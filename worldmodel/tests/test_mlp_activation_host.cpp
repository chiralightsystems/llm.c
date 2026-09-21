#include "llmc/mlp_activation.h"
#include <algorithm>
#include <cstdio>
#include <limits>

double reference(double x, int policy) {
    if (policy == LLMC_MLP_RELU_SQUARED) return x > 0 ? x*x : 0;
    double e = std::exp(-std::abs(x)), s = x >= 0 ? 1/(1+e) : e/(1+e);
    if (policy == LLMC_MLP_SWISH) return x*s;
    double z = 8*(x-1), b = 1+(std::max(z,0.0)+std::log1p(std::exp(-std::abs(z))))/8;
    if (policy == LLMC_MLP_SWISH_POWER2_K8) return x*s*b;
    return x*s*std::pow(b,.25);
}
double power2_derivative_reference(double x) {
    const double e=std::exp(-std::abs(x)),s=x>=0?1/(1+e):e/(1+e);
    const double ds=e/((1+e)*(1+e)),z=8*(x-1),ez=std::exp(-std::abs(z));
    const double b=1+(std::max(z,0.0)+std::log1p(ez))/8,t=z>=0?1/(1+ez):ez/(1+ez);
    return b*(s+x*ds)+x*s*t;
}
int main() {
    int failed=0;
    auto check=[&](bool ok,const char* what){if(!ok){std::fprintf(stderr,"FAIL %s\n",what);++failed;}};
    const float huge=std::numeric_limits<float>::max();
    for (int policy : {LLMC_MLP_SWISH_POWER125_K8, LLMC_MLP_SWISH, LLMC_MLP_RELU_SQUARED, LLMC_MLP_SWISH_POWER2_K8}) {
        for(float x : {-32.f,-15.f,-6.f,-2.f,-.5f,-0.f,0.f,.5f,.875f,1.f,1.125f,2.f,6.f,17.f,32.f}) {
            double y=reference(x,policy), h=1e-5;
            double dy=(reference(double(x)+h,policy)-reference(double(x)-h,policy))/(2*h);
            check(std::abs(llmc_custom_mlp_activation(x,policy)-y)<3e-6*(1+std::abs(y)),"primal reference");
            check(std::abs(llmc_custom_mlp_activation_derivative(x,policy)-dy)<2e-5*(1+std::abs(dy)),"finite-difference derivative");
        }
        check(llmc_custom_mlp_activation(-huge,policy)==0.f,"negative huge finite tail");
        check(llmc_custom_mlp_activation_derivative(-huge,policy)==0.f,"negative huge derivative");
        for(float x:{NAN,INFINITY,-INFINITY}) {
            check(std::isnan(llmc_custom_mlp_activation(x,policy)),"nonfinite primal");
            check(std::isnan(llmc_custom_mlp_activation_derivative(x,policy)),"nonfinite derivative");
        }
        int header[256]={};
        for(int offset:{18,44}) {
            llmc_store_mlp_contract(header,offset,offset==18?7:2,policy);
            check(llmc_mlp_contract_matches(header,offset,policy),"contract roundtrip");
            for(int other : {LLMC_MLP_GELU,LLMC_MLP_SWISH_POWER125_K8,LLMC_MLP_SWISH,LLMC_MLP_RELU_SQUARED,LLMC_MLP_SWISH_POWER2_K8})
                check(llmc_mlp_contract_matches(header,offset,other)==(policy==other),"cross-selector state contract rejected");
            for(int i=1;i<8;++i) {
                header[offset+i]^=1;
                check(!llmc_valid_mlp_contract(header,offset),"contract mutation rejected");
                header[offset+i]^=1;
            }
        }
        int parsed=-1;
        check(llmc_parse_mlp_activation(llmc_mlp_activation_name(policy),&parsed)&&parsed==policy,"custom selector roundtrip");
    }
    check(std::isinf(llmc_swish_power125_k8(huge)),"power positive overflow retained");
    check(std::isfinite(llmc_swish_power125_k8_derivative(huge)),"power huge derivative finite");
    check(llmc_swish(huge)==huge&&llmc_swish_derivative(huge)==1.f,"Swish huge positive tail");
    check(std::isinf(llmc_relu_squared(huge))&&std::isinf(llmc_relu_squared_derivative(huge)),"ReLU true positive overflow retained");
    for(float zero:{-0.f,0.f})
        check(llmc_relu_squared(zero)==0.f&&llmc_relu_squared_derivative(zero)==0.f,"ReLU kink derivative exactly zero");
    check(llmc_swish_derivative(0)==.5f,"Swish derivative at origin");
    const float divide_seam=std::ldexp(1.0f,126),half_max=huge*.5f;
    for(float x:{-huge,-100.f,-32.f,-15.f,-2.f,-.5f,-0.f,0.f,.5f,.875f,1.f,1.125f,2.f,6.f,17.f,32.f,
                 1e18f,1e19f,2e19f,1e30f,nextafterf(divide_seam,0.f),divide_seam,
                 nextafterf(divide_seam,INFINITY),1e38f,half_max,nextafterf(half_max,INFINITY),huge}) {
        const double y=reference(x,LLMC_MLP_SWISH_POWER2_K8),dy=power2_derivative_reference(x);
        const float actual=llmc_swish_power2_k8(x),actual_dy=llmc_swish_power2_k8_derivative(x);
        check(y>huge ? std::isinf(actual)&&actual>0 :
            std::isfinite(actual)&&std::abs(actual-y)<=3e-6*std::abs(y)+std::numeric_limits<float>::min(),"p2 full-range primal oracle");
        check(dy>huge ? std::isinf(actual_dy)&&actual_dy>0 :
            std::isfinite(actual_dy)&&std::abs(actual_dy-dy)<2e-6*(1+std::abs(dy)),"p2 full-range derivative oracle");
    }
    check(llmc_swish_power2_k8_derivative(0)==float(power2_derivative_reference(0)),"p2 derivative at origin");
    const int original_power[]={7,1,1,0x3fa00000,0x41000000,0x3f800000,0x3f800000,1};
    int h[8];llmc_store_mlp_contract(h,0,7);
    check(!memcmp(h,original_power,sizeof(h)),"existing power header bytes unchanged");
    const int original_swish[]={7,2,1,0,0,0,0x3f800000,1};
    const int original_relu[]={7,3,1,0x40000000,0,0,0,1};
    llmc_store_mlp_contract(h,0,7,LLMC_MLP_SWISH);
    check(!memcmp(h,original_swish,sizeof(h)),"existing Swish header bytes unchanged");
    llmc_store_mlp_contract(h,0,7,LLMC_MLP_RELU_SQUARED);
    check(!memcmp(h,original_relu,sizeof(h)),"existing ReLU-squared header bytes unchanged");
    const int power2[]={7,4,1,0x40000000,0x41000000,0x3f800000,0x3f800000,1};
    llmc_store_mlp_contract(h,0,7,LLMC_MLP_SWISH_POWER2_K8);
    check(!memcmp(h,power2,sizeof(h)),"p2 fixed formula/numerics contract bytes");
    int policy=-1;
    check(llmc_parse_mlp_activation("gelu",&policy)&&policy==0,"legacy default selector");
    check(!llmc_parse_mlp_activation("swish_power150_k8",&policy),"unknown selector rejected");
    check(std::isnan(llmc_custom_mlp_activation(1,99))&&std::isnan(llmc_custom_mlp_activation_derivative(1,99)),"unknown custom dispatch fails closed");
    std::printf("llmc_mlp_activation_host: selectors=swish_power125_k8,swish,relu_squared,swish_power2_k8 formula_and_derivative=1 header_contract=1 invalid_selectors=1 result=%s\n",failed?"fail":"pass");
    return failed?1:0;
}
