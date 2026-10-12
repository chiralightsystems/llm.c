#include "llmc/navia_ffn.h"
#include "llmc/ffn_interventions.h"
#include "llmc/ffn_contract.h"
#include "llmc/eos_boundary.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct Checks {
    int failures=0,assertions=0;
    void operator()(bool ok,const char* label) {
        ++assertions;
        if(!ok) { if(failures<12) std::fprintf(stderr,"FAIL %s\n",label); ++failures; }
    }
    void near(double got,double expected,double tol,const char* label) {
        (*this)(std::isfinite(got) && std::abs(got-expected)<=tol*(1+std::abs(expected)),label);
    }
};
using Vec=std::vector<float>;
using DVec=std::vector<double>;
DVec widen(const Vec& x) { return DVec(x.begin(),x.end()); }

// This FP64 oracle spells the equation independently; it calls no production
// pointwise function and contains no analytic backward implementation.
DVec gum_reference(const DVec& h,const DVec& u,const DVec& g,const DVec& v,int b,
        double gate_scale=1.0) {
    DVec z(h.size());
    for(std::size_t base=0;base<h.size();base+=b) for(int o=0;o<b;++o) {
        double a=0,gate=0,value=0;
        for(int k=0;k<b;++k) {
            a+=u[o*b+k]*h[base+k]; gate+=g[o*b+k]*h[base+k]; value+=v[o*b+k]*h[base+k];
        }
        const double sigmoid=1/(1+std::exp(-gate));
        a+=gate_scale*sigmoid*value;
        z[base+o]=std::max(a,0.0)*std::max(a,0.0);
    }
    return z;
}
double loss(const DVec& h,const DVec& u,const DVec& g,const DVec& v,const Vec& dz,int b,
        double gate_scale=1.0) {
    const DVec z=gum_reference(h,u,g,v,b,gate_scale);
    double value=0;
    for(std::size_t i=0;i<z.size();++i) value+=z[i]*double(dz[i]);
    return value;
}

void check_swiglu(Checks& check) {
    constexpr int rows=3,H=5;
    Vec pre(rows*2*H),bias(2*H),out(rows*H),dz(rows*H),dpre(pre.size());
    for(std::size_t i=0;i<pre.size();++i) pre[i]=float(std::sin(double(i)+.2)*3);
    for(int i=0;i<2*H;++i) bias[i]=float(.1*std::cos(i));
    for(std::size_t i=0;i<dz.size();++i) dz[i]=float(std::cos(i+.3)/dz.size());
    const Vec original=pre;
    llmc_swiglu_forward_host(out.data(),pre.data(),bias.data(),rows,H);
    llmc_swiglu_backward_host(dpre.data(),dz.data(),pre.data(),rows,H);
    auto ref=[](double gate,double value) { return gate*value/(1+std::exp(-gate)); };
    for(int r=0;r<rows;++r) for(int j=0;j<H;++j) {
        const int p=r*2*H+j,i=r*H+j;
        check(pre[p]==original[p]+bias[j] && pre[p+H]==original[p+H]+bias[j+H],"packed SwiGLU bias layout");
        const double gate=pre[p],value=pre[p+H],eps=1e-5;
        check.near(out[i],ref(gate,value),2e-6,"SwiGLU independent primal");
        const double dg=dz[i]*(ref(gate+eps,value)-ref(gate-eps,value))/(2*eps);
        const double dv=dz[i]*(ref(gate,value+eps)-ref(gate,value-eps))/(2*eps);
        check.near(dpre[p],dg,2e-6,"SwiGLU gate finite difference");
        check.near(dpre[p+H],dv,2e-6,"SwiGLU value finite difference");
    }
    float s,c;
    llmc_ffn_sigmoid(32.f,&s,&c);
    check(s==1.f && c>0.f,"sigmoid positive-tail complement retained");
    for(float x:{NAN,INFINITY,-INFINITY}) {
        llmc_ffn_sigmoid(x,&s,&c);
        check(std::isnan(s)&&std::isnan(c),"sigmoid nonfinite does not hide NaN");
    }
}

void check_gum_finite_differences(Checks& check,float gate_scale) {
    constexpr int b=32,H=64,rows=2;
    const int count=rows*H;
    Vec h(count),u(b*b),g(b*b),v(b*b),dz(count),z(count),dh(count),du(b*b),dg(b*b),dv(b*b),work(3*count);
    for(int i=0;i<count;++i) { h[i]=float(.3*std::sin(i*.7+.13)); dz[i]=float(std::cos(i*.2)/count); }
    for(int o=0;o<b;++o) for(int k=0;k<b;++k) {
        const int i=o*b+k;
        u[i]=float(.07*std::sin(i*.3)+(o==k?.5:0));
        g[i]=float(.08*std::cos(i*.4)); v[i]=float(.06*std::sin(i*.5+.2));
    }
    llmc_tiled_gum_forward_host(z.data(),h.data(),nullptr,u.data(),g.data(),v.data(),rows,H,b,gate_scale);
    llmc_tiled_gum_backward_host(dh.data(),du.data(),dg.data(),dv.data(),work.data(),dz.data(),h.data(),u.data(),g.data(),v.data(),rows,H,b,gate_scale);
    DVec rh=widen(h),ru=widen(u),rg=widen(g),rv=widen(v);
    const DVec rz=gum_reference(rh,ru,rg,rv,b,gate_scale);
    for(int i=0;i<count;++i) check.near(z[i],rz[i],2e-6,"GUM independent primal");
    auto finite=[&](DVec& parameter,const Vec& derivative,const char* label) {
        const double eps=1e-5;
        for(std::size_t i=0;i<parameter.size();++i) {
            const double old=parameter[i];
            parameter[i]=old+eps; const double plus=loss(rh,ru,rg,rv,dz,b,gate_scale);
            parameter[i]=old-eps; const double minus=loss(rh,ru,rg,rv,dz,b,gate_scale);
            parameter[i]=old;
            check.near(derivative[i],(plus-minus)/(2*eps),2e-6,label);
        }
    };
    finite(rh,dh,"GUM input finite difference"); finite(ru,du,"GUM U finite difference");
    finite(rg,dg,"GUM G finite difference"); finite(rv,dv,"GUM V finite difference");

    // Independent use-by-use pullbacks prove a SUM across tokens and blocks.
    Vec sumu(b*b),sumg(b*b),sumv(b*b),one_dh(b),one_work(3*b);
    for(int base=0;base<count;base+=b)
        llmc_tiled_gum_backward_host(one_dh.data(),sumu.data(),sumg.data(),sumv.data(),one_work.data(),
            dz.data()+base,h.data()+base,u.data(),g.data(),v.data(),1,b,b,gate_scale);
    for(int i=0;i<b*b;++i) {
        check.near(du[i],sumu[i],2e-7,"U summed block uses");
        check.near(dg[i],sumg[i],2e-7,"G summed block uses");
        check.near(dv[i],sumv[i],2e-7,"V summed block uses");
    }
    Vec accu(b*b,.125f),accg(b*b,-.25f),accv(b*b,.5f);
    for(int step=0;step<2;++step) llmc_tiled_gum_backward_host(dh.data(),accu.data(),accg.data(),accv.data(),work.data(),
        dz.data(),h.data(),u.data(),g.data(),v.data(),rows,H,b,gate_scale);
    for(int i=0;i<b*b;++i) {
        check.near(accu[i],.125+2*double(du[i]),2e-7,"U microbatch gradient addition");
        check.near(accg[i],-.25+2*double(dg[i]),2e-7,"G microbatch gradient addition");
        check.near(accv[i],.5+2*double(dv[i]),2e-7,"V microbatch gradient addition");
    }
}

void check_identity_complete_ffn(Checks& check,float gate_scale) {
    constexpr int rows=3,D=4,H=64,b=32;
    Vec x(rows*D),up(H*D),upbias(H),down(D*H),downbias(D),dy(rows*D);
    for(std::size_t i=0;i<x.size();++i) { x[i]=float(std::sin(i*.3)); dy[i]=float(std::cos(i+.2)/x.size()); }
    for(std::size_t i=0;i<up.size();++i) up[i]=float(.1*std::cos(i*.5));
    for(std::size_t i=0;i<down.size();++i) down[i]=float(.08*std::sin(i*.7));
    for(int j=0;j<H;++j) upbias[j]=float(.03*std::sin(j));
    for(int j=0;j<D;++j) downbias[j]=float(.02*std::cos(j));
    Vec h(rows*H),baseh(rows*H),z(rows*H),baseline(rows*H),dz(rows*H),dh(rows*H),basedh(rows*H);
    Vec u(b*b),g(b*b),v(b*b),du(b*b),dg(b*b),dv(b*b),work(3*rows*H);
    llmc_tiled_gum_identity_bank(u.data(),g.data(),v.data(),b);
    // G can be arbitrary when V=0: identity recovery must not depend on G=0.
    for(int i=0;i<b*b;++i) g[i]=float(.11*std::sin(i*.4));
    for(int r=0;r<rows;++r) for(int j=0;j<H;++j) {
        float sum=0;
        for(int k=0;k<D;++k) sum+=up[j*D+k]*x[r*D+k];
        h[r*H+j]=sum; baseh[r*H+j]=sum+upbias[j];
        baseline[r*H+j]=llmc_relu_squared(baseh[r*H+j]);
        float grad=0;
        for(int k=0;k<D;++k) grad+=down[k*H+j]*dy[r*D+k];
        dz[r*H+j]=grad;
        basedh[r*H+j]=grad*llmc_relu_squared_derivative(baseh[r*H+j]);
    }
    llmc_tiled_gum_forward_host(z.data(),h.data(),upbias.data(),u.data(),g.data(),v.data(),rows,H,b,gate_scale);
    llmc_tiled_gum_backward_host(dh.data(),du.data(),dg.data(),dv.data(),work.data(),dz.data(),h.data(),u.data(),g.data(),v.data(),rows,H,b,gate_scale);
    check(!std::memcmp(z.data(),baseline.data(),z.size()*sizeof(float)),"identity GUM exact activation forward");
    check(!std::memcmp(h.data(),baseh.data(),h.size()*sizeof(float)),"identity GUM exact outer bias");
    // Adding zero matrix terms can canonicalize signed-zero gradients. Exact
    // numeric equality is required; signed zero is not an FFN semantic change.
    check(std::equal(dh.begin(),dh.end(),basedh.begin()),"identity GUM exact expansion adjoint");
    for(int r=0;r<rows;++r) for(int j=0;j<D;++j) {
        float actual=downbias[j],expected=downbias[j],dx=0,bdx=0;
        for(int k=0;k<H;++k) {
            actual+=down[j*H+k]*z[r*H+k]; expected+=down[j*H+k]*baseline[r*H+k];
            dx+=up[k*D+j]*dh[r*H+k]; bdx+=up[k*D+j]*basedh[r*H+k];
        }
        check(actual==expected,"identity exact contracted FFN output");
        check(dx==bdx,"identity exact normalized-residual input gradient");
    }
    for(int j=0;j<H;++j) {
        float db=0,bdb=0;
        for(int r=0;r<rows;++r) { db+=dh[r*H+j]; bdb+=basedh[r*H+j]; }
        check(db==bdb,"identity exact up bias gradient");
        for(int k=0;k<D;++k) {
            float dw=0,bdw=0,ddown=0,bddown=0;
            for(int r=0;r<rows;++r) {
                dw+=dh[r*H+j]*x[r*D+k]; bdw+=basedh[r*H+j]*x[r*D+k];
                ddown+=dy[r*D+k]*z[r*H+j]; bddown+=dy[r*D+k]*baseline[r*H+j];
            }
            check(dw==bdw,"identity exact up weight gradient");
            check(ddown==bddown,"identity exact down weight gradient");
        }
    }
    check(std::all_of(dg.begin(),dg.end(),[](float t){return t==0;}),"identity gate gradient zero when V zero");
    check(std::any_of(du.begin(),du.end(),[](float t){return t!=0;}) &&
          std::any_of(dv.begin(),dv.end(),[](float t){return t!=0;}),"identity banks remain trainable");
    Vec zero(b),signed_zero(b),one_dz(b,1.f),zero_work(3*b);
    for(int i=0;i<b;++i) signed_zero[i]=i%2?-0.f:0.f;
    llmc_tiled_gum_forward_host(zero.data(),signed_zero.data(),nullptr,u.data(),g.data(),v.data(),1,b,b,gate_scale);
    llmc_tiled_gum_backward_host(zero.data(),du.data(),dg.data(),dv.data(),zero_work.data(),one_dz.data(),signed_zero.data(),u.data(),g.data(),v.data(),1,b,b,gate_scale);
    check(std::all_of(zero.begin(),zero.end(),[](float t){return t==0;}),"identity signed-zero kink adjoint");
}

void check_gate_scale(Checks& check) {
    // Nonzero V makes scaling observable even at zero gate preactivation.
    constexpr int b=32;
    Vec h(b,.4f),u(b*b),g(b*b),v(b*b),z1(b),z2(b),legacy(b);
    llmc_tiled_gum_identity_bank(u.data(),g.data(),v.data(),b);
    for(int o=0;o<b;++o) v[o*b+o]=.25f;
    llmc_tiled_gum_forward_host(z1.data(),h.data(),nullptr,u.data(),g.data(),v.data(),1,b,b,1.f);
    llmc_tiled_gum_forward_host(z2.data(),h.data(),nullptr,u.data(),g.data(),v.data(),1,b,b,2.f);
    llmc_tiled_gum_forward_host(legacy.data(),h.data(),nullptr,u.data(),g.data(),v.data(),1,b,b);
    check(!std::memcmp(legacy.data(),z1.data(),b*sizeof(float)),"default scale1 preserves explicit scale1 arithmetic");
    const DVec reference1=gum_reference(widen(h),widen(u),widen(g),widen(v),b,1.0);
    const DVec reference2=gum_reference(widen(h),widen(u),widen(g),widen(v),b,2.0);
    for(int o=0;o<b;++o) {
        check.near(z1[o],reference1[o],2e-7,"zero gate scale1 independent forward");
        check.near(z2[o],reference2[o],2e-7,"zero gate scale2 independent forward");
        check(z2[o]>z1[o],"scale2 changes nonzero admitted branch");
    }
    float a,s,c,value,pu,pg,pv;
    llmc_ffn_gum_channel(h.data(),u.data(),g.data(),v.data(),0,b,&a,&s,&c,&value,2.f);
    check(2.f*s==1.f && 2.f*s*c==.5f,"scale2 gate has unity gain and slope one-half at zero");
    llmc_ffn_gum_local_vjp(1.f,a,s,c,value,&pu,&pg,&pv,2.f);
    check.near(pu,2.0*a,2e-7,"scale2 local ungated derivative");
    check.near(pv,2.0*a,2e-7,"scale2 local value derivative at zero gate");
    auto independent=[&](double gate) {
        const double raw=double(h[0])+2.0*double(value)/(1.0+std::exp(-gate));
        return raw*raw;
    };
    const double eps=1e-5;
    check.near(pg,(independent(eps)-independent(-eps))/(2*eps),2e-7,
        "scale2 local gate derivative finite difference at zero");
    for(float scale:{0.f,-1.f,3.f,NAN,INFINITY})
        check(!llmc_ffn_gate_scale_valid(scale),"unsupported gate scale rejected");
}

// Independent FP64 power oracle used only in the CPU test, never authoring.
double norm_reference(const Vec& a) {
    constexpr int b=32;
    double best=0;
    for(int start=0;start<3;++start) {
        DVec x(b),ax(b),next(b);
        for(int k=0;k<b;++k) x[k]=std::sin((k+1)*(start+.4));
        for(int iteration=0;iteration<800;++iteration) {
            std::fill(ax.begin(),ax.end(),0); std::fill(next.begin(),next.end(),0);
            for(int r=0;r<b;++r) for(int k=0;k<b;++k) ax[r]+=a[r*b+k]*x[k];
            for(int k=0;k<b;++k) for(int r=0;r<b;++r) next[k]+=a[r*b+k]*ax[r];
            double length=0; for(double t:next) length+=t*t; length=std::sqrt(length);
            for(int k=0;k<b;++k) x[k]=next[k]/length;
        }
        std::fill(ax.begin(),ax.end(),0);
        for(int r=0;r<b;++r) for(int k=0;k<b;++k) ax[r]+=a[r*b+k]*x[k];
        double norm=0; for(double t:ax) norm+=t*t;
        best=std::max(best,std::sqrt(norm));
    }
    return best;
}
void check_navia(Checks& check) {
    constexpr int b=32;
    check(LLMC_NAVIA_FFN_COLUMN_ACCEPT_LIMIT%31U==0 &&
          std::uint64_t(LLMC_NAVIA_FFN_COLUMN_ACCEPT_LIMIT)+4U==(std::uint64_t(1)<<32),
          "NAVIA accepted uniform domain excludes exactly four draws");
    for(int row=0;row<b;++row) {
        int seen[32]={};
        for(unsigned residue=0;residue<31;++residue) {
            int column=-1;
            check(llmc_navia_ffn_column_from_draw(residue,row,&column),"NAVIA residue accepted");
            check(column>=0 && column<b && column!=row,"NAVIA negative channel excludes positive channel");
            ++seen[column];
            const std::uint64_t preimages=(std::uint64_t(LLMC_NAVIA_FFN_COLUMN_ACCEPT_LIMIT)-1-residue)/31+1;
            check(preimages==LLMC_NAVIA_FFN_COLUMN_ACCEPT_LIMIT/31U,"NAVIA exact equal accepted preimages");
        }
        for(int col=0;col<b;++col) check(seen[col]==(col==row?0:1),"NAVIA excluded-channel map is bijective");
        int column=-1;
        check(llmc_navia_ffn_column_from_draw(LLMC_NAVIA_FFN_COLUMN_ACCEPT_LIMIT-1,row,&column),"NAVIA last accepted draw");
        for(std::uint64_t draw=LLMC_NAVIA_FFN_COLUMN_ACCEPT_LIMIT;draw<=(std::uint64_t(1)<<32)-1;++draw)
            check(!llmc_navia_ffn_column_from_draw(unsigned(draw),row,&column),"NAVIA top-tail draw rejected");
    }
    mt19937_state rng1,rng2; manual_seed(&rng1,42); manual_seed(&rng2,42);
    Vec previous(b*b);
    for(int family=0;family<3;++family) for(int layer=0;layer<2;++layer) {
        Vec bank(b*b),replica(b*b);
        llmc_navia_ffn_bank(bank.data(),b,&rng1); llmc_navia_ffn_bank(replica.data(),b,&rng2);
        check(!std::memcmp(bank.data(),replica.data(),bank.size()*sizeof(float)),"NAVIA deterministic explicit RNG");
        check(std::memcmp(bank.data(),previous.data(),bank.size()*sizeof(float))!=0,"NAVIA independent bank draw");
        check.near(norm_reference(bank),1.0,2e-6,"NAVIA independent spectral norm");
        const float anchor=bank[0];
        check(anchor>0 && anchor<1,"NAVIA nonidentity positive anchor");
        for(int row=0;row<b;++row) {
            check(bank[row*b+row]==anchor,"NAVIA positive row anchor");
            int negative_anchors=0;
            for(int k=0;k<b;++k) if(k!=row && bank[row*b+k]==-anchor) ++negative_anchors;
            check(negative_anchors==1,"NAVIA one random negative anchor per row");
        }
        check(std::all_of(bank.begin(),bank.end(),[](float t){return std::isfinite(t);}),"NAVIA finite bank");
        previous=bank;
    }
}
void check_contract(Checks& check) {
    check(LLMC_MODEL_VERSION_FP32_FFN==18 && LLMC_MODEL_VERSION_BF16_FFN==19 &&
          LLMC_OPTIMIZER_STATE_VERSION_FFN==7,"FFN append-only version identifiers");
    for(int old:{LLMC_MODEL_VERSION_FP32_MLP_ACTIVATION,LLMC_MODEL_VERSION_BF16_MLP_ACTIVATION,
                 LLMC_MODEL_VERSION_FP32_EOS,LLMC_MODEL_VERSION_BF16_EOS})
        check(old!=LLMC_MODEL_VERSION_FP32_FFN && old!=LLMC_MODEL_VERSION_BF16_FFN,"model wrapper version uniqueness");
    check(LLMC_OPTIMIZER_STATE_VERSION_FFN!=LLMC_OPTIMIZER_STATE_VERSION_MLP_ACTIVATION &&
          LLMC_OPTIMIZER_STATE_VERSION_FFN!=LLMC_OPTIMIZER_STATE_VERSION_EOS,"state wrapper version uniqueness");
    for(int mode:{LLMC_MLP_SWIGLU,LLMC_MLP_TIED_GUM_RELU_SQUARED}) {
        const int hidden=mode==LLMC_MLP_SWIGLU?2048:3072,block=mode==LLMC_MLP_SWIGLU?0:32;
        for(int base:{3,5,7,2,4}) {
            int h[256]={};
            llmc_store_ffn_contract(h,base,mode,hidden,block);
            const int historical[9]={base,1,mode,hidden,block,0,1,1,21};
            check(!std::memcmp(h+80,historical,sizeof(historical)) && h[89]==0,
                "unit scale writes historical schema1 bytes");
            check(llmc_valid_ffn_contract(h),"FFN contract serialization round trip");
            check(llmc_ffn_contract_gate_scale(h)==1.f,"schema1 resolves scale1");
            check(llmc_ffn_contract_matches(h,mode,hidden,block),"FFN exact contract match");
            check(!llmc_ffn_contract_matches(h,mode,hidden+128,block),"FFN valid different width rejected by match");
            check(!llmc_ffn_contract_matches(h,mode==LLMC_MLP_SWIGLU?LLMC_MLP_TIED_GUM_RELU_SQUARED:LLMC_MLP_SWIGLU,hidden,block),"FFN cross-mode match rejection");
            // Base version is supplied by the outer model/state codec. Fixed
            // formula/inventory words are owned by this inner contract.
            for(int fixed:{1,5,6,7,8}) {
                h[LLMC_FFN_HEADER_OFFSET+fixed]^=1;
                check(!llmc_valid_ffn_contract(h),"FFN fixed-word mutation rejected");
                h[LLMC_FFN_HEADER_OFFSET+fixed]^=1;
            }
            for(int word:{2,3,4}) {
                const int old=h[LLMC_FFN_HEADER_OFFSET+word];
                h[LLMC_FFN_HEADER_OFFSET+word]=word==2?99:word==3?hidden+1:16;
                check(!llmc_valid_ffn_contract(h),"FFN malformed shape or mode rejected");
                h[LLMC_FFN_HEADER_OFFSET+word]=old;
            }
            check(llmc_ffn_contract_matches(h,mode,hidden,block),"FFN restored after corruption checks");
            for(int reserved:{1,LLMC_FFN_GATE_SCALE_TWO_BITS,0x7fc00000}) {
                h[89]=reserved;
                check(llmc_valid_ffn_contract(h) && llmc_ffn_contract_gate_scale(h)==1.f &&
                    llmc_ffn_contract_matches(h,mode,hidden,block,1.f),
                    "schema1 ignores historically unowned reserved word89");
                check(!llmc_ffn_contract_matches(h,mode,hidden,block,2.f),
                    "schema1 cannot resume a scale2 model");
            }
        }
    }
    for(int base:{3,5,7,2,4}) {
        int h[256]={}; h[90]=123; h[91]=456;
        llmc_store_ffn_contract(h,base,LLMC_MLP_TIED_GUM_RELU_SQUARED,3072,32,2.f);
        check(h[81]==2 && h[89]==0x40000000 && h[90]==123 && h[91]==456,
            "schema2 explicit FP32 scale2 leaves parameter-count words intact");
        check(llmc_valid_ffn_contract(h) && llmc_ffn_contract_gate_scale(h)==2.f,
            "scale2 schema2 serialization round trip");
        check(llmc_ffn_contract_matches(h,LLMC_MLP_TIED_GUM_RELU_SQUARED,3072,32,2.f),
            "scale2 model and optimizer state match");
        check(!llmc_ffn_contract_matches(h,LLMC_MLP_TIED_GUM_RELU_SQUARED,3072,32,1.f),
            "scale2 state cannot resume a scale1 model");
        for(int bits:{0,0x3f800000,0x40400000,0x7f800000,0x7fc00000,int(0xbf800000U)}) {
            h[89]=bits;
            check(!llmc_valid_ffn_contract(h),"schema2 malformed or unit gate-scale word rejected");
        }
        h[89]=0x40000000;
        for(int fixed:{1,5,6,7,8}) {
            h[80+fixed]^=1;
            check(!llmc_valid_ffn_contract(h),"schema2 fixed formula or inventory corruption rejected");
            h[80+fixed]^=1;
        }
        h[82]=LLMC_MLP_SWIGLU; h[83]=2048; h[84]=0;
        check(!llmc_valid_ffn_contract(h),"SwiGLU cannot acquire GUM scale2 schema");
    }
    for(float unsupported:{0.f,-1.f,3.f,NAN,INFINITY}) {
        int h[256]={};
        llmc_store_ffn_contract(h,3,LLMC_MLP_TIED_GUM_RELU_SQUARED,3072,32,unsupported);
        check(!llmc_valid_ffn_contract(h),"unsupported authored scale cannot become valid scale2");
    }
    for(const char* text:{"1","2"}) {
        float scale=0;
        check(llmc_parse_ffn_gate_scale(text,&scale) && (scale==1.f || scale==2.f),
            "explicit gate-scale CLI parser");
    }
    for(const char* text:{"","0","3","-1","+2","02","2.0","2x"," 2","nan","inf"}) {
        float scale=0;
        check(!llmc_parse_ffn_gate_scale(text,&scale),"gate-scale CLI rejects malformed or unsupported values");
    }
    for(const char* text:{"1","32","128","2048","3072"}) {
        int value=-1; check(llmc_parse_ffn_width(text,&value)&&value>0,"FFN positive decimal parser");
    }
    for(const char* text:{"","0","-32","+32","032","32x"," 32","2147483648"}) {
        int value=-1; check(!llmc_parse_ffn_width(text,&value),"FFN invalid integer parser");
    }
    for(int width:{-128,0,1,127,129,2147483647})
        check(!llmc_ffn_valid_shape(LLMC_MLP_SWIGLU,width,0),"FFN malformed hidden shape");
}
int main() {
    Checks check;
    check(llmc_ffn_geometry_valid(LLMC_MLP_SWIGLU,1,2048,0),"SwiGLU admitted geometry");
    check(llmc_ffn_geometry_valid(LLMC_MLP_TIED_GUM_RELU_SQUARED,1,3072,32),"GUM admitted geometry");
    for(int b:{0,16,31,64}) check(!llmc_ffn_geometry_valid(LLMC_MLP_TIED_GUM_RELU_SQUARED,1,3072,b),"GUM fixed-block rejection");
    check(!llmc_ffn_geometry_valid(LLMC_MLP_TIED_GUM_RELU_SQUARED,0,3072,32) &&
          !llmc_ffn_geometry_valid(LLMC_MLP_TIED_GUM_RELU_SQUARED,1,3073,32),"GUM invalid shapes rejected");
    check(!llmc_ffn_geometry_valid(99,1,32,32),"unknown FFN selector rejected");
    check_swiglu(check); check_gum_finite_differences(check,1.f); check_gum_finite_differences(check,2.f);
    check_identity_complete_ffn(check,1.f); check_identity_complete_ffn(check,2.f);
    check_gate_scale(check); check_navia(check); check_contract(check);
    std::printf("llmc_ffn_interventions_host: swiglu=1 gum_fp64_fd=1 gum_gate_scale2=1 identity_outer_backward=1 summed_bank=1 navia=1 header_contract=1 assertions=%d result=%s\n",
        check.assertions,check.failures?"fail":"pass");
    return check.failures?1:0;
}
