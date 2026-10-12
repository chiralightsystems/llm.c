// Synthetic local-kernel timing diagnostic. No dataset, optimizer or training.
// Compile separately against an explicitly selected immutable source snapshot.
#include "llmc/ffn_interventions.cuh"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <vector>

cudaDeviceProp deviceProp;

__global__ void fill_inputs(float* h, float* dz, size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count) {
        h[i]=float(int(i%97)-48)*.01f;
        dz[i]=float(int(i%31)-15)*1e-7f;
    }
}
template<class T> struct DeviceBuffer {
    T* p=nullptr; size_t n;
    explicit DeviceBuffer(size_t size):n(size){cudaCheck(cudaMalloc(&p,n*sizeof(T)));}
    ~DeviceBuffer(){if(p)cudaFree(p);}
};
template<class F> std::vector<float> time_calls(F fn,cudaStream_t stream,int warmup,int repeats) {
    cudaEvent_t start,end;cudaCheck(cudaEventCreate(&start));cudaCheck(cudaEventCreate(&end));
    for(int i=0;i<warmup;++i)fn();
    cudaCheck(cudaStreamSynchronize(stream));
    std::vector<float> result;
    for(int i=0;i<repeats;++i) {
        cudaCheck(cudaEventRecord(start,stream));fn();cudaCheck(cudaEventRecord(end,stream));
        cudaCheck(cudaEventSynchronize(end));float ms=0;cudaCheck(cudaEventElapsedTime(&ms,start,end));
        result.push_back(ms);
    }
    cudaCheck(cudaEventDestroy(start));cudaCheck(cudaEventDestroy(end));return result;
}
void write_array(std::ofstream& file,const std::vector<float>& values) {
    file<<'[';for(size_t i=0;i<values.size();++i){if(i)file<<',';file<<values[i];}file<<']';
}
int main(int argc,char** argv) {
    if(argc!=2){fprintf(stderr,"usage: bench_ffn_interventions <exclusive-output-json>\n");return 2;}
    // The supervising caller owns the exclusive output directory.
    if(FILE* existing=fopen(argv[1],"rb")){fclose(existing);fprintf(stderr,"Refusing existing output\n");return 2;}
    constexpr int rows=256*1024,hidden=3072,b=32,warmup=2,repeats=5;
    constexpr size_t count=size_t(rows)*hidden;
    cudaCheck(cudaSetDevice(0));cudaCheck(cudaGetDeviceProperties(&deviceProp,0));
    cudaStream_t stream;cudaCheck(cudaStreamCreate(&stream));
    DeviceBuffer<float> h(count),dz(count),local(3*count);
    DeviceBuffer<floatX> out(count),dh(count),u(b*b),g(b*b),v(b*b),du(b*b),dg(b*b),dv(b*b);
    std::vector<floatX> bank(b*b);
    for(int i=0;i<b*b;++i)bank[i]=floatX(float((i*7)%23-11)*.017f);
    for(auto p:{u.p,g.p,v.p})cudaCheck(cudaMemcpy(p,bank.data(),bank.size()*sizeof(floatX),cudaMemcpyHostToDevice));
    fill_inputs<<<CEIL_DIV(count,256),256,0,stream>>>(h.p,dz.p,count);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemsetAsync(du.p,0,b*b*sizeof(floatX),stream));
    cudaCheck(cudaMemsetAsync(dg.p,0,b*b*sizeof(floatX),stream));
    cudaCheck(cudaMemsetAsync(dv.p,0,b*b*sizeof(floatX),stream));
#ifdef LLMC_TIED_GUM_OPTIMIZED_KERNELS
    DeviceBuffer<float> partial(llmc_tiled_gum_backward_bank_scratch_floats(rows,hidden,b));
#endif
    const auto forward=time_calls([&]{llmc_tiled_gum_forward(out.p,h.p,nullptr,u.p,g.p,v.p,rows,hidden,b,stream,2.f);},stream,warmup,repeats);
    const auto backward=time_calls([&]{
#ifdef LLMC_TIED_GUM_OPTIMIZED_KERNELS
        llmc_tiled_gum_backward(dh.p,du.p,dg.p,dv.p,local.p,dz.p,h.p,u.p,g.p,v.p,rows,hidden,b,partial.p,partial.n,stream,2.f);
#else
        llmc_tiled_gum_backward(dh.p,du.p,dg.p,dv.p,local.p,dz.p,h.p,u.p,g.p,v.p,rows,hidden,b,stream,2.f);
#endif
    },stream,warmup,repeats);
    const auto input_backward=time_calls([&]{
#ifdef LLMC_TIED_GUM_OPTIMIZED_KERNELS
        llmc_tiled_gum_backward_input_kernel<<<CEIL_DIV(count/b,LLMC_GUM_LOCAL_USES_PER_CTA),LLMC_GUM_LOCAL_WARPS*b,0,stream>>>(dh.p,local.p,dz.p,h.p,u.p,g.p,v.p,count,2.f);
#else
        llmc_tiled_gum_backward_input_kernel<<<count/b,b,0,stream>>>(dh.p,local.p,dz.p,h.p,u.p,g.p,v.p,count,2.f);
#endif
        cudaCheck(cudaGetLastError());
    },stream,warmup,repeats);
    const auto bank_backward=time_calls([&]{
#ifdef LLMC_TIED_GUM_OPTIMIZED_KERNELS
        llmc_tiled_gum_backward_bank(du.p,dg.p,dv.p,local.p,h.p,rows,hidden,b,partial.p,partial.n,stream);
#else
        llmc_tiled_gum_backward_bank_kernel<<<b*b,256,0,stream>>>(du.p,dg.p,dv.p,local.p,h.p,count);
        cudaCheck(cudaGetLastError());
#endif
    },stream,warmup,repeats);
    // Finite publication check is deliberately separate from numerical qualification.
    std::vector<floatX> host(b*b);bool finite=true;
    for(auto p:{du.p,dg.p,dv.p}){
        cudaCheck(cudaMemcpy(host.data(),p,host.size()*sizeof(floatX),cudaMemcpyDeviceToHost));
        for(auto x:host)finite=finite&&std::isfinite(float(x));
    }
    size_t free_bytes,total_bytes;cudaCheck(cudaMemGetInfo(&free_bytes,&total_bytes));
    std::ofstream file(argv[1]);if(!file){fprintf(stderr,"Cannot create report\n");return 2;}
    file<<std::setprecision(9)<<"{\n  \"schema\":\"llmc.worldmodel.ffn_local_timing.v1\",\n"
        <<"  \"scope\":\"synthetic full-shape local kernels only; no training or optimizer; not a numerical oracle\",\n"
        <<"  \"rows\":"<<rows<<",\n  \"hidden\":"<<hidden<<",\n  \"block\":"<<b
        <<",\n  \"gate_scale\":2,\n  \"warmup_calls\":"<<warmup<<",\n  \"timed_calls\":"<<repeats
        <<",\n  \"local_accumulation\":\"FP32\",\n  \"finite_bank_publications\":"<<(finite?"true":"false")
        <<",\n  \"observed_device_global_used_bytes\":"<<(total_bytes-free_bytes)
        <<",\n  \"forward_cuda_event_ms\":";write_array(file,forward);
    file<<",\n  \"backward_input_and_bank_cuda_event_ms\":";write_array(file,backward);
    file<<",\n  \"backward_input_cuda_event_ms\":";write_array(file,input_backward);
    file<<",\n  \"backward_bank_cuda_event_ms\":";write_array(file,bank_backward);file<<"\n}\n";
    file.close();cudaCheck(cudaStreamDestroy(stream));return finite?0:1;
}
