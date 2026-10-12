// nvcc parses the production CUDA headers, but this executable calls ONLY
// their CPU plan/layout functions. It never creates a CUDA context, handle,
// allocation, stream or kernel launch; no dataset or experiment is involved.
#define WIN32_LEAN_AND_MEAN
#include <cassert>
#include "llmc/cuda_common.h"
#include "llmc/cuda_utils.cuh"
#include "llmc/cublas_common.h"
#include "llmc/ffn_interventions.cuh"
#include "llmc/normuon.cuh"
#include "llmc/normuon_batched.cuh"
#include <algorithm>
#include <cstdio>

cudaDeviceProp deviceProp={};

int main() {
    int failures=0,checks=0;
    auto check=[&](bool ok,const char* label) {
        ++checks;
        if(!ok) { std::fprintf(stderr,"FAIL %s\n",label); ++failures; }
    };
    constexpr int L=12,C=768;
    const int modes[]={LLMC_MLP_RELU_SQUARED,LLMC_MLP_SWIGLU,LLMC_MLP_TIED_GUM_RELU_SQUARED};
    for(int mode:modes) {
        const std::size_t H=mode==LLMC_MLP_SWIGLU?2048:3072;
        const std::size_t F=mode==LLMC_MLP_SWIGLU?2*H:H;
        std::size_t elements[LLMC_OPTIMIZER_PARAMETER_TYPE_COUNT]={
            50304U*C,1024U*C,L*C,L*C,L*3U*C*C,L*3U*C,
            L*std::size_t(C)*C,L*C,L*C,L*C,L*F*C,L*F,
            L*std::size_t(C)*H,L*C,C,C,0,0,0,0,0};
        if(mode==LLMC_MLP_TIED_GUM_RELU_SQUARED)
            elements[18]=elements[19]=elements[20]=L*32U*32U;
        LlmcNormuonConfig config;
        llmc_normuon_config_defaults(&config);
        config.optimizer_selection=LLMC_OPTIMIZER_SELECTION_ADAMW_NORMUON;
        config.orthogonalization_mode=LLMC_NORMUON_ORTHO_RECTANGULAR_MUON;
        config.execution_mode=LLMC_NORMUON_EXECUTION_BF16_BATCHED;
        config.retraction_mode=LLMC_NORMUON_TRACKER_RETRACTION_DISABLED;
        config.targeted_family_mask=llmc_normuon_required_target_mask();
        config.learning_rate=5e-4f;
        LlmcOptimizerPlan plan;
        char error[256]={};
        const bool ok=llmc_build_optimizer_plan(&plan,&config,L,C,elements,error,sizeof(error));
        check(ok,"actual optimizer plan accepted");
        if(!ok) { std::fprintf(stderr,"%s\n",error); continue; }
        check(plan.normuon_parameter_type_count==2 && plan.normuon_view_count==2*L,"NorMuon only two outer matrices");
        std::size_t maximum_panel=0,maximum_rows=0;
        for(int id=0;id<LLMC_OPTIMIZER_PARAMETER_TYPE_COUNT;++id) {
            const auto& type=plan.parameter_types[id];
            check(type.tensor_elements==elements[id],"optimizer entire tensor extent");
            check(type.backend_kind==((id==10 || id==12)?LLMC_OPTIMIZER_BACKEND_NORMUON:LLMC_OPTIMIZER_BACKEND_ADAMW),"AdamW rest including local banks");
            if(id>=18) {
                check(type.layer_elements==(mode==LLMC_MLP_TIED_GUM_RELU_SQUARED?32U*32U:0U),"independent per-layer bank extent");
            }
            if(id!=10 && id!=12) continue;
            LlmcOptimizerMatrixView view={};
            check(type.enumerate_matrix_view(&type,0,&view),"production whole rectangular enumerator");
            check(llmc_optimizer_view_within_bounds(&type,&view),"whole rectangle within declared layer");
            check(view.rows==(id==10?F:C) && view.columns==(id==10?C:H),"checkpoint-shaped rectangular dimensions");
            const std::size_t n=view.rows*view.columns;
            check(n==type.layer_elements,"matrix extent equals unique layer extent");
            maximum_panel=std::max(maximum_panel,std::size_t(L)*n);
            maximum_rows=std::max(maximum_rows,std::max(view.rows,view.columns));
            for(int layer=0;layer<L;++layer) {
                const std::size_t start=std::size_t(layer)*n;
                check(llmc_normuon_batch_rectangular_offset(layer,0,n)==start,"rectangular parameter layer start");
                check(llmc_normuon_batch_rectangular_offset(layer,n-1,n)==start+n-1,"rectangular parameter full layer end");
                check(llmc_normuon_batch_rectangular_second_offset(layer,0,n)==start,"rectangular second moment layer start");
                check(llmc_normuon_batch_rectangular_second_offset(layer,view.rows-1,n)==start+view.rows-1,"all output-row gains stay within actual layer");
                check(start+n<=elements[id],"rectangular gradient/master extent stays in tensor");
                if(layer+1<L) check(llmc_normuon_batch_rectangular_offset(layer,n,n)==
                    llmc_normuon_batch_rectangular_offset(layer+1,0,n),"adjacent rectangles have no gaps or overlap");
            }
            auto malformed=type; ++malformed.layer_elements;
            check(!type.enumerate_matrix_view(&malformed,0,&view),"nonintegral rectangular shape rejected");
        }
        check(maximum_panel==std::size_t(L)*F*C,"workspace maximum accounts for packed expansion");
        check(maximum_rows==F,"row statistics account for packed expansion rows");
        std::size_t capacity=0,panel=0,axis=0;
        check(llmc_normuon_workspace_geometry(&plan,LLMC_NORMUON_EXECUTION_BF16_BATCHED,
            &capacity,&panel,&axis),"production batched workspace planner accepted");
        check(capacity==L && panel==maximum_panel && axis==maximum_rows,
            "batched rectangles retain all layers and actual dimensions");
        check(llmc_normuon_workspace_geometry(&plan,LLMC_NORMUON_EXECUTION_FP32_REFERENCE,
            &capacity,&panel,&axis),"production serial workspace planner accepted");
        check(capacity==1 && panel==F*C && axis==F,
            "serial reference retains only maximum actual single rectangle");
        check(5U*panel*sizeof(float)==5U*F*C*sizeof(float),
            "five FP32 reference panels do not multiply by layer count");
        if(mode==LLMC_MLP_SWIGLU) {
            check(F*C!=4U*C*C && H*C!=4U*C*C,"SwiGLU asymmetric extents exercise former fixed-stride bug");
        }
        if(mode==LLMC_MLP_RELU_SQUARED) {
            config.orthogonalization_mode=LLMC_NORMUON_ORTHO_NEWTON_SCHULZ;
            LlmcOptimizerPlan square_plan;
            const bool square_ok=llmc_build_optimizer_plan(&square_plan,&config,L,C,elements,error,sizeof(error));
            check(square_ok,"legacy square-view optimizer plan accepted");
            if(square_ok) {
                check(llmc_normuon_workspace_geometry(&square_plan,LLMC_NORMUON_EXECUTION_BF16_BATCHED,
                    &capacity,&panel,&axis),"square batched workspace planner accepted");
                check(capacity==std::size_t(L)*4 && panel==std::size_t(L)*4*C*C && axis==C,
                    "square batched path retains full view batch");
                check(llmc_normuon_workspace_geometry(&square_plan,LLMC_NORMUON_EXECUTION_FP32_REFERENCE,
                    &capacity,&panel,&axis),"square serial workspace planner accepted");
                check(capacity==1 && panel==std::size_t(C)*C && axis==C,
                    "square serial reference restores one-view allocation");
            }
        }
    }
    std::printf("llmc_ffn_optimizer_host: actual_plan=1 rectangular_addressing=1 bank_adamw=1 workspace_planning=1 cuda_device_calls=0 checks=%d result=%s\n",
        checks,failures?"fail":"pass");
    return failures?1:0;
}
