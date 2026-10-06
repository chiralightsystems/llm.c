/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#include "cuda_common.h"
#include "eos_boundary.h"

struct LlmcEosAttention;
using LlmcEosStatsCopy = void (*)(float*,float*,const int32_t*,const int64_t*,int,int,int,int,int,bool,cudaStream_t);
using LlmcEosQkvRelayout = void (*)(const floatX*,floatX*,size_t,int,bool,cudaStream_t);
LlmcEosAttention* llmc_eos_attention_create(int B, int T, int NH, int C, cudaStream_t stream,
    LlmcEosStatsCopy copy_stats,LlmcEosQkvRelayout relayout_qkv);
void llmc_eos_attention_destroy(LlmcEosAttention* context);
void llmc_eos_attention_prepare(LlmcEosAttention* context, const int* inputs,
                               int B, int T, int eos, cudaStream_t stream);
const int32_t* llmc_eos_attention_positions(const LlmcEosAttention* context);
void llmc_eos_attention_forward(LlmcEosAttention* context, floatX* out, float* stats,
                               floatX* qkv, cudaStream_t stream);
void llmc_eos_attention_backward(LlmcEosAttention* context, floatX* dqkv, floatX* dout,
                                floatX* qkv, floatX* out, float* stats, cudaStream_t stream);

// forward declarations of functions defined in cudnn_att.cpp
void create_cudnn();
void destroy_cudnn();
void attention_forward_cudnn(floatX* out,  // output: (B, T, NH, HS)
                             float* stats, // output for backward pass: (B, NH, T)
                             floatX* inp,  // input: (B, T, 3, NH, HS) QKV
                             int B, int T, int NH, int C, cudaStream_t stream);

// Inference-only causal attention with the current query key and the preceding
// blackout_width - 1 keys denied. Query rows before blackout_width have no
// legal keys and are written as exact zeros.
void attention_forward_cudnn_recent_blackout(
    floatX* out,  // output: (B, T, NH, HS)
    floatX* inp,  // input: (B, T, 3, NH, HS) QKV
    int B,
    int T,
    int NH,
    int C,
    int blackout_width,
    cudaStream_t stream);

void attention_backward_cudnn(floatX* dqkvr,                                       // output
                              floatX* dout, floatX* qkvr, floatX* o, float* stats, // inputs
                              int B, int T, int NH, int C, cudaStream_t stream);

#endif // CUDNN_ATT_H
