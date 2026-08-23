/*
cuDNN (flash) attention
*/
#ifndef CUDNN_ATT_H
#define CUDNN_ATT_H

#include "cuda_common.h"

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
