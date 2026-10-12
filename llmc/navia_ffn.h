#ifndef LLMC_NAVIA_FFN_H
#define LLMC_NAVIA_FFN_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include "rand.h"

// 2^32 has remainder4 modulo31. Reject those four top draws so every
// non-diagonal channel has exactly the same number of accepted preimages.
// Accepted draws retain the usual modulo mapping and explicit RNG stream.
constexpr unsigned int LLMC_NAVIA_FFN_COLUMN_ACCEPT_LIMIT =
    0xffffffffU - (0xffffffffU % 31U);
static_assert(std::numeric_limits<unsigned int>::digits==32,"NAVIA FFN sampler requires uint32 RNG words");
inline bool llmc_navia_ffn_column_from_draw(unsigned int draw,int row,int* column) {
    if(row<0 || row>=32 || !column || draw>=LLMC_NAVIA_FFN_COLUMN_ACCEPT_LIMIT) return false;
    const int compact=int(draw%31U);
    *column=compact+(compact>=row);
    return true;
}
inline int llmc_navia_ffn_negative_column(int row,mt19937_state* rng) {
    if(row<0 || row>=32 || !rng) {
        std::fprintf(stderr,"NAVIA FFN negative channel requires a valid row and explicit RNG\n");
        std::exit(EXIT_FAILURE);
    }
    int column=0;
    while(!llmc_navia_ffn_column_from_draw(randint32(rng),row,&column)) {}
    return column;
}

// CPU-only NAVIA authoring for the fixed 32-coordinate FFN bank. Bank values,
// population sigma, scales and publication are FP32; the converged symmetric
// Jacobi eigensolve uses FP64 Gram arithmetic to keep norm error below the
// published bank's FP32 precision. This realizes NAVIA's mathematical policy,
// not PyTorch SVD/RNG bit identity. No CUDA/cuBLAS or power-iteration fallback.
inline float llmc_navia_ffn_spectral_norm(const float* bank,int b) {
    if(b!=32 || !bank) {
        std::fprintf(stderr,"NAVIA FFN requires a non-null 32x32 bank\n");
        std::exit(EXIT_FAILURE);
    }
    double gram[32*32]={};
    for(int i=0;i<b;++i) for(int j=i;j<b;++j) {
        double sum=0.0;
        for(int r=0;r<b;++r) sum+=double(bank[r*b+i])*double(bank[r*b+j]);
        if(!std::isfinite(sum)) {
            std::fprintf(stderr,"NAVIA FFN encountered a nonfinite Gram matrix\n");
            std::exit(EXIT_FAILURE);
        }
        gram[i*b+j]=gram[j*b+i]=sum;
    }
    bool converged=false;
    for(int sweep=0;sweep<64;++sweep) {
        double scale=0.0,off=0.0;
        for(int p=0;p<b;++p) {
            scale=std::fmax(scale,std::fabs(gram[p*b+p]));
            for(int q=p+1;q<b;++q) off=std::fmax(off,std::fabs(gram[p*b+q]));
        }
        if(scale>0.0 && off<=1e-12*scale) { converged=true; break; }
        for(int p=0;p<b;++p) for(int q=p+1;q<b;++q) {
            const double pq=gram[p*b+q];
            if(std::fabs(pq)<=1e-14*scale) continue;
            const double tau=(gram[q*b+q]-gram[p*b+p])/(2.0*pq);
            const double t=std::copysign(1.0/(std::fabs(tau)+std::hypot(1.0,tau)),tau);
            const double c=1.0/std::sqrt(1.0+t*t),s=t*c;
            gram[p*b+p]-=t*pq;
            gram[q*b+q]+=t*pq;
            gram[p*b+q]=gram[q*b+p]=0.0;
            for(int k=0;k<b;++k) if(k!=p && k!=q) {
                const double kp=gram[k*b+p],kq=gram[k*b+q];
                gram[k*b+p]=gram[p*b+k]=c*kp-s*kq;
                gram[k*b+q]=gram[q*b+k]=s*kp+c*kq;
            }
        }
    }
    double maximum=0.0;
    for(int i=0;i<b;++i) maximum=std::fmax(maximum,gram[i*b+i]);
    const float norm=float(std::sqrt(maximum));
    if(!converged || !std::isfinite(norm) || norm<=0.0f) {
        std::fprintf(stderr,"NAVIA FFN spectral normalization failed to converge to a finite positive norm\n");
        std::exit(EXIT_FAILURE);
    }
    return norm;
}
inline void llmc_navia_ffn_bank(float* bank,int b,mt19937_state* rng) {
    if(b!=32 || !bank || !rng) {
        std::fprintf(stderr,"NAVIA FFN requires a 32x32 bank and explicit independent RNG\n");
        std::exit(EXIT_FAILURE);
    }
    normal_(bank,unsigned(b*b),0.0f,1.0f,rng);
    const float initial_norm=llmc_navia_ffn_spectral_norm(bank,b);
    double mean=0.0;
    for(int i=0;i<b*b;++i) { bank[i]/=initial_norm; mean+=bank[i]; }
    mean/=b*b;
    double variance=0.0;
    for(int i=0;i<b*b;++i) {
        const double centered=double(bank[i])-mean;
        variance+=centered*centered;
    }
    const float sigma=float(std::sqrt(variance/(b*b)));
    if(!std::isfinite(sigma) || sigma<=0.0f) {
        std::fprintf(stderr,"NAVIA FFN population sigma must be finite and positive\n");
        std::exit(EXIT_FAILURE);
    }
    const float anchor=3.0f*sigma;
    for(int row=0;row<b;++row) {
        const int negative=llmc_navia_ffn_negative_column(row,rng);
        bank[row*b+row]=anchor;
        bank[row*b+negative]=-anchor;
    }
    const float final_norm=llmc_navia_ffn_spectral_norm(bank,b);
    for(int i=0;i<b*b;++i) bank[i]/=final_norm;
}
#endif
