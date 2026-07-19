/*
Implements a simple Sampler, used during model inference to sample tokens.
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#include <algorithm>
#include <cmath>
#include <math.h>

// Simple xorshift RNG
unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_softmax(const float* logits, int n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

typedef struct {
    int token_id;
    float logit;
    double weight;
} LlmcSamplingCandidate;

bool llmc_sampling_candidate_greater(
    const LlmcSamplingCandidate& left,
    const LlmcSamplingCandidate& right
) {
    if (left.logit != right.logit) {
        return left.logit > right.logit;
    }
    return left.token_id < right.token_id;
}

// The input logits are already temperature-scaled. Filtering follows the
// native FRNA order: top-k, then top-p, then categorical sampling. Keeping the
// legacy function for the unfiltered case preserves llm.c's default trajectory.
int sample_softmax_top_k_top_p(
    const float* logits,
    int n,
    int top_k,
    float top_p,
    float coin,
    LlmcSamplingCandidate* scratch
) {
    if (top_k == 0 && top_p >= 1.0f) {
        return sample_softmax(logits, n, coin);
    }
    if (logits == nullptr || scratch == nullptr || n <= 0 || top_k < 0 ||
        !(top_p > 0.0f) || !(top_p <= 1.0f) || !std::isfinite(top_p)) {
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(logits[i])) {
            return -1;
        }
        scratch[i].token_id = i;
        scratch[i].logit = logits[i];
        scratch[i].weight = 0.0;
    }

    const int candidate_count = top_k == 0 || top_k > n ? n : top_k;
    if (candidate_count < n) {
        std::partial_sort(
            scratch,
            scratch + candidate_count,
            scratch + n,
            llmc_sampling_candidate_greater);
    } else {
        std::sort(
            scratch,
            scratch + candidate_count,
            llmc_sampling_candidate_greater);
    }

    const double max_logit = (double)scratch[0].logit;
    double total_weight = 0.0;
    for (int i = 0; i < candidate_count; ++i) {
        const double weight = std::exp((double)scratch[i].logit - max_logit);
        scratch[i].weight = weight;
        total_weight += weight;
    }
    if (!std::isfinite(total_weight) || !(total_weight > 0.0)) {
        return -1;
    }

    int retained_count = candidate_count;
    if (top_p < 1.0f) {
        const double threshold = (double)top_p * total_weight;
        double cumulative_weight = 0.0;
        retained_count = 0;
        do {
            cumulative_weight += scratch[retained_count].weight;
            retained_count++;
        } while (retained_count < candidate_count && cumulative_weight < threshold);
    }

    double retained_weight = 0.0;
    for (int i = 0; i < retained_count; ++i) {
        retained_weight += scratch[i].weight;
    }
    const double draw = (double)coin * retained_weight;
    double cumulative_weight = 0.0;
    for (int i = 0; i < retained_count; ++i) {
        cumulative_weight += scratch[i].weight;
        if (draw < cumulative_weight) {
            return scratch[i].token_id;
        }
    }
    return scratch[retained_count - 1].token_id;
}

#endif
