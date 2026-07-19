#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "llmc/sampler.h"

int main(void) {
    LlmcSamplingCandidate scratch[4] = {};
    const float logits[4] = {1.0f, 3.0f, 2.0f, 3.0f};
    const float equal_logits[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Top-k ties resolve to the lowest token id, matching native FRNA.
    assert(sample_softmax_top_k_top_p(logits, 4, 1, 1.0f, 0.9f, scratch) == 1);

    // Nucleus sampling retains the threshold-crossing token and excludes the tail.
    bool saw_crossing_token = false;
    for (int draw_index = 0; draw_index < 100; ++draw_index) {
        const float coin = ((float)draw_index + 0.5f) / 100.0f;
        const int token = sample_softmax_top_k_top_p(
            equal_logits, 4, 0, 0.75f, coin, scratch);
        assert(token >= 0 && token < 3);
        saw_crossing_token = saw_crossing_token || token == 2;
    }
    assert(saw_crossing_token);

    // Explicitly disabled filters preserve the existing llm.c sampler exactly.
    for (int draw_index = 0; draw_index < 100; ++draw_index) {
        const float coin = ((float)draw_index + 0.5f) / 100.0f;
        assert(sample_softmax_top_k_top_p(logits, 4, 0, 1.0f, coin, nullptr) ==
               sample_softmax(logits, 4, coin));
    }

    const float nonfinite_logits[4] = {0.0f, NAN, 1.0f, 2.0f};
    assert(sample_softmax_top_k_top_p(nonfinite_logits, 4, 2, 0.95f, 0.5f, scratch) == -1);

    printf("sampler top-k/top-p tests passed\n");
    return 0;
}
