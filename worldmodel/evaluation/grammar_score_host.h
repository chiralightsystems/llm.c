// CPU-only input alignment and observation around unmodified llm.c validation.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace grammar_score {
constexpr int kVocab = 50257;
constexpr int kBos = 50256;
constexpr int64_t kIgnore = -100;
enum class RowPolicy { GrammarBos, PackedSuffix256, PackedFullRow, PackedFinalWord };

inline void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Keep the model's valid-target contract: excluded positions receive target 0.
// Labels are unshifted cache labels, so logits at t predict labels at t+1.
inline std::vector<int> shifted_targets(const uint32_t* inputs,
        const int64_t* labels, size_t B, size_t T,
        RowPolicy policy = RowPolicy::GrammarBos) {
    require(B > 0 && T >= 2, "Invalid cache batch shape");
    std::vector<int> targets(B * T, 0);
    if (policy == RowPolicy::PackedFinalWord) {
        for (size_t b = 0; b < B; ++b) {
            const size_t row = b * T;
            require(labels[row] == kIgnore, "Final-word row must start with real unscored context");
            bool selected = false, padding = false;
            size_t count = 0;
            for (size_t t = 0; t < T; ++t) {
                require(inputs[row + t] < kVocab, "Final-word input outside vocabulary");
                const int64_t label = labels[row + t];
                if (label == kIgnore) {
                    if (selected) padding = true;
                    if (padding) require(inputs[row + t] == 0, "Final-word right padding must be input 0");
                } else {
                    require(t > 0 && !padding, "Final-word targets must be one contiguous suffix before padding");
                    require(label >= 0 && label < kVocab && inputs[row + t] == static_cast<uint32_t>(label),
                            "Final-word input and selected label mismatch");
                    selected = true;
                    targets[row + t - 1] = static_cast<int>(label);
                    ++count;
                }
            }
            if (!count) {
                for (size_t t = 0; t < T; ++t)
                    require(inputs[row + t] == 0 && labels[row + t] == kIgnore,
                            "An empty final-word lane must have only input 0 and ignored labels");
            }
        }
        return targets;
    }
    if (policy == RowPolicy::PackedFullRow) {
        for (size_t b = 0; b < B; ++b) {
            const size_t row = b * T;
            require(labels[row] == kIgnore, "Packed full-row start must be unscored");
            if (labels[row + 1] == kIgnore) {
                // Batch completion may append explicit empty lanes, never a
                // partially masked source row. Real token-zero rows still
                // have selected labels and take the ordinary branch below.
                for (size_t t = 0; t < T; ++t)
                    require(inputs[row + t] == 0 && labels[row + t] == kIgnore,
                            "A packed dummy row must have only input 0 and ignored labels");
                continue;
            }
            for (size_t t = 0; t < T; ++t) {
                require(inputs[row + t] < kVocab, "Packed input outside vocabulary");
                if (t == 0) continue;
                require(labels[row + t] == static_cast<int64_t>(inputs[row + t]),
                        "Packed full-row labels must score every original shifted input");
                targets[row + t - 1] = static_cast<int>(inputs[row + t]);
            }
        }
        return targets;
    }
    if (policy == RowPolicy::PackedSuffix256) {
        require(T >= 257, "Packed suffix needs a real predecessor and 256 targets");
        for (size_t b = 0; b < B; ++b) {
            const size_t row = b * T;
            require(labels[row] == kIgnore, "Packed start must be unscored");
            bool selected = false, padding = false;
            size_t count = 0;
            for (size_t t = 0; t < T; ++t) {
                require(inputs[row + t] < kVocab, "Packed input outside vocabulary");
                const int64_t label = labels[row + t];
                if (label == kIgnore) {
                    if (selected) padding = true;
                    if (padding) require(inputs[row + t] == 0, "Trailing padding must be input 0");
                } else {
                    require(t > 0 && !padding, "Packed targets must be one contiguous suffix before padding");
                    require(label >= 0 && label < kVocab, "Invalid packed target");
                    require(inputs[row + t] == static_cast<uint32_t>(label), "Packed input and label mismatch");
                    selected = true;
                    targets[row + t - 1] = static_cast<int>(label);
                    ++count;
                }
            }
            require(count == 256, "Packed suffix must contain exactly 256 scored targets");
        }
        return targets;
    }
    for (size_t b = 0; b < B; ++b) {
        const size_t row = b * T;
        require(labels[row] == kIgnore, "BOS must not be a scored target");
        bool padding = false;
        size_t count = 0;
        for (size_t t = 1; t < T; ++t) {
            const int64_t label = labels[row + t];
            if (label == kIgnore) {
                padding = true;
                require(inputs[row + t] == 0, "Ignored inputs must be right padding 0");
            } else {
                require(!padding, "Sentence targets must be a contiguous prefix after BOS");
                require(label >= 0 && label < kVocab && label != kBos,
                        "Invalid sentence target or embedded EOS");
                require(inputs[row + t] == static_cast<uint32_t>(label),
                        "Unshifted input and label mismatch");
                targets[row + t - 1] = static_cast<int>(label);
                ++count;
            }
        }
        require(inputs[row] == static_cast<uint32_t>(count ? kBos : 0), "Expected BOS or zero-input dummy row");
    }
    return targets;
}

struct RowScore {
    double sum = 0;
    size_t count = 0;
    std::vector<double> token_nll;
    std::vector<bool> selected;
    bool greedy_observed = false;
    bool greedy_exact_match = false;
    std::vector<int> greedy_token_ids;
};

// Observe original logits, not probabilities, NLL thresholds, or target/max
// equality. Ascending IDs and strict comparison implement torch.argmax ties.
template <typename Value>
inline int argmax_lowest(const Value* logits, size_t vocab) {
    require(logits != nullptr && vocab > 0 && vocab <= kVocab, "Invalid argmax vocabulary");
    int best = 0;
    float maximum = static_cast<float>(logits[0]);
    require(std::isfinite(maximum), "Nonfinite greedy logit");
    for (size_t v = 1; v < vocab; ++v) {
        const float value = static_cast<float>(logits[v]);
        require(std::isfinite(value), "Nonfinite greedy logit");
        if (value > maximum) { maximum = value; best = static_cast<int>(v); }
    }
    return best;
}

// Independent FP64 logsumexp checks that the observed vocabulary vector is
// the same one scored by the ordinary FP32 classifier. It never determines
// the greedy ID, which comes from the exact argmax above.
template <typename Value>
inline double observed_logit_nll(const Value* logits, size_t vocab, int target) {
    require(target >= 0 && static_cast<size_t>(target) < vocab, "Invalid observed target");
    const int best = argmax_lowest(logits, vocab);
    const double maximum = static_cast<float>(logits[best]);
    double sum = 0;
    for (size_t v = 0; v < vocab; ++v)
        sum += std::exp(static_cast<double>(static_cast<float>(logits[v])) - maximum);
    const double nll = std::log(sum) + maximum - static_cast<float>(logits[target]);
    require(std::isfinite(nll) && nll >= 0, "Invalid observed logsumexp NLL");
    return nll;
}

inline void observe_greedy(RowScore& row, const int64_t* unshifted_labels,
        const std::vector<int>& token_ids) {
    require(unshifted_labels != nullptr && token_ids.size() == row.selected.size(), "Greedy row alignment mismatch");
    bool exact = row.count > 0;
    for (size_t t = 0; t < token_ids.size(); ++t) {
        if (row.selected[t]) {
            require(token_ids[t] >= 0 && token_ids[t] < kVocab, "Invalid selected greedy token");
            exact = exact && token_ids[t] == unshifted_labels[t + 1];
        } else {
            require(token_ids[t] == -1, "Ignored position has a greedy token");
        }
    }
    row.greedy_observed = true;
    row.greedy_exact_match = exact;
    row.greedy_token_ids = token_ids;
}

inline RowScore reduce(const float* losses, const int64_t* unshifted_labels, size_t T) {
    require(T >= 2, "Invalid row length");
    RowScore result;
    result.token_nll.resize(T - 1, 0);
    result.selected.resize(T - 1, false);
    for (size_t t = 0; t + 1 < T; ++t) {
        if (unshifted_labels[t + 1] == kIgnore) continue;
        require(unshifted_labels[t + 1] >= 0 && unshifted_labels[t + 1] < kVocab,
                "Invalid selected label");
        const double value = static_cast<double>(losses[t]);
        require(std::isfinite(value) && value >= 0, "Nonfinite or negative selected NLL");
        result.token_nll[t] = value;
        result.selected[t] = true;
        result.sum += value;
        ++result.count;
    }
    require(std::isfinite(result.sum), "Nonfinite row sum");
    return result;
}

inline void write(FILE* out, const RowScore& row, size_t batch, size_t lane,
        size_t execution_row) {
    fprintf(out, "{\"schema\":\"worldmodel.native_frna.eval_row_score.v1\","
        "\"batch_index\":%zu,\"row_in_batch\":%zu,\"execution_row_index\":%zu,"
        "\"scored_targets\":%zu,\"nll_sum\":%.17g,\"mean_nll\":",
        batch, lane, execution_row, row.count, row.sum);
    if (row.count) fprintf(out, "%.17g", row.sum / static_cast<double>(row.count));
    else fputs("null", out);
    fputs(",\"nll_reduction\":\"host_float64_sum_of_llmc_float32_token_nll\",\"token_nll\":[", out);
    for (size_t t = 0; t < row.token_nll.size(); ++t) {
        if (t) fputc(',', out);
        if (row.selected[t]) fprintf(out, "%.17g", row.token_nll[t]);
        else fputs("null", out);
    }
    fputc(']', out);
    if (row.greedy_observed) {
        fputs(",\"greedy_policy\":\"argmax_lowest_token_id_v1\",\"greedy_exact_match\":", out);
        fputs(row.count ? (row.greedy_exact_match ? "true" : "false") : "null", out);
        fputs(",\"greedy_token_ids\":[", out);
        for (size_t t = 0; t < row.greedy_token_ids.size(); ++t) {
            if (t) fputc(',', out);
            if (row.selected[t]) fprintf(out, "%d", row.greedy_token_ids[t]);
            else fputs("null", out);
        }
        fputc(']', out);
    }
    fputs("}\n", out);
    require(!ferror(out), "Row score output write failed");
}
}  // namespace grammar_score
