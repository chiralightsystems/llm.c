#ifndef LLMC_EOS_BOUNDARY_H
#define LLMC_EOS_BOUNDARY_H

// EOS is the first input of the next segment. Labels are never changed here.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <climits>
#include <limits>
#include <stdexcept>
#include <vector>

enum LlmcAttentionBoundaryPolicy { LLMC_ATTENTION_ROW_CAUSAL = 0, LLMC_ATTENTION_ISOLATE_SEGMENTS = 1 };
constexpr int LLMC_MODEL_VERSION_FP32_EOS = 16;
constexpr int LLMC_MODEL_VERSION_BF16_EOS = 17;
constexpr int LLMC_OPTIMIZER_STATE_VERSION_EOS = 6;
constexpr int LLMC_EOS_SCHEMA = 1;
constexpr int LLMC_EOS_HEADER_OFFSET = 64;
constexpr int LLMC_EOS_GROUP_LIMIT = 16384;
constexpr int LLMC_EOS_TOKEN_BUDGET = 131072;

inline const char* llmc_attention_boundary_name(int policy) {
    return policy == LLMC_ATTENTION_ROW_CAUSAL ? "row_causal_v1" :
           policy == LLMC_ATTENTION_ISOLATE_SEGMENTS ? "isolate_segments_v1" : "invalid";
}
inline bool llmc_parse_attention_boundary(const char* text, int* policy) {
    if (!text || !policy) return false;
    for (int p = 0; p <= 1; ++p) if (std::strcmp(text, llmc_attention_boundary_name(p)) == 0) {
        *policy = p; return true;
    }
    return false;
}
inline bool llmc_valid_eos_policy(int policy, int eos, int vocab) {
    return (policy == LLMC_ATTENTION_ROW_CAUSAL && eos == -1) ||
           (policy == LLMC_ATTENTION_ISOLATE_SEGMENTS && eos >= 0 && eos < vocab);
}
inline bool llmc_parse_eos_id(const char* text,int* eos) {
    if(!text||!text[0]||!eos)return false;
    char* end=nullptr;errno=0;const long value=std::strtol(text,&end,10);
    if(errno==ERANGE||*end||value < -1||value>INT_MAX)return false;
    *eos=(int)value;return true;
}
inline void llmc_store_eos_contract(int* header, int base_version, int eos) {
    header[64] = base_version; header[65] = LLMC_EOS_SCHEMA;
    header[66] = LLMC_ATTENTION_ISOLATE_SEGMENTS; header[67] = eos;
}
inline bool llmc_valid_eos_contract(const int* header, int vocab) {
    return header[65] == LLMC_EOS_SCHEMA && header[66] == LLMC_ATTENTION_ISOLATE_SEGMENTS &&
           llmc_valid_eos_policy(header[66], header[67], vocab);
}
inline bool llmc_eos_contract_matches(const int* header, bool wrapped, int policy, int eos, int vocab) {
    return wrapped ? llmc_valid_eos_contract(header, vocab) && header[66] == policy && header[67] == eos
                   : policy == LLMC_ATTENTION_ROW_CAUSAL && eos == -1;
}

struct LlmcEosGroup { int max_sequence; int count; size_t metadata_begin; };
struct LlmcEosHostPlan {
    int B = 0, T = 0, C = 0;
    std::vector<int32_t> positions;
    std::vector<int32_t> lengths;
    std::vector<int64_t> qkv_offsets, output_offsets;
    std::vector<LlmcEosGroup> groups;
};

inline int llmc_eos_bucket_length(int length, int T) {
    int result = std::min(2, T);
    while (result < length) result = result > T / 2 ? T : result * 2;
    return result;
}
inline int llmc_eos_group_capacity(int max_sequence) {
    return std::min(LLMC_EOS_GROUP_LIMIT, std::max(1, LLMC_EOS_TOKEN_BUDGET / max_sequence));
}
inline int llmc_eos_rounded_count(int actual, int capacity) {
    int count = 1;
    while (count < actual) count = count > capacity / 2 ? capacity : count * 2;
    return count;
}

// Host metadata is immutable from forward through backward/replay. Offsets are
// absolute ELEMENT offsets into the original packed tensors, not byte offsets.
// Zero-length padding points to the terminal physical span and is never read.
inline LlmcEosHostPlan llmc_make_eos_plan(const int* inputs, int B, int T, int C, int eos) {
    if (!inputs || B <= 0 || T <= 0 || C <= 0 || eos < 0 ||
        (uint64_t)B * T > (uint64_t)INT32_MAX ||
        (uint64_t)B * T > (uint64_t)INT64_MAX / (3ull * C))
        throw std::invalid_argument("invalid EOS attention shape/token");
    LlmcEosHostPlan plan;
    plan.B = B; plan.T = T; plan.C = C;
    plan.positions.resize((size_t)B * T);
    struct Segment { int begin, length; };
    std::vector<std::vector<Segment>> buckets;
    std::vector<int> sizes;
    for (int s = std::min(2, T);; s = s > T / 2 ? T : s * 2) {
        sizes.push_back(s); buckets.emplace_back(); if (s == T) break;
    }
    for (int b = 0; b < B; ++b) {
        int start = 0;
        for (int t = 0; t <= T; ++t) {
            if (t == T || (t > 0 && inputs[(size_t)b * T + t] == eos)) {
                const int length = t - start;
                const int bucket = (int)(std::lower_bound(sizes.begin(), sizes.end(), length) - sizes.begin());
                buckets[bucket].push_back({b * T + start, length});
                start = t;
            }
            if (t < T) plan.positions[(size_t)b * T + t] = t - start;
        }
    }
    const int64_t terminal = (int64_t)B * T;
    for (size_t bucket = 0; bucket < buckets.size(); ++bucket) {
        const int cap = llmc_eos_group_capacity(sizes[bucket]);
        for (size_t first = 0; first < buckets[bucket].size(); first += cap) {
            const int actual = (int)std::min((size_t)cap, buckets[bucket].size() - first);
            const int count = llmc_eos_rounded_count(actual, cap);
            const size_t origin = plan.lengths.size();
            plan.groups.push_back({sizes[bucket], count, origin});
            // Each group owns its terminal offset; lengths includes the same
            // unused sentinel slot so all three arrays use the same origin.
            for (int i = 0; i <= count; ++i) {
                const bool valid = i < actual;
                const int64_t token = valid ? buckets[bucket][first + i].begin : terminal;
                plan.lengths.push_back(valid ? buckets[bucket][first + i].length : 0);
                plan.qkv_offsets.push_back(token * 3 * C);
                plan.output_offsets.push_back(token * C);
            }
        }
    }
    return plan;
}
#endif
