#ifndef LLMC_FFN_CONTRACT_H
#define LLMC_FFN_CONTRACT_H
#include "ffn_interventions.h"
#include <cstring>
#include <cerrno>
#include <climits>
#include <cstdlib>

// Append-only wrapper: versions 14/15 are pointwise, 16/17 are EOS.
constexpr int LLMC_MODEL_VERSION_FP32_FFN = 18;
constexpr int LLMC_MODEL_VERSION_BF16_FFN = 19;
constexpr int LLMC_OPTIMIZER_STATE_VERSION_FFN = 7;
constexpr int LLMC_FFN_HEADER_OFFSET = 80;
constexpr int LLMC_FFN_GATE_SCALE_WORD = 89;
constexpr int LLMC_FFN_GATE_SCALE_TWO_BITS = 0x40000000;
inline bool llmc_ffn_is_topology(int mode) {
    return mode == LLMC_MLP_SWIGLU || mode == LLMC_MLP_TIED_GUM_RELU_SQUARED;
}
inline bool llmc_parse_ffn_width(const char* text, int* value) {
    if (!text || !*text || *text < '1' || *text > '9') return false;
    errno = 0; char* end = nullptr;
    long n = std::strtol(text, &end, 10);
    if (errno || *end || n <= 0 || n > INT_MAX) return false;
    *value = int(n); return true;
}
inline bool llmc_parse_ffn_gate_scale(const char* text, float* value) {
    if (!text || !value || (std::strcmp(text,"1") && std::strcmp(text,"2"))) return false;
    *value = text[0] == '2' ? 2.0f : 1.0f;
    return true;
}
inline bool llmc_ffn_valid_shape(int mode, int hidden, int block) {
    return hidden > 0 && hidden <= INT_MAX/2 && hidden % 128 == 0 &&
        ((mode == LLMC_MLP_SWIGLU && block == 0) ||
         (mode == LLMC_MLP_TIED_GUM_RELU_SQUARED && block == 32 && hidden % block == 0));
}
inline const char* llmc_ffn_name(int mode) {
    return mode == LLMC_MLP_SWIGLU ? "swiglu" :
        mode == LLMC_MLP_TIED_GUM_RELU_SQUARED ? "tied_gum_relu_squared" : "invalid";
}
inline void llmc_store_ffn_contract(int* h, int base, int mode, int hidden, int block,
        float gate_scale = 1.0f) {
    const int words[9] = {base, gate_scale == 1.0f ? 1 : 2, mode, hidden, block, 0, 1, 1, 21};
    // no local bias; NAVIA local initialization; FP32 replay; append-only inventory
    std::memcpy(h + LLMC_FFN_HEADER_OFFSET, words, sizeof(words));
    // Unit-scale writers retain the schema1 encoding. Schema2 owns the
    // formerly reserved word89, before the parameter count at90/91.
    static_assert(sizeof(int) == sizeof(float), "FFN gate scale word size mismatch");
    if (gate_scale == 1.0f) h[LLMC_FFN_GATE_SCALE_WORD] = 0;
    else std::memcpy(h + LLMC_FFN_GATE_SCALE_WORD,&gate_scale,sizeof(float));
}
inline float llmc_ffn_contract_gate_scale(const int* h) {
    // Call after llmc_valid_ffn_contract; the valid schema2 value is exactly2.
    // Schema1 never owned word89: preserve its old admission and meaning
    // even if a historical reserved word contains nonzero bytes.
    return h[LLMC_FFN_HEADER_OFFSET+1] == 1 ? 1.0f : 2.0f;
}
inline bool llmc_valid_ffn_contract(const int* h) {
    const int* f = h + LLMC_FFN_HEADER_OFFSET;
    if (!llmc_ffn_valid_shape(f[2], f[3], f[4])) return false;
    if (f[1] != 1 && (f[1] != 2 || f[2] != LLMC_MLP_TIED_GUM_RELU_SQUARED ||
            h[LLMC_FFN_GATE_SCALE_WORD] != LLMC_FFN_GATE_SCALE_TWO_BITS)) return false;
    int expected[256] = {};
    llmc_store_ffn_contract(expected, f[0], f[2], f[3], f[4],llmc_ffn_contract_gate_scale(h));
    return !std::memcmp(f, expected + LLMC_FFN_HEADER_OFFSET, 9*sizeof(int));
}
inline bool llmc_ffn_contract_matches(const int* h, int mode, int hidden, int block,
        float gate_scale = 1.0f) {
    return llmc_valid_ffn_contract(h) && h[82] == mode && h[83] == hidden && h[84] == block &&
        llmc_ffn_contract_gate_scale(h) == gate_scale;
}
#endif
