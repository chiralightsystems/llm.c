#include "grammar_score_host.h"
#include <limits>

int main() {
    using namespace grammar_score;
    const uint32_t x[] = {50256, 17, 21, 0, 0, 0, 0, 0, 50256, 9, 8, 7};
    const int64_t y[] = {-100,17,21,-100, -100,-100,-100,-100, -100,9,8,7};
    const auto target = shifted_targets(x, y, 3, 4);
    require(target == std::vector<int>({17,21,0,0, 0,0,0,0, 9,8,7,0}), "Target shift/row boundary mismatch");
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float loss[] = {0.25f,1.5f,nan,nan, nan,nan,nan,nan, 0.5f,0.125f,2.25f,nan};
    const auto first = reduce(loss,y,4), empty = reduce(loss+4,y+4,4), full = reduce(loss+8,y+8,4);
    require(first.count == 2 && first.sum == 1.75 && !first.selected[2], "BOS/last sentence target selection mismatch");
    require(empty.count == 0 && empty.sum == 0, "Dummy padding contributes loss");
    require(full.count == 3 && full.sum == 2.875, "Final usable target missing");
    const float large[] = {16777216.0f,1.0f,0.5f,0};
    require(reduce(large,y+8,4).sum == 16777217.5, "Must sum FP32 observations in FP64");
    bool failed = false;
    try { reduce(loss+2,y,4); } catch (const std::runtime_error&) { failed = true; }
    require(failed,"Selected nonfinite loss was accepted");
    auto invalid = std::vector<int64_t>(y,y+12); invalid[1] = -100;
    failed = false;
    try { shifted_targets(x,invalid.data(),3,4); } catch (const std::runtime_error&) { failed = true; }
    require(failed,"Broken prefix was accepted");
    // Same 256 targets after four real-history lengths, all physically T2048.
    const size_t width = 2048;
    const size_t histories[] = {1, 128, 512, 1792};
    std::vector<uint32_t> px(4 * width, 0);
    std::vector<int64_t> py(4 * width, kIgnore);
    for (size_t b = 0; b < 4; ++b) {
        const size_t row = b * width, h = histories[b];
        for (size_t t = 0; t < h; ++t) px[row + t] = static_cast<uint32_t>(31 + t);
        for (size_t t = 0; t < 256; ++t) {
            px[row + h + t] = static_cast<uint32_t>(100 + t);
            py[row + h + t] = static_cast<int64_t>(100 + t);
        }
    }
    const auto pt = shifted_targets(px.data(), py.data(), 4, width, RowPolicy::PackedSuffix256);
    for (size_t b = 0; b < 4; ++b) {
        const size_t row = b * width, h = histories[b];
        std::vector<float> ploss(width, nan);
        for (size_t t = 0; t < width; ++t) {
            const bool valid = t >= h - 1 && t < h + 255;
            require(pt[row + t] == (valid ? static_cast<int>(101 + t - h) : 0), "Packed target shift mismatch");
            if (valid) ploss[t] = 0.25f;
        }
        const auto scored = reduce(ploss.data(), py.data() + row, width);
        require(scored.count == 256 && scored.sum == 64, "Packed reduction must ignore all warm-up and padding");
        for (size_t t = 0; t + 1 < width; ++t)
            require(scored.selected[t] == (t >= h - 1 && t < h + 255), "Packed token NLL alignment mismatch");
    }
    auto rejects = [&](std::vector<uint32_t> bx, std::vector<int64_t> by) {
        bool rejected = false;
        try { shifted_targets(bx.data(), by.data(), 4, width, RowPolicy::PackedSuffix256); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "Invalid packed suffix accepted");
    };
    auto bad_y = py; bad_y[5] = kIgnore; rejects(px, bad_y); // Interior gap.
    bad_y = py; bad_y[256] = kIgnore; auto bad_x = px; bad_x[256] = 0; rejects(bad_x, bad_y); // 255 targets.
    bad_y = py; bad_y[257] = 400; bad_x = px; bad_x[257] = 400; rejects(bad_x, bad_y); // 257 targets.
    bad_x = px; bad_x[300] = 17; rejects(bad_x, py); // Nonzero tail padding.
    bad_x = px; bad_x[1] = 18; rejects(bad_x, py); // Label/input disagreement.
    bad_x = px; bad_x[width] = kVocab; rejects(bad_x, py); // Invalid warm-up input.
    bad_y = py; bad_y[width + 3] = -7; rejects(px, bad_y); // Invalid ignored label.
    bad_y = py; bad_y[0] = px[0]; rejects(px, bad_y); // Scored row start.
    failed = false;
    try { shifted_targets(px.data(), py.data(), 4, width); }
    catch (const std::runtime_error&) { failed = true; }
    require(failed, "Grammar default must still reject real packed starts");
    // Original packed rows score all T-1 targets, including real EOS positions.
    std::vector<uint32_t> fx(2 * width);
    for (size_t t = 0; t < fx.size(); ++t) fx[t] = static_cast<uint32_t>(t % kVocab);
    fx[0] = kBos; fx[1] = kBos; fx[147] = kBos; fx[width - 1] = kBos;
    fx[width] = 0; fx[width + 1] = 0; fx[width + 200] = kBos;
    std::vector<int64_t> fy(fx.begin(), fx.end());
    fy[0] = kIgnore; fy[width] = kIgnore;
    const auto ft = shifted_targets(fx.data(), fy.data(), 2, width, RowPolicy::PackedFullRow);
    for (size_t b = 0; b < 2; ++b) {
        const size_t row = b * width;
        std::vector<float> losses(width, 0.5f); losses[width - 1] = nan;
        for (size_t t = 0; t + 1 < width; ++t)
            require(ft[row + t] == static_cast<int>(fx[row + t + 1]), "Full-row target shift mismatch");
        require(ft[row + width - 1] == 0, "Final classifier target must remain a placeholder");
        const auto scored = reduce(losses.data(), fy.data() + row, width);
        require(scored.count == width - 1 && scored.sum == 1023.5, "Full row must score every shifted token");
        for (bool selected : scored.selected) require(selected, "Full row unexpectedly masked a target");
    }
    auto full_rejects = [&](std::vector<uint32_t> bx, std::vector<int64_t> by) {
        bool rejected = false;
        try { shifted_targets(bx.data(), by.data(), 2, width, RowPolicy::PackedFullRow); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "Invalid full row accepted");
    };
    bad_y = fy; bad_y[147] = kIgnore; full_rejects(fx, bad_y); // EOS may not be excluded.
    bad_y = fy; bad_y[width - 1] = kIgnore; full_rejects(fx, bad_y); // Last target is scored.
    bad_y = fy; bad_y[0] = kBos; full_rejects(fx, bad_y); // First input is context only.
    bad_y = fy; bad_y[2] += 1; full_rejects(fx, bad_y); // Input/label disagreement.
    bad_x = fx; bad_x[width] = kVocab; full_rejects(bad_x, fy); // Start ID must be valid.
    // Append one explicit empty lane without changing any real-row targets.
    auto dx = fx; dx.resize(3 * width, 0);
    auto dy = fy; dy.resize(3 * width, kIgnore);
    const auto dt = shifted_targets(dx.data(), dy.data(), 3, width, RowPolicy::PackedFullRow);
    for (size_t t = 0; t < 2 * width; ++t) require(dt[t] == ft[t], "Dummy lane changed real-row alignment");
    for (size_t t = 2 * width; t < 3 * width; ++t) require(dt[t] == 0, "Dummy classifier targets must be placeholders");
    std::vector<float> dummy_losses(width, nan);
    const auto dummy_score = reduce(dummy_losses.data(), dy.data() + 2 * width, width);
    require(dummy_score.count == 0 && dummy_score.sum == 0, "Dummy lane contributes loss or count");
    for (bool selected : dummy_score.selected) require(!selected, "Dummy lane has a selected token");
    auto dummy_rejects = [&](std::vector<uint32_t> bx, std::vector<int64_t> by) {
        bool rejected = false;
        try { shifted_targets(bx.data(), by.data(), 3, width, RowPolicy::PackedFullRow); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "Mixed dummy or partially masked real row was accepted");
    };
    bad_x = dx; bad_x[2 * width] = 17; dummy_rejects(bad_x, dy); // Real context cannot be an empty lane.
    bad_x = dx; bad_x[3 * width - 1] = kBos; dummy_rejects(bad_x, dy); // No concealed source token.
    bad_y = dy; bad_y[2 * width + 2] = 0; dummy_rejects(dx, bad_y); // No mixed selected/ignored dummy.
    bad_y = dy; bad_y[1] = kIgnore; dummy_rejects(dx, bad_y); // No dropping the first real target.
    bad_y = dy; bad_y[2 * width + 1] = 0; dummy_rejects(dx, bad_y); // Partially scored row is invalid.
    // An actual all-token-zero row is fully scored when labels declare it real.
    auto zero_y = dy;
    for (size_t t = 2 * width + 1; t < 3 * width; ++t) zero_y[t] = 0;
    shifted_targets(dx.data(), zero_y.data(), 3, width, RowPolicy::PackedFullRow);
    const std::vector<float> zero_losses(width, 0.5f);
    require(reduce(zero_losses.data(), zero_y.data() + 2 * width, width).count == width - 1,
            "Real token-zero row was mistaken for a dummy");
    // Variable final-word suffixes preserve real starts, final targets and
    // token-zero content without inserting any synthetic BOS/EOS.
    const std::vector<uint32_t> wx = {19,20,31,32,0,0, 0,3,4,5,6,7, 0,0,0,0,0,0};
    const std::vector<int64_t> wy = {-100,-100,31,32,-100,-100,
        -100,-100,-100,-100,-100,7, -100,-100,-100,-100,-100,-100};
    const auto wt = shifted_targets(wx.data(), wy.data(), 3, 6, RowPolicy::PackedFinalWord);
    require(wt == std::vector<int>({0,31,32,0,0,0, 0,0,0,0,7,0, 0,0,0,0,0,0}),
            "Variable final-word target shift mismatch");
    auto word_rejects = [&](std::vector<uint32_t> bx, std::vector<int64_t> by) {
        bool rejected = false;
        try { shifted_targets(bx.data(), by.data(), 3, 6, RowPolicy::PackedFinalWord); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "Invalid final-word row accepted");
    };
    bad_y = wy; bad_y[3] = kIgnore; bad_y[4] = 0; bad_x = wx; bad_x[3] = 0; word_rejects(bad_x, bad_y); // Gap in final word.
    bad_y = wy; bad_y[0] = 19; word_rejects(wx, bad_y); // No preceding context.
    bad_x = wx; bad_x[4] = 100; word_rejects(bad_x, wy); // Nonzero right pad.
    bad_x = wx; bad_x[2] = 33; word_rejects(bad_x, wy); // Wrong selected input.
    bad_x = wx; bad_x[0] = kVocab; word_rejects(bad_x, wy); // Invalid real prefix.
    bad_y = wy; bad_y[1] = -2; word_rejects(wx, bad_y); // Invalid mask value.
    bad_x = wx; bad_x[12] = 3; word_rejects(bad_x, wy); // Empty lane hides context.
    bad_y = wy; bad_y[2] = bad_y[3] = kIgnore; word_rejects(wx, bad_y); // Real row without a final word.
    const float tie[] = {-2.f, 3.f, 3.f, 100.f};
    require(argmax_lowest(tie, 3) == 1, "Argmax must exclude padded vocabulary and choose lowest tied ID");
    const float negative[] = {-3.f, -1.f, -2.f};
    require(argmax_lowest(negative, 3) == 1, "Negative logits were mishandled");
    const float binary_logits[] = {0.f, 1.f};
    require(std::abs(observed_logit_nll(binary_logits, 2, 1) - std::log1p(std::exp(-1.0))) < 1e-14,
            "Observed logsumexp must use the exact scored target and vocabulary");
    const float shifted_logits[] = {-100.f, -99.f};
    require(std::abs(observed_logit_nll(shifted_logits, 2, 1) - observed_logit_nll(binary_logits, 2, 1)) < 1e-14,
            "Observed NLL must be invariant to a common logit offset");
    require(observed_logit_nll(tie, 3, 1) == observed_logit_nll(tie, 3, 2),
            "Equal-probability tied labels remain distinct argmax IDs");
    const float invalid_logits[] = {0.f, nan};
    failed = false;
    try { argmax_lowest(invalid_logits, 2); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "Nonfinite greedy logit was accepted");
    auto greedy_first = first, greedy_empty = empty;
    observe_greedy(greedy_first, y, {17,21,-1});
    observe_greedy(greedy_empty, y + 4, {-1,-1,-1});
    require(greedy_first.greedy_exact_match && !greedy_empty.greedy_exact_match,
            "Greedy all-subtoken/dummy exact-match mismatch");
    auto wrong_last = first;
    observe_greedy(wrong_last, y, {17,22,-1});
    require(!wrong_last.greedy_exact_match && wrong_last.sum == first.sum,
            "One wrong final-word subtoken must fail without changing NLL");
    failed = false;
    try { observe_greedy(wrong_last, y, {17,21,0}); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "Ignored position acquired a greedy observation");
    write(stdout,greedy_first,0,0,0); write(stdout,greedy_empty,0,1,1); write(stdout,full,0,2,2);
    return 0;
}
