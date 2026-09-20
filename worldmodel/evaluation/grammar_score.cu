// Frozen full-row scoring; all model mathematics stay in the pinned source.
#define TESTING
#include "train_gpt2.cu"
#include "grammar_score_host.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

using grammar_score::require;

static void allocate_eval_state(GPT2* model, int B, int T) {
    // Use the stock activation layout without gradients or optimizer allocations.
    model->batch_size = B;
    model->seq_len = T;
    if (model->config.position_encoding == LLMC_POSITION_ENCODING_ROPE) {
        require(llmc_rope_cache_allocate(&model->rope_cache,
                    model->config.max_seq_len, model->config.rope_rotary_dim,
                    model->config.rope_theta, model->config.rope_lowest_frequency_plane_is_dc,
                    main_stream), "RoPE phase cache allocation failed");
    }
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, T, model->config, model->recompute);
    model->acts_memory_bytes = activation_allocation_bytes(model->acts_specs);
    model->acts_memory = malloc_and_point_activations(model->acts_specs);
    cudaCheck(cudaMalloc((void**)&model->inputs, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&model->targets, (size_t)B * T * sizeof(int)));
    cudaCheck(cudaMallocHost((void**)&model->cpu_losses, (size_t)B * T * sizeof(float)));
}

struct GreedyObservation {
    std::vector<int> ids;
    size_t audited_targets = 0;
    double maximum_nll_error = 0;
};

// FP32 exp/log/reduction roundoff versus independent FP64 logsumexp;
// this is an observation-identity check, not a change to reported NLL.
static constexpr double kGreedyNllAuditTolerance = 1e-4;

static GreedyObservation validate_with_greedy_observation(GPT2* model,
        const int* inputs, const int* targets, const int64_t* labels, size_t B, size_t T,
        int blackout_width, bool attention_disabled) {
    // Preserve the qualified validation's forward, classifier and loss
    // arithmetic. Its misleading `False` alias is bool_constant<true> and
    // overwrites logits with gradients; therefore observe BEFORE classifier.
    // No second forward and no change to the pinned alias or CUDA source.
    gpt2_forward(model, inputs, B, T, blackout_width, attention_disabled);
    const size_t V = model->config.vocab_size, Vp = model->config.padded_vocab_size;
    std::vector<floatX> host_logits(V);
    std::vector<double> observed_nll(B * T, 0);
    GreedyObservation result;
    result.ids.assign(B * T, -1);
    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t + 1 < T; ++t) {
            const size_t index = b * T + t;
            if (labels[index + 1] == grammar_score::kIgnore) continue;
            cudaCheck(cudaMemcpy(host_logits.data(), model->acts.output + index * Vp,
                V * sizeof(floatX), cudaMemcpyDeviceToHost));
            result.ids[index] = grammar_score::argmax_lowest(host_logits.data(), V);
            observed_nll[index] = grammar_score::observed_logit_nll(
                host_logits.data(), V, targets[index]);
        }
    }
    NvtxRange classifier_and_loss_range("classifier_and_loss");
    ActivationTensors acts = model->acts;
    const size_t supervised_targets = B * (T - 1);
    const float dloss = 1.0f / supervised_targets;
    cudaCheck(cudaMemset(acts.losses, 0, B * T * sizeof(float)));
    cudaCheck(cudaMemcpy(model->targets, targets, B * T * sizeof(int), cudaMemcpyHostToDevice));
    tokenCheck(targets, B * T, V);
    fused_classifier(acts.output, acts.losses, dloss, model->targets,
        B, T, V, Vp, False, main_stream, true);
    cudaCheck(cudaMemcpy(model->cpu_losses, acts.losses, B * T * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaDeviceSynchronize());
    for (size_t index = 0; index < B * T; ++index) {
        if (result.ids[index] < 0) continue;
        const double actual = model->cpu_losses[index];
        const double error = std::abs(actual - observed_nll[index]);
        require(std::isfinite(actual) && error <= kGreedyNllAuditTolerance,
                "Pre-classifier logit audit disagrees with ordinary token NLL");
        require(actual >= std::log(2.0) - kGreedyNllAuditTolerance || result.ids[index] == targets[index],
                "A greater-than-half-probability target failed exact argmax sanity check");
        result.maximum_nll_error = std::max(result.maximum_nll_error, error);
        ++result.audited_targets;
    }
    return result;
}

static uint64_t number(const std::string& value) {
    require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
            "Expected an unsigned integer option");
    return std::stoull(value);
}

static void seek_read(FILE* file, uint64_t offset, void* data, size_t bytes) {
    require(offset <= INT64_MAX, "Input offset overflow");
    require(_fseeki64(file, static_cast<int64_t>(offset), SEEK_SET) == 0, "Input seek failed");
    require(fread(data, 1, bytes, file) == bytes, "Truncated cache payload");
}

int main(int argc, char** argv) {
    try {
        const char* names[] = {"--checkpoint", "--inputs", "--labels", "--input-offset",
            "--label-offset", "--batch-size", "--sequence-length", "--batches", "--row-offset",
            "--scores", "--result", "--row-policy", "--export-greedy",
            "--blackout-width", "--attention-disabled", "--allow-rope-extrapolation"};
        std::map<std::string, std::string> options;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc, "Missing option value");
            require(std::find(std::begin(names), std::end(names), std::string(argv[i])) != std::end(names),
                    "Unknown option");
            require(options.emplace(argv[i], argv[i + 1]).second, "Duplicate option");
        }
        for (const char* name : names) {
            if (std::string(name) != "--row-policy" && std::string(name) != "--export-greedy" &&
                std::string(name) != "--blackout-width" && std::string(name) != "--attention-disabled" &&
                std::string(name) != "--allow-rope-extrapolation")
                require(options.count(name) == 1, "Missing driver option; use run_grammar_eval.py");
        }
        const std::string row_policy_name = options.count("--row-policy") ? options.at("--row-policy") : "grammar";
        require(row_policy_name == "grammar" || row_policy_name == "packed-suffix-256" ||
                row_policy_name == "packed-full-row" || row_policy_name == "packed-final-word", "Unknown row policy");
        const auto row_policy = row_policy_name == "grammar" ? grammar_score::RowPolicy::GrammarBos :
            row_policy_name == "packed-suffix-256" ? grammar_score::RowPolicy::PackedSuffix256 :
            row_policy_name == "packed-full-row" ? grammar_score::RowPolicy::PackedFullRow :
            grammar_score::RowPolicy::PackedFinalWord;
        const uint64_t greedy_option = options.count("--export-greedy") ? number(options.at("--export-greedy")) : 0;
        require(greedy_option <= 1, "Greedy export must be 0 or 1");
        const bool export_greedy = greedy_option != 0;
        require(row_policy != grammar_score::RowPolicy::PackedFinalWord || export_greedy,
                "Final-word evaluation requires exact greedy observation");
        const size_t B = number(options.at("--batch-size"));
        const size_t T = number(options.at("--sequence-length"));
        const size_t batches = number(options.at("--batches"));
        const size_t row_offset = number(options.at("--row-offset"));
        require(B > 0 && B <= 4096 && T >= 2 && T <= 8192 && batches > 0,
                "Invalid evaluation shape");
        const uint64_t blackout_width = options.count("--blackout-width") ? number(options.at("--blackout-width")) : 0;
        const uint64_t attention_disabled = options.count("--attention-disabled") ? number(options.at("--attention-disabled")) : 0;
        const uint64_t allow_extrapolation = options.count("--allow-rope-extrapolation") ? number(options.at("--allow-rope-extrapolation")) : 0;
        require(blackout_width < T && attention_disabled <= 1 && allow_extrapolation <= 1,
                "Invalid frozen attention or extrapolation setting");
        require(!(blackout_width && attention_disabled), "Blackout and attention-off are separate interventions");
        require(batches <= (SIZE_MAX - row_offset) / B, "Row range overflow");
        FILE* inputs = fopenCheck(options.at("--inputs").c_str(), "rb");
        FILE* labels = fopenCheck(options.at("--labels").c_str(), "rb");
        const uint64_t input_offset = number(options.at("--input-offset"));
        const uint64_t label_offset = number(options.at("--label-offset"));
        const size_t row_end = row_offset + batches * B;
        require(row_end <= UINT64_MAX / (T * sizeof(int64_t)), "Cache row range overflow");
        require(input_offset <= UINT64_MAX - row_end * T * sizeof(uint32_t) &&
                label_offset <= UINT64_MAX - row_end * T * sizeof(int64_t), "Cache offset overflow");
        require(std::filesystem::file_size(options.at("--inputs")) >= input_offset + row_end * T * sizeof(uint32_t) &&
                std::filesystem::file_size(options.at("--labels")) >= label_offset + row_end * T * sizeof(int64_t),
                "Cache range exceeds input payload");
        require(!std::filesystem::exists(options.at("--result")), "Result already exists");
        FILE* scores = fopenCheck(options.at("--scores").c_str(), "wbx");
        const auto start = std::chrono::steady_clock::now();
        multi_gpu_config = multi_gpu_config_init(1, 0, 1, nullptr, nullptr, nullptr);
        common_start(true, true);
        GPT2 model = {};
        gpt2_init_common(&model);
        model.use_master_weights = 0;
        // Preserve the trained controls' declared numerical policy.
        model.gelu_fusion = 2;
        model.recompute = 2;
        gpt2_build_from_checkpoint(&model, options.at("--checkpoint").c_str());
        const int checkpoint_max_seq_len = model.config.max_seq_len;
        if (allow_extrapolation) {
            require(model.config.position_encoding == LLMC_POSITION_ENCODING_ROPE,
                    "Evaluation capacity extension requires a RoPE checkpoint");
            // Same evaluation-only cache extension as the pinned trainer's -mt.
            // No frequency rescaling, position interpolation, or parameter edit.
            model.config.max_seq_len = std::max(model.config.max_seq_len, static_cast<int>(T));
        }
        require(model.config.vocab_size == grammar_score::kVocab && T <= (size_t)model.config.max_seq_len,
                "Checkpoint is incompatible with this token cache");
        allocate_eval_state(&model, static_cast<int>(B), static_cast<int>(T));
        std::vector<uint32_t> x(B * T);
        std::vector<int64_t> y(B * T);
        std::vector<int> model_inputs(B * T);
        size_t selected = 0;
        size_t greedy_audited_targets = 0;
        double greedy_maximum_nll_error = 0;
        const auto eval_start = std::chrono::steady_clock::now();
        for (size_t batch = 0; batch < batches; ++batch) {
            const size_t first = row_offset + batch * B;
            seek_read(inputs, input_offset + first * T * sizeof(uint32_t), x.data(), x.size() * sizeof(uint32_t));
            seek_read(labels, label_offset + first * T * sizeof(int64_t), y.data(), y.size() * sizeof(int64_t));
            const auto targets = grammar_score::shifted_targets(x.data(), y.data(), B, T, row_policy);
            for (size_t i = 0; i < x.size(); ++i) model_inputs[i] = static_cast<int>(x[i]);
            // Aggregate mean includes placeholder padding targets; deliberately unused.
            GreedyObservation observation;
            if (export_greedy) {
                observation = validate_with_greedy_observation(&model,
                    model_inputs.data(), targets.data(), y.data(), B, T,
                    static_cast<int>(blackout_width), attention_disabled != 0);
                greedy_audited_targets += observation.audited_targets;
                greedy_maximum_nll_error = std::max(greedy_maximum_nll_error, observation.maximum_nll_error);
            } else {
                (void)gpt2_validate(&model, model_inputs.data(), targets.data(), B, T, true,
                    static_cast<int>(blackout_width), attention_disabled != 0, 0);
            }
            for (size_t lane = 0; lane < B; ++lane) {
                auto row = grammar_score::reduce(model.cpu_losses + lane * T, y.data() + lane * T, T);
                if (export_greedy) {
                    std::vector<int> greedy(observation.ids.begin() + lane * T,
                        observation.ids.begin() + lane * T + T - 1);
                    grammar_score::observe_greedy(row, y.data() + lane * T, greedy);
                }
                grammar_score::write(scores, row, batch, lane, batch * B + lane);
                selected += row.count;
            }
            require(fflush(scores) == 0, "Failed to flush scores");
            if ((batch + 1) % 25 == 0 || batch + 1 == batches) {
                printf("grammar batches %zu/%zu rows %zu selected_targets %zu\n", batch + 1, batches, (batch + 1) * B, selected);
                fflush(stdout);
            }
        }
        const double eval_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - eval_start).count();
        fcloseCheck(inputs);
        fcloseCheck(labels);
        fcloseCheck(scores);
        FILE* result = fopenCheck(options.at("--result").c_str(), "wbx");
        fprintf(result, "{\"schema\":\"worldmodel.llmc.grammar_eval_result.v1\",\"status\":\"completed\","
            "\"rows\":%zu,\"scored_targets\":%zu,\"batch_size\":%zu,\"batches\":%zu,\"sequence_length\":%zu,"
            "\"source_row_offset\":%zu,\"precision\":\"BF16\",\"gelu_fusion\":2,\"recompute\":2,"
            "\"row_policy\":\"%s\","
            "\"blackout_width\":%llu,\"attention_disabled\":%s,\"allow_rope_extrapolation\":%s,"
            "\"checkpoint_max_sequence_length\":%d,\"effective_max_sequence_length\":%d,"
            "\"greedy_exported\":%s,"
            "\"greedy_observation\":\"pre_classifier_original_bf16_logits_v1\","
            "\"greedy_audited_targets\":%zu,\"greedy_nll_audit_absolute_tolerance\":%.17g,"
            "\"greedy_maximum_nll_audit_error\":%.17g,"
            "\"rope_lowest_frequency_plane_is_dc\":%d,\"activation_bytes\":%zu,"
            "\"evaluation_seconds\":%.17g,\"elapsed_seconds\":%.17g,\"checkpoint_saved\":false}\n",
            batches * B, selected, B, batches, T, row_offset, row_policy_name.c_str(),
            static_cast<unsigned long long>(blackout_width), attention_disabled ? "true" : "false",
            allow_extrapolation ? "true" : "false", checkpoint_max_seq_len, model.config.max_seq_len,
            export_greedy ? "true" : "false",
            greedy_audited_targets, kGreedyNllAuditTolerance, greedy_maximum_nll_error, model.config.rope_lowest_frequency_plane_is_dc,
            model.acts_memory_bytes, eval_seconds,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        require(!ferror(result), "Failed to write result");
        fcloseCheck(result);
        gpt2_free(&model);
        common_free(model);
        multi_gpu_config_free(&multi_gpu_config);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Grammar evaluation failed: %s\n", error.what());
        return 1;
    }
}
