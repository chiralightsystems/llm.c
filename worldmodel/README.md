# Worldmodel integration

Search tags: External experiments; implementation; `wm:llmc`; `wm:decode`;
`wm:muon`; `wm:provenance`.

`kyin/worldmodel-checkout` is the ChiralightSystems working integration branch.
It gathers the downstream llm.c work previously spread across fork branches,
a detached classifier-fix commit, and the parent worldmodel repository.

## Included work

| Surface | Location and origin |
| --- | --- |
| Muon/NorMuon, rectangular CacheMuon, adaptive tracking and LR diagnostics | Trainer and `llmc/` headers; existing history through `5a8207c` |
| Basis-free NorMuon and CacheMuon inverse-root research modes | Merged `15ca124`; retained as opt-in research modes |
| Direct uint32 NPY input, row boundaries/order, attention ablations, RoPE/DC positions, widened lexical embeddings | Existing midpoint history through `247f438` |
| Small-vocabulary classifier tail fix | Merged `0e09f14`, preserving the formerly detached commit |
| cuDNN workspace allocation/diagnostics, validation timing, LayerNorm bounds, Linux C++20 compatibility, explicit inference NoPE descriptor | Parent decoder prerequisite patches integrated into trainer/headers |
| Cached-KV inference and primitive/numerical tests | [`decode/`](decode/), imported unchanged from parent commit `d6d74f6a` |
| Standalone decoder preparation, build, qualification and run tools | [`scripts/`](scripts/), adapted from the parent tools |
| Frozen full-row scoring and CPU target/loss-alignment checks | [`evaluation/`](evaluation/) and [`tests/test_grammar_score_host.cpp`](tests/test_grammar_score_host.cpp) |
| Attention allocation/forward/backward diagnostic | [`tests/test_attention_memory.cu`](tests/test_attention_memory.cu) |

The [import inventory](provenance.json) records full source commit identities,
original paths, and SHA-256 hashes. Archived prerequisite patches in
[`history/decode_patches/`](history/decode_patches/) preserve their exact inputs;
they are already integrated and must not be applied again to this branch.

## Cached decoder

[`decode/cached_decode.cu`](decode/cached_decode.cu) supplies a separate
inference benchmark driver around `train_gpt2.cu`. Each layer has its own BF16
K/V storage, requests have independent positions, and one-token cuDNN SDPA
reads the valid cached prefix. The driver supports true B1/B8, CUDA graph
replay, GPU greedy selection, Medium RoPE with lexical width 4096, and XL NoPE
with lexical width 8192. This does not replace the trainer's original
full-prefix generation loop.

The shared NoPE descriptor is restricted to the inference driver. The ordinary
trainer rejects it because its checkpoint/state formats and multi-GPU gradient
layout do not support NoPE; this consolidation does not introduce a new format.

The control uses cold initialized weights, fixed-length greedy generation,
and no optimizer/gradient/master-parameter allocations. It does not establish
model quality and does not supply a trained-checkpoint serving API, stochastic
sampling, or EOS early termination. Preserve the declared BF16 storage,
FP32 accumulation/normalization, fused GELU, and logits arithmetic when
comparing it with another implementation.

Preparation archives this llm.c checkout's committed HEAD, including decoder
sources. Dirty working files are excluded. Compilation verifies that immutable
snapshot and writes a fresh `attempt_###` with source/dependency/binary hashes.
Later checkout changes do not change an existing prepared snapshot. No parent
worldmodel checkout or historical worktree is needed to prepare the source.

The current build contract remains Linux/WSL, CUDA 13.3, SM120, C++20, BF16,
cuDNN frontend and cuBLASLt. Provide installed dependency paths explicitly:

```bash
python worldmodel/scripts/build_cached_decode.py --prepare --compile \
  --source-dir artifacts/cached_decode_src \
  --build-dir artifacts/cached_decode_build \
  --nvcc /usr/local/cuda-13.3/bin/nvcc \
  --cudnn-root /path/to/cudnn \
  --frontend-include /path/to/cudnn-frontend/include
```

Preparation/build does not launch a GPU workload. Missing dependencies are
errors. A subsequent `--compile` can use the same prepared directories;
changed source requires a fresh pair. Run CPU packaging tests with:

```bash
python -m unittest discover -s worldmodel/tests -p 'test_cached_decode_*.py'
```

GPU qualification and throughput remain separate, explicitly launched work:

```bash
python worldmodel/scripts/check_cached_decode.py \
  --build-attempt artifacts/cached_decode_build/attempt_001 \
  --gpu YOUR_GPU_UUID --output-dir artifacts/decode_checks

python worldmodel/scripts/run_cached_decode.py \
  --binary artifacts/cached_decode_build/attempt_001/cached_decode \
  --build-manifest artifacts/cached_decode_build/attempt_001/compile_manifest.json \
  --gpu YOUR_GPU_UUID --output-dir artifacts/decode_run
```

Use each command's `--help` to select numerical qualification, memory checks,
prefix/capacity, model profile and repetitions. Hardware admission retains
the original RTX 5070 scope. Select an actual successful attempt; every output
directory must be fresh. The timed workload retains complete greedy/body/head
steps, excluding setup, prefill and final download. Explicit model, precision,
hardware, batch, prefix, capacity and timing scope belong in every comparison.

## Qualification and historical evidence

The decoder CUDA sources are unchanged from the preserved parent import.
The combined optimizer/trainer history and standalone Python packaging form a
new integration revision. Historical GPU qualification and benchmark results
do not qualify a binary compiled from this new revision. CPU packaging tests,
source hashes and review establish packaging integrity; a future numerical
qualification must be run and recorded before reporting new measurements.

The original exact recipes, qualification limits, measurements, and build
identities remain in the parent repository:

- [Cached decode contract and historical results](https://github.com/chiralightsystems/worldmodel/blob/d6d74f6a1bcf725acb858a8629e917df2fb73bc2/experiments/llmc_muon/docs/cached_decode.md)
- [Frozen decode campaign archive](https://github.com/chiralightsystems/worldmodel/tree/d6d74f6a1bcf725acb858a8629e917df2fb73bc2/experiments/exp_1/reports/frozen_decode_preservation)
- [LLMC campaign navigation](https://github.com/chiralightsystems/worldmodel/blob/a822579769cd433bb203f96f19d5468389cdacde/experiments/llmc_muon/docs/README.md)

Historical parent harnesses continue to reconstruct their original pinned
inputs. They are preserved reproduction tools, not the new fork build path.
Datasets, checkpoints, comparisons with FURINA, campaign orchestrators and
shared parent Python services remain owned by worldmodel. The trained naming
generator also remains there because it imports FRNA's shared sampling policy;
it is distinct from this cold-weight cached-decode driver. No run artifacts or
machine-specific credentials belong in this branch.

The imported evaluation sources are adjuncts, not additional executables of
the cached-decode builder. `grammar_score.cu` retains its Windows evaluation
I/O contract and parent campaign launcher. Its host-only fixture can be built
with a C++17 compiler and `-Iworldmodel/evaluation`. The attention diagnostic
requires the same trainer/cuDNN build inputs and runs each explicit
`LLMC_CUDNN_ATTENTION_DETERMINISTIC_BACKWARD=0/1` policy in a separate process.

This branch changes downstream implementation packaging. Canonical worldmodel
architecture, training recipes, benchmark definitions and historical results
remain governed by the parent paper and companions.
