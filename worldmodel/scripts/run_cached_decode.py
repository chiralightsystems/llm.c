"""Run isolated cold widened GPT-2 cached decode controls on the RTX 5070.

Search tags: External experiments; wm:llmc; wm:throughput; wm:decode.
Every repetition is a new process and every invocation owns a fresh run directory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import signal
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
GPU_NAME = "NVIDIA GeForce RTX 5070"
PROFILES = {
    "medium-rope-e4096": {"descriptor": "gpt2:rope:d24:t2048:e4096", "layers": 24,
                         "body_width": 1024, "heads": 16, "lexical_width": 4096,
                         "position_policy": "rope", "reference_prefix_lengths": [8, 32, 128, 1024, 1152]},
    "xl-nope-e8192": {"descriptor": "gpt2:nope:d48:t2048:e8192", "layers": 48,
                     "body_width": 1600, "heads": 25, "lexical_width": 8192,
                     "position_policy": "none", "reference_prefix_lengths": [8, 32, 128]},
}


def parameter_bytes(profile_name: str) -> int:
    profile = PROFILES[profile_name]
    C, E, L = (profile[key] for key in ("body_width", "lexical_width", "layers"))
    return 2 * (50304*E + 12*L*C*C + 13*L*C + 2*C + 2*C*E)


def kv_cache_bytes(profile_name: str, batch: int, capacity: int) -> int:
    profile = PROFILES[profile_name]
    return 4 * profile["layers"] * batch * capacity * profile["body_width"]


def profile_matches(metrics: dict, args: argparse.Namespace) -> bool:
    profile = PROFILES[args.model_profile]
    return (metrics.get("model_profile") == args.model_profile and
            all(metrics.get(key) == profile[key] for key in
                ("descriptor", "layers", "body_width", "heads", "lexical_width", "position_policy")) and
            metrics.get("head_dim") == 64 and
            metrics.get("reference_max_prefix") == max(profile["reference_prefix_lengths"]) and
            metrics.get("parameter_bytes") == parameter_bytes(args.model_profile))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    # The worktree is shared with Windows. Match its checkout newline policy.
    return subprocess.check_output(
        ["git", "-c", "core.autocrlf=true", "-c", "diff.ignoreSubmodules=dirty",
         "-C", str(ROOT), *args], text=True
    ).strip()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def resolve_gpu(selector: str) -> str:
    """Resolve an explicit selector while retaining the qualified hardware scope."""
    result = subprocess.run(
        ["nvidia-smi", "--id=" + selector, "--query-gpu=uuid,name",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10)
    rows = list(csv.reader(result.stdout.strip().splitlines()))
    if (len(rows) != 1 or len(rows[0]) != 2 or
            not rows[0][0].strip().startswith("GPU-") or rows[0][1].strip() != GPU_NAME):
        raise RuntimeError("GPU selection must identify one qualified NVIDIA GeForce RTX 5070")
    return rows[0][0].strip()


def runtime_environment(build_record: dict, gpu_uuid: str) -> dict[str, str]:
    """Use the library directories recorded by the successful local build."""
    directories = build_record.get("runtime_library_dirs")
    if not isinstance(directories, list) or not directories:
        raise ValueError("Build manifest must declare runtime_library_dirs")
    resolved = []
    for directory in directories:
        if not isinstance(directory, str) or not directory:
            raise ValueError("Runtime library directories must be nonempty path strings")
        path = Path(directory)
        if not path.is_absolute() or not path.is_dir():
            raise ValueError(f"Runtime library directory is not an existing absolute directory: {directory}")
        if directory not in resolved:
            resolved.append(directory)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=gpu_uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    if env.get("LD_LIBRARY_PATH"):
        resolved.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(resolved)
    return env


def gpu_state(gpu_uuid: str) -> dict:
    command = [
        "nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.used,memory.total,"
        "utilization.gpu,temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits", "--id=" + gpu_uuid,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def memory_state(gpu_uuid: str) -> dict:
    command = ["nvidia-smi", "--id=" + gpu_uuid,
               "--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    rows = list(csv.reader(result.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 7 or rows[0][0].strip() != gpu_uuid or rows[0][1].strip() != GPU_NAME:
        raise RuntimeError("Memory monitor did not identify the authorized RTX 5070")
    row = rows[0]
    return {"utc": datetime.now(timezone.utc).isoformat(), "used_mib": int(row[2]),
            "total_mib": int(row[3]), "utilization_percent": int(row[4]),
            "temperature_c": int(row[5]), "power_w": row[6].strip()}


def terminate_group(process: subprocess.Popen) -> None:
    # Each child owns a new session. Bound cleanup to that process group,
    # including descendants if its leader exits before they do.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def guarded_run(command: list[str], *, env: dict, log, memory_path: Path,
                gpu_uuid: str, batch: int, capacity: int, model_profile: str = "medium-rope-e4096") -> subprocess.CompletedProcess:
    record = {"batch": batch, "capacity": capacity, "model_profile": model_profile, "gpu_uuid": gpu_uuid, "samples": [],
              "reserve_mib": 512, "sample_interval_seconds": 0.25,
              "timeout_seconds": 1200, "stop_reason": None}
    process = None
    try:
        initial = memory_state(gpu_uuid)
        record["idle_settle_samples"] = []
        if initial["used_mib"] <= 256 and initial["utilization_percent"] > 0:
            settle_start = time.monotonic()
            while time.monotonic() - settle_start < 10:
                time.sleep(0.25)
                initial = memory_state(gpu_uuid)
                record["idle_settle_samples"].append(initial)
                if initial["used_mib"] > 256 or initial["utilization_percent"] == 0:
                    break
        record["utilization_settled_to_zero"] = initial["utilization_percent"] == 0
        record["before"] = initial
        # Lower bound preflight here; the driver checks all static allocations
        # and qualification activations after CUDA setup, before weights/KV.
        minimum_bytes = parameter_bytes(model_profile) + kv_cache_bytes(model_profile, batch, capacity) + 1024**3
        record["minimum_model_kv_and_reserve_bytes"] = minimum_bytes
        # Owned earlier children have completed and their groups were cleaned.
        # Utilization reports a recent window, so after bounded settling it is
        # diagnostic; an idle-memory baseline remains mandatory.
        if initial["used_mib"] > 256:
            record["admission_status"] = "device_busy"
            record["stop_reason"] = "Idle-memory preflight rejected; no workload launched"
            raise RuntimeError(record["stop_reason"])
        if minimum_bytes > (initial["total_mib"] - initial["used_mib"] - 512)*1024**2:
            record["admission_status"] = "not_fit"
            record["stop_reason"] = "VRAM preflight rejected: model and KV do not fit with reserves; no workload launched"
            raise RuntimeError(record["stop_reason"])
        record["admission_status"] = "preflight_passed_driver_static_check_pending"
        process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        start = time.monotonic()
        while process.poll() is None:
            sample = memory_state(gpu_uuid)
            record["samples"].append(sample)
            if sample["used_mib"] >= sample["total_mib"] - 512:
                record["stop_reason"] = "VRAM reserve reached; stop suite without retry or smaller batch"
                raise RuntimeError(record["stop_reason"])
            if time.monotonic() - start > 1200:
                record["stop_reason"] = "1200-second per-process watchdog"
                raise RuntimeError(record["stop_reason"])
            time.sleep(0.25)
        record["returncode"] = process.wait(timeout=5)
        if record["returncode"]:
            record["stop_reason"] = "Decoder returned nonzero; stop suite"
        return subprocess.CompletedProcess(command, record["returncode"])
    except BaseException as error:
        if not record["stop_reason"]:
            record["stop_reason"] = "Launch/monitor failure: " + repr(error)
        raise
    finally:
        try:
            if process is not None:
                terminate_group(process)
                record["returncode"] = process.returncode
        except BaseException as error:
            record["cleanup_error"] = repr(error)
            record["stop_reason"] = record["stop_reason"] or "Process-group cleanup failure"
            raise
        finally:
            after_failure = None
            try:
                record["after"] = memory_state(gpu_uuid)
            except BaseException as error:
                record["after_query_error"] = repr(error)
                if not record["stop_reason"]:
                    record["stop_reason"] = "Final memory query failed"
                    after_failure = error
            observations = ([record["before"]] if "before" in record else []) + record["samples"]
            if "after" in record:
                observations.append(record["after"])
            record["sampled_peak_used_mib"] = max((s["used_mib"] for s in observations), default=None)
            write_json(memory_path, record)
            if after_failure is not None:
                raise RuntimeError("Final memory query failed; stop suite") from after_failure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--build-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu", required=True, help="Explicit nvidia-smi GPU index or UUID; must identify an RTX 5070")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--model-profile", choices=PROFILES, default="medium-rope-e4096")
    parser.add_argument("--prefix", type=int, default=1024, help="Resident prompt tokens per request; each measurement generates 128 tokens")
    parser.add_argument("--capacity", type=int, default=2048, help="Allocated KV/RoPE positions; must include all 128 generated commits")
    parser.add_argument("--batches", type=int, nargs="+", choices=[1, 8], default=[1, 8])
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--timing-audit", action="store_true", help="Run a separate reset event/host timing audit after each primary measurement")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if len(args.batches) != len(set(args.batches)):
        raise ValueError("batches must not contain duplicates")
    if not 20 <= args.capacity <= 2**31 - 9:
        raise ValueError("capacity must admit graph warmup and int32 positions")
    if args.qualify:
        required = max(PROFILES[args.model_profile]["reference_prefix_lengths"])
        if args.capacity < required:
            raise ValueError(f"qualification requires capacity at least {required}")
    elif args.prefix < 1 or args.prefix + 128 > args.capacity:
        raise ValueError("prefix must be positive and prefix + 128 must fit capacity")


def decode_command(args: argparse.Namespace, binary: Path, batch: int, output: Path) -> list[str]:
    command = [str(binary), "--batch", str(batch), "--output", str(output),
               "--capacity", str(args.capacity)]
    if args.model_profile != "medium-rope-e4096":
        command.extend(["--model-profile", args.model_profile])
    if args.qualify:
        command.append("--qualify")
    else:
        command.extend(["--prefix", str(args.prefix), "--steps", "128"])
        if args.timing_audit:
            command.append("--timing-audit")
    return command


def workload_matches(metrics: dict, args: argparse.Namespace, batch: int) -> bool:
    if metrics.get("capacity") != args.capacity or metrics.get("memory_admission", {}).get("admitted") is not True:
        return False
    if args.qualify:
        return True
    return (metrics.get("prefix_tokens_per_request") == args.prefix and
            metrics.get("generated_tokens_per_request") == 128 and
            metrics.get("total_generated_tokens") == batch * 128 and
            metrics.get("timer") == "clock_monotonic_raw_graph_submit_and_completion_no_events")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("Cached decode runs require Linux or WSL with CUDA")
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    profile = PROFILES[args.model_profile]
    binary = args.binary.resolve(strict=True)
    manifest = args.build_manifest.resolve(strict=True)
    build_record = json.loads(manifest.read_text())
    binary_hash = digest(binary)
    if (build_record.get("status") != "passed" or
            build_record.get("artifacts", {}).get(binary.name, {}).get("sha256") != binary_hash):
        raise RuntimeError("Binary is not bound to a successful supplied build manifest")
    gpu_uuid = resolve_gpu(args.gpu)
    env = runtime_environment(build_record, gpu_uuid)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "compile_manifest.json").write_bytes(manifest.read_bytes())
    config = {
        "schema": "worldmodel.llmc_cached_decode_run.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short", "--ignore-submodules=dirty"),
        "git_status_scope": "Standalone llm.c checkout; submodule working trees excluded. Dependencies are hashed in build manifest.",
        "cached_submodule_gitlinks": git("ls-files", "--stage", "third_party"),
        "binary": str(binary), "binary_sha256": binary_hash,
        "build_manifest": str(manifest), "build_manifest_sha256": digest(manifest),
        "harness_sha256": digest(Path(__file__)),
        "gpu_selector": args.gpu, "gpu_uuid": gpu_uuid, "gpu_name": GPU_NAME, "gpu_before": gpu_state(gpu_uuid),
        "mode": "qualify" if args.qualify else "benchmark",
        "timing_audit": args.timing_audit,
        "repetitions": 1 if args.qualify else args.repetitions,
        "batches": args.batches, "model_profile": args.model_profile, "descriptor": profile["descriptor"],
        "model_geometry": {key: profile[key] for key in ("layers", "body_width", "heads", "lexical_width", "position_policy")},
        "parameter_bytes": parameter_bytes(args.model_profile),
        "cold_initializer_seed": 42, "checkpoint_lineage": "cold descriptor; no checkpoint",
        "dataset": ("synthetic pattern[(b+t)%4]+b//4, eight distinct rows, generator v1" if args.qualify
                    else "synthetic fixed prefix pattern [464,3797,3332,319], generator v1"),
        "capacity": args.capacity, "gelu_fusion": 2,
        "memory_policy": "Before weights/KV: exact static device bytes plus 1 GiB reserve must fit free VRAM; no offload or batch reduction",
        "memory_monitor": {"sample_interval_seconds": 0.25, "stop_reserve_mib": 512,
                           "preflight_max_used_mib": 256, "utilization_settle_seconds": 10,
                           "utilization_after_settle_is_diagnostic": True,
                           "per_process_timeout_seconds": 1200,
                           "peak_scope": "Sampled device-wide used memory; not an exact process allocation peak"},
        "timing_scope": "128 GPU greedy selections and full cached body/head commits; final head included",
        "excluded_from_decode": ["parameter initialization", "plan and graph creation", "warmup",
                                 "KV reset", "prefill and initial logits", "final status/token download"],
        "samples_policy": "No quality claim; token checksum only",
        "environment": {k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH"]},
    }
    if args.qualify:
        config.update(reference_prefix_lengths=profile["reference_prefix_lengths"], capture_reset_prefix=16,
                      capture_reset_generated_tokens=4, capture_reset_repeats=2)
        config["timing_scope"] = "Untimed numerical qualification; no decode throughput claim"
    else:
        config.update(prefix_tokens_per_request=args.prefix, generated_tokens_per_request=128)
    write_json(output / "config.json", config)
    (output / "implementation.diff").write_text(git("diff", "HEAD", "--", "worldmodel"))
    records = []
    summary: dict = {"schema": "worldmodel.llmc_cached_decode_summary.v1", "config": config,
                     "runs": records, "passed": False}
    for repetition in range(config["repetitions"]):
        for batch in config["batches"]:
            name = f"b{batch}_r{repetition + 1}"
            result_path = output / f"{name}.json"
            command = decode_command(args, binary, batch, result_path)
            record = {"batch": batch, "repetition": repetition + 1, "command": command,
                      "gpu_before": gpu_state(gpu_uuid), "started_utc": datetime.now(timezone.utc).isoformat()}
            print(f"Running {name} ({config['mode']})", flush=True)
            start = time.perf_counter()
            memory_path = output / f"{name}_memory.json"
            record["memory_record"] = str(memory_path)
            try:
                with (output / f"{name}.log").open("w") as log:
                    result = guarded_run(command, env=env, log=log, memory_path=memory_path,
                                         gpu_uuid=gpu_uuid, batch=batch, capacity=args.capacity, model_profile=args.model_profile)
            except BaseException as error:
                record["error"] = repr(error)
                if memory_path.is_file():
                    record["memory"] = json.loads(memory_path.read_text())
                records.append(record)
                write_json(output / "summary.json", summary)
                raise
            record.update(returncode=result.returncode, process_wall_seconds=time.perf_counter() - start,
                          gpu_after=gpu_state(gpu_uuid), stdout=str(output / f"{name}.log"))
            record["memory"] = json.loads(memory_path.read_text())
            records.append(record)
            if result_path.is_file():
                try:
                    record["metrics"] = json.loads(result_path.read_text())
                except json.JSONDecodeError as error:
                    record["metrics_parse_error"] = str(error)
            if result.returncode:
                write_json(output / "summary.json", summary)
                raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")
            metrics = record.get("metrics", {})
            valid = (metrics.get("schema") == "worldmodel.llmc_cached_decode.v1" and
                     metrics.get("mode") == config["mode"] and metrics.get("batch") == batch and
                     metrics.get("descriptor") == config["descriptor"] and
                     metrics.get("hardware") == GPU_NAME and
                     metrics.get("weights") == "cold" and metrics.get("seed") == 42 and
                     metrics.get("gelu_fusion") == 2 and workload_matches(metrics, args, batch) and
                     profile_matches(metrics, args))
            if args.qualify:
                checks = metrics.get("checks", [])
                valid = valid and (
                    [check.get("prefix") for check in checks] == config["reference_prefix_lengths"] and
                    metrics.get("relative_l2_limit") == 0.03 and
                    metrics.get("max_abs_logit_error_limit") == 0.125 and
                    metrics.get("capture_eager_bitwise_equal") is True and
                    metrics.get("reset_repeats") == 2 and
                    all(0 <= check.get("max_row_relative_l2", float("inf")) <= 0.03 and
                        0 <= check.get("max_abs_logit_error", float("inf")) <= 0.125
                        for check in checks))
            if not valid or not metrics.get("passed"):
                write_json(output / "summary.json", summary)
                raise RuntimeError(f"{name} result identity or qualification failed")
            write_json(output / "summary.json", summary)
    if not args.qualify:
        summary["aggregates"] = {}
        for batch in config["batches"]:
            rows = [r["metrics"] for r in records if r["batch"] == batch]
            fields = ["aggregate_tokens_per_second", "tokens_per_second_per_request",
                      "ms_per_decode_step", "decode_wall_ms", "prefill_wall_ms"]
            summary["aggregates"][str(batch)] = {
                key: {"median": statistics.median(r[key] for r in rows),
                      "min": min(r[key] for r in rows), "max": max(r[key] for r in rows)}
                for key in fields
            }
    summary["passed"] = True
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "summary.json", summary)
    print(json.dumps(summary.get("aggregates", {"passed": True}), indent=2), flush=True)


if __name__ == "__main__":
    main()
