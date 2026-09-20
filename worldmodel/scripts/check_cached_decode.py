"""Qualify cached llm.c primitives and whole-stack memory safety (RTX 5070).

Search tags: External experiments; wm:llmc; wm:decode; wm:throughput.
Full-prefix numerical qualification is run separately with run_cached_decode.py.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from run_cached_decode import (
    GPU_NAME, ROOT, digest, git, resolve_gpu, runtime_environment, terminate_group, write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-attempt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True, help="Explicit nvidia-smi GPU index or UUID; must identify an RTX 5070")
    parser.add_argument("--memcheck", action="store_true")
    parser.add_argument("--sanitizer", type=Path, help="Compute Sanitizer executable; otherwise use the build manifest CUDA toolkit")
    return parser


def sanitizer_path(manifest: dict, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.resolve(strict=True)
    else:
        cuda_root = manifest.get("dependencies", {}).get("cuda_root")
        if not isinstance(cuda_root, str) or not Path(cuda_root).is_absolute():
            raise ValueError("Memcheck requires manifest dependencies.cuda_root or explicit --sanitizer")
        path = Path(cuda_root) / "bin" / "compute-sanitizer"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def execute_check(command: list[str], *, env: dict, log) -> subprocess.CompletedProcess:
    process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return subprocess.CompletedProcess(command, process.wait(timeout=1200))
    finally:
        terminate_group(process)


def main() -> None:
    args = build_parser().parse_args()
    if os.name != "posix":
        raise RuntimeError("Cached decode checks require Linux or WSL with CUDA")
    build = args.build_attempt.resolve(strict=True)
    binaries = {name: build / name for name in
                ["test_decode_primitives", "test_cached_attention", "cached_decode"]}
    for path in binaries.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_path = build / "compile_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "passed":
        raise RuntimeError("Checks require a successful build attempt")
    for name, path in binaries.items():
        if manifest.get("artifacts", {}).get(name, {}).get("sha256") != digest(path):
            raise RuntimeError(f"Binary does not match compile manifest: {name}")
    sanitizer = sanitizer_path(manifest, args.sanitizer) if args.memcheck else None
    gpu_uuid = resolve_gpu(args.gpu)
    env = runtime_environment(manifest, gpu_uuid)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "compile_manifest.json").write_bytes(manifest_path.read_bytes())
    jobs = [(name, [str(binaries[name])]) for name in ["test_decode_primitives", "test_cached_attention"]]
    if args.memcheck:
        for batch in [1, 8]:
            jobs.append((f"stack_b{batch}", [str(binaries["cached_decode"]), "--batch", str(batch),
                        "--prefix", "4", "--steps", "4", "--output", str(output / f"stack_b{batch}.json")]))
    report = {"schema": "worldmodel.llmc_cached_decode_checks.v1", "commit": git("rev-parse", "HEAD"),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "gpu_selector": args.gpu, "gpu_uuid": gpu_uuid, "gpu_name": GPU_NAME,
              "memcheck": args.memcheck, "passed": False,
              "per_process_timeout_seconds": 1200,
              "environment": {key: env[key] for key in
                              ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "LD_LIBRARY_PATH")},
              "compile_manifest_sha256": digest(manifest_path), "harness_sha256": digest(Path(__file__)),
              "binaries": {name: {"path": str(path), "sha256": digest(path)} for name, path in binaries.items()},
              "runs": []}
    for name, command in jobs:
        if args.memcheck:
            command = [str(sanitizer), "--tool", "memcheck", "--error-exitcode", "97", *command]
        print(f"Checking {name}, memcheck={args.memcheck}", flush=True)
        log = output / (name + ".log")
        start = time.perf_counter()
        try:
            with log.open("x") as stream:
                result = execute_check(command, env=env, log=stream)
        except BaseException as error:
            report["runs"].append({"name": name, "command": command, "error": repr(error),
                                   "wall_seconds": time.perf_counter() - start, "log": str(log),
                                   "log_sha256": digest(log) if log.is_file() else None, "passed": False})
            write_json(output / "results.json", report)
            raise
        contents = log.read_text()
        passed = result.returncode == 0 and (not args.memcheck or "ERROR SUMMARY: 0 errors" in contents)
        report["runs"].append({"name": name, "command": command, "returncode": result.returncode,
                               "wall_seconds": time.perf_counter() - start, "log": str(log),
                               "log_sha256": digest(log), "passed": passed})
        write_json(output / "results.json", report)
        if not passed:
            raise RuntimeError(f"{name} failed; see {log}")
    report["passed"] = True
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "results.json", report)
    print("All checks passed", flush=True)


if __name__ == "__main__":
    main()
