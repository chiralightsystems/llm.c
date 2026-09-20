"""Build the committed checkout's cached decoder on Linux/WSL; never run a GPU job.

Search tags: External experiments; wm:llmc; wm:decode; wm:provenance.
Preparation archives HEAD, ignoring staged, unstaged and untracked changes.
Compilation verifies that immutable snapshot and writes each attempt into a new
directory. CUDA 13.3 and SM120 remain the historical numerical/build policy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "llmc.worldmodel.cached_decode_build.v1"
DECODE_RELATIVE = Path("worldmodel/decode")
HISTORICAL_BASE_COMMIT = "247f4384b978faad621dc0255d375986e5df09b6"
PARENT_IMPORT_COMMIT = "d6d74f6a1bcf725acb858a8629e917df2fb73bc2"
DECODE_FILES = (
    "cached_decode.cu", "cached_attention.cu", "cached_attention.h", "cached_matmul.h",
    "test_cached_attention.cu", "test_decode_primitives.cu",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def file_inventory(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def verify_inventory(directory: Path, expected: dict[str, str]) -> None:
    actual = file_inventory(directory)
    if actual != expected:
        changed = sorted(key for key in expected.keys() | actual.keys()
                         if actual.get(key) != expected.get(key))
        raise RuntimeError("Prepared source changed: " + ", ".join(changed[:12]))


def validate_locations(source: Path, build: Path) -> None:
    artifacts = (ROOT / "artifacts").resolve()
    for path in (source, build):
        if path == artifacts or not path.is_relative_to(artifacts):
            raise ValueError(f"Source/build output must be below {artifacts}: {path}")
    if source == build or source.is_relative_to(build) or build.is_relative_to(source):
        raise ValueError("Source and build directories must be disjoint")


def git_identity(commit: str = "HEAD") -> tuple[str, str]:
    git = ["git", "-C", str(ROOT), "rev-parse", "--verify"]
    resolved = subprocess.check_output([*git, commit + "^{commit}"], text=True).strip()
    tree = subprocess.check_output([*git, resolved + "^{tree}"], text=True).strip()
    return resolved, tree


def prepare(args: argparse.Namespace) -> None:
    source, build = args.source_dir, args.build_dir
    if source.exists() or build.exists():
        raise FileExistsError("Refusing to overwrite an existing source/build snapshot; use --compile to verify and resume")
    commit, tree = git_identity()
    archive_command = ["git", "-C", str(ROOT), "-c", "core.autocrlf=false", "-c", "core.eol=lf",
                       "archive", "--format=tar", commit]
    archive = subprocess.check_output(archive_command)
    # Reject an incomplete committed package before creating either output.
    required = ["train_gpt2.cu", "llmc/cudnn_att.cpp", *[
        (DECODE_RELATIVE / name).as_posix() for name in DECODE_FILES]]
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        archived = {member.name for member in handle.getmembers() if member.isfile()}
    missing = sorted(set(required) - archived)
    if missing:
        raise FileNotFoundError("Current HEAD lacks committed decoder inputs: " + ", ".join(missing))
    # Every compilation input comes from the committed archive, never working files.
    source.mkdir(parents=True, exist_ok=False)
    build.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        handle.extractall(source, filter="data")
    manifest = {
        "schema": SCHEMA, "prepared_at_utc": now(), "source_commit": commit,
        "source_tree": tree, "source_repository": str(ROOT), "source": str(source),
        "build": str(build), "source_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "archive_command": archive_command, "source_files": file_inventory(source),
        "working_tree_changes_included": False,
        "historical_provenance": {"llmc_base_commit": HISTORICAL_BASE_COMMIT,
                                  "worldmodel_import_commit": PARENT_IMPORT_COMMIT},
        "builder_sha256": sha256(Path(__file__)),
        "precision": "BF16 storage and FP32 compute; stock fast-math compilation",
        "scope": "frozen cached decode and synthetic primitive tests; no execution by this builder",
    }
    write_json(build / "build_manifest.json", manifest, exclusive=True)
    print(json.dumps({"event": "prepared", "manifest": str(build / "build_manifest.json")}), flush=True)


def verify_prepared(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads((args.build_dir / "build_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema") != SCHEMA or manifest.get("source_repository") != str(ROOT)
            or manifest.get("source") != str(args.source_dir)
            or manifest.get("build") != str(args.build_dir)):
        raise RuntimeError("Prepared manifest identity does not match this build request")
    commit = manifest.get("source_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("Prepared source commit is not an immutable Git identity")
    if git_identity(commit) != (commit, manifest.get("source_tree")):
        raise RuntimeError("Prepared source commit/tree identity differs")
    verify_inventory(args.source_dir, manifest["source_files"])
    return manifest


def snapshot_decode(attempt: Path, source: Path) -> dict[str, str]:
    """Each attempt owns a decoder copy from its verified committed archive."""
    decode = source / DECODE_RELATIVE
    for name in DECODE_FILES:
        if not (decode / name).is_file():
            raise FileNotFoundError(f"Required decoder input is missing: {decode / name}")
    snapshot = attempt / "decode"
    snapshot.mkdir()
    for name in DECODE_FILES:
        with (decode / name).open("rb") as origin, (snapshot / name).open("xb") as destination:
            shutil.copyfileobj(origin, destination)
    return file_inventory(snapshot)


def library_directories(args: argparse.Namespace) -> list[Path]:
    cuda = args.nvcc.parent.parent
    candidates = [*args.library_dir, args.cudnn_root / "lib", args.cudnn_root / "lib64",
                  args.cudnn_root / "lib/x86_64-linux-gnu", cuda / "lib64",
                  Path("/usr/lib/wsl/lib"), Path("/usr/lib/x86_64-linux-gnu"), cuda / "lib64/stubs"]
    return list(dict.fromkeys(candidates))


def dependency_inventory(args: argparse.Namespace) -> dict[str, Any]:
    cuda = args.nvcc.parent.parent
    headers = [args.nvcc, args.frontend_include / "cudnn_frontend.h",
               args.cudnn_root / "include/cudnn.h", cuda / "include/nvtx3/nvToolsExt.h"]
    for path in headers:
        if not path.is_file():
            raise FileNotFoundError(f"Required compiler/development dependency is missing: {path}")
    library_dirs = library_directories(args)
    libraries = {}
    for name in ("cublas", "cublasLt", "nvidia-ml", "nvrtc", "cudnn", "cudart"):
        found = next((directory / f"lib{name}.so" for directory in library_dirs
                      if (directory / f"lib{name}.so").is_file()), None)
        if found is None:
            raise FileNotFoundError(f"Required development library lib{name}.so is missing")
        libraries[name] = {"path": str(found), "resolved_path": str(found.resolve()), "sha256": sha256(found)}
    version = subprocess.check_output([str(args.nvcc), "--version"], text=True)
    if not re.search(r"release 13\.3[,\s]", version):
        raise RuntimeError("This pinned build requires nvcc CUDA 13.3")
    return {"nvcc_version": version, "cuda_root": str(cuda), "cudnn_root": str(args.cudnn_root),
            "frontend_include": str(args.frontend_include),
            "headers": {str(path): sha256(path) for path in headers},
            "libraries": libraries, "library_dirs": [str(path) for path in library_dirs]}


def runtime_library_directories(dependencies: dict[str, Any]) -> list[str]:
    # Linker stubs must never shadow real shared libraries at execution time.
    return list(dict.fromkeys(str(Path(library[key]).parent)
                             for library in dependencies["libraries"].values()
                             for key in ("path", "resolved_path")
                             if Path(library[key]).parent.name != "stubs"))


def compile_commands(args: argparse.Namespace, attempt: Path) -> list[list[str]]:
    cuda = args.nvcc.parent.parent
    common = [str(args.nvcc), "--threads=4", "--use_fast_math", "--std=c++20", "-O3",
              "-DENABLE_BF16", "-DENABLE_CUDNN", "-DNO_MULTI_GPU", "-maxrregcount=0",
              "--generate-code=arch=compute_120,code=sm_120",
              # dev/unistd.h is a Windows shim and must not shadow Linux unistd.h.
              "-I" + str(attempt / "decode"), "-I" + str(args.source_dir),
              "-I" + str(args.frontend_include), "-I" + str(args.cudnn_root / "include")]
    stock = attempt / "cudnn_att.o"
    attention = attempt / "cached_attention.o"
    commands = [
        [*common, "-x", "cu", "-c", str(args.source_dir / "llmc/cudnn_att.cpp"), "-o", str(stock)],
        [*common, "-c", str(attempt / "decode/cached_attention.cu"), "-o", str(attention)],
    ]
    libraries = ["-L" + str(directory) for directory in library_directories(args)]
    for directory in library_directories(args):
        if directory.name != "stubs":
            libraries += ["-Xlinker=-rpath", "-Xlinker=" + str(directory)]
    libraries += ["-lcublas", "-lcublasLt", "-lnvidia-ml", "-lnvrtc", "-lcudnn", "-lcudart"]
    for name in ("cached_decode", "test_cached_attention"):
        commands.append([*common, str(attempt / "decode" / (name + ".cu")), str(stock), str(attention),
                         *libraries, "-o", str(attempt / name)])
    commands.append([*common, str(attempt / "decode/test_decode_primitives.cu"),
                     *["-L" + str(directory) for directory in library_directories(args)],
                     "-lcublas", "-lcublasLt", "-lcudart",
                     "-o", str(attempt / "test_decode_primitives")])
    return commands


def object_inputs(index: int, source: Path, attempt: Path) -> dict[str, str]:
    """Hash each object's transitive quoted includes, excluding unrelated training code."""
    roots = (("decode", attempt / "decode"), ("source", source))
    pending = [source / "llmc/cudnn_att.cpp" if index == 0 else attempt / "decode/cached_attention.cu"]
    result: dict[str, str] = {}
    while pending:
        path = pending.pop().resolve()
        owner = next(((label, root.resolve()) for label, root in roots
                      if path.is_relative_to(root.resolve())), None)
        if owner is None:
            raise RuntimeError(f"Object include escaped its source snapshot: {path}")
        key = owner[0] + "/" + path.relative_to(owner[1]).as_posix()
        if key in result:
            continue
        result[key] = sha256(path)
        for include in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', path.read_text(encoding="utf-8"), re.MULTILINE):
            target = next((directory / include for directory in (path.parent, *(root for _, root in roots))
                           if (directory / include).is_file()), None)
            if target is None:
                raise RuntimeError(f"Cannot verify quoted object include {include!r} in {path}")
            pending.append(target)
    return result


def normalized_object_command(command: list[str], source: Path, attempt: Path) -> list[str]:
    return [part.replace(str(attempt), "<attempt>").replace(str(source), "<source>") for part in command]


def reusable_objects(args: argparse.Namespace, attempt: Path, prepared: dict[str, Any],
                     dependencies: dict[str, Any], commands: list[list[str]]) -> dict[int, dict[str, Any]]:
    prior = args.reuse_objects_from
    if prior is None:
        return {}
    if not prior.is_relative_to((ROOT / "artifacts").resolve()) or prior == attempt:
        raise ValueError("Object reuse must name a prior attempt below this worktree's artifacts")
    manifest_path = prior / "compile_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    prepared_path = prior.parent / "build_manifest.json"
    previous_prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if (previous.get("schema") != SCHEMA + ".attempt"
            or previous.get("source_commit") != prepared.get("source_commit")
            or previous.get("prepared_manifest_sha256") != sha256(prepared_path)
            or previous_prepared.get("schema") != SCHEMA
            or previous_prepared.get("source_commit") != prepared.get("source_commit")
            or previous_prepared.get("source_tree") != prepared.get("source_tree")
            or previous_prepared.get("build") != str(prior.parent)):
        raise RuntimeError("Prior object preparation identity could not be verified")
    if previous.get("dependencies") != dependencies:
        raise RuntimeError("Prior object compiler/dependency fingerprints differ")
    prior_source = Path(previous_prepared["source"])
    verify_inventory(prior / "decode", previous["decode_sources"])
    result = {}
    for index, name in enumerate(("cudnn_att.o", "cached_attention.o")):
        if not any(step.get("command_index") == index and step.get("return_code") == 0
                   for step in previous.get("steps", [])):
            raise RuntimeError(f"Prior object has no successful compile step: {name}")
        if normalized_object_command(previous["commands"][index], prior_source, prior) != normalized_object_command(commands[index], args.source_dir, attempt):
            raise RuntimeError(f"Prior object compile flags differ: {name}")
        before = object_inputs(index, prior_source, prior)
        for key, digest in before.items():
            owner, relative = key.split("/", 1)
            expected = previous["decode_sources"] if owner == "decode" else previous_prepared["source_files"]
            if expected.get(relative) != digest:
                raise RuntimeError(f"Prior object input changed: {key}")
        if before != object_inputs(index, args.source_dir, attempt):
            raise RuntimeError(f"Prior object source/include fingerprints differ: {name}")
        artifact = previous.get("artifacts", {}).get(name)
        path = prior / name
        if not artifact or artifact.get("path") != str(path) or sha256(path) != artifact.get("sha256"):
            raise RuntimeError(f"Prior object artifact hash differs: {name}")
        result[index] = {"path": path, "sha256": artifact["sha256"], "object_inputs": before,
                         "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path)}
    return result


def build(args: argparse.Namespace) -> None:
    prepared = verify_prepared(args)
    dependencies = dependency_inventory(args)
    index = 1
    while True:
        attempt = args.build_dir / f"attempt_{index:03d}"
        try:
            attempt.mkdir()
            break
        except FileExistsError:
            index += 1
    temporary = attempt / "temp"
    temporary.mkdir()
    environment = dict(os.environ)
    environment.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
    decode_sources = snapshot_decode(attempt, args.source_dir)
    commands = compile_commands(args, attempt)
    report = {"schema": SCHEMA + ".attempt", "started_at_utc": now(), "status": "building",
              "prepared_manifest_sha256": sha256(args.build_dir / "build_manifest.json"),
              "source_commit": prepared["source_commit"], "source_tree": prepared["source_tree"],
              "dependencies": dependencies, "runtime_library_dirs": runtime_library_directories(dependencies),
              "builder_sha256": sha256(Path(__file__)), "decode_sources": decode_sources,
              "commands": commands, "steps": [], "artifacts": {},
              "environment": {key: environment.get(key) for key in ("PATH", "LD_LIBRARY_PATH", "TMPDIR")}}
    report_path = attempt / "compile_manifest.json"
    write_json(report_path, report, exclusive=True)
    try:
        reused = reusable_objects(args, attempt, prepared, dependencies, commands)
        for index, command in enumerate(commands):
            log = attempt / f"compile_{index:02d}.log"
            reuse = reused.get(index)
            print(json.dumps({"event": "reuse_object" if reuse else "compile", "output": command[-1], "log": str(log)}), flush=True)
            start = time.perf_counter()
            with log.open("x", encoding="utf-8") as handle:
                if reuse:
                    with reuse["path"].open("rb") as origin, Path(command[-1]).open("xb") as destination:
                        shutil.copyfileobj(origin, destination)
                    if sha256(Path(command[-1])) != reuse["sha256"]:
                        raise RuntimeError("Object changed while copying")
                    handle.write("Verified object reuse: " + str(reuse["path"]) + "\n")
                    return_code = 0
                else:
                    return_code = subprocess.run(command, cwd=attempt, env=environment,
                                                 stdout=handle, stderr=subprocess.STDOUT).returncode
            step = {"command_index": index, "return_code": return_code,
                    "elapsed_seconds": time.perf_counter() - start, "log_sha256": sha256(log)}
            if index < 2:
                step["object_inputs"] = object_inputs(index, args.source_dir, attempt)
            if reuse:
                step["reused_from"] = {key: str(value) if isinstance(value, Path) else value
                                       for key, value in reuse.items() if key != "object_inputs"}
            report["steps"].append(step)
            if return_code:
                raise RuntimeError(f"Compiler failed; preserved log: {log}")
            output = Path(command[-1])
            report["artifacts"][output.name] = {"path": str(output), "sha256": sha256(output), "bytes": output.stat().st_size}
            write_json(report_path, report)
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = str(error) or type(error).__name__
        raise
    finally:
        report["completed_at_utc"] = now()
        write_json(report_path, report)
    print(json.dumps({"event": "built", "manifest": str(report_path), "artifacts": report["artifacts"],
                      "gpu_executions": 0}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Exclusively archive committed HEAD; working changes are excluded")
    parser.add_argument("--compile", action="store_true", help="Verify a prepared snapshot and compile in a fresh attempt directory")
    parser.add_argument("--reuse-objects-from", type=Path,
                        help="Explicitly reuse both attention objects after checking a prior attempt's hashes and inputs")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "artifacts/llmc_cached_decode_src")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "artifacts/llmc_cached_decode_build")
    parser.add_argument("--nvcc", type=Path, default=Path("/usr/local/cuda/bin/nvcc"),
                        help="CUDA 13.3 compiler (default: /usr/local/cuda/bin/nvcc)")
    parser.add_argument("--cudnn-root", type=Path, default=Path("/usr"),
                        help="cuDNN development prefix containing include/cudnn.h (default: /usr)")
    parser.add_argument("--frontend-include", type=Path, default=Path("/usr/local/include"),
                        help="Directory containing cudnn_frontend.h (default: /usr/local/include)")
    parser.add_argument("--library-dir", type=Path, action="append", default=[],
                        help="Additional library search directory, searched first; repeat as needed")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.prepare and not args.compile:
        parser.error("Select --prepare and/or --compile")
    if args.reuse_objects_from is not None and not args.compile:
        parser.error("--reuse-objects-from requires --compile")
    if os.name != "posix" or platform.system() != "Linux":
        parser.error("This build helper requires Linux or WSL; it does not build or execute Windows binaries")
    for name in ("source_dir", "build_dir", "nvcc", "cudnn_root", "frontend_include"):
        setattr(args, name, getattr(args, name).resolve())
    args.library_dir = [path.resolve() for path in args.library_dir]
    validate_locations(args.source_dir, args.build_dir)
    if args.reuse_objects_from is not None:
        args.reuse_objects_from = args.reuse_objects_from.resolve(strict=True)
    if args.prepare:
        prepare(args)
    if args.compile:
        build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
