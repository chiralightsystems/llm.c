from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "worldmodel/scripts/build_cached_decode.py"
SPEC = importlib.util.spec_from_file_location("llmc_cached_decode_builder_test", SCRIPT)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


@contextmanager
def workspace():
    # Inherit the workspace ACL; Python 3.14 private temp dirs are inaccessible
    # to the unelevated Windows sandbox used by other repository CPU tests.
    path = ROOT / "artifacts" / ("cached_decode_builder_test_" + uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        if path.resolve().parent != (ROOT / "artifacts").resolve():
            raise RuntimeError("Test cleanup escaped the artifacts directory")
        # Git fixture objects are read-only on Windows; clear that attribute
        # only within the just-validated, test-owned directory before removal.
        for file in path.rglob("*"):
            if file.is_file() and not file.stat().st_mode & stat.S_IWRITE:
                file.chmod(file.stat().st_mode | stat.S_IWRITE)
        shutil.rmtree(path)


def fixture_git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "-c", "user.name=Cached Decode Test", "-c",
         "user.email=cached-decode-test@example.invalid", "-c", "commit.gpgsign=false",
         "-c", "core.autocrlf=false", *args], text=True, stderr=subprocess.STDOUT).strip()


def fixture_repository(root: Path, *, include_decode: bool = True) -> None:
    root.mkdir()
    fixture_git(root, "init", "--quiet")
    (root / ".gitignore").write_text("artifacts/\n")
    (root / "train_gpt2.cu").write_bytes(b"// committed trainer\n")
    (root / "llmc").mkdir()
    (root / "llmc/cudnn_att.cpp").write_bytes(b"// committed attention\n")
    if include_decode:
        decode = root / BUILDER.DECODE_RELATIVE
        decode.mkdir(parents=True)
        for name in BUILDER.DECODE_FILES:
            (decode / name).write_bytes(("// committed " + name + "\n").encode())
    fixture_git(root, "add", ".")
    fixture_git(root, "commit", "--quiet", "-m", "Fixture source")


class CachedDecodeBuildTests(unittest.TestCase):
    def setUp(self):
        silence = mock.patch.object(BUILDER, "print", create=True)
        silence.start()
        self.addCleanup(silence.stop)

    def test_commands_use_attempt_sources_and_stock_attention(self):
        args = BUILDER.build_parser().parse_args([])
        attempt = args.build_dir / "attempt_001"
        commands = BUILDER.compile_commands(args, attempt)
        self.assertEqual(len(commands), 5)
        self.assertEqual([Path(command[-1]).name for command in commands],
                         ["cudnn_att.o", "cached_attention.o", "cached_decode",
                          "test_cached_attention", "test_decode_primitives"])
        for command in commands:
            for flag in ("--std=c++20", "--use_fast_math", "-DENABLE_BF16", "-DENABLE_CUDNN",
                         "-DNO_MULTI_GPU", "--generate-code=arch=compute_120,code=sm_120"):
                self.assertIn(flag, command)
            self.assertNotIn(str(ROOT / BUILDER.DECODE_RELATIVE / "cached_decode.cu"), command)
            self.assertNotIn("-I" + str(args.source_dir / "dev"), command)
        self.assertIn(str(args.source_dir / "llmc/cudnn_att.cpp"), commands[0])
        for command in commands[2:4]:
            self.assertIn(str(attempt / "cudnn_att.o"), command)
            self.assertIn(str(attempt / "cached_attention.o"), command)
            for library in ("-lcublas", "-lcublasLt", "-lnvidia-ml", "-lnvrtc", "-lcudnn"):
                self.assertIn(library, command)
        self.assertNotIn(str(attempt / "cudnn_att.o"), commands[4])
        self.assertIn(str(attempt / "decode/test_decode_primitives.cu"), commands[4])

    def test_prepared_inventory_rejects_changes_and_extra_files(self):
        with workspace() as source:
            file = source / "train_gpt2.cu"
            file.write_bytes(b"pinned\n")
            original = BUILDER.file_inventory(source)
            BUILDER.verify_inventory(source, original)
            file.write_bytes(b"changed\n")
            with self.assertRaisesRegex(RuntimeError, "Prepared source changed"):
                BUILDER.verify_inventory(source, original)
            file.write_bytes(b"pinned\n")
            (source / "unexpected.cuh").write_bytes(b"extra\n")
            with self.assertRaisesRegex(RuntimeError, "unexpected.cuh"):
                BUILDER.verify_inventory(source, original)

    def test_fresh_attempts_snapshot_prepared_decoder_without_overwrite(self):
        with workspace() as root:
            prepared = root / "prepared"
            development = prepared / BUILDER.DECODE_RELATIVE
            development.mkdir(parents=True)
            for name in BUILDER.DECODE_FILES:
                (development / name).write_bytes(name.encode())
            first, second = root / "attempt_001", root / "attempt_002"
            first.mkdir()
            second.mkdir()
            before = BUILDER.snapshot_decode(first, prepared)
            after = BUILDER.snapshot_decode(second, prepared)
            with self.assertRaises(FileExistsError):
                BUILDER.snapshot_decode(first, prepared)
            self.assertEqual(before, after)
            self.assertEqual(BUILDER.file_inventory(first / "decode"), before)
            self.assertEqual(BUILDER.file_inventory(second / "decode"), after)

    def test_outputs_are_disjoint_and_preparation_is_exclusive(self):
        with workspace() as root:
            source, build = root / "source", root / "build"
            BUILDER.validate_locations(source, build)
            for candidate_source, candidate_build in ((source, source), (source, source / "build"),
                                                        (ROOT / "worldmodel", build), (ROOT / "artifacts", build)):
                with self.assertRaises(ValueError):
                    BUILDER.validate_locations(candidate_source, candidate_build)
            source.mkdir()
            (source / "retained").write_bytes(b"keep")
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                BUILDER.prepare(SimpleNamespace(source_dir=source, build_dir=build))
            self.assertEqual((source / "retained").read_bytes(), b"keep")

    def test_object_reuse_verifies_inputs_flags_dependencies_and_binary(self):
        with workspace() as root:
            prior_source, current_source = root / "prior_source", root / "current_source"
            prior, current = root / "prior_build/attempt_002", root / "current_build/attempt_001"
            for source in (prior_source, current_source):
                (source / "llmc").mkdir(parents=True)
                (source / "llmc/cudnn_att.cpp").write_text('#include "cudnn_att.h"\n')
                (source / "llmc/cudnn_att.h").write_text('#include "cuda_common.h"\n')
                (source / "llmc/cuda_common.h").write_text("// common\n")
                (source / "llmc/adamw.cuh").write_text(str(source))
            for attempt in (prior, current):
                (attempt / "decode").mkdir(parents=True)
                (attempt / "decode/cached_attention.cu").write_text('#include "cached_attention.h"\n')
                (attempt / "decode/cached_attention.h").write_text("// cached header\n")
            previous_args = BUILDER.build_parser().parse_args([])
            previous_args.source_dir = prior_source
            args = BUILDER.build_parser().parse_args([])
            args.source_dir = current_source
            args.reuse_objects_from = prior
            commands = BUILDER.compile_commands(args, current)
            prepared = {"schema": BUILDER.SCHEMA, "source_commit": "1" * 40, "source_tree": "2" * 40,
                        "source": str(prior_source),
                        "build": str(prior.parent), "source_files": BUILDER.file_inventory(prior_source)}
            BUILDER.write_json(prior.parent / "build_manifest.json", prepared)
            artifacts = {}
            for name in ("cudnn_att.o", "cached_attention.o"):
                path = prior / name
                path.write_bytes(name.encode())
                artifacts[name] = {"path": str(path), "sha256": BUILDER.sha256(path)}
            dependencies = {"compiler": "fixed", "library": "fixed"}
            manifest = {"schema": BUILDER.SCHEMA + ".attempt", "source_commit": prepared["source_commit"],
                        "prepared_manifest_sha256": BUILDER.sha256(prior.parent / "build_manifest.json"),
                        "dependencies": dependencies, "decode_sources": BUILDER.file_inventory(prior / "decode"),
                        "commands": BUILDER.compile_commands(previous_args, prior),
                        "steps": [{"command_index": i, "return_code": 0} for i in range(2)],
                        "artifacts": artifacts}
            BUILDER.write_json(prior / "compile_manifest.json", manifest)
            reuse = lambda: BUILDER.reusable_objects(args, current, prepared, dependencies, commands)
            self.assertEqual(set(reuse()), {0, 1})
            common = current_source / "llmc/cuda_common.h"
            common.write_text("// changed transitive include\n")
            with self.assertRaisesRegex(RuntimeError, "source/include fingerprints differ"):
                reuse()
            common.write_text("// common\n")
            commands[0].append("--changed-flag")
            with self.assertRaisesRegex(RuntimeError, "compile flags differ"):
                reuse()
            commands[0].pop()
            with self.assertRaisesRegex(RuntimeError, "dependency fingerprints differ"):
                BUILDER.reusable_objects(args, current, prepared, {"compiler": "changed"}, commands)
            with self.assertRaisesRegex(RuntimeError, "preparation identity"):
                BUILDER.reusable_objects(args, current, {**prepared, "source_commit": "3" * 40}, dependencies, commands)
            (prior / "cached_attention.o").write_bytes(b"changed object")
            with self.assertRaisesRegex(RuntimeError, "artifact hash differs"):
                reuse()

    def test_prepare_archives_commit_not_working_changes_and_survives_head_change(self):
        with workspace() as root:
            repository = root / "repository"
            fixture_repository(repository)
            original_commit = fixture_git(repository, "rev-parse", "HEAD")
            (repository / "train_gpt2.cu").write_bytes(b"// unstaged trainer\n")
            (repository / "untracked.txt").write_bytes(b"not archived\n")
            decoder = repository / BUILDER.DECODE_RELATIVE / "cached_decode.cu"
            decoder.write_bytes(b"// staged decoder\n")
            fixture_git(repository, "add", str(BUILDER.DECODE_RELATIVE / "cached_decode.cu"))
            with mock.patch.object(BUILDER, "ROOT", repository):
                args = BUILDER.build_parser().parse_args([])
                BUILDER.prepare(args)
                manifest = BUILDER.verify_prepared(args)
                self.assertEqual(manifest["source_commit"], original_commit)
                self.assertEqual((args.source_dir / "train_gpt2.cu").read_bytes(), b"// committed trainer\n")
                self.assertEqual((args.source_dir / BUILDER.DECODE_RELATIVE / "cached_decode.cu").read_bytes(),
                                 b"// committed cached_decode.cu\n")
                self.assertFalse((args.source_dir / "untracked.txt").exists())
                self.assertFalse(manifest["working_tree_changes_included"])
                fixture_git(repository, "commit", "--quiet", "-m", "Later fixture commit")
                self.assertNotEqual(fixture_git(repository, "rev-parse", "HEAD"), original_commit)
                self.assertEqual(BUILDER.verify_prepared(args), manifest)
                attempt = args.build_dir / "attempt_001"
                attempt.mkdir()
                BUILDER.snapshot_decode(attempt, args.source_dir)
                self.assertEqual((attempt / "decode/cached_decode.cu").read_bytes(),
                                 b"// committed cached_decode.cu\n")
                with self.assertRaises(FileExistsError):
                    BUILDER.prepare(args)

    def test_prepare_requires_committed_decoder_before_creating_outputs(self):
        with workspace() as root:
            repository = root / "repository"
            fixture_repository(repository, include_decode=False)
            with mock.patch.object(BUILDER, "ROOT", repository):
                args = BUILDER.build_parser().parse_args([])
                with self.assertRaisesRegex(FileNotFoundError, "committed decoder inputs"):
                    BUILDER.prepare(args)
                self.assertFalse(args.source_dir.exists())
                self.assertFalse(args.build_dir.exists())

    def test_build_records_archived_source_runtime_paths_and_fresh_attempts(self):
        with workspace() as root:
            repository = root / "repository"
            fixture_repository(repository)
            library = root / "libcudart.so"
            library.write_bytes(b"mock library")
            dependencies = {"cuda_root": str(root), "libraries": {
                "cudart": {"path": str(library), "resolved_path": str(library)}}}
            real_run = subprocess.run

            def compile_fixture(command, **kwargs):
                if command[0] == "git":
                    return real_run(command, **kwargs)
                Path(command[-1]).write_bytes(b"mock compiler output")
                return SimpleNamespace(returncode=0)

            with mock.patch.object(BUILDER, "ROOT", repository):
                args = BUILDER.build_parser().parse_args([])
                BUILDER.prepare(args)
                (repository / BUILDER.DECODE_RELATIVE / "cached_decode.cu").write_bytes(b"working edit")
                with mock.patch.object(BUILDER, "dependency_inventory", return_value=dependencies), \
                     mock.patch.object(BUILDER.subprocess, "run", side_effect=compile_fixture) as compiler:
                    BUILDER.build(args)
                    BUILDER.build(args)
                    self.assertEqual(sum(call.args[0][0] != "git" for call in compiler.call_args_list), 10)
                for index in (1, 2):
                    attempt = args.build_dir / f"attempt_{index:03d}"
                    report = json.loads((attempt / "compile_manifest.json").read_text())
                    self.assertEqual(report["schema"], BUILDER.SCHEMA + ".attempt")
                    self.assertEqual(report["source_commit"], fixture_git(repository, "rev-parse", "HEAD"))
                    self.assertEqual(report["status"], "passed")
                    self.assertEqual(report["runtime_library_dirs"], [str(root)])
                    self.assertEqual((attempt / "decode/cached_decode.cu").read_bytes(),
                                     b"// committed cached_decode.cu\n")
                    self.assertEqual(report["artifacts"]["cached_decode"]["sha256"],
                                     BUILDER.sha256(attempt / "cached_decode"))

    def test_prepared_manifest_and_archived_decoder_tampering_are_rejected(self):
        with workspace() as root:
            repository = root / "repository"
            fixture_repository(repository)
            with mock.patch.object(BUILDER, "ROOT", repository):
                args = BUILDER.build_parser().parse_args([])
                BUILDER.prepare(args)
                manifest_path = args.build_dir / "build_manifest.json"
                original = json.loads(manifest_path.read_text())
                for key, value in (("schema", "old-schema"), ("source_commit", "HEAD"),
                                   ("source_tree", "0" * 40), ("source_repository", "another-repository")):
                    with self.subTest(key=key):
                        BUILDER.write_json(manifest_path, {**original, key: value})
                        with self.assertRaises(RuntimeError):
                            BUILDER.verify_prepared(args)
                BUILDER.write_json(manifest_path, original)
                (args.source_dir / BUILDER.DECODE_RELATIVE / "cached_decode.cu").write_bytes(b"tampered")
                with self.assertRaisesRegex(RuntimeError, "Prepared source changed"):
                    BUILDER.verify_prepared(args)

    def test_missing_dependencies_fail_before_invoking_compiler(self):
        with workspace() as root:
            args = BUILDER.build_parser().parse_args([
                "--nvcc", str(root / "cuda/bin/nvcc"), "--cudnn-root", str(root / "cudnn"),
                "--frontend-include", str(root / "frontend")])
            with mock.patch.object(BUILDER.subprocess, "check_output") as launch:
                with self.assertRaisesRegex(FileNotFoundError, "compiler/development dependency"):
                    BUILDER.dependency_inventory(args)
                launch.assert_not_called()
                for path in (args.nvcc, args.nvcc.parent.parent / "include/nvtx3/nvToolsExt.h",
                             args.cudnn_root / "include/cudnn.h", args.frontend_include / "cudnn_frontend.h"):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture dependency")
                with mock.patch.object(BUILDER, "library_directories", return_value=[root / "missing-libraries"]):
                    with self.assertRaisesRegex(FileNotFoundError, "libcublas.so"):
                        BUILDER.dependency_inventory(args)
                launch.assert_not_called()

    def test_explicit_dependency_locations_record_fingerprints_and_runtime_paths(self):
        with workspace() as root:
            args = BUILDER.build_parser().parse_args([
                "--nvcc", str(root / "cuda/bin/nvcc"), "--cudnn-root", str(root / "cudnn"),
                "--frontend-include", str(root / "frontend"), "--library-dir", str(root / "custom-lib")])
            for path in (args.nvcc, args.nvcc.parent.parent / "include/nvtx3/nvToolsExt.h",
                         args.cudnn_root / "include/cudnn.h", args.frontend_include / "cudnn_frontend.h"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture dependency")
            libraries = args.library_dir[0]
            libraries.mkdir()
            for name in ("cublas", "cublasLt", "nvidia-ml", "nvrtc", "cudnn", "cudart"):
                (libraries / ("lib" + name + ".so")).write_bytes(name.encode())
            with mock.patch.object(BUILDER.subprocess, "check_output", return_value="Cuda compilation tools, release 13.3, V13.3"):
                inventory = BUILDER.dependency_inventory(args)
            self.assertEqual(inventory["cuda_root"], str(args.nvcc.parent.parent))
            self.assertEqual(BUILDER.runtime_library_directories(inventory), [str(libraries)])
            self.assertEqual(inventory["libraries"]["cudnn"]["sha256"], BUILDER.sha256(libraries / "libcudnn.so"))
            commands = BUILDER.compile_commands(args, root / "attempt")
            self.assertEqual(commands[0][0], str(args.nvcc))
            self.assertIn("-I" + str(args.frontend_include), commands[0])
            self.assertIn("-I" + str(args.cudnn_root / "include"), commands[0])
            self.assertIn("-L" + str(libraries), commands[2])
            with mock.patch.object(BUILDER.subprocess, "check_output", return_value="release 13.2, V13.2"):
                with self.assertRaisesRegex(RuntimeError, "requires nvcc CUDA 13.3"):
                    BUILDER.dependency_inventory(args)

    def test_runtime_library_paths_exclude_linker_stubs(self):
        dependencies = {"libraries": {
            "stub": {"path": "/cuda/lib64/stubs/libnvidia-ml.so", "resolved_path": "/cuda/lib64/stubs/libnvidia-ml.so"},
            "runtime": {"path": "/cuda/lib64/libcudart.so", "resolved_path": "/cuda/lib64/libcudart.so.13"}}}
        self.assertEqual(BUILDER.runtime_library_directories(dependencies), [str(Path("/cuda/lib64"))])


if __name__ == "__main__":
    unittest.main()
