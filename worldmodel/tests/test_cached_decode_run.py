from __future__ import annotations

import importlib.util
import contextlib
import io
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "llmc_cached_decode_run_test", ROOT / "scripts/run_cached_decode.py")
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
CHECK_SPEC = importlib.util.spec_from_file_location(
    "llmc_cached_decode_check_test", ROOT / "scripts/check_cached_decode.py")
assert CHECK_SPEC and CHECK_SPEC.loader
CHECKER = importlib.util.module_from_spec(CHECK_SPEC)
with mock.patch.dict(sys.modules, {"run_cached_decode": RUNNER}):
    CHECK_SPEC.loader.exec_module(CHECKER)


@contextlib.contextmanager
def temporary_directory():
    scratch = ROOT.parent / "artifacts"
    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / ("cached_decode_test_" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != scratch.resolve():
            raise RuntimeError("Refusing cleanup outside the test artifact directory")
        shutil.rmtree(path)


class CachedDecodeRunTests(unittest.TestCase):
    def args(self, *options):
        return RUNNER.build_parser().parse_args([
            "--binary", "cached_decode", "--build-manifest", "compile_manifest.json",
            "--output-dir", "fresh", "--gpu", "GPU-test", *options])

    def test_defaults_preserve_prior_workload(self):
        args = self.args()
        RUNNER.validate_args(args)
        self.assertEqual((args.prefix, args.capacity, args.batches, args.repetitions),
                         (1024, 2048, [1, 8], 3))
        self.assertEqual(args.model_profile, "medium-rope-e4096")
        self.assertEqual(RUNNER.parameter_bytes(args.model_profile), 1033490432)

    def test_xl_nope_geometry_memory_and_command(self):
        args = self.args("--model-profile", "xl-nope-e8192", "--prefix", "2048", "--capacity", "2176")
        RUNNER.validate_args(args)
        profile = RUNNER.PROFILES[args.model_profile]
        self.assertEqual((profile["layers"], profile["body_width"], profile["heads"], profile["lexical_width"]),
                         (48, 1600, 25, 8192))
        self.assertEqual(profile["position_policy"], "none")
        self.assertEqual(RUNNER.parameter_bytes(args.model_profile), 3827732736)
        self.assertEqual([RUNNER.kv_cache_bytes(args.model_profile, 8, capacity)
                          for capacity in (1152, 2176, 4224)],
                         [2831155200, 5347737600, 10380902400])
        self.assertEqual(RUNNER.decode_command(args, Path("driver"), 8, Path("b8.json")),
                         ["driver", "--batch", "8", "--output", "b8.json", "--capacity", "2176",
                          "--model-profile", "xl-nope-e8192", "--prefix", "2048", "--steps", "128"])

    def test_xl_qualification_has_explicit_smaller_reference_extent(self):
        args = self.args("--model-profile", "xl-nope-e8192", "--qualify", "--capacity", "1152")
        RUNNER.validate_args(args)
        self.assertEqual(RUNNER.PROFILES[args.model_profile]["reference_prefix_lengths"], [8, 32, 128])
        args.capacity = 127
        with self.assertRaisesRegex(ValueError, "at least 128"):
            RUNNER.validate_args(args)

    def test_result_identity_rejects_wrong_shape_position_or_parameter_count(self):
        args = self.args("--model-profile", "xl-nope-e8192")
        metrics = {**RUNNER.PROFILES[args.model_profile], "model_profile": args.model_profile,
                   "head_dim": 64, "reference_max_prefix": 128, "parameter_bytes": 3827732736}
        self.assertTrue(RUNNER.profile_matches(metrics, args))
        for key, value in [("model_profile", "medium-rope-e4096"), ("body_width", 1024),
                           ("lexical_width", 4096), ("position_policy", "rope"),
                           ("parameter_bytes", 1033490432), ("reference_max_prefix", 1152)]:
            with self.subTest(key=key):
                self.assertFalse(RUNNER.profile_matches({**metrics, key: value}, args))

    def test_long_prefix_includes_every_generated_commit(self):
        for prefix, capacity in [(4096, 4224), (8192, 8320)]:
            with self.subTest(prefix=prefix):
                args = self.args("--prefix", str(prefix), "--capacity", str(capacity), "--batches", "8")
                RUNNER.validate_args(args)
                command = RUNNER.decode_command(args, Path("driver"), 8, Path("b8.json"))
                self.assertEqual(command, ["driver", "--batch", "8", "--output", "b8.json",
                                          "--capacity", str(capacity), "--prefix", str(prefix), "--steps", "128"])
                args.capacity -= 1
                with self.assertRaisesRegex(ValueError, "prefix \\+ 128"):
                    RUNNER.validate_args(args)

    def test_duplicate_batches_and_invalid_integer_geometry_rejected(self):
        for options in [("--batches", "1", "1"), ("--repetitions", "0"),
                        ("--prefix", "0"), ("--capacity", "19"),
                        ("--capacity", str(2**31)), ("--qualify", "--capacity", "1151")]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                RUNNER.validate_args(self.args(*options))

    def test_qualification_retains_fixed_reference_lengths_at_larger_capacity(self):
        args = self.args("--qualify", "--capacity", "8320", "--prefix", "8192")
        RUNNER.validate_args(args)
        command = RUNNER.decode_command(args, Path("driver"), 1, Path("check.json"))
        self.assertEqual(command, ["driver", "--batch", "1", "--output", "check.json",
                                  "--capacity", "8320", "--qualify"])

    def test_result_rejects_stale_capacity_short_or_partial_batch_work(self):
        args = self.args("--prefix", "8192", "--capacity", "8320", "--batches", "8")
        metrics = {"capacity": 8320, "memory_admission": {"admitted": True},
                   "prefix_tokens_per_request": 8192, "generated_tokens_per_request": 128,
                   "total_generated_tokens": 1024,
                   "timer": "clock_monotonic_raw_graph_submit_and_completion_no_events"}
        self.assertTrue(RUNNER.workload_matches(metrics, args, 8))
        for key, value in [("capacity", 2048), ("prefix_tokens_per_request", 1024),
                           ("generated_tokens_per_request", 127), ("total_generated_tokens", 128),
                           ("timer", "steady_clock"), ("memory_admission", {"admitted": False})]:
            with self.subTest(key=key):
                self.assertFalse(RUNNER.workload_matches({**metrics, key: value}, args, 8))

    def run_guard(self, samples, *, elapsed=0, clock_values=None, model_profile="medium-rope-e4096", capacity=8320):
        child = mock.Mock(pid=4321, returncode=0)
        child.poll.side_effect = [None, 0]
        child.wait.return_value = 0
        with mock.patch.object(RUNNER, "memory_state", side_effect=samples), \
             mock.patch.object(RUNNER.subprocess, "Popen", return_value=child) as launch, \
             mock.patch.object(RUNNER, "terminate_group") as cleanup, \
             mock.patch.object(RUNNER, "write_json") as write, \
             mock.patch.object(RUNNER.time, "sleep"), \
             mock.patch.object(RUNNER.time, "monotonic", side_effect=clock_values or [0, elapsed]):
            error = None
            result = None
            try:
                result = RUNNER.guarded_run(["driver"], env={}, log=mock.sentinel.log,
                                           memory_path=Path("memory.json"), batch=8, capacity=capacity,
                                           gpu_uuid="GPU-test", model_profile=model_profile)
            except RuntimeError as caught:
                error = caught
            return result, error, launch, cleanup, write.call_args.args[1]

    def state(self, used):
        return {"used_mib": used, "total_mib": 12227, "utilization_percent": 0}

    def test_monitor_records_peak_and_owns_session(self):
        result, error, launch, cleanup, record = self.run_guard([
            self.state(100), self.state(9000), self.state(100)])
        self.assertIsNone(error)
        self.assertEqual(result.returncode, 0)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        cleanup.assert_called_once()
        self.assertEqual(record["sampled_peak_used_mib"], 9000)
        self.assertIsNone(record["stop_reason"])

    def test_memory_threshold_stops_owned_process_and_preserves_samples(self):
        result, error, launch, cleanup, record = self.run_guard([
            self.state(100), self.state(12227 - 512), self.state(100)])
        self.assertIsNone(result)
        self.assertRegex(str(error), "VRAM reserve reached")
        cleanup.assert_called_once()
        self.assertEqual(record["sampled_peak_used_mib"], 11715)
        self.assertEqual(len(record["samples"]), 1)

    def test_busy_preflight_prevents_launch(self):
        _, error, launch, cleanup, record = self.run_guard([self.state(257), self.state(257)])
        self.assertRegex(str(error), "preflight rejected")
        launch.assert_not_called()
        cleanup.assert_not_called()
        self.assertEqual(record["samples"], [])

    def test_xl_b8_4k_is_retained_as_not_fit_without_launch(self):
        _, error, launch, cleanup, record = self.run_guard([self.state(0), self.state(0)],
            model_profile="xl-nope-e8192", capacity=4224)
        self.assertRegex(str(error), "do not fit")
        launch.assert_not_called()
        cleanup.assert_not_called()
        self.assertEqual(record["admission_status"], "not_fit")
        self.assertEqual(record["minimum_model_kv_and_reserve_bytes"], 15282376960)

    def test_recent_utilization_is_only_diagnostic_after_bounded_settle(self):
        recent = {**self.state(100), "utilization_percent": 20}
        result, error, launch, _, record = self.run_guard([
            recent, self.state(9000), self.state(100)], clock_values=[0, 11, 11, 11])
        self.assertIsNone(error)
        self.assertEqual(result.returncode, 0)
        launch.assert_called_once()
        self.assertFalse(record["utilization_settled_to_zero"])

    def test_final_monitor_failure_stops_suite_after_cleanup(self):
        _, error, _, cleanup, record = self.run_guard([
            self.state(100), self.state(9000), RuntimeError("query failed")])
        self.assertRegex(str(error), "Final memory query failed")
        cleanup.assert_called_once()
        self.assertIn("query failed", record["after_query_error"])

    def test_monitor_failure_and_watchdog_cleanup(self):
        for middle, elapsed, expected in [(RuntimeError("query failed"), 0, "query failed"),
                                          (self.state(9000), 1201, "watchdog")]:
            with self.subTest(expected=expected):
                _, error, _, cleanup, record = self.run_guard([
                    self.state(100), middle, self.state(100)], elapsed=elapsed)
                self.assertRegex(str(error), expected)
                cleanup.assert_called_once()
                self.assertIn(expected, record["stop_reason"])

    def test_termination_uses_only_owned_group_with_bounded_waits(self):
        child = mock.Mock(pid=4321)
        child.wait.side_effect = [RUNNER.subprocess.TimeoutExpired("driver", 5), 0]
        with mock.patch.object(RUNNER.os, "killpg", create=True) as kill:
            # Windows lacks SIGKILL; tests remain CPU-only on either host OS.
            with mock.patch.object(RUNNER.signal, "SIGKILL", 9, create=True):
                RUNNER.terminate_group(child)
        self.assertEqual(kill.call_args_list,
                         [mock.call(4321, RUNNER.signal.SIGTERM), mock.call(4321, 9)])
        self.assertEqual(child.wait.call_args_list, [mock.call(timeout=5), mock.call(timeout=5)])

    def test_gpu_must_be_selected_explicitly(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            RUNNER.build_parser().parse_args([
                "--binary", "driver", "--build-manifest", "manifest.json", "--output-dir", "fresh"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            CHECKER.build_parser().parse_args(["--build-attempt", "build", "--output-dir", "fresh"])

    def test_gpu_index_resolves_to_uuid_without_relaxing_hardware_scope(self):
        completed = RUNNER.subprocess.CompletedProcess([], 0, "GPU-resolved, NVIDIA GeForce RTX 5070\n", "")
        with mock.patch.object(RUNNER.subprocess, "run", return_value=completed) as query:
            self.assertEqual(RUNNER.resolve_gpu("2"), "GPU-resolved")
        self.assertIn("--id=2", query.call_args.args[0])
        self.assertEqual(query.call_args.kwargs["timeout"], 10)

    def test_wrong_or_ambiguous_hardware_is_rejected(self):
        for output in ("GPU-a, NVIDIA GeForce RTX 5090\n", "GPU-a, NVIDIA GeForce RTX 5070\nGPU-b, NVIDIA GeForce RTX 5070\n",
                       "0, NVIDIA GeForce RTX 5070\n", ""):
            with self.subTest(output=output), mock.patch.object(RUNNER.subprocess, "run", return_value=
                    RUNNER.subprocess.CompletedProcess([], 0, output, "")):
                with self.assertRaisesRegex(RuntimeError, "one qualified"):
                    RUNNER.resolve_gpu("0")

    def test_monitor_rejects_gpu_identity_change(self):
        output = "GPU-other, NVIDIA GeForce RTX 5070, 10, 12227, 0, 35, 12.0\n"
        with mock.patch.object(RUNNER.subprocess, "run", return_value=
                RUNNER.subprocess.CompletedProcess([], 0, output, "")):
            with self.assertRaisesRegex(RuntimeError, "authorized RTX 5070"):
                RUNNER.memory_state("GPU-selected")

    def test_runtime_libraries_come_from_manifest_and_keep_existing_search_suffix(self):
        with temporary_directory() as temporary:
            root = Path(temporary).resolve()
            cuda, cudnn = root / "cuda", root / "cudnn"
            cuda.mkdir()
            cudnn.mkdir()
            with mock.patch.dict(RUNNER.os.environ, {"LD_LIBRARY_PATH": "existing-libraries"}, clear=True):
                env = RUNNER.runtime_environment(
                    {"runtime_library_dirs": [str(cuda), str(cudnn), str(cuda)]}, "GPU-selected")
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-selected")
            self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertEqual(env["LD_LIBRARY_PATH"], RUNNER.os.pathsep.join(
                [str(cuda), str(cudnn), "existing-libraries"]))

    def test_missing_runtime_library_manifest_cannot_use_machine_defaults(self):
        for manifest in ({}, {"runtime_library_dirs": []}, {"runtime_library_dirs": "lib"},
                         {"runtime_library_dirs": ["relative/lib"]}, {"runtime_library_dirs": [None]}):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                RUNNER.runtime_environment(manifest, "GPU-selected")

    def test_sanitizer_is_selected_from_build_toolkit_or_explicit_path(self):
        with temporary_directory() as temporary:
            root = Path(temporary).resolve()
            (root / "bin").mkdir()
            sanitizer = root / "bin/compute-sanitizer"
            sanitizer.write_text("fixture")
            self.assertEqual(CHECKER.sanitizer_path({"dependencies": {"cuda_root": str(root)}}, None), sanitizer)
            self.assertEqual(CHECKER.sanitizer_path({}, sanitizer), sanitizer)
        with self.assertRaisesRegex(ValueError, "dependencies.cuda_root"):
            CHECKER.sanitizer_path({}, None)

    def test_checker_watchdog_cleans_only_its_owned_process(self):
        child = mock.Mock(pid=6789)
        child.wait.side_effect = RUNNER.subprocess.TimeoutExpired("check", 1200)
        with mock.patch.object(CHECKER.subprocess, "Popen", return_value=child) as launch, \
             mock.patch.object(CHECKER, "terminate_group") as cleanup:
            with self.assertRaises(RUNNER.subprocess.TimeoutExpired):
                CHECKER.execute_check(["check"], env={}, log=mock.sentinel.log)
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        child.wait.assert_called_once_with(timeout=1200)
        cleanup.assert_called_once_with(child)


if __name__ == "__main__":
    unittest.main()
