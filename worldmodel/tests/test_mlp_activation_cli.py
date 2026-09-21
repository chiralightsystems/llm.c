"""Behavioral CPU checks for the real trainer parser and activation host fixture.

Set LLMC_MLP_ACTIVATION_CLI_BINARY to a built train_gpt2cu executable and
LLMC_MLP_ACTIVATION_HOST_BINARY to test_mlp_activation_host. Supply the build's
normal DLL/library search path when needed. No compiler or GPU test is invoked.
Missing executable configuration skips that class; a configured missing or
failing executable is an error, not a skip.

Every trainer command deliberately fails in argument parsing, before native
CUDA initialization. The subprocess also receives CUDA_VISIBLE_DEVICES="" and
has a timeout. The host fixture checks actual selector math, derivatives and
checkpoint-contract serialization. Full-model replay, checkpoint resume and
cross-selector state rejection belong to test_mlp_activation.cu and require
separately authorized GPU qualification. Cached-decoder rejection is outside
this CPU suite; source-text inspection is not a behavioral qualification for it.
"""

import os
from pathlib import Path
import subprocess
import unittest


SELECTORS = ("gelu", "swish_power125_k8", "swish", "relu_squared", "swish_power2_k8")


def configured_binary(variable: str) -> Path:
    configured = os.environ.get(variable)
    if not configured:
        raise unittest.SkipTest(f"Set {variable} to run the compiled behavior checks")
    binary = Path(configured).resolve(strict=True)
    if not binary.is_file():
        raise ValueError(f"{variable} must name an executable file")
    return binary


def run_cpu_binary(binary: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run(
        [str(binary), *arguments],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )


class ActivationCliDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary = configured_binary("LLMC_MLP_ACTIVATION_CLI_BINARY")

    def assert_rejected(self, arguments: list[str], diagnostic: str) -> None:
        result = run_cpu_binary(self.binary, *arguments)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn(diagnostic, result.stdout)

    def test_unknown_activation_is_rejected_by_real_parser(self) -> None:
        for value in ("swish_power150_k8", "unknown_activation", ""):
            with self.subTest(value=value):
                self.assert_rejected(["-ma", value], f"Unknown -ma activation: {value}")

    def test_all_known_selectors_reach_next_parser_diagnostic(self) -> None:
        for selector in SELECTORS:
            with self.subTest(selector=selector):
                self.assert_rejected(
                    ["-ma", selector, "-pa", "cpu_only_stop"],
                    "-pa expects exactly 0 or 1",
                )

    def test_validation_limit_does_not_consume_activation_option(self) -> None:
        for arguments in (
            ["-m", "7", "-ma", "unknown_activation"],
            ["-ma", "unknown_activation", "-m", "7"],
        ):
            with self.subTest(arguments=arguments):
                self.assert_rejected(arguments, "Unknown -ma activation: unknown_activation")

    def test_preflight_rejects_values_other_than_exact_zero_or_one(self) -> None:
        for value in ("-1", "2", "00", "1x", ""):
            with self.subTest(value=value):
                self.assert_rejected(["-pa", value], "-pa expects exactly 0 or 1")

    def test_valid_preflight_flags_reach_next_parser_diagnostic(self) -> None:
        for value in ("0", "1"):
            with self.subTest(value=value):
                self.assert_rejected(
                    ["-pa", value, "-ma", "unknown_activation"],
                    "Unknown -ma activation: unknown_activation",
                )

    def test_distributed_filesystem_option_retains_its_path_argument(self) -> None:
        self.assert_rejected(
            ["-pf", "unused-filesystem-path", "-ma", "unknown_activation"],
            "Unknown -ma activation: unknown_activation",
        )

    def test_missing_option_values_report_usage(self) -> None:
        for option in ("-ma", "-pa"):
            with self.subTest(option=option):
                self.assert_rejected([option], f"{option} <")


class ActivationHostContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary = configured_binary("LLMC_MLP_ACTIVATION_HOST_BINARY")

    def test_math_derivatives_and_checkpoint_contracts(self) -> None:
        result = run_cpu_binary(self.binary)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("llmc_mlp_activation_host:", result.stdout)
        self.assertIn("result=pass", result.stdout)
        for selector in SELECTORS[1:]:
            self.assertIn(selector, result.stdout)


if __name__ == "__main__":
    unittest.main()
