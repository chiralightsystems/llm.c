"""Run an explicitly configured CPU-only FFN correctness executable.

Set LLMC_FFN_INTERVENTIONS_HOST_BINARY to test_ffn_interventions_host.
Set LLMC_FFN_OPTIMIZER_HOST_BINARY to test_ffn_optimizer_host.
No compilation, CUDA initialization, dataset access or experiment is launched.
"""
import os
from pathlib import Path
import subprocess
import unittest


class FfnInterventionsHostTests(unittest.TestCase):
    def test_independent_finite_differences_baseline_and_navia(self) -> None:
        configured = os.environ.get("LLMC_FFN_INTERVENTIONS_HOST_BINARY")
        if not configured:
            self.skipTest("Set LLMC_FFN_INTERVENTIONS_HOST_BINARY to the CPU fixture")
        binary = Path(configured).resolve(strict=True)
        self.assertTrue(binary.is_file())
        result = subprocess.run(
            [str(binary)], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("result=pass", result.stdout)
        self.assertIn("gum_fp64_fd=1", result.stdout)
        self.assertIn("summed_bank=1", result.stdout)
        self.assertIn("navia=1", result.stdout)

    def test_actual_optimizer_plan_and_rectangular_addressing(self) -> None:
        configured = os.environ.get("LLMC_FFN_OPTIMIZER_HOST_BINARY")
        if not configured:
            self.skipTest("Set LLMC_FFN_OPTIMIZER_HOST_BINARY to the CPU plan fixture")
        binary = Path(configured).resolve(strict=True)
        self.assertTrue(binary.is_file())
        result = subprocess.run(
            [str(binary)], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True, text=True, encoding="utf-8", timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("result=pass", result.stdout)
        self.assertIn("actual_plan=1", result.stdout)
        self.assertIn("bank_adamw=1", result.stdout)
        self.assertIn("cuda_device_calls=0", result.stdout)


if __name__ == "__main__":
    unittest.main()
