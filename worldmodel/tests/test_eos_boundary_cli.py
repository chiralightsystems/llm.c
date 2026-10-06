"""Compiled parser checks; never initializes CUDA. Set LLMC_EOS_CLI_BINARY."""
import os
from pathlib import Path
import subprocess
import unittest

class EosCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value=os.environ.get("LLMC_EOS_CLI_BINARY")
        if not value: raise unittest.SkipTest("Set LLMC_EOS_CLI_BINARY")
        cls.binary=Path(value).resolve(strict=True)

    def rejected(self,args,diagnostic):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="")
        result=subprocess.run([str(self.binary),*args],env=env,capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertIn(diagnostic,result.stdout+result.stderr)

    def test_policy_dispatch(self):
        for policy in ("row_causal_v1","isolate_segments_v1"):
            self.rejected(["-ab",policy,"-pa","cpu_only_stop"],"-pa expects exactly 0 or 1")
        for policy in ("row_reset","opaque","isolate_segments_v2",""):
            self.rejected(["-ab",policy],"Unknown -ab attention boundary:")

    def test_eos_dispatch_does_not_become_model_option(self):
        for value in ("-1","0","50256"):
            self.rejected(["-e","unused.bin","-ei",value,"-pa","cpu_only_stop"],"-pa expects exactly 0 or 1")
        for value in ("-2","922337203685477580899","x","127x",""):
            self.rejected(["-ei",value],"Invalid -ei EOS token:")

    def test_missing_values(self):
        for flag in ("-ab","-ei"):
            self.rejected([flag],flag+" <")

if __name__=="__main__":unittest.main()
