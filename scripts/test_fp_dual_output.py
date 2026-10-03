"""Two-output FP contracts, execution boundaries and assumption reporting."""
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FPDualOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ["lake", "build", "VeriTile.Triton.Float.GuardedIO", "VeriTile.Meta.StatementAudit"],
            cwd=ROOT, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def test_two_outputs_frames_signatures_and_assumption_reporting(self):
        result = subprocess.run(
            ["lake", "env", "lean", "bench/tests/FPDualOutput.lean"],
            cwd=ROOT, text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)
        self.assertIn("FP assumptions used by", result.stdout)
        self.assertEqual(result.stdout[result.stdout.index("FP assumptions used by"):],
                         "FP assumptions used by reordered_outputs:\n  none\n"
                         "FP assumptions used by guarded_reordered_outputs:\n  none\n"
                         "FP assumptions used by opaque_guarded_outputs:\n"
                         "  unresolved FP proof: h (atomic assumptions unavailable)\n")


if __name__ == "__main__":
    unittest.main()
