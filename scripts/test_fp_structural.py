"""Lean regressions for structural execution and the public SiLU FP proof."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FPStructuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ["lake", "build", "VeriTile.Triton.Float.StructuralIO",
             "bench.examples.FusedSiLUFPEquiv", "bench.examples.FusedSiLUEquiv"], cwd=ROOT,
            text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def test_numerical_laws_and_failures_are_not_structural_certificates(self):
        result = subprocess.run(
            ["lake", "env", "lean", "bench/tests/FPStructural.lean"], cwd=ROOT,
            text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_silu_proof_has_no_numerical_assumptions(self):
        source = (ROOT / "bench/examples/FusedSiLUFPEquiv.lean").read_text() + '''
open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `VeriTile.Bench.Examples.FusedSiLUCorrect.fusedIO then
    throwError "FP proof imported its real correctness counterpart"
'''
        result = self.lean(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "FP assumptions used by silu_equiv:\n  none\n")

    def lean(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "StructuralPair.lean"
            path.write_text(source)
            return subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                                  text=True, capture_output=True, timeout=180)

    def test_silu_preserves_both_original_kernels_and_symbolic_shape(self):
        result = self.lean('''
import bench.examples.FusedSiLUEquiv
import bench.examples.FusedSiLUFPEquiv
open VeriTile Triton
open VeriTile.Bench.Examples
open scoped VeriTile.Spec
example (x g r o : RegionName) (B : Nat) :
    FusedSiLUFPEquiv.fusedSiLUKernel x g r o B =
      FusedSiLUEquiv.fusedSiLUKernel x g r o B := rfl
example (x g r z s o : RegionName) (B : Nat) :
    FusedSiLUFPEquiv.unfusedSiLUKernel x g r z s o B =
      FusedSiLUEquiv.unfusedSiLUKernel x g r z s o B := rfl
example (B : Nat) :
    FusedSiLUFPEquiv.fusedIO B ≡[FusedSiLUFPEquiv.R] FusedSiLUFPEquiv.unfusedIO B :=
  FusedSiLUFPEquiv.silu_equiv B
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
