"""Lean regressions for structural execution and independent kernel FP proofs."""
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
             "bench.examples.FusedSiLU.FPEquiv", "bench.examples.FusedSiLU.Proofs.LegacyRealEquiv",
             "bench.examples.FusedSwiglu.FPEquiv", "bench.examples.FusedSwiglu.Proofs.LegacyRealEquiv",
             "bench.examples.RowWiseMax.FPEquiv", "bench.examples.RowWiseMax.Correct"], cwd=ROOT,
            text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def test_numerical_laws_and_failures_are_not_structural_certificates(self):
        result = subprocess.run(
            ["lake", "env", "lean", "bench/tests/FPStructural.lean"], cwd=ROOT,
            text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_silu_proof_has_no_numerical_assumptions(self):
        source = (ROOT / "bench/examples/FusedSiLU/FPEquiv.lean").read_text() + '''
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
import bench.examples.FusedSiLU.Proofs.LegacyRealEquiv
import bench.examples.FusedSiLU.FPEquiv
open VeriTile Triton
open VeriTile.Bench.Examples
open scoped VeriTile.Spec
example (B : Nat) :
    (FusedSiLUFPEquiv.fusedIO B).kernel =
      FusedSiLU.Kernels.fusedSiLUKernel "x" "gate" "residual" "out" B := rfl
example (B : Nat) :
    (FusedSiLUFPEquiv.unfusedIO B).kernel =
      FusedSiLU.Kernels.unfusedSiLUKernel "x" "gate" "residual" "z" "silu" "out" B := rfl
example (B : Nat) :
    FusedSiLUFPEquiv.fusedIO B ≡[FusedSiLUFPEquiv.R] FusedSiLUFPEquiv.unfusedIO B :=
  FusedSiLUFPEquiv.silu_equiv B
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_swiglu_proof_has_no_numerical_assumptions(self):
        source = (ROOT / "bench/examples/FusedSwiglu/FPEquiv.lean").read_text() + '''
open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `VeriTile.Bench.Examples.FusedSwigluCorrect.fusedIO then
    throwError "FP proof imported its real correctness counterpart"
'''
        result = self.lean(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "FP assumptions used by swiglu_equiv:\n  none\n")

    def test_swiglu_preserves_original_kernels_casts_masks_and_symbolic_shape(self):
        result = self.lean('''
import bench.examples.FusedSwiglu.Proofs.LegacyRealEquiv
import bench.examples.FusedSwiglu.FPEquiv
open VeriTile Triton
open VeriTile.Bench.Examples
open scoped VeriTile.Spec
example (n B : Nat) :
    (FusedSwigluFPEquiv.fusedIO n B).kernel =
      FusedSwiglu.Kernels.swiglu_fused "X" "Y" "OUT" n B := rfl
example (n B : Nat) :
    (FusedSwigluFPEquiv.unfusedIO n B).kernel =
      FusedSwiglu.Kernels.swiglu_unfused "X" "Y" "S" "OUT" n B := rfl
example (n B : Nat) :
    FusedSwigluFPEquiv.fusedIO n B ≡[FusedSwigluFPEquiv.R]
      FusedSwigluFPEquiv.unfusedIO n B := FusedSwigluFPEquiv.swiglu_equiv n B
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rowmax_proof_is_independent_and_has_no_numerical_assumptions(self):
        source = (ROOT / "bench/examples/RowWiseMax/FPEquiv.lean").read_text() + '''
open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `VeriTile.Bench.Examples.RowWiseMax.rowWiseMaxIO then
    throwError "FP proof imported its real correctness counterpart"
'''
        result = self.lean(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout, "FP assumptions used by rowwise_max_equiv:\n  none\n")

    def test_rowmax_preserves_original_source_and_symbolic_nonempty_rows(self):
        result = self.lean('''
import bench.examples.RowWiseMax.Correct
import bench.examples.RowWiseMax.FPEquiv
open VeriTile Triton
open VeriTile.Bench.Examples
open scoped VeriTile.Spec
example (nCol B : Nat) :
    (RowWiseMaxFPEquiv.originalIO nCol B).kernel =
      RowWiseMax.Kernels.rowWiseMaxKernel "x" "y" nCol B := rfl
example (nCol B : Nat) (hB : 0 < B) :
    RowWiseMaxFPEquiv.originalIO nCol B ≡[RowWiseMaxFPEquiv.R]
      RowWiseMaxFPEquiv.inlinedIO nCol B := RowWiseMaxFPEquiv.rowwise_max_equiv nCol B hB
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
