"""Scalar-derived reduction permutations, precision, and assumption auditing."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FPEquationalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ["lake", "build", "bench.examples.RowWiseSum.FPEquiv",
             "bench.examples.RowWiseSum.Correct"], cwd=ROOT,
            text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "EquationalCheck.lean"
            path.write_text(source)
            result = subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_scalar_and_reduction_boundaries(self):
        output = self.lean((ROOT / "bench/tests/FPEquational.lean").read_text())
        self.assertIn("FP assumptions used by reduction_reorder:\n", output)
        self.assertIn("unresolved FP proof:", output)
        self.assertNotIn("\n  add_assoc\n", output)
        self.assertIn("FP assumptions used by opaque_term:\n  unresolved FP proof: h", output)
        self.assertIn("FP assumptions used by same_tree:\n  none\n", output)
        self.assertIn("FP assumptions used by reassociation_only:\n  unresolved FP proof:", output)
        self.assertIn("FP assumptions used by inside_opaque_exp:\n  add_commute\n", output)
        self.assertEqual(output.count("unresolved FP proof"), 3)

    def test_current_admissions_do_not_silently_supply_missing_algebra(self):
        self.assertEqual(self.lean((ROOT / "bench/tests/FPAdmissionCoverage.lean").read_text()), "")

    def test_kernel_proof_is_independent_and_reports_both_admitted_atoms(self):
        source = (ROOT / "bench/examples/RowWiseSum/FPEquiv.lean").read_text() + '''
open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `VeriTile.Bench.Examples.RowWiseSum.rowWiseSumIO then
    throwError "FP proof imported its real correctness counterpart"
'''
        output = self.lean(source)
        self.assertIn("FP assumptions used by rowwise_sum_equiv:\n", output)
        self.assertIn("\n  add_commute\n", output)
        self.assertNotIn("unresolved FP proof:", output)
        self.assertIn("\n  add_assoc\n", output)

    def test_same_original_source_and_symbolic_dimensions_including_zero(self):
        self.lean('''
import bench.examples.RowWiseSum.Correct
import bench.examples.RowWiseSum.FPEquiv
open VeriTile Triton
open VeriTile.Bench.Examples
open scoped VeriTile.Spec
example (nCol B : Nat) :
    (RowWiseSum.rowWiseSumIO nCol B).kernel =
      (RowWiseSumFPEquiv.originalIO nCol B).kernel := rfl
example (nCol B : Nat) (R : RowWiseSumFPEquiv.Rules) :
    RowWiseSumFPEquiv.originalIO nCol B ≡[R] RowWiseSumFPEquiv.reversedIO nCol B :=
  RowWiseSumFPEquiv.rowwise_sum_equiv nCol B R
example (nCol : Nat) (R : RowWiseSumFPEquiv.Rules) :
    RowWiseSumFPEquiv.originalIO nCol 0 ≡[R] RowWiseSumFPEquiv.reversedIO nCol 0 :=
  RowWiseSumFPEquiv.rowwise_sum_equiv nCol 0 R
''')

    def test_opaque_execution_premises_are_not_reported_as_no_atoms(self):
        output = self.lean('''
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Meta.StatementAudit
open VeriTile Triton FP.Structural
open scoped VeriTile.Spec
theorem fromOpaque (R : Spec.Assumptions ComputeStmt) (a b : KernelIO₁)
    (hs : io₁Signature a = io₁Signature b) (h : IO₁NumericalEquiv R a b) : a ≡[R] b :=
  Spec.FloatingPoint.ofNumerical (structural := IO₁Equiv) rfl hs h
#print_fp_assumptions fromOpaque
theorem optionalPremise (R : Spec.Assumptions ComputeStmt) (a b : Option (FP.Equational.Term Nat))
    (h : FP.Equational.OptionalEq R a b) : FP.Equational.OptionalEq R a b := h
#print_fp_assumptions optionalPremise
''')
        self.assertEqual(output.count("unresolved FP proof: h"), 2)
        self.assertNotIn("  none", output)

    def test_reduction_precision_survives_explicit_default_and_keepdims_forms(self):
        source = "import VeriTile.Triton.DSL\nopen VeriTile Triton\n"
        for name, kwargs, shape, keep in (
            ("explicitAxis", ", axis=0", "[]", "false"),
            ("defaultAxis", "", "[]", "false"),
            ("keepAxis", ", axis=0, keep_dims=true", "[1]", "true"),
        ):
            output_offset = "tl.arange(0, 1)" if keep == "true" else "0"
            source += f'''
def {name} (B : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  triton {{
    values := tl.load(x_ptr + tl.arange(0, $(B)))
    result := tl.sum(values{kwargs})
    tl.store($("out") + {output_offset}, result)
  }}
example (B : Nat) : ({name} B).surfaceBody[1]? =
    some (.assign .real {shape} "result" (.compute (.alg .fp32
      (.reduceSum ⟨0, by simp⟩ Bool.{keep} (.ref .real [B] "values"))))) := rfl
'''
        self.lean(source)


if __name__ == "__main__":
    unittest.main()
