"""Proof-producing refutation and the boundaries of countermodel search."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DisproveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ["lake", "build", "VeriTile.Meta.Disprove", "VeriTile.Triton.Float.Disprove",
             "bench.examples.StableLogSumExp.Proofs.Unconditional"],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DisproveCheck.lean"
            path.write_text(source)
            return subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=180)

    def succeeds(self, source):
        result = self.lean(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_refutation_declarations_search_registration_and_bounds(self):
        self.succeeds((ROOT / "bench/tests/Disprove.lean").read_text())

    def test_refutation_cannot_prove_or_export_the_positive_claim(self):
        for source in (
            'refutation false_refutation : True := by trivial',
            'refutation non_proposition : Nat := by disprove',
            'refutation rejected_claim : True := by disprove',
            '''refutation result : (1 : Nat) = 2 := by disprove
example : (1 : Nat) = 2 := result''',
        ):
            with self.subTest(source=source):
                result = self.lean('import VeriTile.Meta.Disprove\n' + source)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("maximum number of heartbeats", result.stdout)

    def test_independent_kernels_domains_casts_and_comparisons(self):
        self.succeeds((ROOT / "bench/tests/FPDisprove.lean").read_text())

    def test_core_does_not_import_numerical_semantics(self):
        self.succeeds('''
import VeriTile.Meta.Disprove
open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `VeriTile.Triton.ComputeKernel then
    throwError "Generic disprove imported the Triton numerical model"
structure UserConfig where
  maxDepth : Nat
  maxNodes : Nat
  maxWitnesses : Nat
example : ¬ (∀ n : Nat, n < 5) := by
  disprove (maxWitnesses := 6) (maxNodes := 100) (maxDepth := 20)
''')

    def test_real_example_uses_context_and_refutes_the_universal_claim(self):
        self.succeeds('''
import bench.examples.StableLogSumExp.Proofs.Unconditional
open VeriTile Triton Bench.Examples StableLogSumExpFPContract
open StableLogSumExpUnconditional
open scoped VeriTile.Spec
refutation from_context (R : FP.LogSumExp.Rules) :
    stable "x" "y" 1 ≡[R] direct "x" "y" 1 := by
  have hm := Model.models R
  have hd := Model.domain_holds
  have outputs := Model.stored_outputs
  disprove
refutation universal (R : FP.Exponential.Rules) : ∀ B, equivalenceGoal B R := by
  disprove [Model.exponential_models R, Model.domain_holds, Model.stored_outputs]
example (R : FP.LogSumExp.Rules) :
    ¬ (stable "x" "y" 1 ≡[R] direct "x" "y" 1) := from_context R
example (R : FP.Exponential.Rules) : ¬ (∀ B, equivalenceGoal B R) := universal R
#axiomsClean from_context
#axiomsClean universal
''')

    def test_failure_is_explicitly_inconclusive(self):
        result = self.lean('''
import VeriTile.Meta.Disprove
example (P : Prop) : ¬ P := by disprove
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disprove found no checked counterexample", result.stdout)
        self.assertIn("Failure is inconclusive", result.stdout)
        self.assertNotIn("maximum number of heartbeats", result.stdout)

    def test_positive_goal_is_rejected(self):
        result = self.lean('''
import VeriTile.Meta.Disprove
example : True := by disprove
''')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expects a negated proposition or False", result.stdout)


if __name__ == "__main__":
    unittest.main()
