"""Check proof-producing equivalence tactics and their failure boundaries."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    "VectorAdd", "FlatVectorAdd", "FloatDTypeAdd", "TritonBenchVectorAddition",
    "AdamUpdateGridLaunch", "HyperConnectionsDepth", "HyperConnectionsWidth",
)


class EquivalenceTacticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ["lake", "build", "VeriTile.Meta.FPProve"]
            + [f"bench.examples.{family}.FPEquiv" for family in FAMILIES],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "EquivTacticCheck.lean"
            path.write_text(source)
            return subprocess.run(
                ["lake", "env", "lean", str(path)], cwd=ROOT,
                capture_output=True, text=True, timeout=180,
            )

    def test_composition_metadata_precision_and_admission_boundaries(self):
        result = self.lean((ROOT / "bench/tests/EquivTactics.lean").read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)

    def test_decomposition_is_independent_of_the_fp_model(self):
        result = self.lean("""
import VeriTile.Meta.EquivDecompose
open VeriTile
open scoped VeriTile.Spec
theorem example_decomposition (R : Spec.Assumptions Nat)
    (h : Spec.Derivation R [1] [2]) : [0, 1, 9] ≡[R] [0, 2, 9] := by
  equiv_decompose
  exact h
open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `VeriTile.Triton.ComputeKernel then
    throwError "Generic decomposition imported the Triton numerical model"
""")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_failed_search_explains_the_remaining_obligation(self):
        result = self.lean("""
import VeriTile.Meta.FPProve
example : VeriTile.Spec.Derivation ([] : VeriTile.Spec.Assumptions Nat) [1] [2] := by
  fp_prove
""")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fp_prove could not close this goal", result.stdout)
        self.assertIn("domain premise or structural lemma", result.stdout)
        self.assertNotIn("maximum number of heartbeats", result.stdout)


if __name__ == "__main__":
    unittest.main()
