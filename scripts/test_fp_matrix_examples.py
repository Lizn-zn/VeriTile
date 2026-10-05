"""General mHC sources, executable matrix contexts, and atomic assumptions."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MatrixExampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = [f"bench.examples.HyperConnections{side}.{part}"
                   for side in ("Depth", "Width") for part in ("Correct", "FPEquiv")]
        result = subprocess.run(["lake", "build", *targets], cwd=ROOT,
                                text=True, capture_output=True, timeout=600)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "MatrixExamples.lean"
            path.write_text(source)
            result = subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("warning:", result.stdout)
        return result.stdout

    def test_nonsquare_iterations_backend_and_precision(self):
        self.assertEqual(self.lean((ROOT / "bench/tests/FPMatrixExamples.lean").read_text()), "")

    def test_general_fp_specs_only_use_their_scalar_atoms(self):
        for side, atom in (("Depth", "add_commute"), ("Width", "div_mul_rcp")):
            with self.subTest(side=side):
                name = f"VeriTile.Bench.Examples.HyperConnections{side}FPEquiv.mhc_{side.lower()}_matrix_equiv"
                output = self.lean(f"""
import bench.examples.HyperConnections{side}.FPEquiv
#print_fp_assumptions {name}
open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `VeriTile.Bench.Examples.HyperConnections{side}.MatrixCorrect.mhc_{side.lower()}_matrix_correct then
    throwError "FP specification imported real correctness"
""")
                self.assertEqual(output.splitlines()[1:], [f"  {atom}"])

    def test_general_real_specs_have_no_fp_dependency(self):
        for side in ("Depth", "Width"):
            with self.subTest(side=side):
                self.lean(f"""
import bench.examples.HyperConnections{side}.Correct
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  unless env.contains `VeriTile.Bench.Examples.HyperConnections{side}.MatrixCorrect.mhc_{side.lower()}_matrix_correct do
    throwError "General original correctness is missing"
  unless env.contains `VeriTile.Bench.Examples.HyperConnections{side}.MatrixCorrect.mhc_{side.lower()}_matrix_optimized_correct do
    throwError "General optimized correctness is missing"
  if env.contains `VeriTile.Bench.Examples.HyperConnections{side}FPEquiv.mhc_{side.lower()}_matrix_equiv then
    throwError "Real correctness imported FP equivalence"
""")


if __name__ == "__main__":
    unittest.main()
