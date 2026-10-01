"""Cross-check the independently written real and FP example kernels."""
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    ("VectorAdd", "add_kernel_equiv", "VeriTile.Bench.Examples.VectorAdd.addIO"),
    ("FlatVectorAdd", "add_kernel_masked_equiv", "VeriTile.Bench.Examples.FlatVectorAdd.addMaskedIO"),
    ("FloatDTypeAdd", "float_add_equiv", "VeriTile.Bench.Examples.FloatDTypeAddCorrect.floatAddIO"),
)


class ExamplePairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = [f"bench.examples.{name}{suffix}"
                   for name, _, _ in PAIRS for suffix in ("Correct", "FPEquiv")]
        build = subprocess.run(["lake", "build", *targets], cwd=ROOT,
                               text=True, capture_output=True, timeout=300)
        if build.returncode:
            raise AssertionError(build.stdout + build.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PairCheck.lean"
            path.write_text(source)
            result = subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_same_original_implementation_in_both_files(self):
        """A correct file for one implementation cannot certify a different FP input."""
        imports = "\n".join(f"import bench.examples.{n}{s}"
                            for n, _, _ in PAIRS for s in ("Correct", "FPEquiv"))
        self.lean(imports + '''
open VeriTile.Bench.Examples

example (B : Nat) :
    (VectorAddFPEquiv.originalKernel B).toAlgorithm? =
      (VectorAdd.addKernel "x" "y" "out" B).toAlgorithm? := rfl

example (n B : Nat) :
    (FlatVectorAddFPEquiv.originalKernel n B).toAlgorithm? =
      (FlatVectorAdd.addKernelMasked "x" "y" "out" B n).toAlgorithm? := rfl

example (B : Nat) :
    (FloatDTypeAddFPEquiv.originalKernel B).toAlgorithm? =
      (FloatDTypeAddCorrect.floatAddKernel "x" "y" "out" B).toAlgorithm? := rfl
''')

    def test_fp_files_are_independent_and_print_only_used_atoms(self):
        for name, headline, correct_io in PAIRS:
            with self.subTest(case=name):
                source = (ROOT / f"bench/examples/{name}FPEquiv.lean").read_text()
                source += f'''
open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `{correct_io} then
    throwError "FP example imported the correctness implementation"
'''
                self.assertEqual(self.lean(source),
                                 f"FP assumptions used by {headline}:\n  add_commute\n")

    def test_real_float_add_includes_empty_tiles(self):
        self.lean('''
import bench.examples.FloatDTypeAddCorrect
open VeriTile.Bench.Examples.FloatDTypeAddCorrect
open scoped VeriTile.Triton.KernelIO₂

example : VeriTile.Spec.Real (floatAddIO 0 ⊨ fun xs ys i => xs i + ys i) :=
  float_add_correctness 0
''')


if __name__ == "__main__":
    unittest.main()
