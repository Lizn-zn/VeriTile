"""Library exp identity and the completed guarded softmax equivalence."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LibdeviceExpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(['lake', 'build', 'bench.examples.SoftmaxStableFPEquiv'],
                                cwd=ROOT, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'LibdeviceCheck.lean'
            path.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        return result.stdout

    def test_nested_identity_precision_and_distinct_interpretations(self):
        self.check((ROOT / 'bench/tests/FPLibdeviceExp.lean').read_text())

    def test_public_spec_uses_only_admitted_atoms(self):
        output = self.check('''
import bench.examples.SoftmaxStableFPEquiv
#print_fp_assumptions VeriTile.Bench.Examples.SoftmaxStableFPEquiv.softmax_stable_equiv
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  if env.contains `VeriTile.Bench.Examples.SoftmaxStableCorrect.naiveSoftmaxKernel then
    throwError "FP specification imported its correctness counterpart"
''')
        self.assertEqual(set(output.splitlines()[1:]), {
            '  add_commute', '  add_assoc', '  mul_commute', '  mul_assoc',
            '  mul_distrib', '  cancel', '  add_zero', '  mul_one',
            '  div_mul_rcp', '  mul_rcp_cancel', '  exp_sub'})


if __name__ == '__main__':
    unittest.main()
