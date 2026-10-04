"""Libdevice softmax, scalar normalization and scheduled IO boundaries."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SoftmaxStableFPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ['lake', 'build', 'bench.examples.SoftmaxStable.Contract',
             'bench.examples.SoftmaxStable.Correct', 'VeriTile.Meta.StatementAudit'],
            cwd=ROOT, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check_lean(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'SoftmaxCheck.lean'
            path.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        return result.stdout

    def test_original_sources_and_real_spec_independence(self):
        output = self.check_lean('''
import bench.examples.SoftmaxStable.Contract
import bench.examples.SoftmaxStable.Correct
open VeriTile Triton VeriTile.Bench.Examples
example (B : Nat) :
    (SoftmaxStableCorrect.naiveIO B).kernel =
      (SoftmaxStableFPExecution.naiveIO "x" "y" B).kernel.eraseDType := rfl
example (x y : RegionName) (B : Nat) :
    (SoftmaxStableFPExecution.naiveIO x y B).kernel = SoftmaxStable.Kernels.naiveSoftmaxKernel x y B := rfl
example (B : Nat) :
    (SoftmaxStableCorrect.stableIO B).kernel =
      (SoftmaxStableFPExecution.stableIO "x" "y" B).kernel.eraseDType := rfl
example (x y : RegionName) (B : Nat) :
    (SoftmaxStableFPExecution.stableIO x y B).kernel = SoftmaxStable.Kernels.stableSoftmaxKernel x y B := rfl
''')
        self.assertEqual(output, '')
        self.check_lean('''
import bench.examples.SoftmaxStable.Contract
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  if env.contains `VeriTile.Bench.Examples.SoftmaxStableCorrect.naive_softmax_correct then
    throwError "FP proof imported its correctness counterpart"
''')

    def test_libdevice_domain_execution_and_assumption_reporting(self):
        output = self.check_lean((ROOT / 'bench/tests/FPSoftmaxStable.lean').read_text())
        self.assertEqual(output[output.index('FP assumptions used by'):],
                         'FP assumptions used by opaque_softmax:\n'
                         '  unresolved FP proof: h (atomic assumptions unavailable)\n'
                         'FP assumptions used by naive_self_spec:\n  none\n')


if __name__ == '__main__':
    unittest.main()
