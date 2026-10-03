"""Original online softmax recurrence and its scalar-derived prefix invariant."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OnlineSoftmaxFPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ['lake', 'build', 'bench.examples.support.OnlineSoftmaxContract',
             'bench.examples.OnlineSoftmaxCorrect', 'VeriTile.Meta.StatementAudit'],
            cwd=ROOT, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check_lean(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'OnlineSoftmaxCheck.lean'
            path.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        return result.stdout

    def test_original_source_and_real_spec_independence(self):
        output = self.check_lean('''
import bench.examples.support.OnlineSoftmaxContract
import bench.examples.OnlineSoftmaxCorrect
open VeriTile Triton VeriTile.Bench.Examples
example (x y : RegionName) (N : Nat) :
    OnlineSoftmaxFPExecution.onlineSoftmaxKernel x y N =
      OnlineSoftmax.onlineSoftmaxKernel x y N := rfl
example (x y : RegionName) (N : Nat) :
    OnlineSoftmaxFPBatch.stableSoftmaxKernel x y N =
      OnlineSoftmax.stableSoftmaxKernel x y N := rfl
''')
        self.assertEqual(output, '')
        self.check_lean('''
import bench.examples.support.OnlineSoftmaxContract
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  if env.contains `VeriTile.Bench.Examples.OnlineSoftmax.onlineSoftmaxKernel then
    throwError "FP recurrence proof imported its correctness counterpart"
''')

    def test_invariant_initialization_iteration_domains_and_unchanged_memory(self):
        self.check_lean((ROOT / 'bench/tests/FPOnlineSoftmax.lean').read_text())


if __name__ == '__main__':
    unittest.main()
