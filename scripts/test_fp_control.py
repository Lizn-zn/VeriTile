"""Control-flow boundaries and original FP recurrence execution proofs."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FPControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ['lake', 'build', 'VeriTile.Triton.Float.Control',
             'bench.examples.support.WelfordExecution',
             'bench.examples.support.OnlineSoftmaxExecution',
             'bench.examples.WelfordCorrect', 'bench.examples.OnlineSoftmaxCorrect'],
            cwd=ROOT, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'FPControlCheck.lean'
            path.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)

    def test_loop_order_bounds_failures_and_nested_precision(self):
        self.lean((ROOT / 'bench/tests/FPControl.lean').read_text())

    def test_recurrences_domains_casts_frames_and_independent_sources(self):
        self.lean((ROOT / 'bench/tests/FPRecurrenceExecution.lean').read_text())

    def test_exact_original_kernels_including_both_stores_and_no_softmax_store(self):
        self.lean('''
import bench.examples.support.WelfordExecution
import bench.examples.support.OnlineSoftmaxExecution
import bench.examples.WelfordCorrect
import bench.examples.OnlineSoftmaxCorrect
open VeriTile Triton VeriTile.Bench.Examples
example (x m v : RegionName) (N stride : Nat) :
    WelfordFPExecution.onlineWelfordKernel x m v N stride =
      WelfordCorrect.onlineWelfordKernel x m v N stride := rfl
example (x m v : RegionName) (N stride : Nat) :
    WelfordFPExecution.twopassWelfordKernel x m v N stride =
      WelfordCorrect.twopassWelfordKernel x m v N stride := rfl
example (x y : RegionName) (N : Nat) :
    OnlineSoftmaxFPExecution.onlineSoftmaxKernel x y N =
      OnlineSoftmax.onlineSoftmaxKernel x y N := rfl
''')


if __name__ == '__main__':
    unittest.main()
