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
             'bench.examples.Welford.Proofs.FP',
             'bench.examples.OnlineSoftmax.Proofs.FP',
             'bench.examples.Welford.Correct', 'bench.examples.OnlineSoftmax.Correct'],
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

    def test_exact_original_kernels_and_softmax_normalizer_helper(self):
        self.lean('''
import bench.examples.Welford.Proofs.FP
import bench.examples.OnlineSoftmax.Proofs.FP
import bench.examples.Welford.Correct
import bench.examples.OnlineSoftmax.Correct
open VeriTile Triton VeriTile.Bench.Examples
example (N stride : Nat) :
    (WelfordCorrect.onlineIO N stride).kernel =
      (WelfordFPExecution.onlineIO "x" "mean" "var" N stride).kernel.eraseDType := rfl
example (N stride : Nat) :
    (WelfordCorrect.twopassIO N stride).kernel =
      (WelfordFPExecution.twopassIO "x" "mean" "var" N stride).kernel.eraseDType := rfl
example (x y : RegionName) (N : Nat) :
    (OnlineSoftmax.Kernels.onlineNormalizerKernel x y N).surfaceBody =
      OnlineSoftmaxFPExecution.initialCode ++
        [.forLoop "i" N (OnlineSoftmaxFPExecution.body x N)] :=
  OnlineSoftmaxFPExecution.kernel_body x y N
''')


if __name__ == '__main__':
    unittest.main()
