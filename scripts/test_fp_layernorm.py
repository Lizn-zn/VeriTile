"""Original LayerNorm execution, unrounded statistics and scheduled IO boundaries."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LayerNormFPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ['lake', 'build', 'bench.examples.support.LayerNormContract',
             'bench.examples.FusedLayerNormCorrect', 'VeriTile.Meta.StatementAudit'],
            cwd=ROOT, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check_lean(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'LayerNormCheck.lean'
            path.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        return result.stdout

    def test_original_sources_and_real_spec_independence(self):
        output = self.check_lean('''
import bench.examples.support.LayerNormContract
import bench.examples.FusedLayerNormCorrect
open VeriTile Triton VeriTile.Bench.Examples
example (x g b y : RegionName) (N stride : Nat) (ε : ℝ) :
    LayerNormFPExecution.fusedLayerNormKernel x g b y N stride ε =
      FusedLayerNormCorrect.fusedLayerNormKernel x g b y N stride ε := rfl
example (x g b y : RegionName) (N stride : Nat) (ε : ℝ) :
    LayerNormFPExecution.twoPassLayerNormKernel x g b y N stride ε =
      FusedLayerNormCorrect.twoPassLayerNormKernel x g b y N stride ε := rfl
''')
        self.assertEqual(output, '')
        self.check_lean('''
import bench.examples.support.LayerNormContract
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for name in [
      `VeriTile.Bench.Examples.FusedLayerNormCorrect.twoPassLayerNormKernel,
      `VeriTile.Bench.Examples.WelfordCorrect.onlineWelfordKernel] do
    if env.contains name then throwError "FP proof imported a correctness counterpart: {name}"
''')

    def test_unrounded_statistics_domains_frames_and_assumption_reporting(self):
        output = self.check_lean((ROOT / 'bench/tests/FPLayerNormExecution.lean').read_text())
        self.assertEqual(output[output.index('FP assumptions used by'):],
                         'FP assumptions used by opaque_layernorm:\n'
                         '  unresolved FP proof: h (atomic assumptions unavailable)\n'
                         'FP assumptions used by empty_layernorm:\n  none\n')


if __name__ == '__main__':
    unittest.main()
