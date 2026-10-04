"""Original logsumexp execution, scalar obligations and assumption reporting."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LogSumExpFPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ['lake', 'build', 'bench.examples.support.StableLogSumExpContract',
             'bench.examples.StableLogSumExpCorrect', 'VeriTile.Meta.StatementAudit',
             'VeriTile.Triton.Float.LogExp', 'VeriTile.Triton.Float.LogExpCounterexample'],
            cwd=ROOT, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check_lean(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'LogSumExpCheck.lean'
            path.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        return result.stdout

    def test_original_sources_and_real_spec_independence(self):
        output = self.check_lean('''
import bench.examples.support.StableLogSumExpContract
import bench.examples.StableLogSumExpCorrect
open VeriTile Triton VeriTile.Bench.Examples
example (x y : RegionName) (B : Nat) :
    StableLogSumExpFPExecution.directLSEKernel x y B =
      StableLogSumExpCorrect.directLSEKernel x y B := rfl
example (x y : RegionName) (B : Nat) :
    StableLogSumExpFPExecution.stableLSEKernel x y B =
      StableLogSumExpCorrect.stableLSEKernel x y B := rfl
''')
        self.assertEqual(output, '')
        self.check_lean('''
import bench.examples.support.StableLogSumExpContract
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for name in [
      `VeriTile.Bench.Examples.StableLogSumExpCorrect.directLSEKernel,
      `VeriTile.Bench.Examples.SoftmaxStableCorrect.naiveSoftmaxKernel] do
    if env.contains name then throwError "FP proof imported a correctness counterpart: {name}"
''')

    def test_scalar_gaps_casts_domains_outputs_and_projected_premises(self):
        output = self.check_lean((ROOT / 'bench/tests/FPStableLogSumExp.lean').read_text())
        expected = ''.join(
            f'FP assumptions used by {name}:\n'
            '  unresolved FP proof: h (atomic assumptions unavailable)\n'
            for name in ['opaque_logsumexp', 'rebuilt_exp_sub', 'rebuilt_log_mul',
                         'rebuilt_log_exp', 'rebuilt_count'])
        expected += ('FP assumptions used by wrapped_log_mul:\n'
                     '  unresolved FP proof: h.1 (atomic assumptions unavailable)\n'
                     'FP assumptions used by exp_sub_holds:\n  none\n')
        self.assertEqual(output[output.index('FP assumptions used by'):], expected)

    def test_pr13_branching_precision_and_exact_counterexamples(self):
        self.check_lean((ROOT / 'bench/tests/FPLogExp.lean').read_text())

    def test_pr13_prints_only_the_new_expression_atom(self):
        output = self.check_lean('''
import VeriTile.Triton.Float.LogExp
#print_fp_assumptions VeriTile.Triton.FP.LogExp.log_exp_expm1_equiv
''')
        self.assertEqual(output, 'FP assumptions used by log_exp_expm1_equiv:\n'
                                 '  log_exp_expm1\n')


if __name__ == '__main__':
    unittest.main()
