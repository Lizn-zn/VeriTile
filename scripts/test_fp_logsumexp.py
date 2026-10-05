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
            ['lake', 'build', 'bench.examples.StableLogSumExp.Contract',
             'bench.examples.StableLogSumExp.Correct', 'VeriTile.Meta.StatementAudit',
             'bench.examples.LogExp.Correct', 'bench.examples.LogExp.FPEquiv',
             'VeriTile.Triton.Float.LogExpCounterexample'],
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
import bench.examples.StableLogSumExp.Contract
import bench.examples.StableLogSumExp.Correct
open VeriTile Triton VeriTile.Bench.Examples
example (B : Nat) :
    (StableLogSumExpCorrect.directIO B).kernel =
      (StableLogSumExpFPExecution.directIO "x" "y" B).kernel.eraseDType := rfl
example (x y : RegionName) (B : Nat) :
    (StableLogSumExpFPExecution.directIO x y B).kernel = StableLogSumExp.Kernels.directLSEKernel x y B := rfl
example (B : Nat) :
    (StableLogSumExpCorrect.stableIO B).kernel =
      (StableLogSumExpFPExecution.stableIO "x" "y" B).kernel.eraseDType := rfl
example (x y : RegionName) (B : Nat) :
    (StableLogSumExpFPExecution.stableIO x y B).kernel = StableLogSumExp.Kernels.stableLSEKernel x y B := rfl
''')
        self.assertEqual(output, '')
        self.check_lean('''
import bench.examples.StableLogSumExp.Contract
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for name in [
      `VeriTile.Bench.Examples.StableLogSumExpCorrect.direct_logsumexp_correct,
      `VeriTile.Bench.Examples.SoftmaxStableCorrect.naive_softmax_correct] do
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

    def test_piecewise_branching_precision_and_exact_counterexamples(self):
        self.check_lean((ROOT / 'bench/tests/FPLogExp.lean').read_text())

    def test_logsumexp_candidate_branches_casts_and_source_binding(self):
        self.check_lean((ROOT / 'bench/tests/FPLogSumExpCandidate.lean').read_text())

    def test_log_exp_prints_only_its_elimination_rule(self):
        output = self.check_lean('''
import bench.examples.LogExp.FPEquiv
#print_fp_assumptions VeriTile.Bench.Examples.LogExp.FPEquiv.log_exp_equiv
''')
        self.assertEqual(output, 'FP assumptions used by log_exp_equiv:\n'
                                 '  log_exp_elim\n')

    def test_log_exp_shared_sources_and_independent_proofs(self):
        self.check_lean('''
import bench.examples.LogExp.Correct
import bench.examples.LogExp.FPEquiv
open VeriTile Triton Bench.Examples.LogExp
example (x y : RegionName) (B : Nat) :
    (Correct.originalIO x y B).kernel = (FPEquiv.originalIO x y B).io.kernel := rfl
example (x y : RegionName) (B : Nat) :
    (Correct.optimizedIO x y B).kernel = (FPEquiv.optimizedIO x y B).io.kernel := rfl
#axiomsClean Correct.original_correct
#axiomsClean Correct.optimized_correct
#axiomsClean FPEquiv.log_exp_equiv
''')
        for module, forbidden in [
            ('Kernels', ['Correct.originalIO', 'FPEquiv.originalIO']),
            ('Correct', ['FPEquiv.originalIO']),
            ('FPEquiv', ['Correct.originalIO']),
        ]:
            with self.subTest(module=module):
                names = ', '.join(f'`VeriTile.Bench.Examples.LogExp.{n}' for n in forbidden)
                if module != 'FPEquiv':
                    names += ', `VeriTile.Triton.FP.LogExp.Rules'
                self.check_lean(f'''
import bench.examples.LogExp.{module}
import Lean
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for name in [{names}] do
    if env.contains name then throwError "Unexpected proof dependency: {{name}}"
''')


if __name__ == '__main__':
    unittest.main()
