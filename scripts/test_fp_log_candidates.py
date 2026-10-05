"""Lean candidates exist before admission and preserve their exact syntax."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
# The unadmitted FMA/log1p probe still needs an explicit fused-operation model.
EXPERIMENT_ONLY = {'LOG-MUL-LOG1P'}


class LogCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(['lake', 'build', 'VeriTile.Triton.Float.LogExp',
                                 'bench.examples.LogExp.FPEquiv'], cwd=ROOT,
                                text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'Candidates.lean'
            path.write_text(source)
            return subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                  text=True, capture_output=True, timeout=180)

    def check(self, source):
        result = self.lean(source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        return result.stdout

    def test_syntax_domains_precision_and_conditional_reuse(self):
        output = self.check((ROOT / 'bench/tests/FPLogCandidates.lean').read_text())
        self.assertIn('FP assumptions used by selected_piecewise:\n  log_exp_elim\n', output)
        self.assertIn('FP assumptions used by selected_product_split:\n  log_mul_split\n', output)
        self.assertNotIn('  log_mul_guarded\n', output)
        self.assertNotIn('unresolved FP proof', output)

    def test_catalog_and_selection_match_the_experiment(self):
        output = self.check('''
import VeriTile.Triton.Float.LogExp
open VeriTile.Triton.FP.LogExp
#eval Lean.Json.compress (Lean.toJson (candidates.map Atom.ruleID))
#eval Lean.Json.compress (Lean.toJson
  ((candidates.filter fun a => a.report?.isSome).map Atom.ruleID))
''')
        catalog, selected = [json.loads(json.loads(line)) for line in output.splitlines()]
        registry = json.loads((ROOT / 'experiments/floating_point/supplement/rules.json').read_text())
        expected = {r['id'] for r in registry['rules'] if r['id'].startswith('LOG-')}
        self.assertEqual(expected - set(catalog), EXPERIMENT_ONLY)
        self.assertEqual(set(catalog), expected - EXPERIMENT_ONLY)
        self.assertEqual(len(catalog), len(expected - EXPERIMENT_ONLY))
        report = json.loads((ROOT / 'experiments/floating_point/supplement/log_report/summary.json').read_text())
        accepted = {r['rule'] for r in report['rows'] if r['format'] == 'fp32' and r['accept']}
        self.assertEqual(set(selected), accepted - EXPERIMENT_ONLY)
        self.assertEqual(accepted & EXPERIMENT_ONLY, set())
        self.assertEqual(set(selected), {'LOG-EXP-GUARDED', 'LOG-MUL-GUARDED'})

    def test_product_execution_branches_and_domains(self):
        self.check((ROOT / 'bench/tests/FPLogProduct.lean').read_text())

    def test_detailed_report_preserves_experimental_identity(self):
        output = self.check('''
import VeriTile.Triton.Float.LogExp
open VeriTile.Triton.FP.LogExp
open scoped VeriTile.Spec
specification product_split (R : Rules) :
    [Atom.log_mul_split.lhs] ≡[R] [Atom.log_mul_split.rhs] :=
  VeriTile.Triton.FP.LogExp.rewrite R .log_mul_split (by decide)
#print_spec product_split full
''')
        self.assertIn('LOG-MUL-GUARDED', output)

    def test_catalog_compiles_with_no_admitted_rules(self):
        source = (ROOT / 'VeriTile/Triton/Float/LogExp.lean').read_text()
        source = source.replace('LogAdmission.all.find? a.matches',
                                '([] : List ReportedScalarRule).find? a.matches')
        self.check(source + '''
open VeriTile.Triton.FP.LogExp
example (a : Atom) : ¬ a.Available := by cases a <;> decide
example (R : Rules) : R.assumptions = [] := rfl
''')

    def test_unadmitted_candidate_cannot_be_used_by_decide(self):
        registry = json.loads((ROOT / 'experiments/floating_point/supplement/rules.json').read_text())
        candidates = {r['id'] for r in registry['rules'] if r['id'].startswith('LOG-')}
        report = json.loads((ROOT / 'experiments/floating_point/supplement/log_report/summary.json').read_text())
        accepted = {r['rule'] for r in report['rows'] if r['format'] == 'fp32' and r['accept']}
        prefix = '''
import VeriTile.Triton.Float.LogExp
open VeriTile.Triton.FP.LogExp
open scoped VeriTile.Spec
'''
        for rule in sorted(candidates - accepted - EXPERIMENT_ONLY):
            atom = rule.lower().replace('-', '_')
            with self.subTest(atom=atom):
                result = self.lean(prefix + f'''
example (R : Rules) : [Atom.{atom}.lhs] ≡[R] [Atom.{atom}.rhs] :=
  VeriTile.Triton.FP.LogExp.rewrite R .{atom} (by decide)
''')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('decide', result.stdout)


if __name__ == '__main__':
    unittest.main()
