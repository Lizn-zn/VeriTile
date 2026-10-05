"""Lean candidates exist before admission and preserve their exact syntax."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
# These GPU probes have no Lean fragments yet. Keep the exception explicit so
# adding experiment data cannot silently enable an unconditional product rule.
EXPERIMENT_ONLY = {'LOG-MUL-LOG1P', 'LOG-MUL-GUARDED'}


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
        self.assertIn('FP assumptions used by selected_piecewise:\n  log_exp_expm1\n', output)
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
        self.assertEqual(accepted & EXPERIMENT_ONLY, {'LOG-MUL-GUARDED'})
        self.assertEqual(set(selected), {'LOG-EXP-EXPM1'})

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
