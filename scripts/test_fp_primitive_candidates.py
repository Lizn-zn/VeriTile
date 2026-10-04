"""Candidate libraries survive missing admissions; only selected atoms are usable."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('ScalarArithmetic', 'Reciprocal', 'Exponential', 'CountConversion', 'Maximum')


class PrimitiveCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(['lake', 'build', *(f'VeriTile.Triton.Float.{f}' for f in FAMILIES)],
                                cwd=ROOT, text=True, capture_output=True, timeout=300)
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

    def source(self, family):
        return (ROOT / f'VeriTile/Triton/Float/{family}.lean').read_text()

    def empty(self, source):
        for pool in ('ReportedAdmission.all', 'SupplementalAdmission.all', 'CountAdmission.all'):
            source = source.replace(pool, '[]')
        return source

    def test_syntax_domains_precision_and_actual_dependency(self):
        output = self.check((ROOT / 'bench/tests/FPPrimitiveCandidates.lean').read_text())
        self.assertIn('FP assumptions used by selected_maximum:\n  max_commute\n', output)
        self.assertNotIn('unresolved FP proof', output)
        self.assertNotIn('\n  max_assoc\n', output)

    def test_empty_tables_keep_candidates_and_conditional_proofs(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                kind = 'Format' if family == 'Reciprocal' else 'Atom'
                remaining = 'R.arithmetic.assumptions' if family in ('Exponential', 'CountConversion') else '[]'
                rules = 'Rules .fp32' if family == 'Reciprocal' else 'Rules'
                self.check(self.empty(self.source(family)) + f'''
open VeriTile.Triton.FP.{family}
example (a : {kind}) : ¬ a.Available := by cases a <;> decide
example (R : {rules}) : R.assumptions = {remaining} := rfl
''')

    def test_removed_rules_cannot_be_used(self):
        for family, atom in (
            ('ScalarArithmetic', 'addAssociate'), ('Reciprocal', 'fp32'),
            ('Exponential', 'exp_sub'), ('CountConversion', 'successor'), ('Maximum', 'max_assoc')
        ):
            with self.subTest(family=family):
                rules = 'Rules .fp32' if family == 'Reciprocal' else 'Rules'
                args = 'R' if family == 'Reciprocal' else f'R .{atom}'
                result = self.lean(self.empty(self.source(family)) + f'''
open VeriTile.Triton.FP.{family}
example (R : {rules}) := VeriTile.Triton.FP.{family}.rewrite {args} (by decide)
''')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('decide', result.stdout)

    def test_partial_table_enables_only_the_matching_rule(self):
        for family, pool, row, selected, absent in (
            ('ScalarArithmetic', 'ReportedAdmission.all', 'ReportedAdmission.fp32_add_commute',
             'addCommute', 'addAssociate'),
            ('Exponential', 'SupplementalAdmission.all', 'SupplementalAdmission.fp32_exp_zero',
             'exp_zero', 'exp_sub'),
            ('CountConversion', 'CountAdmission.all', 'CountAdmission.zero', 'zero', 'successor'),
            ('Maximum', 'SupplementalAdmission.all', 'SupplementalAdmission.fp32_max_commute',
             'max_commute', 'max_assoc'),
        ):
            with self.subTest(family=family):
                source = self.source(family)
                for other in ('ReportedAdmission.all', 'SupplementalAdmission.all', 'CountAdmission.all'):
                    source = source.replace(other, f'[{row}]' if other == pool else '[]')
                self.check(source + f'''
open VeriTile.Triton.FP.{family}
example : Atom.{selected}.Available := by decide
example : ¬ Atom.{absent}.Available := by decide
example : (candidates.filterMap Atom.entry?).length = 1 := rfl
''')
        source = self.source('Reciprocal').replace('SupplementalAdmission.all',
                                                  '[SupplementalAdmission.fp32_div_mul_rcp]')
        self.check(source + '''
open VeriTile.Triton.FP.Reciprocal
example : Format.fp32.Available := by decide
example : ¬ Format.fp64_fp32.Available := by decide
''')

    def test_count_domain_is_not_extended_by_a_different_report_range(self):
        source = self.source('CountConversion').replace('CountAdmission.upperExclusive', '(1024 : Nat)')
        self.check(source + '''
open VeriTile.Triton.FP.CountConversion
example : Atom.zero.Available := by decide
example : ¬ Atom.successor.Available := by decide
''')

    def test_catalog_selection_matches_published_results(self):
        rows = []
        for path in ('report', 'supplement/report', 'primitives/report'):
            rows.extend(json.loads((ROOT / f'experiments/floating_point/{path}/summary.json').read_text())['rows'])
        accepted = {r['rule'] for r in rows if r['format'] in ('fp32', 'int32_fp32') and r['accept']}
        registry = json.loads((ROOT / 'experiments/floating_point/supplement/rules.json').read_text())['rules']
        for family, prefix in (('Exponential', 'EXP-'), ('CountConversion', 'COUNT-'), ('Maximum', 'MAX-'),
                               ('ScalarArithmetic', None)):
            with self.subTest(family=family):
                output = self.check(f'''
import VeriTile.Triton.Float.{family}
open VeriTile.Triton.FP.{family}
#eval Lean.Json.compress (Lean.toJson (candidates.map Atom.ruleID))
#eval Lean.Json.compress (Lean.toJson ((candidates.filter fun a => a.report?.isSome).map Atom.ruleID))
''')
                catalog, selected = [json.loads(json.loads(line)) for line in output.splitlines()]
                if prefix:
                    self.assertEqual(set(catalog), {r['id'] for r in registry if r['id'].startswith(prefix)})
                self.assertEqual(len(catalog), len(set(catalog)))
                self.assertEqual(set(selected), accepted & set(catalog))


if __name__ == '__main__':
    unittest.main()
