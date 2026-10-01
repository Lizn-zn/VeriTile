"""Report-to-Lean admission selection and frozen-profile regression checks."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts import export_numerical_rules as exporter


class ExportTests(unittest.TestCase):
    def test_only_accepted_instances_are_generated(self):
        _, _, rows, _, _ = exporter.load_report(exporter.REPORT)
        self.assertEqual(len(rows), 30)
        content = exporter.render()
        self.assertEqual(content, exporter.OUTPUT.read_text())
        self.assertIn('def fp32_add_assoc : ReportedRule', content)
        self.assertNotIn('def bf16_fma_contract', content)
        self.assertNotIn('def fp32_cast_remove', content)
        self.assertNotIn('def fp32_cast_move', content)
        self.assertNotIn('def fp32_sqrt_rsqrt', content)
        self.assertNotIn('def fp32_bf16_widen_return', content)
        self.assertNotIn('axiom ', content)

    def test_warn_fail_domain_and_unsupported_cannot_be_promoted(self):
        for rule, fmt in [('FMA-CONTRACT', 'bf16'), ('CAST-MOVE', 'fp32'),
                          ('SQRT-RSQRT', 'fp32'), ('BF16-WIDEN-RETURN', 'fp32')]:
            with self.subTest(rule=rule), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                shutil.copytree(exporter.REPORT, target, dirs_exist_ok=True)
                path = target / 'summary.json'
                summary = json.loads(path.read_text())
                row = next(r for r in summary['rows'] if (r['rule'], r['format']) == (rule, fmt))
                row['accept'] = True
                path.write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError, 'accept disagrees'):
                    exporter.render(target)

    def test_renamed_unknown_rules_and_duplicate_rows_are_rejected(self):
        for action in ('unknown', 'duplicate', 'missing'):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                shutil.copytree(exporter.REPORT, target, dirs_exist_ok=True)
                path = target / 'summary.json'
                summary = json.loads(path.read_text())
                if action == 'unknown':
                    summary['rows'][0]['rule'] = 'SOFTMAX-ONLINE'
                elif action == 'duplicate':
                    summary['rows'].append(deepcopy(summary['rows'][0]))
                else:
                    summary['rows'].pop()
                path.write_text(json.dumps(summary))
                with self.assertRaises(ValueError):
                    exporter.render(target)

    def test_source_drift_is_rejected_and_profile_changes_change_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            shutil.copytree(exporter.REPORT, target, dirs_exist_ok=True)
            path = target / 'experiment.json'
            settings = json.loads(path.read_text())
            settings['profile']['shape'] = [32, 33]
            path.write_text(json.dumps(settings))
            self.assertNotEqual(exporter.render(), exporter.render(target))
            settings['sources']['scripts/numerical_gates.py'] = '0' * 64
            path.write_text(json.dumps(settings))
            with self.assertRaisesRegex(ValueError, 'source hashes'):
                exporter.render(target)

    def test_trust_report_is_explicit(self):
        result = subprocess.run(['python3', 'scripts/export_numerical_rules.py', '--check'],
                                cwd=exporter.ROOT, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--trust-report', result.stderr)


class LeanExampleTests(unittest.TestCase):
    def test_frozen_example_and_printed_assumption(self):
        result = subprocess.run(['lake', 'env', 'lean',
                                 'bench/examples/TritonBenchVectorAdditionFP.lean'],
                                cwd=exporter.ROOT, text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for text in ('rule ID: "ADD-COMMUTE"', 'bias gate: PASS', 'vars gate: PASS',
                     'Atom 1:', 'fp32:ADD-COMMUTE', 'Rules.add_comm:', 'axiom footprint ⊆ standard base'):
            self.assertIn(text, result.stdout)
        self.assertNotIn('Atom 2:', result.stdout)
        self.assertNotIn('sorryAx', result.stdout)
        self.assertNotIn('Nonstandard axioms:', result.stdout)
        self.assertIn('summary.json#fp32/ADD-COMMUTE', result.stdout)
        self.assertIn('4096', result.stdout)

    def test_shape_and_precision_cannot_silently_change(self):
        source = (exporter.ROOT / 'bench/examples/TritonBenchVectorAdditionFP.lean').read_text()
        # Check only the definitions and binding assertion; no proof/report noise.
        source = source.split('/-- Only this frozen accepted row')[0]
        for old, new in [('def rows : Nat := 4096', 'def rows : Nat := 32'),
                         ('ReportedAdmission.fp32_add_commute', 'ReportedAdmission.bf16_add_commute')]:
            with self.subTest(change=new), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'WrongProfile.lean'
                path.write_text(source.replace(old, new) +
                                '\nend VeriTile.Bench.Examples.TritonBenchVectorAdditionFP\n')
                result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=exporter.ROOT,
                                        text=True, capture_output=True, timeout=180)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('decide', result.stdout)


if __name__ == '__main__':
    unittest.main()
