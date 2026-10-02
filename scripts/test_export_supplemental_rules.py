"""Trusted-report import preserves admission, domains and precision contracts."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts import export_supplemental_rules as exporter


class SupplementalExportTests(unittest.TestCase):
    def mutate(self, filename, update):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        target = Path(tmp.name)
        shutil.copytree(exporter.REPORT, target, dirs_exist_ok=True)
        path = target / filename
        content = json.loads(path.read_text())
        update(content)
        path.write_text(json.dumps(content))
        return target

    def test_exact_accepted_precision_instances_and_reproducible_export(self):
        _, _, rows, _, _ = exporter.load_report(exporter.REPORT)
        self.assertEqual(len(rows), 37)
        self.assertEqual(sum(r['format'] == 'fp64_fp64_fp32' for r in rows), 1)
        self.assertEqual({r['format'] for r in rows if r['rule'] == 'MUL-RCP-CANCEL'},
                         {'bf16_fp32', 'fp32'})
        text = exporter.render()
        self.assertEqual(text, exporter.OUTPUT.read_text())
        for name in ('fp32_log_exp', 'fp32_log_mul', 'bf16_mul_rcp_cancel'):
            self.assertNotIn(f'def {name} :', text)
        self.assertIn('def fp32_exp_sub :', text)
        self.assertIn('def fp32_mul_rcp_cancel :', text)
        self.assertNotIn('axiom ', text)

    def test_nonaccepted_results_cannot_be_promoted(self):
        for key in [('LOG-EXP', 'fp32'), ('LOG-MUL', 'fp32'),
                    ('MUL-RCP-CANCEL', 'bf16'), ('ADD-ZERO', 'fp64_fp64_fp32')]:
            def promote(report):
                row = next(r for r in report['rows'] if (r['rule'], r['format']) == key)
                row['accept'] = True
            with self.subTest(key=key):
                target = self.mutate('summary.json', promote)
                with self.assertRaisesRegex(ValueError, 'accept disagrees'):
                    exporter.render(target)

    def test_pass_labels_do_not_override_the_bias_budget(self):
        def exceed(report):
            next(r for r in report['rows'] if r['accept'])['B'] = .051
        with self.assertRaisesRegex(ValueError, 'budget'):
            exporter.render(self.mutate('summary.json', exceed))

    def test_source_profile_and_manifest_tampering_rejected(self):
        for kind in ('sources', 'shape', 'smoke', 'entries'):
            def alter(settings):
                if kind == 'sources':
                    settings['sources']['scripts/numerical_gates.py'] = '0' * 64
                elif kind == 'shape':
                    settings['profile']['shape'] = [32, 33]
                elif kind == 'smoke':
                    settings['smoke'] = True
                else:
                    settings['entries'].pop()
            with self.subTest(kind=kind):
                target = self.mutate('experiment.json', alter)
                with self.assertRaises(ValueError):
                    exporter.render(target)

    def test_incomplete_unknown_duplicate_or_missing_rows_rejected(self):
        for kind in ('unknown', 'duplicate', 'missing', 'replay', 'replicates'):
            def alter(summary):
                row = next(r for r in summary['rows'] if r['accept'])
                if kind == 'unknown':
                    row['rule'] = 'SOFTMAX-ONLINE'
                elif kind == 'duplicate':
                    summary['rows'].append(deepcopy(row))
                elif kind == 'missing':
                    summary['rows'].pop()
                elif kind == 'replay':
                    row['replayed'] = False
                else:
                    row['replicates'] = 1
            with self.subTest(kind=kind):
                target = self.mutate('summary.json', alter)
                with self.assertRaises(ValueError):
                    exporter.render(target)

    def test_scalar_operand_domains_are_retained(self):
        self.assertEqual(exporter.domain('DIV-MUL-RCP'),
                         [('a', 'finite'), ('b', 'finite'), ('b', 'nonzero')])
        self.assertEqual(exporter.domain('MUL-RCP-CANCEL'), [('a', 'finite'), ('a', 'nonzero')])
        self.assertEqual(exporter.domain('EXP-NEG-INF-SUB'), [('a', 'finite')])

    def test_trust_report_is_explicit(self):
        result = subprocess.run(['python3', 'scripts/export_supplemental_rules.py', '--check'],
                                cwd=exporter.ROOT, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--trust-report', result.stderr)


if __name__ == '__main__':
    unittest.main()
