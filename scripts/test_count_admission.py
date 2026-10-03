"""Reject unsupported count scopes and contradictory published admissions."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from scripts import export_count_rules as exporter


class CountAdmissionTests(unittest.TestCase):
    def fixture(self):
        return {name: json.loads((exporter.REPORT / name).read_text()) for name in
                ('experiment.json', 'summary.json', 'admission.json')}

    def check_rejected(self, files):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            for name, value in files.items():
                (directory / name).write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                exporter.load_report(directory)

    def test_current_report_generates_two_bounded_atoms_without_intrinsic_exp(self):
        output = exporter.render()
        self.assertEqual(exporter.OUTPUT.read_text(), output)
        self.assertIn('def upperExclusive : Nat := 16777216', output)
        self.assertIn('def all : List ReportedRule := [zero, successor]', output)
        self.assertNotIn('axiom ', output)
        self.assertNotIn('def exp', output)

    def test_frozen_pr12_source_identity_and_version_are_retained(self):
        files = self.fixture()
        manifest = files['experiment.json']['bundles']['counts']
        self.assertEqual(exporter.runner.sha(exporter.canonical(manifest['sources'])),
                         exporter.PR12_SOURCE_SNAPSHOT)
        self.assertEqual(manifest['bundle_version'], 'scalar-supplement-7')
        self.assertNotEqual(manifest['sources'], exporter.runner.source_hashes())
        manifest['bundle_version'] = exporter.runner.BUNDLE_VERSION
        self.check_rejected(files)

    def test_invalid_range_precision_smoke_and_sources_are_rejected(self):
        for case in ('range', 'negative', 'precision', 'smoke', 'sources', 'manifest'):
            files = self.fixture()
            m = files['experiment.json']['bundles']['counts']
            if case == 'range':
                m['profile']['distribution']['high'] = 2**24 + 1
            elif case == 'negative':
                m['profile']['distribution']['low'] = -1
            elif case == 'precision':
                m['profile']['formats'][0]['input'] = 'fp32'
            elif case == 'smoke':
                m['smoke'] = True
            elif case == 'sources':
                m['sources']['unknown.py'] = '0' * 64
            else:
                m['backend']['kind'] = 'changed'
            with self.subTest(case=case):
                self.check_rejected(files)

    def test_report_failures_missing_rows_and_inconsistent_statistics_are_rejected(self):
        for case in ('warn', 'unreplayed', 'missing', 'duplicate', 'bias', 'accepted', 'statistics'):
            files = self.fixture()
            rows = files['summary.json']['rows']
            row = next(r for r in rows if r['rule'] == 'COUNT-SUCCESSOR')
            if case == 'warn':
                row['vars'] = 'WARN'
            elif case == 'unreplayed':
                row['replayed'] = False
            elif case == 'missing':
                rows.remove(row)
            elif case == 'duplicate':
                rows.append(deepcopy(row))
            elif case == 'bias':
                row['B'] = 1.0
            elif case == 'accepted':
                files['admission.json']['counts']['accepted'].pop()
            else:
                files['admission.json']['counts']['accepted'][0]['statistics']['bias']['status'] = 'FAIL'
            with self.subTest(case=case):
                self.check_rejected(files)

    def test_rekeying_a_changed_domain_does_not_authorize_it(self):
        for field in ('domain', 'sampling', 'precision'):
            files = self.fixture()
            entries = files['admission.json']['counts']['accepted']
            e = next(e for e in entries if e['rule_id'] == 'COUNT-SUCCESSOR')
            cfg = e['config']
            if field == 'domain':
                cfg['relation']['domain'] = 'all natural numbers'
                e['relation'] = deepcopy(cfg['relation'])
            elif field == 'sampling':
                cfg['probe']['roles']['a']['high'] += 1
            else:
                cfg['numerics']['input_formats']['a'] = 'fp32'
            e['instance_key'] = exporter.runner.supplemental.instance_key(cfg)
            record = next(r for r in files['admission.json']['counts']['rows']
                          if r['rule_id'] == 'COUNT-SUCCESSOR')
            record.update({k: deepcopy(e[k]) for k in record})
            with self.subTest(field=field):
                self.check_rejected(files)


if __name__ == '__main__':
    unittest.main()
