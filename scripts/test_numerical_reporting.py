"""Reporting completeness, explicit missing values, and acceptance after replay."""
from copy import deepcopy
import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import report_numerics as reporting
from scripts.test_numerical_gates import fixture_bundle, profile


class ReportingTests(unittest.TestCase):
    def test_all_requested_instances_and_verified_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / 'first'
            bundle.mkdir()
            fixture_bundle(bundle)
            expected = profile()
            expected['rules'].append('MUL-COMMUTE')
            result = reporting.collect(root, expected, verify_all=True)
            self.assertEqual(result['total'], 2)
            complete, pending = result['rows']
            self.assertEqual((complete['z'], complete['U'], complete['accept']), (0, 0, True))
            self.assertTrue(complete['replayed'])
            self.assertIsNone(pending['z'])
            self.assertIsNone(pending['U'])
            self.assertIsNone(pending['accept'])
            self.assertEqual(pending['state'], 'PENDING')
            reporting.publish(root, result)
            self.assertEqual(json.loads((root / 'summary.json').read_text())['total'], 2)
            self.assertIn('pending', (root / 'summary.md').read_text())
            self.assertIn('empirical_max', (root / 'summary.csv').read_text())

    def test_provisional_results_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'first').mkdir()
            fixture_bundle(root / 'first')
            result = reporting.collect(root, profile())
            self.assertEqual(result['rows'][0]['z'], 0)
            self.assertIsNone(result['rows'][0]['accept'])
            self.assertFalse(result['rows'][0]['replayed'])

    def test_export_reads_evidence_from_bundle_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'bundles'
            report = Path(tmp) / 'report'
            (root / 'first').mkdir(parents=True)
            fixture_bundle(root / 'first')
            result = reporting.refresh(root, profile(), {}, verify_all=True, report_dir=report)
            self.assertTrue(result['complete'])
            self.assertEqual(result['accepted'], 1)
            self.assertFalse((root / 'summary.json').exists())
            self.assertEqual(json.loads((report / 'summary.json').read_text()), result)
            self.assertIn('not proof of strict floating-point equivalence',
                          (report / 'summary.md').read_text())

    def test_smoke_does_not_fill_formal_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'smoke').mkdir()
            fixture_bundle(root / 'smoke', smoke=True)
            result = reporting.collect(root, profile(), verify_all=True)
            self.assertEqual(result['rows'][0]['state'], 'PENDING')
            self.assertEqual(result['accepted'], 0)

    def test_corrupt_bundle_cannot_populate_accept(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'first').mkdir()
            entry = fixture_bundle(root / 'first')
            (entry / 'reference.0.ptx').write_text('changed')
            row = reporting.collect(root, profile(), verify_all=True)['rows'][0]
            self.assertEqual(row['state'], 'REPLAY_ERROR')
            self.assertIsNone(row['accept'])

    def test_duplicate_results_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['first', 'second']:
                (root / name).mkdir()
                fixture_bundle(root / name)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                reporting.collect(root, profile())

    def test_infinite_z_and_missing_values(self):
        self.assertEqual(reporting.maximum_z([0., 3., '+inf']), '+inf')
        self.assertIsNone(reporting.maximum_z([]))
        self.assertEqual(reporting.display_number(None), '—')
        self.assertEqual(reporting.display_number('+inf'), '∞')
        with self.assertRaises(ValueError):
            reporting.maximum_z([float('nan')])

    def test_stale_handle_retries_without_caching_a_replay_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'first').mkdir()
            fixture_bundle(root / 'first')
            replay = reporting.experiment.replay
            calls = 0

            def transient(bundle):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError(errno.ESTALE, 'Stale file handle')
                return replay(bundle)

            with patch.object(reporting.experiment, 'replay', side_effect=transient), \
                    patch.object(reporting.time, 'sleep'):
                result = reporting.refresh(root, profile(), {}, verify_all=True)
            self.assertEqual(calls, 2)
            self.assertTrue(result['rows'][0]['accept'])
            self.assertTrue(result['complete'])

    def test_nontransient_io_error_is_not_retried(self):
        with patch.object(reporting, 'collect', side_effect=PermissionError(errno.EACCES, 'denied')), \
                patch.object(reporting.time, 'sleep') as sleep:
            with self.assertRaises(PermissionError):
                reporting.refresh(Path('/unused'), profile(), {})
            sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
