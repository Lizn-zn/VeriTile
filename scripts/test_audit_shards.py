"""Exercise aggregate audit modes and the real corpus-shard launcher."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AuditAggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        ports = self.root / 'bench/tritonbench_g'
        ports.mkdir(parents=True)
        (self.root / 'scripts').mkdir()
        for name in ['audit_tritonbench_g.sh', 'audit_source.py']:
            shutil.copyfile(ROOT / 'bench' / name, self.root / 'bench' / name)
        for name in ['proof_blockers.md', 'completion_audit.md']:
            (ports / name).touch()
        # Empty source fixtures exercise the actual structural rule dispatch.
        # Replace the manifest and proof executors to inject independent failures.
        for path, label, failure in [
            ('bench/check_proof_gap_manifest.py', 'structural', 'FAIL_STATIC'),
            ('scripts/check_comparator.py', 'library', 'FAIL_LIBRARY')]:
            (self.root / path).write_text(
                'import os\nfrom pathlib import Path\n'
                f'with Path("calls").open("a") as log: log.write("{label}\\n")\n'
                f'raise SystemExit(int(os.environ.get("{failure}", "0")))\n')
        trust = self.root / 'bench/audit_trust.sh'
        trust.write_text('#!/bin/bash\nprintf "corpus\\n" >> calls\nexit "${FAIL_CORPUS:-0}"\n')
        trust.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('AUDIT_TRUST_', 'FAIL_'))}

    def run_aggregate(self, args=(), env=None):
        calls = self.root / 'calls'
        calls.unlink(missing_ok=True)
        result = subprocess.run(['bash', 'bench/audit_tritonbench_g.sh', *args],
                                cwd=self.root, env={**self.env, **(env or {})},
                                text=True, capture_output=True, timeout=20)
        return result, calls.read_text().splitlines() if calls.exists() else []

    def test_default_still_runs_all_gates(self):
        result, calls = self.run_aggregate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ['structural', 'library', 'corpus'])
        self.assertIn('TritonBench-G audit gates passed', result.stdout)

    def test_global_mode_does_not_run_or_claim_corpus_checks(self):
        result, calls = self.run_aggregate(['--global-only'])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ['structural', 'library'])
        self.assertIn('global gates passed; corpus audit is separate', result.stdout)
        self.assertNotIn('TritonBench-G audit gates passed', result.stdout)

    def test_global_failures_fail_both_modes(self):
        for args in [(), ('--global-only',)]:
            for failure in ['FAIL_STATIC', 'FAIL_LIBRARY']:
                with self.subTest(args=args, failure=failure):
                    result, calls = self.run_aggregate(args, {failure: '1'})
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotIn('gates passed', result.stdout)
                    self.assertIn('structural', calls)
                    self.assertIn('library', calls)

    def test_corpus_failure_still_fails_default_mode(self):
        result, calls = self.run_aggregate(env={'FAIL_CORPUS': '1'})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, ['structural', 'library', 'corpus'])
        self.assertNotIn('gates passed', result.stdout)

    def test_invalid_arguments_fail_before_checks(self):
        for args in [['--skip-library'], ['alpha'], ['--global-only', 'extra']]:
            with self.subTest(args=args):
                result, calls = self.run_aggregate(args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(calls, [])

    def test_global_mode_rejects_shard_configuration(self):
        for env in [{'AUDIT_TRUST_SHARD_COUNT': '4'},
                    {'AUDIT_TRUST_SHARD_INDEX': '0'},
                    {'AUDIT_TRUST_SHARD_COUNT': ''},
                    {'AUDIT_TRUST_SHARD_COUNT': '4', 'AUDIT_TRUST_SHARD_INDEX': '0'}]:
            with self.subTest(env=env):
                result, calls = self.run_aggregate(['--global-only'], env)
                self.assertEqual(result.returncode, 2)
                self.assertIn('cannot be combined', result.stderr)
                self.assertEqual(calls, [])


class AuditShardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'bench').mkdir()
        (self.root / 'scripts').mkdir()
        for name in ['audit_trust.sh', 'check_worker_results.py', 'select_audit_shard.py']:
            shutil.copyfile(ROOT / 'bench' / name, self.root / 'bench' / name)
        self.targets = [f'bench/tritonbench_g/{name}/Port.lean' for name in ['alpha', 'beta', 'gamma']]
        self.targets += ['bench/examples/One.lean', 'bench/examples/Two.lean',
                         'bench/tests/One.lean', 'bench/tests/Two.lean']
        for target in self.targets:
            path = self.root / target
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        # Substitute only proof execution; target selection, xargs dispatch,
        # result accounting and failure propagation use the production scripts.
        (self.root / 'scripts/check_comparator.py').write_text('''
import argparse
import os
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--file', required=True)
p.add_argument('--trust', action='store_true', required=True)
p.add_argument('--workspace', required=True)
a = p.parse_args()
with Path('calls').open('a') as log:
    log.write(a.file + '\\n')
print('Comparator: scheduling fixture')
raise SystemExit(1 if a.file == os.environ.get('FAIL_TARGET') else 0)
''')
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith('AUDIT_TRUST_')}
        self.env['AUDIT_TRUST_JOBS'] = '2'

    def run_audit(self, env=None, args=()):
        calls = self.root / 'calls'
        calls.unlink(missing_ok=True)
        result = subprocess.run(['bash', 'bench/audit_trust.sh', *args],
                                cwd=self.root, env={**self.env, **(env or {})},
                                text=True, capture_output=True, timeout=20)
        actual = calls.read_text().splitlines() if calls.exists() else []
        return result, actual

    def test_shards_cover_every_target_exactly_once(self):
        combined = []
        for index in range(4):
            result, calls = self.run_audit({'AUDIT_TRUST_SHARD_COUNT': '4',
                                            'AUDIT_TRUST_SHARD_INDEX': str(index)})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(calls)
            self.assertIn(f'bench corpus shard {index}/4', result.stdout)
            combined.extend(calls)
        self.assertCountEqual(combined, self.targets)
        self.assertEqual(len(combined), len(set(combined)))

    def test_failure_fails_its_shard(self):
        outcomes = []
        for index in range(4):
            result, calls = self.run_audit({'AUDIT_TRUST_SHARD_COUNT': '4',
                                            'AUDIT_TRUST_SHARD_INDEX': str(index),
                                            'FAIL_TARGET': self.targets[0]})
            outcomes.append(result.returncode)
            self.assertEqual(result.returncode != 0, self.targets[0] in calls,
                             result.stdout + result.stderr)
        self.assertEqual(sum(code != 0 for code in outcomes), 1)

    def test_invalid_shards_never_launch_proofs(self):
        for count, index in [('0', '0'), ('4', '-1'), ('4', '4'), ('8', '0'),
                             ('', '0'), ('4', ''), (None, '0'), ('4', None),
                             ('invalid', '0'), ('4', '1.0'), ('4', '00'),
                             ('9' * 100, '0')]:
            with self.subTest(count=count, index=index):
                env = {key: value for key, value in [
                    ('AUDIT_TRUST_SHARD_COUNT', count), ('AUDIT_TRUST_SHARD_INDEX', index)]
                    if value is not None}
                result, calls = self.run_audit(env)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls, [])

    def test_named_targets_reject_sharding(self):
        result, calls = self.run_audit({'AUDIT_TRUST_SHARD_COUNT': '4',
                                        'AUDIT_TRUST_SHARD_INDEX': '0'}, ['alpha'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cannot be combined', result.stderr)
        self.assertEqual(calls, [])

    def test_unsharded_and_named_modes_still_work(self):
        for args, expected in [((), self.targets), (['beta'], [self.targets[1]])]:
            with self.subTest(args=args):
                result, calls = self.run_audit(args=args)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertCountEqual(calls, expected)

    def test_missing_shard_results_fail_closed(self):
        launcher = self.root / 'xargs'
        launcher.write_text('#!/bin/sh\ncat >/dev/null\nexit 0\n')
        launcher.chmod(0o755)
        result, calls = self.run_audit({'AUDIT_TRUST_SHARD_COUNT': '4',
                                        'AUDIT_TRUST_SHARD_INDEX': '0',
                                        'PATH': str(self.root) + os.pathsep + self.env['PATH']})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('incomplete worker results', result.stderr)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
