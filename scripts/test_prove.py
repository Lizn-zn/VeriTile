"""Exercise the runner against real comparator, with a deterministic fake agent.

Requires the tools from setup-comparator.sh and a systemd user service. No model
calls are made. Run explicitly: python3 scripts/test_prove.py
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ComparatorRunnerTests(unittest.TestCase):
    def attempt(self, challenge, candidate, names=('target',), agent_exit=0):
        with tempfile.TemporaryDirectory(prefix='veritile-prove-test-') as temp:
            project = Path(temp)
            (project / 'scripts').mkdir()
            for name in ('prove.py', 'comparator_common.py'):
                shutil.copy(ROOT / 'scripts' / name, project / 'scripts' / name)
            shutil.copy(ROOT / 'lean-toolchain', project / 'lean-toolchain')
            (project / 'lakefile.toml').write_text('name = "veritiletest"\n')
            (project / 'lake-manifest.json').write_text(json.dumps({
                'version': '1.1.0', 'packagesDir': '.lake/packages', 'packages': [],
                'name': 'veritiletest', 'lakeDir': '.lake'}))
            (project / 'Target.lean').write_text(challenge)
            subprocess.run(['git', 'init', '-q', str(project)], check=True)
            binary = project / 'bin'
            binary.mkdir()
            # Simulate edits and a deliberately unreliable success report.
            (binary / 'claude').write_text(
                '#!/usr/bin/env python3\nfrom pathlib import Path\n'
                f'Path("Target.lean").write_text({candidate!r})\n'
                'print(\'{"type":"result","subtype":"success"}\')\n'
                f'raise SystemExit({agent_exit})\n')
            (binary / 'claude').chmod(0o755)
            env = {**os.environ, 'PATH': str(binary) + os.pathsep + os.environ['PATH']}
            command = ['python3', 'scripts/prove.py', 'Target.lean']
            for name in names:
                command += ['--theorem', name]
            result = subprocess.run(command, cwd=project, env=env, text=True,
                                    capture_output=True)
            reports = list(project.glob('Logs/*/result.json'))
            diagnostics = result.stdout + result.stderr
            for log in project.glob('Logs/*/comparator.txt'):
                diagnostics += log.read_text()
            if names:
                self.assertEqual(len(reports), 1, diagnostics)
                report = json.loads(reports[0].read_text())
                self.assertEqual(report['accepted'], result.returncode == 0, diagnostics)
                original = reports[0].parent / 'Challenge.lean'
                self.assertEqual(original.read_text(), challenge)
            return result.returncode, diagnostics

    def test_valid_proof_despite_agent_exit(self):
        code, log = self.attempt('theorem target : True := by sorry\n',
                                 'theorem target : True := by trivial\n', agent_exit=1)
        self.assertEqual(code, 0, log)

    def test_rejects_invalid_submissions(self):
        cases = [
            ('changed_statement', 'theorem target : False := by sorry\n',
             'theorem target : True := by trivial\n', 'theorem statement do not match'),
            ('new_axiom', 'theorem target : False := by sorry\n',
             'axiom injected : False\ntheorem target : False := injected\n', "Illegal axiom detected: 'injected'"),
            ('sorry', 'theorem target : True := by sorry\n',
             'theorem target : True := by sorry\n', "Illegal axiom detected: 'sorryAx'"),
            ('changed_definition', 'def answer : Nat := 1\ntheorem target : answer = 2 := by sorry\n',
             'def answer : Nat := 2\ntheorem target : answer = 2 := rfl\n', 'Const does not match'),
            ('missing_target', 'theorem target : True := by sorry\n',
             'theorem replacement : True := by trivial\n', 'target'),
        ]
        for name, challenge, candidate, reason in cases:
            with self.subTest(name=name):
                code, log = self.attempt(challenge, candidate)
                self.assertNotEqual(code, 0, log)
                self.assertIn('[FAIL]', log)
                self.assertIn(reason, log)

    def test_all_requested_theorems_must_pass(self):
        code, log = self.attempt(
            'theorem first : True := by sorry\ntheorem second : False := by sorry\n',
            'theorem first : True := by trivial\ntheorem second : True := by trivial\n',
            names=('first', 'second'))
        self.assertNotEqual(code, 0, log)
        self.assertIn('theorem statement do not match', log)

    def test_missing_theorem_list_fails_before_agent(self):
        code, log = self.attempt('theorem target : True := by sorry\n', '', names=())
        self.assertNotEqual(code, 0)
        self.assertIn('--theorem', log)


if __name__ == '__main__':
    unittest.main()
