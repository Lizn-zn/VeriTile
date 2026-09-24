"""Integration tests for routine gates using the real official comparator."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ComparatorGateTests(unittest.TestCase):
    def project(self, root, source):
        (root / 'scripts').mkdir()
        (root / 'bench/tritonbench_g/fixture').mkdir(parents=True)
        for name in ('check_comparator.py', 'comparator_common.py', 'check-artifact.sh', 'source_axioms.py'):
            shutil.copyfile(ROOT / 'scripts' / name, root / 'scripts' / name)
        for name in ('audit_trust_prep.py', 'check_ports.sh', 'audit_trust.sh', 'check_worker_results.py'):
            shutil.copyfile(ROOT / 'bench' / name, root / 'bench' / name)
        # The trust-path fixture has a real Lean command boundary without
        # importing this project's Mathlib-heavy library into tiny regressions.
        (root / 'VeriTile/Meta').mkdir(parents=True)
        (root / 'VeriTile/Meta/StatementAudit.lean').write_text(
            'import Lean\nopen Lean Elab Command\n'
            'elab "#auditModuleAxioms" : command => pure ()\n'
            'elab "#auditModuleSpecs" : command => pure ()\n')
        (root / 'bench/tritonbench_g/fixture/Fixture.lean').write_text(source)
        shutil.copyfile(ROOT / 'lean-toolchain', root / 'lean-toolchain')
        (root / 'lakefile.toml').write_text(
            'name = "comparatorgates"\n[[lean_lib]]\nname = "VeriTile"\nglobs = ["VeriTile.+"]\n'
            '[[lean_lib]]\nname = "VeriTileFull"\nglobs = []\n')
        (root / 'lake-manifest.json').write_text(json.dumps({
            'version': '1.1.0', 'packagesDir': '.lake/packages', 'packages': [],
            'name': 'comparatorgates', 'lakeDir': '.lake'}))
        subprocess.run(['git', 'init', '-q', str(root)], check=True)

    def run_check(self, source, script=None, library=False, env=None):
        with tempfile.TemporaryDirectory(prefix='veritile-check-regression-') as temp:
            root = Path(temp)
            self.project(root, source)
            if library:
                (root / 'VeriTile/Fixture.lean').write_text(source)
                (root / 'scripts/kernel-manifest.tsv').write_text(
                    'fixture\tVeriTile/Fixture.lean\ttarget\tcorrect\tproven\tinternal\t-\t-\tFixture\tRegression\n')
                # Keep the artifact's unrelated documentation checks empty;
                # compile, inventory, and comparator remain real executions.
                (root / 'scripts/artifact-doc-terms.tsv').write_text('')
                (root / 'scripts/artifact-axiom-whitelist.txt').write_text(
                    'VeriTile/Fixture.lean:injected\n' if 'axiom injected' in source else '')
                (root / 'site/scripts').mkdir(parents=True)
                (root / 'site/scripts/check-doc-api.py').write_text('pass\n')
                command = ['bash', 'scripts/check-artifact.sh']
            elif script:
                command = ['bash', script, 'fixture']
            else:
                command = ['python3', 'scripts/check_comparator.py', '--file',
                           'bench/tritonbench_g/fixture/Fixture.lean']
            result = subprocess.run(command, cwd=root, text=True, capture_output=True,
                                    timeout=120, env={**os.environ, **(env or {})})
            reports = [json.loads(p.read_text()) for p in root.glob('Logs/*/*/result.json')]
            inventories = [json.loads(p.read_text()) for p in root.glob('Logs/*/*/targets.json')]
            return result, reports, inventories

    def test_private_quoted_and_macro_generated_targets(self):
        result, reports, inventories = self.run_check(
            'private theorem «a.b» : True := True.intro\n'
            'macro "fixtureTheorem" : command => `(theorem generated : True := True.intro)\n'
            'fixtureTheorem\n', script='bench/check_ports.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0]['accepted'])
        self.assertGreaterEqual(reports[0]['theorem_count'], 2)
        self.assertTrue(any('a.b' in name for name in inventories[0]['theorems']))

    def test_compiling_unapproved_proofs_fail_in_every_gate(self):
        for script in ('bench/check_ports.sh', 'bench/audit_trust.sh', None):
            with self.subTest(script=script):
                # This compiles without warnings. It is deliberately not named
                # *_correct: comparator must cover more than the headline net.
                result, reports, _ = self.run_check(
                    'axiom injected : False\ntheorem target : False := injected\n',
                    script=script, library=script is None)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(reports), 1, result.stdout + result.stderr)
                self.assertFalse(reports[0]['accepted'])
                self.assertIn('Illegal axiom detected', result.stderr)

    def test_sorry_rejected(self):
        result, reports, _ = self.run_check('theorem target : True := by sorry\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('sorryAx', result.stderr)
        self.assertFalse(reports[0]['accepted'])

    def test_library_manifest_passes(self):
        result, reports, _ = self.run_check('theorem target : True := True.intro\n', library=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(reports[0]['theorem_count'], 1)

    def test_definition_only_fixture_reports_zero_targets(self):
        result, reports, inventories = self.run_check('def value : Nat := 42\n', script='bench/audit_trust.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(reports[0]['theorem_count'], 0)
        self.assertEqual(inventories[0]['aliases'], ['veritile_comparator_empty_module'])

    def test_missing_comparator_is_a_failure(self):
        result, reports, _ = self.run_check('theorem target : True := True.intro\n',
                                            env={'COMPARATOR_BIN': '/missing/comparator'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Official comparator is required', result.stderr)
        self.assertEqual(reports, [])


if __name__ == '__main__':
    unittest.main()
