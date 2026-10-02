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
        self.assertIn('def bf16_add_assoc : ReportedRule', content)
        self.assertNotIn('def fp32_add_assoc', content)
        self.assertNotIn('def bf16_fp32_add_assoc', content)
        self.assertIn('def bf16_fma_contract', content)
        self.assertNotIn('def fp32_cast_remove', content)
        for name in ('fp32_cast_move', 'fp32_cancel', 'fp32_fma_contract'):
            self.assertNotIn(f'def {name}', content)
        self.assertIn('def bf16_cast_remove', content)
        self.assertNotIn('def fp32_sqrt_rsqrt', content)
        self.assertNotIn('def fp32_bf16_widen_return', content)
        self.assertNotIn('axiom ', content)

    def test_warn_fail_domain_and_unsupported_cannot_be_promoted(self):
        for rule, fmt in [('CAST-REMOVE', 'fp32'), ('ADD-ASSOC', 'fp32'),
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

    def test_pass_labels_cannot_hide_an_exceeded_budget(self):
        for field, value in [('B', .051), ('tau', 1.0), ('U', 3.0)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                target = Path(tmp)
                shutil.copytree(exporter.REPORT, target, dirs_exist_ok=True)
                path = target / 'summary.json'
                report = json.loads(path.read_text())
                next(r for r in report['rows'] if r['accept'])[field] = value
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'budget'):
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
                                 'bench/examples/TritonBenchVectorAdditionFPEquiv.lean'],
                                cwd=exporter.ROOT, text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout,
                         'FP assumptions used by vector_addition_equiv:\n'
                         '  add_commute\n')

    def test_full_example_keeps_the_audit_details(self):
        source = (exporter.ROOT / 'bench/examples/TritonBenchVectorAdditionFPEquiv.lean').read_text()
        source = source.replace('#print_fp_assumptions vector_addition_equiv',
                                '#print_spec vector_addition_equiv full')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'FullSpec.lean'
            path.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=exporter.ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for text in ('rule ID: "ADD-COMMUTE"', 'bias gate: PASS', 'vars gate: PASS',
                     'Atom 1:', 'fp32:ADD-COMMUTE', 'Rules.add_comm:', 'EvidenceValidated',
                     'summary.json#fp32/ADD-COMMUTE', 'Transitive axioms:',
                     'Project dependencies:', 'Trusted library boundary:'):
            self.assertIn(text, result.stdout)

    def test_proof_is_parameterized_independently_of_experiment_dimensions(self):
        source = (exporter.ROOT / 'bench/examples/TritonBenchVectorAdditionFPEquiv.lean').read_text()
        source += '''
open VeriTile.Bench.Examples.TritonBenchVectorAdditionFPEquiv
open scoped VeriTile.Spec

example (n block : Nat) (R : Rules block) :
    originalKernel n block ≡[R] optimizedKernel n block :=
  vector_addition_equiv n block R

example (R : Rules 64) : originalKernel 1001 64 ≡[R] optimizedKernel 1001 64 :=
  vector_addition_equiv 1001 64 R

example (R : Rules 256) : originalKernel 8192 256 ≡[R] optimizedKernel 8192 256 :=
  vector_addition_equiv 8192 256 R

open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect.originalKernel then
    throwError "FP equivalence must not import the correctness example"
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'GenericDimensions.lean'
            path.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=exporter.ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_real_correctness_spec_is_independent_of_fp_assumptions(self):
        source = '''import bench.examples.TritonBenchVectorAdditionCorrect
open VeriTile
open VeriTile.Bench.Examples.TritonBenchVectorAdditionCorrect
open scoped VeriTile.Triton.MaskedKernelIO₂

example (n block : Nat) :
    Spec.Real (addIO n block ⊨ fun xs ys i => xs i + ys i) :=
  vector_addition_correct n block

example (n block : Nat) :
    (originalKernel n block).toAlgorithm? =
      (VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel
        "x" "y" "output" n block).toAlgorithm? := real_projection n block

open Lean Elab Command in
run_cmd do
  let deps ← liftCoreM <| VeriTile.Meta.specDependencies (← getEnv) [``vector_addition_correct]
  if deps.project.contains ``Spec.EvidenceValidated || deps.project.contains ``Spec.Derivation then
    throwError "Real correctness must not depend on FP assumptions"
#axiomsClean vector_addition_correct
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'RealCorrectness.lean'
            path.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=exporter.ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('axiom footprint ⊆ standard base', result.stdout)

    def test_atom_selection_preserves_operation_precision(self):
        source = (exporter.ROOT / 'bench/examples/TritonBenchVectorAdditionFPEquiv.lean').read_text()
        # Check only the definitions and binding assertion; no proof/report noise.
        source = source.split('/-- Only this frozen accepted row')[0]
        for old, new in [('ReportedAdmission.fp32_add_commute', 'ReportedAdmission.bf16_add_commute'),
                         ('ReportedAdmission.fp32_add_commute', 'ReportedAdmission.fp32_mul_commute')]:
            with self.subTest(change=new), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'WrongProfile.lean'
                path.write_text(source.replace(old, new) +
                                '\nend VeriTile.Bench.Examples.TritonBenchVectorAdditionFPEquiv\n')
                result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=exporter.ROOT,
                                        text=True, capture_output=True, timeout=180)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('decide', result.stdout)


if __name__ == '__main__':
    unittest.main()
