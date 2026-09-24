"""Failure-injection regressions for the standalone audit gates."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


coverage = load('coverage_manifest', 'bench/check_proof_gap_manifest.py')
sheets = load('spec_sheets', 'scripts/spec_sheet.py')


class AuditGateTests(unittest.TestCase):
    def run_gate(self, script, env):
        return subprocess.run(['bash', str(ROOT / script), 'vector_addition', 'mean_reduction'],
                              cwd=ROOT, env={**os.environ, **env}, text=True,
                              capture_output=True, timeout=20)

    def test_invalid_concurrency(self):
        for script, variable in [('bench/check_ports.sh', 'CHECK_PORTS_JOBS'),
                                 ('bench/audit_trust.sh', 'AUDIT_TRUST_JOBS')]:
            for value in ['invalid', '0', '-1']:
                with self.subTest(script=script, value=value):
                    result = self.run_gate(script, {variable: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('positive integer', result.stderr)

    def test_worker_failures_and_result_accounting(self):
        for script, variable, targets in [
            ('bench/check_ports.sh', 'CHECK_PORTS_JOBS', ['vector_addition', 'mean_reduction']),
            ('bench/audit_trust.sh', 'AUDIT_TRUST_JOBS', [
                'bench/tritonbench_g/vector_addition/VectorAddition.lean',
                'bench/tritonbench_g/mean_reduction/MeanReduction.lean'])]:
            first = f'  ok    {targets[0]}\n'
            both = first + f'  ok    {targets[1]}\n'
            fixtures = [('launch-failure', '', 'exit 127', False),
                        ('killed-worker', '', 'kill -TERM $$', False),
                        ('no-results', '', 'exit 0', False),
                        ('partial-results', first, 'exit 0', False),
                        ('duplicate-results', first * 2, 'exit 0', False),
                        ('unexpected-result', first + '  ok    surprise\n', 'exit 0', False),
                        ('all-results', both, 'exit 0', True)]
            for name, output, ending, success in fixtures:
                with self.subTest(script=script, fault=name), tempfile.TemporaryDirectory() as temp:
                    launcher = Path(temp) / 'xargs'
                    launcher.write_text("#!/bin/sh\ncat >/dev/null\nprintf '%s' '" + output + "'\n" + ending + '\n')
                    launcher.chmod(0o755)
                    result = self.run_gate(script, {variable: '2', 'PATH': temp + os.pathsep + os.environ['PATH']})
                    self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)

    def test_modified_axioms_are_rejected_by_real_whitelist_function(self):
        source = (ROOT / 'scripts/check-artifact.sh').read_text()
        start = source.index('check_axioms() {')
        function = source[start:source.index('\n}\n', start) + 3]
        for declaration in ['private axiom hidden : False', 'protected axiom hidden : False',
                            'private\naxiom hidden : False', '@[simp] axiom hidden : False']:
            with self.subTest(declaration=declaration), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                (directory / 'VeriTile').mkdir()
                (directory / 'VeriTile/Fixture.lean').write_text(declaration + '\n')
                harness = ('AXIOM_WHITELIST=empty.txt\nfailures=0\n'
                           'ok() { :; }\nfail() { failures=$((failures + 1)); }\n'
                           + function + '\ncheck_axioms\nexit "$failures"\n')
                result = subprocess.run(['bash', '-c', harness], cwd=directory, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b'Fixture.lean:hidden', result.stderr)

    def test_unannotated_proof_is_not_promoted(self):
        self.assertEqual(coverage.classify('kernel_correctness', 'a checked theorem')[0], 'unreviewed')
        rows = {Path(row.file).parent.name: row for row in coverage.collect()}
        self.assertEqual(rows['int8_quantization'].coverage_level, 'precomputed_input_slice')
        for name in ['quantize_global', 'rowwise_quantization_triton']:
            self.assertEqual(rows[name].coverage_level, 'pre_rounding_slice')

    def test_lean_rejects_specs_without_discovered_kernels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'MissingKernel.lean'
            path.write_text('import VeriTile.Meta.StatementAudit\n'
                            'def expectedSpec : Nat := 1\n#auditModuleSpecs\n')
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=90)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('specifications but no ComputeKernel', result.stdout)

    def test_spec_sheet_collisions_are_port_qualified_and_preflighted(self):
        paths = ['rms_norm_triton/RmsNormTriton.lean', 'rmsnorm_triton/RmsnormTriton.lean']
        sheets.check_output_names(paths)
        self.assertNotEqual(*(sheets.sheet_name(p).casefold() for p in paths))
        with self.assertRaises(ValueError):
            sheets.check_output_names(['port/Example.lean', 'Port/example.lean'])


if __name__ == '__main__':
    unittest.main()
