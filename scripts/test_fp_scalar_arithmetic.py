"""Guarded scalar normalization and explicit reduction-tree regressions."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ScalarArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ['lake', 'build', 'VeriTile.Triton.Float.ScalarReduction',
             'VeriTile.Triton.Float.WelfordInit', 'bench.examples.support.WelfordExecution'], cwd=ROOT,
            text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ScalarCheck.lean'
            path.write_text(source)
            return subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                  text=True, capture_output=True, timeout=180)

    def test_domains_padding_precision_and_actual_atom_dependency(self):
        result = self.lean((ROOT / 'bench/tests/FPScalarArithmetic.lean').read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        self.assertIn('FP assumptions used by used_distribution:\n  mul_distrib\n', result.stdout)
        self.assertNotIn('unresolved FP proof', result.stdout)
        self.assertNotIn('\n  add_assoc\n', result.stdout)

    def test_welford_steps_centering_and_count_conversion_boundary(self):
        result = self.lean((ROOT / 'bench/tests/FPWelfordArithmetic.lean').read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)

    def test_wrong_relation_or_precision_cannot_replace_the_selected_row(self):
        source = (ROOT / 'VeriTile/Triton/Float/ScalarArithmetic.lean').read_text()
        for old, new in (
            ('ReportedAdmission.fp32_mul_distrib', 'ReportedAdmission.bf16_fp32_mul_distrib'),
            ('ReportedAdmission.fp32_add_assoc', 'ReportedAdmission.fp32_mul_assoc'),
            ('SupplementalAdmission.fp32_mul_rcp_cancel',
             'SupplementalAdmission.bf16_fp32_mul_rcp_cancel'),
        ):
            with self.subTest(replacement=new):
                result = self.lean(source.replace(old, new))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('decide', result.stdout)


if __name__ == '__main__':
    unittest.main()
