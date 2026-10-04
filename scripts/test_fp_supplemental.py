"""Guarded scalar assumptions and complete reciprocal-softmax execution proofs."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ('SoftmaxReciprocal', 'FloatDTypeSoftmax')


class FPSupplementalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(['lake', 'build'] +
                                [f'bench.examples.{n}.FPEquiv' for n in EXAMPLES],
                                cwd=ROOT, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'Supplement.lean'
            path.write_text(source)
            return subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                  text=True, capture_output=True, timeout=180)

    def test_domain_precision_casts_and_independent_proof_boundaries(self):
        result = self.lean((ROOT / 'bench/tests/FPSupplementalAdmission.lean').read_text())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)

    def test_both_proofs_print_only_the_used_scalar_atom(self):
        for name in EXAMPLES:
            with self.subTest(name=name):
                result = self.lean((ROOT / f'bench/examples/{name}/FPEquiv.lean').read_text())
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stdout,
                                 'FP assumptions used by softmax_reciprocal_equiv:\n'
                                 '  div_mul_rcp\n')

    def test_an_opaque_guarded_equivalence_is_not_an_atomic_certificate(self):
        result = self.lean('''
import bench.examples.SoftmaxReciprocal.FPEquiv
open VeriTile Triton
open VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
open VeriTile.Bench.Examples.SoftmaxReciprocalFPEquiv
open scoped VeriTile.Spec
specification opaque_equiv (B : Nat) (R : Rules)
    (h : FP.Guarded.Equivalent R.assumptions (originalIO B) (reciprocalIO B)) :
    originalIO B ≡[R] reciprocalIO B :=
  Spec.FloatingPoint.ofNumerical (lhs := originalIO B) (rhs := reciprocalIO B)
    (structural := fun _ _ => False) rfl rfl h
#print_fp_assumptions opaque_equiv
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('unresolved FP proof', result.stdout)
        self.assertNotIn('  div_mul_rcp\n', result.stdout)


if __name__ == '__main__':
    unittest.main()
