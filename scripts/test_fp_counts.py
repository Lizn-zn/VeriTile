"""Bounded count admission completes the original Welford and LayerNorm pairs."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FPCountTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(['lake', 'build', 'bench.examples.WelfordFPEquiv',
                                 'bench.examples.FusedLayerNormFPEquiv'],
                                cwd=ROOT, text=True, capture_output=True, timeout=600)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check(self, source):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'CountCheck.lean'
            p.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(p)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        return result.stdout

    def test_count_boundaries_and_independence_from_correct(self):
        self.check((ROOT / 'bench/tests/FPCountConversion.lean').read_text())

    def test_original_public_transforms_and_used_atoms_at_default_print_settings(self):
        output = self.check('''
import bench.examples.WelfordFPEquiv
import bench.examples.FusedLayerNormFPEquiv
#print_fp_assumptions VeriTile.Bench.Examples.WelfordFPEquiv.welford_equiv
#print_fp_assumptions VeriTile.Bench.Examples.FusedLayerNormFPEquiv.layernorm_equiv
''')
        expected = {'count_zero', 'count_successor', 'add_commute', 'add_zero', 'cancel',
                    'div_one', 'add_assoc', 'mul_distrib', 'mul_assoc', 'mul_rcp_cancel',
                    'mul_one', 'mul_commute', 'div_mul_rcp'}
        sections = output.split('FP assumptions used by ')[1:]
        self.assertEqual(len(sections), 2)
        for section in sections:
            lines = section.splitlines()[1:]
            self.assertEqual({s.strip() for s in lines}, expected)
            self.assertEqual(len(lines), len(expected))

    def test_upper_bound_and_layernorm_empty_case_are_public(self):
        self.check('''
import bench.examples.WelfordFPEquiv
import bench.examples.FusedLayerNormFPEquiv
open VeriTile Triton FP.CountConversion
open VeriTile.Bench.Examples
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan)
open scoped VeriTile.Spec
example (stride : Nat) (empty : ReductionPlan 0) (R : Rules) :
    WelfordFPContract.online "x" "mean" "variance" limit stride empty ≡[R]
      WelfordFPContract.twopass "x" "mean" "variance" limit stride empty :=
  WelfordFPEquiv.welford_equiv limit stride (by decide) le_rfl empty R
example (stride : Nat) (ε : ℝ) (empty : ReductionPlan 0) (R : Rules) :
    LayerNormFPContract.fused "x" "gamma" "beta" "y" 0 stride ε empty ≡[R]
      LayerNormFPContract.twopass "x" "gamma" "beta" "y" 0 stride ε empty :=
  FusedLayerNormFPEquiv.layernorm_equiv 0 stride (by decide) ε empty R
''')


if __name__ == '__main__':
    unittest.main()
