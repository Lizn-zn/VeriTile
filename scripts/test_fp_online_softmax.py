"""Original online softmax recurrence and its scalar-derived prefix invariant."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OnlineSoftmaxFPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(
            ['lake', 'build', 'bench.examples.OnlineSoftmax.FPEquiv',
             'bench.examples.OnlineSoftmax.Correct', 'VeriTile.Meta.StatementAudit'],
            cwd=ROOT, text=True, capture_output=True, timeout=300)
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)

    def check_lean(self, source):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'OnlineSoftmaxCheck.lean'
            path.write_text(source)
            result = subprocess.run(['lake', 'env', 'lean', str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('warning:', result.stdout)
        return result.stdout

    def test_original_source_and_real_spec_independence(self):
        output = self.check_lean('''
import bench.examples.OnlineSoftmax.Contract
import bench.examples.OnlineSoftmax.Correct
open VeriTile Triton VeriTile.Bench.Examples
example (x y : RegionName) (N : Nat) :
    (OnlineSoftmax.Kernels.onlineNormalizerKernel x y N).surfaceBody =
      OnlineSoftmaxFPExecution.initialCode ++
        [.forLoop "i" N (OnlineSoftmaxFPExecution.body x N)] :=
  OnlineSoftmaxFPExecution.kernel_body x y N
example (x y : RegionName) (N : Nat) :
    (OnlineSoftmax.Kernels.onlineSoftmaxKernel x y N).surfaceBody =
      (OnlineSoftmax.Kernels.onlineNormalizerKernel x y N).surfaceBody ++
        OnlineSoftmaxFPExecution.normalizationBody x y N :=
  OnlineSoftmaxFPExecution.full_kernel_body x y N
example (x y : RegionName) (N : Nat) :
    (OnlineSoftmaxFPContract.online x y N).io.kernel =
      OnlineSoftmax.Kernels.onlineSoftmaxKernel x y N := rfl
example (N : Nat) :
    (OnlineSoftmax.batchSoftmaxIO N).kernel =
      OnlineSoftmax.Kernels.batchSoftmaxKernel "x" "y" N := rfl
''')
        self.assertEqual(output, '')
        self.check_lean('''
import bench.examples.OnlineSoftmax.Contract
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  if env.contains `VeriTile.Bench.Examples.OnlineSoftmax.online_softmax_correctness then
    throwError "FP recurrence proof imported its correctness counterpart"
''')

    def test_invariant_initialization_iteration_domains_and_unchanged_memory(self):
        self.check_lean((ROOT / 'bench/tests/FPOnlineSoftmax.lean').read_text())

    def test_public_output_spec_uses_only_admitted_scalar_atoms(self):
        output = self.check_lean('''
import bench.examples.OnlineSoftmax.FPEquiv
#print_fp_assumptions VeriTile.Bench.Examples.OnlineSoftmaxFPEquiv.online_softmax_equiv
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  if env.contains `VeriTile.Bench.Examples.OnlineSoftmax.online_softmax_correctness then
    throwError "FP specification imported its correctness counterpart"
''')
        self.assertEqual(set(output.splitlines()[1:]), {
            '  add_commute', '  add_assoc', '  mul_commute', '  mul_assoc',
            '  mul_distrib', '  cancel', '  add_zero', '  mul_one',
            '  div_mul_rcp', '  mul_rcp_cancel', '  exp_sub(libdevice.exp)'})

    def test_observation_requires_final_registers_and_preserves_source_effects(self):
        self.check_lean((ROOT / 'bench/tests/FPObservedRow.lean').read_text())


if __name__ == '__main__':
    unittest.main()
