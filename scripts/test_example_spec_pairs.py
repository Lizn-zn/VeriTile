"""Cross-check the independently written real and FP example kernels."""
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    ("VectorAdd", "add_kernel_equiv", "VeriTile.Bench.Examples.VectorAdd.addIO", "add_commute"),
    ("FlatVectorAdd", "add_kernel_masked_equiv", "VeriTile.Bench.Examples.FlatVectorAdd.addMaskedIO", "add_commute"),
    ("FloatDTypeAdd", "float_add_equiv", "VeriTile.Bench.Examples.FloatDTypeAddCorrect.floatAddIO", "add_commute"),
    ("HyperConnectionsDepth", "mhc_depth_equiv", "VeriTile.Bench.Examples.HyperConnectionsDepth.mhcDepthIO", "add_commute"),
    ("HyperConnectionsWidth", "mhc_width_equiv", "VeriTile.Bench.Examples.HyperConnectionsWidth.mhcWidthIO", "mul_commute"),
    ("AdamUpdateGridLaunch", "adam_update_equiv", "VeriTile.Bench.Examples.AdamUpdateGridLaunch.adamIO", "add_commute"),
)

# The old files remain until the original FP transformation is derived. Check
# that the new real proofs still describe exactly those source kernels.
REDUCTIONS = (
    ("SoftmaxStableCorrect", "SoftmaxStableEquiv", "Softmax",
     (("naiveSoftmaxKernel", "naiveIO", "naive_softmax_correct"),
      ("stableSoftmaxKernel", "stableIO", "stable_softmax_correct"))),
    ("StableLogSumExpCorrect", "StableLogSumExpEquiv", "LogSumExp",
     (("directLSEKernel", "directIO", "direct_logsumexp_correct"),
      ("stableLSEKernel", "stableIO", "stable_logsumexp_correct"))),
    ("SoftmaxReciprocalCorrect", "SoftmaxReciprocalEquiv", "SoftmaxReciprocal",
     (("stableSoftmaxKernel", "divIO", "softmax_div_correct"),
      ("softmaxRecipKernel", "recipIO", "softmax_reciprocal_correct"))),
    ("FloatDTypeSoftmaxCorrect", "FloatDTypeEquiv", "FloatDTypeEquiv",
     (("floatStableSoftmaxKernel", "divIO", "float_softmax_div_correct"),
      ("floatSoftmaxRecipKernel", "recipIO", "float_softmax_recip_correct"))),
)


class ExamplePairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        targets = [f"bench.examples.{name}{suffix}"
                   for name, _, _, _ in PAIRS for suffix in ("Correct", "FPEquiv")]
        targets += [f"bench.examples.{module}"
                    for new, old, _, _ in REDUCTIONS for module in (new, old)]
        targets += [f"bench.examples.{name}{suffix}"
                    for name in ("FusedSiLU", "FusedSwiglu") for suffix in ("Correct", "Equiv")]
        build = subprocess.run(["lake", "build", *targets], cwd=ROOT,
                               text=True, capture_output=True, timeout=300)
        if build.returncode:
            raise AssertionError(build.stdout + build.stderr)

    def lean(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "PairCheck.lean"
            path.write_text(source)
            result = subprocess.run(["lake", "env", "lean", str(path)], cwd=ROOT,
                                    text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_same_original_implementation_in_both_files(self):
        """A correct file for one implementation cannot certify a different FP input."""
        imports = "\n".join(f"import bench.examples.{n}{s}"
                            for n, _, _, _ in PAIRS for s in ("Correct", "FPEquiv"))
        self.lean(imports + '''
open VeriTile.Bench.Examples

example (B : Nat) :
    (VectorAddFPEquiv.originalKernel B).toAlgorithm? =
      (VectorAdd.addKernel "x" "y" "out" B).toAlgorithm? := rfl

example (n B : Nat) :
    (FlatVectorAddFPEquiv.originalKernel n B).toAlgorithm? =
      (FlatVectorAdd.addKernelMasked "x" "y" "out" B n).toAlgorithm? := rfl

example (B : Nat) :
    (FloatDTypeAddFPEquiv.originalKernel B).toAlgorithm? =
      (FloatDTypeAddCorrect.floatAddKernel "x" "y" "out" B).toAlgorithm? := rfl

example (tau : Real) :
    (HyperConnectionsDepthFPEquiv.originalKernel tau).toAlgorithm? =
      (HyperConnectionsDepth.mhcDepthConnectionKernel
        "res_mix" "branch_out" "h_post" "out" 1 1 1 0 tau).toAlgorithm? := rfl

example (tau : Real) :
    (HyperConnectionsWidthFPEquiv.originalKernel tau).toAlgorithm? =
      (HyperConnectionsWidth.mhcWidthConnectionKernel
        "res" "h_res" "h_pre" "res_mix" "branch_in" 1 1 1 0 tau).toAlgorithm? := rfl

example (lr wd beta1 beta2 : Real) (n B : Nat) :
    (AdamUpdateGridLaunchFPEquiv.originalKernel lr wd beta1 beta2 n B).toAlgorithm? =
      (AdamUpdateGridLaunch.update_fn_kernel
        "p" "grad" "exp_avg" lr wd beta1 beta2 n B).toAlgorithm? := rfl
''')

    def test_fp_files_are_independent_and_print_only_used_atoms(self):
        for name, headline, correct_io, atom in PAIRS:
            with self.subTest(case=name):
                source = (ROOT / f"bench/examples/{name}FPEquiv.lean").read_text()
                source += f'''
open Lean Elab Command in
run_cmd do
  if (← getEnv).contains `{correct_io} then
    throwError "FP example imported the correctness implementation"
'''
                self.assertEqual(self.lean(source),
                                 f"FP assumptions used by {headline}:\n  {atom}\n")

    def test_reduction_correctness_preserves_sources_and_states_the_formula(self):
        imports = "\n".join(f"import bench.examples.{module}"
                            for new, old, _, _ in REDUCTIONS for module in (new, old))
        source = imports + '''
open VeriTile.Bench.Examples
open scoped VeriTile.Triton.KernelIO₁
'''
        for new, _, old_ns, kernels in REDUCTIONS:
            for kernel, io, theorem in kernels:
                formula = ("Real.log (∑ j, Real.exp (xs j))" if new == "StableLogSumExpCorrect"
                           else "Real.exp (xs i) / ∑ j, Real.exp (xs j)")
                source += f'''
example (x y : VeriTile.Triton.RegionName) (B : Nat) :
    {new}.{kernel} x y B = {old_ns}.{kernel} x y B := rfl

example (B : Nat) (hB : 0 < B) :
    VeriTile.Spec.Real ({new}.{io} B ⊨ fun xs i => {formula}) :=
  {new}.{theorem} B hB
'''
        self.lean(source)

    def test_real_float_add_includes_empty_tiles(self):
        self.lean('''
import bench.examples.FloatDTypeAddCorrect
open VeriTile.Bench.Examples.FloatDTypeAddCorrect
open scoped VeriTile.Triton.KernelIO₂

example : VeriTile.Spec.Real (floatAddIO 0 ⊨ fun xs ys i => xs i + ys i) :=
  float_add_correctness 0
''')

    def test_fusion_correctness_preserves_both_sources_and_full_shape_scope(self):
        self.lean('''
import bench.examples.FusedSiLUCorrect
import bench.examples.FusedSiLUEquiv
import bench.examples.FusedSwigluCorrect
import bench.examples.FusedSwigluEquiv
open VeriTile Triton
open VeriTile.Bench.Examples

example (x g r o : RegionName) (B : Nat) :
    FusedSiLUCorrect.fusedSiLUKernel x g r o B =
      FusedSiLUEquiv.fusedSiLUKernel x g r o B := rfl
example (x g r z s o : RegionName) (B : Nat) :
    FusedSiLUCorrect.unfusedSiLUKernel x g r z s o B =
      FusedSiLUEquiv.unfusedSiLUKernel x g r z s o B := rfl
example (x y o : RegionName) (n B : Nat) :
    FusedSwigluCorrect.swiglu_fused x y o n B =
      FusedSwigluEquiv.swiglu_fused x y o n B := rfl
example (x y s o : RegionName) (n B : Nat) :
    FusedSwigluCorrect.swiglu_unfused x y s o n B =
      FusedSwigluEquiv.swiglu_unfused x y s o n B := rfl

section
open scoped VeriTile.Triton.KernelIO₃
example (B : Nat) :
    Spec.Real (FusedSiLUCorrect.fusedIO B ⊨ fun xs gs rs i =>
      rs i + (xs i * gs i) * Real.sigmoid (xs i * gs i)) :=
  FusedSiLUCorrect.fused_silu_correct B
example (B : Nat) :
    Spec.Real (FusedSiLUCorrect.unfusedIO B ⊨ fun xs gs rs i =>
      rs i + (xs i * gs i) * Real.sigmoid (xs i * gs i)) :=
  FusedSiLUCorrect.unfused_silu_correct B
end

section
open scoped VeriTile.Triton.MaskedKernelIO₂
example (n B : Nat) :
    Spec.Real (FusedSwigluCorrect.fusedIO n B ⊨ fun xs ys i =>
      (xs i * Real.sigmoid (xs i)) * ys i) :=
  FusedSwigluCorrect.fused_swiglu_correct n B
example (n B : Nat) :
    Spec.Real (FusedSwigluCorrect.unfusedIO n B ⊨ fun xs ys i =>
      (xs i * Real.sigmoid (xs i)) * ys i) :=
  FusedSwigluCorrect.unfused_swiglu_correct n B
end
''')


if __name__ == "__main__":
    unittest.main()
