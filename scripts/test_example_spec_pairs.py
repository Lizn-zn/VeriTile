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
                    for name in ("FusedSiLU", "FusedSwiglu", "Welford", "FusedLayerNorm")
                    for suffix in ("Correct", "Equiv")]
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

    def test_statistics_correctness_preserves_sources_and_formulas(self):
        self.lean('''
import bench.examples.WelfordCorrect
import bench.examples.WelfordEquiv
import bench.examples.FusedLayerNormCorrect
import bench.examples.FusedLayerNormEquiv
open VeriTile Triton
open VeriTile.Bench.Examples

example (x m v : RegionName) (N stride : Nat) :
    WelfordCorrect.twopassWelfordKernel x m v N stride =
      Welford.twopassWelfordKernel x m v N stride := rfl
example (x m v : RegionName) (N stride : Nat) :
    WelfordCorrect.onlineWelfordKernel x m v N stride =
      Welford.onlineWelfordKernel x m v N stride := rfl
example (x g b y : RegionName) (N stride : Nat) (ε : ℝ) :
    FusedLayerNormCorrect.twoPassLayerNormKernel x g b y N stride ε =
      LayerNorm.twoPassLayerNormKernel x g b y N stride ε := rfl
example (x g b y : RegionName) (N stride : Nat) (ε : ℝ) :
    FusedLayerNormCorrect.fusedLayerNormKernel x g b y N stride ε =
      LayerNorm.fusedLayerNormKernel x g b y N stride ε := rfl

section
open scoped VeriTile.Triton.KernelIO₁ₓ₂
example (N stride : Nat) :
    Spec.Real (WelfordCorrect.twopassIO N stride ⊨ fun xs =>
      ((fun _ => (∑ i, xs i) / N),
       (fun _ => (∑ i, (xs i - (∑ j, xs j) / N) ^ 2) / N))) :=
  WelfordCorrect.twopass_welford_correct N stride
example (N stride : Nat) :
    Spec.Real (WelfordCorrect.onlineIO N stride ⊨ fun xs =>
      ((fun _ => (∑ i, xs i) / N),
       (fun _ => (∑ i, (xs i - (∑ j, xs j) / N) ^ 2) / N))) :=
  WelfordCorrect.online_welford_correct N stride
end

section
open scoped VeriTile.Triton.KernelIO₃
example (N stride : Nat) (ε : ℝ) :
    Spec.Real (FusedLayerNormCorrect.twoPassIO N stride ε ⊨ fun xs gs bs i =>
      let μ := (∑ j, xs j) / N
      let variance := (∑ j, (xs j - μ) ^ 2) / N
      (xs i - μ) / Real.sqrt (variance + ε) * gs i + bs i) :=
  FusedLayerNormCorrect.two_pass_layernorm_correct N stride ε
example (N stride : Nat) (ε : ℝ) :
    Spec.Real (FusedLayerNormCorrect.fusedIO N stride ε ⊨ fun xs gs bs i =>
      let μ := (∑ j, xs j) / N
      let variance := (∑ j, (xs j - μ) ^ 2) / N
      (xs i - μ) / Real.sqrt (variance + ε) * gs i + bs i) :=
  FusedLayerNormCorrect.fused_layernorm_correct N stride ε
end
''')

    def test_two_output_correctness_reaches_flat_memory_and_frames_each_window(self):
        self.lean('''
import bench.examples.WelfordCorrect
open VeriTile Triton
open VeriTile.Bench.Examples.WelfordCorrect
open scoped VeriTile.Triton.KernelIO₁ₓ₂

private def slots : List (RegionName × Nat) := [("x", 2), ("mean", 2), ("var", 2)]
private def alloc := FlatAlloc.ofList "flat" slots
private def samples : Fin 2 → ℝ := ![1, 3]
private noncomputable def initial := DenoteSlot.state 0 [DenoteSlot.ofFin "x" 0 samples]

-- This is a flat-memory consumer, not a region-run helper: check both results
-- and an unwritten cell in each output region. Their capacities exceed their
-- one-cell output windows, so framing whole output regions would be caught.
example : ∃ t,
    exec (alloc.flattenKernel (twopassIO 2 2).kernel.toAlgKernel)
      (alloc.flattenState initial) = some t ∧
    t.readMem "flat" (alloc.addr "mean" 0) = 2 ∧
    t.readMem "flat" (alloc.addr "var" 0) = 1 ∧
    t.mem "flat" (alloc.addr "mean" 1) = (alloc.flattenState initial).mem "flat" (alloc.addr "mean" 1) ∧
    t.mem "flat" (alloc.addr "var" 1) = (alloc.flattenState initial).mem "flat" (alloc.addr "var" 1) := by
  have hd : alloc.Disjoint := FlatAlloc.ofList_disjoint _ _ (by decide)
  have hx : ∀ j : Fin 2, initial.readMem "x" j.val = samples j := by
    intro j
    simpa [initial] using DenoteSlot.state_read_ofFin
      (pid := 0) (region := "x") (off := 0) (arr := samples) (slots := [DenoteSlot.ofFin "x" 0 samples]) (by simp) (by simp) j
  obtain ⟨t, he, hm, hv, hf⟩ := twopass_welford_correct 2 2 alloc hd rfl
    (FlatAlloc.ofList_closed _ _) 0 (by decide) (by decide) (by decide)
    (by simp [twopassIO]) samples initial rfl rfl (by simpa only [twopassIO, Nat.zero_mul, Nat.zero_add] using hx)
  refine ⟨t, he, ?_, ?_, ?_, ?_⟩
  · have h := hm ⟨0, Nat.zero_lt_one⟩
    change t.readMem alloc.flat (alloc.addr "mean" 0) = (∑ i : Fin 2, samples i) / 2 at h
    norm_num [samples, Fin.sum_univ_two] at h
    exact h
  · have h := hv ⟨0, Nat.zero_lt_one⟩
    change t.readMem alloc.flat (alloc.addr "var" 0) =
      (∑ i : Fin 2, (samples i - (∑ j : Fin 2, samples j) / 2) ^ 2) / 2 at h
    norm_num [samples, Fin.sum_univ_two] at h
    exact h
  · apply hf
    right
    norm_num [twopassIO, alloc, slots, FlatAlloc.addr, FlatAlloc.listBase]
    decide
  · apply hf
    right
    norm_num [twopassIO, alloc, slots, FlatAlloc.addr, FlatAlloc.listBase]
    decide
''')

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
