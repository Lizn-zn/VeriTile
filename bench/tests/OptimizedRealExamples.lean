import bench.examples.TritonBenchVectorAddition.Correct
import bench.examples.VectorAdd.Correct
import bench.examples.FlatVectorAdd.Correct
import bench.examples.FloatDTypeAdd.Correct
import bench.examples.AdamUpdateGridLaunch.Correct
import bench.examples.HyperConnectionsDepth.Correct
import bench.examples.HyperConnectionsWidth.Correct
import bench.examples.RowWiseSum.Correct
import bench.examples.RowWiseMax.Correct

/-!
Regression contracts for the optimized real specifications: exact source wiring,
independent mathematical formulas, symbolic sizes and no FP-proof dependency.
-/

open VeriTile Triton
open VeriTile.Bench.Examples

-- Each IO contract selects the optimized source, including explicit cast erasure.
example (n B : Nat) :
    (TritonBenchVectorAdditionCorrect.optimizedIO n B).kernel =
      TritonBenchVectorAddition.Kernels.optimizedKernel n B := rfl
example (B : Nat) :
    (VectorAdd.optimizedIO B).kernel = VectorAdd.Kernels.optimizedKernel B := rfl
example (B n : Nat) :
    (FlatVectorAdd.optimizedIO B n).kernel = FlatVectorAdd.Kernels.optimizedKernel n B := rfl
example (B : Nat) :
    (FloatDTypeAddCorrect.optimizedIO B).kernel =
      (FloatDTypeAdd.Kernels.optimizedKernel B).eraseDType := rfl
example (lr wd beta1 beta2 : ℝ) (n B : Nat) :
    (AdamUpdateGridLaunch.optimizedIO lr wd beta1 beta2 n B).kernel =
      AdamUpdateGridLaunch.Kernels.optimizedKernel lr wd beta1 beta2 n B := rfl
example (tau : ℝ) :
    (HyperConnectionsDepth.optimizedIO tau).kernel =
      HyperConnectionsDepth.Kernels.optimizedKernel tau := rfl
example (tau : ℝ) :
    (HyperConnectionsWidth.optimizedIO tau).kernel =
      HyperConnectionsWidth.Kernels.optimizedKernel tau := rfl
example (nCol B : Nat) :
    (RowWiseSum.reversedIO nCol B).kernel = RowWiseSum.Kernels.reversedKernel "x" "y" nCol B := rfl
example (nCol B : Nat) :
    (RowWiseMax.inlinedIO nCol B).kernel = RowWiseMax.Kernels.inlinedKernel "x" "y" nCol B := rfl

-- The formulas and hypotheses are part of the public API. In particular, the
-- symbolic dimensions here cannot acquire an experimental shape restriction.
section
open scoped VeriTile.Triton.KernelIO₂
example (B : Nat) :
    Spec.Real (VectorAdd.optimizedIO B ⊨ fun xs ys i => xs i + ys i) :=
  VectorAdd.add_kernel_optimized_correctness B
example (B : Nat) :
    Spec.Real (FloatDTypeAddCorrect.optimizedIO B ⊨ fun xs ys i => xs i + ys i) :=
  FloatDTypeAddCorrect.float_add_optimized_correctness B
end

section
open scoped VeriTile.Triton.MaskedKernelIO₂
example (n B : Nat) :
    Spec.Real (TritonBenchVectorAdditionCorrect.optimizedIO n B ⊨ fun xs ys i => xs i + ys i) :=
  TritonBenchVectorAdditionCorrect.vector_addition_optimized_correct n B
example (B n : Nat) (hB : 0 < B) :
    Spec.Real (FlatVectorAdd.optimizedIO B n ⊨ fun xs ys i => xs i + ys i) :=
  FlatVectorAdd.add_kernel_masked_optimized_correctness B n hB
end

section
open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂
example (lr wd beta1 beta2 : ℝ) (n B : Nat) :
    Spec.Real (AdamUpdateGridLaunch.optimizedIO lr wd beta1 beta2 n B ⊨ fun p grad m =>
      (fun i => TiledOptimizer.lionParam (p i) (m i) (grad i) lr wd beta1,
       fun i => TiledOptimizer.lionMomentum (m i) (grad i) beta2)) :=
  AdamUpdateGridLaunch.adam_update_optimized_correctness lr wd beta1 beta2 n B
end

section
open scoped VeriTile.Triton.KernelIO₃
example (tau : ℝ) :
    Spec.Real (HyperConnectionsDepth.optimizedIO tau ⊨ fun resMix branchOut hPost _ =>
      resMix 0 + Real.exp (hPost 0 / tau) * branchOut 0) :=
  HyperConnectionsDepth.mhc_depth_optimized_correctness tau
end

section
open scoped VeriTile.Triton.KernelIO₃ₓ₂
example (tau : ℝ) :
    Spec.Real (HyperConnectionsWidth.optimizedIO tau ⊨ fun res hRes hPre =>
      (fun _ => Real.exp (hRes 0 / tau) * res 0,
       fun _ => Real.exp (hPre 0 / tau) * res 0)) :=
  HyperConnectionsWidth.mhc_width_optimized_correctness tau
end

section
open scoped VeriTile.Triton.KernelIO₁
example (nCol B : Nat) :
    Spec.Real (RowWiseSum.reversedIO nCol B ⊨ fun xs _ => ∑ k, xs k) :=
  RowWiseSum.rowWiseSum_reversed_correctness nCol B
example (nCol B : Nat) (hB : 0 < B) :
    Spec.Real (RowWiseMax.inlinedIO nCol B ⊨ fun xs _ => TiledReduction.tileMax hB xs) :=
  RowWiseMax.rowWiseMax_inlined_correctness nCol B hB
end

-- Real correctness must remain available without importing any of these
-- examples' assumption-dependent FP equivalence proofs.
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for name in #[
      `TritonBenchVectorAdditionFPEquiv.vector_addition_equiv,
      `VectorAddFPEquiv.add_kernel_equiv,
      `FlatVectorAddFPEquiv.add_kernel_masked_equiv,
      `FloatDTypeAddFPEquiv.float_add_equiv,
      `AdamUpdateGridLaunchFPEquiv.adam_update_equiv,
      `HyperConnectionsDepthFPEquiv.mhc_depth_equiv,
      `HyperConnectionsWidthFPEquiv.mhc_width_equiv,
      `RowWiseSumFPEquiv.rowwise_sum_equiv,
      `RowWiseMaxFPEquiv.rowwise_max_equiv] do
    let fullName := `VeriTile.Bench.Examples ++ name
    if env.contains fullName then
      throwError "Real correctness imported an FP equivalence proof: {fullName}"
