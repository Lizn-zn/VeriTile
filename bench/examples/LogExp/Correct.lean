import bench.examples.LogExp.Kernels
import bench.examples.LogExp.Proofs.Real
import VeriTile.Triton.Memory.KernelSpec.Basic
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

noncomputable section

namespace VeriTile.Bench.Examples.LogExp.Correct
open Triton
open scoped VeriTile.Triton.KernelIO₁

set_option maxHeartbeats 1600000

/-- The original source, under the real interpretation of KernelIO₁. -/
def originalIO (xReg yReg : RegionName) (B : Nat) : KernelIO₁ where
  kernel := originalKernel xReg yReg B
  inp := xReg
  out := yReg
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- The piecewise candidate shares its source with the FP proof. -/
def optimizedIO (xReg yReg : RegionName) (B : Nat) : KernelIO₁ :=
  { originalIO xReg yReg B with kernel := optimizedKernel xReg yReg B, projection := by rfl }

/-- The original log-exp source implements the identity function over ℝ. -/
specification original_correct (xReg yReg : RegionName) (B : Nat) :
    Spec.Real (originalIO xReg yReg B ⊨ fun xs i => xs i) := by
  refine KernelIO₁.Implements.intro _ (original_flattenOk xReg yReg B) ?_ ?_
  · intro bounds s hx hy _
    exact original_traceSafe xReg yReg B bounds s hx hy
  · intro s xs hx
    obtain ⟨t, ht, hv, hf⟩ := original_run xReg yReg B s xs hx
    exact ⟨t, ht, hv, fun r o ho _ => hf r o ho⟩

/-- The piecewise candidate implements the same identity function over ℝ. -/
specification optimized_correct (xReg yReg : RegionName) (B : Nat) :
    Spec.Real (optimizedIO xReg yReg B ⊨ fun xs i => xs i) := by
  refine KernelIO₁.Implements.intro _ (optimized_flattenOk xReg yReg B) ?_ ?_
  · intro bounds s hx hy _
    exact optimized_traceSafe xReg yReg B bounds s hx hy
  · intro s xs hx
    obtain ⟨t, ht, hv, hf⟩ := optimized_run xReg yReg B s xs hx
    exact ⟨t, ht, hv, fun r o ho _ => hf r o ho⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.LogExp.Correct
