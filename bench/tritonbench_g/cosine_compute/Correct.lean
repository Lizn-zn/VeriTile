import bench.tritonbench_g.cosine_compute.Kernels
import bench.tritonbench_g.cosine_compute.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Both complete typed sources compute the independent lane-wise real cosine
after precision erasure. The masked IO contract includes successful finite flat
execution, active-address bounds and preservation of untouched tagged cells.
Sizes remain symbolic, including empty blocks and partial tails. The source's
unspecified masked values are retained; only active outputs are constrained.
The shared flat IO adapter requires a zero unspecified-value oracle at launch;
Proofs/Real.lean proves region execution and outputs for arbitrary such values.
This is per-program real correctness, not an IEEE or concurrent-launch claim. -/
namespace VeriTile.Bench.TritonBenchG.CosineCompute.Correctness
open VeriTile VeriTile.Triton RealProof
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section
set_option maxHeartbeats 1000000

def originalIO (storage : Storage) (a b : RegionName) (n B : Nat) : MaskedKernelIO₁ where
  kernel := (kernel storage Bool.false a b n B).eraseDType
  projection := RealProof.projection storage Bool.false a b n B
  inp := a
  out := b
  B := B
  read := fun pid => pid*B
  write := fun pid => pid*B
  mask := fun pid j => pid*B+j.val < n

def indexCommutedIO (storage : Storage) (a b : RegionName) (n B : Nat) : MaskedKernelIO₁ :=
  { originalIO storage a b n B with
    kernel := (kernel storage Bool.true a b n B).eraseDType
    projection := RealProof.projection storage Bool.true a b n B }

specification original_correct (storage : Storage) (a b : RegionName) (n B : Nat) :
    Spec.Real (originalIO storage a b n B ⊨ fun xs i => Real.cos (xs i)) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · change (kernel storage Bool.false a b n B).eraseDType.toAlgKernel.FlattenOk
    rw [projects]; exact cos_func_flattenOk a b n B
  · intro bounds s hx ho _
    change Kernel.TraceSafe bounds (kernel storage Bool.false a b n B).eraseDType.toAlgKernel s
    rw [projects]; exact cos_func_traceSafe a b n B bounds s hx ho
  · intro s xs hx
    obtain ⟨t,he,hv,hf⟩ := cos_func_region_run a b n B s xs hx
    refine ⟨t,?_,hv,fun r o hmiss _ => hf r o hmiss⟩
    change exec (kernel storage Bool.false a b n B).eraseDType.toAlgKernel s = some t
    rw [projects]; exact he

specification index_commuted_correct (storage : Storage) (a b : RegionName) (n B : Nat) :
    Spec.Real (indexCommutedIO storage a b n B ⊨ fun xs i => Real.cos (xs i)) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · change (kernel storage Bool.true a b n B).eraseDType.toAlgKernel.FlattenOk
    rw [projects]; exact commuted_flattenOk a b n B
  · intro bounds s hx ho _
    change Kernel.TraceSafe bounds (kernel storage Bool.true a b n B).eraseDType.toAlgKernel s
    rw [projects]; exact commuted_traceSafe a b n B bounds s hx ho
  · intro s xs hx
    obtain ⟨t,he,hv,hf⟩ := cos_func_region_run a b n B s xs hx
    refine ⟨t,?_,hv,fun r o hmiss _ => hf r o hmiss⟩
    change exec (kernel storage Bool.true a b n B).eraseDType.toAlgKernel s = some t
    rw [projects,commuted_exec]; exact he

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CosineCompute.Correctness
