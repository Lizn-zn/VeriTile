import bench.tritonbench_g.adam_update_triton.Kernels
import bench.tritonbench_g.adam_update_triton.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Mathematical correctness of both sources in Kernels.lean.
For the initially loaded parameters p, gradients g and momentum m, the outputs are
  p' = (1 - lr * wd) * p - lr * sign(beta1 * m + (1 - beta1) * g)
  m' = beta2 * m + (1 - beta2) * g.
These are the independent TiledOptimizer.lionParam/lionMomentum formulas.
The two output regions must differ; their roles alias their respective inputs.
The masked contracts include termination, active-address bounds and the frame
outside both output windows. They describe one program, with symbolic sizes. -/

namespace VeriTile.Bench.TritonBenchG.AdamUpdateTriton
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂

/-- All three reads and both writes use the same symbolic block and tail mask.
The allocation list is separate from the in-place input/output roles. -/
def adamIO (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    MaskedKernelIO₃ₓ₂ where
  kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE
  bufs := [p_ptr, grad_ptr, exp_avg_ptr]  -- p and exp_avg are updated in place
  in1 := p_ptr
  in2 := grad_ptr
  in3 := exp_avg_ptr
  out1 := p_ptr          -- = in1: in-place parameter update
  out2 := exp_avg_ptr    -- = in3: in-place momentum update
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  read3 := fun pid => pid * BLOCK_SIZE
  write1 := fun pid => pid * BLOCK_SIZE
  write2 := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements

/-- Both original outputs satisfy the Lion formulas over real arithmetic. -/
specification update_fn_kernel_correctness
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (hRegions : p_ptr ≠ exp_avg_ptr) :
    Spec.Real (adamIO p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n_elements BLOCK_SIZE
      ⊨ fun p grad expAvg =>
        (fun i => TiledOptimizer.lionParam (p i) (expAvg i) (grad i) lr wd beta1,
         fun i => TiledOptimizer.lionMomentum (expAvg i) (grad i) beta2)) := by
  refine MaskedKernelIO₃ₓ₂.Implements.intro _
    (by simp [adamIO]) (by simp [adamIO]) ?_ ?_ ?_
  · exact update_fn_kernel_flattenOk p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE
  · intro bounds s h1 h2 h3 _ _
    exact update_fn_kernel_traceSafe p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE bounds s h1 h2 h3
  · intro s₀ xs ys zs hx hy hz
    exact update_fn_kernel_region_run p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE hRegions s₀ xs ys zs hx hy hz


/-- The candidate retains the allocation list, in-place wiring and tail mask. -/
def commutedIO (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) : MaskedKernelIO₃ₓ₂ :=
  { adamIO p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B with
    kernel := update_commuted_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B
    projection := by rfl }

/-- Both candidate outputs satisfy the same independent real formulas. -/
specification update_commuted_correctness (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n B : Nat) (hRegions : p_ptr ≠ exp_avg_ptr) :
    Spec.Real (commutedIO p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B
      ⊨ fun p grad expAvg =>
        (fun i => TiledOptimizer.lionParam (p i) (expAvg i) (grad i) lr wd beta1,
         fun i => TiledOptimizer.lionMomentum (expAvg i) (grad i) beta2)) := by
  refine MaskedKernelIO₃ₓ₂.Implements.intro _
    (by simp [commutedIO, adamIO]) (by simp [commutedIO, adamIO]) ?_ ?_ ?_
  · exact commuted_flattenOk p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B
  · intro bounds s hp hg he _ _
    exact commuted_traceSafe p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B bounds s hp hg he
  · intro s xs ys zs hx hy hz
    obtain ⟨s1, he, hp, hm, hf⟩ := update_fn_kernel_region_run
      p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B hRegions s xs ys zs hx hy hz
    exact ⟨s1, (commuted_exec p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B s).trans he,
      hp, hm, hf⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton
