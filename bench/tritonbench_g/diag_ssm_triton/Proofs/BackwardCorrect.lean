import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardProjection
import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardReference
import bench.tritonbench_g.diag_ssm_triton.Proofs.Candidate
import bench.tritonbench_g.decay_cumsum.Proofs.GroupedRealIO
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardCorrect
open VeriTile VeriTile.Triton TypedSource
open scoped VeriTile.Triton.StreamGroupedEmitMasked3DKernelIO
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

theorem source_projects (storage : Storage) (swapped : Bool) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    (TypedSource.backward storage swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgorithm? =
      .ok (TypedSource.backward storage swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel := by
  cases storage <;> cases swapped <;> rfl

/-- The branch-dependent previous-state reads, static transition, emitted input
gradients and two terminal outputs are all explicit. Active accesses must fit
the finite allocation. The real adapter also requires zero unspecified values. -/
def originalIO (storage : Storage)
    (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr
      grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) :
    StreamGroupedEmitMasked3DKernelIO where
  kernel := (TypedSource.backward storage Bool.false s_ptr lambda_ptr y_ptr grad_s_ptr
    grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  nIn := 4
  nOut := 3
  bufs := [s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr,
    grad_lambda_ptr, grad_y_ptr]
  inp := fun i => match i with
    | ⟨0, _⟩ => grad_y_ptr
    | ⟨1, _⟩ => y_ptr
    | ⟨2, _⟩ => s_ptr
    | ⟨_ + 3, _⟩ => lambda_ptr
  out := fun o => match o with
    | ⟨0, _⟩ => grad_x_ptr
    | ⟨1, _⟩ => grad_s_ptr
    | ⟨_ + 2, _⟩ => grad_lambda_ptr
  T := length
  B := BLOCK_SIZE
  read := fun i p₀ _ _ t j => match i with
    | ⟨0, _⟩ => t.val * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
    | ⟨1, _⟩ => (t.val - 1) * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
    | ⟨2, _⟩ => p₀ * BLOCK_SIZE + j.val
    | ⟨_ + 3, _⟩ => (p₀ * BLOCK_SIZE + j.val) % dim
  readMask := fun i p₀ _ _ t j => match i with
    | ⟨0, _⟩ => p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨1, _⟩ => 0 < t.val ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨2, _⟩ => t.val = 0 ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨_ + 3, _⟩ => p₀ * BLOCK_SIZE + j.val < batch_size * dim
  write := fun o p₀ _ _ t j => match o with
    | ⟨0, _⟩ => t.val * (batch_size * dim) + (p₀ * BLOCK_SIZE + j.val)
    | ⟨1, _⟩ => p₀ * BLOCK_SIZE + j.val
    | ⟨_ + 2, _⟩ => p₀ * BLOCK_SIZE + j.val
  writeMask := fun o p₀ _ _ t j => match o with
    | ⟨0, _⟩ => p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨1, _⟩ => t.val = 0 ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim
    | ⟨_ + 2, _⟩ => t.val = 0 ∧ p₀ * BLOCK_SIZE + j.val < batch_size * dim

def indexCommutedIO (storage : Storage) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) : StreamGroupedEmitMasked3DKernelIO :=
  { originalIO storage s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE with
    kernel := (TypedSource.backward storage Bool.true s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

theorem original_correct (storage : Storage) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (hLength : 0<length)
    (hGradXSNe : grad_x_ptr≠s_ptr) (hGradXYNe : grad_x_ptr≠y_ptr)
    (hGradXGradYNe : grad_x_ptr≠grad_y_ptr) (hGradXLambdaNe : grad_x_ptr≠lambda_ptr)
    (hGradSGradXNe : grad_s_ptr≠grad_x_ptr) (hGradLambdaGradXNe : grad_lambda_ptr≠grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr≠grad_s_ptr) :
    Spec.Real (originalIO storage s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE ⊨
      fun _ _ _ xs o t j => BackwardReference.gradients xs o t j) := by
  have h := StreamGroupedEmitMasked3DKernelIO.Implements.of_identityRounding
    (RealReference.diagSsmBackwardKernelIO s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE) rfl
    (fun _ _ _ xs o t j => RealReference.diagSsmBackwardStreamSpec xs o t j)
    (RealReference.diag_ssm_backward_io_correctness .triv s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE hLength hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe hGradSGradXNe hGradLambdaGradXNe hGradLambdaGradSNe)
  simpa only [Spec.Real,StreamGroupedEmitMasked3DKernelIO.Implements,originalIO,
    RealReference.diagSsmBackwardKernelIO,BackwardProjection.typed_source_flat_exec,
    BackwardReference.gradients_reference] using h

theorem candidate_correct (storage : Storage) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName)
    (length batch_size dim BLOCK_SIZE : Nat) (hLength : 0<length)
    (hGradXSNe : grad_x_ptr≠s_ptr) (hGradXYNe : grad_x_ptr≠y_ptr)
    (hGradXGradYNe : grad_x_ptr≠grad_y_ptr) (hGradXLambdaNe : grad_x_ptr≠lambda_ptr)
    (hGradSGradXNe : grad_s_ptr≠grad_x_ptr) (hGradLambdaGradXNe : grad_lambda_ptr≠grad_x_ptr)
    (hGradLambdaGradSNe : grad_lambda_ptr≠grad_s_ptr) :
    Spec.Real (indexCommutedIO storage s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE ⊨
      fun _ _ _ xs o t j => BackwardReference.gradients xs o t j) := by
  have h := original_correct storage s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE hLength hGradXSNe hGradXYNe hGradXGradYNe hGradXLambdaNe hGradSGradXNe hGradLambdaGradXNe hGradLambdaGradSNe
  simpa only [Spec.Real,StreamGroupedEmitMasked3DKernelIO.Implements,originalIO,indexCommutedIO,
    ← Candidate.backward_flat_exec] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.BackwardCorrect
