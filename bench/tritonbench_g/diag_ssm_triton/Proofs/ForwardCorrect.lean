import bench.tritonbench_g.diag_ssm_triton.Proofs.ForwardProjection
import bench.tritonbench_g.diag_ssm_triton.Proofs.Candidate
import bench.tritonbench_g.diag_ssm_triton.Proofs.Reference
import bench.tritonbench_g.diag_ssm_triton.Proofs.RealIO
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardCorrect
open VeriTile VeriTile.Triton TypedSource
open scoped VeriTile.Triton.StreamEmitMasked2DKernelIO₃
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

theorem source_projects (storage : Storage) (swapped : Bool) (s x l y : RegionName)
    (length batch dim block : Nat) :
    (TypedSource.forward storage swapped s x l y length batch dim block).toAlgorithm? =
      .ok (TypedSource.forward storage swapped s x l y length batch dim block).toAlgKernel := by
  cases storage <;> cases swapped <;> rfl

/-- Static initial-state/transition windows and time-varying input/output
windows. Every actual access must fit the finite allocation; the real
adapter requires zero unspecified values. Positive length below ensures
that the static pre-loop reads have bounds in this stream interface. -/
def originalIO (storage : Storage) (s x l y : RegionName)
    (length batch dim block : Nat) : StreamEmitMasked2DKernelIO₃ where
  kernel := (TypedSource.forward storage Bool.false s x l y length batch dim block).eraseDType
  projection := by
    simp only [ComputeKernel.eraseDType,source_projects,
      ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody]
  inp1 := s
  inp2 := l
  inp3 := x
  out := y
  T := length
  B1 := block
  B2 := block
  B3 := block
  C := block
  read1 := fun p _ _ j => p*block+j.val
  read2 := fun p _ _ j => (p*block+j.val)%dim
  read3 := fun p _ t j => t.val*(batch*dim)+(p*block+j.val)
  write := fun p _ t j => t.val*(batch*dim)+(p*block+j.val)
  mask1 := fun p _ _ j => p*block+j.val<batch*dim
  mask2 := fun p _ _ j => p*block+j.val<batch*dim
  mask3 := fun p _ _ j => p*block+j.val<batch*dim
  writeMask := fun p _ _ j => p*block+j.val<batch*dim

def indexCommutedIO (storage : Storage) (s x l y : RegionName)
    (length batch dim block : Nat) : StreamEmitMasked2DKernelIO₃ :=
  { originalIO storage s x l y length batch dim block with
    kernel := (TypedSource.forward storage Bool.true s x l y length batch dim block).eraseDType
    projection := by
      simp only [ComputeKernel.eraseDType,source_projects,
        ComputeKernel.toAlgorithm?_fromKernelBody,ComputeKernel.toAlgKernel_fromKernelBody] }

/-- Independent scalar SSM recurrence. Reference.forward_closed_form also
identifies this value as the initial-state power plus a weighted input sum. -/
def scan {length : Nat} (initial transition : ℝ) (input : Fin length → ℝ)
    (t : Fin length) : ℝ :=
  Reference.forwardState initial transition (fun k => if h : k<length then input ⟨k,h⟩ else 0) (t.val+1)

theorem stream_reference {length : Nat} (s l : ℝ) (x : Fin length → ℝ) (n : Nat) :
    RealReference.diagSsmStreamState s l x n =
      Reference.forwardState s l (fun k => if h : k<length then x ⟨k,h⟩ else 0) n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [RealReference.diagSsmStreamState,Reference.forwardState,ih]

theorem original_correct (storage : Storage) (s x l y : RegionName)
    (length batch dim block : Nat) (hLength : 0<length) (hne : x≠y) :
    Spec.Real (originalIO storage s x l y length batch dim block ⊨
      fun _ _ ss ls xs t j => scan (ss t j) (ls t j) (fun u => xs u j) t) := by
  have h := StreamEmitMasked2DKernelIO₃.Implements.of_identityRounding
    (RealReference.diagSsmForwardKernelIO s x l y length batch dim block) rfl
    (fun _ _ ss ls xs t j => RealReference.diagSsmStreamState (ss t j) (ls t j) (fun u => xs u j) (t.val+1))
    (RealReference.diag_ssm_forward_io_correctness .triv s x l y length batch dim block hLength hne)
  simpa only [Spec.Real,StreamEmitMasked2DKernelIO₃.Implements,originalIO,
    RealReference.diagSsmForwardKernelIO,ForwardProjection.typed_source_flat_exec,
    stream_reference,scan] using h

theorem candidate_correct (storage : Storage) (s x l y : RegionName)
    (length batch dim block : Nat) (hLength : 0<length) (hne : x≠y) :
    Spec.Real (indexCommutedIO storage s x l y length batch dim block ⊨
      fun _ _ ss ls xs t j => scan (ss t j) (ls t j) (fun u => xs u j) t) := by
  have h := original_correct storage s x l y length batch dim block hLength hne
  simpa only [Spec.Real,StreamEmitMasked2DKernelIO₃.Implements,originalIO,indexCommutedIO,
    ← Candidate.forward_flat_exec] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ForwardCorrect
