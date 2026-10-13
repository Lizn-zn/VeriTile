import bench.tritonbench_g.log_softmax.Proofs.FP
import bench.tritonbench_g.log_softmax.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.LogSoftmax.FPEquiv
open VeriTile Triton FP.ProfiledRewrite FP.Structural FPProof
open scoped VeriTile.Spec
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def forwardProgram (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) : Program :=
  ⟨typedForward input output v OUT X M N K R C, []⟩

theorem forward_decomposition (input output : Storage) (v : Bool)
    (OUT X : RegionName) (M N K R C : Nat) :
    (typedForward input output v OUT X M N K R C).surfaceBody = sourcePrefix v R ++
      (typedForward input output Bool.false OUT X M N K R C).surfaceBody.drop 3 := by
  cases input <;> cases output <;> cases v <;> rfl

/-- Full-state equivalence, including failure and aliases; no numerical admission. -/
specification forward_index_equiv (input output : Storage)
    (OUT X : RegionName) (M N K R C : Nat) :
    forwardProgram input output Bool.false OUT X M N K R C ≡[assumptions]
      forwardProgram input output Bool.true OUT X M N K R C := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases input <;> cases output <;> rfl)
  intro α _ Mdl D _ st _
  change run (engine Mdl) (typedForward input output Bool.false OUT X M N K R C).surfaceBody st =
    run (engine Mdl) (typedForward input output Bool.true OUT X M N K R C).surfaceBody st
  rw [forward_decomposition input output Bool.false OUT X M N K R C,
    forward_decomposition input output Bool.true OUT X M N K R C]
  exact prefix_run Mdl st R _

#print_fp_assumptions forward_index_equiv

def backwardProgram (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) : Program :=
  ⟨typedBackward input upstream output v Y G DX M N K R C, []⟩

theorem backward_decomposition (input upstream output : Storage) (v : Bool)
    (Y G DX : RegionName) (M N K R C : Nat) :
    (typedBackward input upstream output v Y G DX M N K R C).surfaceBody = sourcePrefix v R ++
      (typedBackward input upstream output Bool.false Y G DX M N K R C).surfaceBody.drop 3 := by
  cases input <;> cases upstream <;> cases output <;> cases v <;> rfl

/-- Full-state equivalence, including failure and aliases; no numerical admission. -/
specification backward_index_equiv (input upstream output : Storage)
    (Y G DX : RegionName) (M N K R C : Nat) :
    backwardProgram input upstream output Bool.false Y G DX M N K R C ≡[assumptions]
      backwardProgram input upstream output Bool.true Y G DX M N K R C := by
  apply Spec.FloatingPoint.ofNumerical (structural := fun _ _ => False) rfl
    (by cases input <;> cases upstream <;> cases output <;> rfl)
  intro α _ Mdl D _ st _
  change run (engine Mdl) (typedBackward input upstream output Bool.false Y G DX M N K R C).surfaceBody st =
    run (engine Mdl) (typedBackward input upstream output Bool.true Y G DX M N K R C).surfaceBody st
  rw [backward_decomposition input upstream output Bool.false Y G DX M N K R C,
    backward_decomposition input upstream output Bool.true Y G DX M N K R C]
  exact prefix_run Mdl st R _

#print_fp_assumptions backward_index_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LogSoftmax.FPEquiv
