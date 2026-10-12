import bench.tritonbench_g.diag_ssm_triton.Proofs.Reference
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardReference
open Reference
noncomputable section

/-- Conjugating the source's internal carry gives the mathematical reverse
recurrence with a conjugated transition and the caller's original grad_y. -/
theorem conjugate_carry (length : Nat) (transition : ℂ) (gradOutput : Nat → ℂ) (k : Nat) :
    starRingEnd ℂ (reverseCarry length transition (fun t => starRingEnd ℂ (gradOutput t)) k) =
      reverseCarry length (starRingEnd ℂ transition) gradOutput k := by
  induction k with
  | zero => simp [reverseCarry]
  | succ k ih => simp [reverseCarry,ih]

theorem conjugate_previous (initial : ℂ) (output : Nat → ℂ) (t : Nat) :
    starRingEnd ℂ (previousState initial output t) =
      previousState (starRingEnd ℂ initial) (fun t => starRingEnd ℂ (output t)) t := by
  by_cases ht : t=0 <;> simp [previousState,ht]

theorem conjugate_transition (length : Nat) (transition initial : ℂ)
    (output gradOutput : Nat → ℂ) (k : Nat) :
    starRingEnd ℂ (transitionGradient length transition initial output
      (fun t => starRingEnd ℂ (gradOutput t)) k) =
      transitionGradient length (starRingEnd ℂ transition) (starRingEnd ℂ initial)
        (fun t => starRingEnd ℂ (output t)) gradOutput k := by
  induction k with
  | zero => simp [transitionGradient]
  | succ k ih => simp [transitionGradient,ih,conjugate_carry,conjugate_previous]

theorem conjugate_input (length : Nat) (transition : ℂ) (gradOutput : Nat → ℂ) (t : Nat) :
    starRingEnd ℂ (inputGradient length transition (fun q => starRingEnd ℂ (gradOutput q)) t) =
      inputGradient length (starRingEnd ℂ transition) gradOutput t := by
  simp [inputGradient,conjugate_carry]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardReference
