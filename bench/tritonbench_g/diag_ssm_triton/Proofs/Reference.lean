import Mathlib.Data.Complex.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

/-! Mathematical lane-wise recurrences, independent of registers, kernel
execution and memory. Complex gradients use conjugated transitions/states;
this specifies the reverse scan performed by the source, without asserting
that a caller's arbitrary grad_y is a derivative of a particular loss. -/
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.Reference
noncomputable section

/-- State after t transitions; output row t contains state t+1. -/
def forwardState {α : Type} [Semiring α] (initial transition : α) (input : Nat → α) : Nat → α
  | 0 => initial
  | t+1 => forwardState initial transition input t * transition + input t

/-- Closed form of the recurrence, valid for real and complex values. -/
theorem forward_closed_form {α : Type} [Semiring α]
    (initial transition : α) (input : Nat → α) (t : Nat) :
    forwardState initial transition input t = initial * transition^t +
      ∑ k ∈ Finset.range t, input k * transition^(t-1-k) := by
  induction t with
  | zero => simp [forwardState]
  | succ t ih =>
    have hsum : (∑ k ∈ Finset.range t, input k * transition^(t-1-k)) * transition =
        ∑ k ∈ Finset.range t, input k * transition^(t-k) := by
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro k hk
      rw [mul_assoc, ← pow_succ]
      have hindex : t-1-k+1 = t-k := by
        have := Finset.mem_range.mp hk
        omega
      rw [hindex]
    rw [forwardState, ih, add_mul, hsum]
    simp [Finset.sum_range_succ, pow_succ, mul_assoc, add_assoc]

/-- Carried adjoint after k reverse steps. For a complex scan, the caller
supplies the conjugated transition. At k=length this is grad_s. -/
def reverseCarry {α : Type} [Semiring α] (length : Nat)
    (adjointTransition : α) (gradOutput : Nat → α) : Nat → α
  | 0 => 0
  | k+1 => (gradOutput (length-1-k) + reverseCarry length adjointTransition gradOutput k) *
      adjointTransition

/-- Input gradient at time t. Only t<length is observed by the IO contract. -/
def inputGradient {α : Type} [Semiring α] (length : Nat)
    (adjointTransition : α) (gradOutput : Nat → α) (t : Nat) : α :=
  gradOutput t + reverseCarry length adjointTransition gradOutput (length-1-t)

/-- Initially supplied previous forward state, optionally conjugated by the
caller. No equation about the output kernel is an input premise. -/
def previousState {α : Type} (initial : α) (output : Nat → α) (t : Nat) : α :=
  if t=0 then initial else output (t-1)

/-- Per-column transition-gradient reduction over reverse time. The source
stores one value per batch/column; a host batch reduction is outside this spec. -/
def transitionGradient {α : Type} [Semiring α] (length : Nat)
    (adjointTransition initial : α) (output gradOutput : Nat → α) : Nat → α
  | 0 => 0
  | k+1 => transitionGradient length adjointTransition initial output gradOutput k +
      (gradOutput (length-1-k) + reverseCarry length adjointTransition gradOutput k) *
      previousState initial output (length-1-k)

/-- Complex state update exposed as the two real expressions in the source. -/
theorem complex_forward_components (s l x : ℂ) :
    (s*l+x).re = s.re*l.re-s.im*l.im+x.re ∧
    (s*l+x).im = s.re*l.im+s.im*l.re+x.im := by
  simp [Complex.mul_re, Complex.mul_im]

/-- The internal conjugated gradient convention implements the usual
conjugated-transition reverse recurrence after output conjugation. -/
theorem complex_carry_conjugation (g carry transition : ℂ) :
    starRingEnd ℂ ((starRingEnd ℂ g + carry) * transition) =
      (g + starRingEnd ℂ carry) * starRingEnd ℂ transition := by
  simp

theorem complex_transition_conjugation (acc g carry state : ℂ) :
    starRingEnd ℂ (acc + (starRingEnd ℂ g + carry) * state) =
      starRingEnd ℂ acc + (g + starRingEnd ℂ carry) * starRingEnd ℂ state := by
  simp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.Reference
