import bench.tritonbench_g.diag_ssm_triton.Proofs.Reference
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ReferenceCongruence
open Reference

/-- Only earlier inputs contribute to a forward state. -/
theorem forward_prefix {α : Type} [Semiring α] (initial transition : α)
    (x y : Nat → α) (k : Nat) (h : ∀ t, t<k → x t=y t) :
    forwardState initial transition x k=forwardState initial transition y k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [forwardState,forwardState,ih (fun t ht => h t (by omega)),h k (by omega)]

/-- Reverse states through length steps use only the specified time window. -/
theorem carry_prefix {α : Type} [Semiring α] (length : Nat) (transition : α)
    (x y : Nat → α) (k : Nat) (hk : k≤length) (h : ∀ t, t<length → x t=y t) :
    reverseCarry length transition x k=reverseCarry length transition y k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [reverseCarry,reverseCarry,ih (by omega),h (length-1-k) (by omega)]

theorem previous_prefix {α : Type} (length : Nat) (initial : α) (x y : Nat → α)
    (t : Nat) (ht : t<length) (h : ∀ q, q+1<length → x q=y q) :
    previousState initial x t=previousState initial y t := by
  by_cases hz : t=0
  · simp [previousState,hz]
  · simp only [previousState,hz,↓reduceIte]
    exact h (t-1) (by omega)

theorem transition_prefix {α : Type} [Semiring α] (length : Nat) (transition initial : α)
    (x y gx gy : Nat → α) (k : Nat) (hk : k≤length)
    (ho : ∀ t, t+1<length → x t=y t) (hg : ∀ t, t<length → gx t=gy t) :
    transitionGradient length transition initial x gx k=
      transitionGradient length transition initial y gy k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [transitionGradient,transitionGradient,ih (by omega),
      carry_prefix length transition gx gy k (by omega) hg,
      hg (length-1-k) (by omega),previous_prefix length initial x y (length-1-k) (by omega) ho]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ReferenceCongruence
