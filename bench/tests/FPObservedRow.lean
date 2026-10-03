/- Readbacks do not add stores or equate failed observations. -/
import bench.examples.OnlineSoftmaxFPEquiv

namespace FPObservedRowTests
open VeriTile Triton FP.Structural
open VeriTile.Bench.Examples.OnlineSoftmaxFPContract
open FP.Equational (Schedules)

theorem missing_register_fails {α : Type} [Inhabited α] (M : Algebra α)
    (plans : Schedules) (s : State α) (x y : RegionName) (N pid : Nat) (i : Fin N)
    (hm : s.regs .real [] "m" = none) :
    FP.Structural.evalOp (FP.Scheduled.fp32.algebra M plans) none
      ((normalizedOnline x y N).observe pid i) s = none := by
  simp [normalizedOnline, loadRow, evalOp_unfold, hm]

theorem online_frame_preserves_all_memory {α : Type} [Inhabited α]
    (s t : State α) (x y : RegionName) (N : Nat)
    (h : FP.ObservedRow.Frame (normalizedOnline x y N) s t) : t.mem = s.mem := by
  funext r o
  exact h r o (Or.inl rfl)

theorem batch_frame_preserves_other_rows {α : Type} [Inhabited α]
    (s t : State α) (x y r : RegionName) (N o : Nat)
    (h : FP.ObservedRow.Frame (batchOutput x y N) s t)
    (ho : r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * N + i.val) : t.mem r o = s.mem r o :=
  h r o (Or.inr ho)

/-- The output observation remains a successful real-valued read, even when
the output region aliases the input; it uses the original program ID. -/
theorem batch_aliased_readback {α : Type} [Inhabited α] (M : Algebra α)
    (plans : Schedules) (s : State α) (r : RegionName) (N pid : Nat)
    (i : Fin N) (v : α) (hv : s.mem r (pid * N + i.val) = .mk .real v) :
    FP.Structural.evalOp (FP.Scheduled.fp32.algebra M plans) none
      ((batchOutput r r N).observe pid i) s = some (fun _ => v) := by
  simp [batchOutput, loadRow, evalOp_unfold, Region.cast, hv]

theorem same_observation_signature (x y : RegionName) (N : Nat) :
    Spec.ProgramSyntax.signature (batchOutput x y N) =
      Spec.ProgramSyntax.signature (normalizedOnline x y N) := rfl

/-- An unchanged kernel body cannot silently replace its readback. -/
theorem different_readback_is_not_same_context (x y : RegionName) (N : Nat)
    (a b : Nat → Fin N → Op .real []) (hab : a ≠ b) :
    ¬ Spec.ProgramSyntax.sameContext
      { batchOutput x y N with observe := a }
      { batchOutput x y N with observe := b } := by
  intro h
  have he := congrArg (fun p : FP.ObservedRow.Program =>
    (⟨p.size, p.observe⟩ : Σ n, Nat → Fin n → Op .real [])) h
  exact hab (eq_of_heq (Sigma.mk.inj_iff.mp he).2)

#guard_msgs (drop info) in
#auditModuleAxioms
end FPObservedRowTests
