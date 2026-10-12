import VeriTile.Meta.Disprove
import VeriTile.Triton.Float.Countermodel
import VeriTile.Triton.Float.ConditionalIO

/-! Semantic adapters for the generic disprove tactic. Each adapter proves a
negation from a model of the full assumption set, an inhabited input domain,
an in-bounds output lane and unequal observations of the actual kernels.
This module contains no example names, countermodels or report-specific laws. -/
namespace VeriTile.Triton.FP.Scheduled
open Structural Guarded

@[disprove_rule]
theorem not_equiv_of_signature {Program : Type u} [view : Spec.ProgramSyntax Program]
    {R : Spec.Assumptions view.Statement} {lhs rhs : Program}
    (different : view.signature lhs ≠ view.signature rhs) :
    ¬ Spec.FloatingPoint R lhs rhs := fun h => different h.sameSignature

@[disprove_rule]
theorem not_equiv_of_observe₁ {α : Type} [Inhabited α]
    {R : Spec.Assumptions GuardedFragment} {lhs rhs : IO₁}
    {M : Algebra α} {D : Domain α} (hM : Models R M D)
    {plans : Equational.Schedules} {s : State α}
    (hd : (lhs.domain plans).Holds M D s) (i : Fin lhs.io.Bout)
    (different : observe₁ M plans s lhs i.val ≠ observe₁ M plans s rhs i.val) :
    ¬ Spec.FloatingPoint R lhs rhs :=
  fun h => different (observe₁_eq_of_equiv R M D hM plans s h hd i.val i.isLt)

/-- The public conditional interface also preserves required comparisons.
An interpreter without those operations is not a counterexample in its domain. -/
theorem observe₁_eq_of_conditional_equiv {α : Type} [Inhabited α]
    (R : Spec.Assumptions GuardedFragment) (M : Algebra α) (D : Domain α)
    (hM : Models R M D) (plans : Equational.Schedules) (s : State α)
    {lhs rhs : ConditionalIO₁} (h : Spec.FloatingPoint R lhs rhs)
    (hc : ∀ c ∈ lhs.comparisons, c.Supported M)
    (hd : (lhs.domain plans).Holds M D s) (i : Nat) (hi : i < lhs.io.Bout) :
    observe₁ M plans s lhs.toIO₁ i = observe₁ M plans s rhs.toIO₁ i := by
  have derivation := h.derivation
  change Spec.ProgramDerivation (fun _ _ : ConditionalIO₁ => False) R lhs rhs at derivation
  have sound {a b : ConditionalIO₁}
      (proof : Spec.ProgramDerivation (fun _ _ : ConditionalIO₁ => False) R a b) :
      Spec.ProgramSyntax.signature a = Spec.ProgramSyntax.signature b ∧
      ((∀ c ∈ a.comparisons, c.Supported M) → (a.domain plans).Holds M D s →
        i < a.io.Bout → observe₁ M plans s a.toIO₁ i = observe₁ M plans s b.toIO₁ i) := by
    induction proof with
    | refl => exact ⟨rfl, fun _ _ _ => rfl⟩
    | «syntax» hs hctx _ => exact ⟨hs, fun _ _ _ => by cases hctx; rfl⟩
    | structural _ hf => exact hf.elim
    | numerical hs hn =>
        refine ⟨hs, fun hc hd hi => ?_⟩
        obtain ⟨a, b, ha, hb, ho, _, _⟩ := hn.2.2 α M D hM hc plans s hd
        simp only [observe₁, ha, hb, Option.map_some]
        exact congrArg some (ho ⟨i, hi⟩)
    | @symm a b _ ih =>
        refine ⟨ih.1.symm, fun hc hd hi => ?_⟩
        have domains := congrArg (fun sig => sig.2.2.1) ih.1
        change a.domain = b.domain at domains
        have comparisons := congrArg (fun sig => sig.2.2.2) ih.1
        change a.comparisons = b.comparisons at comparisons
        have sizes := congrArg (fun sig => sig.1.Bout) ih.1
        change a.io.Bout = b.io.Bout at sizes
        exact (ih.2 (comparisons.symm ▸ hc) (domains.symm ▸ hd) (sizes.symm ▸ hi)).symm
    | @trans a b c _ _ ih₁ ih₂ =>
        refine ⟨ih₁.1.trans ih₂.1, fun hc hd hi => ?_⟩
        have domains := congrArg (fun sig => sig.2.2.1) ih₁.1
        change a.domain = b.domain at domains
        have comparisons := congrArg (fun sig => sig.2.2.2) ih₁.1
        change a.comparisons = b.comparisons at comparisons
        have sizes := congrArg (fun sig => sig.1.Bout) ih₁.1
        change a.io.Bout = b.io.Bout at sizes
        exact (ih₁.2 hc hd hi).trans
          (ih₂.2 (comparisons ▸ hc) (domains ▸ hd) (sizes ▸ hi))
  exact (sound derivation).2 hc hd hi

@[disprove_rule]
theorem not_conditional_equiv_of_observe₁ {α : Type} [Inhabited α]
    {R : Spec.Assumptions GuardedFragment} {lhs rhs : ConditionalIO₁}
    {M : Algebra α} {D : Domain α} (hM : Models R M D)
    {plans : Equational.Schedules} {s : State α}
    (hc : ∀ c ∈ lhs.comparisons, c.Supported M)
    (hd : (lhs.domain plans).Holds M D s) (i : Fin lhs.io.Bout)
    (different : observe₁ M plans s lhs.toIO₁ i.val ≠
      observe₁ M plans s rhs.toIO₁ i.val) :
    ¬ Spec.FloatingPoint R lhs rhs :=
  fun h => different (observe₁_eq_of_conditional_equiv R M D hM plans s h hc hd i.val i.isLt)

end VeriTile.Triton.FP.Scheduled
