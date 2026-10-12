import VeriTile.Triton.Float.ScheduledIO
import Mathlib.Data.List.Forall2

/-! Soundness tools for refuting an FP specification with an interpreted model.
An unequal numerical result alone is insufficient: the model must satisfy every
admitted rule, and the observation must respect every specification constructor. -/
namespace VeriTile.Triton.FP
open Structural Guarded

/-- Equal guards and equal scalar output observations on their domain. -/
def FragmentSound {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (lhs rhs : GuardedFragment) : Prop :=
  lhs.guards = rhs.guards ∧ ∀ s, ScalarDomain D lhs.guards s →
    (run M lhs.code s).map (fun t => t.regs .real [] "out") =
      (run M rhs.code s).map (fun t => t.regs .real [] "out")

theorem models_of_atoms {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (R : Spec.Assumptions GuardedFragment)
    (atoms : ∀ e ∈ R, List.Forall₂ (FragmentSound M D) e.rule.lhs e.rule.rhs) :
    Models R M D := by
  have reflexive (xs : List GuardedFragment) : List.Forall₂ (FragmentSound M D) xs xs := by
    induction xs with
    | nil => exact .nil
    | cons x xs ih => exact .cons ⟨rfl, fun _ _ => rfl⟩ ih
  have chain {xs ys zs : List GuardedFragment}
      (h₁ : List.Forall₂ (FragmentSound M D) xs ys)
      (h₂ : List.Forall₂ (FragmentSound M D) ys zs) :
      List.Forall₂ (FragmentSound M D) xs zs := by
    induction h₁ generalizing zs with
    | nil => exact h₂
    | cons h₁ _ ih =>
        cases h₂ with
        | cons h₂ tail =>
            exact .cons ⟨h₁.1.trans h₂.1, fun s hd =>
              (h₁.2 s hd).trans (h₂.2 s (h₁.1 ▸ hd))⟩ (ih tail)
  have sound {xs ys} (h : Spec.Derivation R xs ys) :
      List.Forall₂ (FragmentSound M D) xs ys := by
    induction h with
    | refl xs => exact reflexive xs
    | atom e he _ => exact atoms e he
    | symm _ ih =>
        exact ih.flip.imp (fun _ _ h => ⟨h.1.symm, fun s hd =>
          (h.2 s (h.1.symm ▸ hd)).symm⟩)
    | trans _ _ ih₁ ih₂ => exact chain ih₁ ih₂
    | frame pre post _ ih => exact List.rel_append (List.rel_append (reflexive pre) ih) (reflexive post)
  intro lhs rhs h s hl _ a b ha hb
  have hs := (List.forall₂_cons.mp (sound h)).1
  simpa only [ha, hb, Option.map_some, Option.some.injEq] using hs.2 s hl

namespace Scheduled

/-- Observe one output lane, including execution failure. -/
noncomputable def observe₁ {α : Type} [Inhabited α] (M : Algebra α)
    (plans : Equational.Schedules) (s : State α) (p : IO₁) (i : Nat) : Option (Cell α) :=
  (Structural.exec (p.profile.algebra M plans) p.io.kernel s).map
    (fun t => t.mem p.io.out (p.io.write (s.pids 0) + i))

/-- All ways of constructing the public FP equivalence preserve observations
in every model. In particular, symmetry and intermediate programs cannot evade
a counterexample to the numerical relation. -/
theorem observe₁_eq_of_equiv {α : Type} [Inhabited α]
    (R : Spec.Assumptions GuardedFragment) (M : Algebra α) (D : Domain α)
    (hM : Models R M D) (plans : Equational.Schedules) (s : State α)
    {lhs rhs : IO₁} (h : Spec.FloatingPoint R lhs rhs)
    (hd : (lhs.domain plans).Holds M D s) (i : Nat) (hi : i < lhs.io.Bout) :
    observe₁ M plans s lhs i = observe₁ M plans s rhs i := by
  have derivation := h.derivation
  change Spec.ProgramDerivation (fun _ _ : IO₁ => False) R lhs rhs at derivation
  have sound {a b : IO₁}
      (proof : Spec.ProgramDerivation (fun _ _ : IO₁ => False) R a b) :
      Spec.ProgramSyntax.signature a = Spec.ProgramSyntax.signature b ∧
      ((a.domain plans).Holds M D s → i < a.io.Bout →
        observe₁ M plans s a i = observe₁ M plans s b i) := by
    induction proof with
    | refl => exact ⟨rfl, fun _ _ => rfl⟩
    | «syntax» hs hc _ => exact ⟨hs, fun _ _ => by cases hc; rfl⟩
    | structural _ hf => exact hf.elim
    | numerical hs hn =>
        refine ⟨hs, fun hd hi => ?_⟩
        obtain ⟨a, b, ha, hb, ho, _, _⟩ := hn.2.2 α M D hM plans s hd
        simp only [observe₁, ha, hb, Option.map_some]
        exact congrArg some (ho ⟨i, hi⟩)
    | @symm a b _ ih =>
        refine ⟨ih.1.symm, fun hd hi => ?_⟩
        have domains := congrArg (fun sig => sig.2.2) ih.1
        change a.domain = b.domain at domains
        have sizes := congrArg (fun sig => sig.1.Bout) ih.1
        change a.io.Bout = b.io.Bout at sizes
        exact (ih.2 (domains.symm ▸ hd) (sizes.symm ▸ hi)).symm
    | @trans a b c _ _ ih₁ ih₂ =>
        refine ⟨ih₁.1.trans ih₂.1, fun hd hi => ?_⟩
        have domains := congrArg (fun sig => sig.2.2) ih₁.1
        change a.domain = b.domain at domains
        have sizes := congrArg (fun sig => sig.1.Bout) ih₁.1
        change a.io.Bout = b.io.Bout at sizes
        exact (ih₁.2 hd hi).trans (ih₂.2 (domains ▸ hd) (sizes ▸ hi))
  exact (sound derivation).2 hd hi

end Scheduled
end VeriTile.Triton.FP
