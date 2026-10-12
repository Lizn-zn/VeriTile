import VeriTile.Triton.Float.ProfiledRewrite

/-! Contextual rewriting of a selected branch. Operand guards are required only
in reachable states of that arm. The other arm, failures, prefix and suffix
are preserved without numerical assumptions. -/
namespace VeriTile.Triton.FP.ProfiledRewrite
open Structural Guarded

theorem contextual_then {α : Type} [Inhabited α] (M : Algebra α)
    (D : Domain α) (s : State α) (before after : List ComputeStmt)
    (condition : Op .bool []) (lhs rhs otherwise : List ComputeStmt) (inner : Site)
    (hd : (Site.branch before condition true inner).Holds M D s)
    (hbody : ∀ t, inner.Holds M D t → run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.ifThenElse condition lhs otherwise)
      (.ifThenElse condition rhs otherwise) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step]
    cases hc : Structural.evalOp (engine M) none condition t with
    | none => rfl
    | some value =>
      simp only [bind, Option.bind_some]
      cases hv : value PUnit.unit with
      | false => rfl
      | true => simp only [↓reduceIte]; rw [hbody t (hd t hp value hc hv)]

theorem contextual_else {α : Type} [Inhabited α] (M : Algebra α)
    (D : Domain α) (s : State α) (before after : List ComputeStmt)
    (condition : Op .bool []) (otherwise lhs rhs : List ComputeStmt) (inner : Site)
    (hd : (Site.branch before condition false inner).Holds M D s)
    (hbody : ∀ t, inner.Holds M D t → run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.ifThenElse condition otherwise lhs)
      (.ifThenElse condition otherwise rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step]
    cases hc : Structural.evalOp (engine M) none condition t with
    | none => rfl
    | some value =>
      simp only [bind, Option.bind_some]
      cases hv : value PUnit.unit with
      | false => simp only [Bool.false_eq_true, ↓reduceIte]; rw [hbody t (hd t hp value hc hv)]
      | true => rfl

/-- Rewriting the body of a one-arm conditional checks the declared domain
only after the common prefix and only when that body is selected. -/
theorem contextual_if_then {α : Type} [Inhabited α] (M : Algebra α)
    (D : Domain α) (s : State α) (before after : List ComputeStmt)
    (condition : Op .bool []) (lhs rhs : List ComputeStmt) (inner : Site)
    (hd : (Site.branch before condition true inner).Holds M D s)
    (hbody : ∀ t, inner.Holds M D t → run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.ifThen condition lhs) (.ifThen condition rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step]
    cases hc : Structural.evalOp (engine M) none condition t with
    | none => rfl
    | some value =>
      simp only [bind, Option.bind_some]
      cases hv : value PUnit.unit with
      | false => rfl
      | true => simp only [↓reduceIte]; rw [hbody t (hd t hp value hc hv)]


end VeriTile.Triton.FP.ProfiledRewrite
