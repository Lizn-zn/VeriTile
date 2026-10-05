import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Meta.Specification

/-! Contextual scalar rewrites in fp32-profiled algorithm-typed source.
The checks refer to actual operands after the indicated prefix. Matrix products,
reductions and loops in the surrounding source retain their operation order. -/
namespace VeriTile.Triton.FP.ProfiledRewrite
open Structural Guarded
open GuardedRewrite (sameShape sameShape_left sameShape_right)

def engine {α : Type} (M : Algebra α) : Algebra α := M.withDefaultPrecision .fp32

/-- Locate a register assignment in the source, counting repeated writes from
zero. A source edit before the assignment does not change the requested site. -/
def prefixBefore (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) : List ComputeStmt :=
  match code with
  | [] => []
  | st :: rest =>
    match st with
    | .assign _ _ out _ =>
      if out = name then
        match occurrence with
        | 0 => []
        | n + 1 => st :: prefixBefore rest name n
      else st :: prefixBefore rest name occurrence
    | _ => st :: prefixBefore rest name occurrence

inductive Site where
  | add (before : List ComputeStmt) (shape : TileShape) (left right : Op .real shape)
  | div (before : List ComputeStmt) (n : Nat) (rest : TileShape)
      (numerator : Op .real (n :: rest)) (denominator : ℝ)

def Site.Holds {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (s : State α) : Site → Prop
  | .add before _ left right =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b, Structural.evalOp (engine M) none left t = some a →
        Structural.evalOp (engine M) none right t = some b →
        (∀ i, D .finite (a i)) ∧ (∀ i, D .finite (b i))
  | .div before _ _ numerator denominator =>
    ∀ t, run (engine M) before s = some t →
      ∀ a, Structural.evalOp (engine M) none numerator t = some a →
        (∀ i, D .finite (a i)) ∧
        D .finite (M.literal (some .fp32) .real denominator) ∧
        D .nonzero (M.literal (some .fp32) .real denominator)

structure Program where
  kernel : ComputeKernel
  domain : List Site

def Equivalent (R : Spec.Assumptions GuardedFragment) (lhs rhs : Program) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ s, (∀ site ∈ lhs.domain, site.Holds M D s) →
      Structural.exec (engine M) lhs.kernel s = Structural.exec (engine M) rhs.kernel s

instance : Spec.ProgramSyntax Program where
  Statement := GuardedFragment
  Signature := (List RegionName × List RegionName) × List Site
  signature p := ((p.kernel.inputs, p.kernel.outputs), p.domain)
  body p := [⟨[], p.kernel.surfaceBody⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  sameContext := fun lhs rhs => lhs = rhs

private theorem eval_add {α : Type} [Inhabited α] (M : Algebra α)
    (bc : Broadcast a b c) (x : Op .real a) (y : Op .real b) (s : State α) :
    Structural.evalOp M none (.add .real bc x y) s =
      ((Structural.evalOp M none x s).bind fun vx =>
        (Structural.evalOp M none y s).bind fun vy =>
          some (bop (M.binary none .real .add) bc vx vy)) := rfl

/-- Pointwise addition commutation, lifted through an arbitrary prefix/suffix. -/
theorem add_commute {α : Type} [Inhabited α] (R : ScalarArithmetic.Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (before after : List ComputeStmt) (out : RegName)
    (a b : Op .real shape) (hd : (Site.add before shape a b).Holds M D s) :
    run (engine M) (before ++ .assign .real shape out (.alg (.add .real (sameShape shape) a b)) :: after) s =
    run (engine M) (before ++ .assign .real shape out (.alg (.add .real (sameShape shape) b a)) :: after) s := by
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr, eval_add]
    cases ha : Structural.evalOp (engine M) none a t <;>
      cases hb : Structural.evalOp (engine M) none b t
    all_goals simp only [Option.bind_none, Option.bind_some]
    rename_i va vb
    have hvalues : bop ((engine M).binary none .real .add) (sameShape shape) va vb =
        bop ((engine M).binary none .real .add) (sameShape shape) vb va := by
      funext i
      simp only [bop, sameShape_left, sameShape_right]
      exact ScalarArithmetic.add_comm R M D hM t _ _ ((hd t hp va vb ha hb).1 i)
        ((hd t hp va vb ha hb).2 i)
    rw [hvalues]

private theorem eval_div_const {α : Type} [Inhabited α] (M : Algebra α)
    (a : Op .real (n :: rest)) (b : ℝ) (s : State α) :
    Structural.evalOp M none (.div .real .scalarR a (.const b)) s =
      (Structural.evalOp M none a s).bind (fun va => some (fun i =>
        M.binary none .real .div (va i) (M.literal none .real b))) := rfl

private theorem eval_mul_rcp {α : Type} [Inhabited α] (M : Algebra α)
    (a : Op .real (n :: rest)) (b : ℝ) (s : State α) :
    Structural.evalOp M none (.mul .real .scalarR a (.div .real .nil (.const 1) (.const b))) s =
      (Structural.evalOp M none a s).bind (fun va => some (fun i =>
        M.binary none .real .mul (va i)
          (M.binary none .real .div (M.literal none .real 1) (M.literal none .real b)))) := rfl

/-- Divide a tile by a scalar versus multiply it by the scalar reciprocal. -/
theorem div_mul_rcp {α : Type} [Inhabited α] (R : ScalarArithmetic.Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (before after : List ComputeStmt) (out : RegName)
    (a : Op .real (n :: rest)) (b : ℝ) (hd : (Site.div before n rest a b).Holds M D s) :
    run (engine M) (before ++ .assign .real (n :: rest) out
      (.alg (.div .real .scalarR a (.const b))) :: after) s =
    run (engine M) (before ++ .assign .real (n :: rest) out
      (.alg (.mul .real .scalarR a (.div .real .nil (.const 1) (.const b)))) :: after) s := by
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr, eval_div_const, eval_mul_rcp]
    cases ha : Structural.evalOp (engine M) none a t with
    | none => rfl
    | some va =>
      obtain ⟨hva, hb, hn⟩ := hd t hp va ha
      have hv : (fun i => (engine M).binary none .real .div (va i) ((engine M).literal none .real b)) =
          (fun i => (engine M).binary none .real .mul (va i)
            ((engine M).binary none .real .div ((engine M).literal none .real 1)
              ((engine M).literal none .real b))) := by
        funext i
        exact ScalarArithmetic.div_mul_rcp R M D hM t _ _ (hva i) hb hn
      simp only [Option.bind_some]
      rw [hv]

/-- One changed statement in its original execution context. The prefix is
retained so guards describe actual intermediate values, not fresh inputs. -/
def Contextual {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (before after : List ComputeStmt) (lhs rhs : ComputeStmt) : Prop :=
  run (engine M) (before ++ lhs :: after) s = run (engine M) (before ++ rhs :: after) s

@[spec_rule] theorem contextual_add_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a b : Op .real shape)
    (hd : (Site.add before shape a b).Holds M D s) :
    Contextual M s before after
      (.assign .real shape out (.alg (.add .real (sameShape shape) a b)))
      (.assign .real shape out (.alg (.add .real (sameShape shape) b a))) :=
  add_commute R M D hM s before after out a b hd

@[spec_rule] theorem contextual_div_mul_rcp {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a : Op .real (n :: rest)) (b : ℝ)
    (hd : (Site.div before n rest a b).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.alg (.div .real .scalarR a (.const b))))
      (.assign .real (n :: rest) out (.alg (.mul .real .scalarR a (.div .real .nil (.const 1) (.const b))))) :=
  div_mul_rcp R M D hM s before after out a b hd

end VeriTile.Triton.FP.ProfiledRewrite
