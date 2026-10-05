import VeriTile.Triton.Float.ScalarArithmeticLaws

/-! Contextual fp32 rewrites under the domains of the admitted scalar atoms.
The domain names the actual expressions at their program point, including
intermediate products and exponentials. Shapes remain symbolic.

These are contextual execution equalities: the two runs return the same state
or both fail. They preserve successful execution, but do not independently
assert that every surrounding instruction is supported by the interpreter. -/

namespace VeriTile.Triton.FP.GuardedRewrite
open Structural Guarded

inductive Operation where
  | add | mul
  deriving DecidableEq

def sameShape : (shape : TileShape) → Broadcast shape shape shape
  | [] => .nil
  | _ :: rest => .consSame (sameShape rest)

@[simp] theorem sameShape_left (shape : TileShape) (i : TileIndex shape) :
    (sameShape shape).leftIndex i = i := by
  induction shape with
  | nil => rfl
  | cons n rest ih => simp [sameShape, ih]

@[simp] theorem sameShape_right (shape : TileShape) (i : TileIndex shape) :
    (sameShape shape).rightIndex i = i := by
  induction shape with
  | nil => rfl
  | cons n rest ih => simp [sameShape, ih]

def expression (op : Operation) (a b : Op .real shape) : Op .real shape :=
  match op with
  | .add => .add .real (sameShape shape) a b
  | .mul => .mul .real (sameShape shape) a b

def assignment (op : Operation) (out : RegName) (a b : Op .real shape) : ComputeStmt :=
  .assign .real shape out (.compute (.alg .fp32 (expression op a b)))

/-- A finite check at a concrete program point. The prefix is part of the
condition, so register overwrites and intermediate expressions are retained. -/
structure Site where
  before : List ComputeStmt
  shape : TileShape
  left : Op .real shape
  right : Op .real shape

/-- Check every value actually evaluated at the rewrite site. A failed prefix
or operand evaluation remains a failed execution on both sides of the rewrite;
it is not used as a premise claiming that the program successfully ran. -/
def Site.Holds {α : Type} [Inhabited α] (site : Site) (M : Algebra α)
    (D : Domain α) (s : State α) : Prop :=
  ∀ t, run M site.before s = some t →
    ∀ a b, Structural.evalOp M (some .fp32) site.left t = some a →
      Structural.evalOp M (some .fp32) site.right t = some b →
        (∀ i, D .finite (a i)) ∧ (∀ i, D .finite (b i))

structure Program where
  kernel : ComputeKernel
  domain : List Site

def Equivalent (R : Spec.Assumptions GuardedFragment) (lhs rhs : Program) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ s, (∀ site ∈ lhs.domain, site.Holds M D s) →
      Structural.exec M lhs.kernel s = Structural.exec M rhs.kernel s

instance : Spec.ProgramSyntax Program where
  Statement := GuardedFragment
  Signature := (List RegionName × List RegionName) × List Site
  signature p := ((p.kernel.inputs, p.kernel.outputs), p.domain)
  body p := [⟨[], p.kernel.surfaceBody⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  sameContext := fun lhs rhs => lhs = rhs

private theorem eval_add {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (bc : Broadcast a b c)
    (x : Op .real a) (y : Op .real b) (s : State α) :
    Structural.evalOp M p (.add .real bc x y) s =
      ((Structural.evalOp M p x s).bind fun vx =>
        (Structural.evalOp M p y s).bind fun vy =>
          some (bop (M.binary p .real .add) bc vx vy)) := rfl

private theorem eval_mul {α : Type} [Inhabited α] (M : Algebra α)
    (p : Option ComputeDType) (bc : Broadcast a b c)
    (x : Op .real a) (y : Op .real b) (s : State α) :
    Structural.evalOp M p (.mul .real bc x y) s =
      ((Structural.evalOp M p x s).bind fun vx =>
        (Structural.evalOp M p y s).bind fun vy =>
          some (bop (M.binary p .real .mul) bc vx vy)) := rfl

private theorem assignment_add_commute {α : Type} [Inhabited α] (R : ScalarArithmetic.Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (out : RegName) (a b : Op .real shape)
    (hd : ∀ va vb, Structural.evalOp M (some .fp32) a s = some va →
      Structural.evalOp M (some .fp32) b s = some vb →
      (∀ i, D .finite (va i)) ∧ (∀ i, D .finite (vb i))) :
    step M (assignment .add out a b) s = step M (assignment .add out b a) s := by
  simp only [assignment, expression, step, Structural.evalExpr, Structural.evalComputeOp,
    ComputeDType.eraseDType, eval_add]
  cases ha : Structural.evalOp M (some .fp32) a s <;>
    cases hb : Structural.evalOp M (some .fp32) b s
  all_goals simp only [Option.bind_none, Option.bind_some]
  rename_i va vb
  obtain ⟨hva, hvb⟩ := hd va vb ha hb
  apply congrArg (fun v => some (s.setReg out .real shape v))
  funext i
  simp only [bop, sameShape_left, sameShape_right]
  exact ScalarArithmetic.add_comm R M D hM s _ _ (hva i) (hvb i)

/-- Lift guarded add commutation through the unchanged prefix and suffix.
Only this scalar atom is consumed, at the specified actual operands. -/
theorem add_commute {α : Type} [Inhabited α] (R : ScalarArithmetic.Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (site : Site) (out : RegName) (after : List ComputeStmt)
    (hd : site.Holds M D s) :
    run M (site.before ++ assignment .add out site.left site.right :: after) s =
      run M (site.before ++ assignment .add out site.right site.left :: after) s := by
  rw [run_append, run_append]
  cases hp : run M site.before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run]
    rw [assignment_add_commute R M D hM t out site.left site.right (hd t hp)]

private theorem assignment_mul_commute {α : Type} [Inhabited α] (R : ScalarArithmetic.Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (out : RegName) (a b : Op .real shape)
    (hd : ∀ va vb, Structural.evalOp M (some .fp32) a s = some va →
      Structural.evalOp M (some .fp32) b s = some vb →
      (∀ i, D .finite (va i)) ∧ (∀ i, D .finite (vb i))) :
    step M (assignment .mul out a b) s = step M (assignment .mul out b a) s := by
  simp only [assignment, expression, step, Structural.evalExpr, Structural.evalComputeOp,
    ComputeDType.eraseDType, eval_mul]
  cases ha : Structural.evalOp M (some .fp32) a s <;>
    cases hb : Structural.evalOp M (some .fp32) b s
  all_goals simp only [Option.bind_none, Option.bind_some]
  rename_i va vb
  obtain ⟨hva, hvb⟩ := hd va vb ha hb
  apply congrArg (fun v => some (s.setReg out .real shape v))
  funext i
  simp only [bop, sameShape_left, sameShape_right]
  exact ScalarArithmetic.mul_comm R M D hM s _ _ (hva i) (hvb i)

/-- Lift guarded mul commutation through the unchanged prefix and suffix.
Only this scalar atom is consumed, at the specified actual operands. -/
theorem mul_commute {α : Type} [Inhabited α] (R : ScalarArithmetic.Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) (site : Site) (out : RegName) (after : List ComputeStmt)
    (hd : site.Holds M D s) :
    run M (site.before ++ assignment .mul out site.left site.right :: after) s =
      run M (site.before ++ assignment .mul out site.right site.left :: after) s := by
  rw [run_append, run_append]
  cases hp : run M site.before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run]
    rw [assignment_mul_commute R M D hM t out site.left site.right (hd t hp)]

end VeriTile.Triton.FP.GuardedRewrite
