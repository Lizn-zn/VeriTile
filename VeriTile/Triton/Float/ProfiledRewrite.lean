import VeriTile.Triton.Float.GuardedRewrite
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.Loop
import VeriTile.Meta.Specification

/-! Contextual scalar rewrites in fp32-profiled algorithm-typed source.
The checks refer to actual operands after the indicated prefix. Matrix products,
reductions and loops in the surrounding source retain their operation order. -/
namespace VeriTile.Triton.FP.ProfiledRewrite
open Structural Guarded
open GuardedRewrite (sameShape sameShape_left sameShape_right)

def engine {α : Type} (M : Algebra α) : Algebra α := M.withDefaultPrecision .fp32

abbrev prefixBefore (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  Source.prefixBefore code name occurrence

inductive Site where
  | add (before : List ComputeStmt) (shape : TileShape) (left right : Op .real shape)
  /-- Addition whose operands are evaluated inside an explicit fp32 computation. -/
  | addFP32 (before : List ComputeStmt) (shape : TileShape) (left right : Op .real shape)
  | addScalar (before : List ComputeStmt) (n : Nat) (rest : TileShape)
      (tile : Op .real (n :: rest)) (scalar : Op .real [])
  | mulScalar (before : List ComputeStmt) (n : Nat) (rest : TileShape)
      (tile : Op .real (n :: rest)) (scalar : Op .real [])
  /-- Scalar multiplication whose operands retain their explicit fp32 context. -/
  | mulScalarFP32 (before : List ComputeStmt) (n : Nat) (rest : TileShape)
      (tile : Op .real (n :: rest)) (scalar : Op .real [])
  /-- Broadcast multiplication under the source's default fp32 profile. -/
  | mulBroadcast (before : List ComputeStmt) (sa sb : TileShape)
      (left : Op .real sa) (right : Op .real sb)
  /-- Broadcast multiplication evaluated in the source's explicit fp32 context. -/
  | mulBroadcastFP32 (before : List ComputeStmt) (sa sb : TileShape)
      (left : Op .real sa) (right : Op .real sb)
  | div (before : List ComputeStmt) (n : Nat) (rest : TileShape)
      (numerator : Op .real (n :: rest)) (denominator : ℝ)
  | divScalar (before : List ComputeStmt) (n : Nat) (rest : TileShape)
      (numerator : Op .real (n :: rest)) (denominator : Op .real [])
  /-- Broadcast division inside an explicit fp32 computation. -/
  | divBroadcast (before : List ComputeStmt) (sa : TileShape) (n : Nat) (rest : TileShape)
      (numerator : Op .real sa) (denominator : Op .real (n :: rest))
  /-- Check a local site in each reachable iteration of the original body.
  The bounds are evaluated once after `before`; the loop index is bound before
  checking the inner site's own prefix. Nested range sites are allowed. -/
  | range (before : List ComputeStmt) (index : RegName)
      (start stop stride : Op .nat []) (body : List ComputeStmt) (inner : Site)
  /-- The same reached-iteration check for signed bounds and a signed index.
  In particular, negative indices are not converted to natural numbers. -/
  | intRange (before : List ComputeStmt) (index : RegName)
      (start stop : Op .int []) (stride : Op .nat [])
      (body : List ComputeStmt) (inner : Site)
  /-- Check an inner site only when this arm is actually selected. The condition
  and the inner prefix are evaluated in the state after the common prefix. -/
  | branch (before : List ComputeStmt) (condition : Op .bool [])
      (selected : Bool) (inner : Site)

def Site.Holds {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (s : State α) : Site → Prop
  | .add before _ left right =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b, Structural.evalOp (engine M) none left t = some a →
        Structural.evalOp (engine M) none right t = some b →
        (∀ i, D .finite (a i)) ∧ (∀ i, D .finite (b i))
  | .addFP32 before _ left right =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b, Structural.evalOp (engine M) (some .fp32) left t = some a →
        Structural.evalOp (engine M) (some .fp32) right t = some b →
        (∀ i, D .finite (a i)) ∧ (∀ i, D .finite (b i))
  | .addScalar before _ _ tile scalar | .mulScalar before _ _ tile scalar =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b, Structural.evalOp (engine M) none tile t = some a →
        Structural.evalOp (engine M) none scalar t = some b →
        (∀ i, D .finite (a i)) ∧ D .finite (b PUnit.unit)
  | .mulScalarFP32 before _ _ tile scalar =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b, Structural.evalOp (engine M) (some .fp32) tile t = some a →
        Structural.evalOp (engine M) (some .fp32) scalar t = some b →
        (∀ i, D .finite (a i)) ∧ D .finite (b PUnit.unit)
  | .mulBroadcast before _ _ left right =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b, Structural.evalOp (engine M) none left t = some a →
        Structural.evalOp (engine M) none right t = some b →
        (∀ i, D .finite (a i)) ∧ (∀ i, D .finite (b i))
  | .mulBroadcastFP32 before _ _ left right =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b, Structural.evalOp (engine M) (some .fp32) left t = some a →
        Structural.evalOp (engine M) (some .fp32) right t = some b →
        (∀ i, D .finite (a i)) ∧ (∀ i, D .finite (b i))
  | .div before _ _ numerator denominator =>
    ∀ t, run (engine M) before s = some t →
      ∀ a, Structural.evalOp (engine M) none numerator t = some a →
        (∀ i, D .finite (a i)) ∧
        D .finite (M.literal (some .fp32) .real denominator) ∧
        D .nonzero (M.literal (some .fp32) .real denominator)
  | .divScalar before _ _ numerator denominator =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b, Structural.evalOp (engine M) none numerator t = some a →
        Structural.evalOp (engine M) none denominator t = some b →
        (∀ i, D .finite (a i)) ∧ D .finite (b PUnit.unit) ∧ D .nonzero (b PUnit.unit)
  | .divBroadcast before _ _ _ numerator denominator =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b, Structural.evalOp (engine M) (some .fp32) numerator t = some a →
        Structural.evalOp (engine M) (some .fp32) denominator t = some b →
        (∀ i, D .finite (a i)) ∧ (∀ i, D .finite (b i)) ∧ (∀ i, D .nonzero (b i))
  | .range before index start stop stride body inner =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b d, Structural.evalOp (engine M) none start t = some a →
        Structural.evalOp (engine M) none stop t = some b →
        Structural.evalOp (engine M) none stride t = some d →
        ∀ cur u, RangeReachable (engine M) index (b PUnit.unit) (d PUnit.unit)
          body (a PUnit.unit) t cur u → d PUnit.unit ≠ 0 → cur < b PUnit.unit →
          inner.Holds M D (u.setReg index .nat [] (fun _ => cur))
  | .intRange before index start stop stride body inner =>
    ∀ t, run (engine M) before s = some t →
      ∀ a b d, Structural.evalOp (engine M) none start t = some a →
        Structural.evalOp (engine M) none stop t = some b →
        Structural.evalOp (engine M) none stride t = some d →
        ∀ cur u, IntRangeReachable (engine M) index (b PUnit.unit) (d PUnit.unit)
          body (a PUnit.unit) t cur u → d PUnit.unit ≠ 0 → cur < b PUnit.unit →
          inner.Holds M D (u.setReg index .int [] (fun _ => cur))
  | .branch before condition selected inner =>
    ∀ t, run (engine M) before s = some t →
      ∀ value, Structural.evalOp (engine M) none condition t = some value →
        value PUnit.unit = selected → inner.Holds M D t

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

private theorem eval_mul {α : Type} [Inhabited α] (M : Algebra α)
    (bc : Broadcast a b c) (x : Op .real a) (y : Op .real b) (s : State α) :
    Structural.evalOp M none (.mul .real bc x y) s =
      ((Structural.evalOp M none x s).bind fun vx =>
        (Structural.evalOp M none y s).bind fun vy =>
          some (bop (M.binary none .real .mul) bc vx vy)) := rfl

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

/-- Commute an explicit fp32 addition after checking its actual operands.
Subexpressions, including reductions, keep their fp32 evaluation context. -/
@[spec_rule] theorem contextual_add_fp32_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a b : Op .real shape)
    (hd : (Site.addFP32 before shape a b).Holds M D s) :
    Contextual M s before after
      (.assign .real shape out (.compute (.alg .fp32 (.add .real (sameShape shape) a b))))
      (.assign .real shape out (.compute (.alg .fp32 (.add .real (sameShape shape) b a)))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr, evalComputeOp]
    have eadd (x y : Op .real shape) :
        Structural.evalOp (engine M) (some .fp32) (.add .real (sameShape shape) x y) t =
          ((Structural.evalOp (engine M) (some .fp32) x t).bind fun vx =>
            (Structural.evalOp (engine M) (some .fp32) y t).bind fun vy =>
              some (bop ((engine M).binary (some .fp32) .real .add) (sameShape shape) vx vy)) := rfl
    erw [eadd, eadd]
    cases ha : Structural.evalOp (engine M) (some .fp32) a t <;>
      cases hb : Structural.evalOp (engine M) (some .fp32) b t
    all_goals simp only [Option.bind_none, Option.bind_some]
    rename_i va vb
    have hvalues : bop ((engine M).binary (some .fp32) .real .add) (sameShape shape) va vb =
        bop ((engine M).binary (some .fp32) .real .add) (sameShape shape) vb va := by
      funext i
      simp only [bop, sameShape_left, sameShape_right]
      exact ScalarArithmetic.add_comm R M D hM t _ _ ((hd t hp va vb ha hb).1 i)
        ((hd t hp va vb ha hb).2 i)
    rw [hvalues]

/-- A numerical body rewrite under the guards of each actual iteration.
The common prefix and suffix, all bounds, and every failed execution are
preserved. This supplies only control-flow congruence; `hbody` must still be
proved from the admitted atoms and the inner site's operand checks. -/
theorem contextual_range {α : Type} [Inhabited α] (M : Algebra α)
    (D : Domain α) (s : State α) (before after : List ComputeStmt)
    (index : RegName) (start stop stride : Op .nat [])
    (lhs rhs : List ComputeStmt) (inner : Site)
    (hd : (Site.range before index start stop stride lhs inner).Holds M D s)
    (hbody : ∀ t, inner.Holds M D t → run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.forRangeDyn index start stop stride lhs)
      (.forRangeDyn index start stop stride rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [bind, Option.bind_some, run, step]
    cases ha : Structural.evalOp (engine M) none start t with
    | none => rfl
    | some a =>
      simp only [Option.bind_some]
      cases hb : Structural.evalOp (engine M) none stop t with
      | none => rfl
      | some b =>
        simp only [Option.bind_some]
        cases hdt : Structural.evalOp (engine M) none stride t with
        | none => rfl
        | some d =>
          simp only [Option.bind_some]
          rw [range_congr_on_reachable (engine M) index (a PUnit.unit)
            (b PUnit.unit) (d PUnit.unit) lhs rhs t (fun cur u hr hz hc =>
              hbody _ (hd t hp a b d ha hb hdt cur u hr hz hc))]

/-- A source range with static natural bounds uses the same reached-state
domain as constant bound operations. Its syntax, suffix and failures remain. -/
theorem contextual_static_range {α : Type} [Inhabited α] (M : Algebra α)
    (D : Domain α) (s : State α) (before after : List ComputeStmt)
    (index : RegName) (start stop stride : Nat)
    (lhs rhs : List ComputeStmt) (inner : Site)
    (hd : (Site.range before index (.constNat start) (.constNat stop)
      (.constNat stride) lhs inner).Holds M D s)
    (hbody : ∀ t, inner.Holds M D t → run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.forRange index start stop stride lhs)
      (.forRange index start stop stride rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [bind, Option.bind_some, run, step]
    rw [range_congr_on_reachable (engine M) index start stop stride lhs rhs t
      (fun cur u hr hz hc => hbody _
        (hd t hp (fun _ => start) (fun _ => stop) (fun _ => stride)
          rfl rfl rfl cur u hr hz hc))]

/-- Lift a checked body rewrite through a signed range. Bounds are evaluated
once after the common prefix; the original body's reachable states supply
the guards. Signed indices, the suffix and execution failure are preserved. -/
theorem contextual_int_range {α : Type} [Inhabited α] (M : Algebra α)
    (D : Domain α) (s : State α) (before after : List ComputeStmt)
    (index : RegName) (start stop : Op .int []) (stride : Op .nat [])
    (lhs rhs : List ComputeStmt) (inner : Site)
    (hd : (Site.intRange before index start stop stride lhs inner).Holds M D s)
    (hbody : ∀ t, inner.Holds M D t → run (engine M) lhs t = run (engine M) rhs t) :
    Contextual M s before after (.forRangeInt index start stop stride lhs)
      (.forRangeInt index start stop stride rhs) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [bind, Option.bind_some, run, step]
    cases ha : Structural.evalOp (engine M) none start t with
    | none => rfl
    | some a =>
      simp only [Option.bind_some]
      cases hb : Structural.evalOp (engine M) none stop t with
      | none => rfl
      | some b =>
        simp only [Option.bind_some]
        cases hdt : Structural.evalOp (engine M) none stride t with
        | none => rfl
        | some d =>
          simp only [Option.bind_some]
          rw [rangeInt_congr_on_reachable (engine M) index (a PUnit.unit)
            (b PUnit.unit) (d PUnit.unit) lhs rhs t (fun cur u hr hz hc =>
              hbody _ (hd t hp a b d ha hb hdt cur u hr hz hc))]

/-- Commuting addition also swaps scalar-right with scalar-left broadcasting.
Both operands are evaluated after the actual prefix. Missing registers and
other execution failures agree, and the unchanged suffix keeps its mask and
memory effects. The only numerical step is the admitted scalar add_commute. -/
@[spec_rule] theorem contextual_add_scalar_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a : Op .real (n :: rest)) (b : Op .real [])
    (hd : (Site.addScalar before n rest a b).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.alg (.add .real .scalarR a b)))
      (.assign .real (n :: rest) out (.alg (.add .real .scalarL b a))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr, eval_add]
    cases ha : Structural.evalOp (engine M) none a t <;>
      cases hb : Structural.evalOp (engine M) none b t
    all_goals simp only [Option.bind_none, Option.bind_some]
    rename_i va vb
    have hvalues : bop ((engine M).binary none .real .add) .scalarR va vb =
        bop ((engine M).binary none .real .add) .scalarL vb va := by
      funext i
      exact ScalarArithmetic.add_comm R M D hM t _ _ ((hd t hp va vb ha hb).1 i)
        (hd t hp va vb ha hb).2
    rw [hvalues]

/-- Commute the addition inside a store value. The address expression, mask
and dtype are identical on both sides; only the evaluated value changes.
The same finite-operand guard as register addition is checked after the actual
prefix. No permission to change a memory access follows from the numeric atom. -/
@[spec_rule] theorem contextual_store_add_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (mem : MemAccess .real shape) (mask : MaskOpt .real shape) (a b : Op .real shape)
    (hd : (Site.add before shape a b).Holds M D s) :
    Contextual M s before after
      (.store .real shape mem (.alg (.add .real (sameShape shape) a b)) mask)
      (.store .real shape mem (.alg (.add .real (sameShape shape) b a)) mask) := by
  unfold Contextual
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

/-- Move a scalar multiplier from the left to the right without changing the
prefix, suffix or broadcast result shape. Both the tile and actual scalar
must be finite; a literal's spelling alone does not establish its domain in
an opaque numerical model. The numerical step uses only admitted mul_commute. -/
@[spec_rule] theorem contextual_mul_scalar_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a : Op .real (n :: rest)) (b : Op .real [])
    (hd : (Site.mulScalar before n rest a b).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.alg (.mul .real .scalarL b a)))
      (.assign .real (n :: rest) out (.alg (.mul .real .scalarR a b))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr, eval_mul]
    cases ha : Structural.evalOp (engine M) none a t <;>
      cases hb : Structural.evalOp (engine M) none b t
    all_goals simp only [Option.bind_none, Option.bind_some]
    rename_i va vb
    have hvalues : bop ((engine M).binary none .real .mul) .scalarL vb va =
        bop ((engine M).binary none .real .mul) .scalarR va vb := by
      funext i
      exact ScalarArithmetic.mul_comm R M D hM t _ _ (hd t hp va vb ha hb).2
        ((hd t hp va vb ha hb).1 i)
    rw [hvalues]

/-- Commute an explicitly fp32 tile/scalar multiplication, including computed
operands. Guards use the same precision as the operation; evaluation failures
and the complete surrounding context are preserved. -/
@[spec_rule] theorem contextual_mul_fp32_scalar_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a : Op .real (n :: rest)) (b : Op .real [])
    (hd : (Site.mulScalarFP32 before n rest a b).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.compute (.alg .fp32 (.mul .real .scalarR a b))))
      (.assign .real (n :: rest) out (.compute (.alg .fp32 (.mul .real .scalarL b a)))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr, evalComputeOp]
    have er : Structural.evalOp (engine M) (some .fp32) (.mul .real .scalarR a b) t =
        ((Structural.evalOp (engine M) (some .fp32) a t).bind fun va =>
          (Structural.evalOp (engine M) (some .fp32) b t).bind fun vb =>
            some (bop ((engine M).binary (some .fp32) .real .mul) .scalarR va vb)) := rfl
    have el : Structural.evalOp (engine M) (some .fp32) (.mul .real .scalarL b a) t =
        ((Structural.evalOp (engine M) (some .fp32) b t).bind fun vb =>
          (Structural.evalOp (engine M) (some .fp32) a t).bind fun va =>
            some (bop ((engine M).binary (some .fp32) .real .mul) .scalarL vb va)) := rfl
    erw [er, el]
    cases ha : Structural.evalOp (engine M) (some .fp32) a t <;>
      cases hb : Structural.evalOp (engine M) (some .fp32) b t
    all_goals simp only [Option.bind_none, Option.bind_some]
    rename_i va vb
    have hvalues : bop ((engine M).binary (some .fp32) .real .mul) .scalarR va vb =
        bop ((engine M).binary (some .fp32) .real .mul) .scalarL vb va := by
      funext i
      exact ScalarArithmetic.mul_comm R M D hM t _ _ ((hd t hp va vb ha hb).1 i)
        (hd t hp va vb ha hb).2
    rw [hvalues]

/-- A computed scalar denominator, including a reduction result. Guards are
checked after the actual prefix; neither evaluation nor nonzero is assumed
from the register name. The surrounding mask, memory and failure behavior stay
in the unchanged prefix and suffix. -/
@[spec_rule] theorem contextual_div_scalar_mul_rcp {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a : Op .real (n :: rest)) (b : Op .real [])
    (hd : (Site.divScalar before n rest a b).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.alg (.div .real .scalarR a b)))
      (.assign .real (n :: rest) out
        (.alg (.mul .real .scalarR a (.div .real .nil (.const 1) b)))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr]
    have ediv : Structural.evalOp (engine M) none (.div .real .scalarR a b) t =
        (Structural.evalOp (engine M) none a t >>= fun va =>
          Structural.evalOp (engine M) none b t >>= fun vb =>
            some (fun i => (engine M).binary none .real .div (va i) (vb PUnit.unit))) := rfl
    have emul : Structural.evalOp (engine M) none
        (.mul .real .scalarR a (.div .real .nil (.const 1) b)) t =
        (Structural.evalOp (engine M) none a t >>= fun va =>
          Structural.evalOp (engine M) none b t >>= fun vb =>
            some (fun i => (engine M).binary none .real .mul (va i)
              ((engine M).binary none .real .div ((engine M).literal none .real 1)
                (vb PUnit.unit)))) := by
      change (Structural.evalOp (engine M) none a t >>= fun va =>
        (Structural.evalOp (engine M) none b t >>= fun vb =>
          some (fun _ : TileIndex [] => (engine M).binary none .real .div
            ((engine M).literal none .real 1) (vb PUnit.unit))) >>= fun inv =>
          some (fun i => (engine M).binary none .real .mul (va i) (inv PUnit.unit))) = _
      cases Structural.evalOp (engine M) none a t <;>
        cases Structural.evalOp (engine M) none b t <;> rfl
    rw [ediv, emul]
    cases ha : Structural.evalOp (engine M) none a t <;>
      cases hb : Structural.evalOp (engine M) none b t
    all_goals simp only [bind, pure, Option.bind_none, Option.bind_some]
    rename_i va vb
    obtain ⟨hva, hvb, hn⟩ := hd t hp va vb ha hb
    congr 2
    funext i
    exact ScalarArithmetic.div_mul_rcp R M D hM t _ _ (hva i) hvb hn

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

/-- Commute a loaded tile and scalar parameter inside an explicit fp32 cast.
The cast remains an opaque operation on both sides. Register lookup is
precision independent, and the parameter literal is interpreted at fp32 in
both the domain and the computation; no cast identity is used. -/
@[spec_rule] theorem contextual_scaled_tile_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out tile : RegName) (scale : ℝ)
    (hd : (Site.mulScalar before n rest (.ref .real (n :: rest) tile)
      (.const scale)).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.compute (.alg .fp32
        (.castFloat .real .real (.mul .real .scalarR
          (.ref .real (n :: rest) tile) (.const scale))))))
      (.assign .real (n :: rest) out (.compute (.alg .fp32
        (.castFloat .real .real (.mul .real .scalarL
          (.const scale) (.ref .real (n :: rest) tile)))))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr, evalComputeOp]
    have hv : Structural.evalOp (engine M) (some .fp32)
        (.castFloat .real .real (.mul .real .scalarR
          (.ref .real (n :: rest) tile) (.const scale))) t =
      Structural.evalOp (engine M) (some .fp32)
        (.castFloat .real .real (.mul .real .scalarL
          (.const scale) (.ref .real (n :: rest) tile))) t := by
      cases ha : t.regs .real (n :: rest) tile with
      | none => simp [Structural.evalOp, ha]
      | some a =>
        obtain ⟨hfinite, hscale⟩ := hd t hp a
          (fun _ => M.literal (some .fp32) .real scale) ha rfl
        simp only [Structural.evalOp, FloatDType.toTileDType, ha, bind, pure, Option.bind_some]
        congr 1
        funext i
        exact congrArg (M.cast (some .fp32) .real .real)
          (ScalarArithmetic.mul_comm R M D hM t _ _ (hfinite i) hscale)
    erw [hv]
    rfl

/-- Commute tile/scalar multiplication inside an eager where expression. The
actual operands are guarded at the reached prefix, including unselected lanes;
the predicate, alternative, failures and complete statement context are retained. -/
@[spec_rule] theorem contextual_where_mul_scalar_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a : Op .real (n :: rest)) (b : Op .real [])
    (condition : Op .bool (n :: rest)) (otherwise : Op .real (n :: rest))
    (hd : (Site.mulScalar before n rest a b).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.alg (.where condition (.mul .real .scalarR a b) otherwise)))
      (.assign .real (n :: rest) out (.alg (.where condition (.mul .real .scalarL b a) otherwise))) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    have hmul : Structural.evalOp (engine M) none (.mul .real .scalarR a b) t =
        Structural.evalOp (engine M) none (.mul .real .scalarL b a) t := by
      change (do
        let va ← Structural.evalOp (engine M) none a t
        let vb ← Structural.evalOp (engine M) none b t
        pure (bop ((engine M).binary none .real .mul) .scalarR va vb)) =
        (do
          let vb ← Structural.evalOp (engine M) none b t
          let va ← Structural.evalOp (engine M) none a t
          pure (bop ((engine M).binary none .real .mul) .scalarL vb va))
      cases ha : Structural.evalOp (engine M) none a t <;>
        cases hb : Structural.evalOp (engine M) none b t
      all_goals simp only [bind, pure, Option.bind_none, Option.bind_some]
      rename_i va vb
      congr 1
      funext i
      exact ScalarArithmetic.mul_comm R M D hM t _ _ ((hd t hp va vb ha hb).1 i)
        (hd t hp va vb ha hb).2
    have hwhere : Structural.evalOp (engine M) none
        (.where condition (.mul .real .scalarR a b) otherwise) t =
        Structural.evalOp (engine M) none
          (.where condition (.mul .real .scalarL b a) otherwise) t := by
      change (do
        let c ← Structural.evalOp (engine M) none condition t
        let v ← Structural.evalOp (engine M) none (.mul .real .scalarR a b) t
        let f ← Structural.evalOp (engine M) none otherwise t
        pure (fun i => if c i then v i else f i)) = _
      rw [hmul]
      rfl
    simp only [Option.bind_some, run, step, Structural.evalExpr, hwhere]


end VeriTile.Triton.FP.ProfiledRewrite
