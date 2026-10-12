import VeriTile.Triton.Float.ProfiledRewrite

/-! Composing admitted rewrites inside profiled expressions. Each relation is
checked after the actual common source prefix. Congruence preserves operand
order, dtype, eager branch evaluation and failure; it supplies no floating law.
Import this module with RewriteTactics to use these lemmas through fp_prove. -/
namespace VeriTile.Triton.FP.ProfiledRewrite
open Structural Guarded

/-- Equal expression evaluation at every state reached by the given prefix. -/
def OpEquivalent {α : Type} [Inhabited α] (M : Algebra α) (s : State α)
    (before : List ComputeStmt) (lhs rhs : Op dtype shape) : Prop :=
  ∀ t, run (engine M) before s = some t →
    Structural.evalOp (engine M) none lhs t = Structural.evalOp (engine M) none rhs t

@[spec_rule] theorem op_refl {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a : Op dtype shape) :
    OpEquivalent M s before a a := fun _ _ => rfl

/-- Changing a block pointer's base by an equal integer expression preserves
its offsets, layout and evaluation failure. This supplies no floating law. -/
@[spec_rule] theorem op_block_ptr_base {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (before : List ComputeStmt)
    (r : RegionName) (a b : Op .nat []) (parent block : TileShape)
    (strides : List Nat) (offsets : List (Op .nat []))
    (h : OpEquivalent M s before a b) :
    OpEquivalent M s before (.makeBlockPtrDynOffsets r a parent block strides offsets)
      (.makeBlockPtrDynOffsets r b parent block strides offsets) := by
  intro t ht
  simp only [Structural.evalOp]
  rw [h t ht]

/-- Lift expression equality through an unchanged assignment and source suffix. -/
@[spec_rule] theorem contextual_assign_op {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (out : RegName)
    (a b : Op dtype shape) (h : OpEquivalent M s before a b) :
    Contextual M s before after (.assign dtype shape out (.alg a))
      (.assign dtype shape out (.alg b)) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr]
    rw [h t hp]

/-- The destination, mask and stored dtype are identical on both sides. -/
@[spec_rule] theorem contextual_store_op {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (mem : MemAccess dtype shape)
    (mask : MaskOpt dtype shape) (a b : Op dtype shape)
    (h : OpEquivalent M s before a b) :
    Contextual M s before after (.store dtype shape mem (.alg a) mask)
      (.store dtype shape mem (.alg b) mask) := by
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr]
    rw [h t hp]

/-- Both arms remain eagerly evaluated; this is not short-circuit branching. -/
@[spec_rule] theorem op_where {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (c c' : Op .bool shape)
    (a a' b b' : Op dtype shape)
    (hc : OpEquivalent M s before c c')
    (ha : OpEquivalent M s before a a') (hb : OpEquivalent M s before b b') :
    OpEquivalent M s before (.where c a b) (.where c' a' b') := by
  intro t ht
  change (Structural.evalOp (engine M) none c t >>= fun vc =>
    Structural.evalOp (engine M) none a t >>= fun va =>
    Structural.evalOp (engine M) none b t >>= fun vb =>
      some (fun i => if vc i then va i else vb i)) =
    (Structural.evalOp (engine M) none c' t >>= fun vc =>
    Structural.evalOp (engine M) none a' t >>= fun va =>
    Structural.evalOp (engine M) none b' t >>= fun vb =>
      some (fun i => if vc i then va i else vb i))
  rw [hc t ht, ha t ht, hb t ht]

@[spec_rule] theorem op_add {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (d : NumericDType dtype)
    (bc : Broadcast sa sb shape) (a a' : Op dtype sa) (b b' : Op dtype sb)
    (ha : OpEquivalent M s before a a') (hb : OpEquivalent M s before b b') :
    OpEquivalent M s before (.add d bc a b) (.add d bc a' b') := by
  intro t ht
  change (Structural.evalOp (engine M) none a t >>= fun va =>
    Structural.evalOp (engine M) none b t >>= fun vb =>
      some (bop (numeric (engine M) none .add d) bc va vb)) =
    (Structural.evalOp (engine M) none a' t >>= fun va =>
    Structural.evalOp (engine M) none b' t >>= fun vb =>
      some (bop (numeric (engine M) none .add d) bc va vb))
  rw [ha t ht, hb t ht]

@[spec_rule] theorem op_sub {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (d : NumericDType dtype)
    (bc : Broadcast sa sb shape) (a a' : Op dtype sa) (b b' : Op dtype sb)
    (ha : OpEquivalent M s before a a') (hb : OpEquivalent M s before b b') :
    OpEquivalent M s before (.sub d bc a b) (.sub d bc a' b') := by
  intro t ht
  change (Structural.evalOp (engine M) none a t >>= fun va =>
    Structural.evalOp (engine M) none b t >>= fun vb =>
      some (bop (numeric (engine M) none .sub d) bc va vb)) =
    (Structural.evalOp (engine M) none a' t >>= fun va =>
    Structural.evalOp (engine M) none b' t >>= fun vb =>
      some (bop (numeric (engine M) none .sub d) bc va vb))
  rw [ha t ht, hb t ht]

@[spec_rule] theorem op_mul {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (d : NumericDType dtype)
    (bc : Broadcast sa sb shape) (a a' : Op dtype sa) (b b' : Op dtype sb)
    (ha : OpEquivalent M s before a a') (hb : OpEquivalent M s before b b') :
    OpEquivalent M s before (.mul d bc a b) (.mul d bc a' b') := by
  intro t ht
  change (Structural.evalOp (engine M) none a t >>= fun va =>
    Structural.evalOp (engine M) none b t >>= fun vb =>
      some (bop (numeric (engine M) none .mul d) bc va vb)) =
    (Structural.evalOp (engine M) none a' t >>= fun va =>
    Structural.evalOp (engine M) none b' t >>= fun vb =>
      some (bop (numeric (engine M) none .mul d) bc va vb))
  rw [ha t ht, hb t ht]

@[spec_rule] theorem op_div {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (d : NumericDType dtype)
    (bc : Broadcast sa sb shape) (a a' : Op dtype sa) (b b' : Op dtype sb)
    (ha : OpEquivalent M s before a a') (hb : OpEquivalent M s before b b') :
    OpEquivalent M s before (.div d bc a b) (.div d bc a' b') := by
  intro t ht
  change (Structural.evalOp (engine M) none a t >>= fun va =>
    Structural.evalOp (engine M) none b t >>= fun vb =>
      some (bop (numeric (engine M) none .div d) bc va vb)) =
    (Structural.evalOp (engine M) none a' t >>= fun va =>
    Structural.evalOp (engine M) none b' t >>= fun vb =>
      some (bop (numeric (engine M) none .div d) bc va vb))
  rw [ha t ht, hb t ht]

/-- The numerical leaf is exactly the admitted scalar div_mul_rcp atom.
Finite numerator lanes and a finite nonzero denominator are checked at this
prefix, even when the rewrite is nested inside an eagerly evaluated branch. -/
@[spec_rule] theorem op_div_scalar_mul_rcp {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before : List ComputeStmt)
    (a : Op .real (n :: rest)) (b : Op .real [])
    (hd : (Site.divScalar before n rest a b).Holds M D s) :
    OpEquivalent M s before (.div .real .scalarR a b)
      (.mul .real .scalarR a (.div .real .nil (.const 1) b)) := by
  intro t ht
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
  all_goals simp only [bind, Option.bind_none, Option.bind_some]
  rename_i va vb
  obtain ⟨hva, hvb, hn⟩ := hd t ht va vb ha hb
  congr 1
  funext i
  exact ScalarArithmetic.div_mul_rcp R M D hM t _ _ (hva i) hvb hn

end VeriTile.Triton.FP.ProfiledRewrite
