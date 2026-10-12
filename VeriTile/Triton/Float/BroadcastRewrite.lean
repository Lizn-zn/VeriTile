import VeriTile.Triton.Float.ProfiledRewrite

/-! Lifting admitted scalar multiplication and division/reciprocal relations
through arbitrary broadcasting inside an explicitly fp32 assignment. Guards inspect evaluated
operands after the actual source prefix, including every evaluated masked lane. -/
namespace VeriTile.Triton.FP.ProfiledRewrite
open Structural Guarded

/-- Equality of literal spellings before they enter the numerical backend. -/
@[spec_rule] theorem scientific_one : (1.0 : ℝ) = 1 := by norm_num

@[spec_rule] theorem contextual_div_broadcast_mul_rcp {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (bc : Broadcast sa (n :: rest) shape)
    (a : Op .real sa) (b : Op .real (n :: rest))
    (one : ℝ) (hone : one = 1)
    (hd : (Site.divBroadcast before sa n rest a b).Holds M D s) :
    Contextual M s before after
      (.assign .real shape out (.compute (.alg .fp32 (.div .real bc a b))))
      (.assign .real shape out (.compute (.alg .fp32
        (.mul .real bc a (.div .real .scalarL (.const one) b))))) := by
  subst one
  unfold Contextual
  rw [run_append, run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step, Structural.evalExpr, evalComputeOp]
    have ediv : Structural.evalOp (engine M) (some .fp32) (.div .real bc a b) t =
        (Structural.evalOp (engine M) (some .fp32) a t >>= fun va =>
          Structural.evalOp (engine M) (some .fp32) b t >>= fun vb =>
            some (fun i => (engine M).binary (some .fp32) .real .div
              (va (bc.leftIndex i)) (vb (bc.rightIndex i)))) := rfl
    have emul : Structural.evalOp (engine M) (some .fp32)
        (.mul .real bc a (.div .real .scalarL (.const 1) b)) t =
        (Structural.evalOp (engine M) (some .fp32) a t >>= fun va =>
          Structural.evalOp (engine M) (some .fp32) b t >>= fun vb =>
            some (fun i => (engine M).binary (some .fp32) .real .mul
              (va (bc.leftIndex i)) ((engine M).binary (some .fp32) .real .div
                ((engine M).literal (some .fp32) .real 1) (vb (bc.rightIndex i))))) := by
      change (Structural.evalOp (engine M) (some .fp32) a t >>= fun va =>
        (Structural.evalOp (engine M) (some .fp32) b t >>= fun vb =>
          some (fun i => (engine M).binary (some .fp32) .real .div
            ((engine M).literal (some .fp32) .real 1) (vb i))) >>= fun inv =>
          some (fun i => (engine M).binary (some .fp32) .real .mul
            (va (bc.leftIndex i)) (inv (bc.rightIndex i)))) = _
      cases Structural.evalOp (engine M) (some .fp32) a t <;>
        cases Structural.evalOp (engine M) (some .fp32) b t <;> rfl
    erw [ediv, emul]
    cases ha : Structural.evalOp (engine M) (some .fp32) a t <;>
      cases hb : Structural.evalOp (engine M) (some .fp32) b t
    all_goals simp only [bind, pure, Option.bind_none, Option.bind_some]
    rename_i va vb
    obtain ⟨hva, hvb, hn⟩ := hd t hp va vb ha hb
    congr 2
    funext i
    exact ScalarArithmetic.div_mul_rcp R M D hM t _ _
      (hva (bc.leftIndex i)) (hvb (bc.rightIndex i)) (hn (bc.rightIndex i))


/-- Exchange the input axes while retaining the same output shape. -/
def reverseBroadcast : Broadcast sa sb shape → Broadcast sb sa shape
  | .nil => .nil
  | .scalarL => .scalarR
  | .scalarR => .scalarL
  | .consSame bc => .consSame (reverseBroadcast bc)
  | .consL bc => .consR (reverseBroadcast bc)
  | .consR bc => .consL (reverseBroadcast bc)
  | .leadL bc => .leadR (reverseBroadcast bc)
  | .leadR bc => .leadL (reverseBroadcast bc)

@[simp] theorem reverseBroadcast_left (bc : Broadcast sa sb shape) (i : TileIndex shape) :
    (reverseBroadcast bc).leftIndex i = bc.rightIndex i := by
  induction bc <;> simp_all [reverseBroadcast]

@[simp] theorem reverseBroadcast_right (bc : Broadcast sa sb shape) (i : TileIndex shape) :
    (reverseBroadcast bc).rightIndex i = bc.leftIndex i := by
  induction bc <;> simp_all [reverseBroadcast]

/-- Lift the admitted mul_commute atom through arbitrary broadcasting. Both
operand tiles are checked after the actual common prefix; either evaluation
failure remains a failure in the other operand order. -/
@[spec_rule] theorem contextual_mul_broadcast_fp32_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (bc : Broadcast sa sb shape) (a : Op .real sa) (b : Op .real sb)
    (hd : (Site.mulBroadcastFP32 before sa sb a b).Holds M D s) :
    Contextual M s before after
      (.assign .real shape out (.compute (.alg .fp32 (.mul .real bc a b))))
      (.assign .real shape out (.compute (.alg .fp32 (.mul .real (reverseBroadcast bc) b a)))) := by
  unfold Contextual
  rw [run_append,run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some,run,step,Structural.evalExpr,evalComputeOp]
    have he : Structural.evalOp (engine M) (some .fp32) (.mul .real bc a b) t =
        Structural.evalOp (engine M) (some .fp32) (.mul .real (reverseBroadcast bc) b a) t := by
      change ((Structural.evalOp (engine M) (some .fp32) a t).bind fun va =>
        (Structural.evalOp (engine M) (some .fp32) b t).bind fun vb =>
          some (bop ((engine M).binary (some .fp32) .real .mul) bc va vb)) =
        ((Structural.evalOp (engine M) (some .fp32) b t).bind fun vb =>
        (Structural.evalOp (engine M) (some .fp32) a t).bind fun va =>
          some (bop ((engine M).binary (some .fp32) .real .mul) (reverseBroadcast bc) vb va))
      cases ha : Structural.evalOp (engine M) (some .fp32) a t <;>
        cases hb : Structural.evalOp (engine M) (some .fp32) b t
      all_goals simp only [Option.bind_none,Option.bind_some]
      rename_i va vb
      obtain ⟨hva,hvb⟩ := hd t hp va vb ha hb
      apply congrArg some
      funext i
      simp only [bop,reverseBroadcast_left,reverseBroadcast_right]
      exact ScalarArithmetic.mul_comm R M D hM t _ _ (hva (bc.leftIndex i)) (hvb (bc.rightIndex i))
    erw [he]
    rfl
/-- Lift the admitted mul_commute atom through arbitrary broadcasting. Both
operand tiles are checked after the actual common prefix; either evaluation
failure remains a failure in the other operand order. -/
@[spec_rule] theorem contextual_mul_broadcast_commute {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (bc : Broadcast sa sb shape) (a : Op .real sa) (b : Op .real sb)
    (hd : (Site.mulBroadcast before sa sb a b).Holds M D s) :
    Contextual M s before after
      (.assign .real shape out (.alg (.mul .real bc a b)))
      (.assign .real shape out (.alg (.mul .real (reverseBroadcast bc) b a))) := by
  unfold Contextual
  rw [run_append,run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some,run,step,Structural.evalExpr]
    have he : Structural.evalOp (engine M) none (.mul .real bc a b) t =
        Structural.evalOp (engine M) none (.mul .real (reverseBroadcast bc) b a) t := by
      change ((Structural.evalOp (engine M) none a t).bind fun va =>
        (Structural.evalOp (engine M) none b t).bind fun vb =>
          some (bop ((engine M).binary none .real .mul) bc va vb)) =
        ((Structural.evalOp (engine M) none b t).bind fun vb =>
        (Structural.evalOp (engine M) none a t).bind fun va =>
          some (bop ((engine M).binary none .real .mul) (reverseBroadcast bc) vb va))
      cases ha : Structural.evalOp (engine M) none a t <;>
        cases hb : Structural.evalOp (engine M) none b t
      all_goals simp only [Option.bind_none,Option.bind_some]
      rename_i va vb
      obtain ⟨hva,hvb⟩ := hd t hp va vb ha hb
      apply congrArg some
      funext i
      simp only [bop,reverseBroadcast_left,reverseBroadcast_right]
      exact ScalarArithmetic.mul_comm R M D hM t _ _ (hva (bc.leftIndex i)) (hvb (bc.rightIndex i))
    erw [he]


end VeriTile.Triton.FP.ProfiledRewrite
