import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

/-! Scalar multiplication within a where branch. The surrounding condition,
other arm, failed evaluations and store suffix remain unchanged. Computation
uses the existing default fp32 engine. The operand
guards apply to every lane, because where evaluates both numeric arms. -/
namespace VeriTile.Triton.FP.ProfiledRewrite
open Structural Guarded
set_option maxHeartbeats 500000
set_option maxRecDepth 10000

@[spec_rule] theorem contextual_where_mul_scalar {α : Type} [Inhabited α]
    (R : ScalarArithmetic.Rules) (M : Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a : Op .real (n :: rest)) (b : Op .real [])
    (condition : Op .bool (n :: rest)) (otherwise : Op .real (n :: rest))
    (hd : (Site.mulScalar before n rest a b).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.alg (.where condition (.mul .real .scalarR a b) otherwise)))
      (.assign .real (n :: rest) out (.alg (.where condition (.mul .real .scalarL b a) otherwise))) := by
  unfold Contextual
  rw [run_append,run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    have hm : Structural.evalOp (engine M) none (.mul .real .scalarR a b) t =
        Structural.evalOp (engine M) none (.mul .real .scalarL b a) t := by
      change ((Structural.evalOp (engine M) none a t).bind fun va =>
        (Structural.evalOp (engine M) none b t).bind fun vb =>
          some (bop ((engine M).binary none .real .mul) .scalarR va vb)) =
        ((Structural.evalOp (engine M) none b t).bind fun vb =>
        (Structural.evalOp (engine M) none a t).bind fun va =>
          some (bop ((engine M).binary none .real .mul) .scalarL vb va))
      cases ha : Structural.evalOp (engine M) none a t <;>
        cases hb : Structural.evalOp (engine M) none b t
      all_goals simp only [Option.bind_none,Option.bind_some]
      rename_i va vb
      apply congrArg some
      funext i
      exact ScalarArithmetic.mul_comm R M D hM t _ _ ((hd t hp va vb ha hb).1 i)
        (hd t hp va vb ha hb).2
    have hw : Structural.evalOp (engine M) none
        (.where condition (.mul .real .scalarR a b) otherwise) t =
        Structural.evalOp (engine M) none
        (.where condition (.mul .real .scalarL b a) otherwise) t := by
      change ((Structural.evalOp (engine M) none condition t).bind fun vc =>
        (Structural.evalOp (engine M) none (.mul .real .scalarR a b) t).bind fun va =>
        (Structural.evalOp (engine M) none otherwise t).bind fun vb =>
          some (fun i => if vc i then va i else vb i)) = _
      rw [hm]
      rfl
    simp only [Option.bind_some,run,step,Structural.evalExpr]
    erw [hw]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Triton.FP.ProfiledRewrite
