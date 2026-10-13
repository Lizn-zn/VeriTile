import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivTriton.IndexRewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

/-- Exact signed index arithmetic, independent of every floating backend. -/
@[spec_rule] theorem signed_product_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a b : Op .int []) :
    OpEquivalent M s before (.mul .int .nil a b) (.mul .int .nil b a) := by
  intro t _
  change (evalOp (engine M) none a t >>= fun x =>
    evalOp (engine M) none b t >>= fun y => some (bop Int.mul .nil x y)) =
    (evalOp (engine M) none b t >>= fun y =>
    evalOp (engine M) none a t >>= fun x => some (bop Int.mul .nil y x))
  cases ha : evalOp (engine M) none a t <;>
    cases hb : evalOp (engine M) none b t
  all_goals first
    | rfl
    | apply congrArg some
      funext i
      exact Int.mul_comm _ _

/-- The natural-address boundary follows the signed product, unchanged on both sides. -/
@[spec_rule] theorem signed_pointer_product_commutes {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (before : List ComputeStmt) (r : RegionName)
    (a b : Op .int []) :
    OpEquivalent M s before
      (.ptrAdd .nil (.ptrBase r) (.castIntToNat (.mul .int .nil a b)))
      (.ptrAdd .nil (.ptrBase r) (.castIntToNat (.mul .int .nil b a))) := by
  intro t ht
  have h := signed_product_commutes M s before a b t ht
  change ((evalOp (engine M) none (.mul .int .nil a b) t >>= fun y =>
    some (fun i => (y i).toNat)) >>= fun y =>
      some (bop (fun (p : RegionName × Nat) (n : Nat) => (p.1, p.2 + n))
        .nil (fun _ => (r, 0)) y)) = _
  rw [h]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivTriton.IndexRewrite
