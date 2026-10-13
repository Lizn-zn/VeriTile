import bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose.IndexRewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

/-- Scalar multiplication of a natural index tile commutes, including failure. -/
@[spec_rule] theorem scalar_product_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a : Op .nat (n :: shape)) (b : Op .nat []) :
    OpEquivalent M s before (.mul .nat .scalarR a b) (.mul .nat .scalarL b a) := by
  intro t _
  change (evalOp (engine M) none a t >>= fun x =>
    evalOp (engine M) none b t >>= fun y => some (bop Nat.mul .scalarR x y)) =
    (evalOp (engine M) none b t >>= fun y =>
    evalOp (engine M) none a t >>= fun x => some (bop Nat.mul .scalarL y x))
  cases ha : evalOp (engine M) none a t <;>
    cases hb : evalOp (engine M) none b t
  all_goals first
    | rfl
    | apply congrArg some
      funext i
      exact Nat.mul_comm _ _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatrixTranspose.IndexRewrite
