import bench.tritonbench_g.matrix_transpose.Proofs.IndexRewrite
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite

namespace VeriTile.Bench.TritonBenchG.MaxReduction.IndexRewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

/-- Complementing the reversed natural bound preserves the complete mask,
including evaluation failure. This is an exact integer rewrite. -/
@[spec_rule] theorem bound_complement {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a : Op .nat (n :: shape)) (b : Op .nat []) :
    OpEquivalent M s before (.lt .nat .scalarR a b)
      (.boolNot (.le .nat .scalarL b a)) := by
  intro t _
  change (evalOp (engine M) none a t >>= fun x =>
    evalOp (engine M) none b t >>= fun y =>
      some (bop (fun a b => decide (a < b)) .scalarR x y)) =
    ((evalOp (engine M) none b t >>= fun y =>
      evalOp (engine M) none a t >>= fun x =>
        some (bop (fun b a => decide (b ≤ a)) .scalarL y x)) >>= fun v =>
          some (fun i => !(v i)))
  cases ha : evalOp (engine M) none a t <;>
    cases hb : evalOp (engine M) none b t
  all_goals first
    | rfl
    | apply congrArg some
      funext i
      simp only [bop]
      rename_i x y
      by_cases h : x i < y PUnit.unit
      · simp [h, Nat.not_le.mpr h]
      · simp [h, Nat.le_of_not_gt h]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MaxReduction.IndexRewrite
