import bench.tritonbench_g.softmax_optimize.Proofs.SignedExpression
import bench.tritonbench_g.softmax_optimize.Proofs.SignedBodies

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton

def naturalMask (N B : Nat) : Stmt := .assign .bool [B] "mask"
  (.lt .nat .scalarR (.ref .nat [B] "n_offsets") (.constNat N))

theorem mask_expression_related (N B : Nat) : ExprRelated B
    (.lt .int .scalarR (.ref .int [B] "n_offsets") (.castNatToInt (.constNat N)))
    (.lt .nat .scalarR (.ref .nat [B] "n_offsets") (.constNat N)) := by
  intro s t h
  simp only [evalOp.eq_def, h.columns_eq]
  cases hv : t.regs .nat [B] "n_offsets" with
  | none => rfl
  | some v =>
    simp only [Option.map_some, Option.bind_some]
    apply congrArg some
    apply Tile.ext
    intro i
    change decide ((v.data i : Int) < (N : Int)) = decide (v.data i < N)
    simp only [Int.ofNat_lt]

theorem mask_related (N B : Nat) : ∀ s t, IndexedRelated B s t →
    ResultsRelated (IndexedRelated B)
      (stepStmt (maskStatement N B) s) (stepStmt (naturalMask N B) t) :=
  assign_related "mask" (by simp [Shared, Scratch]) (mask_expression_related N B)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
