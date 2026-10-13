import bench.tritonbench_g.softmax_optimize.Proofs.SignedRanges
import bench.tritonbench_g.softmax_optimize.Proofs.Geometry

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton FullSource

def naturalPrevious (N B : Nat) : Op .nat [] :=
  .sub .nat .nil (.mul .nat .nil
    (.floorDiv .nat .nil (.constNat (N+B-1)) (.constNat B)) (.constNat B)) (.constNat B)

theorem natural_previous_eval (N B : Nat) (s : BlockState) :
    evalOp (naturalPrevious N B) s = some (Tile.scalar (prevMultiple N B)) := by
  simp only [naturalPrevious, evalOp.eq_def, Option.bind_some]
  apply congrArg some
  apply Tile.ext
  intro i
  rfl

theorem previous_eval (N B : Nat) (hN : 0<N) (hB : 0<B) (s : BlockState) :
    evalOp (previousExpression N B) s = some (Tile.scalar (prevMultiple N B : Int)) := by
  rw [previousExpression_eval N B hB]
  rw [prevMultipleInt_eq_nat N B hN hB]
  rfl

theorem previous_related (N B : Nat) (hN : 0<N) (hB : 0<B) :
    ∀ s t, LiveRelated s t → ResultsRelated LiveRelated
      (stepStmt (.assign .int [] "prev_multiple" (previousExpression N B)) s)
      (stepStmt (.assign .nat [] "prev_multiple" (naturalPrevious N B)) t) := by
  intro s t h
  simp only [stepStmt, previous_eval N B hN hB, natural_previous_eval, Option.bind_some]
  exact h.setPrevious _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
