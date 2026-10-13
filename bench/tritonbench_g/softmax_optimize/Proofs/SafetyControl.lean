import bench.tritonbench_g.softmax_optimize.Proofs.SafetyBodyState
import bench.tritonbench_g.softmax_optimize.Proofs.SafetyLoop

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport FullSource

def Control (N B p : Nat) (s : BlockState) : Prop :=
  s.regs .nat [] "pid_m" = some (Tile.scalar p) ∧
    s.regs .int [] "prev_multiple" = some (Tile.scalar (prevMultipleNat N B : Int))

theorem body_control_preserved (Y X : RegionName) (N B p : Nat) (body : List Stmt)
    (hbody : body ∈ [inputFull X N B, inputTail X N B,
      outputFull Y X N B, outputTail Y X N B, reductionStmts B])
    (s t : BlockState) (hcontrol : Control N B p s)
    (hrun : stepStmts body s = some t) : Control N B p t := by
  constructor
  · rw [source_bodies_preserve "pid_m" (Or.inl rfl) .nat [] Y X N B body hbody s t hrun]
    exact hcontrol.1
  · rw [source_bodies_preserve "prev_multiple" (Or.inr rfl) .int []
      Y X N B body hbody s t hrun]
    exact hcontrol.2

theorem range_control_preserved (Y X : RegionName) (N B p stop c : Nat)
    (body : List Stmt)
    (hbody : body ∈ [inputFull X N B, inputTail X N B,
      outputFull Y X N B, outputTail Y X N B, reductionStmts B])
    (s t : BlockState) (hcontrol : Control N B p s)
    (hrun : stepForRangeIntAux "start_n" (c : Int) (stop : Int) B body s = some t) :
    Control N B p t := by
  apply signed_loop_preserves "start_n" stop B body (Control N B p) _ c s t hcontrol hrun
  intro i u v hi hcontrol hstep
  apply body_control_preserved Y X N B p body hbody _ v _ hstep
  simpa [Control] using hcontrol

theorem previous_run (N B : Nat) (hN : 0 < N) (hB : 0 < B) (s : BlockState) :
    stepStmt (.assign .int [] "prev_multiple" (previousExpression N B)) s =
      some (s.setReg "prev_multiple" .int [] (Tile.scalar (prevMultipleNat N B : Int))) := by
  apply stepStmt_assign_eq_some
  rw [previousExpression_eval N B hB s, prevMultipleInt_eq_nat N B hN hB]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
