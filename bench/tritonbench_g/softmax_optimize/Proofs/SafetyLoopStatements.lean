import bench.tritonbench_g.softmax_optimize.Proofs.SafetyControl
import bench.tritonbench_g.softmax_optimize.Proofs.SafetyRanges

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
open VeriTile.Triton SignedTransport FullSource

def fullLoop (B : Nat) (body : List Stmt) : Stmt :=
  .forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B) body

def tailLoop (N B : Nat) (body : List Stmt) : Stmt :=
  .forRangeInt "start_n" (.ref .int [] "prev_multiple") (.constInt N) (.constNat B) body

theorem full_loop_safe (bounds : RegionBounds) (N B p : Nat) (body : List Stmt)
    (s : BlockState) (hc : Control N B p s)
    (hs : Stmt.forRangeIntTraceSafe bounds "start_n" 0 (prevMultipleNat N B : Int)
      B body s) : Stmt.TraceSafe bounds (fullLoop B body) s := by
  simpa [fullLoop, Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, hc.2] using hs

theorem tail_loop_safe (bounds : RegionBounds) (N B p : Nat) (body : List Stmt)
    (s : BlockState) (hc : Control N B p s)
    (hs : Stmt.forRangeIntTraceSafe bounds "start_n" (prevMultipleNat N B : Int)
      (N : Int) B body s) : Stmt.TraceSafe bounds (tailLoop N B body) s := by
  simpa [tailLoop, Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, hc.2] using hs

theorem full_loop_run (N B p : Nat) (body : List Stmt)
    (s : BlockState) (hc : Control N B p s) :
    stepStmt (fullLoop B body) s =
      stepForRangeIntAux "start_n" 0 (prevMultipleNat N B : Int) B body s := by
  simp [fullLoop, stepStmt, evalOp.eq_def, hc.2]

theorem tail_loop_run (N B p : Nat) (body : List Stmt)
    (s : BlockState) (hc : Control N B p s) :
    stepStmt (tailLoop N B body) s =
      stepForRangeIntAux "start_n" (prevMultipleNat N B : Int) (N : Int) B body s := by
  simp [tailLoop, stepStmt, evalOp.eq_def, hc.2]

theorem full_loop_control (Y X : RegionName) (N B p : Nat) (body : List Stmt)
    (hbody : body ∈ [inputFull X N B, inputTail X N B,
      outputFull Y X N B, outputTail Y X N B, reductionStmts B])
    (s t : BlockState) (hc : Control N B p s)
    (hr : stepStmt (fullLoop B body) s = some t) : Control N B p t := by
  rw [full_loop_run N B p body s hc] at hr
  exact range_control_preserved Y X N B p (prevMultipleNat N B) 0 body hbody s t hc hr

theorem tail_loop_control (Y X : RegionName) (N B p : Nat) (body : List Stmt)
    (hbody : body ∈ [inputFull X N B, inputTail X N B,
      outputFull Y X N B, outputTail Y X N B, reductionStmts B])
    (s t : BlockState) (hc : Control N B p s)
    (hr : stepStmt (tailLoop N B body) s = some t) : Control N B p t := by
  rw [tail_loop_run N B p body s hc] at hr
  exact range_control_preserved Y X N B p N (prevMultipleNat N B) body hbody s t hc hr

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.Safety
