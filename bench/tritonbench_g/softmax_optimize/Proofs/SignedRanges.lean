import bench.tritonbench_g.softmax_optimize.Proofs.SignedReduction

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton
set_option maxHeartbeats 3000000

theorem full_range_related (B : Nat) (left right : List Stmt)
    (hb : ∀ (i : Nat) s t, LiveRelated s t → ResultsRelated LiveRelated
      (stepStmts left (s.setReg "start_n" .int [] (Tile.scalar (i:Int))))
      (stepStmts right (t.setReg "start_n" .nat [] (Tile.scalar i)))) :
    ∀ s t, LiveRelated s t → ResultsRelated LiveRelated
      (stepStmt (.forRangeInt "start_n" (.constInt 0) (.ref .int [] "prev_multiple") (.constNat B) left) s)
      (stepStmt (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "prev_multiple") (.constNat B) right) t) := by
  intro s t h
  simp only [stepStmt, evalOp.eq_def, h.prev_eq]
  cases hp : t.regs .nat [] "prev_multiple" with
  | none => exact True.intro
  | some p =>
    simp only [Option.map_some, Option.bind_some, Tile.scalar_data]
    exact loop_related LiveRelated "start_n" (p.data PUnit.unit) B left right
      (fun i _ s t h ↦ hb i s t h) 0 s t h

theorem tail_range_related (N B : Nat) (left right : List Stmt)
    (hb : ∀ (i : Nat) s t, LiveRelated s t → ResultsRelated LiveRelated
      (stepStmts left (s.setReg "start_n" .int [] (Tile.scalar (i:Int))))
      (stepStmts right (t.setReg "start_n" .nat [] (Tile.scalar i)))) :
    ∀ s t, LiveRelated s t → ResultsRelated LiveRelated
      (stepStmt (.forRangeInt "start_n" (.ref .int [] "prev_multiple") (.constInt N) (.constNat B) left) s)
      (stepStmt (.forRangeDyn "start_n" (.ref .nat [] "prev_multiple") (.constNat N) (.constNat B) right) t) := by
  intro s t h
  simp only [stepStmt, evalOp.eq_def, h.prev_eq]
  cases hp : t.regs .nat [] "prev_multiple" with
  | none => exact True.intro
  | some p =>
    simp only [Option.map_some, Option.bind_some, Tile.scalar_data]
    exact loop_related LiveRelated "start_n" N B left right
      (fun i _ s t h ↦ hb i s t h) (p.data PUnit.unit) s t h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
