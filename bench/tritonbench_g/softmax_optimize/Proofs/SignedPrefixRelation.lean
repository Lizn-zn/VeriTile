import bench.tritonbench_g.softmax_optimize.Proofs.SignedPrefix

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton
set_option maxHeartbeats 3000000

/-- Address setup relates arbitrary initial scratch registers, including failure
when the shared row register is absent. -/
theorem prefix_related (X : RegionName) (N B i : Nat) (s t : BlockState)
    (h : LiveRelated s t) : ResultsRelated (IndexedRelated B)
      (stepStmts (indexPrefix X N B) (s.setReg "start_n" .int [] (Tile.scalar (i:Int))))
      (stepStmts (naturalPrefix X N B) (t.setReg "start_n" .nat [] (Tile.scalar i))) := by
  have hp := h.regs_eq .nat [] "pid_m" (by simp [Shared, Scratch])
  cases ht : t.regs .nat [] "pid_m" with
  | none =>
    have hs : s.regs .nat [] "pid_m" = none := hp.trans ht
    simp [indexPrefix, naturalPrefix, loopAbody, stepStmts, stepStmt, evalOp.eq_def,
      BlockState.setReg, hs, ht, ResultsRelated]
  | some p =>
    have heta : p = Tile.scalar (p.data PUnit.unit) := by
      apply Tile.ext
      intro ix
      cases ix
      rfl
    have ht' : t.regs .nat [] "pid_m" = some (Tile.scalar (p.data PUnit.unit)) := by
      simpa only [← heta] using ht
    rw [prefix_int_run X N B i _ s (hp.trans ht'), prefix_nat_run X N B i _ t ht']
    exact prefix_states_related X N B i _ h

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
