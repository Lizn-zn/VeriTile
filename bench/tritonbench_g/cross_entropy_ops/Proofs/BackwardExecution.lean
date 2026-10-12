import bench.tritonbench_g.cross_entropy_ops.Proofs.BackwardReal
namespace VeriTile.Bench.TritonBenchG.CrossEntropyOps.BackwardExecution
open VeriTile.Triton VeriTile.Examples BackwardFormula BackwardProgram BackwardReal
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem succeeds (p : Layout) (r : Regions) (u : BlockState) :
    ∃ v, exec (mathematical p r).toAlgKernel u = some v := by
  cases hs : p.hasSmoothing <;>
    by_cases hi : u.readMemValue .int r.label (u.pids 0) = p.ignored
  all_goals simp only [Region.cast] at hi
  all_goals simp [exec,mathematical,realKernel,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
      ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,stepStmts,stepStmt,evalOp.eq_def,
      Option.bind,Tile.bop,Tile.uop,Tile.cop,Tile.ptrAdd,NumericDType.add,
      NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,ComparableDType.eq,
      ComparableDType.ne,Region.cast,hs,hi]

theorem frame (p : Layout) (r : Regions) (u v : BlockState)
    (he : exec (mathematical p r).toAlgKernel u = some v)
    (region : RegionName) (offset : Nat)
    (hmiss : ∀ i : Fin p.B, active p (launch u) i.val →
      ¬ (r.gradient=region ∧ gradientAddress p (launch u) i.val=offset)) :
    v.mem region offset = u.mem region offset := by
  cases hs : p.hasSmoothing <;>
    by_cases hi : u.readMemValue .int r.label (u.pids 0) = p.ignored
  all_goals simp only [Region.cast] at hi
  all_goals simp [exec,mathematical,realKernel,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
      ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,stepStmts,stepStmt,evalOp.eq_def,
      Option.bind,Tile.bop,Tile.uop,Tile.cop,Tile.ptrAdd,NumericDType.add,
      NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,ComparableDType.eq,
      ComparableDType.ne,Region.cast,hs,hi] at he
  all_goals rw [← he]
  all_goals
    refine Eq.trans (foldl_store_preserve_cell _ _ _ region offset _ _ ?_) rfl
    intro k _ hm hc
    exact hmiss k.1 hm hc

/-- Complete region execution with the independent gradient formula and every
untouched tagged cell preserved. No precomputed probability is a premise. -/
theorem correct (p : Layout) (r : Regions) (u : BlockState) :
    ∃ v, exec (mathematical p r).toAlgKernel u = some v ∧
      (∀ i : Fin p.B, active p (launch u) i.val →
        v.readMem r.gradient (gradientAddress p (launch u) i.val) =
          expected p (launch u) (inputs u r) i.val) ∧
      (∀ region offset, (∀ i : Fin p.B, active p (launch u) i.val →
        ¬ (r.gradient=region ∧ gradientAddress p (launch u) i.val=offset)) →
          v.mem region offset = u.mem region offset) := by
  obtain ⟨v,he⟩ := succeeds p r u
  refine ⟨v,he,?_,frame p r u v he⟩
  intro i hi
  simpa only [if_pos hi] using BackwardReal.output p r u v he i

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CrossEntropyOps.BackwardExecution
