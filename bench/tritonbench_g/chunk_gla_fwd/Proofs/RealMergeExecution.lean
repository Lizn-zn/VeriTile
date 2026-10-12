import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealMergeFinal

/-! Successful complete real merge execution, including the original early
return. The valid output is the independent sum over the initial NK slices,
with arbitrary unspecified values. Every other tagged cell is preserved. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeExecution
open VeriTile.Triton RealMergeProgram RealMergeStep RealMergeLoop
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem condition_eval (p : Layout) (l : Launch) (s : BlockState)
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar l.chunk))
    (hc : s.regs .nat [] "i_c" = some (Tile.scalar l.subblock)) :
    evalOp (condition p) s = some (Tile.scalar (decide (enabled p l))) := by
  simp only [condition,rowOp,mulRef,evalOp,evalOp_ref,ht,hc,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro i
  simp [-Tile.scalar_eta,Tile.scalar,Tile.uop,Tile.cop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.add,NumericDType.mul,ComparableDType.ge,enabled,rowStart]
  by_cases h : l.chunk*p.BT+l.subblock*p.BC < p.T
  · simp [h,Nat.not_le.mpr h]
  · simp [h,Nat.le_of_not_gt h]

theorem body_executes (p : Layout) (A A2 : RegionName) (swapped : Bool) (s : BlockState)
    (hinj : NoCollision p (launch s)) :
    ∃ sf, stepStmts (body p A A2 swapped) s = some sf ∧
      (∀ i, active p (launch s) i → sf.readMem A2 (outputAddress p (launch s) i) = expected p (launch s) (s.readMem A) i) ∧
      BlockState.WriteWithin (footprint p (launch s) A2) s sf := by
  rw [body,stepStmts.append_some (initialization_executes s)]
  have he := condition_eval p (launch s) (initial s) (by simp [initial,launch]) (by simp [initial,launch])
  simp only [stepStmts,stepStmt,he,Tile.scalar,Option.bind_some,Option.bind_eq_bind]
  by_cases hen : enabled p (launch s)
  · simp only [hen,decide_true,↓reduceIte]
    rw [List.append_assoc,stepStmts.append_some (setup_executes p s)]
    obtain ⟨u,hu,hi⟩ := RealMergeLoop.executes p s A swapped (start p s) (start_holds p s A)
    rw [show [loop p A swapped] ++ finalization p A2 = loop p A swapped :: finalization p A2 from rfl,
      stepStmts.cons_some hu]
    rw [RealMergeFinal.executes p (launch s) A2 (s.readMem A) (s.undef A) u hi.grid hi.carry]
    refine ⟨_,rfl,?_,?_⟩
    · intro i ha
      exact RealMergeFinal.readback p (launch s) A2 (s.readMem A) (s.undef A) _ hinj i ha
    · intro region off hn
      rw [← RealMergeFinal.frame p (launch s) A2 (s.readMem A) (s.undef A) _ hen region off hn]
      simpa only [RealMergeFinal.prepared,BlockState.setReg_mem] using
        (congrFun (congrFun hi.memory region) off).symm
  · simp only [hen,decide_false,Bool.false_eq_true,↓reduceIte]
    refine ⟨initial s,rfl,?_,?_⟩
    · intro i ha
      have hr := ha.1
      exact False.elim (hen (Nat.lt_of_le_of_lt (Nat.le_add_right _ _) hr))
    · intro region off _
      rfl

theorem source_executes (p : Layout) (A A2 : RegionName) (out : FloatDType) (swapped : Bool) (s : BlockState)
    (hinj : NoCollision p (launch s)) :
    ∃ sf, exec (source p A A2 out swapped).eraseDType.toAlgKernel s = some sf ∧
      (∀ i, active p (launch s) i → sf.readMem A2 (outputAddress p (launch s) i) = expected p (launch s) (s.readMem A) i) ∧
      BlockState.WriteWithin (footprint p (launch s) A2) s sf := by
  rw [exec,source_body]
  exact body_executes p A A2 swapped s hinj

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeExecution
