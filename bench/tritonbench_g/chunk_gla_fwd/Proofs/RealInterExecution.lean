import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterFinal

/-! Complete real execution of the inter-subblock producer, including both
skip branches. On the active branch the full K loop computes the independent
unshifted contraction. All other tagged cells are preserved. Zero unspecified
padding is an explicit premise, not a modification of the source loads. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterExecution
open VeriTile.Triton RealInterMath RealInterProgram RealInterPointers RealInterLoop RealInterFinal
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem row_condition (p : Layout) (l : Launch) (s : BlockState) (hg : Grid p l s) :
    evalOp (rowCondition p) s = some (Tile.scalar (decide (rowStart p l < p.T))) := by
  simp only [rowCondition,timeOp,times,evalOp,evalOp_ref,hg.chunk,hg.row,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro i
  simp [-Tile.scalar_eta,Tile.scalar,Tile.uop,Tile.cop,Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    NumericDType.add,NumericDType.mul,ComparableDType.ge,rowStart]
  by_cases h : l.chunk*p.BT+rowBlock p l*p.BC < p.T
  · simp [h,Nat.not_le.mpr h]
  · simp [h,Nat.le_of_not_gt h]

theorem triangle_condition (p : Layout) (l : Launch) (s : BlockState) (hg : Grid p l s) :
    evalOp triangleCondition s = some (Tile.scalar (decide (columnBlock p l < rowBlock p l))) := by
  simp only [triangleCondition,evalOp,evalOp_ref,hg.row,hg.column,Option.bind_some,Option.bind_eq_bind]
  apply congrArg some
  apply Tile.ext
  intro i
  simp [-Tile.scalar_eta,Tile.scalar,Tile.uop,Tile.cop,Broadcast.leftIndex,Broadcast.rightIndex,ComparableDType.le]
  by_cases h : columnBlock p l < rowBlock p l
  · simp [h,Nat.not_le.mpr h]
  · simp [h,Nat.le_of_not_gt h]

theorem body_executes (p : Layout) (r : Regions) (scale : ℝ) (swapped : Bool) (s : BlockState)
    (hu : ∀ region off, s.undef region off = 0) (hinj : NoCollision p (launch s)) :
    ∃ sf, stepStmts (body p r scale swapped) s = some sf ∧
      (enabled p (launch s) → ∀ i : TileIndex [p.BC,p.BC], active p (launch s) i.1.val i.2.1.val →
        sf.readMem r.output (outputAddress p (launch s) i.1.val i.2.1.val) =
          expected p (launch s) (inputs r s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint p (launch s) r.output) s sf := by
  rw [body,stepStmts.append_some (initialization_executes p s)]
  have hg := initial_grid p s
  have hr := row_condition p (launch s) (initial p s) hg
  have ht := triangle_condition p (launch s) (initial p s) hg
  simp only [stepStmts,stepStmt,hr,ht,Tile.scalar,Option.bind_some,Option.bind_eq_bind]
  by_cases hrow : rowStart p (launch s) < p.T
  · simp only [hrow,decide_true,↓reduceIte]
    by_cases htri : columnBlock p (launch s) < rowBlock p (launch s)
    · simp only [htri,decide_true,↓reduceIte]
      rw [List.append_assoc,stepStmts.append_some (setup_executes p s)]
      obtain ⟨u,he,hi⟩ := RealInterLoop.executes p r scale s swapped hu (start p s) (start_holds p r scale s)
      rw [show [loop p r scale swapped] ++ finalization p r = loop p r scale swapped :: finalization p r from rfl,
        stepStmts.cons_some he]
      rw [RealInterFinal.executes p (launch s) r (inputs r s) scale u hi.grid hi.carry]
      refine ⟨_,rfl,?_,?_⟩
      · intro _ i ha
        exact readback p (launch s) r.output (inputs r s) scale _ hinj i ha
      · intro region off hn
        rw [← frame p (launch s) r.output (inputs r s) scale _ ⟨hrow,htri⟩ region off hn]
        simpa only [prepared,BlockState.setReg_mem] using (congrFun (congrFun hi.memory region) off).symm
    · simp only [htri,decide_false,Bool.false_eq_true,↓reduceIte]
      exact ⟨initial p s,rfl,fun h => False.elim (htri h.2),fun _ _ _ => rfl⟩
  · simp only [hrow,decide_false,Bool.false_eq_true,↓reduceIte]
    exact ⟨initial p s,rfl,fun h => False.elim (hrow h.1),fun _ _ _ => rfl⟩

theorem source_executes (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) (swapped : Bool) (s : BlockState)
    (hu : ∀ region off, s.undef region off = 0) :
    ∃ sf, exec (source p r qs ks gs out scale swapped).eraseDType.toAlgKernel s = some sf ∧
      (enabled p (launch s) → ∀ i : TileIndex [p.BC,p.BC], active p (launch s) i.1.val i.2.1.val →
        sf.readMem r.output (outputAddress p (launch s) i.1.val i.2.1.val) =
          expected p (launch s) (inputs r s) scale i.1.val i.2.1.val) ∧
      BlockState.WriteWithin (footprint p (launch s) r.output) s sf := by
  rw [exec,source_body]
  exact body_executes p r scale swapped s hu (no_collision p (launch s))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterExecution
