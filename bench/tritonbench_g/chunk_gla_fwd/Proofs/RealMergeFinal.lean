import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealMergeLoop

/-! Final masked store and independent real result. Only valid output rows
are observed, so arbitrary padding in other accumulator lanes cannot affect
this formula. Input/output aliasing is allowed: all input loads precede this store. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeFinal
open VeriTile.Triton RealMergeProgram RealMergeStep
noncomputable section
set_option maxRecDepth 14000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def pointer (p : Layout) (l : Launch) (A2 : RegionName) : Tile .blockPtr [p.BC,p.BC] :=
  ⟨fun _ => BlockPtr.mk A2 (l.head*p.T*p.BT) [p.T,p.BT] [p.BC,p.BC] [p.BT,1] [rowStart p l,l.subblock*p.BC]⟩
def prepared (p : Layout) (l : Launch) (A2 : RegionName) (s : BlockState) : BlockState :=
  s.setReg "p_A2" .blockPtr [p.BC,p.BC] (pointer p l A2)
def scatter (p : Layout) (l : Launch) (A2 : RegionName) (a padding : Nat → ℝ) (s : BlockState) : BlockState :=
  (TileShape.allIndices [p.BC,p.BC]).foldl (fun u i =>
    if active p l i then u.writeMem A2 (outputAddress p l i) (accumulated p l a padding p.NK i) else u) s

theorem pointer_eval (p : Layout) (l : Launch) (A2 : RegionName) (s : BlockState) (hg : Grid l s) :
    evalOp (outputPointer p A2) s = some (pointer p l A2) := by
  simp only [outputPointer,outputBase,rowOp,mulRef,evalOp,evalOp_ref,hg.chunk,hg.subblock,hg.head,
    List.mapM_cons,List.mapM_nil,Option.bind_some,Option.pure_def,Option.bind_eq_bind]
  rfl

theorem store_eval (p : Layout) (l : Launch) (A2 : RegionName) (a padding : Nat → ℝ) (s : BlockState)
    (hp : s.regs .blockPtr [p.BC,p.BC] "p_A2" = some (pointer p l A2))
    (hc : s.regs .real [p.BC,p.BC] "b_A" = some (accumulator p l a padding p.NK)) :
    stepStmt (.store .real [p.BC,p.BC] (.blockPtr (.ref .blockPtr [p.BC,p.BC] "p_A2") [0,1]) (carry p) .none) s =
      some (scatter p l A2 a padding s) := by
  simp only [stepStmt,carry,evalOp_ref,hp,hc,Option.bind_some,Option.bind_eq_bind,
    Bool.true_and,pointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,decide_eq_true_eq,Nat.mul_one]
  rfl

theorem executes (p : Layout) (l : Launch) (A2 : RegionName) (a padding : Nat → ℝ) (s : BlockState)
    (hg : Grid l s) (hc : s.regs .real [p.BC,p.BC] "b_A" = some (accumulator p l a padding p.NK)) :
    stepStmts (finalization p A2) s = some (scatter p l A2 a padding (prepared p l A2 s)) := by
  unfold finalization
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l A2 s hg))]
  rw [stepStmts.cons_some (store_eval p l A2 a padding _ (by simp) (by simpa using hc)),stepStmts.nil]
  rfl

theorem accumulated_active (p : Layout) (l : Launch) (a padding : Nat → ℝ)
    (i : TileIndex [p.BC,p.BC]) (hi : active p l i) : accumulated p l a padding p.NK i = expected p l a i := by
  apply Finset.sum_congr rfl
  intro kb _
  simp only [value,hi.1,ite_true]

theorem readback (p : Layout) (l : Launch) (A2 : RegionName) (a padding : Nat → ℝ) (s : BlockState)
    (hinj : NoCollision p l) (i : TileIndex [p.BC,p.BC]) (hi : active p l i) :
    (scatter p l A2 a padding s).readMem A2 (outputAddress p l i) = expected p l a i := by
  rw [← accumulated_active p l a padding i hi]
  exact BlockState.scatter_readback_prop_masked_nd_of_true s _ _ _ i hi (fun j hj he => hinj j i hj hi he)

theorem frame (p : Layout) (l : Launch) (A2 : RegionName) (a padding : Nat → ℝ) (s : BlockState)
    (hen : enabled p l) :
    BlockState.WriteWithin (footprint p l A2) s (scatter p l A2 a padding s) := by
  apply BlockState.foldl_writeMem_prop_writeWithin
  intro i _ hi
  exact ⟨rfl,hen,i,hi,rfl⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeFinal
