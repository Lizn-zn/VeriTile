import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealMergeProgram

/-! One complete merge iteration. Invalid row lanes retain their arbitrary
source padding; the eventual valid output formula will remove those lanes.
Both addition orders have the same real accumulator and preserve memory. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeStep
open VeriTile.Triton RealMergeProgram
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Grid (l : Launch) (s : BlockState) : Prop where
  chunk : s.regs .nat [] "i_t" = some (Tile.scalar l.chunk)
  subblock : s.regs .nat [] "i_c" = some (Tile.scalar l.subblock)
  head : s.regs .nat [] "i_bh" = some (Tile.scalar l.head)
  heads : s.regs .nat [] "n_bh" = some (Tile.scalar l.heads)

def pointer (p : Layout) (l : Launch) (A : RegionName) (kb : Nat) : Tile .blockPtr [p.BC,p.BC] :=
  ⟨fun _ => BlockPtr.mk A ((l.head+kb*l.heads)*p.T*p.BC) [p.T,p.BC] [p.BC,p.BC] [p.BC,1] [rowStart p l,0]⟩
def value (p : Layout) (l : Launch) (a padding : Nat → ℝ) (kb : Nat) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  if rowStart p l+i.1.val < p.T then a (inputAddress p l kb i) else padding (inputAddress p l kb i)
def inputTile (p : Layout) (l : Launch) (a padding : Nat → ℝ) (kb : Nat) : Tile .real [p.BC,p.BC] :=
  ⟨fun i => some (value p l a padding kb i)⟩
def accumulated (p : Layout) (l : Launch) (a padding : Nat → ℝ) (n : Nat) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  ∑ kb ∈ Finset.range n, value p l a padding kb i
def accumulator (p : Layout) (l : Launch) (a padding : Nat → ℝ) (n : Nat) : Tile .real [p.BC,p.BC] :=
  ⟨fun i => some (accumulated p l a padding n i)⟩

theorem pointer_eval (p : Layout) (l : Launch) (A : RegionName) (kb : Nat) (s : BlockState)
    (hg : Grid l s) (hk : s.regs .nat [] "i_k" = some (Tile.scalar kb)) :
    evalOp (inputPointer p A) s = some (pointer p l A kb) := by
  simp only [inputPointer,inputBase,rowOp,mulRef,evalOp,evalOp_ref,hg.chunk,hg.subblock,hg.head,hg.heads,hk,
    List.mapM_cons,List.mapM_nil,Option.bind_some,Option.pure_def,Option.bind_eq_bind]
  rfl

theorem load_eval (p : Layout) (l : Launch) (A : RegionName) (a padding : Nat → ℝ)
    (kb : Nat) (s : BlockState)
    (hm : ∀ off, s.readMem A off = a off) (hu : ∀ off, s.undef A off = padding off)
    (hp : s.regs .blockPtr [p.BC,p.BC] "p_A" = some (pointer p l A kb)) :
    evalOp (loaded p) s = some (inputTile p l a padding kb) := by
  simp only [loaded,evalOp,evalOp_ref,hp]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [pointer,inputTile,value,inputAddress,TileShape.indexToList,BlockPtr.inBounds_2d_offsets,
    BlockPtr.address_2d_offsets,BlockState.readMemValue_real,decide_eq_true_eq,
    Nat.zero_add,j.isLt,and_true,Nat.mul_one]
  split <;> simp_all only [BlockState.paddingValue,BlockState.unspecifiedValue]

theorem accumulator_step (p : Layout) (l : Launch) (a padding : Nat → ℝ) (kb : Nat) :
    Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
      (accumulator p l a padding kb) (inputTile p l a padding kb) = accumulator p l a padding (kb+1) := by
  apply Tile.ext
  intro i
  simp [Tile.bop_data,accumulator,inputTile,accumulated,Finset.sum_range_succ,
    NumericDType.add,WithBot.realAdd,Broadcast.leftIndex,Broadcast.rightIndex]

theorem accumulator_step_swapped (p : Layout) (l : Launch) (a padding : Nat → ℝ) (kb : Nat) :
    Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
      (inputTile p l a padding kb) (accumulator p l a padding kb) = accumulator p l a padding (kb+1) := by
  apply Tile.ext
  intro i
  simp [Tile.bop_data,accumulator,inputTile,accumulated,Finset.sum_range_succ,
    NumericDType.add,WithBot.realAdd,Broadcast.leftIndex,Broadcast.rightIndex,add_comm]

def result (p : Layout) (l : Launch) (A : RegionName) (a padding : Nat → ℝ) (kb : Nat) (s : BlockState) : BlockState :=
  (s.setReg "p_A" .blockPtr [p.BC,p.BC] (pointer p l A kb)).setReg "b_A" .real [p.BC,p.BC]
    (accumulator p l a padding (kb+1))

theorem iteration_executes (p : Layout) (l : Launch) (A : RegionName) (a padding : Nat → ℝ)
    (kb : Nat) (swapped : Bool) (s : BlockState)
    (hg : Grid l s) (hk : s.regs .nat [] "i_k" = some (Tile.scalar kb))
    (hm : ∀ off, s.readMem A off = a off) (hu : ∀ off, s.undef A off = padding off)
    (hc : s.regs .real [p.BC,p.BC] "b_A" = some (accumulator p l a padding kb)) :
    stepStmts (iteration p A swapped) s = some (result p l A a padding kb s) := by
  unfold iteration
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (pointer_eval p l A kb s hg hk))]
  cases swapped
  all_goals
    simp only [Bool.false_eq_true,↓reduceIte]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp _ _ = some (accumulator p l a padding (kb+1)) from ?_)),stepStmts.nil]
    rfl
  all_goals
    rw [evalOp_add]
    simp only [carry,evalOp_ref,BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide : ("b_A" : RegName) ≠ "p_A"),hc]
    rw [load_eval p l A a padding kb _ (by simpa [BlockState.readMem,BlockState.setReg] using hm)
      (by simpa [BlockState.setReg] using hu) (by simp)]
    simp only [Option.bind_eq_bind,Option.bind_some]
  · rw [accumulator_step]
  · rw [accumulator_step_swapped]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealMergeStep
