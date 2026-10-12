import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealSetup

/-! The complete scaling and output suffix. Ordinary store overwrites a cell;
atomic addition adds this program's contribution to that cell's initial value.
These are real per-program semantics, with no launch interleaving claim. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealTail
open VeriTile.Triton RealSetupSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def offset (p : Parameters) (batch n : Nat) : Nat :=
  batch * p.outputBatchStride + n * p.outputStride

private def scaled (p : Parameters) (s : BlockState) (g : Fin p.BN → ℝ) : BlockState :=
  s.setReg "accumulator" .real [p.BN] ⟨fun i => some (g i.1 * p.scaling)⟩

private def offsets (p : Parameters) (s : BlockState) (g : Fin p.BN → ℝ) : BlockState :=
  (scaled p s g).setReg "offset_cn" .nat [p.BN] (Tile.vec (fun i => i.val))

private def pointers (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ) : BlockState :=
  (offsets p s g).setReg "c_ptr" .ptr [p.BN] ⟨fun i => (p.output, offset p batch i.1.val)⟩

def prepared (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ) : BlockState :=
  (pointers p s batch g).setReg "c_mask" .bool [p.BN] ⟨fun i => decide (i.1.val < p.N)⟩

private theorem scale_exec (p : Parameters) (swapped : Bool) (s : BlockState) (g : Fin p.BN → ℝ)
    (hg : s.regs .real [p.BN] "accumulator" = some ⟨fun i => some (g i.1)⟩) :
    stepStmt (p.scale swapped) s = some (scaled p s g) := by
  cases swapped
  · exact stepStmt_assign_eq_some (scale_eval p.BN p.scaling g s hg)
  · apply stepStmt_assign_eq_some
    simp only [if_true, evalOp_mul, evalOp_ref, evalOp_const, hg, Option.bind_some, bind,
      Option.bind_eq_bind]
    apply congrArg some
    apply Tile.ext
    intro i
    simp only [Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map, mul_comm]

def scatter (p : Parameters) (s : BlockState) (batch : Nat) (v : Fin p.BN → ℝ) : BlockState :=
  (TileShape.allIndices [p.BN]).foldl (fun acc i =>
    if decide (i.1.val < p.N) then
      acc.writeMem p.output (offset p batch i.1.val)
        (if p.splits = 1 then v i.1 else acc.readMem p.output (offset p batch i.1.val) + v i.1)
    else acc) s

def result (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ) : BlockState :=
  scatter p (prepared p s batch g) batch (fun n => g n * p.scaling)

/-- Both implementations prepare exactly the same real output values and
take the output branch selected by the actual split count. -/
theorem tail_exec (p : Parameters) (swapped : Bool) (s : BlockState)
    (batch : Nat) (g : Fin p.BN → ℝ)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar batch))
    (hg : s.regs .real [p.BN] "accumulator" = some ⟨fun i => some (g i.1)⟩) :
    stepStmts (p.tail swapped) s = some (result p s batch g) := by
  unfold Parameters.tail shrinkTailCommon
  simp only [List.drop_succ_cons, List.drop_zero, List.cons_append, List.nil_append]
  rw [stepStmts.cons_some (scale_exec p swapped s g hg)]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp :
    evalOp (Op.arange p.BN) (scaled p s g) = some (Tile.vec (fun i => i.val))))]
  change stepStmts _ (offsets p s g) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cptr_tail_eval p.output p.outputBatchStride p.outputStride p.BN batch _
      (by simpa [offsets, scaled] using hcb) (by simp [offsets])))]
  change stepStmts _ (pointers p s batch g) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (cmask_eval p.N p.BN _ (by simp [pointers, offsets])))]
  change stepStmts _ (prepared p s batch g) = _
  have hp : (prepared p s batch g).regs .ptr [p.BN] "c_ptr" =
      some ⟨fun i => (p.output, offset p batch i.1.val)⟩ := by simp [prepared, pointers]
  have hm : (prepared p s batch g).regs .bool [p.BN] "c_mask" =
      some ⟨fun i => decide (i.1.val < p.N)⟩ := by simp [prepared]
  have hv : (prepared p s batch g).regs .real [p.BN] "accumulator" =
      some ⟨fun i => some (g i.1 * p.scaling)⟩ := by simp [prepared, pointers, offsets, scaled]
  have hc : evalOp (Op.eq .nat .nil (.constNat p.splits) (.constNat 1)) (prepared p s batch g) =
      some (Tile.scalar (decide (p.splits = 1))) := by
    simp only [evalOp, Tile.cop, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex]
    rfl
  have hb : stepStmt (.ifThenElse (.eq .nat .nil (.constNat p.splits) (.constNat 1))
      [shrinkStoreStmt p.BN] [shrinkAtomicStmt p.BN]) (prepared p s batch g) =
      some (result p s batch g) := by
    simp only [stepStmt, hc, Option.bind_some, Tile.scalar_data]
    by_cases hs : p.splits = 1
    · simp only [hs, decide_true, if_true]
      rw [stepStmts.cons_some (store_tail_step p.output p.N p.BN (fun n => offset p batch n.val)
        (fun n => g n * p.scaling) _ hp hm hv), stepStmts.nil]
      simp only [result, scatter, hs, if_true]
      rfl
    · simp only [hs, decide_false, Bool.false_eq_true, if_false]
      rw [stepStmts.cons_some (atomic_tail_step p.output p.N p.BN (fun n => offset p batch n.val)
        (fun n => g n * p.scaling) _ hp hm hv), stepStmts.nil]
      simp only [result, scatter, hs, if_false]
      rfl
  rw [stepStmts.cons_some hb, stepStmts.nil]

theorem scatter_active (p : Parameters) (s : BlockState) (batch : Nat) (v : Fin p.BN → ℝ)
    (hstride : 0 < p.outputStride) (n : Fin p.BN) (hn : n.val < p.N) :
    (scatter p s batch v).readMem p.output (offset p batch n.val) =
      if p.splits = 1 then v n else s.readMem p.output (offset p batch n.val) + v n := by
  have hu : ∀ b ∈ TileShape.allIndices [p.BN], decide (b.1.val < p.N) = Bool.true →
      offset p batch b.1.val = offset p batch n.val → b = (n, PUnit.unit) := by
    intro b _ _ he
    have hv : b.1.val = n.val :=
      Nat.eq_of_mul_eq_mul_right hstride (Nat.add_left_cancel he)
    exact Prod.ext (Fin.ext hv) rfl
  unfold scatter
  by_cases hs : p.splits = 1
  · simp only [hs, if_true]
    exact foldl_store_at _ _ _ _ _ s (n, PUnit.unit)
      (TileShape.mem_allIndices _ _) (by simpa) rfl hu (TileShape.allIndices_nodup _)
  · simp only [hs, if_false]
    exact foldl_atomicAdd_at _ _ _ _ _ s (n, PUnit.unit)
      (TileShape.mem_allIndices _ _) (by simpa) rfl hu (TileShape.allIndices_nodup _)

/-- Every untouched cell is preserved as a typed cell, including metadata,
masked output lanes and other regions. No alias restriction is needed here. -/
theorem scatter_frame (p : Parameters) (s : BlockState) (batch : Nat) (v : Fin p.BN → ℝ)
    (r : RegionName) (o : Nat)
    (h : ∀ n : Fin p.BN, n.val < p.N → ¬(p.output = r ∧ offset p batch n.val = o)) :
    (scatter p s batch v).mem r o = s.mem r o := by
  unfold scatter
  generalize TileShape.allIndices [p.BN] = lanes
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons, ih]
    by_cases hi : i.1.val < p.N
    · have hmiss : ¬ (r = p.output ∧ o = offset p batch i.1.val) :=
        fun he => h i.1 hi ⟨he.1.symm, he.2.symm⟩
      simp only [hi, decide_true, if_true, BlockState.writeMem_mem, if_neg hmiss]
    · simp [hi]

@[simp] theorem prepared_mem (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ) :
    (prepared p s batch g).mem = s.mem := rfl

theorem tail_readback (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ)
    (hstride : 0 < p.outputStride) (n : Fin p.BN) (hn : n.val < p.N) :
    (result p s batch g).readMem p.output (offset p batch n.val) =
      if p.splits = 1 then g n * p.scaling
      else s.readMem p.output (offset p batch n.val) + g n * p.scaling := by
  exact scatter_active p (prepared p s batch g) batch _ hstride n hn

theorem tail_frame (p : Parameters) (s : BlockState) (batch : Nat) (g : Fin p.BN → ℝ)
    (r : RegionName) (o : Nat)
    (h : ∀ n : Fin p.BN, n.val < p.N → ¬(p.output = r ∧ offset p batch n.val = o)) :
    (result p s batch g).mem r o = s.mem r o :=
  scatter_frame p (prepared p s batch g) batch _ r o h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealTail
