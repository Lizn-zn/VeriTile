import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealLoopSource

/-! Execution of the complete real n-loop body. The source correspondence is
in RealLoopSource.source_loop. These lemmas include both masks, the weight
load, the optional old-output load, arithmetic and the masked store. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealBlockExecution
open VeriTile.Triton
open RealFoundation RealBlock RealLoopSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

private def current (s : BlockState) (BN c : Nat) : BlockState :=
  (s.setReg "n" .nat [] (Tile.scalar (c*BN))).setReg "current_n" .nat [BN]
    (Tile.vec (fun i => c*BN+i.val))
private def weightMask (s : BlockState) (K snl BN BK c : Nat) : BlockState :=
  (current s BN c).setReg "b_ptr_mask" .bool [BN,BK]
    ⟨fun i => decide (c*BN+i.1.val < snl) && decide (i.2.1.val < K)⟩
private def outputMask (s : BlockState) (K snl BN BK c : Nat) : BlockState :=
  (weightMask s K snl BN BK c).setReg "c_mask" .bool [BN]
    (Tile.vec (fun i => decide (c*BN+i.val < snl)))

def afterLoads (s : BlockState) (w : RegionName)
    (base rowStride colStride K snl BN BK c : Nat) : BlockState :=
  (outputMask s K snl BN BK c).setReg "tiled_b" .real [BN,BK]
    (weights (s.readMem w) base rowStride colStride K snl (c*BN) BN BK)

private theorem bmask_from_register (K snl BN BK c : Nat) (s : BlockState)
    (hs : s.regs .nat [] "split_n_length" = some (Tile.scalar snl))
    (hn : s.regs .nat [BN] "current_n" = some (Tile.vec (fun i => c*BN+i.val)))
    (hk : s.regs .nat [BK] "offset_k" = some (Tile.vec (fun k => k.val))) :
    evalOp (.boolAnd (.consR (.consL .nil))
      (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BN] "current_n"))
        (.ref .nat [] "split_n_length"))
      (.lt .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BK] "offset_k")) (.constNat K))) s =
    some ⟨fun i : TileIndex [BN,BK] => decide (c*BN+i.1.val < snl) && decide (i.2.1.val < K)⟩ := by
  have h := bmask_eval K snl BN BK c s hn hk
  simpa [evalOp, hs] using h

private theorem cmask_from_register (snl BN c : Nat) (s : BlockState)
    (hs : s.regs .nat [] "split_n_length" = some (Tile.scalar snl))
    (hn : s.regs .nat [BN] "current_n" = some (Tile.vec (fun i => c*BN+i.val))) :
    evalOp (.lt .nat .scalarR (.ref .nat [BN] "current_n") (.ref .nat [] "split_n_length")) s =
    some (Tile.vec (fun i => decide (c*BN+i.val < snl))) := by
  have h := cmask_eval snl BN c s hn
  simpa [evalOp, hs] using h

theorem loads_exec (s : BlockState) (w : RegionName)
    (base rowStride colStride K snl BN BK c : Nat)
    (hs : s.regs .nat [] "split_n_length" = some (Tile.scalar snl))
    (hn : s.regs .nat [BN] "offset_n" = some (Tile.vec (fun i => i.val)))
    (hk : s.regs .nat [BK] "offset_k" = some (Tile.vec (fun k => k.val)))
    (hp : s.regs .ptr [] "b_ptr" = some (Tile.scalar (w,base))) :
    stepStmts (loadPrefix rowStride colStride K BN BK)
      (s.setReg "n" .nat [] (Tile.scalar (c*BN))) =
    some (afterLoads s w base rowStride colStride K snl BN BK c) := by
  unfold loadPrefix
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (currentn_eval BN c s hn))]
  have h1 := bmask_from_register K snl BN BK c (current s BN c)
    (by simpa [current] using hs) (by simp [current]) (by simpa [current] using hk)
  change stepStmts _ (current s BN c) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h1)]
  have h2 := cmask_from_register snl BN c (weightMask s K snl BN BK c)
    (by simpa [weightMask, current] using hs) (by simp [weightMask, current])
  change stepStmts _ (weightMask s K snl BN BK c) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h2)]
  have h3 := tiledb_eval w base K snl rowStride colStride BN BK c (outputMask s K snl BN BK c)
    (by simpa [outputMask, weightMask, current] using hp)
    (by simp [outputMask, weightMask, current])
    (by simpa [outputMask, weightMask, current] using hk)
    (by simp [outputMask, weightMask])
  change stepStmts _ (outputMask s K snl BN BK c) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some h3), stepStmts.nil]
  simp [afterLoads, weights, outputMask, weightMask, current]
  norm_num

@[simp] theorem loads_mem (s : BlockState) (w r : RegionName)
    (base rowStride colStride K snl BN BK c o : Nat) :
    (afterLoads s w base rowStride colStride K snl BN BK c).mem r o = s.mem r o := by
  simp [afterLoads, outputMask, weightMask, current]

@[simp] theorem loads_read (s : BlockState) (w r : RegionName)
    (base rowStride colStride K snl BN BK c o : Nat) :
    (afterLoads s w base rowStride colStride K snl BN BK c).readMem r o = s.readMem r o := by
  simp [afterLoads, outputMask, weightMask, current]

def afterBlock (s : BlockState) (a : Fin BK → ℝ) (w out : RegionName)
    (base rowStride colStride outBase outStride K snl BN c : Nat) (add : Bool) : BlockState :=
  afterTail (afterLoads s w base rowStride colStride K snl BN BK c)
    (sums a (s.readMem w) base rowStride colStride K snl (c*BN) BN)
    out outBase outStride snl (c*BN) add

/-- Success of every statement in the actual body. Register premises describe
only the loaded input and loop setup, never any expected final output. -/
theorem block_exec (s : BlockState) (a : Fin BK → ℝ) (w out : RegionName)
    (base rowStride colStride outBase outStride K snl BN c : Nat) (add swapped : Bool)
    (hs : s.regs .nat [] "split_n_length" = some (Tile.scalar snl))
    (hn : s.regs .nat [BN] "offset_n" = some (Tile.vec (fun i => i.val)))
    (hk : s.regs .nat [BK] "offset_k" = some (Tile.vec (fun k => k.val)))
    (hp : s.regs .ptr [] "b_ptr" = some (Tile.scalar (w,base)))
    (hc : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out,outBase)))
    (ha : s.regs .real [BK] "tiled_a" = some ⟨fun i => some (a i.1)⟩) :
    stepStmts (body rowStride colStride outStride K BN BK add swapped)
      (s.setReg "n" .nat [] (Tile.scalar (c*BN))) =
    some (afterBlock s a w out base rowStride colStride outBase outStride K snl BN c add) := by
  rw [body, ProgramRewrite.append, loads_exec s w base rowStride colStride K snl BN BK c hs hn hk hp,
    Option.bind_some]
  apply tail_exec
  · simpa [afterLoads, outputMask, weightMask, current] using hc
  · simp [afterLoads, outputMask, weightMask, current]
  · simp [afterLoads, outputMask]
  · apply reduction_eval
    · simpa [afterLoads, outputMask, weightMask, current] using ha
    · simp [afterLoads]

theorem block_readback (s : BlockState) (a : Fin BK → ℝ) (w out : RegionName)
    (base rowStride colStride outBase outStride K snl BN c : Nat) (add : Bool)
    (hstride : 0 < outStride) (i : Fin BN) (hi : c*BN+i.val < snl) :
    (afterBlock s a w out base rowStride colStride outBase outStride K snl BN c add).readMem
      out (outBase+(c*BN+i.val)*outStride) =
      if add then dot a (s.readMem w) base rowStride colStride K (c*BN+i.val) +
        s.readMem out (outBase+(c*BN+i.val)*outStride)
      else dot a (s.readMem w) base rowStride colStride K (c*BN+i.val) := by
  rw [afterBlock, tail_readback _ _ _ _ _ _ _ _ hstride i hi]
  simp [sums, hi]

theorem block_frame (s : BlockState) (a : Fin BK → ℝ) (w out r : RegionName)
    (base rowStride colStride outBase outStride K snl BN c o : Nat) (add : Bool)
    (h : ∀ i : Fin BN, c*BN+i.val < snl → ¬(out = r ∧ outBase+(c*BN+i.val)*outStride = o)) :
    (afterBlock s a w out base rowStride colStride outBase outStride K snl BN c add).mem r o =
      s.mem r o := by
  rw [afterBlock, tail_frame _ _ _ _ _ _ _ _ _ _ h, loads_mem]

theorem block_register (s : BlockState) (a : Fin BK → ℝ) (w out : RegionName)
    (base rowStride colStride outBase outStride K snl BN c : Nat) (add : Bool)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (h : name ∉ (["n", "current_n", "b_ptr_mask", "c_mask", "tiled_b", "tiled_out", "accumulator"] : List RegName)) :
    (afterBlock s a w out base rowStride colStride outBase outStride K snl BN c add).regs dtype shape name =
      s.regs dtype shape name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at h
  rw [afterBlock, tail_register _ _ _ _ _ _ _ _ _ _ _ h.2.2.2.2.2.1 h.2.2.2.2.2.2]
  simp [afterLoads, outputMask, weightMask, current, BlockState.setReg,
    h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealBlockExecution
