import bench.tritonbench_g.attention_llama.Proofs.FP8RealLoopFrame

/-! Lift the checked FP8 iteration through the full dynamic range. Byte
requirements refer to the actual visited block starts and active source masks.
The invariant tracks raw memory and pointer positions, not output equations. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealLoop
open VeriTile.Triton RealKernel FP8RealMemory FP8RealIteration
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- The input addresses at the next streaming counter. Numeric accumulator
registers are deliberately unrestricted by this structural invariant. -/
structure Cursor (K V : RegionName) (initial : BlockState)
    (kBase vBase skn skk svk svn BN BD i : Nat) (s : BlockState) : Prop where
  memory : s.mem = initial.mem
  offsets : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))
  keys : s.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk i BN BD)
  values : s.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn i BN BD)

/-- Finite signed bytes are needed only for reached block starts and active
lanes. Inactive cells are not constrained, including the final partial block. -/
def BytesFrom (R : RegionName) (s : BlockState)
    (base srow scol cur stop N BN BD : Nat) : Prop :=
  ∀ k : Nat, cur + k * BN < stop → ∀ (j : Fin BN) (d : Fin BD),
    cur + k * BN + j.val < N →
      FiniteByte (s.readMemValue .int R
        (base + (cur + k * BN + j.val) * srow + d.val * scol))

private theorem bytes_next (R : RegionName) (s : BlockState)
    (base srow scol cur stop N BN BD : Nat)
    (h : BytesFrom R s base srow scol cur stop N BN BD) :
    BytesFrom R s base srow scol (cur + BN) stop N BN BD := by
  intro k hk j d hj
  have he : cur + BN + k * BN = cur + (k + 1) * BN := by
    simp only [Nat.add_mul, Nat.one_mul]
    omega
  rw [he] at hk hj ⊢
  exact h (k + 1) hk j d hj

theorem cursor_next (K V : RegionName) (initial s t : BlockState) (scale : ℝ)
    (kBase vBase skn skk svk svn i N SP BM BN BD : Nat) (causal candidate : Bool)
    (hc : Cursor K V initial kBase vBase skn skk svk svn BN BD i s)
    (hrun : stepStmts (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate)
      (s.setReg "start_n" .nat [] (Tile.scalar i)) = some t) :
    Cursor K V initial kBase vBase skn skk svk svn BN BD (i + BN) t := by
  have hf := FP8RealLoopFrame.registers K V _ t scale
    kBase vBase skn skk svk svn i N SP BM BN BD causal candidate
    (by simpa using hc.keys) (by simpa using hc.values) hrun
  refine ⟨?_, ?_, hf.2.1, hf.2.2⟩
  · exact (memory_frame scale skn svk N SP BM BN BD causal candidate _ t hrun).trans hc.memory
  · simpa using hf.1.trans (by simpa using hc.offsets)

/-- Every reached iteration of the complete FP8 range has exactly the same
real execution on the decoded input view. The conclusion includes empty
ranges, zero stride and intermediate evaluation failure. -/
theorem range (K V : RegionName) (initial s : BlockState) (scale : ℝ)
    (kBase vBase skn skk svk svn cur stop N SP BM BN BD : Nat) (causal candidate : Bool)
    (hc : Cursor K V initial kBase vBase skn skk svk svn BN BD cur s)
    (hK : BytesFrom K initial kBase skn skk cur stop N BN BD)
    (hV : BytesFrom V initial vBase svk svn cur stop N BN BD) :
    (stepForRangeAux "start_n" cur stop BN
      (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s).map
        (decoded K V) =
      stepForRangeAux "start_n" cur stop BN
        (referenceLoop scale skn svk N SP BM BN BD causal candidate) (decoded K V s) := by
  by_cases hz : BN = 0
  · subst BN
    simp only [stepForRangeAux.step_zero_step, Option.map_some]
  · by_cases hi : cur < stop
    · have hread (r : RegionName) (o : Nat) :
          (s.setReg "start_n" .nat [] (Tile.scalar cur)).readMemValue .int r o =
            initial.readMemValue .int r o := by
        simp only [BlockState.setReg_readMemValue]
        unfold BlockState.readMemValue BlockState.readMemTyped
        rw [hc.memory]
      have hbody := iteration K V (s.setReg "start_n" .nat [] (Tile.scalar cur)) scale
        kBase vBase skn skk svk svn cur N SP BM BN BD causal candidate
        (by simp) (by simpa using hc.offsets) (by simpa using hc.keys) (by simpa using hc.values)
        (by
          intro j d hj
          rw [hread]
          simpa only [Nat.zero_mul, Nat.add_zero] using hK 0 (by simpa using hi) j d (by simpa using hj))
        (by
          intro j d hj
          rw [hread]
          simpa only [Nat.zero_mul, Nat.add_zero] using hV 0 (by simpa using hi) j d (by simpa using hj))
      rw [stepForRangeAux.step_lt hz hi, stepForRangeAux.step_lt hz hi]
      rw [← decoded_setReg, ← hbody]
      cases he : stepStmts
          (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate)
          (s.setReg "start_n" .nat [] (Tile.scalar cur)) with
      | none => simp
      | some t =>
        simp only [Option.map_some, Option.bind_some]
        exact range K V initial t scale kBase vBase skn skk svk svn (cur + BN) stop N SP BM BN BD
          causal candidate (cursor_next K V initial s t scale
            kBase vBase skn skk svk svn cur N SP BM BN BD causal candidate hc he)
          (bytes_next K initial kBase skn skk cur stop N BN BD hK)
          (bytes_next V initial vBase svk svn cur stop N BN BD hV)
    · rw [stepForRangeAux.step_ge hz (by omega), stepForRangeAux.step_ge hz (by omega)]
      rfl
termination_by stop - cur
decreasing_by omega

/-- Lift the range result to the actual source loop statement, whose stop is
read from block_n_end. That register is shared by both input representations. -/
theorem dynamic_range (K V : RegionName) (initial s : BlockState) (scale : ℝ)
    (kBase vBase skn skk svk svn stop N SP BM BN BD : Nat) (causal candidate : Bool)
    (hc : Cursor K V initial kBase vBase skn skk svk svn BN BD 0 s)
    (hstop : s.regs .nat [] "block_n_end" = some (Tile.scalar stop))
    (hK : BytesFrom K initial kBase skn skk 0 stop N BN BD)
    (hV : BytesFrom V initial vBase svk svn 0 stop N BN BD) :
    (stepStmt (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
      (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate)) s).map
        (decoded K V) =
      stepStmt (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
        (referenceLoop scale skn svk N SP BM BN BD causal candidate)) (decoded K V s) := by
  have hstopView : (decoded K V s).regs .nat [] "block_n_end" = some (Tile.scalar stop) := hstop
  simpa only [stepForRangeAux.forRangeDyn_unfold, evalOp_constNat, evalOp_ref,
    hstop, hstopView, bind, Option.bind_some, Tile.scalar_data] using
    range K V initial s scale kBase vBase skn skk svk svn 0 stop N SP BM BN BD causal candidate hc hK hV

/-- The complete range preserves every raw memory cell, independently of the
numerical values and of any decoded-memory equality. -/
theorem range_memory (s t : BlockState) (scale : ℝ)
    (skn svk cur stop N SP BM BN BD : Nat) (causal candidate : Bool)
    (hrun : stepForRangeAux "start_n" cur stop BN
      (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s = some t) :
    t.mem = s.mem := by
  by_cases hz : BN = 0
  · subst BN
    rw [stepForRangeAux.step_zero_step, Option.some.injEq] at hrun
    subst t
    rfl
  · by_cases hi : cur < stop
    · rw [stepForRangeAux.step_lt hz hi] at hrun
      cases he : stepStmts
          (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate)
          (s.setReg "start_n" .nat [] (Tile.scalar cur)) with
      | none => simp [he] at hrun
      | some u =>
        simp only [he, Option.bind_some] at hrun
        exact (range_memory u t scale skn svk (cur + BN) stop N SP BM BN BD causal candidate hrun).trans
          (memory_frame scale skn svk N SP BM BN BD causal candidate
            (s.setReg "start_n" .nat [] (Tile.scalar cur)) u he)
    · rw [stepForRangeAux.step_ge hz (by omega), Option.some.injEq] at hrun
      subst t
      rfl
termination_by stop - cur
decreasing_by omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealLoop
