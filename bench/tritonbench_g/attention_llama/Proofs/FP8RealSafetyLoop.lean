import bench.tritonbench_g.attention_llama.Proofs.FP8RealSetup

/-! Actual byte-load address safety through the complete streaming loop.
Arithmetic registers may be arbitrary, and decode failures are allowed. Only
visited block starts and unmasked lanes impose allocation requirements. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSafetyLoop
open VeriTile.Triton RealKernel FP8RealIteration FP8RealLoop FP8RealLoopFrame
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

private theorem inline_safe (bounds : RegionBounds) (scale : ℝ)
    (skn svk N SP BM BN BD : Nat) (causal candidate : Bool) (s : BlockState) :
    Stmt.TraceSafeList bounds
      (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s ↔
    Stmt.TraceSafeList bounds (inlineLoop scale skn svk N SP BM BN BD causal candidate) s := by
  simp only [FP8RealSource.loopBody, inlineLoop, List.append_assoc, TraceRewrite.append_iff,
    TraceRewrite.constant_branch]

private theorem load_safe (bounds : RegionBounds) (R : RegionName) (s : BlockState)
    (name : RegName) (base sr sc SN N BN BD : Nat)
    (hptr : s.regs .ptr [BN, BD] name = some (alPtrTile R base sr sc SN BN BD))
    (hrows : s.regs .nat [BN] "block_n_offs" = some (Tile.vec (fun j : Fin BN => SN + j.val)))
    (hb : ∀ i : TileIndex [BN, BD], SN + i.1.val < N →
      base + (SN + i.1.val) * sr + i.2.1.val * sc < bounds R) :
    Op.SafeAt bounds s (.decodeFloat8E5 .real
      (.load .int (.ptr (.ref .ptr [BN, BD] name))
        (.maskOther (keyMask N BN BD) (.broadcast (.constInt 0) [BN, BD])))) := by
  simp only [Op.SafeAt.eq_def, keyMask, true_and, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe.eq_def,
    MaskOpt.Active]
  intro ptrs hp i hi
  rw [evalOp_ref, hptr] at hp
  cases hp
  rcases hi with ⟨mask, hm, hi⟩
  rw [al_colmask_eval s SN N hrows] at hm
  cases hm
  exact hb i (of_decide_eq_true hi)

/-- Both complete iterations access only the indicated masked K/V addresses.
No byte value or numerical accumulator equation is a premise. -/
theorem iteration_safe (bounds : RegionBounds) (K V : RegionName) (s : BlockState) (scale : ℝ)
    (kBase vBase skn skk svk svn SN N SP BM BN BD : Nat) (causal candidate : Bool)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoff : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hkp : s.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk SN BN BD))
    (hvp : s.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn SN BN BD))
    (hK : ∀ i : TileIndex [BN, BD], SN + i.1.val < N →
      kBase + (SN + i.1.val) * skn + i.2.1.val * skk < bounds K)
    (hV : ∀ i : TileIndex [BN, BD], SN + i.1.val < N →
      vBase + (SN + i.1.val) * svk + i.2.1.val * svn < bounds V) :
    Stmt.TraceSafeList bounds
      (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s := by
  have hoffEval : evalOp (.add .nat Broadcast.scalarL (.ref .nat [] "start_n")
      (.ref .nat [BN] "offs_n")) s = some (Tile.vec (fun j : Fin BN => SN + j.val)) := by
    rw [evalOp_add]
    simp only [evalOp_ref, hsn, hoff, Option.bind_eq_bind, Option.bind_some]
    apply congrArg some
    apply Tile.ext
    intro i
    rfl
  rw [inline_safe]
  cases causal <;> cases candidate
  all_goals
    simp only [inlineLoop, alLoopPrefixG, alLoopTailG, FP8RealSource.lowerLoad,
      FP8RealSource.byteLoad?, List.flatMap_cons, List.flatMap_nil, List.cons_append,
      List.nil_append, Bool.false_eq_true, ↓reduceIte, Bool.or_eq_true, beq_iff_eq,
      String.reduceEq, false_or, or_false, show (0.0 : ℝ) = 0 from by norm_num]
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some hoffEval)]
    refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
    apply TraceProofs.assign_cons
    · exact load_safe bounds K _ "k_ptrs" kBase skn skk SN N BN BD
        (by simpa using hkp) (by simp) hK
    · intro kt
      repeat' (apply TraceProofs.assign_cons; (solve | simp [Op.SafeAt.eq_def]); intro)
      apply TraceProofs.assign_cons
      · exact load_safe bounds V _ "v_ptrs" vBase svk svn SN N BN BD
          (by simpa using hvp) (by simp) hV
      · intro vt
        apply TraceProofs.all_safe
        simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

/-- Address bounds follow the actual range counter and the source's load mask. -/
def AccessesFrom (bounds : RegionBounds) (R : RegionName)
    (base sr sc cur stop N BN BD : Nat) : Prop :=
  ∀ k : Nat, cur + k * BN < stop → ∀ (j : Fin BN) (d : Fin BD),
    cur + k * BN + j.val < N →
      base + (cur + k * BN + j.val) * sr + d.val * sc < bounds R

private theorem accesses_next (bounds : RegionBounds) (R : RegionName)
    (base sr sc cur stop N BN BD : Nat)
    (h : AccessesFrom bounds R base sr sc cur stop N BN BD) :
    AccessesFrom bounds R base sr sc (cur + BN) stop N BN BD := by
  intro k hk j d hj
  have he : cur + BN + k * BN = cur + (k + 1) * BN := by
    simp only [Nat.add_mul, Nat.one_mul]
    omega
  rw [he] at hk hj ⊢
  exact h (k + 1) hk j d hj

/-- Bounds at every reached iteration, including partial key tails, zero
stride and evaluations that fail before the next iteration. -/
theorem range_safe (bounds : RegionBounds) (K V : RegionName) (initial s : BlockState) (scale : ℝ)
    (kBase vBase skn skk svk svn cur stop N SP BM BN BD : Nat) (causal candidate : Bool)
    (hc : Cursor K V initial kBase vBase skn skk svk svn BN BD cur s)
    (hK : AccessesFrom bounds K kBase skn skk cur stop N BN BD)
    (hV : AccessesFrom bounds V vBase svk svn cur stop N BN BD) :
    Stmt.forRangeTraceSafe bounds "start_n" cur stop BN
      (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s := by
  rw [Stmt.forRangeTraceSafe]
  by_cases hz : BN = 0
  · simp only [hz, ↓reduceIte]
  · simp only [hz, ↓reduceIte]
    by_cases hi : cur < stop
    · simp only [hi, ↓reduceIte]
      refine ⟨?_, ?_⟩
      · apply iteration_safe bounds K V _ scale kBase vBase skn skk svk svn cur N SP BM BN BD
          causal candidate (by simp) (by simpa using hc.offsets)
          (by simpa using hc.keys) (by simpa using hc.values)
        · intro i hj
          simpa using hK 0 (by simpa using hi) i.1 i.2.1 (by simpa using hj)
        · intro i hj
          simpa using hV 0 (by simpa using hi) i.1 i.2.1 (by simpa using hj)
      · cases he : stepStmts
            (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate)
            (s.setReg "start_n" .nat [] (Tile.scalar cur)) with
        | none => trivial
        | some t =>
          exact range_safe bounds K V initial t scale kBase vBase skn skk svk svn
            (cur + BN) stop N SP BM BN BD causal candidate
            (cursor_next K V initial s t scale kBase vBase skn skk svk svn cur N SP BM BN BD causal candidate hc he)
            (accesses_next bounds K kBase skn skk cur stop N BN BD hK)
            (accesses_next bounds V vBase svk svn cur stop N BN BD hV)
    · simp only [hi, ↓reduceIte]
termination_by stop - cur
decreasing_by omega

/-- A successful complete range retains the launch metadata used by Out. -/
theorem range_launch (initial s t : BlockState) (NH : Nat) (scale : ℝ)
    (skn svk cur stop N SP BM BN BD : Nat) (causal candidate : Bool)
    (hl : Launch initial NH s)
    (hrun : stepForRangeAux "start_n" cur stop BN
      (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s = some t) :
    Launch initial NH t := by
  by_cases hz : BN = 0
  · subst BN
    rw [stepForRangeAux.step_zero_step, Option.some.injEq] at hrun
    subst t
    exact hl
  · by_cases hi : cur < stop
    · rw [stepForRangeAux.step_lt hz hi] at hrun
      cases he : stepStmts
          (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate)
          (s.setReg "start_n" .nat [] (Tile.scalar cur)) with
      | none => simp [he] at hrun
      | some mid =>
        simp only [he, Option.bind_some] at hrun
        apply range_launch initial mid t NH scale skn svk (cur + BN) stop N SP BM BN BD causal candidate
          (hrun := hrun)
        apply launch_preserved initial _ mid NH scale skn svk N SP BM BN BD causal candidate (hrun := he)
        exact ⟨hl.pids, by simpa using hl.batch, by simpa using hl.head⟩
    · rw [stepForRangeAux.step_ge hz (by omega), Option.some.injEq] at hrun
      subst t
      exact hl
termination_by stop - cur
decreasing_by omega

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSafetyLoop
