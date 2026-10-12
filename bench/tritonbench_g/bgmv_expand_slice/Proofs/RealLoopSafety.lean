import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealAccessSafety

/-! Complete dynamic-loop access safety. The premises bound actual masked
weight reads and output cells. The proof follows reached loop states and
reuses the successful execution/invariant, including zero-length loops. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealLoopSafety
open VeriTile.Triton RealLoop RealBlock RealBlockExecution RealAccessSafety RealFoundation
noncomputable section
set_option maxHeartbeats 2400000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

private def indexed (p : Geometry) (s : BlockState) (c : Nat) : BlockState :=
  (s.setReg "n" .nat [] (Tile.scalar (c*p.BN))).setReg "current_n" .nat [p.BN]
    (Tile.vec (fun i => c*p.BN+i.val))

private def weightMasked (p : Geometry) (s : BlockState) (c : Nat) : BlockState :=
  (indexed p s c).setReg "b_ptr_mask" .bool [p.BN,p.BK]
    ⟨fun i => decide (c*p.BN+i.1.val < p.length) && decide (i.2.1.val < p.K)⟩

private def masked (p : Geometry) (s : BlockState) (c : Nat) : BlockState :=
  (weightMasked p s c).setReg "c_mask" .bool [p.BN]
    (Tile.vec (fun i => decide (c*p.BN+i.val < p.length)))

private theorem loads_state (p : Geometry) (s : BlockState) (c : Nat) :
    afterLoads s p.weights p.weightBase p.rowStride p.colStride p.K p.length p.BN p.BK c =
      (masked p s c).setReg "tiled_b" .real [p.BN,p.BK]
        (weights (s.readMem p.weights) p.weightBase p.rowStride p.colStride p.K p.length (c*p.BN) p.BN p.BK) := rfl

private theorem weight_mask_eval (p : Geometry) (s : BlockState) (c : Nat)
    (hs : s.regs .nat [] "split_n_length" = some (Tile.scalar p.length))
    (hn : s.regs .nat [p.BN] "current_n" = some (Tile.vec (fun i => c*p.BN+i.val)))
    (hk : s.regs .nat [p.BK] "offset_k" = some (Tile.vec (fun k => k.val))) :
    evalOp (.boolAnd (.consR (.consL .nil))
      (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [p.BN] "current_n"))
        (.ref .nat [] "split_n_length"))
      (.lt .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [p.BK] "offset_k")) (.constNat p.K))) s =
      some ⟨fun i : TileIndex [p.BN,p.BK] => decide (c*p.BN+i.1.val < p.length) && decide (i.2.1.val < p.K)⟩ := by
  simpa [evalOp, hs] using bmask_eval p.K p.length p.BN p.BK c s hn hk

private theorem output_mask_eval (p : Geometry) (s : BlockState) (c : Nat)
    (hs : s.regs .nat [] "split_n_length" = some (Tile.scalar p.length))
    (hn : s.regs .nat [p.BN] "current_n" = some (Tile.vec (fun i => c*p.BN+i.val))) :
    evalOp (.lt .nat .scalarR (.ref .nat [p.BN] "current_n") (.ref .nat [] "split_n_length")) s =
      some (Tile.vec (fun i => decide (c*p.BN+i.val < p.length))) := by
  simpa [evalOp, hs] using cmask_eval p.length p.BN c s hn

theorem loads_safe (bounds : RegionBounds) (p : Geometry) (a : Fin p.BK → ℝ)
    (s : BlockState) (c : Nat) (h : p.Registers a s)
    (hb : ∀ (i : Fin p.BN) (k : Fin p.BK), c*p.BN+i.val < p.length → k.val < p.K →
      p.weightBase+(c*p.BN+i.val)*p.rowStride+k.val*p.colStride < bounds p.weights) :
    Stmt.TraceSafeListR .triv bounds (RealLoopSource.loadPrefix p.rowStride p.colStride p.K p.BN p.BK)
      (s.setReg "n" .nat [] (Tile.scalar (c*p.BN))) := by
  unfold RealLoopSource.loadPrefix
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun q1 hq1 => ?_)
  rw [stepStmtR_triv, stepStmt_assign_eq_some (currentn_eval p.BN c s h.offsetN)] at hq1
  obtain rfl := Option.some.inj hq1
  change Stmt.TraceSafeListR .triv bounds _ (indexed p s c)
  have hm1 := weight_mask_eval p (indexed p s c) c
    (by simpa [indexed] using h.split) (by simp [indexed]) (by simpa [indexed] using h.offsetK)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun q2 hq2 => ?_)
  rw [stepStmtR_triv, stepStmt_assign_eq_some hm1] at hq2
  obtain rfl := Option.some.inj hq2
  change Stmt.TraceSafeListR .triv bounds _ (weightMasked p s c)
  have hm2 := output_mask_eval p (weightMasked p s c) c
    (by simpa [weightMasked, indexed] using h.split) (by simp [weightMasked, indexed])
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun q3 hq3 => ?_)
  rw [stepStmtR_triv, stepStmt_assign_eq_some hm2] at hq3
  obtain rfl := Option.some.inj hq3
  change Stmt.TraceSafeListR .triv bounds _ (masked p s c)
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, true_and, and_true,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR, MaskOpt.ActiveR, evalOpR_triv]
  intro ptrs hptrs i hactive
  rw [bptrs_eval p.weights p.weightBase p.rowStride p.colStride p.BN p.BK c (masked p s c)
    (by simpa [masked, weightMasked, indexed] using h.weightPtr)
    (by simp [masked, weightMasked, indexed])
    (by simpa [masked, weightMasked, indexed] using h.offsetK)] at hptrs
  obtain rfl := Option.some.inj hptrs
  obtain ⟨mask, hmask, hi⟩ := hactive
  have hmask' : (masked p s c).regs .bool [p.BN,p.BK] "b_ptr_mask" =
      some ⟨fun i => decide (c*p.BN+i.1.val < p.length) && decide (i.2.1.val < p.K)⟩ := by
    simp [masked, weightMasked]
  rw [evalOp_ref, hmask'] at hmask
  obtain rfl := Option.some.inj hmask
  have hi' : c*p.BN+i.1.val < p.length ∧ i.2.1.val < p.K := by simpa using hi
  exact hb i.1 i.2.1 hi'.1 hi'.2

theorem block_safe (bounds : RegionBounds) (p : Geometry) (a : Fin p.BK → ℝ)
    (s : BlockState) (c : Nat) (add swapped : Bool) (h : p.Registers a s)
    (hb : ∀ (i : Fin p.BN) (k : Fin p.BK), c*p.BN+i.val < p.length → k.val < p.K →
      p.weightBase+(c*p.BN+i.val)*p.rowStride+k.val*p.colStride < bounds p.weights)
    (hy : ∀ i : Fin p.BN, c*p.BN+i.val < p.length →
      p.outputBase+(c*p.BN+i.val)*p.outputStride < bounds p.output) :
    Stmt.TraceSafeListR .triv bounds (p.body add swapped)
      (s.setReg "n" .nat [] (Tile.scalar (c*p.BN))) := by
  unfold Geometry.body RealLoopSource.body
  refine Stmt.TraceSafeListR.append_intro _ _ (loads_safe bounds p a s c h hb) (fun t ht => ?_)
  rw [stepStmtsR_triv, loads_exec s p.weights p.weightBase p.rowStride p.colStride p.K
    p.length p.BN p.BK c h.split h.offsetN h.offsetK h.weightPtr, loads_state] at ht
  obtain rfl := Option.some.inj ht
  apply tail_safe bounds _ (sums a (s.readMem p.weights) p.weightBase p.rowStride p.colStride
    p.K p.length (c*p.BN) p.BN) p.output p.outputBase p.outputStride p.length c p.BK add swapped
  · simpa [masked, weightMasked, indexed] using h.outputPtr
  · simp [masked, weightMasked, indexed]
  · simp [masked]
  · apply reduction_eval
    · simpa [masked, weightMasked, indexed] using h.input
    · simp
  · exact hy

/-- All reached iterations are safe under bounds for actual active global
rows and rank lanes. The invariant used here needs only the loop registers
and block counter; no output equation or value restriction is required. -/
theorem loop_safe (bounds : RegionBounds) (p : Geometry) (a : Fin p.BK → ℝ)
    (s : BlockState) (add swapped : Bool) (h : p.Registers a s)
    (hb : ∀ g, g < p.length → ∀ k : Fin p.BK, k.val < p.K →
      p.weightBase+g*p.rowStride+k.val*p.colStride < bounds p.weights)
    (hy : ∀ g, g < p.length → p.outputAddress g < bounds p.output) :
    (p.statement add swapped).TraceSafeR .triv bounds s := by
  simp only [Geometry.statement, Stmt.TraceSafeR, Op.SafeAtR.eq_def, true_and,
    evalOpR_triv, evalOp_constNat, evalOp_ref, h.split, Tile.scalar_data]
  apply Stmt.forRangeTraceSafeR_inv .triv bounds "n" p.length p.BN (p.body add swapped)
    (fun n t => p.BN ∣ n ∧ p.Registers a t)
  · intro n t _ ht
    obtain ⟨c, hc⟩ := ht.1
    have hn : n = c*p.BN := hc.trans (Nat.mul_comm _ _)
    clear hc
    subst n
    refine ⟨block_safe bounds p a t c add swapped ht.2
      (fun i k hi hk => hb _ hi k hk) (fun i hi => hy _ hi),
      p.after a t add c, ?_, ⟨c+1, by ring⟩, p.registers_after a t add c ht.2⟩
    rw [stepStmtsR_triv]
    exact block_exec t a p.weights p.output p.weightBase p.rowStride p.colStride
      p.outputBase p.outputStride p.K p.length p.BN c add swapped
      ht.2.split ht.2.offsetN ht.2.offsetK ht.2.weightPtr ht.2.outputPtr ht.2.input
  · exact ⟨dvd_zero _, h⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealLoopSafety
