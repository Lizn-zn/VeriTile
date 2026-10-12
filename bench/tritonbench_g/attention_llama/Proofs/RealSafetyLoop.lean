import bench.tritonbench_g.attention_llama.Proofs.RealSafetyPrologue

/-! Per-iteration safety from the already proved real execution invariants.
These internal invariants are established by source execution; callers supply
only bounds for the streamed K/V rows, not equations for any kernel output. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.RealSafetyLoop
open VeriTile.Triton RealKernel RealSafetyFragments
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

/-- The original non-causal loop's two accesses use the invariant's actual
pointer tiles. All intermediate numerical assignments are allocation-free. -/
theorem noncausal_iteration (bounds : RegionBounds) (Q K V : RegionName)
    (s0 : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H BM BN BD numKV : Nat)
    (hBN : 0 < BN) (i : Nat) (st : BlockState) (hilt : i < BN * numKV)
    (hinv : alInv Q K V s0 sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      NH H BM BN BD numKV i st)
    (hk : ∀ r, r < BN * numKV → ∀ d : Fin BD,
      alBase s0 NH skz skh + r * skn + d.val * skk < bounds K)
    (hv : ∀ r, r < BN * numKV → ∀ d : Fin BD,
      alBase s0 NH svz svh + r * svk + d.val * svn < bounds V) :
    Stmt.TraceSafeList bounds (alLoopBodyG sc skn svk (BN * numKV) BM BN BD)
      (st.setReg "start_n" .nat [] (Tile.scalar i)) := by
  rcases hinv with ⟨hpids, hmod, hile, hmp, hlp, hacc, hq, hoffm, hoffn, hbi, hoh,
    hkp, hvp, hundef, hmem⟩
  have hin : ∀ j : Fin BN, i + j.val < BN * numKV := by
    intro j
    have hi : i = (i / BN) * BN := (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)).symm
    have hn : i / BN < numKV := (Nat.div_lt_iff_lt_mul hBN).mpr (by nlinarith)
    have hmul := Nat.mul_le_mul_right BN hn
    have hj := j.isLt
    rw [Nat.succ_mul] at hmul
    nlinarith
  have hnum : 0 < numKV := by nlinarith
  let se := st.setReg "start_n" .nat [] (Tile.scalar i)
  obtain ⟨sPre, hpre, _, _, _, _, hprebn, _, _, _, _, _, _, _, _, _, hprevp⟩ :=
    al_loopPrefix_steps sc (BN * numKV) BM BN BD i se K (alBase s0 NH skz skh) skn skk
      (⟨fun idx => some (alQTileG s0 Q NH sqz sqh sqm sqk H BM BD idx)⟩)
      (fun idx => se.readMem K (alBase s0 NH skz skh + (i + idx.1.val) * skn + idx.2.1.val * skk))
      (Nat.le_mul_of_pos_right BN hnum) hin (by simp [se])
      (by simpa [se] using hoffn) (by simpa [se] using hq)
      (by simpa [se] using hkp) (by intro; rfl)
      (by simpa [se] using hundef)
  unfold alLoopBodyG
  apply TraceProofs.append_safe
  · exact prefix_safe bounds sc (BN * numKV) BM BN BD i se K (alBase s0 NH skz skh) skn skk
      (by simp [se]) (by simpa [se] using hoffn) (by simpa [se] using hkp)
      (fun idx hi => hk _ hi idx.2.1)
  · intro t ht
    rw [hpre] at ht
    cases ht
    exact tail_safe bounds skn svk (BN * numKV) BM BN BD i sPre V
      (alBase s0 NH svz svh) svn Bool.false hprebn
      (hprevp.trans (by simpa [se] using hvp)) (fun idx hi => hv _ hi idx.2.1)

/-- Causal masking changes qk but leaves the V pointer and global row mask
unchanged. The allocation window is the streamed span, not all N_CTX rows. -/
theorem causal_iteration (bounds : RegionBounds) (Q K V : RegionName)
    (s0 : BlockState) (sc : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N SP BM BN BD numC : Nat)
    (hBN : 0 < BN) (hspan : BN * numC ≤ N)
    (i : Nat) (st : BlockState) (hilt : i < BN * numC)
    (hinv : alcInv Q K V s0 sc sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      NH H SP BM BN BD numC i st)
    (hk : ∀ r, r < BN * numC → ∀ d : Fin BD,
      alBase s0 NH skz skh + r * skn + d.val * skk < bounds K)
    (hv : ∀ r, r < BN * numC → ∀ d : Fin BD,
      alBase s0 NH svz svh + r * svk + d.val * svn < bounds V) :
    Stmt.TraceSafeList bounds (alLoopBodyCausalG sc skn svk N SP BM BN BD)
      (st.setReg "start_n" .nat [] (Tile.scalar i)) := by
  rcases hinv with ⟨hpids, hmod, hile, hmp, hlp, hacc, hq, hoffm, hoffn, hbi, hoh,
    hkp, hvp, hundef, hmem⟩
  have hin : ∀ j : Fin BN, i + j.val < BN * numC := by
    intro j
    have hi : i = (i / BN) * BN := (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)).symm
    have hn : i / BN < numC := (Nat.div_lt_iff_lt_mul hBN).mpr (by nlinarith)
    have hmul := Nat.mul_le_mul_right BN hn
    have hj := j.isLt
    rw [Nat.succ_mul] at hmul
    nlinarith
  have hnum : 0 < numC := by nlinarith
  let se := st.setReg "start_n" .nat [] (Tile.scalar i)
  obtain ⟨sPre, hpre, _, _, _, _, hprebn, _, _, _, _, _, _, _, _, _, hprevp⟩ :=
    al_loopPrefix_steps sc N BM BN BD i se K (alBase s0 NH skz skh) skn skk
      (⟨fun idx => some (alQTileG s0 Q NH sqz sqh sqm sqk H BM BD idx)⟩)
      (fun idx => se.readMem K (alBase s0 NH skz skh + (i + idx.1.val) * skn + idx.2.1.val * skk))
      ((Nat.le_mul_of_pos_right BN hnum).trans hspan) (fun j => (hin j).trans_le hspan)
      (by simp [se]) (by simpa [se] using hoffn) (by simpa [se] using hq)
      (by simpa [se] using hkp) (by intro; rfl)
      (by simpa [se] using hundef)
  unfold alLoopBodyCausalG
  apply TraceProofs.append_safe
  · exact prefix_safe bounds sc N BM BN BD i se K (alBase s0 NH skz skh) skn skk
      (by simp [se]) (by simpa [se] using hoffn) (by simpa [se] using hkp)
      (fun idx _ => hk _ (hin idx.1) idx.2.1)
  · intro t ht
    rw [hpre] at ht
    cases ht
    unfold alCausalWhereStmt
    apply TraceProofs.assign_cons (by simp [Op.SafeAt.eq_def])
    intro qk
    exact tail_safe bounds skn svk N BM BN BD i (sPre.setReg "qk" .real [BM, BN] qk)
      V (alBase s0 NH svz svh) svn Bool.false
      (by simpa using hprebn)
      (by simpa using hprevp.trans (by simpa [se] using hvp))
      (fun idx _ => hv _ (hin idx.1) idx.2.1)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.RealSafetyLoop
