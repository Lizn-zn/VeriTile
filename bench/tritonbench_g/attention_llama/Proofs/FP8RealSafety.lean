import bench.tritonbench_g.attention_llama.Proofs.FP8RealSafetyLoop

/-! Address safety of both complete byte-input sources. Bounds are derived
from the actual initialization, counter, input pointer advances and final
masked store, independently of the real attention formula. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSafety
open VeriTile.Triton RealKernel FP8RealLoop FP8RealLoopFrame FP8RealSafetyLoop
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal candidate : Bool)

/-- Safety of the actual erased byte source, including all intermediate
loads and the output store. Finite-byte assumptions and numerical equations
are unnecessary: a failed decode cannot execute later accesses. -/
theorem source_safe (bounds : RegionBounds) (s : BlockState)
    (hu : ∀ r o, s.undef r o = 0)
    (hQ : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alBase s NH sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < bounds Q)
    (hK : AccessesFrom bounds K (alBase s NH skz skh) skn skk 0
      (FP8RealSetup.stop s N SP BN causal) N BN BD)
    (hV : AccessesFrom bounds V (alBase s NH svz svh) svk svn 0
      (FP8RealSetup.stop s N SP BN causal) N BN BD)
    (hO : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alOutOffset s NH soz soh som son BM i < bounds Out) :
    (FP8RealSource.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨entry, he, hc, hb, hl⟩ := FP8RealSetup.runs_with_launch Q K V sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn NH H N SP BM BN BD causal s hu
  have hsetup : Stmt.TraceSafeList bounds (FP8RealSetup.body Q K V sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn NH H N SP BM BN BD causal) s := by
    unfold FP8RealSetup.body
    apply TraceProofs.append_safe
    · exact RealSafetyPrologue.preamble_safe bounds s Q K V sqz sqh sqm sqk skz skh skn skk
        svz svh svk svn NH H N BM BN BD hQ
    · intro t _
      rw [TraceRewrite.constant_branch]
      apply TraceProofs.all_safe
      cases causal <;> simp [Stmt.TraceSafe, Op.SafeAt.eq_def, alBnEndCausalStmt]
  have hloop : Stmt.TraceSafe bounds
      (.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
        (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate)) entry := by
    simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp_constNat, evalOp_ref, hb,
      Tile.scalar_data, true_and]
    exact range_safe bounds K V s entry scale (alBase s NH skz skh) (alBase s NH svz svh)
      skn skk svk svn 0 (FP8RealSetup.stop s N SP BN causal) N SP BM BN BD causal candidate hc hK hV
  apply (FP8RealSource.fused_safety Q K V Out scale sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate bounds s).mpr
  change Stmt.TraceSafeList bounds (FP8RealSource.body Q K V Out scale sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate Bool.true) s
  have hbody : FP8RealSource.body Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate Bool.true =
      FP8RealSetup.body Q K V sqz sqh sqm sqk skz skh skn skk svz svh svk svn NH H N SP BM BN BD causal ++
      .forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "block_n_end") (.constNat BN)
        (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) ::
        alPostLoopG Out soz soh som son H BM BD := by
    simp [FP8RealSource.body, FP8RealSetup.body, List.append_assoc]
  rw [hbody]
  apply TraceProofs.append_safe _ s hsetup
  intro entry' he'
  rw [he, Option.some.injEq] at he'
  subst entry'
  apply Stmt.TraceSafeList.cons_intro hloop
  intro mid hm
  have hr : stepForRangeAux "start_n" 0 (FP8RealSetup.stop s N SP BN causal) BN
      (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) entry = some mid := by
    simpa only [stepForRangeAux.forRangeDyn_unfold, evalOp_constNat, evalOp_ref, hb,
      bind, Option.bind_some, Tile.scalar_data] using hm
  have hf := range_launch s entry mid NH scale skn svk 0 (FP8RealSetup.stop s N SP BN causal)
    N SP BM BN BD causal candidate hl hr
  exact RealSafetyPrologue.postamble_safe bounds Out s soz soh som son NH H BM BD mid
    hf.pids hf.batch hf.head hO

end

/-- A whole-block span reduces the scheduled address requirements to ordinary
row bounds. N may be larger than the causal span. -/
theorem whole_blocks (bounds : RegionBounds) (R : RegionName)
    (base sr sc N BN BD T : Nat)
    (hb : ∀ r, r < BN * T → ∀ d : Fin BD,
      base + r * sr + d.val * sc < bounds R) :
    AccessesFrom bounds R base sr sc 0 (BN * T) N BN BD := by
  intro k hk j d _
  simp only [Nat.zero_add] at hk ⊢
  apply hb _ ?_ d
  have hkt : k < T := by nlinarith
  have hs := Nat.mul_le_mul_right BN (Nat.succ_le_of_lt hkt)
  have hj := j.isLt
  nlinarith

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealSafety
