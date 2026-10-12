import bench.tritonbench_g.attention_llama.Proofs.RealSource

/-! Successful real execution, independent attention formulas and the full
memory frame for both typed fp16 sources. This is a regional-memory result;
flat-allocation safety and the FP8 representation path are separate obligations.
The source's key-tail and causal-position restrictions are explicit below. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.RealOutput
open VeriTile.Triton RealKernel
set_option maxRecDepth 16000
set_option maxHeartbeats 500000

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat)

/-- Select the actual complete source, before erasing precision. -/
def source (candidate causal : Bool) : ComputeKernel :=
  if candidate then
    TypedSource.carryFirst Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal
  else
    TypedSource.original Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD causal

private theorem noncausal_execution (candidate : Bool) (s : BlockState) :
    exec (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate Bool.false).eraseDType.toAlgKernel s =
      exec (RealKernel.noncausal Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD).toAlgKernel s := by
  cases candidate
  · exact RealSource.original_noncausal_execution ..
  · exact (RealSource.candidate_execution ..).trans (RealSource.original_noncausal_execution ..)

private theorem causal_execution (candidate : Bool) (s : BlockState) :
    exec (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate Bool.true).eraseDType.toAlgKernel s =
      exec (RealKernel.causal Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD).toAlgKernel s := by
  cases candidate
  · exact RealSource.original_causal_execution ..
  · exact (RealSource.candidate_execution ..).trans (RealSource.original_causal_execution ..)

/-- Whole key blocks compute the independent natural-exp weighted sum. Both
source variants succeed and preserve every cell outside the active output. -/
theorem noncausal_correct (candidate : Bool) (s : BlockState) (numKV : Nat)
    (hBM : 0 < BM) (hBN : 0 < BN) (hnum : 0 < numKV) (hN : N = BN * numKV)
    (houtinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
      (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hundef : ∀ r o, s.undef r o = 0) :
    ∃ sF, exec (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate Bool.false).eraseDType.toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BM, BD], alActive s H BM idx →
        sF.readMem Out (alOutOffset s NH soz soh som son BM idx) =
          attentionReal (alQTileG s Q NH sqz sqh sqm sqk H BM BD)
            (alKTileG s K NH skz skh skn skk N BD)
            (alVTileG s V NH svz svh svk svn N BD) scale idx)
      ∧ (∀ r o, (r ≠ Out ∨ ∀ idx : TileIndex [BM, BD], alActive s H BM idx →
          o ≠ alOutOffset s NH soz soh som son BM idx) → sF.mem r o = s.mem r o) := by
  subst N
  rw [noncausal_execution]
  obtain ⟨sF, hrun, hread, hframe⟩ := al_exec Q K V Out s scale
    sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    NH H SP BM BN BD numKV hBM hBN houtinj hundef
  refine ⟨sF, hrun, ?_, hframe⟩
  intro idx hi
  rw [hread idx hi]
  have hl : StreamingAccumulator.lPartial (alQTileG s Q NH sqz sqh sqm sqk H BM BD)
      numKV (alKTileG s K NH skz skh skn skk (BN * numKV) BD) scale numKV idx.1 ≠ 0 := by
    obtain ⟨k, rfl⟩ : ∃ k, numKV = k + 1 := ⟨numKV - 1, by omega⟩
    exact al_lPartial_succ_ne_zero hBN _ _ _ k (le_refl _) idx.1
  exact StreamingAccumulator.streaming_eq_attentionReal hBN _ numKV hnum _ _ scale idx hl

/-- The source's causal predicate is key + SP ≤ query. The streamed span
must be whole blocks within N, with SP no larger than the first query row. -/
theorem causal_correct (candidate : Bool) (s : BlockState) (numC : Nat)
    (hBM : 0 < BM) (hBN : 0 < BN)
    (hspan : (s.pids 0 + 1) * BN + SP = BN * numC)
    (hspanle : BN * numC ≤ N) (hsp : SP ≤ s.pids 0 * BM)
    (houtinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
      (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hundef : ∀ r o, s.undef r o = 0) :
    ∃ sF, exec (source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate Bool.true).eraseDType.toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BM, BD], alActive s H BM idx →
        sF.readMem Out (alOutOffset s NH soz soh som son BM idx) =
          attentionRealCausalBlock (s.pids 0 * BM - SP)
            (alQTileG s Q NH sqz sqh sqm sqk H BM BD)
            (alKTileG s K NH skz skh skn skk (BN * numC) BD)
            (alVTileG s V NH svz svh svk svn (BN * numC) BD) scale idx)
      ∧ (∀ r o, (r ≠ Out ∨ ∀ idx : TileIndex [BM, BD], alActive s H BM idx →
          o ≠ alOutOffset s NH soz soh som son BM idx) → sF.mem r o = s.mem r o) := by
  rw [causal_execution]
  obtain ⟨sF, hrun, hread, hframe⟩ := alc_exec Q K V Out s scale
    sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    NH H N SP BM BN BD numC hBM hBN hspan hspanle hsp houtinj hundef
  refine ⟨sF, hrun, ?_, hframe⟩
  intro idx hi
  rw [hread idx hi]
  have hn : 0 < numC := by nlinarith [Nat.zero_le (s.pids 0), Nat.zero_le SP]
  rw [VeriTile.Examples.FA1MathCausal.streaming_eq_attentionRealCausalBlock hBN (s.pids 0 * BM - SP)
    (alQTileG s Q NH sqz sqh sqm sqk H BM BD) numC hn
    (alKTileG s K NH skz skh skn skk (BN * numC) BD)
    (alVTileG s V NH svz svh svk svn (BN * numC) BD) scale idx]
  exact alc_attentionRealCausalBlock_eq _ _ _ _ _ _

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.RealOutput
