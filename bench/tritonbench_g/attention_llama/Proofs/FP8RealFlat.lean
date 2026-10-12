import bench.tritonbench_g.attention_llama.Proofs.FP8RealOutput
import bench.tritonbench_g.attention_llama.Proofs.FP8RealSafety
import bench.tritonbench_g.attention_llama.Proofs.RealMemory

/-! Execute the actual byte source in one disjoint flat allocation. The
mathematical inputs are decoded from the original integer K/V cells, and the
complete cell frame includes original byte encodings and masked output lanes. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealFlat
open VeriTile.Triton RealKernel
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat)

/-- Successful flat execution, the independent noncausal formula and the full frame. -/
theorem noncausal_correct (A : FlatAlloc) (hd : A.Disjoint)
    (hregions : A.regions = [Q, K, V, Out])
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (candidate : Bool) (s : BlockState) (numKV : Nat)
    (hQK : Q ≠ K) (hQV : Q ≠ V) (hOK : Out ≠ K) (hOV : Out ≠ V)
    (hBM : 0 < BM) (hBN : 0 < BN) (hnum : 0 < numKV) (hN : N = BN * numKV)
    (houtinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
      (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hu : ∀ r o, s.undef r o = 0)
    (hkbytes : FP8RealLoop.BytesFrom K s (alBase s NH skz skh) skn skk 0 N N BN BD)
    (hvbytes : FP8RealLoop.BytesFrom V s (alBase s NH svz svh) svk svn 0 N N BN BD)
    (hQ : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alBase s NH sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < A.extent Q)
    (hK : ∀ r, r < BN * numKV → ∀ d : Fin BD,
      alBase s NH skz skh + r * skn + d.val * skk < A.extent K)
    (hV : ∀ r, r < BN * numKV → ∀ d : Fin BD,
      alBase s NH svz svh + r * svk + d.val * svn < A.extent V)
    (hO : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alOutOffset s NH soz soh som son BM i < A.extent Out) :
    ∃ sF, exec (A.flattenKernel (FP8RealSource.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD Bool.false candidate).eraseDType.toAlgKernel)
        (A.flattenState s) = some sF ∧
      (∀ idx : TileIndex [BM, BD], alActive s H BM idx →
        sF.readMem A.flat (A.addr Out (alOutOffset s NH soz soh som son BM idx)) =
        attentionReal (alQTileG s Q NH sqz sqh sqm sqk H BM BD)
          (FP8RealOutput.byteTile s K NH skz skh skn skk N BD)
          (FP8RealOutput.byteTile s V NH svz svh svk svn N BD) scale idx) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ idx : TileIndex [BM, BD], alActive s H BM idx →
        o ≠ A.addr Out (alOutOffset s NH soz soh som son BM idx)) →
        sF.mem r o = (A.flattenState s).mem r o) := by
  apply RealMemory.transport A hd hcov
  · exact FP8RealSource.flatten_supported ..
  · exact funext fun r => funext fun o => hu r o
  · apply FP8RealSafety.source_safe Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD Bool.false candidate A.extent s hu hQ
    · simpa only [FP8RealSetup.stop, ↓reduceIte, hN] using
        FP8RealSafety.whole_blocks A.extent K (alBase s NH skz skh) skn skk N BN BD numKV hK
    · simpa only [FP8RealSetup.stop, ↓reduceIte, hN] using
        FP8RealSafety.whole_blocks A.extent V (alBase s NH svz svh) svk svn N BN BD numKV hV
    · exact hO
  · rw [hregions]; simp
  · exact hO
  · exact FP8RealOutput.noncausal_correct Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD candidate s numKV
      hQK hQV hOK hOV hBM hBN hnum hN houtinj hu hkbytes hvbytes

/-- Successful flat execution, the independent causal formula and the full frame. -/
theorem causal_correct (A : FlatAlloc) (hd : A.Disjoint)
    (hregions : A.regions = [Q, K, V, Out])
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (candidate : Bool) (s : BlockState) (numC : Nat)
    (hQK : Q ≠ K) (hQV : Q ≠ V) (hOK : Out ≠ K) (hOV : Out ≠ V)
    (hBM : 0 < BM) (hBN : 0 < BN)
    (hspan : (s.pids 0 + 1) * BN + SP = BN * numC)
    (hspanle : BN * numC ≤ N) (hsp : SP ≤ s.pids 0 * BM)
    (houtinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
      (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hu : ∀ r o, s.undef r o = 0)
    (hkbytes : FP8RealLoop.BytesFrom K s (alBase s NH skz skh) skn skk 0 (BN * numC) N BN BD)
    (hvbytes : FP8RealLoop.BytesFrom V s (alBase s NH svz svh) svk svn 0 (BN * numC) N BN BD)
    (hQ : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alBase s NH sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < A.extent Q)
    (hK : ∀ r, r < BN * numC → ∀ d : Fin BD,
      alBase s NH skz skh + r * skn + d.val * skk < A.extent K)
    (hV : ∀ r, r < BN * numC → ∀ d : Fin BD,
      alBase s NH svz svh + r * svk + d.val * svn < A.extent V)
    (hO : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alOutOffset s NH soz soh som son BM i < A.extent Out) :
    ∃ sF, exec (A.flattenKernel (FP8RealSource.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD Bool.true candidate).eraseDType.toAlgKernel)
        (A.flattenState s) = some sF ∧
      (∀ idx : TileIndex [BM, BD], alActive s H BM idx →
        sF.readMem A.flat (A.addr Out (alOutOffset s NH soz soh som son BM idx)) =
        attentionRealCausalBlock (s.pids 0 * BM - SP)
          (alQTileG s Q NH sqz sqh sqm sqk H BM BD)
          (FP8RealOutput.byteTile s K NH skz skh skn skk (BN * numC) BD)
          (FP8RealOutput.byteTile s V NH svz svh svk svn (BN * numC) BD) scale idx) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ idx : TileIndex [BM, BD], alActive s H BM idx →
        o ≠ A.addr Out (alOutOffset s NH soz soh som son BM idx)) →
        sF.mem r o = (A.flattenState s).mem r o) := by
  apply RealMemory.transport A hd hcov
  · exact FP8RealSource.flatten_supported ..
  · exact funext fun r => funext fun o => hu r o
  · apply FP8RealSafety.source_safe Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD Bool.true candidate A.extent s hu hQ
    · simpa only [FP8RealSetup.stop, ↓reduceIte, hspan] using
        FP8RealSafety.whole_blocks A.extent K (alBase s NH skz skh) skn skk N BN BD numC hK
    · simpa only [FP8RealSetup.stop, ↓reduceIte, hspan] using
        FP8RealSafety.whole_blocks A.extent V (alBase s NH svz svh) svk svn N BN BD numC hV
    · exact hO
  · rw [hregions]; simp
  · exact hO
  · exact FP8RealOutput.causal_correct Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD candidate s numC
      hQK hQV hOK hOV hBM hBN hspan hspanle hsp houtinj hu hkbytes hvbytes

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealFlat
