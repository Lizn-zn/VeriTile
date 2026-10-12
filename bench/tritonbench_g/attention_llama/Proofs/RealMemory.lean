import bench.tritonbench_g.attention_llama.Proofs.RealOutput
import bench.tritonbench_g.attention_llama.Proofs.RealSafety

/-! Full typed-source real execution over a flat allocation. Numerical output,
active-address safety and the complete frame come from separate checked proofs.
Causal span conditions concern this program's IDs; no universal-ID premise or
fixed experimental shape is introduced. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.RealMemory
open VeriTile.Triton RealKernel
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

/-- Transport any proved regional real output and full cell frame through a
safe disjoint allocation. The input cell types are unrestricted. -/
theorem transport {α : Type} (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (k : Kernel) (hok : k.FlattenOk)
    (s : BlockState) (hu : s.undef = (fun _ _ => 0)) (hsafe : k.TraceSafe A.extent s)
    (Out : RegionName) (hout : Out ∈ A.regions) (write : α → Nat) (active : α → Prop)
    (expected : α → ℝ) (hb : ∀ i, active i → write i < A.extent Out)
    (hrun : ∃ t, exec k s = some t ∧ (∀ i, active i → t.readMem Out (write i) = expected i) ∧
      (∀ r o, (r ≠ Out ∨ ∀ i, active i → o ≠ write i) → t.mem r o = s.mem r o)) :
    ∃ t, exec (A.flattenKernel k) (A.flattenState s) = some t ∧
      (∀ i, active i → t.readMem A.flat (A.addr Out (write i)) = expected i) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ i, active i → o ≠ A.addr Out (write i)) →
        t.mem r o = (A.flattenState s).mem r o) := by
  obtain ⟨t, he, hv, hf⟩ := hrun
  refine ⟨A.flattenState t, ?_, ?_, ?_⟩
  · rw [A.exec_flatten hd hcov k s hsafe hok hu, he, Option.map_some]
  · intro i hi
    have h := A.flattenState_readMemAs hd t hout (hb i hi) .real
    change (A.flattenState t).readMemAs .real A.flat (A.addr Out (write i)) =
      t.readMemAs .real Out (write i) at h
    rw [BlockState.readMemAs_real, BlockState.readMemAs_real] at h
    exact (Option.some.inj h).trans (hv i hi)
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
        obtain ⟨r, o⟩ := p
        obtain ⟨_, hoeq, _⟩ := A.decode_sound hdec
        show A.trCell (t.mem r o) = A.trCell (s.mem r o)
        refine congrArg A.trCell (hf r o ?_)
        by_cases hro : r = Out
        · subst hro
          refine Or.inr fun i hi => ?_
          rcases hcond with hflat | hn
          · exact absurd rfl hflat
          · intro heq
            exact hn i hi (by rw [hoeq, heq])
        · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat)

/-- Safety of the actual typed source after precision erasure, for either variant. -/
theorem noncausal_safe (bounds : RegionBounds) (candidate : Bool) (s : BlockState) (numKV : Nat)
    (hBM : 0 < BM) (hBN : 0 < BN) (hN : N = BN * numKV)
    (hu : ∀ r o, s.undef r o = 0)
    (hQ : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alBase s NH sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < bounds Q)
    (hK : ∀ r, r < BN * numKV → ∀ d : Fin BD,
      alBase s NH skz skh + r * skn + d.val * skk < bounds K)
    (hV : ∀ r, r < BN * numKV → ∀ d : Fin BD,
      alBase s NH svz svh + r * svk + d.val * svn < bounds V)
    (hO : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alOutOffset s NH soz soh som son BM i < bounds Out) :
    (RealOutput.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate Bool.false).eraseDType.toAlgKernel.TraceSafe bounds s := by
  subst N
  have hm := RealSafety.noncausal_safe bounds Q K V Out s scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    NH H SP BM BN BD numKV hBM hBN hu hQ hK hV hO
  have ho := (RealSource.original_noncausal_safety Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H (BN * numKV) SP BM BN BD bounds s).mpr hm
  cases candidate
  · exact ho
  · exact (RealSource.candidate_safety Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H (BN * numKV) SP BM BN BD bounds Bool.false s).mpr ho

/-- Successful flat execution, the independent noncausal formula and the full frame. -/
theorem noncausal_correct (A : FlatAlloc) (hd : A.Disjoint)
    (hregions : A.regions = [Q, K, V, Out])
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (candidate : Bool) (s : BlockState) (numKV : Nat)
    (hBM : 0 < BM) (hBN : 0 < BN) (hnum : 0 < numKV) (hN : N = BN * numKV)
    (houtinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
      (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hu : ∀ r o, s.undef r o = 0)
    (hQ : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alBase s NH sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < A.extent Q)
    (hK : ∀ r, r < BN * numKV → ∀ d : Fin BD,
      alBase s NH skz skh + r * skn + d.val * skk < A.extent K)
    (hV : ∀ r, r < BN * numKV → ∀ d : Fin BD,
      alBase s NH svz svh + r * svk + d.val * svn < A.extent V)
    (hO : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alOutOffset s NH soz soh som son BM i < A.extent Out) :
    ∃ sF, exec (A.flattenKernel (RealOutput.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate Bool.false).eraseDType.toAlgKernel)
        (A.flattenState s) = some sF ∧
      (∀ idx : TileIndex [BM, BD], alActive s H BM idx →
        sF.readMem A.flat (A.addr Out (alOutOffset s NH soz soh som son BM idx)) =
        attentionReal (alQTileG s Q NH sqz sqh sqm sqk H BM BD)
          (alKTileG s K NH skz skh skn skk N BD)
          (alVTileG s V NH svz svh svk svn N BD) scale idx) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ idx : TileIndex [BM, BD], alActive s H BM idx →
        o ≠ A.addr Out (alOutOffset s NH soz soh som son BM idx)) →
        sF.mem r o = (A.flattenState s).mem r o) := by
  apply transport A hd hcov
  · cases candidate
    · exact RealSource.original_flatten_supported ..
    · exact RealSource.candidate_flatten_supported ..
  · exact funext fun r => funext fun o => hu r o
  · exact noncausal_safe Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD A.extent candidate s numKV
      hBM hBN hN hu hQ hK hV hO
  · rw [hregions]; simp
  · exact hO
  · exact RealOutput.noncausal_correct Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate s numKV
      hBM hBN hnum hN houtinj hu

/-- Safety of the actual typed source after precision erasure, for either variant. -/
theorem causal_safe (bounds : RegionBounds) (candidate : Bool) (s : BlockState) (numC : Nat)
    (hBM : 0 < BM) (hBN : 0 < BN)
    (hspan : (s.pids 0 + 1) * BN + SP = BN * numC)
    (hspanle : BN * numC ≤ N) (hsp : SP ≤ s.pids 0 * BM)
    (hu : ∀ r o, s.undef r o = 0)
    (hQ : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alBase s NH sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < bounds Q)
    (hK : ∀ r, r < BN * numC → ∀ d : Fin BD,
      alBase s NH skz skh + r * skn + d.val * skk < bounds K)
    (hV : ∀ r, r < BN * numC → ∀ d : Fin BD,
      alBase s NH svz svh + r * svk + d.val * svn < bounds V)
    (hO : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alOutOffset s NH soz soh som son BM i < bounds Out) :
    (RealOutput.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate Bool.true).eraseDType.toAlgKernel.TraceSafe bounds s := by
  have hm := RealSafety.causal_safe bounds Q K V Out s scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    NH H N SP BM BN BD numC hBM hBN hspan hspanle hsp hu hQ hK hV hO
  have ho := (RealSource.original_causal_safety Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD bounds s).mpr hm
  cases candidate
  · exact ho
  · exact (RealSource.candidate_safety Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD bounds Bool.true s).mpr ho

/-- Successful flat execution, the independent causal formula and the full frame. -/
theorem causal_correct (A : FlatAlloc) (hd : A.Disjoint)
    (hregions : A.regions = [Q, K, V, Out])
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (candidate : Bool) (s : BlockState) (numC : Nat)
    (hBM : 0 < BM) (hBN : 0 < BN)
    (hspan : (s.pids 0 + 1) * BN + SP = BN * numC)
    (hspanle : BN * numC ≤ N) (hsp : SP ≤ s.pids 0 * BM)
    (houtinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
      (s.pids 0 * BM + idx.1.val) * som + idx.2.1.val * son))
    (hu : ∀ r o, s.undef r o = 0)
    (hQ : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alBase s NH sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < A.extent Q)
    (hK : ∀ r, r < BN * numC → ∀ d : Fin BD,
      alBase s NH skz skh + r * skn + d.val * skk < A.extent K)
    (hV : ∀ r, r < BN * numC → ∀ d : Fin BD,
      alBase s NH svz svh + r * svk + d.val * svn < A.extent V)
    (hO : ∀ i : TileIndex [BM, BD], alActive s H BM i →
      alOutOffset s NH soz soh som son BM i < A.extent Out) :
    ∃ sF, exec (A.flattenKernel (RealOutput.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate Bool.true).eraseDType.toAlgKernel)
        (A.flattenState s) = some sF ∧
      (∀ idx : TileIndex [BM, BD], alActive s H BM idx →
        sF.readMem A.flat (A.addr Out (alOutOffset s NH soz soh som son BM idx)) =
        attentionRealCausalBlock (s.pids 0 * BM - SP)
          (alQTileG s Q NH sqz sqh sqm sqk H BM BD)
          (alKTileG s K NH skz skh skn skk (BN * numC) BD)
          (alVTileG s V NH svz svh svk svn (BN * numC) BD) scale idx) ∧
      (∀ r o, (r ≠ A.flat ∨ ∀ idx : TileIndex [BM, BD], alActive s H BM idx →
        o ≠ A.addr Out (alOutOffset s NH soz soh som son BM idx)) →
        sF.mem r o = (A.flattenState s).mem r o) := by
  apply transport A hd hcov
  · cases candidate
    · exact RealSource.original_flatten_supported ..
    · exact RealSource.candidate_flatten_supported ..
  · exact funext fun r => funext fun o => hu r o
  · exact causal_safe Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD A.extent candidate s numC
      hBM hBN hspan hspanle hsp hu hQ hK hV hO
  · rw [hregions]; simp
  · exact hO
  · exact RealOutput.causal_correct Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son NH H N SP BM BN BD candidate s numC
      hBM hBN hspan hspanle hsp houtinj hu

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.RealMemory
