import bench.tritonbench_g.attention_llama.Proofs.FP8RealExecution

/-! Real attention on the exact finite E5M2 interpretation of the original
K/V bytes. These theorems execute the byte source itself and preserve raw
memory outside its active output, including masked output lanes. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealOutput
open VeriTile.Triton RealKernel FP8RealMemory FP8RealLoop
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

/-- Mathematical input tile obtained by decoding the stored signed bytes. -/
noncomputable def byteTile (s : BlockState) (R : RegionName)
    (NH sz sh sr sc SEQ BD : Nat) : TileIndex [SEQ, BD] → ℝ :=
  fun i => value s R (alBase s NH sz sh + i.1.val * sr + i.2.1.val * sc)

private theorem query_view (Q K V : RegionName) (s : BlockState)
    (NH sqz sqh sqm sqk H BM BD : Nat) (hQK : Q ≠ K) (hQV : Q ≠ V) :
    alQTileG (decoded K V s) Q NH sqz sqh sqm sqk H BM BD =
      alQTileG s Q NH sqz sqh sqm sqk H BM BD := by
  funext i
  simp only [alQTileG, alRow, alBase, decoded]
  split
  · unfold BlockState.readMem
    simp only [hQK, hQV, or_self, ↓reduceIte]
  · rfl

private theorem key_view (K V : RegionName) (s : BlockState)
    (NH sz sh sr sc SEQ BD : Nat) :
    alKTileG (decoded K V s) K NH sz sh sr sc SEQ BD = byteTile s K NH sz sh sr sc SEQ BD := by
  funext i
  exact read_decoded K V s K _ (Or.inl rfl)

private theorem value_view (K V : RegionName) (s : BlockState)
    (NH sz sh sr sc SEQ BD : Nat) :
    alVTileG (decoded K V s) V NH sz sh sr sc SEQ BD = byteTile s V NH sz sh sr sc SEQ BD := by
  funext i
  exact read_decoded K V s V _ (Or.inr rfl)

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat)

private theorem transfer_result (causal candidate : Bool) (s : BlockState)
    (hQK : Q ≠ K) (hQV : Q ≠ V) (hOK : Out ≠ K) (hOV : Out ≠ V)
    (hundef : ∀ r o, s.undef r o = 0)
    (hK : BytesFrom K s (alBase s NH skz skh) skn skk 0 (FP8RealSetup.stop s N SP BN causal) N BN BD)
    (hV : BytesFrom V s (alBase s NH svz svh) svk svn 0 (FP8RealSetup.stop s N SP BN causal) N BN BD)
    (formula : TileIndex [BM, BD] → ℝ)
    (href : ∃ t, exec (RealOutput.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk
        svz svh svk svn soz soh som son NH H N SP BM BN BD candidate causal).eraseDType.toAlgKernel
        (decoded K V s) = some t ∧
      (∀ i, alActive s H BM i → t.readMem Out (alOutOffset s NH soz soh som son BM i) = formula i) ∧
      (∀ r o, (r ≠ Out ∨ ∀ i : TileIndex [BM, BD], alActive s H BM i →
        o ≠ alOutOffset s NH soz soh som son BM i) → t.mem r o = (decoded K V s).mem r o)) :
    ∃ t, exec (FP8RealSource.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk
        svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).eraseDType.toAlgKernel s = some t ∧
      (∀ i, alActive s H BM i → t.readMem Out (alOutOffset s NH soz soh som son BM i) = formula i) ∧
      (∀ r o, (r ≠ Out ∨ ∀ i : TileIndex [BM, BD], alActive s H BM i →
        o ≠ alOutOffset s NH soz soh som son BM i) → t.mem r o = s.mem r o) := by
  obtain ⟨ref, href, hout, hframe⟩ := href
  have hmap := FP8RealExecution.decoded_execution Q K V Out scale sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate s hQK hQV hOK hOV hundef hK hV
  rw [href] at hmap
  cases he : exec (FP8RealSource.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).eraseDType.toAlgKernel s with
  | none => simp [he] at hmap
  | some t =>
    simp only [he, Option.map_some, Option.some.injEq] at hmap
    subst ref
    refine ⟨t, rfl, ?_, ?_⟩
    · intro i hi
      have hv := hout i hi
      unfold BlockState.readMem at hv ⊢
      rw [outside_memory K V t Out _ hOK hOV] at hv
      exact hv
    · intro r o ho
      by_cases hr : r = K ∨ r = V
      · have hro : r ≠ Out := by
          rcases hr with rfl | rfl
          · exact Ne.symm hOK
          · exact Ne.symm hOV
        exact FP8RealExecution.memory_frame Q K V Out scale sqz sqh sqm sqk skz skh skn skk
          svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate s t r o hro hundef he
      · have hf := hframe r o ho
        rw [outside_memory K V t r o (fun h => hr (Or.inl h)) (fun h => hr (Or.inr h)),
          outside_memory K V s r o (fun h => hr (Or.inl h)) (fun h => hr (Or.inr h))] at hf
        exact hf

/-- Whole key blocks compute ordinary natural-exp attention on the decoded
byte inputs. Both actual source variants succeed and preserve every cell
outside the active output, with the original byte encoding retained. -/
theorem noncausal_correct (candidate : Bool) (s : BlockState) (numKV : Nat)
    (hQK : Q ≠ K) (hQV : Q ≠ V) (hOK : Out ≠ K) (hOV : Out ≠ V)
    (hBM : 0 < BM) (hBN : 0 < BN) (hnum : 0 < numKV) (hN : N = BN * numKV)
    (houtinj : Function.Injective (fun i : TileIndex [BM, BD] =>
      (s.pids 0 * BM + i.1.val) * som + i.2.1.val * son))
    (hundef : ∀ r o, s.undef r o = 0)
    (hK : BytesFrom K s (alBase s NH skz skh) skn skk 0 N N BN BD)
    (hV : BytesFrom V s (alBase s NH svz svh) svk svn 0 N N BN BD) :
    ∃ t, exec (FP8RealSource.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk
        svz svh svk svn soz soh som son NH H N SP BM BN BD Bool.false candidate).eraseDType.toAlgKernel s = some t ∧
      (∀ i : TileIndex [BM, BD], alActive s H BM i →
        t.readMem Out (alOutOffset s NH soz soh som son BM i) =
          attentionReal (alQTileG s Q NH sqz sqh sqm sqk H BM BD)
            (byteTile s K NH skz skh skn skk N BD)
            (byteTile s V NH svz svh svk svn N BD) scale i) ∧
      (∀ r o, (r ≠ Out ∨ ∀ i : TileIndex [BM, BD], alActive s H BM i →
        o ≠ alOutOffset s NH soz soh som son BM i) → t.mem r o = s.mem r o) := by
  apply transfer_result Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn
    soz soh som son NH H N SP BM BN BD Bool.false candidate s hQK hQV hOK hOV hundef hK hV
  have h := RealOutput.noncausal_correct Q K V Out scale sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn soz soh som son NH H N SP BM BN BD candidate (decoded K V s) numKV
    hBM hBN hnum hN houtinj hundef
  rw [query_view Q K V s NH sqz sqh sqm sqk H BM BD hQK hQV, key_view, value_view] at h
  exact h

/-- Causal attention retains the source's key + SP ≤ query predicate and its
whole-block streamed span. This is an independent formula on decoded inputs,
with successful execution and a complete raw-memory frame for either variant. -/
theorem causal_correct (candidate : Bool) (s : BlockState) (numC : Nat)
    (hQK : Q ≠ K) (hQV : Q ≠ V) (hOK : Out ≠ K) (hOV : Out ≠ V)
    (hBM : 0 < BM) (hBN : 0 < BN)
    (hspan : (s.pids 0 + 1) * BN + SP = BN * numC)
    (hspanle : BN * numC ≤ N) (hsp : SP ≤ s.pids 0 * BM)
    (houtinj : Function.Injective (fun i : TileIndex [BM, BD] =>
      (s.pids 0 * BM + i.1.val) * som + i.2.1.val * son))
    (hundef : ∀ r o, s.undef r o = 0)
    (hK : BytesFrom K s (alBase s NH skz skh) skn skk 0 (BN * numC) N BN BD)
    (hV : BytesFrom V s (alBase s NH svz svh) svk svn 0 (BN * numC) N BN BD) :
    ∃ t, exec (FP8RealSource.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk
        svz svh svk svn soz soh som son NH H N SP BM BN BD Bool.true candidate).eraseDType.toAlgKernel s = some t ∧
      (∀ i : TileIndex [BM, BD], alActive s H BM i →
        t.readMem Out (alOutOffset s NH soz soh som son BM i) =
          attentionRealCausalBlock (s.pids 0 * BM - SP)
            (alQTileG s Q NH sqz sqh sqm sqk H BM BD)
            (byteTile s K NH skz skh skn skk (BN * numC) BD)
            (byteTile s V NH svz svh svk svn (BN * numC) BD) scale i) ∧
      (∀ r o, (r ≠ Out ∨ ∀ i : TileIndex [BM, BD], alActive s H BM i →
        o ≠ alOutOffset s NH soz soh som son BM i) → t.mem r o = s.mem r o) := by
  apply transfer_result Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn
    soz soh som son NH H N SP BM BN BD Bool.true candidate s hQK hQV hOK hOV hundef
      (by simpa only [FP8RealSetup.stop, ↓reduceIte, hspan] using hK)
      (by simpa only [FP8RealSetup.stop, ↓reduceIte, hspan] using hV)
  have h := RealOutput.causal_correct Q K V Out scale sqz sqh sqm sqk skz skh skn skk
    svz svh svk svn soz soh som son NH H N SP BM BN BD candidate (decoded K V s) numC
    hBM hBN hspan hspanle hsp houtinj hundef
  rw [query_view Q K V s NH sqz sqh sqm sqk H BM BD hQK hQV, key_view, value_view] at h
  exact h

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealOutput
