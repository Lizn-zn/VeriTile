import bench.tritonbench_g.attention_llama.Proofs.FP8RealFlat
import bench.tritonbench_g.attention_llama.Proofs.RealIO
import VeriTile.Triton.Memory.KernelSpec.StreamCellsReal

/-! Independent array specification for real Q and raw signed-byte K/V input.
The input contract pins the original integer cells; only the mathematical
formula decodes those arrays to E5M2 real values. Both complete sources execute
over a disjoint flat allocation, with the full original-cell frame. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealIO
open VeriTile.Triton RealKernel
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

/-- Signed E5M2 input with a finite real interpretation. NaNs, infinities and
integers outside the signed-byte range have no real-correctness contract. -/
abbrev Byte := { value : Int // FP8RealMemory.FiniteByte value }

def byteCell (b : Byte) : MemCell := .of .int b.val

noncomputable def decode (b : Byte) : ℝ :=
  (((Float8E5Bits.ofInt b.val).decodeRat.getD 0 : ℚ) : ℝ)

/-- The source keeps the fp16-path addresses and masks but reads integer K/V
cells and decodes them inside the program. No input buffer is replaced. -/
def io (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool) : StreamMasked3DKernelIO₃ :=
  { RealIO.io Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn
      soz soh som son NH H N SP BM BN BD T candidate causal with
    kernel := (FP8RealSource.source Q K V Out scale sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son NH H N SP BM BN BD causal candidate).eraseDType
    projection := FP8RealSource.projection_supported .. }

/-- Independent natural-exp attention over Q and the exact decoded input
arrays. Causal mode uses the source's shifted visibility predicate. -/
noncomputable def expected (T BM BN BD H SP : Nat) (hBN : 0 < BN) (hT : 0 < T)
    (causal : Bool) (scale : ℝ) (p : Nat)
    (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys zs : Fin T → Fin (BN * BD) → Byte) (j : Fin (BM * BD)) : ℝ :=
  RealIO.expected T BM BN BD H SP hBN hT causal scale p xs
    (fun t i => decode (ys t i)) (fun t i => decode (zs t i)) j

private theorem pinned_byte (s : BlockState) (R : RegionName) (o : Nat) (b : Byte)
    (h : s.mem R o = byteCell b) : s.readMemValue .int R o = b.val := by
  simp [BlockState.readMemValue, BlockState.readMemTyped, h, byteCell]

private theorem read_bytes {BN T BD : Nat} (hBN : 0 < BN)
    (s : BlockState) (R : RegionName) (NH sz sh sr sc : Nat)
    (xs : Fin T → Fin (BN * BD) → Byte)
    (hx : ∀ t j, s.mem R (alBase s NH sz sh + (t.val * BN + j.val / BD) * sr +
      j.val % BD * sc) = byteCell (xs t j)) :
    FP8RealOutput.byteTile s R NH sz sh sr sc (BN * T) BD =
      RealIO.stream T BN BD hBN (fun t j => decode (xs t j)) := by
  funext i
  have hp := hx (RealIO.stepOf i.1) (RealIO.laneOf hBN i)
  rw [RealIO.stream_address hBN] at hp
  unfold FP8RealOutput.byteTile FP8RealMemory.value
  rw [pinned_byte s R _ _ hp]
  rfl

private theorem finite_bytes {BN T BD : Nat}
    (s : BlockState) (R : RegionName) (base sr sc N : Nat)
    (xs : Fin T → Fin (BN * BD) → Byte)
    (hx : ∀ t j, s.mem R (base + (t.val * BN + j.val / BD) * sr +
      j.val % BD * sc) = byteCell (xs t j)) :
    FP8RealLoop.BytesFrom R s base sr sc 0 (BN * T) N BN BD := by
  intro k hk j d _
  have hkT : k < T := by simp only [Nat.zero_add] at hk; nlinarith
  let lane : Fin (BN * BD) := Lane2D.encode (j, d, PUnit.unit)
  have hp := hx ⟨k, hkT⟩ lane
  simp only [lane, Lane2D.encode_div, Lane2D.encode_mod] at hp
  simp only [Nat.zero_add]
  rw [pinned_byte s R _ _ hp]
  exact (xs ⟨k, hkT⟩ lane).property

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool)

/-- Both raw-byte sources implement the independent input-array formula.
All active input cells, successful flat execution and the complete frame are
part of this contract; there is no prepared-decoded-memory premise. -/
theorem correct (hBM : 0 < BM) (hBN : 0 < BN) (hT : 0 < T)
    (hQK : Q ≠ K) (hQV : Q ≠ V) (hOK : Out ≠ K) (hOV : Out ≠ V)
    (hinj : Function.Injective (fun i : TileIndex [BM, BD] =>
      i.1.val * som + i.2.1.val * son)) :
    (io Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T candidate causal).ImplementsCells MemCell.real byteCell byteCell
        (fun p _ _ xs ys zs => expected T BM BN BD H SP hBN hT causal scale p xs ys zs) := by
  intro A hd hregions hcov p h z xs ys zs s hpre hp hh hz hu hQ hK hV hO hx hy hv
  subst p
  subst h
  subst z
  have hquery : alQTileG s Q NH sqz sqh sqm sqk H BM BD =
      RealIO.query T BM BD (s.pids 0) H hT xs := by
    funext i
    unfold alQTileG RealIO.query
    change (if s.pids 0 * BM + i.1.val < H then _ else 0) = _
    split
    next hi =>
      have hm := hx ⟨0, hT⟩ (Lane2D.encode i)
        (by simpa only [io, RealIO.io, Lane2D.encode_div] using hi)
      simp only [io, RealIO.io, Lane2D.encode_div, Lane2D.encode_mod] at hm
      simp only [BlockState.readMem, alBase, alRow]
      rw [hm]
      rfl
    next => rfl
  have hkeys := read_bytes (BN := BN) (T := T) (BD := BD) hBN s K NH skz skh skn skk ys
    (fun t j => hy t j True.intro)
  have hvalues := read_bytes (BN := BN) (T := T) (BD := BD) hBN s V NH svz svh svk svn zs
    (fun t j => hv t j True.intro)
  have hkbytes := finite_bytes (BN := BN) (T := T) (BD := BD) s K (alBase s NH skz skh) skn skk N ys
    (fun t j => hy t j True.intro)
  have hvbytes := finite_bytes (BN := BN) (T := T) (BD := BD) s V (alBase s NH svz svh) svk svn N zs
    (fun t j => hv t j True.intro)
  have hqb (i : TileIndex [BM, BD]) (hi : alActive s H BM i) :
      alBase s NH sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < A.extent Q := by
    simpa only [io, RealIO.io, alBase, alRow, Lane2D.encode_div, Lane2D.encode_mod] using
      hQ ⟨0, hT⟩ (Lane2D.encode i) (by simpa only [io, RealIO.io, Lane2D.encode_div] using hi)
  have hkb (r : Nat) (hr : r < BN * T) (d : Fin BD) :
      alBase s NH skz skh + r * skn + d.val * skk < A.extent K := by
    have hb := hK (RealIO.stepOf ⟨r, hr⟩) (RealIO.laneOf hBN (⟨r, hr⟩, d, PUnit.unit)) True.intro
    simp only [io, RealIO.io] at hb
    rw [RealIO.stream_address hBN (⟨r, hr⟩, d, PUnit.unit)] at hb
    exact hb
  have hvb (r : Nat) (hr : r < BN * T) (d : Fin BD) :
      alBase s NH svz svh + r * svk + d.val * svn < A.extent V := by
    have hb := hV (RealIO.stepOf ⟨r, hr⟩) (RealIO.laneOf hBN (⟨r, hr⟩, d, PUnit.unit)) True.intro
    simp only [io, RealIO.io] at hb
    rw [RealIO.stream_address hBN (⟨r, hr⟩, d, PUnit.unit)] at hb
    exact hb
  have hob (i : TileIndex [BM, BD]) (hi : alActive s H BM i) :
      alOutOffset s NH soz soh som son BM i < A.extent Out := by
    simpa only [io, RealIO.io, alOutOffset, alBase, alRow, Lane2D.encode_div, Lane2D.encode_mod] using
      hO (Lane2D.encode i) (by simpa only [io, RealIO.io, Lane2D.encode_div] using hi)
  have hundef : ∀ r o, s.undef r o = 0 := fun r o => by rw [hu]
  have hoi := RealIO.output_injective (s.pids 0) som son hinj
  cases causal
  case' «false» =>
    have hN : N = BN * T := hpre
    subst N
    obtain ⟨t, he, hread, hframe⟩ := FP8RealFlat.noncausal_correct Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H (BN * T) SP BM BN BD A hd hregions hcov candidate s T
      hQK hQV hOK hOV hBM hBN hT rfl hoi hundef hkbytes hvbytes hqb hkb hvb hob
  case' «true» =>
    obtain ⟨hspan, hspanle, hsp⟩ := hpre
    obtain ⟨t, he, hread, hframe⟩ := FP8RealFlat.causal_correct Q K V Out scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD A hd hregions hcov candidate s T
      hQK hQV hOK hOV hBM hBN hspan hspanle hsp hoi hundef hkbytes hvbytes hqb hkb hvb hob
  all_goals
    refine ⟨t, he, ?_, ?_⟩
    · intro j hj
      have hr := hread (Lane2D.decode j) hj
      rw [hquery, hkeys, hvalues] at hr
      exact hr
    · intro r o hc
      apply hframe r o
      rcases hc with hf | hf
      · exact Or.inl hf
      · refine Or.inr fun i hi => ?_
        simpa only [io, RealIO.io, alOutOffset, alBase, alRow, Lane2D.encode_div, Lane2D.encode_mod]
          using hf (Lane2D.encode i) (by simpa only [io, RealIO.io, Lane2D.encode_div] using hi)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealIO
