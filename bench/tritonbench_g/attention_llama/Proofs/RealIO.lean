import bench.tritonbench_g.attention_llama.Proofs.RealMemory
import VeriTile.Triton.Memory.KernelSpec.StreamAttentionReal

/-! Independent input-array IO contract for the complete fp16 source paths.
The launch condition describes the actual streamed span. Access masks retain
their source meaning, and output values are defined without running a kernel. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.RealIO
open VeriTile.Triton RealKernel
open scoped VeriTile.Triton.StreamMasked3DKernelIO₃
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000

/-- Whole key blocks for the noncausal program; the source's shifted span and
nonempty visible rows for the causal program. This is local to the program ID. -/
def launch (causal : Bool) (N SP BM BN T p : Nat) : Prop :=
  if causal then (p + 1) * BN + SP = BN * T ∧ BN * T ≤ N ∧ SP ≤ p * BM
  else N = BN * T

/-- Q is a static stream; K and V are loaded as [BN, BD] tiles, before K's
transpose. The output uses its own row and column strides. -/
def io (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool) : StreamMasked3DKernelIO₃ where
  kernel := (RealOutput.source Q K V Out scale
    sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
    NH H N SP BM BN BD candidate causal).eraseDType
  projection := by
    cases candidate
    · exact RealSource.original_projection_supported ..
    · exact RealSource.candidate_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  out := Out
  T := T
  B1 := BM * BD
  B2 := BN * BD
  B3 := BN * BD
  C := BM * BD
  outDType := .real
  pre := fun p _ _ => launch causal N SP BM BN T p
  read1 := fun p h _ _ j => h / NH * sqz + h % NH * sqh +
    (p * BM + j.val / BD) * sqm + j.val % BD * sqk
  read2 := fun _ h _ t j => h / NH * skz + h % NH * skh +
    (t.val * BN + j.val / BD) * skn + j.val % BD * skk
  read3 := fun _ h _ t j => h / NH * svz + h % NH * svh +
    (t.val * BN + j.val / BD) * svk + j.val % BD * svn
  write := fun p h _ j => h / NH * soz + h % NH * soh +
    (p * BM + j.val / BD) * som + j.val % BD * son
  mask1 := fun p _ _ _ j => p * BM + j.val / BD < H
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask := fun p _ _ j => p * BM + j.val / BD < H

/-- Read the static query slice from the first input array, padding inactive rows. -/
noncomputable def query (T BM BD p H : Nat) (hT : 0 < T)
    (xs : Fin T → Fin (BM * BD) → ℝ) : TileIndex [BM, BD] → ℝ :=
  fun i => if p * BM + i.1.val < H then xs ⟨0, hT⟩ (Lane2D.encode i) else 0

def stepOf {BN T : Nat} (r : Fin (BN * T)) : Fin T :=
  ⟨r.val / BN, Nat.div_lt_of_lt_mul (by simpa only [Nat.mul_comm] using r.isLt)⟩

def laneOf {BN T BD : Nat} (hBN : 0 < BN) (i : TileIndex [BN * T, BD]) :
    Fin (BN * BD) :=
  Lane2D.encode (⟨i.1.val % BN, Nat.mod_lt _ hBN⟩, i.2)

/-- Reassemble a key or value row from the independent per-step input array. -/
noncomputable def stream (T BN BD : Nat) (hBN : 0 < BN)
    (xs : Fin T → Fin (BN * BD) → ℝ) : TileIndex [BN * T, BD] → ℝ :=
  fun i => xs (stepOf i.1) (laneOf hBN i)

/-- Natural-exp attention over the input arrays, with the source's shifted
causal visibility when requested. No recurrence or execution result is used. -/
noncomputable def expected (T BM BN BD H SP : Nat) (hBN : 0 < BN) (hT : 0 < T)
    (causal : Bool) (scale : ℝ) (p : Nat)
    (xs : Fin T → Fin (BM * BD) → ℝ)
    (ys zs : Fin T → Fin (BN * BD) → ℝ) (j : Fin (BM * BD)) : ℝ :=
  if causal then
    attentionRealCausalBlock (p * BM - SP)
      (query T BM BD p H hT xs) (stream T BN BD hBN ys) (stream T BN BD hBN zs)
      scale (Lane2D.decode j)
  else
    attentionReal (query T BM BD p H hT xs)
      (stream T BN BD hBN ys) (stream T BN BD hBN zs) scale (Lane2D.decode j)

theorem stream_address {BN T BD : Nat} (hBN : 0 < BN)
    (i : TileIndex [BN * T, BD]) (base rowStride colStride : Nat) :
    base + ((stepOf i.1).val * BN + (laneOf hBN i).val / BD) * rowStride +
        (laneOf hBN i).val % BD * colStride =
      base + i.1.val * rowStride + i.2.1.val * colStride := by
  simp only [laneOf, stepOf, Lane2D.encode_div, Lane2D.encode_mod]
  rw [Nat.mul_comm (i.1.val / BN) BN, Nat.div_add_mod]

private theorem read_stream {BN T BD : Nat} (hBN : 0 < BN)
    (s : BlockState) (r : RegionName) (base rowStride colStride : Nat)
    (xs : Fin T → Fin (BN * BD) → ℝ)
    (hx : ∀ t j, s.readMem r (base + (t.val * BN + j.val / BD) * rowStride +
      j.val % BD * colStride) = xs t j) :
    (fun i : TileIndex [BN * T, BD] =>
      s.readMem r (base + i.1.val * rowStride + i.2.1.val * colStride)) =
      stream T BN BD hBN xs := by
  funext i
  have h := hx (stepOf i.1) (laneOf hBN i)
  rw [stream_address] at h
  exact h

theorem output_injective {BM BD : Nat} (p som son : Nat)
    (h : Function.Injective (fun i : TileIndex [BM, BD] =>
      i.1.val * som + i.2.1.val * son)) :
    Function.Injective (fun i : TileIndex [BM, BD] =>
      (p * BM + i.1.val) * som + i.2.1.val * son) := by
  intro i j hij
  apply h
  simp only [Nat.add_mul, Nat.add_assoc] at hij
  exact Nat.add_left_cancel hij

section
variable (Q K V Out : RegionName) (scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T : Nat) (candidate causal : Bool)

/-- Both source variants implement the independent array formula, including
successful flat execution and the complete memory frame. -/
theorem correct (hBM : 0 < BM) (hBN : 0 < BN) (hT : 0 < T)
    (hinj : Function.Injective (fun i : TileIndex [BM, BD] =>
      i.1.val * som + i.2.1.val * son)) :
    io Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD T candidate causal ⊨
        fun p _ _ xs ys zs => expected T BM BN BD H SP hBN hT causal scale p xs ys zs := by
  intro A hd hregions hcov p h z xs ys zs s hpre hp hh hz hu hQ hK hV hO hx hy hv
  subst p
  subst h
  subst z
  have hquery : alQTileG s Q NH sqz sqh sqm sqk H BM BD =
      query T BM BD (s.pids 0) H hT xs := by
    funext i
    unfold alQTileG query
    change (if s.pids 0 * BM + i.1.val < H then _ else 0) = _
    split
    next hi =>
      simpa only [io, alBase, alRow, Lane2D.encode_div, Lane2D.encode_mod] using
        hx ⟨0, hT⟩ (Lane2D.encode i) (by simpa only [io, Lane2D.encode_div] using hi)
    next => rfl
  have hkeys : alKTileG s K NH skz skh skn skk (BN * T) BD = stream T BN BD hBN ys :=
    read_stream hBN s K (alBase s NH skz skh) skn skk ys
      (fun t j => hy t j True.intro)
  have hvalues : alVTileG s V NH svz svh svk svn (BN * T) BD = stream T BN BD hBN zs :=
    read_stream hBN s V (alBase s NH svz svh) svk svn zs
      (fun t j => hv t j True.intro)
  have hqb (i : TileIndex [BM, BD]) (hi : alActive s H BM i) :
      alBase s NH sqz sqh + alRow s BM i.1 * sqm + i.2.1.val * sqk < A.extent Q := by
    simpa only [io, alBase, alRow, Lane2D.encode_div, Lane2D.encode_mod] using
      hQ ⟨0, hT⟩ (Lane2D.encode i) (by simpa only [io, Lane2D.encode_div] using hi)
  have hkb (r : Nat) (hr : r < BN * T) (d : Fin BD) :
      alBase s NH skz skh + r * skn + d.val * skk < A.extent K := by
    have hb := hK (stepOf ⟨r, hr⟩) (laneOf hBN (⟨r, hr⟩, d, PUnit.unit)) True.intro
    simp only [io] at hb
    rw [stream_address hBN (⟨r, hr⟩, d, PUnit.unit)] at hb
    exact hb
  have hvb (r : Nat) (hr : r < BN * T) (d : Fin BD) :
      alBase s NH svz svh + r * svk + d.val * svn < A.extent V := by
    have hb := hV (stepOf ⟨r, hr⟩) (laneOf hBN (⟨r, hr⟩, d, PUnit.unit)) True.intro
    simp only [io] at hb
    rw [stream_address hBN (⟨r, hr⟩, d, PUnit.unit)] at hb
    exact hb
  have hob (i : TileIndex [BM, BD]) (hi : alActive s H BM i) :
      alOutOffset s NH soz soh som son BM i < A.extent Out := by
    simpa only [io, alOutOffset, alBase, alRow, Lane2D.encode_div, Lane2D.encode_mod] using
      hO (Lane2D.encode i) (by simpa only [io, Lane2D.encode_div] using hi)
  have hundef : ∀ r o, s.undef r o = 0 := fun r o => by rw [hu]
  have hoi := output_injective (s.pids 0) som son hinj
  cases causal
  case' «false» =>
    have hN : N = BN * T := hpre
    subst N
    obtain ⟨sF, he, hread, hframe⟩ := RealMemory.noncausal_correct
      Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H (BN * T) SP BM BN BD A hd hregions hcov candidate s T
      hBM hBN hT rfl hoi hundef hqb hkb hvb hob
  case' «true» =>
    obtain ⟨hspan, hspanle, hsp⟩ := hpre
    obtain ⟨sF, he, hread, hframe⟩ := RealMemory.causal_correct
      Q K V Out scale sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD A hd hregions hcov candidate s T
      hBM hBN hspan hspanle hsp hoi hundef hqb hkb hvb hob
  all_goals
    refine ⟨sF, he, ?_, ?_⟩
    · intro j hj
      have hr := hread (Lane2D.decode j) hj
      rw [hquery, hkeys, hvalues] at hr
      exact hr
    · intro r o hc
      apply hframe r o
      rcases hc with hf | hf
      · exact Or.inl hf
      · refine Or.inr fun i hi => ?_
        simpa only [io, alOutOffset, alBase, alRow, Lane2D.encode_div, Lane2D.encode_mod]
          using hf (Lane2D.encode i) (by simpa only [io, Lane2D.encode_div] using hi)
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.RealIO
