import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealIOBridge
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealInputFormula
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealOutputLayout
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealSource

/-! Full real IO composition for the complete original and reciprocal source.
The windows retain independent strides, grouped heads, checked Q tails,
continuation carry reads and unmasked M/Out stores. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealContract
open VeriTile.Triton RealPrologue
noncomputable section

def io (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool) : RealIO.IO BM BD p.layout.kvContext where
  kernel := RealSource.kernel BM BN BD Z finish p reciprocal
  projection := RealSource.projection BM BN BD Z finish p reciprocal
  Q := p.layout.Q
  K := p.layout.K
  V := p.layout.V
  M := p.layout.M
  L := p.layout.L
  Out := p.layout.Out
  queryWindow := fun p₀ p₁ _ i =>
    p₁ / p.heads * p.strides.qBatch + p₁ % p.heads * p.strides.qHead +
      (p₀ * BM + i.1.val) * p.layout.qRow + i.2.1.val * p.layout.qDim
  keyWindow := fun _ p₁ _ i =>
    p₁ / p.heads * p.strides.kBatch + ((p₁ % p.heads) / (p.heads / p.kvHeads)) * p.strides.kHead +
      i.2.1.val * p.layout.kDim + i.1.val * p.layout.kRow
  valueWindow := fun _ p₁ _ i =>
    p₁ / p.heads * p.strides.vBatch + ((p₁ % p.heads) / (p.heads / p.kvHeads)) * p.strides.vHead +
      i.1.val * p.layout.vRow + i.2.1.val * p.layout.vDim
  centerWindow := fun p₀ p₁ _ i => p₁ * p.layout.roundedContext + p₀ * BM + i.1.val
  denominatorWindow := fun p₀ p₁ _ i => p₁ * p.layout.roundedContext + p₀ * BM + i.1.val
  outputWindow := fun p₀ p₁ _ i =>
    p₁ / p.heads * p.strides.oBatch + p₁ % p.heads * p.strides.oHead +
      (p₀ * BM + i.1.val) * p.layout.oRow + i.2.1.val * p.layout.oDim
  queryMask := fun p₀ _ _ i => p.evenM ≠ 0 ∨ p₀ * BM + i.1.val < p.layout.queryContext
  readCarry := p.init = 0
  writeDenominator := finish = 0

def expected (BM BN BD finish start : Nat) (p : Parameters)
    (xs : RealInputs.Inputs BM BD p.layout.kvContext) : RealIO.Outputs BM BD where
  center := RealInputFormula.center BM BN BD finish start p xs
  denominator := (RealInputFormula.carry BM BN BD start p xs).denominator
  output := RealInputFormula.output BM BN BD finish start p xs

theorem input_pins (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool)
    (s : BlockState) (xs : RealInputs.Inputs BM BD p.layout.kvContext)
    (h : RealIO.Pins (io BM BN BD Z finish p reciprocal) s xs) : RealInputs.Pins BM BD p s xs := by
  refine ⟨h.query, h.key, h.value, h.center, h.denominator, ?_⟩
  intro hi r d
  simpa only [io, RealAddressBounds.output_address, RealHeader.bases, RealHeader.coordinates] using h.output hi r d

theorem address_bounds (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool)
    (s : BlockState) (bounds : RegionBounds)
    (h : RealIO.Bounds (io BM BN BD Z finish p reciprocal) s bounds) :
    RealAddressBounds.Windows BM BD finish p s bounds := by
  refine ⟨fun r d ha => h.query (r, d, PUnit.unit) ha,
    fun k d => h.key (k, d, PUnit.unit), fun k d => h.value (k, d, PUnit.unit),
    h.center, h.denominator, ?_⟩
  intro i
  simpa only [io, RealAddressBounds.output_address, RealHeader.bases, RealHeader.coordinates] using h.output i

theorem readback (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool)
    (s t : BlockState) (xs : RealInputs.Inputs BM BD p.layout.kvContext)
    (hp : RealInputs.Pins BM BD p s xs) (h : RealExecution.Results BM BN BD finish p s t) :
    RealIO.Results (io BM BN BD Z finish p reciprocal) s t (expected BM BN BD finish (s.pids 0) p xs) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro i
    exact (h.center i).trans (RealInputFormula.center_pinned BM BN BD finish p s xs hp i.1)
  · intro he i
    have hd := h.denominator he i
    rw [RealInputFormula.carry_pinned BM BN BD p s xs hp] at hd
    exact hd
  · intro i
    have ho := (h.output i).trans (RealInputFormula.output_pinned BM BN BD finish p s xs hp i)
    simpa only [io, expected, RealAddressBounds.output_address, RealHeader.bases, RealHeader.coordinates] using ho
  · intro r a hm hl ho
    apply h.frame r a hm hl
    simpa only [io, RealAddressBounds.output_address, RealHeader.bases, RealHeader.coordinates] using ho

/-- Both complete sources implement the same independent six-input formula.
Bounds and input pinning live in the IO contract. The remaining static premises
are positive BLOCK_N, valid unchecked KV blocks and nonoverlapping output views. -/
theorem implements (BM BN BD Z finish : Nat) (p : Parameters) (reciprocal : Bool)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (hd : 0 < p.layout.oDim) (hr : BD * p.layout.oDim ≤ p.layout.oRow)
    (hm : p.layout.M ≠ p.layout.Out) (hlm : p.layout.L ≠ p.layout.M) (hlo : p.layout.L ≠ p.layout.Out) :
    (io BM BN BD Z finish p reciprocal).Implements (fun start _ xs => expected BM BN BD finish start p xs) := by
  apply RealIO.IO.Implements.intro
  · exact RealSource.flatten_ok BM BN BD Z finish p reciprocal
  · intro bounds s _ _ hb
    exact RealSource.safe BM BN BD Z finish p reciprocal s bounds hN hflag
      (RealAddressBounds.trace_bounds BM BN BD finish p s bounds hN hflag
        (address_bounds BM BN BD Z finish p reciprocal s bounds hb))
  · intro s xs _ hp
    obtain ⟨t, he, hy⟩ := RealSource.executes BM BN BD Z finish p reciprocal s hN hflag
      (RealOutputLayout.output_layout BM BD p s hd hr hm hlm hlo)
    exact ⟨t, he, readback BM BN BD Z finish p reciprocal s t xs
      (input_pins BM BN BD Z finish p reciprocal s xs hp) hy⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealContract
