import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealSafetySource

/-! Ordinary strided input/output windows imply the source trace bounds.
The interface names Q rows, global K/V keys and M/L/Out output lanes; it has no
loop-state or successful-execution premise. Output bounds cover the full block,
since the actual stores do not have boundary masks. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAddressBounds
open VeriTile.Triton RealAccess RealPrologue
noncomputable section

def queryAddress (BM : Nat) (p : Parameters) (s : BlockState) (row lane : Nat) : Nat :=
  let c := RealHeader.coordinates s p.heads p.kvHeads
  let b := RealHeader.bases c p.strides
  b.q + (c.startM * BM + row) * p.layout.qRow + lane * p.layout.qDim

def keyAddress (p : Parameters) (s : BlockState) (key lane : Nat) : Nat :=
  let b := RealHeader.bases (RealHeader.coordinates s p.heads p.kvHeads) p.strides
  b.k + lane * p.layout.kDim + key * p.layout.kRow

def valueAddress (p : Parameters) (s : BlockState) (key lane : Nat) : Nat :=
  let b := RealHeader.bases (RealHeader.coordinates s p.heads p.kvHeads) p.strides
  b.v + key * p.layout.vRow + lane * p.layout.vDim

structure Windows (BM BD finish : Nat) (p : Parameters) (s : BlockState) (bounds : RegionBounds) : Prop where
  query : ∀ (row : Fin BM) (lane : Fin BD),
    (p.evenM ≠ 0 ∨ s.pids 0 * BM + row.val < p.layout.queryContext) →
    queryAddress BM p s row.val lane.val < bounds p.layout.Q
  keys : ∀ (key : Fin p.layout.kvContext) (lane : Fin BD), keyAddress p s key.val lane.val < bounds p.layout.K
  values : ∀ (key : Fin p.layout.kvContext) (lane : Fin BD), valueAddress p s key.val lane.val < bounds p.layout.V
  center : ∀ i, RealExecution.rowAddress BM p s i < bounds p.layout.M
  denominator : p.init = 0 ∨ finish = 0 → ∀ i, RealExecution.rowAddress BM p s i < bounds p.layout.L
  output : ∀ i, RealExecution.outputAddress BM BD p s i < bounds p.layout.Out

theorem output_address (BM BD : Nat) (p : Parameters) (s : BlockState) (i : TileIndex [BM, BD]) :
    RealExecution.outputAddress BM BD p s i =
      (RealHeader.bases (RealHeader.coordinates s p.heads p.kvHeads) p.strides).out +
        (s.pids 0 * BM + i.1.val) * p.layout.oRow + i.2.1.val * p.layout.oDim := by
  simp [RealExecution.outputAddress, RealEpilogue.blockAddress, RealExecution.outputPointer, RealHeader.coordinates]

theorem query_bounds (BM BD finish : Nat) (p : Parameters) (s : BlockState) (bounds : RegionBounds)
    (h : Windows BM BD finish p s bounds) :
    BlockBounds bounds
      (RealPointers.queryPointer BM BD p.layout (RealHeader.coordinates s p.heads p.kvHeads)
        (RealHeader.bases (RealHeader.coordinates s p.heads p.kvHeads) p.strides))
      (RealAccumulation.checks p.evenM) := by
  intro i hi
  have hactive : p.evenM ≠ 0 ∨ s.pids 0 * BM + i.1.val < p.layout.queryContext := by
    by_cases he : p.evenM ≠ 0
    · exact Or.inl he
    · right
      have hb : s.pids 0 * BM + i.1.val < p.layout.queryContext ∧ i.2.1.val < BD := by
        simpa [RealPointers.queryPointer, RealHeader.coordinates, RealAccumulation.checks, he] using hi
      exact hb.1
  simpa [RealPointers.queryPointer, queryAddress] using h.query i.1 i.2.1 hactive

theorem key_bounds (BM BN BD finish : Nat) (p : Parameters) (s : BlockState) (bounds : RegionBounds)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (h : Windows BM BD finish p s bounds) (j : Nat) (hj : j < RealLoop.count (loopParameters BM BN BD p s)) :
    BlockBounds bounds (RealLoop.keyPointer (loopParameters BM BN BD p s) j) (RealAccumulation.checks p.evenN) := by
  let lp := loopParameters BM BN BD p s
  have hf := RealLoopContext.full_blocks_of_source lp hN hflag rfl (RealWindow.upper_matches _ _ _ _ _ _ _)
  intro i hi
  have hkey : RealLoop.cursor lp j + i.2.1.val < p.layout.kvContext := by
    by_cases he : p.evenN ≠ 0
    · have hs := hf he j hj
      have hn := i.2.1.isLt
      change RealLoop.cursor lp j + BN ≤ p.layout.kvContext at hs
      omega
    · have hb : i.1.val < BD ∧ RealLoop.cursor lp j + i.2.1.val < p.layout.kvContext := by
        simpa [RealLoop.keyPointer, RealAccumulation.checks, he] using hi
      exact hb.2
  simpa [RealLoop.keyPointer, lp, loopParameters, keyAddress] using
    h.keys ⟨RealLoop.cursor lp j + i.2.1.val, hkey⟩ i.1

theorem value_bounds (BM BN BD finish : Nat) (p : Parameters) (s : BlockState) (bounds : RegionBounds)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (h : Windows BM BD finish p s bounds) (j : Nat) (hj : j < RealLoop.count (loopParameters BM BN BD p s)) :
    BlockBounds bounds (RealLoop.valuePointer (loopParameters BM BN BD p s) j) (RealAccumulation.checks p.evenN) := by
  let lp := loopParameters BM BN BD p s
  have hf := RealLoopContext.full_blocks_of_source lp hN hflag rfl (RealWindow.upper_matches _ _ _ _ _ _ _)
  intro i hi
  have hkey : RealLoop.cursor lp j + i.1.val < p.layout.kvContext := by
    by_cases he : p.evenN ≠ 0
    · have hs := hf he j hj
      have hn := i.1.isLt
      change RealLoop.cursor lp j + BN ≤ p.layout.kvContext at hs
      omega
    · have hb : RealLoop.cursor lp j + i.1.val < p.layout.kvContext ∧ i.2.1.val < BD := by
        simpa [RealLoop.valuePointer, RealAccumulation.checks, he] using hi
      exact hb.1
  simpa [RealLoop.valuePointer, lp, loopParameters, valueAddress] using
    h.values ⟨RealLoop.cursor lp j + i.1.val, hkey⟩ i.2.1

theorem trace_bounds (BM BN BD finish : Nat) (p : Parameters) (s : BlockState) (bounds : RegionBounds)
    (hN : 0 < BN) (hflag : p.evenN ≠ 0 → p.layout.kvContext % BN = 0)
    (h : Windows BM BD finish p s bounds) : RealSafety.Bounds BM BN BD finish p s bounds := by
  refine ⟨⟨query_bounds BM BD finish p s bounds h, fun _ => h.center,
    fun hi => h.denominator (Or.inl hi), fun _ => h.output⟩,
    key_bounds BM BN BD finish p s bounds hN hflag h,
    value_bounds BM BN BD finish p s bounds hN hflag h,
    h.center, fun he => h.denominator (Or.inr he), h.output⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAddressBounds
