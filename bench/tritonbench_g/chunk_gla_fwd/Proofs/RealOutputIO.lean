import bench.tritonbench_g.chunk_gla_fwd.Proofs.IO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputFlat

/-! Explicit output-consumer IO. The formula reads Q/V/G/H/A from initial
memory, independently of the implementation and its accumulators. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO
open VeriTile.Triton RealIO
open scoped BigOperators RealIO
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

structure Layout where
  keyHead : Nat
  keyTime : Nat
  valueHead : Nat
  valueTime : Nat
  historyHead : Nat
  historyKey : Nat
  T : Nat
  K : Nat
  V : Nat
  BT : Nat
  BK : Nat
  BV : Nat
structure Regions where
  query : RegionName
  value : RegionName
  gate : RegionName
  history : RegionName
  output : RegionName
  attention : RegionName
structure Launch where
  value : Nat
  chunk : Nat
  head : Nat
structure Inputs where
  query : Nat → ℝ
  value : Nat → ℝ
  gate : Nat → ℝ
  history : Nat → ℝ
  attention : Nat → ℝ

def launch (s : BlockState) : Launch := ⟨s.pids 0,s.pids 1,s.pids 2⟩
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BT,p.BV]) :=
  l.head*p.valueHead+(l.chunk*p.BT+i.1.val)*p.valueTime+(l.value*p.BV+i.2.1.val)
def active (p : Layout) (l : Launch) (i : TileIndex [p.BT,p.BV]) : Prop :=
  l.chunk*p.BT+i.1.val < p.T ∧ l.value*p.BV+i.2.1.val < p.V

def source (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) (swapped : Bool) :=
  TypedOutput.source qs vs gs swapped history out r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV

theorem projection (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) (swapped : Bool) :
    (source p r qs vs gs history out scale swapped).eraseDType.toAlgorithm? =
      .ok (source p r qs vs gs history out scale swapped).eraseDType.toAlgKernel := by
  simp only [source,ComputeKernel.eraseDType,RealOutputSource.projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) (swapped : Bool) : IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (source p r qs vs gs history out scale swapped).eraseDType
  projection := projection p r qs vs gs history out scale swapped
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.value,s.readMem r.gate,s.readMem r.history,s.readMem r.attention⟩
  valid := fun s => Function.Injective (outputAddress p (launch s))
  bounds := fun s => RealOutputSafety.Bounds s r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey p.T p.K p.V p.BT p.BK p.BV
  out := r.output
  write := outputAddress p
  mask := active p

/-- Full K-block gated history contraction plus causal A·V, with zero extension
outside the source block pointers' checked parents. -/
def formula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
  let t := l.chunk*p.BT+i.1.val
  let v := l.value*p.BV+i.2.1.val
  (∑ kb : Fin ((p.K+p.BK-1)/p.BK), ∑ e : Fin p.BK,
    let k := kb.val*p.BK+e.val
    (if t < p.T ∧ k < p.K then
      a.query (l.head*p.keyHead+t*p.keyTime+k) * scale * Real.exp (a.gate (l.head*p.keyHead+t*p.keyTime+k)) else 0) *
    (if k < p.K ∧ v < p.V then
      a.history (l.head*p.historyHead+l.chunk*p.K*p.V+k*p.historyKey+v) else 0)) +
  ∑ j : Fin p.BT,
    (if j.val ≤ i.1.val then (if t < p.T then a.attention (l.head*p.T*p.BT+t*p.BT+j.val) else 0) else 0) *
    (if l.chunk*p.BT+j.val < p.T ∧ v < p.V then
      a.value (l.head*p.valueHead+(l.chunk*p.BT+j.val)*p.valueTime+v) else 0)

theorem implements (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) (swapped : Bool) :
    sourceIO p r qs vs gs history out scale swapped ⊨ formula p scale := by
  intro alloc hd hcov s hu hi hb
  have hinj : Function.Injective (RealOutput.outOffset s p.valueHead p.valueTime p.BT p.BV) := by
    intro i j hij
    apply hi
    simpa only [outputAddress,launch,RealOutput.outOffset,RealOutput.tIndex,
      RealOutput.vIndex,Nat.mul_one] using hij
  obtain ⟨sf,he,ho,hf⟩ := RealOutputFlat.source_executes qs vs gs history out r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV
    alloc hd hcov swapped s hu hinj hb
  refine ⟨sf,he,?_,?_⟩
  · intro i ha
    simpa only [sourceIO,launch,formula,outputAddress,RealOutput.outOffset,RealOutput.tIndex,RealOutput.vIndex,
      RealOutput.cgfOutput,RealOutput.numKB,RealOutput.qgElem,RealOutput.hGuarded,RealOutput.vGuarded,RealOutput.aMasked,
      RealOutput.qElem,RealOutput.gElem,RealOutput.hElem,RealOutput.vElem,RealOutput.aElem,RealOutput.kIndex,Nat.mul_one]
      using ho i ha
  · intro region off hn
    apply hf region off
    simpa only [IO.writes,sourceIO,launch,outputAddress,active,RealOutputFlat.footprint,
      RealOutput.outOffset,RealOutput.tIndex,RealOutput.vIndex,Nat.mul_one] using hn

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealOutputIO
