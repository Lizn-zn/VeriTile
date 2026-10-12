import bench.tritonbench_g.chunk_gated_attention.Kernels
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealCumIO
import bench.tritonbench_g.chunk_gated_attention.Proofs.RealHIO

/-! Real correctness of the two complete sources and their candidates in Kernels.
Precision is erased only here. Both contracts conclude successful execution in
disjoint finite allocations, independent mathematical outputs and preservation
of every other tagged cell. Dimensions, strides and program IDs remain symbolic.
Host grid composition and fixed-width address overflow are outside this model.

The cumulative kernel permits input/output aliasing; its valid region outputs
are independent of unspecified padding. The shared flat IO adapter uses a zero
oracle. The recurrent kernel needs that zero-oracle premise for its formula;
Proofs/UndefinedTail gives a complete-source counterexample without it. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.Correctness
open VeriTile.Triton
open scoped BigOperators
noncomputable section

section Cumulative
open scoped VeriTile.Triton.Masked3DTileKernelIO₁
abbrev CumLayout := RealCumProgram.Layout

/-- Feature-block, chunk and head are program axes 0/1/2. Checked-off time and
feature lanes are excluded from the BT×BS output window. -/
def cumulativeIO (p : CumLayout) (s o : RegionName) (dtype out_dtype : FloatDType) : Masked3DTileKernelIO₁ where
  kernel := (TypedSource.cumOriginal dtype out_dtype s o p.headStride p.timeStride p.channelStride
    p.T p.S p.BT p.BS).eraseDType
  projection := (RealCumIO.io Bool.false p s o dtype out_dtype).projection
  inp := s
  out := o
  shape := [p.BT,p.BS]
  read := fun feature c head i =>
    head*p.headStride+(c*p.BT+i.1.val)*p.timeStride+(feature*p.BS+i.2.1.val)*p.channelStride
  write := fun feature c head i =>
    head*p.headStride+(c*p.BT+i.1.val)*p.timeStride+(feature*p.BS+i.2.1.val)*p.channelStride
  mask := fun feature c _ i => c*p.BT+i.1.val < p.T ∧ feature*p.BS+i.2.1.val < p.S

def cumulativeIndexFirstIO (p : CumLayout) (s o : RegionName) (dtype out_dtype : FloatDType) : Masked3DTileKernelIO₁ :=
  { cumulativeIO p s o dtype out_dtype with
    kernel := (TypedSource.cumIndexFirst dtype out_dtype s o p.headStride p.timeStride p.channelStride
      p.T p.S p.BT p.BS).eraseDType
    projection := (RealCumIO.io Bool.true p s o dtype out_dtype).projection }

/-- Prefix sum of initial inputs within one chunk, independently per feature. -/
def cumulativeExpected (p : CumLayout) (_feature _chunk : Nat) (xs : TileIndex [p.BT,p.BS] → ℝ)
    (i : TileIndex [p.BT,p.BS]) : ℝ :=
  ∑ k : Fin p.BT, if k.val ≤ i.1.val then xs (k,i.2.1,PUnit.unit) else 0

/-- Only active output cells of the same program must have distinct addresses.
The IO relation also requires every active input/output address to be allocated. -/
def cumulativeNoOverlap (p : CumLayout) : Prop :=
  ∀ feature c head, RealCumExecution.NoCollision p ⟨feature,head⟩ c

specification cumulative_correctness (p : CumLayout) (s o : RegionName) (dtype out_dtype : FloatDType)
    (hinj : cumulativeNoOverlap p) :
    Spec.Real (cumulativeIO p s o dtype out_dtype ⊨ cumulativeExpected p) := by
  exact RealCumIO.implements Bool.false p s o dtype out_dtype hinj

specification cumulative_index_first_correctness (p : CumLayout) (s o : RegionName) (dtype out_dtype : FloatDType)
    (hinj : cumulativeNoOverlap p) :
    Spec.Real (cumulativeIndexFirstIO p s o dtype out_dtype ⊨ cumulativeExpected p) := by
  exact RealCumIO.implements Bool.true p s o dtype out_dtype hinj
end Cumulative

section Recurrent
open RealHMath RealHProgram
open scoped VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealHIO
abbrev HLayout := RealHMath.Layout

/-- History observes H before each update; optional final output observes H[NT].
K/V/gate/seed arrays are read from initial memory. Access bounds cover the actual
selected gate tile and flattened full-chunk endpoint, plus enabled state loads
and stores. `valid` prevents active history and final-address collisions.
IO.Implements explicitly requires zero unspecified real values. -/
def recurrentIO (p : HLayout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal : Bool) : RealHIO.IO p where
  kernel := (TypedSource.hOriginal dot gate history seed final r.key r.value r.gate r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
    p.historyHead p.historyKey p.historyValue p.T p.K p.V p.BT p.BK p.BV p.NT gateK hasSeed storeFinal).eraseDType
  projection := RealHIO.projection p r dot gate history seed final gateK hasSeed storeFinal Bool.false
  read := fun s => ⟨s.readMem r.key,s.readMem r.value,s.readMem r.gate,s.readMem r.seed⟩
  valid := fun s => RealHLoop.HistoryNoCollision p (launch s) ∧
    (storeFinal = Bool.true → RealHStore.NoCollision p (launch s) (RealHFinal.base p (launch s)) p.V 1)
  bounds := fun s => RealHSafety.Bounds p (launch s) r gateK hasSeed storeFinal
  history := r.history
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j =>
    l.head*p.historyHead+c*p.K*p.V+(l.key*p.BK+e.val)*p.historyKey+(l.value*p.BV+j.val)*p.historyValue
  finalWindow := fun l e j => l.head*p.K*p.V+(l.key*p.BK+e.val)*p.V+(l.value*p.BV+j.val)
  stateMask := fun l e j => l.key*p.BK+e.val < p.K ∧ l.value*p.BV+j.val < p.V

def recurrentSumLeftIO (p : HLayout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal : Bool) : RealHIO.IO p :=
  { recurrentIO p r dot gate history seed final gateK hasSeed storeFinal with
    kernel := (TypedSource.hSumLeft dot gate history seed final r.key r.value r.gate r.history r.seed r.final
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature
      p.historyHead p.historyKey p.historyValue p.T p.K p.V p.BT p.BK p.BV p.NT gateK hasSeed storeFinal).eraseDType
    projection := RealHIO.projection p r dot gate history seed final gateK hasSeed storeFinal Bool.true }

/-- G[c]=exp(full-chunk endpoint), and P[c] is the BT-term gated K/V contraction.
`gateFactor` and `contribution` independently index the initial arrays, with
zero extension outside each source pointer's checked parent. GATEK selects
whether K or V carries the exp(endpoint-gate) weight. The closed form is
H[c]=seed*product(G[0:c])+sum(P[t]*product(G[t+1:c])). -/
def recurrentExpected (p : HLayout) (gateK hasSeed : Bool) (l : Launch) (a : Inputs) : RealHIO.Outputs p :=
  let state := fun c k v =>
    RealHMath.seed p l a hasSeed k v * (∏ t ∈ Finset.range c, gateFactor p l a gateK t k v) +
      ∑ t ∈ Finset.range c, contribution p l a gateK t k v *
        ∏ u ∈ Finset.Ico (t+1) c, gateFactor p l a gateK u k v
  { history := fun c e j => state c (keyIndex p l e.val) (valueIndex p l j.val)
    final := fun e j => state p.NT (keyIndex p l e.val) (valueIndex p l j.val) }

/-- The history store precedes K/V/gate loads, so those inputs cannot alias it.
The final store preserves history. Seed may alias outputs: it is read first. -/
specification recurrent_correctness (p : HLayout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal : Bool)
    (hs : RealHStep.Separate r) (hfsep : storeFinal = Bool.true → r.history ≠ r.final) :
    Spec.Real (recurrentIO p r dot gate history seed final gateK hasSeed storeFinal ⊨ recurrentExpected p gateK hasSeed) := by
  exact RealHIO.implements p r dot gate history seed final gateK hasSeed storeFinal Bool.false hs hfsep

specification recurrent_sum_left_correctness (p : HLayout) (r : Regions) (dot gate history : TypedSource.Storage)
    (seed final : FloatDType) (gateK hasSeed storeFinal : Bool)
    (hs : RealHStep.Separate r) (hfsep : storeFinal = Bool.true → r.history ≠ r.final) :
    Spec.Real (recurrentSumLeftIO p r dot gate history seed final gateK hasSeed storeFinal ⊨ recurrentExpected p gateK hasSeed) := by
  exact RealHIO.implements p r dot gate history seed final gateK hasSeed storeFinal Bool.true hs hfsep
end Recurrent

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.Correctness
