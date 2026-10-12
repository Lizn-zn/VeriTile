import bench.tritonbench_g.chunk_linear_attn.Proofs.ForwardStateFlat
import bench.tritonbench_g.chunk_linear_attn.Proofs.ForwardOutputFlat
import bench.tritonbench_g.chunk_linear_attn.Proofs.BackwardStateFlat
import bench.tritonbench_g.chunk_linear_attn.Proofs.BackwardGradientsFlat

/-! Real correctness of all four complete sources in Kernels. These contracts
include successful execution in finite flat allocations and preservation of all
other tagged cells. Precision is erased only here. Every dimension and stride
remains symbolic; bounds and injectivity apply to actual active accesses.

The source leaves block padding unspecified. Each IO contract explicitly uses
zero unspecified values for the zero-extended mathematical formula. Host launch,
allocation construction, backend tile restrictions and inter-program races are
outside these per-program statements. -/
namespace VeriTile.Bench.TritonBenchG.ChunkLinearAttn.Correct
open VeriTile.Triton
open scoped BigOperators
noncomputable section

section ForwardState
open RealHMath RealHProgram
open scoped RealHIO

/-- Active history stores must be injective across chunks and lanes; the
optional final-forwardStateBefore store must also be injective. Bounds include the optional
seed/final accesses and every K/V/history access in the full NT loop. -/
def forwardState (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) : RealHIO.IO p where
  kernel := (ChunkLinearAttn.forwardH storage seedType finalType r.key r.value r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    p.T p.K p.V p.BT p.BK p.BV p.NT hasSeed storeFinal).eraseDType
  projection := RealHIO.projection p r storage seedType finalType hasSeed storeFinal Bool.false
  read := fun s => ⟨s.readMem r.key,s.readMem r.value,s.readMem r.seed⟩
  valid := fun s => RealHLoop.HistoryNoCollision p (launch s) ∧
    (storeFinal = Bool.true → RealHStore.NoCollision p (launch s) (RealHFinal.base p (launch s)) p.V 1)
  bounds := fun s => RealHSafety.Bounds p (launch s) r hasSeed storeFinal
  history := r.history
  final := r.final
  storeFinal := storeFinal
  historyWindow := fun l c e j => historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)
  finalWindow := fun l e j => stateAddress p l (keyIndex p l e.val) (valueIndex p l j.val)
  stateMask := fun l e j => keyIndex p l e.val < p.K ∧ valueIndex p l j.val < p.V

def forwardStateDotLeft (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) : RealHIO.IO p :=
  { forwardState p r storage seedType finalType hasSeed storeFinal with
    kernel := (ChunkLinearAttn.forwardHDotLeft storage seedType finalType r.key r.value r.history r.seed r.final
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
      p.T p.K p.V p.BT p.BK p.BV p.NT hasSeed storeFinal).eraseDType
    projection := RealHIO.projection p r storage seedType finalType hasSeed storeFinal Bool.true }

/-- Initial seed plus all earlier chunk contractions. The array accessors
apply the source strides and zero extension; they do not execute the kernel. -/
def forwardStateBefore (p : Layout) (hasSeed : Bool) (l : Launch) (a : Inputs) (c k v : Nat) : ℝ :=
  seed p l a hasSeed k v + ∑ t ∈ Finset.range c, ∑ lane : Fin p.BT,
    keyValue p l a k (t*p.BT+lane.val)*valueValue p l a (t*p.BT+lane.val) v

def forwardStateFormula (p : Layout) (hasSeed : Bool) (l : Launch) (a : Inputs) : RealHIO.Outputs p where
  history := fun c e j => forwardStateBefore p hasSeed l a c (keyIndex p l e.val) (valueIndex p l j.val)
  final := fun e j => forwardStateBefore p hasSeed l a p.NT (keyIndex p l e.val) (valueIndex p l j.val)

specification forwardState_correct (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) (hkey : r.key ≠ r.history) (hvalue : r.value ≠ r.history)
    (hfinal : storeFinal = Bool.true → r.history ≠ r.final) :
    forwardState p r storage seedType finalType hasSeed storeFinal ⊨ forwardStateFormula p hasSeed := by
  exact RealHIO.implements p r storage seedType finalType hasSeed storeFinal Bool.false ⟨hkey,hvalue⟩ hfinal

specification forwardStateDotLeft_correct (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) (hkey : r.key ≠ r.history) (hvalue : r.value ≠ r.history)
    (hfinal : storeFinal = Bool.true → r.history ≠ r.final) :
    forwardStateDotLeft p r storage seedType finalType hasSeed storeFinal ⊨ forwardStateFormula p hasSeed := by
  exact RealHIO.implements p r storage seedType finalType hasSeed storeFinal Bool.true ⟨hkey,hvalue⟩ hfinal
end ForwardState

section ForwardOutput
open OutputMath RealOutput ChunkGlaFwd.RealIO
open scoped ChunkGlaFwd.RealIO

/-- Bounds cover every Q/K/H tile, the V tile and output. Output addresses
must be injective among active lanes. Inputs may alias output: all loads
precede its sole store. Both feature strides stay independent. -/
def forwardOutput (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (ChunkLinearAttn.forwardO storage valueType out r.query r.key r.value r.history r.output
    p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey
    scale p.T p.K p.V p.BT p.BK p.BV).eraseDType
  projection := OutputIO.projection p r storage valueType out scale Bool.false
  grid := fun s => ⟨s.pids 0,s.pids 1,s.pids 2⟩
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history⟩
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => OutputSafe.Bounds p (launch s) r
  out := r.output
  write := fun l i => l.head*p.valueHead+(l.chunk*p.BT+i.1.val)*p.valueTime+(l.value*p.BV+i.2.1.val)*p.valueDimension
  mask := fun l i => l.chunk*p.BT+i.1.val < p.T ∧ l.value*p.BV+i.2.1.val < p.V

def forwardOutputScaleFirst (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) :=
  { forwardOutput p r storage valueType out scale with
    kernel := (ChunkLinearAttn.forwardOScaleFirst storage valueType out r.query r.key r.value r.history r.output
      p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey
      scale p.T p.K p.V p.BT p.BK p.BV).eraseDType
    projection := OutputIO.projection p r storage valueType out scale Bool.true }

/-- Q·H plus causal Q·K·V, followed by scale. Sums cover every key block;
all accessors read initial arrays with the source strides and zero extension. -/
def forwardOutputFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
  ((∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
      query p l a i.1.val (kb*p.BK+e.val)*history p l a (kb*p.BK+e.val) i.2.1.val) +
    ∑ j : Fin p.BT,
      (if j.val ≤ i.1.val then
        ∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
          query p l a i.1.val (kb*p.BK+e.val)*key p l a (kb*p.BK+e.val) j.val
      else 0)*value p l a j.val i.2.1.val)*scale

specification forwardOutput_correct (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    forwardOutput p r storage valueType out scale ⊨ forwardOutputFormula p scale := by
  exact OutputIO.implements p r storage valueType out scale Bool.false

specification forwardOutputScaleFirst_correct (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    forwardOutputScaleFirst p r storage valueType out scale ⊨ forwardOutputFormula p scale := by
  exact OutputIO.implements p r storage valueType out scale Bool.true
end ForwardOutput

section BackwardState
open RealHMath (Layout Launch keyIndex valueIndex historyAddress)
open RealDhMath RealDhProgram
open scoped RealDhIO

/-- Stores are injective across active chunks/lanes. Bounds cover Q, dO and
dH at every reverse iteration. Only the two inputs reloaded after each store
must be distinct from history output. -/
def backwardState (p : Layout) (r : Regions) (queryType gradientType : Storage) (out : FloatDType) (scale : ℝ) : RealDhIO.IO p where
  kernel := (ChunkLinearAttn.backwardH queryType gradientType out r.query r.incoming r.history
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    scale p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
  projection := RealDhIO.projection p r queryType gradientType out scale Bool.false
  read := fun s => ⟨s.readMem r.query,s.readMem r.incoming⟩
  valid := fun s => RealDhLoop.HistoryNoCollision p (launch s)
  bounds := fun s => RealDhSafety.Bounds p (launch s) r
  output := r.history
  window := fun l c e j => historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)
  mask := fun l e j => keyIndex p l e.val < p.K ∧ valueIndex p l j.val < p.V

def backwardStateDotLeft (p : Layout) (r : Regions) (queryType gradientType : Storage) (out : FloatDType) (scale : ℝ) : RealDhIO.IO p :=
  { backwardState p r queryType gradientType out scale with
    kernel := (ChunkLinearAttn.backwardHDotLeft queryType gradientType out r.query r.incoming r.history
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
      scale p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
    projection := RealDhIO.projection p r queryType gradientType out scale Bool.true }

/-- Only strictly later chunks contribute. The current chunk is excluded
because the source stores its carry before adding the current contraction. -/
def backwardStateFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : RealDhIO.Values p :=
  fun c e j => ∑ t : Fin p.NT, if c < t.val then
    ∑ lane : Fin p.BT,
      (query p l a (keyIndex p l e.val) (t.val*p.BT+lane.val)*scale)*
        incoming p l a (t.val*p.BT+lane.val) (valueIndex p l j.val)
    else 0

specification backwardState_correct (p : Layout) (r : Regions) (queryType gradientType : Storage) (out : FloatDType) (scale : ℝ)
    (hquery : r.query ≠ r.history) (hgradient : r.incoming ≠ r.history) :
    backwardState p r queryType gradientType out scale ⊨ backwardStateFormula p scale := by
  exact RealDhIO.implements p r queryType gradientType out scale Bool.false ⟨hquery,hgradient⟩

specification backwardStateDotLeft_correct (p : Layout) (r : Regions) (queryType gradientType : Storage) (out : FloatDType) (scale : ℝ)
    (hquery : r.query ≠ r.history) (hgradient : r.incoming ≠ r.history) :
    backwardStateDotLeft p r queryType gradientType out scale ⊨ backwardStateFormula p scale := by
  exact RealDhIO.implements p r queryType gradientType out scale Bool.true ⟨hquery,hgradient⟩
end BackwardState

section BackwardGradients
open RealHMath (Layout)
open RealQKVMath RealQKVProgram
open scoped RealQKVIO

/-- dV writes must be injective across all value blocks; dQ/dK writes must
be injective in their active tile. Bounds cover every input/output access.
The dV base includes the runtime head count as in the source. -/
def backwardGradients (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealQKVIO.IO p where
  kernel := (ChunkLinearAttn.backwardQKV storage r.query r.key r.value r.history r.incoming r.historyGradient r.dq r.dk r.dv
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    scale p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
  projection := RealQKVIO.projection p r storage scale Bool.false
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.value,s.readMem r.history,
    s.readMem r.incoming,s.readMem r.historyGradient⟩
  valid := fun s => RealQKVLoop.NoCollision p (launch s) r ∧ (RealQKVFinal.keyWindow p (launch s) r.dq).NoCollision
  bounds := fun s => RealQKVSafety.Bounds p (launch s) r
  dq := r.dq
  dk := r.dk
  dv := r.dv
  keyWindow := fun l i e => l.head*p.keyHead+(l.chunk*p.BT+i.val)*p.keyTime+(l.key*p.BK+e.val)*p.keyFeature
  valueWindow := fun l n i j => (l.key*l.heads+l.head)*p.valueHead+(l.chunk*p.BT+i.val)*p.valueTime+(n*p.BV+j.val)*p.valueFeature
  keyMask := fun l i e => l.chunk*p.BT+i.val < p.T ∧ l.key*p.BK+e.val < p.K
  valueMask := fun l n i j => l.chunk*p.BT+i.val < p.T ∧ n*p.BV+j.val < p.V

def backwardGradientsDotLeft (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealQKVIO.IO p :=
  { backwardGradients p r storage scale with
    kernel := (ChunkLinearAttn.backwardQKVDotLeft storage r.query r.key r.value r.history r.incoming r.historyGradient r.dq r.dk r.dv
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
      scale p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
    projection := RealQKVIO.projection p r storage scale Bool.true }

/-- Upper causal K·Q score. -/
def backwardGradientsScore (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  if i ≤ j then (∑ e : Fin p.BK, key p l a i e.val*query p l a e.val j)*scale else 0

/-- Lower causal dO·Vᵀ score gradient, summing every value block. -/
def backwardGradientsScoreGradient (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  if i ≥ j then (∑ c ∈ Finset.range (numBlocks p), ∑ v : Fin p.BV,
    incoming p l a c i v.val*value p l a c j v.val)*scale else 0

/-- Independent initial-array contractions for dQ, dK and dV. Accessors
retain the source strides and distinct K versus NT*K history boundary checks. -/
def backwardGradientsFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : RealQKVIO.Values p where
  query := fun i e =>
    (∑ c ∈ Finset.range (numBlocks p),
      (∑ v : Fin p.BV, incoming p l a c i.val v.val*history p l a c v.val e.val)*scale) +
      ∑ j : Fin p.BT, backwardGradientsScoreGradient p scale l a i.val j.val*key p l a j.val e.val
  key := fun i e =>
    (∑ c ∈ Finset.range (numBlocks p), ∑ v : Fin p.BV,
      value p l a c i.val v.val*historyGradient p l a c e.val v.val) +
      ∑ j : Fin p.BT, query p l a e.val j.val*backwardGradientsScoreGradient p scale l a j.val i.val
  value := fun c i j =>
    (∑ e : Fin p.BK, key p l a i.val e.val*historyGradient p l a c e.val j.val) +
      ∑ t : Fin p.BT, backwardGradientsScore p scale l a i.val t.val*incoming p l a c t.val j.val

/-- The four repeatedly loaded inputs are separate from dV; the three
outputs are pairwise distinct. Q/K can alias dV because they are loaded first. -/
specification backwardGradients_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ)
    (hv : r.value ≠ r.dv) (ho : r.incoming ≠ r.dv) (hh : r.history ≠ r.dv) (hdh : r.historyGradient ≠ r.dv)
    (hqk : r.dq ≠ r.dk) (hqv : r.dq ≠ r.dv) (hkv : r.dk ≠ r.dv) :
    backwardGradients p r storage scale ⊨ backwardGradientsFormula p scale := by
  exact RealQKVIO.implements p r storage scale Bool.false ⟨⟨hv,ho,hh,hdh⟩,hqk,hqv,hkv⟩

specification backwardGradientsDotLeft_correct (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ)
    (hv : r.value ≠ r.dv) (ho : r.incoming ≠ r.dv) (hh : r.history ≠ r.dv) (hdh : r.historyGradient ≠ r.dv)
    (hqk : r.dq ≠ r.dk) (hqv : r.dq ≠ r.dv) (hkv : r.dk ≠ r.dv) :
    backwardGradientsDotLeft p r storage scale ⊨ backwardGradientsFormula p scale := by
  exact RealQKVIO.implements p r storage scale Bool.true ⟨⟨hv,ho,hh,hdh⟩,hqk,hqv,hkv⟩
end BackwardGradients

end
end VeriTile.Bench.TritonBenchG.ChunkLinearAttn.Correct

#guard_msgs (drop info) in
#auditModuleAxioms
