import bench.tritonbench_g.chunk_retention_ops.Proofs.BackwardStateFlat
import bench.tritonbench_g.chunk_retention_ops.Proofs.SourceComparison
import bench.tritonbench_g.chunk_retention.Proofs.ForwardStateFlat
import bench.tritonbench_g.chunk_retention.Proofs.ForwardOutputFlat
import bench.tritonbench_g.chunk_retention.Proofs.BackwardGradientsFlat

/-! Real correctness of all four complete sources in Kernels. These contracts
include successful execution in finite flat allocations and preservation of all
other tagged cells. Precision is erased only here. Every dimension and stride
remains symbolic; bounds and injectivity apply to actual active accesses.

The source leaves block padding unspecified. Each IO contract explicitly uses
zero unspecified values for the zero-extended mathematical formula. Host launch,
allocation construction, backend tile restrictions and inter-program races are
outside these per-program statements. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetentionOps.Correct
open VeriTile.Bench.TritonBenchG.ChunkRetention
open VeriTile.Triton
open scoped BigOperators
noncomputable section

section ForwardState
open RealHMath RealHProgram
open scoped RealHIO

/-- Active history stores must be injective across chunks and lanes; the
optional final-state store must also be injective. Bounds include the optional
seed/final accesses and every K/V/history access in the full NT loop. -/
def forwardState (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) : RealHIO.IO p where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardH storage seedType finalType r.key r.value r.history r.seed r.final
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    p.H p.T p.K p.V p.BT p.BK p.BV p.NT hasSeed storeFinal).eraseDType
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
    kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardHDotLeft storage seedType finalType r.key r.value r.history r.seed r.final
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
      p.H p.T p.K p.V p.BT p.BK p.BV p.NT hasSeed storeFinal).eraseDType
    projection := RealHIO.projection p r storage seedType finalType hasSeed storeFinal Bool.true }

/-- Mathematical decayed recurrence before chunk c. Each history tile is
stored before its chunk update. The last ragged chunk uses T % BT for both
its carry decay and signed row decay; the source accessors zero-extend arrays. -/
def forwardStateBefore (p : Layout) (hasSeed : Bool) (l : Launch) (a : Inputs) : Nat → Nat → Nat → ℝ
  | 0, k, v => seed p l a hasSeed k v
  | c+1, k, v => Decay.factor (base p l) (length p c) * forwardStateBefore p hasSeed l a c k v +
      ∑ lane : Fin p.BT, keyValue p l a k (c*p.BT+lane.val) *
        (valueValue p l a (c*p.BT+lane.val) v * Decay.key (base p l) (length p c) lane.val)

def forwardStateFormula (p : Layout) (hasSeed : Bool) (l : Launch) (a : Inputs) : RealHIO.Outputs p where
  history := fun c e j => forwardStateBefore p hasSeed l a c (keyIndex p l e.val) (valueIndex p l j.val)
  final := fun e j => forwardStateBefore p hasSeed l a p.NT (keyIndex p l e.val) (valueIndex p l j.val)

specification forwardState_correct (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) (hkey : r.key ≠ r.history) (hvalue : r.value ≠ r.history)
    (hfinal : storeFinal = Bool.true → r.history ≠ r.final) :
    forwardState p r storage seedType finalType hasSeed storeFinal ⊨ forwardStateFormula p hasSeed := by
  exact RealHIO.implements_recurrence p r storage seedType finalType hasSeed storeFinal Bool.false
    (forwardStateBefore p hasSeed) (by intros; rfl) (by intros; rfl) ⟨hkey,hvalue⟩ hfinal

specification forwardStateDotLeft_correct (p : Layout) (r : Regions) (storage : Storage) (seedType finalType : FloatDType)
    (hasSeed storeFinal : Bool) (hkey : r.key ≠ r.history) (hvalue : r.value ≠ r.history)
    (hfinal : storeFinal = Bool.true → r.history ≠ r.final) :
    forwardStateDotLeft p r storage seedType finalType hasSeed storeFinal ⊨ forwardStateFormula p hasSeed := by
  exact RealHIO.implements_recurrence p r storage seedType finalType hasSeed storeFinal Bool.true
    (forwardStateBefore p hasSeed) (by intros; rfl) (by intros; rfl) ⟨hkey,hvalue⟩ hfinal
end ForwardState

section ForwardOutput
open OutputMath RealOutput VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIO
open scoped VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealIO

/-- Bounds cover every Q/K/H tile, the V tile and output. Output addresses
must be injective among active lanes. Inputs may alias output: all loads
precede its sole store. Both feature strides stay independent. -/
def forwardOutput (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardO storage valueType out r.query r.key r.value r.history r.output
    p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey
    scale p.H p.T p.K p.V p.BT p.BK p.BV).eraseDType
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
    kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardOScaleFirst storage valueType out r.query r.key r.value r.history r.output
      p.keyHead p.keyTime p.keyDimension p.valueHead p.valueTime p.valueDimension p.historyHead p.historyKey
      scale p.H p.T p.K p.V p.BT p.BK p.BV).eraseDType
    projection := OutputIO.projection p r storage valueType out scale Bool.true }

/-- Row-decayed Q·H plus causal, distance-decayed Q·K·V, then scale.
The head base is log2(1 - 2^(-5 - head % H)). All sums read initial arrays. -/
def forwardOutputFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
  ((∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
      (query p l a i.1.val (kb*p.BK+e.val) * Decay.query (base p l) i.1.val) *
        history p l a (kb*p.BK+e.val) i.2.1.val) +
    ∑ j : Fin p.BT,
      ((∑ kb ∈ Finset.range (numBlocks p), ∑ e : Fin p.BK,
        query p l a i.1.val (kb*p.BK+e.val)*key p l a (kb*p.BK+e.val) j.val) *
        Decay.score (base p l) i.1.val j.val) * value p l a j.val i.2.1.val) * scale

specification forwardOutput_correct (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    forwardOutput p r storage valueType out scale ⊨ forwardOutputFormula p scale := by
  exact OutputIO.implements p r storage valueType out scale Bool.false

specification forwardOutputScaleFirst_correct (p : Layout) (r : Regions) (storage valueType : Storage) (out : FloatDType) (scale : ℝ) :
    forwardOutputScaleFirst p r storage valueType out scale ⊨ forwardOutputFormula p scale := by
  exact OutputIO.implements p r storage valueType out scale Bool.true
end ForwardOutput

section BackwardState
open VeriTile.Bench.TritonBenchG.ChunkRetention.RealHMath (Layout Launch keyIndex valueIndex historyAddress timeIndex)
open VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhMath VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhProgram
open scoped VeriTile.Bench.TritonBenchG.ChunkRetentionOps.RealDhIO

/-- Every history store must be injective across chunks and lanes. Bounds
cover the actual Q/dO loads and history stores. The reverse loop may be empty. -/
def backwardState (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ) : RealDhIO.IO p where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardH storage out r.query r.incoming r.history
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
  projection := RealDhIO.projection p r storage out scale Bool.false
  read := fun s => ⟨s.readMem r.query,s.readMem r.incoming⟩
  valid := fun s => RealDhLoop.HistoryNoCollision p (launch s)
  bounds := fun s => RealDhSafety.Bounds p (launch s) r
  output := r.history
  window := fun l c e j => historyAddress p l c (keyIndex p l e.val) (valueIndex p l j.val)
  mask := fun l e j => keyIndex p l e.val < p.K ∧ valueIndex p l j.val < p.V

def backwardStateDotLeft (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ) : RealDhIO.IO p :=
  { backwardState p r storage out scale with
    kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardHDotLeft storage out r.query r.incoming r.history
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
      scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
    projection := RealDhIO.projection p r storage out scale Bool.true }

/-- Carry after n reverse-time updates of initial Q/dO. A history tile is
stored before its update; unlike forward state, all backward decay lengths
remain BT, including the partial last chunk. -/
def backwardStateAfter (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : Nat → Nat → Nat → ℝ
  | 0, _, _ => 0
  | n+1, k, v => Decay.factor (RealHMath.base p l) p.BT * backwardStateAfter p scale l a n k v +
      ∑ t : Fin p.BT,
        (query p l a k (timeIndex p (p.NT-1-n) t.val) * scale) *
          (incoming p l a (timeIndex p (p.NT-1-n) t.val) v * Decay.query (RealHMath.base p l) t.val)

def backwardStateFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : RealDhIO.Values p :=
  fun c e j => backwardStateAfter p scale l a (p.NT-1-c) (keyIndex p l e.val) (valueIndex p l j.val)

specification backwardState_correct (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ)
    (hq : r.query ≠ r.history) (hd : r.incoming ≠ r.history) :
    backwardState p r storage out scale ⊨ backwardStateFormula p scale := by
  exact RealDhIO.implements_recurrence p r storage out scale Bool.false
    (backwardStateAfter p scale) (by intros; rfl) (by intros; rfl) ⟨hq,hd⟩

specification backwardStateDotLeft_correct (p : Layout) (r : Regions) (storage : Storage) (out : FloatDType) (scale : ℝ)
    (hq : r.query ≠ r.history) (hd : r.incoming ≠ r.history) :
    backwardStateDotLeft p r storage out scale ⊨ backwardStateFormula p scale := by
  exact RealDhIO.implements_recurrence p r storage out scale Bool.true
    (backwardStateAfter p scale) (by intros; rfl) (by intros; rfl) ⟨hq,hd⟩
end BackwardState

section BackwardGradients
open RealHMath (Layout)
open RealQKVMath RealQKVProgram
open scoped RealQKVIO

/-- dV writes must be injective across all value blocks; dQ/dK writes must
be injective in their active tile. Bounds cover every input/output access.
The dV base includes the runtime head count as in the source. -/
def backwardGradients (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : RealQKVIO.IO p where
  kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKV storage r.query r.key r.value r.history r.incoming r.historyGradient r.dq r.dk r.dv
    p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
    scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
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
    kernel := (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKVDotLeft storage r.query r.key r.value r.history r.incoming r.historyGradient r.dq r.dk r.dv
      p.keyHead p.keyTime p.keyFeature p.valueHead p.valueTime p.valueFeature p.historyHead p.historyKey
      scale p.H p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
    projection := RealQKVIO.projection p r storage scale Bool.true }

/-- Upper causal K·Q score. -/
def backwardGradientsScore (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  (∑ e : Fin p.BK, key p l a i e.val*query p l a e.val j) * (Decay.score (base p l) j i * scale)

/-- Lower causal dO·Vᵀ score gradient, summing every value block. -/
def backwardGradientsScoreGradient (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  (∑ c ∈ Finset.range (numBlocks p), ∑ v : Fin p.BV,
    incoming p l a c i v.val*value p l a c j v.val) * (Decay.score (base p l) i j * scale)

/-- Independent initial-array contractions for dQ, dK and dV. Accessors
retain the source strides and distinct K versus NT*K history boundary checks. -/
def backwardGradientsFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) : RealQKVIO.Values p where
  query := fun i e =>
    (∑ c ∈ Finset.range (numBlocks p),
      (∑ v : Fin p.BV, incoming p l a c i.val v.val*history p l a c v.val e.val)) *
      (Decay.query (base p l) i.val * scale) +
      ∑ j : Fin p.BT, backwardGradientsScoreGradient p scale l a i.val j.val*key p l a j.val e.val
  key := fun i e =>
    (∑ c ∈ Finset.range (numBlocks p), ∑ v : Fin p.BV,
      value p l a c i.val v.val*historyGradient p l a c e.val v.val) * Decay.key (base p l) p.BT i.val +
      ∑ j : Fin p.BT, query p l a e.val j.val*backwardGradientsScoreGradient p scale l a j.val i.val
  value := fun c i j =>
    (∑ e : Fin p.BK, key p l a i.val e.val*historyGradient p l a c e.val j.val) * Decay.key (base p l) p.BT i.val +
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
end VeriTile.Bench.TritonBenchG.ChunkRetentionOps.Correct

#guard_msgs (drop info) in
#auditModuleAxioms
