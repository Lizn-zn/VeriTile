import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealOutputIO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealInterIO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealIntraIO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealSplitIO
import bench.tritonbench_g.chunk_gla_fwd.Proofs.RealMergeIO

/-! Real correctness of the five complete kernels in `Kernels.lean`.

`io ⊨ formula` means: for disjoint finite allocations covering the declared
regions, zero unspecified values, `valid` and actual-access `bounds`, the
whole erased kernel executes, every masked output equals the formula on the
INITIAL inputs, and every other tagged memory cell is unchanged. Sizes,
strides and launch IDs are parameters. The source itself leaves block-pointer
padding unspecified; the zero-oracle premise is explicit and necessary for
the output consumer (see `Proofs/UndefinedOutput.lean`).

This is per-kernel real correctness. It does not establish the Python host's
allocation/launch composition; in particular its split temporary uses BT rows
where the kernel addresses use T rows. Precision and source casts are preserved
in `Kernels.lean` and erased only in this real specification. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.Correct
open VeriTile.Triton RealIO
open scoped BigOperators RealIO
noncomputable section
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

namespace Output
open RealOutputIO

def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BT,p.BV]) :=
  l.head*p.valueHead+(l.chunk*p.BT+i.1.val)*p.valueTime+(l.value*p.BV+i.2.1.val)
def active (p : Layout) (l : Launch) (i : TileIndex [p.BT,p.BV]) : Prop :=
  l.chunk*p.BT+i.1.val < p.T ∧ l.value*p.BV+i.2.1.val < p.V

/-- Bounds cover every active Q/G/H K-block load, A/V load and output store.
Output addresses must be injective; all inputs are read before the store. -/
def outputOriginal (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) : IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (TypedOutput.original qs vs gs history out r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV).eraseDType
  projection := RealOutputIO.projection p r qs vs gs history out scale Bool.false
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.value,s.readMem r.gate,s.readMem r.history,s.readMem r.attention⟩
  valid := fun s => Function.Injective (outputAddress p (launch s))
  bounds := fun s => RealOutputSafety.Bounds s r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey p.T p.K p.V p.BT p.BK p.BV
  out := r.output
  write := outputAddress p
  mask := active p

/-- Commute the final carry/dot addition; retain the same IO contract. -/
def outputSumLeft (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) : IO Launch Inputs (TileIndex [p.BT,p.BV]) :=
  { outputOriginal p r qs vs gs history out scale with
    kernel := (TypedOutput.sumLeft qs vs gs history out r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV).eraseDType
    projection := RealOutputIO.projection p r qs vs gs history out scale Bool.true }

/-- Full K-block gated history contraction plus causal A·V, with zero extension
outside the source block pointers' checked parents. -/
def outputFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
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

specification output_correct (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) :
    outputOriginal p r qs vs gs history out scale ⊨ outputFormula p scale := by
  exact RealOutputIO.implements p r qs vs gs history out scale Bool.false

specification output_sumLeft_correct (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) :
    outputSumLeft p r qs vs gs history out scale ⊨ outputFormula p scale := by
  exact RealOutputIO.implements p r qs vs gs history out scale Bool.true

end Output

namespace Inter
open RealInterMath RealInterProgram

def rowBlock (p : Layout) (l : Launch) := l.pair / p.NC
def columnBlock (p : Layout) (l : Launch) := l.pair % p.NC
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+rowBlock p l*p.BC
def columnStart (p : Layout) (l : Launch) := l.chunk*p.BT+columnBlock p l*p.BC
def numBlocks (p : Layout) := (p.K+p.BK-1)/p.BK

def enabled (p : Layout) (l : Launch) : Prop :=
  rowStart p l < p.T ∧ columnBlock p l < rowBlock p l

def matrixAddress (p : Layout) (l : Launch) (t k : Nat) :=
  l.head*p.headStride+t*p.timeStride+k

def outputAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.head*p.T*p.BT+(rowStart p l+i)*p.BT+(columnBlock p l*p.BC+j)

def active (p : Layout) (l : Launch) (i j : Nat) : Prop :=
  rowStart p l+i < p.T ∧ columnBlock p l*p.BC+j < p.BT

def matrixValue (p : Layout) (l : Launch) (a : Nat → ℝ) (t k : Nat) : ℝ :=
  if t < p.T ∧ k < p.K then a (matrixAddress p l t k) else 0

/-- Full K-block contraction, with zero extension beyond checked input parents. -/
def interFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  ∑ kb ∈ Finset.range ((p.K+p.BK-1)/p.BK), ∑ e : Fin p.BK,
    let k := kb*p.BK+e.val
    matrixValue p l a.query (rowStart p l+i.1.val) k *
      matrixValue p l a.key (columnStart p l+i.2.1.val) k *
      Real.exp (matrixValue p l a.gate (rowStart p l+i.1.val) k -
        matrixValue p l a.gate (columnStart p l+i.2.1.val) k) * scale

/-- Bounds cover Q/G/K, the separate flat gate reference and the final store
on each reached K block. Inputs may alias output because reads precede the store. -/
def interOriginal (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) : IO Launch Inputs (TileIndex [p.BC,p.BC]) where
  kernel := (TypedInter.original qs ks gs out r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC).eraseDType
  projection := RealInterIO.projection p r qs ks gs out scale Bool.false
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun _ => True
  bounds := fun s => RealInterSafety.Bounds p (launch s) r
  out := r.output
  write := fun l i => outputAddress p l i.1.val i.2.1.val
  mask := fun l i => enabled p l ∧ active p l i.1.val i.2.1.val

def interDotFirst (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) : IO Launch Inputs (TileIndex [p.BC,p.BC]) :=
  { interOriginal p r qs ks gs out scale with
    kernel := (TypedInter.dotFirst qs ks gs out r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC).eraseDType
    projection := RealInterIO.projection p r qs ks gs out scale Bool.true }

specification inter_correct (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) :
    interOriginal p r qs ks gs out scale ⊨ interFormula p scale := by
  exact RealInterIO.implements p r qs ks gs out scale Bool.false

specification inter_dotFirst_correct (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) :
    interDotFirst p r qs ks gs out scale ⊨ interFormula p scale := by
  exact RealInterIO.implements p r qs ks gs out scale Bool.true

end Inter

namespace Intra
open RealIntraMath RealIntraProgram

def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.block*p.BC
def columns (p : Layout) (l : Launch) := min p.BC (p.T-l.chunk*p.BT-l.block*p.BC)
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
def rowActive (p : Layout) (l : Launch) (i : Nat) : Prop := rowStart p l+i < p.T
def matrixAddress (p : Layout) (l : Launch) (i k : Nat) :=
  l.head*p.headStride+(rowStart p l+i)*p.timeStride+k
def keyAddress (p : Layout) (l : Launch) (j k : Nat) :=
  l.head*p.headStride+(rowStart p l+j)*p.K+k
def outputAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.head*p.T*p.BT+(rowStart p l+i)*p.BT+l.block*p.BC+j

/-- Causal query/key contraction on the initial arrays; matrix rows use
`timeStride`, streamed key/gate rows use `K`. -/
def intraFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : Nat × Fin p.BC) : ℝ :=
  if i.1 ≤ i.2.val then (∑ k ∈ Finset.range p.K,
      a.query (matrixAddress p l i.2.val k) * a.key (keyAddress p l i.1 k) *
        Real.exp (a.gate (matrixAddress p l i.2.val k)-a.gate (keyAddress p l i.1 k)))*scale else 0

/-- Bounds cover cached Q/G, each streamed key/gate load and row-masked store.
The column count is at most BT. Key/gate must differ from output; cached Q may alias. -/
def intraOriginal (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) : IO Launch Inputs (Nat × Fin p.BC) where
  kernel := (TypedIntra.original qs ks gs r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK).eraseDType
  projection := RealIntraIO.projection p r qs ks gs scale Bool.false
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun s => columns p (launch s) ≤ p.BT
  bounds := fun s => RealIntraSafety.Bounds p (launch s) r
  out := r.output
  write := fun l i => outputAddress p l i.2.val i.1
  mask := fun l i => enabled p l ∧ i.1 < columns p l ∧ rowActive p l i.2.val

def intraScaleFirst (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) : IO Launch Inputs (Nat × Fin p.BC) :=
  { intraOriginal p r qs ks gs scale with
    kernel := (TypedIntra.scaleFirst qs ks gs r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK).eraseDType
    projection := RealIntraIO.projection p r qs ks gs scale Bool.true }

specification intra_correct (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hK : p.K ≤ p.BK) :
    intraOriginal p r qs ks gs scale ⊨ intraFormula p scale := by
  exact RealIntraIO.implements p r qs ks gs scale Bool.false hkey hgate hK

specification intra_scaleFirst_correct (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hK : p.K ≤ p.BK) :
    intraScaleFirst p r qs ks gs scale ⊨ intraFormula p scale := by
  exact RealIntraIO.implements p r qs ks gs scale Bool.true hkey hgate hK

end Intra

namespace Split
open RealSplitMath RealSplitProgram

def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.block*p.BC
def columns (p : Layout) (l : Launch) := min p.BC (p.T-l.chunk*p.BT-l.block*p.BC)
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
def rowActive (p : Layout) (l : Launch) (i : Nat) : Prop := rowStart p l+i < p.T
def featureIndex (p : Layout) (l : Launch) (e : Nat) := l.feature*p.BK+e
def matrixAddress (p : Layout) (l : Launch) (i k : Nat) :=
  l.head*p.headStride+(rowStart p l+i)*p.timeStride+featureIndex p l k
def keyAddress (p : Layout) (l : Launch) (j k : Nat) :=
  l.head*p.headStride+(rowStart p l+j)*p.K+featureIndex p l k
def outputAddress (p : Layout) (l : Launch) (i j : Nat) :=
  (l.head+l.feature*l.heads)*p.T*p.BC+(rowStart p l+i)*p.BC+j

/-- Causal query/key contraction on the initial arrays; matrix rows use
`timeStride`, streamed key/gate rows use `K`. -/
def splitFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : Nat × Fin p.BC) : ℝ :=
  if i.1 ≤ i.2.val then (∑ e : Fin p.BK, if featureIndex p l e.val < p.K then
      a.query (matrixAddress p l i.2.val e.val) * a.key (keyAddress p l i.1 e.val) *
        Real.exp (a.gate (matrixAddress p l i.2.val e.val)-a.gate (keyAddress p l i.1 e.val)) else 0)*scale else 0

/-- Bounds cover cached Q/G, streamed key/gate and the T-strided temporary
store. Key/gate must differ from output; cached Q may alias. K may exceed BK. -/
def splitOriginal (p : Layout) (r : Regions) (qs ks gs : TypedSplit.Storage)
    (scale : ℝ) : IO Launch Inputs (Nat × Fin p.BC) where
  kernel := (TypedSplit.original qs ks gs r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC).eraseDType
  projection := RealSplitIO.projection p r qs ks gs scale Bool.false
  grid := launch p
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun _ => True
  bounds := fun s => RealSplitSafety.Bounds p (launch p s) r
  out := r.output
  write := fun l i => outputAddress p l i.2.val i.1
  mask := fun l i => enabled p l ∧ i.1 < columns p l ∧ rowActive p l i.2.val

def splitScaleFirst (p : Layout) (r : Regions) (qs ks gs : TypedSplit.Storage)
    (scale : ℝ) : IO Launch Inputs (Nat × Fin p.BC) :=
  { splitOriginal p r qs ks gs scale with
    kernel := (TypedSplit.scaleFirst qs ks gs r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC).eraseDType
    projection := RealSplitIO.projection p r qs ks gs scale Bool.true }

specification split_correct (p : Layout) (r : Regions) (qs ks gs : TypedSplit.Storage)
    (scale : ℝ)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) :
    splitOriginal p r qs ks gs scale ⊨ splitFormula p scale := by
  exact RealSplitIO.implements p r qs ks gs scale Bool.false hkey hgate

specification split_scaleFirst_correct (p : Layout) (r : Regions) (qs ks gs : TypedSplit.Storage)
    (scale : ℝ)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) :
    splitScaleFirst p r qs ks gs scale ⊨ splitFormula p scale := by
  exact RealSplitIO.implements p r qs ks gs scale Bool.true hkey hgate

end Split

namespace Merge
open RealMergeProgram

def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
def active (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Prop :=
  rowStart p l+i.1.val < p.T ∧ l.subblock*p.BC+i.2.1.val < p.BT
instance (p : Layout) (l : Launch) : Decidable (enabled p l) := inferInstanceAs (Decidable (_ < _))
instance (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Decidable (active p l i) :=
  inferInstanceAs (Decidable (_ ∧ _))
def inputAddress (p : Layout) (l : Launch) (kb : Nat) (i : TileIndex [p.BC,p.BC]) :=
  (l.head+kb*l.heads)*p.T*p.BC+(rowStart p l+i.1.val)*p.BC+i.2.1.val
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
def mergeFormula (p : Layout) (l : Launch) (a : Nat → ℝ) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  ∑ kb ∈ Finset.range p.NK, a (inputAddress p l kb i)
/-- Bounds cover all NK temporary slices and the final masked store;
NoCollision requires distinct active output addresses. The two regions may alias. -/
def mergeOriginal (p : Layout) (A A2 : RegionName) (out : FloatDType) :
    IO Launch (Nat → ℝ) (TileIndex [p.BC,p.BC]) where
  kernel := (TypedMerge.original out A A2 p.T p.BT p.BC p.NK).eraseDType
  projection := RealMergeIO.projection p A A2 out Bool.false
  grid := launch
  read := fun s => s.readMem A
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => RealMergeSafety.Bounds p (launch s) A A2
  out := A2
  write := outputAddress p
  mask := active p

def mergeLoadFirst (p : Layout) (A A2 : RegionName) (out : FloatDType) : IO Launch (Nat → ℝ) (TileIndex [p.BC,p.BC]) :=
  { mergeOriginal p A A2 out with
    kernel := (TypedMerge.loadFirst out A A2 p.T p.BT p.BC p.NK).eraseDType
    projection := RealMergeIO.projection p A A2 out Bool.true }

specification merge_correct (p : Layout) (A A2 : RegionName) (out : FloatDType) :
    mergeOriginal p A A2 out ⊨ mergeFormula p := by
  exact RealMergeIO.implements p A A2 out Bool.false

specification merge_loadFirst_correct (p : Layout) (A A2 : RegionName) (out : FloatDType) :
    mergeLoadFirst p A A2 out ⊨ mergeFormula p := by
  exact RealMergeIO.implements p A A2 out Bool.true

end Merge

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.Correct
