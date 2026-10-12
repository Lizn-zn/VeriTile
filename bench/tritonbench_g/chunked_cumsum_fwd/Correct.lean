import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.RealIO

/-! Real correctness of the two complete sources in Kernels. Both outputs
are specified from initial arrays; execution and the complete memory frame
are conclusions. This contract is per program, for finite flat allocations.
It includes the default positive-infinite upper limit. -/
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Correct
open VeriTile.Triton Math Program Initialization
open scoped RealIO BigOperators
noncomputable section

/-- The chunk starts at or before seqlen, as in the host launch. Active output
addresses are injective. Bounds cover only actual dt/bias/A loads and output
stores; the bias buffer is unused when hasBias=false. dt and bias may alias an
output because their complete loads precede the stores. -/
def original (p : Layout) (r : Regions) (bias : BiasStorage) (dtype rateType : FloatDType)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) : RealIO.IO p where
  kernel := (Program.kernel p r bias Bool.false dtype rateType hasBias useSoftplus lower upper).eraseDType
  projection := RealIO.projection p r bias Bool.false dtype rateType hasBias useSoftplus lower upper
  read := fun s => ⟨s.readMem r.dt,s.readMem r.rate,s.readMem r.bias⟩
  valid := fun s => (launch s).chunk*p.chunk ≤ p.sequence ∧
    Stores.NoCollision p (launch s) (Execution.outOffsets p (launch s)) ∧
    Stores.NoCollision p (launch s) (Execution.sumOffsets p (launch s))
  bounds := fun s => Safety.Bounds p (launch s) r hasBias
  preparedOutput := r.out
  cumulativeOutput := r.sum
  preparedWindow := fun l i j => outAddress p l i.val j.val
  cumulativeWindow := fun l i j => sumAddress p l i.val j.val
  mask := fun l i j => head p l i.val < p.heads ∧ j.val < p.chunk

def scaleFirst (p : Layout) (r : Regions) (bias : BiasStorage) (dtype rateType : FloatDType)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) : RealIO.IO p :=
  { original p r bias dtype rateType hasBias useSoftplus lower upper with
    kernel := (Program.kernel p r bias Bool.true dtype rateType hasBias useSoftplus lower upper).eraseDType
    projection := RealIO.projection p r bias Bool.true dtype rateType hasBias useSoftplus lower upper }

/-- Bias, optional cutoff softplus, lower/upper clamp, then remask. A padded
time lane is zero even if its bias or lower clamp would otherwise be nonzero. -/
def preparedFormula (p : Layout) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (l : Launch) (a : Inputs) (i j : Nat) : ℝ :=
  if head p l i < p.heads ∧ j < min p.chunk (p.sequence-l.chunk*p.chunk) then
    let x := a.dt (dtAddress p l i j) + if hasBias then a.bias (head p l i*p.biasHead) else 0
    let y := if useSoftplus then (if x ≤ 20 then Real.log (1+Real.exp x) else x) else x
    upper.cap (max y lower)
  else 0

/-- Prepared dt and its inclusive row prefix sum after multiplying by A.
The prefix sum continues into stored padding lanes; it is not remasked. -/
def formula (p : Layout) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (l : Launch) (a : Inputs) : RealIO.Values p where
  prepared := fun i j => preparedFormula p hasBias useSoftplus lower upper l a i.val j.val
  cumulative := fun i j => ∑ t : Fin p.BC, if t.val ≤ j.val then
    preparedFormula p hasBias useSoftplus lower upper l a i.val t.val *
      (if head p l i.val < p.heads then a.rate (head p l i.val*p.rateHead) else 0)
    else 0

/-- A is loaded after the prepared output is stored, so those regions must
be separate. The two outputs must also be separate to preserve both results. -/
specification original_correct (p : Layout) (r : Regions) (bias : BiasStorage) (dtype rateType : FloatDType)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (hRA : r.rate ≠ r.out) (hOS : r.out ≠ r.sum) :
    original p r bias dtype rateType hasBias useSoftplus lower upper ⊨
      formula p hasBias useSoftplus lower upper := by
  exact RealIO.implements p r bias Bool.false dtype rateType hasBias useSoftplus lower upper hRA hOS

specification scaleFirst_correct (p : Layout) (r : Regions) (bias : BiasStorage) (dtype rateType : FloatDType)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (hRA : r.rate ≠ r.out) (hOS : r.out ≠ r.sum) :
    scaleFirst p r bias dtype rateType hasBias useSoftplus lower upper ⊨
      formula p hasBias useSoftplus lower upper := by
  exact RealIO.implements p r bias Bool.true dtype rateType hasBias useSoftplus lower upper hRA hOS

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Correct
