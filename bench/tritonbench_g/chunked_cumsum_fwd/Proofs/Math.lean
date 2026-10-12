import bench.tritonbench_g.chunked_cumsum_fwd.Kernels
import VeriTile.Triton

/-! Independent initial-array formulas for both outputs. The final partial
chunk stores explicit zero dt values through chunk_size, but its cumulative
output continues the final valid prefix into those padded output lanes.
A is read after the dt_out store: a whole-kernel initial-array proof must keep
that output separate from A, in addition to separating the two output regions. -/
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Math
open VeriTile.Triton
open scoped BigOperators
noncomputable section

structure Layout where
  batches : Nat
  sequence : Nat
  heads : Nat
  chunk : Nat
  BH : Nat
  BC : Nat
  dtBatch : Nat
  dtTime : Nat
  dtHead : Nat
  rateHead : Nat
  biasHead : Nat
  outBatch : Nat
  outChunk : Nat
  outHead : Nat
  outColumn : Nat
  sumBatch : Nat
  sumChunk : Nat
  sumHead : Nat
  sumColumn : Nat

structure Launch where
  batch : Nat
  chunk : Nat
  head : Nat

structure Inputs where
  dt : Nat → ℝ
  rate : Nat → ℝ
  bias : Nat → ℝ

def head (p : Layout) (l : Launch) (i : Nat) := l.head*p.BH+i
def time (p : Layout) (l : Launch) (j : Nat) := l.chunk*p.chunk+j

def dtAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.batch*p.dtBatch+l.chunk*p.chunk*p.dtTime+head p l i*p.dtHead+j*p.dtTime

def outAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.batch*p.outBatch+l.chunk*p.outChunk+head p l i*p.outHead+j*p.outColumn

def sumAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.batch*p.sumBatch+l.chunk*p.sumChunk+head p l i*p.sumHead+j*p.sumColumn

def activeInput (p : Layout) (l : Launch) (i j : Nat) : Prop :=
  head p l i < p.heads ∧ j < min p.chunk (p.sequence-l.chunk*p.chunk)
def activeOutput (p : Layout) (l : Launch) (i j : Nat) : Prop :=
  head p l i < p.heads ∧ j < p.chunk
instance (p : Layout) (l : Launch) (i j : Nat) : Decidable (activeInput p l i j) :=
  inferInstanceAs (Decidable (_ ∧ _))
instance (p : Layout) (l : Launch) (i j : Nat) : Decidable (activeOutput p l i j) :=
  inferInstanceAs (Decidable (_ ∧ _))

def softplus (x : ℝ) : ℝ := if x ≤ 20 then Real.log (1+Real.exp x) else x

def prepared (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (i j : Nat) : ℝ :=
  if activeInput p l i j then
    let x := a.dt (dtAddress p l i j) + if hasBias then a.bias (head p l i*p.biasHead) else 0
    upper.cap (max (if useSoftplus then softplus x else x) lower)
  else 0

def rate (p : Layout) (l : Launch) (a : Inputs) (i : Nat) : ℝ :=
  if head p l i < p.heads then a.rate (head p l i*p.rateHead) else 0

def cumulative (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (i : Fin p.BH) (j : Fin p.BC) : ℝ :=
  ∑ t : Fin p.BC, if t.val ≤ j.val then
    prepared p l a hasBias useSoftplus lower upper i.val t.val * rate p l a i.val else 0

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Math
