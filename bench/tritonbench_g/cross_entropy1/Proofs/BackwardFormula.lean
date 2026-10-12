import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.CrossEntropy1.BackwardFormula
open VeriTile.Triton
noncomputable section

structure Layout where
  nCols : Nat
  B : Nat
  logitsStride : Nat
  gradientStride : Nat
  dlossStride : Nat
  totalClasses : Nat
  classStart : Int
  ignored : Int
  smoothing : ℝ
  squareScale : ℝ
  hasSmoothing : Bool

structure Launch where
  row : Nat
  block : Nat

structure Inputs where
  logits : Nat → ℝ
  lse : Nat → ℝ
  dloss : Nat → ℝ
  label : Nat → Int

def column (p : Layout) (l : Launch) (i : Nat) := l.block*p.B+i
def active (p : Layout) (l : Launch) (i : Nat) : Prop := column p l i < p.nCols
instance (p : Layout) (l : Launch) (i : Nat) : Decidable (active p l i) :=
  inferInstanceAs (Decidable (column p l i < p.nCols))

def logitAddress (p : Layout) (l : Launch) (i : Nat) := l.row*p.logitsStride+column p l i
def gradientAddress (p : Layout) (l : Launch) (i : Nat) := l.row*p.gradientStride+column p l i
def shiftedLabel (p : Layout) (l : Launch) (a : Inputs) : Int := a.label l.row-p.classStart
def upstream (p : Layout) (l : Launch) (a : Inputs) : ℝ :=
  if a.label l.row = p.ignored then 0 else a.dloss (l.row*p.dlossStride)

/-- The LSE and upstream derivative are legitimate inputs of the backward
kernel. Every term reads only these initial arrays and the initial logits. -/
def probability (p : Layout) (l : Launch) (a : Inputs) (i : Nat) : ℝ :=
  Real.exp (a.logits (logitAddress p l i)-a.lse l.row)

def adjusted (p : Layout) (l : Launch) (a : Inputs) (i : Nat) : ℝ :=
  probability p l a i+2*p.squareScale*a.lse l.row*probability p l a i

def expected (p : Layout) (l : Launch) (a : Inputs) (i : Nat) : ℝ :=
  upstream p l a *
    (if p.hasSmoothing then
      (adjusted p l a i - if (column p l i : Int) = shiftedLabel p l a then 1-p.smoothing else 0) -
        p.smoothing / p.totalClasses
     else adjusted p l a i - if (column p l i : Int) = shiftedLabel p l a then 1 else 0)

theorem ignored_zero (p : Layout) (l : Launch) (a : Inputs) (i : Nat)
    (h : a.label l.row = p.ignored) : expected p l a i = 0 := by
  simp [expected,upstream,h]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CrossEntropy1.BackwardFormula
