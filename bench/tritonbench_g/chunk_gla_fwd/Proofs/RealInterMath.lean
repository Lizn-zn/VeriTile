import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent real contraction for the inter-subblock producer. Inputs are
initial arrays; no expression below is defined by running a kernel. Matrix
loads use their actual strides and zero extension. The separate flat gate
reference uses K as its row stride. Its cancellation is a real exponential
identity only; the floating-point equivalence does not use that identity.
The connection to complete source execution is a separate proof obligation. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterMath
open scoped BigOperators
noncomputable section

structure Layout where
  headStride : Nat
  timeStride : Nat
  T : Nat
  K : Nat
  BT : Nat
  BC : Nat
  BK : Nat
  NC : Nat

structure Launch where
  chunk : Nat
  pair : Nat
  head : Nat

structure Inputs where
  query : Nat → ℝ
  key : Nat → ℝ
  gate : Nat → ℝ

def rowBlock (p : Layout) (l : Launch) := l.pair / p.NC
def columnBlock (p : Layout) (l : Launch) := l.pair % p.NC
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+rowBlock p l*p.BC
def columnStart (p : Layout) (l : Launch) := l.chunk*p.BT+columnBlock p l*p.BC
def numBlocks (p : Layout) := (p.K+p.BK-1)/p.BK

def enabled (p : Layout) (l : Launch) : Prop :=
  rowStart p l < p.T ∧ columnBlock p l < rowBlock p l

def matrixAddress (p : Layout) (l : Launch) (t k : Nat) :=
  l.head*p.headStride+t*p.timeStride+k

def referenceAddress (p : Layout) (l : Launch) (k : Nat) :=
  l.head*p.headStride+rowStart p l*p.K+k

def outputAddress (p : Layout) (l : Launch) (i j : Nat) :=
  l.head*p.T*p.BT+(rowStart p l+i)*p.BT+(columnBlock p l*p.BC+j)

def active (p : Layout) (l : Launch) (i j : Nat) : Prop :=
  rowStart p l+i < p.T ∧ columnBlock p l*p.BC+j < p.BT

def matrixValue (p : Layout) (l : Launch) (a : Nat → ℝ) (t k : Nat) : ℝ :=
  if t < p.T ∧ k < p.K then a (matrixAddress p l t k) else 0

def reference (p : Layout) (l : Launch) (a : Inputs) (k : Nat) : ℝ :=
  if k < p.K then a.gate (referenceAddress p l k) else 0

def queryFactor (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i k : Nat) : ℝ :=
  matrixValue p l a.query (rowStart p l+i) k *
    Real.exp (matrixValue p l a.gate (rowStart p l+i) k-reference p l a k) * scale

def keyFactor (p : Layout) (l : Launch) (a : Inputs) (j k : Nat) : ℝ :=
  matrixValue p l a.key (columnStart p l+j) k *
    Real.exp (reference p l a k-matrixValue p l a.gate (columnStart p l+j) k)

/-- An unshifted gated query/key product. No gate-reference input or
source register appears in the target formula. -/
def term (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j k : Nat) : ℝ :=
  matrixValue p l a.query (rowStart p l+i) k *
    matrixValue p l a.key (columnStart p l+j) k *
    Real.exp (matrixValue p l a.gate (rowStart p l+i) k-
      matrixValue p l a.gate (columnStart p l+j) k) * scale

def partialSum (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n i j : Nat) : ℝ :=
  ∑ kb ∈ Finset.range n, ∑ e : Fin p.BK, term p l a scale i j (kb*p.BK+e.val)

def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) : ℝ :=
  partialSum p l a scale (numBlocks p) i j

theorem gate_cancellation (q k row center col scale : ℝ) :
    (q*Real.exp (row-center)*scale)*(k*Real.exp (center-col)) =
      q*k*Real.exp (row-col)*scale := by
  rw [show row-col = (row-center)+(center-col) by ring,Real.exp_add]
  ring

theorem factor_product (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j k : Nat) :
    queryFactor p l a scale i k * keyFactor p l a j k = term p l a scale i j k :=
  gate_cancellation _ _ _ _ _ _

@[simp] theorem partial_zero (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i j : Nat) :
    partialSum p l a scale 0 i j = 0 := by simp [partialSum]

theorem partial_step (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (n i j : Nat) :
    partialSum p l a scale (n+1) i j = partialSum p l a scale n i j +
      ∑ e : Fin p.BK, queryFactor p l a scale i (n*p.BK+e.val) *
        keyFactor p l a j (n*p.BK+e.val) := by
  simp only [factor_product,partialSum,Finset.sum_range_succ]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.RealInterMath
