import bench.tritonbench_g.block_sparse_attn.Proofs.StreamingMath

/-! Independent attention formula from the initial mathematical arrays and
signed CSR schedule. Layout row/column strides are distinct. Both channel
branches, duplicate columns, empty rows and masked prefixes are represented.
The recurrence theorem supplies the mathematical part of the loop invariant;
source execution, input pinning and memory safety remain separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.CSRMath
open VeriTile.Triton
open scoped BigOperators
noncomputable section

/-- Initial mathematical arrays for one batch and query/KV head. Key and
value indices stay signed until memory-safety obligations establish validity. -/
structure Inputs where
  query : Nat → Nat → ℝ
  key : Int → Nat → ℝ
  value : Int → Nat → ℝ
  rowIndex : Int → Int
  columnIndex : Int → Int

/-- The row and column strides are independent CSR layout parameters. -/
structure Schedule where
  layout : Nat
  rowStride : Nat
  columnStride : Nat
  queryBlock : Nat

def first (s : Schedule) (xs : Inputs) : Int :=
  xs.rowIndex ((s.layout * s.rowStride + s.queryBlock : Nat) : Int)

def last (s : Schedule) (xs : Inputs) : Int :=
  xs.rowIndex ((s.layout * s.rowStride + s.queryBlock + 1 : Nat) : Int)

/-- Number of iterations of the source's signed, unit-stride CSR range. -/
def count (s : Schedule) (xs : Inputs) : Nat := (last s xs - first s xs).toNat

/-- The natural visit counter describes the signed loop without clipping
its initial index. This includes negative starts and reversed intervals. -/
theorem visit_before_end (s : Schedule) (xs : Inputs) (visit : Nat) :
    visit < count s xs ↔ first s xs + (visit : Int) < last s xs := by
  unfold count
  omega

theorem count_zero (s : Schedule) (xs : Inputs) :
    count s xs = 0 ↔ last s xs ≤ first s xs := by
  unfold count
  omega

theorem end_reached (s : Schedule) (xs : Inputs) :
    last s xs ≤ first s xs + (count s xs : Int) := by
  unfold count
  omega

/-- Global key index selected by a particular CSR visit and local lane.
Repeated column entries retain their multiplicity in the weighted sum. -/
def selectedKey (s : Schedule) (xs : Inputs) (BN visit : Nat) (lane : Fin BN) : Int :=
  xs.columnIndex ((s.layout * s.columnStride : Nat) + first s xs + (visit : Int)) *
    (BN : Int) + (lane.val : Int)

/-- Exactly one or two dot-product channel blocks, followed by scaling. -/
def dotScore (xs : Inputs) (BD blocks queryRow : Nat) (keyRow : Int) (scale : ℝ) : ℝ :=
  ((∑ d : Fin BD, xs.query queryRow d.val * xs.key keyRow d.val) +
    if 2 ≤ blocks then
      ∑ d : Fin BD, xs.query queryRow (BD + d.val) * xs.key keyRow (BD + d.val)
    else 0) * scale

def score (s : Schedule) (xs : Inputs) (BN BD blocks queryRow : Nat) (scale : ℝ)
    (visit : Nat) (lane : Fin BN) : WithBot ℝ :=
  let key := selectedKey s xs BN visit lane
  if key ≤ (queryRow : Int) then (dotScore xs BD blocks queryRow key scale : WithBot ℝ) else ⊥

/-- Independent formula over the initial CSR-selected keys; no execution
state, loop invariant or final register is used in this definition. -/
def attention (s : Schedule) (xs : Inputs) (BN BD blocks queryRow channel : Nat)
    (scale : ℝ) : ℝ :=
  let w := fun visit (lane : Fin BN) =>
    let key := selectedKey s xs BN visit lane
    if key ≤ (queryRow : Int) then Real.exp (dotScore xs BD blocks queryRow key scale) else 0
  (∑ visit ∈ Finset.range (count s xs), ∑ lane : Fin BN,
    w visit lane * xs.value (selectedKey s xs BN visit lane) channel) /
  (∑ visit ∈ Finset.range (count s xs), ∑ lane : Fin BN, w visit lane)

/-- The normalized row recurrence computes the initial-array formula for
arbitrary CSR order, duplicates, empty rows and masked prefixes. The separate
execution proof must connect these arrays/addresses to the complete kernel. -/
theorem recurrence_eq_attention (s : Schedule) (xs : Inputs)
    (BN BD blocks queryRow channel : Nat) (scale : ℝ) :
    StreamingMath.accumulated (score s xs BN BD blocks queryRow scale)
      (fun visit lane => xs.value (selectedKey s xs BN visit lane) channel) (count s xs) =
      attention s xs BN BD blocks queryRow channel scale := by
  rw [StreamingMath.accumulated_eq_formula]
  simp only [StreamingMath.moment, StreamingMath.mass, attention, score]
  simp only [apply_ite, StreamingMath.weight_coe, StreamingMath.weight_bot]

/-- On a stored query row, a causally visible key also satisfies the source's
upper-bound key mask. Nonnegative addresses remain a memory-safety obligation. -/
theorem visible_key_in_upper_bound {queryRow N : Nat} {key : Int}
    (hq : queryRow < N) (hk : key ≤ (queryRow : Int)) : key < (N : Int) := by
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.CSRMath
