import bench.tritonbench_g.block_sparse_attn.Proofs.StreamingMath

/-! Row formulas for the actual block-sparse normalization statements.
A score or maximum can be bottom; all exponentials and accumulators use the
project's real semantics. No first-visible-key or nonzero-denominator premise
is imposed. Only max reduction requires a nonempty physical key dimension. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealNormalizationValues
open VeriTile.Triton
open scoped BigOperators
noncomputable section

def finite {shape : TileShape} (f : TileIndex shape → ℝ) : Tile .real shape :=
  ⟨fun i => some (f i)⟩

def row {BM : Nat} (f : Fin BM → ℝ) : Tile .real [BM] := finite (fun i => f i.1)
def matrix {BM BN : Nat} (f : Fin BM → Fin BN → ℝ) : Tile .real [BM, BN] :=
  finite (fun i => f i.1 i.2.1)
def extendedRow {BM : Nat} (f : Fin BM → WithBot ℝ) : Tile .real [BM] := ⟨fun i => f i.1⟩
def extendedMatrix {BM BN : Nat} (f : Fin BM → Fin BN → WithBot ℝ) : Tile .real [BM, BN] :=
  ⟨fun i => f i.1 i.2.1⟩

structure Inputs (BM BN BD : Nat) where
  score : Fin BM → Fin BN → WithBot ℝ
  maximum : Fin BM → WithBot ℝ
  normalizer : Fin BM → ℝ
  accumulator : Fin BM → Fin BD → ℝ
  secondAccumulator : Fin BM → Fin BD → ℝ

namespace Inputs
variable {BM BN BD : Nat}
def blockMax (x : Inputs BM BN BD) (r : Fin BM) : WithBot ℝ := Finset.univ.sup (x.score r)
def probability (x : Inputs BM BN BD) (r : Fin BM) (j : Fin BN) : ℝ :=
  StreamingMath.shifted (x.score r j) (x.blockMax r)
def blockNormalizer (x : Inputs BM BN BD) (r : Fin BM) : ℝ := ∑ j : Fin BN, x.probability r j
def nextMax (x : Inputs BM BN BD) (r : Fin BM) : WithBot ℝ := max (x.maximum r) (x.blockMax r)
def alpha (x : Inputs BM BN BD) (r : Fin BM) : ℝ := StreamingMath.shifted (x.maximum r) (x.nextMax r)
def beta (x : Inputs BM BN BD) (r : Fin BM) : ℝ := StreamingMath.shifted (x.blockMax r) (x.nextMax r)
def nextNormalizer (x : Inputs BM BN BD) (r : Fin BM) : ℝ :=
  x.alpha r * x.normalizer r + x.beta r * x.blockNormalizer r
def probabilityScale (x : Inputs BM BN BD) (r : Fin BM) : ℝ := x.beta r / x.nextNormalizer r
def scaledProbability (x : Inputs BM BN BD) (r : Fin BM) (j : Fin BN) : ℝ :=
  x.probability r j * x.probabilityScale r
def accumulatorScale (x : Inputs BM BN BD) (r : Fin BM) : ℝ :=
  x.normalizer r / x.nextNormalizer r * x.alpha r
def scaledAccumulator (x : Inputs BM BN BD) (r : Fin BM) (d : Fin BD) : ℝ :=
  x.accumulator r d * x.accumulatorScale r
def scaledSecond (x : Inputs BM BN BD) (r : Fin BM) (d : Fin BD) : ℝ :=
  x.secondAccumulator r d * x.accumulatorScale r
end Inputs

/-- The exponentiation of a masked difference always has a finite real carrier. -/
theorem shifted_carrier (a b : WithBot ℝ) :
    WithBot.realExp (WithBot.realSub a b) = some (StreamingMath.shifted a b) := by
  cases a <;> cases b <;> rfl

theorem reduce_max {BM BN : Nat} (hN : 0 < BN) (f : Fin BM → Fin BN → WithBot ℝ) :
    Tile.reduceMaxDrop (shape := [BM, BN]) 1 (extendedMatrix f) =
      some (extendedRow (fun r => Finset.univ.sup (f r))) := by
  unfold Tile.reduceMaxDrop
  split
  · congr 1
    apply Tile.ext
    rintro ⟨r, ⟨⟩⟩
    simp only [Finset.sup'_eq_sup]
    rfl
  · rename_i h
    exact False.elim (h hN)

theorem reduce_sum {BM BN : Nat} (f : Fin BM → Fin BN → ℝ) :
    Tile.reduceSumDrop (shape := [BM, BN]) 1 (matrix f) = row (fun r => ∑ j : Fin BN, f r j) := by
  apply Tile.ext
  rintro ⟨r, ⟨⟩⟩
  change (∑ j : Fin BN, ((f r j : ℝ) : WithBot ℝ)) = ((∑ j : Fin BN, f r j : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum _ _).symm

theorem shifted_matrix {BM BN : Nat} (f : Fin BM → Fin BN → WithBot ℝ)
    (m : Fin BM → WithBot ℝ) :
    Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub (.consSame (.consR .nil))
      (extendedMatrix f) (Tile.expandDim (shape := [BM]) 1 (extendedRow m))) =
      matrix (fun r j => StreamingMath.shifted (f r j) (m r)) := by
  apply Tile.ext
  rintro ⟨r, j, ⟨⟩⟩
  exact shifted_carrier _ _

theorem shifted_row {BM : Nat} (a b : Fin BM → WithBot ℝ) :
    Tile.uop WithBot.realExp (Tile.bop NumericDType.real.sub (.consSame .nil)
      (extendedRow a) (extendedRow b)) = row (fun r => StreamingMath.shifted (a r) (b r)) := by
  apply Tile.ext
  rintro ⟨r, ⟨⟩⟩
  exact shifted_carrier _ _

theorem maximum_row {BM : Nat} (a b : Fin BM → WithBot ℝ) :
    Tile.select (Tile.cop ComparableDType.real.gt (.consSame .nil) (extendedRow a) (extendedRow b))
      (extendedRow a) (extendedRow b) = extendedRow (fun r => max (a r) (b r)) := by
  apply Tile.ext
  rintro ⟨r, ⟨⟩⟩
  simp only [Tile.select_data, Tile.cop_data, ComparableDType.gt, extendedRow,
    Broadcast.leftIndex, Broadcast.rightIndex]
  by_cases h : b r < a r
  · simp [h, max_eq_left (le_of_lt h)]
  · simp [h, max_eq_right (le_of_not_gt h)]

theorem row_add {BM : Nat} (a b : Fin BM → ℝ) :
    Tile.bop NumericDType.real.add (.consSame .nil) (row a) (row b) = row (fun r => a r + b r) := rfl

theorem row_mul {BM : Nat} (a b : Fin BM → ℝ) :
    Tile.bop NumericDType.real.mul (.consSame .nil) (row a) (row b) = row (fun r => a r * b r) := rfl

theorem row_div {BM : Nat} (a b : Fin BM → ℝ) :
    Tile.bop NumericDType.real.div (.consSame .nil) (row a) (row b) = row (fun r => a r / b r) := rfl

theorem matrix_scale {BM BN : Nat} (a : Fin BM → Fin BN → ℝ) (b : Fin BM → ℝ) :
    Tile.bop NumericDType.real.mul (.consSame (.consR .nil)) (matrix a)
      (Tile.expandDim (shape := [BM]) 1 (row b)) = matrix (fun r j => a r j * b r) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealNormalizationValues
