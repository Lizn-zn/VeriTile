import VeriTile.Triton.Math.Attention
import VeriTile.Meta.StatementAudit

/-! Independent column-score formula. Q, K and the supplied base-2 row
normalization M are input functions. The formula neither executes a kernel nor
assumes that M was computed by an omitted producer. Source load checks and its
IS_EVEN_N-controlled query mask remain separate predicates. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealFormula
open VeriTile.Triton
noncomputable section

structure Inputs (D : Nat) where
  query : Nat → Fin D → ℝ
  key : Nat → Fin D → ℝ
  normalizer : Nat → ℝ

def distance (row key : Nat) (offset : Int) : Int := (row : Int) - key + offset

/-- The complement arm keeps distance >= size, rather than negating both
bounds of the regular window. Negative offsets and sizes retain their meaning. -/
def windowAllows (window complement : Bool) (offset size : Int) (row key : Nat) : Prop :=
  window = Bool.false ∨ if complement then size ≤ distance row key offset
    else 0 ≤ distance row key offset ∧ distance row key offset < size

instance (w c : Bool) (offset size : Int) (r k : Nat) :
    Decidable (windowAllows w c offset size r k) := by
  unfold windowAllows
  infer_instance

def score {D : Nat} (xs : Inputs D) (scale : ℝ) (N NK : Nat)
    (evenM evenN : Bool) (row key : Nat) : ℝ :=
  (Finset.univ.sum fun d : Fin D =>
    (if evenM || decide (row < N) then xs.query row d else 0) *
    (if evenN || decide (key < NK) then xs.key key d else 0)) *
      (scale * 1.4426950408889634)

def weight {D : Nat} (xs : Inputs D) (scale : ℝ) (N NK : Nat)
    (window complement evenM evenN : Bool) (offset size : Int) (row key : Nat) : ℝ :=
  if windowAllows window complement offset size row key ∧
      (evenN = Bool.true ∨ row < N) then
    pow2 (score xs scale N NK evenM evenN row key - xs.normalizer row)
  else 0

/-- Sum over all query blocks actually visited. The execution bridge must
identify blocks with the dynamic range's trip count; this definition alone
makes no statement about execution, memory bounds or the host grid. -/
def column {D : Nat} (xs : Inputs D) (scale : ℝ) (N NK B blocks : Nat)
    (window complement evenM evenN : Bool) (offset size : Int) (key : Nat) : ℝ :=
  ∑ block : Fin blocks, ∑ row : Fin B,
    weight xs scale N NK window complement evenM evenN offset size
      (block.val * B + row.val) key

/-- The induction step for a running column sum, independent of the source
recurrence. This is the closed formula used by the future execution invariant. -/
theorem column_succ {D : Nat} (xs : Inputs D) (scale : ℝ) (N NK B blocks : Nat)
    (window complement evenM evenN : Bool) (offset size : Int) (key : Nat) :
    column xs scale N NK B (blocks + 1) window complement evenM evenN offset size key =
      column xs scale N NK B blocks window complement evenM evenN offset size key +
        ∑ row : Fin B, weight xs scale N NK window complement evenM evenN offset size
          (blocks * B + row.val) key := by
  simp only [column, Fin.sum_univ_castSucc]
  rfl

theorem no_window (complement : Bool) (offset size : Int) (row key : Nat) :
    windowAllows Bool.false complement offset size row key := Or.inl rfl

/-- In the usual window, future keys are excluded even when the size is
positive. Natural subtraction would turn these negative distances into zero. -/
theorem future_key_excluded (row key : Nat) (size : Int) (hf : row < key) :
    ¬ windowAllows Bool.true Bool.false 0 size row key := by
  simp only [windowAllows, Bool.true_eq_false, Bool.false_eq_true, ↓reduceIte,
    false_or, distance]
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealFormula
