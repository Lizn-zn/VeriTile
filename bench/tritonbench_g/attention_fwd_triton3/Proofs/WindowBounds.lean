import Mathlib.Data.Int.DivMod
import Lean.Elab.Tactic.Omega
import VeriTile.Meta.StatementAudit

/-! Arithmetic needed for the signed window-bound transcription.
Triton runtime integer division truncates toward zero, while the arithmetic
comparison below uses Euclidean division. See the integer-division section of
https://triton-lang.org/main/python-api/triton-semantics.html .
These lemmas justify their agreement only after a nonnegative clamp, with a
positive block size. They are not a correspondence proof for the whole loop
or for fixed-width integer overflow. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.WindowBounds

/-- The Python lower bound is a block-aligned quotient followed by lo >= 0.
Negative numerators may have different quotients, but both clamped bounds are 0. -/
theorem clamped_division_matches (a b : Int) (hb : 0 < b) :
    ((a / b) * b).toNat = (a.tdiv b * b).toNat := by
  by_cases ha : 0 ≤ a
  · rw [Int.tdiv_eq_ediv_of_nonneg ha]
  · have he : a / b < 0 := Int.ediv_neg_of_neg_of_pos (by omega) hb
    have ht : a.tdiv b ≤ 0 := by
      rw [Int.tdiv_eq_ediv]
      split
      · omega
      · rw [Int.sign_eq_one_of_pos hb]
        omega
    rw [Int.toNat_of_nonpos (Int.mul_nonpos_of_nonpos_of_nonneg (by omega) (by omega)),
      Int.toNat_of_nonpos (Int.mul_nonpos_of_nonpos_of_nonneg ht (by omega))]

/-- Exact lower-bound expression used in the non-complement window branch. -/
def lowerBound (startM blockM blockN : Nat) (offset size : Int) : Nat :=
  (((startM * blockM : Nat) + offset - size + 1).tdiv (blockN : Int) * blockN).toNat

theorem lower_bound_matches (startM blockM blockN : Nat) (offset size : Int)
    (hN : 0 < blockN) :
    (((startM * blockM : Nat) + offset - size + 1) / (blockN : Int) * blockN).toNat =
      lowerBound startM blockM blockN offset size := by
  apply clamped_division_matches
  omega

/-- The upper bound is also capped by the nonnegative KV context length. -/
theorem capped_division_matches (a b cap : Int) (hb : 0 < b) :
    (min ((a / b) * b) cap).toNat = (min (a.tdiv b * b) cap).toNat := by
  have h := clamped_division_matches a b hb
  omega

def upperBound (startM blockM blockN context : Nat) (offset : Int) : Nat :=
  (min (((((startM + 1) * blockM : Nat) : Int) - 1 + offset + blockN).tdiv
    (blockN : Int) * blockN) context).toNat

theorem upper_bound_matches (startM blockM blockN context : Nat) (offset : Int)
    (hN : 0 < blockN) :
    (min (((((startM + 1) * blockM : Nat) : Int) - 1 + offset + blockN) /
      (blockN : Int) * blockN) context).toNat =
      upperBound startM blockM blockN context offset := by
  apply capped_division_matches
  omega

/-- Once lo is nonnegative, a nonpositive hi produces no iterations. This
explains why conversion of the upper bound must be justified at the loop,
rather than treated as a general identity of signed integers. -/
theorem loop_condition_matches (lo hi : Int) (hlo : 0 ≤ lo) :
    lo < hi ↔ lo.toNat < hi.toNat := by omega

/-- A positive stride keeps the loop index nonnegative and commutes with the
natural-index representation. This concerns exact control arithmetic only. -/
theorem step_matches (index stride : Int) (hi : 0 ≤ index) (hs : 0 < stride) :
    0 ≤ index + stride ∧ (index + stride).toNat = index.toNat + stride.toNat := by
  omega

/-- Signed query/key distance in global coordinates; local row and key
indices must not use truncated natural subtraction. -/
def distance (startM blockM row key : Nat) (offset : Int) : Int :=
  ((startM * blockM : Nat) : Int) + row - key + offset

def retained (startM blockM row key : Nat) (offset size : Int) : Prop :=
  0 ≤ distance startM blockM row key offset ∧
    distance startM blockM row key offset < size

/-- Every valid key retained by any row lies inside the source's clipped
sliding-window interval. This holds for arbitrary signed offsets and sizes;
no positivity assumption on the window size is introduced. -/
theorem retained_key_in_window (startM blockM blockN context row key : Nat)
    (offset size : Int) (hN : 0 < blockN) (hrow : row < blockM)
    (hkey : key < context) (hkeep : retained startM blockM row key offset size) :
    lowerBound startM blockM blockN offset size ≤ key ∧
      key < upperBound startM blockM blockN context offset := by
  have hbn : (0 : Int) < blockN := by omega
  have hlo : ((startM * blockM : Nat) : Int) + offset - size + 1 ≤ key := by
    dsimp [retained, distance] at hkeep
    omega
  constructor
  · rw [← lower_bound_matches _ _ _ _ _ hN]
    have he := Int.ediv_mul_le (((startM * blockM : Nat) : Int) + offset - size + 1)
      (by omega : (blockN : Int) ≠ 0)
    have ht := Int.toNat_le_toNat (Int.le_trans he hlo)
    simpa using ht
  · let u : Int := (((startM + 1) * blockM : Nat) : Int) - 1 + offset
    have hu : (key : Int) ≤ u := by
      dsimp [u]
      dsimp [retained, distance] at hkeep
      simp only [Int.add_mul, Int.one_mul]
      omega
    have hu0 : 0 ≤ u := by omega
    have he : (u + (blockN : Int)).tdiv blockN = u / blockN + 1 := by
      rw [Int.tdiv_eq_ediv_of_nonneg (by omega)]
      simpa using Int.add_mul_ediv_left u 1 (by omega : (blockN : Int) ≠ 0)
    have hnext : (key : Int) < (u + (blockN : Int)).tdiv blockN * blockN := by
      rw [he]
      have hn := Int.lt_ediv_add_one_mul_self u hbn
      omega
    change key < (min ((u + (blockN : Int)).tdiv blockN * blockN) context).toNat
    omega

/-- Clamping a block-aligned lower bound retains its block alignment. This
will be used for unchecked loads when the KV context consists of full blocks. -/
theorem lower_bound_aligned (startM blockM blockN : Nat) (offset size : Int)
    (hN : 0 < blockN) : blockN ∣ lowerBound startM blockM blockN offset size := by
  rw [← lower_bound_matches _ _ _ _ _ hN]
  let q := (((startM * blockM : Nat) : Int) + offset - size + 1) / blockN
  change blockN ∣ (q * (blockN : Int)).toNat
  by_cases hq : 0 ≤ q
  · rw [Int.toNat_mul hq (by omega)]
    simp
    exact ⟨q.toNat, Nat.mul_comm _ _⟩
  · have hp : q * (blockN : Int) ≤ 0 :=
      Int.mul_nonpos_of_nonpos_of_nonneg (by omega) (by omega)
    rw [Int.toNat_of_nonpos hp]
    exact ⟨0, by simp⟩

/-- A positive-stride loop covers every key between its entry bounds. Its
last block may extend beyond hi; the source's row/KV masks remain necessary. -/
theorem key_has_iteration (lo hi blockN key : Nat) (hN : 0 < blockN)
    (hlo : lo ≤ key) (hhi : key < hi) :
    ∃ iteration : Nat,
      lo + iteration * blockN < hi ∧
      lo + iteration * blockN ≤ key ∧
      key < lo + iteration * blockN + blockN := by
  refine ⟨(key - lo) / blockN, ?_⟩
  have he := Nat.mod_add_div (key - lo) blockN
  have hm := Nat.mod_lt (key - lo) hN
  rw [Nat.mul_comm blockN] at he
  omega

/-- Every kept key belongs to a block of the positive-stride loop interval,
including partial final blocks. Execution of the body is proved separately. -/
theorem retained_key_has_iteration (startM blockM blockN context row key : Nat)
    (offset size : Int) (hN : 0 < blockN) (hrow : row < blockM)
    (hkey : key < context) (hkeep : retained startM blockM row key offset size) :
    ∃ iteration : Nat,
      lowerBound startM blockM blockN offset size + iteration * blockN <
        upperBound startM blockM blockN context offset ∧
      lowerBound startM blockM blockN offset size + iteration * blockN ≤ key ∧
      key < lowerBound startM blockM blockN offset size + iteration * blockN + blockN := by
  obtain ⟨hl, hh⟩ := retained_key_in_window _ _ _ _ _ _ _ _ hN hrow hkey hkeep
  exact key_has_iteration _ _ _ _ hN hl hh

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.WindowBounds
