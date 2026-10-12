import bench.tritonbench_g.attention_fwd_triton3.Proofs.WindowBounds
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealFormula
import Mathlib.Data.List.Sort

/-! Key coverage for the independent real formula. Splitting a clipped source
interval into BLOCK_N lanes and applying the row/KV masks yields exactly the
kept keys of the full context, in order and without duplicates. These are
list/index facts; execution and memory safety remain separate obligations. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealKeys

/-- Both window flags retain the source meaning. The complement branch is
one-sided; disabling the window ignores the complement flag. -/
def keep (startM blockM row window complement : Nat) (offset size : Int) (key : Nat) : Bool :=
  if window = 0 then Bool.true
  else if complement = 0 then
    decide (0 ≤ WindowBounds.distance startM blockM row key offset ∧
      WindowBounds.distance startM blockM row key offset < size)
  else decide (size ≤ WindowBounds.distance startM blockM row key offset)

def lower (startM blockM blockN window complement : Nat) (offset size : Int) : Nat :=
  if window ≠ 0 ∧ complement = 0 then
    WindowBounds.lowerBound startM blockM blockN offset size else 0

def upper (startM blockM blockN context window complement : Nat) (offset : Int) : Nat :=
  if window ≠ 0 ∧ complement = 0 then
    WindowBounds.upperBound startM blockM blockN context offset else context

theorem kept_key_in_interval (startM blockM blockN context row window complement key : Nat)
    (offset size : Int) (hN : 0 < blockN) (hrow : row < blockM)
    (hkey : key < context) (hkeep : keep startM blockM row window complement offset size key = Bool.true) :
    lower startM blockM blockN window complement offset size ≤ key ∧
      key < upper startM blockM blockN context window complement offset := by
  by_cases hw : window = 0
  · simp [lower, upper, hw, hkey]
  · by_cases hc : complement = 0
    · have hk : WindowBounds.retained startM blockM row key offset size := by
        simpa [keep, hw, hc, WindowBounds.retained] using hkeep
      simpa [lower, upper, hw, hc] using
        WindowBounds.retained_key_in_window _ _ _ _ _ _ _ _ hN hrow hkey hk
    · simp [lower, upper, hc, hkey]

/-- Number of positive-stride iterations; the subtraction is only used in
the nonempty case. The last block can extend past the upper bound. -/
def blockCount (lo hi blockN : Nat) : Nat :=
  if lo < hi then (hi - lo - 1) / blockN + 1 else 0

theorem iteration_lt_count (lo hi blockN iteration : Nat) (hN : 0 < blockN) :
    iteration < blockCount lo hi blockN ↔ lo + iteration * blockN < hi := by
  by_cases h : lo < hi
  · simp only [blockCount, if_pos h, Nat.lt_add_one_iff, Nat.le_div_iff_mul_le hN]
    omega
  · simp only [blockCount, if_neg h]
    omega

theorem block_count_covers (lo hi blockN : Nat) (hN : 0 < blockN) :
    hi ≤ lo + blockCount lo hi blockN * blockN := by
  have h := iteration_lt_count lo hi blockN (blockCount lo hi blockN) hN
  omega

/-- Concatenating consecutive blocks introduces neither gaps nor repetitions. -/
theorem concatenate_blocks (lo blockN count : Nat) :
    (List.range count).flatMap (fun i => List.range' (lo + i * blockN) blockN) =
      List.range' lo (count * blockN) := by
  induction count with
  | zero => simp
  | succ count ih =>
    simp only [List.range_succ, List.flatMap_append, List.flatMap_singleton, ih]
    rw [List.range'_append_1]
    simp [Nat.succ_mul]

/-- Generic interval lemma: any clipping that contains all kept keys gives
the same ordered key list after the KV-tail mask, even for a partial block. -/
theorem filtered_blocks_eq_context (lo hi blockN context : Nat) (keepKey : Nat → Bool)
    (hN : 0 < blockN)
    (hcover : ∀ key < context, keepKey key = Bool.true → lo ≤ key ∧ key < hi) :
    (List.range (blockCount lo hi blockN)).flatMap
        (fun i => (List.range' (lo + i * blockN) blockN).filter
          (fun key => decide (key < context) && keepKey key)) =
      (List.range context).filter keepKey := by
  rw [← List.filter_flatMap, concatenate_blocks]
  apply List.Pairwise.eq_of_mem_iff
    ((List.pairwise_lt_range' (s := lo) (n := blockCount lo hi blockN * blockN)).filter _)
    (List.pairwise_lt_range.filter _)
  intro key
  simp only [List.mem_filter, List.mem_range'_1, Bool.and_eq_true, decide_eq_true_eq,
    List.mem_range]
  constructor
  · rintro ⟨_, hk, hp⟩
    exact ⟨hk, hp⟩
  · rintro ⟨hk, hp⟩
    obtain ⟨hl, hh⟩ := hcover key hk hp
    have hbound := block_count_covers lo hi blockN hN
    exact ⟨⟨hl, by omega⟩, hk, hp⟩

/-- Symbolic key partition for all window flags and arbitrary signed offset
and size. This supplies a full-context list to the closed attention formula. -/
theorem source_blocks_eq_context (startM blockM blockN context row window complement : Nat)
    (offset size : Int) (hN : 0 < blockN) (hrow : row < blockM) :
    let lo := lower startM blockM blockN window complement offset size
    let hi := upper startM blockM blockN context window complement offset
    let keepKey := keep startM blockM row window complement offset size
    (List.range (blockCount lo hi blockN)).flatMap
        (fun i => (List.range' (lo + i * blockN) blockN).filter
          (fun key => decide (key < context) && keepKey key)) =
      (List.range context).filter keepKey := by
  apply filtered_blocks_eq_context _ _ _ _ _ hN
  intro key hk hp
  exact kept_key_in_interval _ _ _ _ _ _ _ _ _ _ hN hrow hk hp

noncomputable section

/-- Block-local score/value pairs, retaining an empty list for an executed
block with no kept keys. Dropping those empty blocks would change M/L behavior. -/
def pairedBlocks (lo hi blockN context : Nat) (keepKey : Nat → Bool)
    (pair : Nat → ℝ × ℝ) : List (List (ℝ × ℝ)) :=
  (List.range (blockCount lo hi blockN)).map (fun i =>
    ((List.range' (lo + i * blockN) blockN).filter
      (fun key => decide (key < context) && keepKey key)).map pair)

theorem paired_blocks_flatten (lo hi blockN context : Nat) (keepKey : Nat → Bool)
    (pair : Nat → ℝ × ℝ) (hN : 0 < blockN)
    (hcover : ∀ key < context, keepKey key = Bool.true → lo ≤ key ∧ key < hi) :
    (pairedBlocks lo hi blockN context keepKey pair).flatten =
      ((List.range context).filter keepKey).map pair := by
  change (List.range (blockCount lo hi blockN)).flatMap _ = _
  rw [← List.map_flatMap, filtered_blocks_eq_context lo hi blockN context keepKey hN hcover]

theorem paired_blocks_empty (lo hi blockN context : Nat) (keepKey : Nat → Bool)
    (pair : Nat → ℝ × ℝ) :
    pairedBlocks lo hi blockN context keepKey pair = [] ↔ hi ≤ lo := by
  simp [pairedBlocks, blockCount]

/-- The block update algebra reduces to a full-context maximum and sums.
No iteration preserves the seed, including its initial denominator of one. -/
theorem blocked_result_eq_closed (seed : RealFormula.RowState)
    (lo hi blockN context : Nat) (keepKey : Nat → Bool) (pair : Nat → ℝ × ℝ)
    (hN : 0 < blockN)
    (hcover : ∀ key < context, keepKey key = Bool.true → lo ≤ key ∧ key < hi) :
    (pairedBlocks lo hi blockN context keepKey pair).foldl RealFormula.summarize seed =
      if lo < hi then RealFormula.summarize seed (((List.range context).filter keepKey).map pair)
      else seed := by
  rw [RealFormula.block_updates_eq_closed]
  simp only [RealFormula.result, paired_blocks_empty,
    paired_blocks_flatten lo hi blockN context keepKey pair hN hcover]
  split <;> split <;> (try omega) <;> rfl

/-- The final normalized fresh output is independent of how the key list is
blocked, including empty contexts and entirely masked rows. -/
theorem initialized_blocks_normalized (lo hi blockN context : Nat)
    (keepKey : Nat → Bool) (pair : Nat → ℝ × ℝ) (hN : 0 < blockN)
    (hcover : ∀ key < context, keepKey key = Bool.true → lo ≤ key ∧ key < hi) :
    let row := RealFormula.result RealFormula.initial
      (pairedBlocks lo hi blockN context keepKey pair)
    let keys := ((List.range context).filter keepKey).map pair
    row.numerator / row.denominator =
      RealFormula.unshiftedSum keys Prod.snd / RealFormula.unshiftedSum keys (fun _ => 1) := by
  dsimp only
  rw [RealFormula.initialized_result_normalized,
    paired_blocks_flatten lo hi blockN context keepKey pair hN hcover]

/-- Continuation carries arbitrary finite M/L/Out data; it does not require
a positive denominator or a nonempty new context in the total real semantics. -/
theorem resumed_blocks_normalized (m l a : ℝ) (lo hi blockN context : Nat)
    (keepKey : Nat → Bool) (pair : Nat → ℝ × ℝ) (hN : 0 < blockN)
    (hcover : ∀ key < context, keepKey key = Bool.true → lo ≤ key ∧ key < hi) :
    let row := RealFormula.result ⟨m, l, a⟩ (pairedBlocks lo hi blockN context keepKey pair)
    row.numerator / row.denominator =
      RealFormula.normalized ⟨m, l, a⟩ (((List.range context).filter keepKey).map pair) := by
  dsimp only
  rw [RealFormula.finite_resume_result_normalized,
    paired_blocks_flatten lo hi blockN context keepKey pair hN hcover]

end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealKeys
