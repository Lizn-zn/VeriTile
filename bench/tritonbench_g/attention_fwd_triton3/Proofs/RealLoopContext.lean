import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealLoop

/-! Relate the executed loop's pointer views and tile masks to ordinary
full-context attention inputs. The unchecked-load flag requires valid full
blocks. Checked tails remain allowed. All dimensions and strides are symbolic. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopContext
open VeriTile.Triton
open RealTiles RealScores RealAccumulation RealLoopFormula RealLoop
open scoped BigOperators
noncomputable section

def keyInput {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (key : Nat) (d : Fin BD) : ℝ :=
  s.readMem p.K (p.kBase + d.val * p.strideKD + key * p.strideKN)

def valueInput {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (key : Nat) (d : Fin BD) : ℝ :=
  s.readMem p.V (p.vBase + key * p.strideVN + d.val * p.strideVD)

def fullScore {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (row : Fin BM) (key : Nat) : ℝ :=
  (∑ d : Fin BD, p.q (row, d, PUnit.unit) * keyInput s p key d) * p.scale

def keepKey {BM BN BD : Nat} (p : Parameters BM BN BD) (row : Fin BM) : Nat → Bool :=
  RealKeys.keep p.startM BM row.val p.window p.complement p.offset p.size

/-- The unchecked source load arm is valid only for complete KV blocks. -/
def FullBlocks {BM BN BD : Nat} (p : Parameters BM BN BD) : Prop :=
  p.even ≠ 0 → ∀ i < count p, cursor p i + BN ≤ p.context

/-- The Python launch heuristic sets IS_EVEN_N from context % BLOCK_N = 0.
Together with the source's aligned/clipped bounds, this suffices for every
unchecked iteration; callers need not assume a condition on executed values. -/
theorem full_blocks_of_source {BM BN BD : Nat} (p : Parameters BM BN BD) (hN : 0 < BN)
    (hflag : p.even ≠ 0 → p.context % BN = 0)
    (hlo : p.lo = RealKeys.lower p.startM BM BN p.window p.complement p.offset p.size)
    (hhi : p.hi.toNat = RealKeys.upper p.startM BM BN p.context p.window p.complement p.offset) :
    FullBlocks p := by
  have halign : BN ∣ p.lo := by
    rw [hlo, RealKeys.lower]
    split
    · exact WindowBounds.lower_bound_aligned _ _ _ _ _ hN
    · exact dvd_zero BN
  have hcap : p.hi.toNat ≤ p.context := by
    rw [hhi, RealKeys.upper]
    split
    · dsimp [WindowBounds.upperBound]
      omega
    · exact Nat.le_refl _
  intro he i hi
  have hc : cursor p i < p.context :=
    Nat.lt_of_lt_of_le ((RealKeys.iteration_lt_count p.lo p.hi.toNat BN i hN).mp hi) hcap
  obtain ⟨loBlocks, hloBlocks⟩ := halign
  obtain ⟨contextBlocks, hcontextBlocks⟩ := Nat.dvd_of_mod_eq_zero (hflag he)
  have hcursor : cursor p i = BN * (loBlocks + i) := by simp [cursor, hloBlocks, Nat.mul_add, Nat.mul_comm]
  rw [hcursor, hcontextBlocks] at hc ⊢
  have hblocks : loBlocks + i < contextBlocks := (Nat.mul_lt_mul_left hN).mp hc
  have hnext := Nat.mul_le_mul_left BN (Nat.succ_le_of_lt hblocks)
  simpa [Nat.mul_add, Nat.mul_one] using hnext

theorem key_value_at {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (i : Nat) (d : Fin BD) (key : Fin BN) (hkey : cursor p i + key.val < p.context) :
    values s (keyPointer p i) (checks p.even) (d, key, PUnit.unit) =
      keyInput s p (cursor p i + key.val) d := by
  by_cases he : p.even ≠ 0
  · simp [values, keyPointer, checks, he, keyInput]
  · simp [values, keyPointer, checks, he, keyInput, hkey]

theorem value_value_at {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (i : Nat) (key : Fin BN) (d : Fin BD) (hkey : cursor p i + key.val < p.context) :
    value s p i (key, d, PUnit.unit) = valueInput s p (cursor p i + key.val) d := by
  by_cases he : p.even ≠ 0
  · simp [value, values, valuePointer, checks, he, valueInput]
  · simp [value, values, valuePointer, checks, he, valueInput, hkey]

theorem score_at {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (i : Nat) (row : Fin BM) (key : Fin BN) (hkey : cursor p i + key.val < p.context) :
    score s p i row key = fullScore s p row (cursor p i + key.val) := by
  simp only [score, RealNumeric.scores, fullScore]
  congr 1
  apply Finset.sum_congr rfl
  intro d _
  rw [key_value_at s p i d key hkey]

theorem active_at {BM BN BD : Nat} (p : Parameters BM BN BD) (hf : FullBlocks p)
    (i : Nat) (hi : i < count p) (row : Fin BM) (key : Fin BN) :
    active p i row key =
      (decide (cursor p i + key.val < p.context) && keepKey p row (cursor p i + key.val)) := by
  rw [active, keep_matches_keys _ _ _ _ _ _ _ _ _ _ _ _ (by intro he; have := hf he i hi; omega)]
  exact Bool.and_comm _ _

theorem active_key_in_context {BM BN BD : Nat} (p : Parameters BM BN BD) (hf : FullBlocks p)
    (i : Nat) (hi : i < count p) (row : Fin BM) (key : Fin BN)
    (h : active p i row key = Bool.true) : cursor p i + key.val < p.context := by
  rw [active_at p hf i hi row key] at h
  exact of_decide_eq_true (Bool.and_eq_true_iff.mp h).1

theorem finRange_global (BN pos : Nat) :
    (List.finRange BN).map (fun key => pos + key.val) = List.range' pos BN := by
  rw [show (fun key : Fin BN => pos + key.val) = (fun key : Nat => pos + key) ∘ Fin.val from rfl,
    ← List.map_map, List.map_coe_finRange_eq_range]
  exact List.range'_eq_map_range.symm

/-- Map one local tile's kept pairs to its ordered global key interval.
This uses real loads only on kept in-range lanes; no equality of whole kernel
outputs or all-context experimental sampling is a premise. -/
theorem pairs_global {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (hf : FullBlocks p) (i : Nat) (hi : i < count p) (row : Fin BM) (lane : Fin BD) :
    pairs (score s p i row) (fun key => value s p i (key, lane, PUnit.unit)) (active p i row) =
      ((List.range' (cursor p i) BN).filter
        (fun key => decide (key < p.context) && keepKey p row key)).map
          (fun key => (fullScore s p row key, valueInput s p key lane)) := by
  rw [← finRange_global, List.filter_map, List.map_map]
  unfold pairs
  have hfilter : (List.finRange BN).filter (active p i row) =
      (List.finRange BN).filter (fun key =>
        decide (cursor p i + key.val < p.context) && keepKey p row (cursor p i + key.val)) := by
    congr 1
    funext key
    exact active_at p hf i hi row key
  simp only [Function.comp_def]
  rw [← hfilter]
  apply List.map_congr_left
  intro key hkey
  have hk := active_key_in_context p hf i hi row key (List.mem_filter.mp hkey).2
  exact Prod.ext (score_at s p i row key hk) (value_value_at s p i key lane hk)

theorem blocks_global {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (hf : FullBlocks p) (row : Fin BM) (lane : Fin BD) :
    blocks (fun i => score s p i row) (fun i => active p i row)
        (fun i key => value s p i (key, lane, PUnit.unit)) (count p) =
      RealKeys.pairedBlocks p.lo p.hi.toNat BN p.context (keepKey p row)
        (fun key => (fullScore s p row key, valueInput s p key lane)) := by
  unfold blocks RealKeys.pairedBlocks
  apply List.map_congr_left
  intro i hi
  exact pairs_global s p hf i (List.mem_range.mp hi) row lane

theorem mass_pairs_global {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (hf : FullBlocks p) (i : Nat) (hi : i < count p) (row : Fin BM) :
    pairs (score s p i row) (fun _ => 0) (active p i row) =
      ((List.range' (cursor p i) BN).filter
        (fun key => decide (key < p.context) && keepKey p row key)).map
          (fun key => (fullScore s p row key, 0)) := by
  rw [← finRange_global, List.filter_map, List.map_map]
  unfold pairs
  have hfilter : (List.finRange BN).filter (active p i row) =
      (List.finRange BN).filter (fun key =>
        decide (cursor p i + key.val < p.context) && keepKey p row (cursor p i + key.val)) := by
    congr 1
    funext key
    exact active_at p hf i hi row key
  simp only [Function.comp_def]
  rw [← hfilter]
  apply List.map_congr_left
  intro key hkey
  have hk := active_key_in_context p hf i hi row key (List.mem_filter.mp hkey).2
  exact Prod.ext (score_at s p i row key hk) rfl

theorem mass_blocks_global {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (hf : FullBlocks p) (row : Fin BM) :
    blocks (fun i => score s p i row) (fun i => active p i row) (fun _ _ => 0) (count p) =
      RealKeys.pairedBlocks p.lo p.hi.toNat BN p.context (keepKey p row)
        (fun key => (fullScore s p row key, 0)) := by
  unfold blocks RealKeys.pairedBlocks
  apply List.map_congr_left
  intro i hi
  exact mass_pairs_global s p hf i (List.mem_range.mp hi) row

def fullRow {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) (row : Fin BM) (initialValue : ℝ) (v : Nat → ℝ) : RealFormula.RowState :=
  if (p.lo : Int) < p.hi then
    RealFormula.summarize (rowSeed seed row initialValue)
      (((List.range p.context).filter (keepKey p row)).map (fun key => (fullScore s p row key, v key)))
  else rowSeed seed row initialValue

/-- Full-context formula for all row registers, with an explicit no-iteration
case. It uses input memory and mathematical sums, without kernel execution. -/
def fullResult {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) : Carry BM BD where
  center := fun row => (fullRow s p seed row 0 (fun _ => 0)).center
  denominator := fun row => (fullRow s p seed row 0 (fun _ => 0)).denominator
  numerator := fun i => (fullRow s p seed i.1 (seed.numerator i) (fun key => valueInput s p key i.2.1)).numerator

/-- Closed formula over the complete context, with the actual source window
bounds. The two bound equalities are arithmetic obligations for the prologue. -/
theorem final_row {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) (hN : 0 < BN) (hf : FullBlocks p)
    (hlo : p.lo = RealKeys.lower p.startM BM BN p.window p.complement p.offset p.size)
    (hhi : p.hi.toNat = RealKeys.upper p.startM BM BN p.context p.window p.complement p.offset)
    (row : Fin BM) (lane : Fin BD) :
    rowSeed (after s p seed (count p)) row ((after s p seed (count p)).numerator (row, lane, PUnit.unit)) =
      if (p.lo : Int) < p.hi then
        RealFormula.summarize (rowSeed seed row (seed.numerator (row, lane, PUnit.unit)))
          (((List.range p.context).filter (keepKey p row)).map
            (fun key => (fullScore s p row key, valueInput s p key lane)))
      else rowSeed seed row (seed.numerator (row, lane, PUnit.unit)) := by
  rw [after, closed_row, blocks_global s p hf row lane, ← RealFormula.block_updates_eq_closed]
  have hcover : ∀ key < p.context, keepKey p row key = Bool.true → p.lo ≤ key ∧ key < p.hi.toNat := by
    intro key hk hm
    rw [hlo, hhi]
    exact RealKeys.kept_key_in_interval _ _ _ _ _ _ _ _ _ _ hN row.isLt hk hm
  rw [RealKeys.blocked_result_eq_closed _ _ _ _ _ _ _ hN hcover]
  have ht : p.lo < p.hi.toNat ↔ (p.lo : Int) < p.hi := by omega
  simp only [ht]

theorem final_mass {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) (hN : 0 < BN) (hf : FullBlocks p)
    (hlo : p.lo = RealKeys.lower p.startM BM BN p.window p.complement p.offset p.size)
    (hhi : p.hi.toNat = RealKeys.upper p.startM BM BN p.context p.window p.complement p.offset)
    (row : Fin BM) :
    (after s p seed (count p)).center row = (fullRow s p seed row 0 (fun _ => 0)).center ∧
      (after s p seed (count p)).denominator row = (fullRow s p seed row 0 (fun _ => 0)).denominator := by
  have hcover : ∀ key < p.context, keepKey p row key = Bool.true → p.lo ≤ key ∧ key < p.hi.toNat := by
    intro key hk hm
    rw [hlo, hhi]
    exact RealKeys.kept_key_in_interval _ _ _ _ _ _ _ _ _ _ hN row.isLt hk hm
  have hr : RealFormula.result (rowSeed seed row 0)
      (blocks (fun i => score s p i row) (fun i => active p i row) (fun _ _ => 0) (count p)) =
      fullRow s p seed row 0 (fun _ => 0) := by
    rw [mass_blocks_global s p hf row, ← RealFormula.block_updates_eq_closed,
      RealKeys.blocked_result_eq_closed _ _ _ _ _ _ _ hN hcover]
    have ht : p.lo < p.hi.toNat ↔ (p.lo : Int) < p.hi := by omega
    simp only [fullRow, ht]
  exact ⟨congrArg RealFormula.RowState.center hr, congrArg RealFormula.RowState.denominator hr⟩

theorem after_eq_full_context {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) (hN : 0 < BN) (hf : FullBlocks p)
    (hlo : p.lo = RealKeys.lower p.startM BM BN p.window p.complement p.offset p.size)
    (hhi : p.hi.toNat = RealKeys.upper p.startM BM BN p.context p.window p.complement p.offset) :
    after s p seed (count p) = fullResult s p seed := by
  apply Carry.ext
  · funext row; exact (final_mass s p seed hN hf hlo hhi row).1
  · funext row; exact (final_mass s p seed hN hf hlo hhi row).2
  · funext i
    rcases i with ⟨row, lane, ⟨⟩⟩
    have h := congrArg RealFormula.RowState.numerator (final_row s p seed hN hf hlo hhi row lane)
    exact h

/-- Successful execution of the whole loop with the complete-context
formula in its output registers, including zero iterations and masked tails. -/
theorem loop_full_context {BM BN BD : Nat} (s : BlockState) (p : Parameters BM BN BD)
    (seed : Carry BM BD) (hN : 0 < BN) (hf : FullBlocks p)
    (hlo : p.lo = RealKeys.lower p.startM BM BN p.window p.complement p.offset p.size)
    (hhi : p.hi.toNat = RealKeys.upper p.startM BM BN p.context p.window p.complement p.offset)
    (hregs : Registers p seed 0 s) :
    ∃ s', stepForRangeIntAux "start_n" p.lo p.hi BN
        (RealBody.body BM BN BD p.context p.window p.complement p.even p.offset p.size) s = some s' ∧
      Registers p (fullResult s p seed) (count p) s' ∧ Frame writes s s' := by
  obtain ⟨s', hs, hr, hframe⟩ := loop_executes s p seed hN hregs
  rw [after_eq_full_context s p seed hN hf hlo hhi] at hr
  exact ⟨s', hs, hr, hframe⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopContext
