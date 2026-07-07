/-
VeriTile.Semantics.TiledIndexing

Reusable tile/block indexing helpers for mapping program/lane structure to
pure mathematical indices.
-/

import VeriTile.Math.LogSumExp

namespace VeriTile

namespace TiledLogSumExp

/-! ## Contiguous block indexing -/

/-- Lane-to-row position map for pid axis-1 = `i_d`. Requires the block to fit
in the row (`(i_d + 1) * B ≤ D`). -/
def laneIdx {D B : Nat} (i_d : Nat) (h_no_mask : (i_d + 1) * B ≤ D) (i : Fin B) :
    Fin D :=
  ⟨i_d * B + i.val, by
    calc i_d * B + i.val < i_d * B + B := Nat.add_lt_add_left i.isLt _
      _ = (i_d + 1) * B := by ring
      _ ≤ D := h_no_mask⟩

/-- Optionally-scaled value at a lane in a full, unmasked block. -/
noncomputable def scaledLane
    {D : Nat} (xs : Fin D → ℝ) {B : Nat} (i_d : Nat)
    (h_no_mask : (i_d + 1) * B ≤ D)
    (HAS_SCALE : Bool) (scale : ℝ) (i : Fin B) : ℝ :=
  let raw := xs (laneIdx i_d h_no_mask i)
  if HAS_SCALE then raw * scale else raw

/-- Per-block partial LSE for a full, unmasked block. -/
noncomputable def partialLSE
    {D : Nat} (xs : Fin D → ℝ) {B : Nat} (hB : 0 < B) (i_d : Nat)
    (h_no_mask : (i_d + 1) * B ≤ D)
    (HAS_SCALE : Bool) (scale : ℝ) : ℝ :=
  let scaled := scaledLane xs i_d h_no_mask HAS_SCALE scale
  let m := blockMax hB scaled
  m + Real.log (∑ i, Real.exp (scaled i - m))

/-! ## Tail-block indexing -/

/-- Set of valid lane indices for tail-block pid `i_d` with block size `n+1`. -/
def validLanes (n D i_d : Nat) : Finset (Fin (n+1)) :=
  Finset.univ.filter (fun i => i_d * (n+1) + i.val < D)

theorem validLanes_nonempty {n D i_d : Nat} (h : i_d * (n+1) < D) :
    (validLanes n D i_d).Nonempty := by
  unfold validLanes; rw [Finset.filter_nonempty_iff]
  exact ⟨⟨0, Nat.zero_lt_succ n⟩, Finset.mem_univ _, by simpa using h⟩

/-- Raw/optionally-scaled value at a tail-block lane; invalid lanes use `0`
only as a harmless value outside `validLanes`. -/
noncomputable def scaledLane_full {D : Nat} (xs : Fin D → ℝ) {n : Nat} (i_d : Nat)
    (HAS_SCALE : Bool) (scale : ℝ) (i : Fin (n+1)) : ℝ :=
  let raw := if h : i_d * (n+1) + i.val < D then xs ⟨i_d * (n+1) + i.val, h⟩ else 0
  if HAS_SCALE then raw * scale else raw

/-- Stable implementation form for a tail/masked block. User-facing block specs
should use `blockLSE`; this definition is a proof bridge for max-shifted
kernels. -/
noncomputable def partialLSE_full {D : Nat} (xs : Fin D → ℝ) {n : Nat} (i_d : Nat)
    (h_tail : i_d * (n+1) < D) (HAS_SCALE : Bool) (scale : ℝ) : ℝ :=
  let vl := validLanes n D i_d
  let h_ne := validLanes_nonempty h_tail
  let m := vl.sup' h_ne (scaledLane_full xs i_d HAS_SCALE scale)
  m + Real.log (∑ i ∈ vl, Real.exp (scaledLane_full xs i_d HAS_SCALE scale i - m))

/-- Standard block log-sum-exp over the valid lanes of a masked/tail block. -/
noncomputable def blockLSE {D : Nat} (xs : Fin D → ℝ) {n : Nat} (i_d : Nat)
    (_h_tail : i_d * (n+1) < D) (HAS_SCALE : Bool) (scale : ℝ) : ℝ :=
  Real.log (∑ i ∈ validLanes n D i_d,
    Real.exp (scaledLane_full xs i_d HAS_SCALE scale i))

/-- When the block size is exactly the row length and `i_d = 0`, the masked
stable block form is the complete stable row LSE. -/
theorem partialLSE_full_zero_self_eq_stableLSE {n : Nat}
    (xs : Fin (n+1) → ℝ) (HAS_SCALE : Bool) (scale : ℝ) :
    partialLSE_full (n := n) xs 0 (by simp) HAS_SCALE scale =
      stableLSE xs (Nat.succ_pos n) HAS_SCALE scale := by
  have h_valid : validLanes n (n+1) 0 = Finset.univ := by
    ext i
    simp [validLanes, Nat.lt_succ_iff.mp i.isLt]
  have h_scaled :
      (fun x : Fin (n+1) =>
        if HAS_SCALE = true then
          if (x : Nat) ≤ n then xs x * scale else 0
        else
          if (x : Nat) ≤ n then xs x else 0)
      =
      (fun x : Fin (n+1) =>
        if HAS_SCALE = true then xs x * scale else xs x) := by
    funext x
    have hx : (x : Nat) ≤ n := Nat.lt_succ_iff.mp x.isLt
    by_cases h : HAS_SCALE = true <;> simp [h, hx]
  unfold partialLSE_full stableLSE stableLSEWithShift blockMax
  simp [h_valid, scaledLane_full, h_scaled]
  congr 1
  apply Finset.sum_congr rfl
  intro x _hx
  congr 1
  have hx : (x : Nat) ≤ n := Nat.lt_succ_iff.mp x.isLt
  by_cases h : HAS_SCALE = true <;> simp [h, hx]

/-! ## Tiled aggregation math -/

/-- For `i_d : Fin ((D + n) / (n+1))`, the first lane of block `i_d` is in range:
`i_d.val * (n+1) < D`. -/
theorem h_tail_of_lt_cdiv {n D : Nat}
    (i_d : Fin ((D + n) / (n + 1))) :
    i_d.val * (n + 1) < D := by
  have h1 : 0 < n + 1 := Nat.succ_pos n
  have h := i_d.isLt
  rw [Nat.lt_div_iff_mul_lt h1] at h
  omega

/-- Per-block exponential pull-through: the exponential of the per-block
partial LSE equals the sum over valid lanes of `exp(scaledLane_full ...)`. -/
theorem exp_partialLSE_full_eq_sum
    {D : Nat} (xs : Fin D → ℝ) {n : Nat} (i_d : Nat)
    (h_tail : i_d * (n + 1) < D) (HAS_SCALE : Bool) (scale : ℝ) :
    Real.exp (partialLSE_full xs i_d h_tail HAS_SCALE scale) =
      ∑ i ∈ validLanes n D i_d,
        Real.exp (scaledLane_full xs i_d HAS_SCALE scale i) := by
  let vl : Finset (Fin (n+1)) := validLanes n D i_d
  let h_ne : vl.Nonempty := validLanes_nonempty h_tail
  let m : ℝ := vl.sup' h_ne (scaledLane_full xs i_d HAS_SCALE scale)
  show Real.exp (m + Real.log (∑ i ∈ vl,
        Real.exp (scaledLane_full xs i_d HAS_SCALE scale i - m))) = _
  have h_sum_pos : 0 < ∑ i ∈ vl,
      Real.exp (scaledLane_full xs i_d HAS_SCALE scale i - m) := by
    apply Finset.sum_pos
    · intro i _; exact Real.exp_pos _
    · exact h_ne
  rw [Real.exp_add, Real.exp_log h_sum_pos, Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [← Real.exp_add]
  ring_nf

/-- The stable block implementation form equals the standard block LSE. -/
theorem partialLSE_full_eq_blockLSE
    {D : Nat} (xs : Fin D → ℝ) {n : Nat} (i_d : Nat)
    (h_tail : i_d * (n + 1) < D) (HAS_SCALE : Bool) (scale : ℝ) :
    partialLSE_full xs i_d h_tail HAS_SCALE scale =
      blockLSE xs i_d h_tail HAS_SCALE scale := by
  rw [← Real.log_exp (partialLSE_full xs i_d h_tail HAS_SCALE scale)]
  rw [exp_partialLSE_full_eq_sum xs i_d h_tail HAS_SCALE scale]
  rfl

/-- Membership in `validLanes n D i_d` is exactly the bound check. -/
theorem mem_validLanes_iff {n D i_d : Nat} {i : Fin (n+1)} :
    i ∈ validLanes n D i_d ↔ i_d * (n+1) + i.val < D := by
  unfold validLanes
  simp [Finset.mem_filter, Finset.mem_univ]

/-- A block-indexed function: given `i_d`, lane `i`, and a proof that the lane is
in range, return the value of `g` at the corresponding position in `Fin D`. -/
noncomputable def blockIndex {D : Nat} (n : Nat) (g : Fin D → ℝ)
    (i_d : Nat) (i : Fin (n+1)) : ℝ :=
  if h : i_d * (n+1) + i.val < D then g ⟨i_d * (n+1) + i.val, h⟩ else 0

/-- For valid lanes, `blockIndex` evaluates `g` at the correct position. -/
theorem blockIndex_eq_of_valid {D : Nat} {n : Nat} (g : Fin D → ℝ)
    (i_d : Nat) (i : Fin (n+1)) (h : i_d * (n+1) + i.val < D) :
    blockIndex n g i_d i = g ⟨i_d * (n+1) + i.val, h⟩ := by
  unfold blockIndex; rw [dif_pos h]

/-- A sum over `Fin D` decomposes via the divmod bijection into a double sum over
blocks. -/
theorem sum_exp_partition
    {D : Nat} (n : Nat) (g : Fin D → ℝ) :
    ∑ j : Fin D, g j =
      ∑ i_d : Fin ((D + n) / (n + 1)),
        ∑ i ∈ validLanes n D i_d.val, blockIndex n g i_d.val i := by
  set ND := (D + n) / (n + 1) with hND
  set B := n + 1 with hB
  have hBpos : 0 < B := Nat.succ_pos n
  rw [Finset.sum_sigma' (Finset.univ : Finset (Fin ND))
        (fun i_d => validLanes n D i_d.val)
        (fun i_d i => blockIndex n g i_d.val i)]
  symm
  apply Finset.sum_bij'
    (i := fun (p : Σ (i_d : Fin ND), Fin (n+1)) (hp : p ∈ _) =>
      (⟨p.1.val * B + p.2.val, by
        rw [Finset.mem_sigma] at hp
        have hmem := hp.2
        rw [hB]
        exact mem_validLanes_iff.mp hmem⟩ : Fin D))
    (j := fun (j : Fin D) _ =>
      (⟨⟨j.val / B, by
          show j.val / B < ND
          rw [hND, hB, Nat.div_lt_iff_lt_mul hBpos]
          have hkey : ((D + n) / (n + 1)) * (n + 1) ≥ D := by
            have hBpos' : 0 < (n + 1 : Nat) := Nat.succ_pos n
            have hmod : (D + n) % (n + 1) ≤ n := by
              have hml : (D + n) % (n + 1) < n + 1 := Nat.mod_lt (D + n) hBpos'
              omega
            have hdam : (n + 1) * ((D + n) / (n + 1)) + (D + n) % (n + 1) = D + n :=
              Nat.div_add_mod (D + n) (n + 1)
            have hbd : (n + 1) * ((D + n) / (n + 1)) ≥ D := by omega
            have hcomm : ((D + n) / (n + 1)) * (n + 1) = (n + 1) * ((D + n) / (n + 1)) :=
              Nat.mul_comm _ _
            omega
          have hjlt := j.isLt
          show j.val < ((D + n) / (n + 1)) * (n + 1)
          omega⟩,
        ⟨j.val % B, Nat.mod_lt _ hBpos⟩⟩ : Σ (i_d : Fin ND), Fin (n+1)))
  case hi =>
    intros; exact Finset.mem_univ _
  case hj =>
    intro j _
    rw [Finset.mem_sigma]
    refine ⟨Finset.mem_univ _, ?_⟩
    rw [mem_validLanes_iff]
    show j.val / B * (n + 1) + j.val % B < D
    have hkey : j.val / B * (n + 1) + j.val % B = j.val := by
      have hB1 : (B : Nat) = n + 1 := hB
      conv_lhs => rw [hB1]
      have hdam : (n + 1) * (j.val / (n+1)) + j.val % (n+1) = j.val :=
        Nat.div_add_mod j.val (n + 1)
      linarith [Nat.mul_comm (j.val / (n+1)) (n+1)]
    rw [hkey]
    exact j.isLt
  case left_neg =>
    intro p hp
    rw [Finset.mem_sigma] at hp
    have hmem : p.1.val * (n + 1) + p.2.val < D := mem_validLanes_iff.mp hp.2
    apply Sigma.ext
    · apply Fin.ext
      show (p.1.val * B + p.2.val) / B = p.1.val
      rw [hB]
      rw [show p.1.val * (n+1) + p.2.val = (n+1) * p.1.val + p.2.val from by ring]
      rw [Nat.mul_add_div (Nat.succ_pos n), Nat.div_eq_of_lt p.2.isLt, Nat.add_zero]
    · rw [Fin.heq_ext_iff (by rfl)]
      show (p.1.val * B + p.2.val) % B = p.2.val
      rw [hB]
      rw [show p.1.val * (n+1) + p.2.val = (n+1) * p.1.val + p.2.val from by ring]
      rw [Nat.mul_add_mod, Nat.mod_eq_of_lt p.2.isLt]
  case right_neg =>
    intro j _
    apply Fin.ext
    show j.val / B * B + j.val % B = j.val
    rw [hB]
    rw [show j.val / (n+1) * (n+1) = (n+1) * (j.val / (n+1)) from Nat.mul_comm _ _]
    exact Nat.div_add_mod j.val (n + 1)
  case h =>
    intro p hp
    rw [Finset.mem_sigma] at hp
    have hmem : p.1.val * (n + 1) + p.2.val < D := mem_validLanes_iff.mp hp.2
    rw [blockIndex_eq_of_valid g p.1.val p.2 hmem]

/-- Full-row LSE. Kept in the tiled module as the aggregation endpoint. -/
noncomputable def fullRowLSE
    {D : Nat} (xs : Fin D → ℝ) (HAS_SCALE : Bool) (scale : ℝ) (_hD : 0 < D) : ℝ :=
  LSE xs HAS_SCALE scale

/-- Aggregating per-block partial LSEs with log-sum-exp yields the full-row LSE. -/
theorem aggregate_partialLSE_eq_fullRowLSE
    {D : Nat} (xs : Fin D → ℝ) (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (hD : 0 < D) :
    Real.log (∑ i_d : Fin ((D + n) / (n + 1)),
      Real.exp (partialLSE_full xs i_d.val (h_tail_of_lt_cdiv i_d)
        HAS_SCALE scale))
    = fullRowLSE xs HAS_SCALE scale hD := by
  set f : Fin D → ℝ := fun j => if HAS_SCALE then xs j * scale else xs j with hf
  have step1 : ∀ (i_d : Fin ((D + n) / (n + 1))),
      Real.exp (partialLSE_full xs i_d.val (h_tail_of_lt_cdiv i_d) HAS_SCALE scale) =
      ∑ i ∈ validLanes n D i_d.val,
        Real.exp (scaledLane_full xs i_d.val HAS_SCALE scale i) := fun i_d =>
    exp_partialLSE_full_eq_sum xs i_d.val (h_tail_of_lt_cdiv i_d) HAS_SCALE scale
  rw [Finset.sum_congr rfl (fun i_d _ => step1 i_d)]
  have step2 : ∀ (i_d : Fin ((D + n) / (n + 1))),
      ∑ i ∈ validLanes n D i_d.val,
        Real.exp (scaledLane_full xs i_d.val HAS_SCALE scale i) =
      ∑ i ∈ validLanes n D i_d.val,
        blockIndex n (fun j => Real.exp (f j)) i_d.val i := by
    intro i_d
    apply Finset.sum_congr rfl
    intro i hi
    have hmem : i_d.val * (n+1) + i.val < D := mem_validLanes_iff.mp hi
    rw [blockIndex_eq_of_valid (fun j => Real.exp (f j)) i_d.val i hmem]
    unfold scaledLane_full
    simp only [dif_pos hmem]
    show Real.exp (if HAS_SCALE then xs ⟨_, _⟩ * scale else xs ⟨_, _⟩) =
        Real.exp (f ⟨_, _⟩)
    rcases HAS_SCALE with _ | _
    · simp [hf]
    · simp [hf]
  rw [Finset.sum_congr rfl (fun i_d _ => step2 i_d)]
  rw [← sum_exp_partition n (fun j => Real.exp (f j))]
  unfold fullRowLSE LSE
  simp only [hf]

end TiledLogSumExp

end VeriTile
