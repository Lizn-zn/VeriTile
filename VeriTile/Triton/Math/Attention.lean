/-
VeriTile.Triton.Math.Attention

Pure real-valued attention specifications and shape helpers.
-/

import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic
import VeriTile.Triton.Core.Shape

namespace VeriTile.Triton

/-! ## Pure attention math -/

/-- Scaled score `(Q · K^T · scale)[i, j]` for real-valued Q/K slices. -/
noncomputable def scaledScore {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) : ℝ :=
  scale * Finset.univ.sum (fun d : Fin D =>
    Q (i, d, PUnit.unit) * K (j, d, PUnit.unit))

/-- ℝ-valued reference attention:
`softmax(Q · K^T · scale) · V`. -/
noncomputable def attentionReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let score := fun j : Fin S => scaledScore Q K scale i j
    let denom := Finset.univ.sum (fun j : Fin S => Real.exp (score j))
    let numer := Finset.univ.sum (fun j : Fin S =>
      Real.exp (score j) * V (j, d, PUnit.unit))
    numer / denom

/-- Causal attention for one 2D slice. A key position `j` contributes to
query position `i` exactly when `j ≤ i`; future keys contribute zero
softmax mass. -/
noncomputable def attentionRealCausal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let score := fun j : Fin S => scaledScore Q K scale i j
    let weight := fun j : Fin S =>
      if j.val ≤ i.val then Real.exp (score j) else 0
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S =>
      weight j * V (j, d, PUnit.unit))
    numer / denom

/-- Causal attention for a local Q block whose row `i` corresponds to
global query row `qStart + i`. -/
noncomputable def attentionRealCausalBlock {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let score := fun j : Fin S => scaledScore Q K scale i j
    let weight := fun j : Fin S =>
      if j.val ≤ qStart + i.val then Real.exp (score j) else 0
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S =>
      weight j * V (j, d, PUnit.unit))
    numer / denom

/-! ## 4D layout-independent views -/

/-- Slice a 4D `[B, H, S, D]` tile-as-function at fixed `(batch, head)`,
yielding a 2D `[S, D]` tile-as-function. -/
def sliceBH {B H S D : Nat}
    (T : TileIndex [B, H, S, D] → ℝ)
    (b : Fin B) (h : Fin H) : TileIndex [S, D] → ℝ :=
  fun (i, d, _) => T (b, h, i, d, PUnit.unit)

/-- 4D ℝ-valued reference attention. Each `(batch, head)` slice is
independent and computes ordinary 2D `attentionReal`. -/
noncomputable def attentionReal4D {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionReal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

/-- 4D ℝ-valued causal reference attention. -/
noncomputable def attentionReal4DCausal {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealCausal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

@[simp] theorem attentionReal4D_slice {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4D Q K V scale (b, h, i, d, PUnit.unit)
      = attentionReal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          scale (i, d, PUnit.unit) := rfl

@[simp] theorem attentionReal4DCausal_slice {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DCausal Q K V scale (b, h, i, d, PUnit.unit)
      = attentionRealCausal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          scale (i, d, PUnit.unit) := rfl

/-- M-row slice of the `(b, h)` plane of a 4D tensor. -/
def slice4DQRows {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) (hBnd : start * M + M ≤ S) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    T (b, h, ⟨start * M + i.val, by
      have := i.isLt
      omega⟩, d, PUnit.unit)

/-- Boundary-masked M-row Q block. In-bounds rows read the logical Q tensor;
out-of-bounds rows are the `other=0` value supplied to `tl.load`. -/
def slice4DQRowsBoundary {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    if hIn : start * M + i.val < S then
      T (b, h, ⟨start * M + i.val, hIn⟩, d, PUnit.unit)
    else
      0

@[simp] theorem slice4DQRowsBoundary_of_lt {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) (i : Fin M) (d : Fin D)
    (hIn : start * M + i.val < S) :
    slice4DQRowsBoundary M T b h start (i, d, PUnit.unit) =
      T (b, h, ⟨start * M + i.val, hIn⟩, d, PUnit.unit) := by
  simp [slice4DQRowsBoundary, hIn]

@[simp] theorem slice4DQRowsBoundary_of_not_lt {B H S D : Nat} (M : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (start : Nat) (i : Fin M) (d : Fin D)
    (hOut : ¬ start * M + i.val < S) :
    slice4DQRowsBoundary M T b h start (i, d, PUnit.unit) = 0 := by
  simp [slice4DQRowsBoundary, hOut]

/-- Pad a logical hidden dimension `D` to a block hidden dimension `Bd`.
Out-of-range hidden lanes are zero. -/
def padHeadD {S D Bd : Nat} (X : TileIndex [S, D] → ℝ) :
    TileIndex [S, Bd] → ℝ :=
  fun (i, d, _) =>
    if h : d.val < D then
      X (i, ⟨d.val, h⟩, PUnit.unit)
    else
      0

@[simp] theorem padHeadD_of_lt {S D Bd : Nat}
    (X : TileIndex [S, D] → ℝ) (i : Fin S) (d : Fin Bd)
    (h : d.val < D) :
    padHeadD X (i, d, PUnit.unit) = X (i, ⟨d.val, h⟩, PUnit.unit) := by
  simp [padHeadD, h]

@[simp] theorem padHeadD_of_not_lt {S D Bd : Nat}
    (X : TileIndex [S, D] → ℝ) (i : Fin S) (d : Fin Bd)
    (h : ¬ d.val < D) :
    padHeadD X (i, d, PUnit.unit) = 0 := by
  simp [padHeadD, h]

/-- A zero-padded hidden-dimension sum over `Fin Bd` reduces to the logical
sum over `Fin D` when the logical dimension is covered by the block width. -/
theorem sum_padHeadD_eq {D Bd : Nat} (hDLe : D ≤ Bd) (f : Fin D → ℝ) :
    (Finset.univ : Finset (Fin Bd)).sum (fun d : Fin Bd =>
      if h : d.val < D then f ⟨d.val, h⟩ else 0)
      = (Finset.univ : Finset (Fin D)).sum f := by
  rw [show (Finset.univ.sum (fun d : Fin Bd =>
        if h : d.val < D then f ⟨d.val, h⟩ else 0)) =
        ((Finset.univ : Finset (Fin Bd)).filter (fun d => d.val < D)).sum
          (fun d : Fin Bd => if h : d.val < D then f ⟨d.val, h⟩ else 0)
        from ?_]
  · refine (Finset.sum_bij (fun (d : Fin D) (_ : d ∈ Finset.univ) =>
      Fin.castLE hDLe d) ?_ ?_ ?_ ?_).symm
    · intro d _
      simp [Fin.val_castLE, d.isLt]
    · intro d₁ _ d₂ _ heq
      apply Fin.ext
      have := congrArg Fin.val heq
      simpa [Fin.val_castLE] using this
    · intro d hd
      simp at hd
      refine ⟨⟨d.val, hd⟩, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      simp
    · intro d _
      have hLt : (Fin.castLE hDLe d).val < D := by
        rw [Fin.val_castLE]; exact d.isLt
      rw [dif_pos hLt]
      congr 1
  · refine (Finset.sum_filter_of_ne ?_).symm
    intro d _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

theorem scaledScore_padHeadD_eq {M S D Bd : Nat} (hDLe : D ≤ Bd)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) (j : Fin S) :
    scaledScore (D := Bd) (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) scale i j =
      scaledScore Q K scale i j := by
  unfold scaledScore
  congr 1
  rw [show (Finset.univ : Finset (Fin Bd)).sum (fun d : Fin Bd =>
        padHeadD Q (i, d, PUnit.unit) * padHeadD K (j, d, PUnit.unit)) =
      (Finset.univ : Finset (Fin Bd)).sum (fun d : Fin Bd =>
        if h : d.val < D then
          Q (i, ⟨d.val, h⟩, PUnit.unit) * K (j, ⟨d.val, h⟩, PUnit.unit)
        else
          0) by
    apply Finset.sum_congr rfl
    intro d _
    by_cases h : d.val < D
    · simp [padHeadD, h]
    · simp [padHeadD, h]]
  rw [sum_padHeadD_eq hDLe (fun d : Fin D =>
    Q (i, d, PUnit.unit) * K (j, d, PUnit.unit))]

/-- Causal block attention is invariant under zero-padding the hidden
dimension, on logical output lanes `d < D`. -/
theorem attentionRealCausalBlock_padHeadD_eq {M S D Bd : Nat} (hDLe : D ≤ Bd)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Bd]) (hDIdx : idx.2.1.val < D) :
    attentionRealCausalBlock qStart
        (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
        scale idx =
    attentionRealCausalBlock qStart Q K V scale
        (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  cases u
  have hD : d.val < D := hDIdx
  unfold attentionRealCausalBlock
  have hV : ∀ j : Fin S,
      padHeadD V (j, d, PUnit.unit) = V (j, ⟨d.val, hD⟩, PUnit.unit) := by
    intro j
    simp [padHeadD, hD]
  have hScoreT : ∀ j : Fin S,
      scaledScore (padHeadD (Bd := Bd) Q)
          (padHeadD (Bd := Bd) K) scale i j =
        scaledScore Q K scale i j := by
    intro j
    simpa using scaledScore_padHeadD_eq hDLe Q K scale i j
  simp [hScoreT, hV]

/-- Causal attention is invariant under zero-padding the hidden dimension,
on logical output lanes `d < D`. -/
theorem attentionRealCausal_padHeadD_eq {M S D Bd : Nat} (hDLe : D ≤ Bd)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Bd]) (hDIdx : idx.2.1.val < D) :
    attentionRealCausal
        (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
        scale idx =
    attentionRealCausal Q K V scale
        (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  cases u
  have hD : d.val < D := hDIdx
  unfold attentionRealCausal
  have hV : ∀ j : Fin S,
      padHeadD V (j, d, PUnit.unit) = V (j, ⟨d.val, hD⟩, PUnit.unit) := by
    intro j
    simp [padHeadD, hD]
  have hScoreT : ∀ j : Fin S,
      scaledScore (padHeadD (Bd := Bd) Q)
          (padHeadD (Bd := Bd) K) scale i j =
        scaledScore Q K scale i j := by
    intro j
    simpa using scaledScore_padHeadD_eq hDLe Q K scale i j
  simp [hScoreT, hV]

/-- Non-causal attention is invariant under zero-padding the hidden
dimension, on logical output lanes `d < D`. -/
theorem attentionReal_padHeadD_eq {M S D Bd : Nat} (hDLe : D ≤ Bd)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Bd]) (hDIdx : idx.2.1.val < D) :
    attentionReal
        (padHeadD (Bd := Bd) Q) (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
        scale idx =
      attentionReal Q K V scale
        (idx.1, ⟨idx.2.1.val, hDIdx⟩, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  cases u
  have hD : d.val < D := hDIdx
  have hV : ∀ j : Fin S,
      padHeadD V (j, d, PUnit.unit) = V (j, ⟨d.val, hD⟩, PUnit.unit) := by
    intro j
    simp [padHeadD, hD]
  have hScoreT : ∀ j : Fin S,
      scaledScore (padHeadD (Bd := Bd) Q)
          (padHeadD (Bd := Bd) K) scale i j =
        scaledScore Q K scale i j := by
    intro j
    simpa using scaledScore_padHeadD_eq hDLe Q K scale i j
  unfold attentionReal
  simp [hScoreT, hV]

/-- Reinterpret the `(b, h)` plane of a 4D `[B, H, S, D]` tensor as a
flat `[Bk * numKVBlocks, D]` view, given `Bk * numKVBlocks = S`. -/
def slice4DFlat {B H S D : Nat} (Bk numKVBlocks : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (hSk : Bk * numKVBlocks = S) : TileIndex [Bk * numKVBlocks, D] → ℝ :=
  fun (j, d, _) =>
    T (b, h, ⟨j.val, by
      have := j.isLt
      omega⟩, d, PUnit.unit)

end VeriTile.Triton
