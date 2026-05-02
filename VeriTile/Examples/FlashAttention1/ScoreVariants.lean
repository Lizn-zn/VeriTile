/-
VeriTile.Examples.FlashAttention1.ScoreVariants

Score-level FA-1 realism references for issue #40: ALiBi, sliding-window
masks, and softcap. These are mathematical spec surfaces; the online-softmax
kernel recurrence can target them by changing only the score expression fed
to max/exp.
-/

import VeriTile.Examples.FlashAttention1.Common

namespace VeriTile.Examples

open VeriTile.Triton

namespace FA1Score

/-! ## DSL score-transform smoke kernels -/

/-- FA-1 score block with ALiBi bias. This is a typed surface-shape smoke test:
`scores = QKᵀ * scale - slope * |offs_m - offs_n|`. -/
def alibiScoreSmokeKernel (qReg kReg scoreReg : RegionName)
    (M Bk D : Nat) (scale slope : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_n := tl.arange(0, $(Bk))
  offs_d := tl.arange(0, $(D))
  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  k_ptrs := offs_n[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)
  k      := tl.load($(kReg) + k_ptrs)
  raw    := tl.dot(q, tl.trans(k)) * $ℝ(scale)
  delta  := tl.toReal(offs_n[None, :]) - tl.toReal(offs_m[:, None])
  dist   := tl.max(delta, 0 - delta)
  bias   := 0 - $ℝ(slope) * dist
  scores := raw + bias
  s_ptrs := offs_m[:, None] * $(Bk) + offs_n[None, :]
  tl.store($(scoreReg) + s_ptrs, scores)
}

/-- FA-1 score block with a symmetric sliding-window mask. -/
def slidingWindowScoreSmokeKernel (qReg kReg scoreReg : RegionName)
    (M Bk D window : Nat) (scale : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_n := tl.arange(0, $(Bk))
  offs_d := tl.arange(0, $(D))
  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  k_ptrs := offs_n[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)
  k      := tl.load($(kReg) + k_ptrs)
  raw    := tl.dot(q, tl.trans(k)) * $ℝ(scale)
  delta  := tl.toReal(offs_n[None, :]) - tl.toReal(offs_m[:, None])
  dist   := tl.max(delta, 0 - delta)
  mask   := dist < $ℝ((window : ℝ))
  scores := tl.where(mask, raw, -inf)
  s_ptrs := offs_m[:, None] * $(Bk) + offs_n[None, :]
  tl.store($(scoreReg) + s_ptrs, scores)
}

/-- FA-1 score block with Gemma-style softcap. -/
def softcapScoreSmokeKernel (qReg kReg scoreReg : RegionName)
    (M Bk D : Nat) (scale softcap : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_n := tl.arange(0, $(Bk))
  offs_d := tl.arange(0, $(D))
  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  k_ptrs := offs_n[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)
  k      := tl.load($(kReg) + k_ptrs)
  raw    := tl.dot(q, tl.trans(k)) * $ℝ(scale)
  scores := $ℝ(softcap) * tl.tanh(raw / $ℝ(softcap))
  s_ptrs := offs_m[:, None] * $(Bk) + offs_n[None, :]
  tl.store($(scoreReg) + s_ptrs, scores)
}

/-- Absolute distance on sequence indices, as a natural number. -/
def natAbsDiff (a b : Nat) : Nat :=
  if a ≤ b then b - a else a - b

/-- Generic attention over an explicit real score function. -/
noncomputable def attentionRealScore {M S D : Nat}
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let weight := fun j : Fin S => Real.exp (score i j)
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S =>
      weight j * V (j, d, PUnit.unit))
    numer / denom

/-- Generic attention over an explicit score function and visibility mask.
Masked-out scores contribute zero mass, matching `tl.where(mask, scores, -inf)`.
-/
noncomputable def attentionRealMaskedScore {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let weight := fun j : Fin S =>
      if visible i j then Real.exp (score i j) else 0
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S =>
      weight j * V (j, d, PUnit.unit))
    numer / denom

/-- Ordinary scaled dot-product score as a first-class score function. -/
noncomputable def dotScore {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) : Fin M → Fin S → ℝ :=
  fun i j => FA1Math.scaledScore Q K scale i j

/-- ALiBi additive bias `-slope * |q - k|`. -/
noncomputable def alibiBias (slope : ℝ) (qPos kPos : Nat) : ℝ :=
  -slope * (natAbsDiff qPos kPos : ℝ)

/-- Scaled dot-product score plus ALiBi bias. `qStart` is the global row
offset of the local query block. -/
noncomputable def alibiScore {M S D : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) : Fin M → Fin S → ℝ :=
  fun i j =>
    dotScore Q K scale i j + alibiBias slope (qStart + i.val) j.val

/-- Sliding-window visibility predicate `|q - k| < window`. -/
def slidingVisible (window qStart : Nat) {M S : Nat} : Fin M → Fin S → Bool :=
  fun i j => decide (natAbsDiff (qStart + i.val) j.val < window)

/-- Gemma-style softcap score transform: `softcap * tanh(score / softcap)`. -/
noncomputable def softcapScore (softcap : ℝ) (score : ℝ) : ℝ :=
  softcap * Real.tanh (score / softcap)

noncomputable def softcapDotScore {M S D : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) : Fin M → Fin S → ℝ :=
  fun i j => softcapScore softcap (dotScore Q K scale i j)

noncomputable def attentionRealAlibi {M S D : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  attentionRealScore (alibiScore qStart slope Q K scale) V

noncomputable def attentionRealSlidingWindow {M S D : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  attentionRealMaskedScore (slidingVisible window qStart)
    (dotScore Q K scale) V

noncomputable def attentionRealSoftcap {M S D : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  attentionRealScore (softcapDotScore softcap Q K scale) V

noncomputable def attentionRealAlibiSlidingSoftcap {M S D : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  let score := fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j)
  attentionRealMaskedScore (slidingVisible window qStart) score V

@[simp] theorem attentionRealScore_apply {M S D : Nat}
    (score : Fin M → Fin S → ℝ) (V : TileIndex [S, D] → ℝ)
    (i : Fin M) (d : Fin D) :
    attentionRealScore score V (i, d, PUnit.unit) =
      (Finset.univ.sum (fun j : Fin S =>
        Real.exp (score i j) * V (j, d, PUnit.unit))) /
      (Finset.univ.sum (fun j : Fin S => Real.exp (score i j))) := by
  rfl

@[simp] theorem attentionRealMaskedScore_apply {M S D : Nat}
    (visible : Fin M → Fin S → Bool) (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (i : Fin M) (d : Fin D) :
    attentionRealMaskedScore visible score V (i, d, PUnit.unit) =
      (Finset.univ.sum (fun j : Fin S =>
        (if visible i j then Real.exp (score i j) else 0) *
          V (j, d, PUnit.unit))) /
      (Finset.univ.sum (fun j : Fin S =>
        if visible i j then Real.exp (score i j) else 0)) := by
  rfl

noncomputable def attentionReal4DAlibi {B H S_q S_k D : Nat}
    (slopes : Fin H → ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealAlibi 0 (slopes h)
      (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

noncomputable def attentionReal4DSlidingWindow {B H S_q S_k D : Nat}
    (window : Nat)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealSlidingWindow 0 window
      (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

noncomputable def attentionReal4DSoftcap {B H S_q S_k D : Nat}
    (softcap : ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealSoftcap softcap
      (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

@[simp] theorem attentionReal4DAlibi_slice {B H S_q S_k D : Nat}
    (slopes : Fin H → ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DAlibi slopes Q K V scale (b, h, i, d, PUnit.unit) =
      attentionRealAlibi 0 (slopes h)
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        scale (i, d, PUnit.unit) := rfl

@[simp] theorem attentionReal4DSlidingWindow_slice {B H S_q S_k D : Nat}
    (window : Nat)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DSlidingWindow window Q K V scale (b, h, i, d, PUnit.unit) =
      attentionRealSlidingWindow 0 window
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        scale (i, d, PUnit.unit) := rfl

@[simp] theorem attentionReal4DSoftcap_slice {B H S_q S_k D : Nat}
    (softcap : ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DSoftcap softcap Q K V scale (b, h, i, d, PUnit.unit) =
      attentionRealSoftcap softcap
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        scale (i, d, PUnit.unit) := rfl

end FA1Score

end VeriTile.Examples
