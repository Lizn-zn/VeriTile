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

/-! ## Full forward kernel surfaces -/

/-- FA-1 forward with ALiBi score bias inside the online-softmax loop. -/
def fa1ForwardKernelAlibi
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale slope : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))
  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)
  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    k       := tl.load($(kReg) + k_ptrs)
    v       := tl.load($(vReg) + v_ptrs)
    raw     := tl.dot(q, tl.trans(k)) * $ℝ(scale)
    delta   := tl.toReal(offs_n[None, :]) - tl.toReal(offs_m[:, None])
    dist    := tl.max(delta, 0 - delta)
    scores  := raw + (0 - $ℝ(slope) * dist)
    m_block := tl.max(scores, axis = 1)
    m_new   := tl.max(m_i, m_block)
    alpha   := tl.exp(m_i - m_new)
    p       := tl.exp(scores - m_new[:, None])
    l_new   := alpha * l_i + tl.sum(p, axis = 1)
    o_acc   := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i     := m_new
    l_i     := l_new
  }

  out    := o_acc / l_i[:, None]
  o_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + o_ptrs, out)
}

/-- FA-1 forward with a symmetric sliding-window score mask. -/
def fa1ForwardKernelSlidingWindow
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks window : Nat) (scale : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))
  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)
  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    k       := tl.load($(kReg) + k_ptrs)
    v       := tl.load($(vReg) + v_ptrs)
    raw     := tl.dot(q, tl.trans(k)) * $ℝ(scale)
    delta   := tl.toReal(offs_n[None, :]) - tl.toReal(offs_m[:, None])
    dist    := tl.max(delta, 0 - delta)
    visible := dist < $ℝ((window : ℝ))
    scores  := tl.where(visible, raw, -inf)
    m_block := tl.max(scores, axis = 1)
    m_new   := tl.max(m_i, m_block)
    alpha   := tl.exp(m_i - m_new)
    p       := tl.exp(scores - m_new[:, None])
    l_new   := alpha * l_i + tl.sum(p, axis = 1)
    o_acc   := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i     := m_new
    l_i     := l_new
  }

  out    := o_acc / l_i[:, None]
  o_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + o_ptrs, out)
}

/-- FA-1 forward with Gemma-style softcap applied to scores before softmax. -/
def fa1ForwardKernelSoftcap
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale softcap : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))
  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)
  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    k       := tl.load($(kReg) + k_ptrs)
    v       := tl.load($(vReg) + v_ptrs)
    raw     := tl.dot(q, tl.trans(k)) * $ℝ(scale)
    scores  := $ℝ(softcap) * tl.tanh(raw / $ℝ(softcap))
    m_block := tl.max(scores, axis = 1)
    m_new   := tl.max(m_i, m_block)
    alpha   := tl.exp(m_i - m_new)
    p       := tl.exp(scores - m_new[:, None])
    l_new   := alpha * l_i + tl.sum(p, axis = 1)
    o_acc   := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i     := m_new
    l_i     := l_new
  }

  out    := o_acc / l_i[:, None]
  o_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + o_ptrs, out)
}

/-- FA-1 forward with ALiBi, softcap, and sliding-window masking composed. -/
def fa1ForwardKernelAlibiSlidingSoftcap
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks window : Nat) (scale slope softcap : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))
  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)
  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    k       := tl.load($(kReg) + k_ptrs)
    v       := tl.load($(vReg) + v_ptrs)
    raw     := tl.dot(q, tl.trans(k)) * $ℝ(scale)
    delta   := tl.toReal(offs_n[None, :]) - tl.toReal(offs_m[:, None])
    dist    := tl.max(delta, 0 - delta)
    biased  := raw + (0 - $ℝ(slope) * dist)
    capped  := $ℝ(softcap) * tl.tanh(biased / $ℝ(softcap))
    visible := dist < $ℝ((window : ℝ))
    scores  := tl.where(visible, capped, -inf)
    m_block := tl.max(scores, axis = 1)
    m_new   := tl.max(m_i, m_block)
    alpha   := tl.exp(m_i - m_new)
    p       := tl.exp(scores - m_new[:, None])
    l_new   := alpha * l_i + tl.sum(p, axis = 1)
    o_acc   := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i     := m_new
    l_i     := l_new
  }

  out    := o_acc / l_i[:, None]
  o_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + o_ptrs, out)
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

/-! ## M-free score references

These are the final, non-online sums that online-softmax recurrences should
factor to. They deliberately mention only a score function and a visibility
predicate, so ALiBi, sliding-window masks, and softcap all share the same
ratio bridge.
-/

noncomputable def lFreeScore {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun j : Fin S =>
    if visible i j then Real.exp (score i j) else 0)

noncomputable def oFreeScore {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (idx : TileIndex [M, D]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  Finset.univ.sum (fun j : Fin S =>
    (if visible i j then Real.exp (score i j) else 0) *
      V (j, d, PUnit.unit))

def allVisible {M S : Nat} : Fin M → Fin S → Bool :=
  fun _ _ => Bool.true

theorem oFreeScore_div_lFreeScore_eq_attentionRealMaskedScore {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (idx : TileIndex [M, D]) :
    oFreeScore visible score V idx / lFreeScore visible score idx.1 =
      attentionRealMaskedScore visible score V idx := by
  obtain ⟨i, d, u⟩ := idx
  cases u
  rfl

theorem attentionRealScore_eq_masked_allVisible {M S D : Nat}
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) :
    attentionRealScore score V =
      attentionRealMaskedScore allVisible score V := by
  funext idx
  obtain ⟨i, d, u⟩ := idx
  cases u
  simp [attentionRealScore, attentionRealMaskedScore, allVisible]

theorem oFreeScore_div_lFreeScore_eq_attentionRealScore {M S D : Nat}
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (idx : TileIndex [M, D]) :
    oFreeScore allVisible score V idx / lFreeScore allVisible score idx.1 =
      attentionRealScore score V idx := by
  rw [oFreeScore_div_lFreeScore_eq_attentionRealMaskedScore]
  rw [attentionRealScore_eq_masked_allVisible]

theorem oFreeScore_div_lFreeScore_eq_attentionRealAlibi {M S D : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oFreeScore allVisible (alibiScore qStart slope Q K scale) V idx /
        lFreeScore allVisible (alibiScore qStart slope Q K scale) idx.1 =
      attentionRealAlibi qStart slope Q K V scale idx := by
  rw [oFreeScore_div_lFreeScore_eq_attentionRealScore]
  rfl

theorem oFreeScore_div_lFreeScore_eq_attentionRealSlidingWindow {M S D : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oFreeScore (slidingVisible window qStart) (dotScore Q K scale) V idx /
        lFreeScore (slidingVisible window qStart) (dotScore Q K scale) idx.1 =
      attentionRealSlidingWindow qStart window Q K V scale idx := by
  rw [oFreeScore_div_lFreeScore_eq_attentionRealMaskedScore]
  rfl

theorem oFreeScore_div_lFreeScore_eq_attentionRealSoftcap {M S D : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oFreeScore allVisible (softcapDotScore softcap Q K scale) V idx /
        lFreeScore allVisible (softcapDotScore softcap Q K scale) idx.1 =
      attentionRealSoftcap softcap Q K V scale idx := by
  rw [oFreeScore_div_lFreeScore_eq_attentionRealScore]
  rfl

theorem oFreeScore_div_lFreeScore_eq_attentionRealAlibiSlidingSoftcap {M S D : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oFreeScore (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        V idx /
      lFreeScore (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        idx.1 =
      attentionRealAlibiSlidingSoftcap qStart window slope softcap Q K V scale idx := by
  rw [oFreeScore_div_lFreeScore_eq_attentionRealMaskedScore]
  rfl

/-! ## Score-prefix recurrence

This is the score-level recurrence target shared by ALiBi, sliding-window
masks, and softcap. It intentionally abstracts over the concrete source of
the score so the final prefix ratio can be reused for all score transforms.
-/

noncomputable def lScorePrefix {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (k : Nat) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k =>
    if h : n.val < S then
      if visible i ⟨n.val, h⟩ then Real.exp (score i ⟨n.val, h⟩) else 0
    else
      0)

noncomputable def oScorePrefix {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (k : Nat) (idx : TileIndex [M, D]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  Finset.univ.sum (fun n : Fin k =>
    if h : n.val < S then
      (if visible i ⟨n.val, h⟩ then Real.exp (score i ⟨n.val, h⟩) else 0) *
        V (⟨n.val, h⟩, d, PUnit.unit)
    else
      0)

@[simp] theorem lScorePrefix_zero {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (i : Fin M) :
    lScorePrefix visible score 0 i = 0 := by
  simp [lScorePrefix]

@[simp] theorem oScorePrefix_zero {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (idx : TileIndex [M, D]) :
    oScorePrefix visible score V 0 idx = 0 := by
  simp [oScorePrefix]

theorem lScorePrefix_succ {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (k : Nat) (i : Fin M) :
    lScorePrefix visible score (k + 1) i =
      lScorePrefix visible score k i +
        (if h : k < S then
          if visible i ⟨k, h⟩ then Real.exp (score i ⟨k, h⟩) else 0
        else
          0) := by
  unfold lScorePrefix
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

theorem oScorePrefix_succ {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (k : Nat) (idx : TileIndex [M, D]) :
    oScorePrefix visible score V (k + 1) idx =
      oScorePrefix visible score V k idx +
        (if h : k < S then
          (if visible idx.1 ⟨k, h⟩ then Real.exp (score idx.1 ⟨k, h⟩) else 0) *
            V (⟨k, h⟩, idx.2.1, PUnit.unit)
        else
          0) := by
  unfold oScorePrefix
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

theorem lScorePrefix_final_eq_lFreeScore {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (i : Fin M) :
    lScorePrefix visible score S i = lFreeScore visible score i := by
  unfold lScorePrefix lFreeScore
  apply Finset.sum_congr rfl
  intro j _
  simp [j.isLt]

theorem oScorePrefix_final_eq_oFreeScore {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (idx : TileIndex [M, D]) :
    oScorePrefix visible score V S idx = oFreeScore visible score V idx := by
  unfold oScorePrefix oFreeScore
  apply Finset.sum_congr rfl
  intro j _
  simp [j.isLt]

theorem oScorePrefix_div_lScorePrefix_eq_attentionRealMaskedScore {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (idx : TileIndex [M, D]) :
    oScorePrefix visible score V S idx / lScorePrefix visible score S idx.1 =
      attentionRealMaskedScore visible score V idx := by
  rw [oScorePrefix_final_eq_oFreeScore, lScorePrefix_final_eq_lFreeScore]
  rw [oFreeScore_div_lFreeScore_eq_attentionRealMaskedScore]

theorem oScorePrefix_div_lScorePrefix_eq_attentionRealAlibi {M S D : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScorePrefix allVisible (alibiScore qStart slope Q K scale) V S idx /
        lScorePrefix allVisible (alibiScore qStart slope Q K scale) S idx.1 =
      attentionRealAlibi qStart slope Q K V scale idx := by
  rw [oScorePrefix_final_eq_oFreeScore, lScorePrefix_final_eq_lFreeScore]
  rw [oFreeScore_div_lFreeScore_eq_attentionRealAlibi]

theorem oScorePrefix_div_lScorePrefix_eq_attentionRealSlidingWindow {M S D : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScorePrefix (slidingVisible window qStart) (dotScore Q K scale) V S idx /
        lScorePrefix (slidingVisible window qStart) (dotScore Q K scale) S idx.1 =
      attentionRealSlidingWindow qStart window Q K V scale idx := by
  rw [oScorePrefix_final_eq_oFreeScore, lScorePrefix_final_eq_lFreeScore]
  rw [oFreeScore_div_lFreeScore_eq_attentionRealSlidingWindow]

theorem oScorePrefix_div_lScorePrefix_eq_attentionRealSoftcap {M S D : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScorePrefix allVisible (softcapDotScore softcap Q K scale) V S idx /
        lScorePrefix allVisible (softcapDotScore softcap Q K scale) S idx.1 =
      attentionRealSoftcap softcap Q K V scale idx := by
  rw [oScorePrefix_final_eq_oFreeScore, lScorePrefix_final_eq_lFreeScore]
  rw [oFreeScore_div_lFreeScore_eq_attentionRealSoftcap]

theorem oScorePrefix_div_lScorePrefix_eq_attentionRealAlibiSlidingSoftcap {M S D : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScorePrefix (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        V S idx /
      lScorePrefix (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        S idx.1 =
      attentionRealAlibiSlidingSoftcap qStart window slope softcap Q K V scale idx := by
  rw [oScorePrefix_final_eq_oFreeScore, lScorePrefix_final_eq_lFreeScore]
  rw [oFreeScore_div_lFreeScore_eq_attentionRealAlibiSlidingSoftcap]

/-! ## Shifted online score recurrence

These definitions mirror FA-1's numerically stable loop state for an
arbitrary score function. Masked-out keys are represented as `⊥`, so their
exponentiated contribution is zero.
-/

noncomputable def scoreLane {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (i : Fin M) (j : Fin S) : WithBot ℝ :=
  if visible i j then ((score i j : ℝ) : WithBot ℝ) else ⊥

@[simp] theorem scoreLane_of_true {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (i : Fin M) (j : Fin S)
    (h : visible i j = Bool.true) :
    scoreLane visible score i j = ((score i j : ℝ) : WithBot ℝ) := by
  simp [scoreLane, h]

@[simp] theorem scoreLane_of_false {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (i : Fin M) (j : Fin S)
    (h : visible i j = Bool.false) :
    scoreLane visible score i j = (⊥ : WithBot ℝ) := by
  simp [scoreLane, h]

theorem scoreLane_exp_shift_eq {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (i : Fin M) (j : Fin S) (m : ℝ) :
    (WithBot.realExp
      (WithBot.realSub (scoreLane visible score i j) ((m : ℝ) : WithBot ℝ))).unbotD 0 =
      Real.exp (-m) *
        (if visible i j then Real.exp (score i j) else 0) := by
  by_cases h : visible i j = Bool.true
  · rw [scoreLane_of_true visible score i j h]
    simp
    rw [show score i j - m = -m + score i j by ring, Real.exp_add]
    simp [h]
  · have hfalse : visible i j = Bool.false := by
      cases hv : visible i j
      · rfl
      · exact (h hv).elim
    rw [scoreLane_of_false visible score i j hfalse]
    simp [hfalse]

theorem alphaScore_shift_cancel (mOld mNew x : ℝ) :
    (WithBot.realExp
      (WithBot.realSub ((mOld : ℝ) : WithBot ℝ) ((mNew : ℝ) : WithBot ℝ))).unbotD 0 *
        (Real.exp (-mOld) * x) =
      Real.exp (-mNew) * x := by
  simp
  rw [show mOld - mNew = -mNew + mOld by ring, Real.exp_add]
  rw [show Real.exp (-mNew) * Real.exp mOld * (Real.exp (-mOld) * x) =
      Real.exp (-mNew) * (Real.exp mOld * Real.exp (-mOld) * x) by ring]
  rw [show Real.exp mOld * Real.exp (-mOld) = 1 by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
  ring

noncomputable def mScoreOnline {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) : Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if h : k < S then
        max (mScoreOnline visible score k i) (scoreLane visible score i ⟨k, h⟩)
      else
        mScoreOnline visible score k i

noncomputable def alphaScoreOnline {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (k : Nat) (i : Fin M) : ℝ :=
  (WithBot.realExp
    (WithBot.realSub
      (mScoreOnline visible score k i)
      (mScoreOnline visible score (k + 1) i))).unbotD 0

noncomputable def lScoreOnline {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) : Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if h : k < S then
        let mNew := mScoreOnline visible score (k + 1) i
        alphaScoreOnline visible score k i *
          lScoreOnline visible score k i +
          (WithBot.realExp
            (WithBot.realSub (scoreLane visible score i ⟨k, h⟩) mNew)).unbotD 0
      else
        lScoreOnline visible score k i

noncomputable def oScoreOnline {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) : Nat → TileIndex [M, D] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if h : k < S then
        let i := idx.1
        let d := idx.2.1
        let mNew := mScoreOnline visible score (k + 1) i
        alphaScoreOnline visible score k i *
          oScoreOnline visible score V k idx +
          (WithBot.realExp
            (WithBot.realSub (scoreLane visible score i ⟨k, h⟩) mNew)).unbotD 0 *
            V (⟨k, h⟩, d, PUnit.unit)
      else
        oScoreOnline visible score V k idx

@[simp] theorem mScoreOnline_zero {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (i : Fin M) :
    mScoreOnline visible score 0 i = (⊥ : WithBot ℝ) := rfl

@[simp] theorem lScoreOnline_zero {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (i : Fin M) :
    lScoreOnline visible score 0 i = 0 := rfl

@[simp] theorem oScoreOnline_zero {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (idx : TileIndex [M, D]) :
    oScoreOnline visible score V 0 idx = 0 := rfl

theorem mScoreOnline_succ_of_lt {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (k : Nat) (hk : k < S) (i : Fin M) :
    mScoreOnline visible score (k + 1) i =
      max (mScoreOnline visible score k i) (scoreLane visible score i ⟨k, hk⟩) := by
  change (if h : k < S then
      max (mScoreOnline visible score k i) (scoreLane visible score i ⟨k, h⟩)
    else mScoreOnline visible score k i) = _
  rw [dif_pos hk]

theorem lScoreOnline_succ_of_lt {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (k : Nat) (hk : k < S) (i : Fin M) :
    lScoreOnline visible score (k + 1) i =
      let mNew := mScoreOnline visible score (k + 1) i
      alphaScoreOnline visible score k i *
        lScoreOnline visible score k i +
        (WithBot.realExp
          (WithBot.realSub (scoreLane visible score i ⟨k, hk⟩) mNew)).unbotD 0 := by
  change (if h : k < S then
      (let mNew := mScoreOnline visible score (k + 1) i
       alphaScoreOnline visible score k i *
          lScoreOnline visible score k i +
          (WithBot.realExp
            (WithBot.realSub (scoreLane visible score i ⟨k, h⟩) mNew)).unbotD 0)
    else lScoreOnline visible score k i) = _
  rw [dif_pos hk]

theorem oScoreOnline_succ_of_lt {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (k : Nat) (hk : k < S)
    (idx : TileIndex [M, D]) :
    oScoreOnline visible score V (k + 1) idx =
      let i := idx.1
      let d := idx.2.1
      let mNew := mScoreOnline visible score (k + 1) i
      alphaScoreOnline visible score k i *
        oScoreOnline visible score V k idx +
        (WithBot.realExp
          (WithBot.realSub (scoreLane visible score i ⟨k, hk⟩) mNew)).unbotD 0 *
          V (⟨k, hk⟩, d, PUnit.unit) := by
  change (if h : k < S then
      (let i := idx.1
       let d := idx.2.1
       let mNew := mScoreOnline visible score (k + 1) i
       alphaScoreOnline visible score k i *
          oScoreOnline visible score V k idx +
          (WithBot.realExp
            (WithBot.realSub (scoreLane visible score i ⟨k, h⟩) mNew)).unbotD 0 *
            V (⟨k, h⟩, d, PUnit.unit))
    else oScoreOnline visible score V k idx) = _
  rw [dif_pos hk]

theorem lScorePrefix_eq_zero_of_mScoreOnline_eq_bot {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (k : Nat) (i : Fin M)
    (hm : mScoreOnline visible score k i = (⊥ : WithBot ℝ)) :
    lScorePrefix visible score k i = 0 := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      by_cases hk : k < S
      · have hmStep := hm
        rw [mScoreOnline_succ_of_lt visible score k hk i] at hmStep
        cases hOld : mScoreOnline visible score k i <;>
          cases hLane : scoreLane visible score i ⟨k, hk⟩ <;>
          simp [hOld, hLane] at hmStep
        rw [lScorePrefix_succ visible score k i]
        have hPrefix : lScorePrefix visible score k i = 0 := ih hOld
        have hVisible : visible i ⟨k, hk⟩ = Bool.false := by
          cases hv : visible i ⟨k, hk⟩
          · rfl
          · have hLane' :
                scoreLane visible score i ⟨k, hk⟩ =
                  ((score i ⟨k, hk⟩ : ℝ) : WithBot ℝ) := by
              simp [scoreLane, hv]
            rw [hLane'] at hLane
            exact (WithBot.coe_ne_bot hLane).elim
        simp [hPrefix, hk, hVisible]
      · have hmPrev : mScoreOnline visible score k i = (⊥ : WithBot ℝ) := by
          change (if h : k < S then
              max (mScoreOnline visible score k i) (scoreLane visible score i ⟨k, h⟩)
            else mScoreOnline visible score k i) = (⊥ : WithBot ℝ) at hm
          rw [dif_neg hk] at hm
          exact hm
        rw [lScorePrefix_succ visible score k i]
        have hPrefix : lScorePrefix visible score k i = 0 := ih hmPrev
        simp [hPrefix, hk]

theorem oScorePrefix_eq_zero_of_mScoreOnline_eq_bot {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (V : TileIndex [S, D] → ℝ)
    (k : Nat) (idx : TileIndex [M, D])
    (hm : mScoreOnline visible score k idx.1 = (⊥ : WithBot ℝ)) :
    oScorePrefix visible score V k idx = 0 := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      by_cases hk : k < S
      · have hmStep := hm
        rw [mScoreOnline_succ_of_lt visible score k hk idx.1] at hmStep
        cases hOld : mScoreOnline visible score k idx.1 <;>
          cases hLane : scoreLane visible score idx.1 ⟨k, hk⟩ <;>
          simp [hOld, hLane] at hmStep
        rw [oScorePrefix_succ visible score V k idx]
        have hPrefix : oScorePrefix visible score V k idx = 0 := ih hOld
        have hVisible : visible idx.1 ⟨k, hk⟩ = Bool.false := by
          cases hv : visible idx.1 ⟨k, hk⟩
          · rfl
          · have hLane' :
                scoreLane visible score idx.1 ⟨k, hk⟩ =
                  ((score idx.1 ⟨k, hk⟩ : ℝ) : WithBot ℝ) := by
              simp [scoreLane, hv]
            rw [hLane'] at hLane
            exact (WithBot.coe_ne_bot hLane).elim
        simp [hPrefix, hk, hVisible]
      · have hmPrev : mScoreOnline visible score k idx.1 = (⊥ : WithBot ℝ) := by
          change (if h : k < S then
              max (mScoreOnline visible score k idx.1)
                (scoreLane visible score idx.1 ⟨k, h⟩)
            else mScoreOnline visible score k idx.1) = (⊥ : WithBot ℝ) at hm
          rw [dif_neg hk] at hm
          exact hm
        rw [oScorePrefix_succ visible score V k idx]
        have hPrefix : oScorePrefix visible score V k idx = 0 := ih hmPrev
        simp [hPrefix, hk]

theorem lScoreOnline_eq_zero_of_mScoreOnline_eq_bot {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (k : Nat) (i : Fin M)
    (hm : mScoreOnline visible score k i = (⊥ : WithBot ℝ)) :
    lScoreOnline visible score k i = 0 := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      by_cases hk : k < S
      · have hmStep := hm
        rw [mScoreOnline_succ_of_lt visible score k hk i] at hmStep
        cases hOld : mScoreOnline visible score k i <;>
          cases hLane : scoreLane visible score i ⟨k, hk⟩ <;>
          simp [hOld, hLane] at hmStep
        rw [lScoreOnline_succ_of_lt visible score k hk i]
        have hOnline : lScoreOnline visible score k i = 0 := ih hOld
        simp [hOnline, hOld, hLane, alphaScoreOnline]
      · have hmPrev : mScoreOnline visible score k i = (⊥ : WithBot ℝ) := by
          change (if h : k < S then
              max (mScoreOnline visible score k i) (scoreLane visible score i ⟨k, h⟩)
            else mScoreOnline visible score k i) = (⊥ : WithBot ℝ) at hm
          rw [dif_neg hk] at hm
          exact hm
        change (if h : k < S then _ else lScoreOnline visible score k i) = 0
        rw [dif_neg hk]
        exact ih hmPrev

theorem oScoreOnline_eq_zero_of_mScoreOnline_eq_bot {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (V : TileIndex [S, D] → ℝ)
    (k : Nat) (idx : TileIndex [M, D])
    (hm : mScoreOnline visible score k idx.1 = (⊥ : WithBot ℝ)) :
    oScoreOnline visible score V k idx = 0 := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      by_cases hk : k < S
      · have hmStep := hm
        rw [mScoreOnline_succ_of_lt visible score k hk idx.1] at hmStep
        cases hOld : mScoreOnline visible score k idx.1 <;>
          cases hLane : scoreLane visible score idx.1 ⟨k, hk⟩ <;>
          simp [hOld, hLane] at hmStep
        rw [oScoreOnline_succ_of_lt visible score V k hk idx]
        have hOnline : oScoreOnline visible score V k idx = 0 := ih hOld
        simp [hOnline, hOld, hLane, alphaScoreOnline]
      · have hmPrev : mScoreOnline visible score k idx.1 = (⊥ : WithBot ℝ) := by
          change (if h : k < S then
              max (mScoreOnline visible score k idx.1)
                (scoreLane visible score idx.1 ⟨k, h⟩)
            else mScoreOnline visible score k idx.1) = (⊥ : WithBot ℝ) at hm
          rw [dif_neg hk] at hm
          exact hm
        change (if h : k < S then _ else oScoreOnline visible score V k idx) = 0
        rw [dif_neg hk]
        exact ih hmPrev

theorem lScoreOnline_eq_mShifted {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (k : Nat) (i : Fin M) :
    lScoreOnline visible score k i =
      Real.exp (-(mScoreOnline visible score k i).unbotD 0) *
        lScorePrefix visible score k i := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      by_cases hk : k < S
      · rw [lScoreOnline_succ_of_lt visible score k hk i]
        rw [lScorePrefix_succ visible score k i]
        rw [ih]
        cases hmNew : mScoreOnline visible score (k + 1) i with
        | bot =>
            have hmStep := hmNew
            rw [mScoreOnline_succ_of_lt visible score k hk i] at hmStep
            cases hmOld : mScoreOnline visible score k i <;>
              cases hLane : scoreLane visible score i ⟨k, hk⟩ <;>
              simp [hmOld, hLane] at hmStep
            have hPrefixOld :
                lScorePrefix visible score k i = 0 :=
              lScorePrefix_eq_zero_of_mScoreOnline_eq_bot visible score k i hmOld
            have hVisible : visible i ⟨k, hk⟩ = Bool.false := by
              cases hv : visible i ⟨k, hk⟩
              · rfl
              · have hLane' :
                    scoreLane visible score i ⟨k, hk⟩ =
                      ((score i ⟨k, hk⟩ : ℝ) : WithBot ℝ) := by
                  simp [scoreLane, hv]
                rw [hLane'] at hLane
                exact (WithBot.coe_ne_bot hLane).elim
            simp [alphaScoreOnline, hmOld, hPrefixOld, hk, hVisible]
        | coe mNew =>
            cases hmOld : mScoreOnline visible score k i with
            | bot =>
                have hPrefixOld :
                    lScorePrefix visible score k i = 0 :=
                  lScorePrefix_eq_zero_of_mScoreOnline_eq_bot visible score k i hmOld
                simp [alphaScoreOnline, hmOld, hmNew, hPrefixOld]
                have hShift := scoreLane_exp_shift_eq visible score i ⟨k, hk⟩ mNew
                simp only [WithBot.realSub] at hShift
                rw [hShift]
                simp [hk]
            | coe mOld =>
                simp [alphaScoreOnline, hmOld, hmNew]
                rw [show Real.exp (mOld - mNew) *
                        (Real.exp (-mOld) * lScorePrefix visible score k i) =
                      Real.exp (-mNew) * lScorePrefix visible score k i
                    from alphaScore_shift_cancel mOld mNew
                      (lScorePrefix visible score k i)]
                have hShift := scoreLane_exp_shift_eq visible score i ⟨k, hk⟩ mNew
                simp only [WithBot.realSub] at hShift
                rw [hShift]
                simp [hk]
                by_cases hVisible : visible i ⟨k, hk⟩ = Bool.true
                · simp [hVisible]
                  ring
                · have hFalse : visible i ⟨k, hk⟩ = Bool.false := by
                    cases hv : visible i ⟨k, hk⟩
                    · rfl
                    · exact (hVisible hv).elim
                  simp [hFalse]
      · change (if h : k < S then _ else lScoreOnline visible score k i) =
          Real.exp (-(mScoreOnline visible score (k + 1) i).unbotD 0) *
            lScorePrefix visible score (k + 1) i
        rw [dif_neg hk]
        rw [show mScoreOnline visible score (k + 1) i =
            mScoreOnline visible score k i by
          change (if h : k < S then
              max (mScoreOnline visible score k i) (scoreLane visible score i ⟨k, h⟩)
            else mScoreOnline visible score k i) = _
          rw [dif_neg hk]]
        rw [lScorePrefix_succ visible score k i]
        simp [hk, ih]

theorem oScoreOnline_eq_mShifted {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (k : Nat) (idx : TileIndex [M, D]) :
    oScoreOnline visible score V k idx =
      Real.exp (-(mScoreOnline visible score k idx.1).unbotD 0) *
        oScorePrefix visible score V k idx := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      by_cases hk : k < S
      · rw [oScoreOnline_succ_of_lt visible score V k hk idx]
        rw [oScorePrefix_succ visible score V k idx]
        rw [ih]
        cases hmNew : mScoreOnline visible score (k + 1) idx.1 with
        | bot =>
            have hmStep := hmNew
            rw [mScoreOnline_succ_of_lt visible score k hk idx.1] at hmStep
            cases hmOld : mScoreOnline visible score k idx.1 <;>
              cases hLane : scoreLane visible score idx.1 ⟨k, hk⟩ <;>
              simp [hmOld, hLane] at hmStep
            have hPrefixOld :
                oScorePrefix visible score V k idx = 0 :=
              oScorePrefix_eq_zero_of_mScoreOnline_eq_bot visible score V k idx hmOld
            have hVisible : visible idx.1 ⟨k, hk⟩ = Bool.false := by
              cases hv : visible idx.1 ⟨k, hk⟩
              · rfl
              · have hLane' :
                    scoreLane visible score idx.1 ⟨k, hk⟩ =
                      ((score idx.1 ⟨k, hk⟩ : ℝ) : WithBot ℝ) := by
                  simp [scoreLane, hv]
                rw [hLane'] at hLane
                exact (WithBot.coe_ne_bot hLane).elim
            simp [alphaScoreOnline, hmOld, hPrefixOld, hk, hVisible]
        | coe mNew =>
            cases hmOld : mScoreOnline visible score k idx.1 with
            | bot =>
                have hPrefixOld :
                    oScorePrefix visible score V k idx = 0 :=
                  oScorePrefix_eq_zero_of_mScoreOnline_eq_bot visible score V k idx hmOld
                simp [alphaScoreOnline, hmOld, hmNew, hPrefixOld]
                have hShift := scoreLane_exp_shift_eq visible score idx.1 ⟨k, hk⟩ mNew
                simp only [WithBot.realSub] at hShift
                rw [hShift]
                simp [hk]
                ring_nf
            | coe mOld =>
                simp [alphaScoreOnline, hmOld, hmNew]
                rw [show Real.exp (mOld - mNew) *
                        (Real.exp (-mOld) * oScorePrefix visible score V k idx) =
                      Real.exp (-mNew) * oScorePrefix visible score V k idx
                    from alphaScore_shift_cancel mOld mNew
                      (oScorePrefix visible score V k idx)]
                have hShift := scoreLane_exp_shift_eq visible score idx.1 ⟨k, hk⟩ mNew
                simp only [WithBot.realSub] at hShift
                rw [hShift]
                simp [hk]
                by_cases hVisible : visible idx.1 ⟨k, hk⟩ = Bool.true
                · simp [hVisible]
                  ring
                · have hFalse : visible idx.1 ⟨k, hk⟩ = Bool.false := by
                    cases hv : visible idx.1 ⟨k, hk⟩
                    · rfl
                    · exact (hVisible hv).elim
                  simp [hFalse]
      · change (if h : k < S then _ else oScoreOnline visible score V k idx) =
          Real.exp (-(mScoreOnline visible score (k + 1) idx.1).unbotD 0) *
            oScorePrefix visible score V (k + 1) idx
        rw [dif_neg hk]
        rw [show mScoreOnline visible score (k + 1) idx.1 =
            mScoreOnline visible score k idx.1 by
          change (if h : k < S then
              max (mScoreOnline visible score k idx.1)
                (scoreLane visible score idx.1 ⟨k, h⟩)
            else mScoreOnline visible score k idx.1) = _
          rw [dif_neg hk]]
        rw [oScorePrefix_succ visible score V k idx]
        simp [hk, ih]

theorem oScoreOnline_div_lScoreOnline_eq_attentionRealMaskedScore {M S D : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ)
    (V : TileIndex [S, D] → ℝ) (idx : TileIndex [M, D]) :
    oScoreOnline visible score V S idx / lScoreOnline visible score S idx.1 =
      attentionRealMaskedScore visible score V idx := by
  rw [oScoreOnline_eq_mShifted, lScoreOnline_eq_mShifted]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  rw [oScorePrefix_div_lScorePrefix_eq_attentionRealMaskedScore]

theorem oScoreOnline_div_lScoreOnline_eq_attentionRealAlibi {M S D : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreOnline allVisible (alibiScore qStart slope Q K scale) V S idx /
        lScoreOnline allVisible (alibiScore qStart slope Q K scale) S idx.1 =
      attentionRealAlibi qStart slope Q K V scale idx := by
  rw [oScoreOnline_eq_mShifted, lScoreOnline_eq_mShifted]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  rw [oScorePrefix_div_lScorePrefix_eq_attentionRealAlibi]

theorem oScoreOnline_div_lScoreOnline_eq_attentionRealSlidingWindow {M S D : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreOnline (slidingVisible window qStart) (dotScore Q K scale) V S idx /
        lScoreOnline (slidingVisible window qStart) (dotScore Q K scale) S idx.1 =
      attentionRealSlidingWindow qStart window Q K V scale idx := by
  rw [oScoreOnline_eq_mShifted, lScoreOnline_eq_mShifted]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  rw [oScorePrefix_div_lScorePrefix_eq_attentionRealSlidingWindow]

theorem oScoreOnline_div_lScoreOnline_eq_attentionRealSoftcap {M S D : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreOnline allVisible (softcapDotScore softcap Q K scale) V S idx /
        lScoreOnline allVisible (softcapDotScore softcap Q K scale) S idx.1 =
      attentionRealSoftcap softcap Q K V scale idx := by
  rw [oScoreOnline_eq_mShifted, lScoreOnline_eq_mShifted]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  rw [oScorePrefix_div_lScorePrefix_eq_attentionRealSoftcap]

theorem oScoreOnline_div_lScoreOnline_eq_attentionRealAlibiSlidingSoftcap {M S D : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreOnline (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        V S idx /
      lScoreOnline (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        S idx.1 =
      attentionRealAlibiSlidingSoftcap qStart window slope softcap Q K V scale idx := by
  rw [oScoreOnline_eq_mShifted, lScoreOnline_eq_mShifted]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  rw [oScorePrefix_div_lScorePrefix_eq_attentionRealAlibiSlidingSoftcap]

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

noncomputable def attentionReal4DAlibiSlidingSoftcap {B H S_q S_k D : Nat}
    (window : Nat) (slopes : Fin H → ℝ) (softcap : ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealAlibiSlidingSoftcap 0 window (slopes h) softcap
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

@[simp] theorem attentionReal4DAlibiSlidingSoftcap_slice {B H S_q S_k D : Nat}
    (window : Nat) (slopes : Fin H → ℝ) (softcap : ℝ)
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DAlibiSlidingSoftcap window slopes softcap Q K V scale
        (b, h, i, d, PUnit.unit) =
      attentionRealAlibiSlidingSoftcap 0 window (slopes h) softcap
        (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
        scale (i, d, PUnit.unit) := rfl

end FA1Score

end VeriTile.Examples
