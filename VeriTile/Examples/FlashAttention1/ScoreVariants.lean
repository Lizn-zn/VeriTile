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
    (M Bk D : Nat) (scale slope : ℝ) : ComputeKernel := triton {
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
    (M Bk D window : Nat) (scale : ℝ) : ComputeKernel := triton {
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
    (M Bk D : Nat) (scale softcap : ℝ) : ComputeKernel := triton {
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
    (M D Bk numKVBlocks : Nat) (scale slope : ℝ) : ComputeKernel := triton {
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
    (M D Bk numKVBlocks window : Nat) (scale : ℝ) : ComputeKernel := triton {
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
    (M D Bk numKVBlocks : Nat) (scale softcap : ℝ) : ComputeKernel := triton {
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
    (M D Bk numKVBlocks window : Nat) (scale slope softcap : ℝ) : ComputeKernel := triton {
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

theorem real_max_delta_eq_natAbsDiff (a b : Nat) :
    max ((b : ℝ) - (a : ℝ)) (0 - ((b : ℝ) - (a : ℝ))) =
      (natAbsDiff a b : ℝ) := by
  unfold natAbsDiff
  by_cases h : a ≤ b
  · rw [if_pos h]
    rw [Nat.cast_sub h]
    rw [show 0 - ((b : ℝ) - (a : ℝ)) = (a : ℝ) - (b : ℝ) by ring]
    have hreal : (a : ℝ) ≤ b := by exact_mod_cast h
    have hnonneg : 0 ≤ (b : ℝ) - (a : ℝ) := by nlinarith
    rw [max_eq_left]
    nlinarith [hnonneg]
  · rw [if_neg h]
    have hb : b ≤ a := Nat.le_of_lt (Nat.lt_of_not_ge h)
    rw [Nat.cast_sub hb]
    rw [show 0 - ((b : ℝ) - (a : ℝ)) = (a : ℝ) - (b : ℝ) by ring]
    have hreal : (b : ℝ) ≤ a := by exact_mod_cast hb
    have hle0 : (b : ℝ) - (a : ℝ) ≤ 0 := by nlinarith
    rw [max_eq_right]
    nlinarith [hle0]

theorem withBot_natAbsDiff_lt_window (dist window : Nat) :
    decide ((((dist : ℝ) : WithBot ℝ) < (((window : ℝ) : WithBot ℝ)))) =
      decide (dist < window) := by
  by_cases h : dist < window
  · rw [decide_eq_true h]
    rw [decide_eq_true]
    exact_mod_cast h
  · rw [decide_eq_false h]
    rw [decide_eq_false]
    intro hlt
    apply h
    exact_mod_cast hlt

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
  ring_nf

theorem softcapScore_toWithBot (softcap score : ℝ) :
    WithBot.realMul ((softcap : ℝ) : WithBot ℝ)
      (WithBot.realTanh
        (WithBot.realDiv ((score : ℝ) : WithBot ℝ) ((softcap : ℝ) : WithBot ℝ))) =
      ((softcapScore softcap score : ℝ) : WithBot ℝ) := by
  simp [WithBot.realMul, WithBot.realDiv, softcapScore]

theorem softcapScore_lane_eq {M S D : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) (j : Fin S) :
    WithBot.realMul ((softcap : ℝ) : WithBot ℝ)
      (WithBot.realTanh
        (WithBot.realDiv
          ((FA1Math.scaledScore Q K scale i j : ℝ) : WithBot ℝ)
          ((softcap : ℝ) : WithBot ℝ))) =
      scoreLane allVisible (softcapDotScore softcap Q K scale) i j := by
  rw [softcapScore_toWithBot]
  simp [scoreLane, allVisible, softcapDotScore, dotScore]

theorem alibiBias_toWithBot (slope : ℝ) (qPos kPos : Nat) :
    WithBot.realMul
        (WithBot.realSub ((0 : ℝ) : WithBot ℝ) ((slope : ℝ) : WithBot ℝ))
        (((natAbsDiff qPos kPos : ℝ) : WithBot ℝ)) =
      ((alibiBias slope qPos kPos : ℝ) : WithBot ℝ) := by
  change (((0 - slope) * (natAbsDiff qPos kPos : ℝ) : ℝ) : WithBot ℝ) =
    ((-slope * (natAbsDiff qPos kPos : ℝ) : ℝ) : WithBot ℝ)
  ring_nf

theorem alibiBiasSub_toWithBot (slope : ℝ) (qPos kPos : Nat) :
    WithBot.realSub ((0 : ℝ) : WithBot ℝ)
      (WithBot.realMul ((slope : ℝ) : WithBot ℝ)
        (((natAbsDiff qPos kPos : ℝ) : WithBot ℝ))) =
      ((alibiBias slope qPos kPos : ℝ) : WithBot ℝ) := by
  change (((0 - slope * (natAbsDiff qPos kPos : ℝ) : ℝ)) : WithBot ℝ) =
    ((-slope * (natAbsDiff qPos kPos : ℝ) : ℝ) : WithBot ℝ)
  ring_nf

theorem alibiScore_lane_eq {M S D : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) (j : Fin S) :
    WithBot.realAdd
      ((FA1Math.scaledScore Q K scale i j : ℝ) : WithBot ℝ)
      ((alibiBias slope (qStart + i.val) j.val : ℝ) : WithBot ℝ) =
      scoreLane allVisible (alibiScore qStart slope Q K scale) i j := by
  simp [scoreLane, allVisible, alibiScore, dotScore, WithBot.realAdd]

theorem slidingScore_lane_eq {M S D : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) (j : Fin S) :
    (if slidingVisible window qStart i j then
        ((FA1Math.scaledScore Q K scale i j : ℝ) : WithBot ℝ)
      else
        (⊥ : WithBot ℝ)) =
      scoreLane (slidingVisible window qStart) (dotScore Q K scale) i j := by
  by_cases h : slidingVisible window qStart i j = Bool.true
  · simp [scoreLane, h, dotScore]
  · have hfalse : slidingVisible window qStart i j = Bool.false := by
      cases hv : slidingVisible window qStart i j
      · rfl
      · exact (h hv).elim
    simp [scoreLane, hfalse]

theorem alibiSoftcapSlidingScore_lane_eq {M S D : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) (j : Fin S) :
    (if slidingVisible window qStart i j then
        ((softcapScore softcap (alibiScore qStart slope Q K scale i j) : ℝ) :
          WithBot ℝ)
      else
        (⊥ : WithBot ℝ)) =
      scoreLane (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        i j := by
  by_cases h : slidingVisible window qStart i j = Bool.true
  · simp [scoreLane, h]
  · have hfalse : slidingVisible window qStart i j = Bool.false := by
      cases hv : slidingVisible window qStart i j
      · rfl
      · exact (h hv).elim
    simp [scoreLane, hfalse]

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

/-! ## Generic score loop invariant

`P_fa1_score` is the operational invariant shape needed by score-transform
FA-1 kernels. It mirrors `P_fa1`, but the running accumulator slots are driven
by an explicit `score` function and `visible` predicate instead of being tied
to bare scaled dot-product scores.
-/

private def fa1ScorePreLoop (qReg : RegionName) (M D : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0)
  , Stmt.assign .nat [M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "pid")
          (Op.constNat M))
        (Op.arange M))
  , Stmt.assign .nat [D] "offs_d" (Op.arange D)
  , Stmt.assign .nat [M, D] "q_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [M, D] "q"
      (Op.load .real (MemAccess.region qReg (Op.ref .nat [M, D] "q_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M] "m_i"
      (Op.full [M] Op.negInf)
  , Stmt.assign .real [M] "l_i"
      (Op.full [M] (Op.const 0))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.full [M, D] (Op.const 0))
  ]

private def fa1ScorePostLoop (outReg : RegionName) (M D : Nat) : List Stmt :=
  [ Stmt.assign .real [M, D] "out"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [M, D] "o_acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "l_i")))
  , Stmt.assign .nat [M, D] "o_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.store .real [M, D] (MemAccess.region outReg (Op.ref .nat [M, D] "o_ptrs")) (Op.ref .real [M, D] "out") MaskOpt.none
  ]

private def fa1ScoreLoopBodySoftcap (kReg vReg : RegionName)
    (M D Bk : Nat) (scale softcap : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.mul .real Broadcast.scalarL
        (Op.const softcap)
        (Op.tanh
          (Op.div .real Broadcast.scalarR
            (Op.ref .real [M, Bk] "raw")
            (Op.const softcap))))
  , Stmt.assign .real [M] "m_block"
      (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, Bk] "scores"))
  , Stmt.assign .real [M] "m_new"
      (Op.max2 (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [M] "m_i")
        (Op.ref .real [M] "m_block"))
  , Stmt.assign .real [M] "alpha"
      (Op.exp
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_new")))
  , Stmt.assign .real [M, Bk] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [M, Bk] "scores")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))
  , Stmt.assign .real [M] "l_new"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "alpha")
          (Op.ref .real [M] "l_i"))
        (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
          (Op.ref .real [M, Bk] "p")))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
          (Op.ref .real [M, D] "o_acc"))
        (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
          (Op.ref .real [M, Bk] "p")
          (Op.ref .real [Bk, D] "v")))
  , Stmt.assign .real [M] "m_i"
      (Op.ref .real [M] "m_new")
  , Stmt.assign .real [M] "l_i"
      (Op.ref .real [M] "l_new")
  ]

private def fa1ScoreLoopBodyAlibi (kReg vReg : RegionName)
    (M D Bk : Nat) (scale slope : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "raw")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.mul .real Broadcast.scalarL
            (Op.const slope)
            (Op.ref .real [M, Bk] "dist"))))
  , Stmt.assign .real [M] "m_block"
      (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, Bk] "scores"))
  , Stmt.assign .real [M] "m_new"
      (Op.max2 (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [M] "m_i")
        (Op.ref .real [M] "m_block"))
  , Stmt.assign .real [M] "alpha"
      (Op.exp
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_new")))
  , Stmt.assign .real [M, Bk] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [M, Bk] "scores")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))
  , Stmt.assign .real [M] "l_new"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "alpha")
          (Op.ref .real [M] "l_i"))
        (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
          (Op.ref .real [M, Bk] "p")))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
          (Op.ref .real [M, D] "o_acc"))
        (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
          (Op.ref .real [M, Bk] "p")
          (Op.ref .real [Bk, D] "v")))
  , Stmt.assign .real [M] "m_i"
      (Op.ref .real [M] "m_new")
  , Stmt.assign .real [M] "l_i"
      (Op.ref .real [M] "l_new")
  ]

private def fa1ScoreLoopBodySlidingWindow (kReg vReg : RegionName)
    (M D Bk window : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .bool [M, Bk] "visible"
      (Op.lt ComparableDType.real Broadcast.scalarR
        (Op.ref .real [M, Bk] "dist")
        (Op.const (window : ℝ)))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "visible")
        (Op.ref .real [M, Bk] "raw")
        (Op.broadcast Op.negInf [M, Bk]))
  , Stmt.assign .real [M] "m_block"
      (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, Bk] "scores"))
  , Stmt.assign .real [M] "m_new"
      (Op.max2 (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [M] "m_i")
        (Op.ref .real [M] "m_block"))
  , Stmt.assign .real [M] "alpha"
      (Op.exp
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_new")))
  , Stmt.assign .real [M, Bk] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [M, Bk] "scores")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))
  , Stmt.assign .real [M] "l_new"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "alpha")
          (Op.ref .real [M] "l_i"))
        (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
          (Op.ref .real [M, Bk] "p")))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
          (Op.ref .real [M, D] "o_acc"))
        (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
          (Op.ref .real [M, Bk] "p")
          (Op.ref .real [Bk, D] "v")))
  , Stmt.assign .real [M] "m_i"
      (Op.ref .real [M] "m_new")
  , Stmt.assign .real [M] "l_i"
      (Op.ref .real [M] "l_new")
  ]

private def fa1ScoreLoopBodyAlibiSlidingSoftcap (kReg vReg : RegionName)
    (M D Bk window : Nat) (scale slope softcap : ℝ) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  , Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .real [M, Bk] "biased"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "raw")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.mul .real Broadcast.scalarL
            (Op.const slope)
            (Op.ref .real [M, Bk] "dist"))))
  , Stmt.assign .real [M, Bk] "capped"
      (Op.mul .real Broadcast.scalarL
        (Op.const softcap)
        (Op.tanh
          (Op.div .real Broadcast.scalarR
            (Op.ref .real [M, Bk] "biased")
            (Op.const softcap))))
  , Stmt.assign .bool [M, Bk] "visible"
      (Op.lt ComparableDType.real Broadcast.scalarR
        (Op.ref .real [M, Bk] "dist")
        (Op.const (window : ℝ)))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "visible")
        (Op.ref .real [M, Bk] "capped")
        (Op.broadcast Op.negInf [M, Bk]))
  , Stmt.assign .real [M] "m_block"
      (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, Bk] "scores"))
  , Stmt.assign .real [M] "m_new"
      (Op.max2 (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [M] "m_i")
        (Op.ref .real [M] "m_block"))
  , Stmt.assign .real [M] "alpha"
      (Op.exp
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_new")))
  , Stmt.assign .real [M, Bk] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [M, Bk] "scores")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))
  , Stmt.assign .real [M] "l_new"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "alpha")
          (Op.ref .real [M] "l_i"))
        (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
          (Op.ref .real [M, Bk] "p")))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
          (Op.ref .real [M, D] "o_acc"))
        (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
          (Op.ref .real [M, Bk] "p")
          (Op.ref .real [Bk, D] "v")))
  , Stmt.assign .real [M] "m_i"
      (Op.ref .real [M] "m_new")
  , Stmt.assign .real [M] "l_i"
      (Op.ref .real [M] "l_new")
  ]

private def fa1ScoreLoopLoadBlock (kReg vReg : RegionName)
    (D Bk : Nat) : List Stmt :=
  [ Stmt.assign .nat [Bk] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil
          (Op.ref .nat [] "n")
          (Op.constNat Bk))
        (Op.arange Bk))
  , Stmt.assign .nat [Bk, D] "k_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .nat [Bk, D] "v_ptrs"
      (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.mul .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [Bk] "offs_n"))
          (Op.constNat D))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] "offs_d")))
  , Stmt.assign .real [Bk, D] "k"
      (Op.load .real (MemAccess.region kReg (Op.ref .nat [Bk, D] "k_ptrs")) MaskOpt.none)
  , Stmt.assign .real [Bk, D] "v"
      (Op.load .real (MemAccess.region vReg (Op.ref .nat [Bk, D] "v_ptrs")) MaskOpt.none)
  ]

private def fa1ScoreLoopScoreSoftcap
    (M D Bk : Nat) (scale softcap : ℝ) : List Stmt :=
  [ Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.mul .real Broadcast.scalarL
        (Op.const softcap)
        (Op.tanh
          (Op.div .real Broadcast.scalarR
            (Op.ref .real [M, Bk] "raw")
            (Op.const softcap))))
  ]

private def fa1ScoreLoopScoreAlibi
    (M D Bk : Nat) (scale slope : ℝ) : List Stmt :=
  [ Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "raw")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.mul .real Broadcast.scalarL
            (Op.const slope)
            (Op.ref .real [M, Bk] "dist"))))
  ]

private def fa1ScoreLoopScoreSlidingWindow
    (M D Bk window : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .bool [M, Bk] "visible"
      (Op.lt ComparableDType.real Broadcast.scalarR
        (Op.ref .real [M, Bk] "dist")
        (Op.const (window : ℝ)))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "visible")
        (Op.ref .real [M, Bk] "raw")
        (Op.broadcast Op.negInf [M, Bk]))
  ]

private def fa1ScoreLoopScoreAlibiSlidingSoftcap
    (M D Bk window : Nat) (scale slope softcap : ℝ) : List Stmt :=
  [ Stmt.assign .real [M, Bk] "raw"
      (Op.mul .real Broadcast.scalarR
        (Op.dot (batch := []) (M := M) (K := D) (N := Bk)
          (Op.ref .real [M, D] "q")
          (Op.transpose (batch := []) (M := Bk) (N := D)
            (Op.ref .real [Bk, D] "k")))
        (Op.const scale))
  , Stmt.assign .real [M, Bk] "delta"
      (Op.sub .real (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.natToReal (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [Bk] "offs_n")))
        (Op.natToReal (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offs_m"))))
  , Stmt.assign .real [M, Bk] "dist"
      (Op.max2 (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "delta")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.ref .real [M, Bk] "delta")))
  , Stmt.assign .real [M, Bk] "biased"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, Bk] "raw")
        (Op.sub .real Broadcast.scalarL
          (Op.const 0)
          (Op.mul .real Broadcast.scalarL
            (Op.const slope)
            (Op.ref .real [M, Bk] "dist"))))
  , Stmt.assign .real [M, Bk] "capped"
      (Op.mul .real Broadcast.scalarL
        (Op.const softcap)
        (Op.tanh
          (Op.div .real Broadcast.scalarR
            (Op.ref .real [M, Bk] "biased")
            (Op.const softcap))))
  , Stmt.assign .bool [M, Bk] "visible"
      (Op.lt ComparableDType.real Broadcast.scalarR
        (Op.ref .real [M, Bk] "dist")
        (Op.const (window : ℝ)))
  , Stmt.assign .real [M, Bk] "scores"
      (Op.where
        (Op.ref .bool [M, Bk] "visible")
        (Op.ref .real [M, Bk] "capped")
        (Op.broadcast Op.negInf [M, Bk]))
  ]

private def fa1ScoreLoopTail (M D Bk : Nat) : List Stmt :=
  [ Stmt.assign .real [M] "m_block"
      (Op.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
        (Op.ref .real [M, Bk] "scores"))
  , Stmt.assign .real [M] "m_new"
      (Op.max2 (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [M] "m_i")
        (Op.ref .real [M] "m_block"))
  , Stmt.assign .real [M] "alpha"
      (Op.exp
        (Op.sub .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "m_i")
          (Op.ref .real [M] "m_new")))
  , Stmt.assign .real [M, Bk] "p"
      (Op.exp
        (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ref .real [M, Bk] "scores")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "m_new"))))
  , Stmt.assign .real [M] "l_new"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [M] "alpha")
          (Op.ref .real [M] "l_i"))
        (Op.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ (keepDims := Bool.false)
          (Op.ref .real [M, Bk] "p")))
  , Stmt.assign .real [M, D] "o_acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.mul .real (Broadcast.consSame (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [M] "alpha"))
          (Op.ref .real [M, D] "o_acc"))
        (Op.dot (batch := []) (M := M) (K := Bk) (N := D)
          (Op.ref .real [M, Bk] "p")
          (Op.ref .real [Bk, D] "v")))
  , Stmt.assign .real [M] "m_i"
      (Op.ref .real [M] "m_new")
  , Stmt.assign .real [M] "l_i"
      (Op.ref .real [M] "l_new")
  ]

@[simp] theorem fa1ScoreLoopBodySoftcap_parts_eq
    (kReg vReg : RegionName) (M D Bk : Nat) (scale softcap : ℝ) :
    fa1ScoreLoopBodySoftcap kReg vReg M D Bk scale softcap =
      fa1ScoreLoopLoadBlock kReg vReg D Bk ++
      fa1ScoreLoopScoreSoftcap M D Bk scale softcap ++
      fa1ScoreLoopTail M D Bk := by
  rfl

@[simp] theorem fa1ScoreLoopBodyAlibi_parts_eq
    (kReg vReg : RegionName) (M D Bk : Nat) (scale slope : ℝ) :
    fa1ScoreLoopBodyAlibi kReg vReg M D Bk scale slope =
      fa1ScoreLoopLoadBlock kReg vReg D Bk ++
      fa1ScoreLoopScoreAlibi M D Bk scale slope ++
      fa1ScoreLoopTail M D Bk := by
  rfl

@[simp] theorem fa1ScoreLoopBodySlidingWindow_parts_eq
    (kReg vReg : RegionName) (M D Bk window : Nat) (scale : ℝ) :
    fa1ScoreLoopBodySlidingWindow kReg vReg M D Bk window scale =
      fa1ScoreLoopLoadBlock kReg vReg D Bk ++
      fa1ScoreLoopScoreSlidingWindow M D Bk window scale ++
      fa1ScoreLoopTail M D Bk := by
  rfl

@[simp] theorem fa1ScoreLoopBodyAlibiSlidingSoftcap_parts_eq
    (kReg vReg : RegionName) (M D Bk window : Nat)
    (scale slope softcap : ℝ) :
    fa1ScoreLoopBodyAlibiSlidingSoftcap kReg vReg M D Bk window
        scale slope softcap =
      fa1ScoreLoopLoadBlock kReg vReg D Bk ++
      fa1ScoreLoopScoreAlibiSlidingSoftcap M D Bk window scale slope softcap ++
      fa1ScoreLoopTail M D Bk := by
  rfl

@[simp] theorem fa1ForwardKernelSoftcap_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale softcap : ℝ) :
    (fa1ForwardKernelSoftcap qReg kReg vReg outReg M D Bk numKVBlocks
        scale softcap).toAlgKernel.body =
      fa1ScorePreLoop qReg M D ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1ScoreLoopBodySoftcap kReg vReg M D Bk scale softcap)] ++
      fa1ScorePostLoop outReg M D := by
  rfl

@[simp] theorem fa1ForwardKernelAlibi_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale slope : ℝ) :
    (fa1ForwardKernelAlibi qReg kReg vReg outReg M D Bk numKVBlocks
        scale slope).toAlgKernel.body =
      fa1ScorePreLoop qReg M D ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1ScoreLoopBodyAlibi kReg vReg M D Bk scale slope)] ++
      fa1ScorePostLoop outReg M D := by
  rfl

@[simp] theorem fa1ForwardKernelSlidingWindow_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks window : Nat) (scale : ℝ) :
    (fa1ForwardKernelSlidingWindow qReg kReg vReg outReg M D Bk numKVBlocks
        window scale).toAlgKernel.body =
      fa1ScorePreLoop qReg M D ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1ScoreLoopBodySlidingWindow kReg vReg M D Bk window scale)] ++
      fa1ScorePostLoop outReg M D := by
  rfl

@[simp] theorem fa1ForwardKernelAlibiSlidingSoftcap_body_eq
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks window : Nat) (scale slope softcap : ℝ) :
    (fa1ForwardKernelAlibiSlidingSoftcap qReg kReg vReg outReg M D Bk
        numKVBlocks window scale slope softcap).toAlgKernel.body =
      fa1ScorePreLoop qReg M D ++
      [Stmt.forLoop "n" numKVBlocks
        (fa1ScoreLoopBodyAlibiSlidingSoftcap kReg vReg M D Bk window
          scale slope softcap)] ++
      fa1ScorePostLoop outReg M D := by
  rfl

def P_fa1_score
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ)
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.regs .nat [] "pid" = some (Tile.scalar origPid) ∧
  s.pid = origPid ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => origPid * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        mScoreOnline visible score k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        lScoreOnline visible score k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        oScoreOnline visible score V k idx) ∧
  InputAt s qReg
      (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) Q ∧
  InputAt s kReg
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) K ∧
  InputAt s vReg
      (Offset.rowMajor2D (rows := S) (cols := D) 0 D) V

/-- Block-indexed wrapper around `P_fa1_score`.

The generic online recurrence is indexed by the number of logical key
positions already consumed. The executable FA-1 loop is indexed by KV blocks,
so after `k` loop iterations the recurrence has consumed `Bk * k` key lanes.
-/
def P_fa1_score_blocks
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (s : BlockState) : Prop :=
  P_fa1_score qReg kReg vReg origPid Q K V visible score (Bk * k) s

theorem P_fa1_score_blocks_zero
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (s : BlockState) :
    P_fa1_score_blocks qReg kReg vReg origPid Q K V visible score 0 s =
      P_fa1_score qReg kReg vReg origPid Q K V visible score 0 s := by
  simp [P_fa1_score_blocks]

theorem P_fa1_score_blocks_final
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (s : BlockState) :
    P_fa1_score_blocks qReg kReg vReg origPid Q K V visible score numKVBlocks s =
      P_fa1_score qReg kReg vReg origPid Q K V visible score (Bk * numKVBlocks) s := by
  rfl

/-! ## Block-local score views

The executable loop consumes one `Bk`-wide KV block at a time. These helpers
repackage the generic key-indexed score recurrence into the block-local tile
shape that the loop body computes.
-/

def scoreBlockIndex (Bk numKVBlocks k : Nat) (h : k + 1 ≤ numKVBlocks)
    (jLocal : Fin Bk) : Fin (Bk * numKVBlocks) :=
  FA1Math.blockIndex Bk numKVBlocks k h jLocal

noncomputable def scoreBlockLane {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) :
    Tile .real [M, Bk] :=
  ⟨fun idx : TileIndex [M, Bk] =>
    scoreLane visible score idx.1 (scoreBlockIndex Bk numKVBlocks k h idx.2.1)⟩

def visibleBlock {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) :
    Tile .bool [M, Bk] :=
  ⟨fun idx : TileIndex [M, Bk] =>
    visible idx.1 (scoreBlockIndex Bk numKVBlocks k h idx.2.1)⟩

def valueBlock {D Bk numKVBlocks : Nat}
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) :
    Tile .real [Bk, D] :=
  Tile.ofReal (fun idx : TileIndex [Bk, D] =>
    V (scoreBlockIndex Bk numKVBlocks k h idx.1, idx.2.1, PUnit.unit))

noncomputable def lScoreBlockFree {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k =>
    Finset.univ.sum (fun jLocal : Fin Bk =>
      let j := scoreBlockIndex Bk numKVBlocks n.val
        (Nat.lt_of_lt_of_le n.isLt hk) jLocal
      if visible i j then Real.exp (score i j) else 0))

noncomputable def oScoreBlockFree {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) : ℝ :=
  Finset.univ.sum (fun n : Fin k =>
    Finset.univ.sum (fun jLocal : Fin Bk =>
      let j := scoreBlockIndex Bk numKVBlocks n.val
        (Nat.lt_of_lt_of_le n.isLt hk) jLocal
      (if visible idx.1 j then Real.exp (score idx.1 j) else 0) *
        V (j, idx.2.1, PUnit.unit)))

@[simp] theorem lScoreBlockFree_zero {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (i : Fin M) :
    lScoreBlockFree visible score 0 (Nat.zero_le _) i = 0 := by
  simp [lScoreBlockFree]

@[simp] theorem oScoreBlockFree_zero {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (idx : TileIndex [M, D]) :
    oScoreBlockFree visible score V 0 (Nat.zero_le _) idx = 0 := by
  simp [oScoreBlockFree]

theorem lScoreBlockFree_succ {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    lScoreBlockFree visible score (k + 1) hk i =
      lScoreBlockFree visible score k (Nat.le_of_succ_le hk) i +
      Finset.univ.sum (fun jLocal : Fin Bk =>
        let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
        if visible i j then Real.exp (score i j) else 0) := by
  unfold lScoreBlockFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

theorem oScoreBlockFree_succ {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oScoreBlockFree visible score V (k + 1) hk idx =
      oScoreBlockFree visible score V k (Nat.le_of_succ_le hk) idx +
      Finset.univ.sum (fun jLocal : Fin Bk =>
        let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
        (if visible idx.1 j then Real.exp (score idx.1 j) else 0) *
          V (j, idx.2.1, PUnit.unit)) := by
  unfold oScoreBlockFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

theorem lScoreBlockFree_final_eq_lFreeScore {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (i : Fin M) :
    lScoreBlockFree visible score numKVBlocks (le_refl _) i =
      lFreeScore visible score i := by
  unfold lScoreBlockFree lFreeScore
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j _
    simp [scoreBlockIndex, FA1Math.blockIndex_blockIndexEquiv]

theorem oScoreBlockFree_final_eq_oFreeScore {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (idx : TileIndex [M, D]) :
    oScoreBlockFree visible score V numKVBlocks (le_refl _) idx =
      oFreeScore visible score V idx := by
  unfold oScoreBlockFree oFreeScore
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j _
    simp [scoreBlockIndex, FA1Math.blockIndex_blockIndexEquiv]

theorem scoreBlock_exp_shift_sum_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) (m : ℝ) :
    (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
      (WithBot.realExp
        (WithBot.realSub
          (scoreLane visible score i
            (scoreBlockIndex Bk numKVBlocks k hk jLocal))
          ((m : ℝ) : WithBot ℝ))).unbotD 0) =
      Real.exp (-m) *
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
          if visible i j then Real.exp (score i j) else 0) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro jLocal _
  have hShift :=
    scoreLane_exp_shift_eq visible score i
      (scoreBlockIndex Bk numKVBlocks k hk jLocal) m
  simpa [WithBot.realSub] using hShift

theorem scoreBlock_exp_shift_v_sum_eq {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (idx : TileIndex [M, D]) (m : ℝ) :
    (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
      let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
      (WithBot.realExp
        (WithBot.realSub
          (scoreLane visible score idx.1 j)
          ((m : ℝ) : WithBot ℝ))).unbotD 0 *
        V (j, idx.2.1, PUnit.unit)) =
      Real.exp (-m) *
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
          (if visible idx.1 j then Real.exp (score idx.1 j) else 0) *
            V (j, idx.2.1, PUnit.unit)) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro jLocal _
  let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
  have hShift :=
    scoreLane_exp_shift_eq visible score idx.1 j m
  have hShift' :
      (WithBot.realExp
        (WithBot.realSub
          (scoreLane visible score idx.1 j)
          ((m : ℝ) : WithBot ℝ))).unbotD 0 =
        Real.exp (-m) *
          if visible idx.1 j then
            Real.exp (score idx.1 j)
          else 0 := by
    simpa [WithBot.realSub] using hShift
  change
    (WithBot.realExp
      (WithBot.realSub (scoreLane visible score idx.1 j)
        ((m : ℝ) : WithBot ℝ))).unbotD 0 *
        V (j, idx.2.1, PUnit.unit) =
      Real.exp (-m) *
        ((if visible idx.1 j then Real.exp (score idx.1 j) else 0) *
          V (j, idx.2.1, PUnit.unit))
  rw [hShift']
  by_cases hVisible : visible idx.1 j = Bool.true
  · simp [hVisible]
    ring
  · have hFalse : visible idx.1 j = Bool.false := by
      cases hv : visible idx.1 j
      · rfl
      · exact (hVisible hv).elim
    simp [hFalse]

theorem scoreBlock_visible_false_of_sup_eq_bot {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M)
    (hSup :
      ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
        scoreLane visible score i
          (scoreBlockIndex Bk numKVBlocks k hk jLocal)) = (⊥ : WithBot ℝ))
    (jLocal : Fin Bk) :
    visible i (scoreBlockIndex Bk numKVBlocks k hk jLocal) = Bool.false := by
  have hAll :
      ∀ j ∈ (Finset.univ : Finset (Fin Bk)),
        scoreLane visible score i
          (scoreBlockIndex Bk numKVBlocks k hk j) = (⊥ : WithBot ℝ) := by
    simpa [Finset.sup_eq_bot_iff] using hSup
  have hLane := hAll jLocal (Finset.mem_univ _)
  cases hv : visible i (scoreBlockIndex Bk numKVBlocks k hk jLocal)
  · rfl
  · have hLane' :
        scoreLane visible score i
            (scoreBlockIndex Bk numKVBlocks k hk jLocal) =
          ((score i (scoreBlockIndex Bk numKVBlocks k hk jLocal) : ℝ) :
            WithBot ℝ) := by
      simp [scoreLane, hv]
    rw [hLane'] at hLane
    exact (WithBot.coe_ne_bot hLane).elim

@[simp] theorem scoreBlockLane_data {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) (i : Fin M) (j : Fin Bk) :
    (scoreBlockLane visible score k h).data (i, j, PUnit.unit) =
      scoreLane visible score i (scoreBlockIndex Bk numKVBlocks k h j) := rfl

@[simp] theorem visibleBlock_data {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) (i : Fin M) (j : Fin Bk) :
    (visibleBlock visible k h).data (i, j, PUnit.unit) =
      visible i (scoreBlockIndex Bk numKVBlocks k h j) := rfl

@[simp] theorem valueBlock_data {D Bk numKVBlocks : Nat}
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (h : k + 1 ≤ numKVBlocks) (j : Fin Bk) (d : Fin D) :
    (valueBlock V k h).data (j, d, PUnit.unit) =
      ((V (scoreBlockIndex Bk numKVBlocks k h j, d, PUnit.unit) : ℝ) :
        WithBot ℝ) := rfl

theorem fa1_score_block_read
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (k : Nat) (hk : k < numKVBlocks)
    (j : Fin Bk) (d : Fin D) :
    s.readMem region ((k * Bk + j.val) * D + d.val) =
      X (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j,
        d, PUnit.unit) := by
  rw [BlockState.readMem]
  have haddr :
      (k * Bk + j.val) * D + d.val =
        Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D
          (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j,
            d, PUnit.unit) := by
    simp [Offset.rowMajor2D, Offset.strided, scoreBlockIndex, FA1Math.blockIndex]
  rw [haddr]
  exact hX _

theorem fa1_score_block_load_tile_eq
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (k : Nat) (hk : k < numKVBlocks) :
    (⟨fun idx : TileIndex [Bk, D] =>
        some (s.readMem region ((k * Bk + idx.1.val) * D + idx.2.1.val))⟩
      : Tile .real [Bk, D])
      =
      valueBlock X k (Nat.succ_le_iff.mpr hk) := by
  ext idx
  rw [valueBlock_data]
  exact congrArg some (fa1_score_block_read region s X hX k hk idx.1 idx.2.1)

theorem fa1_score_block_mem_load_tile_eq
    {D Bk numKVBlocks : Nat}
    (region : RegionName) (s : BlockState)
    (X : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (hX : InputAt s region
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) X)
    (k : Nat) (hk : k < numKVBlocks) :
    (⟨fun idx : TileIndex [Bk, D] =>
        some (s.readMem region ((k * Bk + idx.1.val) * D + idx.2.1.val))⟩
      : Tile .real [Bk, D])
      =
      valueBlock X k (Nat.succ_le_iff.mpr hk) := by
  ext idx
  rw [valueBlock_data]
  exact congrArg some (by
    simpa [BlockState.readMem] using
      fa1_score_block_read region s X hX k hk idx.1 idx.2.1)

noncomputable def rawScoreBlockTile {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (h : k + 1 ≤ numKVBlocks) :
    Tile .real [M, Bk] :=
  Tile.bop NumericDType.real.mul Broadcast.scalarR
    (Tile.dot [] (Tile.ofReal Q)
      (Tile.transpose [] (Tile.ofReal
        (fun idx : TileIndex [Bk, D] =>
          K (scoreBlockIndex Bk numKVBlocks k h idx.1, idx.2.1, PUnit.unit)))))
    (Tile.scalar ((scale : ℝ) : WithBot ℝ))

def distanceBlockTile {M Bk : Nat} (qStart : Nat) (k : Nat) :
    Tile .real [M, Bk] :=
  Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
      (Tile.natToReal
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.natToReal
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => qStart + i.val)))))
    (Tile.bop WithBot.realSub Broadcast.scalarL
      (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
      (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Tile.natToReal
          (Tile.expandDim ⟨0, by simp⟩
            (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
        (Tile.natToReal
          (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => qStart + i.val))))))

def distanceBlockTileNumeric {M Bk : Nat} (qStart : Nat) (k : Nat) :
    Tile .real [M, Bk] :=
  Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (Tile.bop NumericDType.real.sub (Broadcast.consL (Broadcast.consR Broadcast.nil))
      (Tile.natToReal
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.natToReal
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => qStart + i.val)))))
    (Tile.bop NumericDType.real.sub Broadcast.scalarL
      (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
      (Tile.bop NumericDType.real.sub (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Tile.natToReal
          (Tile.expandDim ⟨0, by simp⟩
            (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
        (Tile.natToReal
          (Tile.expandDim ⟨1, by simp⟩
            (Tile.vec (fun i : Fin M => qStart + i.val))))))

theorem distanceBlockTileNumeric_eq {M Bk : Nat} (qStart : Nat) (k : Nat) :
    distanceBlockTileNumeric (M := M) (Bk := Bk) qStart k =
      distanceBlockTile (M := M) (Bk := Bk) qStart k := by
  rfl

noncomputable def slidingVisibleBlockTile {M Bk : Nat}
    (qStart window : Nat) (k : Nat) : Tile .bool [M, Bk] :=
  Tile.cop ComparableDType.real.lt Broadcast.scalarR
    (distanceBlockTile (M := M) (Bk := Bk) qStart k)
    (Tile.scalar (((window : ℝ) : WithBot ℝ)))

noncomputable def alibiScoreBlockTile {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (h : k + 1 ≤ numKVBlocks) :
    Tile .real [M, Bk] :=
  Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
    (rawScoreBlockTile Q K scale k h)
    (Tile.bop WithBot.realSub Broadcast.scalarL
      (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
      (Tile.bop WithBot.realMul Broadcast.scalarL
        (Tile.scalar ((slope : ℝ) : WithBot ℝ))
        (distanceBlockTile (M := M) (Bk := Bk) qStart k)))

noncomputable def softcapScoreTile {M Bk : Nat}
    (softcap : ℝ) (scores : Tile .real [M, Bk]) : Tile .real [M, Bk] :=
  Tile.bop WithBot.realMul Broadcast.scalarL
    (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
    (Tile.uop WithBot.realTanh
      (Tile.bop WithBot.realDiv Broadcast.scalarR
        scores
        (Tile.scalar ((softcap : ℝ) : WithBot ℝ))))

theorem softcapScoreTileNumeric_eq {M Bk : Nat}
    (softcap : ℝ) (scores : Tile .real [M, Bk]) :
    Tile.bop NumericDType.real.mul Broadcast.scalarL
      (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
      (Tile.uop WithBot.realTanh
        (Tile.bop NumericDType.real.div Broadcast.scalarR
          scores
          (Tile.scalar ((softcap : ℝ) : WithBot ℝ)))) =
      softcapScoreTile softcap scores := by
  rfl

def maskedScoreTile {M Bk : Nat}
    (visible : Tile .bool [M, Bk]) (scores : Tile .real [M, Bk]) :
    Tile .real [M, Bk] :=
  Tile.select visible scores
    (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])

/-! ## Block-partial score recurrences

These mirror the executable FA-1 loop's block granularity while keeping the
score and visibility predicates generic. They are the math bridge needed by
the loop-step proof: one loop iteration consumes exactly one `Bk`-wide score
block.
-/

noncomputable def mScoreBlockPartial {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        max (mScoreBlockPartial visible score k i)
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            scoreLane visible score i (scoreBlockIndex Bk numKVBlocks k h jLocal))
      else
        mScoreBlockPartial visible score k i

noncomputable def alphaScoreBlockPartial {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  (WithBot.realExp
    (WithBot.realSub
      (mScoreBlockPartial visible score k i)
      (mScoreBlockPartial visible score (k + 1) i))).unbotD 0

noncomputable def lScoreBlockPartial {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) :
    Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        alphaScoreBlockPartial visible score k i *
          lScoreBlockPartial visible score k i +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score i (scoreBlockIndex Bk numKVBlocks k h jLocal))
              (mScoreBlockPartial visible score (k + 1) i))).unbotD 0)
      else
        lScoreBlockPartial visible score k i

noncomputable def oScoreBlockPartial {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ) :
    Nat → TileIndex [M, D] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if h : k + 1 ≤ numKVBlocks then
        alphaScoreBlockPartial visible score k idx.1 *
          oScoreBlockPartial visible score V k idx +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          let j := scoreBlockIndex Bk numKVBlocks k h jLocal
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1 j)
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit))
      else
        oScoreBlockPartial visible score V k idx

theorem mScoreBlockPartial_succ_of_lt {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    mScoreBlockPartial visible score (k + 1) i =
      max (mScoreBlockPartial visible score k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          scoreLane visible score i
            (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal)) := by
  conv_lhs => rw [mScoreBlockPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem mScoreBlockPartial_succ {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    mScoreBlockPartial visible score (k + 1) i =
      max (mScoreBlockPartial visible score k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          scoreLane visible score i
            (scoreBlockIndex Bk numKVBlocks k hk jLocal)) := by
  conv_lhs => rw [mScoreBlockPartial]
  rw [dif_pos hk]

theorem lScoreBlockPartial_succ_of_lt {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    lScoreBlockPartial visible score (k + 1) i =
      alphaScoreBlockPartial visible score k i *
        lScoreBlockPartial visible score k i +
      (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score i
              (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal))
            (mScoreBlockPartial visible score (k + 1) i))).unbotD 0) := by
  conv_lhs => rw [lScoreBlockPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem lScoreBlockPartial_succ {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    lScoreBlockPartial visible score (k + 1) i =
      alphaScoreBlockPartial visible score k i *
        lScoreBlockPartial visible score k i +
      (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score i
              (scoreBlockIndex Bk numKVBlocks k hk jLocal))
            (mScoreBlockPartial visible score (k + 1) i))).unbotD 0) := by
  conv_lhs => rw [lScoreBlockPartial]
  rw [dif_pos hk]

theorem oScoreBlockPartial_succ_of_lt {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
    oScoreBlockPartial visible score V (k + 1) idx =
      alphaScoreBlockPartial visible score k idx.1 *
        oScoreBlockPartial visible score V k idx +
      (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
        let j := scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score idx.1 j)
            (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
          V (j, idx.2.1, PUnit.unit)) := by
  conv_lhs => rw [oScoreBlockPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem oScoreBlockPartial_succ {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oScoreBlockPartial visible score V (k + 1) idx =
      alphaScoreBlockPartial visible score k idx.1 *
        oScoreBlockPartial visible score V k idx +
      (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
        let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score idx.1 j)
            (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
          V (j, idx.2.1, PUnit.unit)) := by
  conv_lhs => rw [oScoreBlockPartial]
  rw [dif_pos hk]

theorem lScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M)
    (hm : mScoreBlockPartial visible score k i = (⊥ : WithBot ℝ)) :
    lScoreBlockFree visible score k hk i = 0 := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      have hkPrev : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      have hkLt : k < numKVBlocks := Nat.lt_of_succ_le hk
      have hmStep := hm
      rw [mScoreBlockPartial_succ visible score k hk i] at hmStep
      cases hOld : mScoreBlockPartial visible score k i <;>
        cases hSup :
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            scoreLane visible score i
              (scoreBlockIndex Bk numKVBlocks k hk jLocal)) <;>
        simp [hOld, hSup] at hmStep
      rw [lScoreBlockFree_succ visible score k hk i]
      have hFreeOld :
          lScoreBlockFree visible score k hkPrev i = 0 := ih hkPrev hOld
      have hBlock :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
            if visible i j then Real.exp (score i j) else 0) = 0 := by
        apply Finset.sum_eq_zero
        intro jLocal _
        have hFalse :=
          scoreBlock_visible_false_of_sup_eq_bot visible score k hk i hSup jLocal
        simp [hFalse]
      simp [hFreeOld, hBlock]

theorem oScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D])
    (hm : mScoreBlockPartial visible score k idx.1 = (⊥ : WithBot ℝ)) :
    oScoreBlockFree visible score V k hk idx = 0 := by
  induction k with
  | zero =>
      simp
  | succ k ih =>
      have hkPrev : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      have hkLt : k < numKVBlocks := Nat.lt_of_succ_le hk
      have hmStep := hm
      rw [mScoreBlockPartial_succ visible score k hk idx.1] at hmStep
      cases hOld : mScoreBlockPartial visible score k idx.1 <;>
        cases hSup :
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            scoreLane visible score idx.1
              (scoreBlockIndex Bk numKVBlocks k hk jLocal)) <;>
        simp [hOld, hSup] at hmStep
      rw [oScoreBlockFree_succ visible score V k hk idx]
      have hFreeOld :
          oScoreBlockFree visible score V k hkPrev idx = 0 := ih hkPrev hOld
      have hBlock :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
            if visible idx.1 j then
              Real.exp (score idx.1 j) * V (j, idx.2.1, PUnit.unit)
            else
              0) = 0 := by
        apply Finset.sum_eq_zero
        intro jLocal _
        have hFalse :=
          scoreBlock_visible_false_of_sup_eq_bot visible score k hk idx.1 hSup jLocal
        simp [hFalse]
      simp [hFreeOld, hBlock]

theorem lScoreBlockPartial_eq_mShifted {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) :
    lScoreBlockPartial visible score k i =
      Real.exp (-(mScoreBlockPartial visible score k i).unbotD 0) *
        lScoreBlockFree visible score k hk i := by
  induction k with
  | zero =>
      rw [lScoreBlockFree_zero]
      simp [lScoreBlockPartial]
  | succ k ih =>
      have hkPrev : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      have hkLt : k < numKVBlocks := Nat.lt_of_succ_le hk
      rw [lScoreBlockPartial_succ visible score k hk i]
      rw [lScoreBlockFree_succ visible score k hk i]
      rw [ih hkPrev]
      cases hmNew : mScoreBlockPartial visible score (k + 1) i with
      | bot =>
          have hmStep := hmNew
          rw [mScoreBlockPartial_succ visible score k hk i] at hmStep
          cases hmOld : mScoreBlockPartial visible score k i <;>
            cases hSup :
              ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
                scoreLane visible score i
                  (scoreBlockIndex Bk numKVBlocks k hk jLocal)) <;>
            simp [hmOld, hSup] at hmStep
          have hFreeOld :
              lScoreBlockFree visible score k hkPrev i = 0 :=
            lScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot
              visible score k hkPrev i hmOld
          have hBlockFree :
              (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
                let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
                if visible i j then Real.exp (score i j) else 0) = 0 := by
            apply Finset.sum_eq_zero
            intro jLocal _
            have hFalse :=
              scoreBlock_visible_false_of_sup_eq_bot visible score k hk i hSup jLocal
            simp [hFalse]
          have hBlockPartial :
              (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
                (WithBot.realExp
                  (WithBot.realSub
                    (scoreLane visible score i
                      (scoreBlockIndex Bk numKVBlocks k hk jLocal))
                    (mScoreBlockPartial visible score (k + 1) i))).unbotD 0) = 0 := by
            apply Finset.sum_eq_zero
            intro jLocal _
            have hFalse :=
              scoreBlock_visible_false_of_sup_eq_bot visible score k hk i hSup jLocal
            simp [hmNew, hFalse, scoreLane_of_false]
          simp [alphaScoreBlockPartial, hmOld, hmNew, hFreeOld, hBlockFree]
          simpa [WithBot.realSub, hmNew] using hBlockPartial
      | coe mNew =>
          cases hmOld : mScoreBlockPartial visible score k i with
          | bot =>
              have hFreeOld :
                  lScoreBlockFree visible score k hkPrev i = 0 :=
                lScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot
                  visible score k hkPrev i hmOld
              simp [alphaScoreBlockPartial, hmOld, hmNew, hFreeOld]
              simpa [WithBot.realSub] using
                scoreBlock_exp_shift_sum_eq visible score k hk i mNew
          | coe mOld =>
              simp [alphaScoreBlockPartial, hmOld, hmNew]
              rw [show Real.exp (mOld - mNew) *
                      (Real.exp (-mOld) *
                        lScoreBlockFree visible score k hkPrev i) =
                    Real.exp (-mNew) *
                        lScoreBlockFree visible score k hkPrev i
                  from alphaScore_shift_cancel mOld mNew
                    (lScoreBlockFree visible score k hkPrev i)]
              rw [show
                  (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                    (WithBot.realExp
                      (Option.map₂ (fun x1 x2 => x1 - x2)
                        (scoreLane visible score i
                          (scoreBlockIndex Bk numKVBlocks k hk x))
                        ((mNew : ℝ) : WithBot ℝ))).unbotD 0) =
                    Real.exp (-mNew) *
                      (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                        let j := scoreBlockIndex Bk numKVBlocks k hk x
                        if visible i j then Real.exp (score i j) else 0) by
                simpa [WithBot.realSub] using
                  scoreBlock_exp_shift_sum_eq visible score k hk i mNew]
              ring

theorem oScoreBlockPartial_eq_mShifted {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oScoreBlockPartial visible score V k idx =
      Real.exp (-(mScoreBlockPartial visible score k idx.1).unbotD 0) *
        oScoreBlockFree visible score V k hk idx := by
  induction k with
  | zero =>
      rw [oScoreBlockFree_zero]
      simp [oScoreBlockPartial]
  | succ k ih =>
      have hkPrev : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [oScoreBlockPartial_succ visible score V k hk idx]
      rw [oScoreBlockFree_succ visible score V k hk idx]
      rw [ih hkPrev]
      cases hmNew : mScoreBlockPartial visible score (k + 1) idx.1 with
      | bot =>
          have hmStep := hmNew
          rw [mScoreBlockPartial_succ visible score k hk idx.1] at hmStep
          cases hmOld : mScoreBlockPartial visible score k idx.1 <;>
            cases hSup :
              ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
                scoreLane visible score idx.1
                  (scoreBlockIndex Bk numKVBlocks k hk jLocal)) <;>
            simp [hmOld, hSup] at hmStep
          have hFreeOld :
              oScoreBlockFree visible score V k hkPrev idx = 0 :=
            oScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot
              visible score V k hkPrev idx hmOld
          have hBlockFree :
              (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
                let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
                if visible idx.1 j then
                  Real.exp (score idx.1 j) * V (j, idx.2.1, PUnit.unit)
                else
                  0) = 0 := by
            apply Finset.sum_eq_zero
            intro jLocal _
            have hFalse :=
              scoreBlock_visible_false_of_sup_eq_bot visible score k hk idx.1 hSup jLocal
            simp [hFalse]
          have hBlockPartial :
              (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
                let j := scoreBlockIndex Bk numKVBlocks k hk jLocal
                (WithBot.realExp
                  (WithBot.realSub
                    (scoreLane visible score idx.1 j)
                    (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
                  V (j, idx.2.1, PUnit.unit)) = 0 := by
            apply Finset.sum_eq_zero
            intro jLocal _
            have hFalse :=
              scoreBlock_visible_false_of_sup_eq_bot visible score k hk idx.1 hSup jLocal
            simp [hmNew, hFalse, scoreLane_of_false]
          simp [alphaScoreBlockPartial, hmOld, hmNew, hFreeOld, hBlockFree]
          simpa [WithBot.realSub, hmNew] using hBlockPartial
      | coe mNew =>
          cases hmOld : mScoreBlockPartial visible score k idx.1 with
          | bot =>
              have hFreeOld :
                  oScoreBlockFree visible score V k hkPrev idx = 0 :=
                oScoreBlockFree_eq_zero_of_mScoreBlockPartial_eq_bot
                  visible score V k hkPrev idx hmOld
              simp [alphaScoreBlockPartial, hmOld, hmNew, hFreeOld]
              simpa [WithBot.realSub] using
                scoreBlock_exp_shift_v_sum_eq visible score V k hk idx mNew
          | coe mOld =>
              simp [alphaScoreBlockPartial, hmOld, hmNew]
              rw [show Real.exp (mOld - mNew) *
                      (Real.exp (-mOld) *
                        oScoreBlockFree visible score V k hkPrev idx) =
                    Real.exp (-mNew) *
                        oScoreBlockFree visible score V k hkPrev idx
                  from alphaScore_shift_cancel mOld mNew
                    (oScoreBlockFree visible score V k hkPrev idx)]
              rw [show
                  (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                    let j := scoreBlockIndex Bk numKVBlocks k hk x
                    (WithBot.realExp
                      (Option.map₂ (fun x1 x2 => x1 - x2)
                        (scoreLane visible score idx.1 j)
                        ((mNew : ℝ) : WithBot ℝ))).unbotD 0 *
                      V (j, idx.2.1, PUnit.unit)) =
                    Real.exp (-mNew) *
                      (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                        let j := scoreBlockIndex Bk numKVBlocks k hk x
                        (if visible idx.1 j then Real.exp (score idx.1 j) else 0) *
                          V (j, idx.2.1, PUnit.unit)) by
                simpa [WithBot.realSub] using
                  scoreBlock_exp_shift_v_sum_eq visible score V k hk idx mNew]
              have hIfMul :
                  (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                    (if visible idx.1 (scoreBlockIndex Bk numKVBlocks k hk x) then
                        Real.exp (score idx.1 (scoreBlockIndex Bk numKVBlocks k hk x))
                      else 0) *
                      V (scoreBlockIndex Bk numKVBlocks k hk x, idx.2.1, PUnit.unit)) =
                  (Finset.univ : Finset (Fin Bk)).sum (fun x =>
                    if visible idx.1 (scoreBlockIndex Bk numKVBlocks k hk x) then
                      Real.exp (score idx.1 (scoreBlockIndex Bk numKVBlocks k hk x)) *
                        V (scoreBlockIndex Bk numKVBlocks k hk x, idx.2.1, PUnit.unit)
                    else 0) := by
                apply Finset.sum_congr rfl
                intro x _
                by_cases hVisible :
                    visible idx.1 (scoreBlockIndex Bk numKVBlocks k hk x) = Bool.true
                · simp [hVisible]
                · have hFalse :
                      visible idx.1 (scoreBlockIndex Bk numKVBlocks k hk x) = Bool.false := by
                    cases hv : visible idx.1 (scoreBlockIndex Bk numKVBlocks k hk x)
                    · rfl
                    · exact (hVisible hv).elim
                  simp [hFalse]
              rw [hIfMul]
              ring

theorem oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealMaskedScore
    {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (idx : TileIndex [M, D]) :
    oScoreBlockPartial visible score V numKVBlocks idx /
        lScoreBlockPartial visible score numKVBlocks idx.1 =
      attentionRealMaskedScore visible score V idx := by
  rw [oScoreBlockPartial_eq_mShifted visible score V numKVBlocks (le_refl _) idx]
  rw [lScoreBlockPartial_eq_mShifted visible score numKVBlocks (le_refl _) idx.1]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  rw [oScoreBlockFree_final_eq_oFreeScore, lScoreBlockFree_final_eq_lFreeScore]
  rw [oFreeScore_div_lFreeScore_eq_attentionRealMaskedScore]

theorem oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealAlibi
    {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreBlockPartial allVisible (alibiScore qStart slope Q K scale) V
        numKVBlocks idx /
      lScoreBlockPartial allVisible (alibiScore qStart slope Q K scale)
        numKVBlocks idx.1 =
      attentionRealAlibi qStart slope Q K V scale idx := by
  rw [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealMaskedScore]
  rfl

theorem oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealSlidingWindow
    {M D Bk numKVBlocks : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale) V
        numKVBlocks idx /
      lScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale)
        numKVBlocks idx.1 =
      attentionRealSlidingWindow qStart window Q K V scale idx := by
  rw [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealMaskedScore]
  rfl

theorem oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealSoftcap
    {M D Bk numKVBlocks : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreBlockPartial allVisible (softcapDotScore softcap Q K scale) V
        numKVBlocks idx /
      lScoreBlockPartial allVisible (softcapDotScore softcap Q K scale)
        numKVBlocks idx.1 =
      attentionRealSoftcap softcap Q K V scale idx := by
  rw [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealMaskedScore]
  rfl

theorem oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealAlibiSlidingSoftcap
    {M D Bk numKVBlocks : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oScoreBlockPartial (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        V numKVBlocks idx /
      lScoreBlockPartial (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        numKVBlocks idx.1 =
      attentionRealAlibiSlidingSoftcap qStart window slope softcap Q K V scale idx := by
  rw [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealMaskedScore]
  rfl

theorem score_block_mNew_tile_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] =>
        mScoreBlockPartial visible score k idx.1⟩ : Tile .real [M])
      (⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun j => scoreLane visible score idx.1
            (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩ :
        Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] =>
        mScoreBlockPartial visible score (k + 1) idx.1⟩ := by
  ext idx
  simp [Tile.bop]
  rw [mScoreBlockPartial_succ_of_lt visible score k hk idx.1]

theorem score_block_alpha_tile_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.uop WithBot.realExp
      (Tile.bop WithBot.realSub (Broadcast.consSame Broadcast.nil)
        (⟨fun idx : TileIndex [M] =>
          mScoreBlockPartial visible score k idx.1⟩ : Tile .real [M])
        (Tile.bop max (Broadcast.consSame Broadcast.nil)
          (⟨fun idx : TileIndex [M] =>
            mScoreBlockPartial visible score k idx.1⟩ : Tile .real [M])
          (⟨fun idx : TileIndex [M] =>
            (Finset.univ : Finset (Fin Bk)).sup
              (fun j => scoreLane visible score idx.1
                (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩ :
            Tile .real [M])))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        alphaScoreBlockPartial visible score k idx.1) := by
  ext idx
  have hmTile := score_block_mNew_tile_eq visible score k hk
  have hm := congrArg (fun t : Tile .real [M] => t.data idx) hmTile
  simp [Tile.bop] at hm
  simp [Tile.uop, Tile.bop, Tile.ofReal, alphaScoreBlockPartial]
  rw [hm]
  exact FA1MathBoundary.realExp_eq_some_unbotD _

theorem score_block_p_tile_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.uop WithBot.realExp
      (Tile.bop WithBot.realSub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (scoreBlockLane visible score k (Nat.succ_le_iff.mpr hk))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.bop max (Broadcast.consSame Broadcast.nil)
            (⟨fun idx : TileIndex [M] =>
              mScoreBlockPartial visible score k idx.1⟩ : Tile .real [M])
            (⟨fun idx : TileIndex [M] =>
              (Finset.univ : Finset (Fin Bk)).sup
                (fun j => scoreLane visible score idx.1
                  (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩ :
              Tile .real [M]))))
      =
      Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score idx.1
              (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
            (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0) := by
  ext idx
  have hmTile := score_block_mNew_tile_eq visible score k hk
  have hm := congrArg (fun t : Tile .real [M] => t.data (idx.1, PUnit.unit)) hmTile
  simp [Tile.bop] at hm
  simp [Tile.uop, Tile.bop, Tile.expandDim, Tile.ofReal,
    TileShape.dropInsertedIndex]
  rw [hm]
  exact FA1MathBoundary.realExp_eq_some_unbotD _

theorem score_block_mBlock_reduceMax_eq {M Bk numKVBlocks : Nat}
    (hBk : 0 < Bk)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      (scoreBlockLane visible score k (Nat.succ_le_iff.mpr hk))
      =
      some
        (⟨fun idx : TileIndex [M] =>
          (Finset.univ : Finset (Fin Bk)).sup
            (fun j => scoreLane visible score idx.1
              (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩ :
          Tile .real [M]) := by
  unfold Tile.reduceMax
  simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBk, scoreBlockLane]
  funext idx
  rw [Finset.sup'_eq_sup]
  apply Finset.sup_congr rfl
  intro j _
  simp [scoreBlockIndex]

theorem score_block_mBlock_sup'_tile_eq {M Bk numKVBlocks : Nat}
    (hU : (Finset.univ : Finset (Fin Bk)).Nonempty)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    (⟨fun outIdx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup' hU
        (fun j =>
          (scoreBlockLane visible score k
            (Nat.succ_le_iff.mpr hk)).data (outIdx.1, j, outIdx.2))⟩ :
      Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun j => scoreLane visible score idx.1
            (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩ := by
  ext idx
  change
    (Finset.univ : Finset (Fin Bk)).sup' hU
      (fun j =>
        (scoreBlockLane visible score k
          (Nat.succ_le_iff.mpr hk)).data (idx.1, j, idx.2))
      =
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => scoreLane visible score idx.1
          (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))
  rw [Finset.sup'_eq_sup]
  apply Finset.sup_congr rfl
  intro j _
  simp [scoreBlockLane, scoreBlockIndex]

theorem score_block_p_rowSum_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      (Tile.ofReal fun idx : TileIndex [M, Bk] =>
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score idx.1
              (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
            (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0)
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1
                (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0)) := by
  ext idx
  simp [Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, Tile.ofReal]
  rfl

theorem score_block_pv_dot_eq {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.dot [] (M := M) (K := Bk) (N := D)
      (Tile.ofReal fun idx : TileIndex [M, Bk] =>
        (WithBot.realExp
          (WithBot.realSub
            (scoreLane visible score idx.1
              (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
            (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0)
      (valueBlock V k (Nat.succ_le_iff.mpr hk))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun jLocal : Fin Bk =>
          let j := scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1 j)
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit))) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.dot, Tile.ofReal, valueBlock]

theorem rawScoreBlockTile_eq {M D Bk numKVBlocks : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk) =
      scoreBlockLane allVisible (dotScore Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [rawScoreBlockTile, Tile.bop, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul]
  unfold scoreBlockIndex
  rw [FA1Math.block_scaled_data_eq Q K scale k hk i j]
  simp [scoreBlockLane, scoreLane, dotScore, allVisible, scoreBlockIndex]

theorem score_block_lNew_tile_eq {M Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        (Tile.ofReal fun idx : TileIndex [M] =>
          alphaScoreBlockPartial visible score k idx.1)
        (Tile.ofReal fun idx : TileIndex [M] =>
          lScoreBlockPartial visible score k idx.1))
      (Tile.ofReal fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1
                (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        lScoreBlockPartial visible score (k + 1) idx.1) := by
  ext idx
  simp [Tile.bop, Tile.ofReal]
  rw [lScoreBlockPartial_succ_of_lt visible score k hk idx.1]
  simp [WithBot.realSub]

theorem score_block_oAcc_tile_eq {M D Bk numKVBlocks : Nat}
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.ofReal fun idx : TileIndex [M] =>
            alphaScoreBlockPartial visible score k idx.1))
        (Tile.ofReal fun idx : TileIndex [M, D] =>
          oScoreBlockPartial visible score V k idx))
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun jLocal : Fin Bk =>
          let j := scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1 j)
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit)))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        oScoreBlockPartial visible score V (k + 1) idx) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [oScoreBlockPartial_succ_of_lt visible score V k hk (i, d, PUnit.unit)]
  simp [WithBot.realSub]

theorem softcapScoreBlock_tile_eq {M D Bk numKVBlocks : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop WithBot.realMul Broadcast.scalarL
      (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
      (Tile.uop WithBot.realTanh
        (Tile.bop WithBot.realDiv Broadcast.scalarR
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.dot [] (Tile.ofReal Q)
              (Tile.transpose [] (Tile.ofReal
                (fun idx : TileIndex [Bk, D] =>
                  K (scoreBlockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) idx.1, idx.2.1, PUnit.unit)))))
            (Tile.scalar ((scale : ℝ) : WithBot ℝ)))
          (Tile.scalar ((softcap : ℝ) : WithBot ℝ))))
      =
      scoreBlockLane allVisible (softcapDotScore softcap Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.bop, Tile.scalar, Tile.uop, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.mul]
  unfold scoreBlockIndex
  rw [FA1Math.block_scaled_data_eq Q K scale k hk i j]
  exact softcapScore_lane_eq softcap Q K scale i
    (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)

theorem softcapScoreBlock_numeric_tile_eq {M D Bk numKVBlocks : Nat}
    (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop NumericDType.real.mul Broadcast.scalarL
      (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
      (Tile.uop WithBot.realTanh
        (Tile.bop NumericDType.real.div Broadcast.scalarR
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
          (Tile.scalar ((softcap : ℝ) : WithBot ℝ))))
      =
      scoreBlockLane allVisible (softcapDotScore softcap Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  unfold rawScoreBlockTile
  exact softcapScoreBlock_tile_eq softcap Q K scale k hk

theorem distanceBlock_tile_eq {M Bk numKVBlocks : Nat}
    (qStart : Nat) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Tile.natToReal
          (Tile.expandDim ⟨0, by simp⟩
            (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
        (Tile.natToReal
          (Tile.expandDim ⟨1, by simp⟩
            (Tile.vec (fun i : Fin M => qStart + i.val)))))
      (Tile.bop WithBot.realSub Broadcast.scalarL
        (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
        (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
          (Tile.natToReal
            (Tile.expandDim ⟨0, by simp⟩
              (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
          (Tile.natToReal
            (Tile.expandDim ⟨1, by simp⟩
              (Tile.vec (fun i : Fin M => qStart + i.val))))))
      =
      Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        (natAbsDiff (qStart + idx.1.val)
          (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1).val :
            Nat)) := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.bop, Tile.scalar, Tile.natToReal, Tile.expandDim_data,
    Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, Tile.ofReal_data]
  unfold scoreBlockIndex
  change (((max (((k * Bk + j.val : Nat) : ℝ) - ((qStart + i.val : Nat) : ℝ))
        (0 - (((k * Bk + j.val : Nat) : ℝ) - ((qStart + i.val : Nat) : ℝ))) : ℝ) :
        WithBot ℝ) =
      (((natAbsDiff (qStart + i.val) (k * Bk + j.val) : Nat) : ℝ) : WithBot ℝ))
  rw [real_max_delta_eq_natAbsDiff]

theorem slidingVisibleBlock_tile_eq {M Bk numKVBlocks : Nat}
    (qStart window : Nat) (k : Nat) (hk : k < numKVBlocks) :
    Tile.cop ComparableDType.real.lt Broadcast.scalarR
      (Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
          (Tile.natToReal
            (Tile.expandDim ⟨0, by simp⟩
              (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
          (Tile.natToReal
            (Tile.expandDim ⟨1, by simp⟩
              (Tile.vec (fun i : Fin M => qStart + i.val)))))
        (Tile.bop WithBot.realSub Broadcast.scalarL
          (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
          (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
            (Tile.natToReal
              (Tile.expandDim ⟨0, by simp⟩
                (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
            (Tile.natToReal
              (Tile.expandDim ⟨1, by simp⟩
                (Tile.vec (fun i : Fin M => qStart + i.val)))))))
      (Tile.scalar (((window : ℝ) : WithBot ℝ)))
      =
      visibleBlock (slidingVisible window qStart) k (Nat.succ_le_iff.mpr hk) := by
  rw [distanceBlock_tile_eq qStart k hk]
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.cop, Tile.scalar, Tile.ofReal_data, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt]
  unfold visibleBlock slidingVisible scoreBlockIndex
  simp [FA1Math.blockIndex]
  constructor
  · intro hlt
    change ((((natAbsDiff (qStart + i.val) (k * Bk + j.val) : Nat) : ℝ) :
      WithBot ℝ) < (((window : Nat) : ℝ) : WithBot ℝ)) at hlt
    rw [WithBot.coe_lt_coe] at hlt
    exact_mod_cast hlt
  · intro hlt
    change ((((natAbsDiff (qStart + i.val) (k * Bk + j.val) : Nat) : ℝ) :
      WithBot ℝ) < (((window : Nat) : ℝ) : WithBot ℝ))
    rw [WithBot.coe_lt_coe]
    exact_mod_cast hlt

theorem alibiScoreBlock_tile_eq {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal
            (fun idx : TileIndex [Bk, D] =>
              K (scoreBlockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) idx.1, idx.2.1, PUnit.unit)))))
        (Tile.scalar ((scale : ℝ) : WithBot ℝ)))
      (Tile.bop WithBot.realSub Broadcast.scalarL
        (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
        (Tile.bop WithBot.realMul Broadcast.scalarL
          (Tile.scalar ((slope : ℝ) : WithBot ℝ))
          (Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
              (Tile.natToReal
                (Tile.expandDim ⟨0, by simp⟩
                  (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
              (Tile.natToReal
                (Tile.expandDim ⟨1, by simp⟩
                  (Tile.vec (fun i : Fin M => qStart + i.val)))))
            (Tile.bop WithBot.realSub Broadcast.scalarL
              (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
              (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
                (Tile.natToReal
                  (Tile.expandDim ⟨0, by simp⟩
                    (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
                (Tile.natToReal
                  (Tile.expandDim ⟨1, by simp⟩
                    (Tile.vec (fun i : Fin M => qStart + i.val)))))))))
      =
      scoreBlockLane allVisible (alibiScore qStart slope Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.bop, Tile.scalar, Tile.natToReal, Tile.expandDim_data,
    Tile.vec_data, Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex, NumericDType.mul]
  unfold scoreBlockIndex
  rw [FA1Math.block_scaled_data_eq Q K scale k hk i j]
  have hdist :
      max (WithBot.realSub (some (((k * Bk + j.val : Nat) : ℝ)))
            (some (((qStart + i.val : Nat) : ℝ))))
          (WithBot.realSub (((0 : ℝ) : WithBot ℝ))
            (WithBot.realSub (some (((k * Bk + j.val : Nat) : ℝ)))
              (some (((qStart + i.val : Nat) : ℝ))))) =
        (((natAbsDiff (qStart + i.val) (k * Bk + j.val) : Nat) : ℝ) :
          WithBot ℝ) := by
    change (((max (((k * Bk + j.val : Nat) : ℝ) - ((qStart + i.val : Nat) : ℝ))
        (0 - (((k * Bk + j.val : Nat) : ℝ) - ((qStart + i.val : Nat) : ℝ))) : ℝ) :
        WithBot ℝ) =
      (((natAbsDiff (qStart + i.val) (k * Bk + j.val) : Nat) : ℝ) :
        WithBot ℝ))
    rw [real_max_delta_eq_natAbsDiff]
  rw [hdist]
  rw [alibiBiasSub_toWithBot slope (qStart + i.val) (k * Bk + j.val)]
  simpa [scoreBlockLane, scoreBlockIndex, FA1Math.blockIndex] using
    alibiScore_lane_eq qStart slope Q K scale i
      (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j)

theorem slidingScoreBlock_tile_eq {M D Bk numKVBlocks : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.select
      (Tile.cop ComparableDType.real.lt Broadcast.scalarR
        (Tile.bop max (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
            (Tile.natToReal
              (Tile.expandDim ⟨0, by simp⟩
                (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
            (Tile.natToReal
              (Tile.expandDim ⟨1, by simp⟩
                (Tile.vec (fun i : Fin M => qStart + i.val)))))
          (Tile.bop WithBot.realSub Broadcast.scalarL
            (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
            (Tile.bop WithBot.realSub (Broadcast.consL (Broadcast.consR Broadcast.nil))
              (Tile.natToReal
                (Tile.expandDim ⟨0, by simp⟩
                  (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
              (Tile.natToReal
                (Tile.expandDim ⟨1, by simp⟩
                  (Tile.vec (fun i : Fin M => qStart + i.val)))))))
        (Tile.scalar (((window : ℝ) : WithBot ℝ))))
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal
            (fun idx : TileIndex [Bk, D] =>
              K (scoreBlockIndex Bk numKVBlocks k
                (Nat.succ_le_iff.mpr hk) idx.1, idx.2.1, PUnit.unit)))))
        (Tile.scalar ((scale : ℝ) : WithBot ℝ)))
      (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])
      =
      scoreBlockLane (slidingVisible window qStart) (dotScore Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  rw [slidingVisibleBlock_tile_eq qStart window k hk]
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.select_data, Tile.bop, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
  unfold scoreBlockIndex
  let jg := FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j
  by_cases hvis : slidingVisible window qStart i jg = Bool.true
  · simp [visibleBlock, scoreBlockIndex, jg, hvis]
    have hraw := FA1Math.block_scaled_data_eq Q K scale k hk i j
    simp only [WithBot.realMul] at hraw
    exact hraw.trans (by
      simp [scoreBlockLane, scoreBlockIndex, dotScore, hvis, jg] at ⊢
    )
  · simp [visibleBlock, scoreBlockIndex, jg, Bool.eq_false_iff.mpr hvis]
    simp [scoreBlockLane, scoreBlockIndex, Bool.eq_false_iff.mpr hvis, jg]

theorem slidingScoreBlock_numeric_tile_eq {M D Bk numKVBlocks : Nat}
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.select
      (slidingVisibleBlockTile (M := M) (Bk := Bk) qStart window k)
      (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
      (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])
      =
      scoreBlockLane (slidingVisible window qStart) (dotScore Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  unfold slidingVisibleBlockTile rawScoreBlockTile distanceBlockTile
  exact slidingScoreBlock_tile_eq qStart window Q K scale k hk

theorem alibiScoreBlockTile_eq {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    alibiScoreBlockTile qStart slope Q K scale k (Nat.succ_le_iff.mpr hk) =
      scoreBlockLane allVisible (alibiScore qStart slope Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  unfold alibiScoreBlockTile rawScoreBlockTile distanceBlockTile
  exact alibiScoreBlock_tile_eq qStart slope Q K scale k hk

theorem alibiScoreBlock_numeric_tile_eq {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
      (Tile.bop NumericDType.real.sub Broadcast.scalarL
        (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
        (Tile.bop NumericDType.real.mul Broadcast.scalarL
          (Tile.scalar ((slope : ℝ) : WithBot ℝ))
          (distanceBlockTile (M := M) (Bk := Bk) qStart k)))
      =
      scoreBlockLane allVisible (alibiScore qStart slope Q K scale) k
        (Nat.succ_le_iff.mpr hk) := by
  unfold rawScoreBlockTile distanceBlockTile
  exact alibiScoreBlock_tile_eq qStart slope Q K scale k hk

theorem alibiScoreBlock_numeric_eq_tile {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
      (Tile.bop NumericDType.real.sub Broadcast.scalarL
        (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
        (Tile.bop NumericDType.real.mul Broadcast.scalarL
          (Tile.scalar ((slope : ℝ) : WithBot ℝ))
          (distanceBlockTile (M := M) (Bk := Bk) qStart k)))
      =
      alibiScoreBlockTile qStart slope Q K scale k (Nat.succ_le_iff.mpr hk) := by
  rfl

theorem slidingVisibleBlockTile_eq {M Bk numKVBlocks : Nat}
    (qStart window : Nat) (k : Nat) (hk : k < numKVBlocks) :
    slidingVisibleBlockTile (M := M) (Bk := Bk) qStart window k =
      visibleBlock (numKVBlocks := numKVBlocks) (slidingVisible window qStart)
        k (Nat.succ_le_iff.mpr hk) := by
  unfold slidingVisibleBlockTile distanceBlockTile
  exact slidingVisibleBlock_tile_eq qStart window k hk

theorem softcapAlibiScoreBlockTile_eq {M D Bk numKVBlocks : Nat}
    (qStart : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    softcapScoreTile softcap
        (alibiScoreBlockTile qStart slope Q K scale k (Nat.succ_le_iff.mpr hk)) =
      scoreBlockLane allVisible
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        k (Nat.succ_le_iff.mpr hk) := by
  rw [alibiScoreBlockTile_eq qStart slope Q K scale k hk]
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [softcapScoreTile, Tile.bop, Tile.scalar, Tile.uop,
    Broadcast.leftIndex, Broadcast.rightIndex, WithBot.realMul_coe_coe,
    WithBot.realDiv_coe_coe, WithBot.realTanh_coe, scoreBlockLane, scoreLane,
    allVisible, softcapScore, if_true]

theorem alibiSlidingSoftcapScoreBlockTile_eq {M D Bk numKVBlocks : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    maskedScoreTile
        (slidingVisibleBlockTile (M := M) (Bk := Bk) qStart window k)
        (softcapScoreTile softcap
          (alibiScoreBlockTile qStart slope Q K scale k (Nat.succ_le_iff.mpr hk))) =
      scoreBlockLane (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        k (Nat.succ_le_iff.mpr hk) := by
  rw [slidingVisibleBlockTile_eq (M := M) (Bk := Bk) (numKVBlocks := numKVBlocks)
      qStart window k hk,
    softcapAlibiScoreBlockTile_eq qStart slope softcap Q K scale k hk]
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  let jg := scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j
  by_cases hvis : slidingVisible window qStart i jg = Bool.true
  · simp [maskedScoreTile, visibleBlock, scoreBlockLane, scoreLane, allVisible,
      jg, hvis]
  · simp [maskedScoreTile, visibleBlock, scoreBlockLane, scoreLane, allVisible, jg,
      Bool.eq_false_iff.mpr hvis]

theorem alibiSlidingSoftcapScoreBlock_numeric_tile_eq {M D Bk numKVBlocks : Nat}
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < numKVBlocks) :
    Tile.select
      (slidingVisibleBlockTile (M := M) (Bk := Bk) qStart window k)
      (Tile.bop NumericDType.real.mul Broadcast.scalarL
        (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
        (Tile.uop WithBot.realTanh
          (Tile.bop NumericDType.real.div Broadcast.scalarR
            (Tile.bop NumericDType.real.add
              (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
              (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
              (Tile.bop NumericDType.real.sub Broadcast.scalarL
                (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
                (Tile.bop NumericDType.real.mul Broadcast.scalarL
                  (Tile.scalar ((slope : ℝ) : WithBot ℝ))
                  (distanceBlockTile (M := M) (Bk := Bk) qStart k))))
            (Tile.scalar ((softcap : ℝ) : WithBot ℝ)))))
      (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])
      =
      scoreBlockLane (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        k (Nat.succ_le_iff.mpr hk) := by
  change maskedScoreTile
      (slidingVisibleBlockTile (M := M) (Bk := Bk) qStart window k)
      (softcapScoreTile softcap
        (alibiScoreBlockTile qStart slope Q K scale k (Nat.succ_le_iff.mpr hk))) =
      scoreBlockLane (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        k (Nat.succ_le_iff.mpr hk)
  exact alibiSlidingSoftcapScoreBlockTile_eq qStart window slope softcap Q K scale k hk

def P_fa1_score_blockrec
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (s : BlockState) : Prop :=
  s.regs .nat [] "pid" = some (Tile.scalar origPid) ∧
  s.pid = origPid ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => origPid * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        mScoreBlockPartial visible score k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        lScoreBlockPartial visible score k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        oScoreBlockPartial visible score V k idx) ∧
  InputAt s qReg
      (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) Q ∧
  InputAt s kReg
      (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K ∧
  InputAt s vReg
      (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V

def P_fa1_score_blockrec_loaded
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) (s : BlockState) : Prop :=
  P_fa1_score_blockrec qReg kReg vReg origPid Q K V visible score k s ∧
  s.regs .nat [Bk] "offs_n" = some
      (Tile.vec fun j : Fin Bk => k * Bk + j.val) ∧
  s.regs .nat [Bk, D] "k_ptrs" = some
      (⟨fun idx : TileIndex [Bk, D] =>
        (k * Bk + idx.1.val) * D + idx.2.1.val⟩ : Tile .nat [Bk, D]) ∧
  s.regs .nat [Bk, D] "v_ptrs" = some
      (⟨fun idx : TileIndex [Bk, D] =>
        (k * Bk + idx.1.val) * D + idx.2.1.val⟩ : Tile .nat [Bk, D]) ∧
  s.regs .real [Bk, D] "k" =
      some (valueBlock K k (Nat.succ_le_iff.mpr hk)) ∧
  s.regs .real [Bk, D] "v" =
      some (valueBlock V k (Nat.succ_le_iff.mpr hk))

def P_fa1_score_blockrec_scored
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (hk : k < numKVBlocks) (s : BlockState) : Prop :=
  P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V visible score
    k hk s ∧
  s.regs .real [M, Bk] "scores" =
    some (scoreBlockLane visible score k (Nat.succ_le_iff.mpr hk))

theorem fa1_score_loop_loadBlock_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (s : BlockState) (hk : k < numKVBlocks)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V visible score k s) :
    ∃ sLoad,
      stepStmts (fa1ScoreLoopLoadBlock kReg vReg D Bk)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some sLoad ∧
      P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V visible score
        k hk sLoad := by
  rcases hP with
    ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
  let offsN : Tile .nat [Bk] :=
    Tile.vec fun j : Fin Bk => k * Bk + j.val
  let ptrs : Tile .nat [Bk, D] :=
    ⟨fun idx : TileIndex [Bk, D] => (k * Bk + idx.1.val) * D + idx.2.1.val⟩
  let kTile : Tile .real [Bk, D] := valueBlock K k (Nat.succ_le_iff.mpr hk)
  let vTile : Tile .real [Bk, D] := valueBlock V k (Nat.succ_le_iff.mpr hk)
  let s0 := s.setReg "n" .nat [] (Tile.scalar k)
  let s1 := s0.setReg "offs_n" .nat [Bk] offsN
  let s2 := s1.setReg "k_ptrs" .nat [Bk, D] ptrs
  let s3 := s2.setReg "v_ptrs" .nat [Bk, D] ptrs
  let s4 := s3.setReg "k" .real [Bk, D] kTile
  let sLoad := s4.setReg "v" .real [Bk, D] vTile
  refine ⟨sLoad, ?_, ?_⟩
  · simp [fa1ScoreLoopLoadBlock, stepStmts, stepStmt, evalOp, Tile.bop,
      Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
      NumericDType.mul, Option.bind, hoffs_d,
      offsN, ptrs, kTile, vTile, s0, s1, s2, s3, s4, sLoad]
    rw [fa1_score_block_mem_load_tile_eq kReg s K hK k hk]
    rw [fa1_score_block_mem_load_tile_eq vReg s V hV k hk]
    simp [Tile.vec]
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
    · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · simp [sLoad, s4, s3, s2, s1, s0, hpidReg]
      · simpa [sLoad, s4, s3, s2, s1, s0] using hpid
      · simp [sLoad, s4, s3, s2, s1, s0, hoffs_m]
      · simp [sLoad, s4, s3, s2, s1, s0, hoffs_d]
      · simp [sLoad, s4, s3, s2, s1, s0, hq]
      · simp [sLoad, s4, s3, s2, s1, s0, hm]
      · simp [sLoad, s4, s3, s2, s1, s0, hl]
      · simp [sLoad, s4, s3, s2, s1, s0, ho]
      · intro idx
        simpa [sLoad, s4, s3, s2, s1, s0] using hQ idx
      · intro idx
        simpa [sLoad, s4, s3, s2, s1, s0] using hK idx
      · intro idx
        simpa [sLoad, s4, s3, s2, s1, s0] using hV idx
    · simp [sLoad, s4, s3, s2, s1, offsN]
    · simp [sLoad, s4, s3, s2, ptrs]
    · simp [sLoad, s4, s3, ptrs]
    · simp [sLoad, s4, kTile]
    · simp [sLoad, vTile]

theorem fa1_score_loop_scoreSoftcap_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale softcap : ℝ)
    (k : Nat) (sLoad : BlockState) (hk : k < numKVBlocks)
    (hP : P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V allVisible
      (softcapDotScore softcap Q K scale) k hk sLoad) :
    ∃ sScore,
      stepStmts (fa1ScoreLoopScoreSoftcap M D Bk scale softcap) sLoad =
        some sScore ∧
      P_fa1_score_blockrec_scored qReg kReg vReg origPid Q K V allVisible
        (softcapDotScore softcap Q K scale) k hk sScore := by
  have hLoaded := hP
  rcases hP with
    ⟨hCore, _hoffs_n, _hk_ptrs, _hv_ptrs, hkTileReg, _hvTileReg⟩
  rcases hCore with
    ⟨_hpidReg, _hpid, _hoffs_m, _hoffs_d, hq, _hm, _hl, _ho, _hQ, _hK, _hV⟩
  let raw : Tile .real [M, Bk] :=
    rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk)
  let scores : Tile .real [M, Bk] :=
    scoreBlockLane allVisible (softcapDotScore softcap Q K scale) k
      (Nat.succ_le_iff.mpr hk)
  let sRaw := sLoad.setReg "raw" .real [M, Bk] raw
  let sScore := sRaw.setReg "scores" .real [M, Bk] scores
  refine ⟨sScore, ?_, ?_⟩
  · simp [fa1ScoreLoopScoreSoftcap, stepStmts, stepStmt, evalOp, Option.bind,
      hq, hkTileReg, raw, scores, sRaw, sScore]
    simp only [valueBlock]
    change (sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "scores" .real [M, Bk]
          (Tile.bop NumericDType.real.mul Broadcast.scalarL
            (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
            (Tile.uop WithBot.realTanh
              (Tile.bop NumericDType.real.div Broadcast.scalarR
                (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
                (Tile.scalar ((softcap : ℝ) : WithBot ℝ))))) =
        (sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "scores" .real [M, Bk]
          (scoreBlockLane allVisible (softcapDotScore softcap Q K scale) k
            (Nat.succ_le_iff.mpr hk))
    rw [softcapScoreBlock_numeric_tile_eq softcap Q K scale k hk]
  · refine ⟨?_, ?_⟩
    · rcases hLoaded with
        ⟨hCore, hoffs_n, hk_ptrs, hv_ptrs, hkReg, hvReg⟩
      rcases hCore with
        ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
      · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · simp [sScore, sRaw, hpidReg]
        · simpa [sScore, sRaw] using hpid
        · simp [sScore, sRaw, hoffs_m]
        · simp [sScore, sRaw, hoffs_d]
        · simp [sScore, sRaw, hq]
        · simp [sScore, sRaw, hm]
        · simp [sScore, sRaw, hl]
        · simp [sScore, sRaw, ho]
        · intro idx
          simpa [sScore, sRaw] using hQ idx
        · intro idx
          simpa [sScore, sRaw] using hK idx
        · intro idx
          simpa [sScore, sRaw] using hV idx
      · simp [sScore, sRaw, hoffs_n]
      · simp [sScore, sRaw, hk_ptrs]
      · simp [sScore, sRaw, hv_ptrs]
      · simp [sScore, sRaw, hkReg]
      · simp [sScore, sRaw, hvReg]
    · simp [sScore, scores]

theorem fa1_score_loop_scoreAlibi_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart : Nat)
    (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (sLoad : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V allVisible
      (alibiScore qStart slope Q K scale) k hk sLoad) :
    ∃ sScore,
      stepStmts (fa1ScoreLoopScoreAlibi M D Bk scale slope) sLoad =
        some sScore ∧
      P_fa1_score_blockrec_scored qReg kReg vReg origPid Q K V allVisible
        (alibiScore qStart slope Q K scale) k hk sScore := by
  subst qStart
  have hLoaded := hP
  rcases hP with
    ⟨hCore, hoffs_n, _hk_ptrs, _hv_ptrs, hkTileReg, _hvTileReg⟩
  rcases hCore with
    ⟨_hpidReg, _hpid, hoffs_m, _hoffs_d, hq, _hm, _hl, _ho, _hQ, _hK, _hV⟩
  let raw : Tile .real [M, Bk] :=
    rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk)
  let delta : Tile .real [M, Bk] :=
    Tile.bop NumericDType.real.sub (Broadcast.consL (Broadcast.consR Broadcast.nil))
      (Tile.natToReal
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.natToReal
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => origPid * M + i.val))))
  let dist : Tile .real [M, Bk] :=
    distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k
  let scores : Tile .real [M, Bk] :=
    scoreBlockLane allVisible (alibiScore (origPid * M) slope Q K scale) k
      (Nat.succ_le_iff.mpr hk)
  let sRaw := sLoad.setReg "raw" .real [M, Bk] raw
  let sDelta := sRaw.setReg "delta" .real [M, Bk] delta
  let sDist := sDelta.setReg "dist" .real [M, Bk] dist
  let sScore := sDist.setReg "scores" .real [M, Bk] scores
  refine ⟨sScore, ?_, ?_⟩
  · simp [fa1ScoreLoopScoreAlibi, stepStmts, stepStmt, evalOp, Option.bind,
      hq, hkTileReg, hoffs_n, hoffs_m, raw, delta, dist, scores, sRaw, sDelta, sDist,
      sScore]
    simp only [valueBlock]
    change (((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "scores" .real [M, Bk]
          (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
            (Tile.bop NumericDType.real.sub Broadcast.scalarL
              (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
              (Tile.bop NumericDType.real.mul Broadcast.scalarL
                (Tile.scalar ((slope : ℝ) : WithBot ℝ))
                (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)))) =
        (((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk] (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "scores" .real [M, Bk]
          (scoreBlockLane allVisible (alibiScore (origPid * M) slope Q K scale) k
            (Nat.succ_le_iff.mpr hk))
    rw [distanceBlockTileNumeric_eq (M := M) (Bk := Bk) (origPid * M) k]
    rw [alibiScoreBlock_numeric_tile_eq (origPid * M) slope Q K scale k hk]
  · refine ⟨?_, ?_⟩
    · rcases hLoaded with
        ⟨hCore, hoffs_n, hk_ptrs, hv_ptrs, hkReg, hvReg⟩
      rcases hCore with
        ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
      · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · simp [sScore, sDist, sDelta, sRaw, hpidReg]
        · simpa [sScore, sDist, sDelta, sRaw] using hpid
        · simp [sScore, sDist, sDelta, sRaw, hoffs_m]
        · simp [sScore, sDist, sDelta, sRaw, hoffs_d]
        · simp [sScore, sDist, sDelta, sRaw, hq]
        · simp [sScore, sDist, sDelta, sRaw, hm]
        · simp [sScore, sDist, sDelta, sRaw, hl]
        · simp [sScore, sDist, sDelta, sRaw, ho]
        · intro idx
          simpa [sScore, sDist, sDelta, sRaw] using hQ idx
        · intro idx
          simpa [sScore, sDist, sDelta, sRaw] using hK idx
        · intro idx
          simpa [sScore, sDist, sDelta, sRaw] using hV idx
      · simp [sScore, sDist, sDelta, sRaw, hoffs_n]
      · simp [sScore, sDist, sDelta, sRaw, hk_ptrs]
      · simp [sScore, sDist, sDelta, sRaw, hv_ptrs]
      · simp [sScore, sDist, sDelta, sRaw, hkReg]
      · simp [sScore, sDist, sDelta, sRaw, hvReg]
    · simp [sScore, scores]

theorem fa1_score_loop_scoreSlidingWindow_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (sLoad : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V
      (slidingVisible window qStart) (dotScore Q K scale) k hk sLoad) :
    ∃ sScore,
      stepStmts (fa1ScoreLoopScoreSlidingWindow M D Bk window scale) sLoad =
        some sScore ∧
      P_fa1_score_blockrec_scored qReg kReg vReg origPid Q K V
        (slidingVisible window qStart) (dotScore Q K scale) k hk sScore := by
  subst qStart
  have hLoaded := hP
  rcases hP with
    ⟨hCore, hoffs_n, _hk_ptrs, _hv_ptrs, hkTileReg, _hvTileReg⟩
  rcases hCore with
    ⟨_hpidReg, _hpid, hoffs_m, _hoffs_d, hq, _hm, _hl, _ho, _hQ, _hK, _hV⟩
  let raw : Tile .real [M, Bk] :=
    rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk)
  let delta : Tile .real [M, Bk] :=
    Tile.bop NumericDType.real.sub (Broadcast.consL (Broadcast.consR Broadcast.nil))
      (Tile.natToReal
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.natToReal
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => origPid * M + i.val))))
  let dist : Tile .real [M, Bk] :=
    distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k
  let vis : Tile .bool [M, Bk] :=
    slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k
  let scores : Tile .real [M, Bk] :=
    scoreBlockLane (slidingVisible window (origPid * M)) (dotScore Q K scale) k
      (Nat.succ_le_iff.mpr hk)
  let sRaw := sLoad.setReg "raw" .real [M, Bk] raw
  let sDelta := sRaw.setReg "delta" .real [M, Bk] delta
  let sDist := sDelta.setReg "dist" .real [M, Bk] dist
  let sVis := sDist.setReg "visible" .bool [M, Bk] vis
  let sScore := sVis.setReg "scores" .real [M, Bk] scores
  refine ⟨sScore, ?_, ?_⟩
  · simp [fa1ScoreLoopScoreSlidingWindow, stepStmts, stepStmt, evalOp,
      Option.bind, hq, hkTileReg, hoffs_n, hoffs_m, raw, delta, dist, vis,
      scores, sRaw, sDelta, sDist, sVis, sScore]
    simp only [valueBlock]
    change ((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "visible" .bool [M, Bk]
          (Tile.cop ComparableDType.real.lt Broadcast.scalarR
            (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)
            (Tile.scalar (((window : ℝ) : WithBot ℝ))))).setReg
        "scores" .real [M, Bk]
          (Tile.select
            (Tile.cop ComparableDType.real.lt Broadcast.scalarR
              (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)
              (Tile.scalar (((window : ℝ) : WithBot ℝ))))
            (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
            (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])) =
        ((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "visible" .bool [M, Bk]
          (slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k)).setReg
        "scores" .real [M, Bk]
          (scoreBlockLane (slidingVisible window (origPid * M)) (dotScore Q K scale) k
            (Nat.succ_le_iff.mpr hk))
    rw [distanceBlockTileNumeric_eq (M := M) (Bk := Bk) (origPid * M) k]
    change ((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "visible" .bool [M, Bk]
          (slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k)).setReg
        "scores" .real [M, Bk]
          (Tile.select
            (slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k)
            (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
            (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])) =
        ((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "visible" .bool [M, Bk]
          (slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k)).setReg
        "scores" .real [M, Bk]
          (scoreBlockLane (slidingVisible window (origPid * M)) (dotScore Q K scale) k
            (Nat.succ_le_iff.mpr hk))
    rw [slidingScoreBlock_numeric_tile_eq (origPid * M) window Q K scale k hk]
  · refine ⟨?_, ?_⟩
    · rcases hLoaded with
        ⟨hCore, hoffs_n, hk_ptrs, hv_ptrs, hkReg, hvReg⟩
      rcases hCore with
        ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
      · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · simp [sScore, sVis, sDist, sDelta, sRaw, hpidReg]
        · simpa [sScore, sVis, sDist, sDelta, sRaw] using hpid
        · simp [sScore, sVis, sDist, sDelta, sRaw, hoffs_m]
        · simp [sScore, sVis, sDist, sDelta, sRaw, hoffs_d]
        · simp [sScore, sVis, sDist, sDelta, sRaw, hq]
        · simp [sScore, sVis, sDist, sDelta, sRaw, hm]
        · simp [sScore, sVis, sDist, sDelta, sRaw, hl]
        · simp [sScore, sVis, sDist, sDelta, sRaw, ho]
        · intro idx
          simpa [sScore, sVis, sDist, sDelta, sRaw] using hQ idx
        · intro idx
          simpa [sScore, sVis, sDist, sDelta, sRaw] using hK idx
        · intro idx
          simpa [sScore, sVis, sDist, sDelta, sRaw] using hV idx
      · simp [sScore, sVis, sDist, sDelta, sRaw, hoffs_n]
      · simp [sScore, sVis, sDist, sDelta, sRaw, hk_ptrs]
      · simp [sScore, sVis, sDist, sDelta, sRaw, hv_ptrs]
      · simp [sScore, sVis, sDist, sDelta, sRaw, hkReg]
      · simp [sScore, sVis, sDist, sDelta, sRaw, hvReg]
    · simp [sScore, scores]

theorem fa1_score_loop_scoreAlibiSlidingSoftcap_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat)
    (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (sLoad : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec_loaded qReg kReg vReg origPid Q K V
      (slidingVisible window qStart)
      (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
      k hk sLoad) :
    ∃ sScore,
      stepStmts
        (fa1ScoreLoopScoreAlibiSlidingSoftcap M D Bk window scale slope softcap)
        sLoad = some sScore ∧
      P_fa1_score_blockrec_scored qReg kReg vReg origPid Q K V
        (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        k hk sScore := by
  subst qStart
  have hLoaded := hP
  rcases hP with
    ⟨hCore, hoffs_n, _hk_ptrs, _hv_ptrs, hkTileReg, _hvTileReg⟩
  rcases hCore with
    ⟨_hpidReg, _hpid, hoffs_m, _hoffs_d, hq, _hm, _hl, _ho, _hQ, _hK, _hV⟩
  let raw : Tile .real [M, Bk] :=
    rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk)
  let delta : Tile .real [M, Bk] :=
    Tile.bop NumericDType.real.sub (Broadcast.consL (Broadcast.consR Broadcast.nil))
      (Tile.natToReal
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.natToReal
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.vec (fun i : Fin M => origPid * M + i.val))))
  let dist : Tile .real [M, Bk] :=
    distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k
  let biased : Tile .real [M, Bk] :=
    alibiScoreBlockTile (origPid * M) slope Q K scale k (Nat.succ_le_iff.mpr hk)
  let capped : Tile .real [M, Bk] :=
    softcapScoreTile softcap biased
  let vis : Tile .bool [M, Bk] :=
    slidingVisibleBlockTile (M := M) (Bk := Bk) (origPid * M) window k
  let scores : Tile .real [M, Bk] :=
    scoreBlockLane (slidingVisible window (origPid * M))
      (fun i j => softcapScore softcap (alibiScore (origPid * M) slope Q K scale i j))
      k (Nat.succ_le_iff.mpr hk)
  let sRaw := sLoad.setReg "raw" .real [M, Bk] raw
  let sDelta := sRaw.setReg "delta" .real [M, Bk] delta
  let sDist := sDelta.setReg "dist" .real [M, Bk] dist
  let sBiased := sDist.setReg "biased" .real [M, Bk] biased
  let sCapped := sBiased.setReg "capped" .real [M, Bk] capped
  let sVis := sCapped.setReg "visible" .bool [M, Bk] vis
  let sScore := sVis.setReg "scores" .real [M, Bk] scores
  refine ⟨sScore, ?_, ?_⟩
  · simp [fa1ScoreLoopScoreAlibiSlidingSoftcap, stepStmts, stepStmt, evalOp,
      Option.bind, hq, hkTileReg, hoffs_n, hoffs_m, raw, delta, dist, biased,
      capped, vis, scores, sRaw, sDelta, sDist, sBiased, sCapped, sVis, sScore]
    simp only [valueBlock]
    change ((((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "biased" .real [M, Bk]
          (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
            (Tile.bop NumericDType.real.sub Broadcast.scalarL
              (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
              (Tile.bop NumericDType.real.mul Broadcast.scalarL
                (Tile.scalar ((slope : ℝ) : WithBot ℝ))
                (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k))))).setReg
        "capped" .real [M, Bk]
          (Tile.bop NumericDType.real.mul Broadcast.scalarL
            (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
            (Tile.uop WithBot.realTanh
              (Tile.bop NumericDType.real.div Broadcast.scalarR
                (Tile.bop NumericDType.real.add
                  (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                  (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
                  (Tile.bop NumericDType.real.sub Broadcast.scalarL
                    (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
                    (Tile.bop NumericDType.real.mul Broadcast.scalarL
                      (Tile.scalar ((slope : ℝ) : WithBot ℝ))
                      (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k))))
                (Tile.scalar ((softcap : ℝ) : WithBot ℝ)))))).setReg
        "visible" .bool [M, Bk]
          (Tile.cop ComparableDType.real.lt Broadcast.scalarR
            (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)
            (Tile.scalar (((window : ℝ) : WithBot ℝ))))).setReg
        "scores" .real [M, Bk]
          (Tile.select
            (Tile.cop ComparableDType.real.lt Broadcast.scalarR
              (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k)
              (Tile.scalar (((window : ℝ) : WithBot ℝ))))
            (Tile.bop NumericDType.real.mul Broadcast.scalarL
              (Tile.scalar ((softcap : ℝ) : WithBot ℝ))
              (Tile.uop WithBot.realTanh
                (Tile.bop NumericDType.real.div Broadcast.scalarR
                  (Tile.bop NumericDType.real.add
                    (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
                    (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))
                    (Tile.bop NumericDType.real.sub Broadcast.scalarL
                      (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
                      (Tile.bop NumericDType.real.mul Broadcast.scalarL
                        (Tile.scalar ((slope : ℝ) : WithBot ℝ))
                        (distanceBlockTileNumeric (M := M) (Bk := Bk) (origPid * M) k))))
                  (Tile.scalar ((softcap : ℝ) : WithBot ℝ)))))
            (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])) =
        ((((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "biased" .real [M, Bk] biased).setReg
        "capped" .real [M, Bk] capped).setReg
        "visible" .bool [M, Bk] vis).setReg
        "scores" .real [M, Bk] scores
    rw [distanceBlockTileNumeric_eq (M := M) (Bk := Bk) (origPid * M) k]
    rw [alibiScoreBlock_numeric_eq_tile (origPid * M) slope Q K scale k hk]
    rw [softcapScoreTileNumeric_eq softcap
      (alibiScoreBlockTile (origPid * M) slope Q K scale k (Nat.succ_le_iff.mpr hk))]
    change ((((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "biased" .real [M, Bk] biased).setReg
        "capped" .real [M, Bk] capped).setReg
        "visible" .bool [M, Bk] vis).setReg
        "scores" .real [M, Bk]
          (Tile.select vis capped
            (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk])) =
        ((((((sLoad.setReg "raw" .real [M, Bk]
          (rawScoreBlockTile Q K scale k (Nat.succ_le_iff.mpr hk))).setReg
        "delta" .real [M, Bk] delta).setReg
        "dist" .real [M, Bk]
          (distanceBlockTile (M := M) (Bk := Bk) (origPid * M) k)).setReg
        "biased" .real [M, Bk] biased).setReg
        "capped" .real [M, Bk] capped).setReg
        "visible" .bool [M, Bk] vis).setReg
        "scores" .real [M, Bk] scores
    rw [show Tile.select vis capped
        (⟨fun _ : TileIndex [M, Bk] => (⊥ : WithBot ℝ)⟩ : Tile .real [M, Bk]) =
        scores by
      simp [vis, capped, biased, scores]
      exact alibiSlidingSoftcapScoreBlock_numeric_tile_eq
        (origPid * M) window slope softcap Q K scale k hk]
  · refine ⟨?_, ?_⟩
    · rcases hLoaded with
        ⟨hCore, hoffs_n, hk_ptrs, hv_ptrs, hkReg, hvReg⟩
      rcases hCore with
        ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
      · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hpidReg]
        · simpa [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw] using hpid
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hoffs_m]
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hoffs_d]
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hq]
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hm]
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hl]
        · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, ho]
        · intro idx
          simpa [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw] using hQ idx
        · intro idx
          simpa [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw] using hK idx
        · intro idx
          simpa [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw] using hV idx
      · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hoffs_n]
      · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hk_ptrs]
      · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hv_ptrs]
      · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hkReg]
      · simp [sScore, sVis, sCapped, sBiased, sDist, sDelta, sRaw, hvReg]
    · simp [sScore, scores]

set_option maxHeartbeats 800000 in
theorem fa1_score_loop_tail_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (k : Nat) (sScore : BlockState) (hk : k < numKVBlocks)
    (hP : P_fa1_score_blockrec_scored qReg kReg vReg origPid Q K V visible score
      k hk sScore) :
    ∃ s',
      stepStmts (fa1ScoreLoopTail M D Bk) sScore = some s' ∧
      P_fa1_score_blockrec qReg kReg vReg origPid Q K V visible score
        (k + 1) s' := by
  rcases hP with
    ⟨hLoaded, hscores⟩
  rcases hLoaded with
    ⟨hCore, _hoffs_n, _hk_ptrs, _hv_ptrs, _hkReg, hvReg⟩
  rcases hCore with
    ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, hm, hl, ho, hQ, hK, hV⟩
  let mOld : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      mScoreBlockPartial visible score k idx.1⟩
  let lOld : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      lScoreBlockPartial visible score k idx.1
  let oOld : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      oScoreBlockPartial visible score V k idx
  let scoresTile : Tile .real [M, Bk] :=
    scoreBlockLane visible score k (Nat.succ_le_iff.mpr hk)
  let mBlockExec : Tile .real [M] :=
    ⟨fun outIdx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup'
        (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
        (fun j => scoresTile.data (outIdx.1, j, outIdx.2))⟩
  let mNewExec : Tile .real [M] :=
    Tile.bop max (Broadcast.consSame Broadcast.nil) mOld mBlockExec
  let alphaExec : Tile .real [M] :=
    Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil)
        mOld mNewExec)
  let pExec : Tile .real [M, Bk] :=
    Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        scoresTile
        (Tile.expandDim ⟨1, by simp⟩ mNewExec))
  let lNewExec : Tile .real [M] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        alphaExec lOld)
      (Tile.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false pExec)
  let oNewExec : Tile .real [M, D] :=
    Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop NumericDType.real.mul
        (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩ alphaExec)
        oOld)
      (Tile.dot [] (M := M) (K := Bk) (N := D) pExec
        (valueBlock V k (Nat.succ_le_iff.mpr hk)))
  let mBlock : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      (Finset.univ : Finset (Fin Bk)).sup
        (fun j => scoreLane visible score idx.1
          (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) j))⟩
  let mNew : Tile .real [M] :=
    ⟨fun idx : TileIndex [M] =>
      mScoreBlockPartial visible score (k + 1) idx.1⟩
  let alpha : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      alphaScoreBlockPartial visible score k idx.1
  let p : Tile .real [M, Bk] :=
    Tile.ofReal fun idx : TileIndex [M, Bk] =>
      (WithBot.realExp
        (WithBot.realSub
          (scoreLane visible score idx.1
            (scoreBlockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) idx.2.1))
          (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0
  let lNew : Tile .real [M] :=
    Tile.ofReal fun idx : TileIndex [M] =>
      lScoreBlockPartial visible score (k + 1) idx.1
  let oNew : Tile .real [M, D] :=
    Tile.ofReal fun idx : TileIndex [M, D] =>
      oScoreBlockPartial visible score V (k + 1) idx
  let s1 := sScore.setReg "m_block" .real [M] mBlockExec
  let s2 := s1.setReg "m_new" .real [M] mNewExec
  let s3 := s2.setReg "alpha" .real [M] alphaExec
  let s4 := s3.setReg "p" .real [M, Bk] pExec
  let s5 := s4.setReg "l_new" .real [M] lNewExec
  let s6 := s5.setReg "o_acc" .real [M, D] oNewExec
  let s7 := s6.setReg "m_i" .real [M] mNewExec
  let s' := s7.setReg "l_i" .real [M] lNewExec
  have hmBlockExec : mBlockExec = mBlock := by
    exact score_block_mBlock_sup'_tile_eq
      (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩) visible score k hk
  have hmNewExec : mNewExec = mNew := by
    change Tile.bop max (Broadcast.consSame Broadcast.nil) mOld mBlockExec = mNew
    rw [hmBlockExec]
    change Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] =>
        mScoreBlockPartial visible score k idx.1⟩ : Tile .real [M])
      mBlock = mNew
    rw [score_block_mNew_tile_eq visible score k hk]
  have halphaExec : alphaExec = alpha := by
    change Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil)
        mOld mNewExec) = alpha
    rw [hmNewExec]
    ext idx
    simp [mOld, mNew, alpha, Tile.uop, Tile.bop,
      Tile.ofReal, alphaScoreBlockPartial]
    exact FA1MathBoundary.realExp_eq_some_unbotD _
  have hpExec : pExec = p := by
    change Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        scoresTile
        (Tile.expandDim ⟨1, by simp⟩ mNewExec)) = p
    rw [hmNewExec]
    ext idx
    simp [scoresTile, mNew, p, Tile.uop, Tile.bop,
      Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
    exact FA1MathBoundary.realExp_eq_some_unbotD _
  have hpRowSumExec :
      Tile.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false pExec =
        Tile.ofReal (fun idx : TileIndex [M] =>
          Finset.univ.sum (fun j : Fin Bk =>
            (WithBot.realExp
              (WithBot.realSub
                (scoreLane visible score idx.1
                  (scoreBlockIndex Bk numKVBlocks k
                    (Nat.succ_le_iff.mpr hk) j))
                (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0)) := by
    rw [hpExec]
    exact score_block_p_rowSum_eq visible score k hk
  have hlNewExec : lNewExec = lNew := by
    change Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        alphaExec lOld)
      (Tile.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false pExec) = lNew
    rw [halphaExec, hpRowSumExec]
    change Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        alpha lOld)
      (Tile.ofReal (fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1
                (scoreBlockIndex Bk numKVBlocks k
                  (Nat.succ_le_iff.mpr hk) j))
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0))) = lNew
    rw [score_block_lNew_tile_eq visible score k hk]
  have hpvDotExec :
      Tile.dot [] (M := M) (K := Bk) (N := D) pExec
        (valueBlock V k (Nat.succ_le_iff.mpr hk)) =
        Tile.ofReal (fun idx : TileIndex [M, D] =>
          Finset.univ.sum (fun jLocal : Fin Bk =>
            let j := scoreBlockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr hk) jLocal
            (WithBot.realExp
              (WithBot.realSub
                (scoreLane visible score idx.1 j)
                (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
              V (j, idx.2.1, PUnit.unit))) := by
    rw [hpExec]
    exact score_block_pv_dot_eq visible score V k hk
  have hoNewExec : oNewExec = oNew := by
    change Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop NumericDType.real.mul
        (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩ alphaExec) oOld)
      (Tile.dot [] (M := M) (K := Bk) (N := D) pExec
        (valueBlock V k (Nat.succ_le_iff.mpr hk))) = oNew
    rw [halphaExec, hpvDotExec]
    change Tile.bop WithBot.realAdd
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul
        (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩ alpha) oOld)
      (Tile.ofReal (fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun jLocal : Fin Bk =>
          let j := scoreBlockIndex Bk numKVBlocks k
            (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (WithBot.realSub
              (scoreLane visible score idx.1 j)
              (mScoreBlockPartial visible score (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit)))) = oNew
    rw [score_block_oAcc_tile_eq visible score V k hk]
  refine ⟨s', ?_, ?_⟩
  · simp [fa1ScoreLoopTail, stepStmts, stepStmt, evalOp, Option.bind,
      Tile.reduceMax, Tile.reduceMaxDrop, TileShape.axisDim,
      TileShape.eraseAxis, TileShape.insertAxisIndex,
      hBk, hscores, hm, hl, ho, hvReg,
      mOld, lOld, oOld, scoresTile, mBlockExec, mNewExec,
      alphaExec, pExec, lNewExec, oNewExec, s1, s2, s3, s4, s5,
      s6, s7, s']
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s', s7, s6, s5, s4, s3, s2, s1, hpidReg]
    · simpa [s', s7, s6, s5, s4, s3, s2, s1] using hpid
    · simp [s', s7, s6, s5, s4, s3, s2, s1, hoffs_m]
    · simp [s', s7, s6, s5, s4, s3, s2, s1, hoffs_d]
    · simp [s', s7, s6, s5, s4, s3, s2, s1, hq]
    · simp [s', s7, hmNewExec, mNew]
    · simp [s', hlNewExec, lNew]
    · simp [s', s7, s6, hoNewExec, oNew]
    · intro idx
      simpa [s', s7, s6, s5, s4, s3, s2, s1] using hQ idx
    · intro idx
      simpa [s', s7, s6, s5, s4, s3, s2, s1] using hK idx
    · intro idx
      simpa [s', s7, s6, s5, s4, s3, s2, s1] using hV idx

theorem fa1_score_loop_stepSoftcap_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale softcap : ℝ)
    (k : Nat) (s : BlockState) (hk : k < numKVBlocks)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V allVisible
      (softcapDotScore softcap Q K scale) k s) :
    ∃ s',
      stepStmts (fa1ScoreLoopBodySoftcap kReg vReg M D Bk scale softcap)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_score_blockrec qReg kReg vReg origPid Q K V allVisible
        (softcapDotScore softcap Q K scale) (k + 1) s' := by
  rcases fa1_score_loop_loadBlock_correct qReg kReg vReg origPid Q K V
      allVisible (softcapDotScore softcap Q K scale) k s hk hP with
    ⟨sLoad, hLoad, hLoaded⟩
  rcases fa1_score_loop_scoreSoftcap_correct qReg kReg vReg origPid Q K V
      scale softcap k sLoad hk hLoaded with
    ⟨sScore, hScore, hScored⟩
  rcases fa1_score_loop_tail_correct hBk qReg kReg vReg origPid Q K V
      allVisible (softcapDotScore softcap Q K scale) k sScore hk hScored with
    ⟨s', hTail, hTailP⟩
  refine ⟨s', ?_, hTailP⟩
  rw [fa1ScoreLoopBodySoftcap_parts_eq]
  rw [List.append_assoc]
  rw [stepStmts.append_some hLoad]
  rw [stepStmts.append_some hScore]
  exact hTail

theorem fa1_score_loop_stepAlibi_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (origPid qStart : Nat)
    (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (s : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V allVisible
      (alibiScore qStart slope Q K scale) k s) :
    ∃ s',
      stepStmts (fa1ScoreLoopBodyAlibi kReg vReg M D Bk scale slope)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_score_blockrec qReg kReg vReg origPid Q K V allVisible
        (alibiScore qStart slope Q K scale) (k + 1) s' := by
  rcases fa1_score_loop_loadBlock_correct qReg kReg vReg origPid Q K V
      allVisible (alibiScore qStart slope Q K scale) k s hk hP with
    ⟨sLoad, hLoad, hLoaded⟩
  rcases fa1_score_loop_scoreAlibi_correct qReg kReg vReg origPid qStart slope Q K V
      scale k sLoad hk hqStart hLoaded with
    ⟨sScore, hScore, hScored⟩
  rcases fa1_score_loop_tail_correct hBk qReg kReg vReg origPid Q K V
      allVisible (alibiScore qStart slope Q K scale) k sScore hk hScored with
    ⟨s', hTail, hTailP⟩
  refine ⟨s', ?_, hTailP⟩
  rw [fa1ScoreLoopBodyAlibi_parts_eq]
  rw [List.append_assoc]
  rw [stepStmts.append_some hLoad]
  rw [stepStmts.append_some hScore]
  exact hTail

theorem fa1_score_loop_stepSlidingWindow_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (s : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V
      (slidingVisible window qStart) (dotScore Q K scale) k s) :
    ∃ s',
      stepStmts (fa1ScoreLoopBodySlidingWindow kReg vReg M D Bk window scale)
        (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_score_blockrec qReg kReg vReg origPid Q K V
        (slidingVisible window qStart) (dotScore Q K scale) (k + 1) s' := by
  rcases fa1_score_loop_loadBlock_correct qReg kReg vReg origPid Q K V
      (slidingVisible window qStart) (dotScore Q K scale) k s hk hP with
    ⟨sLoad, hLoad, hLoaded⟩
  rcases fa1_score_loop_scoreSlidingWindow_correct qReg kReg vReg origPid qStart
      window Q K V scale k sLoad hk hqStart hLoaded with
    ⟨sScore, hScore, hScored⟩
  rcases fa1_score_loop_tail_correct hBk qReg kReg vReg origPid Q K V
      (slidingVisible window qStart) (dotScore Q K scale) k sScore hk hScored with
    ⟨s', hTail, hTailP⟩
  refine ⟨s', ?_, hTailP⟩
  rw [fa1ScoreLoopBodySlidingWindow_parts_eq]
  rw [List.append_assoc]
  rw [stepStmts.append_some hLoad]
  rw [stepStmts.append_some hScore]
  exact hTail

theorem fa1_score_loop_stepAlibiSlidingSoftcap_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat)
    (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (k : Nat) (s : BlockState) (hk : k < numKVBlocks)
    (hqStart : qStart = origPid * M)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V
      (slidingVisible window qStart)
      (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
      k s) :
    ∃ s',
      stepStmts
          (fa1ScoreLoopBodyAlibiSlidingSoftcap kReg vReg M D Bk window scale
            slope softcap)
          (s.setReg "n" .nat [] (Tile.scalar k)) = some s' ∧
      P_fa1_score_blockrec qReg kReg vReg origPid Q K V
        (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        (k + 1) s' := by
  rcases fa1_score_loop_loadBlock_correct qReg kReg vReg origPid Q K V
      (slidingVisible window qStart)
      (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
      k s hk hP with
    ⟨sLoad, hLoad, hLoaded⟩
  rcases fa1_score_loop_scoreAlibiSlidingSoftcap_correct qReg kReg vReg origPid
      qStart window slope softcap Q K V scale k sLoad hk hqStart hLoaded with
    ⟨sScore, hScore, hScored⟩
  rcases fa1_score_loop_tail_correct hBk qReg kReg vReg origPid Q K V
      (slidingVisible window qStart)
      (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
      k sScore hk hScored with
    ⟨s', hTail, hTailP⟩
  refine ⟨s', ?_, hTailP⟩
  rw [fa1ScoreLoopBodyAlibiSlidingSoftcap_parts_eq]
  rw [List.append_assoc]
  rw [stepStmts.append_some hLoad]
  rw [stepStmts.append_some hScore]
  exact hTail

theorem fa1_score_preLoop_correct
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ)
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := S) (cols := D) 0 D) V) :
    ∃ s0,
      stepStmts (fa1ScorePreLoop qReg M D) s = some s0 ∧
      P_fa1_score qReg kReg vReg s.pid Q K V visible score 0 s0 := by
  let qPtrs : Tile .nat [M, D] :=
    ⟨fun idx => (s.pid * M + idx.1.val) * D + idx.2.1.val⟩
  let qLoaded : Tile .real [M, D] :=
    ⟨fun idx => some (s.readMem qReg ((s.pid * M + idx.1.val) * D + idx.2.1.val))⟩
  let s0 :=
    ((((((((s.setReg "pid" .nat [] (Tile.scalar s.pid))
      ).setReg "offs_m" .nat [M] (Tile.vec fun i : Fin M => s.pid * M + i.val)
      ).setReg "offs_d" .nat [D] (Tile.vec fun d : Fin D => d.val)
      ).setReg "q_ptrs" .nat [M, D] qPtrs
      ).setReg "q" .real [M, D] qLoaded
      ).setReg "m_i" .real [M] ⟨fun _ => (⊥ : WithBot ℝ)⟩
      ).setReg "l_i" .real [M] (Tile.ofReal fun _ => 0)
      ).setReg "o_acc" .real [M, D] (Tile.ofReal fun _ => 0)
  have hQ_loaded_eq : qLoaded = Tile.ofReal Q := by
    ext idx
    simp [qLoaded, Tile.ofReal]
    rw [show (s.pid * M + idx.1.val) * D + idx.2.1.val =
        Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D idx by
          simp [Offset.rowMajor2D, Offset.strided, Nat.add_mul, Nat.mul_assoc,
            Nat.add_assoc]]
    exact congrArg some (hQ idx)
  refine ⟨s0, ?_, ?_⟩
  · simp [fa1ScorePreLoop, stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
      NumericDType.add, NumericDType.mul, Option.bind, TileShape.dropInsertedIndex, Tile.vec, Tile.ofReal, qPtrs, qLoaded, s0]
    rfl
  · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0]
    · simp [s0, hQ_loaded_eq]
    · simp [s0, mScoreOnline]
    · simp [s0, lScoreOnline, Tile.ofReal]
    · simp [s0, oScoreOnline, Tile.ofReal]
    · intro idx
      simpa [s0] using hQ idx
    · intro idx
      simpa [s0] using hK idx
    · intro idx
      simpa [s0] using hV idx

theorem fa1_score_preLoop_correct_blocks
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∃ s0,
      stepStmts (fa1ScorePreLoop qReg M D) s = some s0 ∧
      P_fa1_score_blocks qReg kReg vReg s.pid Q K V visible score 0 s0 := by
  rcases fa1_score_preLoop_correct qReg kReg vReg Q K V visible score s hQ hK hV with
    ⟨s0, hStep, hP⟩
  exact ⟨s0, hStep, by simpa [P_fa1_score_blocks] using hP⟩

theorem fa1_score_blockrec_preLoop_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∃ s0,
      stepStmts (fa1ScorePreLoop qReg M D) s = some s0 ∧
      P_fa1_score_blockrec qReg kReg vReg s.pid Q K V visible score 0 s0 := by
  rcases fa1_score_preLoop_correct qReg kReg vReg Q K V visible score s hQ hK hV with
    ⟨s0, hStep, hP⟩
  rcases hP with
    ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, _hm, _hl, _ho, hQ', hK', hV'⟩
  refine ⟨s0, hStep, ?_⟩
  refine ⟨hpidReg, hpid, hoffs_m, hoffs_d, hq, ?_, ?_, ?_, hQ', hK', hV'⟩
  · simpa [mScoreBlockPartial]
  · simpa [lScoreBlockPartial, Tile.ofReal]
  · simpa [oScoreBlockPartial, Tile.ofReal]

theorem fa1_score_postLoop_correct
    {M D S : Nat}
    (qReg kReg vReg outReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ)
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (sLoop : BlockState)
    (hP : P_fa1_score qReg kReg vReg origPid Q K V visible score S sLoop) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1ScorePostLoop outReg M D) sLoop)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) idx
        = some (attentionRealMaskedScore visible score V idx) := by
  intro idx
  rcases hP with
    ⟨_hpidReg, _hpid, hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hQ, _hK, _hV⟩
  have h_inj :
      Function.Injective
        (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) :=
    Offset.rowMajor2D_inj (base := origPid * M * D) (rowStride := D) (le_refl D)
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] => (origPid * M + i.1.val) * D + i.2.1.val) := by
    intro a b h
    apply h_inj
    simpa [Offset.rowMajor2D, Offset.strided, Nat.add_mul, Nat.mul_assoc,
      Nat.add_assoc] using h
  simp [observeTileAt, fa1ScorePostLoop, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho,
        Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, Offset.rowMajor2D, Offset.strided, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show origPid * M * D + idx.1.val * D + idx.2.1.val =
      (origPid * M + idx.1.val) * D + idx.2.1.val by
        rw [Nat.add_mul]]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj_store idx]
  simp [oScoreOnline_div_lScoreOnline_eq_attentionRealMaskedScore visible score V idx]

theorem fa1_score_postLoop_correct_blocks
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (sLoop : BlockState)
    (hP : P_fa1_score_blocks qReg kReg vReg origPid Q K V visible score
      numKVBlocks sLoop) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1ScorePostLoop outReg M D) sLoop)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) idx
        = some (attentionRealMaskedScore visible score V idx) := by
  intro idx
  exact fa1_score_postLoop_correct qReg kReg vReg outReg origPid Q K V
    visible score sLoop (by simpa [P_fa1_score_blocks] using hP) idx

theorem fa1_score_blockrec_postLoop_correct
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ) (sLoop : BlockState)
    (hP : P_fa1_score_blockrec qReg kReg vReg origPid Q K V visible score
      numKVBlocks sLoop) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts (fa1ScorePostLoop outReg M D) sLoop)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) idx
        = some
          (oScoreBlockPartial visible score V numKVBlocks idx /
            lScoreBlockPartial visible score numKVBlocks idx.1) := by
  intro idx
  rcases hP with
    ⟨_hpidReg, _hpid, hoffs_m, hoffs_d, _hq, _hm, hl, ho, _hQ, _hK, _hV⟩
  have h_inj :
      Function.Injective
        (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) :=
    Offset.rowMajor2D_inj (base := origPid * M * D) (rowStride := D) (le_refl D)
  have h_inj_store :
      Function.Injective
        (fun i : TileIndex [M, D] => (origPid * M + i.1.val) * D + i.2.1.val) := by
    intro a b h
    apply h_inj
    simpa [Offset.rowMajor2D, Offset.strided, Nat.add_mul, Nat.mul_assoc,
      Nat.add_assoc] using h
  simp [observeTileAt, fa1ScorePostLoop, stepStmts, stepStmt, evalOp, Tile.ofReal, hoffs_m, hoffs_d, hl, ho,
        Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
        NumericDType.div, Offset.rowMajor2D, Offset.strided, Option.bind,
        TileShape.dropInsertedIndex]
  rw [show origPid * M * D + idx.1.val * D + idx.2.1.val =
      (origPid * M + idx.1.val) * D + idx.2.1.val by
        rw [Nat.add_mul]]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj_store idx]

theorem fa1_score_blockrec_forward_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (visible : Fin M → Fin (Bk * numKVBlocks) → Bool)
    (score : Fin M → Fin (Bk * numKVBlocks) → ℝ)
    (body : List Stmt)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_score_blockrec qReg kReg vReg s.pid Q K V visible score i st →
          ∃ st',
            stepStmts body (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
            P_fa1_score_blockrec qReg kReg vReg s.pid Q K V visible score
              (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (stepStmts
            (fa1ScorePreLoop qReg M D ++
              [Stmt.forLoop "n" numKVBlocks body] ++
              fa1ScorePostLoop outReg M D) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial visible score V numKVBlocks idx /
            lScoreBlockPartial visible score numKVBlocks idx.1) := by
  obtain ⟨s0, hPre, hP0⟩ :=
    fa1_score_blockrec_preLoop_correct qReg kReg vReg Q K V visible score s hQ hK hV
  obtain ⟨sLoop, hLoopStmt, hPLoop⟩ :=
    forLoop_inv
      (P := P_fa1_score_blockrec qReg kReg vReg s.pid Q K V visible score) hP0
      hStep
  intro idx
  rw [List.append_assoc,
      stepStmts.append_some (l1 := fa1ScorePreLoop qReg M D)
        (l2 := [Stmt.forLoop "n" numKVBlocks body] ++ fa1ScorePostLoop outReg M D)
        hPre]
  rw [stepStmts.append_some (l1 := ([Stmt.forLoop "n" numKVBlocks body]))
      (l2 := fa1ScorePostLoop outReg M D) ?_]
  · exact fa1_score_blockrec_postLoop_correct qReg kReg vReg outReg s.pid Q K V
      visible score sLoop hPLoop idx
  · rw [stepStmts.cons_some hLoopStmt]
    exact stepStmts.nil

theorem fa1_forward_softcap_blockrec_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale softcap : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_score_blockrec qReg kReg vReg s.pid Q K V allVisible
          (softcapDotScore softcap Q K scale) i st →
          ∃ st',
            stepStmts (fa1ScoreLoopBodySoftcap kReg vReg M D Bk scale softcap)
              (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
            P_fa1_score_blockrec qReg kReg vReg s.pid Q K V allVisible
              (softcapDotScore softcap Q K scale) (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks scale softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial allVisible (softcapDotScore softcap Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial allVisible (softcapDotScore softcap Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [fa1ForwardKernelSoftcap_body_eq]
  exact fa1_score_blockrec_forward_raw_of_step qReg kReg vReg outReg Q K V
    allVisible (softcapDotScore softcap Q K scale)
    (fa1ScoreLoopBodySoftcap kReg vReg M D Bk scale softcap) s hQ hK hV
    hStep idx

theorem fa1_forward_alibi_blockrec_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_score_blockrec qReg kReg vReg s.pid Q K V allVisible
          (alibiScore qStart slope Q K scale) i st →
          ∃ st',
            stepStmts (fa1ScoreLoopBodyAlibi kReg vReg M D Bk scale slope)
              (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
            P_fa1_score_blockrec qReg kReg vReg s.pid Q K V allVisible
              (alibiScore qStart slope Q K scale) (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibi qReg kReg vReg outReg M D Bk
              numKVBlocks scale slope) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial allVisible (alibiScore qStart slope Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial allVisible (alibiScore qStart slope Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [fa1ForwardKernelAlibi_body_eq]
  exact fa1_score_blockrec_forward_raw_of_step qReg kReg vReg outReg Q K V
    allVisible (alibiScore qStart slope Q K scale)
    (fa1ScoreLoopBodyAlibi kReg vReg M D Bk scale slope) s hQ hK hV
    hStep idx

theorem fa1_forward_slidingWindow_blockrec_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_score_blockrec qReg kReg vReg s.pid Q K V
          (slidingVisible window qStart) (dotScore Q K scale) i st →
          ∃ st',
            stepStmts (fa1ScoreLoopBodySlidingWindow kReg vReg M D Bk window scale)
              (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
            P_fa1_score_blockrec qReg kReg vReg s.pid Q K V
              (slidingVisible window qStart) (dotScore Q K scale) (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSlidingWindow qReg kReg vReg outReg M D Bk
              numKVBlocks window scale) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [fa1ForwardKernelSlidingWindow_body_eq]
  exact fa1_score_blockrec_forward_raw_of_step qReg kReg vReg outReg Q K V
    (slidingVisible window qStart) (dotScore Q K scale)
    (fa1ScoreLoopBodySlidingWindow kReg vReg M D Bk window scale) s hQ hK hV
    hStep idx

theorem fa1_forward_alibiSlidingSoftcap_blockrec_raw_of_step
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V)
    (hStep :
      ∀ i st, i < numKVBlocks →
        P_fa1_score_blockrec qReg kReg vReg s.pid Q K V
          (slidingVisible window qStart)
          (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
          i st →
          ∃ st',
            stepStmts
              (fa1ScoreLoopBodyAlibiSlidingSoftcap kReg vReg M D Bk window
                scale slope softcap)
              (st.setReg "n" .nat [] (Tile.scalar i)) = some st' ∧
            P_fa1_score_blockrec qReg kReg vReg s.pid Q K V
              (slidingVisible window qStart)
              (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
              (i + 1) st') :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibiSlidingSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks window scale slope softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial (slidingVisible window qStart)
              (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
              V numKVBlocks idx /
            lScoreBlockPartial (slidingVisible window qStart)
              (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
              numKVBlocks idx.1) := by
  intro idx
  show observeTileAt (stepStmts _ s) outReg _ idx = _
  rw [fa1ForwardKernelAlibiSlidingSoftcap_body_eq]
  exact fa1_score_blockrec_forward_raw_of_step qReg kReg vReg outReg Q K V
    (slidingVisible window qStart)
    (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
    (fa1ScoreLoopBodyAlibiSlidingSoftcap kReg vReg M D Bk window scale slope softcap)
    s hQ hK hV hStep idx

theorem fa1_forward_softcap_blockrec_raw
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale softcap : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks scale softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial allVisible (softcapDotScore softcap Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial allVisible (softcapDotScore softcap Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  exact fa1_forward_softcap_blockrec_raw_of_step qReg kReg vReg outReg Q K V
    scale softcap s hQ hK hV
    (fun i st hi hP =>
      fa1_score_loop_stepSoftcap_correct hBk qReg kReg vReg s.pid Q K V
        scale softcap i st hi hP) idx

theorem fa1_forward_alibi_blockrec_raw
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibi qReg kReg vReg outReg M D Bk
              numKVBlocks scale slope) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial allVisible (alibiScore qStart slope Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial allVisible (alibiScore qStart slope Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  exact fa1_forward_alibi_blockrec_raw_of_step qReg kReg vReg outReg qStart slope
    Q K V scale s hQ hK hV
    (fun i st hi hP =>
      fa1_score_loop_stepAlibi_correct hBk qReg kReg vReg s.pid qStart slope Q K V
        scale i st hi hqStart hP) idx

theorem fa1_forward_slidingWindow_blockrec_raw
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSlidingWindow qReg kReg vReg outReg M D Bk
              numKVBlocks window scale) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale) V
              numKVBlocks idx /
            lScoreBlockPartial (slidingVisible window qStart) (dotScore Q K scale)
              numKVBlocks idx.1) := by
  intro idx
  exact fa1_forward_slidingWindow_blockrec_raw_of_step qReg kReg vReg outReg
    qStart window Q K V scale s hQ hK hV
    (fun i st hi hP =>
      fa1_score_loop_stepSlidingWindow_correct hBk qReg kReg vReg s.pid qStart
        window Q K V scale i st hi hqStart hP) idx

theorem fa1_forward_alibiSlidingSoftcap_blockrec_raw
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibiSlidingSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks window scale slope softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (oScoreBlockPartial (slidingVisible window qStart)
              (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
              V numKVBlocks idx /
            lScoreBlockPartial (slidingVisible window qStart)
              (fun r c => softcapScore softcap (alibiScore qStart slope Q K scale r c))
              numKVBlocks idx.1) := by
  intro idx
  exact fa1_forward_alibiSlidingSoftcap_blockrec_raw_of_step qReg kReg vReg outReg
    qStart window slope softcap Q K V scale s hQ hK hV
    (fun i st hi hP =>
      fa1_score_loop_stepAlibiSlidingSoftcap_correct hBk qReg kReg vReg s.pid
        qStart window slope softcap Q K V scale i st hi hqStart hP) idx

theorem fa1_forward_softcap_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale softcap : ℝ)
    (s : BlockState)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks scale softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some (attentionRealSoftcap softcap Q K V scale idx) := by
  intro idx
  have hraw :=
    fa1_forward_softcap_blockrec_raw hBk qReg kReg vReg outReg Q K V
      scale softcap s hQ hK hV idx
  simpa [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealSoftcap
    softcap Q K V scale idx] using hraw

theorem fa1_forward_alibi_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibi qReg kReg vReg outReg M D Bk
              numKVBlocks scale slope) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some (attentionRealAlibi qStart slope Q K V scale idx) := by
  intro idx
  have hraw :=
    fa1_forward_alibi_blockrec_raw hBk qReg kReg vReg outReg qStart slope
      Q K V scale s hqStart hQ hK hV idx
  simpa [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealAlibi
    qStart slope Q K V scale idx] using hraw

theorem fa1_forward_slidingWindow_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelSlidingWindow qReg kReg vReg outReg M D Bk
              numKVBlocks window scale) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some (attentionRealSlidingWindow qStart window Q K V scale idx) := by
  intro idx
  have hraw :=
    fa1_forward_slidingWindow_blockrec_raw hBk qReg kReg vReg outReg
      qStart window Q K V scale s hqStart hQ hK hV idx
  simpa [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealSlidingWindow
    qStart window Q K V scale idx] using hraw

theorem fa1_forward_alibiSlidingSoftcap_correct
    {M D Bk numKVBlocks : Nat} (hBk : 0 < Bk)
    (qReg kReg vReg outReg : RegionName)
    (qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ)
    (s : BlockState)
    (hqStart : qStart = s.pid * M)
    (hQ : InputAt s qReg
        (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) Q)
    (hK : InputAt s kReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K)
    (hV : InputAt s vReg
        (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V) :
    ∀ idx : TileIndex [M, D],
      observeTileAt
          (exec
            (fa1ForwardKernelAlibiSlidingSoftcap qReg kReg vReg outReg M D Bk
              numKVBlocks window scale slope softcap) s)
          outReg
          (Offset.rowMajor2D (rows := M) (cols := D) (s.pid * M * D) D) idx
        = some
          (attentionRealAlibiSlidingSoftcap qStart window slope softcap
            Q K V scale idx) := by
  intro idx
  have hraw :=
    fa1_forward_alibiSlidingSoftcap_blockrec_raw hBk qReg kReg vReg outReg
      qStart window slope softcap Q K V scale s hqStart hQ hK hV idx
  simpa [oScoreBlockPartial_div_lScoreBlockPartial_eq_attentionRealAlibiSlidingSoftcap
    qStart window slope softcap Q K V scale idx] using hraw

theorem P_fa1_score_readout_ratio
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ)
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (s : BlockState)
    (_hP : P_fa1_score qReg kReg vReg origPid Q K V visible score S s)
    (idx : TileIndex [M, D]) :
    oScoreOnline visible score V S idx /
        lScoreOnline visible score S idx.1 =
      attentionRealMaskedScore visible score V idx := by
  exact oScoreOnline_div_lScoreOnline_eq_attentionRealMaskedScore
    visible score V idx

theorem P_fa1_score_readout_alibi
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart : Nat) (slope : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ) (scale : ℝ) (s : BlockState)
    (_hP : P_fa1_score qReg kReg vReg origPid Q K V allVisible
      (alibiScore qStart slope Q K scale) S s)
    (idx : TileIndex [M, D]) :
    oScoreOnline allVisible (alibiScore qStart slope Q K scale) V S idx /
        lScoreOnline allVisible (alibiScore qStart slope Q K scale) S idx.1 =
      attentionRealAlibi qStart slope Q K V scale idx := by
  exact oScoreOnline_div_lScoreOnline_eq_attentionRealAlibi
    qStart slope Q K V scale idx

theorem P_fa1_score_readout_slidingWindow
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ) (scale : ℝ) (s : BlockState)
    (_hP : P_fa1_score qReg kReg vReg origPid Q K V
      (slidingVisible window qStart) (dotScore Q K scale) S s)
    (idx : TileIndex [M, D]) :
    oScoreOnline (slidingVisible window qStart) (dotScore Q K scale) V S idx /
        lScoreOnline (slidingVisible window qStart) (dotScore Q K scale) S idx.1 =
      attentionRealSlidingWindow qStart window Q K V scale idx := by
  exact oScoreOnline_div_lScoreOnline_eq_attentionRealSlidingWindow
    qStart window Q K V scale idx

theorem P_fa1_score_readout_softcap
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat) (softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ) (scale : ℝ) (s : BlockState)
    (_hP : P_fa1_score qReg kReg vReg origPid Q K V allVisible
      (softcapDotScore softcap Q K scale) S s)
    (idx : TileIndex [M, D]) :
    oScoreOnline allVisible (softcapDotScore softcap Q K scale) V S idx /
        lScoreOnline allVisible (softcapDotScore softcap Q K scale) S idx.1 =
      attentionRealSoftcap softcap Q K V scale idx := by
  exact oScoreOnline_div_lScoreOnline_eq_attentionRealSoftcap
    softcap Q K V scale idx

theorem P_fa1_score_readout_alibiSlidingSoftcap
    {M D S : Nat}
    (qReg kReg vReg : RegionName)
    (origPid qStart window : Nat) (slope softcap : ℝ)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ) (scale : ℝ) (s : BlockState)
    (_hP : P_fa1_score qReg kReg vReg origPid Q K V
      (slidingVisible window qStart)
      (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
      S s)
    (idx : TileIndex [M, D]) :
    oScoreOnline (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        V S idx /
      lScoreOnline (slidingVisible window qStart)
        (fun i j => softcapScore softcap (alibiScore qStart slope Q K scale i j))
        S idx.1 =
      attentionRealAlibiSlidingSoftcap qStart window slope softcap Q K V scale idx := by
  exact oScoreOnline_div_lScoreOnline_eq_attentionRealAlibiSlidingSoftcap
    qStart window slope softcap Q K V scale idx

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
