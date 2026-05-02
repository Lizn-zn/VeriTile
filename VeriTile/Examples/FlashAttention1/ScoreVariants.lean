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

/-! ## Shifted online score recurrence

These definitions mirror FA-1's numerically stable loop state for an
arbitrary score function. Masked-out keys are represented as `⊥`, so their
exponentiated contribution is zero.
-/

noncomputable def scoreLane {M S : Nat}
    (visible : Fin M → Fin S → Bool)
    (score : Fin M → Fin S → ℝ) (i : Fin M) (j : Fin S) : WithBot ℝ :=
  if visible i j then ((score i j : ℝ) : WithBot ℝ) else ⊥

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
      (Op.load qReg (Op.ref .nat [M, D] "q_ptrs"))
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
  , Stmt.store outReg [M, D]
      (Op.ref .nat [M, D] "o_ptrs")
      (Op.ref .real [M, D] "out")
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
      (Op.load kReg (Op.ref .nat [Bk, D] "k_ptrs"))
  , Stmt.assign .real [Bk, D] "v"
      (Op.load vReg (Op.ref .nat [Bk, D] "v_ptrs"))
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
      (Op.load kReg (Op.ref .nat [Bk, D] "k_ptrs"))
  , Stmt.assign .real [Bk, D] "v"
      (Op.load vReg (Op.ref .nat [Bk, D] "v_ptrs"))
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
      (Op.load kReg (Op.ref .nat [Bk, D] "k_ptrs"))
  , Stmt.assign .real [Bk, D] "v"
      (Op.load vReg (Op.ref .nat [Bk, D] "v_ptrs"))
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
      (Op.load kReg (Op.ref .nat [Bk, D] "k_ptrs"))
  , Stmt.assign .real [Bk, D] "v"
      (Op.load vReg (Op.ref .nat [Bk, D] "v_ptrs"))
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
      (Op.load kReg (Op.ref .nat [Bk, D] "k_ptrs"))
  , Stmt.assign .real [Bk, D] "v"
      (Op.load vReg (Op.ref .nat [Bk, D] "v_ptrs"))
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

end FA1Score

end VeriTile.Examples
