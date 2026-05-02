/-
VeriTile.Examples.FlashAttention1

FlashAttention-1 forward kernel — DSL definition + math model + full
correctness theorem.

v0 scope:
  * One `program_id` ↔ one Q-row block of `[BLOCK_M, D]`.
  * K, V are full `[S, D]` matrices in memory (single-block-output).
  * Inner KV-block loop with the canonical online-softmax update
    (running `m_i`, `l_i`, `O`).
  * Arbitrary user-supplied `scale : ℝ` (the standard `1/√D` Triton
    specialization is the caller's responsibility).
  * Non-causal and causal strided kernels are fully proved against
    their 4D reference specs.
  * Full-tile assumptions remain explicit:
    `Bk * numKVBlocks = S_k` and `q_block * M + M <= S_q`.

`fa1_forward_correct_4D` and `fa1_forward_correct_4D_causal` are fully
proven end-to-end via the four-stage decomposition (math identity +
pre-loop init + loop step via `forLoop_inv` + post-loop readout).

v1 / boundary-mask scope:
  * Boundary-masked non-causal and causal kernels are defined and proved
    through 4D/layout/view theorem surfaces.
  * D-tail kernels split logical head dimension `D` from block width `Bd`;
    their DSL surface and padded-D math bridges are in place.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## FA-1 forward kernel — DSL definition

Online-softmax recurrence over KV blocks:

  m_i, l_i, O ← running max / normalizer / output (all per Q-row)

  for n in 0..numKVBlocks:
    K_b, V_b ← load
    s   = (Q · K_bᵀ) · scale                       -- [M, Bk]
    m'  = max(m_i, rowmax(s))                          -- [M]
    α   = exp(m_i - m')                                -- [M]
    p   = exp(s - m'[:, None])                         -- [M, Bk]
    l'  = α * l_i + rowsum(p)                          -- [M]
    O'  = α[:, None] * O + p · V_b                     -- [M, D]
    (m_i, l_i, O) ← (m', l', O')

  out = O / l_i[:, None]                               -- [M, D]

The kernel is parameterized by the four regions (Q, K, V, output) and
by `M` (Q-block / output rows), `D` (head dim), `Bk` (KV-block size),
and `numKVBlocks` (`S = numKVBlocks * Bk` is the implicit invariant).
The scaling factor `1/√D` enters as a Lean-level `ℝ` antiquote so the
kernel can be parameterized by precomputed constants. -/

def fa1ForwardKernel
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat) (scale : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)

  offs_m := pid * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))

  -- Load Q-block once (reused across all KV blocks).
  q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  q      := tl.load($(qReg) + q_ptrs)

  -- Initialize accumulators at the right tile shape (so the loop
  -- registers stay rank-correct across iterations).
  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    k       := tl.load($(kReg) + k_ptrs)
    v       := tl.load($(vReg) + v_ptrs)

    -- Scaled scores: [M, Bk]
    scores  := tl.dot(q, tl.trans(k)) * $ℝ(scale)

    -- Online-softmax update.
    m_block := tl.max(scores, axis = 1)
    m_new   := tl.max(m_i, m_block)
    alpha   := tl.exp(m_i - m_new)
    p       := tl.exp(scores - m_new[:, None])
    l_new   := alpha * l_i + tl.sum(p, axis = 1)
    o_acc   := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i     := m_new
    l_i     := l_new
  }

  -- Final normalize and store.
  out    := o_acc / l_i[:, None]
  o_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]
  tl.store($(outReg) + o_ptrs, out)
}

/-! ## Strided / 4D-aware kernel

Real-Triton FA-1 kernels are launched on a 3D grid `(numQBlocks, H, B)`
and use `tl.program_id(0/1/2)` plus per-tensor stride parameters for
arbitrary `[B, H, S, D]` memory layouts. The strided kernel below
mirrors that shape: 16 stride parameters (4 per Q/K/V/O), three
`program_id` axes for `(q_block, head, batch)`. Body is otherwise
identical to `fa1ForwardKernel` above — same online-softmax recurrence
on the per-`(b, h, q_block)` slice.

The Step 1 proof (issue #39) targets this kernel. The original
`fa1ForwardKernel` is recovered by instantiating
`stride_q*b = stride_q*h = 0`, `stride_qs = D`, `stride_qd = 1`, etc.
Step 1 (i) lands the definition only — proofs come in subsequent
commits. -/

def fa1ForwardKernelStrided
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat)
    -- Q strides (axes [B, H, S_q, D]):
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    -- K strides (axes [B, H, S_k, D]):
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    -- V strides (axes [B, H, S_k, D]):
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    -- Output strides (axes [B, H, S_q, D]):
    (stride_ob stride_oh stride_om stride_od : Nat)
    (scale : ℝ) : Kernel := triton {
  pid_qb := tl.program_id(0)
  pid_h  := tl.program_id(1)
  pid_b  := tl.program_id(2)

  q_base_off := pid_b * $(stride_qb) + pid_h * $(stride_qh)
  k_base_off := pid_b * $(stride_kb) + pid_h * $(stride_kh)
  v_base_off := pid_b * $(stride_vb) + pid_h * $(stride_vh)
  o_base_off := pid_b * $(stride_ob) + pid_h * $(stride_oh)

  offs_m := pid_qb * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))

  q_ptrs := q_base_off + offs_m[:, None] * $(stride_qs) + offs_d[None, :] * $(stride_qd)
  q      := tl.load($(qReg) + q_ptrs)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
    v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
    k       := tl.load($(kReg) + k_ptrs)
    v       := tl.load($(vReg) + v_ptrs)

    scores  := tl.dot(q, tl.trans(k)) * $ℝ(scale)
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
  o_ptrs := o_base_off + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_od)
  tl.store($(outReg) + o_ptrs, out)
}

/-! ## Causal strided / 4D-aware kernel

The causal variant is identical to `fa1ForwardKernelStrided` except
that each score block is masked before the online-softmax update:

```python
mask = offs_m[:, None] >= offs_n[None, :]
scores = tl.where(mask, scores, -inf)
```

Both `offs_m` and `offs_n` are global sequence indices. This is the
important semantic point: inside the KV loop, `offs_n` is
`n * BLOCK_N + arange(BLOCK_N)`, not the local `arange(BLOCK_N)`.
-/
def fa1ForwardKernelStridedCausal
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks : Nat)
    -- Q strides (axes [B, H, S_q, D]):
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    -- K strides (axes [B, H, S_k, D]):
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    -- V strides (axes [B, H, S_k, D]):
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    -- Output strides (axes [B, H, S_q, D]):
    (stride_ob stride_oh stride_om stride_od : Nat)
    (scale : ℝ) : Kernel := triton {
  pid_qb := tl.program_id(0)
  pid_h  := tl.program_id(1)
  pid_b  := tl.program_id(2)

  q_base_off := pid_b * $(stride_qb) + pid_h * $(stride_qh)
  k_base_off := pid_b * $(stride_kb) + pid_h * $(stride_kh)
  v_base_off := pid_b * $(stride_vb) + pid_h * $(stride_vh)
  o_base_off := pid_b * $(stride_ob) + pid_h * $(stride_oh)

  offs_m := pid_qb * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))

  q_ptrs := q_base_off + offs_m[:, None] * $(stride_qs) + offs_d[None, :] * $(stride_qd)
  q      := tl.load($(qReg) + q_ptrs)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
    v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
    k       := tl.load($(kReg) + k_ptrs)
    v       := tl.load($(vReg) + v_ptrs)

    scores_raw := tl.dot(q, tl.trans(k)) * $ℝ(scale)
    causal     := offs_m[:, None] >= offs_n[None, :]
    scores     := tl.where(causal, scores_raw, -inf)
    m_block    := tl.max(scores, axis = 1)
    m_new      := tl.max(m_i, m_block)
    alpha      := tl.exp(m_i - m_new)
    p          := tl.exp(scores - m_new[:, None])
    l_new      := alpha * l_i + tl.sum(p, axis = 1)
    o_acc      := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i        := m_new
    l_i        := l_new
  }

  out    := o_acc / l_i[:, None]
  o_ptrs := o_base_off + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_od)
  tl.store($(outReg) + o_ptrs, out)
}

/-! ## Boundary-masked strided kernels (FA-1 v1 scaffold)

Step 2 of issue #39 removes the divisibility assumptions for the sequence
axes. These kernels keep the tile sizes `M` and `Bk` fixed, but take logical
sequence lengths `S_q` and `S_k` separately:

* Q loads and output stores are masked by `offs_m < S_q`.
* K/V loads are masked by `offs_n < S_k` with `other = 0`.
* Invalid KV score lanes are explicitly rewritten to `-inf` before the
  online-softmax update, so they contribute zero probability mass.

The head dimension still uses tile size `D` directly. Supporting a separate
`BLOCK_D` with `offs_d < D_head` is a later extension because it changes the
typed matrix multiplication shapes. -/

def fa1ForwardKernelStridedBoundary
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks S_q S_k : Nat)
    -- Q strides (axes [B, H, S_q, D]):
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    -- K strides (axes [B, H, S_k, D]):
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    -- V strides (axes [B, H, S_k, D]):
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    -- Output strides (axes [B, H, S_q, D]):
    (stride_ob stride_oh stride_om stride_od : Nat)
    (scale : ℝ) : Kernel := triton {
  pid_qb := tl.program_id(0)
  pid_h  := tl.program_id(1)
  pid_b  := tl.program_id(2)

  q_base_off := pid_b * $(stride_qb) + pid_h * $(stride_qh)
  k_base_off := pid_b * $(stride_kb) + pid_h * $(stride_kh)
  v_base_off := pid_b * $(stride_vb) + pid_h * $(stride_vh)
  o_base_off := pid_b * $(stride_ob) + pid_h * $(stride_oh)

  offs_m := pid_qb * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))

  q_ptrs := q_base_off + offs_m[:, None] * $(stride_qs) + offs_d[None, :] * $(stride_qd)
  q_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  q      := tl.load($(qReg) + q_ptrs, mask=q_mask, other=0)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
    v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
    kv_mask := (offs_n[:, None] + offs_d[None, :] * $(0)) < $(S_k)
    k       := tl.load($(kReg) + k_ptrs, mask=kv_mask, other=0)
    v       := tl.load($(vReg) + v_ptrs, mask=kv_mask, other=0)

    scores_raw := tl.dot(q, tl.trans(k)) * $ℝ(scale)
    score_mask := (offs_m[:, None] * $(0) + offs_n[None, :]) < $(S_k)
    scores     := tl.where(score_mask, scores_raw, -inf)
    m_block    := tl.max(scores, axis = 1)
    m_new      := tl.max(m_i, m_block)
    alpha      := tl.exp(m_i - m_new)
    p          := tl.exp(scores - m_new[:, None])
    l_new      := alpha * l_i + tl.sum(p, axis = 1)
    o_acc      := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i        := m_new
    l_i        := l_new
  }

  out    := o_acc / l_i[:, None]
  o_ptrs := o_base_off + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_od)
  o_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  tl.store($(outReg) + o_ptrs, out, mask=o_mask)
}

def fa1ForwardKernelStridedCausalBoundary
    (qReg kReg vReg outReg : RegionName)
    (M D Bk numKVBlocks S_q S_k : Nat)
    -- Q strides (axes [B, H, S_q, D]):
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    -- K strides (axes [B, H, S_k, D]):
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    -- V strides (axes [B, H, S_k, D]):
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    -- Output strides (axes [B, H, S_q, D]):
    (stride_ob stride_oh stride_om stride_od : Nat)
    (scale : ℝ) : Kernel := triton {
  pid_qb := tl.program_id(0)
  pid_h  := tl.program_id(1)
  pid_b  := tl.program_id(2)

  q_base_off := pid_b * $(stride_qb) + pid_h * $(stride_qh)
  k_base_off := pid_b * $(stride_kb) + pid_h * $(stride_kh)
  v_base_off := pid_b * $(stride_vb) + pid_h * $(stride_vh)
  o_base_off := pid_b * $(stride_ob) + pid_h * $(stride_oh)

  offs_m := pid_qb * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(D))

  q_ptrs := q_base_off + offs_m[:, None] * $(stride_qs) + offs_d[None, :] * $(stride_qd)
  q_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  q      := tl.load($(qReg) + q_ptrs, mask=q_mask, other=0)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
    v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
    kv_mask := (offs_n[:, None] + offs_d[None, :] * $(0)) < $(S_k)
    k       := tl.load($(kReg) + k_ptrs, mask=kv_mask, other=0)
    v       := tl.load($(vReg) + v_ptrs, mask=kv_mask, other=0)

    scores_raw := tl.dot(q, tl.trans(k)) * $ℝ(scale)
    causal     := offs_m[:, None] >= offs_n[None, :]
    causal_scores := tl.where(causal, scores_raw, -inf)
    score_mask := (offs_m[:, None] * $(0) + offs_n[None, :]) < $(S_k)
    scores     := tl.where(score_mask, causal_scores, -inf)
    m_block    := tl.max(scores, axis = 1)
    m_new      := tl.max(m_i, m_block)
    alpha      := tl.exp(m_i - m_new)
    p          := tl.exp(scores - m_new[:, None])
    l_new      := alpha * l_i + tl.sum(p, axis = 1)
    o_acc      := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i        := m_new
    l_i        := l_new
  }

  out    := o_acc / l_i[:, None]
  o_ptrs := o_base_off + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_od)
  o_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  tl.store($(outReg) + o_ptrs, out, mask=o_mask)
}

/-! ## Boundary-masked strided kernels with D-tail masks

These variants split the hidden-dimension tile width `Bd` from the logical
head dimension `D`. They model the production `BLOCK_D` tail case: lanes with
`offs_d >= D` are masked on Q/K/V loads and output stores, so the dot product
sees zero in padded hidden lanes and stores only logical output columns. -/

def fa1ForwardKernelStridedBoundaryD
    (qReg kReg vReg outReg : RegionName)
    (M Bd Bk numKVBlocks S_q S_k D : Nat)
    -- Q strides (axes [B, H, S_q, D]):
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    -- K strides (axes [B, H, S_k, D]):
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    -- V strides (axes [B, H, S_k, D]):
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    -- Output strides (axes [B, H, S_q, D]):
    (stride_ob stride_oh stride_om stride_od : Nat)
    (scale : ℝ) : Kernel := triton {
  pid_qb := tl.program_id(0)
  pid_h  := tl.program_id(1)
  pid_b  := tl.program_id(2)

  q_base_off := pid_b * $(stride_qb) + pid_h * $(stride_qh)
  k_base_off := pid_b * $(stride_kb) + pid_h * $(stride_kh)
  v_base_off := pid_b * $(stride_vb) + pid_h * $(stride_vh)
  o_base_off := pid_b * $(stride_ob) + pid_h * $(stride_oh)

  offs_m := pid_qb * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(Bd))

  q_ptrs := q_base_off + offs_m[:, None] * $(stride_qs) + offs_d[None, :] * $(stride_qd)
  q_seq_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  q_d_mask   := (offs_m[:, None] * $(0) + offs_d[None, :]) < $(D)
  q_mask     := tl.logical_and(q_seq_mask, q_d_mask)
  q          := tl.load($(qReg) + q_ptrs, mask=q_mask, other=0)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(Bd)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
    v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
    kv_seq_mask := (offs_n[:, None] + offs_d[None, :] * $(0)) < $(S_k)
    kv_d_mask   := (offs_n[:, None] * $(0) + offs_d[None, :]) < $(D)
    kv_mask     := tl.logical_and(kv_seq_mask, kv_d_mask)
    k       := tl.load($(kReg) + k_ptrs, mask=kv_mask, other=0)
    v       := tl.load($(vReg) + v_ptrs, mask=kv_mask, other=0)

    scores_raw := tl.dot(q, tl.trans(k)) * $ℝ(scale)
    score_mask := (offs_m[:, None] * $(0) + offs_n[None, :]) < $(S_k)
    scores     := tl.where(score_mask, scores_raw, -inf)
    m_block    := tl.max(scores, axis = 1)
    m_new      := tl.max(m_i, m_block)
    alpha      := tl.exp(m_i - m_new)
    p          := tl.exp(scores - m_new[:, None])
    l_new      := alpha * l_i + tl.sum(p, axis = 1)
    o_acc      := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i        := m_new
    l_i        := l_new
  }

  out    := o_acc / l_i[:, None]
  o_ptrs := o_base_off + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_od)
  o_seq_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  o_d_mask   := (offs_m[:, None] * $(0) + offs_d[None, :]) < $(D)
  o_mask     := tl.logical_and(o_seq_mask, o_d_mask)
  tl.store($(outReg) + o_ptrs, out, mask=o_mask)
}

def fa1ForwardKernelStridedCausalBoundaryD
    (qReg kReg vReg outReg : RegionName)
    (M Bd Bk numKVBlocks S_q S_k D : Nat)
    -- Q strides (axes [B, H, S_q, D]):
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    -- K strides (axes [B, H, S_k, D]):
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    -- V strides (axes [B, H, S_k, D]):
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    -- Output strides (axes [B, H, S_q, D]):
    (stride_ob stride_oh stride_om stride_od : Nat)
    (scale : ℝ) : Kernel := triton {
  pid_qb := tl.program_id(0)
  pid_h  := tl.program_id(1)
  pid_b  := tl.program_id(2)

  q_base_off := pid_b * $(stride_qb) + pid_h * $(stride_qh)
  k_base_off := pid_b * $(stride_kb) + pid_h * $(stride_kh)
  v_base_off := pid_b * $(stride_vb) + pid_h * $(stride_vh)
  o_base_off := pid_b * $(stride_ob) + pid_h * $(stride_oh)

  offs_m := pid_qb * $(M) + tl.arange(0, $(M))
  offs_d := tl.arange(0, $(Bd))

  q_ptrs := q_base_off + offs_m[:, None] * $(stride_qs) + offs_d[None, :] * $(stride_qd)
  q_seq_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  q_d_mask   := (offs_m[:, None] * $(0) + offs_d[None, :]) < $(D)
  q_mask     := tl.logical_and(q_seq_mask, q_d_mask)
  q          := tl.load($(qReg) + q_ptrs, mask=q_mask, other=0)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(Bd)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
    v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
    kv_seq_mask := (offs_n[:, None] + offs_d[None, :] * $(0)) < $(S_k)
    kv_d_mask   := (offs_n[:, None] * $(0) + offs_d[None, :]) < $(D)
    kv_mask     := tl.logical_and(kv_seq_mask, kv_d_mask)
    k       := tl.load($(kReg) + k_ptrs, mask=kv_mask, other=0)
    v       := tl.load($(vReg) + v_ptrs, mask=kv_mask, other=0)

    scores_raw := tl.dot(q, tl.trans(k)) * $ℝ(scale)
    causal     := offs_m[:, None] >= offs_n[None, :]
    causal_scores := tl.where(causal, scores_raw, -inf)
    score_mask := (offs_m[:, None] * $(0) + offs_n[None, :]) < $(S_k)
    scores     := tl.where(score_mask, causal_scores, -inf)
    m_block    := tl.max(scores, axis = 1)
    m_new      := tl.max(m_i, m_block)
    alpha      := tl.exp(m_i - m_new)
    p          := tl.exp(scores - m_new[:, None])
    l_new      := alpha * l_i + tl.sum(p, axis = 1)
    o_acc      := alpha[:, None] * o_acc + tl.dot(p, v)
    m_i        := m_new
    l_i        := l_new
  }

  out    := o_acc / l_i[:, None]
  o_ptrs := o_base_off + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_od)
  o_seq_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  o_d_mask   := (offs_m[:, None] * $(0) + offs_d[None, :]) < $(D)
  o_mask     := tl.logical_and(o_seq_mask, o_d_mask)
  tl.store($(outReg) + o_ptrs, out, mask=o_mask)
}

/-! ## Naive single-block reference kernels

These kernels are the Step 4 baseline: each program instance computes one
Q-block output directly as `softmax(QKᵀ * scale) · V` over the whole logical
KV sequence. There is no KV-block loop and no online-softmax recurrence.
The D-tail variants keep the production-style `Bd` tile width while masking
logical lanes `d >= D`.
-/

def fa1NaiveForwardKernelStridedBoundaryD
    (qReg kReg vReg outReg : RegionName)
    (M Bd S_q S_k D : Nat)
    -- Q strides (axes [B, H, S_q, D]):
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    -- K strides (axes [B, H, S_k, D]):
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    -- V strides (axes [B, H, S_k, D]):
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    -- Output strides (axes [B, H, S_q, D]):
    (stride_ob stride_oh stride_om stride_od : Nat)
    (scale : ℝ) : Kernel := triton {
  pid_qb := tl.program_id(0)
  pid_h  := tl.program_id(1)
  pid_b  := tl.program_id(2)

  q_base_off := pid_b * $(stride_qb) + pid_h * $(stride_qh)
  k_base_off := pid_b * $(stride_kb) + pid_h * $(stride_kh)
  v_base_off := pid_b * $(stride_vb) + pid_h * $(stride_vh)
  o_base_off := pid_b * $(stride_ob) + pid_h * $(stride_oh)

  offs_m := pid_qb * $(M) + tl.arange(0, $(M))
  offs_n := tl.arange(0, $(S_k))
  offs_d := tl.arange(0, $(Bd))

  q_ptrs := q_base_off + offs_m[:, None] * $(stride_qs) + offs_d[None, :] * $(stride_qd)
  q_seq_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  q_d_mask   := (offs_m[:, None] * $(0) + offs_d[None, :]) < $(D)
  q_mask     := tl.logical_and(q_seq_mask, q_d_mask)
  q          := tl.load($(qReg) + q_ptrs, mask=q_mask, other=0)

  k_ptrs := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
  v_ptrs := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
  kv_d_mask := (offs_n[:, None] * $(0) + offs_d[None, :]) < $(D)
  k      := tl.load($(kReg) + k_ptrs, mask=kv_d_mask, other=0)
  v      := tl.load($(vReg) + v_ptrs, mask=kv_d_mask, other=0)

  scores := tl.dot(q, tl.trans(k)) * $ℝ(scale)
  p      := tl.exp(scores)
  l      := tl.sum(p, axis = 1)
  o_acc  := tl.dot(p, v)
  out    := o_acc / l[:, None]

  o_ptrs := o_base_off + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_od)
  o_seq_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  o_d_mask   := (offs_m[:, None] * $(0) + offs_d[None, :]) < $(D)
  o_mask     := tl.logical_and(o_seq_mask, o_d_mask)
  tl.store($(outReg) + o_ptrs, out, mask=o_mask)
}

def fa1NaiveForwardKernelStridedCausalBoundaryD
    (qReg kReg vReg outReg : RegionName)
    (M Bd S_q S_k D : Nat)
    -- Q strides (axes [B, H, S_q, D]):
    (stride_qb stride_qh stride_qs stride_qd : Nat)
    -- K strides (axes [B, H, S_k, D]):
    (stride_kb stride_kh stride_kn stride_kd : Nat)
    -- V strides (axes [B, H, S_k, D]):
    (stride_vb stride_vh stride_vn stride_vd : Nat)
    -- Output strides (axes [B, H, S_q, D]):
    (stride_ob stride_oh stride_om stride_od : Nat)
    (scale : ℝ) : Kernel := triton {
  pid_qb := tl.program_id(0)
  pid_h  := tl.program_id(1)
  pid_b  := tl.program_id(2)

  q_base_off := pid_b * $(stride_qb) + pid_h * $(stride_qh)
  k_base_off := pid_b * $(stride_kb) + pid_h * $(stride_kh)
  v_base_off := pid_b * $(stride_vb) + pid_h * $(stride_vh)
  o_base_off := pid_b * $(stride_ob) + pid_h * $(stride_oh)

  offs_m := pid_qb * $(M) + tl.arange(0, $(M))
  offs_n := tl.arange(0, $(S_k))
  offs_d := tl.arange(0, $(Bd))

  q_ptrs := q_base_off + offs_m[:, None] * $(stride_qs) + offs_d[None, :] * $(stride_qd)
  q_seq_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  q_d_mask   := (offs_m[:, None] * $(0) + offs_d[None, :]) < $(D)
  q_mask     := tl.logical_and(q_seq_mask, q_d_mask)
  q          := tl.load($(qReg) + q_ptrs, mask=q_mask, other=0)

  k_ptrs := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
  v_ptrs := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
  kv_d_mask := (offs_n[:, None] * $(0) + offs_d[None, :]) < $(D)
  k      := tl.load($(kReg) + k_ptrs, mask=kv_d_mask, other=0)
  v      := tl.load($(vReg) + v_ptrs, mask=kv_d_mask, other=0)

  scores_raw := tl.dot(q, tl.trans(k)) * $ℝ(scale)
  causal     := offs_m[:, None] >= offs_n[None, :]
  scores     := tl.where(causal, scores_raw, -inf)
  p          := tl.exp(scores)
  l          := tl.sum(p, axis = 1)
  o_acc      := tl.dot(p, v)
  out        := o_acc / l[:, None]

  o_ptrs := o_base_off + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_od)
  o_seq_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  o_d_mask   := (offs_m[:, None] * $(0) + offs_d[None, :]) < $(D)
  o_mask     := tl.logical_and(o_seq_mask, o_d_mask)
  tl.store($(outReg) + o_ptrs, out, mask=o_mask)
}

/-! ## Math model — softmax-attention

Spec layer is ℝ-valued: `BlockState.mem` only reads ℝ, never `⊥`, so
the natural type of a `tl.load`-fed kernel input is
`TileIndex shape → ℝ`. We lift to `Tile .real` (whose carrier is
`WithBot ℝ`) only as an internal staging step — `Tile.ofReal` /
`unbotD 0` round-trip — to reuse the existing `Tile.dot` /
`Tile.transpose` machinery. The `⊥` sentinel is reserved for things
like `-inf` / `tl.full(_, -inf)` / masked-off lanes that arise
*inside* a kernel, not at its inputs/outputs. -/

/-- Row-wise softmax along the trailing axis on a `Tile .real`. (Math
reference — no row-max subtraction; correctness is what matters at
the spec layer.) -/
noncomputable def softmaxRow {M N : Nat} (s : Tile .real [M, N]) :
    Tile .real [M, N] :=
  ⟨fun (m, n, _) =>
    let row := fun j : Fin N => (s.data (m, j, PUnit.unit)).unbotD 0
    let num := Real.exp (row n)
    let denom := Finset.univ.sum (fun j : Fin N => Real.exp (row j))
    some (num / denom)⟩

/-- Internal `Tile`-level attention helper. Takes `Tile .real` operands
and reuses `Tile.dot` / `Tile.transpose`. The user-facing spec is
`attentionReal` below; this helper exists so the proof can still pivot
through tile-level lemmas. -/
noncomputable def attention {M S D : Nat}
    (Q : Tile .real [M, D]) (K V : Tile .real [S, D])
    (scale : ℝ) : Tile .real [M, D] :=
  let qkT : Tile .real [M, S] :=
    Tile.dot [] Q (Tile.transpose [] K)
  let scaled : Tile .real [M, S] :=
    ⟨fun idx => Option.map (· * scale) (qkT.data idx)⟩
  let p : Tile .real [M, S] := softmaxRow scaled
  Tile.dot [] p V

/-- ℝ-valued reference attention: `softmax(Q · Kᵀ · scale) · V` on
plain `TileIndex → ℝ` inputs. Lifts through `Tile.ofReal`, runs
`attention`, projects back via `unbotD 0`. This is what
`fa1_forward_correct` compares the kernel against. -/
noncomputable def attentionReal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun idx =>
    ((attention (Tile.ofReal Q) (Tile.ofReal K) (Tile.ofReal V)
        scale).data idx).unbotD 0

/-! ## Causal math model

The causal spec is written directly over `ℝ` instead of routing through
`Tile .real`: masked scores are semantically `-inf`, hence contribute
zero to `exp(score)`. Using `WithBot.unbotD 0` in the generic
`softmaxRow` helper would be wrong for that case, because it would turn
`-inf` into the real number `0` before exponentiation.
-/

/-- Causal attention for one 2D slice. A key position `j` contributes to
query position `i` exactly when `j ≤ i`; future keys contribute zero
softmax mass. -/
noncomputable def attentionRealCausal {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let score := fun j : Fin S =>
      scale * Finset.univ.sum (fun d' : Fin D =>
        Q (i, d', PUnit.unit) * K (j, d', PUnit.unit))
    let weight := fun j : Fin S =>
      if j.val ≤ i.val then Real.exp (score j) else 0
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S =>
      weight j * V (j, d, PUnit.unit))
    numer / denom

/-- Causal attention for a local Q block whose row `i` corresponds to
global query row `qStart + i`. This is the spec shape that matches the
strided causal kernel: `offs_m = pid_qb * M + arange(M)` is global,
while the theorem observes only the local `[M, D]` output tile. -/
noncomputable def attentionRealCausalBlock {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let score := fun j : Fin S =>
      scale * Finset.univ.sum (fun d' : Fin D =>
        Q (i, d', PUnit.unit) * K (j, d', PUnit.unit))
    let weight := fun j : Fin S =>
      if j.val ≤ qStart + i.val then Real.exp (score j) else 0
    let denom := Finset.univ.sum (fun j : Fin S => weight j)
    let numer := Finset.univ.sum (fun j : Fin S =>
      weight j * V (j, d, PUnit.unit))
    numer / denom

/-! ## 4D layout — `[B, H, S, D]` reference attention

Step 1 of the FA-1 realism roadmap (issue #39) lifts Q/K/V/O to real
Triton 4D layout `[B, H, S, D]`. The math spec stays simple: the per-
`(batch, head)` slice computes ordinary 2D `attentionReal`, and the 4D
spec is "do that on every slice". The kernel under verification only
touches a single `(b, h, q_block)` slot per program instance, so the
slice helper is also the natural pivot for the correctness proof. -/

/-- Slice a 4D `[B, H, S, D]` tile-as-function at fixed `(batch, head)`,
yielding a 2D `[S, D]` tile-as-function. Used to thread the 4D spec
through the existing 2D `attentionReal`.

Specialized to 2 leading axes (FA-1's batch + head). A general
ND-prefix slicer (`TileIndex (prefix ++ rest) → ℝ → TileIndex rest → ℝ`)
would require `TileIndex` append helpers; deferred until a kernel
needs more than two leading axes (e.g. grouped attention with an
extra group axis). -/
def sliceBH {B H S D : Nat}
    (T : TileIndex [B, H, S, D] → ℝ)
    (b : Fin B) (h : Fin H) : TileIndex [S, D] → ℝ :=
  fun (i, d, _) => T (b, h, i, d, PUnit.unit)

/-- 4D ℝ-valued reference attention. Each `(batch, head)` slice is
independent and computes the ordinary 2D `attentionReal`. The kernel
spec for Step 1 will only assert this on the single `(b, h, q_block)`
slot determined by the three `program_id` axes. -/
noncomputable def attentionReal4D {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionReal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

/-- 4D ℝ-valued causal reference attention. As with
`attentionReal4D`, each `(batch, head)` slice is independent; the only
difference is the per-row causal restriction `key ≤ query`. -/
noncomputable def attentionReal4DCausal {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) =>
    attentionRealCausal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
      scale (i, d, PUnit.unit)

/-- `attentionReal4D` projects to `attentionReal` on each `(b, h)` slice.
The natural bridge from the 4D spec to the existing 2D correctness
machinery; the Step 1 correctness proof reduces 4D ↦ 2D via this
lemma and then reuses `streaming_eq_attentionReal` unchanged. -/
@[simp] theorem attentionReal4D_slice {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4D Q K V scale (b, h, i, d, PUnit.unit)
      = attentionReal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          scale (i, d, PUnit.unit) := rfl

/-- `attentionReal4DCausal` projects to `attentionRealCausal` on each
`(b, h)` slice. -/
@[simp] theorem attentionReal4DCausal_slice {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (b : Fin B) (h : Fin H) (i : Fin S_q) (d : Fin D) :
    attentionReal4DCausal Q K V scale (b, h, i, d, PUnit.unit)
      = attentionRealCausal (sliceBH Q b h) (sliceBH K b h) (sliceBH V b h)
          scale (i, d, PUnit.unit) := rfl

/-- M-row slice of the `(b, h)` plane of a 4D tensor: pick `M`
consecutive rows starting at `start * M`. The boundary
`start * M + M ≤ S` ensures every row index fits.

This is the natural Q-input view that FA-1's strided kernel sees:
each program-instance `(b, h, qb)` reads the `M` rows
`[qb*M, qb*M+1, ..., qb*M+M-1]` of `Q4D`'s `(b, h)` plane. -/
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
Out-of-range hidden lanes are zero, matching `tl.load(..., other=0)` under
the D-tail mask. -/
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

/-- Reinterpret the `(b, h)` plane of a 4D `[B, H, S, D]` tensor as a
flat `[Bk * numKVBlocks, D]` view, given `Bk * numKVBlocks = S`. The K
and V inputs of FA-1 take this form: the kernel iterates over
`numKVBlocks` blocks of `Bk` rows each, covering all `S` rows of the
`(b, h)` plane. -/
def slice4DFlat {B H S D : Nat} (Bk numKVBlocks : Nat)
    (T : TileIndex [B, H, S, D] → ℝ) (b : Fin B) (h : Fin H)
    (hSk : Bk * numKVBlocks = S) : TileIndex [Bk * numKVBlocks, D] → ℝ :=
  fun (j, d, _) =>
    T (b, h, ⟨j.val, by
      have := j.isLt
      omega⟩, d, PUnit.unit)

/-! ## User-facing 4D layout wrapper

The low-level strided FA-1 theorems expose every stride as a separate
argument. That is useful for proof reuse, but awkward as a public theorem
surface. `FA1Layout4D` bundles the Q/K/V/O strides for `[B, H, S, D]`
style tensors, plus the output-layout validity proof needed to turn the
output offset expression into an injective scatter/readback map.

The wrapper theorems near the end of the file are thin corollaries over
`fa1_forward_correct_4D` / `fa1_forward_correct_4D_causal`; they keep the
same semantics while giving users one layout argument instead of sixteen
stride arguments plus a separate `hOValid`. The input memory contracts are
exposed through `TensorView.loaded`, so theorem users can talk in tensor-view
metadata rather than raw `InputAt` / `Offset.strided` terms. -/

/-- Stride bundle for FA-1 over 4D `[B, H, S, D]` Q/K/V/O tensors.

Fields `q*`, `k*`, `v*`, and `o*` are the batch/head/sequence-or-output-row
/ feature-dimension strides for Q, K, V, and output respectively. -/
structure FA1Layout4D (B H S_q S_k D : Nat) where
  qB : Nat
  qH : Nat
  qS : Nat
  qD : Nat
  kB : Nat
  kH : Nat
  kS : Nat
  kD : Nat
  vB : Nat
  vH : Nat
  vS : Nat
  vD : Nat
  oB : Nat
  oH : Nat
  oS : Nat
  oD : Nat
  hOValid : Offset.StridesValid [B, H, S_q, D] [oB, oH, oS, oD]

namespace FA1Layout4D

def qStrides (layout : FA1Layout4D B H S_q S_k D) : List Nat :=
  [layout.qB, layout.qH, layout.qS, layout.qD]

def kStrides (layout : FA1Layout4D B H S_q S_k D) : List Nat :=
  [layout.kB, layout.kH, layout.kS, layout.kD]

def vStrides (layout : FA1Layout4D B H S_q S_k D) : List Nat :=
  [layout.vB, layout.vH, layout.vS, layout.vD]

def oStrides (layout : FA1Layout4D B H S_q S_k D) : List Nat :=
  [layout.oB, layout.oH, layout.oS, layout.oD]

/-- Tensor view for Q under this FA-1 layout. -/
def qView (layout : FA1Layout4D B H S_q S_k D)
    (qReg : RegionName) : TensorView [B, H, S_q, D] :=
  { region := qReg, base := 0, strides := layout.qStrides }

/-- Tensor view for K under this FA-1 layout. -/
def kView (layout : FA1Layout4D B H S_q S_k D)
    (kReg : RegionName) : TensorView [B, H, S_k, D] :=
  { region := kReg, base := 0, strides := layout.kStrides }

/-- Tensor view for V under this FA-1 layout. -/
def vView (layout : FA1Layout4D B H S_q S_k D)
    (vReg : RegionName) : TensorView [B, H, S_k, D] :=
  { region := vReg, base := 0, strides := layout.vStrides }

/-- Tensor view for output under this FA-1 layout. -/
def oView (layout : FA1Layout4D B H S_q S_k D)
    (outReg : RegionName) : TensorView [B, H, S_q, D] :=
  { region := outReg, base := 0, strides := layout.oStrides }

/-- Full 4D Q offset for an input tensor. -/
def qOffset (layout : FA1Layout4D B H S_q S_k D) :
    TileIndex [B, H, S_q, D] → Nat :=
  Offset.strided [B, H, S_q, D] layout.qStrides 0

/-- Full 4D K offset for an input tensor. -/
def kOffset (layout : FA1Layout4D B H S_q S_k D) :
    TileIndex [B, H, S_k, D] → Nat :=
  Offset.strided [B, H, S_k, D] layout.kStrides 0

/-- Full 4D V offset for an input tensor. -/
def vOffset (layout : FA1Layout4D B H S_q S_k D) :
    TileIndex [B, H, S_k, D] → Nat :=
  Offset.strided [B, H, S_k, D] layout.vStrides 0

/-- Output offset for the local `[M, D]` tile written by the current
`BlockState.pids` program instance. -/
def outBlockOffset (layout : FA1Layout4D B H S_q S_k D)
    (s : BlockState) (M : Nat) : TileIndex [M, D] → Nat :=
  fun idx =>
    s.pids 2 * layout.oB + s.pids 1 * layout.oH + s.pids 0 * M * layout.oS
      + idx.1.val * layout.oS + idx.2.1.val * layout.oD

/-- Output offset for a block-width hidden tile `[M, Bd]`; only lanes
`d < D` are written by D-tail kernels. -/
def outBlockOffsetD (layout : FA1Layout4D B H S_q S_k D)
    (s : BlockState) (M Bd : Nat) : TileIndex [M, Bd] → Nat :=
  fun idx =>
    s.pids 2 * layout.oB + s.pids 1 * layout.oH + s.pids 0 * M * layout.oS
      + idx.1.val * layout.oS + idx.2.1.val * layout.oD

def kernel (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  fa1ForwardKernelStrided qReg kReg vReg outReg M D Bk numKVBlocks
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def causalKernel (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  fa1ForwardKernelStridedCausal qReg kReg vReg outReg M D Bk numKVBlocks
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def boundaryKernel (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  fa1ForwardKernelStridedBoundary qReg kReg vReg outReg M D Bk numKVBlocks S_q S_k
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def causalBoundaryKernel (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  fa1ForwardKernelStridedCausalBoundary qReg kReg vReg outReg
    M D Bk numKVBlocks S_q S_k
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def boundaryKernelD (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bd Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  fa1ForwardKernelStridedBoundaryD qReg kReg vReg outReg
    M Bd Bk numKVBlocks S_q S_k D
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def causalBoundaryKernelD (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bd Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  fa1ForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
    M Bd Bk numKVBlocks S_q S_k D
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def naiveBoundaryKernelD (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bd : Nat) (scale : ℝ) : Kernel :=
  fa1NaiveForwardKernelStridedBoundaryD qReg kReg vReg outReg
    M Bd S_q S_k D
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

def naiveCausalBoundaryKernelD (layout : FA1Layout4D B H S_q S_k D)
    (qReg kReg vReg outReg : RegionName)
    (M Bd : Nat) (scale : ℝ) : Kernel :=
  fa1NaiveForwardKernelStridedCausalBoundaryD qReg kReg vReg outReg
    M Bd S_q S_k D
    layout.qB layout.qH layout.qS layout.qD
    layout.kB layout.kH layout.kS layout.kD
    layout.vB layout.vH layout.vS layout.vD
    layout.oB layout.oH layout.oS layout.oD scale

end FA1Layout4D

/-! ## View-level theorem surface

`FA1Layout4D` removes the raw stride arguments, but still keeps the
Q/K/V/O region names separate. `FA1Views4D` is the preferred public
surface: it bundles the logical layout together with the concrete memory
regions and exposes the four resulting `TensorView`s directly. -/

/-- Full memory contract bundle for one FA-1 call over 4D
`[B, H, S, D]` tensors. -/
structure FA1Views4D (B H S_q S_k D : Nat) where
  layout : FA1Layout4D B H S_q S_k D
  qReg : RegionName
  kReg : RegionName
  vReg : RegionName
  outReg : RegionName

namespace FA1Views4D

def qView (views : FA1Views4D B H S_q S_k D) : TensorView [B, H, S_q, D] :=
  views.layout.qView views.qReg

def kView (views : FA1Views4D B H S_q S_k D) : TensorView [B, H, S_k, D] :=
  views.layout.kView views.kReg

def vView (views : FA1Views4D B H S_q S_k D) : TensorView [B, H, S_k, D] :=
  views.layout.vView views.vReg

def oView (views : FA1Views4D B H S_q S_k D) : TensorView [B, H, S_q, D] :=
  views.layout.oView views.outReg

def outBlockOffset (views : FA1Views4D B H S_q S_k D)
    (s : BlockState) (M : Nat) : TileIndex [M, D] → Nat :=
  views.layout.outBlockOffset s M

def outBlockOffsetD (views : FA1Views4D B H S_q S_k D)
    (s : BlockState) (M Bd : Nat) : TileIndex [M, Bd] → Nat :=
  views.layout.outBlockOffsetD s M Bd

def kernel (views : FA1Views4D B H S_q S_k D)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  views.layout.kernel views.qReg views.kReg views.vReg views.outReg
    M Bk numKVBlocks scale

def causalKernel (views : FA1Views4D B H S_q S_k D)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  views.layout.causalKernel views.qReg views.kReg views.vReg views.outReg
    M Bk numKVBlocks scale

def boundaryKernel (views : FA1Views4D B H S_q S_k D)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  views.layout.boundaryKernel views.qReg views.kReg views.vReg views.outReg
    M Bk numKVBlocks scale

def causalBoundaryKernel (views : FA1Views4D B H S_q S_k D)
    (M Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  views.layout.causalBoundaryKernel views.qReg views.kReg views.vReg views.outReg
    M Bk numKVBlocks scale

def boundaryKernelD (views : FA1Views4D B H S_q S_k D)
    (M Bd Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  views.layout.boundaryKernelD views.qReg views.kReg views.vReg views.outReg
    M Bd Bk numKVBlocks scale

def causalBoundaryKernelD (views : FA1Views4D B H S_q S_k D)
    (M Bd Bk numKVBlocks : Nat) (scale : ℝ) : Kernel :=
  views.layout.causalBoundaryKernelD views.qReg views.kReg views.vReg views.outReg
    M Bd Bk numKVBlocks scale

def naiveBoundaryKernelD (views : FA1Views4D B H S_q S_k D)
    (M Bd : Nat) (scale : ℝ) : Kernel :=
  views.layout.naiveBoundaryKernelD views.qReg views.kReg views.vReg views.outReg
    M Bd scale

def naiveCausalBoundaryKernelD (views : FA1Views4D B H S_q S_k D)
    (M Bd : Nat) (scale : ℝ) : Kernel :=
  views.layout.naiveCausalBoundaryKernelD views.qReg views.kReg views.vReg views.outReg
    M Bd scale

end FA1Views4D

/-! ## Streaming math model (FA-1 online softmax recurrence)

Mirrors what `fa1ForwardKernel` computes block by block. Three running
quantities, each parameterized by the number of completed KV blocks
`k ∈ [0, numKVBlocks]`:

* `mPartial` — running per-row max of scaled scores. Uses `WithBot ℝ`
  (with seed `⊥` at `k = 0`) so the streaming `max` matches the
  kernel's `m_i := tl.full([M], -inf)` initialization exactly.
* `lPartial` — running per-row normalizer (sum of `exp(score - m)`).
  Real-valued (seed `0` at `k = 0`).
* `oPartial` — running unnormalized output (`Σ exp(score - m) · V`).
  Real-valued (seed `0` at `k = 0`).

The final theorem `streaming_eq_attentionReal` (math identity, no
kernel) proves that
`oPartial numKVBlocks idx / lPartial numKVBlocks i = attentionReal idx`,
which the kernel's post-loop `out := o_acc / l_i[:, None]` realizes. -/

namespace FA1Math

/-- Scaled score `(Q · Kᵀ · scale)[i, j]` at a flat KV index `j ∈ [0, S)`.
The streaming form indexes K/V by `k * Bk + jLocal` for the k-th block;
this helper normalizes that to a single `j` so the recurrence can sum
over the whole input length uniformly. -/
noncomputable def scaledScore {M S D : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) : ℝ :=
  scale * Finset.univ.sum (fun d : Fin D =>
    Q (i, d, PUnit.unit) * K (j, d, PUnit.unit))

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
  have hScore : ∀ j : Fin S,
      scale * (Finset.univ.sum fun d' : Fin Bd =>
        padHeadD Q (i, d', PUnit.unit) * padHeadD K (j, d', PUnit.unit)) =
      scale * (Finset.univ.sum fun d' : Fin D =>
        Q (i, d', PUnit.unit) * K (j, d', PUnit.unit)) := by
    intro j
    simpa [scaledScore] using
      scaledScore_padHeadD_eq hDLe Q K scale i j
  have hV : ∀ j : Fin S,
      padHeadD V (j, d, PUnit.unit) = V (j, ⟨d.val, hD⟩, PUnit.unit) := by
    intro j
    simp [padHeadD, hD]
  simp [hScore, hV]

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
  have hScore : ∀ j : Fin S,
      scale * (Finset.univ.sum fun d' : Fin Bd =>
        padHeadD Q (i, d', PUnit.unit) * padHeadD K (j, d', PUnit.unit)) =
      scale * (Finset.univ.sum fun d' : Fin D =>
        Q (i, d', PUnit.unit) * K (j, d', PUnit.unit)) := by
    intro j
    simpa [scaledScore] using
      scaledScore_padHeadD_eq hDLe Q K scale i j
  have hV : ∀ j : Fin S,
      padHeadD V (j, d, PUnit.unit) = V (j, ⟨d.val, hD⟩, PUnit.unit) := by
    intro j
    simp [padHeadD, hD]
  simp [hScore, hV]

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
  have hScore : ∀ j : Fin S,
      (∑ x : Fin Bd, padHeadD Q (i, x, PUnit.unit) * padHeadD K (j, x, PUnit.unit)) =
      (∑ x : Fin D, Q (i, x, PUnit.unit) * K (j, x, PUnit.unit)) := by
    intro j
    rw [show (Finset.univ : Finset (Fin Bd)).sum
          (fun x : Fin Bd => padHeadD Q (i, x, PUnit.unit) * padHeadD K (j, x, PUnit.unit)) =
        (Finset.univ : Finset (Fin Bd)).sum
          (fun x : Fin Bd =>
            if h : x.val < D then
              Q (i, ⟨x.val, h⟩, PUnit.unit) * K (j, ⟨x.val, h⟩, PUnit.unit)
            else
              0) by
      apply Finset.sum_congr rfl
      intro x _
      by_cases hx : x.val < D
      · simp [padHeadD, hx]
      · simp [padHeadD, hx]]
    rw [sum_padHeadD_eq hDLe (fun x : Fin D =>
      Q (i, x, PUnit.unit) * K (j, x, PUnit.unit))]
  have hV : ∀ j : Fin S,
      padHeadD V (j, d, PUnit.unit) = V (j, ⟨d.val, hD⟩, PUnit.unit) := by
    intro j
    simp [padHeadD, hD]
  unfold attentionReal attention softmaxRow
  simp [Tile.dot, Tile.transpose, Tile.ofReal, hScore, hV]

/-- Flat KV index of the `jLocal`-th lane of the `k`-th block. Factored
out as a non-dependent helper so `Finset.le_sup`-style lemmas can
unify on it without tripping over the inline proof. -/
def blockIndex (Bk numKVBlocks k : Nat) (h : k + 1 ≤ numKVBlocks)
    (jLocal : Fin Bk) : Fin (Bk * numKVBlocks) :=
  ⟨k * Bk + jLocal.val, by
    have hk : k < numKVBlocks := by omega
    calc k * Bk + jLocal.val
        < k * Bk + Bk := by omega
      _ = (k + 1) * Bk := by ring
      _ ≤ numKVBlocks * Bk := Nat.mul_le_mul_right Bk (by omega)
      _ = Bk * numKVBlocks := by ring⟩

/-- Running per-row max of scaled scores over the first `k` KV blocks.
Seeded at `⊥` so the `max` is right at `k = 0` (no scores seen).
Closely mirrors `onlineSoftmaxM` from `Examples/OnlineSoftmax.lean`. -/
noncomputable def mPartial {M D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0,     _ => (⊥ : WithBot ℝ)
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        max (mPartial Bk Q numKVBlocks K scale k i)
          (Finset.univ.sup (fun jLocal : Fin Bk =>
            ((scaledScore Q K scale i (blockIndex Bk numKVBlocks k h jLocal)
              : ℝ) : WithBot ℝ)))
      else
        mPartial Bk Q numKVBlocks K scale k i

/-- The α multiplier `exp(m_k - m_{k+1})` produced by FA-1's online
softmax. At `k = 0` this collapses to `0` (since `m_0 = ⊥` and
`exp ⊥ = 0`), zeroing out the first iteration's contribution from the
empty seed. -/
noncomputable def alphaPartial {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  ((WithBot.realExp
      (WithBot.realSub
        (mPartial Bk Q numKVBlocks K scale k i)
      (mPartial Bk Q numKVBlocks K scale (k + 1) i)))).unbotD 0

/-- `alphaPartial` is exactly the real payload of the operational
`exp(m_i - m_new)` expression. The result is always a real `some`: when the
subtraction sees `⊥`, `WithBot.realExp ⊥ = 0`. -/
theorem alphaPartial_toWithBot {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) :
    WithBot.realExp
      (WithBot.realSub
        (mPartial Bk Q numKVBlocks K scale k i)
        (mPartial Bk Q numKVBlocks K scale (k + 1) i))
      =
      ((alphaPartial Q numKVBlocks K scale k i : ℝ) : WithBot ℝ) := by
  unfold alphaPartial
  cases h :
      WithBot.realSub
        (mPartial Bk Q numKVBlocks K scale k i)
        (mPartial Bk Q numKVBlocks K scale (k + 1) i) with
  | bot =>
      simp
  | coe a =>
      simp

/-- Running per-row normalizer `Σ exp(score - m)` over the first `k`
KV blocks. -/
noncomputable def lPartial {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → ℝ
  | 0,     _ => 0
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
        alphaPartial Q numKVBlocks K scale k i *
          lPartial Q numKVBlocks K scale k i +
          Finset.univ.sum (fun jLocal : Fin Bk =>
            Real.exp (scaledScore Q K scale i
                (blockIndex Bk numKVBlocks k h jLocal) - mNew))
      else
        lPartial Q numKVBlocks K scale k i

/-- Running unnormalized per-row output `Σ exp(score - m) · V` over
the first `k` KV blocks. -/
noncomputable def oPartial {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, D] → ℝ
  | 0,     _ => 0
  | k + 1, idx =>
      let i := idx.1
      let d := idx.2.1
      if h : k + 1 ≤ numKVBlocks then
        let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
        alphaPartial Q numKVBlocks K scale k i *
          oPartial Q numKVBlocks K V scale k idx +
          Finset.univ.sum (fun jLocal : Fin Bk =>
            Real.exp (scaledScore Q K scale i
                (blockIndex Bk numKVBlocks k h jLocal) - mNew) *
              V (blockIndex Bk numKVBlocks k h jLocal, d, PUnit.unit))
      else
        oPartial Q numKVBlocks K V scale k idx

/-! ### Stage A — m-free reference sums (`lFree`, `oFree`)

Reference sums *without* the `m` shift, used to bridge the streaming
form (which subtracts `m_k` for numerical stability) to `attentionReal`
(which doesn't). The key identity:

```text
lPartial k i = exp(-m_k) · lFree k i        (m-shift factors out)
oPartial k i d = exp(-m_k) · oFree k i d
```

so the ratio `oPartial / lPartial = oFree / lFree`, which matches
`attentionReal` directly. -/

/-- Σ over the first `k` KV blocks of `exp(scaledScore)`, no `m`-shift. -/
noncomputable def lFree {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ N) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    Real.exp (scaledScore Q K scale i
      (blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL))))

/-- Σ over the first `k` KV blocks of `exp(scaledScore) · V`, no `m`-shift. -/
noncomputable def oFree {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ N) (idx : TileIndex [M, D]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    Real.exp (scaledScore Q K scale i
      (blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL)) *
    V (blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL, d, PUnit.unit)))

/-- Recurrence: `lFree (k+1)` adds the next block's sum on top of `lFree k`.
Uses `Fin.sum_univ_castSucc` to peel the last index off the outer sum;
the remaining `Fin k` sum matches `lFree k` modulo proof irrelevance in
the `blockIndex` proof argument, and the peeled-off term reduces via
`Fin.val_last`. -/
theorem lFree_succ {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ N) (i : Fin M) :
    lFree Q K scale (k + 1) hk i =
      lFree Q K scale k (Nat.le_of_succ_le hk) i +
      Finset.univ.sum (fun jL : Fin Bk =>
        Real.exp (scaledScore Q K scale i (blockIndex Bk N k hk jL))) := by
  unfold lFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Recurrence companion: `oFree (k+1)` adds the next block's `exp · V` sum
on top of `oFree k`. -/
theorem oFree_succ {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ N) (idx : TileIndex [M, D]) :
    oFree Q K V scale (k + 1) hk idx =
      oFree Q K V scale k (Nat.le_of_succ_le hk) idx +
      Finset.univ.sum (fun jL : Fin Bk =>
        Real.exp (scaledScore Q K scale idx.1 (blockIndex Bk N k hk jL)) *
        V (blockIndex Bk N k hk jL, idx.2.1, PUnit.unit)) := by
  unfold oFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- `lFree 0 = 0` (empty sum). -/
@[simp] theorem lFree_zero {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lFree Q K scale 0 (Nat.zero_le _) i = 0 := by
  unfold lFree
  simp

/-- `oFree 0 = 0` (empty sum). -/
@[simp] theorem oFree_zero {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oFree Q K V scale 0 (Nat.zero_le _) idx = 0 := by
  unfold oFree
  simp

/-! ### Stage A foundations — `mPartial` non-`⊥` for `k ≥ 1`

The streaming algebra (α-cancellation in `lPartial` / `oPartial`)
needs `mPartial k` to be a real number, not `⊥`, whenever any block
has been seen. With `0 < Bk`, the block-max `Finset.univ.sup` over
`Fin Bk` is non-`⊥`, so `mPartial 1` is non-`⊥` and the property
propagates upward through `max`. -/

theorem mPartial_succ_ne_bot {M D : Nat} {Bk : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ (⊥ : WithBot ℝ) := by
  unfold mPartial
  simp only [hk, ↓reduceDIte]
  -- Result is `max prev (Finset.sup f)` where `f j = some (scaledScore _)`.
  -- Pick j₀ = ⟨0, hBk⟩; sup ≥ f j₀ = some _, so sup ≠ ⊥, so max ≠ ⊥.
  set f : Fin Bk → WithBot ℝ := fun jLocal =>
    ((scaledScore Q K scale i (blockIndex Bk numKVBlocks k hk jLocal) : ℝ)
      : WithBot ℝ)
  have hSup : f ⟨0, hBk⟩ ≤ Finset.univ.sup f :=
    Finset.le_sup (Finset.mem_univ _)
  intro hMaxBot
  have hSupBot : Finset.univ.sup f ≤ (⊥ : WithBot ℝ) := hMaxBot ▸ le_max_right _ _
  have : f ⟨0, hBk⟩ = ⊥ :=
    le_antisymm (le_trans hSup hSupBot) (OrderBot.bot_le _)
  exact WithBot.coe_ne_bot this

/-- In-bounds recurrence form of `mPartial`; avoids repeatedly unfolding the
guarded definition and discharging `k + 1 ≤ numKVBlocks`. -/
theorem mPartial_succ_of_lt {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    mPartial Bk Q numKVBlocks K scale (k + 1) i =
      max (mPartial Bk Q numKVBlocks K scale k i)
        (Finset.univ.sup (fun jLocal : Fin Bk =>
          ((scaledScore Q K scale i
            (blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal)
            : ℝ) : WithBot ℝ))) := by
  change (if h : k + 1 ≤ numKVBlocks then
      max (mPartial Bk Q numKVBlocks K scale k i)
        (Finset.univ.sup (fun jLocal : Fin Bk =>
          ((scaledScore Q K scale i
            (blockIndex Bk numKVBlocks k h jLocal) : ℝ) : WithBot ℝ)))
    else mPartial Bk Q numKVBlocks K scale k i) = _
  simp [Nat.succ_le_iff.mpr hk]

/-- In-bounds recurrence form of `lPartial`. -/
theorem lPartial_succ_of_lt {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    lPartial Q numKVBlocks K scale (k + 1) i =
      let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
      alphaPartial Q numKVBlocks K scale k i *
        lPartial Q numKVBlocks K scale k i +
        Finset.univ.sum (fun jLocal : Fin Bk =>
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal) - mNew)) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
       alphaPartial Q numKVBlocks K scale k i *
          lPartial Q numKVBlocks K scale k i +
        Finset.univ.sum (fun jLocal : Fin Bk =>
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk numKVBlocks k h jLocal) - mNew)))
    else lPartial Q numKVBlocks K scale k i) = _
  simp [Nat.succ_le_iff.mpr hk]

/-- In-bounds recurrence form of `oPartial`. -/
theorem oPartial_succ_of_lt {M D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Q numKVBlocks K V scale (k + 1) idx =
      let i := idx.1
      let d := idx.2.1
      let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
      alphaPartial Q numKVBlocks K scale k i *
        oPartial Q numKVBlocks K V scale k idx +
        Finset.univ.sum (fun jLocal : Fin Bk =>
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal) - mNew) *
            V (blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal,
              d, PUnit.unit)) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let i := idx.1
       let d := idx.2.1
       let mNew : ℝ := (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
       alphaPartial Q numKVBlocks K scale k i *
          oPartial Q numKVBlocks K V scale k idx +
        Finset.univ.sum (fun jLocal : Fin Bk =>
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk numKVBlocks k h jLocal) - mNew) *
            V (blockIndex Bk numKVBlocks k h jLocal, d, PUnit.unit)))
    else oPartial Q numKVBlocks K V scale k idx) = _
  simp [Nat.succ_le_iff.mpr hk]

/-- The streaming `lPartial k i` equals `exp(-m_k) · lFree k i`, where
`m_k = (mPartial k i).unbotD 0`. By induction on `k`: the α-factor
`exp(m_k - m_{k+1})` absorbs the shift difference at each step. -/
theorem lPartial_eq_mShifted {M D Bk : Nat} (_hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) :
    lPartial Q numKVBlocks K scale k i =
      Real.exp (-(mPartial Bk Q numKVBlocks K scale k i).unbotD 0) *
        lFree Q K scale k hk i := by
  induction k with
  | zero =>
    -- LHS: lPartial 0 i = 0; RHS: exp(-((⊥).unbotD 0)) · lFree 0 = 1 · 0 = 0.
    show (0 : ℝ) =
      Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
        lFree Q K scale 0 hk i
    rw [lFree_zero]
    ring
  | succ k ih =>
    have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
    -- Unfold LHS: αₖ * lPartial k + Σ exp(scaledScore - m_{k+1}).
    unfold lPartial
    simp only [hk, ↓reduceDIte]
    -- Apply IH on the lPartial k subterm.
    rw [ih hk']
    -- Expand RHS via lFree_succ.
    rw [lFree_succ Q K scale k hk i, mul_add]
    -- Goal:
    --   αₖ * (exp(-mₖ) * lFree k _) + Σ_jL exp(scaledScore - m_{k+1})
    --   = exp(-m_{k+1}) * lFree k _ + exp(-m_{k+1}) * Σ_jL exp(scaledScore)
    -- Match each term separately.
    have hExpSub : ∀ s : ℝ,
        Real.exp (s - (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            Real.exp s := by
      intro s
      rw [show s - (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0
            = -(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0 + s by ring,
          Real.exp_add]
    -- Term (b): Σ_jL exp(s - m_{k+1}) = exp(-m_{k+1}) * Σ_jL exp(s).
    have hSumB :
        Finset.univ.sum (fun jL : Fin Bk =>
          Real.exp (scaledScore Q K scale i (blockIndex Bk numKVBlocks k hk jL)
            - (mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0))
        = Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            Finset.univ.sum (fun jL : Fin Bk =>
              Real.exp (scaledScore Q K scale i
                (blockIndex Bk numKVBlocks k hk jL))) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro jL _
      exact hExpSub _
    -- Term (a): αₖ * (exp(-mₖ) * lFree k) = exp(-m_{k+1}) * lFree k.
    -- Case split on k.
    have hSumA :
        alphaPartial Q numKVBlocks K scale k i *
          (Real.exp (-(mPartial Bk Q numKVBlocks K scale k i).unbotD 0) *
            lFree Q K scale k hk' i)
        = Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            lFree Q K scale k hk' i := by
      rcases Nat.eq_zero_or_pos k with hkz | hkpos
      · -- k = 0: lFree 0 _ = 0, so both sides are zero.
        subst hkz
        rw [lFree_zero]
        ring
      · -- k ≥ 1: mPartial k is some real, so αₖ = exp(mₖ - m_{k+1}) and
        -- αₖ · exp(-mₖ) = exp(-m_{k+1}).
        -- Get the underlying reals.
        obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
        -- Now `k = k' + 1`. Have hk : k' + 1 + 1 ≤ numKVBlocks ⇒ k' + 1 ≤ numKVBlocks.
        have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
        have hmk_ne : mPartial Bk Q numKVBlocks K scale (k' + 1) i ≠ ⊥ :=
          mPartial_succ_ne_bot _hBk Q numKVBlocks K scale k' hk_succ i
        have hmk1_ne :
            mPartial Bk Q numKVBlocks K scale (k' + 1 + 1) i ≠ ⊥ :=
          mPartial_succ_ne_bot _hBk Q numKVBlocks K scale (k' + 1) hk i
        -- Extract real values.
        obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
        obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
        -- Compute αₖ.
        have hAlpha :
            alphaPartial Q numKVBlocks K scale (k' + 1) i =
              Real.exp (rk - rk1) := by
          unfold alphaPartial
          rw [← hrk, ← hrk1]
          simp [WithBot.realSub]
        rw [hAlpha, ← hrk, ← hrk1]
        simp only [WithBot.unbotD_coe]
        -- Goal: exp(rk - rk1) * (exp(-rk) * lFree k' k _) = exp(-rk1) * lFree k' k _
        rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
              rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
        rw [show Real.exp (-rk1) * Real.exp rk *
                  (Real.exp (-rk) * lFree Q K scale (k' + 1) hk' i)
              = Real.exp (-rk1) *
                  (Real.exp rk * Real.exp (-rk) * lFree Q K scale (k' + 1) hk' i) by ring]
        rw [show Real.exp rk * Real.exp (-rk) = 1 by
              rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
        ring
    -- Combine.
    linarith [hSumA, hSumB]

/-- Companion identity for `oPartial`: `exp(-m_k) · oFree k i d`. Same
shape as `lPartial_eq_mShifted` plus `· V[j, d]` on each summand. -/
theorem oPartial_eq_mShifted {M D Bk : Nat} (_hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Q numKVBlocks K V scale k idx =
      Real.exp (-(mPartial Bk Q numKVBlocks K scale k idx.1).unbotD 0) *
        oFree Q K V scale k hk idx := by
  induction k with
  | zero =>
    show (0 : ℝ) =
      Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
        oFree Q K V scale 0 hk idx
    rw [oFree_zero]
    ring
  | succ k ih =>
    have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
    unfold oPartial
    simp only [hk, ↓reduceDIte]
    rw [ih hk']
    rw [oFree_succ Q K V scale k hk idx, mul_add]
    -- Match the two terms; identical shape to `lPartial_eq_mShifted` plus `· V`.
    have hExpSub : ∀ s : ℝ,
        Real.exp (s - (mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            Real.exp s := by
      intro s
      rw [show s - (mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0
            = -(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0 + s by ring,
          Real.exp_add]
    have hSumB :
        Finset.univ.sum (fun jL : Fin Bk =>
          Real.exp (scaledScore Q K scale idx.1
              (blockIndex Bk numKVBlocks k hk jL)
            - (mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
          V (blockIndex Bk numKVBlocks k hk jL, idx.2.1, PUnit.unit))
        = Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            Finset.univ.sum (fun jL : Fin Bk =>
              Real.exp (scaledScore Q K scale idx.1
                  (blockIndex Bk numKVBlocks k hk jL)) *
              V (blockIndex Bk numKVBlocks k hk jL, idx.2.1, PUnit.unit)) := by
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro jL _
      rw [hExpSub]
      ring
    have hSumA :
        alphaPartial Q numKVBlocks K scale k idx.1 *
          (Real.exp (-(mPartial Bk Q numKVBlocks K scale k idx.1).unbotD 0) *
            oFree Q K V scale k hk' idx)
        = Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            oFree Q K V scale k hk' idx := by
      rcases Nat.eq_zero_or_pos k with hkz | hkpos
      · subst hkz
        rw [oFree_zero]
        ring
      · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
        have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
        have hmk_ne : mPartial Bk Q numKVBlocks K scale (k' + 1) idx.1 ≠ ⊥ :=
          mPartial_succ_ne_bot _hBk Q numKVBlocks K scale k' hk_succ idx.1
        have hmk1_ne :
            mPartial Bk Q numKVBlocks K scale (k' + 1 + 1) idx.1 ≠ ⊥ :=
          mPartial_succ_ne_bot _hBk Q numKVBlocks K scale (k' + 1) hk idx.1
        obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
        obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
        have hAlpha :
            alphaPartial Q numKVBlocks K scale (k' + 1) idx.1 =
              Real.exp (rk - rk1) := by
          unfold alphaPartial
          rw [← hrk, ← hrk1]
          simp [WithBot.realSub]
        rw [hAlpha, ← hrk, ← hrk1]
        simp only [WithBot.unbotD_coe]
        rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
              rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
        rw [show Real.exp (-rk1) * Real.exp rk *
                  (Real.exp (-rk) * oFree Q K V scale (k' + 1) hk' idx)
              = Real.exp (-rk1) *
                  (Real.exp rk * Real.exp (-rk) * oFree Q K V scale (k' + 1) hk' idx) by ring]
        rw [show Real.exp rk * Real.exp (-rk) = 1 by
              rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
        ring
    linarith [hSumA, hSumB]

/-- Fin (Bk * N) ≃ Fin N × Fin Bk, the bijection underlying
`oFree`/`lFree`'s double-sum form. Composes `finProdFinEquiv.symm` with
a `mul_comm` cast. -/
def blockIndexEquiv (Bk N : Nat) : Fin (Bk * N) ≃ Fin N × Fin Bk :=
  (Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv.trans finProdFinEquiv.symm

/-- Helper: blockIndex round-trips through `blockIndexEquiv`. -/
theorem blockIndex_blockIndexEquiv {Bk N : Nat} (j : Fin (Bk * N)) :
    blockIndex Bk N ((blockIndexEquiv Bk N) j).1.val
        (by have := ((blockIndexEquiv Bk N) j).1.isLt; omega)
        ((blockIndexEquiv Bk N) j).2 = j := by
  apply Fin.ext
  show ((blockIndexEquiv Bk N) j).1.val * Bk +
        ((blockIndexEquiv Bk N) j).2.val = j.val
  -- `blockIndexEquiv = (Fin.castOrderIso mul_comm).toEquiv.trans
  --                     finProdFinEquiv.symm`.
  -- Apply to `j`: cast to `Fin (N * Bk)` then divNat/modNat. So
  -- `(blockIndexEquiv j).1.val = j.val / Bk` and
  -- `(blockIndexEquiv j).2.val = j.val % Bk`. Then standard
  -- `Nat.div_add_mod`.
  show ((finProdFinEquiv.symm
            ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j))).1.val * Bk +
       ((finProdFinEquiv.symm
            ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j))).2.val = j.val
  show (Fin.divNat ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j)).val * Bk +
       (Fin.modNat ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j)).val = j.val
  -- `(castOrderIso _ j).val = j.val` (cast preserves val).
  -- `Fin.divNat j' .val = j'.val / Bk`, `Fin.modNat j' .val = j'.val % Bk`.
  -- (These are unfoldings of definitions; if simp's not pulling them
  -- through, fall back on the well-known Nat identity.)
  have : (((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j)).val = j.val := rfl
  rw [show (Fin.divNat ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j)).val
        = j.val / Bk from rfl,
      show (Fin.modNat ((Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv j)).val
        = j.val % Bk from rfl]
  rw [Nat.mul_comm (j.val / Bk) Bk]
  exact Nat.div_add_mod j.val Bk

/-- The flat `Σ over Fin (Bk*N)` form of `lFree N (le_refl _)`. Bridges
the double-sum form (used by streaming proofs) and the flat form (used
by `attentionReal` through `softmaxRow`). -/
theorem lFree_eq_flat {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lFree Q K scale N (le_refl _) i =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        Real.exp (scaledScore Q K scale i j)) := by
  unfold lFree
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (blockIndexEquiv Bk N) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [blockIndex_blockIndexEquiv]

/-- Companion flat-form for `oFree`. -/
theorem oFree_eq_flat {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oFree Q K V scale N (le_refl _) idx =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        Real.exp (scaledScore Q K scale idx.1 j) *
        V (j, idx.2.1, PUnit.unit)) := by
  unfold oFree
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (blockIndexEquiv Bk N) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [blockIndex_blockIndexEquiv]

/-- Auxiliary: the WithBot ℝ-valued `qkT` tile data computes the real
`scaledScore`-related dot product (no scale yet). All-`some` because
all operands are lifted via `Tile.ofReal`. -/
theorem qkT_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (i : Fin M) (j : Fin (Bk * N)) :
    (Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] (Tile.ofReal K))).data
        (i, j, PUnit.unit)
      = ((Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)) : ℝ) : WithBot ℝ) := by
  rw [Tile.dot_nil_data]
  have hPush : ∑ d : Fin D, (((Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)
                : ℝ) : WithBot ℝ)) =
      (((∑ d : Fin D, Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)
          : ℝ) : WithBot ℝ)) :=
    (map_sum (WithBot.addHom : ℝ →+ WithBot ℝ) _ _).symm
  rw [← hPush]
  apply Finset.sum_congr rfl
  intro k _
  rw [Tile.transpose_nil_data, Tile.ofReal_data, Tile.ofReal_data]
  rfl

/-- The `scaled` tile inside `attention` data-evaluates to a `WithBot ℝ`
coercion of `scaledScore`. -/
theorem scaled_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) (j : Fin (Bk * N)) :
    Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data (i, j, PUnit.unit))
      = ((scaledScore Q K scale i j : ℝ) : WithBot ℝ) := by
  rw [qkT_data_eq]
  unfold scaledScore
  show some _ = some _
  congr 1
  ring

/-- `softmaxRow` of `scaled` evaluates per-cell as
`some(exp(scaledScore) / Σ_j' exp(scaledScore))`. -/
theorem softmaxRow_scaled_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) (n : Fin (Bk * N)) :
    let scaled : Tile .real [M, Bk * N] :=
      ⟨fun idx => Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data idx)⟩
    (softmaxRow scaled).data (i, n, PUnit.unit) =
      ((Real.exp (scaledScore Q K scale i n) /
        Finset.univ.sum (fun j' : Fin (Bk * N) =>
          Real.exp (scaledScore Q K scale i j')) : ℝ) : WithBot ℝ) := by
  -- Unfold softmaxRow; the row function evaluates to scaledScore via
  -- `scaled_data_eq` + `WithBot.unbotD_coe`.
  unfold softmaxRow
  show some _ = some _
  congr 1
  -- Goal: exp(row n) / Σ exp(row j) = exp(scaledScore i n) / Σ exp(scaledScore i j)
  -- where row j = (scaled.data (i, j, _)).unbotD 0
  have hRow : ∀ j' : Fin (Bk * N),
      ((⟨fun idx => Option.map (· * scale)
          ((Tile.dot [] (Tile.ofReal Q)
            (Tile.transpose [] (Tile.ofReal K))).data idx)⟩ : Tile .real _).data
        (i, j', PUnit.unit)).unbotD 0
        = scaledScore Q K scale i j' := by
    intro j'
    show WithBot.unbotD 0 (Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data (i, j', PUnit.unit)))
        = scaledScore Q K scale i j'
    rw [scaled_data_eq]
    rfl
  congr 1
  · exact congrArg Real.exp (hRow n)
  · apply Finset.sum_congr rfl
    intro j _
    exact congrArg Real.exp (hRow j)

/-! ### Shape-polymorphic variants

These mirror `qkT_data_eq` / `scaled_data_eq` / `softmaxRow_scaled_data_eq`
but take a generic `S` (not necessarily of the form `Bk * N`). Used by
the boundary streaming proofs where K/V live in `[S_k, D]` with no
factorization commitment. The proofs are textually identical — Lean's
definitional unfolding is the same; only the implicit shape changes. -/

/-- Shape-polymorphic version of `qkT_data_eq`. -/
theorem qkT_data_eq' {M D S : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (i : Fin M) (j : Fin S) :
    (Tile.dot [] (Tile.ofReal Q) (Tile.transpose [] (Tile.ofReal K))).data
        (i, j, PUnit.unit)
      = ((Finset.univ.sum (fun d : Fin D =>
            Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)) : ℝ) : WithBot ℝ) := by
  rw [Tile.dot_nil_data]
  have hPush : ∑ d : Fin D, (((Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)
                : ℝ) : WithBot ℝ)) =
      (((∑ d : Fin D, Q (i, d, PUnit.unit) * K (j, d, PUnit.unit)
          : ℝ) : WithBot ℝ)) :=
    (map_sum (WithBot.addHom : ℝ →+ WithBot ℝ) _ _).symm
  rw [← hPush]
  apply Finset.sum_congr rfl
  intro k _
  rw [Tile.transpose_nil_data, Tile.ofReal_data, Tile.ofReal_data]
  rfl

/-- Shape-polymorphic version of `scaled_data_eq`. -/
theorem scaled_data_eq' {M D S : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) (j : Fin S) :
    Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data (i, j, PUnit.unit))
      = ((scaledScore Q K scale i j : ℝ) : WithBot ℝ) := by
  rw [qkT_data_eq']
  unfold scaledScore
  show some _ = some _
  congr 1
  ring

/-- Shape-polymorphic version of `softmaxRow_scaled_data_eq`. -/
theorem softmaxRow_scaled_data_eq' {M D S : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ)
    (scale : ℝ) (i : Fin M) (n : Fin S) :
    let scaled : Tile .real [M, S] :=
      ⟨fun idx => Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data idx)⟩
    (softmaxRow scaled).data (i, n, PUnit.unit) =
      ((Real.exp (scaledScore Q K scale i n) /
        Finset.univ.sum (fun j' : Fin S =>
          Real.exp (scaledScore Q K scale i j')) : ℝ) : WithBot ℝ) := by
  unfold softmaxRow
  show some _ = some _
  congr 1
  have hRow : ∀ j' : Fin S,
      ((⟨fun idx => Option.map (· * scale)
          ((Tile.dot [] (Tile.ofReal Q)
            (Tile.transpose [] (Tile.ofReal K))).data idx)⟩ : Tile .real _).data
        (i, j', PUnit.unit)).unbotD 0
        = scaledScore Q K scale i j' := by
    intro j'
    show WithBot.unbotD 0 (Option.map (· * scale)
        ((Tile.dot [] (Tile.ofReal Q)
          (Tile.transpose [] (Tile.ofReal K))).data (i, j', PUnit.unit)))
        = scaledScore Q K scale i j'
    rw [scaled_data_eq']
    rfl
  congr 1
  · exact congrArg Real.exp (hRow n)
  · apply Finset.sum_congr rfl
    intro j _
    exact congrArg Real.exp (hRow j)

/-- Dot product against the `n`-th KV block computes the corresponding
flat-index score before scaling. This is the block-local companion to
`qkT_data_eq`, used by the operational loop proof. -/
theorem block_qkT_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (n : Nat) (hn : n < N)
    (i : Fin M) (j : Fin Bk) :
    (Tile.dot [] (Tile.ofReal Q)
      (Tile.transpose [] (Tile.ofReal
        (fun idx : TileIndex [Bk, D] =>
          K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.1,
            idx.2.1, PUnit.unit))))).data (i, j, PUnit.unit)
      =
        ((Finset.univ.sum (fun d : Fin D =>
          Q (i, d, PUnit.unit) *
          K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
            d, PUnit.unit)) : ℝ) : WithBot ℝ) := by
  rw [Tile.dot_nil_data]
  have hPush : ∑ d : Fin D,
        (((Q (i, d, PUnit.unit) *
            K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
              d, PUnit.unit) : ℝ) : WithBot ℝ)) =
      (((∑ d : Fin D,
          Q (i, d, PUnit.unit) *
            K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
              d, PUnit.unit) : ℝ) : WithBot ℝ)) :=
    (map_sum (WithBot.addHom : ℝ →+ WithBot ℝ) _ _).symm
  rw [← hPush]
  apply Finset.sum_congr rfl
  intro d _
  rw [Tile.transpose_nil_data, Tile.ofReal_data, Tile.ofReal_data]
  rfl

/-- Scaled block score data: the exact value assigned to the loop-local
`scores[i,j]`. The AST multiplies the dot product by a scalar on the right;
`scaledScore` writes the scale on the left, so the proof finishes by
commutativity of real multiplication. -/
theorem block_scaled_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N)
    (i : Fin M) (j : Fin Bk) :
    WithBot.realMul
      ((Tile.dot [] (Tile.ofReal Q)
        (Tile.transpose [] (Tile.ofReal
          (fun idx : TileIndex [Bk, D] =>
            K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.1,
              idx.2.1, PUnit.unit))))).data (i, j, PUnit.unit))
      ((scale : ℝ) : WithBot ℝ)
      = ((scaledScore Q K scale i
          (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) : ℝ) : WithBot ℝ) := by
  rw [block_qkT_data_eq Q K n hn i j]
  unfold scaledScore
  let S : ℝ := Finset.univ.sum (fun d : Fin D =>
    Q (i, d, PUnit.unit) *
      K (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j, d, PUnit.unit))
  change WithBot.realMul ((S : ℝ) : WithBot ℝ) ((scale : ℝ) : WithBot ℝ) =
      ((scale * S : ℝ) : WithBot ℝ)
  simp [WithBot.realMul]
  rw [mul_comm]

/-- Row-max of the current score block, matching the second argument of
`mPartial_succ_of_lt`. -/
theorem block_mBlock_data_eq {M D Bk N : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N) :
    (Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      (Tile.ofReal fun idx : TileIndex [M, Bk] =>
        scaledScore Q K scale idx.1
          (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1)))
      =
        some ⟨fun idx : TileIndex [M] =>
          (Finset.univ : Finset (Fin Bk)).sup
            (fun j => ((scaledScore Q K scale idx.1
              (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) : ℝ) :
                WithBot ℝ))⟩ := by
  unfold Tile.reduceMax
  simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBk, Tile.ofReal]
  funext idx
  change (((Finset.univ : Finset (Fin Bk)).sup'
      (by exact ⟨⟨0, hBk⟩, Finset.mem_univ _⟩)
      (fun j => scaledScore Q K scale idx.1
        (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j)) : ℝ) : WithBot ℝ) =
    (Finset.univ : Finset (Fin Bk)).sup
      (fun j => ((scaledScore Q K scale idx.1
        (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) : ℝ) : WithBot ℝ))
  rw [← WithBot.sup'_coe]
  rw [Finset.sup'_eq_sup]

/-- Data form of the loop-local unnormalized probabilities
`p = exp(scores - m_new[:, None])`. -/
theorem block_p_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N)
    (mNew : Fin M → WithBot ℝ) (i : Fin M) (j : Fin Bk) :
    WithBot.realExp
      (WithBot.realSub
        ((scaledScore Q K scale i
          (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) : ℝ) : WithBot ℝ)
        (mNew i))
      =
      WithBot.realExp
        (WithBot.realSub
          ((scaledScore Q K scale i
            (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) : ℝ) : WithBot ℝ)
          (mNew i)) := rfl

/-- The operational `p` entry is a real value once `mPartial (k+1)` is known
not to be `⊥` (true for non-empty KV blocks). -/
theorem block_p_toWithBot {M D Bk N : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) (i : Fin M) (j : Fin Bk) :
    WithBot.realExp
      (WithBot.realSub
        ((scaledScore Q K scale i
          (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j) : ℝ) : WithBot ℝ)
        (mPartial Bk Q N K scale (k + 1) i))
      =
      ((Real.exp (scaledScore Q K scale i
          (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j) -
        (mPartial Bk Q N K scale (k + 1) i).unbotD 0) : ℝ) :
          WithBot ℝ) := by
  have hm : mPartial Bk Q N K scale (k + 1) i ≠ ⊥ :=
    mPartial_succ_ne_bot hBk Q N K scale k (Nat.succ_le_iff.mpr hk) i
  obtain ⟨m, hm_eq⟩ := WithBot.ne_bot_iff_exists.mp hm
  rw [← hm_eq]
  simp [WithBot.realSub]

/-- Row-sum of the current `p` block in the non-`⊥` case where `m_new` is
known to be real-valued. This is the additive block contribution in
`lPartial_succ_of_lt`. -/
theorem block_p_rowSum_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N)
    (mNew : Fin M → ℝ) :
    (Tile.reduceSum (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      (Tile.ofReal fun idx : TileIndex [M, Bk] =>
        Real.exp (scaledScore Q K scale idx.1
          (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1) - mNew idx.1)))
      =
        (Tile.ofReal fun idx : TileIndex [M] =>
          Finset.univ.sum (fun j : Fin Bk =>
            Real.exp (scaledScore Q K scale idx.1
              (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) - mNew idx.1))) := by
  ext idx
  unfold Tile.reduceSum
  simp [Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.ofReal]
  rfl

/-- Dotting the current probability block with the current V block gives the
block contribution used by `oPartial_succ_of_lt`. -/
theorem block_pv_data_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (n : Nat) (hn : n < N) (mNew : Fin M → ℝ)
    (i : Fin M) (d : Fin D) :
    (Tile.dot [] (M := M) (K := Bk) (N := D)
      (Tile.ofReal fun idx : TileIndex [M, Bk] =>
        Real.exp (scaledScore Q K scale idx.1
          (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1) - mNew idx.1))
      (Tile.ofReal fun idx : TileIndex [Bk, D] =>
        V (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.1,
          idx.2.1, PUnit.unit))).data (i, d, PUnit.unit)
      =
        ((Finset.univ.sum (fun j : Fin Bk =>
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) - mNew i) *
            V (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
              d, PUnit.unit)) : ℝ) : WithBot ℝ) := by
  rw [Tile.dot_nil_data]
  have hPush : ∑ j : Fin Bk,
        (((Real.exp (scaledScore Q K scale i
              (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) - mNew i) *
            V (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
              d, PUnit.unit) : ℝ) : WithBot ℝ)) =
      (((∑ j : Fin Bk,
          Real.exp (scaledScore Q K scale i
              (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j) - mNew i) *
            V (blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j,
              d, PUnit.unit) : ℝ) : WithBot ℝ)) :=
    (map_sum (WithBot.addHom : ℝ →+ WithBot ℝ) _ _).symm
  rw [← hPush]
  apply Finset.sum_congr rfl
  intro j _
  rw [Tile.ofReal_data, Tile.ofReal_data]
  rfl

/-- The loop-local `m_new = max(m_i, m_block)` realizes the streaming
`mPartial` recurrence at `k + 1`. -/
theorem block_mNew_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
      (⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun j => ((scaledScore Q K scale idx.1
            (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j) : ℝ) :
              WithBot ℝ))⟩ : Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale (k + 1) idx.1⟩ := by
  ext idx
  simp [Tile.bop]
  rw [mPartial_succ_of_lt Q N K scale k hk idx.1]

/-- Bridge between `Option ℝ`'s `Max` instance (used by `Tile.bop max bc`
when the carrier is the raw `Option ℝ`) and `WithBot ℝ`'s `Max` instance
(coming from `SemilatticeSup.toMax`). The two are propositionally — but not
definitionally — equal, so we materialize the equality as a rewrite lemma
that `simp_rw` can fire inside the `fa1_step` proof. -/
theorem option_max_eq_withbot_max (a b : WithBot ℝ) :
    @max (Option ℝ) Option.instMax a b = max a b := by
  cases a <;> cases b <;> rfl

/-- Proof irrelevance for `Finset.sup'`: any two `Nonempty` witnesses give the
same value. Used in the `fa1_step` proof to align `Finset.univ.sup'` proof
arguments coming from `Tile.bop` elaboration with proof arguments coming from
the helper hypotheses (`hAlphaDataMaxOf` etc.). -/
theorem sup'_proof_irrel {α ι : Type*} [SemilatticeSup α] {s : Finset ι}
    (h₁ h₂ : s.Nonempty) (f : ι → α) : s.sup' h₁ f = s.sup' h₂ f := rfl

/-- The loop-local `l_new` expression realizes the streaming `lPartial`
recurrence at `k + 1`. -/
theorem block_lNew_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        (Tile.ofReal fun idx : TileIndex [M] =>
          alphaPartial Q N K scale k idx.1)
        (Tile.ofReal fun idx : TileIndex [M] =>
          lPartial Q N K scale k idx.1))
      (Tile.ofReal fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          Real.exp (scaledScore Q K scale idx.1
            (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j) -
              (mPartial Bk Q N K scale (k + 1) idx.1).unbotD 0)))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        lPartial Q N K scale (k + 1) idx.1) := by
  ext idx
  simp [Tile.bop, Tile.ofReal]
  rw [lPartial_succ_of_lt Q N K scale k hk idx.1]

/-- The loop-local `o_acc` update realizes the streaming `oPartial`
recurrence at `k + 1`. -/
theorem block_oAcc_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.ofReal fun idx : TileIndex [M] =>
            alphaPartial Q N K scale k idx.1))
        (Tile.ofReal fun idx : TileIndex [M, D] =>
          oPartial Q N K V scale k idx))
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun j : Fin Bk =>
          Real.exp (scaledScore Q K scale idx.1
              (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j) -
                (mPartial Bk Q N K scale (k + 1) idx.1).unbotD 0) *
            V (blockIndex Bk N k (Nat.succ_le_iff.mpr hk) j,
              idx.2.1, PUnit.unit)))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        oPartial Q N K V scale (k + 1) idx) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [oPartial_succ_of_lt Q N K V scale k hk (i, d, PUnit.unit)]

/-- The m-free reference sums `oFree N` and `lFree N` connect to
`attentionReal` directly through `Tile.dot` / `softmaxRow`. This is the
specification-side identity (no streaming algebra involved). -/
theorem oFree_div_lFree_eq_attentionReal {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D])
    (_hlFree : lFree Q K scale N (le_refl N) idx.1 ≠ 0) :
    oFree Q K V scale N (le_refl N) idx /
        lFree Q K scale N (le_refl N) idx.1
      = attentionReal Q K V scale idx := by
  rw [oFree_eq_flat, lFree_eq_flat]
  unfold attentionReal attention
  rw [Tile.dot_nil_data]
  -- Match the WithBot Σ to a coerced real Σ via Finset.sum_congr +
  -- softmaxRow_scaled_data_eq, then push the coercion out via map_sum,
  -- unbotD it, and factor the constant denominator with Finset.sum_div.
  rw [Finset.sum_congr rfl (g := fun k : Fin (Bk * N) =>
        (((Real.exp (scaledScore Q K scale idx.1 k) /
          Finset.univ.sum (fun j' : Fin (Bk * N) =>
            Real.exp (scaledScore Q K scale idx.1 j')) *
          V (k, idx.2.1, PUnit.unit)) : ℝ) : WithBot ℝ))
    (by
      intro k _
      let scaled : Tile .real [M, Bk * N] :=
        { data := fun idx =>
            Option.map (fun x => x * scale)
              ((Tile.dot [] (Tile.ofReal Q)
                (Tile.transpose [] (Tile.ofReal K))).data idx) }
      change
        Option.map₂ (fun x1 x2 => x1 * x2)
          ((softmaxRow scaled).data (idx.1, k, PUnit.unit))
          (some (V (k, idx.2.1, PUnit.unit))) =
        (((Real.exp (scaledScore Q K scale idx.1 k) /
            Finset.univ.sum (fun j' : Fin (Bk * N) =>
              Real.exp (scaledScore Q K scale idx.1 j')) *
            V (k, idx.2.1, PUnit.unit)) : ℝ) : WithBot ℝ)
      have hsoft := softmaxRow_scaled_data_eq Q K scale idx.1 k
      change
        (softmaxRow scaled).data (idx.1, k, PUnit.unit) =
          ((Real.exp (scaledScore Q K scale idx.1 k) /
            Finset.univ.sum (fun j' : Fin (Bk * N) =>
              Real.exp (scaledScore Q K scale idx.1 j')) : ℝ) : WithBot ℝ) at hsoft
      rw [hsoft]
      rfl)]
  rw [WithBot.sum_some_eq_some]
  rw [WithBot.unbotD_coe]
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- **Math identity (paper centerpiece).** After all `numKVBlocks`
iterations, the streaming `(oPartial, lPartial)` ratio computes the
same value as `attentionReal`. The proof factors `exp(-m_N)` out of
both numerator and denominator (via `lPartial_eq_mShifted` and
`oPartial_eq_mShifted`), cancels it, and matches the residual
`oFree / lFree` against `attentionReal`. -/
theorem streaming_eq_attentionReal {M D Bk : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat) (_hN : 0 < numKVBlocks)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D])
    (hl : lPartial Q numKVBlocks K scale numKVBlocks idx.1 ≠ 0) :
    oPartial Q numKVBlocks K V scale numKVBlocks idx /
        lPartial Q numKVBlocks K scale numKVBlocks idx.1
      = attentionReal Q K V scale idx := by
  -- Factor exp(-m_N) out of both sides, then cancel.
  rw [oPartial_eq_mShifted hBk Q numKVBlocks K V scale numKVBlocks
        (le_refl _) idx,
      lPartial_eq_mShifted hBk Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1]
  -- After rewriting:
  --   (exp(-m) · oFree) / (exp(-m) · lFree) = attentionReal
  -- Use `mul_div_mul_left` to cancel exp(-m), which is non-zero.
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  -- Now goal: oFree N idx / lFree N idx.1 = attentionReal idx.
  -- We need lFree ≠ 0 to invoke the spec-side identity. Derive from `hl`:
  -- `lPartial = exp(-m) · lFree` and `lPartial ≠ 0` ⇒ `lFree ≠ 0`.
  have hlFree : lFree Q K scale numKVBlocks (le_refl _) idx.1 ≠ 0 := by
    intro h
    apply hl
    rw [lPartial_eq_mShifted hBk Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1, h, mul_zero]
  exact oFree_div_lFree_eq_attentionReal Q K V scale idx hlFree

/-- The final unshifted softmax normalizer is strictly positive when the
KV domain is non-empty. -/
theorem lFree_final_pos {M D Bk N : Nat} (hBk : 0 < Bk) (hN : 0 < N)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    0 < lFree Q K scale N (le_refl N) i := by
  rw [lFree_eq_flat]
  apply Finset.sum_pos
  · intro j _
    exact Real.exp_pos _
  · exact ⟨⟨0, Nat.mul_pos hBk hN⟩, Finset.mem_univ _⟩

/-- The final streaming normalizer is non-zero under FA-1's non-empty KV
scope. This is the denominator fact needed by the readout stage. -/
theorem lPartial_final_ne_zero {M D Bk : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (hN : 0 < numKVBlocks)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lPartial Q numKVBlocks K scale numKVBlocks i ≠ 0 := by
  rw [lPartial_eq_mShifted hBk Q numKVBlocks K scale numKVBlocks
      (le_refl _) i]
  exact mul_ne_zero (Real.exp_ne_zero _)
    (ne_of_gt (lFree_final_pos hBk hN Q K scale i))

end FA1Math

/-! ## Boundary-masked streaming math model (FA-1 v1 scaffold)

The full-tile `FA1Math` recurrence indexes K/V by
`Fin (Bk * numKVBlocks)`. Boundary-masked kernels instead iterate over a
padded block domain while the mathematical input has logical length `S_k`.
Invalid local lanes (`k * Bk + jLocal >= S_k`) enter as `⊥`, exactly like
the kernel's score-side `tl.where(score_mask, scores_raw, -inf)`.
-/

namespace FA1MathBoundary

/-- `WithBot.realExp` is never bottom, so it is equal to its `unbotD`
payload rewrapped as `some`. -/
theorem realExp_eq_some_unbotD (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

/-- Logical KV index for a padded loop lane, if it is in bounds. -/
def blockIndex? (S_k Bk k : Nat) (jLocal : Fin Bk) : Option (Fin S_k) :=
  if h : k * Bk + jLocal.val < S_k then
    some ⟨k * Bk + jLocal.val, h⟩
  else
    none

@[simp] theorem blockIndex?_of_lt
    (S_k Bk k : Nat) (jLocal : Fin Bk)
    (h : k * Bk + jLocal.val < S_k) :
    blockIndex? S_k Bk k jLocal = some ⟨k * Bk + jLocal.val, h⟩ := by
  simp [blockIndex?, h]

@[simp] theorem blockIndex?_of_not_lt
    (S_k Bk k : Nat) (jLocal : Fin Bk)
    (h : ¬ k * Bk + jLocal.val < S_k) :
    blockIndex? S_k Bk k jLocal = none := by
  simp [blockIndex?, h]

/-- Boundary-masked score for a padded loop lane. Out-of-range KV lanes are
`⊥`, so exponentiating them contributes zero mass. -/
noncomputable def maskedScore {M S_k D : Nat}
    (Bk k : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk) : WithBot ℝ :=
  match blockIndex? S_k Bk k jLocal with
  | some j => (FA1Math.scaledScore Q K scale i j : ℝ)
  | none   => ⊥

@[simp] theorem maskedScore_of_lt {M S_k D : Nat}
    (Bk k : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk)
    (h : k * Bk + jLocal.val < S_k) :
    maskedScore Bk k Q K scale i jLocal =
      ((FA1Math.scaledScore Q K scale i
        ⟨k * Bk + jLocal.val, h⟩ : ℝ) : WithBot ℝ) := by
  simp [maskedScore, h]

@[simp] theorem maskedScore_of_not_lt {M S_k D : Nat}
    (Bk k : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk)
    (h : ¬ k * Bk + jLocal.val < S_k) :
    maskedScore Bk k Q K scale i jLocal = (⊥ : WithBot ℝ) := by
  simp [maskedScore, h]

/-- Running per-row max over the first `k` padded KV blocks, ignoring
out-of-range lanes by treating them as `⊥`. -/
noncomputable def mPartial {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if _h : k + 1 ≤ numKVBlocks then
        max (mPartial Bk Q numKVBlocks K scale k i)
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            maskedScore Bk k Q K scale i jLocal)
      else
        mPartial Bk Q numKVBlocks K scale k i

/-- Boundary-aware α multiplier `exp(m_k - m_{k+1})`. -/
noncomputable def alphaPartial {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  (WithBot.realExp
    (WithBot.realSub
      (mPartial Bk Q numKVBlocks K scale k i)
      (mPartial Bk Q numKVBlocks K scale (k + 1) i))).unbotD 0

/-- Boundary-aware running normalizer. -/
noncomputable def lPartial {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if _h : k + 1 ≤ numKVBlocks then
        let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
        alphaPartial Bk Q numKVBlocks K scale k i *
          lPartial Bk Q numKVBlocks K scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0)
      else
        lPartial Bk Q numKVBlocks K scale k i

/-- Boundary-aware running unnormalized output accumulator. -/
noncomputable def oPartial {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, D] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if _h : k + 1 ≤ numKVBlocks then
        let i := idx.1
        let d := idx.2.1
        let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
        alphaPartial Bk Q numKVBlocks K scale k i *
          oPartial Bk Q numKVBlocks K V scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match blockIndex? S_k Bk k jLocal with
            | some j =>
                (WithBot.realExp
                  (WithBot.realSub
                    (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0 *
                  V (j, d, PUnit.unit)
            | none => 0)
      else
        oPartial Bk Q numKVBlocks K V scale k idx

theorem mPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    mPartial Bk Q numKVBlocks K scale (k + 1) i =
      max (mPartial Bk Q numKVBlocks K scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore Bk k Q K scale i jLocal) := by
  change (if h : k + 1 ≤ numKVBlocks then
      max (mPartial Bk Q numKVBlocks K scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore Bk k Q K scale i jLocal)
    else mPartial Bk Q numKVBlocks K scale k i) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem lPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    lPartial Bk Q numKVBlocks K scale (k + 1) i =
      let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
      alphaPartial Bk Q numKVBlocks K scale k i *
        lPartial Bk Q numKVBlocks K scale k i +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          (WithBot.realExp
            (WithBot.realSub
              (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
       alphaPartial Bk Q numKVBlocks K scale k i *
          lPartial Bk Q numKVBlocks K scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0))
    else lPartial Bk Q numKVBlocks K scale k i) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem oPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk Q numKVBlocks K V scale (k + 1) idx =
      let i := idx.1
      let d := idx.2.1
      let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
      alphaPartial Bk Q numKVBlocks K scale k i *
        oPartial Bk Q numKVBlocks K V scale k idx +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          match blockIndex? S_k Bk k jLocal with
          | some j =>
              (WithBot.realExp
                (WithBot.realSub
                  (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0 *
                V (j, d, PUnit.unit)
          | none => 0) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let i := idx.1
       let d := idx.2.1
       let mNew := mPartial Bk Q numKVBlocks K scale (k + 1) i
       alphaPartial Bk Q numKVBlocks K scale k i *
          oPartial Bk Q numKVBlocks K V scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match blockIndex? S_k Bk k jLocal with
            | some j =>
                (WithBot.realExp
                  (WithBot.realSub
                    (maskedScore Bk k Q K scale i jLocal) mNew)).unbotD 0 *
                  V (j, d, PUnit.unit)
            | none => 0))
    else oPartial Bk Q numKVBlocks K V scale k idx) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-- Boundary loop-local `m_new = max(m_i, m_block)` realizes the masked
streaming `mPartial` recurrence. -/
theorem block_mNew_tile_eq {M S_k D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
      (⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun j => maskedScore Bk k Q K scale idx.1 j)⟩ : Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale (k + 1) idx.1⟩ := by
  ext idx
  simp [Tile.bop]
  rw [mPartial_succ_of_lt Bk Q N K scale k hk idx.1]

/-- Boundary loop-local `alpha = exp(m_i - m_new)` after folding
`m_new = mPartial(k+1)`. -/
theorem block_alpha_tile_eq {M S_k D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.uop WithBot.realExp
      (Tile.bop WithBot.realSub (Broadcast.consSame Broadcast.nil)
        (⟨fun idx : TileIndex [M] =>
          mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
        (Tile.bop max (Broadcast.consSame Broadcast.nil)
          (⟨fun idx : TileIndex [M] =>
            mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
          (⟨fun idx : TileIndex [M] =>
            (Finset.univ : Finset (Fin Bk)).sup
              (fun j => maskedScore Bk k Q K scale idx.1 j)⟩ : Tile .real [M])))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        alphaPartial Bk Q N K scale k idx.1) := by
  ext idx
  have hmTile :
      Tile.bop max (Broadcast.consSame Broadcast.nil)
        (⟨fun idx : TileIndex [M] =>
          mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
        (⟨fun idx : TileIndex [M] =>
          (Finset.univ : Finset (Fin Bk)).sup
            (fun j => maskedScore Bk k Q K scale idx.1 j)⟩ : Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale (k + 1) idx.1⟩ :=
    block_mNew_tile_eq (Bk := Bk) Q K scale k hk
  have hm :=
    congrArg (fun t : Tile .real [M] => t.data idx)
      hmTile
  simp [Tile.bop] at hm
  simp [Tile.uop, Tile.bop, Tile.ofReal, alphaPartial]
  rw [hm]
  exact realExp_eq_some_unbotD _

/-- Boundary loop-local probability tile `p = exp(scores - m_new[:,None])`
after folding `m_new = mPartial(k+1)`. -/
theorem block_p_tile_eq {M S_k D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.uop WithBot.realExp
      (Tile.bop WithBot.realSub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (⟨fun idx : TileIndex [M, Bk] =>
          maskedScore Bk k Q K scale idx.1 idx.2.1⟩ : Tile .real [M, Bk])
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.bop max (Broadcast.consSame Broadcast.nil)
            (⟨fun idx : TileIndex [M] =>
              mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
            (⟨fun idx : TileIndex [M] =>
              (Finset.univ : Finset (Fin Bk)).sup
                (fun j => maskedScore Bk k Q K scale idx.1 j)⟩ : Tile .real [M]))))
      =
      Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        (WithBot.realExp
          (WithBot.realSub
            (maskedScore Bk k Q K scale idx.1 idx.2.1)
            (mPartial Bk Q N K scale (k + 1) idx.1))).unbotD 0) := by
  ext idx
  have hmTile :
      Tile.bop max (Broadcast.consSame Broadcast.nil)
        (⟨fun idx : TileIndex [M] =>
          mPartial Bk Q N K scale k idx.1⟩ : Tile .real [M])
        (⟨fun idx : TileIndex [M] =>
          (Finset.univ : Finset (Fin Bk)).sup
            (fun j => maskedScore Bk k Q K scale idx.1 j)⟩ : Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] => mPartial Bk Q N K scale (k + 1) idx.1⟩ :=
    block_mNew_tile_eq (Bk := Bk) Q K scale k hk
  have hm :=
    congrArg (fun t : Tile .real [M] => t.data (idx.1, PUnit.unit))
      hmTile
  simp [Tile.bop] at hm
  simp [Tile.uop, Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [hm]
  exact realExp_eq_some_unbotD _

/-- Boundary score mask `offs_n < S_k` lifted to the `[M, Bk]` score tile. -/
theorem block_score_mask_tile_eq {M Bk S_k : Nat} (k : Nat) :
    (Tile.cop (ComparableDType.nat.lt)
      Broadcast.scalarR
      (Tile.bop NumericDType.nat.add (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Tile.bop NumericDType.nat.mul Broadcast.scalarR
          (Tile.expandDim ⟨1, by simp⟩
            (Tile.vec (fun i : Fin M => i.val)))
          (Tile.scalar 0))
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.scalar S_k))
      = ⟨fun idx : TileIndex [M, Bk] =>
          decide (k * Bk + idx.2.1.val < S_k)⟩ := by
  ext idx
  obtain ⟨i, j, _⟩ := idx
  simp only [Tile.cop, Tile.bop, ComparableDType.lt, NumericDType.add,
    NumericDType.mul, Tile.expandDim_data, Tile.vec_data, Tile.scalar_data_index,
    Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex]
  simp

/-- Applying the boundary score mask to the raw score tile produces exactly
`maskedScore`: valid KV lanes carry the real scaled score; padded lanes carry
`⊥` through the explicit `-inf` branch. -/
theorem block_scores_tile_eq {M S_k D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) :
    Tile.select
      (⟨fun idx : TileIndex [M, Bk] =>
        decide (k * Bk + idx.2.1.val < S_k)⟩ : Tile .bool [M, Bk])
      (Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        match blockIndex? S_k Bk k idx.2.1 with
        | some j => FA1Math.scaledScore Q K scale idx.1 j
        | none => 0))
      (⟨fun _ : TileIndex [M, Bk] => (none : WithBot ℝ)⟩ : Tile .real [M, Bk])
    = ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore Bk k Q K scale idx.1 idx.2.1⟩ := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.select_data, Tile.ofReal_data]
  by_cases h : k * Bk + j.val < S_k
  · rw [decide_eq_true h]
    simp [h, maskedScore_of_lt, blockIndex?_of_lt]
    rfl
  · rw [decide_eq_false h]
    simp [h, maskedScore_of_not_lt]
    rfl

/-- Row-max of the boundary-masked scores tile. -/
theorem block_mBlock_tile_eq {M S_k D Bk : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) :
    Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore Bk k Q K scale idx.1 idx.2.1⟩
    = some ⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun jLocal => maskedScore Bk k Q K scale idx.1 jLocal)⟩ := by
  unfold Tile.reduceMax
  simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBk]
  funext idx
  rw [Finset.sup'_eq_sup]
  rfl

/-- Boundary loop-local `l_new` expression realizes the masked streaming
`lPartial` recurrence. -/
theorem block_lNew_tile_eq {M S_k D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        (Tile.ofReal fun idx : TileIndex [M] =>
          alphaPartial Bk Q N K scale k idx.1)
        (Tile.ofReal fun idx : TileIndex [M] =>
          lPartial Bk Q N K scale k idx.1))
      (Tile.ofReal fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          (WithBot.realExp
            (WithBot.realSub
              (maskedScore Bk k Q K scale idx.1 j)
              (mPartial Bk Q N K scale (k + 1) idx.1))).unbotD 0))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        lPartial Bk Q N K scale (k + 1) idx.1) := by
  ext idx
  simp [Tile.bop, Tile.ofReal]
  rw [lPartial_succ_of_lt Bk Q N K scale k hk idx.1]
  simp [WithBot.realSub]

/-- Boundary loop-local `o_acc` update realizes the masked streaming
`oPartial` recurrence. -/
theorem block_oAcc_tile_eq {M S_k D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.ofReal fun idx : TileIndex [M] =>
            alphaPartial Bk Q N K scale k idx.1))
        (Tile.ofReal fun idx : TileIndex [M, D] =>
          oPartial Bk Q N K V scale k idx))
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun j : Fin Bk =>
          match blockIndex? S_k Bk k j with
          | some jGlobal =>
              (WithBot.realExp
                (WithBot.realSub
                  (maskedScore Bk k Q K scale idx.1 j)
                  (mPartial Bk Q N K scale (k + 1) idx.1))).unbotD 0 *
                V (jGlobal, idx.2.1, PUnit.unit)
          | none => 0))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        oPartial Bk Q N K V scale (k + 1) idx) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [oPartial_succ_of_lt Bk Q N K V scale k hk (i, d, PUnit.unit)]
  simp [WithBot.realSub]

/-! ### Boundary m-free reference sums

Mirrors `FA1Math.lFree` / `FA1Math.oFree` but indexes over the logical
`[S_k, D]` domain. Out-of-range loop lanes (`k * Bk + jLocal ≥ S_k`)
contribute zero, so summing all `numKVBlocks * Bk` slots yields the same
total as summing over `Fin S_k` whenever `S_k ≤ Bk * numKVBlocks`. -/

/-- Σ over the first `k` padded KV blocks of `exp(scaledScore)` for
in-range lanes only, no `m`-shift. Out-of-range lanes contribute 0. -/
noncomputable def lFreeBoundary {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    match blockIndex? S_k Bk n.val jL with
    | some j => Real.exp (FA1Math.scaledScore Q K scale i j)
    | none => 0))

/-- Σ over the first `k` padded KV blocks of `exp(scaledScore) · V` for
in-range lanes only, no `m`-shift. Out-of-range lanes contribute 0. -/
noncomputable def oFreeBoundary {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (idx : TileIndex [M, D]) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    match blockIndex? S_k Bk n.val jL with
    | some j =>
        Real.exp (FA1Math.scaledScore Q K scale idx.1 j) *
          V (j, idx.2.1, PUnit.unit)
    | none => 0))

@[simp] theorem lFreeBoundary_zero {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lFreeBoundary Bk Q K scale 0 i = 0 := by
  unfold lFreeBoundary
  simp

@[simp] theorem oFreeBoundary_zero {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oFreeBoundary Bk Q K V scale 0 idx = 0 := by
  unfold oFreeBoundary
  simp

/-- Recurrence: `lFreeBoundary (k+1)` adds the next block's masked sum
on top of `lFreeBoundary k`. -/
theorem lFreeBoundary_succ {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) :
    lFreeBoundary Bk Q K scale (k + 1) i =
      lFreeBoundary Bk Q K scale k i +
      Finset.univ.sum (fun jL : Fin Bk =>
        match blockIndex? S_k Bk k jL with
        | some j => Real.exp (FA1Math.scaledScore Q K scale i j)
        | none => 0) := by
  unfold lFreeBoundary
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Recurrence: `oFreeBoundary (k+1)` adds the next block's masked
`exp · V` sum on top of `oFreeBoundary k`. -/
theorem oFreeBoundary_succ {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (idx : TileIndex [M, D]) :
    oFreeBoundary Bk Q K V scale (k + 1) idx =
      oFreeBoundary Bk Q K V scale k idx +
      Finset.univ.sum (fun jL : Fin Bk =>
        match blockIndex? S_k Bk k jL with
        | some j =>
            Real.exp (FA1Math.scaledScore Q K scale idx.1 j) *
              V (j, idx.2.1, PUnit.unit)
        | none => 0) := by
  unfold oFreeBoundary
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Whenever the loop has reached at least one iteration AND the logical
KV length `S_k > 0` (so the very first lane `0 * Bk + 0 = 0` is in-range),
the boundary `mPartial (k+1)` is non-`⊥`. This propagates through the
running max from the first in-range contribution. -/
theorem mPartial_succ_ne_bot {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    mPartial Bk Q numKVBlocks K scale (k + 1) i ≠ ⊥ := by
  induction k with
  | zero =>
      rw [mPartial_succ_of_lt Bk Q numKVBlocks K scale 0
            (Nat.lt_of_succ_le hk) i]
      have hVis : 0 * Bk + (⟨0, hBk⟩ : Fin Bk).val < S_k := by
        simp; exact hSk
      have h0 : maskedScore Bk 0 Q K scale i (⟨0, hBk⟩ : Fin Bk) ≠ ⊥ := by
        rw [maskedScore_of_lt Bk 0 Q K scale i _ hVis]
        exact WithBot.coe_ne_bot
      show mPartial Bk Q numKVBlocks K scale 0 i ⊔ _ ≠ ⊥
      change ⊥ ⊔ _ ≠ ⊥
      rw [bot_sup_eq]
      simp [Finset.sup_eq_bot_iff]
      exact ⟨⟨0, hBk⟩, h0⟩
  | succ k' ih =>
      have hk' : k' + 1 ≤ numKVBlocks := by omega
      rw [mPartial_succ_of_lt Bk Q numKVBlocks K scale (k' + 1)
            (Nat.lt_of_succ_le hk) i]
      intro hcontra
      have h_left : mPartial Bk Q numKVBlocks K scale (k' + 1) i ≤ ⊥ := by
        rw [← hcontra]; exact le_max_left _ _
      exact ih hk' (le_bot_iff.mp h_left)

/-- The boundary streaming `lPartial k i` equals `exp(-m_k) · lFreeBoundary k i`,
where `m_k = (mPartial k i).unbotD 0`. By induction on `k`, parallel to
`FA1Math.lPartial_eq_mShifted` and `FA1MathCausal.lPartial_eq_mShifted`. -/
theorem lPartial_eq_mShifted {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) :
    lPartial Bk Q numKVBlocks K scale k i =
      Real.exp (-(mPartial Bk Q numKVBlocks K scale k i).unbotD 0) *
        lFreeBoundary Bk Q K scale k i := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          lFreeBoundary Bk Q K scale 0 i
      rw [lFreeBoundary_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [lPartial_succ_of_lt Bk Q numKVBlocks K scale k
        (Nat.lt_of_succ_le hk) i]
      rw [ih hk']
      rw [lFreeBoundary_succ Bk Q K scale k i, mul_add]
      -- Goal:
      --   αₖ * (exp(-mₖ) * lFreeBoundary k) +
      --   Σ exp(maskedScore - m_{k+1}).unbotD 0
      --   = exp(-m_{k+1}) * lFreeBoundary k
      --     + exp(-m_{k+1}) * Σ_{in-range} exp(scaledScore)
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k Q K scale i jLocal)
                (mPartial Bk Q numKVBlocks K scale (k + 1) i))).unbotD 0)
          =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jL : Fin Bk =>
              match blockIndex? S_k Bk k jL with
              | some j => Real.exp (FA1Math.scaledScore Q K scale i j)
              | none => 0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        by_cases hvis : k * Bk + jLocal.val < S_k
        · rw [maskedScore_of_lt Bk k Q K scale i jLocal hvis,
              blockIndex?_of_lt S_k Bk k jLocal hvis]
          obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
            (mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale k hk i)
          rw [← hm]
          simp
          rw [show FA1Math.scaledScore Q K scale i ⟨k * Bk + jLocal.val, hvis⟩ - m =
                -m + FA1Math.scaledScore Q K scale i ⟨k * Bk + jLocal.val, hvis⟩
                by ring,
              Real.exp_add]
        · rw [maskedScore_of_not_lt Bk k Q K scale i jLocal hvis,
              blockIndex?_of_not_lt S_k Bk k jLocal hvis]
          simp
          cases mPartial Bk Q numKVBlocks K scale (k + 1) i <;>
            (left; unfold WithBot.realExp; rfl)
      have hSumA :
          alphaPartial Bk Q numKVBlocks K scale k i *
            (Real.exp (-(mPartial Bk Q numKVBlocks K scale k i).unbotD 0) *
              lFreeBoundary Bk Q K scale k i)
          =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            lFreeBoundary Bk Q K scale k i := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [lFreeBoundary_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne :=
            mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale k' hk_succ i
          have hmk1_ne :=
            mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale (k' + 1) hk i
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          have hAlpha :
              alphaPartial Bk Q numKVBlocks K scale (k' + 1) i =
                Real.exp (rk - rk1) := by
            unfold alphaPartial
            rw [← hrk, ← hrk1]
            simp [WithBot.realSub]
          rw [hAlpha, ← hrk, ← hrk1]
          simp only [WithBot.unbotD_coe]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
                rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * lFreeBoundary Bk Q K scale (k' + 1) i)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      lFreeBoundary Bk Q K scale (k' + 1) i) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
                rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Companion identity for `oPartial`: `exp(-m_k) · oFreeBoundary k idx`.
Same shape as `lPartial_eq_mShifted` plus `· V[j, d]` on each summand. -/
theorem oPartial_eq_mShifted {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk Q numKVBlocks K V scale k idx =
      Real.exp (-(mPartial Bk Q numKVBlocks K scale k idx.1).unbotD 0) *
        oFreeBoundary Bk Q K V scale k idx := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          oFreeBoundary Bk Q K V scale 0 idx
      rw [oFreeBoundary_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [oPartial_succ_of_lt Bk Q numKVBlocks K V scale k
        (Nat.lt_of_succ_le hk) idx]
      rw [ih hk']
      rw [oFreeBoundary_succ Bk Q K V scale k idx, mul_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match blockIndex? S_k Bk k jLocal with
            | some j =>
                (WithBot.realExp
                  (WithBot.realSub
                    (maskedScore Bk k Q K scale idx.1 jLocal)
                    (mPartial Bk Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
                  V (j, idx.2.1, PUnit.unit)
            | none => 0)
          =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jL : Fin Bk =>
              match blockIndex? S_k Bk k jL with
              | some j =>
                  Real.exp (FA1Math.scaledScore Q K scale idx.1 j) *
                    V (j, idx.2.1, PUnit.unit)
              | none => 0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        by_cases hvis : k * Bk + jLocal.val < S_k
        · rw [maskedScore_of_lt Bk k Q K scale idx.1 jLocal hvis,
              blockIndex?_of_lt S_k Bk k jLocal hvis]
          obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
            (mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale k hk idx.1)
          rw [← hm]
          simp
          rw [show FA1Math.scaledScore Q K scale idx.1
                  ⟨k * Bk + jLocal.val, hvis⟩ - m =
                -m + FA1Math.scaledScore Q K scale idx.1
                  ⟨k * Bk + jLocal.val, hvis⟩ by ring,
              Real.exp_add]
          ring
        · rw [blockIndex?_of_not_lt S_k Bk k jLocal hvis]
          simp
      have hSumA :
          alphaPartial Bk Q numKVBlocks K scale k idx.1 *
            (Real.exp (-(mPartial Bk Q numKVBlocks K scale k idx.1).unbotD 0) *
              oFreeBoundary Bk Q K V scale k idx)
          =
          Real.exp (-(mPartial Bk Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            oFreeBoundary Bk Q K V scale k idx := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [oFreeBoundary_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne :=
            mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale k' hk_succ idx.1
          have hmk1_ne :=
            mPartial_succ_ne_bot hBk hSk Q numKVBlocks K scale (k' + 1) hk idx.1
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          have hAlpha :
              alphaPartial Bk Q numKVBlocks K scale (k' + 1) idx.1 =
                Real.exp (rk - rk1) := by
            unfold alphaPartial
            rw [← hrk, ← hrk1]
            simp [WithBot.realSub]
          rw [hAlpha, ← hrk, ← hrk1]
          simp only [WithBot.unbotD_coe]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
                rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * oFreeBoundary Bk Q K V scale (k' + 1) idx)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      oFreeBoundary Bk Q K V scale (k' + 1) idx) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
                rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Flat-index form of `lFreeBoundary numKVBlocks i`: collapse the
double sum over `Fin numKVBlocks × Fin Bk` to a single sum over
`Fin (Bk * numKVBlocks)` using the same bijection as `FA1Math.lFree_eq_flat`.
The masked scores (out-of-range lanes give 0) are preserved exactly
under the bijection, since `(blockIndexEquiv j).1.val * Bk + ... = j.val`. -/
theorem lFreeBoundary_eq_flat {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (i : Fin M) :
    lFreeBoundary Bk Q K scale numKVBlocks i =
      Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
        else
          0) := by
  unfold lFreeBoundary
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j' _
    -- The bijection sends `j'` to `(n, jL)` with `n.val * Bk + jL.val = j'.val`.
    set p := FA1Math.blockIndexEquiv Bk numKVBlocks j' with hp
    show (if h : j'.val < S_k then
            Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
          else
            0)
        = (match blockIndex? S_k Bk p.1.val p.2 with
          | some j => Real.exp (FA1Math.scaledScore Q K scale i j)
          | none => 0)
    have hValEq : p.1.val * Bk + p.2.val = j'.val := by
      have hBI := FA1Math.blockIndex_blockIndexEquiv (Bk := Bk) (N := numKVBlocks) j'
      have h1 := congrArg Fin.val hBI
      change p.1.val * Bk + p.2.val = j'.val at h1
      exact h1
    by_cases hLt : p.1.val * Bk + p.2.val < S_k
    · rw [blockIndex?_of_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : j'.val < S_k := hValEq ▸ hLt
      rw [dif_pos hLt']
      congr 2
      apply Fin.ext
      exact hValEq.symm
    · rw [blockIndex?_of_not_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : ¬ j'.val < S_k := hValEq ▸ hLt
      rw [dif_neg hLt']

/-- Flat-index form of `oFreeBoundary numKVBlocks idx`. Companion to
`lFreeBoundary_eq_flat`. -/
theorem oFreeBoundary_eq_flat {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (idx : TileIndex [M, D]) :
    oFreeBoundary Bk Q K V scale numKVBlocks idx =
      Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩) *
            V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
        else
          0) := by
  unfold oFreeBoundary
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j' _
    set p := FA1Math.blockIndexEquiv Bk numKVBlocks j' with hp
    show (if h : j'.val < S_k then
            Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩) *
              V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
          else
            0)
        = (match blockIndex? S_k Bk p.1.val p.2 with
          | some j =>
              Real.exp (FA1Math.scaledScore Q K scale idx.1 j) *
                V (j, idx.2.1, PUnit.unit)
          | none => 0)
    have hValEq : p.1.val * Bk + p.2.val = j'.val := by
      have hBI := FA1Math.blockIndex_blockIndexEquiv (Bk := Bk) (N := numKVBlocks) j'
      have h1 := congrArg Fin.val hBI
      change p.1.val * Bk + p.2.val = j'.val at h1
      exact h1
    by_cases hLt : p.1.val * Bk + p.2.val < S_k
    · rw [blockIndex?_of_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : j'.val < S_k := hValEq ▸ hLt
      rw [dif_pos hLt']
      have hFinEq : (⟨p.1.val * Bk + p.2.val, hLt⟩ : Fin S_k) =
          ⟨j'.val, hLt'⟩ := Fin.ext hValEq
      rw [hFinEq]
    · rw [blockIndex?_of_not_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : ¬ j'.val < S_k := hValEq ▸ hLt
      rw [dif_neg hLt']

/-- Reindex the flat-form `lFreeBoundary` from `Fin (Bk * numKVBlocks)` to
`Fin S_k`: padded indices `j ≥ S_k` contribute 0, so the embedding
`Fin S_k ↪ Fin (Bk * numKVBlocks)` (valid because `S_k ≤ Bk * numKVBlocks`)
exhausts the support. -/
theorem lFreeBoundary_final_eq_finSk_sum {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks) (i : Fin M) :
    lFreeBoundary Bk Q K scale numKVBlocks i =
      Finset.univ.sum (fun j : Fin S_k =>
        Real.exp (FA1Math.scaledScore Q K scale i j)) := by
  rw [lFreeBoundary_eq_flat]
  -- Use `Fintype.sum_subset` strategy: only `j' : j'.val < S_k` contribute,
  -- and these biject with `Fin S_k` via `Fin.castLE hSkLe`.
  -- Apply `Finset.sum_bij` from `Fin S_k` to `Fin (Bk * numKVBlocks)`-with-filter.
  rw [show (Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
        else
          0)) =
        ((Finset.univ : Finset (Fin (Bk * numKVBlocks))).filter
          (fun j' => j'.val < S_k)).sum
          (fun j' : Fin (Bk * numKVBlocks) =>
            if h : j'.val < S_k then
              Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
            else
              0)
        from ?_]
  · -- Now the sum is over `j' < S_k`. Reindex via `Fin.castLE hSkLe`.
    refine (Finset.sum_bij (fun (j : Fin S_k) (_ : j ∈ Finset.univ) =>
      Fin.castLE hSkLe j) ?_ ?_ ?_ ?_).symm
    · -- (mem) `Fin.castLE hSkLe j ∈ filter`
      intro j _
      simp [Fin.val_castLE, j.isLt]
    · -- (inj) `Fin.castLE hSkLe` is injective on its domain
      intro j₁ _ j₂ _ heq
      apply Fin.ext
      have := congrArg Fin.val heq
      simpa [Fin.val_castLE] using this
    · -- (surj) every `j' < S_k` is `Fin.castLE hSkLe j` for some `j`
      intro j' hj'
      simp at hj'
      refine ⟨⟨j'.val, hj'⟩, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      simp
    · -- (val) value match
      intro j _
      have hLt : (Fin.castLE hSkLe j).val < S_k := by
        rw [Fin.val_castLE]; exact j.isLt
      rw [dif_pos hLt]
      congr 1
  · -- Auxiliary: sum-over-univ with 0-on-failed-dite equals filtered sum.
    refine (Finset.sum_filter_of_ne ?_).symm
    intro j' _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

/-- Companion of `lFreeBoundary_final_eq_finSk_sum` for `oFreeBoundary`. -/
theorem oFreeBoundary_final_eq_finSk_sum {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (idx : TileIndex [M, D]) :
    oFreeBoundary Bk Q K V scale numKVBlocks idx =
      Finset.univ.sum (fun j : Fin S_k =>
        Real.exp (FA1Math.scaledScore Q K scale idx.1 j) *
          V (j, idx.2.1, PUnit.unit)) := by
  rw [oFreeBoundary_eq_flat]
  rw [show (Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩) *
            V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
        else
          0)) =
        ((Finset.univ : Finset (Fin (Bk * numKVBlocks))).filter
          (fun j' => j'.val < S_k)).sum
          (fun j' : Fin (Bk * numKVBlocks) =>
            if h : j'.val < S_k then
              Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩) *
                V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
            else
              0)
        from ?_]
  · refine (Finset.sum_bij (fun (j : Fin S_k) (_ : j ∈ Finset.univ) =>
      Fin.castLE hSkLe j) ?_ ?_ ?_ ?_).symm
    · intro j _
      simp [Fin.val_castLE, j.isLt]
    · intro j₁ _ j₂ _ heq
      apply Fin.ext
      have := congrArg Fin.val heq
      simpa [Fin.val_castLE] using this
    · intro j' hj'
      simp at hj'
      refine ⟨⟨j'.val, hj'⟩, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      simp
    · intro j _
      have hLt : (Fin.castLE hSkLe j).val < S_k := by
        rw [Fin.val_castLE]; exact j.isLt
      rw [dif_pos hLt]
      congr 1
  · refine (Finset.sum_filter_of_ne ?_).symm
    intro j' _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

/-- Boundary spec-side identity: the m-free reference ratio (over `Fin S_k`)
equals `attentionReal` directly. Mirrors `FA1Math.oFree_div_lFree_eq_attentionReal`,
adapted for `K V : [S_k, D] → ℝ`. -/
theorem oFreeBoundary_div_lFreeBoundary_eq_attentionReal
    {M S_k D : Nat} (Bk : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (idx : TileIndex [M, D])
    (_hlFree : lFreeBoundary Bk Q K scale numKVBlocks idx.1 ≠ 0) :
    oFreeBoundary Bk Q K V scale numKVBlocks idx /
        lFreeBoundary Bk Q K scale numKVBlocks idx.1
      = attentionReal Q K V scale idx := by
  rw [oFreeBoundary_final_eq_finSk_sum Bk Q K V scale numKVBlocks hSkLe idx]
  rw [lFreeBoundary_final_eq_finSk_sum Bk Q K scale numKVBlocks hSkLe idx.1]
  unfold attentionReal attention
  rw [Tile.dot_nil_data]
  rw [Finset.sum_congr rfl (g := fun k : Fin S_k =>
        (((Real.exp (FA1Math.scaledScore Q K scale idx.1 k) /
          Finset.univ.sum (fun j' : Fin S_k =>
            Real.exp (FA1Math.scaledScore Q K scale idx.1 j')) *
          V (k, idx.2.1, PUnit.unit)) : ℝ) : WithBot ℝ))
    (by
      intro k _
      let scaled : Tile .real [M, S_k] :=
        { data := fun idx =>
            Option.map (fun x => x * scale)
              ((Tile.dot [] (Tile.ofReal Q)
                (Tile.transpose [] (Tile.ofReal K))).data idx) }
      change
        Option.map₂ (fun x1 x2 => x1 * x2)
          ((softmaxRow scaled).data (idx.1, k, PUnit.unit))
          (some (V (k, idx.2.1, PUnit.unit))) =
        (((Real.exp (FA1Math.scaledScore Q K scale idx.1 k) /
            Finset.univ.sum (fun j' : Fin S_k =>
              Real.exp (FA1Math.scaledScore Q K scale idx.1 j')) *
            V (k, idx.2.1, PUnit.unit)) : ℝ) : WithBot ℝ)
      have hsoft := FA1Math.softmaxRow_scaled_data_eq' Q K scale idx.1 k
      change
        (softmaxRow scaled).data (idx.1, k, PUnit.unit) =
          ((Real.exp (FA1Math.scaledScore Q K scale idx.1 k) /
            Finset.univ.sum (fun j' : Fin S_k =>
              Real.exp (FA1Math.scaledScore Q K scale idx.1 j')) : ℝ) : WithBot ℝ) at hsoft
      rw [hsoft]
      rfl)]
  rw [WithBot.sum_some_eq_some]
  rw [WithBot.unbotD_coe]
  rw [Finset.sum_div]
  apply Finset.sum_congr rfl
  intro k _
  ring

/-- The boundary `lFreeBoundary numKVBlocks` is strictly positive when
the logical KV scope is non-empty (`0 < S_k`): the j=0 lane (in-range
when `0 < S_k`) contributes `Real.exp _ > 0`. -/
theorem lFreeBoundary_final_pos {M S_k D : Nat} (Bk : Nat)
    (hSk : 0 < S_k)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (i : Fin M) :
    0 < lFreeBoundary Bk Q K scale numKVBlocks i := by
  rw [lFreeBoundary_final_eq_finSk_sum Bk Q K scale numKVBlocks hSkLe i]
  apply Finset.sum_pos'
  · intro j _
    exact le_of_lt (Real.exp_pos _)
  · refine ⟨⟨0, hSk⟩, Finset.mem_univ _, ?_⟩
    exact Real.exp_pos _

/-- The final boundary streaming normalizer is non-zero whenever the
KV scope is non-empty. Boundary analog of `FA1Math.lPartial_final_ne_zero`. -/
theorem lPartial_final_ne_zero {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (hSkLe : S_k ≤ Bk * numKVBlocks)
    (i : Fin M) :
    lPartial Bk Q numKVBlocks K scale numKVBlocks i ≠ 0 := by
  rw [lPartial_eq_mShifted hBk hSk Q numKVBlocks K scale numKVBlocks
      (le_refl _) i]
  exact mul_ne_zero (Real.exp_ne_zero _)
    (ne_of_gt (lFreeBoundary_final_pos Bk hSk Q K scale numKVBlocks hSkLe i))

/-- When the logical KV scope is empty (`S_k = 0`), every loop lane is
out-of-range, so the running normalizer stays at zero. This is the
contradiction that excludes the degenerate case from
`streaming_eq_attentionReal`. -/
theorem lPartial_eq_zero_of_S_k_zero {M D : Nat} {Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [0, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) :
    lPartial Bk Q numKVBlocks K scale k i = 0 := by
  induction k with
  | zero => rfl
  | succ k ih =>
      by_cases hk : k + 1 ≤ numKVBlocks
      · rw [lPartial_succ_of_lt Bk Q numKVBlocks K scale k
            (Nat.lt_of_succ_le hk) i]
        simp only
        rw [ih]
        ring_nf
        apply Finset.sum_eq_zero
        intro jLocal _
        have hOOR : ¬ k * Bk + jLocal.val < 0 := by omega
        rw [maskedScore_of_not_lt Bk k Q K scale i jLocal hOOR]
        cases mPartial Bk Q numKVBlocks K scale (1 + k) i with
        | bot => rfl
        | coe _ => rfl
      · -- Out-of-loop branch: lPartial (k+1) = lPartial k = 0.
        change (if h : k + 1 ≤ numKVBlocks then _ else
            lPartial Bk Q numKVBlocks K scale k i) = 0
        rw [dif_neg hk]
        exact ih

/-- **Math identity (boundary).** After all `numKVBlocks` iterations,
the boundary streaming `(oPartial, lPartial)` ratio computes the same
value as `attentionReal` on the logical `[S_k, D]` KV domain. The proof
factors `exp(-m_N)` out of both numerator and denominator and matches
the residual against `attentionReal`. -/
theorem streaming_eq_attentionReal {M D Bk : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) {S_k numKVBlocks : Nat}
    (hSkLe : S_k ≤ Bk * numKVBlocks)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D])
    (hL : lPartial Bk Q numKVBlocks K scale numKVBlocks idx.1 ≠ 0) :
    oPartial Bk Q numKVBlocks K V scale numKVBlocks idx /
        lPartial Bk Q numKVBlocks K scale numKVBlocks idx.1
      = attentionReal Q K V scale idx := by
  -- We need `0 < S_k` to invoke the m-shifted bridge. Derive it from
  -- `hL`: if `S_k = 0`, then `lPartial` is identically zero
  -- (no in-range lane ever contributes), contradicting `hL`.
  rcases Nat.eq_zero_or_pos S_k with hSkz | hSk
  · subst hSkz
    exact (hL (lPartial_eq_zero_of_S_k_zero Q numKVBlocks K scale numKVBlocks
      idx.1)).elim
  · -- 0 < S_k: standard cancellation via lPartial/oPartial_eq_mShifted.
    rw [oPartial_eq_mShifted hBk hSk Q numKVBlocks K V scale numKVBlocks
          (le_refl _) idx,
        lPartial_eq_mShifted hBk hSk Q numKVBlocks K scale numKVBlocks
          (le_refl _) idx.1]
    rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
    have hlFree : lFreeBoundary Bk Q K scale numKVBlocks idx.1 ≠ 0 := by
      intro h
      apply hL
      rw [lPartial_eq_mShifted hBk hSk Q numKVBlocks K scale numKVBlocks
          (le_refl _) idx.1, h, mul_zero]
    exact oFreeBoundary_div_lFreeBoundary_eq_attentionReal Bk Q K V scale
      numKVBlocks hSkLe idx hlFree

end FA1MathBoundary

/-! ## Causal boundary streaming math model

Combines the KV boundary mask with the causal `j <= qStart + i` mask.
Out-of-range or future lanes are represented as `⊥`, matching the
causal-boundary kernel's two `tl.where(..., -inf)` stages. -/

namespace FA1MathCausalBoundary

/-- `WithBot.realExp` is never bottom, so it is equal to its `unbotD`
payload rewrapped as `some`. -/
theorem realExp_eq_some_unbotD (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

@[simp] theorem realExp_realSub_bot_unbotD (m : WithBot ℝ) :
    (WithBot.realExp (WithBot.realSub (⊥ : WithBot ℝ) m)).unbotD 0 = 0 := by
  cases m <;> rfl

@[simp] theorem realExp_optionMap_sub_bot_unbotD (m : WithBot ℝ) :
    (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y) (⊥ : WithBot ℝ) m)).unbotD 0 = 0 := by
  cases m <;> rfl

/-- Causal + boundary masked score for a padded loop lane. -/
noncomputable def maskedScore {M S_k D : Nat}
    (Bk k qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk) : WithBot ℝ :=
  match FA1MathBoundary.blockIndex? S_k Bk k jLocal with
  | some j =>
      if j.val ≤ qStart + i.val then
        (FA1Math.scaledScore Q K scale i j : ℝ)
      else
        ⊥
  | none => ⊥

@[simp] theorem maskedScore_of_lt_of_le {M S_k D : Nat}
    (Bk k qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk)
    (hLt : k * Bk + jLocal.val < S_k)
    (hLe : k * Bk + jLocal.val ≤ qStart + i.val) :
    maskedScore Bk k qStart Q K scale i jLocal =
      ((FA1Math.scaledScore Q K scale i
        ⟨k * Bk + jLocal.val, hLt⟩ : ℝ) : WithBot ℝ) := by
  simp [maskedScore, FA1MathBoundary.blockIndex?_of_lt, hLt, hLe]

@[simp] theorem maskedScore_of_lt_of_not_le {M S_k D : Nat}
    (Bk k qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk)
    (hLt : k * Bk + jLocal.val < S_k)
    (hLe : ¬ k * Bk + jLocal.val ≤ qStart + i.val) :
    maskedScore Bk k qStart Q K scale i jLocal = (⊥ : WithBot ℝ) := by
  simp [maskedScore, FA1MathBoundary.blockIndex?_of_lt, hLt, hLe]

@[simp] theorem maskedScore_of_not_lt {M S_k D : Nat}
    (Bk k qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk)
    (hLt : ¬ k * Bk + jLocal.val < S_k) :
    maskedScore Bk k qStart Q K scale i jLocal = (⊥ : WithBot ℝ) := by
  simp [maskedScore, FA1MathBoundary.blockIndex?_of_not_lt, hLt]

/-- Running per-row max over the first `k` padded KV blocks. -/
noncomputable def mPartial {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if _h : k + 1 ≤ numKVBlocks then
        max (mPartial Bk qStart Q numKVBlocks K scale k i)
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            maskedScore Bk k qStart Q K scale i jLocal)
      else
        mPartial Bk qStart Q numKVBlocks K scale k i

/-- Boundary-causal α multiplier `exp(m_k - m_{k+1})`. -/
noncomputable def alphaPartial {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  (WithBot.realExp
    (WithBot.realSub
      (mPartial Bk qStart Q numKVBlocks K scale k i)
      (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0

/-- Boundary-causal running normalizer. -/
noncomputable def lPartial {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if _h : k + 1 ≤ numKVBlocks then
        let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
        alphaPartial Bk qStart Q numKVBlocks K scale k i *
          lPartial Bk qStart Q numKVBlocks K scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0)
      else
        lPartial Bk qStart Q numKVBlocks K scale k i

/-- Boundary-causal running unnormalized output accumulator. -/
noncomputable def oPartial {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, D] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if _h : k + 1 ≤ numKVBlocks then
        let i := idx.1
        let d := idx.2.1
        let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
        alphaPartial Bk qStart Q numKVBlocks K scale k i *
          oPartial Bk qStart Q numKVBlocks K V scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match FA1MathBoundary.blockIndex? S_k Bk k jLocal with
            | some j =>
                if j.val ≤ qStart + i.val then
                  (WithBot.realExp
                    (WithBot.realSub
                      (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0 *
                    V (j, d, PUnit.unit)
                else
                  0
            | none => 0)
      else
        oPartial Bk qStart Q numKVBlocks K V scale k idx

theorem mPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    mPartial Bk qStart Q numKVBlocks K scale (k + 1) i =
      max (mPartial Bk qStart Q numKVBlocks K scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore Bk k qStart Q K scale i jLocal) := by
  change (if h : k + 1 ≤ numKVBlocks then
      max (mPartial Bk qStart Q numKVBlocks K scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore Bk k qStart Q K scale i jLocal)
    else mPartial Bk qStart Q numKVBlocks K scale k i) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem lPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale (k + 1) i =
      let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
      alphaPartial Bk qStart Q numKVBlocks K scale k i *
        lPartial Bk qStart Q numKVBlocks K scale k i +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          (WithBot.realExp
            (WithBot.realSub
              (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
       alphaPartial Bk qStart Q numKVBlocks K scale k i *
          lPartial Bk qStart Q numKVBlocks K scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0))
    else lPartial Bk qStart Q numKVBlocks K scale k i) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

theorem oPartial_succ_of_lt {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale (k + 1) idx =
      let i := idx.1
      let d := idx.2.1
      let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
      alphaPartial Bk qStart Q numKVBlocks K scale k i *
        oPartial Bk qStart Q numKVBlocks K V scale k idx +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          match FA1MathBoundary.blockIndex? S_k Bk k jLocal with
          | some j =>
              if j.val ≤ qStart + i.val then
                (WithBot.realExp
                  (WithBot.realSub
                    (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0 *
                  V (j, d, PUnit.unit)
              else
                0
          | none => 0) := by
  change (if h : k + 1 ≤ numKVBlocks then
      (let i := idx.1
       let d := idx.2.1
       let mNew := mPartial Bk qStart Q numKVBlocks K scale (k + 1) i
       alphaPartial Bk qStart Q numKVBlocks K scale k i *
          oPartial Bk qStart Q numKVBlocks K V scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match FA1MathBoundary.blockIndex? S_k Bk k jLocal with
            | some j =>
                if j.val ≤ qStart + i.val then
                  (WithBot.realExp
                    (WithBot.realSub
                      (maskedScore Bk k qStart Q K scale i jLocal) mNew)).unbotD 0 *
                    V (j, d, PUnit.unit)
                else
                  0
            | none => 0))
    else oPartial Bk qStart Q numKVBlocks K V scale k idx) = _
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-- Boundary score mask `offs_n < S_k` lifted to the `[M, Bk]` score tile. -/
theorem block_score_mask_tile_eq {M Bk S_k : Nat} (k : Nat) :
    (Tile.cop (ComparableDType.nat.lt)
      Broadcast.scalarR
      (Tile.bop NumericDType.nat.add (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Tile.bop NumericDType.nat.mul Broadcast.scalarR
          (Tile.expandDim ⟨1, by simp⟩
            (Tile.vec (fun i : Fin M => i.val)))
          (Tile.scalar 0))
        (Tile.expandDim ⟨0, by simp⟩
          (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      (Tile.scalar S_k))
      = ⟨fun idx : TileIndex [M, Bk] =>
          decide (k * Bk + idx.2.1.val < S_k)⟩ := by
  ext idx
  obtain ⟨i, j, _⟩ := idx
  simp only [Tile.cop, Tile.bop, ComparableDType.lt, NumericDType.add,
    NumericDType.mul, Tile.expandDim_data, Tile.vec_data, Tile.scalar_data_index,
    Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex]
  simp

/-- Causal mask `offs_m[:, None] >= offs_n[None, :]`, stated against the
same `qStart` used by the recurrence. -/
theorem block_causal_mask_tile_eq {M Bk : Nat} (qStart k : Nat) :
    (Tile.cop (ComparableDType.nat.ge)
      (Broadcast.consR (Broadcast.consL Broadcast.nil))
      (Tile.expandDim ⟨1, by simp⟩
        (Tile.vec (fun i : Fin M => qStart + i.val)))
      (Tile.expandDim ⟨0, by simp⟩
        (Tile.vec (fun j : Fin Bk => k * Bk + j.val))))
      = ⟨fun idx : TileIndex [M, Bk] =>
          decide ((k * Bk + idx.2.1.val) ≤ (qStart + idx.1.val))⟩ := by
  ext idx
  obtain ⟨i, j, _⟩ := idx
  simp [Tile.cop, ComparableDType.ge, Tile.expandDim, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]

/-- Applying the causal mask and then the boundary score mask to the raw
score tile produces exactly `maskedScore`. -/
theorem block_scores_tile_eq {M S_k D Bk : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) :
    Tile.select
      (⟨fun idx : TileIndex [M, Bk] =>
        decide (k * Bk + idx.2.1.val < S_k)⟩ : Tile .bool [M, Bk])
      (Tile.select
        (⟨fun idx : TileIndex [M, Bk] =>
          decide ((k * Bk + idx.2.1.val) ≤ (qStart + idx.1.val))⟩ :
            Tile .bool [M, Bk])
        (Tile.ofReal (fun idx : TileIndex [M, Bk] =>
          match FA1MathBoundary.blockIndex? S_k Bk k idx.2.1 with
          | some j => FA1Math.scaledScore Q K scale idx.1 j
          | none => 0))
        (⟨fun _ : TileIndex [M, Bk] => (none : WithBot ℝ)⟩ : Tile .real [M, Bk]))
      (⟨fun _ : TileIndex [M, Bk] => (none : WithBot ℝ)⟩ : Tile .real [M, Bk])
    = ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore Bk k qStart Q K scale idx.1 idx.2.1⟩ := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  cases u
  simp only [Tile.select_data, Tile.ofReal_data]
  by_cases hLt : k * Bk + j.val < S_k
  · rw [decide_eq_true hLt]
    simp only [if_true]
    by_cases hLe : k * Bk + j.val ≤ qStart + i.val
    · rw [decide_eq_true hLe]
      simp [FA1MathBoundary.blockIndex?_of_lt, maskedScore_of_lt_of_le,
        hLt, hLe]
      rfl
    · rw [decide_eq_false hLe]
      simp [maskedScore_of_lt_of_not_le, hLt, hLe]
      rfl
  · rw [decide_eq_false hLt]
    simp [maskedScore_of_not_lt, hLt]
    rfl

/-- Row-max of the causal-boundary masked scores tile. -/
theorem block_mBlock_tile_eq {M S_k D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) :
    Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore Bk k qStart Q K scale idx.1 idx.2.1⟩
    = some ⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun jLocal => maskedScore Bk k qStart Q K scale idx.1 jLocal)⟩ := by
  unfold Tile.reduceMax
  simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBk]
  funext idx
  rw [Finset.sup'_eq_sup]
  rfl

/-- Boundary-causal loop-local `m_new = max(m_i, m_block)` realizes the
streaming `mPartial` recurrence. -/
theorem block_mNew_tile_eq {M S_k D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] => mPartial Bk qStart Q N K scale k idx.1⟩ : Tile .real [M])
      (⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun j => maskedScore Bk k qStart Q K scale idx.1 j)⟩ : Tile .real [M])
      =
      ⟨fun idx : TileIndex [M] => mPartial Bk qStart Q N K scale (k + 1) idx.1⟩ := by
  ext idx
  simp [Tile.bop]
  rw [mPartial_succ_of_lt Bk qStart Q N K scale k hk idx.1]

/-- Boundary-causal loop-local `l_new` expression realizes the streaming
`lPartial` recurrence. -/
theorem block_lNew_tile_eq {M S_k D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        (Tile.ofReal fun idx : TileIndex [M] =>
          alphaPartial Bk qStart Q N K scale k idx.1)
        (Tile.ofReal fun idx : TileIndex [M] =>
          lPartial Bk qStart Q N K scale k idx.1))
      (Tile.ofReal fun idx : TileIndex [M] =>
        Finset.univ.sum (fun j : Fin Bk =>
          (WithBot.realExp
            (WithBot.realSub
              (maskedScore Bk k qStart Q K scale idx.1 j)
              (mPartial Bk qStart Q N K scale (k + 1) idx.1))).unbotD 0))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        lPartial Bk qStart Q N K scale (k + 1) idx.1) := by
  ext idx
  simp [Tile.bop, Tile.ofReal]
  rw [lPartial_succ_of_lt Bk qStart Q N K scale k hk idx.1]
  simp [WithBot.realSub]

/-- Boundary-causal loop-local `o_acc` update realizes the streaming
`oPartial` recurrence. -/
theorem block_oAcc_tile_eq {M S_k D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.ofReal fun idx : TileIndex [M] =>
            alphaPartial Bk qStart Q N K scale k idx.1))
        (Tile.ofReal fun idx : TileIndex [M, D] =>
          oPartial Bk qStart Q N K V scale k idx))
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun j : Fin Bk =>
          match FA1MathBoundary.blockIndex? S_k Bk k j with
          | some jGlobal =>
              if jGlobal.val ≤ qStart + idx.1.val then
                (WithBot.realExp
                  (WithBot.realSub
                    (maskedScore Bk k qStart Q K scale idx.1 j)
                    (mPartial Bk qStart Q N K scale (k + 1) idx.1))).unbotD 0 *
                  V (jGlobal, idx.2.1, PUnit.unit)
              else
                0
          | none => 0))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        oPartial Bk qStart Q N K V scale (k + 1) idx) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [oPartial_succ_of_lt Bk qStart Q N K V scale k hk (i, d, PUnit.unit)]
  simp [WithBot.realSub]

/-! ### Causal boundary m-free reference sums -/

/-- Causal-boundary m-free normalizer over the first `k` padded KV blocks.
Out-of-range and future lanes contribute zero mass. -/
noncomputable def lFreeBoundary {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    match FA1MathBoundary.blockIndex? S_k Bk n.val jL with
    | some j =>
        if j.val ≤ qStart + i.val then
          Real.exp (FA1Math.scaledScore Q K scale i j)
        else
          0
    | none => 0))

/-- Causal-boundary m-free unnormalized output over the first `k` padded
KV blocks. -/
noncomputable def oFreeBoundary {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (idx : TileIndex [M, D]) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    match FA1MathBoundary.blockIndex? S_k Bk n.val jL with
    | some j =>
        (if j.val ≤ qStart + idx.1.val then
          Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
        else
          0) * V (j, idx.2.1, PUnit.unit)
    | none => 0))

@[simp] theorem lFreeBoundary_zero {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lFreeBoundary Bk qStart Q K scale 0 i = 0 := by
  unfold lFreeBoundary
  simp

@[simp] theorem oFreeBoundary_zero {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oFreeBoundary Bk qStart Q K V scale 0 idx = 0 := by
  unfold oFreeBoundary
  simp

/-- Recurrence: `lFreeBoundary (k+1)` adds the next masked causal block. -/
theorem lFreeBoundary_succ {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) :
    lFreeBoundary Bk qStart Q K scale (k + 1) i =
      lFreeBoundary Bk qStart Q K scale k i +
      Finset.univ.sum (fun jL : Fin Bk =>
        match FA1MathBoundary.blockIndex? S_k Bk k jL with
        | some j =>
            if j.val ≤ qStart + i.val then
              Real.exp (FA1Math.scaledScore Q K scale i j)
            else
              0
        | none => 0) := by
  unfold lFreeBoundary
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Recurrence: `oFreeBoundary (k+1)` adds the next masked causal block. -/
theorem oFreeBoundary_succ {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (idx : TileIndex [M, D]) :
    oFreeBoundary Bk qStart Q K V scale (k + 1) idx =
      oFreeBoundary Bk qStart Q K V scale k idx +
      Finset.univ.sum (fun jL : Fin Bk =>
        match FA1MathBoundary.blockIndex? S_k Bk k jL with
        | some j =>
            (if j.val ≤ qStart + idx.1.val then
              Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
            else
              0) * V (j, idx.2.1, PUnit.unit)
        | none => 0) := by
  unfold oFreeBoundary
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Flat-index form of the final causal-boundary normalizer. -/
theorem lFreeBoundary_eq_flat {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (i : Fin M) :
    lFreeBoundary Bk qStart Q K scale numKVBlocks i =
      Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          if j'.val ≤ qStart + i.val then
            Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
          else
            0
        else
          0) := by
  unfold lFreeBoundary
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j' _
    set p := FA1Math.blockIndexEquiv Bk numKVBlocks j' with hp
    show (if h : j'.val < S_k then
            if j'.val ≤ qStart + i.val then
              Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
            else
              0
          else
            0)
        = (match FA1MathBoundary.blockIndex? S_k Bk p.1.val p.2 with
          | some j =>
              if j.val ≤ qStart + i.val then
                Real.exp (FA1Math.scaledScore Q K scale i j)
              else
                0
          | none => 0)
    have hValEq : p.1.val * Bk + p.2.val = j'.val := by
      have hBI := FA1Math.blockIndex_blockIndexEquiv (Bk := Bk) (N := numKVBlocks) j'
      have h1 := congrArg Fin.val hBI
      change p.1.val * Bk + p.2.val = j'.val at h1
      exact h1
    by_cases hLt : p.1.val * Bk + p.2.val < S_k
    · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : j'.val < S_k := hValEq ▸ hLt
      rw [dif_pos hLt']
      have hFinEq : (⟨p.1.val * Bk + p.2.val, hLt⟩ : Fin S_k) =
          ⟨j'.val, hLt'⟩ := Fin.ext hValEq
      rw [hFinEq]
    · rw [FA1MathBoundary.blockIndex?_of_not_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : ¬ j'.val < S_k := hValEq ▸ hLt
      rw [dif_neg hLt']

/-- Flat-index form of the final causal-boundary output accumulator. -/
theorem oFreeBoundary_eq_flat {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (idx : TileIndex [M, D]) :
    oFreeBoundary Bk qStart Q K V scale numKVBlocks idx =
      Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          (if j'.val ≤ qStart + idx.1.val then
            Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩)
          else
            0) * V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
        else
          0) := by
  unfold oFreeBoundary
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk numKVBlocks) ?_ ?_).symm
  · intro _; simp
  · intro j' _
    set p := FA1Math.blockIndexEquiv Bk numKVBlocks j' with hp
    show (if h : j'.val < S_k then
            (if j'.val ≤ qStart + idx.1.val then
              Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩)
            else
              0) * V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
          else
            0)
        = (match FA1MathBoundary.blockIndex? S_k Bk p.1.val p.2 with
          | some j =>
              (if j.val ≤ qStart + idx.1.val then
                Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
              else
                0) * V (j, idx.2.1, PUnit.unit)
          | none => 0)
    have hValEq : p.1.val * Bk + p.2.val = j'.val := by
      have hBI := FA1Math.blockIndex_blockIndexEquiv (Bk := Bk) (N := numKVBlocks) j'
      have h1 := congrArg Fin.val hBI
      change p.1.val * Bk + p.2.val = j'.val at h1
      exact h1
    by_cases hLt : p.1.val * Bk + p.2.val < S_k
    · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : j'.val < S_k := hValEq ▸ hLt
      rw [dif_pos hLt']
      have hFinEq : (⟨p.1.val * Bk + p.2.val, hLt⟩ : Fin S_k) =
          ⟨j'.val, hLt'⟩ := Fin.ext hValEq
      rw [hFinEq]
    · rw [FA1MathBoundary.blockIndex?_of_not_lt S_k Bk p.1.val p.2 hLt]
      have hLt' : ¬ j'.val < S_k := hValEq ▸ hLt
      rw [dif_neg hLt']

/-- Reindex the final causal-boundary normalizer from padded loop lanes to
the logical KV domain. Out-of-range padded lanes contribute zero. -/
theorem lFreeBoundary_final_eq_finSk_sum {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks) (i : Fin M) :
    lFreeBoundary Bk qStart Q K scale numKVBlocks i =
      Finset.univ.sum (fun j : Fin S_k =>
        if j.val ≤ qStart + i.val then
          Real.exp (FA1Math.scaledScore Q K scale i j)
        else
          0) := by
  rw [lFreeBoundary_eq_flat]
  rw [show (Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          if j'.val ≤ qStart + i.val then
            Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
          else
            0
        else
          0)) =
        ((Finset.univ : Finset (Fin (Bk * numKVBlocks))).filter
          (fun j' => j'.val < S_k)).sum
          (fun j' : Fin (Bk * numKVBlocks) =>
            if h : j'.val < S_k then
              if j'.val ≤ qStart + i.val then
                Real.exp (FA1Math.scaledScore Q K scale i ⟨j'.val, h⟩)
              else
                0
            else
              0)
        from ?_]
  · refine (Finset.sum_bij (fun (j : Fin S_k) (_ : j ∈ Finset.univ) =>
      Fin.castLE hSkLe j) ?_ ?_ ?_ ?_).symm
    · intro j _
      simp [Fin.val_castLE, j.isLt]
    · intro j₁ _ j₂ _ heq
      apply Fin.ext
      have := congrArg Fin.val heq
      simpa [Fin.val_castLE] using this
    · intro j' hj'
      simp at hj'
      refine ⟨⟨j'.val, hj'⟩, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      simp
    · intro j _
      have hLt : (Fin.castLE hSkLe j).val < S_k := by
        rw [Fin.val_castLE]; exact j.isLt
      rw [dif_pos hLt]
      simp [Fin.val_castLE]
  · refine (Finset.sum_filter_of_ne ?_).symm
    intro j' _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

/-- Reindex the final causal-boundary output accumulator from padded loop
lanes to the logical KV domain. -/
theorem oFreeBoundary_final_eq_finSk_sum {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (idx : TileIndex [M, D]) :
    oFreeBoundary Bk qStart Q K V scale numKVBlocks idx =
      Finset.univ.sum (fun j : Fin S_k =>
        (if j.val ≤ qStart + idx.1.val then
          Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
        else
          0) * V (j, idx.2.1, PUnit.unit)) := by
  rw [oFreeBoundary_eq_flat]
  rw [show (Finset.univ.sum (fun j' : Fin (Bk * numKVBlocks) =>
        if h : j'.val < S_k then
          (if j'.val ≤ qStart + idx.1.val then
            Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩)
          else
            0) * V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
        else
          0)) =
        ((Finset.univ : Finset (Fin (Bk * numKVBlocks))).filter
          (fun j' => j'.val < S_k)).sum
          (fun j' : Fin (Bk * numKVBlocks) =>
            if h : j'.val < S_k then
              (if j'.val ≤ qStart + idx.1.val then
                Real.exp (FA1Math.scaledScore Q K scale idx.1 ⟨j'.val, h⟩)
              else
                0) * V (⟨j'.val, h⟩, idx.2.1, PUnit.unit)
            else
              0)
        from ?_]
  · refine (Finset.sum_bij (fun (j : Fin S_k) (_ : j ∈ Finset.univ) =>
      Fin.castLE hSkLe j) ?_ ?_ ?_ ?_).symm
    · intro j _
      simp [Fin.val_castLE, j.isLt]
    · intro j₁ _ j₂ _ heq
      apply Fin.ext
      have := congrArg Fin.val heq
      simpa [Fin.val_castLE] using this
    · intro j' hj'
      simp at hj'
      refine ⟨⟨j'.val, hj'⟩, Finset.mem_univ _, ?_⟩
      apply Fin.ext
      simp
    · intro j _
      have hLt : (Fin.castLE hSkLe j).val < S_k := by
        rw [Fin.val_castLE]; exact j.isLt
      rw [dif_pos hLt]
      simp [Fin.val_castLE]
  · refine (Finset.sum_filter_of_ne ?_).symm
    intro j' _ hNe
    by_contra hLt
    apply hNe
    rw [dif_neg hLt]

/-- Causal-boundary m-free ratio is exactly the local-block causal attention
spec over the logical KV domain. -/
theorem oFreeBoundary_div_lFreeBoundary_eq_attentionRealCausalBlock
    {M S_k D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (idx : TileIndex [M, D])
    (_hlFree : lFreeBoundary Bk qStart Q K scale numKVBlocks idx.1 ≠ 0) :
    oFreeBoundary Bk qStart Q K V scale numKVBlocks idx /
        lFreeBoundary Bk qStart Q K scale numKVBlocks idx.1
      = attentionRealCausalBlock qStart Q K V scale idx := by
  rw [oFreeBoundary_final_eq_finSk_sum Bk qStart Q K V scale numKVBlocks hSkLe idx]
  rw [lFreeBoundary_final_eq_finSk_sum Bk qStart Q K scale numKVBlocks hSkLe idx.1]
  unfold attentionRealCausalBlock
  rfl

/-- The final causal-boundary m-free normalizer is positive when the logical
KV domain is non-empty: key 0 is in range and causally visible. -/
theorem lFreeBoundary_final_pos {M S_k D : Nat} (Bk : Nat)
    (hSk : 0 < S_k)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (numKVBlocks : Nat) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (i : Fin M) :
    0 < lFreeBoundary Bk qStart Q K scale numKVBlocks i := by
  rw [lFreeBoundary_final_eq_finSk_sum Bk qStart Q K scale numKVBlocks hSkLe i]
  apply Finset.sum_pos'
  · intro j _
    by_cases h : j.val ≤ qStart + i.val
    · simp [h, le_of_lt (Real.exp_pos _)]
    · simp [h]
  · refine ⟨⟨0, hSk⟩, Finset.mem_univ _, ?_⟩
    simp [Real.exp_pos]

/-- Once the loop has processed at least one block and `S_k > 0`,
causal-boundary `mPartial` is non-bottom. The first logical key is both
in range and causally visible. -/
theorem mPartial_succ_ne_bot {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    mPartial Bk qStart Q numKVBlocks K scale (k + 1) i ≠ ⊥ := by
  induction k with
  | zero =>
      rw [mPartial_succ_of_lt Bk qStart Q numKVBlocks K scale 0
            (Nat.lt_of_succ_le hk) i]
      have hLt : 0 * Bk + (⟨0, hBk⟩ : Fin Bk).val < S_k := by
        simp; exact hSk
      have hLe : 0 * Bk + (⟨0, hBk⟩ : Fin Bk).val ≤ qStart + i.val := by
        simp
      have h0 : maskedScore Bk 0 qStart Q K scale i (⟨0, hBk⟩ : Fin Bk) ≠ ⊥ := by
        rw [maskedScore_of_lt_of_le Bk 0 qStart Q K scale i _ hLt hLe]
        exact WithBot.coe_ne_bot
      show mPartial Bk qStart Q numKVBlocks K scale 0 i ⊔ _ ≠ ⊥
      change ⊥ ⊔ _ ≠ ⊥
      rw [bot_sup_eq]
      simp [Finset.sup_eq_bot_iff]
      exact ⟨⟨0, hBk⟩, h0⟩
  | succ k' ih =>
      have hk' : k' + 1 ≤ numKVBlocks := by omega
      rw [mPartial_succ_of_lt Bk qStart Q numKVBlocks K scale (k' + 1)
            (Nat.lt_of_succ_le hk) i]
      intro hcontra
      have h_left : mPartial Bk qStart Q numKVBlocks K scale (k' + 1) i ≤ ⊥ := by
        rw [← hcontra]; exact le_max_left _ _
      exact ih hk' (le_bot_iff.mp h_left)

/-- Causal-boundary streaming normalizer equals the m-free normalizer times
the final exponential shift. -/
theorem lPartial_eq_mShifted {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale k i =
      Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k i).unbotD 0) *
        lFreeBoundary Bk qStart Q K scale k i := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          lFreeBoundary Bk qStart Q K scale 0 i
      rw [lFreeBoundary_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [lPartial_succ_of_lt Bk qStart Q numKVBlocks K scale k
        (Nat.lt_of_succ_le hk) i]
      rw [ih hk']
      rw [lFreeBoundary_succ Bk qStart Q K scale k i, mul_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (WithBot.realSub
                (maskedScore Bk k qStart Q K scale i jLocal)
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jL : Fin Bk =>
              match FA1MathBoundary.blockIndex? S_k Bk k jL with
              | some j =>
                  if j.val ≤ qStart + i.val then
                    Real.exp (FA1Math.scaledScore Q K scale i j)
                  else
                    0
              | none => 0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        by_cases hLt : k * Bk + jLocal.val < S_k
        · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk k jLocal hLt]
          by_cases hLe : k * Bk + jLocal.val ≤ qStart + i.val
          · rw [maskedScore_of_lt_of_le Bk k qStart Q K scale i jLocal hLt hLe]
            obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
              (mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale k hk i)
            rw [← hm]
            simp [hLe, WithBot.realSub]
            rw [show FA1Math.scaledScore Q K scale i ⟨k * Bk + jLocal.val, hLt⟩ - m =
                  -m + FA1Math.scaledScore Q K scale i ⟨k * Bk + jLocal.val, hLt⟩
                  by ring,
                Real.exp_add]
          · rw [maskedScore_of_lt_of_not_le Bk k qStart Q K scale i jLocal hLt hLe]
            simp [hLe]
        · rw [maskedScore_of_not_lt Bk k qStart Q K scale i jLocal hLt,
            FA1MathBoundary.blockIndex?_of_not_lt S_k Bk k jLocal hLt]
          simp
      have hSumA :
          alphaPartial Bk qStart Q numKVBlocks K scale k i *
            (Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k i).unbotD 0) *
              lFreeBoundary Bk qStart Q K scale k i)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            lFreeBoundary Bk qStart Q K scale k i := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [lFreeBoundary_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne :=
            mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale k' hk_succ i
          have hmk1_ne :=
            mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale (k' + 1) hk i
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          have hAlpha :
              alphaPartial Bk qStart Q numKVBlocks K scale (k' + 1) i =
                Real.exp (rk - rk1) := by
            unfold alphaPartial
            rw [← hrk, ← hrk1]
            simp [WithBot.realSub]
          rw [hAlpha, ← hrk, ← hrk1]
          simp only [WithBot.unbotD_coe]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
                rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * lFreeBoundary Bk qStart Q K scale (k' + 1) i)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      lFreeBoundary Bk qStart Q K scale (k' + 1) i) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
                rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Causal-boundary streaming output accumulator equals the m-free output
accumulator times the final exponential shift. -/
theorem oPartial_eq_mShifted {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale k idx =
      Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k idx.1).unbotD 0) *
        oFreeBoundary Bk qStart Q K V scale k idx := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          oFreeBoundary Bk qStart Q K V scale 0 idx
      rw [oFreeBoundary_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [oPartial_succ_of_lt Bk qStart Q numKVBlocks K V scale k
        (Nat.lt_of_succ_le hk) idx]
      rw [ih hk']
      rw [oFreeBoundary_succ Bk qStart Q K V scale k idx, mul_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            match FA1MathBoundary.blockIndex? S_k Bk k jLocal with
            | some j =>
                if j.val ≤ qStart + idx.1.val then
                  (WithBot.realExp
                    (WithBot.realSub
                      (maskedScore Bk k qStart Q K scale idx.1 jLocal)
                      (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
                    V (j, idx.2.1, PUnit.unit)
                else
                  0
            | none => 0)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jL : Fin Bk =>
              match FA1MathBoundary.blockIndex? S_k Bk k jL with
              | some j =>
                  (if j.val ≤ qStart + idx.1.val then
                    Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
                  else
                    0) * V (j, idx.2.1, PUnit.unit)
              | none => 0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        by_cases hLt : k * Bk + jLocal.val < S_k
        · rw [FA1MathBoundary.blockIndex?_of_lt S_k Bk k jLocal hLt]
          by_cases hLe : k * Bk + jLocal.val ≤ qStart + idx.1.val
          · rw [maskedScore_of_lt_of_le Bk k qStart Q K scale idx.1 jLocal hLt hLe]
            obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
              (mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale k hk idx.1)
            rw [← hm]
            simp [hLe, WithBot.realSub]
            rw [show FA1Math.scaledScore Q K scale idx.1 ⟨k * Bk + jLocal.val, hLt⟩ - m =
                  -m + FA1Math.scaledScore Q K scale idx.1 ⟨k * Bk + jLocal.val, hLt⟩
                  by ring,
                Real.exp_add]
            ring
          · rw [maskedScore_of_lt_of_not_le Bk k qStart Q K scale idx.1 jLocal hLt hLe]
            simp [hLe]
        · rw [FA1MathBoundary.blockIndex?_of_not_lt S_k Bk k jLocal hLt]
          simp
      have hSumA :
          alphaPartial Bk qStart Q numKVBlocks K scale k idx.1 *
            (Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k idx.1).unbotD 0) *
              oFreeBoundary Bk qStart Q K V scale k idx)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            oFreeBoundary Bk qStart Q K V scale k idx := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [oFreeBoundary_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne :=
            mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale k' hk_succ idx.1
          have hmk1_ne :=
            mPartial_succ_ne_bot hBk hSk qStart Q numKVBlocks K scale (k' + 1) hk idx.1
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          have hAlpha :
              alphaPartial Bk qStart Q numKVBlocks K scale (k' + 1) idx.1 =
                Real.exp (rk - rk1) := by
            unfold alphaPartial
            rw [← hrk, ← hrk1]
            simp [WithBot.realSub]
          rw [hAlpha, ← hrk, ← hrk1]
          simp only [WithBot.unbotD_coe]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
                rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * oFreeBoundary Bk qStart Q K V scale (k' + 1) idx)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      oFreeBoundary Bk qStart Q K V scale (k' + 1) idx) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
                rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- The final causal-boundary streaming normalizer is nonzero when the
logical KV domain is non-empty. -/
theorem lPartial_final_ne_zero {M S_k D : Nat} {Bk : Nat}
    (hBk : 0 < Bk) (hSk : 0 < S_k)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (hSkLe : S_k ≤ Bk * numKVBlocks)
    (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale numKVBlocks i ≠ 0 := by
  rw [lPartial_eq_mShifted hBk hSk qStart Q numKVBlocks K scale numKVBlocks
      (le_refl _) i]
  exact mul_ne_zero (Real.exp_ne_zero _)
    (ne_of_gt (lFreeBoundary_final_pos Bk hSk qStart Q K scale numKVBlocks hSkLe i))

/-- Final causal-boundary streaming ratio equals the local-block causal
attention spec over the logical `[S_k, D]` KV domain. -/
theorem streaming_eq_attentionRealCausalBlock {M D Bk : Nat} (hBk : 0 < Bk)
    {S_k numKVBlocks : Nat} (hSk : 0 < S_k) (hSkLe : S_k ≤ Bk * numKVBlocks)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale numKVBlocks idx /
        lPartial Bk qStart Q numKVBlocks K scale numKVBlocks idx.1
      = attentionRealCausalBlock qStart Q K V scale idx := by
  have hl : lPartial Bk qStart Q numKVBlocks K scale numKVBlocks idx.1 ≠ 0 :=
    lPartial_final_ne_zero hBk hSk qStart Q numKVBlocks K scale hSkLe idx.1
  rw [oPartial_eq_mShifted hBk hSk qStart Q numKVBlocks K V scale numKVBlocks
        (le_refl _) idx,
      lPartial_eq_mShifted hBk hSk qStart Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  have hlFree : lFreeBoundary Bk qStart Q K scale numKVBlocks idx.1 ≠ 0 := by
    intro h
    apply hl
    rw [lPartial_eq_mShifted hBk hSk qStart Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1, h, mul_zero]
  exact oFreeBoundary_div_lFreeBoundary_eq_attentionRealCausalBlock Bk qStart Q K V scale
    numKVBlocks hSkLe idx hlFree

end FA1MathCausalBoundary

/-! ## Causal streaming math model

The causal loop is the same online-softmax recurrence, but each block
score is first filtered by the global predicate
`key_index ≤ qStart + query_lane`. Filtered-out scores are represented
as `⊥`, so `WithBot.realExp` contributes exactly zero softmax mass,
matching the kernel's `tl.where(..., -inf)` followed by `tl.exp`.
-/

namespace FA1MathCausal

/-- Causal scaled score. Returns `⊥` for future keys, otherwise the
ordinary scaled dot-product score. -/
noncomputable def maskedScore {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) : WithBot ℝ :=
  if j.val ≤ qStart + i.val then
    (FA1Math.scaledScore Q K scale i j : ℝ)
  else
    ⊥

@[simp] theorem maskedScore_of_le {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) (h : j.val ≤ qStart + i.val) :
    maskedScore qStart Q K scale i j =
      ((FA1Math.scaledScore Q K scale i j : ℝ) : WithBot ℝ) := by
  simp [maskedScore, h]

@[simp] theorem maskedScore_of_not_le {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) (h : ¬ j.val ≤ qStart + i.val) :
    maskedScore qStart Q K scale i j = (⊥ : WithBot ℝ) := by
  simp [maskedScore, h]

/-- Exponentiating a masked score contributes the ordinary exponent on
causally visible keys and zero on future keys. This is the algebraic
shape produced by `tl.where(..., -inf)` followed by `tl.exp`. -/
theorem realExp_maskedScore_sub {M S D : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i : Fin M) (j : Fin S) (m : WithBot ℝ) :
    WithBot.realExp
        (Option.map₂ (fun x y : ℝ => x - y)
          (maskedScore qStart Q K scale i j) m)
      =
      if j.val ≤ qStart + i.val then
        WithBot.realExp
          (Option.map (fun y : ℝ => FA1Math.scaledScore Q K scale i j - y) m)
      else
        some 0 := by
  by_cases h : j.val ≤ qStart + i.val
  · simp [h, maskedScore]
    cases m <;> rfl
  · simp [h, maskedScore, WithBot.realExp]
    cases m <;> rfl

/-- Running per-row max of causal masked scores over the first `k` KV
blocks. Future keys enter as `⊥`, exactly like `tl.where(..., -inf)`. -/
noncomputable def mPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        max (mPartial Bk qStart Q numKVBlocks K scale k i)
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            maskedScore qStart Q K scale i
              (FA1Math.blockIndex Bk numKVBlocks k h jLocal))
      else
        mPartial Bk qStart Q numKVBlocks K scale k i

/-- Running causal softmax normalizer, shifted by the running max. -/
noncomputable def lPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        let alpha :=
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (mPartial Bk qStart Q numKVBlocks K scale k i)
              (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0
        alpha * lPartial Bk qStart Q numKVBlocks K scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart Q K scale i
                  (FA1Math.blockIndex Bk numKVBlocks k h jLocal))
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0)
      else
        lPartial Bk qStart Q numKVBlocks K scale k i

/-- Running causal unnormalized output accumulator. -/
noncomputable def oPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, D] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if h : k + 1 ≤ numKVBlocks then
        let alpha :=
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (mPartial Bk qStart Q numKVBlocks K scale k idx.1)
              (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0
        alpha * oPartial Bk qStart Q numKVBlocks K V scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := FA1Math.blockIndex Bk numKVBlocks k h jLocal
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart Q K scale idx.1 j)
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
              V (j, idx.2.1, PUnit.unit))
      else
        oPartial Bk qStart Q numKVBlocks K V scale k idx

/-- Recurrence unfold for `mPartial` at iteration `k+1`, when `k < N`. -/
theorem mPartial_succ_of_lt {M D Bk : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    mPartial Bk qStart Q numKVBlocks K scale (k + 1) i =
      max (mPartial Bk qStart Q numKVBlocks K scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore qStart Q K scale i
            (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal)) := by
  conv_lhs => rw [mPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-- Recurrence unfold for `lPartial` at iteration `k+1`, when `k < N`. -/
theorem lPartial_succ_of_lt {M D Bk : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale (k + 1) i =
      let alpha :=
        (WithBot.realExp
          (Option.map₂ (fun x y : ℝ => x - y)
            (mPartial Bk qStart Q numKVBlocks K scale k i)
            (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0
      alpha * lPartial Bk qStart Q numKVBlocks K scale k i +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart Q K scale i
                (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal))
              (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0) := by
  conv_lhs => rw [lPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-- Recurrence unfold for `oPartial` at iteration `k+1`, when `k < N`. -/
theorem oPartial_succ_of_lt {M D Bk : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale (k + 1) idx =
      let alpha :=
        (WithBot.realExp
          (Option.map₂ (fun x y : ℝ => x - y)
            (mPartial Bk qStart Q numKVBlocks K scale k idx.1)
            (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0
      alpha * oPartial Bk qStart Q numKVBlocks K V scale k idx +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          let j := FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart Q K scale idx.1 j)
              (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit)) := by
  conv_lhs => rw [oPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-! ### Causal m-free reference sums

These are the causal counterparts of `FA1Math.lFree` / `FA1Math.oFree`.
They keep the causal mask in the unshifted reference form, so the final
streaming ratio can be compared directly with `attentionRealCausalBlock`.
-/

/-- Causal m-free normalizer over the first `k` KV blocks. Future keys
contribute zero mass. -/
noncomputable def lFree {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k ≤ N) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    let j := FA1Math.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL
    if j.val ≤ qStart + i.val then
      Real.exp (FA1Math.scaledScore Q K scale i j)
    else
      0))

/-- Causal m-free unnormalized output over the first `k` KV blocks. -/
noncomputable def oFree {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k ≤ N)
    (idx : TileIndex [M, D]) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    let j := FA1Math.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL
    (if j.val ≤ qStart + idx.1.val then
      Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
    else
      0) * V (j, idx.2.1, PUnit.unit)))

@[simp] theorem lFree_zero {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    lFree qStart Q K scale 0 (Nat.zero_le _) i = 0 := by
  unfold lFree
  simp

@[simp] theorem oFree_zero {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oFree qStart Q K V scale 0 (Nat.zero_le _) idx = 0 := by
  unfold oFree
  simp

/-- Recurrence for causal `lFree`. -/
theorem lFree_succ {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k + 1 ≤ N) (i : Fin M) :
    lFree qStart Q K scale (k + 1) hk i =
      lFree qStart Q K scale k (Nat.le_of_succ_le hk) i +
      Finset.univ.sum (fun jL : Fin Bk =>
        let j := FA1Math.blockIndex Bk N k hk jL
        if j.val ≤ qStart + i.val then
          Real.exp (FA1Math.scaledScore Q K scale i j)
        else
          0) := by
  unfold lFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Recurrence for causal `oFree`. -/
theorem oFree_succ {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k + 1 ≤ N)
    (idx : TileIndex [M, D]) :
    oFree qStart Q K V scale (k + 1) hk idx =
      oFree qStart Q K V scale k (Nat.le_of_succ_le hk) idx +
      Finset.univ.sum (fun jL : Fin Bk =>
        let j := FA1Math.blockIndex Bk N k hk jL
        (if j.val ≤ qStart + idx.1.val then
          Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
        else
          0) * V (j, idx.2.1, PUnit.unit)) := by
  unfold oFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Flat form of the final causal normalizer. -/
theorem lFree_eq_flat {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    lFree qStart Q K scale N (le_refl N) i =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        if j.val ≤ qStart + i.val then
          Real.exp (FA1Math.scaledScore Q K scale i j)
        else
          0) := by
  unfold lFree
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk N) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [FA1Math.blockIndex_blockIndexEquiv]

/-- Flat form of the final causal output accumulator. -/
theorem oFree_eq_flat {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
    oFree qStart Q K V scale N (le_refl N) idx =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        (if j.val ≤ qStart + idx.1.val then
          Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
        else
          0) * V (j, idx.2.1, PUnit.unit)) := by
  unfold oFree
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (FA1Math.blockIndexEquiv Bk N) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [FA1Math.blockIndex_blockIndexEquiv]

/-- For causal streaming, `mPartial (k+1) i` is non-`⊥` whenever there
is at least one block (`hBk : 0 < Bk`) and at least one iteration
(`hk : k+1 ≤ numKVBlocks`). The first block's first lane has global
index `0`, which is always causally visible (`0 ≤ qStart + i.val`),
giving a `some _` contribution that propagates via running max. -/
theorem mPartial_succ_ne_bot {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M) :
    mPartial Bk qStart Q numKVBlocks K scale (k + 1) i ≠ ⊥ := by
  induction k with
  | zero =>
      rw [mPartial_succ_of_lt qStart Q numKVBlocks K scale 0
            (Nat.lt_of_succ_le hk) i]
      have hVis : (FA1Math.blockIndex Bk numKVBlocks 0
              (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) (⟨0, hBk⟩ : Fin Bk)).val
              ≤ qStart + i.val := by
        simp [FA1Math.blockIndex]
      have h0 : maskedScore qStart Q K scale i
              (FA1Math.blockIndex Bk numKVBlocks 0
                (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk))
                (⟨0, hBk⟩ : Fin Bk)) ≠ ⊥ := by
        rw [maskedScore_of_le qStart Q K scale i _ hVis]
        exact WithBot.coe_ne_bot
      show mPartial Bk qStart Q numKVBlocks K scale 0 i ⊔ _ ≠ ⊥
      change ⊥ ⊔ _ ≠ ⊥
      rw [bot_sup_eq]
      simp [Finset.sup_eq_bot_iff]
      exact ⟨⟨0, hBk⟩, h0⟩
  | succ k' ih =>
      have hk' : k' + 1 ≤ numKVBlocks := by omega
      rw [mPartial_succ_of_lt qStart Q numKVBlocks K scale (k' + 1)
            (Nat.lt_of_succ_le hk) i]
      intro hcontra
      have h_left : mPartial Bk qStart Q numKVBlocks K scale (k' + 1) i ≤ ⊥ := by
        rw [← hcontra]; exact le_max_left _ _
      exact ih hk' (le_bot_iff.mp h_left)

/-- Causal streaming normalizer equals the causal m-free normalizer times
the final exponential shift. -/
theorem lPartial_eq_mShifted {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale k i =
      Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k i).unbotD 0) *
        lFree qStart Q K scale k hk i := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          lFree qStart Q K scale 0 hk i
      rw [lFree_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [lPartial_succ_of_lt qStart Q numKVBlocks K scale k
        (Nat.lt_of_succ_le hk) i]
      rw [ih hk']
      rw [lFree_succ qStart Q K scale k hk i, mul_add]
      have hExpSub : ∀ s : ℝ,
          Real.exp (s - (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) =
            Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) *
              Real.exp s := by
        intro s
        rw [show s - (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0
              = -(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0 + s by ring,
            Real.exp_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart Q K scale i
                  (FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal))
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
              let j := FA1Math.blockIndex Bk numKVBlocks k hk jLocal
              if j.val ≤ qStart + i.val then
                Real.exp (FA1Math.scaledScore Q K scale i j)
              else
                0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        let j := FA1Math.blockIndex Bk numKVBlocks k hk jLocal
        have hjEq :
            FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal = j := by
          rfl
        rw [hjEq]
        by_cases hvis : j.val ≤ qStart + i.val
        · rw [maskedScore_of_le qStart Q K scale i j hvis]
          obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
            (mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale k hk i)
          rw [← hm]
          simp [hvis]
          rw [show FA1Math.scaledScore Q K scale i j - m = -m + FA1Math.scaledScore Q K scale i j by ring,
              Real.exp_add]
        · rw [maskedScore_of_not_le qStart Q K scale i j hvis]
          simp [hvis]
      have hSumA :
          (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (mPartial Bk qStart Q numKVBlocks K scale k i)
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0 *
            (Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k i).unbotD 0) *
              lFree qStart Q K scale k hk' i)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) i).unbotD 0) *
            lFree qStart Q K scale k hk' i := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [lFree_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne := mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale k' hk_succ i
          have hmk1_ne := mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale (k' + 1) hk i
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          rw [← hrk, ← hrk1]
          simp [WithBot.realExp]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
            rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * lFree qStart Q K scale (k' + 1) hk' i)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      lFree qStart Q K scale (k' + 1) hk' i) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
            rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Causal streaming output accumulator equals the causal m-free output
accumulator times the final exponential shift. -/
theorem oPartial_eq_mShifted {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale k idx =
      Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k idx.1).unbotD 0) *
        oFree qStart Q K V scale k hk idx := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          oFree qStart Q K V scale 0 hk idx
      rw [oFree_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [oPartial_succ_of_lt qStart Q numKVBlocks K V scale k
        (Nat.lt_of_succ_le hk) idx]
      rw [ih hk']
      rw [oFree_succ qStart Q K V scale k hk idx, mul_add]
      have hExpSub : ∀ s : ℝ,
          Real.exp (s - (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) =
            Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
              Real.exp s := by
        intro s
        rw [show s - (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0
              = -(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0 + s by ring,
            Real.exp_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := FA1Math.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart Q K scale idx.1 j)
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
              V (j, idx.2.1, PUnit.unit))
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
              let j := FA1Math.blockIndex Bk numKVBlocks k hk jLocal
              (if j.val ≤ qStart + idx.1.val then
                Real.exp (FA1Math.scaledScore Q K scale idx.1 j)
              else
                0) * V (j, idx.2.1, PUnit.unit)) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        let j := FA1Math.blockIndex Bk numKVBlocks k hk jLocal
        have hjEq :
            FA1Math.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal = j := by
          rfl
        rw [hjEq]
        dsimp only
        by_cases hvis : j.val ≤ qStart + idx.1.val
        · rw [maskedScore_of_le qStart Q K scale idx.1 j hvis]
          obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
            (mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale k hk idx.1)
          rw [← hm]
          simp [hvis]
          rw [show FA1Math.scaledScore Q K scale idx.1 j - m =
                -m + FA1Math.scaledScore Q K scale idx.1 j by ring,
              Real.exp_add]
          ring
        · rw [maskedScore_of_not_le qStart Q K scale idx.1 j hvis]
          simp [hvis]
      have hSumA :
          (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (mPartial Bk qStart Q numKVBlocks K scale k idx.1)
                (mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1))).unbotD 0 *
            (Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale k idx.1).unbotD 0) *
              oFree qStart Q K V scale k hk' idx)
          =
          Real.exp (-(mPartial Bk qStart Q numKVBlocks K scale (k + 1) idx.1).unbotD 0) *
            oFree qStart Q K V scale k hk' idx := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [oFree_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne := mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale k' hk_succ idx.1
          have hmk1_ne := mPartial_succ_ne_bot hBk qStart Q numKVBlocks K scale (k' + 1) hk idx.1
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          rw [← hrk, ← hrk1]
          simp [WithBot.realExp]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
            rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * oFree qStart Q K V scale (k' + 1) hk' idx)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      oFree qStart Q K V scale (k' + 1) hk' idx) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
            rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Causal m-free ratio is exactly the local-block causal attention spec. -/
theorem oFree_div_lFree_eq_attentionRealCausalBlock {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K V : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D])
    (_hlFree : lFree qStart Q K scale N (le_refl N) idx.1 ≠ 0) :
    oFree qStart Q K V scale N (le_refl N) idx /
        lFree qStart Q K scale N (le_refl N) idx.1
      = attentionRealCausalBlock qStart Q K V scale idx := by
  rw [oFree_eq_flat, lFree_eq_flat]
  unfold attentionRealCausalBlock
  rfl

/-- The final causal m-free normalizer is positive: key `0` is always
visible to every query row. -/
theorem lFree_final_pos {M D Bk N : Nat} (hBk : 0 < Bk) (hN : 0 < N)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    0 < lFree qStart Q K scale N (le_refl N) i := by
  rw [lFree_eq_flat]
  apply Finset.sum_pos'
  · intro j _
    by_cases h : j.val ≤ qStart + i.val
    · simp [h, le_of_lt (Real.exp_pos _)]
    · simp [h]
  · refine ⟨⟨0, Nat.mul_pos hBk hN⟩, Finset.mem_univ _, ?_⟩
    simp [Real.exp_pos]

/-- The final causal streaming normalizer is nonzero under non-empty KV
scope. -/
theorem lPartial_final_ne_zero {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (hN : 0 < numKVBlocks)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (i : Fin M) :
    lPartial Bk qStart Q numKVBlocks K scale numKVBlocks i ≠ 0 := by
  rw [lPartial_eq_mShifted hBk qStart Q numKVBlocks K scale numKVBlocks
      (le_refl _) i]
  exact mul_ne_zero (Real.exp_ne_zero _)
    (ne_of_gt (lFree_final_pos hBk hN qStart Q K scale i))

/-- Final causal streaming ratio equals the user-facing local-block
causal attention spec. -/
theorem streaming_eq_attentionRealCausalBlock {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D]) :
    oPartial Bk qStart Q numKVBlocks K V scale numKVBlocks idx /
        lPartial Bk qStart Q numKVBlocks K scale numKVBlocks idx.1
      = attentionRealCausalBlock qStart Q K V scale idx := by
  have hl : lPartial Bk qStart Q numKVBlocks K scale numKVBlocks idx.1 ≠ 0 :=
    lPartial_final_ne_zero hBk qStart Q numKVBlocks hN K scale idx.1
  rw [oPartial_eq_mShifted hBk qStart Q numKVBlocks K V scale numKVBlocks
        (le_refl _) idx,
      lPartial_eq_mShifted hBk qStart Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  have hlFree : lFree qStart Q K scale numKVBlocks (le_refl _) idx.1 ≠ 0 := by
    intro h
    apply hl
    rw [lPartial_eq_mShifted hBk qStart Q numKVBlocks K scale numKVBlocks
        (le_refl _) idx.1, h, mul_zero]
  exact oFree_div_lFree_eq_attentionRealCausalBlock qStart Q K V scale idx hlFree

/-! ### Tile-level block helpers — causal kernel body

Each helper packages one statement of the causal loop body's Tile-level
algebra into a single equation, used as a rewrite in
`fa1_step_strided_causal`.

The chain mirrors the kernel:

```text
scores_raw :=  q @ k.T * scale          -- block_scoresRaw_tile_eq
causal     :=  offs_m[:, None] >= offs_n[None, :]   -- block_causal_mask_tile_eq
scores     :=  tl.where(causal, scores_raw, -inf)   -- block_scores_tile_eq
m_block    :=  tl.max(scores, axis=1)               -- block_mBlock_tile_eq
m_new      :=  tl.max(m_i, m_block)                 -- block_mNew_tile_eq
```

These are the parts where the causal kernel diverges from the non-causal
strided kernel; the `m_new`/`alpha`/`l_new`/`o_acc` lemmas (combining
the mask-aware streaming math with the running accumulators) follow. -/

/-- The kernel's `scores_raw = q @ k.T * scale` Tile expression at this
program-instance evaluates to `Tile.ofReal` of the per-element
`scaledScore`. Same structure as the non-causal kernel's `scores`
register; reused here as the un-masked input to the causal mask. -/
theorem block_scoresRaw_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N) :
    Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.dot [] (Tile.ofReal Q)
        (Tile.transpose [] (Tile.ofReal
          (fun idx : TileIndex [Bk, D] =>
            K (FA1Math.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.1,
              idx.2.1, PUnit.unit)))))
      (Tile.scalar ((scale : ℝ) : WithBot ℝ))
    = Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        FA1Math.scaledScore Q K scale idx.1
          (FA1Math.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1)) := by
  ext idx
  obtain ⟨i, j, _⟩ := idx
  simp only [Tile.bop, Tile.scalar, Tile.ofReal_data,
    Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul]
  exact FA1Math.block_scaled_data_eq Q K scale n hn i j

/-- The causal mask tile `offs_m[:, None] >= offs_n[None, :]` evaluates
to the `decide` predicate `(n * Bk + j) ≤ (qb * M + i)` at position
`(i, j)`. -/
theorem block_causal_mask_tile_eq {M Bk : Nat} (qb n : Nat) :
    (Tile.cop (ComparableDType.nat.ge)
      (Broadcast.consR (Broadcast.consL Broadcast.nil))
      (Tile.expandDim ⟨1, by simp⟩
        (Tile.vec (fun i : Fin M => qb * M + i.val)))
      (Tile.expandDim ⟨0, by simp⟩
        (Tile.vec (fun j : Fin Bk => n * Bk + j.val))))
      = ⟨fun idx : TileIndex [M, Bk] =>
          decide ((n * Bk + idx.2.1.val) ≤ (qb * M + idx.1.val))⟩ := by
  ext idx
  obtain ⟨i, j, _⟩ := idx
  simp [Tile.cop, ComparableDType.ge, Tile.expandDim, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex,
    TileShape.dropInsertedIndex]

/-- Combining the causal mask with the raw scores via `tl.where(causal,
scores_raw, -inf)` yields the `Tile` whose data is exactly
`maskedScore`. Visible lanes carry the real `scaledScore`; masked lanes
carry `⊥`. -/
theorem block_scores_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N) (qb : Nat) :
    Tile.select
      (⟨fun idx : TileIndex [M, Bk] =>
        decide ((n * Bk + idx.2.1.val) ≤ (qb * M + idx.1.val))⟩ : Tile .bool [M, Bk])
      (Tile.ofReal (fun idx : TileIndex [M, Bk] =>
        FA1Math.scaledScore Q K scale idx.1
          (FA1Math.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1)))
      (⟨fun _ : TileIndex [M, Bk] => (none : WithBot ℝ)⟩ : Tile .real [M, Bk])
    = ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore (qb * M) Q K scale idx.1
          (FA1Math.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1)⟩ := by
  ext idx
  obtain ⟨i, j, u⟩ := idx
  simp only [Tile.select_data, Tile.ofReal_data]
  have hIdx : (FA1Math.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j).val
            = n * Bk + j.val := by simp [FA1Math.blockIndex]
  by_cases h_mask : n * Bk + j.val ≤ qb * M + i.val
  · rw [decide_eq_true h_mask]
    simp only [if_true]
    rw [maskedScore_of_le (qb * M) Q K scale i _ (by rw [hIdx]; exact h_mask)]
    rfl
  · rw [decide_eq_false h_mask]
    show (none : WithBot ℝ) = maskedScore (qb * M) Q K scale i
        (FA1Math.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) j)
    rw [maskedScore_of_not_le (qb * M) Q K scale i _ (by rw [hIdx]; exact h_mask)]
    rfl

/-- Row-max of the causal scores tile: `tl.max(scores, axis=1)` produces
`Finset.univ.sup` of the per-row `maskedScore`. The reduce-max here uses
`WithBot.sup'` internally; we collapse it to `Finset.sup` since the
`WithBot` type already carries `⊥` as its bottom element. -/
theorem block_mBlock_tile_eq {M D Bk N : Nat} (hBk : 0 < Bk)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N) (qb : Nat) :
    Tile.reduceMax (shape := [M, Bk]) ⟨1, by simp⟩ Bool.false
      ⟨fun idx : TileIndex [M, Bk] =>
        maskedScore (qb * M) Q K scale idx.1
          (FA1Math.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) idx.2.1)⟩
    = some ⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun jLocal => maskedScore (qb * M) Q K scale idx.1
            (FA1Math.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) jLocal))⟩ := by
  unfold Tile.reduceMax
  simp [Tile.reduceMaxDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, hBk]
  funext idx
  rw [Finset.sup'_eq_sup]
  rfl

/-- Combining the running max `m_i` with the per-block max `m_block`
yields the next-iteration `mPartial`. This is `mPartial_succ_of_lt` at
the `Tile` level. -/
theorem block_mNew_tile_eq {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (n : Nat) (hn : n < N) (qb : Nat) :
    Tile.bop max (Broadcast.consSame Broadcast.nil)
      (⟨fun idx : TileIndex [M] =>
        mPartial Bk (qb * M) Q N K scale n idx.1⟩ : Tile .real [M])
      (⟨fun idx : TileIndex [M] =>
        (Finset.univ : Finset (Fin Bk)).sup
          (fun jLocal => maskedScore (qb * M) Q K scale idx.1
            (FA1Math.blockIndex Bk N n (Nat.succ_le_iff.mpr hn) jLocal))⟩ : Tile .real [M])
    = ⟨fun idx : TileIndex [M] =>
        mPartial Bk (qb * M) Q N K scale (n + 1) idx.1⟩ := by
  ext idx
  obtain ⟨i, _⟩ := idx
  simp [Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex]
  rw [mPartial_succ_of_lt (qb * M) Q N K scale n hn i]

/-- Causal `α` multiplier — the unbotted real payload of
`exp(mPartial(k) - mPartial(k+1))`. Same role as
`FA1Math.alphaPartial`, but for the causal streaming. -/
noncomputable def alphaCausal {M D Bk : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (numKVBlocks : Nat)
    (K : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) : ℝ :=
  (WithBot.realExp
      (Option.map₂ (fun x y : ℝ => x - y)
        (mPartial Bk qStart Q numKVBlocks K scale k i)
        (mPartial Bk qStart Q numKVBlocks K scale (k + 1) i))).unbotD 0

/-- The causal `l_new = α * l_i + tl.sum(p, axis=1)` update at the
simplified-input form realizes the causal streaming `lPartial(k+1)`.
This is `lPartial_succ_of_lt` lifted to `Tile` level. -/
theorem block_lNew_tile_eq {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame Broadcast.nil)
      (Tile.bop WithBot.realMul (Broadcast.consSame Broadcast.nil)
        (Tile.ofReal fun idx : TileIndex [M] =>
          alphaCausal qStart Q N K scale k idx.1)
        (Tile.ofReal fun idx : TileIndex [M] =>
          lPartial Bk qStart Q N K scale k idx.1))
      (Tile.ofReal fun idx : TileIndex [M] =>
        Finset.univ.sum (fun jLocal : Fin Bk =>
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart Q K scale idx.1
                (FA1Math.blockIndex Bk N k (Nat.succ_le_iff.mpr hk) jLocal))
              (mPartial Bk qStart Q N K scale (k + 1) idx.1))).unbotD 0))
      =
      Tile.ofReal (fun idx : TileIndex [M] =>
        lPartial Bk qStart Q N K scale (k + 1) idx.1) := by
  ext idx
  simp [Tile.bop, Tile.ofReal]
  rw [lPartial_succ_of_lt qStart Q N K scale k hk idx.1]
  rfl

/-- Bridge `if (decide P : Bool) then a else b ↔ if P then a else b`.
The kernel's `Tile.select` unfolds via `Tile.select_data` to a Bool-`ite`
(its `c.data idx : Bool`), while `maskedScore` and similar Prop-`if`
definitions elaborate to a Prop-`ite`. Both are propositionally equal
but not definitionally; this is the canonical simp bridge between them,
used in `fa1_step_strided_causal`'s operational simp. -/
@[simp] theorem ite_decide_bool {P : Prop} [Decidable P] {α : Sort _}
    (a b : α) :
    (if (decide P : Bool) then a else b) = if P then a else b := by
  by_cases h : P
  · simp [h]
  · simp [h]

/-- The causal `o_acc = α * o_acc + p @ V` update at the
simplified-input form realizes the causal streaming `oPartial(k+1)`.
This is `oPartial_succ_of_lt` lifted to `Tile` level. -/
theorem block_oAcc_tile_eq {M D Bk N : Nat}
    (qStart : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < N) :
    Tile.bop WithBot.realAdd (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (Tile.bop WithBot.realMul (Broadcast.consSame (Broadcast.consL Broadcast.nil))
        (Tile.expandDim ⟨1, by simp⟩
          (Tile.ofReal fun idx : TileIndex [M] =>
            alphaCausal qStart Q N K scale k idx.1))
        (Tile.ofReal fun idx : TileIndex [M, D] =>
          oPartial Bk qStart Q N K V scale k idx))
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        Finset.univ.sum (fun jLocal : Fin Bk =>
          let j := FA1Math.blockIndex Bk N k (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart Q K scale idx.1 j)
              (mPartial Bk qStart Q N K scale (k + 1) idx.1))).unbotD 0 *
            V (j, idx.2.1, PUnit.unit)))
      =
      Tile.ofReal (fun idx : TileIndex [M, D] =>
        oPartial Bk qStart Q N K V scale (k + 1) idx) := by
  ext idx
  rcases idx with ⟨i, d, u⟩
  cases u
  simp [Tile.bop, Tile.expandDim, Tile.ofReal, TileShape.dropInsertedIndex]
  rw [oPartial_succ_of_lt qStart Q N K V scale k hk (i, d, PUnit.unit)]
  rfl

/-- Per-row bridge: the `Finset.sup` of the kernel-side
`if-then-some-else-none` lambda equals the `Finset.sup` of `maskedScore`.
The mask predicate matches because `(blockIndex k j).val = k * Bk + j.val`,
and the `some` payload matches because `scaledScore = scale * Σ Q*K`
(differing from kernel's `(Σ Q*K) * scale` only by `mul_comm`). -/
theorem kernelIfSup_eq_maskedScoreSup {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (qb : Nat) (k : Nat) (hk : k < N) (i : Fin M) :
    ((Finset.univ : Finset (Fin Bk)).sup
      (fun x : Fin Bk =>
        (if k * Bk + ↑x ≤ qb * M + ↑i then
          some
            ((∑ x_1 : Fin D,
              Q (i, x_1, PUnit.unit) *
                K (FA1Math.blockIndex Bk N k (Nat.succ_le_iff.mpr hk) x, x_1, PUnit.unit)) * scale)
        else none : WithBot ℝ))) =
    ((Finset.univ : Finset (Fin Bk)).sup
      (fun x : Fin Bk =>
        maskedScore (qb * M) Q K scale i
          (FA1Math.blockIndex Bk N k (Nat.succ_le_iff.mpr hk) x))) := by
  apply Finset.sup_congr rfl
  intro x _
  by_cases h_mask : k * Bk + x.val ≤ qb * M + i.val
  · rw [maskedScore_of_le (qb * M) Q K scale i _
        (by simp [FA1Math.blockIndex]; exact h_mask)]
    rw [if_pos h_mask]
    unfold FA1Math.scaledScore
    ring_nf
    rfl
  · rw [maskedScore_of_not_le (qb * M) Q K scale i _
        (by simp [FA1Math.blockIndex]; omega)]
    simp [h_mask]
    rfl

/-- Sup-form `mPartial(k+1)` recurrence over the kernel-side
`if-then-some-else-none` lambda. The kernel's `Tile.reduceMaxDrop`
produces `Finset.sup'`; after normalizing to `Finset.sup` via
`Finset.sup'_eq_sup`, this lemma directly closes the m_new conjunct. -/
theorem mPartial_succ_kernelForm {M D Bk N : Nat}
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (qb : Nat) (k : Nat) (hk : k < N) (i : Fin M) :
    max (mPartial Bk (qb * M) Q N K scale k i)
      ((Finset.univ : Finset (Fin Bk)).sup
        (fun x : Fin Bk =>
          (if k * Bk + ↑x ≤ qb * M + ↑i then
            some
              ((∑ x_1 : Fin D,
                Q (i, x_1, PUnit.unit) *
                  K (FA1Math.blockIndex Bk N k (Nat.succ_le_iff.mpr hk) x, x_1, PUnit.unit)) * scale)
          else none : WithBot ℝ))) =
    mPartial Bk (qb * M) Q N K scale (k + 1) i := by
  rw [kernelIfSup_eq_maskedScoreSup Q K scale qb k hk i,
      ← mPartial_succ_of_lt (qb * M) Q N K scale k hk i]

/-- `WithBot.realExp` is never bottom, so it is equal to its `unbotD`
payload rewrapped as `some`. This bridges the kernel's optional value
with the ℝ-clean streaming definitions. -/
theorem realExp_eq_some_unbotD (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

end FA1MathCausal
def P_fa1
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (origPid : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  -- `pid` register and value preserved.
  s.regs .nat [] "pid" = some (Tile.scalar origPid) ∧
  s.pid = origPid ∧
  -- Q-row-block index vector (set pre-loop, reused post-loop for the
  -- final `tl.store` ptrs).
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => origPid * M + i.val)) ∧
  -- D-axis index vector.
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  -- Loaded Q-row-block.
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  -- Streaming accumulators at iteration `k`.
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] => FA1Math.mPartial Bk Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1Math.lPartial Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        FA1Math.oPartial Q numKVBlocks K V scale k idx) ∧
  -- Input regions still loaded as the user's hypothesis says.
  InputAt s qReg
      (Offset.rowMajor2D (rows := M) (cols := D) (origPid * M * D) D) Q ∧
  InputAt s kReg
      (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) K ∧
  InputAt s vReg
      (Offset.rowMajor2D (rows := Bk * numKVBlocks) (cols := D) 0 D) V

/-- Loop-carried predicate for the strided FA-1 forward kernel
(`fa1ForwardKernelStrided`). Same streaming math content as `P_fa1` —
running `mPartial`/`lPartial`/`oPartial` in their register slots, plus
the initial Q-block load and book-keeping — but the InputAt premises
use the strided / 4D-aware addressing computed by the kernel from its
16 stride parameters and three `program_id` axes.

The static `(qb, headIdx, batch)` triple is the program_id values at
proof entry; the invariant asserts they are preserved across loop
iterations and exposes them as `Tile.scalar` registers.

Tile-local offset perspective: Q is observed at addresses
`qBase + i * sQS + d * sQD` where
`qBase = batch * sQB + headIdx * sQH + qb * M * sQS`. Similarly for
K/V/O. The 4D-wrapper corollary (issue #39 step (iv)) supplies these
offsets via `Offset.strided` over `[B, H, S_q, D]` plus
`Offset.strided_inj` for the readback injectivity. -/
def P_fa1_strided
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    -- Static (qb, headIdx, batch) program-id context.
    (qb headIdx batch : Nat)
    -- 16 stride parameters (matching `fa1ForwardKernelStrided`):
    (sQB sQH sQS sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    -- Math inputs:
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  -- Program-id values preserved.
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  -- pid_* registers (set by preLoop's three program_id reads).
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  -- *_base_off registers (set by preLoop's four base-offset assigns).
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  -- Q-block row index vector and D-axis index vector.
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  -- Loaded Q-row-block.
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  -- Streaming accumulators at iter k.
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] => FA1Math.mPartial Bk Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1Math.lPartial Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        FA1Math.oPartial Q numKVBlocks K V scale k idx) ∧
  -- Input regions still loaded under the strided addressing.
  InputAt s qReg
      (fun idx : TileIndex [M, D] =>
        batch * sQB + headIdx * sQH + qb * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD) Q ∧
  InputAt s kReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V

/-- Loop-carried predicate for the causal strided FA-1 kernel. This is
the causal analogue of `P_fa1_strided`: the address bookkeeping and
loaded Q/K/V premises are unchanged, but the accumulator registers are
interpreted with the causal streaming recurrence parameterized by
`qStart = qb * M`. -/
def P_fa1_strided_causal
    {M D Bk numKVBlocks : Nat}
    (qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH sQS sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        FA1MathCausal.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1MathCausal.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        FA1MathCausal.oPartial Bk (qb * M) Q numKVBlocks K V scale k idx) ∧
  InputAt s qReg
      (fun idx : TileIndex [M, D] =>
        batch * sQB + headIdx * sQH + qb * M * sQS
          + idx.1.val * sQS + idx.2.1.val * sQD) Q ∧
  InputAt s kReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [Bk * numKVBlocks, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V

/-- Loop-carried predicate for FA-1 v1 / boundary-masked strided kernels.

Compared with `P_fa1_strided`, K/V are logical `[S_k, D]` tensors rather
than padded `[Bk * numKVBlocks, D]` tensors, and the running accumulators use
`FA1MathBoundary`. The kernel still iterates over `numKVBlocks` padded
blocks; out-of-range lanes are masked into `⊥` / zero by the recurrence. -/
def P_fa1_strided_boundary
    {M D Bk numKVBlocks S_k : Nat}
    (_qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH _sQS _sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        FA1MathBoundary.mPartial Bk Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1MathBoundary.lPartial Bk Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        FA1MathBoundary.oPartial Bk Q numKVBlocks K V scale k idx) ∧
  InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V

/-- Loop-carried predicate for causal boundary-masked strided kernels.

This combines `P_fa1_strided_boundary`'s logical `[S_k, D]` K/V memory
contract with the causal streaming recurrence parameterized by
`qStart = qb * M`. -/
def P_fa1_strided_causal_boundary
    {M D Bk numKVBlocks S_k : Nat}
    (_qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH _sQS _sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [D] "offs_d" = some
      (Tile.vec (fun d : Fin D => d.val)) ∧
  s.regs .real [M, D] "q" = some (Tile.ofReal Q) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        FA1MathCausalBoundary.mPartial Bk (qb * M) Q numKVBlocks K scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1MathCausalBoundary.lPartial Bk (qb * M) Q numKVBlocks K scale k idx.1) ∧
  s.regs .real [M, D] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, D] =>
        FA1MathCausalBoundary.oPartial Bk (qb * M) Q numKVBlocks K V scale k idx) ∧
  InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V

/-- D-tail boundary invariant: run the existing boundary invariant at block
hidden width `Bd`, with Q/K/V register/math state zero-padded outside `D`.
Unlike the plain padded lift, the K/V memory contract remains logical
`[S_k, D]`; out-of-D lanes are supplied by masked loads, not by memory. -/
def P_fa1_strided_boundaryD
    {M D Bd Bk numKVBlocks S_k : Nat}
    (_qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH _sQS _sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [Bd] "offs_d" = some
      (Tile.vec (fun d : Fin Bd => d.val)) ∧
  s.regs .real [M, Bd] "q" = some (Tile.ofReal (padHeadD (Bd := Bd) Q)) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        FA1MathBoundary.mPartial Bk (padHeadD (Bd := Bd) Q) numKVBlocks
          (padHeadD (Bd := Bd) K) scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1MathBoundary.lPartial Bk (padHeadD (Bd := Bd) Q) numKVBlocks
          (padHeadD (Bd := Bd) K) scale k idx.1) ∧
  s.regs .real [M, Bd] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, Bd] =>
        FA1MathBoundary.oPartial Bk (padHeadD (Bd := Bd) Q) numKVBlocks
          (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V) scale k idx) ∧
  InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V

/-- D-tail causal-boundary invariant: same padded hidden-width lift as
`P_fa1_strided_boundaryD`, but with the causal-boundary recurrence. -/
def P_fa1_strided_causal_boundaryD
    {M D Bd Bk numKVBlocks S_k : Nat}
    (_qReg kReg vReg : RegionName)
    (qb headIdx batch : Nat)
    (sQB sQH _sQS _sQD : Nat)
    (sKB sKH sKN sKD : Nat)
    (sVB sVH sVN sVD : Nat)
    (sOB sOH _sOM _sOD : Nat)
    (Q : TileIndex [M, D] → ℝ)
    (K V : TileIndex [S_k, D] → ℝ)
    (scale : ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.pids 0 = qb ∧ s.pids 1 = headIdx ∧ s.pids 2 = batch ∧
  s.regs .nat [] "pid_qb" = some (Tile.scalar qb) ∧
  s.regs .nat [] "pid_h"  = some (Tile.scalar headIdx) ∧
  s.regs .nat [] "pid_b"  = some (Tile.scalar batch) ∧
  s.regs .nat [] "q_base_off" = some (Tile.scalar (batch * sQB + headIdx * sQH)) ∧
  s.regs .nat [] "k_base_off" = some (Tile.scalar (batch * sKB + headIdx * sKH)) ∧
  s.regs .nat [] "v_base_off" = some (Tile.scalar (batch * sVB + headIdx * sVH)) ∧
  s.regs .nat [] "o_base_off" = some (Tile.scalar (batch * sOB + headIdx * sOH)) ∧
  s.regs .nat [M] "offs_m" = some
      (Tile.vec (fun i : Fin M => qb * M + i.val)) ∧
  s.regs .nat [Bd] "offs_d" = some
      (Tile.vec (fun d : Fin Bd => d.val)) ∧
  s.regs .real [M, Bd] "q" = some (Tile.ofReal (padHeadD (Bd := Bd) Q)) ∧
  s.regs .real [M] "m_i" = some
      ⟨fun idx : TileIndex [M] =>
        FA1MathCausalBoundary.mPartial Bk (qb * M) (padHeadD (Bd := Bd) Q)
          numKVBlocks (padHeadD (Bd := Bd) K) scale k idx.1⟩ ∧
  s.regs .real [M] "l_i" = some
      (Tile.ofReal fun idx : TileIndex [M] =>
        FA1MathCausalBoundary.lPartial Bk (qb * M) (padHeadD (Bd := Bd) Q)
          numKVBlocks (padHeadD (Bd := Bd) K) scale k idx.1) ∧
  s.regs .real [M, Bd] "o_acc" = some
      (Tile.ofReal fun idx : TileIndex [M, Bd] =>
        FA1MathCausalBoundary.oPartial Bk (qb * M) (padHeadD (Bd := Bd) Q)
          numKVBlocks (padHeadD (Bd := Bd) K) (padHeadD (Bd := Bd) V)
          scale k idx) ∧
  InputAt s kReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sKB + headIdx * sKH
          + idx.1.val * sKN + idx.2.1.val * sKD) K ∧
  InputAt s vReg
      (fun idx : TileIndex [S_k, D] =>
        batch * sVB + headIdx * sVH
          + idx.1.val * sVN + idx.2.1.val * sVD) V
/-! ## attentionReal4D bridge

The slice-form result of `fa1_forward_correct_4D_slice` equals the
user-facing `attentionReal4D` spec at the corresponding 4D index.
The bridge factors through `attentionReal_row_eq` (`attentionReal`'s
output at row `i` depends only on `Q`'s row at `i`), reusing the
fact that K and V slices are byte-identical (modulo Fin proof
irrelevance) under `Bk * numKVBlocks = S_k`.

Three private helpers do the row-factoring through the `Tile.dot` /
`softmaxRow` / `attention` layers; `attentionReal_row_eq` ties them
together at the user-facing `Tile.ofReal`-lifted level. -/

/-- `Tile.dot`'s output at row `i` of the first argument depends only
on that row's data. -/
private theorem dot_data_row_eq {M1 M2 K N : Nat}
    (a1 : Tile .real [M1, K]) (a2 : Tile .real [M2, K])
    (b : Tile .real [K, N])
    (i1 : Fin M1) (i2 : Fin M2) (n : Fin N)
    (hRow : ∀ k, a1.data (i1, k, PUnit.unit) = a2.data (i2, k, PUnit.unit)) :
    (Tile.dot [] a1 b).data (i1, n, PUnit.unit) =
    (Tile.dot [] a2 b).data (i2, n, PUnit.unit) := by
  rw [Tile.dot_nil_data, Tile.dot_nil_data]
  apply Finset.sum_congr rfl
  intro k _
  rw [hRow k]

/-- `softmaxRow`'s output at row `i` depends only on that row's data. -/
private theorem softmaxRow_data_row_eq {M1 M2 N : Nat}
    (s1 : Tile .real [M1, N]) (s2 : Tile .real [M2, N])
    (i1 : Fin M1) (i2 : Fin M2) (n : Fin N)
    (hRow : ∀ n', s1.data (i1, n', PUnit.unit) = s2.data (i2, n', PUnit.unit)) :
    (softmaxRow s1).data (i1, n, PUnit.unit) =
    (softmaxRow s2).data (i2, n, PUnit.unit) := by
  unfold softmaxRow
  simp only []
  show some _ = some _
  congr 2
  · rw [hRow n]
  · apply Finset.sum_congr rfl
    intro j _
    rw [hRow j]

/-- `attention`'s output at row `i` of `Q` depends only on that
row's data (and on `K`, `V`, `scale`). -/
private theorem attention_data_row_eq {M1 M2 S D : Nat}
    (Q1 : Tile .real [M1, D]) (Q2 : Tile .real [M2, D])
    (K V : Tile .real [S, D]) (scale : ℝ)
    (i1 : Fin M1) (i2 : Fin M2) (d : Fin D)
    (hRow : ∀ d' : Fin D, Q1.data (i1, d', PUnit.unit) = Q2.data (i2, d', PUnit.unit)) :
    (attention Q1 K V scale).data (i1, d, PUnit.unit) =
    (attention Q2 K V scale).data (i2, d, PUnit.unit) := by
  unfold attention
  simp only []
  apply dot_data_row_eq
  intro k
  apply softmaxRow_data_row_eq
  intro n'
  show Option.map (· * scale) ((Tile.dot [] Q1 (Tile.transpose [] K)).data (i1, n', PUnit.unit)) =
       Option.map (· * scale) ((Tile.dot [] Q2 (Tile.transpose [] K)).data (i2, n', PUnit.unit))
  congr 1
  exact dot_data_row_eq Q1 Q2 (Tile.transpose [] K) i1 i2 n' hRow

/-- Row-invariance of `attentionReal`. Two `attentionReal` calls (with
possibly different M dimensions on `Q`) produce the same value at
indices `(i1, d)` and `(i2, d)` whenever `Q1`'s row at `i1` equals
`Q2`'s row at `i2`. The K, V, scale arguments are the same. -/
theorem attentionReal_row_eq {M1 M2 S D : Nat}
    (Q1 : TileIndex [M1, D] → ℝ) (Q2 : TileIndex [M2, D] → ℝ)
    (K V : TileIndex [S, D] → ℝ) (scale : ℝ)
    (i1 : Fin M1) (i2 : Fin M2) (d : Fin D)
    (hRow : ∀ d' : Fin D, Q1 (i1, d', PUnit.unit) = Q2 (i2, d', PUnit.unit)) :
    attentionReal Q1 K V scale (i1, d, PUnit.unit)
      = attentionReal Q2 K V scale (i2, d, PUnit.unit) := by
  unfold attentionReal
  congr 1
  apply attention_data_row_eq
  intro d'
  simp only [Tile.ofReal_data, hRow d']

/-- Bridge: the slice-form `attentionReal` produced by FA-1's strided
kernel equals the `attentionReal4D` spec at the corresponding 4D
index. The proof factors through `attentionReal_row_eq`: K, V slices
become byte-identical to `sliceBH` after substituting `hSk`, so only
the Q-row equality remains, and that is `rfl` by `slice4DQRows`'s
definition. -/
theorem attentionReal_slice_eq_attentionReal4D
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hSk : Bk * numKVBlocks = S_k)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ)
    (b : Fin B) (h : Fin H) (qb : Nat)
    (hQBnd : qb * M + M ≤ S_q)
    (idx : TileIndex [M, D]) :
    attentionReal
        (slice4DQRows M Q4D b h qb hQBnd)
        (slice4DFlat Bk numKVBlocks K4D b h hSk)
        (slice4DFlat Bk numKVBlocks V4D b h hSk)
        scale idx
      = attentionReal4D Q4D K4D V4D scale
          (b, h, ⟨qb * M + idx.1.val, by have := idx.1.isLt; omega⟩,
           idx.2.1, PUnit.unit) := by
  obtain ⟨i, d, _⟩ := idx
  rw [attentionReal4D_slice]
  subst hSk
  have hKeq : slice4DFlat Bk numKVBlocks K4D b h rfl = sliceBH K4D b h := by
    funext idx; obtain ⟨j, d', _⟩ := idx; rfl
  have hVeq : slice4DFlat Bk numKVBlocks V4D b h rfl = sliceBH V4D b h := by
    funext idx; obtain ⟨j, d', _⟩ := idx; rfl
  rw [hKeq, hVeq]
  apply attentionReal_row_eq
  intro d'
  rfl

/-- Causal bridge: the block-local causal spec with global query start
`qb * M` is exactly the user-facing 4D causal spec at the corresponding
global row. This is the causal analogue of
`attentionReal_slice_eq_attentionReal4D`, but it does not need the
row-invariance lemmas because `attentionRealCausalBlock` already carries
the global query offset in its mask. -/
theorem attentionRealCausalBlock_slice_eq_attentionReal4DCausal
    {B H S_q S_k D Bk numKVBlocks M : Nat}
    (hSk : Bk * numKVBlocks = S_k)
    (Q4D : TileIndex [B, H, S_q, D] → ℝ)
    (K4D V4D : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ)
    (b : Fin B) (h : Fin H) (qb : Nat)
    (hQBnd : qb * M + M ≤ S_q)
    (idx : TileIndex [M, D]) :
    attentionRealCausalBlock (qb * M)
        (slice4DQRows M Q4D b h qb hQBnd)
        (slice4DFlat Bk numKVBlocks K4D b h hSk)
        (slice4DFlat Bk numKVBlocks V4D b h hSk)
        scale idx
      = attentionReal4DCausal Q4D K4D V4D scale
          (b, h, ⟨qb * M + idx.1.val, by have := idx.1.isLt; omega⟩,
           idx.2.1, PUnit.unit) := by
  obtain ⟨i, d, _⟩ := idx
  rw [attentionReal4DCausal_slice]
  subst hSk
  unfold attentionRealCausalBlock attentionRealCausal
  simp [slice4DQRows, slice4DFlat, sliceBH]

end VeriTile.Examples
