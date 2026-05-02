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
  q      := tl.load(tl.ptr($(qReg)) + q_ptrs)

  -- Initialize accumulators at the right tile shape (so the loop
  -- registers stay rank-correct across iterations).
  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    v_ptrs  := offs_n[:, None] * $(D) + offs_d[None, :]
    k       := tl.load(tl.ptr($(kReg)) + k_ptrs)
    v       := tl.load(tl.ptr($(vReg)) + v_ptrs)

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
  tl.store(tl.ptr($(outReg)) + o_ptrs, out)
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
  q      := tl.load(tl.ptr($(qReg)) + q_ptrs)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
    v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
    k       := tl.load(tl.ptr($(kReg)) + k_ptrs)
    v       := tl.load(tl.ptr($(vReg)) + v_ptrs)

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
  tl.store(tl.ptr($(outReg)) + o_ptrs, out)
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
  q      := tl.load(tl.ptr($(qReg)) + q_ptrs)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
    v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
    k       := tl.load(tl.ptr($(kReg)) + k_ptrs)
    v       := tl.load(tl.ptr($(vReg)) + v_ptrs)

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
  tl.store(tl.ptr($(outReg)) + o_ptrs, out)
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
  q      := tl.load(tl.ptr($(qReg)) + q_ptrs, mask=q_mask, other=0)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
    v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
    kv_mask := (offs_n[:, None] + offs_d[None, :] * $(0)) < $(S_k)
    k       := tl.load(tl.ptr($(kReg)) + k_ptrs, mask=kv_mask, other=0)
    v       := tl.load(tl.ptr($(vReg)) + v_ptrs, mask=kv_mask, other=0)

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
  tl.store(tl.ptr($(outReg)) + o_ptrs, out, mask=o_mask)
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
  q      := tl.load(tl.ptr($(qReg)) + q_ptrs, mask=q_mask, other=0)

  m_i    := tl.full([$(M)], -inf)
  l_i    := tl.zeros([$(M)])
  o_acc  := tl.zeros([$(M), $(D)])

  tl.for n in $(numKVBlocks) {
    offs_n  := n * $(Bk) + tl.arange(0, $(Bk))
    k_ptrs  := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
    v_ptrs  := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
    kv_mask := (offs_n[:, None] + offs_d[None, :] * $(0)) < $(S_k)
    k       := tl.load(tl.ptr($(kReg)) + k_ptrs, mask=kv_mask, other=0)
    v       := tl.load(tl.ptr($(vReg)) + v_ptrs, mask=kv_mask, other=0)

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
  tl.store(tl.ptr($(outReg)) + o_ptrs, out, mask=o_mask)
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
  q          := tl.load(tl.ptr($(qReg)) + q_ptrs, mask=q_mask, other=0)

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
    k       := tl.load(tl.ptr($(kReg)) + k_ptrs, mask=kv_mask, other=0)
    v       := tl.load(tl.ptr($(vReg)) + v_ptrs, mask=kv_mask, other=0)

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
  tl.store(tl.ptr($(outReg)) + o_ptrs, out, mask=o_mask)
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
  q          := tl.load(tl.ptr($(qReg)) + q_ptrs, mask=q_mask, other=0)

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
    k       := tl.load(tl.ptr($(kReg)) + k_ptrs, mask=kv_mask, other=0)
    v       := tl.load(tl.ptr($(vReg)) + v_ptrs, mask=kv_mask, other=0)

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
  tl.store(tl.ptr($(outReg)) + o_ptrs, out, mask=o_mask)
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
  q          := tl.load(tl.ptr($(qReg)) + q_ptrs, mask=q_mask, other=0)

  k_ptrs := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
  v_ptrs := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
  kv_d_mask := (offs_n[:, None] * $(0) + offs_d[None, :]) < $(D)
  k      := tl.load(tl.ptr($(kReg)) + k_ptrs, mask=kv_d_mask, other=0)
  v      := tl.load(tl.ptr($(vReg)) + v_ptrs, mask=kv_d_mask, other=0)

  scores := tl.dot(q, tl.trans(k)) * $ℝ(scale)
  p      := tl.exp(scores)
  l      := tl.sum(p, axis = 1)
  o_acc  := tl.dot(p, v)
  out    := o_acc / l[:, None]

  o_ptrs := o_base_off + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_od)
  o_seq_mask := (offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
  o_d_mask   := (offs_m[:, None] * $(0) + offs_d[None, :]) < $(D)
  o_mask     := tl.logical_and(o_seq_mask, o_d_mask)
  tl.store(tl.ptr($(outReg)) + o_ptrs, out, mask=o_mask)
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
  q          := tl.load(tl.ptr($(qReg)) + q_ptrs, mask=q_mask, other=0)

  k_ptrs := k_base_off + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kd)
  v_ptrs := v_base_off + offs_n[:, None] * $(stride_vn) + offs_d[None, :] * $(stride_vd)
  kv_d_mask := (offs_n[:, None] * $(0) + offs_d[None, :]) < $(D)
  k      := tl.load(tl.ptr($(kReg)) + k_ptrs, mask=kv_d_mask, other=0)
  v      := tl.load(tl.ptr($(vReg)) + v_ptrs, mask=kv_d_mask, other=0)

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
  tl.store(tl.ptr($(outReg)) + o_ptrs, out, mask=o_mask)
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

/-! ### Stage A foundations — `mPartial` non-`⊥` for `k ≥ 1`

The streaming algebra (α-cancellation in `lPartial` / `oPartial`)
needs `mPartial k` to be a real number, not `⊥`, whenever any block
has been seen. With `0 < Bk`, the block-max `Finset.univ.sup` over
`Fin Bk` is non-`⊥`, so `mPartial 1` is non-`⊥` and the property
propagates upward through `max`. -/

/-- Fin (Bk * N) ≃ Fin N × Fin Bk, the bijection underlying
`oFree`/`lFree`'s double-sum form. Composes `finProdFinEquiv.symm` with
a `mul_comm` cast. -/
def blockIndexEquiv (Bk N : Nat) : Fin (Bk * N) ≃ Fin N × Fin Bk :=
  (Fin.castOrderIso (Nat.mul_comm Bk N)).toEquiv.trans finProdFinEquiv.symm

/-! ### Shape-polymorphic variants

These mirror `qkT_data_eq` / `scaled_data_eq` / `softmaxRow_scaled_data_eq`
but take a generic `S` (not necessarily of the form `Bk * N`). Used by
the boundary streaming proofs where K/V live in `[S_k, D]` with no
factorization commitment. The proofs are textually identical — Lean's
definitional unfolding is the same; only the implicit shape changes. -/

end FA1Math

/-! ## Boundary-masked streaming math model (FA-1 v1 scaffold)

The full-tile `FA1Math` recurrence indexes K/V by
`Fin (Bk * numKVBlocks)`. Boundary-masked kernels instead iterate over a
padded block domain while the mathematical input has logical length `S_k`.
Invalid local lanes (`k * Bk + jLocal >= S_k`) enter as `⊥`, exactly like
the kernel's score-side `tl.where(score_mask, scores_raw, -inf)`.
-/

namespace FA1MathBoundary

/-- Logical KV index for a padded loop lane, if it is in bounds. -/
def blockIndex? (S_k Bk k : Nat) (jLocal : Fin Bk) : Option (Fin S_k) :=
  if h : k * Bk + jLocal.val < S_k then
    some ⟨k * Bk + jLocal.val, h⟩
  else
    none

/-- Boundary-masked score for a padded loop lane. Out-of-range KV lanes are
`⊥`, so exponentiating them contributes zero mass. -/
noncomputable def maskedScore {M S_k D : Nat}
    (Bk k : Nat)
    (Q : TileIndex [M, D] → ℝ) (K : TileIndex [S_k, D] → ℝ) (scale : ℝ)
    (i : Fin M) (jLocal : Fin Bk) : WithBot ℝ :=
  match blockIndex? S_k Bk k jLocal with
  | some j => (FA1Math.scaledScore Q K scale i j : ℝ)
  | none   => ⊥

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

end FA1MathBoundary

/-! ## Causal boundary streaming math model

Combines the KV boundary mask with the causal `j <= qStart + i` mask.
Out-of-range or future lanes are represented as `⊥`, matching the
causal-boundary kernel's two `tl.where(..., -inf)` stages. -/

namespace FA1MathCausalBoundary

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

end VeriTile.Examples
