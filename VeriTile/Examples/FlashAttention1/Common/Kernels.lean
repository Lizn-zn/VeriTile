/-
VeriTile.Examples.FlashAttention1.Common.Kernels

FA-1 forward kernel DSL surfaces shared by the proof modules.
-/

import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Frontend.Triton.DSL
import VeriTile.Kernel
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile

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
    (M D Bk numKVBlocks : Nat) (scale : ℝ) : ComputeKernel := triton {
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
    scores  := tl.dot(q, tl.trans(k)) * $(scale)

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
    (scale : ℝ) : ComputeKernel := triton {
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

    scores  := tl.dot(q, tl.trans(k)) * $(scale)
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
    (scale : ℝ) : ComputeKernel := triton {
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

    scores_raw := tl.dot(q, tl.trans(k)) * $(scale)
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
    (scale : ℝ) : ComputeKernel := triton {
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

    scores_raw := tl.dot(q, tl.trans(k)) * $(scale)
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
    (scale : ℝ) : ComputeKernel := triton {
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

    scores_raw := tl.dot(q, tl.trans(k)) * $(scale)
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
    (scale : ℝ) : ComputeKernel := triton {
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

    scores_raw := tl.dot(q, tl.trans(k)) * $(scale)
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
    (scale : ℝ) : ComputeKernel := triton {
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

    scores_raw := tl.dot(q, tl.trans(k)) * $(scale)
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
    (scale : ℝ) : ComputeKernel := triton {
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

  scores := tl.dot(q, tl.trans(k)) * $(scale)
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
    (scale : ℝ) : ComputeKernel := triton {
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

  scores_raw := tl.dot(q, tl.trans(k)) * $(scale)
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

end VeriTile.Examples
