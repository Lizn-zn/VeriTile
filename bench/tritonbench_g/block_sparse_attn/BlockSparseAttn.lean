import VeriTile.Triton

/-!
# `block_sparse_attn` — strict per-kernel correctness

`block_sparse_attn.py`'s `block_sparse_attention_kernel` is a block-sparse
(CSR-mask) FlashAttention forward for prompt-only / right-padded inputs:
program `(start_m, off_bh)` loads its query tile, walks only the active key
blocks named by the per-head CSR `row_indices`/`col_indices`, runs the
online-softmax recurrence (`m_i`, `l_i`, accumulators `acc`/`acc2`) with causal
masking and `softmax_scale`, and finally stores `acc` (and, when
`NUM_D_BLOCKS ≥ 2`, `acc2` at `+BLOCK_D`) to `out`, masked by `offs_m <
q_seq_len`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`grid = (cdiv(q_seq_len, BLOCK_M), B·num_heads)`, the
GQA head mapping, the CSR sparsity schedule, and how the runtime composes
per-program writes into `out`) is the *trusted boundary*, not a proof
obligation here. Because the program ids `start_m`/`off_bh` are universally
quantified (via `s`), the per-program statements cover every program of the
grid.

## Proof architecture

```
block_sparse_attn_output_closed_form_summary_general                     ← TOP (dimension-general; EVEN_M = EVEN_N = true)
  ├─ bsa_exec                                       full surface exec → streaming accumulators
  │    ├─ bsaPreLoop_eval                           prologue → bsaInvariant 0 (+ start_l/end_l regs)
  │    ├─ bsa_csr_loop (per-block bsaLoopBody step)  forRangeDyn driver: bsaInvariant c → c+1
  │    └─ bsaPostLoop_eval                          two masked out stores = bsaOPartial / bsaLPartial
  └─ bsa_streaming_eq_closedForm                    streaming = blockSparseAttnClosedForm
       ├─ bsaStreaming_eq_bsaAttn                   online-softmax fold = gathered causal softmax
       ├─ bsaAttn_reindex                           16·numKVBlocks = numKVBlocks·16 reshape
       └─ bsaAttn_eq_blockSparseAttnClosedForm      gathered = genuine block-sparse closed form
```
(`bsa_body_split` splits the lowered body; offset injectivity is inlined in
`bsaPostLoop_eval`.)

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; `exp`, `tl.dot`, and the
`softmax_scale` multiply are not modeled at the bit level). The verified result
is the **genuine closed form**: the full
faithful kernel surface (prologue + CSR `forRangeDyn` online-softmax loop + the
two masked `out` stores) is unfolded statement-by-statement and proven to write
`blockSparseAttnClosedForm` (NOT a self-referential executed value) at every
active output lane. The CSR sparsity schedule itself — which key blocks the loop
visits (`start_l`/`end_l`, the per-block selection/load alignment `hstep`, and
first-key causal visibility `hVis0`) — is the trusted host boundary, supplied as
hypotheses of the summary. The general summary holds at arbitrary tile dims and
strides with `NUM_D_BLOCKS = 2` and `EVEN_M = EVEN_N = true`;
the `EVEN_M = EVEN_N = false` masked-load Python variant reuses a
different lowered body and is out of scope here.

## Genuine closed-form spec

`blockSparseAttnClosedForm` is a **genuine, non-self-referential** closed form
for one program's output: the causal natural-exp softmax attention
(`tl.exp`, not `exp2`) of the program's query tile against the key/value rows the
CSR layout selects (`selKeyGlobal`), under grouped-query head mapping
(`offHkv`). The top summary `block_sparse_attn_output_closed_form_summary_general`
proves **sorry-free** that the two masked `out` stores write this closed form to
`Out` at every active lane, via `bsa_execG` (the full execution chain) composed
with `bsa_streaming_eq_closedFormG` (the streaming online-softmax recurrence over
the CSR-selected, causally-masked key blocks collapses to the gathered causal
softmax). The self-referential `produced…Value` carriers are retired.
-/

namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn

open VeriTile.Triton
open scoped VeriTile.Triton.Masked3DTileKernelIO₁

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `block_sparse_attn.py`'s
`block_sparse_attention_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `tl.constexpr` parameters become Lean parameters with `$(...)` at use
  sites.
- The Python `tl.static_print(f"...")` f-string payload is represented by a
  fixed debug string; `tl.static_print` is a compile-time/no-op DSL marker.
- `if NUM_D_BLOCKS >= 2:` is represented as the equivalent Bool antiquote
  `if $((NUM_D_BLOCKS >= 2 : Bool))`. -/
def block_sparse_attention_kernel
    (out Q K V : RegionName)
    (layout_csr_row_indices layout_csr_col_indices : Region .nat)
    (layout_csr_row_stride_h layout_csr_col_stride_h num_layout : Nat)
    (softmax_scale : ℝ)
    (stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
      stride_vb stride_vh stride_vn stride_ob stride_oh stride_om
      num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D
      NUM_D_BLOCKS : Nat)
    (EVEN_M EVEN_N : Bool) :
    ComputeKernel := triton {
  tl.static_print("block_sparse_attention_kernel")
  q_seq_len = $(total_seq_len)
  start_m = tl.program_id(0)
  off_bh = tl.program_id(1)
  off_h = off_bh % $(num_heads)
  off_b = off_bh // $(num_heads)
  head_groups = $(num_heads) // $(num_kv_heads)
  off_h_kv = off_h // head_groups
  Q += off_b * $(stride_qb) + off_h * $(stride_qh)
  K += off_b * $(stride_kb) + off_h_kv * $(stride_kh)
  V += off_b * $(stride_vb) + off_h_kv * $(stride_vh)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_D))
  off_q = offs_m[:, None] * $(stride_qm) + offs_d[None, :]
  off_k = offs_n[None, :] * $(stride_kn) + offs_d[:, None]
  off_v = offs_n[:, None] * $(stride_vn) + offs_d[None, :]
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_D)], dtype=tl.float32)
  if $((NUM_D_BLOCKS >= 2 : Bool)) {
    acc2 = tl.zeros([$(BLOCK_M), $(BLOCK_D)], dtype=tl.float32)
  }
  if EVEN_M {
    q = tl.load(q_ptrs)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      q2 = tl.load(q_ptrs + $(BLOCK_D))
    }
  } else {
    q = tl.load(q_ptrs, mask=offs_m[:, None] < q_seq_len)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      q2 = tl.load(q_ptrs + $(BLOCK_D), mask=offs_m[:, None] < q_seq_len)
    }
  }
  layout_h = off_h % $(num_layout)
  layout_ptr = layout_csr_row_indices + layout_h * $(layout_csr_row_stride_h) + start_m
  start_l = (tl.load(layout_ptr)).to(tl.int32)
  end_l = (tl.load(layout_ptr + $(1))).to(tl.int32)
  for col_idx_idx in range(start_l, end_l) {
    col_idx = (tl.load(layout_csr_col_indices +
      layout_h * $(layout_csr_col_stride_h) + col_idx_idx)).to(tl.int32)
    start_n = col_idx * $(BLOCK_N)
    if EVEN_N {
      k = tl.load(k_ptrs + start_n * $(stride_kn))
    } else {
      k = tl.load(k_ptrs + start_n * $(stride_kn),
        mask=offs_n[None, :] + start_n < $(total_seq_len))
    }
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      if EVEN_N {
        k = tl.load(k_ptrs + start_n * $(stride_kn) + $(BLOCK_D))
      } else {
        k = tl.load(k_ptrs + start_n * $(stride_kn) + $(BLOCK_D),
          mask=offs_n[None, :] + start_n < $(total_seq_len))
      }
      qk += tl.dot(q2, k)
    }
    qk *= $(softmax_scale)
    qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      acc2 = acc2 * acc_scale[:, None]
    }
    p = (p).to(Q.dtype.element_ty)
    if EVEN_N {
      v = tl.load(v_ptrs + start_n * $(stride_vn))
    } else {
      v = tl.load(v_ptrs + start_n * $(stride_vn),
        mask=offs_n[:, None] + start_n < $(total_seq_len))
    }
    acc += tl.dot(p, v)
    if $((NUM_D_BLOCKS >= 2 : Bool)) {
      if EVEN_N {
        v = tl.load(v_ptrs + start_n * $(stride_vn) + $(BLOCK_D))
      } else {
        v = tl.load(v_ptrs + start_n * $(stride_vn) + $(BLOCK_D),
          mask=offs_n[:, None] + start_n < $(total_seq_len))
      }
      acc2 += tl.dot(p, v)
    }
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = off_b * $(stride_ob) + off_h * $(stride_oh) +
    offs_m[:, None] * $(stride_om) + offs_d[None, :]
  out_ptrs = out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
  if $((NUM_D_BLOCKS >= 2 : Bool)) {
    tl.store(out_ptrs + $(BLOCK_D), acc2, mask=offs_m[:, None] < q_seq_len)
  }
}

/-- Surface transcription/proof-oriented first output-block store slice of
`block_sparse_attn.py`'s `block_sparse_attention_kernel`.

The full kernel walks a CSR sparse layout and accumulates one or two D blocks.
This slice starts from a precomputed first-block `Acc` tile and proves the final
masked writeback into `Out`, preserving the source `off_bh` decomposition and
`offs_m < total_seq_len` row mask. The CSR `tl.int32` row/column index casts,
`tl.float32` online-softmax accumulator, and `p.to(Q.dtype.element_ty)` dot input cast
belong to the omitted sparse-attention loop that produces `Acc`. -/
def block_sparse_attn_output_store_slice
    (Acc Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bh = tl.program_id(1)
  off_h = off_bh % $(num_heads)
  off_b = off_bh // $(num_heads)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_D))
  mask = (offs_m[:, None] < $(total_seq_len)) & (offs_d[None, :] < $(BLOCK_D))
  acc = tl.load(Acc + off_b * $(stride_acc_b) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_b * $(stride_ob) + off_h * $(stride_oh) +
      offs_m[:, None] * $(stride_om) + offs_d[None, :], acc, mask=mask)
}

/-- Surface transcription of the second output-block store in
`block_sparse_attn.py`'s `block_sparse_attention_kernel`.

The benchmark uses `NUM_D_BLOCKS = 2`, so Python stores `acc2` at
`out_ptrs + BLOCK_D` after the first output block. This slice starts from a
precomputed second-block `Acc2` tile and preserves the same row mask and
batch/head decomposition as the first store. -/
def block_sparse_attn_output_store_second_slice
    (Acc2 Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bh = tl.program_id(1)
  off_h = off_bh % $(num_heads)
  off_b = off_bh // $(num_heads)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_D))
  mask = (offs_m[:, None] < $(total_seq_len)) & (offs_d[None, :] < $(BLOCK_D))
  acc2 = tl.load(Acc2 + off_b * $(stride_acc_b) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_b * $(stride_ob) + off_h * $(stride_oh) +
      offs_m[:, None] * $(stride_om) + $(BLOCK_D) + offs_d[None, :],
    acc2, mask=mask)
}

def offH (s : BlockState) (num_heads : Nat) : Nat :=
  s.pids 1 % num_heads

def offB (s : BlockState) (num_heads : Nat) : Nat :=
  s.pids 1 / num_heads

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (total_seq_len BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Prop :=
  mIndex s BLOCK_M idx.1 < total_seq_len

instance activeDecidable
    (s : BlockState) (total_seq_len BLOCK_M BLOCK_D : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) :
    Decidable (active s total_seq_len BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (num_heads stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  offB s num_heads * stride_acc_b + offH s num_heads * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d

def outOffset
    (s : BlockState)
    (num_heads stride_ob stride_oh stride_om BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  offB s num_heads * stride_ob + offH s num_heads * stride_oh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx

noncomputable def accStoreValue
    (s : BlockState) (Acc : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m
      stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : ℝ :=
  WithBot.unbotD 0
    (if active s total_seq_len BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s num_heads stride_acc_b stride_acc_h stride_acc_m
          stride_acc_d BLOCK_M idx))
    else some (0.0 : ℝ))

/-! ## Genuine closed-form attention spec (NOT self-referential)

The block-sparse kernel's per-program output is, over `ℝ`, an honest
**causal natural-exp softmax attention** of the program's query tile against the
key/value rows the CSR layout selects — *not* the kernel's own executed value.
This section provides that genuine closed form as a tile-as-function, so the
final-store recipes can be stated and (where the loop fill is provided) bridged
against it.

For a program `(start_m, off_bh)`, with `qStart = start_m · BLOCK_M`:

* The query head `off_h = off_bh % num_heads`, batch `off_b = off_bh //
  num_heads`, and, for grouped-query attention, KV head
  `off_h_kv = off_h // (num_heads // num_kv_heads)`.
* The CSR loop visits `col_idx_idx ∈ [start_l, end_l)`, loading
  `col_idx = layout_csr_col_indices[layout_h · stride_col + col_idx_idx]` and
  attending the `BLOCK_N` keys at global positions `col_idx · BLOCK_N + t`,
  `t ∈ [0, BLOCK_N)`. The flattened selected-key index space therefore has size
  `numSelBlocks · BLOCK_N`, and the `r`-th selected key sits at global position
  `selKeyGlobal r`.
* The score is `softmax_scale · Σ_e Q[qStart+i, e] · K[selKeyGlobal r, e]`,
  causally masked: key `r` contributes iff `selKeyGlobal r ≤ qStart + i` (the
  Python `tl.where(offs_m ≥ start_n + offs_n, 0, -inf)`), and softmaxed with the
  natural exponential (`tl.exp`).
* The two output D-blocks read disjoint head channels: block 0 reads V channels
  `d ∈ [0, BLOCK_D)`, block 1 reads V channels `BLOCK_D + d` (`acc2` is stored at
  `out_ptrs + BLOCK_D`).

This is exactly `VeriTile.Triton.attentionRealCausal` evaluated on the gathered
selected-key tiles, except the causal predicate is on the *global* key position
`selKeyGlobal r` rather than on the gathered index `r`; we therefore inline the
softmax here with that global predicate. -/

/-- Grouped-query head map `num_heads // num_kv_heads`. -/
def headGroups (num_heads num_kv_heads : Nat) : Nat := num_heads / num_kv_heads

/-- KV head index `off_h // headGroups` for grouped-query attention. -/
def offHkv (s : BlockState) (num_heads num_kv_heads : Nat) : Nat :=
  offH s num_heads / headGroups num_heads num_kv_heads

/-- Q tile base offset `off_b · stride_qb + off_h · stride_qh`. -/
def qBase (s : BlockState) (num_heads stride_qb stride_qh : Nat) : Nat :=
  offB s num_heads * stride_qb + offH s num_heads * stride_qh

/-- K/V tile base offset `off_b · stride_kb + off_h_kv · stride_kh`
(grouped-query: the KV head is `off_h // headGroups`). -/
def kvBase (s : BlockState) (num_heads num_kv_heads stride_kb stride_kh : Nat) :
    Nat :=
  offB s num_heads * stride_kb + offHkv s num_heads num_kv_heads * stride_kh

/-- Global key position of the `r`-th flattened selected key:
`col_idx(r / BLOCK_N) · BLOCK_N + (r % BLOCK_N)`, where `col_idx(b)` is the
`b`-th CSR column index visited by the program, read from
`layout_csr_col_indices` at `layout_h · stride_col + start_l + b`. -/
def selKeyGlobal
    (s : BlockState) (layoutCols : Region .nat)
    (layout_h stride_col start_l BLOCK_N : Nat) (r : Nat) : Nat :=
  s.readMemValue .nat (Region.cast layoutCols)
      (layout_h * stride_col + start_l + r / BLOCK_N) * BLOCK_N +
    r % BLOCK_N

/-- Q row `qStart + i`, head channel `e`, read at
`qBase + (qStart+i) · stride_qm + e`. -/
noncomputable def qTileBSA (s : BlockState) (Q : RegionName)
    (num_heads stride_qb stride_qh stride_qm BLOCK_M : Nat)
    (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  s.readMem Q (qBase s num_heads stride_qb stride_qh +
    mIndex s BLOCK_M i * stride_qm + e)

/-- K row at global key position `n`, head channel `e`, read at
`kvBase + n · stride_kn + e`. -/
noncomputable def kRowBSA (s : BlockState) (K : RegionName)
    (num_heads num_kv_heads stride_kb stride_kh stride_kn : Nat)
    (n e : Nat) : ℝ :=
  s.readMem K (kvBase s num_heads num_kv_heads stride_kb stride_kh +
    n * stride_kn + e)

/-- V row at global key position `n`, head channel `d`, read at
`vbase + n · stride_vn + d`. The second D-block reads channel `BLOCK_D + d`. -/
noncomputable def vRowBSA (s : BlockState) (V : RegionName)
    (num_heads num_kv_heads stride_vb stride_vh stride_vn : Nat)
    (n d : Nat) : ℝ :=
  s.readMem V (kvBase s num_heads num_kv_heads stride_vb stride_vh +
    n * stride_vn + d)

/-- Unscaled raw score `Σ_{e<HEAD_DIM} Q[qStart+i,e] · K[n,e]` at global key `n`. -/
noncomputable def rawScoreBSA (s : BlockState) (Q K : RegionName)
    (num_heads num_kv_heads stride_qb stride_qh stride_qm
      stride_kb stride_kh stride_kn HEAD_DIM BLOCK_M : Nat)
    (i : Fin BLOCK_M) (n : Nat) : ℝ :=
  Finset.univ.sum (fun e : Fin HEAD_DIM =>
    qTileBSA s Q num_heads stride_qb stride_qh stride_qm BLOCK_M i e.val *
      kRowBSA s K num_heads num_kv_heads stride_kb stride_kh stride_kn n e.val)

/-- **Genuine closed-form block-sparse attention output** for one program.

`out[i, d] = (Σ_r w(i,r) · V[selKeyGlobal r, dChan d]) / (Σ_r w(i,r))`, where the
sum ranges over the `numSelBlocks · BLOCK_N` flattened selected keys, and
`w(i,r) = exp(softmax_scale · rawScore i (selKeyGlobal r))` when key
`selKeyGlobal r ≤ qStart + i` (causal), else `0`. `dChan d = dBlockBase + d`
selects the head channel for the chosen output D-block (`dBlockBase = 0` for the
first store, `BLOCK_D` for the second). This is `attentionRealCausal` with the
causal predicate evaluated on the global key position. -/
noncomputable def blockSparseAttnClosedForm
    (s : BlockState) (Q K V : RegionName) (layoutCols : Region .nat)
    (num_heads num_kv_heads
      stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
      stride_vb stride_vh stride_vn
      layout_h stride_col start_l numSelBlocks
      HEAD_DIM BLOCK_M BLOCK_N dBlockBase : Nat)
    (softmax_scale : ℝ)
    (i : Fin BLOCK_M) (d : Nat) : ℝ :=
  let n := fun r : Fin (numSelBlocks * BLOCK_N) =>
    selKeyGlobal s layoutCols layout_h stride_col start_l BLOCK_N r.val
  let w := fun r : Fin (numSelBlocks * BLOCK_N) =>
    if n r ≤ mIndex s BLOCK_M i then
      Real.exp (softmax_scale *
        rawScoreBSA s Q K num_heads num_kv_heads stride_qb stride_qh stride_qm
          stride_kb stride_kh stride_kn HEAD_DIM BLOCK_M i (n r))
    else 0
  let denom := Finset.univ.sum (fun r => w r)
  let numer := Finset.univ.sum (fun r =>
    w r * vRowBSA s V num_heads num_kv_heads stride_vb stride_vh stride_vn
      (n r) (dBlockBase + d))
  numer / denom

/-! ## Gathered global-causal streaming math (`BSAMathCausal`)

This is the analog of `VeriTile.Examples.FA1MathCausal`, but the causal predicate
is evaluated on the **gathered global key position** `gpos r` rather than on the
flat streaming index `r`. The block-sparse kernel walks a CSR-selected,
non-contiguous set of `BLOCK_N`-key blocks; after gathering, the `r`-th streamed
key sits at global position `gpos r = selKeyGlobal r`, and the online-softmax
masks key `r` exactly when `gpos r ≤ qStart + i` (the Python
`tl.where(offs_m ≥ start_n + offs_n, 0, -inf)`).

The streaming accumulators (`bsaMPartial`/`bsaLPartial`/`bsaOPartial`) run over
the **gathered** key/value stream `Kg`/`Vg : TileIndex [Bk * N, D] → ℝ`
(`Kg (r, e)` is the `e`-th channel of the key gathered at flat index `r`). Only
the mask predicate consults `gpos`; the scores/values read the gathered tile at
`r` directly. This is a strict generalization of `FA1MathCausal` (recover FA1 by
`gpos = fun j => j.val`), so every lemma mirrors it line-for-lemma, substituting
the gathered-global predicate.

The two output D-blocks share the **same** accumulator recurrence (same scores,
same softmax weights); they differ only in which V projection is read (`Vg` for
block 0, a second `Vg2`/channel offset for block 1). Hence one accumulator
family suffices; the second D-block is obtained by re-instantiating `bsaOPartial`
with the second value tile. -/

namespace BSAMathCausal

open VeriTile.Triton

/-- Gathered scaled score: `scale · Σ_e Q[i,e] · Kg[r,e]`, reading the gathered
key tile at flat stream index `r` (no global remap on the score itself). -/
noncomputable def gScore {M N D Bk : Nat}
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) (r : Fin (Bk * N)) : ℝ :=
  scale * Finset.univ.sum (fun d : Fin D =>
    Q (i, d, PUnit.unit) * Kg (r, d, PUnit.unit))

/-- Causal masked score under the **gathered global position** predicate.
Returns `⊥` when the gathered key's global position `gpos r` is in the future
(`> qStart + i`), otherwise the ordinary gathered scaled score. -/
noncomputable def maskedScore {M N D Bk : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) (r : Fin (Bk * N)) : WithBot ℝ :=
  if gpos r ≤ qStart + i.val then
    ((gScore Q Kg scale i r : ℝ) : WithBot ℝ)
  else
    ⊥

@[simp] theorem maskedScore_of_le {M N D Bk : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) (r : Fin (Bk * N)) (h : gpos r ≤ qStart + i.val) :
    maskedScore qStart gpos Q Kg scale i r =
      ((gScore Q Kg scale i r : ℝ) : WithBot ℝ) := by
  simp [maskedScore, h]

@[simp] theorem maskedScore_of_not_le {M N D Bk : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ) (scale : ℝ)
    (i : Fin M) (r : Fin (Bk * N)) (h : ¬ gpos r ≤ qStart + i.val) :
    maskedScore qStart gpos Q Kg scale i r = (⊥ : WithBot ℝ) := by
  simp [maskedScore, h]

/-- Running per-row max of gathered causal masked scores over the first `k`
KV blocks. Future (gathered-global) keys enter as `⊥`. -/
noncomputable def bsaMPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        max (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i)
          ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
            maskedScore qStart gpos Q Kg scale i
              (StreamingAccumulator.blockIndex Bk numKVBlocks k h jLocal))
      else
        bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i

/-- Running causal softmax normalizer, shifted by the running max. -/
noncomputable def bsaLPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → Fin M → ℝ
  | 0, _ => 0
  | k + 1, i =>
      if h : k + 1 ≤ numKVBlocks then
        let alpha :=
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i)
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i))).unbotD 0
        alpha * bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k i +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart gpos Q Kg scale i
                  (StreamingAccumulator.blockIndex Bk numKVBlocks k h jLocal))
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i))).unbotD 0)
      else
        bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k i

/-- Running causal unnormalized output accumulator over the gathered value
stream `Vg`. The two D-blocks differ only by which `Vg` is supplied. -/
noncomputable def bsaOPartial {M D Dv : Nat} (Bk : Nat)
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (Vg : TileIndex [Bk * numKVBlocks, Dv] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, Dv] → ℝ
  | 0, _ => 0
  | k + 1, idx =>
      if h : k + 1 ≤ numKVBlocks then
        let alpha :=
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1)
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0
        alpha * bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale k idx +
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := StreamingAccumulator.blockIndex Bk numKVBlocks k h jLocal
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart gpos Q Kg scale idx.1 j)
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0 *
              Vg (j, idx.2.1, PUnit.unit))
      else
        bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale k idx

/-- Recurrence unfold for `bsaMPartial` at iteration `k+1`, when `k < N`. -/
theorem bsaMPartial_succ_of_lt {M D Bk : Nat}
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i =
      max (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i)
        ((Finset.univ : Finset (Fin Bk)).sup fun jLocal =>
          maskedScore qStart gpos Q Kg scale i
            (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal)) := by
  conv_lhs => rw [bsaMPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-- Recurrence unfold for `bsaLPartial` at iteration `k+1`, when `k < N`. -/
theorem bsaLPartial_succ_of_lt {M D Bk : Nat}
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (i : Fin M) :
    bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i =
      let alpha :=
        (WithBot.realExp
          (Option.map₂ (fun x y : ℝ => x - y)
            (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i)
            (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i))).unbotD 0
      alpha * bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k i +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart gpos Q Kg scale i
                (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal))
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i))).unbotD 0) := by
  conv_lhs => rw [bsaLPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-- Recurrence unfold for `bsaOPartial` at iteration `k+1`, when `k < N`. -/
theorem bsaOPartial_succ_of_lt {M D Dv Bk : Nat}
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (Vg : TileIndex [Bk * numKVBlocks, Dv] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, Dv]) :
    bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale (k + 1) idx =
      let alpha :=
        (WithBot.realExp
          (Option.map₂ (fun x y : ℝ => x - y)
            (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1)
            (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0
      alpha * bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale k idx +
        (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          let j := StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr hk) jLocal
          (WithBot.realExp
            (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart gpos Q Kg scale idx.1 j)
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0 *
            Vg (j, idx.2.1, PUnit.unit)) := by
  conv_lhs => rw [bsaOPartial]
  rw [dif_pos (Nat.succ_le_iff.mpr hk)]

/-! ### Gathered m-free reference sums -/

/-- Gathered-causal m-free normalizer over the first `k` KV blocks. -/
noncomputable def lFree {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k ≤ N) (i : Fin M) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    let j := StreamingAccumulator.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL
    if gpos j ≤ qStart + i.val then
      Real.exp (gScore Q Kg scale i j)
    else
      0))

/-- Gathered-causal m-free unnormalized output over the first `k` KV blocks. -/
noncomputable def oFree {M D Dv Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (Vg : TileIndex [Bk * N, Dv] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k ≤ N)
    (idx : TileIndex [M, Dv]) : ℝ :=
  Finset.univ.sum (fun n : Fin k => Finset.univ.sum (fun jL : Fin Bk =>
    let j := StreamingAccumulator.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL
    (if gpos j ≤ qStart + idx.1.val then
      Real.exp (gScore Q Kg scale idx.1 j)
    else
      0) * Vg (j, idx.2.1, PUnit.unit)))

@[simp] theorem lFree_zero {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    lFree qStart gpos Q Kg scale 0 (Nat.zero_le _) i = 0 := by
  unfold lFree
  simp

@[simp] theorem oFree_zero {M D Dv Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (Vg : TileIndex [Bk * N, Dv] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Dv]) :
    oFree qStart gpos Q Kg Vg scale 0 (Nat.zero_le _) idx = 0 := by
  unfold oFree
  simp

/-- Recurrence for gathered-causal `lFree`. -/
theorem lFree_succ {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k + 1 ≤ N) (i : Fin M) :
    lFree qStart gpos Q Kg scale (k + 1) hk i =
      lFree qStart gpos Q Kg scale k (Nat.le_of_succ_le hk) i +
      Finset.univ.sum (fun jL : Fin Bk =>
        let j := StreamingAccumulator.blockIndex Bk N k hk jL
        if gpos j ≤ qStart + i.val then
          Real.exp (gScore Q Kg scale i j)
        else
          0) := by
  unfold lFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Recurrence for gathered-causal `oFree`. -/
theorem oFree_succ {M D Dv Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (Vg : TileIndex [Bk * N, Dv] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k + 1 ≤ N)
    (idx : TileIndex [M, Dv]) :
    oFree qStart gpos Q Kg Vg scale (k + 1) hk idx =
      oFree qStart gpos Q Kg Vg scale k (Nat.le_of_succ_le hk) idx +
      Finset.univ.sum (fun jL : Fin Bk =>
        let j := StreamingAccumulator.blockIndex Bk N k hk jL
        (if gpos j ≤ qStart + idx.1.val then
          Real.exp (gScore Q Kg scale idx.1 j)
        else
          0) * Vg (j, idx.2.1, PUnit.unit)) := by
  unfold oFree
  rw [Fin.sum_univ_castSucc]
  simp [Fin.val_last]

/-- Flat form of the final gathered-causal normalizer. -/
theorem lFree_eq_flat {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M) :
    lFree qStart gpos Q Kg scale N (le_refl N) i =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        if gpos j ≤ qStart + i.val then
          Real.exp (gScore Q Kg scale i j)
        else
          0) := by
  unfold lFree
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (StreamingAccumulator.blockIndexEquiv Bk N) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [StreamingAccumulator.blockIndex_blockIndexEquiv]

/-- Flat form of the final gathered-causal output accumulator. -/
theorem oFree_eq_flat {M D Dv Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (Vg : TileIndex [Bk * N, Dv] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Dv]) :
    oFree qStart gpos Q Kg Vg scale N (le_refl N) idx =
      Finset.univ.sum (fun j : Fin (Bk * N) =>
        (if gpos j ≤ qStart + idx.1.val then
          Real.exp (gScore Q Kg scale idx.1 j)
        else
          0) * Vg (j, idx.2.1, PUnit.unit)) := by
  unfold oFree
  rw [← Finset.sum_product', Finset.univ_product_univ]
  refine (Finset.sum_equiv (StreamingAccumulator.blockIndexEquiv Bk N) ?_ ?_).symm
  · intro _; simp
  · intro j _
    rw [StreamingAccumulator.blockIndex_blockIndexEquiv]

/-- `bsaMPartial (k+1) i` is non-`⊥` whenever `0 < Bk`, `k+1 ≤ numKVBlocks`,
and the first gathered key (flat index `0`) is causally visible
(`gpos 0 ≤ qStart + i`). For block-sparse attention the first CSR-selected
block has global base `col_idx · BLOCK_N` and the kernel only selects blocks at
or before the query (so the first selected key is visible); this is supplied as
`hVis0`. -/
theorem bsaMPartial_succ_ne_bot {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat) (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k + 1 ≤ numKVBlocks) (i : Fin M)
    (hVis0 : gpos ⟨0, Nat.mul_pos hBk hN⟩ ≤ qStart + i.val) :
    bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i ≠ ⊥ := by
  induction k with
  | zero =>
      rw [bsaMPartial_succ_of_lt qStart numKVBlocks gpos Q Kg scale 0
            (Nat.lt_of_succ_le hk) i]
      have hidx0 : (StreamingAccumulator.blockIndex Bk numKVBlocks 0
                (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk))
                (⟨0, hBk⟩ : Fin Bk)) = ⟨0, Nat.mul_pos hBk hN⟩ := by
        apply Fin.ext; simp [StreamingAccumulator.blockIndex]
      have h0 : maskedScore qStart gpos Q Kg scale i
              (StreamingAccumulator.blockIndex Bk numKVBlocks 0
                (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk))
                (⟨0, hBk⟩ : Fin Bk)) ≠ ⊥ := by
        rw [maskedScore_of_le qStart gpos Q Kg scale i _ (by rw [hidx0]; exact hVis0)]
        exact WithBot.coe_ne_bot
      show bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale 0 i ⊔ _ ≠ ⊥
      change ⊥ ⊔ _ ≠ ⊥
      rw [bot_sup_eq]
      simp [Finset.sup_eq_bot_iff]
      exact ⟨⟨0, hBk⟩, h0⟩
  | succ k' ih =>
      have hk' : k' + 1 ≤ numKVBlocks := by omega
      rw [bsaMPartial_succ_of_lt qStart numKVBlocks gpos Q Kg scale (k' + 1)
            (Nat.lt_of_succ_le hk) i]
      intro hcontra
      have h_left : bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k' + 1) i ≤ ⊥ := by
        rw [← hcontra]; exact le_max_left _ _
      exact ih hk' (le_bot_iff.mp h_left)

/-- Gathered-causal streaming normalizer equals the gathered m-free normalizer
times the final exponential shift. Mirrors `FA1MathCausal.lPartial_eq_mShifted`
with the global-position predicate. -/
theorem bsaLPartial_eq_mShifted {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat) (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (i : Fin M)
    (hVis0 : gpos ⟨0, Nat.mul_pos hBk hN⟩ ≤ qStart + i.val) :
    bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k i =
      Real.exp (-(bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i).unbotD 0) *
        lFree qStart gpos Q Kg scale k hk i := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          lFree qStart gpos Q Kg scale 0 hk i
      rw [lFree_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [bsaLPartial_succ_of_lt qStart numKVBlocks gpos Q Kg scale k
        (Nat.lt_of_succ_le hk) i]
      rw [ih hk']
      rw [lFree_succ qStart gpos Q Kg scale k hk i, mul_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart gpos Q Kg scale i
                  (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal))
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i))).unbotD 0)
          =
          Real.exp (-(bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
              let j := StreamingAccumulator.blockIndex Bk numKVBlocks k hk jLocal
              if gpos j ≤ qStart + i.val then
                Real.exp (gScore Q Kg scale i j)
              else
                0) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        let j := StreamingAccumulator.blockIndex Bk numKVBlocks k hk jLocal
        have hjEq :
            StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal = j := by
          rfl
        rw [hjEq]
        by_cases hvis : gpos j ≤ qStart + i.val
        · rw [maskedScore_of_le qStart gpos Q Kg scale i j hvis]
          obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
            (bsaMPartial_succ_ne_bot hBk qStart numKVBlocks hN gpos Q Kg scale k hk i
              hVis0)
          rw [← hm]
          simp [hvis]
          rw [show gScore Q Kg scale i j - m = -m + gScore Q Kg scale i j by ring,
              Real.exp_add]
        · rw [maskedScore_of_not_le qStart gpos Q Kg scale i j hvis]
          rw [show (Option.map₂ (fun x y : ℝ => x - y) (⊥ : WithBot ℝ)
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i))
              = (⊥ : WithBot ℝ) from rfl]
          rw [if_neg hvis]
          show (WithBot.realExp (⊥ : WithBot ℝ)).unbotD 0 = _
          simp [WithBot.realExp]
      have hSumA :
          (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i)
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i))).unbotD 0 *
            (Real.exp (-(bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k i).unbotD 0) *
              lFree qStart gpos Q Kg scale k hk' i)
          =
          Real.exp (-(bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) i).unbotD 0) *
            lFree qStart gpos Q Kg scale k hk' i := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [lFree_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne := bsaMPartial_succ_ne_bot hBk qStart numKVBlocks hN gpos Q Kg scale k' hk_succ i hVis0
          have hmk1_ne := bsaMPartial_succ_ne_bot hBk qStart numKVBlocks hN gpos Q Kg scale (k' + 1) hk i hVis0
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          rw [← hrk, ← hrk1]
          simp [WithBot.realExp]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
            rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * lFree qStart gpos Q Kg scale (k' + 1) hk' i)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      lFree qStart gpos Q Kg scale (k' + 1) hk' i) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
            rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Gathered-causal streaming output accumulator equals the gathered m-free
output accumulator times the final exponential shift. -/
theorem bsaOPartial_eq_mShifted {M D Dv Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat) (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (Vg : TileIndex [Bk * numKVBlocks, Dv] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, Dv])
    (hVis0 : gpos ⟨0, Nat.mul_pos hBk hN⟩ ≤ qStart + idx.1.val) :
    bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale k idx =
      Real.exp (-(bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1).unbotD 0) *
        oFree qStart gpos Q Kg Vg scale k hk idx := by
  induction k with
  | zero =>
      show (0 : ℝ) =
        Real.exp (-((⊥ : WithBot ℝ).unbotD 0)) *
          oFree qStart gpos Q Kg Vg scale 0 hk idx
      rw [oFree_zero]
      ring
  | succ k ih =>
      have hk' : k ≤ numKVBlocks := Nat.le_of_succ_le hk
      rw [bsaOPartial_succ_of_lt qStart numKVBlocks gpos Q Kg Vg scale k
        (Nat.lt_of_succ_le hk) idx]
      rw [ih hk']
      rw [oFree_succ qStart gpos Q Kg Vg scale k hk idx, mul_add]
      have hSumB :
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            let j := StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal
            (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (maskedScore qStart gpos Q Kg scale idx.1 j)
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0 *
              Vg (j, idx.2.1, PUnit.unit))
          =
          Real.exp (-(bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1).unbotD 0) *
            (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
              let j := StreamingAccumulator.blockIndex Bk numKVBlocks k hk jLocal
              (if gpos j ≤ qStart + idx.1.val then
                Real.exp (gScore Q Kg scale idx.1 j)
              else
                0) * Vg (j, idx.2.1, PUnit.unit)) := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro jLocal _
        let j := StreamingAccumulator.blockIndex Bk numKVBlocks k hk jLocal
        have hjEq :
            StreamingAccumulator.blockIndex Bk numKVBlocks k
              (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal = j := by
          rfl
        rw [hjEq]
        dsimp only
        by_cases hvis : gpos j ≤ qStart + idx.1.val
        · rw [maskedScore_of_le qStart gpos Q Kg scale idx.1 j hvis]
          obtain ⟨m, hm⟩ := WithBot.ne_bot_iff_exists.mp
            (bsaMPartial_succ_ne_bot hBk qStart numKVBlocks hN gpos Q Kg scale k hk idx.1
              hVis0)
          rw [← hm]
          simp [hvis]
          rw [show gScore Q Kg scale idx.1 j - m =
                -m + gScore Q Kg scale idx.1 j by ring,
              Real.exp_add]
          ring
        · rw [maskedScore_of_not_le qStart gpos Q Kg scale idx.1 j hvis]
          rw [show (Option.map₂ (fun x y : ℝ => x - y) (⊥ : WithBot ℝ)
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))
              = (⊥ : WithBot ℝ) from rfl]
          rw [if_neg hvis]
          show (WithBot.realExp (⊥ : WithBot ℝ)).unbotD 0 * Vg (j, idx.2.1, PUnit.unit) = _
          simp [WithBot.realExp]
      have hSumA :
          (WithBot.realExp
              (Option.map₂ (fun x y : ℝ => x - y)
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1)
                (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0 *
            (Real.exp (-(bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1).unbotD 0) *
              oFree qStart gpos Q Kg Vg scale k hk' idx)
          =
          Real.exp (-(bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1).unbotD 0) *
            oFree qStart gpos Q Kg Vg scale k hk' idx := by
        rcases Nat.eq_zero_or_pos k with hkz | hkpos
        · subst hkz
          rw [oFree_zero]
          ring
        · obtain ⟨k', rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.pos_iff_ne_zero.mp hkpos)
          have hk_succ : k' + 1 ≤ numKVBlocks := Nat.le_of_succ_le hk
          have hmk_ne := bsaMPartial_succ_ne_bot hBk qStart numKVBlocks hN gpos Q Kg scale k' hk_succ idx.1 hVis0
          have hmk1_ne := bsaMPartial_succ_ne_bot hBk qStart numKVBlocks hN gpos Q Kg scale (k' + 1) hk idx.1 hVis0
          obtain ⟨rk, hrk⟩ := WithBot.ne_bot_iff_exists.mp hmk_ne
          obtain ⟨rk1, hrk1⟩ := WithBot.ne_bot_iff_exists.mp hmk1_ne
          rw [← hrk, ← hrk1]
          simp [WithBot.realExp]
          rw [show Real.exp (rk - rk1) = Real.exp (-rk1) * Real.exp rk by
            rw [show rk - rk1 = -rk1 + rk by ring, Real.exp_add]]
          rw [show Real.exp (-rk1) * Real.exp rk *
                    (Real.exp (-rk) * oFree qStart gpos Q Kg Vg scale (k' + 1) hk' idx)
                = Real.exp (-rk1) *
                    (Real.exp rk * Real.exp (-rk) *
                      oFree qStart gpos Q Kg Vg scale (k' + 1) hk' idx) by ring]
          rw [show Real.exp rk * Real.exp (-rk) = 1 by
            rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]]
          ring
      linarith [hSumA, hSumB]

/-- Gathered-causal attention closed form over the flat selected-key stream.
This is the `attentionRealCausalBlock` analog with the causal predicate on the
gathered global position `gpos`. It is exactly the shape of
`blockSparseAttnClosedForm` (gathered keys/values, global-position causal mask). -/
noncomputable def bsaAttn {M D Dv Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (Vg : TileIndex [Bk * N, Dv] → ℝ)
    (scale : ℝ) : TileIndex [M, Dv] → ℝ :=
  fun (i, d, _) =>
    let weight := fun j : Fin (Bk * N) =>
      if gpos j ≤ qStart + i.val then Real.exp (gScore Q Kg scale i j) else 0
    let denom := Finset.univ.sum (fun j => weight j)
    let numer := Finset.univ.sum (fun j => weight j * Vg (j, d, PUnit.unit))
    numer / denom

/-- Gathered m-free ratio is exactly the gathered closed-form attention. -/
theorem oFree_div_lFree_eq_bsaAttn {M D Dv Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (Vg : TileIndex [Bk * N, Dv] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, Dv]) :
    oFree qStart gpos Q Kg Vg scale N (le_refl N) idx /
        lFree qStart gpos Q Kg scale N (le_refl N) idx.1
      = bsaAttn qStart gpos Q Kg Vg scale idx := by
  rw [oFree_eq_flat, lFree_eq_flat]
  unfold bsaAttn
  rfl

/-- The final gathered-causal m-free normalizer is positive, **given** the first
gathered key is causally visible (`hVis0`). This is the block-sparse analog of
`FA1MathCausal.lFree_final_pos`; in the dense FA1 case `gpos 0 = 0` makes
visibility automatic, but with CSR gathering it is supplied as a hypothesis
(discharged at exec-assembly time from the CSR selection schedule). -/
theorem lFree_final_pos {M D Bk N : Nat} (hBk : 0 < Bk) (hN : 0 < N)
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (i : Fin M)
    (hVis0 : gpos ⟨0, Nat.mul_pos hBk hN⟩ ≤ qStart + i.val) :
    0 < lFree qStart gpos Q Kg scale N (le_refl N) i := by
  rw [lFree_eq_flat]
  apply Finset.sum_pos'
  · intro j _
    by_cases h : gpos j ≤ qStart + i.val
    · simp [h, le_of_lt (Real.exp_pos _)]
    · simp [h]
  · refine ⟨⟨0, Nat.mul_pos hBk hN⟩, Finset.mem_univ _, ?_⟩
    simp [hVis0, Real.exp_pos]

/-- The final gathered-causal streaming normalizer is nonzero under non-empty
KV scope and first-key visibility. -/
theorem bsaLPartial_final_ne_zero {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat) (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (i : Fin M)
    (hVis0 : gpos ⟨0, Nat.mul_pos hBk hN⟩ ≤ qStart + i.val) :
    bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale numKVBlocks i ≠ 0 := by
  rw [bsaLPartial_eq_mShifted hBk qStart numKVBlocks hN gpos Q Kg scale numKVBlocks
      (le_refl _) i hVis0]
  exact mul_ne_zero (Real.exp_ne_zero _)
    (ne_of_gt (lFree_final_pos hBk hN qStart gpos Q Kg scale i hVis0))

/-- **Load-bearing streaming-eq bridge.** The final gathered-causal streaming
ratio equals the gathered closed-form attention `bsaAttn` — i.e. the online
softmax over the CSR-selected (non-contiguous, global-causal) key stream computes
exactly the closed-form attention. This is the block-sparse analog of
`FA1MathCausal.streaming_eq_attentionRealCausalBlock`. -/
theorem bsaStreaming_eq_bsaAttn {M D Dv Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat) (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ)
    (Vg : TileIndex [Bk * numKVBlocks, Dv] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, Dv])
    (hVis0 : gpos ⟨0, Nat.mul_pos hBk hN⟩ ≤ qStart + idx.1.val) :
    bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale numKVBlocks idx /
        bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale numKVBlocks idx.1
      = bsaAttn qStart gpos Q Kg Vg scale idx := by
  have hl : bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale numKVBlocks idx.1 ≠ 0 :=
    bsaLPartial_final_ne_zero hBk qStart numKVBlocks hN gpos Q Kg scale idx.1 hVis0
  rw [bsaOPartial_eq_mShifted hBk qStart numKVBlocks hN gpos Q Kg Vg scale numKVBlocks
        (le_refl _) idx hVis0,
      bsaLPartial_eq_mShifted hBk qStart numKVBlocks hN gpos Q Kg scale numKVBlocks
        (le_refl _) idx.1 hVis0]
  rw [mul_div_mul_left _ _ (Real.exp_ne_zero _)]
  exact oFree_div_lFree_eq_bsaAttn qStart gpos Q Kg Vg scale idx

/-- **Bridge: gathered closed form = block-sparse closed form.** Instantiating
the gathered `bsaAttn` with the CSR-selected keys/values (`Kg = kRowBSA ∘
selKeyGlobal`, `Vg = vRowBSA ∘ selKeyGlobal` at the chosen D-block channels,
`gpos = selKeyGlobal`, `qStart = pids 0 · BLOCK_M`) yields exactly
`blockSparseAttnClosedForm`. The streaming math (`bsaStreaming_eq_bsaAttn`) thus
computes the genuine block-sparse closed form. -/
theorem bsaAttn_eq_blockSparseAttnClosedForm
    (s : BlockState) (Q K V : RegionName) (layoutCols : Region .nat)
    (num_heads num_kv_heads
      stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
      stride_vb stride_vh stride_vn
      layout_h stride_col start_l numSelBlocks
      HEAD_DIM BLOCK_M BLOCK_N BLOCK_D dBlockBase : Nat)
    (softmax_scale : ℝ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_D) :
    BSAMathCausal.bsaAttn (M := BLOCK_M) (D := HEAD_DIM) (Dv := BLOCK_D)
        (Bk := numSelBlocks) (N := BLOCK_N)
        (s.pids 0 * BLOCK_M)
        (fun r => selKeyGlobal s layoutCols layout_h stride_col start_l BLOCK_N r.val)
        (fun idx => qTileBSA s Q num_heads stride_qb stride_qh stride_qm BLOCK_M
          idx.1 idx.2.1.val)
        (fun idx => kRowBSA s K num_heads num_kv_heads stride_kb stride_kh stride_kn
          (selKeyGlobal s layoutCols layout_h stride_col start_l BLOCK_N idx.1.val)
          idx.2.1.val)
        (fun idx => vRowBSA s V num_heads num_kv_heads stride_vb stride_vh stride_vn
          (selKeyGlobal s layoutCols layout_h stride_col start_l BLOCK_N idx.1.val)
          (dBlockBase + idx.2.1.val))
        softmax_scale (i, d, PUnit.unit)
      = blockSparseAttnClosedForm s Q K V layoutCols num_heads num_kv_heads
          stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
          stride_vb stride_vh stride_vn layout_h stride_col start_l numSelBlocks
          HEAD_DIM BLOCK_M BLOCK_N dBlockBase softmax_scale i d.val := by
  unfold BSAMathCausal.bsaAttn blockSparseAttnClosedForm
  simp only [BSAMathCausal.gScore, rawScoreBSA, mIndex]

/-- `bsaAttn` only sums over `Fin (Bk * N)` via each index's `.val`, so it is
invariant under any reshaping `Bk₁ * N₁ = Bk₂ * N₂` that preserves the underlying
`Nat`-valued data (`gpos`/`Kg`/`Vg` are `.val`-functions). This bridges the
streaming instantiation (`Bk = BLOCK_N` keys per block × `numKVBlocks` iterations,
flat domain `Fin (BLOCK_N * numKVBlocks)`) to the closed-form layout
(`numSelBlocks` blocks × `BLOCK_N` keys, domain `Fin (numSelBlocks * BLOCK_N)`):
both flat domains have the same size and index the same gathered keys by `.val`. -/
theorem bsaAttn_reindex {M D Dv Bk₁ N₁ Bk₂ N₂ : Nat} (h : Bk₁ * N₁ = Bk₂ * N₂)
    (qStart : Nat)
    (gposVal : Nat → Nat) (KgVal : Nat → Fin D → ℝ) (VgVal : Nat → Fin Dv → ℝ)
    (Q : TileIndex [M, D] → ℝ) (scale : ℝ) (idx : TileIndex [M, Dv]) :
    bsaAttn (M := M) (D := D) (Dv := Dv) (Bk := Bk₁) (N := N₁) qStart
        (fun r => gposVal r.val) Q
        (fun jx => KgVal jx.1.val jx.2.1) (fun jx => VgVal jx.1.val jx.2.1) scale idx
      = bsaAttn (M := M) (D := D) (Dv := Dv) (Bk := Bk₂) (N := N₂) qStart
        (fun r => gposVal r.val) Q
        (fun jx => KgVal jx.1.val jx.2.1) (fun jx => VgVal jx.1.val jx.2.1) scale idx := by
  obtain ⟨i, d, u⟩ := idx
  -- The per-key gScore depends on the index only via `.val` (through `KgVal`).
  set gsv : Nat → ℝ := fun n =>
    scale * Finset.univ.sum (fun e : Fin D => Q (i, e, PUnit.unit) * KgVal n e) with hgsv
  have hgScore₁ : ∀ r : Fin (Bk₁ * N₁),
      gScore Q (fun jx : TileIndex [Bk₁ * N₁, D] => KgVal jx.1.val jx.2.1) scale i r = gsv r.val := by
    intro r; rfl
  have hgScore₂ : ∀ r : Fin (Bk₂ * N₂),
      gScore Q (fun jx : TileIndex [Bk₂ * N₂, D] => KgVal jx.1.val jx.2.1) scale i r = gsv r.val := by
    intro r; rfl
  -- generic `.val`-indexed reshape of a sum
  have hg : ∀ (f : Nat → ℝ),
      (Finset.univ : Finset (Fin (Bk₁ * N₁))).sum (fun j => f j.val)
        = (Finset.univ : Finset (Fin (Bk₂ * N₂))).sum (fun j => f j.val) := by
    intro f
    refine Finset.sum_nbij' (fun j => (finCongr h) j) (fun j => (finCongr h.symm) j)
      ?_ ?_ ?_ ?_ ?_ <;> intro j _ <;> simp [finCongr]
  unfold bsaAttn
  simp only [hgScore₁, hgScore₂]
  refine congrArg₂ (· / ·) ?_ ?_
  · exact hg (fun n => (if gposVal n ≤ qStart + i.val then Real.exp (gsv n) else 0) * VgVal n d)
  · exact hg (fun n => if gposVal n ≤ qStart + i.val then Real.exp (gsv n) else 0)

end BSAMathCausal

/-- Algorithm-layer correctness for the first block-sparse output store. -/
theorem block_sparse_attn_output_store_slice_correct
    (Acc Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      let outAddr := outOffset s num_heads stride_ob stride_oh stride_om
        BLOCK_M idx
      (exec (block_sparse_attn_output_store_slice Acc Out num_heads
            total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
            stride_ob stride_oh stride_om BLOCK_M BLOCK_D) s).map
          (·.readMem Out outAddr)
        = some (if active s total_seq_len BLOCK_M idx then
            accStoreValue s Acc num_heads total_seq_len stride_acc_b
              stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, block_sparse_attn_output_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
        IntegralDType.mod, ComparableDType.lt, offH, offB, mIndex, dIndex,
        active, accOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_D] → Nat :=
    fun idx =>
      s.pids 1 / num_heads * stride_ob + s.pids 1 % num_heads * stride_oh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, BLOCK_D] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
          some (s.readMem Acc
            (s.pids 1 / num_heads * stride_acc_b +
              s.pids 1 % num_heads * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_D] → Prop :=
    fun idx => s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offH, offB, mIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_D])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc num_heads total_seq_len stride_acc_b stride_acc_h
        stride_acc_m stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
  · simp [P, valueFn, accStoreValue, active, accOffset, offB, offH, mIndex,
      dIndex, hActive]
  · simp [P, valueFn, accStoreValue, active, accOffset, offB, offH, mIndex,
      dIndex, hActive]

/-- Compute-facing correctness for the first block-sparse output store. -/
theorem block_sparse_attn_output_store_slice_compute_correct
    (Acc Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := block_sparse_attn_output_store_slice Acc Out num_heads
        total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        stride_ob stride_oh stride_om BLOCK_M BLOCK_D)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
          active s total_seq_len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out,
          outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        accStoreValue s Acc num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [block_sparse_attn_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := block_sparse_attn_output_store_slice_correct Acc Out num_heads
    total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_ob
    stride_oh stride_om BLOCK_M BLOCK_D s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Output offset for the second block-sparse store. -/
def out2Offset
    (s : BlockState)
    (num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D]) : Nat :=
  offB s num_heads * stride_ob + offH s num_heads * stride_oh +
    mIndex s BLOCK_M idx.1 * stride_om + BLOCK_D + dIndex idx

/-- Algorithm-layer correctness for the second block-sparse output store. -/
theorem block_sparse_attn_output_store_second_slice_correct
    (Acc2 Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        out2Offset s num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      let outAddr := out2Offset s num_heads stride_ob stride_oh stride_om
        BLOCK_M BLOCK_D idx
      (exec (block_sparse_attn_output_store_second_slice Acc2 Out num_heads
            total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
            stride_ob stride_oh stride_om BLOCK_M BLOCK_D) s).map
          (·.readMem Out outAddr)
        = some (if active s total_seq_len BLOCK_M idx then
            accStoreValue s Acc2 num_heads total_seq_len stride_acc_b
              stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, block_sparse_attn_output_store_second_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
        IntegralDType.mod, ComparableDType.lt, offH, offB, mIndex, dIndex,
        active, accOffset, out2Offset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_D] → Nat :=
    fun idx =>
      s.pids 1 / num_heads * stride_ob + s.pids 1 % num_heads * stride_oh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om + BLOCK_D + idx.2.1.val
  let valueFn : TileIndex [BLOCK_M, BLOCK_D] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
          some (s.readMem Acc2
            (s.pids 1 / num_heads * stride_acc_b +
              s.pids 1 % num_heads * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_D] → Prop :=
    fun idx => s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, out2Offset, offH, offB, mIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_D])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc2 num_heads total_seq_len stride_acc_b stride_acc_h
        stride_acc_m stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
  · rfl
  · rfl

/-- Compute-facing correctness for the second block-sparse output store. -/
theorem block_sparse_attn_output_store_second_slice_compute_correct
    (Acc2 Out : RegionName)
    (num_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        out2Offset s num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := block_sparse_attn_output_store_second_slice Acc2 Out num_heads
        total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        stride_ob stride_oh stride_om BLOCK_M BLOCK_D)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
          active s total_seq_len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out,
          out2Offset s num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        accStoreValue s Acc2 num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [block_sparse_attn_output_store_second_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := block_sparse_attn_output_store_second_slice_correct Acc2 Out num_heads
    total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_ob
    stride_oh stride_om BLOCK_M BLOCK_D s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

section BSARecipes

open VeriTile.Triton

/-- **`if cond { thenBody } else { elseBody }` step, true branch.** When the
elaborated condition Op evaluates to the scalar boolean `true`, the runtime
`ifThenElse` statement steps to running `thenBody`. Used to discharge the
concrete `if EVEN_N {...} else {...}` branches in the CSR loop body chain
(`EVEN_N = true` ⇒ unmasked loads). -/
theorem stepStmt_ifThenElse_true {cond : Op .bool []}
    {thenBody elseBody : List Stmt} {s s' : BlockState}
    (hcond : evalOp cond s = some (Tile.scalar (Bool.true)))
    (hthen : stepStmts thenBody s = some s') :
    stepStmt (.ifThenElse cond thenBody elseBody) s = some s' := by
  simp only [stepStmt, hcond, Option.bind_some, Tile.scalar_data, if_true]
  exact hthen

/-- **`if cond { body }` step, true branch.** When the elaborated condition Op
evaluates to scalar `true`, the runtime `ifThen` statement steps to running
`body`. Used for the concrete `if NUM_D_BLOCKS ≥ 2 { ... }` D-block branches
(`NUM_D_BLOCKS = 2` ⇒ both D-blocks execute). -/
theorem stepStmt_ifThen_true {cond : Op .bool []}
    {body : List Stmt} {s s' : BlockState}
    (hcond : evalOp cond s = some (Tile.scalar (Bool.true)))
    (hbody : stepStmts body s = some s') :
    stepStmt (.ifThen cond body) s = some s' := by
  simp only [stepStmt, hcond, Option.bind_some, Tile.scalar_data, if_true]
  exact hbody

/-- **`if cond { body }` step, false branch.** When the condition is scalar
`false`, the `ifThen` is a no-op (state unchanged). -/
theorem stepStmt_ifThen_false {cond : Op .bool []}
    {body : List Stmt} {s : BlockState}
    (hcond : evalOp cond s = some (Tile.scalar (Bool.false))) :
    stepStmt (.ifThen cond body) s = some s := by
  simp [stepStmt, hcond]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`start_l = tl.load(layout_ptr).to(tl.int32)` statement eval** (CSR loop
bound, pre-loop L_start): the data-dependent lower loop bound, an unmasked `.nat`
ptr-load of the precomputed `layout_ptr` scalar (= CSR row-pointer base). With
`layout_ptr` holding the pointer `(lpReg, lpOff)`, `start_l` reads
`layout_ptr_region[lpOff]`. The `forRangeDyn`'s `start` operand is `ref "start_l"`. -/
theorem bsa_startl_eval (s : BlockState) (lpReg : RegionName) (lpOff : Nat)
    (hlp : s.regs .ptr [] "layout_ptr" = some (Tile.scalar (lpReg, lpOff))) :
    evalOp (Op.load .nat (MemAccess.ptr (Op.ref .ptr [] "layout_ptr")) MaskOpt.none) s
      = some (Tile.scalar (s.readMemValue .nat lpReg lpOff)) := by
  simp only [evalOp, evalOp_ref, hlp, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.scalar, Tile.scalar_data_index, if_true, if_pos]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`end_l = tl.load(layout_ptr + 1).to(tl.int32)` statement eval** (CSR loop
bound, pre-loop L_end): the data-dependent upper loop bound, an unmasked `.nat`
ptr-load of `layout_ptr + 1` (the next CSR row-pointer). With `layout_ptr` holding
`(lpReg, lpOff)`, `end_l` reads `layout_ptr_region[lpOff + 1]`. The `forRangeDyn`'s
`stop` operand is `ref "end_l"`. -/
theorem bsa_endl_eval (s : BlockState) (lpReg : RegionName) (lpOff : Nat)
    (hlp : s.regs .ptr [] "layout_ptr" = some (Tile.scalar (lpReg, lpOff))) :
    evalOp (Op.load .nat
        (MemAccess.ptr (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "layout_ptr") (Op.constNat 1)))
        MaskOpt.none) s
      = some (Tile.scalar (s.readMemValue .nat lpReg (lpOff + 1))) := by
  simp only [evalOp, evalOp_ref, hlp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.ptrAdd_data, Tile.scalar, Tile.scalar_data_index, Broadcast.leftIndex,
    Broadcast.rightIndex, if_true, if_pos]

end BSARecipes

/-! ## Body split (test shape `BLOCK_M=BLOCK_N=BLOCK_D=16`, `NUM_D_BLOCKS=2`,
`EVEN_M=EVEN_N=true`, layout `(num_heads,num_kv_heads,total_seq_len)=(4,2,16)`,
strides `(qb,qh,qm)=(2048,512,32)`, `(kb,kh,kn)=(vb,vh,vn)=(1024,512,32)`,
`(ob,oh,om)=(2048,512,32)`, `num_layout=1`, row/col CSR strides `3`/`4`).

The lowered algorithm-layer kernel body is `bsaPreLoop ++ forRangeDyn "col_idx_idx"
(ref start_l) (ref end_l) 1 bsaLoopBody :: bsaPostLoop`, an exact transcription of
the elaborated `block_sparse_attention_kernel` AST at this shape. `bsa_body_split`
proves the identity by `rfl`. The loop body is `BSALoopBody`'s 26 algorithm-layer
statements documented in `section BSARecipes`. -/

/-- The 29 lowered pre-loop statements (static_print marker through `end_l`). -/
def bsaPreLoop (Out Q K V : RegionName) (R C : Region .nat) : List Stmt :=
  [ Stmt.ifThen (Op.constBool «false») [],
    Stmt.assign .nat [] "q_seq_len" (Op.constNat 16),
    Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_bh" (Op.programId 1),
    Stmt.assign .nat [] "off_h"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_bh") (Op.constNat 4)),
    Stmt.assign .nat [] "off_b"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_bh") (Op.constNat 4)),
    Stmt.assign .nat [] "head_groups"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat 4) (Op.constNat 2)),
    Stmt.assign .nat [] "off_h_kv"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h")
        (Op.ref .nat [] "head_groups")),
    Stmt.assign .ptr [] "Q"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase Q)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat 2048))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 512)))),
    Stmt.assign .ptr [] "K"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase K)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat 1024))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h_kv") (Op.constNat 512)))),
    Stmt.assign .ptr [] "V"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase V)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat 1024))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h_kv") (Op.constNat 512)))),
    Stmt.assign .nat [16] "offs_m"
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 16))
        (Op.arange 16)),
    Stmt.assign .nat [16] "offs_n" (Op.arange 16),
    Stmt.assign .nat [16] "offs_d" (Op.arange 16),
    Stmt.assign .nat [16, 16] "off_q"
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m")) (Op.constNat 32))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_d"))),
    Stmt.assign .nat [16, 16] "off_k"
      (Op.add NumericDType.nat Broadcast.nil.consR.consL
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_n")) (Op.constNat 32))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_d"))),
    Stmt.assign .nat [16, 16] "off_v"
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_n")) (Op.constNat 32))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_d"))),
    Stmt.assign .ptr [16, 16] "q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Q") (Op.ref .nat [16, 16] "off_q")),
    Stmt.assign .ptr [16, 16] "k_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "K") (Op.ref .nat [16, 16] "off_k")),
    Stmt.assign .ptr [16, 16] "v_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "V") (Op.ref .nat [16, 16] "off_v")),
    Stmt.assign .real [16] "m_i"
      (Op.add NumericDType.real Broadcast.scalarR (Op.full [16] (Op.const 0)) Op.negInf),
    Stmt.assign .real [16] "l_i" (Op.full [16] (Op.const 0)),
    Stmt.assign .real [16, 16] "acc" (Op.full [16, 16] (Op.const 0)),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.assign .real [16, 16] "acc2" (Op.full [16, 16] (Op.const 0))],
    Stmt.ifThenElse (Op.constBool «true»)
      [Stmt.assign .real [16, 16] "q"
          (Op.load .real (MemAccess.ptr (Op.ref .ptr [16, 16] "q_ptrs")) MaskOpt.none),
        Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
          [Stmt.assign .real [16, 16] "q2"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "q_ptrs") (Op.constNat 16)))
                MaskOpt.none)]]
      [Stmt.assign .real [16, 16] "q"
          (Op.load .real (MemAccess.ptr (Op.ref .ptr [16, 16] "q_ptrs"))
            (MaskOpt.mask
              (Op.remap [16, 16] Broadcast.nil.consL.consSame.leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m"))
                  (Op.ref .nat [] "q_seq_len"))))),
        Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
          [Stmt.assign .real [16, 16] "q2"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "q_ptrs") (Op.constNat 16)))
                (MaskOpt.mask
                  (Op.remap [16, 16] Broadcast.nil.consL.consSame.leftIndex
                    (Op.lt ComparableDType.nat Broadcast.scalarR
                      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m"))
                      (Op.ref .nat [] "q_seq_len")))))]],
    Stmt.assign .nat [] "layout_h"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 1)),
    Stmt.assign .ptr [] "layout_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase R)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "layout_h") (Op.constNat 3))
          (Op.ref .nat [] "start_m"))),
    Stmt.assign .nat [] "start_l"
      (Op.load .nat (MemAccess.ptr (Op.ref .ptr [] "layout_ptr")) MaskOpt.none),
    Stmt.assign .nat [] "end_l"
      (Op.load .nat
        (MemAccess.ptr (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "layout_ptr") (Op.constNat 1)))
        MaskOpt.none) ]

/-- The 26 lowered CSR-loop-body statements (see `section BSARecipes` for the
per-statement recipes). `EVEN_N = true` ⇒ unmasked K/V loads; `NUM_D_BLOCKS = 2`
⇒ both `ifThen (2 ≥ 2)` D-block branches are present. -/
def bsaLoopBody (C : Region .nat) : List Stmt :=
  [ Stmt.assign .nat [] "col_idx"
      (Op.load .nat
        (MemAccess.region C
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "layout_h") (Op.constNat 4))
            (Op.ref .nat [] "col_idx_idx")))
        MaskOpt.none),
    Stmt.assign .nat [] "start_n"
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "col_idx") (Op.constNat 16)),
    Stmt.ifThenElse (Op.constBool «true»)
      [Stmt.assign .real [16, 16] "k"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "k_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 32))))
            MaskOpt.none)]
      [Stmt.assign .real [16, 16] "k"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "k_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 32))))
            (MaskOpt.mask
              (Op.remap [16, 16] Broadcast.nil.consSame.consL.leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.add NumericDType.nat Broadcast.scalarR
                    (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_n"))
                    (Op.ref .nat [] "start_n"))
                  (Op.constNat 16)))))],
    Stmt.assign .real [16, 16] "qk" (Op.full [16, 16] (Op.const 0)),
    Stmt.assign .real [16, 16] "qk"
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [16, 16] "qk")
        (Op.dot (batch := []) (Op.ref .real [16, 16] "q") (Op.ref .real [16, 16] "k"))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.ifThenElse (Op.constBool «true»)
          [Stmt.assign .real [16, 16] "k"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "k_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 32)))
                    (Op.constNat 16)))
                MaskOpt.none)]
          [Stmt.assign .real [16, 16] "k"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "k_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 32)))
                    (Op.constNat 16)))
                (MaskOpt.mask
                  (Op.remap [16, 16] Broadcast.nil.consSame.consL.leftIndex
                    (Op.lt ComparableDType.nat Broadcast.scalarR
                      (Op.add NumericDType.nat Broadcast.scalarR
                        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_n"))
                        (Op.ref .nat [] "start_n"))
                      (Op.constNat 16)))))],
        Stmt.assign .real [16, 16] "qk"
          (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [16, 16] "qk")
            (Op.dot (batch := []) (Op.ref .real [16, 16] "q2") (Op.ref .real [16, 16] "k")))],
    Stmt.assign .real [16, 16] "qk"
      (Op.mul NumericDType.real Broadcast.scalarR (Op.ref .real [16, 16] "qk") (Op.const 1.0)),
    Stmt.assign .real [16, 16] "qk"
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [16, 16] "qk")
        ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m"))
              (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_n")))).where
          ((Op.const 0).broadcast [16, 16]) (Op.negInf.broadcast [16, 16]))),
    Stmt.assign .real [16] "m_ij"
      (Op.reduceMax ⟨1, by simp⟩ «false» (Op.ref .real [16, 16] "qk")),
    Stmt.assign .real [16, 16] "p"
      (Op.sub NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [16, 16] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [16] "m_ij"))).exp,
    Stmt.assign .real [16] "l_ij"
      (Op.reduceSum ⟨1, by simp⟩ «false» (Op.ref .real [16, 16] "p")),
    Stmt.assign .real [16] "m_i_new"
      ((Op.gt ComparableDType.real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [16] "m_i") (Op.ref .real [16] "m_ij")).where
        (Op.ref .real [16] "m_i") (Op.ref .real [16] "m_ij")),
    Stmt.assign .real [16] "alpha"
      (Op.sub NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [16] "m_i") (Op.ref .real [16] "m_i_new")).exp,
    Stmt.assign .real [16] "beta"
      (Op.sub NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [16] "m_ij") (Op.ref .real [16] "m_i_new")).exp,
    Stmt.assign .real [16] "l_i_new"
      (Op.add NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.mul NumericDType.real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [16] "alpha") (Op.ref .real [16] "l_i"))
        (Op.mul NumericDType.real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [16] "beta") (Op.ref .real [16] "l_ij"))),
    Stmt.assign .real [16] "p_scale"
      (Op.div NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [16] "beta") (Op.ref .real [16] "l_i_new")),
    Stmt.assign .real [16, 16] "p"
      (Op.mul NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [16, 16] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [16] "p_scale"))),
    Stmt.assign .real [16] "acc_scale"
      (Op.mul NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.div NumericDType.real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [16] "l_i") (Op.ref .real [16] "l_i_new"))
        (Op.ref .real [16] "alpha")),
    Stmt.assign .real [16, 16] "acc"
      (Op.mul NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [16, 16] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [16] "acc_scale"))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.assign .real [16, 16] "acc2"
          (Op.mul NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [16, 16] "acc2")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [16] "acc_scale")))],
    Stmt.assign .real [16, 16] "p" (Op.ref .real [16, 16] "p"),
    Stmt.ifThenElse (Op.constBool «true»)
      [Stmt.assign .real [16, 16] "v"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "v_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 32))))
            MaskOpt.none)]
      [Stmt.assign .real [16, 16] "v"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "v_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 32))))
            (MaskOpt.mask
              (Op.remap [16, 16] Broadcast.nil.consL.consSame.leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.add NumericDType.nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_n"))
                    (Op.ref .nat [] "start_n"))
                  (Op.constNat 16)))))],
    Stmt.assign .real [16, 16] "acc"
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [16, 16] "acc")
        (Op.dot (batch := []) (Op.ref .real [16, 16] "p") (Op.ref .real [16, 16] "v"))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.ifThenElse (Op.constBool «true»)
          [Stmt.assign .real [16, 16] "v"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "v_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 32)))
                    (Op.constNat 16)))
                MaskOpt.none)]
          [Stmt.assign .real [16, 16] "v"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "v_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat 32)))
                    (Op.constNat 16)))
                (MaskOpt.mask
                  (Op.remap [16, 16] Broadcast.nil.consL.consSame.leftIndex
                    (Op.lt ComparableDType.nat Broadcast.scalarR
                      (Op.add NumericDType.nat Broadcast.scalarR
                        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_n"))
                        (Op.ref .nat [] "start_n"))
                      (Op.constNat 16)))))],
        Stmt.assign .real [16, 16] "acc2"
          (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [16, 16] "acc2")
            (Op.dot (batch := []) (Op.ref .real [16, 16] "p") (Op.ref .real [16, 16] "v")))],
    Stmt.assign .real [16] "l_i" (Op.ref .real [16] "l_i_new"),
    Stmt.assign .real [16] "m_i" (Op.ref .real [16] "m_i_new") ]

/-- The 4 lowered post-loop statements (`off_o` through the two masked `out`
stores; the second store is `ifThen (2 ≥ 2)`-gated at `+BLOCK_D`). -/
def bsaPostLoop (Out : RegionName) : List Stmt :=
  [ Stmt.assign .nat [16, 16] "off_o"
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat 2048))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 512)))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m")) (Op.constNat 32)))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_d"))),
    Stmt.assign .ptr [16, 16] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [16, 16] "off_o")),
    Stmt.store .real [16, 16] (MemAccess.ptr (Op.ref .ptr [16, 16] "out_ptrs"))
      (Op.ref .real [16, 16] "acc")
      (MaskOpt.mask
        (Op.remap [16, 16] Broadcast.nil.consL.consSame.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m"))
            (Op.ref .nat [] "q_seq_len")))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.store .real [16, 16]
          (MemAccess.ptr
            (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "out_ptrs") (Op.constNat 16)))
          (Op.ref .real [16, 16] "acc2")
          (MaskOpt.mask
            (Op.remap [16, 16] Broadcast.nil.consL.consSame.leftIndex
              (Op.lt ComparableDType.nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m"))
                (Op.ref .nat [] "q_seq_len"))))] ]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- The lowered test-shape `block_sparse_attention_kernel` body splits as
`bsaPreLoop ++ forRangeDyn "col_idx_idx" (ref start_l) (ref end_l) 1 bsaLoopBody ::
bsaPostLoop`. Pure `List Stmt` identity on the transcription (`rfl`). -/
theorem bsa_body_split
    (Out Q K V : RegionName) (R C : Region .nat) :
    (block_sparse_attention_kernel Out Q K V R C
      3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
      4 2 16 16 16 16 2 Bool.true Bool.true).toAlgKernel.body
      = bsaPreLoop Out Q K V R C
        ++ (Stmt.forRangeDyn "col_idx_idx" (Op.ref .nat [] "start_l") (Op.ref .nat [] "end_l")
              (Op.constNat 1) (bsaLoopBody C)
            :: bsaPostLoop Out) := by
  rfl

/-! ## CSR loop invariant + pre-loop entry

`bsaInvariant … c s` states that, after streaming `c` `BLOCK_N`-blocks of the
CSR-selected key stream, the live registers `m_i`/`l_i`/`acc`/`acc2` hold the
⊥-seeded online-softmax accumulators (`bsaMPartial`/`bsaLPartial`/`bsaOPartial`,
the latter with two value tiles `Vg`/`Vg2` for the two D-blocks) over the gathered
stream, and every pre-loop-seeded register is preserved. The gathered streaming
data (`qStart`/`numKVBlocks`/`gpos`/`Q`/`Kg`/`Vg`/`Vg2`/`scale`) is fixed by the
program (independent of `c`), exactly as `srInvariant` parametrizes by `srQkF`/etc.;
the in-loop `p_scale`/`acc_scale` collapse into this running state. -/

open BSAMathCausal in
/-- The CSR loop invariant after `c` selected `BLOCK_N`-blocks (test shape:
`BLOCK_M = BLOCK_N = BLOCK_D = 16`). -/
noncomputable def bsaInvariant
    (Out Q K V : RegionName) (R C : Region .nat)
    (qStart numKVBlocks : Nat) (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 32] → ℝ)
    (Kg : TileIndex [16 * numKVBlocks, 32] → ℝ)
    (Vg Vg2 : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (s0 : BlockState) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "q_seq_len" = some (Tile.scalar 16)
  ∧ s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "off_bh" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "off_h" = some (Tile.scalar (s0.pids 1 % 4))
  ∧ s.regs .nat [] "off_b" = some (Tile.scalar (s0.pids 1 / 4))
  ∧ s.regs .nat [] "head_groups" = some (Tile.scalar 2)
  ∧ s.regs .nat [] "off_h_kv" = some (Tile.scalar (s0.pids 1 % 4 / 2))
  ∧ s.regs .nat [16] "offs_m" =
      some (Tile.vec (fun i : Fin 16 => s0.pids 0 * 16 + i.val))
  ∧ s.regs .nat [16] "offs_n" = some (Tile.vec (fun j : Fin 16 => j.val))
  ∧ s.regs .nat [16] "offs_d" = some (Tile.vec (fun e : Fin 16 => e.val))
  ∧ s.regs .nat [] "layout_h" = some (Tile.scalar (s0.pids 1 % 4 % 1))
  ∧ s.regs .real [16] "m_i" =
      some (Tile.vec (fun i : Fin 16 =>
        bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale c i))
  ∧ s.regs .real [16] "l_i" =
      some (Tile.vec (fun i : Fin 16 =>
        (some (bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale c i) : WithBot ℝ)))
  ∧ s.regs .real [16, 16] "acc" =
      some (⟨fun idx : TileIndex [16, 16] =>
        (some (bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg scale c idx /
          bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale c idx.1) : WithBot ℝ)⟩
          : Tile .real [16, 16])
  ∧ s.regs .real [16, 16] "acc2" =
      some (⟨fun idx : TileIndex [16, 16] =>
        (some (bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg2 scale c idx /
          bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale c idx.1) : WithBot ℝ)⟩
          : Tile .real [16, 16])
  ∧ s.regs .real [16, 16] "q" =
      some (⟨fun idx : TileIndex [16, 16] =>
        s0.readMemValue .real Q ((s0.pids 1 / 4 * 2048 + s0.pids 1 % 4 * 512) +
          ((s0.pids 0 * 16 + idx.1.val) * 32 + idx.2.1.val))⟩ : Tile .real [16, 16])
  ∧ s.regs .real [16, 16] "q2" =
      some (⟨fun idx : TileIndex [16, 16] =>
        s0.readMemValue .real Q ((s0.pids 1 / 4 * 2048 + s0.pids 1 % 4 * 512) +
          ((s0.pids 0 * 16 + idx.1.val) * 32 + idx.2.1.val) + 16)⟩ : Tile .real [16, 16])
  ∧ s.regs .ptr [16, 16] "k_ptrs" =
      some (⟨fun idx : TileIndex [16, 16] =>
        (K, (s0.pids 1 / 4 * 1024 + s0.pids 1 % 4 / 2 * 512) +
          (idx.2.1.val * 32 + idx.1.val))⟩ : Tile .ptr [16, 16])
  ∧ s.regs .ptr [16, 16] "v_ptrs" =
      some (⟨fun idx : TileIndex [16, 16] =>
        (V, (s0.pids 1 / 4 * 1024 + s0.pids 1 % 4 / 2 * 512) +
          (idx.1.val * 32 + idx.2.1.val))⟩ : Tile .ptr [16, 16])

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **Pre-loop execution.** The 29 deterministic pre-loop statements step a clean
input state (`undef = 0`) to a state satisfying `bsaInvariant … 0` — the loop-entry
base case (`m_i = ⊥`, `l_i = 0`, `acc = acc2 = 0`, via the `bsaMPartial`/
`bsaLPartial`/`bsaOPartial` zero recurrences). The streaming data is arbitrary;
at `c = 0` the accumulators are data-independent. -/
theorem bsaPreLoop_eval
    (s : BlockState) (Out Q K V : RegionName) (R C : Region .nat)
    (qStart numKVBlocks : Nat) (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 32] → ℝ)
    (Kg : TileIndex [16 * numKVBlocks, 32] → ℝ)
    (Vg Vg2 : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (bsaPreLoop Out Q K V R C) s = some s0
      ∧ bsaInvariant Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s 0 s0
      ∧ s0.regs .nat [] "start_l" =
          some (Tile.scalar (s.readMemValue .nat R.cast (s.pids 1 % 4 % 1 * 3 + s.pids 0)))
      ∧ s0.regs .nat [] "end_l" =
          some (Tile.scalar (s.readMemValue .nat R.cast (s.pids 1 % 4 % 1 * 3 + s.pids 0 + 1))) := by
  unfold bsaPreLoop
  -- stmt 0: static_print marker `ifThen false []` is a no-op
  rw [stepStmts.cons_some (stepStmt_ifThen_false (by simp [evalOp]))]
  -- stmt 1: q_seq_len = 16
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.constNat 16) s = some (Tile.scalar 16) from by simp))]
  -- stmt 2: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 _))]
  -- stmt 3: off_bh = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 4: off_h = off_bh % 4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_bh") (Op.constNat 4)) _
        = some (Tile.scalar (s.pids 1 % 4)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.mod]))]
  -- stmt 5: off_b = off_bh // 4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_bh") (Op.constNat 4)) _
        = some (Tile.scalar (s.pids 1 / 4)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.floorDiv]))]
  -- stmt 6: head_groups = 4 // 2
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat 4) (Op.constNat 2)) _
        = some (Tile.scalar 2) from by
      simp only [evalOp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.floorDiv]))]
  -- stmt 7: off_h_kv = off_h // head_groups
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h")
          (Op.ref .nat [] "head_groups")) _
        = some (Tile.scalar (s.pids 1 % 4 / 2)) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.floorDiv]))]
  -- stmt 8: Q += off_b*2048 + off_h*512
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Q)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat 2048))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 512)))) _
        = some (Tile.scalar (Q, s.pids 1 / 4 * 2048 + s.pids 1 % 4 * 512)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.scalar_data_index,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
        NumericDType.mul, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]))]
  -- stmt 9: K += off_b*1024 + off_h_kv*512
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase K)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat 1024))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h_kv") (Op.constNat 512)))) _
        = some (Tile.scalar (K, s.pids 1 / 4 * 1024 + s.pids 1 % 4 / 2 * 512)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.scalar_data_index,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
        NumericDType.mul, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]))]
  -- stmt 10: V += off_b*1024 + off_h_kv*512
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase V)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat 1024))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h_kv") (Op.constNat 512)))) _
        = some (Tile.scalar (V, s.pids 1 / 4 * 1024 + s.pids 1 % 4 / 2 * 512)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.scalar_data_index,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
        NumericDType.mul, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]))]
  -- stmt 11: offs_m = start_m*16 + arange 16
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 16))
          (Op.arange 16)) _
        = some (Tile.vec (fun i : Fin 16 => s.pids 0 * 16 + i.val)) from by
      simp only [evalOp_add, evalOp_mul, evalOp_arange, evalOp_ref, evalOp_constNat,
        BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.vec, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 12: offs_n = arange 16
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 16) _ = some (Tile.vec (fun j : Fin 16 => j.val)) from evalOp_arange 16 _))]
  -- stmt 13: offs_d = arange 16
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange 16) _ = some (Tile.vec (fun e : Fin 16 => e.val)) from evalOp_arange 16 _))]
  -- stmt 14: off_q (irrelevant scalar value)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil.consL.consR
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m")) (Op.constNat 32))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_d"))) _
        = some (⟨fun idx : TileIndex [16, 16] =>
            (s.pids 0 * 16 + idx.1.val) * 32 + idx.2.1.val⟩ : Tile .nat [16, 16]) from by
      rw [evalOp_add, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs .nat [16] ⟨1, by simp⟩ "offs_m" _
            (Tile.vec (fun i : Fin 16 => s.pids 0 * 16 + i.val)) (by simp [Tile.vec]),
        evalOp_expandDim_ref_of_regs .nat [16] ⟨0, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin 16 => e.val)) (by simp [Tile.vec])]
      simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 15: off_k = offs_n[None,:]*32 + offs_d[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil.consR.consL
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_n")) (Op.constNat 32))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_d"))) _
        = some (⟨fun idx : TileIndex [16, 16] =>
            idx.2.1.val * 32 + idx.1.val⟩ : Tile .nat [16, 16]) from by
      rw [evalOp_add, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs .nat [16] ⟨0, by simp⟩ "offs_n" _
            (Tile.vec (fun j : Fin 16 => j.val)) (by simp [Tile.vec]),
        evalOp_expandDim_ref_of_regs .nat [16] ⟨1, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin 16 => e.val)) (by simp [Tile.vec])]
      simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 16: off_v = offs_n[:,None]*32 + offs_d[None,:]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil.consL.consR
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_n")) (Op.constNat 32))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_d"))) _
        = some (⟨fun idx : TileIndex [16, 16] =>
            idx.1.val * 32 + idx.2.1.val⟩ : Tile .nat [16, 16]) from by
      rw [evalOp_add, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs .nat [16] ⟨1, by simp⟩ "offs_n" _
            (Tile.vec (fun j : Fin 16 => j.val)) (by simp [Tile.vec]),
        evalOp_expandDim_ref_of_regs .nat [16] ⟨0, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin 16 => e.val)) (by simp [Tile.vec])]
      simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 17: q_ptrs = Q + off_q (irrelevant)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Q") (Op.ref .nat [16, 16] "off_q")) _
        = some (⟨fun idx : TileIndex [16, 16] =>
            (Q, (s.pids 1 / 4 * 2048 + s.pids 1 % 4 * 512) +
              ((s.pids 0 * 16 + idx.1.val) * 32 + idx.2.1.val))⟩ : Tile .ptr [16, 16]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_)); rfl))]
  -- stmt 18: k_ptrs = K + off_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "K") (Op.ref .nat [16, 16] "off_k")) _
        = some (⟨fun idx : TileIndex [16, 16] =>
            (K, (s.pids 1 / 4 * 1024 + s.pids 1 % 4 / 2 * 512) +
              (idx.2.1.val * 32 + idx.1.val))⟩ : Tile .ptr [16, 16]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_)); rfl))]
  -- stmt 19: v_ptrs = V + off_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "V") (Op.ref .nat [16, 16] "off_v")) _
        = some (⟨fun idx : TileIndex [16, 16] =>
            (V, (s.pids 1 / 4 * 1024 + s.pids 1 % 4 / 2 * 512) +
              (idx.1.val * 32 + idx.2.1.val))⟩ : Tile .ptr [16, 16]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_)); rfl))]
  -- stmt 20: m_i = zeros[16] - inf  (= ⊥-seed)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.real Broadcast.scalarR (Op.full [16] (Op.const 0)) Op.negInf) _
        = some (Tile.vec (fun _ : Fin 16 => (⊥ : WithBot ℝ))) from by
      simp only [evalOp_add, evalOp_full, evalOp_const, evalOp_negInf,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, Tile.scalar_data_index]
      rfl))]
  -- stmt 21: l_i = zeros[16]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [16] (Op.const 0)) _
        = some (Tile.vec (fun _ : Fin 16 => (some (0 : ℝ) : WithBot ℝ))) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx; simp [Tile.vec, Tile.scalar_data_index]))]
  -- stmt 22: acc = zeros[16,16]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [16, 16] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [16, 16] => (some (0 : ℝ) : WithBot ℝ)⟩ : Tile .real [16, 16]) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx; simp [Tile.scalar_data_index]))]
  -- stmt 23: ifThen (2 ≥ 2) { acc2 = zeros }
  rw [stepStmts.cons_some (stepStmt_ifThen_true (by simp [evalOp])
    (show stepStmts [Stmt.assign .real [16, 16] "acc2" (Op.full [16, 16] (Op.const 0))] _
        = some _ from by
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.full [16, 16] (Op.const 0)) _
            = some (⟨fun _ : TileIndex [16, 16] => (some (0 : ℝ) : WithBot ℝ)⟩ : Tile .real [16, 16]) from by
          simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
          refine congrArg some ?_; ext idx; simp [Tile.scalar_data_index]))]
      rw [stepStmts.nil]))]
  -- stmt 24: ifThenElse true { q = load q_ptrs ; ifThen { q2 = load } }
  rw [stepStmts.cons_some (stepStmt_ifThenElse_true (by simp [evalOp])
    (show stepStmts _ _ = some _ from by
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [16, 16] "q_ptrs")) MaskOpt.none) _
            = some (⟨fun idx : TileIndex [16, 16] =>
                s.readMemValue .real Q ((s.pids 1 / 4 * 2048 + s.pids 1 % 4 * 512) +
                  ((s.pids 0 * 16 + idx.1.val) * 32 + idx.2.1.val))⟩ : Tile .real [16, 16]) from by
          simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
            Option.bind_eq_bind, Option.bind_some]
          refine congrArg some ?_; ext idx; rfl))]
      rw [stepStmts.cons_some (stepStmt_ifThen_true (by simp [evalOp])
        (show stepStmts _ _ = some _ from by
          rw [stepStmts.cons_some (stepStmt_assign_eq_some
            (show evalOp (Op.load .real
                  (MemAccess.ptr
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "q_ptrs") (Op.constNat 16)))
                  MaskOpt.none) _
                = some (⟨fun idx : TileIndex [16, 16] =>
                    s.readMemValue .real Q ((s.pids 1 / 4 * 2048 + s.pids 1 % 4 * 512) +
                      ((s.pids 0 * 16 + idx.1.val) * 32 + idx.2.1.val) + 16)⟩
                      : Tile .real [16, 16]) from by
              simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
                BlockState.setReg_ne_name, Option.bind_eq_bind, Option.bind_some]
              refine congrArg some ?_; ext idx
              simp [Tile.ptrAdd_data, Tile.scalar_data_index,
                Broadcast.leftIndex, Broadcast.rightIndex]))]
          rw [stepStmts.nil]))]
      rw [stepStmts.nil]))]
  -- stmt 25..: layout_h = off_h % 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 1)) _
        = some (Tile.scalar (s.pids 1 % 4 % 1)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.mod]))]
  -- stmt 26: layout_ptr = R + (layout_h*3 + start_m)  (irrelevant)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase R)
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "layout_h") (Op.constNat 3))
            (Op.ref .nat [] "start_m"))) _
        = some (Tile.scalar ((R.cast : RegionName), s.pids 1 % 4 % 1 * 3 + s.pids 0)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.scalar_data_index,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
        NumericDType.mul, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]))]
  -- stmt 27: start_l = load layout_ptr (via banked recipe `bsa_startl_eval`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_startl_eval _ R.cast (s.pids 1 % 4 % 1 * 3 + s.pids 0)
      (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true])))]
  -- stmt 28: end_l = load (layout_ptr + 1) (via banked recipe `bsa_endl_eval`)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_endl_eval _ R.cast (s.pids 1 % 4 % 1 * 3 + s.pids 0)
      (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_⟩
  · unfold bsaInvariant
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp only [BlockState.setReg_pids]
    · funext rg o; simp only [BlockState.setReg_mem]
    · intro rg o; simp [hundef]
    all_goals
      simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true]
    all_goals
      first
      | rfl
      | -- acc / acc2 at c = 0: full-zero tile = `some (0/0)` pointwise
        (refine congrArg some (Tile.ext (fun idx => ?_));
         refine congrArg some ?_;
         rw [BSAMathCausal.bsaOPartial, BSAMathCausal.bsaLPartial, zero_div])
  · -- start_l register value (innermost-but-one setReg); mem preserved by setReg
    rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), BlockState.setReg_same]
    refine congrArg some (congrArg Tile.scalar ?_)
    simp only [BlockState.readMemValue, BlockState.readMemTyped, BlockState.setReg_mem]
  · -- end_l register value (innermost setReg); mem preserved by setReg
    rw [BlockState.setReg_same]
    refine congrArg some (congrArg Tile.scalar ?_)
    simp only [BlockState.readMemValue, BlockState.readMemTyped, BlockState.setReg_mem]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open BSAMathCausal in
/-- **CSR loop driver.** From the loop-entry invariant (`bsaInvariant … 0`) and the
per-block gather bridges, `forRangeDyn "col_idx_idx" start_l end_l 1 bsaLoopBody`
runs to a final state at counter `final ≥ end_l` satisfying `bsaInvariant …
(end_l − start_l)`. The bridges at absolute CSR index `i` are supplied for every
`i ∈ [start_l, end_l)`; the block count there is `c = i − start_l`. -/
theorem bsa_csr_loop
    (Out Q K V : RegionName) (R C : Region .nat)
    (qStart numKVBlocks : Nat) (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 32] → ℝ)
    (Kg : TileIndex [16 * numKVBlocks, 32] → ℝ)
    (Vg Vg2 : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (s0 : BlockState) (start_l end_l : Nat)
    (hbound : end_l - start_l = numKVBlocks) (hsle : start_l ≤ end_l)
    (sEntry : BlockState)
    (hStartOp : evalOp (Op.ref .nat [] "start_l") sEntry = some (Tile.scalar start_l))
    (hStopOp : evalOp (Op.ref .nat [] "end_l") sEntry = some (Tile.scalar end_l))
    (hInit : bsaInvariant Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s0 0 sEntry)
    (hstep : ∀ (i : Nat) (st : BlockState), start_l ≤ i → i < end_l →
      bsaInvariant Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s0 (i - start_l) st →
      ∃ st', stepStmts (bsaLoopBody C) (st.setReg "col_idx_idx" .nat [] (Tile.scalar i)) = some st'
        ∧ bsaInvariant Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s0
            (i - start_l + 1) st') :
    ∃ final sFinal,
      stepStmt (Stmt.forRangeDyn "col_idx_idx" (Op.ref .nat [] "start_l") (Op.ref .nat [] "end_l")
        (Op.constNat 1) (bsaLoopBody C)) sEntry = some sFinal
      ∧ end_l ≤ final
      ∧ bsaInvariant Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s0 numKVBlocks sFinal := by
  obtain ⟨final, sFinal, hExec, hfin, hP⟩ :=
    forRangeDyn_inv (idx := "col_idx_idx")
      (P := fun i st => bsaInvariant Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s0
        (i - start_l) st ∧ start_l ≤ i ∧ i ≤ end_l)
      (start := start_l) (stop := end_l) (step := 1)
      hStartOp hStopOp (evalOp_constNat 1 sEntry) (by norm_num)
      ⟨by rw [Nat.sub_self]; exact hInit, le_refl _, by omega⟩
      (fun i st hlt hPi => by
        obtain ⟨hinv, hge, hle⟩ := hPi
        obtain ⟨st', hbody, hinv'⟩ := hstep i st hge hlt hinv
        refine ⟨st', hbody, ?_, ?_, ?_⟩
        · have : i + 1 - start_l = (i - start_l) + 1 := by omega
          rw [this]; exact hinv'
        · omega
        · omega)
  refine ⟨final, sFinal, hExec, hfin, ?_⟩
  obtain ⟨hinv, _, hfle⟩ := hP
  have hfeq : final = end_l := le_antisymm hfle hfin
  have hsub : final - start_l = numKVBlocks := by rw [hfeq]; exact hbound
  rw [hsub] at hinv
  exact hinv

/-- A prop-masked `writeMem` scatter preserves any target offset `o` not hit by
the (mask-active) offset function. -/
private theorem bsa_foldl_writeMem_preserves_off {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (o : Nat) (l : List α) (s : BlockState)
    (h : ∀ k ∈ l, P k → offsetFn k ≠ o) :
    ((l.foldl
        (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem
          region o)
      = s.readMem region o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
    rw [List.foldl_cons]
    have htl : ∀ k ∈ tl, P k → offsetFn k ≠ o := fun k hk hPk => h k (List.mem_cons_of_mem hd hk) hPk
    by_cases hP : P hd
    · simp only [hP, if_true]
      rw [ih _ htl, BlockState.writeMem_readMem]
      rw [if_neg]
      rintro ⟨_, h_eq⟩
      exact h hd List.mem_cons_self hP h_eq.symm
    · simp only [hP, if_false]
      exact ih _ htl

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open BSAMathCausal in
/-- **PostLoop execution + accumulator readback.** Stepping the 4 post-loop
statements (`off_o`, `out_ptrs`, masked store of `acc`, masked store of `acc2` at
`+16`) from a loop-end state satisfying `bsaInvariant … numKVBlocks`, the two
masked `out` stores write the running `acc`/`acc2` (= `bsaOPartial / bsaLPartial`
at the full window) at every active lane and preserve out-of-bounds lanes. BSA does
*not* normalize a final `acc /= l_i`; the accumulators are already normalized
in-loop. The two stores hit disjoint output channels (`outOffset` vs
`out2Offset = outOffset + 16`), so each readback is independent. -/
theorem bsaPostLoop_eval
    (Out Q K V : RegionName) (R C : Region .nat)
    (qStart numKVBlocks : Nat) (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 32] → ℝ)
    (Kg : TileIndex [16 * numKVBlocks, 32] → ℝ)
    (Vg Vg2 : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (s0 : BlockState) (s : BlockState)
    (hinv : bsaInvariant Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s0 numKVBlocks s) :
    ∃ sP, stepStmts (bsaPostLoop Out) s = some sP
      ∧ (∀ idx : TileIndex [16, 16],
          sP.readMem Out (outOffset s0 4 2048 512 32 16 idx)
            = if s0.pids 0 * 16 + idx.1.val < 16 then
                (bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg scale numKVBlocks idx /
                  bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1)
              else s.readMem Out (outOffset s0 4 2048 512 32 16 idx))
      ∧ (∀ idx : TileIndex [16, 16],
          sP.readMem Out (out2Offset s0 4 2048 512 32 16 16 idx)
            = if s0.pids 0 * 16 + idx.1.val < 16 then
                (bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg2 scale numKVBlocks idx /
                  bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1)
              else s.readMem Out (out2Offset s0 4 2048 512 32 16 16 idx)) := by
  obtain ⟨hpids, hmem, hundef, hqsl, hsm, hbh, hoh, hob, hhg, hohkv, hom, hon, hod,
    hlh, hmi, hli, hacc, hacc2, hq, hq2, hkp, hvp⟩ := hinv
  -- abbreviations for the two written value functions
  set accFn : TileIndex [16, 16] → ℝ := fun idx =>
    bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg scale numKVBlocks idx /
      bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1 with haccFn
  set acc2Fn : TileIndex [16, 16] → ℝ := fun idx =>
    bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg2 scale numKVBlocks idx /
      bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1 with hacc2Fn
  set offFn : TileIndex [16, 16] → Nat := fun idx =>
    s0.pids 1 / 4 * 2048 + s0.pids 1 % 4 * 512 + (s0.pids 0 * 16 + idx.1.val) * 32 + idx.2.1.val with hoffFn
  unfold bsaPostLoop
  -- stmt 0: off_o = off_b*2048 + off_h*512 + offs_m*32 + offs_d
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat 2048))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat 512)))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m")) (Op.constNat 32)))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [16] "offs_d"))) s
        = some (⟨fun idx : TileIndex [16, 16] => offFn idx⟩ : Tile .nat [16, 16]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs .nat [16] ⟨1, by simp⟩ "offs_m" _
            (Tile.vec (fun i : Fin 16 => s0.pids 0 * 16 + i.val)) hom,
        evalOp_expandDim_ref_of_regs .nat [16] ⟨0, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin 16 => e.val)) hod]
      simp only [evalOp_ref, hob, hoh, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [hoffFn, Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
        TileShape.dropInsertedIndex]
      try ring))]
  set s1 := s.setReg "off_o" .nat [16, 16] (⟨fun idx : TileIndex [16, 16] => offFn idx⟩ : Tile .nat [16, 16]) with hs1d
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "off_o" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1offo : s1.regs .nat [16, 16] "off_o" = some
      (⟨fun idx : TileIndex [16, 16] => offFn idx⟩ : Tile .nat [16, 16]) := by
    rw [hs1d, BlockState.setReg_same]
  -- stmt 1: out_ptrs = Out + off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [16, 16] "off_o")) s1
        = some (⟨fun idx : TileIndex [16, 16] => (Out, offFn idx)⟩ : Tile .ptr [16, 16]) from by
      simp only [evalOp, evalOp_ref, hs1offo, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨ir, dd, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  set s2 := s1.setReg "out_ptrs" .ptr [16, 16]
    (⟨fun idx : TileIndex [16, 16] => (Out, offFn idx)⟩ : Tile .ptr [16, 16]) with hs2d
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "out_ptrs" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2ptr : s2.regs .ptr [16, 16] "out_ptrs" = some
      (⟨fun idx : TileIndex [16, 16] => (Out, offFn idx)⟩ : Tile .ptr [16, 16]) := by
    rw [hs2d, BlockState.setReg_same]
  have hs2acc : s2.regs .real [16, 16] "acc" = some
      (⟨fun idx : TileIndex [16, 16] => (some (accFn idx) : WithBot ℝ)⟩ : Tile .real [16, 16]) :=
    e2 (by decide) (e1 (by decide) hacc)
  have hs2acc2 : s2.regs .real [16, 16] "acc2" = some
      (⟨fun idx : TileIndex [16, 16] => (some (acc2Fn idx) : WithBot ℝ)⟩ : Tile .real [16, 16]) :=
    e2 (by decide) (e1 (by decide) hacc2)
  have hs2m : s2.regs .nat [16] "offs_m" = some (Tile.vec (fun i : Fin 16 => s0.pids 0 * 16 + i.val)) :=
    e2 (by decide) (e1 (by decide) hom)
  have hs2qsl : s2.regs .nat [] "q_seq_len" = some (Tile.scalar 16) :=
    e2 (by decide) (e1 (by decide) hqsl)
  have hs2mem : s2.mem = s0.mem := by
    funext rg o; rw [hs2d, BlockState.setReg_mem, hs1d, BlockState.setReg_mem]; exact hmem ▸ rfl
  have hs2pids : s2.pids = s0.pids := by
    rw [hs2d, BlockState.setReg_pids, hs1d, BlockState.setReg_pids]; exact hpids
  -- mask op evaluation shared by both stores
  have hexpM : @evalOp .nat [16, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m")) s2
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin 16 => s0.pids 0 * 16 + i.val))) :=
    evalOp_expandDim_ref_of_regs .nat [16] ⟨1, by simp⟩ "offs_m" s2 _ hs2m
  -- stmt 2: masked store of acc at off_o
  have hstore1 : stepStmt (Stmt.store .real [16, 16] (MemAccess.ptr (Op.ref .ptr [16, 16] "out_ptrs"))
      (Op.ref .real [16, 16] "acc")
      (MaskOpt.mask (Op.remap [16, 16] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m"))
          (Op.ref .nat [] "q_seq_len"))))) s2
      = some ((TileShape.allIndices [16, 16]).foldl
          (fun acc idx =>
            if s0.pids 0 * 16 + idx.1.val < 16 then
              acc.writeMem Out (offFn idx) (accFn idx)
            else acc) s2) := by
    unfold stepStmt
    simp only [evalOp_ref, hs2acc, hs2ptr, hs2qsl, evalOp, hexpM, Option.bind, Option.map]
    refine congrArg some ?_
    congr 1
    funext acc idx
    obtain ⟨ir, dd, u⟩ := idx
    simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim, Tile.vec,
      Tile.scalar, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.dropInsertedIndex, BlockState.writeMemTyped_real, FloatDType.real_storeValue,
      decide_eq_true_eq]
    simp only [show (WithBot.unbotD 0 (some (accFn (ir, dd, u))) : ℝ) = accFn (ir, dd, u) from rfl]
  rw [stepStmts.cons_some hstore1]
  set s3 := (TileShape.allIndices [16, 16]).foldl
      (fun acc idx => if s0.pids 0 * 16 + idx.1.val < 16 then
        acc.writeMem Out (offFn idx) (accFn idx) else acc) s2 with hs3d
  -- s3 register/mem facts: scatter only touches Out memory; regs preserved
  have hs3ptr : s3.regs .ptr [16, 16] "out_ptrs" = some
      (⟨fun idx : TileIndex [16, 16] => (Out, offFn idx)⟩ : Tile .ptr [16, 16]) := by
    rw [hs3d, BlockState.foldl_writeMem_prop_masked_regs]; exact hs2ptr
  have hs3acc2 : s3.regs .real [16, 16] "acc2" = some
      (⟨fun idx : TileIndex [16, 16] => (some (acc2Fn idx) : WithBot ℝ)⟩ : Tile .real [16, 16]) := by
    rw [hs3d, BlockState.foldl_writeMem_prop_masked_regs]; exact hs2acc2
  have hs3m : s3.regs .nat [16] "offs_m" = some (Tile.vec (fun i : Fin 16 => s0.pids 0 * 16 + i.val)) := by
    rw [hs3d, BlockState.foldl_writeMem_prop_masked_regs]; exact hs2m
  have hs3qsl : s3.regs .nat [] "q_seq_len" = some (Tile.scalar 16) := by
    rw [hs3d, BlockState.foldl_writeMem_prop_masked_regs]; exact hs2qsl
  have hexpM3 : @evalOp .nat [16, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m")) s3
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin 16 => s0.pids 0 * 16 + i.val))) :=
    evalOp_expandDim_ref_of_regs .nat [16] ⟨1, by simp⟩ "offs_m" s3 _ hs3m
  -- inner store of acc2 at out2 offset (off_o + 16)
  have hstore2inner : stepStmt (Stmt.store .real [16, 16]
      (MemAccess.ptr (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [16, 16] "out_ptrs") (Op.constNat 16)))
      (Op.ref .real [16, 16] "acc2")
      (MaskOpt.mask (Op.remap [16, 16] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [16] "offs_m"))
          (Op.ref .nat [] "q_seq_len"))))) s3
      = some ((TileShape.allIndices [16, 16]).foldl
          (fun acc idx =>
            if s0.pids 0 * 16 + idx.1.val < 16 then
              acc.writeMem Out (offFn idx + 16) (acc2Fn idx)
            else acc) s3) := by
    unfold stepStmt
    simp only [evalOp_ref, hs3acc2, hs3ptr, hs3qsl, evalOp, hexpM3, Option.bind, Option.map]
    refine congrArg some ?_
    congr 1
    funext acc idx
    obtain ⟨ir, dd, u⟩ := idx
    simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim, Tile.vec,
      Tile.scalar, Tile.ptrAdd_data, Tile.scalar_data, ComparableDType.lt, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex, BlockState.writeMemTyped_real,
      FloatDType.real_storeValue, decide_eq_true_eq]
    simp only [show (WithBot.unbotD 0 (some (acc2Fn (ir, dd, u))) : ℝ) = acc2Fn (ir, dd, u) from rfl]
  -- stmt 3: ifThen (2 ≥ 2) [store acc2 at off_o + 16]
  rw [stepStmts.cons_some (stepStmt_ifThen_true (by simp [evalOp])
    (by rw [stepStmts.cons_some hstore2inner, stepStmts.nil])), stepStmts.nil]
  set s4 := (TileShape.allIndices [16, 16]).foldl
      (fun acc idx => if s0.pids 0 * 16 + idx.1.val < 16 then
        acc.writeMem Out (offFn idx + 16) (acc2Fn idx) else acc) s3 with hs4d
  refine ⟨s4, rfl, ?_, ?_⟩
  · -- first store readback at outOffset
    intro idx
    -- offFn = outOffset
    have hoeq : outOffset s0 4 2048 512 32 16 idx = offFn idx := by
      simp only [outOffset, offB, offH, mIndex, dIndex, hoffFn]; try ring
    -- injectivity of offFn
    have hinj : Function.Injective offFn := by
      rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
      simp only [hoffFn] at h
      have hm : ma = mb := by omega
      have hd2 : da = db := by omega
      subst hm; subst hd2; rfl
    rw [hoeq]
    -- s4 = second scatter (at offFn+16) over s3; it doesn't touch offFn lanes
    have hs4read : s4.readMem Out (offFn idx) = s3.readMem Out (offFn idx) := by
      rw [hs4d]
      apply bsa_foldl_writeMem_preserves_off
      intro k _hk _hPk
      simp only [hoffFn]; omega
    rw [hs4read, hs3d]
    rw [BlockState.scatter_readback_prop_masked_nd s2 offFn accFn
      (fun idx : TileIndex [16, 16] => s0.pids 0 * 16 + idx.1.val < 16) hinj idx]
    by_cases hlt : s0.pids 0 * 16 + idx.1.val < 16
    · rw [if_pos hlt, if_pos hlt]
    · rw [if_neg hlt, if_neg hlt]
      simp only [BlockState.readMem, hs2mem, hmem]
  · -- second store readback at out2Offset
    intro idx
    have ho2eq : out2Offset s0 4 2048 512 32 16 16 idx = offFn idx + 16 := by
      simp only [out2Offset, offB, offH, mIndex, dIndex, hoffFn]; try ring
    have hinj2 : Function.Injective (fun idx : TileIndex [16, 16] => offFn idx + 16) := by
      rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
      simp only [hoffFn] at h
      have hm : ma = mb := by omega
      have hd2 : da = db := by omega
      subst hm; subst hd2; rfl
    rw [ho2eq, hs4d]
    rw [BlockState.scatter_readback_prop_masked_nd s3 (fun idx => offFn idx + 16) acc2Fn
      (fun idx : TileIndex [16, 16] => s0.pids 0 * 16 + idx.1.val < 16) hinj2 idx]
    by_cases hlt : s0.pids 0 * 16 + idx.1.val < 16
    · rw [if_pos hlt, if_pos hlt]
    · rw [if_neg hlt, if_neg hlt]
      -- s3 at offFn+16 = s2 at offFn+16 (first scatter at offFn doesn't touch offFn+16)
      rw [hs3d]
      rw [bsa_foldl_writeMem_preserves_off offFn accFn
        (fun idx : TileIndex [16, 16] => s0.pids 0 * 16 + idx.1.val < 16) (offFn idx + 16)
        _ s2 (by intro k _hk _hPk; simp only [hoffFn]; omega)]
      simp only [BlockState.readMem, hs2mem, hmem]

/-! ## Full-kernel execution (`bsa_exec`)

Composing `bsaPreLoop_eval` (→ `bsaInvariant 0`, plus the `start_l`/`end_l`
register values), `bsa_csr_loop` (the `forRangeDyn` driver, instantiated with the
per-block `bsaLoopBody` advance), and `bsaPostLoop_eval` (reads out `acc`/`acc2`
= `bsaOPartial / bsaLPartial`) over `bsa_body_split`, from a clean input state the
whole lowered test-shape `block_sparse_attention_kernel` body runs to a final
state whose two `out` stores hold the streaming online-softmax accumulators at the
active lanes and preserve inactive lanes.

The CSR schedule data — `start_l`/`end_l` (the row-pointer window), the
per-block selection and the dot-product = `gScore` alignment that advances the
accumulators by one block — are *given* together as the `hstep` hypothesis of
`bsa_exec`: this is the trusted host boundary (which key blocks the layout
selects) plus the per-block `bsaLoopBody` advance. The summary then bridges
the streaming output to `blockSparseAttnClosedForm` via `bsaStreaming_eq_bsaAttn`. -/
set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open BSAMathCausal in
theorem bsa_exec
    (Out Q K V : RegionName) (R C : Region .nat)
    (qStart numKVBlocks : Nat) (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 32] → ℝ)
    (Kg : TileIndex [16 * numKVBlocks, 32] → ℝ)
    (Vg Vg2 : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (start_l end_l : Nat)
    (hStartL : s.readMemValue .nat R.cast (s.pids 1 % 4 % 1 * 3 + s.pids 0) = start_l)
    (hEndL : s.readMemValue .nat R.cast (s.pids 1 % 4 % 1 * 3 + s.pids 0 + 1) = end_l)
    (hbound : end_l - start_l = numKVBlocks) (hsle : start_l ≤ end_l)
    -- per-block selection bridges (CSR schedule + dot = gScore), for every block c
    (hstep : ∀ (i : Nat) (st : BlockState), start_l ≤ i → i < end_l →
      bsaInvariant Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s (i - start_l) st →
      ∃ st', stepStmts (bsaLoopBody C) (st.setReg "col_idx_idx" .nat [] (Tile.scalar i)) = some st'
        ∧ bsaInvariant Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s
            (i - start_l + 1) st') :
    ∃ sF, stepStmts ((block_sparse_attention_kernel Out Q K V R C
        3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
        4 2 16 16 16 16 2 Bool.true Bool.true).toAlgKernel.body) s = some sF
      ∧ (∀ idx : TileIndex [16, 16],
          sF.readMem Out (outOffset s 4 2048 512 32 16 idx)
            = if s.pids 0 * 16 + idx.1.val < 16 then
                (bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg scale numKVBlocks idx /
                  bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1)
              else s.readMem Out (outOffset s 4 2048 512 32 16 idx))
      ∧ (∀ idx : TileIndex [16, 16],
          sF.readMem Out (out2Offset s 4 2048 512 32 16 16 idx)
            = if s.pids 0 * 16 + idx.1.val < 16 then
                (bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg2 scale numKVBlocks idx /
                  bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1)
              else s.readMem Out (out2Offset s 4 2048 512 32 16 16 idx)) := by
  rw [bsa_body_split]
  -- preLoop: clean state → invariant at 0, with start_l/end_l register values exposed
  obtain ⟨s0, hpre, hinv0, hsl, hel⟩ :=
    bsaPreLoop_eval s Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale hundef
  rw [stepStmts.append_some hpre]
  -- start_l/end_l register values at s0 (the loop-entry state)
  have hStartOp : evalOp (Op.ref .nat [] "start_l") s0 = some (Tile.scalar start_l) := by
    rw [evalOp_ref, hsl, hStartL]
  have hStopOp : evalOp (Op.ref .nat [] "end_l") s0 = some (Tile.scalar end_l) := by
    rw [evalOp_ref, hel, hEndL]
  -- the loop driver: forRangeDyn over the CSR window → invariant at numKVBlocks.
  -- Loop anchor = the clean input `s` (matches the supplied `hstep`); entry = `s0`.
  obtain ⟨final, sL, hloop, _hfin, hinvL⟩ :=
    bsa_csr_loop Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s
      start_l end_l hbound hsle s0 hStartOp hStopOp hinv0 hstep
  rw [stepStmts.cons_some hloop]
  -- postLoop: reads out acc/acc2 at the active lanes (anchored at the clean input `s`)
  obtain ⟨sP, hpost, hOut, hOut2⟩ :=
    bsaPostLoop_eval Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s sL hinvL
  rw [hpost]
  -- loop-end memory agrees with the clean input (loop preserves Out memory)
  obtain ⟨_, hmemL, _⟩ := hinvL
  refine ⟨sP, rfl, ?_, ?_⟩
  · intro idx
    rw [hOut idx]
    by_cases hlt : s.pids 0 * 16 + idx.1.val < 16
    · rw [if_pos hlt, if_pos hlt]
    · rw [if_neg hlt, if_neg hlt]
      simp only [BlockState.readMem, hmemL]
  · intro idx
    rw [hOut2 idx]
    by_cases hlt : s.pids 0 * 16 + idx.1.val < 16
    · rw [if_pos hlt, if_pos hlt]
    · rw [if_neg hlt, if_neg hlt]
      simp only [BlockState.readMem, hmemL]

/-- **Streaming output = genuine closed form.** For the BSA test-shape
instantiation (`Bk = BLOCK_N = 16` keys per CSR block, `numKVBlocks` selected
blocks, `gpos`/`Kg`/`Vg` gathered along `selKeyGlobal`), the streaming
`bsaOPartial / bsaLPartial` ratio at the full window equals
`blockSparseAttnClosedForm`. Chains `bsaStreaming_eq_bsaAttn` (Bk=16), the
`bsaAttn_reindex` reshape (`16·numKVBlocks = numKVBlocks·16`), and
`bsaAttn_eq_blockSparseAttnClosedForm` (`numSelBlocks = numKVBlocks`,
`BLOCK_N = 16`). `hVis0` (first selected key causally visible) is required for the
normalizer to be nonzero. `dBlockBase ∈ {0, 16}` selects the output D-block. -/
theorem bsa_streaming_eq_closedForm
    (s : BlockState) (Q K V : RegionName) (C : Region .nat)
    (numKVBlocks : Nat) (hN : 0 < numKVBlocks) (start_l dBlockBase : Nat)
    (idx : TileIndex [16, 16])
    (hVis0 : selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 0 ≤ s.pids 0 * 16 + idx.1.val) :
    BSAMathCausal.bsaOPartial 16 (s.pids 0 * 16) numKVBlocks
        (fun r : Fin (16 * numKVBlocks) => selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 r.val)
        (fun jx : TileIndex [16, 32] => qTileBSA s Q 4 2048 512 32 16 jx.1 jx.2.1.val)
        (fun jx : TileIndex [16 * numKVBlocks, 32] => kRowBSA s K 4 2 1024 512 32
          (selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 jx.1.val) jx.2.1.val)
        (fun jx : TileIndex [16 * numKVBlocks, 16] => vRowBSA s V 4 2 1024 512 32
          (selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 jx.1.val) (dBlockBase + jx.2.1.val))
        1.0 numKVBlocks idx /
      BSAMathCausal.bsaLPartial 16 (s.pids 0 * 16) numKVBlocks
        (fun r : Fin (16 * numKVBlocks) => selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 r.val)
        (fun jx : TileIndex [16, 32] => qTileBSA s Q 4 2048 512 32 16 jx.1 jx.2.1.val)
        (fun jx : TileIndex [16 * numKVBlocks, 32] => kRowBSA s K 4 2 1024 512 32
          (selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 jx.1.val) jx.2.1.val)
        1.0 numKVBlocks idx.1
      = blockSparseAttnClosedForm s Q K V C 4 2 2048 512 32 1024 512 32 1024 512 32
          (s.pids 1 % 4 % 1) 4 start_l numKVBlocks 32 16 16 dBlockBase 1.0 idx.1 idx.2.1.val := by
  -- step 1: streaming = bsaAttn (Bk = 16, N = numKVBlocks)
  have h1 := BSAMathCausal.bsaStreaming_eq_bsaAttn (M := 16) (D := 32) (Dv := 16) (Bk := 16)
    (by norm_num) (s.pids 0 * 16) numKVBlocks hN
    (fun r : Fin (16 * numKVBlocks) => selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 r.val)
    (fun jx : TileIndex [16, 32] => qTileBSA s Q 4 2048 512 32 16 jx.1 jx.2.1.val)
    (fun jx : TileIndex [16 * numKVBlocks, 32] => kRowBSA s K 4 2 1024 512 32
      (selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 jx.1.val) jx.2.1.val)
    (fun jx : TileIndex [16 * numKVBlocks, 16] => vRowBSA s V 4 2 1024 512 32
      (selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 jx.1.val) (dBlockBase + jx.2.1.val))
    1.0 idx hVis0
  rw [h1]
  -- step 2: reindex bsaAttn from (Bk=16, N=numKVBlocks) to (Bk=numKVBlocks, N=16)
  rw [BSAMathCausal.bsaAttn_reindex (M := 16) (D := 32) (Dv := 16)
      (Bk₁ := 16) (N₁ := numKVBlocks) (Bk₂ := numKVBlocks) (N₂ := 16) (Nat.mul_comm 16 numKVBlocks)
      (s.pids 0 * 16)
      (fun n => selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 n)
      (fun n e => kRowBSA s K 4 2 1024 512 32 (selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 n) e.val)
      (fun n d => vRowBSA s V 4 2 1024 512 32 (selKeyGlobal s C (s.pids 1 % 4 % 1) 4 start_l 16 n) (dBlockBase + d.val))
      (fun jx : TileIndex [16, 32] => qTileBSA s Q 4 2048 512 32 16 jx.1 jx.2.1.val) 1.0 idx]
  -- step 3: bsaAttn (Bk = numKVBlocks, N = 16) = blockSparseAttnClosedForm
  obtain ⟨i, d, u⟩ := idx
  exact BSAMathCausal.bsaAttn_eq_blockSparseAttnClosedForm s Q K V C 4 2
    2048 512 32 1024 512 32 1024 512 32 (s.pids 1 % 4 % 1) 4 start_l numKVBlocks 32 16 16 16
    dBlockBase 1.0 i d

/-! ## Symbolic-dimension exec stack (`…G`)

The pinned exec stack above (`bsaInvariant`/`bsaPreLoop_eval`/`bsa_csr_loop`/
`bsaPostLoop_eval`/`bsa_exec`/`bsa_streaming_eq_closedForm`/the `case1` summary)
is specialized to the Python test shape (`BLOCK_M=BLOCK_N=BLOCK_D=16`,
`num_heads=4`, `num_kv_heads=2`, strides `2048/512/32/1024`, `num_layout=1`,
row/col CSR strides `3`/`4`, `total_seq_len=16`).

This section adds a **symbolic-dimension** version of the *body split* — the pure
`List Stmt` identity for the lowered kernel body at arbitrary tile dims/strides
(`NUM_D_BLOCKS = 2`, `EVEN_M = EVEN_N = true`). The general pre-loop / loop-body /
post-loop `List Stmt` defs mirror the pinned ones with every literal replaced by
a symbolic parameter; the split holds by `rfl`. -/

/-- General (symbolic-dim) pre-loop statement list — the 29 lowered pre-loop
statements with every dim/stride symbolic. Mirrors `bsaPreLoop` literal-for-literal
against the elaborated body dump (`NUM_D_BLOCKS = 2`, `EVEN_M = true`). -/
def bsaPreLoopG (Out Q K V : RegionName) (R : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len : Nat)
    (rowStrideH sqb sqh sqm skb skh skn svb svh svn : Nat) : List Stmt :=
  [ Stmt.ifThen (Op.constBool «false») [],
    Stmt.assign .nat [] "q_seq_len" (Op.constNat total_seq_len),
    Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_bh" (Op.programId 1),
    Stmt.assign .nat [] "off_h"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_bh") (Op.constNat num_heads)),
    Stmt.assign .nat [] "off_b"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_bh") (Op.constNat num_heads)),
    Stmt.assign .nat [] "head_groups"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat num_heads) (Op.constNat num_kv_heads)),
    Stmt.assign .nat [] "off_h_kv"
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h")
        (Op.ref .nat [] "head_groups")),
    Stmt.assign .ptr [] "Q"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase Q)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat sqb))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat sqh)))),
    Stmt.assign .ptr [] "K"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase K)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat skb))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h_kv") (Op.constNat skh)))),
    Stmt.assign .ptr [] "V"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase V)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat svb))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h_kv") (Op.constNat svh)))),
    Stmt.assign .nat [BLOCK_M] "offs_m"
      (Op.add NumericDType.nat Broadcast.scalarL
        (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
        (Op.arange BLOCK_M)),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_D] "offs_d" (Op.arange BLOCK_D),
    Stmt.assign .nat [BLOCK_M, BLOCK_D] "off_q"
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat sqm))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_D] "offs_d"))),
    Stmt.assign .nat [BLOCK_D, BLOCK_N] "off_k"
      (Op.add NumericDType.nat Broadcast.nil.consR.consL
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat skn))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_D] "offs_d"))),
    Stmt.assign .nat [BLOCK_N, BLOCK_D] "off_v"
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat svn))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_D] "offs_d"))),
    Stmt.assign .ptr [BLOCK_M, BLOCK_D] "q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Q") (Op.ref .nat [BLOCK_M, BLOCK_D] "off_q")),
    Stmt.assign .ptr [BLOCK_D, BLOCK_N] "k_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "K") (Op.ref .nat [BLOCK_D, BLOCK_N] "off_k")),
    Stmt.assign .ptr [BLOCK_N, BLOCK_D] "v_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "V") (Op.ref .nat [BLOCK_N, BLOCK_D] "off_v")),
    Stmt.assign .real [BLOCK_M] "m_i"
      (Op.add NumericDType.real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLOCK_M] "l_i" (Op.full [BLOCK_M] (Op.const 0)),
    Stmt.assign .real [BLOCK_M, BLOCK_D] "acc" (Op.full [BLOCK_M, BLOCK_D] (Op.const 0)),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.assign .real [BLOCK_M, BLOCK_D] "acc2" (Op.full [BLOCK_M, BLOCK_D] (Op.const 0))],
    Stmt.ifThenElse (Op.constBool «true»)
      [Stmt.assign .real [BLOCK_M, BLOCK_D] "q"
          (Op.load .real (MemAccess.ptr (Op.ref .ptr [BLOCK_M, BLOCK_D] "q_ptrs")) MaskOpt.none),
        Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
          [Stmt.assign .real [BLOCK_M, BLOCK_D] "q2"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_M, BLOCK_D] "q_ptrs")
                    (Op.constNat BLOCK_D)))
                MaskOpt.none)]]
      [Stmt.assign .real [BLOCK_M, BLOCK_D] "q"
          (Op.load .real (MemAccess.ptr (Op.ref .ptr [BLOCK_M, BLOCK_D] "q_ptrs"))
            (MaskOpt.mask
              (Op.remap [BLOCK_M, BLOCK_D] Broadcast.nil.consL.consSame.leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
                  (Op.ref .nat [] "q_seq_len"))))),
        Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
          [Stmt.assign .real [BLOCK_M, BLOCK_D] "q2"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_M, BLOCK_D] "q_ptrs")
                    (Op.constNat BLOCK_D)))
                (MaskOpt.mask
                  (Op.remap [BLOCK_M, BLOCK_D] Broadcast.nil.consL.consSame.leftIndex
                    (Op.lt ComparableDType.nat Broadcast.scalarR
                      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
                      (Op.ref .nat [] "q_seq_len")))))]],
    Stmt.assign .nat [] "layout_h"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat num_layout)),
    Stmt.assign .ptr [] "layout_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase R)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "layout_h") (Op.constNat rowStrideH))
          (Op.ref .nat [] "start_m"))),
    Stmt.assign .nat [] "start_l"
      (Op.load .nat (MemAccess.ptr (Op.ref .ptr [] "layout_ptr")) MaskOpt.none),
    Stmt.assign .nat [] "end_l"
      (Op.load .nat
        (MemAccess.ptr (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "layout_ptr") (Op.constNat 1)))
        MaskOpt.none) ]

/-- General (symbolic-dim) CSR-loop body — the 26 lowered loop-body statements with
every dim/stride symbolic. Mirrors `bsaLoopBody` literal-for-literal against the
elaborated body dump (`NUM_D_BLOCKS = 2`, `EVEN_N = true`). -/
def bsaLoopBodyG (C : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads colStrideH skn svn total_seq_len : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [] "col_idx"
      (Op.load .nat
        (MemAccess.region C
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "layout_h") (Op.constNat colStrideH))
            (Op.ref .nat [] "col_idx_idx")))
        MaskOpt.none),
    Stmt.assign .nat [] "start_n"
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "col_idx") (Op.constNat BLOCK_N)),
    Stmt.ifThenElse (Op.constBool «true»)
      [Stmt.assign .real [BLOCK_D, BLOCK_N] "k"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_D, BLOCK_N] "k_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat skn))))
            MaskOpt.none)]
      [Stmt.assign .real [BLOCK_D, BLOCK_N] "k"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_D, BLOCK_N] "k_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat skn))))
            (MaskOpt.mask
              (Op.remap [BLOCK_D, BLOCK_N] Broadcast.nil.consSame.consL.leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.add NumericDType.nat Broadcast.scalarR
                    (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
                    (Op.ref .nat [] "start_n"))
                  (Op.constNat total_seq_len)))))],
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk" (Op.full [BLOCK_M, BLOCK_N] (Op.const 0)),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_D] "q") (Op.ref .real [BLOCK_D, BLOCK_N] "k"))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.ifThenElse (Op.constBool «true»)
          [Stmt.assign .real [BLOCK_D, BLOCK_N] "k"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_D, BLOCK_N] "k_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat skn)))
                    (Op.constNat BLOCK_D)))
                MaskOpt.none)]
          [Stmt.assign .real [BLOCK_D, BLOCK_N] "k"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_D, BLOCK_N] "k_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat skn)))
                    (Op.constNat BLOCK_D)))
                (MaskOpt.mask
                  (Op.remap [BLOCK_D, BLOCK_N] Broadcast.nil.consSame.consL.leftIndex
                    (Op.lt ComparableDType.nat Broadcast.scalarR
                      (Op.add NumericDType.nat Broadcast.scalarR
                        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
                        (Op.ref .nat [] "start_n"))
                      (Op.constNat total_seq_len)))))],
        Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
          (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
            (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_D] "q2") (Op.ref .real [BLOCK_D, BLOCK_N] "k")))],
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.mul NumericDType.real Broadcast.scalarR (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.const scale)),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
              (Op.add NumericDType.nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))).where
          ((Op.const 0).broadcast [BLOCK_M, BLOCK_N]) (Op.negInf.broadcast [BLOCK_M, BLOCK_N]))),
    Stmt.assign .real [BLOCK_M] "m_ij"
      (Op.reduceMax ⟨1, by simp⟩ «false» (Op.ref .real [BLOCK_M, BLOCK_N] "qk")),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.sub NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "m_ij"))).exp,
    Stmt.assign .real [BLOCK_M] "l_ij"
      (Op.reduceSum ⟨1, by simp⟩ «false» (Op.ref .real [BLOCK_M, BLOCK_N] "p")),
    Stmt.assign .real [BLOCK_M] "m_i_new"
      ((Op.gt ComparableDType.real (Broadcast.consSame Broadcast.nil)
            (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij")).where
        (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij")),
    Stmt.assign .real [BLOCK_M] "alpha"
      (Op.sub NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_i_new")).exp,
    Stmt.assign .real [BLOCK_M] "beta"
      (Op.sub NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_M] "m_ij") (Op.ref .real [BLOCK_M] "m_i_new")).exp,
    Stmt.assign .real [BLOCK_M] "l_i_new"
      (Op.add NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.mul NumericDType.real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "alpha") (Op.ref .real [BLOCK_M] "l_i"))
        (Op.mul NumericDType.real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "beta") (Op.ref .real [BLOCK_M] "l_ij"))),
    Stmt.assign .real [BLOCK_M] "p_scale"
      (Op.div NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_M] "beta") (Op.ref .real [BLOCK_M] "l_i_new")),
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.mul NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_N] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "p_scale"))),
    Stmt.assign .real [BLOCK_M] "acc_scale"
      (Op.mul NumericDType.real (Broadcast.consSame Broadcast.nil)
        (Op.div NumericDType.real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "l_i") (Op.ref .real [BLOCK_M] "l_i_new"))
        (Op.ref .real [BLOCK_M] "alpha")),
    Stmt.assign .real [BLOCK_M, BLOCK_D] "acc"
      (Op.mul NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_D] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "acc_scale"))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.assign .real [BLOCK_M, BLOCK_D] "acc2"
          (Op.mul NumericDType.real Broadcast.nil.consR.consSame (Op.ref .real [BLOCK_M, BLOCK_D] "acc2")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "acc_scale")))],
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p" (Op.ref .real [BLOCK_M, BLOCK_N] "p"),
    Stmt.ifThenElse (Op.constBool «true»)
      [Stmt.assign .real [BLOCK_N, BLOCK_D] "v"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_N, BLOCK_D] "v_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat svn))))
            MaskOpt.none)]
      [Stmt.assign .real [BLOCK_N, BLOCK_D] "v"
          (Op.load .real
            (MemAccess.ptr
              (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_N, BLOCK_D] "v_ptrs")
                (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat svn))))
            (MaskOpt.mask
              (Op.remap [BLOCK_N, BLOCK_D] Broadcast.nil.consL.consSame.leftIndex
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.add NumericDType.nat Broadcast.scalarR
                    (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
                    (Op.ref .nat [] "start_n"))
                  (Op.constNat total_seq_len)))))],
    Stmt.assign .real [BLOCK_M, BLOCK_D] "acc"
      (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_D] "acc")
        (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_N] "p") (Op.ref .real [BLOCK_N, BLOCK_D] "v"))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.ifThenElse (Op.constBool «true»)
          [Stmt.assign .real [BLOCK_N, BLOCK_D] "v"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_N, BLOCK_D] "v_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat svn)))
                    (Op.constNat BLOCK_D)))
                MaskOpt.none)]
          [Stmt.assign .real [BLOCK_N, BLOCK_D] "v"
              (Op.load .real
                (MemAccess.ptr
                  (Op.ptrAdd Broadcast.scalarR
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_N, BLOCK_D] "v_ptrs")
                      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat svn)))
                    (Op.constNat BLOCK_D)))
                (MaskOpt.mask
                  (Op.remap [BLOCK_N, BLOCK_D] Broadcast.nil.consL.consSame.leftIndex
                    (Op.lt ComparableDType.nat Broadcast.scalarR
                      (Op.add NumericDType.nat Broadcast.scalarR
                        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
                        (Op.ref .nat [] "start_n"))
                      (Op.constNat total_seq_len)))))],
        Stmt.assign .real [BLOCK_M, BLOCK_D] "acc2"
          (Op.add NumericDType.real Broadcast.nil.consSame.consSame (Op.ref .real [BLOCK_M, BLOCK_D] "acc2")
            (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_N] "p") (Op.ref .real [BLOCK_N, BLOCK_D] "v")))],
    Stmt.assign .real [BLOCK_M] "l_i" (Op.ref .real [BLOCK_M] "l_i_new"),
    Stmt.assign .real [BLOCK_M] "m_i" (Op.ref .real [BLOCK_M] "m_i_new") ]

/-- General (symbolic-dim) post-loop statement list — `off_o`, `out_ptrs`, and the
two masked `out` stores (`NUM_D_BLOCKS = 2`). Mirrors `bsaPostLoop`. -/
def bsaPostLoopG (Out : RegionName)
    (BLOCK_M BLOCK_D : Nat) (sob soh som : Nat) : List Stmt :=
  [ Stmt.assign .nat [BLOCK_M, BLOCK_D] "off_o"
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat sob))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat soh)))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat som)))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_D] "offs_d"))),
    Stmt.assign .ptr [BLOCK_M, BLOCK_D] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLOCK_M, BLOCK_D] "off_o")),
    Stmt.store .real [BLOCK_M, BLOCK_D] (MemAccess.ptr (Op.ref .ptr [BLOCK_M, BLOCK_D] "out_ptrs"))
      (Op.ref .real [BLOCK_M, BLOCK_D] "acc")
      (MaskOpt.mask
        (Op.remap [BLOCK_M, BLOCK_D] Broadcast.nil.consL.consSame.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
            (Op.ref .nat [] "q_seq_len")))),
    Stmt.ifThen (Op.constBool (decide (2 ≥ 2)))
      [Stmt.store .real [BLOCK_M, BLOCK_D]
          (MemAccess.ptr
            (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_M, BLOCK_D] "out_ptrs") (Op.constNat BLOCK_D)))
          (Op.ref .real [BLOCK_M, BLOCK_D] "acc2")
          (MaskOpt.mask
            (Op.remap [BLOCK_M, BLOCK_D] Broadcast.nil.consL.consSame.leftIndex
              (Op.lt ComparableDType.nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
                (Op.ref .nat [] "q_seq_len"))))] ]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General body split.** The lowered `block_sparse_attention_kernel` body at
*arbitrary* tile dims / strides (`NUM_D_BLOCKS = 2`, `EVEN_M = EVEN_N = true`)
splits as `bsaPreLoopG ++ forRangeDyn "col_idx_idx" (ref start_l) (ref end_l) 1
bsaLoopBodyG :: bsaPostLoopG`. Pure `List Stmt` identity (`rfl`). -/
theorem bsa_body_splitG
    (Out Q K V : RegionName) (R C : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len : Nat)
    (rowStrideH colStrideH sqb sqh sqm skb skh skn svb svh svn sob soh som : Nat)
    (scale : ℝ) :
    (block_sparse_attention_kernel Out Q K V R C
      rowStrideH colStrideH num_layout scale sqb sqh sqm skb skh skn svb svh svn sob soh som
      num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D 2 Bool.true Bool.true).toAlgKernel.body
      = bsaPreLoopG Out Q K V R BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len
          rowStrideH sqb sqh sqm skb skh skn svb svh svn
        ++ (Stmt.forRangeDyn "col_idx_idx" (Op.ref .nat [] "start_l") (Op.ref .nat [] "end_l")
              (Op.constNat 1) (bsaLoopBodyG C BLOCK_M BLOCK_D BLOCK_N num_heads colStrideH skn svn total_seq_len scale)
            :: bsaPostLoopG Out BLOCK_M BLOCK_D sob soh som) := by
  unfold block_sparse_attention_kernel bsaPreLoopG bsaLoopBodyG bsaPostLoopG
  rfl

open BSAMathCausal in
/-- General (symbolic-dim) CSR loop invariant after `c` selected `BLOCK_N`-blocks.
Mirrors `bsaInvariant` with every literal replaced by a symbolic dim/stride. The
math accumulators carry `M = BLOCK_M`, contraction `D = 2·BLOCK_D` (the two q/k
D-blocks), value dim `Dv = BLOCK_D`, block size `Bk = BLOCK_N`. -/
noncomputable def bsaInvariantG
    (Out Q K V : RegionName) (R C : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len : Nat)
    (sqb sqh sqm skb skh skn svb svh svn : Nat)
    (qStart numKVBlocks : Nat) (gpos : Fin (BLOCK_N * numKVBlocks) → Nat)
    (Qg : TileIndex [BLOCK_M, 2 * BLOCK_D] → ℝ)
    (Kg : TileIndex [BLOCK_N * numKVBlocks, 2 * BLOCK_D] → ℝ)
    (Vg Vg2 : TileIndex [BLOCK_N * numKVBlocks, BLOCK_D] → ℝ) (scale : ℝ)
    (s0 : BlockState) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "q_seq_len" = some (Tile.scalar total_seq_len)
  ∧ s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "off_bh" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "off_h" = some (Tile.scalar (s0.pids 1 % num_heads))
  ∧ s.regs .nat [] "off_b" = some (Tile.scalar (s0.pids 1 / num_heads))
  ∧ s.regs .nat [] "head_groups" = some (Tile.scalar (num_heads / num_kv_heads))
  ∧ s.regs .nat [] "off_h_kv" =
      some (Tile.scalar (s0.pids 1 % num_heads / (num_heads / num_kv_heads)))
  ∧ s.regs .nat [BLOCK_M] "offs_m" =
      some (Tile.vec (fun i : Fin BLOCK_M => s0.pids 0 * BLOCK_M + i.val))
  ∧ s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
  ∧ s.regs .nat [BLOCK_D] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_D => e.val))
  ∧ s.regs .nat [] "layout_h" = some (Tile.scalar (s0.pids 1 % num_heads % num_layout))
  ∧ s.regs .real [BLOCK_M] "m_i" =
      some (Tile.vec (fun i : Fin BLOCK_M =>
        bsaMPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale c i))
  ∧ s.regs .real [BLOCK_M] "l_i" =
      some (Tile.vec (fun i : Fin BLOCK_M =>
        (some (bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale c i) : WithBot ℝ)))
  ∧ s.regs .real [BLOCK_M, BLOCK_D] "acc" =
      some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        (some (bsaOPartial BLOCK_N qStart numKVBlocks gpos Qg Kg Vg scale c idx /
          bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale c idx.1) : WithBot ℝ)⟩
          : Tile .real [BLOCK_M, BLOCK_D])
  ∧ s.regs .real [BLOCK_M, BLOCK_D] "acc2" =
      some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        (some (bsaOPartial BLOCK_N qStart numKVBlocks gpos Qg Kg Vg2 scale c idx /
          bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale c idx.1) : WithBot ℝ)⟩
          : Tile .real [BLOCK_M, BLOCK_D])
  ∧ s.regs .real [BLOCK_M, BLOCK_D] "q" =
      some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        s0.readMemValue .real Q ((s0.pids 1 / num_heads * sqb + s0.pids 1 % num_heads * sqh) +
          ((s0.pids 0 * BLOCK_M + idx.1.val) * sqm + idx.2.1.val))⟩ : Tile .real [BLOCK_M, BLOCK_D])
  ∧ s.regs .real [BLOCK_M, BLOCK_D] "q2" =
      some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        s0.readMemValue .real Q ((s0.pids 1 / num_heads * sqb + s0.pids 1 % num_heads * sqh) +
          ((s0.pids 0 * BLOCK_M + idx.1.val) * sqm + idx.2.1.val) + BLOCK_D)⟩
          : Tile .real [BLOCK_M, BLOCK_D])
  ∧ s.regs .ptr [BLOCK_D, BLOCK_N] "k_ptrs" =
      some (⟨fun idx : TileIndex [BLOCK_D, BLOCK_N] =>
        (K, (s0.pids 1 / num_heads * skb +
            s0.pids 1 % num_heads / (num_heads / num_kv_heads) * skh) +
          (idx.2.1.val * skn + idx.1.val))⟩ : Tile .ptr [BLOCK_D, BLOCK_N])
  ∧ s.regs .ptr [BLOCK_N, BLOCK_D] "v_ptrs" =
      some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_D] =>
        (V, (s0.pids 1 / num_heads * svb +
            s0.pids 1 % num_heads / (num_heads / num_kv_heads) * svh) +
          (idx.1.val * svn + idx.2.1.val))⟩ : Tile .ptr [BLOCK_N, BLOCK_D])

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General pre-loop execution.** Mirrors `bsaPreLoop_eval` at symbolic dims:
the 29 deterministic pre-loop statements step a clean input state to a state
satisfying `bsaInvariantG … 0`, with the `start_l`/`end_l` register values exposed.
-/
theorem bsaPreLoop_evalG
    (s : BlockState) (Out Q K V : RegionName) (R C : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len : Nat)
    (rowStrideH sqb sqh sqm skb skh skn svb svh svn : Nat)
    (qStart numKVBlocks : Nat) (gpos : Fin (BLOCK_N * numKVBlocks) → Nat)
    (Qg : TileIndex [BLOCK_M, 2 * BLOCK_D] → ℝ)
    (Kg : TileIndex [BLOCK_N * numKVBlocks, 2 * BLOCK_D] → ℝ)
    (Vg Vg2 : TileIndex [BLOCK_N * numKVBlocks, BLOCK_D] → ℝ) (scale : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (bsaPreLoopG Out Q K V R BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads
        num_layout total_seq_len rowStrideH sqb sqh sqm skb skh skn svb svh svn) s = some s0
      ∧ bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
          total_seq_len sqb sqh sqm skb skh skn svb svh svn
          qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s 0 s0
      ∧ s0.regs .nat [] "start_l" =
          some (Tile.scalar (s.readMemValue .nat R.cast
            (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0)))
      ∧ s0.regs .nat [] "end_l" =
          some (Tile.scalar (s.readMemValue .nat R.cast
            (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0 + 1))) := by
  unfold bsaPreLoopG
  -- stmt 0: static_print marker `ifThen false []` is a no-op
  rw [stepStmts.cons_some (stepStmt_ifThen_false (by simp [evalOp]))]
  -- stmt 1: q_seq_len = total_seq_len
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.constNat total_seq_len) s = some (Tile.scalar total_seq_len) from by simp))]
  -- stmt 2: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 _))]
  -- stmt 3: off_bh = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 4: off_h = off_bh % num_heads
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_bh") (Op.constNat num_heads)) _
        = some (Tile.scalar (s.pids 1 % num_heads)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.mod]))]
  -- stmt 5: off_b = off_bh // num_heads
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_bh") (Op.constNat num_heads)) _
        = some (Tile.scalar (s.pids 1 / num_heads)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.floorDiv]))]
  -- stmt 6: head_groups = num_heads // num_kv_heads
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat num_heads) (Op.constNat num_kv_heads)) _
        = some (Tile.scalar (num_heads / num_kv_heads)) from by
      simp only [evalOp, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.floorDiv]))]
  -- stmt 7: off_h_kv = off_h // head_groups
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h")
          (Op.ref .nat [] "head_groups")) _
        = some (Tile.scalar (s.pids 1 % num_heads / (num_heads / num_kv_heads))) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.floorDiv]))]
  -- stmt 8: Q += off_b*sqb + off_h*sqh
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Q)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat sqb))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat sqh)))) _
        = some (Tile.scalar (Q, s.pids 1 / num_heads * sqb + s.pids 1 % num_heads * sqh)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.scalar_data_index,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
        NumericDType.mul, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]))]
  -- stmt 9: K += off_b*skb + off_h_kv*skh
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase K)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat skb))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h_kv") (Op.constNat skh)))) _
        = some (Tile.scalar (K, s.pids 1 / num_heads * skb +
            s.pids 1 % num_heads / (num_heads / num_kv_heads) * skh)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.scalar_data_index,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
        NumericDType.mul, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]))]
  -- stmt 10: V += off_b*svb + off_h_kv*svh
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase V)
        (Op.add NumericDType.nat Broadcast.nil
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat svb))
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h_kv") (Op.constNat svh)))) _
        = some (Tile.scalar (V, s.pids 1 / num_heads * svb +
            s.pids 1 % num_heads / (num_heads / num_kv_heads) * svh)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.scalar_data_index,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
        NumericDType.mul, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]))]
  -- stmt 11: offs_m = start_m*BLOCK_M + arange BLOCK_M
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.scalarL
          (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M))
          (Op.arange BLOCK_M)) _
        = some (Tile.vec (fun i : Fin BLOCK_M => s.pids 0 * BLOCK_M + i.val)) from by
      simp only [evalOp_add, evalOp_mul, evalOp_arange, evalOp_ref, evalOp_constNat,
        BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.vec, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 12: offs_n = arange BLOCK_N
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_N) _ = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) from evalOp_arange BLOCK_N _))]
  -- stmt 13: offs_d = arange BLOCK_D
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_D) _ = some (Tile.vec (fun e : Fin BLOCK_D => e.val)) from evalOp_arange BLOCK_D _))]
  -- stmt 14: off_q = offs_m[:,None]*sqm + offs_d[None,:]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil.consL.consR
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat sqm))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_D] "offs_d"))) _
        = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
            (s.pids 0 * BLOCK_M + idx.1.val) * sqm + idx.2.1.val⟩ : Tile .nat [BLOCK_M, BLOCK_D]) from by
      rw [evalOp_add, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs .nat [BLOCK_M] ⟨1, by simp⟩ "offs_m" _
            (Tile.vec (fun i : Fin BLOCK_M => s.pids 0 * BLOCK_M + i.val)) (by simp [Tile.vec]),
        evalOp_expandDim_ref_of_regs .nat [BLOCK_D] ⟨0, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin BLOCK_D => e.val)) (by simp [Tile.vec])]
      simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 15: off_k = offs_n[None,:]*skn + offs_d[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil.consR.consL
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat skn))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_D] "offs_d"))) _
        = some (⟨fun idx : TileIndex [BLOCK_D, BLOCK_N] =>
            idx.2.1.val * skn + idx.1.val⟩ : Tile .nat [BLOCK_D, BLOCK_N]) from by
      rw [evalOp_add, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs .nat [BLOCK_N] ⟨0, by simp⟩ "offs_n" _
            (Tile.vec (fun j : Fin BLOCK_N => j.val)) (by simp [Tile.vec]),
        evalOp_expandDim_ref_of_regs .nat [BLOCK_D] ⟨1, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin BLOCK_D => e.val)) (by simp [Tile.vec])]
      simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 16: off_v = offs_n[:,None]*svn + offs_d[None,:]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil.consL.consR
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat svn))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_D] "offs_d"))) _
        = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_D] =>
            idx.1.val * svn + idx.2.1.val⟩ : Tile .nat [BLOCK_N, BLOCK_D]) from by
      rw [evalOp_add, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs .nat [BLOCK_N] ⟨1, by simp⟩ "offs_n" _
            (Tile.vec (fun j : Fin BLOCK_N => j.val)) (by simp [Tile.vec]),
        evalOp_expandDim_ref_of_regs .nat [BLOCK_D] ⟨0, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin BLOCK_D => e.val)) (by simp [Tile.vec])]
      simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.expandDim_data, Tile.vec, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 17: q_ptrs = Q + off_q
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "Q") (Op.ref .nat [BLOCK_M, BLOCK_D] "off_q")) _
        = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
            (Q, (s.pids 1 / num_heads * sqb + s.pids 1 % num_heads * sqh) +
              ((s.pids 0 * BLOCK_M + idx.1.val) * sqm + idx.2.1.val))⟩ : Tile .ptr [BLOCK_M, BLOCK_D]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨ir, dd, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  -- stmt 18: k_ptrs = K + off_k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "K") (Op.ref .nat [BLOCK_D, BLOCK_N] "off_k")) _
        = some (⟨fun idx : TileIndex [BLOCK_D, BLOCK_N] =>
            (K, (s.pids 1 / num_heads * skb +
                s.pids 1 % num_heads / (num_heads / num_kv_heads) * skh) +
              (idx.2.1.val * skn + idx.1.val))⟩ : Tile .ptr [BLOCK_D, BLOCK_N]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨ir, dd, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  -- stmt 19: v_ptrs = V + off_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "V") (Op.ref .nat [BLOCK_N, BLOCK_D] "off_v")) _
        = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_D] =>
            (V, (s.pids 1 / num_heads * svb +
                s.pids 1 % num_heads / (num_heads / num_kv_heads) * svh) +
              (idx.1.val * svn + idx.2.1.val))⟩ : Tile .ptr [BLOCK_N, BLOCK_D]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨ir, dd, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  -- stmt 20: m_i = zeros[BLOCK_M] - inf  (= ⊥-seed)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf) _
        = some (Tile.vec (fun _ : Fin BLOCK_M => (⊥ : WithBot ℝ))) from by
      simp only [evalOp_add, evalOp_full, evalOp_const, evalOp_negInf,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, Tile.scalar_data_index]
      rfl))]
  -- stmt 21: l_i = zeros[BLOCK_M]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLOCK_M] (Op.const 0)) _
        = some (Tile.vec (fun _ : Fin BLOCK_M => (some (0 : ℝ) : WithBot ℝ))) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx; simp [Tile.vec, Tile.scalar_data_index]))]
  -- stmt 22: acc = zeros[BLOCK_M,BLOCK_D]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLOCK_M, BLOCK_D] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_D] => (some (0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_D]) from by
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx; simp [Tile.scalar_data_index]))]
  -- stmt 23: ifThen (2 ≥ 2) { acc2 = zeros }
  rw [stepStmts.cons_some (stepStmt_ifThen_true (by simp [evalOp])
    (show stepStmts [Stmt.assign .real [BLOCK_M, BLOCK_D] "acc2" (Op.full [BLOCK_M, BLOCK_D] (Op.const 0))] _
        = some _ from by
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.full [BLOCK_M, BLOCK_D] (Op.const 0)) _
            = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_D] => (some (0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_D]) from by
          simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
          refine congrArg some ?_; ext idx; simp [Tile.scalar_data_index]))]
      rw [stepStmts.nil]))]
  -- stmt 24: ifThenElse true { q = load q_ptrs ; ifThen { q2 = load } }
  rw [stepStmts.cons_some (stepStmt_ifThenElse_true (by simp [evalOp])
    (show stepStmts _ _ = some _ from by
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BLOCK_M, BLOCK_D] "q_ptrs")) MaskOpt.none) _
            = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
                s.readMemValue .real Q ((s.pids 1 / num_heads * sqb + s.pids 1 % num_heads * sqh) +
                  ((s.pids 0 * BLOCK_M + idx.1.val) * sqm + idx.2.1.val))⟩ : Tile .real [BLOCK_M, BLOCK_D]) from by
          simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
            ne_eq, String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
          refine congrArg some (Tile.ext (fun idx => ?_))
          obtain ⟨ir, dd, u⟩ := idx; rfl))]
      rw [stepStmts.cons_some (stepStmt_ifThen_true (by simp [evalOp])
        (show stepStmts _ _ = some _ from by
          rw [stepStmts.cons_some (stepStmt_assign_eq_some
            (show evalOp (Op.load .real
                  (MemAccess.ptr
                    (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_M, BLOCK_D] "q_ptrs") (Op.constNat BLOCK_D)))
                  MaskOpt.none) _
                = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
                    s.readMemValue .real Q ((s.pids 1 / num_heads * sqb + s.pids 1 % num_heads * sqh) +
                      ((s.pids 0 * BLOCK_M + idx.1.val) * sqm + idx.2.1.val) + BLOCK_D)⟩
                      : Tile .real [BLOCK_M, BLOCK_D]) from by
              simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
                BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
                Option.bind_eq_bind, Option.bind_some]
              refine congrArg some (Tile.ext (fun idx => ?_))
              obtain ⟨ir, dd, u⟩ := idx
              simp [Tile.ptrAdd_data, Tile.scalar_data_index,
                Broadcast.leftIndex, Broadcast.rightIndex]))]
          rw [stepStmts.nil]))]
      rw [stepStmts.nil]))]
  -- stmt 25: layout_h = off_h % num_layout
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat num_layout)) _
        = some (Tile.scalar (s.pids 1 % num_heads % num_layout)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.mod]))]
  -- stmt 26: layout_ptr = R + (layout_h*rowStrideH + start_m)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase R)
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "layout_h") (Op.constNat rowStrideH))
            (Op.ref .nat [] "start_m"))) _
        = some (Tile.scalar ((R.cast : RegionName), s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.scalar_data_index,
        castTile_self, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
        NumericDType.mul, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]))]
  -- stmt 27: start_l = load layout_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_startl_eval _ R.cast (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0)
      (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true])))]
  -- stmt 28: end_l = load (layout_ptr + 1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_endl_eval _ R.cast (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0)
      (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
        String.reduceEq, not_false_eq_true])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_⟩
  · unfold bsaInvariantG
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · simp only [BlockState.setReg_pids]
    · funext rg o; simp only [BlockState.setReg_mem]
    · intro rg o; simp [hundef]
    all_goals
      simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true]
    all_goals
      first
      | rfl
      | (refine congrArg some (Tile.ext (fun idx => ?_));
         refine congrArg some ?_;
         rw [BSAMathCausal.bsaOPartial, BSAMathCausal.bsaLPartial, zero_div])
  · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), BlockState.setReg_same]
    refine congrArg some (congrArg Tile.scalar ?_)
    simp only [BlockState.readMemValue, BlockState.readMemTyped, BlockState.setReg_mem]
  · rw [BlockState.setReg_same]
    refine congrArg some (congrArg Tile.scalar ?_)
    simp only [BlockState.readMemValue, BlockState.readMemTyped, BlockState.setReg_mem]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open BSAMathCausal in
/-- **General CSR loop driver.** Mirrors `bsa_csr_loop` at symbolic dims: from the
loop-entry invariant (`bsaInvariantG … 0`) and per-block gather bridges,
`forRangeDyn "col_idx_idx" start_l end_l 1 bsaLoopBodyG` runs to a final state at
counter `final ≥ end_l` satisfying `bsaInvariantG … numKVBlocks`. -/
theorem bsa_csr_loopG
    (Out Q K V : RegionName) (R C : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len : Nat)
    (sqb sqh sqm skb skh skn svb svh svn colStrideH : Nat) (scale : ℝ)
    (qStart numKVBlocks : Nat) (gpos : Fin (BLOCK_N * numKVBlocks) → Nat)
    (Qg : TileIndex [BLOCK_M, 2 * BLOCK_D] → ℝ)
    (Kg : TileIndex [BLOCK_N * numKVBlocks, 2 * BLOCK_D] → ℝ)
    (Vg Vg2 : TileIndex [BLOCK_N * numKVBlocks, BLOCK_D] → ℝ)
    (s0 : BlockState) (start_l end_l : Nat)
    (hbound : end_l - start_l = numKVBlocks) (hsle : start_l ≤ end_l)
    (sEntry : BlockState)
    (hStartOp : evalOp (Op.ref .nat [] "start_l") sEntry = some (Tile.scalar start_l))
    (hStopOp : evalOp (Op.ref .nat [] "end_l") sEntry = some (Tile.scalar end_l))
    (hInit : bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
      total_seq_len sqb sqh sqm skb skh skn svb svh svn qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s0 0 sEntry)
    (hstep : ∀ (i : Nat) (st : BlockState), start_l ≤ i → i < end_l →
      bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
        total_seq_len sqb sqh sqm skb skh skn svb svh svn qStart numKVBlocks gpos Qg Kg Vg Vg2 scale
        s0 (i - start_l) st →
      ∃ st', stepStmts (bsaLoopBodyG C BLOCK_M BLOCK_D BLOCK_N num_heads colStrideH skn svn total_seq_len scale)
          (st.setReg "col_idx_idx" .nat [] (Tile.scalar i)) = some st'
        ∧ bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
            total_seq_len sqb sqh sqm skb skh skn svb svh svn qStart numKVBlocks gpos Qg Kg Vg Vg2 scale
            s0 (i - start_l + 1) st') :
    ∃ final sFinal,
      stepStmt (Stmt.forRangeDyn "col_idx_idx" (Op.ref .nat [] "start_l") (Op.ref .nat [] "end_l")
        (Op.constNat 1) (bsaLoopBodyG C BLOCK_M BLOCK_D BLOCK_N num_heads colStrideH skn svn total_seq_len scale))
        sEntry = some sFinal
      ∧ end_l ≤ final
      ∧ bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
          total_seq_len sqb sqh sqm skb skh skn svb svh svn qStart numKVBlocks gpos Qg Kg Vg Vg2 scale
          s0 numKVBlocks sFinal := by
  obtain ⟨final, sFinal, hExec, hfin, hP⟩ :=
    forRangeDyn_inv (idx := "col_idx_idx")
      (P := fun i st => bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads
        num_layout total_seq_len sqb sqh sqm skb skh skn svb svh svn qStart numKVBlocks gpos Qg Kg Vg Vg2
        scale s0 (i - start_l) st ∧ start_l ≤ i ∧ i ≤ end_l)
      (start := start_l) (stop := end_l) (step := 1)
      hStartOp hStopOp (evalOp_constNat 1 sEntry) (by norm_num)
      ⟨by rw [Nat.sub_self]; exact hInit, le_refl _, by omega⟩
      (fun i st hlt hPi => by
        obtain ⟨hinv, hge, hle⟩ := hPi
        obtain ⟨st', hbody, hinv'⟩ := hstep i st hge hlt hinv
        refine ⟨st', hbody, ?_, ?_, ?_⟩
        · have : i + 1 - start_l = (i - start_l) + 1 := by omega
          rw [this]; exact hinv'
        · omega
        · omega)
  refine ⟨final, sFinal, hExec, hfin, ?_⟩
  obtain ⟨hinv, _, hfle⟩ := hP
  have hfeq : final = end_l := le_antisymm hfle hfin
  have hsub : final - start_l = numKVBlocks := by rw [hfeq]; exact hbound
  rw [hsub] at hinv
  exact hinv

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open BSAMathCausal in
/-- **General PostLoop execution + accumulator readback.** Mirrors `bsaPostLoop_eval`
at symbolic dims. The two masked `out` stores write the running `acc`/`acc2`
(= `bsaOPartial / bsaLPartial` at the full window) at every active lane and
preserve out-of-bounds lanes. Honest side-conditions: output-offset injectivity
(`hinj`) and the two stores hitting disjoint offsets (`hdisj`, since
`out2Offset = outOffset + BLOCK_D`). -/
theorem bsaPostLoop_evalG
    (Out Q K V : RegionName) (R C : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len : Nat)
    (sqb sqh sqm skb skh skn svb svh svn sob soh som : Nat) (scale : ℝ)
    (qStart numKVBlocks : Nat) (gpos : Fin (BLOCK_N * numKVBlocks) → Nat)
    (Qg : TileIndex [BLOCK_M, 2 * BLOCK_D] → ℝ)
    (Kg : TileIndex [BLOCK_N * numKVBlocks, 2 * BLOCK_D] → ℝ)
    (Vg Vg2 : TileIndex [BLOCK_N * numKVBlocks, BLOCK_D] → ℝ)
    (s0 : BlockState) (s : BlockState)
    (hinj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] => outOffset s0 num_heads sob soh som BLOCK_M idx))
    (hdisj : ∀ a b : TileIndex [BLOCK_M, BLOCK_D],
      outOffset s0 num_heads sob soh som BLOCK_M a
        ≠ out2Offset s0 num_heads sob soh som BLOCK_M BLOCK_D b)
    (hinv : bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
      total_seq_len sqb sqh sqm skb skh skn svb svh svn qStart numKVBlocks gpos Qg Kg Vg Vg2 scale
      s0 numKVBlocks s) :
    ∃ sP, stepStmts (bsaPostLoopG Out BLOCK_M BLOCK_D sob soh som) s = some sP
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_D],
          sP.readMem Out (outOffset s0 num_heads sob soh som BLOCK_M idx)
            = if s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
                (bsaOPartial BLOCK_N qStart numKVBlocks gpos Qg Kg Vg scale numKVBlocks idx /
                  bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1)
              else s.readMem Out (outOffset s0 num_heads sob soh som BLOCK_M idx))
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_D],
          sP.readMem Out (out2Offset s0 num_heads sob soh som BLOCK_M BLOCK_D idx)
            = if s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
                (bsaOPartial BLOCK_N qStart numKVBlocks gpos Qg Kg Vg2 scale numKVBlocks idx /
                  bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1)
              else s.readMem Out (out2Offset s0 num_heads sob soh som BLOCK_M BLOCK_D idx)) := by
  obtain ⟨hpids, hmem, hundef, hqsl, hsm, hbh, hoh, hob, hhg, hohkv, hom, hon, hod,
    hlh, hmi, hli, hacc, hacc2, hq, hq2, hkp, hvp⟩ := hinv
  set accFn : TileIndex [BLOCK_M, BLOCK_D] → ℝ := fun idx =>
    bsaOPartial BLOCK_N qStart numKVBlocks gpos Qg Kg Vg scale numKVBlocks idx /
      bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1 with haccFn
  set acc2Fn : TileIndex [BLOCK_M, BLOCK_D] → ℝ := fun idx =>
    bsaOPartial BLOCK_N qStart numKVBlocks gpos Qg Kg Vg2 scale numKVBlocks idx /
      bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1 with hacc2Fn
  set offFn : TileIndex [BLOCK_M, BLOCK_D] → Nat := fun idx =>
    outOffset s0 num_heads sob soh som BLOCK_M idx with hoffFn
  -- out2Offset = offFn + BLOCK_D (definitional)
  have ho2eq : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      out2Offset s0 num_heads sob soh som BLOCK_M BLOCK_D idx = offFn idx + BLOCK_D := by
    intro idx; simp only [hoffFn, outOffset, out2Offset]; omega
  unfold bsaPostLoopG
  -- stmt 0: off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.add NumericDType.nat Broadcast.scalarL
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_b") (Op.constNat sob))
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat soh)))
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat som)))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_D] "offs_d"))) s
        = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] => offFn idx⟩ : Tile .nat [BLOCK_M, BLOCK_D]) from by
      rw [evalOp_add, evalOp_add, evalOp_add, evalOp_mul, evalOp_mul, evalOp_mul]
      erw [evalOp_expandDim_ref_of_regs .nat [BLOCK_M] ⟨1, by simp⟩ "offs_m" _
            (Tile.vec (fun i : Fin BLOCK_M => s0.pids 0 * BLOCK_M + i.val)) hom,
        evalOp_expandDim_ref_of_regs .nat [BLOCK_D] ⟨0, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin BLOCK_D => e.val)) hod]
      simp only [evalOp_ref, hob, hoh, evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [hoffFn, outOffset, offB, offH, mIndex, dIndex, Tile.bop_data, Tile.expandDim_data,
        Tile.vec, Tile.scalar_data_index, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul, TileShape.dropInsertedIndex]
      try ring))]
  set s1 := s.setReg "off_o" .nat [BLOCK_M, BLOCK_D]
    (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] => offFn idx⟩ : Tile .nat [BLOCK_M, BLOCK_D]) with hs1d
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "off_o" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1offo : s1.regs .nat [BLOCK_M, BLOCK_D] "off_o" = some
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] => offFn idx⟩ : Tile .nat [BLOCK_M, BLOCK_D]) := by
    rw [hs1d, BlockState.setReg_same]
  -- stmt 1: out_ptrs = Out + off_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLOCK_M, BLOCK_D] "off_o")) s1
        = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out, offFn idx)⟩ : Tile .ptr [BLOCK_M, BLOCK_D]) from by
      simp only [evalOp, evalOp_ref, hs1offo, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨ir, dd, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  set s2 := s1.setReg "out_ptrs" .ptr [BLOCK_M, BLOCK_D]
    (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out, offFn idx)⟩ : Tile .ptr [BLOCK_M, BLOCK_D]) with hs2d
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "out_ptrs" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2ptr : s2.regs .ptr [BLOCK_M, BLOCK_D] "out_ptrs" = some
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out, offFn idx)⟩ : Tile .ptr [BLOCK_M, BLOCK_D]) := by
    rw [hs2d, BlockState.setReg_same]
  have hs2acc : s2.regs .real [BLOCK_M, BLOCK_D] "acc" = some
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] => (some (accFn idx) : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_D]) :=
    e2 (by decide) (e1 (by decide) hacc)
  have hs2acc2 : s2.regs .real [BLOCK_M, BLOCK_D] "acc2" = some
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] => (some (acc2Fn idx) : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_D]) :=
    e2 (by decide) (e1 (by decide) hacc2)
  have hs2m : s2.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun i : Fin BLOCK_M => s0.pids 0 * BLOCK_M + i.val)) :=
    e2 (by decide) (e1 (by decide) hom)
  have hs2qsl : s2.regs .nat [] "q_seq_len" = some (Tile.scalar total_seq_len) :=
    e2 (by decide) (e1 (by decide) hqsl)
  have hs2mem : s2.mem = s0.mem := by
    funext rg o; rw [hs2d, BlockState.setReg_mem, hs1d, BlockState.setReg_mem]; exact hmem ▸ rfl
  have hexpM : @evalOp .nat [BLOCK_M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) s2
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin BLOCK_M => s0.pids 0 * BLOCK_M + i.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLOCK_M] ⟨1, by simp⟩ "offs_m" s2 _ hs2m
  -- stmt 2: masked store of acc at off_o
  have hstore1 : stepStmt (Stmt.store .real [BLOCK_M, BLOCK_D] (MemAccess.ptr (Op.ref .ptr [BLOCK_M, BLOCK_D] "out_ptrs"))
      (Op.ref .real [BLOCK_M, BLOCK_D] "acc")
      (MaskOpt.mask (Op.remap [BLOCK_M, BLOCK_D] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
          (Op.ref .nat [] "q_seq_len"))))) s2
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_D]).foldl
          (fun acc idx =>
            if s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
              acc.writeMem Out (offFn idx) (accFn idx)
            else acc) s2) := by
    unfold stepStmt
    simp only [evalOp_ref, hs2acc, hs2ptr, hs2qsl, evalOp, hexpM, Option.bind, Option.map]
    refine congrArg some ?_
    congr 1
    funext acc idx
    obtain ⟨ir, dd, u⟩ := idx
    simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim, Tile.vec,
      Tile.scalar, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.dropInsertedIndex, BlockState.writeMemTyped_real, FloatDType.real_storeValue,
      decide_eq_true_eq]
    simp only [show (WithBot.unbotD 0 (some (accFn (ir, dd, u))) : ℝ) = accFn (ir, dd, u) from rfl]
  rw [stepStmts.cons_some hstore1]
  set s3 := (TileShape.allIndices [BLOCK_M, BLOCK_D]).foldl
      (fun acc idx => if s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
        acc.writeMem Out (offFn idx) (accFn idx) else acc) s2 with hs3d
  have hs3ptr : s3.regs .ptr [BLOCK_M, BLOCK_D] "out_ptrs" = some
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out, offFn idx)⟩ : Tile .ptr [BLOCK_M, BLOCK_D]) := by
    rw [hs3d, BlockState.foldl_writeMem_prop_masked_regs]; exact hs2ptr
  have hs3acc2 : s3.regs .real [BLOCK_M, BLOCK_D] "acc2" = some
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_D] => (some (acc2Fn idx) : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_D]) := by
    rw [hs3d, BlockState.foldl_writeMem_prop_masked_regs]; exact hs2acc2
  have hs3m : s3.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun i : Fin BLOCK_M => s0.pids 0 * BLOCK_M + i.val)) := by
    rw [hs3d, BlockState.foldl_writeMem_prop_masked_regs]; exact hs2m
  have hs3qsl : s3.regs .nat [] "q_seq_len" = some (Tile.scalar total_seq_len) := by
    rw [hs3d, BlockState.foldl_writeMem_prop_masked_regs]; exact hs2qsl
  have hexpM3 : @evalOp .nat [BLOCK_M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) s3
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin BLOCK_M => s0.pids 0 * BLOCK_M + i.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLOCK_M] ⟨1, by simp⟩ "offs_m" s3 _ hs3m
  have hstore2inner : stepStmt (Stmt.store .real [BLOCK_M, BLOCK_D]
      (MemAccess.ptr (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_M, BLOCK_D] "out_ptrs") (Op.constNat BLOCK_D)))
      (Op.ref .real [BLOCK_M, BLOCK_D] "acc2")
      (MaskOpt.mask (Op.remap [BLOCK_M, BLOCK_D] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
          (Op.ref .nat [] "q_seq_len"))))) s3
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_D]).foldl
          (fun acc idx =>
            if s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
              acc.writeMem Out (offFn idx + BLOCK_D) (acc2Fn idx)
            else acc) s3) := by
    unfold stepStmt
    simp only [evalOp_ref, hs3acc2, hs3ptr, hs3qsl, evalOp, hexpM3, Option.bind, Option.map]
    refine congrArg some ?_
    congr 1
    funext acc idx
    obtain ⟨ir, dd, u⟩ := idx
    simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim, Tile.vec,
      Tile.scalar, Tile.ptrAdd_data, Tile.scalar_data, ComparableDType.lt, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex, BlockState.writeMemTyped_real,
      FloatDType.real_storeValue, decide_eq_true_eq]
    simp only [show (WithBot.unbotD 0 (some (acc2Fn (ir, dd, u))) : ℝ) = acc2Fn (ir, dd, u) from rfl]
  rw [stepStmts.cons_some (stepStmt_ifThen_true (by simp [evalOp])
    (by rw [stepStmts.cons_some hstore2inner, stepStmts.nil])), stepStmts.nil]
  set s4 := (TileShape.allIndices [BLOCK_M, BLOCK_D]).foldl
      (fun acc idx => if s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
        acc.writeMem Out (offFn idx + BLOCK_D) (acc2Fn idx) else acc) s3 with hs4d
  refine ⟨s4, rfl, ?_, ?_⟩
  · -- first store readback at outOffset
    intro idx
    show s4.readMem Out (offFn idx) = _
    have hs4read : s4.readMem Out (offFn idx) = s3.readMem Out (offFn idx) := by
      rw [hs4d]
      apply bsa_foldl_writeMem_preserves_off
      intro k _hk _hPk
      have hh := hdisj idx k
      simp only [out2Offset, hoffFn, outOffset] at hh ⊢
      omega
    rw [hs4read, hs3d]
    rw [BlockState.scatter_readback_prop_masked_nd s2 offFn accFn
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] => s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len) hinj idx]
    by_cases hlt : s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len
    · rw [if_pos hlt, if_pos hlt]
    · rw [if_neg hlt, if_neg hlt]
      simp only [hoffFn, BlockState.readMem, hs2mem, hmem]
  · -- second store readback at out2Offset
    intro idx
    rw [ho2eq idx, hs4d]
    have hinj2 : Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_D] => offFn idx + BLOCK_D) := by
      intro a b h; exact hinj (by simpa using Nat.add_right_cancel h)
    rw [BlockState.scatter_readback_prop_masked_nd s3 (fun idx => offFn idx + BLOCK_D) acc2Fn
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] => s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len) hinj2 idx]
    by_cases hlt : s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len
    · rw [if_pos hlt, if_pos hlt]
    · rw [if_neg hlt, if_neg hlt]
      rw [hs3d]
      have hpres : ∀ k ∈ TileShape.allIndices [BLOCK_M, BLOCK_D],
          (s0.pids 0 * BLOCK_M + k.1.val < total_seq_len) → offFn k ≠ offFn idx + BLOCK_D := by
        intro k _hk _hPk
        have hh := hdisj k idx
        simp only [out2Offset, hoffFn, outOffset] at hh ⊢
        omega
      rw [bsa_foldl_writeMem_preserves_off offFn accFn
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => s0.pids 0 * BLOCK_M + idx.1.val < total_seq_len)
        (offFn idx + BLOCK_D) _ s2 hpres]
      simp only [BlockState.readMem, hs2mem, hmem]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open BSAMathCausal in
/-- **General full-kernel execution.** Mirrors `bsa_exec` at symbolic dims. From a
clean input state, the whole lowered `block_sparse_attention_kernel` body (at
arbitrary tile dims/strides, `NUM_D_BLOCKS = 2`, `EVEN_M = EVEN_N = true`) runs to
a final state whose two `out` stores hold the streaming online-softmax
accumulators at active lanes and preserve inactive lanes. Composes
`bsaPreLoop_evalG` → `bsa_csr_loopG` → `bsaPostLoop_evalG` over `bsa_body_splitG`.
The CSR schedule (`start_l`/`end_l`, per-block `bsaLoopBodyG` advance) is the
trusted host boundary, supplied as `hstep`. -/
theorem bsa_execG
    (Out Q K V : RegionName) (R C : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len : Nat)
    (rowStrideH colStrideH sqb sqh sqm skb skh skn svb svh svn sob soh som : Nat) (scale : ℝ)
    (qStart numKVBlocks : Nat) (gpos : Fin (BLOCK_N * numKVBlocks) → Nat)
    (Qg : TileIndex [BLOCK_M, 2 * BLOCK_D] → ℝ)
    (Kg : TileIndex [BLOCK_N * numKVBlocks, 2 * BLOCK_D] → ℝ)
    (Vg Vg2 : TileIndex [BLOCK_N * numKVBlocks, BLOCK_D] → ℝ)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (start_l end_l : Nat)
    (hStartL : s.readMemValue .nat R.cast
      (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0) = start_l)
    (hEndL : s.readMemValue .nat R.cast
      (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0 + 1) = end_l)
    (hbound : end_l - start_l = numKVBlocks) (hsle : start_l ≤ end_l)
    (hinj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] => outOffset s num_heads sob soh som BLOCK_M idx))
    (hdisj : ∀ a b : TileIndex [BLOCK_M, BLOCK_D],
      outOffset s num_heads sob soh som BLOCK_M a
        ≠ out2Offset s num_heads sob soh som BLOCK_M BLOCK_D b)
    (hstep : ∀ (i : Nat) (st : BlockState), start_l ≤ i → i < end_l →
      bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
        total_seq_len sqb sqh sqm skb skh skn svb svh svn qStart numKVBlocks gpos Qg Kg Vg Vg2 scale
        s (i - start_l) st →
      ∃ st', stepStmts (bsaLoopBodyG C BLOCK_M BLOCK_D BLOCK_N num_heads colStrideH skn svn total_seq_len scale)
          (st.setReg "col_idx_idx" .nat [] (Tile.scalar i)) = some st'
        ∧ bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
            total_seq_len sqb sqh sqm skb skh skn svb svh svn qStart numKVBlocks gpos Qg Kg Vg Vg2 scale
            s (i - start_l + 1) st') :
    ∃ sF, stepStmts ((block_sparse_attention_kernel Out Q K V R C
        rowStrideH colStrideH num_layout scale sqb sqh sqm skb skh skn svb svh svn sob soh som
        num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D 2 Bool.true Bool.true).toAlgKernel.body) s = some sF
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_D],
          sF.readMem Out (outOffset s num_heads sob soh som BLOCK_M idx)
            = if s.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
                (bsaOPartial BLOCK_N qStart numKVBlocks gpos Qg Kg Vg scale numKVBlocks idx /
                  bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1)
              else s.readMem Out (outOffset s num_heads sob soh som BLOCK_M idx))
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_D],
          sF.readMem Out (out2Offset s num_heads sob soh som BLOCK_M BLOCK_D idx)
            = if s.pids 0 * BLOCK_M + idx.1.val < total_seq_len then
                (bsaOPartial BLOCK_N qStart numKVBlocks gpos Qg Kg Vg2 scale numKVBlocks idx /
                  bsaLPartial BLOCK_N qStart numKVBlocks gpos Qg Kg scale numKVBlocks idx.1)
              else s.readMem Out (out2Offset s num_heads sob soh som BLOCK_M BLOCK_D idx)) := by
  rw [bsa_body_splitG]
  obtain ⟨s0, hpre, hinv0, hsl, hel⟩ :=
    bsaPreLoop_evalG s Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
      total_seq_len rowStrideH sqb sqh sqm skb skh skn svb svh svn qStart numKVBlocks gpos Qg Kg Vg Vg2 scale hundef
  rw [stepStmts.append_some hpre]
  have hStartOp : evalOp (Op.ref .nat [] "start_l") s0 = some (Tile.scalar start_l) := by
    rw [evalOp_ref, hsl, hStartL]
  have hStopOp : evalOp (Op.ref .nat [] "end_l") s0 = some (Tile.scalar end_l) := by
    rw [evalOp_ref, hel, hEndL]
  obtain ⟨final, sL, hloop, _hfin, hinvL⟩ :=
    bsa_csr_loopG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
      total_seq_len sqb sqh sqm skb skh skn svb svh svn colStrideH scale qStart numKVBlocks gpos Qg Kg Vg Vg2
      s start_l end_l hbound hsle s0 hStartOp hStopOp hinv0 hstep
  rw [stepStmts.cons_some hloop]
  obtain ⟨sP, hpost, hOut, hOut2⟩ :=
    bsaPostLoop_evalG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
      total_seq_len sqb sqh sqm skb skh skn svb svh svn sob soh som scale qStart numKVBlocks gpos Qg Kg Vg Vg2
      s sL hinj hdisj hinvL
  rw [hpost]
  obtain ⟨_, hmemL, _⟩ := hinvL
  refine ⟨sP, rfl, ?_, ?_⟩
  · intro idx
    rw [hOut idx]
    by_cases hlt : s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
    · rw [if_pos hlt, if_pos hlt]
    · rw [if_neg hlt, if_neg hlt]
      simp only [BlockState.readMem, hmemL]
  · intro idx
    rw [hOut2 idx]
    by_cases hlt : s.pids 0 * BLOCK_M + idx.1.val < total_seq_len
    · rw [if_pos hlt, if_pos hlt]
    · rw [if_neg hlt, if_neg hlt]
      simp only [BlockState.readMem, hmemL]

set_option maxHeartbeats 4000000 in
/-- **General streaming output = genuine closed form.** Mirrors
`bsa_streaming_eq_closedForm` at symbolic dims: with `Bk = BLOCK_N` keys per CSR
block, contraction `HEAD_DIM = 2·BLOCK_D`, value dim `Dv = BLOCK_D`,
`M = BLOCK_M`, the streaming `bsaOPartial / bsaLPartial` ratio at the full window
equals `blockSparseAttnClosedForm`. Chains `bsaStreaming_eq_bsaAttn`,
`bsaAttn_reindex` (`BLOCK_N·numKVBlocks = numKVBlocks·BLOCK_N`), and
`bsaAttn_eq_blockSparseAttnClosedForm`. `hVis0` (first selected key causally
visible) ensures a nonzero normalizer. -/
theorem bsa_streaming_eq_closedFormG
    (s : BlockState) (Q K V : RegionName) (C : Region .nat)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout : Nat)
    (sqb sqh sqm skb skh skn svb svh svn : Nat) (scale : ℝ)
    (numKVBlocks : Nat) (hBN : 0 < BLOCK_N) (hN : 0 < numKVBlocks) (start_l dBlockBase : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_D])
    (hVis0 : selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N 0
      ≤ s.pids 0 * BLOCK_M + idx.1.val) :
    BSAMathCausal.bsaOPartial BLOCK_N (s.pids 0 * BLOCK_M) numKVBlocks
        (fun r : Fin (BLOCK_N * numKVBlocks) =>
          selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N r.val)
        (fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
          qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val)
        (fun jx : TileIndex [BLOCK_N * numKVBlocks, 2 * BLOCK_D] =>
          kRowBSA s K num_heads num_kv_heads skb skh skn
            (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) jx.2.1.val)
        (fun jx : TileIndex [BLOCK_N * numKVBlocks, BLOCK_D] =>
          vRowBSA s V num_heads num_kv_heads svb svh svn
            (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val)
            (dBlockBase + jx.2.1.val))
        scale numKVBlocks idx /
      BSAMathCausal.bsaLPartial BLOCK_N (s.pids 0 * BLOCK_M) numKVBlocks
        (fun r : Fin (BLOCK_N * numKVBlocks) =>
          selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N r.val)
        (fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
          qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val)
        (fun jx : TileIndex [BLOCK_N * numKVBlocks, 2 * BLOCK_D] =>
          kRowBSA s K num_heads num_kv_heads skb skh skn
            (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) jx.2.1.val)
        scale numKVBlocks idx.1
      = blockSparseAttnClosedForm s Q K V C num_heads num_kv_heads sqb sqh sqm skb skh skn svb svh svn
          (s.pids 1 % num_heads % num_layout) num_heads start_l numKVBlocks (2 * BLOCK_D) BLOCK_M BLOCK_N
          dBlockBase scale idx.1 idx.2.1.val := by
  have h1 := BSAMathCausal.bsaStreaming_eq_bsaAttn (M := BLOCK_M) (D := 2 * BLOCK_D) (Dv := BLOCK_D)
    (Bk := BLOCK_N) hBN (s.pids 0 * BLOCK_M) numKVBlocks hN
    (fun r : Fin (BLOCK_N * numKVBlocks) =>
      selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N r.val)
    (fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
      qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val)
    (fun jx : TileIndex [BLOCK_N * numKVBlocks, 2 * BLOCK_D] =>
      kRowBSA s K num_heads num_kv_heads skb skh skn
        (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) jx.2.1.val)
    (fun jx : TileIndex [BLOCK_N * numKVBlocks, BLOCK_D] =>
      vRowBSA s V num_heads num_kv_heads svb svh svn
        (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val)
        (dBlockBase + jx.2.1.val))
    scale idx hVis0
  rw [h1]
  rw [BSAMathCausal.bsaAttn_reindex (M := BLOCK_M) (D := 2 * BLOCK_D) (Dv := BLOCK_D)
      (Bk₁ := BLOCK_N) (N₁ := numKVBlocks) (Bk₂ := numKVBlocks) (N₂ := BLOCK_N) (Nat.mul_comm BLOCK_N numKVBlocks)
      (s.pids 0 * BLOCK_M)
      (fun n => selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N n)
      (fun n e => kRowBSA s K num_heads num_kv_heads skb skh skn
        (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N n) e.val)
      (fun n d => vRowBSA s V num_heads num_kv_heads svb svh svn
        (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N n) (dBlockBase + d.val))
      (fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
        qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val) scale idx]
  obtain ⟨i, d, u⟩ := idx
  exact BSAMathCausal.bsaAttn_eq_blockSparseAttnClosedForm s Q K V C num_heads num_kv_heads
    sqb sqh sqm skb skh skn svb svh svn (s.pids 1 % num_heads % num_layout) num_heads start_l numKVBlocks
    (2 * BLOCK_D) BLOCK_M BLOCK_N BLOCK_D dBlockBase scale i d

set_option maxHeartbeats 4000000 in
/-- **General (symbolic-dimension) public summary.** At *arbitrary* tile dims and
strides (`NUM_D_BLOCKS = 2`, `EVEN_M = EVEN_N = true`), the full faithful
`block_sparse_attention_kernel` surface (prologue + the CSR `forRangeDyn`
online-softmax loop + the two masked `out` stores) **realizes the genuine causal
block-sparse softmax closed form** `blockSparseAttnClosedForm` at every active
output lane, for both D-blocks (`dBlockBase = 0` / `BLOCK_D`).

This is the symbolic-dimension generalization of the former case-1 Python-shape
summary (the instance `BLOCK_M=BLOCK_N=BLOCK_D=16`, `num_heads=4`,
`num_kv_heads=2`, strides `2048/512/32/1024`, `num_layout=1`, row/col CSR strides
`3`/`4`, `total_seq_len=16`).

Honest side-conditions: clean undef (`hundef`); the CSR row-pointer window
(`hStartL`/`hEndL`/`hsle`) with a nonempty window (`hN`); positivity of the block
size (`hBN`); output-offset injectivity (`hinj`) and the two stores hitting
disjoint offsets (`hdisj`); first-key causal visibility (`hVis0`); and the
per-block selection / load-alignment advance (`hstep`, the trusted host boundary).
This is **not** the self-referential executed value: the streaming
`m_i`/`l_i`/`acc`/`acc2` recurrence is unfolded statement-by-statement
(`bsa_execG`) and proven to collapse to the closed form
(`bsa_streaming_eq_closedFormG`). -/
specification block_sparse_attn_output_closed_form_summary_general
    (Out Q K V : RegionName) (R C : Region .nat) (s : BlockState)
    (BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len : Nat)
    (rowStrideH colStrideH sqb sqh sqm skb skh skn svb svh svn sob soh som : Nat) (scale : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hBN : 0 < BLOCK_N)
    (start_l end_l : Nat)
    (hStartL : s.readMemValue .nat R.cast
      (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0) = start_l)
    (hEndL : s.readMemValue .nat R.cast
      (s.pids 1 % num_heads % num_layout * rowStrideH + s.pids 0 + 1) = end_l)
    (hsle : start_l ≤ end_l) (hN : 0 < end_l - start_l)
    (hinj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] => outOffset s num_heads sob soh som BLOCK_M idx))
    (hdisj : ∀ a b : TileIndex [BLOCK_M, BLOCK_D],
      outOffset s num_heads sob soh som BLOCK_M a
        ≠ out2Offset s num_heads sob soh som BLOCK_M BLOCK_D b)
    (hVis0 : ∀ idx : TileIndex [BLOCK_M, BLOCK_D], active s total_seq_len BLOCK_M idx →
      selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N 0
        ≤ s.pids 0 * BLOCK_M + idx.1.val)
    (hstep : ∀ (i : Nat) (st : BlockState), start_l ≤ i → i < end_l →
      bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
          total_seq_len sqb sqh sqm skb skh skn svb svh svn (s.pids 0 * BLOCK_M) (end_l - start_l)
          (fun r : Fin (BLOCK_N * (end_l - start_l)) =>
            selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N r.val)
          (fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
            qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val)
          (fun jx : TileIndex [BLOCK_N * (end_l - start_l), 2 * BLOCK_D] =>
            kRowBSA s K num_heads num_kv_heads skb skh skn
              (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) jx.2.1.val)
          (fun jx : TileIndex [BLOCK_N * (end_l - start_l), BLOCK_D] =>
            vRowBSA s V num_heads num_kv_heads svb svh svn
              (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) (0 + jx.2.1.val))
          (fun jx : TileIndex [BLOCK_N * (end_l - start_l), BLOCK_D] =>
            vRowBSA s V num_heads num_kv_heads svb svh svn
              (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) (BLOCK_D + jx.2.1.val))
          scale s (i - start_l) st →
      ∃ st', stepStmts (bsaLoopBodyG C BLOCK_M BLOCK_D BLOCK_N num_heads colStrideH skn svn total_seq_len scale)
          (st.setReg "col_idx_idx" .nat [] (Tile.scalar i)) = some st'
        ∧ bsaInvariantG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
            total_seq_len sqb sqh sqm skb skh skn svb svh svn (s.pids 0 * BLOCK_M) (end_l - start_l)
            (fun r : Fin (BLOCK_N * (end_l - start_l)) =>
              selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N r.val)
            (fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
              qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val)
            (fun jx : TileIndex [BLOCK_N * (end_l - start_l), 2 * BLOCK_D] =>
              kRowBSA s K num_heads num_kv_heads skb skh skn
                (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) jx.2.1.val)
            (fun jx : TileIndex [BLOCK_N * (end_l - start_l), BLOCK_D] =>
              vRowBSA s V num_heads num_kv_heads svb svh svn
                (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) (0 + jx.2.1.val))
            (fun jx : TileIndex [BLOCK_N * (end_l - start_l), BLOCK_D] =>
              vRowBSA s V num_heads num_kv_heads svb svh svn
                (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) (BLOCK_D + jx.2.1.val))
            scale s (i - start_l + 1) st') :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := block_sparse_attention_kernel Out Q K V R C
        rowStrideH colStrideH num_layout scale sqb sqh sqm skb skh skn svb svh svn sob soh som
        num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D 2 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => active s total_seq_len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out, outOffset s num_heads sob soh som BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        blockSparseAttnClosedForm s Q K V C num_heads num_kv_heads sqb sqh sqm skb skh skn svb svh svn
          (s.pids 1 % num_heads % num_layout) num_heads start_l (end_l - start_l) (2 * BLOCK_D) BLOCK_M BLOCK_N
          0 scale idx.1 idx.2.1.val)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := block_sparse_attention_kernel Out Q K V R C
        rowStrideH colStrideH num_layout scale sqb sqh sqm skb skh skn svb svh svn sob soh som
        num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D 2 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => active s total_seq_len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_D] => (Out, out2Offset s num_heads sob soh som BLOCK_M BLOCK_D idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        blockSparseAttnClosedForm s Q K V C num_heads num_kv_heads sqb sqh sqm skb skh skn svb svh svn
          (s.pids 1 % num_heads % num_layout) num_heads start_l (end_l - start_l) (2 * BLOCK_D) BLOCK_M BLOCK_N
          BLOCK_D scale idx.1 idx.2.1.val)) := by
  obtain ⟨sF, hstepK, hOut, hOut2⟩ :=
    bsa_execG Out Q K V R C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout total_seq_len
      rowStrideH colStrideH sqb sqh sqm skb skh skn svb svh svn sob soh som scale
      (s.pids 0 * BLOCK_M) (end_l - start_l)
      (fun r : Fin (BLOCK_N * (end_l - start_l)) =>
        selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N r.val)
      (fun jx : TileIndex [BLOCK_M, 2 * BLOCK_D] =>
        qTileBSA s Q num_heads sqb sqh sqm BLOCK_M jx.1 jx.2.1.val)
      (fun jx : TileIndex [BLOCK_N * (end_l - start_l), 2 * BLOCK_D] =>
        kRowBSA s K num_heads num_kv_heads skb skh skn
          (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) jx.2.1.val)
      (fun jx : TileIndex [BLOCK_N * (end_l - start_l), BLOCK_D] =>
        vRowBSA s V num_heads num_kv_heads svb svh svn
          (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) (0 + jx.2.1.val))
      (fun jx : TileIndex [BLOCK_N * (end_l - start_l), BLOCK_D] =>
        vRowBSA s V num_heads num_kv_heads svb svh svn
          (selKeyGlobal s C (s.pids 1 % num_heads % num_layout) num_heads start_l BLOCK_N jx.1.val) (BLOCK_D + jx.2.1.val))
      s hundef start_l end_l hStartL hEndL rfl hsle hinj hdisj hstep
  constructor
  · rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [block_sparse_attention_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx hActive
    rw [show exec _ s = stepStmts _ s from rfl, hstepK] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_real]
    rw [hOut idx, if_pos (show s.pids 0 * BLOCK_M + idx.1.val < total_seq_len from hActive)]
    exact bsa_streaming_eq_closedFormG s Q K V C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
      sqb sqh sqm skb skh skn svb svh svn scale (end_l - start_l) hBN hN start_l 0 idx (hVis0 idx hActive)
  · rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [block_sparse_attention_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx hActive
    rw [show exec _ s = stepStmts _ s from rfl, hstepK] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    simp only [ComputeCorrect.OutputReadable.read_real]
    rw [hOut2 idx, if_pos (show s.pids 0 * BLOCK_M + idx.1.val < total_seq_len from hActive)]
    exact bsa_streaming_eq_closedFormG s Q K V C BLOCK_M BLOCK_D BLOCK_N num_heads num_kv_heads num_layout
      sqb sqh sqm skb skh skn svb svh svn scale (end_l - start_l) hBN hN start_l BLOCK_D idx (hVis0 idx hActive)


/-! ## ════════ `⊨` IO face for the two output-block stores ════════

The summaries above are stated per *declared write map*. This section restates both
output-block stores on the audit-once IO surface
`Masked3DTileKernelIO₁.Implements` (`⊨`), which additionally pins the **flat memory**
placement. Zero new library surface: both slices are masked `[BLOCK_M, BLOCK_D]` tile
copies whose two addresses are built from the two program axes — with `pid₁` split
into `(batch, head)` by `/ num_heads` and `% num_heads`, which the window *functions*
absorb without comment. -/

section IOFace

/-- Cell-level frame of a masked scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, P k → offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k =>
            if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, P k → offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl]
      by_cases hP : P hd
      · rw [if_pos hP, BlockState.writeMem_mem, if_neg ?_]
        rintro ⟨h1, h2⟩
        rcases hc with h | h
        · exact h h1
        · exact h hd List.mem_cons_self hP h2.symm
      · rw [if_neg hP]

theorem out_store_flattenOk (Acc Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) :
    ((block_sparse_attn_output_store_slice Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M
      BLOCK_D).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [block_sparse_attn_output_store_slice, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  and_intros <;> simp [Op.FlattenOk.eq_def]

theorem out_store_terminates (Acc Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) (s : BlockState) :
    ∃ s1, exec (block_sparse_attn_output_store_slice Acc Out num_heads total_seq_len stride_acc_b
      stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D) s = some s1 := by
  simp [exec, block_sparse_attn_output_store_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt, TileShape.dropInsertedIndex]

theorem out_store_frame (Acc Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) (s s' : BlockState)
    (hExec : exec (block_sparse_attn_output_store_slice Acc Out num_heads total_seq_len stride_acc_b
      stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ Out ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
        active s total_seq_len BLOCK_M idx →
        o ≠ outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, block_sparse_attn_output_store_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt, TileShape.dropInsertedIndex] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := Out)
    (fun i : TileIndex [BLOCK_M, BLOCK_D] =>
      s.pids 1 / num_heads * stride_ob + s.pids 1 % num_heads * stride_oh
        + (s.pids 0 * BLOCK_M + i.1.val) * stride_om + i.2.1.val)
    _ (fun i : TileIndex [BLOCK_M, BLOCK_D] =>
      s.pids 0 * BLOCK_M + i.1.val < total_seq_len) r o
    (TileShape.allIndices [BLOCK_M, BLOCK_D]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun idx _ hidx => Ne.symm (h idx hidx)

theorem out_store_traceSafe (Acc Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s total_seq_len BLOCK_M idx →
      accOffset s num_heads stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        BLOCK_M idx < bounds Acc)
    (hout : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s total_seq_len BLOCK_M idx →
      outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx
        < bounds Out) :
    ((block_sparse_attn_output_store_slice Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M
      BLOCK_D).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, block_sparse_attn_output_store_slice, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt,
    MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt,
    MemAccess.MemorySafe, memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt, TileShape.dropInsertedIndex]
  and_intros
  all_goals try exact fun a b ha => hin (a, b, PUnit.unit) ha
  all_goals try exact fun a b ha => hout (a, b, PUnit.unit) ha
  all_goals try (simp [Op.SafeAt.eq_def]; done)

theorem out_store_region_run (Acc Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        outOffset s₀ num_heads stride_ob stride_oh stride_om BLOCK_M idx))
    (xs : TileIndex [BLOCK_M, BLOCK_D] → ℝ)
    (hx : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s₀ total_seq_len BLOCK_M idx →
      s₀.readMem Acc (accOffset s₀ num_heads stride_acc_b stride_acc_h
        stride_acc_m stride_acc_d BLOCK_M idx) = xs idx) :
    ∃ s1, exec (block_sparse_attn_output_store_slice Acc Out num_heads total_seq_len stride_acc_b
        stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
        BLOCK_M BLOCK_D) s₀ = some s1
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_D],
          active s₀ total_seq_len BLOCK_M idx →
          s1.readMem Out (outOffset s₀ num_heads stride_ob stride_oh stride_om
            BLOCK_M idx) = xs idx)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Out ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
            active s₀ total_seq_len BLOCK_M idx →
            o ≠ outOffset s₀ num_heads stride_ob stride_oh stride_om
              BLOCK_M idx) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := out_store_terminates Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D s₀
  refine ⟨s1, hexec, ?_, out_store_frame Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D s₀ s1 hexec⟩
  intro idx hact
  have h := block_sparse_attn_output_store_slice_correct Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D s₀ hOutInj idx
  have h' : s1.readMem Out (outOffset s₀ num_heads stride_ob stride_oh stride_om
        BLOCK_M idx)
      = if active s₀ total_seq_len BLOCK_M idx then
          accStoreValue s₀ Acc num_heads total_seq_len stride_acc_b
            stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
        else s₀.readMem Out (outOffset s₀ num_heads stride_ob stride_oh stride_om
          BLOCK_M idx) := by
    simpa [hexec] using h
  rw [h', if_pos hact, accStoreValue, if_pos hact]
  simpa using hx idx hact

/-- IO signature of `block_sparse_attn_output_store_slice` on the three-axis tile surface. -/
def out_storeIO (Acc Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) : Masked3DTileKernelIO₁ where
  kernel := block_sparse_attn_output_store_slice Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
    stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D
  inp := Acc
  out := Out
  shape := [BLOCK_M, BLOCK_D]
  read := fun p₀ p₁ _p₂ idx =>
    p₁ / num_heads * stride_acc_b + p₁ % num_heads * stride_acc_h
      + (p₀ * BLOCK_M + idx.1.val) * stride_acc_m + idx.2.1.val * stride_acc_d
  write := fun p₀ p₁ _p₂ idx =>
    p₁ / num_heads * stride_ob + p₁ % num_heads * stride_oh
      + (p₀ * BLOCK_M + idx.1.val) * stride_om + idx.2.1.val
  mask := fun p₀ _p₁ _p₂ idx => p₀ * BLOCK_M + idx.1.val < total_seq_len

theorem out2_store_flattenOk (Acc2 Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) :
    ((block_sparse_attn_output_store_second_slice Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M
      BLOCK_D).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [block_sparse_attn_output_store_second_slice, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  and_intros <;> simp [Op.FlattenOk.eq_def]

theorem out2_store_terminates (Acc2 Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) (s : BlockState) :
    ∃ s1, exec (block_sparse_attn_output_store_second_slice Acc2 Out num_heads total_seq_len stride_acc_b
      stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D) s = some s1 := by
  simp [exec, block_sparse_attn_output_store_second_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt, TileShape.dropInsertedIndex]

theorem out2_store_frame (Acc2 Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) (s s' : BlockState)
    (hExec : exec (block_sparse_attn_output_store_second_slice Acc2 Out num_heads total_seq_len stride_acc_b
      stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
      BLOCK_M BLOCK_D) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ Out ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
        active s total_seq_len BLOCK_M idx →
        o ≠ out2Offset s num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, block_sparse_attn_output_store_second_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt, TileShape.dropInsertedIndex] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := Out)
    (fun i : TileIndex [BLOCK_M, BLOCK_D] =>
      s.pids 1 / num_heads * stride_ob + s.pids 1 % num_heads * stride_oh
        + (s.pids 0 * BLOCK_M + i.1.val) * stride_om + BLOCK_D + i.2.1.val)
    _ (fun i : TileIndex [BLOCK_M, BLOCK_D] =>
      s.pids 0 * BLOCK_M + i.1.val < total_seq_len) r o
    (TileShape.allIndices [BLOCK_M, BLOCK_D]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun idx _ hidx => Ne.symm (h idx hidx)

theorem out2_store_traceSafe (Acc2 Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s total_seq_len BLOCK_M idx →
      accOffset s num_heads stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        BLOCK_M idx < bounds Acc2)
    (hout : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s total_seq_len BLOCK_M idx →
      out2Offset s num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx
        < bounds Out) :
    ((block_sparse_attn_output_store_second_slice Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M
      BLOCK_D).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, block_sparse_attn_output_store_second_slice, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt,
    MaskOpt.SafeAt, MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt,
    MemAccess.MemorySafe, memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt, TileShape.dropInsertedIndex]
  and_intros
  all_goals try exact fun a b ha => hin (a, b, PUnit.unit) ha
  all_goals try exact fun a b ha => hout (a, b, PUnit.unit) ha
  all_goals try (simp [Op.SafeAt.eq_def]; done)

theorem out2_store_region_run (Acc2 Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        out2Offset s₀ num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx))
    (xs : TileIndex [BLOCK_M, BLOCK_D] → ℝ)
    (hx : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s₀ total_seq_len BLOCK_M idx →
      s₀.readMem Acc2 (accOffset s₀ num_heads stride_acc_b stride_acc_h
        stride_acc_m stride_acc_d BLOCK_M idx) = xs idx) :
    ∃ s1, exec (block_sparse_attn_output_store_second_slice Acc2 Out num_heads total_seq_len stride_acc_b
        stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
        BLOCK_M BLOCK_D) s₀ = some s1
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_D],
          active s₀ total_seq_len BLOCK_M idx →
          s1.readMem Out (out2Offset s₀ num_heads stride_ob stride_oh stride_om
            BLOCK_M BLOCK_D idx) = xs idx)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Out ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
            active s₀ total_seq_len BLOCK_M idx →
            o ≠ out2Offset s₀ num_heads stride_ob stride_oh stride_om
              BLOCK_M BLOCK_D idx) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := out2_store_terminates Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D s₀
  refine ⟨s1, hexec, ?_, out2_store_frame Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D s₀ s1 hexec⟩
  intro idx hact
  have h := block_sparse_attn_output_store_second_slice_correct Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D s₀ hOutInj idx
  have h' : s1.readMem Out (out2Offset s₀ num_heads stride_ob stride_oh stride_om
        BLOCK_M BLOCK_D idx)
      = if active s₀ total_seq_len BLOCK_M idx then
          accStoreValue s₀ Acc2 num_heads total_seq_len stride_acc_b
            stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx
        else s₀.readMem Out (out2Offset s₀ num_heads stride_ob stride_oh stride_om
          BLOCK_M BLOCK_D idx) := by
    simpa [hexec] using h
  rw [h', if_pos hact, accStoreValue, if_pos hact]
  simpa using hx idx hact

/-- IO signature of `block_sparse_attn_output_store_second_slice` on the three-axis tile surface. -/
def out2_storeIO (Acc2 Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) : Masked3DTileKernelIO₁ where
  kernel := block_sparse_attn_output_store_second_slice Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
    stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D
  inp := Acc2
  out := Out
  shape := [BLOCK_M, BLOCK_D]
  read := fun p₀ p₁ _p₂ idx =>
    p₁ / num_heads * stride_acc_b + p₁ % num_heads * stride_acc_h
      + (p₀ * BLOCK_M + idx.1.val) * stride_acc_m + idx.2.1.val * stride_acc_d
  write := fun p₀ p₁ _p₂ idx =>
    p₁ / num_heads * stride_ob + p₁ % num_heads * stride_oh
      + (p₀ * BLOCK_M + idx.1.val) * stride_om + BLOCK_D + idx.2.1.val
  mask := fun p₀ _p₁ _p₂ idx => p₀ * BLOCK_M + idx.1.val < total_seq_len

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `block_sparse_attn.py`'s two output-block
stores: for every disjoint flat placement of the source and `Out` buffers, every
program coordinate whose active lanes are in bounds, and every launch state whose
accumulator tile holds `xs` at the active lanes, each slice terminates, every active
lane of the output block holds `xs idx`, and every other memory cell is unchanged.

`pid₁` is split into `(batch, head)` by `/ num_heads` and `% num_heads`; the window
*functions* absorb that with no new library surface. The second slice differs only by
the `+ BLOCK_D` column shift of its write window.

Dimension-general in `num_heads`, `total_seq_len`, all seven strides, `BLOCK_M` and
`BLOCK_D`. Honest side-condition: output-address injectivity at every program
coordinate, the same hypothesis the per-write-map summaries take. -/
specification block_sparse_attn_output_stores_io_correctness
    (Acc Acc2 Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat)
    (hInj1 : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        p₁ / num_heads * stride_ob + p₁ % num_heads * stride_oh
          + (p₀ * BLOCK_M + idx.1.val) * stride_om + idx.2.1.val))
    (hInj2 : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        p₁ / num_heads * stride_ob + p₁ % num_heads * stride_oh
          + (p₀ * BLOCK_M + idx.1.val) * stride_om + BLOCK_D + idx.2.1.val)) :
    (out_storeIO Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D
      ⊨ fun _p₀ _p₁ xs idx => xs idx) ∧
    (out2_storeIO Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D
      ⊨ fun _p₀ _p₁ xs idx => xs idx) := by
  constructor
  · refine Masked3DTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
    · exact out_store_flattenOk Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M
          BLOCK_D
    · intro bounds s h1 h2
      exact out_store_traceSafe Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M
          BLOCK_D bounds s
        (fun idx hact => h1 idx hact) (fun idx hact => h2 idx hact)
    · intro s₀ xs hin
      exact out_store_region_run Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M
          BLOCK_D s₀
        (hInj1 (s₀.pids 0) (s₀.pids 1)) xs (fun idx hact => hin idx hact)
  · refine Masked3DTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
    · exact out2_store_flattenOk Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M
          BLOCK_D
    · intro bounds s h1 h2
      exact out2_store_traceSafe Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M
          BLOCK_D bounds s
        (fun idx hact => h1 idx hact) (fun idx hact => h2 idx hact)
    · intro s₀ xs hin
      exact out2_store_region_run Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M
          BLOCK_D s₀
        (hInj2 (s₀.pids 0) (s₀.pids 1)) xs (fun idx hact => hin idx hact)

/-! ### The rounding faces

Both stores are pure copies: no arithmetic on the loaded accumulator tile, and no
`.to(...)` on either store — the tiles stay `.real` end to end. So both slices are
**cast-free** — every statement steps identically under `stepStmtsR R` and
`stepStmts` — and the exact runs transport to `execR R` for *every* rounding
model. -/

/-- The first output store is cast-free: `execR R` is the exact stepper. -/
private theorem out_store_castFree (R : RoundingModel) (Acc Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) (s : BlockState) :
    execR R ((block_sparse_attn_output_store_slice Acc Out num_heads
        total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        stride_ob stride_oh stride_om BLOCK_M BLOCK_D).toAlgKernel) s
      = exec ((block_sparse_attn_output_store_slice Acc Out num_heads
        total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        stride_ob stride_oh stride_om BLOCK_M BLOCK_D).toAlgKernel) s := by
  simp [execR, exec, block_sparse_attn_output_store_slice,
    ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR, evalOpR.eq_def, evalOp,
    evalOp.eq_def, BlockState.writeMemTypedR, BlockState.writeMemAsR,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    Tile.uop, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt, FloatDType.cast,
    TileShape.dropInsertedIndex]

/-- The second output store is cast-free: `execR R` is the exact stepper. -/
private theorem out2_store_castFree (R : RoundingModel) (Acc2 Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat) (s : BlockState) :
    execR R ((block_sparse_attn_output_store_second_slice Acc2 Out num_heads
        total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        stride_ob stride_oh stride_om BLOCK_M BLOCK_D).toAlgKernel) s
      = exec ((block_sparse_attn_output_store_second_slice Acc2 Out num_heads
        total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        stride_ob stride_oh stride_om BLOCK_M BLOCK_D).toAlgKernel) s := by
  simp [execR, exec, block_sparse_attn_output_store_second_slice,
    ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR, evalOpR.eq_def, evalOp,
    evalOp.eq_def, BlockState.writeMemTypedR, BlockState.writeMemAsR,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    Tile.uop, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt, FloatDType.cast,
    TileShape.dropInsertedIndex]

/-- Per-execution safety walk of the first output store **under the rounding
model** — the `hts` obligation of `Masked3DTileKernelIO₁.ImplementsR.intro`. -/
theorem out_store_traceSafeR (R : RoundingModel) (Acc Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s total_seq_len BLOCK_M idx →
      accOffset s num_heads stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        BLOCK_M idx < bounds Acc)
    (hout : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s total_seq_len BLOCK_M idx →
      outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx
        < bounds Out) :
    Kernel.TraceSafeR R bounds
      ((block_sparse_attn_output_store_slice Acc Out num_heads total_seq_len
        stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh
        stride_om BLOCK_M BLOCK_D).toAlgKernel) s := by
  simp only [active, accOffset, outOffset, offB, offH, mIndex, dIndex]
    at hin hout
  unfold Kernel.TraceSafeR
  simp [block_sparse_attn_output_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeListR,
    Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.SafeAtR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR, stepStmtR, evalOpR, evalOpR.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    Tile.uop, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt, FloatDType.cast,
    TileShape.dropInsertedIndex]
  and_intros
  all_goals try exact fun a b ha => hin (a, b, PUnit.unit) ha
  all_goals try exact fun a b ha => hout (a, b, PUnit.unit) ha
  all_goals try simp [Op.SafeAtR.eq_def]

/-- Per-execution safety walk of the second output store **under the rounding
model** — the `hts` obligation of `Masked3DTileKernelIO₁.ImplementsR.intro`. -/
theorem out2_store_traceSafeR (R : RoundingModel) (Acc2 Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s total_seq_len BLOCK_M idx →
      accOffset s num_heads stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        BLOCK_M idx < bounds Acc2)
    (hout : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s total_seq_len BLOCK_M idx →
      out2Offset s num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx
        < bounds Out) :
    Kernel.TraceSafeR R bounds
      ((block_sparse_attn_output_store_second_slice Acc2 Out num_heads
        total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
        stride_ob stride_oh stride_om BLOCK_M BLOCK_D).toAlgKernel) s := by
  simp only [active, accOffset, out2Offset, offB, offH, mIndex, dIndex]
    at hin hout
  unfold Kernel.TraceSafeR
  simp [block_sparse_attn_output_store_second_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeListR,
    Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.SafeAtR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR, stepStmtR, evalOpR, evalOpR.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    Tile.uop, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
    IntegralDType.mod, ComparableDType.lt, FloatDType.cast,
    TileShape.dropInsertedIndex]
  and_intros
  all_goals try exact fun a b ha => hin (a, b, PUnit.unit) ha
  all_goals try exact fun a b ha => hout (a, b, PUnit.unit) ha
  all_goals try simp [Op.SafeAtR.eq_def]

/-! ### ════════ ★ MAIN THEOREM (rounding face) ★ ════════ -/

/-- **The `⊨[R]` headline** for `block_sparse_attn.py`'s two output-block stores:
for **every** rounding model `R`, the same pair of masked Hoare triples as
`block_sparse_attn_output_stores_io_correctness`, but run under `execR R` and read
back as `.real`-typed cells holding `R.round .real (xs idx)`.

Both are pure copies carrying no `.to(...)`, so both slices are cast-free and the
exact runs transport verbatim. The content of the rounding face here is exactly
that: *neither store introduces a rounding event of its own*, at any `R`. -/
specification block_sparse_attn_output_stores_io_correctnessR (R : RoundingModel)
    (Acc Acc2 Out : RegionName)
    (num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om BLOCK_M BLOCK_D : Nat)
    (hInj1 : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        p₁ / num_heads * stride_ob + p₁ % num_heads * stride_oh
          + (p₀ * BLOCK_M + idx.1.val) * stride_om + idx.2.1.val))
    (hInj2 : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        p₁ / num_heads * stride_ob + p₁ % num_heads * stride_oh
          + (p₀ * BLOCK_M + idx.1.val) * stride_om + BLOCK_D + idx.2.1.val)) :
    (out_storeIO Acc Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx) ∧
    (out2_storeIO Acc2 Out num_heads total_seq_len stride_acc_b stride_acc_h
      stride_acc_m stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx) := by
  constructor
  · refine Masked3DTileKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
    · exact out_store_flattenOk Acc Out num_heads total_seq_len stride_acc_b
        stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
        BLOCK_M BLOCK_D
    · intro bounds s h1 h2
      exact out_store_traceSafeR R Acc Out num_heads total_seq_len stride_acc_b
        stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
        BLOCK_M BLOCK_D bounds s
        (fun idx hact => h1 idx hact) (fun idx hact => h2 idx hact)
    · intro s₀ xs hx
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        out_store_region_run Acc Out num_heads total_seq_len stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
          BLOCK_M BLOCK_D s₀ (hInj1 (s₀.pids 0) (s₀.pids 1)) xs
          (fun idx hact => hx idx hact)
      refine ⟨s1, ?_, ?_, hframe⟩
      · simp only [out_storeIO]
        rw [out_store_castFree R Acc Out num_heads total_seq_len stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
          BLOCK_M BLOCK_D s₀]
        exact hexec
      · intro idx hidx
        simp only [out_storeIO]
        rw [BlockState.readMemAs_real]
        have := hval idx hidx
        simp only [outOffset, offB, offH, mIndex, dIndex] at this
        rw [this]
        simp
  · refine Masked3DTileKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
    · exact out2_store_flattenOk Acc2 Out num_heads total_seq_len stride_acc_b
        stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
        BLOCK_M BLOCK_D
    · intro bounds s h1 h2
      exact out2_store_traceSafeR R Acc2 Out num_heads total_seq_len stride_acc_b
        stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
        BLOCK_M BLOCK_D bounds s
        (fun idx hact => h1 idx hact) (fun idx hact => h2 idx hact)
    · intro s₀ xs hx
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        out2_store_region_run Acc2 Out num_heads total_seq_len stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
          BLOCK_M BLOCK_D s₀ (hInj2 (s₀.pids 0) (s₀.pids 1)) xs
          (fun idx hact => hx idx hact)
      refine ⟨s1, ?_, ?_, hframe⟩
      · simp only [out2_storeIO]
        rw [out2_store_castFree R Acc2 Out num_heads total_seq_len stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d stride_ob stride_oh stride_om
          BLOCK_M BLOCK_D s₀]
        exact hexec
      · intro idx hidx
        simp only [out2_storeIO]
        rw [BlockState.readMemAs_real]
        have := hval idx hidx
        simp only [out2Offset, offB, offH, mIndex, dIndex] at this
        rw [this]
        simp

end IOFace

end VeriTile.Bench.TritonBenchG.BlockSparseAttn

