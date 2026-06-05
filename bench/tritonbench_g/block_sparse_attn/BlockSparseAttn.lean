import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.StreamingAccumulator
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
block_sparse_attn_python_case1_output_summary / _case2_output_summary   ← TOP (abbrev, EVEN_M/N = true / false)
  ├─ block_sparse_attn_case{1,2}_surface_first_output_compute_correct    first D-block store of the full surface
  │    └─ block_sparse_attn_python_first_output_compute_correct
  │         └─ block_sparse_attn_output_store_slice_compute_correct
  │              └─ block_sparse_attn_output_store_slice_correct          algorithm-layer readback per lane
  └─ block_sparse_attn_case{1,2}_surface_second_output_compute_correct   second D-block store (+BLOCK_D)
       └─ block_sparse_attn_python_second_output_compute_correct
            └─ block_sparse_attn_output_store_second_slice_compute_correct
                 └─ block_sparse_attn_output_store_second_slice_correct
```
(Offset injectivity discharged by `block_sparse_attn_python_{first,second}_output_offset_injective`.)

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; `exp`, `tl.dot`, and the
`softmax_scale` multiply are not modeled at the bit level); `@triton.autotune`
is not modeled. The verified result is **final-store scoped**: the proof
establishes that the two masked `out` stores write the accumulator slices
`Acc`/`Acc2` at the correct, injective output offsets and preserve inactive
lanes — the written value is `producedBlockSparseAttnCase{1,2}Out{,2}Value` /
`accStoreValue` `Acc`, an opaque carrier for the online-softmax + sparse-mask
recurrence, which is **not** re-derived as a closed-form attention formula here.
The sparse-mask schedule itself (which key blocks the CSR loop visits) lives in
that opaque carrier and the trusted host boundary. Side conditions: the
test-shape summaries fix the concrete layout `(B,H,M,D) = (2,4,16,32)` with
`NUM_D_BLOCKS = 2`; case 1 uses `EVEN_M = EVEN_N = true`, case 2 uses
`EVEN_M = EVEN_N = false`.

## Genuine closed-form spec (banked)

`blockSparseAttnClosedForm` is a **genuine, non-self-referential** closed form
for one program's output: the causal natural-exp softmax attention
(`tl.exp`, not `exp2`) of the program's query tile against the key/value rows the
CSR layout selects (`selKeyGlobal`), under grouped-query head mapping
(`offHkv`). The store recipes `block_sparse_attn_{first,second}_output_closed_form`
prove **sorry-free** that the two masked `out` stores write this closed form to
`Out` — *given* the loop-fill contract `hFill` (the accumulator regions hold the
closed form at each active lane). The remaining proof gap is exactly `hFill`:
the online-softmax recurrence over the CSR-selected, causally-masked key blocks
equals the batch causal softmax. That `exec`-side loop unfolding (data-dependent
CSR bounds + per-step rescale + two D-blocks) is far larger than the existing
non-sparse `AttentionForwardClosedForm` assembly and is **not** discharged here;
the self-referential `produced…Value` summaries are retained as the surface
record until it is. This is tracked alongside the other sparse-attention loop
gaps (cf. `mixed_sparse_attention`).
-/

namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn

open VeriTile.Triton

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
noncomputable def bsaOPartial {M D : Nat} (Bk : Nat)
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg Vg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ) :
    Nat → TileIndex [M, D] → ℝ
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
theorem bsaOPartial_succ_of_lt {M D Bk : Nat}
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg Vg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k < numKVBlocks) (idx : TileIndex [M, D]) :
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
noncomputable def oFree {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg Vg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k ≤ N)
    (idx : TileIndex [M, D]) : ℝ :=
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

@[simp] theorem oFree_zero {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg Vg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
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
theorem oFree_succ {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg Vg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k + 1 ≤ N)
    (idx : TileIndex [M, D]) :
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
theorem oFree_eq_flat {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg Vg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
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
theorem bsaOPartial_eq_mShifted {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat) (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg Vg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D])
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
noncomputable def bsaAttn {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg Vg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) : TileIndex [M, D] → ℝ :=
  fun (i, d, _) =>
    let weight := fun j : Fin (Bk * N) =>
      if gpos j ≤ qStart + i.val then Real.exp (gScore Q Kg scale i j) else 0
    let denom := Finset.univ.sum (fun j => weight j)
    let numer := Finset.univ.sum (fun j => weight j * Vg (j, d, PUnit.unit))
    numer / denom

/-- Gathered m-free ratio is exactly the gathered closed-form attention. -/
theorem oFree_div_lFree_eq_bsaAttn {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg Vg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (idx : TileIndex [M, D]) :
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

/-- If the gathered m-free normalizer `lFree` vanishes, so does the m-free output
`oFree` (no causally-visible key is selected, so every output summand carries a
zero weight). -/
theorem oFree_eq_zero_of_lFree_eq_zero {M D Bk N : Nat}
    (qStart : Nat) (gpos : Fin (Bk * N) → Nat)
    (Q : TileIndex [M, D] → ℝ) (Kg Vg : TileIndex [Bk * N, D] → ℝ)
    (scale : ℝ) (k : Nat) (hk : k ≤ N) (idx : TileIndex [M, D])
    (hl : lFree qStart gpos Q Kg scale k hk idx.1 = 0) :
    oFree qStart gpos Q Kg Vg scale k hk idx = 0 := by
  -- every summand of `lFree` is nonneg, so the sum is 0 ⟹ each is 0 ⟹ each weight 0
  unfold lFree at hl
  unfold oFree
  rw [Finset.sum_eq_zero]
  intro n _
  rw [Finset.sum_eq_zero]
  intro jL _
  -- the corresponding `lFree` term is 0; deduce the weight is 0
  have hterm : (fun jL : Fin Bk =>
      let j := StreamingAccumulator.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL
      if gpos j ≤ qStart + idx.1.val then Real.exp (gScore Q Kg scale idx.1 j) else 0) jL = 0 := by
    have hnonneg_inner : ∀ m : Fin k, (0 : ℝ) ≤ Finset.univ.sum (fun jL : Fin Bk =>
        let j := StreamingAccumulator.blockIndex Bk N m.val (Nat.lt_of_lt_of_le m.isLt hk) jL
        if gpos j ≤ qStart + idx.1.val then Real.exp (gScore Q Kg scale idx.1 j) else 0) := by
      intro m; apply Finset.sum_nonneg; intro jL _
      by_cases h : gpos (StreamingAccumulator.blockIndex Bk N m.val (Nat.lt_of_lt_of_le m.isLt hk) jL) ≤ qStart + idx.1.val
      · simp only [h, if_true]; exact le_of_lt (Real.exp_pos _)
      · simp only [h, if_false]; exact le_refl 0
    have houter := (Finset.sum_eq_zero_iff_of_nonneg (fun m _ => hnonneg_inner m)).mp hl n (Finset.mem_univ _)
    have hinner_nonneg : ∀ jL : Fin Bk, (0 : ℝ) ≤
        (let j := StreamingAccumulator.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL
         if gpos j ≤ qStart + idx.1.val then Real.exp (gScore Q Kg scale idx.1 j) else 0) := by
      intro jL
      by_cases h : gpos (StreamingAccumulator.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL) ≤ qStart + idx.1.val
      · simp only [h, if_true]; exact le_of_lt (Real.exp_pos _)
      · simp only [h, if_false]; exact le_refl 0
    exact (Finset.sum_eq_zero_iff_of_nonneg (fun jL _ => hinner_nonneg jL)).mp houter jL (Finset.mem_univ _)
  -- so the output summand (weight · V) is 0
  simp only [] at hterm
  show (if gpos (StreamingAccumulator.blockIndex Bk N n.val (Nat.lt_of_lt_of_le n.isLt hk) jL) ≤ qStart + idx.1.val
      then Real.exp (gScore Q Kg scale idx.1 _) else 0) * Vg _ = 0
  rw [hterm, zero_mul]

/-- The running ratio is well-defined under cancellation: `(O_c / L_c) · L_c = O_c`,
even when `L_c = 0` (then `O_c = 0` too, by `oFree_eq_zero_of_lFree_eq_zero`). This
is the load-bearing fact making the FA2 normalized acc-rescale advance the
`O / L` invariant. -/
theorem bsaOPartial_div_mul_self {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat) (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg Vg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (hk : k ≤ numKVBlocks) (idx : TileIndex [M, D])
    (hVis0 : gpos ⟨0, Nat.mul_pos hBk hN⟩ ≤ qStart + idx.1.val) :
    bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale k idx /
        bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1 *
        bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1
      = bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale k idx := by
  by_cases hL : bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1 = 0
  · rw [hL, div_zero, mul_zero]
    -- L = 0 ⟹ lFree = 0 (exp(-m) ≠ 0) ⟹ oFree = 0 ⟹ O = 0
    rw [bsaLPartial_eq_mShifted hBk qStart numKVBlocks hN gpos Q Kg scale k hk idx.1 hVis0] at hL
    have hlFree : lFree qStart gpos Q Kg scale k hk idx.1 = 0 := by
      rcases mul_eq_zero.mp hL with h | h
      · exact absurd h (Real.exp_ne_zero _)
      · exact h
    rw [bsaOPartial_eq_mShifted hBk qStart numKVBlocks hN gpos Q Kg Vg scale k hk idx hVis0,
        oFree_eq_zero_of_lFree_eq_zero qStart gpos Q Kg Vg scale k hk idx hlFree, mul_zero]
  · rw [div_mul_cancel₀ _ hL]

/-- **FA2 normalized acc-rescale step (real arithmetic).** Given the running ratio
`Rc` with `Rc · Lc = Oc`, the new normalizer `Lnew = α·Lc + Σ w` (nonzero), and the
new unnormalized output `Onew = α·Oc + Σ w·v`, the kernel's normalized update
`Rc·(Lc/Lnew·α) + Σ (w/Lnew)·v` equals the new ratio `Onew/Lnew`. -/
theorem fa2_acc_step_real {ι : Type*} [Fintype ι]
    (Rc Lc Oc α Lnew : ℝ) (w v : ι → ℝ)
    (hRO : Rc * Lc = Oc)
    (hLnew : Lnew = α * Lc + Finset.univ.sum w)
    (hLnz : Lnew ≠ 0) :
    Rc * (Lc / Lnew * α) + Finset.univ.sum (fun j => w j / Lnew * v j)
      = (α * Oc + Finset.univ.sum (fun j => w j * v j)) / Lnew := by
  rw [add_div, ← hRO]
  rw [Finset.sum_div]
  congr 1
  · field_simp
  · refine Finset.sum_congr rfl (fun j _ => ?_)
    field_simp

/-- **Load-bearing streaming-eq bridge.** The final gathered-causal streaming
ratio equals the gathered closed-form attention `bsaAttn` — i.e. the online
softmax over the CSR-selected (non-contiguous, global-causal) key stream computes
exactly the closed-form attention. This is the block-sparse analog of
`FA1MathCausal.streaming_eq_attentionRealCausalBlock`. -/
theorem bsaStreaming_eq_bsaAttn {M D Bk : Nat} (hBk : 0 < Bk)
    (qStart : Nat) (numKVBlocks : Nat) (hN : 0 < numKVBlocks)
    (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg Vg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (idx : TileIndex [M, D])
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
      HEAD_DIM BLOCK_M BLOCK_N dBlockBase : Nat)
    (softmax_scale : ℝ)
    (i : Fin BLOCK_M) (d : Fin HEAD_DIM) :
    BSAMathCausal.bsaAttn (M := BLOCK_M) (D := HEAD_DIM)
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

/-- `realExp x` is always `some` (it is `0` on `⊥`), so it round-trips through
`unbotD 0`. Mirror of `FA1MathCausal.realExp_eq_some_unbotD`. -/
theorem realExp_eq_some_unbotD (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

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
    ComputeCorrect.Realizes
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
    ComputeCorrect.Realizes
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

theorem block_sparse_attn_python_first_output_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [16, 16] => outOffset s 4 2048 512 32 16 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, offB, offH, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem block_sparse_attn_python_second_output_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [16, 16] => out2Offset s 4 2048 512 32 16 16 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [out2Offset, offB, offH, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem block_sparse_attn_python_first_output_compute_correct
    (Acc Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attn_output_store_slice Acc Out 4 16
        2048 512 16 1 2048 512 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Out, outOffset s 4 2048 512 32 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        accStoreValue s Acc 4 16 2048 512 16 1 16 idx) := by
  exact block_sparse_attn_output_store_slice_compute_correct Acc Out
    4 16 2048 512 16 1 2048 512 32 16 16 s
    (block_sparse_attn_python_first_output_offset_injective s)

theorem block_sparse_attn_python_second_output_compute_correct
    (Acc2 Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attn_output_store_second_slice Acc2 Out 4 16
        2048 512 16 1 2048 512 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] =>
          (Out, out2Offset s 4 2048 512 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        accStoreValue s Acc2 4 16 2048 512 16 1 16 idx) := by
  exact block_sparse_attn_output_store_second_slice_compute_correct Acc2 Out
    4 16 2048 512 16 1 2048 512 32 16 16 s
    (block_sparse_attn_python_second_output_offset_injective s)

/-! ## Genuine closed-form store recipes (sorry-free)

These bridge the final masked `out` stores to the **genuine closed-form**
attention value `blockSparseAttnClosedForm` (NOT the kernel's executed value):
*given* that the accumulator region holds the closed form at each active lane
(the loop-fill contract, which the trusted online-softmax recurrence
establishes), the store-slice surface writes the closed-form attention output to
`Out`. The remaining proof gap is exactly the loop-fill contract `hFill`; the
store-side composition is closed here.

`hFill` is the per-lane statement that the first/second D-block accumulator
equals the causal natural-exp softmax attention over the CSR-selected keys, with
`dBlockBase = 0` for `acc` and `dBlockBase = BLOCK_D` for `acc2`. -/

/-- First D-block store writes the genuine closed-form attention output to `Out`,
given the accumulator `Acc` holds it (loop-fill contract `hFill`). -/
theorem block_sparse_attn_first_output_closed_form
    (Acc Out Q K V : RegionName) (layoutCols : Region .nat)
    (num_heads num_kv_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
      stride_vb stride_vh stride_vn
      layout_h stride_col start_l numSelBlocks
      HEAD_DIM BLOCK_M BLOCK_D : Nat)
    (softmax_scale : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        outOffset s num_heads stride_ob stride_oh stride_om BLOCK_M idx))
    (hFill : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s total_seq_len BLOCK_M idx →
      s.readMem Acc (accOffset s num_heads stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d BLOCK_M idx)
        = blockSparseAttnClosedForm s Q K V layoutCols num_heads num_kv_heads
            stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
            stride_vb stride_vh stride_vn layout_h stride_col start_l numSelBlocks
            HEAD_DIM BLOCK_M (stride_kn) 0 softmax_scale idx.1 (dIndex idx)) :
    ComputeCorrect.Realizes
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
        blockSparseAttnClosedForm s Q K V layoutCols num_heads num_kv_heads
          stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
          stride_vb stride_vh stride_vn layout_h stride_col start_l numSelBlocks
          HEAD_DIM BLOCK_M (stride_kn) 0 softmax_scale idx.1 (dIndex idx)) := by
  have hbase := block_sparse_attn_output_store_slice_compute_correct Acc Out
    num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m stride_acc_d
    stride_ob stride_oh stride_om BLOCK_M BLOCK_D s hOutInj
  rw [ComputeCorrect.realizes_writeIf_iff] at hbase ⊢
  refine ⟨hbase.1, ?_⟩
  intro s0 s' hExec hs0 idx hActive
  have h := hbase.2 s0 s' hExec hs0 idx hActive
  rw [h]
  rw [accStoreValue, if_pos hActive]
  simpa using hFill idx hActive

/-- Second D-block store writes the genuine closed-form attention output to
`Out` at the `+BLOCK_D` channels, given `Acc2` holds it (loop-fill contract). -/
theorem block_sparse_attn_second_output_closed_form
    (Acc2 Out Q K V : RegionName) (layoutCols : Region .nat)
    (num_heads num_kv_heads total_seq_len
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_ob stride_oh stride_om
      stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
      stride_vb stride_vh stride_vn
      layout_h stride_col start_l numSelBlocks
      HEAD_DIM BLOCK_M BLOCK_D : Nat)
    (softmax_scale : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_D] =>
        out2Offset s num_heads stride_ob stride_oh stride_om BLOCK_M BLOCK_D idx))
    (hFill : ∀ idx : TileIndex [BLOCK_M, BLOCK_D],
      active s total_seq_len BLOCK_M idx →
      s.readMem Acc2 (accOffset s num_heads stride_acc_b stride_acc_h
          stride_acc_m stride_acc_d BLOCK_M idx)
        = blockSparseAttnClosedForm s Q K V layoutCols num_heads num_kv_heads
            stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
            stride_vb stride_vh stride_vn layout_h stride_col start_l numSelBlocks
            HEAD_DIM BLOCK_M (stride_kn) BLOCK_D softmax_scale idx.1 (dIndex idx)) :
    ComputeCorrect.Realizes
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
        blockSparseAttnClosedForm s Q K V layoutCols num_heads num_kv_heads
          stride_qb stride_qh stride_qm stride_kb stride_kh stride_kn
          stride_vb stride_vh stride_vn layout_h stride_col start_l numSelBlocks
          HEAD_DIM BLOCK_M (stride_kn) BLOCK_D softmax_scale idx.1 (dIndex idx)) := by
  have hbase := block_sparse_attn_output_store_second_slice_compute_correct Acc2
    Out num_heads total_seq_len stride_acc_b stride_acc_h stride_acc_m
    stride_acc_d stride_ob stride_oh stride_om BLOCK_M BLOCK_D s hOutInj
  rw [ComputeCorrect.realizes_writeIf_iff] at hbase ⊢
  refine ⟨hbase.1, ?_⟩
  intro s0 s' hExec hs0 idx hActive
  have h := hbase.2 s0 s' hExec hs0 idx hActive
  rw [h]
  rw [accStoreValue, if_pos hActive]
  simpa using hFill idx hActive

noncomputable def producedBlockSparseAttnCase1OutValue
    (s : BlockState) (Out Q K V : RegionName)
    (layoutRows layoutCols : Region .nat) (idx : TileIndex [16, 16]) : ℝ :=
  match exec (block_sparse_attention_kernel Out Q K V layoutRows layoutCols
      3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
      4 2 16 16 16 16 2 Bool.true Bool.true).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s 4 2048 512 32 16 idx)
  | none => 0.0

noncomputable def producedBlockSparseAttnCase1Out2Value
    (s : BlockState) (Out Q K V : RegionName)
    (layoutRows layoutCols : Region .nat) (idx : TileIndex [16, 16]) : ℝ :=
  match exec (block_sparse_attention_kernel Out Q K V layoutRows layoutCols
      3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
      4 2 16 16 16 16 2 Bool.true Bool.true).toAlgKernel s with
  | some s' => s'.readMem Out (out2Offset s 4 2048 512 32 16 16 idx)
  | none => 0.0

theorem block_sparse_attn_case1_surface_first_output_compute_correct
    (Out Q K V : RegionName) (layoutRows layoutCols : Region .nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attention_kernel Out Q K V layoutRows layoutCols
        3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
        4 2 16 16 16 16 2 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Out,
          outOffset s 4 2048 512 32 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        producedBlockSparseAttnCase1OutValue s Out Q K V layoutRows
          layoutCols idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [ComputeKernel.toAlgKernel, block_sparse_attention_kernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedBlockSparseAttnCase1OutValue, hExec]

theorem block_sparse_attn_case1_surface_second_output_compute_correct
    (Out Q K V : RegionName) (layoutRows layoutCols : Region .nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attention_kernel Out Q K V layoutRows layoutCols
        3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
        4 2 16 16 16 16 2 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Out,
          out2Offset s 4 2048 512 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        producedBlockSparseAttnCase1Out2Value s Out Q K V layoutRows
          layoutCols idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [ComputeKernel.toAlgKernel, block_sparse_attention_kernel,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedBlockSparseAttnCase1Out2Value, hExec]

noncomputable def producedBlockSparseAttnCase2OutValue
    (s : BlockState) (Out Q K V : RegionName)
    (layoutRows layoutCols : Region .nat) (idx : TileIndex [16, 16]) : ℝ :=
  match exec (block_sparse_attention_kernel Out Q K V layoutRows layoutCols
      3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
      4 2 16 16 16 16 2 Bool.false Bool.false).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s 4 2048 512 32 16 idx)
  | none => 0.0

noncomputable def producedBlockSparseAttnCase2Out2Value
    (s : BlockState) (Out Q K V : RegionName)
    (layoutRows layoutCols : Region .nat) (idx : TileIndex [16, 16]) : ℝ :=
  match exec (block_sparse_attention_kernel Out Q K V layoutRows layoutCols
      3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
      4 2 16 16 16 16 2 Bool.false Bool.false).toAlgKernel s with
  | some s' => s'.readMem Out (out2Offset s 4 2048 512 32 16 16 idx)
  | none => 0.0

theorem block_sparse_attn_case2_surface_first_output_compute_correct
    (Out Q K V : RegionName) (layoutRows layoutCols : Region .nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attention_kernel Out Q K V layoutRows layoutCols
        3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
        4 2 16 16 16 16 2 Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Out,
          outOffset s 4 2048 512 32 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        producedBlockSparseAttnCase2OutValue s Out Q K V layoutRows
          layoutCols idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [block_sparse_attention_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedBlockSparseAttnCase2OutValue, hExec]

theorem block_sparse_attn_case2_surface_second_output_compute_correct
    (Out Q K V : RegionName) (layoutRows layoutCols : Region .nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := block_sparse_attention_kernel Out Q K V layoutRows layoutCols
        3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
        4 2 16 16 16 16 2 Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Out,
          out2Offset s 4 2048 512 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        producedBlockSparseAttnCase2Out2Value s Out Q K V layoutRows
          layoutCols idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [block_sparse_attention_kernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedBlockSparseAttnCase2Out2Value, hExec]

/-- Legacy output-pair summary for the checked `(B,H,M,D)=(2,4,16,32)`
layout. This factors the two observable D-block stores when the sparse
attention loop's accumulators are supplied as precomputed inputs. -/
theorem block_sparse_attn_python_output_pair_compute_correct
    (Out Acc Acc2 : RegionName) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := block_sparse_attn_output_store_slice Acc Out 4 16
        2048 512 16 1 2048 512 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Out, outOffset s 4 2048 512 32 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        accStoreValue s Acc 4 16 2048 512 16 1 16 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := block_sparse_attn_output_store_second_slice Acc2 Out 4 16
        2048 512 16 1 2048 512 32 16 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] =>
          (Out, out2Offset s 4 2048 512 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        accStoreValue s Acc2 4 16 2048 512 16 1 16 idx)) := by
  constructor
  · exact block_sparse_attn_python_first_output_compute_correct Acc Out s
  · exact block_sparse_attn_python_second_output_compute_correct Acc2 Out s

/-- `test_case_1` in `block_sparse_attn.py` uses `EVEN_M = true` and
`EVEN_N = true`. The checked output surface is the full sparse-attention kernel
for the Python shape `(B,H,M,D) = (2,4,16,32)`, observed through its two final
D-block stores. -/
abbrev block_sparse_attn_python_case1_output_summary
    (Out Q K V : RegionName) (layoutRows layoutCols : Region .nat) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := block_sparse_attention_kernel Out Q K V layoutRows layoutCols
        3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
        4 2 16 16 16 16 2 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Out, outOffset s 4 2048 512 32 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        producedBlockSparseAttnCase1OutValue s Out Q K V layoutRows
          layoutCols idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := block_sparse_attention_kernel Out Q K V layoutRows layoutCols
        3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
        4 2 16 16 16 16 2 Bool.true Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] =>
          (Out, out2Offset s 4 2048 512 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        producedBlockSparseAttnCase1Out2Value s Out Q K V layoutRows
          layoutCols idx)) :=
  ⟨block_sparse_attn_case1_surface_first_output_compute_correct Out Q K V
      layoutRows layoutCols s,
    block_sparse_attn_case1_surface_second_output_compute_correct Out Q K V
      layoutRows layoutCols s⟩

/-- `test_case_2` in `block_sparse_attn.py` flips `EVEN_M = false` and
`EVEN_N = false`. The checked output surface is the full sparse-attention kernel
for the Python shape `(B,H,M,D) = (2,4,16,32)`, observed through its two final
D-block stores. -/
abbrev block_sparse_attn_python_case2_output_summary
    (Out Q K V : RegionName) (layoutRows layoutCols : Region .nat) (s : BlockState) :
    (ComputeCorrect.Realizes
      (kernel := block_sparse_attention_kernel Out Q K V layoutRows layoutCols
        3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
        4 2 16 16 16 16 2 Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] => (Out, outOffset s 4 2048 512 32 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        producedBlockSparseAttnCase2OutValue s Out Q K V layoutRows
          layoutCols idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := block_sparse_attention_kernel Out Q K V layoutRows layoutCols
        3 4 1 1.0 2048 512 32 1024 512 32 1024 512 32 2048 512 32
        4 2 16 16 16 16 2 Bool.false Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 16] => active s 16 16 idx)
        (fun idx : TileIndex [16, 16] =>
          (Out, out2Offset s 4 2048 512 32 16 16 idx)))
      (expected := fun idx : TileIndex [16, 16] =>
        producedBlockSparseAttnCase2Out2Value s Out Q K V layoutRows
          layoutCols idx)) :=
  ⟨block_sparse_attn_case2_surface_first_output_compute_correct Out Q K V
      layoutRows layoutCols s,
    block_sparse_attn_case2_surface_second_output_compute_correct Out Q K V
      layoutRows layoutCols s⟩

/-! ## Per-statement op-eval recipes for the CSR loop body (RECIPE LAYER)

These `bsa_*_eval` lemmas are the block-sparse-attention analogues of the
`flash_*_op_eval` / `ta_*_eval` family: each proves that one CSR-loop-body
statement's algorithm-layer `Op` evaluates (under abstract register-readback
hypotheses on a symbolic `BlockState`) to a symbolic result tile. They are the
banked building blocks for the (separate, NEXT-stage) loop invariant/step/
assembly proof; nothing here unfolds the data-dependent `forRangeDyn`.

The exact CSR loop body — `forRangeDyn "col_idx_idx" (ref "start_l")
(ref "end_l") (constNat 1)` — has statement order (extracted from the elaborated
`block_sparse_attention_kernel` AST):

1.  `col_idx`   = load .nat region C at `layout_h·1 + col_idx_idx`           (gather)
2.  `start_n`   = `col_idx · BLOCK_N`
3.  `k`         = ptr-load `k_ptrs + start_n·stride_kn`         (EVEN_N gated mask)
4.  `qk`        = zeros `[BLOCK_M, BLOCK_N]`
5.  `qk`       += dot(q, k)
6.  if D≥2:     `k` = ptr-load `… + BLOCK_D`; `qk += dot(q2, k)`
7.  `qk`       *= softmax_scale
8.  `qk`       += where(offs_m[:,None] ≥ start_n + offs_n[None,:], 0, -inf)
9.  `m_ij`      = reduceMax(qk, axis 1)
10. `p`         = exp(qk - m_ij[:,None])                               (natural exp)
11. `l_ij`      = reduceSum(p, axis 1)
12. `m_i_new`   = where(m_i > m_ij, m_i, m_ij)
13. `alpha`     = exp(m_i - m_i_new)                                  (natural exp)
14. `beta`      = exp(m_ij - m_i_new)                                 (natural exp)
15. `l_i_new`   = alpha·l_i + beta·l_ij
16. `p_scale`   = beta / l_i_new
17. `p`         = p · p_scale[:,None]
18. `acc_scale` = (l_i / l_i_new) · alpha
19. `acc`       = acc · acc_scale[:,None]
20. if D≥2:     `acc2` = acc2 · acc_scale[:,None]
21. `p`         = p                                       (`.to(Q.dtype)`, identity)
22. `v`         = ptr-load `v_ptrs + start_n·stride_vn`        (EVEN_N gated mask)
23. `acc`       = acc + dot(p, v)
24. if D≥2:     `v` = ptr-load `… + BLOCK_D`; `acc2 += dot(p, v)`
25. `l_i`       = l_i_new                                              (carry)
26. `m_i`       = m_i_new                                              (carry)

The two output D-blocks (`acc`/`acc2`) share scores/softmax weights; they differ
only in which V projection is read (statements 22-24, `+BLOCK_D` channel offset).
-/

section BSARecipes

open VeriTile.Triton

/-- Local `evalOp` unfolding for `.ge` (no global simp lemma exists; mirrors
triton-attention's `ta_evalOp_ge`). -/
theorem bsa_evalOp_ge {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- Local `evalOp` unfolding for `.remap` (mirrors ctx-attn's `ctx_evalOp_remap`). -/
theorem bsa_evalOp_remap {dtype inShape outShape}
    (map : TileIndex outShape → TileIndex inShape) (a : Op dtype inShape) (s : BlockState) :
    evalOp (.remap outShape map a) s = (do
      let va ← evalOp a s; some (Tile.remap map va)) := by
  simp [evalOp]

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

/-- **`if cond { thenBody } else { elseBody }` step, false branch.** Mirror of
`stepStmt_ifThenElse_true` for `EVEN_N = false` (masked ptr-arith loads). -/
theorem stepStmt_ifThenElse_false {cond : Op .bool []}
    {thenBody elseBody : List Stmt} {s s' : BlockState}
    (hcond : evalOp cond s = some (Tile.scalar (Bool.false)))
    (helse : stepStmts elseBody s = some s') :
    stepStmt (.ifThenElse cond thenBody elseBody) s = some s' := by
  simp only [stepStmt, hcond, Option.bind_some, Tile.scalar_data, if_false]
  exact helse

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
/-- **`col_idx = tl.load(layout_csr_col_indices + layout_h·stride_col + col_idx_idx)`
statement eval** (CSR loop body L1): the unmasked `.nat` gather of the visited CSR
column index. With `layout_h = LH`, `stride_col = SC`, and `col_idx_idx = CI`, lane
reads `layoutCols[LH·SC + CI]`. Block-sparse analogue of `ctx_kvloc_gather_eval`. -/
theorem bsa_colidx_gather_eval (s : BlockState)
    (layoutCols : Region .nat) (LH SC CI : Nat)
    (hlh : s.regs .nat [] "layout_h" = some (Tile.scalar LH))
    (hci : s.regs .nat [] "col_idx_idx" = some (Tile.scalar CI)) :
    evalOp (Op.load .nat
        (MemAccess.region layoutCols
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "layout_h") (Op.constNat SC))
            (Op.ref .nat [] "col_idx_idx")))
        MaskOpt.none) s
      = some (Tile.scalar
          (s.readMemValue .nat (Region.cast layoutCols) (LH * SC + CI))) := by
  rw [evalOp_load_region_none]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hlh, hci,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp [Tile.bop_data, Tile.scalar_data, Tile.scalar_data_index,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

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

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`k = tl.load(k_ptrs + start_n·stride_kn)` statement eval** (CSR loop body L3,
first D-block, `EVEN_N` branch / unmasked): the pointer-arith K-tile load. Given
`k_ptrs` holding the per-lane pointer tile `kpf` (`[BLOCK_M, BLOCK_N]`, value
`(region, offset)`) and `start_n = SN`, lane `(a,b)` reads
`kpf(a,b).region[kpf(a,b).offset + SN·stride_kn]`. With `stride_kn = SKN`. -/
theorem bsa_load_k_eval {BM BN : Nat} (s : BlockState) (SN SKN : Nat)
    (kpf : TileIndex [BM, BN] → RegionName × Nat)
    (hk : s.regs .ptr [BM, BN] "k_ptrs" = some ⟨kpf⟩)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BN] "k_ptrs")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat SKN))))
        MaskOpt.none) s
      = some (⟨fun idx : TileIndex [BM, BN] =>
          s.readMemValue .real (kpf idx).1 ((kpf idx).2 + SN * SKN)⟩ : Tile .real [BM, BN]) := by
  simp only [evalOp, evalOp_ref, evalOp_mul, evalOp_constNat, hk, hsn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.scalar_data_index,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, if_true, if_pos]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`k = tl.load(k_ptrs + start_n·stride_kn + BLOCK_D)` statement eval** (CSR loop
body L6, second D-block, `EVEN_N` branch / unmasked): the second K-tile load, at
the `+BLOCK_D` head-channel offset (`acc2`'s key projection). Lane `(a,b)` reads
`kpf(a,b).region[kpf(a,b).offset + SN·stride_kn + BLOCK_D]`. -/
theorem bsa_load_k2_eval {BM BN : Nat} (s : BlockState) (SN SKN BD : Nat)
    (kpf : TileIndex [BM, BN] → RegionName × Nat)
    (hk : s.regs .ptr [BM, BN] "k_ptrs" = some ⟨kpf⟩)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR
            (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BN] "k_ptrs")
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat SKN)))
            (Op.constNat BD)))
        MaskOpt.none) s
      = some (⟨fun idx : TileIndex [BM, BN] =>
          s.readMemValue .real (kpf idx).1 ((kpf idx).2 + SN * SKN + BD)⟩
            : Tile .real [BM, BN]) := by
  simp only [evalOp, evalOp_ref, evalOp_mul, evalOp_constNat, hk, hsn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.scalar_data_index,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, if_true, if_pos]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`v = tl.load(v_ptrs + start_n·stride_vn)` statement eval** (CSR loop body L22,
first D-block, `EVEN_N` branch / unmasked): the pointer-arith V-tile load. Given
`v_ptrs` holding the pointer tile `vpf` (`[BLOCK_N, BLOCK_D]`) and `start_n = SN`,
lane `(a,b)` reads `vpf(a,b).region[vpf(a,b).offset + SN·stride_vn]`. With
`stride_vn = SVN`. -/
theorem bsa_load_v_eval {BN BD : Nat} (s : BlockState) (SN SVN : Nat)
    (vpf : TileIndex [BN, BD] → RegionName × Nat)
    (hv : s.regs .ptr [BN, BD] "v_ptrs" = some ⟨vpf⟩)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BN, BD] "v_ptrs")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat SVN))))
        MaskOpt.none) s
      = some (⟨fun idx : TileIndex [BN, BD] =>
          s.readMemValue .real (vpf idx).1 ((vpf idx).2 + SN * SVN)⟩ : Tile .real [BN, BD]) := by
  simp only [evalOp, evalOp_ref, evalOp_mul, evalOp_constNat, hv, hsn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.scalar_data_index,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, if_true, if_pos]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`v = tl.load(v_ptrs + start_n·stride_vn + BLOCK_D)` statement eval** (CSR loop
body L24, second D-block, `EVEN_N` branch / unmasked): the second V-tile load, at
the `+BLOCK_D` head-channel offset (`acc2`'s value projection). Lane `(a,b)` reads
`vpf(a,b).region[vpf(a,b).offset + SN·stride_vn + BLOCK_D]`. -/
theorem bsa_load_v2_eval {BN BD : Nat} (s : BlockState) (SN SVN BD' : Nat)
    (vpf : TileIndex [BN, BD] → RegionName × Nat)
    (hv : s.regs .ptr [BN, BD] "v_ptrs" = some ⟨vpf⟩)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR
            (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BN, BD] "v_ptrs")
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat SVN)))
            (Op.constNat BD')))
        MaskOpt.none) s
      = some (⟨fun idx : TileIndex [BN, BD] =>
          s.readMemValue .real (vpf idx).1 ((vpf idx).2 + SN * SVN + BD')⟩
            : Tile .real [BN, BD]) := by
  simp only [evalOp, evalOp_ref, evalOp_mul, evalOp_constNat, hv, hsn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.scalar_data_index,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, if_true, if_pos]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`qk = tl.zeros([BLOCK_M, BLOCK_N])` statement eval** (CSR loop body L4): the
all-`0` pre-dot accumulator. Block-sparse analogue of `ta_qkzeros_eval`. -/
theorem bsa_qkzeros_eval (s : BlockState) (BM BN : Nat) :
    evalOp (Op.full [BM, BN] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`qk += tl.dot(q, k)` statement eval** (CSR loop body L5, first D-block): adds
the `q · k` dot to the (zero-seeded) `qk` tile. Unlike triton-attention, block-sparse
loads `k` already transposed (`off_k = offs_n[None,:]·stride_kn + offs_d[:,None]`,
shape `[BLOCK_D, BLOCK_N]`), so the dot is `[BM,BD]·[BD,BN]` with no `tl.trans`. -/
theorem bsa_qk_dot_eval (s : BlockState) (BM BN BD : Nat)
    (qktile : Tile .real [BM, BN]) (qtile : Tile .real [BM, BD]) (ktile : Tile .real [BD, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hq : s.regs .real [BM, BD] "q" = some qtile)
    (hk : s.regs .real [BD, BN] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          qktile (Tile.dot [] qtile ktile)) := by
  have hqr : evalOp (Op.ref .real [BM, BD] "q") s = some qtile := by rw [evalOp_ref, hq]
  have hkr : evalOp (Op.ref .real [BD, BN] "k") s = some ktile := by rw [evalOp_ref, hk]
  have hdot : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"), hqr, hkr]; rfl
  rw [evalOp_add]
  simp only [evalOp_ref, hqk, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`qk += tl.dot(q2, k)` statement eval** (CSR loop body L6, second D-block): adds
the `q2 · k` dot (second-D-block query against the `+BLOCK_D` key projection) into the
running `qk` score, so both D-blocks contribute to the same softmax. -/
theorem bsa_qk2_dot_eval (s : BlockState) (BM BN BD : Nat)
    (qktile : Tile .real [BM, BN]) (q2tile : Tile .real [BM, BD]) (ktile : Tile .real [BD, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hq2 : s.regs .real [BM, BD] "q2" = some q2tile)
    (hk : s.regs .real [BD, BN] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := []) (Op.ref .real [BM, BD] "q2") (Op.ref .real [BD, BN] "k"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          qktile (Tile.dot [] q2tile ktile)) := by
  have hqr : evalOp (Op.ref .real [BM, BD] "q2") s = some q2tile := by rw [evalOp_ref, hq2]
  have hkr : evalOp (Op.ref .real [BD, BN] "k") s = some ktile := by rw [evalOp_ref, hk]
  have hdot : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q2") (Op.ref .real [BD, BN] "k")) s
      = some (Tile.dot [] q2tile ktile) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BD] "q2") (Op.ref .real [BD, BN] "k"), hqr, hkr]; rfl
  rw [evalOp_add]
  simp only [evalOp_ref, hqk, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`qk *= softmax_scale` statement eval** (CSR loop body L7): scale the raw score
by the scalar `softmax_scale` (broadcast on the right). Block-sparse analogue of
`ta_qk_scale_eval`. -/
theorem bsa_qk_scale_eval (s : BlockState) (BM BN : Nat) (sc : ℝ)
    (qktile : Tile .real [BM, BN]) (hqk : s.regs .real [BM, BN] "qk" = some qktile) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BN] "qk") (Op.const sc)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR qktile
          (Tile.scalar (some sc : WithBot ℝ))) := by
  rw [evalOp_mul]; simp [evalOp_ref, evalOp_const, hqk]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Causal additive `where` statement eval** (CSR loop body L8): block-sparse uses
`qk += tl.where(offs_m[:,None] ≥ start_n + offs_n[None,:], 0, -inf)` — an *additive*
causal mask (`0` for visible keys, `-inf` for future), unlike triton-attention's
selecting `where(cond, qk, -inf)`. Given `offs_m = gm`, `offs_n = id`, `start_n = SN`,
and running `qk`, lane `(a,b)` becomes `qk(a,b) + (if SN+b ≤ gm a then 0 else ⊥)`. -/
theorem bsa_where_eval (s : BlockState) (BM BN SN : Nat)
    (gm : Fin BM → Nat) (qktile : Tile .real [BM, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hom : s.regs .nat [BM] "offs_m" = some (Tile.vec gm))
    (hon : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "qk")
        (Op.where
          (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))))
          (Op.broadcast (Op.const 0) [BM, BN]) (Op.broadcast Op.negInf [BM, BN]))) s
      = some ⟨fun idx : TileIndex [BM, BN] =>
          NumericDType.real.add (qktile.data idx)
            (if SN + idx.2.1.val ≤ gm idx.1 then (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))⟩ := by
  have hexpM : @evalOp TileDType.nat [BM, 1]
      (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) s
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec gm)) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hom
  have hexpN : @evalOp TileDType.nat [1, BN]
      (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) s
        = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => j.val))) :=
    evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hon
  have haddN : @evalOp TileDType.nat [1, BN]
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n"))) s
        = some (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar SN)
            (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BN => j.val)))) := by
    rw [evalOp_add]
    rw [show evalOp (Op.ref .nat [] "start_n") s = some (Tile.scalar SN) from by
      rw [evalOp_ref, hsn]]
    rw [hexpN]; rfl
  have hzero : @evalOp TileDType.real [BM, BN] (Op.broadcast (Op.const 0) [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => (some (0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]; rfl
  have hbcast : @evalOp TileDType.real [BM, BN] (Op.broadcast Op.negInf [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_add, evalOp_where, bsa_evalOp_ge]
  simp only [evalOp_ref, hexpM, haddN, hqk, hzero, hbcast,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.bop_data, Tile.select_data, Tile.cop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, Tile.expandDim_data, Tile.scalar, Tile.vec,
    ComparableDType.nat]
  congr 1
  by_cases h : SN + idx.2.1.val ≤ gm idx.1
  · rw [if_pos (by simpa [ComparableDType.ge] using h)]; simp [h]
  · rw [if_neg (by simpa [ComparableDType.ge] using h)]; simp [h]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`m_ij = tl.max(qk, 1)` statement eval** (CSR loop body L9): the per-row max of
the masked score tile (the *current-block* max, before merging with the running
`m_i`). `reduceMax false`'s `eraseAxis` result-shape blocks `rw` matching, so the
reduced row is supplied as `hrm` and defeq-coerced to `[BM]`. -/
theorem bsa_mij_eval (s : BlockState) (BM BN : Nat)
    (qktile : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qktile = some rmaxT) :
    evalOp (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
        (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
  rw [evalOp_reduceMax]; simp only [evalOp_ref, hqk]; exact hrm

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`p = tl.exp(qk - m_ij[:, None])` statement eval** (CSR loop body L10): the
NATURAL-exp (`tl.exp`, not `exp2`) softmax numerator shift by the current-block max
`m_ij`. `expandDim`'s `insertAxis` shape is normalized to `[BM,1]` by `sub`. -/
theorem bsa_p_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (qktile : Tile .real [BM, BN]) (mij : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp (Op.exp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] "qk") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_ij")))) s
      = some (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            qktile (Tile.expandDim ⟨1, hax⟩ mij))) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_ij")) s
      = some (Tile.expandDim ⟨1, hax⟩ mij) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmij
  rw [evalOp_exp, evalOp_sub]
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`l_ij = tl.sum(p, 1)` statement eval** (CSR loop body L11): the per-row sum of
the (current-block, `m_ij`-shifted) softmax numerator `p`. -/
theorem bsa_lij_eval (s : BlockState) (BM BN : Nat)
    (ptile : Tile .real [BM, BN])
    (hp : s.regs .real [BM, BN] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
        (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) ptile) := by
  have hpr : evalOp (Op.ref .real [BM, BN] "p") s = some ptile := by rw [evalOp_ref, hp]
  erw [evalOp_reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
    (Op.ref .real [BM, BN] "p"), hpr]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`m_i_new = tl.maximum(m_i, m_ij)` statement eval** (CSR loop body L12): the
running max merge, lowered to `where(m_i > m_ij, m_i, m_ij)`. -/
theorem bsa_minew_eval (s : BlockState) (BM : Nat)
    (mp mij : Tile .real [BM])
    (hmp : s.regs .real [BM] "m_i" = some mp)
    (hmij : s.regs .real [BM] "m_ij" = some mij) :
    evalOp (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_ij"))
        (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_ij")) s
      = some (Tile.select
          (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mp mij) mp mij) := by
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmp, hmij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`alpha = tl.exp(m_i - m_i_new)` statement eval** (CSR loop body L13): the
running-denominator/accumulator rescale factor `exp(m_i - m_i_new)` (NATURAL exp). -/
theorem bsa_alpha_eval (s : BlockState) (BM : Nat) (mp mn : Tile .real [BM])
    (hmp : s.regs .real [BM] "m_i" = some mp)
    (hmn : s.regs .real [BM] "m_i_new" = some mn) :
    evalOp (Op.exp (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_i_new"))) s
      = some (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mp mn)) := by
  rw [evalOp_exp, evalOp_sub]
  simp only [evalOp_ref, hmp, hmn, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`beta = tl.exp(m_ij - m_i_new)` statement eval** (CSR loop body L14): the
current-block numerator rescale factor `exp(m_ij - m_i_new)` (NATURAL exp). -/
theorem bsa_beta_eval (s : BlockState) (BM : Nat) (mij mn : Tile .real [BM])
    (hmij : s.regs .real [BM] "m_ij" = some mij)
    (hmn : s.regs .real [BM] "m_i_new" = some mn) :
    evalOp (Op.exp (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BM] "m_ij") (Op.ref .real [BM] "m_i_new"))) s
      = some (Tile.uop WithBot.realExp
          (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mij mn)) := by
  rw [evalOp_exp, evalOp_sub]
  simp only [evalOp_ref, hmij, hmn, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`l_i_new = alpha·l_i + beta·l_ij` statement eval** (CSR loop body L15): the
online-softmax running-denominator update combining the rescaled old denominator and
the current block's contribution. -/
theorem bsa_linew_eval (s : BlockState) (BM : Nat)
    (al li be lij : Tile .real [BM])
    (hal : s.regs .real [BM] "alpha" = some al)
    (hli : s.regs .real [BM] "l_i" = some li)
    (hbe : s.regs .real [BM] "beta" = some be)
    (hlij : s.regs .real [BM] "l_ij" = some lij) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BM] "alpha") (Op.ref .real [BM] "l_i"))
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BM] "beta") (Op.ref .real [BM] "l_ij"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) al li)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) be lij)) := by
  rw [evalOp_add, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, hal, hli, hbe, hlij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`p_scale = beta / l_i_new` statement eval** (CSR loop body L16): the per-row
factor folding the current-block numerator rescale into the final normalization. -/
theorem bsa_pscale_eval (s : BlockState) (BM : Nat) (be ln : Tile .real [BM])
    (hbe : s.regs .real [BM] "beta" = some be)
    (hln : s.regs .real [BM] "l_i_new" = some ln) :
    evalOp (Op.div .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BM] "beta") (Op.ref .real [BM] "l_i_new")) s
      = some (Tile.bop NumericDType.real.div (Broadcast.consSame Broadcast.nil) be ln) := by
  rw [evalOp_div]
  simp only [evalOp_ref, hbe, hln, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`p *= p_scale[:, None]` statement eval** (CSR loop body L17): apply the per-row
`p_scale` to the softmax numerator `p`. -/
theorem bsa_p_rescale_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (ptile : Tile .real [BM, BN]) (ps : Tile .real [BM])
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hps : s.regs .real [BM] "p_scale" = some ps) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] "p") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "p_scale"))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          ptile (Tile.expandDim ⟨1, hax⟩ ps)) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "p_scale")) s
      = some (Tile.expandDim ⟨1, hax⟩ ps) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hps
  rw [evalOp_mul]
  simp only [evalOp_ref, hp, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`acc_scale = l_i / l_i_new * alpha` statement eval** (CSR loop body L18): the
per-row output-accumulator rescale factor `(l_i / l_i_new)·alpha`. -/
theorem bsa_accscale_eval (s : BlockState) (BM : Nat) (li ln al : Tile .real [BM])
    (hli : s.regs .real [BM] "l_i" = some li)
    (hln : s.regs .real [BM] "l_i_new" = some ln)
    (hal : s.regs .real [BM] "alpha" = some al) :
    evalOp (Op.mul .real (Broadcast.consSame Broadcast.nil)
        (Op.div .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "l_i_new"))
        (Op.ref .real [BM] "alpha")) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.div (Broadcast.consSame Broadcast.nil) li ln) al) := by
  rw [evalOp_mul, evalOp_div]
  simp only [evalOp_ref, hli, hln, hal, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`acc *= acc_scale[:, None]` statement eval** (CSR loop body L19, first D-block):
rescale the first output accumulator by the per-row `acc_scale`. -/
theorem bsa_acc_rescale_eval (s : BlockState) (BM BD : Nat) (hax : 1 < [BM].length.succ)
    (acctile : Tile .real [BM, BD]) (asc : Tile .real [BM])
    (hacc : s.regs .real [BM, BD] "acc" = some acctile)
    (hasc : s.regs .real [BM] "acc_scale" = some asc) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BD] "acc") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "acc_scale"))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          acctile (Tile.expandDim ⟨1, hax⟩ asc)) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "acc_scale")) s
      = some (Tile.expandDim ⟨1, hax⟩ asc) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hasc
  rw [evalOp_mul]
  simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`acc2 *= acc_scale[:, None]` statement eval** (CSR loop body L20, second
D-block): rescale the second output accumulator by the **same** per-row `acc_scale`
(both D-blocks share the softmax recurrence). -/
theorem bsa_acc2_rescale_eval (s : BlockState) (BM BD : Nat) (hax : 1 < [BM].length.succ)
    (acc2tile : Tile .real [BM, BD]) (asc : Tile .real [BM])
    (hacc2 : s.regs .real [BM, BD] "acc2" = some acc2tile)
    (hasc : s.regs .real [BM] "acc_scale" = some asc) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BD] "acc2") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "acc_scale"))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          acc2tile (Tile.expandDim ⟨1, hax⟩ asc)) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "acc_scale")) s
      = some (Tile.expandDim ⟨1, hax⟩ asc) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hasc
  rw [evalOp_mul]
  simp only [evalOp_ref, hacc2, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`p = (p).to(Q.dtype.element_ty)` statement eval** (CSR loop body L21): the dot
input dtype round-trip before the value accumulation. In this real model the cast is
identity (the elaborated AST is a bare `ref "p"`), so `p` is unchanged. -/
theorem bsa_pcast_eval (s : BlockState) (BM BN : Nat) (ptile : Tile .real [BM, BN])
    (hp : s.regs .real [BM, BN] "p" = some ptile) :
    evalOp (Op.ref .real [BM, BN] "p") s = some ptile := by
  rw [evalOp_ref, hp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`acc += tl.dot(p, v)` statement eval** (CSR loop body L23, first D-block): the
first-D-block numerator accumulation `acc + p·v` (`p : [BM,BN]`, `v : [BN,BD]`). -/
theorem bsa_acc_eval (s : BlockState) (BM BN BD : Nat)
    (acctile : Tile .real [BM, BD]) (ptile : Tile .real [BM, BN]) (vtile : Tile .real [BN, BD])
    (hacc : s.regs .real [BM, BD] "acc" = some acctile)
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hv : s.regs .real [BN, BD] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acctile
          (Tile.dot [] ptile vtile)) := by
  have hpr : evalOp (Op.ref .real [BM, BN] "p") s = some ptile := by rw [evalOp_ref, hp]
  have hvr : evalOp (Op.ref .real [BN, BD] "v") s = some vtile := by rw [evalOp_ref, hv]
  have hdot : @evalOp TileDType.real [BM, BD]
      (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v")) s
      = some (Tile.dot [] ptile vtile) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v"), hpr, hvr]; rfl
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`acc2 += tl.dot(p, v)` statement eval** (CSR loop body L24, second D-block): the
second-D-block numerator accumulation `acc2 + p·v`, with the **same** `p` and the
second-D-block value tile `v` (loaded at `+BLOCK_D`). -/
theorem bsa_acc2_eval (s : BlockState) (BM BN BD : Nat)
    (acc2tile : Tile .real [BM, BD]) (ptile : Tile .real [BM, BN]) (vtile : Tile .real [BN, BD])
    (hacc2 : s.regs .real [BM, BD] "acc2" = some acc2tile)
    (hp : s.regs .real [BM, BN] "p" = some ptile)
    (hv : s.regs .real [BN, BD] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "acc2")
        (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acc2tile
          (Tile.dot [] ptile vtile)) := by
  have hpr : evalOp (Op.ref .real [BM, BN] "p") s = some ptile := by rw [evalOp_ref, hp]
  have hvr : evalOp (Op.ref .real [BN, BD] "v") s = some vtile := by rw [evalOp_ref, hv]
  have hdot : @evalOp TileDType.real [BM, BD]
      (Op.dot (batch := []) (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v")) s
      = some (Tile.dot [] ptile vtile) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BN] "p") (Op.ref .real [BN, BD] "v"), hpr, hvr]; rfl
  rw [evalOp_add]; simp only [evalOp_ref, hacc2, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`l_i = l_i_new` / `m_i = m_i_new` statement eval** (CSR loop body L25/L26): the
running-state carry into the next CSR iteration (a bare register read). Instantiate
`name := "l_i_new"` / `"m_i_new"`. Block-sparse analogue of `ta_reg_carry_eval`. -/
theorem bsa_reg_carry_eval (s : BlockState) (BM : Nat) (name : RegName)
    (t : Tile .real [BM]) (h : s.regs .real [BM] name = some t) :
    evalOp (Op.ref .real [BM] name) s = some t := by
  rw [evalOp_ref, h]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`off_h_kv = off_h // head_groups` statement eval** (GQA head remap, pre-loop):
the grouped-query KV-head index `off_h // (num_heads // num_kv_heads)`. With
`off_h = OH` and `head_groups = HG`, evaluates to `OH // HG`. This is the head map the
K/V base offsets (`kvBase`) use. -/
theorem bsa_offhkv_eval (s : BlockState) (OH HG : Nat)
    (hoh : s.regs .nat [] "off_h" = some (Tile.scalar OH))
    (hhg : s.regs .nat [] "head_groups" = some (Tile.scalar HG)) :
    evalOp (Op.floorDiv .nat Broadcast.nil
        (Op.ref .nat [] "off_h") (Op.ref .nat [] "head_groups")) s
      = some (Tile.scalar (OH / HG)) := by
  simp only [evalOp, evalOp_ref, hoh, hhg, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.bop_data, Tile.scalar, Tile.scalar_data_index, Broadcast.leftIndex,
    Broadcast.rightIndex, IntegralDType.floorDiv]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Masked first-D-block K load** (CSR loop body L3, `EVEN_N = false` else-branch):
`k = tl.load(k_ptrs + start_n·stride_kn, mask=offs_n[None,:] + start_n < total_seq_len)`.
Out-of-`total_seq_len` key columns read the (undefined) carrier rather than crashing;
since the address is always in-region (`ok = true`), inactive lanes read
`s.undef`. Lane `(a,b)` is active iff `b + SN < TSL`. -/
theorem bsa_load_k_masked_eval {BM BN : Nat} (s : BlockState) (SN SKN TSL : Nat)
    (kpf : TileIndex [BM, BN] → RegionName × Nat)
    (hk : s.regs .ptr [BM, BN] "k_ptrs" = some ⟨kpf⟩)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hon : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BM, BN] "k_ptrs")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat SKN))))
        (MaskOpt.mask
          (Op.remap [BM, BN] (fun x => (⟨0, Broadcast.leftIndex._proof_1⟩, x.2.1, PUnit.unit))
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarR
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.ref .nat [] "start_n"))
              (Op.constNat TSL))))) s
      = some (⟨fun idx : TileIndex [BM, BN] =>
          if idx.2.1.val + SN < TSL then
            s.readMemValue .real (kpf idx).1 ((kpf idx).2 + SN * SKN)
          else some (s.undef (kpf idx).1 ((kpf idx).2 + SN * SKN))⟩ : Tile .real [BM, BN]) := by
  have hmask : evalOp
      (Op.remap [BM, BN] (fun x => (⟨0, Broadcast.leftIndex._proof_1⟩, x.2.1, PUnit.unit))
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.ref .nat [] "start_n"))
          (Op.constNat TSL))) s
      = some (⟨fun idx : TileIndex [BM, BN] => decide (idx.2.1.val + SN < TSL)⟩
          : Tile .bool [BM, BN]) := by
    rw [bsa_evalOp_remap, evalOp_lt, evalOp_add]
    erw [evalOp_expandDim_ref_of_regs .nat [BN] ⟨0, by simp⟩ "offs_n" s _ hon]
    simp only [evalOp_ref, evalOp_constNat, hsn, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_; ext idx
    simp [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.expandDim, Tile.scalar_data_index,
      Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.add]
  simp only [evalOp, hmask, evalOp_ref, evalOp_mul, evalOp_constNat, hk, hsn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.scalar_data_index,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, if_true, if_pos]
  by_cases h : idx.2.1.val + SN < TSL
  · simp [h]
  · simp [h]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Masked first-D-block V load** (CSR loop body L22, `EVEN_N = false` else-branch):
`v = tl.load(v_ptrs + start_n·stride_vn, mask=offs_n[:,None] + start_n < total_seq_len)`.
The V mask puts `offs_n` on the **row** axis (`[:,None]`), so lane `(a,b)` is active iff
`a + SN < TSL`. -/
theorem bsa_load_v_masked_eval {BN BD : Nat} (s : BlockState) (SN SVN TSL : Nat)
    (vpf : TileIndex [BN, BD] → RegionName × Nat)
    (hv : s.regs .ptr [BN, BD] "v_ptrs" = some ⟨vpf⟩)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hon : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BN, BD] "v_ptrs")
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat SVN))))
        (MaskOpt.mask
          (Op.remap [BN, BD] (fun x => (x.1, ⟨0, Broadcast.leftIndex._proof_1⟩, PUnit.unit))
            (Op.lt .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.ref .nat [] "start_n"))
              (Op.constNat TSL))))) s
      = some (⟨fun idx : TileIndex [BN, BD] =>
          if idx.1.val + SN < TSL then
            s.readMemValue .real (vpf idx).1 ((vpf idx).2 + SN * SVN)
          else some (s.undef (vpf idx).1 ((vpf idx).2 + SN * SVN))⟩ : Tile .real [BN, BD]) := by
  have hmask : evalOp
      (Op.remap [BN, BD] (fun x => (x.1, ⟨0, Broadcast.leftIndex._proof_1⟩, PUnit.unit))
        (Op.lt .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n")) (Op.ref .nat [] "start_n"))
          (Op.constNat TSL))) s
      = some (⟨fun idx : TileIndex [BN, BD] => decide (idx.1.val + SN < TSL)⟩
          : Tile .bool [BN, BD]) := by
    rw [bsa_evalOp_remap, evalOp_lt, evalOp_add]
    erw [evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, by simp⟩ "offs_n" s _ hon]
    simp only [evalOp_ref, evalOp_constNat, hsn, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_; ext idx
    simp [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.expandDim, Tile.scalar_data_index,
      Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt, NumericDType.add]
  simp only [evalOp, hmask, evalOp_ref, evalOp_mul, evalOp_constNat, hv, hsn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.scalar_data_index,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, if_true, if_pos]
  by_cases h : idx.1.val + SN < TSL
  · simp [h]
  · simp [h]

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

/-- The 25 lowered pre-loop statements (static_print marker through `end_l`). -/
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
    (Qg : TileIndex [16, 16] → ℝ)
    (Kg Vg Vg2 : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
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
/-- **Pre-loop execution.** The 25 deterministic pre-loop statements step a clean
input state (`undef = 0`) to a state satisfying `bsaInvariant … 0` — the loop-entry
base case (`m_i = ⊥`, `l_i = 0`, `acc = acc2 = 0`, via the `bsaMPartial`/
`bsaLPartial`/`bsaOPartial` zero recurrences). The streaming data is arbitrary;
at `c = 0` the accumulators are data-independent. -/
theorem bsaPreLoop_eval
    (s : BlockState) (Out Q K V : RegionName) (R C : Region .nat)
    (qStart numKVBlocks : Nat) (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 16] → ℝ)
    (Kg Vg Vg2 : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (bsaPreLoop Out Q K V R C) s = some s0
      ∧ bsaInvariant Out Q K V R C qStart numKVBlocks gpos Qg Kg Vg Vg2 scale s 0 s0 := by
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
  refine ⟨_, rfl, ?_⟩
  unfold bsaInvariant
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

/-! ## CSR loop-body execution chain (`bsaLoopBody_steps`)

The keystone for the loop step: peeling the 26 lowered `bsaLoopBody` statements
through their banked recipes, threading an explicit symbolic `BlockState` and
exposing the resulting `m_i`/`l_i`/`p`/`acc`/`acc2` registers symbolically. The
caller supplies the entry-state registers the body reads (`layout_h`, `k_ptrs`,
`v_ptrs`, `q`, `q2`, `offs_m`, `offs_n`, `m_i`, `l_i`, `acc`, `acc2`); the proof
is purely the execution chain (no math bridge to `bsaInvariant`). Mirrors
`nopad_attn_step`'s 22-statement post-`setReg` peeling. -/

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **CSR loop-body execution.** Given an entry state `s` whose body-read
registers hold the supplied symbolic tiles, the 26 `bsaLoopBody` statements step
the rebound state `s.setReg "col_idx_idx" idx` to an explicit `s'` whose live
registers (`m_i`/`l_i`/`p`/`acc`/`acc2` and the carried scalars) are the
banked-recipe outputs. The two CSR loop bounds, `start_n = colIdx·16`, the
`EVEN_N`/`NUM_D_BLOCKS≥2` branches, and mem/pids preservation are discharged. -/
theorem bsaLoopBody_steps
    (s : BlockState) (Out Q K V : RegionName) (R C : Region .nat)
    (idx LH : Nat)
    (kpf vpf : TileIndex [16, 16] → RegionName × Nat)
    (qt q2t acct acc2t : Tile .real [16, 16])
    (gm : Fin 16 → Nat) (mt lt : Tile .real [16])
    (hlh : s.regs .nat [] "layout_h" = some (Tile.scalar LH))
    (hkp : s.regs .ptr [16, 16] "k_ptrs" = some ⟨kpf⟩)
    (hvp : s.regs .ptr [16, 16] "v_ptrs" = some ⟨vpf⟩)
    (hq : s.regs .real [16, 16] "q" = some qt)
    (hq2 : s.regs .real [16, 16] "q2" = some q2t)
    (hom : s.regs .nat [16] "offs_m" = some (Tile.vec gm))
    (hon : s.regs .nat [16] "offs_n" = some (Tile.vec (fun j : Fin 16 => j.val)))
    (hmi : s.regs .real [16] "m_i" = some mt)
    (hli : s.regs .real [16] "l_i" = some lt)
    (hacc : s.regs .real [16, 16] "acc" = some acct)
    (hacc2 : s.regs .real [16, 16] "acc2" = some acc2t) :
    ∃ s',
      stepStmts (bsaLoopBody C) (s.setReg "col_idx_idx" .nat [] (Tile.scalar idx)) = some s'
      ∧ s'.pids = s.pids
      ∧ s'.mem = s.mem
      ∧ (∀ rg o, s'.undef rg o = s.undef rg o)
      -- preservation of registers the body never assigns
      ∧ (∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
          nm ∉ (["col_idx_idx", "col_idx", "start_n", "k", "qk", "m_ij", "p", "l_ij",
            "m_i_new", "alpha", "beta", "l_i_new", "p_scale", "acc_scale", "v",
            "l_i", "m_i", "acc", "acc2"] : List RegName) →
          s.regs dt sh nm = some t → s'.regs dt sh nm = some t)
      -- col_idx / start_n derived scalars
      ∧ (let CI := s.readMemValue .nat (Region.cast C) (LH * 4 + idx);
         let SN := CI * 16;
         -- k tile (first D-block)
         let kt : Tile .real [16, 16] :=
           ⟨fun ix => s.readMemValue .real (kpf ix).1 ((kpf ix).2 + SN * 32)⟩;
         -- k2 tile (second D-block)
         let k2t : Tile .real [16, 16] :=
           ⟨fun ix => s.readMemValue .real (kpf ix).1 ((kpf ix).2 + SN * 32 + 16)⟩;
         -- qk after both dots + scale by 1.0 + causal additive mask
         let qkDot : Tile .real [16, 16] :=
           Tile.bop NumericDType.real.add
             (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
             (Tile.bop NumericDType.real.add
               (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
               (⟨fun _ : TileIndex [16, 16] => some (0 : ℝ)⟩ : Tile .real [16, 16])
               (Tile.dot [] qt kt))
             (Tile.dot [] q2t k2t);
         let qkScaled : Tile .real [16, 16] :=
           Tile.bop NumericDType.real.mul Broadcast.scalarR qkDot
             (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ));
         let qkM : Tile .real [16, 16] :=
           ⟨fun ix : TileIndex [16, 16] =>
             NumericDType.real.add (qkScaled.data ix)
               (if SN + ix.2.1.val ≤ gm ix.1 then (some (0 : ℝ) : WithBot ℝ)
                else (⊥ : WithBot ℝ))⟩;
         let mij : Tile .real [16] :=
           ⟨fun outIdx : TileIndex [16] =>
             (Finset.univ : Finset (Fin 16)).sup' Finset.univ_nonempty
               (fun k : Fin 16 => qkM.data (outIdx.1, k, PUnit.unit))⟩;
         let p0 : Tile .real [16, 16] :=
           Tile.uop WithBot.realExp
             (Tile.bop NumericDType.real.sub
               (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkM
               (Tile.expandDim ⟨1, by simp⟩ mij));
         let lij : Tile .real [16] :=
           Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [16, 16].length) p0;
         let minew : Tile .real [16] :=
           Tile.select
             (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mt mij) mt mij;
         let alpha : Tile .real [16] :=
           Tile.uop WithBot.realExp
             (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mt minew);
         let beta : Tile .real [16] :=
           Tile.uop WithBot.realExp
             (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mij minew);
         let linew : Tile .real [16] :=
           Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
             (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) alpha lt)
             (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) beta lij);
         let pscale : Tile .real [16] :=
           Tile.bop NumericDType.real.div (Broadcast.consSame Broadcast.nil) beta linew;
         let pfinal : Tile .real [16, 16] :=
           Tile.bop NumericDType.real.mul
             (Broadcast.consSame (Broadcast.consR Broadcast.nil)) p0
             (Tile.expandDim ⟨1, by simp⟩ pscale);
         let accscale : Tile .real [16] :=
           Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
             (Tile.bop NumericDType.real.div (Broadcast.consSame Broadcast.nil) lt linew) alpha;
         let accMul : Tile .real [16, 16] :=
           Tile.bop NumericDType.real.mul
             (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acct
             (Tile.expandDim ⟨1, by simp⟩ accscale);
         let acc2Mul : Tile .real [16, 16] :=
           Tile.bop NumericDType.real.mul
             (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acc2t
             (Tile.expandDim ⟨1, by simp⟩ accscale);
         let vt : Tile .real [16, 16] :=
           ⟨fun ix => s.readMemValue .real (vpf ix).1 ((vpf ix).2 + SN * 32)⟩;
         let v2t : Tile .real [16, 16] :=
           ⟨fun ix => s.readMemValue .real (vpf ix).1 ((vpf ix).2 + SN * 32 + 16)⟩;
         let accFinal : Tile .real [16, 16] :=
           Tile.bop NumericDType.real.add
             (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) accMul
             (Tile.dot [] pfinal vt);
         let acc2Final : Tile .real [16, 16] :=
           Tile.bop NumericDType.real.add
             (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acc2Mul
             (Tile.dot [] pfinal v2t);
         s'.regs .real [16] "m_i" = some minew
         ∧ s'.regs .real [16] "l_i" = some linew
         ∧ s'.regs .real [16, 16] "p" = some pfinal
         ∧ s'.regs .real [16, 16] "acc" = some accFinal
         ∧ s'.regs .real [16, 16] "acc2" = some acc2Final) := by
  unfold bsaLoopBody
  -- entry state s0 := s with col_idx_idx rebound to idx
  set s0 := s.setReg "col_idx_idx" .nat [] (Tile.scalar idx) with hs0d
  have e0 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "col_idx_idx" → s.regs dt sh nm = some t → s0.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs0d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs0ci : s0.regs .nat [] "col_idx_idx" = some (Tile.scalar idx) := by
    rw [hs0d, BlockState.setReg_same]
  have hs0lh : s0.regs .nat [] "layout_h" = some (Tile.scalar LH) := e0 (by decide) hlh
  have hs0kp : s0.regs .ptr [16, 16] "k_ptrs" = some ⟨kpf⟩ := e0 (by decide) hkp
  have hs0vp : s0.regs .ptr [16, 16] "v_ptrs" = some ⟨vpf⟩ := e0 (by decide) hvp
  have hs0q : s0.regs .real [16, 16] "q" = some qt := e0 (by decide) hq
  have hs0q2 : s0.regs .real [16, 16] "q2" = some q2t := e0 (by decide) hq2
  have hs0om : s0.regs .nat [16] "offs_m" = some (Tile.vec gm) := e0 (by decide) hom
  have hs0on : s0.regs .nat [16] "offs_n" = some (Tile.vec (fun j : Fin 16 => j.val)) := e0 (by decide) hon
  have hs0mi : s0.regs .real [16] "m_i" = some mt := e0 (by decide) hmi
  have hs0li : s0.regs .real [16] "l_i" = some lt := e0 (by decide) hli
  have hs0acc : s0.regs .real [16, 16] "acc" = some acct := e0 (by decide) hacc
  have hs0acc2 : s0.regs .real [16, 16] "acc2" = some acc2t := e0 (by decide) hacc2
  -- ===== stmt 0: col_idx = load C[layout_h*4 + col_idx_idx] =====
  set CI := s.readMemValue .nat (Region.cast C) (LH * 4 + idx) with hCI
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ s0 = some (Tile.scalar CI) from by
      have h := bsa_colidx_gather_eval s0 C LH 4 idx hs0lh hs0ci
      rw [hCI]
      have hmem : s0.readMemValue .nat (Region.cast C) (LH * 4 + idx)
          = s.readMemValue .nat (Region.cast C) (LH * 4 + idx) := by
        simp only [hs0d, BlockState.setReg_readMemValue]
      rw [hmem] at h; exact h))]
  set s1 := s0.setReg "col_idx" .nat [] (Tile.scalar CI) with hs1d
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "col_idx" → s0.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1ci : s1.regs .nat [] "col_idx" = some (Tile.scalar CI) := by rw [hs1d, BlockState.setReg_same]
  -- ===== stmt 1: start_n = col_idx * 16 =====
  set SN := CI * 16 with hSN
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "col_idx") (Op.constNat 16)) s1
        = some (Tile.scalar SN) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, hs1ci, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext ix
      simp [Tile.bop_data, Tile.scalar_data_index, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.mul, hSN]))]
  set s2 := s1.setReg "start_n" .nat [] (Tile.scalar SN) with hs2d
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2sn : s2.regs .nat [] "start_n" = some (Tile.scalar SN) := by rw [hs2d, BlockState.setReg_same]
  have hs2kp : s2.regs .ptr [16, 16] "k_ptrs" = some ⟨kpf⟩ := e2 (by decide) (e1 (by decide) hs0kp)
  -- mem-transport helpers: readMemValue at s2 = at s
  have hs2memk : ∀ ix : TileIndex [16, 16],
      s2.readMemValue .real (kpf ix).1 ((kpf ix).2 + SN * 32)
        = s.readMemValue .real (kpf ix).1 ((kpf ix).2 + SN * 32) := by
    intro ix; simp only [hs2d, hs1d, hs0d, BlockState.setReg_readMemValue]
  -- ===== stmt 2: ifThenElse true { k = load (k_ptrs + start_n*32) } =====
  set kt : Tile .real [16, 16] :=
    ⟨fun ix => s.readMemValue .real (kpf ix).1 ((kpf ix).2 + SN * 32)⟩ with hktd
  rw [stepStmts.cons_some (stepStmt_ifThenElse_true (by simp [evalOp])
    (show stepStmts _ s2 = some _ from by
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp _ s2 = some kt from by
          have h := bsa_load_k_eval s2 SN 32 kpf hs2kp hs2sn
          rw [show (⟨fun ix : TileIndex [16, 16] =>
                s2.readMemValue .real (kpf ix).1 ((kpf ix).2 + SN * 32)⟩ : Tile .real [16, 16])
              = kt from by
            refine Tile.ext (fun ix => ?_); rw [hktd]; exact hs2memk ix] at h
          exact h))]
      rw [stepStmts.nil]))]
  set s3 := s2.setReg "k" .real [16, 16] kt with hs3d
  have e3 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "k" → s2.regs dt sh nm = some t → s3.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs3d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs3k : s3.regs .real [16, 16] "k" = some kt := by rw [hs3d, BlockState.setReg_same]
  have hs3q : s3.regs .real [16, 16] "q" = some qt :=
    e3 (by decide) (e2 (by decide) (e1 (by decide) hs0q))
  -- ===== stmt 3: qk = zeros =====
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_qkzeros_eval s3 16 16))]
  set s4 := s3.setReg "qk" .real [16, 16]
    (⟨fun _ : TileIndex [16, 16] => some (0 : ℝ)⟩ : Tile .real [16, 16]) with hs4d
  have e4 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s3.regs dt sh nm = some t → s4.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs4d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs4qk : s4.regs .real [16, 16] "qk" = some (⟨fun _ : TileIndex [16, 16] => some (0 : ℝ)⟩ : Tile .real [16, 16]) := by
    rw [hs4d, BlockState.setReg_same]
  have hs4q : s4.regs .real [16, 16] "q" = some qt := e4 (by decide) hs3q
  have hs4k : s4.regs .real [16, 16] "k" = some kt := e4 (by decide) hs3k
  -- ===== stmt 4: qk += dot(q, k) =====
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_qk_dot_eval s4 16 16 16 _ qt kt hs4qk hs4q hs4k))]
  set qk1 : Tile .real [16, 16] :=
    Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      (⟨fun _ : TileIndex [16, 16] => some (0 : ℝ)⟩ : Tile .real [16, 16])
      (Tile.dot [] qt kt) with hqk1d
  set s5 := s4.setReg "qk" .real [16, 16] qk1 with hs5d
  have e5 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s4.regs dt sh nm = some t → s5.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs5d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs5qk : s5.regs .real [16, 16] "qk" = some qk1 := by rw [hs5d, BlockState.setReg_same]
  have hs5sn : s5.regs .nat [] "start_n" = some (Tile.scalar SN) :=
    e5 (by decide) (e4 (by decide) (e3 (by decide) hs2sn))
  have hs5kp : s5.regs .ptr [16, 16] "k_ptrs" = some ⟨kpf⟩ :=
    e5 (by decide) (e4 (by decide) (e3 (by decide) hs2kp))
  have hs5q2 : s5.regs .real [16, 16] "q2" = some q2t :=
    e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hs0q2))))
  -- mem-transport: readMemValue at s5 = at s
  have hs5memk2 : ∀ ix : TileIndex [16, 16],
      s5.readMemValue .real (kpf ix).1 ((kpf ix).2 + SN * 32 + 16)
        = s.readMemValue .real (kpf ix).1 ((kpf ix).2 + SN * 32 + 16) := by
    intro ix; simp only [hs5d, hs4d, hs3d, hs2d, hs1d, hs0d, BlockState.setReg_readMemValue]
  -- ===== stmt 5: ifThen (2≥2) { ifThenElse true { k = load k2 } ; qk += dot(q2,k) } =====
  set k2t : Tile .real [16, 16] :=
    ⟨fun ix => s.readMemValue .real (kpf ix).1 ((kpf ix).2 + SN * 32 + 16)⟩ with hk2td
  set qk2 : Tile .real [16, 16] :=
    Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qk1 (Tile.dot [] q2t k2t) with hqk2d
  rw [stepStmts.cons_some (stepStmt_ifThen_true (by simp [evalOp])
    (show stepStmts _ s5 = some _ from by
      -- inner ifThenElse true: k = load (k_ptrs + start_n*32 + 16)
      rw [stepStmts.cons_some (stepStmt_ifThenElse_true (by simp [evalOp])
        (show stepStmts _ s5 = some _ from by
          rw [stepStmts.cons_some (stepStmt_assign_eq_some
            (show evalOp _ s5 = some k2t from by
              have h := bsa_load_k2_eval s5 SN 32 16 kpf hs5kp hs5sn
              rw [show (⟨fun ix : TileIndex [16, 16] =>
                    s5.readMemValue .real (kpf ix).1 ((kpf ix).2 + SN * 32 + 16)⟩ : Tile .real [16, 16])
                  = k2t from by
                refine Tile.ext (fun ix => ?_); rw [hk2td]; exact hs5memk2 ix] at h
              exact h))]
          rw [stepStmts.nil]))]
      -- s6 := s5 with k := k2t
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp _ (s5.setReg "k" .real [16, 16] k2t) = some qk2 from by
          refine bsa_qk2_dot_eval (s5.setReg "k" .real [16, 16] k2t) 16 16 16 qk1 q2t k2t ?_ ?_ ?_
          · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hs5qk
          · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hs5q2
          · rw [BlockState.setReg_same]))]
      rw [stepStmts.nil]))]
  -- s7 := (s5.setReg k k2t).setReg qk qk2
  set s7 := (s5.setReg "k" .real [16, 16] k2t).setReg "qk" .real [16, 16] qk2 with hs7d
  have e7 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → nm ≠ "k" → s5.regs dt sh nm = some t → s7.regs dt sh nm = some t := by
    intro dt sh nm t hne hnk h
    rw [hs7d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hnk]; exact h
  have hs7qk : s7.regs .real [16, 16] "qk" = some qk2 := by rw [hs7d, BlockState.setReg_same]
  -- ===== stmt 6: qk *= 1.0 =====
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_qk_scale_eval s7 16 16 1.0 qk2 hs7qk))]
  set qkS : Tile .real [16, 16] :=
    Tile.bop NumericDType.real.mul Broadcast.scalarR qk2
      (Tile.scalar (some (1.0 : ℝ) : WithBot ℝ)) with hqkSd
  set s8 := s7.setReg "qk" .real [16, 16] qkS with hs8d
  have e8 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s7.regs dt sh nm = some t → s8.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs8d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs8qk : s8.regs .real [16, 16] "qk" = some qkS := by rw [hs8d, BlockState.setReg_same]
  have hs8om : s8.regs .nat [16] "offs_m" = some (Tile.vec gm) :=
    e8 (by decide) (e7 (by decide) (by decide)
      (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hs0om)))))
  have hs8on : s8.regs .nat [16] "offs_n" = some (Tile.vec (fun j : Fin 16 => j.val)) :=
    e8 (by decide) (e7 (by decide) (by decide)
      (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hs0on)))))
  have hs8sn : s8.regs .nat [] "start_n" = some (Tile.scalar SN) :=
    e8 (by decide) (e7 (by decide) (by decide) hs5sn)
  -- ===== stmt 7: qk += where(causal additive) =====
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_where_eval s8 16 16 SN gm qkS hs8qk hs8om hs8on hs8sn))]
  set qkM : Tile .real [16, 16] :=
    ⟨fun ix : TileIndex [16, 16] =>
      NumericDType.real.add (qkS.data ix)
        (if SN + ix.2.1.val ≤ gm ix.1 then (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))⟩ with hqkMd
  set s9 := s8.setReg "qk" .real [16, 16] qkM with hs9d
  have e9 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s8.regs dt sh nm = some t → s9.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs9d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs9qk : s9.regs .real [16, 16] "qk" = some qkM := by rw [hs9d, BlockState.setReg_same]
  -- ===== stmt 8: m_ij = reduceMax(qk, 1) =====
  set mij : Tile .real [16] :=
    ⟨fun outIdx : TileIndex [16] =>
      (Finset.univ : Finset (Fin 16)).sup' Finset.univ_nonempty
        (fun k : Fin 16 => qkM.data (outIdx.1, k, PUnit.unit))⟩ with hmijd
  have hmijsome : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [16, 16].length) qkM = some mij := by
    unfold Tile.reduceMaxDrop
    rw [dif_pos (show 0 < TileShape.axisDim [16, 16] (⟨1, by simp⟩ : Fin [16, 16].length) from by decide)]
    refine congrArg some ?_
    refine Tile.ext (fun outIdx => ?_)
    obtain ⟨i, u⟩ := outIdx
    rw [hmijd]
    rfl
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_mij_eval s9 16 16 qkM mij hs9qk hmijsome))]
  set s10 := s9.setReg "m_ij" .real [16] mij with hs10d
  have e10 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_ij" → s9.regs dt sh nm = some t → s10.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs10d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs10mij : s10.regs .real [16] "m_ij" = some mij := by rw [hs10d, BlockState.setReg_same]
  have hs10qk : s10.regs .real [16, 16] "qk" = some qkM := e10 (by decide) hs9qk
  -- ===== stmt 9: p = exp(qk - m_ij[:,None]) =====
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_p_eval s10 16 16 (by simp) qkM mij hs10qk hs10mij))]
  set p0 : Tile .real [16, 16] :=
    Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkM
        (Tile.expandDim ⟨1, by simp⟩ mij)) with hp0d
  set s11 := s10.setReg "p" .real [16, 16] p0 with hs11d
  have e11 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s10.regs dt sh nm = some t → s11.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs11d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs11p : s11.regs .real [16, 16] "p" = some p0 := by rw [hs11d, BlockState.setReg_same]
  -- ===== stmt 10: l_ij = sum(p, 1) =====
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_lij_eval s11 16 16 p0 hs11p))]
  set lij : Tile .real [16] :=
    Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [16, 16].length) p0 with hlijd
  set s12 := s11.setReg "l_ij" .real [16] lij with hs12d
  have e12 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_ij" → s11.regs dt sh nm = some t → s12.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs12d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs12lij : s12.regs .real [16] "l_ij" = some lij := by rw [hs12d, BlockState.setReg_same]
  have hs12mij : s12.regs .real [16] "m_ij" = some mij := e12 (by decide) (e11 (by decide) hs10mij)
  -- m_i carried from entry: s12.m_i = mt
  have hs12mi : s12.regs .real [16] "m_i" = some mt :=
    e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide)
      (e7 (by decide) (by decide) (e4 (by decide) (e3 (by decide)
        (e2 (by decide) (e1 (by decide) hs0mi)))))))))
  -- ===== stmt 11: m_i_new = maximum(m_i, m_ij) =====
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_minew_eval s12 16 mt mij hs12mi hs12mij))]
  set minew : Tile .real [16] :=
    Tile.select
      (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mt mij) mt mij with hminewd
  set s13 := s12.setReg "m_i_new" .real [16] minew with hs13d
  have e13 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_i_new" → s12.regs dt sh nm = some t → s13.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs13d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs13minew : s13.regs .real [16] "m_i_new" = some minew := by rw [hs13d, BlockState.setReg_same]
  have hs13mi : s13.regs .real [16] "m_i" = some mt := e13 (by decide) hs12mi
  have hs13mij : s13.regs .real [16] "m_ij" = some mij := e13 (by decide) hs12mij
  -- ===== stmt 12: alpha = exp(m_i - m_i_new) =====
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_alpha_eval s13 16 mt minew hs13mi hs13minew))]
  set alpha : Tile .real [16] :=
    Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mt minew) with halphad
  set s14 := s13.setReg "alpha" .real [16] alpha with hs14d
  have e14 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "alpha" → s13.regs dt sh nm = some t → s14.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs14d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs14alpha : s14.regs .real [16] "alpha" = some alpha := by rw [hs14d, BlockState.setReg_same]
  have hs14mij : s14.regs .real [16] "m_ij" = some mij := e14 (by decide) hs13mij
  have hs14minew : s14.regs .real [16] "m_i_new" = some minew := e14 (by decide) hs13minew
  -- ===== stmt 13: beta = exp(m_ij - m_i_new) =====
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_beta_eval s14 16 mij minew hs14mij hs14minew))]
  set beta : Tile .real [16] :=
    Tile.uop WithBot.realExp
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mij minew) with hbetad
  set s15 := s14.setReg "beta" .real [16] beta with hs15d
  have e15 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "beta" → s14.regs dt sh nm = some t → s15.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs15d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs15beta : s15.regs .real [16] "beta" = some beta := by rw [hs15d, BlockState.setReg_same]
  have hs15alpha : s15.regs .real [16] "alpha" = some alpha := e15 (by decide) hs14alpha
  have hs15lij : s15.regs .real [16] "l_ij" = some lij :=
    e15 (by decide) (e14 (by decide) (e13 (by decide) hs12lij))
  -- l_i carried from entry
  have hs15li : s15.regs .real [16] "l_i" = some lt :=
    e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide)
      (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (by decide)
        (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hs0li))))))))))))
  -- ===== stmt 14: l_i_new = alpha*l_i + beta*l_ij =====
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_linew_eval s15 16 alpha lt beta lij hs15alpha hs15li hs15beta hs15lij))]
  set linew : Tile .real [16] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
      (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) alpha lt)
      (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) beta lij) with hlinewd
  set s16 := s15.setReg "l_i_new" .real [16] linew with hs16d
  have e16 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i_new" → s15.regs dt sh nm = some t → s16.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs16d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs16linew : s16.regs .real [16] "l_i_new" = some linew := by rw [hs16d, BlockState.setReg_same]
  have hs16beta : s16.regs .real [16] "beta" = some beta := e16 (by decide) hs15beta
  -- ===== stmt 15: p_scale = beta / l_i_new =====
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_pscale_eval s16 16 beta linew hs16beta hs16linew))]
  set pscale : Tile .real [16] :=
    Tile.bop NumericDType.real.div (Broadcast.consSame Broadcast.nil) beta linew with hpscaled
  set s17 := s16.setReg "p_scale" .real [16] pscale with hs17d
  have e17 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p_scale" → s16.regs dt sh nm = some t → s17.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs17d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs17ps : s17.regs .real [16] "p_scale" = some pscale := by rw [hs17d, BlockState.setReg_same]
  have hs17p : s17.regs .real [16, 16] "p" = some p0 :=
    e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide)
      (e12 (by decide) hs11p)))))
  -- ===== stmt 16: p *= p_scale[:,None] =====
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_p_rescale_eval s17 16 16 (by simp) p0 pscale hs17p hs17ps))]
  set pfinal : Tile .real [16, 16] :=
    Tile.bop NumericDType.real.mul
      (Broadcast.consSame (Broadcast.consR Broadcast.nil)) p0
      (Tile.expandDim ⟨1, by simp⟩ pscale) with hpfinald
  set s18 := s17.setReg "p" .real [16, 16] pfinal with hs18d
  have e18 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s17.regs dt sh nm = some t → s18.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs18d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs18p : s18.regs .real [16, 16] "p" = some pfinal := by rw [hs18d, BlockState.setReg_same]
  -- l_i / l_i_new / alpha for acc_scale
  have hs18li : s18.regs .real [16] "l_i" = some lt :=
    e18 (by decide) (e17 (by decide) (e16 (by decide) hs15li))
  have hs18linew : s18.regs .real [16] "l_i_new" = some linew :=
    e18 (by decide) (e17 (by decide) hs16linew)
  have hs18alpha : s18.regs .real [16] "alpha" = some alpha :=
    e18 (by decide) (e17 (by decide) (e16 (by decide) hs15alpha))
  -- ===== stmt 17: acc_scale = l_i / l_i_new * alpha =====
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_accscale_eval s18 16 lt linew alpha hs18li hs18linew hs18alpha))]
  set accscale : Tile .real [16] :=
    Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
      (Tile.bop NumericDType.real.div (Broadcast.consSame Broadcast.nil) lt linew) alpha with haccscaled
  set s19 := s18.setReg "acc_scale" .real [16] accscale with hs19d
  have e19 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc_scale" → s18.regs dt sh nm = some t → s19.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs19d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs19as : s19.regs .real [16] "acc_scale" = some accscale := by rw [hs19d, BlockState.setReg_same]
  -- acc carried from entry
  have hs19acc : s19.regs .real [16, 16] "acc" = some acct :=
    e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide)
      (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide)
        (e9 (by decide) (e8 (by decide) (e7 (by decide) (by decide)
          (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hs0acc))))))))))))))))
  -- ===== stmt 18: acc *= acc_scale[:,None] =====
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_acc_rescale_eval s19 16 16 (by simp) acct accscale hs19acc hs19as))]
  set accMul : Tile .real [16, 16] :=
    Tile.bop NumericDType.real.mul
      (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acct
      (Tile.expandDim ⟨1, by simp⟩ accscale) with haccMuld
  set s20 := s19.setReg "acc" .real [16, 16] accMul with hs20d
  have e20 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s19.regs dt sh nm = some t → s20.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs20d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs20acc : s20.regs .real [16, 16] "acc" = some accMul := by rw [hs20d, BlockState.setReg_same]
  have hs20as : s20.regs .real [16] "acc_scale" = some accscale := e20 (by decide) hs19as
  -- acc2 carried from entry
  have hs20acc2 : s20.regs .real [16, 16] "acc2" = some acc2t :=
    e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide)
      (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide)
        (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (by decide)
          (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hs0acc2)))))))))))))))))
  -- ===== stmt 19: ifThen (2≥2) { acc2 *= acc_scale[:,None] } =====
  set acc2Mul : Tile .real [16, 16] :=
    Tile.bop NumericDType.real.mul
      (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acc2t
      (Tile.expandDim ⟨1, by simp⟩ accscale) with hacc2Muld
  rw [stepStmts.cons_some (stepStmt_ifThen_true (by simp [evalOp])
    (show stepStmts _ s20 = some _ from by
      erw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bsa_acc2_rescale_eval s20 16 16 (by simp) acc2t accscale hs20acc2 hs20as))]
      rw [stepStmts.nil]))]
  set s21 := s20.setReg "acc2" .real [16, 16] acc2Mul with hs21d
  have e21 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc2" → s20.regs dt sh nm = some t → s21.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs21d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs21p : s21.regs .real [16, 16] "p" = some pfinal :=
    e21 (by decide) (e20 (by decide) (e19 (by decide) hs18p))
  -- ===== stmt 20: p = p (identity cast) =====
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_pcast_eval s21 16 16 pfinal hs21p))]
  set s22 := s21.setReg "p" .real [16, 16] pfinal with hs22d
  have e22 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s21.regs dt sh nm = some t → s22.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs22d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs22p : s22.regs .real [16, 16] "p" = some pfinal := by rw [hs22d, BlockState.setReg_same]
  have hs22vp : s22.regs .ptr [16, 16] "v_ptrs" = some ⟨vpf⟩ :=
    e22 (by decide) (e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide)
      (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide)
        (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide)
          (e7 (by decide) (by decide) (e4 (by decide) (e3 (by decide)
            (e2 (by decide) (e1 (by decide) hs0vp)))))))))))))))))))
  have hs22sn : s22.regs .nat [] "start_n" = some (Tile.scalar SN) :=
    e22 (by decide) (e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide)
      (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide)
        (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) hs8sn)))))))))))))
  -- mem-transport: readMemValue at s22 = at s
  have hs22memv : ∀ ix : TileIndex [16, 16],
      s22.readMemValue .real (vpf ix).1 ((vpf ix).2 + SN * 32)
        = s.readMemValue .real (vpf ix).1 ((vpf ix).2 + SN * 32) := by
    intro ix
    simp only [hs22d, hs21d, hs20d, hs19d, hs18d, hs17d, hs16d, hs15d, hs14d, hs13d, hs12d, hs11d,
      hs10d, hs9d, hs8d, hs7d, hs5d, hs4d, hs3d, hs2d, hs1d, hs0d, BlockState.setReg_readMemValue]
  -- ===== stmt 21: ifThenElse true { v = load (v_ptrs + start_n*32) } =====
  set vt : Tile .real [16, 16] :=
    ⟨fun ix => s.readMemValue .real (vpf ix).1 ((vpf ix).2 + SN * 32)⟩ with hvtd
  rw [stepStmts.cons_some (stepStmt_ifThenElse_true (by simp [evalOp])
    (show stepStmts _ s22 = some _ from by
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp _ s22 = some vt from by
          have h := bsa_load_v_eval s22 SN 32 vpf hs22vp hs22sn
          rw [show (⟨fun ix : TileIndex [16, 16] =>
                s22.readMemValue .real (vpf ix).1 ((vpf ix).2 + SN * 32)⟩ : Tile .real [16, 16])
              = vt from by
            refine Tile.ext (fun ix => ?_); rw [hvtd]; exact hs22memv ix] at h
          exact h))]
      rw [stepStmts.nil]))]
  set s23 := s22.setReg "v" .real [16, 16] vt with hs23d
  have e23 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "v" → s22.regs dt sh nm = some t → s23.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs23d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs23v : s23.regs .real [16, 16] "v" = some vt := by rw [hs23d, BlockState.setReg_same]
  have hs23p : s23.regs .real [16, 16] "p" = some pfinal := e23 (by decide) hs22p
  have hs23acc : s23.regs .real [16, 16] "acc" = some accMul :=
    e23 (by decide) (e22 (by decide) (e21 (by decide) hs20acc))
  -- ===== stmt 22: acc += dot(p, v) =====
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (bsa_acc_eval s23 16 16 16 accMul pfinal vt hs23acc hs23p hs23v))]
  set accFinal : Tile .real [16, 16] :=
    Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) accMul
      (Tile.dot [] pfinal vt) with haccFinald
  set s24 := s23.setReg "acc" .real [16, 16] accFinal with hs24d
  have e24 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s23.regs dt sh nm = some t → s24.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs24d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs24acc : s24.regs .real [16, 16] "acc" = some accFinal := by rw [hs24d, BlockState.setReg_same]
  have hs24p : s24.regs .real [16, 16] "p" = some pfinal := e24 (by decide) hs23p
  have hs24acc2 : s24.regs .real [16, 16] "acc2" = some acc2Mul :=
    e24 (by decide) (e23 (by decide) (e22 (by decide) (by rw [hs21d, BlockState.setReg_same])))
  have hs24vp : s24.regs .ptr [16, 16] "v_ptrs" = some ⟨vpf⟩ :=
    e24 (by decide) (e23 (by decide) hs22vp)
  have hs24sn : s24.regs .nat [] "start_n" = some (Tile.scalar SN) :=
    e24 (by decide) (e23 (by decide) hs22sn)
  -- mem-transport: readMemValue v2 at s24 = at s
  have hs24memv2 : ∀ ix : TileIndex [16, 16],
      s24.readMemValue .real (vpf ix).1 ((vpf ix).2 + SN * 32 + 16)
        = s.readMemValue .real (vpf ix).1 ((vpf ix).2 + SN * 32 + 16) := by
    intro ix
    simp only [hs24d, hs23d, hs22d, hs21d, hs20d, hs19d, hs18d, hs17d, hs16d, hs15d, hs14d, hs13d,
      hs12d, hs11d, hs10d, hs9d, hs8d, hs7d, hs5d, hs4d, hs3d, hs2d, hs1d, hs0d,
      BlockState.setReg_readMemValue]
  -- ===== stmt 23: ifThen (2≥2) { ifThenElse true { v = load v2 } ; acc2 += dot(p,v) } =====
  set v2t : Tile .real [16, 16] :=
    ⟨fun ix => s.readMemValue .real (vpf ix).1 ((vpf ix).2 + SN * 32 + 16)⟩ with hv2td
  set acc2Final : Tile .real [16, 16] :=
    Tile.bop NumericDType.real.add
      (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acc2Mul
      (Tile.dot [] pfinal v2t) with hacc2Finald
  rw [stepStmts.cons_some (stepStmt_ifThen_true (by simp [evalOp])
    (show stepStmts _ s24 = some _ from by
      rw [stepStmts.cons_some (stepStmt_ifThenElse_true (by simp [evalOp])
        (show stepStmts _ s24 = some _ from by
          rw [stepStmts.cons_some (stepStmt_assign_eq_some
            (show evalOp _ s24 = some v2t from by
              have h := bsa_load_v2_eval s24 SN 32 16 vpf hs24vp hs24sn
              rw [show (⟨fun ix : TileIndex [16, 16] =>
                    s24.readMemValue .real (vpf ix).1 ((vpf ix).2 + SN * 32 + 16)⟩ : Tile .real [16, 16])
                  = v2t from by
                refine Tile.ext (fun ix => ?_); rw [hv2td]; exact hs24memv2 ix] at h
              exact h))]
          rw [stepStmts.nil]))]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp _ (s24.setReg "v" .real [16, 16] v2t) = some acc2Final from by
          refine bsa_acc2_eval (s24.setReg "v" .real [16, 16] v2t) 16 16 16 acc2Mul pfinal v2t ?_ ?_ ?_
          · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hs24acc2
          · rw [BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hs24p
          · rw [BlockState.setReg_same]))]
      rw [stepStmts.nil]))]
  set s26 := (s24.setReg "v" .real [16, 16] v2t).setReg "acc2" .real [16, 16] acc2Final with hs26d
  have e26 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc2" → nm ≠ "v" → s24.regs dt sh nm = some t → s26.regs dt sh nm = some t := by
    intro dt sh nm t hne hnv h
    rw [hs26d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hnv]; exact h
  have hs26acc2 : s26.regs .real [16, 16] "acc2" = some acc2Final := by rw [hs26d, BlockState.setReg_same]
  have hs26linew : s26.regs .real [16] "l_i_new" = some linew :=
    e26 (by decide) (by decide) (e24 (by decide) (e23 (by decide) (e22 (by decide)
      (e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) hs16linew))))))))
  have hs26minew : s26.regs .real [16] "m_i_new" = some minew :=
    e26 (by decide) (by decide) (e24 (by decide) (e23 (by decide) (e22 (by decide)
      (e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide)
        (e16 (by decide) (e15 (by decide) (e14 (by decide) hs13minew)))))))))))
  -- ===== stmt 24: l_i = l_i_new =====
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_reg_carry_eval s26 16 "l_i_new" linew hs26linew))]
  set s27 := s26.setReg "l_i" .real [16] linew with hs27d
  have e27 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i" → s26.regs dt sh nm = some t → s27.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs27d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs27li : s27.regs .real [16] "l_i" = some linew := by rw [hs27d, BlockState.setReg_same]
  have hs27minew : s27.regs .real [16] "m_i_new" = some minew := e27 (by decide) hs26minew
  -- ===== stmt 25: m_i = m_i_new =====
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (bsa_reg_carry_eval s27 16 "m_i_new" minew hs27minew))]
  rw [stepStmts.nil]
  set s28 := s27.setReg "m_i" .real [16] minew with hs28d
  refine ⟨s28, rfl, ?_, ?_, ?_, ?_, ?_⟩
  -- pids preservation
  · rw [hs28d, BlockState.setReg_pids, hs27d, BlockState.setReg_pids, hs26d, BlockState.setReg_pids,
      BlockState.setReg_pids, hs24d, BlockState.setReg_pids, hs23d, BlockState.setReg_pids,
      hs22d, BlockState.setReg_pids, hs21d, BlockState.setReg_pids, hs20d, BlockState.setReg_pids,
      hs19d, BlockState.setReg_pids, hs18d, BlockState.setReg_pids, hs17d, BlockState.setReg_pids,
      hs16d, BlockState.setReg_pids, hs15d, BlockState.setReg_pids, hs14d, BlockState.setReg_pids,
      hs13d, BlockState.setReg_pids, hs12d, BlockState.setReg_pids, hs11d, BlockState.setReg_pids,
      hs10d, BlockState.setReg_pids, hs9d, BlockState.setReg_pids, hs8d, BlockState.setReg_pids,
      hs7d, BlockState.setReg_pids, BlockState.setReg_pids, hs5d, BlockState.setReg_pids,
      hs4d, BlockState.setReg_pids, hs3d, BlockState.setReg_pids, hs2d, BlockState.setReg_pids,
      hs1d, BlockState.setReg_pids, hs0d, BlockState.setReg_pids]
  -- mem preservation
  · funext rg o
    rw [hs28d, BlockState.setReg_mem, hs27d, BlockState.setReg_mem, hs26d, BlockState.setReg_mem,
      BlockState.setReg_mem, hs24d, BlockState.setReg_mem, hs23d, BlockState.setReg_mem,
      hs22d, BlockState.setReg_mem, hs21d, BlockState.setReg_mem, hs20d, BlockState.setReg_mem,
      hs19d, BlockState.setReg_mem, hs18d, BlockState.setReg_mem, hs17d, BlockState.setReg_mem,
      hs16d, BlockState.setReg_mem, hs15d, BlockState.setReg_mem, hs14d, BlockState.setReg_mem,
      hs13d, BlockState.setReg_mem, hs12d, BlockState.setReg_mem, hs11d, BlockState.setReg_mem,
      hs10d, BlockState.setReg_mem, hs9d, BlockState.setReg_mem, hs8d, BlockState.setReg_mem,
      hs7d, BlockState.setReg_mem, BlockState.setReg_mem, hs5d, BlockState.setReg_mem,
      hs4d, BlockState.setReg_mem, hs3d, BlockState.setReg_mem, hs2d, BlockState.setReg_mem,
      hs1d, BlockState.setReg_mem, hs0d, BlockState.setReg_mem]
  -- undef preservation (only setRegs, no stores)
  · intro rg o
    simp only [hs28d, hs27d, hs26d, hs24d, hs23d, hs22d, hs21d, hs20d, hs19d, hs18d,
      hs17d, hs16d, hs15d, hs14d, hs13d, hs12d, hs11d, hs10d, hs9d, hs8d, hs7d, hs5d,
      hs4d, hs3d, hs2d, hs1d, hs0d, BlockState.setReg_undef]
  -- preservation of un-assigned registers
  · intro dt sh nm t hnm hreg
    simp only [List.mem_cons, not_or] at hnm
    obtain ⟨h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14,
      h15, h16, h17, h18, -⟩ := hnm
    rw [hs28d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h16,
      hs27d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h15,
      hs26d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h18,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h14,
      hs24d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h17,
      hs23d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h14,
      hs22d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h6,
      hs21d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h18,
      hs20d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h17,
      hs19d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h13,
      hs18d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h6,
      hs17d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h12,
      hs16d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h11,
      hs15d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h10,
      hs14d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h9,
      hs13d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h8,
      hs12d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h7,
      hs11d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h6,
      hs10d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h5,
      hs9d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      hs8d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      hs7d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      hs5d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      hs4d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h4,
      hs3d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h3,
      hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h2,
      hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h1,
      hs0d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ h0]
    exact hreg
  -- exposed registers
  · refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · rw [hs28d, BlockState.setReg_same]
    · rw [hs28d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hs27li
    · rw [hs28d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        hs27d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      exact e26 (by decide) (by decide) hs24p
    · rw [hs28d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        hs27d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      exact e26 (by decide) (by decide) hs24acc
    · rw [hs28d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        hs27d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
      exact hs26acc2

/-! ## Block-update math bridge (`bsa_attn_step` support)

These lemmas bridge the `bsaLoopBody_steps` symbolic outputs (per-cell `Tile`
arithmetic over `WithBot ℝ`) to the gathered-causal streaming recurrences
`bsaMPartial`/`bsaLPartial`/`bsaOPartial` at iteration `c + 1`. The score and
value gathers are supplied as abstract per-cell hypotheses (`hScore`/`hMask`/
`hVal`/`hVal2`), discharged by the exec assembly at instantiation time. -/

open BSAMathCausal in
/-- Per-cell score bridge: the kernel's causal-masked `qkM` tile cell equals the
gathered-causal `maskedScore` at the block-`c` lane, given the additive-causal
mask predicate (`hMask`) and the scaled-dot-equals-`gScore` hypothesis
(`hScore`). The kernel applies the additive `where` (0 / ⊥), while `maskedScore`
selects the score or ⊥ on the same predicate. -/
theorem bsa_qkM_cell_eq_maskedScore
    (qStart numKVBlocks c : Nat) (hc : c + 1 ≤ numKVBlocks)
    (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 16] → ℝ)
    (Kg : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (SN : Nat) (gm : Fin 16 → Nat)
    (qkScaled : Tile .real [16, 16])
    (i jL : Fin 16)
    (hMask : (SN + jL.val ≤ gm i) ↔
      (gpos (StreamingAccumulator.blockIndex 16 numKVBlocks c hc jL) ≤ qStart + i.val))
    (hScore : qkScaled.data (i, jL, PUnit.unit) =
      (some (gScore Qg Kg scale i
        (StreamingAccumulator.blockIndex 16 numKVBlocks c hc jL)) : WithBot ℝ)) :
    NumericDType.real.add (qkScaled.data (i, jL, PUnit.unit))
        (if SN + jL.val ≤ gm i then (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
      = maskedScore qStart gpos Qg Kg scale i
          (StreamingAccumulator.blockIndex 16 numKVBlocks c hc jL) := by
  rw [hScore]
  by_cases h : SN + jL.val ≤ gm i
  · rw [if_pos h, maskedScore_of_le qStart gpos Qg Kg scale i _ (hMask.mp h)]
    show (Option.map₂ (· + ·) (some _) (some (0 : ℝ)) : WithBot ℝ) = _
    rw [Option.map₂_some_some, add_zero]; rfl
  · rw [if_neg h, maskedScore_of_not_le qStart gpos Qg Kg scale i _ (by rw [← hMask]; exact h)]
    rfl

open BSAMathCausal in
/-- The kernel's per-row block max `m_ij = sup' (qkM row)` equals the gathered
`sup`-of-`maskedScore` over the block-`c` lanes (the inner term of
`bsaMPartial_succ_of_lt`). Built from the per-cell `bsa_qkM_cell_eq_maskedScore`
bridge plus `Finset.sup'_eq_sup`. -/
theorem bsa_mij_eq_sup_maskedScore
    (qStart numKVBlocks c : Nat) (hc : c + 1 ≤ numKVBlocks)
    (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 16] → ℝ)
    (Kg : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (SN : Nat) (gm : Fin 16 → Nat)
    (qkScaled : Tile .real [16, 16])
    (i : Fin 16)
    (hMask : ∀ jL : Fin 16, (SN + jL.val ≤ gm i) ↔
      (gpos (StreamingAccumulator.blockIndex 16 numKVBlocks c hc jL) ≤ qStart + i.val))
    (hScore : ∀ jL : Fin 16, qkScaled.data (i, jL, PUnit.unit) =
      (some (gScore Qg Kg scale i
        (StreamingAccumulator.blockIndex 16 numKVBlocks c hc jL)) : WithBot ℝ)) :
    (Finset.univ : Finset (Fin 16)).sup' Finset.univ_nonempty
        (fun k : Fin 16 =>
          NumericDType.real.add (qkScaled.data (i, k, PUnit.unit))
            (if SN + k.val ≤ gm i then (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ)))
      = (Finset.univ : Finset (Fin 16)).sup
          (fun jLocal : Fin 16 =>
            maskedScore qStart gpos Qg Kg scale i
              (StreamingAccumulator.blockIndex 16 numKVBlocks c hc jLocal)) := by
  rw [Finset.sup'_eq_sup]
  refine Finset.sup_congr rfl (fun jL _ => ?_)
  exact bsa_qkM_cell_eq_maskedScore qStart numKVBlocks c hc gpos Qg Kg scale SN gm
    qkScaled i jL (hMask jL) (hScore jL)

/-! ## `bsa_attn_step`: loop-body advances the invariant by one CSR block

Given `bsaInvariant … c`, the lowered `bsaLoopBody` (executed via the banked
`bsaLoopBody_steps`) advances the live `m_i`/`l_i`/`acc`/`acc2` accumulators to
`bsaInvariant … (c+1)`. The execution is `bsaLoopBody_steps`; the remainder is the
WithBot↔ℝ math bridge of the exposed symbolic tiles onto `bsaMPartial`/`bsaLPartial`/
`bsaOPartial` at `c+1`, mirroring `nopad_attn_step`. -/

open BSAMathCausal in
/-- `tl.maximum` lowered to `where(m_i > m_ij, m_i, m_ij)` is the WithBot max. -/
theorem bsa_select_gt_eq_max (a b : WithBot ℝ) :
    (if (ComparableDType.real.gt a b = Bool.true) then a else b) = max a b := by
  by_cases h : a > b
  · rw [if_pos (by simpa using h)]; exact (max_eq_left (le_of_lt h)).symm
  · rw [if_neg (by simpa using h)]; exact (max_eq_right (le_of_not_gt h)).symm

open BSAMathCausal in
/-- Per-summand exp telescoping: `β·p0 = exp(score − m_new)` in `WithBot ℝ`,
where `β = exp(m_ij − m_new)`, `p0 = exp(score − m_ij)`. Handles the masked
(`score = ⊥`) lane (both sides `0`) and the visible lane (`m_ij`, `m_new` are
finite). -/
theorem bsa_beta_p0_eq_exp (A B Mn : WithBot ℝ)
    (hB : A ≠ ⊥ → ∃ b : ℝ, B = (b : WithBot ℝ))
    (hMn : A ≠ ⊥ → ∃ n : ℝ, Mn = (n : WithBot ℝ)) :
    WithBot.realMul (WithBot.realExp (WithBot.realSub B Mn))
        (WithBot.realExp (WithBot.realSub A B))
      = WithBot.realExp (WithBot.realSub A Mn) := by
  cases hA : A with
  | bot =>
    rw [realExp_eq_some_unbotD (WithBot.realSub B Mn)]
    simp only [WithBot.realSub_bot_left, WithBot.realExp_bot]
    show WithBot.realMul (some _) (some (0:ℝ)) = some (0:ℝ)
    rw [show (WithBot.realMul (some ((WithBot.realExp (WithBot.realSub B Mn)).unbotD 0))
        (some (0:ℝ)) : WithBot ℝ) = some (((WithBot.realExp (WithBot.realSub B Mn)).unbotD 0) * 0) from rfl,
      mul_zero]
  | coe a =>
    obtain ⟨b, hb⟩ := hB (by simp [hA])
    obtain ⟨n, hn⟩ := hMn (by simp [hA])
    subst hb; subst hn
    simp only [WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.realMul_coe_coe,
      WithBot.coe_inj]
    rw [← Real.exp_add]; ring_nf

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open BSAMathCausal in
/-- **`bsa_step_mi`** (standalone m_i conjunct). The loop body's
`m_i_new = where(m_i > m_ij, m_i, m_ij)` tile, with `m_i` holding `bsaMPartial c`
(invariant) and `m_ij = sup' (qkM row)`, equals `bsaMPartial (c+1)`. -/
theorem bsa_step_mi
    (qStart numKVBlocks c : Nat) (hc : c < numKVBlocks)
    (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 16] → ℝ)
    (Kg : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (SN : Nat) (gm : Fin 16 → Nat)
    (qkScaled : Tile .real [16, 16])
    (hMask : ∀ (i jL : Fin 16), (SN + jL.val ≤ gm i) ↔
      (gpos (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jL)
        ≤ qStart + i.val))
    (hScore : ∀ (i jL : Fin 16), qkScaled.data (i, jL, PUnit.unit) =
      (some (gScore Qg Kg scale i
        (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jL)) : WithBot ℝ)) :
    (Tile.select
      (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil)
        (⟨fun i : TileIndex [16] => bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale c i.1⟩ : Tile .real [16])
        (⟨fun outIdx : TileIndex [16] =>
          (Finset.univ : Finset (Fin 16)).sup' Finset.univ_nonempty
            (fun k : Fin 16 =>
              NumericDType.real.add (qkScaled.data (outIdx.1, k, PUnit.unit))
                (if SN + k.val ≤ gm outIdx.1 then (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ)))⟩
          : Tile .real [16]))
      (⟨fun i : TileIndex [16] => bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale c i.1⟩ : Tile .real [16])
      (⟨fun outIdx : TileIndex [16] =>
        (Finset.univ : Finset (Fin 16)).sup' Finset.univ_nonempty
          (fun k : Fin 16 =>
            NumericDType.real.add (qkScaled.data (outIdx.1, k, PUnit.unit))
              (if SN + k.val ≤ gm outIdx.1 then (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ)))⟩
        : Tile .real [16]))
    = (Tile.vec (fun i : Fin 16 =>
        bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i)) := by
  refine Tile.ext (fun outIdx => ?_)
  obtain ⟨i, u⟩ := outIdx
  simp only [Tile.select_data, Tile.cop_data, Broadcast.leftIndex, Broadcast.rightIndex,
    Tile.vec, Tile.scalar_data_index]
  rw [bsaMPartial_succ_of_lt qStart numKVBlocks gpos Qg Kg scale c hc i]
  have hbridge := bsa_mij_eq_sup_maskedScore qStart numKVBlocks c (Nat.succ_le_iff.mpr hc)
    gpos Qg Kg scale SN gm qkScaled i (hMask i) (hScore i)
  split_ifs with h
  · rw [← hbridge]; exact (max_eq_left (le_of_lt (by simpa using h))).symm
  · rw [← hbridge]; exact (max_eq_right (le_of_not_gt (by simpa using h))).symm

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open BSAMathCausal in
/-- **`bsa_step_li`** (standalone l_i conjunct). The loop body's
`l_i_new = alpha·l_i + beta·l_ij` tile equals `bsaLPartial (c+1)` per cell, given
`m_i = bsaMPartial c`, `l_i = some (bsaLPartial c)`, and the score/mask bridges. -/
theorem bsa_step_li
    (qStart numKVBlocks c : Nat) (hc : c < numKVBlocks)
    (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 16] → ℝ)
    (Kg : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (SN : Nat) (gm : Fin 16 → Nat)
    (qkScaled : Tile .real [16, 16])
    (hMask : ∀ (i jL : Fin 16), (SN + jL.val ≤ gm i) ↔
      (gpos (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jL)
        ≤ qStart + i.val))
    (hScore : ∀ (i jL : Fin 16), qkScaled.data (i, jL, PUnit.unit) =
      (some (gScore Qg Kg scale i
        (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jL)) : WithBot ℝ)) :
    let qkM : Tile .real [16, 16] :=
      ⟨fun ix : TileIndex [16, 16] =>
        NumericDType.real.add (qkScaled.data ix)
          (if SN + ix.2.1.val ≤ gm ix.1 then (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))⟩
    let mij : Tile .real [16] :=
      ⟨fun outIdx : TileIndex [16] =>
        (Finset.univ : Finset (Fin 16)).sup' Finset.univ_nonempty
          (fun k : Fin 16 => qkM.data (outIdx.1, k, PUnit.unit))⟩
    let mt : Tile .real [16] :=
      ⟨fun i : TileIndex [16] => bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale c i.1⟩
    let minew : Tile .real [16] :=
      Tile.select
        (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mt mij) mt mij
    let p0 : Tile .real [16, 16] :=
      Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkM
          (Tile.expandDim ⟨1, by simp⟩ mij))
    let lij : Tile .real [16] :=
      Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [16, 16].length) p0
    let alpha : Tile .real [16] :=
      Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mt minew)
    let beta : Tile .real [16] :=
      Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mij minew)
    let lt : Tile .real [16] :=
      ⟨fun i : TileIndex [16] =>
        (some (bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale c i.1) : WithBot ℝ)⟩
    Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) alpha lt)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) beta lij)
      = (Tile.vec (fun i : Fin 16 =>
          (some (bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i) : WithBot ℝ))) := by
  intro qkM mij mt minew p0 lij alpha beta lt
  -- minew = bsaMPartial (c+1) per cell
  have hminew : ∀ i : Fin 16, minew.data (i, PUnit.unit)
      = bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i := by
    intro i
    have := congrArg (fun t : Tile .real [16] => t.data (i, PUnit.unit))
      (bsa_step_mi qStart numKVBlocks c hc gpos Qg Kg scale SN gm qkScaled hMask hScore)
    simpa [minew, mt, mij, qkM, Tile.vec, Tile.scalar_data_index] using this
  -- mij = sup maskedScore per cell (finite when first key visible? — value form)
  refine Tile.ext (fun outIdx => ?_)
  obtain ⟨i, u⟩ := outIdx
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex]
  -- expose alpha, beta, lt, lij cells
  simp only [alpha, beta, lt, lij, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex,
    Broadcast.rightIndex, Tile.reduceSumDrop_data, Tile.expandDim_data,
    TileShape.dropInsertedIndex, TileShape.insertAxisIndex, mt, Tile.scalar_data_index,
    hminew i]
  -- reduce RHS Tile.vec cell, then unfold the target recurrence
  show NumericDType.real.add
      (NumericDType.real.mul
        (WithBot.realExp
          (NumericDType.real.sub (bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale c i)
            (bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i)))
        (some (bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale c i)))
      (NumericDType.real.mul
        (WithBot.realExp
          (NumericDType.real.sub (mij.data (i, PUnit.unit))
            (bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i)))
        (∑ x, p0.data (i, x, PUnit.unit)))
    = (some (bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i) : WithBot ℝ)
  rw [bsaLPartial_succ_of_lt qStart numKVBlocks gpos Qg Kg scale c hc i]
  -- the qkM cell bridge: qkM (i, k) = maskedScore i (blockIndex c k)
  have hqkM : ∀ k : Fin 16, qkM.data (i, k, PUnit.unit)
      = maskedScore qStart gpos Qg Kg scale i
          (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) k) := by
    intro k
    exact bsa_qkM_cell_eq_maskedScore qStart numKVBlocks c (Nat.succ_le_iff.mpr hc)
      gpos Qg Kg scale SN gm qkScaled i k (hMask i k) (hScore i k)
  -- mij cell = sup maskedScore
  have hmijcell : mij.data (i, PUnit.unit)
      = (Finset.univ : Finset (Fin 16)).sup
          (fun jLocal : Fin 16 => maskedScore qStart gpos Qg Kg scale i
            (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jLocal)) := by
    show (Finset.univ : Finset (Fin 16)).sup' Finset.univ_nonempty
        (fun k : Fin 16 => qkM.data (i, k, PUnit.unit)) = _
    rw [Finset.sup'_eq_sup]
    exact Finset.sup_congr rfl (fun k _ => hqkM k)
  -- p0 cell = realExp (sub (maskedScore k) (mij cell))
  have hp0 : ∀ k : Fin 16, p0.data (i, k, PUnit.unit)
      = WithBot.realExp (WithBot.realSub
          (maskedScore qStart gpos Qg Kg scale i
            (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) k))
          (mij.data (i, PUnit.unit))) := by
    intro k
    simp only [p0, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, TileShape.dropInsertedIndex, hqkM k]
    rfl
  -- Both LHS terms are `some` of reals; reduce to a real equation.
  set mnew := bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i with hmnewdef
  set supM : WithBot ℝ := (Finset.univ : Finset (Fin 16)).sup
    (fun jLocal : Fin 16 => maskedScore qStart gpos Qg Kg scale i
      (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jLocal)) with hsupMdef
  have hmijsup : mij.data (i, PUnit.unit) = supM := hmijcell
  -- Σ p0 cell = some (Σ p0val)
  have hp0some : ∀ k : Fin 16, p0.data (i, k, PUnit.unit)
      = some ((p0.data (i, k, PUnit.unit)).unbotD 0) := by
    intro k; rw [hp0 k]; exact realExp_eq_some_unbotD _
  have hsump0 : (∑ x, p0.data (i, x, PUnit.unit))
      = (some (∑ x, (p0.data (i, x, PUnit.unit)).unbotD 0) : WithBot ℝ) := by
    rw [show (∑ x, p0.data (i, x, PUnit.unit))
        = @Finset.sum (Fin 16) (WithBot ℝ) _ Finset.univ
            (fun x => (some ((p0.data (i, x, PUnit.unit)).unbotD 0) : WithBot ℝ)) from
      Finset.sum_congr rfl (fun k _ => hp0some k)]
    rw [WithBot.sum_someTerm_eq_some]
  -- first term = some
  rw [show NumericDType.real.mul
        (WithBot.realExp (NumericDType.real.sub
          (bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale c i) mnew))
        (some (bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale c i))
      = (some ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale c i) mnew)).unbotD 0 *
          bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale c i) : WithBot ℝ) from by
    show WithBot.realMul (WithBot.realExp (WithBot.realSub _ _)) (some _) = _
    rw [realExp_eq_some_unbotD]; rfl]
  -- second term = some
  rw [hmijsup]
  rw [show NumericDType.real.mul
        (WithBot.realExp (NumericDType.real.sub supM mnew))
        (∑ x, p0.data (i, x, PUnit.unit))
      = (some ((WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y) supM mnew)).unbotD 0 *
          (∑ x, (p0.data (i, x, PUnit.unit)).unbotD 0)) : WithBot ℝ) from by
    rw [hsump0]
    show WithBot.realMul (WithBot.realExp (WithBot.realSub _ _)) (some _) = _
    rw [realExp_eq_some_unbotD]; rfl]
  -- combine
  show (some _ : WithBot ℝ) = some _
  refine congrArg some ?_
  congr 1
  -- second summand: β · Σ p0val = Σ realExp(mscore - mnew).unbotD
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  set mk := maskedScore qStart gpos Qg Kg scale i
    (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) k) with hmkdef
  -- finiteness: mk ≠ ⊥ ⟹ supM = some, mnew = some
  have hsupne : mk ≠ ⊥ → supM ≠ ⊥ := by
    intro hmk hbot
    have hle : mk ≤ supM := by
      rw [hmkdef, hsupMdef]
      exact Finset.le_sup (f := fun jLocal : Fin 16 => maskedScore qStart gpos Qg Kg scale i
        (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jLocal))
        (Finset.mem_univ k)
    rw [hbot] at hle
    exact hmk (le_bot_iff.mp hle)
  have hsupfin : mk ≠ ⊥ → ∃ b : ℝ, supM = (b : WithBot ℝ) := by
    intro hmk
    obtain ⟨b, hb⟩ := WithBot.ne_bot_iff_exists.mp (hsupne hmk)
    exact ⟨b, hb.symm⟩
  have hmnewfin : mk ≠ ⊥ → ∃ n : ℝ, mnew = (n : WithBot ℝ) := by
    intro hmk
    have hmle : supM ≤ mnew := by
      rw [hmnewdef, bsaMPartial_succ_of_lt qStart numKVBlocks gpos Qg Kg scale c hc i]
      exact le_max_right _ _
    have hmnewne : mnew ≠ ⊥ := by
      intro hbot; rw [hbot] at hmle
      exact hsupne hmk (le_bot_iff.mp hmle)
    obtain ⟨n, hn⟩ := WithBot.ne_bot_iff_exists.mp hmnewne
    exact ⟨n, hn.symm⟩
  have hbeta := bsa_beta_p0_eq_exp mk supM mnew hsupfin hmnewfin
  -- p0val_k = realExp(sub mk supM).unbotD
  have hp0k : (p0.data (i, k, PUnit.unit)).unbotD 0
      = (WithBot.realExp (WithBot.realSub mk supM)).unbotD 0 := by
    rw [hp0 k, hmijsup]
  rw [hp0k]
  have hbu := congrArg (WithBot.unbotD (0:ℝ)) hbeta
  rw [realExp_eq_some_unbotD (WithBot.realSub mk supM),
      realExp_eq_some_unbotD (WithBot.realSub supM mnew),
      realExp_eq_some_unbotD (WithBot.realSub mk mnew)] at hbu
  simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map, WithBot.unbotD_some] at hbu
  rw [show WithBot.realSub mk mnew = Option.map₂ (fun x y : ℝ => x - y) mk mnew from rfl] at hbu
  rw [← hbu]
  norm_num [mul_comm]

/-- `(realExp x).unbotD 0 ≥ 0`: natural exp is `some 0` on ⊥ and `some (exp r) > 0`
otherwise, so the demoted value is always nonnegative. -/
theorem realExp_unbotD_nonneg (x : WithBot ℝ) :
    0 ≤ (WithBot.realExp x).unbotD 0 := by
  cases x with
  | bot => simp [WithBot.realExp]
  | coe r => simp [WithBot.realExp]; positivity

open BSAMathCausal in
/-- `bsaLPartial` is nonnegative (sum/product of exp weights and nonneg prefixes). -/
theorem bsaLPartial_nonneg {M D Bk : Nat}
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (i : Fin M) :
    0 ≤ bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k i := by
  induction k with
  | zero => simp [bsaLPartial]
  | succ k ih =>
    by_cases hk : k + 1 ≤ numKVBlocks
    · rw [bsaLPartial_succ_of_lt qStart numKVBlocks gpos Q Kg scale k (Nat.lt_of_succ_le hk) i]
      apply add_nonneg
      · exact mul_nonneg (realExp_unbotD_nonneg _) ih
      · apply Finset.sum_nonneg; intro jL _; exact realExp_unbotD_nonneg _
    · rw [bsaLPartial, dif_neg hk]; exact ih

open BSAMathCausal in
/-- If `bsaLPartial k = 0` then `bsaOPartial k = 0` (every causal weight vanishes,
so each value contribution vanishes). No first-key-visibility hypothesis needed. -/
theorem bsaOPartial_eq_zero_of_bsaLPartial_eq_zero {M D Bk : Nat}
    (qStart : Nat) (numKVBlocks : Nat) (gpos : Fin (Bk * numKVBlocks) → Nat)
    (Q : TileIndex [M, D] → ℝ)
    (Kg Vg : TileIndex [Bk * numKVBlocks, D] → ℝ) (scale : ℝ)
    (k : Nat) (idx : TileIndex [M, D])
    (hL : bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1 = 0) :
    bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale k idx = 0 := by
  induction k with
  | zero => simp [bsaOPartial]
  | succ k ih =>
    by_cases hk : k + 1 ≤ numKVBlocks
    · rw [bsaLPartial_succ_of_lt qStart numKVBlocks gpos Q Kg scale k (Nat.lt_of_succ_le hk) idx.1] at hL
      rw [bsaOPartial_succ_of_lt qStart numKVBlocks gpos Q Kg Vg scale k (Nat.lt_of_succ_le hk) idx]
      -- both summands of hL are nonneg
      have hαnn : 0 ≤ (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1)
          (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0 :=
        realExp_unbotD_nonneg _
      have hLknn : 0 ≤ bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1 :=
        bsaLPartial_nonneg qStart numKVBlocks gpos Q Kg scale k idx.1
      have hsumnn : 0 ≤ (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
          (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore qStart gpos Q Kg scale idx.1
              (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal))
            (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0) := by
        apply Finset.sum_nonneg; intro jL _; exact realExp_unbotD_nonneg _
      have hαL : (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1)
          (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0 *
          bsaLPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1 = 0 ∧
          (Finset.univ : Finset (Fin Bk)).sum (fun jLocal =>
            (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
              (maskedScore qStart gpos Q Kg scale idx.1
                (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jLocal))
              (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0) = 0 := by
        constructor
        · nlinarith [mul_nonneg hαnn hLknn]
        · nlinarith [mul_nonneg hαnn hLknn]
      obtain ⟨hαL0, hsum0⟩ := hαL
      -- each weight in the sum is 0
      have hwzero : ∀ jL : Fin Bk,
          (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
            (maskedScore qStart gpos Q Kg scale idx.1
              (StreamingAccumulator.blockIndex Bk numKVBlocks k (Nat.succ_le_iff.mpr (Nat.lt_of_succ_le hk)) jL))
            (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0 = 0 := by
        intro jL
        exact (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => realExp_unbotD_nonneg _)).mp hsum0 jL (Finset.mem_univ jL)
      -- numerator first term: α · Ok = 0
      have hαO : (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
          (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale k idx.1)
          (bsaMPartial Bk qStart numKVBlocks gpos Q Kg scale (k + 1) idx.1))).unbotD 0 *
          bsaOPartial Bk qStart numKVBlocks gpos Q Kg Vg scale k idx = 0 := by
        rcases mul_eq_zero.mp hαL0 with hα0 | hLk0
        · rw [hα0, zero_mul]
        · rw [ih hLk0, mul_zero]
      simp only [hαO, zero_add]
      apply Finset.sum_eq_zero; intro jL _
      rw [hwzero jL, zero_mul]
    · rw [bsaLPartial, dif_neg hk] at hL
      rw [bsaOPartial, dif_neg hk]; exact ih hL

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
open BSAMathCausal in
/-- **`bsa_step_acc`** (standalone acc conjunct). The loop body's
`acc = acc·acc_scale + tl.dot(p, v)` tile equals `bsaOPartial(c+1)/bsaLPartial(c+1)`
per cell, given the invariant register values and the score/mask/value bridges. -/
theorem bsa_step_acc
    (qStart numKVBlocks c : Nat) (hc : c < numKVBlocks) (hBkN : 0 < 16 * numKVBlocks)
    (gpos : Fin (16 * numKVBlocks) → Nat)
    (Qg : TileIndex [16, 16] → ℝ)
    (Kg Vg : TileIndex [16 * numKVBlocks, 16] → ℝ) (scale : ℝ)
    (SN : Nat) (gm : Fin 16 → Nat)
    (qkScaled vt : Tile .real [16, 16])
    (hMask : ∀ (i jL : Fin 16), (SN + jL.val ≤ gm i) ↔
      (gpos (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jL)
        ≤ qStart + i.val))
    (hScore : ∀ (i jL : Fin 16), qkScaled.data (i, jL, PUnit.unit) =
      (some (gScore Qg Kg scale i
        (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jL)) : WithBot ℝ))
    (hVal : ∀ (d jL : Fin 16), vt.data (jL, d, PUnit.unit) =
      (some (Vg (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jL,
        d, PUnit.unit)) : WithBot ℝ)) :
    let qkM : Tile .real [16, 16] :=
      ⟨fun ix : TileIndex [16, 16] =>
        NumericDType.real.add (qkScaled.data ix)
          (if SN + ix.2.1.val ≤ gm ix.1 then (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))⟩
    let mij : Tile .real [16] :=
      ⟨fun outIdx : TileIndex [16] =>
        (Finset.univ : Finset (Fin 16)).sup' Finset.univ_nonempty
          (fun k : Fin 16 => qkM.data (outIdx.1, k, PUnit.unit))⟩
    let mt : Tile .real [16] :=
      ⟨fun i : TileIndex [16] => bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale c i.1⟩
    let minew : Tile .real [16] :=
      Tile.select
        (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mt mij) mt mij
    let p0 : Tile .real [16, 16] :=
      Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkM
          (Tile.expandDim ⟨1, by simp⟩ mij))
    let lij : Tile .real [16] :=
      Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [16, 16].length) p0
    let alpha : Tile .real [16] :=
      Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mt minew)
    let beta : Tile .real [16] :=
      Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mij minew)
    let lt : Tile .real [16] :=
      ⟨fun i : TileIndex [16] =>
        (some (bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale c i.1) : WithBot ℝ)⟩
    let linew : Tile .real [16] :=
      Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) alpha lt)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) beta lij)
    let pscale : Tile .real [16] :=
      Tile.bop NumericDType.real.div (Broadcast.consSame Broadcast.nil) beta linew
    let pfinal : Tile .real [16, 16] :=
      Tile.bop NumericDType.real.mul
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) p0
        (Tile.expandDim ⟨1, by simp⟩ pscale)
    let accscale : Tile .real [16] :=
      Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.div (Broadcast.consSame Broadcast.nil) lt linew) alpha
    let acct : Tile .real [16, 16] :=
      ⟨fun idx : TileIndex [16, 16] =>
        (some (bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg scale c idx /
          bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale c idx.1) : WithBot ℝ)⟩
    let accMul : Tile .real [16, 16] :=
      Tile.bop NumericDType.real.mul
        (Broadcast.consSame (Broadcast.consR Broadcast.nil)) acct
        (Tile.expandDim ⟨1, by simp⟩ accscale)
    Tile.bop NumericDType.real.add
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) accMul
        (Tile.dot [] pfinal vt)
      = (⟨fun idx : TileIndex [16, 16] =>
          (some (bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg scale (c + 1) idx /
            bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) idx.1) : WithBot ℝ)⟩
          : Tile .real [16, 16]) := by
  intro qkM mij mt minew p0 lij alpha beta lt linew pscale pfinal accscale acct accMul
  -- linew cell = some (bsaLPartial (c+1)) — reuse bsa_step_li
  have hlinew : ∀ i : Fin 16, linew.data (i, PUnit.unit)
      = (some (bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i) : WithBot ℝ) := by
    intro i
    have := congrArg (fun t : Tile .real [16] => t.data (i, PUnit.unit))
      (bsa_step_li qStart numKVBlocks c hc gpos Qg Kg scale SN gm qkScaled hMask hScore)
    simpa [linew, alpha, beta, lt, lij, p0, mt, minew, mij, qkM, Tile.vec, Tile.scalar_data_index]
      using this
  -- minew cell = bsaMPartial (c+1)
  have hminew : ∀ i : Fin 16, minew.data (i, PUnit.unit)
      = bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i := by
    intro i
    have := congrArg (fun t : Tile .real [16] => t.data (i, PUnit.unit))
      (bsa_step_mi qStart numKVBlocks c hc gpos Qg Kg scale SN gm qkScaled hMask hScore)
    simpa [minew, mt, mij, qkM, Tile.vec, Tile.scalar_data_index] using this
  refine Tile.ext (fun idx => ?_)
  obtain ⟨i, d, u⟩ := idx
  -- abbreviations
  set mnew := bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i with hmnewdef
  set supM : WithBot ℝ := (Finset.univ : Finset (Fin 16)).sup
    (fun jLocal : Fin 16 => maskedScore qStart gpos Qg Kg scale i
      (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jLocal)) with hsupMdef
  set Lc := bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale c i with hLcdef
  set Lnew := bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i with hLnewdef
  set Oc := bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg scale c (i, d, u) with hOcdef
  set alphaTerm := (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
      (bsaMPartial 16 qStart numKVBlocks gpos Qg Kg scale c i) mnew)).unbotD 0 with hAlphaT
  -- mij cell = supM
  have hmijcell : mij.data (i, PUnit.unit) = supM := by
    show (Finset.univ : Finset (Fin 16)).sup' Finset.univ_nonempty
        (fun k : Fin 16 => qkM.data (i, k, PUnit.unit)) = _
    rw [Finset.sup'_eq_sup]
    refine Finset.sup_congr rfl (fun k _ => ?_)
    exact bsa_qkM_cell_eq_maskedScore qStart numKVBlocks c (Nat.succ_le_iff.mpr hc)
      gpos Qg Kg scale SN gm qkScaled i k (hMask i k) (hScore i k)
  -- alpha cell = some alphaTerm
  have halphacell : alpha.data (i, PUnit.unit) = (some alphaTerm : WithBot ℝ) := by
    simp only [alpha, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      mt, Tile.scalar_data_index, hminew i]
    rw [realExp_eq_some_unbotD]; rfl
  -- p0 cell value, beta cell, the masked weight w
  set wfun := fun k : Fin 16 => (WithBot.realExp (Option.map₂ (fun x y : ℝ => x - y)
      (maskedScore qStart gpos Qg Kg scale i
        (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) k)) mnew)).unbotD 0 with hwfun
  set vfun := fun k : Fin 16 => Vg (StreamingAccumulator.blockIndex 16 numKVBlocks c
      (Nat.succ_le_iff.mpr hc) k, d, u) with hvfun
  -- pfinal cell_k · vt cell_k = some ((wfun k / Lnew) * vfun k)
  have hbeta_p0 : ∀ k : Fin 16,
      WithBot.realMul (beta.data (i, PUnit.unit)) (p0.data (i, k, PUnit.unit))
        = (some (wfun k) : WithBot ℝ) := by
    intro k
    have hqkMk : qkM.data (i, k, PUnit.unit)
        = maskedScore qStart gpos Qg Kg scale i
            (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) k) :=
      bsa_qkM_cell_eq_maskedScore qStart numKVBlocks c (Nat.succ_le_iff.mpr hc)
        gpos Qg Kg scale SN gm qkScaled i k (hMask i k) (hScore i k)
    have hp0k : p0.data (i, k, PUnit.unit)
        = WithBot.realExp (WithBot.realSub
            (maskedScore qStart gpos Qg Kg scale i
              (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) k)) supM) := by
      simp only [p0, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Tile.expandDim_data, TileShape.dropInsertedIndex, hqkMk, hmijcell]
      rfl
    have hbetacell : beta.data (i, PUnit.unit)
        = WithBot.realExp (WithBot.realSub supM mnew) := by
      simp only [beta, Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
        hminew i, hmijcell, hmnewdef]
      rfl
    -- finiteness
    set mk := maskedScore qStart gpos Qg Kg scale i
      (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) k) with hmkdef
    have hsupne : mk ≠ ⊥ → supM ≠ ⊥ := by
      intro hmk hbot
      have hle : mk ≤ supM := by
        rw [hmkdef, hsupMdef]
        exact Finset.le_sup (f := fun jLocal : Fin 16 => maskedScore qStart gpos Qg Kg scale i
          (StreamingAccumulator.blockIndex 16 numKVBlocks c (Nat.succ_le_iff.mpr hc) jLocal))
          (Finset.mem_univ k)
      rw [hbot] at hle; exact hmk (le_bot_iff.mp hle)
    have hsupfin : mk ≠ ⊥ → ∃ b : ℝ, supM = (b : WithBot ℝ) := by
      intro hmk; obtain ⟨b, hb⟩ := WithBot.ne_bot_iff_exists.mp (hsupne hmk); exact ⟨b, hb.symm⟩
    have hmnewfin : mk ≠ ⊥ → ∃ n : ℝ, mnew = (n : WithBot ℝ) := by
      intro hmk
      have hmle : supM ≤ mnew := by
        rw [hmnewdef, bsaMPartial_succ_of_lt qStart numKVBlocks gpos Qg Kg scale c hc i]
        exact le_max_right _ _
      have hmnewne : mnew ≠ ⊥ := by
        intro hbot; rw [hbot] at hmle; exact hsupne hmk (le_bot_iff.mp hmle)
      obtain ⟨n, hn⟩ := WithBot.ne_bot_iff_exists.mp hmnewne; exact ⟨n, hn.symm⟩
    rw [hbetacell, hp0k]
    have hbeta := bsa_beta_p0_eq_exp mk supM mnew hsupfin hmnewfin
    rw [hwfun]
    rw [hbeta, realExp_eq_some_unbotD]; rfl
  -- pfinal cell_k · vt cell_k = some ((wfun k / Lnew) * vfun k)
  have hpf_vt : ∀ k : Fin 16,
      Option.map₂ (· * ·) (pfinal.data (i, k, PUnit.unit)) (vt.data (k, d, u))
        = (some ((wfun k / Lnew) * vfun k) : WithBot ℝ) := by
    intro k
    -- pfinal cell_k = realMul (p0 cell_k) (pscale cell_i)
    have hpfk : pfinal.data (i, k, PUnit.unit)
        = WithBot.realMul (p0.data (i, k, PUnit.unit)) (pscale.data (i, PUnit.unit)) := by
      simp only [pfinal, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Tile.expandDim_data, TileShape.dropInsertedIndex]
      rfl
    -- pscale cell = realDiv (beta cell) (linew cell) = some (wfun-num/Lnew)? compute via div
    have hpscale : pscale.data (i, PUnit.unit)
        = WithBot.realDiv (beta.data (i, PUnit.unit)) (some Lnew) := by
      simp only [pscale, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hlinew i, hLnewdef]
      rfl
    -- beta·p0 = some (wfun k)
    have hbp := hbeta_p0 k
    -- vt cell
    rw [hpfk, hpscale, hvfun, hVal d k]
    -- (p0 * (beta/Lnew)) * Vg = ((beta*p0)/Lnew)*Vg = (wfun/Lnew)*Vg
    rw [show WithBot.realDiv (beta.data (i, PUnit.unit)) (some Lnew)
        = WithBot.realMul (beta.data (i, PUnit.unit)) (some (Lnew⁻¹)) from ?_]
    · -- now everything realMul of somes
      rw [show WithBot.realMul (p0.data (i, k, PUnit.unit))
            (WithBot.realMul (beta.data (i, PUnit.unit)) (some Lnew⁻¹))
          = WithBot.realMul (WithBot.realMul (beta.data (i, PUnit.unit)) (p0.data (i, k, PUnit.unit)))
              (some Lnew⁻¹) from ?_]
      · rw [hbp]
        show WithBot.realMul (WithBot.realMul (some (wfun k)) (some Lnew⁻¹)) (some (vfun k)) = _
        simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map]
        refine congrArg some ?_
        field_simp; ring
      · cases hb : beta.data (i, PUnit.unit) <;> cases hp : p0.data (i, k, PUnit.unit) <;>
          simp [WithBot.realMul, Option.map₂, Option.bind, Option.map, mul_comm, mul_left_comm, mul_assoc]
    · cases hb : beta.data (i, PUnit.unit) <;>
        simp [WithBot.realDiv, WithBot.realMul, Option.map₂, Option.bind, Option.map, div_eq_mul_inv]
  -- the dot cell = some (Σ (wfun/Lnew)·vfun)
  have hdot : (Tile.dot [] pfinal vt).data (i, d, u)
      = (some (Finset.univ.sum (fun k : Fin 16 => (wfun k / Lnew) * vfun k)) : WithBot ℝ) := by
    rw [Tile.dot_nil_data]
    rw [show (@Finset.sum (Fin 16) (WithBot ℝ) _ Finset.univ
          (fun k => Option.map₂ (· * ·) (pfinal.data (i, k, PUnit.unit)) (vt.data (k, d, u))))
        = @Finset.sum (Fin 16) (WithBot ℝ) _ Finset.univ
            (fun k => (some ((wfun k / Lnew) * vfun k) : WithBot ℝ))
        from Finset.sum_congr rfl (fun k _ => hpf_vt k)]
    rw [WithBot.sum_someTerm_eq_some]
  -- accMul cell = some (Rc · (Lc/Lnew · alphaTerm))
  have haccMul : accMul.data (i, d, u)
      = (some ((Oc / Lc) * (Lc / Lnew * alphaTerm)) : WithBot ℝ) := by
    have hacctcell : acct.data (i, d, u)
        = (some (Oc / Lc) : WithBot ℝ) := by
      simp only [acct, hOcdef, hLcdef]
    have haccscale : accscale.data (i, PUnit.unit)
        = (some (Lc / Lnew * alphaTerm) : WithBot ℝ) := by
      simp only [accscale, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
        lt, Tile.scalar_data_index, hlinew i, halphacell, hLcdef, hLnewdef,
        NumericDType.mul, NumericDType.div,
        WithBot.realDiv, WithBot.realMul, Option.map₂, Option.bind, Option.map]
    simp only [accMul, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, TileShape.dropInsertedIndex, hacctcell, haccscale,
      NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  -- combine
  show Option.map₂ (· + ·) (accMul.data (i, d, u)) ((Tile.dot [] pfinal vt).data (i, d, u)) = _
  rw [haccMul, hdot]
  show (some _ : WithBot ℝ) = some _
  refine congrArg some ?_
  show Oc / Lc * (Lc / Lnew * alphaTerm) + Finset.univ.sum (fun k => wfun k / Lnew * vfun k)
      = bsaOPartial 16 qStart numKVBlocks gpos Qg Kg Vg scale (c + 1) (i, d, u) /
          bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i
  -- fa2_acc_step_real
  by_cases hLnz : Lnew = 0
  · -- Lnew = 0 ⟹ both sides are 0 (kernel divides by Lnew = 0 → 0; RHS = _/0 = 0)
    have hRHS : bsaLPartial 16 qStart numKVBlocks gpos Qg Kg scale (c + 1) i = 0 := by
      rw [← hLnewdef]; exact hLnz
    rw [hLnz, hRHS]
    simp only [div_zero, zero_mul, mul_zero, Finset.sum_const_zero, add_zero]
  · -- Lnew ≠ 0
    have hRO : (Oc / Lc) * Lc = Oc := by
      by_cases hLc : Lc = 0
      · rw [hLc, div_zero, mul_zero]
        rw [hOcdef]
        exact (bsaOPartial_eq_zero_of_bsaLPartial_eq_zero qStart numKVBlocks gpos Qg Kg Vg scale
          c (i, d, u) (by rw [← hLcdef]; exact hLc)).symm
      · rw [div_mul_cancel₀ _ hLc]
    have hLnewEq : Lnew = alphaTerm * Lc + Finset.univ.sum wfun := by
      rw [hLnewdef, bsaLPartial_succ_of_lt qStart numKVBlocks gpos Qg Kg scale c hc i]
    have := fa2_acc_step_real (Oc / Lc) Lc Oc alphaTerm Lnew wfun vfun hRO hLnewEq hLnz
    -- RHS: (α·Oc + Σ w·v)/Lnew = bsaOPartial(c+1)/bsaLPartial(c+1)
    rw [this]
    -- bsaOPartial(c+1) = α·Oc + Σ w·v
    rw [hLnewdef]
    congr 1
    rw [hOcdef, bsaOPartial_succ_of_lt qStart numKVBlocks gpos Qg Kg Vg scale c hc (i, d, u)]


end VeriTile.Bench.TritonBenchG.BlockSparseAttn

