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

end BSARecipes

end VeriTile.Bench.TritonBenchG.BlockSparseAttn
