import VeriTile.Triton

/-!
# `kcache_copy_triton` — strict per-kernel correctness

`_copy_to_kcache_seqlen_n_kernel` is a paged K-cache scatter: program
`(cur_token_idx, cur_kv_head_idx, split_x_idx)` derives the sequence id and a
signed token shift, reads `seq_lengths` and the `BLOCK_TABLES` block table to
locate the destination block / in-block slot, gathers one contiguous
`KCACHE_X` split-x block of `K`, and scatters it into `KCache` for that head /
split. It supports the legacy `[num_blocks, num_kv_heads, block_size, head_dim]`
layout (`KCACHE_X = HEAD_DIM`, one split) and the new split-x
`[num_blocks, num_kv_heads, head_dim // x, block_size, x]` layout.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (grid over `(num_tokens, num_kv_heads, head_dim/x)`, the
block-table / seq-length inputs, and how the runtime composes per-program
scatters into the paged K-cache) is the *trusted boundary*, not a proof
obligation here. Because the program ids are universally quantified, the
per-program statement covers every program of the grid.

## Proof architecture

```
copy_to_kcache_seqlen_n1_correctness            ← TOP SPECIFICATION (kcacheCopyN1IO ⊨ chained paged copy)
  ├─ copy_to_kcache_seqlen_n1_surface_flattenOk       bridge fragment membership
  ├─ copy_to_kcache_seqlen_n1_surface_traceSafe       per-execution safety walk
  └─ copy_to_kcache_seqlen_n1_surface_region_run      region-model chained-metadata triple
       ├─ copy_to_kcache_seqlen_n1_surface_exec_isSome  termination
       ├─ copy_to_kcache_seqlen_n1_surface_correct      executed-state readback per cell
       └─ copy_to_kcache_seqlen_n1_surface_frame        scatter cell frame

(separate consumers of _surface_correct, not on the top-theorem path:
  copy_to_kcache_seqlen_n1_surface_compute_correct            (ComputeCorrect over the scatter)
  copy_to_kcache_seqlen_n1_old_layout_block_compute_correct   (legacy layout)
  copy_to_kcache_seqlen_n1_new_layout_xblock_compute_correct  (split-x layout))
```

The headline is stated on the kernel's chained-metadata **IO signature**
`kcacheCopyN1IO` (`ChainMetaMasked2DKernelIO₂ₓ₂`): slot 1 pins the raw
sequence length `m₁ = seq_lengths[pid₀]`, slot 2 pins the block id
`m₂ = BLOCK_TABLES[pid₀·bts + ((m₁−1)/block_size)·btb]` at the
**`m₁`-dependent** cell (the chain), and the single K tile is copied to the
`m₂`-selected cache window. The kernel is K-only (1 in ↦ 1 out), so the
family's second data channel is instantiated *off* by duplicate-region
wiring: `in2 := in1 = K`, `out2 := out1 = KCache` with
`mask2`/`writeMask2 := False` (sound — `FlatAlloc.Disjoint` only constrains
*distinct* listed regions, and the `False` gates make channel 2's legs
vacuous).

The `n_tokens > 1` (prefill) path is covered at the surface-lowering level only
(`copy_to_kcache_seqlen_n_kernel_toAlgorithm_supported`).
The post-arithmetic split-x slice
`copy_to_kcache_split_x_block_{correct,compute_correct}` proves the store once
the cache slot has been selected.

## Modeling boundary

Arithmetic/values are over `ℝ` (not bit-accurate IEEE float); dtype casts are
erased (post-erasure all numeric dtypes unify to `ℝ`; the signed kernel keeps
explicit `int64`/`uint64` casts on the index path). This is **partial**: full
cellwise readback correctness is proved for the `n_tokens = 1` decode path; the
`n_tokens > 1` path is verified only up to surface lowering (its negative
`cur_token_shift` arithmetic is outside the Nat-only pointer readback).
Python's third grid axis `split_x_idx = tl.program_id(2)` is pinned as the
host parameter `SPLIT_X` on the decode surface (the `kv_cache_copy` sibling's
convention; the `⊨` family is two-pid): the universally quantified `SPLIT_X`
covers every program of that axis. The destination block id is gathered from
`BLOCK_TABLES` and the in-block slot from `seq_lengths`; the headline's
per-cell readback legs are guarded by the family's per-context `WriteInj`
no-aliasing antecedent (the former `hOutInj` side condition — here discharged
for free, since one program's cache window is contiguous). The launch-time
`num_warps` heuristic is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton

open VeriTile.Triton
open scoped VeriTile.Triton.ChainMetaMasked2DKernelIO₂ₓ₂

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/- Lean model of `kcache_copy_triton.py`'s
`_copy_to_kcache_seqlen_n_kernel`.

This version keeps the Python expression shape and typed int regions for
`BLOCK_TABLES` and `seq_lengths`. The explicitly signed variant below pins the
intermediate arithmetic with `tl.int64` casts for the `n_tokens > 1` path. -/
def copy_to_kcache_seqlen_n_kernel
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .int)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size n_tokens
      _HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_token_idx = tl.program_id(0)
  cur_seq_idx = cur_token_idx // $(n_tokens)
  cur_token_shift = cur_token_idx - $(n_tokens) * (cur_seq_idx + $(1))
  cur_kv_head_idx = tl.program_id(1)
  split_x_idx = tl.program_id(2)
  past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx) + cur_token_shift
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_table_ptr = BLOCK_TABLES + cur_seq_idx * $(stride_bts)
  block_id = tl.load(block_table_ptr + last_bt_block_idx * $(stride_btb))
  offset_last_block = past_kv_seq_len % $(block_size)
  offsets_dmodel = split_x_idx * $(KCACHE_X) + tl.arange(0, $(KCACHE_X))
  offsets_k = cur_token_idx * $(stride_kt) +
    cur_kv_head_idx * $(stride_kh) + offsets_dmodel * $(stride_kd)
  k = tl.load(K + offsets_k)
  offsets_kcache = block_id * $(stride_kcb) +
    cur_kv_head_idx * $(stride_kch) +
    split_x_idx * $(stride_kcsplit_x) +
    offset_last_block * $(stride_kcs) + tl.arange(0, $(KCACHE_X))
  tl.store(KCache + offsets_kcache, k)
}

/-- The complete signed `n_tokens` K-cache copy kernel lowers to the algorithm
layer.

This is the faithful surface for Python's `n_tokens > 1` branch: `BLOCK_TABLES`
and `seq_lengths` are typed integer regions, so `cur_token_shift` and
`past_kv_seq_len` stay on the signed path instead of being approximated by a
Nat-only rewrite. -/
theorem copy_to_kcache_seqlen_n_kernel_toAlgorithm_supported
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .int)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size n_tokens
      _HEAD_DIM KCACHE_X : Nat) :
    ∃ alg, (copy_to_kcache_seqlen_n_kernel K KCache BLOCK_TABLES seq_lengths
      stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size n_tokens
      _HEAD_DIM KCACHE_X).toAlgorithm? = Except.ok alg := by
  simp [copy_to_kcache_seqlen_n_kernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Explicitly signed version of Python's `n_tokens > 1` K-cache copy path.

The original Triton code relies on signed `seq_lengths` arithmetic after
`cur_token_shift` is added. This proof surface makes that signed path explicit
by casting the token-position arithmetic to `tl.int64` before forming
`past_kv_seq_len`. -/
def copy_to_kcache_seqlen_n_signed_kernel
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .int)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size n_tokens
      _HEAD_DIM KCACHE_X : Nat) :
  ComputeKernel := triton {
  cur_token_idx_nat = tl.program_id(0)
  cur_token_idx = tl.cast(cur_token_idx_nat, tl.int64)
  cur_seq_idx = cur_token_idx_nat // $(n_tokens)
  cur_seq_idx_i = tl.cast(cur_seq_idx, tl.int64)
  n_tokens_i = tl.cast($(n_tokens), tl.int64)
  one_i = tl.cast($(1), tl.int64)
  cur_token_shift = cur_token_idx - n_tokens_i * (cur_seq_idx_i + one_i)
  cur_kv_head_idx = tl.program_id(1)
  split_x_idx = tl.program_id(2)
  past_kv_seq_len = tl.load($((seq_lengths : Region TileDType.int)) + cur_seq_idx) + cur_token_shift
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  last_bt_block_idx_nat = tl.cast(last_bt_block_idx, tl.uint64)
  block_table_ptr = $((BLOCK_TABLES : Region TileDType.int)) + cur_seq_idx * $(stride_bts)
  block_id = tl.load(block_table_ptr + last_bt_block_idx_nat * $(stride_btb))
  offset_last_block = past_kv_seq_len % $(block_size)
  block_id_nat = tl.cast(block_id, tl.uint64)
  offset_last_block_nat = tl.cast(offset_last_block, tl.uint64)
  offsets_dmodel = split_x_idx * $(KCACHE_X) + tl.arange(0, $(KCACHE_X))
  offsets_k = cur_token_idx_nat * $(stride_kt) +
    cur_kv_head_idx * $(stride_kh) + offsets_dmodel * $(stride_kd)
  k = tl.load(K + offsets_k)
  offsets_kcache = block_id_nat * $(stride_kcb) +
    cur_kv_head_idx * $(stride_kch) +
    split_x_idx * $(stride_kcsplit_x) +
    offset_last_block_nat * $(stride_kcs) + tl.arange(0, $(KCACHE_X))
  tl.store(KCache + offsets_kcache, k)
}

theorem copy_to_kcache_seqlen_n_signed_kernel_toAlgorithm_supported
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .int)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs _stride_kcx stride_bts stride_btb block_size n_tokens
      _HEAD_DIM KCACHE_X : Nat) :
    ∃ alg, (copy_to_kcache_seqlen_n_signed_kernel K KCache BLOCK_TABLES
      seq_lengths stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs _stride_kcx stride_bts stride_btb block_size
      n_tokens _HEAD_DIM KCACHE_X).toAlgorithm? = Except.ok alg := by
  simp [copy_to_kcache_seqlen_n_signed_kernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription of `kcache_copy_triton.py`'s
`_copy_to_kcache_seqlen_n_kernel` for the `n_tokens = 1` decode path.

For `n_tokens = 1`, Python treats `cur_token_idx` as the sequence id and uses
`seq_lengths[cur_seq_idx] - 1` as the position being copied. This surface keeps
that block-table lookup, offset-within-block computation, split-x K load, and
K-cache store. The `n_tokens > 1` path needs negative `cur_token_shift`
arithmetic before the copy and remains outside the current Nat-only pointer
surface. Python's `split_x_idx = tl.program_id(2)` axis is pinned as the host
parameter `SPLIT_X` (the `kv_cache_copy` sibling's convention): the chained
two-pid `⊨` family below has no third program id, and the universally
quantified `SPLIT_X` covers every program of the third grid dimension.
Python's unused `stride_kcx` and `HEAD_DIM` arguments are retained as ignored
parameters; `n_tokens` is fixed to the documented decode value `1` in the
proofs. -/
def copy_to_kcache_seqlen_n1_surface
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .nat)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs _stride_kcx stride_bts stride_btb block_size
      _n_tokens _HEAD_DIM KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_token_idx = tl.program_id(0)
  cur_seq_idx = cur_token_idx
  cur_kv_head_idx = tl.program_id(1)
  past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx) - $(1)
  last_bt_block_idx = past_kv_seq_len // $(block_size)
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    last_bt_block_idx * $(stride_btb))
  offset_last_block = past_kv_seq_len % $(block_size)
  offsets_dmodel = $(SPLIT_X) * $(KCACHE_X) + tl.arange(0, $(KCACHE_X))
  k = tl.load(K + cur_token_idx * $(stride_kt) +
    cur_kv_head_idx * $(stride_kh) + offsets_dmodel * $(stride_kd))
  tl.store(KCache + block_id * $(stride_kcb) +
    cur_kv_head_idx * $(stride_kch) +
    $(SPLIT_X) * $(stride_kcsplit_x) +
    offset_last_block * $(stride_kcs) + tl.arange(0, $(KCACHE_X)), k)
}

/-- The decode-path K-cache copy surface lowers to the algorithm layer,
including sequence-length lookup, block-table lookup, split-x load, and K-cache
store. -/
theorem copy_to_kcache_seqlen_n1_surface_toAlgorithm_supported
    (K KCache : RegionName) (BLOCK_TABLES seq_lengths : Region .nat)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs _stride_kcx stride_bts stride_btb block_size
      _n_tokens _HEAD_DIM KCACHE_X : Nat) :
    ∃ alg,
      (copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES seq_lengths
        SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs _stride_kcx stride_bts stride_btb
        block_size _n_tokens _HEAD_DIM KCACHE_X).toAlgorithm? = Except.ok alg := by
  simp [copy_to_kcache_seqlen_n1_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented split-x slice of `kcache_copy_triton.py`'s
`_copy_to_kcache_seqlen_n_kernel`.

The full kernel computes sequence-local block arithmetic with division and
modulo. This slice starts after that arithmetic has selected the cache slot:
load the block id from `BLOCK_TABLES`, load one contiguous K split-x block, and
store it into `KCache`. -/
def copy_to_kcache_split_x_block
    (K KCache : RegionName) (BLOCK_TABLES : Region .nat)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb KCACHE_X : Nat) :
    ComputeKernel := triton {
  cur_token_idx = tl.program_id(0)
  cur_seq_idx = tl.program_id(0)
  cur_kv_head_idx = tl.program_id(1)
  split_x_idx = tl.program_id(2)
  offsets_dmodel = split_x_idx * $(KCACHE_X) + tl.arange(0, $(KCACHE_X))
  block_id = tl.load(BLOCK_TABLES + cur_seq_idx * $(stride_bts) +
    $(LAST_BLOCK_IDX) * $(stride_btb))
  k = tl.load(K + cur_token_idx * $(stride_kt) +
    cur_kv_head_idx * $(stride_kh) + offsets_dmodel * $(stride_kd))
  tl.store(KCache + block_id * $(stride_kcb) +
    cur_kv_head_idx * $(stride_kch) +
    split_x_idx * $(stride_kcsplit_x) +
    $(OFFSET_LAST_BLOCK) * $(stride_kcs) + tl.arange(0, $(KCACHE_X)), k)
}

def dimIndex (i : Fin KCACHE_X) : Nat :=
  i.val

def blockId (s : BlockState) (BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX stride_bts stride_btb : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb)

def kSourceOffset
    (s : BlockState) (stride_kt stride_kh stride_kd KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  s.pids 0 * stride_kt + s.pids 1 * stride_kh +
    (s.pids 2 * KCACHE_X + dimIndex i) * stride_kd

def kCacheOffset
    (s : BlockState) (BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb : Nat)
    (i : Fin KCACHE_X) : Nat :=
  blockId s BLOCK_TABLES LAST_BLOCK_IDX stride_bts stride_btb * stride_kcb +
    s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
    OFFSET_LAST_BLOCK * stride_kcs + dimIndex i

def n1PastKvSeqLen (s : BlockState) (seq_lengths : RegionName) : Nat :=
  s.readMemValue .nat seq_lengths (s.pids 0) - 1

def n1LastBlockIdx (s : BlockState) (seq_lengths : RegionName)
    (block_size : Nat) : Nat :=
  n1PastKvSeqLen s seq_lengths / block_size

def n1OffsetLastBlock (s : BlockState) (seq_lengths : RegionName)
    (block_size : Nat) : Nat :=
  n1PastKvSeqLen s seq_lengths % block_size

def n1BlockId (s : BlockState) (BLOCK_TABLES seq_lengths : RegionName)
    (stride_bts stride_btb block_size : Nat) : Nat :=
  s.readMemValue .nat BLOCK_TABLES
    (s.pids 0 * stride_bts + n1LastBlockIdx s seq_lengths block_size * stride_btb)

/-- K read offset for the pinned-`SPLIT_X` decode surface: lane `i` of the
split-x block of `K` for program `(cur_token_idx, cur_kv_head_idx)`. -/
def n1KSourceOffset
    (s : BlockState) (SPLIT_X stride_kt stride_kh stride_kd KCACHE_X : Nat)
    (i : Fin KCACHE_X) : Nat :=
  s.pids 0 * stride_kt + s.pids 1 * stride_kh +
    (SPLIT_X * KCACHE_X + dimIndex i) * stride_kd

def n1KCacheOffset
    (s : BlockState) (BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kcb stride_kch stride_kcsplit_x stride_kcs
      stride_bts stride_btb block_size : Nat)
    (i : Fin KCACHE_X) : Nat :=
  n1BlockId s BLOCK_TABLES seq_lengths stride_bts stride_btb block_size * stride_kcb +
    s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
    n1OffsetLastBlock s seq_lengths block_size * stride_kcs + dimIndex i

/-- Algorithm-layer correctness for the `n_tokens = 1` K-cache copy surface.

This theorem covers the Python decode path's dynamic `seq_lengths` read, block
index division, offset modulo, block-table lookup, and split-x K-cache store. -/
theorem copy_to_kcache_seqlen_n1_surface_correct
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        n1KCacheOffset s BLOCK_TABLES seq_lengths SPLIT_X stride_kcb stride_kch
          stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i))
    (hExec : exec (copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0
        KCACHE_X)
        s = some s') :
    ∀ i : Fin KCACHE_X,
      s'.readMem KCache
          (n1KCacheOffset s BLOCK_TABLES seq_lengths SPLIT_X stride_kcb
            stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
            block_size i) =
        s.readMem K (n1KSourceOffset s SPLIT_X stride_kt stride_kh stride_kd
          KCACHE_X i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [KCACHE_X] =>
        s.readMemValue .nat BLOCK_TABLES
            (s.pids 0 * stride_bts +
              ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) / block_size) *
                stride_btb) * stride_kcb +
          s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
          ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) % block_size) *
            stride_kcs + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [n1KCacheOffset, n1BlockId, n1LastBlockIdx, n1OffsetLastBlock,
        n1PastKvSeqLen, blockId, dimIndex, BlockState.readMemValue] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hX : 0 < KCACHE_X
  · simp [exec, copy_to_kcache_seqlen_n1_surface, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          IntegralDType.floorDiv, IntegralDType.mod, BlockState.readMemValue, hX]
        at hExec
    rw [← hExec]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := KCache)
        (shape := [KCACHE_X])
        (s := ((((((((s.setReg "cur_token_idx" TileDType.nat []
              (Tile.scalar (s.pids 0)))
          |>.setReg "cur_seq_idx" TileDType.nat [] (Tile.scalar (s.pids 0)))
          |>.setReg "cur_kv_head_idx" TileDType.nat [] (Tile.scalar (s.pids 1)))
          |>.setReg "past_kv_seq_len" TileDType.nat []
            (Tile.scalar (s.readMemValue .nat seq_lengths (s.pids 0) - 1)))
          |>.setReg "last_bt_block_idx" TileDType.nat []
            (Tile.scalar ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) /
              block_size)))
          |>.setReg "block_id" TileDType.nat []
            (Tile.scalar (s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts +
                ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) / block_size) *
                  stride_btb))))
          |>.setReg "offset_last_block" TileDType.nat []
            (Tile.scalar ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) %
              block_size)))
          |>.setReg "offsets_dmodel" TileDType.nat [KCACHE_X]
            { data := fun i => SPLIT_X * KCACHE_X + i.1.val })
          |>.setReg "k" TileDType.real [KCACHE_X]
            { data := fun i =>
              some (s.readMem K
                (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                  (SPLIT_X * KCACHE_X + i.1.val) * stride_kd)) })
        (offsetFn := fun idx : TileIndex [KCACHE_X] =>
          s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts +
                ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) / block_size) *
                  stride_btb) * stride_kcb +
            s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
            ((s.readMemValue .nat seq_lengths (s.pids 0) - 1) % block_size) *
              stride_kcs + idx.1.val)
        (valueFn := fun idx : TileIndex [KCACHE_X] =>
          WithBot.unbotD 0
            (some (s.readMem K
              (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                (SPLIT_X * KCACHE_X + idx.1.val) * stride_kd))))
        (P := fun _idx : TileIndex [KCACHE_X] => True)
        hRawInj (i, PUnit.unit))
    simp [BlockState.readMemValue] at hScatter
    simpa [n1KCacheOffset, n1BlockId, n1LastBlockIdx, n1OffsetLastBlock,
      n1PastKvSeqLen, n1KSourceOffset, dimIndex, BlockState.readMemValue] using hScatter
  · exact False.elim (hX (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the `n_tokens = 1` K-cache copy surface. -/
theorem copy_to_kcache_seqlen_n1_surface_compute_correct
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        n1KCacheOffset s BLOCK_TABLES seq_lengths SPLIT_X stride_kcb stride_kch
          stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0
        KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin KCACHE_X => True)
        (fun i => (KCache,
          n1KCacheOffset s BLOCK_TABLES seq_lengths SPLIT_X stride_kcb
            stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
            block_size i)))
      (expected := fun i =>
        s.readMem K (n1KSourceOffset s SPLIT_X stride_kt stride_kh stride_kd
          KCACHE_X i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [copy_to_kcache_seqlen_n1_surface]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact copy_to_kcache_seqlen_n1_surface_correct K KCache BLOCK_TABLES
    seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
    stride_kcsplit_x stride_kcs stride_bts stride_btb block_size KCACHE_X
    s s' hOutInj hExec i

/-- Named `n_tokens = 1` K-cache writeback for the legacy Python layout
`[num_blocks, num_kv_heads, block_size, head_dim]`.

The Python wrapper sets `KCACHE_X = HEAD_DIM` and `stride_kcsplit_x = 0` for
this branch, so the single split-x program (`SPLIT_X = 0`) covers the full K
head dimension. -/
theorem copy_to_kcache_seqlen_n1_old_layout_block_compute_correct
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcs
      stride_bts stride_btb block_size HEAD_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HEAD_DIM =>
        n1KCacheOffset s BLOCK_TABLES seq_lengths 0 stride_kcb stride_kch 0
          stride_kcs stride_bts stride_btb block_size i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths 0 stride_kt stride_kh stride_kd stride_kcb stride_kch
        0 stride_kcs 0 stride_bts stride_btb block_size 1 HEAD_DIM HEAD_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HEAD_DIM => True)
        (fun i => (KCache,
          n1KCacheOffset s BLOCK_TABLES seq_lengths 0 stride_kcb stride_kch 0
            stride_kcs stride_bts stride_btb block_size i)))
      (expected := fun i =>
        s.readMem K (n1KSourceOffset s 0 stride_kt stride_kh stride_kd
          HEAD_DIM i)) := by
  exact copy_to_kcache_seqlen_n1_surface_compute_correct K KCache BLOCK_TABLES
    seq_lengths 0 stride_kt stride_kh stride_kd stride_kcb stride_kch 0
    stride_kcs stride_bts stride_btb block_size HEAD_DIM s hOutInj

/-- Named `n_tokens = 1` K-cache writeback for the new Python split-x layout
`[num_blocks, num_kv_heads, head_dim // x, block_size, x]`.

In the checked Python test this is instantiated with `HEAD_DIM = 64` and
`KCACHE_X = 8`, once for each `SPLIT_X < 8`. -/
theorem copy_to_kcache_seqlen_n1_new_layout_xblock_compute_correct
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size HEAD_DIM
      KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        n1KCacheOffset s BLOCK_TABLES seq_lengths SPLIT_X stride_kcb stride_kch
          stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1
        HEAD_DIM KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin KCACHE_X => True)
        (fun i => (KCache,
          n1KCacheOffset s BLOCK_TABLES seq_lengths SPLIT_X stride_kcb
            stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
            block_size i)))
      (expected := fun i =>
        s.readMem K (n1KSourceOffset s SPLIT_X stride_kt stride_kh stride_kd
          KCACHE_X i)) := by
  exact copy_to_kcache_seqlen_n1_surface_compute_correct K KCache BLOCK_TABLES
    seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
    stride_kcsplit_x stride_kcs stride_bts stride_btb block_size KCACHE_X
    s hOutInj

/-- Algorithm-layer correctness for the K-cache split-x copy slice. -/
theorem copy_to_kcache_split_x_block_correct
    (K KCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb KCACHE_X : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
          stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
          stride_btb i))
    (hExec : exec (copy_to_kcache_split_x_block K KCache BLOCK_TABLES
        LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_kt stride_kh stride_kd
        stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
        stride_btb KCACHE_X) s = some s') :
    ∀ i : Fin KCACHE_X,
      s'.readMem KCache
          (kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
            stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
            stride_btb i) =
        s.readMem K (kSourceOffset s stride_kt stride_kh stride_kd KCACHE_X i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [KCACHE_X] =>
        s.readMemValue .nat BLOCK_TABLES
            (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb) * stride_kcb +
          s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
          OFFSET_LAST_BLOCK * stride_kcs + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [kCacheOffset, blockId, dimIndex, BlockState.readMemValue] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hX : 0 < KCACHE_X
  · simp [exec, copy_to_kcache_split_x_block, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, BlockState.readMemValue, hX]
        at hExec
    rw [← hExec]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := KCache)
        (shape := [KCACHE_X])
        (s := (s.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_seq_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_kv_head_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "split_x_idx" TileDType.nat [] (Tile.scalar (s.pids 2))
          |>.setReg "offsets_dmodel" TileDType.nat [KCACHE_X]
            (Tile.vec fun i => s.pids 2 * KCACHE_X + i.val)
          |>.setReg "block_id" TileDType.nat []
            (Tile.scalar (s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb)))
          |>.setReg "k" TileDType.real [KCACHE_X]
            { data := fun i =>
              some (s.readMem K
                (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                  (s.pids 2 * KCACHE_X + i.1.val) * stride_kd)) }))
        (offsetFn := fun idx : TileIndex [KCACHE_X] =>
          s.readMemValue .nat BLOCK_TABLES
              (s.pids 0 * stride_bts + LAST_BLOCK_IDX * stride_btb) * stride_kcb +
            s.pids 1 * stride_kch + s.pids 2 * stride_kcsplit_x +
            OFFSET_LAST_BLOCK * stride_kcs + idx.1.val)
        (valueFn := fun idx : TileIndex [KCACHE_X] =>
          WithBot.unbotD 0
            (some (s.readMem K
              (s.pids 0 * stride_kt + s.pids 1 * stride_kh +
                (s.pids 2 * KCACHE_X + idx.1.val) * stride_kd))))
        (P := fun _idx : TileIndex [KCACHE_X] => True)
        hRawInj (i, PUnit.unit))
    simp [BlockState.readMemValue] at hScatter
    simpa [kSourceOffset, kCacheOffset, blockId, dimIndex,
      BlockState.readMemValue] using hScatter
  · exact False.elim (hX (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the K-cache split-x copy slice. -/
theorem copy_to_kcache_split_x_block_compute_correct
    (K KCache BLOCK_TABLES : RegionName)
    (LAST_BLOCK_IDX OFFSET_LAST_BLOCK
      stride_kt stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x
      stride_kcs stride_bts stride_btb KCACHE_X : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
          stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
          stride_btb i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := copy_to_kcache_split_x_block K KCache BLOCK_TABLES
        LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_kt stride_kh stride_kd
        stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
        stride_btb KCACHE_X)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin KCACHE_X => True)
        (fun i => (KCache,
          kCacheOffset s BLOCK_TABLES LAST_BLOCK_IDX OFFSET_LAST_BLOCK
            stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
            stride_btb i)))
      (expected := fun i =>
        s.readMem K (kSourceOffset s stride_kt stride_kh stride_kd KCACHE_X i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [copy_to_kcache_split_x_block]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact copy_to_kcache_split_x_block_correct K KCache BLOCK_TABLES
    LAST_BLOCK_IDX OFFSET_LAST_BLOCK stride_kt stride_kh stride_kd stride_kcb
    stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb KCACHE_X
    s s' hOutInj hExec i

/-! ## The `⊨` specification (decode path)

The headline states the pinned-`SPLIT_X` decode surface on the
chained-metadata IO skin `ChainMetaMasked2DKernelIO₂ₓ₂`: the raw loaded
sequence length `m₁` and the raw block id `m₂` are named ghost binders
pinned to the two slot cells — `m₂`'s cell address eats `m₁` (the chain) —
and the K tile is copied to the `m₂`-selected cache window. The machinery
below supplies the three intro obligations: the bridge-fragment membership,
the per-execution safety walk, and the region-model run (termination +
readback + cell frame). -/

/-- The decode-path surface sits inside the flat-memory bridge's covered
fragment (pointer arithmetic, two scalar gathers, one tile load, one tile
store). -/
theorem copy_to_kcache_seqlen_n1_surface_flattenOk
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat) :
    ((copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES seq_lengths
      SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0
      KCACHE_X).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [copy_to_kcache_seqlen_n1_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Termination: the straight-line decode surface executes to completion from
any state (the loads and the store are unmasked and total). -/
private theorem copy_to_kcache_seqlen_n1_surface_exec_isSome
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat)
    (s : BlockState) :
    ∃ s1, exec ((copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0
        KCACHE_X).toAlgKernel) s = some s1 := by
  simp [exec, copy_to_kcache_seqlen_n1_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod, BlockState.readMemValue]

/-- An unmasked scatter-store `foldl` leaves every memory cell it does not
hit unchanged (cell-level frame for the unconditional output store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ)
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k))
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons,
        ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem]
      exact if_neg (fun hc =>
        hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell other than the program's `KCache` scatter
window is preserved by the run. -/
private theorem copy_to_kcache_seqlen_n1_surface_frame
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat)
    (s s1 : BlockState)
    (hExec : exec ((copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0
        KCACHE_X).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin KCACHE_X,
      ¬(KCache = r ∧ n1KCacheOffset s BLOCK_TABLES seq_lengths SPLIT_X
          stride_kcb stride_kch stride_kcsplit_x stride_kcs stride_bts
          stride_btb block_size i = o)) :
    s1.mem r o = s.mem r o := by
  simp only [n1KCacheOffset, n1BlockId, n1LastBlockIdx, n1OffsetLastBlock,
    n1PastKvSeqLen, dimIndex, BlockState.readMemValue] at hmiss
  simp [exec, copy_to_kcache_seqlen_n1_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod, BlockState.readMemValue] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ r o _ _ ?_) rfl
  intro k _ hc
  exact hmiss k.1 (by simpa [BlockState.readMemValue] using hc)

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: the four memory accesses — the `seq_lengths`
scalar gather at `pid₀`, the chained `BLOCK_TABLES` scalar gather at the
`m₁`-dependent cell, the unmasked split-x `K` tile load, and the unmasked
`KCache` tile store at the `m₂`-selected window — reduce to the four bounds
hypotheses, stated on the pinned raw slot values `m₁`/`m₂`. -/
theorem copy_to_kcache_seqlen_n1_surface_traceSafe
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat)
    (bounds : RegionBounds) (s : BlockState) (m₁ m₂ : Nat)
    (hm1 : s.readMemValue .nat seq_lengths (s.pids 0) = m₁)
    (hm2 : s.readMemValue .nat BLOCK_TABLES
      (s.pids 0 * stride_bts + ((m₁ - 1) / block_size) * stride_btb) = m₂)
    (hb1 : s.pids 0 < bounds seq_lengths)
    (hb2 : s.pids 0 * stride_bts + ((m₁ - 1) / block_size) * stride_btb
      < bounds BLOCK_TABLES)
    (hbr : ∀ j : Fin KCACHE_X,
      s.pids 0 * stride_kt + s.pids 1 * stride_kh +
        (SPLIT_X * KCACHE_X + j.val) * stride_kd < bounds K)
    (hbw : ∀ j : Fin KCACHE_X,
      m₂ * stride_kcb + s.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
        ((m₁ - 1) % block_size) * stride_kcs + j.val < bounds KCache) :
    Kernel.TraceSafe bounds
      ((copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES seq_lengths
        SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0
        KCACHE_X).toAlgKernel) s := by
  subst hm1
  subst hm2
  unfold Kernel.TraceSafe
  simp [copy_to_kcache_seqlen_n1_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, MaskOpt.Active,
    BlockState.setReg, Option.bind, Option.map,
    Tile.bop, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    IntegralDType.floorDiv, IntegralDType.mod, BlockState.readMemValue]
  exact ⟨hb1, hb2, fun a => hbr a, fun a => hbw a⟩

set_option maxHeartbeats 1600000 in
/-- **The region-model chained triple** — termination, per-lane readback of
the copied K tile at the `m₂`-selected cache window, and the cell frame off
that window, from any launch state pinning the raw slot values `m₁`
(`seq_lengths[pid₀]`) and `m₂` (the chained block-table gather) and the K
tile `xs`. This is the `hrun` obligation of the `⊨` headline; the value half
reuses `copy_to_kcache_seqlen_n1_surface_correct` (the per-context
no-aliasing `WriteInj` is free — the window is contiguous in its lane). -/
theorem copy_to_kcache_seqlen_n1_surface_region_run
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat)
    (s₀ : BlockState) (m₁ m₂ : Nat) (xs : Fin KCACHE_X → ℝ)
    (hm1 : s₀.readMemValue .nat seq_lengths (s₀.pids 0) = m₁)
    (hm2 : s₀.readMemValue .nat BLOCK_TABLES
      (s₀.pids 0 * stride_bts + ((m₁ - 1) / block_size) * stride_btb) = m₂)
    (hx : ∀ j : Fin KCACHE_X,
      s₀.readMem K (s₀.pids 0 * stride_kt + s₀.pids 1 * stride_kh +
        (SPLIT_X * KCACHE_X + j.val) * stride_kd) = xs j) :
    ∃ s1, exec ((copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES
        seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0
        KCACHE_X).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin KCACHE_X,
          s1.readMem KCache
            (m₂ * stride_kcb + s₀.pids 1 * stride_kch +
              SPLIT_X * stride_kcsplit_x +
              ((m₁ - 1) % block_size) * stride_kcs + j.val) = xs j)
      ∧ (∀ r o,
          (r ≠ KCache ∨ ∀ j : Fin KCACHE_X,
            o ≠ m₂ * stride_kcb + s₀.pids 1 * stride_kch +
              SPLIT_X * stride_kcsplit_x +
              ((m₁ - 1) % block_size) * stride_kcs + j.val) →
          s1.mem r o = s₀.mem r o) := by
  subst hm1
  subst hm2
  have hkc : ∀ i : Fin KCACHE_X,
      n1KCacheOffset s₀ BLOCK_TABLES seq_lengths SPLIT_X stride_kcb stride_kch
        stride_kcsplit_x stride_kcs stride_bts stride_btb block_size i
      = s₀.readMemValue .nat BLOCK_TABLES
          (s₀.pids 0 * stride_bts +
            ((s₀.readMemValue .nat seq_lengths (s₀.pids 0) - 1) / block_size) *
              stride_btb) * stride_kcb +
        s₀.pids 1 * stride_kch + SPLIT_X * stride_kcsplit_x +
        ((s₀.readMemValue .nat seq_lengths (s₀.pids 0) - 1) % block_size) *
          stride_kcs + i.val := by
    intro i
    simp [n1KCacheOffset, n1BlockId, n1LastBlockIdx, n1OffsetLastBlock,
      n1PastKvSeqLen, dimIndex]
  have hOutInj : Function.Injective
      (fun i : Fin KCACHE_X =>
        n1KCacheOffset s₀ BLOCK_TABLES seq_lengths SPLIT_X stride_kcb
          stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb
          block_size i) := by
    intro a b h
    simp only [hkc] at h
    exact Fin.ext (Nat.add_left_cancel h)
  obtain ⟨s1, hs1⟩ := copy_to_kcache_seqlen_n1_surface_exec_isSome K KCache
    BLOCK_TABLES seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb
    stride_kch stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
    KCACHE_X s₀
  have hval := copy_to_kcache_seqlen_n1_surface_correct K KCache BLOCK_TABLES
    seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
    stride_kcsplit_x stride_kcs stride_bts stride_btb block_size KCACHE_X
    s₀ s1 hOutInj hs1
  refine ⟨s1, hs1, fun j => ?_, fun r o hcond => ?_⟩
  · have h := hval j
    rw [hkc j] at h
    rw [h, n1KSourceOffset, dimIndex, hx j]
  · refine copy_to_kcache_seqlen_n1_surface_frame K KCache BLOCK_TABLES
      seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size KCACHE_X
      s₀ s1 hs1 r o (fun i hc => ?_)
    rcases hcond with hne | hno
    · exact hne hc.1.symm
    · exact hno i ((hkc i ▸ hc.2).symm)

/-- `copy_to_kcache_seqlen_n1_surface`'s chained-metadata **IO signature** —
the whole kernel-specific audit surface of the `⊨` headline:

* `mbuf1` — the `seq_lengths` slot: program `(cur_token_idx, cur_kv_head_idx)`
  loads the **raw** sequence length `m₁ = seq_lengths[pid₀]` (`mwin1`; the
  decode path's `− 1` lives in the windows, Nat-truncated exactly as in the
  kernel);
* `mbuf2` — the `BLOCK_TABLES` slot, **chained**: its cell
  `pid₀·bts + ((m₁−1)/block_size)·btb` (`mwin2`) eats the first slot's loaded
  value, pinning the raw block id `m₂`;
* `in1 → out1` — the K tile at
  `pid₀·kt + pid₁·kh + (SPLIT_X·KCACHE_X + j)·kd` (`read1`), copied to the
  `m₂`-selected cache window
  `m₂·kcb + pid₁·kch + SPLIT_X·kcsplit_x + ((m₁−1)%block_size)·kcs + j`
  (`write1`), both unmasked (`mask1`/`writeMask1 := True` — the kernel's load
  and store carry no mask);
* `in2 → out2` — the family's second data channel, instantiated **off** for
  this K-only sibling: duplicate-region wiring `in2 := K`, `out2 := KCache`
  with `mask2`/`writeMask2 := False`, so its legs are vacuous.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and chaining match them. -/
def kcacheCopyN1IO (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat) : ChainMetaMasked2DKernelIO₂ₓ₂ where
  kernel := copy_to_kcache_seqlen_n1_surface K KCache BLOCK_TABLES seq_lengths
    SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
    stride_kcsplit_x stride_kcs 0 stride_bts stride_btb block_size 1 0 KCACHE_X
  mbuf1 := seq_lengths
  mbuf2 := BLOCK_TABLES
  in1 := K
  in2 := K
  out1 := KCache
  out2 := KCache
  B := KCACHE_X
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ m₁ =>
    pid₀ * stride_bts + ((m₁ - 1) / block_size) * stride_btb
  read1 := fun pid₀ pid₁ _ _ j =>
    pid₀ * stride_kt + pid₁ * stride_kh +
      (SPLIT_X * KCACHE_X + j.val) * stride_kd
  read2 := fun pid₀ pid₁ _ _ j =>
    pid₀ * stride_kt + pid₁ * stride_kh +
      (SPLIT_X * KCACHE_X + j.val) * stride_kd
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => False
  write1 := fun _ pid₁ m₁ m₂ j =>
    m₂ * stride_kcb + pid₁ * stride_kch + SPLIT_X * stride_kcsplit_x +
      ((m₁ - 1) % block_size) * stride_kcs + j.val
  write2 := fun _ pid₁ m₁ m₂ j =>
    m₂ * stride_kcb + pid₁ * stride_kch + SPLIT_X * stride_kcsplit_x +
      ((m₁ - 1) % block_size) * stride_kcs + j.val
  writeMask1 := fun _ _ _ _ _ => True
  writeMask2 := fun _ _ _ _ _ => False

/-- **The headline**: the `n_tokens = 1` decode path of
`_copy_to_kcache_seqlen_n_kernel` implements the pure paged-cache copy on its
chained-metadata IO signature — for every disjoint flat placement of the
buffers, every program `(cur_token_idx, cur_kv_head_idx)` and split partition
`SPLIT_X` whose declared cells/lanes are in bounds, and every launch state
pinning the **raw** sequence length `m₁` at `seq_lengths[pid₀]`, the **raw**
block id `m₂` at the chained block-table cell
`BLOCK_TABLES[pid₀·bts + ((m₁−1)/block_size)·btb]`, and the K tile `xs`, the
translated pointer kernel terminates and writes

* `KCache[m₂·kcb + pid₁·kch + SPLIT_X·kcsplit_x + ((m₁−1)%block_size)·kcs + j]
  = xs j` — lane `j` of the K tile, verbatim, at the `m₂`-selected cache
  window (the value legs' per-context `WriteInj` no-aliasing antecedents are
  supplied by the skin);

the second data channel is off (`False` gates: its value leg is vacuous), and
every other memory cell is unchanged. Proof:
`ChainMetaMasked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
chained triple with the flat-memory bridge side conditions. -/
specification copy_to_kcache_seqlen_n1_correctness
    (K KCache BLOCK_TABLES seq_lengths : RegionName)
    (SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size
      KCACHE_X : Nat) :
    kcacheCopyN1IO K KCache BLOCK_TABLES seq_lengths SPLIT_X stride_kt
        stride_kh stride_kd stride_kcb stride_kch stride_kcsplit_x stride_kcs
        stride_bts stride_btb block_size KCACHE_X ⊨
      fun _ _ _ _ xs _ => (xs, xs) := by
  refine ChainMetaMasked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact copy_to_kcache_seqlen_n1_surface_flattenOk K KCache BLOCK_TABLES
      seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size KCACHE_X
  · intro bounds s m₁ m₂ hm1 hm2 hb1 hb2 hbr1 _hbr2 hbw1 _hbw2
    exact copy_to_kcache_seqlen_n1_surface_traceSafe K KCache BLOCK_TABLES
      seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
      stride_kcsplit_x stride_kcs stride_bts stride_btb block_size KCACHE_X
      bounds s m₁ m₂ hm1 hm2 hb1 hb2 (fun j => hbr1 j trivial)
      (fun j => hbw1 j trivial)
  · intro s₀ m₁ m₂ xs ys hm1 hm2 hx _hy
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      copy_to_kcache_seqlen_n1_surface_region_run K KCache BLOCK_TABLES
        seq_lengths SPLIT_X stride_kt stride_kh stride_kd stride_kcb stride_kch
        stride_kcsplit_x stride_kcs stride_bts stride_btb block_size KCACHE_X
        s₀ m₁ m₂ xs hm1 hm2 (fun j => hx j trivial)
    refine ⟨s1, hexec, fun _ j _ => hval j, fun _ j hj => hj.elim,
      fun r o hc1 _hc2 => ?_⟩
    refine hframe r o ?_
    rcases hc1 with hne | hno
    · exact Or.inl hne
    · exact Or.inr (fun j => hno j trivial)

end VeriTile.Bench.TritonBenchG.KcacheCopyTriton
