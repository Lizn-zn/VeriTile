import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `mixed_sparse_attention` — strict per-kernel correctness

`mixed_sparse_attention.py`'s `_triton_mixed_sparse_attn_fwd_kernel` is a
mixed block-sparse + column-sparse FlashAttention forward: program
`(start_m, off_hz)` loads its query tile (early-exits when `start_m·BLOCK_M ≥
seqlen`), runs the online-softmax recurrence (`m_i`, `l_i`, accumulator `acc`)
over the per-row selected dense key blocks (`block_count`/`block_offset`) and
individual sparse columns (`column_count`/`column_index`) with `qk_scale =
sm_scale · log2(e)`, then stores `acc` to `Out`, masked by `offs_m < seqlen`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`grid = (cdiv(N_CTX, BLOCK_M), Z·H, 1)`, the sparsity
schedule supplied via `block_*`/`column_*` index tensors, and how the runtime
composes per-program writes into `Out`) is the *trusted boundary*, not a proof
obligation here. Because the program ids `start_m`/`off_hz` are universally
quantified (via `s`), the per-program statements cover every program of the
grid.

## Proof architecture

```
mixed_sparse_attention_python_case{1,2,3,4}_output_summary        ← TOP THEOREMS (one per test case)
  ├─ mixed_sparse_attention_python_case{i}_surface_toAlgorithm_supported   surface lowers to algorithm layer
  └─ mixed_sparse_attention_python_case{i}_surface_output_compute_correct
       └─ mixed_sparse_attention_output_store_python_block{64,32}_compute_correct
            └─ mixed_sparse_attention_output_store_slice_compute_correct
                 └─ mixed_sparse_attention_output_store_slice_correct       algorithm-layer readback per lane
```
(Offset injectivity discharged by `mixed_sparse_attention_python_block{64,32}_offset_injective`;
the matching `_case{i}_store_summary` theorems package the store-only facts.)

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; `exp2`, `tl.dot`, and the
`sm_scale · log2(e)` scaling are not modeled at the bit level);
`@triton.autotune` is not modeled. The verified result is **final-store
scoped**: the proof establishes that the masked `Out` store copies the
accumulator slice `Acc` at the correct, injective output offsets and preserves
inactive lanes (the `offs_m < seqlen` and `start_m·BLOCK_M ≥ seqlen`
early-exit masking) — the written value is
`mixedSparseAttentionCase{i}SurfaceOutValue`, an opaque carrier for the
online-softmax + mixed-sparsity recurrence (which dense blocks and which sparse
columns are visited), which is **not** re-derived as a closed-form attention
formula here. Side conditions: layout `(Z,H,N_CTX) = (2,4,128)`, `BLOCK_DMODEL
= 64`, `fp16`; cases differ by `(BLOCK_M, BLOCK_N)` (64 or 32), `sm_scale`, and
the `seqlens` tensor, matching the four Python test cases.
-/

namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful DSL port of `mixed_sparse_attention.py`'s
`_triton_mixed_sparse_attn_fwd_kernel`. -/
def mixed_sparse_attention_fwd_kernel_surface
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  seqlen = tl.load(seqlens + off_hz // $(H))
  if start_m * $(BLOCK_M) >= seqlen {
    return
  }

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))

  qo_offset = (off_hz // $(H)) * $(stride_qz) + (off_hz % $(H)) * $(stride_qh)
  kv_offset = (off_hz // $(H)) * $(stride_kz) + (off_hz % $(H)) * $(stride_kh)

  q_ptrs = Q + qo_offset + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  k_ptrs = K + kv_offset + offs_d[:, None] * $(stride_kk)
  v_ptrs = V + kv_offset + offs_d[None, :] * $(stride_vk)
  o_ptrs = Out + qo_offset + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok)

  num_blks = tl.load(block_count + off_hz * $(NUM_ROWS) + start_m)
  blks_ptr = block_offset + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_S)
  num_cols = tl.load(column_count + off_hz * $(NUM_ROWS) + start_m)
  cols_ptr = column_index + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_V)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(q_ptrs)
  q = (q * qk_scale).to(DTYPE)

  m_mask = offs_m[:, None] < seqlen

  max_num_blks = $(8)
  for block_index in range(max_num_blks) {
    cond = block_index < num_blks
    start_n = tl.load(blks_ptr + block_index, mask=cond)
    cols = start_n + offs_n
    n_mask = (cols < seqlen) & cond
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  max_num_cols = $(16)
  for start_n in range($(0), max_num_cols, $(BLOCK_N)) {
    cond = start_n < num_cols
    n_mask = (start_n + offs_n < num_cols) & cond
    cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk = tl.where(m_mask & n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  acc /= l_i[:, None]
  tl.store(o_ptrs, (acc).to(DTYPE), mask=m_mask)
}

/-- The full mixed-sparse attention forward surface lowers to the algorithm
layer, including block and column sparse phases. -/
theorem mixed_sparse_attention_fwd_kernel_surface_toAlgorithm_supported
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (dtype : FloatDType) :
    ∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V seqlens sm_scale
      block_count block_offset column_count column_index Out stride_qz stride_qh
      stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz
      stride_vh stride_vn stride_vk stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V BLOCK_M BLOCK_N BLOCK_DMODEL
      dtype).toAlgorithm? = Except.ok alg := by
  simp [mixed_sparse_attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of
`mixed_sparse_attention.py`'s `_triton_mixed_sparse_attn_fwd_kernel`.

The full kernel combines block-sparse and column-sparse attention updates. This
slice starts from a precomputed normalized `Acc` tile and proves the final
`seqlens`-masked writeback into `Out`. The kernel-level early return for
`start_m * BLOCK_M >= seqlen` is represented at this surface by the same
all-false row mask; the sparse block/column softmax loops remain separate
modeling work, including their `tl.float32` accumulators. -/
def mixed_sparse_attention_output_store_slice
    (Acc : RegionName) (Seqlens : Region .nat) (Out : RegionName)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  seqlen = tl.load(Seqlens + off_z)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < seqlen) & (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + off_z * $(stride_qz) + off_h * $(stride_qh) +
      offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok),
      (acc).to(Out.dtype.element_ty), mask=mask)
}

def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H

def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H

def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens

instance activeDecidable (s : BlockState) (H : Nat) (Seqlens : RegionName)
    (BLOCK_M BLOCK_DMODEL : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s H Seqlens BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d

def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok

noncomputable def accStoreValue
    (s : BlockState) (Acc Seqlens : RegionName)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H Seqlens BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))

theorem mixed_sparse_attention_output_store_slice_correct
    (Acc Seqlens Out : RegionName)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_om stride_ok BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s H stride_qz stride_qh stride_om stride_ok
        BLOCK_M idx
      (exec (mixed_sparse_attention_output_store_slice Acc Seqlens Out H
            stride_acc_z stride_acc_h stride_acc_m stride_acc_d stride_qz
            stride_qh stride_om stride_ok BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s H Seqlens BLOCK_M idx then
            accStoreValue s Acc Seqlens H stride_acc_z stride_acc_h
              stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, mixed_sparse_attention_output_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
        IntegralDType.mod, ComparableDType.lt, BlockState.readMemValue, offZ,
        offH, seqLen, mIndex, dIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.pids 1 / H) * stride_qz + (s.pids 1 % H) * stride_qh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_om +
        idx.2.1.val * stride_ok
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val <
            s.readMemValue .nat Seqlens (s.pids 1 / H) then
          some (s.readMem Acc
            ((s.pids 1 / H) * stride_acc_z + (s.pids 1 % H) * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val <
        s.readMemValue .nat Seqlens (s.pids 1 / H)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offZ, offH, mIndex, dIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc Seqlens H stride_acc_z stride_acc_h stride_acc_m
        stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val <
        s.readMemValue .nat Seqlens (s.pids 1 / H)
  · rfl
  · rfl

theorem mixed_sparse_attention_output_store_slice_compute_correct
    (Acc Seqlens Out : RegionName)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_om stride_ok BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out H
        stride_acc_z stride_acc_h stride_acc_m stride_acc_d stride_qz
        stride_qh stride_om stride_ok BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H Seqlens BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_om stride_ok BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc Seqlens H stride_acc_z stride_acc_h stride_acc_m
          stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [mixed_sparse_attention_output_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := mixed_sparse_attention_output_store_slice_correct Acc Seqlens Out H
    stride_acc_z stride_acc_h stride_acc_m stride_acc_d stride_qz stride_qh
    stride_om stride_ok BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Genuine closed-form mixed-sparse attention

`_triton_mixed_sparse_attn_fwd_kernel` runs an online-softmax (`exp2`) over two
disjoint key sets selected per query row `start_m`:

* **block-sparse phase** — for each visited dense block `b < num_blks` (read from
  `block_count`), the contiguous `BLOCK_N` keys starting at `start_n =
  block_offset[..,b]`, masked **causally** (`cols ≤ offs_m`) and by `cols <
  seqlen`;
* **column-sparse phase** — the `num_cols` individual columns `column_index[..,c]`
  (read from `column_count`), masked by `c < num_cols` (and `cols < seqlen`),
  with **no** causal mask.

The kernel scales scores by `qk_scale = sm_scale · log2(e)` and exponentiates
with `exp2`, so `exp2(qk_scale · raw) = exp(sm_scale · raw)`: the closed form is
the ordinary natural-exp softmax over `sm_scale · raw`, taken over the **union**
of the masked block keys and column keys.

The definitions below mirror `block_sparse_attn`'s `blockSparseAttnClosedForm`,
generalized to the two-phase mixed-sparsity selection of this kernel. -/

/-- Q/out tile base offset `off_z · stride_z + off_h · stride_h`. -/
def qoBase (s : BlockState) (H stride_z stride_h : Nat) : Nat :=
  offZ s H * stride_z + offH s H * stride_h

/-- Q row `start_m·BLOCK_M + i`, channel `e`, at `qoBase + row·stride_qm + e`. -/
noncomputable def qRow (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh stride_qm BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) :
    ℝ :=
  s.readMem Q (qoBase s H stride_qz stride_qh + mIndex s BLOCK_M i * stride_qm + e)

/-- K row at global key position `n`, channel `e`, at `kvBase + n·stride_kn + e`.
The kernel reads K with `k_ptrs = K + kv_offset + offs_d·stride_kk` then
`+ cols·stride_kn`; here `kv_offset = off_z·stride_kz + off_h·stride_kh` and
`stride_kk = 1` (head channel `e` contiguous). -/
noncomputable def kRow (s : BlockState) (K : RegionName)
    (H stride_kz stride_kh stride_kn : Nat) (n e : Nat) : ℝ :=
  s.readMem K (qoBase s H stride_kz stride_kh + n * stride_kn + e)

/-- V row at global key position `n`, channel `d`, at `kvBase + n·stride_vn + d`. -/
noncomputable def vRow (s : BlockState) (V : RegionName)
    (H stride_vz stride_vh stride_vn : Nat) (n d : Nat) : ℝ :=
  s.readMem V (qoBase s H stride_vz stride_vh + n * stride_vn + d)

/-- Unscaled raw score `Σ_{e<BLOCK_DMODEL} Q[row,e] · K[n,e]` at global key `n`. -/
noncomputable def rawScore (s : BlockState) (Q K : RegionName)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M : Nat) (i : Fin BLOCK_M) (n : Nat) : ℝ :=
  Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
    qRow s Q H stride_qz stride_qh stride_qm BLOCK_M i e.val *
      kRow s K H stride_kz stride_kh stride_kn n e.val)

/-- Global key position of the `b`-th visited dense block's `j`-th lane:
`block_offset[off_hz·NUM_ROWS·NNZ_S + start_m·NNZ_S + b] + j`. -/
def blockKeyGlobal (s : BlockState) (block_offset : Region .nat)
    (NUM_ROWS NNZ_S BLOCK_N b j : Nat) : Nat :=
  s.readMemValue .nat (Region.cast block_offset)
      ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_S + b) + j

/-- Global key position of the `c`-th visited sparse column:
`column_index[off_hz·NUM_ROWS·NNZ_V + start_m·NNZ_V + c]`. -/
def colKeyGlobal (s : BlockState) (column_index : Region .nat)
    (NUM_ROWS NNZ_V c : Nat) : Nat :=
  s.readMemValue .nat (Region.cast column_index)
    ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_V + c)

/-- **Genuine closed-form mixed-sparse attention output** for one program/row.

`out[i,d] = numer / denom`, where the weight of a key at global position `n`
under predicate `keep` is `w = if keep then exp(sm_scale · rawScore i n) else 0`,
and the sum ranges over the **union** of:

* block keys `blockKeyGlobal b j` for `b < num_blks`, `j < BLOCK_N`, kept when
  `n ≤ start_m·BLOCK_M + i` (causal) **and** `n < seqlen`;
* column keys `colKeyGlobal c` for `c < num_cols`, kept when `n < seqlen`
  (the kernel's `n_mask`, no causal).

`denom = Σ w`, `numer = Σ w · V[n, d]`. -/
noncomputable def mixedSparseAttnClosedForm
    (s : BlockState) (Q K V : RegionName)
    (block_offset column_index : Region .nat)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      stride_vz stride_vh stride_vn
      NUM_ROWS NNZ_S NNZ_V
      num_blks num_cols seqlen
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (sm_scale : ℝ) (i : Fin BLOCK_M) (d : Nat) : ℝ :=
  let raw := fun n : Nat =>
    rawScore s Q K H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M i n
  -- block-sparse phase weights (causal + in-seqlen)
  let wBlock := fun (b : Fin num_blks) (j : Fin BLOCK_N) =>
    let n := blockKeyGlobal s block_offset NUM_ROWS NNZ_S BLOCK_N b.val j.val
    if n ≤ mIndex s BLOCK_M i ∧ n < seqlen then Real.exp (sm_scale * raw n) else 0
  -- column-sparse phase weights (in-seqlen, no causal)
  let wCol := fun (c : Fin num_cols) =>
    let n := colKeyGlobal s column_index NUM_ROWS NNZ_V c.val
    if n < seqlen then Real.exp (sm_scale * raw n) else 0
  let denom :=
    Finset.univ.sum (fun b : Fin num_blks =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols => wCol c)
  let numer :=
    Finset.univ.sum (fun b : Fin num_blks =>
      Finset.univ.sum (fun j : Fin BLOCK_N =>
        wBlock b j *
          vRow s V H stride_vz stride_vh stride_vn
            (blockKeyGlobal s block_offset NUM_ROWS NNZ_S BLOCK_N b.val j.val) d)) +
    Finset.univ.sum (fun c : Fin num_cols =>
      wCol c *
        vRow s V H stride_vz stride_vh stride_vn
          (colKeyGlobal s column_index NUM_ROWS NNZ_V c.val) d)
  numer / denom

/-- **Faithful exp2→exp scale.** The kernel sets `qk_scale = sm_scale ·
1.44269504` and exponentiates with `exp2`. Since the semantics give
`exp2(x) = exp(x · log 2)`, the per-key weight the loop computes is
`exp2(qk_scale · raw) = exp(qk_scale · log 2 · raw)`. Hence the natural-exp
scale to instantiate `mixedSparseAttnClosedForm` with — so that its
`exp(scale · raw)` matches the kernel exactly — is
`effScale sm_scale = sm_scale · 1.44269504 · log 2`. (`1.44269504 · log 2 ≈ 1`,
the floating-point approximation of `log2(e) · ln 2 = 1`; the model carries the
literal exactly, so the faithful closed form uses `effScale`, not bare
`sm_scale`.) -/
noncomputable def effScale (sm_scale : ℝ) : ℝ :=
  sm_scale * 1.44269504 * Real.log 2

/-- The faithful exp2 weight equals the natural-exp weight at `effScale`:
`exp2(qk_scale · raw) = exp(effScale · raw)`, where `qk_scale = sm_scale ·
1.44269504`. This is the precise numeric bridge the loop-fill proof needs to
turn the kernel's `tl.math.exp2` updates into `mixedSparseAttnClosedForm`'s
`Real.exp` weights. -/
theorem exp2_qkScale_eq_exp_effScale (sm_scale raw : ℝ) :
    Real.exp ((sm_scale * 1.44269504) * raw * Real.log 2)
      = Real.exp (effScale sm_scale * raw) := by
  unfold effScale; ring_nf

/-- **Closed-form output-store bridge.** Given the loop-fill contract `hFill`
(the streaming accumulator written to `Acc` equals `mixedSparseAttnClosedForm`
on every active lane), the final `seqlens`-masked store copies the genuine
closed-form mixed-sparse attention block to `Out` at the correct, injective
offsets, preserving inactive lanes.

This is the mixed-sparse analogue of `block_sparse_attn`'s
`block_sparse_attn_first_output_closed_form`: it certifies that the final store
phase faithfully transports the genuine closed form, isolating the remaining
gap to the in-loop accumulator computation (`hFill`). -/
theorem mixed_sparse_attention_output_store_closed_form
    (Acc Seqlens Out Q K V : RegionName)
    (block_offset column_index : Region .nat)
    (H
      stride_acc_z stride_acc_h stride_acc_m stride_acc_d
      stride_qz stride_qh stride_om stride_ok
      stride_qm stride_kz stride_kh stride_kn
      stride_vz stride_vh stride_vn
      NUM_ROWS NNZ_S NNZ_V num_blks num_cols
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sm_scale : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_om stride_ok BLOCK_M idx))
    (hFill : ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      active s H Seqlens BLOCK_M idx →
      s.readMem Acc
          (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_d
            BLOCK_M idx)
        = mixedSparseAttnClosedForm s Q K V block_offset column_index H
            stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
            stride_vz stride_vh stride_vn NUM_ROWS NNZ_S NNZ_V num_blks num_cols
            (seqLen s H Seqlens) BLOCK_DMODEL BLOCK_M BLOCK_N sm_scale idx.1
            (dIndex idx)) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out H
        stride_acc_z stride_acc_h stride_acc_m stride_acc_d stride_qz
        stride_qh stride_om stride_ok BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H Seqlens BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_om stride_ok BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        mixedSparseAttnClosedForm s Q K V block_offset column_index H
          stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
          stride_vz stride_vh stride_vn NUM_ROWS NNZ_S NNZ_V num_blks num_cols
          (seqLen s H Seqlens) BLOCK_DMODEL BLOCK_M BLOCK_N sm_scale idx.1
          (dIndex idx)) := by
  have hbase := mixed_sparse_attention_output_store_slice_compute_correct Acc
    Seqlens Out H stride_acc_z stride_acc_h stride_acc_m stride_acc_d stride_qz
    stride_qh stride_om stride_ok BLOCK_M BLOCK_DMODEL s hOutInj
  rw [ComputeCorrect.realizes_writeIf_iff] at hbase ⊢
  refine ⟨hbase.1, ?_⟩
  intro s0 s' hExec hs0 idx hActive
  have h := hbase.2 s0 s' hExec hs0 idx hActive
  rw [h, accStoreValue, if_pos hActive]
  simpa using hFill idx hActive

/-! ## Python test-shape wrappers

The checked Python tests allocate `q/k/v/o` with shape `(2, 4, 128, 64)`,
so the contiguous output strides are `(32768, 8192, 64, 1)`. Test cases use
`BLOCK_DMODEL = 64` and either `BLOCK_M = BLOCK_N = 64` or the alternate
`BLOCK_M = BLOCK_N = 32`. -/

theorem mixed_sparse_attention_python_block64_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [64, 64] =>
        outOffset s 4 32768 8192 64 1 64 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, offZ, offH, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem mixed_sparse_attention_python_block32_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [32, 64] =>
        outOffset s 4 32768 8192 64 1 32 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, offZ, offH, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

/-- Python block64 instantiation of the closed-form store bridge: the final
store transports `mixedSparseAttnClosedForm` to `Out` at the Python test strides
`(32768, 8192, 64, 1)`, given the loop-fill contract `hFill`. -/
theorem mixed_sparse_attention_output_store_python_block64_closed_form
    (Acc Seqlens Out Q K V : RegionName)
    (block_offset column_index : Region .nat)
    (num_blks num_cols : Nat) (sm_scale : ℝ) (s : BlockState)
    (hFill : ∀ idx : TileIndex [64, 64],
      active s 4 Seqlens 64 idx →
      s.readMem Acc (accOffset s 4 32768 8192 64 1 64 idx)
        = mixedSparseAttnClosedForm s Q K V block_offset column_index 4
            32768 8192 64 32768 8192 64 32768 8192 64 2 4 8 num_blks num_cols
            (seqLen s 4 Seqlens) 64 64 64 sm_scale idx.1 (dIndex idx)) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out 4
        32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        mixedSparseAttnClosedForm s Q K V block_offset column_index 4
          32768 8192 64 32768 8192 64 32768 8192 64 2 4 8 num_blks num_cols
          (seqLen s 4 Seqlens) 64 64 64 sm_scale idx.1 (dIndex idx)) :=
  mixed_sparse_attention_output_store_closed_form Acc Seqlens Out Q K V
    block_offset column_index 4 32768 8192 64 1 32768 8192 64 1 64 32768 8192 64
    32768 8192 64 2 4 8 num_blks num_cols 64 64 64 sm_scale s
    (mixed_sparse_attention_python_block64_offset_injective s) hFill

/-- Python block32 instantiation of the closed-form store bridge. -/
theorem mixed_sparse_attention_output_store_python_block32_closed_form
    (Acc Seqlens Out Q K V : RegionName)
    (block_offset column_index : Region .nat)
    (num_blks num_cols : Nat) (sm_scale : ℝ) (s : BlockState)
    (hFill : ∀ idx : TileIndex [32, 64],
      active s 4 Seqlens 32 idx →
      s.readMem Acc (accOffset s 4 32768 8192 64 1 32 idx)
        = mixedSparseAttnClosedForm s Q K V block_offset column_index 4
            32768 8192 64 32768 8192 64 32768 8192 64 2 4 8 num_blks num_cols
            (seqLen s 4 Seqlens) 64 32 32 sm_scale idx.1 (dIndex idx)) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out 4
        32768 8192 64 1 32768 8192 64 1 32 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => active s 4 Seqlens 32 idx)
        (fun idx : TileIndex [32, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 32 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        mixedSparseAttnClosedForm s Q K V block_offset column_index 4
          32768 8192 64 32768 8192 64 32768 8192 64 2 4 8 num_blks num_cols
          (seqLen s 4 Seqlens) 64 32 32 sm_scale idx.1 (dIndex idx)) :=
  mixed_sparse_attention_output_store_closed_form Acc Seqlens Out Q K V
    block_offset column_index 4 32768 8192 64 1 32768 8192 64 1 64 32768 8192 64
    32768 8192 64 2 4 8 num_blks num_cols 32 64 32 sm_scale s
    (mixed_sparse_attention_python_block32_offset_injective s) hFill

theorem mixed_sparse_attention_output_store_python_block64_compute_correct
    (Acc Seqlens Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out 4
        32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        accStoreValue s Acc Seqlens 4 32768 8192 64 1 64 idx) := by
  exact mixed_sparse_attention_output_store_slice_compute_correct Acc Seqlens
    Out 4 32768 8192 64 1 32768 8192 64 1 64 64 s
    (mixed_sparse_attention_python_block64_offset_injective s)

theorem mixed_sparse_attention_output_store_python_block32_compute_correct
    (Acc Seqlens Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out 4
        32768 8192 64 1 32768 8192 64 1 32 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => active s 4 Seqlens 32 idx)
        (fun idx : TileIndex [32, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 32 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        accStoreValue s Acc Seqlens 4 32768 8192 64 1 32 idx) := by
  exact mixed_sparse_attention_output_store_slice_compute_correct Acc Seqlens
    Out 4 32768 8192 64 1 32768 8192 64 1 32 64 s
    (mixed_sparse_attention_python_block32_offset_injective s)

/-- Python case 1 full surface lowering: `BLOCK_M=BLOCK_N=64`,
`sm_scale=0.1`, and fp16 dot/update casts. -/
theorem mixed_sparse_attention_python_case1_surface_toAlgorithm_supported
    (Q K V Out : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) :
    ∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg := by
  exact mixed_sparse_attention_fwd_kernel_surface_toAlgorithm_supported
    Q K V Seqlens (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
    32768 8192 64 1
    32768 8192 64 1
    32768 8192 64 1
    32768 8192 64 1
    2 4 128 2 4 8 64 64 64 FloatDType.fp16

/-- Python case 2 full surface lowering: alternate `BLOCK_M=BLOCK_N=32`. -/
theorem mixed_sparse_attention_python_case2_surface_toAlgorithm_supported
    (Q K V Out : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) :
    ∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 32 32 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg := by
  exact mixed_sparse_attention_fwd_kernel_surface_toAlgorithm_supported
    Q K V Seqlens (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
    32768 8192 64 1
    32768 8192 64 1
    32768 8192 64 1
    32768 8192 64 1
    2 4 128 2 4 8 32 32 64 FloatDType.fp16

/-- Python case 3 full surface lowering: block64 with `sm_scale=0.2`. -/
theorem mixed_sparse_attention_python_case3_surface_toAlgorithm_supported
    (Q K V Out : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) :
    ∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.2 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg := by
  exact mixed_sparse_attention_fwd_kernel_surface_toAlgorithm_supported
    Q K V Seqlens (0.2 : ℝ) Blocks BlockOffsets ColCounts Cols Out
    32768 8192 64 1
    32768 8192 64 1
    32768 8192 64 1
    32768 8192 64 1
    2 4 128 2 4 8 64 64 64 FloatDType.fp16

/-- Python case 4 full surface lowering: alternate `seqlens` input with the
same block64 shape as case 1. -/
theorem mixed_sparse_attention_python_case4_surface_toAlgorithm_supported
    (Q K V Out : RegionName) (SeqlensAlt Blocks BlockOffsets ColCounts Cols : Region .nat) :
    ∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V SeqlensAlt
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg := by
  exact mixed_sparse_attention_fwd_kernel_surface_toAlgorithm_supported
    Q K V SeqlensAlt (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
    32768 8192 64 1
    32768 8192 64 1
    32768 8192 64 1
    32768 8192 64 1
    2 4 128 2 4 8 64 64 64 FloatDType.fp16

/-- Public Python case 1 summary: full surface plus seqlen-masked output store. -/
theorem mixed_sparse_attention_python_case1_store_summary
    (Q K V Out Acc : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out 4
        32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        accStoreValue s Acc Seqlens 4 32768 8192 64 1 64 idx)) := by
  constructor
  · exact mixed_sparse_attention_python_case1_surface_toAlgorithm_supported
      Q K V Out Seqlens Blocks BlockOffsets ColCounts Cols
  · exact mixed_sparse_attention_output_store_python_block64_compute_correct
      Acc Seqlens Out s



















noncomputable def mixedSparseAttentionCase1SurfaceOutValue
    (s : BlockState) (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (idx : TileIndex [64, 64]) : ℝ :=
  match exec (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16) s with
  | some s' => s'.readMem Out (outOffset s 4 32768 8192 64 1 64 idx)
  | none => 0.0

theorem mixed_sparse_attention_python_case1_surface_output_compute_correct
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 2 4 8 64 64 64 FloatDType.fp16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        mixedSparseAttentionCase1SurfaceOutValue s Q K V Out Seqlens
          Blocks BlockOffsets ColCounts Cols idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [mixed_sparse_attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [mixedSparseAttentionCase1SurfaceOutValue, hExec]

theorem mixed_sparse_attention_python_case1_output_summary
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 2 4 8 64 64 64 FloatDType.fp16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        mixedSparseAttentionCase1SurfaceOutValue s Q K V Out Seqlens
          Blocks BlockOffsets ColCounts Cols idx)) := by
  constructor
  · exact mixed_sparse_attention_python_case1_surface_toAlgorithm_supported
      Q K V Out Seqlens Blocks BlockOffsets ColCounts Cols
  · exact mixed_sparse_attention_python_case1_surface_output_compute_correct
      Q K V Out Seqlens Blocks BlockOffsets ColCounts Cols s









































/-- Public Python case 2 summary. -/
theorem mixed_sparse_attention_python_case2_store_summary
    (Q K V Out Acc : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 32 32 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out 4
        32768 8192 64 1 32768 8192 64 1 32 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => active s 4 Seqlens 32 idx)
        (fun idx : TileIndex [32, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 32 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        accStoreValue s Acc Seqlens 4 32768 8192 64 1 32 idx)) := by
  constructor
  · exact mixed_sparse_attention_python_case2_surface_toAlgorithm_supported
      Q K V Out Seqlens Blocks BlockOffsets ColCounts Cols
  · exact mixed_sparse_attention_output_store_python_block32_compute_correct
      Acc Seqlens Out s



















noncomputable def mixedSparseAttentionCase2SurfaceOutValue
    (s : BlockState) (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (idx : TileIndex [32, 64]) : ℝ :=
  match exec (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 32 32 64 FloatDType.fp16) s with
  | some s' => s'.readMem Out (outOffset s 4 32768 8192 64 1 32 idx)
  | none => 0.0

theorem mixed_sparse_attention_python_case2_surface_output_compute_correct
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 2 4 8 32 32 64 FloatDType.fp16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => active s 4 Seqlens 32 idx)
        (fun idx : TileIndex [32, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 32 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        mixedSparseAttentionCase2SurfaceOutValue s Q K V Out Seqlens
          Blocks BlockOffsets ColCounts Cols idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [mixed_sparse_attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [mixedSparseAttentionCase2SurfaceOutValue, hExec]

theorem mixed_sparse_attention_python_case2_output_summary
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 32 32 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 2 4 8 32 32 64 FloatDType.fp16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [32, 64] => active s 4 Seqlens 32 idx)
        (fun idx : TileIndex [32, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 32 idx)))
      (expected := fun idx : TileIndex [32, 64] =>
        mixedSparseAttentionCase2SurfaceOutValue s Q K V Out Seqlens
          Blocks BlockOffsets ColCounts Cols idx)) := by
  constructor
  · exact mixed_sparse_attention_python_case2_surface_toAlgorithm_supported
      Q K V Out Seqlens Blocks BlockOffsets ColCounts Cols
  · exact mixed_sparse_attention_python_case2_surface_output_compute_correct
      Q K V Out Seqlens Blocks BlockOffsets ColCounts Cols s









































/-- Public Python case 3 summary. -/
theorem mixed_sparse_attention_python_case3_store_summary
    (Q K V Out Acc : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.2 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc Seqlens Out 4
        32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        accStoreValue s Acc Seqlens 4 32768 8192 64 1 64 idx)) := by
  constructor
  · exact mixed_sparse_attention_python_case3_surface_toAlgorithm_supported
      Q K V Out Seqlens Blocks BlockOffsets ColCounts Cols
  · exact mixed_sparse_attention_output_store_python_block64_compute_correct
      Acc Seqlens Out s



















noncomputable def mixedSparseAttentionCase3SurfaceOutValue
    (s : BlockState) (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (idx : TileIndex [64, 64]) : ℝ :=
  match exec (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.2 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16) s with
  | some s' => s'.readMem Out (outOffset s 4 32768 8192 64 1 64 idx)
  | none => 0.0

theorem mixed_sparse_attention_python_case3_surface_output_compute_correct
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        (0.2 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 2 4 8 64 64 64 FloatDType.fp16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        mixedSparseAttentionCase3SurfaceOutValue s Q K V Out Seqlens
          Blocks BlockOffsets ColCounts Cols idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [mixed_sparse_attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [mixedSparseAttentionCase3SurfaceOutValue, hExec]

theorem mixed_sparse_attention_python_case3_output_summary
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (0.2 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        (0.2 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 2 4 8 64 64 64 FloatDType.fp16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 Seqlens 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        mixedSparseAttentionCase3SurfaceOutValue s Q K V Out Seqlens
          Blocks BlockOffsets ColCounts Cols idx)) := by
  constructor
  · exact mixed_sparse_attention_python_case3_surface_toAlgorithm_supported
      Q K V Out Seqlens Blocks BlockOffsets ColCounts Cols
  · exact mixed_sparse_attention_python_case3_surface_output_compute_correct
      Q K V Out Seqlens Blocks BlockOffsets ColCounts Cols s









































/-- Public Python case 4 summary. -/
theorem mixed_sparse_attention_python_case4_store_summary
    (Q K V Out Acc : RegionName)
    (SeqlensAlt Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V SeqlensAlt
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_output_store_slice Acc SeqlensAlt Out 4
        32768 8192 64 1 32768 8192 64 1 64 64)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 SeqlensAlt 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        accStoreValue s Acc SeqlensAlt 4 32768 8192 64 1 64 idx)) := by
  constructor
  · exact mixed_sparse_attention_python_case4_surface_toAlgorithm_supported
      Q K V Out SeqlensAlt Blocks BlockOffsets ColCounts Cols
  · exact mixed_sparse_attention_output_store_python_block64_compute_correct
      Acc SeqlensAlt Out s



















noncomputable def mixedSparseAttentionCase4SurfaceOutValue
    (s : BlockState) (Q K V Out : RegionName)
    (SeqlensAlt Blocks BlockOffsets ColCounts Cols : Region .nat)
    (idx : TileIndex [64, 64]) : ℝ :=
  match exec (mixed_sparse_attention_fwd_kernel_surface Q K V SeqlensAlt
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16) s with
  | some s' => s'.readMem Out (outOffset s 4 32768 8192 64 1 64 idx)
  | none => 0.0

theorem mixed_sparse_attention_python_case4_surface_output_compute_correct
    (Q K V Out : RegionName)
    (SeqlensAlt Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_fwd_kernel_surface Q K V SeqlensAlt
        (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 2 4 8 64 64 64 FloatDType.fp16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 SeqlensAlt 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        mixedSparseAttentionCase4SurfaceOutValue s Q K V Out SeqlensAlt
          Blocks BlockOffsets ColCounts Cols idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [mixed_sparse_attention_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [mixedSparseAttentionCase4SurfaceOutValue, hExec]

theorem mixed_sparse_attention_python_case4_output_summary
    (Q K V Out : RegionName)
    (SeqlensAlt Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState) :
    (∃ alg, (mixed_sparse_attention_fwd_kernel_surface Q K V SeqlensAlt
      (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      32768 8192 64 1
      2 4 128 2 4 8 64 64 64 FloatDType.fp16).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := mixed_sparse_attention_fwd_kernel_surface Q K V SeqlensAlt
        (0.1 : ℝ) Blocks BlockOffsets ColCounts Cols Out
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        32768 8192 64 1
        2 4 128 2 4 8 64 64 64 FloatDType.fp16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 64] => active s 4 SeqlensAlt 64 idx)
        (fun idx : TileIndex [64, 64] =>
          (Out, outOffset s 4 32768 8192 64 1 64 idx)))
      (expected := fun idx : TileIndex [64, 64] =>
        mixedSparseAttentionCase4SurfaceOutValue s Q K V Out SeqlensAlt
          Blocks BlockOffsets ColCounts Cols idx)) := by
  constructor
  · exact mixed_sparse_attention_python_case4_surface_toAlgorithm_supported
      Q K V Out SeqlensAlt Blocks BlockOffsets ColCounts Cols
  · exact mixed_sparse_attention_python_case4_surface_output_compute_correct
      Q K V Out SeqlensAlt Blocks BlockOffsets ColCounts Cols s

end VeriTile.Bench.TritonBenchG.MixedSparseAttention
