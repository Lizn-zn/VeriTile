import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Kernel

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
mixed_sparse_attention_output_closed_form_summary_general              ← TOP THEOREM (dimension-general)
  ├─ (surface lowers to the algorithm layer — discharged inline)
  └─ msa_execGS                                                          streaming exec writes the closed form
       ├─ msaPreLoop_evalGS / msa_attn_stepAGS / msa_attn_stepBGS / msaPostLoop_evalGS   online-softmax fold steps
       │    (dense-block phase A + sparse-column phase B, composed by forRangeDyn_inv)
       └─ mixedSparseAttnClosedForm                                      genuine closed form over loaded Q/K/V
```
(Offset injectivity discharged by `mixed_sparse_attention_offset_injectiveGS`.)

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; `exp2`, `tl.dot`, and the
`sm_scale · log2(e)` scaling are not modeled at the bit level);
`num_warps` / `num_stages` are not modeled. The verified result is the **genuine
closed-form attention output** `mixedSparseAttnClosedForm`:
the full faithful surface kernel, executed statement-by-statement via the
dimension-general `msa_execGS` (symbolic `BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`
streaming), writes
`mixedSparseAttnClosedForm` — the closed-form online-softmax over the visited
dense blocks and sparse columns, a function of memory and never
self-referential — to every active `Out` lane, while preserving inactive lanes
(the `offs_m < seqlen` and `start_m·BLOCK_M ≥ seqlen` early-exit masking) at the
correct, injective output offsets (discharged by
`mixed_sparse_attention_offset_injectiveGS`). The closed-form fact is the top
theorem
`mixed_sparse_attention_output_closed_form_summary_general`, which holds at
symbolic block dims (`0 < BLOCK_N`, `16 ≤ BLOCK_N`, `BLOCK_DMODEL ≤ 64`) under
honest side conditions, subsuming the former pinned per-case Python summaries.
-/

namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention

open VeriTile

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

def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok

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

/-- Global key position of the `c`-th visited sparse column:
`column_index[off_hz·NUM_ROWS·NNZ_V + start_m·NNZ_V + c]`. -/
def colKeyGlobal (s : BlockState) (column_index : Region .nat)
    (NUM_ROWS NNZ_V c : Nat) : Nat :=
  s.readMemValue .nat (Region.cast column_index)
    ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_V + c)

/-- **Faithful exp2→exp scale.** The kernel sets `qk_scale = sm_scale ·
1.44269504` and exponentiates with `exp2`. Since the semantics give
`exp2(x) = exp(x · log 2)`, the per-key weight the loop computes is
`exp2(qk_scale · raw) = exp(qk_scale · log 2 · raw)`. Hence the natural-exp
scale to instantiate the closed form with is
`effScale sm_scale = sm_scale · 1.44269504 · log 2`. (`1.44269504 · log 2 ≈ 1`,
the floating-point approximation of `log2(e) · ln 2 = 1`.) -/
noncomputable def effScale (sm_scale : ℝ) : ℝ :=
  sm_scale * 1.44269504 * Real.log 2

/-- The masked block start `start_n` the kernel reads at block `b` (Loop A's
`tl.load(blks_ptr + b, mask = b < num_blks)`): the real offset for a visited
block, the masked default `0` for a spurious block `b ≥ num_blks`. -/
noncomputable def blockStartN (s : BlockState) (block_offset : Region .nat)
    (NUM_ROWS NNZ_S num_blks b : Nat) : Nat :=
  if b < num_blks then
    s.readMemValue .nat (Region.cast block_offset)
      ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_S + b)
  else BlockState.defaultCarrier .nat

/-- **Genuine (FAITHFUL) closed-form mixed-sparse attention output** for one
program/row. This mirrors *exactly* what
`_triton_mixed_sparse_attn_fwd_kernel` computes — including the faithfulness
quirk that **Loop A always runs `max_num_blks = 8` iterations regardless of
`num_blks`**.

For each iteration `b < 8` the kernel forms `cond = b < num_blks`, the masked
block start `start_n = blockStartN` (the masked default `0` when `cond` is
false), then for each lane `j < BLOCK_N` the key `n = start_n + j`:

* the K-load is masked by `n_mask = (n < seqlen) ∧ cond`, so the effective key
  vector is `K[n]` when `n < seqlen ∧ cond` and the **zero vector** otherwise;
* `qk = where(m_mask ∧ (n ≤ offs_m i), 0, -inf) + dot(q, K_masked)`, so the lane
  contributes weight `w = exp(effScale · rawMasked)` exactly when
  `offs_m i < seqlen ∧ n ≤ offs_m i`, and `0` otherwise. Here `rawMasked = raw n`
  when `n < seqlen ∧ cond` and `rawMasked = 0` (so `w = exp(0) = 1`) otherwise —
  this is the **spurious-block weight-1 path**: a block `b ≥ num_blks` has
  `start_n = 0`, `cond = false`, hence `n = j`, `n ≤ offs_m i` and
  (for active rows) `offs_m i < seqlen`, so it adds `exp(effScale·0) = 1` to the
  DENOMINATOR while its V is the zero vector, leaving the numerator unchanged.

Loop B (column phase) is correctly `n_mask`-guarded (its `where` masks
non-selected lanes to `⊥`), so spurious column lanes contribute nothing.

`numer/denom` is therefore the kernel's true output, **not** the naive
`num_blks`-only union softmax. The natural-exp scale is `effScale sm_scale =
sm_scale · 1.44269504 · log 2` (the faithful `exp2 → exp` bridge). -/
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
  -- block-sparse phase over ALL 8 = max_num_blks kernel iterations.
  -- `keep` = lane kept (causal + active row); `inSeq` = K/V actually loaded.
  let wBlock := fun (b : Fin 8) (j : Fin BLOCK_N) =>
    let SN := blockStartN s block_offset NUM_ROWS NNZ_S num_blks b.val
    let n := SN + j.val
    let inSeq := n < seqlen ∧ b.val < num_blks
    let rawMasked := if inSeq then raw n else 0
    if mIndex s BLOCK_M i < seqlen ∧ n ≤ mIndex s BLOCK_M i then
      Real.exp (effScale sm_scale * rawMasked) else 0
  let vBlock := fun (b : Fin 8) (j : Fin BLOCK_N) =>
    let SN := blockStartN s block_offset NUM_ROWS NNZ_S num_blks b.val
    let n := SN + j.val
    if n < seqlen ∧ b.val < num_blks then
      vRow s V H stride_vz stride_vh stride_vn n d else 0
  -- column-sparse phase weights. Faithful because the kernel `n_mask`-guards
  -- Loop B's `where`: a column lane `c < num_cols` is kept iff the row is active
  -- (`offs_m i < seqlen`). The kernel applies NO `cols < seqlen` mask to the
  -- column keys (only `c < num_cols ∧ 0 < num_cols`), so neither does this.
  let wCol := fun (c : Fin num_cols) =>
    let n := colKeyGlobal s column_index NUM_ROWS NNZ_V c.val
    if mIndex s BLOCK_M i < seqlen then
      Real.exp (effScale sm_scale * raw n) else 0
  let denom :=
    Finset.univ.sum (fun b : Fin 8 =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols => wCol c)
  let numer :=
    Finset.univ.sum (fun b : Fin 8 =>
      Finset.univ.sum (fun j : Fin BLOCK_N => wBlock b j * vBlock b j)) +
    Finset.univ.sum (fun c : Fin num_cols =>
      wCol c *
        vRow s V H stride_vz stride_vh stride_vn
          (colKeyGlobal s column_index NUM_ROWS NNZ_V c.val) d)
  numer / denom

section MSARecipes

open VeriTile

/-- Local `evalOp` unfolding for `.le` (mirrors `block_sparse_attn`'s `bsa_evalOp_ge`;
used for the causal `cols[None,:] ≤ offs_m[:,None]` mask). -/
theorem msa_evalOp_le {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.le h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.le bc vx vy)) := by
  simp [evalOp]

/-- Local `evalOp` unfolding for `.remap` (mirrors `block_sparse_attn`'s
`bsa_evalOp_remap`; used inside the masked K/V/column loads). -/
theorem msa_evalOp_remap {dtype inShape outShape}
    (map : TileIndex outShape → TileIndex inShape) (a : Op dtype inShape) (s : BlockState) :
    evalOp (.remap outShape map a) s = (do
      let va ← evalOp a s; some (Tile.remap map va)) := by
  simp [evalOp]

/-! ### Loop-A shared recipes (block-sparse). The softmax-core recipes
(`A7,A10..A18`) are shape-identical to loop B's `B6,B8..B16`. -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`cond = block_index < num_blks` (A1)** / **`cond = start_n < num_cols` (B1)**:
the loop-guard predicate, a scalar `<` of the loop counter against the per-row
nonzero count. Instantiate `counter := "block_index"`/`"num_blks"` for loop A,
`"start_n"`/`"num_cols"` for loop B. -/
theorem msa_cond_eval (s : BlockState) (counter bound : RegName) (CI NB : Nat)
    (hci : s.regs .nat [] counter = some (Tile.scalar CI))
    (hnb : s.regs .nat [] bound = some (Tile.scalar NB)) :
    evalOp (Op.lt ComparableDType.nat Broadcast.nil
        (Op.ref .nat [] counter) (Op.ref .nat [] bound)) s
      = some (Tile.cop ComparableDType.nat.lt Broadcast.nil
          (Tile.scalar CI) (Tile.scalar NB)) := by
  rw [evalOp_lt]
  simp only [evalOp_ref, hci, hnb, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`cols = start_n + offs_n` (A3)**: the contiguous block-key column indices,
`start_n` broadcast-added to `offs_n`. Given `start_n = SN`, `offs_n = id`, lane
`j` becomes `SN + j`. -/
theorem msa_cols_eval (s : BlockState) (BN SN : Nat)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hon : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))) :
    evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
        (Op.ref .nat [BN] "offs_n")) s
      = some (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar SN)
          (Tile.vec (fun j : Fin BN => j.val))) := by
  rw [evalOp_add]
  simp only [evalOp_ref, hsn, hon, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`n_mask = (cols < seqlen) & cond` (A4)**: the per-column validity mask
combining the in-seqlen test on the gathered columns with the loop guard `cond`.
Given `cols = ct`, `seqlen = SL`, `cond = cb`, lane `j` becomes
`decide (ct j < SL) && cb`. -/
theorem msa_nmask_eval (s : BlockState) (BN SL : Nat) (cb : Bool)
    (ct : Tile .nat [BN])
    (hcols : s.regs .nat [BN] "cols" = some ct)
    (hsl : s.regs .nat [] "seqlen" = some (Tile.scalar SL))
    (hcond : s.regs .bool [] "cond" = some (Tile.scalar cb)) :
    evalOp (Op.boolAnd Broadcast.scalarR
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [BN] "cols") (Op.ref .nat [] "seqlen"))
        (Op.ref .bool [] "cond")) s
      = some (Tile.bop (fun x y : Bool => x && y) Broadcast.scalarR
          (Tile.cop ComparableDType.nat.lt Broadcast.scalarR ct (Tile.scalar SL))
          (Tile.scalar cb)) := by
  have hand :
      evalOp (Op.boolAnd Broadcast.scalarR
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.ref .nat [BN] "cols") (Op.ref .nat [] "seqlen"))
          (Op.ref .bool [] "cond")) s = (do
        let va' ← evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [BN] "cols") (Op.ref .nat [] "seqlen")) s
        let vb' ← evalOp (Op.ref .bool [] "cond") s
        some (Tile.bop (fun x y : Bool => x && y) Broadcast.scalarR va' vb')) := by
    simp only [evalOp]
  rw [hand, evalOp_lt]
  simp only [evalOp_ref, hcols, hsl, hcond, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`qk = tl.zeros([BLOCK_M, BLOCK_N])` (A7/B6)**: the all-`0` pre-`where`/pre-dot
score accumulator. Mixed-sparse analogue of `bsa_qkzeros_eval`. -/
theorem msa_qkzeros_eval (s : BlockState) (BM BN : Nat) :
    evalOp (Op.full [BM, BN] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`causal_mask = cols[None,:] ≤ offs_m[:,None]` (A8, loop A ONLY)**: the causal
visibility predicate, true when key column `cols j` is at or before query row
`offs_m i`. Given `cols = ct`, `offs_m = om`, lane `(i,j)` becomes
`decide (ct j ≤ om i)`. Loop B has no causal mask. -/
theorem msa_causal_eval (s : BlockState) (BM BN : Nat)
    (hax0 : 0 < [BN].length.succ) (hax1 : 1 < [BM].length.succ)
    (ct : Tile .nat [BN]) (om : Tile .nat [BM])
    (hcols : s.regs .nat [BN] "cols" = some ct)
    (hom : s.regs .nat [BM] "offs_m" = some om) :
    evalOp (Op.le ComparableDType.nat Broadcast.nil.consR.consL
        (Op.expandDim ⟨0, hax0⟩ (Op.ref .nat [BN] "cols"))
        (Op.expandDim ⟨1, hax1⟩ (Op.ref .nat [BM] "offs_m"))) s
      = some (Tile.cop ComparableDType.nat.le Broadcast.nil.consR.consL
          (Tile.expandDim ⟨0, hax0⟩ ct) (Tile.expandDim ⟨1, hax1⟩ om)) := by
  rw [msa_evalOp_le]
  erw [evalOp_expandDim_ref_of_regs .nat [BN] ⟨0, hax0⟩ "cols" s _ hcols,
    evalOp_expandDim_ref_of_regs .nat [BM] ⟨1, hax1⟩ "offs_m" s _ hom]
  rfl

/-- Local `evalOp` unfolding for `.exp2` (no global simp lemma; the kernel uses
`tl.math.exp2` for both `alpha` and `p`). -/
theorem msa_evalOp_exp2 (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Causal additive-by-selection `where` (A9, loop A ONLY)**:
`qk = tl.where(m_mask & causal_mask, qk, -inf)` — a *selecting* mask (unlike
block-sparse's additive `where`): visible lanes keep `qk`, masked lanes become
`-inf`. Given `m_mask = mm`, `causal_mask = cm`, running `qk = qkt`, the result is
`select (mm & cm) qkt (-inf)`. -/
theorem msa_where_causal_eval (s : BlockState) (BM BN : Nat)
    (mm : Tile .bool [BM, 1]) (cm : Tile .bool [BM, BN]) (qkt : Tile .real [BM, BN])
    (hmm : s.regs .bool [BM, 1] "m_mask" = some mm)
    (hcm : s.regs .bool [BM, BN] "causal_mask" = some cm)
    (hqk : s.regs .real [BM, BN] "qk" = some qkt) :
    evalOp ((Op.boolAnd Broadcast.nil.consL.consSame
          (Op.ref .bool [BM, 1] "m_mask") (Op.ref .bool [BM, BN] "causal_mask")).where
        (Op.ref .real [BM, BN] "qk") (Op.negInf.broadcast [BM, BN])) s
      = some (Tile.select
          (Tile.bop (fun x y : Bool => x && y) Broadcast.nil.consL.consSame mm cm)
          qkt (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN])) := by
  have hand : evalOp (Op.boolAnd Broadcast.nil.consL.consSame
        (Op.ref .bool [BM, 1] "m_mask") (Op.ref .bool [BM, BN] "causal_mask")) s
      = some (Tile.bop (fun x y : Bool => x && y) Broadcast.nil.consL.consSame mm cm) := by
    simp only [evalOp, evalOp_ref, hmm, hcm, Option.bind_eq_bind, Option.bind_some]
  have hbcast : @evalOp TileDType.real [BM, BN] (Op.broadcast Op.negInf [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where, hand]
  simp only [evalOp_ref, hqk, hbcast, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Selecting `where` with the column mask (B7, loop B ONLY)**:
`qk = tl.where(m_mask & n_mask, qk, -inf)` — same selecting form as A9 but with the
non-causal `n_mask` (broadcast `[64]→[64,64]` via `Broadcast.nil.consL.leadR`). -/
theorem msa_where_col_eval (s : BlockState) (BM BN : Nat)
    (mm : Tile .bool [BM, 1]) (nm : Tile .bool [BN]) (qkt : Tile .real [BM, BN])
    (hmm : s.regs .bool [BM, 1] "m_mask" = some mm)
    (hnm : s.regs .bool [BN] "n_mask" = some nm)
    (hqk : s.regs .real [BM, BN] "qk" = some qkt) :
    evalOp ((Op.boolAnd Broadcast.nil.consL.leadR
          (Op.ref .bool [BM, 1] "m_mask") (Op.ref .bool [BN] "n_mask")).where
        (Op.ref .real [BM, BN] "qk") (Op.negInf.broadcast [BM, BN])) s
      = some (Tile.select
          (Tile.bop (fun x y : Bool => x && y) Broadcast.nil.consL.leadR mm nm)
          qkt (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN])) := by
  have hand : evalOp (Op.boolAnd Broadcast.nil.consL.leadR
        (Op.ref .bool [BM, 1] "m_mask") (Op.ref .bool [BN] "n_mask")) s
      = some (Tile.bop (fun x y : Bool => x && y) Broadcast.nil.consL.leadR mm nm) := by
    simp only [evalOp, evalOp_ref, hmm, hnm, Option.bind_eq_bind, Option.bind_some]
  have hbcast : @evalOp TileDType.real [BM, BN] (Op.broadcast Op.negInf [BM, BN]) s
      = some (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
    simp only [evalOp, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where, hand]
  simp only [evalOp_ref, hqk, hbcast, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`qk += tl.dot(q, k)` (A10/B8)**: adds `dot(q, k)` into the masked `qk`. The
kernel stores the scaled query as fp16 (`q = (q·qk_scale).to(DTYPE)`), so the dot
input is `castFloat fp16→real q`. Given `q = qf16` (`fp16`-tile) and `k = kt`,
result is `qk + dot(castReal qf16, kt)`. -/
theorem msa_qk_dot_eval (s : BlockState) (BM BN BD : Nat)
    (qktile : Tile .real [BM, BN]) (qf16 : Tile FloatDType.fp16.toTileDType [BM, BD])
    (ktile : Tile .real [BD, BN])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hq : s.regs FloatDType.fp16.toTileDType [BM, BD] "q" = some qf16)
    (hk : s.regs .real [BD, BN] "k" = some ktile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := [])
          ((Op.castFloat FloatDType.fp16 FloatDType.real
            (Op.ref FloatDType.fp16.toTileDType [BM, BD] "q")) : Op .real [BM, BD])
          (Op.ref .real [BD, BN] "k"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          qktile
          (Tile.dot []
            (⟨fun i => FloatDType.fp16.cast FloatDType.real (qf16.data i)⟩ : Tile .real [BM, BD])
            ktile)) := by
  have hcast : evalOp (Op.castFloat FloatDType.fp16 FloatDType.real
      (Op.ref FloatDType.fp16.toTileDType [BM, BD] "q")) s
      = some (⟨fun i => FloatDType.fp16.cast FloatDType.real (qf16.data i)⟩
          : Tile FloatDType.real.toTileDType [BM, BD]) := by
    rw [evalOp_castFloat]; simp only [evalOp_ref, hq, Option.bind_eq_bind, Option.bind_some]
  have hkr : evalOp (Op.ref .real [BD, BN] "k") s = some ktile := by rw [evalOp_ref, hk]
  have hdot : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := [])
        ((Op.castFloat FloatDType.fp16 FloatDType.real
          (Op.ref FloatDType.fp16.toTileDType [BM, BD] "q")) : Op .real [BM, BD])
        (Op.ref .real [BD, BN] "k")) s
      = some (Tile.dot []
          (⟨fun i => FloatDType.fp16.cast FloatDType.real (qf16.data i)⟩ : Tile .real [BM, BD])
          ktile) := by
    erw [evalOp_dot []
      ((Op.castFloat FloatDType.fp16 FloatDType.real
        (Op.ref FloatDType.fp16.toTileDType [BM, BD] "q")) : Op .real [BM, BD])
      (Op.ref .real [BD, BN] "k"), hcast, hkr]; rfl
  rw [evalOp_add]
  simp only [evalOp_ref, hqk, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`m_i_new = tl.maximum(m_i, tl.max(qk, 1))` (A11/B9)**: the running row-max
merge. Lowered to `where(m_i > reduceMax(qk,1), m_i, reduceMax(qk,1))` with the
reduced row inlined twice. The `reduceMaxDrop` result is supplied as `hrm` (its
`eraseAxis` shape blocks `rw`); result is `select (m_i > rmaxT) m_i rmaxT`. -/
theorem msa_minew_eval (s : BlockState) (BM BN : Nat)
    (mp : Tile .real [BM]) (qktile : Tile .real [BM, BN]) (rmaxT : Tile .real [BM])
    (hmi : s.regs .real [BM] "m_i" = some mp)
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qktile = some rmaxT) :
    evalOp ((Op.gt ComparableDType.real Broadcast.nil.consSame
          (Op.ref .real [BM] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
            (Op.ref .real [BM, BN] "qk"))).where
        (Op.ref .real [BM] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
          (Op.ref .real [BM, BN] "qk"))) s
      = some (Tile.select
          (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mp rmaxT) mp rmaxT) := by
  have hrmax : evalOp (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
      (Op.ref .real [BM, BN] "qk")) s = some rmaxT := by
    rw [evalOp_reduceMax]
    simp only [evalOp_ref, hqk, Tile.reduceMax_false, Option.bind_eq_bind, Option.bind_some]
    exact hrm
  rw [evalOp_where, evalOp_gt]
  erw [hrmax]
  simp only [evalOp_ref, hmi, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`alpha = tl.math.exp2(m_i - m_i_new)` (A12/B10)**: the running-accumulator
rescale factor `exp2(m_i - m_i_new)` (base-2 exp, not natural). -/
theorem msa_alpha_eval (s : BlockState) (BM : Nat) (mp mn : Tile .real [BM])
    (hmp : s.regs .real [BM] "m_i" = some mp)
    (hmn : s.regs .real [BM] "m_i_new" = some mn) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_i_new"))) s
      = some (Tile.uop WithBot.realExp2
          (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mp mn)) := by
  rw [msa_evalOp_exp2, evalOp_sub]
  simp only [evalOp_ref, hmp, hmn, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`p = tl.math.exp2(qk - m_i_new[:, None])` (A13/B11)**: the base-2 softmax
numerator, shifted by the merged running max `m_i_new`. -/
theorem msa_p_eval (s : BlockState) (BM BN : Nat) (hax : 1 < [BM].length.succ)
    (qktile : Tile .real [BM, BN]) (mn : Tile .real [BM])
    (hqk : s.regs .real [BM, BN] "qk" = some qktile)
    (hmn : s.regs .real [BM] "m_i_new" = some mn) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] "qk") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_i_new")))) s
      = some (Tile.uop WithBot.realExp2
          (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            qktile (Tile.expandDim ⟨1, hax⟩ mn))) := by
  have hexp : @evalOp TileDType.real [BM, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BM] "m_i_new")) s
      = some (Tile.expandDim ⟨1, hax⟩ mn) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmn
  rw [msa_evalOp_exp2, evalOp_sub]
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`acc_scale = l_i * 0 + alpha` (A14/B12)**: the per-row accumulator rescale
factor. The `l_i·0` term is the kernel's literal `l_i * 0 + alpha`, so `acc_scale`
defeq-equals `alpha` but the recipe carries the exact AST. -/
theorem msa_accscale_eval (s : BlockState) (BM : Nat) (li al : Tile .real [BM])
    (hli : s.regs .real [BM] "l_i" = some li)
    (hal : s.regs .real [BM] "alpha" = some al) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BM] "l_i") (Op.const 0))
        (Op.ref .real [BM] "alpha")) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul Broadcast.scalarR li
            (Tile.scalar (some (0 : ℝ) : WithBot ℝ)))
          al) := by
  rw [evalOp_add, evalOp_mul]
  simp only [evalOp_ref, evalOp_const, hli, hal, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`acc *= acc_scale[:, None]` (A15/B13)**: rescale the output accumulator by the
per-row `acc_scale`. -/
theorem msa_acc_rescale_eval (s : BlockState) (BM BD : Nat) (hax : 1 < [BM].length.succ)
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
/-- **`acc += tl.dot((p).to(DTYPE), v)` (A16/B14)**: the numerator accumulation
`acc + dot(p, v)`. The kernel casts `p` to `DTYPE` (fp16) for the dot, so the dot
input is `castFloat fp16→real (castFloat real→fp16 p)` (the fp16 round-trip). Given
`p = pt`, `v = vt`, result is `acc + dot(castReal (castFp16 pt), vt)`. -/
theorem msa_acc_eval (s : BlockState) (BM BN BD : Nat)
    (acctile : Tile .real [BM, BD]) (pt : Tile .real [BM, BN]) (vt : Tile .real [BN, BD])
    (hacc : s.regs .real [BM, BD] "acc" = some acctile)
    (hp : s.regs .real [BM, BN] "p" = some pt)
    (hv : s.regs .real [BN, BD] "v" = some vt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.dot (batch := [])
          ((Op.castFloat FloatDType.fp16 FloatDType.real
            (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "p"))) : Op .real [BM, BN])
          (Op.ref .real [BN, BD] "v"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acctile
          (Tile.dot []
            (⟨fun i => FloatDType.fp16.cast FloatDType.real
              (FloatDType.real.cast FloatDType.fp16 (pt.data i))⟩ : Tile .real [BM, BN])
            vt)) := by
  have hinner : evalOp (Op.castFloat FloatDType.real FloatDType.fp16
      (Op.ref .real [BM, BN] "p")) s
      = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (pt.data i)⟩
          : Tile FloatDType.fp16.toTileDType [BM, BN]) := by
    rw [evalOp_castFloat]
    simp only [FloatDType.toTileDType_real, evalOp_ref, hp,
      Option.bind_eq_bind, Option.bind_some]
  have hcast : evalOp (Op.castFloat FloatDType.fp16 FloatDType.real
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "p"))) s
      = some (⟨fun i => FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (pt.data i))⟩
            : Tile FloatDType.real.toTileDType [BM, BN]) := by
    rw [evalOp_castFloat, hinner]; rfl
  have hvr : evalOp (Op.ref .real [BN, BD] "v") s = some vt := by rw [evalOp_ref, hv]
  have hdot : @evalOp TileDType.real [BM, BD]
      (Op.dot (batch := [])
        ((Op.castFloat FloatDType.fp16 FloatDType.real
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "p"))) : Op .real [BM, BN])
        (Op.ref .real [BN, BD] "v")) s
      = some (Tile.dot []
          (⟨fun i => FloatDType.fp16.cast FloatDType.real
            (FloatDType.real.cast FloatDType.fp16 (pt.data i))⟩ : Tile .real [BM, BN]) vt) := by
    erw [evalOp_dot []
      ((Op.castFloat FloatDType.fp16 FloatDType.real
        (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "p"))) : Op .real [BM, BN])
      (Op.ref .real [BN, BD] "v"), hcast, hvr]; rfl
  rw [evalOp_add]
  simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`l_i = l_i * alpha + tl.sum(p, 1)` (A17/B15, carry)**: the online-softmax
running-denominator update: rescale the old `l_i` by `alpha`, add the current
block/column's `reduceSum(p, 1)`. -/
theorem msa_li_eval (s : BlockState) (BM BN : Nat)
    (li al : Tile .real [BM]) (pt : Tile .real [BM, BN])
    (hli : s.regs .real [BM] "l_i" = some li)
    (hal : s.regs .real [BM] "alpha" = some al)
    (hp : s.regs .real [BM, BN] "p" = some pt) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
          (Op.ref .real [BM, BN] "p"))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) li al)
          (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) pt)) := by
  have hsum : evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
      (Op.ref .real [BM, BN] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) pt) := by
    rw [evalOp_reduceSum]
    simp only [evalOp_ref, hp, Tile.reduceSum_false, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_add, evalOp_mul]
  erw [hsum]
  simp only [evalOp_ref, hli, hal, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`m_i = m_i_new` (A18/B16, carry)**: the running-max carry into the next
iteration — a bare register read. Mixed-sparse analogue of `bsa_reg_carry_eval`. -/
theorem msa_reg_carry_eval (s : BlockState) (BM : Nat) (name : RegName)
    (t : Tile .real [BM]) (h : s.regs .real [BM] name = some t) :
    evalOp (Op.ref .real [BM] name) s = some t := by
  rw [evalOp_ref, h]

/-! ### Loop-specific load / gather recipes -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`start_n = tl.load(blks_ptr + block_index, mask=cond)` (A2, loop A)**: the
masked `.nat` gather of the visited dense block's start column. `blks_ptr` holds a
scalar pointer `(bpReg, bpOff)`; the address is `+ block_index`. With `cond = true`
the lane reads `blks_ptr[bpOff + block_index]`; with `cond = false` it reads the
`.nat` default carrier. Given `block_index = BI`, `cond = cb`. -/
theorem msa_startn_eval (s : BlockState) (bpReg : RegionName) (bpOff BI : Nat) (cb : Bool)
    (hbp : s.regs .ptr [] "blks_ptr" = some (Tile.scalar (bpReg, bpOff)))
    (hbi : s.regs .nat [] "block_index" = some (Tile.scalar BI))
    (hcond : s.regs .bool [] "cond" = some (Tile.scalar cb)) :
    evalOp (Op.load .nat
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "blks_ptr") (Op.ref .nat [] "block_index")))
        (MaskOpt.mask (Op.ref .bool [] "cond"))) s
      = some (Tile.scalar
          (if cb then s.readMemValue .nat bpReg (bpOff + BI)
           else BlockState.defaultCarrier .nat)) := by
  simp only [evalOp, evalOp_ref, hbp, hbi, hcond, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  by_cases h : cb
  · simp only [Tile.ptrAdd_data, Tile.scalar_data_index, Broadcast.leftIndex,
      Broadcast.rightIndex, h, if_true, if_pos]
  · simp only [Tile.ptrAdd_data, Tile.scalar_data_index, Broadcast.leftIndex,
      Broadcast.rightIndex, h, if_false, if_neg, Bool.false_eq_true, not_false_iff]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`n_mask = (start_n + offs_n < num_cols) & cond` (B2, loop B)**: the
column-validity mask. Each lane `j` is kept iff `start_n + j < num_cols` and the
loop guard `cond` holds. Given `start_n = SN`, `offs_n = id`, `num_cols = NC`,
`cond = cb`. -/
theorem msa_nmask_col_eval (s : BlockState) (BN SN NC : Nat) (cb : Bool)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hon : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hnc : s.regs .nat [] "num_cols" = some (Tile.scalar NC))
    (hcond : s.regs .bool [] "cond" = some (Tile.scalar cb)) :
    evalOp (Op.boolAnd Broadcast.scalarR
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n"))
          (Op.ref .nat [] "num_cols"))
        (Op.ref .bool [] "cond")) s
      = some (Tile.bop (fun x y : Bool => x && y) Broadcast.scalarR
          (Tile.cop ComparableDType.nat.lt Broadcast.scalarR
            (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar SN)
              (Tile.vec (fun j : Fin BN => j.val)))
            (Tile.scalar NC))
          (Tile.scalar cb)) := by
  have hand :
      evalOp (Op.boolAnd Broadcast.scalarR
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n"))
            (Op.ref .nat [] "num_cols"))
          (Op.ref .bool [] "cond")) s = (do
        let va' ← evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n"))
          (Op.ref .nat [] "num_cols")) s
        let vb' ← evalOp (Op.ref .bool [] "cond") s
        some (Tile.bop (fun x y : Bool => x && y) Broadcast.scalarR va' vb')) := by
    simp only [evalOp]
  rw [hand, evalOp_lt, evalOp_add]
  simp only [evalOp_ref, hsn, hon, hnc, hcond, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`k = tl.load(k_ptrs + cols[None,:]·stride_kn, mask=n_mask[None,:], other=0.0)`
(A5/B4)**: the masked K-tile load. `k_ptrs` is the `[BM,1]` pointer tile, broadcast
across the `BLOCK_N` key columns; the per-column offset is `cols·stride_kn`, and the
`n_mask[None,:]` mask zeroes out-of-range columns. Given `k_ptrs = kp`, `cols = ct`,
`n_mask = nm`, the result is the lane-wise `if (remapped n_mask) then read(ptrAdd) else 0`
tile — kept symbolic for the assembly layer. With `stride_kn = SKN`. -/
theorem msa_load_k_eval (s : BlockState) (BM BN SKN : Nat)
    (hax0 : 0 < [BN].length.succ)
    (kp : Tile .ptr [BM, 1]) (ct : Tile .nat [BN]) (nm : Tile .bool [BN])
    (hk : s.regs .ptr [BM, 1] "k_ptrs" = some kp)
    (hcols : s.regs .nat [BN] "cols" = some ct)
    (hnm : s.regs .bool [BN] "n_mask" = some nm) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consL.consR (Op.ref .ptr [BM, 1] "k_ptrs")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, hax0⟩ (Op.ref .nat [BN] "cols")) (Op.constNat SKN))))
        (MaskOpt.maskOther
          (Op.remap [BM, BN] Broadcast.nil.consSame.consL.leftIndex
            (Op.expandDim ⟨0, hax0⟩ (Op.ref .bool [BN] "n_mask")))
          ((Op.const 0.0).broadcast [BM, BN]))) s
      = some (⟨fun idx : TileIndex [BM, BN] =>
          let ptrs := Tile.ptrAdd Broadcast.nil.consL.consR kp
            (Tile.bop NumericDType.nat.mul Broadcast.scalarR
              (Tile.expandDim ⟨0, hax0⟩ ct) (Tile.scalar SKN))
          if (Tile.remap Broadcast.nil.consSame.consL.leftIndex
                (Tile.expandDim ⟨0, hax0⟩ nm)).data idx then
            s.readMemValue .real (ptrs.data idx).1 (ptrs.data idx).2
          else (some (0.0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BM, BN]) := by
  simp only [evalOp, hk]
  erw [evalOp_expandDim_ref_of_regs .nat [BN] ⟨0, hax0⟩ "cols" s _ hcols,
    evalOp_expandDim_ref_of_regs .bool [BN] ⟨0, hax0⟩ "n_mask" s _ hnm]
  simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp only [Tile.scalar_data_index]
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`v = tl.load(v_ptrs + cols[:,None]·stride_vn, mask=n_mask[:,None], other=0.0)`
(A6/B5)**: the masked V-tile load. `v_ptrs` is the `[1,BLOCK_N]` pointer tile broadcast
across the `BLOCK_M` query rows; the per-row offset is `cols·stride_vn` (with `cols`
on the row axis, `[:,None]`), masked by `n_mask[:,None]`. Symbolic result kept for the
assembly layer. With `stride_vn = SVN`. -/
theorem msa_load_v_eval (s : BlockState) (BN BD SVN : Nat)
    (hax1 : 1 < [BN].length.succ)
    (vp : Tile .ptr [1, BD]) (ct : Tile .nat [BN]) (nm : Tile .bool [BN])
    (hv : s.regs .ptr [1, BD] "v_ptrs" = some vp)
    (hcols : s.regs .nat [BN] "cols" = some ct)
    (hnm : s.regs .bool [BN] "n_mask" = some nm) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consR.consL (Op.ref .ptr [1, BD] "v_ptrs")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, hax1⟩ (Op.ref .nat [BN] "cols")) (Op.constNat SVN))))
        (MaskOpt.maskOther
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.expandDim ⟨1, hax1⟩ (Op.ref .bool [BN] "n_mask")))
          ((Op.const 0.0).broadcast [BN, BD]))) s
      = some (⟨fun idx : TileIndex [BN, BD] =>
          let ptrs := Tile.ptrAdd Broadcast.nil.consR.consL vp
            (Tile.bop NumericDType.nat.mul Broadcast.scalarR
              (Tile.expandDim ⟨1, hax1⟩ ct) (Tile.scalar SVN))
          if (Tile.remap Broadcast.nil.consL.consSame.leftIndex
                (Tile.expandDim ⟨1, hax1⟩ nm)).data idx then
            s.readMemValue .real (ptrs.data idx).1 (ptrs.data idx).2
          else (some (0.0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BN, BD]) := by
  simp only [evalOp, hv]
  erw [evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, hax1⟩ "cols" s _ hcols,
    evalOp_expandDim_ref_of_regs .bool [BN] ⟨1, hax1⟩ "n_mask" s _ hnm]
  simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp only [Tile.scalar_data_index]
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **`cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:,None], other=0)`
(B3, loop B ONLY)**: the masked `.nat` gather of the visited sparse columns from
`column_index`. `cols_ptr` is a scalar pointer `(cpReg, cpOff)`; lane `j` reads
`column_index[cpOff + start_n + j]` when the loop guard `cond` holds, else the `0`
default. Given `cols_ptr = (cpReg, cpOff)`, `start_n = SN`, `offs_n = id`,
`cond = cb`. The block-sparse loop reads its columns contiguously instead (loop A
has no analogous gather; it uses `cols = start_n + offs_n`). -/
theorem msa_colgather_eval (s : BlockState) (BN cpOff SN : Nat) (cb : Bool)
    (cpReg : RegionName) (hax0 : 0 < ([] : TileShape).length.succ)
    (hcp : s.regs .ptr [] "cols_ptr" = some (Tile.scalar (cpReg, cpOff)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hon : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hcond : s.regs .bool [] "cond" = some (Tile.scalar cb)) :
    evalOp (Op.load .nat
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL
            (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "cols_ptr") (Op.ref .nat [] "start_n"))
            (Op.ref .nat [BN] "offs_n")))
        (MaskOpt.maskOther
          (Op.remap [BN] Broadcast.nil.consL.leftIndex
            (Op.expandDim ⟨0, hax0⟩ (Op.ref .bool [] "cond")))
          ((Op.constNat 0).broadcast [BN]))) s
      = some (⟨fun idx : TileIndex [BN] =>
          let ptrs := Tile.ptrAdd Broadcast.scalarL
            (Tile.ptrAdd Broadcast.nil (Tile.scalar (cpReg, cpOff)) (Tile.scalar SN))
            (Tile.vec (fun j : Fin BN => j.val))
          if (Tile.remap Broadcast.nil.consL.leftIndex
                (Tile.expandDim ⟨0, hax0⟩ (Tile.scalar cb : Tile .bool []))).data idx then
            s.readMemValue .nat (ptrs.data idx).1 (ptrs.data idx).2
          else (0 : Nat)⟩ : Tile .nat [BN]) := by
  simp only [evalOp, hcp, hsn, hon]
  erw [evalOp_expandDim_ref_of_regs .bool [] ⟨0, hax0⟩ "cond" s _ hcond]
  simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def]
  refine congrArg some ?_; ext idx
  simp only [Tile.scalar_data_index]
  rfl

end MSARecipes

/-! ## FOUNDATION: body-split, the two loop bodies, the loop invariants, and preLoop

This section is the execution-layer FOUNDATION for the (NEXT-stage)
step/assembly/top proof of `mixed_sparse_attention_fwd_kernel_surface` at the
Python `block64` shape. It mirrors `context_attn_nopad`'s `nopadPreLoop` /
`nopadLoopBody` / `nopadPostLoop` / `nopad_body_split` / `nopadInvariant` /
`nopadPreLoop_eval` scaffolding, adapted to this kernel's TWO sequential
`forRangeDyn` loops nested inside the `start_m·BLOCK_M ≥ seqlen` early-exit
`ifThen` guard.

**Extracted body structure** (via `rfl`-checked probes at the `block64` shape):
the lowered `toAlgKernel.body` is `[start_m, off_hz, seqlen, ifThen negGuard inner]`
(4 top statements; `negGuard = boolNot (start_m·64 ≥ seqlen)`), and `inner`
(26 statements) is

```
  inner = msaSetup (21 stmts: offs_m … max_num_blks)
            ++ [forRangeDyn "block_index" 0 (ref max_num_blks) 1 msaLoopBodyA]   -- 18-stmt body
            ++ [max_num_cols = 16]
            ++ [forRangeDyn "start_n" 0 (ref max_num_cols) 64 msaLoopBodyB]      -- 16-stmt body
            ++ msaPostLoop (2 stmts: acc /= l_i; masked store)
```

**FA2 normalization note.** Unlike `block_sparse_attn` (whose loop body folds in
`p_scale = beta/l_i_new` and `acc_scale = (l_i/l_i_new)·alpha`, so its `acc`
register holds the *normalized* `OPartial/LPartial` ratio at every step and
`bsaInvariant` records that ratio), THIS kernel uses the classic *unnormalized*
FlashAttention-2 form: `acc_scale = l_i·0 + alpha = alpha`, `acc *= alpha`,
`acc += dot(p, v)`, `l_i = l_i·alpha + sum(p)`, and the single division
`acc /= l_i` happens once in `msaPostLoop`. Hence the in-loop `acc` register is
the *unnormalized* numerator `OPartial`, `l_i` is the denominator `LPartial`, and
`m_i` is the running max `MPartial`; the invariants below record those
unnormalized streaming accumulators (the post-loop divide is what produces the
final closed-form ratio, exactly as `mixedSparseAttnClosedForm = numer/denom`). -/

section MSAFoundation

open VeriTile

/-! ### Streaming online-softmax accumulators (unnormalized, base-2)

The two loops share one online-softmax core (recipes A7,A10–A18 ≡ B6–B16). We
model it by a single generic exp2 fold parametric over a per-iteration *masked
score* row `score k i j : WithBot ℝ` (the `qk` lane value after the loop's
`where` mask; masked-out lanes carry `⊥`) and a per-iteration *value tile*
`vblk k j d : ℝ`. `BM`/`BN`/`BD = 64` at the block64 shape. The fold is the
*unnormalized* FA2 form: `acc` is the numerator, `l_i` the denominator, `m_i` the
running max — the postLoop `acc /= l_i` performs the single normalization.

`score`/`vblk` are supplied by the per-loop recipes (block-A: contiguous causal
keys; column-B: gathered columns), so the invariants below are stated against
opaque `score`/`vblk` carriers exactly as `block_sparse_attn`'s `bsaInvariant`
parametrizes by its gathered `Kg`/`Vg`/`gpos` stream. -/

/-- Running per-row max over the first `k` iterations (base-2 / `⊥`-seeded). -/
noncomputable def msaMPartial (BM BN : Nat)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) : Nat → Fin BM → WithBot ℝ
  | 0, _ => ⊥
  | k + 1, i =>
      max (msaMPartial BM BN score k i)
        ((Finset.univ : Finset (Fin BN)).sup fun j => score k i j)

/-- Running denominator (`l_i`): rescale prior by `alpha = exp2(mOld - mNew)` then
add this iteration's `Σ_j exp2(score - mNew)`. -/
noncomputable def msaLPartial (BM BN : Nat)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) : Nat → Fin BM → ℝ
  | 0, _ => 0
  | k + 1, i =>
      let mNew := msaMPartial BM BN score (k + 1) i
      let alpha := (WithBot.realExp2
        (Option.map₂ (fun x y : ℝ => x - y) (msaMPartial BM BN score k i) mNew)).unbotD 0
      alpha * msaLPartial BM BN score k i +
        (Finset.univ : Finset (Fin BN)).sum (fun j =>
          (WithBot.realExp2
            (Option.map₂ (fun x y : ℝ => x - y) (score k i j) mNew)).unbotD 0)

/-- Running unnormalized numerator (`acc`): rescale prior by `alpha` then add this
iteration's `Σ_j exp2(score - mNew)·vblk`. -/
noncomputable def msaOPartial (BM BN BD : Nat)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) (vblk : Nat → Fin BN → Fin BD → ℝ) :
    Nat → Fin BM → Fin BD → ℝ
  | 0, _, _ => 0
  | k + 1, i, d =>
      let mNew := msaMPartial BM BN score (k + 1) i
      let alpha := (WithBot.realExp2
        (Option.map₂ (fun x y : ℝ => x - y) (msaMPartial BM BN score k i) mNew)).unbotD 0
      alpha * msaOPartial BM BN BD score vblk k i d +
        (Finset.univ : Finset (Fin BN)).sum (fun j =>
          (WithBot.realExp2
            (Option.map₂ (fun x y : ℝ => x - y) (score k i j) mNew)).unbotD 0 * vblk k j d)

/-- Loop-B running max, **seeded** by Loop A's final max `mA` (so iteration 0 is
`mA`, not `⊥`). -/
noncomputable def msaSeedMax (BM BN : Nat) (mA : Fin BM → WithBot ℝ)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) : Nat → Fin BM → WithBot ℝ
  | 0, i => mA i
  | k + 1, i =>
      max (msaSeedMax BM BN mA score k i)
        ((Finset.univ : Finset (Fin BN)).sup fun j => score k i j)

/-- Loop-B running denominator, seeded by Loop A's final `lA` and max `mA`. -/
noncomputable def msaLPartialSeed (BM BN : Nat) (mA : Fin BM → WithBot ℝ) (lA : Fin BM → ℝ)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) : Nat → Fin BM → ℝ
  | 0, i => lA i
  | k + 1, i =>
      let mNew := msaSeedMax BM BN mA score (k + 1) i
      let alpha := (WithBot.realExp2
        (Option.map₂ (fun x y : ℝ => x - y) (msaSeedMax BM BN mA score k i) mNew)).unbotD 0
      alpha * msaLPartialSeed BM BN mA lA score k i +
        (Finset.univ : Finset (Fin BN)).sum (fun j =>
          (WithBot.realExp2
            (Option.map₂ (fun x y : ℝ => x - y) (score k i j) mNew)).unbotD 0)

/-- Loop-B running numerator, seeded by Loop A's final `oA`, `lA`, and max `mA`. -/
noncomputable def msaOPartialSeed (BM BN BD : Nat) (mA : Fin BM → WithBot ℝ) (lA : Fin BM → ℝ)
    (oA : Fin BM → Fin BD → ℝ)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) (vblk : Nat → Fin BN → Fin BD → ℝ) :
    Nat → Fin BM → Fin BD → ℝ
  | 0, i, d => oA i d
  | k + 1, i, d =>
      let mNew := msaSeedMax BM BN mA score (k + 1) i
      let alpha := (WithBot.realExp2
        (Option.map₂ (fun x y : ℝ => x - y) (msaSeedMax BM BN mA score k i) mNew)).unbotD 0
      alpha * msaOPartialSeed BM BN BD mA lA oA score vblk k i d +
        (Finset.univ : Finset (Fin BN)).sum (fun j =>
          (WithBot.realExp2
            (Option.map₂ (fun x y : ℝ => x - y) (score k i j) mNew)).unbotD 0 * vblk k j d)

@[simp] theorem msaMPartial_zero (BM BN : Nat) (score) (i : Fin BM) :
    msaMPartial BM BN score 0 i = ⊥ := rfl

@[simp] theorem msaLPartial_zero (BM BN : Nat) (score) (i : Fin BM) :
    msaLPartial BM BN score 0 i = 0 := rfl

@[simp] theorem msaOPartial_zero (BM BN BD : Nat) (score) (vblk) (i : Fin BM) (d : Fin BD) :
    msaOPartial BM BN BD score vblk 0 i d = 0 := rfl

/-! ### Online-softmax fold collapse (the genuine two-phase closed form)

The streaming `msaMPartial`/`msaLPartial`/`msaOPartial` recurrences are the classic
unnormalized FlashAttention-2 fold. We prove they collapse to the *direct*
masked-softmax weighted sums: writing `msaE x = exp2(x)` (with `exp2(⊥) = 0`), the
per-iteration max shift cancels, so after `k` iterations the running max-shifted
denominator/numerator equal the unshifted direct sums times `exp2(-m_k)`. The
post-loop divide `acc /= l_i` then yields the genuine ratio. -/

/-- `msaE x = exp2(x)` with `exp2(⊥) = 0` — the per-key softmax weight carrier. -/
noncomputable def msaE (x : WithBot ℝ) : ℝ := (WithBot.realExp2 x).unbotD 0

@[simp] theorem msaE_bot : msaE ⊥ = 0 := rfl
@[simp] theorem msaE_some (r : ℝ) : msaE (some r) = Real.exp (r * Real.log 2) := rfl

/-- **Max-shift cancellation.** If `M ≥ x` in `WithBot ℝ` (so `M = ⊥ → x = ⊥`),
then `exp2(M) · exp2(x - M) = exp2(x)`. This is the algebraic identity that makes
the online-softmax max subtraction telescope away. -/
theorem msaE_shift_cancel (x M : WithBot ℝ) (hle : x ≤ M) :
    msaE M * msaE (Option.map₂ (fun a b : ℝ => a - b) x M) = msaE x := by
  cases x with
  | bot =>
    have : Option.map₂ (fun a b : ℝ => a - b) (⊥ : WithBot ℝ) M = (⊥ : WithBot ℝ) := rfl
    rw [this]; simp [msaE_bot]
  | coe a =>
    cases M with
    | bot => exact absurd hle (by simp)
    | coe b =>
      show msaE (some b) * msaE (Option.map₂ (fun a b : ℝ => a - b) (some a) (some b)) = msaE (some a)
      rw [show Option.map₂ (fun a b : ℝ => a - b) (some a) (some b) = some (a - b) from rfl]
      simp only [msaE_some]
      rw [← Real.exp_add]; ring_nf

/-- Direct (unshifted) running denominator: `Σ_{l<k} Σ_j exp2(score l i j)`. -/
noncomputable def msaDenomUpto (BM BN : Nat)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) (k : Nat) (i : Fin BM) : ℝ :=
  (Finset.range k).sum (fun l =>
    (Finset.univ : Finset (Fin BN)).sum (fun j => msaE (score l i j)))

/-- Direct (unshifted) running numerator: `Σ_{l<k} Σ_j exp2(score)·v`. -/
noncomputable def msaNumerUpto (BM BN BD : Nat)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) (vblk : Nat → Fin BN → Fin BD → ℝ)
    (k : Nat) (i : Fin BM) (d : Fin BD) : ℝ :=
  (Finset.range k).sum (fun l =>
    (Finset.univ : Finset (Fin BN)).sum (fun j => msaE (score l i j) * vblk l j d))

/-- The running max dominates the seed and every streamed score (monotone fold). -/
theorem msaMPartial_ge_self (BM BN : Nat) (score) (k : Nat) (i : Fin BM) :
    msaMPartial BM BN score k i ≤ msaMPartial BM BN score (k + 1) i := by
  simp only [msaMPartial]; exact le_max_left _ _

theorem msaMPartial_ge_score (BM BN : Nat) (score) (k : Nat) (i : Fin BM) (j : Fin BN) :
    score k i j ≤ msaMPartial BM BN score (k + 1) i := by
  simp only [msaMPartial]
  exact le_trans (Finset.le_sup (Finset.mem_univ j)) (le_max_right _ _)

/-- **Collapse (denominator).** After `k` iterations the max-shifted denominator
`msaLPartial` equals `exp2(-m_k) · msaDenomUpto`, i.e.
`exp2(m_k) · msaLPartial k = msaDenomUpto k`. -/
theorem msaLPartial_collapse (BM BN : Nat) (score) (k : Nat) (i : Fin BM) :
    msaE (msaMPartial BM BN score k i) * msaLPartial BM BN score k i
      = msaDenomUpto BM BN score k i := by
  induction k with
  | zero => simp [msaLPartial, msaDenomUpto]
  | succ k ih =>
    rw [msaDenomUpto, Finset.sum_range_succ, ← msaDenomUpto, ← ih]
    set mNew := msaMPartial BM BN score (k + 1) i with hmNew
    show msaE mNew *
        ((msaE (Option.map₂ (fun x y : ℝ => x - y) (msaMPartial BM BN score k i) mNew))
            * msaLPartial BM BN score k i
          + (Finset.univ : Finset (Fin BN)).sum (fun j =>
              msaE (Option.map₂ (fun x y : ℝ => x - y) (score k i j) mNew)))
      = msaE (msaMPartial BM BN score k i) * msaLPartial BM BN score k i
        + (Finset.univ : Finset (Fin BN)).sum (fun j => msaE (score k i j))
    rw [mul_add, ← mul_assoc, msaE_shift_cancel _ _ (msaMPartial_ge_self BM BN score k i),
      Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl; intro j _
    rw [msaE_shift_cancel _ _ (msaMPartial_ge_score BM BN score k i j)]

/-- **Collapse (numerator).** `exp2(m_k) · msaOPartial k = msaNumerUpto k`. -/
theorem msaOPartial_collapse (BM BN BD : Nat) (score) (vblk) (k : Nat) (i : Fin BM) (d : Fin BD) :
    msaE (msaMPartial BM BN score k i) * msaOPartial BM BN BD score vblk k i d
      = msaNumerUpto BM BN BD score vblk k i d := by
  induction k with
  | zero => simp [msaOPartial, msaNumerUpto]
  | succ k ih =>
    rw [msaNumerUpto, Finset.sum_range_succ, ← msaNumerUpto, ← ih]
    set mNew := msaMPartial BM BN score (k + 1) i with hmNew
    show msaE mNew *
        ((msaE (Option.map₂ (fun x y : ℝ => x - y) (msaMPartial BM BN score k i) mNew))
            * msaOPartial BM BN BD score vblk k i d
          + (Finset.univ : Finset (Fin BN)).sum (fun j =>
              msaE (Option.map₂ (fun x y : ℝ => x - y) (score k i j) mNew) * vblk k j d))
      = msaE (msaMPartial BM BN score k i) * msaOPartial BM BN BD score vblk k i d
        + (Finset.univ : Finset (Fin BN)).sum (fun j => msaE (score k i j) * vblk k j d)
    rw [mul_add, ← mul_assoc, msaE_shift_cancel _ _ (msaMPartial_ge_self BM BN score k i),
      Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl; intro j _
    rw [← mul_assoc, msaE_shift_cancel _ _ (msaMPartial_ge_score BM BN score k i j)]

/-- **Ratio collapse.** When the direct denominator is positive, the normalized
online-softmax output `msaOPartial / msaLPartial` equals the direct softmax ratio
`msaNumerUpto / msaDenomUpto` — the max shift cancels in the quotient. -/
theorem msaPartial_ratio_collapse (BM BN BD : Nat) (score) (vblk) (k : Nat)
    (i : Fin BM) (d : Fin BD)
    (hpos : 0 < msaDenomUpto BM BN score k i) :
    msaOPartial BM BN BD score vblk k i d / msaLPartial BM BN score k i
      = msaNumerUpto BM BN BD score vblk k i d / msaDenomUpto BM BN score k i := by
  have hL := msaLPartial_collapse BM BN score k i
  have hO := msaOPartial_collapse BM BN BD score vblk k i d
  -- exp2(m_k) > 0 since msaDenomUpto > 0 forces m_k ≠ ⊥
  have hEpos : 0 < msaE (msaMPartial BM BN score k i) := by
    cases hm : msaMPartial BM BN score k i with
    | bot =>
      exfalso
      rw [hm] at hL; simp only [msaE_bot, zero_mul] at hL
      rw [← hL] at hpos; exact lt_irrefl 0 hpos
    | coe r => show 0 < msaE (some r); rw [msaE_some]; exact Real.exp_pos _
  have hLne : msaLPartial BM BN score k i ≠ 0 := by
    intro h; rw [h, mul_zero] at hL; rw [← hL] at hpos; exact lt_irrefl 0 hpos
  rw [← hL, ← hO]
  field_simp

/-! ### Two-phase concatenation: seeded Loop-B fold = single fold over A ++ B

Loop B does not re-seed from `⊥`/`0`; it continues Loop A's accumulators. We show
the seeded partials (`msaSeedMax`/`msaLPartialSeed`/`msaOPartialSeed`) over `c`
column-iterations, seeded by Loop A's `bF`-block finals, equal the plain
(`⊥`-seeded) partials over the *concatenated* stream `catScore bF scoreA scoreB`
run for `bF + c` iterations. This is the genuine two-phase mixed-sparsity fold. -/

/-- Prefix agreement: partials depend only on the streamed prefix `[0, k)`. -/
theorem msaMPartial_congr_prefix (BM BN : Nat) (s1 s2) (k : Nat) (i : Fin BM)
    (h : ∀ l, l < k → ∀ j : Fin BN, s1 l i j = s2 l i j) :
    msaMPartial BM BN s1 k i = msaMPartial BM BN s2 k i := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [msaMPartial, msaMPartial, ih (fun l hl => h l (by omega))]
    congr 1
    apply Finset.sup_congr rfl; intro j _; exact h k (by omega) j

theorem msaLPartial_congr_prefix (BM BN : Nat) (s1 s2) (k : Nat) (i : Fin BM)
    (h : ∀ l, l < k → ∀ j : Fin BN, s1 l i j = s2 l i j) :
    msaLPartial BM BN s1 k i = msaLPartial BM BN s2 k i := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [msaLPartial, msaLPartial, ih (fun l hl => h l (by omega))]
    simp only [msaMPartial_congr_prefix BM BN s1 s2 (k+1) i h,
      msaMPartial_congr_prefix BM BN s1 s2 k i (fun l hl => h l (by omega))]
    congr 1
    apply Finset.sum_congr rfl; intro j _; rw [h k (by omega) j]

theorem msaOPartial_congr_prefix (BM BN BD : Nat) (s1 s2) (v1 v2) (k : Nat) (i : Fin BM) (d : Fin BD)
    (h : ∀ l, l < k → ∀ j : Fin BN, s1 l i j = s2 l i j)
    (hv : ∀ l, l < k → ∀ j : Fin BN, v1 l j d = v2 l j d) :
    msaOPartial BM BN BD s1 v1 k i d = msaOPartial BM BN BD s2 v2 k i d := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [msaOPartial, msaOPartial, ih (fun l hl => h l (by omega)) (fun l hl => hv l (by omega))]
    simp only [msaMPartial_congr_prefix BM BN s1 s2 (k+1) i h,
      msaMPartial_congr_prefix BM BN s1 s2 k i (fun l hl => h l (by omega))]
    congr 1
    apply Finset.sum_congr rfl; intro j _; rw [h k (by omega) j, hv k (by omega) j]

/-- Concatenated score stream: first `bF` iterations from `scoreA`, then `scoreB`. -/
noncomputable def msaCatScore (BM BN bF : Nat)
    (scoreA scoreB : Nat → Fin BM → Fin BN → WithBot ℝ) :
    Nat → Fin BM → Fin BN → WithBot ℝ :=
  fun k => if k < bF then scoreA k else scoreB (k - bF)

/-- Concatenated value stream. -/
noncomputable def msaCatVblk (BN BD bF : Nat)
    (vblkA vblkB : Nat → Fin BN → Fin BD → ℝ) : Nat → Fin BN → Fin BD → ℝ :=
  fun k => if k < bF then vblkA k else vblkB (k - bF)

theorem msaSeedMax_eq_cat (BM BN bF : Nat) (scoreA scoreB) (c : Nat) (i : Fin BM) :
    msaSeedMax BM BN (msaMPartial BM BN scoreA bF) scoreB c i
      = msaMPartial BM BN (msaCatScore BM BN bF scoreA scoreB) (bF + c) i := by
  have hpre : ∀ l, l < bF → ∀ j : Fin BN, scoreA l i j = msaCatScore BM BN bF scoreA scoreB l i j := by
    intro l hl j; rw [msaCatScore, if_pos hl]
  induction c with
  | zero =>
    simp only [msaSeedMax, Nat.add_zero]; exact msaMPartial_congr_prefix BM BN scoreA _ bF i hpre
  | succ c ih =>
    rw [msaSeedMax, ih, show bF + (c + 1) = (bF + c) + 1 from by ring, msaMPartial]
    congr 1
    apply Finset.sup_congr rfl; intro j _
    rw [msaCatScore, if_neg (by omega), show bF + c - bF = c from by omega]

theorem msaLPartialSeed_eq_cat (BM BN bF : Nat) (scoreA scoreB) (lA0 : Fin BM → ℝ) (c : Nat) (i : Fin BM)
    (hlA : lA0 = msaLPartial BM BN scoreA bF) :
    msaLPartialSeed BM BN (msaMPartial BM BN scoreA bF) lA0 scoreB c i
      = msaLPartial BM BN (msaCatScore BM BN bF scoreA scoreB) (bF + c) i := by
  have hpre : ∀ l, l < bF → ∀ j : Fin BN, scoreA l i j = msaCatScore BM BN bF scoreA scoreB l i j := by
    intro l hl j; rw [msaCatScore, if_pos hl]
  induction c with
  | zero =>
    rw [hlA]; simp only [msaLPartialSeed, Nat.add_zero]
    exact msaLPartial_congr_prefix BM BN scoreA _ bF i hpre
  | succ c ih =>
    rw [msaLPartialSeed, ih, show bF + (c + 1) = (bF + c) + 1 from by ring, msaLPartial]
    simp only [msaSeedMax_eq_cat, show bF + (c + 1) = (bF + c) + 1 from by ring]
    have hcat : ∀ j : Fin BN, msaCatScore BM BN bF scoreA scoreB (bF + c) i j = scoreB c i j := by
      intro j; rw [msaCatScore, if_neg (by omega), show bF + c - bF = c from by omega]
    congr 1
    apply Finset.sum_congr rfl; intro j _
    rw [hcat j]

theorem msaOPartialSeed_eq_cat (BM BN BD bF : Nat) (scoreA scoreB) (vblkA vblkB)
    (lA0 : Fin BM → ℝ) (oA0 : Fin BM → Fin BD → ℝ) (c : Nat) (i : Fin BM) (d : Fin BD)
    (hlA : lA0 = msaLPartial BM BN scoreA bF)
    (hoA : oA0 = fun i d => msaOPartial BM BN BD scoreA vblkA bF i d) :
    msaOPartialSeed BM BN BD (msaMPartial BM BN scoreA bF) lA0 oA0 scoreB vblkB c i d
      = msaOPartial BM BN BD (msaCatScore BM BN bF scoreA scoreB)
          (msaCatVblk BN BD bF vblkA vblkB) (bF + c) i d := by
  have hpre : ∀ l, l < bF → ∀ j : Fin BN, scoreA l i j = msaCatScore BM BN bF scoreA scoreB l i j := by
    intro l hl j; rw [msaCatScore, if_pos hl]
  have hprev : ∀ l, l < bF → ∀ j : Fin BN, vblkA l j d = msaCatVblk BN BD bF vblkA vblkB l j d := by
    intro l hl j; rw [msaCatVblk, if_pos hl]
  induction c with
  | zero =>
    rw [hoA]; simp only [msaOPartialSeed, Nat.add_zero]
    exact msaOPartial_congr_prefix BM BN BD scoreA _ vblkA _ bF i d hpre hprev
  | succ c ih =>
    rw [msaOPartialSeed, ih, show bF + (c + 1) = (bF + c) + 1 from by ring, msaOPartial]
    simp only [msaSeedMax_eq_cat, show bF + (c + 1) = (bF + c) + 1 from by ring]
    have hcat : ∀ j : Fin BN, msaCatScore BM BN bF scoreA scoreB (bF + c) i j = scoreB c i j := by
      intro j; rw [msaCatScore, if_neg (by omega), show bF + c - bF = c from by omega]
    have hcatv : ∀ j : Fin BN, msaCatVblk BN BD bF vblkA vblkB (bF + c) j d = vblkB c j d := by
      intro j; rw [msaCatVblk, if_neg (by omega), show bF + c - bF = c from by omega]
    congr 1
    apply Finset.sum_congr rfl; intro j _
    rw [hcat j, hcatv j]

/-- Masked `fp16` scatter at distinct offsets: `readMemValue .fp16` reads the
written `fp16` cell at an active lane, and the prior cell elsewhere. fp16 analogue
of `scatter_readback_nat_prop_masked_nd` (fp16 `storeValue`/`ofReal` are identity in
the model, so the round-tripped value is recovered exactly as `some r`). -/
theorem msa_fp16_scatter_readback {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier FloatDType.fp16.toTileDType)
    (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    BlockState.readMemValue ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemTyped FloatDType.fp16.toTileDType region (offsetFn k) (valueFn k)
         else acc) s)
      .fp16 region (offsetFn i)
    = if P i then FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i))
      else s.readMemValue .fp16 region (offsetFn i) := by
  have hpres : ∀ (l : List (TileIndex shape)) (sX : BlockState) (o : Nat),
      (∀ k ∈ l, P k → offsetFn k ≠ o) →
      BlockState.readMemValue ((l.foldl
          (fun acc k =>
            if P k then acc.writeMemTyped FloatDType.fp16.toTileDType region (offsetFn k) (valueFn k)
            else acc) sX)) .fp16 region o
        = sX.readMemValue .fp16 region o := by
    intro l
    induction l with
    | nil => intro sX o _; rfl
    | cons hd tl ih =>
      intro sX o h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, P k → offsetFn k ≠ o :=
        fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
      by_cases hPhd : P hd
      · have hhd : offsetFn hd ≠ o := h hd List.mem_cons_self hPhd
        simp only [hPhd, if_true]
        rw [ih _ o htl]
        show (sX.writeMemAs .fp16 region (offsetFn hd) (valueFn hd)).readMemAs .fp16 region o
          = sX.readMemAs .fp16 region o
        rw [BlockState.writeMemAs_readMemAs, if_neg (by rintro ⟨_, h_eq⟩; exact hhd h_eq.symm)]
      · simp only [hPhd, if_false]
        exact ih _ o htl
  let l := TileShape.allIndices shape
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  have hl' : l = l₁ ++ i :: l₂ := by simpa [l] using hl
  show BlockState.readMemValue ((l.foldl
       (fun acc k =>
         if P k then acc.writeMemTyped FloatDType.fp16.toTileDType region (offsetFn k) (valueFn k)
         else acc) s)) .fp16 region (offsetFn i) = _
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  rw [hl', List.foldl_append, List.foldl_cons]
  have h_l2_not_in : ∀ k ∈ l₂, P k → offsetFn k ≠ offsetFn i := by
    intro k hk _ heq; have hki : k = i := h_inj heq; subst hki; exact hi_notin_l2 hk
  rw [hpres l₂ _ (offsetFn i) h_l2_not_in]
  have h_l1_not_in : ∀ k ∈ l₁, P k → offsetFn k ≠ offsetFn i := by
    intro k hk _ heq; have hki : k = i := h_inj heq; rw [hki] at hk
    exact (hl1_disj i hk i List.mem_cons_self) rfl
  by_cases hPi : P i
  · simp only [hPi, if_true, if_pos hPi]
    rw [show ∀ sX : BlockState, BlockState.readMemValue
          (sX.writeMemTyped FloatDType.fp16.toTileDType region (offsetFn i) (valueFn i)) .fp16
          region (offsetFn i)
        = FloatDType.fp16.ofReal (FloatDType.fp16.storeValue (valueFn i)) from fun sX => by
      show (sX.writeMemAs .fp16 region (offsetFn i) (valueFn i)).readMemAs .fp16 region (offsetFn i) = _
      rw [BlockState.writeMemAs_readMemAs, if_pos ⟨rfl, rfl⟩]]
  · simp only [hPi, if_false, if_neg hPi]
    rw [hpres l₁ _ (offsetFn i) h_l1_not_in]

/-- `if cond { body }` step (true branch): when `cond` evaluates to scalar `true`,
the `ifThen` runs `body`. -/
theorem stepStmt_ifThen_true {cond : Op .bool []}
    {body : List Stmt} {s s' : BlockState}
    (hcond : evalOp cond s = some (Tile.scalar (Bool.true)))
    (hbody : stepStmts body s = some s') :
    stepStmt (.ifThen cond body) s = some s' := by
  simp only [stepStmt, hcond, Option.bind_some, Tile.scalar_data, if_true]
  exact hbody

end MSAFoundation

open VeriTile

/-! ## FULLY-GENERAL (symbolic strides + layout) AST + body split -/

def msaGuardGS (BM : Nat) : Op .bool [] :=
  Op.ge ComparableDType.nat Broadcast.nil
    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BM))
    (Op.ref .nat [] "seqlen")

/-- Symbolic setup statements (block dims `BM`/`BN`/`BD`; strides pinned). -/
def msaSetupGS (Q K V : RegionName) (seqlens : Region .nat)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName) (BM BN BD : Nat)
    (H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV : Nat)
    (sm_scale : ℝ := 0.1) : List Stmt :=
  [ Stmt.assign .nat [BM] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BM))
        (Op.arange BM)),
    Stmt.assign .nat [BN] "offs_n" (Op.arange BN),
    Stmt.assign .nat [BD] "offs_d" (Op.arange BD),
    Stmt.assign .nat [] "qo_offset"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H))
          (Op.constNat sqz))
        (Op.mul .nat Broadcast.nil
          (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H))
          (Op.constNat sqh))),
    Stmt.assign .nat [] "kv_offset"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H))
          (Op.constNat skz))
        (Op.mul .nat Broadcast.nil
          (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H))
          (Op.constNat skh))),
    Stmt.assign .ptr [BM, BD] "q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qo_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat sqm)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat sqk)))),
    Stmt.assign .ptr [BD, 1] "k_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "kv_offset")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat skk)))),
    Stmt.assign .ptr [1, BD] "v_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "kv_offset")
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat svk)))),
    Stmt.assign .ptr [BM, BD] "o_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add .nat Broadcast.nil.consL.consR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qo_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat som)))
          (Op.mul .nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat sok)))),
    Stmt.assign .nat [] "num_blks"
      (Op.load .nat
        (MemAccess.region block_count
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat NR))
            (Op.ref .nat [] "start_m")))
        MaskOpt.none),
    Stmt.assign .ptr [] "blks_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase block_offset)
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat NR))
            (Op.ref .nat [] "start_m"))
          (Op.constNat NS))),
    Stmt.assign .nat [] "num_cols"
      (Op.load .nat
        (MemAccess.region column_count
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat NR))
            (Op.ref .nat [] "start_m")))
        MaskOpt.none),
    Stmt.assign .ptr [] "cols_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase column_index)
        (Op.mul .nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat NR))
            (Op.ref .nat [] "start_m"))
          (Op.constNat NV))),
    Stmt.assign .real [BM] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [BM] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BM] "l_i" (Op.full [BM] (Op.const 0)),
    Stmt.assign .real [BM, BD] "acc" (Op.full [BM, BD] (Op.const 0)),
    Stmt.assign .real [] "qk_scale"
      (Op.mul .real Broadcast.nil (Op.const (sm_scale : ℝ)) (Op.const 1.44269504)),
    Stmt.assign .real [BM, BD] "q"
      (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM, BD] "q_ptrs")) MaskOpt.none),
    Stmt.assign FloatDType.fp16.toTileDType [BM, BD] "q"
      (Op.castFloat FloatDType.real FloatDType.fp16
        (Op.mul .real Broadcast.scalarR
          (Op.ref .real [BM, BD] "q")
          (Op.ref .real [] "qk_scale"))),
    Stmt.assign .bool [BM, 1] "m_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.ref .nat [] "seqlen")),
    Stmt.assign .nat [] "max_num_blks" (Op.constNat 8) ]

/-- Symbolic Loop-A body. -/
def msaLoopBodyAGS (BM BN BD skn svn : Nat) : List Stmt :=
  [ Stmt.assign .bool [] "cond"
      (Op.lt ComparableDType.nat Broadcast.nil
        (Op.ref .nat [] "block_index") (Op.ref .nat [] "num_blks")),
    Stmt.assign .nat [] "start_n"
      (Op.load .nat
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "blks_ptr") (Op.ref .nat [] "block_index")))
        (MaskOpt.mask (Op.ref .bool [] "cond"))),
    Stmt.assign .nat [BN] "cols"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n")),
    Stmt.assign .bool [BN] "n_mask"
      (Op.boolAnd Broadcast.scalarR
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.ref .nat [BN] "cols") (Op.ref .nat [] "seqlen"))
        (Op.ref .bool [] "cond")),
    Stmt.assign .real [BD, BN] "k"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consL.consR (Op.ref .ptr [BD, 1] "k_ptrs")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "cols")) (Op.constNat skn))))
        (MaskOpt.maskOther
          (Op.remap [BD, BN] Broadcast.nil.consSame.consL.leftIndex
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .bool [BN] "n_mask")))
          ((Op.const 0.0).broadcast [BD, BN]))),
    Stmt.assign .real [BN, BD] "v"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consR.consL (Op.ref .ptr [1, BD] "v_ptrs")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "cols")) (Op.constNat svn))))
        (MaskOpt.maskOther
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .bool [BN] "n_mask")))
          ((Op.const 0.0).broadcast [BN, BD]))),
    Stmt.assign .real [BM, BN] "qk" (Op.full [BM, BN] (Op.const 0)),
    Stmt.assign .bool [BM, BN] "causal_mask"
      (Op.le ComparableDType.nat Broadcast.nil.consR.consL
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "cols"))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m"))),
    Stmt.assign .real [BM, BN] "qk"
      ((Op.boolAnd Broadcast.nil.consL.consSame
          (Op.ref .bool [BM, 1] "m_mask") (Op.ref .bool [BM, BN] "causal_mask")).where
        (Op.ref .real [BM, BN] "qk") (Op.negInf.broadcast [BM, BN])),
    Stmt.assign .real [BM, BN] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := [])
          ((Op.castFloat FloatDType.fp16 FloatDType.real
            (Op.ref FloatDType.fp16.toTileDType [BM, BD] "q")) : Op .real [BM, BD])
          (Op.ref .real [BD, BN] "k"))),
    Stmt.assign .real [BM] "m_i_new"
      ((Op.gt ComparableDType.real Broadcast.nil.consSame
          (Op.ref .real [BM] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
            (Op.ref .real [BM, BN] "qk"))).where
        (Op.ref .real [BM] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
          (Op.ref .real [BM, BN] "qk"))),
    Stmt.assign .real [BM] "alpha"
      (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_i_new"))),
    Stmt.assign .real [BM, BN] "p"
      (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_i_new")))),
    Stmt.assign .real [BM] "acc_scale"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BM] "l_i") (Op.const 0))
        (Op.ref .real [BM] "alpha")),
    Stmt.assign .real [BM, BD] "acc"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "acc_scale"))),
    Stmt.assign .real [BM, BD] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.dot (batch := [])
          ((Op.castFloat FloatDType.fp16 FloatDType.real
            (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "p"))) : Op .real [BM, BN])
          (Op.ref .real [BN, BD] "v"))),
    Stmt.assign .real [BM] "l_i"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
          (Op.ref .real [BM, BN] "p"))),
    Stmt.assign .real [BM] "m_i" (Op.ref .real [BM] "m_i_new") ]

/-- Symbolic Loop-B body. -/
def msaLoopBodyBGS (BM BN BD skn svn : Nat) : List Stmt :=
  [ Stmt.assign .bool [] "cond"
      (Op.lt ComparableDType.nat Broadcast.nil
        (Op.ref .nat [] "start_n") (Op.ref .nat [] "num_cols")),
    Stmt.assign .bool [BN] "n_mask"
      (Op.boolAnd Broadcast.scalarR
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n") (Op.ref .nat [BN] "offs_n"))
          (Op.ref .nat [] "num_cols"))
        (Op.ref .bool [] "cond")),
    Stmt.assign .nat [BN] "cols"
      (Op.load .nat
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL
            (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "cols_ptr") (Op.ref .nat [] "start_n"))
            (Op.ref .nat [BN] "offs_n")))
        (MaskOpt.maskOther
          (Op.remap [BN] Broadcast.nil.consL.leftIndex
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .bool [] "cond")))
          ((Op.constNat 0).broadcast [BN]))),
    Stmt.assign .real [BD, BN] "k"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consL.consR (Op.ref .ptr [BD, 1] "k_ptrs")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "cols")) (Op.constNat skn))))
        (MaskOpt.maskOther
          (Op.remap [BD, BN] Broadcast.nil.consSame.consL.leftIndex
            (Op.expandDim ⟨0, by simp⟩ (Op.ref .bool [BN] "n_mask")))
          ((Op.const 0.0).broadcast [BD, BN]))),
    Stmt.assign .real [BN, BD] "v"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consR.consL (Op.ref .ptr [1, BD] "v_ptrs")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "cols")) (Op.constNat svn))))
        (MaskOpt.maskOther
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .bool [BN] "n_mask")))
          ((Op.const 0.0).broadcast [BN, BD]))),
    Stmt.assign .real [BM, BN] "qk" (Op.full [BM, BN] (Op.const 0)),
    Stmt.assign .real [BM, BN] "qk"
      ((Op.boolAnd Broadcast.nil.consL.leadR
          (Op.ref .bool [BM, 1] "m_mask") (Op.ref .bool [BN] "n_mask")).where
        (Op.ref .real [BM, BN] "qk") (Op.negInf.broadcast [BM, BN])),
    Stmt.assign .real [BM, BN] "qk"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BN] "qk")
        (Op.dot (batch := [])
          ((Op.castFloat FloatDType.fp16 FloatDType.real
            (Op.ref FloatDType.fp16.toTileDType [BM, BD] "q")) : Op .real [BM, BD])
          (Op.ref .real [BD, BN] "k"))),
    Stmt.assign .real [BM] "m_i_new"
      ((Op.gt ComparableDType.real Broadcast.nil.consSame
          (Op.ref .real [BM] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
            (Op.ref .real [BM, BN] "qk"))).where
        (Op.ref .real [BM] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
          (Op.ref .real [BM, BN] "qk"))),
    Stmt.assign .real [BM] "alpha"
      (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BM] "m_i") (Op.ref .real [BM] "m_i_new"))),
    Stmt.assign .real [BM, BN] "p"
      (Op.exp2 (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BN] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "m_i_new")))),
    Stmt.assign .real [BM] "acc_scale"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real Broadcast.scalarR (Op.ref .real [BM] "l_i") (Op.const 0))
        (Op.ref .real [BM] "alpha")),
    Stmt.assign .real [BM, BD] "acc"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "acc_scale"))),
    Stmt.assign .real [BM, BD] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.dot (batch := [])
          ((Op.castFloat FloatDType.fp16 FloatDType.real
            (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BN] "p"))) : Op .real [BM, BN])
          (Op.ref .real [BN, BD] "v"))),
    Stmt.assign .real [BM] "l_i"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BM] "l_i") (Op.ref .real [BM] "alpha"))
        (Op.reduceSum (⟨1, by simp⟩ : Fin [BM, BN].length) Bool.false
          (Op.ref .real [BM, BN] "p"))),
    Stmt.assign .real [BM] "m_i" (Op.ref .real [BM] "m_i_new") ]

/-- Symbolic post-loop. -/
def msaPostLoopGS (BM BD : Nat) : List Stmt :=
  [ Stmt.assign .real [BM, BD] "acc"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "l_i"))),
    Stmt.store FloatDType.fp16.toTileDType [BM, BD]
      (MemAccess.ptr (Op.ref .ptr [BM, BD] "o_ptrs"))
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BD] "acc"))
      (MaskOpt.mask
        (Op.remap [BM, BD] Broadcast.nil.consL.consSame.leftIndex
          (Op.ref .bool [BM, 1] "m_mask"))) ]

set_option maxRecDepth 8000 in
/-- **General body split** (symbolic block dims). -/
theorem msa_body_splitGS
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (BM BN BD : Nat)
    (H NCTX Zc NR NS NV : Nat)
    (sqz sqh sqm sqk skz skh skn skk svz svh svn svk soz soh som sok : Nat)
    (sm_scale : ℝ := 0.1) :
    (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
      (sm_scale : ℝ) Blocks BlockOffsets ColCounts Cols Out
      sqz sqh sqm sqk skz skh skn skk svz svh svn svk soz soh som sok
      Zc H NCTX NR NS NV BM BN BD FloatDType.fp16).toAlgKernel.body
      = [ Stmt.assign .nat [] "start_m" (Op.programId 0),
          Stmt.assign .nat [] "off_hz" (Op.programId 1),
          Stmt.assign .nat [] "seqlen"
            (Op.load .nat
              (MemAccess.region Seqlens
                (Op.floorDiv IntegralDType.nat Broadcast.nil
                  (Op.ref .nat [] "off_hz") (Op.constNat H)))
              MaskOpt.none),
          Stmt.ifThen (Op.boolNot (msaGuardGS BM))
            (msaSetupGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD
              H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV sm_scale
              ++ [ Stmt.forRangeDyn "block_index" (Op.constNat 0)
                    (Op.ref .nat [] "max_num_blks") (Op.constNat 1) (msaLoopBodyAGS BM BN BD skn svn),
                   Stmt.assign .nat [] "max_num_cols" (Op.constNat 16),
                   Stmt.forRangeDyn "start_n" (Op.constNat 0)
                    (Op.ref .nat [] "max_num_cols") (Op.constNat BN) (msaLoopBodyBGS BM BN BD skn svn) ]
              ++ msaPostLoopGS BM BD) ] := by
  rfl

/-! ## FULLY-GENERAL invariants + ptr/val defs (symbolic strides + layout) -/

noncomputable def msaInvariantAGS
    (Q K V : RegionName) (Seqlens : Region .nat)
    (Blocks BlockOffsets ColCounts Cols : Region .nat) (Out : RegionName)
    (BM BN BD H NR NS NV : Nat)
    (scoreA : Nat → Fin BM → Fin BN → WithBot ℝ) (vblkA : Nat → Fin BN → Fin BD → ℝ)
    (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (vpF : TileIndex [1, BD] → RegionName × Nat) (opF : TileIndex [BM, BD] → RegionName × Nat)
    (s0 : BlockState) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "seqlen" = some (Tile.scalar (seqLen s0 H (Region.cast Seqlens)))
  ∧ s.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => s0.pids 0 * BM + i.val))
  ∧ s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))
  ∧ s.regs .nat [BD] "offs_d" = some (Tile.vec (fun e : Fin BD => e.val))
  ∧ s.regs .nat [] "num_blks" =
      some (Tile.scalar (s0.readMemValue .nat (Region.cast Blocks)
        (s0.pids 1 * NR + s0.pids 0)))
  ∧ s.regs .nat [] "num_cols" =
      some (Tile.scalar (s0.readMemValue .nat (Region.cast ColCounts)
        (s0.pids 1 * NR + s0.pids 0)))
  ∧ s.regs .ptr [] "blks_ptr" =
      some (Tile.scalar (Region.cast BlockOffsets, (s0.pids 1 * NR + s0.pids 0) * NS))
  ∧ s.regs .ptr [] "cols_ptr" =
      some (Tile.scalar (Region.cast Cols, (s0.pids 1 * NR + s0.pids 0) * NV))
  ∧ s.regs FloatDType.fp16.toTileDType [BM, BD] "q" =
      some (⟨fun idx : TileIndex [BM, BD] =>
        FloatDType.real.cast FloatDType.fp16 (qF idx)⟩ : Tile FloatDType.fp16.toTileDType [BM, BD])
  ∧ s.regs .ptr [BD, 1] "k_ptrs" =
      some (⟨fun idx : TileIndex [BD, 1] => kpF idx⟩ : Tile .ptr [BD, 1])
  ∧ s.regs .ptr [1, BD] "v_ptrs" =
      some (⟨fun idx : TileIndex [1, BD] => vpF idx⟩ : Tile .ptr [1, BD])
  ∧ s.regs .ptr [BM, BD] "o_ptrs" =
      some (⟨fun idx : TileIndex [BM, BD] => opF idx⟩ : Tile .ptr [BM, BD])
  ∧ s.regs .bool [BM, 1] "m_mask" =
      some (⟨fun idx : TileIndex [BM, 1] =>
        decide (s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))⟩ : Tile .bool [BM, 1])
  ∧ s.regs .nat [] "max_num_blks" = some (Tile.scalar 8)
  ∧ s.regs .real [BM] "m_i" =
      some (⟨fun idx : TileIndex [BM] => msaMPartial BM BN scoreA c idx.1⟩ : Tile .real [BM])
  ∧ s.regs .real [BM] "l_i" =
      some (⟨fun idx : TileIndex [BM] =>
        (some (msaLPartial BM BN scoreA c idx.1) : WithBot ℝ)⟩ : Tile .real [BM])
  ∧ s.regs .real [BM, BD] "acc" =
      some (⟨fun idx : TileIndex [BM, BD] =>
        (some (msaOPartial BM BN BD scoreA vblkA c idx.1 idx.2.1) : WithBot ℝ)⟩
          : Tile .real [BM, BD])

/-- Symbolic Loop-B invariant. -/
noncomputable def msaInvariantBGS
    (Q K V : RegionName) (Seqlens : Region .nat)
    (Blocks BlockOffsets ColCounts Cols : Region .nat) (Out : RegionName)
    (BM BN BD H NR NS NV : Nat)
    (scoreB : Nat → Fin BM → Fin BN → WithBot ℝ) (vblkB : Nat → Fin BN → Fin BD → ℝ)
    (mA : Fin BM → WithBot ℝ) (lA : Fin BM → ℝ) (oA : Fin BM → Fin BD → ℝ)
    (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (vpF : TileIndex [1, BD] → RegionName × Nat) (opF : TileIndex [BM, BD] → RegionName × Nat)
    (s0 : BlockState) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "off_hz" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "seqlen" = some (Tile.scalar (seqLen s0 H (Region.cast Seqlens)))
  ∧ s.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => s0.pids 0 * BM + i.val))
  ∧ s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val))
  ∧ s.regs .nat [BD] "offs_d" = some (Tile.vec (fun e : Fin BD => e.val))
  ∧ s.regs .nat [] "num_blks" =
      some (Tile.scalar (s0.readMemValue .nat (Region.cast Blocks)
        (s0.pids 1 * NR + s0.pids 0)))
  ∧ s.regs .nat [] "num_cols" =
      some (Tile.scalar (s0.readMemValue .nat (Region.cast ColCounts)
        (s0.pids 1 * NR + s0.pids 0)))
  ∧ s.regs .ptr [] "blks_ptr" =
      some (Tile.scalar (Region.cast BlockOffsets, (s0.pids 1 * NR + s0.pids 0) * NS))
  ∧ s.regs .ptr [] "cols_ptr" =
      some (Tile.scalar (Region.cast Cols, (s0.pids 1 * NR + s0.pids 0) * NV))
  ∧ s.regs FloatDType.fp16.toTileDType [BM, BD] "q" =
      some (⟨fun idx : TileIndex [BM, BD] =>
        FloatDType.real.cast FloatDType.fp16 (qF idx)⟩ : Tile FloatDType.fp16.toTileDType [BM, BD])
  ∧ s.regs .ptr [BD, 1] "k_ptrs" =
      some (⟨fun idx : TileIndex [BD, 1] => kpF idx⟩ : Tile .ptr [BD, 1])
  ∧ s.regs .ptr [1, BD] "v_ptrs" =
      some (⟨fun idx : TileIndex [1, BD] => vpF idx⟩ : Tile .ptr [1, BD])
  ∧ s.regs .ptr [BM, BD] "o_ptrs" =
      some (⟨fun idx : TileIndex [BM, BD] => opF idx⟩ : Tile .ptr [BM, BD])
  ∧ s.regs .bool [BM, 1] "m_mask" =
      some (⟨fun idx : TileIndex [BM, 1] =>
        decide (s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))⟩ : Tile .bool [BM, 1])
  ∧ s.regs .nat [] "max_num_blks" = some (Tile.scalar 8)
  ∧ s.regs .nat [] "max_num_cols" = some (Tile.scalar 16)
  ∧ s.regs .real [BM] "m_i" =
      some (⟨fun idx : TileIndex [BM] =>
        msaSeedMax BM BN mA scoreB c idx.1⟩ : Tile .real [BM])
  ∧ s.regs .real [BM] "l_i" =
      some (⟨fun idx : TileIndex [BM] =>
        (some (msaLPartialSeed BM BN mA lA scoreB c idx.1) : WithBot ℝ)⟩ : Tile .real [BM])
  ∧ s.regs .real [BM, BD] "acc" =
      some (⟨fun idx : TileIndex [BM, BD] =>
        (some (msaOPartialSeed BM BN BD mA lA oA scoreB vblkB c idx.1 idx.2.1) : WithBot ℝ)⟩
          : Tile .real [BM, BD])

/-- Symbolic `q_ptrs` lane. -/
def msaQPtrGS (Q : RegionName) (BM BD H sqz sqh sqm sqk : Nat) (s0 : BlockState) : TileIndex [BM, BD] → RegionName × Nat :=
  fun idx => (Q, ((s0.pids 1 / H) * sqz + (s0.pids 1 % H) * sqh)
    + (s0.pids 0 * BM + idx.1.val) * sqm + idx.2.1.val * sqk)

/-- Symbolic `k_ptrs` lane. -/
def msaKPtrGS (K : RegionName) (BD H skz skh skk : Nat) (s0 : BlockState) : TileIndex [BD, 1] → RegionName × Nat :=
  fun idx => (K, ((s0.pids 1 / H) * skz + (s0.pids 1 % H) * skh) + idx.1.val * skk)

/-- Symbolic `v_ptrs` lane. -/
def msaVPtrGS (V : RegionName) (BD H skz skh svk : Nat) (s0 : BlockState) : TileIndex [1, BD] → RegionName × Nat :=
  fun idx => (V, ((s0.pids 1 / H) * skz + (s0.pids 1 % H) * skh) + idx.2.1.val * svk)

/-- Symbolic `o_ptrs` lane. -/
def msaOPtrGS (Out : RegionName) (BM BD H sqz sqh som sok : Nat) (s0 : BlockState) : TileIndex [BM, BD] → RegionName × Nat :=
  fun idx => (Out, ((s0.pids 1 / H) * sqz + (s0.pids 1 % H) * sqh)
    + (s0.pids 0 * BM + idx.1.val) * som + idx.2.1.val * sok)

/-- Symbolic scaled `q` value. -/
noncomputable def msaQValGS (Q : RegionName) (BM BD H sqz sqh sqm sqk : Nat) (s0 : BlockState)
    (sm_scale : ℝ := 0.1) : TileIndex [BM, BD] → WithBot ℝ :=
  fun idx => WithBot.realMul
    (s0.readMemValue .real Q (((s0.pids 1 / H) * sqz + (s0.pids 1 % H) * sqh)
      + (s0.pids 0 * BM + idx.1.val) * sqm + idx.2.1.val * sqk))
    (some (sm_scale * 1.44269504))

/-! ## FULLY-GENERAL preLoop exec (symbolic strides + layout) -/

/-- Symbolic-stride pre-loop statement list. -/
def msaPreLoopGS (Q K V : RegionName) (Seqlens : Region .nat)
    (Blocks BlockOffsets ColCounts Cols : Region .nat) (Out : RegionName)
    (BM BN BD H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV : Nat)
    (sm_scale : ℝ := 0.1) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_hz" (Op.programId 1),
    Stmt.assign .nat [] "seqlen"
      (Op.load .nat
        (MemAccess.region Seqlens
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)))
        MaskOpt.none) ]
    ++ msaSetupGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD
        H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV sm_scale

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
theorem msaPreLoop_evalGS
    (s : BlockState) (Q K V : RegionName) (Seqlens : Region .nat)
    (Blocks BlockOffsets ColCounts Cols : Region .nat) (Out : RegionName)
    (BM BN BD H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV : Nat)
    (scoreA : Nat → Fin BM → Fin BN → WithBot ℝ) (vblkA : Nat → Fin BN → Fin BD → ℝ)
    (sm_scale : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (msaPreLoopGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV sm_scale) s
        = some s0
      ∧ msaInvariantAGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
          scoreA vblkA (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale) (msaKPtrGS K BD H skz skh skk s) (msaVPtrGS V BD H skz skh svk s) (msaOPtrGS Out BM BD H sqz sqh som sok s) s 0 s0 := by
  unfold msaPreLoopGS msaSetupGS
  simp only [List.cons_append, List.nil_append, List.append_assoc]
  -- stmt 0: start_m = programId 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  -- stmt 1: off_hz = programId 1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  -- stmt 2: seqlen = load Seqlens[off_hz / 4]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region Seqlens
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)))
          MaskOpt.none) _
        = some (Tile.scalar (seqLen s H (Region.cast Seqlens))) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      refine congrArg some ?_; ext idx
      simp only [Tile.scalar, Tile.bop, Tile.bop_data, Broadcast.leftIndex,
        Broadcast.rightIndex, IntegralDType.floorDiv, seqLen, offZ,
        BlockState.readMemValue, BlockState.readMemTyped, BlockState.setReg_mem,
        if_true, if_pos]))]
  -- stmt 3: offs_m = start_m*64 + arange 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BM))
          (Op.arange BM)) _
        = some (Tile.vec (fun i : Fin BM => s.pids 0 * BM + i.val)) from by
      rw [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_arange]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  -- stmt 4: offs_n = arange 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BN) _ = some (Tile.vec (fun j : Fin BN => j.val)) from evalOp_arange BN _))]
  -- stmt 5: offs_d = arange 64
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BD) _ = some (Tile.vec (fun e : Fin BD => e.val)) from evalOp_arange BD _))]
  -- stmt 6: qo_offset = (off_hz/4)*32768 + (off_hz%4)*8192
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H))
            (Op.constNat sqz))
          (Op.mul .nat Broadcast.nil
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H))
            (Op.constNat sqh))) _
        = some (Tile.scalar (s.pids 1 / H * sqz + s.pids 1 % H * sqh)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.scalar, Tile.bop, Tile.bop_data, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
        IntegralDType.mod]))]
  -- stmt 7: kv_offset = same as qo_offset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H))
            (Op.constNat skz))
          (Op.mul .nat Broadcast.nil
            (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H))
            (Op.constNat skh))) _
        = some (Tile.scalar (s.pids 1 / H * skz + s.pids 1 % H * skh)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.scalar, Tile.bop, Tile.bop_data, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul, IntegralDType.floorDiv,
        IntegralDType.mod]))]
  -- stmt 8: q_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
          (Op.add .nat Broadcast.nil.consL.consR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qo_offset")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat sqm)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat sqk)))) _
        = some (⟨fun idx : TileIndex [BM, BD] => msaQPtrGS Q BM BD H sqz sqh sqm sqk s idx⟩ : Tile .ptr [BM, BD]) from by
      simp only [evalOp]
      erw [evalOp_expandDim_ref_of_regs .nat [BM] ⟨1, by simp⟩ "offs_m" _
            (Tile.vec (fun i : Fin BM => s.pids 0 * BM + i.val))
            (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true, reduceCtorEq, reduceDIte]),
        evalOp_expandDim_ref_of_regs .nat [BD] ⟨0, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin BD => e.val))
            (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true, reduceCtorEq, reduceDIte])]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨i, e, u⟩ := idx
      simp only [msaQPtrGS, Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar,
        Tile.scalar_data_index, Tile.vec, Tile.expandDim, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
        TileShape.dropInsertedIndex, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]
      ))]
  -- stmt 9: k_ptrs
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "kv_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat skk)))) _
        = some (⟨fun idx : TileIndex [BD, 1] => msaKPtrGS K BD H skz skh skk s idx⟩ : Tile .ptr [BD, 1]) from by
      simp only [evalOp]
      erw [evalOp_expandDim_ref_of_regs .nat [BD] ⟨1, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin BD => e.val))
            (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true, reduceCtorEq, reduceDIte])]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨e, j, u⟩ := idx
      simp only [msaKPtrGS, Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar,
        Tile.scalar_data_index, Tile.vec, Tile.expandDim, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
        TileShape.dropInsertedIndex, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]
      ))]
  -- stmt 10: v_ptrs
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "kv_offset")
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat svk)))) _
        = some (⟨fun idx : TileIndex [1, BD] => msaVPtrGS V BD H skz skh svk s idx⟩ : Tile .ptr [1, BD]) from by
      simp only [evalOp]
      erw [evalOp_expandDim_ref_of_regs .nat [BD] ⟨0, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin BD => e.val))
            (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true, reduceCtorEq, reduceDIte])]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨j, d, u⟩ := idx
      simp only [msaVPtrGS, Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar,
        Tile.scalar_data_index, Tile.vec, Tile.expandDim, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
        TileShape.dropInsertedIndex, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]
      ))]
  -- stmt 11: o_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
          (Op.add .nat Broadcast.nil.consL.consR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qo_offset")
              (Op.mul .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.constNat som)))
            (Op.mul .nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) (Op.constNat sok)))) _
        = some (⟨fun idx : TileIndex [BM, BD] => msaOPtrGS Out BM BD H sqz sqh som sok s idx⟩ : Tile .ptr [BM, BD]) from by
      simp only [evalOp]
      erw [evalOp_expandDim_ref_of_regs .nat [BM] ⟨1, by simp⟩ "offs_m" _
            (Tile.vec (fun i : Fin BM => s.pids 0 * BM + i.val))
            (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true, reduceCtorEq, reduceDIte]),
        evalOp_expandDim_ref_of_regs .nat [BD] ⟨0, by simp⟩ "offs_d" _
            (Tile.vec (fun e : Fin BD => e.val))
            (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true, reduceCtorEq, reduceDIte])]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨i, e, u⟩ := idx
      simp only [msaOPtrGS, Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar,
        Tile.scalar_data_index, Tile.vec, Tile.expandDim, castTile_self,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
        TileShape.dropInsertedIndex, Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]
      ))]
  -- stmt 12: num_blks = load block_count[off_hz*2 + start_m]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region Blocks
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat NR))
            (Op.ref .nat [] "start_m"))) MaskOpt.none) _
        = some (Tile.scalar (s.readMemValue .nat (Region.cast Blocks) (s.pids 1 * NR + s.pids 0))) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      refine congrArg some ?_; ext idx
      simp only [Tile.scalar, Tile.bop, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul, BlockState.readMemValue, BlockState.readMemTyped,
        BlockState.setReg_mem, if_true]))]
  -- stmt 13: blks_ptr = block_offset + (off_hz*2 + start_m)*4
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase BlockOffsets)
          (Op.mul .nat Broadcast.nil
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat NR))
              (Op.ref .nat [] "start_m"))
            (Op.constNat NS))) _
        = some (Tile.scalar (Region.cast BlockOffsets, (s.pids 1 * NR + s.pids 0) * NS)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
        Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]))]
  -- stmt 14: num_cols = load column_count[off_hz*2 + start_m]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region ColCounts
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat NR))
            (Op.ref .nat [] "start_m"))) MaskOpt.none) _
        = some (Tile.scalar (s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * NR + s.pids 0))) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      refine congrArg some ?_; ext idx
      simp only [Tile.scalar, Tile.bop, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, NumericDType.mul, BlockState.readMemValue, BlockState.readMemTyped,
        BlockState.setReg_mem, if_true]))]
  -- stmt 15: cols_ptr = column_index + (off_hz*2 + start_m)*8
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Cols)
          (Op.mul .nat Broadcast.nil
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat NR))
              (Op.ref .nat [] "start_m"))
            (Op.constNat NV))) _
        = some (Tile.scalar (Region.cast Cols, (s.pids 1 * NR + s.pids 0) * NV)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [Tile.ptrAdd_data, Tile.bop_data, Tile.bop, Tile.scalar, Tile.scalar_data_index,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
        Nat.zero_add, Prod.mk.injEq, true_and, Region.cast_id]))]
  -- stmt 16: m_i = full 0 + (-inf) = ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BM] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [BM] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM]) from by
      rw [evalOp_add, evalOp_full, evalOp_const, evalOp_negInf]
      simp only [Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add]
      rfl))]
  -- stmt 17: l_i = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BM] => some (0 : ℝ)⟩ : Tile .real [BM]) from by
      simp [evalOp_full, evalOp_const]))]
  -- stmt 18: acc = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BM, BD] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BM, BD] => some (0 : ℝ)⟩ : Tile .real [BM, BD]) from by
      simp [evalOp_full, evalOp_const]))]
  -- stmt 19: qk_scale = sm_scale * 1.44269504
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .real Broadcast.nil (Op.const (sm_scale : ℝ)) (Op.const 1.44269504)) _
        = some (Tile.scalar (some ((sm_scale : ℝ) * 1.44269504) : WithBot ℝ)) from by
      rw [evalOp_mul, evalOp_const, evalOp_const]
      simp only [Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.mul]
      rfl))]
  -- stmt 20: q = load q_ptrs (real)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .real (MemAccess.ptr (Op.ref .ptr [BM, BD] "q_ptrs")) MaskOpt.none) _
        = some (⟨fun idx : TileIndex [BM, BD] =>
            s.readMemValue .real (msaQPtrGS Q BM BD H sqz sqh sqm sqk s idx).1 (msaQPtrGS Q BM BD H sqz sqh sqm sqk s idx).2⟩ : Tile .real [BM, BD]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, BlockState.setReg_mem, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [msaQPtrGS, BlockState.readMemValue, BlockState.readMemAs, BlockState.setReg_mem,
        if_true]))]
  -- stmt 21: q = castFloat real→fp16 (q * qk_scale)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.castFloat FloatDType.real FloatDType.fp16
          (Op.mul .real Broadcast.scalarR (Op.ref .real [BM, BD] "q") (Op.ref .real [] "qk_scale"))) _
        = some (⟨fun idx : TileIndex [BM, BD] =>
            FloatDType.real.cast FloatDType.fp16 (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale idx)⟩
            : Tile FloatDType.fp16.toTileDType [BM, BD]) from by
      rw [evalOp_castFloat, evalOp_mul]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some, FloatDType.toTileDType_real]
      refine congrArg some (Tile.ext (fun idx => ?_))
      simp only [msaQValGS, msaQPtrGS, Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.mul]))]
  -- stmt 22: m_mask = offs_m[:,None] < seqlen
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BM] "offs_m")) (Op.ref .nat [] "seqlen")) _
        = some (⟨fun idx : TileIndex [BM, 1] =>
            decide (s.pids 0 * BM + idx.1.val < seqLen s H (Region.cast Seqlens))⟩
            : Tile .bool [BM, 1]) from by
      rw [evalOp_lt]
      erw [evalOp_expandDim_ref_of_regs .nat [BM] ⟨1, by simp⟩ "offs_m" _
            (Tile.vec (fun i : Fin BM => s.pids 0 * BM + i.val))
            (by simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
              String.reduceEq, not_false_eq_true, reduceCtorEq, reduceDIte])]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨i, j, u⟩ := idx
      simp only [Tile.cop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim,
        Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.lt,
        TileShape.dropInsertedIndex]
      rfl))]
  -- stmt 23: max_num_blks = 8
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.constNat 8) _ = some (Tile.scalar 8) from by simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  unfold msaInvariantAGS
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp  -- pids
  · funext rg o; simp  -- mem
  · intro rg o; simp [hundef]  -- undef
  · simp  -- start_m
  · simp  -- off_hz
  · simp  -- seqlen
  · simp [Tile.vec]  -- offs_m
  · simp [Tile.vec]  -- offs_n
  · simp [Tile.vec]  -- offs_d
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids,
      BlockState.setReg_mem, ne_eq, String.reduceEq, not_false_eq_true]  -- num_blks
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids,
      BlockState.setReg_mem, ne_eq, String.reduceEq, not_false_eq_true]  -- num_cols
  · simp  -- blks_ptr
  · simp  -- cols_ptr
  · simp  -- q
  · simp only [TileShape.insertAxis, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]  -- k_ptrs
  · simp only [TileShape.insertAxis, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]  -- v_ptrs
  · simp  -- o_ptrs
  · simp only [TileShape.insertAxis, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids,
      BlockState.setReg_mem, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq]  -- m_mask
  · simp  -- max_num_blks
  · -- m_i = msaMPartial 0 = ⊥
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids,
      ne_eq, String.reduceEq, not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [msaMPartial_zero]
  · -- l_i = msaLPartial 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids,
      ne_eq, String.reduceEq, not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [msaLPartial_zero]
  · -- acc = msaOPartial 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids,
      ne_eq, String.reduceEq, not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [msaOPartial_zero]

/-! ## FULLY-GENERAL lane defs (symbolic strides + layout) -/

/-! ## GENERAL lane defs (symbolic block dims) -/

/-- Symbolic Loop-A masked K-load lane `(e,j)`. -/
noncomputable def msaKLaneAGS (K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (BD BN H NR skn : Nat) (kpF : TileIndex [BD, 1] → RegionName × Nat) (s0 : BlockState) (c SN : Nat)
    (e : Fin BD) (j : Fin BN) : WithBot ℝ :=
  if (SN + j.val < seqLen s0 H (Region.cast Seqlens)
      ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + (SN + j.val) * skn) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)

/-- Symbolic Loop-A masked V-load lane `(j,d)`. -/
noncomputable def msaVLaneAGS (V : RegionName) (Seqlens Blocks : Region .nat)
    (BD BN H NR svn : Nat) (vpF : TileIndex [1, BD] → RegionName × Nat) (s0 : BlockState) (c SN : Nat)
    (j : Fin BN) (d : Fin BD) : WithBot ℝ :=
  if (SN + j.val < seqLen s0 H (Region.cast Seqlens)
      ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)) then
    (s0.readMemValue .real (vpF (⟨0, by simp⟩, d, PUnit.unit)).1
      ((vpF (⟨0, by simp⟩, d, PUnit.unit)).2 + (SN + j.val) * svn) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)

/-- Symbolic Loop-A masked `qk` lane `(i,j)`. -/
noncomputable def msaScoreLaneAGS (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (BM BN BD H NR skn : Nat) (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (s0 : BlockState) (c SN : Nat) (i : Fin BM) (j : Fin BN) : WithBot ℝ :=
  Option.map₂ (· + ·)
    (if (s0.pids 0 * BM + i.val < seqLen s0 H (Region.cast Seqlens)
        ∧ SN + j.val ≤ s0.pids 0 * BM + i.val) then
      (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (@Finset.sum (Fin BD) (WithBot ℝ) _ Finset.univ
      (fun e : Fin BD => Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (qF (i, e, PUnit.unit))))
        (msaKLaneAGS K Seqlens Blocks BlockOffsets BD BN H NR skn kpF s0 c SN e j)))

/-- Symbolic Loop-B gathered column lane `j`. -/
noncomputable def msaColLaneBGS (Cols : Region .nat) (ColCounts : Region .nat)
    (BN NR NV : Nat) (s0 : BlockState) (sv : Nat) (j : Fin BN) : Nat :=
  if sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0) then
    s0.readMemValue .nat (Region.cast Cols) ((s0.pids 1 * NR + s0.pids 0) * NV + sv + j.val)
  else 0

/-- Symbolic Loop-B masked K-load lane `(e,j)`. -/
noncomputable def msaKLaneBGS (Blocks ColCounts : Region .nat)
    (BD BN NR skn : Nat) (kpF : TileIndex [BD, 1] → RegionName × Nat) (s0 : BlockState) (sv : Nat)
    (gcol : Fin BN → Nat) (e : Fin BD) (j : Fin BN) : WithBot ℝ :=
  if (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
      ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + gcol j * skn) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)

/-- Symbolic Loop-B masked V-load lane `(j,d)`. -/
noncomputable def msaVLaneBGS (Blocks ColCounts : Region .nat)
    (BD BN NR svn : Nat) (vpF : TileIndex [1, BD] → RegionName × Nat) (s0 : BlockState) (sv : Nat)
    (gcol : Fin BN → Nat) (j : Fin BN) (d : Fin BD) : WithBot ℝ :=
  if (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
      ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)) then
    (s0.readMemValue .real (vpF (⟨0, by simp⟩, d, PUnit.unit)).1
      ((vpF (⟨0, by simp⟩, d, PUnit.unit)).2 + gcol j * svn) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)

/-- Symbolic Loop-B masked `qk` lane `(i,j)` (NON-causal). -/
noncomputable def msaScoreLaneBGS (Blocks ColCounts : Region .nat) (Seqlens : Region .nat)
    (BM BN BD H NR skn : Nat) (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (s0 : BlockState) (sv : Nat) (gcol : Fin BN → Nat) (i : Fin BM) (j : Fin BN) : WithBot ℝ :=
  Option.map₂ (· + ·)
    (if (s0.pids 0 * BM + i.val < seqLen s0 H (Region.cast Seqlens)
        ∧ (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
          ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0))) then
      (some (0 : ℝ) : WithBot ℝ) else (⊥ : WithBot ℝ))
    (@Finset.sum (Fin BD) (WithBot ℝ) _ Finset.univ
      (fun e : Fin BD => Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (qF (i, e, PUnit.unit))))
        (msaKLaneBGS Blocks ColCounts BD BN NR skn kpF s0 sv gcol e j)))


set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
theorem msa_attn_stepAGS
    (Q K V : RegionName) (Seqlens : Region .nat)
    (Blocks BlockOffsets ColCounts Cols : Region .nat) (Out : RegionName)
    (BM BN BD : Nat) (hBN : 0 < BN) (H NR NS NV skn svn : Nat)
    (scoreA : Nat → Fin BM → Fin BN → WithBot ℝ) (vblkA : Nat → Fin BN → Fin BD → ℝ)
    (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (vpF : TileIndex [1, BD] → RegionName × Nat) (opF : TileIndex [BM, BD] → RegionName × Nat)
    (s0 : BlockState) (c SN : Nat) (s : BlockState)
    (hinv : msaInvariantAGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
      scoreA vblkA qF kpF vpF opF s0 c s)
    (hSN : (if c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0) then
        s0.readMemValue .nat (Region.cast BlockOffsets)
          ((s0.pids 1 * NR + s0.pids 0) * NS + c)
       else BlockState.defaultCarrier .nat) = SN)
    (hscore : ∀ (i : Fin BM) (j : Fin BN), scoreA c i j =
      msaScoreLaneAGS Q K Seqlens Blocks BlockOffsets BM BN BD H NR skn qF kpF s0 c SN i j)
    (hvblk : ∀ (j : Fin BN) (d : Fin BD),
      (some (vblkA c j d) : WithBot ℝ) = msaVLaneAGS V Seqlens Blocks BD BN H NR svn vpF s0 c SN j d) :
    ∃ s', stepStmts (msaLoopBodyAGS BM BN BD skn svn) (s.setReg "block_index" .nat [] (Tile.scalar c)) = some s'
      ∧ msaInvariantAGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
          scoreA vblkA qF kpF vpF opF s0 (c + 1) s' := by
  obtain ⟨hpids, hmem, hundef, hsm, hoh, hseq, hoffm, hoffn, hoffd, hnb, hnc,
    hbp, hcp, hq, hkp, hvp, hop, hmmask, hmnb, hmi, hli, hacc⟩ := hinv
  -- readMemValue transports across any mem-equal state
  have rmv : ∀ (sX : BlockState), sX.mem = s0.mem → ∀ (dt : TileDType) (rg : RegionName) (o : Nat),
      sX.readMemValue dt rg o = s0.readMemValue dt rg o := by
    intro sX hX dt rg o
    simp only [BlockState.readMemValue, BlockState.readMemAs, BlockState.readMemTyped, hX]
  -- guard bit
  set NB := s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0) with hNBdef
  set SL := seqLen s0 H (Region.cast Seqlens) with hSLdef
  set cb : Bool := decide (c < NB) with hcbdef
  unfold msaLoopBodyAGS
  -- s1: block_index := c
  set s1 := s.setReg "block_index" .nat [] (Tile.scalar c) with hs1d
  have hs1pids : s1.pids = s0.pids := by rw [hs1d, BlockState.setReg_pids]; exact hpids
  have hs1mem : s1.mem = s0.mem := by funext rg o; rw [hs1d, BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "block_index" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1bi : s1.regs .nat [] "block_index" = some (Tile.scalar c) := by rw [hs1d, BlockState.setReg_same]
  have hs1nb : s1.regs .nat [] "num_blks" = some (Tile.scalar NB) := e1 (by decide) hnb
  -- A1: cond = block_index < num_blks
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_cond_eval s1 "block_index" "num_blks" c NB hs1bi hs1nb))]
  rw [show (Tile.cop ComparableDType.nat.lt Broadcast.nil (Tile.scalar c) (Tile.scalar NB))
        = (Tile.scalar cb : Tile .bool []) from by
    refine Tile.ext (fun idx => ?_)
    simp only [Tile.cop_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
      ComparableDType.lt, hcbdef]]
  set s2 := s1.setReg "cond" .bool [] (Tile.scalar cb) with hs2d
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "cond" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2pids : s2.pids = s0.pids := by rw [hs2d, BlockState.setReg_pids]; exact hs1pids
  have hs2mem : s2.mem = s0.mem := by funext rg o; rw [hs2d, BlockState.setReg_mem]; exact hs1mem ▸ rfl
  have hs2cond : s2.regs .bool [] "cond" = some (Tile.scalar cb) := by rw [hs2d, BlockState.setReg_same]
  have hs2bp : s2.regs .ptr [] "blks_ptr" =
      some (Tile.scalar (Region.cast BlockOffsets, (s0.pids 1 * NR + s0.pids 0) * NS)) :=
    e2 (by decide) (e1 (by decide) hbp)
  have hs2bi : s2.regs .nat [] "block_index" = some (Tile.scalar c) := e2 (by decide) hs1bi
  -- A2: start_n = masked load(blks_ptr + block_index, mask=cond)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_startn_eval s2 (Region.cast BlockOffsets) ((s0.pids 1 * NR + s0.pids 0) * NS) c cb
      hs2bp hs2bi hs2cond))]
  rw [show (Tile.scalar (if cb then
        s2.readMemValue .nat (Region.cast BlockOffsets) ((s0.pids 1 * NR + s0.pids 0) * NS + c)
       else BlockState.defaultCarrier .nat) : Tile .nat [])
      = Tile.scalar SN from by
    rw [rmv s2 hs2mem .nat (Region.cast BlockOffsets) _]
    rw [← hSN]
    by_cases h : c < NB
    · rw [hcbdef, if_pos (by simpa using h), if_pos h]
    · rw [hcbdef, if_neg (by simpa using h), if_neg h]]
  set s3 := s2.setReg "start_n" .nat [] (Tile.scalar SN) with hs3d
  have e3 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s2.regs dt sh nm = some t → s3.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs3d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs3pids : s3.pids = s0.pids := by rw [hs3d, BlockState.setReg_pids]; exact hs2pids
  have hs3mem : s3.mem = s0.mem := by funext rg o; rw [hs3d, BlockState.setReg_mem]; exact hs2mem ▸ rfl
  have hs3sn : s3.regs .nat [] "start_n" = some (Tile.scalar SN) := by rw [hs3d, BlockState.setReg_same]
  have hs3on : s3.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)) :=
    e3 (by decide) (e2 (by decide) (e1 (by decide) hoffn))
  -- A3: cols = start_n + offs_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_cols_eval s3 BN SN hs3sn hs3on))]
  rw [show (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar SN)
        (Tile.vec (fun j : Fin BN => j.val)) : Tile .nat [BN])
      = (Tile.vec (fun j : Fin BN => SN + j.val) : Tile .nat [BN]) from by
    refine Tile.ext (fun idx => ?_)
    simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.add]]
  set ct : Tile .nat [BN] := Tile.vec (fun j : Fin BN => SN + j.val) with hctd
  set s4 := s3.setReg "cols" .nat [BN] ct with hs4d
  have e4 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "cols" → s3.regs dt sh nm = some t → s4.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs4d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs4pids : s4.pids = s0.pids := by rw [hs4d, BlockState.setReg_pids]; exact hs3pids
  have hs4mem : s4.mem = s0.mem := by funext rg o; rw [hs4d, BlockState.setReg_mem]; exact hs3mem ▸ rfl
  have hs4cols : s4.regs .nat [BN] "cols" = some ct := by rw [hs4d, BlockState.setReg_same]
  have hs4seq : s4.regs .nat [] "seqlen" = some (Tile.scalar SL) :=
    e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hseq)))
  have hs4cond : s4.regs .bool [] "cond" = some (Tile.scalar cb) :=
    e4 (by decide) (e3 (by decide) hs2cond)
  -- A4: n_mask = (cols < seqlen) & cond
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_nmask_eval s4 BN SL cb ct hs4cols hs4seq hs4cond))]
  set nm : Tile .bool [BN] := Tile.bop (fun x y : Bool => x && y) Broadcast.scalarR
    (Tile.cop ComparableDType.nat.lt Broadcast.scalarR ct (Tile.scalar SL)) (Tile.scalar cb) with hnmd
  set s5 := s4.setReg "n_mask" .bool [BN] nm with hs5d
  have e5 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "n_mask" → s4.regs dt sh nm' = some t → s5.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs5d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs5pids : s5.pids = s0.pids := by rw [hs5d, BlockState.setReg_pids]; exact hs4pids
  have hs5mem : s5.mem = s0.mem := by funext rg o; rw [hs5d, BlockState.setReg_mem]; exact hs4mem ▸ rfl
  have hs5nm : s5.regs .bool [BN] "n_mask" = some nm := by rw [hs5d, BlockState.setReg_same]
  have hs5cols : s5.regs .nat [BN] "cols" = some ct := e5 (by decide) hs4cols
  have hs5kp : s5.regs .ptr [BD, 1] "k_ptrs" = some (⟨fun idx : TileIndex [BD, 1] => kpF idx⟩ : Tile .ptr [BD, 1]) :=
    e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hkp))))
  -- A5: k = masked K load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_load_k_eval s5 BD BN skn (by simp) (⟨fun idx : TileIndex [BD, 1] => kpF idx⟩ : Tile .ptr [BD, 1])
      ct nm hs5kp hs5cols hs5nm))]
  -- normalize the k-load tile to per-lane msaKLaneA
  set kT : Tile .real [BD, BN] := ⟨fun idx : TileIndex [BD, BN] =>
    msaKLaneAGS K Seqlens Blocks BlockOffsets BD BN H NR skn kpF s0 c SN idx.1 idx.2.1⟩ with hkTd
  rw [show (⟨fun idx : TileIndex [BD, BN] =>
        let ptrs := Tile.ptrAdd Broadcast.nil.consL.consR
          (⟨fun idx : TileIndex [BD, 1] => kpF idx⟩ : Tile .ptr [BD, 1])
          (Tile.bop NumericDType.nat.mul Broadcast.scalarR (Tile.expandDim ⟨0, by simp⟩ ct) (Tile.scalar skn))
        if (Tile.remap Broadcast.nil.consSame.consL.leftIndex (Tile.expandDim ⟨0, by simp⟩ nm)).data idx then
          s5.readMemValue .real (ptrs.data idx).1 (ptrs.data idx).2
        else (some (0.0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BD, BN]) = kT from by
    refine Tile.ext (fun idx => ?_); obtain ⟨e, j, u⟩ := idx
    simp only [hkTd, msaKLaneAGS, hctd, hnmd, Tile.remap, Tile.ptrAdd_data, Tile.bop_data,
      Tile.cop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim_data, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.mul, ComparableDType.lt, TileShape.dropInsertedIndex,
      rmv s5 hs5mem .real]
    by_cases hca : SN + j.val < seqLen s0 H (Region.cast Seqlens)
        ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)
    · rw [if_pos (by simpa [hcbdef] using hca), if_pos hca]
    · rw [if_neg (by simpa [hcbdef] using hca), if_neg hca]]
  set s6 := s5.setReg "k" .real [BD, BN] kT with hs6d
  have e6 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "k" → s5.regs dt sh nm' = some t → s6.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs6d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs6pids : s6.pids = s0.pids := by rw [hs6d, BlockState.setReg_pids]; exact hs5pids
  have hs6mem : s6.mem = s0.mem := by funext rg o; rw [hs6d, BlockState.setReg_mem]; exact hs5mem ▸ rfl
  have hs6k : s6.regs .real [BD, BN] "k" = some kT := by rw [hs6d, BlockState.setReg_same]
  have hs6cols : s6.regs .nat [BN] "cols" = some ct := e6 (by decide) hs5cols
  have hs6nm : s6.regs .bool [BN] "n_mask" = some nm := e6 (by decide) hs5nm
  have hs6vp : s6.regs .ptr [1, BD] "v_ptrs" = some (⟨fun idx : TileIndex [1, BD] => vpF idx⟩ : Tile .ptr [1, BD]) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hvp)))))
  -- A6: v = masked V load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_load_v_eval s6 BN BD svn (by simp) (⟨fun idx : TileIndex [1, BD] => vpF idx⟩ : Tile .ptr [1, BD])
      ct nm hs6vp hs6cols hs6nm))]
  set vT : Tile .real [BN, BD] := ⟨fun idx : TileIndex [BN, BD] =>
    msaVLaneAGS V Seqlens Blocks BD BN H NR svn vpF s0 c SN idx.1 idx.2.1⟩ with hvTd
  rw [show (⟨fun idx : TileIndex [BN, BD] =>
        let ptrs := Tile.ptrAdd Broadcast.nil.consR.consL
          (⟨fun idx : TileIndex [1, BD] => vpF idx⟩ : Tile .ptr [1, BD])
          (Tile.bop NumericDType.nat.mul Broadcast.scalarR (Tile.expandDim ⟨1, by simp⟩ ct) (Tile.scalar svn))
        if (Tile.remap Broadcast.nil.consL.consSame.leftIndex (Tile.expandDim ⟨1, by simp⟩ nm)).data idx then
          s6.readMemValue .real (ptrs.data idx).1 (ptrs.data idx).2
        else (some (0.0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BN, BD]) = vT from by
    refine Tile.ext (fun idx => ?_); obtain ⟨j, d, u⟩ := idx
    simp only [hvTd, msaVLaneAGS, hctd, hnmd, Tile.remap, Tile.ptrAdd_data, Tile.bop_data,
      Tile.cop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim_data, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.mul, ComparableDType.lt, TileShape.dropInsertedIndex,
      rmv s6 hs6mem .real]
    by_cases hca : SN + j.val < seqLen s0 H (Region.cast Seqlens)
        ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)
    · rw [if_pos (by simpa [hcbdef] using hca), if_pos hca]
    · rw [if_neg (by simpa [hcbdef] using hca), if_neg hca]]
  set s7 := s6.setReg "v" .real [BN, BD] vT with hs7d
  have e7 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "v" → s6.regs dt sh nm' = some t → s7.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs7d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs7pids : s7.pids = s0.pids := by rw [hs7d, BlockState.setReg_pids]; exact hs6pids
  have hs7mem : s7.mem = s0.mem := by funext rg o; rw [hs7d, BlockState.setReg_mem]; exact hs6mem ▸ rfl
  have hs7v : s7.regs .real [BN, BD] "v" = some vT := by rw [hs7d, BlockState.setReg_same]
  have hs7k : s7.regs .real [BD, BN] "k" = some kT := e7 (by decide) hs6k
  -- A7: qk = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_qkzeros_eval s7 BM BN))]
  set s8 := s7.setReg "qk" .real [BM, BN] (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) with hs8d
  have e8 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "qk" → s7.regs dt sh nm' = some t → s8.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs8d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs8pids : s8.pids = s0.pids := by rw [hs8d, BlockState.setReg_pids]; exact hs7pids
  have hs8mem : s8.mem = s0.mem := by funext rg o; rw [hs8d, BlockState.setReg_mem]; exact hs7mem ▸ rfl
  have hs8qk0 : s8.regs .real [BM, BN] "qk" = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
    rw [hs8d, BlockState.setReg_same]
  have hs8cols : s8.regs .nat [BN] "cols" = some ct := e8 (by decide) (e7 (by decide) hs6cols)
  have hs8offm : s8.regs .nat [BM] "offs_m" = some (Tile.vec (fun i : Fin BM => s0.pids 0 * BM + i.val)) :=
    e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hoffm)))))))
  -- A8: causal_mask = cols[None,:] ≤ offs_m[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_causal_eval s8 BM BN (by simp) (by simp) ct (Tile.vec (fun i : Fin BM => s0.pids 0 * BM + i.val))
      hs8cols hs8offm))]
  set cmT : Tile .bool [BM, BN] := Tile.cop ComparableDType.nat.le Broadcast.nil.consR.consL
    (Tile.expandDim ⟨0, by simp⟩ ct) (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin BM => s0.pids 0 * BM + i.val))) with hcmTd
  set s9 := s8.setReg "causal_mask" .bool [BM, BN] cmT with hs9d
  have e9 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "causal_mask" → s8.regs dt sh nm' = some t → s9.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs9d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs9pids : s9.pids = s0.pids := by rw [hs9d, BlockState.setReg_pids]; exact hs8pids
  have hs9mem : s9.mem = s0.mem := by funext rg o; rw [hs9d, BlockState.setReg_mem]; exact hs8mem ▸ rfl
  have hs9cm : s9.regs .bool [BM, BN] "causal_mask" = some cmT := by rw [hs9d, BlockState.setReg_same]
  have hs9qk0 : s9.regs .real [BM, BN] "qk" = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) :=
    e9 (by decide) hs8qk0
  have hs9mmask : s9.regs .bool [BM, 1] "m_mask" = some (⟨fun idx : TileIndex [BM, 1] =>
      decide (s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))⟩ : Tile .bool [BM, 1]) :=
    e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hmmask))))))))
  -- A9: qk = where(m_mask & causal_mask, qk, -inf)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_where_causal_eval s9 BM BN
      (⟨fun idx : TileIndex [BM, 1] => decide (s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))⟩ : Tile .bool [BM, 1])
      cmT (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
      hs9mmask hs9cm hs9qk0))]
  set qkSel : Tile .real [BM, BN] :=
    Tile.select (Tile.bop (fun x y : Bool => x && y) Broadcast.nil.consL.consSame
        (⟨fun idx : TileIndex [BM, 1] => decide (s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))⟩ : Tile .bool [BM, 1]) cmT)
      (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
      (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) with hqkSeld
  set s10 := s9.setReg "qk" .real [BM, BN] qkSel with hs10d
  have e10 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "qk" → s9.regs dt sh nm' = some t → s10.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs10d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs10pids : s10.pids = s0.pids := by rw [hs10d, BlockState.setReg_pids]; exact hs9pids
  have hs10mem : s10.mem = s0.mem := by funext rg o; rw [hs10d, BlockState.setReg_mem]; exact hs9mem ▸ rfl
  have hs10qk : s10.regs .real [BM, BN] "qk" = some qkSel := by rw [hs10d, BlockState.setReg_same]
  have hs10q : s10.regs FloatDType.fp16.toTileDType [BM, BD] "q" =
      some (⟨fun idx : TileIndex [BM, BD] => FloatDType.real.cast FloatDType.fp16 (qF idx)⟩ : Tile FloatDType.fp16.toTileDType [BM, BD]) :=
    e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hq)))))))))
  have hs10k : s10.regs .real [BD, BN] "k" = some kT :=
    e10 (by decide) (e9 (by decide) (e8 (by decide) hs7k))
  -- A10: qk += dot(q, k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_qk_dot_eval s10 BM BN BD qkSel (⟨fun idx : TileIndex [BM, BD] => FloatDType.real.cast FloatDType.fp16 (qF idx)⟩ : Tile FloatDType.fp16.toTileDType [BM, BD]) kT
      hs10qk hs10q hs10k))]
  -- normalize the A10 qk tile to `fun i j => scoreA c i j`
  rw [show (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qkSel
        (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real
          ((⟨fun idx : TileIndex [BM, BD] => FloatDType.real.cast FloatDType.fp16 (qF idx)⟩ : Tile FloatDType.fp16.toTileDType [BM, BD]).data i)⟩ : Tile .real [BM, BD]) kT) : Tile .real [BM, BN])
      = (⟨fun idx : TileIndex [BM, BN] => scoreA c idx.1 idx.2.1⟩ : Tile .real [BM, BN]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨i, j, u⟩ := idx
    show (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qkSel
      (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real
        ((⟨fun idx : TileIndex [BM, BD] => FloatDType.real.cast FloatDType.fp16 (qF idx)⟩ : Tile FloatDType.fp16.toTileDType [BM, BD]).data i)⟩ : Tile .real [BM, BD]) kT)).data (i, j, u) = scoreA c i j
    rw [hscore i j, msaScoreLaneAGS]
    simp only [Tile.bop_data, Tile.bop, qkSel, hqkSeld, Tile.select_data, Tile.cop_data,
      Tile.dot_nil_data, hkTd, hcmTd, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, Tile.vec, Tile.scalar, NumericDType.add, ComparableDType.le,
      TileShape.dropInsertedIndex]
    by_cases hcond : s0.pids 0 * BM + i.val < seqLen s0 H (Region.cast Seqlens) ∧ SN + j.val ≤ s0.pids 0 * BM + i.val
    · obtain ⟨h1, h2⟩ := hcond
      rw [if_pos (by simp only [hctd, Tile.vec, Tile.scalar, decide_eq_true_eq,
        Bool.and_eq_true]; exact ⟨h1, h2⟩), if_pos ⟨h1, h2⟩]; rfl
    · rw [if_neg (by simp only [hctd, Tile.vec, Tile.scalar, Bool.and_eq_true,
        decide_eq_true_eq]; exact hcond), if_neg hcond]; rfl]
  set qkS : Tile .real [BM, BN] := ⟨fun idx : TileIndex [BM, BN] => scoreA c idx.1 idx.2.1⟩ with hqkSd
  set s11 := s10.setReg "qk" .real [BM, BN] qkS with hs11d
  have e11 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "qk" → s10.regs dt sh nm' = some t → s11.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs11d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs11pids : s11.pids = s0.pids := by rw [hs11d, BlockState.setReg_pids]; exact hs10pids
  have hs11mem : s11.mem = s0.mem := by funext rg o; rw [hs11d, BlockState.setReg_mem]; exact hs10mem ▸ rfl
  have hs11qk : s11.regs .real [BM, BN] "qk" = some qkS := by rw [hs11d, BlockState.setReg_same]
  have hs11pids : s11.pids = s0.pids := by rw [hs11d, BlockState.setReg_pids]; exact hs10pids
  have hs11mem : s11.mem = s0.mem := by funext rg o; rw [hs11d, BlockState.setReg_mem]; exact hs10mem ▸ rfl
  -- m_i from invariant (must transport across the s setReg chain), reading msaMPartial c
  set mp : Tile .real [BM] := ⟨fun idx : TileIndex [BM] => msaMPartial BM BN scoreA c idx.1⟩ with hmpd
  have hs11mi : s11.regs .real [BM] "m_i" = some mp :=
    e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hmi))))))))))
  -- the row-max reduce of qkS
  set rmaxT : Tile .real [BM] := ⟨fun idx : TileIndex [BM] =>
    (Finset.univ : Finset (Fin BN)).sup (fun j => scoreA c idx.1 j)⟩ with hrmaxTd
  have hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkS = some rmaxT := by
    unfold Tile.reduceMaxDrop
    rw [dif_pos (by simp only [TileShape.axisDim]; exact hBN)]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨i, u⟩ := idx
    simp only [hrmaxTd, hqkSd, TileShape.insertAxisIndex, TileShape.axisDim, TileShape.eraseAxis]
    rw [Finset.sup'_eq_sup]
    rfl
  -- A11: m_i_new = maximum(m_i, max(qk,1)) = select(gt mp rmaxT) mp rmaxT
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_minew_eval s11 BM BN mp qkS rmaxT hs11mi hs11qk hrm))]
  -- bridge select(gt) → ⊔ → msaMPartial (c+1)
  set mnew : Tile .real [BM] := ⟨fun idx : TileIndex [BM] => msaMPartial BM BN scoreA (c + 1) idx.1⟩ with hmnewd
  rw [show (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mp rmaxT) mp rmaxT)
      = mnew from by
    refine Tile.ext (fun idx => ?_); obtain ⟨i, u⟩ := idx
    simp only [hmnewd, hmpd, hrmaxTd, Tile.select_data, Tile.cop_data, Tile.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, msaMPartial]
    by_cases h : (Finset.univ : Finset (Fin BN)).sup (fun j => scoreA c i j) < msaMPartial BM BN scoreA c i
    · rw [if_pos (by simpa using h), max_eq_left (le_of_lt h)]
    · rw [if_neg (by simpa using h), max_eq_right (not_lt.mp h)]]
  set s12 := s11.setReg "m_i_new" .real [BM] mnew with hs12d
  have e12 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "m_i_new" → s11.regs dt sh nm' = some t → s12.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs12d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs12pids : s12.pids = s0.pids := by rw [hs12d, BlockState.setReg_pids]; exact hs11pids
  have hs12mem : s12.mem = s0.mem := by funext rg o; rw [hs12d, BlockState.setReg_mem]; exact hs11mem ▸ rfl
  have hs12mnew : s12.regs .real [BM] "m_i_new" = some mnew := by rw [hs12d, BlockState.setReg_same]
  have hs12mi : s12.regs .real [BM] "m_i" = some mp := e12 (by decide) hs11mi
  -- A12: alpha = exp2(m_i - m_i_new)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_alpha_eval s12 BM mp mnew hs12mi hs12mnew))]
  set alphaT : Tile .real [BM] := Tile.uop WithBot.realExp2
    (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mp mnew) with halphaTd
  set s13 := s12.setReg "alpha" .real [BM] alphaT with hs13d
  have e13 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "alpha" → s12.regs dt sh nm' = some t → s13.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs13d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs13pids : s13.pids = s0.pids := by rw [hs13d, BlockState.setReg_pids]; exact hs12pids
  have hs13mem : s13.mem = s0.mem := by funext rg o; rw [hs13d, BlockState.setReg_mem]; exact hs12mem ▸ rfl
  have hs13alpha : s13.regs .real [BM] "alpha" = some alphaT := by rw [hs13d, BlockState.setReg_same]
  have hs13qk : s13.regs .real [BM, BN] "qk" = some qkS := e13 (by decide) (e12 (by decide) hs11qk)
  have hs13mnew : s13.regs .real [BM] "m_i_new" = some mnew := e13 (by decide) hs12mnew
  -- A13: p = exp2(qk - m_i_new[:,None])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_p_eval s13 BM BN (by simp) qkS mnew hs13qk hs13mnew))]
  set pT : Tile .real [BM, BN] := Tile.uop WithBot.realExp2
    (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkS
      (Tile.expandDim ⟨1, by simp⟩ mnew)) with hpTd
  set s14 := s13.setReg "p" .real [BM, BN] pT with hs14d
  have e14 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "p" → s13.regs dt sh nm' = some t → s14.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs14d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs14pids : s14.pids = s0.pids := by rw [hs14d, BlockState.setReg_pids]; exact hs13pids
  have hs14mem : s14.mem = s0.mem := by funext rg o; rw [hs14d, BlockState.setReg_mem]; exact hs13mem ▸ rfl
  have hs14p : s14.regs .real [BM, BN] "p" = some pT := by rw [hs14d, BlockState.setReg_same]
  have hs14alpha : s14.regs .real [BM] "alpha" = some alphaT := e14 (by decide) hs13alpha
  -- l_i from invariant
  set liT : Tile .real [BM] := ⟨fun idx : TileIndex [BM] => (some (msaLPartial BM BN scoreA c idx.1) : WithBot ℝ)⟩ with hliTd
  have hs14li : s14.regs .real [BM] "l_i" = some liT :=
    e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hli)))))))))))))
  -- A14: acc_scale = l_i*0 + alpha
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_accscale_eval s14 BM liT alphaT hs14li hs14alpha))]
  set ascT : Tile .real [BM] := Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
    (Tile.bop NumericDType.real.mul Broadcast.scalarR liT (Tile.scalar (some (0 : ℝ) : WithBot ℝ))) alphaT with hascTd
  set s15 := s14.setReg "acc_scale" .real [BM] ascT with hs15d
  have e15 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "acc_scale" → s14.regs dt sh nm' = some t → s15.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs15d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs15pids : s15.pids = s0.pids := by rw [hs15d, BlockState.setReg_pids]; exact hs14pids
  have hs15mem : s15.mem = s0.mem := by funext rg o; rw [hs15d, BlockState.setReg_mem]; exact hs14mem ▸ rfl
  have hs15asc : s15.regs .real [BM] "acc_scale" = some ascT := by rw [hs15d, BlockState.setReg_same]
  -- acc from invariant
  set accT : Tile .real [BM, BD] := ⟨fun idx : TileIndex [BM, BD] =>
    (some (msaOPartial BM BN BD scoreA vblkA c idx.1 idx.2.1) : WithBot ℝ)⟩ with haccTd
  have hs15acc : s15.regs .real [BM, BD] "acc" = some accT :=
    e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hacc))))))))))))))
  -- A15: acc *= acc_scale[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_acc_rescale_eval s15 BM BD (by simp) accT ascT hs15acc hs15asc))]
  set accMul : Tile .real [BM, BD] := Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
    accT (Tile.expandDim ⟨1, by simp⟩ ascT) with haccMuld
  set s16 := s15.setReg "acc" .real [BM, BD] accMul with hs16d
  have e16 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "acc" → s15.regs dt sh nm' = some t → s16.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs16d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs16pids : s16.pids = s0.pids := by rw [hs16d, BlockState.setReg_pids]; exact hs15pids
  have hs16mem : s16.mem = s0.mem := by funext rg o; rw [hs16d, BlockState.setReg_mem]; exact hs15mem ▸ rfl
  have hs16acc : s16.regs .real [BM, BD] "acc" = some accMul := by rw [hs16d, BlockState.setReg_same]
  have hs16p : s16.regs .real [BM, BN] "p" = some pT := e16 (by decide) (e15 (by decide) hs14p)
  have hs16v : s16.regs .real [BN, BD] "v" = some vT :=
    e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) hs7v))))))))
  -- A16: acc += dot((p).to(fp16), v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_acc_eval s16 BM BN BD accMul pT vT hs16acc hs16p hs16v))]
  set accFinal : Tile .real [BM, BD] := Tile.bop NumericDType.real.add
    (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) accMul
    (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real
      (FloatDType.real.cast FloatDType.fp16 (pT.data i))⟩ : Tile .real [BM, BN]) vT) with haccFinald
  set s17 := s16.setReg "acc" .real [BM, BD] accFinal with hs17d
  have e17 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "acc" → s16.regs dt sh nm' = some t → s17.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs17d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs17pids : s17.pids = s0.pids := by rw [hs17d, BlockState.setReg_pids]; exact hs16pids
  have hs17mem : s17.mem = s0.mem := by funext rg o; rw [hs17d, BlockState.setReg_mem]; exact hs16mem ▸ rfl
  have hs17acc : s17.regs .real [BM, BD] "acc" = some accFinal := by rw [hs17d, BlockState.setReg_same]
  have hs17li : s17.regs .real [BM] "l_i" = some liT := e17 (by decide) (e16 (by decide) hs14li)
  have hs17alpha : s17.regs .real [BM] "alpha" = some alphaT :=
    e17 (by decide) (e16 (by decide) (e15 (by decide) hs14alpha))
  have hs17p : s17.regs .real [BM, BN] "p" = some pT := e17 (by decide) hs16p
  -- A17: l_i = l_i*alpha + sum(p,1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_li_eval s17 BM BN liT alphaT pT hs17li hs17alpha hs17p))]
  set liFinal : Tile .real [BM] := Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
    (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) liT alphaT)
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) pT) with hliFinald
  set s18 := s17.setReg "l_i" .real [BM] liFinal with hs18d
  have e18 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "l_i" → s17.regs dt sh nm' = some t → s18.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs18d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs18pids : s18.pids = s0.pids := by rw [hs18d, BlockState.setReg_pids]; exact hs17pids
  have hs18mem : s18.mem = s0.mem := by funext rg o; rw [hs18d, BlockState.setReg_mem]; exact hs17mem ▸ rfl
  have hs18li : s18.regs .real [BM] "l_i" = some liFinal := by rw [hs18d, BlockState.setReg_same]
  have hs18mnew : s18.regs .real [BM] "m_i_new" = some mnew :=
    e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) hs12mnew)))))
  -- A18: m_i = m_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_reg_carry_eval s18 BM "m_i_new" mnew hs18mnew))]
  rw [stepStmts.nil]
  set s19 := s18.setReg "m_i" .real [BM] mnew with hs19d
  refine ⟨s19, rfl, ?_⟩
  -- realExp2 always returns `some _`
  have hexp2some : ∀ z : WithBot ℝ, WithBot.realExp2 z = some ((WithBot.realExp2 z).unbotD 0) := by
    intro z; cases z <;> rfl
  have e19 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "m_i" → s18.regs dt sh nm' = some t → s19.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs19d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  -- chain a preserved register from s to s19 (skips all written names)
  have chainAll : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "block_index" → nm' ≠ "cond" → nm' ≠ "start_n" → nm' ≠ "cols" → nm' ≠ "n_mask"
      → nm' ≠ "k" → nm' ≠ "v" → nm' ≠ "qk" → nm' ≠ "causal_mask" → nm' ≠ "m_i_new"
      → nm' ≠ "alpha" → nm' ≠ "p" → nm' ≠ "acc_scale" → nm' ≠ "acc" → nm' ≠ "l_i" → nm' ≠ "m_i"
      → s.regs dt sh nm' = some t → s19.regs dt sh nm' = some t := by
    intro dt sh nm' t h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h
    exact e19 h15 (e18 h14 (e17 h13 (e16 h13 (e15 h12 (e14 h11 (e13 h10 (e12 h9 (e11 h7
      (e10 h7 (e9 h8 (e8 h7 (e7 h6 (e6 h5 (e5 h4 (e4 h3 (e3 h2 (e2 h1 (e1 h0 h))))))))))))))))))
  have hs19pids : s19.pids = s0.pids := by rw [hs19d, BlockState.setReg_pids]; exact hs18pids
  have hs19mem : s19.mem = s0.mem := by funext rg o; rw [hs19d, BlockState.setReg_mem]; exact hs18mem ▸ rfl
  have hs19undef : ∀ rg o, s19.undef rg o = 0 := by
    intro rg o
    have : s19.undef rg o = s.undef rg o := by
      rw [hs19d, BlockState.setReg_undef, hs18d, BlockState.setReg_undef, hs17d, BlockState.setReg_undef,
        hs16d, BlockState.setReg_undef, hs15d, BlockState.setReg_undef, hs14d, BlockState.setReg_undef,
        hs13d, BlockState.setReg_undef, hs12d, BlockState.setReg_undef, hs11d, BlockState.setReg_undef,
        hs10d, BlockState.setReg_undef, hs9d, BlockState.setReg_undef, hs8d, BlockState.setReg_undef,
        hs7d, BlockState.setReg_undef, hs6d, BlockState.setReg_undef, hs5d, BlockState.setReg_undef,
        hs4d, BlockState.setReg_undef, hs3d, BlockState.setReg_undef, hs2d, BlockState.setReg_undef,
        hs1d, BlockState.setReg_undef]
    rw [this]; exact hundef rg o
  unfold msaInvariantAGS
  refine ⟨hs19pids, hs19mem, hs19undef, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hsm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoh
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hseq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoffm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoffn
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoffd
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hnb
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hnc
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hbp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hcp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hkp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hvp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hop
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hmmask
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hmnb
  · -- m_i = msaMPartial (c+1)
    rw [hs19d, BlockState.setReg_same]
  · -- l_i = some (msaLPartial (c+1))
    rw [show s19.regs .real [BM] "l_i" = some liFinal from by
      rw [hs19d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hs18li]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨i, u⟩ := idx
    show liFinal.data (i, u) = (some (msaLPartial BM BN scoreA (c + 1) i) : WithBot ℝ)
    -- the reduce-sum lane → some of real sum
    have hsumL : (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) pT).data (i, PUnit.unit)
        = (some ((Finset.univ : Finset (Fin BN)).sum (fun j : Fin BN =>
            (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (scoreA c i j)
              (msaMPartial BM BN scoreA (c + 1) i))).unbotD 0)) : WithBot ℝ) := by
      rw [Tile.reduceSumDrop_data, ← WithBot.sum_someTerm_eq_some]
      apply Finset.sum_congr rfl; intro j _
      simp only [hpTd, hmnewd, hqkSd, Tile.uop_data, Tile.bop_data, Tile.bop, Tile.expandDim_data,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub, TileShape.insertAxisIndex,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.dropInsertedIndex, WithBot.realSub]
      exact hexp2some _
    simp only [hliFinald, halphaTd, hmpd, hmnewd, hliTd, Tile.bop_data, Tile.bop,
      Tile.uop_data, NumericDType.add, NumericDType.mul, NumericDType.sub,
      Broadcast.leftIndex, Broadcast.rightIndex, WithBot.realSub, WithBot.realAdd, WithBot.realMul]
    rw [hexp2some (Option.map₂ (fun x y : ℝ => x - y) (msaMPartial BM BN scoreA c i) (msaMPartial BM BN scoreA (c+1) i))]
    erw [hsumL]
    rw [Option.map₂_some_some, Option.map₂_some_some]
    rw [show msaLPartial BM BN scoreA (c + 1) i =
        (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (msaMPartial BM BN scoreA c i)
            (msaMPartial BM BN scoreA (c + 1) i))).unbotD 0 * msaLPartial BM BN scoreA c i +
          (Finset.univ : Finset (Fin BN)).sum (fun j =>
            (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (scoreA c i j)
              (msaMPartial BM BN scoreA (c + 1) i))).unbotD 0) from rfl]
    refine congrArg some ?_; ring
  · -- acc = some (msaOPartial (c+1))
    rw [show s19.regs .real [BM, BD] "acc" = some accFinal from by
      rw [hs19d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        hs18d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hs17acc]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨i, d, u⟩ := idx
    show accFinal.data (i, d, u) = (some (msaOPartial BM BN BD scoreA vblkA (c + 1) i d) : WithBot ℝ)
    -- the p·v dot lane → some of real sum
    have hsumO : (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (pT.data i))⟩ : Tile .real [BM, BN]) vT).data (i, d, PUnit.unit)
        = (some ((Finset.univ : Finset (Fin BN)).sum (fun j : Fin BN =>
            (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (scoreA c i j)
              (msaMPartial BM BN scoreA (c + 1) i))).unbotD 0 * vblkA c j d)) : WithBot ℝ) := by
      rw [Tile.dot_nil_data, ← WithBot.sum_someTerm_eq_some]
      apply Finset.sum_congr rfl; intro j _
      simp only [hpTd, hmnewd, hqkSd, hvTd, Tile.uop_data, Tile.bop_data, Tile.bop,
        Tile.expandDim_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub,
        TileShape.dropInsertedIndex, WithBot.realSub]
      rw [hexp2some (Option.map₂ (fun x y : ℝ => x - y) (scoreA c i j) (msaMPartial BM BN scoreA (c+1) i)),
        ← hvblk j d]
      simp only [FloatDType.cast, FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
        FloatDType.real_ofWithBot, FloatDType.real_toWithBot]
      rfl
    simp only [haccFinald, haccMuld, halphaTd, hmpd, hmnewd, haccTd, hascTd, hliTd,
      Tile.bop_data, Tile.bop, Tile.uop_data, Tile.expandDim_data, Tile.scalar,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul, NumericDType.sub,
      TileShape.dropInsertedIndex, WithBot.realSub, WithBot.realAdd, WithBot.realMul]
    rw [hexp2some (Option.map₂ (fun x y : ℝ => x - y) (msaMPartial BM BN scoreA c i) (msaMPartial BM BN scoreA (c+1) i))]
    rw [hsumO]
    rw [Option.map₂_some_some, Option.map₂_some_some, Option.map₂_some_some, Option.map₂_some_some]
    rw [show msaOPartial BM BN BD scoreA vblkA (c + 1) i d =
        (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (msaMPartial BM BN scoreA c i)
            (msaMPartial BM BN scoreA (c + 1) i))).unbotD 0 * msaOPartial BM BN BD scoreA vblkA c i d +
          (Finset.univ : Finset (Fin BN)).sum (fun j =>
            (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (scoreA c i j)
              (msaMPartial BM BN scoreA (c + 1) i))).unbotD 0 * vblkA c j d) from rfl]
    refine congrArg some ?_
    ring



set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
theorem msa_attn_stepBGS
    (Q K V : RegionName) (Seqlens : Region .nat)
    (Blocks BlockOffsets ColCounts Cols : Region .nat) (Out : RegionName)
    (BM BN BD : Nat) (hBN : 0 < BN) (H NR NS NV skn svn : Nat)
    (scoreB : Nat → Fin BM → Fin BN → WithBot ℝ) (vblkB : Nat → Fin BN → Fin BD → ℝ)
    (mA : Fin BM → WithBot ℝ) (lA : Fin BM → ℝ) (oA : Fin BM → Fin BD → ℝ)
    (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (vpF : TileIndex [1, BD] → RegionName × Nat) (opF : TileIndex [BM, BD] → RegionName × Nat)
    (s0 : BlockState) (c sv : Nat) (s : BlockState)
    (hinv : msaInvariantBGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
      scoreB vblkB mA lA oA qF kpF vpF opF s0 c s)
    -- the gathered columns at loop value `sv`
    (gcol : Fin BN → Nat)
    (hgcol : ∀ j : Fin BN, msaColLaneBGS Cols ColCounts BN NR NV s0 sv j = gcol j)
    (hscore : ∀ (i : Fin BM) (j : Fin BN), scoreB c i j =
      msaScoreLaneBGS Blocks ColCounts Seqlens BM BN BD H NR skn qF kpF s0 sv gcol i j)
    (hvblk : ∀ (j : Fin BN) (d : Fin BD),
      (some (vblkB c j d) : WithBot ℝ) = msaVLaneBGS Blocks ColCounts BD BN NR svn vpF s0 sv gcol j d) :
    ∃ s', stepStmts (msaLoopBodyBGS BM BN BD skn svn) (s.setReg "start_n" .nat [] (Tile.scalar sv)) = some s'
      ∧ msaInvariantBGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
          scoreB vblkB mA lA oA qF kpF vpF opF s0 (c + 1) s' := by
  obtain ⟨hpids, hmem, hundef, hsm, hoh, hseq, hoffm, hoffn, hoffd, hnb, hnc,
    hbp, hcp, hq, hkp, hvp, hop, hmmask, hmnb, hmnc, hmi, hli, hacc⟩ := hinv
  have rmv : ∀ (sX : BlockState), sX.mem = s0.mem → ∀ (dt : TileDType) (rg : RegionName) (o : Nat),
      sX.readMemValue dt rg o = s0.readMemValue dt rg o := by
    intro sX hX dt rg o
    simp only [BlockState.readMemValue, BlockState.readMemAs, BlockState.readMemTyped, hX]
  set NC := s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0) with hNCdef
  set NBlk := s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0) with hNBlkdef
  set SL := seqLen s0 H (Region.cast Seqlens) with hSLdef
  set cb : Bool := decide (sv < NC) with hcbdef
  unfold msaLoopBodyBGS
  set s1 := s.setReg "start_n" .nat [] (Tile.scalar sv) with hs1d
  have hs1pids : s1.pids = s0.pids := by rw [hs1d, BlockState.setReg_pids]; exact hpids
  have hs1mem : s1.mem = s0.mem := by funext rg o; rw [hs1d, BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "start_n" → s.regs dt sh nm' = some t → s1.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1sn : s1.regs .nat [] "start_n" = some (Tile.scalar sv) := by rw [hs1d, BlockState.setReg_same]
  have hs1nc : s1.regs .nat [] "num_cols" = some (Tile.scalar NC) := e1 (by decide) hnc
  -- B1: cond = start_n < num_cols
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_cond_eval s1 "start_n" "num_cols" sv NC hs1sn hs1nc))]
  rw [show (Tile.cop ComparableDType.nat.lt Broadcast.nil (Tile.scalar sv) (Tile.scalar NC))
        = (Tile.scalar cb : Tile .bool []) from by
    refine Tile.ext (fun idx => ?_)
    simp only [Tile.cop_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
      ComparableDType.lt, hcbdef]]
  set s2 := s1.setReg "cond" .bool [] (Tile.scalar cb) with hs2d
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "cond" → s1.regs dt sh nm' = some t → s2.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2pids : s2.pids = s0.pids := by rw [hs2d, BlockState.setReg_pids]; exact hs1pids
  have hs2mem : s2.mem = s0.mem := by funext rg o; rw [hs2d, BlockState.setReg_mem]; exact hs1mem ▸ rfl
  have hs2cond : s2.regs .bool [] "cond" = some (Tile.scalar cb) := by rw [hs2d, BlockState.setReg_same]
  have hs2sn : s2.regs .nat [] "start_n" = some (Tile.scalar sv) := e2 (by decide) hs1sn
  have hs2on : s2.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)) :=
    e2 (by decide) (e1 (by decide) hoffn)
  have hs2nc : s2.regs .nat [] "num_cols" = some (Tile.scalar NC) := e2 (by decide) hs1nc
  -- B2: n_mask = (start_n + offs_n < num_cols) & cond
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_nmask_col_eval s2 BN sv NC cb hs2sn hs2on hs2nc hs2cond))]
  set nm : Tile .bool [BN] := Tile.bop (fun x y : Bool => x && y) Broadcast.scalarR
    (Tile.cop ComparableDType.nat.lt Broadcast.scalarR
      (Tile.bop NumericDType.nat.add Broadcast.scalarL (Tile.scalar sv) (Tile.vec (fun j : Fin BN => j.val)))
      (Tile.scalar NC)) (Tile.scalar cb) with hnmd
  set s3 := s2.setReg "n_mask" .bool [BN] nm with hs3d
  have e3 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "n_mask" → s2.regs dt sh nm' = some t → s3.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs3d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs3pids : s3.pids = s0.pids := by rw [hs3d, BlockState.setReg_pids]; exact hs2pids
  have hs3mem : s3.mem = s0.mem := by funext rg o; rw [hs3d, BlockState.setReg_mem]; exact hs2mem ▸ rfl
  have hs3nm : s3.regs .bool [BN] "n_mask" = some nm := by rw [hs3d, BlockState.setReg_same]
  have hs3cp : s3.regs .ptr [] "cols_ptr" =
      some (Tile.scalar (Region.cast Cols, (s0.pids 1 * NR + s0.pids 0) * NV)) :=
    e3 (by decide) (e2 (by decide) (e1 (by decide) hcp))
  have hs3sn : s3.regs .nat [] "start_n" = some (Tile.scalar sv) := e3 (by decide) hs2sn
  have hs3on : s3.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)) :=
    e3 (by decide) hs2on
  have hs3cond : s3.regs .bool [] "cond" = some (Tile.scalar cb) := e3 (by decide) hs2cond
  -- B3: cols = masked column gather
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_colgather_eval s3 BN ((s0.pids 1 * NR + s0.pids 0) * NV) sv cb (Region.cast Cols) (by simp)
      hs3cp hs3sn hs3on hs3cond))]
  -- normalize the gathered cols tile to per-lane gcol
  set ct : Tile .nat [BN] := Tile.vec (fun j : Fin BN => gcol j) with hctd
  rw [show (⟨fun idx : TileIndex [BN] =>
        let ptrs := Tile.ptrAdd Broadcast.scalarL
          (Tile.ptrAdd Broadcast.nil (Tile.scalar (Region.cast Cols, (s0.pids 1 * NR + s0.pids 0) * NV)) (Tile.scalar sv))
          (Tile.vec (fun j : Fin BN => j.val))
        if (Tile.remap Broadcast.nil.consL.leftIndex
              (Tile.expandDim ⟨0, by simp⟩ (Tile.scalar cb : Tile .bool []))).data idx then
          s3.readMemValue .nat (ptrs.data idx).1 (ptrs.data idx).2
        else (0 : Nat)⟩ : Tile .nat [BN]) = ct from by
    refine Tile.ext (fun idx => ?_); obtain ⟨j, u⟩ := idx
    show _ = ct.data (j, u)
    rw [show ct.data (j, u) = gcol j from rfl, ← hgcol j]
    simp only [msaColLaneBGS, Tile.remap, Tile.ptrAdd_data, Tile.scalar,
      Tile.expandDim_data, Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex]
    by_cases h : sv < NC
    · rw [if_pos (by simpa [hcbdef] using h), if_pos h]
      show s3.readMemValue .nat (Region.cast Cols) _ = s0.readMemValue .nat (Region.cast Cols) _
      rw [rmv s3 hs3mem .nat]
      simp only [Tile.vec]
    · rw [if_neg (by simpa [hcbdef] using h), if_neg h]]
  set s4 := s3.setReg "cols" .nat [BN] ct with hs4d
  have e4 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "cols" → s3.regs dt sh nm' = some t → s4.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs4d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs4pids : s4.pids = s0.pids := by rw [hs4d, BlockState.setReg_pids]; exact hs3pids
  have hs4mem : s4.mem = s0.mem := by funext rg o; rw [hs4d, BlockState.setReg_mem]; exact hs3mem ▸ rfl
  have hs4cols : s4.regs .nat [BN] "cols" = some ct := by rw [hs4d, BlockState.setReg_same]
  have hs4nm : s4.regs .bool [BN] "n_mask" = some nm := e4 (by decide) hs3nm
  have hs4kp : s4.regs .ptr [BD, 1] "k_ptrs" = some (⟨fun idx : TileIndex [BD, 1] => kpF idx⟩ : Tile .ptr [BD, 1]) :=
    e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hkp)))
  -- B4: k = masked K load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_load_k_eval s4 BD BN skn (by simp) (⟨fun idx : TileIndex [BD, 1] => kpF idx⟩ : Tile .ptr [BD, 1])
      ct nm hs4kp hs4cols hs4nm))]
  set kT : Tile .real [BD, BN] := ⟨fun idx : TileIndex [BD, BN] =>
    msaKLaneBGS Blocks ColCounts BD BN NR skn kpF s0 sv gcol idx.1 idx.2.1⟩ with hkTd
  rw [show (⟨fun idx : TileIndex [BD, BN] =>
        let ptrs := Tile.ptrAdd Broadcast.nil.consL.consR
          (⟨fun idx : TileIndex [BD, 1] => kpF idx⟩ : Tile .ptr [BD, 1])
          (Tile.bop NumericDType.nat.mul Broadcast.scalarR (Tile.expandDim ⟨0, by simp⟩ ct) (Tile.scalar skn))
        if (Tile.remap Broadcast.nil.consSame.consL.leftIndex (Tile.expandDim ⟨0, by simp⟩ nm)).data idx then
          s4.readMemValue .real (ptrs.data idx).1 (ptrs.data idx).2
        else (some (0.0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BD, BN]) = kT from by
    refine Tile.ext (fun idx => ?_); obtain ⟨e, j, u⟩ := idx
    simp only [hkTd, msaKLaneBGS, hctd, hnmd, Tile.remap, Tile.ptrAdd_data, Tile.bop_data,
      Tile.cop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim_data, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.mul, NumericDType.add, ComparableDType.lt,
      TileShape.dropInsertedIndex, rmv s4 hs4mem .real]
    by_cases hca : sv + j.val < NC ∧ sv < NC
    · rw [if_pos (by simp only [hcbdef, Bool.and_eq_true, decide_eq_true_eq]; exact hca), if_pos hca]
    · rw [if_neg (by simp only [hcbdef, Bool.and_eq_true, decide_eq_true_eq]; exact hca),
        if_neg hca]]
  set s5 := s4.setReg "k" .real [BD, BN] kT with hs5d
  have e5 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "k" → s4.regs dt sh nm' = some t → s5.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs5d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs5pids : s5.pids = s0.pids := by rw [hs5d, BlockState.setReg_pids]; exact hs4pids
  have hs5mem : s5.mem = s0.mem := by funext rg o; rw [hs5d, BlockState.setReg_mem]; exact hs4mem ▸ rfl
  have hs5k : s5.regs .real [BD, BN] "k" = some kT := by rw [hs5d, BlockState.setReg_same]
  have hs5cols : s5.regs .nat [BN] "cols" = some ct := e5 (by decide) hs4cols
  have hs5nm : s5.regs .bool [BN] "n_mask" = some nm := e5 (by decide) hs4nm
  have hs5vp : s5.regs .ptr [1, BD] "v_ptrs" = some (⟨fun idx : TileIndex [1, BD] => vpF idx⟩ : Tile .ptr [1, BD]) :=
    e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hvp))))
  -- B5: v = masked V load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_load_v_eval s5 BN BD svn (by simp) (⟨fun idx : TileIndex [1, BD] => vpF idx⟩ : Tile .ptr [1, BD])
      ct nm hs5vp hs5cols hs5nm))]
  set vT : Tile .real [BN, BD] := ⟨fun idx : TileIndex [BN, BD] =>
    msaVLaneBGS Blocks ColCounts BD BN NR svn vpF s0 sv gcol idx.1 idx.2.1⟩ with hvTd
  rw [show (⟨fun idx : TileIndex [BN, BD] =>
        let ptrs := Tile.ptrAdd Broadcast.nil.consR.consL
          (⟨fun idx : TileIndex [1, BD] => vpF idx⟩ : Tile .ptr [1, BD])
          (Tile.bop NumericDType.nat.mul Broadcast.scalarR (Tile.expandDim ⟨1, by simp⟩ ct) (Tile.scalar svn))
        if (Tile.remap Broadcast.nil.consL.consSame.leftIndex (Tile.expandDim ⟨1, by simp⟩ nm)).data idx then
          s5.readMemValue .real (ptrs.data idx).1 (ptrs.data idx).2
        else (some (0.0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BN, BD]) = vT from by
    refine Tile.ext (fun idx => ?_); obtain ⟨j, d, u⟩ := idx
    simp only [hvTd, msaVLaneBGS, hctd, hnmd, Tile.remap, Tile.ptrAdd_data, Tile.bop_data,
      Tile.cop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim_data, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.mul, NumericDType.add, ComparableDType.lt,
      TileShape.dropInsertedIndex, rmv s5 hs5mem .real]
    by_cases hca : sv + j.val < NC ∧ sv < NC
    · rw [if_pos (by simp only [hcbdef, Bool.and_eq_true, decide_eq_true_eq]; exact hca), if_pos hca]
    · rw [if_neg (by simp only [hcbdef, Bool.and_eq_true, decide_eq_true_eq]; exact hca),
        if_neg hca]]
  set s6 := s5.setReg "v" .real [BN, BD] vT with hs6d
  have e6 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "v" → s5.regs dt sh nm' = some t → s6.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs6d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs6pids : s6.pids = s0.pids := by rw [hs6d, BlockState.setReg_pids]; exact hs5pids
  have hs6mem : s6.mem = s0.mem := by funext rg o; rw [hs6d, BlockState.setReg_mem]; exact hs5mem ▸ rfl
  have hs6v : s6.regs .real [BN, BD] "v" = some vT := by rw [hs6d, BlockState.setReg_same]
  have hs6k : s6.regs .real [BD, BN] "k" = some kT := e6 (by decide) hs5k
  have hs6nm : s6.regs .bool [BN] "n_mask" = some nm := e6 (by decide) hs5nm
  -- B6: qk = zeros
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_qkzeros_eval s6 BM BN))]
  set s7 := s6.setReg "qk" .real [BM, BN] (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) with hs7d
  have e7 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "qk" → s6.regs dt sh nm' = some t → s7.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs7d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs7pids : s7.pids = s0.pids := by rw [hs7d, BlockState.setReg_pids]; exact hs6pids
  have hs7mem : s7.mem = s0.mem := by funext rg o; rw [hs7d, BlockState.setReg_mem]; exact hs6mem ▸ rfl
  have hs7qk0 : s7.regs .real [BM, BN] "qk" = some (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN]) := by
    rw [hs7d, BlockState.setReg_same]
  have hs7nm : s7.regs .bool [BN] "n_mask" = some nm := e7 (by decide) hs6nm
  have hs7mmask : s7.regs .bool [BM, 1] "m_mask" = some (⟨fun idx : TileIndex [BM, 1] =>
      decide (s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))⟩ : Tile .bool [BM, 1]) :=
    e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hmmask))))))
  -- B7: qk = where(m_mask & n_mask, qk, -inf)  (non-causal)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_where_col_eval s7 BM BN
      (⟨fun idx : TileIndex [BM, 1] => decide (s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))⟩ : Tile .bool [BM, 1])
      nm (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
      hs7mmask hs7nm hs7qk0))]
  set qkSel : Tile .real [BM, BN] :=
    Tile.select (Tile.bop (fun x y : Bool => x && y) Broadcast.nil.consL.leadR
        (⟨fun idx : TileIndex [BM, 1] => decide (s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))⟩ : Tile .bool [BM, 1]) nm)
      (⟨fun _ : TileIndex [BM, BN] => some (0 : ℝ)⟩ : Tile .real [BM, BN])
      (⟨fun _ : TileIndex [BM, BN] => (⊥ : WithBot ℝ)⟩ : Tile .real [BM, BN]) with hqkSeld
  set s8 := s7.setReg "qk" .real [BM, BN] qkSel with hs8d
  have e8 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "qk" → s7.regs dt sh nm' = some t → s8.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs8d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs8pids : s8.pids = s0.pids := by rw [hs8d, BlockState.setReg_pids]; exact hs7pids
  have hs8mem : s8.mem = s0.mem := by funext rg o; rw [hs8d, BlockState.setReg_mem]; exact hs7mem ▸ rfl
  have hs8qk : s8.regs .real [BM, BN] "qk" = some qkSel := by rw [hs8d, BlockState.setReg_same]
  have hs8q : s8.regs FloatDType.fp16.toTileDType [BM, BD] "q" =
      some (⟨fun idx : TileIndex [BM, BD] => FloatDType.real.cast FloatDType.fp16 (qF idx)⟩ : Tile FloatDType.fp16.toTileDType [BM, BD]) :=
    e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hq)))))))
  have hs8k : s8.regs .real [BD, BN] "k" = some kT :=
    e8 (by decide) (e7 (by decide) hs6k)
  -- B8: qk += dot(q, k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msa_qk_dot_eval s8 BM BN BD qkSel (⟨fun idx : TileIndex [BM, BD] => FloatDType.real.cast FloatDType.fp16 (qF idx)⟩ : Tile FloatDType.fp16.toTileDType [BM, BD]) kT
      hs8qk hs8q hs8k))]
  rw [show (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qkSel
      (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real
        ((⟨fun idx : TileIndex [BM, BD] => FloatDType.real.cast FloatDType.fp16 (qF idx)⟩ : Tile FloatDType.fp16.toTileDType [BM, BD]).data i)⟩ : Tile .real [BM, BD]) kT) : Tile .real [BM, BN])
      = (⟨fun idx : TileIndex [BM, BN] => scoreB c idx.1 idx.2.1⟩ : Tile .real [BM, BN]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨i, j, u⟩ := idx
    show (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) qkSel
      (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real
        ((⟨fun idx : TileIndex [BM, BD] => FloatDType.real.cast FloatDType.fp16 (qF idx)⟩ : Tile FloatDType.fp16.toTileDType [BM, BD]).data i)⟩ : Tile .real [BM, BD]) kT)).data (i, j, u) = scoreB c i j
    rw [hscore i j, msaScoreLaneBGS]
    simp only [Tile.bop_data, Tile.bop, qkSel, hqkSeld, Tile.select_data, Tile.cop_data,
      Tile.dot_nil_data, hkTd, hnmd, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, TileShape.dropInsertedIndex]
    by_cases hcond : s0.pids 0 * BM + i.val < seqLen s0 H (Region.cast Seqlens)
        ∧ (sv + j.val < NC ∧ sv < NC)
    · obtain ⟨h1, h2, h3⟩ := hcond
      rw [if_pos (by simp only [hcbdef, Bool.and_eq_true, decide_eq_true_eq]; exact ⟨h1, h2, h3⟩), if_pos (by rw [hNCdef] at h2 h3; exact ⟨h1, h2, h3⟩)]; rfl
    · rw [if_neg (by
        simp only [hcbdef, Bool.and_eq_true, decide_eq_true_eq, Nat.add_eq, Nat.add_zero]
        rintro ⟨ha, hb, hcc⟩; exact hcond ⟨ha, hb, hcc⟩),
        if_neg (by rw [hNCdef] at hcond; exact hcond)]; rfl]
  set qkS : Tile .real [BM, BN] := ⟨fun idx : TileIndex [BM, BN] => scoreB c idx.1 idx.2.1⟩ with hqkSd
  set s9 := s8.setReg "qk" .real [BM, BN] qkS with hs9d
  have e9 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "qk" → s8.regs dt sh nm' = some t → s9.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs9d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs9pids : s9.pids = s0.pids := by rw [hs9d, BlockState.setReg_pids]; exact hs8pids
  have hs9mem : s9.mem = s0.mem := by funext rg o; rw [hs9d, BlockState.setReg_mem]; exact hs8mem ▸ rfl
  have hs9qk : s9.regs .real [BM, BN] "qk" = some qkS := by rw [hs9d, BlockState.setReg_same]
  -- m_i from invariant = msaSeedMax c
  set mp : Tile .real [BM] := ⟨fun idx : TileIndex [BM] => msaSeedMax BM BN mA scoreB c idx.1⟩ with hmpd
  have hs9mi : s9.regs .real [BM] "m_i" = some mp :=
    e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hmi))))))))
  set rmaxT : Tile .real [BM] := ⟨fun idx : TileIndex [BM] =>
    (Finset.univ : Finset (Fin BN)).sup (fun j => scoreB c idx.1 j)⟩ with hrmaxTd
  have hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BM, BN].length) qkS = some rmaxT := by
    unfold Tile.reduceMaxDrop
    rw [dif_pos (by simp only [TileShape.axisDim]; exact hBN)]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨i, u⟩ := idx
    simp only [hrmaxTd, hqkSd, TileShape.insertAxisIndex, TileShape.axisDim, TileShape.eraseAxis]
    rw [Finset.sup'_eq_sup]; rfl
  -- B9: m_i_new
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_minew_eval s9 BM BN mp qkS rmaxT hs9mi hs9qk hrm))]
  set mnew : Tile .real [BM] := ⟨fun idx : TileIndex [BM] => msaSeedMax BM BN mA scoreB (c + 1) idx.1⟩ with hmnewd
  rw [show (Tile.select (Tile.cop ComparableDType.real.gt Broadcast.nil.consSame mp rmaxT) mp rmaxT) = mnew from by
    refine Tile.ext (fun idx => ?_); obtain ⟨i, u⟩ := idx
    simp only [hmnewd, hmpd, hrmaxTd, Tile.select_data, Tile.cop_data, Tile.bop,
      Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, msaSeedMax]
    by_cases h : (Finset.univ : Finset (Fin BN)).sup (fun j => scoreB c i j) < msaSeedMax BM BN mA scoreB c i
    · rw [if_pos (by simpa using h), max_eq_left (le_of_lt h)]
    · rw [if_neg (by simpa using h), max_eq_right (not_lt.mp h)]]
  set s10 := s9.setReg "m_i_new" .real [BM] mnew with hs10d
  have e10 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "m_i_new" → s9.regs dt sh nm' = some t → s10.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs10d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs10pids : s10.pids = s0.pids := by rw [hs10d, BlockState.setReg_pids]; exact hs9pids
  have hs10mem : s10.mem = s0.mem := by funext rg o; rw [hs10d, BlockState.setReg_mem]; exact hs9mem ▸ rfl
  have hs10mnew : s10.regs .real [BM] "m_i_new" = some mnew := by rw [hs10d, BlockState.setReg_same]
  have hs10mi : s10.regs .real [BM] "m_i" = some mp := e10 (by decide) hs9mi
  -- B10: alpha
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_alpha_eval s10 BM mp mnew hs10mi hs10mnew))]
  set alphaT : Tile .real [BM] := Tile.uop WithBot.realExp2
    (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mp mnew) with halphaTd
  set s11 := s10.setReg "alpha" .real [BM] alphaT with hs11d
  have e11 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "alpha" → s10.regs dt sh nm' = some t → s11.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs11d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs11pids : s11.pids = s0.pids := by rw [hs11d, BlockState.setReg_pids]; exact hs10pids
  have hs11mem : s11.mem = s0.mem := by funext rg o; rw [hs11d, BlockState.setReg_mem]; exact hs10mem ▸ rfl
  have hs11alpha : s11.regs .real [BM] "alpha" = some alphaT := by rw [hs11d, BlockState.setReg_same]
  have hs11qk : s11.regs .real [BM, BN] "qk" = some qkS := e11 (by decide) (e10 (by decide) hs9qk)
  have hs11mnew : s11.regs .real [BM] "m_i_new" = some mnew := e11 (by decide) hs10mnew
  -- B11: p
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_p_eval s11 BM BN (by simp) qkS mnew hs11qk hs11mnew))]
  set pT : Tile .real [BM, BN] := Tile.uop WithBot.realExp2
    (Tile.bop NumericDType.real.sub (Broadcast.consSame (Broadcast.consR Broadcast.nil)) qkS
      (Tile.expandDim ⟨1, by simp⟩ mnew)) with hpTd
  set s12 := s11.setReg "p" .real [BM, BN] pT with hs12d
  have e12 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "p" → s11.regs dt sh nm' = some t → s12.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs12d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs12pids : s12.pids = s0.pids := by rw [hs12d, BlockState.setReg_pids]; exact hs11pids
  have hs12mem : s12.mem = s0.mem := by funext rg o; rw [hs12d, BlockState.setReg_mem]; exact hs11mem ▸ rfl
  have hs12p : s12.regs .real [BM, BN] "p" = some pT := by rw [hs12d, BlockState.setReg_same]
  have hs12alpha : s12.regs .real [BM] "alpha" = some alphaT := e12 (by decide) hs11alpha
  set liT : Tile .real [BM] := ⟨fun idx : TileIndex [BM] => (some (msaLPartialSeed BM BN mA lA scoreB c idx.1) : WithBot ℝ)⟩ with hliTd
  have hs12li : s12.regs .real [BM] "l_i" = some liT :=
    e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hli)))))))))))
  -- B12: acc_scale = l_i*0 + alpha
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_accscale_eval s12 BM liT alphaT hs12li hs12alpha))]
  set ascT : Tile .real [BM] := Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
    (Tile.bop NumericDType.real.mul Broadcast.scalarR liT (Tile.scalar (some (0 : ℝ) : WithBot ℝ))) alphaT with hascTd
  set s13 := s12.setReg "acc_scale" .real [BM] ascT with hs13d
  have e13 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "acc_scale" → s12.regs dt sh nm' = some t → s13.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs13d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs13pids : s13.pids = s0.pids := by rw [hs13d, BlockState.setReg_pids]; exact hs12pids
  have hs13mem : s13.mem = s0.mem := by funext rg o; rw [hs13d, BlockState.setReg_mem]; exact hs12mem ▸ rfl
  have hs13asc : s13.regs .real [BM] "acc_scale" = some ascT := by rw [hs13d, BlockState.setReg_same]
  set accT : Tile .real [BM, BD] := ⟨fun idx : TileIndex [BM, BD] =>
    (some (msaOPartialSeed BM BN BD mA lA oA scoreB vblkB c idx.1 idx.2.1) : WithBot ℝ)⟩ with haccTd
  have hs13acc : s13.regs .real [BM, BD] "acc" = some accT :=
    e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hacc))))))))))))
  -- B13: acc *= acc_scale[:,None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_acc_rescale_eval s13 BM BD (by simp) accT ascT hs13acc hs13asc))]
  set accMul : Tile .real [BM, BD] := Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
    accT (Tile.expandDim ⟨1, by simp⟩ ascT) with haccMuld
  set s14 := s13.setReg "acc" .real [BM, BD] accMul with hs14d
  have e14 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "acc" → s13.regs dt sh nm' = some t → s14.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs14d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs14pids : s14.pids = s0.pids := by rw [hs14d, BlockState.setReg_pids]; exact hs13pids
  have hs14mem : s14.mem = s0.mem := by funext rg o; rw [hs14d, BlockState.setReg_mem]; exact hs13mem ▸ rfl
  have hs14acc : s14.regs .real [BM, BD] "acc" = some accMul := by rw [hs14d, BlockState.setReg_same]
  have hs14p : s14.regs .real [BM, BN] "p" = some pT := e14 (by decide) (e13 (by decide) hs12p)
  have hs14v : s14.regs .real [BN, BD] "v" = some vT :=
    e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) hs6v)))))))
  -- B14: acc += dot((p).to(fp16), v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_acc_eval s14 BM BN BD accMul pT vT hs14acc hs14p hs14v))]
  set accFinal : Tile .real [BM, BD] := Tile.bop NumericDType.real.add
    (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) accMul
    (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real
      (FloatDType.real.cast FloatDType.fp16 (pT.data i))⟩ : Tile .real [BM, BN]) vT) with haccFinald
  set s15 := s14.setReg "acc" .real [BM, BD] accFinal with hs15d
  have e15 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "acc" → s14.regs dt sh nm' = some t → s15.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs15d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs15pids : s15.pids = s0.pids := by rw [hs15d, BlockState.setReg_pids]; exact hs14pids
  have hs15mem : s15.mem = s0.mem := by funext rg o; rw [hs15d, BlockState.setReg_mem]; exact hs14mem ▸ rfl
  have hs15acc : s15.regs .real [BM, BD] "acc" = some accFinal := by rw [hs15d, BlockState.setReg_same]
  have hs15li : s15.regs .real [BM] "l_i" = some liT := e15 (by decide) (e14 (by decide) hs12li)
  have hs15alpha : s15.regs .real [BM] "alpha" = some alphaT :=
    e15 (by decide) (e14 (by decide) (e13 (by decide) hs12alpha))
  have hs15p : s15.regs .real [BM, BN] "p" = some pT := e15 (by decide) hs14p
  -- B15: l_i = l_i*alpha + sum(p,1)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_li_eval s15 BM BN liT alphaT pT hs15li hs15alpha hs15p))]
  set liFinal : Tile .real [BM] := Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
    (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) liT alphaT)
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) pT) with hliFinald
  set s16 := s15.setReg "l_i" .real [BM] liFinal with hs16d
  have e16 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "l_i" → s15.regs dt sh nm' = some t → s16.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs16d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs16pids : s16.pids = s0.pids := by rw [hs16d, BlockState.setReg_pids]; exact hs15pids
  have hs16mem : s16.mem = s0.mem := by funext rg o; rw [hs16d, BlockState.setReg_mem]; exact hs15mem ▸ rfl
  have hs16li : s16.regs .real [BM] "l_i" = some liFinal := by rw [hs16d, BlockState.setReg_same]
  have hs16mnew : s16.regs .real [BM] "m_i_new" = some mnew :=
    e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) hs10mnew)))))
  -- B16: m_i = m_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (msa_reg_carry_eval s16 BM "m_i_new" mnew hs16mnew))]
  rw [stepStmts.nil]
  set s17 := s16.setReg "m_i" .real [BM] mnew with hs17d
  refine ⟨s17, rfl, ?_⟩
  have hexp2some : ∀ z : WithBot ℝ, WithBot.realExp2 z = some ((WithBot.realExp2 z).unbotD 0) := by
    intro z; cases z <;> rfl
  have e17 : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "m_i" → s16.regs dt sh nm' = some t → s17.regs dt sh nm' = some t := by
    intro dt sh nm' t hne h; rw [hs17d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have chainAll : ∀ {dt : TileDType} {sh : TileShape} {nm' : RegName} {t : Tile dt sh},
      nm' ≠ "start_n" → nm' ≠ "cond" → nm' ≠ "n_mask" → nm' ≠ "cols" → nm' ≠ "k" → nm' ≠ "v"
      → nm' ≠ "qk" → nm' ≠ "m_i_new" → nm' ≠ "alpha" → nm' ≠ "p" → nm' ≠ "acc_scale"
      → nm' ≠ "acc" → nm' ≠ "l_i" → nm' ≠ "m_i"
      → s.regs dt sh nm' = some t → s17.regs dt sh nm' = some t := by
    intro dt sh nm' t h0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h
    exact e17 h13 (e16 h12 (e15 h11 (e14 h11 (e13 h10 (e12 h9 (e11 h8 (e10 h7 (e9 h6 (e8 h6
      (e7 h6 (e6 h5 (e5 h4 (e4 h3 (e3 h2 (e2 h1 (e1 h0 h))))))))))))))))
  have hs17pids : s17.pids = s0.pids := by rw [hs17d, BlockState.setReg_pids]; exact hs16pids
  have hs17mem : s17.mem = s0.mem := by funext rg o; rw [hs17d, BlockState.setReg_mem]; exact hs16mem ▸ rfl
  have hs17undef : ∀ rg o, s17.undef rg o = 0 := by
    intro rg o
    have : s17.undef rg o = s.undef rg o := by
      rw [hs17d, BlockState.setReg_undef, hs16d, BlockState.setReg_undef, hs15d, BlockState.setReg_undef,
        hs14d, BlockState.setReg_undef, hs13d, BlockState.setReg_undef, hs12d, BlockState.setReg_undef,
        hs11d, BlockState.setReg_undef, hs10d, BlockState.setReg_undef, hs9d, BlockState.setReg_undef,
        hs8d, BlockState.setReg_undef, hs7d, BlockState.setReg_undef, hs6d, BlockState.setReg_undef,
        hs5d, BlockState.setReg_undef, hs4d, BlockState.setReg_undef, hs3d, BlockState.setReg_undef,
        hs2d, BlockState.setReg_undef, hs1d, BlockState.setReg_undef]
    rw [this]; exact hundef rg o
  unfold msaInvariantBGS
  refine ⟨hs17pids, hs17mem, hs17undef, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hsm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoh
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hseq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoffm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoffn
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoffd
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hnb
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hnc
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hbp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hcp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hkp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hvp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hop
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hmmask
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hmnb
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hmnc
  · rw [hs17d, BlockState.setReg_same]
  · rw [show s17.regs .real [BM] "l_i" = some liFinal from by
      rw [hs17d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hs16li]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨i, u⟩ := idx
    show liFinal.data (i, u) = (some (msaLPartialSeed BM BN mA lA scoreB (c + 1) i) : WithBot ℝ)
    have hsumL : (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BM, BN].length) pT).data (i, PUnit.unit)
        = (some ((Finset.univ : Finset (Fin BN)).sum (fun j : Fin BN =>
            (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (scoreB c i j)
              (msaSeedMax BM BN mA scoreB (c + 1) i))).unbotD 0)) : WithBot ℝ) := by
      rw [Tile.reduceSumDrop_data, ← WithBot.sum_someTerm_eq_some]
      apply Finset.sum_congr rfl; intro j _
      simp only [hpTd, hmnewd, hqkSd, Tile.uop_data, Tile.bop_data, Tile.bop, Tile.expandDim_data,
        Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub, TileShape.insertAxisIndex,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.dropInsertedIndex, WithBot.realSub]
      exact hexp2some _
    simp only [hliFinald, halphaTd, hmpd, hmnewd, hliTd, Tile.bop_data, Tile.bop,
      Tile.uop_data, NumericDType.add, NumericDType.mul, NumericDType.sub,
      Broadcast.leftIndex, Broadcast.rightIndex, WithBot.realSub, WithBot.realAdd, WithBot.realMul]
    rw [hexp2some (Option.map₂ (fun x y : ℝ => x - y) (msaSeedMax BM BN mA scoreB c i) (msaSeedMax BM BN mA scoreB (c+1) i))]
    erw [hsumL]
    rw [Option.map₂_some_some, Option.map₂_some_some]
    rw [show msaLPartialSeed BM BN mA lA scoreB (c + 1) i =
        (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (msaSeedMax BM BN mA scoreB c i)
            (msaSeedMax BM BN mA scoreB (c + 1) i))).unbotD 0 * msaLPartialSeed BM BN mA lA scoreB c i +
          (Finset.univ : Finset (Fin BN)).sum (fun j =>
            (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (scoreB c i j)
              (msaSeedMax BM BN mA scoreB (c + 1) i))).unbotD 0) from rfl]
    refine congrArg some ?_; ring
  · rw [show s17.regs .real [BM, BD] "acc" = some accFinal from by
      rw [hs17d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
        hs16d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hs15acc]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨i, d, u⟩ := idx
    show accFinal.data (i, d, u) = (some (msaOPartialSeed BM BN BD mA lA oA scoreB vblkB (c + 1) i d) : WithBot ℝ)
    have hsumO : (Tile.dot [] (⟨fun i => FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (pT.data i))⟩ : Tile .real [BM, BN]) vT).data (i, d, PUnit.unit)
        = (some ((Finset.univ : Finset (Fin BN)).sum (fun j : Fin BN =>
            (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (scoreB c i j)
              (msaSeedMax BM BN mA scoreB (c + 1) i))).unbotD 0 * vblkB c j d)) : WithBot ℝ) := by
      rw [Tile.dot_nil_data, ← WithBot.sum_someTerm_eq_some]
      apply Finset.sum_congr rfl; intro j _
      simp only [hpTd, hmnewd, hqkSd, hvTd, Tile.uop_data, Tile.bop_data, Tile.bop,
        Tile.expandDim_data, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub,
        TileShape.dropInsertedIndex, WithBot.realSub]
      rw [hexp2some (Option.map₂ (fun x y : ℝ => x - y) (scoreB c i j) (msaSeedMax BM BN mA scoreB (c+1) i)),
        ← hvblk j d]
      simp only [FloatDType.cast, FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
        FloatDType.real_ofWithBot, FloatDType.real_toWithBot]
      rfl
    simp only [haccFinald, haccMuld, halphaTd, hmpd, hmnewd, haccTd, hascTd, hliTd,
      Tile.bop_data, Tile.bop, Tile.uop_data, Tile.expandDim_data, Tile.scalar,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul, NumericDType.sub,
      TileShape.dropInsertedIndex, WithBot.realSub, WithBot.realAdd, WithBot.realMul]
    rw [hexp2some (Option.map₂ (fun x y : ℝ => x - y) (msaSeedMax BM BN mA scoreB c i) (msaSeedMax BM BN mA scoreB (c+1) i))]
    rw [hsumO]
    rw [Option.map₂_some_some, Option.map₂_some_some, Option.map₂_some_some, Option.map₂_some_some]
    rw [show msaOPartialSeed BM BN BD mA lA oA scoreB vblkB (c + 1) i d =
        (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (msaSeedMax BM BN mA scoreB c i)
            (msaSeedMax BM BN mA scoreB (c + 1) i))).unbotD 0 * msaOPartialSeed BM BN BD mA lA oA scoreB vblkB c i d +
          (Finset.univ : Finset (Fin BN)).sum (fun j =>
            (WithBot.realExp2 (Option.map₂ (fun x y : ℝ => x - y) (scoreB c i j)
              (msaSeedMax BM BN mA scoreB (c + 1) i))).unbotD 0 * vblkB c j d) from rfl]
    refine congrArg some ?_; ring


/-! ## FULLY-GENERAL stream defs + loop drivers + handoff (symbolic strides + layout) -/


/-- Symbolic masked block-start gather. -/
noncomputable def msaSN0GS (s0 : BlockState) (Blocks BlockOffsets : Region .nat) (NR NS c : Nat) : Nat :=
  if c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0) then
    s0.readMemValue .nat (Region.cast BlockOffsets) ((s0.pids 1 * NR + s0.pids 0) * NS + c)
  else BlockState.defaultCarrier .nat

/-- Symbolic block-A score stream. -/
noncomputable def msaScoreA0GS (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (BM BN BD H NR NS skn : Nat) (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin BM → Fin BN → WithBot ℝ :=
  fun c i j => msaScoreLaneAGS Q K Seqlens Blocks BlockOffsets BM BN BD H NR skn qF kpF s0 c
    (msaSN0GS s0 Blocks BlockOffsets NR NS c) i j

/-- Symbolic block-A value stream. -/
noncomputable def msaVblkA0GS (V : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (BD BN H NR NS svn : Nat) (vpF : TileIndex [1, BD] → RegionName × Nat) (s0 : BlockState) :
    Nat → Fin BN → Fin BD → ℝ :=
  fun c j d => (msaVLaneAGS V Seqlens Blocks BD BN H NR svn vpF s0 c (msaSN0GS s0 Blocks BlockOffsets NR NS c) j d).unbotD 0

theorem msaVLaneAGS_some_unbotD (V : RegionName) (Seqlens Blocks : Region .nat)
    (BD BN H NR svn : Nat) (vpF : TileIndex [1, BD] → RegionName × Nat) (s0 : BlockState) (c SN : Nat)
    (j : Fin BN) (d : Fin BD) :
    (some ((msaVLaneAGS V Seqlens Blocks BD BN H NR svn vpF s0 c SN j d).unbotD 0) : WithBot ℝ)
      = msaVLaneAGS V Seqlens Blocks BD BN H NR svn vpF s0 c SN j d := by
  unfold msaVLaneAGS
  split <;> simp [WithBot.unbotD_coe]

theorem msaVLaneBGS_some_unbotD (Blocks ColCounts : Region .nat)
    (BD BN NR svn : Nat) (vpF : TileIndex [1, BD] → RegionName × Nat) (s0 : BlockState) (sv : Nat)
    (gcol : Fin BN → Nat) (j : Fin BN) (d : Fin BD) :
    (some ((msaVLaneBGS Blocks ColCounts BD BN NR svn vpF s0 sv gcol j d).unbotD 0) : WithBot ℝ)
      = msaVLaneBGS Blocks ColCounts BD BN NR svn vpF s0 sv gcol j d := by
  unfold msaVLaneBGS
  split <;> simp [WithBot.unbotD_coe]

/-- Symbolic gathered columns at loop value `sv`. -/
noncomputable def msaGcol0GS (s0 : BlockState) (Cols ColCounts : Region .nat) (BN NR NV sv : Nat) :
    Fin BN → Nat :=
  fun j => msaColLaneBGS Cols ColCounts BN NR NV s0 sv j

/-- Symbolic column-B score stream (loop value `sv = c·BN`). -/
noncomputable def msaScoreB0GS (Q K : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (BM BN BD H NR NV skn : Nat) (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin BM → Fin BN → WithBot ℝ :=
  fun c i j => msaScoreLaneBGS Blocks ColCounts Seqlens BM BN BD H NR skn qF kpF s0 (c * BN)
    (msaGcol0GS s0 Cols ColCounts BN NR NV (c * BN)) i j

/-- Symbolic column-B value stream. -/
noncomputable def msaVblkB0GS (V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (BD BN NR NV svn : Nat) (vpF : TileIndex [1, BD] → RegionName × Nat) (s0 : BlockState) :
    Nat → Fin BN → Fin BD → ℝ :=
  fun c j d => (msaVLaneBGS Blocks ColCounts BD BN NR svn vpF s0 (c * BN)
    (msaGcol0GS s0 Cols ColCounts BN NR NV (c * BN)) j d).unbotD 0

set_option maxHeartbeats 4000000 in
/-- **General Loop-A driver.** -/
theorem msa_loopA_execGS
    (Q K V : RegionName) (Seqlens : Region .nat)
    (Blocks BlockOffsets ColCounts Cols : Region .nat) (Out : RegionName)
    (BM BN BD : Nat) (hBN : 0 < BN) (H NR NS NV skn svn : Nat)
    (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (vpF : TileIndex [1, BD] → RegionName × Nat) (opF : TileIndex [BM, BD] → RegionName × Nat)
    (s0 s : BlockState)
    (hinv : msaInvariantAGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
      (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn qF kpF s0)
      (msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn vpF s0) qF kpF vpF opF s0 0 s) :
    ∃ sF, stepStmt (Stmt.forRangeDyn "block_index" (Op.constNat 0)
        (Op.ref .nat [] "max_num_blks") (Op.constNat 1) (msaLoopBodyAGS BM BN BD skn svn)) s = some sF
      ∧ msaInvariantAGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
          (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn qF kpF s0)
          (msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn vpF s0) qF kpF vpF opF s0 8 sF := by
  set scoreA := msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn qF kpF s0 with hscA
  set vblkA := msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn vpF s0 with hvbA
  have hmnb : s.regs .nat [] "max_num_blks" = some (Tile.scalar 8) := by
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, h, _⟩ := hinv; exact h
  obtain ⟨final, sF, hloop, hfin, hP⟩ :=
    VeriTile.forRangeDyn_inv (idx := "block_index")
      (startOp := Op.constNat 0) (stopOp := Op.ref .nat [] "max_num_blks")
      (stepOp := Op.constNat 1)
      (P := fun i st => msaInvariantAGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
        scoreA vblkA qF kpF vpF opF s0 i st ∧ i ≤ 8)
      (s_init := s) (start := 0) (stop := 8) (step := 1)
      (by rw [evalOp_constNat]) (by rw [evalOp_ref, hmnb]) (by rw [evalOp_constNat])
      (by norm_num) ⟨hinv, by norm_num⟩
      (fun i st hlt hPi => by
        obtain ⟨hPinv, hPle⟩ := hPi
        obtain ⟨s', hstep, hinv'⟩ := msa_attn_stepAGS Q K V Seqlens Blocks BlockOffsets
          ColCounts Cols Out BM BN BD hBN H NR NS NV skn svn scoreA vblkA qF kpF vpF opF s0 i
          (msaSN0GS s0 Blocks BlockOffsets NR NS i) st hPinv rfl
          (fun a b => rfl)
          (fun a b => by rw [hvbA]; exact msaVLaneAGS_some_unbotD V Seqlens Blocks BD BN H NR svn vpF s0 i _ a b)
        exact ⟨s', hstep, hinv', by omega⟩)
  refine ⟨sF, hloop, ?_⟩
  obtain ⟨hPinv, hPle⟩ := hP
  have : final = 8 := by omega
  subst this; exact hPinv

set_option maxHeartbeats 4000000 in
/-- **General Loop-B driver.** Needs `16 ≤ BN` so exactly one column block runs
(`range(0, 16, BN)` has one iteration when `BN ≥ 16`). -/
theorem msa_loopB_execGS
    (Q K V : RegionName) (Seqlens : Region .nat)
    (Blocks BlockOffsets ColCounts Cols : Region .nat) (Out : RegionName)
    (BM BN BD : Nat) (hBN : 0 < BN) (hBN16 : 16 ≤ BN) (H NR NS NV skn svn : Nat)
    (mA : Fin BM → WithBot ℝ) (lA : Fin BM → ℝ) (oA : Fin BM → Fin BD → ℝ)
    (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (vpF : TileIndex [1, BD] → RegionName × Nat) (opF : TileIndex [BM, BD] → RegionName × Nat)
    (s0 s : BlockState)
    (hinv : msaInvariantBGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
      (msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn qF kpF s0)
      (msaVblkB0GS V Seqlens Blocks BlockOffsets ColCounts Cols BD BN NR NV svn vpF s0)
      mA lA oA qF kpF vpF opF s0 0 s) :
    ∃ sF, stepStmt (Stmt.forRangeDyn "start_n" (Op.constNat 0)
        (Op.ref .nat [] "max_num_cols") (Op.constNat BN) (msaLoopBodyBGS BM BN BD skn svn)) s = some sF
      ∧ msaInvariantBGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
          (msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn qF kpF s0)
          (msaVblkB0GS V Seqlens Blocks BlockOffsets ColCounts Cols BD BN NR NV svn vpF s0)
          mA lA oA qF kpF vpF opF s0 1 sF := by
  set scoreB := msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn qF kpF s0 with hscB
  set vblkB := msaVblkB0GS V Seqlens Blocks BlockOffsets ColCounts Cols BD BN NR NV svn vpF s0 with hvbB
  have hmnc : s.regs .nat [] "max_num_cols" = some (Tile.scalar 16) := by
    obtain ⟨_, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, h, _⟩ := hinv; exact h
  obtain ⟨final, sF, hloop, hfin, hP⟩ :=
    VeriTile.forRangeDyn_inv (idx := "start_n")
      (startOp := Op.constNat 0) (stopOp := Op.ref .nat [] "max_num_cols")
      (stepOp := Op.constNat BN)
      (P := fun i st => msaInvariantBGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
        scoreB vblkB mA lA oA qF kpF vpF opF s0 (i / BN) st ∧ i % BN = 0 ∧ i ≤ BN)
      (s_init := s) (start := 0) (stop := 16) (step := BN)
      (by rw [evalOp_constNat]) (by rw [evalOp_ref, hmnc]) (by rw [evalOp_constNat])
      (by exact hBN.ne') ⟨by rw [Nat.zero_div]; exact hinv, by simp, by omega⟩
      (fun i st hlt hPi => by
        obtain ⟨hPinv, hPmod, hPle⟩ := hPi
        obtain ⟨s', hstep, hinv'⟩ := msa_attn_stepBGS Q K V Seqlens Blocks BlockOffsets
          ColCounts Cols Out BM BN BD hBN H NR NS NV skn svn scoreB vblkB mA lA oA qF kpF vpF opF s0 (i / BN) i st hPinv
          (msaGcol0GS s0 Cols ColCounts BN NR NV i) (fun j => rfl)
          (fun a b => by
            have hii : i / BN * BN = i := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)
            rw [hscB, msaScoreB0GS, hii])
          (fun a b => by
            have hii : i / BN * BN = i := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)
            rw [hvbB, msaVblkB0GS, hii]
            exact msaVLaneBGS_some_unbotD Blocks ColCounts BD BN NR svn vpF s0 i _ a b)
        have hi0 : i = 0 := by
          have hltBN : i < BN := lt_of_lt_of_le hlt hBN16
          rw [Nat.mod_eq_of_lt hltBN] at hPmod; exact hPmod
        refine ⟨s', hstep, ?_, by rw [hi0]; simp, by omega⟩
        rw [show (i + BN) / BN = i / BN + 1 from by
          rw [Nat.add_div_right _ hBN]]; exact hinv')
  refine ⟨sF, hloop, ?_⟩
  obtain ⟨hPinv, hPmod, hPle⟩ := hP
  -- final ≥ 16 (it is the first multiple of BN reaching the stop), ≤ BN, and a multiple of BN ⇒ = BN
  have hfge : 16 ≤ final := by omega
  have hdvd : BN ∣ final := Nat.dvd_of_mod_eq_zero hPmod
  have hf : final = BN := by
    rcases (Nat.eq_zero_or_pos final) with h0 | hpos
    · omega
    · exact Nat.le_antisymm hPle (Nat.le_of_dvd hpos hdvd)
  rw [show (1 : Nat) = final / BN from by rw [hf]; rw [Nat.div_self hBN]]; exact hPinv

set_option maxHeartbeats 4000000 in
/-- **General A→B handoff.** -/
theorem msa_handoffGS
    (Q K V : RegionName) (Seqlens : Region .nat)
    (Blocks BlockOffsets ColCounts Cols : Region .nat) (Out : RegionName)
    (BM BN BD H NR NS NV : Nat)
    (scoreA : Nat → Fin BM → Fin BN → WithBot ℝ) (vblkA : Nat → Fin BN → Fin BD → ℝ)
    (scoreB : Nat → Fin BM → Fin BN → WithBot ℝ) (vblkB : Nat → Fin BN → Fin BD → ℝ)
    (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (vpF : TileIndex [1, BD] → RegionName × Nat) (opF : TileIndex [BM, BD] → RegionName × Nat)
    (s0 : BlockState) (bF : Nat) (s : BlockState)
    (hinv : msaInvariantAGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
      scoreA vblkA qF kpF vpF opF s0 bF s) :
    ∃ s', stepStmts [Stmt.assign .nat [] "max_num_cols" (Op.constNat 16)] s = some s'
      ∧ msaInvariantBGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
          scoreB vblkB (msaMPartial BM BN scoreA bF)
          (fun i => msaLPartial BM BN scoreA bF i)
          (fun i d => msaOPartial BM BN BD scoreA vblkA bF i d)
          qF kpF vpF opF s0 0 s' := by
  obtain ⟨hpids, hmem, hundef, hsm, hoh, hseq, hoffm, hoffn, hoffd, hnb, hnc,
    hbp, hcp, hq, hkp, hvp, hop, hmmask, hmnb, hmi, hli, hacc⟩ := hinv
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_constNat 16 s)), stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  set s' := s.setReg "max_num_cols" .nat [] (Tile.scalar 16) with hs'd
  have e : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "max_num_cols" → s.regs dt sh nm = some t → s'.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs'd, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  refine ⟨by rw [hs'd, BlockState.setReg_pids]; exact hpids,
    by funext rg o; rw [hs'd, BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o,
    by intro rg o; rw [hs'd, BlockState.setReg_undef]; exact hundef rg o,
    e (by decide) hsm, e (by decide) hoh, e (by decide) hseq, e (by decide) hoffm,
    e (by decide) hoffn, e (by decide) hoffd, e (by decide) hnb, e (by decide) hnc,
    e (by decide) hbp, e (by decide) hcp, e (by decide) hq, e (by decide) hkp,
    e (by decide) hvp, e (by decide) hop, e (by decide) hmmask, e (by decide) hmnb,
    by rw [hs'd, BlockState.setReg_same], ?_, ?_, ?_⟩
  · rw [e (by decide) hmi]; rfl
  · rw [e (by decide) hli]; rfl
  · rw [e (by decide) hacc]; rfl


/-! ## FULLY-GENERAL closed-form math bridge helpers (symbolic strides + layout; channel strides = 1) -/

theorem msaScoreA_dot_eqGS
    (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (BM BN BD H NR sqz sqh sqm sqk skz skh skn skk : Nat)
    (hsqk : sqk = 1) (hskk : skk = 1)
    (s0 : BlockState) (sm_scale : ℝ) (c SN : Nat) (i : Fin BM) (j : Fin BN) :
    (@Finset.sum (Fin BD) (WithBot ℝ) _ Finset.univ
        (fun e : Fin BD => Option.map₂ (· * ·)
          (FloatDType.fp16.cast FloatDType.real
            (FloatDType.real.cast FloatDType.fp16 (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale (i, e, PUnit.unit))))
          (msaKLaneAGS K Seqlens Blocks BlockOffsets BD BN H NR skn (msaKPtrGS K BD H skz skh skk s0) s0 c SN e j)))
      = some ((sm_scale * 1.44269504) *
          (if SN + j.val < seqLen s0 H (Region.cast Seqlens)
              ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)
            then rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (SN + j.val)
            else 0)) := by
  have hterm : ∀ e : Fin BD, Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale (i, e, PUnit.unit))))
        (msaKLaneAGS K Seqlens Blocks BlockOffsets BD BN H NR skn (msaKPtrGS K BD H skz skh skk s0) s0 c SN e j)
      = some ((sm_scale * 1.44269504) *
          (qRow s0 Q H sqz sqh sqm BM i e.val *
            (if SN + j.val < seqLen s0 H (Region.cast Seqlens)
                ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)
              then kRow s0 K H skz skh skn (SN + j.val) e.val else 0))) := by
    intro e
    have hQoff : ((s0.pids 1 / H) * sqz + (s0.pids 1 % H) * sqh)
        + (s0.pids 0 * BM + i.val) * sqm + e.val * 1
      = qoBase s0 H sqz sqh + mIndex s0 BM i * sqm + e.val := by
      unfold qoBase offZ offH mIndex; ring
    have hKoff : ((s0.pids 1 / H) * skz + (s0.pids 1 % H) * skh) + e.val * 1 + (SN + j.val) * skn
      = qoBase s0 H skz skh + (SN + j.val) * skn + e.val := by
      unfold qoBase offZ offH; ring
    subst hsqk hskk; unfold msaQValGS msaKLaneAGS msaKPtrGS
    by_cases hin : SN + j.val < seqLen s0 H (Region.cast Seqlens)
        ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)
    · rw [if_pos hin, if_pos hin]
      simp only [BlockState.readMemValue_real, FloatDType.cast,
        FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
        FloatDType.real_ofWithBot, WithBot.realMul, Option.map₂_some_some]
      rw [hQoff, hKoff]
      show some _ = some ((sm_scale * 1.44269504) *
        (qRow s0 Q H sqz sqh sqm BM i e.val * kRow s0 K H skz skh skn (SN + j.val) e.val))
      unfold qRow kRow; ring_nf
    · rw [if_neg hin, if_neg hin]
      simp only [BlockState.readMemValue_real, FloatDType.cast,
        FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
        FloatDType.real_ofWithBot, WithBot.realMul, Option.map₂_some_some, mul_zero]
      norm_num
  rw [show (@Finset.sum (Fin BD) (WithBot ℝ) _ Finset.univ
        (fun e : Fin BD => Option.map₂ (· * ·)
          (FloatDType.fp16.cast FloatDType.real
            (FloatDType.real.cast FloatDType.fp16 (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale (i, e, PUnit.unit))))
          (msaKLaneAGS K Seqlens Blocks BlockOffsets BD BN H NR skn (msaKPtrGS K BD H skz skh skk s0) s0 c SN e j)))
      = @Finset.sum (Fin BD) (WithBot ℝ) _ Finset.univ
          (fun e : Fin BD => some ((sm_scale * 1.44269504) *
            (qRow s0 Q H sqz sqh sqm BM i e.val *
              (if SN + j.val < seqLen s0 H (Region.cast Seqlens)
                  ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)
                then kRow s0 K H skz skh skn (SN + j.val) e.val else 0))))
      from Finset.sum_congr rfl (fun e _ => hterm e)]
  rw [WithBot.sum_someTerm_eq_some]
  congr 1
  by_cases hin : SN + j.val < seqLen s0 H (Region.cast Seqlens)
      ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)
  · rw [if_pos hin]
    rw [show rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (SN + j.val)
        = ∑ e : Fin BD, qRow s0 Q H sqz sqh sqm BM i e.val *
            kRow s0 K H skz skh skn (SN + j.val) e.val from rfl]
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl; intro e _; rw [if_pos hin]
  · rw [if_neg hin, mul_zero]
    apply Finset.sum_eq_zero; intro e _; rw [if_neg hin, mul_zero, mul_zero]

/-- The block-A score-lane softmax weight is exactly `mixedSparseAttnClosedForm`'s
`wBlock` term at `effScale 0.1` (including the spurious-block weight-1 path). -/
theorem msaE_scoreLaneA_eqGS
    (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (BM BN BD H NR sqz sqh sqm sqk skz skh skn skk : Nat)
    (hsqk : sqk = 1) (hskk : skk = 1)
    (s0 : BlockState) (sm_scale : ℝ) (c SN : Nat) (i : Fin BM) (j : Fin BN) :
    msaE (msaScoreLaneAGS Q K Seqlens Blocks BlockOffsets BM BN BD H NR skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0)
        s0 c SN i j)
      = (if s0.pids 0 * BM + i.val < seqLen s0 H (Region.cast Seqlens)
            ∧ SN + j.val ≤ s0.pids 0 * BM + i.val then
          Real.exp (effScale sm_scale *
            (if SN + j.val < seqLen s0 H (Region.cast Seqlens)
                ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)
              then rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (SN + j.val) else 0))
        else 0) := by
  unfold msaScoreLaneAGS
  rw [msaScoreA_dot_eqGS (hsqk := hsqk) (hskk := hskk)]
  by_cases hgate : s0.pids 0 * BM + i.val < seqLen s0 H (Region.cast Seqlens)
      ∧ SN + j.val ≤ s0.pids 0 * BM + i.val
  · rw [if_pos hgate, if_pos hgate]
    show msaE (some (0 + (sm_scale * 1.44269504) * _)) = _
    rw [msaE_some]
    rw [zero_add]
    congr 1
    unfold effScale; ring
  · rw [if_neg hgate, if_neg hgate]
    rfl

/-- The column-B `Σ_e q·K` reduces to `some (0.1·1.44269504 · rawMasked)` over the
gathered column `gcol j`; same fp16-identity argument as `msaScoreA_dot_eq`. -/
theorem msaScoreB_dot_eqGS
    (Q K : RegionName) (Blocks ColCounts : Region .nat)
    (BM BN BD H NR sqz sqh sqm sqk skz skh skn skk : Nat)
    (hsqk : sqk = 1) (hskk : skk = 1)
    (s0 : BlockState) (sm_scale : ℝ) (sv : Nat) (gcol : Fin BN → Nat) (i : Fin BM) (j : Fin BN) :
    (@Finset.sum (Fin BD) (WithBot ℝ) _ Finset.univ
        (fun e : Fin BD => Option.map₂ (· * ·)
          (FloatDType.fp16.cast FloatDType.real
            (FloatDType.real.cast FloatDType.fp16 (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale (i, e, PUnit.unit))))
          (msaKLaneBGS Blocks ColCounts BD BN NR skn (msaKPtrGS K BD H skz skh skk s0) s0 sv gcol e j)))
      = some ((sm_scale * 1.44269504) *
          (if sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
              ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
            then rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (gcol j) else 0)) := by
  have hterm : ∀ e : Fin BD, Option.map₂ (· * ·)
        (FloatDType.fp16.cast FloatDType.real
          (FloatDType.real.cast FloatDType.fp16 (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale (i, e, PUnit.unit))))
        (msaKLaneBGS Blocks ColCounts BD BN NR skn (msaKPtrGS K BD H skz skh skk s0) s0 sv gcol e j)
      = some ((sm_scale * 1.44269504) *
          (qRow s0 Q H sqz sqh sqm BM i e.val *
            (if sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
                ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
              then kRow s0 K H skz skh skn (gcol j) e.val else 0))) := by
    intro e
    have hQoff : ((s0.pids 1 / H) * sqz + (s0.pids 1 % H) * sqh)
        + (s0.pids 0 * BM + i.val) * sqm + e.val * 1
      = qoBase s0 H sqz sqh + mIndex s0 BM i * sqm + e.val := by
      unfold qoBase offZ offH mIndex; ring
    have hKoff : ((s0.pids 1 / H) * skz + (s0.pids 1 % H) * skh) + e.val * 1 + gcol j * skn
      = qoBase s0 H skz skh + gcol j * skn + e.val := by
      unfold qoBase offZ offH; ring
    subst hsqk hskk; unfold msaQValGS msaKLaneBGS msaKPtrGS
    by_cases hin : sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
        ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
    · rw [if_pos hin, if_pos hin]
      simp only [BlockState.readMemValue_real, FloatDType.cast,
        FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
        FloatDType.real_ofWithBot, WithBot.realMul, Option.map₂_some_some]
      rw [hQoff, hKoff]
      show some _ = some ((sm_scale * 1.44269504) *
        (qRow s0 Q H sqz sqh sqm BM i e.val * kRow s0 K H skz skh skn (gcol j) e.val))
      unfold qRow kRow; ring_nf
    · rw [if_neg hin, if_neg hin]
      simp only [BlockState.readMemValue_real, FloatDType.cast,
        FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot, FloatDType.fp16_toWithBot,
        FloatDType.real_ofWithBot, WithBot.realMul, Option.map₂_some_some, mul_zero]
      norm_num
  rw [show (@Finset.sum (Fin BD) (WithBot ℝ) _ Finset.univ
        (fun e : Fin BD => Option.map₂ (· * ·)
          (FloatDType.fp16.cast FloatDType.real
            (FloatDType.real.cast FloatDType.fp16 (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale (i, e, PUnit.unit))))
          (msaKLaneBGS Blocks ColCounts BD BN NR skn (msaKPtrGS K BD H skz skh skk s0) s0 sv gcol e j)))
      = @Finset.sum (Fin BD) (WithBot ℝ) _ Finset.univ
          (fun e : Fin BD => some ((sm_scale * 1.44269504) *
            (qRow s0 Q H sqz sqh sqm BM i e.val *
              (if sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
                  ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
                then kRow s0 K H skz skh skn (gcol j) e.val else 0))))
      from Finset.sum_congr rfl (fun e _ => hterm e)]
  rw [WithBot.sum_someTerm_eq_some]
  congr 1
  by_cases hin : sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
      ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
  · rw [if_pos hin]
    rw [show rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (gcol j)
        = ∑ e : Fin BD, qRow s0 Q H sqz sqh sqm BM i e.val *
            kRow s0 K H skz skh skn (gcol j) e.val from rfl]
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl; intro e _; rw [if_pos hin]
  · rw [if_neg hin, mul_zero]
    apply Finset.sum_eq_zero; intro e _; rw [if_neg hin, mul_zero, mul_zero]

/-- The column-B score-lane softmax weight is exactly `mixedSparseAttnClosedForm`'s
`wCol` term at `effScale sm_scale` (no `cols < seqlen` mask — the kernel applies none). -/
theorem msaE_scoreLaneB_eqGS
    (Q K : RegionName) (Blocks ColCounts Seqlens : Region .nat)
    (BM BN BD H NR sqz sqh sqm sqk skz skh skn skk : Nat)
    (hsqk : sqk = 1) (hskk : skk = 1)
    (s0 : BlockState) (sm_scale : ℝ) (sv : Nat) (gcol : Fin BN → Nat) (i : Fin BM) (j : Fin BN) :
    msaE (msaScoreLaneBGS Blocks ColCounts Seqlens BM BN BD H NR skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0)
        s0 sv gcol i j)
      = (if s0.pids 0 * BM + i.val < seqLen s0 H (Region.cast Seqlens)
            ∧ (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
              ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)) then
          Real.exp (effScale sm_scale *
            (if sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
                ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
              then rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (gcol j) else 0))
        else 0) := by
  unfold msaScoreLaneBGS
  rw [msaScoreB_dot_eqGS (hsqk := hsqk) (hskk := hskk)]
  by_cases hgate : s0.pids 0 * BM + i.val < seqLen s0 H (Region.cast Seqlens)
      ∧ (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
        ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0))
  · rw [if_pos hgate, if_pos hgate]
    show msaE (some (0 + (sm_scale * 1.44269504) * _)) = _
    rw [msaE_some, zero_add]
    congr 1
    unfold effScale; ring
  · rw [if_neg hgate, if_neg hgate]
    rfl

/-- The block-A value lane equals `mixedSparseAttnClosedForm`'s `vBlock` term. -/
theorem msaVblkA0_eqGS
    (V : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (BD BN H NR NS skz skh svk svn : Nat) (hsvk : svk = 1) (s0 : BlockState) (c : Nat) (j : Fin BN) (d : Fin BD) :
    msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn (msaVPtrGS V BD H skz skh svk s0) s0 c j d
      = (if msaSN0GS s0 Blocks BlockOffsets NR NS c + j.val < seqLen s0 H (Region.cast Seqlens)
            ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0) then
          vRow s0 V H skz skh svn
            (msaSN0GS s0 Blocks BlockOffsets NR NS c + j.val) d else 0) := by
  subst hsvk
  unfold msaVblkA0GS msaVLaneAGS msaVPtrGS vRow qoBase offZ offH
  by_cases hin : msaSN0GS s0 Blocks BlockOffsets NR NS c + j.val < seqLen s0 H (Region.cast Seqlens)
      ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)
  · rw [if_pos hin, if_pos hin]
    rw [show ((s0.pids 1 / H) * skz + (s0.pids 1 % H) * skh) + d.val * 1
        + (msaSN0GS s0 Blocks BlockOffsets NR NS c + j.val) * svn
      = (s0.pids 1 / H) * skz + (s0.pids 1 % H) * skh
        + (msaSN0GS s0 Blocks BlockOffsets NR NS c + j.val) * svn + d.val from by ring]
    simp [BlockState.readMemValue_real]
  · rw [if_neg hin, if_neg hin]; norm_num

/-- The column-B value lane equals the gathered-column `vRow`. -/
theorem msaVblkB0_eqGS
    (V : RegionName) (Blocks ColCounts : Region .nat)
    (BD BN H NR skz skh svk svn : Nat) (hsvk : svk = 1) (s0 : BlockState) (sv : Nat) (gcol : Fin BN → Nat) (j : Fin BN) (d : Fin BD) :
    (msaVLaneBGS Blocks ColCounts BD BN NR svn (msaVPtrGS V BD H skz skh svk s0) s0 sv gcol j d).unbotD 0
      = (if sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
            ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0) then
          vRow s0 V H skz skh svn (gcol j) d else 0) := by
  subst hsvk
  unfold msaVLaneBGS msaVPtrGS vRow qoBase offZ offH
  by_cases hin : sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
      ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
  · rw [if_pos hin, if_pos hin]
    rw [show ((s0.pids 1 / H) * skz + (s0.pids 1 % H) * skh) + d.val * 1 + gcol j * svn
      = (s0.pids 1 / H) * skz + (s0.pids 1 % H) * skh + gcol j * svn + d.val from by ring]
    simp [BlockState.readMemValue_real]
  · rw [if_neg hin, if_neg hin]; norm_num

/-- A `Fin 64`-sum of column terms that vanish for `j ≥ numCols` collapses to the
`Fin numCols` sum (given `numCols ≤ 64`). The terms depend on `j` only via `j.val`,
so they are presented as a `Nat → ℝ` function `f`. -/
theorem msa_col_sum_collapseGS (BN numCols : Nat) (h : numCols ≤ BN)
    (f : Nat → ℝ) (hz : ∀ j, numCols ≤ j → f j = 0) :
    (∑ j : Fin BN, f j.val) = ∑ c : Fin numCols, f c.val := by
  rw [Fin.sum_univ_eq_sum_range (fun k => f k) BN,
      Fin.sum_univ_eq_sum_range (fun k => f k) numCols]
  rw [← Finset.sum_range_add_sum_Ico (fun k => f k) h]
  rw [Finset.sum_Ico_eq_sum_range]
  rw [show (∑ k ∈ Finset.range (BN - numCols), f (numCols + k)) = 0 from
    Finset.sum_eq_zero (fun k _ => hz _ (by omega))]
  rw [add_zero]

/-- `msaSN0` (the kernel's masked block start) equals the closed form's
`blockStartN` at the python `(NUM_ROWS, NNZ_S) = (2, 4)` layout. -/
theorem msaSN0GS_eq_blockStartN (s0 : BlockState) (Blocks BlockOffsets : Region .nat) (NR NS c : Nat) :
    msaSN0GS s0 Blocks BlockOffsets NR NS c
      = blockStartN s0 BlockOffsets NR NS
          (s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)) c := by
  unfold msaSN0GS blockStartN; rfl

/-- The gathered column at sv=0, lane `j < numCols`, is `colKeyGlobal j`. -/
theorem msaGcol0_eq_colKeyGlobalGS (BN NR NV : Nat) (s0 : BlockState) (Cols ColCounts : Region .nat)
    (j : Nat) (hj : j < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0))
    (hjBN : j < BN) :
    msaGcol0GS s0 Cols ColCounts BN NR NV 0 ⟨j, hjBN⟩
      = colKeyGlobal s0 Cols NR NV j := by
  unfold msaGcol0GS msaColLaneBGS colKeyGlobal
  have h0 : (0 : Nat) < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0) :=
    by omega
  rw [if_pos h0]
  show s0.readMemValue .nat (Region.cast Cols) _ = s0.readMemValue .nat (Region.cast Cols) _
  congr 1









/-! ## FULLY-GENERAL catFold connector + seeded ratio + offset injectivity (symbolic strides + layout) -/

set_option maxHeartbeats 4000000 in
theorem msa_catFold_eq_closedFormGS
    (Q K V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk svz svh svn svk : Nat)
    (hsqk : sqk = 1) (hskk : skk = 1) (hsvk : svk = 1) (hvz : svz = skz) (hvh : svh = skh)
    (hBN16 : 16 ≤ BN) (s0 : BlockState) (sm_scale : ℝ) (i : Fin BM) (d : Fin BD)
    (hNCBN : s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0) ≤ BN) :
    msaNumerUpto BM BN BD
        (msaCatScore BM BN 8
          (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0)
          (msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0))
        (msaCatVblk BN BD 8
          (msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn (msaVPtrGS V BD H skz skh svk s0) s0)
          (msaVblkB0GS V Seqlens Blocks BlockOffsets ColCounts Cols BD BN NR NV svn (msaVPtrGS V BD H skz skh svk s0) s0))
        9 i d
      / msaDenomUpto BM BN
        (msaCatScore BM BN 8
          (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0)
          (msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0))
        9 i
    = mixedSparseAttnClosedForm s0 Q K V BlockOffsets Cols H
        sqz sqh sqm skz skh skn svz svh svn NR NS NV
        (s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0))
        (s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0))
        (seqLen s0 H (Region.cast Seqlens)) BD BM BN sm_scale i d := by
  subst svz svh
  set NB := s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0) with hNB
  set NC := s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0) with hNC
  set SL := seqLen s0 H (Region.cast Seqlens) with hSL
  set scoreA := msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0 with hscA
  set scoreB := msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0 with hscB
  set vblkA := msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn (msaVPtrGS V BD H skz skh svk s0) s0 with hvbA
  set vblkB := msaVblkB0GS V Seqlens Blocks BlockOffsets ColCounts Cols BD BN NR NV svn (msaVPtrGS V BD H skz skh svk s0) s0 with hvbB
  set cat := msaCatScore BM BN 8 scoreA scoreB with hcat
  set catV := msaCatVblk BN BD 8 vblkA vblkB with hcatV
  -- abbreviations matching the closed form's per-lane terms
  set SN : Nat → Nat := fun b => blockStartN s0 BlockOffsets NR NS NB b with hSNd
  -- per-block-A weight (the closed form's `wBlock`)
  have hwA : ∀ (b : Fin 8) (j : Fin BN), msaE (scoreA b.val i j)
      = (if s0.pids 0 * BM + i.val < SL ∧ SN b.val + j.val ≤ s0.pids 0 * BM + i.val then
          Real.exp (effScale sm_scale *
            (if SN b.val + j.val < SL ∧ b.val < NB
              then rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (SN b.val + j.val) else 0))
        else 0) := by
    intro b j
    rw [hscA, msaScoreA0GS, msaE_scoreLaneA_eqGS (hsqk := hsqk) (hskk := hskk), msaSN0GS_eq_blockStartN]
  -- per-block-A value (the closed form's `vBlock`)
  have hvA : ∀ (b : Fin 8) (j : Fin BN), vblkA b.val j d
      = (if SN b.val + j.val < SL ∧ b.val < NB then
          vRow s0 V H skz skh svn (SN b.val + j.val) d else 0) := by
    intro b j
    rw [hvbA, msaVblkA0_eqGS (hsvk := hsvk)]
    rw [show msaSN0GS s0 Blocks BlockOffsets NR NS b.val = SN b.val from by
      rw [hSNd]; exact msaSN0GS_eq_blockStartN s0 Blocks BlockOffsets NR NS b.val]
  -- per-column-B weight (the closed form's `wCol`, as a `Nat → ℝ` over the gathered col)
  set gcol : Fin BN → Nat := msaGcol0GS s0 Cols ColCounts BN NR NV 0 with hgcol
  have hwB : ∀ (j : Fin BN), msaE (scoreB 0 i j)
      = (if s0.pids 0 * BM + i.val < SL ∧ (j.val < NC ∧ (0:Nat) < NC) then
          Real.exp (effScale sm_scale *
            (if j.val < NC ∧ (0:Nat) < NC
              then rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (gcol j) else 0))
        else 0) := by
    intro j
    rw [hscB, msaScoreB0GS]
    have h0 : (0 : Nat) * BN = 0 := by ring
    rw [h0, msaE_scoreLaneB_eqGS (hsqk := hsqk) (hskk := hskk)]
    simp only [Nat.zero_add, ← hSL, ← hNC, ← hgcol]
  have hvB : ∀ (j : Fin BN), vblkB 0 j d
      = (if j.val < NC ∧ (0:Nat) < NC then vRow s0 V H skz skh svn (gcol j) d else 0) := by
    intro j
    rw [hvbB, msaVblkB0GS]
    have h0 : (0 : Nat) * BN = 0 := by ring
    rw [h0, ← hgcol]
    have := msaVblkB0_eqGS V Blocks ColCounts BD BN H NR skz skh svk svn hsvk s0 0 gcol j d
    simp only [Nat.zero_add, ← hNC] at this ⊢
    exact this
  -- numerator split: block-A (Fin 8) + 1 column-B term
  have hnumBlk : (∑ l ∈ Finset.range 8, ∑ j : Fin BN, msaE (cat l i j) * catV l j d)
      = ∑ b : Fin 8, ∑ j : Fin BN, msaE (scoreA b.val i j) * vblkA b.val j d := by
    rw [Fin.sum_univ_eq_sum_range (fun b => ∑ j : Fin BN, msaE (scoreA b i j) * vblkA b j d) 8]
    apply Finset.sum_congr rfl; intro b hb
    rw [Finset.mem_range] at hb
    apply Finset.sum_congr rfl; intro j _
    rw [hcat, msaCatScore, if_pos hb, hcatV, msaCatVblk, if_pos hb]
  have hnum : msaNumerUpto BM BN BD cat catV 9 i d
      = (∑ b : Fin 8, ∑ j : Fin BN, msaE (scoreA b.val i j) * vblkA b.val j d)
        + ∑ j : Fin BN, msaE (scoreB 0 i j) * vblkB 0 j d := by
    rw [msaNumerUpto, show (9 : Nat) = 8 + 1 from rfl, Finset.sum_range_succ, hnumBlk]
    refine congrArg (_ + ·) ?_
    apply Finset.sum_congr rfl; intro j _
    rw [hcat, msaCatScore, if_neg (by omega), hcatV, msaCatVblk, if_neg (by omega)]
  have hdenBlk : (∑ l ∈ Finset.range 8, ∑ j : Fin BN, msaE (cat l i j))
      = ∑ b : Fin 8, ∑ j : Fin BN, msaE (scoreA b.val i j) := by
    rw [Fin.sum_univ_eq_sum_range (fun b => ∑ j : Fin BN, msaE (scoreA b i j)) 8]
    apply Finset.sum_congr rfl; intro b hb
    rw [Finset.mem_range] at hb
    apply Finset.sum_congr rfl; intro j _
    rw [hcat, msaCatScore, if_pos hb]
  have hden : msaDenomUpto BM BN cat 9 i
      = (∑ b : Fin 8, ∑ j : Fin BN, msaE (scoreA b.val i j))
        + ∑ j : Fin BN, msaE (scoreB 0 i j) := by
    rw [msaDenomUpto, show (9 : Nat) = 8 + 1 from rfl, Finset.sum_range_succ, hdenBlk]
    refine congrArg (_ + ·) ?_
    apply Finset.sum_congr rfl; intro j _
    rw [hcat, msaCatScore, if_neg (by omega)]
  have hNCle : NC ≤ BN := hNCBN
  rw [hnum, hden]
  -- expose the closed form's `numer / denom`; fold raw terms back to the set abbrevs
  simp only [mixedSparseAttnClosedForm, mIndex, ← hNB, ← hNC, ← hSL, ← hSNd]
  -- rewrite LHS block/column terms to closed-form per-lane shapes
  simp only [hwA, hvA, hwB, hvB]
  -- the gated column summands (numerator/denominator) as `Nat → ℝ` functions
  set numCol : Nat → ℝ := fun jn =>
    (if s0.pids 0 * BM + i.val < SL ∧ (jn < NC ∧ (0:Nat) < NC) then
        Real.exp (effScale sm_scale *
          (if jn < NC ∧ (0:Nat) < NC
            then rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (colKeyGlobal s0 Cols NR NV jn) else 0))
      else 0) *
      (if jn < NC ∧ (0:Nat) < NC then vRow s0 V H skz skh svn (colKeyGlobal s0 Cols NR NV jn) d else 0)
    with hnumCol
  set denCol : Nat → ℝ := fun jn =>
    (if s0.pids 0 * BM + i.val < SL ∧ (jn < NC ∧ (0:Nat) < NC) then
        Real.exp (effScale sm_scale *
          (if jn < NC ∧ (0:Nat) < NC
            then rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (colKeyGlobal s0 Cols NR NV jn) else 0))
      else 0)
    with hdenCol
  -- gcol-keyed column sum = colKeyGlobal-keyed sum (for j < NC); both vanish for j ≥ NC
  have hgcConv : ∀ j : Fin BN, j.val < NC → gcol j = colKeyGlobal s0 Cols NR NV j.val := by
    intro j hj; rw [hgcol]
    exact msaGcol0_eq_colKeyGlobalGS BN NR NV s0 Cols ColCounts j.val (by rw [← hNC]; exact hj) j.isLt
  have hcolNum :
      (∑ j : Fin BN,
        (if s0.pids 0 * BM + i.val < SL ∧ (j.val < NC ∧ (0:Nat) < NC) then
            Real.exp (effScale sm_scale *
              (if j.val < NC ∧ (0:Nat) < NC
                then rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (gcol j) else 0))
          else 0) *
          (if j.val < NC ∧ (0:Nat) < NC then vRow s0 V H skz skh svn (gcol j) d else 0))
        = ∑ c : Fin NC, numCol c.val := by
    rw [← msa_col_sum_collapseGS BN NC hNCle numCol
      (fun jn hjn => by rw [hnumCol]; simp only [if_neg (by omega : ¬(jn < NC ∧ (0:Nat) < NC)), mul_zero])]
    apply Finset.sum_congr rfl; intro j _
    rw [hnumCol]
    by_cases hj : j.val < NC
    · rw [hgcConv j hj]
    · simp only [if_neg (by omega : ¬(j.val < NC ∧ (0:Nat) < NC)), mul_zero]
  have hcolDen :
      (∑ j : Fin BN,
        (if s0.pids 0 * BM + i.val < SL ∧ (j.val < NC ∧ (0:Nat) < NC) then
            Real.exp (effScale sm_scale *
              (if j.val < NC ∧ (0:Nat) < NC
                then rawScore s0 Q K H sqz sqh sqm skz skh skn BD BM i (gcol j) else 0))
          else 0))
        = ∑ c : Fin NC, denCol c.val := by
    rw [← msa_col_sum_collapseGS BN NC hNCle denCol
      (fun jn hjn => by rw [hdenCol]; simp only []; rw [if_neg (by omega : ¬(s0.pids 0 * BM + i.val < SL ∧ (jn < NC ∧ (0:Nat) < NC)))])]
    apply Finset.sum_congr rfl; intro j _
    rw [hdenCol]; simp only []
    by_cases hj : j.val < NC
    · rw [hgcConv j hj]
    · rw [if_neg (by omega : ¬(s0.pids 0 * BM + i.val < SL ∧ (j.val < NC ∧ (0:Nat) < NC)))]
      simp only [if_neg (by omega : ¬(s0.pids 0 * BM + i.val < SL ∧ (j.val < NC ∧ (0:Nat) < NC)))]
  rw [hcolNum, hcolDen]
  -- final: match Fin 8 block sums + Fin NC column sums to closed form
  congr 1
  · congr 1
    apply Finset.sum_congr rfl; intro c _
    rw [hnumCol]; simp only []
    -- closed form `wCol c * vRow(colKeyGlobal c)`; gate `c.val < NC ∧ 0 < NC` is true
    have hcNC : c.val < NC := c.isLt
    have h0NC : (0:Nat) < NC := by omega
    simp only [if_pos (And.intro hcNC h0NC)]
    by_cases hrow : s0.pids 0 * BM + i.val < SL
    · rw [if_pos hrow, if_pos (And.intro hrow (And.intro hcNC h0NC))]
    · rw [if_neg hrow, if_neg (by tauto : ¬(s0.pids 0 * BM + i.val < SL ∧ (c.val < NC ∧ (0:Nat) < NC)))]
  · congr 1
    apply Finset.sum_congr rfl; intro c _
    rw [hdenCol]; simp only []
    have hcNC : c.val < NC := c.isLt
    have h0NC : (0:Nat) < NC := by omega
    by_cases hrow : s0.pids 0 * BM + i.val < SL
    · rw [if_pos (And.intro hrow (And.intro hcNC h0NC)), if_pos hrow, if_pos (And.intro hcNC h0NC)]
    · rw [if_neg (by tauto : ¬(s0.pids 0 * BM + i.val < SL ∧ (c.val < NC ∧ (0:Nat) < NC))), if_neg hrow]


noncomputable abbrev msaCatScore0GS
    (Q K V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk : Nat) (s0 : BlockState) (sm_scale : ℝ := 0.1) : Nat → Fin BM → Fin BN → WithBot ℝ :=
  msaCatScore BM BN 8
    (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0)
    (msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0)

/-- **Seeded-ratio bridge.** The two-phase seeded online-softmax ratio at the
Loop-B end (`c = 1`, seeded by Loop-A's `bF = 8` finals) equals
`mixedSparseAttnClosedForm`, given the cat denominator is positive at this lane. -/
theorem msaSeeded_ratio_eq_closedFormGS
    (Q K V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk svz svh svn svk : Nat)
    (hsqk : sqk = 1) (hskk : skk = 1) (hsvk : svk = 1) (hvz : svz = skz) (hvh : svh = skh)
    (hBN16 : 16 ≤ BN) (s0 : BlockState) (sm_scale : ℝ) (i : Fin BM) (d : Fin BD)
    (hNCBN : s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0) ≤ BN)
    (hpos : 0 < msaDenomUpto BM BN
      (msaCatScore0GS Q K V Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk s0 sm_scale) 9 i) :
    (msaOPartialSeed BM BN BD
        (msaMPartial BM BN (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0) 8)
        (msaLPartial BM BN (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0) 8)
        (fun ii dd => msaOPartial BM BN BD
          (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0)
          (msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn (msaVPtrGS V BD H skz skh svk s0) s0) 8 ii dd)
        (msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0)
        (msaVblkB0GS V Seqlens Blocks BlockOffsets ColCounts Cols BD BN NR NV svn (msaVPtrGS V BD H skz skh svk s0) s0) 1 i d)
      / (msaLPartialSeed BM BN
        (msaMPartial BM BN (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0) 8)
        (msaLPartial BM BN (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0) 8)
        (msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0) 1 i)
    = mixedSparseAttnClosedForm s0 Q K V BlockOffsets Cols H
        sqz sqh sqm skz skh skn svz svh svn NR NS NV
        (s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0))
        (s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0))
        (seqLen s0 H (Region.cast Seqlens)) BD BM BN sm_scale i d := by
  set scoreA := msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0 with hscA
  set scoreB := msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0 with hscB
  set vblkA := msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn (msaVPtrGS V BD H skz skh svk s0) s0 with hvbA
  set vblkB := msaVblkB0GS V Seqlens Blocks BlockOffsets ColCounts Cols BD BN NR NV svn (msaVPtrGS V BD H skz skh svk s0) s0 with hvbB
  rw [msaOPartialSeed_eq_cat BM BN BD 8 scoreA scoreB vblkA vblkB _ _ 1 i d rfl rfl,
      msaLPartialSeed_eq_cat BM BN 8 scoreA scoreB _ 1 i rfl,
      show (8 + 1 : Nat) = 9 from rfl]
  rw [msaPartial_ratio_collapse BM BN BD _ _ 9 i d (by
    rw [show msaCatScore BM BN 8 scoreA scoreB
        = msaCatScore0GS Q K V Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk s0 sm_scale from rfl]
    exact hpos)]
  exact msa_catFold_eq_closedFormGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk svz svh svn svk hsqk hskk hsvk hvz hvh hBN16 s0 sm_scale i d hNCBN


/-- General output-offset injectivity with symbolic output strides: offset
`= base + m·som + d·sok`. With channel stride `sok = 1` and `BD ≤ som` (the row
stride covers the channel block — the natural contiguous layout `som = head_dim ≥
BD`), the per-lane offset is injective on `TileIndex [BM, BD]`. -/
theorem mixed_sparse_attention_offset_injectiveGS
    (BM BD H sqz sqh som sok : Nat) (hsok : sok = 1) (hBDsom : BD ≤ som) (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [BM, BD] =>
        outOffset s H sqz sqh som sok BM idx) := by
  subst hsok
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp only [outOffset, offZ, offH, mIndex, dIndex, mul_one] at h
  have hdas : da < som := lt_of_lt_of_le hda hBDsom
  have hdbs : db < som := lt_of_lt_of_le hdb hBDsom
  -- generalize the two row-index sums and the shared prefix to opaque atoms
  generalize hA : s.pids 0 * BM + ma = A at h
  generalize hB : s.pids 0 * BM + mb = B at h
  generalize hP : s.pids 1 / H * sqz + s.pids 1 % H * sqh = P at h
  -- h : P + A * som + da = P + B * som + db
  have key : A * som + da = B * som + db := by omega
  have hAB : A = B := by
    rcases Nat.lt_trichotomy A B with hlt | heq | hgt
    · exfalso
      have hle : (A + 1) * som ≤ B * som := Nat.mul_le_mul_right som (by omega)
      have hexp : A * som + som ≤ B * som := by rw [Nat.add_mul, one_mul] at hle; exact hle
      omega
    · exact heq
    · exfalso
      have hle : (B + 1) * som ≤ A * som := Nat.mul_le_mul_right som (by omega)
      have hexp : B * som + som ≤ A * som := by rw [Nat.add_mul, one_mul] at hle; exact hle
      omega
  have hm : ma = mb := by
    have : s.pids 0 * BM + ma = s.pids 0 * BM + mb := by rw [hA, hB]; exact hAB
    omega
  subst hm
  have hd : da = db := by
    have : A * som + da = A * som + db := by rw [hAB] at key ⊢; exact key
    omega
  subst hd; rfl

/-! ## FULLY-GENERAL postLoop exec (symbolic strides + layout) -/

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in


theorem msaPostLoop_evalGS
    (Q K V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (Out : RegionName) (BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk svz svh svn svk som sok : Nat)
    (hsqk : sqk = 1) (hskk : skk = 1) (hsvk : svk = 1) (hvz : svz = skz) (hvh : svh = skh)
    (hsok : sok = 1) (hBDsom : BD ≤ som)
    (hBN16 : 16 ≤ BN) (s0 s : BlockState) (sm_scale : ℝ)
    (hNCBN : s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0) ≤ BN)
    (hpos : ∀ i : Fin BM, s0.pids 0 * BM + i.val < seqLen s0 H (Region.cast Seqlens) →
      0 < msaDenomUpto BM BN
        (msaCatScore0GS Q K V Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk s0 sm_scale) 9 i)
    (hinv : msaInvariantBGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
      (msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0)
      (msaVblkB0GS V Seqlens Blocks BlockOffsets ColCounts Cols BD BN NR NV svn (msaVPtrGS V BD H skz skh svk s0) s0)
      (msaMPartial BM BN (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0) 8)
      (msaLPartial BM BN (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0) 8)
      (fun ii dd => msaOPartial BM BN BD
        (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0)
        (msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn (msaVPtrGS V BD H skz skh svk s0) s0) 8 ii dd)
      (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) (msaVPtrGS V BD H skz skh svk s0) (msaOPtrGS Out BM BD H sqz sqh som sok s0) s0 1 s) :
    ∃ sP, stepStmts (msaPostLoopGS BM BD) s = some sP
      ∧ ∀ idx : TileIndex [BM, BD],
          sP.readMemValue .fp16 Out (outOffset s0 H sqz sqh som sok BM idx)
            = if s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens) then
                (some (mixedSparseAttnClosedForm s0 Q K V BlockOffsets Cols H
                  sqz sqh sqm skz skh skn svz svh svn NR NS NV
                  (s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0))
                  (s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0))
                  (seqLen s0 H (Region.cast Seqlens)) BD BM BN sm_scale idx.1 (dIndex idx)) : WithBot ℝ)
              else s.readMemValue .fp16 Out (outOffset s0 H sqz sqh som sok BM idx) := by
  obtain ⟨hpids, hmem, hundef, hsm, hoh, hseq, hoffm, hoffn, hoffd, hnb, hnc,
    hbp, hcp, hq, hkp, hvp, hop, hmmask, hmnb, hmnc, hmi, hli, hacc⟩ := hinv
  set scoreA := msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0 with hscA
  set scoreB := msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0 with hscB
  set vblkA := msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn (msaVPtrGS V BD H skz skh svk s0) s0 with hvbA
  set vblkB := msaVblkB0GS V Seqlens Blocks BlockOffsets ColCounts Cols BD BN NR NV svn (msaVPtrGS V BD H skz skh svk s0) s0 with hvbB
  set mA := msaMPartial BM BN scoreA 8 with hmA
  set lA := msaLPartial BM BN scoreA 8 with hlA
  set oA := fun ii dd => msaOPartial BM BN BD scoreA vblkA 8 ii dd with hoA
  -- the per-lane ratio register value after the divide
  set ratio : TileIndex [BM, BD] → ℝ := fun idx =>
    msaOPartialSeed BM BN BD mA lA oA scoreB vblkB 1 idx.1 idx.2.1
      / msaLPartialSeed BM BN mA lA scoreB 1 idx.1 with hratio
  unfold msaPostLoopGS
  -- stmt 0: acc = acc / l_i[:, None]
  have hexpand : @evalOp .real [BM, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "l_i")) s
      = some (Tile.expandDim ⟨1, by simp⟩
          (⟨fun idx : TileIndex [BM] =>
            (some (msaLPartialSeed BM BN mA lA scoreB 1 idx.1) : WithBot ℝ)⟩ : Tile .real [BM])) :=
    evalOp_expandDim_ref_of_regs .real [BM] ⟨1, by simp⟩ "l_i" s _ hli
  have hdiv : evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BM] "l_i"))) s
      = some (⟨fun idx : TileIndex [BM, BD] => (some (ratio idx) : WithBot ℝ)⟩ : Tile .real [BM, BD]) := by
    rw [evalOp_div]
    simp only [evalOp_ref, hacc, hexpand, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some (Tile.ext (fun idx => ?_))
    obtain ⟨ir, dd, u⟩ := idx
    simp only [Tile.bop_data, Tile.bop, Tile.expandDim, NumericDType.div,
      Broadcast.leftIndex, Broadcast.rightIndex, TileShape.dropInsertedIndex, hratio]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hdiv)]
  set s1 := s.setReg "acc" .real [BM, BD]
    (⟨fun idx : TileIndex [BM, BD] => (some (ratio idx) : WithBot ℝ)⟩ : Tile .real [BM, BD]) with hs1d
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1acc : s1.regs .real [BM, BD] "acc" = some
      (⟨fun idx : TileIndex [BM, BD] => (some (ratio idx) : WithBot ℝ)⟩ : Tile .real [BM, BD]) := by
    rw [hs1d, BlockState.setReg_same]
  have hs1op : s1.regs .ptr [BM, BD] "o_ptrs" = some
      (⟨fun idx : TileIndex [BM, BD] => msaOPtrGS Out BM BD H sqz sqh som sok s0 idx⟩ : Tile .ptr [BM, BD]) := e1 (by decide) hop
  have hs1mm : s1.regs .bool [BM, 1] "m_mask" = some
      (⟨fun idx : TileIndex [BM, 1] =>
        decide (s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))⟩ : Tile .bool [BM, 1]) :=
    e1 (by decide) hmmask
  have hs1mem : s1.mem = s0.mem := by
    funext rg o; rw [hs1d, BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o
  have hs1pids : s1.pids = s0.pids := by rw [hs1d, BlockState.setReg_pids]; exact hpids
  -- stmt 1: masked fp16 store of acc at o_ptrs
  have hstore : stepStmt (Stmt.store FloatDType.fp16.toTileDType [BM, BD]
      (MemAccess.ptr (Op.ref .ptr [BM, BD] "o_ptrs"))
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BD] "acc"))
      (MaskOpt.mask (Op.remap [BM, BD] Broadcast.nil.consL.consSame.leftIndex
        (Op.ref .bool [BM, 1] "m_mask")))) s1
      = some ((TileShape.allIndices [BM, BD]).foldl
          (fun acc idx =>
            if s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens) then
              acc.writeMemTyped FloatDType.fp16.toTileDType Out (outOffset s0 H sqz sqh som sok BM idx)
                (FloatDType.real.cast FloatDType.fp16 (some (ratio idx)))
            else acc) s1) := by
    have hval : evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BM, BD] "acc")) s1
        = some (⟨fun idx : TileIndex [BM, BD] =>
            FloatDType.real.cast FloatDType.fp16 (some (ratio idx))⟩ : Tile FloatDType.fp16.toTileDType [BM, BD]) := by
      rw [evalOp_castFloat]
      erw [evalOp_ref, hs1acc]
      rfl
    have hmask : @evalOp .bool [BM, BD] (Op.remap [BM, BD] Broadcast.nil.consL.consSame.leftIndex
          (Op.ref .bool [BM, 1] "m_mask")) s1
        = some (Tile.remap Broadcast.nil.consL.consSame.leftIndex
            (⟨fun idx : TileIndex [BM, 1] =>
              decide (s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))⟩ : Tile .bool [BM, 1])) := by
      rw [msa_evalOp_remap]
      erw [evalOp_ref, hs1mm]
      rfl
    have hptr : evalOp (Op.ref .ptr [BM, BD] "o_ptrs") s1
        = some (⟨fun idx : TileIndex [BM, BD] => msaOPtrGS Out BM BD H sqz sqh som sok s0 idx⟩ : Tile .ptr [BM, BD]) := by
      rw [evalOp_ref, hs1op]
    unfold stepStmt
    simp only [hval, hmask, hptr, Option.bind, Option.map]
    refine congrArg some ?_
    congr 1
    funext acc idx
    obtain ⟨ir, dd, u⟩ := idx
    simp only [Tile.cop_data, Tile.remap, Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.dropInsertedIndex, decide_eq_true_eq, outOffset, offZ, offH, mIndex, dIndex,
      msaOPtrGS, mul_one]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hinj : Function.Injective (fun idx : TileIndex [BM, BD] =>
      outOffset s0 H sqz sqh som sok BM idx) :=
    mixed_sparse_attention_offset_injectiveGS BM BD H sqz sqh som sok hsok hBDsom s0
  rw [msa_fp16_scatter_readback s1
    (fun idx : TileIndex [BM, BD] => outOffset s0 H sqz sqh som sok BM idx)
    (fun idx : TileIndex [BM, BD] => FloatDType.real.cast FloatDType.fp16 (some (ratio idx)))
    (fun idx : TileIndex [BM, BD] => s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens))
    hinj idx]
  by_cases hlt : s0.pids 0 * BM + idx.1.val < seqLen s0 H (Region.cast Seqlens)
  · rw [if_pos hlt, if_pos hlt]
    obtain ⟨ir, dd, u⟩ := idx
    -- fp16 round-trip is identity: stored cell value = some ratio = some closedForm
    rw [show FloatDType.fp16.ofReal (FloatDType.fp16.storeValue
          (FloatDType.real.cast FloatDType.fp16 (some (ratio (ir, dd, u)))))
        = (some (ratio (ir, dd, u)) : WithBot ℝ) from rfl]
    rw [hratio]
    exact congrArg some
      (msaSeeded_ratio_eq_closedFormGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk svz svh svn svk hsqk hskk hsvk hvz hvh hBN16 s0 sm_scale ir dd
        hNCBN (hpos ir hlt))
  · rw [if_neg hlt, if_neg hlt]
    simp only [BlockState.readMemValue, BlockState.readMemAs, hs1mem, hmem]




/-! ## FULLY-GENERAL streaming exec (symbolic strides + layout) -/

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
theorem msa_execGS
    (Q K V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (Out : RegionName) (BM BN BD : Nat) (hBN : 0 < BN) (hBN16 : 16 ≤ BN)
    (NCTX Zc H NR NS NV sqz sqh sqm sqk skz skh skn skk svz svh svn svk soz soh som sok : Nat)
    (hsqk : sqk = 1) (hskk : skk = 1) (hsvk : svk = 1) (hvz : svz = skz) (hvh : svh = skh)
    (hsok : sok = 1) (hBDsom : BD ≤ som)
    (s : BlockState) (sm_scale : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hactive : s.pids 0 * BM < seqLen s H (Region.cast Seqlens))
    (hNCBN : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * NR + s.pids 0) ≤ BN)
    (hpos : ∀ i : Fin BM, s.pids 0 * BM + i.val < seqLen s H (Region.cast Seqlens) →
      0 < msaDenomUpto BM BN
        (msaCatScore0GS Q K V Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk s sm_scale) 9 i) :
    ∃ sF, stepStmts ((mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        (sm_scale : ℝ) Blocks BlockOffsets ColCounts Cols Out
        sqz sqh sqm sqk skz skh skn skk svz svh svn svk soz soh som sok
        Zc H NCTX NR NS NV BM BN BD FloatDType.fp16).toAlgKernel.body) s = some sF
      ∧ ∀ idx : TileIndex [BM, BD],
          s.pids 0 * BM + idx.1.val < seqLen s H (Region.cast Seqlens) →
            sF.readMemValue .fp16 Out (outOffset s H sqz sqh som sok BM idx)
              = (some (mixedSparseAttnClosedForm s Q K V BlockOffsets Cols H
                  sqz sqh sqm skz skh skn svz svh svn NR NS NV
                  (s.readMemValue .nat (Region.cast Blocks) (s.pids 1 * NR + s.pids 0))
                  (s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * NR + s.pids 0))
                  (seqLen s H (Region.cast Seqlens)) BD BM BN sm_scale idx.1 (dIndex idx)) : WithBot ℝ) := by
  rw [msa_body_splitGS Q K V Out Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NCTX Zc NR NS NV sqz sqh sqm sqk skz skh skn skk svz svh svn svk soz soh som sok sm_scale]
  -- preLoop = [3 outer] ++ msaSetup; the whole preLoop runs to s0 (invariantA 0).
  obtain ⟨s0, hpre, hinv0⟩ := msaPreLoop_evalGS s Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV
    (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale) (msaKPtrGS K BD H skz skh skk s) s)
    (msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn (msaVPtrGS V BD H skz skh svk s) s) sm_scale hundef
  -- the three outer statements, stepped explicitly to a concrete state s3
  have h0 : stepStmt (Stmt.assign .nat [] "start_m" (Op.programId 0)) s
      = some (s.setReg "start_m" .nat [] (Tile.scalar (s.pids 0))) :=
    stepStmt_assign_eq_some (evalOp_programId 0 s)
  set sA := s.setReg "start_m" .nat [] (Tile.scalar (s.pids 0)) with hsAd
  have hpidsA : sA.pids 1 = s.pids 1 := by rw [hsAd, BlockState.setReg_pids]
  have h1 : stepStmt (Stmt.assign .nat [] "off_hz" (Op.programId 1)) sA
      = some (sA.setReg "off_hz" .nat [] (Tile.scalar (s.pids 1))) := by
    rw [← hpidsA]; exact stepStmt_assign_eq_some (evalOp_programId 1 sA)
  set sB := sA.setReg "off_hz" .nat [] (Tile.scalar (s.pids 1)) with hsBd
  have hseqEval : evalOp (Op.load .nat (MemAccess.region Seqlens
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)))
        MaskOpt.none) sB
      = some (Tile.scalar (seqLen s H (Region.cast Seqlens))) := by
    simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
      BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
      not_false_eq_true, Option.bind_eq_bind, Option.bind_some, Option.pure_def]
    refine congrArg some ?_; ext idx
    simp only [Tile.scalar, Tile.scalar_data, castTile_self, Tile.bop, Tile.bop_data,
      Broadcast.leftIndex, Broadcast.rightIndex, IntegralDType.floorDiv, seqLen, offZ,
      BlockState.readMemValue, BlockState.readMemTyped, hsBd, hsAd,
      BlockState.setReg_mem, BlockState.setReg_pids, if_true, if_pos]
  have h2 : stepStmt (Stmt.assign .nat [] "seqlen"
      (Op.load .nat (MemAccess.region Seqlens
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)))
        MaskOpt.none)) sB
      = some (sB.setReg "seqlen" .nat [] (Tile.scalar (seqLen s H (Region.cast Seqlens)))) :=
    stepStmt_assign_eq_some hseqEval
  set s3 := sB.setReg "seqlen" .nat [] (Tile.scalar (seqLen s H (Region.cast Seqlens))) with hs3d
  -- the 3 outer statements run s → s3
  have h3 : stepStmts [ Stmt.assign .nat [] "start_m" (Op.programId 0),
      Stmt.assign .nat [] "off_hz" (Op.programId 1),
      Stmt.assign .nat [] "seqlen"
        (Op.load .nat (MemAccess.region Seqlens
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)))
          MaskOpt.none) ] s = some s3 := by
    rw [stepStmts.cons_some h0, stepStmts.cons_some h1, stepStmts.cons_some h2, stepStmts.nil]
  -- setup runs s3 → s0 (from the full preLoop run, split at s3)
  have hsetup : stepStmts (msaSetupGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV sm_scale) s3
      = some s0 := by
    have hsplit := stepStmts.append_some h3 (l2 := msaSetupGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV sm_scale)
    rw [show ([ Stmt.assign .nat [] "start_m" (Op.programId 0),
        Stmt.assign .nat [] "off_hz" (Op.programId 1),
        Stmt.assign .nat [] "seqlen"
          (Op.load .nat (MemAccess.region Seqlens
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)))
            MaskOpt.none) ]
      ++ msaSetupGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV sm_scale)
      = msaPreLoopGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV sm_scale from rfl] at hsplit
    rw [← hsplit]; exact hpre
  -- s3 pids/mem agree with s
  have hs3pids : s3.pids = s.pids := by
    rw [hs3d, BlockState.setReg_pids, hsBd, BlockState.setReg_pids, hsAd, BlockState.setReg_pids]
  have hs3start : s3.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0)) := by
    rw [hs3d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
      hsBd, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hsAd, BlockState.setReg_same]
  have hs3seq : s3.regs .nat [] "seqlen" = some (Tile.scalar (seqLen s H (Region.cast Seqlens))) := by
    rw [hs3d, BlockState.setReg_same]
  -- active guard: boolNot (start_m·64 ≥ seqlen) = true
  have hguard : evalOp (Op.boolNot (msaGuardGS BM)) s3 = some (Tile.scalar Bool.true) := by
    simp only [msaGuardGS, evalOp, evalOp_ref, evalOp_constNat, hs3start, hs3seq,
      Option.bind_eq_bind, Option.bind_some, Option.pure_def]
    refine congrArg some ?_; ext idx
    simp only [Tile.scalar, Tile.cop, Tile.cop_data, Tile.uop, Tile.uop_data, Tile.bop, Tile.bop_data,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, ComparableDType.ge,
      Bool.not_eq_true', decide_eq_false_iff_not, not_le]
    exact hactive
  -- Loop A: forRangeDyn over the 8 dense blocks (invariantA 0 → 8)
  obtain ⟨sA1, hloopA, hinvA8⟩ := msa_loopA_execGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD hBN H NR NS NV skn svn
    (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale) (msaKPtrGS K BD H skz skh skk s) (msaVPtrGS V BD H skz skh svk s) (msaOPtrGS Out BM BD H sqz sqh som sok s) s s0 hinv0
  -- handoff: max_num_cols = 16 (invariantA 8 → invariantB 0)
  obtain ⟨sB1, hhand, hinvB0⟩ := msa_handoffGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV
    (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale) (msaKPtrGS K BD H skz skh skk s) s)
    (msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn (msaVPtrGS V BD H skz skh svk s) s)
    (msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale) (msaKPtrGS K BD H skz skh skk s) s)
    (msaVblkB0GS V Seqlens Blocks BlockOffsets ColCounts Cols BD BN NR NV svn (msaVPtrGS V BD H skz skh svk s) s)
    (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale) (msaKPtrGS K BD H skz skh skk s) (msaVPtrGS V BD H skz skh svk s) (msaOPtrGS Out BM BD H sqz sqh som sok s) s 8 sA1 hinvA8
  -- Loop B: forRangeDyn over the 1 column block (invariantB 0 → 1)
  obtain ⟨sC1, hloopB, hinvB1⟩ := msa_loopB_execGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD hBN hBN16 H NR NS NV skn svn
    (msaMPartial BM BN (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale) (msaKPtrGS K BD H skz skh skk s) s) 8)
    (msaLPartial BM BN (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale) (msaKPtrGS K BD H skz skh skk s) s) 8)
    (fun ii dd => msaOPartial BM BN BD
      (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale) (msaKPtrGS K BD H skz skh skk s) s)
      (msaVblkA0GS V Seqlens Blocks BlockOffsets BD BN H NR NS svn (msaVPtrGS V BD H skz skh svk s) s) 8 ii dd)
    (msaQValGS Q BM BD H sqz sqh sqm sqk s sm_scale) (msaKPtrGS K BD H skz skh skk s) (msaVPtrGS V BD H skz skh svk s) (msaOPtrGS Out BM BD H sqz sqh som sok s) s sB1 hinvB0
  -- postLoop: acc /= l_i + masked store (invariantB 1 → genuine closed form)
  obtain ⟨sP, hpost, hOut⟩ := msaPostLoop_evalGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk svz svh svn svk som sok hsqk hskk hsvk hvz hvh hsok hBDsom hBN16
    s sC1 sm_scale hNCBN hpos hinvB1
  -- assemble the inner block: setup ++ [loopA, maxcols, loopB] ++ postLoop
  -- the middle block [loopA, maxcols, loopB] runs s0 → sC1
  have hmid : stepStmts [ Stmt.forRangeDyn "block_index" (Op.constNat 0)
        (Op.ref .nat [] "max_num_blks") (Op.constNat 1) (msaLoopBodyAGS BM BN BD skn svn),
       Stmt.assign .nat [] "max_num_cols" (Op.constNat 16),
       Stmt.forRangeDyn "start_n" (Op.constNat 0)
        (Op.ref .nat [] "max_num_cols") (Op.constNat BN) (msaLoopBodyBGS BM BN BD skn svn) ] s0 = some sC1 := by
    have hhandStmt : stepStmt (Stmt.assign .nat [] "max_num_cols" (Op.constNat 16)) sA1 = some sB1 := by
      rcases hx : stepStmt (Stmt.assign .nat [] "max_num_cols" (Op.constNat 16)) sA1 with _ | s'
      · rw [show stepStmts [Stmt.assign .nat [] "max_num_cols" (Op.constNat 16)] sA1 = none from by
          conv_lhs => unfold stepStmts; rw [hx]] at hhand; exact absurd hhand (by simp)
      · rw [stepStmts.cons_some hx, stepStmts.nil] at hhand; exact hhand
    rw [stepStmts.cons_some hloopA, stepStmts.cons_some hhandStmt,
      stepStmts.cons_some hloopB, stepStmts.nil]
  -- setup ++ mid runs s3 → sC1
  have hsetupmid : stepStmts (msaSetupGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV sm_scale
      ++ [ Stmt.forRangeDyn "block_index" (Op.constNat 0)
            (Op.ref .nat [] "max_num_blks") (Op.constNat 1) (msaLoopBodyAGS BM BN BD skn svn),
           Stmt.assign .nat [] "max_num_cols" (Op.constNat 16),
           Stmt.forRangeDyn "start_n" (Op.constNat 0)
            (Op.ref .nat [] "max_num_cols") (Op.constNat BN) (msaLoopBodyBGS BM BN BD skn svn) ]) s3 = some sC1 := by
    rw [stepStmts.append_some hsetup]; exact hmid
  have hinner : stepStmts ((msaSetupGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out BM BN BD H sqz sqh sqm sqk skz skh skk svk som sok NR NS NV sm_scale
      ++ [ Stmt.forRangeDyn "block_index" (Op.constNat 0)
            (Op.ref .nat [] "max_num_blks") (Op.constNat 1) (msaLoopBodyAGS BM BN BD skn svn),
           Stmt.assign .nat [] "max_num_cols" (Op.constNat 16),
           Stmt.forRangeDyn "start_n" (Op.constNat 0)
            (Op.ref .nat [] "max_num_cols") (Op.constNat BN) (msaLoopBodyBGS BM BN BD skn svn) ])
      ++ msaPostLoopGS BM BD) s3 = some sP := by
    rw [stepStmts.append_some hsetupmid]; exact hpost
  -- close: active ifThen runs the inner block
  rw [stepStmts.cons_some h0, stepStmts.cons_some h1, stepStmts.cons_some h2,
    stepStmts.cons_some (stepStmt_ifThen_true hguard hinner), stepStmts.nil]
  refine ⟨sP, rfl, ?_⟩
  intro idx hlt
  -- readback transport: msaPostLoop_eval's anchor is the clean input `s`
  have := hOut idx
  rw [if_pos hlt] at this
  exact this


/-- **Genuine dimension-general closed-form summary.** For symbolic block dims
`BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL` (under the honest faithful-regime side
conditions `0 < BLOCK_N`, `16 ≤ BLOCK_N` — so the kernel's single column block at
`max_num_cols = 16` covers all visited columns — and `BLOCK_DMODEL ≤ 64` — so the
fixed output strides `(stride_om, stride_ok) = (64, 1)` stay injective), the
executed surface kernel writes the genuine non-self-referential mixed-sparse
closed form `mixedSparseAttnClosedForm` to every active `Out` lane (the concrete
`64/64/64` and `32/32/64` shapes are instances obtained by specializing the
symbolic block dims). Side conditions
(`num_cols ≤ BLOCK_N`, per-active-lane positive online-softmax denominator,
clean `undef`) are honest hypotheses; the spec reads INPUT memory only.

**Why the raw `∃ sF, exec … ∧ …` form (not `ComputeCorrect.Realizes`) — honest
framework blocker, not a proof gap.** The `Out` cell is compared as a *decoded
fp16 value* `sF.readMemValue .fp16 Out … = (some … : WithBot ℝ)`. `Realizes`
requires an `OutputReadable` carrier for the readback type, and the framework
provides carriers only for `MemCell`/`ℝ`/`Nat`/`Int` — there is no
decoded-fp16-value (`TileCarrier .fp16 = WithBot ℝ`) carrier. The engine
(`msa_execGS`) yields a *value*-level equality, so it cannot be wrapped by
`Realizes` without a framework-level `ReadsAsValue`-style carrier. The statement
is nonetheless genuine and non-self-referential (`mixedSparseAttnClosedForm`
reads INPUT memory only), and the `∃ sF, exec … = some sF` conjunct is strictly
stronger than `Realizes` (it asserts execution actually succeeds). See
`bench/MAIN_THEOREM_CONVENTIONS.md` §4/§6. -/
theorem mixed_sparse_attention_output_closed_form_summary_general
    (Q K V Out : RegionName)
    (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat) (s : BlockState)
    (BM BN BD : Nat) (hBN : 0 < BN) (hBN16 : 16 ≤ BN)
    -- memory-layout strides (Q/K/V/O batch z, head h, row m / key n, channel k)
    -- and grid/layout sizes, ALL symbolic
    (NCTX Zc H NR NS NV
      sqz sqh sqm sqk skz skh skn skk svz svh svn svk soz soh som sok : Nat)
    -- honest contiguity hypotheses (the natural row-major attention layout):
    -- channel strides are 1; V reuses K's batch/head base; O channel stride 1 and
    -- its row stride covers the channel block (BD ≤ stride_om)
    (hsqk : sqk = 1) (hskk : skk = 1) (hsvk : svk = 1) (hvz : svz = skz) (hvh : svh = skh)
    (hsok : sok = 1) (hBDsom : BD ≤ som)
    (sm_scale : ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hactive : s.pids 0 * BM < seqLen s H (Region.cast Seqlens))
    (hNCBN : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * NR + s.pids 0) ≤ BN)
    (hpos : ∀ i : Fin BM, s.pids 0 * BM + i.val < seqLen s H (Region.cast Seqlens) →
      0 < msaDenomUpto BM BN
        (msaCatScore0GS Q K V Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD
          H NR NS NV sqz sqh sqm sqk skz skh skn skk s sm_scale) 9 i) :
    ∃ sF, exec (mixed_sparse_attention_fwd_kernel_surface Q K V Seqlens
        sm_scale Blocks BlockOffsets ColCounts Cols Out
        sqz sqh sqm sqk skz skh skn skk svz svh svn svk soz soh som sok
        Zc H NCTX NR NS NV BM BN BD FloatDType.fp16).toAlgKernel s = some sF
      ∧ ∀ idx : TileIndex [BM, BD],
          active s H Seqlens BM idx →
            sF.readMemValue .fp16 Out (outOffset s H sqz sqh som sok BM idx)
              = (some (mixedSparseAttnClosedForm s Q K V BlockOffsets Cols H
                  sqz sqh sqm skz skh skn svz svh svn NR NS NV
                  (s.readMemValue .nat (Region.cast Blocks) (s.pids 1 * NR + s.pids 0))
                  (s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * NR + s.pids 0))
                  (seqLen s H (Region.cast Seqlens)) BD BM BN sm_scale idx.1 (dIndex idx)) : WithBot ℝ) := by
  obtain ⟨sF, hexec, hOut⟩ := msa_execGS Q K V Seqlens Blocks BlockOffsets ColCounts Cols Out
    BM BN BD hBN hBN16 NCTX Zc H NR NS NV sqz sqh sqm sqk skz skh skn skk svz svh svn svk soz soh som sok
    hsqk hskk hsvk hvz hvh hsok hBDsom s sm_scale hundef hactive hNCBN hpos
  exact ⟨sF, hexec, fun idx hact => hOut idx hact⟩

end VeriTile.Bench.TritonBenchG.MixedSparseAttention
