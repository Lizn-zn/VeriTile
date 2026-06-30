# Spec sheet — `bench/tritonbench_g/mixed_sparse_attention/MixedSparseAttention.lean`

**Python source:** `bench/tritonbench_g/mixed_sparse_attention/mixed_sparse_attention.py`

## Public theorem: `mixed_sparse_attention_output_closed_form_summary_general`

<details><summary>docstring</summary>

```
/-- **Genuine dimension-general closed-form summary.** For symbolic block dims
`BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL` (under the honest faithful-regime side
conditions `0 < BLOCK_N`, `16 ≤ BLOCK_N` — so the kernel's single column block at
`max_num_cols = 16` covers all visited columns — and `BLOCK_DMODEL ≤ 64` — so the
fixed output strides `(stride_om, stride_ok) = (64, 1)` stay injective), the
executed surface kernel writes the genuine non-self-referential mixed-sparse
closed form `mixedSparseAttnClosedForm` to every active `Out` lane. Same
value-equality style as the pinned per-case summaries; the case `64/64/64` and
`32/32/64` summaries are instances of this theorem. Side conditions
(`num_cols ≤ BLOCK_N`, per-active-lane positive online-softmax denominator,
clean `undef`) are honest hypotheses; the spec reads INPUT memory only. -/
```
</details>

**Statement:**
```lean
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
                  (seqLen s H (Region.cast Seqlens)) BD BM BN sm_scale idx.1 (dIndex idx)) : WithBot ℝ)
```

**Assumptions / layout contracts:**
- `hBN : 0 < BN`
- `hBN16 : 16 ≤ BN`
- `hsqk : sqk = 1`
- `hskk : skk = 1`
- `hsvk : svk = 1`
- `hvz : svz = skz`
- `hvh : svh = skh`
- `hsok : sok = 1`
- `hBDsom : BD ≤ som`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hactive : s.pids 0 * BM < seqLen s H (Region.cast Seqlens)`
- `hNCBN : s.readMemValue .nat (Region.cast ColCounts) (s.pids 1 * NR + s.pids 0) ≤ BN`
- `hpos : ∀ i : Fin BM, s.pids 0 * BM + i.val < seqLen s H (Region.cast Seqlens) →
      0 < msaDenomUpto BM BN
        (msaCatScore0GS Q K V Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD
          H NR NS NV sqz sqh sqm sqk skz skh skn skk s sm_scale) 9 i`

**Closed-form spec defs (transitive):** `seqLen`, `msaDenomUpto`, `msaCatScore0GS`, `mixed_sparse_attention_fwd_kernel_surface`, `active`, `outOffset`, `mixedSparseAttnClosedForm`, `dIndex`, `offZ`, `msaE`, `msaCatScore`, `msaScoreA0GS`, `msaQValGS`, `msaKPtrGS`, `msaScoreB0GS`, `mIndex`, `offH`, `rawScore`, `blockStartN`, `effScale`, `vRow`, `colKeyGlobal`, `msaScoreLaneAGS`, `msaSN0GS`, `msaScoreLaneBGS`, `msaGcol0GS`, `qRow`, `kRow`, `qoBase`, `msaKLaneAGS`, `msaKLaneBGS`, `msaColLaneBGS`

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (H : Nat) (Seqlens : RegionName) : Nat :=
  s.readMemValue .nat Seqlens (offZ s H)
```
</details>

<details><summary><code>msaDenomUpto</code></summary>

```
/-- Direct (unshifted) running denominator: `Σ_{l<k} Σ_j exp2(score l i j)`. -/
```
```lean
noncomputable def msaDenomUpto (BM BN : Nat)
    (score : Nat → Fin BM → Fin BN → WithBot ℝ) (k : Nat) (i : Fin BM) : ℝ :=
  (Finset.range k).sum (fun l =>
    (Finset.univ : Finset (Fin BN)).sum (fun j => msaE (score l i j)))
```
</details>

<details><summary><code>msaCatScore0GS</code></summary>

```lean
noncomputable abbrev msaCatScore0GS
    (Q K V : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (BM BN BD H NR NS NV sqz sqh sqm sqk skz skh skn skk : Nat) (s0 : BlockState) (sm_scale : ℝ := 0.1) : Nat → Fin BM → Fin BN → WithBot ℝ :=
  msaCatScore BM BN 8
    (msaScoreA0GS Q K Seqlens Blocks BlockOffsets BM BN BD H NR NS skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0)
    (msaScoreB0GS Q K Seqlens Blocks BlockOffsets ColCounts Cols BM BN BD H NR NV skn (msaQValGS Q BM BD H sqz sqh sqm sqk s0 sm_scale) (msaKPtrGS K BD H skz skh skk s0) s0)
```
</details>

<details><summary><code>mixed_sparse_attention_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `mixed_sparse_attention.py`'s
`_triton_mixed_sparse_attn_fwd_kernel`. -/
```
```lean
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
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (H : Nat) (Seqlens : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H Seqlens
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_om stride_ok BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + dIndex idx * stride_ok
```
</details>

<details><summary><code>mixedSparseAttnClosedForm</code></summary>

```
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
```
```lean
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
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>msaE</code></summary>

```
/-- `msaE x = exp2(x)` with `exp2(⊥) = 0` — the per-key softmax weight carrier. -/
```
```lean
noncomputable def msaE (x : WithBot ℝ) : ℝ := (WithBot.realExp2 x).unbotD 0
```
</details>

<details><summary><code>msaCatScore</code></summary>

```
/-- Concatenated score stream: first `bF` iterations from `scoreA`, then `scoreB`. -/
```
```lean
noncomputable def msaCatScore (BM BN bF : Nat)
    (scoreA scoreB : Nat → Fin BM → Fin BN → WithBot ℝ) :
    Nat → Fin BM → Fin BN → WithBot ℝ :=
  fun k => if k < bF then scoreA k else scoreB (k - bF)
```
</details>

<details><summary><code>msaScoreA0GS</code></summary>

```
/-- Symbolic block-A score stream. -/
```
```lean
noncomputable def msaScoreA0GS (Q K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (BM BN BD H NR NS skn : Nat) (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin BM → Fin BN → WithBot ℝ :=
  fun c i j => msaScoreLaneAGS Q K Seqlens Blocks BlockOffsets BM BN BD H NR skn qF kpF s0 c
    (msaSN0GS s0 Blocks BlockOffsets NR NS c) i j
```
</details>

<details><summary><code>msaQValGS</code></summary>

```
/-- Symbolic scaled `q` value. -/
```
```lean
noncomputable def msaQValGS (Q : RegionName) (BM BD H sqz sqh sqm sqk : Nat) (s0 : BlockState)
    (sm_scale : ℝ := 0.1) : TileIndex [BM, BD] → WithBot ℝ :=
  fun idx => WithBot.realMul
    (s0.readMemValue .real Q (((s0.pids 1 / H) * sqz + (s0.pids 1 % H) * sqh)
      + (s0.pids 0 * BM + idx.1.val) * sqm + idx.2.1.val * sqk))
    (some (sm_scale * 1.44269504))
```
</details>

<details><summary><code>msaKPtrGS</code></summary>

```
/-- Symbolic `k_ptrs` lane. -/
```
```lean
def msaKPtrGS (K : RegionName) (BD H skz skh skk : Nat) (s0 : BlockState) : TileIndex [BD, 1] → RegionName × Nat :=
  fun idx => (K, ((s0.pids 1 / H) * skz + (s0.pids 1 % H) * skh) + idx.1.val * skk)
```
</details>

<details><summary><code>msaScoreB0GS</code></summary>

```
/-- Symbolic column-B score stream (loop value `sv = c·BN`). -/
```
```lean
noncomputable def msaScoreB0GS (Q K : RegionName) (Seqlens Blocks BlockOffsets ColCounts Cols : Region .nat)
    (BM BN BD H NR NV skn : Nat) (qF : TileIndex [BM, BD] → WithBot ℝ) (kpF : TileIndex [BD, 1] → RegionName × Nat)
    (s0 : BlockState) : Nat → Fin BM → Fin BN → WithBot ℝ :=
  fun c i j => msaScoreLaneBGS Blocks ColCounts Seqlens BM BN BD H NR skn qF kpF s0 (c * BN)
    (msaGcol0GS s0 Cols ColCounts BN NR NV (c * BN)) i j
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>rawScore</code></summary>

```
/-- Unscaled raw score `Σ_{e<BLOCK_DMODEL} Q[row,e] · K[n,e]` at global key `n`. -/
```
```lean
noncomputable def rawScore (s : BlockState) (Q K : RegionName)
    (H stride_qz stride_qh stride_qm stride_kz stride_kh stride_kn
      BLOCK_DMODEL BLOCK_M : Nat) (i : Fin BLOCK_M) (n : Nat) : ℝ :=
  Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
    qRow s Q H stride_qz stride_qh stride_qm BLOCK_M i e.val *
      kRow s K H stride_kz stride_kh stride_kn n e.val)
```
</details>

<details><summary><code>blockStartN</code></summary>

```
/-- The masked block start `start_n` the kernel reads at block `b` (Loop A's
`tl.load(blks_ptr + b, mask = b < num_blks)`): the real offset for a visited
block, the masked default `0` for a spurious block `b ≥ num_blks`. -/
```
```lean
noncomputable def blockStartN (s : BlockState) (block_offset : Region .nat)
    (NUM_ROWS NNZ_S num_blks b : Nat) : Nat :=
  if b < num_blks then
    s.readMemValue .nat (Region.cast block_offset)
      ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_S + b)
  else BlockState.defaultCarrier .nat
```
</details>

<details><summary><code>effScale</code></summary>

```
/-- **Faithful exp2→exp scale.** The kernel sets `qk_scale = sm_scale ·
1.44269504` and exponentiates with `exp2`. Since the semantics give
`exp2(x) = exp(x · log 2)`, the per-key weight the loop computes is
`exp2(qk_scale · raw) = exp(qk_scale · log 2 · raw)`. Hence the natural-exp
scale to instantiate the closed form with is
`effScale sm_scale = sm_scale · 1.44269504 · log 2`. (`1.44269504 · log 2 ≈ 1`,
the floating-point approximation of `log2(e) · ln 2 = 1`.) -/
```
```lean
noncomputable def effScale (sm_scale : ℝ) : ℝ :=
  sm_scale * 1.44269504 * Real.log 2
```
</details>

<details><summary><code>vRow</code></summary>

```
/-- V row at global key position `n`, channel `d`, at `kvBase + n·stride_vn + d`. -/
```
```lean
noncomputable def vRow (s : BlockState) (V : RegionName)
    (H stride_vz stride_vh stride_vn : Nat) (n d : Nat) : ℝ :=
  s.readMem V (qoBase s H stride_vz stride_vh + n * stride_vn + d)
```
</details>

<details><summary><code>colKeyGlobal</code></summary>

```
/-- Global key position of the `c`-th visited sparse column:
`column_index[off_hz·NUM_ROWS·NNZ_V + start_m·NNZ_V + c]`. -/
```
```lean
def colKeyGlobal (s : BlockState) (column_index : Region .nat)
    (NUM_ROWS NNZ_V c : Nat) : Nat :=
  s.readMemValue .nat (Region.cast column_index)
    ((s.pids 1 * NUM_ROWS + s.pids 0) * NNZ_V + c)
```
</details>

<details><summary><code>msaScoreLaneAGS</code></summary>

```
/-- Symbolic Loop-A masked `qk` lane `(i,j)`. -/
```
```lean
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
```
</details>

<details><summary><code>msaSN0GS</code></summary>

```
/-- Symbolic masked block-start gather. -/
```
```lean
noncomputable def msaSN0GS (s0 : BlockState) (Blocks BlockOffsets : Region .nat) (NR NS c : Nat) : Nat :=
  if c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0) then
    s0.readMemValue .nat (Region.cast BlockOffsets) ((s0.pids 1 * NR + s0.pids 0) * NS + c)
  else BlockState.defaultCarrier .nat
```
</details>

<details><summary><code>msaScoreLaneBGS</code></summary>

```
/-- Symbolic Loop-B masked `qk` lane `(i,j)` (NON-causal). -/
```
```lean
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
```
</details>

<details><summary><code>msaGcol0GS</code></summary>

```
/-- Symbolic gathered columns at loop value `sv`. -/
```
```lean
noncomputable def msaGcol0GS (s0 : BlockState) (Cols ColCounts : Region .nat) (BN NR NV sv : Nat) :
    Fin BN → Nat :=
  fun j => msaColLaneBGS Cols ColCounts BN NR NV s0 sv j
```
</details>

<details><summary><code>qRow</code></summary>

```
/-- Q row `start_m·BLOCK_M + i`, channel `e`, at `qoBase + row·stride_qm + e`. -/
```
```lean
noncomputable def qRow (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh stride_qm BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) :
    ℝ :=
  s.readMem Q (qoBase s H stride_qz stride_qh + mIndex s BLOCK_M i * stride_qm + e)
```
</details>

<details><summary><code>kRow</code></summary>

```
/-- K row at global key position `n`, channel `e`, at `kvBase + n·stride_kn + e`.
The kernel reads K with `k_ptrs = K + kv_offset + offs_d·stride_kk` then
`+ cols·stride_kn`; here `kv_offset = off_z·stride_kz + off_h·stride_kh` and
`stride_kk = 1` (head channel `e` contiguous). -/
```
```lean
noncomputable def kRow (s : BlockState) (K : RegionName)
    (H stride_kz stride_kh stride_kn : Nat) (n e : Nat) : ℝ :=
  s.readMem K (qoBase s H stride_kz stride_kh + n * stride_kn + e)
```
</details>

<details><summary><code>qoBase</code></summary>

```
/-- Q/out tile base offset `off_z · stride_z + off_h · stride_h`. -/
```
```lean
def qoBase (s : BlockState) (H stride_z stride_h : Nat) : Nat :=
  offZ s H * stride_z + offH s H * stride_h
```
</details>

<details><summary><code>msaKLaneAGS</code></summary>

```
/-- Symbolic Loop-A masked K-load lane `(e,j)`. -/
```
```lean
noncomputable def msaKLaneAGS (K : RegionName) (Seqlens Blocks BlockOffsets : Region .nat)
    (BD BN H NR skn : Nat) (kpF : TileIndex [BD, 1] → RegionName × Nat) (s0 : BlockState) (c SN : Nat)
    (e : Fin BD) (j : Fin BN) : WithBot ℝ :=
  if (SN + j.val < seqLen s0 H (Region.cast Seqlens)
      ∧ c < s0.readMemValue .nat (Region.cast Blocks) (s0.pids 1 * NR + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + (SN + j.val) * skn) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)
```
</details>

<details><summary><code>msaKLaneBGS</code></summary>

```
/-- Symbolic Loop-B masked K-load lane `(e,j)`. -/
```
```lean
noncomputable def msaKLaneBGS (Blocks ColCounts : Region .nat)
    (BD BN NR skn : Nat) (kpF : TileIndex [BD, 1] → RegionName × Nat) (s0 : BlockState) (sv : Nat)
    (gcol : Fin BN → Nat) (e : Fin BD) (j : Fin BN) : WithBot ℝ :=
  if (sv + j.val < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)
      ∧ sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0)) then
    (s0.readMemValue .real (kpF (e, ⟨0, by simp⟩, PUnit.unit)).1
      ((kpF (e, ⟨0, by simp⟩, PUnit.unit)).2 + gcol j * skn) : WithBot ℝ)
  else (some (0.0 : ℝ) : WithBot ℝ)
```
</details>

<details><summary><code>msaColLaneBGS</code></summary>

```
/-- Symbolic Loop-B gathered column lane `j`. -/
```
```lean
noncomputable def msaColLaneBGS (Cols : Region .nat) (ColCounts : Region .nat)
    (BN NR NV : Nat) (s0 : BlockState) (sv : Nat) (j : Fin BN) : Nat :=
  if sv < s0.readMemValue .nat (Region.cast ColCounts) (s0.pids 1 * NR + s0.pids 0) then
    s0.readMemValue .nat (Region.cast Cols) ((s0.pids 1 * NR + s0.pids 0) * NV + sv + j.val)
  else 0
```
</details>
