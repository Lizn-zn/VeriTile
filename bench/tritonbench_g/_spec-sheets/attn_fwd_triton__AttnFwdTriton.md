# Spec sheet — `bench/tritonbench_g/attn_fwd_triton/AttnFwdTriton.lean`

**Python source:** `bench/tritonbench_g/attn_fwd_triton/attn_fwd_triton.py`

## Public theorem: `attn_fwd_triton_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general genuine causal closed-form output summary for `attn_fwd_triton`.**

Removes the Python test-shape pin (`Z = 2`, `H = 4`,
`N_CTX = HEAD_DIM = BLOCK_M = BLOCK_DMODEL = 128`, `BLOCK_N = 64`,
`HEAD_ACTIVE = 96`): for arbitrary symbolic block/head dimensions at the
contiguous layout (`stride_qm = HEAD_DIM`, `stride_qk = 1`, `stride_kn = HEAD_DIM`),
the full causal surface lowers to the algorithm layer and its masked `Out`
writeback realizes the genuine closed-form causal attention
`attnFwdTritonOutSpecG` (= `attentionRealBase2PerKeyScalePred … (causalKeep)` over
INPUT memory) at every active output lane. Side conditions: positive blocks,
`N_CTX = BLOCK_N · numKVBlocks` (`numKVBlocks > 0`), clean input (`undef = 0`), the
`-1e6` sentinel score bound `aftgScoreBoundG`, and output-offset injectivity. -/
```
</details>

**Statement:**
```lean
specification attn_fwd_triton_output_summary_general
    (Q K V QScale KScale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z numKVBlocks : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M)
    (hN : N_CTX = BLOCK_N * numKVBlocks) (hnum : 0 < numKVBlocks)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hsb : aftgScoreBoundG
      (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
      (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
      (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
      (keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) (qStartAFT2G s BLOCK_M)) :
    (∃ alg, (attn_fwd_triton_surface Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attn_fwd_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => active s N_CTX HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        attnFwdTritonOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks (keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) idx)
```

**Assumptions / layout contracts:**
- `hBD : 0 < BLOCK_DMODEL`
- `hBN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`
- `hN : N_CTX = BLOCK_N * numKVBlocks`
- `hnum : 0 < numKVBlocks`
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `outOffset`, `aftgScoreBoundG`, `qTileAFT2mG`, `kTileAFT2G`, `vTileAFT2mG`, `keyScaleAFT2G`, `qStartAFT2G`, `attn_fwd_triton_surface`, `active`, `attnFwdTritonOutSpecG`, `offZ`, `offH`, `mIndex`, `kIndex`, `qTileAFT2G`, `baseOffsetAFT2G`, `vTileAFT2G`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + kIndex idx * stride_qk
```
</details>

<details><summary><code>aftgScoreBoundG</code></summary>

```
/-- General sentinel boundedness side-condition (**causal-only, non-strict**).

The kernel masks every non-causal key with the `-1e6` `tl.where` sentinel, so only
the **kept** (causal) keys `j ≤ qStart + i` need a boundedness assumption, and only
`≥ -1e6` (non-strict) is required: the masked keys are pinned to the sentinel and
never raise the running max. -/
```
```lean
def aftgScoreBoundG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart : Nat) : Prop :=
  ∀ (j : Fin SEQ) (i : Fin BLOCK_M), j.val ≤ qStart + i.val →
    (0:ℝ) - 1000000.0
      ≤ keyScale j * Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit))
```
</details>

<details><summary><code>qTileAFT2mG</code></summary>

```
/-- General masked query tile: head-active (`e < HEAD_ACTIVE`) and query-row
boundary (`qStart + i < N_CTX`) masking, mirroring the kernel's `q` load mask. -/
```
```lean
noncomputable def qTileAFT2mG (s : BlockState) (Q : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, u) =>
    if qStartAFT2G s BLOCK_M + i.val < N_CTX ∧ e.val < HEAD_ACTIVE then
      qTileAFT2G s Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL (i, e, u) else 0
```
</details>

<details><summary><code>kTileAFT2G</code></summary>

```
/-- General key tile: row `j` (global key), head lane `e` (`K[base + j·HEAD_DIM + e]`). -/
```
```lean
noncomputable def kTileAFT2G (s : BlockState) (K : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, e, _) => s.readMem K (baseOffsetAFT2G s stride_qz stride_qh H + j.val * HEAD_DIM + e.val)
```
</details>

<details><summary><code>vTileAFT2mG</code></summary>

```
/-- General masked value tile: head-active (`d < HEAD_ACTIVE`) masking. -/
```
```lean
noncomputable def vTileAFT2mG (s : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, d, u) =>
    if d.val < HEAD_ACTIVE then
      vTileAFT2G s V stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL (j, d, u) else 0
```
</details>

<details><summary><code>keyScaleAFT2G</code></summary>

```
/-- General per-key score scale carrier `q_scale · k_scale` (block `j / BLOCK_N`). -/
```
```lean
noncomputable def keyScaleAFT2G (s : BlockState) (QScale KScale : RegionName)
    (N_CTX BLOCK_M BLOCK_N numKVBlocks : Nat) : Fin (BLOCK_N * numKVBlocks) → ℝ :=
  fun j => s.readMem QScale (s.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M) + s.pids 0)
            * s.readMem KScale (s.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N) + j.val / BLOCK_N)
```
</details>

<details><summary><code>qStartAFT2G</code></summary>

```
/-- General global query row for output tile-row `i`. -/
```
```lean
def qStartAFT2G (s : BlockState) (BLOCK_M : Nat) : Nat := s.pids 0 * BLOCK_M
```
</details>

<details><summary><code>attn_fwd_triton_surface</code></summary>

```
/-- Full Lean port of `attn_fwd_triton.py`'s `_attn_fwd` (`STAGE = 3`).

The upstream kernel runs the K/V streaming-softmax loop through a separate
`@triton.jit` helper `_attn_fwd_inner`, invoked twice: stage `4 - STAGE = 1`
streams the strictly-below-diagonal key blocks `range(0, start_m·BLOCK_M)` with
no causal mask, and stage `2` streams the diagonal block
`range(start_m·BLOCK_M, (start_m+1)·BLOCK_M)` under the causal predicate
`offs_m[:, None] ≥ start_n + offs_n[None, :]`.

The DSL has no cross-`@triton.jit` function-call surface, so the helper body is
inlined here as a single `forRange` loop `range(0, N_CTX, BLOCK_N)` with the
causal `where` `offs_m[:, None] ≥ start_n + offs_n[None, :]` applied to every
block. This faithfully composes the two staged helper calls of `STAGE = 3`: on a
stage-1 block (strictly below the diagonal) every lane satisfies
`offs_m ≥ start_n + offs_n`, so the causal `where` is a no-op there, matching the
unmasked stage-1 helper; on the diagonal block it is the stage-2 mask; and on
the strictly-above-diagonal blocks (which the two-call kernel never visits) the
`where` zeroes every probability (`p = where(mask, p, 0)`), so they contribute
nothing to `acc`/`l_i` — making the full-range loop equal to the kernel's
`range(0, (start_m+1)·BLOCK_M)` traversal. -/
```
```lean
def attn_fwd_triton_surface
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk, -1000000.0)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    p = tl.where(mask, p, 0.0)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < N_CTX ∧ kIndex idx < HEAD_ACTIVE
```
</details>

<details><summary><code>attnFwdTritonOutSpecG</code></summary>

```
/-- **General genuine closed form** (exp2, causal): predicate-masked base-2
per-key-scale attention with the `causalKeep qStart` mask, over the kernel's
actually-loaded masked q/v tiles. -/
```
```lean
noncomputable def attnFwdTritonOutSpecG
    (s : BlockState) (Q K V : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  attentionRealBase2PerKeyScalePred
    (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
    (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
    (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
    keyScale (fun i j => causalKeep (qStartAFT2G s BLOCK_M) i j) idx
```
</details>

<details><summary><code>offZ</code></summary>

```lean
def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>offH</code></summary>

```lean
def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>qTileAFT2G</code></summary>

```
/-- General query tile: row `i` = global query `pids0·BLOCK_M + i`, head lane `e`
(`stride_qm = HEAD_DIM`, `stride_qk = 1`). -/
```
```lean
noncomputable def qTileAFT2G (s : BlockState) (Q : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, _) => s.readMem Q (baseOffsetAFT2G s stride_qz stride_qh H
    + (s.pids 0 * BLOCK_M + i.val) * HEAD_DIM + e.val)
```
</details>

<details><summary><code>baseOffsetAFT2G</code></summary>

```
/-- General per-plane base offset: `off_z·stride_qz + off_h·stride_qh` with
`off_z = pids1 / H`, `off_h = pids1 % H`. -/
```
```lean
def baseOffsetAFT2G (s : BlockState) (stride_qz stride_qh H : Nat) : Nat :=
  s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh
```
</details>

<details><summary><code>vTileAFT2G</code></summary>

```
/-- General value tile: row `j` (global key), head lane `d` (`V[base + j·HEAD_DIM + d]`). -/
```
```lean
noncomputable def vTileAFT2G (s : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, d, _) => s.readMem V (baseOffsetAFT2G s stride_qz stride_qh H + j.val * HEAD_DIM + d.val)
```
</details>

## Public theorem: `attn_fwd_triton_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` io headline — on the `StreamMasked3DKernelIO₅` skin (sibling
of `attn_fwd_causal`'s exemplar; the two surfaces are statement-identical).** On
its five-stream single-store signature, `_attn_fwd` implements the genuine
causal closed form: output lane `j = (i, e)` of `Out` holds the
predicate-masked base-2 per-key-scale attention
`attnFwdTritonIOOutSpec` (= the exact stack's `attnFwdTritonOutSpecG`
restated over the five streams, with `keyScale j = Q_scale · K_scale[j/BN]`
assembled from the two scalar-width channels), for every rounding model that
is trivial on the fp16 grid.

**Hypothesis provenance**:
* `hfp16` pins `R.round .fp16 = id` — loop-body statement 16
  (`p = (p).to(tl.float16)`) is an *in-loop* rounding event outside the
  skin's single-boundary-round shape, exactly the file's declared fp16
  modeling boundary (the exact stack already treats this cast as the
  identity), now explicit as a headline hypothesis.
* `hBD`/`hBN`/`hBM` positivity and `hN`/`hnum` (`N_CTX = BLOCK_N ·
  numKVBlocks > 0`) shape the KV walk — inherited verbatim from the exact
  headline `attn_fwd_triton_output_summary_general`.
* `hBDHD : BLOCK_DMODEL ≤ HEAD_DIM` — host launches use
  `HEAD_DIM = BLOCK_DMODEL`; it makes the contiguous `Out` window injective,
  **discharging** the exact headline's `hOutInj` hypothesis.
* `hsb` — the exact stack's `-1e6` sentinel score bound `aftgScoreBoundG`
  restated tile-parametrically over the streams (∀-pids ∀-streams form;
  consumers instantiate it at their loaded tiles): every causally kept key's
  scaled score is `≥ -1e6`, so the kernel's causal `tl.where` sentinel never
  overtakes a genuine score.

The exact headline's `hundef` is **not** a hypothesis here — the skin's
Hoare triple carries the `undef` pin itself. The output grid is the `.real`
default (the store-side `.to(Out.type.element_ty)` cast erases at
translation), so at every such `R` the terminal cells carry the exact fold
values. -/
```
</details>

**Statement:**
```lean
specification attn_fwd_triton_io_correctness (R : RoundingModel)
    (hfp16 : R.round .fp16 = id)
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M)
    (hN : N_CTX = BLOCK_N * numKVBlocks) (hnum : 0 < numKVBlocks)
    (hBDHD : BLOCK_DMODEL ≤ HEAD_DIM)
    (hsb : ∀ (p₀ : Nat) (xs : Fin numKVBlocks → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
      (ys : Fin numKVBlocks → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
      (zs : Fin numKVBlocks → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
      (ws vs : Fin numKVBlocks → Fin 1 → ℝ),
      aftgIOScoreBound p₀ N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs) :
    attnFwdTritonIO Q K V QScale KScale Out stride_qz stride_qh Z H N_CTX HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨[R]
      fun p₀ _ _ xs ys zs ws vs j =>
        attnFwdTritonIOOutSpec p₀ N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
          xs ys zs ws vs (Lane2D.decode j)
```

**Assumptions / layout contracts:**
- `hfp16 : R.round .fp16 = id`
- `hBD : 0 < BLOCK_DMODEL`
- `hBN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`
- `hN : N_CTX = BLOCK_N * numKVBlocks`
- `hnum : 0 < numKVBlocks`
- `hBDHD : BLOCK_DMODEL ≤ HEAD_DIM`
- `xs : Fin numKVBlocks → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ`
- `ys : Fin numKVBlocks → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ`
- `zs : Fin numKVBlocks → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ`
- `ws vs : Fin numKVBlocks → Fin 1 → ℝ`

**Closed-form spec defs (transitive):** `aftgIOScoreBound`, `attnFwdTritonIO`, `attnFwdTritonIOOutSpec`, `aftgScoreBoundG`, `aftgIOqTm`, `aftgIOkTm`, `aftgIOvTm`, `aftgIOkeyScale`, `attn_fwd_triton_surface`

<details><summary><code>aftgIOScoreBound</code></summary>

```
/-- **The tile-parametric sentinel score bound** — the exact stack's
`aftgScoreBoundG` side condition restated over the five streams (∀-pids form;
consumers instantiate it at their loaded tiles): every *causally kept* key's
scaled score is `≥ -1e6`, so the kernel's `tl.where(mask, qk, -1e6)` sentinel
never overtakes a genuine score. -/
```
```lean
def aftgIOScoreBound (p₀ N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (ws vs : Fin T → Fin 1 → ℝ) : Prop :=
  aftgScoreBoundG
    (aftgIOqTm p₀ N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T xs)
    (aftgIOkTm BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T ys)
    (aftgIOvTm BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T zs)
    (aftgIOkeyScale BLOCK_N T ws vs)
    (p₀ * BLOCK_M)
```
</details>

<details><summary><code>attnFwdTritonIO</code></summary>

```
/-- **Streaming IO signature** of `_attn_fwd` on the five-stream single-store
attention fold skin (`T = numKVBlocks` under `N_CTX = BLOCK_N · numKVBlocks`;
the trip count is pid-free, so there is no launch-legality `pre` analog to
carry). -/
```
```lean
def attnFwdTritonIO (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat) : StreamMasked3DKernelIO₅ where
  kernel := attn_fwd_triton_surface Q K V QScale KScale Out
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := QScale
  inp5 := KScale
  out := Out
  T := numKVBlocks
  B1 := BLOCK_M * BLOCK_DMODEL
  B2 := BLOCK_DMODEL * BLOCK_N
  B3 := BLOCK_N * BLOCK_DMODEL
  B4 := 1
  B5 := 1
  C := BLOCK_M * BLOCK_DMODEL
  outDType := .real
  read1 := fun p₀ p₁ _ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read2 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD_DIM
  read3 := fun _ p₁ _ t j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (t.val * BLOCK_N + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  read4 := fun p₀ p₁ _ _ _ => p₁ * ((N_CTX + BLOCK_M - 1) / BLOCK_M) + p₀
  read5 := fun _ p₁ _ t _ => p₁ * ((N_CTX + BLOCK_N - 1) / BLOCK_N) + t.val
  write := fun p₀ p₁ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  mask1 := fun p₀ _ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < N_CTX ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask2 := fun _ _ _ t j =>
    j.val % BLOCK_N < N_CTX - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE
  mask3 := fun _ _ _ t j =>
    j.val / BLOCK_DMODEL < N_CTX - t.val * BLOCK_N ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask4 := fun _ _ _ _ _ => True
  mask5 := fun _ _ _ _ _ => True
  writeMask := fun p₀ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < N_CTX ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
```
</details>

<details><summary><code>attnFwdTritonIOOutSpec</code></summary>

```
/-- **The streamed closed-form spec `f`**: predicate-masked base-2
per-key-scale attention with the `causalKeep (p₀·BLOCK_M)` mask
(`attnFwdTritonOutSpecG` restated tile-parametrically over the five
streams). -/
```
```lean
noncomputable def attnFwdTritonIOOutSpec
    (p₀ N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (ws vs : Fin T → Fin 1 → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  attentionRealBase2PerKeyScalePred
    (aftgIOqTm p₀ N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T xs)
    (aftgIOkTm BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T ys)
    (aftgIOvTm BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T zs)
    (aftgIOkeyScale BLOCK_N T ws vs)
    (fun i j => causalKeep (p₀ * BLOCK_M) i j) idx
```
</details>

<details><summary><code>aftgScoreBoundG</code></summary>

```
/-- General sentinel boundedness side-condition (**causal-only, non-strict**).

The kernel masks every non-causal key with the `-1e6` `tl.where` sentinel, so only
the **kept** (causal) keys `j ≤ qStart + i` need a boundedness assumption, and only
`≥ -1e6` (non-strict) is required: the masked keys are pinned to the sentinel and
never raise the running max. -/
```
```lean
def aftgScoreBoundG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart : Nat) : Prop :=
  ∀ (j : Fin SEQ) (i : Fin BLOCK_M), j.val ≤ qStart + i.val →
    (0:ℝ) - 1000000.0
      ≤ keyScale j * Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit))
```
</details>

<details><summary><code>aftgIOqTm</code></summary>

```
/-- The **masked query tile** read off the static first stream (the window
ignores `t`, so the step-`0` slice carries the whole tile; masked lanes are
`0`, mirroring the kernel's `q` load mask). -/
```
```lean
noncomputable def aftgIOqTm (p₀ N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun idx =>
    if h : (p₀ * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE) ∧ 0 < T then
      xs ⟨0, h.2⟩ (Lane2D.encode idx)
    else 0
```
</details>

<details><summary><code>aftgIOkTm</code></summary>

```
/-- The **head-masked global key tile** read off the transposed `K` stream:
global key `jg` lives in step `jg / BLOCK_N` at block-local column
`jg % BLOCK_N`; head lanes `e ≥ HEAD_ACTIVE` (which the kernel never loads)
are `0`. -/
```
```lean
noncomputable def aftgIOkTm (BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ) :
    TileIndex [BLOCK_N * T, BLOCK_DMODEL] → ℝ :=
  fun idx =>
    if h : idx.2.1.val < HEAD_ACTIVE ∧ idx.1.val / BLOCK_N < T ∧ 0 < BLOCK_N then
      ys ⟨idx.1.val / BLOCK_N, h.2.1⟩
        (Lane2D.encode (idx.2.1, ⟨idx.1.val % BLOCK_N, Nat.mod_lt _ h.2.2⟩, PUnit.unit))
    else 0
```
</details>

<details><summary><code>aftgIOvTm</code></summary>

```
/-- The **head-masked global value tile** read off the `V` stream (row-major
per-step tiles; head lanes `d ≥ HEAD_ACTIVE` are `0`, mirroring the kernel's
`v` load mask). -/
```
```lean
noncomputable def aftgIOvTm (BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) :
    TileIndex [BLOCK_N * T, BLOCK_DMODEL] → ℝ :=
  fun idx =>
    if h : idx.2.1.val < HEAD_ACTIVE ∧ idx.1.val / BLOCK_N < T ∧ 0 < BLOCK_N then
      zs ⟨idx.1.val / BLOCK_N, h.2.1⟩
        (Lane2D.encode (⟨idx.1.val % BLOCK_N, Nat.mod_lt _ h.2.2⟩, idx.2.1, PUnit.unit))
    else 0
```
</details>

<details><summary><code>aftgIOkeyScale</code></summary>

```
/-- The per-key score-scale carrier `q_scale · k_scale` read off the two
scalar streams: the static `Q_scale` slot (step `0`) times key `j`'s
`K_scale` slot (step `j / BLOCK_N`). -/
```
```lean
noncomputable def aftgIOkeyScale (BLOCK_N T : Nat) (ws vs : Fin T → Fin 1 → ℝ) :
    Fin (BLOCK_N * T) → ℝ :=
  fun j =>
    (if h : 0 < T then ws ⟨0, h⟩ 0 else 0)
      * (if h : j.val / BLOCK_N < T then vs ⟨j.val / BLOCK_N, h⟩ 0 else 0)
```
</details>

<details><summary><code>attn_fwd_triton_surface</code></summary>

```
/-- Full Lean port of `attn_fwd_triton.py`'s `_attn_fwd` (`STAGE = 3`).

The upstream kernel runs the K/V streaming-softmax loop through a separate
`@triton.jit` helper `_attn_fwd_inner`, invoked twice: stage `4 - STAGE = 1`
streams the strictly-below-diagonal key blocks `range(0, start_m·BLOCK_M)` with
no causal mask, and stage `2` streams the diagonal block
`range(start_m·BLOCK_M, (start_m+1)·BLOCK_M)` under the causal predicate
`offs_m[:, None] ≥ start_n + offs_n[None, :]`.

The DSL has no cross-`@triton.jit` function-call surface, so the helper body is
inlined here as a single `forRange` loop `range(0, N_CTX, BLOCK_N)` with the
causal `where` `offs_m[:, None] ≥ start_n + offs_n[None, :]` applied to every
block. This faithfully composes the two staged helper calls of `STAGE = 3`: on a
stage-1 block (strictly below the diagonal) every lane satisfies
`offs_m ≥ start_n + offs_n`, so the causal `where` is a no-op there, matching the
unmasked stage-1 helper; on the diagonal block it is the stage-2 mask; and on
the strictly-above-diagonal blocks (which the two-call kernel never visits) the
`where` zeroes every probability (`p = where(mask, p, 0)`), so they contribute
nothing to `acc`/`l_i` — making the full-range loop equal to the kernel's
`range(0, (start_m+1)·BLOCK_M)` traversal. -/
```
```lean
def attn_fwd_triton_surface
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk, -1000000.0)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    p = tl.where(mask, p, 0.0)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}
```
</details>

## Also present (pinned special-case summaries)
- `attn_fwd_triton_final_store_slice_compute_correct`
