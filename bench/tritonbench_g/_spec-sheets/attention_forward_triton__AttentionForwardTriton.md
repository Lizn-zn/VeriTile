# Spec sheet — `bench/tritonbench_g/attention_forward_triton/AttentionForwardTriton.lean`

**Python source:** `bench/tritonbench_g/attention_forward_triton/attention_forward_triton.py`

## Public theorem: `attention_forward_triton_closed_form_correct`

<details><summary>docstring</summary>

```
/-- **Closed-form correctness for `attention_forward_triton` (general statement).**

For arbitrary batch/head strides, head count, block sizes, KV-block count,
head/active dimensions and arbitrary `q_scale`/`k_scale`, every active output
lane of `Out` (`mIndex < N_CTX ∧ head < HEAD_ACTIVE`) equals
`attentionRealBase2PerKeyScale` of the loaded Q/K/V tiles under the per-block key
scale — the genuine base-2, per-key-scaled attention output, NOT the kernel's own
executed value. Inactive lanes are unconstrained (masked out by the write map).

Layout contracts: `N_CTX = BLOCK_N · numKVBlocks`, `stride_qm = stride_kn =
HEAD_DIM` and head stride `1` (so the per-block pointer advance composes into a
per-key address), `0 < BLOCK_N`, `HEAD_ACTIVE ≤ BLOCK_DMODEL`. The Python test
case (`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128, BLOCK_N=64,
HEAD_ACTIVE=96`, `q_scale = k_scale = 1`) is the special case.

**Proven sorry-free**: bridges (via `realizes_writeIf_iff` +
`computeCorrect_of_toAlgKernel`) to
`VeriTile.Examples.AttentionForwardClosedForm.attention_forward_triton_closed_form_correct`,
whose full `exec`-side loop unfolding (preLoop + per-block step + postLoop) and
math core (`Math/Attention.lean`) are both complete. Extra preconditions:
`HEAD_ACTIVE ≤ HEAD_DIM` (store-offset injectivity), clean initial `undef`.
Tracked as `attention-forward-online-softmax-recurrence`, #162. -/
```
</details>

**Statement:**
```lean
specification attention_forward_triton_closed_form_correct
    (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (hBN : 0 < BLOCK_N) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (hHD : HEAD_ACTIVE ≤ HEAD_DIM) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_forward_triton_surface Q K V Q_scale K_scale Out
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1
        Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
        HEAD_ACTIVE STAGE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s (BLOCK_N * numKVBlocks) HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if h : idx.2.1.val < HEAD_ACTIVE then
          attentionRealBase2PerKeyScale
            (qTile s Q H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE)
            (kTile s K H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (vTile s V H stride_qz stride_qh HEAD_DIM (BLOCK_N * numKVBlocks) HEAD_ACTIVE)
            (keyScale s Q_scale K_scale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N
              (BLOCK_N * numKVBlocks))
            (idx.1, ⟨idx.2.1.val, h⟩, PUnit.unit)
        else (0 : ℝ))
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL`
- `hHD : HEAD_ACTIVE ≤ HEAD_DIM`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `attention_forward_triton_surface`, `active`, `outOffset`, `qTile`, `kTile`, `vTile`, `keyScale`, `mIndex`, `kIndex`, `offZ`, `offH`, `baseOffset`

<details><summary><code>attention_forward_triton_surface</code></summary>

```
/-- Full Lean port of `attention_forward_triton.py`'s `_attn_fwd`.

The upstream kernel calls a separate `@triton.jit` helper `_attn_fwd_inner` to
run the K/V streaming-softmax loop. The DSL has no function-call surface, so the
helper body is inlined verbatim into the outer kernel; semantically the two
forms are identical for this fixed-stage path.

The literal `128` and `96` in the upstream kernel correspond to the
`BLOCK_DMODEL` / `HEAD_ACTIVE` parameters threaded through the bundled tests
(`head_dim = 128`, with the inner dot using only the first 96 lanes of the head
dimension). They appear here as explicit Lean parameters. -/
```
```lean
def attention_forward_triton_surface
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
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
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
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
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

<details><summary><code>qTile</code></summary>

```lean
noncomputable def qTile (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE : Nat) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (baseOffset s H stride_qz stride_qh + mIndex s BLOCK_M i * HEAD_DIM + e.val)
```
</details>

<details><summary><code>kTile</code></summary>

```lean
noncomputable def kTile (s : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + e.val)
```
</details>

<details><summary><code>vTile</code></summary>

```lean
noncomputable def vTile (s : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + d.val)
```
</details>

<details><summary><code>keyScale</code></summary>

```
/-- Per-key scale `q_scale · k_scale[block(j)]`, `block(j) = j / BLOCK_N`.
`q_scale` is read at `off_hz · cdiv(N_CTX, BLOCK_M) + pid₀`; `k_scale[b]` at
`off_hz · cdiv(N_CTX, BLOCK_N) + b`. -/
```
```lean
noncomputable def keyScale (s : BlockState) (Q_scale K_scale : RegionName)
    (N_CTX BLOCK_M BLOCK_N S : Nat) :
    Fin S → ℝ :=
  fun j =>
    s.readMem Q_scale (s.pids 1 * cdiv N_CTX BLOCK_M + s.pids 0) *
      s.readMem K_scale (s.pids 1 * cdiv N_CTX BLOCK_N + j.val / BLOCK_N)
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

<details><summary><code>baseOffset</code></summary>

```
/-- Batch/head base offset `off_z · stride_qz + off_h · stride_qh`. -/
```
```lean
def baseOffset (s : BlockState) (H stride_qz stride_qh : Nat) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh
```
</details>

## Public theorem: `attention_forward_triton_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` io headline — on the `StreamMasked3DKernelIO₅` skin.** On its
five-stream single-store signature (`Q`/`K`/`V` tile channels plus the two
scalar-width `Q_scale`/`K_scale` channels), `_attn_fwd` implements the genuine
closed form: output lane `j = (i, e)` of `Out` holds
`attnFwdIOOutSpec` — base-2 per-key-scale attention over the tiles assembled
from the five streams, with `keyScale j = Q_scale · K_scale[j / BLOCK_N]` — for
every rounding model that is trivial on the fp16 grid.

**Hypothesis provenance**:
* `hfp16` pins `R.round .fp16 = id` — loop-body statement 14
  (`p = (p).to(tl.float16)`, and the `out_dtype=tl.float16` dot it feeds) is an
  *in-loop* rounding event outside the skin's single-boundary-round shape;
  this is the file's declared fp16 modeling boundary (the exact stack above
  already treats this cast as the identity), now explicit as a hypothesis.
* `hBD`/`hBN`/`hnum` positivity shape the KV walk (`N_CTX = BLOCK_N ·
  numKVBlocks > 0`) — inherited from the exact headline
  `attention_forward_triton_closed_form_correct`.
* `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL` — the head mask lives inside the
  block width; it is what makes the stream lane `i·BLOCK_DMODEL + e` exist for
  every head-active `e`.
* `hHD : HEAD_ACTIVE ≤ HEAD_DIM` — contiguous-row `Out` window injectivity
  (host launches use `HEAD_DIM = BLOCK_DMODEL`), inherited verbatim.

The exact headline's `hundef` is **not** a hypothesis here — the skin's Hoare
triple carries the `undef` pin itself. The output grid is the `.real` default:
the store-side `.to(Out.type.element_ty)` cast **erases at translation**
(`attnStoreStmt` is a bare `Stmt.store .real …`), so at every such `R` the
terminal cells carry the exact fold values. -/
```
</details>

**Statement:**
```lean
specification attention_forward_triton_io_correctness (R : RoundingModel)
    (hfp16 : R.round .fp16 = id)
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hHD : HEAD_ACTIVE ≤ HEAD_DIM) :
    attnFwdIO Q K V QScale KScale Out stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N
        BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨[R]
      fun p₀ _ _ xs ys zs ws vs j =>
        if h : j.val % BLOCK_DMODEL < HEAD_ACTIVE then
          attnFwdIOOutSpec p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks xs ys zs ws vs
            ((Lane2D.decode j).1, ⟨j.val % BLOCK_DMODEL, h⟩, PUnit.unit)
        else 0
```

**Assumptions / layout contracts:**
- `hfp16 : R.round .fp16 = id`
- `hBD : 0 < BLOCK_DMODEL`
- `hBN : 0 < BLOCK_N`
- `hnum : 0 < numKVBlocks`
- `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL`
- `hHD : HEAD_ACTIVE ≤ HEAD_DIM`

**Closed-form spec defs (transitive):** `attnFwdIO`, `attnFwdIOOutSpec`, `attention_forward_triton_surface`, `attnIOqTm`, `attnIOkTm`, `attnIOvTm`, `attnIOkeyScale`, `qTile`, `kTile`, `vTile`, `keyScale`, `baseOffset`, `mIndex`, `outOffset`, `offZ`, `offH`, `kIndex`

<details><summary><code>attnFwdIO</code></summary>

```
/-- **Streaming IO signature** of `_attn_fwd` on the five-stream single-store
attention fold skin (`T = numKVBlocks` under `N_CTX = BLOCK_N · numKVBlocks`;
the trip count is pid-free, so there is no launch-legality `pre` analog to
carry). -/
```
```lean
def attnFwdIO (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat) : StreamMasked3DKernelIO₅ where
  kernel := attention_forward_triton_surface Q K V QScale KScale Out
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
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
  read4 := fun p₀ p₁ _ _ _ =>
    p₁ * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + p₀
  read5 := fun _ p₁ _ t _ =>
    p₁ * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + t.val
  write := fun p₀ p₁ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  mask1 := fun p₀ _ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask2 := fun _ _ _ t j =>
    j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE
  mask3 := fun _ _ _ t j =>
    j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask4 := fun _ _ _ _ _ => True
  mask5 := fun _ _ _ _ _ => True
  writeMask := fun p₀ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks
      ∧ j.val % BLOCK_DMODEL < HEAD_ACTIVE
```
</details>

<details><summary><code>attnFwdIOOutSpec</code></summary>

```
/-- **The streamed closed-form spec `f`**: base-2 per-key-scale attention over
the tiles assembled from the five streams (`attentionRealBase2PerKeyScale`
restated tile-parametrically; the contraction axis is the `HEAD_ACTIVE`
active head lanes, exactly as in the exact stack). -/
```
```lean
noncomputable def attnFwdIOOutSpec
    (p₀ BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (ws vs : Fin T → Fin 1 → ℝ)
    (idx : TileIndex [BLOCK_M, HEAD_ACTIVE]) : ℝ :=
  attentionRealBase2PerKeyScale
    (attnIOqTm p₀ (BLOCK_N * T) BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T xs)
    (attnIOkTm BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T ys)
    (attnIOvTm BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T zs)
    (attnIOkeyScale BLOCK_N T ws vs) idx

/-! ### Stream-pin tile bridges

Under the skin's input pins the kernel-side tiles/scales of the exact stack
(`VeriTile/Examples/AttentionForwardClosedForm.lean`) coincide with the
stream-built ones. The exact stack's names are opened under an `ex`-prefix to
keep them apart from this file's own layout vocabulary. -/

open VeriTile.Examples.AttentionForwardClosedForm renaming
  qTile → exQTile, kTile → exKTile, vTile → exVTile, keyScale → exKeyScale,
  baseOffset → exBaseOffset, mIndex → exMIndex, outOffset → exOutOffset, cdiv → exCdiv
```
</details>

<details><summary><code>attention_forward_triton_surface</code></summary>

```
/-- Full Lean port of `attention_forward_triton.py`'s `_attn_fwd`.

The upstream kernel calls a separate `@triton.jit` helper `_attn_fwd_inner` to
run the K/V streaming-softmax loop. The DSL has no function-call surface, so the
helper body is inlined verbatim into the outer kernel; semantically the two
forms are identical for this fixed-stage path.

The literal `128` and `96` in the upstream kernel correspond to the
`BLOCK_DMODEL` / `HEAD_ACTIVE` parameters threaded through the bundled tests
(`head_dim = 128`, with the inner dot using only the first 96 lanes of the head
dimension). They appear here as explicit Lean parameters. -/
```
```lean
def attention_forward_triton_surface
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
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
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
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
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

<details><summary><code>attnIOqTm</code></summary>

```
/-- The **row-masked query tile** read off the static first stream (the window
ignores `t`, so the step-`0` slice carries the whole tile; rows whose global
index is `≥ N_CTX` are `0`, mirroring the kernel's `q` row mask). -/
```
```lean
noncomputable def attnIOqTm (p₀ NC BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : p₀ * BLOCK_M + idx.1.val < NC ∧ 0 < T ∧ idx.2.1.val < BLOCK_DMODEL then
      xs ⟨0, h.2.1⟩ (Lane2D.encode (idx.1, ⟨idx.2.1.val, h.2.2⟩, PUnit.unit))
    else 0
```
</details>

<details><summary><code>attnIOkTm</code></summary>

```
/-- The **global key tile** read off the transposed `K` stream: global key `jg`
lives in step `jg / BLOCK_N` at block-local column `jg % BLOCK_N`. -/
```
```lean
noncomputable def attnIOkTm (BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ) :
    TileIndex [BLOCK_N * T, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : idx.2.1.val < BLOCK_DMODEL ∧ idx.1.val / BLOCK_N < T ∧ 0 < BLOCK_N then
      ys ⟨idx.1.val / BLOCK_N, h.2.1⟩
        (Lane2D.encode (⟨idx.2.1.val, h.1⟩, ⟨idx.1.val % BLOCK_N, Nat.mod_lt _ h.2.2⟩, PUnit.unit))
    else 0
```
</details>

<details><summary><code>attnIOvTm</code></summary>

```
/-- The **global value tile** read off the `V` stream (row-major per-step
tiles). -/
```
```lean
noncomputable def attnIOvTm (BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) :
    TileIndex [BLOCK_N * T, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : idx.2.1.val < BLOCK_DMODEL ∧ idx.1.val / BLOCK_N < T ∧ 0 < BLOCK_N then
      zs ⟨idx.1.val / BLOCK_N, h.2.1⟩
        (Lane2D.encode (⟨idx.1.val % BLOCK_N, Nat.mod_lt _ h.2.2⟩, ⟨idx.2.1.val, h.1⟩, PUnit.unit))
    else 0
```
</details>

<details><summary><code>attnIOkeyScale</code></summary>

```
/-- The per-key score-scale carrier `q_scale · k_scale` read off the two
scalar streams: the static `Q_scale` slot (step `0`) times key `j`'s
`K_scale` slot (step `j / BLOCK_N`). -/
```
```lean
noncomputable def attnIOkeyScale (BLOCK_N T : Nat) (ws vs : Fin T → Fin 1 → ℝ) :
    Fin (BLOCK_N * T) → ℝ :=
  fun j =>
    (if h : 0 < T then ws ⟨0, h⟩ 0 else 0)
      * (if h : j.val / BLOCK_N < T then vs ⟨j.val / BLOCK_N, h⟩ 0 else 0)
```
</details>

<details><summary><code>qTile</code></summary>

```lean
noncomputable def qTile (s : BlockState) (Q : RegionName)
    (H stride_qz stride_qh HEAD_DIM BLOCK_M HEAD_ACTIVE : Nat) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (baseOffset s H stride_qz stride_qh + mIndex s BLOCK_M i * HEAD_DIM + e.val)
```
</details>

<details><summary><code>kTile</code></summary>

```lean
noncomputable def kTile (s : BlockState) (K : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + e.val)
```
</details>

<details><summary><code>vTile</code></summary>

```lean
noncomputable def vTile (s : BlockState) (V : RegionName)
    (H stride_qz stride_qh HEAD_DIM S HEAD_ACTIVE : Nat) :
    TileIndex [S, HEAD_ACTIVE] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (baseOffset s H stride_qz stride_qh + j.val * HEAD_DIM + d.val)
```
</details>

<details><summary><code>keyScale</code></summary>

```
/-- Per-key scale `q_scale · k_scale[block(j)]`, `block(j) = j / BLOCK_N`.
`q_scale` is read at `off_hz · cdiv(N_CTX, BLOCK_M) + pid₀`; `k_scale[b]` at
`off_hz · cdiv(N_CTX, BLOCK_N) + b`. -/
```
```lean
noncomputable def keyScale (s : BlockState) (Q_scale K_scale : RegionName)
    (N_CTX BLOCK_M BLOCK_N S : Nat) :
    Fin S → ℝ :=
  fun j =>
    s.readMem Q_scale (s.pids 1 * cdiv N_CTX BLOCK_M + s.pids 0) *
      s.readMem K_scale (s.pids 1 * cdiv N_CTX BLOCK_N + j.val / BLOCK_N)
```
</details>

<details><summary><code>baseOffset</code></summary>

```
/-- Batch/head base offset `off_z · stride_qz + off_h · stride_qh`. -/
```
```lean
def baseOffset (s : BlockState) (H stride_qz stride_qh : Nat) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

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

<details><summary><code>kIndex</code></summary>

```lean
def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

## Also present (pinned special-case summaries)
- `attention_forward_triton_final_store_slice_compute_correct`
