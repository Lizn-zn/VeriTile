# Spec sheet — `bench/tritonbench_g/attention_fwd_triton2/AttentionFwdTriton2.lean`

**Python source:** `bench/tritonbench_g/attention_fwd_triton2/attention_fwd_triton2.py`

## Public theorem: `attention_fwd_triton2_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general output summary for `attention_fwd_triton2` (no test-shape pin).**

Mirrors the reference `attention_forward_triton_closed_form_correct`: over
*symbolic* batch/head strides, head count `H`, block sizes `BLOCK_M`/`BLOCK_N`,
KV-block count `numKVBlocks` (so `N_CTX = BLOCK_N · numKVBlocks`), head/active
dimensions and arbitrary `q_scale`/`k_scale`, this combines

* the checked full-surface lowering to the algorithm layer
  (`attention_fwd_triton2_surface_toAlgorithm_supported`), and
* the genuine closed-form value of every active `Out` lane
  (`attention_fwd_triton2_closed_form_correct`):
  `attentionRealBase2PerKeyScale` of the loaded Q/K/V tiles under the per-block
  key scale — the base-2, per-key-scaled attention output reading INPUT Q/K/V
  memory, NOT the kernel's own executed value.

The only layout assumptions are the contiguity contracts the kernel relies on
(`stride_qm = stride_kn = HEAD_DIM`, head stride `1`), `0 < BLOCK_N`,
`HEAD_ACTIVE ≤ BLOCK_DMODEL`, `HEAD_ACTIVE ≤ HEAD_DIM`, and a clean initial
`undef`. The Python test case (`B=2, H=4, N_CTX=128, HEAD_DIM=128, BLOCK_M=128,
BLOCK_N=64, HEAD_ACTIVE=96, numKVBlocks=2`) is the special case. -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton2_output_summary_general
    (Q K V Q_scale K_scale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh Z H BLOCK_M BLOCK_N numKVBlocks
      HEAD_DIM BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
    (hBN : 0 < BLOCK_N) (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL)
    (hHD : HEAD_ACTIVE ≤ HEAD_DIM) (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (attention_fwd_triton2_surface Q K V Q_scale K_scale Out
      stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1
      Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL
      HEAD_ACTIVE STAGE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL`
- `hHD : HEAD_ACTIVE ≤ HEAD_DIM`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `attention_fwd_triton2_surface`, `cdiv`

<details><summary><code>attention_fwd_triton2_surface</code></summary>

```
/-- Full Lean port of `attention_fwd_triton2.py`'s `_attn_fwd`.

The upstream kernel calls a separate `@triton.jit` helper `_attn_fwd_inner` to
run the K/V streaming-softmax loop. The DSL has no function-call surface, so the
helper body is inlined verbatim into the outer kernel; semantically the two
forms are identical for this fixed-stage path. The upstream `v.to(tl.float16)`
dot-input cast and the `bfloat16` output cast erase to the identity over `ℝ`, so
this surface is the same inlined online-softmax loop verified for
`attention_forward_triton`.

The literal `128` and `96` in the upstream kernel correspond to the
`BLOCK_DMODEL` / `HEAD_ACTIVE` parameters threaded through the bundled tests
(`head_dim = 128`, with the inner dot using only the first 96 lanes of the head
dimension). They appear here as explicit Lean parameters. -/
```
```lean
def attention_fwd_triton2_surface
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

<details><summary><code>cdiv</code></summary>

```
/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
```
```lean
def cdiv (a b : Nat) : Nat := (a + b - 1) / b
```
</details>

## Public theorem: `attention_fwd_triton2_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` io headline — on the `StreamMasked3DKernelIO₅` skin.** On its
five-stream single-store signature (three tile channels `Q`/`K`/`V` plus two
**scalar-width** `B = 1` channels `Q_scale`/`K_scale`), `_attn_fwd` implements
the genuine non-causal closed form: output lane `j = (i, e)` of `Out` holds
`attentionFwdTriton2IOOutSpec` — base-2 per-key-scale attention of the streamed
Q/K/V tiles with `keyScale j = Q_scale · K_scale[j / BLOCK_N]` assembled from the
two scalar channels — for every rounding model that is trivial on the fp16 grid.

**Hypothesis provenance**:
* `hfp16` pins `R.round .fp16 = id` — the in-loop narrowing casts
  (`p = p.to(tl.float16)`, the `v.to(tl.float16)` dot input, `out_dtype=
  tl.float16`) are *in-loop* rounding events outside the skin's
  single-boundary-round shape; this is the file's declared fp16 modeling
  boundary (the exec stack already treats these casts as the identity), now
  explicit as a headline hypothesis.
* `hBN`/`hnum` positivity shape the KV walk (`N_CTX = BLOCK_N · numKVBlocks > 0`)
  — inherited verbatim from the exact headline
  `attention_fwd_triton2_output_summary_general`.
* `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL` — the head mask selects a prefix of
  the head axis (the Python `< 96` inside `tl.arange(0, 128)`); it makes the
  streamed lane encodings land in range.
* `hBDHD : BLOCK_DMODEL ≤ HEAD_DIM` — host launches use
  `HEAD_DIM = BLOCK_DMODEL`; together with `hActiveLe` it supplies the exact
  headline's `HEAD_ACTIVE ≤ HEAD_DIM` contract.

The exact headline's `hundef` is **not** a hypothesis here — the skin's Hoare
triple carries the `undef` pin itself. The output grid is the `.real` default:
the surface's store-side `.to(Out.type.element_ty)` erases at translation, so
the lowered terminal statement is a plain `Stmt.store .real` and the host's
`torch.bfloat16` allocation is invisible to the model (no `hbf16` hypothesis is
needed, or provable). At every such `R` the terminal cells carry the exact fold
values. -/
```
</details>

**Statement:**
```lean
specification attention_fwd_triton2_io_correctness (R : RoundingModel)
    (hfp16 : R.round .fp16 = id)
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL) (hBDHD : BLOCK_DMODEL ≤ HEAD_DIM) :
    attentionFwdTriton2IO Q K V Q_scale K_scale Out stride_qz stride_qh Z H HEAD_DIM
        BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE numKVBlocks ⊨[R]
      fun _ _ _ xs ys zs ws vs j =>
        attentionFwdTriton2IOOutSpec BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
          xs ys zs ws vs (Lane2D.decode j)
```

**Assumptions / layout contracts:**
- `hfp16 : R.round .fp16 = id`
- `hBN : 0 < BLOCK_N`
- `hnum : 0 < numKVBlocks`
- `hActiveLe : HEAD_ACTIVE ≤ BLOCK_DMODEL`
- `hBDHD : BLOCK_DMODEL ≤ HEAD_DIM`

**Closed-form spec defs (transitive):** `attentionFwdTriton2IO`, `attentionFwdTriton2IOOutSpec`, `attention_fwd_triton2_surface`, `cdiv`, `aft2IOqT`, `aft2IOkT`, `aft2IOvT`, `aft2IOkeyScale`

<details><summary><code>attentionFwdTriton2IO</code></summary>

```
/-- **Streaming IO signature** of `_attn_fwd` on the five-stream single-store
attention fold skin (`T = numKVBlocks` under `N_CTX = BLOCK_N · numKVBlocks`;
the trip count is pid-free, so there is no launch-legality `pre` analog). -/
```
```lean
def attentionFwdTriton2IO (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh Z H HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
      numKVBlocks : Nat) : StreamMasked3DKernelIO₅ where
  kernel := attention_fwd_triton2_surface Q K V Q_scale K_scale Out
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
    Z H (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := Q_scale
  inp5 := K_scale
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
  read4 := fun p₀ p₁ _ _ _ => p₁ * cdiv (BLOCK_N * numKVBlocks) BLOCK_M + p₀
  read5 := fun _ p₁ _ t _ => p₁ * cdiv (BLOCK_N * numKVBlocks) BLOCK_N + t.val
  write := fun p₀ p₁ _ j =>
    p₁ / H * stride_qz + p₁ % H * stride_qh
      + (p₀ * BLOCK_M + j.val / BLOCK_DMODEL) * HEAD_DIM + j.val % BLOCK_DMODEL
  mask1 := fun p₀ _ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks ∧
      j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask2 := fun _ _ _ t j =>
    j.val % BLOCK_N < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧ j.val / BLOCK_N < HEAD_ACTIVE
  mask3 := fun _ _ _ t j =>
    j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks - t.val * BLOCK_N ∧
      j.val % BLOCK_DMODEL < HEAD_ACTIVE
  mask4 := fun _ _ _ _ _ => True
  mask5 := fun _ _ _ _ _ => True
  writeMask := fun p₀ _ _ j =>
    p₀ * BLOCK_M + j.val / BLOCK_DMODEL < BLOCK_N * numKVBlocks ∧
      j.val % BLOCK_DMODEL < HEAD_ACTIVE
```
</details>

<details><summary><code>attentionFwdTriton2IOOutSpec</code></summary>

```
/-- **The streamed closed-form spec `f`**: base-2 per-key-scale attention of the
streamed Q/K/V tiles under the streamed per-block key scale. Head-inactive
output lanes (which the store masks off) are `0`. -/
```
```lean
noncomputable def attentionFwdTriton2IOOutSpec
    (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ)
    (ws vs : Fin T → Fin 1 → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  if h : idx.2.1.val < HEAD_ACTIVE then
    attentionRealBase2PerKeyScale
      (aft2IOqT BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T xs)
      (aft2IOkT BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T ys)
      (aft2IOvT BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T zs)
      (aft2IOkeyScale BLOCK_N T ws vs)
      (idx.1, ⟨idx.2.1.val, h⟩, PUnit.unit)
  else 0
```
</details>

<details><summary><code>attention_fwd_triton2_surface</code></summary>

```
/-- Full Lean port of `attention_fwd_triton2.py`'s `_attn_fwd`.

The upstream kernel calls a separate `@triton.jit` helper `_attn_fwd_inner` to
run the K/V streaming-softmax loop. The DSL has no function-call surface, so the
helper body is inlined verbatim into the outer kernel; semantically the two
forms are identical for this fixed-stage path. The upstream `v.to(tl.float16)`
dot-input cast and the `bfloat16` output cast erase to the identity over `ℝ`, so
this surface is the same inlined online-softmax loop verified for
`attention_forward_triton`.

The literal `128` and `96` in the upstream kernel correspond to the
`BLOCK_DMODEL` / `HEAD_ACTIVE` parameters threaded through the bundled tests
(`head_dim = 128`, with the inner dot using only the first 96 lanes of the head
dimension). They appear here as explicit Lean parameters. -/
```
```lean
def attention_fwd_triton2_surface
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

<details><summary><code>cdiv</code></summary>

```
/-- Ceiling division `⌈a / b⌉`, matching Triton's `tl.cdiv`. -/
```
```lean
def cdiv (a b : Nat) : Nat := (a + b - 1) / b
```
</details>

<details><summary><code>aft2IOqT</code></summary>

```
/-- The **query tile** read off the static first stream (the window ignores `t`,
so the step-`0` slice carries the whole tile). -/
```
```lean
noncomputable def aft2IOqT (BLOCK_M BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (xs : Fin T → Fin (BLOCK_M * BLOCK_DMODEL) → ℝ) :
    TileIndex [BLOCK_M, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : 0 < T ∧ idx.1.val * BLOCK_DMODEL + idx.2.1.val < BLOCK_M * BLOCK_DMODEL then
      xs ⟨0, h.1⟩ ⟨idx.1.val * BLOCK_DMODEL + idx.2.1.val, h.2⟩
    else 0
```
</details>

<details><summary><code>aft2IOkT</code></summary>

```
/-- The **global key tile** read off the transposed `K` stream: global key `jg`
lives in step `jg / BLOCK_N` at block-local column `jg % BLOCK_N`. -/
```
```lean
noncomputable def aft2IOkT (BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (ys : Fin T → Fin (BLOCK_DMODEL * BLOCK_N) → ℝ) :
    TileIndex [BLOCK_N * T, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : idx.1.val / BLOCK_N < T ∧
        idx.2.1.val * BLOCK_N + idx.1.val % BLOCK_N < BLOCK_DMODEL * BLOCK_N then
      ys ⟨idx.1.val / BLOCK_N, h.1⟩ ⟨idx.2.1.val * BLOCK_N + idx.1.val % BLOCK_N, h.2⟩
    else 0
```
</details>

<details><summary><code>aft2IOvT</code></summary>

```
/-- The **global value tile** read off the `V` stream (row-major per-step
tiles). -/
```
```lean
noncomputable def aft2IOvT (BLOCK_N BLOCK_DMODEL HEAD_ACTIVE T : Nat)
    (zs : Fin T → Fin (BLOCK_N * BLOCK_DMODEL) → ℝ) :
    TileIndex [BLOCK_N * T, HEAD_ACTIVE] → ℝ :=
  fun idx =>
    if h : idx.1.val / BLOCK_N < T ∧
        idx.1.val % BLOCK_N * BLOCK_DMODEL + idx.2.1.val < BLOCK_N * BLOCK_DMODEL then
      zs ⟨idx.1.val / BLOCK_N, h.1⟩ ⟨idx.1.val % BLOCK_N * BLOCK_DMODEL + idx.2.1.val, h.2⟩
    else 0
```
</details>

<details><summary><code>aft2IOkeyScale</code></summary>

```
/-- The per-key score-scale carrier `q_scale · k_scale` read off the two
scalar-width streams: the static `Q_scale` slot (step `0`) times key `j`'s
`K_scale` slot (step `j / BLOCK_N`). -/
```
```lean
noncomputable def aft2IOkeyScale (BLOCK_N T : Nat) (ws vs : Fin T → Fin 1 → ℝ) :
    Fin (BLOCK_N * T) → ℝ :=
  fun j =>
    (if h : 0 < T then ws ⟨0, h⟩ 0 else 0)
      * (if h : j.val / BLOCK_N < T then vs ⟨j.val / BLOCK_N, h⟩ 0 else 0)
```
</details>

## Also present (pinned special-case summaries)
- `attention_fwd_triton2_final_store_slice_compute_correct`
- `attention_fwd_triton2_closed_form_correct`
