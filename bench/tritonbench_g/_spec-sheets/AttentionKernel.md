# Spec sheet — `bench/tritonbench_g/attention_kernel/AttentionKernel.lean`

**Python source:** `bench/tritonbench_g/attention_kernel/attention_kernel.py`

## Public theorem: `attention_kernel_genuine_output_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **General genuine closed-form `Out`-store correctness** (dimension-parameterized).
Every output lane of `_fwd_kernel_aligned` holds the closed-form base-2
streaming-softmax attention `attentionKernelSpec` (= `attnGenScore fscore vFlat`)
under the kernel's genuine bias-augmented per-key score `fscore` — NOT the
kernel's own executed output. Genuinely general over
`BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE numKVBlocks sm_scale` and the head/bias
strides (`P_SEQ = 0`, contiguous Q/K/V/Out layout per `qRaw`/`kFlat`/`vFlat`). -/
```
</details>

**Statement:**
```lean
specification attention_kernel_genuine_output_compute_correct_general
    (Q K V B0 Out : RegionName) (s : BlockState) (sm_scale : ℝ)
    (stride_qh stride_kh stride_b0h BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB : Nat)
    (hKN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hHD : 0 < HEAD) (hnB : 1 ≤ nB)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
        FloatDType.fp16)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, HEAD] =>
        some (Out, surfaceOutOffset s stride_qh HEAD 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, HEAD] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (attentionKernelSpec s Q K V B0 sm_scale
            (s.pids 1 * stride_qh) (s.pids 1 * stride_kh) (s.pids 1 * stride_b0h)
            BLOCK_M BLOCK_N HEAD (BLOCK_N * nB) BIAS_LAST_SIZE stride_b0m (s.pids 0) idx))))
```

**Assumptions / layout contracts:**
- `hKN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`
- `hHD : 0 < HEAD`
- `hnB : 1 ≤ nB`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `attention_kernel_fwd_kernel_aligned_surface`, `surfaceOutOffset`, `attentionKernelSpec`, `mIndex`, `kIndex`, `fscore`, `vFlat`, `qRaw`, `kFlat`, `b0Val`, `b1Val`

<details><summary><code>attention_kernel_fwd_kernel_aligned_surface</code></summary>

```
/-- Faithful DSL port of `attention_kernel.py`'s `_fwd_kernel_aligned`. -/
```
```lean
def attention_kernel_fwd_kernel_aligned_surface
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn
      stride_oh stride_om stride_on
      stride_b0h stride_b0m
      _Z _H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (out_dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  q_offset = off_hz * $(stride_qh)
  kv_offset = off_hz * $(stride_kh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + kv_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + kv_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(OUT_DTYPE)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = tl.arange(0, $(BLOCK_M))
  b_offset = off_hz * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :])
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=OUT_DTYPE)
    qk += tl.dot(q, k)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N))
    qk += ((b0 + b1) * 1.44269504)

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += tl.dot((p).to(OUT_DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc / l_i[:, None]
  O_block_ptr = tl.make_block_ptr(base=Out + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(OUT_DTYPE))
}
```
</details>

<details><summary><code>surfaceOutOffset</code></summary>

```lean
def surfaceOutOffset
    (s : BlockState)
    (stride_qh stride_om stride_on BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  s.pids 1 * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_om + kIndex idx * stride_on
```
</details>

<details><summary><code>attentionKernelSpec</code></summary>

```
/-- **Closed-form target.** The kernel's genuine specification is
`attnGenScore (fscore …) (vFlat …)`: the base-2 streaming softmax over the
genuine per-key score `fscore` (scaled dot + additive bias). The generalized
math lemma `closed_form_g` establishes that the running `oPg / lPg` recurrence
the kernel's loop maintains converges to exactly this value after all
`numKVBlocks` key blocks — which is what the remaining `exec`-side assembly
discharges. -/
```
```lean
noncomputable def attentionKernelSpec (s0 : BlockState) (Q K V B0 : RegionName)
    (sm_scale : ℝ) (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m : Nat)
    (start_m : Nat) : TileIndex [BLOCK_M, HEAD_DIM] → ℝ :=
  attnGenScore
    (fscore s0 Q K B0 sm_scale q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m start_m)
    (vFlat s0 V kv_offset HEAD_DIM N_CTX)
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

<details><summary><code>fscore</code></summary>

```
/-- **Genuine per-key score** `fscore r j` of `_fwd_kernel_aligned`:
`qk_scale·(Σ_e Q[r,e]·K[e,j]) + (b0[r, j/BN] + b1[r, j%BN])·log2 e`,
with `qk_scale = sm_scale · log2 e` already folded into the pre-scaled `q`. This
is the `score` argument of `VeriTile.Triton.attnGenScore`, whose batch base-2
softmax `(Σ 2^fscore · V) / (Σ 2^fscore)` is the kernel's closed form (see
`closed_form_g`). -/
```
```lean
noncomputable def fscore (s0 : BlockState) (Q K B0 : RegionName)
    (sm_scale : ℝ) (q_offset kv_offset b_offset
      BLOCK_M BLOCK_N HEAD_DIM N_CTX BIAS_LAST_SIZE stride_b0m : Nat)
    (start_m : Nat)
    (r : Fin BLOCK_M) (j : Fin N_CTX) : ℝ :=
  (sm_scale * 1.44269504) *
      Finset.univ.sum (fun e : Fin HEAD_DIM =>
        qRaw s0 Q q_offset BLOCK_M HEAD_DIM start_m (r, e, PUnit.unit)
          * kFlat s0 K kv_offset HEAD_DIM N_CTX e j)
    + (b0Val s0 B0 b_offset BLOCK_M stride_b0m start_m r (j.val / BLOCK_N)
        + b1Val s0 B0 b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE start_m r (j.val % BLOCK_N))
      * 1.44269504
```
</details>

<details><summary><code>vFlat</code></summary>

```
/-- Loaded V tile, flat per-key over `[N_CTX, HEAD_DIM]`. -/
```
```lean
noncomputable def vFlat (s0 : BlockState) (V : RegionName)
    (kv_offset HEAD_DIM N_CTX : Nat) :
    TileIndex [N_CTX, HEAD_DIM] → ℝ :=
  fun (j, d, _) => s0.readMem V (kv_offset + j.val * HEAD_DIM + d.val)
```
</details>

<details><summary><code>qRaw</code></summary>

```
/-- Loaded (pre-scale) Q tile: row `r`, head lane `e`. -/
```
```lean
noncomputable def qRaw (s0 : BlockState) (Q : RegionName)
    (q_offset BLOCK_M HEAD_DIM : Nat) (start_m : Nat) :
    TileIndex [BLOCK_M, HEAD_DIM] → ℝ :=
  fun (r, e, _) => s0.readMem Q (q_offset + (start_m * BLOCK_M + r.val) * HEAD_DIM + e.val)
```
</details>

<details><summary><code>kFlat</code></summary>

```
/-- Loaded K tile as a flat per-key function over `[HEAD_DIM, N_CTX]`. -/
```
```lean
noncomputable def kFlat (s0 : BlockState) (K : RegionName)
    (kv_offset HEAD_DIM N_CTX : Nat) :
    Fin HEAD_DIM → Fin N_CTX → ℝ :=
  fun e j => s0.readMem K (kv_offset + e.val + j.val * HEAD_DIM)
```
</details>

<details><summary><code>b0Val</code></summary>

```
/-- Per-row, per-block-column bias `b0` read (`c = j / BLOCK_N`). -/
```
```lean
noncomputable def b0Val (s0 : BlockState) (B0 : RegionName)
    (b_offset BLOCK_M stride_b0m : Nat) (start_m : Nat)
    (r : Fin BLOCK_M) (c : Nat) : ℝ :=
  s0.readMem B0 (b_offset + (start_m * BLOCK_M + r.val) * stride_b0m + c)
```
</details>

<details><summary><code>b1Val</code></summary>

```
/-- Per-row, per-lane bias `b1` read at lane `jL` (`jL = j % BLOCK_N`). -/
```
```lean
noncomputable def b1Val (s0 : BlockState) (B0 : RegionName)
    (b_offset BLOCK_M stride_b0m BIAS_LAST_SIZE : Nat) (start_m : Nat)
    (r : Fin BLOCK_M) (jL : Nat) : ℝ :=
  s0.readMem B0
    (b_offset + (start_m * BLOCK_M + r.val) * stride_b0m + (jL % BIAS_LAST_SIZE + BIAS_LAST_SIZE))
```
</details>

## Public theorem: `attention_kernel_io_correctness`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` io headline** — the `StreamMasked3DKernelIO₄` exemplar. On
its four-stream single-output signature, `_fwd_kernel_aligned` implements the
genuine base-2 streaming-softmax closed form `attnGenScore` of the streamed
`Q`/`K`/`V` tiles under the kernel's bias-augmented per-key score
(`akIOscoreT`, fed by the two-window `B0` lane pack), quantized once at the
`.fp16` grid — for every rounding model that is trivial on the fp16 grid.

Hypothesis provenance:
* `hfp16` — the kernel has *in-loop* fp16 rounding events (the pre-scaled
  fp16 `q` feeding every step's dot, the fp16 `qk` seed, and the fp16
  round-trip on `p`), outside the skin's single-boundary-round shape; exactly
  the file's declared fp16 modeling boundary, now explicit.
* `hBM`/`hBN`/`hHD`/`hnB` — positivity/nonemptiness forced by the exact
  stack (`attention_kernel_exec_general`): block-max reductions need
  `0 < BLOCK_N`, the store-lane row/column split needs `0 < BLOCK_M` and
  `0 < HEAD`, and the closed form reads off after at least one key block
  (`1 ≤ nB`, which also makes the static `Q`/`b1` windows nonempty). -/
```
</details>

**Statement:**
```lean
specification attention_kernel_io_correctness (R : RoundingModel)
    (hfp16 : R.round .fp16 = id)
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_kh stride_b0h BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB : Nat)
    (hBM : 0 < BLOCK_M) (hBN : 0 < BLOCK_N) (hHD : 0 < HEAD) (hnB : 1 ≤ nB) :
    attentionKernelIO Q K V B0 Out sm_scale stride_qh stride_kh stride_b0h
        BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB ⊨[R]
      fun _ _ _ xs ys zs ws j =>
        attnGenScore (akIOscoreT nB BLOCK_M BLOCK_N HEAD sm_scale xs ys ws)
          (akIOvT nB BLOCK_N HEAD zs) (Lane2D.decode j)
```

**Assumptions / layout contracts:**
- `hfp16 : R.round .fp16 = id`
- `hBM : 0 < BLOCK_M`
- `hBN : 0 < BLOCK_N`
- `hHD : 0 < HEAD`
- `hnB : 1 ≤ nB`

**Closed-form spec defs (transitive):** `attentionKernelIO`, `akIOscoreT`, `akIOvT`, `attention_kernel_fwd_kernel_aligned_surface`, `akIOqT`, `akIOkT`, `akIOb0T`, `akIOb1T`

<details><summary><code>attentionKernelIO</code></summary>

```
/-- **Streaming IO signature** of `_fwd_kernel_aligned` on the four-stream
single-output attention fold skin. -/
```
```lean
def attentionKernelIO (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_kh stride_b0h BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB : Nat) :
    StreamMasked3DKernelIO₄ where
  kernel := attention_kernel_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
    stride_qh HEAD 1 stride_kh HEAD 1 stride_kh HEAD 1 stride_qh HEAD 1
    stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
    FloatDType.fp16
  inp1 := Q
  inp2 := K
  inp3 := V
  inp4 := B0
  out := Out
  T := nB
  B1 := BLOCK_M * HEAD
  B2 := HEAD * BLOCK_N
  B3 := BLOCK_N * HEAD
  B4 := BLOCK_M * (BLOCK_N + 1)
  C := BLOCK_M * HEAD
  outDType := .fp16
  read1 := fun p₀ p₁ _ _ j =>
    p₁ * stride_qh + (p₀ * BLOCK_M + j.val / HEAD) * HEAD + j.val % HEAD
  read2 := fun _ p₁ _ t j =>
    p₁ * stride_kh + j.val / BLOCK_N + (t.val * BLOCK_N + j.val % BLOCK_N) * HEAD
  read3 := fun _ p₁ _ t j =>
    p₁ * stride_kh + (t.val * BLOCK_N + j.val / HEAD) * HEAD + j.val % HEAD
  read4 := fun p₀ p₁ _ t j =>
    p₁ * stride_b0h + (p₀ * BLOCK_M + j.val / (BLOCK_N + 1)) * stride_b0m +
      (if j.val % (BLOCK_N + 1) < BLOCK_N then
        j.val % (BLOCK_N + 1) % BIAS_LAST_SIZE + BIAS_LAST_SIZE
      else t.val)
  write := fun p₀ p₁ _ j =>
    p₁ * stride_qh + (p₀ * BLOCK_M + j.val / HEAD) * HEAD + (j.val % HEAD) * 1
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  mask4 := fun _ _ _ _ _ => True
  writeMask := fun _ _ _ _ => True
```
</details>

<details><summary><code>akIOscoreT</code></summary>

```
/-- **Stream-side genuine per-key score** (the tile-parametric twin of
`ClosedForm.fscore`): `(sm_scale·log₂e)·⟨q_r, k_gj⟩ + (b0 + b1)·log₂e`, with
the `qk_scale` factor folded into the dot exactly as the kernel folds it into
the pre-scaled `q`. -/
```
```lean
noncomputable def akIOscoreT (T BM BN HD : Nat) (sm_scale : ℝ)
    (xs : Fin T → Fin (BM * HD) → ℝ) (ys : Fin T → Fin (HD * BN) → ℝ)
    (ws : Fin T → Fin (BM * (BN + 1)) → ℝ)
    (r : Fin BM) (gj : Fin (BN * T)) : ℝ :=
  (sm_scale * 1.44269504) *
      Finset.univ.sum (fun e : Fin HD =>
        akIOqT T BM HD xs (r, e, PUnit.unit) * akIOkT T HD BN ys e gj)
    + (akIOb0T T BM BN ws r (gj.val / BN) + akIOb1T T BM BN ws r (gj.val % BN))
      * 1.44269504
```
</details>

<details><summary><code>akIOvT</code></summary>

```
/-- The flat `V` value tile off the `[BLOCK_N, HEAD]` step tiles. -/
```
```lean
noncomputable def akIOvT (T BN HD : Nat) (zs : Fin T → Fin (BN * HD) → ℝ) :
    TileIndex [BN * T, HD] → ℝ :=
  fun idx =>
    if h : idx.1.val / BN < T ∧ 0 < BN then
      zs ⟨idx.1.val / BN, h.1⟩
        (Lane2D.encode (⟨idx.1.val % BN, Nat.mod_lt _ h.2⟩, idx.2.1, PUnit.unit))
    else 0
```
</details>

<details><summary><code>attention_kernel_fwd_kernel_aligned_surface</code></summary>

```
/-- Faithful DSL port of `attention_kernel.py`'s `_fwd_kernel_aligned`. -/
```
```lean
def attention_kernel_fwd_kernel_aligned_surface
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk
      stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn
      stride_oh stride_om stride_on
      stride_b0h stride_b0m
      _Z _H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (out_dtype : FloatDType) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  q_offset = off_hz * $(stride_qh)
  kv_offset = off_hz * $(stride_kh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + kv_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + kv_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(OUT_DTYPE)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = tl.arange(0, $(BLOCK_M))
  b_offset = off_hz * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :])
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=OUT_DTYPE)
    qk += tl.dot(q, k)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N))
    qk += ((b0 + b1) * 1.44269504)

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += tl.dot((p).to(OUT_DTYPE), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc / l_i[:, None]
  O_block_ptr = tl.make_block_ptr(base=Out + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(OUT_DTYPE))
}
```
</details>

<details><summary><code>akIOqT</code></summary>

```
/-- The `Q` tile off the (static) first stream — the window ignores `t`, so
the step-`0` slice carries the whole tile (`0` off the empty stream when
`nB = 0`; the headline's `1 ≤ nB` excludes this). -/
```
```lean
noncomputable def akIOqT (T BM HD : Nat) (xs : Fin T → Fin (BM * HD) → ℝ) :
    TileIndex [BM, HD] → ℝ :=
  fun idx => if h : 0 < T then xs ⟨0, h⟩ (Lane2D.encode idx) else 0
```
</details>

<details><summary><code>akIOkT</code></summary>

```
/-- The flat per-key `K` function off the transposed `[HEAD, BLOCK_N]` step
tiles: global key `gj` lives in step `gj / BN` at block-local column
`gj % BN`. -/
```
```lean
noncomputable def akIOkT (T HD BN : Nat) (ys : Fin T → Fin (HD * BN) → ℝ) :
    Fin HD → Fin (BN * T) → ℝ :=
  fun e gj =>
    if h : gj.val / BN < T ∧ 0 < BN then
      ys ⟨gj.val / BN, h.1⟩
        (Lane2D.encode (e, ⟨gj.val % BN, Nat.mod_lt _ h.2⟩, PUnit.unit))
    else 0
```
</details>

<details><summary><code>akIOb0T</code></summary>

```
/-- Per-row block-column bias `b0` off the packed fourth stream (column
`BLOCK_N` of the `[BLOCK_M, BLOCK_N+1]` lane pack at step `c`). -/
```
```lean
noncomputable def akIOb0T (T BM BN : Nat) (ws : Fin T → Fin (BM * (BN + 1)) → ℝ)
    (r : Fin BM) (c : Nat) : ℝ :=
  if h : c < T then
    ws ⟨c, h⟩ (Lane2D.encode (r, ⟨BN, Nat.lt_succ_self BN⟩, PUnit.unit))
  else 0
```
</details>

<details><summary><code>akIOb1T</code></summary>

```
/-- Per-row per-lane bias `b1` off the packed fourth stream (the static
columns `< BLOCK_N`, read at step 0). -/
```
```lean
noncomputable def akIOb1T (T BM BN : Nat) (ws : Fin T → Fin (BM * (BN + 1)) → ℝ)
    (r : Fin BM) (jL : Nat) : ℝ :=
  if h : 0 < T ∧ jL < BN then
    ws ⟨0, h.1⟩ (Lane2D.encode (r, ⟨jL, Nat.lt_succ_of_lt h.2⟩, PUnit.unit))
  else 0
```
</details>
