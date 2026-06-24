# Spec sheet — `bench/tritonbench_g/attention_kernel_aligned/AttentionKernelAligned.lean`

**Python source:** `bench/tritonbench_g/attention_kernel_aligned/attention_kernel_aligned.py`

## Public theorem: `attention_kernel_aligned_python_test_shape_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General public summary for `attention_kernel_aligned.py`
(dimension-parameterized, NON-self-referential).**

Records the faithful aligned attention surface and asserts that every observable
`Out` lane holds the **genuine** closed-form base-2 streaming-softmax attention
`alignedClosedForm` (= `attentionRealBase2ScalarScaleBias` of the loaded
`Q`/`K`/`V` tiles under the scalar score scale `sm_scale · log2(e)` and the fused
`rel_h + rel_w` bias `b0 + b1`) — NOT the kernel's own executed readback.
Genuinely general over `BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE numKVBlocks sm_scale`
and the head/bias strides (`P_SEQ = 0`, contiguous Q/K/V/Out layout). The Python
test shape (`sm_scale = 1.0`, `stride_qh = 8192`, `stride_b0h = 8192`,
`stride_b0m = 128`, `BLOCK_M = 32`, `BLOCK_N = HEAD = 64`, `BIAS_LAST_SIZE = 64`,
`nB = 2`) is the special case. -/
```
</details>

**Statement:**
```lean
theorem attention_kernel_aligned_python_test_shape_output_summary_general
    (Q K V B0 Out : RegionName) (s : BlockState) (sm_scale : ℝ)
    (stride_qh stride_b0h BLOCK_M BLOCK_N HEAD BIAS_LAST_SIZE stride_b0m nB : Nat)
    (hKN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hHD : 0 < HEAD) (hnB : 1 ≤ nB)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
      stride_qh HEAD 1 stride_qh HEAD 1 stride_qh HEAD 1 stride_qh HEAD 1
      stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
      FloatDType.fp16).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_qh HEAD 1 stride_qh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
        FloatDType.fp16)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, HEAD] =>
        some (Out, surfaceOutOffset s stride_qh HEAD 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, HEAD] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (alignedClosedForm s Q K V B0 sm_scale stride_qh stride_b0h stride_b0m
            (BLOCK_N * nB) BIAS_LAST_SIZE HEAD BLOCK_M BLOCK_N idx))))
```

**Assumptions / layout contracts:**
- `hKN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`
- `hHD : 0 < HEAD`
- `hnB : 1 ≤ nB`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `kernel : = attention_kernel_aligned_fwd_kernel_aligned_surface Q K V B0 Out sm_scale
        stride_qh HEAD 1 stride_qh HEAD 1 stride_qh HEAD 1 stride_qh HEAD 1
        stride_b0h stride_b0m 2 4 (BLOCK_N * nB) 0 BIAS_LAST_SIZE 128 HEAD BLOCK_M BLOCK_N
        FloatDType.fp16`
- `initialState : = s`
- `write : = fun idx : TileIndex [BLOCK_M, HEAD] =>
        some (Out, surfaceOutOffset s stride_qh HEAD 1 BLOCK_M idx)`

**Closed-form spec defs (transitive):** `attention_kernel_aligned_fwd_kernel_aligned_surface`, `surfaceOutOffset`, `alignedClosedForm`, `mIndex`, `kIndex`, `alignedQTile`, `alignedKTile`, `alignedVTile`, `log2e`, `alignedBias`

<details><summary><code>attention_kernel_aligned_fwd_kernel_aligned_surface</code></summary>

```
/-- Faithful DSL port of `attention_kernel_aligned.py`'s `_fwd_kernel_aligned`. -/
```
```lean
def attention_kernel_aligned_fwd_kernel_aligned_surface
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
    qk += tl.dot(q, k, out_dtype=OUT_DTYPE)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N))
    qk += (b0 + b1)

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

<details><summary><code>alignedClosedForm</code></summary>

```
/-- **Genuine closed-form `Out`-store value** for `attention_kernel_aligned`: the
base-2 attention of the loaded `Q`/`K`/`V` tiles, with the constant scalar score
scale `sm_scale · log2(e)` and the fused `rel_h + rel_w` bias `b0 + b1`. This is
the value the streaming softmax `acc / l_i` computes — defined over the loaded
tiles, NOT the kernel's own executed output (`producedOutputValue`). -/
```
```lean
noncomputable def alignedClosedForm
    (s : BlockState) (Q K V B0 : RegionName) (sm_scale : ℝ)
    (stride_qh stride_b0h stride_b0m
      N_CTX BIAS_LAST_SIZE BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  attentionRealBase2ScalarScaleBias
    (alignedQTile s Q stride_qh BLOCK_DMODEL BLOCK_M)
    (alignedKTile s K stride_qh BLOCK_DMODEL N_CTX)
    (alignedVTile s V stride_qh BLOCK_DMODEL N_CTX)
    (sm_scale * log2e)
    (alignedBias s B0 stride_b0h stride_b0m BIAS_LAST_SIZE BLOCK_M BLOCK_N N_CTX)
    idx
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

<details><summary><code>alignedQTile</code></summary>

```
/-- Loaded `Q` tile: block row `i`, head lane `e` at
`pid₁ · stride_qh + (pid₀ · BLOCK_M + i) · BLOCK_DMODEL + e`. -/
```
```lean
noncomputable def alignedQTile (s : BlockState) (Q : RegionName)
    (stride_qh BLOCK_DMODEL BLOCK_M : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (s.pids 1 * stride_qh + mIndex s BLOCK_M i * BLOCK_DMODEL + e.val)
```
</details>

<details><summary><code>alignedKTile</code></summary>

```
/-- Loaded `K` tile: key `j`, head lane `e` at
`pid₁ · stride_qh + j · BLOCK_DMODEL + e`. -/
```
```lean
noncomputable def alignedKTile (s : BlockState) (K : RegionName)
    (stride_qh BLOCK_DMODEL S : Nat) :
    TileIndex [S, BLOCK_DMODEL] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (s.pids 1 * stride_qh + j.val * BLOCK_DMODEL + e.val)
```
</details>

<details><summary><code>alignedVTile</code></summary>

```
/-- Loaded `V` tile: key `j`, channel `d` at
`pid₁ · stride_qh + j · BLOCK_DMODEL + d`. -/
```
```lean
noncomputable def alignedVTile (s : BlockState) (V : RegionName)
    (stride_qh BLOCK_DMODEL S : Nat) :
    TileIndex [S, BLOCK_DMODEL] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (s.pids 1 * stride_qh + j.val * BLOCK_DMODEL + d.val)
```
</details>

<details><summary><code>log2e</code></summary>

```
/-- The kernel's literal base-2-`e` constant `1.44269504` (the truncated
`log2(e) = 1 / log 2` the `@triton.jit` source folds into `q` via
`qk_scale = sm_scale · 1.44269504`, `q = (q · qk_scale).to(...)`). The genuine
closed form `alignedClosedForm` below is stated with exactly this constant so
that the executed `qk_scale` register and the spec's score scale coincide
literally (no `log2(e)`-rounding gap), matching the kernel faithfully. -/
```
```lean
noncomputable def log2e : ℝ := 1.44269504
```
</details>

<details><summary><code>alignedBias</code></summary>

```
/-- Fused relative-position bias `bias i j = b0 + b1` added to the `(i, j)`
score. Mirrors the kernel: `b0` indexes the block column `j / BLOCK_N`, `b1`
indexes the per-lane column `(j % BLOCK_N) % BIAS_LAST_SIZE + BIAS_LAST_SIZE`,
both at row `pid₀ · BLOCK_M + i` of the `B0` table
(`b_offset = pid₁ · stride_b0h`, row stride `stride_b0m`). -/
```
```lean
noncomputable def alignedBias (s : BlockState) (B0 : RegionName)
    (stride_b0h stride_b0m BIAS_LAST_SIZE BLOCK_M BLOCK_N S : Nat) :
    Fin BLOCK_M → Fin S → ℝ :=
  fun i j =>
    let row := s.pids 1 * stride_b0h + mIndex s BLOCK_M i * stride_b0m
    s.readMem B0 (row + j.val / BLOCK_N) +
      s.readMem B0 (row + (j.val % BLOCK_N) % BIAS_LAST_SIZE + BIAS_LAST_SIZE)
```
</details>

## Also present (pinned special-case summaries)
- `attention_kernel_aligned_final_store_slice_compute_correct`
- `attention_kernel_aligned_fwd_kernel_aligned_surface_compute_correct`
- `aligned_genuine_output_compute_correct`
- `aligned_genuine_output_compute_correct_general`
