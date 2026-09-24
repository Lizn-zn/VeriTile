# Spec sheet — `bench/tritonbench_g/flash_attn/FlashAttn.lean`

**Python source:** `bench/tritonbench_g/flash_attn/flash_attn.py`

## Public theorem: `flash_attn_genuine_output_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **Genuine GENERAL `O`-store correctness** (both causal cases). Dimension-parameterized
`flash_attn_genuine_output_compute_correct`. -/
```
</details>

**Statement:**
```lean
specification flash_attn_genuine_output_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState) (IS_CAUSAL : Bool)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hHi : flashHiG s IS_CAUSAL SEQLEN BLOCK_M = SEQLEN)
    (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N IS_CAUSAL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] =>
        some (O, outOffset s stride_q_head DIM 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (if IS_CAUSAL then
            flashAttnOValueSpecCausal s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M idx
          else
            flashAttnOValueSpec s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M idx))))
```

**Assumptions / layout contracts:**
- `hDIM : 0 < DIM`
- `hBN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`
- `hBMlen : 1 < [BLOCK_M].length.succ`
- `hdvd : BLOCK_N ∣ SEQLEN`
- `hSEQ : 0 < SEQLEN`
- `hHi : flashHiG s IS_CAUSAL SEQLEN BLOCK_M = SEQLEN`
- `hOL : O ≠ L`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `flashHiG`, `flash_attn_fwd_kernel_surface`, `outOffset`, `flashAttnOValueSpecCausal`, `flashAttnOValueSpec`, `mIndex`, `dIndex`, `qTile`, `kTile`, `vTile`, `log2e`, `flashBaseOffset`

<details><summary><code>flashHiG</code></summary>

```
/-- General resolved `hi`. -/
```
```lean
def flashHiG (s : BlockState) (IS_CAUSAL : Bool) (SEQLEN BLOCK_M : Nat) : Nat :=
  if IS_CAUSAL then (s.pids 0 + 1) * BLOCK_M else SEQLEN
```
</details>

<details><summary><code>flash_attn_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `flash_attn.py`'s `_fwd_kernel`. -/
```
```lean
def flash_attn_fwd_kernel_surface
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      _BS _HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)

  qkv_base_offset = off_bs_head * $(stride_q_head)
  Q_block_ptr = tl.make_block_ptr(base=Q + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_q_seqlen), $(stride_q_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + qkv_base_offset,
    shape=($(DIM), $(SEQLEN)),
    strides=($(stride_k_dim), $(stride_k_seqlen)),
    offsets=(0, 0),
    block_shape=($(DIM), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_k_seqlen), $(stride_v_dim)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(DIM)),
    order=(1, 0))
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(BLOCK_N))
  max = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  denom = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  out_buffer = tl.zeros([$(BLOCK_M), $(DIM)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = ((start_m + $(1)) * $(BLOCK_M) if IS_CAUSAL else $(SEQLEN))
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk += tl.dot(q, k)

    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot((nume).to(tl.float16), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $(SEQLEN) + off_m
  tl.store(l_ptr, max + tl.math.log2(denom))
  O_block_ptr = tl.make_block_ptr(base=O + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_o_seqlen), $(stride_o_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  tl.store(O_block_ptr, (out_buffer).to(tl.float16))
}
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (stride_q_head stride_o_seqlen stride_o_dim BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_q_head +
    mIndex s BLOCK_M idx.1 * stride_o_seqlen + dIndex idx * stride_o_dim
```
</details>

<details><summary><code>flashAttnOValueSpecCausal</code></summary>

```
/-- Genuine causal (`IS_CAUSAL = true`, Python case 1) closed-form `O`-store value:
the base-2 attention restricted to keys `j ≤ pid₀·BLOCK_M + i` — the per-element
`tl.where(off_m ≥ start_n + off_n, qk, -inf)` mask zeroes future keys. -/
```
```lean
noncomputable def flashAttnOValueSpecCausal
    (s : BlockState) (Q K V : RegionName)
    (sm_scale : ℝ) (stride_q_head DIM SEQLEN BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : ℝ :=
  attentionRealBase2PerKeyScaleCausal
    (qTile s Q stride_q_head DIM BLOCK_M)
    (kTile s K stride_q_head DIM SEQLEN)
    (vTile s V stride_q_head DIM SEQLEN)
    (fun _ : Fin SEQLEN => sm_scale * log2e)
    (s.pids 0 * BLOCK_M)
    idx
```
</details>

<details><summary><code>flashAttnOValueSpec</code></summary>

```
/-- Genuine non-causal (`IS_CAUSAL = false`, Python case 2) closed-form `O`-store
value: the base-2 attention of the loaded Q/K/V tiles with the constant per-key
scale `qk_scale = sm_scale · log2(e)`. Every key contributes. -/
```
```lean
noncomputable def flashAttnOValueSpec
    (s : BlockState) (Q K V : RegionName)
    (sm_scale : ℝ) (stride_q_head DIM SEQLEN BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : ℝ :=
  attentionRealBase2PerKeyScale
    (qTile s Q stride_q_head DIM BLOCK_M)
    (kTile s K stride_q_head DIM SEQLEN)
    (vTile s V stride_q_head DIM SEQLEN)
    (fun _ : Fin SEQLEN => sm_scale * log2e)
    idx
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>qTile</code></summary>

```
/-- Loaded `Q` tile as a function of memory. Under the Python layout
(`stride_q_seqlen = DIM`, `stride_q_dim = 1`) row `i`, head lane `e` of the block
sits at `base + (pid₀·BLOCK_M + i)·DIM + e`. -/
```
```lean
noncomputable def qTile (s : BlockState) (Q : RegionName)
    (stride_q_head DIM BLOCK_M : Nat) : TileIndex [BLOCK_M, DIM] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (flashBaseOffset s stride_q_head + mIndex s BLOCK_M i * DIM + e.val)
```
</details>

<details><summary><code>kTile</code></summary>

```
/-- Loaded `K` tile (key `j`, head lane `e`) at `base + j·DIM + e`. -/
```
```lean
noncomputable def kTile (s : BlockState) (K : RegionName)
    (stride_q_head DIM SEQLEN : Nat) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (flashBaseOffset s stride_q_head + j.val * DIM + e.val)
```
</details>

<details><summary><code>vTile</code></summary>

```
/-- Loaded `V` tile (key `j`, channel `d`) at `base + j·DIM + d`. -/
```
```lean
noncomputable def vTile (s : BlockState) (V : RegionName)
    (stride_q_head DIM SEQLEN : Nat) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (flashBaseOffset s stride_q_head + j.val * DIM + d.val)
```
</details>

<details><summary><code>log2e</code></summary>

```
/-- The base-2 log-of-`e` constant the kernel folds into `qk_scale`
(`q = (q · sm_scale · 1.44269504).to(fp16)`). This is the *exact decimal literal*
`1.44269504` the Triton source uses (a truncation of the true `log2(e) = 1/log 2 ≈
1.4426950408889634`); the spec's per-key scale `sm_scale · log2e` is therefore the
genuine scale the kernel actually computes, folded into `q`. -/
```
```lean
def log2e : ℝ := 1.44269504
```
</details>

<details><summary><code>flashBaseOffset</code></summary>

```
/-- Per-(batch,head) base offset `off_bs_head · stride_q_head = pid₁ · 8192` for
the Python layout. -/
```
```lean
def flashBaseOffset (s : BlockState) (stride_q_head : Nat) : Nat :=
  s.pids 1 * stride_q_head
```
</details>

## Public theorem: `flash_attn_genuine_l_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **Genuine GENERAL `L`-store correctness** (both causal cases). Dimension-parameterized
`flash_attn_genuine_l_compute_correct`. -/
```
</details>

**Statement:**
```lean
specification flash_attn_genuine_l_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState) (IS_CAUSAL : Bool)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hHi : flashHiG s IS_CAUSAL SEQLEN BLOCK_M = SEQLEN)
    (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N IS_CAUSAL)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lOffset s SEQLEN BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        Real.log
          (((flashKeysUpto (qTile s Q stride_q_head DIM BLOCK_M) (kTile s K stride_q_head DIM SEQLEN)
              (vTile s V stride_q_head DIM SEQLEN) (sm_scale * log2e) IS_CAUSAL (s.pids 0 * BLOCK_M) SEQLEN i
              ⟨0, hDIM⟩).map (fun p => pow2 p.1)).sum) / Real.log 2)
```

**Assumptions / layout contracts:**
- `hDIM : 0 < DIM`
- `hBN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`
- `hBMlen : 1 < [BLOCK_M].length.succ`
- `hdvd : BLOCK_N ∣ SEQLEN`
- `hSEQ : 0 < SEQLEN`
- `hHi : flashHiG s IS_CAUSAL SEQLEN BLOCK_M = SEQLEN`
- `hOL : O ≠ L`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `flashHiG`, `flash_attn_fwd_kernel_surface`, `lOffset`, `flashKeysUpto`, `qTile`, `kTile`, `vTile`, `log2e`, `mIndex`, `flashKV`, `flashBaseOffset`

<details><summary><code>flashHiG</code></summary>

```
/-- General resolved `hi`. -/
```
```lean
def flashHiG (s : BlockState) (IS_CAUSAL : Bool) (SEQLEN BLOCK_M : Nat) : Nat :=
  if IS_CAUSAL then (s.pids 0 + 1) * BLOCK_M else SEQLEN
```
</details>

<details><summary><code>flash_attn_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `flash_attn.py`'s `_fwd_kernel`. -/
```
```lean
def flash_attn_fwd_kernel_surface
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      _BS _HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)

  qkv_base_offset = off_bs_head * $(stride_q_head)
  Q_block_ptr = tl.make_block_ptr(base=Q + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_q_seqlen), $(stride_q_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + qkv_base_offset,
    shape=($(DIM), $(SEQLEN)),
    strides=($(stride_k_dim), $(stride_k_seqlen)),
    offsets=(0, 0),
    block_shape=($(DIM), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_k_seqlen), $(stride_v_dim)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(DIM)),
    order=(1, 0))
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(BLOCK_N))
  max = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  denom = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  out_buffer = tl.zeros([$(BLOCK_M), $(DIM)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = ((start_m + $(1)) * $(BLOCK_M) if IS_CAUSAL else $(SEQLEN))
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk += tl.dot(q, k)

    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot((nume).to(tl.float16), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $(SEQLEN) + off_m
  tl.store(l_ptr, max + tl.math.log2(denom))
  O_block_ptr = tl.make_block_ptr(base=O + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_o_seqlen), $(stride_o_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  tl.store(O_block_ptr, (out_buffer).to(tl.float16))
}
```
</details>

<details><summary><code>lOffset</code></summary>

```
/-- Output offset for the FlashAttention `L` row store. -/
```
```lean
def lOffset (s : BlockState) (SEQLEN BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * SEQLEN + mIndex s BLOCK_M i
```
</details>

<details><summary><code>flashKeysUpto</code></summary>

```
/-- Causal per-row key list over the *window* `[0, hi)`: keys `j < hi` with
`j ≤ qStart + i`, in index order. After `c` blocks `hi = c · BLOCK_N`, this is
the prefix the kernel has streamed. -/
```
```lean
noncomputable def flashKeysUpto
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    List (ℝ × ℝ) :=
  (List.finRange SEQLEN).filterMap (fun j : Fin SEQLEN =>
    if j.val < hi ∧ (causal → j.val ≤ qStart + i.val) then
      some (flashKV qT kT vT scale i d j)
    else none)
```
</details>

<details><summary><code>qTile</code></summary>

```
/-- Loaded `Q` tile as a function of memory. Under the Python layout
(`stride_q_seqlen = DIM`, `stride_q_dim = 1`) row `i`, head lane `e` of the block
sits at `base + (pid₀·BLOCK_M + i)·DIM + e`. -/
```
```lean
noncomputable def qTile (s : BlockState) (Q : RegionName)
    (stride_q_head DIM BLOCK_M : Nat) : TileIndex [BLOCK_M, DIM] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (flashBaseOffset s stride_q_head + mIndex s BLOCK_M i * DIM + e.val)
```
</details>

<details><summary><code>kTile</code></summary>

```
/-- Loaded `K` tile (key `j`, head lane `e`) at `base + j·DIM + e`. -/
```
```lean
noncomputable def kTile (s : BlockState) (K : RegionName)
    (stride_q_head DIM SEQLEN : Nat) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (flashBaseOffset s stride_q_head + j.val * DIM + e.val)
```
</details>

<details><summary><code>vTile</code></summary>

```
/-- Loaded `V` tile (key `j`, channel `d`) at `base + j·DIM + d`. -/
```
```lean
noncomputable def vTile (s : BlockState) (V : RegionName)
    (stride_q_head DIM SEQLEN : Nat) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (flashBaseOffset s stride_q_head + j.val * DIM + d.val)
```
</details>

<details><summary><code>log2e</code></summary>

```
/-- The base-2 log-of-`e` constant the kernel folds into `qk_scale`
(`q = (q · sm_scale · 1.44269504).to(fp16)`). This is the *exact decimal literal*
`1.44269504` the Triton source uses (a truncation of the true `log2(e) = 1/log 2 ≈
1.4426950408889634`); the spec's per-key scale `sm_scale · log2e` is therefore the
genuine scale the kernel actually computes, folded into `q`. -/
```
```lean
def log2e : ℝ := 1.44269504
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>flashKV</code></summary>

```
/-- The `(score, value)` pair the kernel streams for output `(i, d)` at *global*
key `j`: score `scale · Σ_e q[i,e]·k[j,e]`, value `V[j, d]`. (`scale` already
folds `qk_scale = sm_scale · log2e`.) -/
```
```lean
noncomputable def flashKV
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (i : Fin BLOCK_M) (d : Fin DIM) (j : Fin SEQLEN) : ℝ × ℝ :=
  (scale * Finset.univ.sum (fun e : Fin DIM => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
   vT (j, d, PUnit.unit))
```
</details>

<details><summary><code>flashBaseOffset</code></summary>

```
/-- Per-(batch,head) base offset `off_bs_head · stride_q_head = pid₁ · 8192` for
the Python layout. -/
```
```lean
def flashBaseOffset (s : BlockState) (stride_q_head : Nat) : Nat :=
  s.pids 1 * stride_q_head
```
</details>

## Public theorem: `flash_attn_python_case1_genuine_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **Python case 1 (causal) GENERAL genuine closed-form correctness.** -/
```
</details>

**Statement:**
```lean
specification flash_attn_python_case1_genuine_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hAlign : (s.pids 0 + 1) * BLOCK_M = SEQLEN)
    (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] => some (O, outOffset s stride_q_head DIM 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (flashAttnOValueSpecCausal s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M idx))))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.true)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lOffset s SEQLEN BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        Real.log
          (((flashKeysUpto (qTile s Q stride_q_head DIM BLOCK_M) (kTile s K stride_q_head DIM SEQLEN)
              (vTile s V stride_q_head DIM SEQLEN) (sm_scale * log2e) Bool.true (s.pids 0 * BLOCK_M) SEQLEN i
              ⟨0, hDIM⟩).map (fun p => pow2 p.1)).sum) / Real.log 2))
```

**Assumptions / layout contracts:**
- `hDIM : 0 < DIM`
- `hBN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`
- `hBMlen : 1 < [BLOCK_M].length.succ`
- `hdvd : BLOCK_N ∣ SEQLEN`
- `hSEQ : 0 < SEQLEN`
- `hAlign : (s.pids 0 + 1) * BLOCK_M = SEQLEN`
- `hOL : O ≠ L`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `flash_attn_fwd_kernel_surface`, `outOffset`, `flashAttnOValueSpecCausal`, `lOffset`, `flashKeysUpto`, `qTile`, `kTile`, `vTile`, `log2e`, `mIndex`, `dIndex`, `flashKV`, `flashBaseOffset`

<details><summary><code>flash_attn_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `flash_attn.py`'s `_fwd_kernel`. -/
```
```lean
def flash_attn_fwd_kernel_surface
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      _BS _HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)

  qkv_base_offset = off_bs_head * $(stride_q_head)
  Q_block_ptr = tl.make_block_ptr(base=Q + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_q_seqlen), $(stride_q_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + qkv_base_offset,
    shape=($(DIM), $(SEQLEN)),
    strides=($(stride_k_dim), $(stride_k_seqlen)),
    offsets=(0, 0),
    block_shape=($(DIM), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_k_seqlen), $(stride_v_dim)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(DIM)),
    order=(1, 0))
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(BLOCK_N))
  max = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  denom = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  out_buffer = tl.zeros([$(BLOCK_M), $(DIM)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = ((start_m + $(1)) * $(BLOCK_M) if IS_CAUSAL else $(SEQLEN))
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk += tl.dot(q, k)

    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot((nume).to(tl.float16), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $(SEQLEN) + off_m
  tl.store(l_ptr, max + tl.math.log2(denom))
  O_block_ptr = tl.make_block_ptr(base=O + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_o_seqlen), $(stride_o_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  tl.store(O_block_ptr, (out_buffer).to(tl.float16))
}
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (stride_q_head stride_o_seqlen stride_o_dim BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_q_head +
    mIndex s BLOCK_M idx.1 * stride_o_seqlen + dIndex idx * stride_o_dim
```
</details>

<details><summary><code>flashAttnOValueSpecCausal</code></summary>

```
/-- Genuine causal (`IS_CAUSAL = true`, Python case 1) closed-form `O`-store value:
the base-2 attention restricted to keys `j ≤ pid₀·BLOCK_M + i` — the per-element
`tl.where(off_m ≥ start_n + off_n, qk, -inf)` mask zeroes future keys. -/
```
```lean
noncomputable def flashAttnOValueSpecCausal
    (s : BlockState) (Q K V : RegionName)
    (sm_scale : ℝ) (stride_q_head DIM SEQLEN BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : ℝ :=
  attentionRealBase2PerKeyScaleCausal
    (qTile s Q stride_q_head DIM BLOCK_M)
    (kTile s K stride_q_head DIM SEQLEN)
    (vTile s V stride_q_head DIM SEQLEN)
    (fun _ : Fin SEQLEN => sm_scale * log2e)
    (s.pids 0 * BLOCK_M)
    idx
```
</details>

<details><summary><code>lOffset</code></summary>

```
/-- Output offset for the FlashAttention `L` row store. -/
```
```lean
def lOffset (s : BlockState) (SEQLEN BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * SEQLEN + mIndex s BLOCK_M i
```
</details>

<details><summary><code>flashKeysUpto</code></summary>

```
/-- Causal per-row key list over the *window* `[0, hi)`: keys `j < hi` with
`j ≤ qStart + i`, in index order. After `c` blocks `hi = c · BLOCK_N`, this is
the prefix the kernel has streamed. -/
```
```lean
noncomputable def flashKeysUpto
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    List (ℝ × ℝ) :=
  (List.finRange SEQLEN).filterMap (fun j : Fin SEQLEN =>
    if j.val < hi ∧ (causal → j.val ≤ qStart + i.val) then
      some (flashKV qT kT vT scale i d j)
    else none)
```
</details>

<details><summary><code>qTile</code></summary>

```
/-- Loaded `Q` tile as a function of memory. Under the Python layout
(`stride_q_seqlen = DIM`, `stride_q_dim = 1`) row `i`, head lane `e` of the block
sits at `base + (pid₀·BLOCK_M + i)·DIM + e`. -/
```
```lean
noncomputable def qTile (s : BlockState) (Q : RegionName)
    (stride_q_head DIM BLOCK_M : Nat) : TileIndex [BLOCK_M, DIM] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (flashBaseOffset s stride_q_head + mIndex s BLOCK_M i * DIM + e.val)
```
</details>

<details><summary><code>kTile</code></summary>

```
/-- Loaded `K` tile (key `j`, head lane `e`) at `base + j·DIM + e`. -/
```
```lean
noncomputable def kTile (s : BlockState) (K : RegionName)
    (stride_q_head DIM SEQLEN : Nat) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (flashBaseOffset s stride_q_head + j.val * DIM + e.val)
```
</details>

<details><summary><code>vTile</code></summary>

```
/-- Loaded `V` tile (key `j`, channel `d`) at `base + j·DIM + d`. -/
```
```lean
noncomputable def vTile (s : BlockState) (V : RegionName)
    (stride_q_head DIM SEQLEN : Nat) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (flashBaseOffset s stride_q_head + j.val * DIM + d.val)
```
</details>

<details><summary><code>log2e</code></summary>

```
/-- The base-2 log-of-`e` constant the kernel folds into `qk_scale`
(`q = (q · sm_scale · 1.44269504).to(fp16)`). This is the *exact decimal literal*
`1.44269504` the Triton source uses (a truncation of the true `log2(e) = 1/log 2 ≈
1.4426950408889634`); the spec's per-key scale `sm_scale · log2e` is therefore the
genuine scale the kernel actually computes, folded into `q`. -/
```
```lean
def log2e : ℝ := 1.44269504
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>flashKV</code></summary>

```
/-- The `(score, value)` pair the kernel streams for output `(i, d)` at *global*
key `j`: score `scale · Σ_e q[i,e]·k[j,e]`, value `V[j, d]`. (`scale` already
folds `qk_scale = sm_scale · log2e`.) -/
```
```lean
noncomputable def flashKV
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (i : Fin BLOCK_M) (d : Fin DIM) (j : Fin SEQLEN) : ℝ × ℝ :=
  (scale * Finset.univ.sum (fun e : Fin DIM => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
   vT (j, d, PUnit.unit))
```
</details>

<details><summary><code>flashBaseOffset</code></summary>

```
/-- Per-(batch,head) base offset `off_bs_head · stride_q_head = pid₁ · 8192` for
the Python layout. -/
```
```lean
def flashBaseOffset (s : BlockState) (stride_q_head : Nat) : Nat :=
  s.pids 1 * stride_q_head
```
</details>

## Public theorem: `flash_attn_python_case2_genuine_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **Python case 2 (non-causal) GENERAL genuine closed-form correctness.** -/
```
</details>

**Statement:**
```lean
specification flash_attn_python_case2_genuine_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] => some (O, outOffset s stride_q_head DIM 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (flashAttnOValueSpec s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M idx))))) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.false)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lOffset s SEQLEN BLOCK_M i))
      (expected := fun i : Fin BLOCK_M =>
        Real.log
          (((flashKeysUpto (qTile s Q stride_q_head DIM BLOCK_M) (kTile s K stride_q_head DIM SEQLEN)
              (vTile s V stride_q_head DIM SEQLEN) (sm_scale * log2e) Bool.false (s.pids 0 * BLOCK_M) SEQLEN i
              ⟨0, hDIM⟩).map (fun p => pow2 p.1)).sum) / Real.log 2))
```

**Assumptions / layout contracts:**
- `hDIM : 0 < DIM`
- `hBN : 0 < BLOCK_N`
- `hBM : 0 < BLOCK_M`
- `hBMlen : 1 < [BLOCK_M].length.succ`
- `hdvd : BLOCK_N ∣ SEQLEN`
- `hSEQ : 0 < SEQLEN`
- `hOL : O ≠ L`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `flash_attn_fwd_kernel_surface`, `outOffset`, `flashAttnOValueSpec`, `lOffset`, `flashKeysUpto`, `qTile`, `kTile`, `vTile`, `log2e`, `mIndex`, `dIndex`, `flashKV`, `flashBaseOffset`

<details><summary><code>flash_attn_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `flash_attn.py`'s `_fwd_kernel`. -/
```
```lean
def flash_attn_fwd_kernel_surface
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      _BS _HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)

  qkv_base_offset = off_bs_head * $(stride_q_head)
  Q_block_ptr = tl.make_block_ptr(base=Q + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_q_seqlen), $(stride_q_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + qkv_base_offset,
    shape=($(DIM), $(SEQLEN)),
    strides=($(stride_k_dim), $(stride_k_seqlen)),
    offsets=(0, 0),
    block_shape=($(DIM), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_k_seqlen), $(stride_v_dim)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(DIM)),
    order=(1, 0))
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(BLOCK_N))
  max = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  denom = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  out_buffer = tl.zeros([$(BLOCK_M), $(DIM)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = ((start_m + $(1)) * $(BLOCK_M) if IS_CAUSAL else $(SEQLEN))
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk += tl.dot(q, k)

    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot((nume).to(tl.float16), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $(SEQLEN) + off_m
  tl.store(l_ptr, max + tl.math.log2(denom))
  O_block_ptr = tl.make_block_ptr(base=O + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_o_seqlen), $(stride_o_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  tl.store(O_block_ptr, (out_buffer).to(tl.float16))
}
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (stride_q_head stride_o_seqlen stride_o_dim BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_q_head +
    mIndex s BLOCK_M idx.1 * stride_o_seqlen + dIndex idx * stride_o_dim
```
</details>

<details><summary><code>flashAttnOValueSpec</code></summary>

```
/-- Genuine non-causal (`IS_CAUSAL = false`, Python case 2) closed-form `O`-store
value: the base-2 attention of the loaded Q/K/V tiles with the constant per-key
scale `qk_scale = sm_scale · log2(e)`. Every key contributes. -/
```
```lean
noncomputable def flashAttnOValueSpec
    (s : BlockState) (Q K V : RegionName)
    (sm_scale : ℝ) (stride_q_head DIM SEQLEN BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : ℝ :=
  attentionRealBase2PerKeyScale
    (qTile s Q stride_q_head DIM BLOCK_M)
    (kTile s K stride_q_head DIM SEQLEN)
    (vTile s V stride_q_head DIM SEQLEN)
    (fun _ : Fin SEQLEN => sm_scale * log2e)
    idx
```
</details>

<details><summary><code>lOffset</code></summary>

```
/-- Output offset for the FlashAttention `L` row store. -/
```
```lean
def lOffset (s : BlockState) (SEQLEN BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * SEQLEN + mIndex s BLOCK_M i
```
</details>

<details><summary><code>flashKeysUpto</code></summary>

```
/-- Causal per-row key list over the *window* `[0, hi)`: keys `j < hi` with
`j ≤ qStart + i`, in index order. After `c` blocks `hi = c · BLOCK_N`, this is
the prefix the kernel has streamed. -/
```
```lean
noncomputable def flashKeysUpto
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (causal : Bool) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin DIM) :
    List (ℝ × ℝ) :=
  (List.finRange SEQLEN).filterMap (fun j : Fin SEQLEN =>
    if j.val < hi ∧ (causal → j.val ≤ qStart + i.val) then
      some (flashKV qT kT vT scale i d j)
    else none)
```
</details>

<details><summary><code>qTile</code></summary>

```
/-- Loaded `Q` tile as a function of memory. Under the Python layout
(`stride_q_seqlen = DIM`, `stride_q_dim = 1`) row `i`, head lane `e` of the block
sits at `base + (pid₀·BLOCK_M + i)·DIM + e`. -/
```
```lean
noncomputable def qTile (s : BlockState) (Q : RegionName)
    (stride_q_head DIM BLOCK_M : Nat) : TileIndex [BLOCK_M, DIM] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q (flashBaseOffset s stride_q_head + mIndex s BLOCK_M i * DIM + e.val)
```
</details>

<details><summary><code>kTile</code></summary>

```
/-- Loaded `K` tile (key `j`, head lane `e`) at `base + j·DIM + e`. -/
```
```lean
noncomputable def kTile (s : BlockState) (K : RegionName)
    (stride_q_head DIM SEQLEN : Nat) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (flashBaseOffset s stride_q_head + j.val * DIM + e.val)
```
</details>

<details><summary><code>vTile</code></summary>

```
/-- Loaded `V` tile (key `j`, channel `d`) at `base + j·DIM + d`. -/
```
```lean
noncomputable def vTile (s : BlockState) (V : RegionName)
    (stride_q_head DIM SEQLEN : Nat) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (flashBaseOffset s stride_q_head + j.val * DIM + d.val)
```
</details>

<details><summary><code>log2e</code></summary>

```
/-- The base-2 log-of-`e` constant the kernel folds into `qk_scale`
(`q = (q · sm_scale · 1.44269504).to(fp16)`). This is the *exact decimal literal*
`1.44269504` the Triton source uses (a truncation of the true `log2(e) = 1/log 2 ≈
1.4426950408889634`); the spec's per-key scale `sm_scale · log2e` is therefore the
genuine scale the kernel actually computes, folded into `q`. -/
```
```lean
def log2e : ℝ := 1.44269504
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>flashKV</code></summary>

```
/-- The `(score, value)` pair the kernel streams for output `(i, d)` at *global*
key `j`: score `scale · Σ_e q[i,e]·k[j,e]`, value `V[j, d]`. (`scale` already
folds `qk_scale = sm_scale · log2e`.) -/
```
```lean
noncomputable def flashKV
    (qT : TileIndex [BLOCK_M, DIM] → ℝ) (kT vT : TileIndex [SEQLEN, DIM] → ℝ)
    (scale : ℝ) (i : Fin BLOCK_M) (d : Fin DIM) (j : Fin SEQLEN) : ℝ × ℝ :=
  (scale * Finset.univ.sum (fun e : Fin DIM => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
   vT (j, d, PUnit.unit))
```
</details>

<details><summary><code>flashBaseOffset</code></summary>

```
/-- Per-(batch,head) base offset `off_bs_head · stride_q_head = pid₁ · 8192` for
the Python layout. -/
```
```lean
def flashBaseOffset (s : BlockState) (stride_q_head : Nat) : Nat :=
  s.pids 1 * stride_q_head
```
</details>

## Public theorem: `flash_attn_io_correctness`

<details><summary>docstring</summary>

```
/-- **The non-causal `⊨[R]` io headline.** For every rounding model `R` with
`R.round .fp16 = id`, the FlashAttention forward surface at
`IS_CAUSAL = false` implements, on its `StreamMasked3DKernelIO₃ₓ₂` signature,
the **ideal-ℝ online-softmax attention fold** over the three streamed tiles:
output lane `j = (i, d)` of `O` holds the base-2 attention ratio
(`flashIOOutSpec` = the existing `flashAttnOValueSpec` closed form restated on
the streams), and row `i` of `L` holds the genuine log-sum-exp
`log₂ (Σⱼ 2^scoreⱼ)` (`flashIOLSpec`). `O` is quantized on the `.fp16` grid
(`out1DType := .fp16`) and `L` is the unrounded `.real` statistics row.

**Hypothesis provenance** (all inherited from the exact non-causal headline
`flash_attn_python_case2_genuine_compute_correct_general`):

* `hfp16 : R.round .fp16 = id` — the file's declared fp16 modeling boundary
  (three in-body `Op.castFloat` sites plus the `.fp16` terminal store); the
  exact stack already treats these casts as the identity.
* `hDIM`/`hBN`/`hSEQ` (`0 < DIM`, `0 < BLOCK_N`, `0 < SEQLEN`) and
  `hdvd : BLOCK_N ∣ SEQLEN` shape the KV walk — `T = SEQLEN / BLOCK_N` full
  blocks, exactly the exact stack's side conditions.
* `hOL : O ≠ L` keeps the `O` tile store from clobbering the `L` row.

The exact headline's `hundef` is **not** a hypothesis here — the skin's Hoare
triple carries the `undef` pin itself — and `hpid₀ = 0` is gone (the Phase-1
generalization), so this holds at every program id.

**Scope disclosed**: the **causal** (`IS_CAUSAL = true`) io face is
deliberately deferred (see the section header); it stays on its exact
headline `flash_attn_python_case1_genuine_compute_correct_general`. -/
```
</details>

**Statement:**
```lean
specification flash_attn_io_correctness (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (sqbs skbs svbs sobs BS HEAD SEQLEN BLOCK_M DIM BLOCK_N stride_q_head : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hSEQ : 0 < SEQLEN) (hdvd : BLOCK_N ∣ SEQLEN)
    (hOL : O ≠ L) :
    flashAttnIO Q K V L O sm_scale sqbs skbs svbs sobs BS HEAD SEQLEN BLOCK_M DIM BLOCK_N
        stride_q_head ⊨[R]
      fun _ _ _ xs ys zs =>
        (fun j => flashIOOutSpec BLOCK_M DIM BLOCK_N SEQLEN (SEQLEN / BLOCK_N)
            (Nat.div_pos (Nat.le_of_dvd hSEQ hdvd) hBN) (Nat.div_mul_cancel hdvd) hBN
            sm_scale xs ys zs j,
         fun i => flashIOLSpec BLOCK_M DIM BLOCK_N SEQLEN (SEQLEN / BLOCK_N) hDIM
            (Nat.div_pos (Nat.le_of_dvd hSEQ hdvd) hBN) (Nat.div_mul_cancel hdvd) hBN
            sm_scale xs ys zs i)
```

**Assumptions / layout contracts:**
- `hfp16 : R.round .fp16 = id`
- `hDIM : 0 < DIM`
- `hBN : 0 < BLOCK_N`
- `hSEQ : 0 < SEQLEN`
- `hdvd : BLOCK_N ∣ SEQLEN`
- `hOL : O ≠ L`

**Closed-form spec defs (transitive):** `flashAttnIO`, `flashIOOutSpec`, `flashIOLSpec`, `flash_attn_fwd_kernel_surface`, `flashIOqT`, `flashIOkT`, `flashIOvT`, `log2e`

<details><summary><code>flashAttnIO</code></summary>

```
/-- **Streaming IO signature** of the non-causal (`IS_CAUSAL = false`,
Python case 2) FlashAttention forward surface on the three-stream
two-output attention fold skin. -/
```
```lean
def flashAttnIO (Q K V L O : RegionName) (sm_scale : ℝ)
    (sqbs skbs svbs sobs BS HEAD SEQLEN BLOCK_M DIM BLOCK_N stride_q_head : Nat) :
    StreamMasked3DKernelIO₃ₓ₂ where
  kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
    sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
    sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.false
  inp1 := Q
  inp2 := K
  inp3 := V
  out1 := O
  out2 := L
  T := SEQLEN / BLOCK_N
  B1 := BLOCK_M * DIM
  B2 := DIM * BLOCK_N
  B3 := BLOCK_N * DIM
  C1 := BLOCK_M * DIM
  C2 := BLOCK_M
  out1DType := .fp16
  out2DType := .real
  read1 := fun p₀ p₁ _ _ j =>
    p₁ * stride_q_head + (p₀ * BLOCK_M + j.val / DIM) * DIM + (j.val % DIM) * 1
  read2 := fun _ p₁ _ t j =>
    p₁ * stride_q_head + (j.val / BLOCK_N) * 1 + (t.val * BLOCK_N + j.val % BLOCK_N) * DIM
  read3 := fun _ p₁ _ t j =>
    p₁ * stride_q_head + (t.val * BLOCK_N + j.val / DIM) * DIM + (j.val % DIM) * 1
  write1 := fun p₀ p₁ _ j =>
    p₁ * stride_q_head + (p₀ * BLOCK_M + j.val / DIM) * DIM + (j.val % DIM) * 1
  write2 := fun p₀ p₁ _ i => p₁ * SEQLEN + (p₀ * BLOCK_M + i.val)
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => True
```
</details>

<details><summary><code>flashIOOutSpec</code></summary>

```
/-- **`O` closed form on the streams**: the non-causal base-2 attention
(`attentionRealBase2PerKeyScale`, exactly the existing `flashAttnOValueSpec`)
restated over the three streamed tiles, at output lane `j = (i, d)` row-major
over `[BLOCK_M, DIM]`. -/
```
```lean
noncomputable def flashIOOutSpec (BLOCK_M DIM BLOCK_N SEQLEN T : Nat)
    (hT : 0 < T) (hTB : T * BLOCK_N = SEQLEN) (hBN : 0 < BLOCK_N) (sm_scale : ℝ)
    (xs : Fin T → Fin (BLOCK_M * DIM) → ℝ) (ys : Fin T → Fin (DIM * BLOCK_N) → ℝ)
    (zs : Fin T → Fin (BLOCK_N * DIM) → ℝ) (j : Fin (BLOCK_M * DIM)) : ℝ :=
  attentionRealBase2PerKeyScale (flashIOqT BLOCK_M DIM T hT xs)
    (flashIOkT DIM BLOCK_N SEQLEN T hTB hBN ys) (flashIOvT DIM BLOCK_N SEQLEN T hTB hBN zs)
    (fun _ : Fin SEQLEN => sm_scale * log2e) (Lane2D.decode j)
```
</details>

<details><summary><code>flashIOLSpec</code></summary>

```
/-- **`L` closed form on the streams**: the genuine log-sum-exp
`log₂ (Σⱼ 2^scoreⱼ)` over the full key list, exactly the existing exact
headline's `L` expectation restated over the three streamed tiles. -/
```
```lean
noncomputable def flashIOLSpec (BLOCK_M DIM BLOCK_N SEQLEN T : Nat) (hDIM : 0 < DIM)
    (hT : 0 < T) (hTB : T * BLOCK_N = SEQLEN) (hBN : 0 < BLOCK_N) (sm_scale : ℝ)
    (xs : Fin T → Fin (BLOCK_M * DIM) → ℝ) (ys : Fin T → Fin (DIM * BLOCK_N) → ℝ)
    (zs : Fin T → Fin (BLOCK_N * DIM) → ℝ) (i : Fin BLOCK_M) : ℝ :=
  Real.log (((attnKeyList (flashIOqT BLOCK_M DIM T hT xs)
      (flashIOkT DIM BLOCK_N SEQLEN T hTB hBN ys) (flashIOvT DIM BLOCK_N SEQLEN T hTB hBN zs)
      (fun _ : Fin SEQLEN => sm_scale * log2e) i ⟨0, hDIM⟩).map
    (fun p => pow2 p.1)).sum) / Real.log 2
```
</details>

<details><summary><code>flash_attn_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `flash_attn.py`'s `_fwd_kernel`. -/
```
```lean
def flash_attn_fwd_kernel_surface
    (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      _BS _HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)

  qkv_base_offset = off_bs_head * $(stride_q_head)
  Q_block_ptr = tl.make_block_ptr(base=Q + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_q_seqlen), $(stride_q_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + qkv_base_offset,
    shape=($(DIM), $(SEQLEN)),
    strides=($(stride_k_dim), $(stride_k_seqlen)),
    offsets=(0, 0),
    block_shape=($(DIM), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_k_seqlen), $(stride_v_dim)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(DIM)),
    order=(1, 0))
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_n = tl.arange(0, $(BLOCK_N))
  max = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  denom = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  out_buffer = tl.zeros([$(BLOCK_M), $(DIM)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr)
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = ((start_m + $(1)) * $(BLOCK_M) if IS_CAUSAL else $(SEQLEN))
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    if IS_CAUSAL {
      qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
    }
    qk += tl.dot(q, k)

    max_new = tl.maximum(max, tl.max(qk, 1))
    alpha = tl.math.exp2(max - max_new)
    nume = tl.math.exp2(qk - max_new[:, None])
    out_scale = denom * 0 + alpha
    out_buffer *= out_scale[:, None]
    out_buffer += tl.dot((nume).to(tl.float16), v)
    denom = denom * alpha + tl.sum(nume, 1)
    max = max_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  out_buffer = out_buffer / denom[:, None]
  l_ptr = L + off_bs_head * $(SEQLEN) + off_m
  tl.store(l_ptr, max + tl.math.log2(denom))
  O_block_ptr = tl.make_block_ptr(base=O + qkv_base_offset,
    shape=($(SEQLEN), $(DIM)),
    strides=($(stride_o_seqlen), $(stride_o_dim)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(DIM)),
    order=(1, 0))
  tl.store(O_block_ptr, (out_buffer).to(tl.float16))
}
```
</details>

<details><summary><code>flashIOqT</code></summary>

```
/-- The `Q` tile read off the (static) first stream: the window ignores `t`,
so the step-`0` slice carries the whole tile. -/
```
```lean
noncomputable def flashIOqT (BLOCK_M DIM T : Nat) (hT : 0 < T)
    (xs : Fin T → Fin (BLOCK_M * DIM) → ℝ) : TileIndex [BLOCK_M, DIM] → ℝ :=
  fun idx => xs ⟨0, hT⟩ (Lane2D.encode idx)
```
</details>

<details><summary><code>flashIOkT</code></summary>

```
/-- The global `K` tile read off the second stream: global key `j` lives in
step `j / BLOCK_N`, block-local column `j % BLOCK_N`, at stream lane
`(e, j % BLOCK_N)` of the transposed `[DIM, BLOCK_N]` tile. -/
```
```lean
noncomputable def flashIOkT (DIM BLOCK_N SEQLEN T : Nat)
    (hTB : T * BLOCK_N = SEQLEN) (hBN : 0 < BLOCK_N)
    (ys : Fin T → Fin (DIM * BLOCK_N) → ℝ) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun idx =>
    ys ⟨idx.1.val / BLOCK_N, (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [hTB]; exact idx.1.isLt)⟩
      (Lane2D.encode (idx.2.1, ⟨idx.1.val % BLOCK_N, Nat.mod_lt _ hBN⟩, PUnit.unit))
```
</details>

<details><summary><code>flashIOvT</code></summary>

```
/-- The global `V` tile read off the third stream: global key `j` lives in
step `j / BLOCK_N`, block-local row `j % BLOCK_N`, at stream lane
`(j % BLOCK_N, d)` of the `[BLOCK_N, DIM]` tile. -/
```
```lean
noncomputable def flashIOvT (DIM BLOCK_N SEQLEN T : Nat)
    (hTB : T * BLOCK_N = SEQLEN) (hBN : 0 < BLOCK_N)
    (zs : Fin T → Fin (BLOCK_N * DIM) → ℝ) : TileIndex [SEQLEN, DIM] → ℝ :=
  fun idx =>
    zs ⟨idx.1.val / BLOCK_N, (Nat.div_lt_iff_lt_mul hBN).mpr (by rw [hTB]; exact idx.1.isLt)⟩
      (Lane2D.encode (⟨idx.1.val % BLOCK_N, Nat.mod_lt _ hBN⟩, idx.2.1, PUnit.unit))
```
</details>

<details><summary><code>log2e</code></summary>

```
/-- The base-2 log-of-`e` constant the kernel folds into `qk_scale`
(`q = (q · sm_scale · 1.44269504).to(fp16)`). This is the *exact decimal literal*
`1.44269504` the Triton source uses (a truncation of the true `log2(e) = 1/log 2 ≈
1.4426950408889634`); the spec's per-key scale `sm_scale · log2e` is therefore the
genuine scale the kernel actually computes, folded into `q`. -/
```
```lean
def log2e : ℝ := 1.44269504
```
</details>

## Also present (pinned special-case summaries)
- `flash_attn_output_store_slice_compute_correct`
- `flash_attn_l_store_slice_compute_correct`
- `flash_attn_genuine_output_compute_correct`
- `flash_attn_genuine_l_compute_correct`
