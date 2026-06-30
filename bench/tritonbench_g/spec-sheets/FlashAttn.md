# Spec sheet — `bench/tritonbench_g/flash_attn/FlashAttn.lean`

**Python source:** `bench/tritonbench_g/flash_attn/flash_attn.py`

## Public theorem: `flash_attn_output_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the final FlashAttention output store. -/
```
</details>

**Statement:**
```lean
theorem flash_attn_output_store_slice_compute_correct
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, DIM] =>
        outOffset s stride_q_head stride_o_seqlen stride_o_dim BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O stride_buf_h
        stride_buf_m stride_buf_d stride_q_head stride_o_seqlen stride_o_dim
        BLOCK_M DIM)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] =>
        some (O, outOffset s stride_q_head stride_o_seqlen stride_o_dim
          BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s stride_buf_h stride_buf_m stride_buf_d BLOCK_M idx)))))
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, DIM] =>
        outOffset s stride_q_head stride_o_seqlen stride_o_dim BLOCK_M idx)`

**Closed-form spec defs (transitive):** `outOffset`, `flash_attn_output_store_slice`, `bufferOffset`, `mIndex`, `dIndex`

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

<details><summary><code>flash_attn_output_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `flash_attn.py`'s
`_fwd_kernel`.

The full kernel streams over K/V blocks, computes a numerically stable attention
accumulator, and also writes the log-sum-exp vector `L`. This slice starts after
`out_buffer = out_buffer / denom[:, None]` with a precomputed `OutBuffer` tile
and proves the final unmasked `O_block_ptr` writeback. It preserves the source
base offset, which is derived from `stride_q_head`. The inner `tl.float32`
online-softmax state and K/V dot loop are outside this slice. -/
```
```lean
def flash_attn_output_store_slice
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(DIM))
  out_buffer = tl.load(OutBuffer + off_bs_head * $(stride_buf_h) +
      offs_m[:, None] * $(stride_buf_m) + offs_d[None, :] * $(stride_buf_d))
  tl.store(O + off_bs_head * $(stride_q_head) +
      offs_m[:, None] * $(stride_o_seqlen) + offs_d[None, :] * $(stride_o_dim),
      (out_buffer).to(tl.float16))
}
```
</details>

<details><summary><code>bufferOffset</code></summary>

```lean
def bufferOffset
    (s : BlockState)
    (stride_buf_h stride_buf_m stride_buf_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_buf_h +
    mIndex s BLOCK_M idx.1 * stride_buf_m + dIndex idx * stride_buf_d
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

## Public theorem: `flash_attn_l_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the `L` row store slice. -/
```
</details>

**Statement:**
```lean
theorem flash_attn_l_store_slice_compute_correct
    (Max Denom L : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m
      SEQLEN BLOCK_M : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lOffset s SEQLEN BLOCK_M i)) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_l_store_slice Max Denom L stride_max_h
        stride_max_m stride_den_h stride_den_m SEQLEN BLOCK_M)
      (initialState := s)
      (write := fun i : Fin BLOCK_M => some (L, lOffset s SEQLEN BLOCK_M i))
      (expected := fun i =>
        lStoreSpec s Max Denom stride_max_h stride_max_m stride_den_h
          stride_den_m BLOCK_M i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => lOffset s SEQLEN BLOCK_M i)`

**Closed-form spec defs (transitive):** `lOffset`, `flash_attn_l_store_slice`, `lStoreSpec`, `mIndex`, `maxOffset`, `denomOffset`

<details><summary><code>lOffset</code></summary>

```
/-- Output offset for the FlashAttention `L` row store. -/
```
```lean
def lOffset (s : BlockState) (SEQLEN BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * SEQLEN + mIndex s BLOCK_M i
```
</details>

<details><summary><code>flash_attn_l_store_slice</code></summary>

```
/-- Surface transcription of `flash_attn.py`'s final `L` vector store.

The full kernel computes the streaming row max and denominator, then stores
`max + tl.math.log2(denom)` into `L + off_bs_head * SEQLEN + off_m`. This
surface starts from precomputed `Max` and `Denom` row tiles and preserves that
addressing. -/
```
```lean
def flash_attn_l_store_slice
    (Max Denom L : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m
      SEQLEN BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  max_row = tl.load(Max + off_bs_head * $(stride_max_h) + off_m * $(stride_max_m))
  denom = tl.load(Denom + off_bs_head * $(stride_den_h) + off_m * $(stride_den_m))
  tl.store(L + off_bs_head * $(SEQLEN) + off_m, max_row + tl.log2(denom))
}
```
</details>

<details><summary><code>lStoreSpec</code></summary>

```
/-- Spec for the `L` row store value: `max + log(denom) / log(2)`, mirroring
the kernel's `tl.log2` semantics (`Real.log x / Real.log 2`). -/
```
```lean
noncomputable def lStoreSpec
    (s : BlockState) (Max Denom : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  s.readMem Max (maxOffset s stride_max_h stride_max_m BLOCK_M i) +
    Real.log (s.readMem Denom
      (denomOffset s stride_den_h stride_den_m BLOCK_M i)) / Real.log 2
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>maxOffset</code></summary>

```
/-- Source offset for the precomputed `Max` row read. -/
```
```lean
def maxOffset
    (s : BlockState) (stride_max_h stride_max_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * stride_max_h + mIndex s BLOCK_M i * stride_max_m
```
</details>

<details><summary><code>denomOffset</code></summary>

```
/-- Source offset for the precomputed `Denom` row read. -/
```
```lean
def denomOffset
    (s : BlockState) (stride_den_h stride_den_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * stride_den_h + mIndex s BLOCK_M i * stride_den_m
```
</details>

## Public theorem: `flash_attn_python_output_store_compute_correct`

**Statement:**
```lean
theorem flash_attn_python_output_store_compute_correct
    (OutBuffer O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_output_store_slice OutBuffer O
        8192 64 1 8192 64 1 128 64)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (O, outOffset s 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (s.readMem OutBuffer
              (bufferOffset s 8192 64 1 128 idx)))))
```

**Closed-form spec defs (transitive):** `flash_attn_output_store_slice`, `outOffset`, `bufferOffset`, `mIndex`, `dIndex`

<details><summary><code>flash_attn_output_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `flash_attn.py`'s
`_fwd_kernel`.

The full kernel streams over K/V blocks, computes a numerically stable attention
accumulator, and also writes the log-sum-exp vector `L`. This slice starts after
`out_buffer = out_buffer / denom[:, None]` with a precomputed `OutBuffer` tile
and proves the final unmasked `O_block_ptr` writeback. It preserves the source
base offset, which is derived from `stride_q_head`. The inner `tl.float32`
online-softmax state and K/V dot loop are outside this slice. -/
```
```lean
def flash_attn_output_store_slice
    (OutBuffer O : RegionName)
    (stride_buf_h stride_buf_m stride_buf_d
      stride_q_head stride_o_seqlen stride_o_dim
      BLOCK_M DIM : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(DIM))
  out_buffer = tl.load(OutBuffer + off_bs_head * $(stride_buf_h) +
      offs_m[:, None] * $(stride_buf_m) + offs_d[None, :] * $(stride_buf_d))
  tl.store(O + off_bs_head * $(stride_q_head) +
      offs_m[:, None] * $(stride_o_seqlen) + offs_d[None, :] * $(stride_o_dim),
      (out_buffer).to(tl.float16))
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

<details><summary><code>bufferOffset</code></summary>

```lean
def bufferOffset
    (s : BlockState)
    (stride_buf_h stride_buf_m stride_buf_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, DIM]) : Nat :=
  s.pids 1 * stride_buf_h +
    mIndex s BLOCK_M idx.1 * stride_buf_m + dIndex idx * stride_buf_d
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

## Public theorem: `flash_attn_python_l_store_compute_correct`

**Statement:**
```lean
theorem flash_attn_python_l_store_compute_correct
    (Max Denom L : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_l_store_slice Max Denom L 128 1 128 1 128 128)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lOffset s 128 128 i))
      (expected := fun i =>
        lStoreSpec s Max Denom 128 1 128 1 128 i)
```

**Closed-form spec defs (transitive):** `flash_attn_l_store_slice`, `lOffset`, `lStoreSpec`, `mIndex`, `maxOffset`, `denomOffset`

<details><summary><code>flash_attn_l_store_slice</code></summary>

```
/-- Surface transcription of `flash_attn.py`'s final `L` vector store.

The full kernel computes the streaming row max and denominator, then stores
`max + tl.math.log2(denom)` into `L + off_bs_head * SEQLEN + off_m`. This
surface starts from precomputed `Max` and `Denom` row tiles and preserves that
addressing. -/
```
```lean
def flash_attn_l_store_slice
    (Max Denom L : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m
      SEQLEN BLOCK_M : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_bs_head = tl.program_id(1)
  off_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  max_row = tl.load(Max + off_bs_head * $(stride_max_h) + off_m * $(stride_max_m))
  denom = tl.load(Denom + off_bs_head * $(stride_den_h) + off_m * $(stride_den_m))
  tl.store(L + off_bs_head * $(SEQLEN) + off_m, max_row + tl.log2(denom))
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

<details><summary><code>lStoreSpec</code></summary>

```
/-- Spec for the `L` row store value: `max + log(denom) / log(2)`, mirroring
the kernel's `tl.log2` semantics (`Real.log x / Real.log 2`). -/
```
```lean
noncomputable def lStoreSpec
    (s : BlockState) (Max Denom : RegionName)
    (stride_max_h stride_max_m stride_den_h stride_den_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  s.readMem Max (maxOffset s stride_max_h stride_max_m BLOCK_M i) +
    Real.log (s.readMem Denom
      (denomOffset s stride_den_h stride_den_m BLOCK_M i)) / Real.log 2
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>maxOffset</code></summary>

```
/-- Source offset for the precomputed `Max` row read. -/
```
```lean
def maxOffset
    (s : BlockState) (stride_max_h stride_max_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * stride_max_h + mIndex s BLOCK_M i * stride_max_m
```
</details>

<details><summary><code>denomOffset</code></summary>

```
/-- Source offset for the precomputed `Denom` row read. -/
```
```lean
def denomOffset
    (s : BlockState) (stride_den_h stride_den_m BLOCK_M : Nat)
    (i : Fin BLOCK_M) : Nat :=
  s.pids 1 * stride_den_h + mIndex s BLOCK_M i * stride_den_m
```
</details>

## Public theorem: `flash_attn_genuine_output_compute_correct`

<details><summary>docstring</summary>

```
/-- **Genuine `O`-store correctness** (both causal cases). Every output lane of
the FlashAttention kernel holds the closed-form base-2 attention ratio of the
loaded tiles. -/
```
</details>

**Statement:**
```lean
theorem flash_attn_genuine_output_compute_correct
    (Q K V L O : RegionName) (s : BlockState) (IS_CAUSAL : Bool)
    (hpid0 : s.pids 0 = 0) (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
        16384 8192 64 1 16384 8192 64 1 16384 8192 64 1 16384 8192 64 1
        2 2 128 128 64 64 IS_CAUSAL)
      (initialState := s)
      (write := fun idx : TileIndex [128, 64] =>
        some (O, outOffset s 8192 64 1 128 idx))
      (expected := fun idx : TileIndex [128, 64] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (if IS_CAUSAL then
            flashAttnOValueSpecCausal s Q K V (1.0 : ℝ) 8192 64 128 128 idx
          else
            flashAttnOValueSpec s Q K V (1.0 : ℝ) 8192 64 128 128 idx))))
```

**Assumptions / layout contracts:**
- `hpid0 : s.pids 0 = 0`
- `hOL : O ≠ L`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `flash_attn_fwd_kernel_surface`, `outOffset`, `flashAttnOValueSpecCausal`, `flashAttnOValueSpec`, `mIndex`, `dIndex`, `qTile`, `kTile`, `vTile`, `log2e`, `flashBaseOffset`

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

## Public theorem: `flash_attn_genuine_l_compute_correct`

<details><summary>docstring</summary>

```
/-- **Genuine `L`-store correctness** (both causal cases). Every row lane of the
FlashAttention kernel holds the closed-form log-sum-exp
`log2 (Σ_j pow2 (scoreⱼ))` of the (causally filtered) per-key base-2 scores. -/
```
</details>

**Statement:**
```lean
theorem flash_attn_genuine_l_compute_correct
    (Q K V L O : RegionName) (s : BlockState) (IS_CAUSAL : Bool)
    (hpid0 : s.pids 0 = 0) (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := flash_attn_fwd_kernel_surface Q K V L O (1.0 : ℝ)
        16384 8192 64 1 16384 8192 64 1 16384 8192 64 1 16384 8192 64 1
        2 2 128 128 64 64 IS_CAUSAL)
      (initialState := s)
      (write := fun i : Fin 128 => some (L, lOffset s 128 128 i))
      (expected := fun i : Fin 128 =>
        Real.log
          (((flashKeysUpto (qTile s Q 8192 64 128) (kTile s K 8192 64 128)
              (vTile s V 8192 64 128) (1.0 * log2e) IS_CAUSAL (s.pids 0 * 128) 128 i
              ⟨0, by norm_num⟩).map (fun p => pow2 p.1)).sum) / Real.log 2)
```

**Assumptions / layout contracts:**
- `hpid0 : s.pids 0 = 0`
- `hOL : O ≠ L`
- `hundef : ∀ rg o, s.undef rg o = 0`

**Closed-form spec defs (transitive):** `flash_attn_fwd_kernel_surface`, `lOffset`, `flashKeysUpto`, `qTile`, `kTile`, `vTile`, `log2e`, `mIndex`, `flashKV`, `flashBaseOffset`

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

## Public theorem: `flash_attn_genuine_output_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **Genuine GENERAL `O`-store correctness** (both causal cases). Dimension-parameterized
`flash_attn_genuine_output_compute_correct`. -/
```
</details>

**Statement:**
```lean
theorem flash_attn_genuine_output_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState) (IS_CAUSAL : Bool)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hHi : flashHiG s IS_CAUSAL SEQLEN BLOCK_M = SEQLEN)
    (hpid0 : s.pids 0 = 0) (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
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
- `hpid0 : s.pids 0 = 0`
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
theorem flash_attn_genuine_l_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState) (IS_CAUSAL : Bool)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hHi : flashHiG s IS_CAUSAL SEQLEN BLOCK_M = SEQLEN)
    (hpid0 : s.pids 0 = 0) (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
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
- `hpid0 : s.pids 0 = 0`
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
theorem flash_attn_python_case1_genuine_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hAlign : (s.pids 0 + 1) * BLOCK_M = SEQLEN)
    (hpid0 : s.pids 0 = 0) (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    (ComputeCorrect.Realizes
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.true)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] => some (O, outOffset s stride_q_head DIM 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (flashAttnOValueSpecCausal s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M idx))))) ∧
    (ComputeCorrect.Realizes
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
- `hpid0 : s.pids 0 = 0`
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
theorem flash_attn_python_case2_genuine_compute_correct_general
    (Q K V L O : RegionName) (s : BlockState)
    (sm_scale : ℝ) (stride_q_head SEQLEN BLOCK_M DIM BLOCK_N : Nat)
    (sqbs skbs svbs sobs sosl sod BS HEAD : Nat)
    (hDIM : 0 < DIM) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBMlen : 1 < [BLOCK_M].length.succ)
    (hdvd : BLOCK_N ∣ SEQLEN) (hSEQ : 0 < SEQLEN)
    (hpid0 : s.pids 0 = 0) (hOL : O ≠ L) (hundef : ∀ rg o, s.undef rg o = 0) :
    (ComputeCorrect.Realizes
      (kernel := flash_attn_fwd_kernel_surface Q K V L O sm_scale
        sqbs stride_q_head DIM 1 skbs stride_q_head DIM 1 svbs stride_q_head DIM 1
        sobs stride_q_head DIM 1 BS HEAD SEQLEN BLOCK_M DIM BLOCK_N Bool.false)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK_M, DIM] => some (O, outOffset s stride_q_head DIM 1 BLOCK_M idx))
      (expected := fun idx : TileIndex [BLOCK_M, DIM] =>
        MemCell.of .fp16 (FloatDType.real.cast FloatDType.fp16
          (some (flashAttnOValueSpec s Q K V sm_scale stride_q_head DIM SEQLEN BLOCK_M idx))))) ∧
    (ComputeCorrect.Realizes
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
- `hpid0 : s.pids 0 = 0`
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
