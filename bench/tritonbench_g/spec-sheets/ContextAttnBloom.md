# Spec sheet — `bench/tritonbench_g/context_attn_bloom/ContextAttnBloom.lean`

**Python source:** `bench/tritonbench_g/context_attn_bloom/context_attn_bloom.py`

## Public theorem: `context_attn_bloom_final_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the masked BLOOM context-attention output
store. -/
```
</details>

**Statement:**
```lean
theorem context_attn_bloom_final_store_slice_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out head_dim stride_acc_b stride_acc_h stride_acc_m
        stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len head_dim stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`
- `kernel : = context_attn_bloom_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out head_dim stride_acc_b stride_acc_h stride_acc_m
        stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL`
- `initialState : = s`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`
- `expected : = fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len head_dim stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx`

**Closed-form spec defs (transitive):** `outOffset`, `context_attn_bloom_final_store_slice`, `active`, `accStoreValue`, `startLoc`, `mIndex`, `dIndex`, `seqLen`, `accOffset`, `promptLen`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    s.pids 1 * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>context_attn_bloom_final_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `context_attn_bloom.py`'s
`_fwd_kernel`.

The full kernel computes BLOOM-style context attention with `Req_to_tokens` and
head-dimension padding. This slice starts from a precomputed `Acc` tile and
proves the final masked writeback into `Out`, preserving both source masks:
`offs_m < cur_batch_seq_len` and `offs_d < head_dim`. The inner `tl.float32`
`m_i/l_i/acc` streaming-softmax loop and request-token gathers are outside this
slice. -/
```
```lean
def context_attn_bloom_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen B_Prompt_Cache_Len : Region .nat)
    (Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)
  prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim))
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) + cur_head * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
      cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od), acc, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen B_Prompt_Cache_Len ∧
    dIndex idx < head_dim
```
</details>

<details><summary><code>accStoreValue</code></summary>

```lean
noncomputable def accStoreValue
    (s : BlockState) (Acc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0) - promptLen s B_Prompt_Cache_Len
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0)
```
</details>

## Public theorem: `context_attn_bloom_final_store_python_block128_compute_correct`

<details><summary>docstring</summary>

```
/-! ## Python test-shape wrappers

The checked Python test uses `Z = 1`, `H = 6`, `N_CTX = 500`, `D_HEAD = 96`.
The contiguous `q/k/v/o` layout has row/head/dimension strides `(576, 96, 1)`;
`BLOCK_DMODEL = next_power_of_2(96) = 128`, and `BLOCK_M` is `128` on the
regular path or `64` on the Tesla branch. -/
```
</details>

**Statement:**
```lean
theorem context_attn_bloom_final_store_python_block128_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 96 288000 96 576 1 576 96 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 96 288000 96 576 1
          128 idx)
```

**Assumptions / layout contracts:**
- `kernel : = context_attn_bloom_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 96 288000 96 576 1 576 96 1 128 128`
- `initialState : = s`
- `fun idx : TileIndex [128, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 128 idx`
- `fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 128 idx)`
- `expected : = fun idx : TileIndex [128, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 96 288000 96 576 1
          128 idx`

**Closed-form spec defs (transitive):** `context_attn_bloom_final_store_slice`, `active`, `outOffset`, `accStoreValue`, `mIndex`, `seqLen`, `dIndex`, `startLoc`, `accOffset`, `promptLen`

<details><summary><code>context_attn_bloom_final_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `context_attn_bloom.py`'s
`_fwd_kernel`.

The full kernel computes BLOOM-style context attention with `Req_to_tokens` and
head-dimension padding. This slice starts from a precomputed `Acc` tile and
proves the final masked writeback into `Out`, preserving both source masks:
`offs_m < cur_batch_seq_len` and `offs_d < head_dim`. The inner `tl.float32`
`m_i/l_i/acc` streaming-softmax loop and request-token gathers are outside this
slice. -/
```
```lean
def context_attn_bloom_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen B_Prompt_Cache_Len : Region .nat)
    (Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)
  prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim))
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) + cur_head * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
      cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od), acc, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen B_Prompt_Cache_Len ∧
    dIndex idx < head_dim
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    s.pids 1 * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>accStoreValue</code></summary>

```lean
noncomputable def accStoreValue
    (s : BlockState) (Acc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0) - promptLen s B_Prompt_Cache_Len
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0)
```
</details>

## Public theorem: `context_attn_bloom_final_store_python_block64_compute_correct`

**Statement:**
```lean
theorem context_attn_bloom_final_store_python_block64_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 96 288000 96 576 1 576 96 1 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 96 288000 96 576 1
          64 idx)
```

**Assumptions / layout contracts:**
- `kernel : = context_attn_bloom_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 96 288000 96 576 1 576 96 1 64 128`
- `initialState : = s`
- `fun idx : TileIndex [64, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 64 idx`
- `fun idx : TileIndex [64, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 64 idx)`
- `expected : = fun idx : TileIndex [64, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 96 288000 96 576 1
          64 idx`

**Closed-form spec defs (transitive):** `context_attn_bloom_final_store_slice`, `active`, `outOffset`, `accStoreValue`, `mIndex`, `seqLen`, `dIndex`, `startLoc`, `accOffset`, `promptLen`

<details><summary><code>context_attn_bloom_final_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `context_attn_bloom.py`'s
`_fwd_kernel`.

The full kernel computes BLOOM-style context attention with `Req_to_tokens` and
head-dimension padding. This slice starts from a precomputed `Acc` tile and
proves the final masked writeback into `Out`, preserving both source masks:
`offs_m < cur_batch_seq_len` and `offs_d < head_dim`. The inner `tl.float32`
`m_i/l_i/acc` streaming-softmax loop and request-token gathers are outside this
slice. -/
```
```lean
def context_attn_bloom_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen B_Prompt_Cache_Len : Region .nat)
    (Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)
  prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim))
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) + cur_head * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
      cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od), acc, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen B_Prompt_Cache_Len ∧
    dIndex idx < head_dim
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    s.pids 1 * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>accStoreValue</code></summary>

```lean
noncomputable def accStoreValue
    (s : BlockState) (Acc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0) - promptLen s B_Prompt_Cache_Len
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0)
```
</details>

## Public theorem: `context_attn_bloom_surface_python_block128_compute_correct`

**Statement:**
```lean
theorem context_attn_bloom_surface_python_block128_compute_correct
    (Q K V B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx
      B_Prompt_Cache_Len : RegionName) (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_fwd_kernel_surface Q K V
        ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
        B_req_idx B_Prompt_Cache_Len
        576 96 1 576 96 1 576 96 1 576 96 1
        7500 1 1 96 128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        bloomFwdGenuineOutValue128 s Q K V B_Start_Loc B_Seqlen Out
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `initialState : = s`
- `fun idx : TileIndex [128, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 128 idx`
- `fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 128 idx)`
- `expected : = fun idx : TileIndex [128, 128] =>
        bloomFwdGenuineOutValue128 s Q K V B_Start_Loc B_Seqlen Out
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx`

**Closed-form spec defs (transitive):** `context_attn_bloom_fwd_kernel_surface`, `active`, `outOffset`, `bloomFwdGenuineOutValue128`, `mIndex`, `seqLen`, `dIndex`, `startLoc`, `contextAttnBloomExactFoldM`, `sm_scale_bloom`, `bloomFwdWindow`, `bloomFwdBel`, `promptLen`, `gAccN`, `bloomKVM`, `gStateBot`, `bloomQTileM`, `bloomKTileM`, `bloomVTileM`, `gKeysUpto`, `osStepBot`, `ctxQTile`, `ctxKTile`, `ctxVTile`, `curHead`, `kvLoc`, `reqIdx`

<details><summary><code>context_attn_bloom_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_bloom.py`'s `_fwd_kernel`. -/
```
```lean
def context_attn_bloom_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=((start_n + offs_n[None, :]) < block_end_loc) &
        (offs_d[:, None] < $(head_dim)),
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :],
      qk, -100000000.0)

    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
    acc = acc * acc_scale[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=((start_n + offs_n[:, None]) < block_end_loc) &
        (offs_d[None, :] < $(head_dim)),
      other=0.0)
    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen B_Prompt_Cache_Len ∧
    dIndex idx < head_dim
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    s.pids 1 * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>bloomFwdGenuineOutValue128</code></summary>

```
/-- **Genuine closed-form `Out` value (BLOCK_M = 128)**: the block-causal-guarded
boundary-masked online-softmax fold `contextAttnBloomExactFoldM` of the loaded
Q/K/V memory — a pure function of memory, not the kernel's executed readback. -/
```
```lean
noncomputable def bloomFwdGenuineOutValue128
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  contextAttnBloomExactFoldM s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx
    sm_scale_bloom 7500 1 128 (bloomFwdWindow s B_Seqlen B_Prompt_Cache_Len 128)
    (bloomFwdBel s B_Seqlen B_Prompt_Cache_Len 128) idx
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0) - promptLen s B_Prompt_Cache_Len
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>contextAttnBloomExactFoldM</code></summary>

```
/-- **The faithful kernel value** at output lane `(i,d)`: the block-causal-guarded
normalized accumulator `gAccN` of the ⊥-seeded online-softmax fold over `bloomKVM`
for the full streamed window `[0, S)` (`S / BLOCK_N` blocks of `BLOCK_N = 128`),
with this row's causal limit `qpos = gᵢ + plen`. Guard-fail blocks (`c·128 >
qpos`) contribute via `acc_scale = 1`, exactly matching the kernel's `tl.where`. -/
```
```lean
noncomputable def contextAttnBloomExactFoldM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  gAccN S 128 (s.pids 2 * BLOCK_M + idx.1.val + promptLen s B_Prompt_Cache_Len)
    (bloomKVM s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s BLOCK_M S bel idx.1 idx.2.1)
    (S / 128)
```
</details>

<details><summary><code>sm_scale_bloom</code></summary>

```
/-- The kernel's chosen natural-`exp` `sm_scale` constant at the Python test shape
(`(√96)⁻¹`, fed `/ log 2` into the base-2 fold so `pow2 (score/log2) = exp score`). -/
```
```lean
noncomputable def sm_scale_bloom : ℝ := (Real.sqrt (96 : ℝ))⁻¹
```
</details>

<details><summary><code>bloomFwdWindow</code></summary>

```
/-- The streamed window `S = 128·⌈(block_mask·block_end_loc)/128⌉` (loop step is
`BLOCK_N = 128`; `block_end_loc` uses the query block size `BLOCK_M`). -/
```
```lean
def bloomFwdWindow (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BLOCK_M : Nat) : Nat :=
  let plen := promptLen s B_Prompt_Cache_Len
  let sl := seqLen s B_Seqlen B_Prompt_Cache_Len
  let bel := (let a := (s.pids 2 + 1) * BLOCK_M + plen
              let b := sl + plen
              if a < b then a else b)
  let bm := if BLOCK_M * s.pids 2 < sl then 1 else 0
  128 * ((bm * bel + 127) / 128)
```
</details>

<details><summary><code>bloomFwdBel</code></summary>

```
/-- `block_end_loc = min((start_m+1)·BLOCK_M + plen, cur_batch_seq_len + plen)`. -/
```
```lean
def bloomFwdBel (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BLOCK_M : Nat) : Nat :=
  let plen := promptLen s B_Prompt_Cache_Len
  let sl := seqLen s B_Seqlen B_Prompt_Cache_Len
  let a := (s.pids 2 + 1) * BLOCK_M + plen
  let b := sl + plen
  if a < b then a else b
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0)
```
</details>

<details><summary><code>gAccN</code></summary>

```
/-- **Faithful normalized accumulator** after `c` blocks for a row with causal
limit `qpos`. Mirrors the kernel's `acc` register: `acc_new = acc·acc_scale +
dot(p,v)`, with `acc_scale = (lᵢ/lᵢⁿᵉʷ)·α` on guard-pass blocks (`c·BN ≤ qpos`)
and `1` on guard-fail blocks, and `dot(p,v) = (numerⁿᵉʷ − numer·α)/lᵢⁿᵉʷ`. -/
```
```lean
noncomputable def gAccN (S BN qpos : Nat) (g : Fin S → ℝ × ℝ) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
    let st := gStateBot S (c * BN) g
    let stn := gStateBot S ((c + 1) * BN) g
    let α := (WithBot.realExp2 (WithBot.realSub st.1 stn.1)).unbotD 0
    let accScale := if c * BN ≤ qpos then (st.2.1 / stn.2.1) * α else 1
    gAccN S BN qpos g c * accScale + (stn.2.2 - st.2.2 * α) / stn.2.1
```
</details>

<details><summary><code>bloomKVM</code></summary>

```
/-- **Faithful per-key `(base-2 score, value)`** the loop folds, with the genuine
`-1e8` sentinel and the `block_end_loc` load mask on `k`/`v`. Score is fed in
base-2 (`/ log 2`) so `pow2 score = exp (kernel natural score)`. -/
```
```lean
noncomputable def bloomKVM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128)
    (j : Fin S) : ℝ × ℝ :=
  ((if j.val ≤ s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin 128 =>
        bloomQTileM s Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len BLOCK_M i e
          * bloomKTileM s K Req_to_tokens B_req_idx stride_req_b stride_req_s S bel (j, e, PUnit.unit))
    else (0.0 - 10e7 : ℝ)) / Real.log 2,
    bloomVTileM s V Req_to_tokens B_req_idx stride_req_b stride_req_s S bel (j, d, PUnit.unit))
```
</details>

<details><summary><code>gStateBot</code></summary>

```
/-- Generic ⊥-seeded running `(max, l, acc)` after streaming `[0, hi)`. -/
```
```lean
noncomputable def gStateBot (S hi : Nat) (g : Fin S → ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  (gKeysUpto S hi g).foldl osStepBot (⊥, 0, 0)
```
</details>

<details><summary><code>bloomQTileM</code></summary>

```
/-- Row- and channel-masked query tile: genuine `ctxQTile` on active rows and
channel `e < 96`, else `0` (channel mask `offs_d < head_dim = 96` matches the
kernel). -/
```
```lean
noncomputable def bloomQTileM
    (s : BlockState) (Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Fin 128) : ℝ :=
  if (s.pids 2 * BLOCK_M + i.val < seqLen s B_Seqlen B_Prompt_Cache_Len) ∧ (e.val < 96) then
    ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
  else 0
```
</details>

<details><summary><code>bloomKTileM</code></summary>

```
/-- `block_end_loc`/channel-masked key tile (genuine `ctxKTile` for `j < bel` and
channel `e < 96`, else `0`). The channel mask `e < head_dim = 96` matches the
kernel's `tl.load` mask `offs_d < head_dim`, which zeros padding channels
`e ∈ [96, 128)`. -/
```
```lean
noncomputable def bloomKTileM
    (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S bel : Nat) : TileIndex [S, 128] → ℝ :=
  fun (j, e, u) =>
    if (j.val < bel) ∧ (e.val < 96) then ctxKTile s K Req_to_tokens B_req_idx stride_req_b stride_req_s S (j, e, u)
    else 0
```
</details>

<details><summary><code>bloomVTileM</code></summary>

```
/-- `block_end_loc`/channel-masked value tile (channel mask `d < 96` matches the
kernel's `offs_d < head_dim` load mask). -/
```
```lean
noncomputable def bloomVTileM
    (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S bel : Nat) : TileIndex [S, 128] → ℝ :=
  fun (j, d, u) =>
    if (j.val < bel) ∧ (d.val < 96) then ctxVTile s V Req_to_tokens B_req_idx stride_req_b stride_req_s S (j, d, u)
    else 0
```
</details>

<details><summary><code>gKeysUpto</code></summary>

```
/-- Generic windowed key list `[0, hi)` over an abstract per-key `g`. -/
```
```lean
noncomputable def gKeysUpto (S hi : Nat) (g : Fin S → ℝ × ℝ) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S => if j.val < hi then some (g j) else none)
```
</details>

<details><summary><code>osStepBot</code></summary>

```
/-- One ⊥-seeded online-softmax step: running max in `WithBot ℝ` (seeded `⊥`), so
`α = realExp2(m ⊖ m')` is `0` on the first block. -/
```
```lean
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (sc - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)
```
</details>

<details><summary><code>ctxQTile</code></summary>

```
/-- Coordinate-faithful query tile of this kernel at `(cur_batch, cur_head,
start_m)` for the checked Python layout (`stride_qbs=576, stride_qh=96,
stride_qd=1`). Row `i` is the global prefill row `start_m·BLOCK_M + i` offset by
`cur_batch_in_all_start_index`. -/
```
```lean
noncomputable def ctxQTile
    (s : BlockState) (Q B_Start_Loc : RegionName) (BLOCK_M : Nat) :
    TileIndex [BLOCK_M, 128] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((startLoc s B_Start_Loc + (s.pids 2 * BLOCK_M + i.val)) * 576
        + curHead s * 96 + e.val)
```
</details>

<details><summary><code>ctxKTile</code></summary>

```
/-- Coordinate-faithful key tile: `K[kvloc j, cur_head, e]` at the checked layout
(`stride_kbs=576, stride_kh=96, stride_kd=1`). -/
```
```lean
noncomputable def ctxKTile
    (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, _) =>
    s.readMem K
      (kvLoc s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * 576
        + curHead s * 96 + e.val)
```
</details>

<details><summary><code>ctxVTile</code></summary>

```
/-- Coordinate-faithful value tile: `V[kvloc j, cur_head, d]`. -/
```
```lean
noncomputable def ctxVTile
    (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, _) =>
    s.readMem V
      (kvLoc s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * 576
        + curHead s * 96 + d.val)
```
</details>

<details><summary><code>curHead</code></summary>

```
/-- Head index of this kernel's program (`cur_head = pids 1`, `kv_group_num = 1`
so `cur_kv_head = cur_head`). -/
```
```lean
def curHead (s : BlockState) : Nat := s.pids 1
```
</details>

<details><summary><code>kvLoc</code></summary>

```
/-- Gathered KV token location for streamed key `j`:
`kv_loc = Req_to_tokens[cur_batch_req_idx · stride_b + j · stride_s]`. -/
```
```lean
def kvLoc (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s j : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (reqIdx s B_req_idx * stride_req_b + stride_req_s * j)
```
</details>

<details><summary><code>reqIdx</code></summary>

```
/-- Request index for this batch: `cur_batch_req_idx = B_req_idx[cur_batch]`. -/
```
```lean
def reqIdx (s : BlockState) (B_req_idx : RegionName) : Nat :=
  s.readMemValue .nat B_req_idx (s.pids 0)
```
</details>

## Public theorem: `context_attn_bloom_surface_python_block64_compute_correct`

**Statement:**
```lean
theorem context_attn_bloom_surface_python_block64_compute_correct
    (Q K V B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx
      B_Prompt_Cache_Len : RegionName) (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_fwd_kernel_surface Q K V
        ((Real.sqrt (96 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out Req_to_tokens
        B_req_idx B_Prompt_Cache_Len
        576 96 1 576 96 1 576 96 1 576 96 1
        7500 1 1 96 64 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        bloomFwdGenuineOutValue64 s Q K V B_Start_Loc B_Seqlen Out
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `initialState : = s`
- `fun idx : TileIndex [64, 128] =>
          active s B_Seqlen B_Prompt_Cache_Len 96 64 idx`
- `fun idx : TileIndex [64, 128] =>
          (Out, outOffset s B_Start_Loc 576 96 1 64 idx)`
- `expected : = fun idx : TileIndex [64, 128] =>
        bloomFwdGenuineOutValue64 s Q K V B_Start_Loc B_Seqlen Out
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx`

**Closed-form spec defs (transitive):** `context_attn_bloom_fwd_kernel_surface`, `active`, `outOffset`, `bloomFwdGenuineOutValue64`, `mIndex`, `seqLen`, `dIndex`, `startLoc`, `contextAttnBloomExactFoldM`, `sm_scale_bloom`, `bloomFwdWindow`, `bloomFwdBel`, `promptLen`, `gAccN`, `bloomKVM`, `gStateBot`, `bloomQTileM`, `bloomKTileM`, `bloomVTileM`, `gKeysUpto`, `osStepBot`, `ctxQTile`, `ctxKTile`, `ctxVTile`, `curHead`, `kvLoc`, `reqIdx`

<details><summary><code>context_attn_bloom_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_bloom.py`'s `_fwd_kernel`. -/
```
```lean
def context_attn_bloom_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=((start_n + offs_n[None, :]) < block_end_loc) &
        (offs_d[:, None] < $(head_dim)),
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :],
      qk, -100000000.0)

    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
    acc = acc * acc_scale[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=((start_n + offs_n[:, None]) < block_end_loc) &
        (offs_d[None, :] < $(head_dim)),
      other=0.0)
    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen B_Prompt_Cache_Len ∧
    dIndex idx < head_dim
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    s.pids 1 * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>bloomFwdGenuineOutValue64</code></summary>

```
/-- **Genuine closed-form `Out` value (BLOCK_M = 64)**: the block-causal-guarded
boundary-masked online-softmax fold `contextAttnBloomExactFoldM` of the loaded
Q/K/V memory — a pure function of memory, not the kernel's executed readback. -/
```
```lean
noncomputable def bloomFwdGenuineOutValue64
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [64, 128]) : ℝ :=
  contextAttnBloomExactFoldM s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx
    sm_scale_bloom 7500 1 64 (bloomFwdWindow s B_Seqlen B_Prompt_Cache_Len 64)
    (bloomFwdBel s B_Seqlen B_Prompt_Cache_Len 64) idx
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0) - promptLen s B_Prompt_Cache_Len
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>contextAttnBloomExactFoldM</code></summary>

```
/-- **The faithful kernel value** at output lane `(i,d)`: the block-causal-guarded
normalized accumulator `gAccN` of the ⊥-seeded online-softmax fold over `bloomKVM`
for the full streamed window `[0, S)` (`S / BLOCK_N` blocks of `BLOCK_N = 128`),
with this row's causal limit `qpos = gᵢ + plen`. Guard-fail blocks (`c·128 >
qpos`) contribute via `acc_scale = 1`, exactly matching the kernel's `tl.where`. -/
```
```lean
noncomputable def contextAttnBloomExactFoldM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  gAccN S 128 (s.pids 2 * BLOCK_M + idx.1.val + promptLen s B_Prompt_Cache_Len)
    (bloomKVM s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len
      Req_to_tokens B_req_idx sm_scale stride_req_b stride_req_s BLOCK_M S bel idx.1 idx.2.1)
    (S / 128)
```
</details>

<details><summary><code>sm_scale_bloom</code></summary>

```
/-- The kernel's chosen natural-`exp` `sm_scale` constant at the Python test shape
(`(√96)⁻¹`, fed `/ log 2` into the base-2 fold so `pow2 (score/log2) = exp score`). -/
```
```lean
noncomputable def sm_scale_bloom : ℝ := (Real.sqrt (96 : ℝ))⁻¹
```
</details>

<details><summary><code>bloomFwdWindow</code></summary>

```
/-- The streamed window `S = 128·⌈(block_mask·block_end_loc)/128⌉` (loop step is
`BLOCK_N = 128`; `block_end_loc` uses the query block size `BLOCK_M`). -/
```
```lean
def bloomFwdWindow (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BLOCK_M : Nat) : Nat :=
  let plen := promptLen s B_Prompt_Cache_Len
  let sl := seqLen s B_Seqlen B_Prompt_Cache_Len
  let bel := (let a := (s.pids 2 + 1) * BLOCK_M + plen
              let b := sl + plen
              if a < b then a else b)
  let bm := if BLOCK_M * s.pids 2 < sl then 1 else 0
  128 * ((bm * bel + 127) / 128)
```
</details>

<details><summary><code>bloomFwdBel</code></summary>

```
/-- `block_end_loc = min((start_m+1)·BLOCK_M + plen, cur_batch_seq_len + plen)`. -/
```
```lean
def bloomFwdBel (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BLOCK_M : Nat) : Nat :=
  let plen := promptLen s B_Prompt_Cache_Len
  let sl := seqLen s B_Seqlen B_Prompt_Cache_Len
  let a := (s.pids 2 + 1) * BLOCK_M + plen
  let b := sl + plen
  if a < b then a else b
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0)
```
</details>

<details><summary><code>gAccN</code></summary>

```
/-- **Faithful normalized accumulator** after `c` blocks for a row with causal
limit `qpos`. Mirrors the kernel's `acc` register: `acc_new = acc·acc_scale +
dot(p,v)`, with `acc_scale = (lᵢ/lᵢⁿᵉʷ)·α` on guard-pass blocks (`c·BN ≤ qpos`)
and `1` on guard-fail blocks, and `dot(p,v) = (numerⁿᵉʷ − numer·α)/lᵢⁿᵉʷ`. -/
```
```lean
noncomputable def gAccN (S BN qpos : Nat) (g : Fin S → ℝ × ℝ) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
    let st := gStateBot S (c * BN) g
    let stn := gStateBot S ((c + 1) * BN) g
    let α := (WithBot.realExp2 (WithBot.realSub st.1 stn.1)).unbotD 0
    let accScale := if c * BN ≤ qpos then (st.2.1 / stn.2.1) * α else 1
    gAccN S BN qpos g c * accScale + (stn.2.2 - st.2.2 * α) / stn.2.1
```
</details>

<details><summary><code>bloomKVM</code></summary>

```
/-- **Faithful per-key `(base-2 score, value)`** the loop folds, with the genuine
`-1e8` sentinel and the `block_end_loc` load mask on `k`/`v`. Score is fed in
base-2 (`/ log 2`) so `pow2 score = exp (kernel natural score)`. -/
```
```lean
noncomputable def bloomKVM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens
      B_req_idx : RegionName) (sm_scale : ℝ)
    (stride_req_b stride_req_s BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128)
    (j : Fin S) : ℝ × ℝ :=
  ((if j.val ≤ s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin 128 =>
        bloomQTileM s Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len BLOCK_M i e
          * bloomKTileM s K Req_to_tokens B_req_idx stride_req_b stride_req_s S bel (j, e, PUnit.unit))
    else (0.0 - 10e7 : ℝ)) / Real.log 2,
    bloomVTileM s V Req_to_tokens B_req_idx stride_req_b stride_req_s S bel (j, d, PUnit.unit))
```
</details>

<details><summary><code>gStateBot</code></summary>

```
/-- Generic ⊥-seeded running `(max, l, acc)` after streaming `[0, hi)`. -/
```
```lean
noncomputable def gStateBot (S hi : Nat) (g : Fin S → ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  (gKeysUpto S hi g).foldl osStepBot (⊥, 0, 0)
```
</details>

<details><summary><code>bloomQTileM</code></summary>

```
/-- Row- and channel-masked query tile: genuine `ctxQTile` on active rows and
channel `e < 96`, else `0` (channel mask `offs_d < head_dim = 96` matches the
kernel). -/
```
```lean
noncomputable def bloomQTileM
    (s : BlockState) (Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Fin 128) : ℝ :=
  if (s.pids 2 * BLOCK_M + i.val < seqLen s B_Seqlen B_Prompt_Cache_Len) ∧ (e.val < 96) then
    ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
  else 0
```
</details>

<details><summary><code>bloomKTileM</code></summary>

```
/-- `block_end_loc`/channel-masked key tile (genuine `ctxKTile` for `j < bel` and
channel `e < 96`, else `0`). The channel mask `e < head_dim = 96` matches the
kernel's `tl.load` mask `offs_d < head_dim`, which zeros padding channels
`e ∈ [96, 128)`. -/
```
```lean
noncomputable def bloomKTileM
    (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S bel : Nat) : TileIndex [S, 128] → ℝ :=
  fun (j, e, u) =>
    if (j.val < bel) ∧ (e.val < 96) then ctxKTile s K Req_to_tokens B_req_idx stride_req_b stride_req_s S (j, e, u)
    else 0
```
</details>

<details><summary><code>bloomVTileM</code></summary>

```
/-- `block_end_loc`/channel-masked value tile (channel mask `d < 96` matches the
kernel's `offs_d < head_dim` load mask). -/
```
```lean
noncomputable def bloomVTileM
    (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S bel : Nat) : TileIndex [S, 128] → ℝ :=
  fun (j, d, u) =>
    if (j.val < bel) ∧ (d.val < 96) then ctxVTile s V Req_to_tokens B_req_idx stride_req_b stride_req_s S (j, d, u)
    else 0
```
</details>

<details><summary><code>gKeysUpto</code></summary>

```
/-- Generic windowed key list `[0, hi)` over an abstract per-key `g`. -/
```
```lean
noncomputable def gKeysUpto (S hi : Nat) (g : Fin S → ℝ × ℝ) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S => if j.val < hi then some (g j) else none)
```
</details>

<details><summary><code>osStepBot</code></summary>

```
/-- One ⊥-seeded online-softmax step: running max in `WithBot ℝ` (seeded `⊥`), so
`α = realExp2(m ⊖ m')` is `0` on the first block. -/
```
```lean
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (sc - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)
```
</details>

<details><summary><code>ctxQTile</code></summary>

```
/-- Coordinate-faithful query tile of this kernel at `(cur_batch, cur_head,
start_m)` for the checked Python layout (`stride_qbs=576, stride_qh=96,
stride_qd=1`). Row `i` is the global prefill row `start_m·BLOCK_M + i` offset by
`cur_batch_in_all_start_index`. -/
```
```lean
noncomputable def ctxQTile
    (s : BlockState) (Q B_Start_Loc : RegionName) (BLOCK_M : Nat) :
    TileIndex [BLOCK_M, 128] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((startLoc s B_Start_Loc + (s.pids 2 * BLOCK_M + i.val)) * 576
        + curHead s * 96 + e.val)
```
</details>

<details><summary><code>ctxKTile</code></summary>

```
/-- Coordinate-faithful key tile: `K[kvloc j, cur_head, e]` at the checked layout
(`stride_kbs=576, stride_kh=96, stride_kd=1`). -/
```
```lean
noncomputable def ctxKTile
    (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, _) =>
    s.readMem K
      (kvLoc s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * 576
        + curHead s * 96 + e.val)
```
</details>

<details><summary><code>ctxVTile</code></summary>

```
/-- Coordinate-faithful value tile: `V[kvloc j, cur_head, d]`. -/
```
```lean
noncomputable def ctxVTile
    (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, _) =>
    s.readMem V
      (kvLoc s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * 576
        + curHead s * 96 + d.val)
```
</details>

<details><summary><code>curHead</code></summary>

```
/-- Head index of this kernel's program (`cur_head = pids 1`, `kv_group_num = 1`
so `cur_kv_head = cur_head`). -/
```
```lean
def curHead (s : BlockState) : Nat := s.pids 1
```
</details>

<details><summary><code>kvLoc</code></summary>

```
/-- Gathered KV token location for streamed key `j`:
`kv_loc = Req_to_tokens[cur_batch_req_idx · stride_b + j · stride_s]`. -/
```
```lean
def kvLoc (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s j : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (reqIdx s B_req_idx * stride_req_b + stride_req_s * j)
```
</details>

<details><summary><code>reqIdx</code></summary>

```
/-- Request index for this batch: `cur_batch_req_idx = B_req_idx[cur_batch]`. -/
```
```lean
def reqIdx (s : BlockState) (B_req_idx : RegionName) : Nat :=
  s.readMemValue .nat B_req_idx (s.pids 0)
```
</details>

## Public theorem: `context_attn_bloom_surface_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **General surface compute-correctness** for `context_attn_bloom.py` over symbolic
`BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`/`head_dim`, the per-axis K/V/Q/O strides and the
`Req_to_tokens` gather strides (`kv_group_num = 1`). Every active observable `Out`
write holds the genuine boundary-masked natural-exp in-loop-normalized causal-softmax
closed form `bloomFwdGenuineOutValueG` of the BLOOM gathered Q/K/V memory — a pure
function of memory, NOT the kernel's executed readback. The `-1e8` causal sentinel is
kept exactly. Side conditions: `0 < BLOCK_DMODEL`, `0 < BLOCK_N`, output-offset
injectivity. -/
```
</details>

**Statement:**
```lean
theorem context_attn_bloom_surface_compute_correct_general
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      head_dim BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (s : BlockState)
    (hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_bloom_fwd_kernel_surface Q K V sm_scale
        B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bloomFwdGenuineOutValueG s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
          sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M idx)
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_DMODEL`
- `hBN : 0 < BLOCK_N`
- `hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `kernel : = context_attn_bloom_fwd_kernel_surface Q K V sm_scale
        B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 head_dim BLOCK_M BLOCK_DMODEL BLOCK_N`
- `initialState : = s`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s B_Seqlen B_Prompt_Cache_Len head_dim BLOCK_M idx`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`
- `expected : = fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        bloomFwdGenuineOutValueG s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
          sm_scale stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M idx`

**Closed-form spec defs (transitive):** `outOffset`, `context_attn_bloom_fwd_kernel_surface`, `active`, `bloomFwdGenuineOutValueG`, `startLoc`, `mIndex`, `dIndex`, `seqLen`, `contextAttnBloomExactFoldMG`, `bloomFwdWindowG`, `bloomFwdBel`, `promptLen`, `gAccN`, `bloomKVMG`, `gStateBot`, `bloomQTileMG`, `bloomKTileMG`, `bloomVTileMG`, `gKeysUpto`, `osStepBot`, `bloomQTileG`, `bloomKTileG`, `bloomVTileG`, `curHead`, `bloomKvLocG`, `reqIdx`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    s.pids 1 * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>context_attn_bloom_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_bloom.py`'s `_fwd_kernel`. -/
```
```lean
def context_attn_bloom_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

  for start_n in range($(0), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=((start_n + offs_n[None, :]) < block_end_loc) &
        (offs_d[:, None] < $(head_dim)),
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :],
      qk, -100000000.0)

    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
    acc = acc * acc_scale[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=((start_n + offs_n[:, None]) < block_end_loc) &
        (offs_d[None, :] < $(head_dim)),
      other=0.0)
    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)))
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (head_dim BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen B_Prompt_Cache_Len ∧
    dIndex idx < head_dim
```
</details>

<details><summary><code>bloomFwdGenuineOutValueG</code></summary>

```
/-- **General genuine closed-form `Out` value**: the block-causal-guarded boundary-masked
in-loop-normalized online-softmax fold `contextAttnBloomExactFoldMG` of the gathered
Q/K/V memory — a pure function of memory, not the kernel's executed readback. -/
```
```lean
noncomputable def bloomFwdGenuineOutValueG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  contextAttnBloomExactFoldMG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
    stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M
    (bloomFwdWindowG s B_Seqlen B_Prompt_Cache_Len BLOCK_M BLOCK_N)
    (bloomFwdBel s B_Seqlen B_Prompt_Cache_Len BLOCK_M) idx
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0) - promptLen s B_Prompt_Cache_Len
```
</details>

<details><summary><code>contextAttnBloomExactFoldMG</code></summary>

```
/-- **General faithful kernel value** at output lane `(i,d)`: the block-causal-guarded
normalized accumulator `gAccN` of the ⊥-seeded online-softmax fold over `bloomKVMG`
for the full streamed window `[0, S)`. A pure function of `Q`/`K`/`V` memory. -/
```
```lean
noncomputable def contextAttnBloomExactFoldMG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_N BLOCK_M S bel : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  gAccN S BLOCK_N (s.pids 2 * BLOCK_M + idx.1.val + promptLen s B_Prompt_Cache_Len)
    (bloomKVMG s Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx sm_scale
      stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel idx.1 idx.2.1.val)
    (S / BLOCK_N)
```
</details>

<details><summary><code>bloomFwdWindowG</code></summary>

```
/-- General kernel-decoded streamed window `S = BLOCK_N·⌈(block_mask·block_end_loc)/BLOCK_N⌉`
(loop step `BLOCK_N`; `block_end_loc` uses query block size `BLOCK_M`). -/
```
```lean
def bloomFwdWindowG (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BLOCK_M BLOCK_N : Nat) : Nat :=
  let plen := promptLen s B_Prompt_Cache_Len
  let sl := seqLen s B_Seqlen B_Prompt_Cache_Len
  let bel := bloomFwdBel s B_Seqlen B_Prompt_Cache_Len BLOCK_M
  let bm := if BLOCK_M * s.pids 2 < sl then 1 else 0
  BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N)
```
</details>

<details><summary><code>bloomFwdBel</code></summary>

```
/-- `block_end_loc = min((start_m+1)·BLOCK_M + plen, cur_batch_seq_len + plen)`. -/
```
```lean
def bloomFwdBel (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BLOCK_M : Nat) : Nat :=
  let plen := promptLen s B_Prompt_Cache_Len
  let sl := seqLen s B_Seqlen B_Prompt_Cache_Len
  let a := (s.pids 2 + 1) * BLOCK_M + plen
  let b := sl + plen
  if a < b then a else b
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (s.pids 0)
```
</details>

<details><summary><code>gAccN</code></summary>

```
/-- **Faithful normalized accumulator** after `c` blocks for a row with causal
limit `qpos`. Mirrors the kernel's `acc` register: `acc_new = acc·acc_scale +
dot(p,v)`, with `acc_scale = (lᵢ/lᵢⁿᵉʷ)·α` on guard-pass blocks (`c·BN ≤ qpos`)
and `1` on guard-fail blocks, and `dot(p,v) = (numerⁿᵉʷ − numer·α)/lᵢⁿᵉʷ`. -/
```
```lean
noncomputable def gAccN (S BN qpos : Nat) (g : Fin S → ℝ × ℝ) : Nat → ℝ
  | 0 => 0
  | c + 1 =>
    let st := gStateBot S (c * BN) g
    let stn := gStateBot S ((c + 1) * BN) g
    let α := (WithBot.realExp2 (WithBot.realSub st.1 stn.1)).unbotD 0
    let accScale := if c * BN ≤ qpos then (st.2.1 / stn.2.1) * α else 1
    gAccN S BN qpos g c * accScale + (stn.2.2 - st.2.2 * α) / stn.2.1
```
</details>

<details><summary><code>bloomKVMG</code></summary>

```
/-- General faithful per-key `(base-2 score, value)` the loop folds, with the
`-1e8` sentinel kept and the `block_end_loc`/channel load masks; score fed
`/ log 2` so `pow2 score = exp (natural kernel score)`. -/
```
```lean
noncomputable def bloomKVMG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen B_Prompt_Cache_Len Req_to_tokens B_req_idx : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd head_dim BLOCK_DMODEL BLOCK_M S bel : Nat)
    (i : Fin BLOCK_M) (d : Nat) (j : Fin S) : ℝ × ℝ :=
  ((if j.val ≤ s.pids 2 * BLOCK_M + i.val + promptLen s B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
        bloomQTileMG s Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len stride_qbs stride_qh stride_qd head_dim BLOCK_M i e.val
          * bloomKTileMG s K Req_to_tokens B_req_idx stride_req_b stride_req_s stride_kbs stride_kh stride_kd head_dim S bel j e.val)
    else (0.0 - 10e7 : ℝ)) / Real.log 2,
    bloomVTileMG s V Req_to_tokens B_req_idx stride_req_b stride_req_s stride_vbs stride_vh stride_vd head_dim S bel j d)
```
</details>

<details><summary><code>gStateBot</code></summary>

```
/-- Generic ⊥-seeded running `(max, l, acc)` after streaming `[0, hi)`. -/
```
```lean
noncomputable def gStateBot (S hi : Nat) (g : Fin S → ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  (gKeysUpto S hi g).foldl osStepBot (⊥, 0, 0)
```
</details>

<details><summary><code>bloomQTileMG</code></summary>

```
/-- General row/channel-masked query tile: genuine `bloomQTileG` on active rows
(`gi < seq_len`) and channel `e < head_dim`, else `0`. -/
```
```lean
noncomputable def bloomQTileMG
    (s : BlockState) (Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (stride_qbs stride_qh stride_qd head_dim BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  if (s.pids 2 * BLOCK_M + i.val < seqLen s B_Seqlen B_Prompt_Cache_Len) ∧ (e < head_dim) then
    bloomQTileG s Q B_Start_Loc stride_qbs stride_qh stride_qd BLOCK_M i e
  else 0
```
</details>

<details><summary><code>bloomKTileMG</code></summary>

```
/-- General `block_end_loc`/channel-masked key tile: genuine `bloomKTileG` for
`j < bel` and channel `e < head_dim`, else `0`. -/
```
```lean
noncomputable def bloomKTileMG (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd head_dim S bel : Nat)
    (j : Fin S) (e : Nat) : ℝ :=
  if (j.val < bel) ∧ (e < head_dim) then
    bloomKTileG s K Req_to_tokens B_req_idx stride_req_b stride_req_s stride_kbs stride_kh stride_kd S j e
  else 0
```
</details>

<details><summary><code>bloomVTileMG</code></summary>

```
/-- General `block_end_loc`/channel-masked value tile. -/
```
```lean
noncomputable def bloomVTileMG (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_vbs stride_vh stride_vd head_dim S bel : Nat)
    (j : Fin S) (d : Nat) : ℝ :=
  if (j.val < bel) ∧ (d < head_dim) then
    bloomVTileG s V Req_to_tokens B_req_idx stride_req_b stride_req_s stride_vbs stride_vh stride_vd S j d
  else 0
```
</details>

<details><summary><code>gKeysUpto</code></summary>

```
/-- Generic windowed key list `[0, hi)` over an abstract per-key `g`. -/
```
```lean
noncomputable def gKeysUpto (S hi : Nat) (g : Fin S → ℝ × ℝ) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S => if j.val < hi then some (g j) else none)
```
</details>

<details><summary><code>osStepBot</code></summary>

```
/-- One ⊥-seeded online-softmax step: running max in `WithBot ℝ` (seeded `⊥`), so
`α = realExp2(m ⊖ m')` is `0` on the first block. -/
```
```lean
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (sc - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)
```
</details>

<details><summary><code>bloomQTileG</code></summary>

```
/-- General coordinate-faithful query tile `Q[gi, e]` at head/stride parameters. -/
```
```lean
noncomputable def bloomQTileG
    (s : BlockState) (Q B_Start_Loc : RegionName)
    (stride_qbs stride_qh stride_qd BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  s.readMem Q
    ((startLoc s B_Start_Loc + (s.pids 2 * BLOCK_M + i.val)) * stride_qbs
      + curHead s * stride_qh + e * stride_qd)
```
</details>

<details><summary><code>bloomKTileG</code></summary>

```
/-- General coordinate-faithful key tile `K[kv_loc(j), cur_head, e]`. -/
```
```lean
noncomputable def bloomKTileG (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_kbs stride_kh stride_kd S : Nat) (j : Fin S) (e : Nat) : ℝ :=
  s.readMem K (bloomKvLocG s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * stride_kbs
    + curHead s * stride_kh + e * stride_kd)
```
</details>

<details><summary><code>bloomVTileG</code></summary>

```
/-- General coordinate-faithful value tile `V[kv_loc(j), cur_head, d]`. -/
```
```lean
noncomputable def bloomVTileG (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s stride_vbs stride_vh stride_vd S : Nat) (j : Fin S) (d : Nat) : ℝ :=
  s.readMem V (bloomKvLocG s Req_to_tokens B_req_idx stride_req_b stride_req_s j.val * stride_vbs
    + curHead s * stride_vh + d * stride_vd)
```
</details>

<details><summary><code>curHead</code></summary>

```
/-- Head index of this kernel's program (`cur_head = pids 1`, `kv_group_num = 1`
so `cur_kv_head = cur_head`). -/
```
```lean
def curHead (s : BlockState) : Nat := s.pids 1
```
</details>

<details><summary><code>bloomKvLocG</code></summary>

```
/-- General `Req_to_tokens` gather: physical token slot for streamed key index
`j`, gather strides free:
`kv_loc(j) = Req_to_tokens[stride_req_b·req_idx + stride_req_s·j]`. -/
```
```lean
noncomputable def bloomKvLocG
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (stride_req_b stride_req_s j : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens (reqIdx s B_req_idx * stride_req_b + stride_req_s * j)
```
</details>

<details><summary><code>reqIdx</code></summary>

```
/-- Request index for this batch: `cur_batch_req_idx = B_req_idx[cur_batch]`. -/
```
```lean
def reqIdx (s : BlockState) (B_req_idx : RegionName) : Nat :=
  s.readMemValue .nat B_req_idx (s.pids 0)
```
</details>
