# Spec sheet — `bench/tritonbench_g/context_attn_llama/ContextAttnLlama.lean`

**Python source:** `bench/tritonbench_g/context_attn_llama/context_attn_llama.py`

## Public theorem: `context_attn_llama_final_store_slice_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the masked context-attention output store. -/
```
</details>

**Statement:**
```lean
theorem context_attn_llama_final_store_slice_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (H
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out H stride_acc_b stride_acc_h stride_acc_m
        stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len H stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`
- `kernel : = context_attn_llama_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out H stride_acc_b stride_acc_h stride_acc_m
        stride_acc_d stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL`
- `initialState : = s`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`
- `expected : = fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len H stride_acc_b
          stride_acc_h stride_acc_m stride_acc_d BLOCK_M idx`

**Closed-form spec defs (transitive):** `outOffset`, `context_attn_llama_final_store_slice`, `active`, `accStoreValue`, `startLoc`, `mIndex`, `curHead`, `dIndex`, `seqLen`, `accOffset`, `curBatch`, `promptLen`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (H : Nat) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s H B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    curHead s H * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>context_attn_llama_final_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `context_attn_llama.py`'s
`_fwd_kernel`.

The full kernel computes grouped-KV causal context attention using
`Req_to_tokens`. This slice starts from a precomputed `Acc` tile and proves the
final masked writeback into `Out`, preserving the fused `cur_bh` program-id
decomposition, `B_Start_Loc`, and the prompt-cache-adjusted sequence length.
The inner `tl.float32` `m_i/l_i/acc` streaming-softmax loop and request-token
gathers are outside this slice. -/
```
```lean
def context_attn_llama_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen B_Prompt_Cache_Len : Region .nat)
    (Out : RegionName)
    (H
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)
  prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(BLOCK_DMODEL))
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
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H B_Seqlen B_Prompt_Cache_Len
```
</details>

<details><summary><code>accStoreValue</code></summary>

```lean
noncomputable def accStoreValue
    (s : BlockState) (Acc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (H stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (H : Nat) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (curBatch s H)
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>curHead</code></summary>

```lean
def curHead (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
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
def seqLen
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (curBatch s H) -
    promptLen s H B_Prompt_Cache_Len
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (H stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  curBatch s H * stride_acc_b + curHead s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d
```
</details>

<details><summary><code>curBatch</code></summary>

```lean
def curBatch (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (H : Nat) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (curBatch s H)
```
</details>

## Public theorem: `context_attn_llama_final_store_python_block128_compute_correct`

**Statement:**
```lean
theorem context_attn_llama_final_store_python_block128_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 16 4194304 128
          2048 1 128 idx)
```

**Assumptions / layout contracts:**
- `kernel : = context_attn_llama_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1 128 128`
- `initialState : = s`
- `fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx`
- `fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)`
- `expected : = fun idx : TileIndex [128, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 16 4194304 128
          2048 1 128 idx`

**Closed-form spec defs (transitive):** `context_attn_llama_final_store_slice`, `active`, `outOffset`, `accStoreValue`, `mIndex`, `seqLen`, `startLoc`, `curHead`, `dIndex`, `accOffset`, `curBatch`, `promptLen`

<details><summary><code>context_attn_llama_final_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `context_attn_llama.py`'s
`_fwd_kernel`.

The full kernel computes grouped-KV causal context attention using
`Req_to_tokens`. This slice starts from a precomputed `Acc` tile and proves the
final masked writeback into `Out`, preserving the fused `cur_bh` program-id
decomposition, `B_Start_Loc`, and the prompt-cache-adjusted sequence length.
The inner `tl.float32` `m_i/l_i/acc` streaming-softmax loop and request-token
gathers are outside this slice. -/
```
```lean
def context_attn_llama_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen B_Prompt_Cache_Len : Region .nat)
    (Out : RegionName)
    (H
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)
  prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(BLOCK_DMODEL))
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
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H B_Seqlen B_Prompt_Cache_Len
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (H : Nat) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s H B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    curHead s H * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>accStoreValue</code></summary>

```lean
noncomputable def accStoreValue
    (s : BlockState) (Acc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (H stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (curBatch s H) -
    promptLen s H B_Prompt_Cache_Len
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (H : Nat) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (curBatch s H)
```
</details>

<details><summary><code>curHead</code></summary>

```lean
def curHead (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (H stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  curBatch s H * stride_acc_b + curHead s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d
```
</details>

<details><summary><code>curBatch</code></summary>

```lean
def curBatch (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (H : Nat) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (curBatch s H)
```
</details>

## Public theorem: `context_attn_llama_final_store_python_block64_compute_correct`

**Statement:**
```lean
theorem context_attn_llama_final_store_python_block64_compute_correct
    (Acc B_Start_Loc B_Seqlen B_Prompt_Cache_Len Out : RegionName)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 16 4194304 128
          2048 1 64 idx)
```

**Assumptions / layout contracts:**
- `kernel : = context_attn_llama_final_store_slice Acc B_Start_Loc B_Seqlen
        B_Prompt_Cache_Len Out 16 4194304 128 2048 1 2048 128 1 64 128`
- `initialState : = s`
- `fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx`
- `fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)`
- `expected : = fun idx : TileIndex [64, 128] =>
        accStoreValue s Acc B_Seqlen B_Prompt_Cache_Len 16 4194304 128
          2048 1 64 idx`

**Closed-form spec defs (transitive):** `context_attn_llama_final_store_slice`, `active`, `outOffset`, `accStoreValue`, `mIndex`, `seqLen`, `startLoc`, `curHead`, `dIndex`, `accOffset`, `curBatch`, `promptLen`

<details><summary><code>context_attn_llama_final_store_slice</code></summary>

```
/-- Surface transcription/proof-oriented final output-store slice of `context_attn_llama.py`'s
`_fwd_kernel`.

The full kernel computes grouped-KV causal context attention using
`Req_to_tokens`. This slice starts from a precomputed `Acc` tile and proves the
final masked writeback into `Out`, preserving the fused `cur_bh` program-id
decomposition, `B_Start_Loc`, and the prompt-cache-adjusted sequence length.
The inner `tl.float32` `m_i/l_i/acc` streaming-softmax loop and request-token
gathers are outside this slice. -/
```
```lean
def context_attn_llama_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen B_Prompt_Cache_Len : Region .nat)
    (Out : RegionName)
    (H
      stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)
  prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(BLOCK_DMODEL))
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
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H B_Seqlen B_Prompt_Cache_Len
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (H : Nat) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s H B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    curHead s H * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>accStoreValue</code></summary>

```lean
noncomputable def accStoreValue
    (s : BlockState) (Acc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (H stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s H stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (curBatch s H) -
    promptLen s H B_Prompt_Cache_Len
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (H : Nat) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (curBatch s H)
```
</details>

<details><summary><code>curHead</code></summary>

```lean
def curHead (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (H stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  curBatch s H * stride_acc_b + curHead s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d
```
</details>

<details><summary><code>curBatch</code></summary>

```lean
def curBatch (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (H : Nat) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (curBatch s H)
```
</details>

## Public theorem: `context_attn_llama_surface_python_block128_compute_correct`

**Statement:**
```lean
theorem context_attn_llama_surface_python_block128_compute_correct
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_fwd_kernel_surface Q K V
        (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
        B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
        2048 128 1 2048 128 1 2048 128 1 2048 128 1
        9048 1 1 16 128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        ctxFwdGenuineOutValue128 s Q K V B_Start_Loc B_Seqlen
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `initialState : = s`
- `fun idx : TileIndex [128, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 128 idx`
- `fun idx : TileIndex [128, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 128 idx)`
- `expected : = fun idx : TileIndex [128, 128] =>
        ctxFwdGenuineOutValue128 s Q K V B_Start_Loc B_Seqlen
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx`

**Closed-form spec defs (transitive):** `context_attn_llama_fwd_kernel_surface`, `active`, `outOffset`, `ctxFwdGenuineOutValue128`, `mIndex`, `seqLen`, `startLoc`, `curHead`, `dIndex`, `contextAttnExactFoldM`, `sm_scale_python`, `ctxFwdWindow`, `ctxFwdBel`, `curBatch`, `promptLen`, `gStateBot`, `ctxKVM`, `gKeysUpto`, `osStepBot`, `ctxQTileM`, `ctxKTileM`, `ctxVTileM`, `ctxQTile`, `ctxKTile`, `ctxVTile`, `ctxKvLoc`

<details><summary><code>context_attn_llama_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_llama.py`'s `_fwd_kernel`. -/
```
```lean
def context_attn_llama_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen : Region .nat)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = block_start_loc + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum(block_start_loc + $(BLOCK_M) + prompt_cache_len,
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
      mask=(start_n + offs_n[None, :]) < block_end_loc,
      other=0.0)
    qk = tl.dot(q, k)

    mask = offs_m[:, None] + prompt_cache_len >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk * $((sm_scale : ℝ)), -1.0e8)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=(start_n + offs_n[:, None]) < block_end_loc,
      other=0.0)
    p = (p).to(v.dtype)
    acc = tl.dot(p, v, acc)
    m_i = m_ij
  }

  acc = acc / l_i[:, None]
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H B_Seqlen B_Prompt_Cache_Len
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (H : Nat) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s H B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    curHead s H * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>ctxFwdGenuineOutValue128</code></summary>

```
/-- **Genuine closed-form output value** of `context_attn_llama.py` at the Python test
shape, BLOCK_M = 128: the boundary-masked causal-softmax fold `contextAttnExactFoldM`
of the loaded Q/K/V memory — a pure function of memory, NOT the kernel's executed
readback. -/
```
```lean
noncomputable def ctxFwdGenuineOutValue128
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  contextAttnExactFoldM s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
    sm_scale_python 128 (ctxFwdWindow s B_Seqlen B_Prompt_Cache_Len 128)
    (ctxFwdBel s B_Seqlen B_Prompt_Cache_Len 128) idx
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (curBatch s H) -
    promptLen s H B_Prompt_Cache_Len
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (H : Nat) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (curBatch s H)
```
</details>

<details><summary><code>curHead</code></summary>

```lean
def curHead (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>contextAttnExactFoldM</code></summary>

```
/-- **The faithful kernel value** at output lane `(i, d)`: `acc/l` of the
⊥-seeded online-softmax fold over `ctxKVM` for the full streamed window `[0, S)`
(`S = final`). A pure function of `Q`/`K`/`V` memory — exactly what the loop's
`m_i`/`l_i`/`acc` realize, phantom keys included. -/
```
```lean
noncomputable def contextAttnExactFoldM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let st := gStateBot S S (ctxKVM s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len sm_scale
      BLOCK_M S bel idx.1 idx.2.1)
  st.2.2 / st.2.1
```
</details>

<details><summary><code>sm_scale_python</code></summary>

```
/-- The kernel's chosen `sm_scale` constant at the Python test shape. -/
```
```lean
noncomputable def sm_scale_python : ℝ := ((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634
```
</details>

<details><summary><code>ctxFwdWindow</code></summary>

```
/-- The kernel-decoded streamed window `S = ceil_BM(block_mask·block_end_loc)` at the
Python test shape (BLOCK_N = BLOCK_M): the first multiple of `BM` at/above the
dynamic loop bound. -/
```
```lean
def ctxFwdWindow (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BM : Nat) : Nat :=
  let plen := s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / 16)
  let sl := s.readMemValue .nat B_Seqlen (s.pids 1 / 16) - plen
  let bel := (let a := BM * s.pids 0 + BM + plen
              let b := sl + plen
              if a < b then a else b)
  let bm := if BM * s.pids 0 < sl then 1 else 0
  BM * ((bm * bel + (BM - 1)) / BM)
```
</details>

<details><summary><code>ctxFwdBel</code></summary>

```
/-- The kernel-decoded `block_end_loc = min(BM·start_m + BM + plen, seq_len + plen)`
at the Python test shape. -/
```
```lean
def ctxFwdBel (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BM : Nat) : Nat :=
  let plen := s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / 16)
  let sl := s.readMemValue .nat B_Seqlen (s.pids 1 / 16) - plen
  let a := BM * s.pids 0 + BM + plen
  let b := sl + plen
  if a < b then a else b
```
</details>

<details><summary><code>curBatch</code></summary>

```lean
def curBatch (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (H : Nat) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (curBatch s H)
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

<details><summary><code>ctxKVM</code></summary>

```
/-- **Faithful per-key `(score, value)` the loop folds**: `ctxKV` with the
`block_end_loc` load mask applied to `k`/`v` and the row mask applied to `q`.
Active causal lane (`j ≤ gi+plen`) gets `sm·Σ_e ctxQTileM(i,e)·ctxKTileM(j,e)` (so
phantom `j ≥ bel` get `sm·0 = 0`); future lane gets the `-1e8` sentinel; value is
the masked `ctxVTileM` (`0` for `j ≥ bel`). -/
```
```lean
noncomputable def ctxKVM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128) (j : Fin S) : ℝ × ℝ :=
  (if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin 128 =>
        ctxQTileM s Q B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len BLOCK_M i e
          * ctxKTileM s K Req_to_tokens B_req_idx S bel (j, e, PUnit.unit))
    else (0.0 - 10e7 : ℝ),
    ctxVTileM s V Req_to_tokens B_req_idx S bel (j, d, PUnit.unit))
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
`α = realExp2(m ⊖ m')` is `0` on the first block — faithful to the kernel's
`m_i = tl.zeros − inf` and `l_i`/`acc = 0`. -/
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

<details><summary><code>ctxQTileM</code></summary>

```
/-- Row-masked query tile: `ctxQTile` for active rows (`128·pids0 + i < seq_len`),
else `0` (the kernel's `q` load mask `offs_m < cur_batch_seq_len, other=0`). On
active rows it is the genuine `ctxQTile`, so the closed form is the true causal
softmax there; inactive rows are masked off at the final store. -/
```
```lean
noncomputable def ctxQTileM
    (s : BlockState) (Q B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Fin 128) : ℝ :=
  if s.pids 0 * BLOCK_M + i.val < seqLen s 16 B_Seqlen B_Prompt_Cache_Len then
    ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
  else 0
```
</details>

<details><summary><code>ctxKTileM</code></summary>

```
/-- `block_end_loc`-masked key tile: `ctxKTile` for `j < bel`, else `0` (the
kernel's `k` load mask `(start_n+offs_n) < block_end_loc, other=0`). -/
```
```lean
noncomputable def ctxKTileM (s : BlockState) (K Req_to_tokens B_req_idx : RegionName) (S bel : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, u) => if j.val < bel then ctxKTile s K Req_to_tokens B_req_idx S (j, e, u) else 0
```
</details>

<details><summary><code>ctxVTileM</code></summary>

```
/-- `block_end_loc`-masked value tile: `ctxVTile` for `j < bel`, else `0`. -/
```
```lean
noncomputable def ctxVTileM (s : BlockState) (V Req_to_tokens B_req_idx : RegionName) (S bel : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, u) => if j.val < bel then ctxVTile s V Req_to_tokens B_req_idx S (j, d, u) else 0
```
</details>

<details><summary><code>ctxQTile</code></summary>

```
/-- Coordinate-faithful query tile of this kernel at `(start_m, cur_bh)` for the
checked Python layout (strides `stride_qbs=2048, stride_qh=128, stride_qd=1`,
`H=16`). Row `i` is the *global* prefill row `start_m·BLOCK_M + i` offset by
`cur_batch_in_all_start_index`. -/
```
```lean
noncomputable def ctxQTile
    (s : BlockState) (Q B_Start_Loc : RegionName) (BLOCK_M : Nat) :
    TileIndex [BLOCK_M, 128] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((s.readMemValue .nat B_Start_Loc (curBatch s 16) + (s.pids 0 * BLOCK_M + i.val))
          * 2048 + curHead s 16 * 128 + e.val)
```
</details>

<details><summary><code>ctxKTile</code></summary>

```
/-- Coordinate-faithful key tile: `K[kv_loc(j), cur_head, e]` at the checked
Llama layout (`stride_kbs=2048, stride_kh=128, stride_kd=1`, `kv_group_num=1` so
`cur_kv_head=cur_head`), where `kv_loc(j) = Req_to_tokens[9048·req_idx + j]`. -/
```
```lean
noncomputable def ctxKTile
    (s : BlockState) (K Req_to_tokens B_req_idx : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (ctxKvLoc s Req_to_tokens B_req_idx j.val * 2048
      + curHead s 16 * 128 + e.val)
```
</details>

<details><summary><code>ctxVTile</code></summary>

```
/-- Coordinate-faithful value tile: `V[kv_loc(j), cur_head, d]`. -/
```
```lean
noncomputable def ctxVTile
    (s : BlockState) (V Req_to_tokens B_req_idx : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (ctxKvLoc s Req_to_tokens B_req_idx j.val * 2048
      + curHead s 16 * 128 + d.val)
```
</details>

<details><summary><code>ctxKvLoc</code></summary>

```
/-- The Llama `Req_to_tokens` gather: physical token slot for streamed key index
`j`, decoded at the checked Python layout (`stride_req_to_tokens_b = 9048`,
`stride_req_to_tokens_s = 1`, request id `req_idx = B_req_idx[cur_batch]`):
`kv_loc(j) = Req_to_tokens[9048·req_idx + j]`. The key/value coordinate tiles
read physical token `kv_loc(j)` rather than the streamed index `j` — this is the
sole structural difference from `context_attn_fwd`'s sequential indexing. -/
```
```lean
def ctxKvLoc
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName) (j : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (9048 * s.readMemValue .nat B_req_idx (curBatch s 16) + j)
```
</details>

## Public theorem: `context_attn_llama_surface_python_block64_compute_correct`

**Statement:**
```lean
theorem context_attn_llama_surface_python_block64_compute_correct
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_fwd_kernel_surface Q K V
        (((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634) Out
        B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
        2048 128 1 2048 128 1 2048 128 1 2048 128 1
        9048 1 1 16 128 64 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx)
        (fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)))
      (expected := fun idx : TileIndex [64, 128] =>
        ctxFwdGenuineOutValue64 s Q K V B_Start_Loc B_Seqlen
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx)
```

**Assumptions / layout contracts:**
- `hundef : ∀ rg o, s.undef rg o = 0`
- `initialState : = s`
- `fun idx : TileIndex [64, 128] =>
          active s 16 B_Seqlen B_Prompt_Cache_Len 64 idx`
- `fun idx : TileIndex [64, 128] =>
          (Out, outOffset s 16 B_Start_Loc 2048 128 1 64 idx)`
- `expected : = fun idx : TileIndex [64, 128] =>
        ctxFwdGenuineOutValue64 s Q K V B_Start_Loc B_Seqlen
          Req_to_tokens B_req_idx B_Prompt_Cache_Len idx`

**Closed-form spec defs (transitive):** `context_attn_llama_fwd_kernel_surface`, `active`, `outOffset`, `ctxFwdGenuineOutValue64`, `mIndex`, `seqLen`, `startLoc`, `curHead`, `dIndex`, `contextAttnExactFoldM`, `sm_scale_python`, `ctxFwdWindow64`, `ctxFwdBel`, `curBatch`, `promptLen`, `gStateBot`, `ctxKVM`, `gKeysUpto`, `osStepBot`, `ctxQTileM`, `ctxKTileM`, `ctxVTileM`, `ctxQTile`, `ctxKTile`, `ctxVTile`, `ctxKvLoc`

<details><summary><code>context_attn_llama_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_llama.py`'s `_fwd_kernel`. -/
```
```lean
def context_attn_llama_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen : Region .nat)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = block_start_loc + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum(block_start_loc + $(BLOCK_M) + prompt_cache_len,
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
      mask=(start_n + offs_n[None, :]) < block_end_loc,
      other=0.0)
    qk = tl.dot(q, k)

    mask = offs_m[:, None] + prompt_cache_len >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk * $((sm_scale : ℝ)), -1.0e8)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=(start_n + offs_n[:, None]) < block_end_loc,
      other=0.0)
    p = (p).to(v.dtype)
    acc = tl.dot(p, v, acc)
    m_i = m_ij
  }

  acc = acc / l_i[:, None]
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H B_Seqlen B_Prompt_Cache_Len
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (H : Nat) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s H B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    curHead s H * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>ctxFwdGenuineOutValue64</code></summary>

```
/-- **Genuine closed-form output value** of `context_attn_llama.py` at the Python test
shape, BLOCK_M = 64 (Tesla): the boundary-masked causal-softmax fold
`contextAttnExactFoldM` of the loaded Q/K/V memory — a pure function of memory, NOT
the kernel's executed readback. -/
```
```lean
noncomputable def ctxFwdGenuineOutValue64
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (idx : TileIndex [64, 128]) : ℝ :=
  contextAttnExactFoldM s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
    sm_scale_python 64 (ctxFwdWindow64 s B_Seqlen B_Prompt_Cache_Len)
    (ctxFwdBel s B_Seqlen B_Prompt_Cache_Len 64) idx
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (curBatch s H) -
    promptLen s H B_Prompt_Cache_Len
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (H : Nat) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (curBatch s H)
```
</details>

<details><summary><code>curHead</code></summary>

```lean
def curHead (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val
```
</details>

<details><summary><code>contextAttnExactFoldM</code></summary>

```
/-- **The faithful kernel value** at output lane `(i, d)`: `acc/l` of the
⊥-seeded online-softmax fold over `ctxKVM` for the full streamed window `[0, S)`
(`S = final`). A pure function of `Q`/`K`/`V` memory — exactly what the loop's
`m_i`/`l_i`/`acc` realize, phantom keys included. -/
```
```lean
noncomputable def contextAttnExactFoldM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ) (BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let st := gStateBot S S (ctxKVM s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len sm_scale
      BLOCK_M S bel idx.1 idx.2.1)
  st.2.2 / st.2.1
```
</details>

<details><summary><code>sm_scale_python</code></summary>

```
/-- The kernel's chosen `sm_scale` constant at the Python test shape. -/
```
```lean
noncomputable def sm_scale_python : ℝ := ((Real.sqrt (128 : ℝ))⁻¹) * 1.4426950408889634
```
</details>

<details><summary><code>ctxFwdWindow64</code></summary>

```
/-- The Tesla-shape streamed window: the first multiple of `BLOCK_N = 128` at/above
the dynamic loop bound `block_mask · block_end_loc` (the loop step is `BLOCK_N`,
while `block_end_loc` is computed with the query block size `BLOCK_M = 64`). -/
```
```lean
def ctxFwdWindow64 (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  let plen := s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / 16)
  let sl := s.readMemValue .nat B_Seqlen (s.pids 1 / 16) - plen
  let bel := (let a := 64 * s.pids 0 + 64 + plen
              let b := sl + plen
              if a < b then a else b)
  let bm := if 64 * s.pids 0 < sl then 1 else 0
  128 * ((bm * bel + 127) / 128)
```
</details>

<details><summary><code>ctxFwdBel</code></summary>

```
/-- The kernel-decoded `block_end_loc = min(BM·start_m + BM + plen, seq_len + plen)`
at the Python test shape. -/
```
```lean
def ctxFwdBel (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (BM : Nat) : Nat :=
  let plen := s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / 16)
  let sl := s.readMemValue .nat B_Seqlen (s.pids 1 / 16) - plen
  let a := BM * s.pids 0 + BM + plen
  let b := sl + plen
  if a < b then a else b
```
</details>

<details><summary><code>curBatch</code></summary>

```lean
def curBatch (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (H : Nat) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (curBatch s H)
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

<details><summary><code>ctxKVM</code></summary>

```
/-- **Faithful per-key `(score, value)` the loop folds**: `ctxKV` with the
`block_end_loc` load mask applied to `k`/`v` and the row mask applied to `q`.
Active causal lane (`j ≤ gi+plen`) gets `sm·Σ_e ctxQTileM(i,e)·ctxKTileM(j,e)` (so
phantom `j ≥ bel` get `sm·0 = 0`); future lane gets the `-1e8` sentinel; value is
the masked `ctxVTileM` (`0` for `j ≥ bel`). -/
```
```lean
noncomputable def ctxKVM
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128) (j : Fin S) : ℝ × ℝ :=
  (if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s 16 B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin 128 =>
        ctxQTileM s Q B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len BLOCK_M i e
          * ctxKTileM s K Req_to_tokens B_req_idx S bel (j, e, PUnit.unit))
    else (0.0 - 10e7 : ℝ),
    ctxVTileM s V Req_to_tokens B_req_idx S bel (j, d, PUnit.unit))
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
`α = realExp2(m ⊖ m')` is `0` on the first block — faithful to the kernel's
`m_i = tl.zeros − inf` and `l_i`/`acc = 0`. -/
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

<details><summary><code>ctxQTileM</code></summary>

```
/-- Row-masked query tile: `ctxQTile` for active rows (`128·pids0 + i < seq_len`),
else `0` (the kernel's `q` load mask `offs_m < cur_batch_seq_len, other=0`). On
active rows it is the genuine `ctxQTile`, so the closed form is the true causal
softmax there; inactive rows are masked off at the final store. -/
```
```lean
noncomputable def ctxQTileM
    (s : BlockState) (Q B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Fin 128) : ℝ :=
  if s.pids 0 * BLOCK_M + i.val < seqLen s 16 B_Seqlen B_Prompt_Cache_Len then
    ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
  else 0
```
</details>

<details><summary><code>ctxKTileM</code></summary>

```
/-- `block_end_loc`-masked key tile: `ctxKTile` for `j < bel`, else `0` (the
kernel's `k` load mask `(start_n+offs_n) < block_end_loc, other=0`). -/
```
```lean
noncomputable def ctxKTileM (s : BlockState) (K Req_to_tokens B_req_idx : RegionName) (S bel : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, u) => if j.val < bel then ctxKTile s K Req_to_tokens B_req_idx S (j, e, u) else 0
```
</details>

<details><summary><code>ctxVTileM</code></summary>

```
/-- `block_end_loc`-masked value tile: `ctxVTile` for `j < bel`, else `0`. -/
```
```lean
noncomputable def ctxVTileM (s : BlockState) (V Req_to_tokens B_req_idx : RegionName) (S bel : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, u) => if j.val < bel then ctxVTile s V Req_to_tokens B_req_idx S (j, d, u) else 0
```
</details>

<details><summary><code>ctxQTile</code></summary>

```
/-- Coordinate-faithful query tile of this kernel at `(start_m, cur_bh)` for the
checked Python layout (strides `stride_qbs=2048, stride_qh=128, stride_qd=1`,
`H=16`). Row `i` is the *global* prefill row `start_m·BLOCK_M + i` offset by
`cur_batch_in_all_start_index`. -/
```
```lean
noncomputable def ctxQTile
    (s : BlockState) (Q B_Start_Loc : RegionName) (BLOCK_M : Nat) :
    TileIndex [BLOCK_M, 128] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((s.readMemValue .nat B_Start_Loc (curBatch s 16) + (s.pids 0 * BLOCK_M + i.val))
          * 2048 + curHead s 16 * 128 + e.val)
```
</details>

<details><summary><code>ctxKTile</code></summary>

```
/-- Coordinate-faithful key tile: `K[kv_loc(j), cur_head, e]` at the checked
Llama layout (`stride_kbs=2048, stride_kh=128, stride_kd=1`, `kv_group_num=1` so
`cur_kv_head=cur_head`), where `kv_loc(j) = Req_to_tokens[9048·req_idx + j]`. -/
```
```lean
noncomputable def ctxKTile
    (s : BlockState) (K Req_to_tokens B_req_idx : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, _) =>
    s.readMem K (ctxKvLoc s Req_to_tokens B_req_idx j.val * 2048
      + curHead s 16 * 128 + e.val)
```
</details>

<details><summary><code>ctxVTile</code></summary>

```
/-- Coordinate-faithful value tile: `V[kv_loc(j), cur_head, d]`. -/
```
```lean
noncomputable def ctxVTile
    (s : BlockState) (V Req_to_tokens B_req_idx : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, _) =>
    s.readMem V (ctxKvLoc s Req_to_tokens B_req_idx j.val * 2048
      + curHead s 16 * 128 + d.val)
```
</details>

<details><summary><code>ctxKvLoc</code></summary>

```
/-- The Llama `Req_to_tokens` gather: physical token slot for streamed key index
`j`, decoded at the checked Python layout (`stride_req_to_tokens_b = 9048`,
`stride_req_to_tokens_s = 1`, request id `req_idx = B_req_idx[cur_batch]`):
`kv_loc(j) = Req_to_tokens[9048·req_idx + j]`. The key/value coordinate tiles
read physical token `kv_loc(j)` rather than the streamed index `j` — this is the
sole structural difference from `context_attn_fwd`'s sequential indexing. -/
```
```lean
def ctxKvLoc
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName) (j : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (9048 * s.readMemValue .nat B_req_idx (curBatch s 16) + j)
```
</details>

## Public theorem: `context_attn_llama_surface_compute_correct_general`

<details><summary>docstring</summary>

```
/-- **General surface compute-correctness** for `context_attn_llama.py` over symbolic
`BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`/`H`, the per-axis K/V/Q strides and the gather
strides (`kv_group_num = 1`). Every active observable `Out` write holds the genuine
boundary-masked causal-softmax closed form `ctxFwdGenuineOutValueG` of the Llama
`Req_to_tokens`-gathered Q/K/V memory — a pure function of memory, NOT the kernel's
executed readback. Side conditions: `0 < BLOCK_DMODEL`, `0 < BLOCK_N`, output-offset
injectivity. -/
```
</details>

**Statement:**
```lean
theorem context_attn_llama_surface_compute_correct_general
    (Q K V Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) (hD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N)
    (s : BlockState)
    (hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_llama_fwd_kernel_surface Q K V sm_scale Out
        B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 H BLOCK_DMODEL BLOCK_M BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        ctxFwdGenuineOutValueG s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
          sm_scale H stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd BLOCK_DMODEL BLOCK_M BLOCK_N idx)
```

**Assumptions / layout contracts:**
- `hD : 0 < BLOCK_DMODEL`
- `hBN : 0 < BLOCK_N`
- `hOInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `kernel : = context_attn_llama_fwd_kernel_surface Q K V sm_scale Out
        B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        stride_req_b stride_req_s 1 H BLOCK_DMODEL BLOCK_M BLOCK_N`
- `initialState : = s`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s H B_Seqlen B_Prompt_Cache_Len BLOCK_M idx`
- `fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          (Out, outOffset s H B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)`
- `expected : = fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        ctxFwdGenuineOutValueG s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len
          sm_scale H stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
          stride_vbs stride_vh stride_vd BLOCK_DMODEL BLOCK_M BLOCK_N idx`

**Closed-form spec defs (transitive):** `outOffset`, `context_attn_llama_fwd_kernel_surface`, `active`, `ctxFwdGenuineOutValueG`, `startLoc`, `mIndex`, `curHead`, `dIndex`, `seqLen`, `contextAttnExactFoldMG`, `ctxFwdWindowG`, `ctxFwdBelG`, `curBatch`, `promptLen`, `gStateBot`, `ctxKVMG`, `gKeysUpto`, `osStepBot`, `ctxQTileMG`, `ctxKTileMG`, `ctxVTileMG`, `ctxQTileG`, `ctxKTileG`, `ctxVTileG`, `ctxKvLocG`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState) (H : Nat) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s H B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    curHead s H * stride_oh + dIndex idx * stride_od
```
</details>

<details><summary><code>context_attn_llama_fwd_kernel_surface</code></summary>

```
/-- Faithful DSL port of `context_attn_llama.py`'s `_fwd_kernel`. -/
```
```lean
def context_attn_llama_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen : Region .nat)
    (Req_to_tokens B_req_idx b_prompt_cache_len : Region .nat)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(H)
  cur_head = cur_bh % $(H)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = block_start_loc + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum(block_start_loc + $(BLOCK_M) + prompt_cache_len,
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
      mask=(start_n + offs_n[None, :]) < block_end_loc,
      other=0.0)
    qk = tl.dot(q, k)

    mask = offs_m[:, None] + prompt_cache_len >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk * $((sm_scale : ℝ)), -1.0e8)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=(start_n + offs_n[:, None]) < block_end_loc,
      other=0.0)
    p = (p).to(v.dtype)
    acc = tl.dot(p, v, acc)
    m_i = m_ij
  }

  acc = acc / l_i[:, None]
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName)
    (BLOCK_M : Nat) (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s H B_Seqlen B_Prompt_Cache_Len
```
</details>

<details><summary><code>ctxFwdGenuineOutValueG</code></summary>

```
/-- **General genuine closed-form output value** (Llama gather). -/
```
```lean
noncomputable def ctxFwdGenuineOutValueG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (H stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  contextAttnExactFoldMG s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len sm_scale
    H stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd BLOCK_DMODEL BLOCK_M
    (ctxFwdWindowG s B_Seqlen B_Prompt_Cache_Len H BLOCK_M BLOCK_N)
    (ctxFwdBelG s B_Seqlen B_Prompt_Cache_Len H BLOCK_M) idx
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (H : Nat) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (curBatch s H)
```
</details>

<details><summary><code>mIndex</code></summary>

```lean
def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>curHead</code></summary>

```lean
def curHead (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H
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
def seqLen
    (s : BlockState) (H : Nat) (B_Seqlen B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (curBatch s H) -
    promptLen s H B_Prompt_Cache_Len
```
</details>

<details><summary><code>contextAttnExactFoldMG</code></summary>

```
/-- **General faithful kernel value** at output lane `(i, d)`: `acc/l` of the
⊥-seeded online-softmax fold over `ctxKVMG` for the full streamed window `[0, S)`.
A pure function of `Q`/`K`/`V` memory. -/
```
```lean
noncomputable def contextAttnExactFoldMG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName)
    (sm_scale : ℝ)
    (H stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd BLOCK_DMODEL BLOCK_M S bel : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  let st := gStateBot S S (ctxKVMG s Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len sm_scale
      H stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd BLOCK_DMODEL BLOCK_M S bel idx.1 idx.2.1.val)
  st.2.2 / st.2.1
```
</details>

<details><summary><code>ctxFwdWindowG</code></summary>

```
/-- General kernel-decoded streamed window `S = ceil_{BN}(block_mask·block_end_loc)`. -/
```
```lean
def ctxFwdWindowG (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (H BLOCK_M BLOCK_N : Nat) : Nat :=
  let plen := s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / H)
  let sl := s.readMemValue .nat B_Seqlen (s.pids 1 / H) - plen
  let bel := ctxFwdBelG s B_Seqlen B_Prompt_Cache_Len H BLOCK_M
  let bm := if BLOCK_M * s.pids 0 < sl then 1 else 0
  BLOCK_N * ((bm * bel + (BLOCK_N - 1)) / BLOCK_N)
```
</details>

<details><summary><code>ctxFwdBelG</code></summary>

```
/-- General kernel-decoded `block_end_loc = min(BM·start_m + BM + plen, seq_len + plen)`. -/
```
```lean
def ctxFwdBelG (s : BlockState) (B_Seqlen B_Prompt_Cache_Len : RegionName) (H BLOCK_M : Nat) : Nat :=
  let plen := s.readMemValue .nat B_Prompt_Cache_Len (s.pids 1 / H)
  let sl := s.readMemValue .nat B_Seqlen (s.pids 1 / H) - plen
  let a := BLOCK_M * s.pids 0 + BLOCK_M + plen
  let b := sl + plen
  if a < b then a else b
```
</details>

<details><summary><code>curBatch</code></summary>

```lean
def curBatch (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H
```
</details>

<details><summary><code>promptLen</code></summary>

```lean
def promptLen (s : BlockState) (H : Nat) (B_Prompt_Cache_Len : RegionName) : Nat :=
  s.readMemValue .nat B_Prompt_Cache_Len (curBatch s H)
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

<details><summary><code>ctxKVMG</code></summary>

```
/-- General faithful per-key `(score, value)` the loop folds, channel dim
`BLOCK_DMODEL`. Active causal lane (`j ≤ gi+plen`): `sm·Σ_{e<BLOCK_DMODEL}
ctxQTileMG(i,e)·ctxKTileMG(j,e)`; future lane: the `-1e8` sentinel; value:
`ctxVTileMG`. -/
```
```lean
noncomputable def ctxKVMG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen Req_to_tokens B_req_idx B_Prompt_Cache_Len : RegionName) (sm_scale : ℝ)
    (H stride_qbs stride_qh stride_qd stride_req_b stride_req_s stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd BLOCK_DMODEL BLOCK_M S bel : Nat)
    (i : Fin BLOCK_M) (d : Nat) (j : Fin S) : ℝ × ℝ :=
  (if j.val ≤ s.pids 0 * BLOCK_M + i.val + promptLen s H B_Prompt_Cache_Len then
      sm_scale * Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
        ctxQTileMG s Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len H stride_qbs stride_qh stride_qd BLOCK_M i e.val
          * ctxKTileMG s K Req_to_tokens B_req_idx H stride_req_b stride_req_s stride_kbs stride_kh stride_kd S bel j e.val)
    else (0.0 - 10e7 : ℝ),
    ctxVTileMG s V Req_to_tokens B_req_idx H stride_req_b stride_req_s stride_vbs stride_vh stride_vd S bel j d)
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
`α = realExp2(m ⊖ m')` is `0` on the first block — faithful to the kernel's
`m_i = tl.zeros − inf` and `l_i`/`acc = 0`. -/
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

<details><summary><code>ctxQTileMG</code></summary>

```
/-- General row-masked query tile: `ctxQTileG` for active rows
(`pids0·BLOCK_M + i < seq_len`), else `0`. -/
```
```lean
noncomputable def ctxQTileMG
    (s : BlockState) (Q B_Start_Loc B_Seqlen B_Prompt_Cache_Len : RegionName)
    (H stride_qbs stride_qh stride_qd BLOCK_M : Nat) (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  if s.pids 0 * BLOCK_M + i.val < seqLen s H B_Seqlen B_Prompt_Cache_Len then
    ctxQTileG s Q B_Start_Loc H stride_qbs stride_qh stride_qd BLOCK_M i e
  else 0
```
</details>

<details><summary><code>ctxKTileMG</code></summary>

```
/-- General `block_end_loc`-masked key tile: `ctxKTileG` for `j < bel`, else `0`. -/
```
```lean
noncomputable def ctxKTileMG (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (H stride_req_b stride_req_s stride_kbs stride_kh stride_kd S bel : Nat)
    (j : Fin S) (e : Nat) : ℝ :=
  if j.val < bel then ctxKTileG s K Req_to_tokens B_req_idx H stride_req_b stride_req_s stride_kbs stride_kh stride_kd S j e else 0
```
</details>

<details><summary><code>ctxVTileMG</code></summary>

```
/-- General `block_end_loc`-masked value tile: `ctxVTileG` for `j < bel`, else `0`. -/
```
```lean
noncomputable def ctxVTileMG (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (H stride_req_b stride_req_s stride_vbs stride_vh stride_vd S bel : Nat)
    (j : Fin S) (d : Nat) : ℝ :=
  if j.val < bel then ctxVTileG s V Req_to_tokens B_req_idx H stride_req_b stride_req_s stride_vbs stride_vh stride_vd S j d else 0
```
</details>

<details><summary><code>ctxQTileG</code></summary>

```
/-- General coordinate-faithful query tile `Q[gi, e]` at head/stride parameters
(`gi = pids0·BLOCK_M + i`, offset by `cur_batch_in_all_start_index`). -/
```
```lean
noncomputable def ctxQTileG
    (s : BlockState) (Q B_Start_Loc : RegionName)
    (H stride_qbs stride_qh stride_qd BLOCK_M : Nat)
    (i : Fin BLOCK_M) (e : Nat) : ℝ :=
  s.readMem Q
    ((s.readMemValue .nat B_Start_Loc (curBatch s H) + (s.pids 0 * BLOCK_M + i.val))
        * stride_qbs + curHead s H * stride_qh + e * stride_qd)
```
</details>

<details><summary><code>ctxKTileG</code></summary>

```
/-- General coordinate-faithful key tile `K[kv_loc(j), cur_head, e]`
(`kv_group_num = 1` so `cur_kv_head = cur_head`), the gather reading physical
token `kv_loc(j)`. -/
```
```lean
noncomputable def ctxKTileG (s : BlockState) (K Req_to_tokens B_req_idx : RegionName)
    (H stride_req_b stride_req_s stride_kbs stride_kh stride_kd S : Nat)
    (j : Fin S) (e : Nat) : ℝ :=
  s.readMem K (ctxKvLocG s Req_to_tokens B_req_idx H stride_req_b stride_req_s j.val * stride_kbs
    + curHead s H * stride_kh + e * stride_kd)
```
</details>

<details><summary><code>ctxVTileG</code></summary>

```
/-- General coordinate-faithful value tile `V[kv_loc(j), cur_head, d]`. -/
```
```lean
noncomputable def ctxVTileG (s : BlockState) (V Req_to_tokens B_req_idx : RegionName)
    (H stride_req_b stride_req_s stride_vbs stride_vh stride_vd S : Nat)
    (j : Fin S) (d : Nat) : ℝ :=
  s.readMem V (ctxKvLocG s Req_to_tokens B_req_idx H stride_req_b stride_req_s j.val * stride_vbs
    + curHead s H * stride_vh + d * stride_vd)
```
</details>

<details><summary><code>ctxKvLocG</code></summary>

```
/-- General `Req_to_tokens` gather: physical token slot for streamed key index `j`,
gather strides free: `kv_loc(j) = Req_to_tokens[stride_req_b·req_idx + stride_req_s·j]`,
`req_idx = B_req_idx[cur_batch]`. -/
```
```lean
def ctxKvLocG
    (s : BlockState) (Req_to_tokens B_req_idx : RegionName)
    (H stride_req_b stride_req_s j : Nat) : Nat :=
  s.readMemValue .nat Req_to_tokens
    (stride_req_b * s.readMemValue .nat B_req_idx (curBatch s H) + stride_req_s * j)
```
</details>
