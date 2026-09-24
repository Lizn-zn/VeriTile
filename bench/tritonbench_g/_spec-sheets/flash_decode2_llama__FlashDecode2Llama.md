# Spec sheet — `bench/tritonbench_g/flash_decode2_llama/FlashDecode2Llama.lean`

**Python source:** `bench/tritonbench_g/flash_decode2_llama/flash_decode2_llama.py`

## Public theorem: `flash_decode2_llama_normalization_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General genuine final-normalization output summary.** Fully
dimension-parameterized over every stride and `BLOCK_DMODEL`: the full LLaMA
flash-decode stage2 surface lowers, and the normalization writeback *realizes*
the genuine closed form `O[d] = acc[d] / sum_exp` (`normalizedStoreValue`,
reading the loop-produced `Acc`/`SumExp` input memory) at each output lane. No
hardcoded shape literals. Side condition: output-footprint injectivity. -/
```
</details>

**Statement:**
```lean
specification flash_decode2_llama_normalization_output_summary_general
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Acc SumExp O : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      BLOCK_SEQ BLOCK_DMODEL
      stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    (∃ alg, (flash_decode2_llama_surface B_Seqlen Mid_O Mid_O_LogExpSum O
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od BLOCK_SEQ
      BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_decode2_llama_normalization_store_kernel Acc SumExp O
        stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
        stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := fun i : Fin BLOCK_DMODEL =>
        some (O, outOffset s stride_obs stride_oh stride_od i))
      (expected := fun i : Fin BLOCK_DMODEL =>
        normalizedStoreValue s Acc SumExp stride_acc_b stride_acc_h
          stride_acc_d stride_sum_b stride_sum_h i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)`

**Closed-form spec defs (transitive):** `outOffset`, `flash_decode2_llama_surface`, `flash_decode2_llama_normalization_store_kernel`, `normalizedStoreValue`, `dIndex`, `accOffset`, `sumExpOffset`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od
```
</details>

<details><summary><code>flash_decode2_llama_surface</code></summary>

```
/-- Faithful transcription of `flash_decode2_llama.py`'s
`_fwd_kernel_flash_decode_stage2`.

The Python body passes but does not use `stride_mid_od`, `stride_mid_o_es`, or
`stride_od`; this surface preserves that addressing behavior and keeps those
signature positions underscored. -/
```
```lean
def flash_decode2_llama_surface
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum O : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os _stride_mid_od
      stride_mid_o_eb stride_mid_o_eh _stride_mid_o_es stride_obs stride_oh _stride_od
      BLOCK_SEQ BLOCK_DMODEL : Nat) : ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  block_n_size = tl.where(cur_batch_seq_len <= $(0), $(0),
    cur_batch_seq_len + $(BLOCK_SEQ) - $(1)) // $(BLOCK_SEQ)
  sum_exp = 0.0
  max_logic = -inf
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  offs_v = cur_batch * $(stride_mid_ob) + cur_head * $(stride_mid_oh) + offs_d
  offs_logic = cur_batch * $(stride_mid_o_eb) + cur_head * $(stride_mid_o_eh)
  for block_seq_n in range($(0), block_n_size, $(1)) {
    tv = tl.load(Mid_O + offs_v + block_seq_n * $(stride_mid_os))
    tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
    new_max_logic = tl.maximum(tlogic, max_logic)
    old_scale = tl.exp(max_logic - new_max_logic)
    acc *= old_scale
    exp_logic = tl.exp(tlogic - new_max_logic)
    acc += exp_logic * tv
    sum_exp = sum_exp * old_scale + exp_logic
    max_logic = new_max_logic
  }
  tl.store(O + cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d,
    acc / sum_exp)
}
```
</details>

<details><summary><code>flash_decode2_llama_normalization_store_kernel</code></summary>

```
/-- Final normalization and output writeback for LLaMA flash-decode stage2.

This kernel consumes the loop-produced accumulator vector and `sum_exp` scalar,
computes `acc / sum_exp`, and writes the Python-observable `O` row. -/
```
```lean
def flash_decode2_llama_normalization_store_kernel
    (Acc SumExp O : RegionName)
    (stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d))
  sum_exp = tl.load(SumExp + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h))
  tl.store(O + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), acc / sum_exp)
}
```
</details>

<details><summary><code>normalizedStoreValue</code></summary>

```lean
noncomputable def normalizedStoreValue
    (s : BlockState) (Acc SumExp : RegionName)
    (stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  s.readMem Acc (accOffset s stride_acc_b stride_acc_h stride_acc_d i) /
    s.readMem SumExp (sumExpOffset s stride_sum_b stride_sum_h)
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val
```
</details>

<details><summary><code>accOffset</code></summary>

```lean
def accOffset
    (s : BlockState)
    (stride_acc_b stride_acc_h stride_acc_d : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
    dIndex s i * stride_acc_d
```
</details>

<details><summary><code>sumExpOffset</code></summary>

```lean
def sumExpOffset
    (s : BlockState)
    (stride_sum_b stride_sum_h : Nat) : Nat :=
  s.pids 0 * stride_sum_b + s.pids 1 * stride_sum_h
```
</details>

## Public theorem: `flash_decode2_llama_running_max_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General genuine running-maximum recurrence summary.** Fully
dimension-parameterized over the block index and every stride: the full LLaMA
flash-decode stage2 surface lowers, and one recurrence step *realizes* the
genuine closed form `new_max_logic = max(tlogic, max_logic)`
(`runningMaxStepValue`, reading `Mid_O_LogExpSum`/`MaxLogic` input memory). No
hardcoded shape literals. -/
```
</details>

**Statement:**
```lean
specification flash_decode2_llama_running_max_output_summary_general
    (B_Seqlen : Region .nat)
    (Mid_O Mid_O_LogExpSum MaxLogic NewMaxLogic O : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb' stride_mid_o_eh' stride_mid_o_es stride_obs stride_oh stride_od
      BLOCK_SEQ BLOCK_DMODEL
      block_seq_n stride_mid_o_eb stride_mid_o_eh stride_logic_b stride_logic_h : Nat)
    (s : BlockState) :
    (∃ alg, (flash_decode2_llama_surface B_Seqlen Mid_O Mid_O_LogExpSum O
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb'
      stride_mid_o_eh' stride_mid_o_es stride_obs stride_oh stride_od BLOCK_SEQ
      BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_decode2_llama_running_max_step_kernel Mid_O_LogExpSum
        MaxLogic NewMaxLogic block_seq_n stride_mid_o_eb stride_mid_o_eh
        stride_logic_b stride_logic_h)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar NewMaxLogic
        (sumExpOffset s stride_logic_b stride_logic_h))
      (expected := fun _ : PUnit =>
        runningMaxStepValue s Mid_O_LogExpSum MaxLogic block_seq_n
          stride_mid_o_eb stride_mid_o_eh stride_logic_b stride_logic_h)
```

**Closed-form spec defs (transitive):** `flash_decode2_llama_surface`, `flash_decode2_llama_running_max_step_kernel`, `sumExpOffset`, `runningMaxStepValue`, `logicOffset`

<details><summary><code>flash_decode2_llama_surface</code></summary>

```
/-- Faithful transcription of `flash_decode2_llama.py`'s
`_fwd_kernel_flash_decode_stage2`.

The Python body passes but does not use `stride_mid_od`, `stride_mid_o_es`, or
`stride_od`; this surface preserves that addressing behavior and keeps those
signature positions underscored. -/
```
```lean
def flash_decode2_llama_surface
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum O : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os _stride_mid_od
      stride_mid_o_eb stride_mid_o_eh _stride_mid_o_es stride_obs stride_oh _stride_od
      BLOCK_SEQ BLOCK_DMODEL : Nat) : ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  block_n_size = tl.where(cur_batch_seq_len <= $(0), $(0),
    cur_batch_seq_len + $(BLOCK_SEQ) - $(1)) // $(BLOCK_SEQ)
  sum_exp = 0.0
  max_logic = -inf
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  offs_v = cur_batch * $(stride_mid_ob) + cur_head * $(stride_mid_oh) + offs_d
  offs_logic = cur_batch * $(stride_mid_o_eb) + cur_head * $(stride_mid_o_eh)
  for block_seq_n in range($(0), block_n_size, $(1)) {
    tv = tl.load(Mid_O + offs_v + block_seq_n * $(stride_mid_os))
    tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
    new_max_logic = tl.maximum(tlogic, max_logic)
    old_scale = tl.exp(max_logic - new_max_logic)
    acc *= old_scale
    exp_logic = tl.exp(tlogic - new_max_logic)
    acc += exp_logic * tv
    sum_exp = sum_exp * old_scale + exp_logic
    max_logic = new_max_logic
  }
  tl.store(O + cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d,
    acc / sum_exp)
}
```
</details>

<details><summary><code>flash_decode2_llama_running_max_step_kernel</code></summary>

```
/-- One LLaMA stage2 running-maximum recurrence step. -/
```
```lean
def flash_decode2_llama_running_max_step_kernel
    (Mid_O_LogExpSum MaxLogic NewMaxLogic : RegionName)
    (block_seq_n
      stride_mid_o_eb stride_mid_o_eh
      stride_logic_b stride_logic_h : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  tlogic = tl.load(Mid_O_LogExpSum + cur_batch * $(stride_mid_o_eb) +
      cur_head * $(stride_mid_o_eh) + $(block_seq_n))
  max_logic = tl.load(MaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  new_max_logic = tl.maximum(tlogic, max_logic)
  tl.store(NewMaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h), new_max_logic)
}
```
</details>

<details><summary><code>sumExpOffset</code></summary>

```lean
def sumExpOffset
    (s : BlockState)
    (stride_sum_b stride_sum_h : Nat) : Nat :=
  s.pids 0 * stride_sum_b + s.pids 1 * stride_sum_h
```
</details>

<details><summary><code>runningMaxStepValue</code></summary>

```lean
noncomputable def runningMaxStepValue
    (s : BlockState) (Mid_O_LogExpSum MaxLogic : RegionName)
    (block_seq_n stride_mid_o_eb stride_mid_o_eh
      stride_logic_b stride_logic_h : Nat) : ℝ :=
  let tlogic := s.readMem Mid_O_LogExpSum
    (logicOffset s block_seq_n stride_mid_o_eb stride_mid_o_eh)
  let maxLogic := s.readMem MaxLogic (sumExpOffset s stride_logic_b stride_logic_h)
  if tlogic > maxLogic then tlogic else maxLogic
```
</details>

<details><summary><code>logicOffset</code></summary>

```lean
def logicOffset
    (s : BlockState)
    (block_seq_n stride_mid_o_eb stride_mid_o_eh : Nat) : Nat :=
  s.pids 0 * stride_mid_o_eb + s.pids 1 * stride_mid_o_eh + block_seq_n
```
</details>

## Public theorem: `flash_decode2_llama_final_store_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `flash_decode2_llama.py`'s final
writeback: for every disjoint flat placement of `Final` / `O`, every program
coordinate whose lanes are in bounds, and every launch state whose `Final` row holds
`xs`, the translated pointer kernel terminates, every lane of the `O` row holds
`xs i`, and every other memory cell is unchanged.

Both windows are built from *two* program axes with different strides on the two
buffers — the shape the three-axis tile skin exists for. Dimension-general in all six
strides and `BLOCK_DMODEL`. Honest side-condition: output-address injectivity at every
program coordinate, the same hypothesis the per-write-map summary takes. -/
```
</details>

**Statement:**
```lean
specification flash_decode2_llama_final_store_io_correctness (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_DMODEL =>
        p₀ * stride_obs + p₁ * stride_oh + i.val * stride_od)) :
    finalStoreIO Final O stride_final_b stride_final_h stride_final_d stride_obs
        stride_oh stride_od BLOCK_DMODEL
      ⊨ fun _p₀ _p₁ xs i => xs i
```

**Assumptions / layout contracts:**
- `hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_DMODEL =>
        p₀ * stride_obs + p₁ * stride_oh + i.val * stride_od)`

**Closed-form spec defs (transitive):** `finalStoreIO`, `flash_decode2_llama_final_store_slice`

<details><summary><code>finalStoreIO</code></summary>

```
/-- IO signature of the final writeback on the three-axis tile surface. -/
```
```lean
def finalStoreIO (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat) : Masked3DTileKernelIO₁ where
  kernel := flash_decode2_llama_final_store_slice Final O stride_final_b
    stride_final_h stride_final_d stride_obs stride_oh stride_od BLOCK_DMODEL
  inp := Final
  out := O
  shape := [BLOCK_DMODEL]
  read := fun p₀ p₁ _p₂ i =>
    p₀ * stride_final_b + p₁ * stride_final_h + i.1.val * stride_final_d
  write := fun p₀ p₁ _p₂ i =>
    p₀ * stride_obs + p₁ * stride_oh + i.1.val * stride_od
  mask := fun _p₀ _p₁ _p₂ _ => True
```
</details>

<details><summary><code>flash_decode2_llama_final_store_slice</code></summary>

```
/-- Proof-oriented final output-store slice of
`flash_decode2_llama.py`'s `_fwd_kernel_flash_decode_stage2`.

The full stage2 kernel reduces per-sequence-block partial outputs into a final
`acc / sum_exp` vector. This slice starts from a precomputed normalized `Final`
vector and proves the unmasked writeback into `O`. -/
```
```lean
def flash_decode2_llama_final_store_slice
    (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  final = tl.load(Final + cur_batch * $(stride_final_b) +
      cur_head * $(stride_final_h) + offs_d * $(stride_final_d))
  tl.store(O + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), final)
}
```
</details>

## Public theorem: `flash_decode2_llama_final_store_io_correctnessR`

<details><summary>docstring</summary>

```
/-- **The `⊨[R]` headline** for `flash_decode2_llama.py`'s final writeback: for
**every** rounding model `R`, the same Hoare triple as
`flash_decode2_llama_final_store_io_correctness`, but run under `execR R` and read
back as `.real`-typed cells holding `R.round .real (xs i)`.

The writeback is a pure copy and carries no `.to(...)`, so the slice is cast-free
and the exact run transports verbatim. The content of the rounding face here is
exactly that: *this kernel introduces no rounding event of its own*, at any
`R`. -/
```
</details>

**Statement:**
```lean
specification flash_decode2_llama_final_store_io_correctnessR (R : RoundingModel)
    (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_DMODEL =>
        p₀ * stride_obs + p₁ * stride_oh + i.val * stride_od)) :
    finalStoreIO Final O stride_final_b stride_final_h stride_final_d stride_obs
        stride_oh stride_od BLOCK_DMODEL
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs i => xs i
```

**Assumptions / layout contracts:**
- `hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun i : Fin BLOCK_DMODEL =>
        p₀ * stride_obs + p₁ * stride_oh + i.val * stride_od)`

**Closed-form spec defs (transitive):** `finalStoreIO`, `flash_decode2_llama_final_store_slice`

<details><summary><code>finalStoreIO</code></summary>

```
/-- IO signature of the final writeback on the three-axis tile surface. -/
```
```lean
def finalStoreIO (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d stride_obs stride_oh
      stride_od BLOCK_DMODEL : Nat) : Masked3DTileKernelIO₁ where
  kernel := flash_decode2_llama_final_store_slice Final O stride_final_b
    stride_final_h stride_final_d stride_obs stride_oh stride_od BLOCK_DMODEL
  inp := Final
  out := O
  shape := [BLOCK_DMODEL]
  read := fun p₀ p₁ _p₂ i =>
    p₀ * stride_final_b + p₁ * stride_final_h + i.1.val * stride_final_d
  write := fun p₀ p₁ _p₂ i =>
    p₀ * stride_obs + p₁ * stride_oh + i.1.val * stride_od
  mask := fun _p₀ _p₁ _p₂ _ => True
```
</details>

<details><summary><code>flash_decode2_llama_final_store_slice</code></summary>

```
/-- Proof-oriented final output-store slice of
`flash_decode2_llama.py`'s `_fwd_kernel_flash_decode_stage2`.

The full stage2 kernel reduces per-sequence-block partial outputs into a final
`acc / sum_exp` vector. This slice starts from a precomputed normalized `Final`
vector and proves the unmasked writeback into `O`. -/
```
```lean
def flash_decode2_llama_final_store_slice
    (Final O : RegionName)
    (stride_final_b stride_final_h stride_final_d
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  final = tl.load(Final + cur_batch * $(stride_final_b) +
      cur_head * $(stride_final_h) + offs_d * $(stride_final_d))
  tl.store(O + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), final)
}
```
</details>

## Also present (pinned special-case summaries)
- `flash_decode2_llama_final_store_slice_compute_correct`
- `flash_decode2_llama_normalization_store_kernel_compute_correct`
- `flash_decode2_llama_running_max_step_kernel_compute_correct`
