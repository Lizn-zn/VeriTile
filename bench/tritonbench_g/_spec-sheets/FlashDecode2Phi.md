# Spec sheet — `bench/tritonbench_g/flash_decode2_phi/FlashDecode2Phi.lean`

**Python source:** `bench/tritonbench_g/flash_decode2_phi/flash_decode2_phi.py`

## Public theorem: `flash_decode2_phi_masked_accumulator_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General genuine masked accumulator/`sum_exp` recurrence summary.** Fully
dimension-parameterized over the block index, `head_dim`, `BLOCK_SEQ`,
`BLOCK_DMODEL`, and every stride: the full Phi flash-decode stage2 surface
lowers, and one loop-body step *realizes* both genuine closed forms — the masked
vector accumulator update `accumulatorStepValue` (=
`acc * old_scale + exp_logic * tv` for `offs_d < head_dim`) and the scalar
`sumExpStepValue` (= `sum_exp * old_scale + exp_logic`), both reading
`Mid_O`/`Mid_O_LogExpSum`/`AccIn`/`SumExpIn`/`MaxLogic`/`NewMaxLogic` input
memory. No hardcoded shape literals. Side condition: accumulator-footprint
injectivity. -/
```
</details>

**Statement:**
```lean
specification flash_decode2_phi_masked_accumulator_output_summary_general
    (B_Seqlen : Region .nat)
    (Mid_O Mid_O_LogExpSum AccIn SumExpIn MaxLogic NewMaxLogic AccOut SumExpOut
      Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL
      block_seq_n stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h stride_logic_b stride_logic_h : Nat)
    (s : BlockState)
    (hAccOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL =>
        accOffset s stride_acc_b stride_acc_h stride_acc_d i)) :
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od head_dim
      BLOCK_SEQ BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_decode2_phi_accumulator_step_kernel Mid_O Mid_O_LogExpSum
        AccIn MaxLogic NewMaxLogic AccOut block_seq_n head_dim stride_mid_ob
        stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb stride_mid_o_eh
        stride_acc_b stride_acc_h stride_acc_d stride_logic_b stride_logic_h
        BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_DMODEL => active s head_dim i)
        (fun i => (AccOut, accOffset s stride_acc_b stride_acc_h stride_acc_d i)))
      (expected := fun i : Fin BLOCK_DMODEL =>
        accumulatorStepValue s Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic
          block_seq_n head_dim stride_mid_ob stride_mid_oh stride_mid_os
          stride_mid_od stride_mid_o_eb stride_mid_o_eh stride_acc_b stride_acc_h
          stride_acc_d stride_logic_b stride_logic_h i) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_decode2_phi_sum_exp_step_kernel Mid_O_LogExpSum SumExpIn
        MaxLogic NewMaxLogic SumExpOut block_seq_n stride_mid_o_eb stride_mid_o_eh
        stride_sum_b stride_sum_h stride_logic_b stride_logic_h)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar SumExpOut
        (sumExpOffset s stride_sum_b stride_sum_h))
      (expected := fun _ : PUnit =>
        sumExpStepValue s Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic
          block_seq_n stride_mid_o_eb stride_mid_o_eh stride_sum_b stride_sum_h
          stride_logic_b stride_logic_h)
```

**Assumptions / layout contracts:**
- `hAccOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL =>
        accOffset s stride_acc_b stride_acc_h stride_acc_d i)`
- `fun i : Fin BLOCK_DMODEL => active s head_dim i`

**Closed-form spec defs (transitive):** `accOffset`, `flash_decode2_phi_surface`, `flash_decode2_phi_accumulator_step_kernel`, `active`, `accumulatorStepValue`, `flash_decode2_phi_sum_exp_step_kernel`, `sumExpOffset`, `sumExpStepValue`, `dIndex`, `oldScaleValue`, `expLogicValue`, `midOffset`, `logicOffset`

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

<details><summary><code>flash_decode2_phi_surface</code></summary>

```
/-- Faithful transcription of `flash_decode2_phi.py`'s
`_fwd_kernel_flash_decode_stage2`.

The Python body passes but does not use `stride_mid_od`, `stride_mid_o_es`, or
`stride_od`; this surface preserves that addressing behavior and keeps those
signature positions underscored. -/
```
```lean
def flash_decode2_phi_surface
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os _stride_mid_od
      stride_mid_o_eb stride_mid_o_eh _stride_mid_o_es stride_obs stride_oh _stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL : Nat) : ComputeKernel := triton {
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
    tv = tl.load(Mid_O + offs_v + block_seq_n * $(stride_mid_os),
      mask=offs_d < $(head_dim), other=0.0)
    tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
    new_max_logic = tl.maximum(tlogic, max_logic)
    old_scale = tl.exp(max_logic - new_max_logic)
    acc *= old_scale
    exp_logic = tl.exp(tlogic - new_max_logic)
    acc += exp_logic * tv
    sum_exp = sum_exp * old_scale + exp_logic
    max_logic = new_max_logic
  }
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d,
    acc / sum_exp, mask=offs_d < $(head_dim))
}
```
</details>

<details><summary><code>flash_decode2_phi_accumulator_step_kernel</code></summary>

```
/-- One Phi stage2 accumulator recurrence step.

This kernel models the loop-body arithmetic after the next running maximum has
already been supplied: it loads the masked `Mid_O` row, applies `old_scale` and
`exp_logic`, and writes the updated accumulator vector. -/
```
```lean
def flash_decode2_phi_accumulator_step_kernel
    (Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic AccOut : RegionName)
    (block_seq_n
      head_dim
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh
      stride_acc_b stride_acc_h stride_acc_d
      stride_logic_b stride_logic_h
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = offs_d < $(head_dim)
  tv = tl.load(Mid_O + cur_batch * $(stride_mid_ob) +
      cur_head * $(stride_mid_oh) + $(block_seq_n) * $(stride_mid_os) +
      offs_d * $(stride_mid_od), mask=mask, other=0.0)
  tlogic = tl.load(Mid_O_LogExpSum + cur_batch * $(stride_mid_o_eb) +
      cur_head * $(stride_mid_o_eh) + $(block_seq_n))
  acc = tl.load(AccIn + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d),
      mask=mask, other=0.0)
  max_logic = tl.load(MaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  new_max_logic = tl.load(NewMaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  old_scale = tl.exp(max_logic - new_max_logic)
  acc = acc * old_scale
  exp_logic = tl.exp(tlogic - new_max_logic)
  acc = acc + exp_logic * tv
  tl.store(AccOut + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d), acc, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (_s : BlockState) (head_dim : Nat) (i : Fin BLOCK_DMODEL) : Prop :=
  i.val < head_dim
```
</details>

<details><summary><code>accumulatorStepValue</code></summary>

```lean
noncomputable def accumulatorStepValue
    (s : BlockState) (Mid_O Mid_O_LogExpSum AccIn MaxLogic NewMaxLogic : RegionName)
    (block_seq_n head_dim
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh
      stride_acc_b stride_acc_h stride_acc_d
      stride_logic_b stride_logic_h : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  WithBot.unbotD 0
    (if active s head_dim i then
      some (s.readMem AccIn (accOffset s stride_acc_b stride_acc_h stride_acc_d i) *
          oldScaleValue s MaxLogic NewMaxLogic stride_logic_b stride_logic_h +
        expLogicValue s Mid_O_LogExpSum NewMaxLogic block_seq_n stride_mid_o_eb
          stride_mid_o_eh stride_logic_b stride_logic_h *
        s.readMem Mid_O
          (midOffset s block_seq_n stride_mid_ob stride_mid_oh stride_mid_os
            stride_mid_od i))
    else some (0.0 : ℝ))
```
</details>

<details><summary><code>flash_decode2_phi_sum_exp_step_kernel</code></summary>

```
/-- Scalar companion for the same Phi stage2 recurrence step.

It proves the `sum_exp = sum_exp * old_scale + exp_logic` update with the same
symbolic stride addressing as the vector accumulator step. -/
```
```lean
def flash_decode2_phi_sum_exp_step_kernel
    (Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic SumExpOut : RegionName)
    (block_seq_n
      stride_mid_o_eb stride_mid_o_eh
      stride_sum_b stride_sum_h
      stride_logic_b stride_logic_h : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  tlogic = tl.load(Mid_O_LogExpSum + cur_batch * $(stride_mid_o_eb) +
      cur_head * $(stride_mid_o_eh) + $(block_seq_n))
  sum_exp = tl.load(SumExpIn + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h))
  max_logic = tl.load(MaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  new_max_logic = tl.load(NewMaxLogic + cur_batch * $(stride_logic_b) +
      cur_head * $(stride_logic_h))
  old_scale = tl.exp(max_logic - new_max_logic)
  exp_logic = tl.exp(tlogic - new_max_logic)
  sum_exp = sum_exp * old_scale + exp_logic
  tl.store(SumExpOut + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h), sum_exp)
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

<details><summary><code>sumExpStepValue</code></summary>

```lean
noncomputable def sumExpStepValue
    (s : BlockState) (Mid_O_LogExpSum SumExpIn MaxLogic NewMaxLogic : RegionName)
    (block_seq_n stride_mid_o_eb stride_mid_o_eh
      stride_sum_b stride_sum_h stride_logic_b stride_logic_h : Nat) : ℝ :=
  s.readMem SumExpIn (sumExpOffset s stride_sum_b stride_sum_h) *
      oldScaleValue s MaxLogic NewMaxLogic stride_logic_b stride_logic_h +
    expLogicValue s Mid_O_LogExpSum NewMaxLogic block_seq_n stride_mid_o_eb
      stride_mid_o_eh stride_logic_b stride_logic_h
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (_s : BlockState) (i : Fin BLOCK_DMODEL) : Nat :=
  i.val
```
</details>

<details><summary><code>oldScaleValue</code></summary>

```lean
noncomputable def oldScaleValue
    (s : BlockState) (MaxLogic NewMaxLogic : RegionName)
    (stride_logic_b stride_logic_h : Nat) : ℝ :=
  Real.exp (s.readMem MaxLogic (sumExpOffset s stride_logic_b stride_logic_h) -
    s.readMem NewMaxLogic (sumExpOffset s stride_logic_b stride_logic_h))
```
</details>

<details><summary><code>expLogicValue</code></summary>

```lean
noncomputable def expLogicValue
    (s : BlockState) (Mid_O_LogExpSum NewMaxLogic : RegionName)
    (block_seq_n stride_mid_o_eb stride_mid_o_eh
      stride_logic_b stride_logic_h : Nat) : ℝ :=
  Real.exp (s.readMem Mid_O_LogExpSum
      (logicOffset s block_seq_n stride_mid_o_eb stride_mid_o_eh) -
    s.readMem NewMaxLogic (sumExpOffset s stride_logic_b stride_logic_h))
```
</details>

<details><summary><code>midOffset</code></summary>

```lean
def midOffset
    (s : BlockState)
    (block_seq_n stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_mid_ob + s.pids 1 * stride_mid_oh +
    block_seq_n * stride_mid_os + dIndex s i * stride_mid_od
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

## Public theorem: `flash_decode2_phi_running_max_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General genuine running-maximum recurrence summary.** Fully
dimension-parameterized over every stride, `head_dim`, `BLOCK_SEQ`, and
`BLOCK_DMODEL`: the full Phi flash-decode stage2 surface lowers, and the dynamic
`block_n_size` `max_logic` loop invariant is the genuine recursive value
`runningMaxAfter` — it starts at `-inf` and each iteration joins the next
`Mid_O_LogExpSum` element at contiguous `offs_logic + block_seq_n` (reading
input memory). No hardcoded shape literals. -/
```
</details>

**Statement:**
```lean
specification flash_decode2_phi_running_max_output_summary_general
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL : Nat)
    (s : BlockState) (block_n_size : Nat) :
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od head_dim
      BLOCK_SEQ BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh 0 = ⊥ ∧
    (∀ k, k < block_n_size →
      runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh (k + 1) =
        runningMaxJoin
          (some (s.readMem Mid_O_LogExpSum
            (runningMaxLogicOffset s stride_mid_o_eb stride_mid_o_eh k)) :
            WithBot ℝ)
          (runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh k))
```

**Closed-form spec defs (transitive):** `flash_decode2_phi_surface`, `runningMaxAfter`, `runningMaxJoin`, `runningMaxLogicOffset`

<details><summary><code>flash_decode2_phi_surface</code></summary>

```
/-- Faithful transcription of `flash_decode2_phi.py`'s
`_fwd_kernel_flash_decode_stage2`.

The Python body passes but does not use `stride_mid_od`, `stride_mid_o_es`, or
`stride_od`; this surface preserves that addressing behavior and keeps those
signature positions underscored. -/
```
```lean
def flash_decode2_phi_surface
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os _stride_mid_od
      stride_mid_o_eb stride_mid_o_eh _stride_mid_o_es stride_obs stride_oh _stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL : Nat) : ComputeKernel := triton {
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
    tv = tl.load(Mid_O + offs_v + block_seq_n * $(stride_mid_os),
      mask=offs_d < $(head_dim), other=0.0)
    tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
    new_max_logic = tl.maximum(tlogic, max_logic)
    old_scale = tl.exp(max_logic - new_max_logic)
    acc *= old_scale
    exp_logic = tl.exp(tlogic - new_max_logic)
    acc += exp_logic * tv
    sum_exp = sum_exp * old_scale + exp_logic
    max_logic = new_max_logic
  }
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d,
    acc / sum_exp, mask=offs_d < $(head_dim))
}
```
</details>

<details><summary><code>runningMaxAfter</code></summary>

```lean
noncomputable def runningMaxAfter
    (s : BlockState) (Mid_O_LogExpSum : RegionName)
    (stride_mid_o_eb stride_mid_o_eh : Nat) : Nat → WithBot ℝ
  | 0 => ⊥
  | k + 1 =>
      runningMaxJoin
        (some (s.readMem Mid_O_LogExpSum
          (runningMaxLogicOffset s stride_mid_o_eb stride_mid_o_eh k)) : WithBot ℝ)
        (runningMaxAfter s Mid_O_LogExpSum stride_mid_o_eb stride_mid_o_eh k)
```
</details>

<details><summary><code>runningMaxJoin</code></summary>

```lean
noncomputable def runningMaxJoin (x y : WithBot ℝ) : WithBot ℝ :=
  max x y
```
</details>

<details><summary><code>runningMaxLogicOffset</code></summary>

```lean
def runningMaxLogicOffset
    (s : BlockState) (stride_mid_o_eb stride_mid_o_eh block_seq_n : Nat) :
    Nat :=
  s.pids 0 * stride_mid_o_eb + s.pids 1 * stride_mid_o_eh + block_seq_n
```
</details>

## Public theorem: `flash_decode2_phi_normalization_output_summary_general`

<details><summary>docstring</summary>

```
/-- **General genuine final-normalization output summary.** Fully
dimension-parameterized over every stride, `head_dim`, `BLOCK_SEQ`, and
`BLOCK_DMODEL`: the full Phi flash-decode stage2 surface lowers, and the masked
normalization writeback *realizes* the genuine closed form
`Out[d] = acc[d] / sum_exp` for `offs_d < head_dim` (`normalizedStoreValue`,
reading the loop-produced `Acc`/`SumExp` input memory). No hardcoded shape
literals. Side condition: output-footprint injectivity.

This is the kernel's headline output store; the running-max / accumulator faces
above are the supporting loop-invariant faces (the full loop is not unrolled into
one closed form — see the module Scope note). -/
```
</details>

**Statement:**
```lean
specification flash_decode2_phi_normalization_output_summary_general
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Acc SumExp Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od
      stride_mid_o_eb stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL
      stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)) :
    (∃ alg, (flash_decode2_phi_surface B_Seqlen Mid_O Mid_O_LogExpSum Out
      stride_mid_ob stride_mid_oh stride_mid_os stride_mid_od stride_mid_o_eb
      stride_mid_o_eh stride_mid_o_es stride_obs stride_oh stride_od head_dim
      BLOCK_SEQ BLOCK_DMODEL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := flash_decode2_phi_normalization_store_kernel Acc SumExp Out
        head_dim stride_acc_b stride_acc_h stride_acc_d stride_sum_b stride_sum_h
        stride_obs stride_oh stride_od BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_DMODEL => active s head_dim i)
        (fun i => (Out, outOffset s stride_obs stride_oh stride_od i)))
      (expected := fun i : Fin BLOCK_DMODEL =>
        normalizedStoreValue s Acc SumExp head_dim stride_acc_b stride_acc_h
          stride_acc_d stride_sum_b stride_sum_h i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_obs stride_oh stride_od i)`
- `fun i : Fin BLOCK_DMODEL => active s head_dim i`

**Closed-form spec defs (transitive):** `outOffset`, `flash_decode2_phi_surface`, `flash_decode2_phi_normalization_store_kernel`, `active`, `normalizedStoreValue`, `dIndex`, `accOffset`, `sumExpOffset`

<details><summary><code>outOffset</code></summary>

```lean
def outOffset
    (s : BlockState)
    (stride_obs stride_oh stride_od : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  s.pids 0 * stride_obs + s.pids 1 * stride_oh + dIndex s i * stride_od
```
</details>

<details><summary><code>flash_decode2_phi_surface</code></summary>

```
/-- Faithful transcription of `flash_decode2_phi.py`'s
`_fwd_kernel_flash_decode_stage2`.

The Python body passes but does not use `stride_mid_od`, `stride_mid_o_es`, or
`stride_od`; this surface preserves that addressing behavior and keeps those
signature positions underscored. -/
```
```lean
def flash_decode2_phi_surface
    (B_Seqlen : Region .nat) (Mid_O Mid_O_LogExpSum Out : RegionName)
    (stride_mid_ob stride_mid_oh stride_mid_os _stride_mid_od
      stride_mid_o_eb stride_mid_o_eh _stride_mid_o_es stride_obs stride_oh _stride_od
      head_dim BLOCK_SEQ BLOCK_DMODEL : Nat) : ComputeKernel := triton {
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
    tv = tl.load(Mid_O + offs_v + block_seq_n * $(stride_mid_os),
      mask=offs_d < $(head_dim), other=0.0)
    tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
    new_max_logic = tl.maximum(tlogic, max_logic)
    old_scale = tl.exp(max_logic - new_max_logic)
    acc *= old_scale
    exp_logic = tl.exp(tlogic - new_max_logic)
    acc += exp_logic * tv
    sum_exp = sum_exp * old_scale + exp_logic
    max_logic = new_max_logic
  }
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d,
    acc / sum_exp, mask=offs_d < $(head_dim))
}
```
</details>

<details><summary><code>flash_decode2_phi_normalization_store_kernel</code></summary>

```
/-- Final normalization and output writeback for Phi flash-decode stage2.

This kernel consumes the loop-produced accumulator vector and `sum_exp` scalar,
computes `acc / sum_exp`, and writes the masked Python-observable `Out` row. -/
```
```lean
def flash_decode2_phi_normalization_store_kernel
    (Acc SumExp Out : RegionName)
    (head_dim
      stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h
      stride_obs stride_oh stride_od
      BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = offs_d < $(head_dim)
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) +
      cur_head * $(stride_acc_h) + offs_d * $(stride_acc_d),
      mask=mask, other=0.0)
  sum_exp = tl.load(SumExp + cur_batch * $(stride_sum_b) +
      cur_head * $(stride_sum_h))
  tl.store(Out + cur_batch * $(stride_obs) + cur_head * $(stride_oh) +
      offs_d * $(stride_od), acc / sum_exp, mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (_s : BlockState) (head_dim : Nat) (i : Fin BLOCK_DMODEL) : Prop :=
  i.val < head_dim
```
</details>

<details><summary><code>normalizedStoreValue</code></summary>

```lean
noncomputable def normalizedStoreValue
    (s : BlockState) (Acc SumExp : RegionName)
    (head_dim stride_acc_b stride_acc_h stride_acc_d
      stride_sum_b stride_sum_h : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  WithBot.unbotD 0
    (if active s head_dim i then
      some (s.readMem Acc (accOffset s stride_acc_b stride_acc_h stride_acc_d i) /
        s.readMem SumExp (sumExpOffset s stride_sum_b stride_sum_h))
    else some (0.0 : ℝ))
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

## Also present (pinned special-case summaries)
- `flash_decode2_phi_final_store_slice_compute_correct`
- `flash_decode2_phi_accumulator_step_kernel_compute_correct`
- `flash_decode2_phi_sum_exp_step_kernel_compute_correct`
- `flash_decode2_phi_normalization_store_kernel_compute_correct`
