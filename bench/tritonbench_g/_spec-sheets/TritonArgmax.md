# Spec sheet — `bench/tritonbench_g/triton_argmax/TritonArgmax.lean`

**Python source:** `bench/tritonbench_g/triton_argmax/triton_argmax.py`

## Public theorem: `argmax_kernel_1_value_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the value channel of `argmax_kernel_1`
(`INT64_INDEX=false` branch). The hypothesis `hRegions` rules out aliasing
between the real `mid_value` buffer and the int `mid_index` buffer at the same
offset; without it the `.nat` index store could overwrite the `.real` value
store. -/
```
</details>

**Statement:**
```lean
specification argmax_kernel_1_value_compute_correct
    (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : mid_value ≠ (Region.cast mid_index : RegionName)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := argmax_kernel_1 inp mid_value mid_index M BLOCK_SIZE Bool.false)
      (initialState := s)
      (write := fun _ : PUnit => some (mid_value, s.pid))
      (expected := fun _ => argmaxKernel1ValueSpec s inp M BLOCK_SIZE)
```

**Assumptions / layout contracts:**
- `hRegions : mid_value ≠ (Region.cast mid_index : RegionName)`

**Closed-form spec defs (transitive):** `argmax_kernel_1`, `argmaxKernel1ValueSpec`, `argmaxKernel1InputTile`

<details><summary><code>argmax_kernel_1</code></summary>

```
/-- Faithful transcription of `triton_argmax.py`'s first-stage
`argmax_kernel_1`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `INT64_INDEX: tl.constexpr` -> Lean
  parameters. -/
```
```lean
def argmax_kernel_1
    (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE : Nat) (INT64_INDEX : Bool := Bool.false) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  if INT64_INDEX {
    pid = (pid).to(tl.int64)
  }
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
  max_index = max_index + pid * $(BLOCK_SIZE)
  mid_value_ptr = mid_value + pid
  max_index_ptr = mid_index + pid
  tl.store(mid_value_ptr, max_val)
  tl.store(max_index_ptr, max_index)
}
```
</details>

<details><summary><code>argmaxKernel1ValueSpec</code></summary>

```
/-- Exact `max_val` written by `argmax_kernel_1` at lane `pid`. -/
```
```lean
noncomputable def argmaxKernel1ValueSpec
    (s : BlockState) (inp : RegionName) (M BLOCK_SIZE : Nat) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (argmaxKernel1InputTile s inp M BLOCK_SIZE) with
  | some out => WithBot.unbotD 0 (out.data PUnit.unit)
  | none => 0
```
</details>

<details><summary><code>argmaxKernel1InputTile</code></summary>

```
/-- Masked input tile for `argmax_kernel_1`. Masked lanes evaluate to `none`
matching `other=-float("inf")`. -/
```
```lean
noncomputable def argmaxKernel1InputTile
    (s : BlockState) (inp : RegionName) (M BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * BLOCK_SIZE + idx.1.val
      if off < M then some (s.readMem inp off) else none }
```
</details>

## Public theorem: `argmax_kernel_1_index_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the index channel of `argmax_kernel_1`
(`INT64_INDEX=false` branch). -/
```
</details>

**Statement:**
```lean
specification argmax_kernel_1_index_compute_correct
    (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := argmax_kernel_1 inp mid_value mid_index M BLOCK_SIZE Bool.false)
      (initialState := s)
      (write := fun _ : PUnit =>
        some ((Region.cast mid_index : RegionName), s.pid))
      (expected :=
        fun _ : PUnit =>
          (argmaxKernel1IndexSpec s inp M BLOCK_SIZE : Nat))
```

**Closed-form spec defs (transitive):** `argmax_kernel_1`, `argmaxKernel1IndexSpec`, `argmaxKernel1InputTile`

<details><summary><code>argmax_kernel_1</code></summary>

```
/-- Faithful transcription of `triton_argmax.py`'s first-stage
`argmax_kernel_1`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` / `INT64_INDEX: tl.constexpr` -> Lean
  parameters. -/
```
```lean
def argmax_kernel_1
    (inp mid_value : RegionName) (mid_index : Region .int)
    (M BLOCK_SIZE : Nat) (INT64_INDEX : Bool := Bool.false) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  if INT64_INDEX {
    pid = (pid).to(tl.int64)
  }
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val, max_index = tl.max(inp_val, axis=0, return_indices=True)
  max_index = max_index + pid * $(BLOCK_SIZE)
  mid_value_ptr = mid_value + pid
  max_index_ptr = mid_index + pid
  tl.store(mid_value_ptr, max_val)
  tl.store(max_index_ptr, max_index)
}
```
</details>

<details><summary><code>argmaxKernel1IndexSpec</code></summary>

```
/-- Exact `max_index` (after the `+ pid * BLOCK_SIZE` shift) written by
`argmax_kernel_1` at lane `pid`. -/
```
```lean
noncomputable def argmaxKernel1IndexSpec
    (s : BlockState) (inp : RegionName) (M BLOCK_SIZE : Nat) : Nat :=
  (Tile.argMaxDrop (shape := [BLOCK_SIZE]) ⟨0, by simp⟩
    (argmaxKernel1InputTile s inp M BLOCK_SIZE)).data PUnit.unit
    + s.pid * BLOCK_SIZE
```
</details>

<details><summary><code>argmaxKernel1InputTile</code></summary>

```
/-- Masked input tile for `argmax_kernel_1`. Masked lanes evaluate to `none`
matching `other=-float("inf")`. -/
```
```lean
noncomputable def argmaxKernel1InputTile
    (s : BlockState) (inp : RegionName) (M BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * BLOCK_SIZE + idx.1.val
      if off < M then some (s.readMem inp off) else none }
```
</details>

## Public theorem: `argmax_kernel_2_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for `argmax_kernel_2`. -/
```
</details>

**Statement:**
```lean
specification argmax_kernel_2_compute_correct
    (mid_value mid_index out : RegionName)
    (mid_size BLOCK_MID : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := argmax_kernel_2 mid_value mid_index out mid_size BLOCK_MID)
      (initialState := s)
      (write := fun _ : PUnit => some (out, 0))
      (expected := fun _ => argmaxKernel2Spec s mid_value mid_index mid_size BLOCK_MID)
```

**Closed-form spec defs (transitive):** `argmax_kernel_2`, `argmaxKernel2Spec`, `argmaxKernel2IndexOffset`, `argmaxKernel2InputTile`

<details><summary><code>argmax_kernel_2</code></summary>

```
/-- Faithful transcription of `triton_argmax.py`'s `argmax_kernel_2`.

`mid_index` and `out` are typed Int regions matching the launch site's
`torch.int64` buffers, so their `tl.load` / `tl.store` calls do not need
extra `dtype=` kwargs. -/
```
```lean
def argmax_kernel_2
    (mid_value : RegionName) (mid_index out : Region .int)
    (mid_size BLOCK_MID : Nat) :
    ComputeKernel := triton {
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid_value + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  index_val = tl.argmax(mid_val, axis=0)
  mid_index_ptrs = mid_index + index_val
  out_val = tl.load(mid_index_ptrs)
  tl.store(out, out_val)
}
```
</details>

<details><summary><code>argmaxKernel2Spec</code></summary>

```lean
noncomputable def argmaxKernel2Spec
    (s : BlockState) (mid_value mid_index : RegionName) (mid_size BLOCK_MID : Nat) : Int :=
  s.readMemValue .int mid_index
    (argmaxKernel2IndexOffset s mid_value mid_size BLOCK_MID)
```
</details>

<details><summary><code>argmaxKernel2IndexOffset</code></summary>

```lean
noncomputable def argmaxKernel2IndexOffset
    (s : BlockState) (mid_value : RegionName) (mid_size BLOCK_MID : Nat) : Nat :=
  (Tile.argMaxDrop (shape := [BLOCK_MID]) ⟨0, by simp⟩
    (argmaxKernel2InputTile s mid_value mid_size BLOCK_MID)).data PUnit.unit
```
</details>

<details><summary><code>argmaxKernel2InputTile</code></summary>

```lean
noncomputable def argmaxKernel2InputTile
    (s : BlockState) (mid_value : RegionName) (mid_size BLOCK_MID : Nat) :
    Tile .real [BLOCK_MID] :=
  { data := fun idx =>
      if idx.1.val < mid_size then some (s.readMem mid_value idx.1.val) else none }
```
</details>

## Public theorem: `argmax_kernel_dim_single_block_compute_correct`

<details><summary>docstring</summary>

```
/-- Compute-facing correctness for the dim-specific `argmax_kernel` in the
single-block regime exercised by the Python tests (`0 < N ≤ BLOCK_N`). This
covers the output index store for every active `(m, k)` lane in the launch
tile. -/
```
</details>

**Statement:**
```lean
specification argmax_kernel_dim_single_block_compute_correct
    (inp : RegionName) (out_index : Region .int)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hBN : 0 < BLOCK_N)
    (hNpos : 0 < N)
    (hNle : N ≤ BLOCK_N)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => argmaxKernelOutOffset s K BLOCK_M i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := argmax_kernel inp out_index M N K BLOCK_M BLOCK_N Bool.false)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_M => s.pids 0 * BLOCK_M + i.val < M)
        (fun i => ((Region.cast out_index : RegionName),
          argmaxKernelOutOffset s K BLOCK_M i)))
      (expected := fun i =>
        argmaxKernelDimSingleBlockSpec s inp M N K BLOCK_M BLOCK_N i)
```

**Assumptions / layout contracts:**
- `hBN : 0 < BLOCK_N`
- `hNpos : 0 < N`
- `hNle : N ≤ BLOCK_N`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => argmaxKernelOutOffset s K BLOCK_M i)`
- `fun i : Fin BLOCK_M => s.pids 0 * BLOCK_M + i.val < M`

**Closed-form spec defs (transitive):** `argmaxKernelOutOffset`, `argmax_kernel`, `argmaxKernelDimSingleBlockSpec`, `argmaxKernelDimSingleBlockRowMax`, `argmaxKernelArgmaxSpec`, `argmaxKernelInputTile`

<details><summary><code>argmaxKernelOutOffset</code></summary>

```
/-- Output offset for the dim-specific `argmax_kernel`: lane `i` of the
`[BLOCK_M]` output row writes to `out_index` at position
`(pid_m * BLOCK_M + i) * K + pid_k`. -/
```
```lean
@[reducible] noncomputable def argmaxKernelOutOffset
    (s : BlockState) (K BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  (s.pids 0 * BLOCK_M + i.val) * K + s.pids 1
```
</details>

<details><summary><code>argmax_kernel</code></summary>

```
/-- Faithful transcription of `triton_argmax.py`'s dim-specific
`argmax_kernel`.

The `out_index` region is typed Int, matching the launch site's `torch.int64`
buffer. -/
```
```lean
def argmax_kernel
    (inp : RegionName) (out_index : Region .int)
    (M N K BLOCK_M BLOCK_N : Nat) (INT64_INDEX : Bool := Bool.false) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  if INT64_INDEX {
    pid_m = (pid_m).to(tl.int64)
    pid_k = (pid_k).to(tl.int64)
  }
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  max_values = tl.full([$(BLOCK_M)], dtype=tl.float32, value=float("-inf"))
  argmax_values = tl.full([$(BLOCK_M)], dtype=tl.int64, value=0)
  for start_n in range($(0), $(N), $(BLOCK_N)) {
    n_offset = start_n + tl.arange(0, $(BLOCK_N))
    offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
    mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    local_max, local_argmax := tl.max(inp_vals, 1,
      return_indices=True, return_indices_tie_break_left=True)
    update = local_max > max_values
    max_values = tl.where(update, local_max, max_values)
    argmax_values = tl.where(update, start_n + local_argmax, argmax_values)
  }
  offset_index = m_offset * $(K) + pid_k
  out_index_ptrs = out_index + offset_index
  mask1 = m_offset < $(M)
  tl.store(out_index_ptrs, argmax_values, mask=mask1)
}
```
</details>

<details><summary><code>argmaxKernelDimSingleBlockSpec</code></summary>

```
/-- Exact per-row value carried by `argmax_values` after the single
`start_n = 0` loop iteration. This keeps the kernel's `update` guard explicit:
when every lane in a row is masked, the initial zero is retained. -/
```
```lean
noncomputable def argmaxKernelDimSingleBlockSpec
    (s : BlockState) (inp : RegionName) (M N K BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) : Int := by
  classical
  exact if ComparableDType.real.gt
      (argmaxKernelDimSingleBlockRowMax s inp M N K BLOCK_M BLOCK_N i) none then
    argmaxKernelArgmaxSpec s inp M N K BLOCK_M BLOCK_N i
  else 0
```
</details>

<details><summary><code>argmaxKernelDimSingleBlockRowMax</code></summary>

```lean
noncomputable def argmaxKernelDimSingleBlockRowMax
    (s : BlockState) (inp : RegionName) (M N K BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) : WithBot ℝ := by
  classical
  by_cases h : 0 < BLOCK_N
  · haveI : Nonempty (Fin BLOCK_N) := Fin.pos_iff_nonempty.mp h
    exact Finset.univ.sup' Finset.univ_nonempty
      (fun n : Fin BLOCK_N =>
        (argmaxKernelInputTile s inp M N K BLOCK_M BLOCK_N).data
          (i, n, PUnit.unit))
  · exact none
```
</details>

<details><summary><code>argmaxKernelArgmaxSpec</code></summary>

```
/-- Per-row argmax over the 2D input tile, dropping axis 1 (matching
`tl.max(inp_vals, 1, return_indices=True, return_indices_tie_break_left=True)`). -/
```
```lean
noncomputable def argmaxKernelArgmaxSpec
    (s : BlockState) (inp : RegionName) (M N K BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) : Int :=
  Int.ofNat
    ((Tile.argMaxDrop (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩
      (argmaxKernelInputTile s inp M N K BLOCK_M BLOCK_N)).data (i, PUnit.unit))
```
</details>

<details><summary><code>argmaxKernelInputTile</code></summary>

```
/-- The 2D `[BLOCK_M, BLOCK_N]` input tile loaded inside the `argmax_kernel`
loop body at `start_n = 0`. Masked-out lanes evaluate to `none` matching
`tl.load(..., mask=..., other=-float("inf"))`. -/
```
```lean
noncomputable def argmaxKernelInputTile
    (s : BlockState) (inp : RegionName) (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      let m := idx.1.val
      let n := idx.2.1.val
      let mOff := s.pids 0 * BLOCK_M + m
      let off := mOff * N * K + n * K + s.pids 1
      if mOff < M ∧ n < N then some (s.readMem inp off) else none }
```
</details>
