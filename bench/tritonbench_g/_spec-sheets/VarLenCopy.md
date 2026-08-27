# Spec sheet — `bench/tritonbench_g/var_len_copy/VarLenCopy.lean`

**Python source:** `bench/tritonbench_g/var_len_copy/var_len_copy.py`

## Public theorem: `var_len_copy_kernel_triton_small_length_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `var_len_copy_kernel_triton` under the
Python-tested small-length regime `0 < length ≤ BLOCK_SIZE`: the DSL surface
lowers to the algorithm layer, and the masked segment copy to `new_a_location`
is compute-correct — every active lane (`< length`) holds the matching
`old_a_location` lane, out-of-segment lanes are preserved. -/
```
</details>

**Statement:**
```lean
specification var_len_copy_kernel_triton_small_length_output_summary
    (old_a_start old_a_len : Region .nat) (old_a_location : RegionName)
    (new_a_start : Region .nat) (new_a_location : RegionName)
    (BLOCK_SIZE : Nat)
    (s : BlockState)
    (hBS : 0 < BLOCK_SIZE)
    (hLen :
      s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0)
        ≤ BLOCK_SIZE)
    (hLenPos :
      0 < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0))
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
          + i.val)) :
    (∃ alg, (var_len_copy_kernel_triton old_a_start old_a_len old_a_location
        new_a_start new_a_location BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := var_len_copy_kernel_triton old_a_start old_a_len old_a_location
        new_a_start new_a_location BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE =>
          i.val < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0))
        (fun i =>
          (new_a_location,
            s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
              + i.val)))
      (expected := fun i =>
        s.readMem old_a_location
          (s.readMemValue .nat (Region.cast old_a_start : RegionName) (s.pids 0)
            + i.val))
```

**Assumptions / layout contracts:**
- `hBS : 0 < BLOCK_SIZE`
- `hLen : s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0)
        ≤ BLOCK_SIZE`
- `hLenPos : 0 < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0)`
- `fun i : Fin BLOCK_SIZE =>
        s.readMemValue .nat (Region.cast new_a_start : RegionName) (s.pids 0)
          + i.val`
- `fun i : Fin BLOCK_SIZE =>
          i.val < s.readMemValue .nat (Region.cast old_a_len : RegionName) (s.pids 0)`

**Closed-form spec defs (transitive):** `var_len_copy_kernel_triton`

<details><summary><code>var_len_copy_kernel_triton</code></summary>

```
/-- Faithful transcription of `var_len_copy.py`'s `var_len_copy_kernel_triton`.

Allowed mechanical Lean-syntax-only change:
- The Python test creates the start/length metadata as `int32`; the Lean
  parameters type those metadata buffers as Nat regions so their `tl.load`
  calls do not need extra `dtype=` kwargs. -/
```
```lean
def var_len_copy_kernel_triton
    (old_a_start old_a_len : Region .nat) (old_a_location : RegionName)
    (new_a_start : Region .nat) (new_a_location : RegionName)
    (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  a_id = tl.program_id(0)
  length = tl.load(old_a_len + a_id)
  old_start = tl.load(old_a_start + a_id)
  new_start = tl.load(new_a_start + a_id)
  old_offset = tl.arange(0, $(BLOCK_SIZE))
  new_offset = tl.arange(0, $(BLOCK_SIZE))
  for i in range($(0), length, $(BLOCK_SIZE)) {
    v = tl.load(old_a_location + old_start + i + old_offset,
      mask=old_offset < length)
    tl.store(new_a_location + new_start + i + new_offset, v,
      mask=new_offset < length)
  }
}
```
</details>

## Public theorem: `var_len_copy_one_chunk_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `var_len_copy.py`'s
`var_len_copy_kernel_triton`, one-chunk slice: for every disjoint flat placement
of the five buffers, every program id whose metadata cells and active data lanes
are in bounds, **whatever** the three metadata scalars say, and every launch state
whose source window holds `xs`, the translated pointer kernel terminates, every
active destination lane holds the copied value `xs i`, and every other memory cell
is unchanged.

The three scalars are universally quantified and pinned by the launch state, so
the mask (`lane < length`) and both window bases (`old_start` / `new_start`) are
value-dependent — the shape the metadata skin exists for.

Dimension-general in `BLOCK_SIZE` and the chunk index, with **no**
side-condition: the destination window is `base + lane`, so the one-chunk
readback's output-injectivity precondition is discharged by `destOffset_inj`
rather than assumed. -/
```
</details>

**Statement:**
```lean
specification var_len_copy_one_chunk_io_correctness
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) :
    varLenOneChunkIO old_a_start old_a_len old_a_location new_a_start
        new_a_location chunk BLOCK_SIZE
      ⊨ fun _pid _len _olds _news xs i => xs i
```

**Closed-form spec defs (transitive):** `varLenOneChunkIO`, `var_len_copy_one_chunk`

<details><summary><code>varLenOneChunkIO</code></summary>

```
/-- IO signature of the one-chunk slice on the **three-metadata** surface: the
segment length, the source base and the destination base are read at the
program's own cell, and both data windows plus the mask are functions of them. -/
```
```lean
def varLenOneChunkIO
    (old_a_start old_a_len old_a_location new_a_start new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) : Meta3MaskedTileKernelIO₁ where
  kernel := var_len_copy_one_chunk old_a_start old_a_len old_a_location
    new_a_start new_a_location chunk BLOCK_SIZE
  mbuf1 := old_a_len
  mbuf2 := old_a_start
  mbuf3 := new_a_start
  inp := old_a_location
  out := new_a_location
  shape := [BLOCK_SIZE]
  mwin1 := fun pid => pid
  mwin2 := fun pid => pid
  mwin3 := fun pid => pid
  read := fun _pid _len olds _news i => olds + chunk * BLOCK_SIZE + i.1.val
  write := fun _pid _len _olds news i => news + chunk * BLOCK_SIZE + i.1.val
  mask := fun _pid len _olds _news i => i.1.val < len
```
</details>

<details><summary><code>var_len_copy_one_chunk</code></summary>

```
/-- Proof-oriented one-chunk slice of `var_len_copy.py`'s
`var_len_copy_kernel_triton`.

The Python kernel loops over chunks of a variable-length segment. This slice
fixes one chunk index and proves the masked vector copy from
`old_start + chunk * BLOCK_SIZE` to `new_start + chunk * BLOCK_SIZE`.

The metadata buffers are typed Nat regions so loads from them recover the
`.nat` channel without explicit `dtype=` kwargs. -/
```
```lean
def var_len_copy_one_chunk
    (old_a_start old_a_len : Region .nat) (old_a_location : RegionName)
    (new_a_start : Region .nat) (new_a_location : RegionName)
    (chunk BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  a_id = tl.program_id(0)
  length = tl.load(old_a_len + a_id)
  old_start = tl.load(old_a_start + a_id)
  new_start = tl.load(new_a_start + a_id)
  offset = tl.arange(0, $(BLOCK_SIZE))
  chunk_base = $(chunk) * $(BLOCK_SIZE)
  v = tl.load(old_a_location + old_start + chunk_base + offset,
    mask=offset < length)
  tl.store(new_a_location + new_start + chunk_base + offset, v,
    mask=offset < length)
}
```
</details>

## Also present (pinned special-case summaries)
- `var_len_copy_one_chunk_compute_correct`
- `var_len_copy_kernel_triton_small_length_compute_correct`
