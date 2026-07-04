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
theorem var_len_copy_kernel_triton_small_length_output_summary
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
    ComputeCorrect.Realizes
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

## Also present (pinned special-case summaries)
- `var_len_copy_one_chunk_compute_correct`
- `var_len_copy_kernel_triton_small_length_compute_correct`
