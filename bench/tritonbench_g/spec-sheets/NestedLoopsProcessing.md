# Spec sheet — `bench/tritonbench_g/nested_loops_processing/NestedLoopsProcessing.lean`

**Python source:** `bench/tritonbench_g/nested_loops_processing/nested_loops_processing.py`

## Public theorem: `nested3_python_surfaces_output_summary`

<details><summary>docstring</summary>

```
/-- Public Python surface summary for `nested_loops_processing.py`. -/
```
</details>

**Statement:**
```lean
theorem nested3_python_surfaces_output_summary
    (in_ptr out_ptr : RegionName) :
    nested3_python_surfaces_prop in_ptr out_ptr
```

**Closed-form spec defs (transitive):** `nested3_python_surfaces_prop`, `nested3`

<details><summary><code>nested3_python_surfaces_prop</code></summary>

```
/-- Python-tested surface layouts for `nested_loops_processing.py`.

The nested-loop body store proofs above remain per-store facts; this prop keeps
the Python layout list as a single local theorem surface without promoting
kernel-specific square sizes into `Semantics/`. -/
```
```lean
abbrev nested3_python_surfaces_prop
    (in_ptr out_ptr : RegionName) : Prop :=
  (∃ alg, (nested3 in_ptr out_ptr 8 1).toAlgorithm? = Except.ok alg) ∧
  (∃ alg, (nested3 in_ptr out_ptr 4 1).toAlgorithm? = Except.ok alg) ∧
  (∃ alg, (nested3 in_ptr out_ptr 16 1).toAlgorithm? = Except.ok alg) ∧
  (∃ alg, (nested3 in_ptr out_ptr 2 1).toAlgorithm? = Except.ok alg)
```
</details>

<details><summary><code>nested3</code></summary>

```
/-- Faithful transcription of `nested_loops_processing.py`'s `nested3`.

Allowed mechanical Lean-syntax-only changes:
- Python literal loop bounds `range(0, 2)` are written as
  `range(0, 2, 1)`.
- Scalar constants in pointer increments are antiquoted so they are inferred
  in the Nat offset channel. -/
```
```lean
def nested3 (in_ptr out_ptr : RegionName)
    (stride_m stride_n : Nat) : ComputeKernel := triton {
  offs_am = tl.arange(0, 2)
  offs_an = tl.arange(0, 2)
  a_ptrs = in_ptr + (offs_am[:, None] * $(stride_m) +
    offs_an[None, :] * $(stride_n))
  offs_cm = tl.arange(0, 2)
  offs_cn = tl.arange(0, 2)
  c_ptrs = out_ptr + $(stride_m) * offs_cm[:, None] +
    $(stride_n) * offs_cn[None, :]
  for i in range(0, $(2), $(1)) {
    a1 = tl.load(a_ptrs)
    for j in range(0, $(2), $(1)) {
      a_ptrs += $(2) * $(stride_n)
      a2 = tl.load(a_ptrs)
      for k in range(0, $(2), $(1)) {
        a_ptrs += $(2) * $(stride_n)
        a3 = tl.load(a_ptrs)
        tl.store(c_ptrs, a1)
        c_ptrs += $(2) * $(stride_n)
        tl.store(c_ptrs, a2)
        c_ptrs += $(2) * $(stride_n)
        tl.store(c_ptrs, a3)
        c_ptrs += $(2) * $(stride_n)
      }
    }
    a_ptrs += $(2) * $(stride_n)
  }
}
```
</details>

## Also present (pinned special-case summaries)
- `nested3_first_a1_store_compute_correct`
- `nested3_shifted_store_compute_correct`
- `nested3_first_a2_store_compute_correct`
- `nested3_first_a3_store_compute_correct`
- `nested3_shifted_copy_store_compute_correct`
- `nested3_second_k_a1_store_compute_correct`
- `nested3_second_k_a2_store_compute_correct`
- `nested3_second_k_a3_store_compute_correct`
- `nested3_second_j_first_k_a1_store_compute_correct`
- `nested3_second_j_first_k_a2_store_compute_correct`
- `nested3_second_j_first_k_a3_store_compute_correct`
- `nested3_second_j_second_k_a1_store_compute_correct`
- `nested3_second_j_second_k_a2_store_compute_correct`
- `nested3_second_j_second_k_a3_store_compute_correct`
- `nested3_second_i_first_j_first_k_a1_store_compute_correct`
- `nested3_second_i_first_j_first_k_a2_store_compute_correct`
- `nested3_second_i_first_j_first_k_a3_store_compute_correct`
- `nested3_second_i_first_j_second_k_a1_store_compute_correct`
- `nested3_second_i_first_j_second_k_a2_store_compute_correct`
- `nested3_second_i_first_j_second_k_a3_store_compute_correct`
- `nested3_second_i_second_j_first_k_a1_store_compute_correct`
- `nested3_second_i_second_j_first_k_a2_store_compute_correct`
- `nested3_second_i_second_j_first_k_a3_store_compute_correct`
- `nested3_second_i_second_j_second_k_a1_store_compute_correct`
- `nested3_second_i_second_j_second_k_a2_store_compute_correct`
- `nested3_second_i_second_j_second_k_a3_store_compute_correct`
