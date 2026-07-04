# Spec sheet — `bench/tritonbench_g/cosine_compute/CosineCompute.lean`

**Python source:** `bench/tritonbench_g/cosine_compute/cosine_compute.py`

## Public theorem: `cos_func_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `cos_func`: the DSL surface lowers to the
algorithm layer, and the masked store to `b` is compute-correct — every active
lane holds `Real.cos (xs i)`, out-of-bounds lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem cos_func_output_summary
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s a BLOCK_SIZE xs) :
    (∃ alg, (cos_func a b n_elements BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := cos_func a b n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements)
          (fun i => (b, s.pid * BLOCK_SIZE + i.val)))
      (expected := fun i => Real.cos (xs i))
```

**Assumptions / layout contracts:**
- `hBlockSize : 0 < BLOCK_SIZE`
- `xs : Fin BLOCK_SIZE → ℝ`
- `h_x : InputLoadedAt s a BLOCK_SIZE xs`
- `fun i : Fin BLOCK_SIZE => s.pid * BLOCK_SIZE + i.val < n_elements`

**Closed-form spec defs (transitive):** `cos_func`

<details><summary><code>cos_func</code></summary>

```
/-- Faithful 1:1 transcription of `cosine_compute.py`'s `cos_func`.

Allowed mechanical Lean-syntax-only changes:
-/
```
```lean
def cos_func
    (a b : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  offset = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = offset < $(n_elements)
  a_value = tl.load(a + offset, mask=mask)
  b_value = tl.cos((a_value).to(tl.float32))
  tl.store(b + offset, b_value, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `cos_func_compute_correct`
