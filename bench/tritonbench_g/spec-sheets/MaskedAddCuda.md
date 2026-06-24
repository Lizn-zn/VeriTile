# Spec sheet — `bench/tritonbench_g/masked_add_cuda/MaskedAddCuda.lean`

**Python source:** `bench/tritonbench_g/masked_add_cuda/masked_add_cuda.py`

## Public theorem: `masked_add_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `masked_add_kernel`: the DSL surface lowers to
the algorithm layer, and the masked in-place store to `grad_ptr` is
compute-correct — every active lane (in-bounds and `p_mask` false) holds
`grad + p * alpha`, all other lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem masked_add_kernel_output_summary
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat)
    (s : BlockState) :
    (∃ alg, (masked_add_kernel grad_ptr p_ptr p_mask_ptr
      n_elements alpha BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := masked_add_kernel grad_ptr p_ptr p_mask_ptr
        n_elements alpha BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE =>
          maskedAddActive s p_mask_ptr n_elements BLOCK_SIZE i)
        (fun i => (grad_ptr, maskedAddOffset s BLOCK_SIZE i)))
      (expected := fun i => maskedAddSpec s grad_ptr p_ptr alpha BLOCK_SIZE i)
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_SIZE =>
          maskedAddActive s p_mask_ptr n_elements BLOCK_SIZE i`

**Closed-form spec defs (transitive):** `masked_add_kernel`, `maskedAddActive`, `maskedAddOffset`, `maskedAddSpec`

<details><summary><code>masked_add_kernel</code></summary>

```
/-- Faithful transcription of `masked_add_cuda.py`'s `masked_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
```
```lean
def masked_add_kernel
    (grad_ptr p_ptr p_mask_ptr : RegionName)
    (n_elements : Nat) (alpha : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
  mask = mask & ~p_mask
  p = tl.load(p_ptr + offsets, mask=mask)
  grad = tl.load(grad_ptr + offsets, mask=mask)
  grad += p * $(alpha)
  tl.store(grad_ptr + offsets, grad, mask=mask)
}
```
</details>

<details><summary><code>maskedAddActive</code></summary>

```lean
def maskedAddActive
    (s : BlockState) (p_mask_ptr : RegionName) (n_elements BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Prop :=
  maskedAddOffset s BLOCK_SIZE i < n_elements ∧
    s.readMemValue .bool p_mask_ptr (maskedAddOffset s BLOCK_SIZE i) = Bool.false
```
</details>

<details><summary><code>maskedAddOffset</code></summary>

```lean
def maskedAddOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * BLOCK_SIZE + i.val
```
</details>

<details><summary><code>maskedAddSpec</code></summary>

```lean
noncomputable def maskedAddSpec
    (s : BlockState) (grad_ptr p_ptr : RegionName)
    (alpha : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem grad_ptr (maskedAddOffset s BLOCK_SIZE i) +
    s.readMem p_ptr (maskedAddOffset s BLOCK_SIZE i) * alpha
```
</details>

## Also present (pinned special-case summaries)
- `masked_add_kernel_compute_correct`
