# Spec sheet — `bench/tritonbench_g/vector_addition_custom/VectorAdditionCustom.lean`

**Python source:** `bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py`

## Public theorem: `add_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_add_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `C` is compute-correct — every active
lane holds `as i + bs i`, out-of-bounds lanes are preserved. -/
```
</details>

**Statement:**
```lean
specification add_kernel_output_summary
    (A B C : RegionName)
    (size BLOCK : Nat) (hBlock : 0 < BLOCK)
    (s : BlockState) (as bs : Fin BLOCK → ℝ)
    (h_a : InputLoadedAt s A BLOCK as)
    (h_b : InputLoadedAt s B BLOCK bs) :
    (∃ alg, (_add_kernel A B C size BLOCK).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := _add_kernel A B C size BLOCK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK => s.pid * BLOCK + i.val < size)
          (fun i => (C, s.pid * BLOCK + i.val)))
      (expected := fun i => as i + bs i)
```

**Assumptions / layout contracts:**
- `hBlock : 0 < BLOCK`
- `as bs : Fin BLOCK → ℝ`
- `h_a : InputLoadedAt s A BLOCK as`
- `h_b : InputLoadedAt s B BLOCK bs`
- `fun i : Fin BLOCK => s.pid * BLOCK + i.val < size`

**Closed-form spec defs (transitive):** `_add_kernel`

<details><summary><code>_add_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `vector_addition_custom.py`'s `_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def _add_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, a + b, mask=offs < $(size))
}
```
</details>

## Also present (pinned special-case summaries)
- `add_kernel_compute_correct`
