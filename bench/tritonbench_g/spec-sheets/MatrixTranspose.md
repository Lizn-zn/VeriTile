# Spec sheet — `bench/tritonbench_g/matrix_transpose/MatrixTranspose.lean`

**Python source:** `bench/tritonbench_g/matrix_transpose/matrix_transpose.py`

## Public theorem: `kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `kernel`: the DSL surface lowers to the
algorithm layer, and the cellwise store to `Out` is compute-correct — every
output cell `idx` holds the transposed matrix cell, under the no-alias side
condition `hOutInj`. -/
```
</details>

**Statement:**
```lean
theorem kernel_output_summary
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] => outAddr out_stridex out_stridey idx)) :
    (∃ alg, (kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD)
      (initialState := s)
      (write := fun idx : TileIndex [D_HEAD, SIZE_M] =>
          some (Out, outAddr out_stridex out_stridey idx))
      (expected := fun idx => s.readMem M (matrixAddr matrix_stridex matrix_stridey idx))
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] => outAddr out_stridex out_stridey idx)`

**Closed-form spec defs (transitive):** `outAddr`, `kernel`, `matrixAddr`

<details><summary><code>outAddr</code></summary>

```
/-- Output address written by `kernel` at a logical output tile index. -/
```
```lean
def outAddr (out_stridex out_stridey : Nat)
    (idx : TileIndex [D_HEAD, SIZE_M]) : Nat :=
  idx.1.val * out_stridex + idx.2.1.val * out_stridey
```
</details>

<details><summary><code>kernel</code></summary>

```
/-- Faithful 1:1 transcription of `matrix_transpose.py`'s `kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `SIZE_M: tl.constexpr` / `D_HEAD: tl.constexpr` → Lean `Nat`
  parameters. -/
```
```lean
def kernel
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    ComputeKernel := triton {
  size_m_arange = tl.arange(0, $(SIZE_M))
  d_head_arange = tl.arange(0, $(D_HEAD))
  matrix_ptr = M + size_m_arange[:, None] * $(matrix_stridex)
                + d_head_arange[None, :] * $(matrix_stridey)
  out_ptr = Out + d_head_arange[:, None] * $(out_stridex)
             + size_m_arange[None, :] * $(out_stridey)
  matrix = tl.load(matrix_ptr)
  tl.store(out_ptr, tl.trans(matrix))
}
```
</details>

<details><summary><code>matrixAddr</code></summary>

```
/-- Source address read by `kernel` at a logical output tile index. -/
```
```lean
def matrixAddr (matrix_stridex matrix_stridey : Nat)
    (idx : TileIndex [D_HEAD, SIZE_M]) : Nat :=
  idx.2.1.val * matrix_stridex + idx.1.val * matrix_stridey
```
</details>

## Also present (pinned special-case summaries)
- `kernel_compute_correct`
