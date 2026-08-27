# Spec sheet — `bench/tritonbench_g/matrix_transpose/MatrixTranspose.lean`

**Python source:** `bench/tritonbench_g/matrix_transpose/matrix_transpose.py`

## Public theorem: `kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `kernel` implements the matrix transpose on its IO
signature — for every disjoint flat placement of `M`/`Out`, every program id
whose lanes are in bounds, and every launch state whose transposed source
window holds the tile `xs`, the translated pointer kernel terminates, output
lane `j` holds `xs j` (i.e. the output cell `(d, m)` holds the source cell
`(m, d)` — the transposition lives in the signature's windows, which this
theorem proves the kernel actually uses), and every other memory cell is
unchanged.

`hOutInj` is an honest, **truth-required** side condition: two distinct output
lanes must not alias the same memory cell, otherwise only the last store
survives and the per-lane claim is false. The host's contiguous `out` buffer
satisfies it. -/
```
</details>

**Statement:**
```lean
specification kernel_correctness
    (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] =>
        outAddr out_stridex out_stridey idx)) :
    transposeIO M Out matrix_stridex matrix_stridey out_stridex out_stridey
        SIZE_M D_HEAD
      ⊨ fun _ _ xs j => xs j
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [D_HEAD, SIZE_M] =>
        outAddr out_stridex out_stridey idx)`

**Closed-form spec defs (transitive):** `outAddr`, `transposeIO`, `kernel`, `matrixAddr`

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

<details><summary><code>transposeIO</code></summary>

```
/-- `kernel`'s **IO signature** — the whole kernel-specific audit surface of
the `⊨` headline:

* `inp`/`out` — which buffer is which argument (`M`, `Out`);
* `B = D_HEAD * SIZE_M` — the output tile flattened row-major into lanes,
  lane `j` = logical output cell `(j / SIZE_M, j % SIZE_M)` (`laneIdx`);
* `read` — lane `j`'s **transposed** source address
  `m * matrix_stridex + d * matrix_stridey`;
* `write` — lane `j`'s output address `d * out_stridex + m * out_stridey`;
* `mask` — every lane (the kernel's load and store are both unmasked), so
  `writeMask` keeps the struct default.

The grid is `(1,)`, so no field depends on either program id. -/
```
```lean
def transposeIO (M Out : RegionName)
    (matrix_stridex matrix_stridey out_stridex out_stridey
      SIZE_M D_HEAD : Nat) :
    Masked2DKernelIO₁ where
  kernel := kernel M Out matrix_stridex matrix_stridey out_stridex out_stridey
    SIZE_M D_HEAD
  inp := M
  out := Out
  B := D_HEAD * SIZE_M
  read := fun _ _ j =>
    matrixAddr matrix_stridex matrix_stridey (Lane2D.decode j)
  write := fun _ _ j =>
    outAddr out_stridex out_stridey (Lane2D.decode j)
  mask := fun _ _ _ => True

/-- **The headline**: `kernel` implements the matrix transpose on its IO
signature — for every disjoint flat placement of `M`/`Out`, every program id
whose lanes are in bounds, and every launch state whose transposed source
window holds the tile `xs`, the translated pointer kernel terminates, output
lane `j` holds `xs j` (i.e. the output cell `(d, m)` holds the source cell
`(m, d)` — the transposition lives in the signature's windows, which this
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
