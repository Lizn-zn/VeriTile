# Spec sheet — `bench/tritonbench_g/quant_transpose_kernel/QuantTransposeKernel.lean`

**Python source:** `bench/tritonbench_g/quant_transpose_kernel/quant_transpose_kernel.py`

## Public theorem: `quantize_global_transpose_blocked_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general blocked output summary.** For arbitrary strides, sizes
`M`/`N`, block sizes `BLOCK_M`/`BLOCK_N`, group factor `GROUP_M`, and real scale
`scale127` (and any program coordinates in `s`), the faithful full surface is
recorded as **blocked** at algorithm erasure (it stores CUDA `llrint`/int8
results, not the real-valued pre-rounding expression), while the checked
scaled-store slice realizes the genuine pre-rounding quantity
`scale127 * (a * absmax_inv)` (`quantTransposeScaledSpec`) at every in-range tile
lane, leaving out-of-range lanes unchanged. This holds over arbitrary (symbolic)
dimensions. Output-address injectivity for the transposed writeback is taken as a
hypothesis (`hOutInj`). The `llrint` rounding / int8 cast remain the honest,
unmodeled blocker. -/
```
</details>

**Statement:**
```lean
specification quantize_global_transpose_blocked_output_summary_general
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat)
    (scale127 : ℝ)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)) :
    (∃ err, (quantize_global_transpose_real_surface A AbsmaxInv B
      stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N
      GROUP_M).toAlgorithm? = Except.error err) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        bOffset s stride_bm stride_bn BLOCK_M BLOCK_N idx)`

**Closed-form spec defs (transitive):** `bOffset`, `quantize_global_transpose_real_surface`, `rowIndex`, `colIndex`

<details><summary><code>bOffset</code></summary>

```lean
def bOffset
    (s : BlockState) (stride_bm stride_bn BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  rowIndex s BLOCK_M idx.1 * stride_bm + colIndex s BLOCK_N idx.2.1 * stride_bn
```
</details>

<details><summary><code>quantize_global_transpose_real_surface</code></summary>

```
/-- Real-valued surface of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

This preserves the grouped one-dimensional program-id schedule, masked load,
global scale, CUDA `llrint` surface operation, transposed store addressing, and
masked writeback. The algorithm carrier records the pre-cast real value. -/
```
```lean
def quantize_global_transpose_real_surface
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N GROUP_M : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  grid_m = ($((M : Nat)) + $((BLOCK_M : Nat)) - $((1 : Nat))) // $((BLOCK_M : Nat))
  grid_n = ($((N : Nat)) + $((BLOCK_N : Nat)) - $((1 : Nat))) // $((BLOCK_N : Nat))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // group_size
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  A = A + (rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  a = tl.load(A, mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  B = B + (rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))
  tl.store(B, output, mask=mask)
}
```
</details>

<details><summary><code>rowIndex</code></summary>

```lean
def rowIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val
```
</details>

<details><summary><code>colIndex</code></summary>

```lean
def colIndex (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val
```
</details>

## Public theorem: `quant_transpose_scaled_store_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for the checked scaled-store slice of
`quant_transpose_kernel.py`'s `_quantize_global_transpose`: for every disjoint
flat placement of the three buffers, every program coordinate `(pid₀, pid₁)`
whose active lanes are in bounds, and every launch state whose `A` tile window
holds `xs` at the active lanes and whose `AbsmaxInv` cell holds the broadcast
scalar, the translated pointer kernel terminates, every active lane of the
**transposed** output tile holds the genuine pre-rounding quantity
`scale127 * (xs · ys)`, and every other memory cell is unchanged.

Dimension-general in all four strides, `M`, `N`, `BLOCK_M`, `BLOCK_N`, and the
real scale. Honest side-conditions: output-address injectivity for the
transposed writeback at every program coordinate (`hOutInj` — the same
hypothesis the per-write-map summary takes, here universally quantified over the
two program axes because `Implements` quantifies over them), and a non-degenerate
tile (`0 < BLOCK_M`, `0 < BLOCK_N`), which is what lets the signature's
broadcast-scalar channel witness the `AbsmaxInv` load's own bound. The CUDA
`llrint` rounding / int8 cast remain the honest, unmodeled blocker, exactly as in
the summary above — this face covers the same real-valued slice. -/
```
</details>

**Statement:**
```lean
specification quant_transpose_scaled_store_io_correctness
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ)
    (hM : 0 < BLOCK_M) (hN : 0 < BLOCK_N)
    (hOutInj : ∀ p₀ p₁ : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        (p₀ * BLOCK_M + idx.1.val) * stride_bm
          + (p₁ * BLOCK_N + idx.2.1.val) * stride_bn)) :
    quantTransposeScaledIO A AbsmaxInv B stride_am stride_an stride_bn stride_bm
        M N BLOCK_M BLOCK_N scale127
      ⊨ fun _p₀ _p₁ xs ys idx => scale127 * (xs idx * ys idx)
```

**Assumptions / layout contracts:**
- `hM : 0 < BLOCK_M`
- `hN : 0 < BLOCK_N`
- `fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        (p₀ * BLOCK_M + idx.1.val) * stride_bm
          + (p₁ * BLOCK_N + idx.2.1.val) * stride_bn`

**Closed-form spec defs (transitive):** `quantTransposeScaledIO`, `quantize_global_transpose_scaled_store_slice`

<details><summary><code>quantTransposeScaledIO</code></summary>

```
/-- IO signature of the scaled-store slice on the **tile-indexed** two-input
surface: lane `(i, j)` of program `(pid₀, pid₁)` reads `A` at
`row·stride_am + col·stride_an`, reads the broadcast scalar `AbsmaxInv` at
offset `0` on every lane, writes `B` at the *transposed*
`row·stride_bm + col·stride_bn`, and is read/write-active exactly on the
kernel's `rm < M ∧ rn < N` guard. -/
```
```lean
def quantTransposeScaledIO (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) : MaskedTile2DKernelIO₂ where
  kernel := quantize_global_transpose_scaled_store_slice A AbsmaxInv B
    stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N scale127
  in1 := A
  in2 := AbsmaxInv
  out := B
  shape := [BLOCK_M, BLOCK_N]
  read1 := fun p₀ p₁ idx =>
    (p₀ * BLOCK_M + idx.1.val) * stride_am
      + (p₁ * BLOCK_N + idx.2.1.val) * stride_an
  read2 := fun _ _ _ => 0
  write := fun p₀ p₁ idx =>
    (p₀ * BLOCK_M + idx.1.val) * stride_bm
      + (p₁ * BLOCK_N + idx.2.1.val) * stride_bn
  mask := fun p₀ p₁ idx =>
    p₀ * BLOCK_M + idx.1.val < M ∧ p₁ * BLOCK_N + idx.2.1.val < N
  read2Mask := fun _ _ _ => True
```
</details>

<details><summary><code>quantize_global_transpose_scaled_store_slice</code></summary>

```
/-- Proof-oriented scaled-store tile slice of `quant_transpose_kernel.py`'s
`_quantize_global_transpose`.

The full Triton kernel uses a one-dimensional grouped program-id schedule to
derive `pid_m` and `pid_n`. This slice starts after that scheduling choice, uses
program axes 0/1 for the tile coordinates, loads the `BLOCK_M × BLOCK_N` tile
from `A`, applies the global `absmax_inv` scale, and proves the masked writeback
into `B`. CUDA `llrint` and int8 casting are outside VeriTile's current real-tile
arithmetic layer, matching the other quantization ports. -/
```
```lean
def quantize_global_transpose_scaled_store_slice
    (A AbsmaxInv B : RegionName)
    (stride_am stride_an stride_bn stride_bm M N BLOCK_M BLOCK_N : Nat)
    (scale127 : ℝ) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  mask = (rm[:, None] < $(M)) & (rn[None, :] < $(N))
  a = tl.load(A + rm[:, None] * $(stride_am) + rn[None, :] * $(stride_an),
    mask=mask)
  absmax_inv = tl.load(AbsmaxInv)
  output = $(scale127) * (a * absmax_inv)
  tl.store(B + rm[:, None] * $(stride_bm) + rn[None, :] * $(stride_bn),
    output, mask=mask)
}
```
</details>

## Also present (pinned special-case summaries)
- `quantize_global_transpose_scaled_store_slice_compute_correct`
