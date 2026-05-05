/-
VeriTile.Examples.VectorAdd

Worked correctness example: elementwise add (multi-buffer kernel).

Compared to the softmax / LSE / softmax-reciprocal examples, this is a
**kernel ↔ math** correctness theorem (single-kernel against a math
denotation), not a kernel ↔ kernel refinement.

It exercises:
* Multi-buffer DSL (auto-populated `Kernel.inputs`/`outputs` — see
  `Triton/DSL.lean`): three regions `X`, `Y`, `Out`.
* Two-argument `tl.arange(0, N)` form.
* Tile-tile `Op.add` via `Tile.bop`.
* `BlockState.scatter_readback` for the gather/scatter store readback.

Per RP1 (`Notes/research_problem_pointer_vs_named_region.md`), the
buffers are named regions, not first-class pointers. Per the boundary-
mask discussion, this is the **single-block aligned case**:
`BLOCK_SIZE = N = problem length`, single program_id, no boundary
mask. Proper masked / multi-block extensions are Phase C / D.

Source Triton (`.py` reference, aligned single-block flavour):

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid     = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # mask  = offsets < n_elements   -- omitted: aligned (n_elements = BLOCK_SIZE)
    x       = tl.load(x_ptr + offsets)
    y       = tl.load(y_ptr + offsets)
    output  = x + y
    tl.store(out_ptr + offsets, output)
```
-/

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## Embedded Triton AST

Region names are kernel parameters (`xReg`, `yReg`, `outReg`), threaded
through the DSL via the `$(term) + offset` pointer-like syntax in
`tl.load(...)` / `tl.store(...)`. The same kernel can be instantiated
with any choice of buffer names. -/

/-- Elementwise add of two `blockSize`-element tiles, single-block.

Reads from `xReg` and `yReg`, writes to `outReg`. No aliasing assumption
required: even if `xReg = outReg` (read-modify-write of the same buffer)
the kernel reads first into local registers `x` / `y` before the final
scatter to `outReg`, so the result is still `xs + ys`. -/
def addKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  y    := tl.load($(yReg) + offs)
  out  := x + y
  tl.store($(outReg) + offs, out)
}

/-! ## Math denotation -/

/-- Elementwise add: `out[i] = xs[i] + ys[i]`. -/
def addSpec {N : Nat} (xs ys : Fin N → ℝ) (i : Fin N) : ℝ :=
  xs i + ys i

/-! ## Correctness theorem -/

/-- **`addKernel` correctness against `addSpec`.**

For any choice of region names `xReg`, `yReg`, `outReg` and any state
that has the inputs loaded, the kernel writes the elementwise sum to
`outReg`. No disjointness assumption between the regions: kernel reads
finish before the scatter to `outReg`, so the result is correct even if
some of the regions alias.

Proof structure mirrors `softmax_naive_correct` but is mechanically
shorter — no reduction ops (`reduceMax` / `reduceSum`), no unary lift
(`Tile.uop` for `exp`), only `Tile.bop (·+·)` and tile-tile
gather/scatter. -/
theorem add_kernel_correct
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (_hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_y : InputLoadedAt s yReg blockSize ys) :
    ∀ i : Fin blockSize,
      observeAt (exec (addKernel xReg yReg outReg blockSize) s) outReg blockSize s.pid i
        = some (addSpec xs ys i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, addKernel, stepStmts, stepStmt, evalOp, Tile.bop,
        NumericDType.add, NumericDType.mul, addSpec]
  unfold InputLoadedAt at _h_x _h_y
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_y]

/-- View-level surface for `add_kernel_correct`. -/
theorem add_kernel_correct_exec_view
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg blockSize) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg blockSize) ys) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe (exec (addKernel xReg yReg outReg blockSize) s)
          (programTileView s outReg blockSize) idx
        = some (addSpec xs ys idx.1) := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  have hy := inputLoadedAt_of_programTileView_loaded (s := s) (region := yReg)
    (N := blockSize) (xs := ys) h_y
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using add_kernel_correct xReg yReg outReg blockSize hBlockSize s xs ys hx hy idx.1

/-- Compute-facing correctness theorem for `addKernel`.

From the fixed initial state `s`, the compute-facing kernel writes the
view-level elementwise-add result to `outReg`. -/
theorem add_kernel_compute_correct
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg blockSize) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg blockSize) ys) :
    ComputeKernel.ComputeCorrect
      ((addKernel xReg yReg outReg blockSize))
      (fun s0 s' =>
        s0 = s →
        ∀ idx : TileIndex [blockSize],
          TensorView.observe (some s')
              (programTileView s outReg blockSize) idx
            = some (addSpec xs ys idx.1)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have hview := add_kernel_correct_exec_view xReg yReg outReg blockSize hBlockSize
    s xs ys h_x h_y idx
  rw [hExec] at hview
  simpa using hview

/-- Manifest-compatible view-level surface for `add_kernel_compute_correct`. -/
theorem add_kernel_correct_view
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg blockSize) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg blockSize) ys) :
    ComputeKernel.ComputeCorrect
      ((addKernel xReg yReg outReg blockSize))
      (fun s0 s' =>
        s0 = s →
        ∀ idx : TileIndex [blockSize],
          TensorView.observe (some s')
              (programTileView s outReg blockSize) idx
            = some (addSpec xs ys idx.1)) := by
  exact add_kernel_compute_correct xReg yReg outReg blockSize hBlockSize
    s xs ys h_x h_y



/-! ## Masked variant (boundary mask)

The aligned `addKernel` above only handles `n_elements = block_size`. The
masked variant handles arbitrary `n_elements` by computing
`mask = offsets < n_elements` and threading it through the load and store,
exactly mirroring the canonical Triton tutorial:

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    tl.store(out_ptr + offsets, output, mask=mask)
```

-/

/-- Masked elementwise add. Lanes where `pid * blockSize + i < nElements` are
    loaded, summed, and stored. Lanes outside the bound get Triton's
    `other=None` undefined load value, but the store mask skips those lanes,
    so the undefined values are not observed. -/
def addKernelMasked (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask    := offsets < $(nElements)
  x       := tl.load($(xReg) + offsets, mask=mask)
  y       := tl.load($(yReg) + offsets, mask=mask)
  output  := x + y
  tl.store($(outReg) + offsets, output, mask=mask)
}

/-- **`addKernelMasked` correctness.**

For each lane `i ∈ Fin blockSize`:
* In-bounds (`pid * blockSize + i < nElements`): the output region holds
  `xs i + ys i` at `pid * blockSize + i`.
* Out-of-bounds: the output region's value at `pid * blockSize + i` is
  preserved from the initial state (mask=false → no store).

The hypothesis `InputLoadedAt` constrains memory at every lane in the
`blockSize`-length tile (including out-of-bounds lanes where the data
is irrelevant semantically). This matches Triton's actual behavior:
masked-off loads without `other=` do not read memory and produce
undefined lane values; the matching masked store prevents those values
from reaching memory.

No region-disjointness hypothesis: the kernel reads `x` and `y` into
local registers BEFORE the scatter to `outReg`, so even if `outReg`
aliases `xReg` or `yReg`, the result is correct. -/
theorem add_kernel_masked_correct
    (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) (_hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_y : InputLoadedAt s yReg blockSize ys) :
    ∀ i : Fin blockSize,
      let addr := s.pid * blockSize + i.val
      observeAt (exec (addKernelMasked xReg yReg outReg blockSize nElements) s)
                outReg blockSize s.pid i
        = some (if addr < nElements then xs i + ys i
                else s.readMem outReg addr) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, addKernelMasked, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, NumericDType.add, NumericDType.mul,
        ComparableDType.lt]
  unfold InputLoadedAt at h_x h_y
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : s.pid * blockSize + i.val < nElements
  · simp [hi, h_x, h_y]
  · simp [hi]

/-- View-level surface for `add_kernel_masked_correct`. -/
theorem add_kernel_masked_correct_exec_view
    (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg blockSize) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg blockSize) ys) :
    ∀ idx : TileIndex [blockSize],
      let addr := s.pid * blockSize + idx.1.val
      TensorView.observe (exec (addKernelMasked xReg yReg outReg blockSize nElements) s)
          (programTileView s outReg blockSize) idx
        = some (if addr < nElements then xs idx.1 + ys idx.1
                else s.readMem outReg addr) := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  have hy := inputLoadedAt_of_programTileView_loaded (s := s) (region := yReg)
    (N := blockSize) (xs := ys) h_y
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt, addSpec]
    using add_kernel_masked_correct xReg yReg outReg blockSize nElements
      hBlockSize s xs ys hx hy idx.1

/-- Compute-facing view-level surface for `add_kernel_masked_correct`. -/
theorem add_kernel_masked_correct_view
    (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg blockSize) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg blockSize) ys) :
    ComputeKernel.ComputeCorrect
      ((addKernelMasked xReg yReg outReg blockSize nElements))
      (fun s0 s' =>
        s0 = s →
        ∀ idx : TileIndex [blockSize],
          let addr := s.pid * blockSize + idx.1.val
          TensorView.observe (some s')
              (programTileView s outReg blockSize) idx
            = some (if addr < nElements then xs idx.1 + ys idx.1
                    else s.readMem outReg addr)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have hview := add_kernel_masked_correct_exec_view xReg yReg outReg blockSize nElements
    hBlockSize s xs ys h_x h_y idx
  rw [hExec] at hview
  simpa using hview

end VeriTile.Examples
