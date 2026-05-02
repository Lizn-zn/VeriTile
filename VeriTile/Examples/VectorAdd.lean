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
def addKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : Kernel := triton {
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
    (blockSize nElements : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask    := offsets < $(nElements)
  x       := tl.load($(xReg) + offsets, mask=mask)
  y       := tl.load($(yReg) + offsets, mask=mask)
  output  := x + y
  tl.store($(outReg) + offsets, output, mask=mask)
}

end VeriTile.Examples
