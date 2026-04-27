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
* Tile-tile `Op.add` via `Value.bop`.
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

/-- **`addKernel` correctness against `addSpec`.**

For any choice of region names `xReg`, `yReg`, `outReg` and any state
that has the inputs loaded, the kernel writes the elementwise sum to
`outReg`. No disjointness assumption between the regions: kernel reads
finish before the scatter to `outReg`, so the result is correct even if
some of the regions alias.

Proof structure mirrors `softmax_naive_correct` but is mechanically
shorter — no reduction ops (`reduceMax` / `reduceSum`), no unary lift
(`Value.uop` for `exp`), only `Value.bop (·+·)` and tile-tile
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
  -- The output offsets `s.pid * blockSize + k.val` are injective in `k`.
  have h_inj : Function.Injective (fun k : Fin blockSize => s.pid * blockSize + k.val) := by
    intro a b hab
    exact Fin.ext (Nat.add_left_cancel hab)
  -- Reduce the kernel via the operational semantics. Post-RP2 the offset
  -- arithmetic stays in `Nat` end to end, so no `hcast` lemma is needed.
  simp [observeAt, exec, addKernel, stepStmts, stepStmt, evalOp, Value.bop,
        BlockState.setReg, BlockState.readMem, addSpec]
  -- Substitute the loaded X/Y cells.
  unfold InputLoadedAt at _h_x _h_y
  simp_rw [_h_x, _h_y]
  -- The remaining goal is exactly the readback of an injective scatter store.
  exact BlockState.scatter_readback _ _ _ h_inj i

/-! ## TODO: Mask version (TBD in Phase 3) -/
-- def stableAddKernel (x_ptr y_ptr output_ptr n_elements block_size: Nat) : Kernel := triton {
    -- pid := tl.program_id(0)

    -- offsets := pid * $(block_size) + tl.arange(0, $(block_size))
    -- mask = offsets < n_elements

    -- x = tl.load(x_ptr + offsets, mask=mask)
    -- y = tl.load(y_ptr + offsets, mask=mask)

    -- output = x + y

    -- tl.store(out_ptr + offsets, output, mask=mask)
-- }


end VeriTile.Examples
