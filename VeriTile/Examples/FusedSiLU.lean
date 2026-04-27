/-
VeriTile.Examples.FusedSiLU

Worked correctness example: a fused MLP block with SiLU/Swish activation.

The kernel computes elementwise per program_id

    y[i] = residual[i] + (x[i] * gate[i]) * sigmoid(x[i] * gate[i])

gathering three input tiles (`x`, `gate`, `residual`) and scattering one
output tile (`y`). This is a precursor to the SwiGLU pattern common in
transformer FFN blocks (SwiGLU drops the residual and uses one input as
the sigmoid argument independently of the other multiplier; here we
keep the residual and apply SiLU to the elementwise product).

Compared to softmax / LSE / softmax-reciprocal this is a **kernel ↔ math**
correctness theorem (single kernel against a math denotation), like
`VectorAdd`. It exercises:

* Multi-buffer DSL (3 inputs + 1 output, all parameterized by `RegionName`).
* `Op.sigmoid` on a tile via `Value.uop Real.sigmoid`.
* Tile-tile `Value.bop` for `*` and `+`.
* `BlockState.scatter_readback` for the gather/scatter store readback.

Per RP1 (`Notes/research_problem_pointer_vs_named_region.md`) buffers are
named regions, parameterized via the `tl.load($(reg) + offs)` antiquote.
Per RP2 (`Notes/research_problem_address_typing.md`) address arithmetic
stays in `Nat`. Single-block aligned case (`BLOCK_SIZE = blockSize =
problem length`).

Source Triton (`.py` reference):

```python
@triton.jit
def fused_silu(x_ptr, gate_ptr, residual_ptr, out_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid      = tl.program_id(axis=0)
    offsets  = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x        = tl.load(x_ptr + offsets)
    gate     = tl.load(gate_ptr + offsets)
    residual = tl.load(residual_ptr + offsets)
    z        = x * gate
    silu     = z * tl.sigmoid(z)
    y        = residual + silu
    tl.store(out_ptr + offsets, y)
```
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## Embedded Triton AST -/

/-- Fused MLP block: `y = residual + (x * gate) * sigmoid(x * gate)`.
    Single-block, three input regions (`xReg`, `gateReg`, `residualReg`)
    and one output region (`outReg`); all `blockSize`-element tiles. -/
def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  x        := tl.load($(xReg) + offsets)
  gate     := tl.load($(gateReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  z        := x * gate
  silu     := z * tl.sigmoid(z)
  y        := residual + silu
  tl.store($(outReg) + offsets, y)
}

/-! ## Math denotation -/

/-- Elementwise spec:
    `out[i] = residuals[i] + (xs[i] * gates[i]) * sigmoid(xs[i] * gates[i])`. -/
noncomputable def fusedSiLUSpec {N : Nat}
    (xs gates residuals : Fin N → ℝ) (i : Fin N) : ℝ :=
  residuals i + (xs i * gates i) * Real.sigmoid (xs i * gates i)

/-! ## Correctness theorem -/

/-- **`fusedSiLUKernel` correctness against `fusedSiLUSpec`.**

After running the kernel with all three inputs loaded in their respective
regions, the `outReg` cells equal the elementwise fused-SiLU formula.

Proof structure mirrors `add_kernel_correct` and `softmax_naive_correct`
— purely mechanical operational-semantics walk-through plus
`scatter_readback`. The `Value.uop Real.sigmoid` step is the only new
piece beyond `add_kernel`; no reductions or unbalanced cross-buffer
dependencies. -/
theorem fused_silu_correct
    (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (_h_x   : InputLoadedAt s xReg blockSize xs)
    (_h_g   : InputLoadedAt s gateReg blockSize gates)
    (_h_res : InputLoadedAt s residualReg blockSize residuals) :
    ∀ i : Fin blockSize,
      observeAt (exec (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i
        = some (fusedSiLUSpec xs gates residuals i) := by
  intro i
  -- The output offsets `s.pid * blockSize + k.val` are injective in `k`.
  have h_inj :
      Function.Injective (fun k : Fin blockSize => s.pid * blockSize + k.val) := by
    intro a b hab
    exact Fin.ext (Nat.add_left_cancel hab)
  -- Reduce the kernel via the operational semantics. Compared to add_kernel,
  -- the new piece is `Value.uop` for the sigmoid step. No reductions, so no
  -- `Value.reduceSum` / `Value.reduceMax`.
  simp [observeAt, exec, fusedSiLUKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop,
        BlockState.setReg, BlockState.readMem, fusedSiLUSpec]
  -- Substitute the loaded x / gate / residual cells.
  unfold InputLoadedAt at _h_x _h_g _h_res
  simp_rw [_h_x, _h_g, _h_res]
  -- The remaining goal is exactly the readback of an injective scatter store.
  exact BlockState.scatter_readback _ _ _ h_inj i

end VeriTile.Examples
