/-
VeriTile.Examples.FusedSiLU

Worked equivalence example: fused-SiLU kernel vs manually-expanded
sigmoid kernel. Both compute the same fused MLP block elementwise:

    y[i] = residual[i] + (x[i] * gate[i]) * sigmoid(x[i] * gate[i])

Differing only in how `sigmoid` itself is realized at the kernel level.

The fused version uses `tl.sigmoid` (a single unary op). The separate
version expands sigmoid into its definitional form

    sigmoid(z) = 1 / (1 + exp(-z))

step-by-step (`neg_z`, `exp`, `denom`, `sig`, `silu`). This is the kind
of rewrite a Triton compiler / hand-tuned author may apply when the
target hardware lacks a native sigmoid instruction (or to fuse the
computation with surrounding arithmetic differently).

The refinement theorem `silu_kernels_refinement` shows the two kernels
write the same `outReg` memory.

Structure (mirrors `SoftmaxEq.lean`):

  (a) Embedded Triton ASTs via `triton { ... }` macro
  (b) Math denotation (`fusedSiLUSpec`, `separateSiLUSpec`) and the
      load-bearing identity `fused_eq_separate_silu`
  (c) Per-kernel correctness (`fused_silu_correct`,
      `separate_silu_correct`) — symbolic operational walk-throughs
      finished by `BlockState.scatter_readback`
  (d) Refinement theorem composing (b) and (c)

Per RP1 (`Notes/research_problem_pointer_vs_named_region.md`) buffers are
named regions, parameterized via the `tl.load($(reg) + offs)` antiquote.
Per RP2 (`Notes/research_problem_address_typing.md`) address arithmetic
stays in `Nat`. Single-block aligned case (`BLOCK_SIZE = blockSize =
problem length`).
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## (a) Embedded Triton ASTs -/

/-- Fused MLP block, using `tl.sigmoid` directly:
    `y = residual + (x * gate) * sigmoid(x * gate)`. -/
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

/-- Same MLP block, with `sigmoid` manually expanded as
    `sigmoid(z) = 1 / (1 + exp(-z))`. -/
def separateSiLUKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  x        := tl.load($(xReg) + offsets)
  gate     := tl.load($(gateReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  z        := x * gate
  neg_z    := 0 - z
  e        := tl.exp(neg_z)
  denom    := 1 + e
  sig      := 1 / denom
  silu     := z * sig
  y        := residual + silu
  tl.store($(outReg) + offsets, y)
}

/-! ## (b) Math denotation and equivalence -/

/-- Closed form using `Real.sigmoid` directly. -/
noncomputable def fusedSiLUSpec {N : Nat}
    (xs gates residuals : Fin N → ℝ) (i : Fin N) : ℝ :=
  residuals i + (xs i * gates i) * Real.sigmoid (xs i * gates i)

/-- Closed form with `sigmoid` expanded as `1 / (1 + exp(-z))`. -/
noncomputable def separateSiLUSpec {N : Nat}
    (xs gates residuals : Fin N → ℝ) (i : Fin N) : ℝ :=
  residuals i + (xs i * gates i) * (1 / (1 + Real.exp (-(xs i * gates i))))

/-- The load-bearing math identity: the two specs agree pointwise.
    Direct application of Mathlib's `Real.sigmoid_def`
    (`x.sigmoid = (1 + Real.exp(-x))⁻¹`) plus `one_div`. -/
theorem fused_eq_separate_silu {N : Nat}
    (xs gates residuals : Fin N → ℝ) (i : Fin N) :
    fusedSiLUSpec xs gates residuals i =
    separateSiLUSpec xs gates residuals i := by
  unfold fusedSiLUSpec separateSiLUSpec
  rw [Real.sigmoid_def, ← one_div]

/-! ## (c) Per-kernel correctness -/

/-- **`fusedSiLUKernel` correctness against `fusedSiLUSpec`.** -/
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
  -- Reduce the kernel via the operational semantics.
  simp [observeAt, exec, fusedSiLUKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop,
        BlockState.setReg, BlockState.readMem, fusedSiLUSpec]
  -- Substitute the loaded x / gate / residual cells.
  unfold InputLoadedAt at _h_x _h_g _h_res
  simp_rw [_h_x, _h_g, _h_res]
  -- Readback of an injective scatter store.
  exact BlockState.scatter_readback _ _ _ h_inj i

/-- **`separateSiLUKernel` correctness against `separateSiLUSpec`.**
    Structurally identical proof to `fused_silu_correct`; the only delta
    is the additional `Value.bop` / `Value.uop` steps for the manual
    sigmoid expansion. -/
theorem separate_silu_correct
    (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (_h_x   : InputLoadedAt s xReg blockSize xs)
    (_h_g   : InputLoadedAt s gateReg blockSize gates)
    (_h_res : InputLoadedAt s residualReg blockSize residuals) :
    ∀ i : Fin blockSize,
      observeAt (exec (separateSiLUKernel xReg gateReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i
        = some (separateSiLUSpec xs gates residuals i) := by
  intro i
  have h_inj :
      Function.Injective (fun k : Fin blockSize => s.pid * blockSize + k.val) := by
    intro a b hab
    exact Fin.ext (Nat.add_left_cancel hab)
  -- More intermediate registers (`neg_z`, `e`, `denom`, `sig`) but the
  -- same simp set works — they're all `Value.bop` (×, −, +, /) and one
  -- `Value.uop Real.exp`.
  simp [observeAt, exec, separateSiLUKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop,
        BlockState.setReg, BlockState.readMem, separateSiLUSpec]
  unfold InputLoadedAt at _h_x _h_g _h_res
  simp_rw [_h_x, _h_g, _h_res]
  exact BlockState.scatter_readback _ _ _ h_inj i

/-! ## (d) Refinement: fused ≡ separate -/

/-- **`fusedSiLUKernel` and `separateSiLUKernel` produce the same `outReg`
    memory.**

    Composes the two correctness lemmas through the math identity
    `fused_eq_separate_silu`. Mirrors `softmax_kernels_refinement`. -/
theorem silu_kernels_refinement
    (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (h_x   : InputLoadedAt s xReg blockSize xs)
    (h_g   : InputLoadedAt s gateReg blockSize gates)
    (h_res : InputLoadedAt s residualReg blockSize residuals) :
    ∀ i : Fin blockSize,
      observeAt (exec (fusedSiLUKernel    xReg gateReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i =
      observeAt (exec (separateSiLUKernel xReg gateReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i := by
  intro i
  rw [fused_silu_correct
        xReg gateReg residualReg outReg blockSize hN s xs gates residuals h_x h_g h_res i,
      separate_silu_correct
        xReg gateReg residualReg outReg blockSize hN s xs gates residuals h_x h_g h_res i]
  congr 1
  exact fused_eq_separate_silu xs gates residuals i

end VeriTile.Examples
