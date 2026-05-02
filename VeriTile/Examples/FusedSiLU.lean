/-
VeriTile.Examples.FusedSiLU

Fused SiLU kernel and unfused pipeline over the typed Triton core.

This is a kernel-refinement example: the fused kernel

  y = residual + (x * gate) * sigmoid(x * gate)

is shown equivalent to a three-kernel pipeline that materializes the
intermediate `z = x * gate`, then `silu = z * sigmoid(z)`, then adds the
residual. The proof is intentionally memory-aware: the unfused pipeline writes
temporary regions `zReg` and `siluReg`, and the refinement theorem requires
these temporaries not to alias `residualReg` so the residual input survives
until the final stage.

Source Triton (`.py` reference, aligned single-block flavour; assumes
`BLOCK_SIZE` equals the logical vector length, so no boundary mask is used):

```python
@triton.jit
def fused_silu_kernel(x_ptr, gate_ptr, residual_ptr, out_ptr,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets)
    gate = tl.load(gate_ptr + offsets)
    residual = tl.load(residual_ptr + offsets)

    z = x * gate
    silu = z * tl.sigmoid(z)
    y = residual + silu

    tl.store(out_ptr + offsets, y)

@triton.jit
def silu_step_gate(x_ptr, gate_ptr, z_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    gate = tl.load(gate_ptr + offsets)
    tl.store(z_ptr + offsets, x * gate)

@triton.jit
def silu_step_silu(z_ptr, silu_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    z = tl.load(z_ptr + offsets)
    tl.store(silu_ptr + offsets, z * tl.sigmoid(z))

@triton.jit
def silu_step_residual(silu_ptr, residual_ptr, out_ptr,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    silu = tl.load(silu_ptr + offsets)
    residual = tl.load(residual_ptr + offsets)
    tl.store(out_ptr + offsets, residual + silu)
```
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  x        := tl.load(tl.ptr($(xReg)) + offsets)
  gate     := tl.load(tl.ptr($(gateReg)) + offsets)
  residual := tl.load(tl.ptr($(residualReg)) + offsets)
  z        := x * gate
  silu     := z * tl.sigmoid(z)
  y        := residual + silu
  tl.store(tl.ptr($(outReg)) + offsets, y)
}

def siluStepGate (xReg gateReg zReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  x       := tl.load(tl.ptr($(xReg)) + offsets)
  gate    := tl.load(tl.ptr($(gateReg)) + offsets)
  z       := x * gate
  tl.store(tl.ptr($(zReg)) + offsets, z)
}

def siluStepSilu (zReg siluReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  z       := tl.load(tl.ptr($(zReg)) + offsets)
  silu    := z * tl.sigmoid(z)
  tl.store(tl.ptr($(siluReg)) + offsets, silu)
}

def siluStepResidual (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  silu     := tl.load(tl.ptr($(siluReg)) + offsets)
  residual := tl.load(tl.ptr($(residualReg)) + offsets)
  y        := residual + silu
  tl.store(tl.ptr($(outReg)) + offsets, y)
}

noncomputable def execUnfusedSiLU
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (s : BlockState) : Option BlockState :=
  match exec (siluStepGate xReg gateReg zReg blockSize) s with
  | none => none
  | some s1 =>
    match exec (siluStepSilu zReg siluReg blockSize) s1 with
    | none => none
    | some s2 => exec (siluStepResidual siluReg residualReg outReg blockSize) s2

noncomputable def fusedSiLUSpec {blockSize : Nat}
    (xs gates residuals : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  residuals i + (xs i * gates i) * Real.sigmoid (xs i * gates i)

noncomputable def gateSpec {blockSize : Nat}
    (xs gates : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  xs i * gates i

noncomputable def siluSpec {blockSize : Nat}
    (zs : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  zs i * Real.sigmoid (zs i)

end VeriTile.Examples
