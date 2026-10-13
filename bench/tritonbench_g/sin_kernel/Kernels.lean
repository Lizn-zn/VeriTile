import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `sin_kernel` — strict per-kernel correctness

`kernel_function` is an elementwise sine: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of `x_ptr`, applies `tl.math.sin`
lane-wise, and stores to `output_ptr`, masked by `offsets < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`kernel_function[grid](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`pid` is universally quantified, the per-program statement covers every program
of the grid.

The headline is stated on the kernel's masked **IO signature** `sinIO`
(`MaskedKernelIO₁`): a full masked Hoare triple over **flat pointer memory** —
for every disjoint placement of the two buffers, every program id all of whose
*active* lanes are in bounds (partial blocks may overhang the buffer on their
inactive lanes), and every launch state whose input window holds `xs` at the
active lanes, the translated pointer kernel terminates, every active output
lane holds `Real.sin (xs i)`, and every other flat cell is untouched.

## Proof architecture

```
kernel_function_correctness                   ← TOP SPECIFICATION (sinIO ⊨ sine)
  ├─ kernel_function_flattenOk                bridge fragment membership
  ├─ kernel_function_traceSafe                per-execution lane-wise safety walk
  └─ kernel_function_region_run               region-model masked Hoare triple
       ├─ kernel_function_correct             algorithm-layer readback per lane
       └─ kernel_function_frame               masked scatter-store cell frame
```

The spec is plain elementwise `Real.sin (xs i)`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float), and `tl.math.sin` is
modeled by the exact `Real.sin`. No
output/input disjointness is assumed at the region layer: the input is read
into registers before the scatter, so the result is correct even if
`output_ptr` aliases `x_ptr`.
-/

namespace VeriTile.Bench.TritonBenchG.SinKernel

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Faithful 1:1 transcription of `sin_kernel.py`'s `kernel_function`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def kernel_function
    (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = tl.math.sin(x)
  tl.store(output_ptr + offsets, output, mask=mask)
}



/-- The pinned Triton sine accepts only fp32 and fp64 input tensors. -/
inductive InputStorage where
  | fp32 | fp64
  deriving DecidableEq, Repr

/-- Complete typed source and its natural-index commutation candidate. The
wrapper's empty_like output has the same dtype as its input. -/
def typedKernel (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  match input, v with
  | .fp32, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float32
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.math.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp32, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float32
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.math.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp64, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float64
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.math.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp64, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float64
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.math.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody

end VeriTile.Bench.TritonBenchG.SinKernel
