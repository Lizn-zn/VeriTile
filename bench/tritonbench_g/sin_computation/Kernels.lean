import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `sin_computation` — strict per-kernel correctness

`sin_kernel` is an elementwise sine: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of `in_ptr0`, applies `tl.sin` lane-wise,
and stores to `out_ptr`, masked by `offsets < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`sin_kernel[(n_elements,)](...)`, the grid size, and how
the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `pid` is universally quantified,
the per-program statement covers every program of the grid.

The headline is stated on the kernel's masked **IO signature** `sinIO`
(`MaskedKernelIO₁`): a full masked Hoare triple over flat pointer memory — for
**every** disjoint placement of the two buffers, **every** program id all of
whose *active* lanes are in bounds (partial blocks may overhang the buffer on
their inactive lanes), and **every** launch state whose input window holds `xs`
at the active lanes, the translated pointer kernel terminates, every active
output lane holds `Real.sin (xs i)`, and every other memory cell is unchanged.

## Proof architecture

```
sin_kernel_correctness                        ← TOP SPECIFICATION (sinIO ⊨ sin)
  ├─ sin_kernel_flattenOk                     bridge fragment membership
  ├─ sin_kernel_traceSafe                     per-execution lane-wise safety walk
  └─ sin_kernel_region_run                    region-model masked Hoare triple
       ├─ sin_kernel_correct                  algorithm-layer readback per lane
       └─ sin_kernel_frame                    masked scatter-store cell frame
```

The spec is plain elementwise `Real.sin (xs i)`. `sinIO` is the kernel's masked
IO signature: `read`/`write` windows at `pid * BLOCK_SIZE` and the active-lane
predicate `pid * BLOCK_SIZE + j < n_elements` (the load/store mask; the store
is not gated more tightly, so `writeMask` keeps its default `mask`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float), and `tl.sin` is modeled by
the exact `Real.sin`. No output/input disjointness is assumed at the region
layer: the input is read into registers before the scatter, so the result is
correct even if `out_ptr` aliases `in_ptr0`.
-/

namespace VeriTile.Bench.TritonBenchG.SinComputation

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Faithful 1:1 transcription of `sin_computation.py`'s `sin_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def sin_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  output = tl.sin(x)
  tl.store(out_ptr + offsets, output, mask=mask)
}



/-- The pinned Triton sine accepts only fp32 and fp64 input tensors. -/
inductive InputStorage where
  | fp32 | fp64
  deriving DecidableEq, Repr

/-- Supplied output storage is independent of the sine input storage. -/
inductive OutputStorage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

/-- Complete typed source and its natural-index commutation candidate. Explicit
store casts expose Triton's implicit conversion to the output pointer dtype. -/
def typedKernel (input : InputStorage) (output : OutputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) : ComputeKernel :=
  match input, output, v with
  | .fp32, .fp32, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float32
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp32, .fp32, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float32
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp32, .fp64, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float64
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float64), mask=mask)
    }).surfaceBody
  | .fp32, .fp64, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float64
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float64), mask=mask)
    }).surfaceBody
  | .fp32, .fp16, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float16
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float16), mask=mask)
    }).surfaceBody
  | .fp32, .fp16, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float16
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float16), mask=mask)
    }).surfaceBody
  | .fp32, .bf16, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.bfloat16
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.bfloat16), mask=mask)
    }).surfaceBody
  | .fp32, .bf16, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.bfloat16
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.bfloat16), mask=mask)
    }).surfaceBody
  | .fp32, .f8e4, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float8e4nv
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e4nv), mask=mask)
    }).surfaceBody
  | .fp32, .f8e4, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float8e4nv
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e4nv), mask=mask)
    }).surfaceBody
  | .fp32, .f8e5, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float8e5
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e5), mask=mask)
    }).surfaceBody
  | .fp32, .f8e5, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float32, Y = tl.float8e5
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e5), mask=mask)
    }).surfaceBody
  | .fp64, .fp32, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float32
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float32), mask=mask)
    }).surfaceBody
  | .fp64, .fp32, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float32
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float32), mask=mask)
    }).surfaceBody
  | .fp64, .fp64, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float64
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp64, .fp64, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float64
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output, mask=mask)
    }).surfaceBody
  | .fp64, .fp16, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float16
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float16), mask=mask)
    }).surfaceBody
  | .fp64, .fp16, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float16
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float16), mask=mask)
    }).surfaceBody
  | .fp64, .bf16, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.bfloat16
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.bfloat16), mask=mask)
    }).surfaceBody
  | .fp64, .bf16, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.bfloat16
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.bfloat16), mask=mask)
    }).surfaceBody
  | .fp64, .f8e4, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float8e4nv
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e4nv), mask=mask)
    }).surfaceBody
  | .fp64, .f8e4, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float8e4nv
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e4nv), mask=mask)
    }).surfaceBody
  | .fp64, .f8e5, Bool.false => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float8e5
      pid = tl.program_id(0)
      block_start = pid * $(B)
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e5), mask=mask)
    }).surfaceBody
  | .fp64, .f8e5, Bool.true => ComputeKernel.mk [X] [Y] (triton {
      tl.region X = tl.float64, Y = tl.float8e5
      pid = tl.program_id(0)
      block_start = $(B) * pid
      offsets = block_start + tl.arange(0, $(B))
      mask = offsets < $(N)
      x = tl.load(X + offsets, mask=mask)
      output = tl.sin(x)
      tl.store(Y + offsets, output.to(tl.float8e5), mask=mask)
    }).surfaceBody

end VeriTile.Bench.TritonBenchG.SinComputation
