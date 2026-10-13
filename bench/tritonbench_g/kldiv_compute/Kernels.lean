import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `kldiv_compute` — strict per-kernel correctness

`kldivergence_kernel` is an elementwise pointwise KL term: program `pid` loads
block `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of inputs `x` and `y`, computes
`x · log(x / y)` lane-wise, and stores the result to `output_ptr`, masked by
`offsets < n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`kldivergence_kernel[(num_blocks,)](...)`, the grid size
`cdiv(n_elements, BLOCK_SIZE)`, and how the runtime composes per-program writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`pid` is universally quantified, the per-program statement covers every program
of the grid.

## Proof architecture

```
kldivergence_kernel_correctness               ← TOP THEOREM (kldivIO ⊨ pointwise KL term)
  ├─ kldivergence_kernel_flattenOk            bridge fragment membership
  ├─ kldivergence_kernel_traceSafe            per-execution lane-wise safety walk
  └─ kldivergence_kernel_region_run           region-model masked Hoare triple
       ├─ kldivergence_kernel_correct         algorithm-layer readback per lane
       └─ kldivergence_kernel_frame           masked scatter-store cell frame
```

The headline is stated on the kernel's masked **IO signature** `kldivIO`
(`MaskedKernelIO₂`): which buffer is which argument, where program `pid`
reads/writes its `BLOCK_SIZE`-lane window, and the active-lane predicate
`pid * BLOCK_SIZE + j < n_elements`. `⊨` is the audit-once masked
Hoare-triple combinator (`MaskedKernelIO₂.Implements`): for **every** disjoint
placement of the three buffers in flat memory, **every** program id all of
whose *active* lanes are in bounds (partial blocks may overhang the buffer on
their inactive lanes), and **every** launch state whose input windows hold
`xs`/`ys` at the active lanes, the translated pointer kernel terminates, every
active output lane holds `klDivSpec (xs i) (ys i)`, and every other memory
cell is unchanged.

The per-lane spec is `klDivSpec (xs i) (ys i)` from the
`VeriTile.Triton.Math.Loss` (`TiledLoss`) oracle — the math is *not*
inline-duplicated here.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. No output/input disjointness is assumed at the region layer: both
inputs are read into registers before the scatter, so the result is correct
even if `output_ptr` aliases `x_ptr` or `y_ptr`.
-/

namespace VeriTile.Bench.TritonBenchG.KldivCompute

open VeriTile.Triton VeriTile.Examples
open VeriTile.Triton.TiledLoss
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Faithful 1:1 transcription of `kldiv_compute.py`'s
`kldivergence_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def kldivergence_kernel
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x * tl.log(x / y)
  tl.store(output_ptr + offsets, output, mask=mask)
}

end VeriTile.Bench.TritonBenchG.KldivCompute


namespace VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource
open Lean VeriTile.Triton

/-- Floating storage labels; native support and discrete conversions are separate obligations. -/
inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

/-- Pinned Triton rejects both-fp8 input pairs before the logarithm: division
leaves fp8 or fp16, while `tl.log` accepts only fp32/fp64. This classifier does
not establish target lowering or general native support. -/
def sourceSupported (x y : Storage) : Bool :=
  match x, y with
  | .f8e4, .f8e4 | .f8e4, .f8e5 | .f8e5, .f8e4 | .f8e5, .f8e5 => Bool.false
  | _, _ => Bool.true

syntax "kldivSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(kldivSource%[$xType:tritonDType, $yType:tritonDType, $outType:tritonDType,
      $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let blockStart : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| block_start = pid * $(B))
      else `(tritonStmt| block_start = $(B) * pid)
    let xWide : Bool ← match xType with
      | `(tritonDType| tl.float64) => pure Bool.true
      | _ => pure Bool.false
    let yWide : Bool ← match yType with
      | `(tritonDType| tl.float64) => pure Bool.true
      | _ => pure Bool.false
    let wide : Bool := xWide || yWide
    let same : Bool ← match outType with
      | `(tritonDType| tl.float64) => pure wide
      | `(tritonDType| tl.float32) => pure (!wide)
      | _ => pure Bool.false
    let store : TSyntax `tritonStmt ←
      if same then `(tritonStmt| tl.store(O + offsets, output, mask=mask))
      else `(tritonStmt| tl.store(O + offsets, output.to($outType), mask=mask))
    `(fun (X Y O : RegionName) (N B : Nat) =>
      ComputeKernel.mk [X, Y] [O] (triton {
        tl.region X = $xType, Y = $yType, O = $outType
        pid = tl.program_id(0)
        $blockStart
        offsets = block_start + tl.arange(0, $(B))
        mask = offsets < $(N)
        x = tl.load(X + offsets, mask=mask)
        y = tl.load(Y + offsets, mask=mask)
        output = x * tl.log(x / y)
        $store
      }).surfaceBody)

abbrev Source := RegionName → RegionName → RegionName → Nat → Nat → ComputeKernel

def kernel_fp32_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.float32, tl.float8e5, commuted]

def kernel_fp32_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.float16, tl.float8e5, commuted]

def kernel_fp32_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.bfloat16, tl.float8e5, commuted]

def kernel_fp32_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.float64, tl.float8e5, commuted]

def kernel_fp32_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.float8e4nv, tl.float8e5, commuted]

def kernel_fp32_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float32, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float32, tl.float8e5, tl.float8e5, commuted]

def kernel_fp16_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.float32, tl.float8e5, commuted]

def kernel_fp16_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.float16, tl.float8e5, commuted]

def kernel_fp16_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.bfloat16, tl.float8e5, commuted]

def kernel_fp16_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.float64, tl.float8e5, commuted]

def kernel_fp16_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.float8e4nv, tl.float8e5, commuted]

def kernel_fp16_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float16, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float16, tl.float8e5, tl.float8e5, commuted]

def kernel_bf16_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.float32, tl.float8e5, commuted]

def kernel_bf16_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.float16, tl.float8e5, commuted]

def kernel_bf16_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.bfloat16, tl.float8e5, commuted]

def kernel_bf16_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.float64, tl.float8e5, commuted]

def kernel_bf16_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.float8e4nv, tl.float8e5, commuted]

def kernel_bf16_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.bfloat16, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.bfloat16, tl.float8e5, tl.float8e5, commuted]

def kernel_fp64_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.float32, tl.float8e5, commuted]

def kernel_fp64_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.float16, tl.float8e5, commuted]

def kernel_fp64_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.bfloat16, tl.float8e5, commuted]

def kernel_fp64_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.float64, tl.float8e5, commuted]

def kernel_fp64_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.float8e4nv, tl.float8e5, commuted]

def kernel_fp64_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float64, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float64, tl.float8e5, tl.float8e5, commuted]

def kernel_f8e4_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.float32, tl.float8e5, commuted]

def kernel_f8e4_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.float16, tl.float8e5, commuted]

def kernel_f8e4_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.bfloat16, tl.float8e5, commuted]

def kernel_f8e4_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.float64, tl.float8e5, commuted]

def kernel_f8e4_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, commuted]

def kernel_f8e4_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e4nv, tl.float8e5, tl.float8e5, commuted]

def kernel_f8e5_fp32 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.float32, tl.float8e5, commuted]

def kernel_f8e5_fp16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.float16, tl.float8e5, commuted]

def kernel_f8e5_bf16 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.bfloat16, tl.float8e5, commuted]

def kernel_f8e5_fp64 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.float64, tl.float8e5, commuted]

def kernel_f8e5_f8e4 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.float8e4nv, tl.float8e5, commuted]

def kernel_f8e5_f8e5 (output : Storage) (v : Bool) : Source :=
  match output, v with
  | .fp32, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => kldivSource%[tl.float8e5, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => kldivSource%[tl.float8e5, tl.float8e5, tl.float8e5, commuted]

def kernel (x y output : Storage) (v : Bool) : Source :=
  match x, y with
  | .fp32, .fp32 => kernel_fp32_fp32 output v
  | .fp32, .fp16 => kernel_fp32_fp16 output v
  | .fp32, .bf16 => kernel_fp32_bf16 output v
  | .fp32, .fp64 => kernel_fp32_fp64 output v
  | .fp32, .f8e4 => kernel_fp32_f8e4 output v
  | .fp32, .f8e5 => kernel_fp32_f8e5 output v
  | .fp16, .fp32 => kernel_fp16_fp32 output v
  | .fp16, .fp16 => kernel_fp16_fp16 output v
  | .fp16, .bf16 => kernel_fp16_bf16 output v
  | .fp16, .fp64 => kernel_fp16_fp64 output v
  | .fp16, .f8e4 => kernel_fp16_f8e4 output v
  | .fp16, .f8e5 => kernel_fp16_f8e5 output v
  | .bf16, .fp32 => kernel_bf16_fp32 output v
  | .bf16, .fp16 => kernel_bf16_fp16 output v
  | .bf16, .bf16 => kernel_bf16_bf16 output v
  | .bf16, .fp64 => kernel_bf16_fp64 output v
  | .bf16, .f8e4 => kernel_bf16_f8e4 output v
  | .bf16, .f8e5 => kernel_bf16_f8e5 output v
  | .fp64, .fp32 => kernel_fp64_fp32 output v
  | .fp64, .fp16 => kernel_fp64_fp16 output v
  | .fp64, .bf16 => kernel_fp64_bf16 output v
  | .fp64, .fp64 => kernel_fp64_fp64 output v
  | .fp64, .f8e4 => kernel_fp64_f8e4 output v
  | .fp64, .f8e5 => kernel_fp64_f8e5 output v
  | .f8e4, .fp32 => kernel_f8e4_fp32 output v
  | .f8e4, .fp16 => kernel_f8e4_fp16 output v
  | .f8e4, .bf16 => kernel_f8e4_bf16 output v
  | .f8e4, .fp64 => kernel_f8e4_fp64 output v
  | .f8e4, .f8e4 => kernel_f8e4_f8e4 output v
  | .f8e4, .f8e5 => kernel_f8e4_f8e5 output v
  | .f8e5, .fp32 => kernel_f8e5_fp32 output v
  | .f8e5, .fp16 => kernel_f8e5_fp16 output v
  | .f8e5, .bf16 => kernel_f8e5_bf16 output v
  | .f8e5, .fp64 => kernel_f8e5_fp64 output v
  | .f8e5, .f8e4 => kernel_f8e5_f8e4 output v
  | .f8e5, .f8e5 => kernel_f8e5_f8e5 output v

end VeriTile.Bench.TritonBenchG.KldivCompute.TypedSource
