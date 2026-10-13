import VeriTile.Triton

/-!
# `swiglu_fwd` — strict per-kernel correctness

`_swiglu_fwd_kernel` is the forward SwiGLU: program `(row, col_block)` loads a
tile of gate `X` and value `Y` for its row, computes `swiglu(x, y) = x · σ(x) · y`
lane-wise, and stores the result to `OUT`, masked by `cols < ncols`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_swiglu_fwd_kernel[grid](...)`, the 2-D grid
`(M, cdiv(N, BLOCK_N))`, the `@triton.autotune` choice of `BLOCK_N`, and how the
runtime composes per-program writes into one buffer) is the *trusted boundary*,
not a proof obligation here. Because both program ids (`pids 0`, `pids 1`) are
universally quantified, the per-program statement covers every program of the
grid.

## Proof architecture

```
swiglu_fwd_kernel_correctness                 ← TOP THEOREM (swigluIO ⊨ lane-wise swiglu)
  ├─ swiglu_fwd_kernel_flattenOk              bridge fragment membership
  ├─ swiglu_fwd_kernel_traceSafe              per-execution lane-wise safety walk
  └─ swiglu_fwd_kernel_region_run             region-model masked Hoare triple
       ├─ swiglu_fwd_kernel_exec_isSome       termination
       ├─ swiglu_fwd_kernel_correct           algorithm-layer readback per lane
       └─ swiglu_fwd_kernel_frame             masked scatter frame
```

The headline is stated on the kernel's masked **IO signature** `swigluIO`
(`Masked2DKernelIO₂`, the two-axis family: this kernel reads
`tl.program_id(1)`): which buffer is which argument, where program
`(row, col_block) = (pid₀, pid₁)` reads its `X`/`Y` tiles and writes its `OUT`
tile (lane `j` at `pid₀ * stride + pid₁ * BLOCK_N + j`), and the active-lane
predicate `pid₁ * BLOCK_N + j < ncols`. `⊨` is the audit-once masked
Hoare-triple combinator (`Masked2DKernelIO₂.Implements`): for **every** disjoint
placement of the three buffers in flat memory, **every** program-id pair all of
whose *active* lanes are in bounds (partial blocks may overhang the buffer on
their inactive lanes), and **every** launch state whose input windows hold
`xs`/`ys` at the active lanes, the translated pointer kernel terminates, every
active output lane holds `TiledActivation.swiglu (xs j) (ys j)`, and every
other memory cell is unchanged.

## Modeling boundary

The spec is an **oracle wrapper** over `VeriTile.Triton.Math.Activation`: the
SwiGLU math (`TiledActivation.swiglu`, built on `TiledActivation.silu`) lives
once in `Math.Activation` and is reused here, so this file only checks that the
kernel realizes that oracle lane-wise. Arithmetic is over `ℝ` (not bit-accurate
IEEE float); the `.to(tl.float32)` casts reduce to the identity at the algorithm
layer (post-erasure all dtypes unify to `ℝ`); `@triton.autotune` is not modeled.
No output/input disjointness is assumed in the region model: both inputs are
read into registers before the scatter, so the result is correct even if `OUT`
aliases `X` or `Y`.
-/

namespace VeriTile.Bench.TritonBenchG.SwigluFwd

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂

/-- Faithful transcription of `swiglu_fwd.py`'s `_swiglu_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
def swiglu_fwd_kernel
    (X Y OUT : RegionName)
    (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(BLOCK_N)
  X += row * $(stride_x_row)
  Y += row * $(stride_y_row)
  OUT += row * $(stride_out_row)
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  out = x * tl.sigmoid(x) * y
  tl.store(OUT + cols, out, mask=cols < $(ncols))
}


/-! Complete typed source, including all three independently selected storage
formats. The candidate commutes only the natural row-address products. -/
open Lean
syntax "swigluForward%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(swigluForward%[$xType:tritonDType, $yType:tritonDType,
      $outType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let xBase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| X += row * $(stride_x_row))
      else `(tritonStmt| X += $(stride_x_row) * row)
    let yBase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| Y += row * $(stride_y_row))
      else `(tritonStmt| Y += $(stride_y_row) * row)
    let outBase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| OUT += row * $(stride_out_row))
      else `(tritonStmt| OUT += $(stride_out_row) * row)
    let output : TSyntax `tritonStmt ← match outType with
      | `(tritonDType| tl.float32) =>
        `(tritonStmt| tl.store(OUT + cols, out, mask=cols < $(ncols)))
      | _ => `(tritonStmt| tl.store(OUT + cols, out.to($outType), mask=cols < $(ncols)))
    `(fun (X Y OUT : RegionName)
        (stride_x_row stride_y_row stride_out_row ncols BLOCK_N : Nat) =>
      ComputeKernel.mk [X,Y] [OUT] (triton {
  tl.region X = $xType, Y = $yType, OUT = $outType
  row = tl.program_id(0)
  start_col = tl.program_id(1) * $(BLOCK_N)
  $xBase
  $yBase
  $outBase
  cols = start_col + tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  y = tl.load(Y + cols, mask=cols < $(ncols), other=0.0).to(tl.float32)
  out = x * tl.sigmoid(x) * y
  $output
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp32_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp32_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp32_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp32_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp32_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp32_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp16_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp16_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp16_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp16_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp16_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp16_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_bf16_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_bf16_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_bf16_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_bf16_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_bf16_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_bf16_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp64_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp64_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp64_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp64_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp64_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_fp64_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e4_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e4_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e4_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e4_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e4_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e4_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e5_fp32 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e5_fp16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e5_bf16 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e5_fp64 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e5_f8e4 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Full source with fixed input storage and independent output storage. -/
def typed_f8e5_f8e5 (output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match output, v with
  | .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float32, original] X Y OUT sx sy so N B
  | .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float32, commuted] X Y OUT sx sy so N B
  | .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float16, original] X Y OUT sx sy so N B
  | .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float16, commuted] X Y OUT sx sy so N B
  | .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.bfloat16, original] X Y OUT sx sy so N B
  | .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.bfloat16, commuted] X Y OUT sx sy so N B
  | .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float64, original] X Y OUT sx sy so N B
  | .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float64, commuted] X Y OUT sx sy so N B
  | .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float8e4nv, original] X Y OUT sx sy so N B
  | .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float8e4nv, commuted] X Y OUT sx sy so N B
  | .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float8e5, original] X Y OUT sx sy so N B
  | .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, tl.float8e5, commuted] X Y OUT sx sy so N B

/-- Both complete source variants for all independent floating storage triples. -/
def typedKernel (input partner output : Storage) (v : Bool)
    (X Y OUT : RegionName) (sx sy so N B : Nat) : ComputeKernel :=
  match input, partner with
  | .fp32, .fp32 => typed_fp32_fp32 output v X Y OUT sx sy so N B
  | .fp32, .fp16 => typed_fp32_fp16 output v X Y OUT sx sy so N B
  | .fp32, .bf16 => typed_fp32_bf16 output v X Y OUT sx sy so N B
  | .fp32, .fp64 => typed_fp32_fp64 output v X Y OUT sx sy so N B
  | .fp32, .f8e4 => typed_fp32_f8e4 output v X Y OUT sx sy so N B
  | .fp32, .f8e5 => typed_fp32_f8e5 output v X Y OUT sx sy so N B
  | .fp16, .fp32 => typed_fp16_fp32 output v X Y OUT sx sy so N B
  | .fp16, .fp16 => typed_fp16_fp16 output v X Y OUT sx sy so N B
  | .fp16, .bf16 => typed_fp16_bf16 output v X Y OUT sx sy so N B
  | .fp16, .fp64 => typed_fp16_fp64 output v X Y OUT sx sy so N B
  | .fp16, .f8e4 => typed_fp16_f8e4 output v X Y OUT sx sy so N B
  | .fp16, .f8e5 => typed_fp16_f8e5 output v X Y OUT sx sy so N B
  | .bf16, .fp32 => typed_bf16_fp32 output v X Y OUT sx sy so N B
  | .bf16, .fp16 => typed_bf16_fp16 output v X Y OUT sx sy so N B
  | .bf16, .bf16 => typed_bf16_bf16 output v X Y OUT sx sy so N B
  | .bf16, .fp64 => typed_bf16_fp64 output v X Y OUT sx sy so N B
  | .bf16, .f8e4 => typed_bf16_f8e4 output v X Y OUT sx sy so N B
  | .bf16, .f8e5 => typed_bf16_f8e5 output v X Y OUT sx sy so N B
  | .fp64, .fp32 => typed_fp64_fp32 output v X Y OUT sx sy so N B
  | .fp64, .fp16 => typed_fp64_fp16 output v X Y OUT sx sy so N B
  | .fp64, .bf16 => typed_fp64_bf16 output v X Y OUT sx sy so N B
  | .fp64, .fp64 => typed_fp64_fp64 output v X Y OUT sx sy so N B
  | .fp64, .f8e4 => typed_fp64_f8e4 output v X Y OUT sx sy so N B
  | .fp64, .f8e5 => typed_fp64_f8e5 output v X Y OUT sx sy so N B
  | .f8e4, .fp32 => typed_f8e4_fp32 output v X Y OUT sx sy so N B
  | .f8e4, .fp16 => typed_f8e4_fp16 output v X Y OUT sx sy so N B
  | .f8e4, .bf16 => typed_f8e4_bf16 output v X Y OUT sx sy so N B
  | .f8e4, .fp64 => typed_f8e4_fp64 output v X Y OUT sx sy so N B
  | .f8e4, .f8e4 => typed_f8e4_f8e4 output v X Y OUT sx sy so N B
  | .f8e4, .f8e5 => typed_f8e4_f8e5 output v X Y OUT sx sy so N B
  | .f8e5, .fp32 => typed_f8e5_fp32 output v X Y OUT sx sy so N B
  | .f8e5, .fp16 => typed_f8e5_fp16 output v X Y OUT sx sy so N B
  | .f8e5, .bf16 => typed_f8e5_bf16 output v X Y OUT sx sy so N B
  | .f8e5, .fp64 => typed_f8e5_fp64 output v X Y OUT sx sy so N B
  | .f8e5, .f8e4 => typed_f8e5_f8e4 output v X Y OUT sx sy so N B
  | .f8e5, .f8e5 => typed_f8e5_f8e5 output v X Y OUT sx sy so N B

end VeriTile.Bench.TritonBenchG.SwigluFwd
