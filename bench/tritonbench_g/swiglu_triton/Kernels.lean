import VeriTile.Meta.StatementAudit
import VeriTile.Triton

/-!
# `swiglu_triton` — strict per-kernel correctness

Two SwiGLU kernels. `_swiglu_forward_kernel`: program `row` loads gate `a` and
value `b` for its row and stores `silu(a) · b` to `c`. `_swiglu_backward_kernel`:
recomputes `silu(a)` from the loaded gate `a`, then combines it with the upstream
gradient `dc` and writes the two input gradients in place — `da` into `a_ptr`,
`db` into `b_ptr`. Both are masked by
`col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (`_swiglu_*_kernel[(n_rows,)](...)`, the
1-D grid, the `calculate_settings` choice of `BLOCK_SIZE`/`num_warps`, and how
the runtime composes per-program writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `pid` is universally quantified,
each per-program statement covers every program of the grid.

## Proof architecture

```
swiglu_forward_kernel_correctness             ← TOP THEOREM (swigluFwdIO ⊨ swiglu)
  ├─ swiglu_forward_kernel_flattenOk          bridge fragment membership
  ├─ swiglu_forward_kernel_traceSafe          per-execution lane-wise safety walk
  └─ swiglu_forward_kernel_region_run         region-model masked Hoare triple
       ├─ swiglu_forward_kernel_correct       algorithm-layer readback per lane
       └─ swiglu_forward_kernel_frame         masked scatter cell frame

swiglu_backward_kernel_correctness            ← TOP THEOREM (swigluBwdIO ⊨ (swigluBwdA, swigluBwdB))
  ├─ swiglu_backward_kernel_flattenOk         bridge fragment membership
  ├─ swiglu_backward_kernel_traceSafe         per-execution lane-wise safety walk
  └─ swiglu_backward_kernel_region_run        region-model masked in-place triple
       ├─ swiglu_backward_kernel_correct      algorithm-layer readback per lane, per channel
       └─ swiglu_backward_kernel_frame        two-scatter cell frame
```

Both headlines are stated on the kernels' masked **IO signatures**
(`MaskedKernelIO₂` for the forward, `MaskedKernelIO₃ₓ₂` for the in-place
backward): which buffer is which argument, where program `pid` reads/writes its
`BLOCK_SIZE`-lane row window (`pid * stride`), and the active-lane predicate
`j < n_cols`. `⊨` is the audit-once masked Hoare-triple combinator: for
**every** disjoint placement of the buffers in flat memory, **every** program
id all of whose *active* lanes are in bounds, and **every** launch state whose
input windows hold the input tiles at the active lanes, the translated pointer
kernel terminates, every active output lane holds the SwiGLU oracle value, and
every other memory cell is unchanged.

## Modeling boundary

The specs are **oracle wrappers** over `VeriTile.Triton.Math.Activation`: the
SwiGLU math (`TiledActivation.swiglu`, `TiledActivation.swigluBwdA/B`, built on
`TiledActivation.silu`) lives once in `Math.Activation` and is reused here, so
this file only checks that the kernels realize those oracles lane-wise.
Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `tl.program_id(0)
.to(tl.int64)` cast and `.to(tl.float32)` load casts reduce to the identity at
the algorithm layer (post-erasure all dtypes unify to `ℝ`); `calculate_settings`
is not modeled. The backward kernel writes two channels (`da`→`a_ptr`,
`db`→`b_ptr`) and overwrites its own inputs in place; `MaskedKernelIO₃ₓ₂`
expresses this by decoupling the allocation list (`bufs`, each buffer exactly
once) from the argument roles (`out1 = in2 = a_ptr`, `out2 = in3 = b_ptr`), so
the triple reads the *old* window contents and asserts the *new* ones. Side
condition: `A ≠ B` (the two output buffers do not alias — the second masked
store would otherwise clobber the first output; no kernel caller aliases them).
-/

namespace VeriTile.Bench.TritonBenchG.SwigluTriton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₂
open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂

/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `n_cols: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters. -/
def swiglu_forward_kernel
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  a_ptr += program_id * $(stride)
  b_ptr += program_id * $(stride)
  c_ptr += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  c_row = silu(a_row) * b_row
  tl.store(c_ptr + col_offsets, c_row, mask=mask)
}

/-- Faithful transcription of `swiglu_triton.py`'s `_swiglu_backward_kernel`.

Allowed mechanical Lean-syntax-only changes match `swiglu_forward_kernel`. -/
def swiglu_backward_kernel
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  dc_ptr += program_id * $(stride)
  a_ptr += program_id * $(stride)
  b_ptr += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  sig_a = tl.sigmoid(a_row)
  silu_a = a_row * sig_a
  db_row = dc_row * silu_a
  da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
  tl.store(a_ptr + col_offsets, da_row, mask=mask)
  tl.store(b_ptr + col_offsets, db_row, mask=mask)
}

/-! Storage-specialized full source. The output conversions spell out Triton's
implicit store conversions; no arithmetic statement is omitted or reordered.
The candidate only commutes the three signed row-base products. -/
open Lean
syntax "swigluForward%" "[" tritonDType "," tritonDType "," ident "]" : term
syntax "swigluBackward%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term

set_option hygiene false in
macro_rules
  | `(swigluForward%[$aType:tritonDType, $bType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let aBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| a_ptr += program_id * $(stride))
      else `(tritonStmt| a_ptr += $(stride) * program_id)
    let bBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| b_ptr += program_id * $(stride))
      else `(tritonStmt| b_ptr += $(stride) * program_id)
    let cBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| c_ptr += program_id * $(stride))
      else `(tritonStmt| c_ptr += $(stride) * program_id)
    let wideB := match bType with
      | `(tritonDType| tl.float64) => Bool.true
      | _ => Bool.false
    let cWrite : TSyntax `tritonStmt ← match aType with
      | `(tritonDType| tl.float32) =>
        if wideB then
          `(tritonStmt| tl.store(c_ptr + col_offsets, c_row.to($aType), mask=mask))
        else `(tritonStmt| tl.store(c_ptr + col_offsets, c_row, mask=mask))
      | `(tritonDType| tl.float64) =>
        if wideB then
          `(tritonStmt| tl.store(c_ptr + col_offsets, c_row, mask=mask))
        else `(tritonStmt| tl.store(c_ptr + col_offsets, c_row.to($aType), mask=mask))
      | _ => `(tritonStmt| tl.store(c_ptr + col_offsets, c_row.to($aType), mask=mask))
    `(fun (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [a_ptr, b_ptr] [c_ptr] (triton {
  tl.region a_ptr = $aType, b_ptr = $bType, c_ptr = $aType
  program_id = tl.program_id(0).to(tl.int64)
  $aBase
  $bBase
  $cBase
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  c_row = silu(a_row) * b_row
  $cWrite
}).surfaceBody)

set_option hygiene false in
macro_rules
  | `(swigluBackward%[$aType:tritonDType, $bType:tritonDType, $dcType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let dcBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| dc_ptr += program_id * $(stride))
      else `(tritonStmt| dc_ptr += $(stride) * program_id)
    let aBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| a_ptr += program_id * $(stride))
      else `(tritonStmt| a_ptr += $(stride) * program_id)
    let bBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| b_ptr += program_id * $(stride))
      else `(tritonStmt| b_ptr += $(stride) * program_id)
    let wideB := match bType with
      | `(tritonDType| tl.float64) => Bool.true
      | _ => Bool.false
    let wideDC := match dcType with
      | `(tritonDType| tl.float64) => Bool.true
      | _ => Bool.false
    let aWrite : TSyntax `tritonStmt ← match aType with
      | `(tritonDType| tl.float32) =>
        if wideB || wideDC then
          `(tritonStmt| tl.store(a_ptr + col_offsets, da_row.to($aType), mask=mask))
        else `(tritonStmt| tl.store(a_ptr + col_offsets, da_row, mask=mask))
      | `(tritonDType| tl.float64) =>
        if wideB || wideDC then
          `(tritonStmt| tl.store(a_ptr + col_offsets, da_row, mask=mask))
        else `(tritonStmt| tl.store(a_ptr + col_offsets, da_row.to($aType), mask=mask))
      | _ => `(tritonStmt| tl.store(a_ptr + col_offsets, da_row.to($aType), mask=mask))
    let bWrite : TSyntax `tritonStmt ← match bType with
      | `(tritonDType| tl.float32) =>
        if wideDC then
          `(tritonStmt| tl.store(b_ptr + col_offsets, db_row.to($bType), mask=mask))
        else `(tritonStmt| tl.store(b_ptr + col_offsets, db_row, mask=mask))
      | `(tritonDType| tl.float64) =>
        if wideDC then
          `(tritonStmt| tl.store(b_ptr + col_offsets, db_row, mask=mask))
        else `(tritonStmt| tl.store(b_ptr + col_offsets, db_row.to($bType), mask=mask))
      | _ => `(tritonStmt| tl.store(b_ptr + col_offsets, db_row.to($bType), mask=mask))
    `(fun (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [dc_ptr, a_ptr, b_ptr] [a_ptr, b_ptr] (triton {
  tl.region dc_ptr = $dcType, a_ptr = $aType, b_ptr = $bType
  program_id = tl.program_id(0).to(tl.int64)
  $dcBase
  $aBase
  $bBase
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
  a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
  sig_a = tl.sigmoid(a_row)
  silu_a = a_row * sig_a
  db_row = dc_row * silu_a
  da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row
  $aWrite
  $bWrite
}).surfaceBody)


/-- Floating storage choices; output A/C and backward B follow their input buffers. -/
inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr



/-- Complete forward source family with independent input storage. -/
def typedFwd (storage partner : Storage) (commuted : Bool)
    (a_ptr b_ptr c_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, partner, commuted with
  | .fp32, .fp32, Bool.false =>
    swigluForward%[tl.float32, tl.float32, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .fp32, Bool.true =>
    swigluForward%[tl.float32, tl.float32, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .fp16, Bool.false =>
    swigluForward%[tl.float32, tl.float16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .fp16, Bool.true =>
    swigluForward%[tl.float32, tl.float16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .bf16, Bool.false =>
    swigluForward%[tl.float32, tl.bfloat16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .bf16, Bool.true =>
    swigluForward%[tl.float32, tl.bfloat16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .fp64, Bool.false =>
    swigluForward%[tl.float32, tl.float64, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .fp64, Bool.true =>
    swigluForward%[tl.float32, tl.float64, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .f8e4, Bool.false =>
    swigluForward%[tl.float32, tl.float8e4nv, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .f8e4, Bool.true =>
    swigluForward%[tl.float32, tl.float8e4nv, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .f8e5, Bool.false =>
    swigluForward%[tl.float32, tl.float8e5, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp32, .f8e5, Bool.true =>
    swigluForward%[tl.float32, tl.float8e5, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .fp32, Bool.false =>
    swigluForward%[tl.float16, tl.float32, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .fp32, Bool.true =>
    swigluForward%[tl.float16, tl.float32, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .fp16, Bool.false =>
    swigluForward%[tl.float16, tl.float16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .fp16, Bool.true =>
    swigluForward%[tl.float16, tl.float16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .bf16, Bool.false =>
    swigluForward%[tl.float16, tl.bfloat16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .bf16, Bool.true =>
    swigluForward%[tl.float16, tl.bfloat16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .fp64, Bool.false =>
    swigluForward%[tl.float16, tl.float64, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .fp64, Bool.true =>
    swigluForward%[tl.float16, tl.float64, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .f8e4, Bool.false =>
    swigluForward%[tl.float16, tl.float8e4nv, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .f8e4, Bool.true =>
    swigluForward%[tl.float16, tl.float8e4nv, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .f8e5, Bool.false =>
    swigluForward%[tl.float16, tl.float8e5, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp16, .f8e5, Bool.true =>
    swigluForward%[tl.float16, tl.float8e5, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .fp32, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float32, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .fp32, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float32, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .fp16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .fp16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .bf16, Bool.false =>
    swigluForward%[tl.bfloat16, tl.bfloat16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .bf16, Bool.true =>
    swigluForward%[tl.bfloat16, tl.bfloat16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .fp64, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float64, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .fp64, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float64, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .f8e4, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .f8e4, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e4nv, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .f8e5, Bool.false =>
    swigluForward%[tl.bfloat16, tl.float8e5, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .bf16, .f8e5, Bool.true =>
    swigluForward%[tl.bfloat16, tl.float8e5, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .fp32, Bool.false =>
    swigluForward%[tl.float64, tl.float32, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .fp32, Bool.true =>
    swigluForward%[tl.float64, tl.float32, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .fp16, Bool.false =>
    swigluForward%[tl.float64, tl.float16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .fp16, Bool.true =>
    swigluForward%[tl.float64, tl.float16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .bf16, Bool.false =>
    swigluForward%[tl.float64, tl.bfloat16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .bf16, Bool.true =>
    swigluForward%[tl.float64, tl.bfloat16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .fp64, Bool.false =>
    swigluForward%[tl.float64, tl.float64, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .fp64, Bool.true =>
    swigluForward%[tl.float64, tl.float64, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .f8e4, Bool.false =>
    swigluForward%[tl.float64, tl.float8e4nv, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .f8e4, Bool.true =>
    swigluForward%[tl.float64, tl.float8e4nv, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .f8e5, Bool.false =>
    swigluForward%[tl.float64, tl.float8e5, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .fp64, .f8e5, Bool.true =>
    swigluForward%[tl.float64, tl.float8e5, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .fp32, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float32, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .fp32, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float32, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .fp16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .fp16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .bf16, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .bf16, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.bfloat16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .fp64, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float64, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .fp64, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float64, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .f8e4, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .f8e4, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e4nv, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .f8e5, Bool.false =>
    swigluForward%[tl.float8e4nv, tl.float8e5, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .f8e5, Bool.true =>
    swigluForward%[tl.float8e4nv, tl.float8e5, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .fp32, Bool.false =>
    swigluForward%[tl.float8e5, tl.float32, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .fp32, Bool.true =>
    swigluForward%[tl.float8e5, tl.float32, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .fp16, Bool.false =>
    swigluForward%[tl.float8e5, tl.float16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .fp16, Bool.true =>
    swigluForward%[tl.float8e5, tl.float16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .bf16, Bool.false =>
    swigluForward%[tl.float8e5, tl.bfloat16, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .bf16, Bool.true =>
    swigluForward%[tl.float8e5, tl.bfloat16, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .fp64, Bool.false =>
    swigluForward%[tl.float8e5, tl.float64, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .fp64, Bool.true =>
    swigluForward%[tl.float8e5, tl.float64, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .f8e4, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e4nv, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .f8e4, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e4nv, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .f8e5, Bool.false =>
    swigluForward%[tl.float8e5, tl.float8e5, original] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .f8e5, Bool.true =>
    swigluForward%[tl.float8e5, tl.float8e5, commuted] a_ptr b_ptr c_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float32, tl.float32, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float32, tl.float32, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float32, tl.float32, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float32, tl.float32, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float32, tl.float32, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float32, tl.float32, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float32, tl.float32, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float32, tl.float32, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float32, tl.float32, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float32, tl.float32, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float32, tl.float32, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float32, tl.float32, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float32, tl.float16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float32, tl.float16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float32, tl.float16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float32, tl.float16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float32, tl.float16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float32, tl.float16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float32, tl.float16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float32, tl.float16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float32, tl.float16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float32, tl.float16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float32, tl.float16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float32, tl.float16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float32, tl.bfloat16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float32, tl.float64, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float32, tl.float64, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float32, tl.float64, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float32, tl.float64, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float32, tl.float64, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float32, tl.float64, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float32, tl.float64, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float32, tl.float64, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float32, tl.float64, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float32, tl.float64, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float32, tl.float64, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float32, tl.float64, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e4nv, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e5, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e5, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e5, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e5, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e5, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e5, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e5, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e5, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e5, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e5, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float32, tl.float8e5, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float32, tl.float8e5, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float16, tl.float32, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float16, tl.float32, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float16, tl.float32, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float16, tl.float32, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float16, tl.float32, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float16, tl.float32, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float16, tl.float32, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float16, tl.float32, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float16, tl.float32, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float16, tl.float32, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float16, tl.float32, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float16, tl.float32, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float16, tl.float16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float16, tl.float16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float16, tl.float16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float16, tl.float16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float16, tl.float16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float16, tl.float16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float16, tl.float16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float16, tl.float16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float16, tl.float16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float16, tl.float16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float16, tl.float16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float16, tl.float16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float16, tl.bfloat16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float16, tl.float64, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float16, tl.float64, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float16, tl.float64, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float16, tl.float64, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float16, tl.float64, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float16, tl.float64, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float16, tl.float64, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float16, tl.float64, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float16, tl.float64, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float16, tl.float64, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float16, tl.float64, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float16, tl.float64, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e4nv, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e5, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e5, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e5, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e5, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e5, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e5, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e5, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e5, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e5, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e5, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float16, tl.float8e5, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float16, tl.float8e5, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float32, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float64, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.bfloat16, tl.float8e5, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float64, tl.float32, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float64, tl.float32, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float64, tl.float32, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float64, tl.float32, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float64, tl.float32, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float64, tl.float32, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float64, tl.float32, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float64, tl.float32, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float64, tl.float32, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float64, tl.float32, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float64, tl.float32, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float64, tl.float32, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float64, tl.float16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float64, tl.float16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float64, tl.float16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float64, tl.float16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float64, tl.float16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float64, tl.float16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float64, tl.float16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float64, tl.float16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float64, tl.float16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float64, tl.float16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float64, tl.float16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float64, tl.float16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float64, tl.bfloat16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float64, tl.float64, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float64, tl.float64, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float64, tl.float64, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float64, tl.float64, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float64, tl.float64, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float64, tl.float64, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float64, tl.float64, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float64, tl.float64, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float64, tl.float64, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float64, tl.float64, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float64, tl.float64, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float64, tl.float64, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e4nv, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e5, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e5, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e5, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e5, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e5, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e5, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e5, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e5, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e5, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e5, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float64, tl.float8e5, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float64, tl.float8e5, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float32, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float64, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_fp32 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float32, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float32, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float32, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float32, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float32, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float32, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float32, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float32, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float32, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float32, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float32, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float32, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_fp16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_bf16 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e5, tl.bfloat16, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_fp64 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float64, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float64, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float64, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float64, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float64, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float64, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float64, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float64, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float64, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float64, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float64, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float64, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_f8e4 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_f8e5 (gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.float32, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.float32, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.float16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.float16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.bfloat16, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.bfloat16, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.float64, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.float64, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.float8e4nv, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.float8e4nv, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.float8e5, original] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    swigluBackward%[tl.float8e5, tl.float8e5, tl.float8e5, commuted] dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

/-- Complete in-place backward source family with independent input storage. -/
def typedBwd (storage partner gradient : Storage) (commuted : Bool)
    (dc_ptr a_ptr b_ptr : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, partner with
  | .fp32, .fp32 => typedBwd_fp32_fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, .fp16 => typedBwd_fp32_fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, .bf16 => typedBwd_fp32_bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, .fp64 => typedBwd_fp32_fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, .f8e4 => typedBwd_fp32_f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp32, .f8e5 => typedBwd_fp32_f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, .fp32 => typedBwd_fp16_fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, .fp16 => typedBwd_fp16_fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, .bf16 => typedBwd_fp16_bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, .fp64 => typedBwd_fp16_fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, .f8e4 => typedBwd_fp16_f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp16, .f8e5 => typedBwd_fp16_f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, .fp32 => typedBwd_bf16_fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, .fp16 => typedBwd_bf16_fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, .bf16 => typedBwd_bf16_bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, .fp64 => typedBwd_bf16_fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, .f8e4 => typedBwd_bf16_f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .bf16, .f8e5 => typedBwd_bf16_f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, .fp32 => typedBwd_fp64_fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, .fp16 => typedBwd_fp64_fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, .bf16 => typedBwd_fp64_bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, .fp64 => typedBwd_fp64_fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, .f8e4 => typedBwd_fp64_f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .fp64, .f8e5 => typedBwd_fp64_f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .fp32 => typedBwd_f8e4_fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .fp16 => typedBwd_f8e4_fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .bf16 => typedBwd_f8e4_bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .fp64 => typedBwd_f8e4_fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .f8e4 => typedBwd_f8e4_f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e4, .f8e5 => typedBwd_f8e4_f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .fp32 => typedBwd_f8e5_fp32 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .fp16 => typedBwd_f8e5_fp16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .bf16 => typedBwd_f8e5_bf16 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .fp64 => typedBwd_f8e5_fp64 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .f8e4 => typedBwd_f8e5_f8e4 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE
  | .f8e5, .f8e5 => typedBwd_f8e5_f8e5 gradient commuted dc_ptr a_ptr b_ptr stride n_cols BLOCK_SIZE

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SwigluTriton
