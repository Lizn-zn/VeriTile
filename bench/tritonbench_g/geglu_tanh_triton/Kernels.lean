import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.GegluTanhTriton
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000
open VeriTile.Triton
/-- Faithful transcription of `geglu_tanh_triton.py`'s
`_geglu_tanh_forward_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `n_cols: tl.constexpr` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat`
  parameters.
- Python `from triton.language.extra.libdevice import tanh` is represented by
  the DSL surface function `tanh`. -/
def geglu_tanh_forward_kernel
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  a += program_id * $(stride)
  b += program_id * $(stride)
  c += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  c_row = geglu_a * b_row
  tl.store(c + col_offsets, c_row, mask=mask)
}

/-- Faithful transcription of `geglu_tanh_triton.py`'s
`_geglu_tanh_backward_kernel`.

The Python kernel overwrites `a` and `b` with `da` and `db`; the Lean port keeps
the same region arguments. -/
def geglu_tanh_backward_kernel
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) :
  ComputeKernel := triton {
  program_id = tl.program_id(0).to(tl.int64)
  dc += program_id * $(stride)
  a += program_id * $(stride)
  b += program_id * $(stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  db_row = dc_row * geglu_a
  term1 = 0.5 * (1 + tanh_result)
  tanh_sq = tanh_result * tanh_result
  term2 = 0.5 * a_row * (1 - tanh_sq) *
    (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
  da_row = dc_row * b_row * (term1 + term2)
  tl.store(a + col_offsets, da_row, mask=mask)
  tl.store(b + col_offsets, db_row, mask=mask)
}


/-! Storage-specialized full source. The output conversions spell out Triton's
implicit store conversions; no arithmetic statement is omitted or reordered.
The candidate only commutes the three signed row-base products. -/
open Lean
syntax "gegluForward%" "[" tritonDType "," tritonDType "," ident "]" : term
syntax "gegluBackward%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term

set_option hygiene false in
macro_rules
  | `(gegluForward%[$aType:tritonDType, $bType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let aBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| a += program_id * $(stride))
      else `(tritonStmt| a += $(stride) * program_id)
    let bBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| b += program_id * $(stride))
      else `(tritonStmt| b += $(stride) * program_id)
    let cBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| c += program_id * $(stride))
      else `(tritonStmt| c += $(stride) * program_id)
    let wideB := match bType with
      | `(tritonDType| tl.float64) => Bool.true
      | _ => Bool.false
    let cWrite : TSyntax `tritonStmt ← match aType with
      | `(tritonDType| tl.float32) =>
        if wideB then
          `(tritonStmt| tl.store(c + col_offsets, c_row.to($aType), mask=mask))
        else `(tritonStmt| tl.store(c + col_offsets, c_row, mask=mask))
      | `(tritonDType| tl.float64) =>
        if wideB then
          `(tritonStmt| tl.store(c + col_offsets, c_row, mask=mask))
        else `(tritonStmt| tl.store(c + col_offsets, c_row.to($aType), mask=mask))
      | _ => `(tritonStmt| tl.store(c + col_offsets, c_row.to($aType), mask=mask))
    `(fun (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [a, b] [c] (triton {
  tl.region a = $aType, b = $bType, c = $aType
  program_id = tl.program_id(0).to(tl.int64)
  $aBase
  $bBase
  $cBase
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = libdevice.tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  c_row = geglu_a * b_row
  $cWrite
}).surfaceBody)

set_option hygiene false in
macro_rules
  | `(gegluBackward%[$aType:tritonDType, $bType:tritonDType, $dcType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let dcBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| dc += program_id * $(stride))
      else `(tritonStmt| dc += $(stride) * program_id)
    let aBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| a += program_id * $(stride))
      else `(tritonStmt| a += $(stride) * program_id)
    let bBase : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| b += program_id * $(stride))
      else `(tritonStmt| b += $(stride) * program_id)
    let wideB := match bType with
      | `(tritonDType| tl.float64) => Bool.true
      | _ => Bool.false
    let wideDC := match dcType with
      | `(tritonDType| tl.float64) => Bool.true
      | _ => Bool.false
    let aWrite : TSyntax `tritonStmt ← match aType with
      | `(tritonDType| tl.float32) =>
        if wideB || wideDC then
          `(tritonStmt| tl.store(a + col_offsets, da_row.to($aType), mask=mask))
        else `(tritonStmt| tl.store(a + col_offsets, da_row, mask=mask))
      | `(tritonDType| tl.float64) =>
        if wideB || wideDC then
          `(tritonStmt| tl.store(a + col_offsets, da_row, mask=mask))
        else `(tritonStmt| tl.store(a + col_offsets, da_row.to($aType), mask=mask))
      | _ => `(tritonStmt| tl.store(a + col_offsets, da_row.to($aType), mask=mask))
    let bWrite : TSyntax `tritonStmt ← match bType with
      | `(tritonDType| tl.float32) =>
        if wideDC then
          `(tritonStmt| tl.store(b + col_offsets, db_row.to($bType), mask=mask))
        else `(tritonStmt| tl.store(b + col_offsets, db_row, mask=mask))
      | `(tritonDType| tl.float64) =>
        if wideDC then
          `(tritonStmt| tl.store(b + col_offsets, db_row, mask=mask))
        else `(tritonStmt| tl.store(b + col_offsets, db_row.to($bType), mask=mask))
      | _ => `(tritonStmt| tl.store(b + col_offsets, db_row.to($bType), mask=mask))
    `(fun (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [dc, a, b] [a, b] (triton {
  tl.region dc = $dcType, a = $aType, b = $bType
  program_id = tl.program_id(0).to(tl.int64)
  $dcBase
  $aBase
  $bBase
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
  a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0)
  sqrt_2_over_pi = 0.7978845608028654
  a_cubed = a_row * a_row * a_row
  tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
  tanh_result = libdevice.tanh(tanh_arg)
  geglu_a = 0.5 * a_row * (1 + tanh_result)
  db_row = dc_row * geglu_a
  term1 = 0.5 * (1 + tanh_result)
  tanh_sq = tanh_result * tanh_result
  term2 = 0.5 * a_row * (1 - tanh_sq) *
    (sqrt_2_over_pi * (1 + 0.134145 * a_row * a_row))
  da_row = dc_row * b_row * (term1 + term2)
  $aWrite
  $bWrite
}).surfaceBody)


/-- Floating storage choices; output A/C and backward B follow their input buffers. -/
inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr



/-- Complete forward source family with independent input storage. -/
def typedFwd (storage partner : Storage) (commuted : Bool)
    (a b c : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, partner, commuted with
  | .fp32, .fp32, Bool.false =>
    gegluForward%[tl.float32, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .fp32, Bool.true =>
    gegluForward%[tl.float32, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp32, .fp16, Bool.false =>
    gegluForward%[tl.float32, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .fp16, Bool.true =>
    gegluForward%[tl.float32, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp32, .bf16, Bool.false =>
    gegluForward%[tl.float32, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .bf16, Bool.true =>
    gegluForward%[tl.float32, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp32, .fp64, Bool.false =>
    gegluForward%[tl.float32, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .fp64, Bool.true =>
    gegluForward%[tl.float32, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp32, .f8e4, Bool.false =>
    gegluForward%[tl.float32, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .f8e4, Bool.true =>
    gegluForward%[tl.float32, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp32, .f8e5, Bool.false =>
    gegluForward%[tl.float32, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .fp32, .f8e5, Bool.true =>
    gegluForward%[tl.float32, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp32, Bool.false =>
    gegluForward%[tl.float16, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp32, Bool.true =>
    gegluForward%[tl.float16, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp16, Bool.false =>
    gegluForward%[tl.float16, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp16, Bool.true =>
    gegluForward%[tl.float16, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .bf16, Bool.false =>
    gegluForward%[tl.float16, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .bf16, Bool.true =>
    gegluForward%[tl.float16, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp64, Bool.false =>
    gegluForward%[tl.float16, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .fp64, Bool.true =>
    gegluForward%[tl.float16, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .f8e4, Bool.false =>
    gegluForward%[tl.float16, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .f8e4, Bool.true =>
    gegluForward%[tl.float16, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp16, .f8e5, Bool.false =>
    gegluForward%[tl.float16, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .fp16, .f8e5, Bool.true =>
    gegluForward%[tl.float16, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp32, Bool.false =>
    gegluForward%[tl.bfloat16, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp32, Bool.true =>
    gegluForward%[tl.bfloat16, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp16, Bool.false =>
    gegluForward%[tl.bfloat16, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp16, Bool.true =>
    gegluForward%[tl.bfloat16, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .bf16, Bool.false =>
    gegluForward%[tl.bfloat16, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .bf16, Bool.true =>
    gegluForward%[tl.bfloat16, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp64, Bool.false =>
    gegluForward%[tl.bfloat16, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .fp64, Bool.true =>
    gegluForward%[tl.bfloat16, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .f8e4, Bool.false =>
    gegluForward%[tl.bfloat16, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .f8e4, Bool.true =>
    gegluForward%[tl.bfloat16, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .bf16, .f8e5, Bool.false =>
    gegluForward%[tl.bfloat16, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .bf16, .f8e5, Bool.true =>
    gegluForward%[tl.bfloat16, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp32, Bool.false =>
    gegluForward%[tl.float64, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp32, Bool.true =>
    gegluForward%[tl.float64, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp16, Bool.false =>
    gegluForward%[tl.float64, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp16, Bool.true =>
    gegluForward%[tl.float64, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .bf16, Bool.false =>
    gegluForward%[tl.float64, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .bf16, Bool.true =>
    gegluForward%[tl.float64, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp64, Bool.false =>
    gegluForward%[tl.float64, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .fp64, Bool.true =>
    gegluForward%[tl.float64, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .f8e4, Bool.false =>
    gegluForward%[tl.float64, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .f8e4, Bool.true =>
    gegluForward%[tl.float64, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .fp64, .f8e5, Bool.false =>
    gegluForward%[tl.float64, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .fp64, .f8e5, Bool.true =>
    gegluForward%[tl.float64, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp32, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp32, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp16, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp16, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .bf16, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .bf16, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp64, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .fp64, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .f8e4, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .f8e4, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .f8e5, Bool.false =>
    gegluForward%[tl.float8e4nv, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .f8e4, .f8e5, Bool.true =>
    gegluForward%[tl.float8e4nv, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp32, Bool.false =>
    gegluForward%[tl.float8e5, tl.float32, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp32, Bool.true =>
    gegluForward%[tl.float8e5, tl.float32, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp16, Bool.false =>
    gegluForward%[tl.float8e5, tl.float16, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp16, Bool.true =>
    gegluForward%[tl.float8e5, tl.float16, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .bf16, Bool.false =>
    gegluForward%[tl.float8e5, tl.bfloat16, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .bf16, Bool.true =>
    gegluForward%[tl.float8e5, tl.bfloat16, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp64, Bool.false =>
    gegluForward%[tl.float8e5, tl.float64, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .fp64, Bool.true =>
    gegluForward%[tl.float8e5, tl.float64, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .f8e4, Bool.false =>
    gegluForward%[tl.float8e5, tl.float8e4nv, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .f8e4, Bool.true =>
    gegluForward%[tl.float8e5, tl.float8e4nv, commuted] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .f8e5, Bool.false =>
    gegluForward%[tl.float8e5, tl.float8e5, original] a b c stride n_cols BLOCK_SIZE
  | .f8e5, .f8e5, Bool.true =>
    gegluForward%[tl.float8e5, tl.float8e5, commuted] a b c stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp32_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float32, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp16_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float16, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_bf16_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.bfloat16, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_fp64_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float64, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e4_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e4nv, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_fp32 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float32, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_fp16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_bf16 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.bfloat16, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_fp64 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float64, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_f8e4 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e4nv, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Source specialization with fixed A/B storage and independent gradient storage. -/
def typedBwd_f8e5_f8e5 (gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match gradient, commuted with
  | .fp32, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float32, original] dc a b stride n_cols BLOCK_SIZE
  | .fp32, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float32, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float16, original] dc a b stride n_cols BLOCK_SIZE
  | .fp16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.bfloat16, original] dc a b stride n_cols BLOCK_SIZE
  | .bf16, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.bfloat16, commuted] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float64, original] dc a b stride n_cols BLOCK_SIZE
  | .fp64, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float64, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float8e4nv, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e4, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float8e4nv, commuted] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.false =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float8e5, original] dc a b stride n_cols BLOCK_SIZE
  | .f8e5, Bool.true =>
    gegluBackward%[tl.float8e5, tl.float8e5, tl.float8e5, commuted] dc a b stride n_cols BLOCK_SIZE

/-- Complete in-place backward source family with independent input storage. -/
def typedBwd (storage partner gradient : Storage) (commuted : Bool)
    (dc a b : RegionName) (stride n_cols BLOCK_SIZE : Nat) : ComputeKernel :=
  match storage, partner with
  | .fp32, .fp32 => typedBwd_fp32_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp32, .fp16 => typedBwd_fp32_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp32, .bf16 => typedBwd_fp32_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp32, .fp64 => typedBwd_fp32_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp32, .f8e4 => typedBwd_fp32_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp32, .f8e5 => typedBwd_fp32_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .fp32 => typedBwd_fp16_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .fp16 => typedBwd_fp16_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .bf16 => typedBwd_fp16_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .fp64 => typedBwd_fp16_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .f8e4 => typedBwd_fp16_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp16, .f8e5 => typedBwd_fp16_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .fp32 => typedBwd_bf16_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .fp16 => typedBwd_bf16_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .bf16 => typedBwd_bf16_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .fp64 => typedBwd_bf16_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .f8e4 => typedBwd_bf16_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .bf16, .f8e5 => typedBwd_bf16_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .fp32 => typedBwd_fp64_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .fp16 => typedBwd_fp64_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .bf16 => typedBwd_fp64_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .fp64 => typedBwd_fp64_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .f8e4 => typedBwd_fp64_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .fp64, .f8e5 => typedBwd_fp64_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .fp32 => typedBwd_f8e4_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .fp16 => typedBwd_f8e4_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .bf16 => typedBwd_f8e4_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .fp64 => typedBwd_f8e4_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .f8e4 => typedBwd_f8e4_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e4, .f8e5 => typedBwd_f8e4_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .fp32 => typedBwd_f8e5_fp32 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .fp16 => typedBwd_f8e5_fp16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .bf16 => typedBwd_f8e5_bf16 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .fp64 => typedBwd_f8e5_fp64 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .f8e4 => typedBwd_f8e5_f8e4 gradient commuted dc a b stride n_cols BLOCK_SIZE
  | .f8e5, .f8e5 => typedBwd_f8e5_f8e5 gradient commuted dc a b stride n_cols BLOCK_SIZE

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.GegluTanhTriton
