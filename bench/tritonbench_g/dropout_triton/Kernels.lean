import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: dropout_triton.py, _dropout.
Complete inverted dropout with a caller-supplied keep buffer. The Python
examples use signed int32; both int32 and bool keep buffers are represented.
Integer conditions are converted by nonzero truthiness inside tl.where.
Data/output storage matches at fp16, bf16 or fp32. Omitted load defaults,
eager branch arithmetic and the masked output store are retained. The
explicit store cast represents Triton's implicit destination conversion.
The candidate only commutes the integer block_start product. -/
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.TypedSource
open Lean VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq
inductive KeepStorage where
  | int32 | bool
  deriving DecidableEq

syntax "dropoutSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(dropoutSource%[$storage:tritonDType, $keep:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let blockStart : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| block_start = $(BLOCK_SIZE) * pid)
      else `(tritonStmt| block_start = pid * $(BLOCK_SIZE))
    `(fun (x_ptr x_keep_ptr output_ptr : RegionName) (n_elements BLOCK_SIZE : Nat) (p : ℝ) =>
      ComputeKernel.mk [x_ptr, x_keep_ptr] [output_ptr] (triton {
  tl.region x_ptr = $storage, x_keep_ptr = $keep, output_ptr = $storage
  pid = tl.program_id(axis=0)
  $blockStart
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(x_ptr + offsets, mask=mask)
  x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
  output = tl.where(x_keep, x / (1 - $(p)), 0.0)
  tl.store(output_ptr + offsets, (output).to($storage:tritonDType), mask=mask)
}).surfaceBody)

abbrev sourceType := RegionName → RegionName → RegionName → Nat → Nat → ℝ → ComputeKernel

def source (storage : Storage) (keep : KeepStorage) (swapped : Bool) : sourceType :=
  match storage, keep, swapped with
  | .fp32, .int32, Bool.false => dropoutSource%[tl.float32, tl.int32, original]
  | .fp32, .int32, Bool.true => dropoutSource%[tl.float32, tl.int32, indexCommuted]
  | .fp16, .int32, Bool.false => dropoutSource%[tl.float16, tl.int32, original]
  | .fp16, .int32, Bool.true => dropoutSource%[tl.float16, tl.int32, indexCommuted]
  | .bf16, .int32, Bool.false => dropoutSource%[tl.bfloat16, tl.int32, original]
  | .bf16, .int32, Bool.true => dropoutSource%[tl.bfloat16, tl.int32, indexCommuted]
  | .fp32, .bool, Bool.false => dropoutSource%[tl.float32, tl.int1, original]
  | .fp32, .bool, Bool.true => dropoutSource%[tl.float32, tl.int1, indexCommuted]
  | .fp16, .bool, Bool.false => dropoutSource%[tl.float16, tl.int1, original]
  | .fp16, .bool, Bool.true => dropoutSource%[tl.float16, tl.int1, indexCommuted]
  | .bf16, .bool, Bool.false => dropoutSource%[tl.bfloat16, tl.int1, original]
  | .bf16, .bool, Bool.true => dropoutSource%[tl.bfloat16, tl.int1, indexCommuted]
end VeriTile.Bench.TritonBenchG.DropoutTriton.TypedSource
