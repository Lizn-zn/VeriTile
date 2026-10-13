import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SquareMatrix
open Lean VeriTile.Triton

/-- Faithful 1:1 transcription of `square_matrix.py`'s `square_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def square_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  square_output = row * row
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, square_output, mask=col_offsets < $(n_cols))
}


/-! Storage is explicit in the source-facing family. The candidate commutes
the integer row/stride product before pointer addition; floating multiplication,
negative-infinity padding and both masks are unchanged. -/
syntax "squareSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(squareSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowStart : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| row_start_ptr = input_ptr + row_idx * $(input_row_stride))
      else `(tritonStmt| row_start_ptr = input_ptr + $(input_row_stride) * row_idx)
    `(fun (output_ptr input_ptr : RegionName)
        (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [input_ptr] [output_ptr] (triton {
  tl.region input_ptr = $storage, output_ptr = $storage
  row_idx = tl.program_id(0)
  $rowStart
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  square_output = row * row
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, square_output, mask=col_offsets < $(n_cols))
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

abbrev Source := RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel

def typedKernel (storage : Storage) (commuted : Bool) : Source :=
  match storage, commuted with
  | .fp32, Bool.false => squareSource%[tl.float32, original]
  | .fp32, Bool.true => squareSource%[tl.float32, indexCommuted]
  | .fp16, Bool.false => squareSource%[tl.float16, original]
  | .fp16, Bool.true => squareSource%[tl.float16, indexCommuted]
  | .bf16, Bool.false => squareSource%[tl.bfloat16, original]
  | .bf16, Bool.true => squareSource%[tl.bfloat16, indexCommuted]

end VeriTile.Bench.TritonBenchG.SquareMatrix
