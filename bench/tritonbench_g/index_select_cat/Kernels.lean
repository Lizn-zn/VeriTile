import VeriTile.Triton.DSL
namespace VeriTile.Bench.TritonBenchG.IndexSelectCat
open VeriTile.Triton
open Lean

/-- Faithful transcription of `index_select_cat.py`'s
`index_select_cat_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_INDEX: tl.constexpr` / `BLOCK_SIZE_COL: tl.constexpr`
  → Lean `Nat` parameters.
- `index_ptr` is a typed Lean region so its `tl.load` needs no extra `dtype=`
  kwarg. -/
def index_select_cat_fwd_kernel
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) :
    ComputeKernel := triton {
  pid0 = tl.program_id(axis=0)
  pid1 = tl.program_id(axis=1)
  indices = pid0 * $(BLOCK_SIZE_INDEX) + tl.arange(0, $(BLOCK_SIZE_INDEX))
  rows = tl.load(index_ptr + indices, mask=indices < $(num_indices))
  cols = pid1 * $(BLOCK_SIZE_COL) + tl.arange(0, $(BLOCK_SIZE_COL))
  source_offsets = source_ptr + rows[:, None] * $(stride0) + cols[None, :] * $(stride1)
  mask = (indices[:, None] < $(num_indices)) & (cols[None, :] < $(num_cols))
  output = tl.load(source_offsets, mask=mask)
  output_offsets = output_ptr + indices[:, None] * $(stride0) + cols[None, :] * $(stride1)
  tl.store(output_offsets, output, mask=mask)
}


inductive Storage where
  | fp32 | fp16 | bf16 | f8e4 | f8e5 | fp64
  deriving DecidableEq

inductive IndexStorage where
  | int32 | int64
  deriving DecidableEq

abbrev Source := RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

syntax "indexCat%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(indexCat%[$input:tritonDType, $output:tritonDType, $index:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let indices : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| indices = pid0 * $(BI) + tl.arange(0, $(BI)))
      else `(tritonStmt| indices = $(BI) * pid0 + tl.arange(0, $(BI)))
    let write : TSyntax `tritonStmt ← if input.raw == output.raw then
      `(tritonStmt| tl.store(output_offsets, output, mask=mask))
      else `(tritonStmt| tl.store(output_offsets, output.to($output), mask=mask))
    `(fun (output_ptr index_ptr source_ptr : RegionName)
        (num_indices num_cols stride0 stride1 BI BC : Nat) =>
      ComputeKernel.mk [source_ptr, index_ptr] [output_ptr] (triton {
  tl.region output_ptr = $output, index_ptr = $index, source_ptr = $input
  pid0 = tl.program_id(axis=0)
  pid1 = tl.program_id(axis=1)
  $indices
  rows = tl.load(index_ptr + indices, mask=indices < $(num_indices))
  cols = pid1 * $(BC) + tl.arange(0, $(BC))
  source_offsets = source_ptr + rows[:, None] * $(stride0) + cols[None, :] * $(stride1)
  mask = (indices[:, None] < $(num_indices)) & (cols[None, :] < $(num_cols))
  output = tl.load(source_offsets, mask=mask)
  output_offsets = output_ptr + indices[:, None] * $(stride0) + cols[None, :] * $(stride1)
  $write
}).surfaceBody)

set_option maxHeartbeats 4000000 in
def source (input output : Storage) (index : IndexStorage) (candidate : Bool) : Source :=
  match input, output, index, candidate with
  | .fp32, .fp32, .int32, Bool.false => indexCat%[tl.float32, tl.float32, tl.int32, original]
  | .fp32, .fp32, .int32, Bool.true => indexCat%[tl.float32, tl.float32, tl.int32, indexCommuted]
  | .fp32, .fp32, .int64, Bool.false => indexCat%[tl.float32, tl.float32, tl.int64, original]
  | .fp32, .fp32, .int64, Bool.true => indexCat%[tl.float32, tl.float32, tl.int64, indexCommuted]
  | .fp32, .fp16, .int32, Bool.false => indexCat%[tl.float32, tl.float16, tl.int32, original]
  | .fp32, .fp16, .int32, Bool.true => indexCat%[tl.float32, tl.float16, tl.int32, indexCommuted]
  | .fp32, .fp16, .int64, Bool.false => indexCat%[tl.float32, tl.float16, tl.int64, original]
  | .fp32, .fp16, .int64, Bool.true => indexCat%[tl.float32, tl.float16, tl.int64, indexCommuted]
  | .fp32, .bf16, .int32, Bool.false => indexCat%[tl.float32, tl.bfloat16, tl.int32, original]
  | .fp32, .bf16, .int32, Bool.true => indexCat%[tl.float32, tl.bfloat16, tl.int32, indexCommuted]
  | .fp32, .bf16, .int64, Bool.false => indexCat%[tl.float32, tl.bfloat16, tl.int64, original]
  | .fp32, .bf16, .int64, Bool.true => indexCat%[tl.float32, tl.bfloat16, tl.int64, indexCommuted]
  | .fp32, .f8e4, .int32, Bool.false => indexCat%[tl.float32, tl.float8e4nv, tl.int32, original]
  | .fp32, .f8e4, .int32, Bool.true => indexCat%[tl.float32, tl.float8e4nv, tl.int32, indexCommuted]
  | .fp32, .f8e4, .int64, Bool.false => indexCat%[tl.float32, tl.float8e4nv, tl.int64, original]
  | .fp32, .f8e4, .int64, Bool.true => indexCat%[tl.float32, tl.float8e4nv, tl.int64, indexCommuted]
  | .fp32, .f8e5, .int32, Bool.false => indexCat%[tl.float32, tl.float8e5, tl.int32, original]
  | .fp32, .f8e5, .int32, Bool.true => indexCat%[tl.float32, tl.float8e5, tl.int32, indexCommuted]
  | .fp32, .f8e5, .int64, Bool.false => indexCat%[tl.float32, tl.float8e5, tl.int64, original]
  | .fp32, .f8e5, .int64, Bool.true => indexCat%[tl.float32, tl.float8e5, tl.int64, indexCommuted]
  | .fp32, .fp64, .int32, Bool.false => indexCat%[tl.float32, tl.float64, tl.int32, original]
  | .fp32, .fp64, .int32, Bool.true => indexCat%[tl.float32, tl.float64, tl.int32, indexCommuted]
  | .fp32, .fp64, .int64, Bool.false => indexCat%[tl.float32, tl.float64, tl.int64, original]
  | .fp32, .fp64, .int64, Bool.true => indexCat%[tl.float32, tl.float64, tl.int64, indexCommuted]
  | .fp16, .fp32, .int32, Bool.false => indexCat%[tl.float16, tl.float32, tl.int32, original]
  | .fp16, .fp32, .int32, Bool.true => indexCat%[tl.float16, tl.float32, tl.int32, indexCommuted]
  | .fp16, .fp32, .int64, Bool.false => indexCat%[tl.float16, tl.float32, tl.int64, original]
  | .fp16, .fp32, .int64, Bool.true => indexCat%[tl.float16, tl.float32, tl.int64, indexCommuted]
  | .fp16, .fp16, .int32, Bool.false => indexCat%[tl.float16, tl.float16, tl.int32, original]
  | .fp16, .fp16, .int32, Bool.true => indexCat%[tl.float16, tl.float16, tl.int32, indexCommuted]
  | .fp16, .fp16, .int64, Bool.false => indexCat%[tl.float16, tl.float16, tl.int64, original]
  | .fp16, .fp16, .int64, Bool.true => indexCat%[tl.float16, tl.float16, tl.int64, indexCommuted]
  | .fp16, .bf16, .int32, Bool.false => indexCat%[tl.float16, tl.bfloat16, tl.int32, original]
  | .fp16, .bf16, .int32, Bool.true => indexCat%[tl.float16, tl.bfloat16, tl.int32, indexCommuted]
  | .fp16, .bf16, .int64, Bool.false => indexCat%[tl.float16, tl.bfloat16, tl.int64, original]
  | .fp16, .bf16, .int64, Bool.true => indexCat%[tl.float16, tl.bfloat16, tl.int64, indexCommuted]
  | .fp16, .f8e4, .int32, Bool.false => indexCat%[tl.float16, tl.float8e4nv, tl.int32, original]
  | .fp16, .f8e4, .int32, Bool.true => indexCat%[tl.float16, tl.float8e4nv, tl.int32, indexCommuted]
  | .fp16, .f8e4, .int64, Bool.false => indexCat%[tl.float16, tl.float8e4nv, tl.int64, original]
  | .fp16, .f8e4, .int64, Bool.true => indexCat%[tl.float16, tl.float8e4nv, tl.int64, indexCommuted]
  | .fp16, .f8e5, .int32, Bool.false => indexCat%[tl.float16, tl.float8e5, tl.int32, original]
  | .fp16, .f8e5, .int32, Bool.true => indexCat%[tl.float16, tl.float8e5, tl.int32, indexCommuted]
  | .fp16, .f8e5, .int64, Bool.false => indexCat%[tl.float16, tl.float8e5, tl.int64, original]
  | .fp16, .f8e5, .int64, Bool.true => indexCat%[tl.float16, tl.float8e5, tl.int64, indexCommuted]
  | .fp16, .fp64, .int32, Bool.false => indexCat%[tl.float16, tl.float64, tl.int32, original]
  | .fp16, .fp64, .int32, Bool.true => indexCat%[tl.float16, tl.float64, tl.int32, indexCommuted]
  | .fp16, .fp64, .int64, Bool.false => indexCat%[tl.float16, tl.float64, tl.int64, original]
  | .fp16, .fp64, .int64, Bool.true => indexCat%[tl.float16, tl.float64, tl.int64, indexCommuted]
  | .bf16, .fp32, .int32, Bool.false => indexCat%[tl.bfloat16, tl.float32, tl.int32, original]
  | .bf16, .fp32, .int32, Bool.true => indexCat%[tl.bfloat16, tl.float32, tl.int32, indexCommuted]
  | .bf16, .fp32, .int64, Bool.false => indexCat%[tl.bfloat16, tl.float32, tl.int64, original]
  | .bf16, .fp32, .int64, Bool.true => indexCat%[tl.bfloat16, tl.float32, tl.int64, indexCommuted]
  | .bf16, .fp16, .int32, Bool.false => indexCat%[tl.bfloat16, tl.float16, tl.int32, original]
  | .bf16, .fp16, .int32, Bool.true => indexCat%[tl.bfloat16, tl.float16, tl.int32, indexCommuted]
  | .bf16, .fp16, .int64, Bool.false => indexCat%[tl.bfloat16, tl.float16, tl.int64, original]
  | .bf16, .fp16, .int64, Bool.true => indexCat%[tl.bfloat16, tl.float16, tl.int64, indexCommuted]
  | .bf16, .bf16, .int32, Bool.false => indexCat%[tl.bfloat16, tl.bfloat16, tl.int32, original]
  | .bf16, .bf16, .int32, Bool.true => indexCat%[tl.bfloat16, tl.bfloat16, tl.int32, indexCommuted]
  | .bf16, .bf16, .int64, Bool.false => indexCat%[tl.bfloat16, tl.bfloat16, tl.int64, original]
  | .bf16, .bf16, .int64, Bool.true => indexCat%[tl.bfloat16, tl.bfloat16, tl.int64, indexCommuted]
  | .bf16, .f8e4, .int32, Bool.false => indexCat%[tl.bfloat16, tl.float8e4nv, tl.int32, original]
  | .bf16, .f8e4, .int32, Bool.true => indexCat%[tl.bfloat16, tl.float8e4nv, tl.int32, indexCommuted]
  | .bf16, .f8e4, .int64, Bool.false => indexCat%[tl.bfloat16, tl.float8e4nv, tl.int64, original]
  | .bf16, .f8e4, .int64, Bool.true => indexCat%[tl.bfloat16, tl.float8e4nv, tl.int64, indexCommuted]
  | .bf16, .f8e5, .int32, Bool.false => indexCat%[tl.bfloat16, tl.float8e5, tl.int32, original]
  | .bf16, .f8e5, .int32, Bool.true => indexCat%[tl.bfloat16, tl.float8e5, tl.int32, indexCommuted]
  | .bf16, .f8e5, .int64, Bool.false => indexCat%[tl.bfloat16, tl.float8e5, tl.int64, original]
  | .bf16, .f8e5, .int64, Bool.true => indexCat%[tl.bfloat16, tl.float8e5, tl.int64, indexCommuted]
  | .bf16, .fp64, .int32, Bool.false => indexCat%[tl.bfloat16, tl.float64, tl.int32, original]
  | .bf16, .fp64, .int32, Bool.true => indexCat%[tl.bfloat16, tl.float64, tl.int32, indexCommuted]
  | .bf16, .fp64, .int64, Bool.false => indexCat%[tl.bfloat16, tl.float64, tl.int64, original]
  | .bf16, .fp64, .int64, Bool.true => indexCat%[tl.bfloat16, tl.float64, tl.int64, indexCommuted]
  | .f8e4, .fp32, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.float32, tl.int32, original]
  | .f8e4, .fp32, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.float32, tl.int32, indexCommuted]
  | .f8e4, .fp32, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.float32, tl.int64, original]
  | .f8e4, .fp32, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.float32, tl.int64, indexCommuted]
  | .f8e4, .fp16, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.float16, tl.int32, original]
  | .f8e4, .fp16, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.float16, tl.int32, indexCommuted]
  | .f8e4, .fp16, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.float16, tl.int64, original]
  | .f8e4, .fp16, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.float16, tl.int64, indexCommuted]
  | .f8e4, .bf16, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.bfloat16, tl.int32, original]
  | .f8e4, .bf16, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.bfloat16, tl.int32, indexCommuted]
  | .f8e4, .bf16, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.bfloat16, tl.int64, original]
  | .f8e4, .bf16, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.bfloat16, tl.int64, indexCommuted]
  | .f8e4, .f8e4, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.float8e4nv, tl.int32, original]
  | .f8e4, .f8e4, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.float8e4nv, tl.int32, indexCommuted]
  | .f8e4, .f8e4, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.float8e4nv, tl.int64, original]
  | .f8e4, .f8e4, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.float8e4nv, tl.int64, indexCommuted]
  | .f8e4, .f8e5, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.float8e5, tl.int32, original]
  | .f8e4, .f8e5, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.float8e5, tl.int32, indexCommuted]
  | .f8e4, .f8e5, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.float8e5, tl.int64, original]
  | .f8e4, .f8e5, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.float8e5, tl.int64, indexCommuted]
  | .f8e4, .fp64, .int32, Bool.false => indexCat%[tl.float8e4nv, tl.float64, tl.int32, original]
  | .f8e4, .fp64, .int32, Bool.true => indexCat%[tl.float8e4nv, tl.float64, tl.int32, indexCommuted]
  | .f8e4, .fp64, .int64, Bool.false => indexCat%[tl.float8e4nv, tl.float64, tl.int64, original]
  | .f8e4, .fp64, .int64, Bool.true => indexCat%[tl.float8e4nv, tl.float64, tl.int64, indexCommuted]
  | .f8e5, .fp32, .int32, Bool.false => indexCat%[tl.float8e5, tl.float32, tl.int32, original]
  | .f8e5, .fp32, .int32, Bool.true => indexCat%[tl.float8e5, tl.float32, tl.int32, indexCommuted]
  | .f8e5, .fp32, .int64, Bool.false => indexCat%[tl.float8e5, tl.float32, tl.int64, original]
  | .f8e5, .fp32, .int64, Bool.true => indexCat%[tl.float8e5, tl.float32, tl.int64, indexCommuted]
  | .f8e5, .fp16, .int32, Bool.false => indexCat%[tl.float8e5, tl.float16, tl.int32, original]
  | .f8e5, .fp16, .int32, Bool.true => indexCat%[tl.float8e5, tl.float16, tl.int32, indexCommuted]
  | .f8e5, .fp16, .int64, Bool.false => indexCat%[tl.float8e5, tl.float16, tl.int64, original]
  | .f8e5, .fp16, .int64, Bool.true => indexCat%[tl.float8e5, tl.float16, tl.int64, indexCommuted]
  | .f8e5, .bf16, .int32, Bool.false => indexCat%[tl.float8e5, tl.bfloat16, tl.int32, original]
  | .f8e5, .bf16, .int32, Bool.true => indexCat%[tl.float8e5, tl.bfloat16, tl.int32, indexCommuted]
  | .f8e5, .bf16, .int64, Bool.false => indexCat%[tl.float8e5, tl.bfloat16, tl.int64, original]
  | .f8e5, .bf16, .int64, Bool.true => indexCat%[tl.float8e5, tl.bfloat16, tl.int64, indexCommuted]
  | .f8e5, .f8e4, .int32, Bool.false => indexCat%[tl.float8e5, tl.float8e4nv, tl.int32, original]
  | .f8e5, .f8e4, .int32, Bool.true => indexCat%[tl.float8e5, tl.float8e4nv, tl.int32, indexCommuted]
  | .f8e5, .f8e4, .int64, Bool.false => indexCat%[tl.float8e5, tl.float8e4nv, tl.int64, original]
  | .f8e5, .f8e4, .int64, Bool.true => indexCat%[tl.float8e5, tl.float8e4nv, tl.int64, indexCommuted]
  | .f8e5, .f8e5, .int32, Bool.false => indexCat%[tl.float8e5, tl.float8e5, tl.int32, original]
  | .f8e5, .f8e5, .int32, Bool.true => indexCat%[tl.float8e5, tl.float8e5, tl.int32, indexCommuted]
  | .f8e5, .f8e5, .int64, Bool.false => indexCat%[tl.float8e5, tl.float8e5, tl.int64, original]
  | .f8e5, .f8e5, .int64, Bool.true => indexCat%[tl.float8e5, tl.float8e5, tl.int64, indexCommuted]
  | .f8e5, .fp64, .int32, Bool.false => indexCat%[tl.float8e5, tl.float64, tl.int32, original]
  | .f8e5, .fp64, .int32, Bool.true => indexCat%[tl.float8e5, tl.float64, tl.int32, indexCommuted]
  | .f8e5, .fp64, .int64, Bool.false => indexCat%[tl.float8e5, tl.float64, tl.int64, original]
  | .f8e5, .fp64, .int64, Bool.true => indexCat%[tl.float8e5, tl.float64, tl.int64, indexCommuted]
  | .fp64, .fp32, .int32, Bool.false => indexCat%[tl.float64, tl.float32, tl.int32, original]
  | .fp64, .fp32, .int32, Bool.true => indexCat%[tl.float64, tl.float32, tl.int32, indexCommuted]
  | .fp64, .fp32, .int64, Bool.false => indexCat%[tl.float64, tl.float32, tl.int64, original]
  | .fp64, .fp32, .int64, Bool.true => indexCat%[tl.float64, tl.float32, tl.int64, indexCommuted]
  | .fp64, .fp16, .int32, Bool.false => indexCat%[tl.float64, tl.float16, tl.int32, original]
  | .fp64, .fp16, .int32, Bool.true => indexCat%[tl.float64, tl.float16, tl.int32, indexCommuted]
  | .fp64, .fp16, .int64, Bool.false => indexCat%[tl.float64, tl.float16, tl.int64, original]
  | .fp64, .fp16, .int64, Bool.true => indexCat%[tl.float64, tl.float16, tl.int64, indexCommuted]
  | .fp64, .bf16, .int32, Bool.false => indexCat%[tl.float64, tl.bfloat16, tl.int32, original]
  | .fp64, .bf16, .int32, Bool.true => indexCat%[tl.float64, tl.bfloat16, tl.int32, indexCommuted]
  | .fp64, .bf16, .int64, Bool.false => indexCat%[tl.float64, tl.bfloat16, tl.int64, original]
  | .fp64, .bf16, .int64, Bool.true => indexCat%[tl.float64, tl.bfloat16, tl.int64, indexCommuted]
  | .fp64, .f8e4, .int32, Bool.false => indexCat%[tl.float64, tl.float8e4nv, tl.int32, original]
  | .fp64, .f8e4, .int32, Bool.true => indexCat%[tl.float64, tl.float8e4nv, tl.int32, indexCommuted]
  | .fp64, .f8e4, .int64, Bool.false => indexCat%[tl.float64, tl.float8e4nv, tl.int64, original]
  | .fp64, .f8e4, .int64, Bool.true => indexCat%[tl.float64, tl.float8e4nv, tl.int64, indexCommuted]
  | .fp64, .f8e5, .int32, Bool.false => indexCat%[tl.float64, tl.float8e5, tl.int32, original]
  | .fp64, .f8e5, .int32, Bool.true => indexCat%[tl.float64, tl.float8e5, tl.int32, indexCommuted]
  | .fp64, .f8e5, .int64, Bool.false => indexCat%[tl.float64, tl.float8e5, tl.int64, original]
  | .fp64, .f8e5, .int64, Bool.true => indexCat%[tl.float64, tl.float8e5, tl.int64, indexCommuted]
  | .fp64, .fp64, .int32, Bool.false => indexCat%[tl.float64, tl.float64, tl.int32, original]
  | .fp64, .fp64, .int32, Bool.true => indexCat%[tl.float64, tl.float64, tl.int32, indexCommuted]
  | .fp64, .fp64, .int64, Bool.false => indexCat%[tl.float64, tl.float64, tl.int64, original]
  | .fp64, .fp64, .int64, Bool.true => indexCat%[tl.float64, tl.float64, tl.int64, indexCommuted]

end VeriTile.Bench.TritonBenchG.IndexSelectCat
