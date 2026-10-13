import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.IndexSelectBwd

open VeriTile.Triton
open Lean

/-- Faithful transcription of `index_select_bwd.py`'s
`index_select_cat_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_INDEX: tl.constexpr` / `BLOCK_SIZE_COL: tl.constexpr`
  -> Lean `Nat` parameters.
- Python `.to(tl.float32)` on `grad_output` is represented explicitly in the
  Compute layer; the algorithm-layer theorem observes its Real projection. -/
def index_select_cat_bwd_kernel
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (_num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) :
    ComputeKernel := triton {
  pid0 = tl.program_id(axis=0)
  pid1 = tl.program_id(axis=1)
  cols = pid1 * $(BLOCK_SIZE_COL) + tl.arange(0, $(BLOCK_SIZE_COL))
  grad_output_indices = pid0 * $(BLOCK_SIZE_INDEX) + tl.arange(0, $(BLOCK_SIZE_INDEX))
  grad_output_offsets =
    grad_output_ptr + grad_output_indices[:, None] * $(stride0) +
      cols[None, :] * $(stride1)
  grad_output_mask =
    (grad_output_indices[:, None] < $(num_indices)) &
      (cols[None, :] < $(num_cols))
  grad_output = (tl.load(grad_output_offsets, mask=grad_output_mask)).to(tl.float32)
  grad_source_indices =
    tl.load(index_ptr + grad_output_indices,
      mask=grad_output_indices < $(num_indices))
  grad_source_offsets =
    grad_source_ptr + grad_source_indices[:, None] * $(stride0) +
      cols[None, :] * $(stride1)
  tl.store(grad_source_offsets, grad_output, mask=grad_output_mask)
}



inductive Storage where
  | fp32 | fp16 | bf16 | f8e4 | f8e5 | fp64
  deriving DecidableEq

inductive IndexStorage where
  | int32 | int64
  deriving DecidableEq

abbrev Source := RegionName → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

syntax "indexBwd%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(indexBwd%[$input:tritonDType, $output:tritonDType, $index:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let cols : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| cols = pid1 * $(BC) + tl.arange(0, $(BC)))
      else `(tritonStmt| cols = $(BC) * pid1 + tl.arange(0, $(BC)))
    let write : TSyntax `tritonStmt ← match output with
      | `(tritonDType| tl.float32) =>
        `(tritonStmt| tl.store(grad_source_offsets, grad_output, mask=grad_output_mask))
      | _ => `(tritonStmt| tl.store(grad_source_offsets, grad_output.to($output), mask=grad_output_mask))
    `(fun (grad_source_ptr index_ptr grad_output_ptr : RegionName)
        (_num_rows num_indices num_cols stride0 stride1 BI BC : Nat) =>
      ComputeKernel.mk [grad_output_ptr, index_ptr] [grad_source_ptr] (triton {
  tl.region grad_source_ptr = $output, index_ptr = $index, grad_output_ptr = $input
  pid0 = tl.program_id(axis=0)
  pid1 = tl.program_id(axis=1)
  $cols
  grad_output_indices = pid0 * $(BI) + tl.arange(0, $(BI))
  grad_output_offsets = grad_output_ptr + grad_output_indices[:, None] * $(stride0) + cols[None, :] * $(stride1)
  grad_output_mask = (grad_output_indices[:, None] < $(num_indices)) & (cols[None, :] < $(num_cols))
  grad_output = tl.load(grad_output_offsets, mask=grad_output_mask).to(tl.float32)
  grad_source_indices = tl.load(index_ptr + grad_output_indices, mask=grad_output_indices < $(num_indices))
  grad_source_offsets = grad_source_ptr + grad_source_indices[:, None] * $(stride0) + cols[None, :] * $(stride1)
  $write
}).surfaceBody)

set_option maxHeartbeats 4000000 in
def source (input output : Storage) (index : IndexStorage) (candidate : Bool) : Source :=
  match input, output, index, candidate with
  | .fp32, .fp32, .int32, Bool.false => indexBwd%[tl.float32, tl.float32, tl.int32, original]
  | .fp32, .fp32, .int32, Bool.true => indexBwd%[tl.float32, tl.float32, tl.int32, indexCommuted]
  | .fp32, .fp32, .int64, Bool.false => indexBwd%[tl.float32, tl.float32, tl.int64, original]
  | .fp32, .fp32, .int64, Bool.true => indexBwd%[tl.float32, tl.float32, tl.int64, indexCommuted]
  | .fp32, .fp16, .int32, Bool.false => indexBwd%[tl.float32, tl.float16, tl.int32, original]
  | .fp32, .fp16, .int32, Bool.true => indexBwd%[tl.float32, tl.float16, tl.int32, indexCommuted]
  | .fp32, .fp16, .int64, Bool.false => indexBwd%[tl.float32, tl.float16, tl.int64, original]
  | .fp32, .fp16, .int64, Bool.true => indexBwd%[tl.float32, tl.float16, tl.int64, indexCommuted]
  | .fp32, .bf16, .int32, Bool.false => indexBwd%[tl.float32, tl.bfloat16, tl.int32, original]
  | .fp32, .bf16, .int32, Bool.true => indexBwd%[tl.float32, tl.bfloat16, tl.int32, indexCommuted]
  | .fp32, .bf16, .int64, Bool.false => indexBwd%[tl.float32, tl.bfloat16, tl.int64, original]
  | .fp32, .bf16, .int64, Bool.true => indexBwd%[tl.float32, tl.bfloat16, tl.int64, indexCommuted]
  | .fp32, .f8e4, .int32, Bool.false => indexBwd%[tl.float32, tl.float8e4nv, tl.int32, original]
  | .fp32, .f8e4, .int32, Bool.true => indexBwd%[tl.float32, tl.float8e4nv, tl.int32, indexCommuted]
  | .fp32, .f8e4, .int64, Bool.false => indexBwd%[tl.float32, tl.float8e4nv, tl.int64, original]
  | .fp32, .f8e4, .int64, Bool.true => indexBwd%[tl.float32, tl.float8e4nv, tl.int64, indexCommuted]
  | .fp32, .f8e5, .int32, Bool.false => indexBwd%[tl.float32, tl.float8e5, tl.int32, original]
  | .fp32, .f8e5, .int32, Bool.true => indexBwd%[tl.float32, tl.float8e5, tl.int32, indexCommuted]
  | .fp32, .f8e5, .int64, Bool.false => indexBwd%[tl.float32, tl.float8e5, tl.int64, original]
  | .fp32, .f8e5, .int64, Bool.true => indexBwd%[tl.float32, tl.float8e5, tl.int64, indexCommuted]
  | .fp32, .fp64, .int32, Bool.false => indexBwd%[tl.float32, tl.float64, tl.int32, original]
  | .fp32, .fp64, .int32, Bool.true => indexBwd%[tl.float32, tl.float64, tl.int32, indexCommuted]
  | .fp32, .fp64, .int64, Bool.false => indexBwd%[tl.float32, tl.float64, tl.int64, original]
  | .fp32, .fp64, .int64, Bool.true => indexBwd%[tl.float32, tl.float64, tl.int64, indexCommuted]
  | .fp16, .fp32, .int32, Bool.false => indexBwd%[tl.float16, tl.float32, tl.int32, original]
  | .fp16, .fp32, .int32, Bool.true => indexBwd%[tl.float16, tl.float32, tl.int32, indexCommuted]
  | .fp16, .fp32, .int64, Bool.false => indexBwd%[tl.float16, tl.float32, tl.int64, original]
  | .fp16, .fp32, .int64, Bool.true => indexBwd%[tl.float16, tl.float32, tl.int64, indexCommuted]
  | .fp16, .fp16, .int32, Bool.false => indexBwd%[tl.float16, tl.float16, tl.int32, original]
  | .fp16, .fp16, .int32, Bool.true => indexBwd%[tl.float16, tl.float16, tl.int32, indexCommuted]
  | .fp16, .fp16, .int64, Bool.false => indexBwd%[tl.float16, tl.float16, tl.int64, original]
  | .fp16, .fp16, .int64, Bool.true => indexBwd%[tl.float16, tl.float16, tl.int64, indexCommuted]
  | .fp16, .bf16, .int32, Bool.false => indexBwd%[tl.float16, tl.bfloat16, tl.int32, original]
  | .fp16, .bf16, .int32, Bool.true => indexBwd%[tl.float16, tl.bfloat16, tl.int32, indexCommuted]
  | .fp16, .bf16, .int64, Bool.false => indexBwd%[tl.float16, tl.bfloat16, tl.int64, original]
  | .fp16, .bf16, .int64, Bool.true => indexBwd%[tl.float16, tl.bfloat16, tl.int64, indexCommuted]
  | .fp16, .f8e4, .int32, Bool.false => indexBwd%[tl.float16, tl.float8e4nv, tl.int32, original]
  | .fp16, .f8e4, .int32, Bool.true => indexBwd%[tl.float16, tl.float8e4nv, tl.int32, indexCommuted]
  | .fp16, .f8e4, .int64, Bool.false => indexBwd%[tl.float16, tl.float8e4nv, tl.int64, original]
  | .fp16, .f8e4, .int64, Bool.true => indexBwd%[tl.float16, tl.float8e4nv, tl.int64, indexCommuted]
  | .fp16, .f8e5, .int32, Bool.false => indexBwd%[tl.float16, tl.float8e5, tl.int32, original]
  | .fp16, .f8e5, .int32, Bool.true => indexBwd%[tl.float16, tl.float8e5, tl.int32, indexCommuted]
  | .fp16, .f8e5, .int64, Bool.false => indexBwd%[tl.float16, tl.float8e5, tl.int64, original]
  | .fp16, .f8e5, .int64, Bool.true => indexBwd%[tl.float16, tl.float8e5, tl.int64, indexCommuted]
  | .fp16, .fp64, .int32, Bool.false => indexBwd%[tl.float16, tl.float64, tl.int32, original]
  | .fp16, .fp64, .int32, Bool.true => indexBwd%[tl.float16, tl.float64, tl.int32, indexCommuted]
  | .fp16, .fp64, .int64, Bool.false => indexBwd%[tl.float16, tl.float64, tl.int64, original]
  | .fp16, .fp64, .int64, Bool.true => indexBwd%[tl.float16, tl.float64, tl.int64, indexCommuted]
  | .bf16, .fp32, .int32, Bool.false => indexBwd%[tl.bfloat16, tl.float32, tl.int32, original]
  | .bf16, .fp32, .int32, Bool.true => indexBwd%[tl.bfloat16, tl.float32, tl.int32, indexCommuted]
  | .bf16, .fp32, .int64, Bool.false => indexBwd%[tl.bfloat16, tl.float32, tl.int64, original]
  | .bf16, .fp32, .int64, Bool.true => indexBwd%[tl.bfloat16, tl.float32, tl.int64, indexCommuted]
  | .bf16, .fp16, .int32, Bool.false => indexBwd%[tl.bfloat16, tl.float16, tl.int32, original]
  | .bf16, .fp16, .int32, Bool.true => indexBwd%[tl.bfloat16, tl.float16, tl.int32, indexCommuted]
  | .bf16, .fp16, .int64, Bool.false => indexBwd%[tl.bfloat16, tl.float16, tl.int64, original]
  | .bf16, .fp16, .int64, Bool.true => indexBwd%[tl.bfloat16, tl.float16, tl.int64, indexCommuted]
  | .bf16, .bf16, .int32, Bool.false => indexBwd%[tl.bfloat16, tl.bfloat16, tl.int32, original]
  | .bf16, .bf16, .int32, Bool.true => indexBwd%[tl.bfloat16, tl.bfloat16, tl.int32, indexCommuted]
  | .bf16, .bf16, .int64, Bool.false => indexBwd%[tl.bfloat16, tl.bfloat16, tl.int64, original]
  | .bf16, .bf16, .int64, Bool.true => indexBwd%[tl.bfloat16, tl.bfloat16, tl.int64, indexCommuted]
  | .bf16, .f8e4, .int32, Bool.false => indexBwd%[tl.bfloat16, tl.float8e4nv, tl.int32, original]
  | .bf16, .f8e4, .int32, Bool.true => indexBwd%[tl.bfloat16, tl.float8e4nv, tl.int32, indexCommuted]
  | .bf16, .f8e4, .int64, Bool.false => indexBwd%[tl.bfloat16, tl.float8e4nv, tl.int64, original]
  | .bf16, .f8e4, .int64, Bool.true => indexBwd%[tl.bfloat16, tl.float8e4nv, tl.int64, indexCommuted]
  | .bf16, .f8e5, .int32, Bool.false => indexBwd%[tl.bfloat16, tl.float8e5, tl.int32, original]
  | .bf16, .f8e5, .int32, Bool.true => indexBwd%[tl.bfloat16, tl.float8e5, tl.int32, indexCommuted]
  | .bf16, .f8e5, .int64, Bool.false => indexBwd%[tl.bfloat16, tl.float8e5, tl.int64, original]
  | .bf16, .f8e5, .int64, Bool.true => indexBwd%[tl.bfloat16, tl.float8e5, tl.int64, indexCommuted]
  | .bf16, .fp64, .int32, Bool.false => indexBwd%[tl.bfloat16, tl.float64, tl.int32, original]
  | .bf16, .fp64, .int32, Bool.true => indexBwd%[tl.bfloat16, tl.float64, tl.int32, indexCommuted]
  | .bf16, .fp64, .int64, Bool.false => indexBwd%[tl.bfloat16, tl.float64, tl.int64, original]
  | .bf16, .fp64, .int64, Bool.true => indexBwd%[tl.bfloat16, tl.float64, tl.int64, indexCommuted]
  | .f8e4, .fp32, .int32, Bool.false => indexBwd%[tl.float8e4nv, tl.float32, tl.int32, original]
  | .f8e4, .fp32, .int32, Bool.true => indexBwd%[tl.float8e4nv, tl.float32, tl.int32, indexCommuted]
  | .f8e4, .fp32, .int64, Bool.false => indexBwd%[tl.float8e4nv, tl.float32, tl.int64, original]
  | .f8e4, .fp32, .int64, Bool.true => indexBwd%[tl.float8e4nv, tl.float32, tl.int64, indexCommuted]
  | .f8e4, .fp16, .int32, Bool.false => indexBwd%[tl.float8e4nv, tl.float16, tl.int32, original]
  | .f8e4, .fp16, .int32, Bool.true => indexBwd%[tl.float8e4nv, tl.float16, tl.int32, indexCommuted]
  | .f8e4, .fp16, .int64, Bool.false => indexBwd%[tl.float8e4nv, tl.float16, tl.int64, original]
  | .f8e4, .fp16, .int64, Bool.true => indexBwd%[tl.float8e4nv, tl.float16, tl.int64, indexCommuted]
  | .f8e4, .bf16, .int32, Bool.false => indexBwd%[tl.float8e4nv, tl.bfloat16, tl.int32, original]
  | .f8e4, .bf16, .int32, Bool.true => indexBwd%[tl.float8e4nv, tl.bfloat16, tl.int32, indexCommuted]
  | .f8e4, .bf16, .int64, Bool.false => indexBwd%[tl.float8e4nv, tl.bfloat16, tl.int64, original]
  | .f8e4, .bf16, .int64, Bool.true => indexBwd%[tl.float8e4nv, tl.bfloat16, tl.int64, indexCommuted]
  | .f8e4, .f8e4, .int32, Bool.false => indexBwd%[tl.float8e4nv, tl.float8e4nv, tl.int32, original]
  | .f8e4, .f8e4, .int32, Bool.true => indexBwd%[tl.float8e4nv, tl.float8e4nv, tl.int32, indexCommuted]
  | .f8e4, .f8e4, .int64, Bool.false => indexBwd%[tl.float8e4nv, tl.float8e4nv, tl.int64, original]
  | .f8e4, .f8e4, .int64, Bool.true => indexBwd%[tl.float8e4nv, tl.float8e4nv, tl.int64, indexCommuted]
  | .f8e4, .f8e5, .int32, Bool.false => indexBwd%[tl.float8e4nv, tl.float8e5, tl.int32, original]
  | .f8e4, .f8e5, .int32, Bool.true => indexBwd%[tl.float8e4nv, tl.float8e5, tl.int32, indexCommuted]
  | .f8e4, .f8e5, .int64, Bool.false => indexBwd%[tl.float8e4nv, tl.float8e5, tl.int64, original]
  | .f8e4, .f8e5, .int64, Bool.true => indexBwd%[tl.float8e4nv, tl.float8e5, tl.int64, indexCommuted]
  | .f8e4, .fp64, .int32, Bool.false => indexBwd%[tl.float8e4nv, tl.float64, tl.int32, original]
  | .f8e4, .fp64, .int32, Bool.true => indexBwd%[tl.float8e4nv, tl.float64, tl.int32, indexCommuted]
  | .f8e4, .fp64, .int64, Bool.false => indexBwd%[tl.float8e4nv, tl.float64, tl.int64, original]
  | .f8e4, .fp64, .int64, Bool.true => indexBwd%[tl.float8e4nv, tl.float64, tl.int64, indexCommuted]
  | .f8e5, .fp32, .int32, Bool.false => indexBwd%[tl.float8e5, tl.float32, tl.int32, original]
  | .f8e5, .fp32, .int32, Bool.true => indexBwd%[tl.float8e5, tl.float32, tl.int32, indexCommuted]
  | .f8e5, .fp32, .int64, Bool.false => indexBwd%[tl.float8e5, tl.float32, tl.int64, original]
  | .f8e5, .fp32, .int64, Bool.true => indexBwd%[tl.float8e5, tl.float32, tl.int64, indexCommuted]
  | .f8e5, .fp16, .int32, Bool.false => indexBwd%[tl.float8e5, tl.float16, tl.int32, original]
  | .f8e5, .fp16, .int32, Bool.true => indexBwd%[tl.float8e5, tl.float16, tl.int32, indexCommuted]
  | .f8e5, .fp16, .int64, Bool.false => indexBwd%[tl.float8e5, tl.float16, tl.int64, original]
  | .f8e5, .fp16, .int64, Bool.true => indexBwd%[tl.float8e5, tl.float16, tl.int64, indexCommuted]
  | .f8e5, .bf16, .int32, Bool.false => indexBwd%[tl.float8e5, tl.bfloat16, tl.int32, original]
  | .f8e5, .bf16, .int32, Bool.true => indexBwd%[tl.float8e5, tl.bfloat16, tl.int32, indexCommuted]
  | .f8e5, .bf16, .int64, Bool.false => indexBwd%[tl.float8e5, tl.bfloat16, tl.int64, original]
  | .f8e5, .bf16, .int64, Bool.true => indexBwd%[tl.float8e5, tl.bfloat16, tl.int64, indexCommuted]
  | .f8e5, .f8e4, .int32, Bool.false => indexBwd%[tl.float8e5, tl.float8e4nv, tl.int32, original]
  | .f8e5, .f8e4, .int32, Bool.true => indexBwd%[tl.float8e5, tl.float8e4nv, tl.int32, indexCommuted]
  | .f8e5, .f8e4, .int64, Bool.false => indexBwd%[tl.float8e5, tl.float8e4nv, tl.int64, original]
  | .f8e5, .f8e4, .int64, Bool.true => indexBwd%[tl.float8e5, tl.float8e4nv, tl.int64, indexCommuted]
  | .f8e5, .f8e5, .int32, Bool.false => indexBwd%[tl.float8e5, tl.float8e5, tl.int32, original]
  | .f8e5, .f8e5, .int32, Bool.true => indexBwd%[tl.float8e5, tl.float8e5, tl.int32, indexCommuted]
  | .f8e5, .f8e5, .int64, Bool.false => indexBwd%[tl.float8e5, tl.float8e5, tl.int64, original]
  | .f8e5, .f8e5, .int64, Bool.true => indexBwd%[tl.float8e5, tl.float8e5, tl.int64, indexCommuted]
  | .f8e5, .fp64, .int32, Bool.false => indexBwd%[tl.float8e5, tl.float64, tl.int32, original]
  | .f8e5, .fp64, .int32, Bool.true => indexBwd%[tl.float8e5, tl.float64, tl.int32, indexCommuted]
  | .f8e5, .fp64, .int64, Bool.false => indexBwd%[tl.float8e5, tl.float64, tl.int64, original]
  | .f8e5, .fp64, .int64, Bool.true => indexBwd%[tl.float8e5, tl.float64, tl.int64, indexCommuted]
  | .fp64, .fp32, .int32, Bool.false => indexBwd%[tl.float64, tl.float32, tl.int32, original]
  | .fp64, .fp32, .int32, Bool.true => indexBwd%[tl.float64, tl.float32, tl.int32, indexCommuted]
  | .fp64, .fp32, .int64, Bool.false => indexBwd%[tl.float64, tl.float32, tl.int64, original]
  | .fp64, .fp32, .int64, Bool.true => indexBwd%[tl.float64, tl.float32, tl.int64, indexCommuted]
  | .fp64, .fp16, .int32, Bool.false => indexBwd%[tl.float64, tl.float16, tl.int32, original]
  | .fp64, .fp16, .int32, Bool.true => indexBwd%[tl.float64, tl.float16, tl.int32, indexCommuted]
  | .fp64, .fp16, .int64, Bool.false => indexBwd%[tl.float64, tl.float16, tl.int64, original]
  | .fp64, .fp16, .int64, Bool.true => indexBwd%[tl.float64, tl.float16, tl.int64, indexCommuted]
  | .fp64, .bf16, .int32, Bool.false => indexBwd%[tl.float64, tl.bfloat16, tl.int32, original]
  | .fp64, .bf16, .int32, Bool.true => indexBwd%[tl.float64, tl.bfloat16, tl.int32, indexCommuted]
  | .fp64, .bf16, .int64, Bool.false => indexBwd%[tl.float64, tl.bfloat16, tl.int64, original]
  | .fp64, .bf16, .int64, Bool.true => indexBwd%[tl.float64, tl.bfloat16, tl.int64, indexCommuted]
  | .fp64, .f8e4, .int32, Bool.false => indexBwd%[tl.float64, tl.float8e4nv, tl.int32, original]
  | .fp64, .f8e4, .int32, Bool.true => indexBwd%[tl.float64, tl.float8e4nv, tl.int32, indexCommuted]
  | .fp64, .f8e4, .int64, Bool.false => indexBwd%[tl.float64, tl.float8e4nv, tl.int64, original]
  | .fp64, .f8e4, .int64, Bool.true => indexBwd%[tl.float64, tl.float8e4nv, tl.int64, indexCommuted]
  | .fp64, .f8e5, .int32, Bool.false => indexBwd%[tl.float64, tl.float8e5, tl.int32, original]
  | .fp64, .f8e5, .int32, Bool.true => indexBwd%[tl.float64, tl.float8e5, tl.int32, indexCommuted]
  | .fp64, .f8e5, .int64, Bool.false => indexBwd%[tl.float64, tl.float8e5, tl.int64, original]
  | .fp64, .f8e5, .int64, Bool.true => indexBwd%[tl.float64, tl.float8e5, tl.int64, indexCommuted]
  | .fp64, .fp64, .int32, Bool.false => indexBwd%[tl.float64, tl.float64, tl.int32, original]
  | .fp64, .fp64, .int32, Bool.true => indexBwd%[tl.float64, tl.float64, tl.int32, indexCommuted]
  | .fp64, .fp64, .int64, Bool.false => indexBwd%[tl.float64, tl.float64, tl.int64, original]
  | .fp64, .fp64, .int64, Bool.true => indexBwd%[tl.float64, tl.float64, tl.int64, indexCommuted]

end VeriTile.Bench.TritonBenchG.IndexSelectBwd
