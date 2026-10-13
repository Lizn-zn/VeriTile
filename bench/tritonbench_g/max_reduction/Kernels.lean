import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton
open Lean

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel_1`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def max_kernel_1
    (inp mid : RegionName)
    (M BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val)
}

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel_2`.

Same allowed Lean-syntax-only changes as `max_kernel_1`. -/
def max_kernel_2
    (mid out : RegionName)
    (mid_size BLOCK_MID : Nat) :
    ComputeKernel := triton {
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val)
}

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel`
(autotuned, returns value + index via `tl.max(..., return_indices=True)`).

Allowed mechanical Lean-syntax-only changes apply. The `@triton.autotune`
and `@triton.heuristics` decorators that wrap `max_kernel` are not DSL-side
constructs (they configure the launch grid, not the kernel body); they have
no in-kernel transcription. -/
def max_kernel
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index = m_offset * $(K) + pid_k
  mask1 = m_offset < $(M)
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs = inp + offset
  inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs = out_value + offset_index
  out_index_ptrs = out_index + offset_index
  tl.store(out_value_ptrs, result_value, mask=mask1)
  tl.store(out_index_ptrs, result_index, mask=mask1)
}


/-! Source-facing value reductions load their declared storage, retain fp64 or
promote narrower formats to fp32, and narrow once when writing half/bfloat/fp8 storage.
The first candidate commutes an index product; the second complements the
integer bound comparison, preserving the exact active set. -/
inductive Storage where
  | fp32 | fp16 | bf16 | f8e4 | f8e5 | fp64
  deriving DecidableEq

abbrev ValueSource := RegionName → RegionName → Nat → Nat → ComputeKernel

syntax "maxStage1%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(maxStage1%[$dtype:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let index : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE)))
      else `(tritonStmt| offset = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE)))
    let write : TSyntax `tritonStmt ←
      match dtype with
      | `(tritonDType| tl.float16) | `(tritonDType| tl.bfloat16)
      | `(tritonDType| tl.float8e4nv) | `(tritonDType| tl.float8e5) =>
          `(tritonStmt| tl.store(mid_ptr, max_val.to($dtype)))
      | _ => `(tritonStmt| tl.store(mid_ptr, max_val))
    `(fun (inp mid : RegionName) (M BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [inp] [mid] (triton {
  tl.region inp = $dtype, mid = $dtype
  pid = tl.program_id(0)
  $index
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  $write
}).surfaceBody)

syntax "maxStage2%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(maxStage2%[$dtype:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `maskComplement do
      Macro.throwError "expected original or maskComplement"
    let guard : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| mask = offset < $(mid_size))
      else `(tritonStmt| mask = not ($(mid_size) <= offset))
    let write : TSyntax `tritonStmt ←
      match dtype with
      | `(tritonDType| tl.float16) | `(tritonDType| tl.bfloat16)
      | `(tritonDType| tl.float8e4nv) | `(tritonDType| tl.float8e5) =>
          `(tritonStmt| tl.store(out, max_val.to($dtype)))
      | _ => `(tritonStmt| tl.store(out, max_val))
    `(fun (mid out : RegionName) (mid_size BLOCK_MID : Nat) =>
      ComputeKernel.mk [mid] [out] (triton {
  tl.region mid = $dtype, out = $dtype
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  $guard
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  $write
}).surfaceBody)

def stage1 (storage : Storage) (candidate : Bool) : ValueSource :=
  match storage, candidate with
  | .fp32, Bool.false => maxStage1%[tl.float32, original]
  | .fp32, Bool.true => maxStage1%[tl.float32, indexCommuted]
  | .fp16, Bool.false => maxStage1%[tl.float16, original]
  | .fp16, Bool.true => maxStage1%[tl.float16, indexCommuted]
  | .bf16, Bool.false => maxStage1%[tl.bfloat16, original]
  | .bf16, Bool.true => maxStage1%[tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => maxStage1%[tl.float8e4nv, original]
  | .f8e4, Bool.true => maxStage1%[tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => maxStage1%[tl.float8e5, original]
  | .f8e5, Bool.true => maxStage1%[tl.float8e5, indexCommuted]
  | .fp64, Bool.false => maxStage1%[tl.float64, original]
  | .fp64, Bool.true => maxStage1%[tl.float64, indexCommuted]

def stage2 (storage : Storage) (candidate : Bool) : ValueSource :=
  match storage, candidate with
  | .fp32, Bool.false => maxStage2%[tl.float32, original]
  | .fp32, Bool.true => maxStage2%[tl.float32, maskComplement]
  | .fp16, Bool.false => maxStage2%[tl.float16, original]
  | .fp16, Bool.true => maxStage2%[tl.float16, maskComplement]
  | .bf16, Bool.false => maxStage2%[tl.bfloat16, original]
  | .bf16, Bool.true => maxStage2%[tl.bfloat16, maskComplement]
  | .f8e4, Bool.false => maxStage2%[tl.float8e4nv, original]
  | .f8e4, Bool.true => maxStage2%[tl.float8e4nv, maskComplement]
  | .f8e5, Bool.false => maxStage2%[tl.float8e5, original]
  | .f8e5, Bool.true => maxStage2%[tl.float8e5, maskComplement]
  | .fp64, Bool.false => maxStage2%[tl.float64, original]
  | .fp64, Bool.true => maxStage2%[tl.float64, maskComplement]

/-- Full dimension-reduction source, including both masked output streams.
`out_index` is signed int64 storage as allocated by the Python wrapper.
The explicit store casts expose conversions implicit in Triton's store. -/
syntax "maxDimension%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(maxDimension%[$dtype:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))
      else `(tritonStmt| m_offset = $(BLOCK_M) * pid_m + tl.arange(0, $(BLOCK_M)))
    let writeValue : TSyntax `tritonStmt ← match dtype with
      | `(tritonDType| tl.bfloat16) =>
          `(tritonStmt| tl.store(out_value_ptrs, result_value.to(tl.bfloat16), mask=mask1))
      | _ => `(tritonStmt| tl.store(out_value_ptrs, result_value, mask=mask1))
    `(fun (inp out_value out_index : RegionName) (M N K BLOCK_M BLOCK_N : Nat) =>
      ComputeKernel.mk [inp] [out_value, out_index] (triton {
  tl.region inp = $dtype, out_value = $dtype, out_index = tl.int64
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  $rowIndex
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index = m_offset * $(K) + pid_k
  mask1 = m_offset < $(M)
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs = inp + offset
  inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs = out_value + offset_index
  out_index_ptrs = out_index + offset_index
  $writeValue
  tl.store(out_index_ptrs, result_index.to(tl.int64), mask=mask1)
}).surfaceBody)

def dimensionSource (storage : Storage) (candidate : Bool)
    (inp out_value out_index : RegionName) (M N K BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  (match storage, candidate with
  | .fp32, Bool.false => maxDimension%[tl.float32, original]
  | .fp32, Bool.true => maxDimension%[tl.float32, indexCommuted]
  | .fp16, Bool.false => maxDimension%[tl.float16, original]
  | .fp16, Bool.true => maxDimension%[tl.float16, indexCommuted]
  | .bf16, Bool.false => maxDimension%[tl.bfloat16, original]
  | .bf16, Bool.true => maxDimension%[tl.bfloat16, indexCommuted]
  | .f8e4, Bool.false => maxDimension%[tl.float8e4nv, original]
  | .f8e4, Bool.true => maxDimension%[tl.float8e4nv, indexCommuted]
  | .f8e5, Bool.false => maxDimension%[tl.float8e5, original]
  | .f8e5, Bool.true => maxDimension%[tl.float8e5, indexCommuted]
  | .fp64, Bool.false => maxDimension%[tl.float64, original]
  | .fp64, Bool.true => maxDimension%[tl.float64, indexCommuted])
    inp out_value out_index M N K BLOCK_M BLOCK_N

end VeriTile.Bench.TritonBenchG.MaxReduction
