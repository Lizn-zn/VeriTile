import VeriTile.Triton.DSL

/- Translation-surface blocker: typed fp16/bf16 output stores spell the implicit
destination conversion explicitly because output pointer aliases currently lose
the declared storage dtype. Proofs/Source.lean checks the exact carrier and single
cast for every storage pair, candidate and mask branch; fp32 stores are unchanged. -/

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3
open Lean VeriTile.Triton

/-- Faithful transcription of `softmax_triton3.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- Python `mask_ptr is not None` -> Lean `HAS_MASK : Bool` constexpr gate.
- Python `.to(tl.float32)` casts are represented explicitly in the Compute
  layer; the algorithm-layer theorem observes their Real projection. -/
def softmax_kernel
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = (tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = (mask_ptr + (row_idx * $(row_stride))) + col_offsets
    mask = (tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)).to(tl.float32)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

/-- Real-projection companion for the typed reciprocal candidate. -/
def softmax_reciprocal_kernel
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = (tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = (mask_ptr + (row_idx * $(row_stride))) + col_offsets
    mask = (tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)).to(tl.float32)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator * (1 / denominator)
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

/-! Typed complete source: input/output share storage, while the optional additive
mask has independently chosen storage. Both loads explicitly convert to fp32,
all arithmetic and reductions use fp32, and the output store retains its dtype.
The preceding legacy definition and its theorem signatures remain available. -/
syntax "softmax3Source%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(softmax3Source%[$inputType:tritonDType, $maskType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `reciprocal do
      Macro.throwError "expected original or reciprocal"
    let normalize : TSyntax `tritonStmt ←
      if version.getId == `reciprocal then
        `(tritonStmt| softmax_output = numerator * (1 / denominator))
      else `(tritonStmt| softmax_output = numerator / denominator)
    let store : TSyntax `tritonStmt ←
      match inputType with
      | `(tritonDType| tl.float32) =>
          `(tritonStmt| tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols)))
      | _ => `(tritonStmt| tl.store(output_ptrs, (softmax_output).to($inputType:tritonDType),
          mask=col_offsets < $(n_cols)))
    `(fun (output_ptr input_ptr mask_ptr : RegionName)
        (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) => triton {
  tl.region input_ptr = $inputType, output_ptr = $inputType, mask_ptr = $maskType
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = (tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = (mask_ptr + (row_idx * $(row_stride))) + col_offsets
    mask = (tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)).to(tl.float32)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  $normalize
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  $store
})

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

abbrev Source := RegionName → RegionName → RegionName → Nat → Nat → Nat → Bool → ComputeKernel

def typedKernel (inputType maskType : Storage) (reciprocal : Bool) : Source :=
  match inputType, maskType, reciprocal with
  | .fp32, .fp32, Bool.false => softmax3Source%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => softmax3Source%[tl.float32, tl.float32, reciprocal]
  | .fp32, .fp16, Bool.false => softmax3Source%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => softmax3Source%[tl.float32, tl.float16, reciprocal]
  | .fp32, .bf16, Bool.false => softmax3Source%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => softmax3Source%[tl.float32, tl.bfloat16, reciprocal]
  | .fp16, .fp32, Bool.false => softmax3Source%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => softmax3Source%[tl.float16, tl.float32, reciprocal]
  | .fp16, .fp16, Bool.false => softmax3Source%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => softmax3Source%[tl.float16, tl.float16, reciprocal]
  | .fp16, .bf16, Bool.false => softmax3Source%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => softmax3Source%[tl.float16, tl.bfloat16, reciprocal]
  | .bf16, .fp32, Bool.false => softmax3Source%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => softmax3Source%[tl.bfloat16, tl.float32, reciprocal]
  | .bf16, .fp16, Bool.false => softmax3Source%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => softmax3Source%[tl.bfloat16, tl.float16, reciprocal]
  | .bf16, .bf16, Bool.false => softmax3Source%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => softmax3Source%[tl.bfloat16, tl.bfloat16, reciprocal]

end VeriTile.Bench.TritonBenchG.SoftmaxTriton3
