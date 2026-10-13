import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.LogSoftmax

open Lean VeriTile.Triton

/-- Faithful transcription of `log_softmax.py`'s `log_softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_M: tl.constexpr` / `BLOCK_N: tl.constexpr` -> Lean `Nat`
  parameters. -/
def log_softmax_kernel
    (output_ptr input_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(tl.float32)
  row_minus_max = inp - tl.max(inp, axis=1)[:, None]
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=1)[:, None]
  softmax_output = tl.log(numerator / denominator)
  output_ptrs = output_ptr + offset
  tl.store(output_ptrs, softmax_output, mask=mask)
}

/-- Faithful transcription of `log_softmax.py`'s
`log_softmax_backward_kernel`. -/
def log_softmax_backward_kernel
    (out_ptr out_grad_ptr in_grad_ptr : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offsets = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  out_ptrs = out_ptr + offsets
  out = tl.load(out_ptrs, mask=mask).to(tl.float32)
  out_grad_ptrs = out_grad_ptr + offsets
  out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
  scale = tl.sum(out_grad, 1)
  in_grad = out_grad - tl.exp((out).to(tl.float32)) * scale[:, None]
  in_grad_ptrs = in_grad_ptr + offsets
  tl.store(in_grad_ptrs, in_grad, mask=mask)
}


/-! Complete typed source; the candidate changes only exact row-index arithmetic. -/
syntax "logSoftmaxForward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(logSoftmaxForward%[$inputStorage:tritonDType, $destStorage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let rowOffset : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))
      else `(tritonStmt| m_offset = $(BLOCK_M) * pid_m + tl.arange(0, $(BLOCK_M)))
    let storeStmt : TSyntax `tritonStmt ←
      match destStorage with
      | `(tritonDType| tl.float32) =>
        `(tritonStmt| tl.store(output_ptrs, softmax_output, mask=mask))
      | _ =>
        `(tritonStmt| tl.store(output_ptrs, softmax_output.to($destStorage), mask=mask))
    `(fun (output_ptr input_ptr : RegionName)
        (M N K BLOCK_M BLOCK_N : Nat) =>
      ComputeKernel.mk [input_ptr] [output_ptr] (triton {
  tl.region input_ptr = $inputStorage, output_ptr = $destStorage
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  $rowOffset
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  input_ptrs = input_ptr + offset
  inp = (tl.load(input_ptrs, mask=mask, other=-float("inf"))).to(tl.float32)
  row_minus_max = inp - tl.max(inp, axis=1)[:, None]
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=1)[:, None]
  softmax_output = tl.log(numerator / denominator)
  output_ptrs = output_ptr + offset
  $storeStmt
}).surfaceBody)


/-! Complete typed source; the candidate changes only exact row-index arithmetic. -/
syntax "logSoftmaxBackward%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(logSoftmaxBackward%[$outStorage:tritonDType, $gradStorage:tritonDType, $destStorage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let rowOffset : TSyntax `tritonStmt ←
      if version.getId == `original then
        `(tritonStmt| m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M)))
      else `(tritonStmt| m_offset = $(BLOCK_M) * pid_m + tl.arange(0, $(BLOCK_M)))
    let storeStmt : TSyntax `tritonStmt ←
      match destStorage with
      | `(tritonDType| tl.float32) =>
        `(tritonStmt| tl.store(in_grad_ptrs, in_grad, mask=mask))
      | _ =>
        `(tritonStmt| tl.store(in_grad_ptrs, in_grad.to($destStorage), mask=mask))
    `(fun (out_ptr out_grad_ptr in_grad_ptr : RegionName)
        (M N K BLOCK_M BLOCK_N : Nat) =>
      ComputeKernel.mk [out_ptr, out_grad_ptr] [in_grad_ptr] (triton {
  tl.region out_ptr = $outStorage, out_grad_ptr = $gradStorage, in_grad_ptr = $destStorage
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  $rowOffset
  n_offset = tl.arange(0, $(BLOCK_N))
  offsets = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  out_ptrs = out_ptr + offsets
  out = tl.load(out_ptrs, mask=mask).to(tl.float32)
  out_grad_ptrs = out_grad_ptr + offsets
  out_grad = tl.load(out_grad_ptrs, mask=mask).to(tl.float32)
  scale = tl.sum(out_grad, 1)
  in_grad = out_grad - tl.exp((out).to(tl.float32)) * scale[:, None]
  in_grad_ptrs = in_grad_ptr + offsets
  $storeStmt
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq

abbrev ForwardSource := RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel
abbrev BackwardSource := RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → Nat → ComputeKernel

def typedForward (input output : Storage) (v : Bool) : ForwardSource :=
  match input, output, v with
  | .fp32, .fp32, Bool.false => logSoftmaxForward%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => logSoftmaxForward%[tl.float32, tl.float32, commuted]
  | .fp32, .fp16, Bool.false => logSoftmaxForward%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => logSoftmaxForward%[tl.float32, tl.float16, commuted]
  | .fp32, .bf16, Bool.false => logSoftmaxForward%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => logSoftmaxForward%[tl.float32, tl.bfloat16, commuted]
  | .fp32, .fp64, Bool.false => logSoftmaxForward%[tl.float32, tl.float64, original]
  | .fp32, .fp64, Bool.true => logSoftmaxForward%[tl.float32, tl.float64, commuted]
  | .fp32, .f8e4, Bool.false => logSoftmaxForward%[tl.float32, tl.float8e4nv, original]
  | .fp32, .f8e4, Bool.true => logSoftmaxForward%[tl.float32, tl.float8e4nv, commuted]
  | .fp32, .f8e5, Bool.false => logSoftmaxForward%[tl.float32, tl.float8e5, original]
  | .fp32, .f8e5, Bool.true => logSoftmaxForward%[tl.float32, tl.float8e5, commuted]
  | .fp16, .fp32, Bool.false => logSoftmaxForward%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => logSoftmaxForward%[tl.float16, tl.float32, commuted]
  | .fp16, .fp16, Bool.false => logSoftmaxForward%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => logSoftmaxForward%[tl.float16, tl.float16, commuted]
  | .fp16, .bf16, Bool.false => logSoftmaxForward%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => logSoftmaxForward%[tl.float16, tl.bfloat16, commuted]
  | .fp16, .fp64, Bool.false => logSoftmaxForward%[tl.float16, tl.float64, original]
  | .fp16, .fp64, Bool.true => logSoftmaxForward%[tl.float16, tl.float64, commuted]
  | .fp16, .f8e4, Bool.false => logSoftmaxForward%[tl.float16, tl.float8e4nv, original]
  | .fp16, .f8e4, Bool.true => logSoftmaxForward%[tl.float16, tl.float8e4nv, commuted]
  | .fp16, .f8e5, Bool.false => logSoftmaxForward%[tl.float16, tl.float8e5, original]
  | .fp16, .f8e5, Bool.true => logSoftmaxForward%[tl.float16, tl.float8e5, commuted]
  | .bf16, .fp32, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.float32, commuted]
  | .bf16, .fp16, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.float16, commuted]
  | .bf16, .bf16, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.bfloat16, commuted]
  | .bf16, .fp64, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.float64, original]
  | .bf16, .fp64, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.float64, commuted]
  | .bf16, .f8e4, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.float8e4nv, original]
  | .bf16, .f8e4, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.float8e4nv, commuted]
  | .bf16, .f8e5, Bool.false => logSoftmaxForward%[tl.bfloat16, tl.float8e5, original]
  | .bf16, .f8e5, Bool.true => logSoftmaxForward%[tl.bfloat16, tl.float8e5, commuted]
  | .fp64, .fp32, Bool.false => logSoftmaxForward%[tl.float64, tl.float32, original]
  | .fp64, .fp32, Bool.true => logSoftmaxForward%[tl.float64, tl.float32, commuted]
  | .fp64, .fp16, Bool.false => logSoftmaxForward%[tl.float64, tl.float16, original]
  | .fp64, .fp16, Bool.true => logSoftmaxForward%[tl.float64, tl.float16, commuted]
  | .fp64, .bf16, Bool.false => logSoftmaxForward%[tl.float64, tl.bfloat16, original]
  | .fp64, .bf16, Bool.true => logSoftmaxForward%[tl.float64, tl.bfloat16, commuted]
  | .fp64, .fp64, Bool.false => logSoftmaxForward%[tl.float64, tl.float64, original]
  | .fp64, .fp64, Bool.true => logSoftmaxForward%[tl.float64, tl.float64, commuted]
  | .fp64, .f8e4, Bool.false => logSoftmaxForward%[tl.float64, tl.float8e4nv, original]
  | .fp64, .f8e4, Bool.true => logSoftmaxForward%[tl.float64, tl.float8e4nv, commuted]
  | .fp64, .f8e5, Bool.false => logSoftmaxForward%[tl.float64, tl.float8e5, original]
  | .fp64, .f8e5, Bool.true => logSoftmaxForward%[tl.float64, tl.float8e5, commuted]
  | .f8e4, .fp32, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.float32, original]
  | .f8e4, .fp32, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.float32, commuted]
  | .f8e4, .fp16, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.float16, original]
  | .f8e4, .fp16, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.float16, commuted]
  | .f8e4, .bf16, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.bfloat16, original]
  | .f8e4, .bf16, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.bfloat16, commuted]
  | .f8e4, .fp64, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.float64, original]
  | .f8e4, .fp64, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.float64, commuted]
  | .f8e4, .f8e4, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, .f8e4, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e4, .f8e5, Bool.false => logSoftmaxForward%[tl.float8e4nv, tl.float8e5, original]
  | .f8e4, .f8e5, Bool.true => logSoftmaxForward%[tl.float8e4nv, tl.float8e5, commuted]
  | .f8e5, .fp32, Bool.false => logSoftmaxForward%[tl.float8e5, tl.float32, original]
  | .f8e5, .fp32, Bool.true => logSoftmaxForward%[tl.float8e5, tl.float32, commuted]
  | .f8e5, .fp16, Bool.false => logSoftmaxForward%[tl.float8e5, tl.float16, original]
  | .f8e5, .fp16, Bool.true => logSoftmaxForward%[tl.float8e5, tl.float16, commuted]
  | .f8e5, .bf16, Bool.false => logSoftmaxForward%[tl.float8e5, tl.bfloat16, original]
  | .f8e5, .bf16, Bool.true => logSoftmaxForward%[tl.float8e5, tl.bfloat16, commuted]
  | .f8e5, .fp64, Bool.false => logSoftmaxForward%[tl.float8e5, tl.float64, original]
  | .f8e5, .fp64, Bool.true => logSoftmaxForward%[tl.float8e5, tl.float64, commuted]
  | .f8e5, .f8e4, Bool.false => logSoftmaxForward%[tl.float8e5, tl.float8e4nv, original]
  | .f8e5, .f8e4, Bool.true => logSoftmaxForward%[tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, .f8e5, Bool.false => logSoftmaxForward%[tl.float8e5, tl.float8e5, original]
  | .f8e5, .f8e5, Bool.true => logSoftmaxForward%[tl.float8e5, tl.float8e5, commuted]

def typedBackward_fp32_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.float32, tl.float8e5, commuted]

def typedBackward_fp32_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.float16, tl.float8e5, commuted]

def typedBackward_fp32_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.bfloat16, tl.float8e5, commuted]

def typedBackward_fp32_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.float64, tl.float8e5, commuted]

def typedBackward_fp32_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e4nv, tl.float8e5, commuted]

def typedBackward_fp32_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float32, tl.float8e5, tl.float8e5, commuted]

def typedBackward_fp16_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.float32, tl.float8e5, commuted]

def typedBackward_fp16_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.float16, tl.float8e5, commuted]

def typedBackward_fp16_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.bfloat16, tl.float8e5, commuted]

def typedBackward_fp16_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.float64, tl.float8e5, commuted]

def typedBackward_fp16_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e4nv, tl.float8e5, commuted]

def typedBackward_fp16_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float16, tl.float8e5, tl.float8e5, commuted]

def typedBackward_bf16_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float32, tl.float8e5, commuted]

def typedBackward_bf16_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float16, tl.float8e5, commuted]

def typedBackward_bf16_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.bfloat16, tl.float8e5, commuted]

def typedBackward_bf16_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float64, tl.float8e5, commuted]

def typedBackward_bf16_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, commuted]

def typedBackward_bf16_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.bfloat16, tl.float8e5, tl.float8e5, commuted]

def typedBackward_fp64_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.float32, tl.float8e5, commuted]

def typedBackward_fp64_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.float16, tl.float8e5, commuted]

def typedBackward_fp64_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.bfloat16, tl.float8e5, commuted]

def typedBackward_fp64_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.float64, tl.float8e5, commuted]

def typedBackward_fp64_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e4nv, tl.float8e5, commuted]

def typedBackward_fp64_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float64, tl.float8e5, tl.float8e5, commuted]

def typedBackward_f8e4_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float32, tl.float8e5, commuted]

def typedBackward_f8e4_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float16, tl.float8e5, commuted]

def typedBackward_f8e4_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, commuted]

def typedBackward_f8e4_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float64, tl.float8e5, commuted]

def typedBackward_f8e4_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, commuted]

def typedBackward_f8e4_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e4nv, tl.float8e5, tl.float8e5, commuted]

def typedBackward_f8e5_fp32 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float32, tl.float8e5, commuted]

def typedBackward_f8e5_fp16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float16, tl.float8e5, commuted]

def typedBackward_f8e5_bf16 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.bfloat16, tl.float8e5, commuted]

def typedBackward_f8e5_fp64 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float64, tl.float8e5, commuted]

def typedBackward_f8e5_f8e4 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e4nv, tl.float8e5, commuted]

def typedBackward_f8e5_f8e5 (output : Storage) (v : Bool) : BackwardSource :=
  match output, v with
  | .fp32, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float32, original]
  | .fp32, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float32, commuted]
  | .fp16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float16, original]
  | .fp16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float16, commuted]
  | .bf16, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.bfloat16, original]
  | .bf16, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.bfloat16, commuted]
  | .fp64, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float64, original]
  | .fp64, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float64, commuted]
  | .f8e4, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float8e4nv, original]
  | .f8e4, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, Bool.false => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float8e5, original]
  | .f8e5, Bool.true => logSoftmaxBackward%[tl.float8e5, tl.float8e5, tl.float8e5, commuted]

def typedBackward (input upstream output : Storage) (v : Bool) : BackwardSource :=
  match input, upstream with
  | .fp32, .fp32 => typedBackward_fp32_fp32 output v
  | .fp32, .fp16 => typedBackward_fp32_fp16 output v
  | .fp32, .bf16 => typedBackward_fp32_bf16 output v
  | .fp32, .fp64 => typedBackward_fp32_fp64 output v
  | .fp32, .f8e4 => typedBackward_fp32_f8e4 output v
  | .fp32, .f8e5 => typedBackward_fp32_f8e5 output v
  | .fp16, .fp32 => typedBackward_fp16_fp32 output v
  | .fp16, .fp16 => typedBackward_fp16_fp16 output v
  | .fp16, .bf16 => typedBackward_fp16_bf16 output v
  | .fp16, .fp64 => typedBackward_fp16_fp64 output v
  | .fp16, .f8e4 => typedBackward_fp16_f8e4 output v
  | .fp16, .f8e5 => typedBackward_fp16_f8e5 output v
  | .bf16, .fp32 => typedBackward_bf16_fp32 output v
  | .bf16, .fp16 => typedBackward_bf16_fp16 output v
  | .bf16, .bf16 => typedBackward_bf16_bf16 output v
  | .bf16, .fp64 => typedBackward_bf16_fp64 output v
  | .bf16, .f8e4 => typedBackward_bf16_f8e4 output v
  | .bf16, .f8e5 => typedBackward_bf16_f8e5 output v
  | .fp64, .fp32 => typedBackward_fp64_fp32 output v
  | .fp64, .fp16 => typedBackward_fp64_fp16 output v
  | .fp64, .bf16 => typedBackward_fp64_bf16 output v
  | .fp64, .fp64 => typedBackward_fp64_fp64 output v
  | .fp64, .f8e4 => typedBackward_fp64_f8e4 output v
  | .fp64, .f8e5 => typedBackward_fp64_f8e5 output v
  | .f8e4, .fp32 => typedBackward_f8e4_fp32 output v
  | .f8e4, .fp16 => typedBackward_f8e4_fp16 output v
  | .f8e4, .bf16 => typedBackward_f8e4_bf16 output v
  | .f8e4, .fp64 => typedBackward_f8e4_fp64 output v
  | .f8e4, .f8e4 => typedBackward_f8e4_f8e4 output v
  | .f8e4, .f8e5 => typedBackward_f8e4_f8e5 output v
  | .f8e5, .fp32 => typedBackward_f8e5_fp32 output v
  | .f8e5, .fp16 => typedBackward_f8e5_fp16 output v
  | .f8e5, .bf16 => typedBackward_f8e5_bf16 output v
  | .f8e5, .fp64 => typedBackward_f8e5_fp64 output v
  | .f8e5, .f8e4 => typedBackward_f8e5_f8e4 output v
  | .f8e5, .f8e5 => typedBackward_f8e5_f8e5 output v


end VeriTile.Bench.TritonBenchG.LogSoftmax
