import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Complete typed sources of the four diagonal SSM JITs.
Source: diag_ssm_triton.py, diag_ssm_forward_kernel

Real paths use matching fp16/bf16/fp32 storage across arguments. Complex paths
retain the interleaved fp32 real/imaginary components of complex64 tensors.
The complete recurrences, reverse-time branches, gradient signs, masks and
stores are preserved. The candidate only commutes the integer column-index
product. The complex forward tuple update reads two distinct new-state
registers; neither right-hand side is overwritten by the update.
No host autograd, concurrent launch, fixed-width overflow or GPU rounding
claim is made by this source transcription. -/
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource
open Lean VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

syntax "diagSsmForwardSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(diagSsmForwardSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let columnIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| col_idx = $(BLOCK_SIZE) * tl.program_id(0))
      else `(tritonStmt| col_idx = tl.program_id(0) * $(BLOCK_SIZE))
    `(fun (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [s_ptr, x_ptr, lambda_ptr] [y_ptr] (triton {
  tl.region s_ptr = $storage, x_ptr = $storage, lambda_ptr = $storage, y_ptr = $storage
  $columnIndex
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  tl.for t in $(length) {
    offsets = t * $(batch_size) * $(dim) + col_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    s = s * Lambda + x
    tl.store(y_ptr + offsets, s, mask=mask)
  }
}).surfaceBody)

abbrev forwardSource := RegionName → RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel
def forward (storage : Storage) (swapped : Bool) : forwardSource :=
  match storage, swapped with
  | .fp32, Bool.false => diagSsmForwardSource%[tl.float32, original]
  | .fp32, Bool.true => diagSsmForwardSource%[tl.float32, indexCommuted]
  | .fp16, Bool.false => diagSsmForwardSource%[tl.float16, original]
  | .fp16, Bool.true => diagSsmForwardSource%[tl.float16, indexCommuted]
  | .bf16, Bool.false => diagSsmForwardSource%[tl.bfloat16, original]
  | .bf16, Bool.true => diagSsmForwardSource%[tl.bfloat16, indexCommuted]

/-- Source: diag_ssm_triton.py, diag_ssm_backward_kernel. -/
syntax "diagSsmBackwardSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(diagSsmBackwardSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let columnIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| col_idx = $(BLOCK_SIZE) * tl.program_id(0))
      else `(tritonStmt| col_idx = tl.program_id(0) * $(BLOCK_SIZE))
    `(fun (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [s_ptr, lambda_ptr, y_ptr, grad_y_ptr] [grad_s_ptr, grad_x_ptr, grad_lambda_ptr] (triton {
  tl.region s_ptr = $storage, lambda_ptr = $storage, y_ptr = $storage, grad_s_ptr = $storage, grad_x_ptr = $storage, grad_lambda_ptr = $storage, grad_y_ptr = $storage
  $columnIndex
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  Lambda = tl.load(lambda_ptr + col_offsets % $(dim), mask=mask, other=0)
  grad_s = tl.zeros_like(Lambda)
  grad_Lambda = tl.zeros_like(Lambda)
  tl.for i in $(length) {
    t = $(length) - 1 - i
    offsets = t * $(batch_size) * $(dim) + col_offsets
    grad_y = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    if t > 0 {
      s = tl.load( y_ptr + offsets - $(batch_size) * $(dim), mask=mask, other=0)
    } else {
      s = tl.load(s_ptr + col_offsets, mask=mask, other=0)
    }
    grad_s = grad_y + grad_s
    grad_x = grad_s
    grad_Lambda += grad_s * s
    grad_s = grad_s * Lambda
    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)
  }
  tl.store(grad_s_ptr + col_offsets, grad_s, mask=mask)
  tl.store(grad_lambda_ptr + col_offsets, grad_Lambda, mask=mask)
}).surfaceBody)

abbrev backwardSource := RegionName → RegionName → RegionName → RegionName → RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel
def backward (storage : Storage) (swapped : Bool) : backwardSource :=
  match storage, swapped with
  | .fp32, Bool.false => diagSsmBackwardSource%[tl.float32, original]
  | .fp32, Bool.true => diagSsmBackwardSource%[tl.float32, indexCommuted]
  | .fp16, Bool.false => diagSsmBackwardSource%[tl.float16, original]
  | .fp16, Bool.true => diagSsmBackwardSource%[tl.float16, indexCommuted]
  | .bf16, Bool.false => diagSsmBackwardSource%[tl.bfloat16, original]
  | .bf16, Bool.true => diagSsmBackwardSource%[tl.bfloat16, indexCommuted]

/-- Source: diag_ssm_triton.py, diag_ssm_forward_kernel_complex. -/
syntax "diagSsmComplexForwardSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(diagSsmComplexForwardSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let columnIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| col_idx = $(BLOCK_SIZE) * tl.program_id(0))
      else `(tritonStmt| col_idx = tl.program_id(0) * $(BLOCK_SIZE))
    `(fun (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [s_ptr, x_ptr, lambda_ptr] [y_ptr] (triton {
  tl.region s_ptr = $storage, x_ptr = $storage, y_ptr = $storage, lambda_ptr = $storage
  $columnIndex
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  s_real = tl.load(s_ptr + col_offsets * 2, mask=mask, other=0)
  s_imag = tl.load(s_ptr + col_offsets * 2 + 1, mask=mask, other=0)
  lambda_real = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2, mask=mask, other=0)
  lambda_imag = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2 + 1, mask=mask, other=0)
  tl.for t in $(length) {
    offsets = (t * $(batch_size) * $(dim) + col_offsets) * 2
    x_real = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_imag = tl.load(x_ptr + offsets + 1, mask=mask, other=0)
    new_s_real = s_real * lambda_real - s_imag * lambda_imag + x_real
    new_s_imag = s_real * lambda_imag + s_imag * lambda_real + x_imag
    tl.store(y_ptr + offsets, new_s_real, mask=mask)
    tl.store(y_ptr + offsets + 1, new_s_imag, mask=mask)
    s_real, s_imag = new_s_real, new_s_imag
  }
}).surfaceBody)

abbrev complexForwardSource := RegionName → RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel
def complexForward (swapped : Bool) : complexForwardSource :=
  if swapped then diagSsmComplexForwardSource%[tl.float32, indexCommuted]
  else diagSsmComplexForwardSource%[tl.float32, original]

/-- Source: diag_ssm_triton.py, diag_ssm_backward_kernel_complex. -/
syntax "diagSsmComplexBackwardSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(diagSsmComplexBackwardSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let columnIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| col_idx = $(BLOCK_SIZE) * tl.program_id(0))
      else `(tritonStmt| col_idx = tl.program_id(0) * $(BLOCK_SIZE))
    `(fun (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [s_ptr, lambda_ptr, y_ptr, grad_y_ptr] [grad_s_ptr, grad_x_ptr, grad_lambda_ptr] (triton {
  tl.region s_ptr = $storage, lambda_ptr = $storage, y_ptr = $storage, grad_s_ptr = $storage, grad_x_ptr = $storage, grad_lambda_ptr = $storage, grad_y_ptr = $storage
  $columnIndex
  col_offsets = col_idx + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(batch_size) * $(dim)
  lambda_real = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2, mask=mask, other=0)
  lambda_imag = tl.load( lambda_ptr + (col_offsets % $(dim)) * 2 + 1, mask=mask, other=0)
  grad_s_real = tl.zeros_like(lambda_real)
  grad_s_imag = tl.zeros_like(lambda_imag)
  grad_lambda_real = tl.zeros_like(lambda_real)
  grad_lambda_imag = tl.zeros_like(lambda_imag)
  tl.for i in $(length) {
    t = $(length) - 1 - i
    offsets = (t * $(batch_size) * $(dim) + col_offsets) * 2
    grad_y_real = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    grad_y_imag = -tl.load( grad_y_ptr + offsets + 1, mask=mask, other=0)
    if t > 0 {
      s_real = tl.load( y_ptr + offsets - 2 * $(batch_size) * $(dim), mask=mask, other=0)
      s_imag = tl.load( y_ptr + offsets - 2 * $(batch_size) * $(dim) + 1, mask=mask, other=0)
    } else {
      s_real = tl.load(s_ptr + 2 * col_offsets, mask=mask, other=0)
      s_imag = tl.load( s_ptr + 2 * col_offsets + 1, mask=mask, other=0)
    }
    grad_s_real = grad_y_real + grad_s_real
    grad_s_imag = grad_y_imag + grad_s_imag
    grad_x_real = grad_s_real
    grad_x_imag = grad_s_imag
    grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
    grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
    grad_s_real = grad_x_real * lambda_real - grad_x_imag * lambda_imag
    grad_s_imag = grad_x_real * lambda_imag + grad_x_imag * lambda_real
    tl.store(grad_x_ptr + offsets, grad_x_real, mask=mask)
    tl.store(grad_x_ptr + offsets + 1, -grad_x_imag, mask=mask)
  }
  tl.store(grad_s_ptr + col_offsets * 2, grad_s_real, mask=mask)
  tl.store(grad_s_ptr + col_offsets * 2 + 1, -grad_s_imag, mask=mask)
  tl.store( grad_lambda_ptr + col_offsets * 2, grad_lambda_real, mask=mask)
  tl.store( grad_lambda_ptr + col_offsets * 2 + 1, -grad_lambda_imag, mask=mask)
}).surfaceBody)

abbrev complexBackwardSource := RegionName → RegionName → RegionName → RegionName → RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel
def complexBackward (swapped : Bool) : complexBackwardSource :=
  if swapped then diagSsmComplexBackwardSource%[tl.float32, indexCommuted]
  else diagSsmComplexBackwardSource%[tl.float32, original]

end VeriTile.Bench.TritonBenchG.DiagSsmTriton.TypedSource

namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton
open VeriTile.Triton

/-- Buffer names and logical sizes shared by the real and complex forward
specifications. B is the source block size; length is supplied separately. -/
structure ForwardLayout where
  s : RegionName
  x : RegionName
  y : RegionName
  l : RegionName
  batch : Nat
  dim : Nat
  B : Nat

/-- Backward buffers contain one initial state/transition per column,
time-dependent y/grad_y/grad_x and terminal grad_s/grad_lambda. -/
structure BackwardLayout where
  s : RegionName
  l : RegionName
  y : RegionName
  gs : RegionName
  gx : RegionName
  gl : RegionName
  gy : RegionName
  batch : Nat
  dim : Nat
  B : Nat
end VeriTile.Bench.TritonBenchG.DiagSsmTriton
