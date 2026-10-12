import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: fast_rms_layernorm.py, _rms_layernorm_forward,
_gemma_rms_layernorm_forward, and _rms_layernorm_backward.
The complete candidate commutes only the integer X row-stride product.
Regular forward casts normed to the loaded weight dtype before multiplication.
W_row.dtype is resolved from its unique direct load and explicit W storage.
Gemma forward and backward ignore W_row_stride, as does the Python source;
dW and its stride are passed to backward but never read or written. -/
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.TypedSource
open Lean VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

syntax "fastRmsForward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsForward%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (Y X W r : RegionName) (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) =>
      ComputeKernel.mk [X,W] [Y,r] (triton {
  tl.region Y = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y += row_idx * $(Y_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets * $(W_row_stride), mask=mask, other=0)
  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $((eps : ℝ)))
  tl.store(r, inv_var.to(tl.float32))
  normed = X_row * inv_var
  normed = normed.to($weightTy:tritonDType)
  output = normed * W_row
  tl.store(Y + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)

def forward (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsForward%[tl.float16,tl.float16,indexCommuted] else fastRmsForward%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsForward%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsForward%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsForward%[tl.float16,tl.float32,indexCommuted] else fastRmsForward%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsForward%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsForward%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsForward%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsForward%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsForward%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsForward%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsForward%[tl.float32,tl.float16,indexCommuted] else fastRmsForward%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsForward%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsForward%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsForward%[tl.float32,tl.float32,indexCommuted] else fastRmsForward%[tl.float32,tl.float32,original]

syntax "fastRmsGemma%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsGemma%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (Y X W r : RegionName) (Y_row_stride X_row_stride W_row_stride r_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) =>
      ComputeKernel.mk [X,W] [Y,r] (triton {
  tl.region Y = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y += row_idx * $(Y_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  row_var = tl.sum(X_row * X_row, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $((eps : ℝ)))
  tl.store(r, inv_var.to(tl.float32))
  normed = X_row * inv_var
  output = normed * (W_row + 1.0)
  tl.store(Y + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)

def gemma (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsGemma%[tl.float16,tl.float16,indexCommuted] else fastRmsGemma%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsGemma%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsGemma%[tl.float16,tl.float32,indexCommuted] else fastRmsGemma%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsGemma%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsGemma%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsGemma%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsGemma%[tl.float32,tl.float16,indexCommuted] else fastRmsGemma%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsGemma%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsGemma%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsGemma%[tl.float32,tl.float32,indexCommuted] else fastRmsGemma%[tl.float32,tl.float32,original]

syntax "fastRmsBackward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRmsBackward%[$inputTy:tritonDType,$weightTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (dY X W r dW : RegionName) (dY_row_stride X_row_stride W_row_stride r_row_stride dW_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) (GEMMA : Bool) =>
      ComputeKernel.mk [dY,X,W,r] [dY] (triton {
  tl.region dY = $inputTy, X = $inputTy, W = $weightTy, r = tl.float32, dW = $inputTy
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  $rowIndex
  r += row_idx * $(r_row_stride)
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  normed = X_row * inv_var
  if $((GEMMA : Bool)) {
    dY_W = dY_row * (W_row + 1.0)
  } else {
    dY_W = dY_row * W_row
  }
  rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
  output = inv_var / $(n_cols) * ($(n_cols) * dY_W - normed * rowsum_dY_normed)
  tl.store(dY + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)

def backward (input weight : Storage) (swapped : Bool) := match input,weight with
  | .fp16,.fp16 => if swapped then fastRmsBackward%[tl.float16,tl.float16,indexCommuted] else fastRmsBackward%[tl.float16,tl.float16,original]
  | .fp16,.bf16 => if swapped then fastRmsBackward%[tl.float16,tl.bfloat16,indexCommuted] else fastRmsBackward%[tl.float16,tl.bfloat16,original]
  | .fp16,.fp32 => if swapped then fastRmsBackward%[tl.float16,tl.float32,indexCommuted] else fastRmsBackward%[tl.float16,tl.float32,original]
  | .bf16,.fp16 => if swapped then fastRmsBackward%[tl.bfloat16,tl.float16,indexCommuted] else fastRmsBackward%[tl.bfloat16,tl.float16,original]
  | .bf16,.bf16 => if swapped then fastRmsBackward%[tl.bfloat16,tl.bfloat16,indexCommuted] else fastRmsBackward%[tl.bfloat16,tl.bfloat16,original]
  | .bf16,.fp32 => if swapped then fastRmsBackward%[tl.bfloat16,tl.float32,indexCommuted] else fastRmsBackward%[tl.bfloat16,tl.float32,original]
  | .fp32,.fp16 => if swapped then fastRmsBackward%[tl.float32,tl.float16,indexCommuted] else fastRmsBackward%[tl.float32,tl.float16,original]
  | .fp32,.bf16 => if swapped then fastRmsBackward%[tl.float32,tl.bfloat16,indexCommuted] else fastRmsBackward%[tl.float32,tl.bfloat16,original]
  | .fp32,.fp32 => if swapped then fastRmsBackward%[tl.float32,tl.float32,indexCommuted] else fastRmsBackward%[tl.float32,tl.float32,original]

end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.TypedSource
