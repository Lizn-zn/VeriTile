import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: fast_layernorm.py, layernorm_forward and layernorm_backward.
Complete forward/backward source templates. Input/output storage matches; weight
and bias storage remain independently selected. Every load casts to fp32 before
arithmetic, r/mu use fp32, and the final store converts to the original payload
storage. The candidate only commutes the integer X row-stride product.
The forward variance includes centered padding lanes. The backward formula
uses caller-supplied statistics; no host composition or derivative claim is
implicit in these per-program sources. -/
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.TypedSource
open Lean VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

syntax "fastLayernormForward%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastLayernormForward%[$inputTy:tritonDType,$weightTy:tritonDType,$biasTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (Y X W b r mu : RegionName) (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) =>
      ComputeKernel.mk [X,W,b] [Y,r,mu] (triton {
  tl.region Y = $inputTy, X = $inputTy, W = $weightTy, b = $biasTy, r = tl.float32, mu = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  Y += row_idx * $(Y_row_stride)
  $rowIndex
  r += row_idx
  mu += row_idx
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)
  mean_X = tl.sum(X_row, axis=0) / $(n_cols)
  XX = X_row - mean_X
  row_var = tl.sum(XX * XX, axis=0) / $(n_cols)
  inv_var = tl.math.rsqrt(row_var + $((eps : ℝ)))
  tl.store(r, inv_var.to(tl.float32))
  tl.store(mu, mean_X.to(tl.float32))
  output = (XX * inv_var) * W_row + b_row
  tl.store(Y + col_offsets, output.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)

def forward (input weight bias : Storage) (swapped : Bool) := match input,weight,bias with
  | .fp16,.fp16,.fp16 => if swapped then fastLayernormForward%[tl.float16,tl.float16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float16,tl.float16,original]
  | .fp16,.fp16,.bf16 => if swapped then fastLayernormForward%[tl.float16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float16,tl.bfloat16,original]
  | .fp16,.fp16,.fp32 => if swapped then fastLayernormForward%[tl.float16,tl.float16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float16,tl.float16,tl.float32,original]
  | .fp16,.bf16,.fp16 => if swapped then fastLayernormForward%[tl.float16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float16,tl.bfloat16,tl.float16,original]
  | .fp16,.bf16,.bf16 => if swapped then fastLayernormForward%[tl.float16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float16,tl.bfloat16,tl.bfloat16,original]
  | .fp16,.bf16,.fp32 => if swapped then fastLayernormForward%[tl.float16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float16,tl.bfloat16,tl.float32,original]
  | .fp16,.fp32,.fp16 => if swapped then fastLayernormForward%[tl.float16,tl.float32,tl.float16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float32,tl.float16,original]
  | .fp16,.fp32,.bf16 => if swapped then fastLayernormForward%[tl.float16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float16,tl.float32,tl.bfloat16,original]
  | .fp16,.fp32,.fp32 => if swapped then fastLayernormForward%[tl.float16,tl.float32,tl.float32,indexCommuted] else fastLayernormForward%[tl.float16,tl.float32,tl.float32,original]
  | .bf16,.fp16,.fp16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float16,tl.float16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float16,tl.float16,original]
  | .bf16,.fp16,.bf16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float16,tl.bfloat16,original]
  | .bf16,.fp16,.fp32 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float16,tl.float32,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float16,tl.float32,original]
  | .bf16,.bf16,.fp16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float16,original]
  | .bf16,.bf16,.bf16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,original]
  | .bf16,.bf16,.fp32 => if swapped then fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.bfloat16,tl.float32,original]
  | .bf16,.fp32,.fp16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float32,tl.float16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float32,tl.float16,original]
  | .bf16,.fp32,.bf16 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float32,tl.bfloat16,original]
  | .bf16,.fp32,.fp32 => if swapped then fastLayernormForward%[tl.bfloat16,tl.float32,tl.float32,indexCommuted] else fastLayernormForward%[tl.bfloat16,tl.float32,tl.float32,original]
  | .fp32,.fp16,.fp16 => if swapped then fastLayernormForward%[tl.float32,tl.float16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float16,tl.float16,original]
  | .fp32,.fp16,.bf16 => if swapped then fastLayernormForward%[tl.float32,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float16,tl.bfloat16,original]
  | .fp32,.fp16,.fp32 => if swapped then fastLayernormForward%[tl.float32,tl.float16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float32,tl.float16,tl.float32,original]
  | .fp32,.bf16,.fp16 => if swapped then fastLayernormForward%[tl.float32,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormForward%[tl.float32,tl.bfloat16,tl.float16,original]
  | .fp32,.bf16,.bf16 => if swapped then fastLayernormForward%[tl.float32,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float32,tl.bfloat16,tl.bfloat16,original]
  | .fp32,.bf16,.fp32 => if swapped then fastLayernormForward%[tl.float32,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormForward%[tl.float32,tl.bfloat16,tl.float32,original]
  | .fp32,.fp32,.fp16 => if swapped then fastLayernormForward%[tl.float32,tl.float32,tl.float16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float32,tl.float16,original]
  | .fp32,.fp32,.bf16 => if swapped then fastLayernormForward%[tl.float32,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormForward%[tl.float32,tl.float32,tl.bfloat16,original]
  | .fp32,.fp32,.fp32 => if swapped then fastLayernormForward%[tl.float32,tl.float32,tl.float32,indexCommuted] else fastLayernormForward%[tl.float32,tl.float32,tl.float32,original]

syntax "fastLayernormBackward%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastLayernormBackward%[$inputTy:tritonDType,$weightTy:tritonDType,$biasTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| X += $(X_row_stride) * row_idx)
      else `(tritonStmt| X += row_idx * $(X_row_stride))
    `(fun (dY X W b r mu : RegionName) (dY_row_stride X_row_stride n_cols BLOCK_SIZE : Nat) (eps : ℝ) =>
      ComputeKernel.mk [dY,X,W,b,r,mu] [dY] (triton {
  tl.region dY = $inputTy, X = $inputTy, W = $weightTy, b = $biasTy, r = tl.float32, mu = tl.float32
  row_idx = tl.program_id(0)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(n_cols)
  dY += row_idx * $(dY_row_stride)
  $rowIndex
  r += row_idx
  mu += row_idx
  dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
  X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
  W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
  b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)
  inv_var = tl.load(r).to(tl.float32)
  mean = tl.load(mu).to(tl.float32)
  normed = (X_row - mean) * inv_var
  dY_W = dY_row * W_row
  dX_row = dY_W - tl.sum(dY_W, axis=0) / $(n_cols) - normed * tl.sum(dY_W * normed, axis=0) / $(n_cols)
  dX_row = dX_row * inv_var
  tl.store(dY + col_offsets, dX_row.to($inputTy:tritonDType), mask=mask)
}).surfaceBody)

def backward (input weight bias : Storage) (swapped : Bool) := match input,weight,bias with
  | .fp16,.fp16,.fp16 => if swapped then fastLayernormBackward%[tl.float16,tl.float16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float16,tl.float16,original]
  | .fp16,.fp16,.bf16 => if swapped then fastLayernormBackward%[tl.float16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float16,tl.bfloat16,original]
  | .fp16,.fp16,.fp32 => if swapped then fastLayernormBackward%[tl.float16,tl.float16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float16,tl.float32,original]
  | .fp16,.bf16,.fp16 => if swapped then fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float16,original]
  | .fp16,.bf16,.bf16 => if swapped then fastLayernormBackward%[tl.float16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.bfloat16,tl.bfloat16,original]
  | .fp16,.bf16,.fp32 => if swapped then fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float16,tl.bfloat16,tl.float32,original]
  | .fp16,.fp32,.fp16 => if swapped then fastLayernormBackward%[tl.float16,tl.float32,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float32,tl.float16,original]
  | .fp16,.fp32,.bf16 => if swapped then fastLayernormBackward%[tl.float16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float32,tl.bfloat16,original]
  | .fp16,.fp32,.fp32 => if swapped then fastLayernormBackward%[tl.float16,tl.float32,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float16,tl.float32,tl.float32,original]
  | .bf16,.fp16,.fp16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float16,original]
  | .bf16,.fp16,.bf16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float16,tl.bfloat16,original]
  | .bf16,.fp16,.fp32 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float16,tl.float32,original]
  | .bf16,.bf16,.fp16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float16,original]
  | .bf16,.bf16,.bf16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.bfloat16,original]
  | .bf16,.bf16,.fp32 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.bfloat16,tl.float32,original]
  | .bf16,.fp32,.fp16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float16,original]
  | .bf16,.fp32,.bf16 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float32,tl.bfloat16,original]
  | .bf16,.fp32,.fp32 => if swapped then fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float32,indexCommuted] else fastLayernormBackward%[tl.bfloat16,tl.float32,tl.float32,original]
  | .fp32,.fp16,.fp16 => if swapped then fastLayernormBackward%[tl.float32,tl.float16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float16,tl.float16,original]
  | .fp32,.fp16,.bf16 => if swapped then fastLayernormBackward%[tl.float32,tl.float16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float16,tl.bfloat16,original]
  | .fp32,.fp16,.fp32 => if swapped then fastLayernormBackward%[tl.float32,tl.float16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float16,tl.float32,original]
  | .fp32,.bf16,.fp16 => if swapped then fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float16,original]
  | .fp32,.bf16,.bf16 => if swapped then fastLayernormBackward%[tl.float32,tl.bfloat16,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.bfloat16,tl.bfloat16,original]
  | .fp32,.bf16,.fp32 => if swapped then fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float32,tl.bfloat16,tl.float32,original]
  | .fp32,.fp32,.fp16 => if swapped then fastLayernormBackward%[tl.float32,tl.float32,tl.float16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float32,tl.float16,original]
  | .fp32,.fp32,.bf16 => if swapped then fastLayernormBackward%[tl.float32,tl.float32,tl.bfloat16,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float32,tl.bfloat16,original]
  | .fp32,.fp32,.fp32 => if swapped then fastLayernormBackward%[tl.float32,tl.float32,tl.float32,indexCommuted] else fastLayernormBackward%[tl.float32,tl.float32,tl.float32,original]

end VeriTile.Bench.TritonBenchG.FastLayernorm.TypedSource
