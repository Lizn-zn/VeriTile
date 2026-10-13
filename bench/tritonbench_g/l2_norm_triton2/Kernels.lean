import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.L2NormTriton2
open VeriTile.Triton
/-- Faithful transcription of `l2_norm_triton2.py`'s `_l2_norm_fwd_1pass_kernel`. -/
def l2_norm_fwd_1pass_kernel
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y + cols, y, mask=mask)
}

/-- Faithful transcription of `l2_norm_triton2.py`'s `_l2_norm_bwd_kernel`. -/
def l2_norm_bwd_kernel
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  DX += row * $(stride_x_row)
  DY += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(x * x)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  dy = tl.load(DY + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  dy = tl.where(cols < $(N), dy, 0.0)
  dx = dy * rstd - tl.sum(dy * x) * (1 / (var + $(eps))) * rstd * x
  tl.store(DX + cols, dx, mask=mask)
}

/-! Typed full-source family. The runtime epsilon is fp32. The explicit output
cast spells out the conversion performed by tl.store after pointer rebinding.
The candidate commutes only the two natural-number row-base products. -/
open Lean
syntax "l2TwoFwd%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(l2TwoFwd%[$dtype:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let xbase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| X += row * $(stride_x_row))
      else `(tritonStmt| X += $(stride_x_row) * row)
    let ybase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| Y += row * $(stride_x_row))
      else `(tritonStmt| Y += $(stride_x_row) * row)
    let write : TSyntax `tritonStmt ← match dtype with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(Y + cols, y, mask=mask))
      | _ => `(tritonStmt| tl.store(Y + cols, y.to($dtype), mask=mask))
    `(fun (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) =>
      ComputeKernel.mk [X] [Y] (triton {
  tl.region X = $dtype, Y = $dtype
  row = tl.program_id(0)
  $xbase
  $ybase
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  mask = cols < $(N)
  y = x * rstd
  $write
}).surfaceBody)

/-! Typed full-source family. The runtime epsilon is fp32. The explicit output
cast spells out the conversion performed by tl.store after pointer rebinding.
The candidate commutes only the two natural-number row-base products. -/
open Lean
syntax "l2TwoBwd%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(l2TwoBwd%[$dtype:tritonDType, $grad:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let xbase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| X += row * $(stride_x_row))
      else `(tritonStmt| X += $(stride_x_row) * row)
    let dybase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| DY += row * $(stride_x_row))
      else `(tritonStmt| DY += $(stride_x_row) * row)
    let ybase : TSyntax `tritonStmt ←
      if version.getId == `original then `(tritonStmt| DX += row * $(stride_x_row))
      else `(tritonStmt| DX += $(stride_x_row) * row)
    let write : TSyntax `tritonStmt ← match dtype with
      | `(tritonDType| tl.float32) => `(tritonStmt| tl.store(DX + cols, dx, mask=mask))
      | _ => `(tritonStmt| tl.store(DX + cols, dx.to($dtype), mask=mask))
    `(fun (X DY DX : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) =>
      ComputeKernel.mk [X, DY] [DX] (triton {
  tl.region X = $dtype, DY = $grad, DX = $dtype
  row = tl.program_id(0)
  $xbase
  $ybase
  $dybase
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(x * x)
  rstd = 1 / tl.sqrt(var + $((eps : ℝ)).to(tl.float32))
  mask = cols < $(N)
  dy = tl.load(DY + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  dy = tl.where(cols < $(N), dy, 0.0)
  dx = dy * rstd - tl.sum(dy * x) * (1 / (var + $((eps : ℝ)).to(tl.float32))) * rstd * x
  $write
}).surfaceBody)


inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr

def typedFwd (storage : Storage) (commuted : Bool)
    (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : ComputeKernel :=
  match storage, commuted with
  | .fp32, Bool.false => l2TwoFwd%[tl.float32, original] X Y stride_x_row N eps BLOCK_N
  | .fp32, Bool.true => l2TwoFwd%[tl.float32, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.false => l2TwoFwd%[tl.float16, original] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.true => l2TwoFwd%[tl.float16, commuted] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.false => l2TwoFwd%[tl.bfloat16, original] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.true => l2TwoFwd%[tl.bfloat16, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.false => l2TwoFwd%[tl.float64, original] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.true => l2TwoFwd%[tl.float64, commuted] X Y stride_x_row N eps BLOCK_N

def typedBwd (storage gradient : Storage) (commuted : Bool)
    (X DY DX : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : ComputeKernel :=
  match storage, gradient, commuted with
  | .fp32, .fp32, Bool.false =>
    l2TwoBwd%[tl.float32, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp32, Bool.true =>
    l2TwoBwd%[tl.float32, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp16, Bool.false =>
    l2TwoBwd%[tl.float32, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp16, Bool.true =>
    l2TwoBwd%[tl.float32, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .bf16, Bool.false =>
    l2TwoBwd%[tl.float32, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .bf16, Bool.true =>
    l2TwoBwd%[tl.float32, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp64, Bool.false =>
    l2TwoBwd%[tl.float32, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp32, .fp64, Bool.true =>
    l2TwoBwd%[tl.float32, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp32, Bool.false =>
    l2TwoBwd%[tl.float16, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp32, Bool.true =>
    l2TwoBwd%[tl.float16, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp16, Bool.false =>
    l2TwoBwd%[tl.float16, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp16, Bool.true =>
    l2TwoBwd%[tl.float16, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .bf16, Bool.false =>
    l2TwoBwd%[tl.float16, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .bf16, Bool.true =>
    l2TwoBwd%[tl.float16, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp64, Bool.false =>
    l2TwoBwd%[tl.float16, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp16, .fp64, Bool.true =>
    l2TwoBwd%[tl.float16, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp32, Bool.false =>
    l2TwoBwd%[tl.bfloat16, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp32, Bool.true =>
    l2TwoBwd%[tl.bfloat16, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp16, Bool.false =>
    l2TwoBwd%[tl.bfloat16, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp16, Bool.true =>
    l2TwoBwd%[tl.bfloat16, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .bf16, Bool.false =>
    l2TwoBwd%[tl.bfloat16, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .bf16, Bool.true =>
    l2TwoBwd%[tl.bfloat16, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp64, Bool.false =>
    l2TwoBwd%[tl.bfloat16, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .bf16, .fp64, Bool.true =>
    l2TwoBwd%[tl.bfloat16, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp32, Bool.false =>
    l2TwoBwd%[tl.float64, tl.float32, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp32, Bool.true =>
    l2TwoBwd%[tl.float64, tl.float32, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp16, Bool.false =>
    l2TwoBwd%[tl.float64, tl.float16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp16, Bool.true =>
    l2TwoBwd%[tl.float64, tl.float16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .bf16, Bool.false =>
    l2TwoBwd%[tl.float64, tl.bfloat16, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .bf16, Bool.true =>
    l2TwoBwd%[tl.float64, tl.bfloat16, commuted] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp64, Bool.false =>
    l2TwoBwd%[tl.float64, tl.float64, original] X DY DX stride_x_row N eps BLOCK_N
  | .fp64, .fp64, Bool.true =>
    l2TwoBwd%[tl.float64, tl.float64, commuted] X DY DX stride_x_row N eps BLOCK_N

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton2
