import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.L2NormTriton1
open VeriTile.Triton
/-- Faithful transcription of `l2_norm_triton1.py`'s `_l2_norm_fwd_1pass_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
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

/-! Typed full-source family. The runtime epsilon is fp32. The explicit output
cast spells out the conversion performed by tl.store after pointer rebinding.
The candidate commutes only the two natural-number row-base products. -/
open Lean
syntax "l2Source%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(l2Source%[$dtype:tritonDType, $version:ident]) => do
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

inductive Storage where
  | fp32 | fp16 | bf16 | fp64
  deriving DecidableEq, Repr

def typedKernel (storage : Storage) (commuted : Bool)
    (X Y : RegionName) (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : ComputeKernel :=
  match storage, commuted with
  | .fp32, Bool.false => l2Source%[tl.float32, original] X Y stride_x_row N eps BLOCK_N
  | .fp32, Bool.true => l2Source%[tl.float32, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.false => l2Source%[tl.float16, original] X Y stride_x_row N eps BLOCK_N
  | .fp16, Bool.true => l2Source%[tl.float16, commuted] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.false => l2Source%[tl.bfloat16, original] X Y stride_x_row N eps BLOCK_N
  | .bf16, Bool.true => l2Source%[tl.bfloat16, commuted] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.false => l2Source%[tl.float64, original] X Y stride_x_row N eps BLOCK_N
  | .fp64, Bool.true => l2Source%[tl.float64, commuted] X Y stride_x_row N eps BLOCK_N

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton1
