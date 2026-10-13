import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
open VeriTile.Triton

open Lean
syntax "rmsLlamaTyped%" "[" tritonDType "," tritonDType "," tritonDType "]" : term
set_option hygiene false in
macro_rules
  | `(rmsLlamaTyped%[$input:tritonDType, $weight:tritonDType, $output:tritonDType]) => do
    let half := match output with
      | `(tritonDType| tl.float16) => Bool.true
      | _ => Bool.false
    let store : TSyntax `tritonStmt ← if half then
      `(tritonStmt| tl.store(Y + cols, y.to(tl.float16), mask=mask))
      else `(tritonStmt| tl.store(Y + cols, (y.to(tl.float16)).to($output:tritonDType), mask=mask))
    `(fun (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) =>
      ComputeKernel.mk [X, W] [Y] (triton {
  tl.region X = $input, W = $weight, Y = $output
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(B)], dtype=tl.float32)
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(B)) {
    cols = off + tl.arange(0, $(B))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    $store
  }
}).surfaceBody)

/-- Floating storage varies independently; the wrapper correlates output=input. -/
inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → RegionName → Nat → Nat → Nat → ℝ → ComputeKernel
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

/-- Complete source body with both loops and explicit fp32 input/weight conversions.
The explicit fp16 output conversion precedes the destination storage conversion. -/
def originalTyped (input weight output : Storage) : Source :=
  match input, weight, output with
  | .fp32, .fp32, .fp32 => rmsLlamaTyped%[tl.float32, tl.float32, tl.float32]
  | .fp32, .fp32, .fp64 => rmsLlamaTyped%[tl.float32, tl.float32, tl.float64]
  | .fp32, .fp32, .fp16 => rmsLlamaTyped%[tl.float32, tl.float32, tl.float16]
  | .fp32, .fp32, .bf16 => rmsLlamaTyped%[tl.float32, tl.float32, tl.bfloat16]
  | .fp32, .fp32, .f8e4 => rmsLlamaTyped%[tl.float32, tl.float32, tl.float8e4nv]
  | .fp32, .fp32, .f8e5 => rmsLlamaTyped%[tl.float32, tl.float32, tl.float8e5]
  | .fp32, .fp64, .fp32 => rmsLlamaTyped%[tl.float32, tl.float64, tl.float32]
  | .fp32, .fp64, .fp64 => rmsLlamaTyped%[tl.float32, tl.float64, tl.float64]
  | .fp32, .fp64, .fp16 => rmsLlamaTyped%[tl.float32, tl.float64, tl.float16]
  | .fp32, .fp64, .bf16 => rmsLlamaTyped%[tl.float32, tl.float64, tl.bfloat16]
  | .fp32, .fp64, .f8e4 => rmsLlamaTyped%[tl.float32, tl.float64, tl.float8e4nv]
  | .fp32, .fp64, .f8e5 => rmsLlamaTyped%[tl.float32, tl.float64, tl.float8e5]
  | .fp32, .fp16, .fp32 => rmsLlamaTyped%[tl.float32, tl.float16, tl.float32]
  | .fp32, .fp16, .fp64 => rmsLlamaTyped%[tl.float32, tl.float16, tl.float64]
  | .fp32, .fp16, .fp16 => rmsLlamaTyped%[tl.float32, tl.float16, tl.float16]
  | .fp32, .fp16, .bf16 => rmsLlamaTyped%[tl.float32, tl.float16, tl.bfloat16]
  | .fp32, .fp16, .f8e4 => rmsLlamaTyped%[tl.float32, tl.float16, tl.float8e4nv]
  | .fp32, .fp16, .f8e5 => rmsLlamaTyped%[tl.float32, tl.float16, tl.float8e5]
  | .fp32, .bf16, .fp32 => rmsLlamaTyped%[tl.float32, tl.bfloat16, tl.float32]
  | .fp32, .bf16, .fp64 => rmsLlamaTyped%[tl.float32, tl.bfloat16, tl.float64]
  | .fp32, .bf16, .fp16 => rmsLlamaTyped%[tl.float32, tl.bfloat16, tl.float16]
  | .fp32, .bf16, .bf16 => rmsLlamaTyped%[tl.float32, tl.bfloat16, tl.bfloat16]
  | .fp32, .bf16, .f8e4 => rmsLlamaTyped%[tl.float32, tl.bfloat16, tl.float8e4nv]
  | .fp32, .bf16, .f8e5 => rmsLlamaTyped%[tl.float32, tl.bfloat16, tl.float8e5]
  | .fp32, .f8e4, .fp32 => rmsLlamaTyped%[tl.float32, tl.float8e4nv, tl.float32]
  | .fp32, .f8e4, .fp64 => rmsLlamaTyped%[tl.float32, tl.float8e4nv, tl.float64]
  | .fp32, .f8e4, .fp16 => rmsLlamaTyped%[tl.float32, tl.float8e4nv, tl.float16]
  | .fp32, .f8e4, .bf16 => rmsLlamaTyped%[tl.float32, tl.float8e4nv, tl.bfloat16]
  | .fp32, .f8e4, .f8e4 => rmsLlamaTyped%[tl.float32, tl.float8e4nv, tl.float8e4nv]
  | .fp32, .f8e4, .f8e5 => rmsLlamaTyped%[tl.float32, tl.float8e4nv, tl.float8e5]
  | .fp32, .f8e5, .fp32 => rmsLlamaTyped%[tl.float32, tl.float8e5, tl.float32]
  | .fp32, .f8e5, .fp64 => rmsLlamaTyped%[tl.float32, tl.float8e5, tl.float64]
  | .fp32, .f8e5, .fp16 => rmsLlamaTyped%[tl.float32, tl.float8e5, tl.float16]
  | .fp32, .f8e5, .bf16 => rmsLlamaTyped%[tl.float32, tl.float8e5, tl.bfloat16]
  | .fp32, .f8e5, .f8e4 => rmsLlamaTyped%[tl.float32, tl.float8e5, tl.float8e4nv]
  | .fp32, .f8e5, .f8e5 => rmsLlamaTyped%[tl.float32, tl.float8e5, tl.float8e5]
  | .fp64, .fp32, .fp32 => rmsLlamaTyped%[tl.float64, tl.float32, tl.float32]
  | .fp64, .fp32, .fp64 => rmsLlamaTyped%[tl.float64, tl.float32, tl.float64]
  | .fp64, .fp32, .fp16 => rmsLlamaTyped%[tl.float64, tl.float32, tl.float16]
  | .fp64, .fp32, .bf16 => rmsLlamaTyped%[tl.float64, tl.float32, tl.bfloat16]
  | .fp64, .fp32, .f8e4 => rmsLlamaTyped%[tl.float64, tl.float32, tl.float8e4nv]
  | .fp64, .fp32, .f8e5 => rmsLlamaTyped%[tl.float64, tl.float32, tl.float8e5]
  | .fp64, .fp64, .fp32 => rmsLlamaTyped%[tl.float64, tl.float64, tl.float32]
  | .fp64, .fp64, .fp64 => rmsLlamaTyped%[tl.float64, tl.float64, tl.float64]
  | .fp64, .fp64, .fp16 => rmsLlamaTyped%[tl.float64, tl.float64, tl.float16]
  | .fp64, .fp64, .bf16 => rmsLlamaTyped%[tl.float64, tl.float64, tl.bfloat16]
  | .fp64, .fp64, .f8e4 => rmsLlamaTyped%[tl.float64, tl.float64, tl.float8e4nv]
  | .fp64, .fp64, .f8e5 => rmsLlamaTyped%[tl.float64, tl.float64, tl.float8e5]
  | .fp64, .fp16, .fp32 => rmsLlamaTyped%[tl.float64, tl.float16, tl.float32]
  | .fp64, .fp16, .fp64 => rmsLlamaTyped%[tl.float64, tl.float16, tl.float64]
  | .fp64, .fp16, .fp16 => rmsLlamaTyped%[tl.float64, tl.float16, tl.float16]
  | .fp64, .fp16, .bf16 => rmsLlamaTyped%[tl.float64, tl.float16, tl.bfloat16]
  | .fp64, .fp16, .f8e4 => rmsLlamaTyped%[tl.float64, tl.float16, tl.float8e4nv]
  | .fp64, .fp16, .f8e5 => rmsLlamaTyped%[tl.float64, tl.float16, tl.float8e5]
  | .fp64, .bf16, .fp32 => rmsLlamaTyped%[tl.float64, tl.bfloat16, tl.float32]
  | .fp64, .bf16, .fp64 => rmsLlamaTyped%[tl.float64, tl.bfloat16, tl.float64]
  | .fp64, .bf16, .fp16 => rmsLlamaTyped%[tl.float64, tl.bfloat16, tl.float16]
  | .fp64, .bf16, .bf16 => rmsLlamaTyped%[tl.float64, tl.bfloat16, tl.bfloat16]
  | .fp64, .bf16, .f8e4 => rmsLlamaTyped%[tl.float64, tl.bfloat16, tl.float8e4nv]
  | .fp64, .bf16, .f8e5 => rmsLlamaTyped%[tl.float64, tl.bfloat16, tl.float8e5]
  | .fp64, .f8e4, .fp32 => rmsLlamaTyped%[tl.float64, tl.float8e4nv, tl.float32]
  | .fp64, .f8e4, .fp64 => rmsLlamaTyped%[tl.float64, tl.float8e4nv, tl.float64]
  | .fp64, .f8e4, .fp16 => rmsLlamaTyped%[tl.float64, tl.float8e4nv, tl.float16]
  | .fp64, .f8e4, .bf16 => rmsLlamaTyped%[tl.float64, tl.float8e4nv, tl.bfloat16]
  | .fp64, .f8e4, .f8e4 => rmsLlamaTyped%[tl.float64, tl.float8e4nv, tl.float8e4nv]
  | .fp64, .f8e4, .f8e5 => rmsLlamaTyped%[tl.float64, tl.float8e4nv, tl.float8e5]
  | .fp64, .f8e5, .fp32 => rmsLlamaTyped%[tl.float64, tl.float8e5, tl.float32]
  | .fp64, .f8e5, .fp64 => rmsLlamaTyped%[tl.float64, tl.float8e5, tl.float64]
  | .fp64, .f8e5, .fp16 => rmsLlamaTyped%[tl.float64, tl.float8e5, tl.float16]
  | .fp64, .f8e5, .bf16 => rmsLlamaTyped%[tl.float64, tl.float8e5, tl.bfloat16]
  | .fp64, .f8e5, .f8e4 => rmsLlamaTyped%[tl.float64, tl.float8e5, tl.float8e4nv]
  | .fp64, .f8e5, .f8e5 => rmsLlamaTyped%[tl.float64, tl.float8e5, tl.float8e5]
  | .fp16, .fp32, .fp32 => rmsLlamaTyped%[tl.float16, tl.float32, tl.float32]
  | .fp16, .fp32, .fp64 => rmsLlamaTyped%[tl.float16, tl.float32, tl.float64]
  | .fp16, .fp32, .fp16 => rmsLlamaTyped%[tl.float16, tl.float32, tl.float16]
  | .fp16, .fp32, .bf16 => rmsLlamaTyped%[tl.float16, tl.float32, tl.bfloat16]
  | .fp16, .fp32, .f8e4 => rmsLlamaTyped%[tl.float16, tl.float32, tl.float8e4nv]
  | .fp16, .fp32, .f8e5 => rmsLlamaTyped%[tl.float16, tl.float32, tl.float8e5]
  | .fp16, .fp64, .fp32 => rmsLlamaTyped%[tl.float16, tl.float64, tl.float32]
  | .fp16, .fp64, .fp64 => rmsLlamaTyped%[tl.float16, tl.float64, tl.float64]
  | .fp16, .fp64, .fp16 => rmsLlamaTyped%[tl.float16, tl.float64, tl.float16]
  | .fp16, .fp64, .bf16 => rmsLlamaTyped%[tl.float16, tl.float64, tl.bfloat16]
  | .fp16, .fp64, .f8e4 => rmsLlamaTyped%[tl.float16, tl.float64, tl.float8e4nv]
  | .fp16, .fp64, .f8e5 => rmsLlamaTyped%[tl.float16, tl.float64, tl.float8e5]
  | .fp16, .fp16, .fp32 => rmsLlamaTyped%[tl.float16, tl.float16, tl.float32]
  | .fp16, .fp16, .fp64 => rmsLlamaTyped%[tl.float16, tl.float16, tl.float64]
  | .fp16, .fp16, .fp16 => rmsLlamaTyped%[tl.float16, tl.float16, tl.float16]
  | .fp16, .fp16, .bf16 => rmsLlamaTyped%[tl.float16, tl.float16, tl.bfloat16]
  | .fp16, .fp16, .f8e4 => rmsLlamaTyped%[tl.float16, tl.float16, tl.float8e4nv]
  | .fp16, .fp16, .f8e5 => rmsLlamaTyped%[tl.float16, tl.float16, tl.float8e5]
  | .fp16, .bf16, .fp32 => rmsLlamaTyped%[tl.float16, tl.bfloat16, tl.float32]
  | .fp16, .bf16, .fp64 => rmsLlamaTyped%[tl.float16, tl.bfloat16, tl.float64]
  | .fp16, .bf16, .fp16 => rmsLlamaTyped%[tl.float16, tl.bfloat16, tl.float16]
  | .fp16, .bf16, .bf16 => rmsLlamaTyped%[tl.float16, tl.bfloat16, tl.bfloat16]
  | .fp16, .bf16, .f8e4 => rmsLlamaTyped%[tl.float16, tl.bfloat16, tl.float8e4nv]
  | .fp16, .bf16, .f8e5 => rmsLlamaTyped%[tl.float16, tl.bfloat16, tl.float8e5]
  | .fp16, .f8e4, .fp32 => rmsLlamaTyped%[tl.float16, tl.float8e4nv, tl.float32]
  | .fp16, .f8e4, .fp64 => rmsLlamaTyped%[tl.float16, tl.float8e4nv, tl.float64]
  | .fp16, .f8e4, .fp16 => rmsLlamaTyped%[tl.float16, tl.float8e4nv, tl.float16]
  | .fp16, .f8e4, .bf16 => rmsLlamaTyped%[tl.float16, tl.float8e4nv, tl.bfloat16]
  | .fp16, .f8e4, .f8e4 => rmsLlamaTyped%[tl.float16, tl.float8e4nv, tl.float8e4nv]
  | .fp16, .f8e4, .f8e5 => rmsLlamaTyped%[tl.float16, tl.float8e4nv, tl.float8e5]
  | .fp16, .f8e5, .fp32 => rmsLlamaTyped%[tl.float16, tl.float8e5, tl.float32]
  | .fp16, .f8e5, .fp64 => rmsLlamaTyped%[tl.float16, tl.float8e5, tl.float64]
  | .fp16, .f8e5, .fp16 => rmsLlamaTyped%[tl.float16, tl.float8e5, tl.float16]
  | .fp16, .f8e5, .bf16 => rmsLlamaTyped%[tl.float16, tl.float8e5, tl.bfloat16]
  | .fp16, .f8e5, .f8e4 => rmsLlamaTyped%[tl.float16, tl.float8e5, tl.float8e4nv]
  | .fp16, .f8e5, .f8e5 => rmsLlamaTyped%[tl.float16, tl.float8e5, tl.float8e5]
  | .bf16, .fp32, .fp32 => rmsLlamaTyped%[tl.bfloat16, tl.float32, tl.float32]
  | .bf16, .fp32, .fp64 => rmsLlamaTyped%[tl.bfloat16, tl.float32, tl.float64]
  | .bf16, .fp32, .fp16 => rmsLlamaTyped%[tl.bfloat16, tl.float32, tl.float16]
  | .bf16, .fp32, .bf16 => rmsLlamaTyped%[tl.bfloat16, tl.float32, tl.bfloat16]
  | .bf16, .fp32, .f8e4 => rmsLlamaTyped%[tl.bfloat16, tl.float32, tl.float8e4nv]
  | .bf16, .fp32, .f8e5 => rmsLlamaTyped%[tl.bfloat16, tl.float32, tl.float8e5]
  | .bf16, .fp64, .fp32 => rmsLlamaTyped%[tl.bfloat16, tl.float64, tl.float32]
  | .bf16, .fp64, .fp64 => rmsLlamaTyped%[tl.bfloat16, tl.float64, tl.float64]
  | .bf16, .fp64, .fp16 => rmsLlamaTyped%[tl.bfloat16, tl.float64, tl.float16]
  | .bf16, .fp64, .bf16 => rmsLlamaTyped%[tl.bfloat16, tl.float64, tl.bfloat16]
  | .bf16, .fp64, .f8e4 => rmsLlamaTyped%[tl.bfloat16, tl.float64, tl.float8e4nv]
  | .bf16, .fp64, .f8e5 => rmsLlamaTyped%[tl.bfloat16, tl.float64, tl.float8e5]
  | .bf16, .fp16, .fp32 => rmsLlamaTyped%[tl.bfloat16, tl.float16, tl.float32]
  | .bf16, .fp16, .fp64 => rmsLlamaTyped%[tl.bfloat16, tl.float16, tl.float64]
  | .bf16, .fp16, .fp16 => rmsLlamaTyped%[tl.bfloat16, tl.float16, tl.float16]
  | .bf16, .fp16, .bf16 => rmsLlamaTyped%[tl.bfloat16, tl.float16, tl.bfloat16]
  | .bf16, .fp16, .f8e4 => rmsLlamaTyped%[tl.bfloat16, tl.float16, tl.float8e4nv]
  | .bf16, .fp16, .f8e5 => rmsLlamaTyped%[tl.bfloat16, tl.float16, tl.float8e5]
  | .bf16, .bf16, .fp32 => rmsLlamaTyped%[tl.bfloat16, tl.bfloat16, tl.float32]
  | .bf16, .bf16, .fp64 => rmsLlamaTyped%[tl.bfloat16, tl.bfloat16, tl.float64]
  | .bf16, .bf16, .fp16 => rmsLlamaTyped%[tl.bfloat16, tl.bfloat16, tl.float16]
  | .bf16, .bf16, .bf16 => rmsLlamaTyped%[tl.bfloat16, tl.bfloat16, tl.bfloat16]
  | .bf16, .bf16, .f8e4 => rmsLlamaTyped%[tl.bfloat16, tl.bfloat16, tl.float8e4nv]
  | .bf16, .bf16, .f8e5 => rmsLlamaTyped%[tl.bfloat16, tl.bfloat16, tl.float8e5]
  | .bf16, .f8e4, .fp32 => rmsLlamaTyped%[tl.bfloat16, tl.float8e4nv, tl.float32]
  | .bf16, .f8e4, .fp64 => rmsLlamaTyped%[tl.bfloat16, tl.float8e4nv, tl.float64]
  | .bf16, .f8e4, .fp16 => rmsLlamaTyped%[tl.bfloat16, tl.float8e4nv, tl.float16]
  | .bf16, .f8e4, .bf16 => rmsLlamaTyped%[tl.bfloat16, tl.float8e4nv, tl.bfloat16]
  | .bf16, .f8e4, .f8e4 => rmsLlamaTyped%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv]
  | .bf16, .f8e4, .f8e5 => rmsLlamaTyped%[tl.bfloat16, tl.float8e4nv, tl.float8e5]
  | .bf16, .f8e5, .fp32 => rmsLlamaTyped%[tl.bfloat16, tl.float8e5, tl.float32]
  | .bf16, .f8e5, .fp64 => rmsLlamaTyped%[tl.bfloat16, tl.float8e5, tl.float64]
  | .bf16, .f8e5, .fp16 => rmsLlamaTyped%[tl.bfloat16, tl.float8e5, tl.float16]
  | .bf16, .f8e5, .bf16 => rmsLlamaTyped%[tl.bfloat16, tl.float8e5, tl.bfloat16]
  | .bf16, .f8e5, .f8e4 => rmsLlamaTyped%[tl.bfloat16, tl.float8e5, tl.float8e4nv]
  | .bf16, .f8e5, .f8e5 => rmsLlamaTyped%[tl.bfloat16, tl.float8e5, tl.float8e5]
  | .f8e4, .fp32, .fp32 => rmsLlamaTyped%[tl.float8e4nv, tl.float32, tl.float32]
  | .f8e4, .fp32, .fp64 => rmsLlamaTyped%[tl.float8e4nv, tl.float32, tl.float64]
  | .f8e4, .fp32, .fp16 => rmsLlamaTyped%[tl.float8e4nv, tl.float32, tl.float16]
  | .f8e4, .fp32, .bf16 => rmsLlamaTyped%[tl.float8e4nv, tl.float32, tl.bfloat16]
  | .f8e4, .fp32, .f8e4 => rmsLlamaTyped%[tl.float8e4nv, tl.float32, tl.float8e4nv]
  | .f8e4, .fp32, .f8e5 => rmsLlamaTyped%[tl.float8e4nv, tl.float32, tl.float8e5]
  | .f8e4, .fp64, .fp32 => rmsLlamaTyped%[tl.float8e4nv, tl.float64, tl.float32]
  | .f8e4, .fp64, .fp64 => rmsLlamaTyped%[tl.float8e4nv, tl.float64, tl.float64]
  | .f8e4, .fp64, .fp16 => rmsLlamaTyped%[tl.float8e4nv, tl.float64, tl.float16]
  | .f8e4, .fp64, .bf16 => rmsLlamaTyped%[tl.float8e4nv, tl.float64, tl.bfloat16]
  | .f8e4, .fp64, .f8e4 => rmsLlamaTyped%[tl.float8e4nv, tl.float64, tl.float8e4nv]
  | .f8e4, .fp64, .f8e5 => rmsLlamaTyped%[tl.float8e4nv, tl.float64, tl.float8e5]
  | .f8e4, .fp16, .fp32 => rmsLlamaTyped%[tl.float8e4nv, tl.float16, tl.float32]
  | .f8e4, .fp16, .fp64 => rmsLlamaTyped%[tl.float8e4nv, tl.float16, tl.float64]
  | .f8e4, .fp16, .fp16 => rmsLlamaTyped%[tl.float8e4nv, tl.float16, tl.float16]
  | .f8e4, .fp16, .bf16 => rmsLlamaTyped%[tl.float8e4nv, tl.float16, tl.bfloat16]
  | .f8e4, .fp16, .f8e4 => rmsLlamaTyped%[tl.float8e4nv, tl.float16, tl.float8e4nv]
  | .f8e4, .fp16, .f8e5 => rmsLlamaTyped%[tl.float8e4nv, tl.float16, tl.float8e5]
  | .f8e4, .bf16, .fp32 => rmsLlamaTyped%[tl.float8e4nv, tl.bfloat16, tl.float32]
  | .f8e4, .bf16, .fp64 => rmsLlamaTyped%[tl.float8e4nv, tl.bfloat16, tl.float64]
  | .f8e4, .bf16, .fp16 => rmsLlamaTyped%[tl.float8e4nv, tl.bfloat16, tl.float16]
  | .f8e4, .bf16, .bf16 => rmsLlamaTyped%[tl.float8e4nv, tl.bfloat16, tl.bfloat16]
  | .f8e4, .bf16, .f8e4 => rmsLlamaTyped%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv]
  | .f8e4, .bf16, .f8e5 => rmsLlamaTyped%[tl.float8e4nv, tl.bfloat16, tl.float8e5]
  | .f8e4, .f8e4, .fp32 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e4nv, tl.float32]
  | .f8e4, .f8e4, .fp64 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e4nv, tl.float64]
  | .f8e4, .f8e4, .fp16 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e4nv, tl.float16]
  | .f8e4, .f8e4, .bf16 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16]
  | .f8e4, .f8e4, .f8e4 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv]
  | .f8e4, .f8e4, .f8e5 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e4nv, tl.float8e5]
  | .f8e4, .f8e5, .fp32 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e5, tl.float32]
  | .f8e4, .f8e5, .fp64 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e5, tl.float64]
  | .f8e4, .f8e5, .fp16 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e5, tl.float16]
  | .f8e4, .f8e5, .bf16 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e5, tl.bfloat16]
  | .f8e4, .f8e5, .f8e4 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e5, tl.float8e4nv]
  | .f8e4, .f8e5, .f8e5 => rmsLlamaTyped%[tl.float8e4nv, tl.float8e5, tl.float8e5]
  | .f8e5, .fp32, .fp32 => rmsLlamaTyped%[tl.float8e5, tl.float32, tl.float32]
  | .f8e5, .fp32, .fp64 => rmsLlamaTyped%[tl.float8e5, tl.float32, tl.float64]
  | .f8e5, .fp32, .fp16 => rmsLlamaTyped%[tl.float8e5, tl.float32, tl.float16]
  | .f8e5, .fp32, .bf16 => rmsLlamaTyped%[tl.float8e5, tl.float32, tl.bfloat16]
  | .f8e5, .fp32, .f8e4 => rmsLlamaTyped%[tl.float8e5, tl.float32, tl.float8e4nv]
  | .f8e5, .fp32, .f8e5 => rmsLlamaTyped%[tl.float8e5, tl.float32, tl.float8e5]
  | .f8e5, .fp64, .fp32 => rmsLlamaTyped%[tl.float8e5, tl.float64, tl.float32]
  | .f8e5, .fp64, .fp64 => rmsLlamaTyped%[tl.float8e5, tl.float64, tl.float64]
  | .f8e5, .fp64, .fp16 => rmsLlamaTyped%[tl.float8e5, tl.float64, tl.float16]
  | .f8e5, .fp64, .bf16 => rmsLlamaTyped%[tl.float8e5, tl.float64, tl.bfloat16]
  | .f8e5, .fp64, .f8e4 => rmsLlamaTyped%[tl.float8e5, tl.float64, tl.float8e4nv]
  | .f8e5, .fp64, .f8e5 => rmsLlamaTyped%[tl.float8e5, tl.float64, tl.float8e5]
  | .f8e5, .fp16, .fp32 => rmsLlamaTyped%[tl.float8e5, tl.float16, tl.float32]
  | .f8e5, .fp16, .fp64 => rmsLlamaTyped%[tl.float8e5, tl.float16, tl.float64]
  | .f8e5, .fp16, .fp16 => rmsLlamaTyped%[tl.float8e5, tl.float16, tl.float16]
  | .f8e5, .fp16, .bf16 => rmsLlamaTyped%[tl.float8e5, tl.float16, tl.bfloat16]
  | .f8e5, .fp16, .f8e4 => rmsLlamaTyped%[tl.float8e5, tl.float16, tl.float8e4nv]
  | .f8e5, .fp16, .f8e5 => rmsLlamaTyped%[tl.float8e5, tl.float16, tl.float8e5]
  | .f8e5, .bf16, .fp32 => rmsLlamaTyped%[tl.float8e5, tl.bfloat16, tl.float32]
  | .f8e5, .bf16, .fp64 => rmsLlamaTyped%[tl.float8e5, tl.bfloat16, tl.float64]
  | .f8e5, .bf16, .fp16 => rmsLlamaTyped%[tl.float8e5, tl.bfloat16, tl.float16]
  | .f8e5, .bf16, .bf16 => rmsLlamaTyped%[tl.float8e5, tl.bfloat16, tl.bfloat16]
  | .f8e5, .bf16, .f8e4 => rmsLlamaTyped%[tl.float8e5, tl.bfloat16, tl.float8e4nv]
  | .f8e5, .bf16, .f8e5 => rmsLlamaTyped%[tl.float8e5, tl.bfloat16, tl.float8e5]
  | .f8e5, .f8e4, .fp32 => rmsLlamaTyped%[tl.float8e5, tl.float8e4nv, tl.float32]
  | .f8e5, .f8e4, .fp64 => rmsLlamaTyped%[tl.float8e5, tl.float8e4nv, tl.float64]
  | .f8e5, .f8e4, .fp16 => rmsLlamaTyped%[tl.float8e5, tl.float8e4nv, tl.float16]
  | .f8e5, .f8e4, .bf16 => rmsLlamaTyped%[tl.float8e5, tl.float8e4nv, tl.bfloat16]
  | .f8e5, .f8e4, .f8e4 => rmsLlamaTyped%[tl.float8e5, tl.float8e4nv, tl.float8e4nv]
  | .f8e5, .f8e4, .f8e5 => rmsLlamaTyped%[tl.float8e5, tl.float8e4nv, tl.float8e5]
  | .f8e5, .f8e5, .fp32 => rmsLlamaTyped%[tl.float8e5, tl.float8e5, tl.float32]
  | .f8e5, .f8e5, .fp64 => rmsLlamaTyped%[tl.float8e5, tl.float8e5, tl.float64]
  | .f8e5, .f8e5, .fp16 => rmsLlamaTyped%[tl.float8e5, tl.float8e5, tl.float16]
  | .f8e5, .f8e5, .bf16 => rmsLlamaTyped%[tl.float8e5, tl.float8e5, tl.bfloat16]
  | .f8e5, .f8e5, .f8e4 => rmsLlamaTyped%[tl.float8e5, tl.float8e5, tl.float8e4nv]
  | .f8e5, .f8e5, .f8e5 => rmsLlamaTyped%[tl.float8e5, tl.float8e5, tl.float8e5]

/-- Only the natural multiplication in the output row-pointer update commutes. -/
def typedKernel (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) : ComputeKernel :=
  let source := originalTyped input weight output X Y W stride N B eps
  if v then ComputeKernel.mk source.inputs source.outputs
    (source.surfaceBody.take 1 ++
      [.assign .ptr [] "Y" (.alg (.ptrAdd .nil (.ptrBase Y)
        (.mul .nat .nil (.constNat stride) (.ref .nat [] "row"))))] ++ source.surfaceBody.drop 2)
  else source

end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
