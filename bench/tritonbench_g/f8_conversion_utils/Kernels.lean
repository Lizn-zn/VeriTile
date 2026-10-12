import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Complete fp8 conversion kernels, retaining typed pointers, omitted load
padding and implicit destination conversion. The two identical stores in the
toHalf source are both retained. The candidate commutes only the integer
block index. Host reinterpret/launch operations are outside these kernels. -/
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
open Lean VeriTile.Triton
inductive Format where
  | e4 | e5
  deriving DecidableEq
inductive Input where
  | fp16 | fp32
  deriving DecidableEq

-- Source: f8_conversion_utils.py, kernel_f8_to_f16.
syntax "f8Forward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(f8Forward%[$src:tritonDType,$dst:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let offsets : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| offs = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE)))
      else `(tritonStmt| offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE)))
    `(fun (Y X : RegionName) (N BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [X] [Y] (triton {
  tl.region X = $src, Y = $dst
  pid = tl.program_id(0)
  $offsets
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to($dst), mask=mask)
  tl.store(Y + offs, x.to($dst), mask=mask)
}).surfaceBody)

-- Source: f8_conversion_utils.py, kernel_f16_to_f8.
syntax "f8Reverse%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(f8Reverse%[$src:tritonDType,$dst:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let offsets : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| offs = $(BLOCK_SIZE) * pid + tl.arange(0, $(BLOCK_SIZE)))
      else `(tritonStmt| offs = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE)))
    `(fun (Y X : RegionName) (N BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [X] [Y] (triton {
  tl.region X = $src, Y = $dst
  pid = tl.program_id(0)
  $offsets
  mask = offs < $(N)
  x = tl.load(X + offs, mask=mask)
  tl.store(Y + offs, x.to($dst), mask=mask)
}).surfaceBody)

abbrev sourceType := RegionName → RegionName → Nat → Nat → ComputeKernel

def toHalf (format : Format) (swapped : Bool) : sourceType :=
  match format,swapped with
  | .e4,Bool.false => f8Forward%[tl.float8e4nv,tl.float16,original]
  | .e4,Bool.true => f8Forward%[tl.float8e4nv,tl.float16,indexCommuted]
  | .e5,Bool.false => f8Forward%[tl.float8e5,tl.float16,original]
  | .e5,Bool.true => f8Forward%[tl.float8e5,tl.float16,indexCommuted]

def toFP8 (input : Input) (format : Format) (swapped : Bool) : sourceType :=
  match input,format,swapped with
  | .fp16,.e4,Bool.false => f8Reverse%[tl.float16,tl.float8e4nv,original]
  | .fp16,.e4,Bool.true => f8Reverse%[tl.float16,tl.float8e4nv,indexCommuted]
  | .fp16,.e5,Bool.false => f8Reverse%[tl.float16,tl.float8e5,original]
  | .fp16,.e5,Bool.true => f8Reverse%[tl.float16,tl.float8e5,indexCommuted]
  | .fp32,.e4,Bool.false => f8Reverse%[tl.float32,tl.float8e4nv,original]
  | .fp32,.e4,Bool.true => f8Reverse%[tl.float32,tl.float8e4nv,indexCommuted]
  | .fp32,.e5,Bool.false => f8Reverse%[tl.float32,tl.float8e5,original]
  | .fp32,.e5,Bool.true => f8Reverse%[tl.float32,tl.float8e5,indexCommuted]
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
