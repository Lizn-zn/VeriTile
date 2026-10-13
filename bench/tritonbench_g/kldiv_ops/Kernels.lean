import VeriTile.Triton

/-! Complete KL-divergence ops source families. Runtime Python-float epsilon is
represented at fp32. The accumulator initializer is the original `0.0` in every
profile; frontend-inadmissible fp64 reduction carries are not silently widened.
Native-device lowering is separate from the explicit frontend domain. -/
namespace VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
open Lean VeriTile.Triton

inductive Storage where
  | fp32 | fp64 | fp16 | bf16 | f8e4 | f8e5
  deriving DecidableEq, Repr

/-- Pinned ordinary exp accepts only these raw tensor storage types. -/
inductive MathStorage where
  | fp32 | fp64
  deriving DecidableEq, Repr

def MathStorage.storage : MathStorage → Storage
  | .fp32 => .fp32
  | .fp64 => .fp64

syntax "klOpsForward%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(klOpsForward%[$pred:tritonDType, $target:tritonDType, $output:tritonDType, $branch:ident]) => do
    let logarithmic := branch.getId == `logarithmic
    unless logarithmic || branch.getId == `defaultTarget do
      Macro.throwError "expected logarithmic or defaultTarget"
    let compute : TSyntax `tritonStmt ← if logarithmic then
      `(tritonStmt| loss = tl.exp(y_true) * (y_true - y))
      else `(tritonStmt| loss = y_true * (tl.log(tl.maximum(y_true,
        ($((eps : ℝ))).to(tl.float32))) - y))
    let wide := match pred, target with
      | `(tritonDType| tl.float64), _ | _, `(tritonDType| tl.float64) => Bool.true
      | _, _ => Bool.false
    let same := match output with
      | `(tritonDType| tl.float64) => wide
      | `(tritonDType| tl.float32) => !wide
      | _ => Bool.false
    let elementStore : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(O + offsets, loss, mask=mask))
      else `(tritonStmt| tl.store(O + offsets, loss.to($output), mask=mask))
    -- Admitted reduced profiles keep the initially fp32 carried value fp32.
    -- The formal fp64-carry profiles are explicitly excluded in SourceDomain.
    let accumulatorMatches := match output with
      | `(tritonDType| tl.float32) => Bool.true
      | _ => Bool.false
    let finalStore : TSyntax `tritonStmt ← if accumulatorMatches then
      `(tritonStmt| tl.store(O, loss_sum))
      else `(tritonStmt| tl.store(O, loss_sum.to($output)))
    `(fun (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) =>
      ComputeKernel.mk [Y, T] [O] (triton {
        tl.region Y = $pred, T = $target, O = $output
        pid = tl.program_id(0).to(tl.int64)
        Y += pid * $(sy)
        T += pid * $(st)
        O += pid * $(so)
        base_offsets = tl.arange(0, $(B))
        loss_sum = 0.0
        for i in range(0, $((N : Int)), $(B)) {
          offsets = i + base_offsets
          mask = offsets < $((N : Int))
          y = tl.load(Y + offsets, mask=mask, other=0.0)
          y_true = tl.load(T + offsets, mask=mask, other=0.0)
          $compute
          if $(reduction) == $(0) {
            $elementStore
          } else {
            loss_sum += tl.sum(loss, axis=0)
          }
        }
        if $(reduction) != $(0) {
          $finalStore
        }
      }).surfaceBody)

syntax "klOpsBackward%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(klOpsBackward%[$target:tritonDType, $output:tritonDType, $branch:ident]) => do
    let logarithmic := branch.getId == `logarithmic
    unless logarithmic || branch.getId == `defaultTarget do
      Macro.throwError "expected logarithmic or defaultTarget"
    let compute : TSyntax `tritonStmt ← if logarithmic then
      `(tritonStmt| res = -tl.exp(target))
      else `(tritonStmt| res = target * -1)
    let same := match target, output with
      | `(tritonDType| tl.float32), `(tritonDType| tl.float32)
      | `(tritonDType| tl.float64), `(tritonDType| tl.float64)
      | `(tritonDType| tl.float16), `(tritonDType| tl.float16)
      | `(tritonDType| tl.bfloat16), `(tritonDType| tl.bfloat16)
      | `(tritonDType| tl.float8e4nv), `(tritonDType| tl.float8e4nv)
      | `(tritonDType| tl.float8e5), `(tritonDType| tl.float8e5) => Bool.true
      | _, _ => Bool.false
    let store : TSyntax `tritonStmt ← if same then
      `(tritonStmt| tl.store(O + offsets, res, mask=mask))
      else `(tritonStmt| tl.store(O + offsets, res.to($output), mask=mask))
    `(fun (T O : RegionName) (st so N B : Nat) =>
      ComputeKernel.mk [T] [O] (triton {
        tl.region T = $target, O = $output
        pid = tl.program_id(0).to(tl.int64)
        T += pid * $(st)
        O += pid * $(so)
        offsets = tl.arange(0, $(B))
        mask = offsets < $((N : Int))
        for i in range(0, $((N : Int)), $(B)) {
          offsets = i + tl.arange(0, $(B))
          mask = offsets < $((N : Int))
          target = tl.load(T + offsets, mask=mask, other=0.0)
          $compute
          $store
        }
      }).surfaceBody)

abbrev Forward := RegionName → RegionName → RegionName → Nat → Nat → Nat → Nat → Nat →
  ℝ → Nat → ComputeKernel
abbrev Backward := RegionName → RegionName → Nat → Nat → Nat → Nat → ComputeKernel

def forwardDefault_fp32_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float32, tl.float8e5, defaultTarget]

def forwardDefault_fp32_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float64, tl.float8e5, defaultTarget]

def forwardDefault_fp32_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float16, tl.float8e5, defaultTarget]

def forwardDefault_fp32_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.bfloat16, tl.float8e5, defaultTarget]

def forwardDefault_fp32_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float8e4nv, tl.float8e5, defaultTarget]

def forwardDefault_fp32_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float32, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float32, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float32, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float32, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float32, tl.float8e5, tl.float8e5, defaultTarget]

def forwardDefault_fp64_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float32, tl.float8e5, defaultTarget]

def forwardDefault_fp64_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float64, tl.float8e5, defaultTarget]

def forwardDefault_fp64_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float16, tl.float8e5, defaultTarget]

def forwardDefault_fp64_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.bfloat16, tl.float8e5, defaultTarget]

def forwardDefault_fp64_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float8e4nv, tl.float8e5, defaultTarget]

def forwardDefault_fp64_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float64, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float64, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float64, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float64, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float64, tl.float8e5, tl.float8e5, defaultTarget]

def forwardDefault_fp16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float32, tl.float8e5, defaultTarget]

def forwardDefault_fp16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float64, tl.float8e5, defaultTarget]

def forwardDefault_fp16_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float16, tl.float8e5, defaultTarget]

def forwardDefault_fp16_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.bfloat16, tl.float8e5, defaultTarget]

def forwardDefault_fp16_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float8e4nv, tl.float8e5, defaultTarget]

def forwardDefault_fp16_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float16, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float16, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float16, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float16, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float16, tl.float8e5, tl.float8e5, defaultTarget]

def forwardDefault_bf16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e5, defaultTarget]

def forwardDefault_bf16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e5, defaultTarget]

def forwardDefault_bf16_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float16, tl.float8e5, defaultTarget]

def forwardDefault_bf16_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.bfloat16, tl.float8e5, defaultTarget]

def forwardDefault_bf16_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float8e4nv, tl.float8e5, defaultTarget]

def forwardDefault_bf16_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float8e5, tl.float8e5, defaultTarget]

def forwardDefault_f8e4_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e5, defaultTarget]

def forwardDefault_f8e4_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e5, defaultTarget]

def forwardDefault_f8e4_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float16, tl.float8e5, defaultTarget]

def forwardDefault_f8e4_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.bfloat16, tl.float8e5, defaultTarget]

def forwardDefault_f8e4_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float8e4nv, tl.float8e5, defaultTarget]

def forwardDefault_f8e4_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float8e5, tl.float8e5, defaultTarget]

def forwardDefault_f8e5_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float32, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float32, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float32, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float32, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e5, defaultTarget]

def forwardDefault_f8e5_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float64, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float64, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float64, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float64, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e5, defaultTarget]

def forwardDefault_f8e5_fp16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float16, tl.float8e5, defaultTarget]

def forwardDefault_f8e5_bf16 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.bfloat16, tl.float8e5, defaultTarget]

def forwardDefault_f8e5_f8e4 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float8e4nv, tl.float8e5, defaultTarget]

def forwardDefault_f8e5_f8e5 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float32, defaultTarget]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float64, defaultTarget]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float16, defaultTarget]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float8e5, tl.float8e5, defaultTarget]

/-- Complete forward source for the default target branch; all blocks and the final store remain. -/
def forwardDefaultOriginal (pred : Storage) (target : Storage) (output : Storage) : Forward :=
  match pred, target with
  | .fp32, .fp32 => forwardDefault_fp32_fp32 output
  | .fp32, .fp64 => forwardDefault_fp32_fp64 output
  | .fp32, .fp16 => forwardDefault_fp32_fp16 output
  | .fp32, .bf16 => forwardDefault_fp32_bf16 output
  | .fp32, .f8e4 => forwardDefault_fp32_f8e4 output
  | .fp32, .f8e5 => forwardDefault_fp32_f8e5 output
  | .fp64, .fp32 => forwardDefault_fp64_fp32 output
  | .fp64, .fp64 => forwardDefault_fp64_fp64 output
  | .fp64, .fp16 => forwardDefault_fp64_fp16 output
  | .fp64, .bf16 => forwardDefault_fp64_bf16 output
  | .fp64, .f8e4 => forwardDefault_fp64_f8e4 output
  | .fp64, .f8e5 => forwardDefault_fp64_f8e5 output
  | .fp16, .fp32 => forwardDefault_fp16_fp32 output
  | .fp16, .fp64 => forwardDefault_fp16_fp64 output
  | .fp16, .fp16 => forwardDefault_fp16_fp16 output
  | .fp16, .bf16 => forwardDefault_fp16_bf16 output
  | .fp16, .f8e4 => forwardDefault_fp16_f8e4 output
  | .fp16, .f8e5 => forwardDefault_fp16_f8e5 output
  | .bf16, .fp32 => forwardDefault_bf16_fp32 output
  | .bf16, .fp64 => forwardDefault_bf16_fp64 output
  | .bf16, .fp16 => forwardDefault_bf16_fp16 output
  | .bf16, .bf16 => forwardDefault_bf16_bf16 output
  | .bf16, .f8e4 => forwardDefault_bf16_f8e4 output
  | .bf16, .f8e5 => forwardDefault_bf16_f8e5 output
  | .f8e4, .fp32 => forwardDefault_f8e4_fp32 output
  | .f8e4, .fp64 => forwardDefault_f8e4_fp64 output
  | .f8e4, .fp16 => forwardDefault_f8e4_fp16 output
  | .f8e4, .bf16 => forwardDefault_f8e4_bf16 output
  | .f8e4, .f8e4 => forwardDefault_f8e4_f8e4 output
  | .f8e4, .f8e5 => forwardDefault_f8e4_f8e5 output
  | .f8e5, .fp32 => forwardDefault_f8e5_fp32 output
  | .f8e5, .fp64 => forwardDefault_f8e5_fp64 output
  | .f8e5, .fp16 => forwardDefault_f8e5_fp16 output
  | .f8e5, .bf16 => forwardDefault_f8e5_bf16 output
  | .f8e5, .f8e4 => forwardDefault_f8e5_f8e4 output
  | .f8e5, .f8e5 => forwardDefault_f8e5_f8e5 output

def forwardLog_fp32_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float32, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float32, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float32, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float32, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float32, tl.float32, tl.float8e5, logarithmic]

def forwardLog_fp32_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float32, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float32, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float32, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float32, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float32, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float32, tl.float64, tl.float8e5, logarithmic]

def forwardLog_fp64_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float64, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float64, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float64, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float64, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float64, tl.float32, tl.float8e5, logarithmic]

def forwardLog_fp64_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float64, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float64, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float64, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float64, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float64, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float64, tl.float64, tl.float8e5, logarithmic]

def forwardLog_fp16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float16, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float16, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float16, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float16, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float16, tl.float32, tl.float8e5, logarithmic]

def forwardLog_fp16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float16, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float16, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float16, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float16, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float16, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float16, tl.float64, tl.float8e5, logarithmic]

def forwardLog_bf16_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float32, tl.float8e5, logarithmic]

def forwardLog_bf16_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.bfloat16, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.bfloat16, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.bfloat16, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.bfloat16, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.bfloat16, tl.float64, tl.float8e5, logarithmic]

def forwardLog_f8e4_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float32, tl.float8e5, logarithmic]

def forwardLog_f8e4_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e4nv, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e4nv, tl.float64, tl.float8e5, logarithmic]

def forwardLog_f8e5_fp32 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float32, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float32, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float32, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float32, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float32, tl.float8e5, logarithmic]

def forwardLog_f8e5_fp64 (output : Storage) : Forward :=
  match output with
  | .fp32 => klOpsForward%[tl.float8e5, tl.float64, tl.float32, logarithmic]
  | .fp64 => klOpsForward%[tl.float8e5, tl.float64, tl.float64, logarithmic]
  | .fp16 => klOpsForward%[tl.float8e5, tl.float64, tl.float16, logarithmic]
  | .bf16 => klOpsForward%[tl.float8e5, tl.float64, tl.bfloat16, logarithmic]
  | .f8e4 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e4nv, logarithmic]
  | .f8e5 => klOpsForward%[tl.float8e5, tl.float64, tl.float8e5, logarithmic]

/-- Complete forward source for the log target branch; all blocks and the final store remain. -/
def forwardLogOriginal (pred : Storage) (target : MathStorage) (output : Storage) : Forward :=
  match pred, target with
  | .fp32, .fp32 => forwardLog_fp32_fp32 output
  | .fp32, .fp64 => forwardLog_fp32_fp64 output
  | .fp64, .fp32 => forwardLog_fp64_fp32 output
  | .fp64, .fp64 => forwardLog_fp64_fp64 output
  | .fp16, .fp32 => forwardLog_fp16_fp32 output
  | .fp16, .fp64 => forwardLog_fp16_fp64 output
  | .bf16, .fp32 => forwardLog_bf16_fp32 output
  | .bf16, .fp64 => forwardLog_bf16_fp64 output
  | .f8e4, .fp32 => forwardLog_f8e4_fp32 output
  | .f8e4, .fp64 => forwardLog_f8e4_fp64 output
  | .f8e5, .fp32 => forwardLog_f8e5_fp32 output
  | .f8e5, .fp64 => forwardLog_f8e5_fp64 output

/-- Complete backward default branch, retaining T then O pointer updates and the pre-loop mask. -/
def backwardDefaultOriginal (target : Storage) (output : Storage) : Backward :=
  match target, output with
  | .fp32, .fp32 => klOpsBackward%[tl.float32, tl.float32, defaultTarget]
  | .fp32, .fp64 => klOpsBackward%[tl.float32, tl.float64, defaultTarget]
  | .fp32, .fp16 => klOpsBackward%[tl.float32, tl.float16, defaultTarget]
  | .fp32, .bf16 => klOpsBackward%[tl.float32, tl.bfloat16, defaultTarget]
  | .fp32, .f8e4 => klOpsBackward%[tl.float32, tl.float8e4nv, defaultTarget]
  | .fp32, .f8e5 => klOpsBackward%[tl.float32, tl.float8e5, defaultTarget]
  | .fp64, .fp32 => klOpsBackward%[tl.float64, tl.float32, defaultTarget]
  | .fp64, .fp64 => klOpsBackward%[tl.float64, tl.float64, defaultTarget]
  | .fp64, .fp16 => klOpsBackward%[tl.float64, tl.float16, defaultTarget]
  | .fp64, .bf16 => klOpsBackward%[tl.float64, tl.bfloat16, defaultTarget]
  | .fp64, .f8e4 => klOpsBackward%[tl.float64, tl.float8e4nv, defaultTarget]
  | .fp64, .f8e5 => klOpsBackward%[tl.float64, tl.float8e5, defaultTarget]
  | .fp16, .fp32 => klOpsBackward%[tl.float16, tl.float32, defaultTarget]
  | .fp16, .fp64 => klOpsBackward%[tl.float16, tl.float64, defaultTarget]
  | .fp16, .fp16 => klOpsBackward%[tl.float16, tl.float16, defaultTarget]
  | .fp16, .bf16 => klOpsBackward%[tl.float16, tl.bfloat16, defaultTarget]
  | .fp16, .f8e4 => klOpsBackward%[tl.float16, tl.float8e4nv, defaultTarget]
  | .fp16, .f8e5 => klOpsBackward%[tl.float16, tl.float8e5, defaultTarget]
  | .bf16, .fp32 => klOpsBackward%[tl.bfloat16, tl.float32, defaultTarget]
  | .bf16, .fp64 => klOpsBackward%[tl.bfloat16, tl.float64, defaultTarget]
  | .bf16, .fp16 => klOpsBackward%[tl.bfloat16, tl.float16, defaultTarget]
  | .bf16, .bf16 => klOpsBackward%[tl.bfloat16, tl.bfloat16, defaultTarget]
  | .bf16, .f8e4 => klOpsBackward%[tl.bfloat16, tl.float8e4nv, defaultTarget]
  | .bf16, .f8e5 => klOpsBackward%[tl.bfloat16, tl.float8e5, defaultTarget]
  | .f8e4, .fp32 => klOpsBackward%[tl.float8e4nv, tl.float32, defaultTarget]
  | .f8e4, .fp64 => klOpsBackward%[tl.float8e4nv, tl.float64, defaultTarget]
  | .f8e4, .fp16 => klOpsBackward%[tl.float8e4nv, tl.float16, defaultTarget]
  | .f8e4, .bf16 => klOpsBackward%[tl.float8e4nv, tl.bfloat16, defaultTarget]
  | .f8e4, .f8e4 => klOpsBackward%[tl.float8e4nv, tl.float8e4nv, defaultTarget]
  | .f8e4, .f8e5 => klOpsBackward%[tl.float8e4nv, tl.float8e5, defaultTarget]
  | .f8e5, .fp32 => klOpsBackward%[tl.float8e5, tl.float32, defaultTarget]
  | .f8e5, .fp64 => klOpsBackward%[tl.float8e5, tl.float64, defaultTarget]
  | .f8e5, .fp16 => klOpsBackward%[tl.float8e5, tl.float16, defaultTarget]
  | .f8e5, .bf16 => klOpsBackward%[tl.float8e5, tl.bfloat16, defaultTarget]
  | .f8e5, .f8e4 => klOpsBackward%[tl.float8e5, tl.float8e4nv, defaultTarget]
  | .f8e5, .f8e5 => klOpsBackward%[tl.float8e5, tl.float8e5, defaultTarget]

/-- Complete backward log branch, retaining T then O pointer updates and the pre-loop mask. -/
def backwardLogOriginal (target : MathStorage) (output : Storage) : Backward :=
  match target, output with
  | .fp32, .fp32 => klOpsBackward%[tl.float32, tl.float32, logarithmic]
  | .fp32, .fp64 => klOpsBackward%[tl.float32, tl.float64, logarithmic]
  | .fp32, .fp16 => klOpsBackward%[tl.float32, tl.float16, logarithmic]
  | .fp32, .bf16 => klOpsBackward%[tl.float32, tl.bfloat16, logarithmic]
  | .fp32, .f8e4 => klOpsBackward%[tl.float32, tl.float8e4nv, logarithmic]
  | .fp32, .f8e5 => klOpsBackward%[tl.float32, tl.float8e5, logarithmic]
  | .fp64, .fp32 => klOpsBackward%[tl.float64, tl.float32, logarithmic]
  | .fp64, .fp64 => klOpsBackward%[tl.float64, tl.float64, logarithmic]
  | .fp64, .fp16 => klOpsBackward%[tl.float64, tl.float16, logarithmic]
  | .fp64, .bf16 => klOpsBackward%[tl.float64, tl.bfloat16, logarithmic]
  | .fp64, .f8e4 => klOpsBackward%[tl.float64, tl.float8e4nv, logarithmic]
  | .fp64, .f8e5 => klOpsBackward%[tl.float64, tl.float8e5, logarithmic]

/-- Candidate commutes only the exact signed output-row product. -/
def commuteOutput (source : ComputeKernel) (position : Nat) (O : RegionName)
    (stride : Nat) : ComputeKernel :=
  ComputeKernel.mk source.inputs source.outputs (source.surfaceBody.take position ++
    [.assign .ptr [] "O" (.alg (.ptrAdd .nil (.ptrBase O)
      (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat stride))
        (.ref .int [] "pid")))))] ++ source.surfaceBody.drop (position + 1))

def forwardDefault (pred : Storage) (target : Storage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) : ComputeKernel :=
  let source := forwardDefaultOriginal pred target output Y T O sy st so N B eps reduction
  if v then commuteOutput source 3 O so else source

def backwardDefault (target : Storage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  let source := backwardDefaultOriginal target output T O st so N B
  if v then commuteOutput source 2 O so else source

def forwardLog (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (eps : ℝ) (reduction : Nat) : ComputeKernel :=
  let source := forwardLogOriginal pred target output Y T O sy st so N B eps reduction
  if v then commuteOutput source 3 O so else source

def backwardLog (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) : ComputeKernel :=
  let source := backwardLogOriginal target output T O st so N B
  if v then commuteOutput source 2 O so else source

end VeriTile.Bench.TritonBenchG.KldivOps.TypedSource
