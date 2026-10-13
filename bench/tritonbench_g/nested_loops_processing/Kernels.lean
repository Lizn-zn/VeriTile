import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Full nested3 source, retaining all three loops, cached tiles, and pointer updates.
The candidate normalizes the initial input column multiplication. Signed strides
use the mathematical Int address channel; no fixed-width overflow claim is made. -/
namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedSource
open Lean VeriTile.Triton

syntax "nestedCopy%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(nestedCopy%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let ptr : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| a_ptrs = in_ptr + (offs_am[:, None] * $((stride_m : Int)) +
        offs_an[None, :] * $((stride_n : Int))))
      else
      `(tritonStmt| a_ptrs = in_ptr + (offs_am[:, None] * $((stride_m : Int)) +
        $((stride_n : Int)) * offs_an[None, :]))
    `(fun (in_ptr out_ptr : RegionName) (stride_m stride_n : Int) =>
      ComputeKernel.mk [in_ptr] [out_ptr] (triton {
  tl.region in_ptr = $storage, out_ptr = $storage
  offs_am = tl.arange(0, 2)
  offs_an = tl.arange(0, 2)
  $ptr
  offs_cm = tl.arange(0, 2)
  offs_cn = tl.arange(0, 2)
  c_ptrs = out_ptr + $((stride_m : Int)) * offs_cm[:, None] +
    $((stride_n : Int)) * offs_cn[None, :]
  for i in range(0, $(2), $(1)) {
    a1 = tl.load(a_ptrs)
    for j in range(0, $(2), $(1)) {
      a_ptrs += $(2) * $((stride_n : Int))
      a2 = tl.load(a_ptrs)
      for k in range(0, $(2), $(1)) {
        a_ptrs += $(2) * $((stride_n : Int))
        a3 = tl.load(a_ptrs)
        tl.store(c_ptrs, a1)
        c_ptrs += $(2) * $((stride_n : Int))
        tl.store(c_ptrs, a2)
        c_ptrs += $(2) * $((stride_n : Int))
        tl.store(c_ptrs, a3)
        c_ptrs += $(2) * $((stride_n : Int))
      }
    }
    a_ptrs += $(2) * $((stride_n : Int))
  }
}).surfaceBody)

inductive Storage where
  | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | boolean
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

abbrev Source := RegionName → RegionName → Int → Int → ComputeKernel

def kernel (storage : Storage) (candidate : Bool) : Source :=
  match storage, candidate with
  | .int8, Bool.false => nestedCopy%[tl.int8, original]
  | .int8, Bool.true => nestedCopy%[tl.int8, commuted]
  | .int16, Bool.false => nestedCopy%[tl.int16, original]
  | .int16, Bool.true => nestedCopy%[tl.int16, commuted]
  | .int32, Bool.false => nestedCopy%[tl.int32, original]
  | .int32, Bool.true => nestedCopy%[tl.int32, commuted]
  | .int64, Bool.false => nestedCopy%[tl.int64, original]
  | .int64, Bool.true => nestedCopy%[tl.int64, commuted]
  | .uint8, Bool.false => nestedCopy%[tl.uint8, original]
  | .uint8, Bool.true => nestedCopy%[tl.uint8, commuted]
  | .uint16, Bool.false => nestedCopy%[tl.uint16, original]
  | .uint16, Bool.true => nestedCopy%[tl.uint16, commuted]
  | .uint32, Bool.false => nestedCopy%[tl.uint32, original]
  | .uint32, Bool.true => nestedCopy%[tl.uint32, commuted]
  | .uint64, Bool.false => nestedCopy%[tl.uint64, original]
  | .uint64, Bool.true => nestedCopy%[tl.uint64, commuted]
  | .boolean, Bool.false => nestedCopy%[tl.int1, original]
  | .boolean, Bool.true => nestedCopy%[tl.int1, commuted]
  | .fp32, Bool.false => nestedCopy%[tl.float32, original]
  | .fp32, Bool.true => nestedCopy%[tl.float32, commuted]
  | .fp16, Bool.false => nestedCopy%[tl.float16, original]
  | .fp16, Bool.true => nestedCopy%[tl.float16, commuted]
  | .bf16, Bool.false => nestedCopy%[tl.bfloat16, original]
  | .bf16, Bool.true => nestedCopy%[tl.bfloat16, commuted]
  | .fp64, Bool.false => nestedCopy%[tl.float64, original]
  | .fp64, Bool.true => nestedCopy%[tl.float64, commuted]
  | .f8e4, Bool.false => nestedCopy%[tl.float8e4nv, original]
  | .f8e4, Bool.true => nestedCopy%[tl.float8e4nv, commuted]
  | .f8e5, Bool.false => nestedCopy%[tl.float8e5, original]
  | .f8e5, Bool.true => nestedCopy%[tl.float8e5, commuted]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.TypedSource

#guard_msgs (drop info) in
#auditModuleAxioms
