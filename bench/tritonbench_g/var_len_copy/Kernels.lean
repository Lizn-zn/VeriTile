import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Complete variable-length copy source. Signed metadata uses mathematical Int;
int32/int64 widths do not impose machine-overflow semantics here. The original
mask is lane < length, so partial final chunks of long segments are overcopied.
The candidate only commutes the signed start-plus-loop-index address sum. -/
namespace VeriTile.Bench.TritonBenchG.VarLenCopy.TypedSource
open Lean VeriTile.Triton

syntax "varLenCopy%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(varLenCopy%[$input:tritonDType, $output:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let load : TSyntax `tritonStmt ← if version.getId == `original then
      `(tritonStmt| v = tl.load(src + old_start + i + old_offset, mask=old_offset < length))
      else `(tritonStmt| v = tl.load(src + i + old_start + old_offset, mask=old_offset < length))
    let store : TSyntax `tritonStmt ←
      if input.raw.reprint == output.raw.reprint then
        `(tritonStmt| tl.store(dst + new_start + i + new_offset, v, mask=new_offset < length))
      else `(tritonStmt| tl.store(dst + new_start + i + new_offset, v.to($output), mask=new_offset < length))
    `(fun (os ol ns : Region .int) (src dst : RegionName) (B : Nat) =>
      ComputeKernel.mk [ol.cast, os.cast, ns.cast, src] [dst] (triton {
  tl.region src = $input, dst = $output
  a_id = tl.program_id(0)
  length = tl.load(ol + a_id)
  old_start = tl.load(os + a_id)
  new_start = tl.load(ns + a_id)
  old_offset = tl.arange(0, $(B))
  new_offset = tl.arange(0, $(B))
  for i in range($(0), length, $(B)) {
    $load
    $store
  }
}).surfaceBody)

/-- Floating source and destination storage vary independently. -/
inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq, Repr

/-- Both supported signed metadata widths retain the same mathematical Int channel. -/
inductive Metadata where
  | int32 | int64
  deriving DecidableEq, Repr

abbrev Source := Region .int → Region .int → Region .int → RegionName → RegionName → Nat → ComputeKernel

def kernel (input output : Storage) (v : Bool) : Source :=
  match input, output, v with
  | .fp32, .fp32, Bool.false => varLenCopy%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => varLenCopy%[tl.float32, tl.float32, commuted]
  | .fp32, .fp16, Bool.false => varLenCopy%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => varLenCopy%[tl.float32, tl.float16, commuted]
  | .fp32, .bf16, Bool.false => varLenCopy%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => varLenCopy%[tl.float32, tl.bfloat16, commuted]
  | .fp32, .fp64, Bool.false => varLenCopy%[tl.float32, tl.float64, original]
  | .fp32, .fp64, Bool.true => varLenCopy%[tl.float32, tl.float64, commuted]
  | .fp32, .f8e4, Bool.false => varLenCopy%[tl.float32, tl.float8e4nv, original]
  | .fp32, .f8e4, Bool.true => varLenCopy%[tl.float32, tl.float8e4nv, commuted]
  | .fp32, .f8e5, Bool.false => varLenCopy%[tl.float32, tl.float8e5, original]
  | .fp32, .f8e5, Bool.true => varLenCopy%[tl.float32, tl.float8e5, commuted]
  | .fp16, .fp32, Bool.false => varLenCopy%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => varLenCopy%[tl.float16, tl.float32, commuted]
  | .fp16, .fp16, Bool.false => varLenCopy%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => varLenCopy%[tl.float16, tl.float16, commuted]
  | .fp16, .bf16, Bool.false => varLenCopy%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => varLenCopy%[tl.float16, tl.bfloat16, commuted]
  | .fp16, .fp64, Bool.false => varLenCopy%[tl.float16, tl.float64, original]
  | .fp16, .fp64, Bool.true => varLenCopy%[tl.float16, tl.float64, commuted]
  | .fp16, .f8e4, Bool.false => varLenCopy%[tl.float16, tl.float8e4nv, original]
  | .fp16, .f8e4, Bool.true => varLenCopy%[tl.float16, tl.float8e4nv, commuted]
  | .fp16, .f8e5, Bool.false => varLenCopy%[tl.float16, tl.float8e5, original]
  | .fp16, .f8e5, Bool.true => varLenCopy%[tl.float16, tl.float8e5, commuted]
  | .bf16, .fp32, Bool.false => varLenCopy%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => varLenCopy%[tl.bfloat16, tl.float32, commuted]
  | .bf16, .fp16, Bool.false => varLenCopy%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => varLenCopy%[tl.bfloat16, tl.float16, commuted]
  | .bf16, .bf16, Bool.false => varLenCopy%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => varLenCopy%[tl.bfloat16, tl.bfloat16, commuted]
  | .bf16, .fp64, Bool.false => varLenCopy%[tl.bfloat16, tl.float64, original]
  | .bf16, .fp64, Bool.true => varLenCopy%[tl.bfloat16, tl.float64, commuted]
  | .bf16, .f8e4, Bool.false => varLenCopy%[tl.bfloat16, tl.float8e4nv, original]
  | .bf16, .f8e4, Bool.true => varLenCopy%[tl.bfloat16, tl.float8e4nv, commuted]
  | .bf16, .f8e5, Bool.false => varLenCopy%[tl.bfloat16, tl.float8e5, original]
  | .bf16, .f8e5, Bool.true => varLenCopy%[tl.bfloat16, tl.float8e5, commuted]
  | .fp64, .fp32, Bool.false => varLenCopy%[tl.float64, tl.float32, original]
  | .fp64, .fp32, Bool.true => varLenCopy%[tl.float64, tl.float32, commuted]
  | .fp64, .fp16, Bool.false => varLenCopy%[tl.float64, tl.float16, original]
  | .fp64, .fp16, Bool.true => varLenCopy%[tl.float64, tl.float16, commuted]
  | .fp64, .bf16, Bool.false => varLenCopy%[tl.float64, tl.bfloat16, original]
  | .fp64, .bf16, Bool.true => varLenCopy%[tl.float64, tl.bfloat16, commuted]
  | .fp64, .fp64, Bool.false => varLenCopy%[tl.float64, tl.float64, original]
  | .fp64, .fp64, Bool.true => varLenCopy%[tl.float64, tl.float64, commuted]
  | .fp64, .f8e4, Bool.false => varLenCopy%[tl.float64, tl.float8e4nv, original]
  | .fp64, .f8e4, Bool.true => varLenCopy%[tl.float64, tl.float8e4nv, commuted]
  | .fp64, .f8e5, Bool.false => varLenCopy%[tl.float64, tl.float8e5, original]
  | .fp64, .f8e5, Bool.true => varLenCopy%[tl.float64, tl.float8e5, commuted]
  | .f8e4, .fp32, Bool.false => varLenCopy%[tl.float8e4nv, tl.float32, original]
  | .f8e4, .fp32, Bool.true => varLenCopy%[tl.float8e4nv, tl.float32, commuted]
  | .f8e4, .fp16, Bool.false => varLenCopy%[tl.float8e4nv, tl.float16, original]
  | .f8e4, .fp16, Bool.true => varLenCopy%[tl.float8e4nv, tl.float16, commuted]
  | .f8e4, .bf16, Bool.false => varLenCopy%[tl.float8e4nv, tl.bfloat16, original]
  | .f8e4, .bf16, Bool.true => varLenCopy%[tl.float8e4nv, tl.bfloat16, commuted]
  | .f8e4, .fp64, Bool.false => varLenCopy%[tl.float8e4nv, tl.float64, original]
  | .f8e4, .fp64, Bool.true => varLenCopy%[tl.float8e4nv, tl.float64, commuted]
  | .f8e4, .f8e4, Bool.false => varLenCopy%[tl.float8e4nv, tl.float8e4nv, original]
  | .f8e4, .f8e4, Bool.true => varLenCopy%[tl.float8e4nv, tl.float8e4nv, commuted]
  | .f8e4, .f8e5, Bool.false => varLenCopy%[tl.float8e4nv, tl.float8e5, original]
  | .f8e4, .f8e5, Bool.true => varLenCopy%[tl.float8e4nv, tl.float8e5, commuted]
  | .f8e5, .fp32, Bool.false => varLenCopy%[tl.float8e5, tl.float32, original]
  | .f8e5, .fp32, Bool.true => varLenCopy%[tl.float8e5, tl.float32, commuted]
  | .f8e5, .fp16, Bool.false => varLenCopy%[tl.float8e5, tl.float16, original]
  | .f8e5, .fp16, Bool.true => varLenCopy%[tl.float8e5, tl.float16, commuted]
  | .f8e5, .bf16, Bool.false => varLenCopy%[tl.float8e5, tl.bfloat16, original]
  | .f8e5, .bf16, Bool.true => varLenCopy%[tl.float8e5, tl.bfloat16, commuted]
  | .f8e5, .fp64, Bool.false => varLenCopy%[tl.float8e5, tl.float64, original]
  | .f8e5, .fp64, Bool.true => varLenCopy%[tl.float8e5, tl.float64, commuted]
  | .f8e5, .f8e4, Bool.false => varLenCopy%[tl.float8e5, tl.float8e4nv, original]
  | .f8e5, .f8e4, Bool.true => varLenCopy%[tl.float8e5, tl.float8e4nv, commuted]
  | .f8e5, .f8e5, Bool.false => varLenCopy%[tl.float8e5, tl.float8e5, original]
  | .f8e5, .f8e5, Bool.true => varLenCopy%[tl.float8e5, tl.float8e5, commuted]

/-- Metadata buffers can independently use either signed width. Width abstraction
is explicit; the source has no cast of its loaded signed starts or length. -/
def typedKernel (input output : Storage) (_oldStart _length _newStart : Metadata)
    (v : Bool) : Source := kernel input output v

end VeriTile.Bench.TritonBenchG.VarLenCopy.TypedSource

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteSource
open VeriTile Triton

/-- Width labels distinguish source spellings; Int/Nat retain mathematical carriers. -/
inductive Storage where
  | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | bool
  deriving DecidableEq, Repr

inductive Kind where
  | signed | unsigned | boolean
  deriving DecidableEq, Repr

def Kind.dtype : Kind → TileDType
  | .signed => .int
  | .unsigned => .nat
  | .boolean => .bool

def Storage.kind : Storage → Kind
  | .int8 | .int16 | .int32 | .int64 => .signed
  | .uint8 | .uint16 | .uint32 | .uint64 => .unsigned
  | .bool => .boolean

/-- Homogeneous raw copies have no payload conversion or arithmetic. -/
def kernel (storage : Storage) (v : Bool) : TypedSource.Source :=
  match storage, v with
  | .int8, Bool.false => varLenCopy%[tl.int8, tl.int8, original]
  | .int8, Bool.true => varLenCopy%[tl.int8, tl.int8, commuted]
  | .int16, Bool.false => varLenCopy%[tl.int16, tl.int16, original]
  | .int16, Bool.true => varLenCopy%[tl.int16, tl.int16, commuted]
  | .int32, Bool.false => varLenCopy%[tl.int32, tl.int32, original]
  | .int32, Bool.true => varLenCopy%[tl.int32, tl.int32, commuted]
  | .int64, Bool.false => varLenCopy%[tl.int64, tl.int64, original]
  | .int64, Bool.true => varLenCopy%[tl.int64, tl.int64, commuted]
  | .uint8, Bool.false => varLenCopy%[tl.uint8, tl.uint8, original]
  | .uint8, Bool.true => varLenCopy%[tl.uint8, tl.uint8, commuted]
  | .uint16, Bool.false => varLenCopy%[tl.uint16, tl.uint16, original]
  | .uint16, Bool.true => varLenCopy%[tl.uint16, tl.uint16, commuted]
  | .uint32, Bool.false => varLenCopy%[tl.uint32, tl.uint32, original]
  | .uint32, Bool.true => varLenCopy%[tl.uint32, tl.uint32, commuted]
  | .uint64, Bool.false => varLenCopy%[tl.uint64, tl.uint64, original]
  | .uint64, Bool.true => varLenCopy%[tl.uint64, tl.uint64, commuted]
  | .bool, Bool.false => varLenCopy%[tl.int1, tl.int1, original]
  | .bool, Bool.true => varLenCopy%[tl.int1, tl.int1, commuted]

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteSource

#guard_msgs (drop info) in
#auditModuleAxioms
