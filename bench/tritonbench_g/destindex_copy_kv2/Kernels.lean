import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: destindex_copy_kv2.py, _fwd_kernel_destindex_copy_kv

Complete head-masked multi-head destination-indexed copy. Metadata uses signed
integers; input and output independently support fp16/bf16/fp32. The explicit
output cast records the Python store's implicit destination conversion.
The original scalar metadata load is unmasked. Only the head dimension is masked; every column is accessed for an active
head. The input load retains other=0.0. The host passes head_dim directly
as the arange width; no next-power-of-two repair is introduced.
The candidate commutes only the exact natural-number source-row product.
Integer widths, host launch composition and races are outside the model. -/
namespace VeriTile.Bench.TritonBenchG.DestindexCopyKv2.TypedSource
open Lean VeriTile.Triton
syntax "destindexCopyKv2Source%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(destindexCopyKv2Source%[$inType:tritonDType, $outType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let sourcePointer : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| k_ptrs = K + $(stride_k_bs) * cur_index + $(stride_k_h) * offs_h[:, None] + $(stride_k_d) * offs_d[None, :])
      else
        `(tritonStmt| k_ptrs = K + cur_index * $(stride_k_bs) + $(stride_k_h) * offs_h[:, None] + $(stride_k_d) * offs_d[None, :])
    `(fun (K : RegionName) (Dest_loc : Region .int) (Out : RegionName)
    (stride_k_bs stride_k_h stride_k_d stride_o_bs stride_o_h stride_o_d
      head_num BLOCK_DMODEL BLOCK_HEAD : Nat) =>
      ComputeKernel.mk [K,Dest_loc.cast] [Out] (triton {
  tl.region K = $inType, Out = $outType
  cur_index = tl.program_id(0)
  offs_h = tl.arange(0, $(BLOCK_HEAD))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  dest_index = tl.load(Dest_loc + cur_index)
  $sourcePointer
  o_ptrs = Out + dest_index * $(stride_o_bs) +
    $(stride_o_h) * offs_h[:, None] + $(stride_o_d) * offs_d[None, :]
  k = tl.load(k_ptrs, mask=offs_h[:, None] < $(head_num), other=0.0)
  tl.store(o_ptrs, (k).to($outType:tritonDType), mask=offs_h[:, None] < $(head_num))
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

abbrev Source := RegionName → Region .int → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

def kernel (input output : Storage) (swapped : Bool) : Source :=
  match input, output, swapped with
  | .fp32, .fp32, Bool.false => destindexCopyKv2Source%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => destindexCopyKv2Source%[tl.float32, tl.float32, indexCommuted]
  | .fp32, .fp16, Bool.false => destindexCopyKv2Source%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => destindexCopyKv2Source%[tl.float32, tl.float16, indexCommuted]
  | .fp32, .bf16, Bool.false => destindexCopyKv2Source%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => destindexCopyKv2Source%[tl.float32, tl.bfloat16, indexCommuted]
  | .fp16, .fp32, Bool.false => destindexCopyKv2Source%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => destindexCopyKv2Source%[tl.float16, tl.float32, indexCommuted]
  | .fp16, .fp16, Bool.false => destindexCopyKv2Source%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => destindexCopyKv2Source%[tl.float16, tl.float16, indexCommuted]
  | .fp16, .bf16, Bool.false => destindexCopyKv2Source%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => destindexCopyKv2Source%[tl.float16, tl.bfloat16, indexCommuted]
  | .bf16, .fp32, Bool.false => destindexCopyKv2Source%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => destindexCopyKv2Source%[tl.bfloat16, tl.float32, indexCommuted]
  | .bf16, .fp16, Bool.false => destindexCopyKv2Source%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => destindexCopyKv2Source%[tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, .bf16, Bool.false => destindexCopyKv2Source%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => destindexCopyKv2Source%[tl.bfloat16, tl.bfloat16, indexCommuted]

end VeriTile.Bench.TritonBenchG.DestindexCopyKv2.TypedSource
