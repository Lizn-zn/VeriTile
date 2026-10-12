import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: destindex_copy.py, _fwd_kernel_destindex_copy_kv

Complete `_fwd_kernel_destindex_copy_kv` source. The two payload storage
families are independent; each output has the corresponding input dtype.
Destination metadata uses signed integers. All loads/stores are unmasked,
including power-of-two padding lanes. Head counts and head strides are unused
in the source and retained in the signature: only the first head is copied.

The candidate commutes only the natural-number product in the nope source-row
address. Numeric values, signed destination arithmetic and both stores remain
unchanged. Integer widths are abstracted to mathematical Int/Nat; this is not
a fixed-width overflow or host-allocation/launch proof. -/
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.TypedSource
open Lean VeriTile.Triton
syntax "destindexCopySource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(destindexCopySource%[$nopeType:tritonDType, $ropeType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let sourcePointer : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| kv_nope_ptrs = KV_nope + $(stride_kv_nope_bs) * cur_index + $(stride_kv_nope_d) * offs_d_nope[None, :])
      else `(tritonStmt| kv_nope_ptrs = KV_nope + cur_index * $(stride_kv_nope_bs) + $(stride_kv_nope_d) * offs_d_nope[None, :])
    `(fun (KV_nope KV_rope : RegionName) (Dest_loc : Region .int) (O_nope O_rope : RegionName)
    (stride_kv_nope_bs stride_kv_nope_h stride_kv_nope_d
      stride_kv_rope_bs stride_kv_rope_h stride_kv_rope_d
      stride_o_nope_bs stride_o_nope_h stride_o_nope_d
      stride_o_rope_bs stride_o_rope_h stride_o_rope_d
      kv_nope_head_num kv_rope_head_num
      BLOCK_DMODEL_NOPE BLOCK_DMODEL_ROPE : Nat) =>
      ComputeKernel.mk [KV_nope, KV_rope, Dest_loc.cast] [O_nope, O_rope] (triton {
  tl.region KV_nope = $nopeType, KV_rope = $ropeType, O_nope = $nopeType, O_rope = $ropeType
  cur_index = tl.program_id(0)
  offs_d_nope = tl.arange(0, $(BLOCK_DMODEL_NOPE))
  offs_d_rope = tl.arange(0, $(BLOCK_DMODEL_ROPE))
  dest_index = tl.load(Dest_loc + cur_index)

  $sourcePointer
  kv_rope_ptrs = KV_rope + cur_index * $(stride_kv_rope_bs) +
    $(stride_kv_rope_d) * offs_d_rope[None, :]

  o_nope_ptrs = O_nope + dest_index * $(stride_o_nope_bs) +
    $(stride_o_nope_d) * offs_d_nope[None, :]
  o_rope_ptrs = O_rope + dest_index * $(stride_o_rope_bs) +
    $(stride_o_rope_d) * offs_d_rope[None, :]

  kv_nope = tl.load(kv_nope_ptrs)
  kv_rope = tl.load(kv_rope_ptrs)

  tl.store(o_nope_ptrs, kv_nope)
  tl.store(o_rope_ptrs, kv_rope)
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

abbrev Source := RegionName → RegionName → Region .int → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

def kernel (nopeType ropeType : Storage) (swapped : Bool) : Source :=
  match nopeType, ropeType, swapped with
  | .fp32, .fp32, Bool.false => destindexCopySource%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => destindexCopySource%[tl.float32, tl.float32, indexCommuted]
  | .fp32, .fp16, Bool.false => destindexCopySource%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => destindexCopySource%[tl.float32, tl.float16, indexCommuted]
  | .fp32, .bf16, Bool.false => destindexCopySource%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => destindexCopySource%[tl.float32, tl.bfloat16, indexCommuted]
  | .fp16, .fp32, Bool.false => destindexCopySource%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => destindexCopySource%[tl.float16, tl.float32, indexCommuted]
  | .fp16, .fp16, Bool.false => destindexCopySource%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => destindexCopySource%[tl.float16, tl.float16, indexCommuted]
  | .fp16, .bf16, Bool.false => destindexCopySource%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => destindexCopySource%[tl.float16, tl.bfloat16, indexCommuted]
  | .bf16, .fp32, Bool.false => destindexCopySource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => destindexCopySource%[tl.bfloat16, tl.float32, indexCommuted]
  | .bf16, .fp16, Bool.false => destindexCopySource%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => destindexCopySource%[tl.bfloat16, tl.float16, indexCommuted]
  | .bf16, .bf16, Bool.false => destindexCopySource%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => destindexCopySource%[tl.bfloat16, tl.bfloat16, indexCommuted]

end VeriTile.Bench.TritonBenchG.DestindexCopy.TypedSource
