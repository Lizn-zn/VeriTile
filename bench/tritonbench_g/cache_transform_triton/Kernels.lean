import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Complete typed sources of `cache_transform_triton.py`'s
`prefill_cache_kernel` and `decoding_cache_kernel`. The
candidate commutes the scalar natural-number product in token indexing; all
floating payloads, signed metadata, masks and stores remain unchanged.

Translation-surface blocker: region dtype annotations are syntax-level, so
mixed storage pairs are specialized by a macro. Explicit ComputeKernel
signatures retain both caches and signed metadata while the DSL elaborates
the complete typed bodies. No statement or kernel parameter is dropped.

Cosine and sine storage are independent (fp16/bf16/fp32); each output
inherits its corresponding cache dtype, as in the host wrapper.
Integer widths are abstracted to Int/Nat; this is not an overflow proof.
Prefill keeps the unmasked metadata load, including its allocation obligation.
Decode keeps the complete BLOCK_SIZE × HIDDEN_DIM tile and omitted `other`.
The host subtracts one from lengths before decode; the kernel consumes those
already adjusted signed row indices. Cache strides are also used for outputs,
as in the original source. No padding or contiguous-output repair is inserted.
-/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.TypedSource
open Lean VeriTile.Triton

syntax "cachePrefill%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(cachePrefill%[$cosTy:tritonDType, $sinTy:tritonDType, $variant:ident]) => do
    unless variant.getId == `original || variant.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let tokenIndex : TSyntax `tritonExpr ←
      if variant.getId == `commuted then `(tritonExpr| $(BLOCK_SIZE) * idx0 + idx1)
      else `(tritonExpr| idx0 * $(BLOCK_SIZE) + idx1)
    `(fun (cos_cache sin_cache : RegionName) (cumsum_lengths : Region .int)
        (cos_output sin_output : RegionName)
        (cache_stride hidden_stride total_length HIDDEN_DIM N_ELEMENTS BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [cos_cache, sin_cache, Region.cast cumsum_lengths] [cos_output, sin_output]
      (triton {
  tl.region cos_cache = $cosTy, sin_cache = $sinTy, cos_output = $cosTy, sin_output = $sinTy
  idx0 = tl.program_id(axis=0)
  idx1 = tl.program_id(axis=1)
  idx = $tokenIndex
  cumsum_lens = tl.load(cumsum_lengths + tl.arange(0, $(N_ELEMENTS)))
  ori_seq_idx = idx - tl.max(tl.where(cumsum_lens <= idx, cumsum_lens, $(0)))
  cos_cache_part = tl.load(
    cos_cache + ori_seq_idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    mask=idx < $(total_length))
  sin_cache_part = tl.load(
    sin_cache + ori_seq_idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    mask=idx < $(total_length))
  tl.store(
    cos_output + idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    cos_cache_part,
    mask=idx < $(total_length))
  tl.store(
    sin_output + idx * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM)) * $(hidden_stride),
    sin_cache_part,
    mask=idx < $(total_length))
}).surfaceBody)

syntax "cacheDecode%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(cacheDecode%[$cosTy:tritonDType, $sinTy:tritonDType, $variant:ident]) => do
    unless variant.getId == `original || variant.getId == `commuted do
      Macro.throwError "expected original or commuted"
    let tokenIndex : TSyntax `tritonExpr ←
      if variant.getId == `commuted then
        `(tritonExpr| $(BLOCK_SIZE) * tl.program_id(0) + tl.arange(0, $(BLOCK_SIZE)))
      else `(tritonExpr| tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE)))
    `(fun (cos_cache sin_cache : RegionName) (lengths : Region .int)
        (cos_output sin_output : RegionName)
        (cache_stride hidden_stride HIDDEN_DIM NUM_SEQS BLOCK_SIZE : Nat) =>
      ComputeKernel.mk [cos_cache, sin_cache, Region.cast lengths] [cos_output, sin_output]
      (triton {
  tl.region cos_cache = $cosTy, sin_cache = $sinTy, cos_output = $cosTy, sin_output = $sinTy
  idx = $tokenIndex
  ori_seq_idx = tl.load(lengths + idx,
    mask=idx < $(NUM_SEQS), other=None)
  cos_cache_part = tl.load(cos_cache + ori_seq_idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    mask=idx[:, None] < $(NUM_SEQS))
  sin_cache_part = tl.load(sin_cache + ori_seq_idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    mask=idx[:, None] < $(NUM_SEQS))
  tl.store(cos_output + idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    cos_cache_part, mask=idx[:, None] < $(NUM_SEQS))
  tl.store(sin_output + idx[:, None] * $(cache_stride) +
      tl.arange(0, $(HIDDEN_DIM))[None, :] * $(hidden_stride),
    sin_cache_part, mask=idx[:, None] < $(NUM_SEQS))
}).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

abbrev PrefillSource := RegionName → RegionName → Region .int → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel
abbrev DecodeSource := RegionName → RegionName → Region .int → RegionName → RegionName →
  Nat → Nat → Nat → Nat → Nat → ComputeKernel

def prefill (cosType sinType : Storage) (commuted : Bool) : PrefillSource :=
  match cosType, sinType, commuted with
  | .fp32, .fp32, Bool.false => cachePrefill%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => cachePrefill%[tl.float32, tl.float32, commuted]
  | .fp32, .fp16, Bool.false => cachePrefill%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => cachePrefill%[tl.float32, tl.float16, commuted]
  | .fp32, .bf16, Bool.false => cachePrefill%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => cachePrefill%[tl.float32, tl.bfloat16, commuted]
  | .fp16, .fp32, Bool.false => cachePrefill%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => cachePrefill%[tl.float16, tl.float32, commuted]
  | .fp16, .fp16, Bool.false => cachePrefill%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => cachePrefill%[tl.float16, tl.float16, commuted]
  | .fp16, .bf16, Bool.false => cachePrefill%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => cachePrefill%[tl.float16, tl.bfloat16, commuted]
  | .bf16, .fp32, Bool.false => cachePrefill%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => cachePrefill%[tl.bfloat16, tl.float32, commuted]
  | .bf16, .fp16, Bool.false => cachePrefill%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => cachePrefill%[tl.bfloat16, tl.float16, commuted]
  | .bf16, .bf16, Bool.false => cachePrefill%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => cachePrefill%[tl.bfloat16, tl.bfloat16, commuted]

def decode (cosType sinType : Storage) (commuted : Bool) : DecodeSource :=
  match cosType, sinType, commuted with
  | .fp32, .fp32, Bool.false => cacheDecode%[tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => cacheDecode%[tl.float32, tl.float32, commuted]
  | .fp32, .fp16, Bool.false => cacheDecode%[tl.float32, tl.float16, original]
  | .fp32, .fp16, Bool.true => cacheDecode%[tl.float32, tl.float16, commuted]
  | .fp32, .bf16, Bool.false => cacheDecode%[tl.float32, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => cacheDecode%[tl.float32, tl.bfloat16, commuted]
  | .fp16, .fp32, Bool.false => cacheDecode%[tl.float16, tl.float32, original]
  | .fp16, .fp32, Bool.true => cacheDecode%[tl.float16, tl.float32, commuted]
  | .fp16, .fp16, Bool.false => cacheDecode%[tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => cacheDecode%[tl.float16, tl.float16, commuted]
  | .fp16, .bf16, Bool.false => cacheDecode%[tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => cacheDecode%[tl.float16, tl.bfloat16, commuted]
  | .bf16, .fp32, Bool.false => cacheDecode%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp32, Bool.true => cacheDecode%[tl.bfloat16, tl.float32, commuted]
  | .bf16, .fp16, Bool.false => cacheDecode%[tl.bfloat16, tl.float16, original]
  | .bf16, .fp16, Bool.true => cacheDecode%[tl.bfloat16, tl.float16, commuted]
  | .bf16, .bf16, Bool.false => cacheDecode%[tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => cacheDecode%[tl.bfloat16, tl.bfloat16, commuted]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.TypedSource
