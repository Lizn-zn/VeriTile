import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: fast_rope_embedding.py, _rope_embedding.
Complete four-head group loop, optional sine negation, two loaded Q halves and
two stores. sin1.dtype is statically the declared sine storage; negation leaves
that dtype unchanged. The candidate commutes only the natural-number head-group product; the
complete dynamic loop and both Q offset expressions are retained. -/
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.TypedSource
open Lean VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

syntax "fastRope%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastRope%[$inputTy:tritonDType,$cosTy:tritonDType,$sinTy:tritonDType,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let headStart : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| head_start = ROPE_GROUP_SIZE * group_head_position)
      else `(tritonStmt| head_start = group_head_position * ROPE_GROUP_SIZE)
    `(fun (Q cos sin : RegionName) (Q_row_stride cos_row_stride sin_row_stride seqlen head_dim n_heads BLOCK_SIZE : Nat) (BACKWARD_PASS : Bool) =>
      ComputeKernel.mk [Q,cos,sin] [Q] (triton {
  tl.region Q = $inputTy, cos = $cosTy, sin = $sinTy
  ROPE_GROUP_SIZE = $(4)
  row_position = tl.program_id(0)
  group_head_position = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  half_head_dim = $(head_dim / 2)
  mask = col_offsets < half_head_dim
  sin1 = tl.load(sin + (row_position % $(seqlen)) * $(sin_row_stride) + half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  cos1 = tl.load(cos + (row_position % $(seqlen)) * $(cos_row_stride) + half_head_dim * $(0) + col_offsets, mask=mask, other=0)
  if $((BACKWARD_PASS : Bool)) {
    sin1 = -sin1
  }
  $headStart
  head_end = min(head_start + ROPE_GROUP_SIZE, $(n_heads))
  for k in range(head_start, head_end, $(1)) {
    offs_q1 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets
    offs_q2 = row_position * $(Q_row_stride) + k * $(head_dim) + col_offsets + half_head_dim
    Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to($sinTy:tritonDType)
    Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to($sinTy:tritonDType)
    tl.store(Q + offs_q1, (Q1 * cos1 - Q2 * sin1).to($inputTy:tritonDType), mask=mask)
    tl.store(Q + offs_q2, (Q2 * cos1 + Q1 * sin1).to($inputTy:tritonDType), mask=mask)
  }
}).surfaceBody)

def rope (input cosine sine : Storage) (swapped : Bool) := match input,cosine,sine with
  | .fp16,.fp16,.fp16 => if swapped then fastRope%[tl.float16,tl.float16,tl.float16,indexCommuted] else fastRope%[tl.float16,tl.float16,tl.float16,original]
  | .fp16,.fp16,.bf16 => if swapped then fastRope%[tl.float16,tl.float16,tl.bfloat16,indexCommuted] else fastRope%[tl.float16,tl.float16,tl.bfloat16,original]
  | .fp16,.fp16,.fp32 => if swapped then fastRope%[tl.float16,tl.float16,tl.float32,indexCommuted] else fastRope%[tl.float16,tl.float16,tl.float32,original]
  | .fp16,.bf16,.fp16 => if swapped then fastRope%[tl.float16,tl.bfloat16,tl.float16,indexCommuted] else fastRope%[tl.float16,tl.bfloat16,tl.float16,original]
  | .fp16,.bf16,.bf16 => if swapped then fastRope%[tl.float16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastRope%[tl.float16,tl.bfloat16,tl.bfloat16,original]
  | .fp16,.bf16,.fp32 => if swapped then fastRope%[tl.float16,tl.bfloat16,tl.float32,indexCommuted] else fastRope%[tl.float16,tl.bfloat16,tl.float32,original]
  | .fp16,.fp32,.fp16 => if swapped then fastRope%[tl.float16,tl.float32,tl.float16,indexCommuted] else fastRope%[tl.float16,tl.float32,tl.float16,original]
  | .fp16,.fp32,.bf16 => if swapped then fastRope%[tl.float16,tl.float32,tl.bfloat16,indexCommuted] else fastRope%[tl.float16,tl.float32,tl.bfloat16,original]
  | .fp16,.fp32,.fp32 => if swapped then fastRope%[tl.float16,tl.float32,tl.float32,indexCommuted] else fastRope%[tl.float16,tl.float32,tl.float32,original]
  | .bf16,.fp16,.fp16 => if swapped then fastRope%[tl.bfloat16,tl.float16,tl.float16,indexCommuted] else fastRope%[tl.bfloat16,tl.float16,tl.float16,original]
  | .bf16,.fp16,.bf16 => if swapped then fastRope%[tl.bfloat16,tl.float16,tl.bfloat16,indexCommuted] else fastRope%[tl.bfloat16,tl.float16,tl.bfloat16,original]
  | .bf16,.fp16,.fp32 => if swapped then fastRope%[tl.bfloat16,tl.float16,tl.float32,indexCommuted] else fastRope%[tl.bfloat16,tl.float16,tl.float32,original]
  | .bf16,.bf16,.fp16 => if swapped then fastRope%[tl.bfloat16,tl.bfloat16,tl.float16,indexCommuted] else fastRope%[tl.bfloat16,tl.bfloat16,tl.float16,original]
  | .bf16,.bf16,.bf16 => if swapped then fastRope%[tl.bfloat16,tl.bfloat16,tl.bfloat16,indexCommuted] else fastRope%[tl.bfloat16,tl.bfloat16,tl.bfloat16,original]
  | .bf16,.bf16,.fp32 => if swapped then fastRope%[tl.bfloat16,tl.bfloat16,tl.float32,indexCommuted] else fastRope%[tl.bfloat16,tl.bfloat16,tl.float32,original]
  | .bf16,.fp32,.fp16 => if swapped then fastRope%[tl.bfloat16,tl.float32,tl.float16,indexCommuted] else fastRope%[tl.bfloat16,tl.float32,tl.float16,original]
  | .bf16,.fp32,.bf16 => if swapped then fastRope%[tl.bfloat16,tl.float32,tl.bfloat16,indexCommuted] else fastRope%[tl.bfloat16,tl.float32,tl.bfloat16,original]
  | .bf16,.fp32,.fp32 => if swapped then fastRope%[tl.bfloat16,tl.float32,tl.float32,indexCommuted] else fastRope%[tl.bfloat16,tl.float32,tl.float32,original]
  | .fp32,.fp16,.fp16 => if swapped then fastRope%[tl.float32,tl.float16,tl.float16,indexCommuted] else fastRope%[tl.float32,tl.float16,tl.float16,original]
  | .fp32,.fp16,.bf16 => if swapped then fastRope%[tl.float32,tl.float16,tl.bfloat16,indexCommuted] else fastRope%[tl.float32,tl.float16,tl.bfloat16,original]
  | .fp32,.fp16,.fp32 => if swapped then fastRope%[tl.float32,tl.float16,tl.float32,indexCommuted] else fastRope%[tl.float32,tl.float16,tl.float32,original]
  | .fp32,.bf16,.fp16 => if swapped then fastRope%[tl.float32,tl.bfloat16,tl.float16,indexCommuted] else fastRope%[tl.float32,tl.bfloat16,tl.float16,original]
  | .fp32,.bf16,.bf16 => if swapped then fastRope%[tl.float32,tl.bfloat16,tl.bfloat16,indexCommuted] else fastRope%[tl.float32,tl.bfloat16,tl.bfloat16,original]
  | .fp32,.bf16,.fp32 => if swapped then fastRope%[tl.float32,tl.bfloat16,tl.float32,indexCommuted] else fastRope%[tl.float32,tl.bfloat16,tl.float32,original]
  | .fp32,.fp32,.fp16 => if swapped then fastRope%[tl.float32,tl.float32,tl.float16,indexCommuted] else fastRope%[tl.float32,tl.float32,tl.float16,original]
  | .fp32,.fp32,.bf16 => if swapped then fastRope%[tl.float32,tl.float32,tl.bfloat16,indexCommuted] else fastRope%[tl.float32,tl.float32,tl.bfloat16,original]
  | .fp32,.fp32,.fp32 => if swapped then fastRope%[tl.float32,tl.float32,tl.float32,indexCommuted] else fastRope%[tl.float32,tl.float32,tl.float32,original]

end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.TypedSource
