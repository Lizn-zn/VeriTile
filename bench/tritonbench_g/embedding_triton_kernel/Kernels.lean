import VeriTile.Triton.DSL
import VeriTile.Triton.Source
/-! Source: embedding_triton_kernel.py, embedding_kernel.
Full chunked embedding lookup. Signed int32 token IDs are retained; IDs
outside the local vocabulary produce zero outputs. Data/output precision
matches at fp16, bf16 or fp32. The candidate only commutes the integer product
forming start_n. Loop, load/store masks and explicit padding are unchanged. -/
namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.TypedSource
open Lean VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

syntax "embeddingSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(embeddingSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let start : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| start_n = $(BLOCK_N) * tl.program_id(0))
      else `(tritonStmt| start_n = tl.program_id(0) * $(BLOCK_N))
    `(fun (weight input_ids out : RegionName) (vob_start_id vob_end_id : Int)
        (stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat) =>
      ComputeKernel.mk [weight,input_ids] [out] (triton {
  tl.region weight = $storage, input_ids = tl.int32, out = $storage
  $start
  offs_nn = start_n + tl.arange(0, $(BLOCK_NN))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  for start_nn in range(0, $(BLOCK_N), $(BLOCK_NN)) {
    start_nn = tl.multiple_of(start_nn, $(BLOCK_NN))
    offs_seq = start_nn + offs_nn
    n_ctx_mask = offs_seq < $(n_ctx)
    token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=$((vob_end_id : Int)))
    id_mask = (token_ids >= $((vob_start_id : Int))) & (token_ids < $((vob_end_id : Int)))
    token_ids = token_ids - $((vob_start_id : Int))
    dim_mask = offs_d < $(hiden_size)
    load_mask = id_mask[:, None] & dim_mask[None, :]
    store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
    vecs = tl.load(weight + token_ids[:, None] * $(stride_weight_seq) + offs_d[None, :], mask=load_mask, other=0.0)
    tl.store(out + offs_seq[:, None] * $(stride_out_seq) + offs_d[None, :], vecs, mask=store_mask)
  }
}).surfaceBody)

abbrev sourceType := RegionName → RegionName → RegionName → Int → Int → Nat → Nat → Nat → Nat → Nat → Nat → Nat → ComputeKernel

def source (storage : Storage) (swapped : Bool) : sourceType :=
  match storage, swapped with
  | .fp32, Bool.false => embeddingSource%[tl.float32, original]
  | .fp32, Bool.true => embeddingSource%[tl.float32, indexCommuted]
  | .fp16, Bool.false => embeddingSource%[tl.float16, original]
  | .fp16, Bool.true => embeddingSource%[tl.float16, indexCommuted]
  | .bf16, Bool.false => embeddingSource%[tl.bfloat16, original]
  | .bf16, Bool.true => embeddingSource%[tl.bfloat16, indexCommuted]
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel.TypedSource
