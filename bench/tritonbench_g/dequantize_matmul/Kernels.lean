import VeriTile.Triton.Source
import VeriTile.Triton.DSL
/-! Source: dequantize_matmul.py, dequantize_kernel

The complete dequantization JIT. Signed int8
weights are multiplied by fp32 column scales and stored in the selected
floating output precision. The wider column-only scale mask is preserved.
The candidate commutes only the exact integer block product in n_mask.
The host's subsequent torch.mm call is outside this JIT contract. -/
namespace VeriTile.Bench.TritonBenchG.DequantizeMatmul.TypedSource
open Lean VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq
syntax "dequantizeMatmulSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(dequantizeMatmulSource%[$outType:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let maskStmt : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| n_mask = $(BLOCK_SIZE_N) * n_block_idx + offs_n[None, :] < $(N))
      else `(tritonStmt| n_mask = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :] < $(N))
    `(fun (b_ptr b_scale_ptr fpb_ptr : RegionName)
      (K N stride_bk stride_bn stride_fpbk stride_fpbn BLOCK_SIZE_N BLOCK_SIZE_K : Nat) =>
      ComputeKernel.mk [b_ptr,b_scale_ptr] [fpb_ptr] (triton {
  tl.region b_ptr = tl.int8, b_scale_ptr = tl.float32, fpb_ptr = $outType
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = tl.arange(0, $(BLOCK_SIZE_N))
  b_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_bk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_bn)
  fpb_offs = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None]) * $(stride_fpbk) +
    (n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]) * $(stride_fpbn)
  bs_offs = n_block_idx * $(BLOCK_SIZE_N) + offs_n[None, :]
  $maskStmt
  mask = (k_block_idx * $(BLOCK_SIZE_K) + offs_k[:, None] < $(K)) & n_mask
  int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
  tl.store(fpb_ptr + fpb_offs, (int_b * scale_b).to($outType:tritonDType), mask=mask)
}).surfaceBody)

def kernel (output : Storage) (swapped : Bool) := match output with
  | .fp16 => if swapped then dequantizeMatmulSource%[tl.float16,indexCommuted] else dequantizeMatmulSource%[tl.float16,original]
  | .bf16 => if swapped then dequantizeMatmulSource%[tl.bfloat16,indexCommuted] else dequantizeMatmulSource%[tl.bfloat16,original]
  | .fp32 => if swapped then dequantizeMatmulSource%[tl.float32,indexCommuted] else dequantizeMatmulSource%[tl.float32,original]
end VeriTile.Bench.TritonBenchG.DequantizeMatmul.TypedSource
