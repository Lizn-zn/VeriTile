import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Source: chunk_cumsum_kernel.py, chunk_global_cumsum_scalar_kernel.
Complete typed chunked cumulative sum. The candidate only reverses the
fp32 carry update; every scan, reduction, mask, cast and store is retained.
Python's output dtype attribute is resolved explicitly to OUT_DTYPE. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.TypedSource
open Lean VeriTile.Triton
syntax "chunkCumsumSource%" ident : term
set_option hygiene false in
macro_rules
  | `(chunkCumsumSource% $version:ident) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| b_z = b_zz + b_z)
      else `(tritonStmt| b_z += b_zz)
    `(fun (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat) =>
      ComputeKernel.mk [s] [o] (triton {
        tl.region s = DTYPE, o = OUT_DTYPE
        i_bh = tl.program_id(0)
        b_z = tl.zeros([], dtype=tl.float32)
        for i_t in range($(0), tl.cdiv($(T), $(BT)), $(1)) {
          p_s = tl.make_block_ptr(base=s+i_bh*$(T), shape=($(T)), strides=($(1)),
            offsets=(i_t*$(BT)), block_shape=($(BT)), order=(0))
          p_o = tl.make_block_ptr(base=o+i_bh*$(T), shape=($(T)), strides=($(1)),
            offsets=(i_t*$(BT)), block_shape=($(BT)), order=(0))
          b_s = tl.load(p_s, boundary_check=([0] : List Nat)).to(tl.float32)
          b_o = tl.cumsum(b_s, axis=0) + b_z[None]
          b_zz = tl.sum(b_s, axis=0)
          $update
          tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0] : List Nat))
        }
      }).surfaceBody)

def original := chunkCumsumSource% original

def sumLeft := chunkCumsumSource% sumLeft

end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.TypedSource
