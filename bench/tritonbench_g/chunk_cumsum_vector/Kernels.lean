import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Source: chunk_cumsum_vector.py, chunk_global_cumsum_vector_kernel.
The complete typed candidate reverses only the fp32 vector carry update after
each store. The triangular dot, TF32 setting, masks and source branch remain. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.TypedSource
open Lean VeriTile.Triton
syntax "vectorCumsumSource%" ident : term
set_option hygiene false in
macro_rules
  | `(vectorCumsumSource% $version:ident) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| b_z = tl.sum(b_s, 0) + b_z)
      else `(tritonStmt| b_z += tl.sum(b_s, 0))
    `(fun (dtype out_dtype : FloatDType) (s z : RegionName)
        (s_s_h s_s_t s_s_d T S BT BS : Nat) =>
      ComputeKernel.mk [s] [z] (triton {
        tl.region s = DTYPE, z = OUT_DTYPE
        i_s = tl.program_id(0)
        i_bh = tl.program_id(1)
        o_i = tl.arange(0, $(BT))
        m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
        b_z = tl.zeros([$(BS)], dtype=tl.float32)
        for i_t in range($(0), tl.cdiv($(T), $(BT)), $(1)) {
          p_s = tl.make_block_ptr(base=s+i_bh*$(s_s_h), shape=($(T), $(S)),
            strides=($(s_s_t), $(s_s_d)), offsets=(i_t*$(BT), i_s*$(BS)),
            block_shape=($(BT), $(BS)), order=(1, 0))
          p_z = tl.make_block_ptr(base=z+i_bh*$(s_s_h), shape=($(T), $(S)),
            strides=($(s_s_t), $(s_s_d)), offsets=(i_t*$(BT), i_s*$(BS)),
            block_shape=($(BT), $(BS)), order=(1, 0))
          b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
          b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
          tl.store(p_z, (b_c).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
          if i_t >= $(0) { $update }
        }
      }).surfaceBody)
def original := vectorCumsumSource% original
def sumLeft := vectorCumsumSource% sumLeft
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.TypedSource
