import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Source: chunk_gla_simple.py, chunk_simple_gla_fwd_kernel_o.
Complete source and scale-first candidate. Q/K/H share a compatible dot-input
storage type; V and G have independent storage choices. Output storage is
explicit. Intrinsic exp, half gate subtraction before exp, all three
TF32-disabled dots, the score-to-V cast and unspecified padding are retained.
The source tensor dtype attributes are resolved to the selected storage
parameters before expansion. The candidate only commutes the final fp32
multiplication by scale. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple
open Lean VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq
syntax "simpleGLASource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(simpleGLASource%[$dotTy:tritonDType, $vTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let result : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then
        `(tritonStmt| b_o = $((scale : ℝ)) * (b_o + tl.dot((b_s).to($vTy:tritonDType), b_v, allow_tf32=False)))
      else
        `(tritonStmt| b_o = (b_o + tl.dot((b_s).to($vTy:tritonDType), b_v, allow_tf32=False)) * $((scale : ℝ)))
    `(fun (out_dtype : FloatDType) (q k v h g o : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,k,v,h,g] [o] (triton {
        tl.region q = $dotTy, k = $dotTy, h = $dotTy, v = $vTy, g = $gTy, o = OUT_DTYPE
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        o_i = tl.arange(0, $(BT))
        m_s = o_i[:, None] >= o_i[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
            shape=($(K), $(T)), strides=($(1), $(s_k_t)),
            offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
          p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
            shape=($(K), $(V)), strides=($(s_h_t), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          b_o += tl.dot(b_q, b_h, allow_tf32=False)
          b_s += tl.dot(b_q, b_k, allow_tf32=False)
        }
        p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
          strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
        b_g = tl.load(p_g, boundary_check=([0] : List Nat))
        b_o = b_o * tl.exp(b_g)[:, None]
        b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
        b_s = tl.where(m_s, b_s, 0)
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
        $result
        p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
set_option maxHeartbeats 8000000 in
def source (dotStorage valueStorage gateStorage : Storage) (swapped : Bool) :=
  match dotStorage,valueStorage,gateStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then simpleGLASource%[tl.float32, tl.float32, tl.float32, scaleFirst]
      else simpleGLASource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, .fp32, .fp16 =>
      if swapped then simpleGLASource%[tl.float32, tl.float32, tl.float16, scaleFirst]
      else simpleGLASource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then simpleGLASource%[tl.float32, tl.float32, tl.bfloat16, scaleFirst]
      else simpleGLASource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then simpleGLASource%[tl.float32, tl.float16, tl.float32, scaleFirst]
      else simpleGLASource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then simpleGLASource%[tl.float32, tl.float16, tl.float16, scaleFirst]
      else simpleGLASource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then simpleGLASource%[tl.float32, tl.float16, tl.bfloat16, scaleFirst]
      else simpleGLASource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then simpleGLASource%[tl.float32, tl.bfloat16, tl.float32, scaleFirst]
      else simpleGLASource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then simpleGLASource%[tl.float32, tl.bfloat16, tl.float16, scaleFirst]
      else simpleGLASource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then simpleGLASource%[tl.float32, tl.bfloat16, tl.bfloat16, scaleFirst]
      else simpleGLASource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then simpleGLASource%[tl.float16, tl.float32, tl.float32, scaleFirst]
      else simpleGLASource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then simpleGLASource%[tl.float16, tl.float32, tl.float16, scaleFirst]
      else simpleGLASource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then simpleGLASource%[tl.float16, tl.float32, tl.bfloat16, scaleFirst]
      else simpleGLASource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then simpleGLASource%[tl.float16, tl.float16, tl.float32, scaleFirst]
      else simpleGLASource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then simpleGLASource%[tl.float16, tl.float16, tl.float16, scaleFirst]
      else simpleGLASource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then simpleGLASource%[tl.float16, tl.float16, tl.bfloat16, scaleFirst]
      else simpleGLASource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then simpleGLASource%[tl.float16, tl.bfloat16, tl.float32, scaleFirst]
      else simpleGLASource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then simpleGLASource%[tl.float16, tl.bfloat16, tl.float16, scaleFirst]
      else simpleGLASource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then simpleGLASource%[tl.float16, tl.bfloat16, tl.bfloat16, scaleFirst]
      else simpleGLASource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then simpleGLASource%[tl.bfloat16, tl.float32, tl.float32, scaleFirst]
      else simpleGLASource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then simpleGLASource%[tl.bfloat16, tl.float32, tl.float16, scaleFirst]
      else simpleGLASource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then simpleGLASource%[tl.bfloat16, tl.float32, tl.bfloat16, scaleFirst]
      else simpleGLASource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then simpleGLASource%[tl.bfloat16, tl.float16, tl.float32, scaleFirst]
      else simpleGLASource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then simpleGLASource%[tl.bfloat16, tl.float16, tl.float16, scaleFirst]
      else simpleGLASource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then simpleGLASource%[tl.bfloat16, tl.float16, tl.bfloat16, scaleFirst]
      else simpleGLASource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then simpleGLASource%[tl.bfloat16, tl.bfloat16, tl.float32, scaleFirst]
      else simpleGLASource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then simpleGLASource%[tl.bfloat16, tl.bfloat16, tl.float16, scaleFirst]
      else simpleGLASource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then simpleGLASource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, scaleFirst]
      else simpleGLASource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]

def original (dotStorage valueStorage gateStorage : Storage) := source dotStorage valueStorage gateStorage Bool.false
def scaleFirst (dotStorage valueStorage gateStorage : Storage) := source dotStorage valueStorage gateStorage Bool.true
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple
