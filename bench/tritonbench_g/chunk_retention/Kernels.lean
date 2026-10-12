import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Source: chunk_retention.py, chunk_retention_fwd_kernel_h.
All four per-JIT sources. Original signed decay offsets are retained, including
negative tail lanes. The explicit arange-to-int cast states Triton's signed
index type in the DSL; subsequent float promotion is automatic.
Translation-surface blocker: tl.arange is explicitly cast to tl.int32 to model
the source signed indices; this adds an identity cast to the Python call sequence.
Backward state uses the square tile shape forced by its two untransposed dot
arguments and final accumulator, and preinitializes the post-loop index to 0.
Its correspondence is restricted to NT>0; its unusual final pointer and dead
history load are preserved. Reverse time uses the ascending counter NT-1-j.
-/
namespace VeriTile.Bench.TritonBenchG.ChunkRetention
open Lean VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

/-- Source: chunk_retention_fwd_kernel_h. -/
syntax "retentionForwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionForwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_h = tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False) + d_b * b_h)
      else `(tritonStmt| b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to($dotTy:tritonDType), allow_tf32=False))
    `(fun (dtype out_dtype : FloatDType) (k v h initial_state final_state : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k,v,initial_state] [h,final_state] (triton {
        tl.region k = $dotTy, v = $dotTy, h = $dotTy, initial_state = DTYPE, final_state = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=initial_state + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = (tl.load(p_h0, boundary_check=([0, 1] : List Nat))).to(tl.float32)
        }
        for i_t in range($(NT)) {
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          if i_t == $(NT) - $(1) and $(T) % $(BT) != $(0) {
            d_b = tl.math.exp2(($(T) % $(BT)) * b_b)
            d_i = tl.math.exp2((($(T) % $(BT)) - o_i - $(1)) * b_b)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=final_state + ((i_bh * $(K)) * $(V)), shape=($(K), $(V)), strides=($(V), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)
set_option maxHeartbeats 8000000 in
def forwardHSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 =>
      if swapped then retentionForwardHSource%[tl.float32, dotLeft]
      else retentionForwardHSource%[tl.float32, original]
  | .fp16 =>
      if swapped then retentionForwardHSource%[tl.float16, dotLeft]
      else retentionForwardHSource%[tl.float16, original]
  | .bf16 =>
      if swapped then retentionForwardHSource%[tl.bfloat16, dotLeft]
      else retentionForwardHSource%[tl.bfloat16, original]

def forwardH (dotStorage : Storage) := forwardHSource dotStorage Bool.false
def forwardHDotLeft (dotStorage : Storage) := forwardHSource dotStorage Bool.true

/-- Source: chunk_retention_fwd_kernel_o. -/
syntax "retentionForwardOSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionForwardOSource%[$dotTy:tritonDType, $vTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let update : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then `(tritonStmt| b_o = $((scale : ℝ)) * (b_o + tl.dot((b_s).to($vTy:tritonDType), b_v, allow_tf32=False)))
      else `(tritonStmt| b_o = (b_o + tl.dot((b_s).to($vTy:tritonDType), b_v, allow_tf32=False)) * $((scale : ℝ)))
    `(fun (out_dtype : FloatDType) (q k v h o : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,k,v,h] [o] (triton {
        tl.region q = $dotTy, k = $dotTy, h = $dotTy, v = $vTy, o = OUT_DTYPE
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_i = tl.math.exp2((o_i + $(1)) * b_b)
        m_s = o_i[:, None] >= o_i[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
          p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BK)), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          b_o += tl.dot((b_q * d_i[:, None]).to($dotTy:tritonDType), b_h, allow_tf32=False)
          b_s += tl.dot(b_q, b_k, allow_tf32=False)
        }
        b_s *= d_s
        p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
        $update
        p_o = tl.make_block_ptr(base=o + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
set_option maxHeartbeats 8000000 in
def forwardOSource (dotStorage : Storage) (valueStorage : Storage) (swapped : Bool) :=
  match dotStorage,valueStorage with
  | .fp32, .fp32 =>
      if swapped then retentionForwardOSource%[tl.float32, tl.float32, scaleFirst]
      else retentionForwardOSource%[tl.float32, tl.float32, original]
  | .fp32, .fp16 =>
      if swapped then retentionForwardOSource%[tl.float32, tl.float16, scaleFirst]
      else retentionForwardOSource%[tl.float32, tl.float16, original]
  | .fp32, .bf16 =>
      if swapped then retentionForwardOSource%[tl.float32, tl.bfloat16, scaleFirst]
      else retentionForwardOSource%[tl.float32, tl.bfloat16, original]
  | .fp16, .fp32 =>
      if swapped then retentionForwardOSource%[tl.float16, tl.float32, scaleFirst]
      else retentionForwardOSource%[tl.float16, tl.float32, original]
  | .fp16, .fp16 =>
      if swapped then retentionForwardOSource%[tl.float16, tl.float16, scaleFirst]
      else retentionForwardOSource%[tl.float16, tl.float16, original]
  | .fp16, .bf16 =>
      if swapped then retentionForwardOSource%[tl.float16, tl.bfloat16, scaleFirst]
      else retentionForwardOSource%[tl.float16, tl.bfloat16, original]
  | .bf16, .fp32 =>
      if swapped then retentionForwardOSource%[tl.bfloat16, tl.float32, scaleFirst]
      else retentionForwardOSource%[tl.bfloat16, tl.float32, original]
  | .bf16, .fp16 =>
      if swapped then retentionForwardOSource%[tl.bfloat16, tl.float16, scaleFirst]
      else retentionForwardOSource%[tl.bfloat16, tl.float16, original]
  | .bf16, .bf16 =>
      if swapped then retentionForwardOSource%[tl.bfloat16, tl.bfloat16, scaleFirst]
      else retentionForwardOSource%[tl.bfloat16, tl.bfloat16, original]

def forwardO (dotStorage : Storage) (valueStorage : Storage) := forwardOSource dotStorage valueStorage Bool.false
def forwardOScaleFirst (dotStorage : Storage) (valueStorage : Storage) := forwardOSource dotStorage valueStorage Bool.true

/-- Source: chunk_retention_bwd_kernel_dh. -/
syntax "retentionBackwardHSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionBackwardHSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_dh = tl.dot((b_o * d_i[:, None]).to($dotTy:tritonDType), b_v, allow_tf32=False) + b_dh)
      else `(tritonStmt| b_dh += tl.dot((b_o * d_i[:, None]).to($dotTy:tritonDType), b_v, allow_tf32=False))
    `(fun (out_dtype : FloatDType) (q do_ dh v : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT NT : Nat) =>
      ComputeKernel.mk [q,do_,v,dh] [dh] (triton {
        tl.region q = $dotTy, do_ = $dotTy, v = $dotTy, dh = OUT_DTYPE
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_b = tl.math.exp2($(BT) * b_b)
        d_i = tl.math.exp2((o_i + $(1)) * b_b)
        b_dh = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        i_t = $(0)
        for j in range($(0), $(NT), $(1)) {
          i_t = $(NT) - $(1) - j
          p_o = tl.make_block_ptr(base=do_ + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BT))), block_shape=($(BT), $(BT)), order=(1, 0))
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BT))), block_shape=($(BT), $(BT)), order=(1, 0))
          p_h = tl.make_block_ptr(base=dh + (i_bh * $(s_h_h)) + ((i_t * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_k * $(BT)), (i_v * $(BT))), block_shape=($(BT), $(BT)), order=(1, 0))
          b_o = tl.load(p_o, boundary_check=([0, 1] : List Nat))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          $update
        }
        b_dh *= d_b
        p_dh = tl.make_block_ptr(base=dh + (i_bh * $(s_h_h)) + ((i_k * $(K)) * $(V)), shape=($(K), $(V)), strides=($(s_h_t), 1), offsets=((i_v * $(BT)), (i_t * $(BT))), block_shape=($(BT), $(BT)), order=(1, 0))
        tl.store(p_dh, (b_dh).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
set_option maxHeartbeats 8000000 in
def backwardHSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 => if swapped then retentionBackwardHSource%[tl.float32, dotLeft] else retentionBackwardHSource%[tl.float32, original]
  | .fp16 => if swapped then retentionBackwardHSource%[tl.float16, dotLeft] else retentionBackwardHSource%[tl.float16, original]
  | .bf16 => if swapped then retentionBackwardHSource%[tl.bfloat16, dotLeft] else retentionBackwardHSource%[tl.bfloat16, original]

def backwardH (dotStorage : Storage) := backwardHSource dotStorage Bool.false
def backwardHDotLeft (dotStorage : Storage) := backwardHSource dotStorage Bool.true

/-- Source: chunk_retention_bwd_kernel_dqkv. -/
syntax "retentionBackwardQKVSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(retentionBackwardQKVSource%[$dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotLeft do
      Macro.throwError "expected original or dotLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `dotLeft then `(tritonStmt| b_dq = tl.dot(b_ds, b_k, allow_tf32=False) + b_dq * d_q[:, None])
      else `(tritonStmt| b_dq = b_dq * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False))
    `(fun  (q k v h do_ dh dq dk dv : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV NT : Nat) =>
      ComputeKernel.mk [q,k,v,h,do_,dh] [dq,dk,dv] (triton {
        tl.region q = $dotTy, k = $dotTy, v = $dotTy, h = $dotTy, do_ = $dotTy, dh = $dotTy, dq = $dotTy, dk = $dotTy, dv = $dotTy
        i_k = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_h = i_bh % $(H)
        n_bh = tl.num_programs(2)
        b_b = tl.math.log2(1.0 - tl.math.exp2($((-5 : ℝ)) - i_h * 1.0))
        o_i = (tl.arange(0, $(BT))).to(tl.int32)
        d_q = tl.math.exp2((o_i + $(1)) * b_b)
        d_k = tl.math.exp2(($(BT) - o_i - $(1)) * b_b)
        d_q = (d_q * $((scale : ℝ))).to(tl.float32)
        m_s = o_i[:, None] >= o_i[None, :]
        d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * $((scale : ℝ))
        p_q = tl.make_block_ptr(base=q + (i_bh * $(s_qk_h)), shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)), offsets=((i_k * $(BK)), (i_t * $(BT))), block_shape=($(BK), $(BT)), order=(0, 1))
        p_k = tl.make_block_ptr(base=k + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
        b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
        b_s = (tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s))
        b_dq = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
        b_dk = tl.zeros([$(BT), $(BK)], dtype=tl.float32)
        b_ds = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
        for i_v in range(tl.cdiv($(V), $(BV))) {
          p_v = tl.make_block_ptr(base=v + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + (i_bh * $(s_h_h)), shape=($(V), $(NT * K)), strides=(1, $(s_h_t)), offsets=((i_v * $(BV)), ((i_t * $(K)) + (i_k * $(BK)))), block_shape=($(BV), $(BK)), order=(0, 1))
          p_do = tl.make_block_ptr(base=do_ + (i_bh * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          p_dh = tl.make_block_ptr(base=dh + (i_bh * $(s_h_h)), shape=($(NT * K), $(V)), strides=($(s_h_t), 1), offsets=(((i_t * $(K)) + (i_k * $(BK))), (i_v * $(BV))), block_shape=($(BK), $(BV)), order=(1, 0))
          p_dv = tl.make_block_ptr(base=dv + (((i_k * n_bh) + i_bh) * $(s_vo_h)), shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)), offsets=((i_t * $(BT)), (i_v * $(BV))), block_shape=($(BT), $(BV)), order=(1, 0))
          b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
          b_do = tl.load(p_do, boundary_check=([0, 1] : List Nat))
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          b_dh = tl.load(p_dh, boundary_check=([0, 1] : List Nat))
          b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
          b_dq += tl.dot(b_do, b_h, allow_tf32=False)
          b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
          b_dv = (tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot((b_s).to($dotTy:tritonDType), b_do, allow_tf32=False))
          tl.store(p_dv, (b_dv).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
        }
        b_ds = (b_ds * d_s).to($dotTy:tritonDType)
        $update
        b_dk = b_dk * d_k[:, None] + tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
        p_dq = tl.make_block_ptr(base=dq + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        p_dk = tl.make_block_ptr(base=dk + (i_bh * $(s_qk_h)), shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)), offsets=((i_t * $(BT)), (i_k * $(BK))), block_shape=($(BT), $(BK)), order=(1, 0))
        tl.store(p_dq, (b_dq).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
        tl.store(p_dk, (b_dk).to($dotTy:tritonDType), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
set_option maxHeartbeats 8000000 in
def backwardQKVSource (dotStorage : Storage) (swapped : Bool) :=
  match dotStorage with
  | .fp32 =>
      if swapped then retentionBackwardQKVSource%[tl.float32, dotLeft]
      else retentionBackwardQKVSource%[tl.float32, original]
  | .fp16 =>
      if swapped then retentionBackwardQKVSource%[tl.float16, dotLeft]
      else retentionBackwardQKVSource%[tl.float16, original]
  | .bf16 =>
      if swapped then retentionBackwardQKVSource%[tl.bfloat16, dotLeft]
      else retentionBackwardQKVSource%[tl.bfloat16, original]

def backwardQKV (dotStorage : Storage) := backwardQKVSource dotStorage Bool.false
def backwardQKVDotLeft (dotStorage : Storage) := backwardQKVSource dotStorage Bool.true

end VeriTile.Bench.TritonBenchG.ChunkRetention

#guard_msgs (drop info) in
#auditModuleAxioms
