import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Complete typed source for `chunk_delta_rule_fwd_kernel_h` in
`chunk_delta_fwd.py`. The candidate commutes only the outer fp32 carry update
`b_h += b_h_cumsum`. It retains the entire inner loop, pre-update history store,
corrected-value store, optional initial/final states, both TF32-disabled dots
and all source boundary checks.

K and D share dot-input storage; H uses K storage and V_new uses V storage,
as allocated by the host. Initial/final state storage is independent. Resolved
DTYPE and specialized storage casts preserve the Python dtype attributes, including both
conversions to the loaded K dtype.
Omitted boundary padding remains undefined.
Kernel dimensions/strides and both state flags stay symbolic. Host grid
composition and fixed-width address overflow are outside the per-program model.
-/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.TypedSource
open Lean VeriTile.Triton

syntax "chunkDeltaSource%" "[" tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(chunkDeltaSource%[$valueTy:tritonDType, $finalTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let update : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| b_h = b_h_cumsum + b_h)
      else `(tritonStmt| b_h += b_h_cumsum)
    `(fun (dtype out_dtype : FloatDType)
        (k v d v_new h initial_state final_state : RegionName)
        (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
          H T K V BT BC BK BV NT : Nat)
        (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) =>
      ComputeKernel.mk [k, v, d, initial_state] [v_new, h, final_state] (triton {
        tl.region k = DTYPE, d = DTYPE, h = DTYPE, v = $valueTy,
          v_new = $valueTy, initial_state = OUT_DTYPE, final_state = $finalTy
        i_k = tl.program_id(0)
        i_v = tl.program_id(1)
        i_bh = tl.program_id(2)
        b_h = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
        if USE_INITIAL_STATE {
          p_h0 = tl.make_block_ptr(base=initial_state + i_bh * $(K) * $(V),
            shape=($(K), $(V)), strides=($(V), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          b_h = tl.load(p_h0, boundary_check=([0, 1] : List Nat)).to(tl.float32)
        }
        for i_t in range($(0), $(NT), $(1)) {
          p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
            shape=($(K), $(V)), strides=($(s_h_t), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_h, (b_h).to(DTYPE), boundary_check=([0, 1] : List Nat))
          b_h_cumsum = tl.zeros([$(BK), $(BV)], dtype=tl.float32)
          for i_c in range($(0), tl.cdiv($(BT), $(BC)), $(1)) {
            p_k = tl.make_block_ptr(base=k + i_bh * $(s_qk_h),
              shape=($(K), $(T)), strides=($(s_qk_d), $(s_qk_t)),
              offsets=(i_k * $(BK), i_t * $(BT) + i_c * $(BC)),
              block_shape=($(BK), $(BC)), order=(0, 1))
            p_d = tl.make_block_ptr(base=d + i_bh * $(s_qk_h),
              shape=($(T), $(K)), strides=($(s_qk_t), $(s_qk_d)),
              offsets=(i_t * $(BT) + i_c * $(BC), i_k * $(BK)),
              block_shape=($(BC), $(BK)), order=(1, 0))
            p_v = tl.make_block_ptr(base=v + i_bh * $(s_vo_h),
              shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
              offsets=(i_t * $(BT) + i_c * $(BC), i_v * $(BV)),
              block_shape=($(BC), $(BV)), order=(1, 0))
            p_v_new = tl.make_block_ptr(base=v_new + i_bh * $(s_vo_h),
              shape=($(T), $(V)), strides=($(s_vo_t), $(s_vo_d)),
              offsets=(i_t * $(BT) + i_c * $(BC), i_v * $(BV)),
              block_shape=($(BC), $(BV)), order=(1, 0))
            b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
            b_d = tl.load(p_d, boundary_check=([0, 1] : List Nat))
            b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
            b_v -= tl.dot(b_d, (b_h).to(DTYPE), allow_tf32=false)
            tl.store(p_v_new, (b_v).to($valueTy:tritonDType),
              boundary_check=([0, 1] : List Nat))
            b_h_cumsum += tl.dot(b_k, (b_v).to(DTYPE), allow_tf32=false)
          }
          $update
        }
        if STORE_FINAL_STATE {
          p_ht = tl.make_block_ptr(base=final_state + i_bh * $(K) * $(V),
            shape=($(K), $(V)), strides=($(V), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          tl.store(p_ht, (b_h).to($finalTy:tritonDType),
            boundary_check=([0, 1] : List Nat))
        }
      }).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def Storage.dtype : Storage → FloatDType
  | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16

def specialize (kd value initial final : Storage) (swapped : Bool) :=
  let dtype := kd.dtype
  let out_dtype := initial.dtype
  match value, final, swapped with
  | .fp32, .fp32, Bool.false => chunkDeltaSource%[tl.float32, tl.float32, original] dtype out_dtype
  | .fp32, .fp32, Bool.true => chunkDeltaSource%[tl.float32, tl.float32, sumLeft] dtype out_dtype
  | .fp32, .fp16, Bool.false => chunkDeltaSource%[tl.float32, tl.float16, original] dtype out_dtype
  | .fp32, .fp16, Bool.true => chunkDeltaSource%[tl.float32, tl.float16, sumLeft] dtype out_dtype
  | .fp32, .bf16, Bool.false => chunkDeltaSource%[tl.float32, tl.bfloat16, original] dtype out_dtype
  | .fp32, .bf16, Bool.true => chunkDeltaSource%[tl.float32, tl.bfloat16, sumLeft] dtype out_dtype
  | .fp16, .fp32, Bool.false => chunkDeltaSource%[tl.float16, tl.float32, original] dtype out_dtype
  | .fp16, .fp32, Bool.true => chunkDeltaSource%[tl.float16, tl.float32, sumLeft] dtype out_dtype
  | .fp16, .fp16, Bool.false => chunkDeltaSource%[tl.float16, tl.float16, original] dtype out_dtype
  | .fp16, .fp16, Bool.true => chunkDeltaSource%[tl.float16, tl.float16, sumLeft] dtype out_dtype
  | .fp16, .bf16, Bool.false => chunkDeltaSource%[tl.float16, tl.bfloat16, original] dtype out_dtype
  | .fp16, .bf16, Bool.true => chunkDeltaSource%[tl.float16, tl.bfloat16, sumLeft] dtype out_dtype
  | .bf16, .fp32, Bool.false => chunkDeltaSource%[tl.bfloat16, tl.float32, original] dtype out_dtype
  | .bf16, .fp32, Bool.true => chunkDeltaSource%[tl.bfloat16, tl.float32, sumLeft] dtype out_dtype
  | .bf16, .fp16, Bool.false => chunkDeltaSource%[tl.bfloat16, tl.float16, original] dtype out_dtype
  | .bf16, .fp16, Bool.true => chunkDeltaSource%[tl.bfloat16, tl.float16, sumLeft] dtype out_dtype
  | .bf16, .bf16, Bool.false => chunkDeltaSource%[tl.bfloat16, tl.bfloat16, original] dtype out_dtype
  | .bf16, .bf16, Bool.true => chunkDeltaSource%[tl.bfloat16, tl.bfloat16, sumLeft] dtype out_dtype

def original (kd value initial final : Storage) := specialize kd value initial final Bool.false

def sumLeft (kd value initial final : Storage) := specialize kd value initial final Bool.true

end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.TypedSource
