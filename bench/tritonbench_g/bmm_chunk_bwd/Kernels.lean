import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Complete typed source of `bmm_chunk_bwd.py`'s `_bmm_chunk_bwd_kernel` (line 21). The in-body chunk limit,
both residual choices, every tail mask, dot casts and output conversion are
retained. The candidate only changes the final acc+res to res+acc.

A and Dout may independently use fp32/fp16/bf16. Dot storage follows the Python
host's priority: bf16, then fp16, then fp32. Residual/output dtypes remain
explicit parameters. Dimensions, strides and all three program IDs remain
symbolic. Each load inherits its pointer storage dtype before the explicit
conversion, including nested load/cast expressions.
Size/index arithmetic uses mathematical naturals; matching a valid
host chunk requires pid_c*chunk_size <= seqlen. Fixed-width overflow and host
scheduling are outside this source model.

Translation-surface blocker: this macro specializes host-selected storage and
dot dtypes, declares pointer storage channels and exposes load casts. The
source load kwargs, in-body minimum, loop masks and stores remain unchanged.
Correct.lean states independent real correctness for both versions;
FPEquiv.lean states the admitted residual-add equivalence. -/
open Lean
open VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.TypedSource

syntax "bmmChunkBwdSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(bmmChunkBwdSource%[$aTy:tritonDType, $dTy:tritonDType, $dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `residualLeft do
      Macro.throwError "expected original or residualLeft"
    let add : TSyntax `tritonStmt ←
      if version.getId == `residualLeft then `(tritonStmt| acc = res + acc)
      else `(tritonStmt| acc += res)
    `(fun (dtype out_dtype : FloatDType) (a_ptr dout_ptr db_ptr res_ptr : RegionName)
        (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
          stride_dout_batch stride_dout_chunk stride_dout_head stride_dout_csize_m stride_dout_csize_n
          stride_db_batch stride_db_seqlen stride_db_head stride_db_k
          stride_res_batch stride_res_seqlen stride_res_head stride_res_k
          BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_CS : Nat) (HAS_RESIDUAL : Bool) =>
    ComputeKernel.mk [a_ptr, dout_ptr, res_ptr] [db_ptr] (triton {
      tl.region a_ptr = $aTy, dout_ptr = $dTy, res_ptr = DTYPE, db_ptr = OUT_DTYPE
      pid_b = tl.program_id(axis=1)
      pid_ch = tl.program_id(axis=2)
      pid_c = pid_ch // $(ngroups)
      pid_h = pid_ch - pid_c * $(ngroups)
      num_pid_n = tl.cdiv($(K), $(BLOCK_SIZE_N))
      pid_m = tl.program_id(axis=0) // num_pid_n
      pid_n = tl.program_id(axis=0) % num_pid_n
      a_ptr += pid_b * $(stride_a_batch) + pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
      dout_ptr += pid_b * $(stride_dout_batch) + pid_c * $(stride_dout_chunk) + pid_h * $(stride_dout_head)
      offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
      offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
      offs_cs = tl.arange(0, $(BLOCK_SIZE_CS))
      dout_ptrs = dout_ptr + (offs_m[:, None] * $(stride_dout_csize_n) + offs_cs[None, :] * $(stride_dout_csize_m))
      a_ptrs = a_ptr + (offs_cs[:, None] * $(stride_a_seqlen) + offs_n[None, :] * $(stride_ak))
      chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
      acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
      for cs in range($(0), tl.cdiv(chunk_size_limit, $(BLOCK_SIZE_CS)), $(1)) {
        dout = (tl.load(dout_ptrs, mask=(offs_m[:, None] < $(chunk_size)) &
          (offs_cs[None, :] < chunk_size_limit - cs * $(BLOCK_SIZE_CS)), other=0.0)).to($dotTy:tritonDType)
        a = (tl.load(a_ptrs, mask=(offs_cs[:, None] < chunk_size_limit - cs * $(BLOCK_SIZE_CS)) &
          (offs_n[None, :] < $(K)), other=0.0)).to($dotTy:tritonDType)
        acc += tl.dot(dout, a)
        dout_ptrs += $(BLOCK_SIZE_CS) * $(stride_dout_csize_m)
        a_ptrs += $(BLOCK_SIZE_CS) * $(stride_a_seqlen)
      }
      offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
      offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
      if HAS_RESIDUAL {
        res_ptr += pid_b * $(stride_res_batch) + pid_c * $(chunk_size) * $(stride_res_seqlen) + pid_h * $(stride_res_head)
        res_ptrs = res_ptr + (offs_m[:, None] * $(stride_res_seqlen) + offs_n[None, :] * $(stride_res_k))
        res = (tl.load(res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
          (offs_n[None, :] < $(K)))).to(tl.float32)
        $add
      }
      db = (acc).to(OUT_DTYPE)
      db_ptr += pid_b * $(stride_db_batch) + pid_c * $(chunk_size) * $(stride_db_seqlen) + pid_h * $(stride_db_head)
      db_ptrs = db_ptr + (offs_m[:, None] * $(stride_db_seqlen) + offs_n[None, :] * $(stride_db_k))
      tl.store(db_ptrs, db, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < $(K)))
    }).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def specialize (a dout : Storage) (swapped : Bool) :=
  match a, dout, swapped with
  | .fp32, .fp32, Bool.false => bmmChunkBwdSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => bmmChunkBwdSource%[tl.float32, tl.float32, tl.float32, residualLeft]
  | .fp32, .fp16, Bool.false => bmmChunkBwdSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, Bool.true => bmmChunkBwdSource%[tl.float32, tl.float16, tl.float16, residualLeft]
  | .fp32, .bf16, Bool.false => bmmChunkBwdSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => bmmChunkBwdSource%[tl.float32, tl.bfloat16, tl.bfloat16, residualLeft]
  | .fp16, .fp32, Bool.false => bmmChunkBwdSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, Bool.true => bmmChunkBwdSource%[tl.float16, tl.float32, tl.float16, residualLeft]
  | .fp16, .fp16, Bool.false => bmmChunkBwdSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => bmmChunkBwdSource%[tl.float16, tl.float16, tl.float16, residualLeft]
  | .fp16, .bf16, Bool.false => bmmChunkBwdSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => bmmChunkBwdSource%[tl.float16, tl.bfloat16, tl.bfloat16, residualLeft]
  | .bf16, .fp32, Bool.false => bmmChunkBwdSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp32, Bool.true => bmmChunkBwdSource%[tl.bfloat16, tl.float32, tl.bfloat16, residualLeft]
  | .bf16, .fp16, Bool.false => bmmChunkBwdSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .fp16, Bool.true => bmmChunkBwdSource%[tl.bfloat16, tl.float16, tl.bfloat16, residualLeft]
  | .bf16, .bf16, Bool.false => bmmChunkBwdSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => bmmChunkBwdSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, residualLeft]

def original (a dout : Storage) := specialize a dout Bool.false

def residualLeft (a dout : Storage) := specialize a dout Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.TypedSource
