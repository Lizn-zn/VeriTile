import VeriTile.Triton.DSL
import VeriTile.Triton.Source

/-! Source: chunked_cumsum_fwd.py, _chunk_cumsum_fwd_kernel.
Both complete sources retain optional bias/softplus, clamp, the padded output
store, A load after that store and the within-chunk scan. The candidate only
commutes dt * A[:,None]. Input storage is independent; outputs follow the
host wrapper's fp32 allocations. The upper limit is finite or positive-infinite (the host default).
The positive-infinite minimum remains an explicit, opaque FP operation.
Translation-surface blocker: bias storage and finite/positive-infinite upper
limits are explicit source specializations; the selected clamp and multiply
statements are spliced into the same complete source template.
Host launch supplies pid_c*chunk_size <= seqlen so natural subtraction matches
the Python remaining length. This condition belongs in the real IO contract.
-/
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
open Lean VeriTile.Triton

inductive BiasStorage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

/-- Finite upper limit or the host default +∞. -/
inductive UpperBound where
  | finite (value : ℝ)
  | posInf

instance : Coe ℝ UpperBound := ⟨UpperBound.finite⟩
instance (n : Nat) : OfNat UpperBound n := ⟨.finite n⟩

/-- Independent real meaning of the upper clamp. -/
noncomputable def UpperBound.cap (upper : UpperBound) (x : ℝ) : ℝ :=
  match upper with
  | .finite value => min x value
  | .posInf => x

syntax "chunkedCumsumSource%" "[" tritonDType "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(chunkedCumsumSource%[$biasTy:tritonDType, $version:ident, $bound:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let product : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then `(tritonStmt| dA = A[:, None] * dt)
      else `(tritonStmt| dA = dt * A[:, None])
    unless bound.getId == `finite || bound.getId == `infinite do
      Macro.throwError "expected finite or infinite"
    let clamp : TSyntax `tritonStmt ←
      if bound.getId == `infinite then
        `(tritonStmt| dt = tl.minimum(tl.maximum(dt, $(dt_min)), float("inf")))
      else
        `(tritonStmt| dt = tl.minimum(tl.maximum(dt, $(dt_min)), $(dt_max)))
    `(fun (dtype out_dtype : FloatDType)
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat)
    (dt_min dt_max : ℝ)
    (stride_dt_batch stride_dt_seqlen stride_dt_head
      stride_A_head
      stride_dt_bias_head
      stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize
      stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize : Nat)
    (DT_SOFTPLUS HAS_DT_BIAS : Bool)
    (BLOCK_SIZE_H BLOCK_SIZE_CHUNK : Nat)
    => ComputeKernel.mk [dt_ptr,A_ptr,dt_bias_ptr] [dt_out_ptr,dA_cumsum_ptr] (triton {
  tl.region dt_ptr = DTYPE, A_ptr = OUT_DTYPE, dt_bias_ptr = $biasTy,
    dt_out_ptr = tl.float32, dA_cumsum_ptr = tl.float32
  pid_b = tl.program_id(axis=0)
  pid_c = tl.program_id(axis=1)
  pid_h = tl.program_id(axis=2)
  dt_ptr += pid_b * $(stride_dt_batch) +
    pid_c * $(chunk_size) * $(stride_dt_seqlen)
  dt_out_ptr += pid_b * $(stride_dt_out_batch) +
    pid_c * $(stride_dt_out_chunk)
  dA_cumsum_ptr += pid_b * $(stride_dA_cs_batch) +
    pid_c * $(stride_dA_cs_chunk)
  offs_h = pid_h * $(BLOCK_SIZE_H) + tl.arange(0, $(BLOCK_SIZE_H))
  offs_c = tl.arange(0, $(BLOCK_SIZE_CHUNK))
  dt_ptrs = dt_ptr + offs_h[:, None] * $(stride_dt_head) +
    offs_c[None, :] * $(stride_dt_seqlen)
  A_ptrs = A_ptr + offs_h * $(stride_A_head)
  dt_out_ptrs = dt_out_ptr + offs_h[:, None] * $(stride_dt_out_head) +
    offs_c[None, :] * $(stride_dt_out_csize)
  dA_cs_ptrs = dA_cumsum_ptr + offs_h[:, None] * $(stride_dA_cs_head) +
    offs_c[None, :] * $(stride_dA_cs_csize)
  chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
  dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
  if HAS_DT_BIAS {
    dt_bias = tl.load(dt_bias_ptr + offs_h * $(stride_dt_bias_head),
      mask=offs_h < $(nheads), other=0.0).to(tl.float32)
    dt += dt_bias[:, None]
  }
  if DT_SOFTPLUS {
    dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
  }
  $clamp
  dt = tl.where((offs_h[:, None] < $(nheads)) & (offs_c[None, :] < chunk_size_limit),
    dt, 0.0)
  tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < $(chunk_size)))
  A = tl.load(A_ptrs, mask=offs_h < $(nheads), other=0.0).to(tl.float32)
  $product
  dA_cs = tl.cumsum(dA, axis=1)
  tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < $(nheads)) &
    (offs_c[None, :] < $(chunk_size)))
}).surfaceBody)

def finiteSource (biasStorage : BiasStorage) (swapped : Bool) :=
  match biasStorage with
  | .fp32 => if swapped then chunkedCumsumSource%[tl.float32,scaleFirst,finite] else chunkedCumsumSource%[tl.float32,original,finite]
  | .fp16 => if swapped then chunkedCumsumSource%[tl.float16,scaleFirst,finite] else chunkedCumsumSource%[tl.float16,original,finite]
  | .bf16 => if swapped then chunkedCumsumSource%[tl.bfloat16,scaleFirst,finite] else chunkedCumsumSource%[tl.bfloat16,original,finite]

def infiniteSource (biasStorage : BiasStorage) (swapped : Bool) :=
  match biasStorage with
  | .fp32 => if swapped then chunkedCumsumSource%[tl.float32,scaleFirst,infinite] else chunkedCumsumSource%[tl.float32,original,infinite]
  | .fp16 => if swapped then chunkedCumsumSource%[tl.float16,scaleFirst,infinite] else chunkedCumsumSource%[tl.float16,original,infinite]
  | .bf16 => if swapped then chunkedCumsumSource%[tl.bfloat16,scaleFirst,infinite] else chunkedCumsumSource%[tl.bfloat16,original,infinite]

/-- Both limit forms keep every operation. The unused dummy argument of the
internal infinite template does not enter its AST. -/
def source (biasStorage : BiasStorage) (swapped : Bool) (dtype out_dtype : FloatDType)
    (dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr : RegionName)
    (batch seqlen nheads chunk_size : Nat) (dt_min : ℝ) (dt_max : UpperBound) :=
  match dt_max with
  | .finite hi => finiteSource biasStorage swapped dtype out_dtype
      dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min hi
  | .posInf => infiniteSource biasStorage swapped dtype out_dtype
      dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min 0

def original (biasStorage : BiasStorage) := source biasStorage Bool.false
def scaleFirst (biasStorage : BiasStorage) := source biasStorage Bool.true
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd
