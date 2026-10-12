import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Complete typed source of `bmm_chunk_fwd.py`'s `_bmm_chunk_fwd_kernel`.
The candidate swaps only the operands of the fp32 addition in each K block.
Both sources retain the actual chunk length, K-tail masks, signed sequence-ID
filter, output conversion and all three program IDs. A/B independently use
fp32/fp16/bf16 storage with the host's bf16, then fp16, then fp32 dot priority.

The nested causal return becomes an active-block guard: a skipped tile performs
no load or store; a surviving diagonal tile has no elementwise causal mask.
Loads use the shortened chunk length, while the final store uses chunk_size.
Thus padded output positions are written as zero. The sequence-ID sentinels
are distinct signed values -1 and -2.

Translation-surface blocker: the macro specializes host-selected dtypes,
declares pointer storage and expresses the early return by its complementary
guard. Each explicit load cast keeps the pointer's original storage dtype.
Address arithmetic uses mathematical naturals; valid host chunks, fixed-width
overflow and composition of concurrent grid programs are separate boundaries.
-/
open Lean
open VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.TypedSource
syntax "bmmChunkFwdSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(bmmChunkFwdSource%[$aTy:tritonDType, $bTy:tritonDType, $dotTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `dotFirst do
      Macro.throwError "expected original or dotFirst"
    let add : TSyntax `tritonStmt ←
      if version.getId == `dotFirst then `(tritonStmt| acc = tl.dot(a, b) + acc)
      else `(tritonStmt| acc += tl.dot(a, b))
    `(fun (out_dtype : FloatDType) (a_ptr b_ptr out_ptr : RegionName)
        (seq_idx_ptr : Region .int)
        (seqlen chunk_size K ngroups stride_a_batch stride_a_seqlen stride_a_head stride_ak
          stride_b_batch stride_b_seqlen stride_b_head stride_bk
          stride_out_batch stride_out_chunk stride_out_head stride_outm stride_outn
          stride_seq_idx_batch stride_seq_idx_seqlen
          BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K : Nat) (IS_CAUSAL HAS_SEQ_IDX : Bool) =>
    ComputeKernel.mk [a_ptr, b_ptr, Region.cast seq_idx_ptr] [out_ptr] (triton {
      tl.region a_ptr = $aTy, b_ptr = $bTy, out_ptr = OUT_DTYPE
      pid_b = tl.program_id(axis=1)
      pid_ch = tl.program_id(axis=2)
      pid_c = pid_ch // $(ngroups)
      pid_h = pid_ch - pid_c * $(ngroups)
      num_pid_n = tl.cdiv($(chunk_size), $(BLOCK_SIZE_N))
      pid_m = tl.program_id(axis=0) // num_pid_n
      pid_n = tl.program_id(axis=0) % num_pid_n
      active_block = true
      if IS_CAUSAL {
        active_block = pid_n * $(BLOCK_SIZE_N) < (pid_m + $(1)) * $(BLOCK_SIZE_M)
      }
      if active_block {
        a_ptr += pid_b * $(stride_a_batch) + pid_c * $(chunk_size) * $(stride_a_seqlen) + pid_h * $(stride_a_head)
        b_ptr += pid_b * $(stride_b_batch) + pid_c * $(chunk_size) * $(stride_b_seqlen) + pid_h * $(stride_b_head)
        if HAS_SEQ_IDX {
          seq_idx_ptr += pid_b * $(stride_seq_idx_batch) + pid_c * $(chunk_size) * $(stride_seq_idx_seqlen)
        }
        offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
        offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
        offs_k = tl.arange(0, $(BLOCK_SIZE_K))
        a_ptrs = a_ptr + (offs_m[:, None] * $(stride_a_seqlen) + offs_k[None, :] * $(stride_ak))
        b_ptrs = b_ptr + (offs_k[:, None] * $(stride_bk) + offs_n[None, :] * $(stride_b_seqlen))
        chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
        acc = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
        for k in range($(0), tl.cdiv($(K), $(BLOCK_SIZE_K)), $(1)) {
          a = (tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) &
            (offs_k[None, :] < $(K) - k * $(BLOCK_SIZE_K)), other=0.0)).to($dotTy:tritonDType)
          b = (tl.load(b_ptrs, mask=(offs_k[:, None] < $(K) - k * $(BLOCK_SIZE_K)) &
            (offs_n[None, :] < chunk_size_limit), other=0.0)).to($dotTy:tritonDType)
          $add
          a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
          b_ptrs += $(BLOCK_SIZE_K) * $(stride_bk)
        }
        offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
        offs_n = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
        if HAS_SEQ_IDX {
          chunk_size_limit = min($(chunk_size), $(seqlen) - pid_c * $(chunk_size))
          seq_idx_m = tl.load(seq_idx_ptr + offs_m * $(stride_seq_idx_seqlen), mask=offs_m < chunk_size_limit, other=-1)
          seq_idx_n = tl.load(seq_idx_ptr + offs_n * $(stride_seq_idx_seqlen), mask=offs_n < chunk_size_limit, other=-2)
          acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
        }
        out = (acc).to(OUT_DTYPE)
        out_ptr += pid_b * $(stride_out_batch) + pid_c * $(stride_out_chunk) + pid_h * $(stride_out_head)
        out_ptrs = out_ptr + ($(stride_outm) * offs_m[:, None] + offs_n[None, :] * $(stride_outn))
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < $(chunk_size)) & (offs_n[None, :] < $(chunk_size)))
      }
    }).surfaceBody)

inductive Storage where
  | fp32 | fp16 | bf16
  deriving DecidableEq

def specialize (a b : Storage) (swapped : Bool) :=
  match a, b, swapped with
  | .fp32, .fp32, Bool.false => bmmChunkFwdSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, .fp32, Bool.true => bmmChunkFwdSource%[tl.float32, tl.float32, tl.float32, dotFirst]
  | .fp32, .fp16, Bool.false => bmmChunkFwdSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, Bool.true => bmmChunkFwdSource%[tl.float32, tl.float16, tl.float16, dotFirst]
  | .fp32, .bf16, Bool.false => bmmChunkFwdSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp32, .bf16, Bool.true => bmmChunkFwdSource%[tl.float32, tl.bfloat16, tl.bfloat16, dotFirst]
  | .fp16, .fp32, Bool.false => bmmChunkFwdSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, Bool.true => bmmChunkFwdSource%[tl.float16, tl.float32, tl.float16, dotFirst]
  | .fp16, .fp16, Bool.false => bmmChunkFwdSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, Bool.true => bmmChunkFwdSource%[tl.float16, tl.float16, tl.float16, dotFirst]
  | .fp16, .bf16, Bool.false => bmmChunkFwdSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .bf16, Bool.true => bmmChunkFwdSource%[tl.float16, tl.bfloat16, tl.bfloat16, dotFirst]
  | .bf16, .fp32, Bool.false => bmmChunkFwdSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp32, Bool.true => bmmChunkFwdSource%[tl.bfloat16, tl.float32, tl.bfloat16, dotFirst]
  | .bf16, .fp16, Bool.false => bmmChunkFwdSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .fp16, Bool.true => bmmChunkFwdSource%[tl.bfloat16, tl.float16, tl.bfloat16, dotFirst]
  | .bf16, .bf16, Bool.false => bmmChunkFwdSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .bf16, Bool.true => bmmChunkFwdSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, dotFirst]

def original (a b : Storage) := specialize a b Bool.false

def dotFirst (a b : Storage) := specialize a b Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.TypedSource
