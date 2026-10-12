import bench.tritonbench_g.fast_ce_loss.Source.Stage
import VeriTile.Triton.Source
/-! Source: fast_ce_loss.py, _cross_entropy_forward.
Complete source templates also cover _chunked_cross_entropy_forward and
_cross_entropy_backward. Source/Stage resolves only the two constexpr flags before dtype
inference; label and chunk conditionals remain runtime statements. Each template
has an original and an indexCommuted variant: the latter changes only
row_idx * logits_row_stride to logits_row_stride * row_idx.

Input storage is fp16/bf16/fp32; loss and LSE are fp32, and backward stores in
the input storage type. Explicit destination casts express implicit store
conversions. Signed labels retain -100. The source alias triton_tanh denotes
libdevice.tanh and remains distinct from tl.tanh in FP semantics. Scaling
precedes the fp32 logits cast; chunked gather alone casts before scaling.
Both chunked stores intentionally stay inside chunk_idx == 0.
Translation-surface blocker: DO_SOFTCAPPING and DO_LOGIT_SCALING are specialized
through cap/scale before dtype inference, rather than runtime Boolean binders.
All four flag combinations retain the complete original statement template;
FPFastCeSource checks the differing precision paths. -/
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.TypedSource
open Lean VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

syntax "fastCeForward%" "[" tritonDType "," ident "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastCeForward%[$storage:tritonDType,$capFlag:ident,$scaleFlag:ident,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx)
      else `(tritonStmt| logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64))
    `(fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ)
        => ComputeKernel.mk
      [logits_ptr,Region.cast labels_ptr] [loss_ptr,logsumexp_ptr] (fastCeStage%[$capFlag,$scaleFlag] triton {
  tl.region labels_ptr = tl.int32, logits_ptr = $storage, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
  row_idx = tl.program_id(0)
  $rowIndex
  loss_ptr += row_idx
  logsumexp_ptr += row_idx
  labels_ptr += row_idx
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = tl.load(labels_ptr).to(tl.int32)
  logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    logits = $((LOGIT_SCALE : ℝ)) * logits
  }
  if DO_SOFTCAPPING {
    logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
  }
  logits = logits.to(tl.float32)
  c = tl.max(logits, 0)
  logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
  if label_idx != $((-100 : Int)) {
    x = tl.load(logits_ptr + label_idx)
    if DO_LOGIT_SCALING {
      x = $((LOGIT_SCALE : ℝ)) * x
    }
    if DO_SOFTCAPPING {
      x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
    }
    loss = logsumexp - x.to(tl.float32)
  } else {
    loss = 0.0
  }
  tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
  tl.store(loss_ptr, loss.to(tl.float32))
}).surfaceBody)

def forward (storage : Storage) (cap scale swapped : Bool) := match storage,cap,scale with
  | .fp16,Bool.false,Bool.false => if swapped then fastCeForward%[tl.float16,noCap,noScale,indexCommuted] else fastCeForward%[tl.float16,noCap,noScale,original]
  | .fp16,Bool.false,Bool.true => if swapped then fastCeForward%[tl.float16,noCap,scale,indexCommuted] else fastCeForward%[tl.float16,noCap,scale,original]
  | .fp16,Bool.true,Bool.false => if swapped then fastCeForward%[tl.float16,cap,noScale,indexCommuted] else fastCeForward%[tl.float16,cap,noScale,original]
  | .fp16,Bool.true,Bool.true => if swapped then fastCeForward%[tl.float16,cap,scale,indexCommuted] else fastCeForward%[tl.float16,cap,scale,original]
  | .bf16,Bool.false,Bool.false => if swapped then fastCeForward%[tl.bfloat16,noCap,noScale,indexCommuted] else fastCeForward%[tl.bfloat16,noCap,noScale,original]
  | .bf16,Bool.false,Bool.true => if swapped then fastCeForward%[tl.bfloat16,noCap,scale,indexCommuted] else fastCeForward%[tl.bfloat16,noCap,scale,original]
  | .bf16,Bool.true,Bool.false => if swapped then fastCeForward%[tl.bfloat16,cap,noScale,indexCommuted] else fastCeForward%[tl.bfloat16,cap,noScale,original]
  | .bf16,Bool.true,Bool.true => if swapped then fastCeForward%[tl.bfloat16,cap,scale,indexCommuted] else fastCeForward%[tl.bfloat16,cap,scale,original]
  | .fp32,Bool.false,Bool.false => if swapped then fastCeForward%[tl.float32,noCap,noScale,indexCommuted] else fastCeForward%[tl.float32,noCap,noScale,original]
  | .fp32,Bool.false,Bool.true => if swapped then fastCeForward%[tl.float32,noCap,scale,indexCommuted] else fastCeForward%[tl.float32,noCap,scale,original]
  | .fp32,Bool.true,Bool.false => if swapped then fastCeForward%[tl.float32,cap,noScale,indexCommuted] else fastCeForward%[tl.float32,cap,noScale,original]
  | .fp32,Bool.true,Bool.true => if swapped then fastCeForward%[tl.float32,cap,scale,indexCommuted] else fastCeForward%[tl.float32,cap,scale,original]

syntax "fastCeChunked%" "[" tritonDType "," ident "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastCeChunked%[$storage:tritonDType,$capFlag:ident,$scaleFlag:ident,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx)
      else `(tritonStmt| logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64))
    `(fun (logits_ptr loss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE N_CHUNKS logits_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ)
        => ComputeKernel.mk
      [logits_ptr,Region.cast labels_ptr] [loss_ptr,logsumexp_ptr] (fastCeStage%[$capFlag,$scaleFlag] triton {
  tl.region labels_ptr = tl.int32, logits_ptr = $storage, loss_ptr = tl.float32, logsumexp_ptr = tl.float32
  row_idx = tl.program_id(0)
  chunk_idx = tl.program_id(1)
  $rowIndex
  loss_ptr += row_idx
  logsumexp_ptr += row_idx * $(N_CHUNKS) + chunk_idx
  labels_ptr += row_idx
  col_offsets = chunk_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = tl.load(labels_ptr).to(tl.int32)
  logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    logits = $((LOGIT_SCALE : ℝ)) * logits
  }
  if DO_SOFTCAPPING {
    logits = $((SOFTCAP : ℝ)) * triton_tanh(logits / $((SOFTCAP : ℝ)))
  }
  logits = logits.to(tl.float32)
  c = tl.max(logits, 0)
  logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))
  if chunk_idx == $(0) {
    if label_idx != $((-100 : Int)) {
      x = tl.load(logits_ptr + label_idx).to(tl.float32)
      if DO_LOGIT_SCALING {
        x = $((LOGIT_SCALE : ℝ)) * x
      }
      if DO_SOFTCAPPING {
        x = $((SOFTCAP : ℝ)) * triton_tanh(x / $((SOFTCAP : ℝ)))
      }
      loss = -1.0 * x.to(tl.float32)
    } else {
      loss = 0.0
    }
    tl.store(loss_ptr, loss.to(tl.float32))
    tl.store(logsumexp_ptr, logsumexp.to(tl.float32))
  }
}).surfaceBody)

def chunked (storage : Storage) (cap scale swapped : Bool) := match storage,cap,scale with
  | .fp16,Bool.false,Bool.false => if swapped then fastCeChunked%[tl.float16,noCap,noScale,indexCommuted] else fastCeChunked%[tl.float16,noCap,noScale,original]
  | .fp16,Bool.false,Bool.true => if swapped then fastCeChunked%[tl.float16,noCap,scale,indexCommuted] else fastCeChunked%[tl.float16,noCap,scale,original]
  | .fp16,Bool.true,Bool.false => if swapped then fastCeChunked%[tl.float16,cap,noScale,indexCommuted] else fastCeChunked%[tl.float16,cap,noScale,original]
  | .fp16,Bool.true,Bool.true => if swapped then fastCeChunked%[tl.float16,cap,scale,indexCommuted] else fastCeChunked%[tl.float16,cap,scale,original]
  | .bf16,Bool.false,Bool.false => if swapped then fastCeChunked%[tl.bfloat16,noCap,noScale,indexCommuted] else fastCeChunked%[tl.bfloat16,noCap,noScale,original]
  | .bf16,Bool.false,Bool.true => if swapped then fastCeChunked%[tl.bfloat16,noCap,scale,indexCommuted] else fastCeChunked%[tl.bfloat16,noCap,scale,original]
  | .bf16,Bool.true,Bool.false => if swapped then fastCeChunked%[tl.bfloat16,cap,noScale,indexCommuted] else fastCeChunked%[tl.bfloat16,cap,noScale,original]
  | .bf16,Bool.true,Bool.true => if swapped then fastCeChunked%[tl.bfloat16,cap,scale,indexCommuted] else fastCeChunked%[tl.bfloat16,cap,scale,original]
  | .fp32,Bool.false,Bool.false => if swapped then fastCeChunked%[tl.float32,noCap,noScale,indexCommuted] else fastCeChunked%[tl.float32,noCap,noScale,original]
  | .fp32,Bool.false,Bool.true => if swapped then fastCeChunked%[tl.float32,noCap,scale,indexCommuted] else fastCeChunked%[tl.float32,noCap,scale,original]
  | .fp32,Bool.true,Bool.false => if swapped then fastCeChunked%[tl.float32,cap,noScale,indexCommuted] else fastCeChunked%[tl.float32,cap,noScale,original]
  | .fp32,Bool.true,Bool.true => if swapped then fastCeChunked%[tl.float32,cap,scale,indexCommuted] else fastCeChunked%[tl.float32,cap,scale,original]

syntax "fastCeBackward%" "[" tritonDType "," ident "," ident "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(fastCeBackward%[$storage:tritonDType,$capFlag:ident,$scaleFlag:ident,$version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let rowIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then `(tritonStmt| logits_ptr += ($(logits_row_stride)).to(tl.int64) * row_idx)
      else `(tritonStmt| logits_ptr += row_idx * ($(logits_row_stride)).to(tl.int64))
    `(fun (logits_ptr dloss_ptr logsumexp_ptr : RegionName) (labels_ptr : Region .int)
        (VOCAB_SIZE logits_row_stride dloss_row_stride BLOCK_SIZE : Nat) (SOFTCAP LOGIT_SCALE : ℝ)
        => ComputeKernel.mk
      [logits_ptr,dloss_ptr,logsumexp_ptr,Region.cast labels_ptr] [logits_ptr] (fastCeStage%[$capFlag,$scaleFlag] triton {
  tl.region labels_ptr = tl.int32, logits_ptr = $storage, dloss_ptr = tl.float32, logsumexp_ptr = tl.float32
  row_idx = tl.program_id(0)
  block_idx = tl.program_id(1)
  $rowIndex
  dloss_ptr += row_idx * $(dloss_row_stride)
  col_offsets = block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  mask = col_offsets < $(VOCAB_SIZE)
  label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)
  if label_idx != $((-100 : Int)) {
    dloss = tl.load(dloss_ptr)
  }
  else {
    dloss = 0.0
  }
  x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))
  if DO_LOGIT_SCALING {
    x = x * $((LOGIT_SCALE : ℝ))
  }
  if DO_SOFTCAPPING {
    «partial» = triton_tanh(x / $((SOFTCAP : ℝ)))
    x = $((SOFTCAP : ℝ)) * «partial»
  }
  logsumexp = tl.load(logsumexp_ptr + row_idx)
  y = tl.exp(x.to(tl.float32) - logsumexp)
  y = tl.where(col_offsets == label_idx, y - 1.0, y)
  if DO_LOGIT_SCALING {
    y = y * $((LOGIT_SCALE : ℝ))
  }
  if DO_SOFTCAPPING {
    y = y * (1.0 - «partial» * «partial»)
  }
  tl.store(logits_ptr + col_offsets, (dloss * y).to($storage:tritonDType), mask=mask)
}).surfaceBody)

def backward (storage : Storage) (cap scale swapped : Bool) := match storage,cap,scale with
  | .fp16,Bool.false,Bool.false => if swapped then fastCeBackward%[tl.float16,noCap,noScale,indexCommuted] else fastCeBackward%[tl.float16,noCap,noScale,original]
  | .fp16,Bool.false,Bool.true => if swapped then fastCeBackward%[tl.float16,noCap,scale,indexCommuted] else fastCeBackward%[tl.float16,noCap,scale,original]
  | .fp16,Bool.true,Bool.false => if swapped then fastCeBackward%[tl.float16,cap,noScale,indexCommuted] else fastCeBackward%[tl.float16,cap,noScale,original]
  | .fp16,Bool.true,Bool.true => if swapped then fastCeBackward%[tl.float16,cap,scale,indexCommuted] else fastCeBackward%[tl.float16,cap,scale,original]
  | .bf16,Bool.false,Bool.false => if swapped then fastCeBackward%[tl.bfloat16,noCap,noScale,indexCommuted] else fastCeBackward%[tl.bfloat16,noCap,noScale,original]
  | .bf16,Bool.false,Bool.true => if swapped then fastCeBackward%[tl.bfloat16,noCap,scale,indexCommuted] else fastCeBackward%[tl.bfloat16,noCap,scale,original]
  | .bf16,Bool.true,Bool.false => if swapped then fastCeBackward%[tl.bfloat16,cap,noScale,indexCommuted] else fastCeBackward%[tl.bfloat16,cap,noScale,original]
  | .bf16,Bool.true,Bool.true => if swapped then fastCeBackward%[tl.bfloat16,cap,scale,indexCommuted] else fastCeBackward%[tl.bfloat16,cap,scale,original]
  | .fp32,Bool.false,Bool.false => if swapped then fastCeBackward%[tl.float32,noCap,noScale,indexCommuted] else fastCeBackward%[tl.float32,noCap,noScale,original]
  | .fp32,Bool.false,Bool.true => if swapped then fastCeBackward%[tl.float32,noCap,scale,indexCommuted] else fastCeBackward%[tl.float32,noCap,scale,original]
  | .fp32,Bool.true,Bool.false => if swapped then fastCeBackward%[tl.float32,cap,noScale,indexCommuted] else fastCeBackward%[tl.float32,cap,noScale,original]
  | .fp32,Bool.true,Bool.true => if swapped then fastCeBackward%[tl.float32,cap,scale,indexCommuted] else fastCeBackward%[tl.float32,cap,scale,original]
end VeriTile.Bench.TritonBenchG.FastCeLoss.TypedSource
