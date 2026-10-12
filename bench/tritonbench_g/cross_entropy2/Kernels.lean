import VeriTile.Triton.Source
import VeriTile.Triton.DSL
/-! Complete forward and backward cross-entropy sources. Both candidates only
commute the exact integer product in the column offset. Numerical operations,
label branches, optional smoothing/split paths, masks and writes are retained.
Logit/gradient storage supports fp16/bf16/fp32; loss, LSE, z-loss and dloss are fp32. The scale and gated z-loss store remain.
The real and FP contracts are per program; host tensor-parallel aggregation
and fixed-width index overflow are outside their scope. -/
namespace VeriTile.Bench.TritonBenchG.CrossEntropy2
open Lean VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32
  deriving DecidableEq

/-! Source: cross_entropy2.py, cross_entropy_fwd_kernel. -/
syntax "crossEntropy2FwdSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(crossEntropy2FwdSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let colIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| col_offsets = $(BLOCK_SIZE) * col_block_idx + tl.arange(0, $(BLOCK_SIZE)))
      else `(tritonStmt| col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE)))
    `(fun (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ)
    (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) => ComputeKernel.mk
      [logits_ptr,Region.cast labels_ptr] [loss_ptr,lse_ptr,z_loss_ptr] (triton {
  tl.region logits_ptr = $storage, loss_ptr = tl.float32, lse_ptr = tl.float32, z_loss_ptr = tl.float32
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  logits_ptr = logits_ptr + row_idx * ($(logits_row_stride)).to(tl.int64)
  $colIndex
  label_idx = tl.load(labels_ptr + row_idx)
  logits = tl.load(logits_ptr + col_offsets,
    mask=col_offsets < $(n_cols), other=-float("inf")).to(tl.float32) * $((logit_scale : ℝ))
  max_logits = tl.max(logits, 0)
  if HAS_SMOOTHING {
    sum_logits = tl.sum(tl.where(col_offsets < $(n_cols), logits, 0.0), 0)
  }
  lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
  tl.store(lse_ptr + col_block_idx * $(n_rows) + row_idx, (lse).to(tl.float32))
  if label_idx == $((ignored_index : Int)) {
    loss = 0.0
    z_loss = 0.0
  } else {
    label_idx -= $((class_start_idx : Int))
    if (label_idx >= col_block_idx * $(BLOCK_SIZE)) and
        (label_idx < min($(n_cols), (col_block_idx + $(1)) * $(BLOCK_SIZE))) {
      logits_label = tl.load(logits_ptr + label_idx) * $((logit_scale : ℝ))
      if HAS_SMOOTHING {
        loss = (lse if not SPLIT else 0.0) -
          $((smoothing : ℝ)) * sum_logits / $(total_classes) -
          (1 - $((smoothing : ℝ))) * logits_label
      } else {
        loss = (lse if not SPLIT else 0.0) - logits_label
      }
    } else {
      if HAS_SMOOTHING {
        loss = $((smoothing : ℝ)) *
          ((lse if not SPLIT else 0.0) - sum_logits / $(total_classes))
      } else {
        loss = 0.0
      }
    }
    if not SPLIT {
      z_loss = $((lse_square_scale : ℝ)) * lse * lse
      loss += z_loss
    } else {
      z_loss = 0.0
    }
  }
  tl.store(loss_ptr + col_block_idx * $(n_rows) + row_idx, (loss).to(tl.float32))
  if not SPLIT {
    tl.store(z_loss_ptr + col_block_idx * $(n_rows) + row_idx, (z_loss).to(tl.float32))
  }
}).surfaceBody)

def forward (storage : Storage) (swapped : Bool) := match storage with
  | .fp16 => if swapped then crossEntropy2FwdSource%[tl.float16,indexCommuted] else crossEntropy2FwdSource%[tl.float16,original]
  | .bf16 => if swapped then crossEntropy2FwdSource%[tl.bfloat16,indexCommuted] else crossEntropy2FwdSource%[tl.bfloat16,original]
  | .fp32 => if swapped then crossEntropy2FwdSource%[tl.float32,indexCommuted] else crossEntropy2FwdSource%[tl.float32,original]

/-! Source: cross_entropy2.py, cross_entropy_bwd_kernel. -/
syntax "crossEntropy2BwdSource%" "[" tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(crossEntropy2BwdSource%[$storage:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `indexCommuted do
      Macro.throwError "expected original or indexCommuted"
    let colIndex : TSyntax `tritonStmt ←
      if version.getId == `indexCommuted then
        `(tritonStmt| col_offsets = $(BLOCK_SIZE) * col_block_idx + tl.arange(0, $(BLOCK_SIZE)))
      else `(tritonStmt| col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE)))
    `(fun (dlogits_ptr dloss_ptr logits_ptr lse_ptr : RegionName) (labels_ptr : Region .int)
        (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
        (total_classes : Nat) (class_start_idx : Int)
        (n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE : Nat)
        (HAS_SMOOTHING : Bool) => ComputeKernel.mk
      [dloss_ptr,logits_ptr,lse_ptr,Region.cast labels_ptr] [dlogits_ptr] (triton {
  tl.region logits_ptr = $storage, dlogits_ptr = $storage,
    dloss_ptr = tl.float32, lse_ptr = tl.float32, z_loss_ptr = tl.float32
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  logits_ptr = logits_ptr + row_idx * ($(logits_row_stride)).to(tl.int64)
  dlogits_ptr = dlogits_ptr + row_idx * ($(dlogits_row_stride)).to(tl.int64)
  $colIndex
  label_idx = tl.load(labels_ptr + row_idx)
  if label_idx != $((ignored_index : Int)) {
    dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  } else {
    dloss = 0.0
  }
  logits = tl.load(logits_ptr + col_offsets,
    mask=col_offsets < $(n_cols), other=-float("inf")).to(tl.float32) * $((logit_scale : ℝ))
  lse = tl.load(lse_ptr + row_idx)
  probs = tl.exp(logits - lse)
  probs += 2.0 * $((lse_square_scale : ℝ)) * lse * probs
  label_idx -= $((class_start_idx : Int))
  if HAS_SMOOTHING {
    smooth_negative = $((smoothing : ℝ)) / $(total_classes)
    probs = tl.where(col_offsets == label_idx, probs - (1 - $((smoothing : ℝ))), probs) - smooth_negative
  } else {
    probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
  }
  tl.store(dlogits_ptr + col_offsets, ((dloss * $(logit_scale)) * probs).to($storage:tritonDType), mask=col_offsets < $(n_cols))
}).surfaceBody)

def backward (storage : Storage) (swapped : Bool) := match storage with
  | .fp16 => if swapped then crossEntropy2BwdSource%[tl.float16,indexCommuted] else crossEntropy2BwdSource%[tl.float16,original]
  | .bf16 => if swapped then crossEntropy2BwdSource%[tl.bfloat16,indexCommuted] else crossEntropy2BwdSource%[tl.bfloat16,original]
  | .fp32 => if swapped then crossEntropy2BwdSource%[tl.float32,indexCommuted] else crossEntropy2BwdSource%[tl.float32,original]
end VeriTile.Bench.TritonBenchG.CrossEntropy2
