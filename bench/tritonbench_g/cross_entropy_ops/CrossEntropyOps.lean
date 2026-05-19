import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.CrossEntropyOps

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `cross_entropy_ops.py`'s
`cross_entropy_fwd_kernel`.

This preserves the block logits load, `logit_scale`, optional smoothing sum,
LSE side store, label-in-block loss selection, optional split behavior, z-loss
computation, and non-split `z_loss_ptr` side store. -/
def cross_entropy_fwd_surface
    (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ)
    (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  logits_ptr = logits_ptr + row_idx * ($(logits_row_stride)).to(tl.int64)
  col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  label_idx = tl.load(labels_ptr + row_idx)
  logits = tl.load(logits_ptr + col_offsets,
    mask=col_offsets < $(n_cols), other=-float("inf")).to(tl.float32) * $(logit_scale)
  max_logits = tl.max(logits, 0)
  if HAS_SMOOTHING {
    sum_logits = tl.sum(tl.where(col_offsets < $(n_cols), logits, 0.0), 0)
  }
  lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
  tl.store(lse_ptr + col_block_idx * $(n_rows) + row_idx, lse)
  if label_idx == $((ignored_index : Int)) {
    loss = 0.0
    z_loss = 0.0
  } else {
    label_idx -= $((class_start_idx : Int))
    if (label_idx >= col_block_idx * $(BLOCK_SIZE)) and
        (label_idx < min($(n_cols), (col_block_idx + $(1)) * $(BLOCK_SIZE))) {
      logits_label = tl.load(logits_ptr + label_idx) * $(logit_scale)
      if HAS_SMOOTHING {
        loss = (lse if not SPLIT else 0.0) -
          $(smoothing) * sum_logits / $(total_classes) -
          (1.0 - $(smoothing)) * logits_label
      } else {
        loss = (lse if not SPLIT else 0.0) - logits_label
      }
    } else {
      if HAS_SMOOTHING {
        loss = $(smoothing) *
          ((lse if not SPLIT else 0.0) - sum_logits / $(total_classes))
      } else {
        loss = 0.0
      }
    }
    if not SPLIT {
      z_loss = $(lse_square_scale) * lse * lse
      loss += z_loss
    } else {
      z_loss = 0.0
    }
  }
  tl.store(loss_ptr + col_block_idx * $(n_rows) + row_idx, loss)
  if not SPLIT {
    tl.store(z_loss_ptr + col_block_idx * $(n_rows) + row_idx, z_loss)
  }
}

/-- The faithful full forward surface lowers to the algorithm layer, including
the smoothing, split, ignored-label, LSE/z-loss side-store branches. -/
theorem cross_entropy_fwd_surface_toAlgorithm_supported
    (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ)
    (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) :
    ∃ alg,
      (cross_entropy_fwd_surface loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr
        smoothing logit_scale lse_square_scale ignored_index total_classes
        class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE
        HAS_SMOOTHING SPLIT).toAlgorithm? = Except.ok alg := by
  simp [cross_entropy_fwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final-store slice of `cross_entropy_ops.py`'s
`cross_entropy_bwd_kernel`.

The full kernel computes `probs` from logits/LSE/labels/smoothing. This slice
starts from a precomputed `Probs` row and proves the masked
`dlogits = (dloss * logit_scale) * probs` writeback. -/
def cross_entropy_bwd_store_slice
    (dlogits_ptr dloss_ptr Probs : RegionName)
    (n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE : Nat)
    (logit_scale : ℝ) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  probs = tl.load(Probs + row_idx * $(probs_row_stride) + col_offsets,
    mask=col_offsets < $(n_cols), other=0.0)
  tl.store(dlogits_ptr + row_idx * $(dlogits_row_stride) + col_offsets,
    (dloss * $(logit_scale)) * probs, mask=col_offsets < $(n_cols))
}

def colOffset (s : BlockState) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * BLOCK_SIZE + i.val

def active (s : BlockState) (n_cols BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : Prop :=
  colOffset s BLOCK_SIZE i < n_cols

instance activeDecidable (s : BlockState) (n_cols BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) :
    Decidable (active s n_cols BLOCK_SIZE i) := by
  unfold active
  infer_instance

def outOffset
    (s : BlockState) (dlogits_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * dlogits_row_stride + colOffset s BLOCK_SIZE i

def probsOffset
    (s : BlockState) (probs_row_stride BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * probs_row_stride + colOffset s BLOCK_SIZE i

noncomputable def expectedGrad
    (s : BlockState) (dloss_ptr Probs : RegionName)
    (dloss_row_stride probs_row_stride BLOCK_SIZE : Nat)
    (logit_scale : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  (s.readMem dloss_ptr (s.pids 0 * dloss_row_stride) * logit_scale) *
    s.readMem Probs (probsOffset s probs_row_stride BLOCK_SIZE i)

/-- Algorithm-layer correctness for the final masked `dlogits` store. -/
theorem cross_entropy_bwd_store_slice_correct
    (dlogits_ptr dloss_ptr Probs : RegionName)
    (n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE : Nat)
    (logit_scale : ℝ)
    (s s' : BlockState)
    (hExec : exec (cross_entropy_bwd_store_slice dlogits_ptr dloss_ptr Probs
        n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE
        logit_scale) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem dlogits_ptr (outOffset s dlogits_row_stride BLOCK_SIZE i) =
        if active s n_cols BLOCK_SIZE i then
          expectedGrad s dloss_ptr Probs dloss_row_stride probs_row_stride
            BLOCK_SIZE logit_scale i
        else
          s.readMem dlogits_ptr (outOffset s dlogits_row_stride BLOCK_SIZE i) := by
  intro i
  have hInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * dlogits_row_stride + (s.pids 1 * BLOCK_SIZE + idx.1.val)) := by
    intro a b h
    have hInner :
        s.pids 1 * BLOCK_SIZE + a.1.val =
          s.pids 1 * BLOCK_SIZE + b.1.val := by
      exact Nat.add_left_cancel h
    have hab : a.1 = b.1 := Fin.ext (Nat.add_left_cancel hInner)
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, cross_entropy_bwd_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [outOffset, colOffset, active, expectedGrad, probsOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hActive : s.pids 1 * BLOCK_SIZE + i.val < n_cols
  · simp [hActive]
  · simp [hActive]

/-- Compute-facing correctness for the final masked `dlogits` store. -/
theorem cross_entropy_bwd_store_slice_compute_correct
    (dlogits_ptr dloss_ptr Probs : RegionName)
    (n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE : Nat)
    (logit_scale : ℝ)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_bwd_store_slice dlogits_ptr dloss_ptr Probs
        n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE
        logit_scale)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s n_cols BLOCK_SIZE i)
        (fun i => (dlogits_ptr, outOffset s dlogits_row_stride BLOCK_SIZE i)))
      (expected := fun i =>
        expectedGrad s dloss_ptr Probs dloss_row_stride probs_row_stride
          BLOCK_SIZE logit_scale i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [cross_entropy_bwd_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := cross_entropy_bwd_store_slice_correct dlogits_ptr dloss_ptr Probs
    n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE
    logit_scale s s' hExec i
  simpa [hActive] using h

/-- Proof-oriented LSE-store slice of `cross_entropy_ops.py`'s forward kernel.
Companion to the bwd_store_slice: takes a precomputed `LsePre` scalar and
proves the writeback into `lse_ptr`. -/
def cross_entropy_lse_store_slice
    (LsePre lse_ptr : RegionName) (n_rows : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  lse = tl.load(LsePre + col_block_idx * $(n_rows) + row_idx)
  tl.store(lse_ptr + col_block_idx * $(n_rows) + row_idx, lse)
}

def lseOutOffset (s : BlockState) (n_rows : Nat) : Nat :=
  s.pids 1 * n_rows + s.pids 0

noncomputable def lseStoreSpec (s : BlockState) (LsePre : RegionName)
    (n_rows : Nat) : ℝ :=
  s.readMem LsePre (lseOutOffset s n_rows)

theorem cross_entropy_lse_store_slice_correct
    (LsePre lse_ptr : RegionName) (n_rows : Nat) (s s' : BlockState)
    (hExec : exec (cross_entropy_lse_store_slice LsePre lse_ptr n_rows) s = some s') :
    s'.readMem lse_ptr (lseOutOffset s n_rows) =
      lseStoreSpec s LsePre n_rows := by
  simp [exec, cross_entropy_lse_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul] at hExec
  subst s'
  simp [lseOutOffset, lseStoreSpec]

theorem cross_entropy_lse_store_slice_compute_correct
    (LsePre lse_ptr : RegionName) (n_rows : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_lse_store_slice LsePre lse_ptr n_rows)
      (initialState := s)
      (write := fun _ : PUnit => some (lse_ptr, lseOutOffset s n_rows))
      (expected := fun _ => lseStoreSpec s LsePre n_rows) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [cross_entropy_lse_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact cross_entropy_lse_store_slice_correct LsePre lse_ptr n_rows s s' hExec

/-- Proof-oriented loss-store slice of `cross_entropy_ops.py`'s forward kernel.
Same scalar-copy pattern as the LSE store slice. -/
def cross_entropy_loss_store_slice
    (LossPre loss_ptr : RegionName) (n_rows : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  loss = tl.load(LossPre + col_block_idx * $(n_rows) + row_idx)
  tl.store(loss_ptr + col_block_idx * $(n_rows) + row_idx, loss)
}

noncomputable def lossStoreSpec (s : BlockState) (LossPre : RegionName)
    (n_rows : Nat) : ℝ :=
  s.readMem LossPre (lseOutOffset s n_rows)

theorem cross_entropy_loss_store_slice_correct
    (LossPre loss_ptr : RegionName) (n_rows : Nat) (s s' : BlockState)
    (hExec : exec (cross_entropy_loss_store_slice LossPre loss_ptr n_rows)
      s = some s') :
    s'.readMem loss_ptr (lseOutOffset s n_rows) =
      lossStoreSpec s LossPre n_rows := by
  simp [exec, cross_entropy_loss_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul] at hExec
  subst s'
  simp [lseOutOffset, lossStoreSpec]

theorem cross_entropy_loss_store_slice_compute_correct
    (LossPre loss_ptr : RegionName) (n_rows : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_loss_store_slice LossPre loss_ptr n_rows)
      (initialState := s)
      (write := fun _ : PUnit => some (loss_ptr, lseOutOffset s n_rows))
      (expected := fun _ => lossStoreSpec s LossPre n_rows) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [cross_entropy_loss_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact cross_entropy_loss_store_slice_correct LossPre loss_ptr n_rows s s' hExec

/-- Proof-oriented z_loss-store slice of `cross_entropy_ops.py`'s forward kernel. -/
def cross_entropy_z_loss_store_slice
    (ZLossPre z_loss_ptr : RegionName) (n_rows : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  z_loss = tl.load(ZLossPre + col_block_idx * $(n_rows) + row_idx)
  tl.store(z_loss_ptr + col_block_idx * $(n_rows) + row_idx, z_loss)
}

noncomputable def zLossStoreSpec (s : BlockState) (ZLossPre : RegionName)
    (n_rows : Nat) : ℝ :=
  s.readMem ZLossPre (lseOutOffset s n_rows)

theorem cross_entropy_z_loss_store_slice_correct
    (ZLossPre z_loss_ptr : RegionName) (n_rows : Nat) (s s' : BlockState)
    (hExec : exec (cross_entropy_z_loss_store_slice ZLossPre z_loss_ptr n_rows)
      s = some s') :
    s'.readMem z_loss_ptr (lseOutOffset s n_rows) =
      zLossStoreSpec s ZLossPre n_rows := by
  simp [exec, cross_entropy_z_loss_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul] at hExec
  subst s'
  simp [lseOutOffset, zLossStoreSpec]

theorem cross_entropy_z_loss_store_slice_compute_correct
    (ZLossPre z_loss_ptr : RegionName) (n_rows : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := cross_entropy_z_loss_store_slice ZLossPre z_loss_ptr n_rows)
      (initialState := s)
      (write := fun _ : PUnit => some (z_loss_ptr, lseOutOffset s n_rows))
      (expected := fun _ => zLossStoreSpec s ZLossPre n_rows) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [cross_entropy_z_loss_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact cross_entropy_z_loss_store_slice_correct ZLossPre z_loss_ptr n_rows s s' hExec

end VeriTile.Bench.TritonBenchG.CrossEntropyOps
