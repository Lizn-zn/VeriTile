import VeriTile.Triton

/-!
# `cross_entropy1` — strict per-kernel correctness

`cross_entropy_fwd_kernel` computes, per `(row_idx, col_block_idx)` program, the
block log-sum-exp of the logits row (with optional label smoothing and an
optional `SPLIT`/tensor-parallel mode), stores the LSE side output, selects the
per-row cross-entropy loss for the label that falls in this column block, adds an
optional `lse_square_scale·lse²` term, and stores the loss. The companion
`cross_entropy_bwd_kernel` writes `dlogits = dloss · probs`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (the grid `(n_rows, cdiv(n_cols,
BLOCK_SIZE))`, scheduling, and how the runtime composes per-program writes /
reduces the per-block LSE side outputs across column blocks) is the *trusted
boundary*. Because `row_idx`/`col_block_idx` are universally quantified, the
per-program statements cover every program of the grid.

## Proof architecture

```
cross_entropy_fwd_correctness                          ← TOP THEOREM (fwd, ⊨ headline)
  ├─ crossEntropyFwdIO                                 MetaGatherMasked2DKernelIO₂ₓ₂ signature (label slot + gather)
  ├─ cross_entropy_fwd_flattenOk                       flat-memory bridge coverage
  ├─ cross_entropy_fwd_traceSafe                       per-execution safety walk
  │    ├─ ceFwdLsePrefix_traceSafe                     prefix walk (label slot, masked row, LSE store)
  │    └─ ceFwdLossTail_traceSafe                      branchy walk (gated gather load, loss store)
  └─ cross_entropy_fwd_region_run                      region-model masked Hoare triple (hrun)
       ├─ ceFwdLsePrefix_isSome / _run_facts           prefix termination + registers + cell frame
       ├─ ceFwdLossTail_run / _run_fallback            tail termination + frame (+ ⊥-path loss = 0)
       ├─ cross_entropy_fwd_lse_correct                ← genuine masked-lane LSE store
  │    ├─ ceFwd_body_split                             body = LSE prefix ++ loss tail
  │    ├─ storeFree_stepStmt_mem / _stepStmts_mem      loss tail preserves lse_ptr
  │    └─ ceFwdLsePrefix_correct                       prefix writes partialLSE_full
  └─ cross_entropy_fwd_loss_correct                    ← genuine branchy loss store
       ├─ ceFwdLsePrefix_regs                          mid-state register values
       ├─ ceFwdLsePrefix_mem_of_ne                     prefix store frames logits_ptr
       └─ crossEntropyLossSpec                         five-way closed form (input memory)
cross_entropy_bwd_store_slice_compute_correct          ← masked dlogits = dloss·probs
  └─ cross_entropy_bwd_store_slice_correct
```

The forward kernel is now **genuinely value-correct end-to-end** (both side
outputs, all branches):

* **`lse_ptr`**: executing the *full* surface writes to
  `lse_ptr[col_block·n_rows + row]` exactly the masked-lane stable log-sum-exp
  (`partialLSE_full`) of the INPUT block logits, mirroring the proven
  `logsumexp_fwd` machinery (`other = -inf → ⊥`, `tl.max → sup'`, `exp(⊥-m)=0`,
  `tl.sum` over valid lanes). The loss tail does not disturb the LSE cell — its
  only store targets `loss_ptr ≠ lse_ptr`, established structurally via the
  `storeFree` `mem`-frame lemma.
* **`loss_ptr`**: writes exactly the faithful five-way cross-entropy
  `crossEntropyLossSpec` (`label==ignored` / label-in-block / `HAS_SMOOTHING` /
  `SPLIT` / `lse²`), with every sub-term — `sum_logits`, the data-dependent
  `logits_label` load, and `lse` — read from INPUT memory. Proved by case-split
  on the two `Bool` flags and the runtime label conditions, using the mid-state
  register characterization and the prefix `mem`-frame (under `lse_ptr ≠
  logits_ptr`, so the prefix LSE store doesn't clobber the logit read).

Both value specs read INPUT memory, never `exec(...).readMem` — non-self-referential.
The companion backward kernel's `dlogits = dloss·probs` store is also genuine
(`probs` is a kernel input).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.heuristics`
(`HAS_SMOOTHING`) and `@triton.autotune` are not modeled (`HAS_SMOOTHING`/`SPLIT`
are plain `Bool` parameters). The `.to(tl.float32)`/`.to(tl.int64)` casts erase
to identity at the algorithm layer. The forward LSE uses
`other = -float("inf")` for out-of-block lanes. The masked `dlogits` store leaves
inactive lanes (`col_offsets ≥ n_cols`) untouched and assumes the per-tile output
offset is injective. The spec is built inline; it does not reference a
`VeriTile.Triton.Math.*` oracle.
-/

namespace VeriTile.Bench.TritonBenchG.CrossEntropy1

open VeriTile.Triton
open VeriTile.Triton.TiledLogSumExp

set_option linter.unusedSimpArgs false

/-! ## Memory-frame helper

The forward loss/scale tail is a sequence of register assignments and nested
`if`/`else` blocks (only assignments before the stores), so it writes no memory
and preserves every cell. We use the shared generic frame `storeFree` /
`storeFree_stepStmts_mem` (in `VeriTile.Triton`, `VeriTile/Triton/Semantics/Step.lean`)
to show the genuine `lse` store survives the loss branches. -/

/-- Faithful transcription of `cross_entropy1.py`'s
`cross_entropy_fwd_kernel`.

This preserves the block logits load, optional smoothing sum, LSE side store,
label-in-block loss selection, optional split behavior, and LSE-square term. -/
def cross_entropy_fwd_surface
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ)
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
    mask=col_offsets < $(n_cols), other=-float("inf")).to(tl.float32)
  max_logits = tl.max(logits, 0)
  if HAS_SMOOTHING {
    sum_logits = tl.sum(tl.where(col_offsets < $(n_cols), logits, 0.0), 0)
  }
  lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
  tl.store(lse_ptr + col_block_idx * $(n_rows) + row_idx, lse)
  if label_idx == $((ignored_index : Int)) {
    loss = 0.0
  } else {
    label_idx -= $((class_start_idx : Int))
    if (label_idx >= col_block_idx * $(BLOCK_SIZE)) and
        (label_idx < min($(n_cols), (col_block_idx + $(1)) * $(BLOCK_SIZE))) {
      logits_label = tl.load(logits_ptr + label_idx)
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
      loss += $(lse_square_scale) * lse * lse
    }
  }
  tl.store(loss_ptr + col_block_idx * $(n_rows) + row_idx, loss)
}

/-- The faithful full forward surface lowers to the algorithm layer, including
the smoothing, split, ignored-label, LSE side-store, and LSE-square branches. -/
theorem cross_entropy_fwd_surface_toAlgorithm_supported
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ)
    (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) :
    ∃ alg,
      (cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr
        smoothing lse_square_scale ignored_index total_classes class_start_idx
        n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).toAlgorithm? =
        Except.ok alg := by
  simp [cross_entropy_fwd_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented final-store slice of `cross_entropy1.py`'s
`cross_entropy_bwd_kernel`.

The full kernel computes `probs` from logits/LSE/labels/smoothing. This slice
starts from a precomputed `Probs` row and proves the masked
`dlogits = dloss * probs` writeback. -/
def cross_entropy_bwd_store_slice
    (dlogits_ptr dloss_ptr Probs : RegionName)
    (n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  probs = tl.load(Probs + row_idx * $(probs_row_stride) + col_offsets,
    mask=col_offsets < $(n_cols), other=0.0)
  tl.store(dlogits_ptr + row_idx * $(dlogits_row_stride) + col_offsets,
    dloss * probs, mask=col_offsets < $(n_cols))
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
    (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem dloss_ptr (s.pids 0 * dloss_row_stride) *
    s.readMem Probs (probsOffset s probs_row_stride BLOCK_SIZE i)

theorem cross_entropy_bwd_store_slice_correct
    (dlogits_ptr dloss_ptr Probs : RegionName)
    (n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (cross_entropy_bwd_store_slice dlogits_ptr dloss_ptr Probs
        n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE)
        s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem dlogits_ptr (outOffset s dlogits_row_stride BLOCK_SIZE i) =
        if active s n_cols BLOCK_SIZE i then
          expectedGrad s dloss_ptr Probs dloss_row_stride probs_row_stride
            BLOCK_SIZE i
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
  simp [exec, cross_entropy_bwd_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt] at hExec
  rw [← hExec]
  simp only [outOffset, colOffset, active, expectedGrad, probsOffset]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hInj (i, PUnit.unit)]
  by_cases hActive : s.pids 1 * BLOCK_SIZE + i.val < n_cols
  · simp [hActive]
  · simp [hActive]

theorem cross_entropy_bwd_store_slice_compute_correct
    (dlogits_ptr dloss_ptr Probs : RegionName)
    (n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := cross_entropy_bwd_store_slice dlogits_ptr dloss_ptr Probs
        n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s n_cols BLOCK_SIZE i)
        (fun i => (dlogits_ptr, outOffset s dlogits_row_stride BLOCK_SIZE i)))
      (expected := fun i =>
        expectedGrad s dloss_ptr Probs dloss_row_stride probs_row_stride
          BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [cross_entropy_bwd_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := cross_entropy_bwd_store_slice_correct dlogits_ptr dloss_ptr Probs
    n_cols dlogits_row_stride dloss_row_stride probs_row_stride BLOCK_SIZE
    s s' hExec i
  simpa [hActive] using h

def lseOutOffset (s : BlockState) (n_rows : Nat) : Nat :=
  s.pids 1 * n_rows + s.pids 0

/-- The first ten algorithm statements of `cross_entropy_fwd_surface`: program
ids, the row-offset pointer, the masked block-logits load, the running max, the
optional smoothing sum, the stable LSE, and the LSE store. -/
def ceFwdLsePrefix
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing : ℝ)
    (total_classes : Nat)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING : Bool) : List Stmt :=
  [Stmt.assign TileDType.nat [] "row_idx" (Op.programId 0),
   Stmt.assign TileDType.nat [] "col_block_idx" (Op.programId 1),
   Stmt.assign TileDType.ptr [] "logits_ptr"
     (Op.ptrAdd Broadcast.nil (Op.ptrBase logits_ptr)
       (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "row_idx")
         (Op.constNat logits_row_stride))),
   Stmt.assign TileDType.nat [n + 1] "col_offsets"
     (Op.add NumericDType.nat Broadcast.scalarL
       (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "col_block_idx")
         (Op.constNat (n + 1)))
       (Op.arange (n + 1))),
   Stmt.assign TileDType.int [] "label_idx"
     (Op.load TileDType.int (MemAccess.region labels_ptr (Op.ref TileDType.nat [] "row_idx"))
       MaskOpt.none),
   Stmt.assign TileDType.real [n + 1] "logits"
     (Op.load ComputeDType.fp32.eraseDType
       (MemAccess.ptr
         (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "logits_ptr")
           (Op.ref TileDType.nat [n + 1] "col_offsets")))
       (MaskOpt.maskOther
         (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [n + 1] "col_offsets")
           (Op.constNat n_cols))
         (Op.negInf.broadcast [n + 1]))),
   Stmt.assign TileDType.real [] "max_logits"
     (Op.reduceMax ⟨0, by simp⟩ Bool.false (Op.ref TileDType.real [n + 1] "logits")),
   Stmt.ifThen (Op.constBool HAS_SMOOTHING)
     [Stmt.assign TileDType.real [] "sum_logits"
       (Op.reduceSum ⟨0, by simp⟩ Bool.false
         ((Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [n + 1] "col_offsets")
               (Op.constNat n_cols)).where
           (Op.ref TileDType.real [n + 1] "logits") ((Op.const 0.0).broadcast [n + 1])))],
   Stmt.assign TileDType.real [] "lse"
     (Op.add NumericDType.real Broadcast.nil
       (Op.reduceSum ⟨0, by simp⟩ Bool.false
           (Op.sub NumericDType.real Broadcast.scalarR (Op.ref TileDType.real [n + 1] "logits")
               (Op.ref TileDType.real [] "max_logits")).exp).log
       (Op.ref TileDType.real [] "max_logits")),
   Stmt.store TileDType.real []
     (MemAccess.region lse_ptr
       (Op.add NumericDType.nat Broadcast.nil
         (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "col_block_idx")
           (Op.constNat n_rows))
         (Op.ref TileDType.nat [] "row_idx")))
     (Op.ref TileDType.real [] "lse") MaskOpt.none]

/-- The remaining two algorithm statements of `cross_entropy_fwd_surface`: the
branch-selected `loss` computation (all register assignments, no stores) and the
single store to `loss_ptr`. -/
def ceFwdLossTail
    (loss_ptr logits_ptr : RegionName)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows : Nat) (n : Nat)
    (HAS_SMOOTHING SPLIT : Bool) : List Stmt :=
  [Stmt.ifThenElse
     (Op.eq ComparableDType.int Broadcast.nil (Op.ref TileDType.int [] "label_idx")
       (Op.constInt ignored_index))
     [Stmt.assign TileDType.real [] "loss" (Op.const 0.0)]
     [Stmt.assign TileDType.int [] "label_idx"
         (Op.sub NumericDType.int Broadcast.nil (Op.ref TileDType.int [] "label_idx")
           (Op.constInt class_start_idx)),
       Stmt.ifThenElse
         (Op.boolAnd Broadcast.nil
           (Op.ge ComparableDType.int Broadcast.nil (Op.ref TileDType.int [] "label_idx")
             (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "col_block_idx")
                 (Op.constNat (n + 1))).castNatToInt)
           (Op.lt ComparableDType.int Broadcast.nil (Op.ref TileDType.int [] "label_idx")
             ((Op.lt ComparableDType.nat Broadcast.nil (Op.constNat n_cols)
                     (Op.mul NumericDType.nat Broadcast.nil
                       (Op.add NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "col_block_idx")
                         (Op.constNat 1))
                       (Op.constNat (n + 1)))).where
                 (Op.constNat n_cols)
                 (Op.mul NumericDType.nat Broadcast.nil
                   (Op.add NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "col_block_idx")
                     (Op.constNat 1))
                   (Op.constNat (n + 1)))).castNatToInt))
         [Stmt.assign TileDType.real [] "logits_label"
             (Op.load TileDType.real
               (MemAccess.ptr
                 (Op.ptrAdd Broadcast.nil (Op.ref TileDType.ptr [] "logits_ptr")
                   (Op.ref TileDType.int [] "label_idx").castIntToNat))
               MaskOpt.none),
           Stmt.ifThenElse (Op.constBool HAS_SMOOTHING)
             [Stmt.assign TileDType.real [] "loss"
                 (Op.sub NumericDType.real Broadcast.nil
                   (Op.sub NumericDType.real Broadcast.nil
                     ((Op.constBool SPLIT).boolNot.ite (Op.ref TileDType.real [] "lse") (Op.const 0.0))
                     (Op.div NumericDType.real Broadcast.nil
                       (Op.mul NumericDType.real Broadcast.nil (Op.const smoothing)
                         (Op.ref TileDType.real [] "sum_logits"))
                       (Op.const ↑total_classes)))
                   (Op.mul NumericDType.real Broadcast.nil
                     (Op.sub NumericDType.real Broadcast.nil (Op.const 1.0) (Op.const smoothing))
                     (Op.ref TileDType.real [] "logits_label")))]
             [Stmt.assign TileDType.real [] "loss"
                 (Op.sub NumericDType.real Broadcast.nil
                   ((Op.constBool SPLIT).boolNot.ite (Op.ref TileDType.real [] "lse") (Op.const 0.0))
                   (Op.ref TileDType.real [] "logits_label"))]]
         [Stmt.ifThenElse (Op.constBool HAS_SMOOTHING)
             [Stmt.assign TileDType.real [] "loss"
                 (Op.mul NumericDType.real Broadcast.nil (Op.const smoothing)
                   (Op.sub NumericDType.real Broadcast.nil
                     ((Op.constBool SPLIT).boolNot.ite (Op.ref TileDType.real [] "lse") (Op.const 0.0))
                     (Op.div NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "sum_logits")
                       (Op.const ↑total_classes))))]
             [Stmt.assign TileDType.real [] "loss" (Op.const 0.0)]],
       Stmt.ifThen (Op.constBool SPLIT).boolNot
         [Stmt.assign TileDType.real [] "loss"
             (Op.add NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "loss")
               (Op.mul NumericDType.real Broadcast.nil
                 (Op.mul NumericDType.real Broadcast.nil (Op.const lse_square_scale)
                   (Op.ref TileDType.real [] "lse"))
                 (Op.ref TileDType.real [] "lse")))]],
   Stmt.store TileDType.real []
     (MemAccess.region loss_ptr
       (Op.add NumericDType.nat Broadcast.nil
         (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "col_block_idx")
           (Op.constNat n_rows))
         (Op.ref TileDType.nat [] "row_idx")))
     (Op.ref TileDType.real [] "loss") MaskOpt.none]

/-- The full forward algorithm body splits as prefix (through the LSE store) ++
loss tail. -/
theorem ceFwd_body_split
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING SPLIT : Bool) :
    (cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr smoothing
      lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows
      logits_row_stride (n+1) HAS_SMOOTHING SPLIT).toAlgKernel.body =
      ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr smoothing total_classes
        n_cols n_rows logits_row_stride n HAS_SMOOTHING ++
      ceFwdLossTail loss_ptr logits_ptr smoothing lse_square_scale ignored_index
        total_classes class_start_idx n_cols n_rows n HAS_SMOOTHING SPLIT := by
  rfl

/-- Row-logits function for program `row_idx`: position `j` reads INPUT memory
`logits_ptr` at `row_idx * logits_row_stride + j`. The forward kernel's
`col_block_idx`-th tile views the contiguous block `[col_block_idx·BLOCK_SIZE,
(col_block_idx+1)·BLOCK_SIZE)` of this row. -/
noncomputable def rowLogits
    (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride n_cols : Nat) (j : Fin n_cols) : ℝ :=
  s.readMem logits_ptr (s.pids 0 * logits_row_stride + j.val)

/-- The kernel's `sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0))`:
the sum of in-range block logits, read from INPUT memory. Out-of-range lanes
contribute `0`. -/
noncomputable def blockSumLogits
    (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride n_cols : Nat) (n : Nat) : ℝ :=
  ∑ i : Fin (n+1),
    if h : s.pids 1 * (n+1) + i.val < n_cols then
      rowLogits s logits_ptr logits_row_stride n_cols ⟨s.pids 1 * (n+1) + i.val, h⟩
    else 0

/-- The label logit `logits_label = tl.load(logits_ptr + (label_idx -
class_start_idx))`, read from INPUT memory at the shifted label position. -/
noncomputable def labelLogit
    (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride : Nat) (lblShift : Int) : ℝ :=
  s.readMem logits_ptr (s.pids 0 * logits_row_stride + lblShift.toNat)

/-- The label value loaded by the kernel: `label_idx = tl.load(labels_ptr +
row_idx)` from INPUT memory. -/
noncomputable def labelValue (s : BlockState) (labels_ptr : Region .int) : Int :=
  s.readMemValue .int (Region.cast labels_ptr) (s.pids 0)

/-- The genuine `loss` value computed by `cross_entropy_fwd_surface` for program
`(row_idx, col_block_idx)`, a faithful Lean transcription of the kernel's
five-way branch over `label_idx`/in-block/`HAS_SMOOTHING`/`SPLIT`/`lse²`. All
sub-terms read INPUT memory; `lse` is the genuine `partialLSE_full`. -/
noncomputable def crossEntropyLossSpec
    (s : BlockState) (logits_ptr : RegionName) (labelVal : Int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (lse : ℝ) : ℝ :=
  if labelVal = ignored_index then 0 else
    let lblShift : Int := labelVal - class_start_idx
    let lseTerm : ℝ := if SPLIT then 0 else lse
    let sq : ℝ := if SPLIT then 0 else lse_square_scale * lse * lse
    let core : ℝ :=
      if (lblShift ≥ (s.pids 1 * (n+1) : Nat)) ∧
         (lblShift < (min n_cols ((s.pids 1 + 1) * (n+1)) : Nat)) then
        if HAS_SMOOTHING then
          lseTerm - smoothing * blockSumLogits s logits_ptr logits_row_stride n_cols n
            / total_classes
            - (1 - smoothing) * labelLogit s logits_ptr logits_row_stride lblShift
        else
          lseTerm - labelLogit s logits_ptr logits_row_stride lblShift
      else
        if HAS_SMOOTHING then
          smoothing * (lseTerm - blockSumLogits s logits_ptr logits_row_stride n_cols n
            / total_classes)
        else 0
    core + sq

/-- **Prefix correctness.** Executing the LSE-computing prefix writes the genuine
masked-lane stable log-sum-exp of the block logits (read from INPUT memory) to
`lse_ptr` at offset `col_block_idx * n_rows + row_idx`. Mirrors
`logsumexp_fwd_kernel_correct_full`. -/
theorem ceFwdLsePrefix_correct
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing : ℝ)
    (total_classes : Nat)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING : Bool)
    (s : BlockState)
    (h_tail : s.pids 1 * (n+1) < n_cols) :
    (stepStmts (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr smoothing
        total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING) s).map
        (·.readMem lse_ptr (lseOutOffset s n_rows)) =
      some (partialLSE_full (n := n)
        (rowLogits s logits_ptr logits_row_stride n_cols) (s.pids 1) h_tail
        Bool.false 0) := by
  have h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty := Finset.univ_nonempty
  have h_filter : (validLanes n n_cols (s.pids 1)).Nonempty := validLanes_nonempty h_tail
  -- Raw readMem-based lane function (definitionally equal to s.readMem logits_ptr ...)
  let rm : Fin (n+1) → ℝ := fun i =>
    s.readMem logits_ptr (s.pids 0 * logits_row_stride + (s.pids 1 * (n+1) + i.val))
  have h_fold : ∀ i : Fin (n+1),
      s.readMem logits_ptr (s.pids 0 * logits_row_stride + (s.pids 1 * (n+1) + ↑i)) = rm i :=
    fun _ => rfl
  have h_rm : ∀ (i : Fin (n+1)) (hi : s.pids 1 * (n+1) + i.val < n_cols),
      rm i = rowLogits s logits_ptr logits_row_stride n_cols
        ⟨s.pids 1 * (n+1) + i.val, hi⟩ := by
    intro i hi; rfl
  unfold ceFwdLsePrefix
  cases HAS_SMOOTHING <;>
  · simp [stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop,
      Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt, lseOutOffset]
    simp only [← Int.natCast_one, ← Int.natCast_add, ← Int.natCast_mul,
      ← Int.natCast_ediv, Int.toNat_natCast, Int.ofNat_lt, if_true, h_fold]
    erw [sup'_masked_eq h_ne h_filter rm, sum_exp_masked_eq rm]
    simp only [WithBot.realLog_coe]
    erw [Option.map₂_coe_coe, WithBot.unbotD_coe]
    rw [add_comm]
    unfold partialLSE_full scaledLane_full validLanes
    simp only [Bool.false_eq_true, reduceIte]
    have h_m_eq : (validLanes n n_cols (s.pids 1)).sup' h_filter rm =
        (validLanes n n_cols (s.pids 1)).sup' h_filter
          (fun x => if h : s.pids 1 * (n+1) + x.val < n_cols then
            rowLogits s logits_ptr logits_row_stride n_cols ⟨s.pids 1 * (n+1) + x.val, h⟩
            else 0) := by
      apply Finset.sup'_congr h_filter rfl
      intro i hi
      simp only [validLanes, Finset.mem_filter, Finset.mem_univ, true_and] at hi
      simp only [dif_pos hi]; exact h_rm i hi
    congr 1
    congr 1
    erw [h_m_eq]
    apply Finset.sum_congr rfl
    intro i hi
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hi
    congr 1
    simp only [dif_pos hi]
    rw [h_rm i hi]; rfl

/-- Mid-state register characterization: after the LSE prefix executes, every
register read by the loss tail holds its genuine value — the program ids, the
row-offset pointer, the loaded label, the genuine `lse = some (partialLSE_full)`,
and (under `HAS_SMOOTHING`) the genuine `sum_logits = some (blockSumLogits)`. -/
theorem ceFwdLsePrefix_regs
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing : ℝ)
    (total_classes : Nat)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING : Bool)
    (s smid : BlockState)
    (h_tail : s.pids 1 * (n+1) < n_cols)
    (hpre : stepStmts (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING) s
      = some smid) :
    smid.regs TileDType.nat [] "row_idx" = some (Tile.scalar (s.pids 0)) ∧
    smid.regs TileDType.nat [] "col_block_idx" = some (Tile.scalar (s.pids 1)) ∧
    smid.regs TileDType.ptr [] "logits_ptr" =
      some (Tile.scalar ((Region.cast logits_ptr : RegionName),
        s.pids 0 * logits_row_stride)) ∧
    smid.regs TileDType.int [] "label_idx" =
      some (Tile.scalar (labelValue s labels_ptr)) ∧
    smid.regs TileDType.real [] "lse" =
      some (Tile.scalar (some (partialLSE_full (n := n)
        (rowLogits s logits_ptr logits_row_stride n_cols) (s.pids 1) h_tail
        Bool.false 0))) ∧
    (HAS_SMOOTHING = Bool.true →
      smid.regs TileDType.real [] "sum_logits" =
        some (Tile.scalar (some (blockSumLogits s logits_ptr logits_row_stride
          n_cols n)))) := by
  have h_ne : (Finset.univ : Finset (Fin (n+1))).Nonempty := Finset.univ_nonempty
  have h_filter : (validLanes n n_cols (s.pids 1)).Nonempty := validLanes_nonempty h_tail
  let rm : Fin (n+1) → ℝ := fun i =>
    s.readMem logits_ptr (s.pids 0 * logits_row_stride + (s.pids 1 * (n+1) + i.val))
  have h_rm : ∀ (i : Fin (n+1)) (hi : s.pids 1 * (n+1) + i.val < n_cols),
      rm i = rowLogits s logits_ptr logits_row_stride n_cols
        ⟨s.pids 1 * (n+1) + i.val, hi⟩ := fun _ _ => rfl
  -- The `lse` register carrier equals `some (partialLSE_full …)`.
  have h_lse_carrier :
      Option.map₂ (fun x1 x2 : ℝ => x1 + x2)
        (WithBot.realLog
          (∑ x : Fin (n+1), WithBot.realExp
            (Option.map₂ (fun x1 x2 : ℝ => x1 - x2)
              (if s.pids 1 * (n+1) + x.val < n_cols then ((some (rm x) : WithBot ℝ)) else none)
              (@Finset.sup' (WithBot ℝ) (Fin (n+1)) _ Finset.univ h_ne
                (fun x => if s.pids 1 * (n+1) + x.val < n_cols then ((some (rm x) : WithBot ℝ)) else none)))))
        (@Finset.sup' (WithBot ℝ) (Fin (n+1)) _ Finset.univ h_ne
          (fun x => if s.pids 1 * (n+1) + x.val < n_cols then ((some (rm x) : WithBot ℝ)) else none))
      = some (partialLSE_full (n := n)
          (rowLogits s logits_ptr logits_row_stride n_cols) (s.pids 1) h_tail
          Bool.false 0) := by
    erw [sup'_masked_eq h_ne h_filter rm, sum_exp_masked_eq rm]
    simp only [WithBot.realLog_coe]
    erw [Option.map₂_coe_coe]
    rw [add_comm]
    unfold partialLSE_full scaledLane_full
    simp only [Bool.false_eq_true, reduceIte]
    have h_m_eq : (validLanes n n_cols (s.pids 1)).sup' h_filter rm =
        (validLanes n n_cols (s.pids 1)).sup' h_filter
          (fun x => if h : s.pids 1 * (n+1) + x.val < n_cols then
            rowLogits s logits_ptr logits_row_stride n_cols ⟨s.pids 1 * (n+1) + x.val, h⟩
            else 0) := by
      apply Finset.sup'_congr h_filter rfl
      intro i hi
      simp only [validLanes, Finset.mem_filter, Finset.mem_univ, true_and] at hi
      simp only [dif_pos hi]; exact h_rm i hi
    have h_sum_eq : ∀ M : ℝ,
        (∑ i ∈ validLanes n n_cols (s.pids 1), Real.exp (rm i - M))
        = ∑ i ∈ validLanes n n_cols (s.pids 1),
            Real.exp ((if h : s.pids 1 * (n+1) + i.val < n_cols then
              rowLogits s logits_ptr logits_row_stride n_cols ⟨s.pids 1 * (n+1) + i.val, h⟩ else 0) - M) := by
      intro M
      apply Finset.sum_congr rfl
      intro i hi
      simp only [validLanes, Finset.mem_filter, Finset.mem_univ, true_and] at hi
      rw [h_rm i hi, dif_pos hi]
    rw [h_m_eq, h_sum_eq]
  -- The `sum_logits` register carrier equals `some (blockSumLogits …)`.
  have h_sum_carrier :
      (∑ x : Fin (n+1),
        ((if s.pids 1 * (n+1) + x.val < n_cols then
          (if s.pids 1 * (n+1) + x.val < n_cols then some (rm x) else none)
        else some (0.0 : ℝ)) : WithBot ℝ))
      = some (blockSumLogits s logits_ptr logits_row_stride n_cols n) := by
    unfold blockSumLogits
    rw [show (some (∑ i : Fin (n+1), if h : s.pids 1 * (n+1) + i.val < n_cols then
          rowLogits s logits_ptr logits_row_stride n_cols ⟨s.pids 1 * (n+1) + i.val, h⟩
          else 0) : WithBot ℝ)
        = ((∑ i : Fin (n+1), if h : s.pids 1 * (n+1) + i.val < n_cols then
          rowLogits s logits_ptr logits_row_stride n_cols ⟨s.pids 1 * (n+1) + i.val, h⟩
          else 0 : ℝ) : WithBot ℝ) from rfl, WithBot.coe_sum]
    apply Finset.sum_congr rfl
    intro i _
    by_cases hi : s.pids 1 * (n+1) + i.val < n_cols
    · simp only [hi, if_true, dif_pos hi]; rw [h_rm i hi]; rfl
    · simp only [hi, if_false, dif_neg hi]; congr 1; norm_num
  revert hpre
  unfold ceFwdLsePrefix
  cases hHS : HAS_SMOOTHING <;>
  · intro hpre
    simp [stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop,
      Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt] at hpre
    subst hpre
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [BlockState.writeMem_regs]; rfl
    · rw [BlockState.writeMem_regs]; rfl
    · rw [BlockState.writeMem_regs]
      refine Eq.trans rfl ?_; congr 1
      ext j; simp [Tile.ptrAdd, Region.cast]
    · rw [BlockState.writeMem_regs]; rfl
    · rw [BlockState.writeMem_regs]
      exact congrArg (some ∘ Tile.scalar) h_lse_carrier
    · intro hh
      first
        | exact absurd hh (by decide)
        | (rw [BlockState.writeMem_regs];
           exact congrArg (some ∘ Tile.scalar) h_sum_carrier)

/-- **Prefix memory frame.** The LSE-computing prefix has exactly one store (to
`lse_ptr`); every other region's memory is untouched. Hence for any region
`r ≠ lse_ptr`, the post-prefix state `smid` reads `r` exactly as the input `s`.
This is what lets the data-dependent label-logit load in the loss tail (which
runs *after* the LSE store) recover the pristine INPUT logit. -/
theorem ceFwdLsePrefix_mem_of_ne
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing : ℝ) (total_classes : Nat)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING : Bool)
    (s smid : BlockState)
    (hpre : stepStmts (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING) s
      = some smid)
    (r : RegionName) (hr : r ≠ lse_ptr) (o : Nat) :
    smid.mem r o = s.mem r o := by
  -- The prefix = `front ++ [lseStore]`; `front` (everything before the LSE
  -- store) is store-free, and the only store targets `lse_ptr`.
  obtain ⟨s2, hfrontStep, hstoreStep⟩ :
      ∃ s2, stepStmts (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr smoothing
              total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING).dropLast s
            = some s2 ∧
          stepStmts [(ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr smoothing
              total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING).getLast
              (by cases HAS_SMOOTHING <;> simp [ceFwdLsePrefix])] s2 = some smid := by
    have hsplit : ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr smoothing
        total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING =
        (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr smoothing
          total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING).dropLast ++
        [(ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr smoothing
          total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING).getLast
          (by cases HAS_SMOOTHING <;> simp [ceFwdLsePrefix])] := by
      exact (List.dropLast_append_getLast _).symm
    rw [hsplit, stepStmts.append_some_iff] at hpre
    exact hpre
  have hfront : (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr smoothing
      total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING).dropLast.all
      (fun st => storeFree st) = Bool.true := by
    cases HAS_SMOOTHING <;> simp [ceFwdLsePrefix, storeFree]
  have hmem2 : s2.mem = s.mem := storeFree_stepStmts_mem _ s s2 hfront hfrontStep
  have hgetLast : (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr smoothing
      total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING).getLast
      (by cases HAS_SMOOTHING <;> simp [ceFwdLsePrefix]) =
      Stmt.store TileDType.real []
        (MemAccess.region lse_ptr
          (Op.add NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "col_block_idx")
              (Op.constNat n_rows))
            (Op.ref TileDType.nat [] "row_idx")))
        (Op.ref TileDType.real [] "lse") MaskOpt.none := by
    cases HAS_SMOOTHING <;> rfl
  rw [hgetLast] at hstoreStep
  -- Step the single store; it only writes `lse_ptr`.
  simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hstoreStep
  cases hLse : s2.regs TileDType.real [] "lse" with
  | none => rw [hLse] at hstoreStep; simp at hstoreStep
  | some vLse =>
  cases hCol : s2.regs TileDType.nat [] "col_block_idx" with
  | none => rw [hLse, hCol] at hstoreStep; simp at hstoreStep
  | some vCol =>
  cases hRow : s2.regs TileDType.nat [] "row_idx" with
  | none => rw [hLse, hCol, hRow] at hstoreStep; simp at hstoreStep
  | some vRow =>
      rw [hLse, hCol, hRow] at hstoreStep
      simp only [TileShape.allIndices, List.foldl, bind, Option.bind, if_true] at hstoreStep
      obtain rfl := Option.some.inj hstoreStep
      simp only [BlockState.writeMemTyped, BlockState.writeMemAs]
      rw [if_neg, hmem2]
      rintro ⟨hreg, -⟩
      exact hr (by rw [Region.cast_id] at hreg; exact hreg)

/-- **Genuine forward LSE correctness.** Executing the *full* forward surface
(all loss branches included) writes to `lse_ptr` at offset
`col_block_idx * n_rows + row_idx` exactly the masked-lane stable log-sum-exp of
the block of row logits read from INPUT memory `logits_ptr`. The loss tail does
not disturb the LSE cell: its only store targets `loss_ptr ≠ lse_ptr`. -/
theorem cross_entropy_fwd_lse_correct
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (s s' : BlockState)
    (h_tail : s.pids 1 * (n+1) < n_cols)
    (hne : lse_ptr ≠ loss_ptr)
    (hExec : exec (cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing lse_square_scale ignored_index total_classes class_start_idx
      n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING SPLIT) s = some s') :
    s'.readMem lse_ptr (lseOutOffset s n_rows) =
      partialLSE_full (n := n) (rowLogits s logits_ptr logits_row_stride n_cols)
        (s.pids 1) h_tail Bool.false 0 := by
  rw [exec, ceFwd_body_split, stepStmts.append_some_iff] at hExec
  obtain ⟨smid, hpre, hsuf⟩ := hExec
  -- The loss tail preserves `readMem lse_ptr`: its first statement is store-free,
  -- and its final store targets `loss_ptr ≠ lse_ptr`.
  have hframe : s'.readMem lse_ptr (lseOutOffset s n_rows)
      = smid.readMem lse_ptr (lseOutOffset s n_rows) := by
    obtain ⟨ifStmt, storeStmt, heq, hsf, hStoreEq⟩ :
        ∃ (ifStmt storeStmt : Stmt),
          ceFwdLossTail loss_ptr logits_ptr smoothing lse_square_scale ignored_index
            total_classes class_start_idx n_cols n_rows n HAS_SMOOTHING SPLIT =
            [ifStmt, storeStmt] ∧
            storeFree ifStmt = Bool.true ∧
            storeStmt =
              Stmt.store TileDType.real []
                (MemAccess.region loss_ptr
                  (Op.add NumericDType.nat Broadcast.nil
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "col_block_idx")
                      (Op.constNat n_rows))
                    (Op.ref TileDType.nat [] "row_idx")))
                (Op.ref TileDType.real [] "loss") MaskOpt.none :=
      ⟨_, _, rfl, by simp [storeFree], rfl⟩
    rw [heq] at hsuf
    rw [stepStmts] at hsuf
    cases hstep : stepStmt ifStmt smid with
    | none => rw [hstep] at hsuf; simp at hsuf
    | some s2 =>
        rw [hstep] at hsuf
        have hmem2 : s2.mem = smid.mem := storeFree_stepStmt_mem ifStmt smid s2 hsf hstep
        -- `hsuf : stepStmts [storeStmt] s2 = some s'`
        subst hStoreEq
        simp only [stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
          Tile.bop, Tile.ptrAdd, NumericDType.add, NumericDType.mul] at hsuf
        -- Whatever the store offset/value, `s'` only mutates region `loss_ptr ≠ lse_ptr`.
        cases hLoss : s2.regs TileDType.real [] "loss" with
        | none => rw [hLoss] at hsuf; simp at hsuf
        | some vLoss =>
        cases hCol : s2.regs TileDType.nat [] "col_block_idx" with
        | none => rw [hLoss, hCol] at hsuf; simp at hsuf
        | some vCol =>
        cases hRow : s2.regs TileDType.nat [] "row_idx" with
        | none => rw [hLoss, hCol, hRow] at hsuf; simp at hsuf
        | some vRow =>
            rw [hLoss, hCol, hRow] at hsuf
            simp only [TileShape.allIndices, List.foldl, bind, Option.bind,
              if_true] at hsuf
            obtain rfl := Option.some.inj hsuf
            have hwrite : ∀ X,
                (s2.writeMemTyped TileDType.real (Region.cast loss_ptr)
                  (vCol.data PUnit.unit * (Tile.scalar (n_rows : TileCarrier .nat)).data PUnit.unit
                    + vRow.data PUnit.unit) (vLoss.data PUnit.unit)).mem lse_ptr X = s2.mem lse_ptr X := by
              intro X
              rw [BlockState.writeMemTyped]
              simp only [BlockState.writeMemAs]
              apply if_neg
              rintro ⟨hr, -⟩
              exact hne hr
            simp only [BlockState.readMem, hwrite, hmem2]
  rw [hframe]
  -- Prefix: the LSE store writes the genuine masked-lane log-sum-exp.
  have hprefix := ceFwdLsePrefix_correct loss_ptr lse_ptr logits_ptr labels_ptr
    smoothing total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING s h_tail
  rw [hpre, Option.map_some] at hprefix
  exact Option.some.inj hprefix

/-- **Genuine forward loss correctness.** Executing the *full* forward surface
writes to `loss_ptr` at offset `col_block_idx * n_rows + row_idx` exactly the
genuine cross-entropy `crossEntropyLossSpec` (the faithful five-way branch over
`label`/in-block/`HAS_SMOOTHING`/`SPLIT`/`lse²`), with all sub-terms read from
INPUT memory and `lse` the genuine `partialLSE_full`. -/
theorem cross_entropy_fwd_loss_correct
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (s s' : BlockState)
    (h_tail : s.pids 1 * (n+1) < n_cols)
    (hLL : lse_ptr ≠ logits_ptr)
    (hExec : exec (cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing lse_square_scale ignored_index total_classes class_start_idx
      n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING SPLIT) s = some s') :
    s'.readMem loss_ptr (lseOutOffset s n_rows) =
      crossEntropyLossSpec s logits_ptr (labelValue s labels_ptr) smoothing
        lse_square_scale ignored_index total_classes class_start_idx n_cols
        logits_row_stride n HAS_SMOOTHING SPLIT
        (partialLSE_full (n := n) (rowLogits s logits_ptr logits_row_stride n_cols)
          (s.pids 1) h_tail Bool.false 0) := by
  rw [exec, ceFwd_body_split, stepStmts.append_some_iff] at hExec
  obtain ⟨smid, hpre, hsuf⟩ := hExec
  obtain ⟨hrow, hcol, hlp, hlbl, hlse, hsum⟩ :=
    ceFwdLsePrefix_regs loss_ptr lse_ptr logits_ptr labels_ptr smoothing
      total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING s smid h_tail hpre
  simp only [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def, hrow, hcol, hlp,
    hlbl, hlse, Option.bind, Tile.scalar, Tile.bop, Tile.cop, Tile.uop, Tile.select,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.eq, ComparableDType.ge, ComparableDType.lt, Broadcast.nil,
    BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_ne_dtype,
    BlockState.readMemValue_real] at hsuf
  unfold crossEntropyLossSpec
  cases hHS : HAS_SMOOTHING <;> cases hSP : SPLIT <;>
    simp only [hHS, hSP, Bool.not_true, Bool.not_false, if_true, if_false,
      Bool.false_eq_true, reduceIte, Bool.true_eq_false] at hsuf ⊢
  · simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def,
      hrow, hcol, hlp, hlse, Tile.scalar,
      BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
      reduceCtorEq, not_false_eq_true, String.reduceEq, reduceIte] at hsuf
    by_cases hIgn : labelValue s labels_ptr = ignored_index
    · simp only [hIgn, decide_true, if_true,
        BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
        reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, Tile.scalar,
        Option.bind_some, TileShape.allIndices, List.foldl, Option.bind] at hsuf
      obtain rfl := Option.some.inj hsuf
      rw [if_pos hIgn]
      simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
        BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
        Region.cast_id, and_true, if_pos]
      rw [show (some (0.0:ℝ)) = ((0.0:ℝ) : WithBot ℝ) from rfl, WithBot.unbotD_coe]
      norm_num
    · simp only [hIgn, decide_false, Bool.false_eq_true, if_false] at hsuf ⊢
      have hmin : (if decide (n_cols < (s.pids 1 + 1) * (n + 1)) = «true» then n_cols
          else (s.pids 1 + 1) * (n + 1)) = min n_cols ((s.pids 1 + 1) * (n + 1)) := by
        by_cases hb : n_cols < (s.pids 1 + 1) * (n + 1)
        · simp only [hb, decide_true, if_true]; omega
        · simp only [hb, decide_false, Bool.false_eq_true, if_false]; omega
      simp only [hmin] at hsuf
      by_cases hInb : labelValue s labels_ptr - class_start_idx ≥ (↑(s.pids 1 * (n + 1)) : Int) ∧
          labelValue s labels_ptr - class_start_idx < (↑(min n_cols ((s.pids 1 + 1) * (n + 1))) : Int)
      · have hbool : (decide (labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))) &&
            decide (labelValue s labels_ptr - class_start_idx <
              Int.ofNat (min n_cols ((s.pids 1 + 1) * (n + 1))))) = «true» := by
          simp only [Bool.and_eq_true, decide_eq_true_eq]
          exact ⟨hInb.1, hInb.2⟩
        rw [if_pos hInb]
        simp only [hbool, if_true,
          Bool.not_false,
          BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
          reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, hlse, Tile.scalar,
          Option.bind_eq_bind, Option.bind_some, TileShape.allIndices, List.foldl] at hsuf
        obtain rfl := Option.some.inj hsuf
        simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
          BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
          Region.cast_id, and_true, if_true]
        have hLLframe : smid.readMem logits_ptr
            (s.pids 0 * logits_row_stride + Int.toNat (labelValue s labels_ptr - class_start_idx))
            = s.readMem logits_ptr
            (s.pids 0 * logits_row_stride + Int.toNat (labelValue s labels_ptr - class_start_idx)) := by
          unfold BlockState.readMem
          rw [ceFwdLsePrefix_mem_of_ne loss_ptr lse_ptr logits_ptr labels_ptr smoothing
            total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING s smid hpre
            logits_ptr (Ne.symm hLL) _]
        rw [hLLframe]
        simp only [WithBot.realSub, WithBot.realMul, WithBot.realAdd, Option.map₂_some_some,
          labelLogit]
        rw [show ∀ x : ℝ, WithBot.unbotD 0 (some x) = x from fun x => rfl]
      · have hbool : (decide (labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))) &&
            decide (labelValue s labels_ptr - class_start_idx <
              Int.ofNat (min n_cols ((s.pids 1 + 1) * (n + 1))))) = «false» := by
          rw [Bool.and_eq_false_iff]
          by_cases h1 : labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))
          · right; simp only [decide_eq_false_iff_not, not_lt]
            rcases not_and_or.mp hInb with h | h
            · exact absurd h1 h
            · exact not_lt.mp h
          · left; simp only [decide_eq_false_iff_not]; exact h1
        rw [if_neg hInb]
        simp only [hbool, Bool.false_eq_true, if_false, if_true, reduceIte,
          Bool.not_false,
          BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
          reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, hlse, Tile.scalar,
          Option.bind_eq_bind, Option.bind_some, TileShape.allIndices, List.foldl] at hsuf
        obtain rfl := Option.some.inj hsuf
        simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
          BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
          Region.cast_id, and_true, if_true]
        simp only [WithBot.realSub, WithBot.realMul, WithBot.realAdd, Option.map₂_some_some]
        rw [show ∀ x : ℝ, WithBot.unbotD 0 (some x) = x from fun x => rfl]
        norm_num
  -- Leaf (HAS_SMOOTHING = false, SPLIT = true): lseTerm = 0, sq = 0.
  · simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def,
      hrow, hcol, hlp, hlse, Tile.scalar,
      BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
      reduceCtorEq, not_false_eq_true, String.reduceEq, reduceIte] at hsuf
    by_cases hIgn : labelValue s labels_ptr = ignored_index
    · simp only [hIgn, decide_true, if_true,
        BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
        reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, Tile.scalar,
        Option.bind_some, TileShape.allIndices, List.foldl, Option.bind] at hsuf
      obtain rfl := Option.some.inj hsuf
      rw [if_pos hIgn]
      simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
        BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
        Region.cast_id, and_true, if_pos]
      rw [show (some (0.0:ℝ)) = ((0.0:ℝ) : WithBot ℝ) from rfl, WithBot.unbotD_coe]
      norm_num
    · simp only [hIgn, decide_false, Bool.false_eq_true, if_false] at hsuf ⊢
      have hmin : (if decide (n_cols < (s.pids 1 + 1) * (n + 1)) = «true» then n_cols
          else (s.pids 1 + 1) * (n + 1)) = min n_cols ((s.pids 1 + 1) * (n + 1)) := by
        by_cases hb : n_cols < (s.pids 1 + 1) * (n + 1)
        · simp only [hb, decide_true, if_true]; omega
        · simp only [hb, decide_false, Bool.false_eq_true, if_false]; omega
      simp only [hmin] at hsuf
      by_cases hInb : labelValue s labels_ptr - class_start_idx ≥ (↑(s.pids 1 * (n + 1)) : Int) ∧
          labelValue s labels_ptr - class_start_idx < (↑(min n_cols ((s.pids 1 + 1) * (n + 1))) : Int)
      · have hbool : (decide (labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))) &&
            decide (labelValue s labels_ptr - class_start_idx <
              Int.ofNat (min n_cols ((s.pids 1 + 1) * (n + 1))))) = «true» := by
          simp only [Bool.and_eq_true, decide_eq_true_eq]
          exact ⟨hInb.1, hInb.2⟩
        rw [if_pos hInb]
        simp only [hbool, if_true, if_false, reduceIte,
          Bool.not_true,
          BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
          reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, hlse, Tile.scalar,
          Option.bind_eq_bind, Option.bind_some, TileShape.allIndices, List.foldl] at hsuf
        obtain rfl := Option.some.inj hsuf
        simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
          BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
          Region.cast_id, and_true, if_true]
        have hLLframe : smid.readMem logits_ptr
            (s.pids 0 * logits_row_stride + Int.toNat (labelValue s labels_ptr - class_start_idx))
            = s.readMem logits_ptr
            (s.pids 0 * logits_row_stride + Int.toNat (labelValue s labels_ptr - class_start_idx)) := by
          unfold BlockState.readMem
          rw [ceFwdLsePrefix_mem_of_ne loss_ptr lse_ptr logits_ptr labels_ptr smoothing
            total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING s smid hpre
            logits_ptr (Ne.symm hLL) _]
        rw [hLLframe]
        simp only [WithBot.realSub, WithBot.realMul, WithBot.realAdd, Option.map₂_some_some,
          labelLogit]
        rw [show ∀ x : ℝ, WithBot.unbotD 0 (some x) = x from fun x => rfl]
        norm_num
      · have hbool : (decide (labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))) &&
            decide (labelValue s labels_ptr - class_start_idx <
              Int.ofNat (min n_cols ((s.pids 1 + 1) * (n + 1))))) = «false» := by
          rw [Bool.and_eq_false_iff]
          by_cases h1 : labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))
          · right; simp only [decide_eq_false_iff_not, not_lt]
            rcases not_and_or.mp hInb with h | h
            · exact absurd h1 h
            · exact not_lt.mp h
          · left; simp only [decide_eq_false_iff_not]; exact h1
        rw [if_neg hInb]
        simp only [hbool, Bool.false_eq_true, if_false, if_true, reduceIte,
          Bool.not_true,
          BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
          reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, hlse, Tile.scalar,
          Option.bind_eq_bind, Option.bind_some, TileShape.allIndices, List.foldl] at hsuf
        obtain rfl := Option.some.inj hsuf
        simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
          BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
          Region.cast_id, and_true, if_true]
        rw [show (some (0.0:ℝ)) = ((0.0:ℝ) : WithBot ℝ) from rfl, WithBot.unbotD_coe]
        norm_num
  -- Leaf (HAS_SMOOTHING = true, SPLIT = false): lseTerm = lse, sq = lse²·scale,
  -- smoothing branch with `sum_logits = blockSumLogits`.
  · have hsum' := hsum hHS
    simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def,
      hrow, hcol, hlp, hlse, hsum', Tile.scalar,
      BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
      reduceCtorEq, not_false_eq_true, String.reduceEq, reduceIte] at hsuf
    by_cases hIgn : labelValue s labels_ptr = ignored_index
    · simp only [hIgn, decide_true, if_true,
        BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
        reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, Tile.scalar,
        Option.bind_some, TileShape.allIndices, List.foldl, Option.bind] at hsuf
      obtain rfl := Option.some.inj hsuf
      rw [if_pos hIgn]
      simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
        BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
        Region.cast_id, and_true, if_pos]
      rw [show (some (0.0:ℝ)) = ((0.0:ℝ) : WithBot ℝ) from rfl, WithBot.unbotD_coe]
      norm_num
    · simp only [hIgn, decide_false, Bool.false_eq_true, if_false] at hsuf ⊢
      have hmin : (if decide (n_cols < (s.pids 1 + 1) * (n + 1)) = «true» then n_cols
          else (s.pids 1 + 1) * (n + 1)) = min n_cols ((s.pids 1 + 1) * (n + 1)) := by
        by_cases hb : n_cols < (s.pids 1 + 1) * (n + 1)
        · simp only [hb, decide_true, if_true]; omega
        · simp only [hb, decide_false, Bool.false_eq_true, if_false]; omega
      simp only [hmin] at hsuf
      by_cases hInb : labelValue s labels_ptr - class_start_idx ≥ (↑(s.pids 1 * (n + 1)) : Int) ∧
          labelValue s labels_ptr - class_start_idx < (↑(min n_cols ((s.pids 1 + 1) * (n + 1))) : Int)
      · have hbool : (decide (labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))) &&
            decide (labelValue s labels_ptr - class_start_idx <
              Int.ofNat (min n_cols ((s.pids 1 + 1) * (n + 1))))) = «true» := by
          simp only [Bool.and_eq_true, decide_eq_true_eq]
          exact ⟨hInb.1, hInb.2⟩
        rw [if_pos hInb]
        simp only [hbool, if_true,
          Bool.not_false,
          BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
          reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, hlse, Tile.scalar,
          Option.bind_eq_bind, Option.bind_some, TileShape.allIndices, List.foldl] at hsuf
        obtain rfl := Option.some.inj hsuf
        simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
          BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
          Region.cast_id, and_true, if_true]
        have hLLframe : smid.readMem logits_ptr
            (s.pids 0 * logits_row_stride + Int.toNat (labelValue s labels_ptr - class_start_idx))
            = s.readMem logits_ptr
            (s.pids 0 * logits_row_stride + Int.toNat (labelValue s labels_ptr - class_start_idx)) := by
          unfold BlockState.readMem
          rw [ceFwdLsePrefix_mem_of_ne loss_ptr lse_ptr logits_ptr labels_ptr smoothing
            total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING s smid hpre
            logits_ptr (Ne.symm hLL) _]
        rw [hLLframe]
        simp only [WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realAdd,
          Option.map₂_some_some, labelLogit]
        rw [show ∀ x : ℝ, WithBot.unbotD 0 (some x) = x from fun x => rfl]
        norm_num
      · have hbool : (decide (labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))) &&
            decide (labelValue s labels_ptr - class_start_idx <
              Int.ofNat (min n_cols ((s.pids 1 + 1) * (n + 1))))) = «false» := by
          rw [Bool.and_eq_false_iff]
          by_cases h1 : labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))
          · right; simp only [decide_eq_false_iff_not, not_lt]
            rcases not_and_or.mp hInb with h | h
            · exact absurd h1 h
            · exact not_lt.mp h
          · left; simp only [decide_eq_false_iff_not]; exact h1
        rw [if_neg hInb]
        simp only [hbool, Bool.false_eq_true, if_false, if_true, reduceIte,
          Bool.not_false,
          BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
          reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, hlse, Tile.scalar,
          Option.bind_eq_bind, Option.bind_some, TileShape.allIndices, List.foldl] at hsuf
        obtain rfl := Option.some.inj hsuf
        simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
          BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
          Region.cast_id, and_true, if_true]
        simp only [WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realAdd,
          Option.map₂_some_some]
        rw [show ∀ x : ℝ, WithBot.unbotD 0 (some x) = x from fun x => rfl]
  -- Leaf (HAS_SMOOTHING = true, SPLIT = true): lseTerm = 0, sq = 0, smoothing branch.
  · have hsum' := hsum hHS
    simp only [Option.bind_eq_bind, Option.bind_some, Option.pure_def,
      hrow, hcol, hlp, hlse, hsum', Tile.scalar,
      BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
      reduceCtorEq, not_false_eq_true, String.reduceEq, reduceIte] at hsuf
    by_cases hIgn : labelValue s labels_ptr = ignored_index
    · simp only [hIgn, decide_true, if_true,
        BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
        reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, Tile.scalar,
        Option.bind_some, TileShape.allIndices, List.foldl, Option.bind] at hsuf
      obtain rfl := Option.some.inj hsuf
      rw [if_pos hIgn]
      simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
        BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
        Region.cast_id, and_true, if_pos]
      rw [show (some (0.0:ℝ)) = ((0.0:ℝ) : WithBot ℝ) from rfl, WithBot.unbotD_coe]
      norm_num
    · simp only [hIgn, decide_false, Bool.false_eq_true, if_false] at hsuf ⊢
      have hmin : (if decide (n_cols < (s.pids 1 + 1) * (n + 1)) = «true» then n_cols
          else (s.pids 1 + 1) * (n + 1)) = min n_cols ((s.pids 1 + 1) * (n + 1)) := by
        by_cases hb : n_cols < (s.pids 1 + 1) * (n + 1)
        · simp only [hb, decide_true, if_true]; omega
        · simp only [hb, decide_false, Bool.false_eq_true, if_false]; omega
      simp only [hmin] at hsuf
      by_cases hInb : labelValue s labels_ptr - class_start_idx ≥ (↑(s.pids 1 * (n + 1)) : Int) ∧
          labelValue s labels_ptr - class_start_idx < (↑(min n_cols ((s.pids 1 + 1) * (n + 1))) : Int)
      · have hbool : (decide (labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))) &&
            decide (labelValue s labels_ptr - class_start_idx <
              Int.ofNat (min n_cols ((s.pids 1 + 1) * (n + 1))))) = «true» := by
          simp only [Bool.and_eq_true, decide_eq_true_eq]
          exact ⟨hInb.1, hInb.2⟩
        rw [if_pos hInb]
        simp only [hbool, if_true, if_false, reduceIte,
          Bool.not_true,
          BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
          reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, hlse, Tile.scalar,
          Option.bind_eq_bind, Option.bind_some, TileShape.allIndices, List.foldl] at hsuf
        obtain rfl := Option.some.inj hsuf
        simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
          BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
          Region.cast_id, and_true, if_true]
        have hLLframe : smid.readMem logits_ptr
            (s.pids 0 * logits_row_stride + Int.toNat (labelValue s labels_ptr - class_start_idx))
            = s.readMem logits_ptr
            (s.pids 0 * logits_row_stride + Int.toNat (labelValue s labels_ptr - class_start_idx)) := by
          unfold BlockState.readMem
          rw [ceFwdLsePrefix_mem_of_ne loss_ptr lse_ptr logits_ptr labels_ptr smoothing
            total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING s smid hpre
            logits_ptr (Ne.symm hLL) _]
        rw [hLLframe]
        simp only [WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realAdd,
          Option.map₂_some_some, labelLogit]
        rw [show ∀ x : ℝ, WithBot.unbotD 0 (some x) = x from fun x => rfl]
        norm_num
      · have hbool : (decide (labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))) &&
            decide (labelValue s labels_ptr - class_start_idx <
              Int.ofNat (min n_cols ((s.pids 1 + 1) * (n + 1))))) = «false» := by
          rw [Bool.and_eq_false_iff]
          by_cases h1 : labelValue s labels_ptr - class_start_idx ≥ Int.ofNat (s.pids 1 * (n + 1))
          · right; simp only [decide_eq_false_iff_not, not_lt]
            rcases not_and_or.mp hInb with h | h
            · exact absurd h1 h
            · exact not_lt.mp h
          · left; simp only [decide_eq_false_iff_not]; exact h1
        rw [if_neg hInb]
        simp only [hbool, Bool.false_eq_true, if_false, if_true, reduceIte,
          Bool.not_true,
          BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_ne_dtype, ne_eq,
          reduceCtorEq, not_false_eq_true, String.reduceEq, hcol, hrow, hlse, Tile.scalar,
          Option.bind_eq_bind, Option.bind_some, TileShape.allIndices, List.foldl] at hsuf
        obtain rfl := Option.some.inj hsuf
        simp only [BlockState.writeMemTyped_real, BlockState.setReg_readMem,
          BlockState.writeMem_readMem, lseOutOffset, FloatDType.real_storeValue,
          Region.cast_id, and_true, if_true]
        simp only [WithBot.realSub, WithBot.realMul, WithBot.realDiv, WithBot.realAdd,
          Option.map₂_some_some]
        rw [show ∀ x : ℝ, WithBot.unbotD 0 (some x) = x from fun x => rfl]
        norm_num

/-! ## Bridge to the canonical pure cross-entropy math

In the base regime — a single column block exactly spanning the vocabulary
(`col_block_idx = 0`, `BLOCK_SIZE = n_cols = n+1`, so every class is in the
block), no split (`SPLIT = false`), no `lse²` term (`lse_square_scale = 0`),
the label active and in range, and `total_classes = n_cols` — the kernel's
genuine five-way `crossEntropyLossSpec` collapses to the shared pure
`crossEntropyLoss` (no smoothing) / `crossEntropyLossSmoothed` (with smoothing)
from `VeriTile.Triton.Math.Loss`. -/

open VeriTile.Triton.TiledLoss in
/-- In the full-vocab single-block regime (`s.pids 1 = 0`, `n_cols = n+1`) the
kernel's `blockSumLogits` is the plain total `∑ rowLogits`. -/
theorem blockSumLogits_full_eq_sum
    (s : BlockState) (logits_ptr : RegionName) (logits_row_stride : Nat) (n : Nat)
    (h0 : s.pids 1 = 0) :
    blockSumLogits s logits_ptr logits_row_stride (n+1) n
      = ∑ i : Fin (n+1), rowLogits s logits_ptr logits_row_stride (n+1) i := by
  unfold blockSumLogits rowLogits
  apply Finset.sum_congr rfl
  intro i _
  have hi : s.pids 1 * (n+1) + i.val < n+1 := by rw [h0]; simp [i.isLt]
  rw [dif_pos hi]
  simp only [h0, Nat.zero_mul, Nat.zero_add]

open VeriTile.Triton.TiledLoss in
/-- In the full-vocab single-block regime the kernel's `partialLSE_full` is the
canonical `stableLSE` of the row logits. -/
theorem partialLSE_full_full_eq_stableLSE
    (s : BlockState) (logits_ptr : RegionName) (logits_row_stride : Nat) (n : Nat)
    (h_tail : s.pids 1 * (n+1) < (n+1))
    (h0 : s.pids 1 = 0) :
    partialLSE_full (n := n) (rowLogits s logits_ptr logits_row_stride (n+1))
        (s.pids 1) h_tail Bool.false 0
      = stableLSE (rowLogits s logits_ptr logits_row_stride (n+1)) (Nat.succ_pos n)
          Bool.false 0 := by
  -- replace `s.pids 1` by `0` while transporting the dependent `h_tail`
  rw [show partialLSE_full (n := n) (rowLogits s logits_ptr logits_row_stride (n+1))
        (s.pids 1) h_tail Bool.false 0
      = partialLSE_full (n := n) (rowLogits s logits_ptr logits_row_stride (n+1))
        0 (by rw [h0] at h_tail; exact h_tail) Bool.false 0 from by
    congr 1 <;> simp [h0]]
  exact partialLSE_full_zero_self_eq_stableLSE _ _ _

open VeriTile.Triton.TiledLoss in
/-- In the full-vocab single-block regime the kernel's `labelLogit` is the row
logit at the target class `t` (whose underlying offset is `lblShift.toNat`). -/
theorem labelLogit_eq_rowLogits
    (s : BlockState) (logits_ptr : RegionName) (logits_row_stride : Nat) (n : Nat)
    (lblShift : Int) (t : Fin (n+1)) (ht : (t : Nat) = lblShift.toNat) :
    labelLogit s logits_ptr logits_row_stride lblShift
      = rowLogits s logits_ptr logits_row_stride (n+1) t := by
  unfold labelLogit rowLogits
  rw [ht]

open VeriTile.Triton.TiledLoss in
/-- **Bridge: textbook cross-entropy (no smoothing).** In the base regime
(`HAS_SMOOTHING = false`, `SPLIT = false`, `lse_square_scale = 0`,
`s.pids 1 = 0`, `n_cols = n+1`, `class_start_idx = 0`, label active and in
range as `t : Fin (n+1)`), the kernel's genuine `crossEntropyLossSpec` with
`lse = partialLSE_full` equals the canonical pure `crossEntropyLoss` of the row
logits at `t`. -/
theorem crossEntropyLossSpec_eq_crossEntropyLoss
    (s : BlockState) (logits_ptr : RegionName)
    (labelVal : Int) (smoothing : ℝ) (ignored_index : Int)
    (total_classes : Nat) (n : Nat)
    (h_tail : s.pids 1 * (n+1) < (n+1))
    (h0 : s.pids 1 = 0)
    (t : Fin (n+1))
    (hNotIgn : labelVal ≠ ignored_index)
    (hShift : (t : Nat) = (labelVal - 0).toNat)
    (hNonneg : 0 ≤ labelVal - 0) :
    crossEntropyLossSpec s logits_ptr labelVal smoothing 0 ignored_index
        total_classes 0 (n+1) 0 n Bool.false Bool.false
        (partialLSE_full (n := n) (rowLogits s logits_ptr 0 (n+1)) (s.pids 1) h_tail
          Bool.false 0)
      = crossEntropyLoss (rowLogits s logits_ptr 0 (n+1)) t (Nat.succ_pos n) := by
  unfold crossEntropyLossSpec crossEntropyLoss
  rw [if_neg hNotIgn]
  -- in-block condition holds
  have hin : (labelVal - 0 ≥ (s.pids 1 * (n+1) : Nat)) ∧
      (labelVal - 0 < (min (n+1) ((s.pids 1 + 1) * (n+1)) : Nat)) := by
    constructor
    · rw [h0]; simpa using hNonneg
    · rw [h0]; simp only [Nat.zero_add, Nat.one_mul, Nat.min_self]
      have htlt : (t : Nat) < n+1 := t.isLt
      rw [hShift] at htlt
      have : (labelVal - 0) = ((labelVal - 0).toNat : Int) := (Int.toNat_of_nonneg hNonneg).symm
      rw [this]; exact_mod_cast htlt
  simp only [Bool.false_eq_true, reduceIte, if_pos hin]
  rw [partialLSE_full_full_eq_stableLSE s logits_ptr 0 n h_tail h0]
  rw [labelLogit_eq_rowLogits s logits_ptr 0 n (labelVal - 0) t hShift]
  ring

open VeriTile.Triton.TiledLoss in
/-- **Bridge: textbook cross-entropy with label smoothing.** In the smoothed
base regime (`HAS_SMOOTHING = true`, `SPLIT = false`, `lse_square_scale = 0`,
`s.pids 1 = 0`, `n_cols = n+1`, `class_start_idx = 0`, `total_classes = n+1`,
label active and in range as `t : Fin (n+1)`), the kernel's genuine
`crossEntropyLossSpec` with `lse = partialLSE_full` equals the canonical pure
`crossEntropyLossSmoothed` of the row logits at `t` with smoothing strength
`smoothing`. -/
theorem crossEntropyLossSpec_eq_crossEntropyLossSmoothed
    (s : BlockState) (logits_ptr : RegionName)
    (labelVal : Int) (smoothing : ℝ) (ignored_index : Int)
    (n : Nat)
    (h_tail : s.pids 1 * (n+1) < (n+1))
    (h0 : s.pids 1 = 0)
    (t : Fin (n+1))
    (hNotIgn : labelVal ≠ ignored_index)
    (hShift : (t : Nat) = (labelVal - 0).toNat)
    (hNonneg : 0 ≤ labelVal - 0) :
    crossEntropyLossSpec s logits_ptr labelVal smoothing 0 ignored_index
        (n+1) 0 (n+1) 0 n Bool.true Bool.false
        (partialLSE_full (n := n) (rowLogits s logits_ptr 0 (n+1)) (s.pids 1) h_tail
          Bool.false 0)
      = crossEntropyLossSmoothed (rowLogits s logits_ptr 0 (n+1)) t smoothing
          (Nat.succ_pos n) := by
  unfold crossEntropyLossSpec crossEntropyLossSmoothed
  rw [if_neg hNotIgn]
  have hin : (labelVal - 0 ≥ (s.pids 1 * (n+1) : Nat)) ∧
      (labelVal - 0 < (min (n+1) ((s.pids 1 + 1) * (n+1)) : Nat)) := by
    constructor
    · rw [h0]; simpa using hNonneg
    · rw [h0]; simp only [Nat.zero_add, Nat.one_mul, Nat.min_self]
      have htlt : (t : Nat) < n+1 := t.isLt
      rw [hShift] at htlt
      have : (labelVal - 0) = ((labelVal - 0).toNat : Int) := (Int.toNat_of_nonneg hNonneg).symm
      rw [this]; exact_mod_cast htlt
  simp only [if_true, reduceIte, if_pos hin]
  rw [partialLSE_full_full_eq_stableLSE s logits_ptr 0 n h_tail h0]
  rw [labelLogit_eq_rowLogits s logits_ptr 0 n (labelVal - 0) t hShift]
  rw [blockSumLogits_full_eq_sum s logits_ptr 0 n h0]
  push_cast
  ring



/-! ## The `⊨` specification (forward)

The headline states the forward kernel on the metadata-genre IO skin
`MetaGatherMasked2DKernelIO₂ₓ₂`: the loaded label is a named ghost binder pinned to
the `labels_ptr` slot cell, the masked logits row and the label-gated gather
cell are the two data inputs, and the loss/LSE cells are the two 1-lane
outputs. The machinery below supplies the three intro obligations: the
prefix/tail termination + cell-frame walks (`_isSome`/`_run_facts`/`_run`),
the ⊥-path fallback (programs past the row end store `0` to both cells), and
the per-execution safety walks. -/

/-- Pure block log-sum-exp over the active lanes (`pid₁·B + i < n_cols`) of a
`B`-lane block: the plain (shift-free) form `log (∑ exp xᵢ)`; the stable
kernel form `partialLSE_full` collapses to it via
`partialLSE_full_eq_blockLSE`. -/
noncomputable def ceBlockLSE (n_cols B pid₁ : Nat) (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ i ∈ Finset.univ.filter (fun i : Fin B => pid₁ * B + i.val < n_cols),
    Real.exp (xs i))

/-- Pure masked block sum: the kernel's
`sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0))` over the
pinned block values. -/
noncomputable def ceBlockSum (n_cols B pid₁ : Nat) (xs : Fin B → ℝ) : ℝ :=
  ∑ i : Fin B, if pid₁ * B + i.val < n_cols then xs i else 0

/-- The kernel's five-way loss, as a pure function of the pinned inputs: the
loaded label `lab`, the block values `xs`, and the gather cell `g` (the label
logit, meaningful exactly on the in-block branch that reads it). Mirrors
`crossEntropyLossSpec` with every memory read replaced by its pinned value. -/
noncomputable def ceLossLocal (n_cols total_classes B : Nat)
    (smoothing lse_square_scale : ℝ) (ignored_index class_start_idx : Int)
    (HAS_SMOOTHING SPLIT : Bool)
    (pid₁ : Nat) (lab : Int) (xs : Fin B → ℝ) (g : ℝ) : ℝ :=
  if lab = ignored_index then 0 else
    let lblShift : Int := lab - class_start_idx
    let lse : ℝ := ceBlockLSE n_cols B pid₁ xs
    let lseTerm : ℝ := if SPLIT then 0 else lse
    let sq : ℝ := if SPLIT then 0 else lse_square_scale * lse * lse
    let core : ℝ :=
      if (lblShift ≥ (pid₁ * B : Nat)) ∧
         (lblShift < (min n_cols ((pid₁ + 1) * B) : Nat)) then
        if HAS_SMOOTHING then
          lseTerm - smoothing * ceBlockSum n_cols B pid₁ xs / total_classes
            - (1 - smoothing) * g
        else
          lseTerm - g
      else
        if HAS_SMOOTHING then
          smoothing * (lseTerm - ceBlockSum n_cols B pid₁ xs / total_classes)
        else 0
    core + sq

/-- The stable kernel-form block LSE equals the pure `ceBlockLSE` of any tile
`xs` agreeing with the row on the active lanes. -/
private theorem partialLSE_full_eq_ceBlockLSE
    {n_cols : Nat} (n pid₁ : Nat) (h_tail : pid₁ * (n+1) < n_cols)
    (xsRow : Fin n_cols → ℝ) (xs : Fin (n+1) → ℝ)
    (h : ∀ (j : Fin (n+1)) (hj : pid₁ * (n+1) + j.val < n_cols),
      xsRow ⟨pid₁ * (n+1) + j.val, hj⟩ = xs j) :
    partialLSE_full xsRow pid₁ h_tail Bool.false 0
      = ceBlockLSE n_cols (n+1) pid₁ xs := by
  rw [partialLSE_full_eq_blockLSE]
  unfold TiledLogSumExp.blockLSE ceBlockLSE
  congr 1
  apply Finset.sum_congr
  · rfl
  · intro i hi
    have hmem : pid₁ * (n+1) + i.val < n_cols := (Finset.mem_filter.mp hi).2
    unfold scaledLane_full
    simp only [dif_pos hmem, h i hmem, Bool.false_eq_true, reduceIte]

/-- The kernel's masked block sum equals the pure `ceBlockSum` of the pinned
block values. -/
private theorem blockSumLogits_eq_ceBlockSum
    (s : BlockState) (logits_ptr : RegionName)
    (logits_row_stride n_cols n : Nat) (xs : Fin (n+1) → ℝ)
    (hx : ∀ j : Fin (n+1), s.pids 1 * (n+1) + j.val < n_cols →
      s.readMem logits_ptr
        (s.pids 0 * logits_row_stride + (s.pids 1 * (n+1) + j.val)) = xs j) :
    blockSumLogits s logits_ptr logits_row_stride n_cols n
      = ceBlockSum n_cols (n+1) (s.pids 1) xs := by
  unfold blockSumLogits ceBlockSum
  apply Finset.sum_congr rfl
  intro i _
  by_cases hi : s.pids 1 * (n+1) + i.val < n_cols
  · rw [dif_pos hi, if_pos hi]
    exact hx i hi
  · rw [dif_neg hi, if_neg hi]

/-- The kernel's memory-reading five-way loss equals the pure `ceLossLocal`
of the pinned label, block values, and gather cell. -/
private theorem crossEntropyLossSpec_eq_ceLossLocal
    (s : BlockState) (logits_ptr : RegionName)
    (smoothing lse_square_scale : ℝ) (ignored_index class_start_idx : Int)
    (total_classes n_cols logits_row_stride n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (h_tail : s.pids 1 * (n+1) < n_cols)
    (lab : Int) (xs : Fin (n+1) → ℝ) (g : ℝ)
    (hx : ∀ j : Fin (n+1), s.pids 1 * (n+1) + j.val < n_cols →
      s.readMem logits_ptr
        (s.pids 0 * logits_row_stride + (s.pids 1 * (n+1) + j.val)) = xs j)
    (hg : (lab ≠ ignored_index ∧
        lab - class_start_idx ≥ (↑(s.pids 1 * (n+1)) : Int) ∧
        lab - class_start_idx < (↑(min n_cols ((s.pids 1 + 1) * (n+1))) : Int)) →
      s.readMem logits_ptr
        (s.pids 0 * logits_row_stride + (lab - class_start_idx).toNat) = g) :
    crossEntropyLossSpec s logits_ptr lab smoothing lse_square_scale
        ignored_index total_classes class_start_idx n_cols logits_row_stride n
        HAS_SMOOTHING SPLIT
        (partialLSE_full (n := n) (rowLogits s logits_ptr logits_row_stride n_cols)
          (s.pids 1) h_tail Bool.false 0)
      = ceLossLocal n_cols total_classes (n+1) smoothing lse_square_scale
          ignored_index class_start_idx HAS_SMOOTHING SPLIT (s.pids 1) lab xs g := by
  have hlse := partialLSE_full_eq_ceBlockLSE n (s.pids 1) h_tail
    (rowLogits s logits_ptr logits_row_stride n_cols) xs (fun j hj => hx j hj)
  have hsum := blockSumLogits_eq_ceBlockSum s logits_ptr logits_row_stride
    n_cols n xs hx
  simp only [crossEntropyLossSpec, ceLossLocal]
  by_cases hIgn : lab = ignored_index
  · rw [if_pos hIgn, if_pos hIgn]
  · rw [if_neg hIgn, if_neg hIgn]
    by_cases hInb : (lab - class_start_idx ≥ (↑(s.pids 1 * (n+1)) : Int)) ∧
        (lab - class_start_idx < (↑(min n_cols ((s.pids 1 + 1) * (n+1))) : Int))
    · have hgv : labelLogit s logits_ptr logits_row_stride (lab - class_start_idx)
          = g := hg ⟨hIgn, hInb.1, hInb.2⟩
      rw [if_pos hInb, if_pos hInb, hlse, hsum, hgv]
    · rw [if_neg hInb, if_neg hInb, hlse, hsum]

/-- `TraceSafeList` splits over `++`: prove the front safe, then the back safe
in whatever state the front's execution actually reaches. -/
private theorem traceSafeList_append {bounds : RegionBounds}
    {l1 l2 : List Stmt} {s : BlockState}
    (h1 : Stmt.TraceSafeList bounds l1 s)
    (h2 : ∀ s', stepStmts l1 s = some s' → Stmt.TraceSafeList bounds l2 s') :
    Stmt.TraceSafeList bounds (l1 ++ l2) s := by
  induction l1 generalizing s with
  | nil => exact h2 s (by simp [stepStmts])
  | cons st rest ih =>
      rw [List.cons_append, Stmt.TraceSafeList]
      rw [Stmt.TraceSafeList] at h1
      refine ⟨h1.1, ?_⟩
      cases h : stepStmt st s with
      | none => trivial
      | some s1 =>
          have h1' := h1.2
          simp only [h] at h1'
          refine ih h1' (fun s'' hs'' => h2 s'' ?_)
          simp only [stepStmts, h]
          exact hs''

set_option maxHeartbeats 1600000 in
/-- Termination of the LSE prefix, from any launch state (no valid-lane
hypothesis: the reductions and the masked load are total). -/
private theorem ceFwdLsePrefix_isSome
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing : ℝ) (total_classes : Nat)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING : Bool) (s : BlockState) :
    ∃ smid, stepStmts (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING) s
      = some smid := by
  unfold ceFwdLsePrefix
  cases HAS_SMOOTHING <;>
  · simp [stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop,
      Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt] <;>
    exact ⟨_, rfl⟩

set_option maxHeartbeats 1600000 in
/-- Post-prefix facts, from any launch state: the registers the loss tail
reads (with the `lse`/`sum_logits` carriers existential — the concrete values
belong to `ceFwdLsePrefix_regs` / the `⊥`-path lemma below), and the
cell-level memory frame off the single LSE store cell. -/
private theorem ceFwdLsePrefix_run_facts
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing : ℝ) (total_classes : Nat)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING : Bool) (s smid : BlockState)
    (hpre : stepStmts (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING) s
      = some smid) :
    smid.regs TileDType.nat [] "row_idx" = some (Tile.scalar (s.pids 0)) ∧
    smid.regs TileDType.nat [] "col_block_idx" = some (Tile.scalar (s.pids 1)) ∧
    smid.regs TileDType.ptr [] "logits_ptr" =
      some (Tile.scalar ((Region.cast logits_ptr : RegionName),
        s.pids 0 * logits_row_stride)) ∧
    smid.regs TileDType.int [] "label_idx" =
      some (Tile.scalar (labelValue s labels_ptr)) ∧
    (∃ lseC : WithBot ℝ,
      smid.regs TileDType.real [] "lse" = some (Tile.scalar lseC)) ∧
    (HAS_SMOOTHING = Bool.true → ∃ sumC : WithBot ℝ,
      smid.regs TileDType.real [] "sum_logits" = some (Tile.scalar sumC)) ∧
    (∀ r o, ¬(lse_ptr = r ∧ lseOutOffset s n_rows = o) →
      smid.mem r o = s.mem r o) := by
  revert hpre
  unfold ceFwdLsePrefix
  cases HAS_SMOOTHING <;>
  · intro hpre
    simp [stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop,
      Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt] at hpre
    subst hpre
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [BlockState.writeMem_regs]; rfl
    · rw [BlockState.writeMem_regs]; rfl
    · rw [BlockState.writeMem_regs]
      refine Eq.trans rfl ?_; congr 1
      ext j; simp [Tile.ptrAdd, Region.cast]
    · rw [BlockState.writeMem_regs]; rfl
    · exact ⟨_, by rw [BlockState.writeMem_regs]; rfl⟩
    · first
      | (intro hh; exact absurd hh (by decide))
      | (intro _; exact ⟨_, by rw [BlockState.writeMem_regs]; rfl⟩)
    · intro r o hmiss
      simp only [lseOutOffset] at hmiss
      rw [BlockState.writeMem_mem,
        if_neg (fun hc => hmiss ⟨hc.1.symm, hc.2.symm⟩)]
      simp

set_option maxHeartbeats 1600000 in
/-- **⊥-path prefix facts.** For a program whose block lies entirely past the
row end (no valid lane), the loaded tile is all-`⊥`, so the `lse` register
carrier is `⊥`, the (smoothing) `sum_logits` carrier is `0`, and the unmasked
LSE store writes the IEEE-faithful finite fallback `0`. -/
private theorem ceFwdLsePrefix_run_fallback
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing : ℝ) (total_classes : Nat)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING : Bool) (s smid : BlockState)
    (h_out : ¬ s.pids 1 * (n+1) < n_cols)
    (hpre : stepStmts (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING) s
      = some smid) :
    smid.regs TileDType.real [] "lse" = some (Tile.scalar (none : WithBot ℝ)) ∧
    (HAS_SMOOTHING = Bool.true →
      smid.regs TileDType.real [] "sum_logits"
        = some (Tile.scalar (some (0:ℝ) : WithBot ℝ))) ∧
    smid.readMem lse_ptr (lseOutOffset s n_rows) = 0 := by
  have hno : ∀ i : Fin (n+1), ¬ (s.pids 1 * (n+1) + i.val < n_cols) := fun i hlt =>
    h_out (Nat.lt_of_le_of_lt (Nat.le_add_right _ _) hlt)
  have h_rm : ∀ (i : Fin (n+1)) (hi : s.pids 1 * (n+1) + i.val < n_cols),
      s.readMem logits_ptr
          (s.pids 0 * logits_row_stride + (s.pids 1 * (n+1) + i.val))
        = rowLogits s logits_ptr logits_row_stride n_cols
            ⟨s.pids 1 * (n+1) + i.val, hi⟩ :=
    fun _ _ => rfl
  have hbs : blockSumLogits s logits_ptr logits_row_stride n_cols n = 0 := by
    unfold blockSumLogits
    apply Finset.sum_eq_zero
    intro i _
    rw [dif_neg (hno i)]
  have h_sum_carrier :
      (∑ x : Fin (n+1),
        ((if s.pids 1 * (n+1) + x.val < n_cols then
          (if s.pids 1 * (n+1) + x.val < n_cols then
            some (s.readMem logits_ptr
              (s.pids 0 * logits_row_stride + (s.pids 1 * (n+1) + x.val)))
          else none)
        else some (0.0 : ℝ)) : WithBot ℝ))
      = some (blockSumLogits s logits_ptr logits_row_stride n_cols n) := by
    unfold blockSumLogits
    rw [show (some (∑ i : Fin (n+1), if h : s.pids 1 * (n+1) + i.val < n_cols then
          rowLogits s logits_ptr logits_row_stride n_cols ⟨s.pids 1 * (n+1) + i.val, h⟩
          else 0) : WithBot ℝ)
        = ((∑ i : Fin (n+1), if h : s.pids 1 * (n+1) + i.val < n_cols then
          rowLogits s logits_ptr logits_row_stride n_cols ⟨s.pids 1 * (n+1) + i.val, h⟩
          else 0 : ℝ) : WithBot ℝ) from rfl, WithBot.coe_sum]
    apply Finset.sum_congr rfl
    intro i _
    by_cases hi : s.pids 1 * (n+1) + i.val < n_cols
    · rw [dif_pos hi, if_pos hi, if_pos hi, h_rm i hi]
      rfl
    · simp only [hi, if_false, dif_neg hi]
      congr 1
      norm_num
  revert hpre
  unfold ceFwdLsePrefix
  cases HAS_SMOOTHING <;>
  · intro hpre
    simp [stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.uop, Tile.cop,
      Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt] at hpre
    subst hpre
    refine ⟨?_, ?_, ?_⟩
    · rw [BlockState.writeMem_regs]
      refine Eq.trans rfl ?_; congr 2
      simp [hno, Finset.sup'_const, WithBot.realLog, WithBot.realExp,
        Option.map₂, Real.log_zero] <;>
      rfl
    · first
      | (intro hh; exact absurd hh (by decide))
      | (intro _
         rw [BlockState.writeMem_regs]
         refine Eq.trans rfl ?_
         exact congrArg (some ∘ Tile.scalar)
           (h_sum_carrier.trans (by rw [hbs])))
    · simp only [lseOutOffset, BlockState.writeMem_readMem, and_self, if_pos]
      simp [hno, Finset.sup'_const, WithBot.realLog, WithBot.realExp,
        Option.map₂, Real.log_zero] <;>
      first
        | exact Or.inr rfl
        | rfl

set_option maxHeartbeats 1600000 in
/-- Termination + cell frame of the loss tail, from any post-prefix state
whose relevant registers are pinned (the `lse`/`sum_logits` carriers are
arbitrary: the tail is register arithmetic plus one store). -/
private theorem ceFwdLossTail_run
    (loss_ptr logits_ptr : RegionName)
    (smoothing lse_square_scale : ℝ) (ignored_index class_start_idx : Int)
    (total_classes n_cols n_rows n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (smid : BlockState) (r₀ c₀ poff : Nat) (lv : Int)
    (lseC sumC : WithBot ℝ)
    (hrow : smid.regs TileDType.nat [] "row_idx" = some (Tile.scalar r₀))
    (hcol : smid.regs TileDType.nat [] "col_block_idx" = some (Tile.scalar c₀))
    (hlp : smid.regs TileDType.ptr [] "logits_ptr" =
      some (Tile.scalar ((Region.cast logits_ptr : RegionName), poff)))
    (hlbl : smid.regs TileDType.int [] "label_idx" = some (Tile.scalar lv))
    (hlse : smid.regs TileDType.real [] "lse" = some (Tile.scalar lseC))
    (hsum : HAS_SMOOTHING = Bool.true →
      smid.regs TileDType.real [] "sum_logits" = some (Tile.scalar sumC)) :
    ∃ s2, stepStmts (ceFwdLossTail loss_ptr logits_ptr smoothing lse_square_scale
        ignored_index total_classes class_start_idx n_cols n_rows n
        HAS_SMOOTHING SPLIT) smid = some s2
      ∧ (∀ r o, ¬(loss_ptr = r ∧ c₀ * n_rows + r₀ = o) →
          s2.mem r o = smid.mem r o) := by
  have hmin : (if decide (n_cols < (c₀ + 1) * (n + 1)) = «true» then n_cols
      else (c₀ + 1) * (n + 1)) = min n_cols ((c₀ + 1) * (n + 1)) := by
    by_cases hb : n_cols < (c₀ + 1) * (n + 1)
    · simp only [hb, decide_true, if_true]; omega
    · simp only [hb, decide_false, Bool.false_eq_true, if_false]; omega
  have hmin' : (if n_cols < (c₀ + 1) * (n + 1) then n_cols
      else (c₀ + 1) * (n + 1)) = min n_cols ((c₀ + 1) * (n + 1)) := by
    split <;> omega
  cases hstep : stepStmts (ceFwdLossTail loss_ptr logits_ptr smoothing
      lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows n
      HAS_SMOOTHING SPLIT) smid with
  | none =>
      exfalso
      by_cases hHS : HAS_SMOOTHING = «true» <;>
        by_cases hSP : SPLIT = «true» <;>
        by_cases hIgn : lv = ignored_index <;>
        by_cases hAB : ((c₀ : Int) * ((n : Int) + 1) ≤ lv - class_start_idx ∧
          lv - class_start_idx < (n_cols : Int) ∧
          lv - class_start_idx < ((c₀ : Int) + 1) * ((n : Int) + 1)) <;>
      first
        | (have hsum' := hsum hHS
           simp [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def,
             hrow, hcol, hlp, hlbl, hlse, hsum', hmin, hmin', hHS, hSP, hIgn, hAB,
             Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
             NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
             ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
             TileShape.allIndices] at hstep)
        | simp [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def,
            hrow, hcol, hlp, hlbl, hlse, hmin, hmin', hHS, hSP, hIgn, hAB,
            Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
            TileShape.allIndices] at hstep
  | some s2 =>
      refine ⟨s2, rfl, ?_⟩
      intro r o hmiss
      by_cases hHS : HAS_SMOOTHING = «true» <;>
        by_cases hSP : SPLIT = «true» <;>
        by_cases hIgn : lv = ignored_index <;>
        by_cases hAB : ((c₀ : Int) * ((n : Int) + 1) ≤ lv - class_start_idx ∧
          lv - class_start_idx < (n_cols : Int) ∧
          lv - class_start_idx < ((c₀ : Int) + 1) * ((n : Int) + 1)) <;>
      first
        | (have hsum' := hsum hHS
           simp [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def,
             hrow, hcol, hlp, hlbl, hlse, hsum', hmin, hmin', hHS, hSP, hIgn, hAB,
             Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
             NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
             ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
             TileShape.allIndices] at hstep
           subst hstep
           rw [BlockState.writeMem_mem,
             if_neg (fun hc => hmiss ⟨hc.1.symm, hc.2.symm⟩)]
           simp)
        | (simp [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def,
            hrow, hcol, hlp, hlbl, hlse, hmin, hmin', hHS, hSP, hIgn, hAB,
            Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
            TileShape.allIndices] at hstep
           subst hstep
           rw [BlockState.writeMem_mem,
             if_neg (fun hc => hmiss ⟨hc.1.symm, hc.2.symm⟩)]
           simp)

set_option maxHeartbeats 1600000 in
/-- **⊥-path tail.** With the `lse` carrier `⊥`, the `sum_logits` carrier `0`,
and the label not in the (empty) block window, every branch's `loss` carrier
is `⊥` or `0`, so the unmasked store writes `0`. Termination + the stored `0`
+ cell frame. -/
private theorem ceFwdLossTail_run_fallback
    (loss_ptr logits_ptr : RegionName)
    (smoothing lse_square_scale : ℝ) (ignored_index class_start_idx : Int)
    (total_classes n_cols n_rows n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (smid : BlockState) (r₀ c₀ poff : Nat) (lv : Int)
    (hrow : smid.regs TileDType.nat [] "row_idx" = some (Tile.scalar r₀))
    (hcol : smid.regs TileDType.nat [] "col_block_idx" = some (Tile.scalar c₀))
    (hlp : smid.regs TileDType.ptr [] "logits_ptr" =
      some (Tile.scalar ((Region.cast logits_ptr : RegionName), poff)))
    (hlbl : smid.regs TileDType.int [] "label_idx" = some (Tile.scalar lv))
    (hlse : smid.regs TileDType.real [] "lse" = some (Tile.scalar (none : WithBot ℝ)))
    (hsum : HAS_SMOOTHING = Bool.true →
      smid.regs TileDType.real [] "sum_logits"
        = some (Tile.scalar (some (0:ℝ) : WithBot ℝ)))
    (hnb : ¬((↑(c₀ * (n + 1)) : Int) ≤ lv - class_start_idx ∧
      lv - class_start_idx < (↑(min n_cols ((c₀ + 1) * (n + 1))) : Int))) :
    ∃ s2, stepStmts (ceFwdLossTail loss_ptr logits_ptr smoothing lse_square_scale
        ignored_index total_classes class_start_idx n_cols n_rows n
        HAS_SMOOTHING SPLIT) smid = some s2
      ∧ s2.readMem loss_ptr (c₀ * n_rows + r₀) = 0
      ∧ (∀ r o, ¬(loss_ptr = r ∧ c₀ * n_rows + r₀ = o) →
          s2.mem r o = smid.mem r o) := by
  have hmin : (if decide (n_cols < (c₀ + 1) * (n + 1)) = «true» then n_cols
      else (c₀ + 1) * (n + 1)) = min n_cols ((c₀ + 1) * (n + 1)) := by
    by_cases hb : n_cols < (c₀ + 1) * (n + 1)
    · simp only [hb, decide_true, if_true]; omega
    · simp only [hb, decide_false, Bool.false_eq_true, if_false]; omega
  have hmin' : (if n_cols < (c₀ + 1) * (n + 1) then n_cols
      else (c₀ + 1) * (n + 1)) = min n_cols ((c₀ + 1) * (n + 1)) := by
    split <;> omega
  cases hstep : stepStmts (ceFwdLossTail loss_ptr logits_ptr smoothing
      lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows n
      HAS_SMOOTHING SPLIT) smid with
  | none =>
      exfalso
      by_cases hHS : HAS_SMOOTHING = «true» <;>
        by_cases hSP : SPLIT = «true» <;>
        by_cases hIgn : lv = ignored_index <;>
        by_cases hAB : ((c₀ : Int) * ((n : Int) + 1) ≤ lv - class_start_idx ∧
          lv - class_start_idx < (n_cols : Int) ∧
          lv - class_start_idx < ((c₀ : Int) + 1) * ((n : Int) + 1)) <;>
      first
        | exact (hnb ⟨by push_cast; omega, by push_cast; omega⟩).elim
        | (have hsum' := hsum hHS
           simp [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def,
             hrow, hcol, hlp, hlbl, hlse, hsum', hmin, hmin', hHS, hSP, hIgn, hAB,
             Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
             NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
             ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
             TileShape.allIndices] at hstep)
        | simp [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def,
            hrow, hcol, hlp, hlbl, hlse, hmin, hmin', hHS, hSP, hIgn, hAB,
            Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
            TileShape.allIndices] at hstep
  | some s2 =>
      refine ⟨s2, rfl, ?_, ?_⟩
      · by_cases hHS : HAS_SMOOTHING = «true» <;>
          by_cases hSP : SPLIT = «true» <;>
          by_cases hIgn : lv = ignored_index <;>
          by_cases hAB : ((c₀ : Int) * ((n : Int) + 1) ≤ lv - class_start_idx ∧
            lv - class_start_idx < (n_cols : Int) ∧
            lv - class_start_idx < ((c₀ : Int) + 1) * ((n : Int) + 1)) <;>
        first
          | exact (hnb ⟨by push_cast; omega, by push_cast; omega⟩).elim
          | (have hsum' := hsum hHS
             simp [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def,
               hrow, hcol, hlp, hlbl, hlse, hsum', hmin, hmin', hHS, hSP, hIgn, hAB,
               Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
               NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
               ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
               TileShape.allIndices] at hstep
             subst hstep
             simp [BlockState.writeMem_readMem, Option.map₂, WithBot.unbotD,
               WithBot.recBotCoe]
             try norm_num)
          | (simp [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def,
              hrow, hcol, hlp, hlbl, hlse, hmin, hmin', hHS, hSP, hIgn, hAB,
              Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
              NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
              ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
              TileShape.allIndices] at hstep
             subst hstep
             simp [BlockState.writeMem_readMem, Option.map₂, WithBot.unbotD,
               WithBot.recBotCoe]
             try norm_num)
      · intro r o hmiss
        by_cases hHS : HAS_SMOOTHING = «true» <;>
          by_cases hSP : SPLIT = «true» <;>
          by_cases hIgn : lv = ignored_index <;>
          by_cases hAB : ((c₀ : Int) * ((n : Int) + 1) ≤ lv - class_start_idx ∧
            lv - class_start_idx < (n_cols : Int) ∧
            lv - class_start_idx < ((c₀ : Int) + 1) * ((n : Int) + 1)) <;>
        first
          | exact (hnb ⟨by push_cast; omega, by push_cast; omega⟩).elim
          | (have hsum' := hsum hHS
             simp [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def,
               hrow, hcol, hlp, hlbl, hlse, hsum', hmin, hmin', hHS, hSP, hIgn, hAB,
               Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
               NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
               ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
               TileShape.allIndices] at hstep
             subst hstep
             rw [BlockState.writeMem_mem,
               if_neg (fun hc => hmiss ⟨hc.1.symm, hc.2.symm⟩)]
             simp)
          | (simp [ceFwdLossTail, stepStmts, stepStmt, evalOp.eq_def,
              hrow, hcol, hlp, hlbl, hlse, hmin, hmin', hHS, hSP, hIgn, hAB,
              Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
              NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
              ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
              TileShape.allIndices] at hstep
             subst hstep
             rw [BlockState.writeMem_mem,
               if_neg (fun hc => hmiss ⟨hc.1.symm, hc.2.symm⟩)]
             simp)

set_option maxHeartbeats 3200000 in
/-- Safety walk over the LSE prefix: the label-slot cell, the active lanes of
the masked row load, and the unmasked LSE store cell must be in bounds. -/
private theorem ceFwdLsePrefix_traceSafe
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing : ℝ) (total_classes : Nat)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING : Bool)
    (bounds : RegionBounds) (s : BlockState)
    (hbL : s.pids 0 < bounds (Region.cast labels_ptr))
    (hbr : ∀ j : Fin (n+1), s.pids 1 * (n+1) + j.val < n_cols →
      s.pids 0 * logits_row_stride + (s.pids 1 * (n+1) + j.val)
        < bounds logits_ptr)
    (hbw2 : s.pids 1 * n_rows + s.pids 0 < bounds lse_ptr) :
    Stmt.TraceSafeList bounds
      (ceFwdLsePrefix loss_ptr lse_ptr logits_ptr labels_ptr smoothing
        total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING) s := by
  unfold ceFwdLsePrefix
  cases HAS_SMOOTHING <;>
  · simp (maxSteps := 16000000) [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
      MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
      MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
      BlockState.setReg,
      Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt,
      Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      hbL, hbw2] <;>
    first
      | exact fun a ha => hbr a ha
      | exact ⟨fun a ha => hbr a ha, fun a ha => hbr a ha⟩
      | (refine ⟨fun a ha => hbr a ha, ?_⟩; trivial)

set_option maxHeartbeats 3200000 in
/-- Safety walk over the loss tail: only the gated gather load (active
exactly on the in-block branch, whose condition is the gate) and the unmasked
loss store touch memory. -/
private theorem ceFwdLossTail_traceSafe
    (loss_ptr logits_ptr : RegionName)
    (smoothing lse_square_scale : ℝ) (ignored_index class_start_idx : Int)
    (total_classes n_cols n_rows n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (bounds : RegionBounds) (smid : BlockState) (r₀ c₀ poff : Nat) (lv : Int)
    (lseC sumC : WithBot ℝ)
    (hrow : smid.regs TileDType.nat [] "row_idx" = some (Tile.scalar r₀))
    (hcol : smid.regs TileDType.nat [] "col_block_idx" = some (Tile.scalar c₀))
    (hlp : smid.regs TileDType.ptr [] "logits_ptr" =
      some (Tile.scalar ((Region.cast logits_ptr : RegionName), poff)))
    (hlbl : smid.regs TileDType.int [] "label_idx" = some (Tile.scalar lv))
    (hlse : smid.regs TileDType.real [] "lse" = some (Tile.scalar lseC))
    (hsum : HAS_SMOOTHING = Bool.true →
      smid.regs TileDType.real [] "sum_logits" = some (Tile.scalar sumC))
    (hbg : lv ≠ ignored_index →
      (↑(c₀ * (n + 1)) : Int) ≤ lv - class_start_idx →
      lv - class_start_idx < (↑(min n_cols ((c₀ + 1) * (n + 1))) : Int) →
      poff + (lv - class_start_idx).toNat < bounds logits_ptr)
    (hbw1 : c₀ * n_rows + r₀ < bounds loss_ptr) :
    Stmt.TraceSafeList bounds
      (ceFwdLossTail loss_ptr logits_ptr smoothing lse_square_scale
        ignored_index total_classes class_start_idx n_cols n_rows n
        HAS_SMOOTHING SPLIT) smid := by
  have hmin : (if decide (n_cols < (c₀ + 1) * (n + 1)) = «true» then n_cols
      else (c₀ + 1) * (n + 1)) = min n_cols ((c₀ + 1) * (n + 1)) := by
    by_cases hb : n_cols < (c₀ + 1) * (n + 1)
    · simp only [hb, decide_true, if_true]; omega
    · simp only [hb, decide_false, Bool.false_eq_true, if_false]; omega
  have hmin' : (if n_cols < (c₀ + 1) * (n + 1) then n_cols
      else (c₀ + 1) * (n + 1)) = min n_cols ((c₀ + 1) * (n + 1)) := by
    split <;> omega
  by_cases hHS : HAS_SMOOTHING = «true» <;>
    by_cases hSP : SPLIT = «true» <;>
    by_cases hIgn : lv = ignored_index <;>
    by_cases hAB : ((c₀ : Int) * ((n : Int) + 1) ≤ lv - class_start_idx ∧
      lv - class_start_idx < (n_cols : Int) ∧
      lv - class_start_idx < ((c₀ : Int) + 1) * ((n : Int) + 1)) <;>
  first
    | (have hsum' := hsum hHS
       have hbg' := hbg hIgn (by push_cast; omega) (by push_cast; omega)
       simp (maxSteps := 8000000) [ceFwdLossTail, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
         MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
         MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
         MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
         hrow, hcol, hlp, hlbl, hlse, hsum', hmin, hmin', hHS, hSP, hIgn, hAB,
         hbg', hbw1,
         Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
         NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
         ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
         TileShape.allIndices])
    | (have hsum' := hsum hHS
       simp (maxSteps := 8000000) [ceFwdLossTail, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
         MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
         MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
         MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
         hrow, hcol, hlp, hlbl, hlse, hsum', hmin, hmin', hHS, hSP, hIgn, hAB,
         hbw1,
         Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
         NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
         ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
         TileShape.allIndices])
    | (have hbg' := hbg hIgn (by push_cast; omega) (by push_cast; omega)
       simp (maxSteps := 8000000) [ceFwdLossTail, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
         MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
         MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
         MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
         hrow, hcol, hlp, hlbl, hlse, hmin, hmin', hHS, hSP, hIgn, hAB,
         hbg', hbw1,
         Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
         NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
         ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
         TileShape.allIndices])
    | simp (maxSteps := 8000000) [ceFwdLossTail, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
        MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
        MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
        MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
        hrow, hcol, hlp, hlbl, hlse, hmin, hmin', hHS, hSP, hIgn, hAB, hbw1,
        Tile.bop, Tile.cop, Tile.uop, Tile.select, Tile.ptrAdd, Tile.scalar, -Tile.scalar_eta,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.eq, ComparableDType.ge, ComparableDType.lt,
        TileShape.allIndices]

/-- The forward kernel sits inside the flat-memory bridge's covered fragment. -/
theorem cross_entropy_fwd_flattenOk
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) :
    ((cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr smoothing
      lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows
      logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [cross_entropy_fwd_surface, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- Full-kernel per-execution safety walk, split as prefix ++ tail. The
hypotheses are exactly the `⊨` skin's bounds obligations: the label slot
cell, the active row lanes, the label-gated gather cell, and the two
unconditional output cells. -/
theorem cross_entropy_fwd_traceSafe
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (bounds : RegionBounds) (s : BlockState) (lab : Int)
    (hlab : s.readMemValue .int (Region.cast labels_ptr) (s.pids 0) = lab)
    (hbL : s.pids 0 < bounds (Region.cast labels_ptr))
    (hbr : ∀ j : Fin (n+1), s.pids 1 * (n+1) + j.val < n_cols →
      s.pids 0 * logits_row_stride + (s.pids 1 * (n+1) + j.val)
        < bounds logits_ptr)
    (hbg : lab ≠ ignored_index →
      (↑(s.pids 1 * (n+1)) : Int) ≤ lab - class_start_idx →
      lab - class_start_idx < (↑(min n_cols ((s.pids 1 + 1) * (n+1))) : Int) →
      s.pids 0 * logits_row_stride + (lab - class_start_idx).toNat
        < bounds logits_ptr)
    (hbw1 : s.pids 1 * n_rows + s.pids 0 < bounds loss_ptr)
    (hbw2 : s.pids 1 * n_rows + s.pids 0 < bounds lse_ptr) :
    Kernel.TraceSafe bounds
      ((cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr smoothing
        lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows
        logits_row_stride (n+1) HAS_SMOOTHING SPLIT).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  rw [ceFwd_body_split]
  refine traceSafeList_append ?_ ?_
  · exact ceFwdLsePrefix_traceSafe loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING
      bounds s hbL hbr hbw2
  · intro s' hs'
    obtain ⟨smid, hpre⟩ := ceFwdLsePrefix_isSome loss_ptr lse_ptr logits_ptr
      labels_ptr smoothing total_classes n_cols n_rows logits_row_stride n
      HAS_SMOOTHING s
    obtain ⟨hrow, hcol, hlp, hlbl, ⟨lseC, hlse⟩, hsumEx, _⟩ :=
      ceFwdLsePrefix_run_facts loss_ptr lse_ptr logits_ptr labels_ptr smoothing
        total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING s smid hpre
    obtain rfl : smid = s' := by rw [hpre] at hs'; exact Option.some.inj hs'
    have hlv : labelValue s labels_ptr = lab := hlab
    rw [hlv] at hlbl
    obtain ⟨sumC, hsum⟩ : ∃ sumC : WithBot ℝ, (HAS_SMOOTHING = Bool.true →
        smid.regs TileDType.real [] "sum_logits" = some (Tile.scalar sumC)) := by
      cases hHS : HAS_SMOOTHING
      · exact ⟨0, fun h => absurd h (by decide)⟩
      · obtain ⟨c, hc⟩ := hsumEx hHS
        exact ⟨c, fun _ => hc⟩
    exact ceFwdLossTail_traceSafe loss_ptr logits_ptr smoothing lse_square_scale
      ignored_index class_start_idx total_classes n_cols n_rows n HAS_SMOOTHING
      SPLIT bounds smid (s.pids 0) (s.pids 1) (s.pids 0 * logits_row_stride) lab
      lseC sumC hrow hcol hlp hlbl hlse hsum
      (fun h1 h2 h3 => hbg h1 h2 h3) hbw1

/-- **The region-model masked Hoare triple** — termination, both output
cells' values (in-grid programs: the pure `ceLossLocal`/`ceBlockLSE` of the
pinned inputs; past-the-row programs: the `⊥`-path fallback `0`), and the
cell frame off the two written cells. This is the `hrun` obligation of the
`⊨` headline; the in-grid values reuse `cross_entropy_fwd_loss_correct` /
`cross_entropy_fwd_lse_correct` on the assembled execution. -/
theorem cross_entropy_fwd_region_run
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride : Nat) (n : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (hne : lse_ptr ≠ loss_ptr) (hLL : lse_ptr ≠ logits_ptr)
    (s₀ : BlockState) (lab : Int) (xs : Fin (n+1) → ℝ) (g : ℝ)
    (hlab : s₀.readMemValue .int (Region.cast labels_ptr) (s₀.pids 0) = lab)
    (hx : ∀ j : Fin (n+1), s₀.pids 1 * (n+1) + j.val < n_cols →
      s₀.readMem logits_ptr
        (s₀.pids 0 * logits_row_stride + (s₀.pids 1 * (n+1) + j.val)) = xs j)
    (hg : (lab ≠ ignored_index ∧
        lab - class_start_idx ≥ (↑(s₀.pids 1 * (n+1)) : Int) ∧
        lab - class_start_idx < (↑(min n_cols ((s₀.pids 1 + 1) * (n+1))) : Int)) →
      s₀.readMem logits_ptr
        (s₀.pids 0 * logits_row_stride + (lab - class_start_idx).toNat) = g) :
    ∃ s1, exec ((cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr
        smoothing lse_square_scale ignored_index total_classes class_start_idx
        n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING SPLIT).toAlgKernel) s₀
        = some s1
      ∧ s1.readMem loss_ptr (lseOutOffset s₀ n_rows)
          = (if s₀.pids 1 * (n+1) < n_cols then
              ceLossLocal n_cols total_classes (n+1) smoothing lse_square_scale
                ignored_index class_start_idx HAS_SMOOTHING SPLIT (s₀.pids 1)
                lab xs g
            else 0)
      ∧ s1.readMem lse_ptr (lseOutOffset s₀ n_rows)
          = (if s₀.pids 1 * (n+1) < n_cols then
              ceBlockLSE n_cols (n+1) (s₀.pids 1) xs
            else 0)
      ∧ (∀ r o, ¬(loss_ptr = r ∧ lseOutOffset s₀ n_rows = o) →
          ¬(lse_ptr = r ∧ lseOutOffset s₀ n_rows = o) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨smid, hpre⟩ := ceFwdLsePrefix_isSome loss_ptr lse_ptr logits_ptr
    labels_ptr smoothing total_classes n_cols n_rows logits_row_stride n
    HAS_SMOOTHING s₀
  obtain ⟨hrow, hcol, hlp, hlbl, ⟨lseC, hlse⟩, hsumEx, hpframe⟩ :=
    ceFwdLsePrefix_run_facts loss_ptr lse_ptr logits_ptr labels_ptr smoothing
      total_classes n_cols n_rows logits_row_stride n HAS_SMOOTHING s₀ smid hpre
  have hlv : labelValue s₀ labels_ptr = lab := hlab
  rw [hlv] at hlbl
  obtain ⟨sumC, hsum⟩ : ∃ sumC : WithBot ℝ, (HAS_SMOOTHING = Bool.true →
      smid.regs TileDType.real [] "sum_logits" = some (Tile.scalar sumC)) := by
    cases hHS : HAS_SMOOTHING
    · exact ⟨0, fun h => absurd h (by decide)⟩
    · obtain ⟨c, hc⟩ := hsumEx hHS
      exact ⟨c, fun _ => hc⟩
  by_cases h_tail : s₀.pids 1 * (n+1) < n_cols
  · -- in-grid program: genuine values via the existing exec value lemmas
    obtain ⟨s1, htail, htframe⟩ := ceFwdLossTail_run loss_ptr logits_ptr
      smoothing lse_square_scale ignored_index class_start_idx total_classes
      n_cols n_rows n HAS_SMOOTHING SPLIT smid (s₀.pids 0) (s₀.pids 1)
      (s₀.pids 0 * logits_row_stride) lab lseC sumC
      hrow hcol hlp hlbl hlse hsum
    have hExec : exec ((cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr
        labels_ptr smoothing lse_square_scale ignored_index total_classes
        class_start_idx n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING
        SPLIT).toAlgKernel) s₀ = some s1 := by
      rw [exec, ceFwd_body_split, stepStmts.append_some_iff]
      exact ⟨smid, hpre, htail⟩
    have hloss := cross_entropy_fwd_loss_correct loss_ptr lse_ptr logits_ptr
      labels_ptr smoothing lse_square_scale ignored_index total_classes
      class_start_idx n_cols n_rows logits_row_stride n HAS_SMOOTHING SPLIT
      s₀ s1 h_tail hLL hExec
    have hlse2 := cross_entropy_fwd_lse_correct loss_ptr lse_ptr logits_ptr
      labels_ptr smoothing lse_square_scale ignored_index total_classes
      class_start_idx n_cols n_rows logits_row_stride n HAS_SMOOTHING SPLIT
      s₀ s1 h_tail hne hExec
    refine ⟨s1, hExec, ?_, ?_, ?_⟩
    · rw [if_pos h_tail, hloss, hlv]
      exact crossEntropyLossSpec_eq_ceLossLocal s₀ logits_ptr smoothing
        lse_square_scale ignored_index class_start_idx total_classes n_cols
        logits_row_stride n HAS_SMOOTHING SPLIT h_tail lab xs g hx hg
    · rw [if_pos h_tail, hlse2]
      exact partialLSE_full_eq_ceBlockLSE n (s₀.pids 1) h_tail
        (rowLogits s₀ logits_ptr logits_row_stride n_cols) xs
        (fun j hj => hx j hj)
    · intro r o hm1 hm2
      have h1 : s1.mem r o = smid.mem r o := by
        refine htframe r o (fun hc => hm1 ?_)
        exact ⟨hc.1, by simpa [lseOutOffset] using hc.2⟩
      rw [h1]
      exact hpframe r o hm2
  · -- past-the-row program: the ⊥-path fallback
    obtain ⟨hlseBot, hsum0, hcell0⟩ := ceFwdLsePrefix_run_fallback loss_ptr
      lse_ptr logits_ptr labels_ptr smoothing total_classes n_cols n_rows
      logits_row_stride n HAS_SMOOTHING s₀ smid h_tail hpre
    have hnb : ¬((↑(s₀.pids 1 * (n + 1)) : Int) ≤ lab - class_start_idx ∧
        lab - class_start_idx <
          (↑(min n_cols ((s₀.pids 1 + 1) * (n + 1))) : Int)) := by
      rintro ⟨h1, h2⟩
      have hcols : n_cols ≤ s₀.pids 1 * (n+1) := Nat.le_of_not_lt h_tail
      have hminle : min n_cols ((s₀.pids 1 + 1) * (n+1)) ≤ n_cols :=
        Nat.min_le_left _ _
      omega
    obtain ⟨s1, htail, hloss0, htframe⟩ := ceFwdLossTail_run_fallback loss_ptr
      logits_ptr smoothing lse_square_scale ignored_index class_start_idx
      total_classes n_cols n_rows n HAS_SMOOTHING SPLIT smid (s₀.pids 0)
      (s₀.pids 1) (s₀.pids 0 * logits_row_stride) lab
      hrow hcol hlp hlbl hlseBot hsum0 hnb
    have hExec : exec ((cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr
        labels_ptr smoothing lse_square_scale ignored_index total_classes
        class_start_idx n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING
        SPLIT).toAlgKernel) s₀ = some s1 := by
      rw [exec, ceFwd_body_split, stepStmts.append_some_iff]
      exact ⟨smid, hpre, htail⟩
    refine ⟨s1, hExec, ?_, ?_, ?_⟩
    · rw [if_neg h_tail]
      simpa [lseOutOffset] using hloss0
    · rw [if_neg h_tail]
      have hstep : s1.readMem lse_ptr (lseOutOffset s₀ n_rows)
          = smid.readMem lse_ptr (lseOutOffset s₀ n_rows) := by
        unfold BlockState.readMem
        rw [htframe lse_ptr (lseOutOffset s₀ n_rows)
          (fun hc => hne hc.1.symm)]
      rw [hstep]
      exact hcell0
    · intro r o hm1 hm2
      have h1 : s1.mem r o = smid.mem r o := by
        refine htframe r o (fun hc => hm1 ?_)
        exact ⟨hc.1, by simpa [lseOutOffset] using hc.2⟩
      rw [h1]
      exact hpframe r o hm2

/-- `cross_entropy_fwd_surface`'s metadata-genre **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `mbufL` — the `.int` label slot: program `(row, col_block)` loads
  `labels_ptr[row]` (`mwinL`);
* `inp` — the logits matrix, read twice: the masked row block
  (`read`/`mask`: lane `j` at `row·stride + col_block·B + j`, active while
  `col_block·B + j < n_cols`) and the label-gated single-cell gather
  (`gwin`/`gmask`: cell `row·stride + (lab − class_start_idx)`, read exactly
  when the label is live and its shifted position falls in this block);
* `out1`/`out2` — the loss and LSE cells, both at `col_block·n_rows + row`
  (the host's `(n_splits, n_rows)` layout), written unconditionally
  (`writeMask` defaults).

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, gating, and masking match them. -/
def crossEntropyFwdIO
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) : MetaGatherMasked2DKernelIO₂ₓ₂ where
  kernel := cross_entropy_fwd_surface loss_ptr lse_ptr logits_ptr labels_ptr
    smoothing lse_square_scale ignored_index total_classes class_start_idx
    n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  mbufL := Region.cast labels_ptr
  inp := logits_ptr
  out1 := loss_ptr
  out2 := lse_ptr
  B := BLOCK_SIZE
  mwinL := fun pid₀ _ => pid₀
  read := fun pid₀ pid₁ _ j =>
    pid₀ * logits_row_stride + (pid₁ * BLOCK_SIZE + j.val)
  mask := fun _ pid₁ _ j => pid₁ * BLOCK_SIZE + j.val < n_cols
  gwin := fun pid₀ _ lab =>
    pid₀ * logits_row_stride + (lab - class_start_idx).toNat
  gmask := fun _ pid₁ lab =>
    lab ≠ ignored_index ∧
    lab - class_start_idx ≥ (↑(pid₁ * BLOCK_SIZE) : Int) ∧
    lab - class_start_idx < (↑(min n_cols ((pid₁ + 1) * BLOCK_SIZE)) : Int)
  write1 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀
  write2 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀

open scoped VeriTile.Triton.MetaGatherMasked2DKernelIO₂ₓ₂ in
/-- **The headline**: `cross_entropy_fwd_kernel` implements the pure
per-program cross-entropy pair on its metadata-genre IO signature — for
every disjoint flat placement of the four buffers, every program
`(row, col_block)` whose declared cells/lanes are in bounds, and every launch
state pinning the label `lab` at the slot cell, the block logits `xs` on the
active lanes, and the gather cell `g` under its gate, the translated pointer
kernel terminates and writes

* `loss_ptr[col_block·n_rows + row] = ceLossLocal … lab xs g` — the faithful
  five-way loss (ignored label / label-in-block / `HAS_SMOOTHING` / `SPLIT` /
  `lse²` term) over the pinned inputs, and
* `lse_ptr[col_block·n_rows + row] = ceBlockLSE … xs` — the log-sum-exp of
  the active block lanes,

for in-grid programs (`col_block·B < n_cols`); programs whose block lies past
the row end write the IEEE-faithful `⊥`-path fallback `0` to both cells.
Every other memory cell is unchanged. `0 < BLOCK_SIZE` is required (the `max`
reduce needs a lane); the two output buffers must differ (`lse_ptr ≠
loss_ptr`) so each readback sees through the other store, and `lse_ptr ≠
logits_ptr` so the LSE store does not clobber the gather cell it reads.
Proof: `MetaGatherMasked2DKernelIO₂ₓ₂.Implements.intro` assembles the region-model
masked triple with the flat-memory bridge side conditions. -/
specification cross_entropy_fwd_correctness
    (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (hB : 0 < BLOCK_SIZE)
    (hne : lse_ptr ≠ loss_ptr) (hLL : lse_ptr ≠ logits_ptr) :
    crossEntropyFwdIO loss_ptr lse_ptr logits_ptr labels_ptr smoothing
        lse_square_scale ignored_index total_classes class_start_idx n_cols
        n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT ⊨
      fun _ pid₁ lab xs g =>
        if pid₁ * BLOCK_SIZE < n_cols then
          (ceLossLocal n_cols total_classes BLOCK_SIZE smoothing
             lse_square_scale ignored_index class_start_idx HAS_SMOOTHING SPLIT
             pid₁ lab xs g,
           ceBlockLSE n_cols BLOCK_SIZE pid₁ xs)
        else (0, 0) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  refine MetaGatherMasked2DKernelIO₂ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact cross_entropy_fwd_flattenOk loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing lse_square_scale ignored_index total_classes class_start_idx
      n_cols n_rows logits_row_stride (n+1) HAS_SMOOTHING SPLIT
  · intro bounds s lab hlab hbL hbr hbg hbw1 hbw2
    exact cross_entropy_fwd_traceSafe loss_ptr lse_ptr logits_ptr labels_ptr
      smoothing lse_square_scale ignored_index total_classes class_start_idx
      n_cols n_rows logits_row_stride n HAS_SMOOTHING SPLIT bounds s lab hlab
      hbL (fun j hj => hbr j hj) (fun h1 h2 h3 => hbg ⟨h1, h2, h3⟩)
      (hbw1 trivial) (hbw2 trivial)
  · intro s₀ lab xs g hlab hx hg
    obtain ⟨s1, hexec, hval1, hval2, hframe⟩ := cross_entropy_fwd_region_run
      loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale
      ignored_index total_classes class_start_idx n_cols n_rows
      logits_row_stride n HAS_SMOOTHING SPLIT hne hLL s₀ lab xs g hlab
      (fun j hj => hx j hj) (fun hgm => hg hgm)
    refine ⟨s1, hexec, fun _ => ?_, fun _ => ?_, fun r o h1 h2 => ?_⟩
    · rw [show ((if s₀.pids 1 * (n+1) < n_cols then
          (ceLossLocal n_cols total_classes (n+1) smoothing lse_square_scale
             ignored_index class_start_idx HAS_SMOOTHING SPLIT (s₀.pids 1)
             lab xs g,
           ceBlockLSE n_cols (n+1) (s₀.pids 1) xs)
        else ((0:ℝ), (0:ℝ))).1 : ℝ)
        = (if s₀.pids 1 * (n+1) < n_cols then
            ceLossLocal n_cols total_classes (n+1) smoothing lse_square_scale
              ignored_index class_start_idx HAS_SMOOTHING SPLIT (s₀.pids 1)
              lab xs g
          else 0) from by split <;> rfl]
      exact hval1
    · rw [show ((if s₀.pids 1 * (n+1) < n_cols then
          (ceLossLocal n_cols total_classes (n+1) smoothing lse_square_scale
             ignored_index class_start_idx HAS_SMOOTHING SPLIT (s₀.pids 1)
             lab xs g,
           ceBlockLSE n_cols (n+1) (s₀.pids 1) xs)
        else ((0:ℝ), (0:ℝ))).2 : ℝ)
        = (if s₀.pids 1 * (n+1) < n_cols then
            ceBlockLSE n_cols (n+1) (s₀.pids 1) xs
          else 0) from by split <;> rfl]
      exact hval2
    · refine hframe r o ?_ ?_
      · rcases h1 with hner | hno
        · exact fun hc => hner hc.1.symm
        · exact fun hc => hno trivial hc.2.symm
      · rcases h2 with hner | hno
        · exact fun hc => hner hc.1.symm
        · exact fun hc => hno trivial hc.2.symm

end VeriTile.Bench.TritonBenchG.CrossEntropy1
