import bench.tritonbench_g.cross_entropy_ops.Kernels
import bench.tritonbench_g.cross_entropy_ops.Proofs.ForwardIO
import bench.tritonbench_g.cross_entropy_ops.Proofs.BackwardIO
import VeriTile.Meta.StatementAudit
/-! Real correctness of both complete scaled cross-entropy JITs and candidates.
The independent formulas retain logit_scale and the non-split z-loss output.
Both layers prove successful finite flat execution and the full tagged-cell
frame. Forward retains the shared adapter's zero unspecified-value oracle;
backward only requires actual-access bounds. These are per-program model
contracts, without host split aggregation, IEEE or concurrent-launch claims. -/
namespace VeriTile.Bench.TritonBenchG.CrossEntropyOps.Correctness.Forward
open VeriTile VeriTile.Triton
open scoped VeriTile.Triton.MetaMasked2DKernelIO₂ₓ₃
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000
/-- Label, active block, conditional target gather, loss/LSE and gated z-loss
windows. The third output is written and claimed exactly when SPLIT is false. -/
def forwardOriginalIO (storage : Storage)
    (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) : MetaMasked2DKernelIO₂ₓ₃ where
  kernel := (forward storage Bool.false loss_ptr lse_ptr z_loss_ptr logits_ptr
    labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes
    class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType
  projection := ForwardProjection.projection storage Bool.false loss_ptr lse_ptr z_loss_ptr logits_ptr
    labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes
    class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  mbufL := Region.cast labels_ptr
  inp := logits_ptr
  out1 := loss_ptr
  out2 := lse_ptr
  out3 := z_loss_ptr
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
  write3 := fun pid₀ pid₁ _ => pid₁ * n_rows + pid₀
  writeMask3 := fun _ _ _ => SPLIT = Bool.false

def forwardIndexCommutedIO (storage : Storage)
    (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool) : MetaMasked2DKernelIO₂ₓ₃ :=
  { forwardOriginalIO storage loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT with
    kernel := (forward storage Bool.true loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType
    projection := ForwardProjection.projection storage Bool.true loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT }

/-- Independent functions of initial logits and the initial target logit. -/
noncomputable def blockLSE (n_cols B pid₁ : Nat) (logit_scale : ℝ)
    (xs : Fin B → ℝ) : ℝ :=
  Real.log (∑ i ∈ Finset.univ.filter (fun i : Fin B => pid₁ * B + i.val < n_cols),
    Real.exp (xs i * logit_scale))

noncomputable def blockSum (n_cols B pid₁ : Nat) (logit_scale : ℝ)
    (xs : Fin B → ℝ) : ℝ :=
  ∑ i : Fin B, if pid₁ * B + i.val < n_cols then xs i * logit_scale else 0

noncomputable def loss (n_cols total_classes B : Nat)
    (smoothing logit_scale lse_square_scale : ℝ)
    (ignored_index class_start_idx : Int)
    (HAS_SMOOTHING SPLIT : Bool)
    (pid₁ : Nat) (lab : Int) (xs : Fin B → ℝ) (g : ℝ) : ℝ :=
  if lab = ignored_index then 0 else
    let lblShift : Int := lab - class_start_idx
    let lse : ℝ := blockLSE n_cols B pid₁ logit_scale xs
    let lseTerm : ℝ := if SPLIT then 0 else lse
    let sq : ℝ := if SPLIT then 0 else lse_square_scale * lse * lse
    let core : ℝ :=
      if (lblShift ≥ (pid₁ * B : Nat)) ∧
         (lblShift < (min n_cols ((pid₁ + 1) * B) : Nat)) then
        if HAS_SMOOTHING then
          lseTerm - smoothing * blockSum n_cols B pid₁ logit_scale xs / total_classes
            - (1 - smoothing) * (g * logit_scale)
        else
          lseTerm - g * logit_scale
      else
        if HAS_SMOOTHING then
          smoothing * (lseTerm - blockSum n_cols B pid₁ logit_scale xs / total_classes)
        else 0
    core + sq

noncomputable def zLoss (n_cols B : Nat)
    (logit_scale lse_square_scale : ℝ) (ignored_index : Int)
    (pid₁ : Nat) (lab : Int) (xs : Fin B → ℝ) : ℝ :=
  if lab = ignored_index then 0 else
    lse_square_scale * blockLSE n_cols B pid₁ logit_scale xs
      * blockLSE n_cols B pid₁ logit_scale xs

/-- Positive block size, separate outputs and no LSE overwrite of logits
protect the later gather and all output readbacks. Actual accesses are bounded. -/
specification forward_original_correct (storage : Storage)
    (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (hB : 0 < BLOCK_SIZE)
    (hne : lse_ptr ≠ loss_ptr) (hneZ : lse_ptr ≠ z_loss_ptr)
    (hLL : lse_ptr ≠ logits_ptr) (hLZ : loss_ptr ≠ z_loss_ptr) :
    Spec.Real (forwardOriginalIO storage loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing
        logit_scale lse_square_scale ignored_index total_classes class_start_idx
        n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT ⊨
      fun _ pid₁ lab xs g =>
        if pid₁ * BLOCK_SIZE < n_cols then
          (loss n_cols total_classes BLOCK_SIZE smoothing logit_scale
             lse_square_scale ignored_index class_start_idx HAS_SMOOTHING SPLIT
             pid₁ lab xs g,
           blockLSE n_cols BLOCK_SIZE pid₁ logit_scale xs,
           zLoss n_cols BLOCK_SIZE logit_scale lse_square_scale
             ignored_index pid₁ lab xs)
        else (0, 0, 0)) := by
  exact ForwardIO.implements storage Bool.false loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT hB hne hneZ hLL hLZ

/-- Positive block size, separate outputs and no LSE overwrite of logits
protect the later gather and all output readbacks. Actual accesses are bounded. -/
specification forward_index_commuted_correct (storage : Storage)
    (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING SPLIT : Bool)
    (hB : 0 < BLOCK_SIZE)
    (hne : lse_ptr ≠ loss_ptr) (hneZ : lse_ptr ≠ z_loss_ptr)
    (hLL : lse_ptr ≠ logits_ptr) (hLZ : loss_ptr ≠ z_loss_ptr) :
    Spec.Real (forwardIndexCommutedIO storage loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing
        logit_scale lse_square_scale ignored_index total_classes class_start_idx
        n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT ⊨
      fun _ pid₁ lab xs g =>
        if pid₁ * BLOCK_SIZE < n_cols then
          (loss n_cols total_classes BLOCK_SIZE smoothing logit_scale
             lse_square_scale ignored_index class_start_idx HAS_SMOOTHING SPLIT
             pid₁ lab xs g,
           blockLSE n_cols BLOCK_SIZE pid₁ logit_scale xs,
           zLoss n_cols BLOCK_SIZE logit_scale lse_square_scale
             ignored_index pid₁ lab xs)
        else (0, 0, 0)) := by
  exact ForwardIO.implements storage Bool.true loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT hB hne hneZ hLL hLZ

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CrossEntropyOps.Correctness.Forward

namespace VeriTile.Bench.TritonBenchG.CrossEntropyOps.Correctness.Backward
open VeriTile VeriTile.Triton BackwardFormula BackwardProgram
open scoped VeriTile.Bench.TritonBenchG.CrossEntropyOps.BackwardIO
noncomputable section

/-- Every actual scalar/active-lane access is bounded. An ignored label skips
only dloss; labels, LSE and active logits are still read. Inputs may alias the
output because all reads precede the sole store. The column offsets are
injective, so no extra output-distinctness assumption is needed. -/
def backwardOriginalIO (p : Layout) (r : Regions) (storage : Storage) : BackwardIO.IO p where
  kernel := (source p r storage Bool.false).eraseDType
  projection := BackwardProgram.projection p r storage Bool.false
  read := fun u => {
    logits := u.readMem r.logits, lse := u.readMem r.lse,
    dloss := u.readMem r.dloss, label := u.readMemValue .int r.label }
  bounds := BackwardSafety.Bounds p r
  out := r.gradient
  write := fun l i => l.row*p.gradientStride+(l.block*p.B+i.val)
  mask := fun l i => l.block*p.B+i.val < p.nCols

def backwardIndexCommutedIO (p : Layout) (r : Regions) (storage : Storage) : BackwardIO.IO p :=
  { backwardOriginalIO p r storage with
    kernel := (source p r storage Bool.true).eraseDType
    projection := BackwardProgram.projection p r storage Bool.true }

/-- Independent formula using the backward kernel's legitimate initial inputs;
LSE is supplied by the caller and probabilities are computed by this kernel. -/
def gradient (p : Layout) (l : Launch) (a : Inputs) (i : Fin p.B) : ℝ :=
  let col := l.block*p.B+i.val
  let prob := Real.exp (a.logits (l.row*p.logitsStride+col)*p.logitScale-a.lse l.row)
  let adj := prob+2*p.squareScale*a.lse l.row*prob
  let upstream := if a.label l.row=p.ignored then 0 else a.dloss (l.row*p.dlossStride)
  (upstream * p.logitScale) *
    (if p.hasSmoothing then
      (adj - if (col : Int)=a.label l.row-p.classStart then 1-p.smoothing else 0) -
        p.smoothing / p.totalClasses
     else adj - if (col : Int)=a.label l.row-p.classStart then 1 else 0)

specification backward_original_correct (p : Layout) (r : Regions) (storage : Storage) :
    Spec.Real (backwardOriginalIO p r storage ⊨ gradient p) := by
  exact BackwardIO.implements p r storage Bool.false

specification backward_index_commuted_correct (p : Layout) (r : Regions) (storage : Storage) :
    Spec.Real (backwardIndexCommutedIO p r storage ⊨ gradient p) := by
  exact BackwardIO.implements p r storage Bool.true

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CrossEntropyOps.Correctness.Backward
