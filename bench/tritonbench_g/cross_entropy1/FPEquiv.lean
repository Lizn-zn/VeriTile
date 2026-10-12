import bench.tritonbench_g.cross_entropy1.Kernels
import bench.tritonbench_g.cross_entropy1.Proofs.ForwardFP
import bench.tritonbench_g.cross_entropy1.Proofs.BackwardFP
import VeriTile.Meta.StatementAudit

/-! Whole-program equivalence for both complete typed sources. Only the exact
integer column-offset product is commuted. Numerical operations, signed labels,
all smoothing/split branches, casts, masks, writes and failure are preserved.
No numerical atom or numerical-domain premise is needed. -/
namespace VeriTile.Bench.TritonBenchG.CrossEntropy1.FPEquivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

section
variable (storage : Storage) (loss_ptr lse_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat) (HAS_SMOOTHING SPLIT : Bool)

def forwardOriginal : Program where
  kernel := forward storage Bool.false loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  domain := []
def forwardIndexCommuted : Program where
  kernel := forward storage Bool.true loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  domain := []

specification forward_index_commuted_equiv :
    forwardOriginal storage loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT ≡[assumptions] forwardIndexCommuted storage loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT := by
  exact ForwardFP.equivalent storage loss_ptr lse_ptr logits_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
end
#print_fp_assumptions forward_index_commuted_equiv

section
variable (storage : Storage) (dlogits_ptr dloss_ptr logits_ptr lse_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING : Bool)

def backwardOriginal : Program where
  kernel := backward storage Bool.false dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING
  domain := []
def backwardIndexCommuted : Program where
  kernel := backward storage Bool.true dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING
  domain := []

specification backward_index_commuted_equiv :
    backwardOriginal storage dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING ≡[assumptions] backwardIndexCommuted storage dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING := by
  exact BackwardFP.equivalent storage dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING
end
#print_fp_assumptions backward_index_commuted_equiv

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CrossEntropy1.FPEquivalence
