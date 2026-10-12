import bench.tritonbench_g.cross_entropy2.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.CrossEntropy2.BackwardFP
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
section
variable (storage : Storage) (dlogits_ptr dloss_ptr logits_ptr lse_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE : Nat)
    (HAS_SMOOTHING : Bool)

def original : Program where
  kernel := backward storage Bool.false dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING
  domain := []
def indexCommuted : Program where
  kernel := backward storage Bool.true dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING
  domain := []

theorem equivalent :
    original storage dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING ≡[assumptions] indexCommuted storage dlogits_ptr dloss_ptr logits_ptr lse_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE HAS_SMOOTHING := by
  cases storage <;> cases HAS_SMOOTHING <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CrossEntropy2.BackwardFP
