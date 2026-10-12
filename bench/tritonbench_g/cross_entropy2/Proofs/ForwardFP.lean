import bench.tritonbench_g.cross_entropy2.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.CrossEntropy2.ForwardFP
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def assumptions : Spec.Assumptions FP.GuardedFragment := []
section
variable (storage : Storage) (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat) (HAS_SMOOTHING SPLIT : Bool)

def original : Program where
  kernel := forward storage Bool.false loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  domain := []
def indexCommuted : Program where
  kernel := forward storage Bool.true loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT
  domain := []

theorem equivalent :
    original storage loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT ≡[assumptions] indexCommuted storage loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT := by
  cases storage <;> cases HAS_SMOOTHING <;> cases SPLIT <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions equivalent
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CrossEntropy2.ForwardFP
