import bench.tritonbench_g.cross_entropy2.Kernels
import bench.tritonbench_g.cross_entropy2.Proofs.Index
import bench.tritonbench_g.cross_entropy2.Proofs.ForwardReal
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.CrossEntropy2.ForwardProjection
open VeriTile.Triton Index
set_option maxHeartbeats 12000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false


section
variable (storage : Storage) (swapped : Bool) (loss_ptr lse_ptr z_loss_ptr logits_ptr : RegionName) (labels_ptr : Region .int)
    (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
    (total_classes : Nat) (class_start_idx : Int)
    (n_cols n_rows logits_row_stride BLOCK_SIZE : Nat) (HAS_SMOOTHING SPLIT : Bool)


def base := (ForwardReal.cross_entropy_fwd_surface loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).toAlgKernel

theorem source_body :
    (forward storage swapped loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType.toAlgKernel.body =
      (base loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).body.take 3 ++ [index BLOCK_SIZE swapped] ++
        (base loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).body.drop 4 := by
  cases storage <;> cases swapped <;> cases HAS_SMOOTHING <;> cases SPLIT
  all_goals simp only [forward,base,ForwardReal.cross_entropy_fwd_surface,index,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind,eraseDType_float]
  all_goals simp only [Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType]
  all_goals rfl
theorem base_split : (base loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).body =
    (base loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).body.take 3 ++ index BLOCK_SIZE Bool.false :: (base loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).body.drop 4 := by
  cases HAS_SMOOTHING <;> cases SPLIT <;> rfl

theorem executes_as_base (u : BlockState) :
    exec (forward storage swapped loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType.toAlgKernel u = exec (base loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT) u := by
  simp only [exec,source_body,List.append_assoc,List.singleton_append]
  cases swapped
  · rw [← base_split]
  · rw [Index.context_exec,← base_split]

theorem safe_as_base (bounds : RegionBounds) (u : BlockState)
    (h : Kernel.TraceSafe bounds (base loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT) u) :
    Kernel.TraceSafe bounds (forward storage swapped loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType.toAlgKernel u := by
  change Stmt.TraceSafeList bounds _ u at h ⊢
  rw [base_split] at h
  rw [source_body,List.append_assoc,List.singleton_append]
  cases swapped
  · exact h
  · exact (Index.context_safe _ _ _ _ _).mpr h

theorem flatten_as_base (h : (base loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).FlattenOk) :
    (forward storage swapped loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType.toAlgKernel.FlattenOk := by
  change StmtList.FlattenOk _ at h ⊢
  rw [base_split] at h
  rw [source_body,List.append_assoc,List.singleton_append]
  cases swapped
  · exact h
  · exact (Index.context_flatten _ _ _).mpr h

theorem projects : (forward storage swapped loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).toAlgorithm? = .ok (forward storage swapped loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).toAlgKernel := by
  cases storage <;> cases swapped <;> cases HAS_SMOOTHING <;> cases SPLIT <;> rfl

theorem projection : (forward storage swapped loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType.toAlgorithm? = .ok (forward storage swapped loss_ptr lse_ptr z_loss_ptr logits_ptr labels_ptr smoothing logit_scale lse_square_scale ignored_index total_classes class_start_idx n_cols n_rows logits_row_stride BLOCK_SIZE HAS_SMOOTHING SPLIT).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CrossEntropy2.ForwardProjection
