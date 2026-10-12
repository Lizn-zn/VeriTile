import bench.tritonbench_g.cross_entropy_ops.Kernels
import bench.tritonbench_g.cross_entropy_ops.Proofs.Index
import bench.tritonbench_g.cross_entropy_ops.Proofs.BackwardFormula
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.CrossEntropyOps.BackwardProgram
open VeriTile.Triton BackwardFormula Index
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

structure Regions where
  gradient : RegionName
  dloss : RegionName
  logits : RegionName
  lse : RegionName
  label : Region .int

/-- Real projection of the complete backward source, with precision erased. -/
def realKernel := fun (dlogits_ptr dloss_ptr logits_ptr lse_ptr : RegionName) (labels_ptr : Region .int)
        (smoothing logit_scale lse_square_scale : ℝ) (ignored_index : Int)
        (total_classes : Nat) (class_start_idx : Int)
        (n_cols logits_row_stride dlogits_row_stride dloss_row_stride BLOCK_SIZE : Nat)
        (HAS_SMOOTHING : Bool) => ComputeKernel.mk
      [dloss_ptr,logits_ptr,lse_ptr,Region.cast labels_ptr] [dlogits_ptr] (triton {
  row_idx = tl.program_id(0)
  col_block_idx = tl.program_id(1)
  logits_ptr = logits_ptr + row_idx * ($(logits_row_stride)).to(tl.int64)
  dlogits_ptr = dlogits_ptr + row_idx * ($(dlogits_row_stride)).to(tl.int64)
  col_offsets = col_block_idx * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  label_idx = tl.load(labels_ptr + row_idx)
  if label_idx != $((ignored_index : Int)) {
    dloss = tl.load(dloss_ptr + row_idx * $(dloss_row_stride))
  } else {
    dloss = 0.0
  }
  logits = tl.load(logits_ptr + col_offsets,
    mask=col_offsets < $(n_cols), other=-float("inf")) * $((logit_scale : ℝ))
  lse = tl.load(lse_ptr + row_idx)
  probs = tl.exp(logits - lse)
  probs += 2.0 * $((lse_square_scale : ℝ)) * lse * probs
  label_idx -= $((class_start_idx : Int))
  if HAS_SMOOTHING {
    smooth_positive = 1.0 - $((smoothing : ℝ))
    smooth_negative = $((smoothing : ℝ)) / $(total_classes)
    probs = tl.where(col_offsets == label_idx, probs - (1 - $((smoothing : ℝ))), probs) - smooth_negative
  } else {
    probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
  }
  tl.store(dlogits_ptr + col_offsets, ((dloss * $((logit_scale : ℝ))) * probs), mask=col_offsets < $(n_cols))
}).surfaceBody

def source (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) :=
  backward storage swapped r.gradient r.dloss r.logits r.lse r.label p.smoothing p.logitScale p.squareScale p.ignored p.totalClasses p.classStart p.nCols p.logitsStride p.gradientStride p.dlossStride p.B p.hasSmoothing
def mathematical (p : Layout) (r : Regions) := realKernel r.gradient r.dloss r.logits r.lse r.label p.smoothing p.logitScale p.squareScale p.ignored p.totalClasses p.classStart p.nCols p.logitsStride p.gradientStride p.dlossStride p.B p.hasSmoothing

def launch (u : BlockState) : Launch := ⟨u.pids 0,u.pids 1⟩
def inputs (u : BlockState) (r : Regions) : Inputs where
  logits := u.readMem r.logits
  lse := u.readMem r.lse
  dloss := u.readMem r.dloss
  label := u.readMemValue .int r.label

theorem source_body (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) :
    (source p r storage swapped).eraseDType.toAlgKernel.body =
      (mathematical p r).toAlgKernel.body.take 4 ++ [index p.B swapped] ++
        (mathematical p r).toAlgKernel.body.drop 5 := by
  cases storage <;> cases swapped <;> cases hs : p.hasSmoothing
  all_goals simp only [source,backward,mathematical,realKernel,index,hs,
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

theorem base_split (p : Layout) (r : Regions) : (mathematical p r).toAlgKernel.body =
    (mathematical p r).toAlgKernel.body.take 4 ++ index p.B Bool.false ::
      (mathematical p r).toAlgKernel.body.drop 5 := by rfl

theorem executes_as_base (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) (u : BlockState) :
    exec (source p r storage swapped).eraseDType.toAlgKernel u =
      exec (mathematical p r).toAlgKernel u := by
  simp only [exec,source_body,List.append_assoc,List.singleton_append]
  cases swapped
  · rw [← base_split]
  · rw [Index.context_exec,← base_split]

theorem safe_as_base (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool)
    (extent : RegionBounds) (u : BlockState)
    (h : Kernel.TraceSafe extent (mathematical p r).toAlgKernel u) :
    Kernel.TraceSafe extent (source p r storage swapped).eraseDType.toAlgKernel u := by
  change Stmt.TraceSafeList extent _ u at h ⊢
  rw [base_split] at h
  rw [source_body,List.append_assoc,List.singleton_append]
  cases swapped
  · exact h
  · exact (Index.context_safe _ _ _ _ _).mpr h

theorem flatten_as_base (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool)
    (h : (mathematical p r).toAlgKernel.FlattenOk) :
    (source p r storage swapped).eraseDType.toAlgKernel.FlattenOk := by
  change StmtList.FlattenOk _ at h ⊢
  rw [base_split] at h
  rw [source_body,List.append_assoc,List.singleton_append]
  cases swapped
  · exact h
  · exact (Index.context_flatten _ _ _).mpr h

theorem projects (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) :
    (source p r storage swapped).toAlgorithm? = .ok (source p r storage swapped).toAlgKernel := by
  cases storage <;> cases swapped <;> rfl

theorem projection (p : Layout) (r : Regions) (storage : Storage) (swapped : Bool) :
    (source p r storage swapped).eraseDType.toAlgorithm? =
      .ok (source p r storage swapped).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType,projects,ComputeKernel.toAlgorithm?_fromKernelBody,
    ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CrossEntropyOps.BackwardProgram
