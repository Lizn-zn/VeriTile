import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerInitialize
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerLoopSafety
import bench.tritonbench_g.softmax_flaggems.Proofs.RealTrace

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem prefix_safe (bounds : RegionBounds) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) :
    Stmt.TraceSafeList bounds (pass1Prefix Y X M N K B C) s := by
  rw [pass1Prefix_extract]
  simp [pass1Prefix, softmax_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  apply RealTrace.of_all
  intro statement hstatement st
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at hstatement
  rcases hstatement with rfl | rfl | rfl | rfl | rfl
  all_goals simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem reduction_safe (bounds : RegionBounds) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) :
    Stmt.TraceSafeList bounds (reductionBody Y X M N K B C) s := by
  simp [reductionBody, softmax_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  apply RealTrace.of_all
  intro statement hstatement st
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at hstatement
  rcases hstatement with rfl | rfl | rfl
  all_goals simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem previous_safe (bounds : RegionBounds) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) :
    Stmt.TraceSafeList bounds (previousBody Y X M N K B C) s := by
  simp [previousBody, softmax_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  apply RealTrace.of_all
  intro statement hstatement st
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at hstatement
  rcases hstatement with rfl
  all_goals simp [Stmt.TraceSafe, Op.SafeAt.eq_def]


theorem firstPass_safe (bounds : RegionBounds) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B)
    (hr : s.pids 0 < M) (hX : M * N * K ≤ bounds X) :
    Stmt.TraceSafeList bounds (firstPass Y X M N K B C) s := by
  obtain ⟨st, hexec, hinv⟩ := prefix_run s Y X M N K B C
  rw [firstPass_eq]
  apply RealTrace.append bounds _ _ s st (prefix_safe bounds s Y X M N K B C) hexec
  apply RealTrace.singleton
  rw [Stmt.TraceSafe]
  exact pass1_loop_safe bounds s st Y X M (s.pids 0) N K (s.pids 1 * C) B C hN hB hr hX hinv

theorem prepare_safe (bounds : RegionBounds) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B)
    (hr : s.pids 0 < M) (hX : M * N * K ≤ bounds X) :
    Stmt.TraceSafeList bounds (prepareBody Y X M N K B C) s := by
  obtain ⟨st, hexec, _⟩ := firstPass_run s Y X M N K B C hN hB
  exact RealTrace.append bounds _ _ s st (firstPass_safe bounds s Y X M N K B C hN hB hr hX)
    hexec (reduction_safe bounds st Y X M N K B C)

theorem linear_safe (bounds : RegionBounds) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (hr : s.pids 0 < M) (hX : M * N * K ≤ bounds X) (hY : M * N * K ≤ bounds Y) :
    Stmt.TraceSafeList bounds (linearBody Y X M N K B C) s := by
  obtain ⟨prepared, hexec, hinv⟩ := prepare_run s Y X M N K B C hN hB
  obtain ⟨initialized, hprevious, hinit⟩ :=
    previous_run s prepared Y X M (s.pids 0) N K (s.pids 1 * C) B C hB hinv
  rw [linearBody, List.append_assoc]
  apply RealTrace.append bounds _ _ s prepared
    (prepare_safe bounds s Y X M N K B C hN hB hr hX) hexec
  apply RealTrace.append bounds _ _ prepared initialized
    (previous_safe bounds prepared Y X M N K B C) hprevious
  apply RealTrace.singleton
  rw [Stmt.TraceSafe]
  exact pass2_loop_safe bounds s initialized Y X M (s.pids 0) N K (s.pids 1 * C) B C
    hN hB hXY hr hX hY hinit

theorem source_safe (bounds : RegionBounds) (s : BlockState) (Y X : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y)
    (hr : s.pids 0 < M) (hX : M * N * K ≤ bounds X) (hY : M * N * K ≤ bounds Y) :
    (softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨pre, yes, no, hsource, hlinear⟩ := source_decomposition Y X M N K B C
  unfold Kernel.TraceSafe
  rw [hsource, RealTrace.false_branch, ← hlinear]
  exact linear_safe bounds s Y X M N K B C hN hB hXY hr hX hY

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
