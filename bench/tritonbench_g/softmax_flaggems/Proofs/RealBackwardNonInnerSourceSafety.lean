import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerInitialize
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerLoopSafety
import bench.tritonbench_g.softmax_flaggems.Proofs.RealTrace

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem prefix_safe (bounds : RegionBounds) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) :
    Stmt.TraceSafeList bounds (pass1Prefix P G Y M N K B C) s := by
  simp [pass1Prefix, branch, source, softmax_backward_kernel_non_inner,
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
  rcases hstatement with rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp [Stmt.TraceSafe, Op.SafeAt.eq_def]

theorem reduction_safe (bounds : RegionBounds) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) :
    Stmt.TraceSafeList bounds (reductionBody P G Y M N K B C) s := by
  simp [reductionBody, branch, source, softmax_backward_kernel_non_inner,
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

theorem firstPass_safe (bounds : RegionBounds) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B)
    (hr : s.pids 0 < M) (hP : M * N * K ≤ bounds P) (hG : M * N * K ≤ bounds G) :
    Stmt.TraceSafeList bounds (firstPass P G Y M N K B C) s := by
  obtain ⟨st, hexec, hinv⟩ := prefix_run s P G Y M N K B C
  rw [firstPass_eq]
  apply RealTrace.append bounds _ _ s st (prefix_safe bounds s P G Y M N K B C) hexec
  apply RealTrace.singleton
  rw [Stmt.TraceSafe]
  exact pass1_loop_safe bounds s st P G Y M (s.pids 0) N K (s.pids 1 * C) B C hN hB hr hP hG hinv

theorem prepare_safe (bounds : RegionBounds) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B)
    (hr : s.pids 0 < M) (hP : M * N * K ≤ bounds P) (hG : M * N * K ≤ bounds G) :
    Stmt.TraceSafeList bounds (prepareBody P G Y M N K B C) s := by
  obtain ⟨st, hexec, _⟩ := firstPass_run s P G Y M N K B C hN hB
  exact RealTrace.append bounds _ _ s st (firstPass_safe bounds s P G Y M N K B C hN hB hr hP hG)
    hexec (reduction_safe bounds st P G Y M N K B C)

theorem linear_safe (bounds : RegionBounds) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hPY : P ≠ Y) (hGY : G ≠ Y)
    (hr : s.pids 0 < M) (hP : M * N * K ≤ bounds P) (hG : M * N * K ≤ bounds G) (hY : M * N * K ≤ bounds Y) :
    Stmt.TraceSafeList bounds (linearBody P G Y M N K B C) s := by
  obtain ⟨prepared, hexec, hinv⟩ := prepare_run s P G Y M N K B C hN hB
  rw [linearBody]
  apply RealTrace.append bounds _ _ s prepared
    (prepare_safe bounds s P G Y M N K B C hN hB hr hP hG) hexec
  apply RealTrace.singleton
  rw [Stmt.TraceSafe]
  exact pass2_loop_safe bounds s prepared P G Y M (s.pids 0) N K (s.pids 1*C) B C
    hN hB hPY hGY hr hP hG hY (reduced_pass2 s prepared P G Y _ _ _ _ _ _ hinv)

theorem source_safe (bounds : RegionBounds) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (hN : 0 < N) (hB : 0 < B) (hPY : P ≠ Y) (hGY : G ≠ Y)
    (hr : s.pids 0 < M) (hP : M * N * K ≤ bounds P) (hG : M * N * K ≤ bounds G) (hY : M * N * K ≤ bounds Y) :
    (softmax_backward_kernel_non_inner .fp32 P G Y M N K B C Bool.false).eraseDType.toAlgKernel.TraceSafe bounds s := by
  obtain ⟨pre, yes, no, hsource, hlinear⟩ := source_decomposition P G Y M N K B C
  unfold Kernel.TraceSafe
  rw [hsource, RealTrace.false_branch, ← hlinear]
  exact linear_safe bounds s P G Y M N K B C hN hB hPY hGY hr hP hG hY

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
