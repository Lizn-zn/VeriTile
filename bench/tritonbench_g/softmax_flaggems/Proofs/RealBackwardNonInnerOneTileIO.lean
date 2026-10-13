import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerOneTile
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerIO

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem oneSum_eq (sb : BlockState) (P G : RegionName)
    (row N K kbase B C : Nat) (k : Fin C) :
    oneSum sb P G row N K kbase B C k =
      ∑ n ∈ Finset.range B, sample sb P row N K n (kbase+k.val) *
        sample sb G row N K n (kbase+k.val) := by
  exact (Finset.sum_range (n := B) (fun j ↦
    sample sb P row N K j (kbase+k.val) * sample sb G row N K j (kbase+k.val))).symm

def oneLinear (P G Y : RegionName) (M N K B C : Nat) : List Stmt :=
  onePrefix P G Y M N K B C ++ oneTail P G Y M N K B C

theorem one_decomposition (P G Y : RegionName) (M N K B C : Nat) :
    ∃ pre yes no : List Stmt,
      (source P G Y M N K B C Bool.true).body =
        pre ++ [.ifThenElse (.constBool Bool.true) yes no] ∧
      oneLinear P G Y M N K B C = pre ++ yes := by
  let ast := source P G Y M N K B C Bool.true
  let pre := ast.body.take 3
  let yes := branch P G Y M N K B C Bool.true
  let no := match (ast.body[3]? : Option Stmt) with
    | some (.ifThenElse _ _ no) => no
    | _ => []
  refine ⟨pre, yes, no, ?_, ?_⟩
  · simp [ast, pre, yes, no, source, branch, softmax_backward_kernel_non_inner,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  · simp only [oneLinear, onePrefix, oneTail, List.append_assoc, List.take_append_drop]
    rfl

theorem one_source_linear (s : BlockState) (P G Y : RegionName) (M N K B C : Nat) :
    stepStmts (source P G Y M N K B C Bool.true).body s =
      stepStmts (oneLinear P G Y M N K B C) s := by
  obtain ⟨pre, yes, no, hs, hl⟩ := one_decomposition P G Y M N K B C
  rw [hs, hl, ProgramRewrite.append, ProgramRewrite.append]
  cases stepStmts pre s with
  | none => rfl
  | some st =>
    simp only [Option.bind_some, stepStmts, stepStmt, evalOp.eq_def,
      Tile.scalar_data, ↓reduceIte, Option.bind_eq_bind, Option.bind, Option.pure_def]
    cases stepStmts yes st <;> rfl

theorem one_original_io (storage : Storage) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (hNB : N≤B) :
    ∃ final, exec (softmax_backward_kernel_non_inner storage P G Y M N K B C Bool.true).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1*C) C) s final ∧
      (∀ n k, n<N → k<C → s.pids 1*C+k<K →
        final.readMem Y (offset (s.pids 0) N K n (s.pids 1*C+k)) =
          expected s P G (s.pids 0) N K B n (s.pids 1*C+k)) ∧
      final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef := by
  rw [RealProjection.softmax_backward_kernel_non_inner_erased]
  change ∃ final, stepStmts (source P G Y M N K B C Bool.true).body s = some final ∧ _
  rw [one_source_linear]
  obtain ⟨st, hp, hi⟩ := onePrefix_run s P G Y M N K B C
  let z := oneSum s P G (s.pids 0) N K (s.pids 1*C) B C
  let value : TileIndex [B,C] → ℝ := storeValue s P G (s.pids 0) N K (s.pids 1*C) 0 z
  let mid := st.setReg "in_grad_tile" .real [B,C] ⟨fun i ↦ some (value i)⟩
  let final := scatter mid Y (tileOffset (s.pids 0) N K (s.pids 1*C) 0) value
    (tileActive N K (s.pids 1*C) 0)
  have ht := oneTail_run s st P G Y M (s.pids 0) N K (s.pids 1*C) B C hi
  have hf : BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1*C) C) mid final := by
    apply RealForwardNonInner.scatter_frame
    intro i hactive
    have ha : 0+i.1.val<N ∧ s.pids 1*C+i.2.1.val<K := of_decide_eq_true hactive
    exact ⟨rfl, 0+i.1.val, i.2.1.val, ha.1, i.2.1.isLt, ha.2, rfl⟩
  have hl := RealForwardNonInner.scatter_launch_padding mid Y
    (tileOffset (s.pids 0) N K (s.pids 1*C) 0) value (tileActive N K (s.pids 1*C) 0)
  refine ⟨final, ?_, ?_, ?_, ?_, hl.1.trans hi.numPids, hl.2.trans hi.undef⟩
  · rw [oneLinear, ProgramRewrite.append, hp]
    exact ht
  · intro r o ho
    exact (congrFun (congrFun hi.memory r) o).symm.trans (hf r o ho)
  · intro n k hn hkC hkK
    let i : TileIndex [B,C] := (⟨n, lt_of_lt_of_le hn hNB⟩, ⟨k,hkC⟩, PUnit.unit)
    have ha : tileActive N K (s.pids 1*C) 0 i = Bool.true := by
      apply decide_eq_true
      exact ⟨by simpa [i] using hn, hkK⟩
    have hv := RealForwardNonInner.scatter_value mid Y (s.pids 0) N K (s.pids 1*C) 0 value i ha
    have hformula : value i = expected s P G (s.pids 0) N K B n (s.pids 1*C+k) := by
      simp only [value, storeValue, tileSample, z, oneSum_eq, i, Nat.zero_add,
        expected, RealMath.backwardPadded]
    simpa only [final, RealForwardNonInner.tileOffset, tileOffset, offset, i, Nat.zero_add]
      using hv.trans hformula
  · exact (RealForwardNonInner.scatter_pids mid Y _ value _).trans hi.pids

theorem one_candidate_io (storage : Storage) (s : BlockState) (P G Y : RegionName)
    (M N K B C : Nat) (hNB : N≤B) :
    ∃ final, exec (softmax_backward_kernel_non_inner_candidate storage P G Y M N K B C Bool.true).eraseDType.toAlgKernel s = some final ∧
      BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1*C) C) s final ∧
      (∀ n k, n<N → k<C → s.pids 1*C+k<K →
        final.readMem Y (offset (s.pids 0) N K n (s.pids 1*C+k)) =
          expected s P G (s.pids 0) N K B n (s.pids 1*C+k)) ∧
      final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef := by
  rw [← candidate_exec]
  exact one_original_io storage s P G Y M N K B C hNB

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
