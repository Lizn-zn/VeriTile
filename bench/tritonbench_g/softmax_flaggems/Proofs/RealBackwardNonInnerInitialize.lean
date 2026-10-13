import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerWriteback

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem reduced_pass2 (sb st : BlockState) (P G Y : RegionName)
    (row N K kbase B C : Nat) (h : ReducedInv sb P G row N K kbase B C st) :
    Pass2Inv sb P G Y row N K kbase B C 0 st := by
  constructor
  · intro r o _
    exact (congrFun (congrFun h.memory r) o).symm
  · intro o
    exact congrFun (congrFun h.memory P) o
  · intro o
    exact congrFun (congrFun h.memory G) o
  · exact h.pids
  · exact h.numPids
  · exact h.undef
  · exact h.columnReg
  · simpa using h.nReg
  · simpa using h.offsetReg
  · exact h.sumReg
  · intro n k hn hkC hkK hlt
    exact (Nat.not_lt_zero _ hlt).elim

def linearBody (P G Y : RegionName) (M N K B C : Nat) : List Stmt :=
  prepareBody P G Y M N K B C ++
    [.forRange "unused_start_n" 0 N B (pass2Body P G Y M N K B C)]

theorem linear_run (s : BlockState) (P G Y : RegionName) (M N K B C : Nat)
    (hN : 0<N) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y) :
    ∃ next, stepStmts (linearBody P G Y M N K B C) s = some next ∧
      Pass2Inv s P G Y (s.pids 0) N K (s.pids 1*C) B C (RealAddress.rounds N B) next := by
  obtain ⟨st, he, hi⟩ := prepare_run s P G Y M N K B C hN hB
  obtain ⟨next, hl, hn⟩ := pass2_loop s st P G Y M (s.pids 0) N K (s.pids 1*C)
    B C hN hB hPY hGY (reduced_pass2 s st P G Y _ _ _ _ _ _ hi)
  refine ⟨next, ?_, hn⟩
  simp only [linearBody, ProgramRewrite.append, he, Option.bind_some,
    stepStmts, stepStmt, hl]

/-- Exact syntactic decomposition of the source and its selected branch. -/
theorem source_decomposition (P G Y : RegionName) (M N K B C : Nat) :
    ∃ pre yes no : List Stmt,
      (softmax_backward_kernel_non_inner .fp32 P G Y M N K B C Bool.false).eraseDType.toAlgKernel.body =
        pre ++ [.ifThenElse (.constBool Bool.false) yes no] ∧
      linearBody P G Y M N K B C = pre ++ no := by
  let ast := (softmax_backward_kernel_non_inner .fp32 P G Y M N K B C Bool.false).eraseDType.toAlgKernel
  let pre := ast.body.take 3
  let yes := match (ast.body[3]? : Option Stmt) with
    | some (.ifThenElse _ yes _) => yes
    | _ => []
  let no := match (ast.body[3]? : Option Stmt) with
    | some (.ifThenElse _ _ no) => no
    | _ => []
  have hsource : ast.body = pre ++ [.ifThenElse (.constBool Bool.false) yes no] := by
    simp [ast, pre, yes, no, softmax_backward_kernel_non_inner,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  have hfirst : firstPass P G Y M N K B C = pre ++ no.take 4 := rfl
  have hlinear : linearBody P G Y M N K B C = pre ++ no := by
    unfold linearBody prepareBody
    rw [hfirst]
    unfold reductionBody pass2Body branch source
    simp [ast, pre, yes, no, softmax_backward_kernel_non_inner,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  exact ⟨pre, yes, no, hsource, hlinear⟩

/-- The linearized proof program is the actual source with its static branch selected. -/
theorem source_linear (s : BlockState) (P G Y : RegionName) (M N K B C : Nat) :
    stepStmts ((softmax_backward_kernel_non_inner .fp32 P G Y M N K B C Bool.false).eraseDType.toAlgKernel.body) s = stepStmts (linearBody P G Y M N K B C) s := by
  obtain ⟨pre, yes, no, hsource, hlinear⟩ := source_decomposition P G Y M N K B C
  rw [hsource, hlinear]
  rw [ProgramRewrite.append, ProgramRewrite.append]
  cases stepStmts pre s with
  | none => rfl
  | some st =>
    simp only [Option.bind_some, stepStmts, stepStmt, evalOp.eq_def,
      Tile.scalar_data, Bool.false_eq_true, ↓reduceIte,
      Option.bind_eq_bind, Option.bind, Option.pure_def]
    cases stepStmts no st <;> rfl

theorem source_run (s : BlockState) (P G Y : RegionName) (M N K B C : Nat)
    (hN : 0<N) (hB : 0<B) (hPY : P≠Y) (hGY : G≠Y) :
    ∃ next, stepStmts (source P G Y M N K B C Bool.false).body s = some next ∧
      Pass2Inv s P G Y (s.pids 0) N K (s.pids 1*C) B C (RealAddress.rounds N B) next := by
  rw [source, source_linear]
  exact linear_run s P G Y M N K B C hN hB hPY hGY

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
