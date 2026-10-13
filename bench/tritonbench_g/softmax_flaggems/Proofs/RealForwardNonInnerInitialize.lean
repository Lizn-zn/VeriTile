import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerWriteback

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 1200000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def previousBody (Y X : RegionName) (M N K B C : Nat) : List Stmt :=
  match ((softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.body[3]? : Option Stmt) with
  | some (.ifThenElse _ _ no) => (no.drop 6).take 1
  | _ => []

theorem previous_run (sb st : BlockState) (Y X : RegionName)
    (M row N K kbase B C : Nat) (hB : 0 < B)
    (h : ReducedInv sb X row N K kbase B C hB st) :
    ∃ next, stepStmts (previousBody Y X M N K B C) st = some next ∧
      Pass2Inv sb Y X row N K kbase B C hB 0 next := by
  simp [previousBody, softmax_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def,
    Option.bind_eq_bind, Option.bind, Option.pure_def]
  refine ⟨_, rfl, ?_⟩
  constructor
  · intro r o _
    exact (congrFun (congrFun h.memory r) o).symm
  · intro o
    exact congrFun (congrFun h.memory X) o
  · exact h.pids
  · exact h.numPids
  · exact h.undef
  · simpa using h.rowReg
  · simpa using h.columnReg
  · simp only [BlockState.setReg_same]
    rfl
  · simpa using h.maxReg
  · simpa using h.sumReg
  · intro n k hn _ _ hq
    have hlt := RealAddress.active_tile_lt hB hn
    omega

/-- Successful full two-pass branch after splicing the constant-false source branch. -/
def linearBody (Y X : RegionName) (M N K B C : Nat) : List Stmt :=
  prepareBody Y X M N K B C ++ previousBody Y X M N K B C ++
    [.forRange "start_n" 0 N B (pass2Body Y X M N K B C)]

theorem linear_run (s : BlockState) (Y X : RegionName) (M N K B C : Nat)
    (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y) :
    ∃ next, stepStmts (linearBody Y X M N K B C) s = some next ∧
      Pass2Inv s Y X (s.pids 0) N K (s.pids 1 * C) B C hB (RealAddress.rounds N B) next := by
  obtain ⟨prepared, hprepare, hinv⟩ := prepare_run s Y X M N K B C hN hB
  obtain ⟨initialized, hprevious, hinit⟩ :=
    previous_run s prepared Y X M (s.pids 0) N K (s.pids 1 * C) B C hB hinv
  obtain ⟨next, hloop, hnext⟩ :=
    pass2_loop s initialized Y X M (s.pids 0) N K (s.pids 1 * C) B C hN hB hXY hinit
  refine ⟨next, ?_, hnext⟩
  simp only [linearBody, ProgramRewrite.append, hprepare, hprevious, Option.bind_some,
    stepStmts, stepStmt, hloop]

/-- Exact syntactic decomposition of the source and its selected branch. -/
theorem source_decomposition (Y X : RegionName) (M N K B C : Nat) :
    ∃ pre yes no : List Stmt,
      (softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.body =
        pre ++ [.ifThenElse (.constBool Bool.false) yes no] ∧
      linearBody Y X M N K B C = pre ++ no := by
  let ast := (softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel
  let pre := ast.body.take 3
  let yes := match (ast.body[3]? : Option Stmt) with
    | some (.ifThenElse _ yes _) => yes
    | _ => []
  let no := match (ast.body[3]? : Option Stmt) with
    | some (.ifThenElse _ _ no) => no
    | _ => []
  have hsource : ast.body = pre ++ [.ifThenElse (.constBool Bool.false) yes no] := by
    simp [ast, pre, yes, no, softmax_kernel_non_inner,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  have hfirst : firstPass Y X M N K B C = pre ++ no.take 3 := rfl
  have hlinear : linearBody Y X M N K B C = pre ++ no := by
    unfold linearBody prepareBody
    rw [hfirst]
    unfold reductionBody previousBody pass2Body
    simp [ast, pre, yes, no, softmax_kernel_non_inner,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  exact ⟨pre, yes, no, hsource, hlinear⟩

/-- The linearized proof program is the actual source with its static branch selected. -/
theorem source_linear (s : BlockState) (Y X : RegionName) (M N K B C : Nat) :
    stepStmts ((softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.body) s = stepStmts (linearBody Y X M N K B C) s := by
  obtain ⟨pre, yes, no, hsource, hlinear⟩ := source_decomposition Y X M N K B C
  rw [hsource, hlinear]
  rw [ProgramRewrite.append, ProgramRewrite.append]
  cases stepStmts pre s with
  | none => rfl
  | some st =>
    simp only [Option.bind_some, stepStmts, stepStmt, evalOp.eq_def,
      Tile.scalar_data, Bool.false_eq_true, ↓reduceIte,
      Option.bind_eq_bind, Option.bind, Option.pure_def]
    cases stepStmts no st <;> rfl

theorem source_run (s : BlockState) (Y X : RegionName) (M N K B C : Nat)
    (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y) :
    ∃ next, stepStmts ((softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.body) s = some next ∧
      Pass2Inv s Y X (s.pids 0) N K (s.pids 1 * C) B C hB (RealAddress.rounds N B) next := by
  rw [source_linear]
  exact linear_run s Y X M N K B C hN hB hXY

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
