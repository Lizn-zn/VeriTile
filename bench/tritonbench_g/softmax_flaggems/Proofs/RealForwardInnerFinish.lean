import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerWriteLoop

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 1200000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def previousBody (Y X : RegionName) (M N B : Nat) : List Stmt :=
  match ((softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.false).eraseDType.toAlgKernel.body[1]? : Option Stmt) with
  | some (.ifThenElse _ _ no) => (no.drop 10).take 1
  | _ => []

theorem previous_run (sb st : BlockState) (Y X : RegionName)
    (M row N B : Nat) (hB : 0 < B)
    (h : ReducedInv sb X Y row N B hB st) :
    ∃ next, stepStmts (previousBody Y X M N B) st = some next ∧
      Pass2Inv sb Y X row N B hB 0 next := by
  simp [previousBody, softmax_kernel_inner,
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
  · simpa using h.inputReg
  · simpa using h.outputReg
  · simp only [BlockState.setReg_same]
    rfl
  · simpa using h.maxReg
  · simpa using h.sumReg
  · intro n hn hq
    have hlt := RealAddress.active_tile_lt hB hn
    omega

/-- Successful full two-pass branch after splicing the constant-false source branch. -/
def linearBody (Y X : RegionName) (M N B : Nat) : List Stmt :=
  prepareBody Y X M N B ++ previousBody Y X M N B ++
    [.forRange "start_n" 0 B B (pass2Body Y X M N B),
     .forRange "start_n" B N B (fullStoreBody Y X M N B)]

theorem linear_run (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y) :
    ∃ next, stepStmts (linearBody Y X M N B) s = some next ∧
      Pass2Inv s Y X (s.pids 0) N B hB (RealAddress.rounds N B) next := by
  obtain ⟨prepared, hprepare, hinv⟩ := prepare_run s Y X M N B hN hB
  obtain ⟨initialized, hprevious, hinit⟩ :=
    previous_run s prepared Y X M (s.pids 0) N B hB hinv
  obtain ⟨mid, hmasked, hmid⟩ :=
    masked_write_loop s initialized Y X M (s.pids 0) N B hN hB hXY hinit
  obtain ⟨next, hloop, hnext⟩ :=
    full_write_loop s mid Y X M (s.pids 0) N B hN hB hXY hmid
  refine ⟨next, ?_, hnext⟩
  simp only [linearBody, ProgramRewrite.append, hprepare, hprevious, Option.bind_some,
    stepStmts, stepStmt, hmasked, hloop]

/-- Exact syntactic decomposition of the source and its selected branch. -/
theorem source_decomposition (Y X : RegionName) (M N B : Nat) :
    ∃ pre yes no : List Stmt,
      (softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.false).eraseDType.toAlgKernel.body =
        pre ++ [.ifThenElse (.constBool Bool.false) yes no] ∧
      linearBody Y X M N B = pre ++ no := by
  let ast := (softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.false).eraseDType.toAlgKernel
  let pre := ast.body.take 1
  let yes := match (ast.body[1]? : Option Stmt) with
    | some (.ifThenElse _ yes _) => yes
    | _ => []
  let no := match (ast.body[1]? : Option Stmt) with
    | some (.ifThenElse _ _ no) => no
    | _ => []
  have hsource : ast.body = pre ++ [.ifThenElse (.constBool Bool.false) yes no] := by
    simp [ast, pre, yes, no, source, fallback, softmax_kernel_inner,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  have hfirst : firstPass Y X M N B = pre ++ no.take 7 := rfl
  have hlinear : linearBody Y X M N B = pre ++ no := by
    unfold linearBody prepareBody
    rw [hfirst]
    unfold reductionBody previousBody pass2Body fullStoreBody
    simp [ast, pre, yes, no, source, fallback, softmax_kernel_inner,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  exact ⟨pre, yes, no, hsource, hlinear⟩

/-- The linearized proof program is the actual source with its static branch selected. -/
theorem source_linear (s : BlockState) (Y X : RegionName) (M N B : Nat) :
    stepStmts ((softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.false).eraseDType.toAlgKernel.body) s = stepStmts (linearBody Y X M N B) s := by
  obtain ⟨pre, yes, no, hsource, hlinear⟩ := source_decomposition Y X M N B
  rw [hsource, hlinear]
  rw [ProgramRewrite.append, ProgramRewrite.append]
  cases stepStmts pre s with
  | none => rfl
  | some st =>
    simp only [Option.bind_some, stepStmts, stepStmt, evalOp.eq_def,
      Tile.scalar_data, Bool.false_eq_true, ↓reduceIte,
      Option.bind_eq_bind, Option.bind, Option.pure_def]
    cases stepStmts no st <;> rfl

theorem source_run (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hN : 0 < N) (hB : 0 < B) (hXY : X ≠ Y) :
    ∃ next, stepStmts ((softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.false).eraseDType.toAlgKernel.body) s = some next ∧
      Pass2Inv s Y X (s.pids 0) N B hB (RealAddress.rounds N B) next := by
  rw [source_linear]
  exact linear_run s Y X M N B hN hB hXY

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
