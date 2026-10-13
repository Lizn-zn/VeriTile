import bench.tritonbench_g.nested_loops_processing.Proofs.RealMiddle
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def advances : Nat → PointerTile → Nat → PointerTile
  | 0, p, _ => p
  | n + 1, p, delta => bump (advances n p delta) delta

theorem middleLoop_registers (s : BlockState) (a c : PointerTile) (v1 : IntTile) (delta : Nat)
    (_h1 : s.regs .int [2,2] "a1" = some ⟨v1⟩) :
    (middleLoopState s a c v1 delta).regs .ptr [2,2] "a_ptrs" = some ⟨advances 6 a delta⟩ ∧
    (middleLoopState s a c v1 delta).regs .ptr [2,2] "c_ptrs" = some ⟨advances 12 c delta⟩ := by
  simp [middleLoopState, middleState, middlePrefixState, innerLoopState, innerState,
    scatter, advances]

def outerPrefix (src dst : RegionName) (sm sn : Int) : List Stmt :=
  (outer src dst sm sn).take 1

def outerSuffix (src dst : RegionName) (sm sn : Int) : List Stmt :=
  (outer src dst sm sn).drop 2

noncomputable def outerState (s : BlockState) (a c : PointerTile) (delta : Nat) : BlockState :=
  let t := s.setReg "a1" .int [2,2] ⟨loaded s a⟩
  let t := middleLoopState t a c (loaded s a) delta
  t.setReg "a_ptrs" .ptr [2,2] ⟨advances 7 a delta⟩

theorem outer_decomposition (src dst : RegionName) (sm sn : Int) :
    outer src dst sm sn = outerPrefix src dst sm sn ++
      [.forRange "j" 0 2 1 (middle src dst sm sn)] ++ outerSuffix src dst sm sn := by
  simp [outerPrefix, outerSuffix, outer, middle, source, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem outer_prefix_step (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩) :
    stepStmts (outerPrefix src dst sm sn) s =
      some (s.setReg "a1" .int [2,2] ⟨loaded s a⟩) := by
  simp [outerPrefix, outer, source, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp [stepStmts, stepStmt, evalOp.eq_def, ha, loaded]
  all_goals rfl

theorem outer_suffix_step (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩) :
    stepStmts (outerSuffix src dst sm sn) s =
      some (s.setReg "a_ptrs" .ptr [2,2] ⟨bump a (2 * sn).toNat⟩) := by
  simp [outerSuffix, outer, source, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp [stepStmts, stepStmt, evalOp.eq_def, ha, bump, Tile.ptrAdd, Tile.bop,
    Tile.uop, NumericDType.mul]
  all_goals rfl

theorem outer_step (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩) :
    stepStmts (outer src dst sm sn) s = some (outerState s a c (2 * sn).toNat) := by
  have hp := outer_prefix_step src dst sm sn s a ha
  have hl := middle_loop src dst sm sn (s.setReg "a1" .int [2,2] ⟨loaded s a⟩)
    a c (loaded s a) (by simpa using ha) (by simpa using hc) (by simp)
  have regs := middleLoop_registers (s.setReg "a1" .int [2,2] ⟨loaded s a⟩)
    a c (loaded s a) (2 * sn).toNat (by simp)
  have hs := outer_suffix_step src dst sm sn _ _ regs.1
  rw [outer_decomposition]
  simp only [ProgramRewrite.append, hp, stepStmts, stepStmt, hl, Option.bind_some]
  exact hs

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
