import bench.tritonbench_g.nested_loops_processing.Proofs.PayloadMiddle
import bench.tritonbench_g.nested_loops_processing.Proofs.RealOuter

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal
open VeriTile.Triton FullReal
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem middleLoop_registers (dtype : TileDType) (s : BlockState) (a c : PointerTile) (v1 : Payload dtype) (delta : Nat)
    (_h1 : s.regs dtype [2,2] "a1" = some ⟨v1⟩) :
    (middleLoopState dtype s a c v1 delta).regs .ptr [2,2] "a_ptrs" = some ⟨advances 6 a delta⟩ ∧
    (middleLoopState dtype s a c v1 delta).regs .ptr [2,2] "c_ptrs" = some ⟨advances 12 c delta⟩ := by
  simp [middleLoopState, middleState, middlePrefixState, innerLoopState, innerState,
    scatter, advances]

def outerPrefix (dtype : TileDType) (src dst : RegionName) (sm sn : Int) : List Stmt :=
  (PayloadAlgorithm.outer dtype src dst sm sn).take 1

def outerSuffix (dtype : TileDType) (src dst : RegionName) (sm sn : Int) : List Stmt :=
  (PayloadAlgorithm.outer dtype src dst sm sn).drop 2

noncomputable def outerState (dtype : TileDType) (s : BlockState) (a c : PointerTile) (delta : Nat) : BlockState :=
  let t := s.setReg "a1" dtype [2,2] ⟨loaded dtype s a⟩
  let t := middleLoopState dtype t a c (loaded dtype s a) delta
  t.setReg "a_ptrs" .ptr [2,2] ⟨advances 7 a delta⟩

theorem outer_decomposition (dtype : TileDType) (src dst : RegionName) (sm sn : Int) :
    PayloadAlgorithm.outer dtype src dst sm sn = outerPrefix dtype src dst sm sn ++
      [.forRange "j" 0 2 1 (PayloadAlgorithm.middle dtype src dst sm sn)] ++ outerSuffix dtype src dst sm sn := by
  rfl

theorem outer_prefix_step (dtype : TileDType) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩) :
    stepStmts (outerPrefix dtype src dst sm sn) s =
      some (s.setReg "a1" dtype [2,2] ⟨loaded dtype s a⟩) := by
  simp [outerPrefix, PayloadAlgorithm.outer, PayloadAlgorithm.load,
    stepStmts, stepStmt, evalOp.eq_def, ha, loaded]
  all_goals rfl

theorem outer_suffix_step (dtype : TileDType) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩) :
    stepStmts (outerSuffix dtype src dst sm sn) s =
      some (s.setReg "a_ptrs" .ptr [2,2] ⟨bump a (2 * sn).toNat⟩) := by
  simp [outerSuffix, PayloadAlgorithm.outer, PayloadAlgorithm.inputAdvance,
    FullReal.inner, FullReal.middle, FullReal.outer, FullReal.source, TypedSource.kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp [stepStmts, stepStmt, evalOp.eq_def, ha, bump, Tile.ptrAdd, Tile.bop,
    Tile.uop, NumericDType.mul]
  all_goals rfl

theorem outer_step (dtype : TileDType) (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩) :
    stepStmts (PayloadAlgorithm.outer dtype src dst sm sn) s = some (outerState dtype s a c (2 * sn).toNat) := by
  have hp := outer_prefix_step dtype src dst sm sn s a ha
  have hl := middle_loop dtype src dst sm sn (s.setReg "a1" dtype [2,2] ⟨loaded dtype s a⟩)
    a c (loaded dtype s a) (by simpa using ha) (by simpa using hc) (by simp)
  have regs := middleLoop_registers dtype (s.setReg "a1" dtype [2,2] ⟨loaded dtype s a⟩)
    a c (loaded dtype s a) (2 * sn).toNat (by simp)
  have hs := outer_suffix_step dtype src dst sm sn _ _ regs.1
  rw [outer_decomposition]
  simp only [ProgramRewrite.append, hp, stepStmts, stepStmt, hl, Option.bind_some]
  exact hs

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.PayloadReal

#guard_msgs (drop info) in
#auditModuleAxioms
