import bench.tritonbench_g.nested_loops_processing.Proofs.RealLoop

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def middlePrefix (src dst : RegionName) (sm sn : Int) : List Stmt :=
  (middle src dst sm sn).take 2

noncomputable def middlePrefixState (s : BlockState) (a : PointerTile) (delta : Nat) : BlockState :=
  let t := s.setReg "a_ptrs" .ptr [2,2] ⟨bump a delta⟩
  t.setReg "a2" .int [2,2] ⟨loaded s (bump a delta)⟩

noncomputable def middleState (s : BlockState) (a c : PointerTile)
    (v1 : IntTile) (delta : Nat) : BlockState :=
  innerLoopState (middlePrefixState s a delta) (bump a delta) c v1
    (loaded s (bump a delta)) delta

theorem middle_decomposition (src dst : RegionName) (sm sn : Int) :
    middle src dst sm sn = middlePrefix src dst sm sn ++
      [.forRange "k" 0 2 1 (inner src dst sm sn)] := by
  simp [middlePrefix, middle, outer, inner, source, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem middle_prefix_step (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a : PointerTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩) :
    stepStmts (middlePrefix src dst sm sn) s =
      some (middlePrefixState s a (2 * sn).toNat) := by
  simp [middlePrefix, middle, outer, source, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp [stepStmts, stepStmt, evalOp.eq_def, ha, middlePrefixState, loaded, bump,
    Tile.bop, Tile.uop, Tile.vec, Tile.ptrAdd, NumericDType.mul]
  rfl

theorem middle_step (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile) (v1 : IntTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (h1 : s.regs .int [2,2] "a1" = some ⟨v1⟩) :
    stepStmts (middle src dst sm sn) s = some (middleState s a c v1 (2 * sn).toNat) := by
  rw [middle_decomposition, stepStmts.append_some (middle_prefix_step src dst sm sn s a ha)]
  have h := inner_loop src dst sm sn (middlePrefixState s a (2 * sn).toNat)
    (bump a (2 * sn).toNat) c v1 (loaded s (bump a (2 * sn).toNat))
    (by simp [middlePrefixState]) (by simpa [middlePrefixState] using hc)
    (by simpa [middlePrefixState] using h1) (by simp [middlePrefixState])
  simpa only [stepStmts, stepStmt, h, middleState] using h

theorem middle_registers (s : BlockState) (a c : PointerTile) (v1 : IntTile) (delta : Nat)
    (h1 : s.regs .int [2,2] "a1" = some ⟨v1⟩) :
    (middleState s a c v1 delta).regs .ptr [2,2] "a_ptrs" =
      some ⟨bump (bump (bump a delta) delta) delta⟩ ∧
    (middleState s a c v1 delta).regs .ptr [2,2] "c_ptrs" =
      some ⟨bump (bump (bump (bump (bump (bump c delta) delta) delta) delta) delta) delta⟩ ∧
    (middleState s a c v1 delta).regs .int [2,2] "a1" = some ⟨v1⟩ := by
  simp [middleState, middlePrefixState, innerLoopState, innerState, scatter, h1]

noncomputable def middleLoopState (s : BlockState) (a c : PointerTile)
    (v1 : IntTile) (delta : Nat) : BlockState :=
  let t := middleState (s.setReg "j" .nat [] (Tile.scalar 0)) a c v1 delta
  middleState (t.setReg "j" .nat [] (Tile.scalar 1))
    (bump (bump (bump a delta) delta) delta)
    (bump (bump (bump (bump (bump (bump c delta) delta) delta) delta) delta) delta) v1 delta

theorem middle_loop (src dst : RegionName) (sm sn : Int)
    (s : BlockState) (a c : PointerTile) (v1 : IntTile)
    (ha : s.regs .ptr [2,2] "a_ptrs" = some ⟨a⟩)
    (hc : s.regs .ptr [2,2] "c_ptrs" = some ⟨c⟩)
    (h1 : s.regs .int [2,2] "a1" = some ⟨v1⟩) :
    stepForRangeAux "j" 0 2 1 (middle src dst sm sn) s =
      some (middleLoopState s a c v1 (2 * sn).toNat) := by
  rw [range_two]
  have e0 := middle_step src dst sm sn (s.setReg "j" .nat [] (Tile.scalar 0)) a c v1
    (by simpa using ha) (by simpa using hc) (by simpa using h1)
  rw [e0]
  have regs := middle_registers (s.setReg "j" .nat [] (Tile.scalar 0)) a c v1
    (2 * sn).toNat (by simpa using h1)
  exact middle_step src dst sm sn _ _ _ _
    (by simpa using regs.1) (by simpa using regs.2.1) (by simpa using regs.2.2)

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
