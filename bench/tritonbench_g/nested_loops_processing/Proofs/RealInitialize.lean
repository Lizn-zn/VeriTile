import bench.tritonbench_g.nested_loops_processing.Proofs.RealStep

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal
open VeriTile.Triton TypedSource
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def prefixBody (src dst : RegionName) (sm sn : Int) : List Stmt :=
  (source src dst sm sn).body.take 6

def initialA (src : RegionName) (sm sn : Int) : PointerTile :=
  fun i ↦ (src, ((i.1.val : Int) * sm + (i.2.1.val : Int) * sn).toNat)

def initialC (dst : RegionName) (sm sn : Int) : PointerTile :=
  fun i ↦ (dst, (sm * (i.1.val : Int) + sn * (i.2.1.val : Int)).toNat)

noncomputable def prefixState (s : BlockState) (src dst : RegionName) (sm sn : Int) : BlockState :=
  let s := s.setReg "offs_am" .nat [2] (Tile.vec fun i ↦ i.val)
  let s := s.setReg "offs_an" .nat [2] (Tile.vec fun i ↦ i.val)
  let s := s.setReg "a_ptrs" .ptr [2,2] ⟨initialA src sm sn⟩
  let s := s.setReg "offs_cm" .nat [2] (Tile.vec fun i ↦ i.val)
  let s := s.setReg "offs_cn" .nat [2] (Tile.vec fun i ↦ i.val)
  s.setReg "c_ptrs" .ptr [2,2] ⟨initialC dst sm sn⟩

theorem prefix_step (src dst : RegionName) (sm sn : Int) (s : BlockState) :
    stepStmts (prefixBody src dst sm sn) s = some (prefixState s src dst sm sn) := by
  simp [prefixBody, source, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp [stepStmts, stepStmt, evalOp.eq_def, prefixState, initialA, initialC,
    Tile.bop, Tile.uop, Tile.vec, Tile.ptrAdd, Tile.expandDim,
    NumericDType.mul, NumericDType.add]
  all_goals rfl

theorem source_decomposition (src dst : RegionName) (sm sn : Int) :
    (source src dst sm sn).body = prefixBody src dst sm sn ++
      [.forRange "i" 0 2 1 (outer src dst sm sn)] := by
  simp [prefixBody, outer, source, kernel, ComputeKernel.surfaceBody,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
