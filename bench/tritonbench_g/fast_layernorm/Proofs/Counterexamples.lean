import bench.tritonbench_g.fast_layernorm.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton
import VeriTile.Meta.Disprove
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.HostCounterexamples
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def kernel : ComputeKernel :=
  (forward .fp32 .fp32 .fp32 Bool.false "y" "x" "w" "b" "r" "mu" 3 3 3 4 1).eraseDType

def initial : BlockState where
  mem := fun reg _ => .real (if reg="x" ∨ reg="w" then 3 else 0)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun _ _ => 0

private theorem sum_four (f : Fin 4 → WithBot ℝ) : ∑ j, f j = f 0 + (f 1 + (f 2 + f 3)) := by
  simp [Fin.sum_univ_succ]

/-- A host-valid N=3, B=next_power_of_two(3)=4 row with all active inputs
three and eps=1 writes inverse standard deviation one half. Its ordinary
population variance is zero, so ordinary LayerNorm would write one. -/
theorem padded_inverse_std :
    (exec kernel initial).map (·.readMem "r" 0)=some (1/2) := by
  simp only [kernel,forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  simp [exec,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,Option.map,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
    Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices,
    List.finRange_succ,Fin.sum_univ_succ,NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
    ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,
    initial,Region.cast,BlockState.readMemValue,BlockState.setReg,BlockState.readMem,BlockState.readMemTyped,
    BlockState.writeMem,
    MemCell.real,MemCell.of,MemCell.readAs,WithBot.realRsqrt]
  erw [sum_four, sum_four]
  change (Real.sqrt (((3 - ((3 + (3 + (3 + 0)))/3)) * (3 - ((3 + (3 + (3 + 0)))/3)) + ((3 - ((3 + (3 + (3 + 0)))/3)) * (3 - ((3 + (3 + (3 + 0)))/3)) + ((3 - ((3 + (3 + (3 + 0)))/3)) * (3 - ((3 + (3 + (3 + 0)))/3)) + (0 - ((3 + (3 + (3 + 0)))/3)) * (0 - ((3 + (3 + (3 + 0)))/3)))))/3+1 : ℝ))⁻¹ = (2 : ℝ)⁻¹
  norm_num

refutation padded_source_has_ordinary_inverse_std :
    (exec kernel initial).map (·.readMem "r" 0)=some 1 := by
  rw [padded_inverse_std]
  norm_num

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastLayernorm.HostCounterexamples
