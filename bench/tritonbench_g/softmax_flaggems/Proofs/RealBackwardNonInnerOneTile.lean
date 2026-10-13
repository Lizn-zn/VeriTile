import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerStore
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerPrepare

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def oneSum (sb : BlockState) (P G : RegionName)
    (row N K kbase B C : Nat) (k : Fin C) : ℝ :=
  ∑ i : Fin B, sample sb P row N K i.val (kbase+k.val) *
    sample sb G row N K i.val (kbase+k.val)

def onePrefix (P G Y : RegionName) (M N K B C : Nat) : List Stmt :=
  (source P G Y M N K B C Bool.true).body.take 3 ++
    (branch P G Y M N K B C Bool.true).take 6

structure OneInv (sb : BlockState) (P G : RegionName)
    (row N K kbase B C : Nat) (st : BlockState) : Prop where
  memory : st.mem=sb.mem
  pids : st.pids=sb.pids
  numPids : st.numPids=sb.numPids
  undef : st.undef=sb.undef
  offsetReg : st.regs .nat [B,C] "offsets" = some ⟨tileOffset row N K kbase 0⟩
  maskReg : st.regs .bool [B,C] "mask" = some ⟨tileActive N K kbase 0⟩
  pReg : st.regs .real [B,C] "out_tile" =
    some ⟨fun i ↦ some (tileSample sb P row N K kbase 0 i)⟩
  gReg : st.regs .real [B,C] "out_grad_tile" =
    some ⟨fun i ↦ some (tileSample sb G row N K kbase 0 i)⟩
  zReg : st.regs .real [C] "scale" =
    some ⟨fun i ↦ some (oneSum sb P G row N K kbase B C i.1)⟩

theorem onePrefix_run (s : BlockState) (P G Y : RegionName) (M N K B C : Nat) :
    ∃ next, stepStmts (onePrefix P G Y M N K B C) s = some next ∧
      OneInv s P G (s.pids 0) N K (s.pids 1*C) B C next := by
  simp [onePrefix, branch, source, softmax_backward_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  refine ⟨_, rfl, ?_⟩
  constructor
  · rfl
  · rfl
  · rfl
  · rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    congr 1
    apply Tile.ext
    intro i
    simp [Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul, tileOffset, offset]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    congr 1
    apply Tile.ext
    intro i
    simp [TileShape.insertAxisIndex, Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, tileActive, RealForwardNonInner.tileActive]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    congr 1
    apply Tile.ext
    intro i
    simp [TileShape.insertAxisIndex, Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, tileSample, sample, offset, BlockState.paddingValue,
      BlockState.readMemValue_real]
    split_ifs <;> rfl
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    congr 1
    apply Tile.ext
    intro i
    simp [TileShape.insertAxisIndex, Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
      ComparableDType.lt, tileSample, sample, offset, BlockState.paddingValue,
      BlockState.readMemValue_real]
    split_ifs <;> rfl
  · simp only [BlockState.setReg_same]
    congr 1
    apply Tile.ext
    intro i
    simp only [Tile.reduceSum, Tile.reduceSumDrop_data]
    calc
      _ = ∑ j : Fin B, ((sample s P (s.pids 0) N K j.val (s.pids 1*C+i.1.val) *
          sample s G (s.pids 0) N K j.val (s.pids 1*C+i.1.val) : ℝ) : WithBot ℝ) := by
        apply Finset.sum_congr rfl
        intro j _
        simp [TileShape.insertAxisIndex, Tile.bop, Tile.cop, Tile.expandDim, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, tileSample, sample, offset, BlockState.paddingValue,
          BlockState.readMemValue_real]
        split_ifs <;> rfl
      _ = _ := (WithBot.coe_sum Finset.univ _).symm

def oneTail (P G Y : RegionName) (M N K B C : Nat) : List Stmt :=
  (branch P G Y M N K B C Bool.true).drop 6

theorem oneTail_run (sb st : BlockState) (P G Y : RegionName)
    (M row N K kbase B C : Nat) (h : OneInv sb P G row N K kbase B C st) :
    stepStmts (oneTail P G Y M N K B C) st =
      some (scatter (B := B) (C := C)
        (st.setReg "in_grad_tile" .real [B,C] ⟨fun i ↦
          some (storeValue sb P G row N K kbase 0 (oneSum sb P G row N K kbase B C) i)⟩)
        Y (tileOffset row N K kbase 0)
        (storeValue sb P G row N K kbase 0 (oneSum sb P G row N K kbase B C))
        (tileActive N K kbase 0)) := by
  simp [oneTail, branch, source, softmax_backward_kernel_non_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, h.offsetReg, h.maskReg, h.pReg, h.gReg, h.zReg,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  simp [scatter, RealForwardNonInner.scatter, storeValue, Tile.bop, Tile.expandDim,
    NumericDType.mul, NumericDType.sub, BlockState.writeMemTyped_real]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
