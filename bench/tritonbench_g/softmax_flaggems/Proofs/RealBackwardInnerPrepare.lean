import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerLoop

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def reductionBody (P G Y : RegionName) (M N B C : Nat) : List Stmt :=
  ((branch P G Y M N B C Bool.false).drop 4).take 3

noncomputable def columnSum {B C : Nat} (z : TileIndex [C,B] → ℝ) (k : Fin C) : ℝ :=
  ∑ i : Fin B, z (k,i,PUnit.unit)

noncomputable def finalSum (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (k : Fin C) : ℝ :=
  columnSum (laneSum sb P G M N rbase B C (RealAddress.rounds N B)) k

theorem finalSum_eq (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (hB : 0 < B) (k : Fin C) :
    finalSum sb P G M N rbase B C k =
      ∑ n ∈ Finset.range (RealAddress.rounds N B * B),
        sample sb P M N (rbase+k.val) n * sample sb G M N (rbase+k.val) n := by
  unfold finalSum columnSum laneSum
  exact RealMath.backwardLaneAccumulator_reduce B (RealAddress.rounds N B) hB
    (fun n ↦ sample sb P M N (rbase+k.val) n *
      sample sb G M N (rbase+k.val) n)

theorem reduction_step (P G Y : RegionName) (M N B C rbase : Nat)
    (s : BlockState) (z : TileIndex [C,B] → ℝ)
    (hk : s.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase+i.1.val⟩)
    (hz : s.regs .real [C,B] "scale" = some ⟨fun i ↦ some (z i)⟩) :
    ∃ next, stepStmts (reductionBody P G Y M N B C) s = some next ∧
      next.mem=s.mem ∧ next.pids=s.pids ∧ next.numPids=s.numPids ∧ next.undef=s.undef ∧
      next.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase+i.1.val⟩ ∧
      next.regs .nat [B] "n_offsets" = some ⟨fun i ↦ i.1.val⟩ ∧
      next.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase 0⟩ ∧
      next.regs .real [C] "scale" = some ⟨fun i ↦ some (columnSum z i.1)⟩ := by
  simp [reductionBody, branch, source, softmax_backward_kernel_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, hk, hz,
    BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq,
    String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind, Option.pure_def]
  refine ⟨_, rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_⟩
  · simpa using hk
  · simp [Tile.vec]
  · simp only [BlockState.setReg_same]
    congr 1
    apply Tile.ext
    intro i
    simp [Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul, tileOffset, offset]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    congr 1
    apply Tile.ext
    intro i
    simp only [Tile.reduceSum, Tile.reduceSumDrop_data]
    exact (WithBot.coe_sum Finset.univ _).symm

structure ReducedInv (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  columnReg : st.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase+i.1.val⟩
  nReg : st.regs .nat [B] "n_offsets" = some ⟨fun i ↦ i.1.val⟩
  offsetReg : st.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase 0⟩
  sumReg : st.regs .real [C] "scale" =
    some ⟨fun i ↦ some (finalSum sb P G M N rbase B C i.1)⟩

def prepareBody (P G Y : RegionName) (M N B C : Nat) : List Stmt :=
  firstPass P G Y M N B C ++ reductionBody P G Y M N B C

theorem prepare_run (s : BlockState) (P G Y : RegionName) (M N B C : Nat)
    (hN : 0 < N) (hB : 0 < B) :
    ∃ next, stepStmts (prepareBody P G Y M N B C) s = some next ∧
      ReducedInv s P G M N (s.pids 0*C) B C next := by
  obtain ⟨st, he, hi⟩ := firstPass_run s P G Y M N B C hN hB
  obtain ⟨next, hr, hm, hp, hnp, hu, hk, hn, ho, hz⟩ :=
    reduction_step P G Y M N B C (s.pids 0*C) st
      (laneSum s P G M N (s.pids 0*C) B C (RealAddress.rounds N B))
      hi.columnReg hi.sumReg
  refine ⟨next, ?_, hm.trans hi.memory, hp.trans hi.pids, hnp.trans hi.numPids,
    hu.trans hi.undef, hk, hn, ho, hz⟩
  rw [prepareBody, ProgramRewrite.append, he]
  exact hr

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
