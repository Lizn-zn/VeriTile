import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInner
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

noncomputable def laneSum (sb : BlockState) (P G : RegionName)
    (M N rbase B C c : Nat) (i : TileIndex [C,B]) : ℝ :=
  RealMath.backwardLaneAccumulator B c (fun n ↦
    sample sb P M N (rbase+i.1.val) n *
      sample sb G M N (rbase+i.1.val) n) i.2.1

theorem laneSum_zero (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (i : TileIndex [C,B]) :
    laneSum sb P G M N rbase B C 0 i = 0 := by
  simp [laneSum, RealMath.backwardLaneAccumulator]

theorem laneSum_succ (sb : BlockState) (P G : RegionName)
    (M N rbase B C c : Nat) (i : TileIndex [C,B]) :
    laneSum sb P G M N rbase B C (c+1) i =
      laneSum sb P G M N rbase B C c i +
        tileSample sb P M N rbase (c*B) i *
        tileSample sb G M N rbase (c*B) i := by
  simp [laneSum, RealMath.backwardLaneAccumulator_eq, Finset.sum_range_succ, tileSample]

structure Pass1Inv (sb : BlockState) (P G : RegionName)
    (M N rbase B C c : Nat) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  columnReg : st.regs .nat [C] "m_offsets" = some ⟨fun i ↦ rbase + i.1.val⟩
  nReg : st.regs .nat [B] "n_offsets" = some ⟨fun i ↦ c*B + i.1.val⟩
  offsetReg : st.regs .nat [C,B] "offsets" = some ⟨tileOffset N rbase (c*B)⟩
  sumReg : st.regs .real [C,B] "scale" =
    some ⟨fun i ↦ some (laneSum sb P G M N rbase B C c i)⟩

theorem pass1_invariant_step (sb st : BlockState) (P G Y : RegionName)
    (M N rbase B C c : Nat) (h : Pass1Inv sb P G M N rbase B C c st) :
    ∃ next, stepStmts (pass1Body P G Y M N B C)
      (st.setReg "unused_start_n" .nat [] (Tile.scalar (c*B))) = some next ∧
      Pass1Inv sb P G M N rbase B C (c+1) next := by
  obtain ⟨next, he, hm, hp, hnp, hu, hk, hn, ho, hz⟩ :=
    pass1_step P G Y M N B C rbase (c*B)
      (st.setReg "unused_start_n" .nat [] (Tile.scalar (c*B)))
      (laneSum sb P G M N rbase B C c)
      (by simpa using h.columnReg)
      (by simpa using h.nReg) (by simpa using h.offsetReg) (by simpa using h.sumReg)
  refine ⟨next, he, hm.trans h.memory, hp.trans h.pids, hnp.trans h.numPids,
    hu.trans h.undef, hk, ?_, ?_, ?_⟩
  · simpa only [Nat.add_mul, Nat.one_mul] using hn
  · simpa only [Nat.add_mul, Nat.one_mul] using ho
  · rw [hz]
    apply congrArg some
    apply Tile.ext
    intro i
    dsimp only
    rw [laneSum_succ]
    simp only [tileSample, sample, BlockState.readMem, BlockState.setReg_mem,
      h.memory, BlockState.setReg, h.undef]

theorem pass1_loop (sb st : BlockState) (P G Y : RegionName)
    (M N rbase B C : Nat) (hN : 0 < N) (hB : 0 < B)
    (h : Pass1Inv sb P G M N rbase B C 0 st) :
    ∃ next, stepForRangeAux "unused_start_n" 0 N B (pass1Body P G Y M N B C) st =
      some next ∧ Pass1Inv sb P G M N rbase B C (RealAddress.rounds N B) next := by
  simpa only [Nat.zero_mul] using RealLoop.range_tiles hN hB
    (fun c s _ hs ↦ pass1_invariant_step sb s P G Y M N rbase B C c hs)
    0 st (Nat.zero_le _) h

def pass1Prefix (P G Y : RegionName) (M N B C : Nat) : List Stmt :=
  (source P G Y M N B C Bool.false).body.take 2 ++
    (branch P G Y M N B C Bool.false).take 3

theorem prefix_run (s : BlockState) (P G Y : RegionName) (M N B C : Nat) :
    ∃ st, stepStmts (pass1Prefix P G Y M N B C) s = some st ∧
      Pass1Inv s P G M N (s.pids 0 * C) B C 0 st := by
  simp [pass1Prefix, source, branch, softmax_backward_kernel_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  simp only [stepStmts, stepStmt, evalOp.eq_def, BlockState.setReg_same,
    BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
    Option.bind_eq_bind, Option.bind, Option.pure_def]
  refine ⟨_, rfl, ?_⟩
  constructor
  · rfl
  · rfl
  · rfl
  · rfl
  · simp [Tile.bop, NumericDType.add, NumericDType.mul]
  · simp [Tile.bop, Tile.vec, NumericDType.add]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    congr 1
    apply Tile.ext
    intro i
    simp [Tile.bop, Tile.expandDim, NumericDType.add, NumericDType.mul,
      tileOffset, offset]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
      String.reduceEq, not_false_eq_true]
    congr 1

def firstPass (P G Y : RegionName) (M N B C : Nat) : List Stmt :=
  (source P G Y M N B C Bool.false).body.take 2 ++
    (branch P G Y M N B C Bool.false).take 4

theorem firstPass_eq (P G Y : RegionName) (M N B C : Nat) :
    firstPass P G Y M N B C = pass1Prefix P G Y M N B C ++
      [.forRange "unused_start_n" 0 N B (pass1Body P G Y M N B C)] := by
  simp [firstPass, pass1Prefix, source, branch, pass1Body,
    softmax_backward_kernel_inner, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem firstPass_run (s : BlockState) (P G Y : RegionName) (M N B C : Nat)
    (hN : 0 < N) (hB : 0 < B) :
    ∃ next, stepStmts (firstPass P G Y M N B C) s = some next ∧
      Pass1Inv s P G M N (s.pids 0 * C) B C (RealAddress.rounds N B) next := by
  obtain ⟨st, hprefix, hinv⟩ := prefix_run s P G Y M N B C
  obtain ⟨next, hloop, hnext⟩ :=
    pass1_loop s st P G Y M N (s.pids 0 * C) B C hN hB hinv
  refine ⟨next, ?_, hnext⟩
  rw [firstPass_eq, ProgramRewrite.append, hprefix]
  simp only [Option.bind_some, stepStmts, stepStmt, hloop]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
