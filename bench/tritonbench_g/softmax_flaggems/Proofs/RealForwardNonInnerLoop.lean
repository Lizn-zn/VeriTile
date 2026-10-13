import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInner
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardModel
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

noncomputable def laneState (sb : BlockState) (X : RegionName)
    (row N K kbase B C c : Nat) (i : TileIndex [B, C]) : WithBot ℝ × ℝ :=
  if kbase + i.2.1.val < K then
    RealForwardModel.state N B i.1.val c
      (fun n ↦ sb.readMem X (row * N * K + n * K + (kbase + i.2.1.val)))
  else (⊥, 0)

theorem laneState_zero (sb : BlockState) (X : RegionName)
    (row N K kbase B C : Nat) (i : TileIndex [B, C]) :
    laneState sb X row N K kbase B C 0 i = (⊥, 0) := by
  simp [laneState, RealForwardModel.state, RealForwardModel.values]

theorem laneState_succ (sb : BlockState) (X : RegionName)
    (row N K kbase B C c : Nat) (i : TileIndex [B, C]) :
    laneState sb X row N K kbase B C (c + 1) i =
      RealMath.maskedOnlineStep (laneState sb X row N K kbase B C c i)
        (score sb X row N K kbase (c * B) B C i) := by
  by_cases hk : kbase + i.2.1.val < K
  · simpa [laneState, score, hk] using
      RealForwardModel.state_succ N B i.1.val c
        (fun n ↦ sb.readMem X (row * N * K + n * K + (kbase + i.2.1.val)))
  · simp only [laneState, if_neg hk, score, hk, and_false, if_false]
    exact (RealMath.maskedOnlineStep_none ⊥ 0 (fun _ ↦ rfl)).symm

structure Pass1Inv (sb : BlockState) (X : RegionName)
    (row N K kbase B C c : Nat) (st : BlockState) : Prop where
  memory : st.mem = sb.mem
  pids : st.pids = sb.pids
  numPids : st.numPids = sb.numPids
  undef : st.undef = sb.undef
  rowReg : st.regs .nat [] "pid_m" = some (Tile.scalar row)
  columnReg : st.regs .nat [C] "k_offsets" = some ⟨fun i ↦ kbase + i.1.val⟩
  maxReg : st.regs .real [B, C] "m" =
    some ⟨fun i ↦ (laneState sb X row N K kbase B C c i).1⟩
  sumReg : st.regs .real [B, C] "z" =
    some ⟨fun i ↦ some (laneState sb X row N K kbase B C c i).2⟩

 theorem pass1_invariant_step (sb st : BlockState) (Y X : RegionName)
    (M row N K kbase B C c : Nat) (h : Pass1Inv sb X row N K kbase B C c st) :
    ∃ next, stepStmts (pass1Body Y X M N K B C)
        (st.setReg "start_n" .nat [] (Tile.scalar (c * B))) = some next ∧
      Pass1Inv sb X row N K kbase B C (c + 1) next := by
  obtain ⟨next, hexec, hmem, hpids, hnum, hundef, hrow, hk, hm, hz⟩ :=
    pass1_step Y X M N K B C row kbase (c * B)
      (st.setReg "start_n" .nat [] (Tile.scalar (c * B)))
      (fun i ↦ (laneState sb X row N K kbase B C c i).1)
      (fun i ↦ (laneState sb X row N K kbase B C c i).2)
      (by simpa using h.rowReg) (by simpa using h.columnReg)
      (by simp) (by simpa using h.maxReg) (by simpa using h.sumReg)
  refine ⟨next, hexec, ?_⟩
  refine ⟨hmem.trans h.memory, hpids.trans h.pids, hnum.trans h.numPids,
    hundef.trans h.undef, hrow, hk, ?_, ?_⟩
  · rw [hm]
    apply congrArg some
    apply Tile.ext
    intro i
    dsimp only
    rw [laneState_succ]
    simp only [score, BlockState.setReg_readMem, BlockState.readMem, BlockState.setReg_mem, h.memory]
  · rw [hz]
    apply congrArg some
    apply Tile.ext
    intro i
    dsimp only
    rw [laneState_succ]
    simp only [score, BlockState.setReg_readMem, BlockState.readMem, BlockState.setReg_mem, h.memory]

 theorem pass1_loop (sb st : BlockState) (Y X : RegionName)
    (M row N K kbase B C : Nat) (hN : 0 < N) (hB : 0 < B)
    (h : Pass1Inv sb X row N K kbase B C 0 st) :
    ∃ next, stepForRangeAux "start_n" 0 N B (pass1Body Y X M N K B C) st = some next ∧
      Pass1Inv sb X row N K kbase B C (RealAddress.rounds N B) next := by
  simpa only [Nat.zero_mul] using RealLoop.range_tiles hN hB
    (fun c s _ hs ↦ pass1_invariant_step sb s Y X M row N K kbase B C c hs)
    0 st (Nat.zero_le _) h

private def source (Y X : RegionName) (M N K B C : Nat) : Kernel :=
  (softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel

private def fallback (Y X : RegionName) (M N K B C : Nat) : List Stmt :=
  match ((source Y X M N K B C).body[3]? : Option Stmt) with
  | some (.ifThenElse _ _ no) => no
  | _ => []

def pass1Prefix (Y X : RegionName) (M N K B C : Nat) : List Stmt :=
  (source Y X M N K B C).body.take 3 ++ (fallback Y X M N K B C).take 2

theorem pass1Prefix_extract (Y X : RegionName) (M N K B C : Nat) :
    pass1Prefix Y X M N K B C =
      (softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.body.take 3 ++
        (match ((softmax_kernel_non_inner .fp32 .fp32 Y X M N K B C Bool.false).eraseDType.toAlgKernel.body[3]? : Option Stmt) with
        | some (.ifThenElse _ _ no) => no
        | _ => []).take 2 := rfl

 theorem prefix_run (s : BlockState) (Y X : RegionName) (M N K B C : Nat) :
    ∃ st, stepStmts (pass1Prefix Y X M N K B C) s = some st ∧
      Pass1Inv s X (s.pids 0) N K (s.pids 1 * C) B C 0 st := by
  simp [pass1Prefix, source, fallback, softmax_kernel_non_inner,
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
  · simp
  · simp [Tile.bop, NumericDType.add, NumericDType.mul]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    congr 1
    apply Tile.ext
    intro i
    dsimp only
    rw [laneState_zero]
    rfl
  · simp only [BlockState.setReg_same]
    congr 1
    apply Tile.ext
    intro i
    dsimp only
    rw [laneState_zero]
    norm_num

/-- The initialization and first loop cut directly from the complete source branch. -/
def firstPass (Y X : RegionName) (M N K B C : Nat) : List Stmt :=
  (source Y X M N K B C).body.take 3 ++ (fallback Y X M N K B C).take 3

theorem firstPass_eq (Y X : RegionName) (M N K B C : Nat) :
    firstPass Y X M N K B C = pass1Prefix Y X M N K B C ++
      [.forRange "start_n" 0 N B (pass1Body Y X M N K B C)] := by
  simp [firstPass, pass1Prefix, source, fallback, pass1Body,
    softmax_kernel_non_inner, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]
  rfl

/-- Successful complete accumulation pass from arbitrary initial registers and memory. -/
theorem firstPass_run (s : BlockState) (Y X : RegionName) (M N K B C : Nat)
    (hN : 0 < N) (hB : 0 < B) :
    ∃ next, stepStmts (firstPass Y X M N K B C) s = some next ∧
      Pass1Inv s X (s.pids 0) N K (s.pids 1 * C) B C (RealAddress.rounds N B) next := by
  obtain ⟨st, hprefix, hinv⟩ := prefix_run s Y X M N K B C
  obtain ⟨next, hloop, hnext⟩ :=
    pass1_loop s st Y X M (s.pids 0) N K (s.pids 1 * C) B C hN hB hinv
  refine ⟨next, ?_, hnext⟩
  rw [firstPass_eq, ProgramRewrite.append, hprefix]
  simp only [Option.bind_some, stepStmts, stepStmt, hloop]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
