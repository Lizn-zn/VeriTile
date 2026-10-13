import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerLoop

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def source (Y X : RegionName) (M N B : Nat) : Kernel :=
  (softmax_kernel_inner .fp32 .fp32 Y X M N B Bool.false).eraseDType.toAlgKernel

def fallback (Y X : RegionName) (M N B : Nat) : List Stmt :=
  match ((source Y X M N B).body[1]? : Option Stmt) with
  | some (.ifThenElse _ _ no) => no
  | _ => []

def pass1Prefix (Y X : RegionName) (M N B : Nat) : List Stmt :=
  (source Y X M N B).body.take 1 ++ (fallback Y X M N B).take 5

theorem prefix_run (s : BlockState) (Y X : RegionName) (M N B : Nat) :
    ∃ st, stepStmts (pass1Prefix Y X M N B) s = some st ∧
      Pass1Inv s X Y (s.pids 0) N B 0 st := by
  simp [pass1Prefix, source, fallback, softmax_kernel_inner,
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
  · simp [Tile.ptrAdd, Tile.bop, NumericDType.mul]
  · simp [Tile.ptrAdd, Tile.bop, NumericDType.mul]
  · simp [Tile.bop, NumericDType.add, NumericDType.sub, NumericDType.mul,
      NumericDType.div, RealAddress.previous, RealAddress.rounds]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    congr 1
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same,
      ne_eq, String.reduceEq, not_false_eq_true]
    congr 1
    apply Tile.ext
    intro i
    dsimp only
    rw [laneState_zero]
    norm_num

def firstPass (Y X : RegionName) (M N B : Nat) : List Stmt :=
  (source Y X M N B).body.take 1 ++ (fallback Y X M N B).take 7

theorem firstPass_eq (Y X : RegionName) (M N B : Nat) :
    firstPass Y X M N B = pass1Prefix Y X M N B ++
      [.forRangeDyn "start_n" (.constNat 0) (.ref .nat [] "previous_multiple") (.constNat B)
        (fullBody Y X M N B),
       .forRangeDyn "start_n" (.ref .nat [] "previous_multiple") (.constNat N) (.constNat B)
        (pass1Body Y X M N B)] := by
  rw [fullBody_extract, pass1Body_extract]
  simp [firstPass, pass1Prefix, source, fallback, softmax_kernel_inner,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem firstPass_run (s : BlockState) (Y X : RegionName) (M N B : Nat)
    (hN : 0 < N) (hB : 0 < B) :
    ∃ next, stepStmts (firstPass Y X M N B) s = some next ∧
      Pass1Inv s X Y (s.pids 0) N B (RealAddress.rounds N B) next := by
  obtain ⟨st, hprefix, hinv⟩ := prefix_run s Y X M N B
  obtain ⟨mid, hfull, hmid⟩ := full_loop s st Y X M (s.pids 0) N B hN hB hinv
  obtain ⟨last, htail, hlast⟩ := tail_loop s mid Y X M (s.pids 0) N B hN hB hmid
  refine ⟨last, ?_, hlast⟩
  rw [firstPass_eq, ProgramRewrite.append, hprefix]
  simp only [Option.bind_some, stepStmts, stepStmt, evalOp.eq_def,
    hinv.previousReg, hmid.previousReg, Option.bind_eq_bind, Option.bind,
    Option.pure_def, Tile.scalar_data, hfull, htail]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
