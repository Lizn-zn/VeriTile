import bench.tritonbench_g.kldiv_ops.Kernels
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivOps.Source
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def rowPointer (v : Bool) (name : RegName) (R : RegionName) (stride : Nat) : Stmt :=
  .assign .ptr [] name (.ptrAdd .nil (.ptrBase R) (.castIntToNat
    (if v then .mul .int .nil (.castNatToInt (.constNat stride)) (.ref .int [] "pid")
    else .mul .int .nil (.ref .int [] "pid") (.castNatToInt (.constNat stride)))))

def forwardPrefix (v : Bool) (Y T O : RegionName) (sy st so : Nat) : List Stmt :=
  [.assign .int [] "pid" (.castNatToInt (.programId 0)),
   rowPointer Bool.false "Y" Y sy, rowPointer Bool.false "T" T st, rowPointer v "O" O so]

def backwardPrefix (v : Bool) (T O : RegionName) (st so : Nat) : List Stmt :=
  [.assign .int [] "pid" (.castNatToInt (.programId 0)),
   rowPointer Bool.false "T" T st, rowPointer v "O" O so]

theorem forward_prefix_exec (Y T O : RegionName) (sy st so : Nat) (s : BlockState) :
    stepStmts (forwardPrefix Bool.true Y T O sy st so) s =
      stepStmts (forwardPrefix Bool.false Y T O sy st so) s := by
  simp [forwardPrefix, rowPointer, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.mul, Int.mul_comm]

theorem backward_prefix_exec (T O : RegionName) (st so : Nat) (s : BlockState) :
    stepStmts (backwardPrefix Bool.true T O st so) s =
      stepStmts (backwardPrefix Bool.false T O st so) s := by
  simp [backwardPrefix, rowPointer, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.mul, Int.mul_comm]

theorem forward_prefix_safe (bounds : RegionBounds) (v : Bool)
    (Y T O : RegionName) (sy st so : Nat) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (forwardPrefix v Y T O sy st so) s := by
  cases v <;> simp [forwardPrefix, rowPointer, Stmt.TraceSafeListR, Stmt.TraceSafeR,
    Op.SafeAtR.eq_def, stepStmtR_triv, stepStmt, evalOp.eq_def]

theorem backward_prefix_safe (bounds : RegionBounds) (v : Bool)
    (T O : RegionName) (st so : Nat) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (backwardPrefix v T O st so) s := by
  cases v <;> simp [backwardPrefix, rowPointer, Stmt.TraceSafeListR, Stmt.TraceSafeR,
    Op.SafeAtR.eq_def, stepStmtR_triv, stepStmt, evalOp.eq_def]

theorem safe_append_iff (bounds : RegionBounds) (before after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (before ++ after) s ↔
      Stmt.TraceSafeListR .triv bounds before s ∧
        ∀ t, stepStmts before s = some t → Stmt.TraceSafeListR .triv bounds after t := by
  induction before generalizing s with
  | nil => simp [Stmt.TraceSafeListR, stepStmts]
  | cons st rest ih =>
    cases he : stepStmt st s <;>
      simp [Stmt.TraceSafeListR, stepStmts, stepStmtR_triv, he, ih, and_assoc]

end VeriTile.Bench.TritonBenchG.KldivOps.Source
