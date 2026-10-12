import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.CrossEntropy1.Index
open VeriTile.Triton
def index (B : Nat) (swapped : Bool) : Stmt :=
  .assign .nat [B] "col_offsets" (.add .nat .scalarL
    (if swapped then .mul .nat .nil (.constNat B) (.ref .nat [] "col_block_idx")
     else .mul .nat .nil (.ref .nat [] "col_block_idx") (.constNat B)) (.arange B))

theorem index_step (B : Nat) (u : BlockState) :
    stepStmt (index B Bool.true) u = stepStmt (index B Bool.false) u := by
  simp [index,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.mul,Nat.mul_comm]

/-- Exact index equality composes through arbitrary statement context. -/
theorem context_exec (before after : List Stmt) (B : Nat) (u : BlockState) :
    stepStmts (before ++ index B Bool.true :: after) u =
      stepStmts (before ++ index B Bool.false :: after) u := by
  induction before generalizing u with
  | nil => simp only [List.nil_append,stepStmts,index_step]
  | cons st rest ih =>
    simp only [List.cons_append,stepStmts]
    cases h : stepStmt st u with
    | none => rfl
    | some t => exact ih t

theorem context_safe (before after : List Stmt) (B : Nat) (bounds : RegionBounds) (u : BlockState) :
    Stmt.TraceSafeList bounds (before ++ index B Bool.true :: after) u ↔
      Stmt.TraceSafeList bounds (before ++ index B Bool.false :: after) u := by
  induction before generalizing u with
  | nil =>
    simp only [List.nil_append,Stmt.TraceSafeList,index_step]
    simp [Stmt.TraceSafe,index,Op.SafeAt.eq_def]
  | cons st rest ih =>
    simp only [List.cons_append,Stmt.TraceSafeList]
    cases h : stepStmt st u with
    | none => rfl
    | some t => exact and_congr_right (fun _ => ih t)

theorem context_flatten (before after : List Stmt) (B : Nat) :
    StmtList.FlattenOk (before ++ index B Bool.true :: after) ↔
      StmtList.FlattenOk (before ++ index B Bool.false :: after) := by
  induction before with
  | nil => simp [StmtList.FlattenOk,Stmt.FlattenOk,index,Op.FlattenOk.eq_def]
  | cons st rest ih => simp only [List.cons_append,StmtList.FlattenOk,ih]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CrossEntropy1.Index
