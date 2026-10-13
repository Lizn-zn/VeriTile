import bench.tritonbench_g.rotary_emb.Proofs.Parameters

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def range (base width : Nat) : Tile .nat [width] :=
  ⟨fun i ↦ base*width+i.1.val⟩

def dims (D : Nat) (odd : Bool) : Tile .nat [D/2] :=
  ⟨fun i ↦ i.1.val*2 + if odd then 1 else 0⟩

def prefix (p : Parameters) : List Stmt := [
  .assign .nat [] "cur_head_index" (.programId 0),
  .assign .nat [] "cur_seq_index" (.programId 1),
  .assign .nat [p.blockHead] "cur_head_range"
    (.add .nat .scalarL
      (.mul .nat .nil (.ref .nat [] "cur_head_index") (.constNat p.blockHead))
      (.arange p.blockHead)),
  .assign .nat [p.blockSeq] "cur_seq_range"
    (.add .nat .scalarL
      (.mul .nat .nil (.ref .nat [] "cur_seq_index") (.constNat p.blockSeq))
      (.arange p.blockSeq)),
  .assign .nat [p.dimension/2] "dim_range0"
    (.mul .nat .scalarR (.arange (p.dimension/2)) (.constNat 2)),
  .assign .nat [p.dimension/2] "dim_range1"
    (.add .nat .scalarR
      (.mul .nat .scalarR (.arange (p.dimension/2)) (.constNat 2)) (.constNat 1))]

def prefixState (p : Parameters) (s : BlockState) : BlockState :=
  (((((s.setReg "cur_head_index" .nat [] (Tile.scalar (s.pids 0))).setReg
    "cur_seq_index" .nat [] (Tile.scalar (s.pids 1))).setReg
    "cur_head_range" .nat [p.blockHead] (range (s.pids 0) p.blockHead)).setReg
    "cur_seq_range" .nat [p.blockSeq] (range (s.pids 1) p.blockSeq)).setReg
    "dim_range0" .nat [p.dimension/2] (dims p.dimension Bool.false)).setReg
    "dim_range1" .nat [p.dimension/2] (dims p.dimension Bool.true)

theorem source_prefix (p : Parameters) (Q K Cos Sin : RegionName) :
    (mathKernel p Q K Cos Sin).body.take 6 = prefix p := by
  rfl

theorem prefix_run (p : Parameters) (s : BlockState) :
    stepStmts (prefix p) s = some (prefixState p s) := by
  simp only [prefix]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId _ _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId _ _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := range (s.pids 0) p.blockHead) (by
    simp only [evalOp.eq_def, BlockState.setReg_ne_name, BlockState.setReg_same,
      String.reduceEq, not_false_eq_true, ne_eq, Option.bind_some, Option.bind_eq_bind]
    rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := range (s.pids 1) p.blockSeq) (by
    simp only [evalOp.eq_def, BlockState.setReg_ne_name, BlockState.setReg_same,
      String.reduceEq, not_false_eq_true, ne_eq, Option.bind_some, Option.bind_eq_bind]
    rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := dims p.dimension Bool.false) (by
    simp only [evalOp.eq_def, Option.bind_some, Option.bind_eq_bind]
    rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (v := dims p.dimension Bool.true) (by
    simp only [evalOp.eq_def, Option.bind_some, Option.bind_eq_bind]
    rfl))]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
