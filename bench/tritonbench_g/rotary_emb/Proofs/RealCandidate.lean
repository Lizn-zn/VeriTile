import bench.tritonbench_g.rotary_emb.Proofs.Prefix
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton

def headStatement (v : Bool) (BH : Nat) : Stmt :=
  .assign .nat [BH] "cur_head_range" (.add .nat .scalarL
    (if v then .mul .nat .nil (.constNat BH) (.ref .nat [] "cur_head_index") else
      .mul .nat .nil (.ref .nat [] "cur_head_index") (.constNat BH)) (.arange BH))

theorem head_statement_run (BH : Nat) (s : BlockState) :
    stepStmt (headStatement Bool.false BH) s = stepStmt (headStatement Bool.true BH) s := by
  cases h : s.regs .nat [] "cur_head_index" <;>
    simp [headStatement, stepStmt, evalOp.eq_def, h, Tile.bop,
      NumericDType.mul, Nat.mul_comm]

theorem head_statement_flat (A : FlatAlloc) (BH : Nat) (s : BlockState) :
    stepStmt (A.flattenStmt (headStatement Bool.false BH)) s =
      stepStmt (A.flattenStmt (headStatement Bool.true BH)) s := by
  cases h : s.regs .nat [] "cur_head_index" <;>
    simp [headStatement, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def,
      stepStmt, evalOp.eq_def, h, Tile.bop, NumericDType.mul, Nat.mul_comm]

def mathVariant (v : Bool) (p : Parameters) (Q K Cos Sin : RegionName) : Kernel :=
  let a := mathKernel p Q K Cos Sin
  { a with body := a.body.take 2 ++ [headStatement v p.blockHead] ++ a.body.drop 3 }

theorem math_original (p : Parameters) (Q K Cos Sin : RegionName) :
    mathVariant Bool.false p Q K Cos Sin = mathKernel p Q K Cos Sin := rfl

/-- All memory effects and failures agree; no numerical rewrite is involved. -/
theorem math_candidate_run (p : Parameters) (Q K Cos Sin : RegionName) (s : BlockState) :
    exec (mathVariant Bool.true p Q K Cos Sin) s = exec (mathKernel p Q K Cos Sin) s := by
  rw [← math_original p Q K Cos Sin]
  simp only [exec, mathVariant, ProgramRewrite.append, stepStmts,
    ← head_statement_run]

theorem math_candidate_flat (A : FlatAlloc) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) :
    exec (A.flattenKernel (mathVariant Bool.true p Q K Cos Sin)) s =
      exec (A.flattenKernel (mathKernel p Q K Cos Sin)) s := by
  rw [← math_original p Q K Cos Sin]
  simp only [exec, mathVariant, FlatAlloc.flattenKernel, FlatAlloc.flattenStmts_append,
    ProgramRewrite.append, FlatAlloc.flattenStmts, List.map_cons, List.map_nil,
    stepStmts, ← head_statement_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
