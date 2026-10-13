import bench.tritonbench_g.kldiv_triton.Proofs.Source
import VeriTile.Triton.Semantics.ProgramRewrite

namespace VeriTile.Bench.TritonBenchG.KldivTriton.Candidate
open VeriTile.Triton TypedSource Source
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def rowPointer (v : Bool) (name : RegName) (R : RegionName) (stride : Nat) : Stmt :=
  .assign .ptr [] name (.ptrAdd .nil (.ptrBase R) (.castIntToNat
    (if v then .mul .int .nil (.castNatToInt (.constNat stride)) (.ref .int [] "pid")
    else .mul .int .nil (.ref .int [] "pid") (.castNatToInt (.constNat stride)))))

def forwardPrefix (v : Bool) (Y T O : RegionName) (sy st so : Nat) : List Stmt :=
  [.assign .int [] "pid" (.castNatToInt (.programId 0)),
   rowPointer Bool.false "Y" Y sy, rowPointer Bool.false "T" T st, rowPointer v "O" O so]

def backwardPrefix (v : Bool) (O : RegionName) (so : Nat) : List Stmt :=
  [.assign .int [] "pid" (.castNatToInt (.programId 0)), rowPointer v "O" O so]

theorem forward_prefix_source (v : Bool) (Y T O : RegionName) (sy st so N B : Nat)
    (flag : Bool) (reduction : Nat) :
    (forwardMath v Y T O sy st so N B flag reduction).body.take 4 =
      forwardPrefix v Y T O sy st so := by
  cases v <;> rfl

theorem backward_prefix_source (v : Bool) (T O : RegionName) (st so N B : Nat) (flag : Bool) :
    (backwardMath v T O st so N B flag).body.take 2 = backwardPrefix v O so := by
  cases v <;> cases flag <;> rfl

theorem forward_prefix_flat (A : FlatAlloc) (Y T O : RegionName) (sy st so : Nat)
    (s : BlockState) :
    stepStmts (A.flattenStmts (forwardPrefix Bool.true Y T O sy st so)) s =
      stepStmts (A.flattenStmts (forwardPrefix Bool.false Y T O sy st so)) s := by
  simp [forwardPrefix, rowPointer, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.mul, Int.mul_comm]

theorem backward_prefix_flat (A : FlatAlloc) (O : RegionName) (so : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (backwardPrefix Bool.true O so)) s =
      stepStmts (A.flattenStmts (backwardPrefix Bool.false O so)) s := by
  simp [backwardPrefix, rowPointer, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.uop, NumericDType.mul, Int.mul_comm]

/-- Whole-state equality in every flat allocation, leaving both full branches and loops intact. -/
theorem forward_flat (A : FlatAlloc) (v : Bool) (Y T O : RegionName)
    (sy st so N B : Nat) (flag : Bool) (reduction : Nat) (s : BlockState) :
    exec (A.flattenKernel (forwardMath v Y T O sy st so N B flag reduction)) s =
      exec (A.flattenKernel (FullAlgorithm.forward Y T O sy st so N B flag reduction)) s := by
  cases v
  · rfl
  · change exec (A.flattenKernel (forwardMath Bool.true Y T O sy st so N B flag reduction)) s =
      exec (A.flattenKernel (forwardMath Bool.false Y T O sy st so N B flag reduction)) s
    unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 4 (forwardMath Bool.true Y T O sy st so N B flag reduction).body,
      ← List.take_append_drop 4 (forwardMath Bool.false Y T O sy st so N B flag reduction).body]
    have ht : (forwardMath Bool.true Y T O sy st so N B flag reduction).body.drop 4 =
        (forwardMath Bool.false Y T O sy st so N B flag reduction).body.drop 4 := rfl
    rw [forward_prefix_source, forward_prefix_source]
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, forward_prefix_flat, ht]

theorem backward_flat (A : FlatAlloc) (v : Bool) (T O : RegionName)
    (st so N B : Nat) (flag : Bool) (s : BlockState) :
    exec (A.flattenKernel (backwardMath v T O st so N B flag)) s =
      exec (A.flattenKernel (FullAlgorithm.backward T O st so N B flag)) s := by
  cases v
  · rfl
  · change exec (A.flattenKernel (backwardMath Bool.true T O st so N B flag)) s =
      exec (A.flattenKernel (backwardMath Bool.false T O st so N B flag)) s
    unfold exec FlatAlloc.flattenKernel
    rw [← List.take_append_drop 2 (backwardMath Bool.true T O st so N B flag).body,
      ← List.take_append_drop 2 (backwardMath Bool.false T O st so N B flag).body]
    have ht : (backwardMath Bool.true T O st so N B flag).body.drop 2 =
        (backwardMath Bool.false T O st so N B flag).body.drop 2 := by cases flag <;> rfl
    rw [backward_prefix_source, backward_prefix_source]
    simp only [FlatAlloc.flattenStmts_append, ProgramRewrite.append, backward_prefix_flat, ht]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivTriton.Candidate
