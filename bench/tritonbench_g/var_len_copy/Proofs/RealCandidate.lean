import bench.tritonbench_g.var_len_copy.Proofs.Source
import bench.tritonbench_g.var_len_copy.Proofs.RealFlat

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def candidate (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : Kernel :=
  (kernel .fp32 .fp32 Bool.true os ol ns src dst B).eraseDType.toAlgKernel

def candidateBody (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : List Stmt :=
  match ((candidate os ol ns src dst B).body[6]? : Option Stmt) with
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []

theorem body_commutes (A : FlatAlloc) (os ol ns : Region .int)
    (src dst : RegionName) (B : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (body os ol ns src dst B)) s =
      stepStmts (A.flattenStmts (candidateBody os ol ns src dst B)) s := by
  cases ho : s.regs .int [] "old_start" <;> cases hi : s.regs .int [] "i"
  all_goals simp [ho,hi,body,candidateBody,source,candidate,kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind,
    FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp.eq_def,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask,
    stepStmts,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.add,Int.add_comm]

theorem candidate_decomposition (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    (candidate os ol ns src dst B).body = prefixBody os ol ns src dst B ++
      [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat B)
        (candidateBody os ol ns src dst B)] := by
  simp [prefixBody,candidateBody,source,candidate,kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind]

theorem rangeInt_congr (idx : RegName) (stop : Int) (stride : Nat) (a b : List Stmt)
    (hbody : ∀ s, stepStmts a s = stepStmts b s) :
    ∀ cur s, stepForRangeIntAux idx cur stop stride a s =
      stepForRangeIntAux idx cur stop stride b s := by
  intro cur s
  induction h : (stop - cur).toNat using Nat.strong_induction_on generalizing cur s with
  | _ remaining ih =>
    by_cases hz : stride = 0
    · rw [stepForRangeIntAux, stepForRangeIntAux, if_pos hz, if_pos hz]
    · by_cases hc : cur < stop
      · rw [stepForRangeIntAux, stepForRangeIntAux, if_neg hz, if_neg hz, if_pos hc, if_pos hc, hbody]
        cases he : stepStmts b (s.setReg idx .int [] (Tile.scalar cur)) with
        | none => rfl
        | some next => exact ih (stop - (cur + stride)).toNat (by omega) (cur + stride) next rfl
      · rw [stepForRangeIntAux, stepForRangeIntAux, if_neg hz, if_neg hz, if_neg hc, if_neg hc]

theorem flatten_append (A : FlatAlloc) (before after : List Stmt) :
    A.flattenStmts (before ++ after) = A.flattenStmts before ++ A.flattenStmts after := by
  induction before with
  | nil => rfl
  | cons head tail ih => simp only [List.cons_append, FlatAlloc.flattenStmts, ih]

theorem flat_exec_commutes (A : FlatAlloc) (os ol ns : Region .int)
    (src dst : RegionName) (B : Nat) (s : BlockState) :
    exec (A.flattenKernel (source os ol ns src dst B)) s =
      exec (A.flattenKernel (candidate os ol ns src dst B)) s := by
  change stepStmts (A.flattenStmts (source os ol ns src dst B).body) s =
    stepStmts (A.flattenStmts (candidate os ol ns src dst B).body) s
  rw [source_decomposition, candidate_decomposition, flatten_append, flatten_append,
    ProgramRewrite.append, ProgramRewrite.append]
  apply congrArg (Option.bind (stepStmts (A.flattenStmts (prefixBody os ol ns src dst B)) s))
  funext t
  simp only [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def]
  change stepStmts [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat B)
    (A.flattenStmts (body os ol ns src dst B))] t =
    stepStmts [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat B)
      (A.flattenStmts (candidateBody os ol ns src dst B))] t
  simp only [stepStmts, stepStmt, evalOp.eq_def]
  cases h : t.regs .int [] "length" with
  | none => rfl
  | some stop =>
    simp only [Option.bind_eq_bind, Option.bind, Tile.scalar_data]
    rw [rangeInt_congr "i" (stop.data PUnit.unit) B _ _ (body_commutes A os ol ns src dst B)]

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
