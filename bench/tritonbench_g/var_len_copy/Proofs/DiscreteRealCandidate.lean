import bench.tritonbench_g.var_len_copy.Proofs.RealCandidate
import bench.tritonbench_g.var_len_copy.Proofs.DiscreteRealFlat

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal
open VeriTile Triton DiscreteSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000

def candidate (k : Kind) (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : Kernel :=
  (kernel (representative k) Bool.true os ol ns src dst B).eraseDType.toAlgKernel

def candidateBody (k : Kind) (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : List Stmt :=
  match ((candidate k os ol ns src dst B).body[6]? : Option Stmt) with
  | some (.forRangeInt _ _ _ _ body) => body
  | _ => []

theorem body_commutes (k : Kind) (A : FlatAlloc) (os ol ns : Region .int)
    (src dst : RegionName) (B : Nat) (s : BlockState) :
    stepStmts (A.flattenStmts (body k os ol ns src dst B)) s =
      stepStmts (A.flattenStmts (candidateBody k os ol ns src dst B)) s := by
  cases k
  all_goals cases ho : s.regs .int [] "old_start" <;> cases hi : s.regs .int [] "i"
  all_goals simp [ho,hi,body,candidateBody,source,candidate,representative,kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind,
    FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp.eq_def,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask,
    stepStmts,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.add,Int.add_comm]

theorem candidate_decomposition (k : Kind) (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    (candidate k os ol ns src dst B).body = prefixBody os ol ns src dst B ++
      [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat B)
        (candidateBody k os ol ns src dst B)] := by
  cases k <;> simp [prefixBody,FullReal.prefixBody,FullReal.source,TypedSource.kernel,candidateBody,source,candidate,representative,kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind]

theorem flat_exec_commutes (k : Kind) (A : FlatAlloc) (os ol ns : Region .int)
    (src dst : RegionName) (B : Nat) (s : BlockState) :
    exec (A.flattenKernel (source k os ol ns src dst B)) s =
      exec (A.flattenKernel (candidate k os ol ns src dst B)) s := by
  change stepStmts (A.flattenStmts (source k os ol ns src dst B).body) s =
    stepStmts (A.flattenStmts (candidate k os ol ns src dst B).body) s
  rw [source_decomposition, candidate_decomposition, FullReal.flatten_append, FullReal.flatten_append,
    ProgramRewrite.append, ProgramRewrite.append]
  apply congrArg (Option.bind (stepStmts (A.flattenStmts (prefixBody os ol ns src dst B)) s))
  funext t
  simp only [FlatAlloc.flattenStmts, FlatAlloc.flattenStmt, FlatAlloc.flattenOp.eq_def]
  change stepStmts [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat B)
    (A.flattenStmts (body k os ol ns src dst B))] t =
    stepStmts [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat B)
      (A.flattenStmts (candidateBody k os ol ns src dst B))] t
  simp only [stepStmts, stepStmt, evalOp.eq_def]
  cases h : t.regs .int [] "length" with
  | none => rfl
  | some stop =>
    simp only [Option.bind_eq_bind, Option.bind, Tile.scalar_data]
    rw [FullReal.rangeInt_congr "i" (stop.data PUnit.unit) B _ _ (body_commutes k A os ol ns src dst B)]

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal

#guard_msgs (drop info) in
#auditModuleAxioms
