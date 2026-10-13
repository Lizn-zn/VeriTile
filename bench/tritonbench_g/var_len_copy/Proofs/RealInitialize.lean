import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.var_len_copy.Proofs.RealLoop

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def prefixBody (os ol ns : Region .int) (src dst : RegionName) (B : Nat) : List Stmt :=
  (source os ol ns src dst B).body.take 6

 theorem source_decomposition (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    (source os ol ns src dst B).body = prefixBody os ol ns src dst B ++
      [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat B)
        (body os ol ns src dst B)] := by
  simp [prefixBody,body,source,kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind]

 theorem prefix_run (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)) :
    ∃ mid, stepStmts (prefixBody os ol ns src dst B) s = some mid ∧
      Inv s src dst B length oldStart newStart 0 mid := by
  simp [prefixBody,source,kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind]
  simp only [stepStmts,stepStmt,evalOp.eq_def,BlockState.setReg_same,
    BlockState.setReg_ne_name,ne_eq,String.reduceEq,not_false_eq_true,
    Option.bind_eq_bind,Option.bind,Option.pure_def]
  refine ⟨_,rfl,?_⟩
  constructor
  · intro r o _; rfl
  · intro o; rfl
  · rfl
  · rfl
  · rfl
  · simpa using congrArg (fun n ↦ some (Tile.scalar n)) hL
  · simpa using congrArg (fun n ↦ some (Tile.scalar n)) hO
  · simpa using congrArg (fun n ↦ some (Tile.scalar n)) hN
  · simp
  · simp
  · intro t lane ht; omega

 theorem source_run (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)) :
    ∃ final, exec (source os ol ns src dst B) s = some final ∧
      Inv s src dst B length oldStart newStart (Math.rounds length B) final := by
  obtain ⟨mid,hp,hm⟩ := prefix_run s os ol ns src dst B length oldStart newStart hL hO hN
  obtain ⟨final,he,hf⟩ := loop_run s os ol ns src dst B length oldStart newStart hB hsd
    0 mid (Nat.zero_le _) hm
  refine ⟨final,?_,hf⟩
  simp only [exec,source_decomposition,ProgramRewrite.append,hp,Option.bind_some,
    stepStmts,stepStmt,evalOp.eq_def,hm.lengthReg,Option.bind_eq_bind,Option.bind,
    Tile.scalar_data,Nat.zero_mul,Nat.cast_zero] at he ⊢
  rw [he]

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
