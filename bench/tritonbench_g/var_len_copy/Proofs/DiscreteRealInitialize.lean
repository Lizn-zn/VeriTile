import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.var_len_copy.Proofs.DiscreteRealLoop
import bench.tritonbench_g.var_len_copy.Proofs.RealInitialize

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal
open VeriTile Triton DiscreteSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

abbrev prefixBody := FullReal.prefixBody

 theorem source_decomposition (k : Kind) (os ol ns : Region .int) (src dst : RegionName) (B : Nat) :
    (source k os ol ns src dst B).body = prefixBody os ol ns src dst B ++
      [.forRangeInt "i" (.constInt 0) (.ref .int [] "length") (.constNat B)
        (body k os ol ns src dst B)] := by
  cases k <;> simp [prefixBody,FullReal.prefixBody,FullReal.source,TypedSource.kernel,body,source,representative,kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind]

 theorem prefix_run (k : Kind) (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)) :
    ∃ mid, stepStmts (prefixBody os ol ns src dst B) s = some mid ∧
      Inv k s src dst B length oldStart newStart 0 mid := by
  obtain ⟨mid,hp,hi⟩ := FullReal.prefix_run s os ol ns src dst B length oldStart newStart hL hO hN
  refine ⟨mid,hp,hi.frame,hi.input,hi.pids,hi.numPids,hi.undef,hi.lengthReg,
    hi.oldReg,hi.newReg,hi.oldOffset,hi.newOffset,?_⟩
  intro t lane ht
  omega

 theorem source_run (k : Kind) (s : BlockState) (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int)) :
    ∃ final, exec (source k os ol ns src dst B) s = some final ∧
      Inv k s src dst B length oldStart newStart (Math.rounds length B) final := by
  obtain ⟨mid,hp,hm⟩ := prefix_run k s os ol ns src dst B length oldStart newStart hL hO hN
  obtain ⟨final,he,hf⟩ := loop_run k s os ol ns src dst B length oldStart newStart hB hsd
    0 mid (Nat.zero_le _) hm
  refine ⟨final,?_,hf⟩
  simp only [exec,source_decomposition,ProgramRewrite.append,hp,Option.bind_some,
    stepStmts,stepStmt,evalOp.eq_def,hm.lengthReg,Option.bind_eq_bind,Option.bind,
    Tile.scalar_data,Nat.zero_mul,Nat.cast_zero] at he ⊢
  rw [he]

end VeriTile.Bench.TritonBenchG.VarLenCopy.DiscreteReal

#guard_msgs (drop info) in
#auditModuleAxioms
