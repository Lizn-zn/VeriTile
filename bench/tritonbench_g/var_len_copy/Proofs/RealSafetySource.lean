import bench.tritonbench_g.var_len_copy.Proofs.RealInitialize
import bench.tritonbench_g.var_len_copy.Proofs.RealSafetyLoop

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem trace_append_iff (bounds : RegionBounds) (before after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeList bounds (before ++ after) s ↔
      Stmt.TraceSafeList bounds before s ∧
        ∀ t, stepStmts before s = some t → Stmt.TraceSafeList bounds after t := by
  induction before generalizing s with
  | nil => simp [Stmt.TraceSafeList, stepStmts]
  | cons head tail ih =>
    simp only [List.cons_append, Stmt.TraceSafeList, stepStmts]
    cases heq : stepStmt head s with
    | none => simp
    | some next => simp only [Option.bind_some, ih, and_assoc]

theorem prefix_safe (bounds : RegionBounds) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName) (B : Nat)
    (hL : s.pids 0 < bounds ol.cast) (hO : s.pids 0 < bounds os.cast)
    (hN : s.pids 0 < bounds ns.cast) :
    Stmt.TraceSafeList bounds (prefixBody os ol ns src dst B) s := by
  simp [prefixBody,source,kernel,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Kernel.eraseDType,
    Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    NumericDType.eraseDType,ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType,Region.cast,Except.bind,bind]
  simp [Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt, Op.SafeAt.eq_def, MaskOpt.SafeAt,
    MaskOpt.Active, MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, stepStmts, stepStmt, evalOp.eq_def,
    BlockState.setReg_same, BlockState.setReg_ne_name,
    Option.bind_eq_bind, Option.bind, Option.map, hL, hO, hN]

theorem source_safe (bounds : RegionBounds) (s : BlockState)
    (os ol ns : Region .int) (src dst : RegionName)
    (B : Nat) (length : Int) (oldStart newStart : Nat)
    (hB : 0 < B) (hsd : src ≠ dst)
    (hL : s.readMemValue .int ol.cast (s.pids 0) = length)
    (hO : s.readMemValue .int os.cast (s.pids 0) = (oldStart : Int))
    (hN : s.readMemValue .int ns.cast (s.pids 0) = (newStart : Int))
    (hLb : s.pids 0 < bounds ol.cast) (hOb : s.pids 0 < bounds os.cast)
    (hNb : s.pids 0 < bounds ns.cast)
    (hsrc : oldStart + Math.copiedLength length B ≤ bounds src)
    (hdst : newStart + Math.copiedLength length B ≤ bounds dst) :
    (source os ol ns src dst B).TraceSafe bounds s := by
  obtain ⟨mid,hp,hm⟩ := prefix_run s os ol ns src dst B length oldStart newStart hL hO hN
  unfold Kernel.TraceSafe
  rw [source_decomposition, trace_append_iff]
  refine ⟨prefix_safe bounds s os ol ns src dst B hLb hOb hNb, ?_⟩
  intro t ht
  have : t = mid := Option.some.inj (ht.symm.trans hp)
  subst t
  rw [Stmt.TraceSafeList]
  constructor
  · simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, hm.lengthReg,
      Tile.scalar_data, Option.bind_eq_bind, Option.bind, true_and]
    simpa only [Nat.zero_mul, Nat.cast_zero] using
      loop_safe bounds s os ol ns src dst B length oldStart newStart hB hsd hsrc hdst
        0 mid (Nat.zero_le _) hm
  · cases stepStmt _ mid <;> simp [Stmt.TraceSafeList]

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
