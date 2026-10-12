import bench.tritonbench_g.fast_rope_embedding.Proofs.Step
import VeriTile.Triton.Memory.TraceProofs
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.StepSafety
open VeriTile.Triton TypedSource Code
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem trace_safe (Q C S : RegionName) (QR CR SR seq H heads B row k : Nat) (backwards : Bool)
    (cs ss : Fin B → ℝ) (u : BlockState) (bounds : RegionBounds) (hc : Context u row H B cs ss)
    (hfirst : ∀ j : Fin B,j.val<H/2 → Reference.first row QR H k j.val<bounds Q)
    (hsecond : ∀ j : Fin B,j.val<H/2 → Reference.second row QR H k j.val<bounds Q) :
    Stmt.TraceSafeList bounds (body Q C S QR CR SR seq H heads B backwards)
      (u.setReg "k" .nat [] (Tile.scalar k)) := by
  rcases hc with ⟨hrow,hcols,hhalf,hmask,hcos,hsin⟩
  simp [body,Code.original,rope,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,
    Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,
    MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MemAccess.SafeAt,MaskOpt.Active,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,hrow,hcols,hhalf,hmask,hcos,hsin]
  exact ⟨hfirst,hsecond,hfirst,hsecond⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.StepSafety
