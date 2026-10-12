import FifthSphDraft.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Candidate
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

inductive Kind where
  | forward | backward

def source (kind : Kind) (swapped : Bool) (X Y g : RegionName) (B N M off stride : Nat) : Kernel :=
  (match kind with
    | .forward => forward .fp32 .fp32 swapped X Y B N M off stride
    | .backward => backward .fp32 .fp32 .fp32 swapped X Y g B N M off stride).eraseDType.toAlgKernel

def indexStatement (swapped : Bool) (B : Nat) : Stmt :=
  .assign .nat [B] "coord_row_offset" (.add .nat .scalarR (.ref .nat [B] "coord_striding")
    (if swapped then
      .mul .nat .nil (.ref .nat [] "block_id") (.mul .nat .nil (.constNat B) (.ref .nat [] "coord_stride"))
    else
      .mul .nat .nil (.mul .nat .nil (.constNat B) (.ref .nat [] "coord_stride")) (.ref .nat [] "block_id")))

theorem index_flat (A : FlatAlloc) (B : Nat) (u : BlockState) :
    stepStmts (A.flattenStmts [indexStatement Bool.false B]) u=
    stepStmts (A.flattenStmts [indexStatement Bool.true B]) u := by
  cases hc : u.regs .nat [B] "coord_striding" <;>
    cases hp : u.regs .nat [] "block_id" <;> cases hs : u.regs .nat [] "coord_stride"
  all_goals simp [indexStatement,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    stepStmts,stepStmt,evalOp.eq_def,Tile.bop,NumericDType.add,NumericDType.mul,hc,hp,hs,Nat.mul_comm]

theorem decomposition (kind : Kind) (swapped : Bool) (X Y g : RegionName) (B N M off stride : Nat) :
    (source kind swapped X Y g B N M off stride).body=
      (source kind Bool.false X Y g B N M off stride).body.take 3 ++
      [indexStatement swapped B] ++
      (source kind Bool.false X Y g B N M off stride).body.drop 4 := by
  cases kind <;> cases swapped
  all_goals simp only [source,indexStatement,forward,backward,forward_fp32_fp32,backward_fp32_fp32_fp32,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

theorem flat_exec (A : FlatAlloc) (kind : Kind) (X Y g : RegionName) (B N M off stride : Nat) (u : BlockState) :
    exec (A.flattenKernel (source kind Bool.false X Y g B N M off stride)) u=
    exec (A.flattenKernel (source kind Bool.true X Y g B N M off stride)) u := by
  unfold exec FlatAlloc.flattenKernel
  rw [decomposition kind Bool.false,decomposition kind Bool.true]
  simp only [FlatAlloc.flattenStmts_append,ProgramRewrite.append,index_flat]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Candidate
