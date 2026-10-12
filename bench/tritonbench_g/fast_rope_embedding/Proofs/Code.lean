import bench.tritonbench_g.fast_rope_embedding.Proofs.Reference
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Code
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def original (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) : Kernel :=
  (rope .fp32 .fp32 .fp32 Bool.false Q C S QR CR SR seq H heads B backwards).eraseDType.toAlgKernel

def prelude (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) : List Stmt :=
  (original Q C S QR CR SR seq H heads B backwards).body.take 11

def body (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) : List Stmt :=
  match (original Q C S QR CR SR seq H heads B backwards).body.drop 11 with
  | [.forRangeDyn _ _ _ _ stmts] => stmts
  | _ => []

theorem decomposition (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) :
    (original Q C S QR CR SR seq H heads B backwards).body=
      prelude Q C S QR CR SR seq H heads B backwards ++
      [.forRangeDyn "k" (.ref .nat [] "head_start") (.ref .nat [] "head_end") (.constNat 1)
        (body Q C S QR CR SR seq H heads B backwards)] := by
  cases backwards
  all_goals simp only [prelude,body,original,rope,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
  all_goals rfl

structure Context (u : BlockState) (row H B : Nat) (cs ss : Fin B → ℝ) : Prop where
  row : u.regs .nat [] "row_position"=some (Tile.scalar row)
  columns : u.regs .nat [B] "col_offsets"=some ⟨fun j => j.1.val⟩
  half : u.regs .nat [] "half_head_dim"=some (Tile.scalar (H/2))
  mask : u.regs .bool [B] "mask"=some ⟨fun j => decide (j.1.val<H/2)⟩
  cosine : u.regs .real [B] "cos1"=some ⟨fun j => some (cs j.1)⟩
  sine : u.regs .real [B] "sin1"=some ⟨fun j => some (ss j.1)⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Code
