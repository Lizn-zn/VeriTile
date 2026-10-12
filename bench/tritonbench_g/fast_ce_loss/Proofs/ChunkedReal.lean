import bench.tritonbench_g.fast_ce_loss.Proofs.Erasure
import bench.tritonbench_g.fast_ce_loss.Proofs.RealMath
import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.ChunkedReal
open VeriTile.Triton TypedSource RealMath
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def original (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) : Kernel :=
  (chunked .fp32 cap scale Bool.false X Y Z L V C R B SC SL).eraseDType.toAlgKernel

def partialLoss (cap scale : Bool) (SC SL : ℝ) (lab : Int) (gather : ℝ) : ℝ :=
  if lab = -100 then 0 else -transform cap scale SC SL gather

theorem zero_read_lse (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) (u v : BlockState) (hb : 0<B)
    (hcz : u.pids 1=0)
    (he : exec (original cap scale X Y Z L V C R B SC SL) u=some v) (hv : 0<V) :
    v.readMem Z (u.pids 0*C) = lse cap scale SC SL V 0
      (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) := by
  have hl := transformed_lse cap scale SC SL V 0
    (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) hb hv
  cases cap <;> cases scale
  all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
  all_goals
    simp only [original,chunked,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
    simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi,hb,hcz,
      Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices] at he
    subst v
    rw [BlockState.writeMem_readMem,if_pos ⟨rfl,rfl⟩]
    simp only [transformCarrier,Bool.false_eq_true,↓reduceIte,Nat.zero_add] at hl
    erw [hl]
    rfl

theorem zero_read_loss (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) (u v : BlockState) (hb : 0<B)
    (hcz : u.pids 1=0)
    (he : exec (original cap scale X Y Z L V C R B SC SL) u=some v) (hne : Y≠Z) :
    v.readMem Y (u.pids 0) = partialLoss cap scale SC SL
      (u.readMemValue .int (Region.cast L) (u.pids 0))
      (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) := by
  cases cap <;> cases scale
  all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
  all_goals
    simp only [original,chunked,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
    simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi,hb,hcz,
      Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices] at he
    subst v
    rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne]
    rw [BlockState.writeMem_readMem,if_pos ⟨rfl,rfl⟩]
    norm_num [partialLoss,transform,hi,WithBot.realTanh,WithBot.realDiv,WithBot.realMul]

theorem zero_frame (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) (u v : BlockState) (hb : 0<B)
    (hcz : u.pids 1=0)
    (he : exec (original cap scale X Y Z L V C R B SC SL) u=some v)
    (r : RegionName) (o : Nat) (hy : ¬(r=Y ∧ o=u.pids 0)) (hz : ¬(r=Z ∧ o=u.pids 0*C)) :
    v.mem r o=u.mem r o := by
  cases cap <;> cases scale
  all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
  all_goals
    simp only [original,chunked,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
    simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi,hb,hcz,
      Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices] at he
    subst v
    rw [BlockState.writeMem_mem,if_neg hz,BlockState.writeMem_mem,if_neg hy]
    rfl

/-- Every nonzero chunk executes its full computations and leaves all memory unchanged. -/
theorem later_run (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) (u : BlockState) (hb : 0<B) (hcz : u.pids 1≠0) :
    ∃ v, exec (original cap scale X Y Z L V C R B SC SL) u=some v ∧ v.mem=u.mem := by
  cases he : exec (original cap scale X Y Z L V C R B SC SL) u with
  | none =>
    cases cap <;> cases scale
    all_goals
      simp only [original,chunked,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
        ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
        Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
        Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
        VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
        IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
      simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
        Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
        NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
        ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hb,hcz,
        Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
        TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices] at he
  | some v =>
    refine ⟨v,rfl,?_⟩
    cases cap <;> cases scale
    all_goals
      simp only [original,chunked,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
        ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
        Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
        Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
        VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
        IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
      simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
        Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
        NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
        ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hb,hcz,
        Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
        TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices] at he
      subst v
      rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.ChunkedReal
