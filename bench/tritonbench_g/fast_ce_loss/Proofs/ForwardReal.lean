import bench.tritonbench_g.fast_ce_loss.Proofs.Erasure
import bench.tritonbench_g.fast_ce_loss.Proofs.RealMath
import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.ForwardReal
open VeriTile.Triton TypedSource RealMath
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def original (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) : Kernel :=
  (forward .fp32 cap scale Bool.false X Y Z L V R B SC SL).eraseDType.toAlgKernel

theorem read_lse (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) (u v : BlockState) (hb : 0<B) (hv : 0<V)
    (hne : Y≠Z)
    (he : exec (original cap scale X Y Z L V R B SC SL) u=some v) :
    v.readMem Z (u.pids 0) = lse cap scale SC SL V 0
      (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) := by
  have hl := transformed_lse cap scale SC SL V 0
    (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) hb hv
  cases cap <;> cases scale
  all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
  all_goals
    simp only [original,forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
    simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi,hb,
      Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices] at he
    subst v
    rw [BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hne.symm]
    rw [BlockState.writeMem_readMem,if_pos ⟨rfl,rfl⟩]
    simp only [transformCarrier,Bool.false_eq_true,↓reduceIte,Nat.zero_add] at hl
    erw [hl]
    rfl

theorem read_loss (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) (u v : BlockState) (hb : 0<B) (hv : 0<V)
    (he : exec (original cap scale X Y Z L V R B SC SL) u=some v) :
    v.readMem Y (u.pids 0) = RealMath.loss cap scale SC SL V
      (u.readMemValue .int (Region.cast L) (u.pids 0))
      (fun j : Fin B => u.readMem X (u.pids 0*R+j.val))
      (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) := by
  have hl := transformed_lse cap scale SC SL V 0
    (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) hb hv
  cases cap <;> cases scale
  all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
  all_goals
    simp only [original,forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
    simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi,hb,
      Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices] at he
    subst v
    rw [BlockState.writeMem_readMem,if_pos ⟨rfl,rfl⟩]
    simp only [transformCarrier,Bool.false_eq_true,↓reduceIte,Nat.zero_add] at hl
    try erw [hl]
    norm_num [RealMath.loss,transform,hi,WithBot.realTanh,WithBot.realDiv,WithBot.realMul,WithBot.realSub]

theorem frame (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) (u v : BlockState) (hb : 0<B)
    (he : exec (original cap scale X Y Z L V R B SC SL) u=some v)
    (r : RegionName) (o : Nat) (hy : ¬(r=Y ∧ o=u.pids 0)) (hz : ¬(r=Z ∧ o=u.pids 0)) :
    v.mem r o=u.mem r o := by
  cases cap <;> cases scale
  all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
  all_goals
    simp only [original,forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
    simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi,hb,
      Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices] at he
    subst v
    rw [BlockState.writeMem_mem,if_neg hy,BlockState.writeMem_mem,if_neg hz]
    rfl

/-- Complete row execution with both input-derived outputs and the tagged-cell frame. -/
theorem run (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) (u : BlockState) (hb : 0<B) (hv : 0<V) (hne : Y≠Z) :
    ∃ v, exec (original cap scale X Y Z L V R B SC SL) u=some v ∧
      v.readMem Z (u.pids 0) = lse cap scale SC SL V 0 (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) ∧
      v.readMem Y (u.pids 0) = RealMath.loss cap scale SC SL V
        (u.readMemValue .int (Region.cast L) (u.pids 0))
        (fun j : Fin B => u.readMem X (u.pids 0*R+j.val))
        (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) ∧
      (∀ r o, ¬(r=Y ∧ o=u.pids 0) → ¬(r=Z ∧ o=u.pids 0) → v.mem r o=u.mem r o) := by
  cases he : exec (original cap scale X Y Z L V R B SC SL) u with
  | none =>
    cases cap <;> cases scale
    all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
    all_goals
      simp only [original,forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
        ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
        ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
        Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
        Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
        VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
        IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind] at he
      simp [exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
        Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
        NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
        ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi,hb,
        Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
        TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices] at he
  | some v =>
    exact ⟨v,rfl,read_lse cap scale X Y Z L V R B SC SL u v hb hv hne he,
      read_loss cap scale X Y Z L V R B SC SL u v hb hv he,
      frame cap scale X Y Z L V R B SC SL u v hb he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.ForwardReal
