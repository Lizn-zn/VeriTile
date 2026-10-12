import bench.tritonbench_g.fast_ce_loss.Proofs.ChunkedReal
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.ChunkedReal
open VeriTile.Triton TypedSource RealMath
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem zero_run (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V C R B : Nat) (SC SL : ℝ) (u : BlockState) (hb : 0<B) (hv : 0<V)
    (hne : Y≠Z) (hcz : u.pids 1=0) :
    ∃ v, exec (original cap scale X Y Z L V C R B SC SL) u=some v ∧
      v.readMem Z (u.pids 0*C) = lse cap scale SC SL V 0 (fun j : Fin B => u.readMem X (u.pids 0*R+j.val)) ∧
      v.readMem Y (u.pids 0) = partialLoss cap scale SC SL
        (u.readMemValue .int (Region.cast L) (u.pids 0))
        (u.readMem X (u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat)) ∧
      (∀ r o, ¬(r=Y ∧ o=u.pids 0) → ¬(r=Z ∧ o=u.pids 0*C) → v.mem r o=u.mem r o) := by
  cases he : exec (original cap scale X Y Z L V C R B SC SL) u with
  | none =>
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
  | some v =>
    exact ⟨v,rfl,zero_read_lse cap scale X Y Z L V C R B SC SL u v hb hcz he hv,
      zero_read_loss cap scale X Y Z L V C R B SC SL u v hb hcz he hne,
      zero_frame cap scale X Y Z L V C R B SC SL u v hb hcz he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.ChunkedReal
