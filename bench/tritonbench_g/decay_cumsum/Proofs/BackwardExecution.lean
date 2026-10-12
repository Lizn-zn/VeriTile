import bench.tritonbench_g.decay_cumsum.Proofs.BackwardPrelude
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardProjection
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardExecution
open VeriTile.Triton TypedSource
set_option maxHeartbeats 4000000
set_option maxRecDepth 10000

theorem reference_body (dqi dqo dki dko q k g dg : RegionName)
    (sH DK BT BK : Nat) (scale : ℝ) :
    (RealReference.bwd_decay_global_cumsum_surface dqi dqo dki dko q k g dg sH DK BT BK).toAlgKernel.body =
      BackwardPrelude.referencePrelude dqi dqo dki dko q k g dg sH DK BT BK scale ++
        [.forRangeDyn "__rev_j" (.constNat 0) (BackwardProjection.stop BT) (.constNat 1)
          (RealReference.bwdIterBodyG DK BT BK)] := by rfl

theorem singleton (st : Stmt) (u : BlockState) :
    stepStmts [st] u = stepStmt st u := by
  simp only [stepStmts]
  cases stepStmt st u <;> rfl

theorem loop (DK BT BK : Nat) (u : BlockState) :
    stepStmts [.forRangeDyn "__rev_j" (.constNat 0) (BackwardProjection.stop BT) (.constNat 1)
      (RealReference.bwdIterBodyG DK BT BK)] (BackwardRegisterFrame.setOffs BK u) =
    (stepStmts [.forRangeDyn "__rev_j" (.constNat 0) (BackwardProjection.stop BT) (.constNat 1)
      (RealReference.bwdIterBodyG DK BT BK)] u).map (BackwardRegisterFrame.setOffs BK) := by
  simp only [singleton,stepStmt,BackwardProjection.stop,evalOp,Option.bind_some,
    NumericDType.add,NumericDType.sub,NumericDType.div,Tile.bop,Tile.scalar_data]
  exact BackwardRegisterFrame.loop DK BT BK 0 _ u

/-- Exact source/reference execution correspondence, including the final
pointer decrements. The reference only adds an unused arange register. -/
theorem region (source gate : Storage) (dqi dqo dki dko q k g dg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (u : BlockState) :
    exec (RealReference.bwd_decay_global_cumsum_surface dqi dqo dki dko q k g dg sH DK BT BK).toAlgKernel u =
      (exec (backward source gate Bool.false dqi dqo dki dko q k g dg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel u).map
        (BackwardRegisterFrame.setOffs BK) := by
  simp only [exec,reference_body dqi dqo dki dko q k g dg sH DK BT BK scale,
    BackwardProjection.original_body,BackwardProjection.canonicalBody,
    ProgramRewrite.append,BackwardPrelude.equivalent]
  cases h : stepStmts (BackwardProjection.prelude dqi dqo dki dko q k g dg sH DK BT BK scale) u
  · rfl
  · simpa only [Option.map_some,Option.bind_some] using loop DK BT BK _

theorem flat_body (A : FlatAlloc) (DK BT BK : Nat) :
    A.flattenStmts (RealReference.bwdIterBodyG DK BT BK) =
      RealReference.bwdIterBodyG DK BT BK := by
  simp only [RealReference.bwdIterBodyG,ComputeDType.eraseDType,FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,
    Op.broadcast]

/-- Flattening preserves the correspondence and every tagged memory cell. -/
theorem flat (A : FlatAlloc) (source gate : Storage) (dqi dqo dki dko q k g dg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (u : BlockState) :
    exec (A.flattenKernel (RealReference.bwd_decay_global_cumsum_surface dqi dqo dki dko q k g dg sH DK BT BK).toAlgKernel) u =
      (exec (A.flattenKernel (backward source gate Bool.false dqi dqo dki dko q k g dg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel) u).map
        (BackwardRegisterFrame.setOffs BK) := by
  have hl : A.flattenStmts [.forRangeDyn "__rev_j" (.constNat 0) (BackwardProjection.stop BT) (.constNat 1)
      (RealReference.bwdIterBodyG DK BT BK)] =
      [.forRangeDyn "__rev_j" (.constNat 0) (BackwardProjection.stop BT) (.constNat 1)
        (RealReference.bwdIterBodyG DK BT BK)] := by
    simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,flat_body,
      BackwardProjection.stop]
  simp only [exec,FlatAlloc.flattenKernel,
    reference_body dqi dqo dki dko q k g dg sH DK BT BK scale,
    BackwardProjection.original_body,BackwardProjection.canonicalBody,
    FlatAlloc.flattenStmts_append,hl,ProgramRewrite.append,BackwardPrelude.flat_equivalent]
  cases h : stepStmts (A.flattenStmts (BackwardProjection.prelude dqi dqo dki dko q k g dg sH DK BT BK scale)) u
  · rfl
  · simpa only [Option.map_some,Option.bind_some] using loop DK BT BK _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.BackwardExecution
