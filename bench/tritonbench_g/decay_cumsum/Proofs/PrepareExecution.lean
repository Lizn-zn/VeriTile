import bench.tritonbench_g.decay_cumsum.Proofs.PreparePrelude
import bench.tritonbench_g.decay_cumsum.Proofs.ForwardProjection
namespace VeriTile.Bench.TritonBenchG.DecayCumsum.PrepareExecution
open VeriTile.Triton TypedSource

theorem reference_body (q k g qg kg : RegionName) (sH DK BT BK : Nat) (scale : ℝ) :
    (RealReference.prepare_qg_kg_surface q k g qg kg sH DK BT BK scale).toAlgKernel.body =
      PreparePrelude.referencePrelude q k g qg kg sH DK BT BK scale ++
        [.forRange "i" 0 BT 1 (RealReference.prepBody q k g qg kg sH DK BT BK scale)] := by
  rfl

theorem singleton (st : Stmt) (u : BlockState) :
    stepStmts [st] u = stepStmt st u := by
  simp only [stepStmts]
  cases stepStmt st u <;> rfl

theorem loop (q k g qg kg : RegionName) (sH DK BT BK : Nat) (scale : ℝ)
    (u : BlockState) :
    stepStmts [.forRange "i" 0 BT 1 (RealReference.prepBody q k g qg kg sH DK BT BK scale)]
      (PrepareRegisterFrame.setOffs BK u) =
    (stepStmts [.forRangeDyn "i" (.constNat 0) (.constNat BT) (.constNat 1)
      (RealReference.prepBody q k g qg kg sH DK BT BK scale)] u).map
        (PrepareRegisterFrame.setOffs BK) := by
  simp only [singleton,ForwardProjection.const_dynamic_step,
    stepForRangeAux.forRange_unfold]
  exact PrepareRegisterFrame.loop q k g qg kg sH DK BT BK 0 scale u

/-- The full source execution and reference differ only by the reference's
cached arange register. Memory, program IDs, unspecified values and failure
are all preserved. No shape-specific execution assumption is used. -/
theorem region (source gate : Storage) (q k g qg kg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (u : BlockState) :
    exec (RealReference.prepare_qg_kg_surface q k g qg kg sH DK BT BK scale).toAlgKernel u =
      (exec (prepare source gate Bool.false q k g qg kg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel u).map
        (PrepareRegisterFrame.setOffs BK) := by
  simp only [exec,reference_body,PrepareProjection.original_body,
    PrepareProjection.canonicalBody,ProgramRewrite.append,PreparePrelude.equivalent]
  cases h : stepStmts (PrepareProjection.prelude q k g qg kg sH DK BT BK scale) u
  · rfl
  · simpa only [Option.map_some,Option.bind_some] using loop q k g qg kg sH DK BT BK scale _

theorem flat_body (A : FlatAlloc) (q k g qg kg : RegionName)
    (sH DK BT BK : Nat) (scale : ℝ) :
    A.flattenStmts (RealReference.prepBody q k g qg kg sH DK BT BK scale) =
      RealReference.prepBody q k g qg kg sH DK BT BK scale := by
  simp only [RealReference.prepBody,ComputeDType.eraseDType,FlatAlloc.flattenStmts,
    FlatAlloc.flattenStmt,FlatAlloc.flattenOp,FlatAlloc.flattenAccess,FlatAlloc.flattenMask,
    Op.broadcast]


/-- The same correspondence holds after the finite allocation is flattened;
the cached register never changes a tagged memory cell. -/
theorem flat (A : FlatAlloc) (source gate : Storage) (q k g qg kg : RegionName)
    (sH sT sD B H T BT BK DK : Nat) (scale : ℝ) (u : BlockState) :
    exec (A.flattenKernel (RealReference.prepare_qg_kg_surface q k g qg kg sH DK BT BK scale).toAlgKernel) u =
      (exec (A.flattenKernel (prepare source gate Bool.false q k g qg kg sH sT sD B H T BT BK DK scale).eraseDType.toAlgKernel) u).map
        (PrepareRegisterFrame.setOffs BK) := by
  have hs : A.flattenStmts [.forRange "i" 0 BT 1 (RealReference.prepBody q k g qg kg sH DK BT BK scale)] =
      [.forRange "i" 0 BT 1 (RealReference.prepBody q k g qg kg sH DK BT BK scale)] := by
    simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,flat_body]
  have hd : A.flattenStmts [.forRangeDyn "i" (.constNat 0) (.constNat BT) (.constNat 1)
      (RealReference.prepBody q k g qg kg sH DK BT BK scale)] =
      [.forRangeDyn "i" (.constNat 0) (.constNat BT) (.constNat 1)
        (RealReference.prepBody q k g qg kg sH DK BT BK scale)] := by
    simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,flat_body]
  simp only [exec,FlatAlloc.flattenKernel,reference_body,PrepareProjection.original_body,
    PrepareProjection.canonicalBody,FlatAlloc.flattenStmts_append,hs,hd,
    ProgramRewrite.append,PreparePrelude.flat_equivalent]
  cases h : stepStmts (A.flattenStmts (PrepareProjection.prelude q k g qg kg sH DK BT BK scale)) u
  · rfl
  · simpa only [Option.map_some,Option.bind_some] using loop q k g qg kg sH DK BT BK scale _

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DecayCumsum.PrepareExecution
