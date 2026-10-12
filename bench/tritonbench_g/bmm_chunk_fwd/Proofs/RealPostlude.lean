import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealStore
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealFinish

/-! Connect the exact post-loop statements to refresh, optional sequence
filter, output conversion/addressing and the full-chunk terminal store. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealPostlude
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

abbrev refreshCode := BmmChunkBwd.RealFinish.refreshCode
abbrev refreshed := BmmChunkBwd.RealFinish.refreshed
abbrev rows := BmmChunkBwd.RealFinish.rows
abbrev columns := BmmChunkBwd.RealFinish.columns

def outputCode (BM BN : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .real [BM, BN] "out" (.ref .real [BM, BN] "acc"),
   .assign .ptr [] "out_ptr" (RealStore.basePointer p.Out p.oBatch p.oChunk p.oHead),
   .assign .ptr [BM, BN] "out_ptrs" (RealStore.pointers BM BN p.oRow p.oColumn)]

def code (BM BN : Nat) (p : RealProgram.Parameters) : List Stmt :=
  refreshCode BM BN ++ [.ifThen (.constBool p.hasSeq) (RealSequence.code BM BN p)] ++
    outputCode BM BN p ++ [RealStore.code BM BN p.chunk]

theorem matches_source (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BK p).after = code BM BN p := by
  cases swapped
  all_goals
    simp [code, refreshCode, BmmChunkBwd.RealFinish.refreshCode,
      BmmChunkBwd.RealResidual.rowOffsets, BmmChunkBwd.RealResidual.columnOffsets,
      outputCode, RealStore.code, RealStore.basePointer, RealStore.pointers, RealStore.mask,
      RealSequence.code, RealSequence.indexLoad, RealSequence.limitOp, RealSequence.filtered,
      RealProgram.realPart, RealProgram.parts, RealProgram.branch, RealProgram.source,
      TypedSource.specialize, Source.conditional?, Source.dynamicRange?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
      ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    try norm_num

def outputPrepared (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (rs : Fin BM → Nat) (ns : Fin BN → Nat)
    (values : Fin BM → Fin BN → ℝ) : BlockState :=
  let base := pb * p.oBatch + pc * p.oChunk + ph * p.oHead
  let ptrs := RealStore.pointerValues p.Out BM BN base p.oRow p.oColumn rs ns
  let t := s.setReg "out" .real [BM, BN] (Tile.mat fun i j => some (values i j))
  let t := t.setReg "out_ptr" .ptr [] (Tile.scalar (p.Out, base))
  t.setReg "out_ptrs" .ptr [BM, BN] ptrs

theorem output_preparation (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph : Nat) (rs : Fin BM → Nat) (ns : Fin BN → Nat) (values : Fin BM → Fin BN → ℝ)
    (hb : s.regs .nat [] "pid_b" = some (Tile.scalar pb))
    (hc : s.regs .nat [] "pid_c" = some (Tile.scalar pc))
    (hh : s.regs .nat [] "pid_h" = some (Tile.scalar ph))
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rs))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec ns))
    (hv : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (values i j))) :
    stepStmts (outputCode BM BN p) s = some (outputPrepared s BM BN p pb pc ph rs ns values) := by
  let so := s.setReg "out" .real [BM, BN] (Tile.mat fun i j => some (values i j))
  let base := pb * p.oBatch + pc * p.oChunk + ph * p.oHead
  let sb := so.setReg "out_ptr" .ptr [] (Tile.scalar (p.Out, base))
  have hbval : evalOp (RealStore.basePointer p.Out p.oBatch p.oChunk p.oHead) so =
      some (Tile.scalar (p.Out, base)) := by
    apply RealStore.base_eval so p.Out p.oBatch p.oChunk p.oHead pb pc ph
    all_goals simpa [so] using (by assumption)
  have hpval : evalOp (RealStore.pointers BM BN p.oRow p.oColumn) sb =
      some (RealStore.pointerValues p.Out BM BN base p.oRow p.oColumn rs ns) := by
    apply RealStore.pointers_eval sb p.Out BM BN base p.oRow p.oColumn rs ns
    · simp [sb]
    · simpa [sb, so] using hr
    · simpa [sb, so] using hn
  dsimp only [so, sb, base] at hbval hpval
  simp only [outputCode, stepStmts, stepStmt, evalOp_ref, hv, hbval, hpval,
    bind, Option.bind_some]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealPostlude
