import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealAccess
import VeriTile.Triton.Memory.TraceProofs

/-! The optional residual branch reads exactly the source's output-live lanes.
Address safety is obtained after executing its own pointer preparation; no
prepared pointer, successful branch, or output value is assumed publicly. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealResidualSafety
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

theorem safe (swapped : Bool) (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph length : Nat) (rows : Fin BM → Nat) (columns : Fin BN → Nat) (bounds : RegionBounds)
    (hb : s.regs .nat [] "pid_b" = some (Tile.scalar pb))
    (hc : s.regs .nat [] "pid_c" = some (Tile.scalar pc))
    (hh : s.regs .nat [] "pid_h" = some (Tile.scalar ph))
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hbound : ∀ i j, rows i < length → columns j < p.K →
      RealResidual.baseValue p.chunk p.rBatch p.rRow p.rHead pb pc ph +
        (rows i * p.rRow + columns j * p.rColumn) < bounds p.Res) :
    Stmt.TraceSafeList bounds (RealResidual.residual swapped BM BN p) s := by
  let base := RealResidual.baseValue p.chunk p.rBatch p.rRow p.rHead pb pc ph
  let ptrs := RealResidual.pointerValues p.Res BM BN base p.rRow p.rColumn rows columns
  let s₁ := s.setReg "res_ptr" .ptr [] (Tile.scalar (p.Res, base))
  let s₂ := s₁.setReg "res_ptrs" .ptr [BM, BN] ptrs
  have hp := RealResidual.base_eval s p.Res p.chunk p.rBatch p.rRow p.rHead pb pc ph hb hc hh
  have hps : evalOp (RealResidual.pointers BM BN "res_ptr" p.rRow p.rColumn) s₁ = some ptrs := by
    apply RealResidual.pointers_eval s₁ p.Res BM BN base p.rRow p.rColumn "res_ptr" rows columns
    · simp [s₁]
    · simpa [s₁] using hr
    · simpa [s₁] using hn
  rw [RealResidual.residual, TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some hp)]
  refine ⟨by simp [Stmt.TraceSafe, RealResidual.basePointer, Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some hps)]
  refine ⟨by simp [Stmt.TraceSafe, RealResidual.pointers, Op.SafeAt.eq_def], ?_⟩
  apply TraceProofs.assign_cons
  · rw [Op.SafeAt.eq_def]
    refine ⟨by simp [Op.SafeAt.eq_def], by simp [RealResidual.mask, Op.SafeAt.eq_def], ?_⟩
    apply RealAccess.output_address_safe bounds s₂ BM BN p.K length "res_ptrs" ptrs rows columns
    · simp [s₂]
    · simpa [s₂, s₁] using hr
    · simpa [s₂, s₁] using hn
    · simpa [s₂, s₁] using hl
    · exact hbound
  · intro _
    apply TraceProofs.all_safe
    cases swapped <;> simp [RealResidual.addition, Stmt.TraceSafe, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealResidualSafety
