import bench.tritonbench_g.attention_score.Proofs.RealSafetyParts
import bench.tritonbench_g.attention_score.Proofs.RealProgram

/-! Safety of the exact masked output store. Only active columns require
allocated addresses; numerical output values do not occur in these premises. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealSafetyOutput
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
noncomputable section

theorem correct (bounds : RegionBounds) (s : BlockState) (Out : RegionName)
    (B NK soz soh batch head startN : Nat)
    (hz : s.regs .nat [] "off_z" = some (Tile.scalar batch))
    (hh : s.regs .nat [] "off_h" = some (Tile.scalar head))
    (hn : s.regs .nat [] "start_n" = some (Tile.scalar startN))
    (hb : ∀ i : Fin B, startN*B+i.val < NK → batch*soz+head*soh+startN*B+i.val < bounds Out) :
    Stmt.TraceSafeList bounds (RealProgram.postlude Out B NK soz soh) s := by
  let base := batch*soz+head*soh
  let so := s.setReg "o_offset" .nat [] (Tile.scalar base)
  let sr := so.setReg "o_range" .nat [B] (Tile.vec (fun i => i.val+startN*B))
  let sp := sr.setReg "o_ptrs" .ptr [B] (Tile.vec (fun i => (Out, base+startN*B+i.val)))
  have eo : evalOp (.add .nat .nil
      (.mul .nat .nil (.ref .nat [] "off_z") (.constNat soz))
      (.mul .nat .nil (.ref .nat [] "off_h") (.constNat soh))) s = some (Tile.scalar base) := by
    simp only [evalOp.eq_def, hz, hh, bind, Option.bind]
    rfl
  have er : evalOp (.add .nat .scalarR (.arange B)
      (.mul .nat .nil (.ref .nat [] "start_n") (.constNat B))) so =
      some (Tile.vec (fun i => i.val+startN*B)) := by
    simp [evalOp.eq_def, so, hn]
    rfl
  have ep : evalOp (.ptrAdd .scalarL (.ptrBase Out)
      (.add .nat .scalarL (.ref .nat [] "o_offset") (.ref .nat [B] "o_range"))) sr =
      some (Tile.vec (fun i => (Out, base+startN*B+i.val))) := by
    simp [evalOp.eq_def, sr, so, Tile.bop, Tile.ptrAdd, Tile.scalar, Tile.vec,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, Nat.add_assoc,
      Nat.add_left_comm, Nat.add_comm, -Tile.scalar_eta]
  have em : evalOp (.lt .nat .scalarR (.ref .nat [B] "o_range") (.constNat NK)) sp =
      some (Tile.vec (fun i => decide (startN*B+i.val < NK))) := by
    simp [evalOp.eq_def, sp, sr, Tile.cop, Tile.scalar, Tile.vec, ComparableDType.lt,
      Broadcast.leftIndex, Broadcast.rightIndex, Nat.add_comm, -Tile.scalar_eta]
    rfl
  rw [RealProgram.postlude, TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some eo)]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some er)]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some ep)]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt.eq_def], ?_⟩
  apply Stmt.TraceSafeList.cons_intro
  · change Stmt.TraceSafe bounds (.store .real [B] (.ptr (.ref .ptr [B] "o_ptrs"))
      (.ref .real [B] "o") (.mask (.lt .nat .scalarR (.ref .nat [B] "o_range") (.constNat NK)))) sp
    simp only [Stmt.TraceSafe, MemAccess.SafeAt, Op.SafeAt.eq_def, MaskOpt.SafeAt, true_and]
    unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
    refine ⟨by simp [Op.MemorySafe], ?_⟩
    intro ptrs hp i active
    have hp' : evalOp (.ref .ptr [B] "o_ptrs") sp =
        some (Tile.vec (fun i => (Out, base+startN*B+i.val))) := by simp [sp]
    rw [hp'] at hp
    cases hp
    obtain ⟨mask,hm,hi⟩ := active
    rw [em] at hm
    cases hm
    simp only [Tile.vec_data, decide_eq_true_eq] at hi
    exact hb i.1 hi
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealSafetyOutput
