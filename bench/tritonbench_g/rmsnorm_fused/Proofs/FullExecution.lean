import bench.tritonbench_g.rmsnorm_fused.Proofs.FullWrite

namespace VeriTile.Bench.TritonBenchG.RmsnormFused.General
open VeriTile.Triton
set_option maxHeartbeats 3000000

/-- Successful execution and the independent complete-row formula for arbitrary
numbers of source tiles. Aliasing restrictions preserve the initial input snapshot. -/
theorem real_execution (s0 : BlockState) (X Y W : RegionName)
    (stride N B : Nat) (eps : ℝ) (hB : 0 < B) (hXY : X ≠ Y) (hWY : W ≠ Y) :
    ∃ t, exec (rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel s0 = some t ∧
      (∀ k : Fin N, t.readMem Y (s0.pids 0 * stride + k.val) = rowSpec s0 X W stride N eps k.val) ∧
      OutputFrame s0 t Y stride N := by
  obtain ⟨c, sv, hv, hc, hi⟩ := var_loop s0 (preState s0 X Y stride B) X Y
    stride N B hB (pre_inv s0 X Y stride N B)
  have hp := post_exec s0 sv X Y stride N B c eps hB hc hi
  have hpost : WriteInv s0 X Y W stride N eps 0 (postState s0 sv X stride N eps) := by
    refine ⟨post_context s0 sv X Y W stride N B c eps hi, ?_, ?_⟩
    · intro k
      simp [postState, BlockState.readMem, hi.mem]
    · intro r o _
      simpa [postState] using congrFun (congrFun hi.mem r) o
  obtain ⟨d, t, hw, hd, hout⟩ := write_loop s0 (postState s0 sv X stride N eps)
    X Y W stride N B eps hB hXY hWY hpost
  refine ⟨t, ?_, ?_, hout.frame⟩
  · unfold exec
    rw [body_decomp]
    simp only [List.append_assoc, List.cons_append, List.nil_append]
    rw [stepStmts.append_some (pre_exec s0 X Y stride B), stepStmts.cons_some hv,
      stepStmts.append_some hp, stepStmts.cons_some hw, stepStmts.nil]
  · intro k
    have hk : k.val < d * B := lt_of_lt_of_le k.isLt hd
    simpa [hk] using hout.output k

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFused.General
