import bench.tritonbench_g.softmax_optimize.Proofs.SignedExpression

namespace VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
open VeriTile.Triton
set_option maxHeartbeats 3000000

/-- Equal named pointers read the same complete mathematical memory cells. -/
theorem load_none_related (B : Nat) :
    ExprRelated B
      (.load .real (.ptr (.ref .ptr [B] "input_ptrs")) .none)
      (.load .real (.ptr (.ref .ptr [B] "input_ptrs")) .none) := by
  intro s t h
  have hp := h.regs_eq .ptr [B] "input_ptrs" (by simp [Shared, Scratch])
  simp only [evalOp.eq_def, hp, BlockState.readMemValue_real, h.toLiveRelated.readMem, BlockState.paddingValue_zero]

/-- The literal -infinity tail padding is preserved, without any zero-undef premise. -/
theorem load_tail_related (B : Nat) :
    ExprRelated B
      (.load .real (.ptr (.ref .ptr [B] "input_ptrs"))
        (.maskOther (.ref .bool [B] "mask") (Op.negInf.broadcast [B])))
      (.load .real (.ptr (.ref .ptr [B] "input_ptrs"))
        (.maskOther (.ref .bool [B] "mask") (Op.negInf.broadcast [B]))) := by
  intro s t h
  have hp := h.regs_eq .ptr [B] "input_ptrs" (by simp [Shared, Scratch])
  have hm := h.regs_eq .bool [B] "mask" (by simp [Shared, Scratch])
  simp only [evalOp.eq_def, hp, hm, BlockState.readMemValue_real, h.toLiveRelated.readMem, BlockState.paddingValue_zero]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxOptimize.SignedTransport
