import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealSetupSource

/-! Execution of the actual signed-metadata prefix. A nonnegative slot starts
the split-K loop with zero accumulator and unchanged memory. The sentinel is
handled separately; no natural-valued substitute metadata is introduced. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetup
open VeriTile.Triton RealSetupSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def afterPrelude (s : BlockState) (slot : Int) : BlockState :=
  ((s.setReg "pid_sk" .nat [] (Tile.scalar (s.pids 0))).setReg
    "cur_batch" .nat [] (Tile.scalar (s.pids 1))).setReg
    "lora_index" .int [] (Tile.scalar slot)

theorem prelude_exec (p : Parameters) (initial : BlockState) (slot : Int)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = slot) :
    stepStmts p.prelude initial = some (afterPrelude initial slot) := by
  unfold Parameters.prelude
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp :
    evalOp (Op.programId 0) initial = some (Tile.scalar (initial.pids 0))))]
  set s1 := initial.setReg "pid_sk" .nat [] (Tile.scalar (initial.pids 0)) with hs1
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp [hs1] :
    evalOp (Op.programId 1) s1 = some (Tile.scalar (initial.pids 1))))]
  set s2 := s1.setReg "cur_batch" .nat [] (Tile.scalar (initial.pids 1)) with hs2
  have hl : evalOp (Op.load .int (.region p.indices (.ref .nat [] "cur_batch")) .none) s2 =
      some (Tile.scalar slot) := by
    simp only [evalOp_load_region_none, evalOp_ref]
    have hc : s2.regs .nat [] "cur_batch" = some (Tile.scalar (initial.pids 1)) := by simp [hs2]
    rw [hc]
    apply congrArg some
    ext i
    change s2.readMemValue .int (Region.cast p.indices) (initial.pids 1) = slot
    simpa [hs2, hs1] using hm
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hl), stepStmts.nil]
  rfl

theorem guard_eval (s : BlockState) (slot : Int)
    (h : s.regs .int [] "lora_index" = some (Tile.scalar slot)) :
    evalOp Parameters.guard s = some (Tile.scalar (decide (slot ≠ -1))) := by
  simp only [Parameters.guard, evalOp, h, Tile.cop, Tile.scalar_data,
    Broadcast.leftIndex, Broadcast.rightIndex]
  rfl

private def offsets (p : Parameters) (initial : BlockState) (slot : Nat) : BlockState :=
  ((afterPrelude initial (Int.ofNat slot)).setReg "offset_n" .nat [p.BN]
    (Tile.vec (fun n => n.val))).setReg "offset_k" .nat [p.BK]
    (Tile.vec (fun k => k.val + initial.pids 0 * p.BK))

private def inputState (p : Parameters) (initial : BlockState) (slot : Nat) : BlockState :=
  (offsets p initial slot).setReg "a_ptr" .ptr []
    (Tile.scalar (p.input, initial.pids 1 * p.inputBatchStride))

private def weightState (p : Parameters) (initial : BlockState) (slot : Nat) : BlockState :=
  (inputState p initial slot).setReg "b_ptr" .ptr []
    (Tile.scalar (p.weights, p.slotStride * slot))

def afterSetup (p : Parameters) (initial : BlockState) (slot : Nat) : BlockState :=
  (weightState p initial slot).setReg "accumulator" .real [p.BN] ⟨fun _ => some 0⟩

private theorem weight_eval (p : Parameters) (s : BlockState) (slot : Nat)
    (hlx : s.regs .int [] "lora_index" = some (Tile.scalar (Int.ofNat slot))) :
    evalOp p.weightPointer s = some (Tile.scalar (p.weights, p.slotStride * slot)) := by
  simp only [Parameters.weightPointer, evalOp, hlx, Tile.bop, Tile.uop,
    Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
  apply congrArg some
  apply Tile.ext
  intro i
  change (p.weights, 0 + (Int.ofNat p.slotStride * Int.ofNat slot).toNat) =
    (p.weights, p.slotStride * slot)
  simp only [Nat.zero_add, Int.ofNat_eq_natCast, ← Nat.cast_mul, Int.toNat_natCast]

theorem setup_exec (p : Parameters) (initial : BlockState) (slot : Nat) :
    stepStmts p.setup (afterPrelude initial (Int.ofNat slot)) =
      some (afterSetup p initial slot) := by
  unfold Parameters.setup
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp :
    evalOp (Op.arange p.BN) (afterPrelude initial (Int.ofNat slot)) =
      some (Tile.vec (fun n => n.val))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (offk_eval p.BK (initial.pids 0) _ (by simp [afterPrelude])))]
  change stepStmts _ (offsets p initial slot) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aptr_eval p.input p.inputBatchStride (initial.pids 1) _
      (by simp [offsets, afterPrelude])))]
  change stepStmts _ (inputState p initial slot) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (weight_eval p _ slot
    (by simp [inputState, offsets, afterPrelude])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval p.BN _)), stepStmts.nil]
  rfl

@[simp] theorem setup_mem (p : Parameters) (initial : BlockState) (slot : Nat) :
    (afterSetup p initial slot).mem = initial.mem := rfl

@[simp] theorem setup_pids (p : Parameters) (initial : BlockState) (slot : Nat) :
    (afterSetup p initial slot).pids = initial.pids := rfl

/-- The actual signed source prefix establishes the invariant for arbitrary
nonnegative slots, program IDs and dimensions, including a masked K tail. -/
theorem setup_invariant (p : Parameters) (initial : BlockState) (slot : Nat)
    (hS : 0 < p.BK * p.splits) :
    RealLoop.shrinkInv p.input p.weights slot p.N p.K p.inputBatchStride p.slotStride
      p.rowStride p.colStride p.BN p.BK p.splits initial 0 (afterSetup p initial slot) := by
  simp [RealLoop.shrinkInv, afterSetup, weightState, inputState, offsets, afterPrelude,
    RealLoop.accPartial, Nat.add_pos_right p.K hS]
  rfl

/-- The entire K loop executes, including a partial final block. The final
counter identifies the ceiling trip count used by the independent sum. -/
theorem loop_exec (p : Parameters) (initial : BlockState) (slot : Nat)
    (hS : 0 < p.BK * p.splits) :
    ∃ final st, stepStmt p.loop (afterSetup p initial slot) = some st ∧
      final / (p.BK * p.splits) = RealLoop.numKIters p.K (p.BK * p.splits) ∧
      RealLoop.shrinkInv p.input p.weights slot p.N p.K p.inputBatchStride p.slotStride
        p.rowStride p.colStride p.BN p.BK p.splits initial final st := by
  obtain ⟨final, st, he, hf, hi⟩ :=
    forRange_inv (idx := "k") (start := 0) (stop := p.K) (step := p.BK * p.splits)
      (Nat.pos_iff_ne_zero.mp hS) (setup_invariant p initial slot hS)
      (fun i s hlt hinv => RealLoop.shrink_step p.input p.weights slot p.N p.K
        p.inputBatchStride p.slotStride p.rowStride p.colStride p.BN p.BK p.splits
        initial hS i s hlt hinv)
  refine ⟨final, st, he, ?_, hi⟩
  apply kloop_final_iters p.K (p.BK * p.splits) _ hS
  · rw [← hi.2.1]; exact hf
  · rw [← hi.2.1]; exact hi.2.2.1

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetup
