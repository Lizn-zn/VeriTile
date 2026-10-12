import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealSetupSource

/-! Exact execution of the full read-only prefix. The input tile is obtained
from initial memory, not supplied as an equation about final output. Metadata
is signed; the active path below uses a nonnegative slot and the caller handles
the -1 sentinel separately. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSetup
open VeriTile.Triton RealLoop RealSetupSource
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def afterPrelude (s : BlockState) (slot : Int) : BlockState :=
  ((s.setReg "pid_sn" .nat [] (Tile.scalar (s.pids 0))).setReg
    "cur_batch" .nat [] (Tile.scalar (s.pids 1))).setReg
    "lora_index" .int [] (Tile.scalar slot)

theorem prelude_exec (p : Parameters) (initial : BlockState) (slot : Int)
    (hm : initial.readMemValue .int (Region.cast p.indices) (initial.pids 1) = slot) :
    stepStmts p.prelude initial = some (afterPrelude initial slot) := by
  unfold Parameters.prelude
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp :
    evalOp (Op.programId 0) initial = some (Tile.scalar (initial.pids 0))))]
  set s1 := initial.setReg "pid_sn" .nat [] (Tile.scalar (initial.pids 0)) with hs1
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

def inputValues (p : Parameters) (initial : BlockState) (even : Bool) (k : Fin p.BK) : ℝ :=
  if even || decide (k.val < p.K) then
    initial.readMem p.input (initial.pids 1 * p.inputBatchStride + k.val * p.inputStride)
  else 0

private def offsets (p : Parameters) (initial : BlockState) (slot : Nat) : BlockState :=
  ((afterPrelude initial (Int.ofNat slot)).setReg "offset_k" .nat [p.BK]
    (Tile.vec (fun k => k.val))).setReg "offset_n" .nat [p.BN] (Tile.vec (fun i => i.val))

private def loaded (p : Parameters) (initial : BlockState) (slot : Nat) (even : Bool) : BlockState :=
  (offsets p initial slot).setReg "tiled_a" .real [p.BK] ⟨fun i => some (inputValues p initial even i.1)⟩

private def splitState (p : Parameters) (initial : BlockState) (slot : Nat) (even : Bool) : BlockState :=
  (loaded p initial slot even).setReg "split_n_length" .nat [] (Tile.scalar p.length)

private def castState (p : Parameters) (initial : BlockState) (slot : Nat) (cast even : Bool) : BlockState :=
  if cast then (splitState p initial slot even).setReg "tiled_a" .real [p.BK]
    ⟨fun i => some (inputValues p initial even i.1)⟩ else splitState p initial slot even

private def weightState (p : Parameters) (initial : BlockState) (slot : Nat) (cast even : Bool) : BlockState :=
  (castState p initial slot cast even).setReg "b_ptr" .ptr []
    (Tile.scalar (p.weights, (p.geometry initial slot).weightBase))

def afterSetup (p : Parameters) (initial : BlockState) (slot : Nat) (cast even : Bool) : BlockState :=
  (weightState p initial slot cast even).setReg "c_ptr" .ptr []
    (Tile.scalar (p.output, (p.geometry initial slot).outputBase))

private theorem input_eval (p : Parameters) (initial s : BlockState) (even : Bool)
    (hm : s.readMem p.input = initial.readMem p.input)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar (initial.pids 1)))
    (hk : s.regs .nat [p.BK] "offset_k" = some (Tile.vec (fun k => k.val))) :
    evalOp (p.inputLoad even) s = some ⟨fun i => some (inputValues p initial even i.1)⟩ := by
  cases even <;>
    simp only [Parameters.inputLoad, Parameters.inputAddress, Bool.false_eq_true,
      if_false, if_true, evalOp, hcb, hk, Tile.bop, Tile.cop, Tile.scalar_data, Tile.vec,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      NumericDType.add, NumericDType.mul, ComparableDType.lt, BlockState.readMemValue_real]
  all_goals
    apply congrArg some
    ext i
    by_cases h : i.1.val < p.K <;> simp [inputValues, h, hm]

private theorem split_eval (p : Parameters) (s : BlockState) :
    evalOp p.split s = some (Tile.scalar p.length) := by
  simp only [Parameters.split, evalOp, Tile.bop, Tile.scalar_data,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub, NumericDType.div]
  rfl

private theorem weight_eval (p : Parameters) (initial s : BlockState) (slot : Nat)
    (hps : s.regs .nat [] "pid_sn" = some (Tile.scalar (initial.pids 0)))
    (hlx : s.regs .int [] "lora_index" = some (Tile.scalar (Int.ofNat slot)))
    (hs : s.regs .nat [] "split_n_length" = some (Tile.scalar p.length)) :
    evalOp p.weightPointer s = some (Tile.scalar (p.weights, (p.geometry initial slot).weightBase)) := by
  simp only [Parameters.weightPointer, evalOp, hps, hlx, hs, Tile.bop, Tile.uop,
    Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]
  apply congrArg some
  apply Tile.ext
  intro i
  change (p.weights, 0 + (Int.ofNat p.slotStride * Int.ofNat slot +
    (initial.pids 0 * p.length * p.rowStride : Nat)).toNat) =
    (p.weights, p.slotStride * slot + initial.pids 0 * p.length * p.rowStride)
  simp only [Nat.zero_add, Int.ofNat_eq_natCast, ← Nat.cast_mul, ← Nat.cast_add, Int.toNat_natCast]


private theorem output_eval (p : Parameters) (initial s : BlockState) (slot : Nat)
    (hps : s.regs .nat [] "pid_sn" = some (Tile.scalar (initial.pids 0)))
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar (initial.pids 1)))
    (hs : s.regs .nat [] "split_n_length" = some (Tile.scalar p.length)) :
    evalOp p.outputPointer s = some (Tile.scalar (p.output, (p.geometry initial slot).outputBase)) := by
  simp only [Parameters.outputPointer, evalOp, hps, hcb, hs, Tile.bop,
    Tile.ptrAdd, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]
  apply congrArg some
  apply Tile.ext
  intro i
  change (p.output, 0 + (initial.pids 1 * p.outputBatchStride + initial.pids 0 * p.length +
    p.slice * p.outputStride)) =
    (p.output, initial.pids 1 * p.outputBatchStride + initial.pids 0 * p.length + p.slice * p.outputStride)
  simp

theorem setup_exec (p : Parameters) (initial : BlockState) (slot : Nat) (cast even : Bool) :
    stepStmts (p.setup cast even) (afterPrelude initial (Int.ofNat slot)) =
      some (afterSetup p initial slot cast even) := by
  unfold Parameters.setup
  simp only [List.append_assoc, List.cons_append, List.nil_append]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp :
    evalOp (Op.arange p.BK) (afterPrelude initial (Int.ofNat slot)) =
      some (Tile.vec (fun k => k.val))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by simp :
    evalOp (Op.arange p.BN) _ = some (Tile.vec (fun i => i.val))))]
  change stepStmts _ (offsets p initial slot) = _
  have hi := input_eval p initial (offsets p initial slot) even
    (by rfl) (by simp [offsets, afterPrelude]) (by simp [offsets])
  have hb : stepStmt (.ifThenElse (.constBool even)
      [.assign .real [p.BK] "tiled_a" (p.inputLoad Bool.true)]
      [.assign .real [p.BK] "tiled_a" (p.inputLoad Bool.false)]) (offsets p initial slot) =
      some (loaded p initial slot even) := by
    cases even <;> simp only [stepStmt, evalOp, Tile.scalar_data,
      Option.bind_some, Bool.false_eq_true, if_false, if_true]
    all_goals rw [stepStmts.cons_some (stepStmt_assign_eq_some hi), stepStmts.nil]
    all_goals rfl
  rw [stepStmts.cons_some hb,
    stepStmts.cons_some (stepStmt_assign_eq_some (split_eval p (loaded p initial slot even)))]
  change stepStmts _ (splitState p initial slot even) = _
  have hc : stepStmts (if cast then [.assign .real [p.BK] "tiled_a" (.ref .real [p.BK] "tiled_a")] else [])
      (splitState p initial slot even) = some (castState p initial slot cast even) := by
    cases cast
    · simp [castState]
    · simp only [if_true]
      rw [stepStmts.cons_some (stepStmt_assign_eq_some (by
        simp [splitState, loaded] : evalOp (.ref .real [p.BK] "tiled_a")
          (splitState p initial slot even) = some ⟨fun i => some (inputValues p initial even i.1)⟩)), stepStmts.nil]
      rfl
  rw [stepStmts.append_some hc]
  have hw := weight_eval p initial (castState p initial slot cast even) slot
    (by cases cast <;> simp [castState, splitState, loaded, offsets, afterPrelude])
    (by cases cast <;> simp [castState, splitState, loaded, offsets, afterPrelude])
    (by cases cast <;> simp [castState, splitState])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hw)]
  have ho := output_eval p initial (weightState p initial slot cast even) slot
    (by cases cast <;> simp [weightState, castState, splitState, loaded, offsets, afterPrelude])
    (by cases cast <;> simp [weightState, castState, splitState, loaded, offsets, afterPrelude])
    (by cases cast <;> simp [weightState, castState, splitState])
  change stepStmts _ (weightState p initial slot cast even) = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some ho), stepStmts.nil]
  rfl

theorem setup_registers (p : Parameters) (initial : BlockState) (slot : Nat) (cast even : Bool) :
    (p.geometry initial slot).Registers (inputValues p initial even) (afterSetup p initial slot cast even) := by
  constructor <;> cases cast <;>
    simp [afterSetup, weightState, castState, splitState, loaded, offsets, Parameters.geometry]

@[simp] theorem setup_mem (p : Parameters) (initial : BlockState) (slot : Nat) (cast even : Bool) :
    (afterSetup p initial slot cast even).mem = initial.mem := by
  cases cast <;> rfl

@[simp] theorem setup_read (p : Parameters) (initial : BlockState) (slot : Nat) (cast even : Bool) :
    (afterSetup p initial slot cast even).readMem = initial.readMem := by
  cases cast <;> rfl

@[simp] theorem setup_pids (p : Parameters) (initial : BlockState) (slot : Nat) (cast even : Bool) :
    (afterSetup p initial slot cast even).pids = initial.pids := by
  cases cast <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealSetup
