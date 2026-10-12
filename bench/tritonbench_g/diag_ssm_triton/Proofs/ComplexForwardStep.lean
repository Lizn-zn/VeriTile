import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexForwardSource
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexMemory
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardStep
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def realTile {B : Nat} (z : TileIndex [B] → ℂ) : Tile .real [B] :=
  ⟨fun j => some (z j).re⟩
def imagTile {B : Nat} (z : TileIndex [B] → ℂ) : Tile .real [B] :=
  ⟨fun j => some (z j).im⟩

def offset {B : Nat} (batch dim t : Nat) (col : TileIndex [B] → Nat) : TileIndex [B] → Nat :=
  fun j => t*(batch*dim)+col j

/-- Values read by the two masked loads, independent of their register names. -/
def input {B : Nat} (u : BlockState) (x : RegionName) (off : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ := fun j =>
  ⟨if mask j then u.readMem x (off j*2) else 0,
   if mask j then u.readMem x (off j*2+1) else 0⟩

def next {B : Nat} (u : BlockState) (x : RegionName) (off : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (state transition : TileIndex [B] → ℂ) : TileIndex [B] → ℂ :=
  fun j => state j*transition j+input u x off mask j

/-- Concrete successor, including the source's temporary registers, both
stores and the final tuple assignment. -/
def result {B : Nat} (u : BlockState) (x y : RegionName) (off : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (state transition : TileIndex [B] → ℂ) : BlockState :=
  let xs := input u x off mask
  let ns := next u x off mask state transition
  let indexed := u.setReg "offsets" .nat [B] ⟨fun j => off j*2⟩
  let realLoaded := indexed.setReg "x_real" .real [B] (realTile xs)
  let imagLoaded := realLoaded.setReg "x_imag" .real [B] (imagTile xs)
  let realUpdated := imagLoaded.setReg "new_s_real" .real [B] (realTile ns)
  let loaded := realUpdated.setReg "new_s_imag" .real [B] (imagTile ns)
  let stored := ComplexMemory.store y off ns mask loaded
  let realAssigned := stored.setReg "s_real" .real [B] (realTile ns)
  realAssigned.setReg "s_imag" .real [B] (imagTile ns)

/-- One complete source iteration is complex multiplication followed by
addition, with the source mask and the two interleaved stores unchanged. -/
theorem body_step (x y : RegionName) (batch dim B t : Nat) (u : BlockState)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (state transition : TileIndex [B] → ℂ)
    (ht : u.regs .nat [] "t" = some (Tile.scalar t))
    (hc : u.regs .nat [B] "col_offsets" = some ⟨col⟩)
    (hm : u.regs .bool [B] "mask" = some ⟨mask⟩)
    (hsr : u.regs .real [B] "s_real" = some (realTile state))
    (hsi : u.regs .real [B] "s_imag" = some (imagTile state))
    (hlr : u.regs .real [B] "lambda_real" = some (realTile transition))
    (hli : u.regs .real [B] "lambda_imag" = some (imagTile transition)) :
    stepStmts (ComplexForwardSource.body x y batch dim B) u =
      some (result u x y (offset batch dim t col) mask state transition) := by
  simp only [ComplexForwardSource.body,Erasure.complexForward,
    ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,ComputeKernel.surfaceBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    Except.bind,bind,Bool.false_eq_true,↓reduceIte,List.drop]
  simp only [stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,bind,pure,
    Option.map,Option.pure_def,Broadcast.leftIndex,Broadcast.rightIndex,Tile.vec,
    String.reduceEq,Tile.bop,Tile.cop,Tile.scalar,NumericDType.mul,NumericDType.add,NumericDType.sub,
    WithBot.realMul,WithBot.realAdd,WithBot.realSub,Option.map₂_some_some,
    Region.cast,BlockState.writeMemTyped_real,BlockState.foldl_writeMem_masked_regs,
    BlockState.setReg_same,BlockState.setReg_ne_name,BlockState.setReg_pids,
    BlockState.setReg_readMemValue,BlockState.setReg_readMem,BlockState.readMemValue_real,
    Option.bind_some,Option.map_some,Bool.true_eq_false,↓reduceIte,ne_eq,not_false_eq_true,
    ht,hc,hm,hsr,hsi,hlr,hli,realTile,imagTile,Nat.mul_assoc,
    FloatDType.storeValue,FloatDType.toWithBot,WithBot.unbotD_some]
  simp only [result,ComplexMemory.store,realTile,imagTile,next,input,offset,
    Complex.add_re,Complex.add_im,Complex.mul_re,Complex.mul_im]
  simp only [apply_ite,Option.map₂_some_some,WithBot.unbotD_some]
  rfl


/-- Register facts carried by the loop. The loop counter is supplied by
forLoop at each step, so it is not part of this persistent context. -/
structure Registers {B : Nat} (u : BlockState) (col : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (state transition : TileIndex [B] → ℂ) : Prop where
  columns : u.regs .nat [B] "col_offsets" = some ⟨col⟩
  active : u.regs .bool [B] "mask" = some ⟨mask⟩
  realState : u.regs .real [B] "s_real" = some (realTile state)
  imagState : u.regs .real [B] "s_imag" = some (imagTile state)
  realTransition : u.regs .real [B] "lambda_real" = some (realTile transition)
  imagTransition : u.regs .real [B] "lambda_imag" = some (imagTile transition)

theorem Registers.setCounter {B : Nat} {u : BlockState} {col : TileIndex [B] → Nat}
    {mask : TileIndex [B] → Bool} {state transition : TileIndex [B] → ℂ}
    (h : Registers u col mask state transition) (t : Nat) :
    Registers (u.setReg "t" .nat [] (Tile.scalar t)) col mask state transition := by
  constructor
  · simpa using h.columns
  · simpa using h.active
  · simpa using h.realState
  · simpa using h.imagState
  · simpa using h.realTransition
  · simpa using h.imagTransition

theorem result_registers {B : Nat} (u : BlockState) (x y : RegionName)
    (off col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (state transition : TileIndex [B] → ℂ) (h : Registers u col mask state transition) :
    Registers (result u x y off mask state transition) col mask
      (next u x off mask state transition) transition := by
  constructor
  · simpa [result] using h.columns
  · simpa [result] using h.active
  · simp [result]
  · simp [result]
  · simpa [result] using h.realTransition
  · simpa [result] using h.imagTransition

@[simp] theorem result_pids {B : Nat} (u : BlockState) (x y : RegionName)
    (off : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (state transition : TileIndex [B] → ℂ) :
    (result u x y off mask state transition).pids = u.pids := by
  simp [result]

theorem result_read {B : Nat} (u : BlockState) (x y : RegionName)
    (off : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (state transition : TileIndex [B] → ℂ) (hi : Function.Injective off)
    (j : TileIndex [B]) (hm : mask j = Bool.true) :
    ComplexMemory.read (result u x y off mask state transition) y (off j) =
      next u x off mask state transition j := by
  simp only [result,ComplexMemory.read_setReg]
  exact ComplexMemory.store_read _ _ _ _ _ hi j hm

theorem result_frame {B : Nat} (u : BlockState) (x y : RegionName)
    (off : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (state transition : TileIndex [B] → ℂ) (q : RegionName) (o : Nat)
    (h : q ≠ y ∨ ∀ j, mask j = Bool.true → off j*2 ≠ o ∧ off j*2+1 ≠ o) :
    (result u x y off mask state transition).mem q o = u.mem q o := by
  simp only [result,BlockState.setReg_mem]
  rw [ComplexMemory.store_frame _ _ _ _ _ q o h]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardStep
