import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardSource
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexForwardStep
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardStep
open VeriTile.Triton ComplexForwardStep
noncomputable section
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

/-- Carry and transition-gradient accumulators use the source's conjugated
convention. The two stores conjugate them back to observable gradients. -/
structure Registers {B : Nat} (u : BlockState) (col : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (carry acc transition : TileIndex [B] → ℂ) : Prop where
  columns : u.regs .nat [B] "col_offsets" = some ⟨col⟩
  active : u.regs .bool [B] "mask" = some ⟨mask⟩
  carryReal : u.regs .real [B] "grad_s_real" = some (realTile carry)
  carryImag : u.regs .real [B] "grad_s_imag" = some (imagTile carry)
  accReal : u.regs .real [B] "grad_lambda_real" = some (realTile acc)
  accImag : u.regs .real [B] "grad_lambda_imag" = some (imagTile acc)
  transitionReal : u.regs .real [B] "lambda_real" = some (realTile transition)
  transitionImag : u.regs .real [B] "lambda_imag" = some (imagTile transition)

def previous {B : Nat} (u : BlockState) (s y : RegionName) (batch dim t : Nat)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool) : TileIndex [B] → ℂ :=
  if t=0 then input u s col mask else input u y (offset batch dim (t-1) col) mask

def gradient {B : Nat} (u : BlockState) (gy : RegionName) (off : TileIndex [B] → Nat)
    (mask : TileIndex [B] → Bool) (carry : TileIndex [B] → ℂ) : TileIndex [B] → ℂ :=
  fun j => starRingEnd ℂ (input u gy off mask j)+carry j

/-- Exact temporary registers and stores after one complete reverse step. -/
def result {B : Nat} (u : BlockState) (s y gx gy : RegionName) (batch dim t : Nat)
    (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (carry acc transition : TileIndex [B] → ℂ) : BlockState :=
  let off := offset batch dim t col
  let outputGradient := fun j => starRingEnd ℂ (input u gy off mask j)
  let prev := previous u s y batch dim t col mask
  let grad := gradient u gy off mask carry
  let newAcc := fun j => acc j+grad j*prev j
  let newCarry := fun j => grad j*transition j
  let timeSet := u.setReg "t" .nat [] (Tile.scalar t)
  let indexed := timeSet.setReg "offsets" .nat [B] ⟨fun j => off j*2⟩
  let gyr := indexed.setReg "grad_y_real" .real [B] (realTile outputGradient)
  let gyi := gyr.setReg "grad_y_imag" .real [B] (imagTile outputGradient)
  let sr := gyi.setReg "s_real" .real [B] (realTile prev)
  let si := sr.setReg "s_imag" .real [B] (imagTile prev)
  let gr := si.setReg "grad_s_real" .real [B] (realTile grad)
  let gi := gr.setReg "grad_s_imag" .real [B] (imagTile grad)
  let xr := gi.setReg "grad_x_real" .real [B] (realTile grad)
  let xi := xr.setReg "grad_x_imag" .real [B] (imagTile grad)
  let ar := xi.setReg "grad_lambda_real" .real [B] (realTile newAcc)
  let ai := ar.setReg "grad_lambda_imag" .real [B] (imagTile newAcc)
  let cr := ai.setReg "grad_s_real" .real [B] (realTile newCarry)
  let ci := cr.setReg "grad_s_imag" .real [B] (imagTile newCarry)
  ComplexMemory.store gx off (fun j => starRingEnd ℂ (grad j)) mask ci

/-- Source pointer subtraction is performed in signed integer arithmetic,
then converted to a natural offset. A positive time index puts it exactly
at the preceding row, including the imaginary component. -/
theorem previous_address (batch dim t c a : Nat) (ht : 0<t) :
    ((Int.ofNat ((t*(batch*dim)+c)*2) - 2*Int.ofNat batch*Int.ofNat dim) + (a : Int)).toNat =
      ((t-1)*(batch*dim)+c)*2+a := by
  have he : (t*(batch*dim)+c)*2 = ((t-1)*(batch*dim)+c)*2+2*batch*dim := by
    have h : t=(t-1)+1 := by omega
    conv_lhs => rw [h]
    ring
  have hi : (Int.ofNat ((t*(batch*dim)+c)*2) - 2*Int.ofNat batch*Int.ofNat dim) + (a : Int) =
      Int.ofNat (((t-1)*(batch*dim)+c)*2+a) := by
    rw [he]
    simp only [Int.ofNat_eq_natCast]
    push_cast
    ring
  rw [hi]
  rfl

theorem previous_address_zero (batch dim t c : Nat) (ht : 0<t) :
    (Int.ofNat ((t*(batch*dim)+c)*2) - 2*Int.ofNat batch*Int.ofNat dim).toNat =
      ((t-1)*(batch*dim)+c)*2 := by
  simpa using previous_address batch dim t c 0 ht


theorem Registers.setCounter {B : Nat} {u : BlockState} {col : TileIndex [B] → Nat}
    {mask : TileIndex [B] → Bool} {carry acc transition : TileIndex [B] → ℂ}
    (h : Registers u col mask carry acc transition) (k : Nat) :
    Registers (u.setReg "i" .nat [] (Tile.scalar k)) col mask carry acc transition := by
  rcases h with ⟨hc,hm,hcr,hci,har,hai,hlr,hli⟩
  constructor
  all_goals simpa using (by assumption)

theorem result_registers {B : Nat} (u : BlockState) (s y gx gy : RegionName)
    (batch dim t : Nat) (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (carry acc transition : TileIndex [B] → ℂ) (h : Registers u col mask carry acc transition) :
    Registers (result u s y gx gy batch dim t col mask carry acc transition) col mask
      (fun j => gradient u gy (offset batch dim t col) mask carry j*transition j)
      (fun j => acc j+gradient u gy (offset batch dim t col) mask carry j *
        previous u s y batch dim t col mask j) transition := by
  constructor
  · simpa [result] using h.columns
  · simpa [result] using h.active
  · simp [result]
  · simp [result]
  · simp [result]
  · simp [result]
  · simpa [result] using h.transitionReal
  · simpa [result] using h.transitionImag

@[simp] theorem result_pids {B : Nat} (u : BlockState) (s y gx gy : RegionName)
    (batch dim t : Nat) (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (carry acc transition : TileIndex [B] → ℂ) :
    (result u s y gx gy batch dim t col mask carry acc transition).pids = u.pids := by
  simp [result]

theorem result_read {B : Nat} (u : BlockState) (s y gx gy : RegionName)
    (batch dim t : Nat) (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (carry acc transition : TileIndex [B] → ℂ) (hi : Function.Injective (offset batch dim t col))
    (j : TileIndex [B]) (hm : mask j=Bool.true) :
    ComplexMemory.read (result u s y gx gy batch dim t col mask carry acc transition)
      gx (offset batch dim t col j) =
        starRingEnd ℂ (gradient u gy (offset batch dim t col) mask carry j) := by
  unfold result
  exact ComplexMemory.store_read _ _ _ _ _ hi j hm

theorem result_frame {B : Nat} (u : BlockState) (s y gx gy : RegionName)
    (batch dim t : Nat) (col : TileIndex [B] → Nat) (mask : TileIndex [B] → Bool)
    (carry acc transition : TileIndex [B] → ℂ) (q : RegionName) (o : Nat)
    (h : q≠gx ∨ ∀ j, mask j=Bool.true → offset batch dim t col j*2≠o ∧ offset batch dim t col j*2+1≠o) :
    (result u s y gx gy batch dim t col mask carry acc transition).mem q o = u.mem q o := by
  unfold result
  rw [ComplexMemory.store_frame _ _ _ _ _ q o h]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardStep
