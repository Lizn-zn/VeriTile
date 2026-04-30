/-
VeriTile.Examples.FusedSiLU

Fused SiLU kernel and unfused pipeline over the typed Triton core.
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  x        := tl.load($(xReg) + offsets)
  gate     := tl.load($(gateReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  z        := x * gate
  silu     := z * tl.sigmoid(z)
  y        := residual + silu
  tl.store($(outReg) + offsets, y)
}

def siluStepGate (xReg gateReg zReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  x       := tl.load($(xReg) + offsets)
  gate    := tl.load($(gateReg) + offsets)
  z       := x * gate
  tl.store($(zReg) + offsets, z)
}

def siluStepSilu (zReg siluReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  z       := tl.load($(zReg) + offsets)
  silu    := z * tl.sigmoid(z)
  tl.store($(siluReg) + offsets, silu)
}

def siluStepResidual (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  silu     := tl.load($(siluReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  y        := residual + silu
  tl.store($(outReg) + offsets, y)
}

noncomputable def execUnfusedSiLU
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (s : BlockState) : Option BlockState :=
  match exec (siluStepGate xReg gateReg zReg blockSize) s with
  | none => none
  | some s1 =>
    match exec (siluStepSilu zReg siluReg blockSize) s1 with
    | none => none
    | some s2 => exec (siluStepResidual siluReg residualReg outReg blockSize) s2

noncomputable def fusedSiLUSpec {blockSize : Nat}
    (xs gates residuals : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  residuals i + (xs i * gates i) * Real.sigmoid (xs i * gates i)

noncomputable def gateSpec {blockSize : Nat}
    (xs gates : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  xs i * gates i

noncomputable def siluSpec {blockSize : Nat}
    (zs : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  zs i * Real.sigmoid (zs i)

theorem fused_silu_correct
    (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (_h_x   : InputLoadedAt s xReg blockSize xs)
    (_h_g   : InputLoadedAt s gateReg blockSize gates)
    (_h_res : InputLoadedAt s residualReg blockSize residuals) :
    ∀ i : Fin blockSize,
      observeAt (exec (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i
        = some (fusedSiLUSpec xs gates residuals i) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, fusedSiLUKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        BlockState.setReg, BlockState.readMem, fusedSiLUSpec]
  unfold InputLoadedAt at _h_x _h_g _h_res
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_g, _h_res]

theorem silu_step_gate_correct
    (xReg gateReg zReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (xs gates : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_g : InputLoadedAt s gateReg blockSize gates) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepGate xReg gateReg zReg blockSize) s)
                zReg blockSize s.pid i
        = some (gateSpec xs gates i) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, siluStepGate, stepStmts, stepStmt, evalOp,
        Tile.bop, NumericDType.add, NumericDType.mul,
        BlockState.setReg, BlockState.readMem, gateSpec]
  unfold InputLoadedAt at _h_x _h_g
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_g]

theorem silu_step_gate_preserves
    (xReg gateReg zReg keepReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (xs gates : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_g : InputLoadedAt s gateReg blockSize gates)
    (h_keep : keepReg ≠ zReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepGate xReg gateReg zReg blockSize) s)
                keepReg blockSize s.pid i
        = some (s.readMem keepReg (s.pid * blockSize + i.val)) := by
  intro i
  simp [observeAt, exec, siluStepGate, stepStmts, stepStmt, evalOp,
        Tile.bop, NumericDType.add, NumericDType.mul,
        BlockState.setReg, BlockState.readMem]
  unfold InputLoadedAt at _h_x _h_g
  simp_rw [_h_x, _h_g]
  rw [BlockState.scatter_preserves_other_region zReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => xs k.1 * gates k.1)
        keepReg h_keep (s.pid * blockSize + i.val)]

theorem silu_step_silu_correct
    (zReg siluReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (zs : Fin blockSize → ℝ)
    (_h_z : InputLoadedAt s zReg blockSize zs) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepSilu zReg siluReg blockSize) s)
                siluReg blockSize s.pid i
        = some (siluSpec zs i) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, siluStepSilu, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        BlockState.setReg, BlockState.readMem, siluSpec]
  unfold InputLoadedAt at _h_z
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_z]

theorem silu_step_silu_preserves
    (zReg siluReg keepReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (zs : Fin blockSize → ℝ)
    (_h_z : InputLoadedAt s zReg blockSize zs)
    (h_keep : keepReg ≠ siluReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepSilu zReg siluReg blockSize) s)
                keepReg blockSize s.pid i
        = some (s.readMem keepReg (s.pid * blockSize + i.val)) := by
  intro i
  simp [observeAt, exec, siluStepSilu, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
        BlockState.setReg, BlockState.readMem]
  unfold InputLoadedAt at _h_z
  simp_rw [_h_z]
  rw [BlockState.scatter_preserves_other_region siluReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => zs k.1 * Real.sigmoid (zs k.1))
        keepReg h_keep (s.pid * blockSize + i.val)]

theorem silu_step_residual_correct
    (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (silus residuals : Fin blockSize → ℝ)
    (_h_silu : InputLoadedAt s siluReg blockSize silus)
    (_h_res : InputLoadedAt s residualReg blockSize residuals) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepResidual siluReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i
        = some (residuals i + silus i) := by
  intro i
  have h_inj :
      Function.Injective (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) :=
    injective_offset_singleton (s.pid * blockSize)
  simp [observeAt, exec, siluStepResidual, stepStmts, stepStmt, evalOp,
        Tile.bop, NumericDType.add, NumericDType.mul,
        BlockState.setReg, BlockState.readMem]
  unfold InputLoadedAt at _h_silu _h_res
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_silu, _h_res]

theorem silu_step_residual_preserves
    (siluReg residualReg outReg keepReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (silus residuals : Fin blockSize → ℝ)
    (_h_silu : InputLoadedAt s siluReg blockSize silus)
    (_h_res : InputLoadedAt s residualReg blockSize residuals)
    (h_keep : keepReg ≠ outReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (siluStepResidual siluReg residualReg outReg blockSize) s)
                keepReg blockSize s.pid i
        = some (s.readMem keepReg (s.pid * blockSize + i.val)) := by
  intro i
  simp [observeAt, exec, siluStepResidual, stepStmts, stepStmt, evalOp,
        Tile.bop, NumericDType.add, NumericDType.mul,
        BlockState.setReg, BlockState.readMem]
  unfold InputLoadedAt at _h_silu _h_res
  simp_rw [_h_silu, _h_res]
  rw [BlockState.scatter_preserves_other_region outReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => residuals k.1 + silus k.1)
        keepReg h_keep (s.pid * blockSize + i.val)]

set_option maxHeartbeats 800000 in
theorem unfused_silu_correct
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (_hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (_h_x   : InputLoadedAt s xReg blockSize xs)
    (_h_g   : InputLoadedAt s gateReg blockSize gates)
    (_h_res : InputLoadedAt s residualReg blockSize residuals)
    (_h_zRes    : zReg ≠ residualReg)
    (_h_siluRes : siluReg ≠ residualReg) :
    ∀ i : Fin blockSize,
      observeAt
        (execUnfusedSiLU xReg gateReg residualReg zReg siluReg outReg blockSize s)
        outReg blockSize s.pid i
        = some (fusedSiLUSpec xs gates residuals i) := by
  intro i
  unfold execUnfusedSiLU
  cases h_gate : exec (siluStepGate xReg gateReg zReg blockSize) s with
  | none =>
      have hz :=
        silu_step_gate_correct xReg gateReg zReg blockSize _hN s xs gates
          _h_x _h_g i
      rw [h_gate] at hz
      simp [observeAt] at hz
  | some s1 =>
      have hpid1 : s1.pid = s.pid := exec_pid h_gate
      have h_z : InputLoadedAt s1 zReg blockSize (gateSpec xs gates) := by
        intro j
        have hzj :=
          silu_step_gate_correct xReg gateReg zReg blockSize _hN s xs gates
            _h_x _h_g j
        rw [h_gate] at hzj
        simp [observeAt, BlockState.readMem] at hzj
        rw [hpid1]
        exact hzj
      have h_res1 : InputLoadedAt s1 residualReg blockSize residuals := by
        intro j
        have hrj :=
          silu_step_gate_preserves xReg gateReg zReg residualReg blockSize _hN s
            xs gates _h_x _h_g (fun h => _h_zRes h.symm) j
        rw [h_gate] at hrj
        simp [observeAt, BlockState.readMem] at hrj
        rw [hpid1]
        rw [hrj]
        exact _h_res j
      cases h_silu : exec (siluStepSilu zReg siluReg blockSize) s1 with
      | none =>
          have hsj :=
            silu_step_silu_correct zReg siluReg blockSize _hN s1
              (gateSpec xs gates) h_z i
          rw [h_silu] at hsj
          simp [observeAt] at hsj
      | some s2 =>
          have hpid2 : s2.pid = s1.pid := exec_pid h_silu
          have hpid2s : s2.pid = s.pid := hpid2.trans hpid1
          have h_silu_loaded :
              InputLoadedAt s2 siluReg blockSize (siluSpec (gateSpec xs gates)) := by
            intro j
            have hsj :=
              silu_step_silu_correct zReg siluReg blockSize _hN s1
                (gateSpec xs gates) h_z j
            rw [h_silu] at hsj
            simp [observeAt, BlockState.readMem] at hsj
            rw [hpid2]
            exact hsj
          have h_res2 : InputLoadedAt s2 residualReg blockSize residuals := by
            intro j
            have hrj :=
              silu_step_silu_preserves zReg siluReg residualReg blockSize _hN s1
                (gateSpec xs gates) h_z (fun h => _h_siluRes h.symm) j
            rw [h_silu] at hrj
            simp [observeAt, BlockState.readMem] at hrj
            rw [hpid2]
            rw [hrj]
            exact h_res1 j
          have hout :=
            silu_step_residual_correct siluReg residualReg outReg blockSize _hN s2
              (siluSpec (gateSpec xs gates)) residuals h_silu_loaded h_res2 i
          rw [hpid2s] at hout
          simp [h_silu]
          rw [hout]
          simp [fusedSiLUSpec, siluSpec, gateSpec]

theorem silu_kernels_refinement
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (h_x   : InputLoadedAt s xReg blockSize xs)
    (h_g   : InputLoadedAt s gateReg blockSize gates)
    (h_res : InputLoadedAt s residualReg blockSize residuals)
    (h_zRes    : zReg ≠ residualReg)
    (h_siluRes : siluReg ≠ residualReg) :
    ∀ i : Fin blockSize,
      observeAt (exec (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s)
                outReg blockSize s.pid i =
      observeAt
        (execUnfusedSiLU xReg gateReg residualReg zReg siluReg outReg blockSize s)
        outReg blockSize s.pid i := by
  intro i
  rw [fused_silu_correct
        xReg gateReg residualReg outReg blockSize hN s xs gates residuals h_x h_g h_res i,
      unfused_silu_correct
        xReg gateReg residualReg zReg siluReg outReg blockSize hN s xs gates residuals
        h_x h_g h_res h_zRes h_siluRes i]

end VeriTile.Examples
