/-
VeriTile.Examples.FusedSiLU

Fused SiLU kernel and unfused pipeline over the typed Triton core.

This is a kernel-refinement example: the fused kernel

  y = residual + (x * gate) * sigmoid(x * gate)

is shown equivalent to a three-kernel pipeline that materializes the
intermediate `z = x * gate`, then `silu = z * sigmoid(z)`, then adds the
residual. The proof is intentionally memory-aware: the unfused pipeline writes
temporary regions `zReg` and `siluReg`, and the refinement theorem requires
these temporaries not to alias `residualReg` so the residual input survives
until the final stage.

Source Triton (`.py` reference, aligned single-block flavour; assumes
`BLOCK_SIZE` equals the logical vector length, so no boundary mask is used):

```python
@triton.jit
def fused_silu_kernel(x_ptr, gate_ptr, residual_ptr, out_ptr,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets)
    gate = tl.load(gate_ptr + offsets)
    residual = tl.load(residual_ptr + offsets)

    z = x * gate
    silu = z * tl.sigmoid(z)
    y = residual + silu

    tl.store(out_ptr + offsets, y)

@triton.jit
def silu_step_gate(x_ptr, gate_ptr, z_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    gate = tl.load(gate_ptr + offsets)
    tl.store(z_ptr + offsets, x * gate)

@triton.jit
def silu_step_silu(z_ptr, silu_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    z = tl.load(z_ptr + offsets)
    tl.store(silu_ptr + offsets, z * tl.sigmoid(z))

@triton.jit
def silu_step_residual(silu_ptr, residual_ptr, out_ptr,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    silu = tl.load(silu_ptr + offsets)
    residual = tl.load(residual_ptr + offsets)
    tl.store(out_ptr + offsets, residual + silu)
```
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

def fusedSiLUKernel (xReg gateReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
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

def siluStepGate (xReg gateReg zReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  x       := tl.load($(xReg) + offsets)
  gate    := tl.load($(gateReg) + offsets)
  z       := x * gate
  tl.store($(zReg) + offsets, z)
}

def siluStepSilu (zReg siluReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange($(blockSize))
  z       := tl.load($(zReg) + offsets)
  silu    := z * tl.sigmoid(z)
  tl.store($(siluReg) + offsets, silu)
}

def siluStepResidual (siluReg residualReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid      := tl.program_id(0)
  offsets  := pid * $(blockSize) + tl.arange($(blockSize))
  silu     := tl.load($(siluReg) + offsets)
  residual := tl.load($(residualReg) + offsets)
  y        := residual + silu
  tl.store($(outReg) + offsets, y)
}

def unfusedSiLUKernel
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) : ComputeKernel :=
  let body : List Stmt :=
    (siluStepGate xReg gateReg zReg blockSize).body ++
    (siluStepSilu zReg siluReg blockSize).body ++
    (siluStepResidual siluReg residualReg outReg blockSize).body
  ComputeKernel.fromKernelBody [xReg, gateReg, residualReg] [outReg] body

private theorem stepStmts_append (xs ys : List Stmt) (s : BlockState) :
    stepStmts (xs ++ ys) s = (stepStmts xs s).bind (fun s' => stepStmts ys s') := by
  induction xs generalizing s with
  | nil => simp
  | cons st rest ih =>
      simp [stepStmts]
      cases stepStmt st s <;> simp [ih]

noncomputable def execUnfusedSiLU
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (s : BlockState) : Option BlockState :=
  match exec (siluStepGate xReg gateReg zReg blockSize) s with
  | none => none
  | some s1 =>
    match exec (siluStepSilu zReg siluReg blockSize) s1 with
    | none => none
    | some s2 => exec (siluStepResidual siluReg residualReg outReg blockSize) s2

theorem exec_unfusedSiLUKernel
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (s : BlockState) :
    exec (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize) s =
      execUnfusedSiLU xReg gateReg residualReg zReg siluReg outReg blockSize s := by
  simp [unfusedSiLUKernel, execUnfusedSiLU, exec, stepStmts_append]
  cases h1 : stepStmts (siluStepGate xReg gateReg zReg blockSize).body s <;> simp
  case some s1 =>
    cases h2 : stepStmts (siluStepSilu zReg siluReg blockSize).body s1 <;> simp

/-- Per-lane fused-SiLU output. Bound here to the kernel's `(xs, gates, residuals)`
input layout; delegates the math to `Triton.TiledActivation.fusedSiLU`. -/
noncomputable def fusedSiLUSpec {blockSize : Nat}
    (xs gates residuals : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  Triton.TiledActivation.fusedSiLU (xs i) (gates i) (residuals i)

/-- Per-lane gate output `xs i * gates i`. Pure pointwise multiply; kept local
because there is no reusable math operator at this granularity. -/
noncomputable def gateSpec {blockSize : Nat}
    (xs gates : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  xs i * gates i

/-- Per-lane SiLU output. Delegates to `Triton.TiledActivation.silu`. -/
noncomputable def siluSpec {blockSize : Nat}
    (zs : Fin blockSize → ℝ) (i : Fin blockSize) : ℝ :=
  Triton.TiledActivation.silu (zs i)

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
        fusedSiLUSpec, Triton.TiledActivation.fusedSiLU,
        Triton.TiledActivation.silu]
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
        gateSpec]
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
        Tile.bop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_x _h_g
  simp_rw [_h_x, _h_g]
  rw [BlockState.scatter_preserves_other_region zReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => xs k.1 * gates k.1)
        keepReg h_keep (s.pid * blockSize + i.val)]
  simp [BlockState.pid_eq]

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
        siluSpec, Triton.TiledActivation.silu]
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
        Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_z
  simp_rw [_h_z]
  rw [BlockState.scatter_preserves_other_region siluReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => zs k.1 * Real.sigmoid (zs k.1))
        keepReg h_keep (s.pid * blockSize + i.val)]
  simp [BlockState.pid_eq]

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
        Tile.bop, NumericDType.add, NumericDType.mul]
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
        Tile.bop, NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_silu _h_res
  simp_rw [_h_silu, _h_res]
  rw [BlockState.scatter_preserves_other_region outReg
        (fun k : TileIndex [blockSize] => s.pid * blockSize + k.1.val)
        (fun k : TileIndex [blockSize] => residuals k.1 + silus k.1)
        keepReg h_keep (s.pid * blockSize + i.val)]
  simp [BlockState.pid_eq]

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
        simp [observeAt] at hzj
        rw [hpid1]
        exact hzj
      have h_res1 : InputLoadedAt s1 residualReg blockSize residuals := by
        intro j
        have hrj :=
          silu_step_gate_preserves xReg gateReg zReg residualReg blockSize _hN s
            xs gates _h_x _h_g (fun h => _h_zRes h.symm) j
        rw [h_gate] at hrj
        simp [observeAt] at hrj
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
            simp [observeAt] at hsj
            rw [hpid2]
            exact hsj
          have h_res2 : InputLoadedAt s2 residualReg blockSize residuals := by
            intro j
            have hrj :=
              silu_step_silu_preserves zReg siluReg residualReg blockSize _hN s1
                (gateSpec xs gates) h_z (fun h => _h_siluRes h.symm) j
            rw [h_silu] at hrj
            simp [observeAt] at hrj
            rw [hpid2]
            rw [hrj]
            exact h_res1 j
          have hout :=
            silu_step_residual_correct siluReg residualReg outReg blockSize _hN s2
              (siluSpec (gateSpec xs gates)) residuals h_silu_loaded h_res2 i
          rw [hpid2s] at hout
          simp [h_silu]
          rw [hout]
          simp [fusedSiLUSpec, siluSpec, gateSpec,
                Triton.TiledActivation.fusedSiLU, Triton.TiledActivation.silu]

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

/-- View-level surface for `silu_kernels_refinement`. -/
theorem silu_kernels_refinement_exec_view
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_g : TensorView.loaded s (programTileView s gateReg blockSize)
      (fun idx : TileIndex [blockSize] => gates idx.1))
    (h_res : TensorView.loaded s (programTileView s residualReg blockSize)
      (fun idx : TileIndex [blockSize] => residuals idx.1))
    (h_zRes : zReg ≠ residualReg)
    (h_siluRes : siluReg ≠ residualReg) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe
          (exec (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s)
          (programTileView s outReg blockSize) idx =
      TensorView.observe
          (execUnfusedSiLU xReg gateReg residualReg zReg siluReg outReg blockSize s)
          (programTileView s outReg blockSize) idx := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  have hg := inputLoadedAt_of_programTileView_loaded (s := s) (region := gateReg)
    (N := blockSize) (xs := gates) h_g
  have hres := inputLoadedAt_of_programTileView_loaded (s := s) (region := residualReg)
    (N := blockSize) (xs := residuals) h_res
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using silu_kernels_refinement xReg gateReg residualReg zReg siluReg outReg
      blockSize hN s xs gates residuals hx hg hres h_zRes h_siluRes idx.1

/-! ### Whole-memory frame lemmas for the writes-equality refinement surface

`ComputeRefine.Realizes` compares the two final memories cell-by-cell, so the
refinement proof needs `.mem`-level frames: each kernel touches only its own
store target. Proved by symbolic execution down to the store's `writeMem`
scatter, then the generic scatter frame lemma. -/

private theorem fused_silu_mem_frame
    (xReg gateReg residualReg outReg : RegionName) (blockSize : Nat)
    {s s' : BlockState}
    (h : exec (fusedSiLUKernel xReg gateReg residualReg outReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ outReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [exec, fusedSiLUKernel, stepStmts, stepStmt, evalOp, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _

private theorem silu_step_gate_mem_frame
    (xReg gateReg zReg : RegionName) (blockSize : Nat)
    {s s' : BlockState}
    (h : exec (siluStepGate xReg gateReg zReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ zReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [exec, siluStepGate, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _

private theorem silu_step_silu_mem_frame
    (zReg siluReg : RegionName) (blockSize : Nat)
    {s s' : BlockState}
    (h : exec (siluStepSilu zReg siluReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ siluReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [exec, siluStepSilu, stepStmts, stepStmt, evalOp, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _

private theorem silu_step_residual_mem_frame
    (siluReg residualReg outReg : RegionName) (blockSize : Nat)
    {s s' : BlockState}
    (h : exec (siluStepResidual siluReg residualReg outReg blockSize) s = some s')
    {r : RegionName} (hr : r ≠ outReg) (o : Nat) :
    s'.mem r o = s.mem r o := by
  simp [exec, siluStepResidual, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul] at h
  subst h
  exact BlockState.foldl_writeMem_mem_preserve_other_region _ _ _ r hr o _

set_option maxHeartbeats 1600000 in
/-- Compute-facing surface for `silu_kernels_refinement`: writes-equality
refinement. From the same initial state, the fused kernel and the unfused
pipeline perform the same writes — the two final memories agree at every cell
outside the declared scratch regions `zReg`/`siluReg` (the pipeline's
temporaries). -/
theorem silu_kernels_refinement_view
    (xReg gateReg residualReg zReg siluReg outReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState)
    (xs gates residuals : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_g : TensorView.loaded s (programTileView s gateReg blockSize)
      (fun idx : TileIndex [blockSize] => gates idx.1))
    (h_res : TensorView.loaded s (programTileView s residualReg blockSize)
      (fun idx : TileIndex [blockSize] => residuals idx.1))
    (h_zRes : zReg ≠ residualReg)
    (h_siluRes : siluReg ≠ residualReg) :
    ComputeRefine.Realizes
      (fusedSiLUKernel xReg gateReg residualReg outReg blockSize)
      (unfusedSiLUKernel xReg gateReg residualReg zReg siluReg outReg blockSize)
      s [zReg, siluReg] := by
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  have hg := inputLoadedAt_of_programTileView_loaded (s := s) (region := gateReg)
    (N := blockSize) (xs := gates) h_g
  have hres := inputLoadedAt_of_programTileView_loaded (s := s) (region := residualReg)
    (N := blockSize) (xs := residuals) h_res
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r hr o
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hr
  obtain ⟨hrz, hrsilu⟩ := hr
  have hRsplit := hR
  rw [exec_unfusedSiLUKernel] at hRsplit
  unfold execUnfusedSiLU at hRsplit
  cases h1 : exec (siluStepGate xReg gateReg zReg blockSize) s with
  | none => simp [h1] at hRsplit
  | some s1 =>
    simp only [h1] at hRsplit
    cases h2 : exec (siluStepSilu zReg siluReg blockSize) s1 with
    | none => simp [h2] at hRsplit
    | some s2 =>
      simp only [h2] at hRsplit
      -- hRsplit : exec (siluStepResidual …) s2 = some rhs'
      have hpid1 : s1.pid = s.pid := exec_pid h1
      have hpid2 : s2.pid = s1.pid := exec_pid h2
      have hpid2s : s2.pid = s.pid := hpid2.trans hpid1
      by_cases hrOut : r = outReg
      · -- both sides scatter the same values to the same `outReg` offsets
        subst hrOut
        -- pipeline stage facts at `s2` (mirrors `unfused_silu_correct`)
        have h_z : InputLoadedAt s1 zReg blockSize (gateSpec xs gates) := by
          intro j
          have hzj :=
            silu_step_gate_correct xReg gateReg zReg blockSize hN s xs gates hx hg j
          rw [h1] at hzj
          simp [observeAt] at hzj
          rw [hpid1]
          exact hzj
        have h_silu_loaded :
            InputLoadedAt s2 siluReg blockSize (siluSpec (gateSpec xs gates)) := by
          intro j
          have hsj :=
            silu_step_silu_correct zReg siluReg blockSize hN s1 (gateSpec xs gates) h_z j
          rw [h2] at hsj
          simp [observeAt] at hsj
          rw [hpid2]
          exact hsj
        have h_res1 : InputLoadedAt s1 residualReg blockSize residuals := by
          intro j
          have hrj :=
            silu_step_gate_preserves xReg gateReg zReg residualReg blockSize hN s
              xs gates hx hg (fun h => h_zRes h.symm) j
          rw [h1] at hrj
          simp [observeAt] at hrj
          rw [hpid1, hrj]
          exact hres j
        have h_res2 : InputLoadedAt s2 residualReg blockSize residuals := by
          intro j
          have hrj :=
            silu_step_silu_preserves zReg siluReg residualReg blockSize hN s1
              (gateSpec xs gates) h_z (fun h => h_siluRes h.symm) j
          rw [h2] at hrj
          simp [observeAt] at hrj
          rw [hpid2, hrj]
          exact h_res1 j
        -- symbolic execution of the two stores
        simp [exec, fusedSiLUKernel, stepStmts, stepStmt, evalOp, Tile.bop, Tile.uop,
              NumericDType.add, NumericDType.mul] at hL
        simp [exec, siluStepResidual, stepStmts, stepStmt, Tile.bop,
              NumericDType.add, NumericDType.mul] at hRsplit
        subst hL
        subst hRsplit
        have hpids : s2.pids 0 = s.pids 0 := hpid2s
        rw [hpids]
        refine BlockState.foldl_writeMem_mem_congr _ _ _ _ ?_ r o _ _ ?_
        · -- per-lane written values agree
          intro k _
          have hxk := hx k.1
          have hgk := hg k.1
          have hresk := hres k.1
          have hsiluk := h_silu_loaded k.1
          have hres2k := h_res2 k.1
          rw [hpid2s] at hsiluk hres2k
          simp [hxk, hgk, hresk, hsiluk, hres2k, siluSpec, gateSpec,
                Triton.TiledActivation.silu]
        · -- the two scatter bases agree at the observed cell
          simp only [BlockState.setReg_mem]
          rw [silu_step_silu_mem_frame zReg siluReg blockSize h2 hrsilu o,
              silu_step_gate_mem_frame xReg gateReg zReg blockSize h1 hrz o]
      · -- `r` is none of `zReg`/`siluReg`/`outReg`: both sides preserve the cell
        rw [fused_silu_mem_frame xReg gateReg residualReg outReg blockSize hL hrOut o,
            silu_step_residual_mem_frame siluReg residualReg outReg blockSize hRsplit hrOut o,
            silu_step_silu_mem_frame zReg siluReg blockSize h2 hrsilu o,
            silu_step_gate_mem_frame xReg gateReg zReg blockSize h1 hrz o]

end VeriTile.Examples
