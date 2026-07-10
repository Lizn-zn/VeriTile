/-
VeriTile.Examples.WelfordKernels

Welford math identities and kernels over the typed Triton core.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.KernelLemmas
import VeriTile.Triton.Math.Reduction
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton
open VeriTile.Triton.TiledReduction.WelfordRec

/-! ## Kernels -/

/-- Two-pass mean/variance kernel.

For each program id, this kernel loads one aligned `blockSize`-element row,
computes the mean with one `tl.sum`, then computes the population variance
with a second `tl.sum` over squared deviations. It writes one scalar mean and
one scalar variance for the row. In the current named-region model those
outputs are observed at offset `0`; the theorem below is for one fixed
`BlockState.pid`.

Source Triton (`.py` reference, aligned single-block flavour; assumes
`BLOCK_SIZE` equals the row length and uses no boundary mask):

```python
@triton.jit
def twopass_welford_kernel(x_ptr, mean_ptr, var_ptr,
                           BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offs)
    s_x = tl.sum(x, axis=0)
    mu = s_x / BLOCK_SIZE
    d = x - mu
    s_d2 = tl.sum(d * d, axis=0)
    v = s_d2 / BLOCK_SIZE

    tl.store(mean_ptr, mu)
    tl.store(var_ptr, v)
```
-/
def twopassWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange($(blockSize))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(blockSize))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(blockSize))
  tl.store($(meanReg), μ)
  tl.store($(varReg), v)
}

/-- Online Welford mean/variance kernel.

This kernel computes the same population mean/variance as
`twopassWelfordKernel`, but uses Welford's one-pass recurrence inside a
Triton `for` loop. The loop maintains scalar registers:

* `M`: running mean after the processed prefix
* `S`: running sum of squared deviations

After all `blockSize` elements, it writes `M` and `S / blockSize`. As above,
this is the aligned single-block version used by the proof; no boundary mask
is modeled in this example.

Source Triton (`.py` reference):

```python
@triton.jit
def online_welford_kernel(x_ptr, mean_ptr, var_ptr,
                          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    M = 0.0
    S = 0.0

    for i in range(0, BLOCK_SIZE):
        xi = tl.load(x_ptr + (pid * BLOCK_SIZE + i))
        delta = xi - M
        M = M + delta / (i + 1)
        delta2 = xi - M
        S = S + delta * delta2

    tl.store(mean_ptr, M)
    tl.store(var_ptr, S / BLOCK_SIZE)
```
-/
def onlineWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(blockSize) {
    xi     := tl.load($(xReg) + (pid * $(blockSize) + i))
    delta  := xi - M
    M      := M + delta / (tl.toReal(i) + 1)
    delta2 := xi - M
    S      := S + delta * delta2
  }
  tl.store($(meanReg), M)
  tl.store($(varReg), S / tl.toReal($(blockSize)))
}

/-! ## Pure Welford Math -/

-- The two-pass/recurrence Welford math (twoPassMean/twoPassS/welfordMean/welfordS +
-- welford_eq_two_pass) now lives in `VeriTile.Triton.TiledReduction.WelfordRec`,
-- opened below; shared with the bench/examples Welford + LayerNorm showcases.

def onlineWelfordLoopBody (xReg : RegionName) (blockSize : Nat) : List Stmt :=
  [Stmt.assign .real [] "xi"
      (Op.load .real (MemAccess.region xReg
        (Op.add .nat .nil
          (Op.mul .nat .nil (Op.ref .nat [] "pid")
            (Op.constNat blockSize))
          (Op.ref .nat [] "i"))) MaskOpt.none),
    Stmt.assign .real [] "delta"
      (Op.sub .real .nil (Op.ref .real [] "xi")
        (Op.ref .real [] "M")),
    Stmt.assign .real [] "M"
      (Op.add .real .nil (Op.ref .real [] "M")
        (Op.div .real .nil (Op.ref .real [] "delta")
          (Op.add .real .nil (Op.ref .nat [] "i").natToReal
            (Op.const 1)))),
    Stmt.assign .real [] "delta2"
      (Op.sub .real .nil (Op.ref .real [] "xi")
        (Op.ref .real [] "M")),
    Stmt.assign .real [] "S"
      (Op.add .real .nil (Op.ref .real [] "S")
        (Op.mul .real .nil (Op.ref .real [] "delta")
          (Op.ref .real [] "delta2")))]

/-- Per-row mean spec. Thin alias for `Triton.TiledReduction.welfordMean`. -/
noncomputable def welfordMeanSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  Triton.TiledReduction.welfordMean xs

/-- Per-row population variance spec. Thin alias for
`Triton.TiledReduction.welfordVar`. -/
noncomputable def welfordVarSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  Triton.TiledReduction.welfordVar xs

theorem twopass_welford_correct
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (_hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_mv : meanReg ≠ varReg) :
    let final := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    final.bind (fun s' => some (s'.readMem meanReg 0))
        = some (welfordMeanSpec xs)
    ∧ final.bind (fun s' => some (s'.readMem varReg 0))
        = some (welfordVarSpec xs) := by
  constructor
  · simp [exec, twopassWelfordKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.natToReal, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        welfordMeanSpec, Triton.TiledReduction.welfordMean,
        Triton.TiledReduction.tileSum]
    unfold InputLoadedAt at _h_x
    simp_rw [_h_x]
    repeat unfold evalOp
    simp [Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      WithBot.realAdd, WithBot.realMul, WithBot.realSub, WithBot.realDiv, _h_mv]
    rfl
  · simp [exec, twopassWelfordKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.natToReal, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        welfordVarSpec, Triton.TiledReduction.welfordVar,
        Triton.TiledReduction.welfordSumSq,
        Triton.TiledReduction.welfordMean, Triton.TiledReduction.tileSum]
    unfold InputLoadedAt at _h_x
    simp_rw [_h_x]
    repeat unfold evalOp
    simp [Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      WithBot.realAdd, WithBot.realMul, WithBot.realSub, WithBot.realDiv, _h_mv, pow_two]
    rfl

/-- Loop invariant for `onlineWelfordKernel`: after `k` body iterations,
    register `M` holds `welfordMean xs k`, register `S` holds `welfordS xs k`,
    register `pid` holds the original program id, and the input region is
    unchanged. -/
def P_welford {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs .real [] "M" = some (Tile.scalar (welfordMean xs k))
  ∧ s.regs .real [] "S" = some (Tile.scalar (welfordS xs k))
  ∧ s.regs .nat [] "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputLoadedAt s xReg N xs

theorem online_welford_step
    {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName) (origPid i : Nat)
    (s : BlockState) (hi : i < N)
    (hP : P_welford xs xReg origPid i s) :
    ∃ s',
      stepStmts (onlineWelfordLoopBody xReg N)
        (s.setReg "i" .nat [] (Tile.scalar i)) = some s' ∧
      P_welford xs xReg origPid (i + 1) s' := by
  rcases hP with ⟨hM, hS, hpidReg, hpid, hX⟩
  let xi : ℝ := s.readMem xReg (origPid * N + i)
  have hxi : xi = xs ⟨i, hi⟩ := by
    have hx := hX ⟨i, hi⟩
    rw [hpid] at hx
    exact hx
  let m : ℝ := welfordMean xs i
  let ssum : ℝ := welfordS xs i
  let delta : ℝ := xi - m
  let m' : ℝ := m + delta / ((i : ℝ) + 1)
  let delta2 : ℝ := xi - m'
  let ssum' : ℝ := ssum + delta * delta2
  let s' :=
    (((((s.setReg "i" .nat [] (Tile.scalar i)).setReg
      "xi" .real [] (Tile.scalar xi)).setReg
      "delta" .real [] (Tile.scalar delta)).setReg
      "M" .real [] (Tile.scalar m')).setReg
      "delta2" .real [] (Tile.scalar delta2)).setReg
      "S" .real [] (Tile.scalar ssum')
  refine ⟨s', ?_, ?_⟩
  · -- Reduce loop body via simp; remaining goal is structural equality of
    -- WithBot ℝ arithmetic terms (↑a + ↑b vs ↑(a + b) etc.) which matches via
    -- WithBot.coe_add / coe_mul in reverse, plus rfl on the outer setReg shell.
    simp [onlineWelfordLoopBody, stepStmts, stepStmt, evalOp, Tile.bop,
      Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
      NumericDType.div, BlockState.readMem, hM, hS, hpidReg,
      xi, m, ssum, delta, m', delta2, ssum', s',
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
      BlockState.setReg]
    rfl
  · simp [P_welford, s', InputLoadedAt, welfordMean, welfordS, hi, xi, m,
      ssum, delta, m', delta2, ssum', hpidReg, hpid, hxi]
    intro j
    have hx := hX j
    rw [hpid] at hx
    exact hx

theorem online_welford_correct
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_mv : meanReg ≠ varReg) :
    let final := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    final.bind (fun s' => some (s'.readMem meanReg 0))
        = some (welfordMeanSpec xs)
    ∧ final.bind (fun s' => some (s'.readMem varReg 0))
        = some (welfordVarSpec xs) := by
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
      "M" .real [] (Tile.scalar 0)).setReg
      "S" .real [] (Tile.scalar 0)
  have h_init : P_welford xs xReg s.pid 0 s0 := by
    simp [P_welford, s0, welfordMean, welfordS]
    exact h_x
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    forLoop_inv
      (idx := "i") (n := blockSize)
      (body := onlineWelfordLoopBody xReg blockSize)
      (P := P_welford xs xReg s.pid)
      (s_init := s0)
      h_init
      (fun i st hi hP => online_welford_step xs xReg s.pid i st hi hP)
  rcases hPloop with ⟨hM, hS, _hpidReg, _hpid, _hX⟩
  obtain ⟨hMeanEq, hSEq⟩ := welford_eq_two_pass hN xs
  have hLoopAux :
      stepForLoopAux "i" 0 blockSize (onlineWelfordLoopBody xReg blockSize) s0 =
        some sLoop := by
    simpa [stepForLoopAux.forLoop_unfold] using hLoop
  have hLoopAuxExpanded :
      stepForLoopAux "i" 0 blockSize
        [Stmt.assign .real [] "xi"
            (Op.load .real (MemAccess.region xReg
              (Op.add .nat .nil
                (Op.mul .nat .nil (Op.ref .nat [] "pid")
                  (Op.constNat blockSize))
                (Op.ref .nat [] "i"))) MaskOpt.none),
          Stmt.assign .real [] "delta"
            (Op.sub .real .nil (Op.ref .real [] "xi")
              (Op.ref .real [] "M")),
          Stmt.assign .real [] "M"
            (Op.add .real .nil (Op.ref .real [] "M")
              (Op.div .real .nil (Op.ref .real [] "delta")
                (Op.add .real .nil (Op.ref .nat [] "i").natToReal
                  (Op.const 1)))),
          Stmt.assign .real [] "delta2"
            (Op.sub .real .nil (Op.ref .real [] "xi")
              (Op.ref .real [] "M")),
          Stmt.assign .real [] "S"
            (Op.add .real .nil (Op.ref .real [] "S")
              (Op.mul .real .nil (Op.ref .real [] "delta")
                (Op.ref .real [] "delta2")))]
        s0 = some sLoop := by
    simpa [onlineWelfordLoopBody] using hLoopAux
  have hExec :
      exec (onlineWelfordKernel xReg meanReg varReg blockSize) s =
        some ((sLoop.writeMem meanReg 0 (welfordMean xs blockSize)).writeMem
          varReg 0 (welfordS xs blockSize / blockSize)) := by
    -- Walk through pre-loop assigns, forLoop, post-loop stores explicitly.
    have hpid : stepStmt (.assign .nat [] "pid" (.programId 0)) s
                  = some (s.setReg "pid" .nat [] (Tile.scalar s.pid)) := by
      simp [stepStmt, evalOp]
    have hM0 : stepStmt (.assign .real [] "M" (.const 0))
                  (s.setReg "pid" .nat [] (Tile.scalar s.pid))
                = some ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                    "M" .real [] (Tile.scalar 0)) := by
      simp [stepStmt, evalOp]
      rfl
    have hS0 : stepStmt (.assign .real [] "S" (.const 0))
                  ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                    "M" .real [] (Tile.scalar 0))
                = some s0 := by
      simp [stepStmt, evalOp, s0]
      rfl
    have hMeanStore : stepStmt
        (.store .real [] (MemAccess.region meanReg (.constNat 0)) (.ref .real [] "M") MaskOpt.none) sLoop
        = some (sLoop.writeMem meanReg 0 (welfordMean xs blockSize)) := by
      simp [stepStmt, evalOp, hM]
    set sMean : BlockState := sLoop.writeMem meanReg 0 (welfordMean xs blockSize)
    have hSat_S : sMean.regs .real [] "S"
                  = some (Tile.scalar (welfordS xs blockSize)) := by
      show sLoop.regs .real [] "S" = _
      exact hS
    have hVarStore : stepStmt
        (.store .real [] (MemAccess.region varReg (.constNat 0))
            (.div .real .nil (.ref .real [] "S")
              (.natToReal (.constNat blockSize))) MaskOpt.none) sMean
        = some (sMean.writeMem varReg 0 (welfordS xs blockSize / blockSize)) := by
      simp [stepStmt, evalOp, hSat_S, Tile.bop, NumericDType.div, Tile.natToReal,
            WithBot.realDiv]
    -- Chain: 3 assigns → s0; forLoop → sLoop; 2 stores → sMean.writeMem ...
    show stepStmts (onlineWelfordKernel xReg meanReg varReg blockSize).body s = _
    show stepStmts
        [ .assign .nat [] "pid" (.programId 0)
        , .assign .real [] "M" (.const 0)
        , .assign .real [] "S" (.const 0)
        , .forLoop "i" blockSize (onlineWelfordLoopBody xReg blockSize)
        , .store .real [] (MemAccess.region meanReg (.constNat 0)) (.ref .real [] "M") MaskOpt.none
        , .store .real [] (MemAccess.region varReg (.constNat 0))
            (.div .real .nil (.ref .real [] "S")
              (.natToReal (.constNat blockSize))) MaskOpt.none
        ] s = _
    rw [stepStmts.cons_some hpid]
    rw [stepStmts.cons_some hM0]
    rw [stepStmts.cons_some hS0]
    rw [stepStmts.cons_some hLoop]
    rw [stepStmts.cons_some hMeanStore]
    rw [stepStmts.cons_some hVarStore]
    exact stepStmts.nil
  constructor
  · rw [hExec]
    simp [BlockState.readMem, BlockState.writeMem, h_mv, welfordMeanSpec,
      Triton.TiledReduction.welfordMean, Triton.TiledReduction.tileSum,
      twoPassMean, hMeanEq]
  · rw [hExec]
    simp [BlockState.readMem, BlockState.writeMem, welfordVarSpec,
      Triton.TiledReduction.welfordVar, Triton.TiledReduction.welfordSumSq,
      Triton.TiledReduction.welfordMean, Triton.TiledReduction.tileSum,
      twoPassS, twoPassMean, hSEq]

theorem welford_kernels_refinement
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_mv : meanReg ≠ varReg) :
    let final_2p := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    let final_on := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    final_2p.bind (fun s' => some (s'.readMem meanReg 0))
        = final_on.bind (fun s' => some (s'.readMem meanReg 0))
    ∧ final_2p.bind (fun s' => some (s'.readMem varReg 0))
        = final_on.bind (fun s' => some (s'.readMem varReg 0)) := by
  obtain ⟨h_2p_mean, h_2p_var⟩ :=
    twopass_welford_correct xReg meanReg varReg blockSize hN s xs h_x h_mv
  obtain ⟨h_on_mean, h_on_var⟩ :=
    online_welford_correct xReg meanReg varReg blockSize hN s xs h_x h_mv
  exact ⟨h_2p_mean.trans h_on_mean.symm, h_2p_var.trans h_on_var.symm⟩

/-- View-level surface for `welford_kernels_refinement`. -/
theorem welford_kernels_refinement_exec_view
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_mv : meanReg ≠ varReg) :
    let final_2p := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    let final_on := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    TensorView.observe final_2p (scalarCellView meanReg 0) PUnit.unit
        = TensorView.observe final_on (scalarCellView meanReg 0) PUnit.unit
    ∧ TensorView.observe final_2p (scalarCellView varReg 0) PUnit.unit
        = TensorView.observe final_on (scalarCellView varReg 0) PUnit.unit := by
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  obtain ⟨hm, hv⟩ :=
    welford_kernels_refinement xReg meanReg varReg blockSize hN s xs hx h_mv
  constructor
  · cases h2p : exec (twopassWelfordKernel xReg meanReg varReg blockSize) s <;>
      cases hon : exec (onlineWelfordKernel xReg meanReg varReg blockSize) s <;>
      simp [TensorView.observe, observeTileAt, scalarCellView,
            TensorView.offset, Offset.strided, BlockState.readMem,
            h2p, hon] at hm ⊢
    exact hm
  · cases h2p : exec (twopassWelfordKernel xReg meanReg varReg blockSize) s <;>
      cases hon : exec (onlineWelfordKernel xReg meanReg varReg blockSize) s <;>
      simp [TensorView.observe, observeTileAt, scalarCellView,
            TensorView.offset, Offset.strided, BlockState.readMem,
            h2p, hon] at hv ⊢
    exact hv

end VeriTile.Examples
