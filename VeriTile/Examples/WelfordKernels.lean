/-
VeriTile.Examples.WelfordKernels

Welford kernels over the typed Triton core.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common
import VeriTile.Examples.WelfordMath

namespace VeriTile.Examples

open VeriTile.Triton

def twopassWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
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

def onlineWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
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

def onlineWelfordLoopBody (xReg : RegionName) (blockSize : Nat) : List Stmt :=
  [Stmt.assign .real [] "xi"
      (Op.load xReg
        (Op.add .nat .nil
          (Op.mul .nat .nil (Op.ref .nat [] "pid")
            (Op.constNat blockSize))
          (Op.ref .nat [] "i"))),
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

noncomputable def welfordMeanSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  twoPassMean xs

noncomputable def welfordVarSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  twoPassS xs / N

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
        Tile.bop, Tile.reduceSum, Tile.natToReal, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        welfordMeanSpec, twoPassMean]
    unfold InputLoadedAt at _h_x
    simp_rw [_h_x]
    simp [_h_mv]
  · simp [exec, twopassWelfordKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.reduceSum, Tile.natToReal, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        welfordVarSpec, twoPassS, twoPassMean]
    unfold InputLoadedAt at _h_x
    simp_rw [_h_x]
    simp [pow_two]

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
  let xi : ℝ := s.mem xReg (origPid * N + i)
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
            (Op.load xReg
              (Op.add .nat .nil
                (Op.mul .nat .nil (Op.ref .nat [] "pid")
                  (Op.constNat blockSize))
                (Op.ref .nat [] "i"))),
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
    have stmts_cons : ∀ (st : Stmt) (rest : List Stmt) (s s' : BlockState),
        stepStmt st s = some s' →
        stepStmts (st :: rest) s = stepStmts rest s' := by
      intro st rest s s' h
      conv_lhs => unfold stepStmts
      rw [h]
    have stmts_nil : ∀ (s : BlockState), stepStmts [] s = some s := by
      intro s; conv_lhs => unfold stepStmts
    have hpid : stepStmt (.assign .nat [] "pid" .programId) s
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
        (.store meanReg [] (.constNat 0) (.ref .real [] "M")) sLoop
        = some (sLoop.writeMem meanReg 0 (welfordMean xs blockSize)) := by
      simp [stepStmt, evalOp, hM]
    set sMean : BlockState := sLoop.writeMem meanReg 0 (welfordMean xs blockSize)
    have hSat_S : sMean.regs .real [] "S"
                  = some (Tile.scalar (welfordS xs blockSize)) := by
      show sLoop.regs .real [] "S" = _
      exact hS
    have hVarStore : stepStmt
        (.store varReg [] (.constNat 0)
            (.div .real .nil (.ref .real [] "S")
              (.natToReal (.constNat blockSize)))) sMean
        = some (sMean.writeMem varReg 0 (welfordS xs blockSize / blockSize)) := by
      simp [stepStmt, evalOp, hSat_S, Tile.bop, NumericDType.div, Tile.natToReal,
            WithBot.realDiv]
    -- Chain: 3 assigns → s0; forLoop → sLoop; 2 stores → sMean.writeMem ...
    show stepStmts (onlineWelfordKernel xReg meanReg varReg blockSize).body s = _
    show stepStmts
        [ .assign .nat [] "pid" .programId
        , .assign .real [] "M" (.const 0)
        , .assign .real [] "S" (.const 0)
        , .forLoop "i" blockSize (onlineWelfordLoopBody xReg blockSize)
        , .store meanReg [] (.constNat 0) (.ref .real [] "M")
        , .store varReg [] (.constNat 0)
            (.div .real .nil (.ref .real [] "S")
              (.natToReal (.constNat blockSize)))
        ] s = _
    rw [stmts_cons _ _ _ _ hpid]
    rw [stmts_cons _ _ _ _ hM0]
    rw [stmts_cons _ _ _ _ hS0]
    rw [stmts_cons _ _ _ _ hLoop]
    rw [stmts_cons _ _ _ _ hMeanStore]
    rw [stmts_cons _ _ _ _ hVarStore]
    exact stmts_nil _
  constructor
  · rw [hExec]
    simp [BlockState.readMem, BlockState.writeMem, h_mv, welfordMeanSpec,
      hMeanEq]
  · rw [hExec]
    simp [BlockState.readMem, BlockState.writeMem, welfordVarSpec, hSEq]

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

end VeriTile.Examples
