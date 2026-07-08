import VeriTile.Examples.WelfordKernels

/-!
Writes-equality refinement (`ComputeRefine.Refines` whole-memory surface) for
the two-pass vs online Welford kernel pair: from the same initial state, both
kernels perform the same writes — their final memories agree at every cell.
The kernels, specs, correctness lemmas, and the exec-level refinement live in
`VeriTile.Examples.WelfordKernels`.

Headline: `welford_kernels_refinement_view`. For the rounding-model (∀R) variant of the
writes-equality surface see `bench/examples/Swiglu.lean`.
-/

namespace VeriTile.Examples

open VeriTile.Triton

/-- A successful `assign` step only touches registers, never memory. -/
private theorem stepStmt_assign_mem {dtype : TileDType} {shape : TileShape}
    {name : RegName} {e : Op dtype shape} {t t' : BlockState}
    (h : stepStmt (Stmt.assign dtype shape name e) t = some t')
    (r : RegionName) (o : Nat) : t'.mem r o = t.mem r o := by
  simp only [stepStmt] at h
  cases hv : evalOp e t with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      injection h with h
      subst h
      rfl

/-- A successful run of a list of `assign` statements leaves memory unchanged. -/
private theorem stepStmts_assigns_mem :
    ∀ (l : List Stmt),
      (∀ st ∈ l, ∃ (dtype : TileDType) (shape : TileShape) (name : RegName)
        (e : Op dtype shape), st = Stmt.assign dtype shape name e) →
      ∀ t t' : BlockState, stepStmts l t = some t' →
        ∀ (r : RegionName) (o : Nat), t'.mem r o = t.mem r o := by
  intro l
  induction l with
  | nil =>
      intro _ t t' h r o
      simp only [stepStmts.nil, Option.some.injEq] at h
      subst h
      rfl
  | cons hd tl ih =>
      intro hall t t' h r o
      unfold stepStmts at h
      cases hhd : stepStmt hd t with
      | none => simp [hhd] at h
      | some tmid =>
          simp only [hhd] at h
          obtain ⟨dt, sh, nm, e, heq⟩ := hall hd (by simp)
          subst heq
          rw [ih (fun st hst => hall st (List.mem_cons_of_mem _ hst)) tmid t' h r o,
            stepStmt_assign_mem hhd r o]

/-- Every statement of the online Welford loop body is an `assign`. -/
private theorem onlineWelfordLoopBody_assigns (xReg : RegionName) (blockSize : Nat) :
    ∀ st ∈ onlineWelfordLoopBody xReg blockSize,
      ∃ (dtype : TileDType) (shape : TileShape) (name : RegName)
        (e : Op dtype shape), st = Stmt.assign dtype shape name e := by
  intro st hst
  simp only [onlineWelfordLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩

/-- Explicit final-state characterization of the online kernel's execution:
the loop only updates registers, then the two scalar stores land the Welford
running mean and variance. -/
private theorem online_exec_writes
    (xReg meanReg varReg : RegionName) (blockSize : Nat)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs) :
    ∃ sLoop : BlockState,
      (∀ r o, sLoop.mem r o = s.mem r o) ∧
      exec (onlineWelfordKernel xReg meanReg varReg blockSize) s =
        some ((sLoop.writeMem meanReg 0 (welfordMean xs blockSize)).writeMem
          varReg 0 (welfordS xs blockSize / blockSize)) := by
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
      "M" .real [] (Tile.scalar 0)).setReg
      "S" .real [] (Tile.scalar 0)
  have h_init : P_welford xs xReg s.pid 0 s0 ∧ ∀ r o, s0.mem r o = s.mem r o := by
    refine ⟨?_, fun r o => rfl⟩
    simp [P_welford, s0, welfordMean, welfordS]
    exact h_x
  obtain ⟨sLoop, hLoop, hPloop, hMemLoop⟩ :=
    forLoop_inv
      (idx := "i") (n := blockSize)
      (body := onlineWelfordLoopBody xReg blockSize)
      (P := fun k st => P_welford xs xReg s.pid k st ∧ ∀ r o, st.mem r o = s.mem r o)
      (s_init := s0)
      h_init
      (fun i st hi hP => by
        obtain ⟨hPw, hMem⟩ := hP
        obtain ⟨st', hstep, hPw'⟩ := online_welford_step xs xReg s.pid i st hi hPw
        refine ⟨st', hstep, hPw', fun r o => ?_⟩
        rw [stepStmts_assigns_mem _ (onlineWelfordLoopBody_assigns xReg blockSize)
          _ _ hstep r o]
        exact hMem r o)
  rcases hPloop with ⟨hM, hS, _hpidReg, _hpid, _hX⟩
  refine ⟨sLoop, hMemLoop, ?_⟩
  -- Walk through pre-loop assigns, forLoop, post-loop stores explicitly,
  -- mirroring `online_welford_correct`.
  have hpid : stepStmt (.assign .nat [] "pid" (.programId 0)) s
                = some (s.setReg "pid" .nat [] (Tile.scalar s.pid)) := by
    simp [stepStmt]
  have hM0 : stepStmt (.assign .real [] "M" (.const 0))
                (s.setReg "pid" .nat [] (Tile.scalar s.pid))
              = some ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                  "M" .real [] (Tile.scalar 0)) := by
    simp [stepStmt]
    rfl
  have hS0 : stepStmt (.assign .real [] "S" (.const 0))
                ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                  "M" .real [] (Tile.scalar 0))
              = some s0 := by
    simp [stepStmt, s0]
    rfl
  have hMeanStore : stepStmt
      (.store .real [] (MemAccess.region meanReg (.constNat 0)) (.ref .real [] "M") MaskOpt.none) sLoop
      = some (sLoop.writeMem meanReg 0 (welfordMean xs blockSize)) := by
    simp [stepStmt, hM]
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
    simp [stepStmt, hSat_S, Tile.bop, NumericDType.div, Tile.natToReal,
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

/-- Cell-level memory characterization of the online kernel's final state:
the var store (last) wins at `(varReg, 0)`, the mean store at `(meanReg, 0)`,
and every other cell is untouched. -/
private theorem online_exec_mem
    (xReg meanReg varReg : RegionName) (blockSize : Nat)
    (s rhs' : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoadedAt s xReg blockSize xs)
    (h_mv : meanReg ≠ varReg)
    (hR : exec (onlineWelfordKernel xReg meanReg varReg blockSize) s = some rhs')
    (r : RegionName) (o : Nat) :
    rhs'.mem r o =
      if r = varReg ∧ o = 0 then MemCell.real (rhs'.readMem varReg 0)
      else if r = meanReg ∧ o = 0 then MemCell.real (rhs'.readMem meanReg 0)
      else s.mem r o := by
  obtain ⟨sLoop, hMemLoop, hExec⟩ :=
    online_exec_writes xReg meanReg varReg blockSize s xs h_x
  rw [hExec, Option.some.injEq] at hR
  subst hR
  by_cases hvar : r = varReg ∧ o = 0
  · obtain ⟨rfl, rfl⟩ := hvar
    simp [BlockState.writeMem_mem]
  · by_cases hmean : r = meanReg ∧ o = 0
    · obtain ⟨rfl, rfl⟩ := hmean
      simp [BlockState.writeMem_mem, h_mv]
    · simp [BlockState.writeMem_mem, hvar, hmean, hMemLoop r o]

/-- Cell-level memory characterization of the two-pass kernel's final state. -/
private theorem twopass_exec_mem
    (xReg meanReg varReg : RegionName) (blockSize : Nat)
    (s lhs' : BlockState)
    (h_mv : meanReg ≠ varReg)
    (hL : exec (twopassWelfordKernel xReg meanReg varReg blockSize) s = some lhs')
    (r : RegionName) (o : Nat) :
    lhs'.mem r o =
      if r = varReg ∧ o = 0 then MemCell.real (lhs'.readMem varReg 0)
      else if r = meanReg ∧ o = 0 then MemCell.real (lhs'.readMem meanReg 0)
      else s.mem r o := by
  simp [exec, twopassWelfordKernel, stepStmts, stepStmt,
    Tile.bop, Tile.natToReal, NumericDType.add,
    NumericDType.mul, NumericDType.sub, NumericDType.div] at hL
  repeat unfold evalOp at hL
  simp [Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    NumericDType.mul, WithBot.realMul] at hL
  subst hL
  by_cases hvar : r = varReg ∧ o = 0
  · obtain ⟨rfl, rfl⟩ := hvar
    simp [BlockState.writeMem_mem]
  · by_cases hmean : r = meanReg ∧ o = 0
    · obtain ⟨rfl, rfl⟩ := hmean
      simp [BlockState.writeMem_mem, h_mv]
    · simp [BlockState.writeMem_mem, hvar, hmean]

/-- Compute-facing writes-equality refinement for the Welford kernel pair:
from the same initial state, `twopassWelfordKernel` and `onlineWelfordKernel`
perform the same writes — their final memories agree at every cell (no
scratch regions excluded). -/
theorem welford_kernels_refinement_view
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_mv : meanReg ≠ varReg) :
    ComputeRefine.Refines
      (twopassWelfordKernel xReg meanReg varReg blockSize)
      (onlineWelfordKernel xReg meanReg varReg blockSize)
      s [] := by
  apply ComputeKernel.computeRefine_of_toAlgKernel rfl rfl
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro r _hr o
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := blockSize) (xs := xs) h_x
  obtain ⟨hm, hv⟩ :=
    welford_kernels_refinement xReg meanReg varReg blockSize hN s xs hx h_mv
  rw [hL, hR] at hm hv
  simp only [Option.bind_some, Option.some.injEq] at hm hv
  rw [twopass_exec_mem xReg meanReg varReg blockSize s lhs' h_mv hL r o,
    online_exec_mem xReg meanReg varReg blockSize s rhs' xs hx h_mv hR r o,
    hm, hv]

end VeriTile.Examples
