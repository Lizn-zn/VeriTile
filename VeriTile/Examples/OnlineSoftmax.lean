/-
VeriTile.Examples.OnlineSoftmax

Online softmax recurrence and typed Triton kernel skeleton.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton

def batchSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel :=
  stableSoftmaxKernel xReg yReg N

def onlineSoftmaxKernel (xReg _yReg : RegionName) (N : Nat) : Kernel := triton {
  pid := tl.program_id(0)
  m   := -inf
  l   := 0
  tl.for i in $(N) {
    xi    := tl.load($(xReg) + (pid * $(N) + i))
    m_new := tl.max(m, xi)
    l     := tl.exp(m - m_new) * l + tl.exp(xi - m_new)
    m     := m_new
  }
}

private def onlineSoftmaxLoopBody (xReg : RegionName) (N : Nat) : List Stmt :=
  [Stmt.assign .real [] "xi"
      (Op.load .real (MemAccess.region xReg
        (Op.add .nat .nil
          (Op.mul .nat .nil (Op.ref .nat [] "pid")
            (Op.constNat N))
          (Op.ref .nat [] "i"))) MaskOpt.none),
    Stmt.assign .real [] "m_new"
      (Op.max2 .nil (Op.ref .real [] "m") (Op.ref .real [] "xi")),
    Stmt.assign .real [] "l"
      (Op.add .real .nil
        (Op.mul .real .nil
          (Op.exp (Op.sub .real .nil
            (Op.ref .real [] "m") (Op.ref .real [] "m_new")))
          (Op.ref .real [] "l"))
        (Op.exp (Op.sub .real .nil
          (Op.ref .real [] "xi") (Op.ref .real [] "m_new")))),
    Stmt.assign .real [] "m" (Op.ref .real [] "m_new")]

/-! ## Math model — `WithBot ℝ`-valued

The seed `M_0 = ⊥` is genuinely below every real number, so the first
iteration's `max ⊥ (xs 0) = some (xs 0)` reproduces the batch base case
*without* a magnitude precondition on the input data. This is the entire
point of the `WithBot` refactor (issue #21): no more `h_lo`. -/

noncomputable def onlineSoftmaxM {N : Nat} (xs : Fin N → ℝ) : Nat → WithBot ℝ
  | 0     => ⊥
  | k + 1 =>
      if h : k < N then max (onlineSoftmaxM xs k) (((xs ⟨k, h⟩ : ℝ) : WithBot ℝ))
      else onlineSoftmaxM xs k

/-- L-recurrence in `WithBot ℝ`. After `k ≥ 1` iterations the result is
`some (∑ exp(xs i - M_k))`; at `k = 0` the seed is `some 0`. -/
noncomputable def onlineSoftmaxL {N : Nat} (xs : Fin N → ℝ) : Nat → WithBot ℝ
  | 0     => ((0 : ℝ) : WithBot ℝ)
  | k + 1 =>
      if h : k < N then
        let m_old := onlineSoftmaxM xs k
        let m_new := onlineSoftmaxM xs (k + 1)
        WithBot.realAdd
          (WithBot.realMul (WithBot.realExp (WithBot.realSub m_old m_new))
                           (onlineSoftmaxL xs k))
          (WithBot.realExp (WithBot.realSub (((xs ⟨k, h⟩ : ℝ) : WithBot ℝ)) m_new))
      else onlineSoftmaxL xs k

noncomputable def batchSoftmaxM {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  tileMax hN xs

noncomputable def batchSoftmaxL {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  ∑ i, Real.exp (xs i - batchSoftmaxM hN xs)

/-! ### Prefix lemmas — no `h_lo` precondition needed -/

/-- `M_k = some (max over first k inputs)` for `k ≥ 1`. The k = 0 case is
`⊥`, which is why we start the inductive case from `k = 1`. -/
private theorem onlineSoftmaxM_succ_eq_sup' {N : Nat} (xs : Fin N → ℝ) :
    ∀ k : Nat, ∀ (hk : k + 1 ≤ N),
      onlineSoftmaxM xs (k + 1) =
        ((((Finset.univ : Finset (Fin (k + 1))).sup' Finset.univ_nonempty
            (fun i => xs (castFin hk i))) : ℝ) : WithBot ℝ) := by
  intro k
  induction k with
  | zero =>
      intro hk
      have h0 : 0 < N := hk
      -- M_1 = max ⊥ ↑(xs 0) = ↑(xs 0) = ↑(sup' over Fin 1)
      show onlineSoftmaxM xs 1 = _
      have : onlineSoftmaxM xs 1 =
          max (onlineSoftmaxM xs 0) (((xs ⟨0, h0⟩ : ℝ) : WithBot ℝ)) := by
        show (if h : 0 < N then _ else _) = _
        simp [h0]
      rw [this]
      show max (⊥ : WithBot ℝ) _ = _
      simp [bot_le]
      rfl
  | succ j ih =>
      intro hk
      have hjN : j + 1 ≤ N := Nat.le_of_succ_le hk
      have hjlt : j + 1 < N := hk
      have ihx := ih hjN
      show onlineSoftmaxM xs (j + 1 + 1) = _
      have step : onlineSoftmaxM xs (j + 1 + 1)
          = max (onlineSoftmaxM xs (j + 1))
                (((xs ⟨j + 1, hjlt⟩ : ℝ) : WithBot ℝ)) := by
        show (if h : j + 1 < N then _ else _) = _
        simp [hjlt]
      rw [step, ihx]
      -- max ↑a ↑b = ↑(max a b)
      rw [show ∀ a b : ℝ, max ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ) =
            (((max a b : ℝ)) : WithBot ℝ) from fun _ _ => rfl]
      congr 1
      rw [Finset.sup'_congr Finset.univ_nonempty (Fin.univ_castSuccEmb (j+1))
            (fun _ _ => rfl)]
      rw [Finset.sup'_cons (H := by simp)]
      rw [Finset.sup'_map]
      simp only [Function.comp, Fin.castSuccEmb_apply, max_comm]
      rfl

private theorem onlineSoftmaxM_eq_tileMax
    {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) :
    onlineSoftmaxM xs N = (((tileMax hN xs : ℝ)) : WithBot ℝ) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  rw [onlineSoftmaxM_succ_eq_sup' xs n (le_refl _)]
  rfl

/-- For `k+1 ≤ N`, the running L equals the prefix-sum form (in `WithBot ℝ`,
which after `M` is no longer ⊥ collapses to `↑(real sum)`). -/
private theorem onlineSoftmaxL_succ_eq_sum {N : Nat} (xs : Fin N → ℝ) :
    ∀ k : Nat, ∀ (hk : k + 1 ≤ N),
      onlineSoftmaxL xs (k + 1) =
        (((∑ i : Fin (k + 1), Real.exp (xs (castFin hk i) -
            (Finset.univ : Finset (Fin (k + 1))).sup' Finset.univ_nonempty
              (fun i => xs (castFin hk i)))) : ℝ) : WithBot ℝ) := by
  intro k
  induction k with
  | zero =>
      intro hk
      have h0 : 0 < N := hk
      have hL : onlineSoftmaxL xs 1 = WithBot.realAdd
          (WithBot.realMul
            (WithBot.realExp (WithBot.realSub
              (onlineSoftmaxM xs 0) (onlineSoftmaxM xs 1)))
            (onlineSoftmaxL xs 0))
          (WithBot.realExp (WithBot.realSub
            (((xs ⟨0, h0⟩ : ℝ) : WithBot ℝ)) (onlineSoftmaxM xs 1))) := by
        show (if h : 0 < N then _ else _) = _
        simp [h0]
      have hM0 : onlineSoftmaxM xs 0 = (⊥ : WithBot ℝ) := rfl
      have hL0 : onlineSoftmaxL xs 0 = ((0 : ℝ) : WithBot ℝ) := rfl
      have hM1 := onlineSoftmaxM_succ_eq_sup' xs 0 hk
      rw [hL, hM0, hL0, hM1]
      simp only [WithBot.realSub_bot_left, WithBot.realExp_bot,
        WithBot.realMul_coe_zero, WithBot.realSub_coe_coe,
        WithBot.realExp_coe, WithBot.realAdd_coe_coe]
      congr 1
      -- Goal: 0 + exp(xs ⟨0, h0⟩ - sup'_Fin1 _) = ∑_{i ∈ Fin 1} exp(xs (castFin hk i) - sup')
      -- Both sides use the same sup' and reduce to `1 = exp 0`.
      have sup_eq : (Finset.univ : Finset (Fin 1)).sup' Finset.univ_nonempty
            (fun i => xs (castFin hk i)) = xs ⟨0, h0⟩ := by
        show xs (castFin hk _) = xs ⟨0, h0⟩
        rfl
      rw [sup_eq]
      rw [Fin.sum_univ_one]
      show 0 + Real.exp (xs ⟨0, h0⟩ - xs ⟨0, h0⟩) = Real.exp (xs ⟨0, h0⟩ - xs ⟨0, h0⟩)
      rw [zero_add]
  | succ j ih =>
      intro hk
      have hj : j + 1 ≤ N := Nat.le_of_succ_le hk
      have hjlt : j + 1 < N := hk
      have ih' := ih hj
      have hM := onlineSoftmaxM_succ_eq_sup' xs j hj
      have hM_next := onlineSoftmaxM_succ_eq_sup' xs (j + 1) hk
      have hL : onlineSoftmaxL xs (j + 1 + 1) = WithBot.realAdd
          (WithBot.realMul
            (WithBot.realExp (WithBot.realSub
              (onlineSoftmaxM xs (j + 1)) (onlineSoftmaxM xs (j + 1 + 1))))
            (onlineSoftmaxL xs (j + 1)))
          (WithBot.realExp (WithBot.realSub
            (((xs ⟨j + 1, hjlt⟩ : ℝ) : WithBot ℝ))
            (onlineSoftmaxM xs (j + 1 + 1)))) := by
        show (if h : j + 1 < N then _ else _) = _
        simp [hjlt]
      rw [hL, hM, hM_next, ih']
      simp only [WithBot.realSub_coe_coe, WithBot.realExp_coe,
        WithBot.realMul_coe_coe, WithBot.realAdd_coe_coe]
      congr 1
      -- Expand only the RHS outer sum
      conv_rhs => rw [Fin.sum_univ_castSucc]
      rw [Finset.mul_sum]
      have hxs_castSucc : ∀ i : Fin (j + 1),
          xs (castFin hk i.castSucc) = xs (castFin hj i) := fun _ => rfl
      have hxs_last : xs (castFin hk (Fin.last (j + 1))) = xs ⟨j + 1, hjlt⟩ := rfl
      simp_rw [hxs_castSucc, hxs_last]
      congr 1
      apply Finset.sum_congr rfl
      intros i _
      rw [← Real.exp_add]
      ring_nf

private theorem onlineSoftmaxL_eq_batch
    {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) :
    onlineSoftmaxL xs N = (((batchSoftmaxL hN xs : ℝ)) : WithBot ℝ) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  rw [onlineSoftmaxL_succ_eq_sum xs n (le_refl _)]
  rfl

private def P_online_softmax {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  -- `onlineSoftmaxM/L xs k : WithBot ℝ` directly populates the tile.
  s.regs .real [] "m" = some (Tile.scalar (onlineSoftmaxM xs k))
  ∧ s.regs .real [] "l" = some (Tile.scalar (onlineSoftmaxL xs k))
  ∧ s.regs .nat [] "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputLoadedAt s xReg N xs

private theorem online_softmax_step
    {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName) (origPid i : Nat)
    (s : BlockState) (hi : i < N)
    (hP : P_online_softmax xs xReg origPid i s) :
    ∃ s',
      stepStmts (onlineSoftmaxLoopBody xReg N)
        (s.setReg "i" .nat [] (Tile.scalar i)) = some s' ∧
      P_online_softmax xs xReg origPid (i + 1) s' := by
  rcases hP with ⟨hm, hl, hpidReg, hpid, hX⟩
  let xi : ℝ := s.readMem xReg (origPid * N + i)
  have hxi : xi = xs ⟨i, hi⟩ := by
    have hx := hX ⟨i, hi⟩
    rw [hpid] at hx
    exact hx
  -- Math model values now live in `WithBot ℝ`.
  let mOld : WithBot ℝ := onlineSoftmaxM xs i
  let lOld : WithBot ℝ := onlineSoftmaxL xs i
  let mNew : WithBot ℝ := max mOld ((xi : ℝ) : WithBot ℝ)
  let lNew : WithBot ℝ :=
    WithBot.realAdd
      (WithBot.realMul (WithBot.realExp (WithBot.realSub mOld mNew)) lOld)
      (WithBot.realExp (WithBot.realSub ((xi : ℝ) : WithBot ℝ) mNew))
  let s' :=
    (((s.setReg "i" .nat [] (Tile.scalar i)).setReg
      "xi" .real [] (Tile.scalar ((xi : ℝ) : WithBot ℝ))).setReg
      "m_new" .real [] (Tile.scalar mNew)).setReg
      "l" .real [] (Tile.scalar lNew)
  let s'' := s'.setReg "m" .real [] (Tile.scalar mNew)
  refine ⟨s'', ?_, ?_⟩
  · simp [onlineSoftmaxLoopBody, stepStmts, stepStmt, evalOp, Tile.bop,
      Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
      BlockState.readMem, hm, hl, hpidReg, xi, mOld, lOld, mNew, lNew,
      s', s'', BlockState.setReg]
    rfl
  · -- P_online_softmax holds at i+1 in s''.
    have h_recM : onlineSoftmaxM xs (i + 1) = mNew := by
      show (if h : i < N then max (onlineSoftmaxM xs i) _ else _) = mNew
      simp [hi]
      show max mOld _ = mNew
      simp [mNew, hxi]
    have h_recL : onlineSoftmaxL xs (i + 1) = lNew := by
      show (if h : i < N then _ else _) = lNew
      simp [hi]
      -- Substitute the recurrence value of M_{i+1}
      rw [h_recM]
      simp [mOld, mNew, lNew, lOld, hxi]
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · show s''.regs .real [] "m" = some (Tile.scalar (onlineSoftmaxM xs (i+1)))
      rw [h_recM]
      simp [s'', s', BlockState.setReg]
    · show s''.regs .real [] "l" = some (Tile.scalar (onlineSoftmaxL xs (i+1)))
      rw [h_recL]
      simp [s'', s', BlockState.setReg]
    · simp [s'', s', BlockState.setReg, hpidReg]
    · simp [s'', s', BlockState.setReg, hpid]
    · intro j
      simpa [s'', s', BlockState.setReg] using hX j

/-- **Math identity (paper centerpiece, h_lo-free version).** Online softmax's
streaming `(M, L)` recurrence equals the batch form's `(tileMax, ∑ exp)` —
without any range precondition on the input. The `WithBot ℝ` math model uses
`⊥` as the seed of `M`, which is genuinely below every real, so the first
iteration's `max ⊥ (xs 0) = some (xs 0)` reproduces the batch base case. -/
theorem online_softmax_recurrence_eq_batch
    {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) :
    onlineSoftmaxM xs N = (((batchSoftmaxM hN xs : ℝ)) : WithBot ℝ) ∧
    onlineSoftmaxL xs N = (((batchSoftmaxL hN xs : ℝ)) : WithBot ℝ) := by
  refine ⟨onlineSoftmaxM_eq_tileMax hN xs, onlineSoftmaxL_eq_batch hN xs⟩

/-- **Operational correctness, h_lo-free.** The kernel's `m` register at
termination equals `onlineSoftmaxM xs N : WithBot ℝ`, which by the math
identity equals `↑(tileMax xs)`. Same for `l`. No range precondition on
input data. -/
theorem online_softmax_correct
    (xReg yReg : RegionName) (N : Nat) (_hN : 0 < N)
    (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    let final := exec (onlineSoftmaxKernel xReg yReg N) s
    final.bind (fun s' => (s'.regs .real [] "m").map (fun t => t.data PUnit.unit))
        = some (onlineSoftmaxM xs N)
    ∧ final.bind (fun s' => (s'.regs .real [] "l").map (fun t => t.data PUnit.unit))
        = some (onlineSoftmaxL xs N) := by
  let s0 :=
    ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
      "m" .real [] (Tile.scalar (⊥ : WithBot ℝ))).setReg
      "l" .real [] (Tile.scalar (((0 : ℝ) : WithBot ℝ)))
  have h_init : P_online_softmax xs xReg s.pid 0 s0 := by
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · show s0.regs .real [] "m" = some (Tile.scalar (onlineSoftmaxM xs 0))
      simp [s0, BlockState.setReg]
      rfl
    · show s0.regs .real [] "l" = some (Tile.scalar (onlineSoftmaxL xs 0))
      simp [s0, BlockState.setReg]
      rfl
    · simp [s0, BlockState.setReg]
    · simp [s0, BlockState.setReg]
    · intro j
      have := _h_x j
      simpa [s0, BlockState.setReg] using this
  obtain ⟨sLoop, hLoop, hPloop⟩ :=
    forLoop_inv
      (idx := "i") (n := N)
      (body := onlineSoftmaxLoopBody xReg N)
      (P := P_online_softmax xs xReg s.pid)
      (s_init := s0)
      h_init
      (fun i st hi hP => online_softmax_step xs xReg s.pid i st hi hP)
  rcases hPloop with ⟨hm, hl, _hpidReg, _hpid, _hX⟩
  have hLoopAux :
      stepForLoopAux "i" 0 N (onlineSoftmaxLoopBody xReg N) s0 =
        some sLoop := by
    simpa [stepForLoopAux.forLoop_unfold] using hLoop
  have hLoopAuxExpanded :
      stepForLoopAux "i" 0 N
        [Stmt.assign .real [] "xi"
            (Op.load .real (MemAccess.region xReg
              (Op.add .nat .nil
                (Op.mul .nat .nil (Op.ref .nat [] "pid")
                  (Op.constNat N))
                (Op.ref .nat [] "i"))) MaskOpt.none),
          Stmt.assign .real [] "m_new"
            (Op.max2 .nil (Op.ref .real [] "m") (Op.ref .real [] "xi")),
          Stmt.assign .real [] "l"
            (Op.add .real .nil
              (Op.mul .real .nil
                (Op.exp (Op.sub .real .nil
                  (Op.ref .real [] "m") (Op.ref .real [] "m_new")))
                (Op.ref .real [] "l"))
              (Op.exp (Op.sub .real .nil
                (Op.ref .real [] "xi") (Op.ref .real [] "m_new")))),
          Stmt.assign .real [] "m" (Op.ref .real [] "m_new")]
        s0 = some sLoop := by
    simpa [onlineSoftmaxLoopBody] using hLoopAux
  have hExec : exec (onlineSoftmaxKernel xReg yReg N) s = some sLoop := by
    -- Walk through each pre-loop statement explicitly, then forLoop via hLoop.
    have hpid : stepStmt (.assign .nat [] "pid" (.programId 0)) s
                  = some (s.setReg "pid" .nat [] (Tile.scalar s.pid)) := by
      simp [stepStmt, evalOp]
    have hm0 : stepStmt (.assign .real [] "m" .negInf)
                  (s.setReg "pid" .nat [] (Tile.scalar s.pid))
                = some ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                    "m" .real [] (Tile.scalar (⊥ : WithBot ℝ))) := by
      simp [stepStmt, evalOp]
      rfl
    have hl0 : stepStmt (.assign .real [] "l" (.const 0))
                  ((s.setReg "pid" .nat [] (Tile.scalar s.pid)).setReg
                    "m" .real [] (Tile.scalar (⊥ : WithBot ℝ)))
                = some s0 := by
      simp [stepStmt, evalOp, s0]
      rfl
    -- Chain the pre-loop assignments and the loop statement explicitly.
    show stepStmts (onlineSoftmaxKernel xReg yReg N).body s = some sLoop
    show stepStmts
        [ .assign .nat [] "pid" (.programId 0)
        , .assign .real [] "m" .negInf
        , .assign .real [] "l" (.const 0)
        , .forLoop "i" N (onlineSoftmaxLoopBody xReg N) ] s = some sLoop
    rw [stepStmts.cons_some hpid]
    rw [stepStmts.cons_some hm0]
    rw [stepStmts.cons_some hl0]
    rw [stepStmts.cons_some hLoop]
    exact stepStmts.nil
  constructor
  · rw [hExec]
    simp [hm]
  · rw [hExec]
    simp [hl]

/-- View-level surface for `online_softmax_correct`. -/
theorem online_softmax_correct_view
    (xReg yReg : RegionName) (N : Nat) (hN : 0 < N)
    (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    let final := exec (onlineSoftmaxKernel xReg yReg N) s
    final.bind (fun s' => (s'.regs .real [] "m").map (fun t => t.data PUnit.unit))
        = some (onlineSoftmaxM xs N)
    ∧ final.bind (fun s' => (s'.regs .real [] "l").map (fun t => t.data PUnit.unit))
        = some (onlineSoftmaxL xs N) := by
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := N) (xs := xs) h_x
  exact online_softmax_correct xReg yReg N hN s xs hx

end VeriTile.Examples
