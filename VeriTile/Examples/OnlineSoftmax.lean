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
import VeriTile.Examples.WelfordMath

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
  [Stmt.assign .real .scalar "xi"
      (Op.load xReg
        (Op.add .nat .same
          (Op.mul .nat .same (Op.ref .nat .scalar "pid")
            (Op.constNat N))
          (Op.ref .nat .scalar "i"))),
    Stmt.assign .real .scalar "m_new"
      (Op.max2 .same (Op.ref .real .scalar "m") (Op.ref .real .scalar "xi")),
    Stmt.assign .real .scalar "l"
      (Op.add .real .same
        (Op.mul .real .same
          (Op.exp (Op.sub .real .same
            (Op.ref .real .scalar "m") (Op.ref .real .scalar "m_new")))
          (Op.ref .real .scalar "l"))
        (Op.exp (Op.sub .real .same
          (Op.ref .real .scalar "xi") (Op.ref .real .scalar "m_new")))),
    Stmt.assign .real .scalar "m" (Op.ref .real .scalar "m_new")]

noncomputable def onlineSoftmaxM {N : Nat} (xs : Fin N → ℝ) : Nat → ℝ
  | 0     => -1e38
  | k + 1 =>
      if h : k < N then max (onlineSoftmaxM xs k) (xs ⟨k, h⟩)
      else onlineSoftmaxM xs k

noncomputable def onlineSoftmaxL {N : Nat} (xs : Fin N → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < N then
        let m_old := onlineSoftmaxM xs k
        let m_new := onlineSoftmaxM xs (k + 1)
        Real.exp (m_old - m_new) * onlineSoftmaxL xs k + Real.exp (xs ⟨k, h⟩ - m_new)
      else onlineSoftmaxL xs k

noncomputable def batchSoftmaxM {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  tileMax hN xs

noncomputable def batchSoftmaxL {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  ∑ i, Real.exp (xs i - batchSoftmaxM hN xs)

private theorem onlineSoftmaxM_ge
    {N : Nat} (xs : Fin N → ℝ) :
    ∀ k, k ≤ N → (-1e38 : ℝ) ≤ onlineSoftmaxM xs k := by
  intro k hk
  induction k with
  | zero =>
      simp [onlineSoftmaxM]
  | succ j ih =>
      have hj : j ≤ N := Nat.le_of_succ_le hk
      by_cases hlt : j < N
      · simp [onlineSoftmaxM, hlt]
        exact Or.inl (ih hj)
      · simp [onlineSoftmaxM, hlt]
        exact ih hj

private theorem onlineSoftmaxM_input_le_of_lt
    {N : Nat} (xs : Fin N → ℝ) :
    ∀ k, k ≤ N → ∀ i : Fin N, i.val < k → xs i ≤ onlineSoftmaxM xs k := by
  intro k hk
  induction k with
  | zero =>
      intro i hi
      exact False.elim (Nat.not_lt_zero _ hi)
  | succ j ih =>
      intro i hi
      have hjN : j ≤ N := Nat.le_of_succ_le hk
      have hjltN : j < N := hk
      by_cases hij : i.val = j
      · have hi_eq : i = ⟨j, hjltN⟩ := Fin.ext hij
        subst hi_eq
        simp [onlineSoftmaxM, hjltN]
      · have hiltj : i.val < j := by omega
        have hprev := ih hjN i hiltj
        simp [onlineSoftmaxM, hjltN]
        exact Or.inl hprev

private theorem onlineSoftmaxM_le_tileMax_succ
    {n : Nat} (xs : Fin (n + 1) → ℝ)
    (h_lo : ∀ i, (-1e38 : ℝ) ≤ xs i) :
    onlineSoftmaxM xs (n + 1) ≤ tileMax (Nat.succ_pos n) xs := by
  have h :
      ∀ k, k ≤ n + 1 → onlineSoftmaxM xs k ≤ tileMax (Nat.succ_pos n) xs := by
    intro k hk
    induction k with
    | zero =>
        have hmem : (⟨0, Nat.succ_pos n⟩ : Fin (n + 1)) ∈
            (Finset.univ : Finset (Fin (n + 1))) := by simp
        simpa [tileMax] using
          le_trans (h_lo ⟨0, Nat.succ_pos n⟩) (Finset.le_sup' xs hmem)
    | succ j ih =>
        have hj : j ≤ n + 1 := Nat.le_of_succ_le hk
        by_cases hlt : j < n + 1
        · simp [onlineSoftmaxM, hlt]
          constructor
          · exact ih hj
          · have hmem : (⟨j, hlt⟩ : Fin (n + 1)) ∈
                (Finset.univ : Finset (Fin (n + 1))) := by simp
            simpa [tileMax] using (Finset.le_sup' xs hmem)
        · simp [onlineSoftmaxM, hlt]
          exact ih hj
  exact h (n + 1) (le_refl _)

private theorem tileMax_le_onlineSoftmaxM_succ
    {n : Nat} (xs : Fin (n + 1) → ℝ) :
    tileMax (Nat.succ_pos n) xs ≤ onlineSoftmaxM xs (n + 1) := by
  simp [tileMax]
  intro i
  exact onlineSoftmaxM_input_le_of_lt xs (n + 1) (le_refl _) i i.isLt

private theorem onlineSoftmaxM_eq_tileMax
    {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ)
    (h_lo : ∀ i, (-1e38 : ℝ) ≤ xs i) :
    onlineSoftmaxM xs N = tileMax hN xs := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  exact le_antisymm
    (onlineSoftmaxM_le_tileMax_succ xs h_lo)
    (tileMax_le_onlineSoftmaxM_succ xs)

private theorem onlineSoftmaxL_eq_sum_prefix
    {N : Nat} (xs : Fin N → ℝ) :
    ∀ k, ∀ (hk : k ≤ N),
      onlineSoftmaxL xs k =
        ∑ i : Fin k, Real.exp (xs (castFin hk i) - onlineSoftmaxM xs k) := by
  intro k
  induction k with
  | zero =>
      intro _
      simp [onlineSoftmaxL]
  | succ j ih =>
      intro hk
      have hj : j ≤ N := Nat.le_of_succ_le hk
      have hlt : j < N := hk
      have ih' := ih hj
      rw [Fin.sum_univ_castSucc]
      simp [onlineSoftmaxL, onlineSoftmaxM, hlt]
      rw [ih']
      rw [Finset.mul_sum]
      apply congrArg (fun z =>
        z + Real.exp (xs ⟨j, hlt⟩ -
          max (onlineSoftmaxM xs j) (xs ⟨j, hlt⟩)))
      apply Finset.sum_congr rfl
      intro i _
      have hcast : xs (castFin hk i.castSucc) = xs (castFin hj i) := rfl
      rw [hcast]
      rw [← Real.exp_add]
      congr 1
      ring

private def P_online_softmax {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs .real .scalar "m" = some (Tile.scalar (onlineSoftmaxM xs k))
  ∧ s.regs .real .scalar "l" = some (Tile.scalar (onlineSoftmaxL xs k))
  ∧ s.regs .nat .scalar "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputLoadedAt s xReg N xs

private theorem online_softmax_step
    {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName) (origPid i : Nat)
    (s : BlockState) (hi : i < N)
    (hP : P_online_softmax xs xReg origPid i s) :
    ∃ s',
      stepStmts (onlineSoftmaxLoopBody xReg N)
        (s.setReg "i" .nat .scalar (Tile.scalar i)) = some s' ∧
      P_online_softmax xs xReg origPid (i + 1) s' := by
  rcases hP with ⟨hm, hl, hpidReg, hpid, hX⟩
  let xi : ℝ := s.mem xReg (origPid * N + i)
  have hxi : xi = xs ⟨i, hi⟩ := by
    have hx := hX ⟨i, hi⟩
    rw [hpid] at hx
    exact hx
  let mOld : ℝ := onlineSoftmaxM xs i
  let lOld : ℝ := onlineSoftmaxL xs i
  let mNew : ℝ := max mOld xi
  let lNew : ℝ := Real.exp (mOld - mNew) * lOld + Real.exp (xi - mNew)
  let s' :=
    (((s.setReg "i" .nat .scalar (Tile.scalar i)).setReg
      "xi" .real .scalar (Tile.scalar xi)).setReg
      "m_new" .real .scalar (Tile.scalar mNew)).setReg
      "l" .real .scalar (Tile.scalar lNew)
  let s'' := s'.setReg "m" .real .scalar (Tile.scalar mNew)
  refine ⟨s'', ?_, ?_⟩
  · simp [onlineSoftmaxLoopBody, stepStmts, stepStmt, evalOp, Tile.bop,
      Tile.uop, NumericDType.add, NumericDType.mul, NumericDType.sub,
      BlockState.readMem, hm, hl, hpidReg, xi, mOld, lOld, mNew, lNew,
      s', s'']
  · simp [P_online_softmax, s'', s', InputLoadedAt, onlineSoftmaxM,
      onlineSoftmaxL, hi, xi, hxi, mOld, lOld, mNew, lNew, hpidReg, hpid]
    intro j
    have hx := hX j
    rw [hpid] at hx
    exact hx

theorem online_softmax_recurrence_eq_batch
    {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ)
    (_h_lo : ∀ i, (-1e38 : ℝ) ≤ xs i) :
    onlineSoftmaxM xs N = batchSoftmaxM hN xs ∧
    onlineSoftmaxL xs N = batchSoftmaxL hN xs := by
  have hM : onlineSoftmaxM xs N = batchSoftmaxM hN xs := by
    exact onlineSoftmaxM_eq_tileMax hN xs _h_lo
  refine ⟨hM, ?_⟩
  have hL := onlineSoftmaxL_eq_sum_prefix xs N (le_refl N)
  unfold batchSoftmaxL
  rw [hL, hM]
  apply Finset.sum_congr rfl
  intro i _
  rfl

theorem online_softmax_correct
    (xReg yReg : RegionName) (N : Nat) (_hN : 0 < N)
    (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    let final := exec (onlineSoftmaxKernel xReg yReg N) s
    final.bind (fun s' => (s'.regs .real .scalar "m").map (fun t => t.data PUnit.unit))
        = some (onlineSoftmaxM xs N)
    ∧ final.bind (fun s' => (s'.regs .real .scalar "l").map (fun t => t.data PUnit.unit))
        = some (onlineSoftmaxL xs N) := by
  let s0 :=
    ((s.setReg "pid" .nat .scalar (Tile.scalar s.pid)).setReg
      "m" .real .scalar (Tile.scalar (-1e38))).setReg
      "l" .real .scalar (Tile.scalar 0)
  have h_init : P_online_softmax xs xReg s.pid 0 s0 := by
    simp [P_online_softmax, s0, onlineSoftmaxM, onlineSoftmaxL]
    exact _h_x
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
        [Stmt.assign .real .scalar "xi"
            (Op.load xReg
              (Op.add .nat .same
                (Op.mul .nat .same (Op.ref .nat .scalar "pid")
                  (Op.constNat N))
                (Op.ref .nat .scalar "i"))),
          Stmt.assign .real .scalar "m_new"
            (Op.max2 .same (Op.ref .real .scalar "m") (Op.ref .real .scalar "xi")),
          Stmt.assign .real .scalar "l"
            (Op.add .real .same
              (Op.mul .real .same
                (Op.exp (Op.sub .real .same
                  (Op.ref .real .scalar "m") (Op.ref .real .scalar "m_new")))
                (Op.ref .real .scalar "l"))
              (Op.exp (Op.sub .real .same
                (Op.ref .real .scalar "xi") (Op.ref .real .scalar "m_new")))),
          Stmt.assign .real .scalar "m" (Op.ref .real .scalar "m_new")]
        s0 = some sLoop := by
    simpa [onlineSoftmaxLoopBody] using hLoopAux
  have hExec : exec (onlineSoftmaxKernel xReg yReg N) s = some sLoop := by
    simp [exec, onlineSoftmaxKernel, stepStmts, stepStmt, evalOp,
      hLoopAuxExpanded, s0]
  constructor
  · rw [hExec]
    simp [hm]
  · rw [hExec]
    simp [hl]

end VeriTile.Examples
