/-
bench/examples/OnlineSoftmax

**Online softmax**: the streaming `(M, L)` recurrence, in three parts.

1. **The math** — `onlineSoftmaxM` / `onlineSoftmaxL`, the `WithBot ℝ`-valued
   streaming recurrence (seeding `M` at `⊥` removes every range
   precondition on the input), and `online_softmax_recurrence_eq_batch`:
   the recurrence equals the batch form `(tileMax, ∑ exp)`.
2. **The online kernel** — `onlineSoftmaxKernel`, a typed Triton loop that
   maintains `(m, l)` in registers; `online_softmax_correct` proves those
   registers hold exactly `onlineSoftmaxM/L` after the run.
3. **The headline** — the file's single `specification`:

       online_softmax_correctness : batchSoftmaxIO B ⊨ fun xs i =>
         Real.exp (xs i - (onlineSoftmaxM xs B).unbotD 0)
           / (onlineSoftmaxL xs B).unbotD 0

   `⊨` is the audit-once Hoare-triple combinator (`KernelIO₁.Implements`):
   ∀ disjoint buffer placement (∀ base pointers, ∀ sizes), ∀ program id in
   bounds, ∀ launch state with the input window loaded and everything else
   arbitrary — the pointer kernel terminates, the output window holds the
   value pointwise, and every other cell is unchanged. The mathematical
   function is the **online recurrence itself** — that is the point of the
   file: the batch kernel provably implements the streaming formulation
   (via part 1's identity), the streaming kernel provably computes it in
   registers (part 2).
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.KernelLemmas
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Math.Softmax
import VeriTile.Examples.Common
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.OnlineSoftmax

open VeriTile.Triton VeriTile.Triton.TiledSoftmax
open VeriTile.Examples

/-! ## Batch reference kernel

`stableSoftmaxKernel` is this file's own copy of the max-subtracted softmax
kernel (per the "each showcase self-contained" convention for
`bench/examples/`; the original lived in `VeriTile.Examples.SoftmaxEq`). -/

/-- Stable softmax: subtract the max before exponentiating.
(This file's own copy — see the section header above.) -/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

def batchSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : ComputeKernel :=
  stableSoftmaxKernel xReg yReg N

def onlineSoftmaxKernel (xReg _yReg : RegionName) (N : Nat) : ComputeKernel := triton {
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

/-! ## KernelIO spec — `batchSoftmaxIO ⊨` the online recurrence

The `io ⊨ f` correctness headline for the batch kernel, following the
canonical KernelIO showcase `bench/examples/VectorAdd.lean` (softmax is
one-input/one-output, so `KernelIO₁`). The mathematical function `f` is the
**online `(M, L)` recurrence itself** — `exp (xs i − M_B) / L_B` with
`M_B = onlineSoftmaxM xs B` and `L_B = onlineSoftmaxL xs B` (read back from
`WithBot ℝ` via `unbotD 0`; at step `B ≥ 1` both are genuine reals) — which
is the whole point of this file: by `online_softmax_recurrence_eq_batch`
this equals batch softmax, but the headline keeps the streaming
formulation. Three parts, as in `VectorAdd`:

1. **Region-model Hoare triple** (`batchSoftmax_region_run`) — termination,
   output-window values (the online recurrence, via the batch closed form
   plus the math identity), and the cell-level frame, from any launch state
   whose input window is loaded. Exactly the `hrun` obligation of
   `KernelIO₁.Implements.intro`.
2. **Flat-memory bridge side conditions** — `TraceSafe` (the per-execution
   safety walk: the register ops `max`/`exp`/`sum`/`div` are memory-silent;
   only the one load and the one store carry bounds obligations, both
   through the unmasked `offs` tile) and `FlattenOk` (bridge fragment
   membership).
3. **The spec** — `online_softmax_correctness : batchSoftmaxIO B ⊨ …`,
   spelled out: for **every** disjoint placement of the two buffers in flat
   memory, **every** program id whose windows are in bounds, and **every**
   launch state whose input window holds `xs` — all other buffer cells and
   all registers arbitrary — the translated pointer kernel terminates, its
   output window holds the online-recurrence value pointwise, and every
   other memory cell is unchanged. -/
section OnlineSoftmax.kernelIO

open VeriTile.Triton.KernelIO₁ (Implements)
open scoped VeriTile.Triton.KernelIO₁

/-! ### Part 1 — the region-model Hoare triple -/

/-- Closed form of the batch kernel's stores: after the run, the output
window holds the max-shifted softmax `stableSpec xs (tileMax hB xs)`.
Statement-walk proof; the online-recurrence reading is layered on in
`batchSoftmax_region_run` via `online_softmax_recurrence_eq_batch`. -/
private theorem batchSoftmax_correct
    (xReg yReg : RegionName)
    (B : Nat) (hB : 0 < B) (s : BlockState) (xs : Fin B → ℝ)
    (_h_x : InputLoadedAt s xReg B xs) :
    ∀ i : Fin B,
      observeAt (exec (batchSoftmaxKernel xReg yReg B) s) yReg B s.pid i
        = some (stableSpec xs (tileMax hB xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [n + 1] => s.pid * (n + 1) + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, batchSoftmaxKernel, stableSoftmaxKernel, stepStmts,
        stepStmt, Tile.bop, Tile.uop,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        NumericDType.div, stableSpec, tileMax]
  repeat unfold evalOp
  simp [Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  unfold InputLoadedAt at _h_x
  simp [_h_x]
  rfl

/-- A scatter-store `foldl` leaves every memory cell it does not hit
unchanged (cell-level frame for the unmasked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ)
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        acc.writeMem region (offsetFn k) (valueFn k)) s).mem r o
      = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons,
        ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem]
      exact if_neg (fun hc =>
        hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

/-- Frame half: every memory cell other than the output window is preserved
by the run. Non-emptiness is needed because `Tile.reduceMax` (hence the
kernel's termination) is only defined on positive-length axes. -/
private theorem batchSoftmax_frame (xReg yReg : RegionName)
    (B : Nat) (hB : 0 < B) (s s1 : BlockState)
    (hExec : exec ((batchSoftmaxKernel xReg yReg B).toAlgKernel) s
      = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin B, ¬(yReg = r ∧ s.pid * B + i.val = o)) :
    s1.mem r o = s.mem r o := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  simp [exec, batchSoftmaxKernel, stableSoftmaxKernel,
    ComputeKernel.toAlgKernel, stepStmts,
    stepStmt, Tile.bop, Tile.uop,
    NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div] at hExec
  repeat unfold evalOp at hExec
  simp [Tile.reduceSum, Tile.reduceSumDrop, Tile.reduceMax, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ r o _ _ ?_) rfl
  intro k _ hc
  exact hmiss k.1 hc

/-- **Region-model Hoare triple for the batch kernel** — termination, the
output window holding the **online recurrence** `exp (xs j − M_B) / L_B`
(the batch closed form rewritten through
`online_softmax_recurrence_eq_batch`), and frame, from any launch state
whose input window is loaded. This is what the `⊨` headline transports to
flat memory. -/
theorem batchSoftmax_region_run (B : Nat) (hB : 0 < B)
    (s₀ : BlockState) (xs : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.readMem ⟨"x"⟩ (s₀.pid * B + j.val) = xs j) :
    ∃ s1, exec ((batchSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B).toAlgKernel) s₀
        = some s1
      ∧ (∀ j : Fin B,
          s1.readMem ⟨"y"⟩ (s₀.pid * B + j.val)
            = Real.exp (xs j - (onlineSoftmaxM xs B).unbotD 0)
                / (onlineSoftmaxL xs B).unbotD 0)
      ∧ (∀ r o,
          (r ≠ ⟨"y"⟩ ∨ ∀ j : Fin B, o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  -- The online recurrence at step B collapses to the batch closed form.
  have hM : (onlineSoftmaxM xs B).unbotD 0 = tileMax hB xs := by
    rw [(online_softmax_recurrence_eq_batch hB xs).1]
    exact WithBot.unbotD_coe 0 _
  have hL : (onlineSoftmaxL xs B).unbotD 0
      = ∑ k, Real.exp (xs k - tileMax hB xs) := by
    rw [(online_softmax_recurrence_eq_batch hB xs).2]
    exact WithBot.unbotD_coe 0 _
  have hobs := batchSoftmax_correct ⟨"x"⟩ ⟨"y"⟩ B hB s₀ xs hx
  rw [show exec (batchSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B) s₀
      = exec ((batchSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B).toAlgKernel) s₀ from rfl]
    at hobs
  cases hsrc : exec ((batchSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B).toAlgKernel) s₀ with
  | none =>
      have := hobs ⟨0, hB⟩
      rw [hsrc] at this
      simp [observeAt] at this
  | some s1 =>
      refine ⟨s1, rfl, fun j => ?_, fun r o hcond => ?_⟩
      · have := hobs j
        rw [hsrc] at this
        rw [hM, hL]
        simpa [observeAt, stableSpec] using this
      · refine batchSoftmax_frame ⟨"x"⟩ ⟨"y"⟩ B hB s₀ s1 hsrc r o
          (fun i ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i ho.symm

/-! ### Part 2 — flat-memory bridge side conditions -/

/-- Inversion for a successful `assign` step. -/
private theorem stepStmt_assign_inv {d : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op d sh} {s s' : BlockState}
    (h : stepStmt (.assign d sh nm e) s = some s') :
    ∃ v, evalOp e s = some v ∧ s' = s.setReg nm d sh v := by
  simp only [stepStmt] at h
  cases hv : evalOp e s with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      replace h : some (s.setReg nm d sh v) = some s' := h
      exact ⟨v, rfl, (Option.some_inj.mp h).symm⟩

/-- Bounds discharge for accesses through the `offs` register: in any state
whose `offs` register holds the program tile's offsets
`pid₀ * B + i`, all addressed cells lie below `pid₀ * B + B ≤ bounds reg`. -/
private theorem offs_activeAddressSafe (B : Nat) (bounds : RegionBounds)
    (pid₀ : Nat) (t : BlockState) (active : TileIndex [B] → Prop)
    (hread : t.regs .nat [B] "offs" = some ⟨fun i => pid₀ * B + i.1.val⟩)
    (reg : RegionName) (hreg : pid₀ * B + B ≤ bounds reg) :
    MemAccess.ActiveAddressSafe bounds
      (MemAccess.region reg (Op.ref .nat [B] "offs")) t active := by
  simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp (Op.ref .nat [B] "offs") t
      = some ⟨fun i => pid₀ * B + i.1.val⟩ from by simp [hread]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hreg

set_option maxHeartbeats 1600000 in
/-- The batch kernel is trace-safe: of its eight statements, only the `x`
load and the final store touch memory — both through the unmasked `offs`
tile, so each region's bound must cover the whole program tile. The
register ops (`max`/`exp`/`sum`/`div`) are memory-silent. -/
theorem batchSoftmax_traceSafe (xReg yReg : RegionName)
    (B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg) :
    Kernel.TraceSafe bounds
      ((batchSoftmaxKernel xReg yReg B).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe batchSoftmaxKernel
  -- statement 1: pid := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp (Op.programId 0) s
      = some (Tile.scalar (dtype := .nat) (s.pids 0)) from by simp] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- statement 2: offs := pid * B + arange B
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  rw [show evalOp (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat B))
      (Op.arange B))
      (s.setReg "pid" .nat [] (Tile.scalar (dtype := .nat) (s.pids 0)))
      = some ⟨fun i => s.pids 0 * B + i.1.val⟩ from by
    simp [Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  -- statement 3: x := load(xReg + offs)   (unmasked: all lanes active)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨by simp, trivial,
      offs_activeAddressSafe B bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) xReg hx⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  -- statement 4: m := max(x, axis=0)   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 5: e := exp(x - m)   (register op)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  -- statement 6: s := sum(e, axis=0)   (register op)
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  -- statement 7: y := e / s   (register op)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s7 hs7
  obtain ⟨v7, hv7, rfl⟩ := stepStmt_assign_inv hs7
  -- statement 8: store(yReg + offs, y)   (unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
    offs_activeAddressSafe B bounds (s.pids 0) _ _ ?_ yReg hy⟩
  simp [BlockState.setReg]

/-- The batch kernel sits inside the bridge's covered fragment. -/
theorem batchSoftmax_flattenOk (xReg yReg : RegionName) (B : Nat) :
    ((batchSoftmaxKernel xReg yReg B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [batchSoftmaxKernel, stableSoftmaxKernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### Part 3 — the spec: `batchSoftmaxIO ⊨` the online recurrence -/

/-- `batchSoftmaxKernel`'s **IO signature** — the whole kernel-specific
audit surface of its headline: which buffer is the input (`x`) and which
the output (`y`), the tile lengths (`Bin = Bout = B`: softmax is a
whole-tile map), and where program `pid` reads/writes its tile (`pid * B`,
the host-side launch convention `offs = pid * N + arange`). The windows are
declared, not parsed from the kernel: the headline **proves** the kernel's
actual addressing matches them. -/
def batchSoftmaxIO (B : Nat) : KernelIO₁ where
  kernel := batchSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- **The headline**: the batch softmax kernel implements the **online
`(M, L)` recurrence**, pointwise `exp (xs i − M_B) / L_B` with
`M_B = onlineSoftmaxM xs B` and `L_B = onlineSoftmaxL xs B` (read back from
`WithBot ℝ` via `unbotD 0`), on its IO signature — see the section
docstring for the full Hoare triple `⊨` unfolds to. This is the file's
story stated as one `io ⊨ f` triple: the streaming recurrence *is* a
correct softmax spec, and the batch kernel provably computes it. Proof:
`Implements.intro` assembles the region-model triple (Part 1) with the
bridge side conditions (Part 2). -/
specification online_softmax_correctness (B : Nat) (hB : 0 < B) :
    batchSoftmaxIO B ⊨ fun xs i =>
      Real.exp (xs i - (onlineSoftmaxM xs B).unbotD 0)
        / (onlineSoftmaxL xs B).unbotD 0 := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact batchSoftmax_flattenOk ⟨"x"⟩ ⟨"y"⟩ B
  · intro bounds s h1 h2 _
    exact batchSoftmax_traceSafe ⟨"x"⟩ ⟨"y"⟩ B bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := batchSoftmax_region_run B hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end OnlineSoftmax.kernelIO

/-! ## Trust gates -/

-- No `sorry`, no smuggled axiom, in the headline's transitive proof
-- (and in the online kernel's register-level theorem).
#axiomsClean online_softmax_correctness
#axiomsClean online_softmax_correct

/- The headline's statement surface is the IO signature, the audit-once
Hoare-triple combinator, and the online-recurrence math constants
(`onlineSoftmaxM`/`onlineSoftmaxL`; `Real.exp` and `WithBot.unbotD` are
core-listed; `Zero.toOfNat0` is the Mathlib numeral-`0` instance behind the
`unbotD 0` default) — no other project constant. -/
#stmtSurfaceSubset online_softmax_correctness ⊆
  [batchSoftmaxIO, VeriTile.Triton.KernelIO₁.Implements,
   VeriTile.Triton.KernelIO₁.Bin, VeriTile.Triton.KernelIO₁.Bout,
   onlineSoftmaxM, onlineSoftmaxL, Zero.toOfNat0]

end VeriTile.Bench.Examples.OnlineSoftmax
