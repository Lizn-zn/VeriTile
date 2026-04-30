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
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

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

/-! ## Pure Welford Math -/

/-- Two-pass mean: μ = (∑ xᵢ) / n. -/
noncomputable def twoPassMean {n : Nat} (x : Fin n → ℝ) : ℝ :=
  (∑ i, x i) / n

/-- Two-pass sum-of-squared-deviations: S = ∑ (xᵢ − μ)². -/
noncomputable def twoPassS {n : Nat} (x : Fin n → ℝ) : ℝ :=
  ∑ i, (x i - twoPassMean x) ^ 2

/-- Welford recurrence: running mean M_k after processing x[0..k-1].
    M_0 = 0, M_{k+1} = M_k + (x_k − M_k) / (k+1).
    Returns 0 if k > n (out-of-range). -/
noncomputable def welfordMean {n : Nat} (x : Fin n → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < n then
        let prev := welfordMean x k
        prev + (x ⟨k, h⟩ - prev) / (k + 1)
      else welfordMean x k

/-- Welford recurrence: running sum-of-squared-deviations S_k.
    S_0 = 0, S_{k+1} = S_k + (x_k − M_k) · (x_k − M_{k+1}). -/
noncomputable def welfordS {n : Nat} (x : Fin n → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < n then
        let prevM := welfordMean x k
        let curM  := welfordMean x (k + 1)
        welfordS x k + (x ⟨k, h⟩ - prevM) * (x ⟨k, h⟩ - curM)
      else welfordS x k

/-- Helper: for any prefix length k ≤ n, the running Welford mean times k
    equals the sum of the first k inputs. Used to derive the final
    `welfordMean x n = twoPassMean x` claim. -/
private theorem welford_mean_mul_eq_sum {n : Nat} (x : Fin n → ℝ) :
    ∀ k : Nat, ∀ (h : k ≤ n),
      welfordMean x k * k = ∑ i : Fin k, x (castFin h i) := by
  intro k
  induction k with
  | zero =>
    intro _
    simp [welfordMean]
  | succ j ih =>
    intro hk
    have hj : j ≤ n := Nat.le_of_succ_le hk
    have hj_lt : j < n := hk
    have ih' := ih hj
    have hwm : welfordMean x (j + 1) =
        welfordMean x j + (x ⟨j, hj_lt⟩ - welfordMean x j) / (j + 1) := by
      simp [welfordMean, hj_lt]
    rw [hwm]
    have hjp1_ne : ((j : ℝ) + 1) ≠ 0 := by
      have : (0 : ℝ) ≤ j := Nat.cast_nonneg j
      linarith
    have hcast : ((j + 1 : Nat) : ℝ) = (j : ℝ) + 1 := by push_cast; ring
    rw [hcast]
    have hlhs : (welfordMean x j + (x ⟨j, hj_lt⟩ - welfordMean x j) / ((j : ℝ) + 1))
                  * ((j : ℝ) + 1) = welfordMean x j * j + x ⟨j, hj_lt⟩ := by
      field_simp
      ring
    rw [hlhs]
    rw [Fin.sum_univ_castSucc]
    have h_last : x (castFin hk (Fin.last j)) = x ⟨j, hj_lt⟩ := by
      rfl
    have h_cs : ∀ i : Fin j, x (castFin hk i.castSucc) = x (castFin hj i) := by
      intro i; rfl
    simp only [h_cs, h_last]
    rw [ih']

/-- Welford's variance identity. For any prefix length k ≤ n,
    the running Welford S_k equals the sum-of-squared-deviations from M_k. -/
private theorem welford_S_eq_sum_sq_dev {n : Nat} (x : Fin n → ℝ) :
    ∀ k : Nat, ∀ (hk : k ≤ n),
      welfordS x k = ∑ i : Fin k, (x (castFin hk i) - welfordMean x k) ^ 2 := by
  intro k
  induction k with
  | zero =>
    intro _
    simp [welfordS]
  | succ j ih =>
    intro hk
    have hj : j ≤ n := Nat.le_of_succ_le hk
    have hj_lt : j < n := hk
    have ih' := ih hj
    set M := welfordMean x j with hMdef
    set M' := welfordMean x (j + 1) with hM'def
    set xj := x ⟨j, hj_lt⟩ with hxjdef
    have hM' : M' = M + (xj - M) / ((j : ℝ) + 1) := by
      simp [hM'def, welfordMean, hj_lt, hMdef, hxjdef]
    have hS' : welfordS x (j + 1) = welfordS x j + (xj - M) * (xj - M') := by
      simp [welfordS, hj_lt, hMdef, hM'def, hxjdef]
    have hjp1_pos : (0 : ℝ) < (j : ℝ) + 1 := by
      have : (0 : ℝ) ≤ j := Nat.cast_nonneg j
      linarith
    have hjp1_ne : ((j : ℝ) + 1) ≠ 0 := ne_of_gt hjp1_pos
    have hMean := welford_mean_mul_eq_sum x j hj
    have hMM' : ((j : ℝ) + 1) * (M' - M) = xj - M := by
      rw [hM']; field_simp; ring
    have hxj_M' : xj - M' = (j : ℝ) * (M' - M) := by
      have : xj - M' = (xj - M) - (M' - M) := by ring
      rw [this, ← hMM']; ring
    have hxj_M : xj - M = ((j : ℝ) + 1) * (M' - M) := hMM'.symm
    rw [hS', ih']
    rw [Fin.sum_univ_castSucc]
    have h_last : x (castFin hk (Fin.last j)) = xj := rfl
    have h_cs : ∀ i : Fin j, x (castFin hk i.castSucc) = x (castFin hj i) := by
      intro i; rfl
    simp only [h_cs, h_last]
    have key :
        (∑ i : Fin j, (x (castFin hj i) - M') ^ 2) -
          (∑ i : Fin j, (x (castFin hj i) - M) ^ 2)
        = (j : ℝ) * (M - M') ^ 2 := by
      rw [← Finset.sum_sub_distrib]
      have h_per : ∀ i : Fin j,
          (x (castFin hj i) - M') ^ 2 - (x (castFin hj i) - M) ^ 2 =
          (M - M') * (2 * x (castFin hj i) - M - M') := by
        intro i; ring
      simp_rw [h_per]
      rw [← Finset.mul_sum]
      have h_sum_split : ∀ i : Fin j,
          2 * x (castFin hj i) - M - M' =
            2 * x (castFin hj i) + (- M - M') := by
        intro i; ring
      simp_rw [h_sum_split]
      rw [Finset.sum_add_distrib, ← Finset.mul_sum]
      rw [Finset.sum_const]
      simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
      have hSumX : ∑ i : Fin j, x (castFin hj i) = M * j := hMean.symm
      rw [hSumX]
      ring
    have lhs_alg : (xj - M) * (xj - M') - (xj - M') ^ 2
                 = (j : ℝ) * (M - M') ^ 2 := by
      rw [hxj_M, hxj_M']
      ring
    linarith [key, lhs_alg]

/-- The load-bearing identity for Welford kernel refinement: after processing
all `n` inputs, Welford's running `(M, S)` equals the two-pass `(μ, S)`. -/
theorem welford_eq_two_pass {n : Nat} (hn : 0 < n) (x : Fin n → ℝ) :
    welfordMean x n = twoPassMean x ∧ welfordS x n = twoPassS x := by
  refine ⟨?_, ?_⟩
  · have hMul := welford_mean_mul_eq_sum x n (le_refl n)
    have hn_pos : (0 : ℝ) < n := by exact_mod_cast hn
    have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
    unfold twoPassMean
    rw [eq_div_iff hn_ne, hMul]
    apply Finset.sum_congr rfl
    intro i _
    rfl
  · have hS := welford_S_eq_sum_sq_dev x n (le_refl n)
    have hM : welfordMean x n = twoPassMean x := by
      have hMul := welford_mean_mul_eq_sum x n (le_refl n)
      have hn_pos : (0 : ℝ) < n := by exact_mod_cast hn
      have hn_ne : (n : ℝ) ≠ 0 := ne_of_gt hn_pos
      unfold twoPassMean
      rw [eq_div_iff hn_ne, hMul]
      apply Finset.sum_congr rfl
      intro i _; rfl
    rw [hS, hM]
    unfold twoPassS
    apply Finset.sum_congr rfl
    intro i _
    rfl



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
        Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.natToReal, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        welfordMeanSpec, twoPassMean]
    unfold InputLoadedAt at _h_x
    simp_rw [_h_x]
    simp [_h_mv]
    rfl
  · simp [exec, twopassWelfordKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.natToReal, NumericDType.add,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        welfordVarSpec, twoPassS, twoPassMean]
    unfold InputLoadedAt at _h_x
    simp_rw [_h_x]
    simp [pow_two]
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
