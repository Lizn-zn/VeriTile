import VeriTile.Triton

/-!
# `l2_norm_triton2` — strict per-kernel correctness

Two L2-normalization kernels. `_l2_norm_fwd_1pass_kernel` (forward): program
`row` loads its row of `X`, computes the row sum of squares `var`, the reciprocal
std `rstd = 1/√(var + eps)`, and stores `x · rstd` to `Y`. `_l2_norm_bwd_kernel`
(backward): recomputes `var`/`rstd` and stores the input gradient
`dx = dy·rstd − (Σ dy·x)·(1/(var+eps))·rstd·x` to `DX`. Both are masked by
`cols < N`.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies. The host launch (`_l2_norm_*_kernel[(M,)](...)`, the 1-D
grid over rows, the `BLOCK_N = min(MAX_FUSED_SIZE, next_power_of_2(N))` choice,
the `N > BLOCK_N` guard, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here. Because `pid`
(the row) is universally quantified, each per-program statement covers every row
of the grid.

## Proof architecture

```
l2_norm_fwd_1pass_kernel_correctness          ← TOP THEOREM (l2FwdIO ⊨ l2FwdSpec)
  ├─ l2_norm_fwd_1pass_kernel_flattenOk        bridge fragment membership
  ├─ l2_norm_fwd_1pass_kernel_traceSafe        per-execution lane-wise safety walk
  └─ l2_norm_fwd_1pass_kernel_region_run       region-model masked Hoare triple
       ├─ l2_norm_fwd_1pass_kernel_exec_isSome termination
       ├─ l2_norm_fwd_1pass_kernel_correct    ← algorithm-layer readback per lane
       │    └─ l2VarCarrier_eq_l2NormSqSum     masked reduce-sum = oracle Σx²
       ├─ l2FwdSpec_congr                      only active lanes feed the spec
       └─ l2_norm_fwd_1pass_kernel_frame       masked scatter frame

l2_norm_bwd_kernel_correctness                ← TOP THEOREM (l2BwdIO ⊨ l2BwdSpec)
  ├─ l2_norm_bwd_kernel_flattenOk              bridge fragment membership
  ├─ l2_norm_bwd_kernel_traceSafe              per-execution lane-wise safety walk
  └─ l2_norm_bwd_kernel_region_run             region-model masked Hoare triple
       ├─ l2_norm_bwd_kernel_exec_isSome       termination
       ├─ l2_norm_bwd_kernel_correct          ← algorithm-layer readback per lane
       │    ├─ l2VarCarrier_eq_l2NormSqSum     masked reduce-sum = oracle Σx²
       │    └─ l2BwdDotCarrier_eq_l2NormDot    masked reduce-sum = oracle ⟨dy,x⟩
       ├─ l2BwdSpec_congr                      only active lanes feed the spec
       └─ l2_norm_bwd_kernel_frame             masked scatter frame
```

Each headline is stated on the kernel's masked **IO signature** (`l2FwdIO` :
`MaskedKernelIO₁`, `l2BwdIO` : `MaskedKernelIO₂`) with the audit-once masked
Hoare-triple combinator `⊨`: for every disjoint flat placement of the declared
buffers, every program id all of whose *active* lanes (`j < N`) are in bounds,
and every launch state whose active input-row lanes hold the given rows, the
translated pointer kernel terminates, every active output-row lane holds the
oracle L2-norm (resp. L2-norm-backward) value, and every other memory cell is
unchanged.

## Modeling boundary

The specs are **oracle wrappers** over `VeriTile.Triton.Math` L2-norm definitions
(`TiledL2Norm.l2Norm` / `l2NormBwd`, built on `l2NormSqSum` / `l2NormDot` /
`l2NormRstd`): the L2-norm math lives once in `Math.*` and is reused here, so this
file only checks that the kernels realize those oracles lane-wise. The
intra-kernel reductions `tl.sum(x·x)` and `tl.sum(dy·x)` are connected to the
oracle sum-of-squares and dot product by the bridging lemmas
`l2VarCarrier_eq_l2NormSqSum` and `l2BwdDotCarrier_eq_l2NormDot` (over
`Semantics.MaskedReduction`). Arithmetic is over `ℝ` (not bit-accurate IEEE
float); the masked loads `other=0.0`, the `tl.where(cols < N, ·, 0.0)` masking,
and the `.to(tl.float32)` casts all reduce to the algorithm-layer behavior
(post-erasure all dtypes unify to `ℝ`); `√`/`⁻¹` are the real square root and
inverse. Masked lanes enter the reductions as `0` (matching `other=0.0` +
`tl.where`), so the reduction-over-padded-block matches the upstream semantics.
No output/input disjointness is assumed in the region model: the rows are read
into registers before the masked scatter.
-/

namespace VeriTile.Bench.TritonBenchG.L2NormTriton2

open VeriTile.Triton
open VeriTile.Triton.TiledL2Norm
open scoped VeriTile.Triton.MaskedKernelIO₁
open scoped VeriTile.Triton.MaskedKernelIO₂

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `l2_norm_triton2.py`'s `_l2_norm_fwd_1pass_kernel`. -/
def l2_norm_fwd_1pass_kernel
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  Y += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  xbar = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(xbar * xbar, axis=0)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  y = x * rstd
  tl.store(Y + cols, y, mask=mask)
}

/-- Faithful transcription of `l2_norm_triton2.py`'s `_l2_norm_bwd_kernel`. -/
def l2_norm_bwd_kernel
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  X += row * $(stride_x_row)
  DX += row * $(stride_x_row)
  DY += row * $(stride_x_row)
  cols = tl.arange(0, $(BLOCK_N))
  x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  x = tl.where(cols < $(N), x, 0.0)
  var = tl.sum(x * x)
  rstd = 1 / tl.sqrt(var + $(eps))
  mask = cols < $(N)
  dy = tl.load(DY + cols, mask=cols < $(N), other=0.0).to(tl.float32)
  dy = tl.where(cols < $(N), dy, 0.0)
  dx = dy * rstd - tl.sum(dy * x) * (1 / (var + $(eps))) * rstd * x
  tl.store(DX + cols, dx, mask=mask)
}

noncomputable def l2InputTile
    (s : BlockState) (R : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      let off := s.pids 0 * stride_x_row + idx.1.val
      if idx.1.val < N then
        if idx.1.val < N then some (s.readMem R off) else some (0.0 : ℝ)
      else
        some (0.0 : ℝ) }

/-- Masked row element `R[pid0, idx]`: lane `idx` of **this program's row**
(row = `pids 0`, row stride `stride_x_row`, unit column stride), `0` beyond
the `N` bound (`mask=cols < N, other=0.0`). Shared by the `X` and `dY` loads. -/
noncomputable def l2Load
    (s : BlockState) (R : RegionName) (stride_x_row N BLOCK_N : Nat)
    (idx : Fin BLOCK_N) : ℝ :=
  if idx.val < N then
    s.readMem R (s.pids 0 * stride_x_row + idx.val)
  else
    0

def l2OutOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride_x_row + i.val

noncomputable def l2VarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (l2InputTile s X stride_x_row N BLOCK_N)
      (l2InputTile s X stride_x_row N BLOCK_N))).data PUnit.unit

theorem l2VarCarrier_eq_l2NormSqSum
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    l2VarCarrier s X stride_x_row N BLOCK_N =
      some (l2NormSqSum (l2Load s X stride_x_row N BLOCK_N)) := by
  simp [l2VarCarrier, l2InputTile, Tile.reduceSum,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop, NumericDType.mul]
  refine (reduceSum_masked_sq_eq_some_sum
    (fun k : Fin BLOCK_N => s.readMem X (s.pids 0 * stride_x_row + k.val))
    (fun k : Fin BLOCK_N => k.val < N)).trans ?_
  congr 1
  unfold l2NormSqSum l2Load
  apply Finset.sum_congr rfl
  intro k _hk
  by_cases h : k.val < N <;> simp [h]

/-- Exact L2-normalized value at lane `idx`, as a pure function of the active
row prefix `xs j`, `j < N`: masked lanes enter the oracle sum-of-squares as `0`
(matching `other=0.0` + `tl.where`). -/
noncomputable def l2FwdSpec (N BLOCK_N : Nat) (eps : ℝ)
    (xs : Fin BLOCK_N → ℝ) (idx : Fin BLOCK_N) : ℝ :=
  l2Norm (fun j => if j.val < N then xs j else 0) eps idx

/-- `l2FwdSpec` only reads the active lanes of its row: two rows agreeing below
`N` yield the same value (masked lanes are zeroed in the spec either way). -/
theorem l2FwdSpec_congr (N BLOCK_N : Nat) (eps : ℝ) (xs ys : Fin BLOCK_N → ℝ)
    (h : ∀ j : Fin BLOCK_N, j.val < N → xs j = ys j) (i : Fin BLOCK_N) :
    l2FwdSpec N BLOCK_N eps xs i = l2FwdSpec N BLOCK_N eps ys i := by
  have hrow : (fun j : Fin BLOCK_N => if j.val < N then xs j else 0)
      = (fun j : Fin BLOCK_N => if j.val < N then ys j else 0) := by
    funext j
    by_cases hj : j.val < N
    · simp only [if_pos hj, h j hj]
    · simp only [if_neg hj]
  unfold l2FwdSpec
  rw [hrow]

noncomputable def l2BwdDotCarrier
    (s : BlockState) (X DY : RegionName) (stride_x_row N BLOCK_N : Nat) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (l2InputTile s DY stride_x_row N BLOCK_N)
      (l2InputTile s X stride_x_row N BLOCK_N))).data PUnit.unit

theorem l2BwdDotCarrier_eq_l2NormDot
    (s : BlockState) (X DY : RegionName) (stride_x_row N BLOCK_N : Nat) :
    l2BwdDotCarrier s X DY stride_x_row N BLOCK_N =
      some (l2NormDot
        (l2Load s DY stride_x_row N BLOCK_N)
        (l2Load s X stride_x_row N BLOCK_N)) := by
  simp [l2BwdDotCarrier, l2InputTile, Tile.reduceSum,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop, NumericDType.mul]
  refine (reduceSum_masked_dot_eq_some_sum
    (fun k : Fin BLOCK_N => s.readMem DY (s.pids 0 * stride_x_row + k.val))
    (fun k : Fin BLOCK_N => s.readMem X (s.pids 0 * stride_x_row + k.val))
    (fun k : Fin BLOCK_N => k.val < N)).trans ?_
  congr 1
  unfold l2NormDot l2Load
  apply Finset.sum_congr rfl
  intro k _hk
  by_cases h : k.val < N <;> simp [h]

/-- Exact L2-norm input gradient at lane `idx`, as a pure function of the
active row prefixes `xs j` / `dys j`, `j < N`: masked lanes enter the oracle
sum-of-squares and dot product as `0`. -/
noncomputable def l2BwdSpec (N BLOCK_N : Nat) (eps : ℝ)
    (xs dys : Fin BLOCK_N → ℝ) (idx : Fin BLOCK_N) : ℝ :=
  l2NormBwd
    (fun j => if j.val < N then xs j else 0)
    (fun j => if j.val < N then dys j else 0)
    eps idx

/-- `l2BwdSpec` only reads the active lanes of its two rows. -/
theorem l2BwdSpec_congr (N BLOCK_N : Nat) (eps : ℝ)
    (xs xs' dys dys' : Fin BLOCK_N → ℝ)
    (hx : ∀ j : Fin BLOCK_N, j.val < N → xs j = xs' j)
    (hdy : ∀ j : Fin BLOCK_N, j.val < N → dys j = dys' j) (i : Fin BLOCK_N) :
    l2BwdSpec N BLOCK_N eps xs dys i = l2BwdSpec N BLOCK_N eps xs' dys' i := by
  have hxrow : (fun j : Fin BLOCK_N => if j.val < N then xs j else 0)
      = (fun j : Fin BLOCK_N => if j.val < N then xs' j else 0) := by
    funext j
    by_cases hj : j.val < N
    · simp only [if_pos hj, hx j hj]
    · simp only [if_neg hj]
  have hdyrow : (fun j : Fin BLOCK_N => if j.val < N then dys j else 0)
      = (fun j : Fin BLOCK_N => if j.val < N then dys' j else 0) := by
    funext j
    by_cases hj : j.val < N
    · simp only [if_pos hj, hdy j hj]
    · simp only [if_neg hj]
  unfold l2BwdSpec
  rw [hxrow, hdyrow]

/-- Algorithm-layer correctness for `_l2_norm_fwd_1pass_kernel`: in-bounds
lanes hold `l2FwdSpec` of the loaded row, out-of-bounds lanes are preserved. -/
theorem l2_norm_fwd_1pass_kernel_correct
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      let outAddr := l2OutOffset s stride_x_row i
      s'.readMem Y outAddr =
        if i.val < N then
          l2FwdSpec N BLOCK_N eps
            (fun j => s.readMem X (s.pids 0 * stride_x_row + j.val)) i
        else s.readMem Y outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, l2_norm_fwd_1pass_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, FloatDType.cast,
          FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    rw [← hExec]
    simp only [l2OutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · have hvar := l2VarCarrier_eq_l2NormSqSum s X stride_x_row N BLOCK_N
      simp [l2VarCarrier, l2InputTile, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.bop, NumericDType.mul] at hvar
      simp [hi, l2FwdSpec, WithBot.realSqrt, l2Load, l2Norm, l2NormRstd]
      erw [hvar]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Algorithm-layer correctness for `_l2_norm_bwd_kernel`: in-bounds lanes
hold `l2BwdSpec` of the loaded rows, out-of-bounds lanes are preserved. -/
theorem l2_norm_bwd_kernel_correct
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      let outAddr := l2OutOffset s stride_x_row i
      s'.readMem DX outAddr =
        if i.val < N then
          l2BwdSpec N BLOCK_N eps
            (fun j => s.readMem X (s.pids 0 * stride_x_row + j.val))
            (fun j => s.readMem DY (s.pids 0 * stride_x_row + j.val)) i
        else s.readMem DX outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, l2_norm_bwd_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.sub, NumericDType.div, ComparableDType.lt,
          FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    rw [← hExec]
    simp only [l2OutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · have hvar := l2VarCarrier_eq_l2NormSqSum s X stride_x_row N BLOCK_N
      have hdot := l2BwdDotCarrier_eq_l2NormDot s X DY stride_x_row N BLOCK_N
      simp [l2VarCarrier, l2InputTile, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.bop, NumericDType.mul] at hvar
      simp [l2BwdDotCarrier, l2InputTile, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.bop, NumericDType.mul] at hdot
      simp [hi, l2BwdSpec, WithBot.realSqrt, l2Load, l2NormBwd, l2NormRstd]
      erw [hvar, hdot]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked stores). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively written by the masked `Y`
store — every cell of every region other than `Y`, and the *inactive* lanes
of the output row itself — is preserved by the forward run. -/
private theorem l2_norm_fwd_1pass_kernel_frame
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) (s s1 : BlockState)
    (hExec : exec ((l2_norm_fwd_1pass_kernel X Y stride_x_row N eps
        BLOCK_N).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_N, i.val < N →
      ¬(Y = r ∧ s.pid * stride_x_row + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, l2_norm_fwd_1pass_kernel, ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the forward kernel executes to completion from any state
(`tl.sum` is total, so no non-emptiness side condition is needed). -/
private theorem l2_norm_fwd_1pass_kernel_exec_isSome
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) (s : BlockState) :
    ∃ s1, exec ((l2_norm_fwd_1pass_kernel X Y stride_x_row N eps
        BLOCK_N).toAlgKernel) s = some s1 := by
  simp [exec, l2_norm_fwd_1pass_kernel, ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked Hoare triple** (forward) — termination,
active-lane output values, and frame off the active output lanes, from any
launch state whose input row is loaded at the **active lanes only**
(`j < N`). This is the `hrun` obligation of the `⊨` headline. -/
theorem l2_norm_fwd_1pass_kernel_region_run
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s₀ : BlockState) (xs : Fin BLOCK_N → ℝ)
    (hx : ∀ j : Fin BLOCK_N, j.val < N →
      s₀.readMem X (s₀.pid * stride_x_row + j.val) = xs j) :
    ∃ s1, exec ((l2_norm_fwd_1pass_kernel X Y stride_x_row N eps
          BLOCK_N).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_N, j.val < N →
          s1.readMem Y (s₀.pid * stride_x_row + j.val)
            = l2FwdSpec N BLOCK_N eps xs j)
      ∧ (∀ r o,
          (r ≠ Y ∨ ∀ j : Fin BLOCK_N, j.val < N →
            o ≠ s₀.pid * stride_x_row + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := l2_norm_fwd_1pass_kernel_exec_isSome X Y stride_x_row
    N eps BLOCK_N s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := l2_norm_fwd_1pass_kernel_correct X Y stride_x_row N eps BLOCK_N
      s₀ s1 hs1 j
    simp only [l2OutOffset, hj, if_pos] at h
    simp only [BlockState.pid_eq]
    rw [h]
    exact l2FwdSpec_congr N BLOCK_N eps _ xs (fun k hk => hx k hk) j
  · refine l2_norm_fwd_1pass_kernel_frame X Y stride_x_row N eps BLOCK_N
      s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The forward kernel sits inside the flat-memory bridge's covered fragment
(pointer arithmetic, masked load with `other`, `where`, reduction, cast,
masked store). -/
theorem l2_norm_fwd_1pass_kernel_flattenOk
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ((l2_norm_fwd_1pass_kernel X Y stride_x_row N eps
        BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [l2_norm_fwd_1pass_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk (forward): one computational unfold walks all
eleven statements — nine are memory-silent (`program_id`, pointer staging,
`arange`, `where`, the reduction, and the register arithmetic) — and reduces
the two masked accesses (row load, row store) to the **lane-wise** bounds
hypotheses: every *active* lane's address is below the region bound. -/
theorem l2_norm_fwd_1pass_kernel_traceSafe
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_N, j.val < N →
      s.pid * stride_x_row + j.val < bounds X)
    (hout : ∀ j : Fin BLOCK_N, j.val < N →
      s.pid * stride_x_row + j.val < bounds Y) :
    Kernel.TraceSafe bounds
      ((l2_norm_fwd_1pass_kernel X Y stride_x_row N eps
        BLOCK_N).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [l2_norm_fwd_1pass_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.div,
    ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
    FloatDType.toWithBot,
    Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

/-- `_l2_norm_fwd_1pass_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_N` — the row window each program owns;
* `read`/`write` — program `pid` reads and writes its row at
  `pid * stride_x_row` (the host-side one-program-per-row launch convention;
  `Y` shares `X`'s row stride upstream);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = next_power_of_2(N)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def l2FwdIO (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    MaskedKernelIO₁ where
  kernel := l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N
  inp := X
  out := Y
  B := BLOCK_N
  read := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N

/-- **The forward headline**: `_l2_norm_fwd_1pass_kernel` implements the exact
L2-normalization `l2FwdSpec` (the `Math.TiledL2Norm.l2Norm` oracle over the
zero-padded active prefix) on its masked IO signature — for every disjoint
flat placement of the two buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane `j` holds
`l2FwdSpec N BLOCK_N eps xs j`, and every other memory cell is unchanged.
Proof: `Implements.intro` assembles the region-model masked triple with the
bridge side conditions. -/
specification l2_norm_fwd_1pass_kernel_correctness
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2FwdIO X Y stride_x_row N eps BLOCK_N ⊨
      fun xs i => l2FwdSpec N BLOCK_N eps xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact l2_norm_fwd_1pass_kernel_flattenOk X Y stride_x_row N eps BLOCK_N
  · intro bounds s h1 h2 _
    exact l2_norm_fwd_1pass_kernel_traceSafe X Y stride_x_row N eps BLOCK_N
      bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := l2_norm_fwd_1pass_kernel_region_run
      X Y stride_x_row N eps BLOCK_N s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

set_option maxHeartbeats 1600000 in
/-- Frame half (backward): every memory cell not actively written by the
masked `DX` store is preserved by the run. -/
private theorem l2_norm_bwd_kernel_frame
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) (s s1 : BlockState)
    (hExec : exec ((l2_norm_bwd_kernel X DY DX stride_x_row N eps
        BLOCK_N).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_N, i.val < N →
      ¬(DX = r ∧ s.pid * stride_x_row + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, l2_norm_bwd_kernel, ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the backward kernel executes to completion from any state
(`tl.sum` is total, so no non-emptiness side condition is needed). -/
private theorem l2_norm_bwd_kernel_exec_isSome
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) (s : BlockState) :
    ∃ s1, exec ((l2_norm_bwd_kernel X DY DX stride_x_row N eps
        BLOCK_N).toAlgKernel) s = some s1 := by
  simp [exec, l2_norm_bwd_kernel, ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked Hoare triple** (backward) — termination,
active-lane output values, and frame off the active output lanes, from any
launch state whose `X` and `DY` rows are loaded at the **active lanes only**
(`j < N`). This is the `hrun` obligation of the `⊨` headline. -/
theorem l2_norm_bwd_kernel_region_run
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s₀ : BlockState) (xs dys : Fin BLOCK_N → ℝ)
    (hx : ∀ j : Fin BLOCK_N, j.val < N →
      s₀.readMem X (s₀.pid * stride_x_row + j.val) = xs j)
    (hdy : ∀ j : Fin BLOCK_N, j.val < N →
      s₀.readMem DY (s₀.pid * stride_x_row + j.val) = dys j) :
    ∃ s1, exec ((l2_norm_bwd_kernel X DY DX stride_x_row N eps
          BLOCK_N).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_N, j.val < N →
          s1.readMem DX (s₀.pid * stride_x_row + j.val)
            = l2BwdSpec N BLOCK_N eps xs dys j)
      ∧ (∀ r o,
          (r ≠ DX ∨ ∀ j : Fin BLOCK_N, j.val < N →
            o ≠ s₀.pid * stride_x_row + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := l2_norm_bwd_kernel_exec_isSome X DY DX stride_x_row
    N eps BLOCK_N s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := l2_norm_bwd_kernel_correct X DY DX stride_x_row N eps BLOCK_N
      s₀ s1 hs1 j
    simp only [l2OutOffset, hj, if_pos] at h
    simp only [BlockState.pid_eq]
    rw [h]
    exact l2BwdSpec_congr N BLOCK_N eps _ xs _ dys
      (fun k hk => hx k hk) (fun k hk => hdy k hk) j
  · refine l2_norm_bwd_kernel_frame X DY DX stride_x_row N eps BLOCK_N
      s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The backward kernel sits inside the flat-memory bridge's covered fragment. -/
theorem l2_norm_bwd_kernel_flattenOk
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ((l2_norm_bwd_kernel X DY DX stride_x_row N eps
        BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [l2_norm_bwd_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk (backward): one computational unfold walks all
fifteen statements and reduces the three masked accesses (`X` load, `DY`
load, `DX` store) to the **lane-wise** bounds hypotheses: every *active*
lane's address is below the region bound of the buffer it touches. -/
theorem l2_norm_bwd_kernel_traceSafe
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hinX : ∀ j : Fin BLOCK_N, j.val < N →
      s.pid * stride_x_row + j.val < bounds X)
    (hinDY : ∀ j : Fin BLOCK_N, j.val < N →
      s.pid * stride_x_row + j.val < bounds DY)
    (hout : ∀ j : Fin BLOCK_N, j.val < N →
      s.pid * stride_x_row + j.val < bounds DX) :
    Kernel.TraceSafe bounds
      ((l2_norm_bwd_kernel X DY DX stride_x_row N eps
        BLOCK_N).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hinX hinDY hout
  simp [l2_norm_bwd_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
    FloatDType.toWithBot,
    Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hinX a ha, fun a ha => hinDY a ha, fun a ha => hout a ha⟩

/-- `_l2_norm_bwd_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring: `X`, `DY`
  in, `DX` out);
* `B = BLOCK_N` — the row window each program owns;
* `read1`/`read2`/`write` — program `pid` reads and writes all three rows at
  `pid * stride_x_row` (the host-side one-program-per-row launch convention;
  `DY`/`DX` share `X`'s row stride upstream);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes carry no
  obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. -/
def l2BwdIO (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    MaskedKernelIO₂ where
  kernel := l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N
  in1 := X
  in2 := DY
  out := DX
  B := BLOCK_N
  read1 := fun pid => pid * stride_x_row
  read2 := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N

/-- **The backward headline**: `_l2_norm_bwd_kernel` implements the exact
L2-norm input gradient `l2BwdSpec` (the `Math.TiledL2Norm.l2NormBwd` oracle
over the zero-padded active prefixes) on its masked IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active `X`/`DY` row lanes
hold `xs`/`dys`, the translated pointer kernel terminates, every active
output-row lane `j` holds `l2BwdSpec N BLOCK_N eps xs dys j`, and every other
memory cell is unchanged. Proof: `Implements.intro` assembles the
region-model masked triple with the bridge side conditions. -/
specification l2_norm_bwd_kernel_correctness
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2BwdIO X DY DX stride_x_row N eps BLOCK_N ⊨
      fun xs dys i => l2BwdSpec N BLOCK_N eps xs dys i := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact l2_norm_bwd_kernel_flattenOk X DY DX stride_x_row N eps BLOCK_N
  · intro bounds s h1 h2 h3 _
    exact l2_norm_bwd_kernel_traceSafe X DY DX stride_x_row N eps BLOCK_N
      bounds s h1 h2 h3
  · intro s₀ xs dys hx hdy
    obtain ⟨s1, hexec, hval, hframe⟩ := l2_norm_bwd_kernel_region_run
      X DY DX stride_x_row N eps BLOCK_N s₀ xs dys hx hdy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.L2NormTriton2
