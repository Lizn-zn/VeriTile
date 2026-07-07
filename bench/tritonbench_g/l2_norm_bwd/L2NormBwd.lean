import VeriTile.Triton

/-!
# `l2_norm_bwd` — strict per-kernel correctness

`_l2_norm_bwd_kernel` is the backward pass of L2 normalization: program `row`
loads its row of input `X` and upstream gradient `DY` (length `N`, masked into a
`BLOCK_N` tile), recomputes `var`/`rstd`, and stores the input gradient
`dx = dy·rstd − (Σ dy·x)·(1/(var+eps))·rstd·x` to `DX`, masked by `cols < N`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_l2_norm_bwd_kernel[(M,)](...)`, the 1-D grid over rows,
the `BLOCK_N = min(MAX_FUSED_SIZE, next_power_of_2(N))` choice, the `N > BLOCK_N`
guard, and how the runtime composes per-program writes into one buffer) is the
*trusted boundary*, not a proof obligation here. Because `pid` (the row) is
universally quantified, the per-program statement covers every row of the grid.

## Proof architecture

```
l2_norm_bwd_kernel_output_summary             ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ l2_norm_bwd_kernel_compute_correct       ← ComputeCorrect over the masked store
       └─ l2_norm_bwd_kernel_correct          ← algorithm-layer readback per lane
            ├─ l2BwdVarCarrier_eq_l2NormSqSum  ← masked reduce-sum = oracle sum-of-squares
            └─ l2BwdDotCarrier_eq_l2NormDot    ← masked reduce-sum = oracle dot product
```

## Modeling boundary

The spec is an **oracle wrapper** over `VeriTile.Triton.Math` L2-norm definitions
(`TiledL2Norm.l2NormBwd`, built on `l2NormSqSum` / `l2NormDot` / `l2NormRstd`):
the L2-norm backward math lives once in `Math.*` and is reused here, so this file
only checks that the kernel realizes that oracle lane-wise. The two intra-kernel
reductions `tl.sum(x·x)` and `tl.sum(dy·x)` are connected to the oracle
sum-of-squares and dot product by the bridging lemmas
`l2BwdVarCarrier_eq_l2NormSqSum` and `l2BwdDotCarrier_eq_l2NormDot` (over
`Semantics.MaskedReduction`). Arithmetic is over `ℝ` (not bit-accurate IEEE
float); the masked loads `other=0.0`, the `tl.where(cols < N, ·, 0.0)` masking,
and the `.to(tl.float32)` casts all reduce to the algorithm-layer behavior
(post-erasure all dtypes unify to `ℝ`); `√`/`⁻¹` are the real square root and
inverse. Inputs are presented via the
`s.readMem`-resolved tile `l2BwdLoad`.
-/

namespace VeriTile.Bench.TritonBenchG.L2NormBwd

open VeriTile.Triton
open VeriTile.Triton.TiledL2Norm

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `l2_norm_bwd.py`'s `_l2_norm_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` -> Lean `Nat` parameter. -/
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

noncomputable def l2BwdInputTile
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
noncomputable def l2BwdLoad
    (s : BlockState) (R : RegionName) (stride_x_row N BLOCK_N : Nat)
    (idx : Fin BLOCK_N) : ℝ :=
  if idx.val < N then
    s.readMem R (s.pids 0 * stride_x_row + idx.val)
  else
    0

noncomputable def l2BwdVarCarrier
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (l2BwdInputTile s X stride_x_row N BLOCK_N)
      (l2BwdInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit

theorem l2BwdVarCarrier_eq_l2NormSqSum
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    l2BwdVarCarrier s X stride_x_row N BLOCK_N =
      some (l2NormSqSum (l2BwdLoad s X stride_x_row N BLOCK_N)) := by
  simp [l2BwdVarCarrier, l2BwdInputTile, Tile.reduceSum,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop, NumericDType.mul]
  refine (reduceSum_masked_sq_eq_some_sum
    (fun k : Fin BLOCK_N => s.readMem X (s.pids 0 * stride_x_row + k.val))
    (fun k : Fin BLOCK_N => k.val < N)).trans ?_
  congr 1
  unfold l2NormSqSum l2BwdLoad
  apply Finset.sum_congr rfl
  intro k _hk
  by_cases h : k.val < N <;> simp [h]

noncomputable def l2BwdDotCarrier
    (s : BlockState) (X DY : RegionName) (stride_x_row N BLOCK_N : Nat) : WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (l2BwdInputTile s DY stride_x_row N BLOCK_N)
      (l2BwdInputTile s X stride_x_row N BLOCK_N))).data PUnit.unit

theorem l2BwdDotCarrier_eq_l2NormDot
    (s : BlockState) (X DY : RegionName) (stride_x_row N BLOCK_N : Nat) :
    l2BwdDotCarrier s X DY stride_x_row N BLOCK_N =
      some (l2NormDot
        (l2BwdLoad s DY stride_x_row N BLOCK_N)
        (l2BwdLoad s X stride_x_row N BLOCK_N)) := by
  simp [l2BwdDotCarrier, l2BwdInputTile, Tile.reduceSum,
    Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop, NumericDType.mul]
  refine (reduceSum_masked_dot_eq_some_sum
    (fun k : Fin BLOCK_N => s.readMem DY (s.pids 0 * stride_x_row + k.val))
    (fun k : Fin BLOCK_N => s.readMem X (s.pids 0 * stride_x_row + k.val))
    (fun k : Fin BLOCK_N => k.val < N)).trans ?_
  congr 1
  unfold l2NormDot l2BwdLoad
  apply Finset.sum_congr rfl
  intro k _hk
  by_cases h : k.val < N <;> simp [h]

noncomputable def l2BwdSpec
    (s : BlockState) (X DY : RegionName)
    (stride_x_row N BLOCK_N : Nat) (eps : ℝ) (idx : Fin BLOCK_N) : ℝ :=
  l2NormBwd
    (l2BwdLoad s X stride_x_row N BLOCK_N)
    (l2BwdLoad s DY stride_x_row N BLOCK_N)
    eps idx

def l2BwdOutOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 0 * stride_x_row + i.val

/-- Algorithm-layer correctness for `_l2_norm_bwd_kernel`. -/
theorem l2_norm_bwd_kernel_correct
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      let outAddr := l2BwdOutOffset s stride_x_row i
      s'.readMem DX outAddr =
        if i.val < N then
          l2BwdSpec s X DY stride_x_row N BLOCK_N eps i
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
    simp only [l2BwdOutOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · have hvar := l2BwdVarCarrier_eq_l2NormSqSum s X stride_x_row N BLOCK_N
      have hdot := l2BwdDotCarrier_eq_l2NormDot s X DY stride_x_row N BLOCK_N
      simp [l2BwdVarCarrier, l2BwdInputTile, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.bop, NumericDType.mul] at hvar
      simp [l2BwdDotCarrier, l2BwdInputTile, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.bop, NumericDType.mul] at hdot
      simp [hi, l2BwdSpec, WithBot.realSqrt, l2BwdLoad, l2NormBwd, l2NormRstd]
      erw [hvar, hdot]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for `_l2_norm_bwd_kernel`. -/
theorem l2_norm_bwd_kernel_compute_correct
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, l2BwdOutOffset s stride_x_row i)))
      (expected := fun i => l2BwdSpec s X DY stride_x_row N BLOCK_N eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [l2_norm_bwd_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := l2_norm_bwd_kernel_correct X DY DX stride_x_row N eps BLOCK_N s s' hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `_l2_norm_bwd_kernel`: the DSL surface lowers to
the algorithm layer, and the masked store to `DX` is compute-correct — every
active lane holds `l2BwdSpec` (the oracle L2-norm backward value), out-of-bounds
lanes are preserved. -/
theorem l2_norm_bwd_kernel_output_summary
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s : BlockState) :
    (∃ alg, (l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, l2BwdOutOffset s stride_x_row i)))
      (expected := fun i => l2BwdSpec s X DY stride_x_row N BLOCK_N eps i) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact l2_norm_bwd_kernel_compute_correct X DY DX stride_x_row N eps BLOCK_N s

end VeriTile.Bench.TritonBenchG.L2NormBwd
