import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.Math.RMSNorm
import VeriTile.Triton.DSL

/-!
# `rmsnorm_fused` — strict per-kernel correctness

`rms_norm_fwd_fused` is a fused RMSNorm forward: each program `row` normalizes
one row of `X` by its root-mean-square, scales by per-column weights `W`, and
writes `Y[row] = (x / sqrt(mean(x²) + eps)) * w`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, for one program (one row). The host launch
(`rms_norm_fwd_fused[(M,)](...)`, the grid over rows `M`, the host-side
`BLOCK_SIZE` choice, scheduling, and how the runtime composes per-row writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`row = tl.program_id(0)` is universally quantified (via `s.pid`), the per-program
statement covers every row of the grid.

## Proof architecture

```
rms_norm_fwd_fused_output_summary             ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ rms_norm_fwd_fused_compute_correct       ← ComputeCorrect over the masked store
       └─ rms_norm_fwd_fused_correct          ← algorithm-layer readback per lane
            └─ rmsnormCarrierSpec_eq_rmsnormSpec
                 └─ rmsVarCarrier_eq_rmsMeanSq (uses VeriTile.Triton.Math.RMSNorm /
                                                TiledL2Norm reduction lemmas)
```

The row spec is `TiledRMSNorm.rmsAffine` from `VeriTile.Triton.Math.RMSNorm`
(reusing `TiledRMSNorm.rmsMeanSq` / `TiledL2Norm` reduction lemmas rather than
inlining the reduction math).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)`
casts reduce to identity at the algorithm layer (post-erasure all dtypes unify
to `ℝ`). The reduction `tl.sum(_var) / N` sums over the *padded* `BLOCK_SIZE`
block, but out-of-range lanes are masked to `0` (load `other=0.0` plus the
explicit `tl.where`), so the sum equals the logical row length `N`. The
reciprocal std is `rstd = 1 / sqrt(meanSq + eps)`; the affine step is
`x_hat * w`. The Python wrapper picks `BLOCK_SIZE ≥ N` (raising otherwise), so
correctness is stated under the `0 < N ≤ BLOCK_SIZE` precondition, where both
`range(0, N, BLOCK_SIZE)` loops execute exactly the `off = 0` iteration (single
block).
-/

namespace VeriTile.Bench.TritonBenchG.RmsnormFused

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful `forRange` transcription of `rmsnorm_fused.py`'s
`rms_norm_fwd_fused`.

The Python wrapper chooses `BLOCK_SIZE >= N` and raises otherwise, so the
correctness theorem below proves the full loop-shaped kernel under that
runtime precondition. Under the precondition both `range(0, N, BLOCK_SIZE)`
loops execute exactly the `off = 0` iteration.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameters. -/
def rms_norm_fwd_fused
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    x = tl.where(cols < $(N), x, 0.0)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, y, mask=mask)
  }
}

def xOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

def yOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  Tile.maskedRowTile
    (fun i : Fin BLOCK_SIZE =>
      Tile.maskedRowLoad
        (fun k : Fin BLOCK_SIZE => some (s.readMem X (xOffset s stride k)))
        (fun k : Fin BLOCK_SIZE => k.val < N)
        (some (0.0 : ℝ) : WithBot ℝ)
        i)
    (fun i : Fin BLOCK_SIZE => i.val < N)
    (some (0.0 : ℝ) : WithBot ℝ)

noncomputable def rmsLoad
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat)
    (i : Fin BLOCK_SIZE) : ℝ :=
  Tile.maskedRowLoad
    (fun k : Fin BLOCK_SIZE => s.readMem X (xOffset s stride k))
    (fun k : Fin BLOCK_SIZE => k.val < N)
    0
    i

noncomputable def rmsWeight
    (s : BlockState) (W : RegionName) (i : Fin BLOCK_SIZE) : ℝ :=
  s.readMem W i.val

noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride N BLOCK_SIZE)
        (rmsInputTile s X stride N BLOCK_SIZE))).data PUnit.unit)
    ((Tile.scalar (dtype := .real) (some (N : ℝ) : WithBot ℝ)).data PUnit.unit)

theorem rmsVarCarrier_eq_rmsMeanSq
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    rmsVarCarrier s X stride N BLOCK_SIZE =
      some (TiledRMSNorm.rmsMeanSq (rmsLoad s X stride N BLOCK_SIZE) N) := by
  simp [rmsVarCarrier, rmsInputTile, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    Tile.bop, NumericDType.mul]
  have hsum := TiledL2Norm.reduceSum_masked_sq_eq_some_sum
    (fun k : Fin BLOCK_SIZE => s.readMem X (xOffset s stride k))
    (fun k : Fin BLOCK_SIZE => k.val < N)
  refine (congrArg
    (fun a : WithBot ℝ => Option.map (fun x : ℝ => x / (N : ℝ)) a)
    hsum).trans ?_
  simp [TiledRMSNorm.rmsMeanSq, TiledL2Norm.l2NormSqSum, rmsLoad,
    Tile.maskedRowLoad]

noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride N BLOCK_SIZE)))

noncomputable def rmsnormCarrierSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem X (xOffset s stride i)))
        (rmsInvCarrier s X stride N BLOCK_SIZE eps))
      (some (s.readMem W i.val)))

noncomputable def rmsnormSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  TiledRMSNorm.rmsAffine
    (rmsLoad s X stride N BLOCK_SIZE)
    (rmsWeight s W)
    N eps i

theorem rmsnormCarrierSpec_eq_rmsnormSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE)
    (hi : i.val < N) :
    rmsnormCarrierSpec s X W stride N BLOCK_SIZE eps i =
      rmsnormSpec s X W stride N BLOCK_SIZE eps i := by
  unfold rmsnormCarrierSpec rmsnormSpec rmsInvCarrier
  rw [rmsVarCarrier_eq_rmsMeanSq]
  simp [TiledRMSNorm.rmsAffine, TiledRMSNorm.rmsRstd,
    rmsLoad, rmsWeight, hi]

/-- Algorithm-layer correctness for the fused RMSNorm kernel under the Python
wrapper's `N <= BLOCK_SIZE` launch precondition. -/
theorem rms_norm_fwd_fused_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i))
    (hExec : exec (rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps) s =
        some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem Y (yOffset s stride i) =
        if i.val < N then
          rmsnormSpec s X W stride N BLOCK_SIZE eps i
        else s.readMem Y (yOffset s stride i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · have hStep : BLOCK_SIZE ≠ 0 := Nat.ne_of_gt hB
    simp [exec, rms_norm_fwd_fused, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepForRangeAux.step_lt, stepForRangeAux.step_ge,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, hNpos, hNle,
          Nat.not_lt.mpr hNle, hStep] at hExec
    subst s'
    simp only [yOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · rw [← rmsnormCarrierSpec_eq_rmsnormSpec s X W stride N BLOCK_SIZE eps i hi]
      simp only [hi, ↓reduceIte]
      simp [hi, rmsnormCarrierSpec, rmsInvCarrier, rmsVarCarrier, rmsInputTile,
            xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul, FloatDType.cast]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the fused RMSNorm kernel under the Python
wrapper's `N <= BLOCK_SIZE` launch precondition. -/
theorem rms_norm_fwd_fused_compute_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    ComputeCorrect.Realizes
      (kernel := rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i => rmsnormSpec s X W stride N BLOCK_SIZE eps i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_norm_fwd_fused, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rms_norm_fwd_fused_correct X Y W stride N BLOCK_SIZE eps
    s s' hNpos hNle hOutInj hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `rms_norm_fwd_fused`: the DSL surface lowers to
the algorithm layer, and the masked store to `Y` is compute-correct — every
active lane (`i.val < N`) holds the RMSNorm spec `rmsnormSpec`, out-of-bounds
lanes are preserved. Stated under the `0 < N ≤ BLOCK_SIZE` single-block launch
precondition chosen by the Python wrapper. -/
theorem rms_norm_fwd_fused_output_summary
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    (∃ alg, (rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := rms_norm_fwd_fused X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i => rmsnormSpec s X W stride N BLOCK_SIZE eps i) := by
  refine ⟨?_, ?_⟩
  · simp only [rms_norm_fwd_fused, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    exact ⟨_, rfl⟩
  · exact rms_norm_fwd_fused_compute_correct X Y W stride N BLOCK_SIZE eps
      s hNpos hNle hOutInj

end VeriTile.Bench.TritonBenchG.RmsnormFused
