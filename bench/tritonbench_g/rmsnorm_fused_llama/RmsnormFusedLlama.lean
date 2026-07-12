import VeriTile.Triton

/-!
# `rmsnorm_fused_llama` — strict per-kernel correctness

`_rms_norm_fwd_fused` is the Llama-style fused RMSNorm forward: each program
`row` normalizes one row of `X` by its root-mean-square, scales by per-column
weights `W`, casts the result to float16, and writes
`Y[row] = fp16((x / sqrt(mean(x²) + eps)) * w)`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, for one program (one row). The host launch
(`_rms_norm_fwd_fused[(M,)](...)`, the grid over rows `M`, the host-fixed
`BLOCK_SIZE = 16384`, scheduling, and how the runtime composes per-row writes
into one buffer) is the *trusted boundary*, not a proof obligation here. Because
`row = tl.program_id(0)` is universally quantified (via `s.pid`), the per-program
statement covers every row of the grid.

## Proof architecture

```
rms_norm_fwd_fused_llama_output_summary       ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ rms_norm_fwd_fused_llama_compute_correct ← ComputeCorrect over the masked fp16 store
       └─ rms_norm_fwd_fused_llama_correct    ← algorithm-layer readback per lane
            └─ scatter_memcell_fp16_prop_masked_nd  (fp16 masked-scatter readback)
                 └─ foldl_writeMemTyped_fp16_preserve_masked
```

The RMSNorm row math (`rmsInputTile`, `rmsVarCarrier`, `rmsInvCarrier`,
`rmsnormCarrierSpec` / `rmsnormSpec`) is defined inline in this file rather than
reusing `VeriTile.Triton.Math.RMSNorm`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the `.to(tl.float32)`
input/weight casts reduce to identity at the algorithm layer. The **store cast
to float16 is modeled explicitly**: the spec writes the `MemCell.of .fp16`
obtained by `FloatDType.real.cast FloatDType.fp16`, so the fp16 rounding of the
final value is part of the proved statement (not erased). The reduction
`tl.sum(_var) / N` sums over the *padded* `BLOCK_SIZE` block, but out-of-range
lanes are masked to `0` (load `other=0.0`), so the sum equals the logical row
length `N`. The reciprocal std is `rstd = 1 / sqrt(meanSq + eps)`; the affine
step is `x_hat * w`. Correctness is stated under the `0 < N ≤ BLOCK_SIZE`
single-block precondition (Python fixes `BLOCK_SIZE = 16384` after checking
`N ≤ BLOCK_SIZE`), where both `range(0, N, BLOCK_SIZE)` loops execute exactly the
`off = 0` iteration. `@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful `forRange` transcription of `rmsnorm_fused_llama.py`'s
`_rms_norm_fwd_fused`.

The Python wrapper fixes `BLOCK_SIZE = 16384` after checking `N <= BLOCK_SIZE`,
so the correctness theorem below proves the loop-shaped kernel under that
runtime precondition. Under the precondition both `range(0, N, BLOCK_SIZE)`
loops execute exactly the `off = 0` iteration.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameters. -/
def rms_norm_fwd_fused_llama
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(tl.float16), mask=mask)
  }
}

def xOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

def yOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val

noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride idx.1))
      else some (0.0 : ℝ) }

noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride N BLOCK_SIZE)
        (rmsInputTile s X stride N BLOCK_SIZE))).data PUnit.unit)
    ((Tile.scalar (dtype := .real) (some (N : ℝ) : WithBot ℝ)).data PUnit.unit)

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
  rmsnormCarrierSpec s X W stride N BLOCK_SIZE eps i

/-- Algorithm-layer correctness for the Llama RMSNorm kernel under the Python
wrapper's `N <= BLOCK_SIZE` launch precondition. -/
theorem rms_norm_fwd_fused_llama_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s s' : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i))
    (hExec : exec (rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps)
        s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.mem Y (yOffset s stride i) =
        if i.val < N then
          MemCell.of .fp16
            (FloatDType.real.cast FloatDType.fp16
              (some (rmsnormSpec s X W stride N BLOCK_SIZE eps i)))
        else s.mem Y (yOffset s stride i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] => s.pids 0 * stride + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [yOffset] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hB : 0 < BLOCK_SIZE
  · have hStep : BLOCK_SIZE ≠ 0 := Nat.ne_of_gt hB
    simp [exec, rms_norm_fwd_fused_llama, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          stepForRangeAux.step_lt, stepForRangeAux.step_ge,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
          Tile.reduceSumDrop, Tile.select, TileShape.axisDim, TileShape.eraseAxis,
          TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
          NumericDType.div, ComparableDType.lt, hNpos, hNle,
          Nat.not_lt.mpr hNle, hStep] at hExec
    subst s'
    simp only [yOffset]
    rw [scatter_memcell_fp16_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
    by_cases hi : i.val < N
    · simp only [hi, ↓reduceIte]
      simp [hi, rmsnormSpec, rmsnormCarrierSpec, rmsInvCarrier, rmsVarCarrier,
            rmsInputTile, xOffset, Tile.reduceSum, Tile.reduceSumDrop, Tile.select,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            WithBot.realSqrt, NumericDType.mul, FloatDType.cast]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Llama RMSNorm kernel under the Python
wrapper's `N <= BLOCK_SIZE` launch precondition. -/
theorem rms_norm_fwd_fused_llama_compute_correct
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsnormSpec s X W stride N BLOCK_SIZE eps i)))) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rms_norm_fwd_fused_llama, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rms_norm_fwd_fused_llama_correct X Y W stride N BLOCK_SIZE eps
    s s' hNpos hNle hOutInj hExec i
  simpa [hActive] using h

/-- Per-kernel output summary for `_rms_norm_fwd_fused` (Llama): the DSL surface
lowers to the algorithm layer, and the masked fp16 store to `Y` is
compute-correct — every active lane (`i.val < N`) holds the fp16-cast RMSNorm
spec, out-of-bounds lanes are preserved. Stated under the `0 < N ≤ BLOCK_SIZE`
single-block launch precondition chosen by the Python wrapper. -/
specification rms_norm_fwd_fused_llama_output_summary
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    (∃ alg, (rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsnormSpec s X W stride N BLOCK_SIZE eps i)))) := by
  refine ⟨?_, ?_⟩
  · simp only [rms_norm_fwd_fused_llama, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
    exact ⟨_, rfl⟩
  · exact rms_norm_fwd_fused_llama_compute_correct X Y W stride N BLOCK_SIZE eps
      s hNpos hNle hOutInj

end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama
