import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

/-!
# `mul_exponent_compensator` — strict per-kernel correctness

`mul_kernel` scales a vector by a constant: program `pid` loads block
`[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of `src` (no mask), multiplies each lane
by the compile-time constant `exponent_compensator = 2.0 ** (127 - 15)`, and
stores the result to `dst`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`mul_kernel[(shape[0] // BLOCK_SIZE,)](...)`, the grid
size, and how the runtime composes per-program writes into one buffer) is the
*trusted boundary*, not a proof obligation here. Because `pid` is universally
quantified (as `s.pid`), the per-program statement covers every program of the
grid. Note the kernel uses an unmasked load/store, so the host is trusted to
supply a buffer that is an exact multiple of `BLOCK_SIZE`.

## Proof architecture

```
mul_kernel_output_summary                     ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ mul_kernel_compute_correct               ← ComputeCorrect over the unmasked store
       └─ mul_kernel_correct                  ← algorithm-layer readback per lane
```

The spec is plain elementwise `xs i * exponentCompensator` — no
optimizer/reduction oracle applies. Inputs are presented via `InputLoadedAt`
(the values each lane loads).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The Python compile-time literal `2.0 ** (127 - 15)` is modeled by the
noncomputable real constant `exponentCompensator` and injected as a real scalar
antiquote — its bit pattern (a float32 exponent-bias compensator) is not
otherwise interpreted. No output/input disjointness is assumed: the input is
read into registers before the scatter, so the result is correct even if `dst`
aliases `src`.
-/

namespace VeriTile.Bench.TritonBenchG.MulExponentCompensator

open VeriTile.Triton VeriTile.Examples

/-- The constexpr multiplier from `mul_exponent_compensator.py`. -/
noncomputable def exponentCompensator : ℝ :=
  (2 : ℝ) ^ (127 - 15)

/-- Faithful 1:1 transcription of `mul_exponent_compensator.py`'s
`mul_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python local constexpr literal `2.0 ** (127 - 15)` is represented by the
  Lean constant `exponentCompensator` and injected as a real scalar antiquote.
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
noncomputable def mul_kernel
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  exponent_compensator = $((exponentCompensator : ℝ))
  idxs = tl.program_id(0) * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(src + idxs)
  y = x * exponent_compensator
  tl.store(dst + idxs, y)
}

/-- Algorithm-layer correctness for `mul_kernel`. -/
theorem mul_kernel_correct
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) (_hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s src BLOCK_SIZE xs) :
    ∀ i : Fin BLOCK_SIZE,
      observeAt (exec (mul_kernel src dst BLOCK_SIZE) s)
          dst BLOCK_SIZE s.pid i
        = some (xs i * exponentCompensator) := by
  intro i
  have h_inj := injective_offset_singleton (n := BLOCK_SIZE) (s.pid * BLOCK_SIZE)
  simp [observeAt, exec, mul_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, NumericDType.add, NumericDType.mul,
        exponentCompensator]
  unfold InputLoadedAt at h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [h_x]

/-- Compute-facing correctness for `mul_kernel`. -/
theorem mul_kernel_compute_correct
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s src BLOCK_SIZE xs) :
    ComputeCorrect.Realizes
      (kernel := mul_kernel src dst BLOCK_SIZE)
      (initialState := s)
      (write := fun i : Fin BLOCK_SIZE =>
          some (dst, s.pid * BLOCK_SIZE + i.val))
      (expected := fun i => xs i * exponentCompensator) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i
  have hi := mul_kernel_correct src dst BLOCK_SIZE hBlockSize s xs h_x i
  rw [hExec] at hi
  simpa [observeAt] using hi

/-- Per-kernel output summary for `mul_kernel`: the DSL surface lowers to the
algorithm layer, and the unmasked store to `dst` is compute-correct — every lane
holds `xs i * exponentCompensator`. -/
theorem mul_kernel_output_summary
    (src dst : RegionName)
    (BLOCK_SIZE : Nat) (hBlockSize : 0 < BLOCK_SIZE)
    (s : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (h_x : InputLoadedAt s src BLOCK_SIZE xs) :
    (∃ alg, (mul_kernel src dst BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := mul_kernel src dst BLOCK_SIZE)
      (initialState := s)
      (write := fun i : Fin BLOCK_SIZE =>
          some (dst, s.pid * BLOCK_SIZE + i.val))
      (expected := fun i => xs i * exponentCompensator) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact mul_kernel_compute_correct src dst BLOCK_SIZE hBlockSize s xs h_x

end VeriTile.Bench.TritonBenchG.MulExponentCompensator
