import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Math.Activation
import VeriTile.Examples.Common

/-!
# `relu_triton_kernel` — strict per-kernel correctness

`relu_kernel` is an elementwise ReLU: program `pid` loads block
`[pid·block_size, (pid+1)·block_size)` of `x_ptr`, computes
`tl.where(x >= 0, x, 0.0)` lane-wise, and — *only when `pid == 0`* — stores the
result to `out_ptr`, masked by `offsets < N`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`relu_kernel[grid](...)`, the grid size
`cdiv(N, BLOCK_SIZE)`, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here. Because `pid` is
universally quantified, the per-program statement covers every program of the
grid — including the `pid == 0` write-gate, which is part of the verified body.

## Proof architecture

```
relu_kernel_output_summary                    ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)             surface lowers to the algorithm layer
  └─ relu_kernel_compute_correct              ← ComputeCorrect over the masked store
       └─ relu_kernel_correct                 ← algorithm-layer readback per lane
```

The spec is the reusable `TiledActivation.relu` oracle applied to the values
this lane loads. Inputs are presented via `InputLoadedAt`. The store is gated by
`pid == 0`, so the write predicate is `s.pid = 0 ∧ addr < N`; programs with
`pid ≠ 0` leave the observed output cells untouched.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. No output/input disjointness is assumed: the input is read into
registers before the (gated) scatter, so the result is correct even if `out_ptr`
aliases `x_ptr`.
-/

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel

open VeriTile VeriTile.Examples

/-- Faithful 1:1 transcription of `relu_triton_kernel.py`'s `relu_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `N: tl.constexpr` / `block_size: tl.constexpr` → Lean `Nat`.
- Python `if cond: body` → `if cond { body }`, the DSL-side gate (block syntax
  uses braces because Lean is whitespace-insensitive; the `if` keyword itself
  is shared). -/
def relu_kernel
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = pid * $(block_size)
  offsets = block_start + tl.arange(0, $(block_size))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  result = tl.where(x >= 0, x, 0.0)
  if pid == 0 {
    tl.store(out_ptr + offsets, result, mask=mask)
  }
}

/-- Algorithm-layer correctness for `relu_kernel`.

Only `pid = 0` writes. In that case, active lanes write the conditional-form ReLU
and inactive tail lanes are preserved; nonzero pids preserve the observed
output cells. -/
theorem relu_kernel_correct
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) (_hBlockSize : 0 < block_size)
    (s : BlockState) (xs : Fin block_size → ℝ)
    (h_x : InputLoadedAt s x_ptr block_size xs) :
    ∀ i : Fin block_size,
      let addr := s.pid * block_size + i.val
      observeAt (exec (relu_kernel x_ptr out_ptr N block_size) s)
          out_ptr block_size s.pid i
        = some (if s.pid = 0 ∧ addr < N then TiledActivation.relu (xs i)
                else s.readMem out_ptr addr) := by
  intro i
  by_cases hpid : s.pid = 0
  · simp [observeAt, exec, relu_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid]
    unfold InputLoadedAt at h_x
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          BlockState.tileIndex1d_offset_injective (i, PUnit.unit)]
    have hx0 : s.readMem x_ptr i.val = xs i := by
      simpa [hpid] using h_x i
    by_cases hi : i.val < N
    · simp [hi, hx0, TiledActivation.relu]
      split_ifs with hge
      · -- hge : (some 0 : WithBot ℝ) ≤ some (xs i)
        have h : (0 : ℝ) ≤ xs i :=
          WithBot.coe_le_coe.mp (hge : (↑(0 : ℝ) : WithBot ℝ) ≤ ↑(xs i))
        change xs i = max 0 (xs i)
        exact (max_eq_right h).symm
      · have h : ¬ ((0 : ℝ) ≤ xs i) := fun h' =>
          hge (WithBot.coe_le_coe.mpr h' :
            (↑(0 : ℝ) : WithBot ℝ) ≤ ↑(xs i))
        rw [not_le] at h
        change (0.0 : ℝ) = max 0 (xs i)
        rw [max_eq_left h.le]
        norm_num
    · simp [hi]
  · simp [observeAt, exec, relu_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.select, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, ComparableDType.eq, ComparableDType.ge, hpid]

/-- Compute-facing correctness for `relu_kernel`. -/
theorem relu_kernel_compute_correct
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) (hBlockSize : 0 < block_size)
    (s : BlockState) (xs : Fin block_size → ℝ)
    (h_x : InputLoadedAt s x_ptr block_size xs) :
    ComputeCorrect.Realizes
      (kernel := relu_kernel x_ptr out_ptr N block_size)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin block_size => s.pid = 0 ∧ s.pid * block_size + i.val < N)
          (fun i => (out_ptr, s.pid * block_size + i.val)))
      (expected := fun i => TiledActivation.relu (xs i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have hi := relu_kernel_correct x_ptr out_ptr N block_size hBlockSize s xs h_x i
  rw [hExec] at hi
  have hlt : i.val < N := by
    simpa [hActive.1] using hActive.2
  simpa [observeAt, hActive.1, hlt] using hi

/-- Per-kernel output summary for `relu_kernel`: the DSL surface lowers to the
algorithm layer, and the `pid == 0`-gated masked store to `out_ptr` is
compute-correct — every active lane (with `pid = 0` and in bounds) holds
`TiledActivation.relu (xs i)`, all other observed cells are preserved. -/
theorem relu_kernel_output_summary
    (x_ptr out_ptr : RegionName)
    (N block_size : Nat) (hBlockSize : 0 < block_size)
    (s : BlockState) (xs : Fin block_size → ℝ)
    (h_x : InputLoadedAt s x_ptr block_size xs) :
    (∃ alg, (relu_kernel x_ptr out_ptr N block_size).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := relu_kernel x_ptr out_ptr N block_size)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin block_size => s.pid = 0 ∧ s.pid * block_size + i.val < N)
          (fun i => (out_ptr, s.pid * block_size + i.val)))
      (expected := fun i => TiledActivation.relu (xs i)) := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact relu_kernel_compute_correct x_ptr out_ptr N block_size hBlockSize s xs h_x

end VeriTile.Bench.TritonBenchG.ReluTritonKernel
