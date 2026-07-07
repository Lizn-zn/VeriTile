import VeriTile.Triton

/-!
# `adam_update_triton` — strict per-kernel correctness

The Python kernel is named "adam" but implements **Lion** (Chen et al.): a
sign-based parameter update driven by a single momentum buffer, with decoupled
weight decay. The `@triton.jit` function body is an SPMD program: program `pid`
updates the block `[pid·BLOCK_SIZE, (pid+1)·BLOCK_SIZE)` of two buffers —
`p_ptr` (parameters) and `exp_avg_ptr` (momentum) — masked to `< n_elements`.

## Scope

This file verifies **the Triton kernel itself** — the per-program body — and
nothing on the host side. The verification target is exactly the artifact the
user wrote (`@triton.jit def update_fn_kernel`); the launch `kernel[grid](...)`,
the grid size `triton.cdiv(...)`, and how the runtime composes per-program
memories into one buffer are the *trusted boundary*, not proof obligations here.
(A whole-grid / launch-composition treatment lives separately as the worked
example `VeriTile.Examples.AdamUpdateGridLaunch`.)

## Proof architecture

```
update_fn_kernel_output_summary                  ← TOP THEOREM (this file)
  ├─ update_fn_kernel_surface_toAlgorithm_supported   (the surface lowers)
  └─ update_fn_kernel_all_outputs_compute_correct
       ├─ update_fn_kernel_p_compute_correct
       │    └─ update_fn_kernel_p_correct          ← p_ptr store realizes pFullSpec
       └─ update_fn_kernel_exp_avg_compute_correct
            └─ update_fn_kernel_exp_avg_correct    ← exp_avg_ptr store realizes expAvgFullSpec
```

For an arbitrary program (`s.pid` is universally quantified), each masked store
writes, at offset `linearOffset s BLOCK_SIZE i = pid·BLOCK_SIZE + i`, the value
given by the spec — and leaves out-of-range lanes (`≥ n_elements`) untouched.
Because `pid` is arbitrary, this covers every program of the grid.

The specs `pFullSpec` / `expAvgFullSpec` are **oracle wrappers**: they apply the
layout-free Lion oracle (`VeriTile.Triton.Math.Optimizer.lionParam` /
`lionMomentum`) to the values this lane loads, so the optimizer math is stated
once, in `Math.Optimizer`, and never re-derived here.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. Side condition: `p_ptr ≠ exp_avg_ptr` (the two output buffers do not
alias — no kernel caller relies on aliasing them).
-/

namespace VeriTile.Bench.TritonBenchG.AdamUpdateTriton

open VeriTile.Triton

set_option maxHeartbeats 5000000

/-- Faithful 1:1 transcription of `adam_update_triton.py`'s `update_fn_kernel`. -/
def update_fn_kernel
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  offset_p_ptr = p_ptr + offsets
  offset_grad_ptr = grad_ptr + offsets
  offset_exp_avg_ptr = exp_avg_ptr + offsets
  p = tl.load(offset_p_ptr, mask=mask)
  grad = tl.load(offset_grad_ptr, mask=mask)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
  p = p * (1 - $((lr : ℝ)) * $((wd : ℝ)))
  diff = exp_avg - grad
  update = diff * $(beta1) + grad
  can_update = update != 0
  update_sign = tl.where(update > 0, -$((lr : ℝ)), $((lr : ℝ)))
  p = p + update_sign * can_update
  exp_avg = diff * $(beta2) + grad
  tl.store(offset_p_ptr, p, mask=mask)
  tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
}

/-- The full Adam update surface lowers to the algorithm layer, including both
masked stores to `p_ptr` and `exp_avg_ptr`. -/
theorem update_fn_kernel_surface_toAlgorithm_supported
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ∃ alg, (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg := by
  simp [update_fn_kernel, ComputeExpr.toAlgorithm?]

/-! ## Full-kernel output correctness

The Python test runs the full `update_fn_kernel` which writes both `p` and
`exp_avg`. Per #139's audit, slice proofs are insufficient. This theorem
characterizes the observable stores for the full kernel under the assumption
that `p_ptr ≠ exp_avg_ptr` (no kernel callers rely on aliasing those
buffers). -/

/-- Per-lane `exp_avg` output spec: the reusable Lion momentum oracle applied
to the values this lane loads. The math lives once in `Math.Optimizer`; this
only names the memory reads. -/
noncomputable def expAvgFullSpec
    (s : BlockState) (grad_ptr exp_avg_ptr : RegionName)
    (beta2 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  TiledOptimizer.lionMomentum
    (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
    (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) beta2

/-- Per-lane `p` output spec: the reusable Lion parameter-update oracle applied
to the values this lane loads. -/
noncomputable def pFullSpec
    (s : BlockState) (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 : ℝ) (BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  TiledOptimizer.lionParam
    (s.readMem p_ptr (linearOffset s BLOCK_SIZE i))
    (s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i))
    (s.readMem grad_ptr (linearOffset s BLOCK_SIZE i)) lr wd beta1

/-- Algorithm-layer correctness for the `exp_avg` store of the full
`update_fn_kernel`. -/
theorem update_fn_kernel_exp_avg_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hRegions : exp_avg_ptr ≠ p_ptr)
    (hExec : exec (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i) =
        if linearOffset s BLOCK_SIZE i < n_elements then
          expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i
        else s.readMem exp_avg_ptr (linearOffset s BLOCK_SIZE i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, ComparableDType.gt, ComparableDType.ne] at hExec
    subst s'
    simp only [linearOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          p_ptr _ _ _ _ _ _ _ hRegions]
    by_cases hi : s.pids 0 * BLOCK_SIZE + i.val < n_elements
    · simp only [hi, if_true, expAvgFullSpec, TiledOptimizer.lionMomentum,
        linearOffset, BlockState.pid_eq, Option.map₂, Option.map, Option.bind,
        WithBot.some_eq_coe, WithBot.unbotD_coe]
      ring
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the `exp_avg` store of the full
`update_fn_kernel`. -/
theorem update_fn_kernel_exp_avg_compute_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : exp_avg_ptr ≠ p_ptr) :
    ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i => expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [update_fn_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := update_fn_kernel_exp_avg_correct p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE s s' hRegions hExec i
  simpa [hActive] using h

/-- Algorithm-layer correctness for the `p` store of the full
`update_fn_kernel`. -/
theorem update_fn_kernel_p_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hRegions : p_ptr ≠ exp_avg_ptr)
    (hExec : exec (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      s'.readMem p_ptr (linearOffset s BLOCK_SIZE i) =
        if linearOffset s BLOCK_SIZE i < n_elements then
          pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i
        else s.readMem p_ptr (linearOffset s BLOCK_SIZE i) := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, update_fn_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, ComparableDType.gt, ComparableDType.ne] at hExec
    subst s'
    rw [BlockState.foldl_writeMem_const_region_prop_masked_readMem_other
          exp_avg_ptr _ _ _ _ _ _ _ hRegions]
    simp only [linearOffset]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : s.pids 0 * BLOCK_SIZE + i.val < n_elements
    · -- The realized masked `p` store equals the Lion parameter-update oracle.
      unfold pFullSpec TiledOptimizer.lionParam TiledOptimizer.lionUpdateDir
      simp only [linearOffset, hi, if_true, BlockState.pid_eq,
        Option.map₂, Option.map, Option.bind]
      set m := s.readMem exp_avg_ptr (s.pids 0 * BLOCK_SIZE + i.val) with hm
      set g := s.readMem grad_ptr (s.pids 0 * BLOCK_SIZE + i.val) with hg
      set p := s.readMem p_ptr (s.pids 0 * BLOCK_SIZE + i.val) with hp
      have heq : (m - g) * beta1 + g = beta1 * m + (1 - beta1) * g := by ring
      split_ifs with c1 c2
      · have hd : beta1 * m + (1 - beta1) * g = 0 := heq ▸ Option.some_inj.mp c1
        simp only [WithBot.some_eq_coe, WithBot.unbotD_coe, hd, Real.sign_zero]
        ring
      · have hd : (0 : ℝ) < beta1 * m + (1 - beta1) * g :=
          heq ▸ WithBot.coe_lt_coe.mp c2
        simp only [WithBot.some_eq_coe, WithBot.unbotD_coe, Real.sign_of_pos hd]
        norm_num
        ring
      · have hne : (m - g) * beta1 + g ≠ 0 := fun hc => c1 (congrArg some hc)
        have hnp : ¬ (0 : ℝ) < (m - g) * beta1 + g :=
          fun hc => c2 (WithBot.coe_lt_coe.mpr hc)
        have hd : beta1 * m + (1 - beta1) * g < 0 :=
          heq ▸ lt_of_le_of_ne (not_lt.mp hnp) hne
        simp only [WithBot.some_eq_coe, WithBot.unbotD_coe, Real.sign_of_neg hd]
        ring
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the `p` store of the full
`update_fn_kernel`. -/
theorem update_fn_kernel_p_compute_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : p_ptr ≠ exp_avg_ptr) :
    ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (p_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [update_fn_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := update_fn_kernel_p_correct p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE s s' hRegions hExec i
  simpa [hActive] using h

/-- Full compute-facing output coverage for Python `update_fn`: the parameter
buffer `p_ptr` and momentum buffer `exp_avg_ptr` stores are both characterized
for the full `update_fn_kernel`. -/
theorem update_fn_kernel_all_outputs_compute_correct
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : p_ptr ≠ exp_avg_ptr) :
    (ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (p_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i)) ∧
    (ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i)) := by
  constructor
  · exact update_fn_kernel_p_compute_correct p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE s hRegions
  · exact update_fn_kernel_exp_avg_compute_correct p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE s (fun h => hRegions h.symm)

/-- Single-program output summary for Python `update_fn`.

This is a local, per-`BlockState` correctness statement, not the whole-grid
launch theorem.  It says three things about executing one Triton program/block:

* the DSL surface for `update_fn_kernel` successfully lowers to the algorithm
  layer (`toAlgorithm? = Except.ok alg`);
* for every active lane `i : Fin BLOCK_SIZE`, where
  `linearOffset s BLOCK_SIZE i < n_elements` is the kernel mask
  `offsets < n_elements`, the store to
  `(p_ptr, linearOffset s BLOCK_SIZE i)` realizes `pFullSpec`, i.e. the reusable
  Lion parameter-update oracle applied to the values loaded by that lane;
* for the same active lanes, the store to
  `(exp_avg_ptr, linearOffset s BLOCK_SIZE i)` realizes `expAvgFullSpec`, i.e.
  the reusable Lion momentum oracle.

The side condition `p_ptr ≠ exp_avg_ptr` rules out aliasing between the two
output regions, so the second masked store cannot overwrite the first output.
Grid coverage, `cdiv n_elements BLOCK_SIZE`, and cross-program disjointness are
separate whole-grid obligations handled by the worked example
`VeriTile.Examples.AdamUpdateGridLaunch`, outside the scope of this per-kernel
theorem. -/
theorem update_fn_kernel_output_summary
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (s : BlockState)
    (hRegions : p_ptr ≠ exp_avg_ptr) :
    (∃ alg, (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgorithm? =
        Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (p_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        pFullSpec s p_ptr grad_ptr exp_avg_ptr lr wd beta1 BLOCK_SIZE i)) ∧
    (ComputeCorrect.Realizes
      (kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => linearOffset s BLOCK_SIZE i < n_elements)
        (fun i => (exp_avg_ptr, linearOffset s BLOCK_SIZE i)))
      (expected := fun i =>
        expAvgFullSpec s grad_ptr exp_avg_ptr beta2 BLOCK_SIZE i))) := by
  constructor
  · exact update_fn_kernel_surface_toAlgorithm_supported p_ptr grad_ptr
      exp_avg_ptr lr wd beta1 beta2 n_elements BLOCK_SIZE
  · exact update_fn_kernel_all_outputs_compute_correct p_ptr grad_ptr
      exp_avg_ptr lr wd beta1 beta2 n_elements BLOCK_SIZE s hRegions

end VeriTile.Bench.TritonBenchG.AdamUpdateTriton
