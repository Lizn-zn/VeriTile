import VeriTile.Triton

/-!
# `max_reduction` — strict per-kernel correctness

`max_reduction.py` provides three `@triton.jit` kernels. `max_kernel_1` and
`max_kernel_2` form a two-pass global max (per-block max into `mid`, then max of
`mid` into `out`). `max_kernel` (autotuned + heuristic-blocked) is the dimension
reduction: for a `[M, N, K]` logical tensor it computes, per `(m, k)`, the
max over the `N` axis together with the argmax index via
`tl.max(..., return_indices=True)`, writing `out_value` and `out_index`.

## Scope

This file verifies **the Triton kernels themselves** — each per-program
`@triton.jit` body. The host launches (`max_kernel_1[(mid_size,1,1)]`,
`max_kernel_2[(1,1,1)]`, `max_kernel[grid]` with `grid = (cdiv(M, BLOCK_M), K)`),
the `@triton.autotune`/`@triton.heuristics` decorators that pick `BLOCK_M`/
`BLOCK_N`, and the cross-program composition of `mid`/`out`/`out_value`/
`out_index` are the *trusted boundary*, not proof obligations here. Program ids
enter only through `BlockState`, so each per-program statement covers every
program of its grid.

## Proof architecture

```
max_kernel_output_summary                       ← TOP THEOREM (the 2D value/index kernel)
  ├─ (toAlgorithm? = Except.ok _)               surface lowers to the algorithm layer
  └─ max_kernel_compute_correct                 ← ComputeCorrect.OutputPairWhere over the
       │                                          masked value/index stores
       ├─ maxKernelValueSpec  (row-wise `Tile.reduceMax`)
       └─ maxKernelIndexSpec  (row-wise `Tile.argMaxDrop`)

max_kernel_1_compute_correct  → max_kernel_1_correct  (per-block `Tile.reduceMax` → `mid`)
max_kernel_2_compute_correct  → max_kernel_2_correct  (max of `mid` → `out`)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` and
`@triton.heuristics` configure the launch grid only and have no in-kernel
transcription. `other=-float("inf")` masked lanes are modeled as `⊥`
(`WithBot`). The `max_kernel` output summary requires its two side conditions:
`hOutInj` (the per-row output offset map is injective) and `hOutRegions`
(`out_value ≠ out_index`, so the value store does not clobber the index store).
-/

namespace VeriTile.Bench.TritonBenchG.MaxReduction

open VeriTile.Triton

set_option maxHeartbeats 5000000

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel_1`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def max_kernel_1
    (inp mid : RegionName)
    (M BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  offset = pid * $(BLOCK_SIZE) + tl.arange(0, $(BLOCK_SIZE))
  inp_ptrs = inp + offset
  mask = offset < $(M)
  inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(inp_val)
  mid_ptr = mid + pid
  tl.store(mid_ptr, max_val)
}

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel_2`.

Same allowed Lean-syntax-only changes as `max_kernel_1`. -/
def max_kernel_2
    (mid out : RegionName)
    (mid_size BLOCK_MID : Nat) :
    ComputeKernel := triton {
  offset = tl.arange(0, $(BLOCK_MID))
  mid_ptrs = mid + offset
  mask = offset < $(mid_size)
  mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
  max_val = tl.max(mid_val)
  tl.store(out, max_val)
}

/-- Faithful 1:1 transcription of `max_reduction.py`'s `max_kernel`
(autotuned, returns value + index via `tl.max(..., return_indices=True)`).

Allowed mechanical Lean-syntax-only changes apply. The `@triton.autotune`
and `@triton.heuristics` decorators that wrap `max_kernel` are not DSL-side
constructs (they configure the launch grid, not the kernel body); they have
no in-kernel transcription. -/
def max_kernel
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_k = tl.program_id(1)
  m_offset = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  n_offset = tl.arange(0, $(BLOCK_N))
  offset = m_offset[:, None] * $(N) * $(K) + n_offset[None, :] * $(K) + pid_k
  offset_index = m_offset * $(K) + pid_k
  mask1 = m_offset < $(M)
  mask = m_offset[:, None] < $(M) and n_offset[None, :] < $(N)
  inp_ptrs = inp + offset
  inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
  result_value, result_index := tl.max(inp_vals, axis=1, return_indices=True)
  out_value_ptrs = out_value + offset_index
  out_index_ptrs = out_index + offset_index
  tl.store(out_value_ptrs, result_value, mask=mask1)
  tl.store(out_index_ptrs, result_index, mask=mask1)
}

/-- Masked input tile for the first-stage max reduction. -/
noncomputable def maxKernel1InputTile
    (s : BlockState) (inp : RegionName) (M BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pid * BLOCK_SIZE + idx.1.val
      if off < M then some (s.readMem inp off) else none }

/-- Exact first-stage max value written by `max_kernel_1`. -/
noncomputable def maxKernel1Spec
    (s : BlockState) (inp : RegionName) (M BLOCK_SIZE : Nat) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (maxKernel1InputTile s inp M BLOCK_SIZE) with
  | some out => WithBot.unbotD 0 (out.data PUnit.unit)
  | none => 0

/-- Masked input tile for the second-stage max reduction. -/
noncomputable def maxKernel2InputTile
    (s : BlockState) (mid : RegionName) (mid_size BLOCK_MID : Nat) :
    Tile .real [BLOCK_MID] :=
  { data := fun idx =>
      if idx.1.val < mid_size then some (s.readMem mid idx.1.val) else none }

/-- Exact second-stage max value written by `max_kernel_2`. -/
noncomputable def maxKernel2Spec
    (s : BlockState) (mid : RegionName) (mid_size BLOCK_MID : Nat) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_MID]) ⟨0, by simp⟩ Bool.false
      (maxKernel2InputTile s mid mid_size BLOCK_MID) with
  | some out => WithBot.unbotD 0 (out.data PUnit.unit)
  | none => 0

/-- Input element `inp[m, n, pid1]` of the row-major `[M, N, K]` input tensor
(this program reduces along `n` at fixed trailing index `k = pids 1`). -/
noncomputable def maxInpElem (s : BlockState) (inp : RegionName)
    (N K m n : Nat) : ℝ :=
  s.readMem inp (m * N * K + n * K + s.pids 1)

/-- Input tile for the 2D value/index max kernel. Masked lanes are `⊥`,
matching `other=-float("inf")`. -/
noncomputable def maxKernelInputTile
    (s : BlockState) (inp : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      let m := s.pids 0 * BLOCK_M + idx.1.val
      let n := idx.2.1.val
      if m < M ∧ n < N then
        some (maxInpElem s inp N K m n)
      else none }

/-- Output offset for the 2D value/index max kernel at row lane `i`. -/
def maxKernelOutOffset (s : BlockState) (K BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  (s.pids 0 * BLOCK_M + i.val) * K + s.pids 1

/-- Exact max value written by `max_kernel` at row lane `i`. -/
noncomputable def maxKernelValueSpec
    (s : BlockState) (inp : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩ Bool.false
      (maxKernelInputTile s inp M N K BLOCK_M BLOCK_N) with
  | some out => WithBot.unbotD 0 (out.data (i, PUnit.unit))
  | none => 0

/-- Exact argmax index written by `max_kernel` at row lane `i`. -/
noncomputable def maxKernelIndexSpec
    (s : BlockState) (inp : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) : Nat :=
  (Tile.argMaxDrop (shape := [BLOCK_M, BLOCK_N]) ⟨1, by simp⟩
    (maxKernelInputTile s inp M N K BLOCK_M BLOCK_N)).data (i, PUnit.unit)

/-- Algorithm-layer correctness for `max_kernel_1`. -/
theorem max_kernel_1_correct
    (inp mid : RegionName)
    (M BLOCK_SIZE : Nat) (s s' : BlockState)
    (hExec : exec (max_kernel_1 inp mid M BLOCK_SIZE) s = some s') :
    s'.readMem mid s.pid = maxKernel1Spec s inp M BLOCK_SIZE := by
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, max_kernel_1, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        maxKernel1Spec, maxKernel1InputTile, hB] at hExec ⊢
    cases hExec
    simp [BlockState.writeMem_readMem]
    congr
  · simp [exec, max_kernel_1, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis,
        NumericDType.add, NumericDType.mul, ComparableDType.lt, hB] at hExec

/-- Compute-facing correctness for `max_kernel_1`. -/
theorem max_kernel_1_compute_correct
    (inp mid : RegionName)
    (M BLOCK_SIZE : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := max_kernel_1 inp mid M BLOCK_SIZE)
      (initialState := s)
      (write := fun _ : PUnit => some (mid, s.pid))
      (expected := fun _ => maxKernel1Spec s inp M BLOCK_SIZE) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact max_kernel_1_correct inp mid M BLOCK_SIZE s s' hExec

/-- Algorithm-layer correctness for `max_kernel_2`. -/
theorem max_kernel_2_correct
    (mid out : RegionName)
    (mid_size BLOCK_MID : Nat) (s s' : BlockState)
    (hExec : exec (max_kernel_2 mid out mid_size BLOCK_MID) s = some s') :
    s'.readMem out 0 = maxKernel2Spec s mid mid_size BLOCK_MID := by
  by_cases hB : 0 < BLOCK_MID
  · simp [exec, max_kernel_2, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis,
        ComparableDType.lt, maxKernel2Spec, maxKernel2InputTile, hB] at hExec ⊢
    cases hExec
    simp [BlockState.writeMem_readMem]
    congr
  · simp [exec, max_kernel_2, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis,
        ComparableDType.lt, hB] at hExec

/-- Compute-facing correctness for `max_kernel_2`. -/
theorem max_kernel_2_compute_correct
    (mid out : RegionName)
    (mid_size BLOCK_MID : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := max_kernel_2 mid out mid_size BLOCK_MID)
      (initialState := s)
      (write := fun _ : PUnit => some (out, 0))
      (expected := fun _ => maxKernel2Spec s mid mid_size BLOCK_MID) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro _
  exact max_kernel_2_correct mid out mid_size BLOCK_MID s s' hExec

/-- Compute-facing cellwise correctness for the 2D value/index `max_kernel`. -/
theorem max_kernel_compute_correct
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => maxKernelOutOffset s K BLOCK_M i))
    (hOutRegions : out_value ≠ out_index) :
    ComputeCorrect.OutputPairWhere
      (max_kernel inp out_value out_index M N K BLOCK_M BLOCK_N)
      s out_value out_index
      (maxKernelOutOffset s K BLOCK_M)
      (fun i : Fin BLOCK_M => s.pids 0 * BLOCK_M + i.val < M)
      (maxKernelValueSpec s inp M N K BLOCK_M BLOCK_N)
      (maxKernelIndexSpec s inp M N K BLOCK_M BLOCK_N) := by
  unfold ComputeCorrect.OutputPairWhere ComputeKernel.ExecCorrect
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  by_cases hBM : 0 < BLOCK_M
  · by_cases hBN : 0 < BLOCK_N
    · simp [exec, max_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.argMaxDrop, Tile.argBestDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
            ComparableDType.lt, hBN] at hExec
      rw [← hExec]
      have hOffsetInj : Function.Injective
          (fun idx : TileIndex [BLOCK_M] =>
            (s.pids 0 * BLOCK_M + idx.1.val) * K + s.pids 1) := by
        intro a b h
        have hab : a.1 = b.1 := by
          apply hOutInj
          simpa [maxKernelOutOffset] using h
        rcases a with ⟨a, _⟩
        rcases b with ⟨b, _⟩
        simp at hab ⊢
        exact hab
      constructor
      · rw [BlockState.foldl_writeMemTyped_natAt_prop_masked_readMem_other_region]
        · simp only [maxKernelOutOffset]
          rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
          simp [hActive, maxKernelValueSpec,
            maxKernelInputTile, maxInpElem, Tile.reduceMax, Tile.reduceMaxDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, hBN]
          congr
        · intro k _hk _hPk hEq
          exact hOutRegions hEq
      · simp only [maxKernelOutOffset]
        rw [BlockState.scatter_readback_nat_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
        simp [hActive, maxKernelIndexSpec,
          maxKernelInputTile, maxInpElem, Tile.argMaxDrop, Tile.argBestDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, hBN]
    · simp [exec, max_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
          Tile.reduceMax, Tile.reduceMaxDrop,
          TileShape.axisDim, TileShape.eraseAxis,
          TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
          ComparableDType.lt, hBN] at hExec
  · omega

/-- Per-kernel output summary for the 2D value/index `max_kernel`: the DSL
surface lowers to the algorithm layer, and the masked value/index stores are
compute-correct — every active row lane `i` holds the row-wise maximum
(`maxKernelValueSpec`) in `out_value` and its argmax (`maxKernelIndexSpec`) in
`out_index`. Carries the existing side conditions of
`max_kernel_compute_correct`: `hOutInj` (injective output offsets) and
`hOutRegions` (`out_value ≠ out_index`). -/
theorem max_kernel_output_summary
    (inp out_value out_index : RegionName)
    (M N K BLOCK_M BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_M => maxKernelOutOffset s K BLOCK_M i))
    (hOutRegions : out_value ≠ out_index) :
    (∃ alg, (max_kernel inp out_value out_index M N K BLOCK_M BLOCK_N).toAlgorithm?
        = Except.ok alg) ∧
    ComputeCorrect.OutputPairWhere
      (max_kernel inp out_value out_index M N K BLOCK_M BLOCK_N)
      s out_value out_index
      (maxKernelOutOffset s K BLOCK_M)
      (fun i : Fin BLOCK_M => s.pids 0 * BLOCK_M + i.val < M)
      (maxKernelValueSpec s inp M N K BLOCK_M BLOCK_N)
      (maxKernelIndexSpec s inp M N K BLOCK_M BLOCK_N) := by
  refine ⟨?_, ?_⟩
  · simp [max_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  · exact max_kernel_compute_correct inp out_value out_index M N K BLOCK_M BLOCK_N
      s hOutInj hOutRegions

end VeriTile.Bench.TritonBenchG.MaxReduction
