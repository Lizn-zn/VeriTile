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
max_kernel_1_io_correctness                     ← TOP THEOREM (`⊨`, first-stage block max)
  ├─ max_kernel_1_flattenOk                     inside the flat-memory bridge
  ├─ max_kernel_1_traceSafe                     per-execution address safety
  └─ max_kernel_1_region_run                    region-model run
       ├─ max_kernel_1_terminates
       ├─ max_kernel_1_correct                  per-block readback (shared, below)
       ├─ maxKernel1Spec_eq_of                  memory spec = value spec under the pins
       └─ max_kernel_1_frame                    cell-level frame (only `mid[pid]` is touched)

max_kernel_output_summary                       per-write-map summary (the 2D value/index kernel)
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
open scoped VeriTile.Triton.MaskedTileKernelIO₁

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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
specification max_kernel_output_summary
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

/-! ## ════════ `⊨` IO face for the first-stage block max ════════ -/

section IOFace

/-- Value-level first-stage max spec: the `Tile.reduceMax` of the tile that holds
`xs` on the active lanes (`pid·BLOCK_SIZE + i < M`) and `⊥` elsewhere — the
`other=-float("inf")` padding. Written over the *loaded values* rather than over
memory, which is what the IO surface quantifies. -/
noncomputable def maxTileSpecOf (M BLOCK_SIZE pid : Nat)
    (xs : TileIndex [BLOCK_SIZE] → ℝ) : ℝ :=
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      ⟨fun idx =>
        if pid * BLOCK_SIZE + idx.1.val < M then some (xs idx) else none⟩ with
  | some out => WithBot.unbotD 0 (out.data PUnit.unit)
  | none => 0

/-- The memory-level and value-level first-stage specs agree once the active
lanes of the input window are pinned to `xs`. -/
theorem maxKernel1Spec_eq_of (inp : RegionName) (M BLOCK_SIZE : Nat)
    (s : BlockState) (xs : TileIndex [BLOCK_SIZE] → ℝ)
    (hx : ∀ idx : TileIndex [BLOCK_SIZE], s.pid * BLOCK_SIZE + idx.1.val < M →
      s.readMem inp (s.pid * BLOCK_SIZE + idx.1.val) = xs idx) :
    maxKernel1Spec s inp M BLOCK_SIZE = maxTileSpecOf M BLOCK_SIZE s.pid xs := by
  have htile : maxKernel1InputTile s inp M BLOCK_SIZE
      = ⟨fun idx : TileIndex [BLOCK_SIZE] =>
          if s.pid * BLOCK_SIZE + idx.1.val < M then some (xs idx)
          else none⟩ := by
    refine congrArg Tile.mk (funext fun idx => ?_)
    simp only [maxKernel1InputTile]
    by_cases h : s.pid * BLOCK_SIZE + idx.1.val < M
    · rw [if_pos h, if_pos h, hx idx h]
    · rw [if_neg h, if_neg h]
  rw [maxKernel1Spec, maxTileSpecOf, htile]

/-- `Op.FlattenOk` of a `reduceMax` reduces to its argument's obligation.
Stated as its own lemma and used as a *term*: the per-case equation does not
fire under `simp` on a `reduceMax` node, only under `rw`, and `rw` only succeeds
in an isolated goal. -/
private theorem flattenOk_reduceMax {shape : TileShape}
    (ax : Fin shape.length) (keepDims : Bool) (e : Op .real shape)
    (h : e.FlattenOk) : (Op.reduceMax ax keepDims e).FlattenOk := by
  rw [Op.FlattenOk]
  exact h

/-- `Op.FlattenOk` of a register reference is vacuous. -/
private theorem flattenOk_ref (dtype : TileDType) (shape : TileShape)
    (name : RegName) : (Op.ref dtype shape name).FlattenOk := by
  rw [Op.FlattenOk]
  trivial

/-- `Op.SafeAt` of a `reduceMax` is its argument's obligation (same `rw`-only
caveat as `flattenOk_reduceMax`). -/
private theorem safeAt_reduceMax {shape : TileShape} (bounds : RegionBounds)
    (s : BlockState) (ax : Fin shape.length) (keepDims : Bool)
    (e : Op .real shape) (h : Op.SafeAt bounds s e) :
    Op.SafeAt bounds s (Op.reduceMax ax keepDims e) := by
  rw [Op.SafeAt]
  exact h

/-- `Op.SafeAt` of a register reference is vacuous. -/
private theorem safeAt_ref (bounds : RegionBounds) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName) :
    Op.SafeAt bounds s (Op.ref dtype shape name) := by
  rw [Op.SafeAt]
  trivial

/-- The first-stage kernel sits inside the flat-memory bridge's covered
fragment. -/
theorem max_kernel_1_flattenOk (inp mid : RegionName) (M BLOCK_SIZE : Nat) :
    ((max_kernel_1 inp mid M BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [max_kernel_1, ComputeKernel.toAlgKernel, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  exact flattenOk_reduceMax _ _ _ (flattenOk_ref _ _ _)

/-- Termination of the first-stage kernel (needs a non-degenerate tile: with
`BLOCK_SIZE = 0` the `tl.max` reduction has no axis to reduce and the kernel
faults). -/
theorem max_kernel_1_terminates (inp mid : RegionName) (M BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s : BlockState) :
    ∃ s1, exec (max_kernel_1 inp mid M BLOCK_SIZE) s = some s1 := by
  simp [exec, max_kernel_1, stepStmts, stepStmt, evalOp.eq_def, Tile.bop,
    Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, hB]

/-- Cell-level frame of the first-stage kernel: the only cell it touches is
`mid[pid]`. -/
theorem max_kernel_1_frame (inp mid : RegionName) (M BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s s' : BlockState)
    (hExec : exec (max_kernel_1 inp mid M BLOCK_SIZE) s = some s') :
    ∀ (r : RegionName) (o : Nat), (r ≠ mid ∨ o ≠ s.pid) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, max_kernel_1, stepStmts, stepStmt, evalOp.eq_def, Tile.bop,
    Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, NumericDType.add, NumericDType.mul,
    ComparableDType.lt, hB] at hExec
  subst hExec
  rw [BlockState.writeMem_mem, if_neg ?_]
  · simp
  · rintro ⟨h1, h2⟩
    rcases hcond with h | h
    · exact h h1
    · exact h h2

/-- Per-execution safety walk for the first-stage kernel: the masked load
addresses `inp` at `pid·BLOCK_SIZE + i`, active exactly on `< M`; the scalar
store addresses `mid` at `pid`. -/
theorem max_kernel_1_traceSafe (inp mid : RegionName) (M BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hIn : ∀ i : Fin BLOCK_SIZE, s.pid * BLOCK_SIZE + i.val < M →
      s.pid * BLOCK_SIZE + i.val < bounds inp)
    (hMid : s.pid < bounds mid) :
    ((max_kernel_1 inp mid M BLOCK_SIZE).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, max_kernel_1, Stmt.TraceSafeList, Stmt.TraceSafe,
    Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active, MemAccess.SafeAt,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    Op.PointerAddressesSafeOn, Op.MemorySafe, stepStmt, evalOp, evalOp.eq_def,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.reduceMax, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, hB]
  -- what survives: the load window's bound, the `reduceMax` node (simp cannot
  -- peel it), and the scalar store's bound
  exact ⟨fun a ha => hIn a ha,
    safeAt_reduceMax _ _ _ _ _ (safeAt_ref _ _ _ _ _), hMid⟩

/-- Region-model run of the first-stage kernel, in the shape
`MaskedTileKernelIO₁.Implements.intro` consumes. -/
theorem max_kernel_1_region_run (inp mid : RegionName) (M BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s₀ : BlockState)
    (xs : TileIndex [BLOCK_SIZE] → ℝ)
    (hx : ∀ idx : TileIndex [BLOCK_SIZE],
      s₀.pid * BLOCK_SIZE + idx.1.val < M →
      s₀.readMem inp (s₀.pid * BLOCK_SIZE + idx.1.val) = xs idx) :
    ∃ s1, exec (max_kernel_1 inp mid M BLOCK_SIZE) s₀ = some s1
      ∧ (∀ idx : TileIndex [BLOCK_SIZE], idx.1.val = 0 →
          s1.readMem mid s₀.pid = maxTileSpecOf M BLOCK_SIZE s₀.pid xs)
      ∧ (∀ (r : RegionName) (o : Nat), (r ≠ mid ∨ o ≠ s₀.pid) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := max_kernel_1_terminates inp mid M BLOCK_SIZE hB s₀
  refine ⟨s1, hexec, ?_,
    max_kernel_1_frame inp mid M BLOCK_SIZE hB s₀ s1 hexec⟩
  intro _idx _
  rw [max_kernel_1_correct inp mid M BLOCK_SIZE s₀ s1 hexec,
    maxKernel1Spec_eq_of inp M BLOCK_SIZE s₀ xs hx]

/-- IO signature of `max_kernel_1` on the tile-indexed surface: every lane of
the `BLOCK_SIZE` window reads `inp` at `pid·BLOCK_SIZE + i` and is read-active on
the `< M` guard, while **only lane 0** is write-active and it writes the single
cell `mid[pid]` — the block's max. A single-cell store expressed as a one-lane
write mask over the read tile is exactly what a reduction's IO signature is. -/
def maxKernel1IO (inp mid : RegionName) (M BLOCK_SIZE : Nat) :
    MaskedTileKernelIO₁ where
  kernel := max_kernel_1 inp mid M BLOCK_SIZE
  inp := inp
  out := mid
  shape := [BLOCK_SIZE]
  read := fun pid idx => pid * BLOCK_SIZE + idx.1.val
  write := fun pid _ => pid
  mask := fun pid idx => pid * BLOCK_SIZE + idx.1.val < M
  writeMask := fun _ idx => idx.1.val = 0

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `max_reduction.py`'s `max_kernel_1`:
for every disjoint flat placement of the two buffers, every program id whose
active lanes are in bounds, and every launch state whose input window holds `xs`
at the active lanes, the translated pointer kernel terminates, `mid[pid]` holds
the genuine block max — `Tile.reduceMax` of `xs` over the active lanes, with the
`other=-float("inf")` padding modeled as `⊥` — and every other memory cell is
unchanged.

This is the first *reduction* on an `io ⊨ f` face, and it is what the spec `f`'s
program-id argument is for: the reduced index set is `{i | pid·BLOCK_SIZE + i < M}`,
so the value is irreducibly pid-dependent (the tail block reduces fewer lanes
than a full block) and a pid-independent spec would be falsifiable.

Dimension-general in `M` and `BLOCK_SIZE`. Honest side-condition:
`0 < BLOCK_SIZE` — with an empty tile `tl.max` has no axis to reduce and the
kernel faults, so termination genuinely fails there. -/
specification max_kernel_1_io_correctness
    (inp mid : RegionName) (M BLOCK_SIZE : Nat) (hB : 0 < BLOCK_SIZE) :
    maxKernel1IO inp mid M BLOCK_SIZE
      ⊨ fun pid xs _ => maxTileSpecOf M BLOCK_SIZE pid xs := by
  refine MaskedTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact max_kernel_1_flattenOk inp mid M BLOCK_SIZE
  · intro bounds s h1 h2
    exact max_kernel_1_traceSafe inp mid M BLOCK_SIZE hB bounds s
      (fun i hi => h1 (i, PUnit.unit) hi)
      (h2 (⟨0, hB⟩, PUnit.unit) rfl)
  · intro s₀ xs hin
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      max_kernel_1_region_run inp mid M BLOCK_SIZE hB s₀ xs
        (fun idx hact => hin idx hact)
    refine ⟨s1, hexec, hval, ?_⟩
    intro r o hcond
    refine hframe r o ?_
    rcases hcond with h | h
    · exact Or.inl h
    -- a non-degenerate tile always has a write-active lane, which turns the
    -- skin's lane-wise disjunct into the single-cell one
    · exact Or.inr (h (⟨0, hB⟩, PUnit.unit) rfl)

end IOFace

end VeriTile.Bench.TritonBenchG.MaxReduction
