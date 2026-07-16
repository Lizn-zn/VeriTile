import VeriTile.Triton

/-!
# `l2_norm_triton1` — strict per-kernel correctness

`_l2_norm_fwd_1pass_kernel` is the single-pass forward L2 normalization: program
`row` loads its row of `X` (length `N`, masked into a `BLOCK_N` tile), computes
the row sum of squares `var`, the reciprocal std `rstd = 1/√(var + eps)`, and
stores `x · rstd` to `Y`, masked by `cols < N`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_l2_norm_fwd_1pass_kernel[(M,)](...)`, the 1-D grid over
rows, the `BLOCK_N = min(MAX_FUSED_SIZE, next_power_of_2(N))` choice, the
`N > BLOCK_N` guard, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here. Because `pid`
(the row) is universally quantified, the per-program statement covers every row
of the grid.

## Proof architecture

```
l2_norm_fwd_1pass_kernel_correctness          ← TOP THEOREM (l2NormIO ⊨ l2NormSpec)
  ├─ l2_norm_fwd_1pass_kernel_flattenOk       bridge fragment membership
  ├─ l2_norm_fwd_1pass_kernel_traceSafe       per-execution lane-wise safety walk
  └─ l2_norm_fwd_1pass_kernel_region_run      region-model masked Hoare triple
       ├─ l2_norm_fwd_1pass_kernel_exec_isSome  termination
       ├─ l2_norm_fwd_1pass_kernel_correct      ← algorithm-layer readback per lane
       │    └─ l2VarCarrier_eq_l2NormSqSum       masked reduce-sum = oracle sum-of-squares
       ├─ l2Spec_eq_l2NormSpec                   loaded active lanes → pure spec
       └─ l2_norm_fwd_1pass_kernel_frame         masked scatter frame
```

The headline is the masked Hoare-triple combinator `l2NormIO … ⊨ l2NormSpec`
(`MaskedKernelIO₁.Implements`): for every disjoint flat placement of the two
buffers, every program id all of whose *active* lanes (`j < N`) are in bounds,
and every launch state whose active input-row lanes hold `xs`, the translated
pointer kernel terminates, every active output-row lane holds
`l2NormSpec N BLOCK_N eps xs j`, and every other memory cell is unchanged.

## Modeling boundary

The spec is an **oracle wrapper** over `VeriTile.Triton.Math` L2-norm definitions
(`TiledL2Norm.l2Norm`, built on `l2NormSqSum` / `l2NormRstd`): the L2-norm math
lives once in `Math.*` and is reused here, so this file only checks that the
kernel realizes that oracle lane-wise. The intra-kernel reduction
`tl.sum(xbar·xbar)` is connected to the oracle sum-of-squares by the bridging
lemma `l2VarCarrier_eq_l2NormSqSum` (over `Semantics.MaskedReduction`). Arithmetic
is over `ℝ` (not bit-accurate IEEE float); the masked load `other=0.0`, the
`tl.where(cols < N, x, 0.0)` masking, and the `.to(tl.float32)` cast all reduce to
the algorithm-layer behavior (post-erasure all dtypes unify to `ℝ`); `√`/`⁻¹` are
the real square root and inverse. Inputs are
presented via the `s.readMem`-resolved tile `l2Load`.
-/

namespace VeriTile.Bench.TritonBenchG.L2NormTriton1

open VeriTile.Triton
open VeriTile.Triton.TiledL2Norm
open scoped VeriTile.Triton.MaskedKernelIO₁

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `l2_norm_triton1.py`'s `_l2_norm_fwd_1pass_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N: tl.constexpr` → Lean `Nat` parameter. -/
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

noncomputable def l2InputTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      let off := s.pid * stride_x_row + idx.1.val
      if idx.1.val < N then
        if idx.1.val < N then some (s.readMem X off) else some (0.0 : ℝ)
      else
        some (0.0 : ℝ) }

/-- Masked row element `X[pid, idx]`: lane `idx` of **this program's row**
(row = `pid`, row stride `stride_x_row`, unit column stride), `0` beyond the
`N` bound (`mask=cols < N, other=0.0`). -/
noncomputable def l2Load
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat)
    (idx : Fin BLOCK_N) : ℝ :=
  if idx.val < N then
    s.readMem X (s.pid * stride_x_row + idx.val)
  else
    0

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
    TileShape.insertAxisIndex, Tile.bop, NumericDType.mul,
    BlockState.pid_eq]
  refine (reduceSum_masked_sq_eq_some_sum
    (fun k : Fin BLOCK_N => s.readMem X (s.pids 0 * stride_x_row + k.val))
    (fun k : Fin BLOCK_N => k.val < N)).trans ?_
  congr 1
  unfold l2NormSqSum l2Load
  apply Finset.sum_congr rfl
  intro k _hk
  by_cases h : k.val < N <;> simp [h, BlockState.pid_eq]

/-- Kernel-coupled spec value at lane `idx`: the `Math.*` oracle `l2Norm`
applied to this program's masked row load. Internal plumbing for
`l2_norm_fwd_1pass_kernel_correct`; the headline states the pure
`l2NormSpec` instead (connected by `l2Spec_eq_l2NormSpec`). -/
noncomputable def l2Spec
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) (eps : ℝ)
    (idx : Fin BLOCK_N) : ℝ :=
  l2Norm (l2Load s X stride_x_row N BLOCK_N) eps idx

/-- Exact L2-normalization value computed by the kernel at lane `idx`, as a
pure function of the active row prefix `xs j`, `j < N`: the `Math.*` oracle
`l2Norm` over the masked row (lanes `≥ N` enter the sum-of-squares as `0`,
matching `mask=cols < N, other=0.0`). -/
noncomputable def l2NormSpec (N BLOCK_N : Nat) (eps : ℝ)
    (xs : Fin BLOCK_N → ℝ) (idx : Fin BLOCK_N) : ℝ :=
  l2Norm (fun j : Fin BLOCK_N => if j.val < N then xs j else 0) eps idx

/-- Active-lane congruence: from any launch state whose **active** input-row
lanes hold `xs`, the kernel-coupled spec is the pure `l2NormSpec` of `xs`
(inactive lanes are masked to `0` on both sides). -/
theorem l2Spec_eq_l2NormSpec
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) (eps : ℝ)
    (xs : Fin BLOCK_N → ℝ)
    (hx : ∀ j : Fin BLOCK_N, j.val < N →
      s.readMem X (s.pid * stride_x_row + j.val) = xs j)
    (i : Fin BLOCK_N) :
    l2Spec s X stride_x_row N BLOCK_N eps i = l2NormSpec N BLOCK_N eps xs i := by
  have hload : l2Load s X stride_x_row N BLOCK_N
      = fun j : Fin BLOCK_N => if j.val < N then xs j else 0 := by
    funext j
    unfold l2Load
    by_cases h : j.val < N <;> simp [h, hx j]
  unfold l2Spec l2NormSpec
  rw [hload]

theorem l2_norm_fwd_1pass_kernel_correct
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      let outAddr := s.pid * stride_x_row + i.val
      s'.readMem Y outAddr =
        if i.val < N then
          l2Spec s X stride_x_row N BLOCK_N eps i
        else s.readMem Y outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_N
  · simp [exec, l2_norm_fwd_1pass_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.div,
          ComparableDType.lt, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
    rw [← hExec]
    simp [BlockState.pid_eq]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < N
    · have hvar := l2VarCarrier_eq_l2NormSqSum s X stride_x_row N BLOCK_N
      simp [l2VarCarrier, l2InputTile, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        Tile.bop, NumericDType.mul, BlockState.pid_eq] at hvar
      simp [hi, l2Spec, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, WithBot.realSqrt, l2Load, l2Norm, l2NormRstd]
      erw [hvar]
      rfl
    · simp [hi]
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked store). -/
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
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `Y`, and the *inactive* lanes of
the output row itself — is preserved by the run. -/
private theorem l2_norm_fwd_1pass_kernel_frame
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s s1 : BlockState)
    (hExec : exec ((l2_norm_fwd_1pass_kernel X Y stride_x_row N eps
        BLOCK_N).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_N, i.val < N →
      ¬(Y = r ∧ s.pid * stride_x_row + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, l2_norm_fwd_1pass_kernel, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state — including
`BLOCK_N = 0`, since the only reduction is a `sum` (total on empty axes), so no
`0 < BLOCK_N` side condition is needed. -/
private theorem l2_norm_fwd_1pass_kernel_exec_isSome
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s : BlockState) :
    ∃ s1, exec ((l2_norm_fwd_1pass_kernel X Y stride_x_row N eps
        BLOCK_N).toAlgKernel) s = some s1 := by
  simp [exec, l2_norm_fwd_1pass_kernel, ComputeKernel.toAlgKernel, stepStmts,
        stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.div,
        ComparableDType.lt, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input row is loaded at the **active lanes only** (`j < N`). This is the `hrun`
obligation of the `⊨` headline. -/
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
            = l2NormSpec N BLOCK_N eps xs j)
      ∧ (∀ r o,
          (r ≠ Y ∨ ∀ j : Fin BLOCK_N, j.val < N →
            o ≠ s₀.pid * stride_x_row + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := l2_norm_fwd_1pass_kernel_exec_isSome X Y
    stride_x_row N eps BLOCK_N s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := l2_norm_fwd_1pass_kernel_correct X Y stride_x_row N eps BLOCK_N
      s₀ s1 hs1 j
    simp only [hj, if_pos] at h
    rw [h]
    exact l2Spec_eq_l2NormSpec s₀ X stride_x_row N BLOCK_N eps xs hx j
  · refine l2_norm_fwd_1pass_kernel_frame X Y stride_x_row N eps BLOCK_N
      s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked load with `other`, dtype cast, `where`, sum reduction,
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
/-- Per-execution safety walk: one computational unfold walks all eleven
statements — nine are memory-silent (`program_id`, the pointer/index staging,
the `where`/reduction/`rsqrt` register arithmetic) — and reduces the two masked
accesses (row load, row store) to the **lane-wise** bounds hypotheses: every
*active* lane's address is below the region bound. -/
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
    ComparableDType.lt,
    Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

/-- `_l2_norm_fwd_1pass_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_N` — the row window each program owns;
* `read`/`write` — program `row` reads its row of `X` and writes its row of
  `Y`, both at `row * stride_x_row` (the host passes the same row stride for
  both buffers; the one-program-per-row launch convention);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = min(MAX_FUSED_SIZE, next_power_of_2(N))`) carry no obligations on
  either side. The store mask equals the load mask, so `writeMask` keeps its
  default.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def l2NormIO (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    MaskedKernelIO₁ where
  kernel := l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N
  inp := X
  out := Y
  B := BLOCK_N
  read := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N

/-- **The headline**: `_l2_norm_fwd_1pass_kernel` implements the exact L2
normalization over the active row prefix on its masked IO signature — for
every disjoint flat placement of the two buffers, every program id whose
active lanes are in bounds, and every launch state whose active input-row
lanes hold `xs`, the translated pointer kernel terminates, every active
output-row lane `j` holds `l2NormSpec N BLOCK_N eps xs j` (the `Math.*`
oracle `l2Norm` over the masked row), and every other memory cell is
unchanged. No `0 < BLOCK_N` side condition: the kernel's only reduction is a
`sum`, total on empty tiles. Proof: `Implements.intro` assembles the
region-model masked triple with the bridge side conditions. -/
specification l2_norm_fwd_1pass_kernel_correctness
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2NormIO X Y stride_x_row N eps BLOCK_N ⊨
      fun xs i => l2NormSpec N BLOCK_N eps xs i := by
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

end VeriTile.Bench.TritonBenchG.L2NormTriton1
