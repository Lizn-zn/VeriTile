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
l2_norm_bwd_kernel_correctness                ← TOP THEOREM (l2BwdIO ⊨ l2BwdSpec)
  ├─ l2_norm_bwd_kernel_flattenOk             bridge fragment membership
  ├─ l2_norm_bwd_kernel_traceSafe             per-execution lane-wise safety walk
  └─ l2_norm_bwd_kernel_region_run            region-model masked Hoare triple
       ├─ l2_norm_bwd_kernel_exec_isSome      termination
       ├─ l2_norm_bwd_kernel_correct          algorithm-layer readback per lane
       │    ├─ l2BwdVarCarrier_eq_l2NormSqSum  masked reduce-sum = oracle sum-of-squares
       │    ├─ l2BwdDotCarrier_eq_l2NormDot    masked reduce-sum = oracle dot product
       │    └─ l2BwdSpec_congr                 only active lanes feed the spec
       └─ l2_norm_bwd_kernel_frame            masked scatter frame
```

The headline is stated on the kernel's masked **IO signature** `l2BwdIO`
(`MaskedKernelIO₂`): which buffer is which argument, where program `pid`
(the row) reads its `X`/`DY` tiles and writes its `DX` tile (all three at
`pid * stride_x_row`, the host's one-program-per-row launch convention), and
the active-lane predicate `j < N`. `⊨` is the audit-once masked Hoare-triple
combinator (`MaskedKernelIO₂.Implements`): for **every** disjoint placement of
the three buffers in flat memory, **every** program id all of whose *active*
lanes are in bounds (partial blocks may overhang the buffer on their inactive
lanes), and **every** launch state whose input windows hold `xs`/`dys` at the
active lanes, the translated pointer kernel terminates, every active output
lane holds `l2BwdSpec N BLOCK_N eps xs dys j`, and every other memory cell is
unchanged.

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
inverse. Both reductions run over the full `BLOCK_N` block, but masked lanes
enter as `0`, neutral for the sums, so `l2BwdSpec` zero-fills its inputs beyond
`N` the same way. No output/input disjointness is assumed in the region model:
both rows are read into registers before the masked scatter.
-/

namespace VeriTile.Bench.TritonBenchG.L2NormBwd

open VeriTile.Triton
open VeriTile.Triton.TiledL2Norm
open scoped VeriTile.Triton.MaskedKernelIO₂

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

/-- Oracle L2-norm backward value at lane `i`, as a **pure** function of the
active row prefixes `xs j` / `dys j`, `j < N` (masked lanes enter both
reductions as `0`, matching the kernel's `other=0.0` loads and
`tl.where(cols < N, ·, 0.0)` zero-fill). -/
noncomputable def l2BwdSpec (N BLOCK_N : Nat) (eps : ℝ)
    (xs dys : Fin BLOCK_N → ℝ) (i : Fin BLOCK_N) : ℝ :=
  l2NormBwd
    (fun j => if j.val < N then xs j else 0)
    (fun j => if j.val < N then dys j else 0)
    eps i

/-- `l2BwdSpec` only reads the active lanes of its inputs: rows agreeing below
`N` yield the same backward value (masked lanes are zero-filled either way). -/
theorem l2BwdSpec_congr (N BLOCK_N : Nat) (eps : ℝ)
    (xs xs' dys dys' : Fin BLOCK_N → ℝ)
    (hx : ∀ j : Fin BLOCK_N, j.val < N → xs j = xs' j)
    (hy : ∀ j : Fin BLOCK_N, j.val < N → dys j = dys' j)
    (i : Fin BLOCK_N) :
    l2BwdSpec N BLOCK_N eps xs dys i = l2BwdSpec N BLOCK_N eps xs' dys' i := by
  unfold l2BwdSpec
  have hxe : (fun j : Fin BLOCK_N => if j.val < N then xs j else 0)
      = fun j => if j.val < N then xs' j else 0 := by
    funext j
    by_cases h : j.val < N
    · simp [h, hx j h]
    · simp [h]
  have hye : (fun j : Fin BLOCK_N => if j.val < N then dys j else 0)
      = fun j => if j.val < N then dys' j else 0 := by
    funext j
    by_cases h : j.val < N
    · simp [h, hy j h]
    · simp [h]
  rw [hxe, hye]

/-- Algorithm-layer cellwise correctness for `_l2_norm_bwd_kernel`: in-bounds
lanes hold `l2BwdSpec` of the loaded rows, out-of-bounds lanes are preserved. -/
theorem l2_norm_bwd_kernel_correct
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s s' : BlockState)
    (hExec : exec (l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N) s = some s') :
    ∀ i : Fin BLOCK_N,
      let outAddr := s.pid * stride_x_row + i.val
      s'.readMem DX outAddr =
        if i.val < N then
          l2BwdSpec N BLOCK_N eps
            (fun j => s.readMem X (s.pid * stride_x_row + j.val))
            (fun j => s.readMem DY (s.pid * stride_x_row + j.val)) i
        else s.readMem DX outAddr := by
  intro i
  simp [exec, l2_norm_bwd_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?] at hExec
  rw [← hExec]
  simp only [BlockState.pid_eq]
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
store — every cell of every region other than `DX`, and the *inactive* lanes
of the output row itself — is preserved by the run. -/
private theorem l2_norm_bwd_kernel_frame
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s s1 : BlockState)
    (hExec : exec ((l2_norm_bwd_kernel X DY DX stride_x_row N eps
        BLOCK_N).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_N, i.val < N →
      ¬(DX = r ∧ s.pid * stride_x_row + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, l2_norm_bwd_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def,
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
/-- Termination: the kernel executes to completion from any state (both
reductions are `sum`s, defined on any axis length, so no `0 < BLOCK_N` side
condition is needed). -/
private theorem l2_norm_bwd_kernel_exec_isSome
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s : BlockState) :
    ∃ s1, exec ((l2_norm_bwd_kernel X DY DX stride_x_row N eps
        BLOCK_N).toAlgKernel) s = some s1 := by
  simp [exec, l2_norm_bwd_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input rows are loaded at the **active lanes only** (`j < N`). This is the
`hrun` obligation of the `⊨` headline; the value half reuses
`l2_norm_bwd_kernel_correct` via `l2BwdSpec_congr`. -/
theorem l2_norm_bwd_kernel_region_run
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (s₀ : BlockState) (xs dys : Fin BLOCK_N → ℝ)
    (hx : ∀ j : Fin BLOCK_N, j.val < N →
      s₀.readMem X (s₀.pid * stride_x_row + j.val) = xs j)
    (hy : ∀ j : Fin BLOCK_N, j.val < N →
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
  obtain ⟨s1, hs1⟩ := l2_norm_bwd_kernel_exec_isSome X DY DX stride_x_row N eps
    BLOCK_N s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := l2_norm_bwd_kernel_correct X DY DX stride_x_row N eps BLOCK_N
      s₀ s1 hs1 j
    simp only [hj, if_pos] at h
    rw [h]
    exact l2BwdSpec_congr N BLOCK_N eps _ xs _ dys
      (fun k hk => hx k hk) (fun k hk => hy k hk) j
  · refine l2_norm_bwd_kernel_frame X DY DX stride_x_row N eps BLOCK_N
      s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads with `other`, `.to(tl.float32)` casts, `where`,
reductions, masked store). -/
theorem l2_norm_bwd_kernel_flattenOk
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    ((l2_norm_bwd_kernel X DY DX stride_x_row N eps
        BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [l2_norm_bwd_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all fourteen
statements — eleven are memory-silent (`program_id`, the pointer staging,
`arange`, the `where` zero-fills, the two reductions, and the register
arithmetic) — and reduces the three masked accesses (`X` load, `DY` load,
`DX` store, all at `pid * stride_x_row + j`, active iff `j < N`) to the
**lane-wise** bounds hypotheses: every *active* lane's address is below the
region bound of the buffer it touches. -/
theorem l2_norm_bwd_kernel_traceSafe
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hin1 : ∀ j : Fin BLOCK_N, j.val < N →
      s.pid * stride_x_row + j.val < bounds X)
    (hin2 : ∀ j : Fin BLOCK_N, j.val < N →
      s.pid * stride_x_row + j.val < bounds DY)
    (hout : ∀ j : Fin BLOCK_N, j.val < N →
      s.pid * stride_x_row + j.val < bounds DX) :
    Kernel.TraceSafe bounds
      ((l2_norm_bwd_kernel X DY DX stride_x_row N eps
        BLOCK_N).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin1 hin2 hout
  simp [l2_norm_bwd_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
    Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin1 a ha, fun a ha => hin2 a ha, fun a ha => hout a ha⟩

/-- `_l2_norm_bwd_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring: input `X`,
  upstream gradient `DY`, input gradient `DX`);
* `B = BLOCK_N` — the row window each program owns;
* `read1`/`read2`/`write` — program `pid` (the row) reads and writes its row
  at `pid * stride_x_row` in all three buffers (the host-side
  one-program-per-row launch convention);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = next_power_of_2(N)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def l2BwdIO (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : MaskedKernelIO₂ where
  kernel := l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N
  in1 := X
  in2 := DY
  out := DX
  B := BLOCK_N
  read1 := fun pid => pid * stride_x_row
  read2 := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N

/-- **The headline**: `_l2_norm_bwd_kernel` implements the oracle L2-norm
backward `l2BwdSpec` (i.e. `dx = dy·rstd − (Σ dy·x)·(1/(var+eps))·rstd·x` over
the zero-filled active prefix) on its masked IO signature — for every disjoint
flat placement of the three buffers, every program id whose active lanes are
in bounds, and every launch state whose active input-row lanes hold
`xs`/`dys`, the translated pointer kernel terminates, every active output-row
lane `j` holds `l2BwdSpec N BLOCK_N eps xs dys j`, and every other memory cell
is unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification l2_norm_bwd_kernel_correctness
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2BwdIO X DY DX stride_x_row N eps BLOCK_N
      ⊨ fun xs dys i => l2BwdSpec N BLOCK_N eps xs dys i := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact l2_norm_bwd_kernel_flattenOk X DY DX stride_x_row N eps BLOCK_N
  · intro bounds s h1 h2 h3 _
    exact l2_norm_bwd_kernel_traceSafe X DY DX stride_x_row N eps BLOCK_N
      bounds s h1 h2 h3
  · intro s₀ xs dys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := l2_norm_bwd_kernel_region_run
      X DY DX stride_x_row N eps BLOCK_N s₀ xs dys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.L2NormBwd
