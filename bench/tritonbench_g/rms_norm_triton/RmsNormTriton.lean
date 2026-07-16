import VeriTile.Triton

/-!
# `rms_norm_triton` — strict per-kernel correctness

`rms_norm_kernel` is a row-wise RMS normalization: program `pid` advances `X`/`Y`
by its row stride, loads one row (masked by `arange < N`, masked lanes read as
`0`), computes `var = sum(x*x)/N`, `rrms = 1/sqrt(var + eps)`, and stores
`(x * rrms) * w` back, masked by `arange < N`, where `w` is the per-column
weight tile.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`rms_norm_kernel[M,](...)`, the one-program-per-row grid,
the `BLOCK_SIZE = next_power_of_2(N)` choice, and how the runtime composes
per-row writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because `pid` is universally quantified, the per-program
statement covers every row of the grid.

## Proof architecture

```
rms_norm_kernel_correctness                   ← TOP THEOREM (rmsNormIO ⊨ rmsNormSpec)
  ├─ rms_norm_kernel_flattenOk                bridge fragment membership
  ├─ rms_norm_kernel_traceSafe                per-execution lane-wise safety walk
  └─ rms_norm_kernel_region_run               region-model masked Hoare triple
       ├─ rms_norm_kernel_exec_isSome         termination
       ├─ rms_norm_kernel_correct             ← algorithm-layer readback per lane
       ├─ rmsNormSpec_congr                   only active lanes feed the spec
       └─ rms_norm_kernel_frame               masked scatter frame
```

The headline is the masked Hoare-triple combinator `rmsNormIO … ⊨ rmsNormSpec`
(`Masked2DKernelIO₂.Implements`, the general-window struct: the `X`/`Y` rows
are **column-strided**, lane `j` sits at `row_base + j * stride_c`): for every
disjoint flat placement of the three buffers, every program id all of whose
*active* lanes (`j < N`) are in bounds, and every launch state whose active
input lanes hold `xs` (the `X` row) and `ws` (the weights), the translated
pointer kernel terminates, every active output lane holds
`rmsNormSpec N BLOCK_SIZE eps xs ws j`, and every other memory cell is
unchanged.

The spec `rmsNormSpec` threads the carriers `rmsSumCarrier` (`reduceSum (x*x)`
over the masked row) and `rmsRrmsCarrier` (`(sqrt (sum/N + eps))⁻¹`), then
forms `(x * rrms) * w` lane-wise — a pure function of the active prefixes.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.jit
do_not_specialize` / autotuning are not modeled. The `.to(tl.float32)` load cast
and the `(x * rrms).to(Y.dtype.element_ty)` store-precision cast reduce to the
identity at the algorithm layer (post-erasure all dtypes unify to `ℝ`). The
`sum(x*x)` reduction runs over the full `BLOCK_SIZE` block, but masked lanes load
`0` (matching `other=0.0`), so the reduction-over-padded-block matches the
upstream semantics. The headline takes `0 < y_stride_c`: distinct lanes must
scatter to distinct output cells (a zero column stride would make the whole row
self-clobber one cell). The grid is 1-D; the signature's second program-id axis
is unused (windows and mask are constant in `pid₁`).
-/

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton

open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `rms_norm_triton.py`'s `rms_norm_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
def rms_norm_kernel
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  Y += pid * $(y_stride_r)
  X += pid * $(x_stride_r)
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $(eps))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to(Y.dtype.element_ty) * w
  tl.store(Y + cols * $(y_stride_c), y, mask=mask)
}

/-- Masked input row tile: lane `j < N` holds `xs j`, masked lanes are `0`,
matching `mask=…, other=0.0`. -/
noncomputable def rmsInputTile (N BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then some (xs idx.1)
      else some (0.0 : ℝ) }

/-- `sum(x*x)` over the masked row: masked lanes enter as `0`, neutral for the
sum, so the reduction-over-padded-block equals the sum over the active
prefix. -/
noncomputable def rmsSumCarrier (N BLOCK_SIZE : Nat) (xs : Fin BLOCK_SIZE → ℝ) :
    WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsInputTile N BLOCK_SIZE xs)
      (rmsInputTile N BLOCK_SIZE xs))).data PUnit.unit

/-- `rrms = 1 / sqrt(sum(x*x)/N + eps)`. -/
noncomputable def rmsRrmsCarrier (N BLOCK_SIZE : Nat) (eps : ℝ)
    (xs : Fin BLOCK_SIZE → ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt (Option.map ((fun a => a + eps) ∘ fun a => a / (N : ℝ))
      (rmsSumCarrier N BLOCK_SIZE xs)))

/-- Exact RMS-normalization value computed by the kernel at lane `idx`, as a
pure function of the active row prefix `xs j` and weight prefix `ws j`,
`j < N`: `(x · rrms) · w` with `rrms` threaded through `rmsRrmsCarrier`. -/
noncomputable def rmsNormSpec (N BLOCK_SIZE : Nat) (eps : ℝ)
    (xs ws : Fin BLOCK_SIZE → ℝ) (idx : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun x w => x * w)
      (Option.map₂ (fun x rrms => x * rrms)
        (some (xs idx))
        (rmsRrmsCarrier N BLOCK_SIZE eps xs))
      (some (ws idx)))

/-- `rmsNormSpec` at an **active** lane only reads the active lanes of its
inputs: rows/weights agreeing below `N` yield the same value (masked lanes are
`0` in the sum-of-squares tile either way). -/
theorem rmsNormSpec_congr (N BLOCK_SIZE : Nat) (eps : ℝ)
    (xs xs' ws ws' : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < N → xs j = xs' j)
    (hw : ∀ j : Fin BLOCK_SIZE, j.val < N → ws j = ws' j)
    (i : Fin BLOCK_SIZE) (hi : i.val < N) :
    rmsNormSpec N BLOCK_SIZE eps xs ws i
      = rmsNormSpec N BLOCK_SIZE eps xs' ws' i := by
  have htile : rmsInputTile N BLOCK_SIZE xs = rmsInputTile N BLOCK_SIZE xs' := by
    unfold rmsInputTile
    congr 1
    funext idx
    by_cases hj : idx.1.val < N
    · simp only [if_pos hj, hx idx.1 hj]
    · simp only [if_neg hj]
  unfold rmsNormSpec rmsRrmsCarrier rmsSumCarrier
  rw [htile, hx i hi, hw i hi]

/-- Algorithm-layer correctness for `rms_norm_kernel`: in-bounds lanes hold
`rmsNormSpec` of the loaded row and weights, out-of-bounds lanes are
preserved. -/
theorem rms_norm_kernel_correct
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => s.pids 0 * y_stride_r + i.val * y_stride_c))
    (hExec : exec (rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
          N eps BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := s.pids 0 * y_stride_r + i.val * y_stride_c
      s'.readMem Y outAddr =
        if i.val < N then
          rmsNormSpec N BLOCK_SIZE eps
            (fun j => s.readMem X (s.pids 0 * x_stride_r + j.val * x_stride_c))
            (fun j => s.readMem W j.val) i
        else s.readMem Y outAddr := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE] =>
        s.pids 0 * y_stride_r + idx.1.val * y_stride_c) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  simp [exec, rms_norm_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot] at hExec
  subst s'
  dsimp only
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ h_inj (i, PUnit.unit)]
  by_cases hi : i.val < N
  · simp [hi, rmsNormSpec, rmsRrmsCarrier, rmsSumCarrier,
          rmsInputTile, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          WithBot.realSqrt, NumericDType.mul]
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
store — every cell of every region other than `Y`, and the *inactive* lanes of
the output row itself — is preserved by the run. -/
private theorem rms_norm_kernel_frame
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (s s1 : BlockState)
    (hExec : exec ((rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r
        x_stride_c N eps BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, i.val < N →
      ¬(Y = r ∧ s.pids 0 * y_stride_r + i.val * y_stride_c = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, rms_norm_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state — including
`BLOCK_SIZE = 0`, since the only reduction is a `sum` (total on empty axes), so
no `0 < BLOCK_SIZE` side condition is needed. -/
private theorem rms_norm_kernel_exec_isSome
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState) :
    ∃ s1, exec ((rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r
        x_stride_c N eps BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, rms_norm_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.div, ComparableDType.lt, FloatDType.cast,
        FloatDType.ofWithBot, FloatDType.toWithBot]

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
active input lanes hold `xs` (the `X` row) and `ws` (the weights). This is the
`hrun` obligation of the `⊨` headline. `0 < y_stride_c` makes the strided
output addresses lane-injective (the scatter readback needs it). -/
theorem rms_norm_kernel_region_run
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (hyc : 0 < y_stride_c)
    (s₀ : BlockState) (xs ws : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < N →
      s₀.readMem X (s₀.pids 0 * x_stride_r + j.val * x_stride_c) = xs j)
    (hw : ∀ j : Fin BLOCK_SIZE, j.val < N →
      s₀.readMem W j.val = ws j) :
    ∃ s1, exec ((rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r
          x_stride_c N eps BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < N →
          s1.readMem Y (s₀.pids 0 * y_stride_r + j.val * y_stride_c)
            = rmsNormSpec N BLOCK_SIZE eps xs ws j)
      ∧ (∀ r o,
          (r ≠ Y ∨ ∀ j : Fin BLOCK_SIZE, j.val < N →
            o ≠ s₀.pids 0 * y_stride_r + j.val * y_stride_c) →
          s1.mem r o = s₀.mem r o) := by
  have hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        s₀.pids 0 * y_stride_r + i.val * y_stride_c) := by
    intro a b h
    simp only at h
    exact Fin.ext (Nat.eq_of_mul_eq_mul_right hyc (Nat.add_left_cancel h))
  obtain ⟨s1, hs1⟩ := rms_norm_kernel_exec_isSome Y X W y_stride_r y_stride_c
    x_stride_r x_stride_c N BLOCK_SIZE eps s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := rms_norm_kernel_correct Y X W y_stride_r y_stride_c x_stride_r
      x_stride_c N BLOCK_SIZE eps s₀ s1 hOutInj hs1 j
    simp only [hj, if_pos] at h
    rw [h]
    exact rmsNormSpec_congr N BLOCK_SIZE eps _ xs _ ws
      (fun k hk => hx k hk) (fun k hk => hw k hk) j hj
  · refine rms_norm_kernel_frame Y X W y_stride_r y_stride_c x_stride_r
      x_stride_c N BLOCK_SIZE eps s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads with `other`, dtype casts, sum reduction, masked
store). -/
theorem rms_norm_kernel_flattenOk
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) :
    ((rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
        N eps BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [rms_norm_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all eleven
statements — eight are memory-silent (`program_id`, the pointer/mask/index
staging, the reduction and register arithmetic) — and reduces the three masked
accesses (strided row load of `X`, weight load of `W`, strided row store to
`Y`) to the **lane-wise** bounds hypotheses: every *active* lane's address is
below the region bound of the buffer it touches. -/
theorem rms_norm_kernel_traceSafe
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hin1 : ∀ j : Fin BLOCK_SIZE, j.val < N →
      s.pids 0 * x_stride_r + j.val * x_stride_c < bounds X)
    (hin2 : ∀ j : Fin BLOCK_SIZE, j.val < N → j.val < bounds W)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < N →
      s.pids 0 * y_stride_r + j.val * y_stride_c < bounds Y) :
    Kernel.TraceSafe bounds
      ((rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
        N eps BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [rms_norm_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.div,
    ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
    FloatDType.toWithBot,
    Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin1 a ha, fun a ha => hin2 a ha, fun a ha => hout a ha⟩

/-- `rms_norm_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring): the input
  matrix `X`, the per-column weights `W`, the output matrix `Y`;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`write` — **per-lane strided windows**: program `pid` reads its `X`
  row at `pid * x_stride_r + j * x_stride_c` and writes its `Y` row at
  `pid * y_stride_r + j * y_stride_c` (the host-side one-program-per-row
  launch convention with general row/column strides);
* `read2` — the weight window is **absolute and pid-independent**: every
  program reads `W[j]` (the host passes the same weight vector to all rows);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(N)`) carry no obligations on either side. The
  store mask equals the load mask, so `writeMask` keeps its default.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and mask are constant in `pid₁` (the headline's `∀ pid₁` quantification is
vacuous but honest). The windows and mask are declared, not parsed from the
kernel; the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
def rmsNormIO (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₂ where
  kernel := rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
    N eps BLOCK_SIZE
  in1 := X
  in2 := W
  out := Y
  B := BLOCK_SIZE
  read1 := fun pid _ j => pid * x_stride_r + j.val * x_stride_c
  read2 := fun _ _ j => j.val
  write := fun pid _ j => pid * y_stride_r + j.val * y_stride_c
  mask := fun _ _ j => j.val < N

/-- **The headline**: `rms_norm_kernel` implements the exact RMS normalization
over the active row prefix on its masked strided IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active input lanes hold `xs`
(the `X` row) and `ws` (the weights), the translated pointer kernel
terminates, every active output lane `j` holds
`rmsNormSpec N BLOCK_SIZE eps xs ws j = (xs j · rrms) · ws j`, and every other
memory cell is unchanged. `0 < y_stride_c` is required: distinct lanes must
scatter to distinct output cells (a zero column stride would self-clobber the
row). No `0 < BLOCK_SIZE` side condition: the kernel's only reduction is a
`sum`, total on empty tiles. Proof: `Masked2DKernelIO₂.Implements.intro`
assembles the region-model masked triple with the flat-memory bridge side
conditions. -/
specification rms_norm_kernel_correctness
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (hyc : 0 < y_stride_c) :
    rmsNormIO Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps
        BLOCK_SIZE ⊨
      fun _ _ xs ws i => rmsNormSpec N BLOCK_SIZE eps xs ws i := by
  refine Masked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact rms_norm_kernel_flattenOk Y X W y_stride_r y_stride_c x_stride_r
      x_stride_c N BLOCK_SIZE eps
  · intro bounds s h1 h2 h3 _
    exact rms_norm_kernel_traceSafe Y X W y_stride_r y_stride_c x_stride_r
      x_stride_c N BLOCK_SIZE eps bounds s h1 h2 h3
  · intro s₀ xs ws hx hw
    obtain ⟨s1, hexec, hval, hframe⟩ := rms_norm_kernel_region_run Y X W
      y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE eps hyc s₀
      xs ws hx hw
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.RmsNormTriton
