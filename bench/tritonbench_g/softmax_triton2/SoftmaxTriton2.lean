import VeriTile.Triton

/-!
# `softmax_triton2` — strict per-kernel correctness

`softmax_kernel` is a row-wise stable softmax: program `row_idx` loads one row
(block `[0, BLOCK_SIZE)`, masked by `col_offsets < n_cols`, with masked lanes
read as `-inf`), subtracts the row max, exponentiates, divides by the row sum,
and stores the normalized row back, masked by `col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`softmax_kernel[(n_rows,)](...)`, the one-program-per-row
grid, the `BLOCK_SIZE = next_power_of_2(n_cols)` choice, `num_warps` autotuning,
and how the runtime composes per-row writes into one buffer) is the *trusted
boundary*, not a proof obligation here. Because `row_idx`/`pid` is universally
quantified, the per-program statement covers every row of the grid.

## Proof architecture

```
softmax_kernel_correctness                    ← TOP THEOREM (softmaxIO ⊨ softmaxSpec)
  ├─ softmax_kernel_flattenOk                 bridge fragment membership
  ├─ softmax_kernel_traceSafe                 per-execution lane-wise safety walk
  └─ softmax_kernel_region_run                region-model masked Hoare triple
       ├─ softmax_kernel_exec_isSome          termination
       ├─ softmax_kernel_correct              ← algorithm-layer readback per lane
       │    └─ softmaxSpec_congr              only active lanes feed the spec
       └─ softmax_kernel_frame                masked scatter frame
```

The headline is the masked Hoare-triple combinator `softmaxIO … ⊨ softmaxSpec`
(`MaskedKernelIO₁.Implements`): for every disjoint flat placement of the two
buffers, every program id all of whose *active* lanes (`j < n_cols`) are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane holds
`softmaxSpec n_cols BLOCK_SIZE xs j`, and every other memory cell is unchanged.

The spec `softmaxSpec` is the exact stable softmax over the active prefix:
`reduceMax` over the masked row, lane-wise `exp(row - max)`, `reduceSum` for
the denominator, then the lane-wise quotient.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the approximate `tl.exp`
is modeled as exact `WithBot.realExp`; the manual `num_warps` heuristic is not
modeled. The reduction runs over the full `BLOCK_SIZE` block, but masked lanes
load `⊥` (matching `other=-float("inf")`), so the reduction-over-padded-block
matches the upstream `-inf` semantics. No output/input disjointness is assumed
in the region model: the row is read into registers before the masked scatter.
-/

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton2

open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Faithful 1:1 transcription of `softmax_triton2.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter. -/
def softmax_kernel
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(input_row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(output_row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

/-- Masked input row tile used by `softmax_kernel`: lane `j < n_cols` holds
`xs j`, masked lanes are `⊥`, matching `other=-float("inf")`. -/
noncomputable def softmaxInputTile (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < n_cols then some (xs idx.1) else none }

/-- Exact stable-softmax value computed by the kernel at lane `idx`, as a pure
function of the active row prefix `xs j`, `j < n_cols` (masked lanes enter the
reductions as `⊥`, neutral for both `max` and the `exp`-sum). -/
noncomputable def softmaxSpec (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile n_cols BLOCK_SIZE xs
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (idx, PUnit.unit))
  | none => 0

/-- `softmaxSpec` only reads the active lanes of its input: two rows agreeing
below `n_cols` yield the same softmax value (masked lanes are `⊥` in the tile
either way). -/
theorem softmaxSpec_congr (n_cols BLOCK_SIZE : Nat) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → xs j = ys j) (i : Fin BLOCK_SIZE) :
    softmaxSpec n_cols BLOCK_SIZE xs i = softmaxSpec n_cols BLOCK_SIZE ys i := by
  have htile : softmaxInputTile n_cols BLOCK_SIZE xs
      = softmaxInputTile n_cols BLOCK_SIZE ys := by
    unfold softmaxInputTile
    congr 1
    funext idx
    by_cases hj : idx.1.val < n_cols
    · simp only [if_pos hj, h idx.1 hj]
    · simp only [if_neg hj]
  unfold softmaxSpec
  rw [htile]

/-- Algorithm-layer cellwise correctness for `softmax_kernel`: in-bounds lanes
hold `softmaxSpec` of the loaded row, out-of-bounds lanes are preserved. -/
theorem softmax_kernel_correct
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
          n_cols BLOCK_SIZE) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := s.pid * output_row_stride + i.val
      s'.readMem output_ptr outAddr =
        if i.val < n_cols then
          softmaxSpec n_cols BLOCK_SIZE
            (fun j => s.readMem input_ptr (s.pid * input_row_stride + j.val)) i
        else s.readMem output_ptr outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, softmax_kernel, stepStmts, stepStmt, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hB] at hExec
    subst s'
    simp [BlockState.pid_eq]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
          (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
    by_cases hi : i.val < n_cols
    · simp [hi, softmaxSpec, softmaxInputTile, Tile.reduceMax, Tile.reduceMaxDrop,
            Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
            TileShape.eraseAxis, TileShape.insertAxisIndex, hB]
      congr
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
store — every cell of every region other than `output_ptr`, and the *inactive*
lanes of the output row itself — is preserved by the run. -/
private theorem softmax_kernel_frame
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s s1 : BlockState)
    (hExec : exec ((softmax_kernel output_ptr input_ptr input_row_stride
        output_row_stride n_cols BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(output_ptr = r ∧ s.pid * output_row_stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, softmax_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, hB] at hExec
  subst s1
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa using hmk) hc

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state. `0 <
BLOCK_SIZE` is required because the `max` reduce (like `Finset.sup'`) is only
defined on non-empty axes. -/
private theorem softmax_kernel_exec_isSome
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s : BlockState) :
    ∃ s1, exec ((softmax_kernel output_ptr input_ptr input_row_stride
        output_row_stride n_cols BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, softmax_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, hB]

/-- **The region-model masked Hoare triple** — termination, active-lane output
values, and frame off the active output lanes, from any launch state whose
input row is loaded at the **active lanes only** (`j < n_cols`). This is the
`hrun` obligation of the `⊨` headline. -/
theorem softmax_kernel_region_run
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem input_ptr (s₀.pid * input_row_stride + j.val) = xs j) :
    ∃ s1, exec ((softmax_kernel output_ptr input_ptr input_row_stride
          output_row_stride n_cols BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem output_ptr (s₀.pid * output_row_stride + j.val)
            = softmaxSpec n_cols BLOCK_SIZE xs j)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * output_row_stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := softmax_kernel_exec_isSome output_ptr input_ptr
    input_row_stride output_row_stride n_cols BLOCK_SIZE hB s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := softmax_kernel_correct output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE s₀ s1 hs1 j
    simp only [hj, if_pos] at h
    rw [h]
    exact softmaxSpec_congr n_cols BLOCK_SIZE _ xs (fun k hk => hx k hk) j
  · refine softmax_kernel_frame output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE hB s₀ s1 hs1 r o
      (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked load with `other`, reductions, masked store). -/
theorem softmax_kernel_flattenOk
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    ((softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all twelve
statements — ten are memory-silent (`program_id`, the pointer/index staging,
the reductions, and the register arithmetic) — and reduces the two masked
accesses (row load, row store) to the **lane-wise** bounds hypotheses: every
*active* lane's address is below the region bound. -/
theorem softmax_kernel_traceSafe
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * input_row_stride + j.val < bounds input_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * output_row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((softmax_kernel output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

/-- `softmax_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads its row at `pid * input_row_stride` and
  writes it at `pid * output_row_stride` (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either
  side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def softmaxIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := softmax_kernel output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * input_row_stride
  write := fun pid => pid * output_row_stride
  mask := fun _ j => j.val < n_cols

/-- **The headline**: `softmax_kernel` implements the exact stable softmax
over the active row prefix on its masked IO signature — for every disjoint
flat placement of the two buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane `j` holds
`softmaxSpec n_cols BLOCK_SIZE xs j`, and every other memory cell is
unchanged. `0 < BLOCK_SIZE` is required: the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof: `Implements.intro`
assembles the region-model masked triple with the bridge side conditions. -/
specification softmax_kernel_correctness
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    softmaxIO output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE ⊨
      fun xs i => softmaxSpec n_cols BLOCK_SIZE xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact softmax_kernel_flattenOk output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact softmax_kernel_traceSafe output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE hB bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := softmax_kernel_region_run output_ptr
      input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end VeriTile.Bench.TritonBenchG.SoftmaxTriton2
