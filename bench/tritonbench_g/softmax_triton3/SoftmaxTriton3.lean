import VeriTile.Triton

/-!
# `softmax_triton3` — strict per-kernel correctness

`softmax_kernel` is a row-wise stable softmax with an optional additive mask:
program `row_idx` loads one row (masked by `col_offsets < n_cols`, masked lanes
read as `-inf`), subtracts the row max, optionally (`mask_ptr is not None`, here
the `HAS_MASK` constexpr) adds a per-lane additive mask, exponentiates, divides
by the row sum, and stores back masked by `col_offsets < n_cols`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`softmax_kernel[grid](...)`, the one-program-per-row grid
(or the large-input `BLOCK_M` tiled grid), the `BLOCK_SIZE = next_power_of_2`
choice, `num_warps` autotuning, and how the runtime composes per-row writes) is
the *trusted boundary*, not a proof obligation here. Because `row_idx`/`pid` is
universally quantified, the per-program statement covers every row of the grid.

## Proof architecture

```
softmax_kernel_correctness            ← TOP THEOREM, HAS_MASK = false
                                        (softmaxIO ⊨ softmaxSpec … none)
softmax_kernel_masked_correctness     ← TOP THEOREM, HAS_MASK = true
                                        (softmaxMaskedIO ⊨ softmaxSpec … (some ms))
  ├─ softmax_kernel_flattenOk                 bridge fragment membership (both branches)
  ├─ softmax_kernel_traceSafe /               per-execution lane-wise safety walk
  │  softmax_kernel_masked_traceSafe
  └─ softmax_kernel_region_run /              region-model masked Hoare triple
     softmax_kernel_masked_region_run
       ├─ softmax_kernel_exec_isSome          termination (both branches)
       ├─ softmax_kernel_correct              ← algorithm-layer readback per lane
       │    └─ softmaxSpec_congr /              only active lanes feed the spec
       │       softmaxSpec_mask_congr
       └─ softmax_kernel_frame                masked scatter frame (both branches)
```

The two headlines are the masked Hoare-triple combinators — one per value of
the `HAS_MASK` constexpr (the Python `mask_ptr is not None` compile-time
branch), each dimension-general:

* `HAS_MASK = false`: `softmaxIO … ⊨ softmaxSpec n_cols BLOCK_SIZE none`
  (`MaskedKernelIO₁.Implements`) — one input row in, one output row out.
* `HAS_MASK = true`: `softmaxMaskedIO … ⊨ fun xs ms i =>
  softmaxSpec n_cols BLOCK_SIZE (some ms) xs i` (`MaskedKernelIO₂.Implements`)
  — the additive-mask row is a genuine second input buffer.

In both: for every disjoint flat placement of the declared buffers, every
program id all of whose *active* lanes (`j < n_cols`) are in bounds, and every
launch state whose active input lanes hold the quantified rows, the translated
pointer kernel terminates, every active output-row lane holds `softmaxSpec`,
and every other memory cell is unchanged.

The spec `softmaxSpec` is the exact stable softmax over the active prefix:
`reduceMax` over the masked row, lane-wise `exp(row - max [+ mask])`,
`reduceSum` for the denominator, then the lane-wise quotient.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); the approximate `tl.exp`
is modeled as exact `WithBot.realExp`; the manual `num_warps` heuristic is not
modeled. The `.to(tl.float32)` casts on the loaded row and mask reduce to the
identity at the algorithm layer (post-erasure all dtypes unify to `ℝ`). The
reduction runs over the full `BLOCK_SIZE` block, but masked input lanes load `⊥`
(matching `other=-float("inf")`) and masked mask lanes load `0`, so the
reduction-over-padded-block matches the upstream semantics. No output/input
disjointness is assumed in the region model: the row is read into registers
before the masked scatter.
-/

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3

open VeriTile.Triton

/-- Faithful transcription of `softmax_triton3.py`'s `softmax_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter.
- Python `mask_ptr is not None` -> Lean `HAS_MASK : Bool` constexpr gate.
- Python `.to(tl.float32)` casts are represented explicitly in the Compute
  layer; the algorithm-layer theorem observes their Real projection. -/
def softmax_kernel
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ComputeKernel := triton {
  row_idx = tl.program_id(0)
  row_start_ptr = input_ptr + row_idx * $(row_stride)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  input_ptrs = row_start_ptr + col_offsets
  row = (tl.load(input_ptrs, mask=col_offsets < $(n_cols), other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  if HAS_MASK {
    mask_ptrs = (mask_ptr + (row_idx * $(row_stride))) + col_offsets
    mask = (tl.load(mask_ptrs, mask=col_offsets < $(n_cols), other=0)).to(tl.float32)
    row_minus_max = row_minus_max + mask
  }
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  output_row_start_ptr = output_ptr + row_idx * $(row_stride)
  output_ptrs = output_row_start_ptr + col_offsets
  tl.store(output_ptrs, softmax_output, mask=col_offsets < $(n_cols))
}

/-- Masked input row tile used by `softmax_kernel`: lane `j < n_cols` holds
`xs j`, masked lanes are `⊥`, matching `other=-float("inf")`. -/
noncomputable def softmaxInputTile (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < n_cols then some (xs idx.1) else none }

/-- Optional additive-mask row tile: lane `j < n_cols` holds `ms j`, inactive
lanes are `0`, matching `tl.load(..., other=0)` (the final store is still
masked by `n_cols`). -/
noncomputable def softmaxMaskTile (n_cols BLOCK_SIZE : Nat)
    (ms : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < n_cols then some (ms idx.1) else some 0 }

/-- Exact stable-softmax value computed by `softmax_kernel` at lane `idx`, as a
pure function of the active row prefix `xs j`, `j < n_cols`, and the optional
additive-mask row `mask?` (`some ms` ↔ the `HAS_MASK` branch; masked input
lanes enter the reductions as `⊥`, masked mask lanes as `0`). -/
noncomputable def softmaxSpec (n_cols BLOCK_SIZE : Nat)
    (mask? : Option (Fin BLOCK_SIZE → ℝ)) (xs : Fin BLOCK_SIZE → ℝ)
    (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile n_cols BLOCK_SIZE xs
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let shifted :=
        match mask? with
        | some ms =>
            Tile.bop (NumericDType.add .real) (Broadcast.consSame Broadcast.nil)
              shifted (softmaxMaskTile n_cols BLOCK_SIZE ms)
        | none => shifted
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (idx, PUnit.unit))
  | none => 0

/-- `softmaxSpec` only reads the active lanes of its input row: two rows
agreeing below `n_cols` yield the same softmax value (masked lanes are `⊥` in
the tile either way). -/
theorem softmaxSpec_congr (n_cols BLOCK_SIZE : Nat)
    (mask? : Option (Fin BLOCK_SIZE → ℝ)) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → xs j = ys j) (i : Fin BLOCK_SIZE) :
    softmaxSpec n_cols BLOCK_SIZE mask? xs i
      = softmaxSpec n_cols BLOCK_SIZE mask? ys i := by
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

/-- `softmaxSpec` only reads the active lanes of the additive-mask row: two
mask rows agreeing below `n_cols` yield the same softmax value (inactive mask
lanes are `0` in the tile either way). -/
theorem softmaxSpec_mask_congr (n_cols BLOCK_SIZE : Nat)
    (xs ms₁ ms₂ : Fin BLOCK_SIZE → ℝ)
    (h : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → ms₁ j = ms₂ j) (i : Fin BLOCK_SIZE) :
    softmaxSpec n_cols BLOCK_SIZE (some ms₁) xs i
      = softmaxSpec n_cols BLOCK_SIZE (some ms₂) xs i := by
  have htile : softmaxMaskTile n_cols BLOCK_SIZE ms₁
      = softmaxMaskTile n_cols BLOCK_SIZE ms₂ := by
    unfold softmaxMaskTile
    congr 1
    funext idx
    by_cases hj : idx.1.val < n_cols
    · simp only [if_pos hj, h idx.1 hj]
    · simp only [if_neg hj]
  simp only [softmaxSpec, htile]

/-- Algorithm-layer cellwise correctness for `softmax_kernel`: in-bounds lanes
hold `softmaxSpec` of the loaded row (and, under `HAS_MASK`, the loaded mask
row), out-of-bounds lanes are preserved. -/
theorem softmax_kernel_correct
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool)
    (s s' : BlockState)
    (hExec : exec (softmax_kernel output_ptr input_ptr mask_ptr row_stride
          n_cols BLOCK_SIZE HAS_MASK) s = some s') :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := s.pid * row_stride + i.val
      s'.readMem output_ptr outAddr =
        if i.val < n_cols then
          softmaxSpec n_cols BLOCK_SIZE
            (cond HAS_MASK
              (some (fun j => s.readMem mask_ptr (s.pid * row_stride + j.val)))
              none)
            (fun j => s.readMem input_ptr (s.pid * row_stride + j.val)) i
        else s.readMem output_ptr outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · cases HAS_MASK <;>
      simp [exec, softmax_kernel, stepStmts, stepStmt, evalOp.eq_def,
            Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
            Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
            TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
            NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
            ComparableDType.lt, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
            FloatDType.cast, hB] at hExec
    · subst s'
      simp [BlockState.pid_eq]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
            (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
      by_cases hi : i.val < n_cols
      · simp [hi, softmaxSpec, softmaxInputTile,
              Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
              TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
              NumericDType.sub, hB]
        congr
      · simp [hi]
    · subst s'
      simp [BlockState.pid_eq]
      rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
            (BlockState.tileIndex1d_base_offset_injective _) (i, PUnit.unit)]
      by_cases hi : i.val < n_cols
      · simp [hi, softmaxSpec, softmaxInputTile, softmaxMaskTile,
              Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
              TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
              NumericDType.add, NumericDType.sub, hB]
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
lanes of the output row itself — is preserved by the run (both `HAS_MASK`
branches; the mask load writes no memory). -/
private theorem softmax_kernel_frame
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool)
    (hB : 0 < BLOCK_SIZE) (s s1 : BlockState)
    (hExec : exec ((softmax_kernel output_ptr input_ptr mask_ptr row_stride
        n_cols BLOCK_SIZE HAS_MASK).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(output_ptr = r ∧ s.pid * row_stride + i.val = o)) :
    s1.mem r o = s.mem r o := by
  cases HAS_MASK <;>
    (simp [exec, softmax_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          FloatDType.cast,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hB] at hExec
     subst s1
     refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
     intro k _ hmk hc
     exact hmiss k.1 (by simpa using hmk) hc)

set_option maxHeartbeats 1600000 in
/-- Termination: the kernel executes to completion from any state (both
`HAS_MASK` branches). `0 < BLOCK_SIZE` is required because the `max` reduce
(like `Finset.sup'`) is only defined on non-empty axes. -/
private theorem softmax_kernel_exec_isSome
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool)
    (hB : 0 < BLOCK_SIZE) (s : BlockState) :
    ∃ s1, exec ((softmax_kernel output_ptr input_ptr mask_ptr row_stride
        n_cols BLOCK_SIZE HAS_MASK).toAlgKernel) s = some s1 := by
  cases HAS_MASK <;>
    simp [exec, softmax_kernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
          evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          FloatDType.cast,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, hB]

/-- **The region-model masked Hoare triple, `HAS_MASK = false`** — termination,
active-lane output values, and frame off the active output lanes, from any
launch state whose input row is loaded at the **active lanes only**
(`j < n_cols`). This is the `hrun` obligation of the unmasked-branch `⊨`
headline. -/
theorem softmax_kernel_region_run
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s₀ : BlockState) (xs : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem input_ptr (s₀.pid * row_stride + j.val) = xs j) :
    ∃ s1, exec ((softmax_kernel output_ptr input_ptr mask_ptr row_stride
          n_cols BLOCK_SIZE Bool.false).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem output_ptr (s₀.pid * row_stride + j.val)
            = softmaxSpec n_cols BLOCK_SIZE none xs j)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * row_stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := softmax_kernel_exec_isSome output_ptr input_ptr mask_ptr
    row_stride n_cols BLOCK_SIZE Bool.false hB s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := softmax_kernel_correct output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.false s₀ s1 hs1 j
    simp only [hj, if_pos] at h
    rw [h]
    exact softmaxSpec_congr n_cols BLOCK_SIZE none _ xs (fun k hk => hx k hk) j
  · refine softmax_kernel_frame output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.false hB s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- **The region-model masked Hoare triple, `HAS_MASK = true`** — same as
`softmax_kernel_region_run`, from any launch state whose input row **and**
additive-mask row are loaded at the active lanes only. This is the `hrun`
obligation of the masked-branch `⊨` headline. -/
theorem softmax_kernel_masked_region_run
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s₀ : BlockState) (xs ms : Fin BLOCK_SIZE → ℝ)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem input_ptr (s₀.pid * row_stride + j.val) = xs j)
    (hm : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s₀.readMem mask_ptr (s₀.pid * row_stride + j.val) = ms j) :
    ∃ s1, exec ((softmax_kernel output_ptr input_ptr mask_ptr row_stride
          n_cols BLOCK_SIZE Bool.true).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < n_cols →
          s1.readMem output_ptr (s₀.pid * row_stride + j.val)
            = softmaxSpec n_cols BLOCK_SIZE (some ms) xs j)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
            o ≠ s₀.pid * row_stride + j.val) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := softmax_kernel_exec_isSome output_ptr input_ptr mask_ptr
    row_stride n_cols BLOCK_SIZE Bool.true hB s₀
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := softmax_kernel_correct output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.true s₀ s1 hs1 j
    simp only [hj, if_pos] at h
    rw [h]
    exact (softmaxSpec_congr n_cols BLOCK_SIZE _ _ xs (fun k hk => hx k hk) j).trans
      (softmaxSpec_mask_congr n_cols BLOCK_SIZE xs _ ms (fun k hk => hm k hk) j)
  · refine softmax_kernel_frame output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.true hB s₀ s1 hs1 r o (fun i hi ⟨hr, ho⟩ => ?_)
    rcases hcond with hne | hno
    · exact hne hr.symm
    · exact hno i hi ho.symm

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, masked loads with `other`, dtype casts, the constexpr `if`,
reductions, masked store) — both `HAS_MASK` branches. -/
theorem softmax_kernel_flattenOk
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) (HAS_MASK : Bool) :
    ((softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE HAS_MASK).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  cases HAS_MASK <;>
    simp [softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
      Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk, `HAS_MASK = false`: one computational unfold
walks all the statements — the pointer/index staging, the reductions, and the
register arithmetic are memory-silent, the constexpr `if` is inert — and
reduces the two masked accesses (row load, row store) to the **lane-wise**
bounds hypotheses: every *active* lane's address is below the region bound. -/
theorem softmax_kernel_traceSafe
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds input_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE Bool.false).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hout
  simp [softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk, `HAS_MASK = true`: as
`softmax_kernel_traceSafe`, plus the mask row load — three masked accesses
(row load, mask load, row store), each reduced to its lane-wise bounds
hypothesis. -/
theorem softmax_kernel_masked_traceSafe
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds input_ptr)
    (hmask : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds mask_ptr)
    (hout : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pid * row_stride + j.val < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
        BLOCK_SIZE Bool.true).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp only [BlockState.pid_eq] at hin hmask hout
  simp [softmax_kernel, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, FloatDType.cast,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmts, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin a ha, fun a ha => hmask a ha, fun a ha => hout a ha⟩

section Unmasked

open scoped VeriTile.Triton.MaskedKernelIO₁

/-- `softmax_kernel`'s masked **IO signature for the `HAS_MASK = false`
branch** — the whole kernel-specific audit surface of the unmasked `⊨`
headline:

* `inp`/`out` — which buffer is which argument (the wiring); `mask_ptr` is
  wired into the kernel but never touched on this branch, so it is not an
  interface buffer;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads and writes its row at
  `pid * row_stride` (the host-side one-program-per-row launch convention;
  this kernel uses one `row_stride` for both buffers);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either
  side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def softmaxIO (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
    BLOCK_SIZE Bool.false
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * row_stride
  write := fun pid => pid * row_stride
  mask := fun _ j => j.val < n_cols

/-- **Headline, `HAS_MASK = false`** (the Python `mask_ptr is None` compile-time
branch): `softmax_kernel` implements the exact stable softmax over the active
row prefix on its masked IO signature — for every disjoint flat placement of
the two buffers, every program id whose active lanes are in bounds, and every
launch state whose active input-row lanes hold `xs`, the translated pointer
kernel terminates, every active output-row lane `j` holds
`softmaxSpec n_cols BLOCK_SIZE none xs j`, and every other memory cell is
unchanged. `0 < BLOCK_SIZE` is required: the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof: `Implements.intro`
assembles the region-model masked triple with the bridge side conditions. -/
specification softmax_kernel_correctness
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    softmaxIO output_ptr input_ptr mask_ptr row_stride n_cols BLOCK_SIZE ⊨
      fun xs i => softmaxSpec n_cols BLOCK_SIZE none xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact softmax_kernel_flattenOk output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.false
  · intro bounds s h1 h2 _
    exact softmax_kernel_traceSafe output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE hB bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := softmax_kernel_region_run output_ptr
      input_ptr mask_ptr row_stride n_cols BLOCK_SIZE hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end Unmasked

section Masked

open scoped VeriTile.Triton.MaskedKernelIO₂

/-- `softmax_kernel`'s masked **IO signature for the `HAS_MASK = true`
branch** — as `softmaxIO`, but the additive-mask row is a genuine second input
buffer: `in1` is the input matrix, `in2` the mask matrix (both read at
`pid * row_stride`), `out` the output matrix. Active lanes are `j < n_cols`
for every program, as on the unmasked branch. -/
def softmaxMaskedIO (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
    BLOCK_SIZE Bool.true
  in1 := input_ptr
  in2 := mask_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * row_stride
  read2 := fun pid => pid * row_stride
  write := fun pid => pid * row_stride
  mask := fun _ j => j.val < n_cols

/-- **Headline, `HAS_MASK = true`** (the Python `mask_ptr is not None`
compile-time branch): `softmax_kernel` implements the exact stable softmax
with the additive mask on its masked IO signature — for every disjoint flat
placement of the three buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs` and
active mask-row lanes hold `ms`, the translated pointer kernel terminates,
every active output-row lane `j` holds
`softmaxSpec n_cols BLOCK_SIZE (some ms) xs j`, and every other memory cell is
unchanged. `0 < BLOCK_SIZE` is required: the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof: `Implements.intro`
assembles the region-model masked triple with the bridge side conditions. -/
specification softmax_kernel_masked_correctness
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    softmaxMaskedIO output_ptr input_ptr mask_ptr row_stride n_cols BLOCK_SIZE ⊨
      fun xs ms i => softmaxSpec n_cols BLOCK_SIZE (some ms) xs i := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact softmax_kernel_flattenOk output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.true
  · intro bounds s h1 h2 h3 _
    exact softmax_kernel_masked_traceSafe output_ptr input_ptr mask_ptr
      row_stride n_cols BLOCK_SIZE hB bounds s h1 h2 h3
  · intro s₀ xs ms hx hm
    obtain ⟨s1, hexec, hval, hframe⟩ := softmax_kernel_masked_region_run
      output_ptr input_ptr mask_ptr row_stride n_cols BLOCK_SIZE hB s₀ xs ms hx hm
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end Masked

end VeriTile.Bench.TritonBenchG.SoftmaxTriton3
