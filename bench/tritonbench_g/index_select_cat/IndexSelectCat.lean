import VeriTile.Triton

/-!
# `index_select_cat` — strict per-kernel correctness

`index_select_cat_fwd_kernel` is the forward index-select/cat: for the
`(pid0, pid1)` tile of `(index, column)` blocks it gathers the source row from
`index_ptr`, loads `source[index[i], col]`, and stores it contiguously into
`output[i, col]`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`index_select_cat_fwd_kernel[grid](...)`, the 2D grid
sizes `cdiv(num_indices, BLOCK_SIZE_INDEX)` × `cdiv(num_cols, BLOCK_SIZE_COL)`,
the index-truncation logic, and how the runtime composes per-program writes) is
the *trusted boundary*, not a proof obligation here. Because the program ids
`pids 0`, `pids 1` are universally quantified, the per-program statement covers
every program of the grid.

The headline is stated on the kernel's gather IO signature
`indexSelectCatIO` (`GatherMasked2DKernelIO₁`, the index-tile genre's skin):
which buffer is which argument — data `source_ptr` (`inp`), row-index tile
`index_ptr` (`idxbuf`, the `.nat` channel), destination `output_ptr` (`out`) —
the **static** index-tile window `readx = pid0 * BI + j / BC`, the
**data-dependent** gather source `read … ids j = ids j * stride0 +
(pid1 * BC + j % BC) * stride1`, and the **static** contiguous destination
`write = (pid0 * BI + j / BC) * stride0 + (pid1 * BC + j % BC) * stride1`.
`⊨` (`GatherMasked2DKernelIO₁.Implements`) is the audit-once Hoare-triple
combinator: for **every** disjoint flat placement of the three buffers,
**every** pair of program ids whose active lanes are in bounds, and **every**
launch state whose windows hold the index tile `ids` and the data tile `xs`,
the translated pointer kernel terminates, every write-active lane's output cell
holds `xs j`, and every cell off the write window is unchanged.

The skin's lane space is one-dimensional, so the kernel's `[BI, BC]` store tile
is carried as the flattened lane space `Fin (BI * BC)` — lane `j` is tile cell
`(j / BC, j % BC)` (`Lane2D.decode`, with `Lane2D.encode` the inverse). This is a
bijection, so nothing is lost: every tile cell is some lane.

## Proof architecture

```
index_select_cat_fwd_kernel_correctness           ← TOP SPECIFICATION (indexSelectCatIO ⊨ gather)
  ├─ index_select_cat_fwd_kernel_flattenOk         bridge fragment membership
  ├─ index_select_cat_fwd_kernel_traceSafe         per-execution lane-wise safety walk
  └─ index_select_cat_fwd_kernel_region_run        region-model gather Hoare triple
       ├─ index_select_cat_fwd_kernel_exec_isSome  termination
       ├─ index_select_cat_fwd_kernel_correct_of_exec  executed-state readback per active cell
       │    └─ index_select_cat_fwd_kernel_correct  ← algorithm-layer readback per active cell
       └─ index_select_cat_fwd_kernel_frame        store-cell frame
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The gathered source row
offset is data-dependent (read from `index_ptr`) but only feeds the *value*, not
the destination; the destination `i·stride0 + col·stride1` is **static**. The
skin's readback leg is guarded by the per-context `WriteInj` antecedent (no two
write-active lanes share a destination) because the skin's `write` window may in
general eat the loaded index tile; on this pure-gather side that antecedent is
about the *static* contiguous output window only — it is exactly the old
`hOutInj` side condition, now demanded only at the write-active lanes and
carried inside `⊨` rather than as a headline hypothesis, so the headline is
free of scope-narrowing hypotheses. The one genuine hypothesis is
`0 < BLOCK_SIZE_COL`: the index-tile load runs over the `[BI]` tile
independently of `BC`, so with `BC = 0` the lane space `Fin (BI * BC)` is empty
and carries no bound for it. The statement is scoped to *active* lanes
(`row < num_indices` and `col < num_cols`).
-/

namespace VeriTile.Bench.TritonBenchG.IndexSelectCat

open VeriTile.Triton
open scoped VeriTile.Triton.GatherMasked2DKernelIO₁

/-- Faithful transcription of `index_select_cat.py`'s
`index_select_cat_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_INDEX: tl.constexpr` / `BLOCK_SIZE_COL: tl.constexpr`
  → Lean `Nat` parameters.
- `index_ptr` is a typed Lean region so its `tl.load` needs no extra `dtype=`
  kwarg. -/
def index_select_cat_fwd_kernel
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) :
    ComputeKernel := triton {
  pid0 = tl.program_id(axis=0)
  pid1 = tl.program_id(axis=1)
  indices = pid0 * $(BLOCK_SIZE_INDEX) + tl.arange(0, $(BLOCK_SIZE_INDEX))
  rows = tl.load(index_ptr + indices, mask=indices < $(num_indices))
  cols = pid1 * $(BLOCK_SIZE_COL) + tl.arange(0, $(BLOCK_SIZE_COL))
  source_offsets = source_ptr + rows[:, None] * $(stride0) + cols[None, :] * $(stride1)
  mask = (indices[:, None] < $(num_indices)) & (cols[None, :] < $(num_cols))
  output = tl.load(source_offsets, mask=mask)
  output_offsets = output_ptr + indices[:, None] * $(stride0) + cols[None, :] * $(stride1)
  tl.store(output_offsets, output, mask=mask)
}

/-! ### Flattened lane space

The skin's lane space is `Fin B`; the kernel's store tile is `[BI, BC]`. Lane
`j` is tile cell `(j / BC, j % BC)`, and `Lane2D.encode` is the inverse. -/

/-! ### Kernel-coupled addressing vocabulary -/

def indexBase (s : BlockState) (BLOCK_SIZE_INDEX : Nat) (i : Fin BLOCK_SIZE_INDEX) : Nat :=
  s.pids 0 * BLOCK_SIZE_INDEX + i.val

def colBase (s : BlockState) (BLOCK_SIZE_COL : Nat) (j : Fin BLOCK_SIZE_COL) : Nat :=
  s.pids 1 * BLOCK_SIZE_COL + j.val

def outputAddr (s : BlockState) (stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Nat :=
  indexBase s BLOCK_SIZE_INDEX idx.1 * stride0 +
    colBase s BLOCK_SIZE_COL idx.2.1 * stride1

def sourceAddr (s : BlockState) (index_ptr : RegionName)
    (stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Nat :=
  s.readMemValue .nat index_ptr (indexBase s BLOCK_SIZE_INDEX idx.1) * stride0 +
    colBase s BLOCK_SIZE_COL idx.2.1 * stride1

def active
    (s : BlockState) (num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Prop :=
  indexBase s BLOCK_SIZE_INDEX idx.1 < num_indices ∧
    colBase s BLOCK_SIZE_COL idx.2.1 < num_cols

instance activeDecidable
    (s : BlockState) (num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) :
    Decidable (active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx) := by
  unfold active
  infer_instance

set_option maxHeartbeats 1600000 in
/-- Algorithm-layer cellwise correctness at an **active** cell, under the
no-collision hypothesis restricted to active cells (the state-coupled form of
the skin's `WriteInj` on the static output window). -/
theorem index_select_cat_fwd_kernel_correct
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState) (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])
    (hActive : active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)
    (hNoCol : ∀ k : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL k →
      outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL k
        = outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx →
      k = idx) :
    (exec (index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL) s).map
        (fun s' => s'.readMem output_ptr
          (outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx))
      = some (s.readMem source_ptr
          (sourceAddr s index_ptr stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) := by
  obtain ⟨hIndex, hCol⟩ := hActive
  have hIndexRaw : s.pids 0 * BLOCK_SIZE_INDEX + idx.1.val < num_indices := hIndex
  have hColRaw : s.pids 1 * BLOCK_SIZE_COL + idx.2.1.val < num_cols := hCol
  have hcol : ∀ k : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      (s.pids 0 * BLOCK_SIZE_INDEX + k.1.val < num_indices ∧
        s.pids 1 * BLOCK_SIZE_COL + k.2.1.val < num_cols) →
      (s.pids 0 * BLOCK_SIZE_INDEX + k.1.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + k.2.1.val) * stride1
        = (s.pids 0 * BLOCK_SIZE_INDEX + idx.1.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + idx.2.1.val) * stride1 →
      k = idx := by
    intro k hk heq
    exact hNoCol k hk heq
  simp [exec, index_select_cat_fwd_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex]
  simp [outputAddr, sourceAddr, indexBase, colBase, BlockState.readMemValue]
  simpa [hIndexRaw, hColRaw] using
    (BlockState.scatter_readback_prop_masked_nd_of_true
      (region := output_ptr)
      (shape := [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])
      (offsetFn := fun i : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        (s.pids 0 * BLOCK_SIZE_INDEX + i.1.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + i.2.1.val) * stride1)
      (valueFn := fun i : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        WithBot.unbotD 0
          (if s.pids 0 * BLOCK_SIZE_INDEX + i.1.val < num_indices ∧
              s.pids 1 * BLOCK_SIZE_COL + i.2.1.val < num_cols then
            some (s.readMem source_ptr
              ((if s.pids 0 * BLOCK_SIZE_INDEX + i.1.val < num_indices then
                  (match s.readMemTyped TileDType.nat index_ptr.cast
                      (s.pids 0 * BLOCK_SIZE_INDEX + i.1.val) with
                    | some value => value
                    | none => BlockState.defaultCarrier TileDType.nat) * stride0
                else BlockState.defaultCarrier TileDType.nat * stride0) +
                (s.pids 1 * BLOCK_SIZE_COL + i.2.1.val) * stride1))
          else
            some (s.undef source_ptr
              ((if s.pids 0 * BLOCK_SIZE_INDEX + i.1.val < num_indices then
                  (match s.readMemTyped TileDType.nat index_ptr.cast
                      (s.pids 0 * BLOCK_SIZE_INDEX + i.1.val) with
                    | some value => value
                    | none => BlockState.defaultCarrier TileDType.nat) * stride0
                else BlockState.defaultCarrier TileDType.nat * stride0) +
                (s.pids 1 * BLOCK_SIZE_COL + i.2.1.val) * stride1))))
      (P := fun i : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        s.pids 0 * BLOCK_SIZE_INDEX + i.1.val < num_indices ∧
          s.pids 1 * BLOCK_SIZE_COL + i.2.1.val < num_cols)
      _ idx ⟨hIndexRaw, hColRaw⟩ hcol)

/-- Executed-state form of `index_select_cat_fwd_kernel_correct`. -/
theorem index_select_cat_fwd_kernel_correct_of_exec
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState) (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])
    (hActive : active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)
    (hNoCol : ∀ k : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL k →
      outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL k
        = outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx →
      k = idx)
    (s' : BlockState)
    (hExec : exec (index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL) s
      = some s') :
    s'.readMem output_ptr
        (outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)
      = s.readMem source_ptr
        (sourceAddr s index_ptr stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx) := by
  have h := index_select_cat_fwd_kernel_correct output_ptr source_ptr index_ptr
    num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL s idx
    hActive hNoCol
  rw [hExec] at h
  simpa using h

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
/-- Termination: the kernel executes to completion from any state (straight-line
masked loads plus one masked store). -/
private theorem index_select_cat_fwd_kernel_exec_isSome
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState) :
    ∃ s1, exec ((index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX
        BLOCK_SIZE_COL).toAlgKernel) s = some s1 := by
  simp [exec, index_select_cat_fwd_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.insertAxis, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively hit by the store — every cell of
every region other than `output_ptr`, and every `output_ptr` cell off the
active cells' destinations — is preserved by the run. -/
private theorem index_select_cat_fwd_kernel_frame
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s s1 : BlockState)
    (hExec : exec ((index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX
        BLOCK_SIZE_COL).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx →
      ¬(output_ptr = r ∧
        outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, index_select_cat_fwd_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k hmk ⟨hc.1, hc.2⟩

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all nine
statements — five are memory-silent (`program_id` ×2, the two window
arithmetics, the mask comparison) — and reduces the masked index-tile load, the
data-dependent masked gather load and the masked contiguous store to the three
lane-wise bounds hypotheses. -/
theorem index_select_cat_fwd_kernel_traceSafe
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ i : Fin BLOCK_SIZE_INDEX,
      s.pids 0 * BLOCK_SIZE_INDEX + i.val < num_indices →
      s.pids 0 * BLOCK_SIZE_INDEX + i.val < bounds index_ptr.cast)
    (h2 : ∀ (i : Fin BLOCK_SIZE_INDEX) (c : Fin BLOCK_SIZE_COL),
      s.pids 0 * BLOCK_SIZE_INDEX + i.val < num_indices →
      s.pids 1 * BLOCK_SIZE_COL + c.val < num_cols →
      s.readMemValue .nat index_ptr.cast (s.pids 0 * BLOCK_SIZE_INDEX + i.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + c.val) * stride1
        < bounds source_ptr)
    (h3 : ∀ (i : Fin BLOCK_SIZE_INDEX) (c : Fin BLOCK_SIZE_COL),
      s.pids 0 * BLOCK_SIZE_INDEX + i.val < num_indices →
      s.pids 1 * BLOCK_SIZE_COL + c.val < num_cols →
      (s.pids 0 * BLOCK_SIZE_INDEX + i.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + c.val) * stride1
        < bounds output_ptr) :
    Kernel.TraceSafe bounds
      ((index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX
        BLOCK_SIZE_COL).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [index_select_cat_fwd_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    Op.PointerAddressesSafeOn, Op.MemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.insertAxis, TileShape.dropInsertedIndex,
    BlockState.readMemValue]
  refine ⟨fun a ha => h1 a ha, fun a c ha hc => ?_, fun a c ha hc => h3 a c ha hc⟩
  have h := h2 a c ha hc
  simpa [BlockState.readMemValue, ha] using h

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, a masked `.nat`-channel index load, a data-dependent masked gather
load and a masked store). -/
theorem index_select_cat_fwd_kernel_flattenOk
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) :
    ((index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX
        BLOCK_SIZE_COL).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [index_select_cat_fwd_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- **The region-model gather Hoare triple** — termination, write-active-lane
values under the per-context no-collision antecedent, and frame off the store
cells, from any launch state whose windows hold the index tile `ids` and the
data tile `xs`. This is the `hrun` obligation of
`GatherMasked2DKernelIO₁.Implements.intro`; the value half reuses
`index_select_cat_fwd_kernel_correct_of_exec` (whose state-coupled gather
address collapses to `ids j * stride0 + col * stride1` under the index-tile
pin). -/
theorem index_select_cat_fwd_kernel_region_run
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BI BC : Nat)
    (s₀ : BlockState) (ids : Fin (BI * BC) → Nat) (xs : Fin (BI * BC) → ℝ)
    (hi : ∀ j : Fin (BI * BC), s₀.pids 0 * BI + j.val / BC < num_indices →
      s₀.readMemValue .nat index_ptr.cast (s₀.pids 0 * BI + j.val / BC) = ids j)
    (hx : ∀ j : Fin (BI * BC),
      (s₀.pids 0 * BI + j.val / BC < num_indices ∧
        s₀.pids 1 * BC + j.val % BC < num_cols) →
      s₀.readMem source_ptr
        (ids j * stride0 + (s₀.pids 1 * BC + j.val % BC) * stride1) = xs j) :
    ∃ s1, exec ((index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
          num_indices num_cols stride0 stride1 BI BC).toAlgKernel) s₀ = some s1
      ∧ ((∀ j k : Fin (BI * BC),
            (s₀.pids 0 * BI + j.val / BC < num_indices ∧
              s₀.pids 1 * BC + j.val % BC < num_cols) →
            (s₀.pids 0 * BI + k.val / BC < num_indices ∧
              s₀.pids 1 * BC + k.val % BC < num_cols) →
            (s₀.pids 0 * BI + j.val / BC) * stride0 +
                (s₀.pids 1 * BC + j.val % BC) * stride1
              = (s₀.pids 0 * BI + k.val / BC) * stride0 +
                (s₀.pids 1 * BC + k.val % BC) * stride1 → j = k) →
          ∀ j : Fin (BI * BC),
            (s₀.pids 0 * BI + j.val / BC < num_indices ∧
              s₀.pids 1 * BC + j.val % BC < num_cols) →
            s1.readMem output_ptr
                ((s₀.pids 0 * BI + j.val / BC) * stride0 +
                  (s₀.pids 1 * BC + j.val % BC) * stride1)
              = xs j)
      ∧ (∀ r o,
          (r ≠ output_ptr ∨ ∀ j : Fin (BI * BC),
            (s₀.pids 0 * BI + j.val / BC < num_indices ∧
              s₀.pids 1 * BC + j.val % BC < num_cols) →
            o ≠ (s₀.pids 0 * BI + j.val / BC) * stride0 +
              (s₀.pids 1 * BC + j.val % BC) * stride1) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := index_select_cat_fwd_kernel_exec_isSome output_ptr source_ptr
    index_ptr num_indices num_cols stride0 stride1 BI BC s₀
  -- the lane view of a tile cell is `active`-faithful in both directions
  have hlaneAct : ∀ k : TileIndex [BI, BC],
      active s₀ num_indices num_cols BI BC k →
      (s₀.pids 0 * BI + (Lane2D.encode k).val / BC < num_indices ∧
        s₀.pids 1 * BC + (Lane2D.encode k).val % BC < num_cols) := by
    intro k hk
    rw [Lane2D.encode_div, Lane2D.encode_mod]
    exact hk
  have hlaneAddr : ∀ k : TileIndex [BI, BC],
      (s₀.pids 0 * BI + (Lane2D.encode k).val / BC) * stride0 +
          (s₀.pids 1 * BC + (Lane2D.encode k).val % BC) * stride1
        = outputAddr s₀ stride0 stride1 BI BC k := by
    intro k
    rw [Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  refine ⟨s1, hs1, fun hinj j hj => ?_, fun r o hcond => ?_⟩
  · have hActive : active s₀ num_indices num_cols BI BC (Lane2D.decode j) := hj
    have hNoCol : ∀ k : TileIndex [BI, BC],
        active s₀ num_indices num_cols BI BC k →
        outputAddr s₀ stride0 stride1 BI BC k
          = outputAddr s₀ stride0 stride1 BI BC (Lane2D.decode j) →
        k = Lane2D.decode j := by
      intro k hk heq
      have hjk : Lane2D.encode k = j := by
        refine hinj (Lane2D.encode k) j (hlaneAct k hk) hj ?_
        rw [hlaneAddr k, heq]
        rfl
      rw [← Lane2D.decode_encode k, hjk]
    have h := index_select_cat_fwd_kernel_correct_of_exec output_ptr source_ptr
      index_ptr num_indices num_cols stride0 stride1 BI BC s₀ (Lane2D.decode j)
      hActive hNoCol s1 hs1
    have hsrc : sourceAddr s₀ index_ptr.cast stride0 stride1 BI BC (Lane2D.decode j)
        = ids j * stride0 + (s₀.pids 1 * BC + j.val % BC) * stride1 := by
      show s₀.readMemValue .nat index_ptr.cast (s₀.pids 0 * BI + j.val / BC) * stride0 +
          (s₀.pids 1 * BC + j.val % BC) * stride1
        = ids j * stride0 + (s₀.pids 1 * BC + j.val % BC) * stride1
      rw [hi j hj.1]
    have hout : outputAddr s₀ stride0 stride1 BI BC (Lane2D.decode j)
        = (s₀.pids 0 * BI + j.val / BC) * stride0 +
          (s₀.pids 1 * BC + j.val % BC) * stride1 := rfl
    rw [← hout, h, hsrc]
    exact hx j hj
  · refine index_select_cat_fwd_kernel_frame output_ptr source_ptr index_ptr
      num_indices num_cols stride0 stride1 BI BC s₀ s1 hs1 r o ?_
    intro k hk hc
    rcases hcond with hne | hno
    · exact hne hc.1.symm
    · exact hno (Lane2D.encode k) (hlaneAct k hk)
        (hc.2.symm.trans (hlaneAddr k).symm)

/-- `index_select_cat_fwd_kernel`'s gather **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`idxbuf`/`out` — which buffer is which argument (data `source_ptr`,
  `.nat` row-index channel `index_ptr`, destination `output_ptr`);
* `B = BLOCK_SIZE_INDEX * BLOCK_SIZE_COL` — the kernel's `[BI, BC]` tile,
  flattened: lane `j` is tile cell `(j / BC, j % BC)`;
* `readx` — the **static** index-tile window `pid0 * BI + j / BC`, i.e. the
  kernel's `indices` (row block of the 2D grid's first axis);
* `read` — the **data-dependent** gather source
  `ids j * stride0 + (pid1 * BC + j % BC) * stride1`: the loaded row times
  `stride0` plus the column offset, exactly the kernel's `source_offsets`;
* `write` — the **static** contiguous destination
  `(pid0 * BI + j / BC) * stride0 + (pid1 * BC + j % BC) * stride1`: the
  kernel's `output_offsets`, which ignores the loaded index tile;
* `mask` — the index-tile gate `pid0 * BI + j / BC < num_indices`;
* `readMask`/`writeMask` — the kernel's 2D `mask`, `indices < num_indices ∧
  cols < num_cols`.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking and index-channel plumbing
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
def indexSelectCatIO (output_ptr source_ptr index_ptr : RegionName)
    (num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) : GatherMasked2DKernelIO₁ where
  kernel := index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
    num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
  inp := source_ptr
  idxbuf := index_ptr
  out := output_ptr
  B := BLOCK_SIZE_INDEX * BLOCK_SIZE_COL
  readx := fun p₀ _ j => p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL
  read := fun _ p₁ ids j =>
    ids j * stride0 + (p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL) * stride1
  write := fun p₀ p₁ _ j =>
    (p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL) * stride0 +
      (p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL) * stride1
  mask := fun p₀ _ j =>
    p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL < num_indices
  readMask := fun p₀ p₁ _ j =>
    p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL < num_indices ∧
      p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL < num_cols
  writeMask := fun p₀ p₁ _ j =>
    p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL < num_indices ∧
      p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL < num_cols

/-- **The headline**: `index_select_cat_fwd_kernel` implements the pure gather
`output[i, col] = source[index[i], col]` on its gather IO signature — for every
disjoint flat placement of `source_ptr`/`index_ptr`/`output_ptr`, every pair of
program ids whose active lanes are in bounds, and every launch state whose
windows hold the row-index tile `ids` (`.nat` channel) and the gathered data
tile `xs`, the translated pointer kernel terminates; every write-active lane
`j` (`indices < num_indices ∧ cols < num_cols`, the kernel's 2D `mask`) has
`xs j` at its contiguous destination cell; and every memory cell off the store
window is unchanged — unconditionally.

The skin's readback leg carries a per-context `WriteInj` antecedent because a
`GatherMasked2DKernelIO₁` `write` window may in general eat the loaded index
tile. On this pure-gather side the `write` window is *static*, so `WriteInj` is
just the old `hOutInj` no-duplicate-destination side condition on
`i·stride0 + col·stride1` at the write-active lanes — a fact about the strides,
not about the index data, and it stays inside `⊨` rather than narrowing the
headline.

`hBC : 0 < BLOCK_SIZE_COL` is genuinely forced: the index-tile load runs over
the `[BLOCK_SIZE_INDEX]` tile regardless of `BLOCK_SIZE_COL`, so when
`BLOCK_SIZE_COL = 0` the flattened lane space `Fin (BI * BC)` is empty and
carries no in-bounds witness for it.

Proof: `GatherMasked2DKernelIO₁.Implements.intro` assembles the region-model
gather triple with the flat-memory bridge side conditions. -/
specification index_select_cat_fwd_kernel_correctness
    (output_ptr source_ptr index_ptr : RegionName)
    (num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (hBC : 0 < BLOCK_SIZE_COL) :
    indexSelectCatIO output_ptr source_ptr index_ptr num_indices num_cols
      stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL ⊨
        fun _ _ _ xs j => xs j := by
  refine GatherMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact index_select_cat_fwd_kernel_flattenOk output_ptr source_ptr index_ptr
      num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
  · intro bounds s ids hidsPin hbx hbr hbw
    simp only [indexSelectCatIO] at hidsPin hbx hbr hbw ⊢
    -- the lane of tile cell `(i, c)` — well-formed exactly because `BC > 0`
    have hlane : ∀ (i : Fin BLOCK_SIZE_INDEX) (c : Fin BLOCK_SIZE_COL),
        ((Lane2D.encode ((i, c, PUnit.unit) :
            TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])).val
              / BLOCK_SIZE_COL = i.val ∧
          (Lane2D.encode ((i, c, PUnit.unit) :
            TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])).val
              % BLOCK_SIZE_COL = c.val) :=
      fun i c => ⟨Lane2D.encode_div _, Lane2D.encode_mod _⟩
    refine index_select_cat_fwd_kernel_traceSafe output_ptr source_ptr index_ptr
      num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
      bounds s ?_ ?_ ?_
    · intro i hi
      have hc : Fin BLOCK_SIZE_COL := ⟨0, hBC⟩
      obtain ⟨hd, hm⟩ := hlane i hc
      have h := hbx (Lane2D.encode ((i, hc, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd]; exact hi)
      rw [hd] at h
      exact h
    · intro i c hi hcol
      obtain ⟨hd, hm⟩ := hlane i c
      have h := hbr (Lane2D.encode ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd, hm]; exact ⟨hi, hcol⟩)
      rw [hm] at h
      have hpin := hidsPin (Lane2D.encode ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd]; exact hi)
      rw [hd] at hpin
      show s.readMemValue TileDType.nat index_ptr
            (s.pids 0 * BLOCK_SIZE_INDEX + i.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + c.val) * stride1 < bounds source_ptr
      rw [hpin]
      exact h
    · intro i c hi hcol
      obtain ⟨hd, hm⟩ := hlane i c
      have h := hbw (Lane2D.encode ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd, hm]; exact ⟨hi, hcol⟩)
      rw [hd, hm] at h
      exact h
  · intro s₀ ids xs hi hx
    exact index_select_cat_fwd_kernel_region_run output_ptr source_ptr index_ptr
      num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
      s₀ ids xs hi hx

end VeriTile.Bench.TritonBenchG.IndexSelectCat
