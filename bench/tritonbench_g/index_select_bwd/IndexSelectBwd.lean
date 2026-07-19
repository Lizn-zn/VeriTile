import VeriTile.Triton

/-!
# `index_select_bwd` — strict per-kernel correctness

`index_select_cat_bwd_kernel` is the backward of an index-select/cat: for the
`(pid0, pid1)` tile of `(index, column)` blocks it loads the `grad_output` tile,
gathers the destination row from `index_ptr`, and scatters the gradient into
`grad_source` at the indexed row, column-by-column.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`index_select_cat_bwd_kernel[grid](...)`, the 2D grid
sizes `cdiv(num_indices, BLOCK_SIZE_INDEX)` × `cdiv(num_cols, BLOCK_SIZE_COL)`,
the stride/shape validation, and how the runtime composes per-program writes) is
the *trusted boundary*, not a proof obligation here. Because the program ids
`pids 0`, `pids 1` are universally quantified, the per-program statement covers
every program of the grid.

The headline is stated on the kernel's scatter IO signature
`indexSelectBwdIO` (`GatherMasked2DKernelIO₁`, the index-tile genre's skin):
which buffer is which argument — incoming gradient `grad_output_ptr` (`inp`),
row-index tile `index_ptr` (`idxbuf`, the `.nat` channel), scatter target
`grad_source_ptr` (`out`) — the **static** index-tile window
`readx = pid0 * BI + j / BC`, the **static** data read window
`read = (pid0 * BI + j / BC) * stride0 + (pid1 * BC + j % BC) * stride1`
(which ignores the loaded index tile), and the **data-dependent** scatter
destination `write … ids j = ids j * stride0 + (pid1 * BC + j % BC) * stride1`.
`⊨` (`GatherMasked2DKernelIO₁.Implements`) is the audit-once Hoare-triple
combinator: for **every** disjoint flat placement of the three buffers,
**every** pair of program ids whose active lanes are in bounds, and **every**
launch state whose windows hold the index tile `ids` and the gradient tile
`xs`, the translated pointer kernel terminates, the readback leg — guarded by
the per-context `WriteInj` antecedent — puts `xs j` at every write-active
lane's scatter cell, and every memory cell off the raw scatter cells is
unchanged.

The skin's lane space is one-dimensional, so the kernel's `[BI, BC]` store tile
is carried as the flattened lane space `Fin (BI * BC)` — lane `j` is tile cell
`(j / BC, j % BC)` (`laneIdx`, with `idxLane` the inverse). This is a
bijection, so nothing is lost: every tile cell is some lane.

## Proof architecture

```
index_select_cat_bwd_kernel_correctness           ← TOP SPECIFICATION (indexSelectBwdIO ⊨ indexed scatter)
  ├─ index_select_cat_bwd_kernel_flattenOk         bridge fragment membership
  ├─ index_select_cat_bwd_kernel_traceSafe         per-execution lane-wise safety walk
  └─ index_select_cat_bwd_kernel_region_run        region-model scatter Hoare triple
       ├─ index_select_cat_bwd_kernel_exec_isSome  termination
       ├─ index_select_cat_bwd_kernel_correct_of_exec  executed-state readback per active cell
       │    └─ index_select_cat_bwd_kernel_correct  ← algorithm-layer readback per active cell
       └─ index_select_cat_bwd_kernel_frame        scatter-store cell frame
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The `.to(tl.float32)`
cast on `grad_output` reduces to the identity at the algorithm layer
(post-erasure all dtypes unify to `ℝ`). The destination row is **data-dependent**
(read from `index_ptr`), so the readback leg of the `⊨` headline is guarded by
the skin's per-context `WriteInj` antecedent — no two **write-active** lanes
share a destination. On this scatter side that antecedent is **genuinely
non-vacuous**: the benchmark's own test cases pass repeated indices
(`[0, 0, 0, 0, 0]`, `[9, 9, 9, 9, 9]`), for which two lanes of the same column
block really do collide, the underlying scatter races, and no pointwise
readback can hold. It is the honest no-duplicate-destination precondition, and
it is carried *inside* `⊨` rather than as a headline hypothesis, so the
headline stays free of scope-narrowing hypotheses. The frame and the
trace-safety write bound are stated at the *ungated* write-active lanes, so
they hold with or without `WriteInj`.

`hBC : 0 < BLOCK_SIZE_COL` is genuinely forced: the index-tile load runs over
the `[BLOCK_SIZE_INDEX]` tile regardless of `BLOCK_SIZE_COL`, so when
`BLOCK_SIZE_COL = 0` the flattened lane space `Fin (BI * BC)` is empty and
carries no in-bounds witness for it.
-/

namespace VeriTile.Bench.TritonBenchG.IndexSelectBwd

open VeriTile.Triton
open scoped VeriTile.Triton.GatherMasked2DKernelIO₁

/-- Faithful transcription of `index_select_bwd.py`'s
`index_select_cat_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_INDEX: tl.constexpr` / `BLOCK_SIZE_COL: tl.constexpr`
  -> Lean `Nat` parameters.
- Python `.to(tl.float32)` on `grad_output` is represented explicitly in the
  Compute layer; the algorithm-layer theorem observes its Real projection. -/
def index_select_cat_bwd_kernel
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (_num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) :
    ComputeKernel := triton {
  pid0 = tl.program_id(axis=0)
  pid1 = tl.program_id(axis=1)
  cols = pid1 * $(BLOCK_SIZE_COL) + tl.arange(0, $(BLOCK_SIZE_COL))
  grad_output_indices = pid0 * $(BLOCK_SIZE_INDEX) + tl.arange(0, $(BLOCK_SIZE_INDEX))
  grad_output_offsets =
    grad_output_ptr + grad_output_indices[:, None] * $(stride0) +
      cols[None, :] * $(stride1)
  grad_output_mask =
    (grad_output_indices[:, None] < $(num_indices)) &
      (cols[None, :] < $(num_cols))
  grad_output = (tl.load(grad_output_offsets, mask=grad_output_mask)).to(tl.float32)
  grad_source_indices =
    tl.load(index_ptr + grad_output_indices,
      mask=grad_output_indices < $(num_indices))
  grad_source_offsets =
    grad_source_ptr + grad_source_indices[:, None] * $(stride0) +
      cols[None, :] * $(stride1)
  tl.store(grad_source_offsets, grad_output, mask=grad_output_mask)
}

/-! ### Flattened lane space

The skin's lane space is `Fin B`; the kernel's store tile is `[BI, BC]`. Lane
`j` is tile cell `(j / BC, j % BC)`, and `idxLane` is the inverse. -/

/-- Tile cell of lane `j` of the flattened `[BI, BC]` lane space. -/
def laneIdx {BI BC : Nat} (j : Fin (BI * BC)) : TileIndex [BI, BC] :=
  (⟨j.val / BC, Nat.div_lt_of_lt_mul
      (Nat.lt_of_lt_of_le j.isLt (Nat.le_of_eq (Nat.mul_comm BI BC)))⟩,
    ⟨j.val % BC, Nat.mod_lt _ (by
      rcases Nat.eq_zero_or_pos BC with h | h
      · exact absurd j.isLt (by simp [h])
      · exact h)⟩,
    PUnit.unit)

/-- Lane of tile cell `k` — the inverse of `laneIdx`. -/
def idxLane {BI BC : Nat} (k : TileIndex [BI, BC]) : Fin (BI * BC) :=
  ⟨k.1.val * BC + k.2.1.val, by
    have h1 : k.1.val * BC + k.2.1.val < (k.1.val + 1) * BC := by
      have := k.2.1.isLt
      rw [Nat.succ_mul]
      omega
    exact Nat.lt_of_lt_of_le h1 (Nat.mul_le_mul_right BC k.1.isLt)⟩

theorem idxLane_div {BI BC : Nat} (k : TileIndex [BI, BC]) :
    (idxLane k).val / BC = k.1.val := by
  have hBC : 0 < BC := Nat.lt_of_le_of_lt (Nat.zero_le _) k.2.1.isLt
  show (k.1.val * BC + k.2.1.val) / BC = k.1.val
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hBC, Nat.div_eq_of_lt k.2.1.isLt,
    Nat.zero_add]

theorem idxLane_mod {BI BC : Nat} (k : TileIndex [BI, BC]) :
    (idxLane k).val % BC = k.2.1.val := by
  show (k.1.val * BC + k.2.1.val) % BC = k.2.1.val
  rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt k.2.1.isLt]

theorem laneIdx_idxLane {BI BC : Nat} (k : TileIndex [BI, BC]) :
    laneIdx (idxLane k) = k := by
  refine Prod.ext (Fin.ext ?_) (Prod.ext (Fin.ext ?_) rfl)
  · exact idxLane_div k
  · exact idxLane_mod k

/-! ### Kernel-coupled addressing vocabulary -/

def outputIndex (s : BlockState) (BLOCK_SIZE_INDEX : Nat)
    (i : Fin BLOCK_SIZE_INDEX) : Nat :=
  s.pids 0 * BLOCK_SIZE_INDEX + i.val

def colIndex (s : BlockState) (BLOCK_SIZE_COL : Nat)
    (j : Fin BLOCK_SIZE_COL) : Nat :=
  s.pids 1 * BLOCK_SIZE_COL + j.val

def gradOutputAddr (s : BlockState)
    (stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Nat :=
  outputIndex s BLOCK_SIZE_INDEX idx.1 * stride0 +
    colIndex s BLOCK_SIZE_COL idx.2.1 * stride1

def gradSourceStoreAddr (s : BlockState) (index_ptr : RegionName)
    (num_indices stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Nat :=
  (if outputIndex s BLOCK_SIZE_INDEX idx.1 < num_indices then
      s.readMemValue .nat index_ptr (outputIndex s BLOCK_SIZE_INDEX idx.1) * stride0
    else
      BlockState.defaultCarrier TileDType.nat * stride0) +
    colIndex s BLOCK_SIZE_COL idx.2.1 * stride1

def active
    (s : BlockState) (num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Prop :=
  outputIndex s BLOCK_SIZE_INDEX idx.1 < num_indices ∧
    colIndex s BLOCK_SIZE_COL idx.2.1 < num_cols

instance activeDecidable
    (s : BlockState) (num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) :
    Decidable (active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx) := by
  unfold active
  infer_instance

set_option maxHeartbeats 1600000 in
/-- Algorithm-layer cellwise correctness at an **active** cell, under the
no-collision hypothesis restricted to active cells (the state-coupled form of
the skin's `WriteInj` on the data-dependent scatter window — genuinely
non-vacuous here: repeated indices really do collide). -/
theorem index_select_cat_bwd_kernel_correct
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState) (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])
    (hActive : active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)
    (hNoCol : ∀ k : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL k →
      gradSourceStoreAddr s index_ptr.cast num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL k
        = gradSourceStoreAddr s index_ptr.cast num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx →
      k = idx) :
    (exec (index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
        num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL) s).map
        (fun s' => s'.readMem grad_source_ptr
          (gradSourceStoreAddr s index_ptr.cast num_indices stride0 stride1
            BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx))
      = some (s.readMem grad_output_ptr
          (gradOutputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) := by
  obtain ⟨hIndex, hCol⟩ := hActive
  have hIndexRaw : s.pids 0 * BLOCK_SIZE_INDEX + idx.1.val < num_indices := hIndex
  have hColRaw : s.pids 1 * BLOCK_SIZE_COL + idx.2.1.val < num_cols := hCol
  have hcol : ∀ k : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      (s.pids 0 * BLOCK_SIZE_INDEX + k.1.val < num_indices ∧
        s.pids 1 * BLOCK_SIZE_COL + k.2.1.val < num_cols) →
      (if s.pids 0 * BLOCK_SIZE_INDEX + k.1.val < num_indices then
            (match s.readMemTyped TileDType.nat index_ptr.cast
                (s.pids 0 * BLOCK_SIZE_INDEX + k.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat) * stride0
          else BlockState.defaultCarrier TileDType.nat * stride0) +
          (s.pids 1 * BLOCK_SIZE_COL + k.2.1.val) * stride1
        = (if s.pids 0 * BLOCK_SIZE_INDEX + idx.1.val < num_indices then
            (match s.readMemTyped TileDType.nat index_ptr.cast
                (s.pids 0 * BLOCK_SIZE_INDEX + idx.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat) * stride0
          else BlockState.defaultCarrier TileDType.nat * stride0) +
          (s.pids 1 * BLOCK_SIZE_COL + idx.2.1.val) * stride1 →
      k = idx := by
    intro k hk heq
    refine hNoCol k hk ?_
    simpa [gradSourceStoreAddr, outputIndex, colIndex, BlockState.readMemValue]
      using heq
  simp [exec, index_select_cat_bwd_kernel, stepStmts, stepStmt, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  simp [gradOutputAddr, gradSourceStoreAddr, outputIndex, colIndex,
        BlockState.readMemValue]
  simpa [hIndexRaw, hColRaw] using
    (BlockState.scatter_readback_prop_masked_nd_of_true
      (region := grad_source_ptr)
      (shape := [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])
      (offsetFn := fun i : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        (if s.pids 0 * BLOCK_SIZE_INDEX + i.1.val < num_indices then
            (match s.readMemTyped TileDType.nat index_ptr.cast
                (s.pids 0 * BLOCK_SIZE_INDEX + i.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat) * stride0
          else BlockState.defaultCarrier TileDType.nat * stride0) +
          (s.pids 1 * BLOCK_SIZE_COL + i.2.1.val) * stride1)
      (valueFn := fun i : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        WithBot.unbotD 0
          (if s.pids 0 * BLOCK_SIZE_INDEX + i.1.val < num_indices ∧
              s.pids 1 * BLOCK_SIZE_COL + i.2.1.val < num_cols then
            some (s.readMem grad_output_ptr
              ((s.pids 0 * BLOCK_SIZE_INDEX + i.1.val) * stride0 +
                (s.pids 1 * BLOCK_SIZE_COL + i.2.1.val) * stride1))
          else
            some (s.undef grad_output_ptr
              ((s.pids 0 * BLOCK_SIZE_INDEX + i.1.val) * stride0 +
                (s.pids 1 * BLOCK_SIZE_COL + i.2.1.val) * stride1))))
      (P := fun i : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        s.pids 0 * BLOCK_SIZE_INDEX + i.1.val < num_indices ∧
          s.pids 1 * BLOCK_SIZE_COL + i.2.1.val < num_cols)
      _ idx ⟨hIndexRaw, hColRaw⟩ hcol)

/-- Executed-state form of `index_select_cat_bwd_kernel_correct`. -/
theorem index_select_cat_bwd_kernel_correct_of_exec
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState) (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])
    (hActive : active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)
    (hNoCol : ∀ k : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL k →
      gradSourceStoreAddr s index_ptr.cast num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL k
        = gradSourceStoreAddr s index_ptr.cast num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx →
      k = idx)
    (s' : BlockState)
    (hExec : exec (index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
        num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL) s = some s') :
    s'.readMem grad_source_ptr
        (gradSourceStoreAddr s index_ptr.cast num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)
      = s.readMem grad_output_ptr
        (gradOutputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx) := by
  have h := index_select_cat_bwd_kernel_correct grad_source_ptr index_ptr
    grad_output_ptr num_rows num_indices num_cols stride0 stride1
    BLOCK_SIZE_INDEX BLOCK_SIZE_COL s idx hActive hNoCol
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
masked loads plus one masked scatter store). -/
private theorem index_select_cat_bwd_kernel_exec_isSome
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState) :
    ∃ s1, exec ((index_select_cat_bwd_kernel grad_source_ptr index_ptr
        grad_output_ptr num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL).toAlgKernel) s = some s1 := by
  simp [exec, index_select_cat_bwd_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.insertAxis, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- Frame half: every memory cell not actively hit by the indexed scatter is
preserved by the run. Unconditional in the injectivity: colliding raw scatter
cells are still scatter cells, so the exclusion set needs no `WriteInj`. -/
private theorem index_select_cat_bwd_kernel_frame
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s s1 : BlockState)
    (hExec : exec ((index_select_cat_bwd_kernel grad_source_ptr index_ptr
        grad_output_ptr num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx →
      ¬(grad_source_ptr = r ∧
        gradSourceStoreAddr s index_ptr.cast num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, index_select_cat_bwd_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, stepStmts, stepStmt,
    evalOp.eq_def, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.insertAxis, TileShape.dropInsertedIndex] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  refine hmiss k hmk ⟨hc.1, ?_⟩
  simpa [gradSourceStoreAddr, outputIndex, colIndex, BlockState.readMemValue]
    using hc.2

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk: one computational unfold walks all ten
statements — six are memory-silent (`program_id` ×2, the two window
arithmetics, the mask comparison, the scatter-address arithmetic) — and reduces
the masked `grad_output` load, the masked index-tile load and the
data-dependent masked scatter store to the three lane-wise bounds
hypotheses. -/
theorem index_select_cat_bwd_kernel_traceSafe
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (h1 : ∀ (i : Fin BLOCK_SIZE_INDEX) (c : Fin BLOCK_SIZE_COL),
      s.pids 0 * BLOCK_SIZE_INDEX + i.val < num_indices →
      s.pids 1 * BLOCK_SIZE_COL + c.val < num_cols →
      (s.pids 0 * BLOCK_SIZE_INDEX + i.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + c.val) * stride1
        < bounds grad_output_ptr)
    (h2 : ∀ i : Fin BLOCK_SIZE_INDEX,
      s.pids 0 * BLOCK_SIZE_INDEX + i.val < num_indices →
      s.pids 0 * BLOCK_SIZE_INDEX + i.val < bounds index_ptr.cast)
    (h3 : ∀ (i : Fin BLOCK_SIZE_INDEX) (c : Fin BLOCK_SIZE_COL),
      s.pids 0 * BLOCK_SIZE_INDEX + i.val < num_indices →
      s.pids 1 * BLOCK_SIZE_COL + c.val < num_cols →
      s.readMemValue .nat index_ptr.cast (s.pids 0 * BLOCK_SIZE_INDEX + i.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + c.val) * stride1
        < bounds grad_source_ptr) :
    Kernel.TraceSafe bounds
      ((index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
        num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [index_select_cat_bwd_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    Op.PointerAddressesSafeOn, Op.MemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
    NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.insertAxis, TileShape.dropInsertedIndex,
    BlockState.readMemValue]
  refine ⟨fun a c ha hc => h1 a c ha hc, fun a ha => h2 a ha, fun a c ha hc => ?_⟩
  have h := h3 a c ha hc
  simpa [BlockState.readMemValue, ha] using h

/-- The kernel sits inside the flat-memory bridge's covered fragment (pointer
arithmetic, a masked `.nat`-channel index load, a masked data load through the
`.to(tl.float32)` cast, and the data-dependent masked scatter store). -/
theorem index_select_cat_bwd_kernel_flattenOk
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) :
    ((index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
        num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [index_select_cat_bwd_kernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- **The region-model scatter Hoare triple** — termination, write-active-lane
scatter values under the per-context no-collision antecedent, and frame off the
raw scatter cells, from any launch state whose windows hold the index tile
`ids` and the gradient tile `xs`. This is the `hrun` obligation of
`GatherMasked2DKernelIO₁.Implements.intro`; the value half reuses
`index_select_cat_bwd_kernel_correct_of_exec` (whose state-coupled store
address collapses to `ids j * stride0 + col * stride1` under the index-tile
pin). -/
theorem index_select_cat_bwd_kernel_region_run
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1 BI BC : Nat)
    (s₀ : BlockState) (ids : Fin (BI * BC) → Nat) (xs : Fin (BI * BC) → ℝ)
    (hi : ∀ j : Fin (BI * BC), s₀.pids 0 * BI + j.val / BC < num_indices →
      s₀.readMemValue .nat index_ptr.cast (s₀.pids 0 * BI + j.val / BC) = ids j)
    (hx : ∀ j : Fin (BI * BC),
      (s₀.pids 0 * BI + j.val / BC < num_indices ∧
        s₀.pids 1 * BC + j.val % BC < num_cols) →
      s₀.readMem grad_output_ptr
        ((s₀.pids 0 * BI + j.val / BC) * stride0 +
          (s₀.pids 1 * BC + j.val % BC) * stride1) = xs j) :
    ∃ s1, exec ((index_select_cat_bwd_kernel grad_source_ptr index_ptr
          grad_output_ptr num_rows num_indices num_cols stride0 stride1
          BI BC).toAlgKernel) s₀ = some s1
      ∧ ((∀ j k : Fin (BI * BC),
            (s₀.pids 0 * BI + j.val / BC < num_indices ∧
              s₀.pids 1 * BC + j.val % BC < num_cols) →
            (s₀.pids 0 * BI + k.val / BC < num_indices ∧
              s₀.pids 1 * BC + k.val % BC < num_cols) →
            ids j * stride0 + (s₀.pids 1 * BC + j.val % BC) * stride1
              = ids k * stride0 + (s₀.pids 1 * BC + k.val % BC) * stride1 →
            j = k) →
          ∀ j : Fin (BI * BC),
            (s₀.pids 0 * BI + j.val / BC < num_indices ∧
              s₀.pids 1 * BC + j.val % BC < num_cols) →
            s1.readMem grad_source_ptr
                (ids j * stride0 + (s₀.pids 1 * BC + j.val % BC) * stride1)
              = xs j)
      ∧ (∀ r o,
          (r ≠ grad_source_ptr ∨ ∀ j : Fin (BI * BC),
            (s₀.pids 0 * BI + j.val / BC < num_indices ∧
              s₀.pids 1 * BC + j.val % BC < num_cols) →
            o ≠ ids j * stride0 + (s₀.pids 1 * BC + j.val % BC) * stride1) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := index_select_cat_bwd_kernel_exec_isSome grad_source_ptr
    index_ptr grad_output_ptr num_rows num_indices num_cols stride0 stride1
    BI BC s₀
  have hlaneAct : ∀ k : TileIndex [BI, BC],
      active s₀ num_indices num_cols BI BC k →
      (s₀.pids 0 * BI + (idxLane k).val / BC < num_indices ∧
        s₀.pids 1 * BC + (idxLane k).val % BC < num_cols) := by
    intro k hk
    rw [idxLane_div, idxLane_mod]
    exact hk
  -- under the index-tile pin the state-coupled store address is the declared
  -- data-dependent scatter destination
  have hstore : ∀ k : TileIndex [BI, BC],
      active s₀ num_indices num_cols BI BC k →
      gradSourceStoreAddr s₀ index_ptr.cast num_indices stride0 stride1 BI BC k
        = ids (idxLane k) * stride0 +
          (s₀.pids 1 * BC + (idxLane k).val % BC) * stride1 := by
    intro k hk
    have hpin := hi (idxLane k) (by rw [idxLane_div]; exact hk.1)
    rw [idxLane_div] at hpin
    rw [idxLane_mod]
    unfold gradSourceStoreAddr
    rw [if_pos hk.1]
    show s₀.readMemValue .nat index_ptr.cast (s₀.pids 0 * BI + k.1.val) * stride0 +
        (s₀.pids 1 * BC + k.2.1.val) * stride1
      = ids (idxLane k) * stride0 + (s₀.pids 1 * BC + k.2.1.val) * stride1
    rw [hpin]
  refine ⟨s1, hs1, fun hinj j hj => ?_, fun r o hcond => ?_⟩
  · have hActive : active s₀ num_indices num_cols BI BC (laneIdx j) := hj
    have hlj : idxLane (laneIdx j) = j := by
      refine Fin.ext ?_
      show (laneIdx j).1.val * BC + (laneIdx j).2.1.val = j.val
      show j.val / BC * BC + j.val % BC = j.val
      rw [Nat.mul_comm, Nat.add_comm]
      exact Nat.mod_add_div j.val BC
    have hNoCol : ∀ k : TileIndex [BI, BC],
        active s₀ num_indices num_cols BI BC k →
        gradSourceStoreAddr s₀ index_ptr.cast num_indices stride0 stride1 BI BC k
          = gradSourceStoreAddr s₀ index_ptr.cast num_indices stride0 stride1 BI BC
            (laneIdx j) →
        k = laneIdx j := by
      intro k hk heq
      have hjk : idxLane k = j := by
        have h := hinj (idxLane k) j (hlaneAct k hk) hj ?_
        · exact h
        · rw [← hstore k hk]
          have h2 := hstore (laneIdx j) hActive
          rw [hlj] at h2
          rw [heq, h2]
      rw [← laneIdx_idxLane k, hjk]
    have h := index_select_cat_bwd_kernel_correct_of_exec grad_source_ptr index_ptr
      grad_output_ptr num_rows num_indices num_cols stride0 stride1 BI BC s₀
      (laneIdx j) hActive hNoCol s1 hs1
    have hsa := hstore (laneIdx j) hActive
    rw [hlj] at hsa
    rw [← hsa, h]
    have hgo : gradOutputAddr s₀ stride0 stride1 BI BC (laneIdx j)
        = (s₀.pids 0 * BI + j.val / BC) * stride0 +
          (s₀.pids 1 * BC + j.val % BC) * stride1 := rfl
    rw [hgo]
    exact hx j hj
  · refine index_select_cat_bwd_kernel_frame grad_source_ptr index_ptr
      grad_output_ptr num_rows num_indices num_cols stride0 stride1 BI BC
      s₀ s1 hs1 r o ?_
    intro k hk hc
    rcases hcond with hne | hno
    · exact hne hc.1.symm
    · exact hno (idxLane k) (hlaneAct k hk)
        (hc.2.symm.trans (hstore k hk))

/-- `index_select_cat_bwd_kernel`'s scatter **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`idxbuf`/`out` — which buffer is which argument (incoming gradient
  `grad_output_ptr`, `.nat` row-index channel `index_ptr`, scatter target
  `grad_source_ptr`);
* `B = BLOCK_SIZE_INDEX * BLOCK_SIZE_COL` — the kernel's `[BI, BC]` tile,
  flattened: lane `j` is tile cell `(j / BC, j % BC)`;
* `readx` — the **static** index-tile window `pid0 * BI + j / BC`, i.e. the
  kernel's `grad_output_indices`;
* `read` — the **static** data window
  `(pid0 * BI + j / BC) * stride0 + (pid1 * BC + j % BC) * stride1`: the
  kernel's `grad_output_offsets`, which ignores the loaded index tile;
* `write` — the **data-dependent** scatter destination
  `ids j * stride0 + (pid1 * BC + j % BC) * stride1`: the loaded row times
  `stride0` plus the column offset, exactly the kernel's
  `grad_source_offsets`;
* `mask` — the index-tile gate `pid0 * BI + j / BC < num_indices`;
* `readMask`/`writeMask` — the kernel's `grad_output_mask`,
  `grad_output_indices < num_indices ∧ cols < num_cols`.

The windows and masks are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing, masking and index-channel plumbing
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes and the
write-active scatter cells. -/
def indexSelectBwdIO (grad_source_ptr index_ptr grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat) : GatherMasked2DKernelIO₁ where
  kernel := index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
    num_rows num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
  inp := grad_output_ptr
  idxbuf := index_ptr
  out := grad_source_ptr
  B := BLOCK_SIZE_INDEX * BLOCK_SIZE_COL
  readx := fun p₀ _ j => p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL
  read := fun p₀ p₁ _ j =>
    (p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL) * stride0 +
      (p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL) * stride1
  write := fun _ p₁ ids j =>
    ids j * stride0 + (p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL) * stride1
  mask := fun p₀ _ j =>
    p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL < num_indices
  readMask := fun p₀ p₁ _ j =>
    p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL < num_indices ∧
      p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL < num_cols
  writeMask := fun p₀ p₁ _ j =>
    p₀ * BLOCK_SIZE_INDEX + j.val / BLOCK_SIZE_COL < num_indices ∧
      p₁ * BLOCK_SIZE_COL + j.val % BLOCK_SIZE_COL < num_cols

/-- **The headline**: `index_select_cat_bwd_kernel` implements the indexed
gradient scatter `grad_source[index[i], col] = grad_output[i, col]` on its
scatter IO signature — for every disjoint flat placement of
`grad_output_ptr`/`index_ptr`/`grad_source_ptr`, every pair of program ids
whose active lanes are in bounds (the write bound at the *ungated* scatter
cells), and every launch state whose windows hold the row-index tile `ids`
(`.nat` channel) and the gradient tile `xs`, the translated pointer kernel
terminates; under the per-context `WriteInj` antecedent every write-active lane
`j` (`grad_output_indices < num_indices ∧ cols < num_cols`, the kernel's
`grad_output_mask`) has `xs j` at its scatter cell
`ids j * stride0 + col * stride1`; and every memory cell off the raw scatter
cells is unchanged — unconditionally.

`WriteInj` here is **genuinely non-vacuous and genuinely required**: the
benchmark's own test cases pass repeated indices (`[0, 0, 0, 0, 0]`,
`[9, 9, 9, 9, 9]`), two lanes of one column block then scatter to the same cell,
and no pointwise readback can hold. It is the honest no-duplicate-destination
precondition and it lives inside `⊨`, so the headline itself stays free of
scope-narrowing hypotheses.

`hBC : 0 < BLOCK_SIZE_COL` is genuinely forced: the index-tile load runs over
the `[BLOCK_SIZE_INDEX]` tile regardless of `BLOCK_SIZE_COL`, so when
`BLOCK_SIZE_COL = 0` the flattened lane space is empty and carries no in-bounds
witness for it.

Proof: `GatherMasked2DKernelIO₁.Implements.intro` assembles the region-model
scatter triple with the flat-memory bridge side conditions. -/
specification index_select_cat_bwd_kernel_correctness
    (grad_source_ptr index_ptr grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (hBC : 0 < BLOCK_SIZE_COL) :
    indexSelectBwdIO grad_source_ptr index_ptr grad_output_ptr num_rows
      num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL ⊨
        fun _ _ _ xs j => xs j := by
  refine GatherMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact index_select_cat_bwd_kernel_flattenOk grad_source_ptr index_ptr
      grad_output_ptr num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL
  · intro bounds s ids hidsPin hbx hbr hbw
    simp only [indexSelectBwdIO] at hidsPin hbx hbr hbw ⊢
    have hlane : ∀ (i : Fin BLOCK_SIZE_INDEX) (c : Fin BLOCK_SIZE_COL),
        ((idxLane ((i, c, PUnit.unit) :
            TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])).val
              / BLOCK_SIZE_COL = i.val ∧
          (idxLane ((i, c, PUnit.unit) :
            TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])).val
              % BLOCK_SIZE_COL = c.val) :=
      fun i c => ⟨idxLane_div _, idxLane_mod _⟩
    refine index_select_cat_bwd_kernel_traceSafe grad_source_ptr index_ptr
      grad_output_ptr num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL bounds s ?_ ?_ ?_
    · intro i c hi hcol
      obtain ⟨hd, hm⟩ := hlane i c
      have h := hbr (idxLane ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]))
        (by rw [hd, hm]; exact ⟨hi, hcol⟩)
      rw [hd, hm] at h
      exact h
    · intro i hi
      have hc : Fin BLOCK_SIZE_COL := ⟨0, hBC⟩
      obtain ⟨hd, hm⟩ := hlane i hc
      have h := hbx (idxLane ((i, hc, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd]; exact hi)
      rw [hd] at h
      exact h
    · intro i c hi hcol
      obtain ⟨hd, hm⟩ := hlane i c
      have h := hbw (idxLane ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]))
        (by rw [hd, hm]; exact ⟨hi, hcol⟩)
      rw [hm] at h
      have hpin := hidsPin (idxLane ((i, c, PUnit.unit) :
        TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])) (by rw [hd]; exact hi)
      rw [hd] at hpin
      show s.readMemValue TileDType.nat index_ptr
            (s.pids 0 * BLOCK_SIZE_INDEX + i.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + c.val) * stride1 < bounds grad_source_ptr
      rw [hpin]
      exact h
  · intro s₀ ids xs hi hx
    exact index_select_cat_bwd_kernel_region_run grad_source_ptr index_ptr
      grad_output_ptr num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL s₀ ids xs hi hx

end VeriTile.Bench.TritonBenchG.IndexSelectBwd
