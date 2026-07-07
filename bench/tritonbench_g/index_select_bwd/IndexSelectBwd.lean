import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL

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

## Proof architecture

```
index_select_cat_bwd_kernel_compute_correct       ← ComputeCorrect over the indexed scatter
  └─ index_select_cat_bwd_kernel_correct_of_exec   executed-state readback per active cell
       └─ index_select_cat_bwd_kernel_correct      ← algorithm-layer readback per active cell
```

The spec is the gather/scatter `grad_source[index[row], col] = grad_output[row, col]`
on active cells — no optimizer/reduction oracle applies.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The destination row is
data-dependent (read from `index_ptr`), so a
per-cell injectivity hypothesis `hStoreInj` on the scatter offsets is required as
a side condition — it captures the no-duplicate-destination property the caller
must supply (overlapping indices would race in the underlying scatter). The
`.to(tl.float32)` cast on `grad_output` reduces to the identity at the algorithm
layer (post-erasure all dtypes unify to `ℝ`). The statement is scoped to *active*
cells (`row < num_indices` and `col < num_cols`); masked-out cells are not part of
the realized write map.
-/

namespace VeriTile.Bench.TritonBenchG.IndexSelectBwd

open VeriTile

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

/-- Algorithm-layer cellwise correctness for `index_select_cat_bwd_kernel`.

The injectivity hypothesis captures the no-duplicate-destination condition
needed for a pointwise readback theorem over the scatter store. -/
theorem index_select_cat_bwd_kernel_correct
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState)
    (hStoreInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        gradSourceStoreAddr s index_ptr num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) :
    ∀ idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx →
      (exec (index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
          num_rows num_indices num_cols stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL) s).map
          (fun s' => s'.readMem grad_source_ptr
            (gradSourceStoreAddr s index_ptr num_indices stride0 stride1
              BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx))
        = some (s.readMem grad_output_ptr
            (gradOutputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) := by
  intro idx hActive
  simp [exec, index_select_cat_bwd_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        (if s.pids 0 * BLOCK_SIZE_INDEX + idx.1.val < num_indices then
            (match s.readMemTyped TileDType.nat index_ptr
                (s.pids 0 * BLOCK_SIZE_INDEX + idx.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat) * stride0
          else
            BlockState.defaultCarrier TileDType.nat * stride0) +
          (s.pids 1 * BLOCK_SIZE_COL + idx.2.1.val) * stride1) := by
    simpa [gradSourceStoreAddr, outputIndex, colIndex, BlockState.readMemValue] using hStoreInj
  simp [gradOutputAddr, gradSourceStoreAddr, outputIndex, colIndex,
        BlockState.readMemValue]
  rcases hActive with ⟨hIndex, hCol⟩
  have hIndexRaw : s.pids 0 * BLOCK_SIZE_INDEX + idx.1.val < num_indices := by
    simpa [outputIndex] using hIndex
  have hColRaw : s.pids 1 * BLOCK_SIZE_COL + idx.2.1.val < num_cols := by
    simpa [colIndex] using hCol
  simpa [hIndexRaw, hColRaw] using
    (BlockState.scatter_readback_prop_masked_nd
      (region := grad_source_ptr)
      (shape := [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL])
      (offsetFn := fun i : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        (if s.pids 0 * BLOCK_SIZE_INDEX + i.1.val < num_indices then
            (match s.readMemTyped TileDType.nat index_ptr
                (s.pids 0 * BLOCK_SIZE_INDEX + i.1.val) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat) * stride0
          else
            BlockState.defaultCarrier TileDType.nat * stride0) +
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
      _ hRawInj idx)

/-- Executed-state form of `index_select_cat_bwd_kernel_correct`. -/
theorem index_select_cat_bwd_kernel_correct_of_exec
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState)
    (hStoreInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        gradSourceStoreAddr s index_ptr num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx))
    (s' : BlockState)
    (hExec : exec (index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
        num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL) s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx →
      s'.readMem grad_source_ptr
          (gradSourceStoreAddr s index_ptr num_indices stride0 stride1
            BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)
        = s.readMem grad_output_ptr
            (gradOutputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx) := by
  intro idx hActive
  have h := index_select_cat_bwd_kernel_correct grad_source_ptr index_ptr
    grad_output_ptr num_rows num_indices num_cols stride0 stride1
    BLOCK_SIZE_INDEX BLOCK_SIZE_COL s hStoreInj idx hActive
  rw [hExec] at h
  simpa using h

/-- Compute-facing cellwise correctness for `index_select_cat_bwd_kernel`. -/
theorem index_select_cat_bwd_kernel_compute_correct
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState)
    (hStoreInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        gradSourceStoreAddr s index_ptr num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) :
    ComputeCorrect.Realizes
      (kernel := index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
        num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL)
        (fun idx => (grad_source_ptr,
          gradSourceStoreAddr s index_ptr num_indices stride0 stride1
            BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)))
      (expected := fun idx =>
        s.readMem grad_output_ptr
          (gradOutputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [index_select_cat_bwd_kernel, ComputeExpr.toAlgorithm?,
          ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := index_select_cat_bwd_kernel_correct_of_exec grad_source_ptr index_ptr
    grad_output_ptr num_rows num_indices num_cols stride0 stride1
    BLOCK_SIZE_INDEX BLOCK_SIZE_COL s hStoreInj s' hExec idx hActive
  exact h

/-- Per-kernel output summary for `index_select_cat_bwd_kernel`: the DSL surface
lowers to the algorithm layer, and the indexed scatter into `grad_source` is
compute-correct — under the no-duplicate-destination hypothesis `hStoreInj`, every
active cell writes `grad_output[row, col]` to `grad_source[index[row], col]`. -/
theorem index_select_cat_bwd_kernel_output_summary
    (grad_source_ptr : RegionName) (index_ptr : Region .nat) (grad_output_ptr : RegionName)
    (num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState)
    (hStoreInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        gradSourceStoreAddr s index_ptr num_indices stride0 stride1
          BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) :
    (∃ alg, (index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
        num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := index_select_cat_bwd_kernel grad_source_ptr index_ptr grad_output_ptr
        num_rows num_indices num_cols stride0 stride1
        BLOCK_SIZE_INDEX BLOCK_SIZE_COL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL)
        (fun idx => (grad_source_ptr,
          gradSourceStoreAddr s index_ptr num_indices stride0 stride1
            BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)))
      (expected := fun idx =>
        s.readMem grad_output_ptr
          (gradOutputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) := by
  refine ⟨?_, ?_⟩
  · simp [index_select_cat_bwd_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  · exact index_select_cat_bwd_kernel_compute_correct grad_source_ptr index_ptr
      grad_output_ptr num_rows num_indices num_cols stride0 stride1
      BLOCK_SIZE_INDEX BLOCK_SIZE_COL s hStoreInj

end VeriTile.Bench.TritonBenchG.IndexSelectBwd
