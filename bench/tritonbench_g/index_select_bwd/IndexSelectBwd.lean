import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.IndexSelectBwd

open VeriTile.Triton

/-- Faithful transcription of `index_select_bwd.py`'s
`index_select_cat_bwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_INDEX: tl.constexpr` / `BLOCK_SIZE_COL: tl.constexpr`
  -> Lean `Nat` parameters.
- Python `.to(tl.float32)` on `grad_output` is erased in the algorithm layer,
  matching the existing Real-first TritonBench-G correctness policy. -/
def index_select_cat_bwd_kernel
    (grad_source_ptr index_ptr grad_output_ptr : RegionName)
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
    (grad_output_indices[:, None] < $(num_indices)) and
      (cols[None, :] < $(num_cols))
  grad_output = tl.load(grad_output_offsets, mask=grad_output_mask)
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

def gradSourceAddr (s : BlockState) (index_ptr : RegionName)
    (stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL]) : Nat :=
  s.readMemValue .nat index_ptr (outputIndex s BLOCK_SIZE_INDEX idx.1) * stride0 +
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
    (grad_source_ptr index_ptr grad_output_ptr : RegionName)
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
  simp [exec, index_select_cat_bwd_kernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex]
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
    (grad_source_ptr index_ptr grad_output_ptr : RegionName)
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
    (grad_source_ptr index_ptr grad_output_ptr : RegionName)
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
  · simp [index_select_cat_bwd_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := index_select_cat_bwd_kernel_correct_of_exec grad_source_ptr index_ptr
    grad_output_ptr num_rows num_indices num_cols stride0 stride1
    BLOCK_SIZE_INDEX BLOCK_SIZE_COL s hStoreInj s' hExec idx hActive
  exact h

end VeriTile.Bench.TritonBenchG.IndexSelectBwd
