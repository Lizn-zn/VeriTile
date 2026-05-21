import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.IndexSelectCat

open VeriTile.Triton

/-- Faithful transcription of `index_select_cat.py`'s
`index_select_cat_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE_INDEX: tl.constexpr` / `BLOCK_SIZE_COL: tl.constexpr`
  → Lean `Nat` parameters. -/
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

/-- Algorithm-layer cellwise correctness for `index_select_cat_fwd_kernel`. -/
theorem index_select_cat_fwd_kernel_correct
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) :
    ∀ idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      (exec (index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
          num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL) s).map
          (fun s' => s'.readMem output_ptr
            (outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx))
        = some (if active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx then
            s.readMem source_ptr
              (sourceAddr s index_ptr stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)
          else
            s.readMem output_ptr
              (outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) := by
  intro idx
  simp [exec, index_select_cat_fwd_kernel, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.expandDim,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, Option.bind, Option.map,
        TileShape.insertAxis, TileShape.dropInsertedIndex]
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        (s.pids 0 * BLOCK_SIZE_INDEX + idx.1.val) * stride0 +
          (s.pids 1 * BLOCK_SIZE_COL + idx.2.1.val) * stride1) := by
    simpa [outputAddr, indexBase, colBase] using hOutInj
  simp [active, outputAddr, sourceAddr, indexBase, colBase, BlockState.readMemValue]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj idx]
  by_cases hIndex : s.pids 0 * BLOCK_SIZE_INDEX + idx.1.val < num_indices
  · by_cases hCol : s.pids 1 * BLOCK_SIZE_COL + idx.2.1.val < num_cols
    · simp [hIndex, hCol]
    · simp [hIndex, hCol]
  · simp [hIndex]

/-- Executed-state form of `index_select_cat_fwd_kernel_correct`. -/
theorem index_select_cat_fwd_kernel_correct_of_exec
    (output_ptr source_ptr : RegionName) (index_ptr : Region .nat)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx))
    (s' : BlockState)
    (hExec : exec (index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL) s = some s') :
    ∀ idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL],
      s'.readMem output_ptr
          (outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)
        = if active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx then
            s.readMem source_ptr
              (sourceAddr s index_ptr stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)
          else
            s.readMem output_ptr
              (outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx) := by
  intro idx
  have h := index_select_cat_fwd_kernel_correct output_ptr source_ptr index_ptr
    num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL s hOutInj idx
  rw [hExec] at h
  simpa using h

/-- Compute-facing cellwise correctness for `index_select_cat_fwd_kernel`. -/
theorem index_select_cat_fwd_kernel_compute_correct
    (output_ptr source_ptr index_ptr : RegionName)
    (num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_INDEX, BLOCK_SIZE_COL] =>
        outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) :
    ComputeCorrect.Realizes
      (kernel := index_select_cat_fwd_kernel output_ptr source_ptr index_ptr
        num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s num_indices num_cols BLOCK_SIZE_INDEX BLOCK_SIZE_COL)
        (fun idx => (output_ptr,
          outputAddr s stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)))
      (expected := fun idx =>
        s.readMem source_ptr
          (sourceAddr s index_ptr stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [index_select_cat_fwd_kernel]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := index_select_cat_fwd_kernel_correct_of_exec output_ptr source_ptr index_ptr
    num_indices num_cols stride0 stride1 BLOCK_SIZE_INDEX BLOCK_SIZE_COL
    s hOutInj s' hExec idx
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.IndexSelectCat
