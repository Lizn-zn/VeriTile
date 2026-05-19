import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `chunk_cumsum_kernel.py`'s
`chunk_global_cumsum_scalar_kernel`.

The final cast targets the block pointer destination dtype. -/
def chunk_cumsum_scalar_surface
  (S O : RegionName) (T BT : Nat) : ComputeKernel := triton {
  i_bh = tl.program_id(0)
  b_z = tl.zeros([], dtype=tl.float32)
  for i_t in range($(0), tl.cdiv($(T), $(BT)), $(1)) {
    p_s = tl.make_block_ptr(base=S + i_bh * $(T), shape=($(T)),
      strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
    p_o = tl.make_block_ptr(base=O + i_bh * $(T), shape=($(T)),
      strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
    b_s = tl.load(p_s, boundary_check=([0] : List Nat)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0) + b_z[None]
    b_zz = tl.sum(b_s, axis=0)
    b_z += b_zz
    tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0] : List Nat))
  }
}

/-- The full scalar chunk-cumsum surface lowers to the algorithm layer,
including the carried scalar accumulator across chunks. -/
theorem chunk_cumsum_scalar_surface_toAlgorithm_supported
    (S O : RegionName) (T BT : Nat) :
    ∃ alg, (chunk_cumsum_scalar_surface S O T BT).toAlgorithm? = Except.ok alg := by
  simp [chunk_cumsum_scalar_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented block store slice of `chunk_cumsum_kernel.py`'s
`chunk_global_cumsum_scalar_kernel`.

The full kernel scans chunks while carrying `b_z`. This slice models one chunk
iteration with a precomputed `BO` vector and proves the boundary-checked store
into `O`. -/
def chunk_cumsum_scalar_store_slice
    (BO O : RegionName) (T BT : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  i_t = tl.program_id(1)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  mask = offs_t < $(T)
  b_o = tl.load(BO + i_bh * $(T) + offs_t, mask=mask, other=0.0)
  tl.store(O + i_bh * $(T) + offs_t, (b_o).to(O.dtype.element_ty), mask=mask)
}

def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 1 * BT + i.val

def active (s : BlockState) (T BT : Nat) (i : Fin BT) : Prop :=
  tIndex s BT i < T

instance activeDecidable (s : BlockState) (T BT : Nat) (i : Fin BT) :
    Decidable (active s T BT i) := by
  unfold active
  infer_instance

def vecOffset (s : BlockState) (T BT : Nat) (i : Fin BT) : Nat :=
  s.pids 0 * T + tIndex s BT i

noncomputable def storeValue (s : BlockState) (BO : RegionName) (T BT : Nat)
    (i : Fin BT) : ℝ :=
  WithBot.unbotD 0
    (if active s T BT i then some (s.readMem BO (vecOffset s T BT i))
    else some (0.0 : ℝ))

theorem chunk_cumsum_scalar_store_slice_correct
    (BO O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => vecOffset s T BT i)) :
    ∀ i : Fin BT,
      let outAddr := vecOffset s T BT i
      (exec (chunk_cumsum_scalar_store_slice BO O T BT) s).map
          (·.readMem O outAddr)
        = some (if active s T BT i then storeValue s BO T BT i
          else s.readMem O outAddr) := by
  intro i
  simp [exec, chunk_cumsum_scalar_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, tIndex, active, vecOffset]
  let offsetFn : TileIndex [BT] → Nat :=
    fun idx => s.pids 0 * T + (s.pids 1 * BT + idx.1.val)
  let valueFn : TileIndex [BT] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 1 * BT + idx.1.val < T then
        some (s.readMem BO (s.pids 0 * T + (s.pids 1 * BT + idx.1.val)))
      else some (0.0 : ℝ))
  let P : TileIndex [BT] → Prop := fun idx => s.pids 1 * BT + idx.1.val < T
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : vecOffset s T BT a = vecOffset s T BT b := by
      simpa [offsetFn, vecOffset, tIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem O (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BT])).readMem O (offsetFn (i, PUnit.unit)) =
    if active s T BT i then storeValue s BO T BT i
    else s.readMem O (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : i.val + s.pids 1 * BT < T
  · simp [P, valueFn, active, storeValue, vecOffset, tIndex, Nat.add_comm,
      Nat.add_left_comm, Nat.add_assoc, hi]
  · have hi' : ¬ s.pids 1 * BT + i.val < T := by
      simpa [Nat.add_comm] using hi
    simp [P, active, storeValue, tIndex, hi']

theorem chunk_cumsum_scalar_store_slice_compute_correct
    (BO O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => vecOffset s T BT i)) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_store_slice BO O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => active s T BT i)
        (fun i => (O, vecOffset s T BT i)))
      (expected := fun i : Fin BT => storeValue s BO T BT i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_cumsum_scalar_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := chunk_cumsum_scalar_store_slice_correct BO O T BT s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented single-iteration cumsum slice of
`chunk_cumsum_kernel.py`'s `chunk_global_cumsum_scalar_kernel`.

This models one loop iteration after the carried scalar `b_z` has been
materialized in `Carry`: it loads the current source block, computes
`tl.cumsum(b_s, axis=0) + b_z`, and stores the masked output block. -/
def chunk_cumsum_scalar_cumsum_slice
    (S Carry O : RegionName) (T BT : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  i_t = tl.program_id(1)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  mask = offs_t < $(T)
  b_s = tl.load(S + i_bh * $(T) + offs_t, mask=mask, other=0.0).to(tl.float32)
  b_z = tl.load(Carry + i_bh).to(tl.float32)
  b_o = tl.cumsum(b_s, axis=0) + b_z[None]
  tl.store(O + i_bh * $(T) + offs_t, (b_o).to(O.dtype.element_ty), mask=mask)
}

noncomputable def cumsumInputTile (s : BlockState) (S : RegionName) (T BT : Nat) :
    Tile .real [BT] :=
  { data := fun idx =>
      if s.pids 1 * BT + idx.1.val < T then
        some (s.readMem S (s.pids 0 * T + (s.pids 1 * BT + idx.1.val)))
      else some (0.0 : ℝ) }

noncomputable def cumsumStoreValue
    (s : BlockState) (S Carry : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun a b : ℝ => a + b)
      ((Tile.scan .sum ⟨0, by simp⟩ (cumsumInputTile s S T BT)).data (i, PUnit.unit))
      (some (s.readMem Carry (s.pids 0))))

theorem chunk_cumsum_scalar_cumsum_slice_correct
    (S Carry O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => vecOffset s T BT i)) :
    ∀ i : Fin BT,
      let outAddr := vecOffset s T BT i
      (exec (chunk_cumsum_scalar_cumsum_slice S Carry O T BT) s).map
          (·.readMem O outAddr)
        = some (if active s T BT i then cumsumStoreValue s S Carry T BT i
          else s.readMem O outAddr) := by
  intro i
  simp [exec, chunk_cumsum_scalar_cumsum_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.scan,
        NumericDType.add, NumericDType.mul, ComparableDType.lt, tIndex, active,
        vecOffset, cumsumInputTile, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BT] → Nat :=
    fun idx => s.pids 0 * T + (s.pids 1 * BT + idx.1.val)
  let valueFn : TileIndex [BT] → ℝ :=
    fun idx => cumsumStoreValue s S Carry T BT idx.1
  let P : TileIndex [BT] → Prop := fun idx => s.pids 1 * BT + idx.1.val < T
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : vecOffset s T BT a = vecOffset s T BT b := by
      simpa [offsetFn, vecOffset, tIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : s.pids 1 * BT + i.val < T
  · simp [cumsumStoreValue, cumsumInputTile, active, vecOffset, tIndex, Option.map, hi]
  · simp [cumsumStoreValue, cumsumInputTile, active, vecOffset, tIndex, offsetFn, hi]

theorem chunk_cumsum_scalar_cumsum_slice_compute_correct
    (S Carry O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => vecOffset s T BT i)) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_cumsum_slice S Carry O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => active s T BT i)
        (fun i => (O, vecOffset s T BT i)))
      (expected := fun i : Fin BT => cumsumStoreValue s S Carry T BT i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_cumsum_scalar_cumsum_slice, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := chunk_cumsum_scalar_cumsum_slice_correct S Carry O T BT s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrappers

`chunk_cumsum_kernel.py`'s checked tests use `B = 2`, `H = 3`, `T = 4`.
The autotune set includes `BT = 16`, which covers the single Python chunk for
this test shape. -/

theorem chunk_cumsum_scalar_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 16 => vecOffset s 4 16 i) := by
  intro a b h
  simp [vecOffset, tIndex] at h
  exact Fin.ext (by omega)

theorem chunk_cumsum_scalar_store_python_test_shape_compute_correct
    (BO O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_store_slice BO O 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 4 16 i)
        (fun i => (O, vecOffset s 4 16 i)))
      (expected := fun i : Fin 16 => storeValue s BO 4 16 i) := by
  exact chunk_cumsum_scalar_store_slice_compute_correct BO O 4 16 s
    (chunk_cumsum_scalar_python_test_shape_offset_injective s)

theorem chunk_cumsum_scalar_cumsum_python_test_shape_compute_correct
    (S Carry O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_cumsum_slice S Carry O 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 4 16 i)
        (fun i => (O, vecOffset s 4 16 i)))
      (expected := fun i : Fin 16 => cumsumStoreValue s S Carry 4 16 i) := by
  exact chunk_cumsum_scalar_cumsum_slice_compute_correct S Carry O 4 16 s
    (chunk_cumsum_scalar_python_test_shape_offset_injective s)

end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
