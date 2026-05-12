import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of `chunk_cumsum_kernel.py`'s
`chunk_global_cumsum_scalar_kernel`.

Python writes `b_z[None]`; this surface uses scalar broadcast `b_z`, which is
the same shape behavior in the DSL. The final cast targets `O.dtype.element_ty`
to mirror the block pointer destination dtype. -/
def chunk_cumsum_scalar_surface
  (S O : RegionName) (T BT : Nat) : ComputeKernel := triton {
  i_bh = tl.program_id(0)
  b_z = tl.zeros([], dtype=tl.float32)
  for i_t in range($(0), tl.cdiv($(T), $(BT)), $(1)) {
    p_s = tl.make_block_ptr(S + i_bh * $(T), base=$(0), shape=[$(T)],
      strides=[$(1)], offsets=[i_t * $(BT)], block_shape=[$(BT)])
    p_o = tl.make_block_ptr(O + i_bh * $(T), base=$(0), shape=[$(T)],
      strides=[$(1)], offsets=[i_t * $(BT)], block_shape=[$(BT)])
    b_s = tl.load(p_s, boundary_check=([0] : List Nat)).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0) + b_z
    b_zz = tl.sum(b_s, axis=0)
    b_z += b_zz
    tl.store(p_o, (b_o).to(O.dtype.element_ty), boundary_check=([0] : List Nat))
  }
}

/-- Proof-oriented block store slice of `chunk_cumsum_kernel.py`'s
`chunk_global_cumsum_scalar_kernel`.

The full kernel scans chunks while carrying `b_z`. This slice models one chunk
iteration with a precomputed `BO` vector and proves the boundary-checked store
into `O`. -/
def chunk_cumsum_scalar_store_slice
    (BO O : RegionName) (T BT : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(axis=0)
  i_t = tl.program_id(axis=1)
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

end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
