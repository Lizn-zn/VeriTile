import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `chunk_cumsum_vector.py`'s
`chunk_global_cumsum_vector_kernel`.

The final cast targets the block pointer destination dtype in Python. -/
def chunk_cumsum_vector_surface
    (S Z : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  b_z = tl.zeros([$(BS)], dtype=tl.float32)
  for i_t in range($(0), tl.cdiv($(T), $(BT)), $(1)) {
    p_s = tl.make_block_ptr(base=S + i_bh * $(s_s_h), shape=($(T), $(SSize)),
      strides=($(s_s_t), $(s_s_d)), offsets=(i_t * $(BT), i_s * $(BS)),
      block_shape=($(BT), $(BS)), order=(1, 0))
    p_z = tl.make_block_ptr(base=Z + i_bh * $(s_s_h), shape=($(T), $(SSize)),
      strides=($(s_s_t), $(s_s_d)), offsets=(i_t * $(BT), i_s * $(BS)),
      block_shape=($(BT), $(BS)), order=(1, 0))
    b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
    b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
    tl.store(p_z, (b_c).to(p_z.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    if i_t >= $(0) {
      b_z += tl.sum(b_s, 0)
    }
  }
}

/-- The full vector chunk-cumsum surface lowers to the algorithm layer,
including the carried vector accumulator across chunks. -/
theorem chunk_cumsum_vector_surface_toAlgorithm_supported
    (S Z : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ∃ alg, (chunk_cumsum_vector_surface S Z s_s_h s_s_t s_s_d T SSize BT
      BS).toAlgorithm? = Except.ok alg := by
  simp [chunk_cumsum_vector_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented block store slice of `chunk_cumsum_vector.py`'s
`chunk_global_cumsum_vector_kernel`.

The full kernel computes a per-feature chunk cumsum tile. This slice starts from
a precomputed `BC` tile for one `(i_s, i_bh, i_t)` block and proves the
boundary-checked writeback into `Z`. -/
def chunk_cumsum_vector_store_slice
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_c = tl.load(BC + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), b_c, mask=mask)
}

def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat :=
  s.pids 2 * BT + i.val

def sIndex (s : BlockState) (BS : Nat) (j : Fin BS) : Nat :=
  s.pids 0 * BS + j.val

def active (s : BlockState) (T S BT BS : Nat) (idx : TileIndex [BT, BS]) : Prop :=
  tIndex s BT idx.1 < T ∧ sIndex s BS idx.2.1 < S

instance activeDecidable (s : BlockState) (T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Decidable (active s T S BT BS idx) := by
  unfold active
  infer_instance

def tileOffset (s : BlockState) (s_s_h s_s_t s_s_d BT BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + tIndex s BT idx.1 * s_s_t +
    sIndex s BS idx.2.1 * s_s_d

noncomputable def storeValue (s : BlockState) (BC : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (if active s T S BT BS idx then
      some (s.readMem BC (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
    else some (0.0 : ℝ))

theorem chunk_cumsum_vector_store_slice_correct
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ∀ idx : TileIndex [BT, BS],
      let outAddr := tileOffset s s_s_h s_s_t s_s_d BT BS idx
      (exec (chunk_cumsum_vector_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS)
          s).map (·.readMem Z outAddr)
        = some (if active s T S BT BS idx then
            storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx
          else s.readMem Z outAddr) := by
  intro idx
  simp [exec, chunk_cumsum_vector_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        tIndex, sIndex, active, tileOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BS] → Nat :=
    fun idx => s.pids 1 * s_s_h + (s.pids 2 * BT + idx.1.val) * s_s_t +
      (s.pids 0 * BS + idx.2.1.val) * s_s_d
  let valueFn : TileIndex [BT, BS] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 2 * BT + idx.1.val < T ∧
          s.pids 0 * BS + idx.2.1.val < S then
        some (s.readMem BC (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BT, BS] → Prop :=
    fun idx => s.pids 2 * BT + idx.1.val < T ∧
      s.pids 0 * BS + idx.2.1.val < S
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, tIndex, sIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Z (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BT, BS])).readMem Z (offsetFn idx) =
    if P idx then storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx
    else s.readMem Z (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 2 * BT + idx.1.val < T ∧ s.pids 0 * BS + idx.2.1.val < S
  · rfl
  · rfl

theorem chunk_cumsum_vector_store_slice_compute_correct
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] => (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_cumsum_vector_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_cumsum_vector_store_slice_correct BC Z s_s_h s_s_t s_s_d T S BT BS
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Computed per-chunk cumsum slice

The store slice above starts from a precomputed `BC` tile. This slice models one
loop iteration of the Python kernel after the carry vector `b_z` has been
materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)`, and stores the masked
result. -/

def chunk_cumsum_vector_cumsum_slice
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_z = tl.load(Carry + i_bh * $(s_s_h) + offs_s * $(s_s_d),
    mask=offs_s < $(S), other=0.0)
  b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), b_c, mask=mask)
}

noncomputable def lowerTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val >= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }

noncomputable def sourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if active s T S BT BS idx then
        some (s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS idx))
      else some (0.0 : ℝ) }

noncomputable def carryValue
    (s : BlockState) (Carry : RegionName) (s_s_h s_s_d S BS : Nat)
    (j : Fin BS) : WithBot ℝ :=
  if sIndex s BS j < S then
    some (s.readMem Carry (s.pids 1 * s_s_h + sIndex s BS j * s_s_d))
  else some (0.0 : ℝ)

noncomputable def cumsumStoreValue
    (s : BlockState) (SReg Carry : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun carry dot : ℝ => carry + dot)
      (carryValue s Carry s_s_h s_s_d S BS idx.2.1)
      ((Tile.dot [] (lowerTriTile BT)
        (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
          (idx.1, idx.2.1, PUnit.unit)))

theorem chunk_cumsum_vector_cumsum_slice_correct
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ∀ idx : TileIndex [BT, BS],
      let outAddr := tileOffset s s_s_h s_s_t s_s_d BT BS idx
      (exec (chunk_cumsum_vector_cumsum_slice SReg Carry Z s_s_h s_s_t s_s_d
            T S BT BS) s).map (·.readMem Z outAddr)
        = some (if active s T S BT BS idx then
            cumsumStoreValue s SReg Carry s_s_h s_s_t s_s_d T S BT BS idx
          else s.readMem Z outAddr) := by
  intro idx
  simp [exec, chunk_cumsum_vector_cumsum_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, ComparableDType.ge, tIndex, sIndex, active,
        tileOffset, sourceTile, lowerTriTile, carryValue,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BS] → Nat :=
    fun idx => s.pids 1 * s_s_h + (s.pids 2 * BT + idx.1.val) * s_s_t +
      (s.pids 0 * BS + idx.2.1.val) * s_s_d
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, tIndex, sIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 2 * BT + idx.1.val < T ∧ s.pids 0 * BS + idx.2.1.val < S
  · simp [offsetFn, active, tIndex, sIndex, tileOffset, cumsumStoreValue,
      sourceTile, lowerTriTile, carryValue, Tile.dot, hActive]
  · simp [offsetFn, active, tIndex, sIndex, tileOffset, hActive]

theorem chunk_cumsum_vector_cumsum_slice_compute_correct
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        cumsumStoreValue s SReg Carry s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_cumsum_vector_cumsum_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_cumsum_vector_cumsum_slice_correct SReg Carry Z s_s_h s_s_t
    s_s_d T S BT BS s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Active-masked readback variants

For the Python tests with `S < BS`, padded feature lanes can alias active
addresses when considered over the whole `[BT, BS]` block. The store contract
only observes `active` lanes, so the theorem below uses the weaker semantic
lemma requiring no collision among active writes that target the active lane
being read. -/

theorem chunk_cumsum_vector_store_slice_active_compute_correct
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], active s T S BT BS idx →
      ∀ k : TileIndex [BT, BS], active s T S BT BS k →
        tileOffset s s_s_h s_s_t s_s_d BT BS k =
          tileOffset s s_s_h s_s_t s_s_d BT BS idx → k = idx) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] => (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_cumsum_vector_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  simp [exec, chunk_cumsum_vector_store_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        tIndex, sIndex, active, tileOffset, TileShape.dropInsertedIndex] at hExec
  subst s'
  let offsetFn : TileIndex [BT, BS] → Nat :=
    fun idx => s.pids 1 * s_s_h + (s.pids 2 * BT + idx.1.val) * s_s_t +
      (s.pids 0 * BS + idx.2.1.val) * s_s_d
  let valueFn : TileIndex [BT, BS] → ℝ :=
    fun idx => WithBot.unbotD 0
      (if s.pids 2 * BT + idx.1.val < T ∧
          s.pids 0 * BS + idx.2.1.val < S then
        some (s.readMem BC (offsetFn idx))
      else some (0.0 : ℝ))
  let P : TileIndex [BT, BS] → Prop :=
    fun idx => s.pids 2 * BT + idx.1.val < T ∧
      s.pids 0 * BS + idx.2.1.val < S
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Z (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BT, BS])).readMem Z (offsetFn idx) =
    storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _ P idx]
  · rfl
  · simpa [P, active, tIndex, sIndex] using hActive
  · intro k hPk heq
    exact hNoCollision idx (by simpa [active, tIndex, sIndex] using hActive) k
      (by simpa [P, active, tIndex, sIndex] using hPk)
      (by simpa [offsetFn, tileOffset, tIndex, sIndex] using heq)

theorem chunk_cumsum_vector_cumsum_slice_active_compute_correct
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], active s T S BT BS idx →
      ∀ k : TileIndex [BT, BS], active s T S BT BS k →
        tileOffset s s_s_h s_s_t s_s_d BT BS k =
          tileOffset s s_s_h s_s_t s_s_d BT BS idx → k = idx) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        cumsumStoreValue s SReg Carry s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_cumsum_vector_cumsum_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  simp [exec, chunk_cumsum_vector_cumsum_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, ComparableDType.ge, tIndex, sIndex, active,
        tileOffset, sourceTile, lowerTriTile, carryValue,
        TileShape.dropInsertedIndex] at hExec
  subst s'
  let offsetFn : TileIndex [BT, BS] → Nat :=
    fun idx => s.pids 1 * s_s_h + (s.pids 2 * BT + idx.1.val) * s_s_t +
      (s.pids 0 * BS + idx.2.1.val) * s_s_d
  change (List.foldl
      (fun (acc : BlockState) i =>
        if active s T S BT BS i then
          acc.writeMem Z (offsetFn i)
            (cumsumStoreValue s SReg Carry s_s_h s_s_t s_s_d T S BT BS i)
        else acc)
      _ (TileShape.allIndices [BT, BS])).readMem Z (offsetFn idx) =
    cumsumStoreValue s SReg Carry s_s_h s_s_t s_s_d T S BT BS idx
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _
      (active s T S BT BS) idx hActive]
  intro k hk heq
  exact hNoCollision idx hActive k hk
    (by simpa [offsetFn, tileOffset, tIndex, sIndex] using heq)

theorem chunk_cumsum_vector_python_case1_active_no_collision
    (s : BlockState) :
    ∀ idx : TileIndex [16, 32], active s 4 5 16 32 idx →
      ∀ k : TileIndex [16, 32], active s 4 5 16 32 k →
        tileOffset s 20 5 1 16 32 k = tileOffset s 20 5 1 16 32 idx →
          k = idx := by
  rintro ⟨⟨ti, hti⟩, ⟨si, hsi⟩, _⟩ hi ⟨⟨tk, htk⟩, ⟨sk, hsk⟩, _⟩ hk h
  simp [active, tIndex, sIndex] at hi hk
  simp [tileOffset, tIndex, sIndex] at h
  have ht : tk = ti := by omega
  have hs : sk = si := by omega
  subst tk
  subst sk
  rfl

theorem chunk_cumsum_vector_python_case2_active_no_collision
    (s : BlockState) :
    ∀ idx : TileIndex [16, 32], active s 8 10 16 32 idx →
      ∀ k : TileIndex [16, 32], active s 8 10 16 32 k →
        tileOffset s 80 10 1 16 32 k = tileOffset s 80 10 1 16 32 idx →
          k = idx := by
  rintro ⟨⟨ti, hti⟩, ⟨si, hsi⟩, _⟩ hi ⟨⟨tk, htk⟩, ⟨sk, hsk⟩, _⟩ hk h
  simp [active, tIndex, sIndex] at hi hk
  simp [tileOffset, tIndex, sIndex] at h
  have ht : tk = ti := by omega
  have hs : sk = si := by omega
  subst tk
  subst sk
  rfl

theorem chunk_cumsum_vector_python_case3_active_no_collision
    (s : BlockState) :
    ∀ idx : TileIndex [16, 32], active s 1 5 16 32 idx →
      ∀ k : TileIndex [16, 32], active s 1 5 16 32 k →
        tileOffset s 5 5 1 16 32 k = tileOffset s 5 5 1 16 32 idx →
          k = idx := by
  rintro ⟨⟨ti, hti⟩, ⟨si, hsi⟩, _⟩ hi ⟨⟨tk, htk⟩, ⟨sk, hsk⟩, _⟩ hk h
  simp [active, tIndex, sIndex] at hi hk
  simp [tileOffset, tIndex, sIndex] at h
  have ht : tk = ti := by omega
  have hs : sk = si := by omega
  subst tk
  subst sk
  rfl

theorem chunk_cumsum_vector_store_python_case1_compute_correct
    (BC Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 20 5 1 4 5 16 32 idx) := by
  exact chunk_cumsum_vector_store_slice_active_compute_correct BC Z
    20 5 1 4 5 16 32 s
    (chunk_cumsum_vector_python_case1_active_no_collision s)

theorem chunk_cumsum_vector_cumsum_python_case1_compute_correct
    (SReg Carry Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 20 5 1 4 5 16 32 idx) := by
  exact chunk_cumsum_vector_cumsum_slice_active_compute_correct SReg Carry Z
    20 5 1 4 5 16 32 s
    (chunk_cumsum_vector_python_case1_active_no_collision s)

theorem chunk_cumsum_vector_store_python_case2_compute_correct
    (BC Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z 80 10 1 8 10 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 10 16 32 idx)
        (fun idx => (Z, tileOffset s 80 10 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 80 10 1 8 10 16 32 idx) := by
  exact chunk_cumsum_vector_store_slice_active_compute_correct BC Z
    80 10 1 8 10 16 32 s
    (chunk_cumsum_vector_python_case2_active_no_collision s)

theorem chunk_cumsum_vector_cumsum_python_case2_compute_correct
    (SReg Carry Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z 80 10 1 8 10 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 10 16 32 idx)
        (fun idx => (Z, tileOffset s 80 10 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 80 10 1 8 10 16 32 idx) := by
  exact chunk_cumsum_vector_cumsum_slice_active_compute_correct SReg Carry Z
    80 10 1 8 10 16 32 s
    (chunk_cumsum_vector_python_case2_active_no_collision s)

theorem chunk_cumsum_vector_store_python_case3_compute_correct
    (BC Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z 5 5 1 1 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 1 5 16 32 idx)
        (fun idx => (Z, tileOffset s 5 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 5 5 1 1 5 16 32 idx) := by
  exact chunk_cumsum_vector_store_slice_active_compute_correct BC Z
    5 5 1 1 5 16 32 s
    (chunk_cumsum_vector_python_case3_active_no_collision s)

theorem chunk_cumsum_vector_cumsum_python_case3_compute_correct
    (SReg Carry Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z 5 5 1 1 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 1 5 16 32 idx)
        (fun idx => (Z, tileOffset s 5 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 5 5 1 1 5 16 32 idx) := by
  exact chunk_cumsum_vector_cumsum_slice_active_compute_correct SReg Carry Z
    5 5 1 1 5 16 32 s
    (chunk_cumsum_vector_python_case3_active_no_collision s)

end VeriTile.Bench.TritonBenchG.ChunkCumsumVector
