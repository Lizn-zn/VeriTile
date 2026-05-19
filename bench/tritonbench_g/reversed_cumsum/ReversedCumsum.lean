import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ReversedCumsum

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `reversed_cumsum.py`'s
`chunk_global_reversed_cumsum_vector_kernel`.

The source traverses chunks in reverse. This surface preserves the reverse
range and carries `b_z` across chunks. -/
def reversed_cumsum_surface
    (s z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
  b_z = tl.zeros([$(BS)], dtype=tl.float32)
  for i_t in range(tl.cdiv($(T), $(BT)) - $(1), -$(1), -$(1)) {
    p_s = tl.make_block_ptr(base=s + i_bh * $(s_s_h),
      shape=($(T), $(S)), strides=($(s_s_t), $(s_s_d)),
      offsets=(i_t * $(BT), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
    p_z = tl.make_block_ptr(base=z + i_bh * $(s_s_h),
      shape=($(T), $(S)), strides=($(s_s_t), $(s_s_d)),
      offsets=(i_t * $(BT), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
    b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
    b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
    tl.store(p_z, (b_c).to(p_z.dtype.element_ty), boundary_check=([0, 1] : List Nat))
    if i_t >= $(0) {
      b_z += tl.sum(b_s, 0)
    }
  }
}

/-- The full reverse traversal cumsum surface lowers to the algorithm layer,
including the negative-step loop and carried `b_z` accumulator. -/
theorem reversed_cumsum_surface_toAlgorithm_supported
    (s z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ∃ alg, (reversed_cumsum_surface s z s_s_h s_s_t s_s_d T S BT BS).toAlgorithm?
      = Except.ok alg := by
  simp [reversed_cumsum_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Specialized transcription of `reversed_cumsum.py`'s
`chunk_global_reversed_cumsum_vector_kernel` for the single-`BT` block path.

When the time dimension fits in one `BT` tile, the Python reverse traversal has
one iteration with `b_z = 0`. This surface preserves the triangular mask
`o_i[:, None] <= o_i[None, :]`, the boundary-checked input load, the
`tl.dot(m_s, b_s, allow_tf32=false)` reversed cumulative sum, and the
boundary-checked store. -/
def reversed_cumsum_single_block_surface
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
  p_s = tl.make_block_ptr(base=SReg + i_bh * $(s_s_h),
    shape=($(T), $(SSize)), strides=($(s_s_t), $(s_s_d)),
    offsets=($(0), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
  p_z = tl.make_block_ptr(base=Z + i_bh * $(s_s_h),
    shape=($(T), $(SSize)), strides=($(s_s_t), $(s_s_d)),
    offsets=($(0), i_s * $(BS)), block_shape=($(BT), $(BS)), order=(1, 0))
  b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  b_c = tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(p_z, (b_c).to(p_z.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- The single-block reverse cumsum surface lowers to the algorithm layer. -/
theorem reversed_cumsum_single_block_surface_toAlgorithm_supported
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ∃ alg, (reversed_cumsum_single_block_surface SReg Z s_s_h s_s_t s_s_d T
      SSize BT BS).toAlgorithm? = Except.ok alg := by
  simp [reversed_cumsum_single_block_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented block store surface slice of `reversed_cumsum.py`'s
`chunk_global_reversed_cumsum_vector_kernel`.

The full kernel computes a per-feature reversed chunk cumsum tile. This slice
starts from a precomputed `BC` tile for one `(i_s, i_bh, i_t)` block and proves
the boundary-checked writeback into `Z`. -/
def reversed_cumsum_store_slice
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
      offs_s[None, :] * $(s_s_d), (b_c).to(Z.dtype.element_ty), mask=mask)
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

theorem reversed_cumsum_store_slice_correct
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ∀ idx : TileIndex [BT, BS],
      let outAddr := tileOffset s s_s_h s_s_t s_s_d BT BS idx
      (exec (reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS)
          s).map (·.readMem Z outAddr)
        = some (if active s T S BT BS idx then
            storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx
          else s.readMem Z outAddr) := by
  intro idx
  simp [exec, reversed_cumsum_store_slice, stepStmts, stepStmt, evalOp,
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

theorem reversed_cumsum_store_slice_compute_correct
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] => (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [reversed_cumsum_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := reversed_cumsum_store_slice_correct BC Z s_s_h s_s_t s_s_d T S BT BS
    s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Computed per-chunk reversed-cumsum slice

This slice models one reverse loop iteration after the carry vector `b_z` has
been materialized in `Carry`: it loads the source block, computes
`b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)` with the upper-triangular
reverse-cumsum mask, and stores the masked result. -/

def reversed_cumsum_cumsum_slice
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  i_t = tl.program_id(2)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  offs_s = i_s * $(BS) + tl.arange(0, $(BS))
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] <= o_i[None, :], 1.0, 0.0)
  mask = (offs_t[:, None] < $(T)) & (offs_s[None, :] < $(S))
  b_s = tl.load(SReg + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), mask=mask, other=0.0)
  b_z = tl.load(Carry + i_bh * $(s_s_h) + offs_s * $(s_s_d),
    mask=offs_s < $(S), other=0.0)
  b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(Z + i_bh * $(s_s_h) + offs_t[:, None] * $(s_s_t) +
      offs_s[None, :] * $(s_s_d), (b_c).to(Z.dtype.element_ty), mask=mask)
}

noncomputable def upperTriTile (BT : Nat) : Tile .real [BT, BT] :=
  { data := fun idx =>
      if idx.1.val <= idx.2.1.val then some (1.0 : ℝ) else some (0.0 : ℝ) }

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
      ((Tile.dot [] (upperTriTile BT)
        (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
          (idx.1, idx.2.1, PUnit.unit)))

theorem reversed_cumsum_cumsum_slice_correct
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ∀ idx : TileIndex [BT, BS],
      let outAddr := tileOffset s s_s_h s_s_t s_s_d BT BS idx
      (exec (reversed_cumsum_cumsum_slice SReg Carry Z s_s_h s_s_t s_s_d
            T S BT BS) s).map (·.readMem Z outAddr)
        = some (if active s T S BT BS idx then
            cumsumStoreValue s SReg Carry s_s_h s_s_t s_s_d T S BT BS idx
          else s.readMem Z outAddr) := by
  intro idx
  simp [exec, reversed_cumsum_cumsum_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, ComparableDType.le, tIndex, sIndex, active,
        tileOffset, sourceTile, upperTriTile, carryValue,
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
      sourceTile, upperTriTile, carryValue, Tile.dot, hActive]
  · simp [offsetFn, active, tIndex, sIndex, tileOffset, hActive]

theorem reversed_cumsum_cumsum_slice_compute_correct
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s s_s_h s_s_t s_s_d BT BS idx)) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z s_s_h s_s_t
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
  · simp [reversed_cumsum_cumsum_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := reversed_cumsum_cumsum_slice_correct SReg Carry Z s_s_h s_s_t
    s_s_d T S BT BS s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Active-masked readback variants

For Python cases with `S < BS`, padded feature lanes can alias active addresses
if the whole `[BT, BS]` block is considered. The final store only observes lanes
where `(offs_t < T) & (offs_s < S)`, so these variants require collision
freedom only among active lanes. -/

theorem reversed_cumsum_store_slice_active_compute_correct
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], active s T S BT BS idx →
      ∀ k : TileIndex [BT, BS], active s T S BT BS k →
        tileOffset s s_s_h s_s_t s_s_d BT BS k =
          tileOffset s s_s_h s_s_t s_s_d BT BS idx → k = idx) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] => (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        storeValue s BC s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [reversed_cumsum_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  simp [exec, reversed_cumsum_store_slice, stepStmts, stepStmt, evalOp,
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

theorem reversed_cumsum_cumsum_slice_active_compute_correct
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], active s T S BT BS idx →
      ∀ k : TileIndex [BT, BS], active s T S BT BS k →
        tileOffset s s_s_h s_s_t s_s_d BT BS k =
          tileOffset s s_s_h s_s_t s_s_d BT BS idx → k = idx) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z s_s_h s_s_t
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
  · simp [reversed_cumsum_cumsum_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  simp [exec, reversed_cumsum_cumsum_slice, stepStmts, stepStmt, evalOp,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, Tile.dot, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, ComparableDType.le, tIndex, sIndex, active,
        tileOffset, sourceTile, upperTriTile, carryValue,
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

theorem reversed_cumsum_python_case1_active_no_collision
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

theorem reversed_cumsum_python_case2_active_no_collision
    (s : BlockState) :
    ∀ idx : TileIndex [16, 32], active s 8 8 16 32 idx →
      ∀ k : TileIndex [16, 32], active s 8 8 16 32 k →
        tileOffset s 64 8 1 16 32 k = tileOffset s 64 8 1 16 32 idx →
          k = idx := by
  rintro ⟨⟨ti, hti⟩, ⟨si, hsi⟩, _⟩ hi ⟨⟨tk, htk⟩, ⟨sk, hsk⟩, _⟩ hk h
  simp [active, tIndex, sIndex] at hi hk
  simp [tileOffset, tIndex, sIndex] at h
  have ht : tk = ti := by omega
  have hs : sk = si := by omega
  subst tk
  subst sk
  rfl

theorem reversed_cumsum_python_case3_active_no_collision
    (s : BlockState) :
    ∀ idx : TileIndex [16, 32], active s 16 16 16 32 idx →
      ∀ k : TileIndex [16, 32], active s 16 16 16 32 k →
        tileOffset s 256 16 1 16 32 k = tileOffset s 256 16 1 16 32 idx →
          k = idx := by
  rintro ⟨⟨ti, hti⟩, ⟨si, hsi⟩, _⟩ hi ⟨⟨tk, htk⟩, ⟨sk, hsk⟩, _⟩ hk h
  simp [active, tIndex, sIndex] at hi hk
  simp [tileOffset, tIndex, sIndex] at h
  have ht : tk = ti := by omega
  have hs : sk = si := by omega
  subst tk
  subst sk
  rfl

theorem reversed_cumsum_store_python_case1_compute_correct
    (BC Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_store_slice BC Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 20 5 1 4 5 16 32 idx) := by
  exact reversed_cumsum_store_slice_active_compute_correct BC Z
    20 5 1 4 5 16 32 s
    (reversed_cumsum_python_case1_active_no_collision s)

theorem reversed_cumsum_cumsum_python_case1_compute_correct
    (SReg Carry Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 20 5 1 4 5 16 32 idx) := by
  exact reversed_cumsum_cumsum_slice_active_compute_correct SReg Carry Z
    20 5 1 4 5 16 32 s
    (reversed_cumsum_python_case1_active_no_collision s)

theorem reversed_cumsum_store_python_case2_compute_correct
    (BC Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_store_slice BC Z 64 8 1 8 8 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 8 16 32 idx)
        (fun idx => (Z, tileOffset s 64 8 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 64 8 1 8 8 16 32 idx) := by
  exact reversed_cumsum_store_slice_active_compute_correct BC Z
    64 8 1 8 8 16 32 s
    (reversed_cumsum_python_case2_active_no_collision s)

theorem reversed_cumsum_cumsum_python_case2_compute_correct
    (SReg Carry Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z 64 8 1 8 8 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 8 16 32 idx)
        (fun idx => (Z, tileOffset s 64 8 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 64 8 1 8 8 16 32 idx) := by
  exact reversed_cumsum_cumsum_slice_active_compute_correct SReg Carry Z
    64 8 1 8 8 16 32 s
    (reversed_cumsum_python_case2_active_no_collision s)

theorem reversed_cumsum_store_python_case3_compute_correct
    (BC Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_store_slice BC Z 256 16 1 16 16 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 16 16 16 32 idx)
        (fun idx => (Z, tileOffset s 256 16 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 256 16 1 16 16 16 32 idx) := by
  exact reversed_cumsum_store_slice_active_compute_correct BC Z
    256 16 1 16 16 16 32 s
    (reversed_cumsum_python_case3_active_no_collision s)

theorem reversed_cumsum_cumsum_python_case3_compute_correct
    (SReg Carry Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z 256 16 1 16 16 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 16 16 16 32 idx)
        (fun idx => (Z, tileOffset s 256 16 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 256 16 1 16 16 16 32 idx) := by
  exact reversed_cumsum_cumsum_slice_active_compute_correct SReg Carry Z
    256 16 1 16 16 16 32 s
    (reversed_cumsum_python_case3_active_no_collision s)

/-! ## Python test-shape wrappers

`reversed_cumsum.py` includes a checked case `B = 3`, `H = 3`, `T = 32`,
`S = 32`. For contiguous `[B, H, T, S]` tensors, the per-head strides passed to
the kernel are `(1024, 32, 1)`. With the autotune `BT = 16` and fixed `BS = 32`,
the full block address map is injective. -/

theorem reversed_cumsum_python_case4_offset_injective
    (s : BlockState) :
    Function.Injective
      (fun idx : TileIndex [16, 32] => tileOffset s 1024 32 1 16 32 idx) := by
  rintro ⟨⟨ta, hta⟩, ⟨sa, hsa⟩, _⟩ ⟨⟨tb, htb⟩, ⟨sb, hsb⟩, _⟩ h
  simp [tileOffset, tIndex, sIndex] at h
  have ht : ta = tb := by omega
  have hs : sa = sb := by omega
  subst tb
  subst sb
  rfl

theorem reversed_cumsum_store_python_case4_compute_correct
    (BC Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_store_slice BC Z 1024 32 1 32 32 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 32 32 16 32 idx)
        (fun idx : TileIndex [16, 32] => (Z, tileOffset s 1024 32 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 1024 32 1 32 32 16 32 idx) := by
  exact reversed_cumsum_store_slice_compute_correct BC Z 1024 32 1 32 32 16 32 s
    (reversed_cumsum_python_case4_offset_injective s)

theorem reversed_cumsum_cumsum_python_case4_compute_correct
    (SReg Carry Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := reversed_cumsum_cumsum_slice SReg Carry Z 1024 32 1 32 32 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 32 32 16 32 idx)
        (fun idx : TileIndex [16, 32] => (Z, tileOffset s 1024 32 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 1024 32 1 32 32 16 32 idx) := by
  exact reversed_cumsum_cumsum_slice_compute_correct SReg Carry Z
    1024 32 1 32 32 16 32 s
    (reversed_cumsum_python_case4_offset_injective s)

end VeriTile.Bench.TritonBenchG.ReversedCumsum
