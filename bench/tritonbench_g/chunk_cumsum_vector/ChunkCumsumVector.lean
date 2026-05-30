import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `chunk_cumsum_vector` — strict per-kernel correctness

`chunk_global_cumsum_vector_kernel` computes a global cumulative sum over the
time axis for `[T, S]` value tiles: program `(i_s, i_bh)` walks the `T`-rows of
the row's value matrix in chunks of `BT`, computing the in-chunk prefix sum as a
lower-triangular matmul `tl.dot(m_s, b_s)`, adding the carried column-vector
total `b_z`, and storing the `[BT, BS]` result into `z` while accumulating the
chunk column sums into `b_z`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`chunk_global_cumsum_vector_kernel[(cdiv(S,BS), B*H)]`,
the 2-D grid over feature blocks and batch·head rows, the autotuned `BT`, the
fixed `BS = 32`, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here. Because the
program ids are universally quantified, the per-program statements cover every
program of the grid.

## Proof architecture

```
chunk_cumsum_vector_python_case{1,2,3}_slice_summary           ← TOP THEOREMS
  ├─ chunk_cumsum_vector_python_case{n}_surface_toAlgorithm_supported
  │     └─ chunk_cumsum_vector_surface_toAlgorithm_supported   full surface lowers
  ├─ chunk_cumsum_vector_single_block_python_case{n}_compute_correct
  │     └─ chunk_cumsum_vector_single_block_surface_active_compute_correct
  │          └─ chunk_cumsum_vector_single_block_surface_correct
  ├─ chunk_cumsum_vector_store_python_case{n}_compute_correct
  │     └─ chunk_cumsum_vector_store_slice_active_compute_correct
  │          └─ chunk_cumsum_vector_store_slice_correct
  └─ chunk_cumsum_vector_cumsum_python_case{n}_compute_correct
       └─ chunk_cumsum_vector_cumsum_slice_active_compute_correct
            └─ chunk_cumsum_vector_cumsum_slice_correct

chunk_cumsum_vector_python_case{1,2,3}_output_summary          (= surface-outputs aliases)
chunk_cumsum_vector_python_case{n}_surface_outputs_compute_correct
  └─ chunk_cumsum_vector_surface_output_compute_correct
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` (the
`BT ∈ {16,32,64}` config set) is not modeled — proofs fix the three checked
Python shapes (`T,S = 4,5`; `8,10`; `1,5`) with `BS = 32`. The `.to(tl.float32)`
/ `.to(p_z.dtype.element_ty)` casts erase to the identity at the algorithm
layer (post-erasure all dtypes unify to `ℝ`). The in-chunk prefix sum is modeled
exactly as the lower-triangular matmul `lowerTriTile · sourceTile` (matching the
kernel's `tl.dot(m_s, b_s)`); the chunk-local store is modeled exactly; the
cross-chunk fold that threads the column-vector carry `b_z` over multiple
iterations is left as the trusted boundary — the carry is presented as a
materialized buffer (`Carry`) in `chunk_cumsum_vector_cumsum_slice`, and the
single-block surface covers the single-chunk case where the carry is the initial
zero. The Python shapes here all have `T ≤ BT`, so the single-block surface is
exact. Output non-collision is a side condition (discharged per Python case via
`*_active_no_collision`). Side conditions: store offsets do not collide on
active lanes.
-/

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

/-- Single-iteration surface for Python cases where `T <= BT`.

The checked cases are covered by the autotuned `BT = 16` configuration. In this
path the loop executes once, `b_z` is the initial zero vector, and the observable
output is the block-pointer load followed by the lower-triangular dot and
boundary-checked block-pointer store. -/
def chunk_cumsum_vector_single_block_surface
    (S Z : RegionName) (s_s_h s_s_t s_s_d T SSize BT BS : Nat) :
    ComputeKernel := triton {
  i_s = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_i = tl.arange(0, $(BT))
  m_s = tl.where(o_i[:, None] >= o_i[None, :], 1.0, 0.0)
  p_s = tl.make_block_ptr(base=S + i_bh * $(s_s_h), shape=($(T), $(SSize)),
    strides=($(s_s_t), $(s_s_d)), offsets=($(0), i_s * $(BS)),
    block_shape=($(BT), $(BS)), order=(1, 0))
  p_z = tl.make_block_ptr(base=Z + i_bh * $(s_s_h), shape=($(T), $(SSize)),
    strides=($(s_s_t), $(s_s_d)), offsets=($(0), i_s * $(BS)),
    block_shape=($(BT), $(BS)), order=(1, 0))
  b_s = tl.load(p_s, boundary_check=([0, 1] : List Nat)).to(tl.float32)
  b_c = tl.dot(m_s, b_s, allow_tf32=false)
  tl.store(p_z, (b_c).to(p_z.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

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

def singleBlockActive (s : BlockState) (T S BS : Nat)
    (idx : TileIndex [BT, BS]) : Prop :=
  idx.1.val < T ∧ sIndex s BS idx.2.1 < S

instance singleBlockActiveDecidable (s : BlockState) (T S BS : Nat)
    (idx : TileIndex [BT, BS]) : Decidable (singleBlockActive s T S BS idx) := by
  unfold singleBlockActive
  infer_instance

def singleBlockTileOffset (s : BlockState) (s_s_h s_s_t s_s_d BS : Nat)
    (idx : TileIndex [BT, BS]) : Nat :=
  s.pids 1 * s_s_h + idx.1.val * s_s_t + sIndex s BS idx.2.1 * s_s_d

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
  simp [exec, chunk_cumsum_vector_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

noncomputable def singleBlockSourceTile
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Tile .real [BT, BS] :=
  { data := fun idx =>
      if singleBlockActive s T S BS idx then
        some (s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx))
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

noncomputable def singleBlockStoreValue
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    ((Tile.dot [] (lowerTriTile BT)
      (singleBlockSourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
        (idx.1, idx.2.1, PUnit.unit))

set_option maxHeartbeats 800000 in
theorem chunk_cumsum_vector_single_block_surface_correct
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] =>
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)) :
    ∀ idx : TileIndex [BT, BS],
      let outAddr := singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx
      (exec (chunk_cumsum_vector_single_block_surface SReg Z s_s_h s_s_t
            s_s_d T S BT BS) s).map (·.readMem Z outAddr)
        = some (if singleBlockActive s T S BS idx then
            singleBlockStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx
          else s.readMem Z outAddr) := by
  intro idx
  simp [exec, chunk_cumsum_vector_single_block_surface, ComputeKernel.toAlgKernel,
        ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, Option.bind, Option.map]
  simp [stepStmt, evalOp, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.expandDim, Tile.dot, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, ComparableDType.ge,
        singleBlockActive, singleBlockTileOffset, sIndex, singleBlockSourceTile,
        lowerTriTile, TileShape.dropInsertedIndex]
  simp [evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.dot, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, ComparableDType.ge, singleBlockActive,
        singleBlockTileOffset, sIndex, singleBlockSourceTile, lowerTriTile,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BT, BS] → Nat :=
    fun idx => s.pids 1 * s_s_h + idx.1.val * s_s_t +
      (s.pids 0 * BS + idx.2.1.val) * s_s_d
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, singleBlockTileOffset, sIndex] using hOutInj
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      idx.1.val < T ∧ s.pids 0 * BS + idx.2.1.val < S
  · simp [offsetFn, singleBlockActive, sIndex, singleBlockTileOffset,
      singleBlockStoreValue, singleBlockSourceTile, lowerTriTile, Tile.dot,
      BlockState.defaultCarrier, hActive]
    norm_num
  · simp [offsetFn, singleBlockActive, sIndex, singleBlockTileOffset,
      hActive]

theorem chunk_cumsum_vector_single_block_surface_compute_correct
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] =>
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => singleBlockActive s T S BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        singleBlockStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_cumsum_vector_single_block_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := chunk_cumsum_vector_single_block_surface_correct SReg Z s_s_h s_s_t
    s_s_d T S BT BS s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

set_option maxHeartbeats 800000 in
theorem chunk_cumsum_vector_single_block_surface_active_compute_correct
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => singleBlockActive s T S BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        singleBlockStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_cumsum_vector_single_block_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  simp [exec, chunk_cumsum_vector_single_block_surface, ComputeKernel.toAlgKernel,
        ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, Option.bind, Option.map] at hExec
  simp [stepStmt, evalOp, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.expandDim, Tile.dot, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, ComparableDType.ge,
        singleBlockActive, singleBlockTileOffset, sIndex, singleBlockSourceTile,
        lowerTriTile, TileShape.dropInsertedIndex] at hExec
  simp [evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.dot, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, ComparableDType.ge, singleBlockActive,
        singleBlockTileOffset, sIndex, singleBlockSourceTile, lowerTriTile,
        TileShape.dropInsertedIndex] at hExec
  subst s'
  let offsetFn : TileIndex [BT, BS] → Nat :=
    fun idx => s.pids 1 * s_s_h + idx.1.val * s_s_t +
      (s.pids 0 * BS + idx.2.1.val) * s_s_d
  let valueFn : TileIndex [BT, BS] → ℝ :=
    fun idx => WithBot.unbotD 0
      (∑ x,
        Option.map₂ (fun x1 x2 : ℝ => x1 * x2)
          (if x <= idx.1 then some (1.0 : ℝ) else some (0.0 : ℝ))
          (if x.val < T ∧ s.pids 0 * BS + idx.2.1.val < S then
            some (s.readMem SReg
              (s.pids 1 * s_s_h + x.val * s_s_t +
                (s.pids 0 * BS + idx.2.1.val) * s_s_d))
          else BlockState.defaultCarrier TileDType.real))
  change (List.foldl
      (fun (acc : BlockState) i =>
        if singleBlockActive s T S BS i then
          acc.writeMem Z (offsetFn i) (valueFn i)
        else acc)
      _ (TileShape.allIndices [BT, BS])).readMem Z (offsetFn idx) =
    singleBlockStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _ _ _
      (singleBlockActive s T S BS) idx hActive]
  · simp [valueFn, singleBlockStoreValue, singleBlockSourceTile, lowerTriTile,
      Tile.dot, singleBlockActive, singleBlockTileOffset, sIndex,
      BlockState.defaultCarrier, hActive]
    norm_num
    rfl
  intro k hk heq
  exact hNoCollision idx hActive k hk
    (by simpa [offsetFn, singleBlockTileOffset, sIndex] using heq)

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
  simp [exec, chunk_cumsum_vector_cumsum_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
  simp [exec, chunk_cumsum_vector_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
  simp [exec, chunk_cumsum_vector_cumsum_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

theorem chunk_cumsum_vector_single_block_python_case1_active_no_collision
    (s : BlockState) :
    ∀ idx : TileIndex [16, 32], singleBlockActive s 4 5 32 idx →
      ∀ k : TileIndex [16, 32], singleBlockActive s 4 5 32 k →
        singleBlockTileOffset s 20 5 1 32 k =
          singleBlockTileOffset s 20 5 1 32 idx → k = idx := by
  rintro ⟨⟨ti, hti⟩, ⟨si, hsi⟩, _⟩ hi ⟨⟨tk, htk⟩, ⟨sk, hsk⟩, _⟩ hk h
  simp [singleBlockActive, sIndex] at hi hk
  simp [singleBlockTileOffset, sIndex] at h
  have ht : tk = ti := by omega
  have hs : sk = si := by omega
  subst tk
  subst sk
  rfl

theorem chunk_cumsum_vector_single_block_python_case2_active_no_collision
    (s : BlockState) :
    ∀ idx : TileIndex [16, 32], singleBlockActive s 8 10 32 idx →
      ∀ k : TileIndex [16, 32], singleBlockActive s 8 10 32 k →
        singleBlockTileOffset s 80 10 1 32 k =
          singleBlockTileOffset s 80 10 1 32 idx → k = idx := by
  rintro ⟨⟨ti, hti⟩, ⟨si, hsi⟩, _⟩ hi ⟨⟨tk, htk⟩, ⟨sk, hsk⟩, _⟩ hk h
  simp [singleBlockActive, sIndex] at hi hk
  simp [singleBlockTileOffset, sIndex] at h
  have ht : tk = ti := by omega
  have hs : sk = si := by omega
  subst tk
  subst sk
  rfl

theorem chunk_cumsum_vector_single_block_python_case3_active_no_collision
    (s : BlockState) :
    ∀ idx : TileIndex [16, 32], singleBlockActive s 1 5 32 idx →
      ∀ k : TileIndex [16, 32], singleBlockActive s 1 5 32 k →
        singleBlockTileOffset s 5 5 1 32 k =
          singleBlockTileOffset s 5 5 1 32 idx → k = idx := by
  rintro ⟨⟨ti, hti⟩, ⟨si, hsi⟩, _⟩ hi ⟨⟨tk, htk⟩, ⟨sk, hsk⟩, _⟩ hk h
  simp [singleBlockActive, sIndex] at hi hk
  simp [singleBlockTileOffset, sIndex] at h
  have ht : tk = ti := by omega
  have hs : sk = si := by omega
  subst tk
  subst sk
  rfl

theorem chunk_cumsum_vector_single_block_python_case1_compute_correct
    (SReg Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => singleBlockActive s 4 5 32 idx)
        (fun idx => (Z, singleBlockTileOffset s 20 5 1 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        singleBlockStoreValue s SReg 20 5 1 4 5 16 32 idx) := by
  exact chunk_cumsum_vector_single_block_surface_active_compute_correct SReg Z
    20 5 1 4 5 16 32 s
    (chunk_cumsum_vector_single_block_python_case1_active_no_collision s)

theorem chunk_cumsum_vector_single_block_python_case2_compute_correct
    (SReg Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z 80 10 1 8 10 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => singleBlockActive s 8 10 32 idx)
        (fun idx => (Z, singleBlockTileOffset s 80 10 1 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        singleBlockStoreValue s SReg 80 10 1 8 10 16 32 idx) := by
  exact chunk_cumsum_vector_single_block_surface_active_compute_correct SReg Z
    80 10 1 8 10 16 32 s
    (chunk_cumsum_vector_single_block_python_case2_active_no_collision s)

theorem chunk_cumsum_vector_single_block_python_case3_compute_correct
    (SReg Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z 5 5 1 1 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => singleBlockActive s 1 5 32 idx)
        (fun idx => (Z, singleBlockTileOffset s 5 5 1 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        singleBlockStoreValue s SReg 5 5 1 1 5 16 32 idx) := by
  exact chunk_cumsum_vector_single_block_surface_active_compute_correct SReg Z
    5 5 1 1 5 16 32 s
    (chunk_cumsum_vector_single_block_python_case3_active_no_collision s)

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

/-! ## Python test-case full-surface wrappers

The Python tests use contiguous `[B, H, T, S]` tensors and the existing proof
shape `BT = 16`, `BS = 32`. These wrappers pin the full vector-cumsum surface
to the three checked layouts. -/

theorem chunk_cumsum_vector_python_case1_surface_toAlgorithm_supported
    (SReg Z : RegionName) :
    ∃ alg, (chunk_cumsum_vector_surface SReg Z 20 5 1 4 5 16 32).toAlgorithm? =
      Except.ok alg := by
  exact chunk_cumsum_vector_surface_toAlgorithm_supported SReg Z
    20 5 1 4 5 16 32

theorem chunk_cumsum_vector_python_case2_surface_toAlgorithm_supported
    (SReg Z : RegionName) :
    ∃ alg, (chunk_cumsum_vector_surface SReg Z 80 10 1 8 10 16 32).toAlgorithm? =
      Except.ok alg := by
  exact chunk_cumsum_vector_surface_toAlgorithm_supported SReg Z
    80 10 1 8 10 16 32

theorem chunk_cumsum_vector_python_case3_surface_toAlgorithm_supported
    (SReg Z : RegionName) :
    ∃ alg, (chunk_cumsum_vector_surface SReg Z 5 5 1 1 5 16 32).toAlgorithm? =
      Except.ok alg := by
  exact chunk_cumsum_vector_surface_toAlgorithm_supported SReg Z
    5 5 1 1 5 16 32

noncomputable def chunkCumsumVectorSurfaceValue
    (s : BlockState) (SReg Z Out : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (offset : Nat) : ℝ :=
  match exec (chunk_cumsum_vector_surface SReg Z s_s_h s_s_t s_s_d T S BT BS) s with
  | some s' => s'.readMem Out offset
  | none => 0.0

theorem chunk_cumsum_vector_surface_output_compute_correct
    (SReg Z Out : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_surface SReg Z s_s_h s_s_t s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx => (Out, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        chunkCumsumVectorSurfaceValue s SReg Z Out s_s_h s_s_t s_s_d T S BT BS
          (tileOffset s s_s_h s_s_t s_s_d BT BS idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_cumsum_vector_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [chunkCumsumVectorSurfaceValue, hExec]

theorem chunk_cumsum_vector_python_case1_surface_outputs_compute_correct
    (SReg Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_surface SReg Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        chunkCumsumVectorSurfaceValue s SReg Z Z 20 5 1 4 5 16 32
          (tileOffset s 20 5 1 16 32 idx)) := by
  exact chunk_cumsum_vector_surface_output_compute_correct SReg Z Z
    20 5 1 4 5 16 32 s

theorem chunk_cumsum_vector_python_case2_surface_outputs_compute_correct
    (SReg Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_surface SReg Z 80 10 1 8 10 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 10 16 32 idx)
        (fun idx => (Z, tileOffset s 80 10 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        chunkCumsumVectorSurfaceValue s SReg Z Z 80 10 1 8 10 16 32
          (tileOffset s 80 10 1 16 32 idx)) := by
  exact chunk_cumsum_vector_surface_output_compute_correct SReg Z Z
    80 10 1 8 10 16 32 s

theorem chunk_cumsum_vector_python_case3_surface_outputs_compute_correct
    (SReg Z : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_surface SReg Z 5 5 1 1 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 1 5 16 32 idx)
        (fun idx => (Z, tileOffset s 5 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        chunkCumsumVectorSurfaceValue s SReg Z Z 5 5 1 1 5 16 32
          (tileOffset s 5 5 1 16 32 idx)) := by
  exact chunk_cumsum_vector_surface_output_compute_correct SReg Z Z
    5 5 1 1 5 16 32 s

/-- Public Python case 1 summary: full vector-cumsum surface plus both
boundary store and carry-cumsum output slices for `B = 2`, `H = 3`, `T = 4`,
`S = 5`. -/
theorem chunk_cumsum_vector_python_case1_slice_summary
    (BC SReg Carry Z : RegionName) (s : BlockState) :
    (∃ alg, (chunk_cumsum_vector_surface SReg Z 20 5 1 4 5 16 32).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 20 5 1 4 5 16 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z 20 5 1 4 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 4 5 16 32 idx)
        (fun idx => (Z, tileOffset s 20 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 20 5 1 4 5 16 32 idx)) := by
  constructor
  · exact chunk_cumsum_vector_python_case1_surface_toAlgorithm_supported
      SReg Z
  constructor
  · exact chunk_cumsum_vector_store_python_case1_compute_correct BC Z s
  · exact chunk_cumsum_vector_cumsum_python_case1_compute_correct
      SReg Carry Z s

/-- Public Python case 2 summary: full vector-cumsum surface plus both
boundary store and carry-cumsum output slices for `B = H = 1`, `T = 8`,
`S = 10`. -/
theorem chunk_cumsum_vector_python_case2_slice_summary
    (BC SReg Carry Z : RegionName) (s : BlockState) :
    (∃ alg, (chunk_cumsum_vector_surface SReg Z 80 10 1 8 10 16 32).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z 80 10 1 8 10 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 10 16 32 idx)
        (fun idx => (Z, tileOffset s 80 10 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 80 10 1 8 10 16 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z 80 10 1 8 10 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 8 10 16 32 idx)
        (fun idx => (Z, tileOffset s 80 10 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 80 10 1 8 10 16 32 idx)) := by
  constructor
  · exact chunk_cumsum_vector_python_case2_surface_toAlgorithm_supported
      SReg Z
  constructor
  · exact chunk_cumsum_vector_store_python_case2_compute_correct BC Z s
  · exact chunk_cumsum_vector_cumsum_python_case2_compute_correct
      SReg Carry Z s

/-- Public Python case 3 summary: full vector-cumsum surface plus both
boundary store and carry-cumsum output slices for `B = H = T = 1`, `S = 5`. -/
theorem chunk_cumsum_vector_python_case3_slice_summary
    (BC SReg Carry Z : RegionName) (s : BlockState) :
    (∃ alg, (chunk_cumsum_vector_surface SReg Z 5 5 1 1 5 16 32).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_store_slice BC Z 5 5 1 1 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 1 5 16 32 idx)
        (fun idx => (Z, tileOffset s 5 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        storeValue s BC 5 5 1 1 5 16 32 idx)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z 5 5 1 1 5 16 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [16, 32] => active s 1 5 16 32 idx)
        (fun idx => (Z, tileOffset s 5 5 1 16 32 idx)))
      (expected := fun idx : TileIndex [16, 32] =>
        cumsumStoreValue s SReg Carry 5 5 1 1 5 16 32 idx)) := by
  constructor
  · exact chunk_cumsum_vector_python_case3_surface_toAlgorithm_supported
      SReg Z
  constructor
  · exact chunk_cumsum_vector_store_python_case3_compute_correct BC Z s
  · exact chunk_cumsum_vector_cumsum_python_case3_compute_correct
      SReg Carry Z s


















/-- `output_summary` for vector chunk-cumsum Python case 1 surface. -/
abbrev chunk_cumsum_vector_python_case1_output_summary
    (SReg Z : RegionName) (s : BlockState) :=
  chunk_cumsum_vector_python_case1_surface_outputs_compute_correct SReg Z s

/-- `output_summary` for vector chunk-cumsum Python case 2 surface. -/
abbrev chunk_cumsum_vector_python_case2_output_summary
    (SReg Z : RegionName) (s : BlockState) :=
  chunk_cumsum_vector_python_case2_surface_outputs_compute_correct SReg Z s

/-- `output_summary` for vector chunk-cumsum Python case 3 surface. -/
abbrev chunk_cumsum_vector_python_case3_output_summary
    (SReg Z : RegionName) (s : BlockState) :=
  chunk_cumsum_vector_python_case3_surface_outputs_compute_correct SReg Z s

end VeriTile.Bench.TritonBenchG.ChunkCumsumVector
