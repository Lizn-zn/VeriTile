import VeriTile.Triton

/-!
# `reversed_cumsum` — strict per-kernel correctness

`chunk_global_reversed_cumsum_vector_kernel` computes a global reversed
cumulative sum over the time axis of a `[B, H, T, S]` tensor. For each
`(i_s, i_bh)` program it traverses the `T` axis in reverse `BT`-sized chunks,
carrying an across-chunk accumulator `b_z`; per chunk it loads `b_s`, forms
`b_z[None, :] + tl.dot(m_s, b_s)` with the upper-triangular mask `m_s` (the
within-chunk reversed prefix sum), stores the result, then updates `b_z`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`kernel[grid]` with `grid = (cdiv(S, BS), B·H)`, the
`BS = 32` choice, the per-head strides, and cross-program composition of `z`) is
the *trusted boundary*, not a proof obligation here. Program ids enter only via
`BlockState`, so the per-program statement covers every program of the grid.

## Proof architecture

```
reversed_cumsum_output_summary_general  ← TOP THEOREM (dimension-general, single-chunk)
  ├─ (toAlgorithm? = Except.ok _) via
  │    reversed_cumsum_surface_toAlgorithm_supported  (full reverse-loop surface lowers)
  └─ GENUINE reversed-cumsum closed form: each active output lane (i, j) holds
       reversedCumsumClosed = Σ_{k ≥ i, k < T} x[k, j]   (NOT a kernel read-back)
         ├─ single chunk (T ≤ BT): the single-block surface realizes it
         │    via reversed_cumsum_single_block_surface_active_closed_form
         └─ multi-chunk: per-chunk cumsumStoreValue realizes (carried scan)
              + singleBlockStoreValue_eq_reversedCumsumClosed (dot = reversed cumsum)

mathematical core:
  singleBlockStoreValue_eq_reversedCumsumClosed   (`tl.dot(m_s, b_s)` = Σ_{k ≥ i} x[k])

per-slice value-level correctness (the computational content):
  reversed_cumsum_single_block_surface_compute_correct → ..._correct   (single-BT chunk, b_z = 0)
  reversed_cumsum_cumsum_slice_compute_correct         → ..._correct   (one chunk with carry)
  reversed_cumsum_store_slice_compute_correct          → ..._correct   (boundary-checked writeback)
       └─ cumsumStoreValue (`Tile.dot` with upperTriTile + carryValue)
          / storeValue (plain boundary-checked echo)
  + active-masked variants (reversed_cumsum_*_active_compute_correct).
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled (the kernel runs at fixed `BT`/`BS`). The `tl.dot(m_s, b_s,
allow_tf32=False)` reversed prefix sum is modeled by `Tile.dot` with
`upperTriTile`; the `.to(tl.float32)` / `.to(p_z.dtype.element_ty)` casts reduce
to the identity post-erasure. `boundary_check` / masked loads use `other=0.0`.
The genuine reversed-cumsum closed form (`reversedCumsumClosed`,
`Σ_{k ≥ i, k < T} x[k, j]`) is realized end-to-end by the single-chunk surface
(`reversed_cumsum_single_block_surface_*_closed_form`); the multi-chunk surface
(`T > BT`) is verified to lower, with the carried scan proven per chunk
(`cumsumStoreValue`) and the within-chunk matrix product identified with the
reversed cumsum (`singleBlockStoreValue_eq_reversedCumsumClosed`). The store
scatter requires only active-lane collision freedom (taken as a hypothesis).
-/

namespace VeriTile.Bench.TritonBenchG.ReversedCumsum

open VeriTile.Triton
open scoped VeriTile.Triton.Masked3DTileKernelIO₁

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
  simp [exec, reversed_cumsum_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
    ComputeCorrect.Realizes_without_Rounding
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
      ((Tile.dot [] (upperTriTile BT)
        (sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
          (idx.1, idx.2.1, PUnit.unit)))

noncomputable def singleBlockStoreValue
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  WithBot.unbotD 0
    ((Tile.dot [] (upperTriTile BT)
      (singleBlockSourceTile s SReg s_s_h s_s_t s_s_d T S BT BS)).data
        (idx.1, idx.2.1, PUnit.unit))

/-! ## Genuine reversed-cumsum closed form

The Triton kernel computes, for output row `i` and feature column `j`, the
*reversed* cumulative sum along the time axis: the sum of the source value at
every row `k ≥ i` (that is still inside the tensor, `k < T`) in the same
feature column. The matrix product `tl.dot(m_s, b_s)` with the upper-triangular
mask `m_s[i,k] = [i ≤ k]` realizes exactly this directional scan.

`reversedCumsumClosed` is the genuine mathematical specification — a `Finset.sum`
over `{k : k ≥ i ∧ k < T}` of the loaded source value. It is *not* a read-back
of the kernel's own output, so realizing it is a true correctness statement. -/
noncomputable def reversedCumsumClosed
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  ∑ k : Fin BT,
    if idx.1.val ≤ k.val ∧ k.val < T ∧ sIndex s BS idx.2.1 < S then
      s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
        ((k, idx.2.1, PUnit.unit) : TileIndex [BT, BS]))
    else 0

/-- The matrix-product store value (`Tile.dot` with the upper-triangular mask)
equals the genuine reversed cumulative sum `Σ_{k ≥ i, k < T} x[k, j]`. This is
the mathematical heart: it certifies that the kernel's `tl.dot(m_s, b_s)` is in
fact a reversed directional scan. -/
theorem singleBlockStoreValue_eq_reversedCumsumClosed
    (s : BlockState) (SReg : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) :
    singleBlockStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx
      = reversedCumsumClosed s SReg s_s_h s_s_t s_s_d T S BT BS idx := by
  unfold singleBlockStoreValue reversedCumsumClosed
  rw [Tile.dot_nil_data]
  have hmap : ∀ k : Fin BT,
      Option.map₂ (· * ·)
        ((upperTriTile BT).data (idx.1, k, PUnit.unit))
        ((singleBlockSourceTile s SReg s_s_h s_s_t s_s_d T S BT BS).data
          (k, idx.2.1, PUnit.unit))
      = (some (if idx.1.val ≤ k.val ∧ k.val < T ∧ sIndex s BS idx.2.1 < S then
            s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
              ((k, idx.2.1, PUnit.unit) : TileIndex [BT, BS]))
          else 0) : WithBot ℝ) := by
    intro k
    simp only [upperTriTile, singleBlockSourceTile, singleBlockActive]
    by_cases hik : idx.1.val ≤ k.val
    · by_cases hact : k.val < T ∧ sIndex s BS idx.2.1 < S
      · rw [if_pos hik, if_pos hact,
          if_pos (And.intro hik (And.intro hact.1 hact.2)),
          Option.map₂_some_some]
        rw [show (1.0 : ℝ) *
            s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
              (k, idx.2.1, PUnit.unit)) =
          s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
            (k, idx.2.1, PUnit.unit)) from by norm_num]
      · rw [if_pos hik, if_neg hact,
          if_neg (fun h : idx.1.val ≤ k.val ∧
            k.val < T ∧ sIndex s BS idx.2.1 < S => hact h.2),
          Option.map₂_some_some]
        rw [show (1.0 : ℝ) * (0.0 : ℝ) = 0 from by norm_num]
    · rw [if_neg hik,
        if_neg (fun h : idx.1.val ≤ k.val ∧
          k.val < T ∧ sIndex s BS idx.2.1 < S => hik h.1)]
      by_cases hact : k.val < T ∧ sIndex s BS idx.2.1 < S
      · rw [if_pos hact, Option.map₂_some_some]
        rw [show (0.0 : ℝ) *
            s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
              (k, idx.2.1, PUnit.unit)) = 0 from by norm_num]
      · rw [if_neg hact, Option.map₂_some_some]
        rw [show (0.0 : ℝ) * (0.0 : ℝ) = 0 from by norm_num]
  simp only [hmap]
  rw [WithBot.unbotD_sum_some Finset.univ (fun k : Fin BT =>
    if idx.1.val ≤ k.val ∧ k.val < T ∧ sIndex s BS idx.2.1 < S then
      s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
        ((k, idx.2.1, PUnit.unit) : TileIndex [BT, BS]))
    else 0)]

set_option maxHeartbeats 800000 in
theorem reversed_cumsum_single_block_surface_correct
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] =>
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)) :
    ∀ idx : TileIndex [BT, BS],
      let outAddr := singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx
      (exec (reversed_cumsum_single_block_surface SReg Z s_s_h s_s_t
            s_s_d T S BT BS) s).map (·.readMem Z outAddr)
        = some (if singleBlockActive s T S BS idx then
            singleBlockStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx
          else s.readMem Z outAddr) := by
  intro idx
  simp [exec, reversed_cumsum_single_block_surface, ComputeKernel.toAlgKernel,
        ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, Option.bind, Option.map]
  simp [stepStmt, evalOp, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.expandDim, Tile.dot, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, ComparableDType.le,
        singleBlockActive, singleBlockTileOffset, sIndex, singleBlockSourceTile,
        upperTriTile, TileShape.dropInsertedIndex]
  simp [evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.dot, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, ComparableDType.le, singleBlockActive,
        singleBlockTileOffset, sIndex, singleBlockSourceTile, upperTriTile,
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
      singleBlockStoreValue, singleBlockSourceTile, upperTriTile, Tile.dot,
      BlockState.defaultCarrier, hActive]
    norm_num
  · simp [offsetFn, singleBlockActive, sIndex, singleBlockTileOffset,
      hActive]

theorem reversed_cumsum_single_block_surface_compute_correct
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] =>
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := reversed_cumsum_single_block_surface SReg Z s_s_h s_s_t
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
  · simp [reversed_cumsum_single_block_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := reversed_cumsum_single_block_surface_correct SReg Z s_s_h s_s_t
    s_s_d T S BT BS s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

set_option maxHeartbeats 800000 in
theorem reversed_cumsum_single_block_surface_active_closed_form
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := reversed_cumsum_single_block_surface SReg Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => singleBlockActive s T S BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        reversedCumsumClosed s SReg s_s_h s_s_t s_s_d T S BT BS idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [reversed_cumsum_single_block_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  simp [exec, reversed_cumsum_single_block_surface, ComputeKernel.toAlgKernel,
        ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        stepStmts, Option.bind, Option.map] at hExec
  simp [stepStmt, evalOp, Option.bind, Option.map, Tile.bop,
        Tile.cop, Tile.expandDim, Tile.dot, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, ComparableDType.le,
        singleBlockActive, singleBlockTileOffset, sIndex, singleBlockSourceTile,
        upperTriTile, TileShape.dropInsertedIndex] at hExec
  simp [evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.dot, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, ComparableDType.le, singleBlockActive,
        singleBlockTileOffset, sIndex, singleBlockSourceTile, upperTriTile,
        TileShape.dropInsertedIndex] at hExec
  subst s'
  simp only [ComputeCorrect.OutputReadable.read_real, singleBlockTileOffset,
    sIndex]
  -- Abstract the executed value-function and the (register-only) init state.
  rw [show (s.pids 1 * s_s_h + idx.1.val * s_s_t +
        (s.pids 0 * BS + idx.2.1.val) * s_s_d)
      = (fun i : TileIndex [BT, BS] => s.pids 1 * s_s_h + i.1.val * s_s_t +
          (s.pids 0 * BS + i.2.1.val) * s_s_d) idx from rfl]
  rw [BlockState.scatter_readback_prop_masked_nd_of_true _
    (fun i : TileIndex [BT, BS] => s.pids 1 * s_s_h + i.1.val * s_s_t +
      (s.pids 0 * BS + i.2.1.val) * s_s_d)
    _ (fun i : TileIndex [BT, BS] =>
        i.1.val < T ∧ s.pids 0 * BS + i.2.1.val < S) idx
    (by simpa [singleBlockActive, sIndex] using hActive)
    (fun k hk heq =>
      hNoCollision idx hActive k
        (by simpa [singleBlockActive, sIndex] using hk)
        (by simpa [singleBlockTileOffset, sIndex] using heq))]
  have hbridge := singleBlockStoreValue_eq_reversedCumsumClosed s SReg
    s_s_h s_s_t s_s_d T S BT BS idx
  simp only [singleBlockStoreValue, singleBlockSourceTile,
    singleBlockActive, upperTriTile, sIndex, singleBlockTileOffset] at hbridge
  rw [← hbridge, Tile.dot_nil_data]
  congr 1
  apply Finset.sum_congr rfl
  intro x _
  have h00 : BlockState.defaultCarrier TileDType.real = (some (0.0 : ℝ)) := by
    show (some (0 : ℝ)) = some (0.0 : ℝ); norm_num
  rw [h00]
  rfl

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
  simp [exec, reversed_cumsum_cumsum_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
  simp [exec, reversed_cumsum_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
    ComputeCorrect.Realizes_without_Rounding
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
  simp [exec, reversed_cumsum_cumsum_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

/-- **Dimension-general single-chunk output summary.** Holds at fully symbolic
dimensions (arbitrary `s_s_h s_s_t s_s_d T S BT BS`).
For any single-chunk shape (`T ≤ BT`, so the reverse loop runs once with carry
`b_z = 0`), the full reverse-traversal surface lowers to the algorithm layer, and
the faithful single-chunk surface writes into every active lane `(i, j)` the
genuine reversed cumulative sum `Σ_{k ≥ i, k < T} x[k, j]` (`reversedCumsumClosed`,
a standalone `Finset.sum`, not a read-back of the kernel output). The active-lane
collision-freedom of the block address map is the explicit hypothesis. -/
specification reversed_cumsum_output_summary_general
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx) :
    (∃ alg, (reversed_cumsum_surface SReg Z s_s_h s_s_t s_s_d T S BT BS).toAlgorithm? =
      Except.ok alg) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := reversed_cumsum_single_block_surface SReg Z s_s_h s_s_t s_s_d
        T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => singleBlockActive s T S BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        reversedCumsumClosed s SReg s_s_h s_s_t s_s_d T S BT BS idx)) := by
  refine ⟨reversed_cumsum_surface_toAlgorithm_supported SReg Z
    s_s_h s_s_t s_s_d T S BT BS, ?_⟩
  exact reversed_cumsum_single_block_surface_active_closed_form SReg Z
    s_s_h s_s_t s_s_d T S BT BS s hNoCollision

/-! ## ════════ `⊨` IO face for the masked block store ════════

The summary above is stated per *declared write map*. This section restates the
`BC → Z` block store on the audit-once IO surface
`Masked3DTileKernelIO₁.Implements` (`⊨`), which additionally pins the **flat memory**
placement.

The slice owns a `[BT, BS]` block whose address is built from **all three** program
axes (`i_s`, `i_bh`, `i_t`) — the shape the three-axis tile skin exists for. Read and
write share the one address, and both are gated by the kernel's own
`offs_t < T ∧ offs_s < S` guard. -/

section IOFace

/-- Cell-level frame of a masked scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, P k → offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k =>
            if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, P k → offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl]
      by_cases hP : P hd
      · rw [if_pos hP, BlockState.writeMem_mem, if_neg ?_]
        rintro ⟨h1, h2⟩
        rcases hc with h | h
        · exact h h1
        · exact h hd List.mem_cons_self hP h2.symm
      · rw [if_neg hP]

theorem block_store_flattenOk (BC Z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    ((reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT
      BS).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [reversed_cumsum_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  refine ⟨?_, ?_⟩ <;> simp [Op.FlattenOk.eq_def]

theorem block_store_terminates (BC Z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (s : BlockState) :
    ∃ s1, exec (reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS)
      s = some s1 := by
  simp [exec, reversed_cumsum_store_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt]

theorem block_store_frame (BC Z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (s s' : BlockState)
    (hExec : exec (reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT
      BS) s = some s') :
    ∀ (r : RegionName) (o : Nat),
      (r ≠ Z ∨ ∀ idx : TileIndex [BT, BS], active s T S BT BS idx →
        o ≠ tileOffset s s_s_h s_s_t s_s_d BT BS idx) →
      s'.mem r o = s.mem r o := by
  intro r o hcond
  simp [exec, reversed_cumsum_store_slice, stepStmts, stepStmt,
    evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    TileShape.dropInsertedIndex] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := Z)
    (fun idx : TileIndex [BT, BS] =>
      s.pids 1 * s_s_h + (s.pids 2 * BT + idx.1.val) * s_s_t
        + (s.pids 0 * BS + idx.2.1.val) * s_s_d)
    _ (fun idx : TileIndex [BT, BS] =>
      s.pids 2 * BT + idx.1.val < T ∧ s.pids 0 * BS + idx.2.1.val < S) r o
    (TileShape.allIndices [BT, BS]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun idx _ hidx => Ne.symm (h idx hidx)

theorem block_store_traceSafe (BC Z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BT, BS], active s T S BT BS idx →
      tileOffset s s_s_h s_s_t s_s_d BT BS idx < bounds BC)
    (hout : ∀ idx : TileIndex [BT, BS], active s T S BT BS idx →
      tileOffset s s_s_h s_s_t s_s_d BT BS idx < bounds Z) :
    ((reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT
      BS).toAlgKernel).TraceSafe bounds s := by
  simp [Kernel.TraceSafe, reversed_cumsum_store_slice, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, NumericDType.add,
    NumericDType.mul, ComparableDType.lt]
  -- the surviving goals are the four `expandDim` nodes (simp cannot peel them)
  -- and the two window bounds; `and_intros` flattens the nesting so the shapes
  -- can be dispatched without guessing it
  and_intros
  all_goals first
    | simp [Op.SafeAt.eq_def]
    | exact fun a b ha hb => hin (a, b, PUnit.unit) ⟨ha, hb⟩
    | exact fun a b ha hb => hout (a, b, PUnit.unit) ⟨ha, hb⟩

/-- Region-model run of the masked block store. -/
theorem block_store_region_run (BC Z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BT, BS] => tileOffset s₀ s_s_h s_s_t s_s_d BT BS idx))
    (xs : TileIndex [BT, BS] → ℝ)
    (hx : ∀ idx : TileIndex [BT, BS], active s₀ T S BT BS idx →
      s₀.readMem BC (tileOffset s₀ s_s_h s_s_t s_s_d BT BS idx) = xs idx) :
    ∃ s1, exec (reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS)
        s₀ = some s1
      ∧ (∀ idx : TileIndex [BT, BS], active s₀ T S BT BS idx →
          s1.readMem Z (tileOffset s₀ s_s_h s_s_t s_s_d BT BS idx) = xs idx)
      ∧ (∀ (r : RegionName) (o : Nat),
          (r ≠ Z ∨ ∀ idx : TileIndex [BT, BS], active s₀ T S BT BS idx →
            o ≠ tileOffset s₀ s_s_h s_s_t s_s_d BT BS idx) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hexec⟩ := block_store_terminates BC Z s_s_h s_s_t s_s_d T S BT BS s₀
  refine ⟨s1, hexec, ?_, block_store_frame BC Z s_s_h s_s_t s_s_d T S BT BS s₀ s1
    hexec⟩
  intro idx hact
  have h := reversed_cumsum_store_slice_correct BC Z s_s_h s_s_t s_s_d T S BT
    BS s₀ hOutInj idx
  have h' : s1.readMem Z (tileOffset s₀ s_s_h s_s_t s_s_d BT BS idx)
      = if active s₀ T S BT BS idx then
          storeValue s₀ BC s_s_h s_s_t s_s_d T S BT BS idx
        else s₀.readMem Z (tileOffset s₀ s_s_h s_s_t s_s_d BT BS idx) := by
    simpa [hexec] using h
  rw [h', if_pos hact, storeValue, if_pos hact]
  simpa using hx idx hact

/-- IO signature of the masked block store on the three-axis tile surface: the
`[BT, BS]` block's one address is built from all three program axes, and the same
address serves the read and the write. -/
def blockStoreIO (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) :
    Masked3DTileKernelIO₁ where
  kernel := reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT BS
  inp := BC
  out := Z
  shape := [BT, BS]
  read := fun p₀ p₁ p₂ idx =>
    p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t + (p₀ * BS + idx.2.1.val) * s_s_d
  write := fun p₀ p₁ p₂ idx =>
    p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t + (p₀ * BS + idx.2.1.val) * s_s_d
  mask := fun p₀ _p₁ p₂ idx =>
    p₂ * BT + idx.1.val < T ∧ p₀ * BS + idx.2.1.val < S

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for `reversed_cumsum.py`'s masked block
store: for every disjoint flat placement of `BC` / `Z`, every program coordinate
whose active lanes are in bounds, and every launch state whose `BC` block holds `xs`
at the active lanes, the translated pointer kernel terminates, every active lane of
the `Z` block holds `xs idx`, and every other memory cell is unchanged.

The block address is built from **all three** program axes (`i_s`, `i_bh`, `i_t`) —
the shape the three-axis tile skin exists for. Dimension-general in the three
strides, `T`, `S`, `BT`, `BS`. Honest side-condition: output-address injectivity at
every program coordinate, the same hypothesis the per-write-map summary takes. -/
specification reversed_cumsum_block_store_io_correctness (BC Z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BT, BS] =>
        p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t
          + (p₀ * BS + idx.2.1.val) * s_s_d)) :
    blockStoreIO BC Z s_s_h s_s_t s_s_d T S BT BS
      ⊨ fun _p₀ _p₁ xs idx => xs idx := by
  refine Masked3DTileKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact block_store_flattenOk BC Z s_s_h s_s_t s_s_d T S BT BS
  · intro bounds s h1 h2
    exact block_store_traceSafe BC Z s_s_h s_s_t s_s_d T S BT BS bounds s
      (fun idx hact => h1 idx hact) (fun idx hact => h2 idx hact)
  · intro s₀ xs hin
    exact block_store_region_run BC Z s_s_h s_s_t s_s_d T S BT BS s₀
      (hOutInj (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) xs
      (fun idx hact => hin idx hact)

/-! ### The rounding face

The store is a pure copy: no arithmetic on the loaded block, and the store's
`.to(Z.dtype.element_ty)` erases to `.real`. So the slice is **cast-free** — every
statement steps identically under `stepStmtsR R` and `stepStmts` — and the exact
run transports to `execR R` for *every* rounding model. -/

/-- The slice is cast-free: `execR R` is the exact stepper. -/
private theorem block_store_castFree (R : RoundingModel) (BC Z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (s : BlockState) :
    execR R ((reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT
        BS).toAlgKernel) s
      = exec ((reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT
        BS).toAlgKernel) s := by
  simp [execR, exec, reversed_cumsum_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, stepStmtsR, stepStmts,
    stepStmtR, stepStmt, evalOpR, evalOpR.eq_def, evalOp, evalOp.eq_def,
    BlockState.writeMemTypedR, BlockState.writeMemAsR, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd, Tile.uop,
    NumericDType.add, NumericDType.mul, ComparableDType.lt, FloatDType.cast,
    TileShape.dropInsertedIndex]

/-- Per-execution safety walk **under the rounding model** — the `hts`
obligation of `Masked3DTileKernelIO₁.ImplementsR.intro`. -/
theorem block_store_traceSafeR (R : RoundingModel) (BC Z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat) (bounds : RegionBounds) (s : BlockState)
    (hin : ∀ idx : TileIndex [BT, BS], active s T S BT BS idx →
      tileOffset s s_s_h s_s_t s_s_d BT BS idx < bounds BC)
    (hout : ∀ idx : TileIndex [BT, BS], active s T S BT BS idx →
      tileOffset s s_s_h s_s_t s_s_d BT BS idx < bounds Z) :
    Kernel.TraceSafeR R bounds
      ((reversed_cumsum_store_slice BC Z s_s_h s_s_t s_s_d T S BT
        BS).toAlgKernel) s := by
  simp only [active, tIndex, sIndex, tileOffset] at hin hout
  unfold Kernel.TraceSafeR
  simp [reversed_cumsum_store_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeListR,
    Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.SafeAtR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR, stepStmtR, evalOpR, evalOpR.eq_def,
    Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.ptrAdd,
    Tile.uop, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    FloatDType.cast, TileShape.dropInsertedIndex]
  and_intros
  all_goals first
    | simp [Op.SafeAtR.eq_def]
    | exact fun a b ha hb => hin (a, b, PUnit.unit) ⟨ha, hb⟩
    | exact fun a b ha hb => hout (a, b, PUnit.unit) ⟨ha, hb⟩

/-! ### ════════ ★ MAIN THEOREM (rounding face) ★ ════════ -/

/-- **The `⊨[R]` headline** for `reversed_cumsum.py`'s masked block store: for
**every** rounding model `R`, the same masked Hoare triple as
`reversed_cumsum_block_store_io_correctness`, but run under `execR R` and read
back as `.real`-typed cells holding `R.round .real (xs idx)`.

The store is a pure copy — no arithmetic, and the `.to(Z.dtype.element_ty)`
erases to `.real` — so the slice is cast-free and the exact run transports
verbatim. The content of the rounding face here is exactly that: *this kernel
introduces no rounding event of its own*, at any `R`. -/
specification reversed_cumsum_block_store_io_correctnessR (R : RoundingModel)
    (BC Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (hOutInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BT, BS] =>
        p₁ * s_s_h + (p₂ * BT + idx.1.val) * s_s_t
          + (p₀ * BS + idx.2.1.val) * s_s_d)) :
    blockStoreIO BC Z s_s_h s_s_t s_s_d T S BT BS
      ⊨[R, FloatDType.real] fun _p₀ _p₁ xs idx => xs idx := by
  refine Masked3DTileKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact block_store_flattenOk BC Z s_s_h s_s_t s_s_d T S BT BS
  · intro bounds s h1 h2
    exact block_store_traceSafeR R BC Z s_s_h s_s_t s_s_d T S BT BS bounds s
      (fun idx hact => h1 idx hact) (fun idx hact => h2 idx hact)
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      block_store_region_run BC Z s_s_h s_s_t s_s_d T S BT BS s₀
        (hOutInj (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) xs
        (fun idx hact => hx idx hact)
    refine ⟨s1, ?_, ?_, hframe⟩
    · simp only [blockStoreIO]
      rw [block_store_castFree R BC Z s_s_h s_s_t s_s_d T S BT BS s₀]
      exact hexec
    · intro idx hidx
      simp only [blockStoreIO]
      rw [BlockState.readMemAs_real]
      have := hval idx hidx
      simp only [tileOffset, tIndex, sIndex] at this
      rw [this]
      simp

end IOFace

end VeriTile.Bench.TritonBenchG.ReversedCumsum
