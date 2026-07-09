import VeriTile.Triton

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

This kernel is a **global, per-column** cumulative sum: each output `[flat, j]`
holds `Σ_{m ≤ flat} s[m, j]`. The genuine closed-form spec is a standalone
`Finset.sum` (`globalCumsumVectorClosed` / `singleBlockCumsumVectorClosed`),
never a read-back of the kernel's own output.

```
chunk_cumsum_vector_output_summary_general                    ← TOP THEOREM (dimension-general)
  ├─ chunk_cumsum_vector_surface_toAlgorithm_supported        full surface lowers
  ├─ chunk_cumsum_vector_single_block_surface_closed_form
  │    ├─ chunk_cumsum_vector_single_block_surface_active_compute_correct
  │    └─ singleBlockStoreValue_eq_closed   (lower-tri dot = prefix Σ)
  ├─ chunk_cumsum_vector_store_slice_active_compute_correct
  └─ chunk_cumsum_vector_cumsum_slice_closed_form  (under per-column carry hyp.)
            ├─ chunk_cumsum_vector_cumsum_slice_active_compute_correct
            └─ cumsumStoreValue_eq_globalCumsumVectorClosed  (carry + dot = global Σ)

mathematical core (the carry-fold + within-chunk identity):
  dotLowerTri_sum_if_some                        `tl.dot(m_s,·)` = guarded prefix Σ
  singleBlockStoreValue_eq_closed                single chunk (carry=0) = global prefix Σ
  cumsumStoreValue_eq_globalCumsumVectorClosed   carry[j] + within-chunk Σ = global prefix Σ,
                                                 given carry[j] = Σ_{flat<c·BT, flat<T} s[flat,j]
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` (the
`BT ∈ {16,32,64}` config set) is not modeled — the public summary
`chunk_cumsum_vector_output_summary_general` covers the single-Python-chunk regime
(`T ≤ BT`, so each chunk loop runs once with carry `= 0`) at symbolic shapes; the closed-form
lemmas (`dotLowerTri_sum_if_some`, `*_eq_closed`,
`cumsumStoreValue_eq_globalCumsumVectorClosed`, and the
`*_surface_closed_form` / `*_cumsum_slice_closed_form` realizers) are stated and
proven **general over `T`, `S`, `BT`, `BS` and the number of chunks**. The
`.to(tl.float32)` / `.to(p_z.dtype.element_ty)` casts erase to the identity at
the algorithm layer. The in-chunk prefix sum is modeled exactly as the
lower-triangular matmul `lowerTriTile · sourceTile` (matching `tl.dot(m_s, b_s)`)
and shown equal to the genuine per-column prefix `Finset.sum`
(`dotLowerTri_sum_if_some`). The cross-chunk carry recurrence threaded by `b_z` is
`carry_{c+1}[j] = carry_c[j] + Σ chunk_c[·,j]`; its invariant
`carry_c[j] = Σ_{flat < c·BT, flat < T} s[i_bh·s_s_h + flat·s_s_t + j·s_s_d]` is
the explicit hypothesis of `cumsumStoreValue_eq_globalCumsumVectorClosed` — under
it, each chunk's store equals the genuine global cumulative sum. The carry is
materialized in a buffer (`Carry`) in `chunk_cumsum_vector_cumsum_slice`; the
single-Python-chunk surface realizes the global prefix sum end-to-end with
`carry = 0`. Output non-collision is a side condition (discharged per Python
case via `*_active_no_collision`).
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
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## Within-chunk lower-triangular dot = guarded prefix `Finset.sum`

`tl.dot(m_s, b_s)` with `m_s` the lower-triangular ones matrix computes, at row
`i` and column `j`, the within-chunk prefix sum `∑_{k ≤ i} b_s[k, j]`. The
following lemma turns that register-level matmul into a genuine `Finset.sum`
over the guarded prefix, for any column-keyed source `f`. -/

/-- A `Finset.sum` of coerced lanes over `WithBot ℝ` collapses to the coercion
of the underlying real sum. (Instance-safe restatement of `WithBot.coe_sum`.) -/
private theorem sum_some_eq_some {ι : Type*} (s : Finset ι) (g : ι → ℝ) :
    (∑ k ∈ s, ((g k : ℝ) : WithBot ℝ)) = ((∑ k ∈ s, g k : ℝ) : WithBot ℝ) :=
  (WithBot.coe_sum s g).symm

/-- **Lower-triangular dot = guarded prefix sum (`some`-valued).** For a source
tile whose lane `(k, j)` holds `some (if P k j then f k j else 0)`, the
lower-triangular dot at `(i, j)` is `some` of the sum of `f · j` over the guarded
prefix `{k : k ≤ i ∧ P k j}`. This is the 2D (per-column) analogue of
`scan1d_sum_if`. -/
theorem dotLowerTri_sum_if_some (BT BS : Nat) (f : Fin BT → Fin BS → ℝ)
    (P : Fin BT → Fin BS → Prop) [∀ k j, Decidable (P k j)]
    (i : Fin BT) (j : Fin BS) :
    ((Tile.dot [] (lowerTriTile BT)
        (⟨fun idx => some (if P idx.1 idx.2.1 then f idx.1 idx.2.1 else 0)⟩ :
          Tile .real [BT, BS])).data (i, j, PUnit.unit))
      = some (∑ k ∈ (Finset.univ.filter
          (fun k : Fin BT => k.val ≤ i.val ∧ P k j)), f k j) := by
  rw [Tile.dot_nil_data]
  have e : (fun k : Fin BT => Option.map₂ (fun x1 x2 => x1 * x2)
      ((lowerTriTile BT).data (i, k, PUnit.unit))
      ((⟨fun idx => some (if P idx.1 idx.2.1 then f idx.1 idx.2.1 else 0)⟩ :
        Tile .real [BT, BS]).data (k, j, PUnit.unit)))
      = (fun k : Fin BT => ((((if k.val ≤ i.val then (1:ℝ) else 0) *
          (if P k j then f k j else 0)) : ℝ) : WithBot ℝ)) := by
    funext k
    simp only [lowerTriTile, ge_iff_le]
    split <;> (rw [Option.map₂_some_some, WithBot.some_eq_coe]; congr 1; norm_num)
  rw [e, sum_some_eq_some, WithBot.some_eq_coe]
  congr 1
  rw [← Finset.sum_filter_of_ne (p := fun k : Fin BT => k.val ≤ i.val ∧ P k j)]
  · apply Finset.sum_congr rfl
    intro k hk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
    simp [hk.1, hk.2]
  · intro k _ hne
    by_contra hc
    push Not at hc
    by_cases hki : k.val ≤ i.val <;> simp_all

/-! ## Genuine chunked-cumsum closed form (vector / per-column)

`globalCumsumVectorClosed` is the genuine mathematical specification — for each
feature column `j` it is a `Finset.sum` over all *flat* time indices
`flat ≤ i_t·BT + i` (with `flat < T`) of the source value at the 2D address
`i_bh·s_s_h + flat·s_s_t + (i_s·BS + j)·s_s_d`. It is **not** a read-back of the
kernel's own output, so realizing it is a true correctness statement. Padded
feature lanes (`i_s·BS + j ≥ S`) hold `0`, matching the masked store. This is
the 2D analogue of the scalar `globalCumsumClosed`, carrying the column index
`j` through the prefix sum. -/
noncomputable def globalCumsumVectorClosed
    (s : BlockState) (SReg : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  if sIndex s BS idx.2.1 < S then
    ∑ flat ∈ (Finset.range T).filter
        (fun flat => flat ≤ s.pids 2 * BT + idx.1.val),
      s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
        sIndex s BS idx.2.1 * s_s_d)
  else 0

/-- Genuine closed form for the single-Python-chunk path (`i_t = 0`, carry `= 0`):
for each feature column `j`, the prefix sum of all source entries up to and
including flat index `i`. -/
noncomputable def singleBlockCumsumVectorClosed
    (s : BlockState) (SReg : RegionName) (s_s_h s_s_t s_s_d T S BS : Nat)
    (idx : TileIndex [BT, BS]) : ℝ :=
  if sIndex s BS idx.2.1 < S then
    ∑ flat ∈ (Finset.range T).filter (fun flat => flat ≤ idx.1.val),
      s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
        sIndex s BS idx.2.1 * s_s_d)
  else 0

/-- **The carry-fold recurrence (vector).** When the carry buffer `Carry` holds,
per active feature column, the genuine prefix sum of *all prior chunks* (every
flat index `< i_t·BT`, clamped to `< T`), the per-chunk store value
`cumsumStoreValue` — the within-chunk lower-triangular dot plus that carry —
equals the genuine global cumulative sum `globalCumsumVectorClosed`. This is the
exact recurrence threaded by `b_z` across the `forRange` loop:
`carry_{c+1}[j] = carry_c[j] + Σ chunk_c[·,j]`, with the invariant
`carry_c[j] = Σ_{flat < c·BT, flat < T} s[i_bh·s_s_h + flat·s_s_t + j·s_s_d]`. -/
theorem cumsumStoreValue_eq_globalCumsumVectorClosed
    (s : BlockState) (SReg Carry : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS])
    (hcarry : sIndex s BS idx.2.1 < S →
      s.readMem Carry (s.pids 1 * s_s_h + sIndex s BS idx.2.1 * s_s_d)
        = ∑ flat ∈ (Finset.range T).filter (fun flat => flat < s.pids 2 * BT),
            s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
              sIndex s BS idx.2.1 * s_s_d)) :
    cumsumStoreValue s SReg Carry s_s_h s_s_t s_s_d T S BT BS idx
      = globalCumsumVectorClosed s SReg s_s_h s_s_t s_s_d T S BT BS idx := by
  unfold cumsumStoreValue globalCumsumVectorClosed carryValue
  -- The dot operand `sourceTile` is exactly the guarded-`some` tile.
  have hsrc : sourceTile s SReg s_s_h s_s_t s_s_d T S BT BS
      = (⟨fun p => some (if active s T S BT BS (p.1, p.2.1, PUnit.unit) then
          s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS (p.1, p.2.1, PUnit.unit))
          else 0)⟩ : Tile .real [BT, BS]) := by
    unfold sourceTile; congr 1; funext p
    obtain ⟨a, b, u⟩ := p
    by_cases h : active s T S BT BS (a, b, PUnit.unit)
    · simp [h]
    · simp only [h, if_false]; norm_num
  rw [hsrc]
  have hdot := dotLowerTri_sum_if_some BT BS
    (fun k j => s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS (k, j, PUnit.unit)))
    (fun k j => active s T S BT BS (k, j, PUnit.unit)) idx.1 idx.2.1
  rw [hdot]
  by_cases hcol : sIndex s BS idx.2.1 < S
  · -- active column: carry + within-chunk dot = global prefix sum
    rw [if_pos hcol, if_pos hcol, Option.map₂_some_some]
    show _ + _ = _
    rw [hcarry hcol]
    -- now: (Σ within-chunk) + carry = global prefix sum
    -- Step 1: reindex the within-chunk sum to flat segment indices.
    have hreindex :
        (∑ k ∈ Finset.univ.filter
            (fun k : Fin BT => k.val ≤ idx.1.val ∧
              active s T S BT BS (k, idx.2.1, PUnit.unit)),
          s.readMem SReg (tileOffset s s_s_h s_s_t s_s_d BT BS (k, idx.2.1, PUnit.unit)))
        = ∑ flat ∈ (Finset.range T).filter
            (fun flat => s.pids 2 * BT ≤ flat ∧ flat ≤ s.pids 2 * BT + idx.1.val),
            s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
              sIndex s BS idx.2.1 * s_s_d) := by
      simp only [active, tIndex, tileOffset, hcol, and_true]
      apply Finset.sum_nbij'
        (i := fun k : Fin BT => s.pids 2 * BT + k.val)
        (j := fun flat => (⟨if h : flat - s.pids 2 * BT < BT then
            flat - s.pids 2 * BT else 0, by
            split
            · assumption
            · exact idx.1.pos⟩ : Fin BT))
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        simp only [Finset.mem_range, Finset.mem_filter]
        exact ⟨ha.2, by omega, by omega⟩
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_range] at ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        have hd : a - s.pids 2 * BT < BT := by omega
        rw [dif_pos hd]; exact ⟨by omega, by omega⟩
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        apply Fin.ext
        have hd : s.pids 2 * BT + a.val - s.pids 2 * BT < BT := by omega
        simp only [dif_pos hd]; omega
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_range] at ha
        have hd : a - s.pids 2 * BT < BT := by omega
        simp only [dif_pos hd]; omega
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        rfl
    rw [hreindex]
    -- Step 2: split `globalCumsumClosed` by `flat < i_t·BT`. The store value is
    -- `carry + within`, so the carry segment (`flat < i_t·BT`) comes first.
    rw [← Finset.sum_filter_add_sum_filter_not
          ((Finset.range T).filter (fun flat => flat ≤ s.pids 2 * BT + idx.1.val))
          (fun flat => flat < s.pids 2 * BT)]
    congr 1
    · rw [Finset.filter_filter]
      apply Finset.sum_congr ?_ (fun _ _ => rfl)
      apply Finset.filter_congr
      intro flat hflat
      simp only [Finset.mem_range] at hflat
      constructor
      · intro h; exact ⟨by omega, h⟩
      · rintro ⟨_, h2⟩; exact h2
    · rw [Finset.filter_filter]
      apply Finset.sum_congr ?_ (fun _ _ => rfl)
      apply Finset.filter_congr
      intro flat hflat
      simp only [Finset.mem_range, not_lt] at hflat ⊢
      constructor
      · rintro ⟨h1, h2⟩; exact ⟨h2, h1⟩
      · rintro ⟨h1, h2⟩; exact ⟨h2, h1⟩
  · rw [if_neg hcol, if_neg hcol]
    have hempty : Finset.univ.filter
        (fun k : Fin BT => k.val ≤ idx.1.val ∧
          active s T S BT BS (k, idx.2.1, PUnit.unit)) = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro k _
      simp only [active, not_and]
      intro _
      simp only [not_lt] at *
      omega
    rw [hempty, Finset.sum_empty, Option.map₂_some_some]
    show _ + _ = _
    norm_num

/-- **Single-chunk correctness against the genuine closed form (vector).** With
the carry at its initial zero (`i_t = 0`), the within-chunk lower-triangular dot
store value is, per active feature column, the genuine global prefix sum
`Σ_{flat ≤ i, flat < T} s[i_bh·s_s_h + flat·s_s_t + j·s_s_d]`. This is the
sorry-free end-to-end correctness of the actual Python single-chunk path. -/
theorem singleBlockStoreValue_eq_closed
    (s : BlockState) (SReg : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (idx : TileIndex [BT, BS]) :
    singleBlockStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx
      = singleBlockCumsumVectorClosed s SReg s_s_h s_s_t s_s_d T S BS idx := by
  unfold singleBlockStoreValue singleBlockCumsumVectorClosed
  have hsrc : singleBlockSourceTile s SReg s_s_h s_s_t s_s_d T S BT BS
      = (⟨fun p => some (if singleBlockActive s T S BS (p.1, p.2.1, PUnit.unit) then
          s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
            (p.1, p.2.1, PUnit.unit))
          else 0)⟩ : Tile .real [BT, BS]) := by
    unfold singleBlockSourceTile; congr 1; funext p
    obtain ⟨a, b, u⟩ := p
    by_cases h : singleBlockActive s T S BS (a, b, PUnit.unit)
    · simp [h]
    · simp only [h, if_false]; norm_num
  rw [hsrc, dotLowerTri_sum_if_some BT BS
    (fun k j => s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
      (k, j, PUnit.unit)))
    (fun k j => singleBlockActive s T S BS (k, j, PUnit.unit)) idx.1 idx.2.1]
  show WithBot.unbotD 0 (some _) = _
  by_cases hcol : sIndex s BS idx.2.1 < S
  · rw [if_pos hcol]
    show (∑ k ∈ Finset.univ.filter
        (fun k : Fin BT => k.val ≤ idx.1.val ∧
          singleBlockActive s T S BS (k, idx.2.1, PUnit.unit)),
        s.readMem SReg (singleBlockTileOffset s s_s_h s_s_t s_s_d BS
          (k, idx.2.1, PUnit.unit))) = _
    -- reindex `{k : Fin BT | k ≤ i ∧ (k < T ∧ col)}` to `{flat ∈ range T | flat ≤ i}`
    simp only [singleBlockActive, hcol, and_true, singleBlockTileOffset, sIndex]
    apply Finset.sum_nbij' (i := fun k : Fin BT => k.val)
      (j := fun flat => (⟨if h : flat < BT then flat else 0, by
          split
          · assumption
          · exact idx.1.pos⟩ : Fin BT))
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      simp only [Finset.mem_range, Finset.mem_filter]
      exact ⟨ha.2.1, ha.1⟩
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_range] at ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      have hd : a < BT := by have : a ≤ idx.1.val := ha.2; omega
      rw [dif_pos hd]; exact ⟨ha.2, ha.1, hcol⟩
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      apply Fin.ext
      simp only [dif_pos a.isLt]
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_range] at ha
      have hd : a < BT := by have : a ≤ idx.1.val := ha.2; omega
      simp only [dif_pos hd]
    · intro a _; rfl
  · rw [if_neg hcol]
    have hempty : Finset.univ.filter
        (fun k : Fin BT => k.val ≤ idx.1.val ∧
          singleBlockActive s T S BS (k, idx.2.1, PUnit.unit)) = ∅ := by
      apply Finset.filter_eq_empty_iff.mpr
      intro k _
      simp only [singleBlockActive, not_and]
      intro _
      simp only [not_lt] at *
      omega
    rw [hempty, Finset.sum_empty]
    rfl

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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## Genuine closed-form realizers

These wrap the `*_active_compute_correct` realizers with the closed-form
equivalence lemmas, so the realized `expected` is a standalone `Finset.sum`
specification (the genuine chunked cumulative sum), never a read-back of the
kernel's own output. -/

/-- **Genuine single-chunk correctness (vector).** The single-Python-chunk
surface (the actual `S → Z` path, carry `= 0`) realizes the genuine closed-form
per-column global prefix sum `singleBlockCumsumVectorClosed`. -/
theorem chunk_cumsum_vector_single_block_surface_closed_form
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => singleBlockActive s T S BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        singleBlockCumsumVectorClosed s SReg s_s_h s_s_t s_s_d T S BS idx) := by
  have h := chunk_cumsum_vector_single_block_surface_active_compute_correct
    SReg Z s_s_h s_s_t s_s_d T S BT BS s hNoCollision
  have hcong : (fun idx : TileIndex [BT, BS] =>
      singleBlockStoreValue s SReg s_s_h s_s_t s_s_d T S BT BS idx)
      = (fun idx : TileIndex [BT, BS] =>
        singleBlockCumsumVectorClosed s SReg s_s_h s_s_t s_s_d T S BS idx) := by
    funext idx; exact singleBlockStoreValue_eq_closed s SReg s_s_h s_s_t s_s_d T S BT BS idx
  rwa [hcong] at h

/-- **Genuine per-chunk carry-fold correctness (vector).** Given the carry
buffer holds, per active feature column, the genuine prefix sum of all prior
chunks, the cumsum slice realizes the genuine global cumulative sum
`globalCumsumVectorClosed`. This is the inductive step of the carry recurrence
threaded by `b_z`. -/
theorem chunk_cumsum_vector_cumsum_slice_closed_form
    (SReg Carry Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
    (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], active s T S BT BS idx →
      ∀ k : TileIndex [BT, BS], active s T S BT BS k →
        tileOffset s s_s_h s_s_t s_s_d BT BS k =
          tileOffset s s_s_h s_s_t s_s_d BT BS idx → k = idx)
    (hcarry : ∀ idx : TileIndex [BT, BS], sIndex s BS idx.2.1 < S →
      s.readMem Carry (s.pids 1 * s_s_h + sIndex s BS idx.2.1 * s_s_d)
        = ∑ flat ∈ (Finset.range T).filter (fun flat => flat < s.pids 2 * BT),
            s.readMem SReg (s.pids 1 * s_s_h + flat * s_s_t +
              sIndex s BS idx.2.1 * s_s_d)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_cumsum_vector_cumsum_slice SReg Carry Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => active s T S BT BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, tileOffset s s_s_h s_s_t s_s_d BT BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        globalCumsumVectorClosed s SReg s_s_h s_s_t s_s_d T S BT BS idx) := by
  have h := chunk_cumsum_vector_cumsum_slice_active_compute_correct
    SReg Carry Z s_s_h s_s_t s_s_d T S BT BS s hNoCollision
  have hcong : (fun idx : TileIndex [BT, BS] =>
      cumsumStoreValue s SReg Carry s_s_h s_s_t s_s_d T S BT BS idx)
      = (fun idx : TileIndex [BT, BS] =>
        globalCumsumVectorClosed s SReg s_s_h s_s_t s_s_d T S BT BS idx) := by
    funext idx
    exact cumsumStoreValue_eq_globalCumsumVectorClosed s SReg Carry s_s_h s_s_t
      s_s_d T S BT BS idx (hcarry idx)
  rwa [hcong] at h

/-- **Dimension-general single-chunk output summary.** Subsumes the former
per-shape Python-case summaries (which differed only in the
concrete `s_s_h s_s_t s_s_d T S BT BS` numerals) at fully symbolic dimensions.
For any single-Python-chunk shape (`T ≤ BT`, so the loop runs once with carry
`= 0`), the single-block `S → Z` surface realizes the genuine per-column global
prefix sum `singleBlockCumsumVectorClosed` — a standalone `Finset.sum`, never a
read-back of the kernel's own output. The active-lane collision-freedom of the
block address map is the explicit hypothesis. -/
theorem chunk_cumsum_vector_output_summary_general
    (SReg Z : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat) (s : BlockState)
    (hNoCollision : ∀ idx : TileIndex [BT, BS], singleBlockActive s T S BS idx →
      ∀ k : TileIndex [BT, BS], singleBlockActive s T S BS k →
        singleBlockTileOffset s s_s_h s_s_t s_s_d BS k =
          singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx → k = idx) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_cumsum_vector_single_block_surface SReg Z s_s_h s_s_t
        s_s_d T S BT BS)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BS] => singleBlockActive s T S BS idx)
        (fun idx : TileIndex [BT, BS] =>
          (Z, singleBlockTileOffset s s_s_h s_s_t s_s_d BS idx)))
      (expected := fun idx : TileIndex [BT, BS] =>
        singleBlockCumsumVectorClosed s SReg s_s_h s_s_t s_s_d T S BS idx) :=
  chunk_cumsum_vector_single_block_surface_closed_form SReg Z s_s_h s_s_t s_s_d
    T S BT BS s hNoCollision

end VeriTile.Bench.TritonBenchG.ChunkCumsumVector
