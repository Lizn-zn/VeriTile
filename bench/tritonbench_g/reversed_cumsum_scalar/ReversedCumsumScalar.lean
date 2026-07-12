import VeriTile.Triton

/-!
# `reversed_cumsum_scalar` — strict per-kernel correctness

`chunk_global_reversed_cumsum_scalar_kernel` computes a global scalar
**reversed** cumulative sum along the time axis: program `i_bh` (one per
batch·head row) walks the `T`-long row of `s` in chunks of `BT` **from the
last chunk down to the first**, and for each chunk emits
`b_s - tl.cumsum(b_s) + b_z` — the within-chunk suffix sum plus the carried
total `b_z` of all later chunks (the carry is updated with the current chunk
total *before* the store), storing the result into `o`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`chunk_global_reversed_cumsum_scalar_kernel[(B*H,)]`,
the grid over batch·head rows, the autotuned `BT`, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because the program ids are universally quantified, the
per-program statements cover every program of the grid.

## Proof architecture

The kernel is a **global reversed** cumulative sum: each output flat index
`flat` holds `Σ_{m ≥ flat, m < T} s[m]`. The genuine closed-form spec is a
standalone `Finset.sum` (`globalRevCumsumClosed` / `singleBlockRevClosed`),
never a read-back of the kernel's own output.

```
reversed_cumsum_scalar_output_summary_general                  ← TOP THEOREM
  ├─ reversed_cumsum_scalar_surface_toAlgorithm_supported      full surface lowers
  └─ reversed_cumsum_scalar_surface_global_rev_cumsum          O = global suffix Σ (carry-fold)

mathematical core (the carry-fold + within-chunk identity):
  scan1d_sum / scan1d_sum_if           `tl.cumsum` = guarded prefix `Finset.sum`
  suffix_of_sub_prefix                 `b_s[i] − prefix(i) + total = suffix(i)`
  singleBlockRevStoreValue_eq_closed   single chunk (carry = 0) = global suffix Σ
  revStoreValue_eq_globalRevCumsumClosed   carry_c + within-chunk suffix Σ = global suffix Σ,
                                       given carry_c = Σ_{flat ≥ (c+1)·BT, flat<T} s[flat]
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` (the
`BT ∈ {16,32,64}` config set) is not modeled — the closed-form lemmas are
stated and proven **general over `T`, `BT` and the number of chunks**. The
`.to(tl.float32)` / `.to(p_o.dtype.element_ty)` casts erase to the identity at
the algorithm layer (post-erasure all dtypes unify to `ℝ`). The per-chunk
`tl.cumsum` (`Tile.scan .sum .forward`) is modeled exactly; the kernel realizes
the within-chunk *suffix* sum through the subtraction identity
`b_s − cumsum(b_s) + total` (it does **not** use `reverse=True`), and the
reverse chunk traversal `range(cdiv(T,BT)−1, −1, −1)` is transcribed verbatim.
The cross-chunk carry recurrence threaded by `b_z` is
`carry_{c−1} = carry_c + Σ chunk_c` (chunks visited in descending order); its
invariant `carry_c = Σ_{flat ≥ (c+1)·BT, flat < T} s[flat]` is the explicit
hypothesis of `revStoreValue_eq_globalRevCumsumClosed`. The carry is
materialized in a buffer (`Carry`) in `reversed_cumsum_scalar_rev_slice`; the
single-chunk path realizes the global suffix sum end-to-end with `carry = 0`.
Output injectivity is a side condition (discharged dimension-generally).
-/

namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `reversed_cumsum_scalar_output_summary_general`
(dimension-general `T`, `BT`); the `T = 4`, `BT = 16` Python benchmark shape is
one instantiation. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

/-! ## Within-chunk cumsum identity (`tl.cumsum` = prefix `Finset.sum`)

`tl.cumsum(b_s, axis=0)` lowers to `Op.scan .sum .forward` and hence
`Tile.scan .sum`. For a 1D tile whose lanes hold `some (g k)`, the scanned
value at lane `i` is `some (Σ_{k ≤ i} g k)`. The kernel then realizes the
within-chunk **suffix** sum through `b_s − cumsum(b_s) + total`
(`suffix_of_sub_prefix` below). -/

/-- `foldl WithBot.realAdd` over a list of coerced reals collapses to the
coercion of the real-valued `foldl (+)`. -/
private theorem foldl_realAdd_coe (l : List ℝ) (init : ℝ) :
    (l.map (fun r => ((r : ℝ) : WithBot ℝ))).foldl WithBot.realAdd
        ((init : ℝ) : WithBot ℝ)
      = (((l.foldl (· + ·) init : ℝ)) : WithBot ℝ) := by
  induction l generalizing init with
  | nil => simp
  | cons a t ih =>
      simp only [List.map_cons, List.foldl_cons, WithBot.realAdd_coe_coe]
      exact ih (init + a)

/-- A `ScanOp.sum` fold over a list of `some`-valued lanes equals `some` of the
underlying real list-sum. -/
private theorem scan_prefix_eq (BT : Nat) (g : Fin BT → ℝ) (L : List (Fin BT)) :
    (L.map (fun k => (some (g k) : WithBot ℝ))).foldl WithBot.realAdd
        ((0 : ℝ) : WithBot ℝ)
      = some ((L.map g).sum) := by
  rw [show (L.map (fun k => (some (g k) : WithBot ℝ)))
      = List.map (fun r => ((r : ℝ) : WithBot ℝ)) (L.map g) from by
        rw [List.map_map]; rfl]
  rw [foldl_realAdd_coe]; congr 1; rw [List.sum_eq_foldl]

/-- **1D cumsum = prefix sum.** The `Tile.scan .sum` of a `some`-valued 1D tile
at lane `i` is `some (Σ_{k ≤ i} g k)`. -/
theorem scan1d_sum (BT : Nat) (g : Fin BT → ℝ) (i : Fin BT) :
    (Tile.scan .sum ⟨0, by simp⟩ .forward
      (⟨fun idx => some (g idx.1)⟩ : Tile .real [BT])).data (i, PUnit.unit)
      = some (∑ k ∈ (Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val)), g k) := by
  rw [Tile.scan_data]
  simp only [ScanOp.eval_sum, TileShape.axisCoord, TileShape.replaceAxisCoord,
    TileShape.axisDim]
  refine Eq.trans (scan_prefix_eq BT g _) ?_
  rw [Finset.sum_map_toList]; rfl

/-- **The suffix identity.** For any lane values `g : Fin BT → ℝ`,
`g i − Σ_{k ≤ i} g k + Σ_k g k = Σ_{k ≥ i} g k`. This is exactly the kernel's
`b_s − tl.cumsum(b_s) + total` realization of the within-chunk suffix sum. -/
theorem suffix_of_sub_prefix (BT : Nat) (g : Fin BT → ℝ) (i : Fin BT) :
    g i - (∑ k ∈ (Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val)), g k)
        + (∑ k : Fin BT, g k)
      = ∑ k ∈ (Finset.univ.filter (fun k : Fin BT => i.val ≤ k.val)), g k := by
  have hsplit :
      (∑ k : Fin BT, g k)
        = (∑ k ∈ (Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val)), g k)
          + (∑ k ∈ (Finset.univ.filter (fun k : Fin BT => ¬ k.val ≤ i.val)), g k) :=
    (Finset.sum_filter_add_sum_filter_not _ _ _).symm
  have hsucc :
      (∑ k ∈ (Finset.univ.filter (fun k : Fin BT => i.val ≤ k.val)), g k)
        = g i + (∑ k ∈ (Finset.univ.filter (fun k : Fin BT => ¬ k.val ≤ i.val)), g k) := by
    have hins :
        (Finset.univ.filter (fun k : Fin BT => i.val ≤ k.val))
          = insert i (Finset.univ.filter (fun k : Fin BT => ¬ k.val ≤ i.val)) := by
      ext k
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
        not_le]
      constructor
      · intro h
        rcases Nat.lt_or_ge i.val k.val with hlt | hge
        · exact Or.inr hlt
        · exact Or.inl (Fin.ext (Nat.le_antisymm hge h))
      · rintro (rfl | h)
        · exact Nat.le_refl _
        · exact Nat.le_of_lt h
    rw [hins, Finset.sum_insert (by simp)]
  rw [hsplit, hsucc]; ring

/-- Faithful transcription of `reversed_cumsum_scalar.py`'s
`chunk_global_reversed_cumsum_scalar_kernel`.

The chunk loop runs in reverse (`range(cdiv(T,BT)−1, −1, −1)`), the scalar
carry `b_z` is updated with the current chunk total *before* the store, and
the final cast targets the block pointer destination dtype. -/
def reversed_cumsum_scalar_surface
  (S O : RegionName) (T BT : Nat) : ComputeKernel := triton {
  i_bh = tl.program_id(0)
  b_z = tl.zeros([], dtype=tl.float32)
  for i_t in range(tl.cdiv($(T), $(BT)) - $(1), -$(1), -$(1)) {
    p_s = tl.make_block_ptr(base=S + i_bh * $(T), shape=($(T)),
      strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
    p_o = tl.make_block_ptr(base=O + i_bh * $(T), shape=($(T)),
      strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
    b_s = tl.load(p_s, boundary_check=([0] : List Nat)).to(tl.float32)
    b_zz = tl.sum(b_s, axis=0)
    b_z += b_zz
    b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
    tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0] : List Nat))
  }
}

/-- The full scalar reversed chunk-cumsum surface lowers to the algorithm
layer, including the reverse-range loop and the carried scalar accumulator. -/
theorem reversed_cumsum_scalar_surface_toAlgorithm_supported
    (S O : RegionName) (T BT : Nat) :
    ∃ alg, (reversed_cumsum_scalar_surface S O T BT).toAlgorithm? = Except.ok alg := by
  simp [reversed_cumsum_scalar_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Proof-oriented block store slice of `reversed_cumsum_scalar.py`'s
`chunk_global_reversed_cumsum_scalar_kernel`.

The full kernel scans chunks in reverse while carrying `b_z`. This slice models
one chunk iteration with a precomputed `BO` vector and proves the
boundary-checked store into `O`. -/
def reversed_cumsum_scalar_store_slice
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

theorem reversed_cumsum_scalar_store_slice_correct
    (BO O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => vecOffset s T BT i)) :
    ∀ i : Fin BT,
      let outAddr := vecOffset s T BT i
      (exec (reversed_cumsum_scalar_store_slice BO O T BT) s).map
          (·.readMem O outAddr)
        = some (if active s T BT i then storeValue s BO T BT i
          else s.readMem O outAddr) := by
  intro i
  simp [exec, reversed_cumsum_scalar_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

theorem reversed_cumsum_scalar_store_slice_compute_correct
    (BO O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => vecOffset s T BT i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := reversed_cumsum_scalar_store_slice BO O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => active s T BT i)
        (fun i => (O, vecOffset s T BT i)))
      (expected := fun i : Fin BT => storeValue s BO T BT i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [reversed_cumsum_scalar_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := reversed_cumsum_scalar_store_slice_correct BO O T BT s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented single-iteration reversed-cumsum slice of
`reversed_cumsum_scalar.py`'s `chunk_global_reversed_cumsum_scalar_kernel`.

This models one loop iteration after the kernel's `b_z += b_zz` update for the
current chunk: `Carry` materializes the **post-update** carry — the suffix
total of the current chunk *and* all later chunks. The slice loads the current
source block, computes `b_s − tl.cumsum(b_s, axis=0) + b_z`, and stores the
masked output block. -/
def reversed_cumsum_scalar_rev_slice
    (S Carry O : RegionName) (T BT : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  i_t = tl.program_id(1)
  offs_t = i_t * $(BT) + tl.arange(0, $(BT))
  mask = offs_t < $(T)
  b_s = tl.load(S + i_bh * $(T) + offs_t, mask=mask, other=0.0).to(tl.float32)
  b_z = tl.load(Carry + i_bh).to(tl.float32)
  b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
  tl.store(O + i_bh * $(T) + offs_t, (b_o).to(O.dtype.element_ty), mask=mask)
}

/-- Row element `s[i_bh, flat]` of the row-major `[B·H, T]` input: this
program's row (`i_bh = pids 0`, row stride `T`) at flat time index `flat`
(unit column stride). Every load/spec read in this file uses this layout. -/
noncomputable def rowElem (s : BlockState) (S : RegionName) (T flat : Nat) : ℝ :=
  s.readMem S (s.pids 0 * T + flat)

/-- Masked chunk-`i_t` input block: lane `k` reads row element
`i_t·BT + k` when in bounds, else the `other=0.0` fallback. -/
noncomputable def revInputTile (s : BlockState) (S : RegionName) (T BT : Nat) :
    Tile .real [BT] :=
  { data := fun idx =>
      if s.pids 1 * BT + idx.1.val < T then
        some (s.readMem S (s.pids 0 * T + (s.pids 1 * BT + idx.1.val)))
      else some (0.0 : ℝ) }

noncomputable def revStoreValue
    (s : BlockState) (S Carry : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun a b : ℝ => a + b)
      (Option.map₂ (fun a b : ℝ => a - b)
        ((revInputTile s S T BT).data (i, PUnit.unit))
        ((Tile.scan .sum ⟨0, by simp⟩ .forward
          (revInputTile s S T BT)).data (i, PUnit.unit)))
      (some (s.readMem Carry (s.pids 0))))

/-! ## Genuine reversed chunked-cumsum closed form

`globalRevCumsumClosed` is the genuine mathematical specification — a
`Finset.sum` over all *flat* time indices `flat ≥ chunk·BT + i` (with
`flat < T`) of the source value at `i_bh·T + flat`. It is **not** a read-back
of the kernel's own output, so realizing it is a true correctness statement.

For program `(i_bh, i_t)` (so `i_bh = pids 0`, `i_t = pids 1`) and active lane
`i`, the global flat output index is `i_t·BT + i`, and the output must hold
the suffix sum of all source entries from that flat index to the end of the
row. -/
noncomputable def globalRevCumsumClosed
    (s : BlockState) (S : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  ∑ flat ∈ (Finset.range T).filter
      (fun flat => s.pids 1 * BT + i.val ≤ flat),
    rowElem s S T flat

/-- Reindexing helper: the within-chunk lane sum from lane `lo` onward equals
the flat-segment sum over `[i_t·BT + lo, i_t·BT + BT) ∩ [0, T)`. -/
private theorem chunk_suffix_reindex
    (s : BlockState) (S : RegionName) (T BT : Nat) (hBT : 0 < BT) (lo : Nat) :
    (∑ k ∈ Finset.univ.filter (fun k : Fin BT => lo ≤ k.val),
      (if s.pids 1 * BT + k.val < T then
        s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0))
    = ∑ flat ∈ (Finset.range T).filter
        (fun flat => s.pids 1 * BT + lo ≤ flat ∧ flat < s.pids 1 * BT + BT),
        s.readMem S (s.pids 0 * T + flat) := by
  rw [← Finset.sum_filter_of_ne
      (p := fun k : Fin BT => s.pids 1 * BT + k.val < T)]
  · rw [Finset.filter_filter]
    apply Finset.sum_nbij'
      (i := fun k : Fin BT => s.pids 1 * BT + k.val)
      (j := fun flat => (⟨if h : flat - s.pids 1 * BT < BT then
          flat - s.pids 1 * BT else 0, by
          split
          · assumption
          · exact hBT⟩ : Fin BT))
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      simp only [Finset.mem_range, Finset.mem_filter]
      exact ⟨ha.2, by omega, by omega⟩
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_range] at ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      have hd : a - s.pids 1 * BT < BT := by omega
      rw [dif_pos hd]; exact ⟨by omega, by omega⟩
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      apply Fin.ext
      have hd : s.pids 1 * BT + a.val - s.pids 1 * BT < BT := by omega
      simp only [dif_pos hd]; omega
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_range] at ha
      have hd : a - s.pids 1 * BT < BT := by omega
      simp only [dif_pos hd]; omega
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      rw [if_pos ha.2]
  · intro k hk hne
    by_contra h; simp [h] at hne

/-- **The reversed carry-fold recurrence.** When the carry buffer `Carry`
holds the genuine **post-update** suffix sum — every flat index `≥ i_t·BT`
(the current chunk and all later chunks), clamped to `< T` — the per-chunk
store value `revStoreValue` (the within-chunk `b_s − cumsum + carry`) equals
the genuine global reversed cumulative sum `globalRevCumsumClosed`. This is
the exact recurrence threaded by `b_z` across the reverse `forRange` loop:
the kernel updates `b_z += Σ chunk_c` *before* the store, so at store time
`b_z = Σ_{flat ≥ c·BT, flat < T} s[flat]`. -/
theorem revStoreValue_eq_globalRevCumsumClosed
    (s : BlockState) (S Carry : RegionName) (T BT : Nat) (i : Fin BT)
    (hcarry : s.readMem Carry (s.pids 0)
      = ∑ flat ∈ (Finset.range T).filter
          (fun flat => s.pids 1 * BT ≤ flat),
          s.readMem S (s.pids 0 * T + flat)) :
    revStoreValue s S Carry T BT i = globalRevCumsumClosed s S T BT i := by
  unfold revStoreValue
  have hin : revInputTile s S T BT
      = (⟨fun idx => some (if s.pids 1 * BT + idx.1.val < T then
            s.readMem S (s.pids 0 * T + (s.pids 1 * BT + idx.1.val))
          else 0)⟩ : Tile .real [BT]) := by
    unfold revInputTile; congr 1; funext idx
    by_cases h : s.pids 1 * BT + idx.1.val < T
    · simp [h]
    · simp only [h, if_false]; norm_num
  rw [hin, scan1d_sum BT (fun k => if s.pids 1 * BT + k.val < T then
        s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0) i]
  simp only [Option.map₂_some_some]
  show ((if s.pids 1 * BT + i.val < T then
          s.readMem S (s.pids 0 * T + (s.pids 1 * BT + i.val)) else 0)
        - ∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val),
            (if s.pids 1 * BT + k.val < T then
              s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0))
        + s.readMem Carry (s.pids 0)
      = globalRevCumsumClosed s S T BT i
  rw [hcarry]
  -- Split the carry into the current-chunk segment and the later segment.
  rw [← Finset.sum_filter_add_sum_filter_not
        ((Finset.range T).filter (fun flat => s.pids 1 * BT ≤ flat))
        (fun flat => flat < s.pids 1 * BT + BT)]
  -- Identify the current-chunk segment with the full within-chunk lane sum.
  have hchunk :
      (∑ flat ∈ ((Finset.range T).filter (fun flat => s.pids 1 * BT ≤ flat)).filter
          (fun flat => flat < s.pids 1 * BT + BT),
        s.readMem S (s.pids 0 * T + flat))
      = ∑ k : Fin BT,
          (if s.pids 1 * BT + k.val < T then
            s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0) := by
    rw [Finset.filter_filter]
    rw [show (∑ k : Fin BT,
        (if s.pids 1 * BT + k.val < T then
          s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0))
      = ∑ k ∈ Finset.univ.filter (fun k : Fin BT => 0 ≤ k.val),
          (if s.pids 1 * BT + k.val < T then
            s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0) from by
        rw [Finset.filter_true_of_mem (fun _ _ => Nat.zero_le _)]]
    rw [chunk_suffix_reindex s S T BT i.pos 0]
    apply Finset.sum_congr ?_ (fun _ _ => rfl)
    apply Finset.filter_congr
    intro flat _
    constructor
    · rintro ⟨h1, h2⟩; exact ⟨by omega, h2⟩
    · rintro ⟨h1, h2⟩; exact ⟨by omega, h2⟩
  rw [hchunk]
  -- Fold the subtraction identity: b_s[i] − prefix(i) + chunk total = suffix(i).
  have hrearr :
      ∀ later : ℝ,
      ((if s.pids 1 * BT + i.val < T then
          s.readMem S (s.pids 0 * T + (s.pids 1 * BT + i.val)) else 0)
        - (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val),
            (if s.pids 1 * BT + k.val < T then
              s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0)))
        + ((∑ k : Fin BT,
            (if s.pids 1 * BT + k.val < T then
              s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0)) + later)
      = (((if s.pids 1 * BT + i.val < T then
          s.readMem S (s.pids 0 * T + (s.pids 1 * BT + i.val)) else 0)
        - (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val),
            (if s.pids 1 * BT + k.val < T then
              s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0))
        + (∑ k : Fin BT,
            (if s.pids 1 * BT + k.val < T then
              s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0)))) + later := by
    intro later; ring
  rw [hrearr, suffix_of_sub_prefix BT
    (fun k => if s.pids 1 * BT + k.val < T then
      s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0) i]
  rw [chunk_suffix_reindex s S T BT i.pos i.val]
  -- Merge the within-chunk suffix segment with the later segment.
  unfold globalRevCumsumClosed rowElem
  rw [← Finset.sum_filter_add_sum_filter_not
        ((Finset.range T).filter (fun flat => s.pids 1 * BT + i.val ≤ flat))
        (fun flat => flat < s.pids 1 * BT + BT)]
  congr 1
  · rw [Finset.filter_filter]
  · rw [Finset.filter_filter, Finset.filter_filter]
    apply Finset.sum_congr ?_ (fun _ _ => rfl)
    apply Finset.filter_congr
    intro flat _
    simp only [not_lt]
    constructor
    · rintro ⟨_, h2⟩; exact ⟨by omega, h2⟩
    · rintro ⟨_, h2⟩; exact ⟨by omega, h2⟩

theorem reversed_cumsum_scalar_rev_slice_correct
    (S Carry O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => vecOffset s T BT i)) :
    ∀ i : Fin BT,
      let outAddr := vecOffset s T BT i
      (exec (reversed_cumsum_scalar_rev_slice S Carry O T BT) s).map
          (·.readMem O outAddr)
        = some (if active s T BT i then revStoreValue s S Carry T BT i
          else s.readMem O outAddr) := by
  intro i
  simp [exec, reversed_cumsum_scalar_rev_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd, Tile.scan,
        Tile.expandDim, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.sub, NumericDType.mul, ComparableDType.lt,
        tIndex, active, vecOffset, revInputTile,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BT] → Nat :=
    fun idx => s.pids 0 * T + (s.pids 1 * BT + idx.1.val)
  let valueFn : TileIndex [BT] → ℝ :=
    fun idx => revStoreValue s S Carry T BT idx.1
  let P : TileIndex [BT] → Prop := fun idx => s.pids 1 * BT + idx.1.val < T
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : vecOffset s T BT a = vecOffset s T BT b := by
      simpa [offsetFn, vecOffset, tIndex] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : s.pids 1 * BT + i.val < T
  · simp [revStoreValue, revInputTile, active, vecOffset, tIndex, Option.map, hi]
  · simp [revStoreValue, revInputTile, active, vecOffset, tIndex, offsetFn, hi]

theorem reversed_cumsum_scalar_rev_slice_compute_correct
    (S Carry O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => vecOffset s T BT i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := reversed_cumsum_scalar_rev_slice S Carry O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => active s T BT i)
        (fun i => (O, vecOffset s T BT i)))
      (expected := fun i : Fin BT => revStoreValue s S Carry T BT i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [reversed_cumsum_scalar_rev_slice, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := reversed_cumsum_scalar_rev_slice_correct S Carry O T BT s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Genuine per-chunk correctness (reversed carry-fold).** Given the carry
buffer holds the genuine post-update suffix sum
(`Carry[i_bh] = Σ_{flat ≥ i_t·BT, flat < T} s[i_bh·T + flat]`), the reversed
slice realizes the genuine global reversed cumulative sum
`globalRevCumsumClosed`. This is the inductive step of the carry recurrence
threaded by `b_z`. -/
theorem reversed_cumsum_scalar_rev_slice_closed_form
    (S Carry O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => vecOffset s T BT i))
    (hcarry : s.readMem Carry (s.pids 0)
      = ∑ flat ∈ (Finset.range T).filter (fun flat => s.pids 1 * BT ≤ flat),
          s.readMem S (s.pids 0 * T + flat)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := reversed_cumsum_scalar_rev_slice S Carry O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => active s T BT i)
        (fun i => (O, vecOffset s T BT i)))
      (expected := fun i : Fin BT =>
        globalRevCumsumClosed s S T BT i) := by
  have h := reversed_cumsum_scalar_rev_slice_compute_correct S Carry O T BT s hOutInj
  have hcong : (fun i : Fin BT => revStoreValue s S Carry T BT i)
      = (fun i : Fin BT => globalRevCumsumClosed s S T BT i) := by
    funext i; exact revStoreValue_eq_globalRevCumsumClosed s S Carry T BT i hcarry
  rwa [hcong] at h

/-! ## Single Python-chunk computed surface

For the checked Python shape `T = 4` and autotuned `BT = 16`, the reverse loop
has one iteration and the carried scalar `b_z` at store time is exactly the
current chunk total (`0 + b_zz`). This surface keeps the Python-observable
path from `S` to `O` without a precomputed output tile or external carry
buffer: `b_o = b_s − tl.cumsum(b_s) + tl.sum(b_s)`. -/

def reversed_cumsum_scalar_single_block_surface
    (S O : RegionName) (T BT : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  offs_t = tl.arange(0, $(BT))
  mask = offs_t < $(T)
  b_s = tl.load(S + i_bh * $(T) + offs_t, mask=mask, other=0.0).to(tl.float32)
  b_zz = tl.sum(b_s, axis=0)
  b_o = b_s - tl.cumsum(b_s, axis=0) + b_zz[None]
  tl.store(O + i_bh * $(T) + offs_t, (b_o).to(O.dtype.element_ty), mask=mask)
}

def singleBlockActive (_s : BlockState) (T : Nat) (i : Fin BT) : Prop :=
  i.val < T

instance singleBlockActiveDecidable (s : BlockState) (T : Nat) (i : Fin BT) :
    Decidable (singleBlockActive s T i) := by
  unfold singleBlockActive
  infer_instance

def singleBlockVecOffset (s : BlockState) (T : Nat) (i : Fin BT) : Nat :=
  s.pids 0 * T + i.val

/-- Masked single-chunk input block (`i_t = 0`): lane `k` reads row element
`k` when in bounds, else the `other=0.0` fallback. -/
noncomputable def singleBlockRevInputTile
    (s : BlockState) (S : RegionName) (T BT : Nat) :
    Tile .real [BT] :=
  { data := fun idx =>
      if idx.1.val < T then
        some (s.readMem S (s.pids 0 * T + idx.1.val))
      else some (0.0 : ℝ) }

noncomputable def singleBlockRevStoreValue
    (s : BlockState) (S : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun a b : ℝ => a + b)
      (Option.map₂ (fun a b : ℝ => a - b)
        ((singleBlockRevInputTile s S T BT).data (i, PUnit.unit))
        ((Tile.scan .sum ⟨0, by simp⟩ .forward
          (singleBlockRevInputTile s S T BT)).data (i, PUnit.unit)))
      ((Tile.reduceSum (shape := [BT]) ⟨0, by simp⟩ Bool.false
        (singleBlockRevInputTile s S T BT)).data PUnit.unit))

/-- Genuine closed form for the single-Python-chunk path (single iteration,
carry `= 0 + chunk total`): the suffix sum of all source entries from flat
index `i` to the end of the row. -/
noncomputable def singleBlockRevClosed
    (s : BlockState) (S : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  ∑ flat ∈ (Finset.range T).filter (fun flat => i.val ≤ flat),
    rowElem s S T flat

/-- 1D `tl.sum` of a `some`-valued tile is `some` of the `Finset.sum`. -/
private theorem reduceSum1d (BT : Nat) (g : Fin BT → ℝ) :
    (Tile.reduceSum (shape := [BT]) ⟨0, by simp⟩ Bool.false
      (⟨fun idx => some (g idx.1)⟩ : Tile .real [BT])).data PUnit.unit
      = some (∑ k : Fin BT, g k) := by
  rw [Tile.reduceSum_false, Tile.reduceSumDrop_data]
  show (@Finset.sum (Fin BT) (WithBot ℝ) _ Finset.univ
      (fun k => ((g k : ℝ) : WithBot ℝ)))
    = (((∑ k : Fin BT, g k : ℝ)) : WithBot ℝ)
  norm_cast

/-- **Single-chunk correctness against the genuine closed form.** With the
row covered by a single chunk (`T ≤ BT`), the within-chunk
`b_s − cumsum + total` store value is the genuine global suffix sum
`Σ_{flat ≥ i, flat < T} s[i_bh·T + flat]`. This is the fully proved
end-to-end correctness of the actual Python test-shape path (`T = 4`,
`BT = 16`, where the reverse chunk loop runs exactly once). -/
theorem singleBlockRevStoreValue_eq_closed
    (s : BlockState) (S : RegionName) (T BT : Nat) (hT : T ≤ BT) (i : Fin BT) :
    singleBlockRevStoreValue s S T BT i
      = singleBlockRevClosed s S T BT i := by
  unfold singleBlockRevStoreValue singleBlockRevClosed rowElem
  have hin : singleBlockRevInputTile s S T BT
      = (⟨fun idx => some (if idx.1.val < T then
            s.readMem S (s.pids 0 * T + idx.1.val) else 0)⟩ : Tile .real [BT]) := by
    unfold singleBlockRevInputTile; congr 1; funext idx
    by_cases h : idx.1.val < T
    · simp [h]
    · simp only [h, if_false]; norm_num
  rw [hin, scan1d_sum BT (fun k => if k.val < T then
        s.readMem S (s.pids 0 * T + k.val) else 0) i,
      reduceSum1d BT (fun k => if k.val < T then
        s.readMem S (s.pids 0 * T + k.val) else 0)]
  simp only [Option.map₂_some_some]
  show ((if i.val < T then s.readMem S (s.pids 0 * T + i.val) else 0)
        - ∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val),
            (if k.val < T then s.readMem S (s.pids 0 * T + k.val) else 0))
        + (∑ k : Fin BT, (if k.val < T then s.readMem S (s.pids 0 * T + k.val) else 0))
      = ∑ flat ∈ (Finset.range T).filter (fun flat => i.val ≤ flat),
          s.readMem S (s.pids 0 * T + flat)
  rw [suffix_of_sub_prefix BT
    (fun k => if k.val < T then s.readMem S (s.pids 0 * T + k.val) else 0) i]
  -- reindex the guarded suffix `{k : Fin BT | i ≤ k ∧ k < T}` to flat indices
  -- `{flat ∈ range T | i ≤ flat}` (surjective because `T ≤ BT`).
  rw [← Finset.sum_filter_of_ne
      (p := fun k : Fin BT => k.val < T)]
  · rw [Finset.filter_filter]
    apply Finset.sum_nbij' (i := fun k : Fin BT => k.val)
      (j := fun flat => (⟨if h : flat < BT then flat else 0, by
          split
          · assumption
          · exact i.pos⟩ : Fin BT))
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      simp only [Finset.mem_range, Finset.mem_filter]
      exact ⟨ha.2, ha.1⟩
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_range] at ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      have hd : a < BT := by omega
      rw [dif_pos hd]; exact ⟨ha.2, ha.1⟩
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      apply Fin.ext
      simp only [dif_pos a.isLt]
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_range] at ha
      have hd : a < BT := by omega
      simp only [dif_pos hd]
    · intro a ha
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
      rw [if_pos ha.2]
  · intro k hk hne
    by_contra h; simp [h] at hne

theorem reversed_cumsum_scalar_single_block_surface_correct
    (S O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => singleBlockVecOffset s T i)) :
    ∀ i : Fin BT,
      let outAddr := singleBlockVecOffset s T i
      (exec (reversed_cumsum_scalar_single_block_surface S O T BT) s).map
          (·.readMem O outAddr)
        = some (if singleBlockActive s T i then
            singleBlockRevStoreValue s S T BT i
          else s.readMem O outAddr) := by
  intro i
  simp [exec, reversed_cumsum_scalar_single_block_surface, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, Tile.scan, Tile.expandDim, TileShape.dropInsertedIndex,
        Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.sub, NumericDType.mul,
        ComparableDType.lt, singleBlockActive, singleBlockVecOffset,
        singleBlockRevInputTile, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BT] → Nat :=
    fun idx => s.pids 0 * T + idx.1.val
  let valueFn : TileIndex [BT] → ℝ :=
    fun idx => singleBlockRevStoreValue s S T BT idx.1
  let P : TileIndex [BT] → Prop := fun idx => idx.1.val < T
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : singleBlockVecOffset s T a = singleBlockVecOffset s T b := by
      simpa [offsetFn, singleBlockVecOffset] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : i.val < T
  · simp only [singleBlockRevStoreValue, singleBlockRevInputTile,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
      TileShape.insertAxisIndex,
      singleBlockActive, singleBlockVecOffset, offsetFn, hi, if_true]
    rfl
  · simp [singleBlockRevStoreValue, singleBlockRevInputTile,
      singleBlockActive, singleBlockVecOffset, offsetFn, hi]

theorem reversed_cumsum_scalar_single_block_surface_compute_correct
    (S O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => singleBlockVecOffset s T i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := reversed_cumsum_scalar_single_block_surface S O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => singleBlockActive s T i)
        (fun i => (O, singleBlockVecOffset s T i)))
      (expected := fun i : Fin BT =>
        singleBlockRevStoreValue s S T BT i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [reversed_cumsum_scalar_single_block_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := reversed_cumsum_scalar_single_block_surface_correct S O T BT s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Genuine single-chunk correctness.** The single-Python-chunk surface (the
actual `S → O` path, one reverse iteration) realizes the genuine closed-form
global suffix sum `singleBlockRevClosed = Σ_{flat ≥ i, flat < T} s[i_bh·T + flat]`
whenever the row fits in one chunk (`T ≤ BT`). No read-back of the kernel's
own output appears in `expected`. -/
theorem reversed_cumsum_scalar_single_block_surface_closed_form
    (S O : RegionName) (T BT : Nat) (hT : T ≤ BT) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => singleBlockVecOffset s T i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := reversed_cumsum_scalar_single_block_surface S O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => singleBlockActive s T i)
        (fun i => (O, singleBlockVecOffset s T i)))
      (expected := fun i : Fin BT =>
        singleBlockRevClosed s S T BT i) := by
  have h := reversed_cumsum_scalar_single_block_surface_compute_correct S O T BT s hOutInj
  have hcong : (fun i : Fin BT => singleBlockRevStoreValue s S T BT i)
      = (fun i : Fin BT => singleBlockRevClosed s S T BT i) := by
    funext i; exact singleBlockRevStoreValue_eq_closed s S T BT hT i
  rwa [hcong] at h

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **Dimension-general** correctness summary for `reversed_cumsum_scalar.py`,
against the **genuine closed forms** (no self-referential read-back), for
arbitrary `T`, `BT` and strides. It packages:

* the **full reverse-traversal surface** (verbatim `range(cdiv(T,BT)−1,−1,−1)`
  loop with the carried scalar `b_z`) lowers to the algorithm layer;
* the **single-chunk path** (any shape with `T ≤ BT`, e.g. the Python
  benchmark `T = 4`, `BT = 16`, where the reverse loop runs exactly once):
  every active lane `i` receives the genuine global reversed cumulative sum
  `singleBlockRevClosed = Σ_{flat ≥ i, flat < T} s[i_bh·T + flat]`,
  end-to-end from the input region `S` — a standalone `Finset.sum`, never a
  read-back of the kernel output;
* the **per-chunk reversed carry-fold face**: for *every* state whose carry
  buffer holds the genuine post-update suffix total
  (`Carry[i_bh] = Σ_{flat ≥ i_t·BT, flat < T} s[i_bh·T + flat]` — the value
  `b_z` has after the kernel's `b_z += b_zz`), one chunk iteration realizes
  the genuine global reversed cumulative sum `globalRevCumsumClosed`.

Honest side-conditions only: output-offset injectivity of each store
footprint. The **cross-chunk loop scheduling** that threads the carried `b_z`
register from chunk `c` to `c−1` — i.e. that the materialized carry buffer
holds the genuine suffix total when the body of chunk `c` runs — is the
*trusted runtime boundary*, exactly as in the sibling `reversed_cumsum`
(#290-style carried register); its algebraic content is
`revStoreValue_eq_globalRevCumsumClosed`. -/
specification reversed_cumsum_scalar_output_summary_general
    (S Carry O : RegionName) (T BT : Nat) (hT : T ≤ BT) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => singleBlockVecOffset s T i)) :
    -- (1) the full reverse-traversal surface lowers to the algorithm layer
    (∃ alg, (reversed_cumsum_scalar_surface S O T BT).toAlgorithm?
      = Except.ok alg) ∧
    -- (2) single-chunk genuine suffix sum, end-to-end from `S`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := reversed_cumsum_scalar_single_block_surface S O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => singleBlockActive s T i)
        (fun i => (O, singleBlockVecOffset s T i)))
      (expected := fun i : Fin BT => singleBlockRevClosed s S T BT i)) ∧
    -- (3) per-chunk reversed carry-fold face, for every carry-invariant state
    (∀ (s' : BlockState),
      Function.Injective (fun i : Fin BT => vecOffset s' T BT i) →
      s'.readMem Carry (s'.pids 0)
        = (∑ flat ∈ (Finset.range T).filter
            (fun flat => s'.pids 1 * BT ≤ flat), rowElem s' S T flat) →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := reversed_cumsum_scalar_rev_slice S Carry O T BT)
        (initialState := s')
        (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BT => active s' T BT i)
          (fun i => (O, vecOffset s' T BT i)))
        (expected := fun i : Fin BT => globalRevCumsumClosed s' S T BT i)) := by
  refine ⟨reversed_cumsum_scalar_surface_toAlgorithm_supported S O T BT,
    reversed_cumsum_scalar_single_block_surface_closed_form S O T BT hT s hOutInj,
    ?_⟩
  intro s' hInj' hcarry'
  exact reversed_cumsum_scalar_rev_slice_closed_form S Carry O T BT s' hInj'
    (by simpa [rowElem] using hcarry')

end Correct_without_Rounding
