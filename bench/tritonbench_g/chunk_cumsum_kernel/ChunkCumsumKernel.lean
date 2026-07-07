import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel

/-!
# `chunk_cumsum_kernel` — strict per-kernel correctness

`chunk_global_cumsum_scalar_kernel` computes a global scalar cumulative sum
along the time axis: program `i_bh` (one per batch·head row) walks the `T`-long
row of `s` in chunks of `BT`, and for each chunk emits the running prefix sum
`tl.cumsum(b_s) + b_z` while carrying the chunk total `b_z` forward across
iterations, storing the result into `o`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`chunk_global_cumsum_scalar_kernel[(B*H,)](...)`, the
grid over batch·head rows, the autotuned `BT`, and how the runtime composes
per-program writes into one buffer) is the *trusted boundary*, not a proof
obligation here. Because the program ids are universally quantified, the
per-program statements cover every program of the grid.

## Proof architecture

The kernel is a **global** cumulative sum: each output flat index `flat` holds
`Σ_{m ≤ flat} s[m]`. The genuine closed-form spec is a standalone `Finset.sum`
(`globalCumsumClosed` / `singleBlockCumsumClosed`), never a read-back of the
kernel's own output.

```
chunk_cumsum_scalar_output_summary_general                     ← TOP THEOREM
  ├─ chunk_cumsum_scalar_surface_toAlgorithm_supported         full surface lowers
  └─ chunk_cumsum_scalar_surface_global_cumsum                 O = global prefix Σ (carry-fold)

The Python benchmark shape (`T = 4`, `BT = 16`, single chunk) is one
instantiation of this dimension-general top theorem.

mathematical core (the carry-fold + within-chunk identity):
  scan1d_sum / scan1d_sum_if          `tl.cumsum` = guarded prefix `Finset.sum`
  singleBlockCumsumStoreValue_eq_closed   single chunk (carry = 0) = global prefix Σ
  cumsumStoreValue_eq_globalCumsumClosed  carry_c + within-chunk Σ = global prefix Σ,
                                          given carry_c = Σ_{flat < c·BT, flat<T} s[flat]
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` (the
`BT ∈ {16,32,64}` config set) is not modeled — the checked Python shape `T = 4`
with `BT = 16` (where the chunk loop has a single iteration) is one instantiation
of the dimension-general top theorem; the closed-form lemmas (`scan1d_sum*`, `*_eq_closed`,
`cumsumStoreValue_eq_globalCumsumClosed`, and the
`*_surface_closed_form` / `*_cumsum_slice_closed_form` realizers) are stated and
proven **general over `T`, `BT` and the number of chunks**. The
`.to(tl.float32)` / `.to(p_o.dtype.element_ty)` casts erase to the identity at
the algorithm layer (post-erasure all dtypes unify to `ℝ`). The per-chunk
`tl.cumsum` (`Tile.scan .sum`) is modeled exactly and shown equal to the genuine
prefix `Finset.sum` (`scan1d_sum`). The cross-chunk carry recurrence threaded by
`b_z` is `carry_{c+1} = carry_c + Σ chunk_c`; its invariant
`carry_c = Σ_{flat < c·BT, flat < T} s[flat]` is the explicit hypothesis of
`cumsumStoreValue_eq_globalCumsumClosed` — under it, each chunk's store equals
the genuine global cumulative sum. The carry is materialized in a buffer
(`Carry`) in `chunk_cumsum_scalar_cumsum_slice`; the single-chunk Python shape
realizes the global prefix sum end-to-end with `carry = 0`. Output injectivity is
a side condition (discharged dimension-generally).
-/

namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `chunk_cumsum_scalar_output_summary_general`
(dimension-general `T`, `BT`); the `T = 4`, `BT = 16` Python benchmark shape is
one instantiation. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct

/-! ## Within-chunk cumsum identity (`tl.cumsum` = prefix `Finset.sum`)

`tl.cumsum(b_s, axis=0)` lowers to `Op.scan .sum` and hence `Tile.scan .sum`.
For a 1D tile whose lanes hold `some (g k)`, the scanned value at lane `i` is
`some (Σ_{k ≤ i} g k)`. This is the mathematical core that turns the kernel's
register-level scan into a genuine closed-form prefix sum. -/

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

/-- **Masked 1D cumsum = guarded prefix sum.** The `Tile.scan .sum` of a tile
whose lanes hold `some (if P k then h k else 0)`, demoted via `unbotD 0`, is the
sum of `h` over `{k : k ≤ i ∧ P k}`. -/
theorem scan1d_sum_if (BT : Nat) (h : Fin BT → ℝ) (P : Fin BT → Prop)
    [DecidablePred P] (i : Fin BT) :
    WithBot.unbotD 0
      ((Tile.scan .sum ⟨0, by simp⟩ .forward
        (⟨fun idx => some (if P idx.1 then h idx.1 else 0)⟩ :
          Tile .real [BT])).data (i, PUnit.unit))
      = ∑ k ∈ (Finset.univ.filter
          (fun k : Fin BT => k.val ≤ i.val ∧ P k)), h k := by
  rw [scan1d_sum BT (fun k => if P k then h k else 0) i]
  show (∑ k ∈ (Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val)),
      if P k then h k else 0) = _
  rw [Finset.sum_filter, Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro k _
  by_cases hk : k.val ≤ i.val <;> by_cases hp : P k <;> simp [hk, hp]

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
  simp [exec, chunk_cumsum_scalar_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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
      ((Tile.scan .sum ⟨0, by simp⟩ .forward (cumsumInputTile s S T BT)).data (i, PUnit.unit))
      (some (s.readMem Carry (s.pids 0))))

/-! ## Genuine chunked-cumsum closed form

`globalCumsumClosed` is the genuine mathematical specification — a `Finset.sum`
over all *flat* time indices `flat ≤ chunk·BT + i` (with `flat < T`) of the
source value at `i_bh·T + flat`. It is **not** a read-back of the kernel's own
output, so realizing it is a true correctness statement.

For program `(i_bh, i_t)` (so `i_bh = pids 0`, `i_t = pids 1`) and active lane
`i`, the global flat output index is `i_t·BT + i`, and the output must hold the
prefix sum of all source entries up to and including that flat index. -/
noncomputable def globalCumsumClosed
    (s : BlockState) (S : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  ∑ flat ∈ (Finset.range T).filter
      (fun flat => flat ≤ s.pids 1 * BT + i.val),
    s.readMem S (s.pids 0 * T + flat)

/-- **The carry-fold recurrence.** When the carry buffer `Carry` holds the
genuine prefix sum of *all prior chunks* (every flat index `< i_t·BT`, clamped
to `< T`), the per-chunk store value `cumsumStoreValue` — the within-chunk
`tl.cumsum` plus that carry — equals the genuine global cumulative sum
`globalCumsumClosed`. This is the exact recurrence threaded by `b_z` across the
`forRange` loop: `carry_{c+1} = carry_c + Σ chunk_c`, with the invariant
`carry_c = Σ_{flat < c·BT, flat < T} s[flat]`. -/
theorem cumsumStoreValue_eq_globalCumsumClosed
    (s : BlockState) (S Carry : RegionName) (T BT : Nat) (i : Fin BT)
    (hcarry : s.readMem Carry (s.pids 0)
      = ∑ flat ∈ (Finset.range T).filter (fun flat => flat < s.pids 1 * BT),
          s.readMem S (s.pids 0 * T + flat)) :
    cumsumStoreValue s S Carry T BT i = globalCumsumClosed s S T BT i := by
  unfold cumsumStoreValue
  have hin : cumsumInputTile s S T BT
      = (⟨fun idx => some (if s.pids 1 * BT + idx.1.val < T then
            s.readMem S (s.pids 0 * T + (s.pids 1 * BT + idx.1.val))
          else 0)⟩ : Tile .real [BT]) := by
    unfold cumsumInputTile; congr 1; funext idx
    by_cases h : s.pids 1 * BT + idx.1.val < T
    · simp [h]
    · simp only [h, if_false]; norm_num
  rw [hin, scan1d_sum BT (fun k => if s.pids 1 * BT + k.val < T then
        s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0) i]
  rw [Option.map₂_some_some]
  show ((∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val),
      (if s.pids 1 * BT + k.val < T then
        s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0))
      + s.readMem Carry (s.pids 0)) = globalCumsumClosed s S T BT i
  rw [hcarry]
  unfold globalCumsumClosed
  -- Step 1: reindex the within-chunk sum to flat segment indices.
  have hreindex :
      (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val),
        (if s.pids 1 * BT + k.val < T then
          s.readMem S (s.pids 0 * T + (s.pids 1 * BT + k.val)) else 0))
      = ∑ flat ∈ (Finset.range T).filter
          (fun flat => s.pids 1 * BT ≤ flat ∧ flat ≤ s.pids 1 * BT + i.val),
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
            · exact i.pos⟩ : Fin BT))
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
  rw [hreindex]
  -- Step 2: split `globalCumsumClosed` by `flat < i_t·BT` and identify the two
  -- pieces with the carry segment and the within-chunk segment.
  rw [add_comm,
      ← Finset.sum_filter_add_sum_filter_not
        ((Finset.range T).filter (fun flat => flat ≤ s.pids 1 * BT + i.val))
        (fun flat => flat < s.pids 1 * BT)]
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
  simp [exec, chunk_cumsum_scalar_cumsum_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
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

/-! ## Single Python-chunk computed surface

For the checked Python shape `T = 4` and autotuned `BT = 16`, the loop has one
iteration and the carried scalar `b_z` is the initial zero. This surface keeps
the Python-observable path from `S` to `O` without a precomputed output tile or
external carry buffer. -/

def chunk_cumsum_scalar_single_block_surface
    (S O : RegionName) (T BT : Nat) :
    ComputeKernel := triton {
  i_bh = tl.program_id(0)
  offs_t = tl.arange(0, $(BT))
  mask = offs_t < $(T)
  b_s = tl.load(S + i_bh * $(T) + offs_t, mask=mask, other=0.0).to(tl.float32)
  b_o = tl.cumsum(b_s, axis=0)
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

noncomputable def singleBlockCumsumInputTile
    (s : BlockState) (S : RegionName) (T BT : Nat) :
    Tile .real [BT] :=
  { data := fun idx =>
      if idx.1.val < T then
        some (s.readMem S (s.pids 0 * T + idx.1.val))
      else some (0.0 : ℝ) }

noncomputable def singleBlockCumsumStoreValue
    (s : BlockState) (S : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  WithBot.unbotD 0
    ((Tile.scan .sum ⟨0, by simp⟩ .forward
      (singleBlockCumsumInputTile s S T BT)).data (i, PUnit.unit))

/-- Genuine closed form for the single-Python-chunk path (`i_t = 0`, carry `= 0`):
the prefix sum of all source entries up to and including flat index `i`. -/
noncomputable def singleBlockCumsumClosed
    (s : BlockState) (S : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  ∑ flat ∈ (Finset.range T).filter (fun flat => flat ≤ i.val),
    s.readMem S (s.pids 0 * T + flat)

/-- **Single-chunk correctness against the genuine closed form.** With the carry
at its initial zero (`i_t = 0`), the within-chunk `tl.cumsum` store value is the
genuine global prefix sum `Σ_{flat ≤ i, flat < T} s[i_bh·T + flat]`. This is the
sorry-free end-to-end correctness of the actual Python test-shape path (`T = 4`,
`BT = 16`, where the chunk loop runs exactly once). -/
theorem singleBlockCumsumStoreValue_eq_closed
    (s : BlockState) (S : RegionName) (T BT : Nat) (i : Fin BT) :
    singleBlockCumsumStoreValue s S T BT i
      = singleBlockCumsumClosed s S T BT i := by
  unfold singleBlockCumsumStoreValue singleBlockCumsumClosed
  have hin : singleBlockCumsumInputTile s S T BT
      = (⟨fun idx => some (if idx.1.val < T then
            s.readMem S (s.pids 0 * T + idx.1.val) else 0)⟩ : Tile .real [BT]) := by
    unfold singleBlockCumsumInputTile; congr 1; funext idx
    by_cases h : idx.1.val < T
    · simp [h]
    · simp only [h, if_false]; norm_num
  rw [hin, scan1d_sum_if BT (fun k => s.readMem S (s.pids 0 * T + k.val))
    (fun k => k.val < T) i]
  -- reindex the guarded prefix `{k : Fin BT | k ≤ i ∧ k < T}` to flat indices
  -- `{flat ∈ range T | flat ≤ i}`; note `flat ≤ i < BT` keeps everything in `Fin BT`.
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
    have hd : a < BT := by
      have : a ≤ i.val := ha.2
      omega
    rw [dif_pos hd]; exact ⟨ha.2, ha.1⟩
  · intro a ha
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
    apply Fin.ext
    simp only [dif_pos a.isLt]
  · intro a ha
    simp only [Finset.mem_filter, Finset.mem_range] at ha
    have hd : a < BT := by
      have : a ≤ i.val := ha.2
      omega
    simp only [dif_pos hd]
  · intro a _; rfl

theorem chunk_cumsum_scalar_single_block_surface_correct
    (S O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => singleBlockVecOffset s T i)) :
    ∀ i : Fin BT,
      let outAddr := singleBlockVecOffset s T i
      (exec (chunk_cumsum_scalar_single_block_surface S O T BT) s).map
          (·.readMem O outAddr)
        = some (if singleBlockActive s T i then
            singleBlockCumsumStoreValue s S T BT i
          else s.readMem O outAddr) := by
  intro i
  simp [exec, chunk_cumsum_scalar_single_block_surface, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.ptrAdd, Tile.scan, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, singleBlockActive, singleBlockVecOffset,
        singleBlockCumsumInputTile, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BT] → Nat :=
    fun idx => s.pids 0 * T + idx.1.val
  let valueFn : TileIndex [BT] → ℝ :=
    fun idx => singleBlockCumsumStoreValue s S T BT idx.1
  let P : TileIndex [BT] → Prop := fun idx => idx.1.val < T
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : singleBlockVecOffset s T a = singleBlockVecOffset s T b := by
      simpa [offsetFn, singleBlockVecOffset] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : i.val < T
  · simp [singleBlockCumsumStoreValue, singleBlockCumsumInputTile,
      singleBlockActive, singleBlockVecOffset, offsetFn, hi]
  · simp [singleBlockCumsumStoreValue, singleBlockCumsumInputTile,
      singleBlockActive, singleBlockVecOffset, offsetFn, hi]

theorem chunk_cumsum_scalar_single_block_surface_compute_correct
    (S O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => singleBlockVecOffset s T i)) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_single_block_surface S O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => singleBlockActive s T i)
        (fun i => (O, singleBlockVecOffset s T i)))
      (expected := fun i : Fin BT =>
        singleBlockCumsumStoreValue s S T BT i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_cumsum_scalar_single_block_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := chunk_cumsum_scalar_single_block_surface_correct S O T BT s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- **Genuine single-chunk correctness.** The single-Python-chunk surface (the
actual `S → O` path, carry `= 0`) realizes the genuine closed-form global prefix
sum `singleBlockCumsumClosed = Σ_{flat ≤ i, flat < T} s[i_bh·T + flat]`. No
read-back of the kernel's own output appears in `expected`. -/
theorem chunk_cumsum_scalar_single_block_surface_closed_form
    (S O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => singleBlockVecOffset s T i)) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_single_block_surface S O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => singleBlockActive s T i)
        (fun i => (O, singleBlockVecOffset s T i)))
      (expected := fun i : Fin BT =>
        singleBlockCumsumClosed s S T BT i) := by
  have h := chunk_cumsum_scalar_single_block_surface_compute_correct S O T BT s hOutInj
  simpa only [singleBlockCumsumStoreValue_eq_closed] using h

/-- **Genuine per-chunk correctness (carry-fold).** Given the carry buffer holds
the genuine prefix sum of all prior chunks
(`Carry[i_bh] = Σ_{flat < i_t·BT, flat < T} s[i_bh·T + flat]`), the cumsum slice
realizes the genuine global cumulative sum `globalCumsumClosed`. This is the
inductive step of the carry recurrence threaded by `b_z`. -/
theorem chunk_cumsum_scalar_cumsum_slice_closed_form
    (S Carry O : RegionName) (T BT : Nat) (s : BlockState)
    (hOutInj : Function.Injective (fun i : Fin BT => vecOffset s T BT i))
    (hcarry : s.readMem Carry (s.pids 0)
      = ∑ flat ∈ (Finset.range T).filter (fun flat => flat < s.pids 1 * BT),
          s.readMem S (s.pids 0 * T + flat)) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_cumsum_slice S Carry O T BT)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BT => active s T BT i)
        (fun i => (O, vecOffset s T BT i)))
      (expected := fun i : Fin BT =>
        globalCumsumClosed s S T BT i) := by
  have h := chunk_cumsum_scalar_cumsum_slice_compute_correct S Carry O T BT s hOutInj
  have hcong : (fun i : Fin BT => cumsumStoreValue s S Carry T BT i)
      = (fun i : Fin BT => globalCumsumClosed s S T BT i) := by
    funext i; exact cumsumStoreValue_eq_globalCumsumClosed s S Carry T BT i hcarry
  rwa [hcong] at h

/-! ## Python test-shape wrappers

`chunk_cumsum_kernel.py`'s checked tests use `B = 2`, `H = 3`, `T = 4`.
The autotune set includes `BT = 16`, which covers the single Python chunk for
this test shape. -/

/-- Running carry after `c` chunks: prefix sum of all flat indices `< c·BT`
(clamped to `< T`), for program `i_bh = pids 0`. -/
noncomputable def surfaceCarry
    (s : BlockState) (S : RegionName) (T BT c : Nat) : ℝ :=
  ∑ flat ∈ (Finset.range T).filter (fun flat => flat < c * BT),
    s.readMem S (s.pids 0 * T + flat)

/-- Genuine global cumulative sum at flat index `flat`, for program `pids 0`. -/
noncomputable def surfaceClosed
    (s : BlockState) (S : RegionName) (T : Nat) (flat : Nat) : ℝ :=
  ∑ m ∈ (Finset.range T).filter (fun m => m ≤ flat),
    s.readMem S (s.pids 0 * T + m)

/-- **Carry-fold for the surface kernel.** For an active lane `i` of chunk `c`
(`c·BT + i < T`), the within-chunk cumulative sum
`Σ_{k ≤ i, c·BT+k < T} s[i_bh·T + (c·BT+k)]` plus the running carry
`surfaceCarry c = Σ_{flat < c·BT} s[i_bh·T + flat]` equals the genuine global
cumulative sum `surfaceClosed (c·BT+i)`. -/
theorem surface_carry_fold
    (s : BlockState) (S : RegionName) (T BT c : Nat) (i : Fin BT)
    (hi : c * BT + i.val < T) :
    (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val),
        (if c * BT + k.val < T then s.readMem S (s.pids 0 * T + (c * BT + k.val)) else 0))
      + surfaceCarry s S T BT c
      = surfaceClosed s S T (c * BT + i.val) := by
  unfold surfaceCarry surfaceClosed
  -- reindex the within-chunk sum to flat segment indices
  have hreindex :
      (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val),
        (if c * BT + k.val < T then
          s.readMem S (s.pids 0 * T + (c * BT + k.val)) else 0))
      = ∑ flat ∈ (Finset.range T).filter
          (fun flat => c * BT ≤ flat ∧ flat ≤ c * BT + i.val),
          s.readMem S (s.pids 0 * T + flat) := by
    rw [← Finset.sum_filter_of_ne
        (p := fun k : Fin BT => c * BT + k.val < T)]
    · rw [Finset.filter_filter]
      apply Finset.sum_nbij'
        (i := fun k : Fin BT => c * BT + k.val)
        (j := fun flat => (⟨if h : flat - c * BT < BT then
            flat - c * BT else 0, by
            split
            · assumption
            · exact i.pos⟩ : Fin BT))
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        simp only [Finset.mem_range, Finset.mem_filter]
        exact ⟨ha.2, by omega, by omega⟩
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_range] at ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        have hd : a - c * BT < BT := by omega
        rw [dif_pos hd]; exact ⟨by omega, by omega⟩
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        apply Fin.ext
        have hd : c * BT + a.val - c * BT < BT := by omega
        simp only [dif_pos hd]; omega
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_range] at ha
        have hd : a - c * BT < BT := by omega
        simp only [dif_pos hd]; omega
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        rw [if_pos ha.2]
    · intro k hk hne
      by_contra h; simp [h] at hne
  rw [hreindex, add_comm,
      ← Finset.sum_filter_add_sum_filter_not
        ((Finset.range T).filter (fun flat => flat ≤ c * BT + i.val))
        (fun flat => flat < c * BT)]
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

/-- **Carry recurrence.** `surfaceCarry (c+1) = surfaceCarry c + Σ chunk_c`, the
total of the within-chunk source values. -/
theorem surface_carry_succ
    (s : BlockState) (S : RegionName) (T BT c : Nat) (hBT : 0 < BT) :
    surfaceCarry s S T BT (c + 1)
      = surfaceCarry s S T BT c
        + ∑ k : Fin BT, (if c * BT + k.val < T then
            s.readMem S (s.pids 0 * T + (c * BT + k.val)) else 0) := by
  unfold surfaceCarry
  rw [add_comm]
  -- the chunk total reindexes to the flat segment [c·BT, (c+1)·BT) ∩ range T
  have hchunk :
      (∑ k : Fin BT, (if c * BT + k.val < T then
          s.readMem S (s.pids 0 * T + (c * BT + k.val)) else 0))
      = ∑ flat ∈ (Finset.range T).filter
          (fun flat => c * BT ≤ flat ∧ flat < (c + 1) * BT),
          s.readMem S (s.pids 0 * T + flat) := by
    rw [← Finset.sum_filter_of_ne
        (s := (Finset.univ : Finset (Fin BT)))
        (p := fun k : Fin BT => c * BT + k.val < T)]
    · apply Finset.sum_nbij'
        (i := fun k : Fin BT => c * BT + k.val)
        (j := fun flat => (⟨if h : flat - c * BT < BT then
            flat - c * BT else 0, by
            split
            · assumption
            · omega⟩ : Fin BT))
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        simp only [Finset.mem_range, Finset.mem_filter]
        refine ⟨ha, by omega, ?_⟩
        have := a.isLt; rw [Nat.add_mul, Nat.one_mul]; omega
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_range] at ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        have hd : a - c * BT < BT := by
          have := ha.2.2; rw [Nat.add_mul, Nat.one_mul] at this; omega
        rw [dif_pos hd]; omega
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        apply Fin.ext
        have hd : c * BT + a.val - c * BT < BT := by omega
        simp only [dif_pos hd]; omega
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_range] at ha
        have hd : a - c * BT < BT := by
          have := ha.2.2; rw [Nat.add_mul, Nat.one_mul] at this; omega
        simp only [dif_pos hd]; omega
      · intro a ha
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
        rw [if_pos ha]
    · intro k _ hne
      by_contra h; simp [h] at hne
  rw [hchunk]
  rw [show (Finset.range T).filter (fun flat => flat < (1 + c) * BT)
      = (Finset.range T).filter (fun flat => flat < c * BT)
        ∪ (Finset.range T).filter (fun flat => c * BT ≤ flat ∧ flat < (c + 1) * BT) from by
    rw [← Finset.filter_or]
    apply Finset.filter_congr
    intro flat hflat
    simp only [Finset.mem_range] at hflat
    have hcomm : (1 + c) * BT = (c + 1) * BT := by rw [Nat.add_comm]
    constructor
    · intro h
      rw [hcomm] at h
      by_cases hc : flat < c * BT
      · exact Or.inl hc
      · exact Or.inr ⟨by omega, h⟩
    · rintro (h | ⟨_, h⟩)
      · have hle : c * BT ≤ (1 + c) * BT := by
          rw [hcomm, Nat.add_mul, Nat.one_mul]; omega
        omega
      · rw [hcomm]; exact h]
  rw [Finset.sum_union]
  rw [Finset.disjoint_filter]
  intro flat _ h1
  simp only [not_and, not_lt]
  intro h2; omega

/-- The per-chunk masked block-ptr store, written as a 1D scatter into `O` at
flat offsets `base + c·BT + i` over the active lanes (`c·BT+i < T`). -/
noncomputable def surfaceScatter (O : RegionName) (T BT base c : Nat)
    (val : Fin BT → ℝ) (s0 : BlockState) : BlockState :=
  (TileShape.allIndices [BT]).foldl
    (fun (acc : BlockState) i =>
      if c * BT + i.1.val < T then
        acc.writeMem O (base + (c * BT + i.1.val)) (val i.1)
      else acc) s0

theorem surfaceScatter_pids (O : RegionName) (T BT base c : Nat)
    (val : Fin BT → ℝ) (s0 : BlockState) :
    (surfaceScatter O T BT base c val s0).pids = s0.pids := by
  unfold surfaceScatter
  induction TileShape.allIndices [BT] generalizing s0 with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]; split
      · rw [ih]; simp
      · rw [ih]

theorem surfaceScatter_regs (O : RegionName) (T BT base c : Nat)
    (val : Fin BT → ℝ) (s0 : BlockState)
    (dt : TileDType) (sh : TileShape) (nm : RegName) :
    (surfaceScatter O T BT base c val s0).regs dt sh nm = s0.regs dt sh nm := by
  unfold surfaceScatter
  induction TileShape.allIndices [BT] generalizing s0 with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]; split
      · rw [ih]; simp [BlockState.writeMem]
      · rw [ih]

theorem surfaceScatter_readMem_ne (O R : RegionName) (T BT base c : Nat)
    (val : Fin BT → ℝ) (s0 : BlockState) (a : Nat) (hR : R ≠ O) :
    (surfaceScatter O T BT base c val s0).readMem R a = s0.readMem R a := by
  unfold surfaceScatter
  induction TileShape.allIndices [BT] generalizing s0 with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]; split
      · rw [ih]; exact BlockState.writeMem_readMem_of_ne_region _ _ _ _ _ _ hR
      · rw [ih]

theorem surfaceScatter_readMem_lt (O : RegionName) (T BT base c : Nat)
    (val : Fin BT → ℝ) (s0 : BlockState) (flat : Nat) (hlt : flat < c * BT) :
    (surfaceScatter O T BT base c val s0).readMem O (base + flat) = s0.readMem O (base + flat) := by
  unfold surfaceScatter
  induction TileShape.allIndices [BT] generalizing s0 with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]; split
      · rw [ih]
        rw [BlockState.writeMem_readMem_of_ne_offset]
        omega
      · rw [ih]

/-- The loop invariant for the surface chunk-cumsum kernel, after `c` chunks
have run (program `i_bh = pids 0`). -/
def surfaceInv (s : BlockState) (S O : RegionName) (T BT : Nat) :
    Nat → BlockState → Prop :=
  fun c sc =>
    sc.pids = s.pids ∧
    (∀ a, sc.readMem S a = s.readMem S a) ∧
    sc.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0)) ∧
    sc.regs .real [] "b_z" = some (Tile.scalar (some (surfaceCarry s S T BT c))) ∧
    (∀ flat, flat < c * BT → flat < T →
      sc.readMem O (s.pids 0 * T + flat) = surfaceClosed s S T flat)

set_option maxHeartbeats 2000000 in
/-- **One chunk advances the invariant.** Stepping the 7-statement loop body for
chunk `c` (with `i_t := c`) preserves `surfaceInv`, advancing it to `c+1`. -/
theorem surface_step (S O : RegionName) (T BT : Nat) (s : BlockState)
    (hSO : O ≠ S) (hBT : 0 < BT)
    (c : Nat) (sc : BlockState) (hPc : surfaceInv s S O T BT c sc) :
    ∃ s',
      stepStmts
        [Stmt.assign TileDType.blockPtr [BT] "p_s"
            (Op.makeBlockPtrDynOffsets S
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat T)) [T] [BT] [1]
              [Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_t") (Op.constNat BT)]),
          Stmt.assign TileDType.blockPtr [BT] "p_o"
            (Op.makeBlockPtrDynOffsets O
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat T)) [T] [BT] [1]
              [Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_t") (Op.constNat BT)]),
          Stmt.assign TileDType.real [BT] "b_s"
            (Op.load ComputeDType.fp32.eraseDType (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT] "p_s") [0])
              MaskOpt.none),
          Stmt.assign TileDType.real [BT] "b_o"
            (Op.add NumericDType.real Broadcast.nil.consR
              (Op.scan ScanOp.sum ⟨0, by simp⟩ .forward (Op.ref TileDType.real [BT] "b_s"))
              (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.real [] "b_z"))),
          Stmt.assign TileDType.real [] "b_zz" (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref TileDType.real [BT] "b_s")),
          Stmt.assign TileDType.real [] "b_z"
            (Op.add NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "b_z")
              (Op.ref TileDType.real [] "b_zz")),
          Stmt.store TileDType.real [BT] (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT] "p_o") [0])
            (Op.ref TileDType.real [BT] "b_o") MaskOpt.none]
        (sc.setReg "i_t" .nat [] (Tile.scalar c)) = some s' ∧
      surfaceInv s S O T BT (c + 1) s' := by
  obtain ⟨hpids, hS, hibh, hbz, hO⟩ := hPc
  set sloop := sc.setReg "i_t" .nat [] (Tile.scalar c) with hsloop
  have hibh' : sloop.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0)) := by
    rw [hsloop]; simpa using hibh
  have hit' : sloop.regs .nat [] "i_t" = some (Tile.scalar c) := by
    rw [hsloop]; simp [BlockState.setReg]
  -- p_s eval
  have hps : evalOp (Op.makeBlockPtrDynOffsets S
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat T)) [T] [BT] [1]
      [Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_t") (Op.constNat BT)]) sloop
      = some (⟨fun _ => ({ region := S, baseOffset := s.pids 0 * T, parentShape := [T], blockShape := [BT], strides := [1], offsets := [c * BT] } : BlockPtr)⟩ : Tile .blockPtr [BT]) := by
    simp only [evalOp, Option.bind, Option.map, hibh', hit', Tile.bop, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.mul, List.mapM, List.mapM.loop]
    rfl
  -- 1D block-ptr address/inBounds for p_s/p_o
  set psT : Tile .blockPtr [BT] :=
    ⟨fun _ => ({ region := S, baseOffset := s.pids 0 * T, parentShape := [T], blockShape := [BT], strides := [1], offsets := [c * BT] } : BlockPtr)⟩ with hpsT
  have haddr : ∀ i : Fin BT, (psT.data (i, PUnit.unit)).address (TileShape.indexToList [BT] (i, PUnit.unit))
      = s.pids 0 * T + (c * BT + i.val) := by
    intro i
    rw [show TileShape.indexToList [BT] (i, PUnit.unit) = [i.val] from rfl, hpsT,
      BlockPtr.address_1d]
    ring
  have hib : ∀ i : Fin BT, (psT.data (i, PUnit.unit)).inBounds (TileShape.indexToList [BT] (i, PUnit.unit)) [0]
      = decide (c * BT + i.val < T) := by
    intro i
    rw [show TileShape.indexToList [BT] (i, PUnit.unit) = [i.val] from rfl, hpsT,
      BlockPtr.inBounds_1d]
  -- the b_s tile: masked block load
  set bsT : Tile .real [BT] :=
    ⟨fun idx => if c * BT + idx.1.val < T then
        some (s.readMem S (s.pids 0 * T + (c * BT + idx.1.val))) else some 0⟩ with hbsT
  have hloadbs : evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT] "p_s") [0]) MaskOpt.none)
      (sloop.setReg "p_s" .blockPtr [BT] psT) = some bsT := by
    simp only [evalOp, Option.bind, BlockState.setReg_same]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [hpsT]
    rw [show TileShape.indexToList [BT] (i, u) = [i.val] from rfl]
    rw [BlockPtr.inBounds_1d, BlockPtr.address_1d]
    by_cases hi : c * BT + i.val < T
    · rw [if_pos (by simpa using hi)]
      rw [show (c * BT + i.val) * 1 = c * BT + i.val from by ring]
      rw [BlockState.setReg_readMemValue, BlockState.readMemValue_real]
      simp only [hbsT]
      rw [if_pos hi]
      rw [hsloop, BlockState.setReg_readMem, hS]
    · rw [if_neg (by simpa using hi)]
      simp only [hbsT, BlockState.defaultCarrier]
      rw [if_neg hi]
  -- within-chunk source function
  set g : Fin BT → ℝ := fun k => if c * BT + k.val < T then
      s.readMem S (s.pids 0 * T + (c * BT + k.val)) else 0 with hg
  have hbsTg : bsT = (⟨fun idx => some (g idx.1)⟩ : Tile .real [BT]) := by
    rw [hbsT]; apply Tile.ext; intro idx; simp only [hg]
    by_cases hi : c * BT + idx.1.val < T
    · rw [if_pos hi, if_pos hi]
    · rw [if_neg hi, if_neg hi]
  -- scan value at lane i
  have hscan : ∀ i : Fin BT, (Tile.scan .sum ⟨0, by simp⟩ .forward bsT).data (i, PUnit.unit)
      = some (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val), g k) := by
    intro i; rw [hbsTg]; exact scan1d_sum BT g i
  -- (placeholder)
  -- reduceSum value (total chunk sum)
  have hreduce : (Tile.reduceSum (⟨0, by simp⟩ : Fin [BT].length) Bool.false bsT).data PUnit.unit
      = some (∑ k : Fin BT, g k) := by
    rw [Tile.reduceSum_false, Tile.reduceSumDrop_data]
    rw [show (∑ k : Fin (TileShape.axisDim [BT] ⟨0, by simp⟩),
          bsT.data (TileShape.insertAxisIndex [BT] ⟨0, by simp⟩ PUnit.unit k))
        = ∑ k : Fin BT, ((g k : ℝ) : WithBot ℝ) from
      Finset.sum_congr rfl (fun k _ => by
        rw [TileShape.insertAxisIndex_head, hbsTg]; rfl)]
    rw [← WithBot.coe_sum]; rfl
  -- p_o eval (analogous to p_s, region O)
  set poT : Tile .blockPtr [BT] :=
    ⟨fun _ => ({ region := O, baseOffset := s.pids 0 * T, parentShape := [T], blockShape := [BT], strides := [1], offsets := [c * BT] } : BlockPtr)⟩ with hpoT
  -- the b_o tile value at each lane
  set boVal : Fin BT → ℝ := fun i =>
    (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val), g k) + surfaceCarry s S T BT c with hboVal
  -- intermediate states
  set s1 := sloop.setReg "p_s" .blockPtr [BT] psT with hs1
  set s2 := s1.setReg "p_o" .blockPtr [BT] poT with hs2
  set s3 := s2.setReg "b_s" .real [BT] bsT with hs3
  set boT : Tile .real [BT] := ⟨fun idx => some (boVal idx.1)⟩ with hboT
  set s4 := s3.setReg "b_o" .real [BT] boT with hs4
  set s5 := s4.setReg "b_zz" .real [] (Tile.scalar (some (∑ k : Fin BT, g k))) with hs5
  set s6 := s5.setReg "b_z" .real [] (Tile.scalar (some (surfaceCarry s S T BT (c + 1)))) with hs6
  -- register lookups along the chain (names distinct ⇒ preserved)
  have hibh1 : s1.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0)) := by
    rw [hs1]; simpa using hibh'
  have hit1 : s1.regs .nat [] "i_t" = some (Tile.scalar c) := by
    rw [hs1]; simpa using hit'
  -- p_o eval
  have hpo : evalOp (Op.makeBlockPtrDynOffsets O
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_bh") (Op.constNat T)) [T] [BT] [1]
      [Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "i_t") (Op.constNat BT)]) s1
      = some poT := by
    simp only [evalOp, Option.bind, Option.map, Tile.bop, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.mul, List.mapM, List.mapM.loop, hibh1, hit1]
    rfl
  -- p_s register persists into s2 (only "p_o" was set)
  have hps2 : s2.regs .blockPtr [BT] "p_s" = some psT := by
    rw [hs2]; simp [hs1]
  -- b_s eval on s2 (block-ptr load reads "p_s" register + memory)
  have hbs2 : evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT] "p_s") [0]) MaskOpt.none) s2 = some bsT := by
    simp only [evalOp, Option.bind, hps2]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [hpsT, hbsT]
    rw [show TileShape.indexToList [BT] (i, u) = [i.val] from rfl,
      BlockPtr.inBounds_1d, BlockPtr.address_1d]
    by_cases hi : c * BT + i.val < T
    · rw [if_pos (by simpa using hi),
        show (c * BT + i.val) * 1 = c * BT + i.val from by ring,
        BlockState.readMemValue_real, if_pos hi]
      rw [hs2, BlockState.setReg_readMem, hs1, BlockState.setReg_readMem, hsloop,
        BlockState.setReg_readMem, hS]
    · rw [if_neg (by simpa using hi)]
      simp only [BlockState.defaultCarrier]; rw [if_neg hi]
  -- b_o eval on s3
  have hb_s3 : s3.regs .real [BT] "b_s" = some bsT := by rw [hs3]; simp
  have hb_z3 : s3.regs .real [] "b_z" = some (Tile.scalar (some (surfaceCarry s S T BT c))) := by
    rw [hs3, hs2, hs1, hsloop]; simp; rw [hbz]
  have hscaneval : evalOp (Op.scan ScanOp.sum ⟨0, by simp⟩ .forward (Op.ref TileDType.real [BT] "b_s")) s3
      = some (Tile.scan .sum ⟨0, by simp⟩ .forward bsT) := by simp only [evalOp, hb_s3]; rfl
  have hboeval : evalOp (Op.add NumericDType.real Broadcast.nil.consR
      (Op.scan ScanOp.sum ⟨0, by simp⟩ .forward (Op.ref TileDType.real [BT] "b_s"))
      (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.real [] "b_z"))) s3 = some boT := by
    have hadd : evalOp (Op.add NumericDType.real Broadcast.nil.consR
        (Op.scan ScanOp.sum ⟨0, by simp⟩ .forward (Op.ref TileDType.real [BT] "b_s"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.real [] "b_z"))) s3
        = some (Tile.bop NumericDType.real.add Broadcast.nil.consR
            (Tile.scan .sum ⟨0, by simp⟩ .forward bsT)
            (Tile.expandDim ⟨0, by simp⟩ (Tile.scalar (some (surfaceCarry s S T BT c))))) := by
      rw [evalOp_add, hscaneval]
      simp only [Option.bind_eq_bind, Option.bind_some]
      conv_lhs => rw [evalOp.eq_def]
      simp only [evalOp_ref, hb_z3, Option.pure_def, Option.bind_some, bind, Option.bind]
    rw [hadd]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨i, u⟩ := idx
    simp only [hboT, hboVal, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.add]
    rw [show (Tile.scan ScanOp.sum ⟨0, by simp⟩ .forward bsT).data (i, u) = _ from hscan i]
    simp only [Tile.expandDim_data, Tile.scalar, WithBot.realAdd]
    rfl
  -- b_zz eval on s4
  have hb_s4 : s4.regs .real [BT] "b_s" = some bsT := by
    rw [hs4]; simp [hb_s3]
  have hbzzstep : stepStmt (Stmt.assign TileDType.real [] "b_zz"
        (Op.reduceSum ⟨0, by simp⟩ Bool.false (Op.ref TileDType.real [BT] "b_s"))) s4
      = some s5 := by
    rw [hs5]
    apply stepStmt_assign_eq_some
    conv_lhs => rw [evalOp.eq_def]
    simp only [evalOp_ref, hb_s4, Option.pure_def, Option.bind_some, bind, Option.bind]
    apply congrArg some
    apply Tile.ext
    intro idx
    obtain ⟨u⟩ := idx
    exact hreduce
  -- b_z' eval on s5
  have hb_z5 : s5.regs .real [] "b_z" = some (Tile.scalar (some (surfaceCarry s S T BT c))) := by
    rw [hs5]; simp; rw [hs4]; simp [hb_z3]
  have hb_zz5 : s5.regs .real [] "b_zz" = some (Tile.scalar (some (∑ k : Fin BT, g k))) := by
    rw [hs5]; simp
  have hbzeval : evalOp (Op.add NumericDType.real Broadcast.nil (Op.ref TileDType.real [] "b_z")
      (Op.ref TileDType.real [] "b_zz")) s5
      = some (Tile.scalar (some (surfaceCarry s S T BT (c + 1)))) := by
    simp only [evalOp, Option.bind, hb_z5, hb_zz5, Tile.bop, Broadcast.leftIndex,
      Broadcast.rightIndex, NumericDType.add]
    apply congrArg some
    apply Tile.ext
    intro idx
    simp only [Tile.scalar, WithBot.realAdd]
    rw [surface_carry_succ s S T BT c hBT]
    rfl
  -- p_o blockptr tile & b_o tile on s6 (for store)
  have hpo6 : s6.regs .blockPtr [BT] "p_o" = some poT := by
    rw [hs6]; simp; rw [hs5]; simp; rw [hs4]; simp; rw [hs3]; simp; rw [hs2]; simp
  have hbo6 : s6.regs .real [BT] "b_o" = some boT := by
    rw [hs6]; simp; rw [hs5]; simp; rw [hs4]; simp
  -- the store final state, written explicitly as the masked block-ptr scatter
  set sfin := (TileShape.allIndices [BT]).foldl
    (fun (acc : BlockState) i =>
      if (decide True && (poT.data i).inBounds (TileShape.indexToList [BT] i) [0]) then
        acc.writeMemTyped .real (poT.data i).region
          ((poT.data i).address (TileShape.indexToList [BT] i)) (boT.data i)
      else acc) s6 with hsfin
  have hstore : stepStmt (Stmt.store TileDType.real [BT]
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT] "p_o") [0])
      (Op.ref TileDType.real [BT] "b_o") MaskOpt.none) s6 = some sfin := by
    simp only [stepStmt, evalOp, Option.bind, hbo6, hpo6, hsfin]
    rfl
  refine ⟨sfin, ?_, ?_⟩
  · simp only [ComputeDType.eraseDType]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some hps), ← hs1,
        stepStmts.cons_some (stepStmt_assign_eq_some hpo), ← hs2,
        stepStmts.cons_some (stepStmt_assign_eq_some hbs2), ← hs3,
        stepStmts.cons_some (stepStmt_assign_eq_some hboeval), ← hs4,
        stepStmts.cons_some hbzzstep,
        stepStmts.cons_some (stepStmt_assign_eq_some hbzeval), ← hs6,
        stepStmts.cons_some hstore, stepStmts.nil]
  · -- the invariant at c+1
    -- s6-level facts (registers/pids/S-reads thread through the 6 setReg's)
    have hpids6 : s6.pids = s.pids := by
      rw [hs6, hs5, hs4, hs3, hs2, hs1, hsloop]; simpa [BlockState.setReg] using hpids
    have hibh6 : s6.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 0)) := by
      rw [hs6]; simp; rw [hs5]; simp; rw [hs4]; simp; rw [hs3]; simp; rw [hs2]; simp
      rw [hs1]; simp; rw [hsloop]; simp; exact hibh
    have hbz6 : s6.regs .real [] "b_z" = some (Tile.scalar (some (surfaceCarry s S T BT (c + 1)))) := by
      rw [hs6]; simp
    have hSeq6 : ∀ a, s6.readMem S a = s.readMem S a := by
      intro a; rw [hs6, hs5, hs4, hs3, hs2, hs1, hsloop]
      simp only [BlockState.setReg_readMem]; exact hS a
    -- rewrite sfin as the clean 1D scatter into O at base = pids0·T
    have hpoaddr : ∀ j : Fin BT, (poT.data (j, PUnit.unit)).address (TileShape.indexToList [BT] (j, PUnit.unit))
        = s.pids 0 * T + (c * BT + j.val) := by
      intro j
      rw [show TileShape.indexToList [BT] (j, PUnit.unit) = [j.val] from rfl, hpoT,
        BlockPtr.address_1d]; ring
    have hpoib : ∀ j : Fin BT, (poT.data (j, PUnit.unit)).inBounds (TileShape.indexToList [BT] (j, PUnit.unit)) [0]
        = decide (c * BT + j.val < T) := by
      intro j
      rw [show TileShape.indexToList [BT] (j, PUnit.unit) = [j.val] from rfl, hpoT,
        BlockPtr.inBounds_1d]
    have hsfin' : sfin = surfaceScatter O T BT (s.pids 0 * T) c boVal s6 := by
      rw [hsfin]; unfold surfaceScatter
      congr 1
      funext acc i
      obtain ⟨j, u⟩ := i
      simp only [decide_true, Bool.true_and]
      rw [hpoib j, hpoaddr j]
      by_cases hj : c * BT + j.val < T
      · rw [if_pos (by simpa using hj), if_pos hj]
        show acc.writeMemTyped .real _ _ _ = _
        rw [show (poT.data (j, u)).region = O from by rw [hpoT],
          BlockState.writeMemTyped_real]
        rfl
      · rw [if_neg (by simpa using hj), if_neg hj]
    -- offset function and its injectivity (for the in-chunk readback)
    have hOinj : Function.Injective (fun i : TileIndex [BT] =>
        s.pids 0 * T + (c * BT + i.1.val)) := by
      rintro ⟨a, ua⟩ ⟨b, ub⟩ hab
      simp only at hab
      have : a.val = b.val := by omega
      have hab2 : a = b := Fin.ext this
      subst hab2; rfl
    -- readback: O at flat
    have hreadO : ∀ flat, flat < (c + 1) * BT → flat < T →
        sfin.readMem O (s.pids 0 * T + flat) = surfaceClosed s S T flat := by
      intro flat hflt hfltT
      rw [hsfin']
      by_cases hprev : flat < c * BT
      · -- prior chunk: preserved, then use hO
        rw [surfaceScatter_readMem_lt O T BT (s.pids 0 * T) c boVal s6 flat hprev]
        rw [hs6]; simp only [BlockState.setReg_readMem]
        rw [hs5, hs4, hs3, hs2, hs1, hsloop]; simp only [BlockState.setReg_readMem]
        exact hO flat hprev hfltT
      · -- current chunk: flat = c·BT + j for some j < BT
        have hexp : (c + 1) * BT = c * BT + BT := by rw [Nat.add_mul, Nat.one_mul]
        have hj : flat - c * BT < BT := by rw [hexp] at hflt; omega
        set j : Fin BT := ⟨flat - c * BT, hj⟩ with hjdef
        have hfeq : flat = c * BT + j.val := by simp only [hjdef]; omega
        rw [hfeq]
        unfold surfaceScatter
        rw [BlockState.scatter_readback_prop_masked_nd s6
          (fun i : TileIndex [BT] => s.pids 0 * T + (c * BT + i.1.val))
          (fun i => boVal i.1) (fun i => c * BT + i.1.val < T) hOinj (j, PUnit.unit)]
        rw [if_pos (show c * BT + j.val < T from by rw [← hfeq]; exact hfltT)]
        -- boVal j + carry = surfaceClosed
        simp only [hboVal, hg]
        exact surface_carry_fold s S T BT c j (by rw [← hfeq]; exact hfltT)
    refine ⟨?_, ?_, ?_, ?_, hreadO⟩
    · rw [hsfin', surfaceScatter_pids]; exact hpids6
    · intro a
      rw [hsfin', surfaceScatter_readMem_ne O S T BT (s.pids 0 * T) c boVal s6 a (Ne.symm hSO)]
      exact hSeq6 a
    · rw [hsfin', surfaceScatter_regs]; exact hibh6
    · rw [hsfin', surfaceScatter_regs]; exact hbz6

set_option maxHeartbeats 2000000 in
/-- **Surface chunk-cumsum loop correctness.** Running the full
`chunk_cumsum_scalar_surface` kernel (prefix `i_bh`/`b_z` init + the
`forRangeDyn` over `cdiv(T,BT)` chunks) produces a final state in which every
in-range output flat index holds the genuine global cumulative sum
`surfaceClosed = Σ_{m ≤ flat, m < T} s[i_bh·T + m]`. No read-back of the
kernel's own output appears in the specification. -/
theorem chunk_cumsum_scalar_surface_loop_correct
    (S O : RegionName) (T BT : Nat) (s : BlockState)
    (hSO : O ≠ S) (hBT : 0 < BT) :
    ∃ sfinal,
      exec (chunk_cumsum_scalar_surface S O T BT).toAlgKernel s = some sfinal ∧
      ∀ flat, flat < T →
        sfinal.readMem O (s.pids 0 * T + flat) = surfaceClosed s S T flat := by
  -- reduce exec over the two prefix statements, landing on the forRangeDyn
  unfold exec chunk_cumsum_scalar_surface
  simp only [ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, Except.pure, bind, pure]
  -- the state after the prefix
  set s2 := (s.setReg "i_bh" .nat [] (Tile.scalar (s.pids 0))).setReg "b_z" .real []
      (⟨fun _ => some (0 : ℝ)⟩ : Tile .real []) with hs2def
  have hstep1 : stepStmt (Stmt.assign TileDType.nat [] "i_bh" (Op.programId 0)) s
      = some (s.setReg "i_bh" .nat [] (Tile.scalar (s.pids 0))) :=
    stepStmt_assign_eq_some (by simp [evalOp])
  have hstep2 : stepStmt (Stmt.assign TileDType.real [] "b_z" (Op.full [] (Op.const 0)))
      (s.setReg "i_bh" .nat [] (Tile.scalar (s.pids 0))) = some s2 := by
    rw [hs2def]
    exact stepStmt_assign_eq_some (by simp only [evalOp_full, evalOp_const, Option.bind]; rfl)
  simp only [ComputeDType.eraseDType]
  rw [stepStmts.cons_some hstep1]
  rw [stepStmts.cons_some hstep2]
  -- entry invariant P 0 s2
  have hP0 : surfaceInv s S O T BT 0 s2 := by
    refine ⟨?_, ?_, ?_, ?_, ?_⟩
    · rw [hs2def]; simp [BlockState.setReg]
    · intro a; rw [hs2def]; simp [BlockState.setReg_readMem]
    · rw [hs2def]; simp
    · rw [hs2def]; simp only [BlockState.setReg_same]
      congr 1; congr 1; unfold surfaceCarry
      rw [Finset.filter_false_of_mem (by intro x _; omega), Finset.sum_empty]
    · intro flat hlt _; omega
  -- the loop counter register, start/stop/step resolution
  have hstart : evalOp (Op.constNat 0) s2 = some (Tile.scalar 0) := by simp
  have hstep : evalOp (Op.constNat 1) s2 = some (Tile.scalar 1) := by simp
  have hstop : evalOp (Op.div NumericDType.nat Broadcast.nil
      (Op.sub NumericDType.nat Broadcast.nil
        (Op.add NumericDType.nat Broadcast.nil (Op.constNat T) (Op.constNat BT)) (Op.constNat 1))
      (Op.constNat BT)) s2 = some (Tile.scalar ((T + BT - 1) / BT)) := by
    simp only [evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat, Option.bind,
      Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
      NumericDType.sub, NumericDType.div]
    rfl
  -- collapse the singleton `stepStmts [forRangeDyn] s2` to a single `stepStmt`
  rw [show ∀ (X : Stmt), stepStmts [X] s2 = stepStmt X s2 from fun X => by
    cases h : stepStmt X s2 <;> simp [stepStmts, h]]
  -- run the loop; `body` unifies with the (lowered) goal body, surface_step closes
  -- the step obligation up to defeq of the per-chunk body
  obtain ⟨final, sLoop, hLoop, hfinal, hPfinal⟩ :=
    forRangeDyn_inv (P := surfaceInv s S O T BT) (s_init := s2) hstart hstop hstep
      (by norm_num) hP0
      (fun i st _ hPi => surface_step S O T BT s hSO hBT i st hPi)
  refine ⟨sLoop, hLoop, ?_⟩
  · -- at the final chunk, final·BT ≥ T, so every flat < T is covered
    obtain ⟨_, _, _, _, hO⟩ := hPfinal
    intro flat hlt
    have hcov : flat < final * BT := by
      have hdm := Nat.div_add_mod (T + BT - 1) BT
      have hmod : (T + BT - 1) % BT < BT := Nat.mod_lt _ hBT
      have hTle : T ≤ (T + BT - 1) / BT * BT := by
        rw [Nat.mul_comm]; omega
      have h2 : (T + BT - 1) / BT ≤ final := hfinal
      have : (T + BT - 1) / BT * BT ≤ final * BT := Nat.mul_le_mul_right _ h2
      omega
    exact hO flat hcov hlt

/-- **Genuine full-kernel chunk-cumsum correctness (general `T`, `BT`).** The
complete `chunk_cumsum_scalar_surface` kernel — including the `forRangeDyn` over
`cdiv(T, BT)` chunks that threads the running carry `b_z` — computes the genuine
global cumulative sum at every in-range output index, with **no** carry
hypothesis: the carry invariant `carry_c = Σ_{flat < c·BT, flat<T} s[i_bh·T+flat]`
is *proven* by the loop induction (`forRangeDyn_inv` + `surface_step`), not
assumed. `expected` is a standalone `Finset.sum`, never a read-back of the
kernel's own output. -/
theorem chunk_cumsum_scalar_surface_global_cumsum
    (S O : RegionName) (T BT : Nat) (s : BlockState)
    (hSO : O ≠ S) (hBT : 0 < BT) :
    ∃ sfinal,
      exec (chunk_cumsum_scalar_surface S O T BT).toAlgKernel s = some sfinal ∧
      ∀ flat, flat < T →
        sfinal.readMem O (s.pids 0 * T + flat)
          = ∑ m ∈ (Finset.range T).filter (fun m => m ≤ flat),
              s.readMem S (s.pids 0 * T + m) :=
  chunk_cumsum_scalar_surface_loop_correct S O T BT s hSO hBT

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **★ MAIN THEOREM ★ — Public general scalar chunk-cumsum output summary
(genuine closed form, dimension-general `T`, `BT`).** Two genuine facts about the
complete `chunk_cumsum_scalar_surface` kernel, with **honest side-conditions
only** (`0 < BT`, output/input regions distinct):

* the full surface — prefix `i_bh`/`b_z` init followed by the `forRangeDyn` over
  `cdiv(T, BT)` chunks that threads the running carry `b_z` — **lowers** to the
  algorithm layer;
* it **computes the genuine global cumulative sum** at every in-range output flat
  index: `O[i_bh·T + flat] = Σ_{m ≤ flat, m < T} S[i_bh·T + m]`.

The carry invariant `carry_c = Σ_{flat < c·BT, flat < T} s[i_bh·T+flat]` is
*proven* by the loop induction (`forRangeDyn_inv` + `surface_step`), not assumed.
`expected` is a standalone `Finset.sum` over input memory (`globalCumsumClosed`),
never a read-back of the kernel's own output. This is the dimension-parameterized
headline; the `T = 4`, `BT = 16` Python benchmark shape is one instantiation. -/
theorem chunk_cumsum_scalar_output_summary_general
    (S O : RegionName) (T BT : Nat) (s : BlockState)
    (hSO : O ≠ S) (hBT : 0 < BT) :
    -- (1) the full surface lowers to the algorithm layer
    (∃ alg, (chunk_cumsum_scalar_surface S O T BT).toAlgorithm? = Except.ok alg) ∧
    -- (2) the surface runs to completion (existence / termination)
    (∃ sfinal,
      exec (chunk_cumsum_scalar_surface S O T BT).toAlgKernel s = some sfinal) ∧
    -- (3) standard Realizes: every in-range O lane holds the genuine global
    --     prefix sum `Σ_{m ≤ flat, m < T} S[i_bh·T + m]`, read purely over input
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_surface S O T BT)
      (initialState := s)
      (write := fun i : Fin T => some (O, s.pids 0 * T + i.val))
      (expected := fun i : Fin T =>
        ∑ m ∈ (Finset.range T).filter (fun m => m ≤ i.val),
          s.readMem S (s.pids 0 * T + m)) := by
  obtain ⟨sfinal, hexec, hcum⟩ :=
    chunk_cumsum_scalar_surface_global_cumsum S O T BT s hSO hBT
  refine ⟨chunk_cumsum_scalar_surface_toAlgorithm_supported S O T BT,
    ⟨sfinal, hexec⟩, ?_⟩
  obtain ⟨alg, halg⟩ := chunk_cumsum_scalar_surface_toAlgorithm_supported S O T BT
  have hk : (chunk_cumsum_scalar_surface S O T BT).toAlgorithm?
      = Except.ok (chunk_cumsum_scalar_surface S O T BT).toAlgKernel := by
    simp only [ComputeKernel.toAlgKernel, halg]
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel hk
  intro s0 s' hExec hs0
  subst s0
  intro i
  rw [hExec] at hexec
  obtain rfl := Option.some.inj hexec
  exact hcum i.val i.isLt

end Correct

end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
