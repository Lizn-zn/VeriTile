import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

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
chunk_cumsum_scalar_python_test_shape_summary                  ← TOP THEOREM
  ├─ chunk_cumsum_scalar_python_test_shape_surface_toAlgorithm_supported
  │     └─ chunk_cumsum_scalar_surface_toAlgorithm_supported   full surface lowers
  └─ chunk_cumsum_scalar_python_test_shape_all_outputs_compute_correct
       ├─ chunk_cumsum_scalar_single_block_python_test_shape_compute_correct
       │     └─ chunk_cumsum_scalar_single_block_surface_closed_form
       │          ├─ chunk_cumsum_scalar_single_block_surface_compute_correct
       │          └─ singleBlockCumsumStoreValue_eq_closed  (cumsum = prefix Σ)
       ├─ chunk_cumsum_scalar_store_python_test_shape_compute_correct
       │     └─ chunk_cumsum_scalar_store_slice_compute_correct
       └─ chunk_cumsum_scalar_cumsum_python_test_shape_compute_correct
            └─ chunk_cumsum_scalar_cumsum_slice_closed_form  (under carry hyp.)
                 ├─ chunk_cumsum_scalar_cumsum_slice_compute_correct
                 └─ cumsumStoreValue_eq_globalCumsumClosed   (carry + cumsum = global Σ)

mathematical core (the carry-fold + within-chunk identity):
  scan1d_sum / scan1d_sum_if          `tl.cumsum` = guarded prefix `Finset.sum`
  singleBlockCumsumStoreValue_eq_closed   single chunk (carry = 0) = global prefix Σ
  cumsumStoreValue_eq_globalCumsumClosed  carry_c + within-chunk Σ = global prefix Σ,
                                          given carry_c = Σ_{flat < c·BT, flat<T} s[flat]
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` (the
`BT ∈ {16,32,64}` config set) is not modeled — the public test-shape theorems fix
the checked Python shape `T = 4` with `BT = 16`, where the chunk loop has a single
iteration; the closed-form lemmas (`scan1d_sum*`, `*_eq_closed`,
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
(`Carry`) in `chunk_cumsum_scalar_cumsum_slice`; the single-Python-chunk surface
realizes the global prefix sum end-to-end with `carry = 0`. Output injectivity is
a side condition (discharged for the test shape).
-/

namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel

open VeriTile.Triton

set_option linter.unusedSimpArgs false

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
    (Tile.scan .sum ⟨0, by simp⟩
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
      ((Tile.scan .sum ⟨0, by simp⟩
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
      ((Tile.scan .sum ⟨0, by simp⟩ (cumsumInputTile s S T BT)).data (i, PUnit.unit))
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
    ((Tile.scan .sum ⟨0, by simp⟩
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

theorem chunk_cumsum_scalar_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 16 => vecOffset s 4 16 i) := by
  intro a b h
  simp [vecOffset, tIndex] at h
  exact Fin.ext (by omega)

theorem chunk_cumsum_scalar_single_block_python_test_shape_offset_injective
    (s : BlockState) :
    Function.Injective (fun i : Fin 16 => singleBlockVecOffset s 4 i) := by
  intro a b h
  simp [singleBlockVecOffset] at h
  exact Fin.ext (by omega)

theorem chunk_cumsum_scalar_python_test_shape_surface_toAlgorithm_supported
    (S O : RegionName) :
    ∃ alg, (chunk_cumsum_scalar_surface S O 4 16).toAlgorithm? =
      Except.ok alg := by
  exact chunk_cumsum_scalar_surface_toAlgorithm_supported S O 4 16

/-- **Genuine Python test-shape correctness (`T = 4`, `BT = 16`).** For the
checked Python shape the chunk loop runs exactly once (`cdiv 4 16 = 1`) with the
carry at its initial zero, so the single-Python-chunk surface is the actual
`S → O` path. It realizes the genuine global prefix-sum closed form
`singleBlockCumsumClosed = Σ_{flat ≤ i, flat < 4} s[i_bh·4 + flat]`. The
`expected` value is a standalone `Finset.sum` — not a read-back of the kernel's
own output. -/
theorem chunk_cumsum_scalar_single_block_python_test_shape_compute_correct
    (S O : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_single_block_surface S O 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => singleBlockActive s 4 i)
        (fun i => (O, singleBlockVecOffset s 4 i)))
      (expected := fun i : Fin 16 =>
        singleBlockCumsumClosed s S 4 16 i) := by
  exact chunk_cumsum_scalar_single_block_surface_closed_form S O 4 16 s
    (chunk_cumsum_scalar_single_block_python_test_shape_offset_injective s)

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

/-- **Genuine per-chunk carry-fold correctness (`T = 4`, `BT = 16`).** Given the
carry buffer holds the genuine prefix sum of all prior chunks, the cumsum slice
realizes the genuine global cumulative sum `globalCumsumClosed`. -/
theorem chunk_cumsum_scalar_cumsum_python_test_shape_compute_correct
    (S Carry O : RegionName) (s : BlockState)
    (hcarry : s.readMem Carry (s.pids 0)
      = ∑ flat ∈ (Finset.range 4).filter (fun flat => flat < s.pids 1 * 16),
          s.readMem S (s.pids 0 * 4 + flat)) :
    ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_cumsum_slice S Carry O 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 4 16 i)
        (fun i => (O, vecOffset s 4 16 i)))
      (expected := fun i : Fin 16 => globalCumsumClosed s S 4 16 i) := by
  exact chunk_cumsum_scalar_cumsum_slice_closed_form S Carry O 4 16 s
    (chunk_cumsum_scalar_python_test_shape_offset_injective s) hcarry

/-- Python test-shape output coverage for scalar chunk cumsum: the single-block
surface, precomputed store slice, and cumsum-with-carry slice all realize their
checked masked output shapes. -/
theorem chunk_cumsum_scalar_python_test_shape_all_outputs_compute_correct
    (S BO Carry O : RegionName) (s : BlockState)
    (hcarry : s.readMem Carry (s.pids 0)
      = ∑ flat ∈ (Finset.range 4).filter (fun flat => flat < s.pids 1 * 16),
          s.readMem S (s.pids 0 * 4 + flat)) :
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_single_block_surface S O 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => singleBlockActive s 4 i)
        (fun i => (O, singleBlockVecOffset s 4 i)))
      (expected := fun i : Fin 16 =>
        singleBlockCumsumClosed s S 4 16 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_store_slice BO O 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 4 16 i)
        (fun i => (O, vecOffset s 4 16 i)))
      (expected := fun i : Fin 16 => storeValue s BO 4 16 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_cumsum_slice S Carry O 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 4 16 i)
        (fun i => (O, vecOffset s 4 16 i)))
      (expected := fun i : Fin 16 => globalCumsumClosed s S 4 16 i)) := by
  refine ⟨?_, ?_, ?_⟩
  · exact chunk_cumsum_scalar_single_block_python_test_shape_compute_correct S O s
  · exact chunk_cumsum_scalar_store_python_test_shape_compute_correct BO O s
  · exact chunk_cumsum_scalar_cumsum_python_test_shape_compute_correct
      S Carry O s hcarry

/-- **Public Python test-shape summary for scalar chunk cumsum.** The full
surface lowers to the algorithm layer, and every output slice realizes a genuine
specification for the checked `T = 4`, `BT = 16` shape:

* the single-Python-chunk surface (the actual `S → O` path, where the chunk loop
  runs once with carry `= 0`) realizes the genuine global prefix sum
  `singleBlockCumsumClosed`;
* the boundary-checked store slice passes a precomputed tile through;
* the cumsum-with-carry slice realizes the genuine global cumulative sum
  `globalCumsumClosed` when the carry buffer holds the prior-chunk prefix sum. -/
theorem chunk_cumsum_scalar_python_test_shape_summary
    (S BO Carry O : RegionName) (s : BlockState)
    (hcarry : s.readMem Carry (s.pids 0)
      = ∑ flat ∈ (Finset.range 4).filter (fun flat => flat < s.pids 1 * 16),
          s.readMem S (s.pids 0 * 4 + flat)) :
    (∃ alg, (chunk_cumsum_scalar_surface S O 4 16).toAlgorithm? =
      Except.ok alg) ∧
    ((ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_single_block_surface S O 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => singleBlockActive s 4 i)
        (fun i => (O, singleBlockVecOffset s 4 i)))
      (expected := fun i : Fin 16 =>
        singleBlockCumsumClosed s S 4 16 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_store_slice BO O 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 4 16 i)
        (fun i => (O, vecOffset s 4 16 i)))
      (expected := fun i : Fin 16 => storeValue s BO 4 16 i)) ∧
    (ComputeCorrect.Realizes
      (kernel := chunk_cumsum_scalar_cumsum_slice S Carry O 4 16)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 16 => active s 4 16 i)
        (fun i => (O, vecOffset s 4 16 i)))
      (expected := fun i : Fin 16 => globalCumsumClosed s S 4 16 i))) := by
  refine ⟨chunk_cumsum_scalar_python_test_shape_surface_toAlgorithm_supported S O,
    chunk_cumsum_scalar_python_test_shape_all_outputs_compute_correct
      S BO Carry O s hcarry⟩



















/-- `output_summary` for the scalar Python chunk-cumsum surface: the genuine
single-Python-chunk closed-form correctness (`S → O` realizes the prefix sum). -/
abbrev chunk_cumsum_scalar_python_test_shape_output_summary
    (S O : RegionName) (s : BlockState) :=
  chunk_cumsum_scalar_single_block_python_test_shape_compute_correct S O s

end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
