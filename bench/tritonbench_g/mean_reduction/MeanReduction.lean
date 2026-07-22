import VeriTile.Triton
import VeriTile.Examples.Common

/-!
# `mean_reduction` — strict per-kernel correctness

`mean_dim_kernel` reduces a `[M, N]` row-major matrix `X` along its `N` axis:
program `pid` owns rows `[pid·BLOCK_M, …)`, accumulates each row over `N` in
`BLOCK_N`-wide chunks inside `for off in range(0, N, BLOCK_N)`, then writes the
row mean `tl.sum(_mean, axis=1) / N` to `Mean`, masked by `row < M`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, including the `for off` accumulation loop, proven by a loop invariant
(`meanLoopContextInvariant`). The host launch (`mean_dim_kernel[grid](...)`,
the grid size `cdiv(M, BLOCK_M)`, the host `BLOCK_M = BLOCK_N = 8` choice, the
`dim_compress` permutation/contiguity, and cross-program composition of `Mean`)
is the *trusted boundary*, not a proof obligation here. The program id enters
only via `BlockState`, so the per-program statement covers every program.

## Proof architecture

```
mean_dim_kernel_output_summary                  ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)               surface lowers to the algorithm layer
  └─ mean_dim_kernel_compute_correct            ← mean_dim_kernel_correct_target
       └─ mean_dim_kernel_compute_correct_of_algorithm
            └─ mean_dim_kernel_alg_post_of_exec
                 └─ meanProjectedBody_alg_post   (pre-loop ∘ forRange ∘ post-loop)
                      └─ meanLoopContextInvariant (forRange_inv loop invariant)
                           └─ meanSpec  (per-row `Finset.sum` over `Fin N`, / N)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled (the kernel is launched with fixed `BLOCK_M`/`BLOCK_N`). The
`.to(tl.float32)` cast on each loaded chunk reduces to the identity at the
algorithm layer (post-erasure all dtypes unify to `ℝ`). The masked load uses
`other=0.0`, so out-of-range columns contribute `0` to the row sum. The output
scatter relies on `meanOutOffset_injective_col1` (the per-row output offset map
is injective). The only side condition is `BLOCK_N ≠ 0` (a nonempty inner chunk).
-/

namespace VeriTile.Bench.TritonBenchG.MeanReduction

open VeriTile.Triton VeriTile.Examples

/-- Faithful transcription of `mean_reduction.py`'s `mean_dim_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `[:, None]` / `[None, :]` dimension annotations preserved.
- Python `BLOCK_M` / `BLOCK_N: tl.constexpr` → Lean `Nat` parameters.

The proof below connects the full-row spec to a loop invariant for the
`for off in range(...)` accumulation.
-/
def mean_dim_kernel
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0) * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))[:, None]
  X = X + pid * $(N)
  Mean = Mean + pid
  row_mask = pid < $(M)
  _mean = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_N)) {
    cols = off + tl.arange(0, $(BLOCK_N))[None, :]
    col_mask = cols < $(N)
    mask = row_mask and col_mask
    a = tl.load(X + cols, mask, other=0.0).to(tl.float32)
    _mean += a
  }
  mean = tl.sum(_mean, axis=1) / $(N)
  mean = mean[:, None]
  tl.store(Mean, mean, row_mask)
}

def meanOutOffset (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def meanRowActive (s : BlockState) (M BLOCK_M : Nat) (i : Fin BLOCK_M) : Prop :=
  meanOutOffset s BLOCK_M i < M

instance meanRowActiveDecidable
    (s : BlockState) (M BLOCK_M : Nat) (i : Fin BLOCK_M) :
    Decidable (meanRowActive s M BLOCK_M i) := by
  unfold meanRowActive
  infer_instance

/-- Input element `X[row, col]` of the row-major `[M, N]` input, at global row
`meanOutOffset s BLOCK_M i = pid0·BLOCK_M + i` (row stride `N`, unit column
stride). -/
noncomputable def meanInpElem
    (s : BlockState) (X : RegionName) (N BLOCK_M : Nat)
    (i : Fin BLOCK_M) (col : Nat) : ℝ :=
  s.readMem X (meanOutOffset s BLOCK_M i * N + col)

noncomputable def meanSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  ((Finset.univ : Finset (Fin N)).sum fun j =>
    meanInpElem s X N BLOCK_M i j.val) / (N : ℝ)

noncomputable def meanLanePrefix
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) : ℝ :=
  ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_N = j.val).sum
    fun col => meanInpElem s X N BLOCK_M i col

noncomputable def meanMaskedAccumulatorSpec
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      some
        (if meanRowActive s M BLOCK_M idx.1 then
          meanLanePrefix s X N BLOCK_M BLOCK_N off idx.1 idx.2.1
        else
          0) }

noncomputable def meanChunkLoadSpec
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      some
        (if meanRowActive s M BLOCK_M idx.1 ∧ off + idx.2.1.val < N then
          meanInpElem s X N BLOCK_M idx.1 (off + idx.2.1.val)
        else
          0) }

noncomputable def meanFromAccumulatorSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  ((Finset.univ : Finset (Fin BLOCK_N)).sum fun j =>
    meanLanePrefix s X N BLOCK_M BLOCK_N off i j) / (N : ℝ)

noncomputable def meanFromMaskedAccumulatorSpec
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  ((Finset.univ : Finset (Fin BLOCK_N)).sum fun j =>
    if meanRowActive s M BLOCK_M i then
      meanLanePrefix s X N BLOCK_M BLOCK_N off i j
    else
      0) / (N : ℝ)

@[simp] theorem meanLanePrefix_zero
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) :
    meanLanePrefix s X N BLOCK_M BLOCK_N 0 i j = 0 := by
  simp [meanLanePrefix]

theorem meanMaskedAccumulatorSpec_zero
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N : Nat) :
    meanMaskedAccumulatorSpec s X M N BLOCK_M BLOCK_N 0 =
      { data := fun _ : TileIndex [BLOCK_M, BLOCK_N] => some 0 } := by
  ext idx
  simp [meanMaskedAccumulatorSpec]

theorem meanChunkLane_mod
    (off BLOCK_N : Nat) (j : Fin BLOCK_N) (hOff : off % BLOCK_N = 0) :
    (off + j.val) % BLOCK_N = j.val := by
  rw [Nat.add_mod]
  rw [hOff]
  simp [Nat.mod_eq_of_lt j.isLt]

theorem meanLoopOffset_mod_step
    (off BLOCK_N : Nat) (hOff : off % BLOCK_N = 0) :
    (off + BLOCK_N) % BLOCK_N = 0 := by
  rw [Nat.add_mod, hOff]
  simp

theorem meanChunkLane_not_mem_current
    (N off BLOCK_N : Nat) (j : Fin BLOCK_N) :
    off + j.val ∉ (Finset.range off).filter
      (fun col => col < N ∧ col % BLOCK_N = j.val) := by
  simp

theorem meanLanePrefix_step
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N)
    (hOff : off % BLOCK_N = 0) :
    meanLanePrefix s X N BLOCK_M BLOCK_N (off + BLOCK_N) i j =
      meanLanePrefix s X N BLOCK_M BLOCK_N off i j +
        if off + j.val < N then
          meanInpElem s X N BLOCK_M i (off + j.val)
        else
          0 := by
  classical
  let pred : Nat → Prop := fun col => col < N ∧ col % BLOCK_N = j.val
  let f : Nat → ℝ := fun col => meanInpElem s X N BLOCK_M i col
  have hunique :
      ∀ col, off ≤ col → col < off + BLOCK_N → col % BLOCK_N = j.val →
        col = off + j.val := by
    intro col hle hlt hmod
    obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hle
    have hklt : k < BLOCK_N := by omega
    have hmodk : (off + k) % BLOCK_N = k := by
      rw [Nat.add_mod, hOff, Nat.mod_eq_of_lt hklt]
      simpa using Nat.mod_eq_of_lt hklt
    rw [hmodk] at hmod
    omega
  by_cases hjN : off + j.val < N
  · have hset :
        (Finset.range (off + BLOCK_N)).filter pred =
          insert (off + j.val) ((Finset.range off).filter pred) := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_insert]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact Or.inr ⟨hltOff, h.2⟩
        · exact Or.inl (hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2)
      · intro h
        rcases h with h | h
        · subst h
          exact ⟨by omega, hjN, meanChunkLane_mod off BLOCK_N j hOff⟩
        · exact ⟨by omega, h.2⟩
    unfold meanLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_N)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f + f (off + j.val)
    rw [hset]
    rw [Finset.sum_insert]
    · ring
    · exact meanChunkLane_not_mem_current N off BLOCK_N j
  · have hset :
        (Finset.range (off + BLOCK_N)).filter pred =
          (Finset.range off).filter pred := by
      ext col
      simp only [Finset.mem_filter, Finset.mem_range]
      constructor
      · intro h
        by_cases hltOff : col < off
        · exact ⟨hltOff, h.2⟩
        · have hcol : col = off + j.val :=
            hunique col (Nat.le_of_not_gt hltOff) h.1 h.2.2
          omega
      · intro h
        exact ⟨by omega, h.2⟩
    unfold meanLanePrefix
    simp [hjN]
    change ((Finset.range (off + BLOCK_N)).filter pred).sum f =
      ((Finset.range off).filter pred).sum f
    rw [hset]

def meanLoopInvariant
    (s0 : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N : Nat)
    (off : Nat) (st : BlockState) : Prop :=
  off % BLOCK_N = 0 ∧
    st.regs .real [BLOCK_M, BLOCK_N] "_mean" =
      some (meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off)

theorem meanLoopInvariant_init_of_zero_reg
    (s0 st : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N : Nat)
    (hReg :
      st.regs .real [BLOCK_M, BLOCK_N] "_mean" =
        some { data := fun _ : TileIndex [BLOCK_M, BLOCK_N] => some 0 }) :
    meanLoopInvariant s0 X M N BLOCK_M BLOCK_N 0 st := by
  constructor
  · simp
  · simpa [meanMaskedAccumulatorSpec_zero] using hReg

theorem meanMaskedAccumulatorSpec_step_add
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (hOff : off % BLOCK_N = 0) :
    meanMaskedAccumulatorSpec s X M N BLOCK_M BLOCK_N (off + BLOCK_N) =
      { data := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          some
            (WithBot.unbotD 0
                ((meanMaskedAccumulatorSpec s X M N BLOCK_M BLOCK_N off).data idx) +
              WithBot.unbotD 0
                ((meanChunkLoadSpec s X M N BLOCK_M BLOCK_N off).data idx)) } := by
  ext idx
  by_cases hrow : meanRowActive s M BLOCK_M idx.1
  · by_cases hcol : off + idx.2.1.val < N
    · simp [meanMaskedAccumulatorSpec, meanChunkLoadSpec, hrow, hcol,
        meanLanePrefix_step s X N BLOCK_M BLOCK_N off idx.1 idx.2.1 hOff]
    · simp [meanMaskedAccumulatorSpec, meanChunkLoadSpec, hrow, hcol,
        meanLanePrefix_step s X N BLOCK_M BLOCK_N off idx.1 idx.2.1 hOff]
  · simp [meanMaskedAccumulatorSpec, meanChunkLoadSpec, hrow]

theorem meanLoopInvariant_step_of_accumulator_update
    (s0 st' : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (hOff : off % BLOCK_N = 0)
    (hUpdate :
      st'.regs .real [BLOCK_M, BLOCK_N] "_mean" =
        some
          { data := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
              some
                (WithBot.unbotD 0
                    ((meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off).data idx) +
                  WithBot.unbotD 0
                    ((meanChunkLoadSpec s0 X M N BLOCK_M BLOCK_N off).data idx)) }) :
    meanLoopInvariant s0 X M N BLOCK_M BLOCK_N (off + BLOCK_N) st' := by
  constructor
  · exact meanLoopOffset_mod_step off BLOCK_N hOff
  · simpa [meanMaskedAccumulatorSpec_step_add s0 X M N BLOCK_M BLOCK_N off hOff]
      using hUpdate

private theorem sum_range_eq_sum_fin (N : Nat) (f : Nat → ℝ) :
    (Finset.range N).sum f =
      (Finset.univ : Finset (Fin N)).sum (fun j => f j.val) := by
  classical
  apply Finset.sum_bij (fun n hn => (⟨n, by simpa using hn⟩ : Fin N))
  · intro _ _
    exact Finset.mem_univ _
  · intro _ _ _ _ h
    exact Fin.ext_iff.mp h
  · intro j _
    refine ⟨j.val, Finset.mem_range.mpr j.isLt, ?_⟩
    simp
  · intro _ _
    simp

private theorem sum_lane_prefix_eq_sum_range
    (N BLOCK_N off : Nat) (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off)
    (f : Nat → ℝ) :
    ((Finset.univ : Finset (Fin BLOCK_N)).sum fun j =>
      ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_N = j.val).sum f) =
    (Finset.range N).sum f := by
  classical
  let g : Nat → Fin BLOCK_N := fun col =>
    ⟨col % BLOCK_N, Nat.mod_lt col hBLOCK_N⟩
  have hinner : ∀ j : Fin BLOCK_N,
      ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_N = j.val).sum f =
      (((Finset.range off).filter fun col => col < N).filter fun col => g col = j).sum f := by
    intro j
    apply Finset.sum_congr
    · ext col
      simp [g, Fin.ext_iff, and_left_comm, and_assoc]
    · intro _ _
      rfl
  rw [Finset.sum_congr rfl (fun j _ => hinner j)]
  rw [Finset.sum_fiberwise]
  have hfilter : (Finset.range off).filter (fun col => col < N) = Finset.range N := by
    ext col
    simp only [Finset.mem_filter, Finset.mem_range]
    constructor
    · intro h
      exact h.2
    · intro h
      exact ⟨Nat.lt_of_lt_of_le h hoff, h⟩
  rw [hfilter]

theorem meanFromAccumulatorSpec_eq_meanSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off) :
    meanFromAccumulatorSpec s X N BLOCK_M BLOCK_N off i =
      meanSpec s X N BLOCK_M i := by
  unfold meanFromAccumulatorSpec meanSpec meanLanePrefix
  rw [sum_lane_prefix_eq_sum_range N BLOCK_N off hBLOCK_N hoff]
  rw [sum_range_eq_sum_fin]

theorem meanFromMaskedAccumulatorSpec_eq_meanSpec
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) (hrow : meanRowActive s M BLOCK_M i)
    (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off) :
    meanFromMaskedAccumulatorSpec s X M N BLOCK_M BLOCK_N off i =
      meanSpec s X N BLOCK_M i := by
  simpa [meanFromMaskedAccumulatorSpec, hrow, meanFromAccumulatorSpec] using
    meanFromAccumulatorSpec_eq_meanSpec s X N BLOCK_M BLOCK_N off i hBLOCK_N hoff

def mean_dim_kernel_correct_target
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState) : Prop :=
  ComputeCorrect.Realizes_without_Rounding
    (kernel := mean_dim_kernel X Mean M N BLOCK_M BLOCK_N)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (fun i : Fin BLOCK_M => meanOutOffset s BLOCK_M i < M)
      (fun i => (Mean, meanOutOffset s BLOCK_M i)))
    (expected := fun i => meanSpec s X N BLOCK_M i)

def mean_dim_kernel_alg_post
    (X Mean : RegionName)
    (M N BLOCK_M _BLOCK_N : Nat) (s s' : BlockState) : Prop :=
  ∀ i : Fin BLOCK_M,
    meanOutOffset s BLOCK_M i < M →
    s'.readMem Mean (meanOutOffset s BLOCK_M i) =
      meanSpec s X N BLOCK_M i

theorem meanOutOffset_injective_col1
    (s : BlockState) (BLOCK_M : Nat) :
    Function.Injective
      (fun idx : TileIndex [BLOCK_M, 1] => meanOutOffset s BLOCK_M idx.1) := by
  intro a b h
  apply Prod.ext
  · apply Fin.ext
    simpa [meanOutOffset] using Nat.add_left_cancel h
  · apply Prod.ext
    · apply Fin.ext
      omega
    · cases a.2.2
      cases b.2.2
      rfl

theorem meanStoreFromExpandedMaskedAccumulator_alg_post
    (s0 stBase : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off) :
    mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s0
      ((TileShape.allIndices [BLOCK_M, 1]).foldl
        (fun acc idx =>
          if meanRowActive s0 M BLOCK_M idx.1 then
            acc.writeMem Mean (meanOutOffset s0 BLOCK_M idx.1)
              (WithBot.unbotD 0
                ((Tile.expandDim
                  (⟨1, by simp⟩ : Fin ([BLOCK_M].length + 1))
                  (Tile.bop NumericDType.real.div Broadcast.scalarR
                    (Tile.reduceSumDrop
                      (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length)
                      (meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off))
                    (Tile.scalar (some (N : ℝ) : WithBot ℝ)))).data idx)
              )
          else
            acc)
        stBase) := by
  intro i hi
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _
    (meanOutOffset_injective_col1 s0 BLOCK_M)
    ((i, ⟨0, by omega⟩, PUnit.unit) : TileIndex [BLOCK_M, 1])]
  simp [meanRowActive, hi, Tile.expandDim, TileShape.dropInsertedIndex, Tile.bop,
    NumericDType.div]
  simpa [TileShape.axisDim, TileShape.insertAxisIndex,
    meanMaskedAccumulatorSpec, meanFromMaskedAccumulatorSpec, meanRowActive, hi] using
    meanFromMaskedAccumulatorSpec_eq_meanSpec s0 X M N BLOCK_M BLOCK_N off i
      (by simpa [meanRowActive] using hi) hBLOCK_N hoff

def meanPreLoop
    (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) : List Stmt :=
  [ .assign .nat [BLOCK_M, 1] "pid"
      (.add NumericDType.nat Broadcast.scalarL
        (.mul NumericDType.nat Broadcast.nil (.programId 0) (.constNat BLOCK_M))
        (.expandDim (⟨1, by simp⟩ : Fin ([BLOCK_M].length + 1))
          (.arange BLOCK_M)))
  , .assign .ptr [BLOCK_M, 1] "X"
      (.ptrAdd Broadcast.scalarL (.ptrBase X)
        (.mul NumericDType.nat Broadcast.scalarR
          (.ref .nat [BLOCK_M, 1] "pid") (.constNat N)))
  , .assign .ptr [BLOCK_M, 1] "Mean"
      (.ptrAdd Broadcast.scalarL (.ptrBase Mean)
        (.ref .nat [BLOCK_M, 1] "pid"))
  , .assign .bool [BLOCK_M, 1] "row_mask"
      (.lt ComparableDType.nat Broadcast.scalarR
        (.ref .nat [BLOCK_M, 1] "pid") (.constNat M))
  , .assign .real [BLOCK_M, BLOCK_N] "_mean"
      (.full [BLOCK_M, BLOCK_N] (.const 0))
  ]

theorem meanPreLoop_step_regs
    (s st : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat)
    (hStep :
      stepStmts (meanPreLoop X Mean M N BLOCK_M BLOCK_N) s = some st) :
    st.regs .real [BLOCK_M, BLOCK_N] "_mean" =
        some { data := fun _ : TileIndex [BLOCK_M, BLOCK_N] => some 0 } ∧
      st.regs .ptr [BLOCK_M, 1] "X" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          (X, meanOutOffset s BLOCK_M idx.1 * N) } ∧
      st.regs .ptr [BLOCK_M, 1] "Mean" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          (Mean, meanOutOffset s BLOCK_M idx.1) } ∧
      st.regs .bool [BLOCK_M, 1] "row_mask" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          meanRowActive s M BLOCK_M idx.1 } ∧
      (∀ offset, st.readMem X offset = s.readMem X offset) := by
  unfold meanPreLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, Tile.expandDim,
    TileShape.dropInsertedIndex, Tile.ptrAdd, NumericDType.add,
    NumericDType.mul, BlockState.setReg, Option.bind] at hStep
  subst st
  constructor
  · simp
  · constructor
    · simp [meanOutOffset]
    · constructor
      · simp [meanOutOffset]
      · constructor
        · simp [Tile.cop, ComparableDType.lt, meanOutOffset, meanRowActive]
          funext idx
          rfl
        · intro offset
          rfl

def meanLoopBody (N BLOCK_M BLOCK_N : Nat) : List Stmt :=
  [ .assign .nat [1, BLOCK_N] "cols"
      (.add NumericDType.nat Broadcast.scalarL
        (.ref .nat [] "off")
        (.expandDim (⟨0, by simp⟩ : Fin ([BLOCK_N].length + 1))
          (.arange BLOCK_N)))
  , .assign .bool [1, BLOCK_N] "col_mask"
      (.lt ComparableDType.nat Broadcast.scalarR
        (.ref .nat [1, BLOCK_N] "cols") (.constNat N))
  , .assign .bool [BLOCK_M, BLOCK_N] "mask"
      (.boolAnd Broadcast.nil.consL.consR
        (.ref .bool [BLOCK_M, 1] "row_mask")
        (.ref .bool [1, BLOCK_N] "col_mask"))
  , .assign .real [BLOCK_M, BLOCK_N] "a"
      (.load .real
        (.ptr
          (.ptrAdd Broadcast.nil.consL.consR
            (.ref .ptr [BLOCK_M, 1] "X")
            (.ref .nat [1, BLOCK_N] "cols")))
        (.maskOther
          (.ref .bool [BLOCK_M, BLOCK_N] "mask")
          ((Op.const 0.0).broadcast [BLOCK_M, BLOCK_N])))
  , .assign .real [BLOCK_M, BLOCK_N] "_mean"
      (.add NumericDType.real Broadcast.nil.consSame.consSame
        (.ref .real [BLOCK_M, BLOCK_N] "_mean")
        (.ref .real [BLOCK_M, BLOCK_N] "a"))
  ]

theorem meanLoopBody_step_accumulator_update
    (s0 st st' : BlockState) (X : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (hAcc :
      st.regs .real [BLOCK_M, BLOCK_N] "_mean" =
        some (meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off))
    (hX :
      st.regs .ptr [BLOCK_M, 1] "X" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          (X, meanOutOffset s0 BLOCK_M idx.1 * N) })
    (hRow :
      st.regs .bool [BLOCK_M, 1] "row_mask" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          meanRowActive s0 M BLOCK_M idx.1 })
    (hRead : ∀ offset, st.readMem X offset = s0.readMem X offset)
    (hStep :
      stepStmts (meanLoopBody N BLOCK_M BLOCK_N)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .real [BLOCK_M, BLOCK_N] "_mean" =
      some
        { data := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            some
              (WithBot.unbotD 0
                  ((meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off).data idx) +
                WithBot.unbotD 0
                  ((meanChunkLoadSpec s0 X M N BLOCK_M BLOCK_N off).data idx)) } := by
  unfold meanLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hAcc, hX, hRow, Tile.bop, Tile.cop,
    Tile.expandDim, TileShape.dropInsertedIndex, Tile.ptrAdd,
    NumericDType.add, ComparableDType.lt, Option.bind, meanOutOffset,
    meanRowActive] at hStep
  subst st'
  simp [BlockState.setReg]
  ext idx
  by_cases hmask :
      meanOutOffset s0 BLOCK_M idx.1 < M ∧ off + idx.2.1.val < N
  · rcases hmask with ⟨hrow, hcol⟩
    have hrow' : s0.pids 0 * BLOCK_M + idx.1.val < M := by
      simpa [meanOutOffset] using hrow
    have hread' :
        st.readMem X ((s0.pids 0 * BLOCK_M + idx.1.val) * N +
            (off + idx.2.1.val)) =
          s0.readMem X ((s0.pids 0 * BLOCK_M + idx.1.val) * N +
            (off + idx.2.1.val)) :=
      hRead _
    simp [meanMaskedAccumulatorSpec, meanChunkLoadSpec, meanInpElem,
      meanRowActive, hcol, hrow', hread', meanOutOffset]
  · have hmask' :
        ¬(s0.pids 0 * BLOCK_M + idx.1.val < M ∧ off + idx.2.1.val < N) := by
      simpa [meanOutOffset] using hmask
    simp [meanMaskedAccumulatorSpec, meanChunkLoadSpec, meanInpElem,
      meanRowActive, hmask', meanOutOffset]
    norm_num
    rfl

theorem meanLoopBody_step_preserves_context
    (s0 st st' : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (hAcc :
      st.regs .real [BLOCK_M, BLOCK_N] "_mean" =
        some (meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off))
    (hX :
      st.regs .ptr [BLOCK_M, 1] "X" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          (X, meanOutOffset s0 BLOCK_M idx.1 * N) })
    (hMean :
      st.regs .ptr [BLOCK_M, 1] "Mean" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          (Mean, meanOutOffset s0 BLOCK_M idx.1) })
    (hRow :
      st.regs .bool [BLOCK_M, 1] "row_mask" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          meanRowActive s0 M BLOCK_M idx.1 })
    (hStep :
      stepStmts (meanLoopBody N BLOCK_M BLOCK_N)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.regs .ptr [BLOCK_M, 1] "X" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          (X, meanOutOffset s0 BLOCK_M idx.1 * N) } ∧
      st'.regs .ptr [BLOCK_M, 1] "Mean" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          (Mean, meanOutOffset s0 BLOCK_M idx.1) } ∧
      st'.regs .bool [BLOCK_M, 1] "row_mask" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          meanRowActive s0 M BLOCK_M idx.1 } ∧
      (∀ offset, st'.readMem X offset = st.readMem X offset) := by
  unfold meanLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hAcc, hX, hRow, Tile.bop, Tile.cop,
    Tile.expandDim, TileShape.dropInsertedIndex, Tile.ptrAdd,
    NumericDType.add, ComparableDType.lt, Option.bind, meanOutOffset,
    meanRowActive] at hStep
  subst st'
  constructor
  · exact hX
  · constructor
    · simp [hMean]
    · constructor
      · simp [hRow]
      · intro offset
        rfl

def meanLoopContextInvariant
    (s0 : BlockState) (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat)
    (off : Nat) (st : BlockState) : Prop :=
  meanLoopInvariant s0 X M N BLOCK_M BLOCK_N off st ∧
    st.regs .ptr [BLOCK_M, 1] "X" =
      some { data := fun idx : TileIndex [BLOCK_M, 1] =>
        (X, meanOutOffset s0 BLOCK_M idx.1 * N) } ∧
    st.regs .ptr [BLOCK_M, 1] "Mean" =
      some { data := fun idx : TileIndex [BLOCK_M, 1] =>
        (Mean, meanOutOffset s0 BLOCK_M idx.1) } ∧
    st.regs .bool [BLOCK_M, 1] "row_mask" =
      some { data := fun idx : TileIndex [BLOCK_M, 1] =>
        meanRowActive s0 M BLOCK_M idx.1 } ∧
    (∀ offset, st.readMem X offset = s0.readMem X offset)

theorem meanLoopContextInvariant_init_of_preloop
    (s0 st : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat)
    (hStep :
      stepStmts (meanPreLoop X Mean M N BLOCK_M BLOCK_N) s0 = some st) :
    meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N 0 st := by
  rcases meanPreLoop_step_regs s0 st X Mean M N BLOCK_M BLOCK_N hStep with
    ⟨hZero, hX, hMean, hRow, hRead⟩
  exact ⟨meanLoopInvariant_init_of_zero_reg s0 st X M N BLOCK_M BLOCK_N hZero,
    hX, hMean, hRow, hRead⟩

theorem meanLoopContextInvariant_step_of_body
    (s0 st st' : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (hCtx : meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N off st)
    (hStep :
      stepStmts (meanLoopBody N BLOCK_M BLOCK_N)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N
      (off + BLOCK_N) st' := by
  rcases hCtx with ⟨hInv, hX, hMean, hRow, hRead⟩
  have hUpdate :
      st'.regs .real [BLOCK_M, BLOCK_N] "_mean" =
        some
          { data := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
              some
                (WithBot.unbotD 0
                    ((meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off).data idx) +
                  WithBot.unbotD 0
                    ((meanChunkLoadSpec s0 X M N BLOCK_M BLOCK_N off).data idx)) } :=
    meanLoopBody_step_accumulator_update s0 st st' X M N BLOCK_M BLOCK_N off
      hInv.2 hX hRow hRead hStep
  rcases meanLoopBody_step_preserves_context s0 st st' X Mean M N BLOCK_M
      BLOCK_N off hInv.2 hX hMean hRow hStep with
    ⟨hX', hMean', hRow', hReadStep⟩
  refine ⟨?_, hX', hMean', hRow', ?_⟩
  · exact meanLoopInvariant_step_of_accumulator_update s0 st' X M N BLOCK_M
      BLOCK_N off hInv.1 hUpdate
  · intro offset
    rw [hReadStep offset, hRead offset]

theorem meanLoopContextInvariant_body_step_exists
    (s0 st : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (hCtx : meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N off st) :
    ∃ st',
      stepStmts (meanLoopBody N BLOCK_M BLOCK_N)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st' ∧
      meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N
        (off + BLOCK_N) st' := by
  rcases hCtx with ⟨hInv, hX, hMean, hRow, hRead⟩
  cases hStep :
      stepStmts (meanLoopBody N BLOCK_M BLOCK_N)
        (st.setReg "off" .nat [] (Tile.scalar off)) with
  | none =>
      unfold meanLoopBody at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hInv.2, hX, hRow, Tile.bop, Tile.cop,
        Tile.expandDim, TileShape.dropInsertedIndex, Tile.ptrAdd,
        NumericDType.add, ComparableDType.lt, Option.bind, meanOutOffset,
        meanRowActive] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact meanLoopContextInvariant_step_of_body s0 st st' X Mean M N
        BLOCK_M BLOCK_N off ⟨hInv, hX, hMean, hRow, hRead⟩ hStep

theorem meanForRange_context_of_preloop
    (s0 stPre stLoop : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat)
    (hStepNe : BLOCK_N ≠ 0)
    (hPre :
      stepStmts (meanPreLoop X Mean M N BLOCK_M BLOCK_N) s0 = some stPre)
    (hLoop :
      stepStmt (.forRange "off" 0 N BLOCK_N (meanLoopBody N BLOCK_M BLOCK_N))
        stPre = some stLoop) :
    ∃ final,
      N ≤ final ∧
        meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N final stLoop := by
  have hInit :
      meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N 0 stPre :=
    meanLoopContextInvariant_init_of_preloop s0 stPre X Mean M N BLOCK_M
      BLOCK_N hPre
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "off") (start := 0) (stop := N) (step := BLOCK_N)
      (body := meanLoopBody N BLOCK_M BLOCK_N)
      (P := meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N)
      (s_init := stPre)
      hStepNe hInit
      (by
        intro off st hlt hCtx
        exact meanLoopContextInvariant_body_step_exists s0 st X Mean M N
          BLOCK_M BLOCK_N off hCtx)
  have hEq : stFinal = stLoop := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact ⟨final, hFinal, hCtx⟩

def meanPostLoop (N BLOCK_M BLOCK_N : Nat) : List Stmt :=
  [ .assign .real [BLOCK_M] "mean"
      (.div NumericDType.real Broadcast.scalarR
        (.reduceSum (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
          (.ref .real [BLOCK_M, BLOCK_N] "_mean"))
        (.const (N : ℝ)))
  , .assign .real [BLOCK_M, 1] "mean"
      (.expandDim (⟨1, by simp⟩ : Fin ([BLOCK_M].length + 1))
        (.ref .real [BLOCK_M] "mean"))
  , .store .real [BLOCK_M, 1]
      (.ptr (.ref .ptr [BLOCK_M, 1] "Mean"))
      (.ref .real [BLOCK_M, 1] "mean")
      (.mask (.ref .bool [BLOCK_M, 1] "row_mask"))
  ]

theorem meanPostLoop_step_alg_post
    (s0 st st' : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (hAcc :
      st.regs .real [BLOCK_M, BLOCK_N] "_mean" =
        some (meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off))
    (hMean :
      st.regs .ptr [BLOCK_M, 1] "Mean" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          (Mean, meanOutOffset s0 BLOCK_M idx.1) })
    (hMask :
      st.regs .bool [BLOCK_M, 1] "row_mask" =
        some { data := fun idx : TileIndex [BLOCK_M, 1] =>
          meanRowActive s0 M BLOCK_M idx.1 })
    (hStep : stepStmts (meanPostLoop N BLOCK_M BLOCK_N) st = some st')
    (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off) :
    mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s0 st' := by
  unfold meanPostLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hAcc, hMean, hMask, Tile.bop,
    Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.div,
    BlockState.setReg, Option.bind, Option.map] at hStep
  subst st'
  simpa [Tile.expandDim, TileShape.dropInsertedIndex, Tile.bop,
    Broadcast.scalarR, NumericDType.div] using
    meanStoreFromExpandedMaskedAccumulator_alg_post s0 _ X Mean M N
      BLOCK_M BLOCK_N off hBLOCK_N hoff

theorem meanPostLoop_step_alg_post_of_context
    (s0 st st' : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (hCtx : meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N off st)
    (hStep : stepStmts (meanPostLoop N BLOCK_M BLOCK_N) st = some st')
    (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off) :
    mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s0 st' := by
  rcases hCtx with ⟨hInv, _hX, hMean, hRow, _hRead⟩
  exact meanPostLoop_step_alg_post s0 st st' X Mean M N BLOCK_M BLOCK_N off
    hInv.2 hMean hRow hStep hBLOCK_N hoff

theorem meanPreLoop_forRange_postLoop_alg_post
    (s0 stPre stLoop stPost : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat)
    (hStepNe : BLOCK_N ≠ 0)
    (hPre :
      stepStmts (meanPreLoop X Mean M N BLOCK_M BLOCK_N) s0 = some stPre)
    (hLoop :
      stepStmt (.forRange "off" 0 N BLOCK_N (meanLoopBody N BLOCK_M BLOCK_N))
        stPre = some stLoop)
    (hPost : stepStmts (meanPostLoop N BLOCK_M BLOCK_N) stLoop = some stPost) :
    mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s0 stPost := by
  obtain ⟨final, hFinal, hCtx⟩ :=
    meanForRange_context_of_preloop s0 stPre stLoop X Mean M N BLOCK_M
      BLOCK_N hStepNe hPre hLoop
  exact meanPostLoop_step_alg_post_of_context s0 stLoop stPost X Mean M N
    BLOCK_M BLOCK_N final hCtx hPost (Nat.pos_of_ne_zero hStepNe) hFinal

def meanProjectedBody
    (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) : List Stmt :=
  meanPreLoop X Mean M N BLOCK_M BLOCK_N ++
    [.forRange "off" 0 N BLOCK_N (meanLoopBody N BLOCK_M BLOCK_N)] ++
    meanPostLoop N BLOCK_M BLOCK_N

theorem meanProjectedBody_alg_post
    (s s' : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat)
    (hStepNe : BLOCK_N ≠ 0)
    (hExec :
      stepStmts (meanProjectedBody X Mean M N BLOCK_M BLOCK_N) s = some s') :
    mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s s' := by
  unfold meanProjectedBody at hExec
  rw [List.append_assoc] at hExec
  rcases (stepStmts.append_some_iff).mp hExec with ⟨stPre, hPre, hRest⟩
  rcases (stepStmts.append_some_iff).mp hRest with ⟨stLoop, hLoopStmt, hPost⟩
  simp [stepStmts] at hLoopStmt
  have hLoop :
      stepStmt (.forRange "off" 0 N BLOCK_N (meanLoopBody N BLOCK_M BLOCK_N))
        stPre = some stLoop := by
    cases hAux :
        stepForRangeAux "off" 0 N BLOCK_N (meanLoopBody N BLOCK_M BLOCK_N)
          stPre with
    | none =>
        simp [hAux] at hLoopStmt
    | some stMid =>
        simp [hAux] at hLoopStmt
        subst hLoopStmt
        simp [hAux]
  exact meanPreLoop_forRange_postLoop_alg_post s stPre stLoop s' X Mean M N
    BLOCK_M BLOCK_N hStepNe hPre hLoop hPost

theorem mean_dim_kernel_toAlg_body
    (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) :
    (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N).toAlgKernel.body =
      meanProjectedBody X Mean M N BLOCK_M BLOCK_N := by
  rfl

theorem mean_dim_kernel_alg_post_of_exec
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s s' : BlockState)
    (hStepNe : BLOCK_N ≠ 0)
    (hExec : exec (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N) s = some s') :
    mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s s' := by
  unfold exec at hExec
  rw [mean_dim_kernel_toAlg_body X Mean M N BLOCK_M BLOCK_N] at hExec
  exact meanProjectedBody_alg_post s s' X Mean M N BLOCK_M BLOCK_N hStepNe hExec

theorem mean_dim_kernel_compute_correct_of_algorithm
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (hStepNe : BLOCK_N ≠ 0) :
    mean_dim_kernel_correct_target X Mean M N BLOCK_M BLOCK_N s := by
  unfold mean_dim_kernel_correct_target
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hi
  exact mean_dim_kernel_alg_post_of_exec X Mean M N BLOCK_M BLOCK_N s s'
    hStepNe hExec i hi

/-- Algorithm-layer correctness for the mean reduction kernel. -/
theorem mean_dim_kernel_correct
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (hStepNe : BLOCK_N ≠ 0) :
    mean_dim_kernel_correct_target X Mean M N BLOCK_M BLOCK_N s :=
  mean_dim_kernel_compute_correct_of_algorithm X Mean M N BLOCK_M BLOCK_N s hStepNe

/-- Compute-facing correctness for the mean reduction kernel. -/
theorem mean_dim_kernel_compute_correct
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (hStepNe : BLOCK_N ≠ 0) :
    mean_dim_kernel_correct_target X Mean M N BLOCK_M BLOCK_N s :=
  mean_dim_kernel_compute_correct_of_algorithm X Mean M N BLOCK_M BLOCK_N s hStepNe

/-- Per-kernel output summary for `mean_dim_kernel`: the DSL surface lowers to
the algorithm layer, and the masked store to `Mean` is compute-correct — every
active row lane holds the row mean `meanSpec`, out-of-bounds rows are preserved.
The only side condition is `BLOCK_N ≠ 0`. -/
specification mean_dim_kernel_output_summary
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (hStepNe : BLOCK_N ≠ 0) :
    (∃ alg, (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N).toAlgorithm? =
        Except.ok alg) ∧
    mean_dim_kernel_correct_target X Mean M N BLOCK_M BLOCK_N s := by
  refine ⟨?_, ?_⟩
  · simp [mean_dim_kernel, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  · exact mean_dim_kernel_compute_correct X Mean M N BLOCK_M BLOCK_N s hStepNe

/-! ## The `⊨[R]` streaming headline (wave-5 S1 fold genre, single stream)

Everything below is purely additive; the exact stack above is untouched. The
kernel is entirely **cast-free** under `execR R`: the pre-loop and the
`for off` body are register/load statements with no `castFloat` (the
`.to(tl.float32)` erases at the algorithm layer), and the terminal store is
`.real`-typed (`stepStmtR` delegates `.real` writes to the exact semantics).
So every segment collapses verbatim onto the exact stepper and the proven
`meanPreLoop` / `meanLoopContextInvariant` / `meanPostLoop` stack above is
reused unchanged — the `⊨[R]` face adds only the safety walk, the memory
frame, and the stream-lane spec bridge. -/

open scoped VeriTile.Triton.StreamMasked2DKernelIO₁

/-! ### Stream geometry: trip count, lanes, windows -/

/-- Trip count of the `for off in range(0, N, BLOCK_N)` stream:
`⌈N / BLOCK_N⌉`. -/
def meanNumSteps (N BLOCK_N : Nat) : Nat := (N + BLOCK_N - 1) / BLOCK_N

private theorem meanNumSteps_mul_ge (N BLOCK_N : Nat) (hB : BLOCK_N ≠ 0) :
    N ≤ meanNumSteps N BLOCK_N * BLOCK_N := by
  have hBpos : 0 < BLOCK_N := Nat.pos_of_ne_zero hB
  rcases Nat.eq_zero_or_pos N with rfl | hN
  · exact Nat.zero_le _
  · unfold meanNumSteps
    have heq : N + BLOCK_N - 1 = (N - 1) + BLOCK_N := by omega
    rw [heq, Nat.add_div_right _ hBpos]
    have h2 : (N - 1) % BLOCK_N + 1 ≤ BLOCK_N := Nat.mod_lt _ hBpos
    calc N = (N - 1) + 1 := by omega
      _ = (N - 1) / BLOCK_N * BLOCK_N + ((N - 1) % BLOCK_N + 1) := by
          rw [← Nat.add_assoc, Nat.div_add_mod']
      _ ≤ (N - 1) / BLOCK_N * BLOCK_N + BLOCK_N := Nat.add_le_add_left h2 _
      _ = ((N - 1) / BLOCK_N + 1) * BLOCK_N := (Nat.succ_mul _ _).symm

private theorem meanStep_lt_numSteps (N BLOCK_N off : Nat) (hB : BLOCK_N ≠ 0)
    (hoff : off < N) : off / BLOCK_N < meanNumSteps N BLOCK_N := by
  have h2 : off / BLOCK_N * BLOCK_N < meanNumSteps N BLOCK_N * BLOCK_N :=
    Nat.lt_of_le_of_lt (Nat.div_mul_le_self off BLOCK_N)
      (Nat.lt_of_lt_of_le hoff (meanNumSteps_mul_ge N BLOCK_N hB))
  exact Nat.lt_of_mul_lt_mul_right h2

/-- The `X`-stream lane holding row `i`, in-block column `j` of the
`[BLOCK_M, BLOCK_N]` per-step tile (row-major, via the shared `Lane2D`
bridge). -/
def meanXLane (BLOCK_M BLOCK_N : Nat) (i : Fin BLOCK_M) (j : Fin BLOCK_N) :
    Fin (BLOCK_M * BLOCK_N) :=
  Lane2D.encode (i, j, PUnit.unit)

/-- Step `t`, lane `l = (i, j)`'s `X` read address: row `pid·BLOCK_M + i`
(row stride `N`), global column `t·BLOCK_N + j`. -/
def meanReadAddr (N BLOCK_M BLOCK_N p₀ t : Nat)
    (l : Fin (BLOCK_M * BLOCK_N)) : Nat :=
  (p₀ * BLOCK_M + l.val / BLOCK_N) * N + (t * BLOCK_N + l.val % BLOCK_N)

/-- Step `t`, lane `l = (i, j)`'s read-active window: the kernel's
`row_mask and col_mask` — row in `[0, M)`, global column in `[0, N)`. -/
def meanReadActive (M N BLOCK_M BLOCK_N p₀ t : Nat)
    (l : Fin (BLOCK_M * BLOCK_N)) : Prop :=
  p₀ * BLOCK_M + l.val / BLOCK_N < M ∧ t * BLOCK_N + l.val % BLOCK_N < N

/-- Output lane `i`'s terminal write address: `Mean + pid·BLOCK_M + i`. -/
def meanWriteAddr (BLOCK_M p₀ : Nat) (i : Fin BLOCK_M) : Nat :=
  p₀ * BLOCK_M + i.val

/-- The terminal store's write-active window: the kernel's `row_mask`. -/
def meanWriteActive (M BLOCK_M p₀ : Nat) (i : Fin BLOCK_M) : Prop :=
  p₀ * BLOCK_M + i.val < M

/-- The guarded per-step summand of the stream spec: lane `(i, j)` of step
`t` contributes its streamed value inside the column window
`t·BLOCK_N + j < N`, and `0` outside (the kernel's `other=0.0`). -/
noncomputable def meanStreamTerm (N BLOCK_M BLOCK_N : Nat)
    (xs : Fin (meanNumSteps N BLOCK_N) → Fin (BLOCK_M * BLOCK_N) → ℝ)
    (i : Fin BLOCK_M) (t : Fin (meanNumSteps N BLOCK_N)) (j : Fin BLOCK_N) : ℝ :=
  if t.val * BLOCK_N + j.val < N then xs t (meanXLane BLOCK_M BLOCK_N i j) else 0

/-- The stream-level row-mean spec: output lane `i` holds the double fold
`(∑ t, ∑ j, in-window xs) / N` over the whole curried stream. -/
noncomputable def meanStreamSpec (N BLOCK_M BLOCK_N : Nat)
    (xs : Fin (meanNumSteps N BLOCK_N) → Fin (BLOCK_M * BLOCK_N) → ℝ)
    (i : Fin BLOCK_M) : ℝ :=
  (∑ t : Fin (meanNumSteps N BLOCK_N), ∑ j : Fin BLOCK_N,
    meanStreamTerm N BLOCK_M BLOCK_N xs i t j) / (N : ℝ)

/-- **Streaming IO signature** of `mean_dim_kernel` on the single-stream fold
skin (S1: fold + terminal store). Step `t` of the `for off` loop (at
`off = t·BLOCK_N`) reads the masked `[BLOCK_M, BLOCK_N]` `X`-tile; after the
loop one `BLOCK_M`-lane row-mean tile is stored to `Mean` at the `.real`
grid (the kernel's store is untyped/`.real`, so the default
`outDType := .real` applies and the readback is exact). The windows
transcribe the kernel's pointer arithmetic and masks exactly (pid axis 0;
lanes row-major over `[BLOCK_M, BLOCK_N]` via `Lane2D`). -/
def meanKernelIO (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) :
    StreamMasked2DKernelIO₁ where
  kernel := mean_dim_kernel X Mean M N BLOCK_M BLOCK_N
  inp1 := X
  out := Mean
  T := meanNumSteps N BLOCK_N
  B1 := BLOCK_M * BLOCK_N
  C := BLOCK_M
  read1 := fun p₀ _ t l => meanReadAddr N BLOCK_M BLOCK_N p₀ t.val l
  write := fun p₀ _ i => meanWriteAddr BLOCK_M p₀ i
  mask1 := fun p₀ _ t l => meanReadActive M N BLOCK_M BLOCK_N p₀ t.val l
  writeMask := fun p₀ _ i => meanWriteActive M BLOCK_M p₀ i

/-! ### Cast-free collapses and the covered fragment -/

/-- The pre-loop is cast-free: it steps identically under `stepStmtsR R`. -/
private theorem meanPreLoop_castFree (R : RoundingModel) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (t : BlockState) :
    stepStmtsR R (meanPreLoop X Mean M N BLOCK_M BLOCK_N) t
      = stepStmts (meanPreLoop X Mean M N BLOCK_M BLOCK_N) t := by
  simp only [meanPreLoop, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The loop body is cast-free (masked `.real` load, nat index math, a real
add — no `castFloat`, no narrow-float store): it steps identically under
`stepStmtsR R`, so the exact invariant stack transports to `execR`. -/
private theorem meanLoopBody_castFree (R : RoundingModel)
    (N BLOCK_M BLOCK_N : Nat) (t : BlockState) :
    stepStmtsR R (meanLoopBody N BLOCK_M BLOCK_N) t
      = stepStmts (meanLoopBody N BLOCK_M BLOCK_N) t := by
  simp only [meanLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The store tail is cast-free: the reduce/divide/expand assigns carry no
`castFloat`, and the masked store is `.real`-typed, which `stepStmtR`
delegates to the exact write. -/
private theorem meanPostLoop_castFree (R : RoundingModel)
    (N BLOCK_M BLOCK_N : Nat) (t : BlockState) :
    stepStmtsR R (meanPostLoop N BLOCK_M BLOCK_N) t
      = stepStmts (meanPostLoop N BLOCK_M BLOCK_N) t := by
  simp only [meanPostLoop, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-- The full surface sits inside the flat-memory bridge's covered fragment
(`FlattenOk`; the `forRange` clause recurses into the cast-free body). -/
theorem mean_dim_kernel_flattenOk (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) :
    ((mean_dim_kernel X Mean M N BLOCK_M BLOCK_N).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [mean_dim_kernel_toAlg_body]
  simp [meanProjectedBody, meanPreLoop, meanLoopBody, meanPostLoop,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### Segment existence and memory frames

The exact stack above proves values given a successful step; the `⊨[R]`
Hoare triple additionally needs termination and a per-cell frame, so each
segment gets an existence twin and a `mem`-preservation twin here. -/

/-- The pre-loop steps successfully from any state (register-only assigns of
total ops). -/
private theorem meanPreLoop_step_exists (s : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) :
    ∃ st, stepStmts (meanPreLoop X Mean M N BLOCK_M BLOCK_N) s = some st := by
  cases hStep : stepStmts (meanPreLoop X Mean M N BLOCK_M BLOCK_N) s with
  | none =>
      unfold meanPreLoop at hStep
      simp [stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.expandDim,
        TileShape.dropInsertedIndex, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, BlockState.setReg, Option.bind] at hStep
  | some st => exact ⟨st, rfl⟩

/-- The pre-loop touches no memory. -/
private theorem meanPreLoop_step_mem (s st : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat)
    (hStep : stepStmts (meanPreLoop X Mean M N BLOCK_M BLOCK_N) s = some st) :
    st.mem = s.mem := by
  unfold meanPreLoop at hStep
  simp [stepStmts, stepStmt, evalOp.eq_def, Tile.bop, Tile.expandDim,
    TileShape.dropInsertedIndex, Tile.ptrAdd, NumericDType.add,
    NumericDType.mul, BlockState.setReg, Option.bind] at hStep
  subst st
  rfl

/-- One body iteration touches no memory (loads and register updates only). -/
private theorem meanLoopBody_step_mem
    (s0 st st' : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (hCtx : meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N off st)
    (hStep :
      stepStmts (meanLoopBody N BLOCK_M BLOCK_N)
        (st.setReg "off" .nat [] (Tile.scalar off)) = some st') :
    st'.mem = st.mem := by
  rcases hCtx with ⟨hInv, hX, _hMean, hRow, _hRead⟩
  unfold meanLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp.eq_def, hInv.2, hX, hRow, Tile.bop,
    Tile.cop, Tile.expandDim, TileShape.dropInsertedIndex, Tile.ptrAdd,
    NumericDType.add, ComparableDType.lt, Option.bind, meanOutOffset,
    meanRowActive] at hStep
  subst st'
  rfl

/-- The store tail steps successfully from any context-invariant state. -/
private theorem meanPostLoop_step_exists
    (s0 st : BlockState) (X Mean : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (hCtx : meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N off st) :
    ∃ st', stepStmts (meanPostLoop N BLOCK_M BLOCK_N) st = some st' := by
  rcases hCtx with ⟨hInv, _hX, hMean, hRow, _hRead⟩
  cases hStep : stepStmts (meanPostLoop N BLOCK_M BLOCK_N) st with
  | none =>
      unfold meanPostLoop at hStep
      simp [stepStmts, stepStmt, evalOp.eq_def, hInv.2, hMean, hRow,
        Tile.bop, Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.div,
        BlockState.setReg, Option.bind, Option.map] at hStep
  | some st' => exact ⟨st', rfl⟩

/-- Cell-level frame of a `P`-masked exact `writeMem` scatter `foldl`: every
cell not hit by an active lane is untouched (the exact-store sibling of
`BlockState.foldl_writeMemAsR_preserve_masked`, at the `MemCell` layer). -/
private theorem foldl_writeMem_prop_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP, ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))]
        show (if r = region ∧ o = offsetFn hd then _ else s.mem r o) = s.mem r o
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- Frame of the store tail: every cell off the write-active `Mean` window is
untouched. -/
private theorem meanPostLoop_step_frame
    (s0 st st' : BlockState) (X Mean : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (hCtx : meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N off st)
    (hStep : stepStmts (meanPostLoop N BLOCK_M BLOCK_N) st = some st')
    (r : RegionName) (o : Nat)
    (hcond : r ≠ Mean ∨ ∀ i : Fin BLOCK_M, meanRowActive s0 M BLOCK_M i →
      o ≠ meanOutOffset s0 BLOCK_M i) :
    st'.mem r o = st.mem r o := by
  rcases hCtx with ⟨hInv, _hX, hMean, hRow, _hRead⟩
  unfold meanPostLoop at hStep
  simp [stepStmts, stepStmt, evalOp.eq_def, hInv.2, hMean, hRow,
    Tile.bop, Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.div,
    BlockState.setReg, Option.bind, Option.map] at hStep
  subst st'
  refine Eq.trans (foldl_writeMem_prop_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _hk hP hc
  rcases hcond with hne | hno
  · exact hne hc.1.symm
  · exact hno k.1 hP hc.2.symm

/-- **R-postLoop**: from the context invariant at a finished offset, the
`execR R` store tail terminates, writes the row mean at every active lane
(the values of `meanPostLoop_step_alg_post_of_context`, valid under `R` by
the cast-free collapse), and preserves every off-window cell. -/
private theorem mean_postLoopR (R : RoundingModel) (X Mean : RegionName)
    (s0 : BlockState) (M N BLOCK_M BLOCK_N off : Nat) (st : BlockState)
    (hB : 0 < BLOCK_N) (hoff : N ≤ off)
    (hCtx : meanLoopContextInvariant s0 X Mean M N BLOCK_M BLOCK_N off st) :
    ∃ sfin, stepStmtsR R (meanPostLoop N BLOCK_M BLOCK_N) st = some sfin
      ∧ (∀ i : Fin BLOCK_M, meanRowActive s0 M BLOCK_M i →
          sfin.readMem Mean (meanOutOffset s0 BLOCK_M i)
            = meanSpec s0 X N BLOCK_M i)
      ∧ (∀ r o,
          (r ≠ Mean ∨ ∀ i : Fin BLOCK_M, meanRowActive s0 M BLOCK_M i →
            o ≠ meanOutOffset s0 BLOCK_M i) →
          sfin.mem r o = st.mem r o) := by
  rw [meanPostLoop_castFree R N BLOCK_M BLOCK_N st]
  obtain ⟨sfin, hStep⟩ :=
    meanPostLoop_step_exists s0 st X Mean M N BLOCK_M BLOCK_N off hCtx
  refine ⟨sfin, hStep, ?_, ?_⟩
  · intro i hi
    exact meanPostLoop_step_alg_post_of_context s0 st sfin X Mean M N BLOCK_M
      BLOCK_N off hCtx hStep hB hoff i hi
  · intro r o hcond
    exact meanPostLoop_step_frame s0 st sfin X Mean M N BLOCK_M BLOCK_N off
      hCtx hStep r o hcond

/-! ### The rounded Hoare triple (`hrun`) -/

/-- Termination, values and frame of the whole kernel under `execR R`, from an
**arbitrary** launch state: the exact `meanPreLoop` / `forRange_inv` /
`meanPostLoop` stack runs verbatim (cast-free collapses), extended with the
per-segment memory frames. -/
private theorem mean_kernel_runR (R : RoundingModel) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (hStepNe : BLOCK_N ≠ 0) (s₀ : BlockState) :
    ∃ sfin,
      execR R (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N).toAlgKernel s₀
        = some sfin
      ∧ (∀ i : Fin BLOCK_M, meanRowActive s₀ M BLOCK_M i →
          sfin.readMem Mean (meanOutOffset s₀ BLOCK_M i)
            = meanSpec s₀ X N BLOCK_M i)
      ∧ (∀ r o,
          (r ≠ Mean ∨ ∀ i : Fin BLOCK_M, meanRowActive s₀ M BLOCK_M i →
            o ≠ meanOutOffset s₀ BLOCK_M i) →
          sfin.mem r o = s₀.mem r o) := by
  obtain ⟨sPre, hpre⟩ := meanPreLoop_step_exists s₀ X Mean M N BLOCK_M BLOCK_N
  have hP0 : meanLoopContextInvariant s₀ X Mean M N BLOCK_M BLOCK_N 0 sPre :=
    meanLoopContextInvariant_init_of_preloop s₀ sPre X Mean M N BLOCK_M
      BLOCK_N hpre
  have hpreMem : sPre.mem = s₀.mem :=
    meanPreLoop_step_mem s₀ sPre X Mean M N BLOCK_M BLOCK_N hpre
  obtain ⟨final, sLoop, hFor, hFinal, hCtxL, hMemL⟩ :
      ∃ final sLoop,
        stepStmt (.forRange "off" 0 N BLOCK_N (meanLoopBody N BLOCK_M BLOCK_N))
          sPre = some sLoop
        ∧ N ≤ final
        ∧ meanLoopContextInvariant s₀ X Mean M N BLOCK_M BLOCK_N final sLoop
        ∧ sLoop.mem = s₀.mem := by
    obtain ⟨final, sLoop, hFor, hFinal, hP⟩ :=
      forRange_inv (idx := "off") (start := 0) (stop := N) (step := BLOCK_N)
        (body := meanLoopBody N BLOCK_M BLOCK_N)
        (P := fun off stt =>
          meanLoopContextInvariant s₀ X Mean M N BLOCK_M BLOCK_N off stt ∧
            stt.mem = s₀.mem)
        hStepNe ⟨hP0, hpreMem⟩
        (fun off stt hlt hP => by
          obtain ⟨st', hstep, hCtx'⟩ :=
            meanLoopContextInvariant_body_step_exists s₀ stt X Mean M N BLOCK_M
              BLOCK_N off hP.1
          refine ⟨st', hstep, hCtx', ?_⟩
          rw [meanLoopBody_step_mem s₀ stt st' X Mean M N BLOCK_M BLOCK_N off
            hP.1 hstep]
          exact hP.2)
    exact ⟨final, sLoop, hFor, hFinal, hP.1, hP.2⟩
  obtain ⟨sfin, hpostR, hval, hframe⟩ :=
    mean_postLoopR R X Mean s₀ M N BLOCK_M BLOCK_N final sLoop
      (Nat.pos_of_ne_zero hStepNe) hFinal hCtxL
  have hLoopR :
      stepStmtR R (.forRange "off" 0 N BLOCK_N (meanLoopBody N BLOCK_M BLOCK_N))
        sPre = some sLoop := by
    rw [stepStmtR_forRange,
      stepForRangeAuxR_castFree R _ (meanLoopBody_castFree R N BLOCK_M BLOCK_N)
        "off",
      ← stepForRangeAux.forRange_unfold]
    exact hFor
  refine ⟨sfin, ?_, hval, ?_⟩
  · show execR R (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N).toAlgKernel s₀
      = some sfin
    unfold execR
    rw [mean_dim_kernel_toAlg_body]
    unfold meanProjectedBody
    rw [List.append_assoc, stepStmtsR_append,
      meanPreLoop_castFree R X Mean M N BLOCK_M BLOCK_N s₀, hpre,
      Option.bind_some,
      show ([Stmt.forRange "off" 0 N BLOCK_N (meanLoopBody N BLOCK_M BLOCK_N)]
          ++ meanPostLoop N BLOCK_M BLOCK_N)
        = Stmt.forRange "off" 0 N BLOCK_N (meanLoopBody N BLOCK_M BLOCK_N)
          :: meanPostLoop N BLOCK_M BLOCK_N from rfl,
      stepStmtsR_cons_some hLoopR]
    exact hpostR
  · intro r o hcond
    rw [hframe r o hcond, hMemL]

/-! ### The `TraceSafeR` walk -/

/-- Per-iteration `TraceSafeListR` for the loop body: the three index/mask
assigns and the accumulator add are register-only; the masked `X` load's
**active** lanes are exactly the skin's `mask1` window at step
`t = off / BLOCK_N`, in bounds by the `read1` window bounds. -/
private theorem mean_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (X : RegionName) (s0 : BlockState) (M N BLOCK_M BLOCK_N off : Nat)
    (hB : BLOCK_N ≠ 0) (hoffN : off < N) (hmod : off % BLOCK_N = 0)
    (st : BlockState)
    (hAcc : st.regs .real [BLOCK_M, BLOCK_N] "_mean" =
      some (meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off))
    (hX : st.regs .ptr [BLOCK_M, 1] "X" =
      some { data := fun idx : TileIndex [BLOCK_M, 1] =>
        (X, meanOutOffset s0 BLOCK_M idx.1 * N) })
    (hRow : st.regs .bool [BLOCK_M, 1] "row_mask" =
      some { data := fun idx : TileIndex [BLOCK_M, 1] =>
        meanRowActive s0 M BLOCK_M idx.1 })
    (hbX : ∀ (t : Fin (meanNumSteps N BLOCK_N)) (l : Fin (BLOCK_M * BLOCK_N)),
      meanReadActive M N BLOCK_M BLOCK_N (s0.pids 0) t.val l →
      meanReadAddr N BLOCK_M BLOCK_N (s0.pids 0) t.val l < bounds X) :
    Stmt.TraceSafeListR R bounds (meanLoopBody N BLOCK_M BLOCK_N)
      (st.setReg "off" .nat [] (Tile.scalar off)) := by
  simp only [meanLoopBody]
  simp [Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.ActiveR,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    stepStmtR, evalOpR.eq_def, hAcc, hX, hRow,
    Tile.bop, Tile.cop, Tile.expandDim, TileShape.dropInsertedIndex,
    Tile.ptrAdd, NumericDType.add, ComparableDType.lt, Option.bind,
    meanOutOffset, meanRowActive]
  intro i₁ i₂ hrow hcol
  have hb := hbX ⟨off / BLOCK_N, meanStep_lt_numSteps N BLOCK_N off hB hoffN⟩
    (Lane2D.encode (i₁, i₂, PUnit.unit)) (by
      constructor
      · simpa [Lane2D.encode_div] using hrow
      · simpa [Lane2D.encode_mod,
          Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)] using hcol)
  simpa [meanReadAddr, Lane2D.encode_div, Lane2D.encode_mod,
    Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hmod)] using hb

/-- `TraceSafeListR` for the store tail: the reduce/divide/expand assigns are
register-only, and the masked `.real` store's active lanes are the skin's
`writeMask` window. -/
private theorem mean_postLoopSafeR (R : RoundingModel) (bounds : RegionBounds)
    (Mean : RegionName) (s0 : BlockState) (M N BLOCK_M BLOCK_N : Nat)
    (st : BlockState)
    (hAcc : ∃ zT : Tile .real [BLOCK_M, BLOCK_N],
      st.regs .real [BLOCK_M, BLOCK_N] "_mean" = some zT)
    (hMean : st.regs .ptr [BLOCK_M, 1] "Mean" =
      some { data := fun idx : TileIndex [BLOCK_M, 1] =>
        (Mean, meanOutOffset s0 BLOCK_M idx.1) })
    (hRow : st.regs .bool [BLOCK_M, 1] "row_mask" =
      some { data := fun idx : TileIndex [BLOCK_M, 1] =>
        meanRowActive s0 M BLOCK_M idx.1 })
    (hbM : ∀ i : Fin BLOCK_M, meanWriteActive M BLOCK_M (s0.pids 0) i →
      meanWriteAddr BLOCK_M (s0.pids 0) i < bounds Mean) :
    Stmt.TraceSafeListR R bounds (meanPostLoop N BLOCK_M BLOCK_N) st := by
  obtain ⟨zT, hz⟩ := hAcc
  simp only [meanPostLoop]
  simp [Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, MaskOpt.ActiveR,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    stepStmtR, evalOpR.eq_def, hz, hMean, hRow,
    Tile.bop, Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.div,
    Option.bind, Option.map, meanOutOffset, meanRowActive]
  intro i hrow
  simpa [meanWriteAddr] using hbM i hrow

/-- **The `TraceSafeR` walk for the whole kernel**, driven by
`Stmt.forRangeTraceSafeR_inv` over the (launch-state-robust)
`meanLoopContextInvariant`. The two bound groups are the skin's
`read1`/`write` windows. -/
private theorem mean_kernel_traceSafeR (R : RoundingModel)
    (bounds : RegionBounds) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (hStepNe : BLOCK_N ≠ 0) (s : BlockState)
    (hbX : ∀ (t : Fin (meanNumSteps N BLOCK_N)) (l : Fin (BLOCK_M * BLOCK_N)),
      meanReadActive M N BLOCK_M BLOCK_N (s.pids 0) t.val l →
      meanReadAddr N BLOCK_M BLOCK_N (s.pids 0) t.val l < bounds X)
    (hbM : ∀ i : Fin BLOCK_M, meanWriteActive M BLOCK_M (s.pids 0) i →
      meanWriteAddr BLOCK_M (s.pids 0) i < bounds Mean) :
    ((mean_dim_kernel X Mean M N BLOCK_M BLOCK_N).toAlgKernel).TraceSafeR R
      bounds s := by
  unfold Kernel.TraceSafeR
  rw [mean_dim_kernel_toAlg_body]
  unfold meanProjectedBody
  rw [List.append_assoc]
  have hstep : ∀ c stt, c < N →
      meanLoopContextInvariant s X Mean M N BLOCK_M BLOCK_N c stt →
      Stmt.TraceSafeListR R bounds (meanLoopBody N BLOCK_M BLOCK_N)
        (stt.setReg "off" .nat [] (Tile.scalar c)) ∧
      ∃ stt', stepStmtsR R (meanLoopBody N BLOCK_M BLOCK_N)
          (stt.setReg "off" .nat [] (Tile.scalar c)) = some stt' ∧
        meanLoopContextInvariant s X Mean M N BLOCK_M BLOCK_N (c + BLOCK_N)
          stt' := by
    intro c stt hc hP
    refine ⟨mean_bodySafeR R bounds X s M N BLOCK_M BLOCK_N c hStepNe hc
      hP.1.1 stt hP.1.2 hP.2.1 hP.2.2.2.1 hbX, ?_⟩
    obtain ⟨stt', hstep', hP'⟩ :=
      meanLoopContextInvariant_body_step_exists s stt X Mean M N BLOCK_M
        BLOCK_N c hP
    exact ⟨stt', by
      rw [meanLoopBody_castFree R N BLOCK_M BLOCK_N]; exact hstep', hP'⟩
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- pre-loop: register-only assigns, safe at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stt hst s'
    simp only [meanPreLoop, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s1 hs1
    obtain ⟨s1x, hpre⟩ := meanPreLoop_step_exists s X Mean M N BLOCK_M BLOCK_N
    rw [meanPreLoop_castFree R X Mean M N BLOCK_M BLOCK_N s, hpre] at hs1
    obtain rfl := Option.some.inj hs1
    have hP0 : meanLoopContextInvariant s X Mean M N BLOCK_M BLOCK_N 0 s1x :=
      meanLoopContextInvariant_init_of_preloop s s1x X Mean M N BLOCK_M
        BLOCK_N hpre
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- the streaming loop (invariant principle)
      simp only [Stmt.TraceSafeR]
      exact Stmt.forRangeTraceSafeR_inv R bounds "off" N BLOCK_N
        (meanLoopBody N BLOCK_M BLOCK_N)
        (meanLoopContextInvariant s X Mean M N BLOCK_M BLOCK_N) hstep 0 s1x hP0
    · intro s2 hs2
      obtain ⟨final, sLoop, hFor, hFinal, hPL⟩ :=
        forRange_inv (idx := "off") (start := 0) (stop := N) (step := BLOCK_N)
          (body := meanLoopBody N BLOCK_M BLOCK_N)
          (P := meanLoopContextInvariant s X Mean M N BLOCK_M BLOCK_N)
          hStepNe hP0
          (fun off stt hlt hP =>
            meanLoopContextInvariant_body_step_exists s stt X Mean M N BLOCK_M
              BLOCK_N off hP)
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _
          (meanLoopBody_castFree R N BLOCK_M BLOCK_N) "off",
        ← stepForRangeAux.forRange_unfold, hFor] at hs2
      obtain rfl := Option.some.inj hs2
      rcases hPL with ⟨hInvL, _hXL, hMeanL, hRowL, _hReadL⟩
      exact mean_postLoopSafeR R bounds Mean s M N BLOCK_M BLOCK_N sLoop
        ⟨_, hInvL.2⟩ hMeanL hRowL hbM

/-! ### The stream-lane spec bridge -/

/-- Under the stream pin, the stream-level double fold at an active output
lane **is** the exact per-row spec `meanSpec`: the guarded double sum
re-blocks to the flat column range (`StreamLane.sum_range_mul`), the
out-of-window guards drop (`N ≤ T·BLOCK_N`), and the flat range sum is the
`Fin N` universe sum. -/
private theorem meanStreamSpec_eq_meanSpec (X : RegionName) (s₀ : BlockState)
    (M N BLOCK_M BLOCK_N : Nat) (hB : BLOCK_N ≠ 0)
    (xs : Fin (meanNumSteps N BLOCK_N) → Fin (BLOCK_M * BLOCK_N) → ℝ)
    (hx : ∀ (t : Fin (meanNumSteps N BLOCK_N)) (l : Fin (BLOCK_M * BLOCK_N)),
      meanReadActive M N BLOCK_M BLOCK_N (s₀.pids 0) t.val l →
      s₀.readMem X (meanReadAddr N BLOCK_M BLOCK_N (s₀.pids 0) t.val l)
        = xs t l)
    (i : Fin BLOCK_M) (hi : meanRowActive s₀ M BLOCK_M i) :
    meanStreamSpec N BLOCK_M BLOCK_N xs i = meanSpec s₀ X N BLOCK_M i := by
  unfold meanStreamSpec meanSpec
  congr 1
  have hterm : ∀ (t : Fin (meanNumSteps N BLOCK_N)) (j : Fin BLOCK_N),
      meanStreamTerm N BLOCK_M BLOCK_N xs i t j
        = if t.val * BLOCK_N + j.val < N then
            meanInpElem s₀ X N BLOCK_M i (t.val * BLOCK_N + j.val)
          else 0 := by
    intro t j
    unfold meanStreamTerm
    by_cases hc : t.val * BLOCK_N + j.val < N
    · rw [if_pos hc, if_pos hc,
        ← hx t (meanXLane BLOCK_M BLOCK_N i j) (by
          constructor
          · simpa [meanXLane, Lane2D.encode_div] using hi
          · simpa [meanXLane, Lane2D.encode_mod] using hc)]
      unfold meanInpElem meanReadAddr meanXLane meanOutOffset
      simp [Lane2D.encode_div, Lane2D.encode_mod]
    · rw [if_neg hc, if_neg hc]
  rw [Finset.sum_congr rfl fun t _ =>
    Finset.sum_congr rfl fun j _ => hterm t j]
  rw [← StreamLane.sum_range_mul (meanNumSteps N BLOCK_N) BLOCK_N
    (fun k => if k < N then meanInpElem s₀ X N BLOCK_M i k else 0)]
  have hsub : Finset.range N ⊆ Finset.range (meanNumSteps N BLOCK_N * BLOCK_N) := by
    intro k hk
    rw [Finset.mem_range] at hk ⊢
    exact Nat.lt_of_lt_of_le hk (meanNumSteps_mul_ge N BLOCK_N hB)
  rw [← Finset.sum_subset hsub
    (fun k _ hk => by
      rw [if_neg (by simpa using hk)])]
  rw [Finset.sum_congr rfl
    (fun k hk => if_pos (Finset.mem_range.mp hk)),
    sum_range_eq_sum_fin N (fun k => meanInpElem s₀ X N BLOCK_M i k)]

/-! ### The headline -/

/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre; single-stream
skin).** For every rounding model `R`, the faithful `mean_dim_kernel`
surface implements, on its `StreamMasked2DKernelIO₁` signature, the **ideal
ℝ row-mean fold** over the streamed masked tiles: output lane `i` holds
`(∑ t, ∑ j, in-window X-tile[t](i, j)) / N` — the spec `f` is exact real
arithmetic; the kernel is entirely cast-free and its store is `.real`-typed,
so the skin's readback contract at the default `outDType := .real` grid is
exact for every `R` (`R.round .real = id`): this is the exact streaming
genre carried on the single rounding surface.

Layer map: the pre-loop, the whole `for off` loop and the store tail are
cast-free, so under `execR R` they collapse verbatim onto the exact stepper
and the proven `meanPreLoop` / `meanLoopContextInvariant` / `forRange_inv` /
`meanPostLoop` stack above is reused unchanged; the `⊨[R]` face adds only
the `TraceSafeR` walk, the memory frame, and the stream-lane spec bridge.

The single hypothesis `hStepNe : BLOCK_N ≠ 0` is truth-forced: it is the
exact stack's own side condition (`mean_dim_kernel_output_summary`) — with
`BLOCK_N = 0` and `N > 0` the `for off in range(0, N, 0)` loop does not
terminate, `execR` returns `none`, and the statement is false. The output
window `pid·BLOCK_M + i` on `Fin BLOCK_M` is injective outright
(`meanOutOffset_injective_col1`), so no width hypothesis is needed. -/
specification mean_dim_kernel_io_correctness (R : RoundingModel)
    (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat)
    (hStepNe : BLOCK_N ≠ 0) :
    meanKernelIO X Mean M N BLOCK_M BLOCK_N ⊨[R] fun _ _ xs i =>
      meanStreamSpec N BLOCK_M BLOCK_N xs i := by
  refine StreamMasked2DKernelIO₁.ImplementsR.intro _ ?_ ?_ ?_
  · exact mean_dim_kernel_flattenOk X Mean M N BLOCK_M BLOCK_N
  · -- safety walk
    intro bounds s xs _hx hbr1 hbw
    simp only [meanKernelIO] at hbr1 hbw ⊢
    exact mean_kernel_traceSafeR R bounds X Mean M N BLOCK_M BLOCK_N hStepNe s
      hbr1 hbw
  · -- the rounded Hoare triple
    intro s₀ xs _hundef hx
    simp only [meanKernelIO] at hx ⊢
    obtain ⟨sfin, hexec, hval, hframe⟩ :=
      mean_kernel_runR R X Mean M N BLOCK_M BLOCK_N hStepNe s₀
    refine ⟨sfin, hexec, ?_, ?_⟩
    · intro j hj
      have hjrow : meanRowActive s₀ M BLOCK_M j := hj
      rw [BlockState.readMemAs_real,
        show meanWriteAddr BLOCK_M (s₀.pids 0) j = meanOutOffset s₀ BLOCK_M j
          from rfl,
        hval j hjrow,
        ← meanStreamSpec_eq_meanSpec X s₀ M N BLOCK_M BLOCK_N hStepNe xs hx j
          hjrow]
      simp
    · intro r o hcond
      refine hframe r o ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · exact Or.inr fun i hi => hno i hi

end VeriTile.Bench.TritonBenchG.MeanReduction
