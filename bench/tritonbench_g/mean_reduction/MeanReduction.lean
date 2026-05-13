import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common

namespace VeriTile.Bench.TritonBenchG.MeanReduction

open VeriTile.Triton VeriTile.Examples

/-- Faithful transcription of `mean_reduction.py`'s `mean_dim_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `[:, None]` / `[None, :]` dimension annotations preserved.
- Python `BLOCK_M` / `BLOCK_N: tl.constexpr` → Lean `Nat` parameters.

Known proof blocker: see `bench/tritonbench_g/proof_blockers.md`. The full-row
spec below matches the mathematical effect of the `for off in range(...)`
accumulation, but the theorem still needs a loop invariant for that accumulation.
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

noncomputable def meanOneColSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M : Nat) (i : Fin BLOCK_M) : ℝ :=
  s.readMem X (meanOutOffset s BLOCK_M i * N) / (N : ℝ)

noncomputable def meanSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M : Nat)
    (i : Fin BLOCK_M) : ℝ :=
  ((Finset.univ : Finset (Fin N)).sum fun j =>
    s.readMem X (meanOutOffset s BLOCK_M i * N + j.val)) / (N : ℝ)

noncomputable def meanLanePrefix
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) (j : Fin BLOCK_N) : ℝ :=
  ((Finset.range off).filter fun col => col < N ∧ col % BLOCK_N = j.val).sum
    fun col => s.readMem X (meanOutOffset s BLOCK_M i * N + col)

noncomputable def meanAccumulatorSpec
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat) :
    Tile .real [BLOCK_M, BLOCK_N] :=
  { data := fun idx =>
      some (meanLanePrefix s X N BLOCK_M BLOCK_N off idx.1 idx.2.1) }

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
          s.readMem X (meanOutOffset s BLOCK_M idx.1 * N +
            (off + idx.2.1.val))
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

theorem meanAccumulatorSpec_zero
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N : Nat) :
    meanAccumulatorSpec s X N BLOCK_M BLOCK_N 0 =
      { data := fun _ : TileIndex [BLOCK_M, BLOCK_N] => some 0 } := by
  ext idx
  simp [meanAccumulatorSpec]

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

theorem meanChunkLane_mem_next
    (N off BLOCK_N : Nat) (j : Fin BLOCK_N)
    (hOff : off % BLOCK_N = 0) (hjN : off + j.val < N) :
    off + j.val ∈ (Finset.range (off + BLOCK_N)).filter
      (fun col => col < N ∧ col % BLOCK_N = j.val) := by
  simp [hjN]
  exact meanChunkLane_mod off BLOCK_N j hOff

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
          s.readMem X (meanOutOffset s BLOCK_M i * N + (off + j.val))
        else
          0 := by
  classical
  let pred : Nat → Prop := fun col => col < N ∧ col % BLOCK_N = j.val
  let f : Nat → ℝ := fun col => s.readMem X (meanOutOffset s BLOCK_M i * N + col)
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

theorem meanAccumulatorSpec_step
    (s : BlockState) (X : RegionName) (N BLOCK_M BLOCK_N off : Nat)
    (hOff : off % BLOCK_N = 0) :
    meanAccumulatorSpec s X N BLOCK_M BLOCK_N (off + BLOCK_N) =
      { data := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          some
            (meanLanePrefix s X N BLOCK_M BLOCK_N off idx.1 idx.2.1 +
              if off + idx.2.1.val < N then
                s.readMem X (meanOutOffset s BLOCK_M idx.1 * N +
                  (off + idx.2.1.val))
              else
                0) } := by
  ext idx
  simp [meanAccumulatorSpec, meanLanePrefix_step s X N BLOCK_M BLOCK_N off
    idx.1 idx.2.1 hOff]

theorem meanMaskedAccumulatorSpec_step
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (hOff : off % BLOCK_N = 0) :
    meanMaskedAccumulatorSpec s X M N BLOCK_M BLOCK_N (off + BLOCK_N) =
      { data := fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          some
            (if meanRowActive s M BLOCK_M idx.1 then
              meanLanePrefix s X N BLOCK_M BLOCK_N off idx.1 idx.2.1 +
                if off + idx.2.1.val < N then
                  s.readMem X (meanOutOffset s BLOCK_M idx.1 * N +
                    (off + idx.2.1.val))
                else
                  0
            else
              0) } := by
  ext idx
  by_cases hrow : meanRowActive s M BLOCK_M idx.1
  · simp [meanMaskedAccumulatorSpec, hrow,
      meanLanePrefix_step s X N BLOCK_M BLOCK_N off idx.1 idx.2.1 hOff]
  · simp [meanMaskedAccumulatorSpec, hrow]

theorem meanMaskedAccumulatorSpec_active
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N])
    (hrow : meanRowActive s M BLOCK_M idx.1) :
    (meanMaskedAccumulatorSpec s X M N BLOCK_M BLOCK_N off).data idx =
      some (meanLanePrefix s X N BLOCK_M BLOCK_N off idx.1 idx.2.1) := by
  simp [meanMaskedAccumulatorSpec, hrow]

theorem meanChunkLoadSpec_active
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N])
    (hrow : meanRowActive s M BLOCK_M idx.1)
    (hcol : off + idx.2.1.val < N) :
    (meanChunkLoadSpec s X M N BLOCK_M BLOCK_N off).data idx =
      some (s.readMem X (meanOutOffset s BLOCK_M idx.1 * N +
        (off + idx.2.1.val))) := by
  simp [meanChunkLoadSpec, hrow, hcol]

theorem meanChunkLoadSpec_inactive
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N])
    (hmask : ¬(meanRowActive s M BLOCK_M idx.1 ∧ off + idx.2.1.val < N)) :
    (meanChunkLoadSpec s X M N BLOCK_M BLOCK_N off).data idx =
      some 0 := by
  simp [meanChunkLoadSpec, hmask]

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

theorem meanMaskedAccumulatorSpec_reduceSum_axis1
    (s : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M) :
    (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length)
      (meanMaskedAccumulatorSpec s X M N BLOCK_M BLOCK_N off)).data
        ((i, PUnit.unit) : TileIndex [BLOCK_M]) =
      some ((Finset.univ : Finset (Fin BLOCK_N)).sum fun j =>
        if meanRowActive s M BLOCK_M i then
          meanLanePrefix s X N BLOCK_M BLOCK_N off i j
        else
          0) := by
  simp [Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, meanMaskedAccumulatorSpec]
  rfl

theorem meanLoopInvariant_register_reduceSum_to_meanSpec
    (s0 st : BlockState) (X : RegionName) (M N BLOCK_M BLOCK_N off : Nat)
    (i : Fin BLOCK_M)
    (hInv : meanLoopInvariant s0 X M N BLOCK_M BLOCK_N off st)
    (hrow : meanRowActive s0 M BLOCK_M i)
    (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off) :
    ∃ acc : Tile .real [BLOCK_M, BLOCK_N],
      st.regs .real [BLOCK_M, BLOCK_N] "_mean" = some acc ∧
        WithBot.unbotD 0
            ((Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length)
              acc).data ((i, PUnit.unit) : TileIndex [BLOCK_M])) / (N : ℝ) =
          meanSpec s0 X N BLOCK_M i := by
  refine ⟨meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off, hInv.2, ?_⟩
  rw [meanMaskedAccumulatorSpec_reduceSum_axis1]
  simpa [meanFromMaskedAccumulatorSpec, hrow] using
    meanFromMaskedAccumulatorSpec_eq_meanSpec s0 X M N BLOCK_M BLOCK_N off i
      hrow hBLOCK_N hoff

def mean_dim_kernel_correct_target
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState) : Prop :=
  ComputeCorrect.Realizes
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

theorem meanOutOffset_injective
    (s : BlockState) (BLOCK_M : Nat) :
    Function.Injective
      (fun idx : TileIndex [BLOCK_M] => meanOutOffset s BLOCK_M idx.1) := by
  intro a b h
  apply Prod.ext
  · apply Fin.ext
    simpa [meanOutOffset] using Nat.add_left_cancel h
  · cases a.2
    cases b.2
    rfl

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

theorem meanStoreFromMaskedAccumulator_alg_post
    (s0 stBase : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (hOffsetInj :
      Function.Injective
        (fun idx : TileIndex [BLOCK_M] => meanOutOffset s0 BLOCK_M idx.1))
    (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off) :
    mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s0
      ((TileShape.allIndices [BLOCK_M]).foldl
        (fun acc idx =>
          if meanRowActive s0 M BLOCK_M idx.1 then
            acc.writeMem Mean (meanOutOffset s0 BLOCK_M idx.1)
              (WithBot.unbotD 0
                ((Tile.reduceSumDrop
                  (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length)
                  (meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off)).data
                    idx) / (N : ℝ))
          else
            acc)
        stBase) := by
  intro i hi
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj
    ((i, PUnit.unit) : TileIndex [BLOCK_M])]
  simp [meanRowActive, hi]
  simpa [TileShape.axisDim, TileShape.insertAxisIndex,
    meanMaskedAccumulatorSpec, meanFromMaskedAccumulatorSpec, meanRowActive, hi] using
    meanFromMaskedAccumulatorSpec_eq_meanSpec s0 X M N BLOCK_M BLOCK_N off i
      (by simpa [meanRowActive] using hi) hBLOCK_N hoff

theorem meanStoreFromMaskedAccumulator_alg_post_default
    (s0 stBase : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off) :
    mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s0
      ((TileShape.allIndices [BLOCK_M]).foldl
        (fun acc idx =>
          if meanRowActive s0 M BLOCK_M idx.1 then
            acc.writeMem Mean (meanOutOffset s0 BLOCK_M idx.1)
              (WithBot.unbotD 0
                ((Tile.reduceSumDrop
                  (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length)
                  (meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off)).data
                    idx) / (N : ℝ))
          else
            acc)
        stBase) :=
  meanStoreFromMaskedAccumulator_alg_post s0 stBase X Mean M N BLOCK_M BLOCK_N
    off (meanOutOffset_injective s0 BLOCK_M) hBLOCK_N hoff

theorem meanLoopInvariant_to_scatter_alg_post
    (s0 st stBase : BlockState) (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N off : Nat)
    (_hInv : meanLoopInvariant s0 X M N BLOCK_M BLOCK_N off st)
    (hBLOCK_N : 0 < BLOCK_N) (hoff : N ≤ off) :
    mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s0
      ((TileShape.allIndices [BLOCK_M]).foldl
        (fun acc idx =>
          if meanRowActive s0 M BLOCK_M idx.1 then
            acc.writeMem Mean (meanOutOffset s0 BLOCK_M idx.1)
              (WithBot.unbotD 0
                ((Tile.reduceSumDrop
                  (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length)
                  (meanMaskedAccumulatorSpec s0 X M N BLOCK_M BLOCK_N off)).data
                    idx) / (N : ℝ))
          else
            acc)
        stBase) :=
  meanStoreFromMaskedAccumulator_alg_post_default s0 stBase X Mean M N
    BLOCK_M BLOCK_N off hBLOCK_N hoff

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
  simp [stepStmts, stepStmt, evalOp, Tile.bop, Tile.expandDim,
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
  simp [stepStmts, stepStmt, evalOp, hAcc, hX, hRow, Tile.bop, Tile.cop,
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
    simp [meanMaskedAccumulatorSpec, meanChunkLoadSpec, meanRowActive, hcol,
      hrow', hread', meanOutOffset]
  · have hmask' :
        ¬(s0.pids 0 * BLOCK_M + idx.1.val < M ∧ off + idx.2.1.val < N) := by
      simpa [meanOutOffset] using hmask
    simp [meanMaskedAccumulatorSpec, meanChunkLoadSpec, meanRowActive, hmask',
      meanOutOffset]
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
  simp [stepStmts, stepStmt, evalOp, hAcc, hX, hRow, Tile.bop, Tile.cop,
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
      simp [stepStmts, stepStmt, evalOp, hInv.2, hX, hRow, Tile.bop, Tile.cop,
        Tile.expandDim, TileShape.dropInsertedIndex, Tile.ptrAdd,
        NumericDType.add, ComparableDType.lt, Option.bind, meanOutOffset,
        meanRowActive] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact meanLoopContextInvariant_step_of_body s0 st st' X Mean M N
        BLOCK_M BLOCK_N off ⟨hInv, hX, hMean, hRow, hRead⟩ hStep

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
  simp [stepStmts, stepStmt, evalOp, hAcc, hMean, hMask, Tile.bop,
    Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.div,
    BlockState.setReg, Option.bind, Option.map] at hStep
  subst st'
  simpa [Tile.expandDim, TileShape.dropInsertedIndex, Tile.bop,
    Broadcast.scalarR, NumericDType.div] using
    meanStoreFromExpandedMaskedAccumulator_alg_post s0 _ X Mean M N
      BLOCK_M BLOCK_N off hBLOCK_N hoff

def meanProjectedBody
    (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) : List Stmt :=
  meanPreLoop X Mean M N BLOCK_M BLOCK_N ++
    [.forRange "off" 0 N BLOCK_N (meanLoopBody N BLOCK_M BLOCK_N)] ++
    meanPostLoop N BLOCK_M BLOCK_N

theorem mean_dim_kernel_toAlg_body
    (X Mean : RegionName) (M N BLOCK_M BLOCK_N : Nat) :
    (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N).toAlgKernel.body =
      meanProjectedBody X Mean M N BLOCK_M BLOCK_N := by
  rfl

theorem mean_dim_kernel_compute_correct_of_algorithm
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (hAlg :
      ∀ s',
        exec (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N) s = some s' →
        mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s s') :
    mean_dim_kernel_correct_target X Mean M N BLOCK_M BLOCK_N s := by
  unfold mean_dim_kernel_correct_target
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro i hi
  exact hAlg s' hExec i hi

/-- Algorithm-layer correctness for the mean reduction kernel. -/
theorem mean_dim_kernel_correct
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (hAlg :
      ∀ s',
        exec (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N) s = some s' →
        mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s s') :
    mean_dim_kernel_correct_target X Mean M N BLOCK_M BLOCK_N s :=
  mean_dim_kernel_compute_correct_of_algorithm X Mean M N BLOCK_M BLOCK_N s hAlg

/-- Compute-facing correctness for the mean reduction kernel. -/
theorem mean_dim_kernel_compute_correct
    (X Mean : RegionName)
    (M N BLOCK_M BLOCK_N : Nat) (s : BlockState)
    (hAlg :
      ∀ s',
        exec (mean_dim_kernel X Mean M N BLOCK_M BLOCK_N) s = some s' →
        mean_dim_kernel_alg_post X Mean M N BLOCK_M BLOCK_N s s') :
    mean_dim_kernel_correct_target X Mean M N BLOCK_M BLOCK_N s :=
  mean_dim_kernel_compute_correct_of_algorithm X Mean M N BLOCK_M BLOCK_N s hAlg

end VeriTile.Bench.TritonBenchG.MeanReduction
