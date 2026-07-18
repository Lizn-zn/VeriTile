import VeriTile.Triton

/-!
# `token_softmax_bloom` — strict per-kernel correctness

`_fwd_kernel_token_softmax` is the LightLLM/Bloom token-attention softmax:
program `(cur_batch, cur_head)` loads the two per-batch **metadata scalars**
`cur_batch_seq_len = B_Seqlen[cur_batch]` and
`cur_batch_in_all_start_index = B_Start_Loc[cur_batch]`, loads one row of
`Logics` (the batch's slice given by those scalars, masked with `-inf` past
`cur_batch_seq_len`), computes the numerically-stable softmax
(`row - max`, `exp`, `/ sum`), and stores the result into `Prob_Out`, masked by
`col_offsets < cur_batch_seq_len`. (Kernel body is identical to the LLaMA
variant; only the Python test shapes differ.)

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_token_softmax[(batch, head_num)](...)`, the
grid over `(batch, head)`, the host `BLOCK_SIZE = next_power_of_2(max_input_len)`
choice, `num_warps`, and how the runtime composes per-program writes into
`Prob_Out`) is the *trusted boundary*, not a proof obligation here. Because the
program ids `cur_batch`/`cur_head` are universally quantified (via `BlockState`),
the per-program statements cover every program of the grid.

## Proof architecture

```
token_softmax_bloom_correctness              ← TOP SPECIFICATION (tokenSoftmaxIO ⊨ tokenSoftmaxSpecPure)
  ├─ token_softmax_surface_flattenOk         bridge fragment membership
  ├─ token_softmax_surface_traceSafe         safety walk: slot cells + metadata-window lanes
  └─ token_softmax_surface_region_run        region-model metadata-driven masked triple
       ├─ token_softmax_surface_exec_isSome  termination
       ├─ token_softmax_surface_correct      ← algorithm-layer readback per lane
       │    └─ tokenSoftmaxSpec_eq_pure      active-lane congruence: state row → pure (m₂, xs)
       └─ token_softmax_surface_frame        masked scatter frame
```

The headline is stated on the kernel's **metadata-driven IO signature**
`tokenSoftmaxIO` (`MetaMasked2DKernelIO₁`, the metadata genre: windows and
masks that *read memory*). The two per-program `.nat` scalar slots
`B_Start_Loc[pid₀]` / `B_Seqlen[pid₀]` are declared as slot cells
(`mwin1 = mwin2 = pid₀`), and their loaded values enter the headline as
honest named binders: `m₁` (the batch's start index) parametrizes the
`Logics`/`Prob_Out` data windows `pid₁ * stride_*_h + (m₁ + j) * stride_*_bs`,
and `m₂` (the batch's sequence length) both the active-lane mask `j < m₂` and
the spec value. `⊨` (`MetaMasked2DKernelIO₁.Implements`) pins `m₁`/`m₂` to
the slot cells by `readMemValue .nat` preconditions (the ghost-variable
discipline): for every disjoint flat placement of the four buffers, every
program id whose slot cells and active window lanes are in bounds, and every
launch state whose slots hold `m₁`/`m₂` and whose active `Logics` window
holds `xs`, the translated pointer kernel terminates, every active `Prob_Out`
lane `j` holds the pure stable softmax `tokenSoftmaxSpecPure m₂ xs j`, and
every other memory cell is unchanged.

The `Prob_Out`-offset injectivity side condition `hOutInj` (discharged by the
trusted host layout for any concrete shape) is taken as a hypothesis,
quantified over the head id and the loaded start index. `0 < BLOCK_SIZE` is
required: the kernel's `max` reduce (like `Finset.sup'`) is only defined on
non-empty tiles.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float, so the `-inf` masking and
the `exp`/division are real-valued); `num_warps` is not modeled. The
`(...).to(tl.float32)` cast reduces to the identity at the algorithm layer
(post-erasure all dtypes unify to `ℝ`).

## Genuine closed-form spec (banked recipes)

`tokenSoftmaxSpecPure m₂ xs j` is the genuine, self-contained stable-softmax
value at a lane, built **only** from the loaded sequence length and the input
row: input tile lane `k` holds `xs k` for `k < m₂` and `⊥` otherwise (matching
`other=-inf`), then reduceMax-shift, `exp`, `/ reduceSum`.
`token_softmax_surface_correct` closes the whole per-lane chain `row →
reduceMax → sub → exp → reduceSum → div → masked store` against the
state-coupled `tokenSoftmaxSpec` (decoded in one `simp` that unfolds the
`tl.max`/`tl.sum` reductions through `Tile.reduceMaxDrop`/`reduceSumDrop`,
then reads back through the injective `probOffset` scatter), and
`tokenSoftmaxSpec_eq_pure` transports it to the pure spec by active-lane
congruence: the pins give `s₀.readMem Logics (read …) = xs j` at the active
lanes, and inactive lanes are `none`/`⊥` in both tiles.

The secondary `token_softmax_final_store_slice` kernel (the proof-oriented
store slice) keeps its own algorithm-layer lemmas below, unchanged.
-/

namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom

open VeriTile.Triton
open scoped VeriTile.Triton.MetaMasked2DKernelIO₁

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `token_softmax_bloom.py`'s
`_fwd_kernel_token_softmax`.

The metadata buffers are typed Nat regions so their `tl.load` calls do not need
extra `dtype=` kwargs. -/
def token_softmax_surface
    (Logics : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  row = (tl.load(Logics + cur_head * $(stride_logic_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_logic_bs),
    mask=col_offsets < cur_batch_seq_len, other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(Prob_Out + cur_head * $(stride_prob_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_prob_bs),
    softmax_output, mask=col_offsets < cur_batch_seq_len)
}

/-- Proof-oriented final probability store slice of `token_softmax_bloom.py`'s
`_fwd_kernel_token_softmax`.

The full kernel computes the row softmax with max/exp/sum reductions. This
slice starts from a precomputed `Softmax` region and proves the masked
destination writeback using `B_Start_Loc` and `B_Seqlen`. -/
def token_softmax_final_store_slice
    (Softmax : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Prob_Out : RegionName)
    (stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  softmax_output = tl.load(Softmax + cur_head * $(stride_softmax_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_softmax_bs),
    mask=col_offsets < cur_batch_seq_len, other=0.0)
  tl.store(Prob_Out + cur_head * $(stride_prob_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_prob_bs),
    softmax_output, mask=col_offsets < cur_batch_seq_len)
}

def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)

def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

def tokenIndex (s : BlockState) (B_Start_Loc : RegionName) (i : Fin BLOCK_SIZE) : Nat :=
  startLoc s B_Start_Loc + i.val

def active
    (s : BlockState) (B_Seqlen : RegionName) (i : Fin BLOCK_SIZE) : Prop :=
  i.val < seqLen s B_Seqlen

instance activeDecidable
    (s : BlockState) (B_Seqlen : RegionName) (i : Fin BLOCK_SIZE) :
    Decidable (active s B_Seqlen i) := by
  unfold active
  infer_instance

def softmaxOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_softmax_h stride_softmax_bs : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_softmax_h + tokenIndex s B_Start_Loc i * stride_softmax_bs

def probOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_prob_h stride_prob_bs : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_prob_h + tokenIndex s B_Start_Loc i * stride_prob_bs

def logicOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_logic_h stride_logic_bs : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_logic_h + tokenIndex s B_Start_Loc i * stride_logic_bs

/-- Masked input row for the Python token-softmax path. Inactive lanes are `⊥`,
matching the `other=-float("inf")` load used before the stable softmax. -/
noncomputable def tokenSoftmaxInputTile
    (s : BlockState) (Logics B_Start_Loc B_Seqlen : RegionName)
    (stride_logic_h stride_logic_bs BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < seqLen s B_Seqlen then
        some (s.readMem Logics
          (logicOffset s B_Start_Loc stride_logic_h stride_logic_bs idx.1))
      else none }

/-- Exact stable-softmax value produced at one active token lane. -/
noncomputable def tokenSoftmaxSpec
    (s : BlockState) (Logics B_Start_Loc B_Seqlen : RegionName)
    (stride_logic_h stride_logic_bs BLOCK_SIZE : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  let row :=
    tokenSoftmaxInputTile s Logics B_Start_Loc B_Seqlen stride_logic_h
      stride_logic_bs BLOCK_SIZE
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (i, PUnit.unit))
  | none => 0

noncomputable def softmaxStoreValue
    (s : BlockState) (Softmax B_Start_Loc B_Seqlen : RegionName)
    (stride_softmax_h stride_softmax_bs : Nat) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (if i.val < seqLen s B_Seqlen then
      some (s.readMem Softmax
        (softmaxOffset s B_Start_Loc stride_softmax_h stride_softmax_bs i))
    else some (0.0 : ℝ))

/-- The full Python-path token softmax kernel lowers to the algorithm layer:
the `.to(tl.float32)`, max/exp/sum/div reductions, and masked store are all
translation-supported. The value-level store theorem below remains the
proof-facing entry point for destination correctness. -/
theorem token_softmax_surface_toAlgorithm_supported
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat) :
    (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
        BLOCK_SIZE).toAlgorithm? =
      Except.ok
        (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
          BLOCK_SIZE).toAlgKernel := by
  simp [token_softmax_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Algorithm-layer correctness for the masked token-softmax probability store. -/
theorem token_softmax_final_store_slice_correct
    (Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)) :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := probOffset s B_Start_Loc stride_prob_h stride_prob_bs i
      (exec (token_softmax_final_store_slice Softmax B_Start_Loc B_Seqlen Prob_Out
            stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
            BLOCK_SIZE) s).map (·.readMem Prob_Out outAddr)
        = some (if active s B_Seqlen i then
            softmaxStoreValue s Softmax B_Start_Loc B_Seqlen
              stride_softmax_h stride_softmax_bs i
          else s.readMem Prob_Out outAddr) := by
  intro i
  simp [exec, token_softmax_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.mul, ComparableDType.lt, BlockState.readMemValue,
        seqLen, startLoc, tokenIndex, softmaxOffset, probOffset]
  let offsetFn : TileIndex [BLOCK_SIZE] → Nat :=
    fun idx =>
      s.pids 1 * stride_prob_h +
        (s.readMemValue .nat B_Start_Loc (s.pids 0) + idx.1.val) * stride_prob_bs
  let valueFn : TileIndex [BLOCK_SIZE] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if idx.1.val < s.readMemValue .nat B_Seqlen (s.pids 0) then
          some (s.readMem Softmax
            (s.pids 1 * stride_softmax_h +
              (s.readMemValue .nat B_Start_Loc (s.pids 0) + idx.1.val) *
                stride_softmax_bs))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_SIZE] → Prop :=
    fun idx => idx.1.val < s.readMemValue .nat B_Seqlen (s.pids 0)
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : probOffset s B_Start_Loc stride_prob_h stride_prob_bs a =
        probOffset s B_Start_Loc stride_prob_h stride_prob_bs b := by
      simpa [offsetFn, probOffset, tokenIndex, startLoc, BlockState.readMemValue] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem Prob_Out (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BLOCK_SIZE])).readMem Prob_Out
        (offsetFn (i, PUnit.unit)) =
    if active s B_Seqlen i then
      softmaxStoreValue s Softmax B_Start_Loc B_Seqlen stride_softmax_h
        stride_softmax_bs i
    else s.readMem Prob_Out (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : P (i, PUnit.unit)
  · rw [if_pos hi]
    have hraw :
        i.val <
          (match s.readMemTyped TileDType.nat B_Seqlen (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.nat) := by
      simpa [P, BlockState.readMemValue] using hi
    simp [valueFn, P, active, softmaxStoreValue, seqLen, startLoc, tokenIndex,
      softmaxOffset, BlockState.readMemValue, hi, hraw]
    intro hle
    exact False.elim ((not_lt_of_ge hle) hraw)
  · rw [if_neg hi]
    have hraw :
        ¬ i.val <
          (match s.readMemTyped TileDType.nat B_Seqlen (s.pids 0) with
          | some value => value
          | none => BlockState.defaultCarrier TileDType.nat) := by
      simpa [P, BlockState.readMemValue] using hi
    simp [P, active, softmaxStoreValue, seqLen, BlockState.readMemValue, hi, hraw]
    intro hcontr
    exact False.elim (hraw hcontr)

/-- Compute-facing correctness for the masked token-softmax probability store. -/
theorem token_softmax_final_store_slice_compute_correct
    (Softmax B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := token_softmax_final_store_slice Softmax B_Start_Loc B_Seqlen Prob_Out
        stride_softmax_h stride_softmax_bs stride_prob_h stride_prob_bs
        BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s B_Seqlen i)
        (fun i => (Prob_Out, probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)))
      (expected := fun i =>
        softmaxStoreValue s Softmax B_Start_Loc B_Seqlen stride_softmax_h
          stride_softmax_bs i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_softmax_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := token_softmax_final_store_slice_correct Softmax B_Start_Loc
    B_Seqlen Prob_Out stride_softmax_h stride_softmax_bs stride_prob_h
    stride_prob_bs BLOCK_SIZE s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- Algorithm-layer cellwise correctness for the full token-softmax surface. -/
theorem token_softmax_surface_correct
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat)
    (s s' : BlockState)
    (hExec : exec (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE) s
        = some s')
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)) :
    ∀ i : Fin BLOCK_SIZE,
      let outAddr := probOffset s B_Start_Loc stride_prob_h stride_prob_bs i
      s'.readMem Prob_Out outAddr =
        if active s B_Seqlen i then
          tokenSoftmaxSpec s Logics B_Start_Loc B_Seqlen stride_logic_h
            stride_logic_bs BLOCK_SIZE i
        else s.readMem Prob_Out outAddr := by
  intro i
  by_cases hB : 0 < BLOCK_SIZE
  · simp [exec, token_softmax_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
          ComputeKernel.toAlgKernel,
          stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
          Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
          TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
          NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
          ComparableDType.lt, BlockState.readMemValue, hB] at hExec
    subst s'
    -- The decoded store is a masked scatter over `TileShape.allIndices`.
    -- Read back through the injective offset map.
    have hOffsetInj : Function.Injective
        (fun idx : TileIndex [BLOCK_SIZE] =>
          s.pids 1 * stride_prob_h +
            ((match s.readMemTyped TileDType.nat B_Start_Loc (s.pids 0) with
              | some value => value
              | none => BlockState.defaultCarrier TileDType.nat) + idx.1.val) *
              stride_prob_bs) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hab
      have habFin : probOffset s B_Start_Loc stride_prob_h stride_prob_bs a =
          probOffset s B_Start_Loc stride_prob_h stride_prob_bs b := by
        simpa [probOffset, tokenIndex, startLoc, BlockState.readMemValue] using hab
      obtain rfl : a = b := hOutInj habFin
      rfl
    simp only [probOffset, tokenIndex, startLoc, BlockState.readMemValue]
    erw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
    simp only [active, seqLen, BlockState.readMemValue, probOffset, tokenIndex, startLoc]
    -- Both sides are guarded by the same `↑i < seqLen` test; the else-branches are
    -- syntactically equal and the then-branches (computed softmax value vs spec)
    -- agree lane-wise.  Reduce to the then-branch equality.
    refine if_congr Iff.rfl ?_ rfl
    simp [tokenSoftmaxSpec, tokenSoftmaxInputTile, seqLen,
      BlockState.readMemValue, logicOffset, tokenIndex, startLoc,
      Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
      Tile.bop, Tile.uop, NumericDType.sub, NumericDType.div, hB]
    congr
  · exact False.elim (hB (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Exact stable-softmax value at lane `j`, as a **pure** function of the
loaded sequence length `m₂` and the input row `xs`: input tile lane `k` holds
`xs k` when `k < m₂` and `⊥` otherwise (matching `other=-float("inf")`), then
reduceMax-shift, `exp`, `/ reduceSum`. This is `tokenSoftmaxSpec` with the
state-coupled row replaced by the named binders of the `⊨` headline. -/
noncomputable def tokenSoftmaxSpecPure (m₂ : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) (j : Fin BLOCK_SIZE) : ℝ :=
  let row : Tile .real [BLOCK_SIZE] :=
    { data := fun idx => if idx.1.val < m₂ then some (xs idx.1) else none }
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (j, PUnit.unit))
  | none => 0

/-- Active-lane congruence: once the loaded scalars are pinned, the
state-coupled `tokenSoftmaxSpec` equals the pure `tokenSoftmaxSpecPure` — the
kernel's `-inf`-padded input tile and the pure `(m₂, xs)` tile agree lane-wise
(active lanes by the `xs` pin, inactive lanes are `none`/`⊥` in both). -/
theorem tokenSoftmaxSpec_eq_pure
    (s : BlockState) (Logics B_Start_Loc B_Seqlen : RegionName)
    (stride_logic_h stride_logic_bs BLOCK_SIZE : Nat)
    (m₂ : Nat) (xs : Fin BLOCK_SIZE → ℝ)
    (hm₂ : seqLen s B_Seqlen = m₂)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < m₂ →
      s.readMem Logics
        (logicOffset s B_Start_Loc stride_logic_h stride_logic_bs j) = xs j)
    (i : Fin BLOCK_SIZE) :
    tokenSoftmaxSpec s Logics B_Start_Loc B_Seqlen stride_logic_h
        stride_logic_bs BLOCK_SIZE i = tokenSoftmaxSpecPure m₂ xs i := by
  have htile : tokenSoftmaxInputTile s Logics B_Start_Loc B_Seqlen
      stride_logic_h stride_logic_bs BLOCK_SIZE
      = ({ data := fun idx =>
            if idx.1.val < m₂ then some (xs idx.1) else none } :
          Tile .real [BLOCK_SIZE]) := by
    unfold tokenSoftmaxInputTile
    congr 1
    funext idx
    rw [hm₂]
    by_cases hj : idx.1.val < m₂
    · simp only [if_pos hj, hx idx.1 hj]
    · simp only [if_neg hj]
  unfold tokenSoftmaxSpec tokenSoftmaxSpecPure
  rw [htile]

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
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
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- Frame half: every memory cell not actively written by the masked output
store — every cell of every region other than `Prob_Out`, and the *inactive*
lanes of the output window itself — is preserved by the run. -/
private theorem token_softmax_surface_frame
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s s1 : BlockState)
    (hExec : exec ((token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
        BLOCK_SIZE).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_SIZE,
      i.val < s.readMemValue .nat B_Seqlen (s.pids 0) →
      ¬(Prob_Out = r ∧
        s.pids 1 * stride_prob_h +
          (s.readMemValue .nat B_Start_Loc (s.pids 0) + i.val) * stride_prob_bs
          = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, token_softmax_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, BlockState.readMemValue, hB] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  exact hmiss k.1 (by simpa [BlockState.readMemValue] using hmk)
    (by simpa [BlockState.readMemValue] using hc)

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- Termination: the kernel executes to completion from any state.
`0 < BLOCK_SIZE` is required because the `max` reduce (like `Finset.sup'`) is
only defined on non-empty axes. -/
private theorem token_softmax_surface_exec_isSome
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (s : BlockState) :
    ∃ s1, exec ((token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
        BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, token_softmax_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        ComputeKernel.toAlgKernel,
        stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop,
        Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
        ComparableDType.lt, BlockState.readMemValue, hB]

/-- **The region-model metadata-driven masked Hoare triple** — termination,
active-lane output values, and frame off the active output lanes, from any
launch state whose slot cells hold the pinned scalars `m₁`/`m₂` and whose
active `Logics` window holds `xs`. This is the `hrun` obligation of the `⊨`
headline; the value half reuses `token_softmax_surface_correct` (fed the
host-layout injectivity `hOutInj` at the loaded start index) and transports
its state-coupled spec to the pure one via `tokenSoftmaxSpec_eq_pure`. -/
theorem token_softmax_surface_region_run
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE)
    (hOutInj : ∀ pid₁ m₁, Function.Injective
      (fun j : Fin BLOCK_SIZE =>
        pid₁ * stride_prob_h + (m₁ + j.val) * stride_prob_bs))
    (s₀ : BlockState) (m₁ m₂ : Nat) (xs : Fin BLOCK_SIZE → ℝ)
    (hm₁ : s₀.readMemValue .nat B_Start_Loc (s₀.pids 0) = m₁)
    (hm₂ : s₀.readMemValue .nat B_Seqlen (s₀.pids 0) = m₂)
    (hx : ∀ j : Fin BLOCK_SIZE, j.val < m₂ →
      s₀.readMem Logics
        (s₀.pids 1 * stride_logic_h + (m₁ + j.val) * stride_logic_bs) = xs j) :
    ∃ s1, exec ((token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
          BLOCK_SIZE).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin BLOCK_SIZE, j.val < m₂ →
          s1.readMem Prob_Out
            (s₀.pids 1 * stride_prob_h + (m₁ + j.val) * stride_prob_bs)
            = tokenSoftmaxSpecPure m₂ xs j)
      ∧ (∀ r o,
          (r ≠ Prob_Out ∨ ∀ j : Fin BLOCK_SIZE, j.val < m₂ →
            o ≠ s₀.pids 1 * stride_prob_h + (m₁ + j.val) * stride_prob_bs) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := token_softmax_surface_exec_isSome Logics B_Start_Loc
    B_Seqlen Prob_Out stride_logic_h stride_logic_bs stride_prob_h
    stride_prob_bs BLOCK_SIZE hB s₀
  -- restate the pins in the file's own vocabulary (definitional)
  have hstart : startLoc s₀ B_Start_Loc = m₁ := hm₁
  have hlen : seqLen s₀ B_Seqlen = m₂ := hm₂
  have haddr : ∀ i : Fin BLOCK_SIZE,
      probOffset s₀ B_Start_Loc stride_prob_h stride_prob_bs i
        = s₀.pids 1 * stride_prob_h + (m₁ + i.val) * stride_prob_bs := by
    intro i
    simp only [probOffset, tokenIndex, hstart]
  have hInj : Function.Injective
      (fun i : Fin BLOCK_SIZE =>
        probOffset s₀ B_Start_Loc stride_prob_h stride_prob_bs i) := by
    simp only [haddr]
    exact hOutInj (s₀.pids 1) m₁
  refine ⟨s1, hs1, fun j hj => ?_, fun r o hcond => ?_⟩
  · have h := token_softmax_surface_correct Logics B_Start_Loc B_Seqlen Prob_Out
      stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
      s₀ s1 hs1 hInj j
    have hact : active s₀ B_Seqlen j := by
      show j.val < seqLen s₀ B_Seqlen
      rw [hlen]
      exact hj
    simp only [if_pos hact] at h
    rw [haddr j] at h
    rw [h]
    refine tokenSoftmaxSpec_eq_pure s₀ Logics B_Start_Loc B_Seqlen
      stride_logic_h stride_logic_bs BLOCK_SIZE m₂ xs hlen (fun k hk => ?_) j
    have haddrL : logicOffset s₀ B_Start_Loc stride_logic_h stride_logic_bs k
        = s₀.pids 1 * stride_logic_h + (m₁ + k.val) * stride_logic_bs := by
      simp only [logicOffset, tokenIndex, hstart]
    rw [haddrL]
    exact hx k hk
  · refine token_softmax_surface_frame Logics B_Start_Loc B_Seqlen Prob_Out
      stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
      hB s₀ s1 hs1 r o (fun i hi hc => ?_)
    rw [hm₂] at hi
    rcases hcond with hne | hno
    · exact hne hc.1.symm
    · refine hno i hi ?_
      rw [← hc.2, hm₁]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- Per-execution safety walk: one computational unfold walks all ten
statements — six are memory-silent (`program_id` ×2, `arange`, the reductions
and register arithmetic) — and reduces the four memory accesses to the
skin-shaped hypotheses: the two **unmasked scalar-slot loads**
(`B_Seqlen[pid₀]`, `B_Start_Loc[pid₀]`) need their slot cells in bounds, and
the masked `Logics` load / `Prob_Out` store need the metadata window
`pid₁ * stride_*_h + (start + j) * stride_*_bs` in bounds at the active lanes
`j < seq_len`, where `start`/`seq_len` are the values sitting in the slot
cells (kernel-level spelling: `s.readMemValue .nat …`). -/
theorem token_softmax_surface_traceSafe
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (bounds : RegionBounds) (s : BlockState)
    (hb₁ : s.pids 0 < bounds B_Start_Loc)
    (hb₂ : s.pids 0 < bounds B_Seqlen)
    (hin : ∀ j : Fin BLOCK_SIZE,
      j.val < s.readMemValue .nat B_Seqlen (s.pids 0) →
      s.pids 1 * stride_logic_h +
        (s.readMemValue .nat B_Start_Loc (s.pids 0) + j.val) * stride_logic_bs
        < bounds Logics)
    (hout : ∀ j : Fin BLOCK_SIZE,
      j.val < s.readMemValue .nat B_Seqlen (s.pids 0) →
      s.pids 1 * stride_prob_h +
        (s.readMemValue .nat B_Start_Loc (s.pids 0) + j.val) * stride_prob_bs
        < bounds Prob_Out) :
    Kernel.TraceSafe bounds
      ((token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
        BLOCK_SIZE).toAlgKernel) s := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  unfold Kernel.TraceSafe
  simp [token_softmax_surface, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    BlockState.readMemValue, hb₁, hb₂]
  exact ⟨fun a ha => hin a ha, fun a ha => hout a ha⟩

/-- The kernel sits inside the flat-memory bridge's covered fragment (scalar
`.nat` slot loads, pointer arithmetic, masked load with `other`, the
`.to(tl.float32)` cast, reductions, masked store). -/
theorem token_softmax_surface_flattenOk
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat) :
    ((token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [token_softmax_surface, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-- `token_softmax_surface`'s **metadata-driven IO signature** — the whole
kernel-specific audit surface of the `⊨` headline (`MetaMasked2DKernelIO₁`,
the metadata genre: the data windows and the mask read two per-program `.nat`
scalar slots):

* `mbuf1 = B_Start_Loc`, `mbuf2 = B_Seqlen` — the scalar slots; program
  `(cur_batch, cur_head) = (pid₀, pid₁)` reads both at cell `pid₀`
  (`mwin1 = mwin2 = pid₀`), yielding the named scalars
  `m₁ = cur_batch_in_all_start_index` and `m₂ = cur_batch_seq_len`;
* `inp = Logics`, `out = Prob_Out` — the data channels, `B = BLOCK_SIZE`;
* `read`/`write` — lane `j` at
  `pid₁ * stride_logic_h + (m₁ + j) * stride_logic_bs` resp.
  `pid₁ * stride_prob_h + (m₁ + j) * stride_prob_bs`: the `m₁`-shifted token
  window of the batch's slice, head-indexed by `pid₁`;
* `mask` — the active lanes `j < m₂` (`col_offsets < cur_batch_seq_len`),
  shared by load and store (`writeMask` defaults to `mask`).

The slot cells, windows, and mask are declared, not parsed from the kernel;
the headline **proves** the kernel's actual slot loads, addressing, and
masking match them. Buffer sizes are not signature content: the headline
quantifies over every allocation whose extents cover the slot cells and the
active lanes. -/
def tokenSoftmaxIO (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs
      BLOCK_SIZE : Nat) : MetaMasked2DKernelIO₁ where
  kernel := token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
    stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
  mbuf1 := B_Start_Loc
  mbuf2 := B_Seqlen
  inp := Logics
  out := Prob_Out
  B := BLOCK_SIZE
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ => pid₀
  read := fun _ pid₁ m₁ _ j =>
    pid₁ * stride_logic_h + (m₁ + j.val) * stride_logic_bs
  write := fun _ pid₁ m₁ _ j =>
    pid₁ * stride_prob_h + (m₁ + j.val) * stride_prob_bs
  mask := fun _ _ _ m₂ j => j.val < m₂

set_option maxHeartbeats 1600000 in
/-- **The headline**: `token_softmax_surface` implements the pure stable
softmax on its metadata-driven IO signature — for every disjoint flat
placement of the four buffers, every program `(cur_batch, cur_head)` whose
slot cells and active window lanes are in bounds, and every launch state
whose `B_Start_Loc`/`B_Seqlen` slot cells hold `m₁`/`m₂` and whose active
`Logics` window holds `xs`, the translated pointer kernel terminates, every
active (`j < m₂`) `Prob_Out` lane holds `tokenSoftmaxSpecPure m₂ xs j`, and
every other memory cell is unchanged. The loaded scalars `m₁`/`m₂` are honest
named binders of the spec, pinned to the slot cells by the skin's
`readMemValue .nat` preconditions. `hOutInj` is the trusted host-layout
side condition (output-offset injectivity at any head id and loaded start
index); `0 < BLOCK_SIZE` is required because the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof:
`MetaMasked2DKernelIO₁.Implements.intro` assembles the region-model triple
with the flat-memory bridge side conditions. -/
specification token_softmax_bloom_correctness
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE)
    (hOutInj : ∀ pid₁ m₁, Function.Injective
      (fun j : Fin BLOCK_SIZE =>
        pid₁ * stride_prob_h + (m₁ + j.val) * stride_prob_bs)) :
    tokenSoftmaxIO Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
      ⊨ fun _ _ _ m₂ xs j => tokenSoftmaxSpecPure m₂ xs j := by
  refine MetaMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact token_softmax_surface_flattenOk Logics B_Start_Loc B_Seqlen Prob_Out
      stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
  · intro bounds s m₁ m₂ hm₁ hm₂ hb₁ hb₂ hin hout
    -- restate the skin-projection facts in the kernel-level spelling
    -- (definitional), so `rw` sees the walk lemma's patterns
    have hm₁' : s.readMemValue .nat B_Start_Loc (s.pids 0) = m₁ := hm₁
    have hm₂' : s.readMemValue .nat B_Seqlen (s.pids 0) = m₂ := hm₂
    have hin' : ∀ j : Fin BLOCK_SIZE, j.val < m₂ →
        s.pids 1 * stride_logic_h + (m₁ + j.val) * stride_logic_bs
          < bounds Logics := hin
    have hout' : ∀ j : Fin BLOCK_SIZE, j.val < m₂ →
        s.pids 1 * stride_prob_h + (m₁ + j.val) * stride_prob_bs
          < bounds Prob_Out := hout
    refine token_softmax_surface_traceSafe Logics B_Start_Loc B_Seqlen Prob_Out
      stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
      hB bounds s hb₁ hb₂ (fun j hj => ?_) (fun j hj => ?_)
    · rw [hm₂'] at hj
      rw [hm₁']
      exact hin' j hj
    · rw [hm₂'] at hj
      rw [hm₁']
      exact hout' j hj
  · intro s₀ m₁ m₂ xs hm₁ hm₂ hx
    have hm₁' : s₀.readMemValue .nat B_Start_Loc (s₀.pids 0) = m₁ := hm₁
    have hm₂' : s₀.readMemValue .nat B_Seqlen (s₀.pids 0) = m₂ := hm₂
    have hx' : ∀ j : Fin BLOCK_SIZE, j.val < m₂ →
        s₀.readMem Logics
          (s₀.pids 1 * stride_logic_h + (m₁ + j.val) * stride_logic_bs)
          = xs j := hx
    exact token_softmax_surface_region_run Logics B_Start_Loc B_Seqlen Prob_Out
      stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
      hB hOutInj s₀ m₁ m₂ xs hm₁' hm₂' hx'

end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom
