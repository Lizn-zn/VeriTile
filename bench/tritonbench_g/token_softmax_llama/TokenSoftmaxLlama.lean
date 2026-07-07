import VeriTile.Triton

/-!
# `token_softmax_llama` — strict per-kernel correctness

`_fwd_kernel_token_softmax` is the LightLLM/LLaMA token-attention softmax:
program `(cur_batch, cur_head)` loads one row of `Logics` (the `cur_batch`'s
slice given by `B_Start_Loc`/`B_Seqlen`, masked with `-inf` past
`cur_batch_seq_len`), computes the numerically-stable softmax
(`row - max`, `exp`, `/ sum`), and stores the result into `Prob_Out`, masked by
`col_offsets < cur_batch_seq_len`. (Kernel body is identical to the Bloom
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
token_softmax_llama_output_summary_general            ← TOP THEOREM (symbolic dims)
  ├─ token_softmax_surface_toAlgorithm_supported         surface lowers (general)
  └─ token_softmax_surface_spec_compute_correct          ← ComputeCorrect = tokenSoftmaxSpec
       └─ token_softmax_surface_correct  ← full-body decode = closed-form spec, per lane
```

The top theorem is dimension-general (symbolic strides + `BLOCK_SIZE`); the
`Prob_Out` slice offset-injectivity is taken as an explicit hypothesis.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float, so the `-inf` masking and
the `exp`/division are real-valued); `num_warps` is not
modeled. The `(...).to(tl.float32)` cast reduces to the identity at the algorithm
layer (post-erasure all dtypes unify to `ℝ`). The verified statement is scoped to
the **final masked probability store** into `Prob_Out`: the expected value is the
genuine closed-form `tokenSoftmaxSpec` (reduceMax-shift, `exp`, `/ reduceSum` over
the masked input row) read off at each `probOffset`, with the write-mask
`active s B_Seqlen i` (`col_offsets < cur_batch_seq_len`). The max/exp/sum
reduction body produces that value; the side condition is the offset-injectivity
of the `Prob_Out` slice at the test shapes. Out-of-range lanes are preserved
(mask=false ⇒ no store).

## Genuine closed-form spec (banked recipes)

`tokenSoftmaxSpec` is the genuine, self-contained stable-softmax value at a lane
(reduceMax-shift, `exp`, `/ reduceSum`) over the masked input row
`tokenSoftmaxInputTile` (inactive lanes `⊥`, matching `other=-inf`): the
`-inf`-masked region load lowers to the `⊥`-padded tile, and the `castFloat(load
…)` `row` register equals `tokenSoftmaxInputTile` modulo the model-identity real
cast.

`token_softmax_surface_correct` closes the whole chain `row → reduceMax → sub →
exp → reduceSum → div → masked store = tokenSoftmaxSpec`: the entire body is
decoded in one `simp` that unfolds the `tl.max`/`tl.sum` reductions through
`Tile.reduceMaxDrop`/`reduceSumDrop` so the `Finset.univ.sup'`/`Finset.univ.sum`
are exposed directly (sidestepping the `Fin` reduce-axis proof-irrelevance
friction), then reads back through the injective `probOffset` scatter and matches
the closed form lane-wise.  `token_softmax_surface_spec_compute_correct` lifts
that to `ComputeCorrect.Realizes`, and `token_softmax_llama_output_summary_general`
states it over symbolic dims — fully self-contained, no surface self-reference.
-/

namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `token_softmax_llama.py`'s
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

/-- Proof-oriented final probability store slice of `token_softmax_llama.py`'s
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
    ComputeCorrect.Realizes
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

/-- Compute-facing correctness for the full token-softmax surface kernel: every
active lane of `Prob_Out` holds the genuine stable-softmax value
`tokenSoftmaxSpec`, inactive lanes are preserved. -/
theorem token_softmax_surface_spec_compute_correct
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)) :
    ComputeCorrect.Realizes
      (kernel := token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s B_Seqlen i)
        (fun i => (Prob_Out, probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)))
      (expected := fun i =>
        tokenSoftmaxSpec s Logics B_Start_Loc B_Seqlen stride_logic_h
          stride_logic_bs BLOCK_SIZE i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · exact token_softmax_surface_toAlgorithm_supported Logics B_Start_Loc
      B_Seqlen Prob_Out stride_logic_h stride_logic_bs stride_prob_h
      stride_prob_bs BLOCK_SIZE
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := token_softmax_surface_correct Logics B_Start_Loc B_Seqlen Prob_Out
    stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE
    s s' hExec hOutInj i
  simpa [hActive] using h

/-- Public dimension-general coverage summary: for symbolic strides and
`BLOCK_SIZE`, the full stable-softmax surface lowers to the algorithm layer, and
the masked `Prob_Out` store is compute-correct — every active lane holds the
genuine closed-form stable-softmax value `tokenSoftmaxSpec` (read off the input
memory), inactive lanes are preserved. The `Prob_Out` slice offset-injectivity
that holds at the Python test shapes is taken as an explicit hypothesis here. -/
theorem token_softmax_llama_output_summary_general
    (Logics B_Start_Loc B_Seqlen Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)) :
    (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE).toAlgorithm? =
      Except.ok
        (token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
          stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE).toAlgKernel ∧
    (ComputeCorrect.Realizes
      (kernel := token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s B_Seqlen i)
        (fun i : Fin BLOCK_SIZE =>
          (Prob_Out, probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)))
      (expected := fun i : Fin BLOCK_SIZE =>
        tokenSoftmaxSpec s Logics B_Start_Loc B_Seqlen stride_logic_h
          stride_logic_bs BLOCK_SIZE i)) := by
  refine ⟨token_softmax_surface_toAlgorithm_supported Logics B_Start_Loc B_Seqlen
    Prob_Out stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE, ?_⟩
  exact token_softmax_surface_spec_compute_correct Logics B_Start_Loc B_Seqlen
    Prob_Out stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE s
    hOutInj

end VeriTile.Bench.TritonBenchG.TokenSoftmaxLlama
