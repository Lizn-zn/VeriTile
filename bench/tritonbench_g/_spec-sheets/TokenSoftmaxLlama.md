# Spec sheet — `bench/tritonbench_g/token_softmax_llama/TokenSoftmaxLlama.lean`

**Python source:** `bench/tritonbench_g/token_softmax_llama/token_softmax_llama.py`

## Public theorem: `token_softmax_llama_output_summary_general`

<details><summary>docstring</summary>

```
/-- Public dimension-general coverage summary: for symbolic strides and
`BLOCK_SIZE`, the full stable-softmax surface lowers to the algorithm layer, and
the masked `Prob_Out` store is compute-correct — every active lane holds the
genuine closed-form stable-softmax value `tokenSoftmaxSpec` (read off the input
memory), inactive lanes are preserved. The `Prob_Out` slice offset-injectivity
that holds at the Python test shapes is taken as an explicit hypothesis here. -/
```
</details>

**Statement:**
```lean
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
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := token_softmax_surface Logics B_Start_Loc B_Seqlen Prob_Out
        stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => active s B_Seqlen i)
        (fun i : Fin BLOCK_SIZE =>
          (Prob_Out, probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)))
      (expected := fun i : Fin BLOCK_SIZE =>
        tokenSoftmaxSpec s Logics B_Start_Loc B_Seqlen stride_logic_h
          stride_logic_bs BLOCK_SIZE i))
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)`
- `fun i : Fin BLOCK_SIZE => active s B_Seqlen i`
- `fun i : Fin BLOCK_SIZE =>
          (Prob_Out, probOffset s B_Start_Loc stride_prob_h stride_prob_bs i)`

**Closed-form spec defs (transitive):** `probOffset`, `token_softmax_surface`, `active`, `tokenSoftmaxSpec`, `tokenIndex`, `seqLen`, `tokenSoftmaxInputTile`, `startLoc`, `logicOffset`

<details><summary><code>probOffset</code></summary>

```lean
def probOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_prob_h stride_prob_bs : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_prob_h + tokenIndex s B_Start_Loc i * stride_prob_bs
```
</details>

<details><summary><code>token_softmax_surface</code></summary>

```
/-- Faithful transcription of `token_softmax_llama.py`'s
`_fwd_kernel_token_softmax`.

The metadata buffers are typed Nat regions so their `tl.load` calls do not need
extra `dtype=` kwargs. -/
```
```lean
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
```
</details>

<details><summary><code>active</code></summary>

```lean
def active
    (s : BlockState) (B_Seqlen : RegionName) (i : Fin BLOCK_SIZE) : Prop :=
  i.val < seqLen s B_Seqlen
```
</details>

<details><summary><code>tokenSoftmaxSpec</code></summary>

```
/-- Exact stable-softmax value produced at one active token lane. -/
```
```lean
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
```
</details>

<details><summary><code>tokenIndex</code></summary>

```lean
def tokenIndex (s : BlockState) (B_Start_Loc : RegionName) (i : Fin BLOCK_SIZE) : Nat :=
  startLoc s B_Start_Loc + i.val
```
</details>

<details><summary><code>seqLen</code></summary>

```lean
def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)
```
</details>

<details><summary><code>tokenSoftmaxInputTile</code></summary>

```
/-- Masked input row for the Python token-softmax path. Inactive lanes are `⊥`,
matching the `other=-float("inf")` load used before the stable softmax. -/
```
```lean
noncomputable def tokenSoftmaxInputTile
    (s : BlockState) (Logics B_Start_Loc B_Seqlen : RegionName)
    (stride_logic_h stride_logic_bs BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < seqLen s B_Seqlen then
        some (s.readMem Logics
          (logicOffset s B_Start_Loc stride_logic_h stride_logic_bs idx.1))
      else none }
```
</details>

<details><summary><code>startLoc</code></summary>

```lean
def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)
```
</details>

<details><summary><code>logicOffset</code></summary>

```lean
def logicOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_logic_h stride_logic_bs : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 1 * stride_logic_h + tokenIndex s B_Start_Loc i * stride_logic_bs
```
</details>

## Also present (pinned special-case summaries)
- `token_softmax_final_store_slice_compute_correct`
- `token_softmax_surface_spec_compute_correct`
