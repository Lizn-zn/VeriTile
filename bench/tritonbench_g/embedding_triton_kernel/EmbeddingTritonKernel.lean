import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Proof-oriented `BLOCK_NN = 1` slice of `embedding_triton_kernel.py`'s
`embedding_kernel`.

The full Python kernel loops through token chunks inside a `BLOCK_N` program.
This slice captures one token row: load its token id, range-check it against
the vocab interval, load the selected embedding vector, and store the output
row under the sequence/dimension masks. -/
def embedding_kernel_one_token
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size start_nn BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_n = tl.program_id(0)
  offs_seq = start_n + $(start_nn)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  n_ctx_mask = offs_seq < $(n_ctx)
  token_id = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=$(vob_end_id))
  id_mask = ($(vob_start_id) <= token_id) and (token_id < $(vob_end_id))
  token_id = token_id - $(vob_start_id)
  dim_mask = offs_d < $(hiden_size)
  load_mask = id_mask and dim_mask
  store_mask = n_ctx_mask and dim_mask
  vecs = tl.load(weight + token_id * $(stride_weight_seq) + offs_d,
    mask=load_mask, other=0.0)
  tl.store(out + offs_seq * $(stride_out_seq) + offs_d, vecs, mask=store_mask)
}

def seqIndex (s : BlockState) (start_nn : Nat) : Nat :=
  s.pids 0 + start_nn

def dimIndex (i : Fin BLOCK_DMODEL) : Nat :=
  i.val

def tokenRaw (s : BlockState) (input_ids : RegionName) (start_nn : Nat) : Nat :=
  s.readMemValue .nat input_ids (seqIndex s start_nn)

def tokenIndex
    (s : BlockState) (input_ids : RegionName) (vob_start_id start_nn : Nat) : Nat :=
  tokenRaw s input_ids start_nn - vob_start_id

def outOffset (s : BlockState) (stride_out_seq start_nn : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  seqIndex s start_nn * stride_out_seq + dimIndex i

def weightOffset
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id stride_weight_seq start_nn : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  tokenIndex s input_ids vob_start_id start_nn * stride_weight_seq + dimIndex i

def active
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id vob_end_id n_ctx hiden_size start_nn BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) : Prop :=
  seqIndex s start_nn < n_ctx ∧
    vob_start_id ≤ tokenRaw s input_ids start_nn ∧
    tokenRaw s input_ids start_nn < vob_end_id ∧
    dimIndex i < hiden_size

def storeActive
    (s : BlockState) (n_ctx hiden_size start_nn BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) : Prop :=
  seqIndex s start_nn < n_ctx ∧ dimIndex i < hiden_size

instance activeDecidable
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id vob_end_id n_ctx hiden_size start_nn BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) :
    Decidable (active s input_ids vob_start_id vob_end_id n_ctx hiden_size
      start_nn BLOCK_DMODEL i) := by
  unfold active
  infer_instance

instance storeActiveDecidable
    (s : BlockState) (n_ctx hiden_size start_nn BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) :
    Decidable (storeActive s n_ctx hiden_size start_nn BLOCK_DMODEL i) := by
  unfold storeActive
  infer_instance

noncomputable def embeddingSpec
    (s : BlockState) (weight input_ids : RegionName)
    (vob_start_id vob_end_id stride_weight_seq start_nn BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  WithBot.unbotD 0
    (if vob_start_id ≤ tokenRaw s input_ids start_nn ∧
        tokenRaw s input_ids start_nn < vob_end_id then
      some (s.readMem weight
        (weightOffset s input_ids vob_start_id stride_weight_seq start_nn i))
    else
      some (0.0 : ℝ))

/-- Algorithm-layer correctness for the one-token embedding slice. -/
theorem embedding_kernel_one_token_correct
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size start_nn BLOCK_DMODEL : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_out_seq start_nn i))
    (hExec : exec (embedding_kernel_one_token weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size start_nn BLOCK_DMODEL) s = some s') :
    ∀ i : Fin BLOCK_DMODEL,
      s'.readMem out (outOffset s stride_out_seq start_nn i) =
        if storeActive s n_ctx hiden_size start_nn BLOCK_DMODEL i then
          embeddingSpec s weight input_ids vob_start_id vob_end_id stride_weight_seq
            start_nn BLOCK_DMODEL i
        else
          s.readMem out (outOffset s stride_out_seq start_nn i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_DMODEL] =>
        (s.pids 0 + start_nn) * stride_out_seq + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [outOffset, seqIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBD : 0 < BLOCK_DMODEL
  · simp [exec, embedding_kernel_one_token, stepStmts, stepStmt, evalOp,
          Option.bind, Option.map, Tile.bop, Tile.cop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, ComparableDType.le, BlockState.readMemValue,
          hBD] at hExec
    rw [← hExec]
    simp only [outOffset, seqIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj (i, PUnit.unit)]
    by_cases hSeq : s.pids 0 + start_nn < n_ctx
    · by_cases hLow : vob_start_id ≤ tokenRaw s input_ids start_nn
      · by_cases hHigh : tokenRaw s input_ids start_nn < vob_end_id
        · by_cases hDim : i.val < hiden_size
          · have hLowRaw :
                vob_start_id ≤
                  (match s.readMemTyped TileDType.nat input_ids (s.pids 0 + start_nn) with
                  | some value => value
                  | none => BlockState.defaultCarrier TileDType.nat) := by
              simpa [tokenRaw, seqIndex, BlockState.readMemValue] using hLow
            have hHighRaw :
                (match s.readMemTyped TileDType.nat input_ids (s.pids 0 + start_nn) with
                  | some value => value
                  | none => BlockState.defaultCarrier TileDType.nat) < vob_end_id := by
              simpa [tokenRaw, seqIndex, BlockState.readMemValue] using hHigh
            simp [storeActive, embeddingSpec, weightOffset, tokenIndex, tokenRaw,
                  outOffset, seqIndex, dimIndex, hSeq, hLow, hHigh, hLowRaw,
                  hHighRaw, hDim, BlockState.readMemValue, WithBot.unbotD]
          · simp [storeActive, seqIndex, dimIndex, hSeq, hDim]
        · by_cases hDim : i.val < hiden_size
          · have hHighRaw :
                ¬ (match s.readMemTyped TileDType.nat input_ids (s.pids 0 + start_nn) with
                  | some value => value
                  | none => BlockState.defaultCarrier TileDType.nat) < vob_end_id := by
              simpa [tokenRaw, seqIndex, BlockState.readMemValue] using hHigh
            simp [storeActive, embeddingSpec, weightOffset, tokenIndex, tokenRaw,
                  seqIndex, dimIndex, hSeq, hLow, hHigh, hHighRaw, hDim,
                  BlockState.readMemValue, WithBot.unbotD]
          · simp [storeActive, seqIndex, dimIndex, hSeq, hDim]
      · by_cases hDim : i.val < hiden_size
        · have hLowRaw :
              ¬ vob_start_id ≤
                (match s.readMemTyped TileDType.nat input_ids (s.pids 0 + start_nn) with
                | some value => value
                | none => BlockState.defaultCarrier TileDType.nat) := by
            simpa [tokenRaw, seqIndex, BlockState.readMemValue] using hLow
          simp [storeActive, embeddingSpec, weightOffset, tokenIndex, tokenRaw,
                seqIndex, dimIndex, hSeq, hLow, hLowRaw, hDim,
                BlockState.readMemValue, WithBot.unbotD]
        · simp [storeActive, seqIndex, dimIndex, hSeq, hDim]
    · simp [storeActive, seqIndex, dimIndex, hSeq]
  · exact False.elim (hBD (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-token embedding slice. -/
theorem embedding_kernel_one_token_compute_correct
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size start_nn BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_DMODEL => outOffset s stride_out_seq start_nn i)) :
    ComputeCorrect.Realizes
      (kernel := embedding_kernel_one_token weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size start_nn BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_DMODEL =>
          storeActive s n_ctx hiden_size start_nn BLOCK_DMODEL i)
        (fun i => (out, outOffset s stride_out_seq start_nn i)))
      (expected := fun i =>
        embeddingSpec s weight input_ids vob_start_id vob_end_id stride_weight_seq
          start_nn BLOCK_DMODEL i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [embedding_kernel_one_token]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := embedding_kernel_one_token_correct weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
    start_nn BLOCK_DMODEL s s' hOutInj hExec i
  simpa [hActive] using h

end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel
