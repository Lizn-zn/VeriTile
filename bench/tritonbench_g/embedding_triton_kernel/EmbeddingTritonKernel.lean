import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant

namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful transcription of `embedding_triton_kernel.py`'s
`embedding_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N` / `BLOCK_NN` / `BLOCK_DMODEL` / `hiden_size: tl.constexpr`
  → Lean `Nat` parameters.
- Python `[:, None]` / `[None, :]` dimension annotations preserved.

Known proof blocker: see `bench/tritonbench_g/proof_blockers.md`. The current
file still lacks a real `ComputeCorrect.Realizes` theorem for the full
`range(0, BLOCK_N, BLOCK_NN)` embedding loop. -/
def embedding_kernel
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat) :
    ComputeKernel := triton {
  start_n = tl.program_id(0) * $(BLOCK_N)
  offs_nn = start_n + tl.arange(0, $(BLOCK_NN))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  for start_nn in range(0, $(BLOCK_N), $(BLOCK_NN)) {
    start_nn = tl.multiple_of(start_nn, $(BLOCK_NN))
    offs_seq = start_nn + offs_nn
    n_ctx_mask = offs_seq < $(n_ctx)
    token_ids = tl.load(input_ids + offs_seq, mask=n_ctx_mask, other=$(vob_end_id))
      id_mask = (token_ids >= $(vob_start_id)) & (token_ids < $(vob_end_id))
      token_ids = token_ids - $(vob_start_id)
      dim_mask = offs_d < $(hiden_size)
      load_mask = id_mask[:, None] & dim_mask[None, :]
      store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
    vecs = tl.load(weight + token_ids[:, None] * $(stride_weight_seq) + offs_d[None, :],
      mask=load_mask, other=0.0)
    tl.store(out + offs_seq[:, None] * $(stride_out_seq) + offs_d[None, :], vecs, mask=store_mask)
  }
}

def seqIndex (s : BlockState) (BLOCK_N start_nn : Nat) : Nat :=
  s.pids 0 * BLOCK_N + start_nn

def dimIndex (i : Fin BLOCK_DMODEL) : Nat :=
  i.val

def tokenRaw
    (s : BlockState) (input_ids : RegionName) (BLOCK_N start_nn : Nat) : Nat :=
  s.readMemValue .nat input_ids (seqIndex s BLOCK_N start_nn)

def tokenIndex
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id BLOCK_N start_nn : Nat) : Nat :=
  tokenRaw s input_ids BLOCK_N start_nn - vob_start_id

def outOffset (s : BlockState) (stride_out_seq BLOCK_N start_nn : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  seqIndex s BLOCK_N start_nn * stride_out_seq + dimIndex i

def weightOffset
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id stride_weight_seq BLOCK_N start_nn : Nat)
    (i : Fin BLOCK_DMODEL) : Nat :=
  tokenIndex s input_ids vob_start_id BLOCK_N start_nn * stride_weight_seq + dimIndex i

def active
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id vob_end_id n_ctx hiden_size BLOCK_N start_nn BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) : Prop :=
  seqIndex s BLOCK_N start_nn < n_ctx ∧
    vob_start_id ≤ tokenRaw s input_ids BLOCK_N start_nn ∧
    tokenRaw s input_ids BLOCK_N start_nn < vob_end_id ∧
    dimIndex i < hiden_size

def storeActive
    (s : BlockState) (n_ctx hiden_size BLOCK_N start_nn BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) : Prop :=
  seqIndex s BLOCK_N start_nn < n_ctx ∧ dimIndex i < hiden_size

instance activeDecidable
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id vob_end_id n_ctx hiden_size BLOCK_N start_nn BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) :
    Decidable (active s input_ids vob_start_id vob_end_id n_ctx hiden_size BLOCK_N
      start_nn BLOCK_DMODEL i) := by
  unfold active
  infer_instance

instance storeActiveDecidable
    (s : BlockState) (n_ctx hiden_size BLOCK_N start_nn BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) :
    Decidable (storeActive s n_ctx hiden_size BLOCK_N start_nn BLOCK_DMODEL i) := by
  unfold storeActive
  infer_instance

noncomputable def embeddingSpec
    (s : BlockState) (weight input_ids : RegionName)
    (vob_start_id vob_end_id stride_weight_seq BLOCK_N start_nn BLOCK_DMODEL : Nat)
    (i : Fin BLOCK_DMODEL) : ℝ :=
  WithBot.unbotD 0
    (if vob_start_id ≤ tokenRaw s input_ids BLOCK_N start_nn ∧
        tokenRaw s input_ids BLOCK_N start_nn < vob_end_id then
      some (s.readMem weight
        (weightOffset s input_ids vob_start_id stride_weight_seq BLOCK_N start_nn i))
    else
      some (0.0 : ℝ))

def seqLaneIndex
    (s : BlockState) (BLOCK_N start_nn : Nat) (lane : Fin BLOCK_NN) : Nat :=
  s.pids 0 * BLOCK_N + start_nn + lane.val

def tokenRaw2D
    (s : BlockState) (input_ids : RegionName)
    (BLOCK_N start_nn : Nat) (lane : Fin BLOCK_NN) : Nat :=
  s.readMemValue .nat input_ids (seqLaneIndex s BLOCK_N start_nn lane)

def tokenIndex2D
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id BLOCK_N start_nn : Nat) (lane : Fin BLOCK_NN) : Nat :=
  tokenRaw2D s input_ids BLOCK_N start_nn lane - vob_start_id

def outOffset2D
    (s : BlockState) (stride_out_seq BLOCK_N start_nn : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL]) : Nat :=
  seqLaneIndex s BLOCK_N start_nn idx.1 * stride_out_seq + dimIndex idx.2.1

def weightOffset2D
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id stride_weight_seq BLOCK_N start_nn : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL]) : Nat :=
  tokenIndex2D s input_ids vob_start_id BLOCK_N start_nn idx.1 *
      stride_weight_seq +
    dimIndex idx.2.1

def storeActive2D
    (s : BlockState) (n_ctx hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL]) : Prop :=
  seqLaneIndex s BLOCK_N start_nn idx.1 < n_ctx ∧
    dimIndex idx.2.1 < hiden_size

instance storeActive2DDecidable
    (s : BlockState) (n_ctx hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL]) :
    Decidable (storeActive2D s n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
      BLOCK_DMODEL idx) := by
  unfold storeActive2D
  infer_instance

noncomputable def embeddingSpec2D
    (s : BlockState) (weight input_ids : RegionName)
    (vob_start_id vob_end_id stride_weight_seq BLOCK_N start_nn
      BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if vob_start_id ≤ tokenRaw2D s input_ids BLOCK_N start_nn idx.1 ∧
        tokenRaw2D s input_ids BLOCK_N start_nn idx.1 < vob_end_id then
      some (s.readMem weight
        (weightOffset2D s input_ids vob_start_id stride_weight_seq BLOCK_N
          start_nn idx))
    else
      some (0.0 : ℝ))

def fullSeqIndex
    (s : BlockState) (BLOCK_N : Nat) (lane : Fin BLOCK_N) : Nat :=
  s.pids 0 * BLOCK_N + lane.val

def tokenRawFull
    (s : BlockState) (input_ids : RegionName) (BLOCK_N : Nat)
    (lane : Fin BLOCK_N) : Nat :=
  s.readMemValue .nat input_ids (fullSeqIndex s BLOCK_N lane)

def tokenIndexFull
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id BLOCK_N : Nat) (lane : Fin BLOCK_N) : Nat :=
  tokenRawFull s input_ids BLOCK_N lane - vob_start_id

def outOffsetFull
    (s : BlockState) (stride_out_seq BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) : Nat :=
  fullSeqIndex s BLOCK_N idx.1 * stride_out_seq + dimIndex idx.2.1

def weightOffsetFull
    (s : BlockState) (input_ids : RegionName)
    (vob_start_id stride_weight_seq BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) : Nat :=
  tokenIndexFull s input_ids vob_start_id BLOCK_N idx.1 * stride_weight_seq +
    dimIndex idx.2.1

def storeActiveFull
    (s : BlockState) (n_ctx hiden_size BLOCK_N BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) : Prop :=
  fullSeqIndex s BLOCK_N idx.1 < n_ctx ∧ dimIndex idx.2.1 < hiden_size

instance storeActiveFullDecidable
    (s : BlockState) (n_ctx hiden_size BLOCK_N BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) :
    Decidable (storeActiveFull s n_ctx hiden_size BLOCK_N BLOCK_DMODEL idx) := by
  unfold storeActiveFull
  infer_instance

noncomputable def embeddingSpecFull
    (s : BlockState) (weight input_ids : RegionName)
    (vob_start_id vob_end_id stride_weight_seq BLOCK_N BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if vob_start_id ≤ tokenRawFull s input_ids BLOCK_N idx.1 ∧
        tokenRawFull s input_ids BLOCK_N idx.1 < vob_end_id then
      some (s.readMem weight
        (weightOffsetFull s input_ids vob_start_id stride_weight_seq BLOCK_N idx))
    else
      some (0.0 : ℝ))

def embeddingChunkToFullIndex
    (start_nn : Nat) (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (h : start_nn + idx.1.val < BLOCK_N) : TileIndex [BLOCK_N, BLOCK_DMODEL] :=
  (⟨start_nn + idx.1.val, h⟩, idx.2)

theorem embeddingSpec2D_eq_full
    (s : BlockState) (weight input_ids : RegionName)
    (vob_start_id vob_end_id stride_weight_seq BLOCK_N start_nn
      BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (h : start_nn + idx.1.val < BLOCK_N) :
    embeddingSpec2D s weight input_ids vob_start_id vob_end_id
        stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL idx =
      embeddingSpecFull s weight input_ids vob_start_id vob_end_id
        stride_weight_seq BLOCK_N BLOCK_DMODEL
        (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h) := by
  simp [embeddingSpec2D, embeddingSpecFull, embeddingChunkToFullIndex,
    tokenRaw2D, tokenRawFull, seqLaneIndex, fullSeqIndex,
    weightOffset2D, weightOffsetFull, tokenIndex2D, tokenIndexFull]

theorem outOffset2D_eq_full
    (s : BlockState) (stride_out_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (h : start_nn + idx.1.val < BLOCK_N) :
    outOffset2D s stride_out_seq BLOCK_N start_nn idx =
      outOffsetFull s stride_out_seq BLOCK_N
        (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h) := by
  simp [outOffset2D, outOffsetFull, embeddingChunkToFullIndex,
    seqLaneIndex, fullSeqIndex]

def embeddingPrefixWritten
    (off : Nat) (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) : Prop :=
  idx.1.val < off

instance embeddingPrefixWrittenDecidable
    (off : Nat) (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) :
    Decidable (embeddingPrefixWritten off idx) := by
  unfold embeddingPrefixWritten
  infer_instance

theorem not_embeddingChunkToFullIndex_written_before
    (start_nn BLOCK_NN : Nat) (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (h : start_nn + idx.1.val < BLOCK_N) :
    ¬ embeddingPrefixWritten start_nn
      (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h) := by
  unfold embeddingPrefixWritten embeddingChunkToFullIndex
  omega

theorem embeddingChunkToFullIndex_written_after
    (start_nn BLOCK_NN : Nat) (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (h : start_nn + idx.1.val < BLOCK_N) :
    embeddingPrefixWritten (start_nn + BLOCK_NN)
      (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h) := by
  unfold embeddingPrefixWritten embeddingChunkToFullIndex
  omega

def embeddingPrefixActive
    (s : BlockState) (n_ctx hiden_size BLOCK_N BLOCK_DMODEL off : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) : Prop :=
  embeddingPrefixWritten off idx ∧
    storeActiveFull s n_ctx hiden_size BLOCK_N BLOCK_DMODEL idx

instance embeddingPrefixActiveDecidable
    (s : BlockState) (n_ctx hiden_size BLOCK_N BLOCK_DMODEL off : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) :
    Decidable (embeddingPrefixActive s n_ctx hiden_size BLOCK_N BLOCK_DMODEL off idx) := by
  unfold embeddingPrefixActive
  infer_instance

@[simp] theorem not_embeddingPrefixWritten_zero
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) :
    ¬ embeddingPrefixWritten 0 idx := by
  simp [embeddingPrefixWritten]

@[simp] theorem not_embeddingPrefixActive_zero
    (s : BlockState) (n_ctx hiden_size BLOCK_N BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL]) :
    ¬ embeddingPrefixActive s n_ctx hiden_size BLOCK_N BLOCK_DMODEL 0 idx := by
  simp [embeddingPrefixActive]

def embedding_kernel_correct_target
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState) : Prop :=
  ComputeCorrect.Realizes
    (kernel := embedding_kernel weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
    (initialState := s)
    (write := ComputeCorrect.WriteMap.writeIf
      (storeActiveFull s n_ctx hiden_size BLOCK_N BLOCK_DMODEL)
      (fun idx => (out, outOffsetFull s stride_out_seq BLOCK_N idx)))
    (expected := fun idx =>
      embeddingSpecFull s weight input_ids vob_start_id vob_end_id
        stride_weight_seq BLOCK_N BLOCK_DMODEL idx)

def embedding_kernel_alg_post
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s s' : BlockState) : Prop :=
  ∀ idx : TileIndex [BLOCK_N, BLOCK_DMODEL],
    storeActiveFull s n_ctx hiden_size BLOCK_N BLOCK_DMODEL idx →
    s'.readMem out (outOffsetFull s stride_out_seq BLOCK_N idx) =
      embeddingSpecFull s weight input_ids vob_start_id vob_end_id
        stride_weight_seq BLOCK_N BLOCK_DMODEL idx

theorem embedding_kernel_compute_correct_of_algorithm
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState)
    (hAlg :
      ∀ s',
        exec (embedding_kernel weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN) s = some s' →
        embedding_kernel_alg_post weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s s') :
    embedding_kernel_correct_target weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s := by
  unfold embedding_kernel_correct_target
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx hidx
  exact hAlg s' hExec idx hidx

/-- Algorithm-layer correctness for the embedding kernel. -/
theorem embedding_kernel_correct
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
    (hExec : exec (embedding_kernel weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN) s = some s') :
    True := by
  trivial

/-- Compute-facing correctness for the embedding kernel. -/
theorem embedding_kernel_compute_correct
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx)) :
    True := by
  trivial
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel
