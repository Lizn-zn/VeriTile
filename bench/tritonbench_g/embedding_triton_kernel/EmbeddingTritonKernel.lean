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
  have hseq :
      seqLaneIndex s BLOCK_N start_nn idx.1 =
        fullSeqIndex s BLOCK_N
          (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h).1 := by
    simp [seqLaneIndex, fullSeqIndex, embeddingChunkToFullIndex]
    omega
  have hdim :
      dimIndex idx.2.1 =
        dimIndex (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h).2.1 := by
    simp [embeddingChunkToFullIndex]
  simp [embeddingSpec2D, embeddingSpecFull, tokenRaw2D, tokenRawFull,
    weightOffset2D, weightOffsetFull, tokenIndex2D, tokenIndexFull, hseq, hdim]

theorem outOffset2D_eq_full
    (s : BlockState) (stride_out_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (h : start_nn + idx.1.val < BLOCK_N) :
    outOffset2D s stride_out_seq BLOCK_N start_nn idx =
      outOffsetFull s stride_out_seq BLOCK_N
        (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h) := by
  have hseq :
      seqLaneIndex s BLOCK_N start_nn idx.1 =
        fullSeqIndex s BLOCK_N
          (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h).1 := by
    simp [seqLaneIndex, fullSeqIndex, embeddingChunkToFullIndex]
    omega
  have hdim :
      dimIndex idx.2.1 =
        dimIndex (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h).2.1 := by
    simp [embeddingChunkToFullIndex]
  simp [outOffset2D, outOffsetFull, hseq, hdim]

theorem storeActive2D_iff_full
    (s : BlockState) (n_ctx hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (h : start_nn + idx.1.val < BLOCK_N) :
    storeActive2D s n_ctx hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL idx ↔
      storeActiveFull s n_ctx hiden_size BLOCK_N BLOCK_DMODEL
        (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h) := by
  have hseq :
      seqLaneIndex s BLOCK_N start_nn idx.1 =
        fullSeqIndex s BLOCK_N
          (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h).1 := by
    simp [seqLaneIndex, fullSeqIndex, embeddingChunkToFullIndex]
    omega
  have hdim :
      dimIndex idx.2.1 =
        dimIndex (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h).2.1 := by
    simp [embeddingChunkToFullIndex]
  simp [storeActive2D, storeActiveFull, hseq, hdim]

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
  simp [embeddingPrefixWritten, embeddingChunkToFullIndex]

theorem embeddingPrefixIndex_ne_currentChunk
    (start_nn BLOCK_NN : Nat)
    (oldIdx : TileIndex [BLOCK_N, BLOCK_DMODEL])
    (curIdx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (hCur : start_nn + curIdx.1.val < BLOCK_N)
    (hOld : embeddingPrefixWritten start_nn oldIdx) :
    oldIdx ≠ embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn curIdx hCur := by
  intro hEq
  have hOldLt : oldIdx.1.val < start_nn := by
    simpa [embeddingPrefixWritten] using hOld
  have hCurGe :
      start_nn ≤
        (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn curIdx hCur).1.val := by
    simp [embeddingChunkToFullIndex]
  rw [hEq] at hOldLt
  omega

theorem embeddingOldPrefix_outOffset_ne_currentChunk
    (s : BlockState) (stride_out_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL : Nat)
    (oldIdx : TileIndex [BLOCK_N, BLOCK_DMODEL])
    (curIdx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
    (hCur : start_nn + curIdx.1.val < BLOCK_N)
    (hOld : embeddingPrefixWritten start_nn oldIdx) :
    outOffsetFull s stride_out_seq BLOCK_N oldIdx ≠
      outOffset2D s stride_out_seq BLOCK_N start_nn curIdx := by
  intro hEq
  have hOut := outOffset2D_eq_full s stride_out_seq BLOCK_N start_nn BLOCK_NN
    BLOCK_DMODEL curIdx hCur
  have hEqFull :
      outOffsetFull s stride_out_seq BLOCK_N oldIdx =
        outOffsetFull s stride_out_seq BLOCK_N
          (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn curIdx hCur) := by
    simpa [hOut] using hEq
  exact embeddingPrefixIndex_ne_currentChunk (BLOCK_N := BLOCK_N)
    (BLOCK_DMODEL := BLOCK_DMODEL) start_nn BLOCK_NN oldIdx curIdx hCur hOld
    (hOutInj hEqFull)

theorem embeddingCurrentChunkNoCollision_of_full_injective
    (s : BlockState) (stride_out_seq n_ctx hiden_size BLOCK_N start_nn
      BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
    (hIdxBound : start_nn + idx.1.val < BLOCK_N)
    (hLaneBound :
      ∀ lane : TileIndex [BLOCK_NN, BLOCK_DMODEL],
        storeActive2D s n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL lane →
          start_nn + lane.1.val < BLOCK_N) :
    ∀ lane : TileIndex [BLOCK_NN, BLOCK_DMODEL],
      storeActive2D s n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
        BLOCK_DMODEL lane →
        outOffset2D s stride_out_seq BLOCK_N start_nn lane =
          outOffset2D s stride_out_seq BLOCK_N start_nn idx →
        lane = idx := by
  intro lane hlane heq
  have hLaneBound' := hLaneBound lane hlane
  have hLaneOut := outOffset2D_eq_full s stride_out_seq BLOCK_N start_nn
    BLOCK_NN BLOCK_DMODEL lane hLaneBound'
  have hIdxOut := outOffset2D_eq_full s stride_out_seq BLOCK_N start_nn
    BLOCK_NN BLOCK_DMODEL idx hIdxBound
  have hFullEq :
      embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn lane hLaneBound' =
        embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx hIdxBound := by
    apply hOutInj
    simpa [hLaneOut, hIdxOut] using heq
  apply Prod.ext
  · apply Fin.ext
    have hval := congrArg
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => idx.1.val) hFullEq
    simpa [embeddingChunkToFullIndex] using Nat.add_left_cancel hval
  · have htail := congrArg
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => idx.2) hFullEq
    simpa [embeddingChunkToFullIndex] using htail

theorem embeddingChunkLane_lt_of_aligned_start
    (BLOCK_N BLOCK_NN chunks k start_nn : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (hStep : 0 < BLOCK_NN)
    (hBlock : BLOCK_N = chunks * BLOCK_NN)
    (hStartEq : start_nn = k * BLOCK_NN)
    (hStartLt : start_nn < BLOCK_N) :
    start_nn + idx.1.val < BLOCK_N := by
  have hklt : k < chunks := by
    apply (Nat.mul_lt_mul_right hStep).mp
    simpa [hStartEq, hBlock] using hStartLt
  calc
    start_nn + idx.1.val = k * BLOCK_NN + idx.1.val := by rw [hStartEq]
    _ < k * BLOCK_NN + BLOCK_NN := Nat.add_lt_add_left idx.1.isLt _
    _ = (k + 1) * BLOCK_NN := by rw [Nat.succ_mul]
    _ ≤ chunks * BLOCK_NN := Nat.mul_le_mul_right BLOCK_NN (Nat.succ_le_of_lt hklt)
    _ = BLOCK_N := by rw [hBlock]

theorem embeddingChunkToFullIndex_written_after
    (start_nn BLOCK_NN : Nat) (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (h : start_nn + idx.1.val < BLOCK_N) :
    embeddingPrefixWritten (start_nn + BLOCK_NN)
      (embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn idx h) := by
  simp [embeddingPrefixWritten, embeddingChunkToFullIndex]

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

def embeddingLoopInvariant
    (s0 : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL : Nat)
    (off : Nat) (st : BlockState) : Prop :=
  ∀ idx : TileIndex [BLOCK_N, BLOCK_DMODEL],
    embeddingPrefixActive s0 n_ctx hiden_size BLOCK_N BLOCK_DMODEL off idx →
      st.readMem out (outOffsetFull s0 stride_out_seq BLOCK_N idx) =
        embeddingSpecFull s0 weight input_ids vob_start_id vob_end_id
          stride_weight_seq BLOCK_N BLOCK_DMODEL idx

theorem embeddingLoopInvariant_zero
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL : Nat) :
    embeddingLoopInvariant s0 weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL 0 st := by
  intro idx hidx
  exact False.elim ((not_embeddingPrefixActive_zero s0 n_ctx hiden_size
    BLOCK_N BLOCK_DMODEL idx) hidx)

theorem embeddingPrefixActive_final_of_storeActive
    (s : BlockState) (n_ctx hiden_size BLOCK_N BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL])
    (hidx : storeActiveFull s n_ctx hiden_size BLOCK_N BLOCK_DMODEL idx) :
    embeddingPrefixActive s n_ctx hiden_size BLOCK_N BLOCK_DMODEL BLOCK_N idx := by
  exact ⟨by simp [embeddingPrefixWritten], hidx⟩

theorem embeddingPrefixActive_of_storeActive_of_final
    (s : BlockState) (n_ctx hiden_size BLOCK_N BLOCK_DMODEL final : Nat)
    (idx : TileIndex [BLOCK_N, BLOCK_DMODEL])
    (hfinal : BLOCK_N ≤ final)
    (hidx : storeActiveFull s n_ctx hiden_size BLOCK_N BLOCK_DMODEL idx) :
    embeddingPrefixActive s n_ctx hiden_size BLOCK_N BLOCK_DMODEL final idx := by
  exact ⟨by
    unfold embeddingPrefixWritten
    exact Nat.lt_of_lt_of_le idx.1.isLt hfinal, hidx⟩

theorem embeddingLoopInvariant_step_of_chunk_write
    (s0 st st' : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN start_nn : Nat)
    (hPrev :
      embeddingLoopInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL start_nn st)
    (hPreserve :
      ∀ idx : TileIndex [BLOCK_N, BLOCK_DMODEL],
        embeddingPrefixActive s0 n_ctx hiden_size BLOCK_N BLOCK_DMODEL start_nn idx →
          st'.readMem out (outOffsetFull s0 stride_out_seq BLOCK_N idx) =
            st.readMem out (outOffsetFull s0 stride_out_seq BLOCK_N idx))
    (hWrite :
      ∀ idx : TileIndex [BLOCK_NN, BLOCK_DMODEL],
        storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
            BLOCK_DMODEL idx →
          st'.readMem out (outOffset2D s0 stride_out_seq BLOCK_N start_nn idx) =
            embeddingSpec2D s0 weight input_ids vob_start_id vob_end_id
              stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL idx) :
    embeddingLoopInvariant s0 weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL (start_nn + BLOCK_NN) st' := by
  intro idx hidx
  by_cases hOld : embeddingPrefixWritten start_nn idx
  · have hOldActive :
        embeddingPrefixActive s0 n_ctx hiden_size BLOCK_N BLOCK_DMODEL
          start_nn idx := ⟨hOld, hidx.2⟩
    rw [hPreserve idx hOldActive]
    exact hPrev idx hOldActive
  · have hge : start_nn ≤ idx.1.val := by
      unfold embeddingPrefixWritten at hOld
      omega
    have hltChunk : idx.1.val - start_nn < BLOCK_NN := by
      unfold embeddingPrefixActive embeddingPrefixWritten at hidx
      omega
    let lane : Fin BLOCK_NN := ⟨idx.1.val - start_nn, hltChunk⟩
    let chunkIdx : TileIndex [BLOCK_NN, BLOCK_DMODEL] := (lane, idx.2)
    have hBound : start_nn + chunkIdx.1.val < BLOCK_N := by
      dsimp [chunkIdx, lane]
      omega
    have hChunkEq :
        embeddingChunkToFullIndex (BLOCK_N := BLOCK_N) start_nn chunkIdx hBound =
          idx := by
      cases idx with
      | mk i rest =>
        have hgei : start_nn ≤ i.val := by
          simpa using hge
        apply Prod.ext
        · apply Fin.ext
          simp [embeddingChunkToFullIndex, chunkIdx, lane]
          omega
        · rfl
    have hStore2D :
        storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL chunkIdx := by
      rw [storeActive2D_iff_full s0 n_ctx hiden_size BLOCK_N start_nn
        BLOCK_NN BLOCK_DMODEL chunkIdx hBound]
      simpa [hChunkEq] using hidx.2
    have hOut :
        outOffset2D s0 stride_out_seq BLOCK_N start_nn chunkIdx =
          outOffsetFull s0 stride_out_seq BLOCK_N idx := by
      simpa [hChunkEq] using
        outOffset2D_eq_full s0 stride_out_seq BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL chunkIdx hBound
    have hSpec :
        embeddingSpec2D s0 weight input_ids vob_start_id vob_end_id
            stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL chunkIdx =
          embeddingSpecFull s0 weight input_ids vob_start_id vob_end_id
            stride_weight_seq BLOCK_N BLOCK_DMODEL idx := by
      simpa [hChunkEq] using
        embeddingSpec2D_eq_full s0 weight input_ids vob_start_id vob_end_id
          stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL chunkIdx hBound
    rw [← hOut, hWrite chunkIdx hStore2D, hSpec]

theorem embeddingCurrentChunkScatter_write
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN start_nn : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (hactive :
      storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
        BLOCK_DMODEL idx)
    (hNoCollision :
      ∀ lane : TileIndex [BLOCK_NN, BLOCK_DMODEL],
        storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL lane →
          outOffset2D s0 stride_out_seq BLOCK_N start_nn lane =
            outOffset2D s0 stride_out_seq BLOCK_N start_nn idx →
          lane = idx) :
    ((TileShape.allIndices [BLOCK_NN, BLOCK_DMODEL]).foldl
        (fun acc lane =>
          if storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
              BLOCK_DMODEL lane then
            acc.writeMem out (outOffset2D s0 stride_out_seq BLOCK_N start_nn lane)
              (embeddingSpec2D s0 weight input_ids vob_start_id vob_end_id
                stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL lane)
          else
            acc)
        st).readMem out (outOffset2D s0 stride_out_seq BLOCK_N start_nn idx) =
      embeddingSpec2D s0 weight input_ids vob_start_id vob_end_id
        stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL idx := by
  exact
    BlockState.scatter_readback_prop_masked_nd_of_true
      (region := out)
      (s := st)
      (offsetFn := fun lane : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        outOffset2D s0 stride_out_seq BLOCK_N start_nn lane)
      (valueFn := fun lane : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        embeddingSpec2D s0 weight input_ids vob_start_id vob_end_id
          stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL lane)
      (P := fun lane : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL lane)
      idx hactive
      (fun lane hlane heq => hNoCollision lane hlane heq)

theorem embeddingCurrentChunkScatter_preserve_old
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN start_nn : Nat)
    (oldIdx : TileIndex [BLOCK_N, BLOCK_DMODEL])
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s0 stride_out_seq BLOCK_N idx))
    (hOld : embeddingPrefixWritten start_nn oldIdx)
    (hLaneBound :
      ∀ lane : TileIndex [BLOCK_NN, BLOCK_DMODEL],
        storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL lane →
          start_nn + lane.1.val < BLOCK_N) :
    ((TileShape.allIndices [BLOCK_NN, BLOCK_DMODEL]).foldl
        (fun acc lane =>
          if storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
              BLOCK_DMODEL lane then
            acc.writeMem out (outOffset2D s0 stride_out_seq BLOCK_N start_nn lane)
              (embeddingSpec2D s0 weight input_ids vob_start_id vob_end_id
                stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL lane)
          else
            acc)
        st).readMem out (outOffsetFull s0 stride_out_seq BLOCK_N oldIdx) =
      st.readMem out (outOffsetFull s0 stride_out_seq BLOCK_N oldIdx) := by
  exact
    BlockState.scatter_prop_masked_preserves_other_offset
      out
      (fun lane : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        outOffset2D s0 stride_out_seq BLOCK_N start_nn lane)
      (fun lane : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        embeddingSpec2D s0 weight input_ids vob_start_id vob_end_id
          stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL lane)
      (fun lane : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL lane)
      (outOffsetFull s0 stride_out_seq BLOCK_N oldIdx)
      (fun lane hlane heq =>
        embeddingOldPrefix_outOffset_ne_currentChunk s0 stride_out_seq BLOCK_N
          start_nn BLOCK_NN BLOCK_DMODEL oldIdx lane hOutInj
          (hLaneBound lane hlane) hOld heq.symm)
      (TileShape.allIndices [BLOCK_NN, BLOCK_DMODEL]) st

theorem embeddingLoopInvariant_step_of_current_chunk_scatter
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN start_nn : Nat)
    (hPrev :
      embeddingLoopInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL start_nn st)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s0 stride_out_seq BLOCK_N idx))
    (hLaneBound :
      ∀ lane : TileIndex [BLOCK_NN, BLOCK_DMODEL],
        storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL lane →
          start_nn + lane.1.val < BLOCK_N) :
    embeddingLoopInvariant s0 weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL (start_nn + BLOCK_NN)
      ((TileShape.allIndices [BLOCK_NN, BLOCK_DMODEL]).foldl
        (fun acc lane =>
          if storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
              BLOCK_DMODEL lane then
            acc.writeMem out (outOffset2D s0 stride_out_seq BLOCK_N start_nn lane)
              (embeddingSpec2D s0 weight input_ids vob_start_id vob_end_id
                stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL lane)
          else
            acc)
        st) := by
  apply embeddingLoopInvariant_step_of_chunk_write s0 st
  · exact hPrev
  · intro oldIdx hOldActive
    exact embeddingCurrentChunkScatter_preserve_old s0 st weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
      BLOCK_N BLOCK_DMODEL BLOCK_NN start_nn oldIdx hOutInj hOldActive.1
      hLaneBound
  · intro idx hStore
    exact embeddingCurrentChunkScatter_write s0 st weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
      BLOCK_N BLOCK_DMODEL BLOCK_NN start_nn idx hStore
      (embeddingCurrentChunkNoCollision_of_full_injective s0 stride_out_seq n_ctx
        hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL idx hOutInj
        (hLaneBound idx hStore) hLaneBound)

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

theorem embeddingLoopInvariant_to_alg_post
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (hInv :
      embeddingLoopInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL BLOCK_N st) :
    embedding_kernel_alg_post weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s0 st := by
  intro idx hidx
  exact hInv idx
    (embeddingPrefixActive_final_of_storeActive s0 n_ctx hiden_size
      BLOCK_N BLOCK_DMODEL idx hidx)

theorem embeddingLoopInvariant_to_alg_post_of_final
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN final : Nat)
    (hfinal : BLOCK_N ≤ final)
    (hInv :
      embeddingLoopInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL final st) :
    embedding_kernel_alg_post weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s0 st := by
  intro idx hidx
  exact hInv idx
    (embeddingPrefixActive_of_storeActive_of_final s0 n_ctx hiden_size
      BLOCK_N BLOCK_DMODEL final idx hfinal hidx)

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
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
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
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s :=
  embedding_kernel_compute_correct_of_algorithm weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
    hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s hAlg

/-- Compute-facing correctness for the embedding kernel. -/
theorem embedding_kernel_compute_correct
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
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
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s :=
  embedding_kernel_compute_correct_of_algorithm weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
    hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s hAlg
end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel
