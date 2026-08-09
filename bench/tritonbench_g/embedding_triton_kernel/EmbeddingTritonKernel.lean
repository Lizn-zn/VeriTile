import VeriTile.Triton

/-!
# `embedding_triton_kernel` — strict per-kernel correctness

`embedding_kernel` is a vocabulary embedding lookup: program `pid` walks its
`BLOCK_N`-wide sequence window in `BLOCK_NN`-sized chunks, loads each token id,
shifts it by `vob_start_id`, and gathers the corresponding `weight` row into the
`out` buffer, masked by the context length and hidden size.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body, including the full `for start_nn in range(0, BLOCK_N, BLOCK_NN)` loop. The
host launch (`embedding_kernel[grid](...)`, the grid size `cdiv(n_ctx, BLOCK_N)`,
the host-side `BLOCK_DMODEL = next_power_of_2(hidden)` choice, and how the runtime
composes per-program writes) is the *trusted boundary*, not a proof obligation
here. Because `pids 0` is universally quantified, the per-program statement
covers every program of the grid.

## Proof architecture

```
embedding_kernel_output_summary                       ← TOP THEOREM
  ├─ (toAlgorithm? = Except.ok _)                      surface lowers to the algorithm layer
  └─ embedding_kernel_compute_correct                  ← ComputeCorrect over the masked gather
       └─ embedding_kernel_compute_correct_of_algorithm
            └─ embedding_kernel_alg_post_of_exec       loop-invariant readback per active cell
                 └─ (embeddingLoopInvariant / embeddingProjectedBody_alg_post …)
```

The loop is discharged via an explicit `embeddingLoopInvariant` carried across the
`range` iterations; the final state realizes the per-cell gather spec
`embeddingSpecFull`.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The proof is established for the `BLOCK_NN = 1` chunking (`hOne`), which
is exactly the host configuration. Correctness requires a per-cell injectivity
hypothesis `hOutInj` on the output offsets (no-duplicate-destination across loop
iterations) and the alias side conditions `input_ids ≠ out`, `weight ≠ out` (the
gathered token ids and weight rows are read before the scatter; aliasing the
output would corrupt later iterations). The gathered weight-row offset is
data-dependent (read from `input_ids`). The `other=vob_end_id`/`other=0.0`
out-of-range fills and dtype casts reduce to identity at the algorithm layer. The
statement is scoped to *store-active* cells; masked-out cells are not part of the
realized write map.
-/

namespace VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel

open VeriTile.Triton
open scoped VeriTile.Triton.GatherTileKernelIO

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful transcription of `embedding_triton_kernel.py`'s
`embedding_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_N` / `BLOCK_NN` / `BLOCK_DMODEL` / `hiden_size: tl.constexpr`
  → Lean `Nat` parameters.
- Python `[:, None]` / `[None, :]` dimension annotations preserved.

The proof below connects the full `range(0, BLOCK_N, BLOCK_NN)` embedding loop
to `ComputeCorrect.Realizes_without_Rounding` under the stated no-collision/no-alias
hypotheses. -/
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
    token_ids = tl.load($((input_ids : Region .nat)) + offs_seq, mask=n_ctx_mask, other=$(vob_end_id))
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

theorem seqLaneIndex_eval_add
    (s : BlockState) (BLOCK_N start_nn : Nat) (lane : Fin BLOCK_NN) :
    start_nn + (s.pids 0 * BLOCK_N + lane.val) =
      seqLaneIndex s BLOCK_N start_nn lane := by
  simp [seqLaneIndex]
  omega

theorem outOffset2D_eval_add
    (s : BlockState) (stride_out_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL]) :
    (start_nn + (s.pids 0 * BLOCK_N + idx.1.val)) * stride_out_seq +
        idx.2.1.val =
      outOffset2D s stride_out_seq BLOCK_N start_nn idx := by
  simp [outOffset2D, seqLaneIndex, dimIndex]
  omega

theorem storeActive2D_eval_iff
    (s : BlockState) (n_ctx hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL]) :
    (start_nn + (s.pids 0 * BLOCK_N + idx.1.val) < n_ctx ∧
        idx.2.1.val < hiden_size) ↔
      storeActive2D s n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
        BLOCK_DMODEL idx := by
  simp [storeActive2D, seqLaneIndex, dimIndex]
  omega

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

theorem embeddingLoopStoreValue_eval_eq_spec2D
    (s0 st : BlockState) (weight input_ids : RegionName)
    (vob_start_id vob_end_id stride_weight_seq n_ctx hiden_size BLOCK_N
      start_nn BLOCK_NN BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (hInputRead :
      ∀ offset,
        st.readMemValue .nat (Region.cast input_ids) offset =
          s0.readMemValue .nat (Region.cast input_ids) offset)
    (hactive :
      storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
        BLOCK_DMODEL idx) :
    WithBot.unbotD 0
        (if
            ((vob_start_id ≤
                if start_nn + (s0.pids 0 * BLOCK_N + idx.1.val) < n_ctx then
                  st.readMemValue .nat (Region.cast input_ids)
                    (start_nn + (s0.pids 0 * BLOCK_N + idx.1.val))
                else vob_end_id) ∧
              (if start_nn + (s0.pids 0 * BLOCK_N + idx.1.val) < n_ctx then
                  st.readMemValue .nat (Region.cast input_ids)
                    (start_nn + (s0.pids 0 * BLOCK_N + idx.1.val))
                else vob_end_id) < vob_end_id) ∧
              idx.2.1.val < hiden_size then
            some
              (s0.readMem weight
                (((if start_nn + (s0.pids 0 * BLOCK_N + idx.1.val) < n_ctx then
                      st.readMemValue .nat (Region.cast input_ids)
                        (start_nn + (s0.pids 0 * BLOCK_N + idx.1.val))
                    else vob_end_id) - vob_start_id) * stride_weight_seq +
                  idx.2.1.val))
          else some (0.0 : ℝ)) =
      embeddingSpec2D s0 weight input_ids vob_start_id vob_end_id
        stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL idx := by
  have hEvalActive :
      start_nn + (s0.pids 0 * BLOCK_N + idx.1.val) < n_ctx ∧
        idx.2.1.val < hiden_size := by
    exact (storeActive2D_eval_iff s0 n_ctx hiden_size BLOCK_N start_nn
      BLOCK_NN BLOCK_DMODEL idx).2 hactive
  rcases hEvalActive with ⟨hSeq, hDim⟩
  have hSeqIndex :
      start_nn + (s0.pids 0 * BLOCK_N + idx.1.val) =
        seqLaneIndex s0 BLOCK_N start_nn idx.1 :=
    seqLaneIndex_eval_add s0 BLOCK_N start_nn idx.1
  have hSeqAssoc : s0.pids 0 * BLOCK_N + start_nn + idx.1.val < n_ctx := by
    omega
  have hReadAssoc :
      st.readMemValue .nat (Region.cast input_ids)
          (s0.pids 0 * BLOCK_N + start_nn + idx.1.val) =
        s0.readMemValue .nat (Region.cast input_ids)
          (s0.pids 0 * BLOCK_N + start_nn + idx.1.val) :=
    hInputRead _
  have hReadAssoc' :
      st.readMemValue .nat input_ids
          (s0.pids 0 * BLOCK_N + start_nn + idx.1.val) =
        s0.readMemValue .nat input_ids
          (s0.pids 0 * BLOCK_N + start_nn + idx.1.val) :=
    hInputRead _
  simp [embeddingSpec2D, tokenRaw2D, tokenIndex2D, weightOffset2D,
    seqLaneIndex, dimIndex, hSeq, hDim, hInputRead,
    hSeqIndex, hSeqAssoc, hReadAssoc, hReadAssoc']

theorem foldl_writeMem_prop_preserves_readMemValue_of_region_ne
    {α : Type} (l : List α) (s : BlockState) (writeRegion readRegion : RegionName)
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (dtype : TileDType) (offset : Nat)
    (hNe : readRegion ≠ writeRegion) :
    ((l.foldl
        (fun acc k =>
          if P k then acc.writeMem writeRegion (offsetFn k) (valueFn k) else acc)
        s).readMemValue dtype readRegion offset) =
      s.readMemValue dtype readRegion offset := by
  induction l generalizing s with
  | nil =>
      rfl
  | cons hd tl ih =>
      simp only [List.foldl_cons]
      by_cases hp : P hd
      · simp only [hp, if_true]
        rw [ih]
        cases dtype <;>
          simp [BlockState.readMemValue, BlockState.readMemTyped,
            BlockState.readMemAs, BlockState.readMem, BlockState.writeMem, hNe]
      · simp only [hp, if_false]
        exact ih s

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

def embedding_kernel_correct_target
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState) : Prop :=
  ComputeCorrect.Realizes_without_Rounding
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

def embeddingPreLoop
    (BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat) : List Stmt :=
  [ .assign .nat [] "start_n"
      (.mul NumericDType.nat Broadcast.nil (.programId 0) (.constNat BLOCK_N))
  , .assign .nat [BLOCK_NN] "offs_nn"
      (.add NumericDType.nat Broadcast.scalarL
        (.ref .nat [] "start_n") (.arange BLOCK_NN))
  , .assign .nat [BLOCK_DMODEL] "offs_d"
      (.arange BLOCK_DMODEL)
  ]

def embeddingLoopBody
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat) : List Stmt :=
  [ .assign .nat [] "start_nn"
      (.ref .nat [] "start_nn")
  , .assign .nat [BLOCK_NN] "offs_seq"
      (.add NumericDType.nat Broadcast.scalarL
        (.ref .nat [] "start_nn") (.ref .nat [BLOCK_NN] "offs_nn"))
  , .assign .bool [BLOCK_NN] "n_ctx_mask"
      (.lt ComparableDType.nat Broadcast.scalarR
        (.ref .nat [BLOCK_NN] "offs_seq") (.constNat n_ctx))
  , .assign .nat [BLOCK_NN] "token_ids"
      (.load .nat
        (.region (Region.cast input_ids)
          (.ref .nat [BLOCK_NN] "offs_seq"))
        (.maskOther
          (.ref .bool [BLOCK_NN] "n_ctx_mask")
          ((Op.constNat vob_end_id).broadcast [BLOCK_NN])))
  , .assign .bool [BLOCK_NN] "id_mask"
      (.boolAnd Broadcast.nil.consSame
        (.ge ComparableDType.nat Broadcast.scalarR
          (.ref .nat [BLOCK_NN] "token_ids") (.constNat vob_start_id))
        (.lt ComparableDType.nat Broadcast.scalarR
          (.ref .nat [BLOCK_NN] "token_ids") (.constNat vob_end_id)))
  , .assign .nat [BLOCK_NN] "token_ids"
      (.sub NumericDType.nat Broadcast.scalarR
        (.ref .nat [BLOCK_NN] "token_ids") (.constNat vob_start_id))
  , .assign .bool [BLOCK_DMODEL] "dim_mask"
      (.lt ComparableDType.nat Broadcast.scalarR
        (.ref .nat [BLOCK_DMODEL] "offs_d") (.constNat hiden_size))
  , .assign .bool [BLOCK_NN, BLOCK_DMODEL] "load_mask"
      (.boolAnd Broadcast.nil.consL.consR
        (.expandDim (⟨1, by simp⟩ : Fin ([BLOCK_NN].length + 1))
          (.ref .bool [BLOCK_NN] "id_mask"))
        (.expandDim (⟨0, by simp⟩ : Fin ([BLOCK_DMODEL].length + 1))
          (.ref .bool [BLOCK_DMODEL] "dim_mask")))
  , .assign .bool [BLOCK_NN, BLOCK_DMODEL] "store_mask"
      (.boolAnd Broadcast.nil.consL.consR
        (.expandDim (⟨1, by simp⟩ : Fin ([BLOCK_NN].length + 1))
          (.ref .bool [BLOCK_NN] "n_ctx_mask"))
        (.expandDim (⟨0, by simp⟩ : Fin ([BLOCK_DMODEL].length + 1))
          (.ref .bool [BLOCK_DMODEL] "dim_mask")))
  , .assign .real [BLOCK_NN, BLOCK_DMODEL] "vecs"
      (.load .real
        (.region weight
          (.add NumericDType.nat Broadcast.nil.consL.consR
            (.mul NumericDType.nat Broadcast.scalarR
              (.expandDim (⟨1, by simp⟩ : Fin ([BLOCK_NN].length + 1))
                (.ref .nat [BLOCK_NN] "token_ids"))
              (.constNat stride_weight_seq))
            (.expandDim (⟨0, by simp⟩ : Fin ([BLOCK_DMODEL].length + 1))
              (.ref .nat [BLOCK_DMODEL] "offs_d"))))
        (.maskOther
          (.ref .bool [BLOCK_NN, BLOCK_DMODEL] "load_mask")
          ((Op.const 0.0).broadcast [BLOCK_NN, BLOCK_DMODEL])))
  , (.store .real [BLOCK_NN, BLOCK_DMODEL]
      (.region out
        (.add NumericDType.nat Broadcast.nil.consL.consR
          (.mul NumericDType.nat Broadcast.scalarR
            (.expandDim (⟨1, by simp⟩ : Fin ([BLOCK_NN].length + 1))
              (.ref .nat [BLOCK_NN] "offs_seq"))
            (.constNat stride_out_seq))
          (.expandDim (⟨0, by simp⟩ : Fin ([BLOCK_DMODEL].length + 1))
            (.ref .nat [BLOCK_DMODEL] "offs_d"))))
      (.ref .real [BLOCK_NN, BLOCK_DMODEL] "vecs")
      (MaskOpt.mask (.ref .bool [BLOCK_NN, BLOCK_DMODEL] "store_mask")))
  ]

def embeddingProjectedBody
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat) : List Stmt :=
  embeddingPreLoop BLOCK_DMODEL BLOCK_N BLOCK_NN ++
    [.forRange "start_nn" 0 BLOCK_N BLOCK_NN
      (embeddingLoopBody weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)]

theorem embedding_kernel_toAlg_body
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat) :
    (embedding_kernel weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN).toAlgKernel.body =
      embeddingProjectedBody weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN := by
  rfl

theorem embeddingPreLoop_step_regs
    (s st : BlockState) (input_ids weight : RegionName)
    (BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (hStep :
      stepStmts (embeddingPreLoop BLOCK_DMODEL BLOCK_N BLOCK_NN) s = some st) :
    st.regs .nat [] "start_n" =
        some (Tile.scalar (s.pids 0 * BLOCK_N)) ∧
      st.regs .nat [BLOCK_NN] "offs_nn" =
        some { data := fun lane : TileIndex [BLOCK_NN] =>
          s.pids 0 * BLOCK_N + lane.1.val } ∧
      st.regs .nat [BLOCK_DMODEL] "offs_d" =
        some { data := fun lane : TileIndex [BLOCK_DMODEL] => lane.1.val } ∧
      (∀ offset,
        st.readMemValue .nat (Region.cast input_ids) offset =
          s.readMemValue .nat (Region.cast input_ids) offset) ∧
      (∀ offset, st.readMem weight offset = s.readMem weight offset) := by
  unfold embeddingPreLoop at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, Tile.bop, NumericDType.mul,
    NumericDType.add, BlockState.setReg, Option.bind] at hStep
  subst st
  constructor
  · simp [BlockState.setReg]
  · constructor
    · simp [BlockState.setReg]
    · constructor
      · simp [BlockState.setReg]
        rfl
      · constructor
        · intro offset
          rfl
        · intro offset
          rfl

theorem embeddingLoopBody_step_current_chunk_write
    (s0 st st' : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hOffsNN :
      st.regs .nat [BLOCK_NN] "offs_nn" =
        some { data := fun lane : TileIndex [BLOCK_NN] =>
          s0.pids 0 * BLOCK_N + lane.1.val })
    (hOffsD :
      st.regs .nat [BLOCK_DMODEL] "offs_d" =
        some { data := fun lane : TileIndex [BLOCK_DMODEL] => lane.1.val })
    (hInputRead :
      ∀ offset,
        st.readMemValue .nat (Region.cast input_ids) offset =
          s0.readMemValue .nat (Region.cast input_ids) offset)
    (hWeightRead : ∀ offset, st.readMem weight offset = s0.readMem weight offset)
    (hStep :
      stepStmts
        (embeddingLoopBody weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
        (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) = some st')
    (idx : TileIndex [BLOCK_NN, BLOCK_DMODEL])
    (hNoCollision :
      ∀ lane : TileIndex [BLOCK_NN, BLOCK_DMODEL],
        storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL lane →
          outOffset2D s0 stride_out_seq BLOCK_N start_nn lane =
            outOffset2D s0 stride_out_seq BLOCK_N start_nn idx →
          lane = idx)
    (hactive :
      storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
        BLOCK_DMODEL idx) :
    st'.readMem out (outOffset2D s0 stride_out_seq BLOCK_N start_nn idx) =
      embeddingSpec2D s0 weight input_ids vob_start_id vob_end_id
        stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL idx := by
  unfold embeddingLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hOffsNN, hOffsD, Tile.bop, Tile.cop,
    Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
    NumericDType.sub, NumericDType.mul, ComparableDType.lt,
    ComparableDType.ge, Bool.and_eq_true, Option.bind, hInputRead,
    hWeightRead] at hStep
  subst st'
  let actualValue : TileIndex [BLOCK_NN, BLOCK_DMODEL] → ℝ := fun i =>
    WithBot.unbotD 0
      (if
            ((vob_start_id ≤
                if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                st.readMemValue .nat input_ids
                  (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
              else vob_end_id) ∧
            (if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                st.readMemValue .nat input_ids
                  (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
              else vob_end_id) < vob_end_id) ∧
            i.2.1.val < hiden_size then
          some
            (s0.readMem weight
              (((if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                    st.readMemValue .nat input_ids
                      (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
                  else vob_end_id) - vob_start_id) * stride_weight_seq +
                i.2.1.val))
        else some (0.0 : ℝ))
  let actualActive : TileIndex [BLOCK_NN, BLOCK_DMODEL] → Prop := fun i =>
    start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx ∧
      i.2.1.val < hiden_size
  let actualOffset : TileIndex [BLOCK_NN, BLOCK_DMODEL] → Nat := fun i =>
    (start_nn + (s0.pids 0 * BLOCK_N + i.1.val)) * stride_out_seq + i.2.1.val
  have hIdxOffset :
      actualOffset idx = outOffset2D s0 stride_out_seq BLOCK_N start_nn idx := by
    exact outOffset2D_eval_add s0 stride_out_seq BLOCK_N start_nn BLOCK_NN
      BLOCK_DMODEL idx
  rw [← hIdxOffset]
  have hPi : actualActive idx := by
    exact (storeActive2D_eval_iff s0 n_ctx hiden_size BLOCK_N start_nn
      BLOCK_NN BLOCK_DMODEL idx).2 hactive
  rw [BlockState.scatter_readback_prop_masked_nd_of_true
        (region := out)
        (shape := [BLOCK_NN, BLOCK_DMODEL])
        _
        (offsetFn := actualOffset)
        (valueFn := actualValue)
        (P := actualActive)
        idx hPi
        (fun lane hlane heq =>
          hNoCollision lane
            ((storeActive2D_eval_iff s0 n_ctx hiden_size BLOCK_N start_nn
              BLOCK_NN BLOCK_DMODEL lane).1 hlane)
            (by
              rw [← outOffset2D_eval_add s0 stride_out_seq BLOCK_N start_nn
                BLOCK_NN BLOCK_DMODEL lane]
              rw [← outOffset2D_eval_add s0 stride_out_seq BLOCK_N start_nn
                BLOCK_NN BLOCK_DMODEL idx]
              exact heq))]
  exact embeddingLoopStoreValue_eval_eq_spec2D s0 st weight input_ids
    vob_start_id vob_end_id stride_weight_seq n_ctx hiden_size BLOCK_N
    start_nn BLOCK_NN BLOCK_DMODEL idx hInputRead hactive

theorem embeddingLoopBody_step_preserve_old
    (s0 st st' : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hOffsNN :
      st.regs .nat [BLOCK_NN] "offs_nn" =
        some { data := fun lane : TileIndex [BLOCK_NN] =>
          s0.pids 0 * BLOCK_N + lane.1.val })
    (hOffsD :
      st.regs .nat [BLOCK_DMODEL] "offs_d" =
        some { data := fun lane : TileIndex [BLOCK_DMODEL] => lane.1.val })
    (hInputRead :
      ∀ offset,
        st.readMemValue .nat (Region.cast input_ids) offset =
          s0.readMemValue .nat (Region.cast input_ids) offset)
    (hWeightRead : ∀ offset, st.readMem weight offset = s0.readMem weight offset)
    (hStep :
      stepStmts
        (embeddingLoopBody weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
        (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) = some st')
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
    st'.readMem out (outOffsetFull s0 stride_out_seq BLOCK_N oldIdx) =
      st.readMem out (outOffsetFull s0 stride_out_seq BLOCK_N oldIdx) := by
  unfold embeddingLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hOffsNN, hOffsD, Tile.bop, Tile.cop,
    Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
    NumericDType.sub, NumericDType.mul, ComparableDType.lt,
    ComparableDType.ge, Bool.and_eq_true, Option.bind, hInputRead,
    hWeightRead] at hStep
  subst st'
  let actualValue : TileIndex [BLOCK_NN, BLOCK_DMODEL] → ℝ := fun i =>
    WithBot.unbotD 0
      (if
          ((vob_start_id ≤
              if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                st.readMemValue .nat input_ids
                  (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
              else vob_end_id) ∧
            (if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                st.readMemValue .nat input_ids
                  (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
              else vob_end_id) < vob_end_id) ∧
            i.2.1.val < hiden_size then
          some
            (s0.readMem weight
              (((if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                    st.readMemValue .nat input_ids
                      (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
                  else vob_end_id) - vob_start_id) * stride_weight_seq +
                i.2.1.val))
        else some (0.0 : ℝ))
  let actualActive : TileIndex [BLOCK_NN, BLOCK_DMODEL] → Prop := fun i =>
    start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx ∧
      i.2.1.val < hiden_size
  let actualOffset : TileIndex [BLOCK_NN, BLOCK_DMODEL] → Nat := fun i =>
    (start_nn + (s0.pids 0 * BLOCK_N + i.1.val)) * stride_out_seq + i.2.1.val
  simpa [actualActive, actualOffset, actualValue, BlockState.setReg] using
    BlockState.scatter_prop_masked_preserves_other_offset
      out actualOffset actualValue actualActive
      (outOffsetFull s0 stride_out_seq BLOCK_N oldIdx)
      (fun lane hlane heq =>
        embeddingOldPrefix_outOffset_ne_currentChunk s0 stride_out_seq BLOCK_N
          start_nn BLOCK_NN BLOCK_DMODEL oldIdx lane hOutInj
          (hLaneBound lane
            ((storeActive2D_eval_iff s0 n_ctx hiden_size BLOCK_N start_nn
              BLOCK_NN BLOCK_DMODEL lane).1 hlane))
          hOld
          (by
            rw [← outOffset2D_eval_add s0 stride_out_seq BLOCK_N start_nn
              BLOCK_NN BLOCK_DMODEL lane]
            exact heq.symm))
      (TileShape.allIndices [BLOCK_NN, BLOCK_DMODEL]) _

theorem embeddingLoopInvariant_step_of_concrete_body
    (s0 st st' : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hPrev :
      embeddingLoopInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL start_nn st)
    (hOffsNN :
      st.regs .nat [BLOCK_NN] "offs_nn" =
        some { data := fun lane : TileIndex [BLOCK_NN] =>
          s0.pids 0 * BLOCK_N + lane.1.val })
    (hOffsD :
      st.regs .nat [BLOCK_DMODEL] "offs_d" =
        some { data := fun lane : TileIndex [BLOCK_DMODEL] => lane.1.val })
    (hInputRead :
      ∀ offset,
        st.readMemValue .nat (Region.cast input_ids) offset =
          s0.readMemValue .nat (Region.cast input_ids) offset)
    (hWeightRead : ∀ offset, st.readMem weight offset = s0.readMem weight offset)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s0 stride_out_seq BLOCK_N idx))
    (hLaneBound :
      ∀ lane : TileIndex [BLOCK_NN, BLOCK_DMODEL],
        storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL lane →
          start_nn + lane.1.val < BLOCK_N)
    (hStep :
      stepStmts
        (embeddingLoopBody weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
        (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) = some st') :
    embeddingLoopInvariant s0 weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL (start_nn + BLOCK_NN) st' := by
  apply embeddingLoopInvariant_step_of_chunk_write s0 st
  · exact hPrev
  · intro oldIdx hOldActive
    exact embeddingLoopBody_step_preserve_old s0 st st' weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
      BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn hOffsNN hOffsD hInputRead
      hWeightRead hStep oldIdx hOutInj hOldActive.1 hLaneBound
  · intro idx hStore
    exact embeddingLoopBody_step_current_chunk_write s0 st st' weight input_ids
      out vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn hOffsNN hOffsD
      hInputRead hWeightRead hStep idx
      (embeddingCurrentChunkNoCollision_of_full_injective s0 stride_out_seq
        n_ctx hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL idx hOutInj
        (hLaneBound idx hStore) hLaneBound)
      hStore

theorem embeddingLoopBody_step_preserves_context
    (s0 st st' : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hOffsNN :
      st.regs .nat [BLOCK_NN] "offs_nn" =
        some { data := fun lane : TileIndex [BLOCK_NN] =>
          s0.pids 0 * BLOCK_N + lane.1.val })
    (hOffsD :
      st.regs .nat [BLOCK_DMODEL] "offs_d" =
        some { data := fun lane : TileIndex [BLOCK_DMODEL] => lane.1.val })
    (hInputRead :
      ∀ offset,
        st.readMemValue .nat (Region.cast input_ids) offset =
          s0.readMemValue .nat (Region.cast input_ids) offset)
    (hWeightRead : ∀ offset, st.readMem weight offset = s0.readMem weight offset)
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out)
    (hStep :
      stepStmts
        (embeddingLoopBody weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
        (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) = some st') :
    st'.regs .nat [BLOCK_NN] "offs_nn" =
        some { data := fun lane : TileIndex [BLOCK_NN] =>
          s0.pids 0 * BLOCK_N + lane.1.val } ∧
      st'.regs .nat [BLOCK_DMODEL] "offs_d" =
        some { data := fun lane : TileIndex [BLOCK_DMODEL] => lane.1.val } ∧
      (∀ offset,
        st'.readMemValue .nat (Region.cast input_ids) offset =
          s0.readMemValue .nat (Region.cast input_ids) offset) ∧
      (∀ offset, st'.readMem weight offset = s0.readMem weight offset) := by
  unfold embeddingLoopBody at hStep
  simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hOffsNN, hOffsD, Tile.bop, Tile.cop,
    Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
    NumericDType.sub, NumericDType.mul, ComparableDType.lt,
    ComparableDType.ge, Bool.and_eq_true, Option.bind, hInputRead,
    hWeightRead] at hStep
  subst st'
  constructor
  · simp [BlockState.setReg, hOffsNN]
  · constructor
    · simp [BlockState.setReg, hOffsD]
    · constructor
      · intro offset
        trans st.readMemValue .nat (Region.cast input_ids) offset
        · simpa [BlockState.setReg] using
            foldl_writeMem_prop_preserves_readMemValue_of_region_ne
              (TileShape.allIndices [BLOCK_NN, BLOCK_DMODEL]) _ out input_ids
              (fun i : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
                (start_nn + (s0.pids 0 * BLOCK_N + i.1.val)) *
                    stride_out_seq + i.2.1.val)
              (fun i : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
                WithBot.unbotD 0
                  (if
                      ((vob_start_id ≤
                          if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                            st.readMemValue .nat input_ids
                              (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
                          else vob_end_id) ∧
                        (if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                            st.readMemValue .nat input_ids
                              (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
                          else vob_end_id) < vob_end_id) ∧
                        i.2.1.val < hiden_size then
                      some
                        (s0.readMem weight
                          (((if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                                st.readMemValue .nat input_ids
                                  (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
                              else vob_end_id) - vob_start_id) *
                              stride_weight_seq + i.2.1.val))
                    else some (0.0 : ℝ)))
              (fun i : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
                start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx ∧
                  i.2.1.val < hiden_size)
              .nat offset hInputOutNe
        · exact hInputRead offset
      · intro offset
        trans st.readMem weight offset
        · simpa [BlockState.setReg] using
            BlockState.scatter_prop_masked_preserves_other_region
              (region := out) (R := weight)
              (offsetFn := fun i : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
                (start_nn + (s0.pids 0 * BLOCK_N + i.1.val)) *
                    stride_out_seq + i.2.1.val)
              (valueFn := fun i : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
                WithBot.unbotD 0
                  (if
                      ((vob_start_id ≤
                          if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                            st.readMemValue .nat input_ids
                              (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
                          else vob_end_id) ∧
                        (if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                            st.readMemValue .nat input_ids
                              (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
                          else vob_end_id) < vob_end_id) ∧
                        i.2.1.val < hiden_size then
                      some
                        (s0.readMem weight
                          (((if start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx then
                                st.readMemValue .nat input_ids
                                  (start_nn + (s0.pids 0 * BLOCK_N + i.1.val))
                              else vob_end_id) - vob_start_id) *
                              stride_weight_seq + i.2.1.val))
                    else some (0.0 : ℝ)))
              (P := fun i : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
                start_nn + (s0.pids 0 * BLOCK_N + i.1.val) < n_ctx ∧
                  i.2.1.val < hiden_size)
              (off := offset)
              hWeightOutNe (TileShape.allIndices [BLOCK_NN, BLOCK_DMODEL]) _
        · exact hWeightRead offset

def embeddingLoopContextInvariant
    (s0 : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN : Nat)
    (off : Nat) (st : BlockState) : Prop :=
  embeddingLoopInvariant s0 weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
    hiden_size BLOCK_N BLOCK_DMODEL off st ∧
    st.regs .nat [BLOCK_NN] "offs_nn" =
      some { data := fun lane : TileIndex [BLOCK_NN] =>
        s0.pids 0 * BLOCK_N + lane.1.val } ∧
    st.regs .nat [BLOCK_DMODEL] "offs_d" =
      some { data := fun lane : TileIndex [BLOCK_DMODEL] => lane.1.val } ∧
    (∀ offset,
      st.readMemValue .nat (Region.cast input_ids) offset =
        s0.readMemValue .nat (Region.cast input_ids) offset) ∧
    (∀ offset, st.readMem weight offset = s0.readMem weight offset)

theorem embeddingLoopContextInvariant_init_of_preloop
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (hStep : stepStmts (embeddingPreLoop BLOCK_DMODEL BLOCK_N BLOCK_NN) s0 = some st) :
    embeddingLoopContextInvariant s0 weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
      BLOCK_N BLOCK_DMODEL BLOCK_NN 0 st := by
  rcases embeddingPreLoop_step_regs s0 st input_ids weight BLOCK_DMODEL BLOCK_N
      BLOCK_NN hStep with
    ⟨_hStart, hOffsNN, hOffsD, hInputRead, hWeightRead⟩
  exact ⟨embeddingLoopInvariant_zero s0 st weight input_ids out vob_start_id
      vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_N
      BLOCK_DMODEL,
    hOffsNN, hOffsD, hInputRead, hWeightRead⟩

theorem embeddingLoopContextInvariant_step_of_body
    (s0 st st' : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hCtx :
      embeddingLoopContextInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN start_nn st)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s0 stride_out_seq BLOCK_N idx))
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out)
    (hLaneBound :
      ∀ lane : TileIndex [BLOCK_NN, BLOCK_DMODEL],
        storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
          BLOCK_DMODEL lane →
          start_nn + lane.1.val < BLOCK_N)
    (hStep :
      stepStmts
        (embeddingLoopBody weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
        (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) = some st') :
    embeddingLoopContextInvariant s0 weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
      BLOCK_N BLOCK_DMODEL BLOCK_NN (start_nn + BLOCK_NN) st' := by
  rcases hCtx with ⟨hInv, hOffsNN, hOffsD, hInputRead, hWeightRead⟩
  have hInv' :
      embeddingLoopInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL (start_nn + BLOCK_NN) st' :=
    embeddingLoopInvariant_step_of_concrete_body s0 st st' weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
      BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn hInv hOffsNN hOffsD hInputRead
      hWeightRead hOutInj hLaneBound hStep
  rcases embeddingLoopBody_step_preserves_context s0 st st' weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
      BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn hOffsNN hOffsD hInputRead
      hWeightRead hInputOutNe hWeightOutNe hStep with
    ⟨hOffsNN', hOffsD', hInputRead', hWeightRead'⟩
  exact ⟨hInv', hOffsNN', hOffsD', hInputRead', hWeightRead'⟩

theorem embeddingLaneBound_of_blockNN_one
    (s0 : BlockState) (n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
      BLOCK_DMODEL : Nat)
    (hOne : BLOCK_NN = 1) (hStartLt : start_nn < BLOCK_N) :
    ∀ lane : TileIndex [BLOCK_NN, BLOCK_DMODEL],
      storeActive2D s0 n_ctx hiden_size BLOCK_N start_nn BLOCK_NN
        BLOCK_DMODEL lane →
      start_nn + lane.1.val < BLOCK_N := by
  subst hOne
  intro lane _hactive
  have hlane : lane.1.val = 0 := by
    omega
  omega

theorem embeddingLoopContextInvariant_body_step_exists
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hOne : BLOCK_NN = 1)
    (hStartLt : start_nn < BLOCK_N)
    (hCtx :
      embeddingLoopContextInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN start_nn st)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s0 stride_out_seq BLOCK_N idx))
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out) :
    ∃ st',
      stepStmts
        (embeddingLoopBody weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
        (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) = some st' ∧
      embeddingLoopContextInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN (start_nn + BLOCK_NN) st' := by
  rcases hCtx with ⟨hInv, hOffsNN, hOffsD, hInputRead, hWeightRead⟩
  cases hStep :
      stepStmts
        (embeddingLoopBody weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
        (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) with
  | none =>
      unfold embeddingLoopBody at hStep
      simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hOffsNN, hOffsD, Tile.bop, Tile.cop,
        Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
        NumericDType.sub, NumericDType.mul, ComparableDType.lt,
        ComparableDType.ge, Bool.and_eq_true, Option.bind, hInputRead,
        hWeightRead] at hStep
  | some st' =>
      refine ⟨st', rfl, ?_⟩
      exact embeddingLoopContextInvariant_step_of_body s0 st st' weight
        input_ids out vob_start_id vob_end_id stride_weight_seq stride_out_seq
        n_ctx hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn
        ⟨hInv, hOffsNN, hOffsD, hInputRead, hWeightRead⟩ hOutInj
        hInputOutNe hWeightOutNe
        (embeddingLaneBound_of_blockNN_one s0 n_ctx hiden_size BLOCK_N
          start_nn BLOCK_NN BLOCK_DMODEL hOne hStartLt)
        hStep

theorem embeddingForRange_context_of_preloop
    (s0 stPre stLoop : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (hOne : BLOCK_NN = 1)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s0 stride_out_seq BLOCK_N idx))
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out)
    (hPre : stepStmts (embeddingPreLoop BLOCK_DMODEL BLOCK_N BLOCK_NN) s0 = some stPre)
    (hLoop :
      stepStmt (.forRange "start_nn" 0 BLOCK_N BLOCK_NN
        (embeddingLoopBody weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)) stPre = some stLoop) :
    ∃ final,
      BLOCK_N ≤ final ∧
        embeddingLoopContextInvariant s0 weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN final stLoop := by
  have hStepNe : BLOCK_NN ≠ 0 := by
    subst hOne
    simp
  have hInit :
      embeddingLoopContextInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN 0 stPre :=
    embeddingLoopContextInvariant_init_of_preloop s0 stPre weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
      BLOCK_DMODEL BLOCK_N BLOCK_NN hPre
  obtain ⟨final, stFinal, hFor, hFinal, hCtx⟩ :=
    forRange_inv
      (idx := "start_nn") (start := 0) (stop := BLOCK_N) (step := BLOCK_NN)
      (body := embeddingLoopBody weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
      (P := embeddingLoopContextInvariant s0 weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN)
      (s_init := stPre)
      hStepNe hInit
      (by
        intro start st hlt hCtx
        exact embeddingLoopContextInvariant_body_step_exists s0 st weight
          input_ids out vob_start_id vob_end_id stride_weight_seq
          stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start
          hOne hlt hCtx hOutInj hInputOutNe hWeightOutNe)
  have hEq : stFinal = stLoop := by
    rw [hLoop] at hFor
    injection hFor with h
    exact h.symm
  subst hEq
  exact ⟨final, hFinal, hCtx⟩

theorem embeddingProjectedBody_alg_post
    (s s' : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (hOne : BLOCK_NN = 1)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out)
    (hExec :
      stepStmts
        (embeddingProjectedBody weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN) s = some s') :
    embedding_kernel_alg_post weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s s' := by
  unfold embeddingProjectedBody at hExec
  rcases (stepStmts.append_some_iff).mp hExec with ⟨stPre, hPre, hLoopStmt⟩
  simp [stepStmts] at hLoopStmt
  have hLoop :
      stepStmt (.forRange "start_nn" 0 BLOCK_N BLOCK_NN
        (embeddingLoopBody weight input_ids out
          vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
          hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)) stPre = some s' := by
    cases hAux :
        stepForRangeAux "start_nn" 0 BLOCK_N BLOCK_NN
          (embeddingLoopBody weight input_ids out
            vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
            hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN)
          stPre with
    | none =>
        simp [hAux] at hLoopStmt
    | some stLoop =>
        simp [hAux] at hLoopStmt
        subst hLoopStmt
        simp [hAux]
  obtain ⟨final, hFinal, hCtx⟩ :=
    embeddingForRange_context_of_preloop s stPre s' weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
      BLOCK_DMODEL BLOCK_N BLOCK_NN hOne hOutInj hInputOutNe hWeightOutNe
      hPre hLoop
  exact embeddingLoopInvariant_to_alg_post_of_final s s' weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
    BLOCK_DMODEL BLOCK_N BLOCK_NN final hFinal hCtx.1

theorem embedding_kernel_alg_post_of_exec
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s s' : BlockState)
    (hOne : BLOCK_NN = 1)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out)
    (hExec :
      exec (embedding_kernel weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN) s = some s') :
    embedding_kernel_alg_post weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s s' := by
  unfold exec at hExec
  rw [embedding_kernel_toAlg_body weight input_ids out vob_start_id vob_end_id
    stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
    BLOCK_NN] at hExec
  exact embeddingProjectedBody_alg_post s s' weight input_ids out vob_start_id
    vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL
    BLOCK_N BLOCK_NN hOne hOutInj hInputOutNe hWeightOutNe hExec

theorem embedding_kernel_compute_correct_of_algorithm
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState)
    (hOne : BLOCK_NN = 1)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out) :
    embedding_kernel_correct_target weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s := by
  unfold embedding_kernel_correct_target
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  intro idx hidx
  exact embedding_kernel_alg_post_of_exec weight input_ids out vob_start_id
    vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL
    BLOCK_N BLOCK_NN s s' hOne hOutInj hInputOutNe hWeightOutNe hExec idx hidx

/-- Algorithm-layer correctness for the embedding kernel. -/
theorem embedding_kernel_correct
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
    (hOne : BLOCK_NN = 1)
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out) :
    embedding_kernel_correct_target weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s :=
  embedding_kernel_compute_correct_of_algorithm weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
    hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s hOne hOutInj hInputOutNe
    hWeightOutNe

/-- Compute-facing correctness for the embedding kernel. -/
theorem embedding_kernel_compute_correct
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
    (hOne : BLOCK_NN = 1)
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out) :
    embedding_kernel_correct_target weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s :=
  embedding_kernel_compute_correct_of_algorithm weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
    hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s hOne hOutInj hInputOutNe
    hWeightOutNe

/-- Per-kernel output summary for `embedding_kernel`: the DSL surface lowers to
the algorithm layer, and the full embedding-gather loop is compute-correct — under
the no-duplicate-destination hypothesis `hOutInj`, the chunking `hOne`, and the
alias side conditions, every store-active cell holds the gathered weight row
`embeddingSpecFull`. -/
specification embedding_kernel_output_summary
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
        outOffsetFull s stride_out_seq BLOCK_N idx))
    (hOne : BLOCK_NN = 1)
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out) :
    (∃ alg, (embedding_kernel weight input_ids out
        vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
        hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN).toAlgorithm? = Except.ok alg) ∧
    embedding_kernel_correct_target weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s := by
  refine ⟨⟨_, rfl⟩, ?_⟩
  exact embedding_kernel_compute_correct weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
    hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN s hOutInj hOne hInputOutNe hWeightOutNe
/-! ## ════════ `⊨` IO face for one loop body ════════

The launched kernel is a `range(0, BLOCK_N, BLOCK_NN)` loop, so its `⊨` face would
need a full `forRange` safety walk. What the *body* does, though, is a
straight-line masked **gather**: load a `[BLOCK_NN]` row of token ids, use them as
row indices into `weight`, store a `[BLOCK_NN, BLOCK_DMODEL]` tile into `out`.
`embedding_body_slice` transcribes one body with the loop variable `start_nn` as a
`Nat` parameter — the same hand-cut single-step idiom the fused-recurrent ports
use — and that slice is what goes on the IO surface. -/

section IOFace

/-- One body of the `range(0, BLOCK_N, BLOCK_NN)` loop of `embedding_kernel`,
with the loop variable `start_nn` as a parameter. Statement for statement the
Python loop body, including the `other = vob_end_id` sentinel on the token load
(which makes an out-of-context lane fail `id_mask`) and the ℕ-truncated
`token_ids - vob_start_id`.

Two mechanical changes, both value-preserving: the pinned loop offset is folded
into the base (`start_n = pid·BLOCK_N + start_nn`, so `offs_seq = start_n +
tl.arange(0, BLOCK_NN)` is Python's `start_nn + offs_nn` lane for lane — the
resulting lane address is exactly `seqLaneIndex`), and the intermediate `offs_nn`
register disappears with it. -/
def embedding_body_slice
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) :
    ComputeKernel := triton {
  start_n = tl.program_id(0) * $(BLOCK_N) + $(start_nn)
  offs_seq = start_n + tl.arange(0, $(BLOCK_NN))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  n_ctx_mask = offs_seq < $(n_ctx)
  token_ids = tl.load($((input_ids : Region .nat)) + offs_seq, mask=n_ctx_mask, other=$(vob_end_id))
  id_mask = (token_ids >= $(vob_start_id)) & (token_ids < $(vob_end_id))
  token_ids = token_ids - $(vob_start_id)
  dim_mask = offs_d < $(hiden_size)
  load_mask = id_mask[:, None] & dim_mask[None, :]
  store_mask = n_ctx_mask[:, None] & dim_mask[None, :]
  vecs = tl.load(weight + token_ids[:, None] * $(stride_weight_seq) + offs_d[None, :],
    mask=load_mask, other=0.0)
  tl.store(out + offs_seq[:, None] * $(stride_out_seq) + offs_d[None, :], vecs, mask=store_mask)
}

theorem embedding_body_slice_toAlgorithm_supported
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) :
    ∃ alg, (embedding_body_slice weight input_ids out vob_start_id vob_end_id
      stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
      BLOCK_NN start_nn).toAlgorithm? = Except.ok alg := by
  simp [embedding_body_slice, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

theorem embedding_body_slice_correct
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        outOffset2D s stride_out_seq BLOCK_N start_nn idx))
    (hExec : exec (embedding_body_slice weight input_ids out vob_start_id
      vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL
      BLOCK_N BLOCK_NN start_nn) s = some s') :
    ∀ idx : TileIndex [BLOCK_NN, BLOCK_DMODEL],
      storeActive2D s n_ctx hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL
        idx →
      s'.readMem out (outOffset2D s stride_out_seq BLOCK_N start_nn idx) =
        embeddingSpec2D s weight input_ids vob_start_id vob_end_id
          stride_weight_seq BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL idx := by
  intro idx hactive
  simp [exec, embedding_body_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Tile.bop, Tile.cop,
    Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
    NumericDType.sub, NumericDType.mul, ComparableDType.lt, ComparableDType.ge,
    Bool.and_eq_true, Option.bind] at hExec
  subst hExec
  have hOffInj : Function.Injective
      (fun i : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        (s.pids 0 * BLOCK_N + start_nn + i.1.val) * stride_out_seq +
          i.2.1.val) := by
    simpa [outOffset2D, seqLaneIndex, dimIndex] using hOutInj
  have hPi : s.pids 0 * BLOCK_N + start_nn + idx.1.val < n_ctx ∧
      idx.2.1.val < hiden_size := hactive
  simp only [outOffset2D, seqLaneIndex, dimIndex]
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffInj idx,
    if_pos hPi]
  rcases hPi with ⟨hSeq, hDim⟩
  simp [embeddingSpec2D, tokenRaw2D, tokenIndex2D, weightOffset2D, seqLaneIndex,
    dimIndex, hSeq, hDim]

/-- Cell-level frame of a masked scatter (private copy — `bench` files are
standalone). -/
private theorem foldl_writeMem_frame {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (R : RegionName) (off : Nat) :
    ∀ l : List α, (R ≠ region ∨ ∀ k ∈ l, P k → offsetFn k ≠ off) →
      ∀ s : BlockState,
        ((l.foldl (fun acc k =>
            if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
            s).mem R off) = s.mem R off := by
  intro l
  induction l with
  | nil => intro _ s; rfl
  | cons hd tl ih =>
      intro hc s
      have htl : R ≠ region ∨ ∀ k ∈ tl, P k → offsetFn k ≠ off := by
        rcases hc with h | h
        · exact Or.inl h
        · exact Or.inr fun k hk => h k (List.mem_cons_of_mem hd hk)
      rw [List.foldl_cons, ih htl]
      by_cases hP : P hd
      · rw [if_pos hP, BlockState.writeMem_mem, if_neg ?_]
        rintro ⟨h1, h2⟩
        rcases hc with h | h
        · exact h h1
        · exact h hd List.mem_cons_self hP h2.symm
      · rw [if_neg hP]

theorem embeddingBody_flattenOk
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) :
    ((embedding_body_slice weight input_ids out vob_start_id vob_end_id
      stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
      BLOCK_NN start_nn).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [embedding_body_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  and_intros <;> simp [Op.FlattenOk.eq_def]

theorem embeddingBody_terminates
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) (s : BlockState) :
    ∃ s1, exec (embedding_body_slice weight input_ids out vob_start_id vob_end_id
      stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
      BLOCK_NN start_nn) s = some s1 := by
  simp [exec, embedding_body_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Tile.bop, Tile.cop,
    Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
    NumericDType.sub, NumericDType.mul, ComparableDType.lt, ComparableDType.ge,
    Bool.and_eq_true, Option.bind]

theorem embeddingBody_frame
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (s s' : BlockState)
    (hExec : exec (embedding_body_slice weight input_ids out vob_start_id
      vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL
      BLOCK_N BLOCK_NN start_nn) s = some s') :
    ∀ (r : RegionName) (n : Nat),
      (r ≠ out ∨ ∀ idx : TileIndex [BLOCK_NN, BLOCK_DMODEL],
        storeActive2D s n_ctx hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL
          idx → n ≠ outOffset2D s stride_out_seq BLOCK_N start_nn idx) →
      s'.mem r n = s.mem r n := by
  intro r n hcond
  simp only [storeActive2D, outOffset2D, seqLaneIndex, dimIndex] at hcond
  simp [exec, embedding_body_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Tile.bop, Tile.cop,
    Tile.expandDim, TileShape.dropInsertedIndex, NumericDType.add,
    NumericDType.sub, NumericDType.mul, ComparableDType.lt, ComparableDType.ge,
    Bool.and_eq_true, Option.bind] at hExec
  subst hExec
  rw [foldl_writeMem_frame (region := out)
    (fun i : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
      (s.pids 0 * BLOCK_N + start_nn + i.1.val) * stride_out_seq + i.2.1.val)
    _ (fun i : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
      s.pids 0 * BLOCK_N + start_nn + i.1.val < n_ctx ∧
        i.2.1.val < hiden_size) r n
    (TileShape.allIndices [BLOCK_NN, BLOCK_DMODEL]) ?_]
  · simp
  · rcases hcond with h | h
    · exact Or.inl h
    · exact Or.inr fun idx _ hidx => Ne.symm (h idx hidx)

theorem embeddingBody_traceSafe
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hidx : ∀ i : Fin BLOCK_NN, s.pids 0 * BLOCK_N + start_nn + i.val < n_ctx →
      s.pids 0 * BLOCK_N + start_nn + i.val < bounds input_ids)
    (hw : ∀ idx : TileIndex [BLOCK_NN, BLOCK_DMODEL],
      s.pids 0 * BLOCK_N + start_nn + idx.1.val < n_ctx →
      (vob_start_id ≤ tokenRaw2D s input_ids BLOCK_N start_nn idx.1 ∧
          tokenRaw2D s input_ids BLOCK_N start_nn idx.1 < vob_end_id) ∧
        idx.2.1.val < hiden_size →
      weightOffset2D s input_ids vob_start_id stride_weight_seq BLOCK_N start_nn
        idx < bounds weight)
    (hout : ∀ idx : TileIndex [BLOCK_NN, BLOCK_DMODEL],
      storeActive2D s n_ctx hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL
        idx → outOffset2D s stride_out_seq BLOCK_N start_nn idx < bounds out) :
    ((embedding_body_slice weight input_ids out vob_start_id vob_end_id
      stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
      BLOCK_NN start_nn).toAlgKernel).TraceSafe bounds s := by
  simp only [storeActive2D, outOffset2D, seqLaneIndex, dimIndex, tokenRaw2D,
    weightOffset2D, tokenIndex2D] at hw hout
  simp [Kernel.TraceSafe, embedding_body_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeList,
    Stmt.TraceSafe, Op.SafeAt, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe,
    memAccessMemorySafe, MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe, Op.PointerAddressesSafeOn, Op.MemorySafe,
    stepStmts, stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.sub, NumericDType.mul, ComparableDType.lt,
    ComparableDType.ge, Bool.and_eq_true]
  and_intros
  all_goals try exact fun a h => hidx a h
  all_goals try simp [Op.SafeAt.eq_def]
  -- the store's window
  all_goals try exact fun a b h1 h2 => hout (a, b, PUnit.unit) ⟨h1, h2⟩
  -- the gather's window: on an out-of-context lane the `other = vob_end_id`
  -- sentinel fails `id_mask` outright, so only in-context lanes need a bound
  all_goals
    intro a b h1 h2 h3
    by_cases hn : s.pids 0 * BLOCK_N + start_nn + a.val < n_ctx
    · simp only [if_pos hn] at h1 h2 ⊢
      exact hw (a, b, PUnit.unit) hn ⟨⟨h1, h2⟩, h3⟩
    · simp only [if_neg hn] at h2
      exact absurd h2 (lt_irrefl _)

/-- Region-model run of one loop body. -/
theorem embeddingBody_region_run
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (s₀ : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        outOffset2D s₀ stride_out_seq BLOCK_N start_nn idx))
    (ids : TileIndex [BLOCK_NN] → Nat)
    (xs : TileIndex [BLOCK_NN, BLOCK_DMODEL] → ℝ)
    (hid : ∀ i : TileIndex [BLOCK_NN],
      s₀.pids 0 * BLOCK_N + start_nn + i.1.val < n_ctx →
      s₀.readMemValue .nat input_ids (s₀.pids 0 * BLOCK_N + start_nn + i.1.val)
        = ids i)
    (hx : ∀ j : TileIndex [BLOCK_NN, BLOCK_DMODEL],
      (vob_start_id ≤ ids (j.1, PUnit.unit) ∧
          ids (j.1, PUnit.unit) < vob_end_id) ∧ j.2.1.val < hiden_size →
      s₀.readMem weight
          ((ids (j.1, PUnit.unit) - vob_start_id) * stride_weight_seq
            + j.2.1.val) = xs j) :
    ∃ s1, exec (embedding_body_slice weight input_ids out vob_start_id vob_end_id
        stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
        BLOCK_NN start_nn) s₀ = some s1
      ∧ (∀ j : TileIndex [BLOCK_NN, BLOCK_DMODEL],
          s₀.pids 0 * BLOCK_N + start_nn + j.1.val < n_ctx ∧
            j.2.1.val < hiden_size →
          s1.readMem out
              ((s₀.pids 0 * BLOCK_N + start_nn + j.1.val) * stride_out_seq
                + j.2.1.val)
            = if vob_start_id ≤ ids (j.1, PUnit.unit) ∧
                  ids (j.1, PUnit.unit) < vob_end_id then xs j else 0)
      ∧ (∀ (r : RegionName) (n : Nat),
          (r ≠ out ∨ ∀ j : TileIndex [BLOCK_NN, BLOCK_DMODEL],
            (s₀.pids 0 * BLOCK_N + start_nn + j.1.val < n_ctx ∧
              j.2.1.val < hiden_size) →
            n ≠ (s₀.pids 0 * BLOCK_N + start_nn + j.1.val) * stride_out_seq
              + j.2.1.val) →
          s1.mem r n = s₀.mem r n) := by
  obtain ⟨s1, hexec⟩ := embeddingBody_terminates weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
    BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn s₀
  refine ⟨s1, hexec, ?_, ?_⟩
  · intro j hact
    have h := embedding_body_slice_correct weight input_ids out vob_start_id
      vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL
      BLOCK_N BLOCK_NN start_nn s₀ s1 hOutInj hexec j hact
    have hraw : tokenRaw2D s₀ input_ids BLOCK_N start_nn j.1
        = ids (j.1, PUnit.unit) := hid (j.1, PUnit.unit) hact.1
    simp only [outOffset2D, seqLaneIndex, dimIndex] at h
    rw [h, embeddingSpec2D, hraw]
    by_cases hidm : vob_start_id ≤ ids (j.1, PUnit.unit) ∧
        ids (j.1, PUnit.unit) < vob_end_id
    · rw [if_pos hidm, if_pos hidm]
      have hwadr : weightOffset2D s₀ input_ids vob_start_id stride_weight_seq
          BLOCK_N start_nn j
          = (ids (j.1, PUnit.unit) - vob_start_id) * stride_weight_seq
            + j.2.1.val := by
        simp only [weightOffset2D, tokenIndex2D, dimIndex, hraw]
      rw [hwadr, hx j ⟨hidm, hact.2⟩]
      simp
    · rw [if_neg hidm, if_neg hidm]
      norm_num
  · intro r n hc
    refine embeddingBody_frame weight input_ids out vob_start_id vob_end_id
      stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
      BLOCK_NN start_nn s₀ s1 hexec r n ?_
    rcases hc with hr | hn
    · exact Or.inl hr
    · exact Or.inr fun j hj => hn j hj

/-- IO signature of one loop body: the `[BLOCK_NN]` token-id row is a `.nat`
channel, and the `weight` window is a **gather** through it. -/
def bodyIO (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) :
    GatherTileKernelIO where
  kernel := embedding_body_slice weight input_ids out vob_start_id vob_end_id
    stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
    BLOCK_NN start_nn
  idxbuf := input_ids
  inp := weight
  out := out
  shapeIdx := [BLOCK_NN]
  shape := [BLOCK_NN, BLOCK_DMODEL]
  readx := fun pid i => pid * BLOCK_N + start_nn + i.1.val
  read := fun _pid ids j =>
    (ids (j.1, PUnit.unit) - vob_start_id) * stride_weight_seq + j.2.1.val
  write := fun pid j =>
    (pid * BLOCK_N + start_nn + j.1.val) * stride_out_seq + j.2.1.val
  maskx := fun pid i => pid * BLOCK_N + start_nn + i.1.val < n_ctx
  readMask := fun _pid ids j =>
    (vob_start_id ≤ ids (j.1, PUnit.unit) ∧
      ids (j.1, PUnit.unit) < vob_end_id) ∧ j.2.1.val < hiden_size
  writeMask := fun pid j =>
    pid * BLOCK_N + start_nn + j.1.val < n_ctx ∧ j.2.1.val < hiden_size

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

/-- **The headline on the IO surface** for one body of `embedding_kernel`'s
`range(0, BLOCK_N, BLOCK_NN)` loop: for every disjoint flat placement of
`input_ids` / `weight` / `out`, every program id whose windows are in bounds,
every token-id row `ids` and weight tile `xs` the launch state holds on the
active lanes, the body terminates, every store-active cell of `out` holds

    if vob_start_id ≤ ids[row] < vob_end_id then xs[row, col] else 0

and every other memory cell is unchanged.

The token ids are a **channel**, not an assumption: the face quantifies over what
`input_ids` may hold and the `weight` window is a *gather* through it
(`(ids[row] − vob_start_id)·stride_weight_seq + col`), which is what
`GatherTileKernelIO` exists for. The out-of-vocabulary branch is Python's
`id_mask`, and it is genuine: an out-of-context lane loads the `other =
vob_end_id` sentinel, which fails `id_mask` outright, so no out-of-bounds row of
`weight` is ever addressed.

Dimension-general in `n_ctx`, `hiden_size`, `BLOCK_N`, `BLOCK_NN`,
`BLOCK_DMODEL`, both strides and the vocabulary window, and general in the loop
variable `start_nn`. Honest side-condition: destination injectivity at every
program id, the same hypothesis the launched kernel's summary takes. -/
specification embedding_body_io_correctness
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hOutInj : ∀ pid : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        (pid * BLOCK_N + start_nn + idx.1.val) * stride_out_seq
          + idx.2.1.val)) :
    bodyIO weight input_ids out vob_start_id vob_end_id stride_weight_seq
        stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn
      ⊨ fun _pid ids xs j =>
          if vob_start_id ≤ ids (j.1, PUnit.unit) ∧
              ids (j.1, PUnit.unit) < vob_end_id then xs j else 0 := by
  refine GatherTileKernelIO.Implements.intro _ ?_ ?_ ?_
  · exact embeddingBody_flattenOk weight input_ids out vob_start_id vob_end_id
      stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
      BLOCK_NN start_nn
  · intro bounds s ids hidpin hbx hbr hbw
    simp only [bodyIO] at hidpin hbx hbr hbw
    refine embeddingBody_traceSafe weight input_ids out vob_start_id vob_end_id
      stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
      BLOCK_NN start_nn bounds s (fun i hi => hbx (i, PUnit.unit) hi) ?_ ?_
    · intro idx hn hactive
      have hraw : tokenRaw2D s input_ids BLOCK_N start_nn idx.1
          = ids (idx.1, PUnit.unit) := hidpin (idx.1, PUnit.unit) hn
      have := hbr idx (by rw [← hraw]; exact hactive)
      simpa [weightOffset2D, tokenIndex2D, dimIndex, hraw] using this
    · intro idx hactive
      exact hbw idx hactive
  · intro s₀ ids xs hidpin hxpin
    simp only [bodyIO] at hidpin hxpin
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      embeddingBody_region_run weight input_ids out vob_start_id vob_end_id
        stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
        BLOCK_NN start_nn s₀
        (by
          have := hOutInj (s₀.pids 0)
          simpa [outOffset2D, seqLaneIndex, dimIndex] using this)
        ids xs (fun i hi => hidpin i hi) (fun j hj => hxpin j hj)
    exact ⟨s1, hexec, fun j hj => hval j hj, fun r n hc => hframe r n hc⟩

/-! ### The rounding face

The body is a masked **gather** with no arithmetic on the loaded weight row and
no `.to(...)` on the store — the data tile stays `.real` end to end, and the only
arithmetic anywhere is ℕ index arithmetic (`token_ids - vob_start_id`), which is
not a float rounding site. So the slice is **cast-free** and the exact run
transports to `execR R` for *every* rounding model.

Note the index channel needs nothing extra: it is `.nat`, and `ChanTy.nat` reads
are exact under `execR` as well. -/

/-- The slice is cast-free: `execR R` is the exact stepper. -/
private theorem embeddingBody_castFree (R : RoundingModel)
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) (s : BlockState) :
    execR R ((embedding_body_slice weight input_ids out vob_start_id vob_end_id
        stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
        BLOCK_NN start_nn).toAlgKernel) s
      = exec ((embedding_body_slice weight input_ids out vob_start_id vob_end_id
        stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
        BLOCK_NN start_nn).toAlgKernel) s := by
  simp [execR, exec, embedding_body_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, stepStmtsR, stepStmts,
    stepStmtR, stepStmt, evalOpR, evalOpR.eq_def, evalOp, evalOp.eq_def,
    BlockState.writeMemTypedR, BlockState.writeMemAsR, Option.bind, Option.map,
    Tile.bop, Tile.cop, Tile.expandDim, Tile.uop, TileShape.dropInsertedIndex,
    NumericDType.add, NumericDType.sub, NumericDType.mul, ComparableDType.lt,
    ComparableDType.ge, FloatDType.cast, Bool.and_eq_true]

/-- Per-execution safety walk **under the rounding model** — the `hts`
obligation of `GatherTileKernelIO.ImplementsR.intro`. -/
theorem embeddingBody_traceSafeR (R : RoundingModel)
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hidx : ∀ i : Fin BLOCK_NN, s.pids 0 * BLOCK_N + start_nn + i.val < n_ctx →
      s.pids 0 * BLOCK_N + start_nn + i.val < bounds input_ids)
    (hw : ∀ idx : TileIndex [BLOCK_NN, BLOCK_DMODEL],
      s.pids 0 * BLOCK_N + start_nn + idx.1.val < n_ctx →
      (vob_start_id ≤ tokenRaw2D s input_ids BLOCK_N start_nn idx.1 ∧
          tokenRaw2D s input_ids BLOCK_N start_nn idx.1 < vob_end_id) ∧
        idx.2.1.val < hiden_size →
      weightOffset2D s input_ids vob_start_id stride_weight_seq BLOCK_N start_nn
        idx < bounds weight)
    (hout : ∀ idx : TileIndex [BLOCK_NN, BLOCK_DMODEL],
      storeActive2D s n_ctx hiden_size BLOCK_N start_nn BLOCK_NN BLOCK_DMODEL
        idx → outOffset2D s stride_out_seq BLOCK_N start_nn idx < bounds out) :
    Kernel.TraceSafeR R bounds
      ((embedding_body_slice weight input_ids out vob_start_id vob_end_id
        stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
        BLOCK_NN start_nn).toAlgKernel) s := by
  simp only [storeActive2D, outOffset2D, seqLaneIndex, dimIndex, tokenRaw2D,
    weightOffset2D, tokenIndex2D] at hw hout
  simp [Kernel.TraceSafeR, embedding_body_slice, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.TraceSafeListR,
    Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
    MemAccess.SafeAtR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR, stepStmtsR, stepStmtR, evalOpR,
    evalOpR.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
    Tile.uop, TileShape.dropInsertedIndex, NumericDType.add, NumericDType.sub,
    NumericDType.mul, ComparableDType.lt, ComparableDType.ge, FloatDType.cast,
    Bool.and_eq_true]
  and_intros
  all_goals try exact fun a h => hidx a h
  all_goals try simp [Op.SafeAtR.eq_def]
  all_goals try exact fun a b h1 h2 => hout (a, b, PUnit.unit) ⟨h1, h2⟩
  -- the gather's window: on an out-of-context lane the `other = vob_end_id`
  -- sentinel fails `id_mask` outright, so only in-context lanes need a bound
  all_goals
    intro a b h1 h2 h3
    by_cases hn : s.pids 0 * BLOCK_N + start_nn + a.val < n_ctx
    · simp only [if_pos hn] at h1 h2 ⊢
      exact hw (a, b, PUnit.unit) hn ⟨⟨h1, h2⟩, h3⟩
    · simp only [if_neg hn] at h2
      exact absurd h2 (lt_irrefl _)

/-! ### ════════ ★ MAIN THEOREM (rounding face) ★ ════════ -/

/-- **The `⊨[R]` headline** for one body of `embedding_kernel`'s
`range(0, BLOCK_N, BLOCK_NN)` loop: for **every** rounding model `R`, the same
gather Hoare triple as `embedding_body_io_correctness`, but run under `execR R`
and read back as `.real`-typed cells holding `R.round .real (f …)`.

The body carries no float cast and no arithmetic on the gathered row, so the
slice is cast-free and the exact run transports verbatim. The content of the
rounding face here is exactly that: *this body introduces no rounding event of
its own*, at any `R` — an embedding gather moves weight rows bit-for-bit. -/
specification embedding_body_io_correctnessR (R : RoundingModel)
    (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hOutInj : ∀ pid : Nat, Function.Injective
      (fun idx : TileIndex [BLOCK_NN, BLOCK_DMODEL] =>
        (pid * BLOCK_N + start_nn + idx.1.val) * stride_out_seq
          + idx.2.1.val)) :
    bodyIO weight input_ids out vob_start_id vob_end_id stride_weight_seq
        stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn
      ⊨[R, FloatDType.real] fun _pid ids xs j =>
          if vob_start_id ≤ ids (j.1, PUnit.unit) ∧
              ids (j.1, PUnit.unit) < vob_end_id then xs j else 0 := by
  refine GatherTileKernelIO.ImplementsR.intro _ ?_ ?_ ?_
  · exact embeddingBody_flattenOk weight input_ids out vob_start_id vob_end_id
      stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
      BLOCK_NN start_nn
  · intro bounds s ids hidpin hbx hbr hbw
    simp only [bodyIO] at hidpin hbx hbr hbw
    refine embeddingBody_traceSafeR R weight input_ids out vob_start_id
      vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL
      BLOCK_N BLOCK_NN start_nn bounds s (fun i hi => hbx (i, PUnit.unit) hi)
      ?_ ?_
    · intro idx hn hactive
      have hraw : tokenRaw2D s input_ids BLOCK_N start_nn idx.1
          = ids (idx.1, PUnit.unit) := hidpin (idx.1, PUnit.unit) hn
      have := hbr idx (by rw [← hraw]; exact hactive)
      simpa [weightOffset2D, tokenIndex2D, dimIndex, hraw] using this
    · intro idx hactive
      exact hbw idx hactive
  · intro s₀ ids xs hidpin hxpin
    simp only [bodyIO] at hidpin hxpin
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      embeddingBody_region_run weight input_ids out vob_start_id vob_end_id
        stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
        BLOCK_NN start_nn s₀
        (by
          have := hOutInj (s₀.pids 0)
          simpa [outOffset2D, seqLaneIndex, dimIndex] using this)
        ids xs (fun i hi => hidpin i hi) (fun j hj => hxpin j hj)
    refine ⟨s1, ?_, ?_, fun r n hc => hframe r n hc⟩
    · simp only [bodyIO]
      rw [embeddingBody_castFree R weight input_ids out vob_start_id vob_end_id
        stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
        BLOCK_NN start_nn s₀]
      exact hexec
    · intro j hj
      simp only [bodyIO]
      rw [BlockState.readMemAs_real, hval j hj]
      simp

/-! ### Toward the whole-kernel face: the loop body's safety walk

The value half of the whole-kernel contract is already proven for the entire
`range(0, BLOCK_N, BLOCK_NN)` loop (`embedding_kernel_alg_post`). What an IO face
additionally needs is the **safety** walk, and `Stmt.TraceSafeR` is a different
recursive predicate from the value post-condition, so it needs its own induction.

This section supplies the per-iteration obligation: from a state whose `offs_nn` /
`offs_d` registers are the pre-loop ones and whose `input_ids` / `weight` reads
still agree with the launch state, the loop body is `TraceSafeListR`. Those four
facts are exactly the non-value conjuncts of the port's existing
`embeddingLoopContextInvariant`, so the eventual `Stmt.forRangeTraceSafeR_inv`
walk can run on the weaker context below and reuse the value invariant untouched.

Addresses appear here in the shape the evaluator produces —
`start_nn + (pid·BLOCK_N + lane)` — rather than the `seqLaneIndex` normal form;
`outOffset2D_eval_add` / `storeActive2D_eval_iff` convert between them. -/

/-- The non-value part of the loop context: the two pre-loop index registers, and
agreement of the read-only buffers with the launch state. The loop writes only
`out`, so this is preserved across iterations. -/
def embeddingSafeContext (s0 : BlockState) (weight input_ids : RegionName)
    (BLOCK_DMODEL BLOCK_N BLOCK_NN : Nat) (st : BlockState) : Prop :=
  st.regs .nat [BLOCK_NN] "offs_nn" =
      some { data := fun lane : TileIndex [BLOCK_NN] =>
        s0.pids 0 * BLOCK_N + lane.1.val } ∧
    st.regs .nat [BLOCK_DMODEL] "offs_d" =
      some { data := fun lane : TileIndex [BLOCK_DMODEL] => lane.1.val } ∧
    (∀ offset,
      st.readMemValue .nat (Region.cast input_ids) offset =
        s0.readMemValue .nat (Region.cast input_ids) offset) ∧
    (∀ offset, st.readMem weight offset = s0.readMem weight offset)

/-- The non-value context is exactly the tail of the port's full loop context. -/
theorem embeddingSafeContext_of_loopContext
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN off : Nat)
    (hCtx : embeddingLoopContextInvariant s0 weight input_ids out
      vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_N BLOCK_DMODEL BLOCK_NN off st) :
    embeddingSafeContext s0 weight input_ids BLOCK_DMODEL BLOCK_N BLOCK_NN st :=
  ⟨hCtx.2.1, hCtx.2.2.1, hCtx.2.2.2.1, hCtx.2.2.2.2⟩

/-- **The per-iteration safety obligation.** From a safe context, one loop body is
trace-safe under the rounding model, given the three window bounds at this
iteration's lanes (index reads, the token-dependent gather, and the store).

The gather bound is only required on in-context lanes: an out-of-context lane
loads the `other = vob_end_id` sentinel, which fails `id_mask` outright, so that
lane is never read-active — the same sentinel argument the one-body face makes. -/
theorem embeddingLoopBody_traceSafeR (R : RoundingModel)
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (bounds : RegionBounds)
    (hCtx : embeddingSafeContext s0 weight input_ids BLOCK_DMODEL BLOCK_N
      BLOCK_NN st)
    (hidx : ∀ lane : Fin BLOCK_NN,
      start_nn + (s0.pids 0 * BLOCK_N + lane.val) < n_ctx →
      start_nn + (s0.pids 0 * BLOCK_N + lane.val) < bounds input_ids)
    (hw : ∀ idx : TileIndex [BLOCK_NN, BLOCK_DMODEL],
      start_nn + (s0.pids 0 * BLOCK_N + idx.1.val) < n_ctx →
      (vob_start_id ≤ s0.readMemValue .nat input_ids
            (start_nn + (s0.pids 0 * BLOCK_N + idx.1.val)) ∧
          s0.readMemValue .nat input_ids
            (start_nn + (s0.pids 0 * BLOCK_N + idx.1.val)) < vob_end_id) ∧
        idx.2.1.val < hiden_size →
      (s0.readMemValue .nat input_ids
            (start_nn + (s0.pids 0 * BLOCK_N + idx.1.val)) - vob_start_id)
          * stride_weight_seq + idx.2.1.val < bounds weight)
    (hout : ∀ idx : TileIndex [BLOCK_NN, BLOCK_DMODEL],
      start_nn + (s0.pids 0 * BLOCK_N + idx.1.val) < n_ctx →
      idx.2.1.val < hiden_size →
      (start_nn + (s0.pids 0 * BLOCK_N + idx.1.val)) * stride_out_seq
        + idx.2.1.val < bounds out) :
    Stmt.TraceSafeListR R bounds
      (embeddingLoopBody weight input_ids out vob_start_id vob_end_id
        stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
        BLOCK_NN)
      (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) := by
  obtain ⟨hOffsNN, hOffsD, hInputRead, hWeightRead⟩ := hCtx
  -- the pins arrive wrapped in `Region.cast`; the body's loads mention the bare
  -- region name, so strip the wrapper before it reaches the simp set
  have hIR : ∀ offset,
      st.readMemValue .nat input_ids offset
        = s0.readMemValue .nat input_ids offset := hInputRead
  unfold embeddingLoopBody
  simp [Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.SafeAtR,
    MaskOpt.ActiveR, MemAccess.SafeAtR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR, stepStmtsR, stepStmtR, evalOpR,
    evalOpR.eq_def, hOffsNN, hOffsD, hIR, hWeightRead, Option.bind,
    Option.map, Tile.bop, Tile.cop, Tile.expandDim, Tile.uop,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.sub,
    NumericDType.mul, ComparableDType.lt, ComparableDType.ge, FloatDType.cast,
    Bool.and_eq_true]
  and_intros
  all_goals try exact fun a h => hidx a h
  all_goals try simp [Op.SafeAtR.eq_def]
  all_goals try exact fun a b h1 h2 => hout (a, b, PUnit.unit) h1 h2
  all_goals
    intro a b h1 h2 h3
    by_cases hn : start_nn + (s0.pids 0 * BLOCK_N + a.val) < n_ctx
    · simp only [if_pos hn] at h1 h2 ⊢
      exact hw (a, b, PUnit.unit) hn ⟨⟨h1, h2⟩, h3⟩
    · simp only [if_neg hn] at h2
      exact absurd h2 (lt_irrefl _)

/-- The loop body list is cast-free: the two steppers agree statement for
statement (no float cast anywhere in the body). -/
private theorem embeddingLoopBody_castFreeList (R : RoundingModel)
    (s0 : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat) (st : BlockState)
    (hOffsNN :
      st.regs .nat [BLOCK_NN] "offs_nn" =
        some { data := fun lane : TileIndex [BLOCK_NN] =>
          s0.pids 0 * BLOCK_N + lane.1.val })
    (hOffsD :
      st.regs .nat [BLOCK_DMODEL] "offs_d" =
        some { data := fun lane : TileIndex [BLOCK_DMODEL] => lane.1.val }) :
    stepStmtsR R
        (embeddingLoopBody weight input_ids out vob_start_id vob_end_id
          stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
          BLOCK_NN) (st.setReg "start_nn" .nat [] (Tile.scalar start_nn))
      = stepStmts
        (embeddingLoopBody weight input_ids out vob_start_id vob_end_id
          stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N
          BLOCK_NN) (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) := by
  unfold embeddingLoopBody
  simp [stepStmtsR, stepStmts, stepStmtR, stepStmt, evalOpR, evalOpR.eq_def,
    evalOp, evalOp.eq_def, BlockState.writeMemTypedR, BlockState.writeMemAsR,
    hOffsNN, hOffsD, Option.bind, Option.map, Tile.bop, Tile.cop,
    Tile.expandDim, Tile.uop, TileShape.dropInsertedIndex, NumericDType.add,
    NumericDType.sub, NumericDType.mul, ComparableDType.lt, ComparableDType.ge,
    FloatDType.cast, Bool.and_eq_true]

/-- One body step under the rounding model exists and re-establishes the safe
context. The step is the exact one (`embeddingLoopBody_castFreeList`), so the
port's existing preservation lemma is reused unchanged. -/
theorem embeddingSafeContext_stepR (R : RoundingModel)
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn : Nat)
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out)
    (hCtx : embeddingSafeContext s0 weight input_ids BLOCK_DMODEL BLOCK_N
      BLOCK_NN st) :
    ∃ st',
      stepStmtsR R
          (embeddingLoopBody weight input_ids out vob_start_id vob_end_id
            stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL
            BLOCK_N BLOCK_NN)
          (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) = some st'
        ∧ embeddingSafeContext s0 weight input_ids BLOCK_DMODEL BLOCK_N BLOCK_NN
            st' := by
  obtain ⟨hOffsNN, hOffsD, hInputRead, hWeightRead⟩ := hCtx
  rw [embeddingLoopBody_castFreeList R s0 weight input_ids out vob_start_id
    vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL
    BLOCK_N BLOCK_NN start_nn st hOffsNN hOffsD]
  -- the body is straight-line: with the two index registers pinned it always steps
  have hex : ∃ st',
      stepStmts
          (embeddingLoopBody weight input_ids out vob_start_id vob_end_id
            stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL
            BLOCK_N BLOCK_NN)
          (st.setReg "start_nn" .nat [] (Tile.scalar start_nn)) = some st' := by
    unfold embeddingLoopBody
    simp [stepStmts, stepStmt, evalOp, evalOp.eq_def, hOffsNN, hOffsD,
      Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
      TileShape.dropInsertedIndex, NumericDType.add, NumericDType.sub,
      NumericDType.mul, ComparableDType.lt, ComparableDType.ge,
      Bool.and_eq_true]
  obtain ⟨st', hstep⟩ := hex
  refine ⟨st', hstep, ?_⟩
  exact embeddingLoopBody_step_preserves_context s0 st st' weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
    BLOCK_DMODEL BLOCK_N BLOCK_NN start_nn hOffsNN hOffsD hInputRead
    hWeightRead hInputOutNe hWeightOutNe hstep

/-- **The whole loop is trace-safe under the rounding model.** The
`Stmt.forRangeTraceSafeR_inv` walk over the safe context: every in-range
iteration has a trace-safe body (`embeddingLoopBody_traceSafeR`) and steps to a
state that re-establishes the context (`embeddingSafeContext_stepR`).

The three window bounds are taken at **whole-tile** granularity — over
`TileIndex [BLOCK_N, BLOCK_DMODEL]` — which is what an IO face supplies; with
`hOne : BLOCK_NN = 1` iteration `c` is exactly full-tile row `c`, so the
per-iteration bounds are instantiations of the whole-tile ones. -/
theorem embedding_loop_traceSafeR (R : RoundingModel)
    (s0 st : BlockState) (weight input_ids out : RegionName)
    (vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx
      hiden_size BLOCK_DMODEL BLOCK_N : Nat)
    (bounds : RegionBounds)
    (hInputOutNe : input_ids ≠ out)
    (hWeightOutNe : weight ≠ out)
    (hCtx : embeddingSafeContext s0 weight input_ids BLOCK_DMODEL BLOCK_N 1 st)
    (hidxAll : ∀ row : Fin BLOCK_N,
      s0.pids 0 * BLOCK_N + row.val < n_ctx →
      s0.pids 0 * BLOCK_N + row.val < bounds input_ids)
    (hwAll : ∀ idx : TileIndex [BLOCK_N, BLOCK_DMODEL],
      s0.pids 0 * BLOCK_N + idx.1.val < n_ctx →
      (vob_start_id ≤ s0.readMemValue .nat input_ids
            (s0.pids 0 * BLOCK_N + idx.1.val) ∧
          s0.readMemValue .nat input_ids
            (s0.pids 0 * BLOCK_N + idx.1.val) < vob_end_id) ∧
        idx.2.1.val < hiden_size →
      (s0.readMemValue .nat input_ids (s0.pids 0 * BLOCK_N + idx.1.val)
          - vob_start_id) * stride_weight_seq + idx.2.1.val < bounds weight)
    (houtAll : ∀ idx : TileIndex [BLOCK_N, BLOCK_DMODEL],
      s0.pids 0 * BLOCK_N + idx.1.val < n_ctx →
      idx.2.1.val < hiden_size →
      (s0.pids 0 * BLOCK_N + idx.1.val) * stride_out_seq + idx.2.1.val
        < bounds out) :
    Stmt.forRangeTraceSafeR R bounds "start_nn" 0 BLOCK_N 1
      (embeddingLoopBody weight input_ids out vob_start_id vob_end_id
        stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N 1)
      st := by
  refine Stmt.forRangeTraceSafeR_inv R bounds "start_nn" BLOCK_N 1
    (embeddingLoopBody weight input_ids out vob_start_id vob_end_id
      stride_weight_seq stride_out_seq n_ctx hiden_size BLOCK_DMODEL BLOCK_N 1)
    (fun _c stc =>
      embeddingSafeContext s0 weight input_ids BLOCK_DMODEL BLOCK_N 1 stc)
    ?_ 0 st hCtx
  intro c stc hc hCtxc
  refine ⟨?_, embeddingSafeContext_stepR R s0 stc weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
    BLOCK_DMODEL BLOCK_N 1 c hInputOutNe hWeightOutNe hCtxc⟩
  -- `BLOCK_NN = 1`, so the single lane is `0` and `c + (pid·BLOCK_N + 0)` is
  -- full-tile row `c`
  refine embeddingLoopBody_traceSafeR R s0 stc weight input_ids out
    vob_start_id vob_end_id stride_weight_seq stride_out_seq n_ctx hiden_size
    BLOCK_DMODEL BLOCK_N 1 c bounds hCtxc ?_ ?_ ?_
  · intro lane hlt
    have hl0 : lane.val = 0 := by omega
    have := hidxAll ⟨c, hc⟩ (by simpa [hl0, Nat.add_comm] using hlt)
    simpa [hl0, Nat.add_comm] using this
  · intro idx hn hact
    have hl0 : idx.1.val = 0 := by omega
    have := hwAll (⟨c, hc⟩, idx.2.1, PUnit.unit)
      (by simpa [hl0, Nat.add_comm] using hn)
      (by simpa [hl0, Nat.add_comm] using hact)
    simpa [hl0, Nat.add_comm] using this
  · intro idx hn hd
    have hl0 : idx.1.val = 0 := by omega
    have := houtAll (⟨c, hc⟩, idx.2.1, PUnit.unit)
      (by simpa [hl0, Nat.add_comm] using hn) hd
    simpa [hl0, Nat.add_comm] using this

end IOFace

end VeriTile.Bench.TritonBenchG.EmbeddingTritonKernel
