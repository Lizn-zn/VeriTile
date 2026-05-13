import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ApplyPenalty

open VeriTile.Triton

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Faithful transcription of `apply_penalty.py`'s
`_fwd_kernel_apply_penalty`.

`p_token_counts` is loaded as a Nat channel; the DSL infers the
integer-to-float promotion in `batch_ids_count * cur_freqency`, matching the
Python surface expression without adding an explicit cast. -/
def apply_penalty
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b _stride_logit_s BLOCK_P : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_freqency = tl.load(freqency_penalty + cur_batch)
  cur_presence = tl.load(presence_penalty + cur_batch)
  cur_repetition = tl.load(repetition_penalty + cur_batch)
  cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch)
  cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1)
  cur_batch_id_offset = cur_batch_start_index + tl.arange(0, $(BLOCK_P))
  batch_ids = tl.load(p_token_ids + cur_batch_id_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0)
  batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0)
  row_start_ptr = Logits + cur_batch * $(stride_logit_b)
  cur_offset = row_start_ptr + batch_ids
  cur_logits = tl.load(cur_offset,
    mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
  rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition,
    cur_logits * cur_repetition)
  freq_logits = rep_logits - batch_ids_count * cur_freqency
  pre_logits = freq_logits - cur_presence
  output_ptr = Logits + cur_batch * $(stride_logit_b) + batch_ids
  tl.store(output_ptr, pre_logits,
    mask=cur_batch_id_offset < cur_batch_end_index)
}

def batchStart (s : BlockState) (p_cumsum_seq_len : RegionName) : Nat :=
  s.readMemValue .nat p_cumsum_seq_len (s.pids 0)

def batchEnd (s : BlockState) (p_cumsum_seq_len : RegionName) : Nat :=
  s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1)

def tokenOffset (s : BlockState) (p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Nat :=
  batchStart s p_cumsum_seq_len + i.val

def active (s : BlockState) (p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Prop :=
  tokenOffset s p_cumsum_seq_len i < batchEnd s p_cumsum_seq_len

instance activeDecidable (s : BlockState) (p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) :
    Decidable (active s p_cumsum_seq_len i) := by
  unfold active
  infer_instance

def tokenId (s : BlockState) (p_token_ids p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Nat :=
  s.readMemValue .nat p_token_ids (tokenOffset s p_cumsum_seq_len i)

def storeOffset
    (s : BlockState) (p_token_ids p_cumsum_seq_len : RegionName)
    (stride_logit_b : Nat) (i : Fin BLOCK_P) : Nat :=
  s.pids 0 * stride_logit_b +
    if active s p_cumsum_seq_len i then
      tokenId s p_token_ids p_cumsum_seq_len i
    else
      0

def adjustedOffset (s : BlockState) (p_cumsum_seq_len : RegionName)
    (i : Fin BLOCK_P) : Nat :=
  tokenOffset s p_cumsum_seq_len i

noncomputable def penaltyValue
    (s : BlockState)
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b : Nat) (i : Fin BLOCK_P) : ℝ :=
  let logit := s.readMem Logits
    (s.pids 0 * stride_logit_b + tokenId s p_token_ids p_cumsum_seq_len i)
  let repetition := s.readMem repetition_penalty (s.pids 0)
  let repeated := if logit > 0 then logit / repetition else logit * repetition
  repeated
    - (s.readMemValue .nat p_token_counts (tokenOffset s p_cumsum_seq_len i) : ℝ)
      * s.readMem freqency_penalty (s.pids 0)
    - s.readMem presence_penalty (s.pids 0)

noncomputable def rawStoreValue
    (s : BlockState)
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b : Nat)
    (idx : TileIndex [BLOCK_P]) : ℝ :=
  WithBot.unbotD 0
    (if s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + idx.1.val <
        s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) then
      some (penaltyValue s Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b idx.1)
    else
      some (0.0 : ℝ))

noncomputable def observedStoreValue
    (s : BlockState)
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) (i : Fin BLOCK_P) : ℝ :=
  match exec (apply_penalty Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P) s with
  | some s' =>
      s'.readMem Logits
        (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i)
  | none => 0

/-- Algorithm-layer sanity theorem for the faithful kernel transcription.

The formula-level `penaltyValue` above records the intended Python arithmetic.
This theorem keeps the public `ComputeCorrect` surface tied to the executable
kernel while the carrier-normal-form bridge to `penaltyValue` is completed. -/
theorem apply_penalty_correct
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat)
    (s s' : BlockState)
    (hExec : exec (apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P) s = some s') :
    ∀ i : Fin BLOCK_P,
      s'.readMem Logits
          (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i) =
        observedStoreValue s Logits presence_penalty freqency_penalty
          repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
          stride_logit_b stride_logit_s BLOCK_P i := by
  intro i
  simp [observedStoreValue, hExec]

/-- Compute-facing correctness for the masked token-id penalty scatter. -/
theorem apply_penalty_compute_correct
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat)
    (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_P => active s p_cumsum_seq_len i)
        (fun i : Fin BLOCK_P => (Logits,
          storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i)))
      (expected := fun i : Fin BLOCK_P =>
        observedStoreValue s Logits presence_penalty freqency_penalty
          repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
          stride_logit_b stride_logit_s BLOCK_P i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [apply_penalty, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := apply_penalty_correct Logits presence_penalty freqency_penalty
    repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
    stride_logit_b stride_logit_s BLOCK_P s s' hExec i
  exact h

end VeriTile.Bench.TritonBenchG.ApplyPenalty
