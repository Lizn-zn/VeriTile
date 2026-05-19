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

/-- The full apply-penalty surface lowers to the algorithm layer. -/
theorem apply_penalty_toAlgorithm_supported
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    ∃ alg, (apply_penalty Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P).toAlgorithm? = Except.ok alg := by
  simp [apply_penalty, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

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

/-- The "active" store address (no mask conditional) used for the readback
spec. When `active s i` holds, this equals `storeOffset s ... i`. -/
def activeStoreAddr
    (s : BlockState) (p_token_ids p_cumsum_seq_len : RegionName)
    (stride_logit_b : Nat) (i : Fin BLOCK_P) : Nat :=
  s.pids 0 * stride_logit_b + tokenId s p_token_ids p_cumsum_seq_len i

theorem storeOffset_eq_active
    (s : BlockState) (p_token_ids p_cumsum_seq_len : RegionName)
    (stride_logit_b : Nat) (i : Fin BLOCK_P)
    (hAct : active s p_cumsum_seq_len i) :
    storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i
      = activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i := by
  unfold storeOffset activeStoreAddr
  simp [hAct]

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

/-- Formula-level correctness target for the Python apply-penalty store.

The remaining proof work is the data-dependent masked scatter readback from the
actual kernel state to this target under a token-id uniqueness hypothesis. -/
def apply_penalty_correct_target
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b BLOCK_P : Nat) (s s' : BlockState) : Prop :=
  ∀ i : Fin BLOCK_P,
    s'.readMem Logits
        (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i) =
      if active s p_cumsum_seq_len i then
        penaltyValue s Logits presence_penalty freqency_penalty
          repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
          stride_logit_b i
      else
        s.readMem Logits
          (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i)

/-! ### Bench-local masked-injectivity scatter readback

`BlockState.scatter_readback_prop_masked_nd` requires *global* injectivity of
the offset function. Apply-penalty's offset function `λ i. base + (if active i
then tokenId i else 0)` is not globally injective (all inactive lanes alias to
`base + 0`), but inactive lanes never write — the masked store guards them.
The following helper weakens the injectivity precondition to injectivity over
the masked-true subset, which is all that the underlying proof uses. -/

private theorem foldl_writeMem_masked_preserves_local {α : Type}
    {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k, k ∈ l → (mask k = Bool.true) → offsetFn k ≠ o) →
      ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region o)
      = s.readMem region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s h
    rw [List.foldl_cons]
    have htl : ∀ k, k ∈ tl → (mask k = Bool.true) → offsetFn k ≠ o :=
      fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
    by_cases hmaskhd : mask hd = Bool.true
    · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
      simp only [hmaskhd, if_true]
      rw [ih _ htl]
      rw [BlockState.writeMem_readMem]
      show (if region = region ∧ o = offsetFn hd then valueFn hd else s.readMem region o)
          = s.readMem region o
      rw [if_neg]
      rintro ⟨_, h_eq⟩
      exact hhd h_eq.symm
    · have hmaskhd' : mask hd = Bool.false := by
        rcases hmaskFalse : mask hd
        · rfl
        · exact absurd hmaskFalse hmaskhd
      simp only [hmaskhd', if_false, Bool.false_eq_true]
      exact ih _ htl

/-! ## apply_penalty correctness — status

A direct algorithm-layer cellwise correctness theorem against `penaltyValue`
under a uniqueness hypothesis (`tokenId` injective) is feasible in principle
but requires bridging the bench-local masked-injectivity helper through the
kernel's data-dependent address chain (multi-day work). The pieces in place:

- `tokenId`, `storeOffset`, `activeStoreAddr`, `penaltyValue` (above): the
  full formula-level spec.
- `storeOffset_eq_active` (above): bridges active-lane offset forms.
- `foldl_writeMem_masked_preserves_local` (above): masked-preservation under
  per-element offset-disjointness.

Future closure: state `apply_penalty_correct` against `penaltyValue` under
`hUniq : Function.Injective (tokenId s p_token_ids p_cumsum_seq_len)` and
discharge via masked scatter-readback. -/

end VeriTile.Bench.TritonBenchG.ApplyPenalty
