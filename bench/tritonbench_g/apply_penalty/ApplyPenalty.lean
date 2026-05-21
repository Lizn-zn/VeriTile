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

/-- Store-value shape produced by the expanded kernel execution.

This keeps the load-mask `other=0.0` and masked count contribution explicit,
matching the expression generated by `simp [exec, apply_penalty, ...]`. For
active lanes it reduces to `penaltyValue`; inactive lanes are not written. -/
noncomputable def penaltyStoreValue
    (s : BlockState)
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b : Nat) (i : Fin BLOCK_P) : ℝ := by
  classical
  exact WithBot.unbotD 0
    (match
      match
        if some (0.0 : ℝ) <
            (if active s p_cumsum_seq_len i then
              some (s.readMem Logits
                (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i))
            else
              some (0.0 : ℝ)) then
          match
            if active s p_cumsum_seq_len i then
              some (s.readMem Logits
                (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i))
            else
              some (0.0 : ℝ) with
          | some logit => some (logit / s.readMem repetition_penalty (s.pids 0))
          | none => none
        else
          match
            if active s p_cumsum_seq_len i then
              some (s.readMem Logits
                (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b i))
            else
              some (0.0 : ℝ) with
          | some logit => some (logit * s.readMem repetition_penalty (s.pids 0))
          | none => none with
      | some repeated =>
        some (repeated -
          (if active s p_cumsum_seq_len i then
            (s.readMemValue .nat p_token_counts
              (tokenOffset s p_cumsum_seq_len i) : ℝ)
              * s.readMem freqency_penalty (s.pids 0)
          else
            0))
      | none => none with
    | some freq => some (freq - s.readMem presence_penalty (s.pids 0))
    | none => none)

theorem penaltyStoreValue_active_eq_penaltyValue
    (s : BlockState)
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b : Nat) (i : Fin BLOCK_P)
    (hAct : active s p_cumsum_seq_len i) :
    penaltyStoreValue s Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b i =
      penaltyValue s Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b i := by
  simp [penaltyStoreValue, penaltyValue, hAct, storeOffset_eq_active,
    activeStoreAddr]
  by_cases hPos :
      0.0 < s.readMem Logits
        (s.pids 0 * stride_logit_b +
          tokenId s p_token_ids p_cumsum_seq_len i)
  · simp [hPos]
    intro hle
    nlinarith
  · simp [hPos]
    intro hPos'
    nlinarith

/-- Formula-level correctness target for the Python apply-penalty store.

The masked scatter readback part is captured by
`apply_penalty_masked_scatter_readback`; the remaining full-kernel bridge is
to reduce the executed kernel state to that foldl/write form. -/
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

private theorem scatter_readback_prop_masked_list_active_injective {α : Type}
    {region : RegionName}
    (l : List α) (s : BlockState)
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (i : α) (h_nodup : l.Nodup) (h_mem : i ∈ l)
    (h_no_collision :
      ∀ k, k ∈ l → P k → offsetFn k = offsetFn i → k = i) :
    (l.foldl
       (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region (offsetFn i)
    = if P i then valueFn i else s.readMem region (offsetFn i) := by
  by_cases hPi : P i
  · simp only [hPi, if_true]
    exact BlockState.scatter_readback_prop_masked_list_of_true l s offsetFn valueFn P
      i h_nodup h_mem hPi h_no_collision
  · simp only [hPi, if_false]
    have h_preserve :
        ∀ k, k ∈ l → (decide (P k) = Bool.true) → offsetFn k ≠ offsetFn i := by
      intro k hk hPkDec heq
      have hPk : P k := by simpa only [decide_eq_true_eq] using hPkDec
      have hki := h_no_collision k hk hPk heq
      have hPi' : P i := by simpa [hki] using hPk
      exact hPi hPi'
    have hstep :
        (fun (acc : BlockState) k =>
          if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
          =
        (fun (acc : BlockState) k =>
          if decide (P k) then acc.writeMem region (offsetFn k) (valueFn k) else acc) := by
      funext acc k
      by_cases hk : P k <;> simp [hk]
    rw [show List.foldl
        (fun (acc : BlockState) k =>
          if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s l =
      List.foldl
        (fun (acc : BlockState) k =>
          if decide (P k) then acc.writeMem region (offsetFn k) (valueFn k) else acc) s l by
          rw [hstep]]
    exact foldl_writeMem_masked_preserves_local offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l s h_preserve

theorem apply_penalty_masked_scatter_store_value_readback_from
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b BLOCK_P : Nat) (sInit s : BlockState)
    (hUniq :
      Function.Injective
        (fun i : Fin BLOCK_P => tokenId s p_token_ids p_cumsum_seq_len i)) :
    ∀ i : Fin BLOCK_P,
      ((List.finRange BLOCK_P).foldl
          (fun (acc : BlockState) (k : Fin BLOCK_P) =>
            if active s p_cumsum_seq_len k then
              acc.writeMem Logits
                (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b k)
                (penaltyStoreValue s Logits presence_penalty freqency_penalty
                  repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
                  stride_logit_b k)
            else
              acc)
          sInit).readMem Logits
          (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i) =
        if active s p_cumsum_seq_len i then
          penaltyStoreValue s Logits presence_penalty freqency_penalty
            repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
            stride_logit_b i
        else
          sInit.readMem Logits
            (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i) := by
  intro i
  have h_readback :=
    scatter_readback_prop_masked_list_active_injective
      (region := (Logits : RegionName))
      (List.finRange BLOCK_P) sInit
      (fun k : Fin BLOCK_P =>
        activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b k)
      (fun k : Fin BLOCK_P =>
        penaltyStoreValue s Logits presence_penalty freqency_penalty
          repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
          stride_logit_b k)
      (fun k : Fin BLOCK_P => active s p_cumsum_seq_len k)
      i (List.nodup_finRange BLOCK_P) (List.mem_finRange i)
      (by
        intro k _hk _hAct heq
        apply hUniq
        unfold activeStoreAddr at heq
        exact Nat.add_left_cancel heq)
  have hfun :
      (fun (acc : BlockState) (k : Fin BLOCK_P) =>
        if active s p_cumsum_seq_len k then
          acc.writeMem Logits
            (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b k)
            (penaltyStoreValue s Logits presence_penalty freqency_penalty
              repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
              stride_logit_b k)
        else
          acc)
        =
      (fun (acc : BlockState) (k : Fin BLOCK_P) =>
        if (fun k : Fin BLOCK_P => active s p_cumsum_seq_len k) k then
          acc.writeMem Logits
            ((fun k : Fin BLOCK_P =>
              activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b k) k)
            ((fun k : Fin BLOCK_P =>
              penaltyStoreValue s Logits presence_penalty freqency_penalty
                repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
                stride_logit_b k) k)
        else
          acc) := by
    funext (acc : BlockState) (k : Fin BLOCK_P)
    by_cases hAct : active s p_cumsum_seq_len k
    · simp [hAct, storeOffset_eq_active]
    · simp [hAct]
  rw [hfun]
  simpa using h_readback

theorem apply_penalty_masked_scatter_store_value_readback_tile_from
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b BLOCK_P : Nat) (sInit s : BlockState)
    (hUniq :
      Function.Injective
        (fun i : Fin BLOCK_P => tokenId s p_token_ids p_cumsum_seq_len i)) :
    ∀ idx : TileIndex [BLOCK_P],
      ((TileShape.allIndices [BLOCK_P]).foldl
          (fun (acc : BlockState) (k : TileIndex [BLOCK_P]) =>
            if active s p_cumsum_seq_len k.1 then
              acc.writeMem Logits
                (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b k.1)
                (penaltyStoreValue s Logits presence_penalty freqency_penalty
                  repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
                  stride_logit_b k.1)
            else
              acc)
          sInit).readMem Logits
          (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b idx.1) =
        if active s p_cumsum_seq_len idx.1 then
          penaltyStoreValue s Logits presence_penalty freqency_penalty
            repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
            stride_logit_b idx.1
        else
          sInit.readMem Logits
            (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b idx.1) := by
  intro idx
  have h_readback :=
    scatter_readback_prop_masked_list_active_injective
      (region := (Logits : RegionName))
      (TileShape.allIndices [BLOCK_P]) sInit
      (fun k : TileIndex [BLOCK_P] =>
        activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b k.1)
      (fun k : TileIndex [BLOCK_P] =>
        penaltyStoreValue s Logits presence_penalty freqency_penalty
          repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
          stride_logit_b k.1)
      (fun k : TileIndex [BLOCK_P] => active s p_cumsum_seq_len k.1)
      idx (TileShape.allIndices_nodup [BLOCK_P]) (TileShape.mem_allIndices [BLOCK_P] idx)
      (by
        intro k _hk _hAct heq
        have hFin : k.1 = idx.1 := by
          apply hUniq
          unfold activeStoreAddr at heq
          exact Nat.add_left_cancel heq
        cases k with
        | mk kHead kTail =>
          cases idx with
          | mk idxHead idxTail =>
            simp only at hFin
            subst idxHead
            cases kTail
            cases idxTail
            rfl)
  have hfun :
      (fun (acc : BlockState) (k : TileIndex [BLOCK_P]) =>
        if active s p_cumsum_seq_len k.1 then
          acc.writeMem Logits
            (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b k.1)
            (penaltyStoreValue s Logits presence_penalty freqency_penalty
              repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
              stride_logit_b k.1)
        else
          acc)
        =
      (fun (acc : BlockState) (k : TileIndex [BLOCK_P]) =>
        if (fun k : TileIndex [BLOCK_P] => active s p_cumsum_seq_len k.1) k then
          acc.writeMem Logits
            ((fun k : TileIndex [BLOCK_P] =>
              activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b k.1) k)
            ((fun k : TileIndex [BLOCK_P] =>
              penaltyStoreValue s Logits presence_penalty freqency_penalty
                repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
                stride_logit_b k.1) k)
        else
          acc) := by
    funext (acc : BlockState) (k : TileIndex [BLOCK_P])
    by_cases hAct : active s p_cumsum_seq_len k.1
    · simp [hAct, storeOffset_eq_active]
    · simp [hAct]
  rw [hfun]
  simpa using h_readback

theorem apply_penalty_masked_scatter_store_value_readback_tile_from'
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b BLOCK_P : Nat) {sInit : BlockState} (s : BlockState)
    (hUniq :
      Function.Injective
        (fun i : Fin BLOCK_P => tokenId s p_token_ids p_cumsum_seq_len i)) :
    ∀ idx : TileIndex [BLOCK_P],
      ((TileShape.allIndices [BLOCK_P]).foldl
          (fun (acc : BlockState) (k : TileIndex [BLOCK_P]) =>
            if active s p_cumsum_seq_len k.1 then
              acc.writeMem Logits
                (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b k.1)
                (penaltyStoreValue s Logits presence_penalty freqency_penalty
                  repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
                  stride_logit_b k.1)
            else
              acc)
          sInit).readMem Logits
          (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b idx.1) =
        if active s p_cumsum_seq_len idx.1 then
          penaltyStoreValue s Logits presence_penalty freqency_penalty
            repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
            stride_logit_b idx.1
        else
          sInit.readMem Logits
            (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b idx.1) := by
  exact apply_penalty_masked_scatter_store_value_readback_tile_from
    Logits presence_penalty freqency_penalty repetition_penalty p_token_ids
    p_token_counts p_cumsum_seq_len stride_logit_b BLOCK_P sInit s hUniq

theorem apply_penalty_masked_scatter_readback_from
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b BLOCK_P : Nat) (sInit s : BlockState)
    (hUniq :
      Function.Injective
        (fun i : Fin BLOCK_P => tokenId s p_token_ids p_cumsum_seq_len i)) :
    ∀ i : Fin BLOCK_P,
      ((List.finRange BLOCK_P).foldl
          (fun (acc : BlockState) (k : Fin BLOCK_P) =>
            if active s p_cumsum_seq_len k then
              acc.writeMem Logits
                (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b k)
                (penaltyValue s Logits presence_penalty freqency_penalty
                  repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
                  stride_logit_b k)
            else
              acc)
          sInit).readMem Logits
          (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i) =
        if active s p_cumsum_seq_len i then
          penaltyValue s Logits presence_penalty freqency_penalty
            repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
            stride_logit_b i
        else
          sInit.readMem Logits
            (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i) := by
  intro i
  have h_readback :=
    scatter_readback_prop_masked_list_active_injective
      (region := (Logits : RegionName))
      (List.finRange BLOCK_P) sInit
      (fun k : Fin BLOCK_P =>
        activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b k)
      (fun k : Fin BLOCK_P =>
        penaltyValue s Logits presence_penalty freqency_penalty
          repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
          stride_logit_b k)
      (fun k : Fin BLOCK_P => active s p_cumsum_seq_len k)
      i (List.nodup_finRange BLOCK_P) (List.mem_finRange i)
      (by
        intro k _hk _hAct heq
        apply hUniq
        unfold activeStoreAddr at heq
        exact Nat.add_left_cancel heq)
  have hfun :
      (fun (acc : BlockState) (k : Fin BLOCK_P) =>
        if active s p_cumsum_seq_len k then
          acc.writeMem Logits
            (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b k)
            (penaltyValue s Logits presence_penalty freqency_penalty
              repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
              stride_logit_b k)
        else
          acc)
        =
      (fun (acc : BlockState) (k : Fin BLOCK_P) =>
        if (fun k : Fin BLOCK_P => active s p_cumsum_seq_len k) k then
          acc.writeMem Logits
            ((fun k : Fin BLOCK_P =>
              activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b k) k)
            ((fun k : Fin BLOCK_P =>
              penaltyValue s Logits presence_penalty freqency_penalty
                repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
                stride_logit_b k) k)
        else
          acc) := by
    funext (acc : BlockState) (k : Fin BLOCK_P)
    by_cases hAct : active s p_cumsum_seq_len k
    · simp [hAct, storeOffset_eq_active]
    · simp [hAct]
  rw [hfun]
  simpa using h_readback

theorem apply_penalty_masked_scatter_readback
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b BLOCK_P : Nat) (s : BlockState)
    (hUniq :
      Function.Injective
        (fun i : Fin BLOCK_P => tokenId s p_token_ids p_cumsum_seq_len i)) :
    ∀ i : Fin BLOCK_P,
      ((List.finRange BLOCK_P).foldl
          (fun (acc : BlockState) (k : Fin BLOCK_P) =>
            if active s p_cumsum_seq_len k then
              acc.writeMem Logits
                (storeOffset s p_token_ids p_cumsum_seq_len stride_logit_b k)
                (penaltyValue s Logits presence_penalty freqency_penalty
                  repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
                  stride_logit_b k)
            else
              acc)
          s).readMem Logits
          (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i) =
        if active s p_cumsum_seq_len i then
          penaltyValue s Logits presence_penalty freqency_penalty
            repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
            stride_logit_b i
        else
          s.readMem Logits
            (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i) := by
  exact apply_penalty_masked_scatter_readback_from Logits presence_penalty
    freqency_penalty repetition_penalty p_token_ids p_token_counts
    p_cumsum_seq_len stride_logit_b BLOCK_P s s hUniq

/-! ## apply_penalty correctness — status

A direct algorithm-layer cellwise correctness theorem against `penaltyValue`
under a uniqueness hypothesis (`tokenId` injective) is feasible in principle
but still requires bridging the executed kernel's data-dependent load/address
chain to the explicit foldl shape used by the readback theorem. The pieces in
place:

- `tokenId`, `storeOffset`, `activeStoreAddr`, `penaltyValue` (above): the
  full formula-level spec.
- `penaltyStoreValue` and `penaltyStoreValue_active_eq_penaltyValue` (above):
  the value expression shape exposed by kernel execution, plus its active-lane
  reduction to the formula-level spec.
- `storeOffset_eq_active` (above): bridges active-lane offset forms.
- `apply_penalty_masked_scatter_store_value_readback_from` (above): the
  execution-shaped scatter readback theorem, writing `penaltyStoreValue` from
  an arbitrary register-set pre-store state.
- `apply_penalty_masked_scatter_store_value_readback_tile_from` (above): the
  same readback shape over `TileShape.allIndices [BLOCK_P]`, matching the
  one-dimensional `tl.store` expansion directly.
- `apply_penalty_masked_scatter_readback_from` (above): the active-lane scatter
  readback theorem under `tokenId` injectivity, generalized to the register-set
  pre-store state produced by the executed kernel.

Future closure: state `apply_penalty_correct` against `penaltyValue` under
`hUniq : Function.Injective (tokenId s p_token_ids p_cumsum_seq_len)` and
discharge by rewriting the kernel execution to the store-value scatter readback
shape, then reducing active-lane `penaltyStoreValue` to `penaltyValue`. -/

end VeriTile.Bench.TritonBenchG.ApplyPenalty
