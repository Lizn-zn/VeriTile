import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Optimizer

/-!
# `apply_penalty` — strict per-kernel correctness

`_fwd_kernel_apply_penalty` applies repetition / frequency / presence penalties
to a row of logits in place. For program `cur_batch`, it gathers that batch's
token ids/counts from `[cur_batch_start, cur_batch_end)`, loads the corresponding
logits, applies `rep = logit/rep_pen if logit>0 else logit*rep_pen`, subtracts
`count·freq_pen` and `pres_pen`, and scatters the result back into `Logits` at
the gathered token positions (masked by the per-batch length).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_apply_penalty[(Logits.shape[0],)](...)`, the
grid size, the `BLOCK_P`/`num_warps` choices, and the `Logits.is_contiguous()`
assumption) is the *trusted boundary*, not a proof obligation here. The batch id
`cur_batch` (= `s.pids 0`) is universally quantified, so the per-program
statement covers every batch of the grid.

## Proof architecture

This is a **data-dependent gather-scatter** kernel: the store address itself
comes from a load (`batch_ids = tl.load(p_token_ids + ...)`). Cellwise
correctness holds under distinct active token ids (`hUniq`, i.e. no duplicate
token within a batch's window), which rules out write-write aliasing of the
in-place `Logits` update.

```
apply_penalty_output_summary                  ← TOP THEOREM
  ├─ apply_penalty_toAlgorithm_supported          surface lowers to algorithm layer
  └─ apply_penalty_compute_correct                ← ComputeCorrect over the masked store
       └─ apply_penalty_correct                    ← algorithm-layer readback (discharges
            │                                         apply_penalty_correct_target)
            ├─ foldl_step_readMem_congr             base-memory + per-lane step congruence
            ├─ apply_penalty_masked_scatter_store_value_readback_tile_from'
            │     └─ scatter_readback_prop_masked_list_active_injective
            └─ penaltyStoreValue_active_eq_penaltyValue   exec value → clean penaltyValue
```

The spec is the Lion-style penalty `penaltyValue`
(`rep = logit/rep_pen if logit>0 else logit*rep_pen`, then `− count·freq_pen
− pres_pen`); `penaltyStoreValue` is the raw exec-shaped value, bridged to
`penaltyValue` on active lanes. `setReg` preserves memory, so the executed
register-set base and `s` read back identically (`foldl_step_readMem_congr`).

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The masked load uses `other=0`/`other=0.0` for out-of-range token slots.
Because the store address is data-dependent (gathered token ids), the masked
scatter readback weakens global offset injectivity to **injectivity over the
active (masked-true) lanes only** (`hUniq : ∀ i j, active i → active j →
tokenId i = tokenId j → i = j`, i.e. distinct tokens *within the batch window*).
This is satisfiable for normal padded blocks (inactive/out-of-window lanes may
hold default or duplicated token ids and are simply not constrained); the
correctness claim is correspondingly restricted to the active lanes. The
integer-to-float promotion in `batch_ids_count * cur_freqency` and the
`tl.where(logit > 0, …)` branch are modeled directly. The specs reference the
`ℝ` ordered-field operations directly, not `VeriTile.Triton.Math.*`.
-/

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

/-- Per-lane `Logits` output spec: the reusable Lion penalty oracle
(`VeriTile.Triton.Math.Optimizer.lionPenalty`) applied to the values this lane
loads — the logit at the gathered token, its count, and the three penalties. -/
noncomputable def penaltyValue
    (s : BlockState)
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b : Nat) (i : Fin BLOCK_P) : ℝ :=
  TiledOptimizer.lionPenalty
    (s.readMem Logits
      (s.pids 0 * stride_logit_b + tokenId s p_token_ids p_cumsum_seq_len i))
    (s.readMemValue .nat p_token_counts (tokenOffset s p_cumsum_seq_len i) : ℝ)
    (s.readMem repetition_penalty (s.pids 0))
    (s.readMem freqency_penalty (s.pids 0))
    (s.readMem presence_penalty (s.pids 0))

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
  simp [penaltyStoreValue, penaltyValue, TiledOptimizer.lionPenalty, hAct,
    storeOffset_eq_active, activeStoreAddr]
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

/-- Formula-level correctness target for the Python apply-penalty store: every
**active** token position holds `penaltyValue`. Discharged by
`apply_penalty_correct`. Only active lanes are claimed — inactive (out-of-window)
lanes neither store nor have a well-defined gathered address, and the kernel /
`WriteMap.writeIf` only assert the active subset. -/
def apply_penalty_correct_target
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b BLOCK_P : Nat) (s s' : BlockState) : Prop :=
  ∀ i : Fin BLOCK_P, active s p_cumsum_seq_len i →
    s'.readMem Logits
        (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i) =
      penaltyValue s Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b i

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

theorem apply_penalty_masked_scatter_store_value_readback_tile_from
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b BLOCK_P : Nat) (sInit s : BlockState)
    (hUniq : ∀ i j : Fin BLOCK_P,
      active s p_cumsum_seq_len i → active s p_cumsum_seq_len j →
      tokenId s p_token_ids p_cumsum_seq_len i =
        tokenId s p_token_ids p_cumsum_seq_len j → i = j) :
    ∀ idx : TileIndex [BLOCK_P], active s p_cumsum_seq_len idx.1 →
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
          penaltyStoreValue s Logits presence_penalty freqency_penalty
            repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
            stride_logit_b idx.1 := by
  intro idx hActIdx
  have h_readback :=
    BlockState.scatter_readback_prop_masked_list_of_true
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
      hActIdx
      (by
        intro k _hk hAct heq
        have hFin : k.1 = idx.1 := by
          apply hUniq k.1 idx.1 hAct hActIdx
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
    (hUniq : ∀ i j : Fin BLOCK_P,
      active s p_cumsum_seq_len i → active s p_cumsum_seq_len j →
      tokenId s p_token_ids p_cumsum_seq_len i =
        tokenId s p_token_ids p_cumsum_seq_len j → i = j) :
    ∀ idx : TileIndex [BLOCK_P], active s p_cumsum_seq_len idx.1 →
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
          penaltyStoreValue s Logits presence_penalty freqency_penalty
            repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
            stride_logit_b idx.1 := by
  exact apply_penalty_masked_scatter_store_value_readback_tile_from
    Logits presence_penalty freqency_penalty repetition_penalty p_token_ids
    p_token_counts p_cumsum_seq_len stride_logit_b BLOCK_P sInit s hUniq

theorem apply_penalty_correct
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) (s s' : BlockState)
    (hUniq : ∀ i j : Fin BLOCK_P,
      active s p_cumsum_seq_len i → active s p_cumsum_seq_len j →
      tokenId s p_token_ids p_cumsum_seq_len i =
        tokenId s p_token_ids p_cumsum_seq_len j → i = j)
    (hExec : exec (apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P) s = some s') :
    apply_penalty_correct_target Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b BLOCK_P s s' := by
  unfold apply_penalty_correct_target
  intro i hActive
  simp [exec, apply_penalty, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, ComparableDType.gt] at hExec
  subst s'
  -- (2) Target's `penaltyValue` → exec-shaped `penaltyStoreValue` (active lane).
  rw [show penaltyValue s Logits presence_penalty freqency_penalty
            repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
            stride_logit_b i
        = penaltyStoreValue s Logits presence_penalty freqency_penalty
            repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
            stride_logit_b i
        from (penaltyStoreValue_active_eq_penaltyValue _ _ _ _ _ _ _ _ _ _
          hActive).symm]
  -- (3) Bridge to the exec-shaped tile readback lemma (RHS matches up to defeq).
  refine Eq.trans ?bridge
    (apply_penalty_masked_scatter_store_value_readback_tile_from'
      (sInit := s)
      Logits presence_penalty freqency_penalty repetition_penalty
      p_token_ids p_token_counts p_cumsum_seq_len stride_logit_b BLOCK_P
      s hUniq (i, PUnit.unit) hActive)
  -- (4) The executed foldl (over the register-set base `execBase`) and the
  -- spec foldl (over `s`) read back equally: base memories agree (`setReg`
  -- preserves memory) and the per-lane offset/value agree on active lanes.
  apply BlockState.foldl_step_readMem_congr
  · intro r oo; simp
  · -- per-lane step equality on active lanes
    intro k _hk hAct
    have h00 : (0.0 : ℝ) = (0 : ℝ) := by norm_num
    refine ⟨?_, ?_⟩
    · -- offset: executed `storeOffset` equals `storeOffset` (active form).
      simp [active, storeOffset, tokenId, tokenOffset, batchStart, batchEnd, hAct]
    · -- value: executed raw value equals `penaltyStoreValue`; bridge the latter
      -- to the clean `penaltyValue`, then normalise the executed encoding.
      rw [penaltyStoreValue_active_eq_penaltyValue _ _ _ _ _ _ _ _ _ _ hAct]
      by_cases hPos : (0 : ℝ) < s.readMem Logits
          (s.pids 0 * stride_logit_b +
            s.readMemValue .nat p_token_ids.cast
              (s.readMemValue .nat p_cumsum_seq_len.cast (s.pids 0) + k.1.val)) <;>
        · simp only [penaltyValue, TiledOptimizer.lionPenalty, storeOffset,
            tokenId, tokenOffset, batchStart, batchEnd, NumericDType.div,
            NumericDType.mul, WithBot.realDiv, WithBot.realMul, Option.map₂,
            Option.map, Option.bind, Option.bind_some, Function.comp, hAct,
            if_true, if_false, WithBot.coe_lt_coe, WithBot.unbotD_coe,
            WithBot.some_eq_coe, hPos, h00]

/-- **Compute-facing `ComputeCorrect` for `apply_penalty`.** Under distinct
active token ids, the masked in-place store to `Logits` realizes the Lion-style
penalty value `penaltyValue` at every active token position. -/
theorem apply_penalty_compute_correct
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) (s : BlockState)
    (hUniq : ∀ i j : Fin BLOCK_P,
      active s p_cumsum_seq_len i → active s p_cumsum_seq_len j →
      tokenId s p_token_ids p_cumsum_seq_len i =
        tokenId s p_token_ids p_cumsum_seq_len j → i = j) :
    ComputeCorrect.Realizes
      (kernel := apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_P => active s p_cumsum_seq_len i)
          (fun i => ((Logits : RegionName),
            activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i)))
      (expected := fun i => penaltyValue s Logits presence_penalty
        freqency_penalty repetition_penalty p_token_ids p_token_counts
        p_cumsum_seq_len stride_logit_b i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [apply_penalty, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := apply_penalty_correct Logits presence_penalty freqency_penalty
    repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
    stride_logit_b stride_logit_s BLOCK_P s s' hUniq hExec i hActive
  simpa [ComputeCorrect.OutputReadable.read] using h

/-- **Per-kernel output summary for `apply_penalty`.** The DSL surface lowers to
the algorithm layer, and (under distinct active token ids) the masked in-place
`Logits` store is compute-correct against the Lion penalty value. -/
theorem apply_penalty_output_summary
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) (s : BlockState)
    (hUniq : ∀ i j : Fin BLOCK_P,
      active s p_cumsum_seq_len i → active s p_cumsum_seq_len j →
      tokenId s p_token_ids p_cumsum_seq_len i =
        tokenId s p_token_ids p_cumsum_seq_len j → i = j) :
    (∃ alg, (apply_penalty Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
          (fun i : Fin BLOCK_P => active s p_cumsum_seq_len i)
          (fun i => ((Logits : RegionName),
            activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i)))
      (expected := fun i => penaltyValue s Logits presence_penalty
        freqency_penalty repetition_penalty p_token_ids p_token_counts
        p_cumsum_seq_len stride_logit_b i) := by
  refine ⟨?_, ?_⟩
  · exact apply_penalty_toAlgorithm_supported Logits presence_penalty
      freqency_penalty repetition_penalty p_token_ids p_token_counts
      p_cumsum_seq_len stride_logit_b stride_logit_s BLOCK_P
  · exact apply_penalty_compute_correct Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P s hUniq

/-! ## apply_penalty correctness — closed

Cellwise correctness is fully discharged: `apply_penalty_correct` proves
`apply_penalty_correct_target` (every active token position of `Logits` holds
`penaltyValue`) under the *active-lane* uniqueness premise
`hUniq : ∀ i j, active i → active j → tokenId i = tokenId j → i = j` — satisfiable
for padded blocks, unlike global injectivity. `apply_penalty_compute_correct` /
`apply_penalty_output_summary` lift it to the `ComputeCorrect.Realizes` /
output-summary surface. The supporting readback and value-bridge lemmas above
(`*_scatter_readback*`, `penaltyStoreValue_active_eq_penaltyValue`,
`foldl_step_readMem_congr`) are the reusable pieces of that proof. -/

end VeriTile.Bench.TritonBenchG.ApplyPenalty
