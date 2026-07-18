import VeriTile.Triton

/-!
# `apply_penalty` — strict per-kernel correctness

`_fwd_kernel_apply_penalty` applies repetition / frequency / presence penalties
to a row of logits in place. For program `cur_batch`, it loads the three
per-batch penalty scalars, gathers that batch's token ids/counts from
`[cur_batch_start, cur_batch_end)` (the cumsum window), loads the
corresponding logits, applies `rep = logit/rep_pen if logit>0 else
logit*rep_pen`, subtracts `count·freq_pen` and `pres_pen`, and scatters the
result back into `Logits` at the gathered token positions (masked by the
per-batch length).

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_apply_penalty[(Logits.shape[0],)](...)`, the
grid size, the `BLOCK_P`/`num_warps` choices, and the `Logits.is_contiguous()`
assumption) is the *trusted boundary*, not a proof obligation here. The batch id
`cur_batch` (= `s.pids 0`) is universally quantified, so the per-program
statement covers every batch of the grid.

## Proof architecture

```
apply_penalty_correctness              ← TOP SPECIFICATION (applyPenaltyIO ⊨ penaltyValuePure)
  ├─ apply_penalty_flattenOk           bridge fragment membership
  ├─ apply_penalty_traceSafe           safety walk: slot cells + gather/scatter lanes
  └─ apply_penalty_region_run          region-model gather–scatter masked triple (hrun)
       ├─ apply_penalty_exec_isSome    termination
       ├─ apply_penalty_correct        ← algorithm-layer readback (discharges
       │    │                             apply_penalty_correct_target)
       │    ├─ foldl_step_readMem_congr  base-memory + per-lane step congruence
       │    ├─ apply_penalty_masked_scatter_store_value_readback_tile_from'
       │    │     └─ BlockState.scatter_readback_prop_masked_list_of_true
       │    └─ penaltyStoreValue_active_eq_penaltyValue
       ├─ penaltyValue_eq_pure         active-lane congruence: state reads → pinned values
       └─ apply_penalty_frame          masked scatter frame
```

The headline is stated on the kernel's **penalty gather–scatter IO
signature** `applyPenaltyIO` (`MetaScatterMasked2DKernelIO₁`, the metadata
genre's `Meta` slots + `Scatter` writes skin): three per-program float
scalar slots `presence/freqency/repetition_penalty[pid₀]` (`g₁`/`g₂`/`g₃`),
the two `.nat` cumsum slots `p_cumsum_seq_len[pid₀]`/`[pid₀+1]`
(`m₁`/`m₂` — the batch window bounds), the `.nat` index tile
`p_token_ids[m₁+j]` (`ids` — the gathered token ids), the `.nat` payload
tile `p_token_counts[m₁+j]` (`cnts`), and the float data channel `Logits`,
**gather-read and scatter-written in place** (`out := inp`) at
`pid₀ * stride_logit_b + ids j`, masked by the batch window `m₁ + j < m₂`.
`⊨` (`MetaScatterMasked2DKernelIO₁.Implements`) pins all loaded values by
memory preconditions (the ghost-variable discipline): for every disjoint
flat placement of the buffers, every program id whose slot cells and active
lanes are in bounds, and every launch state whose slots/windows hold the
pinned values, the translated pointer kernel terminates, every active
scatter lane of `Logits` holds the pure Lion penalty
`penaltyValuePure g₁ g₂ g₃ cnts xs j`, and every other memory cell is
unchanged.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The masked loads use
`other=0`/`other=0.0` for out-of-range token slots. Because the store address
is data-dependent (gathered token ids), the scatter readback leg of `⊨` is
guarded by the skin's per-pinned-context `WriteInj` antecedent — injectivity
of `pid₀ * stride_logit_b + ids j` over the **active (masked-true) lanes
only** (i.e. distinct tokens *within the batch window*; the same `hUniq`
write-map-injectivity side condition the pre-`⊨` summary carried as a
hypothesis, now quantified over the pinned values inside `⊨` itself). This
is satisfiable for normal padded blocks (inactive/out-of-window lanes may
hold default or duplicated token ids and are simply not constrained); the
correctness claim is correspondingly restricted to the active lanes. The
integer-to-float promotion in `batch_ids_count * cur_freqency` and the
`tl.where(logit > 0, …)` branch are modeled directly. The penalty spec
`penaltyValuePure` goes through `VeriTile.Triton.Math.Optimizer`'s Lion
oracle (`TiledOptimizer.lionPenalty`); the surrounding arithmetic is the `ℝ`
ordered-field operations directly.
-/

namespace VeriTile.Bench.TritonBenchG.ApplyPenalty

open VeriTile.Triton
open scoped VeriTile.Triton.MetaScatterMasked2DKernelIO₁

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
      (activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i))
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
            activeStoreAddr, tokenId, tokenOffset, batchStart, batchEnd, NumericDType.div,
            NumericDType.mul, WithBot.realDiv, WithBot.realMul, Option.map₂,
            Option.map, Option.bind, Option.bind_some, Function.comp, hAct,
            if_true, if_false, WithBot.coe_lt_coe, WithBot.unbotD_coe,
            WithBot.some_eq_coe, hPos, h00]

/-- Per-lane `Logits` output spec as a **pure** function of the pinned values
of the `⊨` headline: the reusable Lion penalty oracle
(`VeriTile.Triton.Math.Optimizer.lionPenalty`) applied to the gathered logit
`xs j`, its count `cnts j`, and the three loaded penalty scalars —
repetition `g₃`, frequency `g₂`, presence `g₁`. This is `penaltyValue` with
the state-coupled reads replaced by the named binders of the headline. -/
noncomputable def penaltyValuePure (g₁ g₂ g₃ : ℝ)
    (cnts : Fin BLOCK_P → Nat) (xs : Fin BLOCK_P → ℝ) (j : Fin BLOCK_P) : ℝ :=
  TiledOptimizer.lionPenalty (xs j) (cnts j : ℝ) g₃ g₂ g₁

/-- Active-lane congruence: once the loaded scalars and tiles are pinned, the
state-coupled `penaltyValue` equals the pure `penaltyValuePure` — every
memory read in `penaltyValue` is one of the pinned values. -/
theorem penaltyValue_eq_pure
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b : Nat) (s : BlockState)
    (g₁ g₂ g₃ : ℝ) (m₁ m₂ : Nat) (ids cnts : Fin BLOCK_P → Nat)
    (xs : Fin BLOCK_P → ℝ)
    (hg₁ : s.readMem presence_penalty (s.pids 0) = g₁)
    (hg₂ : s.readMem freqency_penalty (s.pids 0) = g₂)
    (hg₃ : s.readMem repetition_penalty (s.pids 0) = g₃)
    (hm₁ : s.readMemValue .nat p_cumsum_seq_len (s.pids 0) = m₁)
    (hid : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s.readMemValue .nat p_token_ids (m₁ + j.val) = ids j)
    (hcnt : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s.readMemValue .nat p_token_counts (m₁ + j.val) = cnts j)
    (hx : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s.readMem Logits (s.pids 0 * stride_logit_b + ids j) = xs j)
    (i : Fin BLOCK_P) (hi : m₁ + i.val < m₂) :
    penaltyValue s Logits presence_penalty freqency_penalty repetition_penalty
        p_token_ids p_token_counts p_cumsum_seq_len stride_logit_b i
      = penaltyValuePure g₁ g₂ g₃ cnts xs i := by
  have hoff : tokenOffset s p_cumsum_seq_len i = m₁ + i.val := by
    unfold tokenOffset batchStart
    rw [hm₁]
  have htok : tokenId s p_token_ids p_cumsum_seq_len i = ids i := by
    unfold tokenId
    rw [hoff]
    exact hid i hi
  have haddr : activeStoreAddr s p_token_ids p_cumsum_seq_len stride_logit_b i
      = s.pids 0 * stride_logit_b + ids i := by
    unfold activeStoreAddr
    rw [htok]
  unfold penaltyValue penaltyValuePure
  rw [haddr, hx i hi, hoff, hcnt i hi, hg₁, hg₂, hg₃]

/-- A masked scatter-store `foldl` leaves every memory cell it does not
actively hit unchanged (cell-level frame for the masked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, P k → ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s).mem r o = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hP : P hd
      · rw [if_pos hP,
          ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
          BlockState.writeMem_mem]
        exact if_neg (fun hc =>
          hnot hd List.mem_cons_self hP ⟨hc.1.symm, hc.2.symm⟩)
      · rw [if_neg hP]
        exact ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk))

/-- Termination: the kernel executes to completion from any state
(straight-line masked gather/scatter body, so no `0 < BLOCK_P` side
condition is needed). -/
private theorem apply_penalty_exec_isSome
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) (s : BlockState) :
    ∃ s1, exec ((apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P).toAlgKernel) s = some s1 := by
  simp [exec, apply_penalty, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, ComparableDType.gt]

/-- Frame half: every memory cell not actively written by the masked scatter
store — every cell of every region other than `Logits`, and the cells of
`Logits` other than the active lanes' gathered token addresses — is
preserved by the run. -/
private theorem apply_penalty_frame
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat)
    (s s1 : BlockState)
    (hExec : exec ((apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin BLOCK_P,
      s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + i.val
          < s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) →
      ¬((Logits : RegionName) = r ∧
        s.pids 0 * stride_logit_b +
          s.readMemValue .nat p_token_ids
            (s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + i.val) = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, apply_penalty, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.select,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        ComparableDType.lt, ComparableDType.gt] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ _ r o _ _ ?_) rfl
  intro k _ hmk hc
  rw [if_pos hmk] at hc
  exact hmiss k.1 hmk hc

/-- **The region-model gather–scatter masked Hoare triple** — termination,
active-lane scatter values (guarded by the write-map injectivity of the
pinned index tile, the skin's `WriteInj` antecedent), and frame off the
active scatter lanes, from any launch state whose slots and windows hold
the pinned values. This is the `hrun` obligation of the `⊨` headline; the
value half reuses `apply_penalty_correct` (fed the state-coupled `hUniq` it
derives from the `WriteInj` antecedent) and transports its state-coupled
spec to the pure one via `penaltyValue_eq_pure`. -/
theorem apply_penalty_region_run
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat)
    (s₀ : BlockState) (g₁ g₂ g₃ : ℝ) (m₁ m₂ : Nat)
    (ids cnts : Fin BLOCK_P → Nat) (xs : Fin BLOCK_P → ℝ)
    (hg₁ : s₀.readMem presence_penalty (s₀.pids 0) = g₁)
    (hg₂ : s₀.readMem freqency_penalty (s₀.pids 0) = g₂)
    (hg₃ : s₀.readMem repetition_penalty (s₀.pids 0) = g₃)
    (hm₁ : s₀.readMemValue .nat p_cumsum_seq_len (s₀.pids 0) = m₁)
    (hm₂ : s₀.readMemValue .nat p_cumsum_seq_len (s₀.pids 0 + 1) = m₂)
    (hid : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s₀.readMemValue .nat p_token_ids (m₁ + j.val) = ids j)
    (hcnt : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s₀.readMemValue .nat p_token_counts (m₁ + j.val) = cnts j)
    (hx : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
      s₀.readMem Logits (s₀.pids 0 * stride_logit_b + ids j) = xs j) :
    ∃ s1, exec ((apply_penalty Logits presence_penalty freqency_penalty
          repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
          stride_logit_b stride_logit_s BLOCK_P).toAlgKernel) s₀ = some s1
      ∧ ((∀ j k : Fin BLOCK_P, m₁ + j.val < m₂ → m₁ + k.val < m₂ →
            s₀.pids 0 * stride_logit_b + ids j
              = s₀.pids 0 * stride_logit_b + ids k → j = k) →
          ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
            s1.readMem Logits (s₀.pids 0 * stride_logit_b + ids j)
              = penaltyValuePure g₁ g₂ g₃ cnts xs j)
      ∧ (∀ r o,
          (r ≠ (Logits : RegionName) ∨ ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
            o ≠ s₀.pids 0 * stride_logit_b + ids j) →
          s1.mem r o = s₀.mem r o) := by
  obtain ⟨s1, hs1⟩ := apply_penalty_exec_isSome Logits presence_penalty
    freqency_penalty repetition_penalty p_token_ids p_token_counts
    p_cumsum_seq_len stride_logit_b stride_logit_s BLOCK_P s₀
  -- restate the pins in the file's own vocabulary (definitional)
  have hstart : batchStart s₀ p_cumsum_seq_len = m₁ := hm₁
  have hend : batchEnd s₀ p_cumsum_seq_len = m₂ := hm₂
  have hact : ∀ i : Fin BLOCK_P,
      active s₀ p_cumsum_seq_len i ↔ m₁ + i.val < m₂ := by
    intro i
    unfold active tokenOffset
    rw [hstart, hend]
  have htok : ∀ i : Fin BLOCK_P, m₁ + i.val < m₂ →
      tokenId s₀ p_token_ids p_cumsum_seq_len i = ids i := by
    intro i hi
    unfold tokenId tokenOffset
    rw [hstart]
    exact hid i hi
  have haddr : ∀ i : Fin BLOCK_P, m₁ + i.val < m₂ →
      activeStoreAddr s₀ p_token_ids p_cumsum_seq_len stride_logit_b i
        = s₀.pids 0 * stride_logit_b + ids i := by
    intro i hi
    unfold activeStoreAddr
    rw [htok i hi]
  refine ⟨s1, hs1, fun hinj j hj => ?_, fun r o hcond => ?_⟩
  · -- WriteInj over the pinned ids → state-coupled distinct active token ids
    have hUniq : ∀ i k : Fin BLOCK_P,
        active s₀ p_cumsum_seq_len i → active s₀ p_cumsum_seq_len k →
        tokenId s₀ p_token_ids p_cumsum_seq_len i =
          tokenId s₀ p_token_ids p_cumsum_seq_len k → i = k := by
      intro i k hi hk hik
      have hi' := (hact i).mp hi
      have hk' := (hact k).mp hk
      refine hinj i k hi' hk' ?_
      rw [htok i hi', htok k hk'] at hik
      rw [hik]
    have h := apply_penalty_correct Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P s₀ s1 hUniq hs1 j
      ((hact j).mpr hj)
    rw [haddr j hj] at h
    rw [h]
    exact penaltyValue_eq_pure Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b s₀ g₁ g₂ g₃ m₁ m₂ ids cnts xs hg₁ hg₂ hg₃ hm₁ hid hcnt
      hx j hj
  · refine apply_penalty_frame Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P s₀ s1 hs1 r o
      (fun i hi hc => ?_)
    rw [hm₁, hm₂] at hi
    rw [hm₁, hid i hi] at hc
    rcases hcond with hne | hno
    · exact hne hc.1.symm
    · exact hno i hi hc.2.symm

set_option maxRecDepth 8000 in
/-- Per-execution safety walk: one computational unfold walks all fifteen
statements — seven are memory-silent (`program_id`, `arange`, the offset /
pointer arithmetic and the `tl.where` penalty arithmetic) — and reduces the
memory accesses to the skin-shaped hypotheses: the five **unmasked
scalar-slot loads** (`presence/freqency/repetition_penalty[pid₀]` and the
two cumsum cells `p_cumsum_seq_len[pid₀]`/`[pid₀+1]`) need their slot cells
in bounds, the masked `p_token_ids`/`p_token_counts` loads need the batch
window `start + j` in bounds at the active lanes `start + j < end`, and the
masked `Logits` gather load / scatter store need the data-dependent address
`pid₀ * stride_logit_b + p_token_ids[start + j]` in bounds at the active
lanes, where `start`/`end` are the values sitting in the cumsum cells
(kernel-level spelling: `s.readMemValue .nat …`). -/
theorem apply_penalty_traceSafe
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat)
    (bounds : RegionBounds) (s : BlockState)
    (hb₁ : s.pids 0 < bounds presence_penalty)
    (hb₂ : s.pids 0 < bounds freqency_penalty)
    (hb₃ : s.pids 0 < bounds repetition_penalty)
    (hb₄ : s.pids 0 < bounds p_cumsum_seq_len)
    (hb₅ : s.pids 0 + 1 < bounds p_cumsum_seq_len)
    (hbi : ∀ j : Fin BLOCK_P,
      s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + j.val
          < s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) →
      s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + j.val
        < bounds p_token_ids)
    (hbc : ∀ j : Fin BLOCK_P,
      s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + j.val
          < s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) →
      s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + j.val
        < bounds p_token_counts)
    (hbl : ∀ j : Fin BLOCK_P,
      s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + j.val
          < s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) →
      s.pids 0 * stride_logit_b +
        s.readMemValue .nat p_token_ids
          (s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + j.val)
        < bounds Logits) :
    Kernel.TraceSafe bounds
      ((apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [apply_penalty, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, stepStmts, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg, tile_elementwise, Bool.and_eq_true,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.select,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt, ComparableDType.gt,
    BlockState.readMemValue, hb₁, hb₂, hb₃, hb₄, hb₅]
  refine ⟨fun a ha => hbi a ha, fun a ha => hbc a ha, fun a ha => ?_⟩
  -- restate the residual gather/scatter obligation in the kernel-level
  -- spelling (definitional), so `rw` sees the masked-address conditional
  have ha' : s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + a.val
      < s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) := ha
  show (s.pids 0 * stride_logit_b +
      if s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + a.val
          < s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) then
        s.readMemValue .nat p_token_ids
          (s.readMemValue .nat p_cumsum_seq_len (s.pids 0) + a.val)
      else 0) < bounds Logits
  rw [if_pos ha']
  exact hbl a ha'

/-- The kernel sits inside the flat-memory bridge's covered fragment (scalar
float / `.nat` slot loads, pointer arithmetic, masked loads with `other`,
`tl.where`, and the masked data-dependent scatter store). -/
theorem apply_penalty_flattenOk
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    ((apply_penalty Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [apply_penalty, ComputeKernel.toAlgKernel, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

/-- `apply_penalty`'s **penalty gather–scatter IO signature** — the whole
kernel-specific audit surface of the `⊨` headline
(`MetaScatterMasked2DKernelIO₁`, the metadata genre's `Meta` slots +
`Scatter` writes skin):

* `fbuf1`/`fbuf2`/`fbuf3` — the three per-batch float penalty slots
  (`presence_penalty`/`freqency_penalty`/`repetition_penalty`), all read at
  cell `pid₀` (`fwin1 = fwin2 = fwin3 = pid₀`), yielding the named scalars
  `g₁ = cur_presence`, `g₂ = cur_freqency`, `g₃ = cur_repetition`;
* `mbuf = p_cumsum_seq_len` — the `.nat` metadata buffer carrying both
  cumsum slots: `mwin1 = pid₀` and `mwin2 = pid₀ + 1`, yielding
  `m₁ = cur_batch_start_index` and `m₂ = cur_batch_end_index`;
* `idbuf = p_token_ids`, `cntbuf = p_token_counts` — the `.nat` tiles, lane
  `j` at the batch-window cell `m₁ + j` (`readi = readc`), yielding the
  index tile `ids` (= `batch_ids`) and the counts `cnts`;
* `inp = out = Logits` — the float data channel is gather-read **and**
  scatter-written **in place** (duplicate-region wiring), lane `j` at the
  data-dependent address `pid₀ * stride_logit_b + ids j` (`read = write`);
  `B = BLOCK_P`;
* `mask` — the active lanes `m₁ + j < m₂`
  (`cur_batch_id_offset < cur_batch_end_index`), shared by all masked
  accesses (`writeMask` defaults to `mask`).

`stride_logit_s` is unused by the kernel body (rows are contiguous), and
the 1-D launch ignores the family's second program id. The slot cells,
windows, and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual slot loads, addressing, and masking match
them. Buffer sizes are not signature content: the headline quantifies over
every allocation whose extents cover the slot cells and the active lanes. -/
def applyPenaltyIO
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    MetaScatterMasked2DKernelIO₁ where
  kernel := apply_penalty Logits presence_penalty freqency_penalty
    repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
    stride_logit_b stride_logit_s BLOCK_P
  fbuf1 := presence_penalty
  fbuf2 := freqency_penalty
  fbuf3 := repetition_penalty
  mbuf := p_cumsum_seq_len
  idbuf := p_token_ids
  cntbuf := p_token_counts
  inp := Logits
  out := Logits
  B := BLOCK_P
  fwin1 := fun pid₀ _ => pid₀
  fwin2 := fun pid₀ _ => pid₀
  fwin3 := fun pid₀ _ => pid₀
  mwin1 := fun pid₀ _ => pid₀
  mwin2 := fun pid₀ _ => pid₀ + 1
  readi := fun _ _ m₁ _ j => m₁ + j.val
  readc := fun _ _ m₁ _ j => m₁ + j.val
  read := fun pid₀ _ _ _ ids j => pid₀ * stride_logit_b + ids j
  mask := fun _ _ m₁ m₂ j => m₁ + j.val < m₂
  write := fun pid₀ _ _ _ ids j => pid₀ * stride_logit_b + ids j

/-- **The headline**: `apply_penalty` implements the pure Lion penalty on
its penalty gather–scatter IO signature — for every disjoint flat placement
of the buffers, every program `cur_batch = pid₀` whose slot cells and
active lanes are in bounds, and every launch state whose penalty slots hold
`g₁`/`g₂`/`g₃`, whose cumsum cells hold `m₁`/`m₂`, whose token-id/count
windows hold `ids`/`cnts`, and whose gathered `Logits` window holds `xs`,
the translated pointer kernel terminates, every active (`m₁ + j < m₂`)
scatter lane `pid₀ * stride_logit_b + ids j` of `Logits` holds
`penaltyValuePure g₁ g₂ g₃ cnts xs j` — **guarded by the skin's `WriteInj`
antecedent** (distinct active token ids over the pinned values: the
write-map injectivity the pre-`⊨` summary carried as its `hUniq`
hypothesis) — and every other memory cell is unchanged. Proof:
`MetaScatterMasked2DKernelIO₁.Implements.intro` assembles the region-model
triple with the flat-memory bridge side conditions. -/
specification apply_penalty_correctness
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    applyPenaltyIO Logits presence_penalty freqency_penalty
        repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
        stride_logit_b stride_logit_s BLOCK_P
      ⊨ fun _ _ g₁ g₂ g₃ _ _ _ cnts xs j =>
          penaltyValuePure g₁ g₂ g₃ cnts xs j := by
  refine MetaScatterMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact apply_penalty_flattenOk Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
  · intro bounds s m₁ m₂ ids hm₁ hm₂ hids hbf₁ hbf₂ hbf₃ hbm₁ hbm₂ hbi hbc
      hbr _hbw
    -- restate the skin-projection facts in the kernel-level spelling
    -- (definitional), so `rw` sees the walk lemma's patterns
    have hm₁' : s.readMemValue .nat p_cumsum_seq_len (s.pids 0) = m₁ := hm₁
    have hm₂' : s.readMemValue .nat p_cumsum_seq_len (s.pids 0 + 1) = m₂ := hm₂
    have hids' : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
        s.readMemValue .nat p_token_ids (m₁ + j.val) = ids j := hids
    have hbf₁' : s.pids 0 < bounds presence_penalty := hbf₁
    have hbf₂' : s.pids 0 < bounds freqency_penalty := hbf₂
    have hbf₃' : s.pids 0 < bounds repetition_penalty := hbf₃
    have hbm₁' : s.pids 0 < bounds p_cumsum_seq_len := hbm₁
    have hbm₂' : s.pids 0 + 1 < bounds p_cumsum_seq_len := hbm₂
    have hbi' : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
        m₁ + j.val < bounds p_token_ids := hbi
    have hbc' : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
        m₁ + j.val < bounds p_token_counts := hbc
    have hbr' : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
        s.pids 0 * stride_logit_b + ids j < bounds Logits := hbr
    refine apply_penalty_traceSafe Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P bounds s hbf₁' hbf₂' hbf₃'
      hbm₁' hbm₂' (fun j hj => ?_) (fun j hj => ?_) (fun j hj => ?_)
    · rw [hm₁', hm₂'] at hj
      rw [hm₁']
      exact hbi' j hj
    · rw [hm₁', hm₂'] at hj
      rw [hm₁']
      exact hbc' j hj
    · rw [hm₁', hm₂'] at hj
      rw [hm₁', hids' j hj]
      exact hbr' j hj
  · intro s₀ g₁ g₂ g₃ m₁ m₂ ids cnts xs hg₁ hg₂ hg₃ hm₁ hm₂ hid hcnt hx
    have hg₁' : s₀.readMem presence_penalty (s₀.pids 0) = g₁ := hg₁
    have hg₂' : s₀.readMem freqency_penalty (s₀.pids 0) = g₂ := hg₂
    have hg₃' : s₀.readMem repetition_penalty (s₀.pids 0) = g₃ := hg₃
    have hm₁' : s₀.readMemValue .nat p_cumsum_seq_len (s₀.pids 0) = m₁ := hm₁
    have hm₂' : s₀.readMemValue .nat p_cumsum_seq_len (s₀.pids 0 + 1) = m₂ :=
      hm₂
    have hid' : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
        s₀.readMemValue .nat p_token_ids (m₁ + j.val) = ids j := hid
    have hcnt' : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
        s₀.readMemValue .nat p_token_counts (m₁ + j.val) = cnts j := hcnt
    have hx' : ∀ j : Fin BLOCK_P, m₁ + j.val < m₂ →
        s₀.readMem Logits (s₀.pids 0 * stride_logit_b + ids j) = xs j := hx
    exact apply_penalty_region_run Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P s₀ g₁ g₂ g₃ m₁ m₂ ids cnts xs
      hg₁' hg₂' hg₃' hm₁' hm₂' hid' hcnt' hx'

/-! ## apply_penalty correctness — closed

The `⊨` headline `apply_penalty_correctness` states the whole story on the
kernel's IO signature `applyPenaltyIO`: termination, active-lane scatter
values (`penaltyValuePure`, guarded by the skin's `WriteInj`
distinct-active-token-ids antecedent — the old summary's `hUniq` side
condition, now quantified over the pinned values inside `⊨` itself), and
the frame, over every disjoint flat placement and every launch state
matching the pins. `apply_penalty_correct` remains the algorithm-layer
readback engine underneath (`apply_penalty_correct_target` in the
state-coupled vocabulary), bridged to the pinned-value spec by
`penaltyValue_eq_pure` inside `apply_penalty_region_run`. -/

end VeriTile.Bench.TritonBenchG.ApplyPenalty
