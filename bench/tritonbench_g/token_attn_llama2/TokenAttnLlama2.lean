import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `token_attn_llama2` — strict per-kernel correctness

`_fwd_kernel_token_att1` is the Llama2 token-decode QK-score stage. Each program
`(cur_batch, cur_head, start_n)` loads the single query vector `q`, gathers a
`BLOCK_N` block of cached key tokens through `B_Loc` (offset by
`cur_batch_start_index = max_input_len - cur_batch_seq_len`), computes the
per-token score `att_value = sum(q * k) * sm_scale`, and stores it to `Att_Out`,
masked by `offs_n_new < cur_batch_end_index`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel_token_att1[grid](...)` with
`grid = (batch, head, cdiv(max_input_len, BLOCK))`, the scheduling, and how the
runtime composes per-program writes into one buffer) is the *trusted boundary*,
not a proof obligation here. Because the program ids
`(cur_batch, cur_head, start_n)` are universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
token_attn_llama2_output_summary_general                     ← HEADLINE (symbolic dims / sm_scale / strides)
  ├─ token_attn_llama2_surface_toAlgorithm_supported                  surface lowers to the algorithm layer
  └─ token_attn_llama2_surface_output_compute_correct                 full surface, masked score store
       └─ token_attn_llama2_closed_form_correct                       exec readback per lane = closed form
            ├─ llama2_preLoop            prelude assigns → loop-entry invariant
            └─ llama2_loop_body_store    single active iteration writes the masked dot score
(supporting slice spec: token_attn_llama2_score_store_slice_compute_correct
  └─ token_attn_llama2_score_store_slice_correct            algorithm-layer readback per lane)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` are not modeled. The dimension-general headline
`token_attn_llama2_output_summary_general` shows the surface kernel lowers to the
algorithm layer AND the masked store to `Att_Out` is compute-correct against a
**genuine, self-reference-free closed form** `tokenAttnLlama2ClosedForm`: every
active lane (`offs_n_new < cur_batch_end_index`) of an active block holds the
QK-dot score `sm_scale · Σ_d q[d]·k[i,d]` (`tokenAttnLlama2DotScore`, with the
`B_Loc` gather and varlen `start_index` offset folded in), and out-of-bounds
lanes (or any lane of an inactive `block_mask = 0` block) are preserved. This is
proven by fully executing the surface kernel — the prelude decode, the
`block_mask`-guarded single iteration, the masked `B_Loc`/`K` gathers, the
`tl.sum(q·k, 1)` reduce, and the masked store readback — never by re-asserting
the kernel's own executed value. The `start_mark` loop is the `block_mask`-guarded
single iteration of the upstream kernel. The headline is symbolic in
`BLOCK_DMODEL`/`BLOCK_N`/`sm_scale`/strides, so it covers arbitrary shapes (under
the honest side-conditions: a clean `undef` state and output-offset injectivity).
-/

namespace VeriTile.Bench.TritonBenchG.TokenAttnLlama2

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `token_attn_llama2.py`'s
`_fwd_kernel_token_att1`.

Typed-region note: metadata/gather buffers are `Region .nat`, matching their
index role without adding source-level `dtype=` kwargs. -/
def token_attn_llama2_surface
    (Q K : RegionName) (sm_scale : ℝ) (B_Loc B_Start_Loc B_Seqlen : Region .nat)
    (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_n = tl.program_id(2)
  cur_kv_head = cur_head // $(kv_group_num)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_start_index = $(max_input_len) - cur_batch_seq_len
  cur_batch_end_index = $(max_input_len)
  off_q = cur_batch * $(stride_qbs) + cur_head * $(stride_qh) + offs_d * $(stride_qd)
  offs_n = start_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  block_stard_index = start_n * $(BLOCK_N)
  block_mask = tl.where(block_stard_index < cur_batch_seq_len, $(1), $(0))
  for start_mark in range($(0), block_mask, $(1)) {
    q = tl.load(Q + off_q + start_mark)
    offs_n_new = cur_batch_start_index + offs_n
    k_loc = tl.load(B_Loc + $(stride_b_loc_b) * cur_batch +
      $(stride_b_loc_s) * offs_n_new,
      mask=offs_n_new < cur_batch_end_index, other=$(0))
    off_k = k_loc[:, None] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[None, :] * $(stride_kd)
    k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
    att_value = tl.sum(q[None, :] * k, 1)
    att_value *= $((sm_scale : ℝ))
    off_o = cur_head * $(att_stride_h) +
      (cur_batch_in_all_start_index + offs_n) * $(att_stride_bs)
    tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
  }
}

/-- The full token-attention LLaMA2 score surface lowers to the algorithm layer. -/
theorem token_attn_llama2_surface_toAlgorithm_supported
    (Q K : RegionName) (sm_scale : ℝ) (B_Loc B_Start_Loc B_Seqlen : Region .nat)
    (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen
      Att_Out max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh
      stride_qd stride_kbs stride_kh stride_kd att_stride_h att_stride_bs
      kv_group_num BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [token_attn_llama2_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented attention-score store slice of `token_attn_llama2.py`'s
`_fwd_kernel_token_att1`.

The full kernel gathers K, computes `sum(q * k) * sm_scale`, and stores a block
of attention scores. This slice starts from a precomputed `AttValue` vector and
proves the masked writeback into `Att_Out`, preserving the source sequence
window mask. -/
def token_attn_llama2_score_store_slice
    (AttValue : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Att_Out : RegionName)
    (max_input_len att_value_stride_h att_value_stride_bs
      att_stride_h att_stride_bs BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_n = tl.program_id(2)
  offs_n = start_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_start_index = $(max_input_len) - cur_batch_seq_len
  cur_batch_end_index = $(max_input_len)
  offs_n_new = cur_batch_start_index + offs_n
  att_value = tl.load(AttValue + cur_head * $(att_value_stride_h) +
      (cur_batch_in_all_start_index + offs_n) * $(att_value_stride_bs),
    mask=offs_n_new < cur_batch_end_index, other=0.0)
  tl.store(Att_Out + cur_head * $(att_stride_h) +
      (cur_batch_in_all_start_index + offs_n) * $(att_stride_bs),
    att_value, mask=offs_n_new < cur_batch_end_index)
}

def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)

def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

def blockOffset (s : BlockState) (BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 2 * BLOCK_N + i.val

def active
    (s : BlockState) (B_Seqlen : RegionName) (max_input_len BLOCK_N : Nat)
    (i : Fin BLOCK_N) : Prop :=
  max_input_len - seqLen s B_Seqlen + blockOffset s BLOCK_N i < max_input_len

instance activeDecidable
    (s : BlockState) (B_Seqlen : RegionName) (max_input_len BLOCK_N : Nat)
    (i : Fin BLOCK_N) :
    Decidable (active s B_Seqlen max_input_len BLOCK_N i) := by
  unfold active
  infer_instance

def attValueOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (att_value_stride_h att_value_stride_bs BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * att_value_stride_h +
    (startLoc s B_Start_Loc + blockOffset s BLOCK_N i) * att_value_stride_bs

def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (att_stride_h att_stride_bs BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pids 1 * att_stride_h +
    (startLoc s B_Start_Loc + blockOffset s BLOCK_N i) * att_stride_bs

noncomputable def attStoreValue
    (s : BlockState) (AttValue B_Start_Loc B_Seqlen : RegionName)
    (max_input_len att_value_stride_h att_value_stride_bs BLOCK_N : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (if active s B_Seqlen max_input_len BLOCK_N i then
      some (s.readMem AttValue
        (attValueOffset s B_Start_Loc att_value_stride_h att_value_stride_bs
          BLOCK_N i))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the masked Llama token-attention score store. -/
theorem token_attn_llama2_score_store_slice_correct
    (AttValue B_Start_Loc B_Seqlen Att_Out : RegionName)
    (max_input_len att_value_stride_h att_value_stride_bs
      att_stride_h att_stride_bs BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)) :
    ∀ i : Fin BLOCK_N,
      let outAddr := outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i
      (exec (token_attn_llama2_score_store_slice AttValue B_Start_Loc B_Seqlen
            Att_Out max_input_len att_value_stride_h att_value_stride_bs
            att_stride_h att_stride_bs BLOCK_N) s).map (·.readMem Att_Out outAddr)
        = some (if active s B_Seqlen max_input_len BLOCK_N i then
            attStoreValue s AttValue B_Start_Loc B_Seqlen max_input_len
              att_value_stride_h att_value_stride_bs BLOCK_N i
          else s.readMem Att_Out outAddr) := by
  intro i
  simp [exec, token_attn_llama2_score_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.ptrAdd, NumericDType.add,
        NumericDType.sub, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, seqLen, startLoc, blockOffset, active,
        attValueOffset, outOffset]
  let offsetFn : TileIndex [BLOCK_N] → Nat :=
    fun idx =>
      s.pids 1 * att_stride_h +
        (s.readMemValue .nat B_Start_Loc (s.pids 0) +
          (s.pids 2 * BLOCK_N + idx.1.val)) * att_stride_bs
  let valueFn : TileIndex [BLOCK_N] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if max_input_len - s.readMemValue .nat B_Seqlen (s.pids 0) +
              (s.pids 2 * BLOCK_N + idx.1.val) < max_input_len then
          some (s.readMem AttValue
            (s.pids 1 * att_value_stride_h +
              (s.readMemValue .nat B_Start_Loc (s.pids 0) +
                (s.pids 2 * BLOCK_N + idx.1.val)) * att_value_stride_bs))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_N] → Prop :=
    fun idx =>
      max_input_len - s.readMemValue .nat B_Seqlen (s.pids 0) +
        (s.pids 2 * BLOCK_N + idx.1.val) < max_input_len
  have hOffsetInj : Function.Injective offsetFn := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    have habFin : outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N a =
        outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N b := by
      simpa [offsetFn, outOffset, startLoc, blockOffset, BlockState.readMemValue] using hab
    obtain rfl : a = b := hOutInj habFin
    rfl
  change (List.foldl
      (fun (acc : BlockState) idx =>
        if P idx then acc.writeMem Att_Out (offsetFn idx) (valueFn idx) else acc)
      _ (TileShape.allIndices [BLOCK_N])).readMem Att_Out
        (offsetFn (i, PUnit.unit)) =
    if P (i, PUnit.unit) then
      attStoreValue s AttValue B_Start_Loc B_Seqlen max_input_len
        att_value_stride_h att_value_stride_bs BLOCK_N i
    else s.readMem Att_Out (offsetFn (i, PUnit.unit))
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj (i, PUnit.unit)]
  by_cases hi : P (i, PUnit.unit)
  · rw [if_pos hi]
    have hraw :
        max_input_len -
            (match s.readMemTyped TileDType.nat B_Seqlen (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) +
          (s.pids 2 * BLOCK_N + i.val) < max_input_len := by
      simpa [P, BlockState.readMemValue] using hi
    simp [valueFn, P, active, attStoreValue, seqLen, startLoc, blockOffset,
      attValueOffset, outOffset, BlockState.readMemValue, hi, hraw]
    intro hle
    exact False.elim ((not_lt_of_ge hle) hraw)
  · rw [if_neg hi]
    have hraw :
        ¬ max_input_len -
            (match s.readMemTyped TileDType.nat B_Seqlen (s.pids 0) with
            | some value => value
            | none => BlockState.defaultCarrier TileDType.nat) +
          (s.pids 2 * BLOCK_N + i.val) < max_input_len := by
      simpa [P, BlockState.readMemValue] using hi
    simp [P, active, attStoreValue, seqLen, startLoc, blockOffset,
      BlockState.readMemValue, hi, hraw]
    intro hcontr
    exact False.elim (hraw hcontr)

/-- Compute-facing correctness for the masked Llama token-attention score store. -/
theorem token_attn_llama2_score_store_slice_compute_correct
    (AttValue B_Start_Loc B_Seqlen Att_Out : RegionName)
    (max_input_len att_value_stride_h att_value_stride_bs
      att_stride_h att_stride_bs BLOCK_N : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := token_attn_llama2_score_store_slice AttValue B_Start_Loc B_Seqlen
        Att_Out max_input_len att_value_stride_h att_value_stride_bs
        att_stride_h att_stride_bs BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => active s B_Seqlen max_input_len BLOCK_N i)
        (fun i => (Att_Out, outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)))
      (expected := fun i =>
        attStoreValue s AttValue B_Start_Loc B_Seqlen max_input_len
          att_value_stride_h att_value_stride_bs BLOCK_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_llama2_score_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := token_attn_llama2_score_store_slice_correct AttValue B_Start_Loc
    B_Seqlen Att_Out max_input_len att_value_stride_h att_value_stride_bs
    att_stride_h att_stride_bs BLOCK_N s hOutInj i
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Genuine closed-form score spec

The Triton kernel computes, per output lane `i` of the `start_n` block, the
scalar QK dot score `sm_scale · Σ_d q[d]·k[i,d]`, where the query `q[d]` is
loaded from `Q` at `cur_batch·stride_qbs + cur_head·stride_qh + d·stride_qd`,
and the key `k[i,d]` is the *gathered* cache row: the page index
`k_loc[i] = B_Loc[stride_b_loc_b·cur_batch + stride_b_loc_s·offs_n_new[i]]`
selects the KV row, and `k[i,d] = K[k_loc[i]·stride_kbs + cur_kv_head·stride_kh +
d·stride_kd]`.  Both gathers are masked by `offs_n_new[i] < max_input_len`
(`active`), with masked-off lanes reading `0`.  The whole block is gated by the
`block_mask = (start_n·BLOCK_N < cur_batch_seq_len)` single-iteration loop: when
`block_mask = 0` the kernel performs no store and `Att_Out` is preserved.

These definitions are a *genuine closed form* — they never execute the kernel —
and (via `token_attn_llama2_closed_form_correct`) fully replace the former
self-referential surface-value spec. -/

/-- `block_mask = 1` predicate: the `start_n` block has at least one in-range
key token (`start_n·BLOCK_N < cur_batch_seq_len`). When false the kernel's
`range(0, block_mask, 1)` loop is empty and nothing is stored. -/
def blockActive (s : BlockState) (B_Seqlen : RegionName) (BLOCK_N : Nat) : Prop :=
  s.pids 2 * BLOCK_N < seqLen s B_Seqlen

instance blockActiveDecidable (s : BlockState) (B_Seqlen : RegionName) (BLOCK_N : Nat) :
    Decidable (blockActive s B_Seqlen BLOCK_N) := by unfold blockActive; infer_instance

/-- The query-load offset for head dim `d`: `cur_batch·stride_qbs +
cur_head·stride_qh + d·stride_qd` (the `start_mark = 0` loop index folds away). -/
def qOffset (s : BlockState) (stride_qbs stride_qh stride_qd : Nat) (d : Nat) : Nat :=
  s.pids 0 * stride_qbs + s.pids 1 * stride_qh + d * stride_qd

/-- The gathered KV page index for lane `i`:
`B_Loc[stride_b_loc_b·cur_batch + stride_b_loc_s·offs_n_new[i]]`. -/
def kLoc
    (s : BlockState) (B_Loc B_Seqlen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s BLOCK_N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.readMemValue .nat B_Loc
    (stride_b_loc_b * s.pids 0 +
      stride_b_loc_s * (max_input_len - seqLen s B_Seqlen + blockOffset s BLOCK_N i))

/-- The key-load offset for lane `i`, head dim `d`:
`k_loc[i]·stride_kbs + cur_kv_head·stride_kh + d·stride_kd`,
with `cur_kv_head = cur_head / kv_group_num`. -/
def kOffset
    (s : BlockState) (B_Loc B_Seqlen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_kbs stride_kh stride_kd
      kv_group_num BLOCK_N : Nat) (i : Fin BLOCK_N) (d : Nat) : Nat :=
  kLoc s B_Loc B_Seqlen max_input_len stride_b_loc_b stride_b_loc_s BLOCK_N i * stride_kbs +
    (s.pids 1 / kv_group_num) * stride_kh + d * stride_kd

/-- The genuine per-lane closed-form QK score for an *active* lane `i` in an
*active* block: `sm_scale · Σ_{d < BLOCK_DMODEL} Q[qOffset d] · K[kOffset i d]`. -/
noncomputable def tokenAttnLlama2DotScore
    (s : BlockState) (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Seqlen : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  (∑ d : Fin BLOCK_DMODEL,
      s.readMem Q (qOffset s stride_qbs stride_qh stride_qd d.val) *
      s.readMem K (kOffset s B_Loc B_Seqlen max_input_len stride_b_loc_b stride_b_loc_s
        stride_kbs stride_kh stride_kd kv_group_num BLOCK_N i d.val)) * sm_scale

/-- Genuine closed-form value written to `Att_Out` for lane `i`. For an active
lane in an active block it is the QK score; otherwise (inactive lane, or an
inactive `block_mask = 0` block in which the kernel stores nothing) the original
`Att_Out` cell at `offset` is preserved. -/
noncomputable def tokenAttnLlama2ClosedForm
    (s : BlockState) (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : RegionName) (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  if blockActive s B_Seqlen BLOCK_N ∧ active s B_Seqlen max_input_len BLOCK_N i then
    tokenAttnLlama2DotScore s Q K sm_scale B_Loc B_Seqlen max_input_len
      stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd kv_group_num BLOCK_DMODEL BLOCK_N i
  else
    s.readMem Att_Out (outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)

/-! ## Per-statement op-eval recipes (the tedious recipe layer)

Standalone, sorry-free `*_eval` recipe lemmas for the body statements of
`token_attn_llama2_surface`, mirroring the recipe-building patterns of
`VeriTile.Examples.AttentionForwardClosedForm` and the closed `token_attn_mistral`
sibling. Each lemma takes abstract register-readback hypotheses and a symbolic
`BlockState`, and decodes a single statement's `evalOp` to its closed-form tile. -/

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`q` load recipe** (`q = tl.load(Q + off_q + start_mark)`, unmasked, shape
`[BLOCK_DMODEL]`). `off_q` lane `d` is `qOffset d`, `start_mark = SM`, and on a
clean state the loaded address is `qOffset d + SM`; the result lane `d` is
`readMem Q (qOffset d + SM)`. -/
theorem llama2_q_load_eval (s : BlockState) (Q : RegionName) (BD SM : Nat)
    (offq : Fin BD → Nat)
    (hoffq : s.regs .nat [BD] "off_q" = some (Tile.vec offq))
    (hsm : s.regs .nat [] "start_mark" = some (Tile.scalar SM)) :
    evalOp (Op.load .real
        (MemAccess.region Q
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BD] "off_q") (Op.ref .nat [] "start_mark")))
        MaskOpt.none) s
      = some (⟨fun idx : TileIndex [BD] => some (s.readMem Q (offq idx.1 + SM))⟩ : Tile .real [BD]) := by
  simp only [evalOp, hoffq, hsm, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨d, u⟩ := idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, NumericDType.add,
    Broadcast.leftIndex, Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    if_true]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`k_loc` masked gather recipe** (`tl.load(B_Loc + sblb*cur_batch +
sbls*offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)`, shape
`[BLOCK_N]`, dtype `.nat`). On an active lane it gathers `kLoc i`; otherwise `0`. -/
theorem llama2_kloc_gather_eval (s : BlockState)
    (B_Loc B_Seqlen : RegionName) (BN mil sblb sbls : Nat)
    (onn : Fin BN → Nat)
    (honn : s.regs .nat [BN] "offs_n_new" = some (Tile.vec onn))
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 0)))
    (hend : s.regs .nat [] "cur_batch_end_index" = some (Tile.scalar mil)) :
    evalOp (Op.load .nat
        (MemAccess.region B_Loc
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.constNat sblb) (Op.ref .nat [] "cur_batch"))
            (Op.mul .nat Broadcast.scalarL (Op.constNat sbls) (Op.ref .nat [BN] "offs_n_new"))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_n_new")
            (Op.ref .nat [] "cur_batch_end_index"))
          ((Op.constNat 0).broadcast [BN]))) s
      = some (⟨fun idx : TileIndex [BN] =>
          if onn idx.1 < mil then
            s.readMemValue .nat B_Loc (sblb * s.pids 0 + sbls * onn idx.1)
          else 0⟩ : Tile .nat [BN]) := by
  simp only [evalOp, honn, hcb, hend, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, u⟩ := idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, NumericDType.add,
    NumericDType.mul, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
    Region.cast_cast, Region.cast_id]
  by_cases hlt : onn j < mil
  · simp only [hlt, decide_true, if_true, if_pos hlt]
  · simp only [hlt, decide_false, if_neg hlt, if_false, Bool.false_eq_true]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`off_k` 2D recipe** (`off_k = k_loc[:,None]*skbs + cur_kv_head*skh +
offs_d[None,:]*skd`, shape `[BLOCK_N, BLOCK_DMODEL]`). Lane `(j,d)` holds
`kloc j * skbs + kvh * skh + d * skd`. -/
theorem llama2_offk_eval (s : BlockState) (BN BD skbs skh skd : Nat)
    (kloc : Fin BN → Nat) (kvh : Nat)
    (hkloc : s.regs .nat [BN] "k_loc" = some (Tile.vec kloc))
    (hkvh : s.regs .nat [] "cur_kv_head" = some (Tile.scalar kvh))
    (hd : s.regs .nat [BD] "offs_d" = some (Tile.vec (fun e : Fin BD => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "k_loc"))
            (Op.constNat skbs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat skh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d"))
          (Op.constNat skd))) s
      = some (⟨fun idx : TileIndex [BN, BD] =>
          kloc idx.1 * skbs + kvh * skh + idx.2.1.val * skd⟩ : Tile .nat [BN, BD]) := by
  have hexpk : @evalOp .nat [BN, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "k_loc")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec kloc)) :=
    evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, by simp⟩ "k_loc" s _ hkloc
  have hexpd : @evalOp .nat [1, BD] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BD] "offs_d")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun e : Fin BD => e.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BD] ⟨0, by simp⟩ "offs_d" s _ hd
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hexpk, hexpd, hkvh,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, d, u⟩ := idx
  simp only [Tile.bop, Tile.vec, Tile.scalar, Tile.expandDim_data,
    TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    TileShape.dropInsertedIndex_zero_cons, NumericDType.add, NumericDType.mul,
    Broadcast.leftIndex, Broadcast.rightIndex]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`k` 2D masked gather recipe** (`tl.load(K + off_k,
mask=offs_n_new[:,None] < cur_batch_end_index, other=0.0)`, shape
`[BLOCK_N, BLOCK_DMODEL]`). The mask is an `Op.remap` of a `[BN,1]` boundary test
broadcast across head-dim columns. On an active lane `(j,d)` the gathered value is
`readMem K (offk j d)`; otherwise `0`. -/
theorem llama2_k_gather_eval (s : BlockState) (K : RegionName) (BN BD mil : Nat)
    (offk : Fin BN → Fin BD → Nat) (onn : Fin BN → Nat)
    (hoffk : s.regs .nat [BN, BD] "off_k" =
      some (⟨fun idx : TileIndex [BN, BD] => offk idx.1 idx.2.1⟩ : Tile .nat [BN, BD]))
    (honn : s.regs .nat [BN] "offs_n_new" = some (Tile.vec onn))
    (hend : s.regs .nat [] "cur_batch_end_index" = some (Tile.scalar mil)) :
    evalOp (Op.load .real (MemAccess.region K (Op.ref .nat [BN, BD] "off_k"))
        (MaskOpt.maskOther
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n_new"))
              (Op.ref .nat [] "cur_batch_end_index")))
          ((Op.const 0.0).broadcast [BN, BD]))) s
      = some (⟨fun idx : TileIndex [BN, BD] =>
          some (if onn idx.1 < mil then s.readMem K (offk idx.1 idx.2.1) else 0)⟩
          : Tile .real [BN, BD]) := by
  have hexpn : @evalOp .nat [BN, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n_new")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec onn)) :=
    evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, by simp⟩ "offs_n_new" s _ honn
  simp only [evalOp, hoffk, hexpn, hend, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, d, u⟩ := idx
  simp only [Tile.remap, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar,
    Tile.expandDim_data, TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    TileShape.dropInsertedIndex_zero_cons, ComparableDType.lt, Broadcast.leftIndex,
    Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id]
  by_cases hlt : onn j < mil
  · simp only [hlt, decide_true, if_true, if_pos hlt]
  · simp only [hlt, decide_false, if_neg hlt, if_false, Bool.false_eq_true]; norm_num

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`att_value = tl.sum(q[None,:]·k, 1)` dot-reduce recipe** (shape `[BLOCK_N]`).
`expandDim ⟨0,_⟩` lifts `q` to a `[1,BD]` row; the elementwise product with `k`
gives `[BN,BD]`; the `reduceSum` over axis 1 contracts the head-dim. Given
`q` lane `d = qVal d` and `k` lane `(j,d) = kVal j d`, the result lane `j` is
`Σ_{d} qVal d · kVal j d`. -/
theorem llama2_attvalue_eval (s : BlockState) (BN BD : Nat)
    (qVal : Fin BD → ℝ) (kVal : Fin BN → Fin BD → ℝ)
    (hq : s.regs .real [BD] "q" =
      some (⟨fun idx : TileIndex [BD] => some (qVal idx.1)⟩ : Tile .real [BD]))
    (hk : s.regs .real [BN, BD] "k" =
      some (⟨fun idx : TileIndex [BN, BD] => some (kVal idx.1 idx.2.1)⟩ : Tile .real [BN, BD])) :
    evalOp (Op.reduceSum ⟨1, by simp⟩ Bool.false
        (Op.mul .real Broadcast.nil.consSame.consL
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BD] "q"))
          (Op.ref .real [BN, BD] "k"))) s
      = some (⟨fun idx : TileIndex [BN] =>
          some (∑ d : Fin BD, qVal d * kVal idx.1 d)⟩ : Tile .real [BN]) := by
  have hexp : @evalOp .real [1, BD] (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BD] "q")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (⟨fun idx : TileIndex [BD] => some (qVal idx.1)⟩ : Tile .real [BD])) :=
    evalOp_expandDim_ref_of_regs .real [BD] ⟨0, by simp⟩ "q" s _ hq
  have hmul : evalOp (Op.mul .real Broadcast.nil.consSame.consL
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BD] "q"))
        (Op.ref .real [BN, BD] "k")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.nil.consSame.consL
          (Tile.expandDim ⟨0, by simp⟩ (⟨fun idx : TileIndex [BD] => some (qVal idx.1)⟩ : Tile .real [BD]))
          (⟨fun idx : TileIndex [BN, BD] => some (kVal idx.1 idx.2.1)⟩ : Tile .real [BN, BD])) := by
    rw [evalOp_mul, hexp, evalOp_ref, hk]; rfl
  rw [evalOp_reduceSum, hmul]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceSum_false]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, u⟩ := idx
  simp only [Tile.reduceSumDrop_data, Tile.bop_data, Tile.bop, Tile.expandDim_data,
    TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    TileShape.dropInsertedIndex_zero_cons, TileShape.insertAxisIndex, TileShape.axisDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul]
  have hcoe : ∀ d : Fin BD, WithBot.realMul (some (qVal d)) (some (kVal j d))
      = ((qVal d * kVal j d : ℝ) : WithBot ℝ) := fun d => rfl
  rw [Finset.sum_congr rfl (fun d _ => hcoe d), ← WithBot.coe_sum]
  rfl

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`off_q` recipe** (`off_q = cur_batch·stride_qbs + cur_head·stride_qh +
offs_d·stride_qd`, shape `[BLOCK_DMODEL]`). Lane `d` holds `qOffset … d`. -/
theorem llama2_offq_eval (s : BlockState) (BD batch head stride_qbs stride_qh stride_qd : Nat)
    (hcb : s.regs .nat [] "cur_batch" = some (Tile.scalar batch))
    (hch : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hd : s.regs .nat [BD] "offs_d" = some (Tile.vec (fun e : Fin BD => e.val))) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat stride_qbs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_qh)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BD] "offs_d") (Op.constNat stride_qd))) s
      = some (Tile.vec (fun d : Fin BD =>
          batch * stride_qbs + head * stride_qh + d.val * stride_qd)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hcb, hch, hd,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul,
    Broadcast.leftIndex, Broadcast.rightIndex]

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **`offs_n` recipe** (`offs_n = start_n·BLOCK_N + arange BLOCK_N`, shape
`[BLOCK_N]`). Lane `j` holds `sn·BLOCK_N + j`. -/
theorem llama2_offsn_eval (s : BlockState) (BN sn BLOCK_N : Nat)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar sn)) :
    evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BLOCK_N)) (Op.arange BN)) s
      = some (Tile.vec (fun j : Fin BN => sn * BLOCK_N + j.val)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_arange, evalOp_constNat, evalOp_ref, hsn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul,
    Broadcast.leftIndex, Broadcast.rightIndex]

/-! ## Operational `exec` decode: prelude + single-iteration store readback

`tokenAttnLlama2ClosedForm` is the genuine, self-reference-free score spec. The
bridge fully executes the surface kernel: the 13-assign prelude decode, the
`block_mask`-guarded single-iteration loop (`step_one_iter` when active,
`step_ge` when inactive), the loop body's `B_Loc`/`K` masked gathers, the
`tl.sum(q[None,:]·k, 1)` dot reduce, the `sm_scale` mul, and the masked store
readback — never re-asserting the kernel's own executed value. -/

/-- The loop-entry invariant: the 13-assign prelude has run, seeding every
loop-invariant register from `s0`. -/
def llama2Invariant
    (Q K : RegionName) (B_Loc B_Start_Loc B_Seqlen : RegionName)
    (s0 : BlockState)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "cur_kv_head" = some (Tile.scalar (s0.pids 1 / kv_group_num))
  ∧ s.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
  ∧ s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s0 B_Seqlen))
  ∧ s.regs .nat [] "cur_batch_start_index" =
      some (Tile.scalar (max_input_len - seqLen s0 B_Seqlen))
  ∧ s.regs .nat [] "cur_batch_end_index" = some (Tile.scalar max_input_len)
  ∧ s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s0 B_Start_Loc))
  ∧ s.regs .nat [BLOCK_DMODEL] "off_q" =
      some (Tile.vec (fun d : Fin BLOCK_DMODEL => qOffset s0 stride_qbs stride_qh stride_qd d.val))
  ∧ s.regs .nat [BLOCK_N] "offs_n" =
      some (Tile.vec (fun j : Fin BLOCK_N => s0.pids 2 * BLOCK_N + j.val))
  ∧ s.regs .nat [] "block_mask" =
      some (Tile.scalar (if blockActive s0 B_Seqlen BLOCK_N then 1 else 0))

set_option maxHeartbeats 1000000 in
set_option maxRecDepth 8000 in
/-- **preLoop** (13 prelude assigns): from a clean input state, the prologue steps
to a state satisfying `llama2Invariant`. -/
theorem llama2_preLoop
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts ((token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
        B_Seqlen Att_Out max_input_len stride_b_loc_b stride_b_loc_s stride_qbs
        stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h att_stride_bs
        kv_group_num BLOCK_DMODEL BLOCK_N).toAlgKernel.body.take 13) s = some s'
      ∧ llama2Invariant Q K B_Loc B_Start_Loc B_Seqlen s max_input_len stride_b_loc_b stride_b_loc_s
          stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h
          att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N s' := by
  rw [show ((token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
        max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd stride_kbs
        stride_kh stride_kd att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N).toAlgKernel.body.take 13)
      = [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
          Stmt.assign .nat [] "cur_head" (Op.programId 1),
          Stmt.assign .nat [] "start_n" (Op.programId 2),
          Stmt.assign .nat [] "cur_kv_head"
            (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat kv_group_num)),
          Stmt.assign .nat [BLOCK_DMODEL] "offs_d" (Op.arange BLOCK_DMODEL),
          Stmt.assign .nat [] "cur_batch_seq_len"
            (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
          Stmt.assign .nat [] "cur_batch_in_all_start_index"
            (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
          Stmt.assign .nat [] "cur_batch_start_index"
            (Op.sub .nat Broadcast.nil (Op.constNat max_input_len) (Op.ref .nat [] "cur_batch_seq_len")),
          Stmt.assign .nat [] "cur_batch_end_index" (Op.constNat max_input_len),
          Stmt.assign .nat [BLOCK_DMODEL] "off_q"
            (Op.add .nat Broadcast.scalarL
              (Op.add .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat stride_qbs))
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_qh)))
              (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_DMODEL] "offs_d") (Op.constNat stride_qd))),
          Stmt.assign .nat [BLOCK_N] "offs_n"
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BLOCK_N)) (Op.arange BLOCK_N)),
          Stmt.assign .nat [] "block_stard_index"
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BLOCK_N)),
          Stmt.assign .nat [] "block_mask"
            ((Op.lt .nat Broadcast.nil (Op.ref .nat [] "block_stard_index")
                  (Op.ref .nat [] "cur_batch_seq_len")).where
              (Op.constNat 1) (Op.constNat 0)) ] from rfl]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat kv_group_num)) _
        = some (Tile.scalar (s.pids 1 / kv_group_num)) from by
      simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind]
      refine congrArg some ?_
      ext idx
      simp only [Tile.bop, Tile.scalar, BlockState.setReg_pids, IntegralDType.nat_floorDiv,
        castTile_self, Tile.scalar_data_index, Broadcast.leftIndex, Broadcast.rightIndex]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_DMODEL) _ = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) from by
      rw [evalOp_arange]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (seqLen s B_Seqlen)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (startLoc s B_Start_Loc)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.sub .nat Broadcast.nil (Op.constNat max_input_len) (Op.ref .nat [] "cur_batch_seq_len")) _
        = some (Tile.scalar (max_input_len - seqLen s B_Seqlen)) from by
      simp only [evalOp_sub, evalOp_constNat, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind]
      refine congrArg some ?_
      ext idx
      simp only [Tile.bop, Tile.scalar, NumericDType.sub, Broadcast.leftIndex, Broadcast.rightIndex,
        castTile_self, Tile.scalar_data, Tile.scalar_data_index]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.constNat max_input_len) _ = some (Tile.scalar max_input_len) from evalOp_constNat _ _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat stride_qbs))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat stride_qh)))
            (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BLOCK_DMODEL] "offs_d") (Op.constNat stride_qd))) _
        = some (Tile.vec (fun d : Fin BLOCK_DMODEL => qOffset s stride_qbs stride_qh stride_qd d.val)) from by
      rw [llama2_offq_eval _ BLOCK_DMODEL (s.pids 0) (s.pids 1) stride_qbs stride_qh stride_qd
        (by simp) (by simp) (by simp [Tile.vec])]
      refine congrArg some ?_
      ext idx
      simp [Tile.vec, qOffset]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BLOCK_N)) (Op.arange BLOCK_N)) _
        = some (Tile.vec (fun j : Fin BLOCK_N => s.pids 2 * BLOCK_N + j.val)) from by
      rw [llama2_offsn_eval _ BLOCK_N (s.pids 2) BLOCK_N (by simp)]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BLOCK_N)) _
        = some (Tile.scalar (s.pids 2 * BLOCK_N)) from by
      simp only [evalOp_mul, evalOp_constNat, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind]
      refine congrArg some ?_
      ext idx
      simp only [Tile.bop, Tile.scalar, NumericDType.mul, Broadcast.leftIndex, Broadcast.rightIndex,
        BlockState.setReg_pids, castTile_self, Tile.scalar_data_index]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp ((Op.lt .nat Broadcast.nil (Op.ref .nat [] "block_stard_index")
              (Op.ref .nat [] "cur_batch_seq_len")).where (Op.constNat 1) (Op.constNat 0)) _
        = some (Tile.scalar (if blockActive s B_Seqlen BLOCK_N then 1 else 0)) from by
      simp only [evalOp_where, evalOp_lt, evalOp_constNat, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind]
      refine congrArg some ?_
      ext idx
      simp only [Tile.select_data, Tile.cop_data, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, ComparableDType.lt, blockActive, seqLen]
      by_cases h : s.pids 2 * BLOCK_N < s.readMemValue .nat B_Seqlen (s.pids 0)
      · rw [if_pos (by simpa using h), if_pos h]
      · rw [if_neg (by simpa using h), if_neg h]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp, ?_, ?_, by simp, by simp, by simp, by simp [Tile.vec], by simp, by simp, by simp,
    by simp, by simp [Tile.vec], by simp [Tile.vec], by simp⟩
  · funext rg o; simp
  · intro rg o; simp [hundef]

/-- The 9-statement loop body of `token_attn_llama2_surface` (q load, offs_n_new,
k_loc gather, off_k, k gather, att_value reduce, sm_scale mul, off_o, masked
store), transcribed. Independent of region names except `Q`/`K`/`B_Loc`/`Att_Out`. -/
def llama2LoopBody
    (Q K B_Loc Att_Out : RegionName) (sm_scale : ℝ)
    (stride_b_loc_b stride_b_loc_s stride_kbs stride_kh stride_kd att_stride_h att_stride_bs
      BLOCK_DMODEL BLOCK_N : Nat) : List Stmt :=
  [ Stmt.assign .real [BLOCK_DMODEL] "q"
      (Op.load .real
        (MemAccess.region Q
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BLOCK_DMODEL] "off_q")
            (Op.ref .nat [] "start_mark")))
        MaskOpt.none),
    Stmt.assign .nat [BLOCK_N] "offs_n_new"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_start_index")
        (Op.ref .nat [BLOCK_N] "offs_n")),
    Stmt.assign .nat [BLOCK_N] "k_loc"
      (Op.load .nat
        (MemAccess.region B_Loc
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.constNat stride_b_loc_b) (Op.ref .nat [] "cur_batch"))
            (Op.mul .nat Broadcast.scalarL (Op.constNat stride_b_loc_s)
              (Op.ref .nat [BLOCK_N] "offs_n_new"))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offs_n_new")
            (Op.ref .nat [] "cur_batch_end_index"))
          ((Op.constNat 0).broadcast [BLOCK_N]))),
    Stmt.assign .nat [BLOCK_N, BLOCK_DMODEL] "off_k"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "k_loc"))
            (Op.constNat stride_kbs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat stride_kh)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_d"))
          (Op.constNat stride_kd))),
    Stmt.assign .real [BLOCK_N, BLOCK_DMODEL] "k"
      (Op.load .real (MemAccess.region K (Op.ref .nat [BLOCK_N, BLOCK_DMODEL] "off_k"))
        (MaskOpt.maskOther
          (Op.remap [BLOCK_N, BLOCK_DMODEL] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n_new"))
              (Op.ref .nat [] "cur_batch_end_index")))
          ((Op.const 0.0).broadcast [BLOCK_N, BLOCK_DMODEL]))),
    Stmt.assign .real [BLOCK_N] "att_value"
      (Op.reduceSum ⟨1, by simp⟩ Bool.false
        (Op.mul .real Broadcast.nil.consSame.consL
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BLOCK_DMODEL] "q")) (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "k"))),
    Stmt.assign .real [BLOCK_N] "att_value"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BLOCK_N] "att_value") (Op.const sm_scale)),
    Stmt.assign .nat [BLOCK_N] "off_o"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat att_stride_h))
        (Op.mul .nat Broadcast.scalarR
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
            (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.constNat att_stride_bs))),
    Stmt.store .real [BLOCK_N] (MemAccess.region Att_Out (Op.ref .nat [BLOCK_N] "off_o"))
      (Op.ref .real [BLOCK_N] "att_value")
      (MaskOpt.mask
        (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offs_n_new")
          (Op.ref .nat [] "cur_batch_end_index"))) ]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Loop-body store readback** (the single active iteration). Given the
loop-entry invariant on `st` (= the prelude post-state), which pins
`cur_batch_in_all_start_index = startLoc s0 B_Start_Loc`, the body run from
`st.setReg "start_mark" 0`
writes, at every `Att_Out[outOffset … i]`, the dot score on active lanes and the
original value on inactive lanes. -/
theorem llama2_loop_body_store
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : RegionName) (Att_Out : RegionName)
    (s0 : BlockState)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat)
    (st : BlockState)
    (hinv : llama2Invariant Q K B_Loc B_Start_Loc B_Seqlen s0 max_input_len stride_b_loc_b stride_b_loc_s
        stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h att_stride_bs
        kv_group_num BLOCK_DMODEL BLOCK_N st)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s0 B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)) :
    ∃ sfin, stepStmts (llama2LoopBody Q K B_Loc Att_Out sm_scale stride_b_loc_b stride_b_loc_s
        stride_kbs stride_kh stride_kd att_stride_h att_stride_bs BLOCK_DMODEL BLOCK_N)
        (st.setReg "start_mark" .nat [] (Tile.scalar 0)) = some sfin
      ∧ ∀ i : Fin BLOCK_N,
          sfin.readMem Att_Out (outOffset s0 B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)
            = if active s0 B_Seqlen max_input_len BLOCK_N i then
                tokenAttnLlama2DotScore s0 Q K sm_scale B_Loc B_Seqlen max_input_len
                  stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
                  stride_kbs stride_kh stride_kd kv_group_num BLOCK_DMODEL BLOCK_N i
              else s0.readMem Att_Out (outOffset s0 B_Start_Loc att_stride_h att_stride_bs BLOCK_N i) := by
  obtain ⟨hpids, hmem, hundef, hcb, hch, hckv, hd, hsl, hsi, hend, hSL, hoffq, hoffn, _hmask⟩ := hinv
  -- abbreviations
  set onn : Fin BLOCK_N → Nat :=
    fun j => (max_input_len - seqLen s0 B_Seqlen) + (s0.pids 2 * BLOCK_N + j.val) with honnDef
  set kloc : Fin BLOCK_N → Nat :=
    fun j => if onn j < max_input_len then
        s0.readMemValue .nat B_Loc (stride_b_loc_b * s0.pids 0 + stride_b_loc_s * onn j) else 0
    with klocDef
  set offk : Fin BLOCK_N → Fin BLOCK_DMODEL → Nat :=
    fun j d => kloc j * stride_kbs + (s0.pids 1 / kv_group_num) * stride_kh + d.val * stride_kd
    with offkDef
  set kval : Fin BLOCK_N → Fin BLOCK_DMODEL → ℝ :=
    fun j d => if onn j < max_input_len then s0.readMem K (offk j d) else 0 with kvalDef
  set qval : Fin BLOCK_DMODEL → ℝ := fun d => s0.readMem Q (qOffset s0 stride_qbs stride_qh stride_qd d.val)
    with qvalDef
  -- entry state with start_mark = 0
  set sm0 : BlockState := st.setReg "start_mark" .nat [] (Tile.scalar 0) with hsm0
  have hsm0mem : sm0.mem = s0.mem := by rw [hsm0]; exact hmem
  have hsm0pids : sm0.pids = s0.pids := by rw [hsm0]; exact hpids
  unfold llama2LoopBody
  -- statement 1: q load
  have hsm : sm0.regs .nat [] "start_mark" = some (Tile.scalar 0) := by rw [hsm0]; simp
  have hoffqSm : sm0.regs .nat [BLOCK_DMODEL] "off_q" =
      some (Tile.vec (fun d : Fin BLOCK_DMODEL => qOffset s0 stride_qbs stride_qh stride_qd d.val)) := by
    rw [hsm0]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hoffq
  set s1 : BlockState := sm0.setReg "q" .real [BLOCK_DMODEL]
    (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (qval idx.1)⟩ : Tile .real [BLOCK_DMODEL]) with hs1
  rw [stepStmts.cons_some (show stepStmt _ sm0 = some s1 from stepStmt_assign_eq_some (by
    have h := llama2_q_load_eval sm0 Q BLOCK_DMODEL 0
      (fun d => qOffset s0 stride_qbs stride_qh stride_qd d.val) hoffqSm hsm
    rw [h]
    refine congrArg some ?_
    ext idx
    simp only [hs1, qvalDef]
    rw [show sm0.readMem = s0.readMem from by funext rg o; simp only [BlockState.readMem, hsm0mem]]
    simp))]
  have hs1mem : s1.mem = s0.mem := by rw [hs1]; exact hsm0mem
  have hs1pids : s1.pids = s0.pids := by rw [hs1]; exact hsm0pids
  -- statement 2: offs_n_new = cur_batch_start_index + offs_n
  have hsiS1 : s1.regs .nat [] "cur_batch_start_index" =
      some (Tile.scalar (max_input_len - seqLen s0 B_Seqlen)) := by
    rw [hs1, hsm0]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hsi
  have hoffnS1 : s1.regs .nat [BLOCK_N] "offs_n" =
      some (Tile.vec (fun j : Fin BLOCK_N => s0.pids 2 * BLOCK_N + j.val)) := by
    rw [hs1, hsm0]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hoffn
  set s2 : BlockState := s1.setReg "offs_n_new" .nat [BLOCK_N] (Tile.vec onn) with hs2
  rw [stepStmts.cons_some (show stepStmt _ s1 = some s2 from stepStmt_assign_eq_some (by
    simp only [evalOp_add, evalOp_ref, hsiS1, hoffnS1, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext idx
    simp only [hs2, Tile.bop, Tile.vec, Tile.scalar, NumericDType.add, Broadcast.leftIndex,
      Broadcast.rightIndex, honnDef]))]
  have hs2mem : s2.mem = s0.mem := by rw [hs2]; exact hs1mem
  have hs2pids : s2.pids = s0.pids := by rw [hs2]; exact hs1pids
  -- statement 3: k_loc gather
  have honnS2 : s2.regs .nat [BLOCK_N] "offs_n_new" = some (Tile.vec onn) := by rw [hs2]; simp
  have hcbS2 : s2.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0)) := by
    rw [hs2, hs1, hsm0]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hcb
  have hendS2 : s2.regs .nat [] "cur_batch_end_index" = some (Tile.scalar max_input_len) := by
    rw [hs2, hs1, hsm0]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hend
  set s3 : BlockState := s2.setReg "k_loc" .nat [BLOCK_N] (Tile.vec kloc) with hs3
  rw [stepStmts.cons_some (show stepStmt _ s2 = some s3 from stepStmt_assign_eq_some (by
    have h := llama2_kloc_gather_eval s2 B_Loc B_Seqlen BLOCK_N max_input_len stride_b_loc_b
      stride_b_loc_s onn honnS2 (by rw [hs2pids]; exact hcbS2) hendS2
    rw [h]
    refine congrArg some ?_
    ext idx
    simp only [hs3, Tile.vec, klocDef, hs2pids]
    rw [show s2.readMemValue .nat = s0.readMemValue .nat from by
      funext rg o; simp only [BlockState.readMemValue, BlockState.readMemTyped, hs2mem]]))]
  have hs3mem : s3.mem = s0.mem := by rw [hs3]; exact hs2mem
  have hs3pids : s3.pids = s0.pids := by rw [hs3]; exact hs2pids
  -- statement 4: off_k
  have hklocS3 : s3.regs .nat [BLOCK_N] "k_loc" = some (Tile.vec kloc) := by rw [hs3]; simp
  have hckvS3 : s3.regs .nat [] "cur_kv_head" = some (Tile.scalar (s0.pids 1 / kv_group_num)) := by
    rw [hs3, hs2, hs1, hsm0]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hckv
  have hdS3 : s3.regs .nat [BLOCK_DMODEL] "offs_d" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) := by
    rw [hs3, hs2, hs1, hsm0]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hd
  set s4 : BlockState := s3.setReg "off_k" .nat [BLOCK_N, BLOCK_DMODEL]
    (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => offk idx.1 idx.2.1⟩ : Tile .nat [BLOCK_N, BLOCK_DMODEL]) with hs4
  rw [stepStmts.cons_some (show stepStmt _ s3 = some s4 from stepStmt_assign_eq_some (by
    exact llama2_offk_eval s3 BLOCK_N BLOCK_DMODEL stride_kbs stride_kh stride_kd kloc
      (s0.pids 1 / kv_group_num) hklocS3 hckvS3 hdS3))]
  have hs4mem : s4.mem = s0.mem := by rw [hs4]; exact hs3mem
  have hs4pids : s4.pids = s0.pids := by rw [hs4]; exact hs3pids
  -- statement 5: k gather
  have hoffkS4 : s4.regs .nat [BLOCK_N, BLOCK_DMODEL] "off_k" =
      some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => offk idx.1 idx.2.1⟩ : Tile .nat [BLOCK_N, BLOCK_DMODEL]) := by
    rw [hs4]; simp
  have honnS4 : s4.regs .nat [BLOCK_N] "offs_n_new" = some (Tile.vec onn) := by
    rw [hs4, hs3]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact honnS2
  have hendS4 : s4.regs .nat [] "cur_batch_end_index" = some (Tile.scalar max_input_len) := by
    rw [hs4, hs3]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; exact hendS2
  set s5 : BlockState := s4.setReg "k" .real [BLOCK_N, BLOCK_DMODEL]
    (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => some (kval idx.1 idx.2.1)⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]) with hs5
  rw [stepStmts.cons_some (show stepStmt _ s4 = some s5 from stepStmt_assign_eq_some (by
    have h := llama2_k_gather_eval s4 K BLOCK_N BLOCK_DMODEL max_input_len offk onn hoffkS4 honnS4 hendS4
    rw [h]
    refine congrArg some ?_
    ext idx
    obtain ⟨j, e, u⟩ := idx
    simp only [hs5, kvalDef]
    rw [show s4.readMem = s0.readMem from by funext rg o; simp only [BlockState.readMem, hs4mem]]))]
  have hs5mem : s5.mem = s0.mem := by rw [hs5]; exact hs4mem
  have hs5pids : s5.pids = s0.pids := by rw [hs5]; exact hs4pids
  -- statement 6: att_value = reduceSum (q[None,:] · k)
  have hqS5 : s5.regs .real [BLOCK_DMODEL] "q" =
      some (⟨fun idx : TileIndex [BLOCK_DMODEL] => some (qval idx.1)⟩ : Tile .real [BLOCK_DMODEL]) := by
    rw [hs5, hs4, hs3, hs2]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; rw [hs1]; simp
  have hkS5 : s5.regs .real [BLOCK_N, BLOCK_DMODEL] "k" =
      some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] => some (kval idx.1 idx.2.1)⟩ : Tile .real [BLOCK_N, BLOCK_DMODEL]) := by
    rw [hs5]; simp
  set avraw : Fin BLOCK_N → ℝ := fun j => ∑ d : Fin BLOCK_DMODEL, qval d * kval j d with avrawDef
  set s6 : BlockState := s5.setReg "att_value" .real [BLOCK_N]
    (⟨fun idx : TileIndex [BLOCK_N] => some (avraw idx.1)⟩ : Tile .real [BLOCK_N]) with hs6
  rw [stepStmts.cons_some (show stepStmt _ s5 = some s6 from stepStmt_assign_eq_some (by
    exact llama2_attvalue_eval s5 BLOCK_N BLOCK_DMODEL qval kval hqS5 hkS5))]
  have hs6mem : s6.mem = s0.mem := by rw [hs6]; exact hs5mem
  have hs6pids : s6.pids = s0.pids := by rw [hs6]; exact hs5pids
  -- statement 7: att_value *= sm_scale
  have havS6 : s6.regs .real [BLOCK_N] "att_value" =
      some (⟨fun idx : TileIndex [BLOCK_N] => some (avraw idx.1)⟩ : Tile .real [BLOCK_N]) := by
    rw [hs6]; simp
  set s7 : BlockState := s6.setReg "att_value" .real [BLOCK_N]
    (⟨fun idx : TileIndex [BLOCK_N] => some (avraw idx.1 * sm_scale)⟩ : Tile .real [BLOCK_N]) with hs7
  rw [stepStmts.cons_some (show stepStmt _ s6 = some s7 from stepStmt_assign_eq_some (by
    simp only [evalOp_mul, evalOp_ref, evalOp_const, havS6, Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext idx
    simp only [hs7, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]
    rfl))]
  have hs7mem : s7.mem = s0.mem := by rw [hs7]; exact hs6mem
  have hs7pids : s7.pids = s0.pids := by rw [hs7]; exact hs6pids
  -- statement 8: off_o = cur_head*ash + (cur_batch_in_all_start_index + offs_n)*asbs
  have hchS7 : s7.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1)) := by
    rw [hs7, hs6, hs5, hs4, hs3, hs2]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; rw [hs1, hsm0]
    simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
    exact hch
  have hSLS7 : s7.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s0 B_Start_Loc)) := by
    rw [hs7, hs6, hs5, hs4, hs3, hs2]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; rw [hs1, hsm0]
    simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
    exact hSL
  have hoffnS7 : s7.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => s0.pids 2 * BLOCK_N + j.val)) := by
    rw [hs7, hs6, hs5, hs4, hs3, hs2]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; rw [hs1]
    simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true, String.reduceEq]
    exact hoffnS1
  set ooFn : Fin BLOCK_N → Nat :=
    fun j => s0.pids 1 * att_stride_h + (startLoc s0 B_Start_Loc + (s0.pids 2 * BLOCK_N + j.val)) * att_stride_bs
    with ooFnDef
  set s8 : BlockState := s7.setReg "off_o" .nat [BLOCK_N] (Tile.vec ooFn) with hs8
  rw [stepStmts.cons_some (show stepStmt _ s7 = some s8 from stepStmt_assign_eq_some (by
    simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hchS7, hSLS7, hoffnS7,
      Option.bind_eq_bind, Option.bind_some]
    refine congrArg some ?_
    ext idx
    simp only [hs8, Tile.bop, Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul,
      Broadcast.leftIndex, Broadcast.rightIndex, ooFnDef]))]
  have hs8mem : s8.mem = s0.mem := by rw [hs8]; exact hs7mem
  have hs8pids : s8.pids = s0.pids := by rw [hs8]; exact hs7pids
  -- statement 9: masked store
  have hoffoS8 : s8.regs .nat [BLOCK_N] "off_o" = some (Tile.vec ooFn) := by rw [hs8]; simp
  have havS8 : s8.regs .real [BLOCK_N] "att_value" =
      some (⟨fun idx : TileIndex [BLOCK_N] => some (avraw idx.1 * sm_scale)⟩ : Tile .real [BLOCK_N]) := by
    rw [hs8]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq, not_false_eq_true,
      String.reduceEq]; rw [hs7]; simp
  have honnS8 : s8.regs .nat [BLOCK_N] "offs_n_new" = some (Tile.vec onn) := by
    rw [hs8, hs7, hs6, hs5]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact honnS4
  have hendS8 : s8.regs .nat [] "cur_batch_end_index" = some (Tile.scalar max_input_len) := by
    rw [hs8, hs7, hs6, hs5]; simp only [BlockState.setReg_ne_name, ne_eq, reduceCtorEq,
      not_false_eq_true, String.reduceEq]; exact hendS4
  -- ooFn injective (= outOffset)
  have hooInj : Function.Injective ooFn := by
    intro a b hab
    apply hOutInj
    simp only [outOffset, startLoc, blockOffset, ooFnDef] at hab ⊢
    exact hab
  -- ooFn injective as a TileIndex offset function
  have hooFnInj : Function.Injective (fun k : TileIndex [BLOCK_N] => ooFn k.1) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := hooInj hab
    rfl
  -- decode the masked store via the slice-style scatter readback
  set valFn : TileIndex [BLOCK_N] → ℝ := fun k => avraw k.1 * sm_scale with hvalFn
  set P : TileIndex [BLOCK_N] → Prop := fun k => onn k.1 < max_input_len with hPdef
  have hPdec : DecidablePred P := by intro k; rw [hPdef]; infer_instance
  have hstoreStep : stepStmt (Stmt.store .real [BLOCK_N] (MemAccess.region Att_Out (Op.ref .nat [BLOCK_N] "off_o"))
      (Op.ref .real [BLOCK_N] "att_value")
      (MaskOpt.mask (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offs_n_new")
        (Op.ref .nat [] "cur_batch_end_index")))) s8
      = some ((TileShape.allIndices [BLOCK_N]).foldl
          (fun acc k => if P k then acc.writeMem Att_Out (ooFn k.1) (valFn k) else acc) s8) := by
    simp only [stepStmt, evalOp_ref, hoffoS8, havS8, evalOp_lt, honnS8, hendS8,
      Option.bind_some, Option.bind_eq_bind, Option.map_some, Option.some.injEq,
      ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex, Tile.cop_data,
      Tile.scalar_data, Tile.vec, decide_eq_true_eq, hPdef]
    rfl
  have hstore := @stepStmts.cons_some _ [] _ _ hstoreStep
  rw [stepStmts.nil] at hstore
  refine ⟨_, hstore, ?_⟩
  intro i
  have hooEq : outOffset s0 B_Start_Loc att_stride_h att_stride_bs BLOCK_N i = ooFn i := by
    simp only [outOffset, startLoc, blockOffset, ooFnDef]
  rw [hooEq]
  rw [show (ooFn i) = (fun k : TileIndex [BLOCK_N] => ooFn k.1) (i, PUnit.unit) from rfl,
    BlockState.scatter_readback_prop_masked_nd (region := Att_Out) s8
      (fun k : TileIndex [BLOCK_N] => ooFn k.1) valFn P hooFnInj (i, PUnit.unit)]
  -- relate s8.readMem to s0.readMem and DotScore to valFn on active lanes
  have hs8read : s8.readMem Att_Out (ooFn i) = s0.readMem Att_Out (ooFn i) := by
    simp only [BlockState.readMem, hs8mem]
  by_cases hac : active s0 B_Seqlen max_input_len BLOCK_N i
  · have honnlt : P (i, PUnit.unit) := by
      simp only [hPdef, honnDef]
      simpa only [active, seqLen, blockOffset] using hac
    rw [if_pos honnlt, if_pos hac]
    have honnlt' : onn i < max_input_len := honnlt
    -- offk i d = kOffset i d for the active lane
    have hklocEq : kloc i =
        kLoc s0 B_Loc B_Seqlen max_input_len stride_b_loc_b stride_b_loc_s BLOCK_N i := by
      simp only [klocDef]
      rw [if_pos honnlt']
      simp only [kLoc, seqLen, blockOffset, honnDef]
    have hoffk : ∀ d : Fin BLOCK_DMODEL, offk i d =
        kOffset s0 B_Loc B_Seqlen max_input_len stride_b_loc_b stride_b_loc_s stride_kbs stride_kh
          stride_kd kv_group_num BLOCK_N i d.val := by
      intro d
      simp only [offkDef, kOffset, hklocEq]
    simp only [hvalFn, avrawDef, qvalDef, kvalDef, tokenAttnLlama2DotScore]
    congr 1
    apply Finset.sum_congr rfl
    intro d _
    rw [if_pos honnlt', hoffk d]
  · have honnlt : ¬ P (i, PUnit.unit) := by
      simp only [hPdef, honnDef]
      simpa only [active, seqLen, blockOffset] using hac
    rw [if_neg honnlt, if_neg hac, hs8read]

/-- Body decomposition: `prelude(13) ++ [forRangeDyn body]`. By `rfl`. -/
theorem llama2_body_split
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat) :
    (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out max_input_len
        stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd stride_kbs stride_kh
        stride_kd att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N).toAlgKernel.body
      = (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out max_input_len
          stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd stride_kbs stride_kh
          stride_kd att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N).toAlgKernel.body.take 13
        ++ [Stmt.forRangeDyn "start_mark" (Op.constNat 0) (Op.ref .nat [] "block_mask") (Op.constNat 1)
              (llama2LoopBody Q K B_Loc Att_Out sm_scale stride_b_loc_b stride_b_loc_s stride_kbs
                stride_kh stride_kd att_stride_h att_stride_bs BLOCK_DMODEL BLOCK_N)] := by
  rfl

set_option maxHeartbeats 1000000 in
/-- **Genuine closed-form correctness for `token_attn_llama2` (general).** Every
output lane `i` of the masked score store holds the genuine QK-dot closed form
`tokenAttnLlama2ClosedForm` — the dot score on an active lane of an active block,
the preserved `Att_Out` cell otherwise — NOT the kernel's own executed value. -/
theorem token_attn_llama2_closed_form_correct
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i))
    (i : Fin BLOCK_N) :
    (match exec (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
        max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd stride_kbs
        stride_kh stride_kd att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N) s with
      | some s' => s'.readMem Att_Out (outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)
      | none => (0.0 : ℝ)) =
      tokenAttnLlama2ClosedForm s Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out max_input_len
        stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd stride_kbs stride_kh
        stride_kd att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N i := by
  obtain ⟨s', hpre, hinv⟩ := llama2_preLoop Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
    max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd stride_kbs
    stride_kh stride_kd att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N s hundef
  -- the prelude post-state s' satisfies the register invariant; extract the pieces we need
  have hpids : s'.pids = s.pids := hinv.1
  have hmem : s'.mem = s.mem := hinv.2.1
  have hmask : s'.regs .nat [] "block_mask" =
      some (Tile.scalar (if blockActive s B_Seqlen BLOCK_N then 1 else 0)) :=
    hinv.2.2.2.2.2.2.2.2.2.2.2.2.2
  -- block_mask dynamic-loop bound
  have hstop : evalOp (Op.ref .nat [] "block_mask") s'
      = some (Tile.scalar (if blockActive s B_Seqlen BLOCK_N then 1 else 0)) := by
    rw [evalOp_ref]; exact hmask
  -- assemble exec = prelude ++ loop
  rw [exec, llama2_body_split, stepStmts.append_some hpre]
  rw [show stepStmts [Stmt.forRangeDyn "start_mark" (Op.constNat 0) (Op.ref .nat [] "block_mask")
        (Op.constNat 1) (llama2LoopBody Q K B_Loc Att_Out sm_scale stride_b_loc_b stride_b_loc_s
          stride_kbs stride_kh stride_kd att_stride_h att_stride_bs BLOCK_DMODEL BLOCK_N)] s'
      = stepStmt (Stmt.forRangeDyn "start_mark" (Op.constNat 0) (Op.ref .nat [] "block_mask")
          (Op.constNat 1) (llama2LoopBody Q K B_Loc Att_Out sm_scale stride_b_loc_b stride_b_loc_s
            stride_kbs stride_kh stride_kd att_stride_h att_stride_bs BLOCK_DMODEL BLOCK_N)) s' from by
    conv_lhs => unfold stepStmts
    cases hb : stepStmt _ s' <;> simp [hb]]
  rw [stepForRangeAux.forRangeDyn_unfold]
  simp only [evalOp_constNat, hstop, Tile.scalar_data, Option.bind_some]
  by_cases hba : blockActive s B_Seqlen BLOCK_N
  · -- active block: block_mask = 1, one iteration runs the body
    rw [if_pos hba]
    rw [stepForRangeAux.step_one_iter (by norm_num) (by norm_num) (by norm_num)]
    obtain ⟨sfin, hbody, hread⟩ := llama2_loop_body_store Q K sm_scale B_Loc B_Start_Loc B_Seqlen
      Att_Out s max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N
      s' hinv hOutInj
    simp only [hbody]
    rw [hread i]
    -- closed form, active block branch
    simp only [tokenAttnLlama2ClosedForm, hba, true_and]
  · -- inactive block: block_mask = 0, no iteration, Att_Out preserved
    rw [if_neg hba]
    rw [stepForRangeAux.step_ge (by norm_num) (by norm_num)]
    -- s' preserves Att_Out (prelude writes no memory), so readback = original
    have hrm : s'.readMem Att_Out (outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)
        = s.readMem Att_Out (outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i) := by
      simp only [BlockState.readMem, hmem]
    simp only [hrm]
    simp only [tokenAttnLlama2ClosedForm, hba, false_and, if_false]

set_option maxHeartbeats 1000000 in
/-- **Compute-facing genuine closed-form correctness for `token_attn_llama2`.** The
full surface kernel realizes the genuine QK-dot closed form
`tokenAttnLlama2ClosedForm` at every active output lane of the masked store. -/
theorem token_attn_llama2_surface_output_compute_correct
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)) :
    ComputeCorrect.Realizes
      (kernel := token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
        B_Seqlen Att_Out max_input_len stride_b_loc_b stride_b_loc_s stride_qbs
        stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h
        att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => active s B_Seqlen max_input_len BLOCK_N i)
        (fun i => (Att_Out, outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)))
      (expected := fun i =>
        tokenAttnLlama2ClosedForm s Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
          max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
          stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
          BLOCK_DMODEL BLOCK_N i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_llama2_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  have h := token_attn_llama2_closed_form_correct Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
    max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd stride_kbs stride_kh
    stride_kd att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N s hundef hOutInj i
  rw [show exec _ s = some s' from hExec] at h
  exact h


/-! ### ════════ ★ MAIN THEOREM ★ ════════

**Dimension-general headline.** For *symbolic* head width `BLOCK_DMODEL`, block
size `BLOCK_N`, `sm_scale`, and *all* strides, the full Llama2 token-decode
QK-score surface kernel (1) lowers to the algorithm layer, and (2) realizes the
genuine, self-reference-free QK-dot closed form `tokenAttnLlama2ClosedForm` at
every active output lane of the masked store: each active lane
(`offs_n_new < max_input_len`) of an active `block_mask = 1` block holds the score
`sm_scale · Σ_d q[d]·k[i,d]` (with the `B_Loc` gather and varlen `start_index`
offset folded in), and an inactive lane (or any lane of an inactive block) is
preserved.

Honest side-conditions only: a clean `undef` state `hundef` and output-offset
injectivity `hOutInj`. The headline is dimension-general, so the concrete Python
test shapes are recovered as instances. -/
theorem token_attn_llama2_output_summary_general
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_N => outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)) :
    (∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen
      Att_Out max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh
      stride_qd stride_kbs stride_kh stride_kd att_stride_h att_stride_bs
      kv_group_num BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
        B_Seqlen Att_Out max_input_len stride_b_loc_b stride_b_loc_s stride_qbs
        stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h
        att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => active s B_Seqlen max_input_len BLOCK_N i)
        (fun i => (Att_Out, outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)))
      (expected := fun i : Fin BLOCK_N =>
        tokenAttnLlama2ClosedForm s Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
          max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
          stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
          BLOCK_DMODEL BLOCK_N i)) := by
  refine ⟨?_, ?_⟩
  · exact token_attn_llama2_surface_toAlgorithm_supported Q K sm_scale B_Loc
      B_Start_Loc B_Seqlen Att_Out max_input_len stride_b_loc_b stride_b_loc_s
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h
      att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N
  · exact token_attn_llama2_surface_output_compute_correct Q K sm_scale B_Loc
      B_Start_Loc B_Seqlen Att_Out max_input_len stride_b_loc_b stride_b_loc_s
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h
      att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N s hundef hOutInj

end VeriTile.Bench.TritonBenchG.TokenAttnLlama2
