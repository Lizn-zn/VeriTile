import VeriTile.Triton

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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
specification token_attn_llama2_output_summary_general
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
    (ComputeCorrect.Realizes_without_Rounding
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

section IOFace

open scoped VeriTile.Triton.StreamMetaGatherMasked3DKernelIO₂

/-! ## Slot table, step budget and IO signature

`token_attn_llama2` is a consumer of `StreamMetaGatherMasked3DKernelIO₂`, the
**gather-indexed** two-stream fold skin. The 3-D grid is
`(cur_batch, cur_head, start_n) = (pid₀, pid₁, pid₂)`; the **two** `.nat`
metadata slots are read at cell `cur_batch = pid₀` of their own regions, in the
kernel's own load order — slot `0` = `B_Seqlen[pid₀]` (`cur_batch_seq_len`, the
varlen right edge) and slot `1` = `B_Start_Loc[pid₀]`
(`cur_batch_in_all_start_index`, the `Att_Out` row base).

**Step budget.** The kernel's only loop is `range(0, block_mask, 1)` with
`block_mask ∈ {0, 1}`, so the pid-free upper bound of the walk is `T = 1`: the
single step `t = 0` is live exactly when `block_mask = 1`, and that gating lives
in `writeMask` (an inactive block writes nothing). No launch restriction is
needed, so `pre ≡ True`.

**Store genre.** The `tl.store` sits syntactically *inside* the loop, but the
loop runs at most once and the store is its only memory effect, so the kernel
still writes **one** window and the skin's terminal-store frame is honest: on an
active block the footprint is exactly `{write i | active i}`, and on an inactive
block `writeMask` is empty everywhere and nothing is written at all.

**The gather channel** is the paged-KV page table `B_Loc` at `gty = .nat` with
`gother = 0` (the kernel's own `other=0`): the step loads `BLOCK_N` page indices
masked by `offs_n_new i < max_input_len`, and those *values* address the `K`
load. Like the `token_attn_mistral` sibling — and unlike the softmax_reducev
exemplar — this port's `K` load carries **the same mask as the gather**
(`offs_n_new[:, None] < cur_batch_end_index`), so a dead lane never dereferences
the `other=` sentinel and no bound on `gother` is needed anywhere.

Stride abbreviations used throughout this section: `mil` = `max_input_len`,
`sblb`/`sbls` = `stride_b_loc_b`/`_s`, `sqbs`/`sqh`/`sqd` = the three `Q`
strides, `skbs`/`skh`/`skd` = the three `K` strides, `ash`/`asbs` =
`att_stride_h`/`att_stride_bs`, `kvg` = `kv_group_num`, `BD` = `BLOCK_DMODEL`,
`BN` = `BLOCK_N`. -/

/-- Slot-region table of the two per-batch metadata slots, in the kernel's own
load order (`B_Seqlen`, `B_Start_Loc`). A shared def, never an inline `match` in
a window position. -/
def llama2IOMetaBuf (B_Start_Loc B_Seqlen : Region .nat) : Fin 2 → RegionName
  | ⟨0, _⟩ => B_Seqlen.cast
  | ⟨_ + 1, _⟩ => B_Start_Loc.cast

/-- The kernel's single streaming step (`T = 1`). -/
def llama2IOStep : Fin 1 := ⟨0, by omega⟩

/-- **Gather-indexed IO signature** of `token_attn_llama2` on the gather-indexed
two-stream fold skin (S1: one-step QK fold + masked store, 3-D pid grid), at
fully **symbolic per-axis strides**.

Windows transcribe the kernel's pointer arithmetic VERBATIM, with the loaded
slot vector `m` in place of the in-state metadata reads:

* `gread` (`B_Loc`, the page table): lane `j` reads
  `sblb · pid₀ + sbls · ((mil − m 0) + (pid₂ · BN + j))`; `gmask` is
  `(mil − m 0) + (pid₂ · BN + j) < mil` and `gother = 0`.
* `read1` (`Q`, the query row): lane `d` of step `t` reads
  `pid₀ · sqbs + pid₁ · sqh + d · sqd + t` (the `+ start_mark` the kernel
  spells; at the single live step it is `+ 0`). The load is **unmasked**, so
  `mask1 ≡ True`.
* `read2` (`K`, the **gather-addressed** cache rows, lane `j = (jL, d)`
  row-major over `[BLOCK_N, BLOCK_DMODEL]`) reads
  `G t jL · skbs + (pid₁ / kvg) · skh + d · skd`. `mask2` repeats `gmask`'s
  predicate on the row coordinate.
* `write` (`Att_Out`): lane `i` writes `pid₁ · ash + (m 1 + (pid₂ · BN + i)) · asbs`,
  with `writeMask` = `block_mask = 1` **and** the store's own lane mask.

`outDType` is the `.real` default: the terminal `tl.store` lowers to
`Stmt.store .real` (no `.to(...)` cast anywhere in this kernel), so there is no
quantization event. -/
def llama2IO (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN : Nat) :
    StreamMetaGatherMasked3DKernelIO₂ where
  kernel := token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
    mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN
  inp1 := Q
  inp2 := K
  out := Att_Out
  nMeta := 2
  sty := fun _ => ChanTy.nat
  mbuf := llama2IOMetaBuf B_Start_Loc B_Seqlen
  mwin := fun _ pid₀ _ _ => pid₀
  gbuf := B_Loc.cast
  gty := ChanTy.nat
  Bg := BN
  gother := 0
  T := 1
  B1 := BD
  B2 := BN * BD
  C := BN
  outDType := .real
  pre := fun _ _ _ _ => True
  gread := fun pid₀ _ pid₂ m _ j =>
    sblb * pid₀ + sbls * ((mil - m (⟨0, by omega⟩ : Fin 2)) + (pid₂ * BN + j.val))
  gmask := fun _ _ pid₂ m _ j =>
    (mil - m (⟨0, by omega⟩ : Fin 2)) + (pid₂ * BN + j.val) < mil
  read1 := fun pid₀ pid₁ _ _ t d => pid₀ * sqbs + pid₁ * sqh + d.val * sqd + t.val
  mask1 := fun _ _ _ _ _ _ => True
  read2 := fun _ pid₁ _ _ G t j =>
    G t (Lane2D.decode j).1 * skbs + (pid₁ / kvg) * skh + (Lane2D.decode j).2.1.val * skd
  mask2 := fun _ _ pid₂ m _ j =>
    (mil - m (⟨0, by omega⟩ : Fin 2)) + (pid₂ * BN + (Lane2D.decode j).1.val) < mil
  write := fun _ pid₁ pid₂ m i =>
    pid₁ * ash + (m (⟨1, by omega⟩ : Fin 2) + (pid₂ * BN + i.val)) * asbs
  writeMask := fun _ _ pid₂ m i =>
    pid₂ * BN < m (⟨0, by omega⟩ : Fin 2) ∧
      (mil - m (⟨0, by omega⟩ : Fin 2)) + (pid₂ * BN + i.val) < mil

/-! ## The streamed closed form -/

/-- **The streamed closed form**: `tokenAttnLlama2DotScore` restated over the two
pinned streams — `att_out[i] = (Σ_d q[d] · k[i, d]) · sm_scale`, where the page
indirection lives in the *window* (`read2` eats `G`), so the second stream cell
is already the gathered cache row at flat lane `(i, d)`. -/
noncomputable def llama2IOSpec (BD BN : Nat) (sm_scale : ℝ)
    (xs : Fin 1 → Fin BD → ℝ) (ys : Fin 1 → Fin (BN * BD) → ℝ) (i : Fin BN) : ℝ :=
  (∑ d : Fin BD,
      xs llama2IOStep d *
        ys llama2IOStep
          (Lane2D.encode ((i, d, PUnit.unit) : TileIndex [BN, BD]))) * sm_scale

/-! ## The spec-equals-genuine bridge

The exact stack's genuine closed form names the kernel's own memory-side reads
(`qOffset` / `kOffset` — the latter **inlines the gather** `kLoc` inside the `K`
address). The io face states the same value over the pinned streams; the bridge
rewrites `kLoc ↦ G t j` through the gather pin, which is the one real proof
content of the gather skin. Because `mask2` repeats `gmask`'s predicate, only the
pin's **active** leg is ever used. -/

set_option maxHeartbeats 1000000 in
/-- **Streamed spec = genuine closed form.** Under the slot/gather/data pins the
exact headline's `tokenAttnLlama2DotScore` is the streamed `llama2IOSpec` at
every store-active lane. -/
theorem llama2IOSpec_eq_genuine (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Seqlen : RegionName)
    (mil sblb sbls sqbs sqh sqd skbs skh skd kvg BD BN : Nat)
    (s₀ : BlockState) (seq : Nat) (hseq : seqLen s₀ B_Seqlen = seq)
    (G : Fin 1 → Fin BN → Nat) (xs : Fin 1 → Fin BD → ℝ)
    (ys : Fin 1 → Fin (BN * BD) → ℝ)
    (hg : ∀ (t : Fin 1) (j : Fin BN),
      (mil - seq) + (s₀.pids 2 * BN + j.val) < mil →
      s₀.readMemValue .nat B_Loc
          (sblb * s₀.pids 0 + sbls * ((mil - seq) + (s₀.pids 2 * BN + j.val))) = G t j)
    (hx : ∀ (t : Fin 1) (d : Fin BD),
      s₀.readMem Q (s₀.pids 0 * sqbs + s₀.pids 1 * sqh + d.val * sqd + t.val) = xs t d)
    (hy : ∀ (t : Fin 1) (j : Fin (BN * BD)),
      (mil - seq) + (s₀.pids 2 * BN + (Lane2D.decode j).1.val) < mil →
      s₀.readMem K (G t (Lane2D.decode j).1 * skbs + (s₀.pids 1 / kvg) * skh
          + (Lane2D.decode j).2.1.val * skd) = ys t j)
    (i : Fin BN) (hact : active s₀ B_Seqlen mil BN i) :
    tokenAttnLlama2DotScore s₀ Q K sm_scale B_Loc B_Seqlen mil sblb sbls sqbs sqh sqd
        skbs skh skd kvg BD BN i
      = llama2IOSpec BD BN sm_scale xs ys i := by
  have hlive : (mil - seq) + (s₀.pids 2 * BN + i.val) < mil := by
    rw [← hseq]
    simpa only [active, blockOffset] using hact
  have hkloc : kLoc s₀ B_Loc B_Seqlen mil sblb sbls BN i = G llama2IOStep i := by
    rw [kLoc, hseq]
    exact hg llama2IOStep i hlive
  unfold tokenAttnLlama2DotScore llama2IOSpec
  refine congrArg (· * sm_scale) (Finset.sum_congr rfl (fun d _ => ?_))
  have hq : s₀.readMem Q (qOffset s₀ sqbs sqh sqd d.val) = xs llama2IOStep d := by
    rw [qOffset]
    exact hx llama2IOStep d
  have hk : s₀.readMem K (kOffset s₀ B_Loc B_Seqlen mil sblb sbls skbs skh skd kvg BN i d.val)
      = ys llama2IOStep (Lane2D.encode ((i, d, PUnit.unit) : TileIndex [BN, BD])) := by
    have h := hy llama2IOStep (Lane2D.encode ((i, d, PUnit.unit) : TileIndex [BN, BD]))
    rw [Lane2D.decode_encode] at h
    rw [kOffset, hkloc]
    exact h hlive
  rw [hq, hk]

/-! ## Flat-bridge coverage and cast-freedom

The port is cast-free in the `R` sense: every load/store is at `.real` or
`.nat`, there is no `.to(...)` cast, no `Op.castFloat`, no `ptrSub` and no block
pointer anywhere in the file. So `stepStmtsR R` collapses verbatim onto the exact
stepper on every segment, and no `R.round … = id` boundary hypothesis is
needed. -/

/-- The 13-statement prelude of `token_attn_llama2_surface`, transcribed
(`rfl`-equal to `…toAlgKernel.body.take 13`). -/
private def llama2IOPrelude (B_Start_Loc B_Seqlen : Region .nat)
    (mil sqbs sqh sqd kvg BD BN : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
    Stmt.assign .nat [] "cur_head" (Op.programId 1),
    Stmt.assign .nat [] "start_n" (Op.programId 2),
    Stmt.assign .nat [] "cur_kv_head"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat kvg)),
    Stmt.assign .nat [BD] "offs_d" (Op.arange BD),
    Stmt.assign .nat [] "cur_batch_seq_len"
      (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_in_all_start_index"
      (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_start_index"
      (Op.sub .nat Broadcast.nil (Op.constNat mil) (Op.ref .nat [] "cur_batch_seq_len")),
    Stmt.assign .nat [] "cur_batch_end_index" (Op.constNat mil),
    Stmt.assign .nat [BD] "off_q"
      (Op.add .nat Broadcast.scalarL
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sqbs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat sqh)))
        (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BD] "offs_d") (Op.constNat sqd))),
    Stmt.assign .nat [BN] "offs_n"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BN))
        (Op.arange BN)),
    Stmt.assign .nat [] "block_stard_index"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BN)),
    Stmt.assign .nat [] "block_mask"
      ((Op.lt .nat Broadcast.nil (Op.ref .nat [] "block_stard_index")
            (Op.ref .nat [] "cur_batch_seq_len")).where
        (Op.constNat 1) (Op.constNat 0)) ]

/-- Body decomposition against the literal prelude list. By `rfl`. -/
private theorem llama2IO_body_split (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN : Nat) :
    (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out mil sblb
        sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN).toAlgKernel.body
      = llama2IOPrelude B_Start_Loc B_Seqlen mil sqbs sqh sqd kvg BD BN
        ++ [Stmt.forRangeDyn "start_mark" (Op.constNat 0) (Op.ref .nat [] "block_mask")
              (Op.constNat 1)
              (llama2LoopBody Q K B_Loc Att_Out sm_scale sblb sbls skbs skh skd ash asbs
                BD BN)] := rfl

set_option maxHeartbeats 4000000 in
/-- The surface sits inside the flat-memory bridge's covered fragment. -/
private theorem llama2IO_flattenOk (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN : Nat) :
    ((token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out mil sblb
      sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [llama2IO_body_split]
  simp [llama2IOPrelude, llama2LoopBody, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-- Per-statement cast-free collapse lifts to statement lists (walks the actual
successor chain; a failing step collapses on both sides). Private copy of the
family helper (bench files never import each other). -/
private theorem llama2IO_stepStmtsR_castFree_of_stmts (R : RoundingModel) :
    ∀ (l : List Stmt), (∀ st ∈ l, ∀ u, stepStmtR R st u = stepStmt st u) →
      ∀ s, stepStmtsR R l s = stepStmts l s
  | [], _, s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, s => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self s]
      cases stepStmt st s with
      | none => rfl
      | some s' =>
          exact llama2IO_stepStmtsR_castFree_of_stmts R rest
            (fun st' h' u => h st' (List.mem_cons_of_mem _ h') u) s'

set_option maxHeartbeats 4000000 in
/-- Every prelude statement is cast-free (two `.nat` slot loads, register-only
`.nat` address arithmetic and the `block_mask` select). -/
private theorem llama2IO_prelude_stmt_castFree (R : RoundingModel)
    (B_Start_Loc B_Seqlen : Region .nat) (mil sqbs sqh sqd kvg BD BN : Nat) :
    ∀ st ∈ llama2IOPrelude B_Start_Loc B_Seqlen mil sqbs sqh sqd kvg BD BN,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [llama2IOPrelude, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 4000000 in
/-- Every loop-body statement is cast-free: the unmasked `.real` `Q` load, the
`.nat` page-table gather (`other = 0`), the gather-addressed `.real` `K` load
(`other = 0.0`), the `.real` dot reduce and scale, and the masked `.real` store
(`writeMemTypedR R .real` *is* `writeMemTyped .real`). -/
private theorem llama2IO_body_stmt_castFree (R : RoundingModel)
    (Q K B_Loc Att_Out : RegionName) (sm_scale : ℝ)
    (sblb sbls skbs skh skd ash asbs BD BN : Nat) :
    ∀ st ∈ llama2LoopBody Q K B_Loc Att_Out sm_scale sblb sbls skbs skh skd ash asbs BD BN,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [llama2LoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
      BlockState.writeMemTypedR]

/-- The loop body is cast-free as a list. -/
private theorem llama2IO_body_castFree (R : RoundingModel)
    (Q K B_Loc Att_Out : RegionName) (sm_scale : ℝ)
    (sblb sbls skbs skh skd ash asbs BD BN : Nat) (t : BlockState) :
    stepStmtsR R (llama2LoopBody Q K B_Loc Att_Out sm_scale sblb sbls skbs skh skd ash
        asbs BD BN) t
      = stepStmts (llama2LoopBody Q K B_Loc Att_Out sm_scale sblb sbls skbs skh skd ash
          asbs BD BN) t :=
  llama2IO_stepStmtsR_castFree_of_stmts R _
    (llama2IO_body_stmt_castFree R Q K B_Loc Att_Out sm_scale sblb sbls skbs skh skd ash
      asbs BD BN) t

/-- `evalOpR` of a `constNat` (R-independent). -/
private theorem llama2IO_evalOpR_constNat (R : RoundingModel) (n : Nat) (u : BlockState) :
    evalOpR R (Op.constNat n) u = some (Tile.scalar n) := by
  simp [evalOpR]

/-- `evalOpR` of the `forRangeDyn` stop expression (a bare `.nat` register read —
R-independent). -/
private theorem llama2IO_stopOpR_castFree (R : RoundingModel) (u : BlockState) :
    evalOpR R (Op.ref .nat [] "block_mask") u = evalOp (Op.ref .nat [] "block_mask") u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 1600000 in
/-- The guard `forRangeDyn` statement is cast-free per-state: its bound
expressions are two literals and a `.nat` register read, and its body is
cast-free, so `stepStmtR R` on the whole loop *is* `stepStmt`. -/
private theorem llama2IO_dyn_castFree (R : RoundingModel)
    (Q K B_Loc Att_Out : RegionName) (sm_scale : ℝ)
    (sblb sbls skbs skh skd ash asbs BD BN : Nat) :
    ∀ u, stepStmtR R (Stmt.forRangeDyn "start_mark" (Op.constNat 0)
        (Op.ref .nat [] "block_mask") (Op.constNat 1)
        (llama2LoopBody Q K B_Loc Att_Out sm_scale sblb sbls skbs skh skd ash asbs BD BN)) u
      = stepStmt (Stmt.forRangeDyn "start_mark" (Op.constNat 0)
          (Op.ref .nat [] "block_mask") (Op.constNat 1)
          (llama2LoopBody Q K B_Loc Att_Out sm_scale sblb sbls skbs skh skd ash asbs BD BN)) u := by
  intro u
  rw [stepForRangeAux.forRangeDyn_unfold]
  simp only [stepStmtR, llama2IO_evalOpR_constNat, llama2IO_stopOpR_castFree,
    evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
  cases hstop : evalOp (Op.ref .nat [] "block_mask") u with
  | none => rfl
  | some t =>
      simp only [Option.bind_some]
      exact stepForRangeAuxR_castFree R _
        (llama2IO_body_castFree R Q K B_Loc Att_Out sm_scale sblb sbls skbs skh skd ash
          asbs BD BN) "start_mark" _ _ _ u

/-- The whole lowered body is cast-free, statement by statement: `execR R` on the
surface *is* the exact `stepStmts` run. -/
private theorem llama2IO_execR_collapse (R : RoundingModel) (Q K : RegionName)
    (sm_scale : ℝ) (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN : Nat) (s : BlockState) :
    execR R ((token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out mil
        sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN).toAlgKernel) s
      = stepStmts ((token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen
          Att_Out mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN).toAlgKernel.body) s := by
  unfold execR
  rw [llama2IO_body_split]
  refine llama2IO_stepStmtsR_castFree_of_stmts R _ ?_ s
  intro st hst
  rcases List.mem_append.mp hst with hpre | hrest
  · exact llama2IO_prelude_stmt_castFree R B_Start_Loc B_Seqlen mil sqbs sqh sqd kvg BD BN
      st hpre
  · rcases List.mem_cons.mp hrest with rfl | hnil
    · exact llama2IO_dyn_castFree R Q K B_Loc Att_Out sm_scale sblb sbls skbs skh skd ash
        asbs BD BN
    · exact absurd hnil (List.not_mem_nil)

/-! ## The combined walk (`hts` + `hrun` in one pass)

Both skin obligations ride the *same* statement walk. To share it, the walk's
conclusion pairs the run (`∃ sF, stepStmtsR R … = some sF ∧ P sF`, which needs no
region bounds at all) with a **bounds-decoupled** safety clause
`∀ bounds, C bounds → TraceSafeListR R bounds …`: the bound hypotheses are
collected into one predicate `C` on `RegionBounds` so that `hrun` — which has no
allocator in scope — can use the run half and simply ignore the safety half. -/

/-- Combined walk cons: safety of the head under `C`, the R-step it actually
takes, and the pair (run, `C`-safety) of the tail from the successor give the
pair for the whole list. -/
private theorem llama2IO_walkCons {R : RoundingModel} {C : RegionBounds → Prop}
    {P : BlockState → Prop} {st : Stmt} {rest : List Stmt} {s s' : BlockState}
    (hsafe : ∀ bounds : RegionBounds, C bounds → Stmt.TraceSafeR R bounds st s)
    (hstep : stepStmtR R st s = some s')
    (h2 : (∃ sF, stepStmtsR R rest s' = some sF ∧ P sF)
      ∧ ∀ bounds : RegionBounds, C bounds → Stmt.TraceSafeListR R bounds rest s') :
    (∃ sF, stepStmtsR R (st :: rest) s = some sF ∧ P sF)
      ∧ ∀ bounds : RegionBounds, C bounds →
          Stmt.TraceSafeListR R bounds (st :: rest) s := by
  refine ⟨by rw [stepStmtsR_cons_some hstep]; exact h2.1, ?_⟩
  intro bounds hC
  refine Stmt.TraceSafeListR.cons_intro (hsafe bounds hC) (fun u hu => ?_)
  rw [hstep] at hu
  exact (Option.some.inj hu) ▸ h2.2 bounds hC

/-- Walk terminator: the empty tail runs to the current state and is safe. -/
private theorem llama2IO_walkNil {R : RoundingModel} {C : RegionBounds → Prop}
    {P : BlockState → Prop} (s : BlockState) (h : P s) :
    (∃ sF, stepStmtsR R [] s = some sF ∧ P sF)
      ∧ ∀ bounds : RegionBounds, C bounds → Stmt.TraceSafeListR R bounds [] s :=
  ⟨⟨s, by simp only [stepStmtsR], h⟩, fun _ _ => Stmt.TraceSafeListR.nil_intro⟩

/-- R-step of an assign whose op is cast-free: the two collapse into one
walk-ready equation. -/
private theorem llama2IO_stepR_of_assign {R : RoundingModel} {dt : TileDType}
    {sh : TileShape} {nm : RegName} {e : Op dt sh} {s : BlockState} {v : Tile dt sh}
    (hcf : evalOpR R e s = evalOp e s) (h : evalOp e s = some v) :
    stepStmtR R (.assign dt sh nm e) s = some (s.setReg nm dt sh v) :=
  stepStmtR_assign_eq_some (hcf.trans h)

/-- `readMem` depends on `mem` only (re-anchors memory-derived quantities at the
launch state across a `setReg` chain). -/
private theorem llama2IO_readMem_of_mem (u s : BlockState) (h : u.mem = s.mem)
    (r : RegionName) (a : Nat) : u.readMem r a = s.readMem r a := by
  simp only [BlockState.readMem, BlockState.readMemValue, BlockState.readMemTyped, h]

/-- `readMemValue` depends on `mem` only, at every channel dtype. -/
private theorem llama2IO_readMemValue_of_mem (u s : BlockState) (h : u.mem = s.mem)
    (dt : TileDType) (r : RegionName) (a : Nat) :
    u.readMemValue dt r a = s.readMemValue dt r a := by
  cases dt <;>
    simp only [BlockState.readMemValue, BlockState.readMemAs, BlockState.readMemTyped,
      BlockState.readMem, h]

/-- Weak (walk) invariant: exact pins for every address-bearing register, all
anchored to the launch state `s` through `readMemValue`, so they survive the
whole run. This is `llama2Invariant` with the clean-`undef` conjunct dropped (the
skin's `hts` runs from an arbitrary launch state) and the unused
`cur_batch_seq_len` pin omitted. -/
private def llama2IOInvW (B_Start_Loc B_Seqlen : RegionName)
    (mil sqbs sqh sqd kvg BD BN : Nat) (s u : BlockState) : Prop :=
  u.mem = s.mem
  ∧ u.pids = s.pids
  ∧ u.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 0))
  ∧ u.regs .nat [] "cur_head" = some (Tile.scalar (s.pids 1))
  ∧ u.regs .nat [] "cur_kv_head" = some (Tile.scalar (s.pids 1 / kvg))
  ∧ u.regs .nat [BD] "offs_d" = some (Tile.vec (fun e : Fin BD => e.val))
  ∧ u.regs .nat [] "cur_batch_start_index"
      = some (Tile.scalar (mil - seqLen s B_Seqlen))
  ∧ u.regs .nat [] "cur_batch_end_index" = some (Tile.scalar mil)
  ∧ u.regs .nat [] "cur_batch_in_all_start_index"
      = some (Tile.scalar (startLoc s B_Start_Loc))
  ∧ u.regs .nat [BD] "off_q"
      = some (Tile.vec (fun d : Fin BD => qOffset s sqbs sqh sqd d.val))
  ∧ u.regs .nat [BN] "offs_n"
      = some (Tile.vec (fun j : Fin BN => s.pids 2 * BN + j.val))
  ∧ u.regs .nat [] "block_mask"
      = some (Tile.scalar (if blockActive s B_Seqlen BN then 1 else 0))

/-- The prelude's bound obligations: the two `.nat` metadata slots, read at cell
`cur_batch = pid₀` of their own regions. -/
private def llama2IOPreBounds (B_Start_Loc B_Seqlen : RegionName) (s : BlockState)
    (bounds : RegionBounds) : Prop :=
  s.pids 0 < bounds B_Seqlen ∧ s.pids 0 < bounds B_Start_Loc

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **Prelude walk** (single pass): from an **arbitrary** launch state the 13
prelude statements step to a state satisfying `llama2IOInvW`, and are trace-safe
whenever the two slot cells are in bounds. -/
private theorem llama2IO_preLoopW (R : RoundingModel)
    (B_Start_Loc B_Seqlen : Region .nat) (mil sqbs sqh sqd kvg BD BN : Nat)
    (s : BlockState) :
    (∃ s0, stepStmtsR R (llama2IOPrelude B_Start_Loc B_Seqlen mil sqbs sqh sqd kvg BD BN) s
        = some s0
      ∧ llama2IOInvW B_Start_Loc.cast B_Seqlen.cast mil sqbs sqh sqd kvg BD BN s s0)
    ∧ ∀ bounds : RegionBounds,
        llama2IOPreBounds B_Start_Loc.cast B_Seqlen.cast s bounds →
        Stmt.TraceSafeListR R bounds
          (llama2IOPrelude B_Start_Loc B_Seqlen mil sqbs sqh sqd kvg BD BN) s := by
  unfold llama2IOPrelude
  -- stmt 0: cur_batch = programId 0
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (evalOp_programId 0 s)) ?_
  -- stmt 1: cur_head = programId 1
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (evalOp_programId 1 _)) ?_
  -- stmt 2: start_n = programId 2
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (evalOp_programId 2 _)) ?_
  -- stmt 3: cur_kv_head = cur_head // kv_group_num
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head")
            (Op.constNat kvg)) _
          = some (Tile.scalar (s.pids 1 / kvg)) from by
        simp only [evalOp, evalOp_ref, evalOp_constNat, BlockState.setReg_same,
          BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind]
        refine congrArg some ?_
        ext idx
        simp only [Tile.bop, Tile.scalar, BlockState.setReg_pids, IntegralDType.nat_floorDiv,
          castTile_self, Tile.scalar_data_index, Broadcast.leftIndex,
          Broadcast.rightIndex])) ?_
  -- stmt 4: offs_d = arange BD
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.arange BD) _ = some (Tile.vec (fun e : Fin BD => e.val)) from
        evalOp_arange _ _)) ?_
  -- stmt 5: cur_batch_seq_len = load(B_Seqlen + cur_batch)   [slot 0]
  refine llama2IO_walkCons ?_
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch"))
            MaskOpt.none) _
          = some (Tile.scalar (seqLen s B_Seqlen.cast)) from by
        simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind, Option.pure_def]
        rfl)) ?_
  · intro bounds hC
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq, BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hC.1
  -- stmt 6: cur_batch_in_all_start_index = load(B_Start_Loc + cur_batch)   [slot 1]
  refine llama2IO_walkCons ?_
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch"))
            MaskOpt.none) _
          = some (Tile.scalar (startLoc s B_Start_Loc.cast)) from by
        simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind, Option.pure_def]
        rfl)) ?_
  · intro bounds hC
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [MaskOpt.SafeAtR], ?_⟩
    intro offsets hoff idx _
    rw [evalOpR_ref] at hoff
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      reduceCtorEq, BlockState.setReg_same] at hoff
    obtain rfl := Option.some.inj hoff
    exact hC.2
  -- stmt 7: cur_batch_start_index = max_input_len - cur_batch_seq_len
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.sub .nat Broadcast.nil (Op.constNat mil)
            (Op.ref .nat [] "cur_batch_seq_len")) _
          = some (Tile.scalar (mil - seqLen s B_Seqlen.cast)) from by
        simp only [evalOp_sub, evalOp_constNat, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, Option.bind]
        refine congrArg some ?_
        ext idx
        simp only [Tile.bop, Tile.scalar, NumericDType.sub, Broadcast.leftIndex,
          Broadcast.rightIndex, castTile_self, Tile.scalar_data,
          Tile.scalar_data_index])) ?_
  -- stmt 8: cur_batch_end_index = max_input_len
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.constNat mil) _ = some (Tile.scalar mil) from
        evalOp_constNat _ _)) ?_
  -- stmt 9: off_q
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.add .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_batch") (Op.constNat sqbs))
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat sqh)))
            (Op.mul .nat Broadcast.scalarR (Op.ref .nat [BD] "offs_d")
              (Op.constNat sqd))) _
          = some (Tile.vec (fun d : Fin BD => qOffset s sqbs sqh sqd d.val)) from by
        rw [llama2_offq_eval _ BD (s.pids 0) (s.pids 1) sqbs sqh sqd
          (by simp) (by simp) (by simp [Tile.vec])]
        refine congrArg some ?_
        ext idx
        simp [Tile.vec, qOffset])) ?_
  -- stmt 10: offs_n
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n") (Op.constNat BN))
            (Op.arange BN)) _
          = some (Tile.vec (fun j : Fin BN => s.pids 2 * BN + j.val)) from by
        rw [llama2_offsn_eval _ BN (s.pids 2) BN (by simp)])) ?_
  -- stmt 11: block_stard_index
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_n")
            (Op.constNat BN)) _
          = some (Tile.scalar (s.pids 2 * BN)) from by
        simp only [evalOp_mul, evalOp_constNat, evalOp_ref, BlockState.setReg_same,
          BlockState.setReg_ne_name, BlockState.setReg_pids, Option.bind]
        refine congrArg some ?_
        ext idx
        simp only [Tile.bop, Tile.scalar, NumericDType.mul, Broadcast.leftIndex,
          Broadcast.rightIndex, BlockState.setReg_pids, castTile_self,
          Tile.scalar_data_index])) ?_
  -- stmt 12: block_mask
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp ((Op.lt .nat Broadcast.nil (Op.ref .nat [] "block_stard_index")
                (Op.ref .nat [] "cur_batch_seq_len")).where (Op.constNat 1)
              (Op.constNat 0)) _
          = some (Tile.scalar (if blockActive s B_Seqlen.cast BN then 1 else 0)) from by
        simp only [evalOp_where, evalOp_lt, evalOp_constNat, evalOp_ref,
          BlockState.setReg_same, BlockState.setReg_ne_name, Option.bind]
        refine congrArg some ?_
        ext idx
        simp only [Tile.select_data, Tile.cop_data, Tile.scalar, Broadcast.leftIndex,
          Broadcast.rightIndex, ComparableDType.lt, blockActive, seqLen]
        by_cases h : s.pids 2 * BN < s.readMemValue .nat B_Seqlen.cast (s.pids 0)
        · rw [if_pos (by simpa using h), if_pos h]
        · rw [if_neg (by simpa using h), if_neg h])) ?_
  refine llama2IO_walkNil _ ?_
  refine ⟨by funext rg o; simp, by simp, by simp, by simp, by simp, by simp [Tile.vec],
    by simp, by simp, by simp, by simp [Tile.vec], by simp [Tile.vec], by simp⟩

/-! ### `evalOpR` decoders and memory-transparent load recipes

The address/mask trees of the loop body, decoded at pinned registers. Each
memory-touching load gets a **rewrite-based** recipe carrying
`hmem : u.mem = s.mem` (never a `show … = some <load>` through the `setReg`
tower), so the whole walk speaks in the launch state's vocabulary. -/

/-- A register write is transparent to `readMem`. -/
private theorem llama2IO_readMem_setReg (u : BlockState) {dt : TileDType}
    {sh : TileShape} (nm : RegName) (t : Tile dt sh) (r : RegionName) (a : Nat) :
    (u.setReg nm dt sh t).readMem r a = u.readMem r a :=
  llama2IO_readMem_of_mem _ u rfl r a

/-- A register write is transparent to `readMemValue`. -/
private theorem llama2IO_readMemValue_setReg (u : BlockState) {dt : TileDType}
    {sh : TileShape} (nm : RegName) (t : Tile dt sh) (d : TileDType) (r : RegionName)
    (a : Nat) :
    (u.setReg nm dt sh t).readMemValue d r a = u.readMemValue d r a :=
  llama2IO_readMemValue_of_mem _ u rfl d r a

set_option maxHeartbeats 1600000 in
/-- **`q` load recipe under `R`** (unmasked `.real` load at `off_q + start_mark`),
anchored at the launch state through `hmem`. -/
private theorem llama2IO_qLoadR_eval (R : RoundingModel) (Q : RegionName) (BD c : Nat)
    (u s : BlockState) (hmem : u.mem = s.mem) (offq : Fin BD → Nat) (qv : Fin BD → ℝ)
    (hoffq : u.regs .nat [BD] "off_q" = some (Tile.vec offq))
    (hsm : u.regs .nat [] "start_mark" = some (Tile.scalar c))
    (hqv : ∀ d : Fin BD, s.readMem Q (offq d + c) = qv d) :
    evalOpR R (Op.load .real
        (MemAccess.region Q
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BD] "off_q")
            (Op.ref .nat [] "start_mark"))) MaskOpt.none) u
      = some (⟨fun idx : TileIndex [BD] => some (qv idx.1)⟩ : Tile .real [BD]) := by
  rw [show evalOpR R (Op.load .real
        (MemAccess.region Q
          (Op.add .nat Broadcast.scalarR (Op.ref .nat [BD] "off_q")
            (Op.ref .nat [] "start_mark"))) MaskOpt.none) u
      = evalOp (Op.load .real
          (MemAccess.region Q
            (Op.add .nat Broadcast.scalarR (Op.ref .nat [BD] "off_q")
              (Op.ref .nat [] "start_mark"))) MaskOpt.none) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  rw [llama2_q_load_eval u Q BD c offq hoffq hsm]
  refine congrArg some ?_
  ext idx
  simp only [llama2IO_readMem_of_mem u s hmem, hqv]

set_option maxHeartbeats 1600000 in
/-- **`k_loc` gather recipe under `R`**, anchored at the launch state. -/
private theorem llama2IO_klocR_eval (R : RoundingModel) (B_Loc B_Seqlen : RegionName)
    (BN mil sblb sbls : Nat) (u s : BlockState) (hmem : u.mem = s.mem)
    (hpids : u.pids = s.pids) (onn : Fin BN → Nat) (kl : Fin BN → Nat)
    (honn : u.regs .nat [BN] "offs_n_new" = some (Tile.vec onn))
    (hcb : u.regs .nat [] "cur_batch" = some (Tile.scalar (s.pids 0)))
    (hend : u.regs .nat [] "cur_batch_end_index" = some (Tile.scalar mil))
    (hkl : ∀ j : Fin BN,
      (if onn j < mil then s.readMemValue .nat B_Loc (sblb * s.pids 0 + sbls * onn j)
        else 0) = kl j) :
    evalOpR R (Op.load .nat
        (MemAccess.region B_Loc
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.constNat sblb) (Op.ref .nat [] "cur_batch"))
            (Op.mul .nat Broadcast.scalarL (Op.constNat sbls)
              (Op.ref .nat [BN] "offs_n_new"))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_n_new")
            (Op.ref .nat [] "cur_batch_end_index"))
          ((Op.constNat 0).broadcast [BN]))) u
      = some (Tile.vec kl) := by
  rw [show evalOpR R (Op.load .nat
        (MemAccess.region B_Loc
          (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.constNat sblb) (Op.ref .nat [] "cur_batch"))
            (Op.mul .nat Broadcast.scalarL (Op.constNat sbls)
              (Op.ref .nat [BN] "offs_n_new"))))
        (MaskOpt.maskOther
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_n_new")
            (Op.ref .nat [] "cur_batch_end_index"))
          ((Op.constNat 0).broadcast [BN]))) u
      = evalOp (Op.load .nat
          (MemAccess.region B_Loc
            (Op.add .nat Broadcast.scalarL
              (Op.mul .nat Broadcast.nil (Op.constNat sblb) (Op.ref .nat [] "cur_batch"))
              (Op.mul .nat Broadcast.scalarL (Op.constNat sbls)
                (Op.ref .nat [BN] "offs_n_new"))))
          (MaskOpt.maskOther
            (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_n_new")
              (Op.ref .nat [] "cur_batch_end_index"))
            ((Op.constNat 0).broadcast [BN]))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  rw [llama2_kloc_gather_eval u B_Loc B_Seqlen BN mil sblb sbls onn honn
    (by rw [hpids]; exact hcb) hend]
  refine congrArg some ?_
  ext idx
  simp only [Tile.vec, hpids, llama2IO_readMemValue_of_mem u s hmem, hkl]

set_option maxHeartbeats 1600000 in
/-- **`k` gather recipe under `R`** (the gather-addressed `.real` load, masked by
the gather's own predicate), anchored at the launch state. -/
private theorem llama2IO_kLoadR_eval (R : RoundingModel) (K : RegionName)
    (BN BD mil : Nat) (u s : BlockState) (hmem : u.mem = s.mem)
    (offk : Fin BN → Fin BD → Nat) (onn : Fin BN → Nat) (kv : Fin BN → Fin BD → ℝ)
    (hoffk : u.regs .nat [BN, BD] "off_k" =
      some (⟨fun idx : TileIndex [BN, BD] => offk idx.1 idx.2.1⟩ : Tile .nat [BN, BD]))
    (honn : u.regs .nat [BN] "offs_n_new" = some (Tile.vec onn))
    (hend : u.regs .nat [] "cur_batch_end_index" = some (Tile.scalar mil))
    (hkv : ∀ (j : Fin BN) (d : Fin BD),
      (if onn j < mil then s.readMem K (offk j d) else 0) = kv j d) :
    evalOpR R (Op.load .real (MemAccess.region K (Op.ref .nat [BN, BD] "off_k"))
        (MaskOpt.maskOther
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n_new"))
              (Op.ref .nat [] "cur_batch_end_index")))
          ((Op.const 0.0).broadcast [BN, BD]))) u
      = some (⟨fun idx : TileIndex [BN, BD] => some (kv idx.1 idx.2.1)⟩
          : Tile .real [BN, BD]) := by
  rw [show evalOpR R (Op.load .real (MemAccess.region K (Op.ref .nat [BN, BD] "off_k"))
        (MaskOpt.maskOther
          (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n_new"))
              (Op.ref .nat [] "cur_batch_end_index")))
          ((Op.const 0.0).broadcast [BN, BD]))) u
      = evalOp (Op.load .real (MemAccess.region K (Op.ref .nat [BN, BD] "off_k"))
          (MaskOpt.maskOther
            (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
              (Op.lt .nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n_new"))
                (Op.ref .nat [] "cur_batch_end_index")))
            ((Op.const 0.0).broadcast [BN, BD]))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  rw [llama2_k_gather_eval u K BN BD mil offk onn hoffk honn hend]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, d, uu⟩ := idx
  simp only [llama2IO_readMem_of_mem u s hmem, hkv]

set_option maxHeartbeats 1600000 in
/-- The masked-store step under `R`: the `off_o` scatter over the store's own
lane mask. -/
private theorem llama2IO_storeR_eval (R : RoundingModel) (Att_Out : RegionName)
    (BN mil : Nat) (u : BlockState) (ooF : Fin BN → Nat) (av : Fin BN → ℝ)
    (onn : Fin BN → Nat)
    (hoffo : u.regs .nat [BN] "off_o" = some (Tile.vec ooF))
    (hav : u.regs .real [BN] "att_value" =
      some (⟨fun idx : TileIndex [BN] => some (av idx.1)⟩ : Tile .real [BN]))
    (honn : u.regs .nat [BN] "offs_n_new" = some (Tile.vec onn))
    (hend : u.regs .nat [] "cur_batch_end_index" = some (Tile.scalar mil)) :
    stepStmtR R (Stmt.store .real [BN] (MemAccess.region Att_Out (Op.ref .nat [BN] "off_o"))
        (Op.ref .real [BN] "att_value")
        (MaskOpt.mask
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_n_new")
            (Op.ref .nat [] "cur_batch_end_index")))) u
      = some ((TileShape.allIndices [BN]).foldl
          (fun acc k => if onn k.1 < mil then acc.writeMem Att_Out (ooF k.1) (av k.1)
            else acc) u) := by
  rw [show stepStmtR R (Stmt.store .real [BN]
        (MemAccess.region Att_Out (Op.ref .nat [BN] "off_o"))
        (Op.ref .real [BN] "att_value")
        (MaskOpt.mask
          (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_n_new")
            (Op.ref .nat [] "cur_batch_end_index")))) u
      = stepStmt (Stmt.store .real [BN]
          (MemAccess.region Att_Out (Op.ref .nat [BN] "off_o"))
          (Op.ref .real [BN] "att_value")
          (MaskOpt.mask
            (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_n_new")
              (Op.ref .nat [] "cur_batch_end_index")))) u from by
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def,
      BlockState.writeMemTypedR]]
  simp only [stepStmt, evalOp_ref, hoffo, hav, evalOp_lt, honn, hend, Option.bind_some,
    Option.bind_eq_bind, Option.map_some, Option.some.injEq, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, Tile.cop_data, Tile.scalar_data,
    Tile.vec, decide_eq_true_eq]
  rfl

/-- A masked `writeMem` scatter `foldl` leaves every cell not hit by an *active*
lane untouched. -/
private theorem llama2IO_foldl_writeMem_frame {α : Type} (region : RegionName)
    (offFn : α → Nat) (valFn : α → ℝ) (Pb : α → Prop) [DecidablePred Pb] :
    ∀ (l : List α) (st : BlockState) (r : RegionName) (o : Nat),
      (r = region → ∀ k ∈ l, Pb k → offFn k ≠ o) →
      ((l.foldl (fun acc k => if Pb k then acc.writeMem region (offFn k) (valFn k) else acc)
          st).mem r o = st.mem r o)
  | [], _, _, _, _ => rfl
  | k :: rest, st, r, o, h => by
      rw [List.foldl_cons]
      by_cases hk : Pb k
      · rw [if_pos hk,
          llama2IO_foldl_writeMem_frame region offFn valFn Pb rest _ r o
            (fun hr k' hk' => h hr k' (List.mem_cons_of_mem _ hk')),
          BlockState.writeMem_mem,
          if_neg (fun hro => h hro.1 k List.mem_cons_self hk hro.2.symm)]
      · rw [if_neg hk]
        exact llama2IO_foldl_writeMem_frame region offFn valFn Pb rest _ r o
          (fun hr k' hk' => h hr k' (List.mem_cons_of_mem _ hk'))

/-- The loop body's bound obligations, in the launch state's vocabulary: the
unmasked `Q` row, the page-table gather and the gather-addressed `K` rows on the
gather's own window, and the store's active lanes. -/
private def llama2IOBounds (Q K B_Loc B_Start_Loc B_Seqlen Att_Out : RegionName)
    (mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN : Nat)
    (s : BlockState) (bounds : RegionBounds) : Prop :=
  (∀ d : Fin BD, qOffset s sqbs sqh sqd d.val < bounds Q)
  ∧ (∀ i : Fin BN, active s B_Seqlen mil BN i →
      sblb * s.pids 0 + sbls * (mil - seqLen s B_Seqlen + (s.pids 2 * BN + i.val))
        < bounds B_Loc)
  ∧ (∀ (i : Fin BN) (d : Fin BD), active s B_Seqlen mil BN i →
      kOffset s B_Loc B_Seqlen mil sblb sbls skbs skh skd kvg BN i d.val < bounds K)
  ∧ (∀ i : Fin BN, active s B_Seqlen mil BN i →
      outOffset s B_Start_Loc ash asbs BN i < bounds Att_Out)

set_option maxHeartbeats 1600000 in
/-- The `q` load's address tree `off_q + start_mark`. -/
private theorem llama2IO_qAddrR_eval (R : RoundingModel) {BD : Nat} (u : BlockState)
    (offq : Fin BD → Nat) (c : Nat)
    (hoffq : u.regs .nat [BD] "off_q" = some (Tile.vec offq))
    (hsm : u.regs .nat [] "start_mark" = some (Tile.scalar c)) :
    evalOpR R (Op.add .nat Broadcast.scalarR (Op.ref .nat [BD] "off_q")
        (Op.ref .nat [] "start_mark")) u
      = some (⟨fun idx : TileIndex [BD] => offq idx.1 + c⟩ : Tile .nat [BD]) := by
  rw [show evalOpR R (Op.add .nat Broadcast.scalarR (Op.ref .nat [BD] "off_q")
        (Op.ref .nat [] "start_mark")) u
      = evalOp (Op.add .nat Broadcast.scalarR (Op.ref .nat [BD] "off_q")
          (Op.ref .nat [] "start_mark")) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp_add, evalOp_ref, hoffq, hsm, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add]

set_option maxHeartbeats 1600000 in
/-- The page-table gather's address tree
`sblb · cur_batch + sbls · offs_n_new`. -/
private theorem llama2IO_gAddrR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (onn : Fin BN → Nat) (cb sblb sbls : Nat)
    (hcb : u.regs .nat [] "cur_batch" = some (Tile.scalar cb))
    (honn : u.regs .nat [BN] "offs_n_new" = some (Tile.vec onn)) :
    evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.constNat sblb) (Op.ref .nat [] "cur_batch"))
        (Op.mul .nat Broadcast.scalarL (Op.constNat sbls)
          (Op.ref .nat [BN] "offs_n_new"))) u
      = some (⟨fun idx : TileIndex [BN] => sblb * cb + sbls * onn idx.1⟩
          : Tile .nat [BN]) := by
  rw [show evalOpR R (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.constNat sblb) (Op.ref .nat [] "cur_batch"))
        (Op.mul .nat Broadcast.scalarL (Op.constNat sbls)
          (Op.ref .nat [BN] "offs_n_new"))) u
      = evalOp (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.constNat sblb) (Op.ref .nat [] "cur_batch"))
          (Op.mul .nat Broadcast.scalarL (Op.constNat sbls)
            (Op.ref .nat [BN] "offs_n_new"))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, hcb, honn,
    Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 1600000 in
/-- The shared window mask `offs_n_new < cur_batch_end_index` — the predicate the
gather, the gather-addressed `K` load and the store all carry. -/
private theorem llama2IO_maskR_eval (R : RoundingModel) {BN : Nat} (u : BlockState)
    (onn : Fin BN → Nat) (mil : Nat)
    (honn : u.regs .nat [BN] "offs_n_new" = some (Tile.vec onn))
    (hend : u.regs .nat [] "cur_batch_end_index" = some (Tile.scalar mil)) :
    evalOpR R (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_n_new")
        (Op.ref .nat [] "cur_batch_end_index")) u
      = some (⟨fun idx : TileIndex [BN] => decide (onn idx.1 < mil)⟩
          : Tile .bool [BN]) := by
  rw [show evalOpR R (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_n_new")
        (Op.ref .nat [] "cur_batch_end_index")) u
      = evalOp (Op.lt .nat Broadcast.scalarR (Op.ref .nat [BN] "offs_n_new")
          (Op.ref .nat [] "cur_batch_end_index")) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  simp only [evalOp, honn, hend, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  simp only [Tile.cop_data, Tile.vec, Tile.scalar, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex]
  rfl

set_option maxHeartbeats 1600000 in
/-- The gather-addressed `K` load's mask: the `[BN, 1]` window test remapped
across the head-dim columns. It is the **same predicate** as the gather's mask,
which is why no dead lane ever dereferences a sentinel address. -/
private theorem llama2IO_kMaskR_eval (R : RoundingModel) (BN BD : Nat) (u : BlockState)
    (onn : Fin BN → Nat) (mil : Nat)
    (honn : u.regs .nat [BN] "offs_n_new" = some (Tile.vec onn))
    (hend : u.regs .nat [] "cur_batch_end_index" = some (Tile.scalar mil)) :
    evalOpR R (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n_new"))
          (Op.ref .nat [] "cur_batch_end_index"))) u
      = some (⟨fun idx : TileIndex [BN, BD] => decide (onn idx.1 < mil)⟩
          : Tile .bool [BN, BD]) := by
  rw [show evalOpR R (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt .nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n_new"))
          (Op.ref .nat [] "cur_batch_end_index"))) u
      = evalOp (Op.remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
          (Op.lt .nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n_new"))
            (Op.ref .nat [] "cur_batch_end_index"))) u from by
    simp only [evalOpR.eq_def, evalOp.eq_def]]
  have hexpn : @evalOp .nat [BN, 1]
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BN] "offs_n_new")) u
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec onn)) :=
    evalOp_expandDim_ref_of_regs .nat [BN] ⟨1, by simp⟩ "offs_n_new" u _ honn
  simp only [evalOp, hexpn, hend, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨j, d, uu⟩ := idx
  simp only [Tile.remap, Tile.cop_data, Tile.vec, Tile.scalar, Tile.expandDim_data,
    TileShape.dropInsertedIndex_succ, TileShape.dropInsertedIndex_nil,
    TileShape.dropInsertedIndex_zero_cons, ComparableDType.lt, Broadcast.leftIndex,
    Broadcast.rightIndex]
  rfl

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 8000 in
/-- **The loop-body walk** (single pass, from an `llama2IOInvW` state at counter
`start_mark = 0`): the 9 body statements step to a state whose `Att_Out` readback
is the genuine dot score on every store-active lane and the original cell
elsewhere, whose memory is framed outside the store window, and which is
trace-safe whenever `llama2IOBounds` holds. -/
private theorem llama2IO_bodyW (R : RoundingModel) (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen Att_Out : RegionName)
    (mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN : Nat)
    (s stt : BlockState)
    (hP : llama2IOInvW B_Start_Loc B_Seqlen mil sqbs sqh sqd kvg BD BN s stt)
    (hOutInj : Function.Injective
      (fun i : Fin BN => outOffset s B_Start_Loc ash asbs BN i)) :
    (∃ sF, stepStmtsR R (llama2LoopBody Q K B_Loc Att_Out sm_scale sblb sbls skbs skh skd
          ash asbs BD BN) (stt.setReg "start_mark" .nat [] (Tile.scalar 0)) = some sF
      ∧ (∀ i : Fin BN,
          sF.readMem Att_Out (outOffset s B_Start_Loc ash asbs BN i)
            = if active s B_Seqlen mil BN i then
                tokenAttnLlama2DotScore s Q K sm_scale B_Loc B_Seqlen mil sblb sbls sqbs
                  sqh sqd skbs skh skd kvg BD BN i
              else s.readMem Att_Out (outOffset s B_Start_Loc ash asbs BN i))
      ∧ (∀ r o, (r ≠ Att_Out ∨ ∀ i : Fin BN, active s B_Seqlen mil BN i →
            o ≠ outOffset s B_Start_Loc ash asbs BN i) → sF.mem r o = s.mem r o))
    ∧ ∀ bounds : RegionBounds,
        llama2IOBounds Q K B_Loc B_Start_Loc B_Seqlen Att_Out mil sblb sbls sqbs sqh sqd
          skbs skh skd ash asbs kvg BD BN s bounds →
        Stmt.TraceSafeListR R bounds (llama2LoopBody Q K B_Loc Att_Out sm_scale sblb sbls
            skbs skh skd ash asbs BD BN)
          (stt.setReg "start_mark" .nat [] (Tile.scalar 0)) := by
  obtain ⟨hmem, hpids, hcb, hch, hckv, hd, hsi, hend, hSL, hoffq, hoffn, _hmask⟩ := hP
  set onn : Fin BN → Nat :=
    fun j => (mil - seqLen s B_Seqlen) + (s.pids 2 * BN + j.val) with honnDef
  set kloc : Fin BN → Nat :=
    fun j => if onn j < mil then
        s.readMemValue .nat B_Loc (sblb * s.pids 0 + sbls * onn j) else 0 with klocDef
  set offk : Fin BN → Fin BD → Nat :=
    fun j e => kloc j * skbs + (s.pids 1 / kvg) * skh + e.val * skd with offkDef
  set kval : Fin BN → Fin BD → ℝ :=
    fun j e => if onn j < mil then s.readMem K (offk j e) else 0 with kvalDef
  set qval : Fin BD → ℝ :=
    fun e => s.readMem Q (qOffset s sqbs sqh sqd e.val) with qvalDef
  set avraw : Fin BN → ℝ :=
    fun j => ∑ e : Fin BD, qval e * kval j e with avrawDef
  -- the shared window predicate, in both vocabularies
  have hactIff : ∀ j : Fin BN, active s B_Seqlen mil BN j ↔ onn j < mil := by
    intro j
    simp only [active, blockOffset, honnDef]
  unfold llama2LoopBody
  -- stmt 0: q = tl.load(Q + off_q + start_mark)   [read1 window, unmasked]
  refine llama2IO_walkCons ?_
    (stepStmtR_assign_eq_some
      (llama2IO_qLoadR_eval R Q BD 0 _ s (by exact hmem)
        (fun e : Fin BD => qOffset s sqbs sqh sqd e.val) qval
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hoffq)
        (by simp only [BlockState.setReg_same])
        (fun e => by simp only [qvalDef, Nat.add_zero]))) ?_
  · intro bounds hC
    obtain ⟨hbQ, _, _, _⟩ := hC
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], trivial, ?_⟩
    intro offsets hoffs idx _
    rw [llama2IO_qAddrR_eval R _ (fun e : Fin BD => qOffset s sqbs sqh sqd e.val) 0
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hoffq)
      (by simp only [BlockState.setReg_same])] at hoffs
    obtain rfl := Option.some.inj hoffs
    obtain ⟨e, uu⟩ := idx
    simpa using hbQ e
  -- stmt 1: offs_n_new = cur_batch_start_index + offs_n
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.ref .nat [] "cur_batch_start_index") (Op.ref .nat [BN] "offs_n")) _
          = some (Tile.vec onn) from by
        simp only [evalOp_add, evalOp_ref, BlockState.setReg_ne_name, ne_eq,
          String.reduceEq, not_false_eq_true, reduceCtorEq, hsi, hoffn,
          Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [Tile.bop, Tile.vec, Tile.scalar, NumericDType.add, Broadcast.leftIndex,
          Broadcast.rightIndex, honnDef])) ?_
  -- stmt 2: k_loc = masked page-table gather   [gread window]
  refine llama2IO_walkCons ?_
    (stepStmtR_assign_eq_some
      (llama2IO_klocR_eval R B_Loc B_Seqlen BN mil sblb sbls _ s (by exact hmem) (by exact hpids) onn kloc
        (by simp only [BlockState.setReg_same])
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hcb)
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hend)
        (fun j => by simp only [klocDef]))) ?_
  · intro bounds hC
    obtain ⟨_, hbG, _, _⟩ := hC
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoffs idx hactive
    rw [llama2IO_gAddrR_eval R _ onn (s.pids 0) sblb sbls
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hcb)
      (by simp only [BlockState.setReg_same])] at hoffs
    obtain rfl := Option.some.inj hoffs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [llama2IO_maskR_eval R _ onn mil
      (by simp only [BlockState.setReg_same])
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hend)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨j, uu⟩ := idx
    have hlive : onn j < mil := by simpa using hactl
    have h := hbG j ((hactIff j).mpr hlive)
    simpa only [honnDef] using h
  -- stmt 3: off_k = k_loc[:, None]·skbs + cur_kv_head·skh + offs_d[None, :]·skd
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (llama2_offk_eval _ BN BD skbs skh skd kloc (s.pids 1 / kvg)
        (by simp only [BlockState.setReg_same])
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hckv)
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hd))) ?_
  -- stmt 4: k = gather-addressed masked K load   [read2 window]
  refine llama2IO_walkCons ?_
    (stepStmtR_assign_eq_some
      (llama2IO_kLoadR_eval R K BN BD mil _ s (by exact hmem) offk onn kval
        (by simp only [BlockState.setReg_same, offkDef])
        (by
          simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
            String.reduceEq, not_false_eq_true, reduceCtorEq])
        (by
          simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            reduceCtorEq]
          exact hend)
        (fun j e => by simp only [kvalDef]))) ?_
  · intro bounds hC
    obtain ⟨_, _, hbK, _⟩ := hC
    simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def],
      ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def]⟩, ?_⟩
    intro offsets hoffs idx hactive
    rw [evalOpR_ref] at hoffs
    simp only [BlockState.setReg_same] at hoffs
    obtain rfl := Option.some.inj hoffs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [llama2IO_kMaskR_eval R BN BD _ onn mil
      (by
        simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
          String.reduceEq, not_false_eq_true, reduceCtorEq])
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hend)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨j, e, uu⟩ := idx
    have hlive : onn j < mil := by simpa using hactl
    have hlive' : mil - seqLen s B_Seqlen + (s.pids 2 * BN + j.val) < mil := by
      simpa only [honnDef] using hlive
    have hklocEq : kloc j = kLoc s B_Loc B_Seqlen mil sblb sbls BN j := by
      simp only [klocDef, kLoc, blockOffset, honnDef]
      exact if_pos hlive'
    have hoffkEq : offk j e
        = kOffset s B_Loc B_Seqlen mil sblb sbls skbs skh skd kvg BN j e.val := by
      simp only [offkDef, kOffset, hklocEq]
    show offk j e < bounds (Region.cast K)
    rw [hoffkEq]
    exact hbK j e ((hactIff j).mpr hlive)
  -- stmt 5: att_value = tl.sum(q[None, :] · k, 1)
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (llama2_attvalue_eval _ BN BD qval kval
        (by
          simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
            String.reduceEq, not_false_eq_true, reduceCtorEq])
        (by simp only [BlockState.setReg_same]))) ?_
  -- stmt 6: att_value *= sm_scale
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BN] "att_value")
            (Op.const sm_scale)) _
          = some (⟨fun idx : TileIndex [BN] => some (avraw idx.1 * sm_scale)⟩
              : Tile .real [BN]) from by
        simp only [evalOp_mul, evalOp_ref, evalOp_const, BlockState.setReg_same,
          Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]
        rfl)) ?_
  -- stmt 7: off_o = cur_head·ash + (cur_batch_in_all_start_index + offs_n)·asbs
  refine llama2IO_walkCons (fun _ _ => by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
    (llama2IO_stepR_of_assign (by simp only [evalOpR.eq_def, evalOp.eq_def])
      (show evalOp (Op.add .nat Broadcast.scalarL
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat ash))
            (Op.mul .nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL
                (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [BN] "offs_n"))
              (Op.constNat asbs))) _
          = some (Tile.vec (fun j : Fin BN => outOffset s B_Start_Loc ash asbs BN j))
          from by
        simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref,
          BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq, hch, hSL, hoffn, Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        ext idx
        simp only [Tile.bop, Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul,
          Broadcast.leftIndex, Broadcast.rightIndex, outOffset, startLoc,
          blockOffset])) ?_
  -- stmt 8: the masked store
  refine llama2IO_walkCons ?_
    (llama2IO_storeR_eval R Att_Out BN mil _
      (fun j : Fin BN => outOffset s B_Start_Loc ash asbs BN j)
      (fun j => avraw j * sm_scale) onn
      (by simp only [BlockState.setReg_same])
      (by
        simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
          String.reduceEq, not_false_eq_true, reduceCtorEq])
      (by
        simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
          String.reduceEq, not_false_eq_true, reduceCtorEq])
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hend)) ?_
  · intro bounds hC
    obtain ⟨_, _, _, hbO⟩ := hC
    simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨by simp [MemAccess.SafeAtR, Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
      by simp [MaskOpt.SafeAtR, Op.SafeAtR.eq_def], ?_⟩
    intro offsets hoffs idx hactive
    rw [evalOpR_ref] at hoffs
    simp only [BlockState.setReg_same] at hoffs
    obtain rfl := Option.some.inj hoffs
    obtain ⟨masks, hmask, hactl⟩ := hactive
    rw [llama2IO_maskR_eval R _ onn mil
      (by
        simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq,
          String.reduceEq, not_false_eq_true, reduceCtorEq])
      (by
        simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
          reduceCtorEq]
        exact hend)] at hmask
    obtain rfl := Option.some.inj hmask
    obtain ⟨j, uu⟩ := idx
    have hlive : onn j < mil := by simpa using hactl
    show outOffset s B_Start_Loc ash asbs BN j < bounds (Region.cast Att_Out)
    exact hbO j ((hactIff j).mpr hlive)
  refine llama2IO_walkNil _ ⟨?_, ?_⟩
  · -- readback
    intro i
    have hooFnInj : Function.Injective
        (fun k : TileIndex [BN] => outOffset s B_Start_Loc ash asbs BN k.1) := by
      rintro ⟨a, _⟩ ⟨b, _⟩ hab
      obtain rfl : a = b := hOutInj hab
      rfl
    rw [show outOffset s B_Start_Loc ash asbs BN i
          = (fun k : TileIndex [BN] => outOffset s B_Start_Loc ash asbs BN k.1)
              (i, PUnit.unit) from rfl,
      BlockState.scatter_readback_prop_masked_nd (region := Att_Out) _
        (fun k : TileIndex [BN] => outOffset s B_Start_Loc ash asbs BN k.1)
        (fun k : TileIndex [BN] => avraw k.1 * sm_scale)
        (fun k : TileIndex [BN] => onn k.1 < mil) hooFnInj (i, PUnit.unit)]
    by_cases hac : active s B_Seqlen mil BN i
    · have hlive : onn i < mil := (hactIff i).mp hac
      have hlive' : mil - seqLen s B_Seqlen + (s.pids 2 * BN + i.val) < mil := by
        simpa only [honnDef] using hlive
      rw [if_pos hlive, if_pos hac]
      have hklocEq : kloc i = kLoc s B_Loc B_Seqlen mil sblb sbls BN i := by
        simp only [klocDef, kLoc, blockOffset, honnDef]
        exact if_pos hlive'
      have hoffkEq : ∀ e : Fin BD, offk i e
          = kOffset s B_Loc B_Seqlen mil sblb sbls skbs skh skd kvg BN i e.val := by
        intro e
        simp only [offkDef, kOffset, hklocEq]
      simp only [avrawDef, qvalDef, kvalDef, tokenAttnLlama2DotScore]
      refine congrArg (· * sm_scale) (Finset.sum_congr rfl (fun e _ => ?_))
      rw [if_pos hlive, hoffkEq e]
    · have hlive : ¬ onn i < mil := fun h => hac ((hactIff i).mpr h)
      rw [if_neg hlive, if_neg hac]
      simp only [llama2IO_readMem_setReg, llama2IO_readMem_of_mem stt s hmem]
  · -- frame
    intro r o hcond
    refine (llama2IO_foldl_writeMem_frame Att_Out
      (fun k : TileIndex [BN] => outOffset s B_Start_Loc ash asbs BN k.1)
      (fun k : TileIndex [BN] => avraw k.1 * sm_scale)
      (fun k : TileIndex [BN] => onn k.1 < mil) _ _ r o ?_).trans ?_
    · intro hr k _ hk
      rcases hcond with hne | hno
      · exact absurd hr hne
      · exact fun heq => hno k.1 ((hactIff k.1).mpr hk) heq.symm
    · exact congrFun (congrFun hmem r) o

/-! ## The guard loop and the whole-kernel run

`range(0, block_mask, 1)` is an `if` encoded as a loop: the trip count is `0` or
`1`, so no loop-invariant principle is needed — `stepForRangeAux.step_one_iter` /
`step_ge` on the run side and two unfoldings of `Stmt.forRangeTraceSafeR` on the
safety side settle both branches. -/

/-- `stepStmtsR` append chaining. -/
private theorem llama2IO_stepStmtsR_append_some {R : RoundingModel} :
    ∀ (l1 : List Stmt) {l2 : List Stmt} {s s' : BlockState},
      stepStmtsR R l1 s = some s' → stepStmtsR R (l1 ++ l2) s = stepStmtsR R l2 s'
  | [], _, s, s', h => by
      simp only [stepStmtsR] at h
      rw [List.nil_append, Option.some.inj h]
  | st :: rest, l2, s, s', h => by
      cases hst : stepStmtR R st s with
      | none => simp [stepStmtsR, hst] at h
      | some u =>
          rw [List.cons_append, stepStmtsR_cons_some hst]
          rw [stepStmtsR_cons_some hst] at h
          exact llama2IO_stepStmtsR_append_some rest h

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **The whole-kernel walk**: prelude ++ the guard loop. Yields the framed
genuine closed-form run (what `hrun` rides) and the trace-safety of the whole
kernel under `llama2IOPreBounds` + `llama2IOBounds` (what `hts` rides). -/
private theorem llama2IO_runW (R : RoundingModel) (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BN => outOffset s B_Start_Loc.cast ash asbs BN i)) :
    (∃ sF, execR R ((token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen
          Att_Out mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN).toAlgKernel) s
        = some sF
      ∧ (∀ i : Fin BN,
          sF.readMem Att_Out (outOffset s B_Start_Loc.cast ash asbs BN i)
            = tokenAttnLlama2ClosedForm s Q K sm_scale B_Loc.cast B_Start_Loc.cast
                B_Seqlen.cast Att_Out mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg
                BD BN i)
      ∧ (∀ r o, (r ≠ Att_Out ∨ ∀ i : Fin BN, blockActive s B_Seqlen.cast BN →
            active s B_Seqlen.cast mil BN i →
            o ≠ outOffset s B_Start_Loc.cast ash asbs BN i) → sF.mem r o = s.mem r o))
    ∧ ∀ bounds : RegionBounds,
        llama2IOPreBounds B_Start_Loc.cast B_Seqlen.cast s bounds →
        llama2IOBounds Q K B_Loc.cast B_Start_Loc.cast B_Seqlen.cast Att_Out mil sblb sbls
          sqbs sqh sqd skbs skh skd ash asbs kvg BD BN s bounds →
        ((token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out mil
          sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN).toAlgKernel).TraceSafeR
            R bounds s := by
  obtain ⟨⟨s0, hpre, hInv0⟩, hpreSafe⟩ :=
    llama2IO_preLoopW R B_Start_Loc B_Seqlen mil sqbs sqh sqd kvg BD BN s
  have hInv0' := hInv0
  obtain ⟨hmem0, hpids0, _, _, _, _, _, _, _, _, _, hmask0⟩ := hInv0
  obtain ⟨⟨sB, hbodyRunR, hbodyRead, hbodyFrame⟩, hbodySafe⟩ :=
    llama2IO_bodyW R Q K sm_scale B_Loc.cast B_Start_Loc.cast B_Seqlen.cast Att_Out mil
      sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN s s0 hInv0' hOutInj
  have hbodyRun : stepStmts (llama2LoopBody Q K B_Loc.cast Att_Out sm_scale sblb sbls skbs
      skh skd ash asbs BD BN) (s0.setReg "start_mark" .nat [] (Tile.scalar 0)) = some sB := by
    rw [← llama2IO_body_castFree R Q K B_Loc.cast Att_Out sm_scale sblb sbls skbs skh skd
      ash asbs BD BN]
    exact hbodyRunR
  have hstopExact : evalOp (Op.ref .nat [] "block_mask") s0
      = some (Tile.scalar (if blockActive s B_Seqlen.cast BN then 1 else 0)) := by
    rw [evalOp_ref]; exact hmask0
  -- the guard loop's exact successor, by branch
  have hloop : stepStmt (Stmt.forRangeDyn "start_mark" (Op.constNat 0)
        (Op.ref .nat [] "block_mask") (Op.constNat 1)
        (llama2LoopBody Q K B_Loc.cast Att_Out sm_scale sblb sbls skbs skh skd ash asbs
          BD BN)) s0
      = some (if blockActive s B_Seqlen.cast BN then sB else s0) := by
    rw [stepForRangeAux.forRangeDyn_unfold]
    simp only [evalOp_constNat, hstopExact, Tile.scalar_data, Option.bind_some]
    by_cases hba : blockActive s B_Seqlen.cast BN
    · rw [if_pos hba, if_pos hba,
        stepForRangeAux.step_one_iter (by norm_num) (by norm_num) (by norm_num), hbodyRun]
    · rw [if_neg hba, if_neg hba, stepForRangeAux.step_ge (by norm_num) (by norm_num)]
  refine ⟨⟨if blockActive s B_Seqlen.cast BN then sB else s0, ?_, ?_, ?_⟩, ?_⟩
  · -- termination
    rw [llama2IO_execR_collapse, llama2IO_body_split,
      stepStmts.append_some
        (show stepStmts (llama2IOPrelude B_Start_Loc B_Seqlen mil sqbs sqh sqd kvg BD BN) s
            = some s0 from by
          rw [← llama2IO_stepStmtsR_castFree_of_stmts R _
            (llama2IO_prelude_stmt_castFree R B_Start_Loc B_Seqlen mil sqbs sqh sqd kvg
              BD BN) s]
          exact hpre),
      stepStmts.cons_some hloop, stepStmts.nil]
  · -- readback
    intro i
    by_cases hba : blockActive s B_Seqlen.cast BN
    · rw [if_pos hba, hbodyRead i]
      simp only [tokenAttnLlama2ClosedForm, hba, true_and]
    · rw [if_neg hba]
      simp only [tokenAttnLlama2ClosedForm, hba, false_and, if_false]
      exact llama2IO_readMem_of_mem s0 s hmem0 _ _
  · -- frame
    intro r o hcond
    by_cases hba : blockActive s B_Seqlen.cast BN
    · rw [if_pos hba]
      refine hbodyFrame r o ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · exact Or.inr (fun i hact => hno i hba hact)
    · rw [if_neg hba]
      exact congrFun (congrFun hmem0 r) o
  · -- the safety walk
    intro bounds hCpre hC
    unfold Kernel.TraceSafeR
    rw [llama2IO_body_split]
    refine Stmt.TraceSafeListR.append_intro _ s (hpreSafe bounds hCpre) (fun s1 hs1 => ?_)
    rw [hpre] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
    simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [llama2IO_evalOpR_constNat, llama2IO_evalOpR_constNat, llama2IO_stopOpR_castFree,
      hstopExact]
    simp only [Tile.scalar_data]
    -- the guard loop runs at most once, so a `c = 0 → · = s0` invariant suffices
    refine Stmt.forRangeTraceSafeR_inv R bounds "start_mark" _ 1 _
      (fun c u => c = 0 → u = s0) ?_ 0 s0 (fun _ => rfl)
    intro c u hc hPc
    have hc0 : c = 0 := by
      by_cases hba : blockActive s B_Seqlen.cast BN
      · rw [if_pos hba] at hc; omega
      · rw [if_neg hba] at hc; omega
    subst hc0
    obtain rfl := hPc rfl
    exact ⟨hbodySafe bounds hC, sB, hbodyRunR, fun h => absurd h (by omega)⟩

/-! ### ════════ ★ MAIN THEOREM (io face) ★ ════════ -/

set_option maxHeartbeats 1600000 in
/-- **The `⊨[R]` gather-skin headline** — `token_attn_llama2` on
`StreamMetaGatherMasked3DKernelIO₂`, at fully symbolic per-axis strides. For
every rounding model `R`, the faithful surface implements, on its gather-indexed
signature, the streamed closed form `llama2IOSpec`: every store-active output
lane `i` holds `(Σ_d q[d] · k[i, d]) · sm_scale`, read off the two pinned
streams. The kernel has **zero rounding events** (two `.nat` slot loads, a `.nat`
page-table gather, an unmasked `.real` `Q` load, an `other = 0.0`-defaulted
`.real` `K` load, `.real` dot arithmetic and an untyped `.real` store — no
`.to(...)` cast, no `Op.castFloat`), so the skin's boundary quantization
degenerates: the readback's `R.round .real` is the identity by the model's
defining `round_real`.

**The gather channel.** `B_Loc` enters as the skin's index channel
(`gty = .nat`, `gother = 0`), and the `K` window `read2` eats the gathered tile:
`G t jL · stride_kbs + …`. Like the `token_attn_mistral` sibling — and unlike the
softmax_reducev exemplar — the Python `K` load carries **the gather's own mask**
(`mask2 = gmask` on the row coordinate), so a masked-off lane never dereferences
the substituted `other=` address and this port needs **no hypothesis at all** on
`gother`; only the gather pin's *active* leg is used.

**The guard loop and `writeMask`.** The kernel's `for start_mark in range(0,
block_mask, 1)` has trip count `block_mask ∈ {0, 1}`, so `T = 1` and the store —
though syntactically inside the loop — executes at most once. The `block_mask = 0`
launches are handled **without any launch restriction**: `writeMask` carries the
`block_mask = 1` conjunct `pid₂ · BN < m 0`, so an inactive block claims no cell
and the skin's frame (everything outside the write-active window is untouched) is
exactly the "nothing was stored" fact. `pre ≡ True`.

**Hypothesis provenance**: `hOutInj` restates the exact headline's **open**
output-offset injectivity side condition in ∀-pids form (per-axis strides are
symbolic, so no contiguity discharge is available; it is what makes the masked
scatter's readback well-defined). The exact headline's `hundef` is **not** a
hypothesis: the skin's Hoare triple carries the `undef` pin itself, and this
kernel's only unmasked load never consults it.

Relation to the exact surface: the `Realizes_without_Rounding` headline above is
retained unchanged; this `⊨[R]` face restates the same genuine closed form on the
gather skin, for every `R` at once. -/
specification token_attn_llama2_io_correctness (R : RoundingModel)
    (Q K : RegionName) (sm_scale : ℝ) (B_Loc B_Start_Loc B_Seqlen : Region .nat)
    (Att_Out : RegionName)
    (mil sblb sbls sqbs sqh sqd skbs skh skd ash asbs kvg BD BN : Nat)
    (hOutInj : ∀ (pid₁ pid₂ base : Nat), Function.Injective
      (fun i : Fin BN => pid₁ * ash + (base + (pid₂ * BN + i.val)) * asbs)) :
    llama2IO Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out mil sblb sbls sqbs sqh sqd
        skbs skh skd ash asbs kvg BD BN ⊨[R]
      fun _ _ _ _ xs ys i => llama2IOSpec BD BN sm_scale xs ys i := by
  refine StreamMetaGatherMasked3DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact llama2IO_flattenOk Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out mil sblb sbls
      sqbs sqh sqd skbs skh skd ash asbs kvg BD BN
  · -- the safety walk
    intro bounds s m G xs ys _hpre hm hg _hgo _hx _hy hbm hbrG hbr1 hbr2 hbw
    simp only [llama2IO, llama2IOMetaBuf] at hm hg hbm hbrG hbr1 hbr2 hbw ⊢
    have hm0 : seqLen s B_Seqlen.cast = m (⟨0, by omega⟩ : Fin 2) :=
      hm (⟨0, by omega⟩ : Fin 2)
    have hm1 : startLoc s B_Start_Loc.cast = m (⟨1, by omega⟩ : Fin 2) :=
      hm (⟨1, by omega⟩ : Fin 2)
    have hOInj : Function.Injective
        (fun i : Fin BN => outOffset s B_Start_Loc.cast ash asbs BN i) := by
      simpa only [outOffset, startLoc, blockOffset] using
        hOutInj (s.pids 1) (s.pids 2) (startLoc s B_Start_Loc.cast)
    refine (llama2IO_runW R Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out mil sblb sbls
      sqbs sqh sqd skbs skh skd ash asbs kvg BD BN s hOInj).2 bounds
      ⟨by simpa using hbm (⟨0, by omega⟩ : Fin 2),
        by simpa using hbm (⟨1, by omega⟩ : Fin 2)⟩ ⟨?_, ?_, ?_, ?_⟩
    · -- the unmasked `Q` row
      intro d
      have h := hbr1 llama2IOStep d trivial
      simpa only [qOffset, llama2IOStep, Nat.add_zero] using h
    · -- the page-table gather's live window
      intro i hact
      have hlive : (mil - m (⟨0, by omega⟩ : Fin 2)) + (s.pids 2 * BN + i.val) < mil := by
        rw [← hm0]
        simpa only [active, blockOffset] using hact
      have h := hbrG llama2IOStep i hlive
      rw [← hm0] at h
      exact h
    · -- the gather-addressed `K` rows, on the gather's own window
      intro i d hact
      have hlive : (mil - m (⟨0, by omega⟩ : Fin 2)) + (s.pids 2 * BN + i.val) < mil := by
        rw [← hm0]
        simpa only [active, blockOffset] using hact
      have hGeq : kLoc s B_Loc.cast B_Seqlen.cast mil sblb sbls BN i = G llama2IOStep i := by
        rw [kLoc, hm0, blockOffset]
        exact hg llama2IOStep i hlive
      have h := hbr2 llama2IOStep
        (Lane2D.encode ((i, d, PUnit.unit) : TileIndex [BN, BD]))
        (by rw [Lane2D.decode_encode]; exact hlive)
      rw [Lane2D.decode_encode] at h
      rw [kOffset, hGeq]
      exact h
    · -- the store's active lanes
      intro i hact
      have hlive : (mil - m (⟨0, by omega⟩ : Fin 2)) + (s.pids 2 * BN + i.val) < mil := by
        rw [← hm0]
        simpa only [active, blockOffset] using hact
      have hba : s.pids 2 * BN < m (⟨0, by omega⟩ : Fin 2) := by
        rw [← hm0]
        rw [← hm0] at hlive
        omega
      have h := hbw i ⟨hba, hlive⟩
      simpa only [outOffset, blockOffset, hm1] using h
  · -- the rounded Hoare triple: the framed run + cast-free collapse
    intro s₀ m G xs ys _hpre _hu hm hg _hgo hx hy
    simp only [llama2IO, llama2IOMetaBuf] at hm hg hx hy ⊢
    have hm0 : seqLen s₀ B_Seqlen.cast = m (⟨0, by omega⟩ : Fin 2) :=
      hm (⟨0, by omega⟩ : Fin 2)
    have hm1 : startLoc s₀ B_Start_Loc.cast = m (⟨1, by omega⟩ : Fin 2) :=
      hm (⟨1, by omega⟩ : Fin 2)
    have hOInj : Function.Injective
        (fun i : Fin BN => outOffset s₀ B_Start_Loc.cast ash asbs BN i) := by
      simpa only [outOffset, startLoc, blockOffset] using
        hOutInj (s₀.pids 1) (s₀.pids 2) (startLoc s₀ B_Start_Loc.cast)
    obtain ⟨⟨sF, hexec, hread, hframe⟩, _⟩ :=
      llama2IO_runW R Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out mil sblb sbls sqbs
        sqh sqd skbs skh skd ash asbs kvg BD BN s₀ hOInj
    refine ⟨sF, hexec, ?_, ?_⟩
    · -- readback: the genuine closed form = the streamed closed form
      intro i hwm
      obtain ⟨hba, hlive⟩ := hwm
      have hbaS : blockActive s₀ B_Seqlen.cast BN := by
        rw [blockActive, hm0]; exact hba
      have hact : active s₀ B_Seqlen.cast mil BN i := by
        simp only [active, blockOffset, hm0]; exact hlive
      have hout : sF.readMem Att_Out (outOffset s₀ B_Start_Loc.cast ash asbs BN i)
          = tokenAttnLlama2DotScore s₀ Q K sm_scale B_Loc.cast B_Seqlen.cast mil sblb sbls
              sqbs sqh sqd skbs skh skd kvg BD BN i := by
        rw [hread i]
        simp only [tokenAttnLlama2ClosedForm, hbaS, hact, and_self, if_true]
      have haddr : s₀.pids 1 * ash
            + (startLoc s₀ B_Start_Loc.cast + (s₀.pids 2 * BN + i.val)) * asbs
          = outOffset s₀ B_Start_Loc.cast ash asbs BN i := by
        simp only [outOffset, blockOffset]
      have hval : sF.readMem Att_Out (outOffset s₀ B_Start_Loc.cast ash asbs BN i)
          = llama2IOSpec BD BN sm_scale xs ys i := by
        rw [hout]
        exact llama2IOSpec_eq_genuine Q K sm_scale B_Loc.cast B_Seqlen.cast mil sblb sbls
          sqbs sqh sqd skbs skh skd kvg BD BN s₀ (m (⟨0, by omega⟩ : Fin 2)) hm0 G xs ys
          (fun t j h => hg t j h) (fun t d => hx t d trivial) (fun t j h => hy t j h) i hact
      simp only [BlockState.readMemAs_real, R.round_real_apply, ← hm1, haddr, hval,
        FloatDType.real_ofReal]
    · -- the frame
      intro r o hcond
      refine hframe r o ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr (fun i hba hact => ?_)
        have haddr : s₀.pids 1 * ash
              + (startLoc s₀ B_Start_Loc.cast + (s₀.pids 2 * BN + i.val)) * asbs
            = outOffset s₀ B_Start_Loc.cast ash asbs BN i := by
          simp only [outOffset, blockOffset]
        have h := hno i ⟨by rw [← hm0]; exact hba,
          by simpa only [active, blockOffset, hm0] using hact⟩
        rw [← hm1, haddr] at h
        exact h

end IOFace

end VeriTile.Bench.TritonBenchG.TokenAttnLlama2
