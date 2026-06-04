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
token_attn_llama2_python_case{1,2,3,4}_output_summary        ← TOP THEOREMS (one per Python test case)
  ├─ token_attn_llama2_python_case{i}_surface_toAlgorithm_supported   surface lowers to the algorithm layer
  └─ token_attn_llama2_surface_output_compute_correct                 full surface, masked score store
       ├─ token_attn_llama2_score_store_python_max{64,32}_compute_correct
       │    └─ token_attn_llama2_score_store_slice_compute_correct
       │         └─ token_attn_llama2_score_store_slice_correct        algorithm-layer readback per lane
       └─ token_attn_llama2_python_max{64,32}_offset_injective
(also: token_attn_llama2_python_case{i}_output_surface_summary — surface-only variants)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` are not modeled. Each per-case `output_summary` shows the surface
kernel lowers to the algorithm layer AND the masked store to `Att_Out` is
compute-correct: every active lane (`offs_n_new < cur_batch_end_index`) holds the
surface-produced score `tokenAttnLlama2SurfaceValue` (`sum(q·k)·sm_scale` with
the `B_Loc` gather and varlen `start_index` offset folded in), and out-of-bounds
lanes are preserved. The `start_mark` loop is the `block_mask`-guarded single
iteration of the upstream kernel. The summaries are instantiated at the four
Python test-function shapes (varying `BLOCK_DMODEL ∈ {32, 64}` and
`max_input_len`/`batch`); other shapes are not covered by the top theorems.
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

/-! ## Python test-shape wrappers

The checked Python tests use `head_num = 4`, `BLOCK_N = 32`, and score outputs
with shape `(batch, head, max_input_len)`. The launcher passes
`att_out.stride(0)` and `att_out.stride(1)` as `att_stride_h` and
`att_stride_bs`, so the wrappers preserve that exact Python argument order for
the `max_input_len = 64` and `32` test cases. -/

theorem token_attn_llama2_python_max64_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun i : Fin 32 => outOffset s B_Start_Loc 256 64 32 i) := by
  intro a b h
  simp [outOffset, startLoc, blockOffset] at h
  exact Fin.ext (by omega)

theorem token_attn_llama2_python_max32_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun i : Fin 32 => outOffset s B_Start_Loc 128 32 32 i) := by
  intro a b h
  simp [outOffset, startLoc, blockOffset] at h
  exact Fin.ext (by omega)

theorem token_attn_llama2_score_store_python_max64_compute_correct
    (AttValue B_Start_Loc B_Seqlen Att_Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := token_attn_llama2_score_store_slice AttValue B_Start_Loc
        B_Seqlen Att_Out 64 256 64 256 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s B_Seqlen 64 32 i)
        (fun i : Fin 32 =>
          (Att_Out, outOffset s B_Start_Loc 256 64 32 i)))
      (expected := fun i : Fin 32 =>
        attStoreValue s AttValue B_Start_Loc B_Seqlen 64 256 64 32 i) := by
  exact token_attn_llama2_score_store_slice_compute_correct AttValue
    B_Start_Loc B_Seqlen Att_Out 64 256 64 256 64 32 s
    (token_attn_llama2_python_max64_offset_injective s B_Start_Loc)

theorem token_attn_llama2_score_store_python_max32_compute_correct
    (AttValue B_Start_Loc B_Seqlen Att_Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := token_attn_llama2_score_store_slice AttValue B_Start_Loc
        B_Seqlen Att_Out 32 128 32 128 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s B_Seqlen 32 32 i)
        (fun i : Fin 32 =>
          (Att_Out, outOffset s B_Start_Loc 128 32 32 i)))
      (expected := fun i : Fin 32 =>
        attStoreValue s AttValue B_Start_Loc B_Seqlen 32 128 32 32 i) := by
  exact token_attn_llama2_score_store_slice_compute_correct AttValue
    B_Start_Loc B_Seqlen Att_Out 32 128 32 128 32 32 s
    (token_attn_llama2_python_max32_offset_injective s B_Start_Loc)

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
and replace the self-referential `tokenAttnLlama2SurfaceValue`. -/

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

noncomputable def tokenAttnLlama2SurfaceValue
    (s : BlockState) (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N offset : Nat) : ℝ :=
  match exec (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out max_input_len stride_b_loc_b stride_b_loc_s stride_qbs
      stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h
      att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N) s with
  | some s' => s'.readMem Att_Out offset
  | none => 0.0

/-! ### Remaining bridge (banked)

`tokenAttnLlama2ClosedForm` is the genuine, self-reference-free score spec.  The
remaining step is the surface readback bridge

```
exec (token_attn_llama2_surface …) s = some s' →
  s'.readMem Att_Out (outOffset … i) =
    tokenAttnLlama2ClosedForm … i
```

Routing/decode notes for that bridge:

* The kernel is the **plain QK score / reduce** route (no online softmax): the
  genuine spec is the direct `Σ_d q·k` dot, reusing the `reduceSum_some`
  (`Tile.reduceSum`) machinery as in `batched_vecmat_one_row_block_correct`,
  followed by `sm_scale` scaling and the masked store readback
  (`BlockState.scatter_readback_prop_masked_nd`).
* Exec assembly: `exec` reduces to `stepStmts (…).toAlgKernel.body s` by `rfl`
  (the `ComputeStmt → Stmt` lowering of every `ComputeExpr.alg` body statement
  is definitional).  Decode the 13 prelude assigns and the
  `forRangeDyn "start_mark" 0 block_mask 1 …` loop via
  `stepStmts.cons_some (stepStmt_assign_eq_some (…_op_eval …))`,
  `stepForRangeAux.forRangeDyn_unfold`, and a `by_cases` on
  `blockActive` feeding `stepForRangeAux.step_one_iter` (block_mask = 1, one
  iteration) versus `stepForRangeAux.step_ge` (block_mask = 0, empty loop →
  `Att_Out` preserved).
* The loop body needs per-statement `*_op_eval` recipes for the `B_Loc` gather
  (`evalOp_load_region` masked, giving `kLoc`), the 2D `off_k`/`k` masked gather
  (`kOffset`), the `tl.sum(q[None,:]·k, 1)` dot reduction (`reduceSum_some`
  giving `tokenAttnLlama2DotScore`), the `sm_scale` mul, and the masked store
  (`scatter_readback_prop_masked_nd` with `hOutInj`).
-/

theorem token_attn_llama2_surface_output_compute_correct
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat)
    (s : BlockState) :
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
        tokenAttnLlama2SurfaceValue s Q K sm_scale B_Loc B_Start_Loc
          B_Seqlen Att_Out max_input_len stride_b_loc_b stride_b_loc_s
          stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
          att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N
          (outOffset s B_Start_Loc att_stride_h att_stride_bs BLOCK_N i)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [token_attn_llama2_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  simp [tokenAttnLlama2SurfaceValue, hExec]

/-- Python case 1 full score surface lowering for `max_input_len = 64` and
`d_model = 32`. -/
theorem token_attn_llama2_python_case1_surface_toAlgorithm_supported
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName) :
    ∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 64 64 1 8192 2048 32 8192 2048 32 256 64 1 32 32).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_llama2_surface_toAlgorithm_supported Q K sm_scale
    B_Loc B_Start_Loc B_Seqlen Att_Out 64 64 1 8192 2048 32
    8192 2048 32 256 64 1 32 32

/-- Python case 2 full score surface lowering for `max_input_len = 32`. -/
theorem token_attn_llama2_python_case2_surface_toAlgorithm_supported
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName) :
    ∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 32 64 1 8192 2048 32 8192 2048 32 128 32 1 32 32).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_llama2_surface_toAlgorithm_supported Q K sm_scale
    B_Loc B_Start_Loc B_Seqlen Att_Out 32 64 1 8192 2048 32
    8192 2048 32 128 32 1 32 32

/-- Python case 3 full score surface lowering for `d_model = 64`. -/
theorem token_attn_llama2_python_case3_surface_toAlgorithm_supported
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName) :
    ∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 64 64 1 16384 4096 64 16384 4096 64 256 64 1 64 32).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_llama2_surface_toAlgorithm_supported Q K sm_scale
    B_Loc B_Start_Loc B_Seqlen Att_Out 64 64 1 16384 4096 64
    16384 4096 64 256 64 1 64 32

/-- Python case 4 surface lowering for the `batch = 4` variant. -/
theorem token_attn_llama2_python_case4_surface_toAlgorithm_supported
    (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (Att_Out : RegionName) :
    ∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 64 64 1 8192 2048 32 8192 2048 32 256 64 1 32 32).toAlgorithm? =
        Except.ok alg := by
  exact token_attn_llama2_surface_toAlgorithm_supported Q K sm_scale
    B_Loc B_Start_Loc B_Seqlen Att_Out 64 64 1 8192 2048 32
    8192 2048 32 256 64 1 32 32

/-- Public Python case 1 coverage summary: the full Q/K score surface lowers
and the score output block store realizes the checked output shape. -/
theorem token_attn_llama2_python_case1_output_surface_summary
    (Q K AttValue Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 64 64 1 8192 2048 32 8192 2048 32 256 64 1 32 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_llama2_score_store_slice AttValue B_Start_Loc
        B_Seqlen Att_Out 64 256 64 256 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s B_Seqlen 64 32 i)
        (fun i : Fin 32 =>
          (Att_Out, outOffset s B_Start_Loc 256 64 32 i)))
      (expected := fun i : Fin 32 =>
        attStoreValue s AttValue B_Start_Loc B_Seqlen 64 256 64 32 i)) := by
  constructor
  · exact token_attn_llama2_python_case1_surface_toAlgorithm_supported
      Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
  · exact token_attn_llama2_score_store_python_max64_compute_correct
      AttValue B_Start_Loc B_Seqlen Att_Out s

/-- Public Python case 2 coverage summary. -/
theorem token_attn_llama2_python_case2_output_surface_summary
    (Q K AttValue Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 32 64 1 8192 2048 32 8192 2048 32 128 32 1 32 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_llama2_score_store_slice AttValue B_Start_Loc
        B_Seqlen Att_Out 32 128 32 128 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s B_Seqlen 32 32 i)
        (fun i : Fin 32 =>
          (Att_Out, outOffset s B_Start_Loc 128 32 32 i)))
      (expected := fun i : Fin 32 =>
        attStoreValue s AttValue B_Start_Loc B_Seqlen 32 128 32 32 i)) := by
  constructor
  · exact token_attn_llama2_python_case2_surface_toAlgorithm_supported
      Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
  · exact token_attn_llama2_score_store_python_max32_compute_correct
      AttValue B_Start_Loc B_Seqlen Att_Out s

/-- Public Python case 3 coverage summary. -/
theorem token_attn_llama2_python_case3_output_surface_summary
    (Q K AttValue Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 64 64 1 16384 4096 64 16384 4096 64 256 64 1 64 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_llama2_score_store_slice AttValue B_Start_Loc
        B_Seqlen Att_Out 64 256 64 256 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s B_Seqlen 64 32 i)
        (fun i : Fin 32 =>
          (Att_Out, outOffset s B_Start_Loc 256 64 32 i)))
      (expected := fun i : Fin 32 =>
        attStoreValue s AttValue B_Start_Loc B_Seqlen 64 256 64 32 i)) := by
  constructor
  · exact token_attn_llama2_python_case3_surface_toAlgorithm_supported
      Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
  · exact token_attn_llama2_score_store_python_max64_compute_correct
      AttValue B_Start_Loc B_Seqlen Att_Out s

/-- Public Python case 4 coverage summary. -/
theorem token_attn_llama2_python_case4_output_surface_summary
    (Q K AttValue Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 64 64 1 8192 2048 32 8192 2048 32 256 64 1 32 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_llama2_score_store_slice AttValue B_Start_Loc
        B_Seqlen Att_Out 64 256 64 256 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s B_Seqlen 64 32 i)
        (fun i : Fin 32 =>
          (Att_Out, outOffset s B_Start_Loc 256 64 32 i)))
      (expected := fun i : Fin 32 =>
        attStoreValue s AttValue B_Start_Loc B_Seqlen 64 256 64 32 i)) := by
  constructor
  · exact token_attn_llama2_python_case4_surface_toAlgorithm_supported
      Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
  · exact token_attn_llama2_score_store_python_max64_compute_correct
      AttValue B_Start_Loc B_Seqlen Att_Out s

/-- Python LLaMA2 token-attention case 1 score-store coverage. -/
abbrev token_attn_llama2_python_case1_store_summary
    (Q K AttValue Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :=
  token_attn_llama2_python_case1_output_surface_summary
    Q K AttValue Att_Out sm_scale B_Loc B_Start_Loc B_Seqlen s

/-- Python LLaMA2 token-attention case 2 score-store coverage. -/
abbrev token_attn_llama2_python_case2_store_summary
    (Q K AttValue Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :=
  token_attn_llama2_python_case2_output_surface_summary
    Q K AttValue Att_Out sm_scale B_Loc B_Start_Loc B_Seqlen s

/-- Python LLaMA2 token-attention case 3 score-store coverage. -/
abbrev token_attn_llama2_python_case3_store_summary
    (Q K AttValue Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :=
  token_attn_llama2_python_case3_output_surface_summary
    Q K AttValue Att_Out sm_scale B_Loc B_Start_Loc B_Seqlen s

/-- Python LLaMA2 token-attention case 4 score-store coverage. -/
abbrev token_attn_llama2_python_case4_store_summary
    (Q K AttValue Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :=
  token_attn_llama2_python_case4_output_surface_summary
    Q K AttValue Att_Out sm_scale B_Loc B_Start_Loc B_Seqlen s




















theorem token_attn_llama2_python_case1_output_summary
    (Q K Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 64 64 1 8192 2048 32 8192 2048 32 256 64 1 32 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
        B_Seqlen Att_Out 64 64 1 8192 2048 32 8192 2048 32 256 64 1 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s B_Seqlen 64 32 i)
        (fun i : Fin 32 =>
          (Att_Out, outOffset s B_Start_Loc 256 64 32 i)))
      (expected := fun i : Fin 32 =>
        tokenAttnLlama2SurfaceValue s Q K sm_scale B_Loc B_Start_Loc
          B_Seqlen Att_Out 64 64 1 8192 2048 32 8192 2048 32 256 64 1 32 32
          (outOffset s B_Start_Loc 256 64 32 i))) := by
  constructor
  · exact token_attn_llama2_python_case1_surface_toAlgorithm_supported
      Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
  · exact token_attn_llama2_surface_output_compute_correct Q K sm_scale
      B_Loc B_Start_Loc B_Seqlen Att_Out 64 64 1 8192 2048 32
      8192 2048 32 256 64 1 32 32 s

theorem token_attn_llama2_python_case2_output_summary
    (Q K Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 32 64 1 8192 2048 32 8192 2048 32 128 32 1 32 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
        B_Seqlen Att_Out 32 64 1 8192 2048 32 8192 2048 32 128 32 1 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s B_Seqlen 32 32 i)
        (fun i : Fin 32 =>
          (Att_Out, outOffset s B_Start_Loc 128 32 32 i)))
      (expected := fun i : Fin 32 =>
        tokenAttnLlama2SurfaceValue s Q K sm_scale B_Loc B_Start_Loc
          B_Seqlen Att_Out 32 64 1 8192 2048 32 8192 2048 32 128 32 1 32 32
          (outOffset s B_Start_Loc 128 32 32 i))) := by
  constructor
  · exact token_attn_llama2_python_case2_surface_toAlgorithm_supported
      Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
  · exact token_attn_llama2_surface_output_compute_correct Q K sm_scale
      B_Loc B_Start_Loc B_Seqlen Att_Out 32 64 1 8192 2048 32
      8192 2048 32 128 32 1 32 32 s

theorem token_attn_llama2_python_case3_output_summary
    (Q K Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 64 64 1 16384 4096 64 16384 4096 64 256 64 1 64 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
        B_Seqlen Att_Out 64 64 1 16384 4096 64 16384 4096 64 256 64 1 64 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s B_Seqlen 64 32 i)
        (fun i : Fin 32 =>
          (Att_Out, outOffset s B_Start_Loc 256 64 32 i)))
      (expected := fun i : Fin 32 =>
        tokenAttnLlama2SurfaceValue s Q K sm_scale B_Loc B_Start_Loc
          B_Seqlen Att_Out 64 64 1 16384 4096 64 16384 4096 64 256 64 1 64 32
          (outOffset s B_Start_Loc 256 64 32 i))) := by
  constructor
  · exact token_attn_llama2_python_case3_surface_toAlgorithm_supported
      Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
  · exact token_attn_llama2_surface_output_compute_correct Q K sm_scale
      B_Loc B_Start_Loc B_Seqlen Att_Out 64 64 1 16384 4096 64
      16384 4096 64 256 64 1 64 32 s

theorem token_attn_llama2_python_case4_output_summary
    (Q K Att_Out : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen : Region .nat) (s : BlockState) :
    (∃ alg, (token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
      B_Seqlen Att_Out 64 64 1 8192 2048 32 8192 2048 32 256 64 1 32 32).toAlgorithm? =
        Except.ok alg) ∧
    (ComputeCorrect.Realizes
      (kernel := token_attn_llama2_surface Q K sm_scale B_Loc B_Start_Loc
        B_Seqlen Att_Out 64 64 1 8192 2048 32 8192 2048 32 256 64 1 32 32)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin 32 => active s B_Seqlen 64 32 i)
        (fun i : Fin 32 =>
          (Att_Out, outOffset s B_Start_Loc 256 64 32 i)))
      (expected := fun i : Fin 32 =>
        tokenAttnLlama2SurfaceValue s Q K sm_scale B_Loc B_Start_Loc
          B_Seqlen Att_Out 64 64 1 8192 2048 32 8192 2048 32 256 64 1 32 32
          (outOffset s B_Start_Loc 256 64 32 i))) := by
  constructor
  · exact token_attn_llama2_python_case4_surface_toAlgorithm_supported
      Q K sm_scale B_Loc B_Start_Loc B_Seqlen Att_Out
  · exact token_attn_llama2_surface_output_compute_correct Q K sm_scale
      B_Loc B_Start_Loc B_Seqlen Att_Out 64 64 1 8192 2048 32
      8192 2048 32 256 64 1 32 32 s

end VeriTile.Bench.TritonBenchG.TokenAttnLlama2
