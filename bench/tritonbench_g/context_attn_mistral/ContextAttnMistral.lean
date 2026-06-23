import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel

/-!
# `context_attn_mistral` — strict per-kernel correctness

`_fwd_kernel` is Mistral-style varlen context (prefill) attention with a
**sliding-window** mask. Each program `(cur_batch, cur_head, start_m)` loads a
`[BLOCK_M, BLOCK_DMODEL]` query tile, runs an online-softmax
(`m_i`/`l_i`/`acc`) loop over the key/value tokens with a causal mask combined
with the `sliding_window` band mask (`start_n + offs_n > offs_m - sliding_window`),
and stores the accumulated `acc` tile to `Out`, masked by
`offs_m < cur_batch_seq_len`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_fwd_kernel[grid](...)` with
`grid = (batch, head, cdiv(...))`, the scheduling over batch / head / sequence
blocks, and how the runtime composes per-program writes into one buffer) is the
*trusted boundary*, not a proof obligation here. Because the program ids
`(cur_batch, cur_head, start_m)` are universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
context_attn_mistral_genuine_output_summary_general        ← TOP THEOREM (symbolic, dimension-general)
  └─ mistral_exec_general                                     whole-kernel exec → genuine closed form
  └─ context_attn_mistral_genuine_output_summary             concrete shape corollary (sliding_window ∈ {10, 20})

(separate per-lane store-slice chain, not a dependency of the summary above:
  context_attn_mistral_final_store_slice_compute_correct
    └─ context_attn_mistral_final_store_slice_correct         algorithm-layer readback per lane)
(supporting: context_attn_mistral_fwd_kernel_surface_toAlgorithm_supported)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The verified compute claim is scoped to the **final masked writeback**
of the accumulated `acc` tile into `Out`: every active lane
(`offs_m < cur_batch_seq_len`, with `offs_d < head_dim` folded into the slice)
holds the **genuine** sliding-window causal-softmax closed form
`mistralGenuineOutValueG` (the boundary-masked sliding-window-softmax fold of the
loaded Q/K/V memory, `-1e9` sentinel kept), and out-of-bounds lanes are preserved
— a pure function of memory, NOT a self-referential executed readback. The
online-softmax streaming loop (`m_i`/`l_i`/`acc` updates, `tl.dot`, the causal +
sliding-window band mask, and the `m_ij == -1e9 → 0` numerical guard) is decoded
statement-by-statement and proven to collapse to that closed form. Note the top
theorem bundles only the compute-correct claim; `toAlgorithm?` lowering is
available separately as
`context_attn_mistral_fwd_kernel_surface_toAlgorithm_supported` but is not folded
into this summary. The top theorem is dimension-general: it is stated over
symbolic `BLK`/`DM` (with `BLOCK_M = BLOCK_N = BLK`, `BLOCK_DMODEL = DM`), the
sliding window `sw`, and the layout strides `(rs, hs, 1)`. The Python test shape
(`BLK = DM = 128`, `rs = 768`, `hs = 128`, `sliding_window ∈ {10, 20}`) is
recovered as a concrete special case.
-/

namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral

open VeriTile.Triton

set_option linter.unusedSimpArgs false


/-! **★ Main theorem:** `context_attn_mistral_genuine_output_summary_general` -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct

/-- Faithful DSL port of `context_attn_mistral.py`'s `_fwd_kernel`. -/
def context_attn_mistral_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  off_k = offs_n[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
    offs_d[:, None] * $(stride_kd)
  off_v = offs_n[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
    offs_d[None, :] * $(stride_vd)

  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  k_ptrs = K + off_k
  v_ptrs = V + off_v

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))

  for start_n in range($(0), block_mask * (start_m + $(1)) * $(BLOCK_M), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_kbs),
      mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, -1e9)
    qk = tl.where((start_n + offs_n[None, :]) > (offs_m[:, None] - $(sliding_window)),
      qk, -1e9)

    m_ij = tl.max(qk, 1)
    m_ij = tl.where(m_ij == -1e9, 0.0, m_ij)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)

    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    l_i_new = tl.where(l_i_new == 0.0, 1e-9, l_i_new)

    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
      other=0.0)

    p = (p).to(v.dtype)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
}

/-- The full Mistral sliding-window context-attention surface lowers to the
algorithm layer. -/
theorem context_attn_mistral_fwd_kernel_surface_toAlgorithm_supported
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (context_attn_mistral_fwd_kernel_surface Q K V sm_scale B_Start_Loc
      B_Seqlen Out stride_qbs stride_qh stride_qd stride_kbs stride_kh
      stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgorithm?
        = Except.ok alg := by
  simp [context_attn_mistral_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `context_attn_mistral.py`'s
`_fwd_kernel`.

The full kernel computes sliding-window causal context attention. This slice
starts from a precomputed `Acc` tile and proves the final masked writeback into
`Out`, preserving the source address shape using `B_Start_Loc`, `B_Seqlen`,
`cur_batch`, `cur_head`, and `start_m`. The inner `tl.float32`
streaming-softmax accumulator and sliding-window score masks are outside this
slice. -/
def context_attn_mistral_final_store_slice
    (Acc : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(BLOCK_DMODEL))
  acc = tl.load(Acc + cur_batch * $(stride_acc_b) + cur_head * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_d[None, :] * $(stride_acc_d),
      mask=mask, other=0.0)
  tl.store(Out + (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
      cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od), acc, mask=mask)
}

def seqLen (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  s.readMemValue .nat B_Seqlen (s.pids 0)

def startLoc (s : BlockState) (B_Start_Loc : RegionName) : Nat :=
  s.readMemValue .nat B_Start_Loc (s.pids 0)

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 2 * BLOCK_M + i.val

def dIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (B_Seqlen : RegionName) (BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < seqLen s B_Seqlen

instance activeDecidable
    (s : BlockState) (B_Seqlen : RegionName) (BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s B_Seqlen BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + dIndex idx * stride_acc_d

def outOffset
    (s : BlockState) (B_Start_Loc : RegionName)
    (stride_obs stride_oh stride_od BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  (startLoc s B_Start_Loc + mIndex s BLOCK_M idx.1) * stride_obs +
    s.pids 1 * stride_oh + dIndex idx * stride_od

noncomputable def accStoreValue
    (s : BlockState) (Acc B_Seqlen : RegionName)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s B_Seqlen BLOCK_M idx then
      some (s.readMem Acc
        (accOffset s stride_acc_b stride_acc_h stride_acc_m stride_acc_d
          BLOCK_M idx))
    else some (0.0 : ℝ))

/-- Algorithm-layer correctness for the masked context-attention output store. -/
theorem context_attn_mistral_final_store_slice_correct
    (Acc B_Start_Loc B_Seqlen Out : RegionName)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s B_Start_Loc stride_obs stride_oh stride_od
        BLOCK_M idx
      (exec (context_attn_mistral_final_store_slice Acc B_Start_Loc B_Seqlen Out
            stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_obs
            stride_oh stride_od BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s B_Seqlen BLOCK_M idx then
            accStoreValue s Acc B_Seqlen stride_acc_b stride_acc_h
              stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, context_attn_mistral_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        BlockState.readMemValue, seqLen, startLoc, mIndex, dIndex, active,
        accOffset, outOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      (s.readMemValue .nat B_Start_Loc (s.pids 0) +
          (s.pids 2 * BLOCK_M + idx.1.val)) * stride_obs +
        s.pids 1 * stride_oh + idx.2.1.val * stride_od
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 2 * BLOCK_M + idx.1.val <
            s.readMemValue .nat B_Seqlen (s.pids 0) then
          some (s.readMem Acc
            (s.pids 0 * stride_acc_b + s.pids 1 * stride_acc_h +
              (s.pids 2 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_d))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 2 * BLOCK_M + idx.1.val <
        s.readMemValue .nat B_Seqlen (s.pids 0)
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, startLoc, mIndex, dIndex,
      BlockState.readMemValue] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      accStoreValue s Acc B_Seqlen stride_acc_b stride_acc_h stride_acc_m
        stride_acc_d BLOCK_M idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 2 * BLOCK_M + idx.1.val <
        s.readMemValue .nat B_Seqlen (s.pids 0)
  · rfl
  · rfl

/-- Compute-facing correctness for the masked context-attention output store. -/
theorem context_attn_mistral_final_store_slice_compute_correct
    (Acc B_Start_Loc B_Seqlen Out : RegionName)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := context_attn_mistral_final_store_slice Acc B_Start_Loc B_Seqlen
        Out stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_obs
        stride_oh stride_od BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s B_Seqlen BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        accStoreValue s Acc B_Seqlen stride_acc_b stride_acc_h stride_acc_m
          stride_acc_d BLOCK_M idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_mistral_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := context_attn_mistral_final_store_slice_correct Acc B_Start_Loc
    B_Seqlen Out stride_acc_b stride_acc_h stride_acc_m stride_acc_d
    stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h


section MistralGeneral

/-! ## Mistral sliding-window genuine closed-form correctness (general dims)

This section adds a GENUINE closed-form correctness proof for
`context_attn_mistral_fwd_kernel_surface`: the final `acc` store equals a genuine
sliding-window softmax fold of the loaded `Q`/`K`/`V` memory (NOT a kernel
readback). The streaming `m_i`/`l_i`/`acc` recurrence (online-normalized, plain
`exp`, ⊥-seeded, with the two causal+window `where`s against the finite `-1e9`
sentinel and the `m_ij == -1e9 → 0` / `l_i_new == 0 → 1e-9` numerical guards) is
decoded statement-by-statement and proven to collapse to the closed form. The
genuine value keeps the `-1e9` sentinel exactly (every key in `Fin S` present),
exactly as the kernel does.

It is self-contained: the pure-math online-softmax lemmas (`osNormStepBot*`) and
the per-statement eval lemmas are copied with `mistral` prefixes from the
`context_attn_nopad` General stack (which is not importable as a module here),
edited for the two-`where` sentinel scoring, the block-max guard, and the
`l_i_new` guard. -/

/-- One ⊥-seeded online-normalized softmax step absorbing key `(sc, v)`
(identical to nopad's `osNormStepBot`). -/
noncomputable def osNormStepBot
    (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp (WithBot.realSub m m')).unbotD 0
  let l' := l * α + Real.exp (sc - m'.unbotD 0)
  let acc' := acc * (l / l' * α) + (Real.exp (sc - m'.unbotD 0) / l') * v
  (m', l', acc')

theorem osNormStepBot_foldl_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osNormStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

theorem osNormStepBot_foldl_consistent
    (xs : List (ℝ × ℝ)) (m : WithBot ℝ) (l acc T L : ℝ)
    (hL0 : 0 ≤ L)
    (hxpos : ∀ p ∈ xs, True)
    (hl : l = (m.elim 0 (fun r => Real.exp (-r))) * L)
    (hacc : acc * L = T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0)
    (hLpos : 0 < L → l ≠ 0) :
    let st := xs.foldl osNormStepBot (m, l, acc)
    let L' := L + (xs.map (fun p => Real.exp p.1)).sum
    let T' := T + (xs.map (fun p => Real.exp p.1 * p.2)).sum
    st.2.1 = (st.1.elim 0 (fun r => Real.exp (-r))) * L'
      ∧ st.2.2 * L' = T'
      ∧ 0 ≤ L'
      ∧ (st.1 = ⊥ → L' = 0)
      ∧ (st.1 = ⊥ → T' = 0)
      ∧ (0 < L' → st.2.1 ≠ 0) := by
  induction xs generalizing m l acc T L with
  | nil => exact ⟨by simpa using hl, by simpa using hacc, by simpa using hL0,
      by simpa using hmL, by simpa using hmT, by simpa using hLpos⟩
  | cons x xs ih =>
    obtain ⟨sc, v⟩ := x
    set m' : WithBot ℝ := m ⊔ ((sc : ℝ) : WithBot ℝ) with hm'
    have hm'r : ∃ r : ℝ, m' = (r : WithBot ℝ) := by
      cases m with
      | bot => exact ⟨sc, by rw [hm']; rfl⟩
      | coe a => exact ⟨max a sc, by rw [hm']; rw [← WithBot.coe_max]⟩
    obtain ⟨mr, hmr⟩ := hm'r
    have hκm' : m'.elim 0 (fun r => Real.exp (-r)) = Real.exp (-mr) := by rw [hmr]; rfl
    have hunbot : m'.unbotD 0 = mr := by rw [hmr]; rfl
    set L' := L + Real.exp sc with hL'd
    set T' := T + Real.exp sc * v with hT'd
    set α : ℝ := (WithBot.realExp (WithBot.realSub m m')).unbotD 0 with hαd
    set p : ℝ := Real.exp (sc - m'.unbotD 0) with hpd
    set l' : ℝ := l * α + p with hl'd
    set acc' : ℝ := acc * (l / l' * α) + (p / l') * v with hacc'd
    have hlα : l * α = Real.exp (-mr) * L := by
      cases m with
      | bot =>
        have hL0' : L = 0 := hmL rfl
        rw [hl, hL0']; simp
      | coe a =>
        have hm'a : m' = ((max a sc : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a sc := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαv : α = Real.exp (a - max a sc) := by
          rw [hαd, hm'a, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
        rw [hl, hαv,
          show ((↑a : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-a) from rfl]
        rw [show Real.exp (-a) * L * Real.exp (a - max a sc)
            = (Real.exp (-a) * Real.exp (a - max a sc)) * L from by ring,
          ← Real.exp_add, hmra]
        ring_nf
    have hl'eq : l' = Real.exp (-mr) * L' := by
      rw [hl'd, hlα, hpd, hunbot, hL'd]
      have e2 : Real.exp (sc - mr) = Real.exp (-mr) * Real.exp sc := by
        rw [← Real.exp_add]; ring_nf
      rw [e2]; ring
    have hL'pos : 0 ≤ L' := by rw [hL'd]; positivity
    have hL'strict : 0 < L' := by rw [hL'd]; positivity
    have hl'ne : l' ≠ 0 := by rw [hl'eq]; positivity
    have hacc'eq : acc' * L' = T' := by
      rw [hacc'd, hpd, hunbot]
      rw [add_mul]
      have hl'val : l' = Real.exp (-mr) * L' := hl'eq
      have e2 : Real.exp (sc - mr) / l' * v * L'
          = Real.exp sc * v := by
        rw [hl'val]
        rw [show Real.exp (sc - mr) = Real.exp (-mr) * Real.exp sc from by
          rw [← Real.exp_add]; ring_nf]
        have hexpne : Real.exp (-mr) ≠ 0 := Real.exp_ne_zero _
        have hL'ne : (L' : ℝ) ≠ 0 := ne_of_gt hL'strict
        field_simp
      have e1 : acc * (l / l' * α) * L' = T := by
        have hexpne : Real.exp (-mr) ≠ 0 := Real.exp_ne_zero _
        have hL'ne : (L' : ℝ) ≠ 0 := ne_of_gt hL'strict
        have hrw : acc * (l / l' * α) * L'
            = acc * (l * α) * L' / l' := by ring
        rw [hrw, hlα, hl'val]
        rw [show acc * (Real.exp (-mr) * L) * L' / (Real.exp (-mr) * L')
            = acc * L * (Real.exp (-mr) * L' / (Real.exp (-mr) * L')) from by ring]
        rw [div_self (mul_ne_zero hexpne hL'ne), mul_one, hacc]
      rw [e1, e2, hT'd]
    have hL'bot : m' = ⊥ → L' = 0 := fun h => absurd h (by rw [hmr]; simp)
    have hT'bot : m' = ⊥ → T' = 0 := fun h => absurd h (by rw [hmr]; simp)
    have hL'ne0 : 0 < L' → l' ≠ 0 := fun _ => hl'ne
    have step := ih m' l' acc' T' L' hL'pos (fun _ _ => trivial)
      (by rw [hl'eq, hκm']) hacc'eq hL'bot hT'bot hL'ne0
    simpa [List.foldl_cons, osNormStepBot, hm', hαd, hpd, hl'd, hacc'd,
      List.map_cons, List.sum_cons, hL'd, hT'd, add_assoc, add_comm, add_left_comm,
      mul_comm, mul_left_comm] using step

theorem nopad_realExp_eq_some_unbotD (x : WithBot ℝ) :
    WithBot.realExp x = some ((WithBot.realExp x).unbotD 0) := by
  cases x <;> rfl

theorem osNormStepBot_block_fst (m : WithBot ℝ) (l acc : ℝ) (block : List (ℝ × ℝ)) :
    (block.foldl osNormStepBot (m, l, acc)).1
      = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  rw [osNormStepBot_foldl_fst]
  induction block generalizing m with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldl_cons, List.foldr_cons]
    rw [ih]
    rw [show (m ⊔ ((a.1 : ℝ) : WithBot ℝ)) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))
          = m ⊔ (((a.1 : ℝ) : WithBot ℝ) ⊔ (List.foldr (· ⊔ ·) ⊥ (List.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) t))) from by
      rw [sup_assoc]]

theorem foldr_sup_coe_bot_iff (xs : List (ℝ × ℝ)) :
    (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ = ⊥ ↔ xs = [] := by
  induction xs with
  | nil => simp
  | cons a t ih =>
    simp only [List.map_cons, List.foldr_cons]
    constructor
    · intro h
      exact absurd (le_bot_iff.mp (h ▸ le_sup_left)) WithBot.coe_ne_bot
    · intro h; exact absurd h (by simp)

private theorem osNormStepBot_blockSum_rescale (block : List (ℝ × ℝ)) (Mr : ℝ)
    (g : ℝ × ℝ → ℝ) :
    let blockSup := (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    (WithBot.realExp (WithBot.realSub blockSup ((Mr : ℝ) : WithBot ℝ))).unbotD 0
        * (block.map (fun p => Real.exp (p.1 - blockSup.unbotD 0) * g p)).sum
      = Real.exp (-Mr) * (block.map (fun p => Real.exp p.1 * g p)).sum := by
  intro blockSup
  cases hbs : blockSup with
  | bot =>
    have hempty : block = [] := by
      rcases block with _ | ⟨a, t⟩
      · rfl
      · exfalso
        have hle : ((a.1 : ℝ) : WithBot ℝ) ≤ blockSup := by
          simp only [blockSup, List.map_cons, List.foldr_cons]; exact le_sup_left
        rw [hbs] at hle
        exact absurd (le_bot_iff.mp hle) (WithBot.coe_ne_bot)
    subst hempty; simp
  | coe bsr =>
    have hbsunbot : blockSup.unbotD 0 = bsr := by rw [hbs]; rfl
    simp only [hbs, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe, hbsunbot]
    rw [show (block.map (fun p => Real.exp (p.1 - bsr) * g p)).sum
          = Real.exp (-bsr) * (block.map (fun p => Real.exp p.1 * g p)).sum from by
      rw [← List.sum_map_mul_left]
      congr 1; apply List.map_congr_left; intro p _
      rw [show p.1 - bsr = -bsr + p.1 from by ring, Real.exp_add]; ring]
    rw [show Real.exp (bsr - Mr) * (Real.exp (-bsr) * (block.map (fun p => Real.exp p.1 * g p)).sum)
          = (Real.exp (bsr - Mr) * Real.exp (-bsr)) * (block.map (fun p => Real.exp p.1 * g p)).sum from by ring,
      ← Real.exp_add]
    ring_nf

end MistralGeneral


section MistralGeneralDefs

/-- General coordinate-faithful query tile (contiguous strides `(rs, hs, 1)`). -/
noncomputable def ctxQTileG
    (s : BlockState) (Q B_Start_Loc : RegionName) (rs hs BLK DM : Nat) :
    TileIndex [BLK, DM] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((startLoc s B_Start_Loc + (s.pids 2 * BLK + i.val)) * rs
        + s.pids 1 * hs + e.val)

/-- General coordinate-faithful key tile. -/
noncomputable def ctxKTileG
    (s : BlockState) (K B_Start_Loc : RegionName) (rs hs S DM : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, e, _) =>
    s.readMem K
      ((startLoc s B_Start_Loc + j.val) * rs + s.pids 1 * hs + e.val)

/-- General coordinate-faithful value tile. -/
noncomputable def ctxVTileG
    (s : BlockState) (V B_Start_Loc : RegionName) (rs hs S DM : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, d, _) =>
    s.readMem V
      ((startLoc s B_Start_Loc + j.val) * rs + s.pids 1 * hs + d.val)

/-- General sequence-length-masked key tile. -/
noncomputable def ctxKTileMG
    (s : BlockState) (K B_Start_Loc : RegionName) (rs hs S DM bel : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, e, u) => if j.val < bel then ctxKTileG s K B_Start_Loc rs hs S DM (j, e, u) else 0

/-- General sequence-length-masked value tile. -/
noncomputable def ctxVTileMG
    (s : BlockState) (V B_Start_Loc : RegionName) (rs hs S DM bel : Nat) :
    TileIndex [S, DM] → ℝ :=
  fun (j, d, u) => if j.val < bel then ctxVTileG s V B_Start_Loc rs hs S DM (j, d, u) else 0

/-- General row-masked query tile. -/
noncomputable def ctxQTileMRowG
    (s : BlockState) (Q B_Start_Loc : RegionName) (rs hs BLK DM bel : Nat) :
    TileIndex [BLK, DM] → ℝ :=
  fun (i, e, u) =>
    if s.pids 2 * BLK + i.val < bel then ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, u) else 0

/-- The Mistral sliding-window+causal "active" predicate for cell `(gi, j)`:
`j ≤ gi` (causal) AND `gi ∸ sliding_window < j` (sliding window band). On inactive
cells the kernel keeps the finite `-1e9` sentinel score. -/
def mistralActive (gi sw j : Nat) : Prop := j ≤ gi ∧ gi - sw < j

instance (gi sw j : Nat) : Decidable (mistralActive gi sw j) := by
  unfold mistralActive; infer_instance

/-- The per-cell sentinel-kept score: real scaled dot on active cells, the finite
`-1e9` sentinel on masked cells. -/
noncomputable def mistralScore
    (s : BlockState) (Q K B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw : Nat) (i : Fin BLK) (j : Fin S) : ℝ :=
  if mistralActive (s.pids 2 * BLK + i.val) sw j.val then
    sm_scale * Finset.univ.sum (fun e : Fin DM =>
      ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit)
        * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit))
  else (-1e9 : ℝ)

/-- The row-masked variant of the sentinel-kept score (uses `ctxQTileMRowG`). -/
noncomputable def mistralScoreM
    (s : BlockState) (Q K B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw : Nat) (i : Fin BLK) (j : Fin S) : ℝ :=
  if mistralActive (s.pids 2 * BLK + i.val) sw j.val then
    sm_scale * Finset.univ.sum (fun e : Fin DM =>
      ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
        * ctxKTileMG s K B_Start_Loc rs hs S DM bel (j, e, PUnit.unit))
  else (-1e9 : ℝ)

/-- General boundary-masked sliding-window-softmax fold (the faithful kernel value).
Every key in `Fin S` contributes weight `exp(score j)` (sentinel kept), so the
denominator is always positive. -/
noncomputable def contextAttnMistralExactFoldMG
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM S bel sw : Nat) (idx : TileIndex [BLK, DM]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let weight := fun j : Fin S => Real.exp (mistralScore s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j)
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))
  numer / denom

/-- General kernel-decoded streamed window `S = block_mask·(start_m+1)·BLK`. -/
def ctxMistralWindowG (s : BlockState) (B_Seqlen : RegionName) (BLK : Nat) : Nat :=
  let sl := seqLen s B_Seqlen
  let bm := if BLK * s.pids 2 < sl then 1 else 0
  bm * (s.pids 2 + 1) * BLK

/-- General k/v load boundary `bel = cur_batch_seq_len`. -/
def ctxMistralBel (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  seqLen s B_Seqlen

/-- General genuine closed-form output value (boundary-masked sliding-window-softmax
fold), with the `-1e9` sentinel kept on masked cells. -/
noncomputable def mistralGenuineOutValueG
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM sw : Nat) (idx : TileIndex [BLK, DM]) : ℝ :=
  contextAttnMistralExactFoldMG s Q K V B_Start_Loc sm_scale rs hs BLK DM
    (ctxMistralWindowG s B_Seqlen BLK) (ctxMistralBel s B_Seqlen) sw idx

/-! ### Sentinel-kept key lists (every key present, no causal drop). -/

noncomputable def mistralBlockMG
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d : Fin DM) : List (ℝ × ℝ) :=
  (List.finRange S).filterMap (fun j : Fin S =>
    if c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then
      some (mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j,
            ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))
    else none)

end MistralGeneralDefs


section MistralBlockStep

/-- The guarded block max (kernel: `m_ij = where(m_ij == -1e9, 0, m_ij)`), as a
`WithBot ℝ`. `blockSup` is the unguarded block max; on an all-masked block
(`blockSup = -1e9`) it is reset to `0`. -/
noncomputable def mistralGuard (blockSup : WithBot ℝ) : WithBot ℝ :=
  if blockSup.unbotD 0 = (-1e9 : ℝ) then ((0 : ℝ) : WithBot ℝ) else blockSup

/-- The guard of a nonempty (`≠ ⊥`) block max is finite. -/
theorem mistralGuard_ne_bot {blockSup : WithBot ℝ} (h : blockSup ≠ ⊥) :
    mistralGuard blockSup ≠ ⊥ := by
  unfold mistralGuard
  by_cases hb : blockSup.unbotD 0 = (-1e9 : ℝ)
  · rw [if_pos hb]; exact WithBot.coe_ne_bot
  · rw [if_neg hb]; exact h

/-- One Mistral block step matching the kernel exactly: block max `m_ij` is the
GUARDED block sup `G`; `p = exp(score − G)`, `l_ij = Σ p`; `m_i_new = m ⊔ G`,
`α = exp(m ⊖ m')`, `β = exp(G ⊖ m')`, `l_i_new = α·l + β·l_ij`,
`acc' = acc·(l/l'·α) + Σ (exp(score−G)·β/l'·v)`. -/
noncomputable def mistralBlockStepBot
    (st : WithBot ℝ × ℝ × ℝ) (block : List (ℝ × ℝ)) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let blockSup := (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
  let G := mistralGuard blockSup
  let m' := m ⊔ G
  let α := (WithBot.realExp (WithBot.realSub m m')).unbotD 0
  let β := (WithBot.realExp (WithBot.realSub G m')).unbotD 0
  let lij := (block.map (fun p => Real.exp (p.1 - G.unbotD 0))).sum
  let l' := α * l + β * lij
  let acc' := acc * (l / l' * α)
    + (block.map (fun p => Real.exp (p.1 - G.unbotD 0) * β / l' * p.2)).sum
  (m', l', acc')

/-- **Per-block consistency for `mistralBlockStepBot`.** Given an anchor state
`(m, ↑l, ↑acc)` consistent with batch denom `L` (`l = κ(m)·L`) and accumulator `T`
(`acc·L = T`), one Mistral block step preserves the anchor at `L' = L + Σexp(score)`,
`T' = T + Σexp(score)·v`, regardless of the guarded pivot `G` (path-independence:
the `G` cancels in `β·l_ij = exp(−mr)·Σexp(score)`). -/
theorem mistralBlockStepBot_consistent
    (m : WithBot ℝ) (l acc T L : ℝ) (block : List (ℝ × ℝ))
    (hbne : block ≠ [])
    (hL0 : 0 ≤ L)
    (hl : l = (m.elim 0 (fun r => Real.exp (-r))) * L)
    (hacc : acc * L = T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0)
    (hLpos : 0 < L → l ≠ 0) :
    let st := mistralBlockStepBot (m, l, acc) block
    let L' := L + (block.map (fun p => Real.exp p.1)).sum
    let T' := T + (block.map (fun p => Real.exp p.1 * p.2)).sum
    st.2.1 = (st.1.elim 0 (fun r => Real.exp (-r))) * L'
      ∧ st.2.2 * L' = T'
      ∧ 0 ≤ L'
      ∧ (st.1 = ⊥ → L' = 0)
      ∧ (st.1 = ⊥ → T' = 0)
      ∧ (0 < L' → st.2.1 ≠ 0) := by
  simp only [mistralBlockStepBot]
  set blockSup := (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ with hbsd
  set G := mistralGuard blockSup with hGd
  set m' := m ⊔ G with hm'd
  set α := (WithBot.realExp (WithBot.realSub m m')).unbotD 0 with hαd
  set β := (WithBot.realExp (WithBot.realSub G m')).unbotD 0 with hβd
  set lij := (block.map (fun p => Real.exp (p.1 - G.unbotD 0))).sum with hlijd
  set L' := L + (block.map (fun p => Real.exp p.1)).sum with hL'd
  set T' := T + (block.map (fun p => Real.exp p.1 * p.2)).sum with hT'd
  -- blockSup ≠ ⊥, hence G ≠ ⊥, hence m' ≠ ⊥
  have hbsne : blockSup ≠ ⊥ := fun h => hbne ((foldr_sup_coe_bot_iff block).mp h)
  have hGne : G ≠ ⊥ := mistralGuard_ne_bot hbsne
  obtain ⟨Gr, hGr⟩ : ∃ r : ℝ, G = (r : WithBot ℝ) := by
    cases hG : G with
    | bot => exact absurd hG hGne
    | coe r => exact ⟨r, rfl⟩
  have hGunbot : G.unbotD 0 = Gr := by rw [hGr]; rfl
  obtain ⟨mr, hmr⟩ : ∃ r : ℝ, m' = (r : WithBot ℝ) := by
    cases hm : m with
    | bot => exact ⟨Gr, by rw [hm'd, hm, hGr]; rfl⟩
    | coe a => exact ⟨max a Gr, by rw [hm'd, hm, hGr, ← WithBot.coe_max]⟩
  have hm'unbot : m'.unbotD 0 = mr := by rw [hmr]; rfl
  have hκm' : m'.elim 0 (fun r => Real.exp (-r)) = Real.exp (-mr) := by rw [hmr]; rfl
  -- α·l = exp(-mr)·L
  have hαl : α * l = Real.exp (-mr) * L := by
    cases hm : m with
    | bot => have : L = 0 := hmL hm; rw [hl, hm]; simp [this]
    | coe a =>
      have hαv : α = Real.exp (a - mr) := by
        simp only [hαd, hm, hmr, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
      rw [hαv, hl, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => Real.exp (-r))) = Real.exp (-a) from rfl]
      rw [show Real.exp (a - mr) * (Real.exp (-a) * L) = (Real.exp (a - mr) * Real.exp (-a)) * L from by ring,
        ← Real.exp_add]
      ring_nf
  -- β·lij = exp(-mr)·Σexp(score)  (G cancels)
  have hβlij : β * lij = Real.exp (-mr) * (block.map (fun p => Real.exp p.1)).sum := by
    have hr := osNormStepBot_blockSum_rescale block mr (fun _ => 1)
    simp only [mul_one] at hr
    -- hr : (realExp (realSub blockSup ↑mr)).unbotD 0 * Σ exp(p.1 - blockSup.unbotD 0)
    --        = exp(-mr) * Σ exp(p.1)
    -- but we use G, not blockSup. Show directly:
    have hβv : β = Real.exp (Gr - mr) := by
      simp only [hβd, hGr, hmr, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
    rw [hβv, hlijd, hGunbot]
    rw [show (block.map (fun p => Real.exp (p.1 - Gr))).sum
          = Real.exp (-Gr) * (block.map (fun p => Real.exp p.1)).sum from by
      rw [← List.sum_map_mul_left]
      congr 1; apply List.map_congr_left; intro p _
      rw [show p.1 - Gr = -Gr + p.1 from by ring, Real.exp_add]]
    rw [show Real.exp (Gr - mr) * (Real.exp (-Gr) * (block.map (fun p => Real.exp p.1)).sum)
          = (Real.exp (Gr - mr) * Real.exp (-Gr)) * (block.map (fun p => Real.exp p.1)).sum from by ring,
      ← Real.exp_add]
    ring_nf
  set l' := α * l + β * lij with hl'd
  have hl'eq : l' = Real.exp (-mr) * L' := by
    rw [hl'd, hαl, hβlij, hL'd]; ring
  have hL'strict : 0 < L' := by
    rw [hL'd]
    have hsumpos : 0 < (block.map (fun p => Real.exp p.1)).sum := by
      rcases block with _ | ⟨a, t⟩
      · exact absurd rfl hbne
      · rw [List.map_cons, List.sum_cons]
        have h1 : 0 ≤ (t.map (fun p => Real.exp p.1)).sum := by
          apply List.sum_nonneg; intro x hx
          simp only [List.mem_map] at hx; obtain ⟨p, _, rfl⟩ := hx; exact le_of_lt (Real.exp_pos _)
        have := Real.exp_pos a.1; linarith
    linarith
  have hL'pos : 0 ≤ L' := le_of_lt hL'strict
  have hl'ne : l' ≠ 0 := by rw [hl'eq]; positivity
  -- acc' * L' = T'
  set acc' := acc * (l / l' * α)
    + (block.map (fun p => Real.exp (p.1 - G.unbotD 0) * β / l' * p.2)).sum with hacc'd
  have hacc'eq : acc' * L' = T' := by
    rw [hacc'd, add_mul]
    have hβsum : (block.map (fun p => Real.exp (p.1 - G.unbotD 0) * β / l' * p.2)).sum
        = (Real.exp (-mr) * (block.map (fun p => Real.exp p.1 * p.2)).sum) / l' := by
      have hmapeq : (block.map (fun p => Real.exp (p.1 - G.unbotD 0) * β / l' * p.2))
          = (block.map (fun p => (Real.exp (p.1 - G.unbotD 0) * (β * p.2)) * (1 / l'))) := by
        apply List.map_congr_left; intro p _; ring
      rw [hmapeq, List.sum_map_mul_right, ← div_eq_mul_one_div]
      congr 1
      rw [show (block.map (fun p => Real.exp (p.1 - G.unbotD 0) * (β * p.2))).sum
            = β * (block.map (fun p => Real.exp (p.1 - G.unbotD 0) * p.2)).sum from by
        rw [← List.sum_map_mul_left]; congr 1; apply List.map_congr_left; intro p _; ring]
      -- β * Σ exp(score - Gr)·v = exp(-mr) * Σ exp(score)·v
      have hβv : β = Real.exp (Gr - mr) := by
        simp only [hβd, hGr, hmr, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
      rw [hβv, hGunbot]
      rw [show (block.map (fun p => Real.exp (p.1 - Gr) * p.2)).sum
            = Real.exp (-Gr) * (block.map (fun p => Real.exp p.1 * p.2)).sum from by
        rw [← List.sum_map_mul_left]; congr 1; apply List.map_congr_left; intro p _
        rw [show p.1 - Gr = -Gr + p.1 from by ring, Real.exp_add]; ring]
      rw [show Real.exp (Gr - mr) * (Real.exp (-Gr) * (block.map (fun p => Real.exp p.1 * p.2)).sum)
            = (Real.exp (Gr - mr) * Real.exp (-Gr)) * (block.map (fun p => Real.exp p.1 * p.2)).sum from by ring,
        ← Real.exp_add]
      ring_nf
    rw [hβsum]
    have e1 : acc * (l / l' * α) * L' = T := by
      have hL'ne : (L' : ℝ) ≠ 0 := ne_of_gt hL'strict
      have hexpne : Real.exp (-mr) ≠ 0 := Real.exp_ne_zero _
      rw [show acc * (l / l' * α) * L' = acc * (α * l) * L' / l' from by ring, hαl, hl'eq]
      rw [show acc * (Real.exp (-mr) * L) * L' / (Real.exp (-mr) * L')
          = acc * L * (Real.exp (-mr) * L' / (Real.exp (-mr) * L')) from by ring]
      rw [div_self (mul_ne_zero hexpne hL'ne), mul_one, hacc]
    have e2 : Real.exp (-mr) * (block.map (fun p => Real.exp p.1 * p.2)).sum / l' * L'
        = (block.map (fun p => Real.exp p.1 * p.2)).sum := by
      have hL'ne : (L' : ℝ) ≠ 0 := ne_of_gt hL'strict
      have hexpne : Real.exp (-mr) ≠ 0 := Real.exp_ne_zero _
      rw [hl'eq]; field_simp
    rw [e1, e2, hT'd]
  -- assemble
  refine ⟨?_, hacc'eq, hL'pos, ?_, ?_, ?_⟩
  · -- l' = κ(m')·L'
    show l' = (m'.elim 0 (fun r => Real.exp (-r))) * L'
    rw [hκm', hl'eq]
  · intro hbot; exact absurd (hmr ▸ hbot) (by simp)
  · intro hbot; exact absurd (hmr ▸ hbot) (by simp)
  · intro _; show l' ≠ 0; exact hl'ne

end MistralBlockStep





section MistralFold

/-- Masked fold by BLOCK COUNT: fold `mistralBlockStepBot` over the first `c`
row-masked sentinel-kept blocks. -/
noncomputable def mistralFoldUptoG
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d : Fin DM) : WithBot ℝ × ℝ × ℝ :=
  ((List.range c).map (fun cc =>
      mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw cc i d)).foldl
    mistralBlockStepBot (⊥, 0, 0)

theorem mistralFoldUptoG_zero
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw : Nat) (i : Fin BLK) (d : Fin DM) :
    mistralFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw 0 i d = (⊥, 0, 0) := by
  simp [mistralFoldUptoG]

theorem mistralFoldUptoG_succ
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d : Fin DM) :
    mistralFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw (c + 1) i d
      = mistralBlockStepBot
          (mistralFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d)
          (mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d) := by
  simp only [mistralFoldUptoG, List.range_succ, List.map_append, List.map_cons, List.map_nil,
    List.foldl_append, List.foldl_cons, List.foldl_nil]

/-- Threshold split on a `.val`-ascending `Fin` list under a `< hi` window. -/
private theorem mistral_filterMap_threshold_split {n : Nat} (l : List (Fin n))
    (hsorted : l.Pairwise (fun a b => a.val < b.val))
    (t hi₂ : Nat) (g : Fin n → ℝ × ℝ) (hle : t ≤ hi₂) :
    l.filterMap (fun j => if j.val < hi₂ then some (g j) else none)
      = l.filterMap (fun j => if j.val < t then some (g j) else none)
        ++ l.filterMap (fun j => if t ≤ j.val ∧ j.val < hi₂ then some (g j) else none) := by
  induction l with
  | nil => simp
  | cons a tl ih =>
    have htl : tl.Pairwise (fun x y => x.val < y.val) := (List.pairwise_cons.mp hsorted).2
    have hahead : ∀ b ∈ tl, a.val < b.val := (List.pairwise_cons.mp hsorted).1
    rw [List.filterMap_cons, List.filterMap_cons, List.filterMap_cons]
    by_cases hlt : a.val < t
    · rw [ih htl, if_pos (lt_of_lt_of_le hlt hle), if_pos hlt,
        if_neg (fun h : t ≤ a.val ∧ a.val < hi₂ => by omega)]
      rfl
    · have hge : t ≤ a.val := Nat.not_lt.mp hlt
      have htail_prefix : tl.filterMap (fun j => if j.val < t then some (g j) else none) = [] := by
        apply List.filterMap_eq_nil_iff.mpr
        intro b hb
        have := hahead b hb
        rw [if_neg (fun h : b.val < t => by omega)]
      rw [ih htl, htail_prefix, if_neg hlt]
      by_cases h2 : a.val < hi₂
      · rw [if_pos h2, if_pos ⟨hge, h2⟩]; rfl
      · rw [if_neg h2, if_neg (fun h => h2 h.2)]

theorem mistralBlocks_concat_map_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw nb : Nat) (i : Fin BLK) (d : Fin DM) (g : ℝ × ℝ → ℝ)
    (hnb : nb * BLK = S) :
    (((List.range nb).map (fun cc =>
        mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw cc i d)).flatten.map g).sum
      = ∑ j : Fin S, g (mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j,
            ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit)) := by
  classical
  set kf : Fin S → ℝ × ℝ := fun j =>
    (mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j,
     ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit)) with hkf
  have hblk : ∀ cc, mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw cc i d
      = (List.finRange S).filterMap (fun j : Fin S =>
          if cc * BLK ≤ j.val ∧ j.val < (cc + 1) * BLK then some (kf j) else none) := by
    intro cc; rfl
  have hflat : ((List.range nb).map (fun cc =>
        mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw cc i d)).flatten
      = (List.finRange S).filterMap (fun j : Fin S =>
          if j.val < nb * BLK then some (kf j) else none) := by
    simp only [hblk]
    clear hnb
    induction nb with
    | zero => simp
    | succ n ih =>
      rw [List.range_succ, List.map_append, List.map_cons, List.map_nil, List.flatten_append,
        List.flatten_cons, List.flatten_nil, List.append_nil, ih,
        mistral_filterMap_threshold_split (List.finRange S) (List.pairwise_lt_finRange S)
          (n * BLK) ((n + 1) * BLK) kf (by nlinarith [Nat.zero_le BLK])]
  rw [hflat]
  rw [show (fun j : Fin S => if j.val < nb * BLK then some (kf j) else none)
        = (fun j : Fin S => some (kf j)) from by
    funext j; rw [if_pos (by rw [hnb]; exact j.isLt)]]
  rw [show (List.finRange S).filterMap (fun j : Fin S => some (kf j))
        = (List.finRange S).map kf from
    List.filterMap_eq_map_iff_forall_eq_some.mpr fun x => congrFun rfl]
  rw [List.map_map, ← List.sum_ofFn]
  congr 1
  rw [List.ofFn_eq_map]
  rfl

end MistralFold


section MistralAnchor

/-- Block `c` is non-empty when fully in-window (`(c+1)·BLK ≤ S`). -/
theorem mistralBlockMG_ne_nil
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d : Fin DM)
    (hwin : (c + 1) * BLK ≤ S) (hBLK : 0 < BLK) :
    mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d ≠ [] := by
  intro hnil
  have h0 : (⟨c * BLK, by nlinarith [Nat.zero_le BLK]⟩ : Fin S) ∈ List.finRange S := List.mem_finRange _
  have := List.filterMap_eq_nil_iff.mp hnil (⟨c * BLK, by nlinarith [Nat.zero_le BLK]⟩ : Fin S) h0
  simp only [Fin.val_mk] at this
  rw [if_pos ⟨by omega, by nlinarith⟩] at this
  exact absurd this (by simp)

/-- **Anchor for `mistralFoldUptoG`.** After folding `c` (fully-in-window) blocks,
the running `(l, acc)` are consistent with the batch denominator `L` and numerator
`T` of the flattened first-`c`-block key stream. -/
theorem mistralFoldUptoG_anchor
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d : Fin DM)
    (hBLK : 0 < BLK) (hc : c * BLK ≤ S) :
    let blocks := ((List.range c).map (fun cc =>
        mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw cc i d)).flatten
    let st := mistralFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d
    let L := (blocks.map (fun p => Real.exp p.1)).sum
    let T := (blocks.map (fun p => Real.exp p.1 * p.2)).sum
    st.2.1 = (st.1.elim 0 (fun r => Real.exp (-r))) * L
      ∧ st.2.2 * L = T
      ∧ 0 ≤ L
      ∧ (st.1 = ⊥ → L = 0)
      ∧ (st.1 = ⊥ → T = 0)
      ∧ (0 < L → st.2.1 ≠ 0) := by
  induction c with
  | zero =>
    simp only [List.range_zero, List.map_nil, List.flatten_nil, List.map_nil, List.sum_nil,
      mistralFoldUptoG_zero]
    refine ⟨by simp, by simp, le_refl 0, ?_, ?_, fun h => absurd h (lt_irrefl 0)⟩
    · intro _; trivial
    · intro _; trivial
  | succ n ih =>
    have hcn : n * BLK ≤ S := by nlinarith [Nat.zero_le BLK]
    obtain ⟨hl, hacc, hL0, hbotL, hbotT, hne⟩ := ih hcn
    simp only [mistralFoldUptoG_succ, List.range_succ, List.map_append, List.map_cons,
      List.map_nil, List.flatten_append, List.flatten_cons, List.flatten_nil, List.append_nil]
    set fold0 := mistralFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw n i d with hf0
    set blk := mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw n i d with hblk
    have hbne : blk ≠ [] := mistralBlockMG_ne_nil s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw n i d hc hBLK
    have hcons := mistralBlockStepBot_consistent fold0.1 fold0.2.1 fold0.2.2
      ((((List.range n).map (fun cc => mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw cc i d)).flatten.map
          (fun p => Real.exp p.1 * p.2)).sum)
      ((((List.range n).map (fun cc => mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw cc i d)).flatten.map
          (fun p => Real.exp p.1)).sum)
      blk hbne hL0 hl hacc hbotL hbotT hne
    -- rewrite the goal's prefix-fold and flattened sums
    rw [List.sum_append, List.sum_append]
    exact hcons

end MistralAnchor


section MistralBridge

/-- On an active row (`gi < bel`) the row-masked score equals the genuine score
(the row mask `ctxQTileMRowG` is the identity). -/
theorem mistralScoreM_eq_score_active
    (s : BlockState) (Q K B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw : Nat) (i : Fin BLK) (j : Fin S)
    (hact : s.pids 2 * BLK + i.val < bel) :
    mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j
      = mistralScore s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j := by
  unfold mistralScoreM mistralScore
  by_cases hj : mistralActive (s.pids 2 * BLK + i.val) sw j.val
  · rw [if_pos hj, if_pos hj]
    refine congrArg (sm_scale * ·) (Finset.sum_congr rfl (fun e _ => ?_))
    rw [show ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM bel (i, e, PUnit.unit)
          = ctxQTileG s Q B_Start_Loc rs hs BLK DM (i, e, PUnit.unit) from by
      simp only [ctxQTileMRowG, hact, if_true]]
  · rw [if_neg hj, if_neg hj]

/-- The all-block flatten denominator is strictly positive: every key in `Fin S` is
present with a finite score (real or the `-1e9` sentinel), so `exp(score) > 0`. -/
theorem mistralBlocks_denom_pos
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw nb : Nat) (i : Fin BLK) (d : Fin DM)
    (hnb : nb * BLK = S) (hS : 0 < S) :
    0 < (((List.range nb).map (fun cc =>
        mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw cc i d)).flatten.map
          (fun p => Real.exp p.1)).sum := by
  rw [mistralBlocks_concat_map_sum s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw nb i d
    (fun p => Real.exp p.1) hnb]
  apply Finset.sum_pos
  · intro j _; exact Real.exp_pos _
  · exact ⟨⟨0, hS⟩, Finset.mem_univ _⟩

/-- **Fold = genuine closed form (active row, full window).** At the full block count
`nb` (`nb·BLK = S`), on an active row the streamed block-fold accumulator equals the
genuine boundary-masked sliding-window softmax closed form. -/
theorem mistralFoldUptoG_full_eq_genuine
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw nb : Nat) (idx : TileIndex [BLK, DM])
    (hBLK : 0 < BLK) (hnb : nb * BLK = S) (hS : 0 < S)
    (hact : s.pids 2 * BLK + idx.1.val < bel) :
    (mistralFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw nb idx.1 idx.2.1).2.2
      = contextAttnMistralExactFoldMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw idx := by
  obtain ⟨i, d, u⟩ := idx
  have hc : nb * BLK ≤ S := le_of_eq hnb
  obtain ⟨_hl, hacc, _hL0, _hbL, _hbT, _hne⟩ :=
    mistralFoldUptoG_anchor s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw nb i d hBLK hc
  -- the flattened all-block denom L and numerator T
  set L := (((List.range nb).map (fun cc =>
      mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw cc i d)).flatten.map
        (fun p => Real.exp p.1)).sum with hLd
  set T := (((List.range nb).map (fun cc =>
      mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw cc i d)).flatten.map
        (fun p => Real.exp p.1 * p.2)).sum with hTd
  have hLpos : 0 < L := mistralBlocks_denom_pos s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw nb i d hnb hS
  -- rewrite L and T as Fin-S sums
  have hLsum : L = Finset.univ.sum (fun j : Fin S =>
      Real.exp (mistralScore s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j)) := by
    rw [hLd, mistralBlocks_concat_map_sum s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw nb i d
      (fun p => Real.exp p.1) hnb]
    apply Finset.sum_congr rfl; intro j _
    rw [mistralScoreM_eq_score_active s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j hact]
  have hTsum : T = Finset.univ.sum (fun j : Fin S =>
      Real.exp (mistralScore s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j)
        * ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit)) := by
    rw [hTd, mistralBlocks_concat_map_sum s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw nb i d
      (fun p => Real.exp p.1 * p.2) hnb]
    apply Finset.sum_congr rfl; intro j _
    rw [mistralScoreM_eq_score_active s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j hact]
  -- closed form
  rw [contextAttnMistralExactFoldMG]
  simp only
  rw [← hLsum, ← hTsum, eq_div_iff (ne_of_gt hLpos)]
  -- hacc : st.2.2 * L = T  (modulo the let-bindings)
  exact hacc

end MistralBridge


section MistralGenuineValue

/-- **Genuine closed-form output value at the Python test shape.** This is the
boundary-masked sliding-window causal-softmax fold of the *loaded* Q/K/V memory
(keeping the `-1e9` sentinel exactly on causally/window-masked cells), NOT a
kernel readback. Strides `(768, 128, 1)`, `BLOCK_M = BLOCK_N = BLOCK_DMODEL = 128`,
`sm_scale = (√128)⁻¹`, and the run-specific `sliding_window`. -/
noncomputable def ctxMistralGenuineOutValue
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName)
    (sliding_window : Nat) (idx : TileIndex [128, 128]) : ℝ :=
  mistralGenuineOutValueG s Q K V B_Start_Loc B_Seqlen
    ((Real.sqrt (128 : ℝ))⁻¹) 768 128 128 128 sliding_window idx

end MistralGenuineValue


section MistralLaneBridges

/-- Generic filterMap-finRange sum (re-port; nopad's is private/inaccessible). -/
theorem mistral_filterMap_finRange_sum {α : Type*} (n : Nat)
    (p : Fin n → Prop) [DecidablePred p] (g : Fin n → α) (h : α → ℝ) :
    (((List.finRange n).filterMap (fun j => if p j then some (g j) else none)).map h).sum
      = ∑ j : Fin n, if p j then h (g j) else 0 := by
  rw [List.map_filterMap]
  rw [show (fun j : Fin n => Option.map h (if p j then some (g j) else none))
        = (fun j : Fin n => if p j then some (h (g j)) else none) from by
    funext j; by_cases hj : p j <;> simp [hj]]
  rw [show (((List.finRange n).filterMap (fun j => if p j then some (h (g j)) else none))).sum
        = ((List.finRange n).map (fun j => if p j then h (g j) else 0)).sum from by
    induction (List.finRange n) with
    | nil => simp
    | cons a t ih => by_cases ha : p a <;> simp [ha, ih]]
  rw [← List.sum_ofFn]; congr 1; rw [List.ofFn_eq_map]

/-- Generic filterMap foldr-sup (re-port). -/
theorem mistral_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
    (sc : Fin n → ℝ) :
    (((List.finRange n).filterMap (fun j => if P j then some (sc j) else none)).map
        (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun j : Fin n => if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) := by
  rw [show (((List.finRange n).filterMap (fun j => if P j then some (sc j) else none)).map
        (fun x => ((x : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = (List.finRange n).foldr (fun j a => (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) ⊔ a) ⊥ from by
    induction (List.finRange n) with
    | nil => simp
    | cons a t ih => by_cases ha : P a <;> simp [ha, ih]]
  apply le_antisymm
  · induction (List.finRange n) with
    | nil => simp
    | cons a t ih =>
      simp only [List.foldr_cons]
      exact sup_le (Finset.le_sup (f := fun j : Fin n => if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥)
        (Finset.mem_univ a)) ih
  · apply Finset.sup_le
    intro j _
    have key : ∀ (l : List (Fin n)), j ∈ l →
        (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥)
          ≤ l.foldr (fun j a => (if P j then ((sc j : ℝ) : WithBot ℝ) else ⊥) ⊔ a) ⊥ := by
      intro l hl
      induction l with
      | nil => simp at hl
      | cons a t ih =>
        simp only [List.foldr_cons]
        rcases List.mem_cons.mp hl with h | h
        · subst h; exact le_sup_left
        · exact le_trans (ih h) le_sup_right
    exact key _ (List.mem_finRange j)

/-- Reindex a window `Finset.sup` over `Fin S` (`c·BLK ≤ j < (c+1)·BLK`) onto
`Fin BLK` lanes (lane `jL` ↦ key `c·BLK + jL`). Every lane present (no causal/window
drop: the sentinel is carried in the score `F`). -/
private theorem mistral_window_sup_reindexG (c S BLK : Nat) (hwin : (c + 1) * BLK ≤ S)
    (F : Nat → WithBot ℝ) :
    Finset.univ.sup (fun j : Fin S =>
        if c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then F j.val else ⊥)
      = Finset.univ.sup (fun jL : Fin BLK => F (c * BLK + jL.val)) := by
  apply le_antisymm
  · apply Finset.sup_le
    intro j _
    by_cases hj : c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK
    · rw [if_pos hj]
      have hjL : j.val - c * BLK < BLK := by
        have : j.val < c * BLK + BLK := by have := hj.2; nlinarith [Nat.zero_le BLK]
        omega
      refine le_trans ?_ (Finset.le_sup
        (f := fun jL : Fin BLK => F (c * BLK + jL.val))
        (Finset.mem_univ (⟨j.val - c * BLK, hjL⟩ : Fin BLK)))
      simp only [show c * BLK + (j.val - c * BLK) = j.val from by omega, le_refl]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le
    intro jL _
    have hb : c * BLK + jL.val < S := by have := jL.isLt; nlinarith [Nat.zero_le BLK]
    refine le_trans ?_ (Finset.le_sup
      (f := fun j : Fin S => if c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then F j.val else ⊥)
      (Finset.mem_univ (⟨c * BLK + jL.val, hb⟩ : Fin S)))
    have := jL.isLt
    simp only
    rw [if_pos (show c * BLK ≤ c * BLK + jL.val ∧ c * BLK + jL.val < (c + 1) * BLK from
      ⟨by omega, by nlinarith⟩)]

/-- Lane bound. -/
theorem mistral_lane_lt_SG (c S BLK : Nat) (hwin : (c + 1) * BLK ≤ S) (jL : Fin BLK) :
    c * BLK + jL.val < S := by have := jL.isLt; nlinarith [Nat.zero_le BLK]

end MistralLaneBridges


section MistralBlockBridges

/-- `mistralBlockMG` map-and-sum: reindex the window `[c·BLK,(c+1)·BLK) ⊆ Fin S`
onto lanes `jL : Fin BLK`. Every lane present (the sentinel is inside the score). -/
theorem mistralBlockMG_map_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d : Fin DM)
    (hwin : (c + 1) * BLK ≤ S) (h : ℝ × ℝ → ℝ) :
    ((mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d).map h).sum
      = ∑ jL : Fin BLK,
          h (mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i
              ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩,
            ctxVTileMG s V B_Start_Loc rs hs S DM bel
              (⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩, d, PUnit.unit)) := by
  classical
  rw [mistralBlockMG, mistral_filterMap_finRange_sum S
    (fun j : Fin S => c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK)
    (fun j => (mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j,
              ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit))) h]
  rw [← Finset.sum_filter
        (fun j : Fin S => c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK)
        (f := fun j : Fin S => h (mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j,
              ctxVTileMG s V B_Start_Loc rs hs S DM bel (j, d, PUnit.unit)))]
  symm
  refine Finset.sum_bij'
    (i := fun (jL : Fin BLK) (_ : jL ∈ (Finset.univ : Finset (Fin BLK))) =>
      (⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩ : Fin S))
    (j := fun (j : Fin S) (_ : j ∈ Finset.univ.filter
        (fun j : Fin S => c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK)) =>
      (⟨j.val - c * BLK, by
          have hj := (Finset.mem_filter.mp ‹_›).2
          have : j.val < c * BLK + BLK := by nlinarith [Nat.zero_le BLK, hj.2]
          omega⟩ : Fin BLK))
    ?_ ?_ ?_ ?_ ?_
  · intro jL _
    rw [Finset.mem_filter]
    have hjL := jL.isLt
    refine ⟨Finset.mem_univ _, ?_, ?_⟩
    · show c * BLK ≤ c * BLK + jL.val; omega
    · show c * BLK + jL.val < (c + 1) * BLK; nlinarith
  · intro j _; exact Finset.mem_univ _
  · intro jL _; apply Fin.ext
    show c * BLK + jL.val - c * BLK = jL.val
    omega
  · intro j hj
    have hj2 := (Finset.mem_filter.mp hj).2
    have hjlt : j.val < c * BLK + BLK := by nlinarith [Nat.zero_le BLK, hj2.2]
    apply Fin.ext
    show c * BLK + (j.val - c * BLK) = j.val
    omega
  · intro jL _; rfl

/-- `mistralBlockMG` running-sup bridge: the lane-`Fin BLK` sup of the coerced
`mistralScoreM` scores equals the `WithBot ⊔`-foldr of the block's coerced scores. -/
theorem mistralBlockMG_sup_eq
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d : Fin DM)
    (hwin : (c + 1) * BLK ≤ S) :
    Finset.univ.sup (fun jL : Fin BLK =>
        ((mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i
            ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩ : ℝ) : WithBot ℝ))
      = ((mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d).map
          (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  classical
  set F : Nat → WithBot ℝ := fun jg =>
    if h : jg < S then
      ((mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i ⟨jg, h⟩ : ℝ) : WithBot ℝ)
    else ⊥ with hF
  rw [show (mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
        = ((List.finRange S).filterMap (fun j : Fin S =>
            if c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then
              some (mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j)
            else none)).map (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold mistralBlockMG
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK <;> simp [hj]]
  rw [mistral_filterMap_foldr_sup S
    (fun j => c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK)
    (fun j => mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j)]
  rw [show (Finset.univ.sup (fun j : Fin S =>
        if c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then
          ((mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i j : ℝ) : WithBot ℝ)
        else ⊥))
      = Finset.univ.sup (fun j : Fin S =>
          if c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK then F j.val else ⊥) from by
    apply Finset.sup_congr rfl
    intro j _
    by_cases hw : c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK
    · rw [if_pos hw, if_pos hw, hF]; simp only [dif_pos j.isLt]
    · rw [if_neg hw, if_neg hw]]
  rw [mistral_window_sup_reindexG c S BLK hwin F]
  apply Finset.sup_congr rfl
  intro jL _
  have hb : c * BLK + jL.val < S := mistral_lane_lt_SG c S BLK hwin jL
  rw [hF]; simp only [dif_pos hb]

end MistralBlockBridges


section MistralLijAccBridges

/-- `mistralBlockMG` `l_ij` lane-sum bridge: the kernel's per-lane
`realExp(realSub (coe scoreM) (coe Gr))` summed over `Fin BLK` equals
`some ((blk.map (exp(p.1 − Gr))).sum)`. Every lane is a real coe (sentinel kept). -/
theorem mistralBlockMG_lij_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d : Fin DM) (Gr : ℝ)
    (hwin : (c + 1) * BLK ≤ S) :
    (∑ jL : Fin BLK, WithBot.realExp (WithBot.realSub
        ((mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i
            ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩ : ℝ) : WithBot ℝ)
        ((Gr : ℝ) : WithBot ℝ)))
      = some ((mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d).map
          (fun p => Real.exp (p.1 - Gr))).sum := by
  have hcell : ∀ jL : Fin BLK,
      WithBot.realExp (WithBot.realSub
        ((mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i
            ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩ : ℝ) : WithBot ℝ)
        ((Gr : ℝ) : WithBot ℝ))
        = some (Real.exp ((mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i
            ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩) - Gr)) := by
    intro jL
    rw [WithBot.realSub_coe_coe, WithBot.realExp_coe]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [mistralBlockMG_map_sum s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d hwin
    (fun p => Real.exp (p.1 - Gr))]

/-- `mistralBlockMG` `acc` lane-sum bridge. -/
theorem mistralBlockMG_acc_sum
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d : Fin DM) (Gr : ℝ)
    (hwin : (c + 1) * BLK ≤ S)
    (rawV : Fin BLK → ℝ)
    (hrawV : ∀ jL : Fin BLK,
      rawV jL = ctxVTileMG s V B_Start_Loc rs hs S DM bel
        (⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩, d, PUnit.unit)) :
    (∑ jL : Fin BLK, WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          ((mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i
              ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩ : ℝ) : WithBot ℝ)
          ((Gr : ℝ) : WithBot ℝ)))
        ((rawV jL : ℝ) : WithBot ℝ))
      = some ((mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d).map
          (fun p => Real.exp (p.1 - Gr) * p.2)).sum := by
  have hcell : ∀ jL : Fin BLK,
      WithBot.realMul
        (WithBot.realExp (WithBot.realSub
          ((mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i
              ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩ : ℝ) : WithBot ℝ)
          ((Gr : ℝ) : WithBot ℝ)))
        ((rawV jL : ℝ) : WithBot ℝ)
        = some (Real.exp ((mistralScoreM s Q K B_Start_Loc sm_scale rs hs BLK DM S bel sw i
            ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩) - Gr)
              * ctxVTileMG s V B_Start_Loc rs hs S DM bel
                  (⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩, d, PUnit.unit)) := by
    intro jL
    rw [WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.realMul_coe_coe, hrawV jL]; rfl
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [mistralBlockMG_map_sum s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d hwin
    (fun p => Real.exp (p.1 - Gr) * p.2)]

end MistralLijAccBridges


section MistralExec

/-! ## Mistral genuine exec-stepping stack (test shape, BLOCK=128)

Connects the kernel's DSL loop execution to the genuine block-fold foundation,
landing the final `Out` store on `ctxMistralGenuineOutValue`. Mirrors the
`context_attn_nopad` exec stack, with the Mistral deltas: a `cur_kv_head =
cur_head // 1` preLoop statement, the two causal+sliding-window `tl.where`s
against the finite `-1e9` sentinel, the `m_ij == -1e9 → 0` block-max guard, and
the `l_i_new == 0 → 1e-9` denominator guard. The block-step semantics is
`mistralBlockStepBot` (per-`BLOCK_N`-block), matched against the kernel registers
through the `mistralBlockMG_{sup,lij,acc}` lane bridges. -/

/-- `sm_scale` at the Python test shape. -/
noncomputable def mistral_sm_scale : ℝ := (Real.sqrt (128 : ℝ))⁻¹

/-- Eval helper for `floorDiv` (the `cur_kv_head = cur_head // kv_group_num`). -/
theorem mistral_evalOp_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

def mistralLoopBody (sc : ℝ) (sw : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    Stmt.assign .real [128, 128] "k"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 128] "k_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat 768))))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consSame.consL.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))),
    Stmt.assign .real [128, 128] "qk" (Op.full [128, 128] (Op.const 0)),
    Stmt.assign .real [128, 128] "qk"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [128, 128] "qk")
        (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k"))),
    Stmt.assign .real [128, 128] "qk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 128] "qk") (Op.const sc)),
    Stmt.assign .real [128, 128] "qk"
      ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))).where
        (Op.ref .real [128, 128] "qk")
        ((Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9)).broadcast [128, 128])),
    Stmt.assign .real [128, 128] "qk"
      ((Op.gt ComparableDType.nat Broadcast.nil.consR.consL
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
            (Op.sub .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
              (Op.constNat sw))).where
        (Op.ref .real [128, 128] "qk")
        ((Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9)).broadcast [128, 128])),
    Stmt.assign .real [128] "m_ij" (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk")),
    Stmt.assign .real [128] "m_ij"
      ((Op.eq ComparableDType.real Broadcast.scalarR (Op.ref .real [128] "m_ij")
            (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9))).where
        ((Op.const 0.0).broadcast [128]) (Op.ref .real [128] "m_ij")),
    Stmt.assign .real [128, 128] "p"
      (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "qk")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij"))).exp,
    Stmt.assign .real [128] "l_ij" (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "p")),
    Stmt.assign .real [128] "m_i_new"
      ((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref .real [128] "m_i")
            (Op.ref .real [128] "m_ij")).where
        (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij")),
    Stmt.assign .real [128] "alpha"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_i")
          (Op.ref .real [128] "m_i_new")).exp,
    Stmt.assign .real [128] "beta"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_ij")
          (Op.ref .real [128] "m_i_new")).exp,
    Stmt.assign .real [128] "l_i_new"
      (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "alpha")
          (Op.ref .real [128] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "beta")
          (Op.ref .real [128] "l_ij"))),
    Stmt.assign .real [128] "l_i_new"
      ((Op.eq ComparableDType.real Broadcast.scalarR (Op.ref .real [128] "l_i_new")
            (Op.const 0.0)).where
        ((Op.const 1e-9).broadcast [128]) (Op.ref .real [128] "l_i_new")),
    Stmt.assign .real [128] "p_scale"
      (Op.div .real Broadcast.nil.consSame (Op.ref .real [128] "beta")
        (Op.ref .real [128] "l_i_new")),
    Stmt.assign .real [128, 128] "p"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "p_scale"))),
    Stmt.assign .real [128] "acc_scale"
      (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [128] "l_i")
          (Op.ref .real [128] "l_i_new"))
        (Op.ref .real [128] "alpha")),
    Stmt.assign .real [128, 128] "acc"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "acc_scale"))),
    Stmt.assign .real [128, 128] "v"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 128] "v_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat 768))))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))),
    Stmt.assign .real [128, 128] "p" (Op.ref .real [128, 128] "p"),
    Stmt.assign .real [128, 128] "acc"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [128, 128] "acc")
        (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v"))),
    Stmt.assign .real [128] "l_i" (Op.ref .real [128] "l_i_new"),
    Stmt.assign .real [128] "m_i" (Op.ref .real [128] "m_i_new") ]

end MistralExec


section MistralOpEvals

/-! ## Per-statement op-eval recipes (test shape). Reused from `context_attn_nopad`
(re-ported with `mistral_` prefix), plus the four Mistral-specific recipes for the
two `-1e9` `where`s and the two guards. -/

set_option maxHeartbeats 1600000 in
/-- `k` masked-load recipe. -/
theorem mistral_k_load_eval (s : BlockState) (K B_Start_Loc B_Seqlen : RegionName) (SN : Nat)
    (hkp : s.regs .ptr [128, 128] "k_ptrs" =
      some (⟨fun idx : TileIndex [128, 128] =>
        (K, idx.2.1.val * 768 + s.pids 1 * 128 + idx.1.val)⟩
        : Tile .ptr [128, 128]))
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s B_Start_Loc)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 128] "k_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat 768))))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consSame.consL.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          if SN + idx.2.1.val < seqLen s B_Seqlen then
            some (s.readMem K ((startLoc s B_Start_Loc + (SN + idx.2.1.val)) * 768
              + s.pids 1 * 128 + idx.1.val))
          else some (0.0 : ℝ)⟩ : Tile .real [128, 128]) := by
  have hexp : @evalOp .nat [1, 128] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin 128 => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [128] ⟨0, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hkp, hsl, hsn, hexp, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, jL, u⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim,
    Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    BlockState.readMem, TileShape.dropInsertedIndex]
  by_cases hlt : SN + jL.val < seqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMem]
    rw [show jL.val * 768 + s.pids 1 * 128 + e.val + (startLoc s B_Start_Loc + SN) * 768
        = (startLoc s B_Start_Loc + (SN + jL.val)) * 768 + s.pids 1 * 128 + e.val from by ring]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- `qk += tl.dot(q, k)` recipe (over the seeded `qk = full 0`). -/
theorem mistral_qk_dot_eval (s : BlockState) (qFn : Fin 128 → Fin 128 → ℝ)
    (kFn : Fin 128 → Fin 128 → ℝ)
    (hqk0 : s.regs .real [128, 128] "qk"
      = some (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]))
    (hq : s.regs .real [128, 128] "q"
      = some (⟨fun idx : TileIndex [128, 128] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]))
    (hk : s.regs .real [128, 128] "k"
      = some (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [128, 128] "qk")
        (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k"))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          some (Finset.univ.sum (fun e : Fin 128 => qFn idx.1 e * kFn e idx.2.1))⟩
          : Tile .real [128, 128]) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k")) s
      = some (Tile.dot [] (⟨fun idx : TileIndex [128, 128] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])
          (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])) := by
    rw [evalOp_dot]; erw [evalOp_ref, hq, evalOp_ref, hk]; rfl
  rw [evalOp_add, evalOp_ref, hqk0]
  show Option.bind (evalOp (Op.dot (batch := []) (Op.ref .real [128, 128] "q") (Op.ref .real [128, 128] "k")) s) _ = _
  rw [hdot]
  simp only [Option.bind]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ fun e : Fin 128 =>
        Option.map₂ (· * ·) ((⟨fun idx : TileIndex [128, 128] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]).data (i, e, PUnit.unit))
          ((⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]).data (e, jL, PUnit.unit)))
      = (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ fun e : Fin 128 => ((qFn i e * kFn e jL : ℝ) : WithBot ℝ)) from by
    apply Finset.sum_congr rfl; intro e _; rfl]
  rw [WithBot.sum_some_eq_some]
  show Option.map₂ (· + ·) (some 0) (some _) = _
  simp only [Option.map₂, Option.bind, Option.map, zero_add]

set_option maxHeartbeats 1600000 in
/-- `qk *= sm_scale` recipe. -/
theorem mistral_qk_scale_eval (s : BlockState) (sc : ℝ) (rawFn : Fin 128 → Fin 128 → ℝ)
    (hqk : s.regs .real [128, 128] "qk"
      = some (⟨fun idx : TileIndex [128, 128] => some (rawFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [128, 128] "qk") (Op.const sc)) s
      = some (⟨fun idx : TileIndex [128, 128] => some (rawFn idx.1 idx.2.1 * sc)⟩
          : Tile .real [128, 128]) := by
  rw [evalOp_mul, evalOp_ref, hqk, evalOp_const]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul]
  rfl

set_option maxHeartbeats 1600000 in
/-- **First `where` recipe (causal, `-1e9` sentinel).**
`qk = tl.where(offs_m[:,None] ≥ start_n+offs_n[None,:], qk, -1e9)`.  Keeps
`some (qkFn i jL)` when `SN+jL ≤ offsM i`, else the finite `some (-1e9)`. -/
theorem mistral_qk_where1_eval (s : BlockState) (SN : Nat) (qkFn : Fin 128 → Fin 128 → ℝ)
    (offsM : Fin 128 → Nat)
    (hqk : s.regs .real [128, 128] "qk"
      = some (⟨fun idx : TileIndex [128, 128] => some (qkFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]))
    (hm : s.regs .nat [128] "offs_m" = some (Tile.vec offsM))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))) :
    evalOp ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))).where
        (Op.ref .real [128, 128] "qk")
        ((Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9)).broadcast [128, 128])) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          if SN + idx.2.1.val ≤ offsM idx.1 then some (qkFn idx.1 idx.2.1)
          else some (-1e9 : ℝ)⟩ : Tile .real [128, 128]) := by
  have hexpM : @evalOp .nat [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec offsM)) :=
    evalOp_expandDim_ref_of_regs .nat [128] ⟨1, by simp⟩ "offs_m" s _ hm
  have hexpN : @evalOp .nat [1, 128] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin 128 => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [128] ⟨0, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hexpM, hsn, hexpN, hqk, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Tile.vec,
    Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.ge, NumericDType.add,
    NumericDType.sub, WithBot.realSub, TileShape.dropInsertedIndex]
  by_cases hle : SN + jL.val ≤ offsM i
  · rw [if_pos (by simp only [decide_eq_true_eq]; omega), if_pos hle]
  · rw [if_neg (by simp only [decide_eq_true_eq]; omega), if_neg hle]
    norm_num

set_option maxHeartbeats 1600000 in
/-- **Second `where` recipe (sliding window, `-1e9` sentinel).**
`qk = tl.where(start_n+offs_n[None,:] > offs_m[:,None] − sliding_window, qk, -1e9)`.
Keeps `some (gFn i jL)` (the prior cell) when `offsM i ∸ sw < SN+jL`, else `some
(-1e9)`. -/
theorem mistral_qk_where2_eval (s : BlockState) (SN sw : Nat) (gFn : Fin 128 → Fin 128 → ℝ)
    (offsM : Fin 128 → Nat)
    (hqk : s.regs .real [128, 128] "qk"
      = some (⟨fun idx : TileIndex [128, 128] => some (gFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]))
    (hm : s.regs .nat [128] "offs_m" = some (Tile.vec offsM))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))) :
    evalOp ((Op.gt ComparableDType.nat Broadcast.nil.consR.consL
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")))
            (Op.sub .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
              (Op.constNat sw))).where
        (Op.ref .real [128, 128] "qk")
        ((Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9)).broadcast [128, 128])) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          if offsM idx.1 - sw < SN + idx.2.1.val then some (gFn idx.1 idx.2.1)
          else some (-1e9 : ℝ)⟩ : Tile .real [128, 128]) := by
  have hexpM : @evalOp .nat [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec offsM)) :=
    evalOp_expandDim_ref_of_regs .nat [128] ⟨1, by simp⟩ "offs_m" s _ hm
  have hexpN : @evalOp .nat [1, 128] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin 128 => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [128] ⟨0, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hexpM, hsn, hexpN, hqk, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Tile.vec,
    Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, NumericDType.add,
    NumericDType.sub, WithBot.realSub, TileShape.dropInsertedIndex]
  by_cases hlt : offsM i - sw < SN + jL.val
  · rw [if_pos (by simp only [decide_eq_true_eq]; omega), if_pos hlt]
  · rw [if_neg (by simp only [decide_eq_true_eq]; omega), if_neg hlt]
    norm_num

set_option maxHeartbeats 1600000 in
/-- **`m_ij` guard recipe.** `m_ij = tl.where(m_ij == -1e9, 0.0, m_ij)`.
On a finite (`≠ ⊥`) block max it lands on `mistralGuard (mijFn i)`. -/
theorem mistral_mij_guard_eval (s : BlockState) (mijFn : Fin 128 → WithBot ℝ)
    (hmij : s.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128])) :
    evalOp ((Op.eq ComparableDType.real Broadcast.scalarR (Op.ref .real [128] "m_ij")
            (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9))).where
        ((Op.const 0.0).broadcast [128]) (Op.ref .real [128] "m_ij")) s
      = some (⟨fun idx : TileIndex [128] => mistralGuard (mijFn idx.1)⟩ : Tile .real [128]) := by
  simp only [evalOp, hmij, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.cop, Tile.bop_data, Tile.bop, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub, WithBot.realSub,
    Option.map₂_some_some]
  unfold mistralGuard
  by_cases hb : (mijFn i).unbotD 0 = (-1e9 : ℝ)
  · rw [if_pos hb]
    have hcoe : mijFn i = (((-1e9 : ℝ)) : WithBot ℝ) := by
      cases hm : mijFn i with
      | bot => rw [hm] at hb; simp at hb; norm_num at hb
      | coe r => rw [hm] at hb; rw [WithBot.unbotD_coe] at hb; rw [hb]
    rw [if_pos (show ComparableDType.real.eq (mijFn i) (some ((0.0:ℝ) - 1e9)) = Bool.true by
      rw [ComparableDType.real_eq_eq_true, hcoe]
      show ((-1e9 : ℝ) : WithBot ℝ) = some ((0.0:ℝ) - 1e9)
      rw [show ((0.0:ℝ) - 1e9) = (-1e9:ℝ) from by norm_num]; rfl)]
    show some (0.0 : ℝ) = ((0:ℝ) : WithBot ℝ)
    rw [show (0.0:ℝ) = (0:ℝ) from by norm_num]; rfl
  · rw [if_neg hb]
    rw [if_neg (show ¬ (ComparableDType.real.eq (mijFn i) (some ((0.0:ℝ) - 1e9)) = Bool.true) by
      rw [ComparableDType.real_eq_eq_true]
      intro heq
      apply hb
      rw [heq]; norm_num)]

set_option maxHeartbeats 1600000 in
/-- `m_ij = tl.max(qk, 1)` recipe (per-row block max over the key axis). -/
theorem mistral_mij_eval (s : BlockState) (qkFn : Fin 128 → Fin 128 → WithBot ℝ)
    (hqk : s.regs .real [128, 128] "qk"
      = some (⟨fun idx : TileIndex [128, 128] => qkFn idx.1 idx.2.1⟩ : Tile .real [128, 128])) :
    evalOp (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk")) s
      = some (⟨fun idx : TileIndex [128] =>
          Finset.univ.sup' Finset.univ_nonempty (fun jL : Fin 128 => qkFn idx.1 jL)⟩ : Tile .real [128]) := by
  rw [evalOp_reduceMax, evalOp_ref, hqk]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceMax_false]
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show 0 < TileShape.axisDim [128, 128] (⟨1, by simp⟩ : Fin [128,128].length) from by decide)]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  rfl

set_option maxHeartbeats 1600000 in
/-- `p = tl.exp(qk - m_ij[:, None])` recipe. -/
theorem mistral_p_eval (s : BlockState) (qkFn : Fin 128 → Fin 128 → WithBot ℝ)
    (mijFn : Fin 128 → WithBot ℝ)
    (hqk : s.regs .real [128, 128] "qk"
      = some (⟨fun idx : TileIndex [128, 128] => qkFn idx.1 idx.2.1⟩ : Tile .real [128, 128]))
    (hmij : s.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij"))).exp s
      = some (⟨fun idx : TileIndex [128, 128] =>
          WithBot.realExp (WithBot.realSub (qkFn idx.1 idx.2.1) (mijFn idx.1))⟩ : Tile .real [128, 128]) := by
  have hexp : @evalOp .real [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "m_ij")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128])) :=
    evalOp_expandDim_ref_of_regs .real [128] ⟨1, by simp⟩ "m_ij" s _ hmij
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hqk, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.sub, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- `l_ij = tl.sum(p, 1)` recipe (per-row block denominator). -/
theorem mistral_lij_eval (s : BlockState) (pFn : Fin 128 → Fin 128 → WithBot ℝ)
    (hp : s.regs .real [128, 128] "p"
      = some (⟨fun idx : TileIndex [128, 128] => pFn idx.1 idx.2.1⟩ : Tile .real [128, 128])) :
    evalOp (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "p")) s
      = some (⟨fun idx : TileIndex [128] =>
          (Finset.univ.sum (fun jL : Fin 128 => pFn idx.1 jL))⟩ : Tile .real [128]) := by
  rw [evalOp_reduceSum, evalOp_ref, hp]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceSum_false]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  rfl

set_option maxHeartbeats 1600000 in
/-- `m_i_new = tl.maximum(m_i, m_ij)` recipe, per-row `m_i[i] ⊔ m_ij[i]`. -/
theorem mistral_minew_eval (s : BlockState) (miFn mijFn : Fin 128 → WithBot ℝ)
    (hmi : s.regs .real [128] "m_i" = some (⟨fun idx : TileIndex [128] => miFn idx.1⟩ : Tile .real [128]))
    (hmij : s.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128])) :
    evalOp ((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref .real [128] "m_i")
            (Op.ref .real [128] "m_ij")).where
        (Op.ref .real [128] "m_i") (Op.ref .real [128] "m_ij")) s
      = some (⟨fun idx : TileIndex [128] => miFn idx.1 ⊔ mijFn idx.1⟩ : Tile .real [128]) := by
  rw [evalOp_where, evalOp_gt]
  simp only [evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.gt]
  by_cases h : mijFn i < miFn i
  · rw [if_pos (by simpa using h), max_eq_left (le_of_lt h)]
  · rw [if_neg (by simpa using h), max_eq_right (not_lt.mp h)]

set_option maxHeartbeats 1600000 in
/-- `alpha = tl.exp(m_i - m_i_new)` recipe. -/
theorem mistral_alpha_eval (s : BlockState) (miFn minewFn : Fin 128 → WithBot ℝ)
    (hmi : s.regs .real [128] "m_i" = some (⟨fun idx : TileIndex [128] => miFn idx.1⟩ : Tile .real [128]))
    (hminew : s.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_i")
        (Op.ref .real [128] "m_i_new")).exp s
      = some (⟨fun idx : TileIndex [128] => WithBot.realExp (WithBot.realSub (miFn idx.1) (minewFn idx.1))⟩
          : Tile .real [128]) := by
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hmi, evalOp_ref, hminew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub]

set_option maxHeartbeats 1600000 in
/-- `beta = tl.exp(m_ij - m_i_new)` recipe. -/
theorem mistral_beta_eval (s : BlockState) (mijFn minewFn : Fin 128 → WithBot ℝ)
    (hmij : s.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]))
    (hminew : s.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [128] "m_ij")
        (Op.ref .real [128] "m_i_new")).exp s
      = some (⟨fun idx : TileIndex [128] => WithBot.realExp (WithBot.realSub (mijFn idx.1) (minewFn idx.1))⟩
          : Tile .real [128]) := by
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hmij, evalOp_ref, hminew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub]

set_option maxHeartbeats 1600000 in
/-- `l_i_new = alpha*l_i + beta*l_ij` recipe. -/
theorem mistral_linew_eval (s : BlockState) (alphaFn liFn betaFn lijFn : Fin 128 → WithBot ℝ)
    (halpha : s.regs .real [128] "alpha" = some (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128]))
    (hli : s.regs .real [128] "l_i" = some (⟨fun idx : TileIndex [128] => liFn idx.1⟩ : Tile .real [128]))
    (hbeta : s.regs .real [128] "beta" = some (⟨fun idx : TileIndex [128] => betaFn idx.1⟩ : Tile .real [128]))
    (hlij : s.regs .real [128] "l_ij" = some (⟨fun idx : TileIndex [128] => lijFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "alpha") (Op.ref .real [128] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [128] "beta") (Op.ref .real [128] "l_ij"))) s
      = some (⟨fun idx : TileIndex [128] =>
          WithBot.realAdd (WithBot.realMul (alphaFn idx.1) (liFn idx.1))
            (WithBot.realMul (betaFn idx.1) (lijFn idx.1))⟩ : Tile .real [128]) := by
  rw [evalOp_add, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, halpha, hli, hbeta, hlij, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 1600000 in
/-- **`l_i_new` guard recipe.** `l_i_new = tl.where(l_i_new == 0, 1e-9, l_i_new)`.
Output cell: `some 1e-9` when `linewFn i == 0`, else `linewFn i`. -/
theorem mistral_linew_guard_eval (s : BlockState) (linewFn : Fin 128 → WithBot ℝ)
    (hlinew : s.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128])) :
    evalOp ((Op.eq ComparableDType.real Broadcast.scalarR (Op.ref .real [128] "l_i_new")
            (Op.const 0.0)).where
        ((Op.const 1e-9).broadcast [128]) (Op.ref .real [128] "l_i_new")) s
      = some (⟨fun idx : TileIndex [128] =>
          if linewFn idx.1 = ((0 : ℝ) : WithBot ℝ) then ((1e-9 : ℝ) : WithBot ℝ) else linewFn idx.1⟩
          : Tile .real [128]) := by
  simp only [evalOp, hlinew, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.cop, Tile.bop_data, Tile.bop, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex]
  by_cases hz : linewFn i = ((0 : ℝ) : WithBot ℝ)
  · rw [if_pos (show ComparableDType.real.eq (linewFn i) (some (0.0:ℝ)) = Bool.true by
        rw [ComparableDType.real_eq_eq_true, hz]
        show ((0:ℝ) : WithBot ℝ) = some (0.0:ℝ)
        rw [show (0.0:ℝ) = (0:ℝ) from by norm_num]; rfl),
      if_pos hz]
    rfl
  · rw [if_neg (show ¬ ComparableDType.real.eq (linewFn i) (some (0.0:ℝ)) = Bool.true by
        rw [ComparableDType.real_eq_eq_true]; intro h; apply hz
        rw [h]; show ((0.0:ℝ) : WithBot ℝ) = ((0:ℝ) : WithBot ℝ)
        rw [show (0.0:ℝ) = (0:ℝ) from by norm_num]),
      if_neg hz]

set_option maxHeartbeats 1600000 in
/-- `p_scale = beta / l_i_new` recipe. -/
theorem mistral_pscale_eval (s : BlockState) (betaFn linewFn : Fin 128 → WithBot ℝ)
    (hbeta : s.regs .real [128] "beta" = some (⟨fun idx : TileIndex [128] => betaFn idx.1⟩ : Tile .real [128]))
    (hlinew : s.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.div .real Broadcast.nil.consSame (Op.ref .real [128] "beta")
        (Op.ref .real [128] "l_i_new")) s
      = some (⟨fun idx : TileIndex [128] => WithBot.realDiv (betaFn idx.1) (linewFn idx.1)⟩ : Tile .real [128]) := by
  rw [evalOp_div, evalOp_ref, hbeta, evalOp_ref, hlinew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div]

set_option maxHeartbeats 1600000 in
/-- `p = p * p_scale[:, None]` recipe. -/
theorem mistral_pmul_eval (s : BlockState) (pFn : Fin 128 → Fin 128 → WithBot ℝ)
    (psFn : Fin 128 → WithBot ℝ)
    (hp : s.regs .real [128, 128] "p"
      = some (⟨fun idx : TileIndex [128, 128] => pFn idx.1 idx.2.1⟩ : Tile .real [128, 128]))
    (hps : s.regs .real [128] "p_scale" = some (⟨fun idx : TileIndex [128] => psFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "p_scale"))) s
      = some (⟨fun idx : TileIndex [128, 128] => WithBot.realMul (pFn idx.1 idx.2.1) (psFn idx.1)⟩
          : Tile .real [128, 128]) := by
  have hexp : @evalOp .real [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "p_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [128] => psFn idx.1⟩ : Tile .real [128])) :=
    evalOp_expandDim_ref_of_regs .real [128] ⟨1, by simp⟩ "p_scale" s _ hps
  rw [evalOp_mul, evalOp_ref, hp, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- `acc_scale = l_i / l_i_new * alpha` recipe. -/
theorem mistral_accscale_eval (s : BlockState) (liFn linewFn alphaFn : Fin 128 → WithBot ℝ)
    (hli : s.regs .real [128] "l_i" = some (⟨fun idx : TileIndex [128] => liFn idx.1⟩ : Tile .real [128]))
    (hlinew : s.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]))
    (halpha : s.regs .real [128] "alpha" = some (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [128] "l_i") (Op.ref .real [128] "l_i_new"))
        (Op.ref .real [128] "alpha")) s
      = some (⟨fun idx : TileIndex [128] =>
          WithBot.realMul (WithBot.realDiv (liFn idx.1) (linewFn idx.1)) (alphaFn idx.1)⟩ : Tile .real [128]) := by
  rw [evalOp_mul, evalOp_div]
  simp only [evalOp_ref, hli, hlinew, halpha, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, NumericDType.div]

set_option maxHeartbeats 1600000 in
/-- `acc = acc * acc_scale[:, None]` recipe. -/
theorem mistral_accmul_eval (s : BlockState) (accFn : Fin 128 → Fin 128 → WithBot ℝ)
    (asFn : Fin 128 → WithBot ℝ)
    (hacc : s.regs .real [128, 128] "acc"
      = some (⟨fun idx : TileIndex [128, 128] => accFn idx.1 idx.2.1⟩ : Tile .real [128, 128]))
    (has : s.regs .real [128] "acc_scale" = some (⟨fun idx : TileIndex [128] => asFn idx.1⟩ : Tile .real [128])) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [128, 128] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "acc_scale"))) s
      = some (⟨fun idx : TileIndex [128, 128] => WithBot.realMul (accFn idx.1 idx.2.1) (asFn idx.1)⟩
          : Tile .real [128, 128]) := by
  have hexp : @evalOp .real [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [128] "acc_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [128] => asFn idx.1⟩ : Tile .real [128])) :=
    evalOp_expandDim_ref_of_regs .real [128] ⟨1, by simp⟩ "acc_scale" s _ has
  rw [evalOp_mul, evalOp_ref, hacc, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- `v` masked-load recipe. -/
theorem mistral_v_load_eval (s : BlockState) (V B_Start_Loc B_Seqlen : RegionName) (SN : Nat)
    (hvp : s.regs .ptr [128, 128] "v_ptrs" =
      some (⟨fun idx : TileIndex [128, 128] =>
        (V, idx.1.val * 768 + s.pids 1 * 128 + idx.2.1.val)⟩ : Tile .ptr [128, 128]))
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s B_Start_Loc)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [128, 128] "v_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat 768))))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          if SN + idx.1.val < seqLen s B_Seqlen then
            some (s.readMem V ((startLoc s B_Start_Loc + (SN + idx.1.val)) * 768
              + s.pids 1 * 128 + idx.2.1.val))
          else some (0.0 : ℝ)⟩ : Tile .real [128, 128]) := by
  have hexp : @evalOp .nat [128, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin 128 => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [128] ⟨1, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hvp, hsl, hsn, hexp, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨jL, d, u⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim,
    Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    BlockState.readMem, TileShape.dropInsertedIndex]
  by_cases hlt : SN + jL.val < seqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMem]
    rw [show jL.val * 768 + s.pids 1 * 128 + d.val + (startLoc s B_Start_Loc + SN) * 768
        = (startLoc s B_Start_Loc + (SN + jL.val)) * 768 + s.pids 1 * 128 + d.val from by ring]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- `acc += tl.dot(p, v)` recipe. -/
theorem mistral_acc_dot_eval (s : BlockState) (accFn : Fin 128 → Fin 128 → WithBot ℝ)
    (pFn : Fin 128 → Fin 128 → ℝ) (vFn : Fin 128 → Fin 128 → ℝ)
    (hacc : s.regs .real [128, 128] "acc"
      = some (⟨fun idx : TileIndex [128, 128] => accFn idx.1 idx.2.1⟩ : Tile .real [128, 128]))
    (hp : s.regs .real [128, 128] "p"
      = some (⟨fun idx : TileIndex [128, 128] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]))
    (hv : s.regs .real [128, 128] "v"
      = some (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [128, 128] "acc")
        (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v"))) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          WithBot.realAdd (accFn idx.1 idx.2.1)
            (some (Finset.univ.sum (fun jL : Fin 128 => pFn idx.1 jL * vFn jL idx.2.1)))⟩
          : Tile .real [128, 128]) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v")) s
      = some (Tile.dot [] (⟨fun idx : TileIndex [128, 128] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])
          (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])) := by
    rw [evalOp_dot]; erw [evalOp_ref, hp, evalOp_ref, hv]; rfl
  rw [evalOp_add, evalOp_ref, hacc]
  show Option.bind (evalOp (Op.dot (batch := []) (Op.ref .real [128, 128] "p") (Op.ref .real [128, 128] "v")) s) _ = _
  rw [hdot]
  simp only [Option.bind]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ fun jL : Fin 128 =>
        Option.map₂ (· * ·) ((⟨fun idx : TileIndex [128, 128] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]).data (i, jL, PUnit.unit))
          ((⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]).data (jL, d, PUnit.unit)))
      = (@Finset.sum (Fin 128) (WithBot ℝ) _ Finset.univ fun jL : Fin 128 => ((pFn i jL * vFn jL d : ℝ) : WithBot ℝ)) from by
    apply Finset.sum_congr rfl; intro jL _; rfl]
  rw [WithBot.sum_some_eq_some]; rfl

end MistralOpEvals


section MistralInvariant

/-! ## Invariant, preLoop execution, and re-anchor (test shape). -/

/-- `seqLen` depends only on `mem`/`pids`. -/
theorem mistral_seqLen_eq_of_mem_pids (s s0 : BlockState) (B_Seqlen : RegionName)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    seqLen s B_Seqlen = seqLen s0 B_Seqlen := by
  simp only [seqLen, BlockState.readMemValue, BlockState.readMemTyped, hmem, hpids]

/-- `startLoc` depends only on `mem`/`pids`. -/
theorem mistral_startLoc_eq_of_mem_pids (s s0 : BlockState) (B_Start_Loc : RegionName)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    startLoc s B_Start_Loc = startLoc s0 B_Start_Loc := by
  simp only [startLoc, BlockState.readMemValue, BlockState.readMemTyped, hmem, hpids]

noncomputable def mistralInvariant
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName) (sw : Nat)
    (s0 : BlockState) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 2))
  ∧ s.regs .nat [] "cur_kv_head" = some (Tile.scalar (s0.pids 1 / 1))
  ∧ s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s0 B_Seqlen))
  ∧ s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s0 B_Start_Loc))
  ∧ s.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val))
  ∧ s.regs .nat [128] "offs_d" = some (Tile.vec (fun e : Fin 128 => e.val))
  ∧ s.regs .nat [128] "offs_m" = some (Tile.vec (fun i : Fin 128 => s0.pids 2 * 128 + i.val))
  ∧ s.regs .real [128, 128] "q" =
      some (⟨fun idx : TileIndex [128, 128] =>
        some (ctxQTileMRowG s0 Q B_Start_Loc 768 128 128 128 (seqLen s0 B_Seqlen) (idx.1, idx.2.1, PUnit.unit))⟩
        : Tile .real [128, 128])
  ∧ s.regs .ptr [128, 128] "k_ptrs" =
      some (⟨fun idx : TileIndex [128, 128] =>
        (K, idx.2.1.val * 768 + s0.pids 1 * 128 + idx.1.val)⟩ : Tile .ptr [128, 128])
  ∧ s.regs .ptr [128, 128] "v_ptrs" =
      some (⟨fun idx : TileIndex [128, 128] =>
        (V, idx.1.val * 768 + s0.pids 1 * 128 + idx.2.1.val)⟩ : Tile .ptr [128, 128])
  ∧ s.regs .nat [] "block_mask" =
      some (Tile.scalar (if 128 * s0.pids 2 < seqLen s0 B_Seqlen then 1 else 0))
  ∧ s.regs .real [128] "m_i" =
      some (⟨fun idx : TileIndex [128] =>
        (mistralFoldUptoG s0 Q K V B_Start_Loc mistral_sm_scale 768 128 128 128
          (ctxMistralWindowG s0 B_Seqlen 128) (seqLen s0 B_Seqlen) sw c idx.1 ⟨0, by norm_num⟩).1⟩
        : Tile .real [128])
  ∧ s.regs .real [128] "l_i" =
      some (⟨fun idx : TileIndex [128] =>
        ((mistralFoldUptoG s0 Q K V B_Start_Loc mistral_sm_scale 768 128 128 128
          (ctxMistralWindowG s0 B_Seqlen 128) (seqLen s0 B_Seqlen) sw c idx.1 ⟨0, by norm_num⟩).2.1 : WithBot ℝ)⟩
        : Tile .real [128])
  ∧ s.regs .real [128, 128] "acc" =
      some (⟨fun idx : TileIndex [128, 128] =>
        ((mistralFoldUptoG s0 Q K V B_Start_Loc mistral_sm_scale 768 128 128 128
          (ctxMistralWindowG s0 B_Seqlen 128) (seqLen s0 B_Seqlen) sw c idx.1 idx.2.1).2.2 : WithBot ℝ)⟩
        : Tile .real [128, 128])

end MistralInvariant


section MistralAttnStep

/-! ## One-block invariant advance (`mistral_attn_step`). -/

/-- The block-`c` score list (`.map fst`) of `mistralBlockMG` is channel-independent. -/
theorem mistralBlockMG_fst_channel_indep
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d d' : Fin DM) :
    (mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d).map Prod.fst
      = (mistralBlockMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d').map Prod.fst := by
  unfold mistralBlockMG
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr
  intro j _
  by_cases hj : c * BLK ≤ j.val ∧ j.val < (c + 1) * BLK <;> simp [hj]

/-- `mistralBlockStepBot`'s running max (`.1`) and denominator (`.2.1`) depend on the
state only through `(.1, .2.1)` and on the block only through its scores (`.map fst`). -/
theorem mistralBlockStepBot_fst_snd_fst_congr
    (st st' : WithBot ℝ × ℝ × ℝ) (b b' : List (ℝ × ℝ))
    (hst1 : st.1 = st'.1) (hst2 : st.2.1 = st'.2.1) (hf : b.map Prod.fst = b'.map Prod.fst) :
    (mistralBlockStepBot st b).1 = (mistralBlockStepBot st' b').1
    ∧ (mistralBlockStepBot st b).2.1 = (mistralBlockStepBot st' b').2.1 := by
  have hcoe : b.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) = b'.map (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    rw [show (fun p : ℝ × ℝ => ((p.1 : ℝ) : WithBot ℝ)) = (fun r : ℝ => ((r : ℝ) : WithBot ℝ)) ∘ Prod.fst from rfl,
      ← List.map_map, ← List.map_map, hf]
  have hexp : ∀ G : WithBot ℝ, b.map (fun p => Real.exp (p.1 - G.unbotD 0)) = b'.map (fun p => Real.exp (p.1 - G.unbotD 0)) := by
    intro G
    rw [show (fun p : ℝ × ℝ => Real.exp (p.1 - G.unbotD 0)) = (fun r : ℝ => Real.exp (r - G.unbotD 0)) ∘ Prod.fst from rfl,
      ← List.map_map, ← List.map_map, hf]
  constructor
  · simp only [mistralBlockStepBot, hcoe, hst1]
  · simp only [mistralBlockStepBot, hcoe, hst1, hst2, hexp]

/-- `mistralFoldUptoG`'s running max (`.1`) and denominator (`.2.1`) are
channel-independent. -/
theorem mistralFoldUptoG_channel_indep
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM S bel sw c : Nat) (i : Fin BLK) (d d' : Fin DM) :
    (mistralFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d).1
      = (mistralFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d').1
    ∧ (mistralFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d).2.1
      = (mistralFoldUptoG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw c i d').2.1 := by
  induction c with
  | zero => constructor <;> simp only [mistralFoldUptoG_zero]
  | succ n ih =>
    obtain ⟨ih1, ih2⟩ := ih
    rw [mistralFoldUptoG_succ, mistralFoldUptoG_succ]
    exact mistralBlockStepBot_fst_snd_fst_congr _ _ _ _ ih1 ih2
      (mistralBlockMG_fst_channel_indep s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw n i d d')

set_option maxHeartbeats 3200000 in
set_option maxRecDepth 8000 in
/-- **One loop-body step advances the invariant by one block.** -/
theorem mistral_attn_step
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName) (sw : Nat)
    (s0 : BlockState) (i : Nat) (s : BlockState)
    (hilt : i < ctxMistralWindowG s0 B_Seqlen 128)
    (hinv : mistralInvariant Q K V B_Start_Loc B_Seqlen sw s0 (i / 128) s)
    (hi : i = (i / 128) * 128) :
    ∃ s', stepStmts (mistralLoopBody mistral_sm_scale sw) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ mistralInvariant Q K V B_Start_Loc B_Seqlen sw s0 (i / 128 + 1) s' := by
  set S := ctxMistralWindowG s0 B_Seqlen 128 with hSdef
  set bel := seqLen s0 B_Seqlen with hbeldef
  set c := i / 128 with hc_def
  set sc := mistral_sm_scale with hscdef
  have hwin : (c + 1) * 128 ≤ S := by
    have hSmul : S = (if 128 * s0.pids 2 < bel then 1 else 0) * (s0.pids 2 + 1) * 128 := by
      simp only [hSdef, ctxMistralWindowG, hbeldef, seqLen]; rfl
    by_cases hbm : 128 * s0.pids 2 < bel
    · rw [hSmul, if_pos hbm, one_mul] at hilt ⊢
      have : c < s0.pids 2 + 1 := by omega
      nlinarith
    · rw [hSmul, if_neg hbm] at hilt; omega
  obtain ⟨hpids, hmem, hundef, hcb, hch, hsm, hckv, hseq, hsl, hn, hd, hoffm, hq, hkp, hvp, hbmask, hmi, hli, hacc⟩ := hinv
  set fold0 : Fin 128 → Fin 128 → WithBot ℝ × ℝ × ℝ := fun ir dd =>
    mistralFoldUptoG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd with hfold0
  set blk : Fin 128 → Fin 128 → List (ℝ × ℝ) := fun ir dd =>
    mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd with hblk
  -- the sentinel-kept score per row/lane (= mistralScoreM at key c*128+jL)
  set scoreFn : Fin 128 → Fin 128 → ℝ := fun ir jL =>
    mistralScoreM s0 Q K B_Start_Loc sc 768 128 128 128 S bel sw ir
      ⟨c * 128 + jL.val, mistral_lane_lt_SG c S 128 hwin jL⟩ with hscoreFn
  unfold mistralLoopBody
  set s1 := s.setReg "start_n" .nat [] (Tile.scalar i) with hs1d
  have hs1pids : s1.pids = s0.pids := by rw [hs1d, BlockState.setReg_pids]; exact hpids
  have hs1mem : s1.mem = s0.mem := by funext rg o; rw [hs1d, BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1sn : s1.regs .nat [] "start_n" = some (Tile.scalar i) := by rw [hs1d, BlockState.setReg_same]
  -- stmt 0: start_n = ref start_n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") s1 = some (Tile.scalar i) from by rw [evalOp_ref]; exact hs1sn))]
  set s2 := s1.setReg "start_n" .nat [] (Tile.scalar i) with hs2d
  have hs2pids : s2.pids = s0.pids := by rw [hs2d, BlockState.setReg_pids]; exact hs1pids
  have hs2mem : s2.mem = s0.mem := by funext rg o; rw [hs2d, BlockState.setReg_mem]; exact hs1mem ▸ rfl
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2sn : s2.regs .nat [] "start_n" = some (Tile.scalar i) := by rw [hs2d, BlockState.setReg_same]
  have hs2seq : s2.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar bel) := e2 (by decide) (e1 (by decide) hseq)
  have hs2slStart : startLoc s2 B_Start_Loc = startLoc s0 B_Start_Loc := mistral_startLoc_eq_of_mem_pids s2 s0 B_Start_Loc hs2mem hs2pids
  have hs2sl : s2.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s2 B_Start_Loc)) := by
    rw [hs2slStart]; exact e2 (by decide) (e1 (by decide) hsl)
  have hs2n : s2.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)) := e2 (by decide) (e1 (by decide) hn)
  have hs2m : s2.regs .nat [128] "offs_m" = some (Tile.vec (fun ir : Fin 128 => s0.pids 2 * 128 + ir.val)) := e2 (by decide) (e1 (by decide) hoffm)
  have hs2kp : s2.regs .ptr [128, 128] "k_ptrs" = some (⟨fun idx : TileIndex [128, 128] =>
      (K, idx.2.1.val * 768 + s2.pids 1 * 128 + idx.1.val)⟩ : Tile .ptr [128, 128]) := by
    rw [hs2pids]; exact e2 (by decide) (e1 (by decide) hkp)
  have hs2seqB : seqLen s2 B_Seqlen = bel := by rw [hbeldef]; exact mistral_seqLen_eq_of_mem_pids s2 s0 B_Seqlen hs2mem hs2pids
  -- stmt 1: k = masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ s2 = _ from by
      have h := mistral_k_load_eval s2 K B_Start_Loc B_Seqlen i hs2kp hs2sl hs2sn hs2n (by rw [hs2seqB]; exact hs2seq)
      rw [hs2seqB] at h; exact h))]
  set kFn : Fin 128 → Fin 128 → ℝ := fun e jL =>
    ctxKTileMG s0 K B_Start_Loc 768 128 S 128 bel (⟨c * 128 + jL.val, mistral_lane_lt_SG c S 128 hwin jL⟩, e, PUnit.unit) with hkFn
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        if i + idx.2.1.val < bel then
          some (s2.readMem K ((startLoc s2 B_Start_Loc + (i + idx.2.1.val)) * 768 + s2.pids 1 * 128 + idx.1.val))
        else some (0.0 : ℝ)⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_)
    obtain ⟨e, jL, u⟩ := idx
    simp only [hkFn, ctxKTileMG, ctxKTileG, hs2pids,
      show startLoc s2 B_Start_Loc = startLoc s0 B_Start_Loc from mistral_startLoc_eq_of_mem_pids s2 s0 B_Start_Loc hs2mem hs2pids,
      show i + jL.val = c * 128 + jL.val from by rw [hi]]
    by_cases hlt : c * 128 + jL.val < bel
    · rw [if_pos hlt, if_pos hlt]
      simp only [BlockState.readMem, hs2mem]
    · rw [if_neg hlt, if_neg hlt]; norm_num]
  set s3 := s2.setReg "k" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) with hs3d
  have e3 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "k" → s2.regs dt sh nm = some t → s3.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs3d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs3pids : s3.pids = s0.pids := by rw [hs3d, BlockState.setReg_pids]; exact hs2pids
  have hs3mem : s3.mem = s0.mem := by funext rg o; rw [hs3d, BlockState.setReg_mem]; exact hs2mem ▸ rfl
  have hs3k : s3.regs .real [128, 128] "k" = some (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := by
    rw [hs3d, BlockState.setReg_same]
  set qFn : Fin 128 → Fin 128 → ℝ := fun ir e => ctxQTileMRowG s0 Q B_Start_Loc 768 128 128 128 bel (ir, e, PUnit.unit) with hqFn
  have hs3q : s3.regs .real [128, 128] "q" = some (⟨fun idx : TileIndex [128, 128] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) :=
    e3 (by decide) (e2 (by decide) (e1 (by decide) hq))
  -- stmt 2: qk = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [128, 128] (Op.const 0)) s3
        = some (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]) from by
      simp [evalOp_full, evalOp_const]))]
  set s4 := s3.setReg "qk" .real [128, 128] (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]) with hs4d
  have e4 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s3.regs dt sh nm = some t → s4.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs4d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs4qk0 : s4.regs .real [128, 128] "qk" = some (⟨fun _ : TileIndex [128, 128] => some (0 : ℝ)⟩ : Tile .real [128, 128]) := by
    rw [hs4d, BlockState.setReg_same]
  have hs4q : s4.regs .real [128, 128] "q" = some (⟨fun idx : TileIndex [128, 128] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := e4 (by decide) hs3q
  have hs4k : s4.regs .real [128, 128] "k" = some (⟨fun idx : TileIndex [128, 128] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := e4 (by decide) hs3k
  -- stmt 3: qk += dot(q,k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_qk_dot_eval s4 qFn kFn hs4qk0 hs4q hs4k))]
  set rawSum : Fin 128 → Fin 128 → ℝ := fun ir jL =>
    Finset.univ.sum (fun e : Fin 128 => qFn ir e * kFn e jL) with hrawSum
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        some (Finset.univ.sum (fun e : Fin 128 => qFn idx.1 e * kFn e idx.2.1))⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx; simp only [hrawSum]]
  set s5 := s4.setReg "qk" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [128, 128]) with hs5d
  have e5 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s4.regs dt sh nm = some t → s5.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs5d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs5qk : s5.regs .real [128, 128] "qk" = some (⟨fun idx : TileIndex [128, 128] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := by
    rw [hs5d, BlockState.setReg_same]
  -- stmt 4: qk *= sc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_qk_scale_eval s5 sc rawSum hs5qk))]
  set s6 := s5.setReg "qk" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (rawSum idx.1 idx.2.1 * sc)⟩ : Tile .real [128, 128]) with hs6d
  have e6 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s5.regs dt sh nm = some t → s6.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs6d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs6qk : s6.regs .real [128, 128] "qk" = some (⟨fun idx : TileIndex [128, 128] => some (rawSum idx.1 idx.2.1 * sc)⟩ : Tile .real [128, 128]) := by
    rw [hs6d, BlockState.setReg_same]
  have hs6m : s6.regs .nat [128] "offs_m" = some (Tile.vec (fun ir : Fin 128 => s0.pids 2 * 128 + ir.val)) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2m)))
  have hs6sn : s6.regs .nat [] "start_n" = some (Tile.scalar i) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2sn)))
  have hs6n : s6.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2n)))
  -- stmt 5: first where (causal, -1e9)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_qk_where1_eval s6 i (fun ir jL => rawSum ir jL * sc) (fun ir => s0.pids 2 * 128 + ir.val)
      hs6qk hs6m hs6sn hs6n))]
  -- after where1: cell = some(if SN+jL ≤ gi then sc·rawSum else -1e9)
  set g1 : Fin 128 → Fin 128 → ℝ := fun ir jL =>
    if i + jL.val ≤ s0.pids 2 * 128 + ir.val then rawSum ir jL * sc else (-1e9 : ℝ) with hg1
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        if i + idx.2.1.val ≤ s0.pids 2 * 128 + idx.1.val then some (rawSum idx.1 idx.2.1 * sc)
        else some (-1e9 : ℝ)⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => some (g1 idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx
    simp only [hg1]
    by_cases hca : i + jL.val ≤ s0.pids 2 * 128 + ir.val
    · rw [if_pos hca, if_pos hca]
    · rw [if_neg hca, if_neg hca]]
  set s7 := s6.setReg "qk" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (g1 idx.1 idx.2.1)⟩ : Tile .real [128, 128]) with hs7d
  have e7 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s6.regs dt sh nm = some t → s7.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs7d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs7qk : s7.regs .real [128, 128] "qk" = some (⟨fun idx : TileIndex [128, 128] => some (g1 idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := by
    rw [hs7d, BlockState.setReg_same]
  have hs7m : s7.regs .nat [128] "offs_m" = some (Tile.vec (fun ir : Fin 128 => s0.pids 2 * 128 + ir.val)) := e7 (by decide) hs6m
  have hs7sn : s7.regs .nat [] "start_n" = some (Tile.scalar i) := e7 (by decide) hs6sn
  have hs7n : s7.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)) := e7 (by decide) hs6n
  -- stmt 6: second where (sliding window, -1e9) → final qkW = some(scoreFn ir jL)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_qk_where2_eval s7 i sw g1 (fun ir => s0.pids 2 * 128 + ir.val)
      hs7qk hs7m hs7sn hs7n))]
  -- after where2: cell = some(if gi-sw < SN+jL then g1 else -1e9) = some(scoreFn ir jL)
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        if s0.pids 2 * 128 + idx.1.val - sw < i + idx.2.1.val then some (g1 idx.1 idx.2.1)
        else some (-1e9 : ℝ)⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => some (scoreFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx
    show (if s0.pids 2 * 128 + ir.val - sw < i + jL.val then some (g1 ir jL) else some (-1e9 : ℝ))
      = some (scoreFn ir jL)
    simp only [hscoreFn, mistralScoreM, mistralActive, Fin.val_mk, hg1]
    rw [show i + jL.val = c * 128 + jL.val from by rw [hi]]
    by_cases hwd : s0.pids 2 * 128 + ir.val - sw < c * 128 + jL.val
    · rw [if_pos hwd]
      by_cases hca : c * 128 + jL.val ≤ s0.pids 2 * 128 + ir.val
      · rw [if_pos hca, if_pos ⟨hca, hwd⟩, hrawSum, hqFn, hkFn, mul_comm]
      · rw [if_neg hca, if_neg (fun h => hca h.1)]
    · rw [if_neg hwd, if_neg (fun h => hwd h.2)]]
  set qkW : Fin 128 → Fin 128 → WithBot ℝ := fun ir jL => ((scoreFn ir jL : ℝ) : WithBot ℝ) with hqkW
  rw [show (⟨fun idx : TileIndex [128, 128] => some (scoreFn idx.1 idx.2.1)⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => qkW idx.1 idx.2.1⟩ : Tile .real [128, 128]) from rfl]
  set s8 := s7.setReg "qk" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => qkW idx.1 idx.2.1⟩ : Tile .real [128, 128]) with hs8d
  have e8 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s7.regs dt sh nm = some t → s8.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs8d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs8qk : s8.regs .real [128, 128] "qk" = some (⟨fun idx : TileIndex [128, 128] => qkW idx.1 idx.2.1⟩ : Tile .real [128, 128]) := by
    rw [hs8d, BlockState.setReg_same]
  -- blockSup ir = foldr-sup of blk ir scores ; via mistralBlockMG_sup_eq
  set bsup : Fin 128 → WithBot ℝ := fun ir =>
    (blk ir ⟨0, by norm_num⟩).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) |>.foldr (· ⊔ ·) ⊥ with hbsup
  -- stmt 7: m_ij = max(qk, 1) = sup' = bsup
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_mij_eval s8 qkW hs8qk))]
  rw [show (⟨fun idx : TileIndex [128] =>
        Finset.univ.sup' Finset.univ_nonempty (fun jL : Fin 128 => qkW idx.1 jL)⟩ : Tile .real [128])
      = (⟨fun idx : TileIndex [128] => bsup idx.1⟩ : Tile .real [128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, u⟩ := idx
    show Finset.univ.sup' Finset.univ_nonempty (fun jL : Fin 128 => qkW ir jL) = bsup ir
    rw [Finset.sup'_eq_sup, hbsup]
    simp only [hblk, hqkW, hscoreFn]
    rw [← mistralBlockMG_sup_eq s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir ⟨0, by norm_num⟩ hwin]]
  set s9 := s8.setReg "m_ij" .real [128] (⟨fun idx : TileIndex [128] => bsup idx.1⟩ : Tile .real [128]) with hs9d
  have e9 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_ij" → s8.regs dt sh nm = some t → s9.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs9d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs9mij0 : s9.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => bsup idx.1⟩ : Tile .real [128]) := by
    rw [hs9d, BlockState.setReg_same]
  -- stmt 8: m_ij guard = mistralGuard (bsup)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_mij_guard_eval s9 bsup hs9mij0))]
  set mijFn : Fin 128 → WithBot ℝ := fun ir => mistralGuard (bsup ir) with hmijFn
  set s10 := s9.setReg "m_ij" .real [128] (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) with hs10d
  have e10 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_ij" → s9.regs dt sh nm = some t → s10.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs10d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs10mij : s10.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) := by
    rw [hs10d, BlockState.setReg_same]
  have hs10qk : s10.regs .real [128, 128] "qk" = some (⟨fun idx : TileIndex [128, 128] => qkW idx.1 idx.2.1⟩ : Tile .real [128, 128]) :=
    e10 (by decide) (e9 (by decide) hs8qk)
  -- stmt 9: p = exp(qk - m_ij[:,None])
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_p_eval s10 qkW mijFn hs10qk hs10mij))]
  set pFn : Fin 128 → Fin 128 → WithBot ℝ := fun ir jL => WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir)) with hpFn
  set s11 := s10.setReg "p" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => pFn idx.1 idx.2.1⟩ : Tile .real [128, 128]) with hs11d
  have e11 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s10.regs dt sh nm = some t → s11.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs11d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs11p : s11.regs .real [128, 128] "p" = some (⟨fun idx : TileIndex [128, 128] => pFn idx.1 idx.2.1⟩ : Tile .real [128, 128]) := by
    rw [hs11d, BlockState.setReg_same]
  -- stmt 10: l_ij = sum(p, 1)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_lij_eval s11 pFn hs11p))]
  set lijFn : Fin 128 → WithBot ℝ := fun ir => Finset.univ.sum (fun jL : Fin 128 => pFn ir jL) with hlijFn
  set s12 := s11.setReg "l_ij" .real [128] (⟨fun idx : TileIndex [128] => lijFn idx.1⟩ : Tile .real [128]) with hs12d
  have e12 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_ij" → s11.regs dt sh nm = some t → s12.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs12d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs12lij : s12.regs .real [128] "l_ij" = some (⟨fun idx : TileIndex [128] => lijFn idx.1⟩ : Tile .real [128]) := by
    rw [hs12d, BlockState.setReg_same]
  set miFn : Fin 128 → WithBot ℝ := fun ir => (fold0 ir ⟨0, by norm_num⟩).1 with hmiFn
  have hs12mi : s12.regs .real [128] "m_i" = some (⟨fun idx : TileIndex [128] => miFn idx.1⟩ : Tile .real [128]) :=
    e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hmi)))))))))))
  have hs12mij : s12.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) := e12 (by decide) (e11 (by decide) hs10mij)
  -- stmt 11: m_i_new = maximum(m_i, m_ij)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_minew_eval s12 miFn mijFn hs12mi hs12mij))]
  set minewFn : Fin 128 → WithBot ℝ := fun ir => miFn ir ⊔ mijFn ir with hminewFn
  set s13 := s12.setReg "m_i_new" .real [128] (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) with hs13d
  have e13 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_i_new" → s12.regs dt sh nm = some t → s13.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs13d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs13minew : s13.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) := by
    rw [hs13d, BlockState.setReg_same]
  have hs13mi : s13.regs .real [128] "m_i" = some (⟨fun idx : TileIndex [128] => miFn idx.1⟩ : Tile .real [128]) := e13 (by decide) hs12mi
  have hs13mij : s13.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) := e13 (by decide) hs12mij
  -- stmt 12: alpha = exp(m_i - m_i_new)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_alpha_eval s13 miFn minewFn hs13mi hs13minew))]
  set alphaFn : Fin 128 → WithBot ℝ := fun ir => WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir)) with halphaFn
  set s14 := s13.setReg "alpha" .real [128] (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128]) with hs14d
  have e14 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "alpha" → s13.regs dt sh nm = some t → s14.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs14d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs14alpha : s14.regs .real [128] "alpha" = some (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128]) := by
    rw [hs14d, BlockState.setReg_same]
  have hs14mij : s14.regs .real [128] "m_ij" = some (⟨fun idx : TileIndex [128] => mijFn idx.1⟩ : Tile .real [128]) := e14 (by decide) hs13mij
  have hs14minew : s14.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) := e14 (by decide) hs13minew
  -- stmt 13: beta = exp(m_ij - m_i_new)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_beta_eval s14 mijFn minewFn hs14mij hs14minew))]
  set betaFn : Fin 128 → WithBot ℝ := fun ir => WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir)) with hbetaFn
  set s15 := s14.setReg "beta" .real [128] (⟨fun idx : TileIndex [128] => betaFn idx.1⟩ : Tile .real [128]) with hs15d
  have e15 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "beta" → s14.regs dt sh nm = some t → s15.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs15d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs15beta : s15.regs .real [128] "beta" = some (⟨fun idx : TileIndex [128] => betaFn idx.1⟩ : Tile .real [128]) := by
    rw [hs15d, BlockState.setReg_same]
  have hs15alpha : s15.regs .real [128] "alpha" = some (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128]) := e15 (by decide) hs14alpha
  set liFn : Fin 128 → WithBot ℝ := fun ir => ((fold0 ir ⟨0, by norm_num⟩).2.1 : WithBot ℝ) with hliFn
  have hs15li : s15.regs .real [128] "l_i" = some (⟨fun idx : TileIndex [128] => liFn idx.1⟩ : Tile .real [128]) :=
    e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hli))))))))))))))
  have hs15lij : s15.regs .real [128] "l_ij" = some (⟨fun idx : TileIndex [128] => lijFn idx.1⟩ : Tile .real [128]) :=
    e15 (by decide) (e14 (by decide) (e13 (by decide) hs12lij))
  -- stmt 14: l_i_new = alpha*l_i + beta*l_ij
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_linew_eval s15 alphaFn liFn betaFn lijFn hs15alpha hs15li hs15beta hs15lij))]
  set linewFn : Fin 128 → WithBot ℝ := fun ir =>
    WithBot.realAdd (WithBot.realMul (alphaFn ir) (liFn ir)) (WithBot.realMul (betaFn ir) (lijFn ir)) with hlinewFn
  set s16 := s15.setReg "l_i_new" .real [128] (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) with hs16d
  have e16 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i_new" → s15.regs dt sh nm = some t → s16.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs16d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs16linew0 : s16.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) := by
    rw [hs16d, BlockState.setReg_same]
  -- ===== real-extraction (every kernel WithBot quantity is `some` of a real) =====
  set pReal : Fin 128 → Fin 128 → ℝ := fun ir jL =>
    (WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0 with hpReal
  have hpFnSome : ∀ ir jL : Fin 128, pFn ir jL = some (pReal ir jL) := by
    intro ir jL
    show WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))
      = some ((WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0)
    exact nopad_realExp_eq_some_unbotD _
  set lijReal : Fin 128 → ℝ := fun ir => Finset.univ.sum (fun jL : Fin 128 => pReal ir jL) with hlijReal
  have hlijSome : ∀ ir : Fin 128, lijFn ir = some (lijReal ir) := by
    intro ir
    show Finset.univ.sum (fun jL : Fin 128 => pFn ir jL) = some (Finset.univ.sum (fun jL : Fin 128 => pReal ir jL))
    rw [show (fun jL : Fin 128 => pFn ir jL) = (fun jL : Fin 128 => (some (pReal ir jL) : WithBot ℝ)) from by
      funext jL; exact hpFnSome ir jL]
    rw [WithBot.sum_someTerm_eq_some]
  set αReal : Fin 128 → ℝ := fun ir => (WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir))).unbotD 0 with hαReal
  set βReal : Fin 128 → ℝ := fun ir => (WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir))).unbotD 0 with hβReal
  have hαSome : ∀ ir, alphaFn ir = some (αReal ir) := fun ir => nopad_realExp_eq_some_unbotD _
  have hβSome : ∀ ir, betaFn ir = some (βReal ir) := fun ir => nopad_realExp_eq_some_unbotD _
  have hliSome : ∀ ir, liFn ir = some ((fold0 ir ⟨0, by norm_num⟩).2.1) := by intro ir; rfl
  set linewReal : Fin 128 → ℝ := fun ir => αReal ir * (fold0 ir ⟨0, by norm_num⟩).2.1 + βReal ir * lijReal ir with hlinewReal
  have hlinewSome : ∀ ir, linewFn ir = some (linewReal ir) := by
    intro ir
    show WithBot.realAdd (WithBot.realMul (alphaFn ir) (liFn ir)) (WithBot.realMul (betaFn ir) (lijFn ir))
      = some (linewReal ir)
    rw [hαSome, hliSome, hβSome, hlijSome]; rfl
  -- mijFn = G(bsup) ≠ ⊥ ; bsup ≠ ⊥ (block nonempty)
  have hbsupNe : ∀ ir, bsup ir ≠ ⊥ := by
    intro ir
    have hbne : blk ir ⟨0, by norm_num⟩ ≠ [] := by
      simp only [hblk]; exact mistralBlockMG_ne_nil s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir ⟨0, by norm_num⟩ hwin (by norm_num)
    simp only [hbsup]; exact fun h => hbne ((foldr_sup_coe_bot_iff _).mp h)
  -- linewReal ir > 0 : β·lij > 0
  have hlinewPos : ∀ ir, 0 < linewReal ir := by
    intro ir
    -- minewFn ≠ ⊥ since mijFn(=G(bsup)) ≠ ⊥
    have hGne : mijFn ir ≠ ⊥ := by simp only [hmijFn]; exact mistralGuard_ne_bot (hbsupNe ir)
    have hminewNe : minewFn ir ≠ ⊥ := by
      simp only [hminewFn]; intro h; exact hGne (le_bot_iff.mp (h ▸ (le_sup_right : mijFn ir ≤ miFn ir ⊔ mijFn ir)))
    obtain ⟨mr, hmr⟩ : ∃ r : ℝ, minewFn ir = (r : WithBot ℝ) := by
      cases hm : minewFn ir with
      | bot => exact absurd hm hminewNe
      | coe r => exact ⟨r, rfl⟩
    obtain ⟨Gr, hGr⟩ : ∃ r : ℝ, mijFn ir = (r : WithBot ℝ) := by
      cases hm : mijFn ir with
      | bot => exact absurd hm hGne
      | coe r => exact ⟨r, rfl⟩
    -- αReal ≥ 0 (unbotD of realExp)
    have hαnn : 0 ≤ αReal ir := by
      simp only [hαReal]
      cases WithBot.realSub (miFn ir) (minewFn ir) with
      | bot => rw [WithBot.realExp_bot, WithBot.unbotD_coe]
      | coe r => simp only [WithBot.realExp_coe, WithBot.unbotD_coe]; exact le_of_lt (Real.exp_pos _)
    -- l ≥ 0
    obtain ⟨_ha_l, _ha_acc, ha_L0, _ha_botL, _ha_botT, _ha_ne⟩ :=
      mistralFoldUptoG_anchor s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir ⟨0, by norm_num⟩ (by norm_num) (le_of_lt (by nlinarith [hwin] : c * 128 < S))
    have hl0 : 0 ≤ (fold0 ir ⟨0, by norm_num⟩).2.1 := by
      simp only [hfold0] at _ha_l ⊢
      rw [_ha_l]
      apply mul_nonneg ?_ ha_L0
      cases (mistralFoldUptoG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir ⟨0, by norm_num⟩).1 with
      | bot => exact le_of_eq rfl
      | coe r => exact le_of_lt (Real.exp_pos _)
    have hαl_nn : 0 ≤ αReal ir * (fold0 ir ⟨0, by norm_num⟩).2.1 := mul_nonneg hαnn hl0
    have hβpos : 0 < βReal ir := by
      simp only [hβReal, hGr, hmr, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
      exact Real.exp_pos _
    have hlijpos : 0 < lijReal ir := by
      simp only [hlijReal]
      have hpr : ∀ jL : Fin 128, 0 < pReal ir jL := by
        intro jL
        show 0 < (WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0
        rw [hqkW, hGr, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
        exact Real.exp_pos _
      exact Finset.sum_pos (fun jL _ => hpr jL) ⟨⟨0, by norm_num⟩, Finset.mem_univ _⟩
    have hblpos : 0 < βReal ir * lijReal ir := mul_pos hβpos hlijpos
    simp only [hlinewReal]; linarith
  -- l_i_new guard (stmt 15): identity since linewFn ≠ some 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_linew_guard_eval s16 linewFn hs16linew0))]
  rw [show (⟨fun idx : TileIndex [128] =>
        if linewFn idx.1 = ((0 : ℝ) : WithBot ℝ) then ((1e-9 : ℝ) : WithBot ℝ) else linewFn idx.1⟩ : Tile .real [128])
      = (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, u⟩ := idx
    show (if linewFn ir = ((0 : ℝ) : WithBot ℝ) then ((1e-9 : ℝ) : WithBot ℝ) else linewFn ir) = linewFn ir
    rw [if_neg (by rw [hlinewSome ir]; intro h; have hp := hlinewPos ir
                   have h2 : linewReal ir = 0 := Option.some.inj h; linarith)]]
  set s17 := s16.setReg "l_i_new" .real [128] (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) with hs17d
  have e17 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i_new" → s16.regs dt sh nm = some t → s17.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs17d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs17linew : s17.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) := by
    rw [hs17d, BlockState.setReg_same]
  have hs17beta : s17.regs .real [128] "beta" = some (⟨fun idx : TileIndex [128] => betaFn idx.1⟩ : Tile .real [128]) := e17 (by decide) hs15beta
  -- stmt 16: p_scale = beta / l_i_new
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_pscale_eval s17 betaFn linewFn hs17beta hs17linew))]
  set psFn : Fin 128 → WithBot ℝ := fun ir => WithBot.realDiv (betaFn ir) (linewFn ir) with hpsFn
  set s18 := s17.setReg "p_scale" .real [128] (⟨fun idx : TileIndex [128] => psFn idx.1⟩ : Tile .real [128]) with hs18d
  have e18 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p_scale" → s17.regs dt sh nm = some t → s18.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs18d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs18ps : s18.regs .real [128] "p_scale" = some (⟨fun idx : TileIndex [128] => psFn idx.1⟩ : Tile .real [128]) := by
    rw [hs18d, BlockState.setReg_same]
  have hs18p : s18.regs .real [128, 128] "p" = some (⟨fun idx : TileIndex [128, 128] => pFn idx.1 idx.2.1⟩ : Tile .real [128, 128]) :=
    e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) hs11p)))))
  -- stmt 17: p = p * p_scale[:,None]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_pmul_eval s18 pFn psFn hs18p hs18ps))]
  set pFinal : Fin 128 → Fin 128 → WithBot ℝ := fun ir jL => WithBot.realMul (pFn ir jL) (psFn ir) with hpFinal
  set s19 := s18.setReg "p" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) with hs19d
  have e19 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s18.regs dt sh nm = some t → s19.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs19d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs19p : s19.regs .real [128, 128] "p" = some (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) := by
    rw [hs19d, BlockState.setReg_same]
  have hs19li : s19.regs .real [128] "l_i" = some (⟨fun idx : TileIndex [128] => liFn idx.1⟩ : Tile .real [128]) :=
    e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) hs15li)))
  have hs19linew : s19.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) :=
    e19 (by decide) (e18 (by decide) hs17linew)
  have hs19alpha : s19.regs .real [128] "alpha" = some (⟨fun idx : TileIndex [128] => alphaFn idx.1⟩ : Tile .real [128]) :=
    e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) hs15alpha)))
  -- stmt 18: acc_scale = l_i / l_i_new * alpha
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_accscale_eval s19 liFn linewFn alphaFn hs19li hs19linew hs19alpha))]
  set asFn : Fin 128 → WithBot ℝ := fun ir => WithBot.realMul (WithBot.realDiv (liFn ir) (linewFn ir)) (alphaFn ir) with hasFn
  set s20 := s19.setReg "acc_scale" .real [128] (⟨fun idx : TileIndex [128] => asFn idx.1⟩ : Tile .real [128]) with hs20d
  have e20 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc_scale" → s19.regs dt sh nm = some t → s20.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs20d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs20as : s20.regs .real [128] "acc_scale" = some (⟨fun idx : TileIndex [128] => asFn idx.1⟩ : Tile .real [128]) := by
    rw [hs20d, BlockState.setReg_same]
  set accFn : Fin 128 → Fin 128 → WithBot ℝ := fun ir dd => ((fold0 ir dd).2.2 : WithBot ℝ) with haccFn
  have hs20acc : s20.regs .real [128, 128] "acc" = some (⟨fun idx : TileIndex [128, 128] => accFn idx.1 idx.2.1⟩ : Tile .real [128, 128]) :=
    e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hacc)))))))))))))))))
  -- stmt 19: acc = acc * acc_scale[:,None]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_accmul_eval s20 accFn asFn hs20acc hs20as))]
  set accMul : Fin 128 → Fin 128 → WithBot ℝ := fun ir dd => WithBot.realMul (accFn ir dd) (asFn ir) with haccMul
  set s21 := s20.setReg "acc" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => accMul idx.1 idx.2.1⟩ : Tile .real [128, 128]) with hs21d
  have e21 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s20.regs dt sh nm = some t → s21.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs21d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs21acc : s21.regs .real [128, 128] "acc" = some (⟨fun idx : TileIndex [128, 128] => accMul idx.1 idx.2.1⟩ : Tile .real [128, 128]) := by
    rw [hs21d, BlockState.setReg_same]
  -- s21 pids/mem agree with s0
  have hs21mem : s21.mem = s0.mem := by
    funext rg o; rw [hs21d, BlockState.setReg_mem, hs20d, BlockState.setReg_mem, hs19d, BlockState.setReg_mem,
      hs18d, BlockState.setReg_mem, hs17d, BlockState.setReg_mem, hs16d, BlockState.setReg_mem,
      hs15d, BlockState.setReg_mem, hs14d, BlockState.setReg_mem, hs13d, BlockState.setReg_mem,
      hs12d, BlockState.setReg_mem, hs11d, BlockState.setReg_mem, hs10d, BlockState.setReg_mem,
      hs9d, BlockState.setReg_mem, hs8d, BlockState.setReg_mem, hs7d, BlockState.setReg_mem,
      hs6d, BlockState.setReg_mem, hs5d, BlockState.setReg_mem, hs4d, BlockState.setReg_mem]; exact hs3mem ▸ rfl
  have hs21pids : s21.pids = s0.pids := by
    rw [hs21d, BlockState.setReg_pids, hs20d, BlockState.setReg_pids, hs19d, BlockState.setReg_pids,
      hs18d, BlockState.setReg_pids, hs17d, BlockState.setReg_pids, hs16d, BlockState.setReg_pids,
      hs15d, BlockState.setReg_pids, hs14d, BlockState.setReg_pids, hs13d, BlockState.setReg_pids,
      hs12d, BlockState.setReg_pids, hs11d, BlockState.setReg_pids, hs10d, BlockState.setReg_pids,
      hs9d, BlockState.setReg_pids, hs8d, BlockState.setReg_pids, hs7d, BlockState.setReg_pids,
      hs6d, BlockState.setReg_pids, hs5d, BlockState.setReg_pids, hs4d, BlockState.setReg_pids, hs3pids]
  have hs21pids1 : s21.pids 1 = s0.pids 1 := by rw [hs21pids]
  have hs21seqB : seqLen s21 B_Seqlen = bel := by rw [hbeldef]; exact mistral_seqLen_eq_of_mem_pids s21 s0 B_Seqlen hs21mem hs21pids
  have hs21slB : startLoc s21 B_Start_Loc = startLoc s0 B_Start_Loc := mistral_startLoc_eq_of_mem_pids s21 s0 B_Start_Loc hs21mem hs21pids
  have hs21vp : s21.regs .ptr [128, 128] "v_ptrs" = some (⟨fun idx : TileIndex [128, 128] =>
      (V, idx.1.val * 768 + s21.pids 1 * 128 + idx.2.1.val)⟩ : Tile .ptr [128, 128]) := by
    rw [hs21pids1]
    exact e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hvp))))))))))))))))))))
  have hs21sl : s21.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s21 B_Start_Loc)) := by
    rw [hs21slB, ← hs2slStart]
    exact e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2sl))))))))))))))))))
  have hs21sn : s21.regs .nat [] "start_n" = some (Tile.scalar i) :=
    e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) hs7sn)))))))))))))
  have hs21n : s21.regs .nat [128] "offs_n" = some (Tile.vec (fun j : Fin 128 => j.val)) :=
    e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) hs7n)))))))))))))
  have hs21seq : s21.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s21 B_Seqlen)) := by
    rw [hs21seqB]
    exact e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2seq))))))))))))))))))
  -- stmt 20: v = masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_v_load_eval s21 V B_Start_Loc B_Seqlen i hs21vp hs21sl hs21sn hs21n hs21seq))]
  set vFn : Fin 128 → Fin 128 → ℝ := fun jL dd =>
    if i + jL.val < seqLen s21 B_Seqlen then
      s21.readMem V ((startLoc s21 B_Start_Loc + (i + jL.val)) * 768 + s21.pids 1 * 128 + dd.val)
    else (0.0 : ℝ) with hvFn
  set s22 := s21.setReg "v" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) with hs22d
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        if i + idx.1.val < seqLen s21 B_Seqlen then
          some (s21.readMem V ((startLoc s21 B_Start_Loc + (i + idx.1.val)) * 768 + s21.pids 1 * 128 + idx.2.1.val))
        else some (0.0 : ℝ)⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨jL, dd, u⟩ := idx
    show (if i + jL.val < seqLen s21 B_Seqlen then
        some (s21.readMem V ((startLoc s21 B_Start_Loc + (i + jL.val)) * 768 + s21.pids 1 * 128 + dd.val))
      else some (0.0 : ℝ)) = some (vFn jL dd)
    by_cases hlt : i + jL.val < seqLen s21 B_Seqlen
    · simp only [hvFn, hlt, if_true]
    · simp only [hvFn, hlt, if_false]]
  have e22 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "v" → s21.regs dt sh nm = some t → s22.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs22d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs22v : s22.regs .real [128, 128] "v" = some (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := by
    rw [hs22d, BlockState.setReg_same]
  have hs22p : s22.regs .real [128, 128] "p" = some (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) := e22 (by decide) hs19p
  -- stmt 21: p = p (rebind)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128, 128] "p") s22 = some (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) from by
      rw [evalOp_ref]; exact hs22p))]
  set s23 := s22.setReg "p" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) with hs23d
  have e23 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s22.regs dt sh nm = some t → s23.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs23d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs23p : s23.regs .real [128, 128] "p" = some (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128]) := by
    rw [hs23d, BlockState.setReg_same]
  have hs23v : s23.regs .real [128, 128] "v" = some (⟨fun idx : TileIndex [128, 128] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := e23 (by decide) hs22v
  have hs23acc : s23.regs .real [128, 128] "acc" = some (⟨fun idx : TileIndex [128, 128] => accMul idx.1 idx.2.1⟩ : Tile .real [128, 128]) :=
    e23 (by decide) (e22 (by decide) hs21acc)
  -- real-extraction for acc and pFinal
  set psReal : Fin 128 → ℝ := fun ir => βReal ir / linewReal ir with hpsReal
  have hpsSome : ∀ ir, psFn ir = some (psReal ir) := by
    intro ir; show WithBot.realDiv (betaFn ir) (linewFn ir) = some (psReal ir)
    rw [hβSome, hlinewSome]; rfl
  have hpFinalSome : ∀ ir jL, pFinal ir jL = some (pReal ir jL * psReal ir) := by
    intro ir jL; show WithBot.realMul (pFn ir jL) (psFn ir) = some (pReal ir jL * psReal ir)
    rw [hpFnSome, hpsSome]; rfl
  set asReal : Fin 128 → ℝ := fun ir => (fold0 ir ⟨0, by norm_num⟩).2.1 / linewReal ir * αReal ir with hasReal
  have hasSome : ∀ ir, asFn ir = some (asReal ir) := by
    intro ir; show WithBot.realMul (WithBot.realDiv (liFn ir) (linewFn ir)) (alphaFn ir) = some (asReal ir)
    rw [hliSome, hlinewSome, hαSome]; rfl
  have haccFnSome : ∀ ir dd, accFn ir dd = some ((fold0 ir dd).2.2) := by intro ir dd; rfl
  have haccMulSome : ∀ ir dd, accMul ir dd = some ((fold0 ir dd).2.2 * asReal ir) := by
    intro ir dd; show WithBot.realMul (accFn ir dd) (asFn ir) = some ((fold0 ir dd).2.2 * asReal ir)
    rw [haccFnSome, hasSome]; rfl
  rw [show (⟨fun idx : TileIndex [128, 128] => pFinal idx.1 idx.2.1⟩ : Tile .real [128, 128])
        = (⟨fun idx : TileIndex [128, 128] => some (pReal idx.1 idx.2.1 * psReal idx.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx; exact hpFinalSome ir jL] at hs23p
  rw [show (⟨fun idx : TileIndex [128, 128] => accMul idx.1 idx.2.1⟩ : Tile .real [128, 128])
        = (⟨fun idx : TileIndex [128, 128] => some ((fold0 idx.1 idx.2.1).2.2 * asReal idx.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, dd, u⟩ := idx; exact haccMulSome ir dd] at hs23acc
  -- stmt 22: acc += dot(p, v)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_acc_dot_eval s23 (fun ir dd => some ((fold0 ir dd).2.2 * asReal ir))
      (fun ir jL => pReal ir jL * psReal ir) vFn hs23acc hs23p hs23v))]
  set accFinal : Fin 128 → Fin 128 → ℝ := fun ir dd =>
    (fold0 ir dd).2.2 * asReal ir + Finset.univ.sum (fun jL : Fin 128 => (pReal ir jL * psReal ir) * vFn jL dd) with haccFinal
  rw [show (⟨fun idx : TileIndex [128, 128] =>
        WithBot.realAdd (some ((fold0 idx.1 idx.2.1).2.2 * asReal idx.1))
          (some (Finset.univ.sum (fun jL : Fin 128 => (pReal idx.1 jL * psReal idx.1) * vFn jL idx.2.1)))⟩ : Tile .real [128, 128])
      = (⟨fun idx : TileIndex [128, 128] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, dd, u⟩ := idx; rfl]
  set s24 := s23.setReg "acc" .real [128, 128] (⟨fun idx : TileIndex [128, 128] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [128, 128]) with hs24d
  have e24 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s23.regs dt sh nm = some t → s24.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs24d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs24acc : s24.regs .real [128, 128] "acc" = some (⟨fun idx : TileIndex [128, 128] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [128, 128]) := by
    rw [hs24d, BlockState.setReg_same]
  have hs24linew : s24.regs .real [128] "l_i_new" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) :=
    e24 (by decide) (e23 (by decide) (e22 (by decide) (e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) hs17linew))))))
  have hs24minew : s24.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) :=
    e24 (by decide) (e23 (by decide) (e22 (by decide) (e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) hs13minew)))))))))) 
  -- stmt 23: l_i = l_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128] "l_i_new") s24 = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) from by
      rw [evalOp_ref]; exact hs24linew))]
  set s25 := s24.setReg "l_i" .real [128] (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) with hs25d
  have e25 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i" → s24.regs dt sh nm = some t → s25.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs25d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs25minew : s25.regs .real [128] "m_i_new" = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) := e25 (by decide) hs24minew
  -- stmt 24: m_i = m_i_new
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [128] "m_i_new") s25 = some (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) from by
      rw [evalOp_ref]; exact hs25minew))]
  rw [stepStmts.nil]
  set s26 := s25.setReg "m_i" .real [128] (⟨fun idx : TileIndex [128] => minewFn idx.1⟩ : Tile .real [128]) with hs26d
  refine ⟨s26, rfl, ?_⟩
  -- ======= the math bridge: new (m, l, acc) = mistralFoldUptoG (c+1) =======
  have hbridge : ∀ ir dd : Fin 128,
      (minewFn ir, linewReal ir, accFinal ir dd)
        = mistralFoldUptoG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw (c + 1) ir dd := by
    intro ir dd
    rw [mistralFoldUptoG_succ]
    have hbne : mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd ≠ [] :=
      mistralBlockMG_ne_nil s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd hwin (by norm_num)
    obtain ⟨hci1, hci2⟩ := mistralFoldUptoG_channel_indep s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd ⟨0, by norm_num⟩
    set bsupDD : WithBot ℝ := (mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) |>.foldr (· ⊔ ·) ⊥ with hbsupDD
    have hbsupEq : bsup ir = bsupDD := by
      simp only [hbsup, hbsupDD, hblk]
      rw [show (fun p : ℝ × ℝ => ((p.1 : ℝ) : WithBot ℝ)) = (fun r : ℝ => ((r : ℝ) : WithBot ℝ)) ∘ Prod.fst from rfl,
        ← List.map_map, ← List.map_map,
        mistralBlockMG_fst_channel_indep s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir ⟨0, by norm_num⟩ dd]
    have hmiEq : miFn ir = (fold0 ir dd).1 := by simp only [hmiFn, hfold0]; exact hci1.symm
    have hmijGEq : mijFn ir = mistralGuard bsupDD := by simp only [hmijFn]; rw [hbsupEq]
    have hMnew : minewFn ir = (fold0 ir dd).1 ⊔ mistralGuard bsupDD := by
      simp only [hminewFn]; rw [hmiEq, hmijGEq]
    have hliEq : (fold0 ir ⟨0, by norm_num⟩).2.1 = (fold0 ir dd).2.1 := by simp only [hfold0]; exact hci2.symm
    set Gdd : ℝ := (mistralGuard bsupDD).unbotD 0 with hGddDef
    have hGddNe : mistralGuard bsupDD ≠ ⊥ := by
      apply mistralGuard_ne_bot; simp only [hbsupDD]; exact fun h => hbne ((foldr_sup_coe_bot_iff _).mp h)
    have hGddCoe : mistralGuard bsupDD = ((Gdd : ℝ) : WithBot ℝ) := by
      cases hg : mistralGuard bsupDD with
      | bot => exact absurd hg hGddNe
      | coe r => simp only [hGddDef, hg]; rfl
    have hlijEqB : lijReal ir = ((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd))).sum := by
      have hMr := mistralBlockMG_lij_sum s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd Gdd hwin
      have hsum : ((lijReal ir : ℝ) : WithBot ℝ)
          = some (((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd))).sum) := by
        rw [← hMr, hlijReal, ← WithBot.sum_some_eq_some]
        apply Finset.sum_congr rfl; intro jL _
        have hpr : pReal ir jL = (WithBot.realExp (WithBot.realSub
            ((mistralScoreM s0 Q K B_Start_Loc sc 768 128 128 128 S bel sw ir ⟨c * 128 + jL.val, mistral_lane_lt_SG c S 128 hwin jL⟩ : ℝ) : WithBot ℝ)
            ((Gdd : ℝ) : WithBot ℝ))).unbotD 0 := by
          show (WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0 = _
          rw [hqkW, hscoreFn, hmijGEq, hGddCoe]
        rw [hpr]; exact (nopad_realExp_eq_some_unbotD _).symm
      exact WithBot.coe_inj.mp hsum
    have hacccsum : Finset.univ.sum (fun jL : Fin 128 => pReal ir jL * vFn jL dd)
        = ((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd) * p.2)).sum := by
      have hvEq : ∀ jL : Fin 128, vFn jL dd
          = ctxVTileMG s0 V B_Start_Loc 768 128 S 128 bel (⟨c * 128 + jL.val, mistral_lane_lt_SG c S 128 hwin jL⟩, dd, PUnit.unit) := by
        intro jL
        show (if i + jL.val < seqLen s21 B_Seqlen then
            s21.readMem V ((startLoc s21 B_Start_Loc + (i + jL.val)) * 768 + s21.pids 1 * 128 + dd.val)
          else (0.0 : ℝ))
          = ctxVTileMG s0 V B_Start_Loc 768 128 S 128 bel (⟨c * 128 + jL.val, mistral_lane_lt_SG c S 128 hwin jL⟩, dd, PUnit.unit)
        rw [ctxVTileMG, ctxVTileG, hs21seqB, hs21slB, hs21pids1,
          show i + jL.val = c * 128 + jL.val from by rw [hi]]
        by_cases hlt : c * 128 + jL.val < bel
        · rw [if_pos hlt, if_pos hlt]; simp only [BlockState.readMem, hs21mem]
        · rw [if_neg hlt, if_neg hlt]; norm_num
      have hbr := mistralBlockMG_acc_sum s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd Gdd hwin
        (fun jL => vFn jL dd) (fun jL => hvEq jL)
      have hcoe : ((Finset.univ.sum (fun jL : Fin 128 => pReal ir jL * vFn jL dd) : ℝ) : WithBot ℝ)
          = some (((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd) * p.2)).sum) := by
        rw [← hbr, ← WithBot.sum_some_eq_some]
        apply Finset.sum_congr rfl; intro jL _
        have hpr : pReal ir jL = (WithBot.realExp (WithBot.realSub
            ((mistralScoreM s0 Q K B_Start_Loc sc 768 128 128 128 S bel sw ir ⟨c * 128 + jL.val, mistral_lane_lt_SG c S 128 hwin jL⟩ : ℝ) : WithBot ℝ)
            ((Gdd : ℝ) : WithBot ℝ))).unbotD 0 := by
          show (WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0 = _
          rw [hqkW, hscoreFn, hmijGEq, hGddCoe]
        rw [show WithBot.realMul
              (WithBot.realExp (WithBot.realSub
                ((mistralScoreM s0 Q K B_Start_Loc sc 768 128 128 128 S bel sw ir ⟨c * 128 + jL.val, mistral_lane_lt_SG c S 128 hwin jL⟩ : ℝ) : WithBot ℝ)
                ((Gdd : ℝ) : WithBot ℝ)))
              ((vFn jL dd : ℝ) : WithBot ℝ)
            = ((pReal ir jL * vFn jL dd : ℝ) : WithBot ℝ) from by
          rw [hpr, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe, WithBot.realMul_coe_coe]]
      exact WithBot.coe_inj.mp hcoe
    -- assemble
    show (minewFn ir, linewReal ir, accFinal ir dd)
      = mistralBlockStepBot (fold0 ir dd) (mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd)
    have hα : αReal ir = (WithBot.realExp (WithBot.realSub (fold0 ir dd).1 ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0 := by
      show (WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir))).unbotD 0 = _
      rw [hmiEq, hMnew]
    have hβ : βReal ir = (WithBot.realExp (WithBot.realSub (mistralGuard bsupDD) ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0 := by
      show (WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir))).unbotD 0 = _
      rw [hmijGEq, hMnew]
    have hl'eq : linewReal ir
        = (WithBot.realExp (WithBot.realSub (fold0 ir dd).1 ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0 * (fold0 ir dd).2.1
          + (WithBot.realExp (WithBot.realSub (mistralGuard bsupDD) ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0
            * ((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map (fun p => Real.exp (p.1 - (mistralGuard bsupDD).unbotD 0))).sum := by
      rw [← hα, ← hβ, ← hGddDef, ← hlijEqB]; simp only [hlinewReal]; rw [hliEq]
    simp only [mistralBlockStepBot]
    refine Prod.ext ?_ (Prod.ext ?_ ?_)
    · show minewFn ir = (fold0 ir dd).1 ⊔ mistralGuard bsupDD; exact hMnew
    · exact hl'eq
    · -- acc component
      show accFinal ir dd
        = (fold0 ir dd).2.2
            * ((fold0 ir dd).2.1
              / ((WithBot.realExp (WithBot.realSub (fold0 ir dd).1 ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0 * (fold0 ir dd).2.1
                  + (WithBot.realExp (WithBot.realSub (mistralGuard bsupDD) ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0
                    * ((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map (fun p => Real.exp (p.1 - (mistralGuard bsupDD).unbotD 0))).sum)
              * (WithBot.realExp (WithBot.realSub (fold0 ir dd).1 ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0)
          + ((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map
              (fun p => Real.exp (p.1 - (mistralGuard bsupDD).unbotD 0)
                * (WithBot.realExp (WithBot.realSub (mistralGuard bsupDD) ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0
                / ((WithBot.realExp (WithBot.realSub (fold0 ir dd).1 ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0 * (fold0 ir dd).2.1
                    + (WithBot.realExp (WithBot.realSub (mistralGuard bsupDD) ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0
                      * ((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map (fun p => Real.exp (p.1 - (mistralGuard bsupDD).unbotD 0))).sum)
                * p.2)).sum
      -- abbreviate the block value-sum
      set BV : ℝ := ((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd) * p.2)).sum with hBV
      -- RHS: rewrite raw realExp into αReal/βReal and the combined denom into linewReal
      rw [← hα, ← hβ, ← hGddDef]
      rw [show ((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd))).sum
            = lijReal ir from hlijEqB.symm]
      rw [show αReal ir * (fold0 ir dd).2.1 + βReal ir * lijReal ir = linewReal ir from by
        simp only [hlinewReal]; rw [hliEq]]
      -- RHS block sum factor: Σ block (exp(p.1-Gdd)·β/l'·p.2) = (β/l')·BV
      rw [show ((mistralBlockMG s0 Q K V B_Start_Loc sc 768 128 128 128 S bel sw c ir dd).map
            (fun p => Real.exp (p.1 - Gdd) * βReal ir / linewReal ir * p.2)).sum
          = βReal ir / linewReal ir * BV from by
        rw [hBV, ← List.sum_map_mul_left]
        apply congrArg; apply List.map_congr_left; intro p _; ring]
      -- LHS accFinal
      simp only [haccFinal, hasReal]
      rw [hliEq]
      rw [show psReal ir = βReal ir / linewReal ir from rfl]
      rw [show (fun jL : Fin 128 => pReal ir jL * (βReal ir / linewReal ir) * vFn jL dd)
            = (fun jL : Fin 128 => (βReal ir / linewReal ir) * (pReal ir jL * vFn jL dd)) from by funext jL; ring]
      rw [← Finset.mul_sum, hacccsum]
  -- ======= reconstruct the invariant =======
  have e26 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_i" → s25.regs dt sh nm = some t → s26.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs26d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have chainAll : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → nm ≠ "k" → nm ≠ "qk" → nm ≠ "m_ij" → nm ≠ "p" → nm ≠ "l_ij"
      → nm ≠ "m_i_new" → nm ≠ "alpha" → nm ≠ "beta" → nm ≠ "l_i_new" → nm ≠ "p_scale"
      → nm ≠ "acc_scale" → nm ≠ "acc" → nm ≠ "v" → nm ≠ "l_i" → nm ≠ "m_i"
      → s.regs dt sh nm = some t → s26.regs dt sh nm = some t := by
    intro dt sh nm t h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h
    exact e26 h16 (e25 h15 (e24 h13 (e23 h5 (e22 h14 (e21 h13 (e20 h12 (e19 h5 (e18 h11 (e17 h10 (e16 h10 (e15 h9 (e14 h8 (e13 h7 (e12 h6 (e11 h5 (e10 h4 (e9 h4 (e8 h3 (e7 h3 (e6 h3 (e5 h3 (e4 h3 (e3 h2 (e2 h1 (e1 h1 h)))))))))))))))))))))))))
  have hs26pids : s26.pids = s0.pids := by
    rw [hs26d, BlockState.setReg_pids, hs25d, BlockState.setReg_pids, hs24d, BlockState.setReg_pids,
      hs23d, BlockState.setReg_pids, hs22d, BlockState.setReg_pids, hs21pids]
  have hs26mem : s26.mem = s0.mem := by
    funext rg o; rw [hs26d, BlockState.setReg_mem, hs25d, BlockState.setReg_mem, hs24d, BlockState.setReg_mem,
      hs23d, BlockState.setReg_mem, hs22d, BlockState.setReg_mem]; exact hs21mem ▸ rfl
  have hs26undef : ∀ rg o, s26.undef rg o = 0 := by
    intro rg o
    rw [hs26d, BlockState.setReg_undef, hs25d, BlockState.setReg_undef, hs24d, BlockState.setReg_undef,
      hs23d, BlockState.setReg_undef, hs22d, BlockState.setReg_undef, hs21d, BlockState.setReg_undef,
      hs20d, BlockState.setReg_undef, hs19d, BlockState.setReg_undef, hs18d, BlockState.setReg_undef,
      hs17d, BlockState.setReg_undef, hs16d, BlockState.setReg_undef, hs15d, BlockState.setReg_undef,
      hs14d, BlockState.setReg_undef, hs13d, BlockState.setReg_undef, hs12d, BlockState.setReg_undef,
      hs11d, BlockState.setReg_undef, hs10d, BlockState.setReg_undef, hs9d, BlockState.setReg_undef,
      hs8d, BlockState.setReg_undef, hs7d, BlockState.setReg_undef, hs6d, BlockState.setReg_undef,
      hs5d, BlockState.setReg_undef, hs4d, BlockState.setReg_undef, hs3d, BlockState.setReg_undef,
      hs2d, BlockState.setReg_undef, hs1d, BlockState.setReg_undef]
    exact hundef rg o
  refine ⟨hs26pids, hs26mem, hs26undef, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hcb
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hch
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hsm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hckv
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hseq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hsl
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hn
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hd
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoffm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hkp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hvp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hbmask
  · rw [hs26d, BlockState.setReg_same]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, u⟩ := idx
    show minewFn ir = (mistralFoldUptoG s0 Q K V B_Start_Loc mistral_sm_scale 768 128 128 128 (ctxMistralWindowG s0 B_Seqlen 128) (seqLen s0 B_Seqlen) sw (c + 1) ir ⟨0, by norm_num⟩).1
    exact congrArg (Prod.fst) (hbridge ir ⟨0, by norm_num⟩)
  · rw [show s26.regs .real [128] "l_i" = some (⟨fun idx : TileIndex [128] => linewFn idx.1⟩ : Tile .real [128]) from by
      rw [hs26d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs25d, BlockState.setReg_same]]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, u⟩ := idx
    show linewFn ir = ((mistralFoldUptoG s0 Q K V B_Start_Loc mistral_sm_scale 768 128 128 128 (ctxMistralWindowG s0 B_Seqlen 128) (seqLen s0 B_Seqlen) sw (c + 1) ir ⟨0, by norm_num⟩).2.1 : WithBot ℝ)
    rw [hlinewSome ir]
    exact congrArg (fun p => ((p.2.1 : ℝ) : WithBot ℝ)) (hbridge ir ⟨0, by norm_num⟩)
  · rw [show s26.regs .real [128, 128] "acc" = some (⟨fun idx : TileIndex [128, 128] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [128, 128]) from by
      rw [hs26d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs25d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs24d, BlockState.setReg_same]]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, dd, u⟩ := idx
    show some (accFinal ir dd) = ((mistralFoldUptoG s0 Q K V B_Start_Loc mistral_sm_scale 768 128 128 128 (ctxMistralWindowG s0 B_Seqlen 128) (seqLen s0 B_Seqlen) sw (c + 1) ir dd).2.2 : WithBot ℝ)
    exact congrArg (fun p => ((p.2.2 : ℝ) : WithBot ℝ)) (hbridge ir dd)


end MistralAttnStep


section MistralPostExec

/-! ## PostLoop store, full-kernel exec assembly, genuine summary. -/

/-- `contextAttnMistralExactFoldMG` at FIXED `S`/`bel` transports across mem/pids-equal states. -/
theorem contextAttnMistralExactFoldMG_eq_of_mem_pids
    (s s0 : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ) (S bel sw : Nat)
    (idx : TileIndex [128, 128]) (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    contextAttnMistralExactFoldMG s Q K V B_Start_Loc sm_scale 768 128 128 128 S bel sw idx
      = contextAttnMistralExactFoldMG s0 Q K V B_Start_Loc sm_scale 768 128 128 128 S bel sw idx := by
  have hsl : startLoc s B_Start_Loc = startLoc s0 B_Start_Loc := mistral_startLoc_eq_of_mem_pids s s0 B_Start_Loc hmem hpids
  have hQt : ctxQTileG s Q B_Start_Loc 768 128 128 128 = ctxQTileG s0 Q B_Start_Loc 768 128 128 128 := by
    funext j; simp only [ctxQTileG, hsl, hpids, BlockState.readMem, hmem]
  have hKt : ctxKTileMG s K B_Start_Loc 768 128 S 128 bel = ctxKTileMG s0 K B_Start_Loc 768 128 S 128 bel := by
    funext j; simp only [ctxKTileMG, ctxKTileG, hsl, hpids, BlockState.readMem, hmem]
  have hVt : ctxVTileMG s V B_Start_Loc 768 128 S 128 bel = ctxVTileMG s0 V B_Start_Loc 768 128 S 128 bel := by
    funext j; simp only [ctxVTileMG, ctxVTileG, hsl, hpids, BlockState.readMem, hmem]
  simp only [contextAttnMistralExactFoldMG, mistralScore, ctxQTileG, ctxKTileMG, ctxKTileG,
    ctxVTileMG, ctxVTileG, hsl, hpids, BlockState.readMem, hmem]

end MistralPostExec


section MistralGeneralExec

/-! ## Dimension-general Mistral genuine exec-stepping stack.

A parallel exec stack to `MistralExec`/`MistralOpEvals`/`MistralInvariant`/
`MistralAttnStep`/`MistralPostExec`, parameterized over `(rs hs BLK DM : Nat)` with
`sm_scale : ℝ` a free parameter (NOT pinned to `(√128)⁻¹`). Map: `BLOCK_M =
BLOCK_N = BLK`, `BLOCK_DMODEL = DM`, strides `(768,128,1) → (rs,hs,1)`,
`kv_group_num = 1`. Reuses the dimension-general math foundation
(`mistralFoldUptoG`, `mistralBlockStepBot`, the bridges, the tile defs) directly. -/

/-- General lowered preLoop statements (through `block_mask`), strides `(rs, hs, 1)`,
including `cur_kv_head = cur_head // 1`. -/
def mistralPreLoopG (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (rs hs BLK DM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
    Stmt.assign .nat [] "cur_head" (Op.programId 1),
    Stmt.assign .nat [] "start_m" (Op.programId 2),
    Stmt.assign .nat [] "cur_kv_head"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 1)),
    Stmt.assign .nat [] "cur_batch_seq_len"
      (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_in_all_start_index"
      (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "block_start_loc"
      (Op.mul .nat Broadcast.nil (Op.constNat BLK) (Op.ref .nat [] "start_m")),
    Stmt.assign .nat [BLK] "offs_n" (Op.arange BLK),
    Stmt.assign .nat [DM] "offs_d" (Op.arange DM),
    Stmt.assign .nat [BLK] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLK))
        (Op.arange BLK)),
    Stmt.assign .nat [BLK, DM] "off_q"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .nat [DM, BLK] "off_k"
      (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n"))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .nat [BLK, DM] "off_v"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n"))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .real [BLK, DM] "q"
      (Op.load .real (MemAccess.region Q (Op.ref .nat [BLK, DM] "off_q"))
        (MaskOpt.maskOther
          (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [BLK, DM]))),
    Stmt.assign .ptr [DM, BLK] "k_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K) (Op.ref .nat [DM, BLK] "off_k")),
    Stmt.assign .ptr [BLK, DM] "v_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [BLK, DM] "off_v")),
    Stmt.assign .real [BLK] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLK] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLK] "l_i" (Op.full [BLK] (Op.const 0)),
    Stmt.assign .real [BLK, DM] "acc" (Op.full [BLK, DM] (Op.const 0)),
    Stmt.assign .nat [] "block_mask"
      ((Op.lt ComparableDType.nat Broadcast.nil (Op.ref .nat [] "block_start_loc")
            (Op.ref .nat [] "cur_batch_seq_len")).where
        (Op.constNat 1) (Op.constNat 0)) ]

/-- General lowered loop-body statements (25), strides `(rs, hs, 1)`: two
causal+sliding-window `where`s against the `-1e9 = 0-1e9` sentinel, the `m_ij ==
-1e9 → 0` guard, and the `l_i_new == 0 → 1e-9` guard. -/
def mistralLoopBodyG (sc : ℝ) (sw rs hs BLK DM : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    Stmt.assign .real [DM, BLK] "k"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [DM, BLK] "k_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat rs))))
        (MaskOpt.maskOther
          (Op.remap [DM, BLK] Broadcast.nil.consSame.consL.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [DM, BLK]))),
    Stmt.assign .real [BLK, BLK] "qk" (Op.full [BLK, BLK] (Op.const 0)),
    Stmt.assign .real [BLK, BLK] "qk"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLK, BLK] "qk")
        (Op.dot (batch := []) (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k"))),
    Stmt.assign .real [BLK, BLK] "qk"
      (Op.mul .real Broadcast.scalarR (Op.ref .real [BLK, BLK] "qk") (Op.const sc)),
    Stmt.assign .real [BLK, BLK] "qk"
      ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))).where
        (Op.ref .real [BLK, BLK] "qk")
        ((Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9)).broadcast [BLK, BLK])),
    Stmt.assign .real [BLK, BLK] "qk"
      ((Op.gt ComparableDType.nat Broadcast.nil.consR.consL
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
            (Op.sub .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
              (Op.constNat sw))).where
        (Op.ref .real [BLK, BLK] "qk")
        ((Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9)).broadcast [BLK, BLK])),
    Stmt.assign .real [BLK] "m_ij" (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BLK, BLK] "qk")),
    Stmt.assign .real [BLK] "m_ij"
      ((Op.eq ComparableDType.real Broadcast.scalarR (Op.ref .real [BLK] "m_ij")
            (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9))).where
        ((Op.const 0.0).broadcast [BLK]) (Op.ref .real [BLK] "m_ij")),
    Stmt.assign .real [BLK, BLK] "p"
      (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, BLK] "qk")
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "m_ij"))).exp,
    Stmt.assign .real [BLK] "l_ij" (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [BLK, BLK] "p")),
    Stmt.assign .real [BLK] "m_i_new"
      ((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref .real [BLK] "m_i")
            (Op.ref .real [BLK] "m_ij")).where
        (Op.ref .real [BLK] "m_i") (Op.ref .real [BLK] "m_ij")),
    Stmt.assign .real [BLK] "alpha"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BLK] "m_i")
          (Op.ref .real [BLK] "m_i_new")).exp,
    Stmt.assign .real [BLK] "beta"
      (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BLK] "m_ij")
          (Op.ref .real [BLK] "m_i_new")).exp,
    Stmt.assign .real [BLK] "l_i_new"
      (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BLK] "alpha")
          (Op.ref .real [BLK] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BLK] "beta")
          (Op.ref .real [BLK] "l_ij"))),
    Stmt.assign .real [BLK] "l_i_new"
      ((Op.eq ComparableDType.real Broadcast.scalarR (Op.ref .real [BLK] "l_i_new")
            (Op.const 0.0)).where
        ((Op.const 1e-9).broadcast [BLK]) (Op.ref .real [BLK] "l_i_new")),
    Stmt.assign .real [BLK] "p_scale"
      (Op.div .real Broadcast.nil.consSame (Op.ref .real [BLK] "beta")
        (Op.ref .real [BLK] "l_i_new")),
    Stmt.assign .real [BLK, BLK] "p"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, BLK] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "p_scale"))),
    Stmt.assign .real [BLK] "acc_scale"
      (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [BLK] "l_i")
          (Op.ref .real [BLK] "l_i_new"))
        (Op.ref .real [BLK] "alpha")),
    Stmt.assign .real [BLK, DM] "acc"
      (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, DM] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "acc_scale"))),
    Stmt.assign .real [BLK, DM] "v"
      (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLK, DM] "v_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat rs))))
        (MaskOpt.maskOther
          (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [BLK, DM]))),
    Stmt.assign .real [BLK, BLK] "p" (Op.ref .real [BLK, BLK] "p"),
    Stmt.assign .real [BLK, DM] "acc"
      (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLK, DM] "acc")
        (Op.dot (batch := []) (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v"))),
    Stmt.assign .real [BLK] "l_i" (Op.ref .real [BLK] "l_i_new"),
    Stmt.assign .real [BLK] "m_i" (Op.ref .real [BLK] "m_i_new") ]

/-- General lowered postLoop statements (3 stmts), strides `(rs, hs, 1)`. -/
def mistralPostLoopG (Out : RegionName) (rs hs BLK DM : Nat) : List Stmt :=
  [ Stmt.assign .nat [BLK, DM] "off_o"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .ptr [BLK, DM] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLK, DM] "off_o")),
    Stmt.store .real [BLK, DM] (MemAccess.ptr (Op.ref .ptr [BLK, DM] "out_ptrs"))
      (Op.ref .real [BLK, DM] "acc")
      (MaskOpt.mask
        (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len")))) ]

set_option maxRecDepth 8000 in
/-- General body split. -/
theorem mistral_body_splitG
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName) (sc : ℝ)
    (rs hs BLK DM sw : Nat) :
    (context_attn_mistral_fwd_kernel_surface Q K V sc B_Start_Loc B_Seqlen Out
      rs hs 1 rs hs 1 rs hs 1 rs hs 1 1 sw BLK DM BLK).toAlgKernel.body
      = mistralPreLoopG Q K V B_Start_Loc B_Seqlen rs hs BLK DM
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
              (Op.mul .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
                  (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
                (Op.constNat BLK))
              (Op.constNat BLK) (mistralLoopBodyG sc sw rs hs BLK DM)
            :: mistralPostLoopG Out rs hs BLK DM) := by
  rfl

set_option maxHeartbeats 1600000 in
/-- General `off_q` recipe. -/
theorem mistral_offq_evalG (s : BlockState) (sl head pid2 rs hs BLK DM : Nat)
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar sl))
    (hch : s.regs .nat [] "cur_head" = some (Tile.scalar head))
    (hm : s.regs .nat [BLK] "offs_m" = some (Tile.vec (fun i : Fin BLK => pid2 * BLK + i.val)))
    (hd : s.regs .nat [DM] "offs_d" = some (Tile.vec (fun e : Fin DM => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          (sl + (pid2 * BLK + idx.1.val)) * rs + head * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) := by
  have hexpM := evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" s _ hm
  have hexpD := evalOp_expandDim_ref_of_regs .nat [DM] ⟨0, by simp⟩ "offs_d" s _ hd
  rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_add, evalOp_ref, hsl]
  erw [hexpM]
  rw [evalOp_mul, evalOp_ref, hch, evalOp_mul]
  erw [hexpD]
  rw [evalOp_constNat, evalOp_constNat, evalOp_constNat]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  obtain ⟨i, e, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    TileShape.dropInsertedIndex]
  ring

set_option maxHeartbeats 1600000 in
/-- General `off_k` recipe (lane `(e, jL)`). -/
theorem mistral_offk_evalG (s : BlockState) (kvhead rs hs BLK DM : Nat)
    (hch : s.regs .nat [] "cur_kv_head" = some (Tile.scalar kvhead))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)))
    (hd : s.regs .nat [DM] "offs_d" = some (Tile.vec (fun e : Fin DM => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n"))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [DM, BLK] =>
          idx.2.1.val * rs + kvhead * hs + idx.1.val⟩ : Tile .nat [DM, BLK]) := by
  have hexpN := evalOp_expandDim_ref_of_regs .nat [BLK] ⟨0, by simp⟩ "offs_n" s _ hn
  have hexpD := evalOp_expandDim_ref_of_regs .nat [DM] ⟨1, by simp⟩ "offs_d" s _ hd
  rw [evalOp_add, evalOp_add, evalOp_mul]
  erw [hexpN]
  rw [evalOp_mul, evalOp_ref, hch, evalOp_mul]
  erw [hexpD]
  rw [evalOp_constNat, evalOp_constNat, evalOp_constNat]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  obtain ⟨e, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    TileShape.dropInsertedIndex]
  ring

set_option maxHeartbeats 1600000 in
/-- General `off_v` recipe (lane `(jL, d)`). -/
theorem mistral_offv_evalG (s : BlockState) (kvhead rs hs BLK DM : Nat)
    (hch : s.regs .nat [] "cur_kv_head" = some (Tile.scalar kvhead))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)))
    (hd : s.regs .nat [DM] "offs_d" = some (Tile.vec (fun e : Fin DM => e.val))) :
    evalOp (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n"))
            (Op.constNat rs))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_kv_head") (Op.constNat hs)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [DM] "offs_d"))
          (Op.constNat 1))) s
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          idx.1.val * rs + kvhead * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) := by
  have hexpN := evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_n" s _ hn
  have hexpD := evalOp_expandDim_ref_of_regs .nat [DM] ⟨0, by simp⟩ "offs_d" s _ hd
  rw [evalOp_add, evalOp_add, evalOp_mul]
  erw [hexpN]
  rw [evalOp_mul, evalOp_ref, hch, evalOp_mul]
  erw [hexpD]
  rw [evalOp_constNat, evalOp_constNat, evalOp_constNat]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_; ext idx
  obtain ⟨jL, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Tile.expandDim,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    TileShape.dropInsertedIndex]
  ring

set_option maxHeartbeats 1600000 in
/-- General `q` masked-load recipe. -/
theorem mistral_q_load_evalG (s : BlockState) (Q B_Start_Loc B_Seqlen : RegionName) (rs hs BLK DM : Nat)
    (hoffq : s.regs .nat [BLK, DM] "off_q" = some (⟨fun idx : TileIndex [BLK, DM] =>
        (startLoc s B_Start_Loc + (s.pids 2 * BLK + idx.1.val)) * rs + s.pids 1 * hs + idx.2.1.val⟩
        : Tile .nat [BLK, DM]))
    (hm : s.regs .nat [BLK] "offs_m" = some (Tile.vec (fun i : Fin BLK => s.pids 2 * BLK + i.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real (MemAccess.region Q (Op.ref .nat [BLK, DM] "off_q"))
        (MaskOpt.maskOther
          (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [BLK, DM]))) s
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          some (ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM (seqLen s B_Seqlen) (idx.1, idx.2.1, PUnit.unit))⟩
          : Tile .real [BLK, DM]) := by
  have hexpM := evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" s _ hm
  simp only [evalOp, hoffq, hseq, Option.bind]
  erw [hexpM]
  refine congrArg some ?_; ext idx
  obtain ⟨i, e, u⟩ := idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim, Tile.vec,
    Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt, Broadcast.leftIndex,
    Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    TileShape.dropInsertedIndex, ctxQTileMRowG, ctxQTileG]
  by_cases hlt : s.pids 2 * BLK + i.val < seqLen s B_Seqlen
  · rw [if_pos hlt]
    simp only [hlt, decide_true, if_true, if_pos hlt]
  · rw [if_neg hlt]
    simp only [hlt, decide_false, if_false, Bool.false_eq_true]
    norm_num

set_option maxHeartbeats 1600000 in
/-- General `k` masked-load recipe (lane `(e, jL)`). -/
theorem mistral_k_load_evalG (s : BlockState) (K B_Start_Loc B_Seqlen : RegionName) (SN rs hs BLK DM : Nat)
    (hkp : s.regs .ptr [DM, BLK] "k_ptrs" =
      some (⟨fun idx : TileIndex [DM, BLK] =>
        (K, idx.2.1.val * rs + s.pids 1 * hs + idx.1.val)⟩
        : Tile .ptr [DM, BLK]))
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s B_Start_Loc)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [DM, BLK] "k_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat rs))))
        (MaskOpt.maskOther
          (Op.remap [DM, BLK] Broadcast.nil.consSame.consL.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [DM, BLK]))) s
      = some (⟨fun idx : TileIndex [DM, BLK] =>
          if SN + idx.2.1.val < seqLen s B_Seqlen then
            some (s.readMem K ((startLoc s B_Start_Loc + (SN + idx.2.1.val)) * rs
              + s.pids 1 * hs + idx.1.val))
          else some (0.0 : ℝ)⟩ : Tile .real [DM, BLK]) := by
  have hexp : @evalOp .nat [1, BLK] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BLK => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨0, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hkp, hsl, hsn, hexp, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨e, jL, u⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim,
    Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    BlockState.readMem, TileShape.dropInsertedIndex]
  by_cases hlt : SN + jL.val < seqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMem]
    rw [show jL.val * rs + s.pids 1 * hs + e.val + (startLoc s B_Start_Loc + SN) * rs
        = (startLoc s B_Start_Loc + (SN + jL.val)) * rs + s.pids 1 * hs + e.val from by ring]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- General `v` masked-load recipe (lane `(jL, d)`). -/
theorem mistral_v_load_evalG (s : BlockState) (V B_Start_Loc B_Seqlen : RegionName) (SN rs hs BLK DM : Nat)
    (hvp : s.regs .ptr [BLK, DM] "v_ptrs" =
      some (⟨fun idx : TileIndex [BLK, DM] =>
        (V, idx.1.val * rs + s.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]))
    (hsl : s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s B_Start_Loc)))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)))
    (hseq : s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s B_Seqlen))) :
    evalOp (Op.load .real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLK, DM] "v_ptrs")
            (Op.mul .nat Broadcast.nil
              (Op.add .nat Broadcast.nil (Op.ref .nat [] "cur_batch_in_all_start_index")
                (Op.ref .nat [] "start_n"))
              (Op.constNat rs))))
        (MaskOpt.maskOther
          (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
                (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [BLK, DM]))) s
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          if SN + idx.1.val < seqLen s B_Seqlen then
            some (s.readMem V ((startLoc s B_Start_Loc + (SN + idx.1.val)) * rs
              + s.pids 1 * hs + idx.2.1.val))
          else some (0.0 : ℝ)⟩ : Tile .real [BLK, DM]) := by
  have hexp : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_n")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun j : Fin BLK => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hvp, hsl, hsn, hexp, hseq, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨jL, d, u⟩ := idx
  simp only [Tile.ptrAdd_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim,
    Tile.vec, Tile.scalar, NumericDType.add, NumericDType.mul, ComparableDType.lt,
    Broadcast.leftIndex, Broadcast.rightIndex, BlockState.readMemValue_real, Region.cast_id,
    BlockState.readMem, TileShape.dropInsertedIndex]
  by_cases hlt : SN + jL.val < seqLen s B_Seqlen
  · simp only [hlt, decide_true, if_true, if_pos hlt, BlockState.readMem]
    rw [show jL.val * rs + s.pids 1 * hs + d.val + (startLoc s B_Start_Loc + SN) * rs
        = (startLoc s B_Start_Loc + (SN + jL.val)) * rs + s.pids 1 * hs + d.val from by ring]
  · simp only [hlt, decide_false, if_false, if_neg hlt, Bool.false_eq_true]

set_option maxHeartbeats 1600000 in
/-- General `qk += tl.dot(q, k)` recipe (contracts over `Fin DM`). -/
theorem mistral_qk_dot_evalG (s : BlockState) (BLK DM : Nat) (qFn : Fin BLK → Fin DM → ℝ)
    (kFn : Fin DM → Fin BLK → ℝ)
    (hqk0 : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun _ : TileIndex [BLK, BLK] => some (0 : ℝ)⟩ : Tile .real [BLK, BLK]))
    (hq : s.regs .real [BLK, DM] "q"
      = some (⟨fun idx : TileIndex [BLK, DM] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]))
    (hk : s.regs .real [DM, BLK] "k"
      = some (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK])) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLK, BLK] "qk")
        (Op.dot (batch := []) (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k"))) s
      = some (⟨fun idx : TileIndex [BLK, BLK] =>
          some (Finset.univ.sum (fun e : Fin DM => qFn idx.1 e * kFn e idx.2.1))⟩
          : Tile .real [BLK, BLK]) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k")) s
      = some (Tile.dot [] (⟨fun idx : TileIndex [BLK, DM] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM])
          (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK])) := by
    rw [evalOp_dot]; erw [evalOp_ref, hq, evalOp_ref, hk]; rfl
  rw [evalOp_add, evalOp_ref, hqk0]
  show Option.bind (evalOp (Op.dot (batch := []) (Op.ref .real [BLK, DM] "q") (Op.ref .real [DM, BLK] "k")) s) _ = _
  rw [hdot]
  simp only [Option.bind]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, WithBot.realAdd]
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin DM) (WithBot ℝ) _ Finset.univ fun e : Fin DM =>
        Option.map₂ (· * ·) ((⟨fun idx : TileIndex [BLK, DM] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]).data (i, e, PUnit.unit))
          ((⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK]).data (e, jL, PUnit.unit)))
      = (@Finset.sum (Fin DM) (WithBot ℝ) _ Finset.univ fun e : Fin DM => ((qFn i e * kFn e jL : ℝ) : WithBot ℝ)) from by
    apply Finset.sum_congr rfl; intro e _; rfl]
  rw [WithBot.sum_some_eq_some]
  show Option.map₂ (· + ·) (some 0) (some _) = _
  simp only [Option.map₂, Option.bind, Option.map, zero_add]

set_option maxHeartbeats 1600000 in
/-- General `qk *= sm_scale` recipe. -/
theorem mistral_qk_scale_evalG (s : BlockState) (BLK : Nat) (sc : ℝ) (rawFn : Fin BLK → Fin BLK → ℝ)
    (hqk : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun idx : TileIndex [BLK, BLK] => some (rawFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK])) :
    evalOp (Op.mul .real Broadcast.scalarR (Op.ref .real [BLK, BLK] "qk") (Op.const sc)) s
      = some (⟨fun idx : TileIndex [BLK, BLK] => some (rawFn idx.1 idx.2.1 * sc)⟩
          : Tile .real [BLK, BLK]) := by
  rw [evalOp_mul, evalOp_ref, hqk, evalOp_const]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul]
  rfl

set_option maxHeartbeats 1600000 in
/-- General **first `where`** recipe (causal, `-1e9` sentinel). -/
theorem mistral_qk_where1_evalG (s : BlockState) (SN BLK : Nat) (qkFn : Fin BLK → Fin BLK → ℝ)
    (offsM : Fin BLK → Nat)
    (hqk : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun idx : TileIndex [BLK, BLK] => some (qkFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]))
    (hm : s.regs .nat [BLK] "offs_m" = some (Tile.vec offsM))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val))) :
    evalOp ((Op.ge ComparableDType.nat Broadcast.nil.consL.consR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))).where
        (Op.ref .real [BLK, BLK] "qk")
        ((Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9)).broadcast [BLK, BLK])) s
      = some (⟨fun idx : TileIndex [BLK, BLK] =>
          if SN + idx.2.1.val ≤ offsM idx.1 then some (qkFn idx.1 idx.2.1)
          else some (-1e9 : ℝ)⟩ : Tile .real [BLK, BLK]) := by
  have hexpM : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec offsM)) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" s _ hm
  have hexpN : @evalOp .nat [1, BLK] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BLK => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨0, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hexpM, hsn, hexpN, hqk, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Tile.vec,
    Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.ge, NumericDType.add,
    NumericDType.sub, WithBot.realSub, TileShape.dropInsertedIndex]
  by_cases hle : SN + jL.val ≤ offsM i
  · rw [if_pos (by simp only [decide_eq_true_eq]; omega), if_pos hle]
  · rw [if_neg (by simp only [decide_eq_true_eq]; omega), if_neg hle]
    norm_num

set_option maxHeartbeats 1600000 in
/-- General **second `where`** recipe (sliding window, `-1e9` sentinel). -/
theorem mistral_qk_where2_evalG (s : BlockState) (SN sw BLK : Nat) (gFn : Fin BLK → Fin BLK → ℝ)
    (offsM : Fin BLK → Nat)
    (hqk : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun idx : TileIndex [BLK, BLK] => some (gFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]))
    (hm : s.regs .nat [BLK] "offs_m" = some (Tile.vec offsM))
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hn : s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val))) :
    evalOp ((Op.gt ComparableDType.nat Broadcast.nil.consR.consL
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
              (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")))
            (Op.sub .nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
              (Op.constNat sw))).where
        (Op.ref .real [BLK, BLK] "qk")
        ((Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9)).broadcast [BLK, BLK])) s
      = some (⟨fun idx : TileIndex [BLK, BLK] =>
          if offsM idx.1 - sw < SN + idx.2.1.val then some (gFn idx.1 idx.2.1)
          else some (-1e9 : ℝ)⟩ : Tile .real [BLK, BLK]) := by
  have hexpM : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec offsM)) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" s _ hm
  have hexpN : @evalOp .nat [1, BLK] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLK] "offs_n")) s
      = some (Tile.expandDim ⟨0, by simp⟩ (Tile.vec (fun j : Fin BLK => j.val))) :=
    evalOp_expandDim_ref_of_regs .nat [BLK] ⟨0, by simp⟩ "offs_n" s _ hn
  simp only [evalOp, hexpM, hsn, hexpN, hqk, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Tile.vec,
    Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, NumericDType.add,
    NumericDType.sub, WithBot.realSub, TileShape.dropInsertedIndex]
  by_cases hlt : offsM i - sw < SN + jL.val
  · rw [if_pos (by simp only [decide_eq_true_eq]; omega), if_pos hlt]
  · rw [if_neg (by simp only [decide_eq_true_eq]; omega), if_neg hlt]
    norm_num

set_option maxHeartbeats 1600000 in
/-- General **`m_ij` guard** recipe. -/
theorem mistral_mij_guard_evalG (s : BlockState) (BLK : Nat) (mijFn : Fin BLK → WithBot ℝ)
    (hmij : s.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK])) :
    evalOp ((Op.eq ComparableDType.real Broadcast.scalarR (Op.ref .real [BLK] "m_ij")
            (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1e9))).where
        ((Op.const 0.0).broadcast [BLK]) (Op.ref .real [BLK] "m_ij")) s
      = some (⟨fun idx : TileIndex [BLK] => mistralGuard (mijFn idx.1)⟩ : Tile .real [BLK]) := by
  simp only [evalOp, hmij, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.cop, Tile.bop_data, Tile.bop, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.sub, WithBot.realSub,
    Option.map₂_some_some]
  unfold mistralGuard
  by_cases hb : (mijFn i).unbotD 0 = (-1e9 : ℝ)
  · rw [if_pos hb]
    have hcoe : mijFn i = (((-1e9 : ℝ)) : WithBot ℝ) := by
      cases hm : mijFn i with
      | bot => rw [hm] at hb; simp at hb; norm_num at hb
      | coe r => rw [hm] at hb; rw [WithBot.unbotD_coe] at hb; rw [hb]
    rw [if_pos (show ComparableDType.real.eq (mijFn i) (some ((0.0:ℝ) - 1e9)) = Bool.true by
      rw [ComparableDType.real_eq_eq_true, hcoe]
      show ((-1e9 : ℝ) : WithBot ℝ) = some ((0.0:ℝ) - 1e9)
      rw [show ((0.0:ℝ) - 1e9) = (-1e9:ℝ) from by norm_num]; rfl)]
    show some (0.0 : ℝ) = ((0:ℝ) : WithBot ℝ)
    rw [show (0.0:ℝ) = (0:ℝ) from by norm_num]; rfl
  · rw [if_neg hb]
    rw [if_neg (show ¬ (ComparableDType.real.eq (mijFn i) (some ((0.0:ℝ) - 1e9)) = Bool.true) by
      rw [ComparableDType.real_eq_eq_true]
      intro heq
      apply hb
      rw [heq]; norm_num)]

set_option maxHeartbeats 1600000 in
/-- General `m_ij = tl.max(qk, 1)` recipe. -/
theorem mistral_mij_evalG (s : BlockState) (BLK : Nat) (hBLK : 0 < BLK) (qkFn : Fin BLK → Fin BLK → WithBot ℝ)
    (hqk : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun idx : TileIndex [BLK, BLK] => qkFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK])) :
    evalOp (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [BLK, BLK] "qk")) s
      = some (⟨fun idx : TileIndex [BLK] =>
          Finset.univ.sup' (⟨⟨0, hBLK⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLK)).Nonempty)
            (fun jL : Fin BLK => qkFn idx.1 jL)⟩ : Tile .real [BLK]) := by
  rw [evalOp_reduceMax, evalOp_ref, hqk]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceMax_false]
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show 0 < TileShape.axisDim [BLK, BLK] (⟨1, by simp⟩ : Fin [BLK,BLK].length) from hBLK)]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  rfl

set_option maxHeartbeats 1600000 in
/-- General `p = tl.exp(qk - m_ij[:, None])` recipe. -/
theorem mistral_p_evalG (s : BlockState) (BLK : Nat) (qkFn : Fin BLK → Fin BLK → WithBot ℝ)
    (mijFn : Fin BLK → WithBot ℝ)
    (hqk : s.regs .real [BLK, BLK] "qk"
      = some (⟨fun idx : TileIndex [BLK, BLK] => qkFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]))
    (hmij : s.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.sub .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, BLK] "qk")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "m_ij"))).exp s
      = some (⟨fun idx : TileIndex [BLK, BLK] =>
          WithBot.realExp (WithBot.realSub (qkFn idx.1 idx.2.1) (mijFn idx.1))⟩ : Tile .real [BLK, BLK]) := by
  have hexp : @evalOp .real [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "m_ij")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK])) :=
    evalOp_expandDim_ref_of_regs .real [BLK] ⟨1, by simp⟩ "m_ij" s _ hmij
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hqk, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex,
    Broadcast.rightIndex, NumericDType.sub, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- General `l_ij = tl.sum(p, 1)` recipe. -/
theorem mistral_lij_evalG (s : BlockState) (BLK : Nat) (pFn : Fin BLK → Fin BLK → WithBot ℝ)
    (hp : s.regs .real [BLK, BLK] "p"
      = some (⟨fun idx : TileIndex [BLK, BLK] => pFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK])) :
    evalOp (Op.reduceSum ⟨1, by simp⟩ Bool.false (Op.ref .real [BLK, BLK] "p")) s
      = some (⟨fun idx : TileIndex [BLK] =>
          (Finset.univ.sum (fun jL : Fin BLK => pFn idx.1 jL))⟩ : Tile .real [BLK]) := by
  rw [evalOp_reduceSum, evalOp_ref, hp]
  simp only [Option.bind_eq_bind, Option.bind_some, Tile.reduceSum_false]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  rfl

set_option maxHeartbeats 1600000 in
/-- General `m_i_new = tl.maximum(m_i, m_ij)` recipe. -/
theorem mistral_minew_evalG (s : BlockState) (BLK : Nat) (miFn mijFn : Fin BLK → WithBot ℝ)
    (hmi : s.regs .real [BLK] "m_i" = some (⟨fun idx : TileIndex [BLK] => miFn idx.1⟩ : Tile .real [BLK]))
    (hmij : s.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK])) :
    evalOp ((Op.gt ComparableDType.real Broadcast.nil.consSame (Op.ref .real [BLK] "m_i")
            (Op.ref .real [BLK] "m_ij")).where
        (Op.ref .real [BLK] "m_i") (Op.ref .real [BLK] "m_ij")) s
      = some (⟨fun idx : TileIndex [BLK] => miFn idx.1 ⊔ mijFn idx.1⟩ : Tile .real [BLK]) := by
  rw [evalOp_where, evalOp_gt]
  simp only [evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.gt]
  by_cases h : mijFn i < miFn i
  · rw [if_pos (by simpa using h), max_eq_left (le_of_lt h)]
  · rw [if_neg (by simpa using h), max_eq_right (not_lt.mp h)]

set_option maxHeartbeats 1600000 in
/-- General `alpha = tl.exp(m_i - m_i_new)` recipe. -/
theorem mistral_alpha_evalG (s : BlockState) (BLK : Nat) (miFn minewFn : Fin BLK → WithBot ℝ)
    (hmi : s.regs .real [BLK] "m_i" = some (⟨fun idx : TileIndex [BLK] => miFn idx.1⟩ : Tile .real [BLK]))
    (hminew : s.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BLK] "m_i")
        (Op.ref .real [BLK] "m_i_new")).exp s
      = some (⟨fun idx : TileIndex [BLK] => WithBot.realExp (WithBot.realSub (miFn idx.1) (minewFn idx.1))⟩
          : Tile .real [BLK]) := by
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hmi, evalOp_ref, hminew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub]

set_option maxHeartbeats 1600000 in
/-- General `beta = tl.exp(m_ij - m_i_new)` recipe. -/
theorem mistral_beta_evalG (s : BlockState) (BLK : Nat) (mijFn minewFn : Fin BLK → WithBot ℝ)
    (hmij : s.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]))
    (hminew : s.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.sub .real Broadcast.nil.consSame (Op.ref .real [BLK] "m_ij")
        (Op.ref .real [BLK] "m_i_new")).exp s
      = some (⟨fun idx : TileIndex [BLK] => WithBot.realExp (WithBot.realSub (mijFn idx.1) (minewFn idx.1))⟩
          : Tile .real [BLK]) := by
  rw [evalOp_exp, evalOp_sub, evalOp_ref, hmij, evalOp_ref, hminew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.uop_data, Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.sub]

set_option maxHeartbeats 1600000 in
/-- General `l_i_new = alpha*l_i + beta*l_ij` recipe. -/
theorem mistral_linew_evalG (s : BlockState) (BLK : Nat) (alphaFn liFn betaFn lijFn : Fin BLK → WithBot ℝ)
    (halpha : s.regs .real [BLK] "alpha" = some (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK]))
    (hli : s.regs .real [BLK] "l_i" = some (⟨fun idx : TileIndex [BLK] => liFn idx.1⟩ : Tile .real [BLK]))
    (hbeta : s.regs .real [BLK] "beta" = some (⟨fun idx : TileIndex [BLK] => betaFn idx.1⟩ : Tile .real [BLK]))
    (hlij : s.regs .real [BLK] "l_ij" = some (⟨fun idx : TileIndex [BLK] => lijFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.add .real Broadcast.nil.consSame
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BLK] "alpha") (Op.ref .real [BLK] "l_i"))
        (Op.mul .real Broadcast.nil.consSame (Op.ref .real [BLK] "beta") (Op.ref .real [BLK] "l_ij"))) s
      = some (⟨fun idx : TileIndex [BLK] =>
          WithBot.realAdd (WithBot.realMul (alphaFn idx.1) (liFn idx.1))
            (WithBot.realMul (betaFn idx.1) (lijFn idx.1))⟩ : Tile .real [BLK]) := by
  rw [evalOp_add, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, halpha, hli, hbeta, hlij, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]

set_option maxHeartbeats 1600000 in
/-- General **`l_i_new` guard** recipe. -/
theorem mistral_linew_guard_evalG (s : BlockState) (BLK : Nat) (linewFn : Fin BLK → WithBot ℝ)
    (hlinew : s.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK])) :
    evalOp ((Op.eq ComparableDType.real Broadcast.scalarR (Op.ref .real [BLK] "l_i_new")
            (Op.const 0.0)).where
        ((Op.const 1e-9).broadcast [BLK]) (Op.ref .real [BLK] "l_i_new")) s
      = some (⟨fun idx : TileIndex [BLK] =>
          if linewFn idx.1 = ((0 : ℝ) : WithBot ℝ) then ((1e-9 : ℝ) : WithBot ℝ) else linewFn idx.1⟩
          : Tile .real [BLK]) := by
  simp only [evalOp, hlinew, Option.bind, Option.some.injEq]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.select_data, Tile.cop_data, Tile.cop, Tile.bop_data, Tile.bop, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex]
  by_cases hz : linewFn i = ((0 : ℝ) : WithBot ℝ)
  · rw [if_pos (show ComparableDType.real.eq (linewFn i) (some (0.0:ℝ)) = Bool.true by
        rw [ComparableDType.real_eq_eq_true, hz]
        show ((0:ℝ) : WithBot ℝ) = some (0.0:ℝ)
        rw [show (0.0:ℝ) = (0:ℝ) from by norm_num]; rfl),
      if_pos hz]
    rfl
  · rw [if_neg (show ¬ ComparableDType.real.eq (linewFn i) (some (0.0:ℝ)) = Bool.true by
        rw [ComparableDType.real_eq_eq_true]; intro h; apply hz
        rw [h]; show ((0.0:ℝ) : WithBot ℝ) = ((0:ℝ) : WithBot ℝ)
        rw [show (0.0:ℝ) = (0:ℝ) from by norm_num]),
      if_neg hz]

set_option maxHeartbeats 1600000 in
/-- General `p_scale = beta / l_i_new` recipe. -/
theorem mistral_pscale_evalG (s : BlockState) (BLK : Nat) (betaFn linewFn : Fin BLK → WithBot ℝ)
    (hbeta : s.regs .real [BLK] "beta" = some (⟨fun idx : TileIndex [BLK] => betaFn idx.1⟩ : Tile .real [BLK]))
    (hlinew : s.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.div .real Broadcast.nil.consSame (Op.ref .real [BLK] "beta")
        (Op.ref .real [BLK] "l_i_new")) s
      = some (⟨fun idx : TileIndex [BLK] => WithBot.realDiv (betaFn idx.1) (linewFn idx.1)⟩ : Tile .real [BLK]) := by
  rw [evalOp_div, evalOp_ref, hbeta, evalOp_ref, hlinew]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div]

set_option maxHeartbeats 1600000 in
/-- General `p = p * p_scale[:, None]` recipe. -/
theorem mistral_pmul_evalG (s : BlockState) (BLK : Nat) (pFn : Fin BLK → Fin BLK → WithBot ℝ)
    (psFn : Fin BLK → WithBot ℝ)
    (hp : s.regs .real [BLK, BLK] "p"
      = some (⟨fun idx : TileIndex [BLK, BLK] => pFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]))
    (hps : s.regs .real [BLK] "p_scale" = some (⟨fun idx : TileIndex [BLK] => psFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, BLK] "p")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "p_scale"))) s
      = some (⟨fun idx : TileIndex [BLK, BLK] => WithBot.realMul (pFn idx.1 idx.2.1) (psFn idx.1)⟩
          : Tile .real [BLK, BLK]) := by
  have hexp : @evalOp .real [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "p_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [BLK] => psFn idx.1⟩ : Tile .real [BLK])) :=
    evalOp_expandDim_ref_of_regs .real [BLK] ⟨1, by simp⟩ "p_scale" s _ hps
  rw [evalOp_mul, evalOp_ref, hp, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, jL, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- General `acc_scale = l_i / l_i_new * alpha` recipe. -/
theorem mistral_accscale_evalG (s : BlockState) (BLK : Nat) (liFn linewFn alphaFn : Fin BLK → WithBot ℝ)
    (hli : s.regs .real [BLK] "l_i" = some (⟨fun idx : TileIndex [BLK] => liFn idx.1⟩ : Tile .real [BLK]))
    (hlinew : s.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]))
    (halpha : s.regs .real [BLK] "alpha" = some (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.mul .real Broadcast.nil.consSame
        (Op.div .real Broadcast.nil.consSame (Op.ref .real [BLK] "l_i") (Op.ref .real [BLK] "l_i_new"))
        (Op.ref .real [BLK] "alpha")) s
      = some (⟨fun idx : TileIndex [BLK] =>
          WithBot.realMul (WithBot.realDiv (liFn idx.1) (linewFn idx.1)) (alphaFn idx.1)⟩ : Tile .real [BLK]) := by
  rw [evalOp_mul, evalOp_div]
  simp only [evalOp_ref, hli, hlinew, halpha, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, NumericDType.div]

set_option maxHeartbeats 1600000 in
/-- General `acc = acc * acc_scale[:, None]` recipe. -/
theorem mistral_accmul_evalG (s : BlockState) (BLK DM : Nat) (accFn : Fin BLK → Fin DM → WithBot ℝ)
    (asFn : Fin BLK → WithBot ℝ)
    (hacc : s.regs .real [BLK, DM] "acc"
      = some (⟨fun idx : TileIndex [BLK, DM] => accFn idx.1 idx.2.1⟩ : Tile .real [BLK, DM]))
    (has : s.regs .real [BLK] "acc_scale" = some (⟨fun idx : TileIndex [BLK] => asFn idx.1⟩ : Tile .real [BLK])) :
    evalOp (Op.mul .real Broadcast.nil.consR.consSame (Op.ref .real [BLK, DM] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "acc_scale"))) s
      = some (⟨fun idx : TileIndex [BLK, DM] => WithBot.realMul (accFn idx.1 idx.2.1) (asFn idx.1)⟩
          : Tile .real [BLK, DM]) := by
  have hexp : @evalOp .real [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLK] "acc_scale")) s
      = some (Tile.expandDim ⟨1, by simp⟩ (⟨fun idx : TileIndex [BLK] => asFn idx.1⟩ : Tile .real [BLK])) :=
    evalOp_expandDim_ref_of_regs .real [BLK] ⟨1, by simp⟩ "acc_scale" s _ has
  rw [evalOp_mul, evalOp_ref, hacc, hexp]
  simp only [Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Tile.expandDim, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.mul, TileShape.dropInsertedIndex]

set_option maxHeartbeats 1600000 in
/-- General `acc += tl.dot(p, v)` recipe (contracts over `Fin BLK`). -/
theorem mistral_acc_dot_evalG (s : BlockState) (BLK DM : Nat) (accFn : Fin BLK → Fin DM → WithBot ℝ)
    (pFn : Fin BLK → Fin BLK → ℝ) (vFn : Fin BLK → Fin DM → ℝ)
    (hacc : s.regs .real [BLK, DM] "acc"
      = some (⟨fun idx : TileIndex [BLK, DM] => accFn idx.1 idx.2.1⟩ : Tile .real [BLK, DM]))
    (hp : s.regs .real [BLK, BLK] "p"
      = some (⟨fun idx : TileIndex [BLK, BLK] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]))
    (hv : s.regs .real [BLK, DM] "v"
      = some (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM])) :
    evalOp (Op.add .real Broadcast.nil.consSame.consSame (Op.ref .real [BLK, DM] "acc")
        (Op.dot (batch := []) (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v"))) s
      = some (⟨fun idx : TileIndex [BLK, DM] =>
          WithBot.realAdd (accFn idx.1 idx.2.1)
            (some (Finset.univ.sum (fun jL : Fin BLK => pFn idx.1 jL * vFn jL idx.2.1)))⟩
          : Tile .real [BLK, DM]) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v")) s
      = some (Tile.dot [] (⟨fun idx : TileIndex [BLK, BLK] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK])
          (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM])) := by
    rw [evalOp_dot]; erw [evalOp_ref, hp, evalOp_ref, hv]; rfl
  rw [evalOp_add, evalOp_ref, hacc]
  show Option.bind (evalOp (Op.dot (batch := []) (Op.ref .real [BLK, BLK] "p") (Op.ref .real [BLK, DM] "v")) s) _ = _
  rw [hdot]
  simp only [Option.bind]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, d, u⟩ := idx
  simp only [Tile.bop_data, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin BLK) (WithBot ℝ) _ Finset.univ fun jL : Fin BLK =>
        Option.map₂ (· * ·) ((⟨fun idx : TileIndex [BLK, BLK] => some (pFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]).data (i, jL, PUnit.unit))
          ((⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]).data (jL, d, PUnit.unit)))
      = (@Finset.sum (Fin BLK) (WithBot ℝ) _ Finset.univ fun jL : Fin BLK => ((pFn i jL * vFn jL d : ℝ) : WithBot ℝ)) from by
    apply Finset.sum_congr rfl; intro jL _; rfl]
  rw [WithBot.sum_some_eq_some]; rfl

/-- General output offset (contiguous layout, head stride `hs`). -/
def mistralOutOffsetG
    (s : BlockState) (B_Start_Loc : RegionName)
    (rs hs BLK DM : Nat) (idx : TileIndex [BLK, DM]) : Nat :=
  (startLoc s B_Start_Loc + (s.pids 2 * BLK + idx.1.val)) * rs
    + s.pids 1 * hs + idx.2.1.val

/-- **General streaming-loop invariant.** After streaming `c` `BLOCK_N=BLK` blocks,
the `m_i`/`l_i`/`acc` registers hold `mistralFoldUptoG … c`; pointers, masks, and the
loaded `q` tile are preserved; state is otherwise clean. -/
noncomputable def mistralInvariantG
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (hDM : 0 < DM) (sw : Nat)
    (s0 : BlockState) (c : Nat) (s : BlockState) : Prop :=
  s.pids = s0.pids
  ∧ s.mem = s0.mem
  ∧ (∀ rg o, s.undef rg o = 0)
  ∧ s.regs .nat [] "cur_batch" = some (Tile.scalar (s0.pids 0))
  ∧ s.regs .nat [] "cur_head" = some (Tile.scalar (s0.pids 1))
  ∧ s.regs .nat [] "start_m" = some (Tile.scalar (s0.pids 2))
  ∧ s.regs .nat [] "cur_kv_head" = some (Tile.scalar (s0.pids 1 / 1))
  ∧ s.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s0 B_Seqlen))
  ∧ s.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s0 B_Start_Loc))
  ∧ s.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val))
  ∧ s.regs .nat [DM] "offs_d" = some (Tile.vec (fun e : Fin DM => e.val))
  ∧ s.regs .nat [BLK] "offs_m" = some (Tile.vec (fun i : Fin BLK => s0.pids 2 * BLK + i.val))
  ∧ s.regs .real [BLK, DM] "q" =
      some (⟨fun idx : TileIndex [BLK, DM] =>
        some (ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM (seqLen s0 B_Seqlen) (idx.1, idx.2.1, PUnit.unit))⟩
        : Tile .real [BLK, DM])
  ∧ s.regs .ptr [DM, BLK] "k_ptrs" =
      some (⟨fun idx : TileIndex [DM, BLK] =>
        (K, idx.2.1.val * rs + s0.pids 1 * hs + idx.1.val)⟩ : Tile .ptr [DM, BLK])
  ∧ s.regs .ptr [BLK, DM] "v_ptrs" =
      some (⟨fun idx : TileIndex [BLK, DM] =>
        (V, idx.1.val * rs + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM])
  ∧ s.regs .nat [] "block_mask" =
      some (Tile.scalar (if BLK * s0.pids 2 < seqLen s0 B_Seqlen then 1 else 0))
  ∧ s.regs .real [BLK] "m_i" =
      some (⟨fun idx : TileIndex [BLK] =>
        (mistralFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM
          (ctxMistralWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) sw c idx.1 ⟨0, hDM⟩).1⟩
        : Tile .real [BLK])
  ∧ s.regs .real [BLK] "l_i" =
      some (⟨fun idx : TileIndex [BLK] =>
        ((mistralFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM
          (ctxMistralWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) sw c idx.1 ⟨0, hDM⟩).2.1 : WithBot ℝ)⟩
        : Tile .real [BLK])
  ∧ s.regs .real [BLK, DM] "acc" =
      some (⟨fun idx : TileIndex [BLK, DM] =>
        ((mistralFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM
          (ctxMistralWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) sw c idx.1 idx.2.1).2.2 : WithBot ℝ)⟩
        : Tile .real [BLK, DM])

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General preLoop execution.** -/
theorem mistralPreLoop_evalG
    (s : BlockState) (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (hBLK : 0 < BLK) (hDM : 0 < DM) (sw : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (mistralPreLoopG Q K V B_Start_Loc B_Seqlen rs hs BLK DM) s = some s0
      ∧ mistralInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM sw s 0 s0 := by
  unfold mistralPreLoopG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 1)) _
        = some (Tile.scalar (s.pids 1 / 1)) from by
      rw [mistral_evalOp_floorDiv]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_ne_name,
        BlockState.setReg_pids, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (seqLen s B_Seqlen)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def, seqLen, BlockState.readMemValue]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none) _
        = some (Tile.scalar (startLoc s B_Start_Loc)) from by
      simp only [evalOp_load_region_none, evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, Option.bind, Option.pure_def, startLoc, BlockState.readMemValue]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.constNat BLK) (Op.ref .nat [] "start_m")) _
        = some (Tile.scalar (BLK * s.pids 2)) from by
      rw [evalOp_mul, evalOp_constNat, evalOp_ref]
      simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLK) _ = some (Tile.vec (fun j : Fin BLK => j.val)) from evalOp_arange BLK _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange DM) _ = some (Tile.vec (fun e : Fin DM => e.val)) from evalOp_arange DM _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLK))
          (Op.arange BLK)) _
        = some (Tile.vec (fun i : Fin BLK => s.pids 2 * BLK + i.val)) from by
      rw [evalOp_add, evalOp_mul, evalOp_ref, evalOp_constNat, evalOp_arange]
      simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Tile.vec, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some (⟨fun idx : TileIndex [BLK, DM] =>
            (startLoc s B_Start_Loc + (s.pids 2 * BLK + idx.1.val)) * rs
              + s.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) from
      mistral_offq_evalG _ (startLoc s B_Start_Loc) (s.pids 1) (s.pids 2) rs hs BLK DM
        (by simp) (by simp) (by simp) (by simp [Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some (⟨fun idx : TileIndex [DM, BLK] =>
            idx.2.1.val * rs + s.pids 1 / 1 * hs + idx.1.val⟩ : Tile .nat [DM, BLK]) from
      mistral_offk_evalG _ (s.pids 1 / 1) rs hs BLK DM
        (by simp) (by simp [Tile.vec]) (by simp [Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some (⟨fun idx : TileIndex [BLK, DM] =>
            idx.1.val * rs + s.pids 1 / 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) from
      mistral_offv_evalG _ (s.pids 1 / 1) rs hs BLK DM
        (by simp) (by simp [Tile.vec]) (by simp [Tile.vec])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_q_load_evalG _ Q B_Start_Loc B_Seqlen rs hs BLK DM
      (by simp [startLoc, BlockState.readMemValue, BlockState.readMemTyped])
      (by simp)
      (by simp [seqLen, BlockState.readMemValue, BlockState.readMemTyped])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K) (Op.ref .nat [DM, BLK] "off_k")) _
        = some (⟨fun idx : TileIndex [DM, BLK] =>
            (K, idx.2.1.val * rs + s.pids 1 * hs + idx.1.val)⟩ : Tile .ptr [DM, BLK]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
        String.reduceEq, not_false_eq_true, BlockState.setReg_pids, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨e, jL, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and, Nat.div_one]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [BLK, DM] "off_v")) _
        = some (⟨fun idx : TileIndex [BLK, DM] =>
            (V, idx.1.val * rs + s.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) from by
      simp only [evalOp, evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
        String.reduceEq, not_false_eq_true, BlockState.setReg_pids, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨jL, d, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and, Nat.div_one]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BLK] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [BLK] => (⊥ : WithBot ℝ)⟩ : Tile .real [BLK]) from by
      rw [evalOp_add, evalOp_full, evalOp_const, evalOp_negInf]
      simp only [Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLK] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLK] => some (0 : ℝ)⟩ : Tile .real [BLK]) from by
      simp [evalOp_full, evalOp_const]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLK, DM] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLK, DM] => some (0 : ℝ)⟩ : Tile .real [BLK, DM]) from by
      simp [evalOp_full, evalOp_const]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp ((Op.lt ComparableDType.nat Broadcast.nil (Op.ref .nat [] "block_start_loc")
            (Op.ref .nat [] "cur_batch_seq_len")).where (Op.constNat 1) (Op.constNat 0)) _
        = some (Tile.scalar (if BLK * s.pids 2 < seqLen s B_Seqlen then 1 else 0)) from by
      rw [evalOp_where, evalOp_lt, evalOp_constNat, evalOp_constNat]
      simp only [evalOp_ref, BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
        String.reduceEq, not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.select_data, Tile.cop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, ComparableDType.lt]
      by_cases hlt : BLK * s.pids 2 < seqLen s B_Seqlen
      · rw [if_pos (by simpa [seqLen, BlockState.readMemValue] using hlt), if_pos hlt]
      · rw [if_neg (by simpa [seqLen, BlockState.readMemValue] using hlt), if_neg hlt]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  refine ⟨by simp, ?_, ?_, by simp, by simp, by simp, by simp, by simp, by simp, by simp [Tile.vec],
    by simp [Tile.vec], by simp [Tile.vec], ?_, by simp, by simp, by simp, ?_, ?_, ?_⟩
  · funext rg o; simp
  · intro rg o; simp [hundef]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq,
      String.reduceEq, not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [ctxQTileMRowG, ctxQTileG, seqLen, startLoc, BlockState.readMem, BlockState.readMemValue,
      BlockState.readMemTyped, BlockState.setReg_mem, BlockState.setReg_pids]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
      not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [mistralFoldUptoG_zero]
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
      not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [mistralFoldUptoG_zero]; rfl
  · simp only [BlockState.setReg_same, BlockState.setReg_ne_name, BlockState.setReg_pids, ne_eq, String.reduceEq,
      not_false_eq_true]
    refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [mistralFoldUptoG_zero]; rfl

/-- General `c = 0` invariant re-anchor to `s0`. -/
theorem mistralInvariantG_zero_reanchor
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName) (sm_scale : ℝ) (rs hs BLK DM : Nat)
    (hDM : 0 < DM) (sw : Nat) (s s0 : BlockState)
    (h : mistralInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM sw s 0 s0) :
    mistralInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM sw s0 0 s0 := by
  obtain ⟨hpids, hmem, hundef, hcb, hch, hsm, hckv, hseq, hsl, hn, hd, hoffm, hq, hkp, hvp, hbmask, hmi, hli, hacc⟩ := h
  have hseqEq : seqLen s0 B_Seqlen = seqLen s B_Seqlen := mistral_seqLen_eq_of_mem_pids s0 s B_Seqlen hmem hpids
  have hslEq : startLoc s0 B_Start_Loc = startLoc s B_Start_Loc := mistral_startLoc_eq_of_mem_pids s0 s B_Start_Loc hmem hpids
  refine ⟨rfl, rfl, hundef, ?_, ?_, ?_, ?_, ?_, ?_, hn, hd, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hcb, hpids]
  · rw [hch, hpids]
  · rw [hsm, hpids]
  · rw [hckv, hpids]
  · rw [hseq, hseqEq]
  · rw [hsl, hslEq]
  · rw [hoffm, hpids]
  · rw [hq]; refine congrArg some (Tile.ext (fun idx => ?_))
    show some (ctxQTileMRowG s Q B_Start_Loc rs hs BLK DM (seqLen s B_Seqlen) (idx.1, idx.2.1, PUnit.unit))
      = some (ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM (seqLen s0 B_Seqlen) (idx.1, idx.2.1, PUnit.unit))
    simp only [ctxQTileMRowG, ctxQTileG, startLoc, seqLen, BlockState.readMem, BlockState.readMemValue,
      BlockState.readMemTyped, hmem, hpids]
  · rw [hkp, hpids]
  · rw [hvp, hpids]
  · rw [hbmask, hpids, hseqEq]
  · rw [hmi]; refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [mistralFoldUptoG_zero]
  · rw [hli]; refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [mistralFoldUptoG_zero]
  · rw [hacc]; refine congrArg some (Tile.ext (fun idx => ?_))
    simp only [mistralFoldUptoG_zero]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General one loop-body step advances the invariant by one block.** -/
theorem mistral_attn_stepG
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM : Nat) (hBLK : 0 < BLK) (hDM : 0 < DM) (sw : Nat)
    (s0 : BlockState) (i : Nat) (s : BlockState)
    (hilt : i < ctxMistralWindowG s0 B_Seqlen BLK)
    (hinv : mistralInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM sw s0 (i / BLK) s)
    (hi : i = (i / BLK) * BLK) :
    ∃ s', stepStmts (mistralLoopBodyG sm_scale sw rs hs BLK DM) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ mistralInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM sw s0 (i / BLK + 1) s' := by
  set S := ctxMistralWindowG s0 B_Seqlen BLK with hSdef
  set bel := seqLen s0 B_Seqlen with hbeldef
  set c := i / BLK with hc_def
  set sc := sm_scale with hscdef
  have hwin : (c + 1) * BLK ≤ S := by
    have hSmul : S = (if BLK * s0.pids 2 < bel then 1 else 0) * (s0.pids 2 + 1) * BLK := by
      simp only [hSdef, ctxMistralWindowG, hbeldef, seqLen]; rfl
    by_cases hbm : BLK * s0.pids 2 < bel
    · rw [hSmul, if_pos hbm, one_mul] at hilt ⊢
      have : c < s0.pids 2 + 1 := by
        by_contra hcon; push_neg at hcon
        have : (s0.pids 2 + 1) * BLK ≤ c * BLK := Nat.mul_le_mul_right BLK hcon
        omega
      have : c + 1 ≤ s0.pids 2 + 1 := by omega
      exact Nat.mul_le_mul_right BLK this
    · rw [hSmul, if_neg hbm] at hilt; omega
  obtain ⟨hpids, hmem, hundef, hcb, hch, hsm, hckv, hseq, hsl, hn, hd, hoffm, hq, hkp, hvp, hbmask, hmi, hli, hacc⟩ := hinv
  set fold0 : Fin BLK → Fin DM → WithBot ℝ × ℝ × ℝ := fun ir dd =>
    mistralFoldUptoG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd with hfold0
  set blk : Fin BLK → Fin DM → List (ℝ × ℝ) := fun ir dd =>
    mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd with hblk
  set scoreFn : Fin BLK → Fin BLK → ℝ := fun ir jL =>
    mistralScoreM s0 Q K B_Start_Loc sc rs hs BLK DM S bel sw ir
      ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩ with hscoreFn
  unfold mistralLoopBodyG
  set s1 := s.setReg "start_n" .nat [] (Tile.scalar i) with hs1d
  have hs1pids : s1.pids = s0.pids := by rw [hs1d, BlockState.setReg_pids]; exact hpids
  have hs1mem : s1.mem = s0.mem := by funext rg o; rw [hs1d, BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1sn : s1.regs .nat [] "start_n" = some (Tile.scalar i) := by rw [hs1d, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") s1 = some (Tile.scalar i) from by rw [evalOp_ref]; exact hs1sn))]
  set s2 := s1.setReg "start_n" .nat [] (Tile.scalar i) with hs2d
  have hs2pids : s2.pids = s0.pids := by rw [hs2d, BlockState.setReg_pids]; exact hs1pids
  have hs2mem : s2.mem = s0.mem := by funext rg o; rw [hs2d, BlockState.setReg_mem]; exact hs1mem ▸ rfl
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2sn : s2.regs .nat [] "start_n" = some (Tile.scalar i) := by rw [hs2d, BlockState.setReg_same]
  have hs2seq : s2.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar bel) := e2 (by decide) (e1 (by decide) hseq)
  have hs2slStart : startLoc s2 B_Start_Loc = startLoc s0 B_Start_Loc := mistral_startLoc_eq_of_mem_pids s2 s0 B_Start_Loc hs2mem hs2pids
  have hs2sl : s2.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s2 B_Start_Loc)) := by
    rw [hs2slStart]; exact e2 (by decide) (e1 (by decide) hsl)
  have hs2n : s2.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)) := e2 (by decide) (e1 (by decide) hn)
  have hs2m : s2.regs .nat [BLK] "offs_m" = some (Tile.vec (fun ir : Fin BLK => s0.pids 2 * BLK + ir.val)) := e2 (by decide) (e1 (by decide) hoffm)
  have hs2kp : s2.regs .ptr [DM, BLK] "k_ptrs" = some (⟨fun idx : TileIndex [DM, BLK] =>
      (K, idx.2.1.val * rs + s2.pids 1 * hs + idx.1.val)⟩ : Tile .ptr [DM, BLK]) := by
    rw [hs2pids]; exact e2 (by decide) (e1 (by decide) hkp)
  have hs2seqB : seqLen s2 B_Seqlen = bel := by rw [hbeldef]; exact mistral_seqLen_eq_of_mem_pids s2 s0 B_Seqlen hs2mem hs2pids
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ s2 = _ from by
      have h := mistral_k_load_evalG s2 K B_Start_Loc B_Seqlen i rs hs BLK DM hs2kp hs2sl hs2sn hs2n (by rw [hs2seqB]; exact hs2seq)
      rw [hs2seqB] at h; exact h))]
  set kFn : Fin DM → Fin BLK → ℝ := fun e jL =>
    ctxKTileMG s0 K B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩, e, PUnit.unit) with hkFn
  rw [show (⟨fun idx : TileIndex [DM, BLK] =>
        if i + idx.2.1.val < bel then
          some (s2.readMem K ((startLoc s2 B_Start_Loc + (i + idx.2.1.val)) * rs + s2.pids 1 * hs + idx.1.val))
        else some (0.0 : ℝ)⟩ : Tile .real [DM, BLK])
      = (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK]) from by
    refine Tile.ext (fun idx => ?_)
    obtain ⟨e, jL, u⟩ := idx
    simp only [hkFn, ctxKTileMG, ctxKTileG, hs2pids,
      show startLoc s2 B_Start_Loc = startLoc s0 B_Start_Loc from mistral_startLoc_eq_of_mem_pids s2 s0 B_Start_Loc hs2mem hs2pids,
      show i + jL.val = c * BLK + jL.val from by rw [hi]]
    by_cases hlt : c * BLK + jL.val < bel
    · rw [if_pos hlt, if_pos hlt]
      simp only [BlockState.readMem, hs2mem]
    · rw [if_neg hlt, if_neg hlt]; norm_num]
  set s3 := s2.setReg "k" .real [DM, BLK] (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK]) with hs3d
  have e3 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "k" → s2.regs dt sh nm = some t → s3.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs3d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs3pids : s3.pids = s0.pids := by rw [hs3d, BlockState.setReg_pids]; exact hs2pids
  have hs3mem : s3.mem = s0.mem := by funext rg o; rw [hs3d, BlockState.setReg_mem]; exact hs2mem ▸ rfl
  have hs3k : s3.regs .real [DM, BLK] "k" = some (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK]) := by
    rw [hs3d, BlockState.setReg_same]
  set qFn : Fin BLK → Fin DM → ℝ := fun ir e => ctxQTileMRowG s0 Q B_Start_Loc rs hs BLK DM bel (ir, e, PUnit.unit) with hqFn
  have hs3q : s3.regs .real [BLK, DM] "q" = some (⟨fun idx : TileIndex [BLK, DM] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) :=
    e3 (by decide) (e2 (by decide) (e1 (by decide) hq))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLK, BLK] (Op.const 0)) s3
        = some (⟨fun _ : TileIndex [BLK, BLK] => some (0 : ℝ)⟩ : Tile .real [BLK, BLK]) from by
      simp [evalOp_full, evalOp_const]))]
  set s4 := s3.setReg "qk" .real [BLK, BLK] (⟨fun _ : TileIndex [BLK, BLK] => some (0 : ℝ)⟩ : Tile .real [BLK, BLK]) with hs4d
  have e4 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s3.regs dt sh nm = some t → s4.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs4d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs4qk0 : s4.regs .real [BLK, BLK] "qk" = some (⟨fun _ : TileIndex [BLK, BLK] => some (0 : ℝ)⟩ : Tile .real [BLK, BLK]) := by
    rw [hs4d, BlockState.setReg_same]
  have hs4q : s4.regs .real [BLK, DM] "q" = some (⟨fun idx : TileIndex [BLK, DM] => some (qFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) := e4 (by decide) hs3q
  have hs4k : s4.regs .real [DM, BLK] "k" = some (⟨fun idx : TileIndex [DM, BLK] => some (kFn idx.1 idx.2.1)⟩ : Tile .real [DM, BLK]) := e4 (by decide) hs3k
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_qk_dot_evalG s4 BLK DM qFn kFn hs4qk0 hs4q hs4k))]
  set rawSum : Fin BLK → Fin BLK → ℝ := fun ir jL =>
    Finset.univ.sum (fun e : Fin DM => qFn ir e * kFn e jL) with hrawSum
  rw [show (⟨fun idx : TileIndex [BLK, BLK] =>
        some (Finset.univ.sum (fun e : Fin DM => qFn idx.1 e * kFn e idx.2.1))⟩ : Tile .real [BLK, BLK])
      = (⟨fun idx : TileIndex [BLK, BLK] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx; simp only [hrawSum]]
  set s5 := s4.setReg "qk" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]) with hs5d
  have e5 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s4.regs dt sh nm = some t → s5.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs5d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs5qk : s5.regs .real [BLK, BLK] "qk" = some (⟨fun idx : TileIndex [BLK, BLK] => some (rawSum idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]) := by
    rw [hs5d, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_qk_scale_evalG s5 BLK sc rawSum hs5qk))]
  set s6 := s5.setReg "qk" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => some (rawSum idx.1 idx.2.1 * sc)⟩ : Tile .real [BLK, BLK]) with hs6d
  have e6 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s5.regs dt sh nm = some t → s6.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs6d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs6qk : s6.regs .real [BLK, BLK] "qk" = some (⟨fun idx : TileIndex [BLK, BLK] => some (rawSum idx.1 idx.2.1 * sc)⟩ : Tile .real [BLK, BLK]) := by
    rw [hs6d, BlockState.setReg_same]
  have hs6m : s6.regs .nat [BLK] "offs_m" = some (Tile.vec (fun ir : Fin BLK => s0.pids 2 * BLK + ir.val)) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2m)))
  have hs6sn : s6.regs .nat [] "start_n" = some (Tile.scalar i) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2sn)))
  have hs6n : s6.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)) :=
    e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2n)))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_qk_where1_evalG s6 i BLK (fun ir jL => rawSum ir jL * sc) (fun ir => s0.pids 2 * BLK + ir.val)
      hs6qk hs6m hs6sn hs6n))]
  set g1 : Fin BLK → Fin BLK → ℝ := fun ir jL =>
    if i + jL.val ≤ s0.pids 2 * BLK + ir.val then rawSum ir jL * sc else (-1e9 : ℝ) with hg1
  rw [show (⟨fun idx : TileIndex [BLK, BLK] =>
        if i + idx.2.1.val ≤ s0.pids 2 * BLK + idx.1.val then some (rawSum idx.1 idx.2.1 * sc)
        else some (-1e9 : ℝ)⟩ : Tile .real [BLK, BLK])
      = (⟨fun idx : TileIndex [BLK, BLK] => some (g1 idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx
    simp only [hg1]
    by_cases hca : i + jL.val ≤ s0.pids 2 * BLK + ir.val
    · rw [if_pos hca, if_pos hca]
    · rw [if_neg hca, if_neg hca]]
  set s7 := s6.setReg "qk" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => some (g1 idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]) with hs7d
  have e7 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s6.regs dt sh nm = some t → s7.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs7d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs7qk : s7.regs .real [BLK, BLK] "qk" = some (⟨fun idx : TileIndex [BLK, BLK] => some (g1 idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]) := by
    rw [hs7d, BlockState.setReg_same]
  have hs7m : s7.regs .nat [BLK] "offs_m" = some (Tile.vec (fun ir : Fin BLK => s0.pids 2 * BLK + ir.val)) := e7 (by decide) hs6m
  have hs7sn : s7.regs .nat [] "start_n" = some (Tile.scalar i) := e7 (by decide) hs6sn
  have hs7n : s7.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)) := e7 (by decide) hs6n
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_qk_where2_evalG s7 i sw BLK g1 (fun ir => s0.pids 2 * BLK + ir.val)
      hs7qk hs7m hs7sn hs7n))]
  rw [show (⟨fun idx : TileIndex [BLK, BLK] =>
        if s0.pids 2 * BLK + idx.1.val - sw < i + idx.2.1.val then some (g1 idx.1 idx.2.1)
        else some (-1e9 : ℝ)⟩ : Tile .real [BLK, BLK])
      = (⟨fun idx : TileIndex [BLK, BLK] => some (scoreFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx
    show (if s0.pids 2 * BLK + ir.val - sw < i + jL.val then some (g1 ir jL) else some (-1e9 : ℝ))
      = some (scoreFn ir jL)
    simp only [hscoreFn, mistralScoreM, mistralActive, Fin.val_mk, hg1]
    rw [show i + jL.val = c * BLK + jL.val from by rw [hi]]
    by_cases hwd : s0.pids 2 * BLK + ir.val - sw < c * BLK + jL.val
    · rw [if_pos hwd]
      by_cases hca : c * BLK + jL.val ≤ s0.pids 2 * BLK + ir.val
      · rw [if_pos hca, if_pos ⟨hca, hwd⟩, hrawSum, hqFn, hkFn, mul_comm]
      · rw [if_neg hca, if_neg (fun h => hca h.1)]
    · rw [if_neg hwd, if_neg (fun h => hwd h.2)]]
  set qkW : Fin BLK → Fin BLK → WithBot ℝ := fun ir jL => ((scoreFn ir jL : ℝ) : WithBot ℝ) with hqkW
  rw [show (⟨fun idx : TileIndex [BLK, BLK] => some (scoreFn idx.1 idx.2.1)⟩ : Tile .real [BLK, BLK])
      = (⟨fun idx : TileIndex [BLK, BLK] => qkW idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) from rfl]
  set s8 := s7.setReg "qk" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => qkW idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) with hs8d
  have e8 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "qk" → s7.regs dt sh nm = some t → s8.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs8d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs8qk : s8.regs .real [BLK, BLK] "qk" = some (⟨fun idx : TileIndex [BLK, BLK] => qkW idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := by
    rw [hs8d, BlockState.setReg_same]
  set bsup : Fin BLK → WithBot ℝ := fun ir =>
    (blk ir ⟨0, hDM⟩).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) |>.foldr (· ⊔ ·) ⊥ with hbsup
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_mij_evalG s8 BLK hBLK qkW hs8qk))]
  rw [show (⟨fun idx : TileIndex [BLK] =>
        Finset.univ.sup' (⟨⟨0, hBLK⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLK)).Nonempty)
          (fun jL : Fin BLK => qkW idx.1 jL)⟩ : Tile .real [BLK])
      = (⟨fun idx : TileIndex [BLK] => bsup idx.1⟩ : Tile .real [BLK]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, u⟩ := idx
    show Finset.univ.sup' (⟨⟨0, hBLK⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin BLK)).Nonempty)
          (fun jL : Fin BLK => qkW ir jL) = bsup ir
    rw [Finset.sup'_eq_sup, hbsup]
    simp only [hblk, hqkW, hscoreFn]
    rw [← mistralBlockMG_sup_eq s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir ⟨0, hDM⟩ hwin]]
  set s9 := s8.setReg "m_ij" .real [BLK] (⟨fun idx : TileIndex [BLK] => bsup idx.1⟩ : Tile .real [BLK]) with hs9d
  have e9 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_ij" → s8.regs dt sh nm = some t → s9.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs9d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs9mij0 : s9.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => bsup idx.1⟩ : Tile .real [BLK]) := by
    rw [hs9d, BlockState.setReg_same]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_mij_guard_evalG s9 BLK bsup hs9mij0))]
  set mijFn : Fin BLK → WithBot ℝ := fun ir => mistralGuard (bsup ir) with hmijFn
  set s10 := s9.setReg "m_ij" .real [BLK] (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) with hs10d
  have e10 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_ij" → s9.regs dt sh nm = some t → s10.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs10d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs10mij : s10.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs10d, BlockState.setReg_same]
  have hs10qk : s10.regs .real [BLK, BLK] "qk" = some (⟨fun idx : TileIndex [BLK, BLK] => qkW idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) :=
    e10 (by decide) (e9 (by decide) hs8qk)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_p_evalG s10 BLK qkW mijFn hs10qk hs10mij))]
  set pFn : Fin BLK → Fin BLK → WithBot ℝ := fun ir jL => WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir)) with hpFn
  set s11 := s10.setReg "p" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => pFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) with hs11d
  have e11 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s10.regs dt sh nm = some t → s11.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs11d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs11p : s11.regs .real [BLK, BLK] "p" = some (⟨fun idx : TileIndex [BLK, BLK] => pFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := by
    rw [hs11d, BlockState.setReg_same]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_lij_evalG s11 BLK pFn hs11p))]
  set lijFn : Fin BLK → WithBot ℝ := fun ir => Finset.univ.sum (fun jL : Fin BLK => pFn ir jL) with hlijFn
  set s12 := s11.setReg "l_ij" .real [BLK] (⟨fun idx : TileIndex [BLK] => lijFn idx.1⟩ : Tile .real [BLK]) with hs12d
  have e12 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_ij" → s11.regs dt sh nm = some t → s12.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs12d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs12lij : s12.regs .real [BLK] "l_ij" = some (⟨fun idx : TileIndex [BLK] => lijFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs12d, BlockState.setReg_same]
  set miFn : Fin BLK → WithBot ℝ := fun ir => (fold0 ir ⟨0, hDM⟩).1 with hmiFn
  have hs12mi : s12.regs .real [BLK] "m_i" = some (⟨fun idx : TileIndex [BLK] => miFn idx.1⟩ : Tile .real [BLK]) :=
    e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hmi)))))))))))
  have hs12mij : s12.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) := e12 (by decide) (e11 (by decide) hs10mij)
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_minew_evalG s12 BLK miFn mijFn hs12mi hs12mij))]
  set minewFn : Fin BLK → WithBot ℝ := fun ir => miFn ir ⊔ mijFn ir with hminewFn
  set s13 := s12.setReg "m_i_new" .real [BLK] (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) with hs13d
  have e13 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_i_new" → s12.regs dt sh nm = some t → s13.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs13d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs13minew : s13.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs13d, BlockState.setReg_same]
  have hs13mi : s13.regs .real [BLK] "m_i" = some (⟨fun idx : TileIndex [BLK] => miFn idx.1⟩ : Tile .real [BLK]) := e13 (by decide) hs12mi
  have hs13mij : s13.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) := e13 (by decide) hs12mij
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_alpha_evalG s13 BLK miFn minewFn hs13mi hs13minew))]
  set alphaFn : Fin BLK → WithBot ℝ := fun ir => WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir)) with halphaFn
  set s14 := s13.setReg "alpha" .real [BLK] (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK]) with hs14d
  have e14 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "alpha" → s13.regs dt sh nm = some t → s14.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs14d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs14alpha : s14.regs .real [BLK] "alpha" = some (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs14d, BlockState.setReg_same]
  have hs14mij : s14.regs .real [BLK] "m_ij" = some (⟨fun idx : TileIndex [BLK] => mijFn idx.1⟩ : Tile .real [BLK]) := e14 (by decide) hs13mij
  have hs14minew : s14.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) := e14 (by decide) hs13minew
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_beta_evalG s14 BLK mijFn minewFn hs14mij hs14minew))]
  set betaFn : Fin BLK → WithBot ℝ := fun ir => WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir)) with hbetaFn
  set s15 := s14.setReg "beta" .real [BLK] (⟨fun idx : TileIndex [BLK] => betaFn idx.1⟩ : Tile .real [BLK]) with hs15d
  have e15 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "beta" → s14.regs dt sh nm = some t → s15.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs15d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs15beta : s15.regs .real [BLK] "beta" = some (⟨fun idx : TileIndex [BLK] => betaFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs15d, BlockState.setReg_same]
  have hs15alpha : s15.regs .real [BLK] "alpha" = some (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK]) := e15 (by decide) hs14alpha
  set liFn : Fin BLK → WithBot ℝ := fun ir => ((fold0 ir ⟨0, hDM⟩).2.1 : WithBot ℝ) with hliFn
  have hs15li : s15.regs .real [BLK] "l_i" = some (⟨fun idx : TileIndex [BLK] => liFn idx.1⟩ : Tile .real [BLK]) :=
    e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hli))))))))))))))
  have hs15lij : s15.regs .real [BLK] "l_ij" = some (⟨fun idx : TileIndex [BLK] => lijFn idx.1⟩ : Tile .real [BLK]) :=
    e15 (by decide) (e14 (by decide) (e13 (by decide) hs12lij))
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_linew_evalG s15 BLK alphaFn liFn betaFn lijFn hs15alpha hs15li hs15beta hs15lij))]
  set linewFn : Fin BLK → WithBot ℝ := fun ir =>
    WithBot.realAdd (WithBot.realMul (alphaFn ir) (liFn ir)) (WithBot.realMul (betaFn ir) (lijFn ir)) with hlinewFn
  set s16 := s15.setReg "l_i_new" .real [BLK] (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) with hs16d
  have e16 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i_new" → s15.regs dt sh nm = some t → s16.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs16d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs16linew0 : s16.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs16d, BlockState.setReg_same]
  set pReal : Fin BLK → Fin BLK → ℝ := fun ir jL =>
    (WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0 with hpReal
  have hpFnSome : ∀ ir jL : Fin BLK, pFn ir jL = some (pReal ir jL) := by
    intro ir jL
    show WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))
      = some ((WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0)
    exact nopad_realExp_eq_some_unbotD _
  set lijReal : Fin BLK → ℝ := fun ir => Finset.univ.sum (fun jL : Fin BLK => pReal ir jL) with hlijReal
  have hlijSome : ∀ ir : Fin BLK, lijFn ir = some (lijReal ir) := by
    intro ir
    show Finset.univ.sum (fun jL : Fin BLK => pFn ir jL) = some (Finset.univ.sum (fun jL : Fin BLK => pReal ir jL))
    rw [show (fun jL : Fin BLK => pFn ir jL) = (fun jL : Fin BLK => (some (pReal ir jL) : WithBot ℝ)) from by
      funext jL; exact hpFnSome ir jL]
    rw [WithBot.sum_someTerm_eq_some]
  set αReal : Fin BLK → ℝ := fun ir => (WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir))).unbotD 0 with hαReal
  set βReal : Fin BLK → ℝ := fun ir => (WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir))).unbotD 0 with hβReal
  have hαSome : ∀ ir, alphaFn ir = some (αReal ir) := fun ir => nopad_realExp_eq_some_unbotD _
  have hβSome : ∀ ir, betaFn ir = some (βReal ir) := fun ir => nopad_realExp_eq_some_unbotD _
  have hliSome : ∀ ir, liFn ir = some ((fold0 ir ⟨0, hDM⟩).2.1) := by intro ir; rfl
  set linewReal : Fin BLK → ℝ := fun ir => αReal ir * (fold0 ir ⟨0, hDM⟩).2.1 + βReal ir * lijReal ir with hlinewReal
  have hlinewSome : ∀ ir, linewFn ir = some (linewReal ir) := by
    intro ir
    show WithBot.realAdd (WithBot.realMul (alphaFn ir) (liFn ir)) (WithBot.realMul (betaFn ir) (lijFn ir))
      = some (linewReal ir)
    rw [hαSome, hliSome, hβSome, hlijSome]; rfl
  have hbsupNe : ∀ ir, bsup ir ≠ ⊥ := by
    intro ir
    have hbne : blk ir ⟨0, hDM⟩ ≠ [] := by
      simp only [hblk]; exact mistralBlockMG_ne_nil s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir ⟨0, hDM⟩ hwin hBLK
    simp only [hbsup]; exact fun h => hbne ((foldr_sup_coe_bot_iff _).mp h)
  have hlinewPos : ∀ ir, 0 < linewReal ir := by
    intro ir
    have hGne : mijFn ir ≠ ⊥ := by simp only [hmijFn]; exact mistralGuard_ne_bot (hbsupNe ir)
    have hminewNe : minewFn ir ≠ ⊥ := by
      simp only [hminewFn]; intro h; exact hGne (le_bot_iff.mp (h ▸ (le_sup_right : mijFn ir ≤ miFn ir ⊔ mijFn ir)))
    obtain ⟨mr, hmr⟩ : ∃ r : ℝ, minewFn ir = (r : WithBot ℝ) := by
      cases hm : minewFn ir with
      | bot => exact absurd hm hminewNe
      | coe r => exact ⟨r, rfl⟩
    obtain ⟨Gr, hGr⟩ : ∃ r : ℝ, mijFn ir = (r : WithBot ℝ) := by
      cases hm : mijFn ir with
      | bot => exact absurd hm hGne
      | coe r => exact ⟨r, rfl⟩
    have hαnn : 0 ≤ αReal ir := by
      simp only [hαReal]
      cases WithBot.realSub (miFn ir) (minewFn ir) with
      | bot => rw [WithBot.realExp_bot, WithBot.unbotD_coe]
      | coe r => simp only [WithBot.realExp_coe, WithBot.unbotD_coe]; exact le_of_lt (Real.exp_pos _)
    obtain ⟨_ha_l, _ha_acc, ha_L0, _ha_botL, _ha_botT, _ha_ne⟩ :=
      mistralFoldUptoG_anchor s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir ⟨0, hDM⟩ hBLK (le_of_lt (by nlinarith [hwin] : c * BLK < S))
    have hl0 : 0 ≤ (fold0 ir ⟨0, hDM⟩).2.1 := by
      simp only [hfold0] at _ha_l ⊢
      rw [_ha_l]
      apply mul_nonneg ?_ ha_L0
      cases (mistralFoldUptoG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir ⟨0, hDM⟩).1 with
      | bot => exact le_of_eq rfl
      | coe r => exact le_of_lt (Real.exp_pos _)
    have hαl_nn : 0 ≤ αReal ir * (fold0 ir ⟨0, hDM⟩).2.1 := mul_nonneg hαnn hl0
    have hβpos : 0 < βReal ir := by
      simp only [hβReal, hGr, hmr, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
      exact Real.exp_pos _
    have hlijpos : 0 < lijReal ir := by
      simp only [hlijReal]
      have hpr : ∀ jL : Fin BLK, 0 < pReal ir jL := by
        intro jL
        show 0 < (WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0
        rw [hqkW, hGr, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe]
        exact Real.exp_pos _
      exact Finset.sum_pos (fun jL _ => hpr jL) ⟨⟨0, hBLK⟩, Finset.mem_univ _⟩
    have hblpos : 0 < βReal ir * lijReal ir := mul_pos hβpos hlijpos
    simp only [hlinewReal]; linarith
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_linew_guard_evalG s16 BLK linewFn hs16linew0))]
  rw [show (⟨fun idx : TileIndex [BLK] =>
        if linewFn idx.1 = ((0 : ℝ) : WithBot ℝ) then ((1e-9 : ℝ) : WithBot ℝ) else linewFn idx.1⟩ : Tile .real [BLK])
      = (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, u⟩ := idx
    show (if linewFn ir = ((0 : ℝ) : WithBot ℝ) then ((1e-9 : ℝ) : WithBot ℝ) else linewFn ir) = linewFn ir
    rw [if_neg (by rw [hlinewSome ir]; intro h; have hp := hlinewPos ir
                   have h2 : linewReal ir = 0 := Option.some.inj h; linarith)]]
  set s17 := s16.setReg "l_i_new" .real [BLK] (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) with hs17d
  have e17 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i_new" → s16.regs dt sh nm = some t → s17.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs17d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs17linew : s17.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs17d, BlockState.setReg_same]
  have hs17beta : s17.regs .real [BLK] "beta" = some (⟨fun idx : TileIndex [BLK] => betaFn idx.1⟩ : Tile .real [BLK]) := e17 (by decide) hs15beta
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_pscale_evalG s17 BLK betaFn linewFn hs17beta hs17linew))]
  set psFn : Fin BLK → WithBot ℝ := fun ir => WithBot.realDiv (betaFn ir) (linewFn ir) with hpsFn
  set s18 := s17.setReg "p_scale" .real [BLK] (⟨fun idx : TileIndex [BLK] => psFn idx.1⟩ : Tile .real [BLK]) with hs18d
  have e18 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p_scale" → s17.regs dt sh nm = some t → s18.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs18d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs18ps : s18.regs .real [BLK] "p_scale" = some (⟨fun idx : TileIndex [BLK] => psFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs18d, BlockState.setReg_same]
  have hs18p : s18.regs .real [BLK, BLK] "p" = some (⟨fun idx : TileIndex [BLK, BLK] => pFn idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) :=
    e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) hs11p)))))
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_pmul_evalG s18 BLK pFn psFn hs18p hs18ps))]
  set pFinal : Fin BLK → Fin BLK → WithBot ℝ := fun ir jL => WithBot.realMul (pFn ir jL) (psFn ir) with hpFinal
  set s19 := s18.setReg "p" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) with hs19d
  have e19 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s18.regs dt sh nm = some t → s19.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs19d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs19p : s19.regs .real [BLK, BLK] "p" = some (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := by
    rw [hs19d, BlockState.setReg_same]
  have hs19li : s19.regs .real [BLK] "l_i" = some (⟨fun idx : TileIndex [BLK] => liFn idx.1⟩ : Tile .real [BLK]) :=
    e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) hs15li)))
  have hs19linew : s19.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) :=
    e19 (by decide) (e18 (by decide) hs17linew)
  have hs19alpha : s19.regs .real [BLK] "alpha" = some (⟨fun idx : TileIndex [BLK] => alphaFn idx.1⟩ : Tile .real [BLK]) :=
    e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) hs15alpha)))
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_accscale_evalG s19 BLK liFn linewFn alphaFn hs19li hs19linew hs19alpha))]
  set asFn : Fin BLK → WithBot ℝ := fun ir => WithBot.realMul (WithBot.realDiv (liFn ir) (linewFn ir)) (alphaFn ir) with hasFn
  set s20 := s19.setReg "acc_scale" .real [BLK] (⟨fun idx : TileIndex [BLK] => asFn idx.1⟩ : Tile .real [BLK]) with hs20d
  have e20 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc_scale" → s19.regs dt sh nm = some t → s20.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs20d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs20as : s20.regs .real [BLK] "acc_scale" = some (⟨fun idx : TileIndex [BLK] => asFn idx.1⟩ : Tile .real [BLK]) := by
    rw [hs20d, BlockState.setReg_same]
  set accFn : Fin BLK → Fin DM → WithBot ℝ := fun ir dd => ((fold0 ir dd).2.2 : WithBot ℝ) with haccFn
  have hs20acc : s20.regs .real [BLK, DM] "acc" = some (⟨fun idx : TileIndex [BLK, DM] => accFn idx.1 idx.2.1⟩ : Tile .real [BLK, DM]) :=
    e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hacc)))))))))))))))))
  erw [stepStmts.cons_some (stepStmt_assign_eq_some (mistral_accmul_evalG s20 BLK DM accFn asFn hs20acc hs20as))]
  set accMul : Fin BLK → Fin DM → WithBot ℝ := fun ir dd => WithBot.realMul (accFn ir dd) (asFn ir) with haccMul
  set s21 := s20.setReg "acc" .real [BLK, DM] (⟨fun idx : TileIndex [BLK, DM] => accMul idx.1 idx.2.1⟩ : Tile .real [BLK, DM]) with hs21d
  have e21 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s20.regs dt sh nm = some t → s21.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs21d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs21acc : s21.regs .real [BLK, DM] "acc" = some (⟨fun idx : TileIndex [BLK, DM] => accMul idx.1 idx.2.1⟩ : Tile .real [BLK, DM]) := by
    rw [hs21d, BlockState.setReg_same]
  have hs21mem : s21.mem = s0.mem := by
    funext rg o; rw [hs21d, BlockState.setReg_mem, hs20d, BlockState.setReg_mem, hs19d, BlockState.setReg_mem,
      hs18d, BlockState.setReg_mem, hs17d, BlockState.setReg_mem, hs16d, BlockState.setReg_mem,
      hs15d, BlockState.setReg_mem, hs14d, BlockState.setReg_mem, hs13d, BlockState.setReg_mem,
      hs12d, BlockState.setReg_mem, hs11d, BlockState.setReg_mem, hs10d, BlockState.setReg_mem,
      hs9d, BlockState.setReg_mem, hs8d, BlockState.setReg_mem, hs7d, BlockState.setReg_mem,
      hs6d, BlockState.setReg_mem, hs5d, BlockState.setReg_mem, hs4d, BlockState.setReg_mem]; exact hs3mem ▸ rfl
  have hs21pids : s21.pids = s0.pids := by
    rw [hs21d, BlockState.setReg_pids, hs20d, BlockState.setReg_pids, hs19d, BlockState.setReg_pids,
      hs18d, BlockState.setReg_pids, hs17d, BlockState.setReg_pids, hs16d, BlockState.setReg_pids,
      hs15d, BlockState.setReg_pids, hs14d, BlockState.setReg_pids, hs13d, BlockState.setReg_pids,
      hs12d, BlockState.setReg_pids, hs11d, BlockState.setReg_pids, hs10d, BlockState.setReg_pids,
      hs9d, BlockState.setReg_pids, hs8d, BlockState.setReg_pids, hs7d, BlockState.setReg_pids,
      hs6d, BlockState.setReg_pids, hs5d, BlockState.setReg_pids, hs4d, BlockState.setReg_pids, hs3pids]
  have hs21pids1 : s21.pids 1 = s0.pids 1 := by rw [hs21pids]
  have hs21seqB : seqLen s21 B_Seqlen = bel := by rw [hbeldef]; exact mistral_seqLen_eq_of_mem_pids s21 s0 B_Seqlen hs21mem hs21pids
  have hs21slB : startLoc s21 B_Start_Loc = startLoc s0 B_Start_Loc := mistral_startLoc_eq_of_mem_pids s21 s0 B_Start_Loc hs21mem hs21pids
  have hs21vp : s21.regs .ptr [BLK, DM] "v_ptrs" = some (⟨fun idx : TileIndex [BLK, DM] =>
      (V, idx.1.val * rs + s21.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) := by
    rw [hs21pids1]
    exact e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) (e2 (by decide) (e1 (by decide) hvp))))))))))))))))))))
  have hs21sl : s21.regs .nat [] "cur_batch_in_all_start_index" = some (Tile.scalar (startLoc s21 B_Start_Loc)) := by
    rw [hs21slB, ← hs2slStart]
    exact e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2sl))))))))))))))))))
  have hs21sn : s21.regs .nat [] "start_n" = some (Tile.scalar i) :=
    e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) hs7sn)))))))))))))
  have hs21n : s21.regs .nat [BLK] "offs_n" = some (Tile.vec (fun j : Fin BLK => j.val)) :=
    e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) hs7n)))))))))))))
  have hs21seq : s21.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s21 B_Seqlen)) := by
    rw [hs21seqB]
    exact e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) (e13 (by decide) (e12 (by decide) (e11 (by decide) (e10 (by decide) (e9 (by decide) (e8 (by decide) (e7 (by decide) (e6 (by decide) (e5 (by decide) (e4 (by decide) (e3 (by decide) hs2seq))))))))))))))))))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_v_load_evalG s21 V B_Start_Loc B_Seqlen i rs hs BLK DM hs21vp hs21sl hs21sn hs21n hs21seq))]
  set vFn : Fin BLK → Fin DM → ℝ := fun jL dd =>
    if i + jL.val < seqLen s21 B_Seqlen then
      s21.readMem V ((startLoc s21 B_Start_Loc + (i + jL.val)) * rs + s21.pids 1 * hs + dd.val)
    else (0.0 : ℝ) with hvFn
  set s22 := s21.setReg "v" .real [BLK, DM] (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) with hs22d
  rw [show (⟨fun idx : TileIndex [BLK, DM] =>
        if i + idx.1.val < seqLen s21 B_Seqlen then
          some (s21.readMem V ((startLoc s21 B_Start_Loc + (i + idx.1.val)) * rs + s21.pids 1 * hs + idx.2.1.val))
        else some (0.0 : ℝ)⟩ : Tile .real [BLK, DM])
      = (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨jL, dd, u⟩ := idx
    show (if i + jL.val < seqLen s21 B_Seqlen then
        some (s21.readMem V ((startLoc s21 B_Start_Loc + (i + jL.val)) * rs + s21.pids 1 * hs + dd.val))
      else some (0.0 : ℝ)) = some (vFn jL dd)
    by_cases hlt : i + jL.val < seqLen s21 B_Seqlen
    · simp only [hvFn, hlt, if_true]
    · simp only [hvFn, hlt, if_false]]
  have e22 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "v" → s21.regs dt sh nm = some t → s22.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs22d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs22v : s22.regs .real [BLK, DM] "v" = some (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) := by
    rw [hs22d, BlockState.setReg_same]
  have hs22p : s22.regs .real [BLK, BLK] "p" = some (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := e22 (by decide) hs19p
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLK, BLK] "p") s22 = some (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) from by
      rw [evalOp_ref]; exact hs22p))]
  set s23 := s22.setReg "p" .real [BLK, BLK] (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) with hs23d
  have e23 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "p" → s22.regs dt sh nm = some t → s23.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs23d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs23p : s23.regs .real [BLK, BLK] "p" = some (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK]) := by
    rw [hs23d, BlockState.setReg_same]
  have hs23v : s23.regs .real [BLK, DM] "v" = some (⟨fun idx : TileIndex [BLK, DM] => some (vFn idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) := e23 (by decide) hs22v
  have hs23acc : s23.regs .real [BLK, DM] "acc" = some (⟨fun idx : TileIndex [BLK, DM] => accMul idx.1 idx.2.1⟩ : Tile .real [BLK, DM]) :=
    e23 (by decide) (e22 (by decide) hs21acc)
  set psReal : Fin BLK → ℝ := fun ir => βReal ir / linewReal ir with hpsReal
  have hpsSome : ∀ ir, psFn ir = some (psReal ir) := by
    intro ir; show WithBot.realDiv (betaFn ir) (linewFn ir) = some (psReal ir)
    rw [hβSome, hlinewSome]; rfl
  have hpFinalSome : ∀ ir jL, pFinal ir jL = some (pReal ir jL * psReal ir) := by
    intro ir jL; show WithBot.realMul (pFn ir jL) (psFn ir) = some (pReal ir jL * psReal ir)
    rw [hpFnSome, hpsSome]; rfl
  set asReal : Fin BLK → ℝ := fun ir => (fold0 ir ⟨0, hDM⟩).2.1 / linewReal ir * αReal ir with hasReal
  have hasSome : ∀ ir, asFn ir = some (asReal ir) := by
    intro ir; show WithBot.realMul (WithBot.realDiv (liFn ir) (linewFn ir)) (alphaFn ir) = some (asReal ir)
    rw [hliSome, hlinewSome, hαSome]; rfl
  have haccFnSome : ∀ ir dd, accFn ir dd = some ((fold0 ir dd).2.2) := by intro ir dd; rfl
  have haccMulSome : ∀ ir dd, accMul ir dd = some ((fold0 ir dd).2.2 * asReal ir) := by
    intro ir dd; show WithBot.realMul (accFn ir dd) (asFn ir) = some ((fold0 ir dd).2.2 * asReal ir)
    rw [haccFnSome, hasSome]; rfl
  rw [show (⟨fun idx : TileIndex [BLK, BLK] => pFinal idx.1 idx.2.1⟩ : Tile .real [BLK, BLK])
        = (⟨fun idx : TileIndex [BLK, BLK] => some (pReal idx.1 idx.2.1 * psReal idx.1)⟩ : Tile .real [BLK, BLK]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, jL, u⟩ := idx; exact hpFinalSome ir jL] at hs23p
  rw [show (⟨fun idx : TileIndex [BLK, DM] => accMul idx.1 idx.2.1⟩ : Tile .real [BLK, DM])
        = (⟨fun idx : TileIndex [BLK, DM] => some ((fold0 idx.1 idx.2.1).2.2 * asReal idx.1)⟩ : Tile .real [BLK, DM]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, dd, u⟩ := idx; exact haccMulSome ir dd] at hs23acc
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_acc_dot_evalG s23 BLK DM (fun ir dd => some ((fold0 ir dd).2.2 * asReal ir))
      (fun ir jL => pReal ir jL * psReal ir) vFn hs23acc hs23p hs23v))]
  set accFinal : Fin BLK → Fin DM → ℝ := fun ir dd =>
    (fold0 ir dd).2.2 * asReal ir + Finset.univ.sum (fun jL : Fin BLK => (pReal ir jL * psReal ir) * vFn jL dd) with haccFinal
  rw [show (⟨fun idx : TileIndex [BLK, DM] =>
        WithBot.realAdd (some ((fold0 idx.1 idx.2.1).2.2 * asReal idx.1))
          (some (Finset.univ.sum (fun jL : Fin BLK => (pReal idx.1 jL * psReal idx.1) * vFn jL idx.2.1)))⟩ : Tile .real [BLK, DM])
      = (⟨fun idx : TileIndex [BLK, DM] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) from by
    refine Tile.ext (fun idx => ?_); obtain ⟨ir, dd, u⟩ := idx; rfl]
  set s24 := s23.setReg "acc" .real [BLK, DM] (⟨fun idx : TileIndex [BLK, DM] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) with hs24d
  have e24 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "acc" → s23.regs dt sh nm = some t → s24.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs24d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs24acc : s24.regs .real [BLK, DM] "acc" = some (⟨fun idx : TileIndex [BLK, DM] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) := by
    rw [hs24d, BlockState.setReg_same]
  have hs24linew : s24.regs .real [BLK] "l_i_new" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) :=
    e24 (by decide) (e23 (by decide) (e22 (by decide) (e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) hs17linew))))))
  have hs24minew : s24.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) :=
    e24 (by decide) (e23 (by decide) (e22 (by decide) (e21 (by decide) (e20 (by decide) (e19 (by decide) (e18 (by decide) (e17 (by decide) (e16 (by decide) (e15 (by decide) (e14 (by decide) hs13minew))))))))))
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLK] "l_i_new") s24 = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) from by
      rw [evalOp_ref]; exact hs24linew))]
  set s25 := s24.setReg "l_i" .real [BLK] (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) with hs25d
  have e25 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "l_i" → s24.regs dt sh nm = some t → s25.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs25d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs25minew : s25.regs .real [BLK] "m_i_new" = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) := e25 (by decide) hs24minew
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .real [BLK] "m_i_new") s25 = some (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) from by
      rw [evalOp_ref]; exact hs25minew))]
  rw [stepStmts.nil]
  set s26 := s25.setReg "m_i" .real [BLK] (⟨fun idx : TileIndex [BLK] => minewFn idx.1⟩ : Tile .real [BLK]) with hs26d
  refine ⟨s26, rfl, ?_⟩
  have hbridge : ∀ (ir : Fin BLK) (dd : Fin DM),
      (minewFn ir, linewReal ir, accFinal ir dd)
        = mistralFoldUptoG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw (c + 1) ir dd := by
    intro ir dd
    rw [mistralFoldUptoG_succ]
    have hbne : mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd ≠ [] :=
      mistralBlockMG_ne_nil s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd hwin hBLK
    obtain ⟨hci1, hci2⟩ := mistralFoldUptoG_channel_indep s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd ⟨0, hDM⟩
    set bsupDD : WithBot ℝ := (mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map (fun p => ((p.1 : ℝ) : WithBot ℝ)) |>.foldr (· ⊔ ·) ⊥ with hbsupDD
    have hbsupEq : bsup ir = bsupDD := by
      simp only [hbsup, hbsupDD, hblk]
      rw [show (fun p : ℝ × ℝ => ((p.1 : ℝ) : WithBot ℝ)) = (fun r : ℝ => ((r : ℝ) : WithBot ℝ)) ∘ Prod.fst from rfl,
        ← List.map_map, ← List.map_map,
        mistralBlockMG_fst_channel_indep s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir ⟨0, hDM⟩ dd]
    have hmiEq : miFn ir = (fold0 ir dd).1 := by simp only [hmiFn, hfold0]; exact hci1.symm
    have hmijGEq : mijFn ir = mistralGuard bsupDD := by simp only [hmijFn]; rw [hbsupEq]
    have hMnew : minewFn ir = (fold0 ir dd).1 ⊔ mistralGuard bsupDD := by
      simp only [hminewFn]; rw [hmiEq, hmijGEq]
    have hliEq : (fold0 ir ⟨0, hDM⟩).2.1 = (fold0 ir dd).2.1 := by simp only [hfold0]; exact hci2.symm
    set Gdd : ℝ := (mistralGuard bsupDD).unbotD 0 with hGddDef
    have hGddNe : mistralGuard bsupDD ≠ ⊥ := by
      apply mistralGuard_ne_bot; simp only [hbsupDD]; exact fun h => hbne ((foldr_sup_coe_bot_iff _).mp h)
    have hGddCoe : mistralGuard bsupDD = ((Gdd : ℝ) : WithBot ℝ) := by
      cases hg : mistralGuard bsupDD with
      | bot => exact absurd hg hGddNe
      | coe r => simp only [hGddDef, hg]; rfl
    have hlijEqB : lijReal ir = ((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd))).sum := by
      have hMr := mistralBlockMG_lij_sum s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd Gdd hwin
      have hsum : ((lijReal ir : ℝ) : WithBot ℝ)
          = some (((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd))).sum) := by
        rw [← hMr, hlijReal, ← WithBot.sum_some_eq_some]
        apply Finset.sum_congr rfl; intro jL _
        have hpr : pReal ir jL = (WithBot.realExp (WithBot.realSub
            ((mistralScoreM s0 Q K B_Start_Loc sc rs hs BLK DM S bel sw ir ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩ : ℝ) : WithBot ℝ)
            ((Gdd : ℝ) : WithBot ℝ))).unbotD 0 := by
          show (WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0 = _
          rw [hqkW, hscoreFn, hmijGEq, hGddCoe]
        rw [hpr]; exact (nopad_realExp_eq_some_unbotD _).symm
      exact WithBot.coe_inj.mp hsum
    have hacccsum : Finset.univ.sum (fun jL : Fin BLK => pReal ir jL * vFn jL dd)
        = ((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd) * p.2)).sum := by
      have hvEq : ∀ jL : Fin BLK, vFn jL dd
          = ctxVTileMG s0 V B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩, dd, PUnit.unit) := by
        intro jL
        show (if i + jL.val < seqLen s21 B_Seqlen then
            s21.readMem V ((startLoc s21 B_Start_Loc + (i + jL.val)) * rs + s21.pids 1 * hs + dd.val)
          else (0.0 : ℝ))
          = ctxVTileMG s0 V B_Start_Loc rs hs S DM bel (⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩, dd, PUnit.unit)
        rw [ctxVTileMG, ctxVTileG, hs21seqB, hs21slB, hs21pids1,
          show i + jL.val = c * BLK + jL.val from by rw [hi]]
        by_cases hlt : c * BLK + jL.val < bel
        · rw [if_pos hlt, if_pos hlt]; simp only [BlockState.readMem, hs21mem]
        · rw [if_neg hlt, if_neg hlt]; norm_num
      have hbr := mistralBlockMG_acc_sum s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd Gdd hwin
        (fun jL => vFn jL dd) (fun jL => hvEq jL)
      have hcoe : ((Finset.univ.sum (fun jL : Fin BLK => pReal ir jL * vFn jL dd) : ℝ) : WithBot ℝ)
          = some (((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd) * p.2)).sum) := by
        rw [← hbr, ← WithBot.sum_some_eq_some]
        apply Finset.sum_congr rfl; intro jL _
        have hpr : pReal ir jL = (WithBot.realExp (WithBot.realSub
            ((mistralScoreM s0 Q K B_Start_Loc sc rs hs BLK DM S bel sw ir ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩ : ℝ) : WithBot ℝ)
            ((Gdd : ℝ) : WithBot ℝ))).unbotD 0 := by
          show (WithBot.realExp (WithBot.realSub (qkW ir jL) (mijFn ir))).unbotD 0 = _
          rw [hqkW, hscoreFn, hmijGEq, hGddCoe]
        rw [show WithBot.realMul
              (WithBot.realExp (WithBot.realSub
                ((mistralScoreM s0 Q K B_Start_Loc sc rs hs BLK DM S bel sw ir ⟨c * BLK + jL.val, mistral_lane_lt_SG c S BLK hwin jL⟩ : ℝ) : WithBot ℝ)
                ((Gdd : ℝ) : WithBot ℝ)))
              ((vFn jL dd : ℝ) : WithBot ℝ)
            = ((pReal ir jL * vFn jL dd : ℝ) : WithBot ℝ) from by
          rw [hpr, WithBot.realSub_coe_coe, WithBot.realExp_coe, WithBot.unbotD_coe, WithBot.realMul_coe_coe]]
      exact WithBot.coe_inj.mp hcoe
    show (minewFn ir, linewReal ir, accFinal ir dd)
      = mistralBlockStepBot (fold0 ir dd) (mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd)
    have hα : αReal ir = (WithBot.realExp (WithBot.realSub (fold0 ir dd).1 ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0 := by
      show (WithBot.realExp (WithBot.realSub (miFn ir) (minewFn ir))).unbotD 0 = _
      rw [hmiEq, hMnew]
    have hβ : βReal ir = (WithBot.realExp (WithBot.realSub (mistralGuard bsupDD) ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0 := by
      show (WithBot.realExp (WithBot.realSub (mijFn ir) (minewFn ir))).unbotD 0 = _
      rw [hmijGEq, hMnew]
    have hl'eq : linewReal ir
        = (WithBot.realExp (WithBot.realSub (fold0 ir dd).1 ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0 * (fold0 ir dd).2.1
          + (WithBot.realExp (WithBot.realSub (mistralGuard bsupDD) ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0
            * ((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map (fun p => Real.exp (p.1 - (mistralGuard bsupDD).unbotD 0))).sum := by
      rw [← hα, ← hβ, ← hGddDef, ← hlijEqB]; simp only [hlinewReal]; rw [hliEq]
    simp only [mistralBlockStepBot]
    refine Prod.ext ?_ (Prod.ext ?_ ?_)
    · show minewFn ir = (fold0 ir dd).1 ⊔ mistralGuard bsupDD; exact hMnew
    · exact hl'eq
    · show accFinal ir dd
        = (fold0 ir dd).2.2
            * ((fold0 ir dd).2.1
              / ((WithBot.realExp (WithBot.realSub (fold0 ir dd).1 ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0 * (fold0 ir dd).2.1
                  + (WithBot.realExp (WithBot.realSub (mistralGuard bsupDD) ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0
                    * ((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map (fun p => Real.exp (p.1 - (mistralGuard bsupDD).unbotD 0))).sum)
              * (WithBot.realExp (WithBot.realSub (fold0 ir dd).1 ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0)
          + ((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map
              (fun p => Real.exp (p.1 - (mistralGuard bsupDD).unbotD 0)
                * (WithBot.realExp (WithBot.realSub (mistralGuard bsupDD) ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0
                / ((WithBot.realExp (WithBot.realSub (fold0 ir dd).1 ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0 * (fold0 ir dd).2.1
                    + (WithBot.realExp (WithBot.realSub (mistralGuard bsupDD) ((fold0 ir dd).1 ⊔ mistralGuard bsupDD))).unbotD 0
                      * ((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map (fun p => Real.exp (p.1 - (mistralGuard bsupDD).unbotD 0))).sum)
                * p.2)).sum
      set BV : ℝ := ((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd) * p.2)).sum with hBV
      rw [← hα, ← hβ, ← hGddDef]
      rw [show ((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map (fun p => Real.exp (p.1 - Gdd))).sum
            = lijReal ir from hlijEqB.symm]
      rw [show αReal ir * (fold0 ir dd).2.1 + βReal ir * lijReal ir = linewReal ir from by
        simp only [hlinewReal]; rw [hliEq]]
      rw [show ((mistralBlockMG s0 Q K V B_Start_Loc sc rs hs BLK DM S bel sw c ir dd).map
            (fun p => Real.exp (p.1 - Gdd) * βReal ir / linewReal ir * p.2)).sum
          = βReal ir / linewReal ir * BV from by
        rw [hBV, ← List.sum_map_mul_left]
        apply congrArg; apply List.map_congr_left; intro p _; ring]
      simp only [haccFinal, hasReal]
      rw [hliEq]
      rw [show psReal ir = βReal ir / linewReal ir from rfl]
      rw [show (fun jL : Fin BLK => pReal ir jL * (βReal ir / linewReal ir) * vFn jL dd)
            = (fun jL : Fin BLK => (βReal ir / linewReal ir) * (pReal ir jL * vFn jL dd)) from by funext jL; ring]
      rw [← Finset.mul_sum, hacccsum]
  have e26 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "m_i" → s25.regs dt sh nm = some t → s26.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs26d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have chainAll : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "start_n" → nm ≠ "k" → nm ≠ "qk" → nm ≠ "m_ij" → nm ≠ "p" → nm ≠ "l_ij"
      → nm ≠ "m_i_new" → nm ≠ "alpha" → nm ≠ "beta" → nm ≠ "l_i_new" → nm ≠ "p_scale"
      → nm ≠ "acc_scale" → nm ≠ "acc" → nm ≠ "v" → nm ≠ "l_i" → nm ≠ "m_i"
      → s.regs dt sh nm = some t → s26.regs dt sh nm = some t := by
    intro dt sh nm t h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11 h12 h13 h14 h15 h16 h
    exact e26 h16 (e25 h15 (e24 h13 (e23 h5 (e22 h14 (e21 h13 (e20 h12 (e19 h5 (e18 h11 (e17 h10 (e16 h10 (e15 h9 (e14 h8 (e13 h7 (e12 h6 (e11 h5 (e10 h4 (e9 h4 (e8 h3 (e7 h3 (e6 h3 (e5 h3 (e4 h3 (e3 h2 (e2 h1 (e1 h1 h)))))))))))))))))))))))))
  have hs26pids : s26.pids = s0.pids := by
    rw [hs26d, BlockState.setReg_pids, hs25d, BlockState.setReg_pids, hs24d, BlockState.setReg_pids,
      hs23d, BlockState.setReg_pids, hs22d, BlockState.setReg_pids, hs21pids]
  have hs26mem : s26.mem = s0.mem := by
    funext rg o; rw [hs26d, BlockState.setReg_mem, hs25d, BlockState.setReg_mem, hs24d, BlockState.setReg_mem,
      hs23d, BlockState.setReg_mem, hs22d, BlockState.setReg_mem]; exact hs21mem ▸ rfl
  have hs26undef : ∀ rg o, s26.undef rg o = 0 := by
    intro rg o
    rw [hs26d, BlockState.setReg_undef, hs25d, BlockState.setReg_undef, hs24d, BlockState.setReg_undef,
      hs23d, BlockState.setReg_undef, hs22d, BlockState.setReg_undef, hs21d, BlockState.setReg_undef,
      hs20d, BlockState.setReg_undef, hs19d, BlockState.setReg_undef, hs18d, BlockState.setReg_undef,
      hs17d, BlockState.setReg_undef, hs16d, BlockState.setReg_undef, hs15d, BlockState.setReg_undef,
      hs14d, BlockState.setReg_undef, hs13d, BlockState.setReg_undef, hs12d, BlockState.setReg_undef,
      hs11d, BlockState.setReg_undef, hs10d, BlockState.setReg_undef, hs9d, BlockState.setReg_undef,
      hs8d, BlockState.setReg_undef, hs7d, BlockState.setReg_undef, hs6d, BlockState.setReg_undef,
      hs5d, BlockState.setReg_undef, hs4d, BlockState.setReg_undef, hs3d, BlockState.setReg_undef,
      hs2d, BlockState.setReg_undef, hs1d, BlockState.setReg_undef]
    exact hundef rg o
  refine ⟨hs26pids, hs26mem, hs26undef, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hcb
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hch
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hsm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hckv
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hseq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hsl
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hn
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hd
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hoffm
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hq
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hkp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hvp
  · exact chainAll (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) (by decide) hbmask
  · rw [hs26d, BlockState.setReg_same]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, u⟩ := idx
    show minewFn ir = (mistralFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxMistralWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) sw (c + 1) ir ⟨0, hDM⟩).1
    exact congrArg (Prod.fst) (hbridge ir ⟨0, hDM⟩)
  · rw [show s26.regs .real [BLK] "l_i" = some (⟨fun idx : TileIndex [BLK] => linewFn idx.1⟩ : Tile .real [BLK]) from by
      rw [hs26d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs25d, BlockState.setReg_same]]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, u⟩ := idx
    show linewFn ir = ((mistralFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxMistralWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) sw (c + 1) ir ⟨0, hDM⟩).2.1 : WithBot ℝ)
    rw [hlinewSome ir]
    exact congrArg (fun p => ((p.2.1 : ℝ) : WithBot ℝ)) (hbridge ir ⟨0, hDM⟩)
  · rw [show s26.regs .real [BLK, DM] "acc" = some (⟨fun idx : TileIndex [BLK, DM] => some (accFinal idx.1 idx.2.1)⟩ : Tile .real [BLK, DM]) from by
      rw [hs26d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs25d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide), hs24d, BlockState.setReg_same]]
    refine congrArg some (Tile.ext (fun idx => ?_)); obtain ⟨ir, dd, u⟩ := idx
    show some (accFinal ir dd) = ((mistralFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxMistralWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) sw (c + 1) ir dd).2.2 : WithBot ℝ)
    exact congrArg (fun p => ((p.2.2 : ℝ) : WithBot ℝ)) (hbridge ir dd)

/-- General `contextAttnMistralExactFoldMG` transports across mem/pids-equal states. -/
theorem contextAttnMistralExactFoldMG_eq_of_mem_pidsG
    (s s0 : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ) (rs hs BLK DM S bel sw : Nat)
    (idx : TileIndex [BLK, DM]) (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) :
    contextAttnMistralExactFoldMG s Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw idx
      = contextAttnMistralExactFoldMG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw idx := by
  have hsl : startLoc s B_Start_Loc = startLoc s0 B_Start_Loc := mistral_startLoc_eq_of_mem_pids s s0 B_Start_Loc hmem hpids
  simp only [contextAttnMistralExactFoldMG, mistralScore, ctxQTileG, ctxKTileMG, ctxKTileG,
    ctxVTileMG, ctxVTileG, hsl, hpids, BlockState.readMem, hmem]

/-- General `mistralGenuineOutValueG` transports across mem/pids-equal states. -/
theorem mistralGenuineOutValueG_eq_of_mem_pids
    (s s0 : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM sw : Nat)
    (hmem : s.mem = s0.mem) (hpids : s.pids = s0.pids) (idx : TileIndex [BLK, DM]) :
    mistralGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM sw idx
      = mistralGenuineOutValueG s0 Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM sw idx := by
  have hseqE : seqLen s B_Seqlen = seqLen s0 B_Seqlen := mistral_seqLen_eq_of_mem_pids s s0 B_Seqlen hmem hpids
  have hWin : ctxMistralWindowG s B_Seqlen BLK = ctxMistralWindowG s0 B_Seqlen BLK := by
    simp only [ctxMistralWindowG, hseqE, hpids]
  have hBel : ctxMistralBel s B_Seqlen = ctxMistralBel s0 B_Seqlen := by simp only [ctxMistralBel, hseqE]
  simp only [mistralGenuineOutValueG, hWin, hBel]
  exact contextAttnMistralExactFoldMG_eq_of_mem_pidsG s s0 Q K V B_Start_Loc sm_scale rs hs BLK DM _ _ _ idx hmem hpids

/-- **General: at the full window, an active row's running `acc` is the genuine
closed form.** -/
theorem mistralFoldUpto_full_eq_genuineG
    (s0 : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName) (sm_scale : ℝ)
    (rs hs BLK DM : Nat) (hBLK : 0 < BLK) (sw : Nat)
    (idx : TileIndex [BLK, DM])
    (hact : s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen) :
    (mistralFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM
        (ctxMistralWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) sw
        (ctxMistralWindowG s0 B_Seqlen BLK / BLK) idx.1 idx.2.1).2.2
      = mistralGenuineOutValueG s0 Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM sw idx := by
  have hbm : BLK * s0.pids 2 < seqLen s0 B_Seqlen := by have := idx.1.isLt; nlinarith [Nat.zero_le BLK]
  have hpos : 0 < ctxMistralWindowG s0 B_Seqlen BLK := by
    rw [ctxMistralWindowG]
    have hbm' : BLK * s0.pids 2 < seqLen s0 B_Seqlen := hbm
    simp only [hbm', if_true, one_mul]
    positivity
  set S := ctxMistralWindowG s0 B_Seqlen BLK with hSdef
  set bel := seqLen s0 B_Seqlen with hbeldef
  have hSmul : S % BLK = 0 := by
    rw [hSdef, ctxMistralWindowG]
    by_cases hb : BLK * s0.pids 2 < seqLen s0 B_Seqlen
    · simp only [hb, if_true, one_mul]; exact Nat.mul_mod_left _ _
    · simp only [hb, if_false, Nat.zero_mul]; simp
  have hnb : (S / BLK) * BLK = S := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hSmul)
  have hbridge := mistralFoldUptoG_full_eq_genuine s0 Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw (S / BLK) idx hBLK hnb hpos (by rw [hbeldef]; exact hact)
  rw [hbridge]
  show contextAttnMistralExactFoldMG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM S bel sw idx
    = mistralGenuineOutValueG s0 Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM sw idx
  rw [mistralGenuineOutValueG, ctxMistralBel, ← hSdef, ← hbeldef]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General postLoop execution + genuine readback** (contiguous `(rs, hs, 1)`,
`DM ≤ rs` for output-offset injectivity). -/
theorem mistralPostLoop_evalG
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs) (sw : Nat)
    (s0 : BlockState) (cF : Nat) (s : BlockState)
    (hfull : cF * BLK = ctxMistralWindowG s0 B_Seqlen BLK)
    (hinv : mistralInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM sw s0 cF s) :
    ∃ sP, stepStmts (mistralPostLoopG Out rs hs BLK DM) s = some sP
      ∧ ∀ idx : TileIndex [BLK, DM],
          sP.readMem Out (mistralOutOffsetG s0 B_Start_Loc rs hs BLK DM idx)
            = if s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen then
                mistralGenuineOutValueG s0 Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM sw idx
              else s.readMem Out (mistralOutOffsetG s0 B_Start_Loc rs hs BLK DM idx) := by
  obtain ⟨hpids, hmem, hundef, hcb, hch, hsm, hckv, hseq, hsl, hn, hd, hoffm, hq, hkp, hvp, hbmask, hmi, hli, hacc⟩ := hinv
  unfold mistralPostLoopG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mistral_offq_evalG s (startLoc s0 B_Start_Loc) (s0.pids 1) (s0.pids 2) rs hs BLK DM hsl hch hoffm hd))]
  set s1 := s.setReg "off_o" .nat [BLK, DM]
    (⟨fun idx : TileIndex [BLK, DM] => (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) with hs1d
  have e1 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "off_o" → s.regs dt sh nm = some t → s1.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs1d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs1offo : s1.regs .nat [BLK, DM] "off_o" = some
      (⟨fun idx : TileIndex [BLK, DM] => (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val⟩ : Tile .nat [BLK, DM]) := by
    rw [hs1d, BlockState.setReg_same]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [BLK, DM] "off_o")) s1
        = some (⟨fun idx : TileIndex [BLK, DM] =>
            (Out, (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) from by
      simp only [evalOp, evalOp_ref, hs1offo, Option.bind]
      refine congrArg some (Tile.ext (fun idx => ?_))
      obtain ⟨ir, dd, u⟩ := idx
      simp only [Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Region.cast_id, Nat.zero_add, Prod.mk.injEq, true_and]))]
  set s2 := s1.setReg "out_ptrs" .ptr [BLK, DM]
    (⟨fun idx : TileIndex [BLK, DM] =>
      (Out, (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) with hs2d
  have e2 : ∀ {dt : TileDType} {sh : TileShape} {nm : RegName} {t : Tile dt sh},
      nm ≠ "out_ptrs" → s1.regs dt sh nm = some t → s2.regs dt sh nm = some t := by
    intro dt sh nm t hne h; rw [hs2d, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]; exact h
  have hs2ptr : s2.regs .ptr [BLK, DM] "out_ptrs" = some
      (⟨fun idx : TileIndex [BLK, DM] => (Out, (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)⟩ : Tile .ptr [BLK, DM]) := by
    rw [hs2d, BlockState.setReg_same]
  have hs2acc : s2.regs .real [BLK, DM] "acc" = some
      (⟨fun idx : TileIndex [BLK, DM] =>
        ((mistralFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxMistralWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) sw cF idx.1 idx.2.1).2.2 : WithBot ℝ)⟩ : Tile .real [BLK, DM]) :=
    e2 (by decide) (e1 (by decide) hacc)
  have hs2m : s2.regs .nat [BLK] "offs_m" = some (Tile.vec (fun i : Fin BLK => s0.pids 2 * BLK + i.val)) :=
    e2 (by decide) (e1 (by decide) hoffm)
  have hs2seq : s2.regs .nat [] "cur_batch_seq_len" = some (Tile.scalar (seqLen s0 B_Seqlen)) :=
    e2 (by decide) (e1 (by decide) hseq)
  have hs2mem : s2.mem = s0.mem := by
    funext rg o; rw [hs2d, BlockState.setReg_mem, hs1d, BlockState.setReg_mem]; exact hmem ▸ rfl
  have hs2pids : s2.pids = s0.pids := by rw [hs2d, BlockState.setReg_pids, hs1d, BlockState.setReg_pids]; exact hpids
  have hstore : stepStmt (Stmt.store .real [BLK, DM] (MemAccess.ptr (Op.ref .ptr [BLK, DM] "out_ptrs"))
      (Op.ref .real [BLK, DM] "acc")
      (MaskOpt.mask (Op.remap [BLK, DM] Broadcast.nil.consL.consSame.leftIndex
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m"))
          (Op.ref .nat [] "cur_batch_seq_len"))))) s2
      = some ((TileShape.allIndices [BLK, DM]).foldl
          (fun acc idx =>
            if s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen then
              acc.writeMem Out ((startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)
                ((mistralFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxMistralWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) sw cF idx.1 idx.2.1).2.2 : ℝ)
            else acc) s2) := by
    have hexpM : @evalOp .nat [BLK, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLK] "offs_m")) s2
        = some (Tile.expandDim ⟨1, by simp⟩ (Tile.vec (fun i : Fin BLK => s0.pids 2 * BLK + i.val))) :=
      evalOp_expandDim_ref_of_regs .nat [BLK] ⟨1, by simp⟩ "offs_m" s2 _ hs2m
    unfold stepStmt
    simp only [evalOp_ref, hs2acc, hs2ptr, hs2seq, evalOp, hexpM, Option.bind, Option.map]
    refine congrArg some ?_
    congr 1
    funext acc idx
    obtain ⟨ir, dd, u⟩ := idx
    simp only [Tile.cop_data, Tile.bop_data, Tile.bop, Tile.remap, Tile.expandDim, Tile.vec,
      Tile.scalar, ComparableDType.lt, Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.dropInsertedIndex, BlockState.writeMemTyped_real, FloatDType.real_storeValue,
      decide_eq_true_eq, WithBot.unbotD_coe]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hinj : Function.Injective (fun idx : TileIndex [BLK, DM] =>
      (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val) := by
    rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
    simp only at h
    have hda' : da < rs := lt_of_lt_of_le hda hDMrs
    have hdb' : db < rs := lt_of_lt_of_le hdb hDMrs
    have hm : ma = mb := by nlinarith [h, hda', hdb', Nat.zero_le rs]
    subst hm
    have hd2 : da = db := by omega
    subst hd2; rfl
  rw [show mistralOutOffsetG s0 B_Start_Loc rs hs BLK DM idx
        = (fun idx : TileIndex [BLK, DM] => (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val) idx from by
    simp only [mistralOutOffsetG]]
  rw [BlockState.scatter_readback_prop_masked_nd s2
    (fun idx : TileIndex [BLK, DM] => (startLoc s0 B_Start_Loc + (s0.pids 2 * BLK + idx.1.val)) * rs + s0.pids 1 * hs + idx.2.1.val)
    (fun idx : TileIndex [BLK, DM] =>
      ((mistralFoldUptoG s0 Q K V B_Start_Loc sm_scale rs hs BLK DM (ctxMistralWindowG s0 B_Seqlen BLK) (seqLen s0 B_Seqlen) sw cF idx.1 idx.2.1).2.2 : ℝ))
    (fun idx : TileIndex [BLK, DM] => s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen) hinj idx]
  by_cases hact : s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen
  · rw [if_pos hact, if_pos hact]
    rw [show cF = ctxMistralWindowG s0 B_Seqlen BLK / BLK from by rw [← hfull, Nat.mul_div_cancel _ hBLK]]
    exact mistralFoldUpto_full_eq_genuineG s0 Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hBLK sw idx hact
  · rw [if_neg hact, if_neg hact]
    simp only [BlockState.readMem, hs2mem, hmem]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General full-kernel execution chain.** -/
theorem mistral_exec_general
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (sm_scale : ℝ) (rs hs BLK DM : Nat) (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs) (sw : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sF, stepStmts ((context_attn_mistral_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        rs hs 1 rs hs 1 rs hs 1 rs hs 1 1 sw BLK DM BLK).toAlgKernel.body) s = some sF
      ∧ ∀ idx : TileIndex [BLK, DM],
          s.pids 2 * BLK + idx.1.val < seqLen s B_Seqlen →
            sF.readMem Out (mistralOutOffsetG s B_Start_Loc rs hs BLK DM idx)
              = mistralGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM sw idx := by
  rw [mistral_body_splitG]
  obtain ⟨s0, hpre, hinv0⟩ := mistralPreLoop_evalG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hBLK hDM sw hundef
  rw [stepStmts.append_some hpre]
  obtain ⟨hpids0, hmem0, hundef0, hcb0, hch0, hsm0, hckv0, hseq0, hsl0, hn0, hd0, hoffm0, hq0, hkp0, hvp0, hbmask0, hmi0, hli0, hacc0⟩ := hinv0
  have hs0seq : seqLen s0 B_Seqlen = seqLen s B_Seqlen := mistral_seqLen_eq_of_mem_pids s0 s B_Seqlen hmem0 hpids0
  set S := ctxMistralWindowG s0 B_Seqlen BLK with hSdef
  have hSmulBLK : S % BLK = 0 := by
    rw [hSdef, ctxMistralWindowG]
    by_cases hbm : BLK * s0.pids 2 < seqLen s0 B_Seqlen
    · simp only [hbm, if_true, one_mul]; exact Nat.mul_mod_left _ _
    · simp only [hbm, if_false, Nat.zero_mul]; simp
  have hinv0' : mistralInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM sw s0 0 s0 :=
    mistralInvariantG_zero_reanchor Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM sw s s0 ⟨hpids0, hmem0, hundef0, hcb0, hch0, hsm0, hckv0, hseq0, hsl0, hn0, hd0, hoffm0, hq0, hkp0, hvp0, hbmask0, hmi0, hli0, hacc0⟩
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    VeriTile.Triton.forRangeDyn_inv (idx := "start_n")
      (startOp := Op.constNat 0)
      (stopOp := Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
          (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
        (Op.constNat BLK))
      (stepOp := Op.constNat BLK)
      (P := fun i st => mistralInvariantG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hDM sw s0 (i / BLK) st ∧ i % BLK = 0 ∧ i ≤ S)
      (s_init := s0) (start := 0) (stop := S) (step := BLK)
      (by rw [evalOp_constNat])
      (by
        rw [evalOp_mul, evalOp_mul, evalOp_ref, hbmask0, evalOp_add, evalOp_ref, hsm0, evalOp_constNat, evalOp_constNat]
        simp only [Option.bind_eq_bind, Option.bind_some]
        refine congrArg some ?_
        refine Tile.ext (fun idx => ?_)
        simp only [Tile.bop_data, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.mul, NumericDType.add]
        show (if BLK * s.pids 2 < seqLen s B_Seqlen.cast then 1 else 0) * (s.pids 2 + 1) * BLK = S
        rw [hSdef, ctxMistralWindowG, hpids0, hs0seq])
      (by rw [evalOp_constNat])
      (by omega)
      ⟨by rw [Nat.zero_div]; exact hinv0', Nat.zero_mod BLK, Nat.zero_le _⟩
      (fun i st hi hP => by
        obtain ⟨hPinv, hPmod, hPle⟩ := hP
        have hdvd : i = (i / BLK) * BLK := (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hPmod)).symm
        obtain ⟨s', hstep, hinv'⟩ := mistral_attn_stepG Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM hBLK hDM sw s0 i st hi hPinv hdvd
        have hmod' : (i + BLK) % BLK = 0 := by rw [Nat.add_mod_right]; exact hPmod
        have hle' : i + BLK ≤ S := by
          have hlt : i < S := hi
          obtain ⟨a, ha⟩ : BLK ∣ i := Nat.dvd_of_mod_eq_zero hPmod
          obtain ⟨b, hb⟩ : BLK ∣ S := Nat.dvd_of_mod_eq_zero hSmulBLK
          have hab : a < b := by
            rw [ha, hb] at hlt
            by_contra hc; push_neg at hc
            have : BLK * b ≤ BLK * a := Nat.mul_le_mul_left BLK hc
            omega
          rw [ha, hb]
          calc BLK * a + BLK = BLK * (a + 1) := by ring
            _ ≤ BLK * b := Nat.mul_le_mul_left BLK (by omega)
        refine ⟨s', hstep, ?_, hmod', hle'⟩
        rw [show (i + BLK) / BLK = i / BLK + 1 from by
          rw [Nat.add_div_right i hBLK]]; exact hinv')
  rw [stepStmts.cons_some hloop]
  obtain ⟨hinvLinv, hinvLmod, hinvLle⟩ := hinvL
  have hfinS : final = S := Nat.le_antisymm hinvLle hfin
  rw [hfinS] at hinvLinv
  have hfull : (S / BLK) * BLK = S := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hSmulBLK)
  obtain ⟨sP, hpost, hOut⟩ := mistralPostLoop_evalG Q K V B_Start_Loc B_Seqlen Out sm_scale rs hs BLK DM hBLK hDM hDMrs sw s0 (S / BLK) sL
    (by rw [hfull]) hinvLinv
  rw [hpost]
  refine ⟨sP, rfl, ?_⟩
  intro idx hact
  have hslEq0 : startLoc s B_Start_Loc = startLoc s0 B_Start_Loc :=
    mistral_startLoc_eq_of_mem_pids s s0 B_Start_Loc hmem0.symm hpids0.symm
  have hoeq : mistralOutOffsetG s B_Start_Loc rs hs BLK DM idx = mistralOutOffsetG s0 B_Start_Loc rs hs BLK DM idx := by
    simp only [mistralOutOffsetG, ← hpids0, hslEq0]
  have hactS0 : s0.pids 2 * BLK + idx.1.val < seqLen s0 B_Seqlen := by
    rw [hs0seq, hpids0]; exact hact
  rw [hoeq, hOut idx, if_pos hactS0]
  exact mistralGenuineOutValueG_eq_of_mem_pids s0 s Q K V B_Start_Loc.cast B_Seqlen.cast sm_scale rs hs BLK DM sw hmem0 hpids0 idx

/-- General active-output predicate. -/
def mistralActiveG (s : BlockState) (B_Seqlen : RegionName) (BLK DM : Nat)
    (idx : TileIndex [BLK, DM]) : Prop :=
  s.pids 2 * BLK + idx.1.val < seqLen s B_Seqlen

instance mistralActiveGDecidable (s : BlockState) (B_Seqlen : RegionName) (BLK DM : Nat)
    (idx : TileIndex [BLK, DM]) : Decidable (mistralActiveG s B_Seqlen BLK DM idx) := by
  unfold mistralActiveG; infer_instance


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
/-- **General Public summary for `context_attn_mistral.py`.**

The full faithful `_fwd_kernel` surface (preLoop + streaming-softmax `forRangeDyn`
loop + masked store) *realizes* the genuine sliding-window causal-softmax closed
form `mistralGenuineOutValueG` — the boundary-masked sliding-window-softmax fold of
the loaded Q/K/V memory with the `-1e9` sentinel kept — at every active output
lane, at the **dimension-parameterized** contiguous layout `(stride_*bs, stride_*h,
stride_*d) = (rs, hs, 1)`, `kv_group_num = 1`, with `BLOCK_M = BLOCK_N = BLK`,
`BLOCK_DMODEL = DM`, `sliding_window = sw`, and a free `sm_scale`. NOT a
self-referential executed value: the streaming `m_i`/`l_i`/`acc` recurrence (two
`where` masks, the `-1e9` block-max guard, and the `l_i_new` denominator guard) is
decoded statement-by-statement and proven to collapse to the closed form. Side
conditions: `0 < BLK`, `0 < DM`, `DM ≤ rs` (output-offset injectivity), `hundef`.
Instantiating `BLK = DM = 128`, `rs = 768`, `hs = 128`, `sm_scale = (√128)⁻¹`,
`sw ∈ {10, 20}` recovers the Python test-shape summary. -/
theorem context_attn_mistral_genuine_output_summary_general
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (sm_scale : ℝ) (rs hs BLK DM sw : Nat)
    (hBLK : 0 < BLK) (hDM : 0 < DM) (hDMrs : DM ≤ rs)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_mistral_fwd_kernel_surface Q K V sm_scale B_Start_Loc B_Seqlen Out
        rs hs 1 rs hs 1 rs hs 1 rs hs 1 1 sw BLK DM BLK)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLK, DM] => mistralActiveG s B_Seqlen BLK DM idx)
        (fun idx : TileIndex [BLK, DM] => (Out, mistralOutOffsetG s B_Start_Loc rs hs BLK DM idx)))
      (expected := fun idx : TileIndex [BLK, DM] =>
        mistralGenuineOutValueG s Q K V B_Start_Loc B_Seqlen sm_scale rs hs BLK DM sw idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_mistral_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  obtain ⟨sF, hstep, hOut⟩ := mistral_exec_general Q K V B_Start_Loc B_Seqlen Out sm_scale rs hs BLK DM hBLK hDM hDMrs sw s hundef
  rw [show exec _ s = stepStmts _ s from rfl, hstep] at hExec
  obtain rfl : sF = s' := Option.some.inj hExec
  simp only [ComputeCorrect.OutputReadable.read_real]
  exact hOut idx hActive

/-- **Public genuine Python test-shape summary for `context_attn_mistral.py`.**
The full faithful `_fwd_kernel` surface realizes the genuine sliding-window
causal-softmax closed form `ctxMistralGenuineOutValue` (boundary-masked
sliding-window-softmax fold of the loaded Q/K/V memory, `-1e9` sentinel kept) at
every active output lane — NOT a self-referential executed value: the streaming
`m_i`/`l_i`/`acc` recurrence (two `where` masks, the `-1e9` block-max guard, and
the `l_i_new` denominator guard) is decoded statement-by-statement and proven to
collapse to the closed form. Bundles `sliding_window ∈ {10, 20}`.

This is now a thin corollary of the dimension-parameterized
`context_attn_mistral_genuine_output_summary_general`, instantiated at the test
shape `rs = 768`, `hs = 128`, `BLK = DM = 128`, `sm_scale = (√128)⁻¹`,
`sw ∈ {10, 20}`. The side conditions `0 < BLK`, `0 < DM`, `DM ≤ rs` discharge by
`norm_num`; `ctxMistralGenuineOutValue` is *definitionally* the general
`mistralGenuineOutValueG` at these args (and `active`/`outOffset` likewise reduce
to `mistralActiveG`/`mistralOutOffsetG`). -/
theorem context_attn_mistral_genuine_output_summary
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := context_attn_mistral_fwd_kernel_surface Q K V
        ((Real.sqrt (128 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out
        768 128 1 768 128 1 768 128 1 768 128 1 1 10 128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s B_Seqlen 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 768 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        ctxMistralGenuineOutValue s Q K V B_Start_Loc B_Seqlen 10 idx) ∧
    ComputeCorrect.Realizes
      (kernel := context_attn_mistral_fwd_kernel_surface Q K V
        ((Real.sqrt (128 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out
        768 128 1 768 128 1 768 128 1 768 128 1 1 20 128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s B_Seqlen 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 768 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        ctxMistralGenuineOutValue s Q K V B_Start_Loc B_Seqlen 20 idx) := by
  have hoff : ∀ idx : TileIndex [128, 128],
      outOffset s B_Start_Loc 768 128 1 128 idx
        = mistralOutOffsetG s B_Start_Loc 768 128 128 128 idx := by
    intro idx
    simp only [outOffset, mIndex, dIndex, mistralOutOffsetG, Nat.mul_one]
  constructor
  · have hgen :=
      context_attn_mistral_genuine_output_summary_general Q K V B_Start_Loc B_Seqlen Out
        ((Real.sqrt (128 : ℝ))⁻¹) 768 128 128 128 10 (by norm_num) (by norm_num) (by norm_num)
        s hundef
    simpa only [ctxMistralGenuineOutValue, active, mIndex, mistralActiveG,
      ComputeCorrect.WriteMap.writeIf, hoff] using hgen
  · have hgen :=
      context_attn_mistral_genuine_output_summary_general Q K V B_Start_Loc B_Seqlen Out
        ((Real.sqrt (128 : ℝ))⁻¹) 768 128 128 128 20 (by norm_num) (by norm_num) (by norm_num)
        s hundef
    simpa only [ctxMistralGenuineOutValue, active, mIndex, mistralActiveG,
      ComputeCorrect.WriteMap.writeIf, hoff] using hgen

end MistralGeneralExec

end Correct


end VeriTile.Bench.TritonBenchG.ContextAttnMistral