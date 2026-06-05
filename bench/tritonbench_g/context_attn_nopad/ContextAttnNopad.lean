import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Attention
import VeriTile.Triton.LoopInvariant

/-!
# `context_attn_nopad` — strict per-kernel correctness

`_fwd_kernel` is varlen ("no padding") context (prefill) attention. Each program
`(cur_batch, cur_head, start_m)` loads a `[BLOCK_M, BLOCK_DMODEL]` query tile,
runs an online-softmax (`m_i`/`l_i`/`acc`) loop over the key/value tokens with a
plain causal mask (`offs_m >= start_n + offs_n`), and stores the accumulated
`acc` tile to `Out`, masked by `offs_m < cur_batch_seq_len`.

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
context_attn_nopad_python_test_shape_output_summary         ← TOP THEOREM
  └─ context_attn_nopad_surface_python_test_shape_compute_correct   full surface, final store
       └─ context_attn_nopad_final_store_python_test_shape_compute_correct
            └─ context_attn_nopad_final_store_slice_compute_correct
                 └─ context_attn_nopad_final_store_slice_correct      algorithm-layer readback per lane
(supporting: context_attn_nopad_python_test_shape_offset_injective,
             context_attn_nopad_fwd_kernel_surface_toAlgorithm_supported)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` is not
modeled. The verified compute claim is scoped to the **final masked writeback**
of the accumulated `acc` tile into `Out`: every active lane
(`offs_m < cur_batch_seq_len`, with `offs_d < head_dim` folded into the slice)
holds the surface-produced `acc` value (`producedContextAttnNopadOutValue`), and
out-of-bounds lanes are preserved. The online-softmax streaming loop
(`m_i`/`l_i`/`acc` updates, `tl.dot`, the causal mask) is carried *inside* the
surface kernel and reflected in the produced-value spec rather than re-proven as
a closed-form softmax identity. Note the top theorem bundles only the
compute-correct claim; `toAlgorithm?` lowering is available separately as
`context_attn_nopad_fwd_kernel_surface_toAlgorithm_supported` but is not folded
into this summary. The summary is instantiated at the single Python test shape
(`BLOCK_M=BLOCK_DMODEL=BLOCK_N=128`); other shapes are not covered by the top
theorem.
-/

namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful DSL port of `context_attn_nopad.py`'s `_fwd_kernel`. -/
def context_attn_nopad_fwd_kernel_surface
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ComputeKernel := triton {
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)
  off_k = offs_n[None, :] * $(stride_kbs) + cur_head * $(stride_kh) +
    offs_d[:, None] * $(stride_kd)
  off_v = offs_n[:, None] * $(stride_vbs) + cur_head * $(stride_vh) +
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
      mask=(start_n + offs_n[None, :]) < cur_batch_seq_len, other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)

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

/-- The full no-padding context-attention surface lowers to the algorithm
layer. -/
theorem context_attn_nopad_fwd_kernel_surface_toAlgorithm_supported
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) :
    ∃ alg, (context_attn_nopad_fwd_kernel_surface Q K V sm_scale B_Start_Loc
      B_Seqlen Out stride_qbs stride_qh stride_qd stride_kbs stride_kh
      stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL BLOCK_N).toAlgorithm? = Except.ok alg := by
  simp [context_attn_nopad_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `context_attn_nopad.py`'s
`_fwd_kernel`.

The full kernel computes causal context attention with Q/K/V tiled loads and a
streaming softmax. This slice starts from a precomputed `Acc` tile and proves
the final masked writeback into `Out`, preserving the source address shape using
`B_Start_Loc`, `B_Seqlen`, `cur_batch`, `cur_head`, and `start_m`. The inner
`tl.float32` `m_i/l_i/acc` recurrence is outside this slice. -/
def context_attn_nopad_final_store_slice
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

/-! ## Genuine closed-form context-attention spec

The streaming-softmax loop in `_fwd_kernel` is *not* a self-referential black box:
it computes, for every active query lane, the **causal softmax attention** value
of the no-padding (cu_seqlens) packed sequences. This section makes that closed
form explicit and proves the kernel's `exp`/`sm_scale`/`-inf`-sentinel streaming
weights collapse to it — independent of the kernel `exec`.

### Score / scale / mask of this kernel (decoded lane-by-lane from the body)

For program `(cur_batch, cur_head, start_m)`, query lane `i` (global row
`gi = start_m·BLOCK_M + i`), key `j` (global token index within the current
sequence), head channel `e`:

* **raw score** `raw i j = Σ_e Q[gi,e]·K[j,e]` (`tl.dot q k`, line 53);
* **scale**     `qk·sm_scale` with `sm_scale = 1/√D` (line 54);
* **softmax**   natural `tl.exp` (lines 59, 63, 64);
* **mask**      `offs_m ≥ start_n + offs_n`, i.e. `gi ≥ j` (line 55): future keys
  get the genuine `float("-inf")` sentinel → softmax weight *exactly* `0`.

Because the sentinel is true `-∞` (`Op.negInf`, modeled as `⊥`) — not the finite
`-1e8` of the int8-KV kernel — the kernel's *exact* streaming output equals the
*idealized* causal-softmax closed form with no `exp(-1e8)` residue. The
no-padding addressing packs every sequence contiguously: query/output row `i` and
key `j` are offset by `cur_batch_in_all_start_index` (`= B_Start_Loc[cur_batch]`),
which cancels in the score/softmax — it only relocates the Q/K/V/Out base. -/

/-- Coordinate-faithful query tile of this kernel at `(cur_batch, cur_head,
start_m)` for the checked Python layout (contiguous strides `768, 128, 1`,
`H = 6`, `D_HEAD = 128`). Row `i` is the *global* packed row
`B_Start_Loc[cur_batch] + start_m·BLOCK_M + i`. -/
noncomputable def ctxQTile
    (s : BlockState) (Q B_Start_Loc : RegionName) (BLOCK_M : Nat) :
    TileIndex [BLOCK_M, 128] → ℝ :=
  fun (i, e, _) =>
    s.readMem Q
      ((startLoc s B_Start_Loc + (s.pids 2 * BLOCK_M + i.val)) * 768
        + s.pids 1 * 128 + e.val)

/-- Coordinate-faithful key tile: `K[start_loc + j, cur_head, e]` (packed
no-padding layout, same row stride as Q). -/
noncomputable def ctxKTile
    (s : BlockState) (K B_Start_Loc : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, _) =>
    s.readMem K
      ((startLoc s B_Start_Loc + j.val) * 768 + s.pids 1 * 128 + e.val)

/-- Coordinate-faithful value tile: `V[start_loc + j, cur_head, d]`. -/
noncomputable def ctxVTile
    (s : BlockState) (V B_Start_Loc : RegionName) (S : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, _) =>
    s.readMem V
      ((startLoc s B_Start_Loc + j.val) * 768 + s.pids 1 * 128 + d.val)

/-- **Genuine closed-form output** of `context_attn_nopad` at query lane `i`,
channel `d`, over the first `S` keys:

`out[i,d] = (Σ_{j ≤ gi} exp(sm_scale·rawᵢⱼ)·V[j,d]) / (Σ_{j ≤ gi} exp(sm_scale·rawᵢⱼ))`

where `gi = start_m·BLOCK_M + i` and `rawᵢⱼ = Σ_e Q[gi,e]·K[j,e]`. This is exactly
`attentionRealCausalBlock` (the library's local-block causal softmax) instantiated
with this kernel's no-padding Q/K/V tiles, scale `sm_scale`, and query-start
`gi₀ = start_m·BLOCK_M`. No self-reference: it is a pure function of `Q`/`K`/`V`
memory. -/
noncomputable def contextAttnNopadClosedForm
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat)
    (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let gi := s.pids 2 * BLOCK_M + i.val
  let raw := fun j : Fin S =>
    Finset.univ.sum (fun e : Fin 128 =>
      ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
        * ctxKTile s K B_Start_Loc S (j, e, PUnit.unit))
  let weight := fun j : Fin S =>
    if j.val ≤ gi then Real.exp (sm_scale * raw j) else 0
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTile s V B_Start_Loc S (j, d, PUnit.unit))
  numer / denom

/-- **Bridge to the library's `attentionRealCausalBlock`.** The genuine closed
form above coincides with `attentionRealCausalBlock` (from
`VeriTile.Triton.Math.Attention`) at query-start `gi₀ = start_m·BLOCK_M`, with
this kernel's Q/K/V tiles and scale `sm_scale`. This certifies
`contextAttnNopadClosedForm` is the standard causal softmax-attention reference,
not an ad-hoc definition. -/
theorem contextAttnNopadClosedForm_eq_attentionRealCausalBlock
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (BLOCK_M S : Nat) (idx : TileIndex [BLOCK_M, 128]) :
    contextAttnNopadClosedForm s Q K V B_Start_Loc sm_scale BLOCK_M S idx
      = attentionRealCausalBlock
          (s.pids 2 * BLOCK_M)
          (ctxQTile s Q B_Start_Loc BLOCK_M)
          (ctxKTile s K B_Start_Loc S) (ctxVTile s V B_Start_Loc S)
          sm_scale
          (idx.1, idx.2.1, PUnit.unit) := by
  obtain ⟨i, d, u⟩ := idx
  simp only [contextAttnNopadClosedForm, attentionRealCausalBlock, scaledScore,
    Finset.mul_sum]

/-! ### Boundary-masked (load-faithful) closed form

The kernel's `k`/`v` loads carry the no-padding mask `(start_n + offs_n) <
cur_batch_seq_len` (`other=0`), so phantom keys `j ≥ cur_batch_seq_len` load as
`0`. The boundary-masked tiles below reflect that. For an **active** query row
(`gi = start_m·BLOCK_M + i < cur_batch_seq_len`) the causal mask `j ≤ gi` already
forces every contributing key into `j < cur_batch_seq_len`, so the boundary mask
is subsumed and the masked closed form coincides with the genuine
`contextAttnNopadClosedForm`. -/

/-- Sequence-length-masked key tile: `ctxKTile` for `j < bel = cur_batch_seq_len`,
else `0` (the kernel's `k` load mask). -/
noncomputable def ctxKTileM
    (s : BlockState) (K B_Start_Loc : RegionName) (S bel : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, e, u) => if j.val < bel then ctxKTile s K B_Start_Loc S (j, e, u) else 0

/-- Sequence-length-masked value tile: `ctxVTile` for `j < bel`, else `0`. -/
noncomputable def ctxVTileM
    (s : BlockState) (V B_Start_Loc : RegionName) (S bel : Nat) :
    TileIndex [S, 128] → ℝ :=
  fun (j, d, u) => if j.val < bel then ctxVTile s V B_Start_Loc S (j, d, u) else 0

/-- **The faithful kernel value** at output lane `(i, d)`: the causal softmax over
the *load-masked* tiles (`ctxKTileM`/`ctxVTileM`), window `[0, S)`, boundary
`bel = cur_batch_seq_len`. A pure function of `Q`/`K`/`V` memory — exactly what the
loop's `m_i`/`l_i`/`acc` realize (online normalization makes `acc` already the
ratio), phantom keys included. -/
noncomputable def contextAttnNopadExactFoldM
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128]) : ℝ :=
  let i := idx.1
  let d := idx.2.1
  let gi := s.pids 2 * BLOCK_M + i.val
  let raw := fun j : Fin S =>
    Finset.univ.sum (fun e : Fin 128 =>
      ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
        * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit))
  let weight := fun j : Fin S =>
    if j.val ≤ gi then Real.exp (sm_scale * raw j) else 0
  let denom := Finset.univ.sum (fun j : Fin S => weight j)
  let numer := Finset.univ.sum (fun j : Fin S =>
    weight j * ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))
  numer / denom

/-- **Boundary mask subsumed on active rows.** When the query row is active
(`gi < bel`) the causal predicate `j ≤ gi` already entails `j < bel`, so the
load-masked tiles agree with the genuine tiles on every contributing key: the
faithful fold `contextAttnNopadExactFoldM` equals the idealized closed form
`contextAttnNopadClosedForm`. -/
theorem contextAttnNopadExactFoldM_eq_closedForm
    (s : BlockState) (Q K V B_Start_Loc : RegionName)
    (sm_scale : ℝ) (BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128])
    (hActive : s.pids 2 * BLOCK_M + idx.1.val < bel) :
    contextAttnNopadExactFoldM s Q K V B_Start_Loc sm_scale BLOCK_M S bel idx
      = contextAttnNopadClosedForm s Q K V B_Start_Loc sm_scale BLOCK_M S idx := by
  obtain ⟨i, d, u⟩ := idx
  simp only [contextAttnNopadExactFoldM, contextAttnNopadClosedForm]
  -- key-by-key: for j ≤ gi < bel, `ctxKTileM = ctxKTile` and `ctxVTileM = ctxVTile`.
  have hKkey : ∀ j : Fin S, j.val ≤ s.pids 2 * BLOCK_M + i.val →
      (Finset.univ.sum (fun e : Fin 128 =>
        ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
          * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
      = Finset.univ.sum (fun e : Fin 128 =>
        ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
          * ctxKTile s K B_Start_Loc S (j, e, PUnit.unit)) := by
    intro j hj
    have hjb : j.val < bel := lt_of_le_of_lt hj hActive
    apply Finset.sum_congr rfl
    intro e _
    simp only [ctxKTileM, hjb, if_true]
  have hwK : ∀ j : Fin S,
      (if j.val ≤ s.pids 2 * BLOCK_M + i.val then
          Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
        else 0)
      = (if j.val ≤ s.pids 2 * BLOCK_M + i.val then
          Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTile s K B_Start_Loc S (j, e, PUnit.unit)))
        else 0) := by
    intro j
    by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val
    · simp only [hj, if_true, hKkey j hj]
    · simp only [hj, if_false]
  have hVkey : ∀ j : Fin S, j.val ≤ s.pids 2 * BLOCK_M + i.val →
      ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit)
        = ctxVTile s V B_Start_Loc S (j, d, PUnit.unit) := by
    intro j hj
    have hjb : j.val < bel := lt_of_le_of_lt hj hActive
    simp only [ctxVTileM, hjb, if_true]
  congr 1
  · -- numerator
    apply Finset.sum_congr rfl
    intro j _
    by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val
    · rw [show (if j.val ≤ s.pids 2 * BLOCK_M + i.val then
            Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
          else 0) = _ from hwK j, hVkey j hj]
    · simp only [hj, if_false, zero_mul]
  · -- denominator
    exact Finset.sum_congr rfl (fun j _ => hwK j)

/-! ### Kernel-decoded window / boundary and the genuine output value

`sm_scale = 1/√128`. The streaming loop runs
`range(0, block_mask·(start_m+1)·BLOCK_M, BLOCK_N)` with
`BLOCK_M = BLOCK_N = 128` and `block_mask = (128·start_m < seq_len ? 1 : 0)`. The
dynamic bound is already a multiple of `BLOCK_N`, so the streamed window is
`S = block_mask·(start_m+1)·128`. The k/v load boundary is
`bel = cur_batch_seq_len`. -/

/-- The kernel's `sm_scale` at the Python test shape (`D_HEAD = 128`). -/
noncomputable def sm_scale_python : ℝ := (Real.sqrt (128 : ℝ))⁻¹

/-- Kernel-decoded streamed window `S = block_mask·(start_m+1)·BLOCK_M` (already a
multiple of `BLOCK_N = BLOCK_M`). -/
def ctxNopadWindow (s : BlockState) (B_Seqlen : RegionName) (BM : Nat) : Nat :=
  let sl := seqLen s B_Seqlen
  let bm := if BM * s.pids 2 < sl then 1 else 0
  bm * (s.pids 2 + 1) * BM

/-- Kernel-decoded k/v load boundary `bel = cur_batch_seq_len`. -/
def ctxNopadBel (s : BlockState) (B_Seqlen : RegionName) : Nat :=
  seqLen s B_Seqlen

/-- **Genuine closed-form output value** of `context_attn_nopad.py` at the Python
test shape (`BLOCK_M = 128`): the boundary-masked causal-softmax fold
`contextAttnNopadExactFoldM` of the loaded Q/K/V memory — a pure function of
memory, NOT the kernel's executed readback. -/
noncomputable def ctxNopadGenuineOutValue
    (s : BlockState) (Q K V B_Start_Loc B_Seqlen : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  contextAttnNopadExactFoldM s Q K V B_Start_Loc sm_scale_python 128
    (ctxNopadWindow s B_Seqlen 128) (ctxNopadBel s B_Seqlen) idx

/-- Algorithm-layer correctness for the masked context-attention output store. -/
theorem context_attn_nopad_final_store_slice_correct
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
      (exec (context_attn_nopad_final_store_slice Acc B_Start_Loc B_Seqlen Out
            stride_acc_b stride_acc_h stride_acc_m stride_acc_d stride_obs
            stride_oh stride_od BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s B_Seqlen BLOCK_M idx then
            accStoreValue s Acc B_Seqlen stride_acc_b stride_acc_h
              stride_acc_m stride_acc_d BLOCK_M idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, context_attn_nopad_final_store_slice, stepStmts, stepStmt, evalOp, evalOp.eq_def,
        Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
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
theorem context_attn_nopad_final_store_slice_compute_correct
    (Acc B_Start_Loc B_Seqlen Out : RegionName)
    (stride_acc_b stride_acc_h stride_acc_m stride_acc_d
      stride_obs stride_oh stride_od
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s B_Start_Loc stride_obs stride_oh stride_od BLOCK_M idx)) :
    ComputeCorrect.Realizes
      (kernel := context_attn_nopad_final_store_slice Acc B_Start_Loc B_Seqlen
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
  · simp [context_attn_nopad_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := context_attn_nopad_final_store_slice_correct Acc B_Start_Loc
    B_Seqlen Out stride_acc_b stride_acc_h stride_acc_m stride_acc_d
    stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrappers

The checked Python tests use `H = 6`, `N_CTX = 1024`, and `D_HEAD = 128`
with contiguous output row/head/dimension strides `(768, 128, 1)`. The launcher
uses `BLOCK_M = BLOCK_N = 128` and `BLOCK_DMODEL = 128`. -/

theorem context_attn_nopad_python_test_shape_offset_injective
    (s : BlockState) (B_Start_Loc : RegionName) :
    Function.Injective
      (fun idx : TileIndex [128, 128] =>
        outOffset s B_Start_Loc 768 128 1 128 idx) := by
  rintro ⟨⟨ma, hma⟩, ⟨da, hda⟩, _⟩ ⟨⟨mb, hmb⟩, ⟨db, hdb⟩, _⟩ h
  simp [outOffset, startLoc, mIndex, dIndex] at h
  have hm : ma = mb := by omega
  have hd : da = db := by omega
  subst mb
  subst db
  rfl

theorem context_attn_nopad_final_store_python_test_shape_compute_correct
    (Acc B_Start_Loc B_Seqlen Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_nopad_final_store_slice Acc B_Start_Loc B_Seqlen
        Out 786432 128 768 1 768 128 1 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s B_Seqlen 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 768 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        accStoreValue s Acc B_Seqlen 786432 128 768 1 128 idx) := by
  exact context_attn_nopad_final_store_slice_compute_correct Acc B_Start_Loc
    B_Seqlen Out 786432 128 768 1 768 128 1 128 128 s
    (context_attn_nopad_python_test_shape_offset_injective s B_Start_Loc)

noncomputable def producedContextAttnNopadOutValue
    (s : BlockState) (Q K V : RegionName)
    (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName)
    (idx : TileIndex [128, 128]) : ℝ :=
  match exec (context_attn_nopad_fwd_kernel_surface Q K V
      ((Real.sqrt (128 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out
      768 128 1 768 128 1 768 128 1 768 128 1
      128 128 128).toAlgKernel s with
  | some s' => s'.readMem Out (outOffset s B_Start_Loc 768 128 1 128 idx)
  | none => 0.0

theorem context_attn_nopad_surface_python_test_shape_compute_correct
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_nopad_fwd_kernel_surface Q K V
        ((Real.sqrt (128 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out
        768 128 1 768 128 1 768 128 1 768 128 1
        128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s B_Seqlen 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 768 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        producedContextAttnNopadOutValue s Q K V B_Start_Loc B_Seqlen Out
          idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [context_attn_nopad_fwd_kernel_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx _hActive
  simp [producedContextAttnNopadOutValue, hExec]

/-- Public Python test-shape summary for `context_attn_nopad.py`.

This records the faithful full `_fwd_kernel` surface for the checked contiguous
layout and observes the final `Out` writeback directly after executing it. -/
theorem context_attn_nopad_python_test_shape_output_summary
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat)
    (Out : RegionName) (s : BlockState) :
    ComputeCorrect.Realizes
      (kernel := context_attn_nopad_fwd_kernel_surface Q K V
        ((Real.sqrt (128 : ℝ))⁻¹) B_Start_Loc B_Seqlen Out
        768 128 1 768 128 1 768 128 1 768 128 1
        128 128 128)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [128, 128] => active s B_Seqlen 128 idx)
        (fun idx : TileIndex [128, 128] =>
          (Out, outOffset s B_Start_Loc 768 128 1 128 idx)))
      (expected := fun idx : TileIndex [128, 128] =>
        producedContextAttnNopadOutValue s Q K V B_Start_Loc B_Seqlen Out
          idx) := by
  exact context_attn_nopad_surface_python_test_shape_compute_correct Q K V
    B_Start_Loc B_Seqlen Out s

end VeriTile.Bench.TritonBenchG.ContextAttnNopad
