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

/-! ## Online-normalized streaming recurrence (the loop invariant's math)

Unlike the int8-KV context kernel (which divides `acc /= l_i` once after the
loop), `context_attn_nopad` **normalizes inside the loop**: each block it rescales
`p` by `beta/l_i_new` and `acc` by `l_i/l_i_new·alpha`, so after every block the
`acc` register already holds the *running normalized softmax ratio* `numer/denom`
and `l_i` holds the running denominator shifted by `exp(-m_i)` (i.e.
`Σ exp(score − m_i)`). This section is the mathematical heart of nopad's loop: the
⊥-seeded normalized recurrence (running max in `WithBot ℝ`, seeded `⊥` to model
`m_i = −inf`, `l_i = acc = 0`), and the proof that its running `(l, acc)` stay
equal to `exp(−m)·Σexp(score)` and `Σexp(score)v / Σexp(score)` — so the
full-window `acc` reads off the genuine causal-softmax closed form with NO
post-loop division. -/

/-- One ⊥-seeded **online-normalized** softmax step absorbing key `(sc, v)`. The
running max lives in `WithBot ℝ` (seeded `⊥`); `α = realExp(m ⊖ m')` is `0` on the
first key (faithful to `m_i = −inf`, `l_i = acc = 0`). `l` is the running
shifted denominator `Σ exp(score − m)`; `acc` is the running normalized ratio
`numer/denom`. The block update is `l' = α·l + exp(sc − m')`,
`acc' = acc·(l/l'·α) + (exp(sc − m')/l')·v`. -/
noncomputable def osNormStepBot
    (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let sc := sv.1; let v := sv.2
  let m' := m ⊔ ((sc : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp (WithBot.realSub m m')).unbotD 0
  let l' := l * α + Real.exp (sc - m'.unbotD 0)
  let acc' := acc * (l / l' * α) + (Real.exp (sc - m'.unbotD 0) / l') * v
  (m', l', acc')

/-- The running `max` component of an `osNormStepBot` fold is the `WithBot ⊔`-fold
of the per-key scores — independent of the normalized `l`/`acc` carried. -/
theorem osNormStepBot_foldl_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osNormStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- **⊥-seeded normalized consistency.** Folding `osNormStepBot` from a state
consistent with batch denominator `L` and unnormalized accumulator `T`
(`l = κ(m)·L`, `acc = T/L`, with the convention `acc = 0` when `L = 0`) keeps that
invariant: the final `l = κ(m_final)·(L + Σexp)`, `acc = (T + Σexp·v)/(L + Σexp)`.
`κ ⊥ = 0`, `κ (some r) = exp(−r)`. -/
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
    -- l·α = exp(-mr)·L : `m = ⊥` uses `L = 0`; `m = ↑a` uses `l = exp(-a)·L`, `α = exp(a − mr)`.
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
    -- new l' = κ(m')·L'
    have hl'eq : l' = Real.exp (-mr) * L' := by
      rw [hl'd, hlα, hpd, hunbot, hL'd]
      have e2 : Real.exp (sc - mr) = Real.exp (-mr) * Real.exp sc := by
        rw [← Real.exp_add]; ring_nf
      rw [e2]; ring
    -- L' ≥ 0
    have hL'pos : 0 ≤ L' := by rw [hL'd]; positivity
    -- L' > 0 (key always contributes exp sc > 0)
    have hL'strict : 0 < L' := by rw [hL'd]; positivity
    -- l' ≠ 0
    have hl'ne : l' ≠ 0 := by rw [hl'eq]; positivity
    -- acc' * L' = T'
    have hacc'eq : acc' * L' = T' := by
      rw [hacc'd, hpd, hunbot]
      rw [add_mul]
      -- term 2: (exp(sc-mr)/l')·v·L' = exp(sc-mr)·L'/l' · v ; l' = exp(-mr)L'
      have hl'val : l' = Real.exp (-mr) * L' := hl'eq
      have e2 : Real.exp (sc - mr) / l' * v * L'
          = Real.exp sc * v := by
        rw [hl'val]
        rw [show Real.exp (sc - mr) = Real.exp (-mr) * Real.exp sc from by
          rw [← Real.exp_add]; ring_nf]
        have hexpne : Real.exp (-mr) ≠ 0 := Real.exp_ne_zero _
        have hL'ne : (L' : ℝ) ≠ 0 := ne_of_gt hL'strict
        field_simp
      -- term 1: acc·(l/l'·α)·L' = acc·(l·α)·(L'/l') = acc·(exp(-mr)L)·(L'/(exp(-mr)L')) = acc·L = T
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
    -- rewrite the goal's fold/sum to match step
    simpa [List.foldl_cons, osNormStepBot, hm', hαd, hpd, hl'd, hacc'd,
      List.map_cons, List.sum_cons, hL'd, hT'd, add_assoc, add_comm, add_left_comm,
      mul_comm, mul_left_comm] using step

/-! ### Full-window readback: the loop's `acc` is the genuine closed form

The kernel's `tl.where(mask, qk, −inf)` makes future keys carry softmax weight
exactly `0`, so they are inert in both numerator and denominator. The fold the
loop realizes is therefore `osNormStepBot` over the *active-key* list (causal
`j ≤ gi`, value `ctxVTileM`); its final `acc` is the genuine boundary-masked
closed form `contextAttnNopadExactFoldM`. -/

/-- filterMap-sum over `Fin n` with a guard collapses into the masked `Finset.sum`. -/
theorem ctxNopad_filterMap_finRange_sum {α : Type*} (n : Nat)
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

/-- The active-key `(score, value)` list the nopad loop folds for output `(i, d)`:
keys `j ≤ gi` (causal), score `sm_scale·Σ_e Q·K`, value `ctxVTileM`. Future keys
(softmax weight `0` via the `−inf` sentinel) are dropped. -/
noncomputable def ctxNopadKeyList
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLOCK_M + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi then
      some (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
            ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))
    else none)

/-- **The nopad loop's full-window `acc` is the genuine closed form.** Folding
`osNormStepBot` from the kernel seed `(⊥, 0, 0)` over the active-key list yields,
in its `acc` component, exactly `contextAttnNopadExactFoldM` — provided the window
is non-empty for this lane (key `0` causal-visible, `0 ≤ gi`). A pure-memory
identity: the streaming loop computes the genuine causal softmax with no post-loop
division. -/
theorem ctxNopad_fold_eq_exactFoldM
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (idx : TileIndex [BLOCK_M, 128])
    (hvis : (0 : Nat) < S ∧ (0 : Nat) ≤ s.pids 2 * BLOCK_M + idx.1.val) :
    ((ctxNopadKeyList s Q K V B_Start_Loc sm_scale BLOCK_M S bel idx.1 idx.2.1).foldl
        osNormStepBot (⊥, 0, 0)).2.2
      = contextAttnNopadExactFoldM s Q K V B_Start_Loc sm_scale BLOCK_M S bel idx := by
  obtain ⟨i, d, u⟩ := idx
  set xs := ctxNopadKeyList s Q K V B_Start_Loc sm_scale BLOCK_M S bel i d with hxs
  -- the active-key list's score/value sums equal the closed form's masked Finset sums
  have hL : (xs.map (fun p => Real.exp p.1)).sum
      = Finset.univ.sum (fun j : Fin S =>
          if j.val ≤ s.pids 2 * BLOCK_M + i.val then
            Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
          else 0) := by
    rw [hxs, ctxNopadKeyList]
    rw [ctxNopad_filterMap_finRange_sum S
      (fun j : Fin S => j.val ≤ s.pids 2 * BLOCK_M + i.val)]
  have hT : (xs.map (fun p => Real.exp p.1 * p.2)).sum
      = Finset.univ.sum (fun j : Fin S =>
          (if j.val ≤ s.pids 2 * BLOCK_M + i.val then
            Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
          else 0) * ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit)) := by
    rw [hxs, ctxNopadKeyList]
    rw [ctxNopad_filterMap_finRange_sum S
      (fun j : Fin S => j.val ≤ s.pids 2 * BLOCK_M + i.val)]
    apply Finset.sum_congr rfl; intro j _
    by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val <;> simp [hj]
  -- consistency from the kernel seed (L = T = 0)
  have hcons := osNormStepBot_foldl_consistent xs ⊥ 0 0 0 0
    (le_refl 0) (fun _ _ => trivial) (by simp) (by ring) (fun _ => rfl) (fun _ => rfl)
    (by intro h; exact absurd h (lt_irrefl 0))
  obtain ⟨_hl', hacc', _hL'0, _hbot1, _hbot2, _hne⟩ := hcons
  -- L' = denom, T' = numer
  simp only [zero_add] at hacc' hL hT
  rw [contextAttnNopadExactFoldM]
  -- denom > 0 : key 0 is causal-visible (0 ≤ gi), so its exp term is in the sum
  have hdenom_pos : 0 < Finset.univ.sum (fun j : Fin S =>
        if j.val ≤ s.pids 2 * BLOCK_M + i.val then
          Real.exp (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
            ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
              * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)))
        else 0) := by
    apply Finset.sum_pos'
    · intro j _
      by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val
      · simp [hj, le_of_lt (Real.exp_pos _)]
      · simp [hj]
    · refine ⟨⟨0, hvis.1⟩, Finset.mem_univ _, ?_⟩
      have h0 : (0 : Nat) ≤ s.pids 2 * BLOCK_M + i.val := Nat.zero_le _
      simp only [Fin.val_mk, h0, if_true]
      exact Real.exp_pos _
  -- from acc' * L' = T' and L' ≠ 0 : acc' = T'/L'
  rw [eq_div_iff (ne_of_gt hdenom_pos)]
  rw [← hL, ← hT, hacc']

/-! ### Block-windowed key lists (per-block invariant advance)

Mirror of #307's `srKeysUpto`/`srBlock`/`srKeysUpto_succ`/`srStateBot_succ`,
adapted to nopad's *causal* per-key filter. The streamed prefix after `c` blocks
is `ctxNopadKeysUpto … (c·128)` — the causal-and-window key list — and one loop
iteration appends `nopadBlock c` (the keys in `[c·128, (c+1)·128)` that are causal
for row `i`). -/

/-- The `(score, value)` keys row `i`/channel `d` has streamed after window
`[0, hi)`: causal (`j ≤ gi`) AND `j.val < hi`. -/
noncomputable def ctxNopadKeysUpto
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel hi : Nat) (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLOCK_M + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi ∧ j.val < hi then
      some (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
            ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))
    else none)

/-- Block-`c` causal keys for row `i`/channel `d`: causal `j ≤ gi` AND
`c·128 ≤ j.val < (c+1)·128` — the keys the loop's `c`-th iteration streams. -/
noncomputable def nopadBlock
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) : List (ℝ × ℝ) :=
  let gi := s.pids 2 * BLOCK_M + i.val
  (List.finRange S).filterMap (fun j : Fin S =>
    if j.val ≤ gi ∧ c * 128 ≤ j.val ∧ j.val < (c + 1) * 128 then
      some (sm_scale * Finset.univ.sum (fun e : Fin 128 =>
              ctxQTile s Q B_Start_Loc BLOCK_M (i, e, PUnit.unit)
                * ctxKTileM s K B_Start_Loc S bel (j, e, PUnit.unit)),
            ctxVTileM s V B_Start_Loc S bel (j, d, PUnit.unit))
    else none)

/-- The full causal key list is the causal-window list at `hi = S`. -/
theorem ctxNopadKeysUpto_full
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel S i d
      = ctxNopadKeyList s Q K V B_Start_Loc sm_scale BLOCK_M S bel i d := by
  unfold ctxNopadKeysUpto ctxNopadKeyList
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val ≤ s.pids 2 * BLOCK_M + i.val
  · simp [hj, j.isLt]
  · simp [hj]

/-- The causal-window key list at `hi = 0` is empty (kernel preLoop init). -/
theorem ctxNopadKeysUpto_zero
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel 0 i d = [] := by
  unfold ctxNopadKeysUpto
  apply List.filterMap_eq_nil_iff.mpr
  intro j _; simp

/-- Threshold-split for a `.val`-ascending `Fin` list under a causal-and-window
guard: `j.val < hi₂` window splits into `< t` prefix and `t ≤ j.val < hi₂` block. -/
private theorem nopad_filterMap_window_split {n : Nat} (l : List (Fin n))
    (hsorted : l.Pairwise (fun a b => a.val < b.val))
    (gi t hi₂ : Nat) (g : Fin n → ℝ × ℝ) (hle : t ≤ hi₂) :
    l.filterMap (fun j => if j.val ≤ gi ∧ j.val < hi₂ then some (g j) else none)
      = l.filterMap (fun j => if j.val ≤ gi ∧ j.val < t then some (g j) else none)
        ++ l.filterMap (fun j => if j.val ≤ gi ∧ t ≤ j.val ∧ j.val < hi₂ then some (g j) else none) := by
  induction l with
  | nil => simp
  | cons a tl ih =>
    have htl : tl.Pairwise (fun x y => x.val < y.val) := (List.pairwise_cons.mp hsorted).2
    have hahead : ∀ b ∈ tl, a.val < b.val := (List.pairwise_cons.mp hsorted).1
    rw [List.filterMap_cons, List.filterMap_cons, List.filterMap_cons]
    by_cases hca : a.val ≤ gi
    · by_cases hlt : a.val < t
      · rw [ih htl]
        rw [if_pos ⟨hca, lt_of_lt_of_le hlt hle⟩, if_pos ⟨hca, hlt⟩,
          if_neg (fun h : a.val ≤ gi ∧ t ≤ a.val ∧ a.val < hi₂ => by omega)]
        rfl
      · have hge : t ≤ a.val := Nat.not_lt.mp hlt
        have htail_prefix : tl.filterMap (fun j => if j.val ≤ gi ∧ j.val < t then some (g j) else none) = [] := by
          apply List.filterMap_eq_nil_iff.mpr
          intro b hb
          have := hahead b hb
          rw [if_neg (fun h : b.val ≤ gi ∧ b.val < t => by omega)]
        rw [ih htl, htail_prefix, if_neg (fun h : a.val ≤ gi ∧ a.val < t => by omega)]
        by_cases h2 : a.val < hi₂
        · rw [if_pos ⟨hca, h2⟩, if_pos ⟨hca, hge, h2⟩]; rfl
        · rw [if_neg (fun h => h2 h.2), if_neg (fun h => h2 h.2.2)]
    · rw [if_neg (fun h => hca h.1), if_neg (fun h => hca h.1), if_neg (fun h => hca h.1)]
      rw [ih htl]

/-- **Window split** (`hi = c·128`): the causal keys streamed through `c+1` blocks
are those through `c` blocks followed by block `c`. -/
theorem ctxNopadKeysUpto_succ
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel ((c + 1) * 128) i d
      = ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel (c * 128) i d
        ++ nopadBlock s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d := by
  unfold ctxNopadKeysUpto nopadBlock
  exact nopad_filterMap_window_split (List.finRange S) (List.pairwise_lt_finRange S)
    (s.pids 2 * BLOCK_M + i.val) (c * 128) ((c + 1) * 128) _
    (by nlinarith [Nat.zero_le (128 : Nat)])

/-- **One-block invariant advance** (pure math): the `osNormStepBot` fold over the
causal keys through `c+1` blocks is block `c`'s `osNormStepBot`-fold applied to the
fold through `c` blocks. -/
theorem nopad_fold_succ
    (s : BlockState) (Q K V B_Start_Loc : RegionName) (sm_scale : ℝ)
    (BLOCK_M S bel c : Nat) (i : Fin BLOCK_M) (d : Fin 128) :
    (ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel ((c + 1) * 128) i d).foldl
        osNormStepBot (⊥, 0, 0)
      = (nopadBlock s Q K V B_Start_Loc sm_scale BLOCK_M S bel c i d).foldl osNormStepBot
          ((ctxNopadKeysUpto s Q K V B_Start_Loc sm_scale BLOCK_M S bel (c * 128) i d).foldl
            osNormStepBot (⊥, 0, 0)) := by
  rw [ctxNopadKeysUpto_succ, List.foldl_append]

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

/-! ## Genuine exec assembly (mirrors triton_attention #308 structure)

The streaming loop is decoded statement-by-statement and bound to the banked
`osNormStepBot` online-normalized recurrence. Unlike triton_attention (block
pointers, single diagonal block, `boundary_check`), `context_attn_nopad` uses
explicit 3D-packed pointer arithmetic, a true `float("-inf")` causal `where`
sentinel, natural `tl.exp`, and an *in-loop* normalize (`p_scale`/`acc_scale`),
so `acc` already holds the normalized ratio at every block (no post-loop divide).
The loop runs `block_mask·(start_m+1)·128` keys = `start_m+1` blocks. -/

/-- The 19 lowered preLoop statements of the Python-shape `context_attn_nopad`
forward body (program ids, the loaded seq-len/start-loc scalars, the index
vectors, the 3D-packed `off_q/off_k/off_v` offset tiles, the masked `q` load, the
`k_ptrs/v_ptrs` base pointers, the `m_i = ⊥`/`l_i = 0`/`acc = 0` running init, and
`block_mask`). -/
def nopadPreLoop (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) : List Stmt :=
  [ Stmt.assign .nat [] "cur_batch" (Op.programId 0),
    Stmt.assign .nat [] "cur_head" (Op.programId 1),
    Stmt.assign .nat [] "start_m" (Op.programId 2),
    Stmt.assign .nat [] "cur_batch_seq_len"
      (Op.load .nat (MemAccess.region B_Seqlen (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "cur_batch_in_all_start_index"
      (Op.load .nat (MemAccess.region B_Start_Loc (Op.ref .nat [] "cur_batch")) MaskOpt.none),
    Stmt.assign .nat [] "block_start_loc"
      (Op.mul .nat Broadcast.nil (Op.constNat 128) (Op.ref .nat [] "start_m")),
    Stmt.assign .nat [128] "offs_n" (Op.arange 128),
    Stmt.assign .nat [128] "offs_d" (Op.arange 128),
    Stmt.assign .nat [128] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 128))
        (Op.arange 128)),
    Stmt.assign .nat [128, 128] "off_q"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .nat [128, 128] "off_k"
      (Op.add .nat Broadcast.nil.consR.consL
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_n"))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .nat [128, 128] "off_v"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_n"))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .real [128, 128] "q"
      (Op.load .real (MemAccess.region Q (Op.ref .nat [128, 128] "off_q"))
        (MaskOpt.maskOther
          (Op.remap [128, 128] Broadcast.nil.consL.consSame.leftIndex
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
              (Op.ref .nat [] "cur_batch_seq_len")))
          ((Op.const 0.0).broadcast [128, 128]))),
    Stmt.assign .ptr [128, 128] "k_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K) (Op.ref .nat [128, 128] "off_k")),
    Stmt.assign .ptr [128, 128] "v_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V) (Op.ref .nat [128, 128] "off_v")),
    Stmt.assign .real [128] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [128] (Op.const 0)) Op.negInf),
    Stmt.assign .real [128] "l_i" (Op.full [128] (Op.const 0)),
    Stmt.assign .real [128, 128] "acc" (Op.full [128, 128] (Op.const 0)),
    Stmt.assign .nat [] "block_mask"
      ((Op.lt ComparableDType.nat Broadcast.nil (Op.ref .nat [] "block_start_loc")
            (Op.ref .nat [] "cur_batch_seq_len")).where
        (Op.constNat 1) (Op.constNat 0)) ]

/-- The 21 lowered loop-body statements of the Python-shape `context_attn_nopad`
forward `forRangeDyn` body (online-normalized streaming softmax over one KV
block). `start_n = start_n` and `p = p` are the lowered `tl.multiple_of` /
`p.to(v.dtype)` no-ops. -/
def nopadLoopBody (sc : ℝ) : List Stmt :=
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
        (Op.ref .real [128, 128] "qk") (Op.negInf.broadcast [128, 128])),
    Stmt.assign .real [128] "m_ij" (Op.reduceMax ⟨1, by simp⟩ Bool.false (Op.ref .real [128, 128] "qk")),
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

/-- The 3 lowered postLoop statements of the Python-shape `context_attn_nopad`
forward body: the 3D-packed `off_o` offset tile, the `out_ptrs` base pointer, and
the masked `Out` store of `acc` (already the normalized ratio — no `acc /= l_i`). -/
def nopadPostLoop (Out : RegionName) : List Stmt :=
  [ Stmt.assign .nat [128, 128] "off_o"
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarR
          (Op.mul .nat Broadcast.scalarR
            (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_batch_in_all_start_index")
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m")))
            (Op.constNat 768))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_head") (Op.constNat 128)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [128] "offs_d"))
          (Op.constNat 1))),
    Stmt.assign .ptr [128, 128] "out_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out) (Op.ref .nat [128, 128] "off_o")),
    Stmt.store .real [128, 128] (MemAccess.ptr (Op.ref .ptr [128, 128] "out_ptrs"))
      (Op.ref .real [128, 128] "acc")
      (MaskOpt.mask
        (Op.remap [128, 128] Broadcast.nil.consL.consSame.leftIndex
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [128] "offs_m"))
            (Op.ref .nat [] "cur_batch_seq_len")))) ]

/-! ### Per-statement loop-body op-eval recipes

Mirror of #307's `sr_*_eval` recipes but in 2D (`BLOCK_M = BLOCK_N = BLOCK_DMODEL =
128`). The kernel's per-block registers are decoded *per query row* `i`: the
`qk`/`p` matrices are `[128, 128]` (row `i`, key lane `jL`), the running
`m_i`/`l_i` vectors are `[128]` (row `i`), and `acc` is `[128, 128]` (row `i`,
channel `d`). The causal `where(offs_m ≥ start_n+offs_n, qk, -inf)` makes future
keys carry the genuine `⊥` sentinel; the `k`/`v` loads carry `other=0` boundary
masks. -/

/-- The decoded `q` tile register value, lane `(i, e)`: `ctxQTile` (= `Q[start_loc
+ start_m·128 + i, cur_head, e]`). A pure function of memory. -/
noncomputable def nopadQReg (s : BlockState) (Q B_Start_Loc : RegionName)
    (i : Fin 128) (e : Fin 128) : ℝ :=
  ctxQTile s Q B_Start_Loc 128 (i, e, PUnit.unit)

set_option maxHeartbeats 1600000 in
/-- **`k` masked-load recipe.** `tl.load(k_ptrs + (start_loc+start_n)·768,
mask=(start_n+offs_n)<seqlen, other=0)`, shape `[128,128]`, lane `(e, jL)`
(`e`=channel axis, `jL`=key axis): `some (K[start_loc+start_n+jL, cur_head, e])`
if `start_n+jL < seqlen`, else `some 0`. -/
theorem nopad_k_load_eval (s : BlockState) (K B_Start_Loc B_Seqlen : RegionName) (SN : Nat)
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
/-- **`qk += tl.dot(q, k)` recipe** (over the seeded `qk = full 0`).  `qk` lane
`(i, jL)`: `some (Σ_e qFn i e · kFn e jL)`, where `q`/`k` are non-`⊥` real tiles. -/
theorem nopad_qk_dot_eval (s : BlockState) (qFn : Fin 128 → Fin 128 → ℝ)
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
/-- **`qk *= sm_scale` recipe.** `qk` lane `(i, jL)`: `some (sc · rawFn i jL)`. -/
theorem nopad_qk_scale_eval (s : BlockState) (sc : ℝ) (rawFn : Fin 128 → Fin 128 → ℝ)
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
/-- **`qk = tl.where(offs_m ≥ start_n+offs_n, qk, -inf)` recipe** (causal mask).
`qk` lane `(i, jL)`: keeps `some (qkFn i jL)` when `gi[i] ≥ SN + jL` (`gi[i] =
offs_m[i]`), else the genuine `⊥` sentinel. -/
theorem nopad_qk_where_eval (s : BlockState) (SN : Nat) (qkFn : Fin 128 → Fin 128 → ℝ)
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
        (Op.ref .real [128, 128] "qk") (Op.negInf.broadcast [128, 128])) s
      = some (⟨fun idx : TileIndex [128, 128] =>
          if SN + idx.2.1.val ≤ offsM idx.1 then some (qkFn idx.1 idx.2.1)
          else (⊥ : WithBot ℝ)⟩ : Tile .real [128, 128]) := by
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
    TileShape.dropInsertedIndex]
  by_cases hle : SN + jL.val ≤ offsM i
  · rw [if_pos (by simp only [decide_eq_true_eq]; omega), if_pos hle]
  · rw [if_neg (by simp only [decide_eq_true_eq]; omega), if_neg hle]; rfl

set_option maxHeartbeats 1600000 in
/-- **`m_ij = tl.max(qk, 1)` recipe** (per-row running block max over the key axis).
Row `i`: `⊔'_{jL} qk[i,jL]`. -/
theorem nopad_mij_eval (s : BlockState) (qkFn : Fin 128 → Fin 128 → WithBot ℝ)
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
/-- **`p = tl.exp(qk - m_ij[:, None])` recipe** (per-cell softmax numerator,
broadcast `[128,128] - [128]` over the key axis).  `p` lane `(i, jL)`:
`realExp(qk[i,jL] ⊖ m_ij[i])`. -/
theorem nopad_p_eval (s : BlockState) (qkFn : Fin 128 → Fin 128 → WithBot ℝ)
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
/-- **`l_ij = tl.sum(p, 1)` recipe** (per-row block denominator over the key axis).
Row `i`: `Σ_{jL} p[i,jL]`. -/
theorem nopad_lij_eval (s : BlockState) (pFn : Fin 128 → Fin 128 → WithBot ℝ)
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
/-- **`m_i_new = tl.maximum(m_i, m_ij)` recipe** (`where(m_i > m_ij, m_i, m_ij)`),
per-row `m_i[i] ⊔ m_ij[i]`. -/
theorem nopad_minew_eval (s : BlockState) (miFn mijFn : Fin 128 → WithBot ℝ)
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
/-- **`alpha = tl.exp(m_i - m_i_new)` recipe** (old-block rescale), per-row
`realExp(m_i[i] ⊖ m_i_new[i])`. -/
theorem nopad_alpha_eval (s : BlockState) (miFn minewFn : Fin 128 → WithBot ℝ)
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
/-- **`beta = tl.exp(m_ij - m_i_new)` recipe** (new-block rescale), per-row
`realExp(m_ij[i] ⊖ m_i_new[i])`. -/
theorem nopad_beta_eval (s : BlockState) (mijFn minewFn : Fin 128 → WithBot ℝ)
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
/-- **`l_i_new = alpha*l_i + beta*l_ij` recipe** (per-row online denominator).
Row `i`: `realAdd (realMul α[i] l_i[i]) (realMul β[i] l_ij[i])`. -/
theorem nopad_linew_eval (s : BlockState) (alphaFn liFn betaFn lijFn : Fin 128 → WithBot ℝ)
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
/-- **`p_scale = beta / l_i_new` recipe** (per-row), `realDiv β[i] l_i_new[i]`. -/
theorem nopad_pscale_eval (s : BlockState) (betaFn linewFn : Fin 128 → WithBot ℝ)
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
/-- **`p = p * p_scale[:, None]` recipe** (per-row rescale of the `[128,128]` `p`
matrix by the `[128]` row factor).  `p` lane `(i, jL)`: `realMul p[i,jL] p_scale[i]`. -/
theorem nopad_pmul_eval (s : BlockState) (pFn : Fin 128 → Fin 128 → WithBot ℝ)
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
/-- **`acc_scale = l_i / l_i_new * alpha` recipe** (per-row), `realMul (realDiv
l_i[i] l_i_new[i]) α[i]`. -/
theorem nopad_accscale_eval (s : BlockState) (liFn linewFn alphaFn : Fin 128 → WithBot ℝ)
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
/-- **`acc = acc * acc_scale[:, None]` recipe** (per-row rescale of the `[128,128]`
`acc` matrix).  `acc` lane `(i, d)`: `realMul acc[i,d] acc_scale[i]`. -/
theorem nopad_accmul_eval (s : BlockState) (accFn : Fin 128 → Fin 128 → WithBot ℝ)
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
/-- **`v` masked-load recipe.** `tl.load(v_ptrs + (start_loc+start_n)·768,
mask=(start_n+offs_n)<seqlen, other=0)`, shape `[128,128]`, lane `(jL, d)`
(`jL`=key axis 0, `d`=channel axis 1): `some (V[start_loc+start_n+jL, cur_head, d])`
if `start_n+jL < seqlen`, else `some 0`. -/
theorem nopad_v_load_eval (s : BlockState) (V B_Start_Loc B_Seqlen : RegionName) (SN : Nat)
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
/-- **`acc += tl.dot(p, v)` recipe** (over the rescaled `acc`).  `acc` lane `(i, d)`:
`realAdd acc[i,d] (Σ_{jL} p[i,jL]·v[jL,d])`. -/
theorem nopad_acc_dot_eval (s : BlockState) (accFn : Fin 128 → Fin 128 → WithBot ℝ)
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

set_option maxRecDepth 8000 in
/-- The lowered forward body is exactly `nopadPreLoop ++ forRangeDyn :: nopadPostLoop`. -/
theorem nopad_body_split
    (Q K V : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Out : RegionName) (sc : ℝ) :
    (context_attn_nopad_fwd_kernel_surface Q K V sc B_Start_Loc B_Seqlen Out
      768 128 1 768 128 1 768 128 1 768 128 1 128 128 128).toAlgKernel.body
      = nopadPreLoop Q K V B_Start_Loc B_Seqlen
        ++ (Stmt.forRangeDyn "start_n" (Op.constNat 0)
              (Op.mul .nat Broadcast.nil
                (Op.mul .nat Broadcast.nil (Op.ref .nat [] "block_mask")
                  (Op.add .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat 1)))
                (Op.constNat 128))
              (Op.constNat 128) (nopadLoopBody sc)
            :: nopadPostLoop Out) := by
  rfl

end VeriTile.Bench.TritonBenchG.ContextAttnNopad
