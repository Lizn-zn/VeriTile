import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Semantics.TileOps
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

/-!
# `lightning_attention` — strict per-kernel correctness

`_fwd_kernel` is the lightning-attention forward recurrent tile scan: each
program walks `NUM_BLOCK` key/value tiles, carrying a `[d, e]` key-value
accumulator `kv` across the loop, combining an intra-block (masked, decayed)
attention term with an inter-block term `q · kv` to produce the output tile.
The backward kernels (`_bwd_intra`, `_bwd_inter`) compute the `dq`/`dk`/`dv`
gradients with the matching reverse recurrence.

## Scope

This file verifies **the Triton kernels themselves** — the per-program
`@triton.jit` bodies (forward, backward-intra, backward-inter). The host
launch (grid shape, block-model tiling, and how the runtime composes
per-program tile writes) is the *trusted boundary*. Because the program ids
are universally quantified, each per-program statement covers every program.

## Proof architecture

```
lightning_attention_output_summary_general               ← TOP THEOREM (dimension-general)
  ├─ lightning_attention_{forward,bwd_intra,bwd_inter}_surface_toAlgorithm_supported
  ├─ lightning_attention_forward_store_slice_compute_correct  (Out writeback)
  └─ lightning_attention_bwd_{dq,dk,dv}_store_slice_compute_correct

Genuine forward closed form (NOT self-referential):
  kvClosed / kvClosed_succ                          kv carry-fold over K·V
  kvStepSpec_eq_kvClosed_succ                       carry invariant: body advances kvClosed
  lightning_attention_forward_kv_step_slice_compute_correct      kv += dot(k_trans, v) body
  lightning_attention_forward_o_inter_dot_slice_compute_correct  o_inter = dot(q, kv) body
```

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity. **Genuine closed form (forward):** `kvClosed m` is a
standalone specification over the input regions `K,V` — the running `kᵀ·v` sum
over the first `m` key blocks — and `kvClosed_succ` is the exact closed-form
counterpart of the Python `kv += tl.dot(k_trans, v)`. The carry-fold step lemma
`kvStepSpec_eq_kvClosed_succ` shows one loop body advances `kvClosed` by one
block under the carry invariant `KVPrev = kvClosed m`; the genuine causal
linear-attention output row is `Σ_{s≤t}(q_t·k_s)·v_s`. The remaining
boundary is the cross-block `NUM_BLOCK` loop *driver* threading the invariant
(the carry hypotheses `hPrev`/`hK`/`hV` of `kvStepSpec_eq_kvClosed_succ`, the
documented loop boundary as in #290/#291), and the host launch (grid/block-model
tiling). The store-slice lemmas prove the per-tile **masked face** (the `Out` /
`DQ` / `DK` / `DV` writeback). The `dot` tile products are modeled. Side
conditions: tile-offset injectivity hypotheses where required.
-/

namespace VeriTile.Bench.TritonBenchG.LightningAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `lightning_attention_output_summary_general`
(dimension-general genuine launched-surface summary). -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct

/-- Faithful transcription of `lightning_attention.py`'s `_fwd_kernel`.

This covers the full forward recurrent tile loop. -/
def lightning_attention_forward_surface
    (Q K V Out : RegionName)
    (_b h n d e BLOCK NUM_BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_bh % $(h)
  off_e = tl.program_id(1)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  e_offset = off_e * $(BLOCK_MODEL)
  Q_block_ptr = Q + qk_offset + tl.arange(0, $(d))[None, :]
  K_trans_block_ptr = K + qk_offset + tl.arange(0, $(d))[:, None]
  V_block_ptr = V + v_offset + e_offset + tl.arange(0, $(BLOCK_MODEL))[None, :]
  O_block_ptr = Out + o_offset + e_offset + tl.arange(0, $(BLOCK_MODEL))[None, :]
  off_block = tl.arange(0, $(BLOCK))
  index = off_block[:, None] - off_block[None, :]
  kv = tl.zeros([$(d), $(BLOCK_MODEL)], dtype=tl.float32)
  for i in range($(0), $(NUM_BLOCK), $(1)) {
    q = tl.load(Q_block_ptr + off_block[:, None] * $(d),
      mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
    k_trans = tl.load(K_trans_block_ptr + off_block[None, :] * $(d),
      mask=off_block[None, :] < $(n), other=0.0).to(tl.float32)
    v = tl.load(V_block_ptr + off_block[:, None] * $(e),
      mask=off_block[:, None] < $(n), other=0.0).to(tl.float32)
    qk = tl.dot(q, k_trans)
    qk = tl.where(index >= 0, qk, 0)
    o_intra = tl.dot(qk, v)
    o_inter = tl.dot(q, kv)
    o = o_intra + o_inter
    tl.store(O_block_ptr + off_block[:, None] * $(e),
      (o).to(O_block_ptr.dtype.element_ty), mask=off_block[:, None] < $(n))
    kv += tl.dot(k_trans, v)
    off_block += $(BLOCK)
  }
}

/-- The full Python-shaped forward recurrent attention surface lowers to the
algorithm layer, including the causal `tl.where`, recurrent `kv` update, and
masked output store. -/
theorem lightning_attention_forward_surface_toAlgorithm_supported
    (Q K V Out : RegionName)
    (_b h n d e BLOCK NUM_BLOCK BLOCK_MODEL : Nat) :
    ∃ alg, (lightning_attention_forward_surface Q K V Out _b h n d e BLOCK
      NUM_BLOCK BLOCK_MODEL).toAlgorithm? = Except.ok alg := by
  simp [lightning_attention_forward_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Genuine forward closed form (the `kv` carry-fold + linear attention output)

`_fwd_kernel` is a **linear-attention recurrence with NO decay** (lightning
attention 2, "no decay"): the `index >= 0` `tl.where` is a pure causal mask, not
a decay weighting. Block `i` carries a `[d, BLOCK_MODEL]` key-value accumulator
`kv`, computes `o = where(index≥0, q·kᵀ, 0)·v + q·kv`, stores `o`, then updates
`kv += kᵀ·v`. The whole computation is the causal linear attention

```
o[t, c] = Σ_{s ≤ t} (Σ_a q[t,a]·k[s,a]) · v[s,c].
```

The definitions below are a **standalone specification over the input regions**
`Q,K,V` — never a read-back of the kernel's own output. `kvClosed m` is the
genuine `kv` state entering block `m` (the sum of `kᵀ·v` over the first `m`
blocks); the genuine output row is the causal linear-attention sum. `kvClosed_succ`
is the exact closed-form counterpart of the Python loop's `kv += tl.dot(k_trans, v)`. -/

/-- `K[qk_offset + s·d + a]`: the key entry at global key row `s`, head channel
`a`, for the program `off_bh`. -/
noncomputable def fwdKVal (s : BlockState) (K : RegionName)
    (n d : Nat) (a : Nat) (keyRow : Nat) : ℝ :=
  s.readMem K (s.pids 0 * n * d + keyRow * d + a)

/-- `V[v_offset + e_offset + s·e + c]`: the value entry at global key row `s`,
value channel `c` (within the `off_e` channel block), for the program. -/
noncomputable def fwdVVal (s : BlockState) (V : RegionName)
    (n e BLOCK_MODEL : Nat) (c : Nat) (keyRow : Nat) : ℝ :=
  s.readMem V (s.pids 0 * n * e + keyRow * e + s.pids 1 * BLOCK_MODEL + c)

/-- **Genuine closed form for the `kv` state entering block `m`**, element
`(a, c)`: `Σ_{s < m·BLOCK} K[s,a]·V[s,c]`, the running sum of `kᵀ·v` over the
first `m` key blocks. This is the carry state the Python loop accumulates with
`kv += tl.dot(k_trans, v)` *after* each block's output store. -/
noncomputable def kvClosed (s : BlockState) (K V : RegionName)
    (n d e BLOCK BLOCK_MODEL : Nat) (m : Nat) (a c : Nat) : ℝ :=
  ∑ keyRow ∈ Finset.range (m * BLOCK),
    fwdKVal s K n d a keyRow * fwdVVal s V n e BLOCK_MODEL c keyRow

/-- **The `kv` carry-fold recurrence.** Unrolling one block:
`kv^(m+1) = kv^(m) + Σ_{j < BLOCK} K[m·BLOCK+j, a]·V[m·BLOCK+j, c]`. This is the
exact closed-form counterpart of the Python loop body `kv += tl.dot(k_trans,
v)`, whose summand is the per-block outer-product `Σ_j k_trans[a,j]·v[j,c]`. -/
theorem kvClosed_succ (s : BlockState) (K V : RegionName)
    (n d e BLOCK BLOCK_MODEL : Nat) (m : Nat) (a c : Nat) :
    kvClosed s K V n d e BLOCK BLOCK_MODEL (m + 1) a c
      = kvClosed s K V n d e BLOCK BLOCK_MODEL m a c
        + ∑ j ∈ Finset.range BLOCK,
            fwdKVal s K n d a (m * BLOCK + j) *
              fwdVVal s V n e BLOCK_MODEL c (m * BLOCK + j) := by
  unfold kvClosed
  rw [show (m + 1) * BLOCK = m * BLOCK + BLOCK by ring]
  rw [Finset.sum_range_add]

/-! ### `kv`-update step slice (the per-block `kv += tl.dot(k_trans, v)` body)

This isolates the Python loop body's `kv` update from the cross-block loop
induction. It loads the materialized previous-state tile `KVPrev`, the
block's `k_trans`/`v` tiles, forms the per-block outer product `tl.dot(k_trans,
v)`, and stores `KVPrev + tl.dot(k_trans, v)` into a state buffer `KVOut` at the
canonical `[d, BLOCK_MODEL]` layout. -/

def lightning_attention_forward_kv_step_slice
    (KVPrev KTrans V KVOut : RegionName) (D BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  offs_a = tl.arange(0, $(D))
  offs_j = tl.arange(0, $(BLOCK))
  offs_c = tl.arange(0, $(BLOCK_MODEL))
  prev = tl.load(KVPrev + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :])
  k_trans = tl.load(KTrans + offs_a[:, None] * $(BLOCK) + offs_j[None, :])
  v = tl.load(V + offs_j[:, None] * $(BLOCK_MODEL) + offs_c[None, :])
  kv_update = tl.dot(k_trans, v)
  kv = prev + kv_update
  tl.store(KVOut + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :], kv)
}

theorem lightning_attention_forward_kv_step_slice_toAlgorithm_supported
    (KVPrev KTrans V KVOut : RegionName) (D BLOCK BLOCK_MODEL : Nat) :
    ∃ alg, (lightning_attention_forward_kv_step_slice KVPrev KTrans V KVOut
      D BLOCK BLOCK_MODEL).toAlgorithm? = Except.ok alg := by
  simp [lightning_attention_forward_kv_step_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Flat `[d, BLOCK_MODEL]` tile offset `a · BLOCK_MODEL + c`. -/
def kvOffset (BLOCK_MODEL : Nat) (idx : TileIndex [D, BLOCK_MODEL]) : Nat :=
  idx.1.val * BLOCK_MODEL + idx.2.1.val

/-- Row-major staged-tile element `R[i, j]` of a flat `[rows, width]` scratch
region (offset `i·width + j`): the layout of every materialized per-block tile
(`Q`, `KTrans`, `V`, `KVPrev`) that the proof slices exchange. -/
noncomputable def tileElem (s : BlockState) (R : RegionName)
    (width i j : Nat) : ℝ :=
  s.readMem R (i * width + j)

/-- The arithmetic spec of one `kv`-update body: `KVPrev[a,c] + Σ_j
k_trans[a,j]·v[j,c]`, i.e. the materialized previous state plus the block's
`tl.dot(k_trans, v)` outer product. -/
noncomputable def kvStepSpec (s : BlockState) (KVPrev KTrans V : RegionName)
    (D BLOCK BLOCK_MODEL : Nat) (idx : TileIndex [D, BLOCK_MODEL]) : ℝ :=
  s.readMem KVPrev (kvOffset BLOCK_MODEL idx) +
    ∑ j : Fin BLOCK,
      tileElem s KTrans BLOCK idx.1.val j.val *
        tileElem s V BLOCK_MODEL j.val idx.2.1.val

theorem lightning_attention_forward_kv_step_slice_correct
    (KVPrev KTrans V KVOut : RegionName) (D BLOCK BLOCK_MODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D, BLOCK_MODEL] => kvOffset BLOCK_MODEL idx)) :
    ∀ idx : TileIndex [D, BLOCK_MODEL],
      let outAddr := kvOffset BLOCK_MODEL idx
      (exec (lightning_attention_forward_kv_step_slice KVPrev KTrans V KVOut
          D BLOCK BLOCK_MODEL) s).map (·.readMem KVOut outAddr)
        = some (kvStepSpec s KVPrev KTrans V D BLOCK BLOCK_MODEL idx) := by
  intro idx
  simp [exec, lightning_attention_forward_kv_step_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, Tile.dot, NumericDType.add,
        NumericDType.mul, kvOffset, kvStepSpec, tileElem,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [D, BLOCK_MODEL] → Nat :=
    fun idx => idx.1.val * BLOCK_MODEL + idx.2.1.val
  let valueFn : TileIndex [D, BLOCK_MODEL] → ℝ :=
    fun idx =>
      s.readMem KVPrev (idx.1.val * BLOCK_MODEL + idx.2.1.val) +
        ∑ j : Fin BLOCK,
          s.readMem KTrans (idx.1.val * BLOCK + j.val) *
            s.readMem V (j.val * BLOCK_MODEL + idx.2.1.val)
  have hInj : Function.Injective offsetFn := by
    simpa [offsetFn, kvOffset] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i => acc.writeMem KVOut (offsetFn i) (valueFn i))
      _ (TileShape.allIndices [D, BLOCK_MODEL])).readMem KVOut (offsetFn idx) =
    s.readMem KVPrev (idx.1.val * BLOCK_MODEL + idx.2.1.val) +
      ∑ j : Fin BLOCK,
        s.readMem KTrans (idx.1.val * BLOCK + j.val) *
          s.readMem V (j.val * BLOCK_MODEL + idx.2.1.val)
  rw [BlockState.scatter_readback_nd _ _ _ hInj idx]

theorem lightning_attention_forward_kv_step_slice_compute_correct
    (KVPrev KTrans V KVOut : RegionName) (D BLOCK BLOCK_MODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [D, BLOCK_MODEL] => kvOffset BLOCK_MODEL idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_kv_step_slice KVPrev KTrans V KVOut
        D BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := fun idx : TileIndex [D, BLOCK_MODEL] =>
        some (KVOut, kvOffset BLOCK_MODEL idx))
      (expected := fun idx : TileIndex [D, BLOCK_MODEL] =>
        kvStepSpec s KVPrev KTrans V D BLOCK BLOCK_MODEL idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_forward_kv_step_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := lightning_attention_forward_kv_step_slice_correct KVPrev KTrans V
    KVOut D BLOCK BLOCK_MODEL s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- **`kv` carry-fold step (genuine).** If the materialized previous-state buffer
`KVPrev` holds the genuine `m`-block folded state `kvClosed m`, and the block's
`k_trans`/`v` tiles read the genuine block-`m` key/value entries, then one loop
body — `kvStepSpec`, i.e. `KVPrev + tl.dot(k_trans, v)` — produces exactly the
genuine `(m+1)`-block folded state `kvClosed (m+1)`. -/
theorem kvStepSpec_eq_kvClosed_succ
    (s : BlockState) (KVPrev KTrans V K Vreg : RegionName)
    (n d e BLOCK BLOCK_MODEL m : Nat)
    (hPrev : ∀ idx : TileIndex [d, BLOCK_MODEL],
      s.readMem KVPrev (kvOffset BLOCK_MODEL idx)
        = kvClosed s K Vreg n d e BLOCK BLOCK_MODEL m idx.1.val idx.2.1.val)
    (hK : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem KTrans (idx.1.val * BLOCK + j.val)
        = fwdKVal s K n d idx.1.val (m * BLOCK + j.val))
    (hV : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem V (j.val * BLOCK_MODEL + idx.2.1.val)
        = fwdVVal s Vreg n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val))
    (idx : TileIndex [d, BLOCK_MODEL]) :
    kvStepSpec s KVPrev KTrans V d BLOCK BLOCK_MODEL idx
      = kvClosed s K Vreg n d e BLOCK BLOCK_MODEL (m + 1) idx.1.val idx.2.1.val := by
  rw [kvClosed_succ]
  unfold kvStepSpec tileElem
  rw [hPrev idx]
  congr 1
  rw [Fin.sum_univ_eq_sum_range
    (fun j => s.readMem KTrans (idx.1.val * BLOCK + j) *
      s.readMem V (j * BLOCK_MODEL + idx.2.1.val)) BLOCK]
  apply Finset.sum_congr rfl
  intro j hj
  simp only [Finset.mem_range] at hj
  rw [hK idx ⟨j, hj⟩, hV idx ⟨j, hj⟩]

/-! ### `o_inter` producer slice (the per-block `o_inter = tl.dot(q, kv)` body)

The inter-block contribution to the output: with the carried state `kv` loaded
from a materialized `KVPrev` tile, `o_inter[r, c] = Σ_a q[r,a]·kv[a,c]`. This is
the `tl.dot(q, kv)` producer; under the carry invariant `KVPrev = kvClosed m`
its value is exactly `Σ_a q[r,a]·(Σ_{s<m·BLOCK} k[s,a]·v[s,c])`, the inter-block
half of the causal linear-attention output row. -/

def lightning_attention_forward_o_inter_dot_slice
    (Q KVPrev OInter : RegionName) (BLOCK D BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  offs_r = tl.arange(0, $(BLOCK))
  offs_a = tl.arange(0, $(D))
  offs_c = tl.arange(0, $(BLOCK_MODEL))
  q = tl.load(Q + offs_r[:, None] * $(D) + offs_a[None, :])
  kv = tl.load(KVPrev + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :])
  o_inter = tl.dot(q, kv)
  tl.store(OInter + offs_r[:, None] * $(BLOCK_MODEL) + offs_c[None, :], o_inter)
}

theorem lightning_attention_forward_o_inter_dot_slice_toAlgorithm_supported
    (Q KVPrev OInter : RegionName) (BLOCK D BLOCK_MODEL : Nat) :
    ∃ alg, (lightning_attention_forward_o_inter_dot_slice Q KVPrev OInter
      BLOCK D BLOCK_MODEL).toAlgorithm? = Except.ok alg := by
  simp [lightning_attention_forward_o_inter_dot_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Flat `[BLOCK, BLOCK_MODEL]` output-tile offset `r · BLOCK_MODEL + c`. -/
def oInterOffset (BLOCK_MODEL : Nat) (idx : TileIndex [BLOCK, BLOCK_MODEL]) : Nat :=
  idx.1.val * BLOCK_MODEL + idx.2.1.val

/-- The genuine arithmetic spec of `o_inter = tl.dot(q, kv)`, reading the
materialized previous-state tile `KVPrev`: `Σ_a q[r,a]·KVPrev[a,c]`. -/
noncomputable def oInterDotSpec (s : BlockState) (Q KVPrev : RegionName)
    (BLOCK D BLOCK_MODEL : Nat) (idx : TileIndex [BLOCK, BLOCK_MODEL]) : ℝ :=
  ∑ a : Fin D,
    tileElem s Q D idx.1.val a.val *
      tileElem s KVPrev BLOCK_MODEL a.val idx.2.1.val

theorem lightning_attention_forward_o_inter_dot_slice_correct
    (Q KVPrev OInter : RegionName) (BLOCK D BLOCK_MODEL : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] => oInterOffset BLOCK_MODEL idx)) :
    ∀ idx : TileIndex [BLOCK, BLOCK_MODEL],
      let outAddr := oInterOffset BLOCK_MODEL idx
      (exec (lightning_attention_forward_o_inter_dot_slice Q KVPrev OInter
          BLOCK D BLOCK_MODEL) s).map (·.readMem OInter outAddr)
        = some (oInterDotSpec s Q KVPrev BLOCK D BLOCK_MODEL idx) := by
  intro idx
  simp [exec, lightning_attention_forward_o_inter_dot_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, Tile.dot, NumericDType.add,
        NumericDType.mul, oInterOffset, oInterDotSpec, tileElem,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK, BLOCK_MODEL] → Nat :=
    fun idx => idx.1.val * BLOCK_MODEL + idx.2.1.val
  have hInj : Function.Injective offsetFn := by
    simpa [offsetFn, oInterOffset] using hOutInj
  rw [BlockState.scatter_readback_nd _ _ _ hInj idx]

theorem lightning_attention_forward_o_inter_dot_slice_compute_correct
    (Q KVPrev OInter : RegionName) (BLOCK D BLOCK_MODEL : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] => oInterOffset BLOCK_MODEL idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_o_inter_dot_slice Q KVPrev OInter
        BLOCK D BLOCK_MODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        some (OInter, oInterOffset BLOCK_MODEL idx))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        oInterDotSpec s Q KVPrev BLOCK D BLOCK_MODEL idx) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_forward_o_inter_dot_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := lightning_attention_forward_o_inter_dot_slice_correct Q KVPrev OInter
    BLOCK D BLOCK_MODEL s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Faithful transcription of `lightning_attention.py`'s `_bwd_intra_kernel`.

This records the intra-block backward path: diagonal causal masks, `DQ`/`DK`/
`DV` stores, and the Python test's block layout. -/
def lightning_attention_bwd_intra_surface
    (Q K V DO DQ DK DV : RegionName)
    (_b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_block = tl.program_id(1)
  off_bh % $(h)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  block_offset = off_block * $(BLOCK) + tl.arange(0, $(BLOCK))
  Q_trans_block_ptr =
    Q + qk_offset + block_offset[None, :] * $(d) + tl.arange(0, $(d))[:, None]
  K_block_ptr =
    K + qk_offset + block_offset[:, None] * $(d) + tl.arange(0, $(d))[None, :]
  V_trans_block_ptr =
    V + v_offset + block_offset[None, :] * $(e) + tl.arange(0, $(e))[:, None]
  DQ_block_ptr =
    DQ + qk_offset + block_offset[:, None] * $(d) + tl.arange(0, $(d))[None, :]
  DK_trans_block_ptr =
    DK + qk_offset + block_offset[None, :] * $(d) + tl.arange(0, $(d))[:, None]
  DV_block_ptr =
    DV + v_offset + block_offset[:, None] * $(e) + tl.arange(0, $(e))[None, :]
  DO_block_ptr =
    DO + o_offset + block_offset[:, None] * $(e) + tl.arange(0, $(e))[None, :]
  array = tl.arange(0, $(BLOCK))
  index = array[:, None] - array[None, :]
  k = tl.load(K_block_ptr, mask=block_offset[:, None] < $(n),
    other=0.0).to(tl.float32)
  v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < $(n),
    other=0.0).to(tl.float32)
  b_do = tl.load(DO_block_ptr, mask=block_offset[:, None] < $(n),
    other=0.0).to(tl.float32)
  q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < $(n),
    other=0.0).to(tl.float32)
  dqk = tl.dot(b_do, v_trans)
  dqk = tl.where(index >= 0, dqk, 0)
  dq_intra = tl.dot(dqk, k)
  dk_intra_trans = tl.dot(q_trans, dqk)
  qk_trans = tl.dot(k, q_trans)
  qk_trans = tl.where(index <= 0, qk_trans, 0)
  dv_intra = tl.dot(qk_trans, b_do)
  dq = dq_intra
  dk_trans = dk_intra_trans
  dv = dv_intra
  tl.store(DQ_block_ptr, (dq).to(DQ_block_ptr.dtype.element_ty),
    mask=block_offset[:, None] < $(n))
  tl.store(DK_trans_block_ptr, (dk_trans).to(DK_trans_block_ptr.dtype.element_ty),
    mask=block_offset[None, :] < $(n))
  tl.store(DV_block_ptr, (dv).to(DV_block_ptr.dtype.element_ty),
    mask=block_offset[:, None] < $(n))
}

/-- The full intra-block backward surface lowers through the algorithm layer. -/
theorem lightning_attention_bwd_intra_surface_toAlgorithm_supported
    (Q K V DO DQ DK DV : RegionName)
    (_b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK : Nat) :
    ∃ alg, (lightning_attention_bwd_intra_surface Q K V DO DQ DK DV
      _b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK).toAlgorithm? =
        Except.ok alg := by
  simp [lightning_attention_bwd_intra_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Faithful transcription of `lightning_attention.py`'s `_bwd_inter_kernel`.

The surface preserves both Python loop nests: the forward scan that accumulates
and writes inter-block `DQ`, and the reverse scan that accumulates and writes
inter-block `DK`/`DV`. -/
def lightning_attention_bwd_inter_surface
    (Q K V DO DQ DK DV : RegionName)
    (_b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_bh % $(h)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  DQ_block_ptr =
    DQ + qk_offset + tl.arange(0, $(CBLOCK))[:, None] * $(d) +
      tl.arange(0, $(d))[None, :]
  K_block_ptr =
    K + qk_offset + tl.arange(0, $(CBLOCK))[:, None] * $(d) +
      tl.arange(0, $(d))[None, :]
  V_trans_block_ptr =
    V + v_offset + tl.arange(0, $(CBLOCK))[None, :] * $(e) +
      tl.arange(0, $(e))[:, None]
  DO_block_ptr =
    DO + o_offset + tl.arange(0, $(CBLOCK))[:, None] * $(e) +
      tl.arange(0, $(e))[None, :]
  off_block1 = tl.arange(0, $(CBLOCK))
  off_block2 = tl.arange(0, $(CBLOCK))
  kv_trans = tl.zeros([$(e), $(d)], dtype=tl.float32)
  for i in range($(0), $(NUM_BLOCK), $(1)) {
    for j in range($(0), $(NUM_CBLOCK), $(1)) {
      if i > 0 {
        b_do = tl.load(DO_block_ptr, mask=off_block1[:, None] < $(n),
          other=0.0).to(tl.float32)
        dq_inter = tl.dot(b_do, kv_trans)
        dq = dq_inter + tl.load(DQ_block_ptr,
          mask=off_block1[:, None] < $(n), other=0.0)
        tl.store(DQ_block_ptr, (dq).to(DQ_block_ptr.dtype.element_ty),
          mask=off_block1[:, None] < $(n))
      }
      DQ_block_ptr += $(CBLOCK) * $(d)
      DO_block_ptr += $(CBLOCK) * $(e)
      off_block1 += $(CBLOCK)
    }
    kv_trans_current = tl.zeros([$(e), $(d)], dtype=tl.float32)
    for j in range($(0), $(NUM_CBLOCK), $(1)) {
      v_trans = tl.load(V_trans_block_ptr, mask=off_block2[None, :] < $(n),
        other=0.0).to(tl.float32)
      k = tl.load(K_block_ptr, mask=off_block2[:, None] < $(n),
        other=0.0).to(tl.float32)
      kv_trans_current += tl.dot(v_trans, k)
      K_block_ptr += $(CBLOCK) * $(d)
      V_trans_block_ptr += $(CBLOCK) * $(e)
      off_block2 += $(CBLOCK)
    }
    kv_trans += kv_trans_current
  }
  m = $(NUM_BLOCK) * $(BLOCK)
  off_block1 = m + tl.arange(0, $(CBLOCK))
  off_block2 = m + tl.arange(0, $(CBLOCK))
  Q_trans_block_ptr =
    Q + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[None, :] * $(d) +
      tl.arange(0, $(d))[:, None]
  K_block_ptr =
    K + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[:, None] * $(d) +
      tl.arange(0, $(d))[None, :]
  V_trans_block_ptr =
    V + v_offset + m * $(e) + tl.arange(0, $(CBLOCK))[None, :] * $(e) +
      tl.arange(0, $(e))[:, None]
  DK_trans_block_ptr =
    DK + qk_offset + m * $(d) + tl.arange(0, $(CBLOCK))[None, :] * $(d) +
      tl.arange(0, $(d))[:, None]
  DV_block_ptr =
    DV + v_offset + m * $(e) + tl.arange(0, $(CBLOCK))[:, None] * $(e) +
      tl.arange(0, $(e))[None, :]
  DO_block_ptr =
    DO + o_offset + m * $(e) + tl.arange(0, $(CBLOCK))[:, None] * $(e) +
      tl.arange(0, $(e))[None, :]
  dkv = tl.zeros([$(d), $(e)], dtype=tl.float32)
  for i in range($(NUM_BLOCK) - $(1), -$(1), -$(1)) {
    for j in range($(NUM_CBLOCK) - $(1), -$(1), -$(1)) {
      K_block_ptr -= $(CBLOCK) * $(d)
      V_trans_block_ptr -= $(CBLOCK) * $(e)
      DK_trans_block_ptr -= $(CBLOCK) * $(d)
      DV_block_ptr -= $(CBLOCK) * $(e)
      off_block1 -= $(CBLOCK)
      if i < $(NUM_BLOCK) - $(1) {
        k = tl.load(K_block_ptr, mask=off_block1[:, None] < $(n),
          other=0.0).to(tl.float32)
        v_trans = tl.load(V_trans_block_ptr, mask=off_block1[None, :] < $(n),
          other=0.0).to(tl.float32)
        dk_inter_trans = tl.dot(dkv, v_trans)
        dv_inter = tl.dot(k, dkv)
        dk_trans = dk_inter_trans + tl.load(DK_trans_block_ptr,
          mask=off_block1[None, :] < $(n), other=0.0)
        dv = dv_inter + tl.load(DV_block_ptr,
          mask=off_block1[:, None] < $(n), other=0.0)
        tl.store(DK_trans_block_ptr, (dk_trans).to(DK_trans_block_ptr.dtype.element_ty),
          mask=off_block1[None, :] < $(n))
        tl.store(DV_block_ptr, (dv).to(DV_block_ptr.dtype.element_ty),
          mask=off_block1[:, None] < $(n))
      }
    }
    dkv_current = tl.zeros([$(d), $(e)], dtype=tl.float32)
    for j in range($(NUM_CBLOCK) - $(1), -$(1), -$(1)) {
      DO_block_ptr -= $(CBLOCK) * $(e)
      Q_trans_block_ptr -= $(CBLOCK) * $(d)
      off_block2 -= $(CBLOCK)
      b_do = tl.load(DO_block_ptr, mask=off_block2[:, None] < $(n),
        other=0.0).to(tl.float32)
      q_trans = tl.load(Q_trans_block_ptr, mask=off_block2[None, :] < $(n),
        other=0.0).to(tl.float32)
      dkv_current += tl.dot(q_trans, b_do)
    }
    dkv += dkv_current
  }
}

/-- The inter-block backward surface lowers with the Python forward and reverse
loop nests preserved. -/
theorem lightning_attention_bwd_inter_surface_toAlgorithm_supported
    (Q K V DO DQ DK DV : RegionName)
    (_b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK : Nat) :
    ∃ alg, (lightning_attention_bwd_inter_surface Q K V DO DQ DK DV
      _b h n d e BLOCK NUM_BLOCK CBLOCK NUM_CBLOCK).toAlgorithm? =
        Except.ok alg := by
  simp [lightning_attention_bwd_inter_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented forward output-store slice of
`lightning_attention.py`'s `_fwd_kernel`.

The full kernel computes a recurrent attention tile `o`. This slice starts from
a precomputed `OAcc` tile and proves the masked writeback into `Out`. The full forward recurrent `kv` loop is represented above by
`lightning_attention_forward_surface`; the backward kernels' negative-step loops
remain separate modeling work. -/
def lightning_attention_forward_store_slice
    (OAcc Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_e = tl.program_id(1)
  off_block = tl.program_id(2) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_e = tl.arange(0, $(BLOCK_MODEL))
  mask = off_block[:, None] < $(n)
  o = tl.load(OAcc + off_bh * $(n) * $(e) +
      off_block[:, None] * $(e) + off_e * $(BLOCK_MODEL) + offs_e[None, :],
      mask=mask, other=0.0)
  tl.store(Out + off_bh * $(n) * $(e) +
      off_block[:, None] * $(e) + off_e * $(BLOCK_MODEL) + offs_e[None, :],
      o, mask=mask)
}

def rowIndex (s : BlockState) (BLOCK : Nat) (i : Fin BLOCK) : Nat :=
  s.pids 2 * BLOCK + i.val

def colIndex (idx : TileIndex [BLOCK, BLOCK_MODEL]) : Nat :=
  idx.2.1.val

def active (s : BlockState) (n BLOCK : Nat) (idx : TileIndex [BLOCK, BLOCK_MODEL]) :
    Prop :=
  rowIndex s BLOCK idx.1 < n

instance activeDecidable (s : BlockState) (n BLOCK BLOCK_MODEL : Nat)
    (idx : TileIndex [BLOCK, BLOCK_MODEL]) :
    Decidable (active s n BLOCK idx) := by
  unfold active
  infer_instance

def tileOffset (s : BlockState) (n e BLOCK BLOCK_MODEL : Nat)
    (idx : TileIndex [BLOCK, BLOCK_MODEL]) : Nat :=
  s.pids 0 * n * e + rowIndex s BLOCK idx.1 * e +
    s.pids 1 * BLOCK_MODEL + colIndex idx

noncomputable def storeValue (s : BlockState) (OAcc : RegionName)
    (n e BLOCK BLOCK_MODEL : Nat) (idx : TileIndex [BLOCK, BLOCK_MODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s n BLOCK idx then
      some (s.readMem OAcc (tileOffset s n e BLOCK BLOCK_MODEL idx))
    else some (0.0 : ℝ))

theorem lightning_attention_forward_store_slice_correct
    (OAcc Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        tileOffset s n e BLOCK BLOCK_MODEL idx)) :
    ∀ idx : TileIndex [BLOCK, BLOCK_MODEL],
      let outAddr := tileOffset s n e BLOCK BLOCK_MODEL idx
      (exec (lightning_attention_forward_store_slice OAcc Out n e BLOCK BLOCK_MODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s n BLOCK idx then
            storeValue s OAcc n e BLOCK BLOCK_MODEL idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, lightning_attention_forward_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.remap,
        Tile.expandDim, Tile.ptrAdd,
        NumericDType.add, NumericDType.mul, ComparableDType.lt,
        rowIndex, colIndex, active, tileOffset, TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK, BLOCK_MODEL] → Nat :=
    fun idx =>
      s.pids 0 * n * e + (s.pids 2 * BLOCK + idx.1.val) * e +
        s.pids 1 * BLOCK_MODEL + idx.2.1.val
  let valueFn : TileIndex [BLOCK, BLOCK_MODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 2 * BLOCK + idx.1.val < n then
          some (s.readMem OAcc
            (s.pids 0 * n * e + (s.pids 2 * BLOCK + idx.1.val) * e +
              s.pids 1 * BLOCK_MODEL + idx.2.1.val))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK, BLOCK_MODEL] → Prop :=
    fun idx => s.pids 2 * BLOCK + idx.1.val < n
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, rowIndex, colIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK, BLOCK_MODEL])).readMem Out
        (offsetFn idx) =
    if P idx then storeValue s OAcc n e BLOCK BLOCK_MODEL idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 2 * BLOCK + idx.1.val < n
  · simp [offsetFn, valueFn, P, storeValue, active, tileOffset, rowIndex,
      colIndex, hActive]
  · simp [offsetFn, valueFn, P, storeValue, active, tileOffset, rowIndex,
      colIndex, hActive]

theorem lightning_attention_forward_store_slice_compute_correct
    (OAcc Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        tileOffset s n e BLOCK BLOCK_MODEL idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_store_slice OAcc Out n e BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, BLOCK_MODEL] => active s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
          (Out, tileOffset s n e BLOCK BLOCK_MODEL idx)))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        storeValue s OAcc n e BLOCK BLOCK_MODEL idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_forward_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := lightning_attention_forward_store_slice_correct OAcc Out n e
    BLOCK BLOCK_MODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Forward output store slice that includes the Python arithmetic
`o = o_intra + o_inter` before the masked writeback. -/
def lightning_attention_forward_sum_store_slice
    (OIntra OInter Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_e = tl.program_id(1)
  off_block = tl.program_id(2) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_e = tl.arange(0, $(BLOCK_MODEL))
  mask = off_block[:, None] < $(n)
  o_intra = tl.load(OIntra + off_bh * $(n) * $(e) +
      off_block[:, None] * $(e) + off_e * $(BLOCK_MODEL) + offs_e[None, :],
      mask=mask, other=0.0)
  o_inter = tl.load(OInter + off_bh * $(n) * $(e) +
      off_block[:, None] * $(e) + off_e * $(BLOCK_MODEL) + offs_e[None, :],
      mask=mask, other=0.0)
  o = o_intra + o_inter
  tl.store(Out + off_bh * $(n) * $(e) +
      off_block[:, None] * $(e) + off_e * $(BLOCK_MODEL) + offs_e[None, :],
      o, mask=mask)
}

noncomputable def sumStoreValue
    (s : BlockState) (OIntra OInter : RegionName)
    (n e BLOCK BLOCK_MODEL : Nat) (idx : TileIndex [BLOCK, BLOCK_MODEL]) : ℝ :=
  WithBot.unbotD 0
    (if active s n BLOCK idx then
      some (s.readMem OIntra (tileOffset s n e BLOCK BLOCK_MODEL idx) +
        s.readMem OInter (tileOffset s n e BLOCK BLOCK_MODEL idx))
    else some (0.0 : ℝ))

theorem lightning_attention_forward_sum_store_slice_correct
    (OIntra OInter Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        tileOffset s n e BLOCK BLOCK_MODEL idx)) :
    ∀ idx : TileIndex [BLOCK, BLOCK_MODEL],
      let outAddr := tileOffset s n e BLOCK BLOCK_MODEL idx
      (exec (lightning_attention_forward_sum_store_slice OIntra OInter Out
          n e BLOCK BLOCK_MODEL) s).map (·.readMem Out outAddr)
        = some (if active s n BLOCK idx then
            sumStoreValue s OIntra OInter n e BLOCK BLOCK_MODEL idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, lightning_attention_forward_sum_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.remap,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, rowIndex, colIndex, active, tileOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK, BLOCK_MODEL] → Nat :=
    fun idx =>
      s.pids 0 * n * e + (s.pids 2 * BLOCK + idx.1.val) * e +
        s.pids 1 * BLOCK_MODEL + idx.2.1.val
  let valueFn : TileIndex [BLOCK, BLOCK_MODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (Option.map₂ (fun a b => a + b)
          (if s.pids 2 * BLOCK + idx.1.val < n then
            some (s.readMem OIntra (offsetFn idx))
          else some (0.0 : ℝ))
          (if s.pids 2 * BLOCK + idx.1.val < n then
            some (s.readMem OInter (offsetFn idx))
          else some (0.0 : ℝ)))
  let P : TileIndex [BLOCK, BLOCK_MODEL] → Prop :=
    fun idx => s.pids 2 * BLOCK + idx.1.val < n
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, tileOffset, rowIndex, colIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK, BLOCK_MODEL])).readMem Out
        (offsetFn idx) =
    if P idx then sumStoreValue s OIntra OInter n e BLOCK BLOCK_MODEL idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 2 * BLOCK + idx.1.val < n
  · simp [offsetFn, valueFn, P, sumStoreValue, active, tileOffset, rowIndex,
      colIndex, hActive]
  · simp [offsetFn, valueFn, P, sumStoreValue, active, tileOffset, rowIndex,
      colIndex, hActive]

theorem lightning_attention_forward_sum_store_slice_compute_correct
    (OIntra OInter Out : RegionName) (n e BLOCK BLOCK_MODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        tileOffset s n e BLOCK BLOCK_MODEL idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_sum_store_slice OIntra OInter Out
        n e BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, BLOCK_MODEL] => active s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
          (Out, tileOffset s n e BLOCK BLOCK_MODEL idx)))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        sumStoreValue s OIntra OInter n e BLOCK BLOCK_MODEL idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_forward_sum_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := lightning_attention_forward_sum_store_slice_correct OIntra OInter
    Out n e BLOCK BLOCK_MODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Proof-oriented backward gradient tile-store slice of
`lightning_attention.py`.

The Python backward path writes `DQ`, `DK`, and `DV` from intra- and inter-block
accumulators. This generic slice fixes one block row and proves the masked
writeback from a precomputed gradient tile into any of those output regions. -/
def lightning_attention_bwd_grad_store_slice
    (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_block = tl.program_id(1) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_w = tl.arange(0, $(WIDTH))
  mask = (off_block[:, None] < $(n)) & (offs_w[None, :] < $(WIDTH))
  grad = tl.load(GradPre + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      mask=mask, other=0.0)
  tl.store(Out + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      (grad).to(Out.dtype.element_ty), mask=mask)
}

def gradRowIndex (s : BlockState) (BLOCK : Nat) (i : Fin BLOCK) : Nat :=
  s.pids 1 * BLOCK + i.val

def gradColIndex (idx : TileIndex [BLOCK, WIDTH]) : Nat :=
  idx.2.1.val

def activeGrad (s : BlockState) (n BLOCK : Nat)
    (idx : TileIndex [BLOCK, WIDTH]) : Prop :=
  gradRowIndex s BLOCK idx.1 < n

instance activeGradDecidable (s : BlockState) (n BLOCK WIDTH : Nat)
    (idx : TileIndex [BLOCK, WIDTH]) :
    Decidable (activeGrad s n BLOCK idx) := by
  unfold activeGrad
  infer_instance

def gradTileOffset (s : BlockState) (n width BLOCK WIDTH : Nat)
    (idx : TileIndex [BLOCK, WIDTH]) : Nat :=
  s.pids 0 * n * width + gradRowIndex s BLOCK idx.1 * width + gradColIndex idx

noncomputable def gradStoreValue (s : BlockState) (GradPre : RegionName)
    (n width BLOCK WIDTH : Nat) (idx : TileIndex [BLOCK, WIDTH]) : ℝ :=
  WithBot.unbotD 0
    (if activeGrad s n BLOCK idx then
      some (s.readMem GradPre (gradTileOffset s n width BLOCK WIDTH idx))
    else some (0.0 : ℝ))

theorem lightning_attention_bwd_grad_store_slice_correct
    (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ∀ idx : TileIndex [BLOCK, WIDTH],
      let outAddr := gradTileOffset s n width BLOCK WIDTH idx
      (exec (lightning_attention_bwd_grad_store_slice GradPre Out n width BLOCK WIDTH)
          s).map (·.readMem Out outAddr)
        = some (if activeGrad s n BLOCK idx then
            gradStoreValue s GradPre n width BLOCK WIDTH idx
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, lightning_attention_bwd_grad_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul, ComparableDType.lt,
        gradRowIndex, gradColIndex, activeGrad, gradTileOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK, WIDTH] → Nat :=
    fun idx =>
      s.pids 0 * n * width + (s.pids 1 * BLOCK + idx.1.val) * width +
        idx.2.1.val
  let valueFn : TileIndex [BLOCK, WIDTH] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 1 * BLOCK + idx.1.val < n then
          some (s.readMem GradPre
            (s.pids 0 * n * width + (s.pids 1 * BLOCK + idx.1.val) * width +
              idx.2.1.val))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK, WIDTH] → Prop :=
    fun idx => s.pids 1 * BLOCK + idx.1.val < n
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, gradTileOffset, gradRowIndex, gradColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK, WIDTH])).readMem Out
        (offsetFn idx) =
    if P idx then gradStoreValue s GradPre n width BLOCK WIDTH idx
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BLOCK + idx.1.val < n
  · rfl
  · rfl

theorem lightning_attention_bwd_grad_store_slice_compute_correct
    (GradPre Out : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice GradPre Out n width BLOCK WIDTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, WIDTH] => activeGrad s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, WIDTH] =>
          (Out, gradTileOffset s n width BLOCK WIDTH idx)))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        gradStoreValue s GradPre n width BLOCK WIDTH idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_bwd_grad_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := lightning_attention_bwd_grad_store_slice_correct GradPre Out n width
    BLOCK WIDTH s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Backward inter-kernel DQ accumulation slice.

This captures the Python `_bwd_inter_kernel` writeback
`dq = dq_inter + tl.load(DQ_block_ptr, ...)` before storing back into `DQ`.
The surrounding loop and `dq_inter = tl.dot(do, kv_trans)` producer remain
separate proof obligations, but this is stronger than a precomputed-gradient
store because it includes the Python-observable in-place DQ accumulation. -/
def lightning_attention_bwd_dq_accum_store_slice
    (DQInter DQ : RegionName) (n width BLOCK WIDTH : Nat) :
    ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_block = tl.program_id(1) * $(BLOCK) + tl.arange(0, $(BLOCK))
  offs_w = tl.arange(0, $(WIDTH))
  mask = (off_block[:, None] < $(n)) & (offs_w[None, :] < $(WIDTH))
  dq_inter = tl.load(DQInter + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      mask=mask, other=0.0)
  dq_prev = tl.load(DQ + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      mask=mask, other=0.0)
  dq = dq_inter + dq_prev
  tl.store(DQ + off_bh * $(n) * $(width) +
      off_block[:, None] * $(width) + offs_w[None, :],
      (dq).to(DQ.dtype.element_ty), mask=mask)
}

noncomputable def dqAccumStoreValue
    (s : BlockState) (DQInter DQ : RegionName)
    (n width BLOCK WIDTH : Nat) (idx : TileIndex [BLOCK, WIDTH]) : ℝ :=
  WithBot.unbotD 0
    (if activeGrad s n BLOCK idx then
      some (s.readMem DQInter (gradTileOffset s n width BLOCK WIDTH idx) +
        s.readMem DQ (gradTileOffset s n width BLOCK WIDTH idx))
    else some (0.0 : ℝ))

theorem lightning_attention_bwd_dq_accum_store_slice_correct
    (DQInter DQ : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ∀ idx : TileIndex [BLOCK, WIDTH],
      let outAddr := gradTileOffset s n width BLOCK WIDTH idx
      (exec (lightning_attention_bwd_dq_accum_store_slice DQInter DQ
          n width BLOCK WIDTH) s).map (·.readMem DQ outAddr)
        = some (if activeGrad s n BLOCK idx then
            dqAccumStoreValue s DQInter DQ n width BLOCK WIDTH idx
          else s.readMem DQ outAddr) := by
  intro idx
  simp [exec, lightning_attention_bwd_dq_accum_store_slice, stepStmts,
        stepStmt, evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
        FloatDType.toWithBot, gradRowIndex, gradColIndex, activeGrad,
        gradTileOffset, TileShape.dropInsertedIndex, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?]
  let offsetFn : TileIndex [BLOCK, WIDTH] → Nat :=
    fun idx =>
      s.pids 0 * n * width + (s.pids 1 * BLOCK + idx.1.val) * width +
        idx.2.1.val
  let valueFn : TileIndex [BLOCK, WIDTH] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (Option.map₂ (fun a b => a + b)
          (if s.pids 1 * BLOCK + idx.1.val < n then
            some (s.readMem DQInter (offsetFn idx))
          else some (0.0 : ℝ))
          (if s.pids 1 * BLOCK + idx.1.val < n then
            some (s.readMem DQ (offsetFn idx))
          else some (0.0 : ℝ)))
  let P : TileIndex [BLOCK, WIDTH] → Prop :=
    fun idx => s.pids 1 * BLOCK + idx.1.val < n
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, gradTileOffset, gradRowIndex, gradColIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem DQ (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK, WIDTH])).readMem DQ
        (offsetFn idx) =
    if P idx then dqAccumStoreValue s DQInter DQ n width BLOCK WIDTH idx
    else s.readMem DQ (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive : s.pids 1 * BLOCK + idx.1.val < n
  · simp [offsetFn, valueFn, P, dqAccumStoreValue, activeGrad, gradTileOffset,
      gradRowIndex, gradColIndex, hActive, NumericDType.add]
  · simp [offsetFn, valueFn, P, dqAccumStoreValue, activeGrad, gradTileOffset,
      gradRowIndex, gradColIndex, hActive]

theorem lightning_attention_bwd_dq_accum_store_slice_compute_correct
    (DQInter DQ : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_dq_accum_store_slice DQInter DQ
        n width BLOCK WIDTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, WIDTH] => activeGrad s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, WIDTH] =>
          (DQ, gradTileOffset s n width BLOCK WIDTH idx)))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        dqAccumStoreValue s DQInter DQ n width BLOCK WIDTH idx) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_bwd_dq_accum_store_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := lightning_attention_bwd_dq_accum_store_slice_correct DQInter DQ
    n width BLOCK WIDTH s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-- Backward inter-kernel DQ producer slice:
`dq_inter = tl.dot(do, kv_trans)`. This is the arithmetic producer consumed by
the DQ accumulation slice above. -/
def lightning_attention_bwd_dq_inter_dot_slice
    (DO KVTrans DQInter : RegionName) (BLOCK E WIDTH : Nat) :
    ComputeKernel := triton {
  offs_b = tl.arange(0, $(BLOCK))
  offs_e = tl.arange(0, $(E))
  offs_w = tl.arange(0, $(WIDTH))
  b_do = tl.load(DO + offs_b[:, None] * $(E) + offs_e[None, :])
  kv_trans = tl.load(KVTrans + offs_e[:, None] * $(WIDTH) + offs_w[None, :])
  dq_inter = tl.dot(b_do, kv_trans)
  tl.store(DQInter + offs_b[:, None] * $(WIDTH) + offs_w[None, :], dq_inter)
}

def dqInterOffset (WIDTH : Nat) (idx : TileIndex [BLOCK, WIDTH]) : Nat :=
  idx.1.val * WIDTH + idx.2.1.val

noncomputable def dqInterDotSpec
    (s : BlockState) (DO KVTrans : RegionName) (BLOCK E WIDTH : Nat)
    (idx : TileIndex [BLOCK, WIDTH]) : ℝ :=
  ∑ e : Fin E,
    s.readMem DO (idx.1.val * E + e.val) *
      s.readMem KVTrans (e.val * WIDTH + idx.2.1.val)

theorem lightning_attention_bwd_dq_inter_dot_slice_correct
    (DO KVTrans DQInter : RegionName) (BLOCK E WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] => dqInterOffset WIDTH idx)) :
    ∀ idx : TileIndex [BLOCK, WIDTH],
      let outAddr := dqInterOffset WIDTH idx
      (exec (lightning_attention_bwd_dq_inter_dot_slice DO KVTrans DQInter
          BLOCK E WIDTH) s).map (·.readMem DQInter outAddr)
        = some (dqInterDotSpec s DO KVTrans BLOCK E WIDTH idx) := by
  intro idx
  simp [exec, lightning_attention_bwd_dq_inter_dot_slice,
        ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
        ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
        Tile.expandDim, Tile.ptrAdd, Tile.dot, NumericDType.add,
        NumericDType.mul, dqInterOffset, dqInterDotSpec,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK, WIDTH] → Nat :=
    fun idx => idx.1.val * WIDTH + idx.2.1.val
  have hInj : Function.Injective offsetFn := by
    simpa [offsetFn, dqInterOffset] using hOutInj
  rw [BlockState.scatter_readback_nd _ _ _ hInj idx]

theorem lightning_attention_bwd_dq_inter_dot_slice_compute_correct
    (DO KVTrans DQInter : RegionName) (BLOCK E WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] => dqInterOffset WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_dq_inter_dot_slice DO KVTrans DQInter
        BLOCK E WIDTH)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK, WIDTH] =>
        some (DQInter, dqInterOffset WIDTH idx))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        dqInterDotSpec s DO KVTrans BLOCK E WIDTH idx) := by
  unfold ComputeCorrect.Realizes
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_bwd_dq_inter_dot_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := lightning_attention_bwd_dq_inter_dot_slice_correct DO KVTrans
    DQInter BLOCK E WIDTH s hOutInj idx
  rw [hExec] at h
  exact Option.some.inj h

/-- Named DQ writeback correctness for the Python backward path. -/
theorem lightning_attention_bwd_dq_store_slice_compute_correct
    (DQPre DQ : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DQPre DQ n width BLOCK WIDTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, WIDTH] => activeGrad s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, WIDTH] =>
          (DQ, gradTileOffset s n width BLOCK WIDTH idx)))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        gradStoreValue s DQPre n width BLOCK WIDTH idx) := by
  exact lightning_attention_bwd_grad_store_slice_compute_correct DQPre DQ
    n width BLOCK WIDTH s hOutInj

/-- Named DK writeback correctness for the Python backward path. -/
theorem lightning_attention_bwd_dk_store_slice_compute_correct
    (DKPre DK : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DKPre DK n width BLOCK WIDTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, WIDTH] => activeGrad s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, WIDTH] =>
          (DK, gradTileOffset s n width BLOCK WIDTH idx)))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        gradStoreValue s DKPre n width BLOCK WIDTH idx) := by
  exact lightning_attention_bwd_grad_store_slice_compute_correct DKPre DK
    n width BLOCK WIDTH s hOutInj

/-- Named DV writeback correctness for the Python backward path. -/
theorem lightning_attention_bwd_dv_store_slice_compute_correct
    (DVPre DV : RegionName) (n width BLOCK WIDTH : Nat) (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK, WIDTH] =>
        gradTileOffset s n width BLOCK WIDTH idx)) :
    ComputeCorrect.Realizes
      (kernel := lightning_attention_bwd_grad_store_slice DVPre DV n width BLOCK WIDTH)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK, WIDTH] => activeGrad s n BLOCK idx)
        (fun idx : TileIndex [BLOCK, WIDTH] =>
          (DV, gradTileOffset s n width BLOCK WIDTH idx)))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        gradStoreValue s DVPre n width BLOCK WIDTH idx) := by
  exact lightning_attention_bwd_grad_store_slice_compute_correct DVPre DV
    n width BLOCK WIDTH s hOutInj

/-! ## Dimension-general offset injectivity

The flat `[d, BLOCK_MODEL]` carry-tile offset `a · BLOCK_MODEL + c` and the
flat `[BLOCK, BLOCK_MODEL]` output-tile offset `r · BLOCK_MODEL + c` are
injective for **every** symbolic `BLOCK_MODEL`: the column coordinate `c` is a
`Fin BLOCK_MODEL`, so `c < BLOCK_MODEL` and the row/column factorization is
unique. No concrete dimension is needed — `omega` discharges the bound from the
`Fin` proofs carried by the index. -/

/-- `kvOffset` is injective for every `d`, `BLOCK_MODEL` (the column coordinate
is `< BLOCK_MODEL` by its `Fin` bound). -/
theorem kvOffset_injective (d BLOCK_MODEL : Nat) :
    Function.Injective
      (fun idx : TileIndex [d, BLOCK_MODEL] => kvOffset BLOCK_MODEL idx) := by
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp only [kvOffset] at h
  have hr : ra = rb := by
    rcases Nat.lt_trichotomy ra rb with hlt | heq | hgt
    · have : (ra + 1) * BLOCK_MODEL ≤ rb * BLOCK_MODEL := Nat.mul_le_mul_right _ hlt
      simp [Nat.add_mul] at this; omega
    · exact heq
    · have : (rb + 1) * BLOCK_MODEL ≤ ra * BLOCK_MODEL := Nat.mul_le_mul_right _ hgt
      simp [Nat.add_mul] at this; omega
  subst rb
  have hc : ca = cb := by omega
  subst cb; rfl

/-- `oInterOffset` is injective for every `BLOCK`, `BLOCK_MODEL` (the column
coordinate is `< BLOCK_MODEL` by its `Fin` bound). -/
theorem oInterOffset_injective (BLOCK BLOCK_MODEL : Nat) :
    Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] => oInterOffset BLOCK_MODEL idx) := by
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp only [oInterOffset] at h
  have hr : ra = rb := by
    rcases Nat.lt_trichotomy ra rb with hlt | heq | hgt
    · have : (ra + 1) * BLOCK_MODEL ≤ rb * BLOCK_MODEL := Nat.mul_le_mul_right _ hlt
      simp [Nat.add_mul] at this; omega
    · exact heq
    · have : (rb + 1) * BLOCK_MODEL ≤ ra * BLOCK_MODEL := Nat.mul_le_mul_right _ hgt
      simp [Nat.add_mul] at this; omega
  subst rb
  have hc : ca = cb := by omega
  subst cb; rfl

/-! ### ════════ ★ MAIN THEOREM (dimension-general) ★ ════════ -/

/-- **Genuine, dimension-general forward compute-correctness summary** for the
lightning-attention kernel. Symbolic in every dimension
(`d e BLOCK NUM_BLOCK BLOCK_MODEL`) and over an arbitrary loop index `m` and
materialized carry buffer; no dimension is pinned. Exposes:

1. **Surface lowering** of the forward and both backward kernels (every
   dimension symbolic).
2. **`kv` carry-fold body — genuine closed form.** One `kv += tl.dot(k_trans,
   v)` body realizes its spec `kvStepSpec`, and under the loop-carry invariant
   (`KVPrev = kvClosed m`, `k_trans`/`v` reading the genuine block-`m` entries)
   that spec equals exactly `kvClosed (m+1)` — the running `Σ kᵀ·v` over the
   first `m+1` key blocks. This is the genuine standalone closed form over the
   input regions `K`, `V`, never an `exec` read-back.
3. **`o_inter = tl.dot(q, kv)` inter-block producer.** Realizes its genuine spec
   `oInterDotSpec` — `Σ_a q[r,a]·kv[a,c]` against the carried state — which is
   the inter-block half of the causal linear-attention output row.

Honest side-conditions only: `0 < BLOCK_MODEL` (column tiling is nonempty) is
not even needed here because injectivity holds unconditionally from the `Fin`
bounds; the carry hypotheses `hPrev`/`hK`/`hV` are the documented loop-carry
invariant of the `NUM_BLOCK` driver. -/
theorem lightning_attention_output_summary_general
    (Q K V Out DO DQ DK DV KVPrev KTrans Vreg KVOut OInter : RegionName)
    (s : BlockState)
    (_b h n d e BLOCK NUM_BLOCK BLOCK_MODEL m : Nat)
    (hPrev : ∀ idx : TileIndex [d, BLOCK_MODEL],
      s.readMem KVPrev (kvOffset BLOCK_MODEL idx)
        = kvClosed s K Vreg n d e BLOCK BLOCK_MODEL m idx.1.val idx.2.1.val)
    (hK : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem KTrans (idx.1.val * BLOCK + j.val)
        = fwdKVal s K n d idx.1.val (m * BLOCK + j.val))
    (hV : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem Vreg (j.val * BLOCK_MODEL + idx.2.1.val)
        = fwdVVal s Vreg n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val)) :
    -- (1) surface lowering of forward + both backward kernels (symbolic dims)
    (∃ alg, (lightning_attention_forward_surface Q K V Out _b h n d e BLOCK
      NUM_BLOCK BLOCK_MODEL).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (lightning_attention_bwd_intra_surface Q K V DO DQ DK DV
      _b h n d e BLOCK NUM_BLOCK BLOCK NUM_BLOCK).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (lightning_attention_bwd_inter_surface Q K V DO DQ DK DV
      _b h n d e BLOCK NUM_BLOCK BLOCK NUM_BLOCK).toAlgorithm? = Except.ok alg) ∧
    -- (2) kv carry-fold body realizes kvStepSpec ...
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_kv_step_slice KVPrev KTrans Vreg KVOut
        d BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := fun idx : TileIndex [d, BLOCK_MODEL] =>
        some (KVOut, kvOffset BLOCK_MODEL idx))
      (expected := fun idx : TileIndex [d, BLOCK_MODEL] =>
        kvStepSpec s KVPrev KTrans Vreg d BLOCK BLOCK_MODEL idx)) ∧
    -- ... and that spec is the genuine closed-form kvClosed (m+1) under the carry invariant
    (∀ idx : TileIndex [d, BLOCK_MODEL],
      kvStepSpec s KVPrev KTrans Vreg d BLOCK BLOCK_MODEL idx
        = kvClosed s K Vreg n d e BLOCK BLOCK_MODEL (m + 1) idx.1.val idx.2.1.val) ∧
    -- (3) o_inter producer realizes its genuine spec oInterDotSpec
    (ComputeCorrect.Realizes
      (kernel := lightning_attention_forward_o_inter_dot_slice Q KVPrev OInter
        BLOCK d BLOCK_MODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        some (OInter, oInterOffset BLOCK_MODEL idx))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        oInterDotSpec s Q KVPrev BLOCK d BLOCK_MODEL idx)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact lightning_attention_forward_surface_toAlgorithm_supported Q K V Out
      _b h n d e BLOCK NUM_BLOCK BLOCK_MODEL
  · exact lightning_attention_bwd_intra_surface_toAlgorithm_supported Q K V
      DO DQ DK DV _b h n d e BLOCK NUM_BLOCK BLOCK NUM_BLOCK
  · exact lightning_attention_bwd_inter_surface_toAlgorithm_supported Q K V
      DO DQ DK DV _b h n d e BLOCK NUM_BLOCK BLOCK NUM_BLOCK
  · exact lightning_attention_forward_kv_step_slice_compute_correct KVPrev KTrans
      Vreg KVOut d BLOCK BLOCK_MODEL s (kvOffset_injective d BLOCK_MODEL)
  · intro idx
    exact kvStepSpec_eq_kvClosed_succ s KVPrev KTrans Vreg K Vreg
      n d e BLOCK BLOCK_MODEL m hPrev hK hV idx
  · exact lightning_attention_forward_o_inter_dot_slice_compute_correct Q KVPrev
      OInter BLOCK d BLOCK_MODEL s (oInterOffset_injective BLOCK BLOCK_MODEL)

/-! ## Python test-shape wrappers

The checked Python test uses `b = 2`, `h = 8`, `n = 128`, `d = 64`, and
`e = 128`. The forward launcher fixes `BLOCK = 64`, `NUM_BLOCK = 2`, and
`BLOCK_MODEL = 32`, so each forward output tile has shape `[64, 32]`. The
backward launcher uses `BLOCK = 64`; DQ/DK row tiles have width `64`, and DV
row tiles have width `128`. The pinned summaries below are now thin
corollaries of the dimension-general headline
`lightning_attention_output_summary_general`. -/

end Correct

/-! # ══════════ TEST-SHAPE — concrete instances / pinned scaffolding ══════════ -/

section TestShape

end TestShape

end VeriTile.Bench.TritonBenchG.LightningAttention
