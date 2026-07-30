import VeriTile.Triton

/-!
# `lightning_attention` — strict per-kernel correctness

`_fwd_kernel` is the lightning-attention forward recurrent tile scan: each
program walks `NUM_BLOCK` key/value tiles, carrying a `[d, e]` key-value
accumulator `kv` across the loop, combining an intra-block (masked, decayed)
attention term with an inter-block term `q · kv` to produce the output tile.
The backward kernels (`_bwd_intra`, `_bwd_inter`) compute the `dq`/`dk`/`dv`
gradients with the matching reverse recurrence.

## Scope — what is and is not verified

Verified:

* **The whole `_fwd_kernel` loop body at a fixed block index `m`**
  (`lightning_attention_forward_body_slice`), over the *launched* regions `Q`, `K`,
  `V`, `Out` at their real addresses: the three masked tile loads,
  `qk = tl.dot(q, k_trans)`, the causal mask, `o_intra = tl.dot(qk, v)`,
  `o_inter = tl.dot(q, kv)`, `o = o_intra + o_inter`, and the masked `tl.store`.
  Under the assumed loop-carry invariant `KVPrev = kvClosed m` and the full-block
  scope `m·BLOCK + BLOCK ≤ n`, it delivers exactly the causal linear-attention row
  `o[t,c] = Σ_{s ≤ t}(Σ_a q[t,a]·k[s,a])·v[s,c]` (`fwdOutClosed`).
* Two smaller **hand-cut single-block slices** of the same body — the
  `kv += tl.dot(k_trans, v)` carry update and the `o_inter = tl.dot(q, kv)`
  inter-block product — plus the algebraic fact that, under an *assumed* carry
  invariant, the first of them advances `kvClosed` by one block.

All three `@triton.jit` bodies are additionally shown to *lower* to the algorithm
layer, which is well-formedness, not correctness.

The host launch (grid shape, block-model tiling, and how the runtime composes
per-program tile writes) is the *trusted boundary*. Because the program ids are
universally quantified, each per-program statement covers every program.

## Proof architecture

```
lightning_attention_output_summary_general               ← TOP THEOREM
  ├─ lightning_attention_{forward,bwd_intra,bwd_inter}_surface_toAlgorithm_supported
  ├─ lightning_attention_forward_kv_step_slice_compute_correct   kv += dot(k_trans, v) body
  ├─ kvStepSpec_eq_kvClosed_succ                  carry invariant: body advances kvClosed
  │    └─ kvClosed / kvClosed_succ                kv carry-fold over K·V (genuine closed form)
  ├─ lightning_attention_forward_o_inter_dot_slice_compute_correct  o_inter = dot(q, kv) body
  └─ lightning_attention_forward_body_closed_form_general   ★ whole body → Out
       ├─ lightning_attention_forward_body_slice_correct    (expected := bodyStoreValue)
       │    └─ bodyOutOffset_injective                      needs BLOCK_MODEL ≤ e
       ├─ bodyStoreValue_eq_bodyStepSpec       WithBot cell layer → readable ℝ form
       └─ bodyStepSpec_eq_fwdOutClosed         o_intra + o_inter = causal linear attention
            └─ fwdOutClosed / kvClosed                      genuine closed forms

-- present in this file but NOT reachable from the headline (backward writebacks):
lightning_attention_bwd_{dq,dk,dv}_store_slice_compute_correct
lightning_attention_bwd_dq_{accum_store,inter_dot}_slice_compute_correct
```

## Modeling boundary — read before trusting anything below

Arithmetic is over `ℝ`, not bit-accurate IEEE float; dtype `.to(...)` casts
erase to the identity. **Genuine closed forms (forward):** `kvClosed m` is a
standalone specification over the input regions `K, V` — the running `kᵀ·v` sum
over the first `m` key blocks — and `kvClosed_succ` is the exact closed-form
counterpart of the Python `kv += tl.dot(k_trans, v)`. `fwdOutClosed t c` is the
causal linear-attention row `Σ_{s ≤ t}(Σ_a q[t,a]·k[s,a])·v[s,c]` over `Q, K, V`,
what `_fwd_kernel` stores into `Out`.

Scope limits and what is **outside** the claims in this file:

* **The causal predicate's spelling.** Python computes
  `index = off_block[:, None] - off_block[None, :]` in *signed* int32 and tests
  `index >= 0`. This model's `-` on `.nat` tiles truncates at zero, which would
  make `index >= 0` a tautology — so `lightning_attention_forward_body_slice`
  spells the same predicate as the direct comparison
  `offs_r[:, None] >= offs_r[None, :]`, equal to `r - j ≥ 0` over ℤ for the
  non-negative lane indices involved. `lightning_attention_forward_surface` keeps
  the *literal* Python spelling and therefore models that mask as a no-op; the
  surface carries only the lowering claim, never a semantic one, but a reader
  comparing the two must know why they differ.
* **Full-block scope.** The `Out`-writeback face assumes
  `hFullBlock : m·BLOCK + BLOCK ≤ n`, under which every load/store mask lane is
  live. Since `NUM_BLOCK = cdiv(n, BLOCK)` (`lightning_attention.py:378`) this
  covers every block except a partial tail one; the partial tail block is trusted
  boundary. `hBM : BLOCK_MODEL ≤ e` is the store-offset no-aliasing condition
  (`BLOCK_MODEL = min(next_power_of_2(e), 32)`, `lightning_attention.py:380`).
* **The cross-block loop.** The `NUM_BLOCK` driver threading the carry is not
  modeled. `hPrev`/`hK`/`hV` are *assumed*: nothing proves `kvClosed 0 = 0`, and
  nothing chains block `m`'s output buffer into block `m+1`'s input buffer. This
  is the one remaining hole in the forward story — with `hPrev` supplied, block
  `m`'s `Out` row is fully determined.
* **Two earlier store-slice families were deleted, not restored.** Both read a
  phantom `tl.program_id(2) * BLOCK` block index while `_fwd_kernel`'s launch grid
  is two-dimensional (`grid = (b*h, cdiv(e, BLOCK_MODEL))`,
  `lightning_attention.py:381`) and the block index is the *loop* variable, not a
  program id. One of them was additionally a pure memcpy. The face that replaces
  them takes the block index `m` as an explicit parameter.
* **Staging vs. launched regions.** The step slices exchange per-block
  materialized tiles (`KTrans`, `VTile`, `KVPrev`, `KVOut`) in a flat
  `[rows, width]` layout. The `hK`/`hV` hypotheses are exactly the bridge
  asserting those tiles hold block `m` of the launched `K`/`V` tensors; they are
  assumptions, not lemmas.
* **The backward kernels.** Only lowering. The `DQ`/`DK`/`DV` writeback lemmas
  present below are not headline conjuncts and their gradient inputs are fiction
  regions.
* **Region distinctness.** No `≠` hypotheses are stated and none are needed:
  each slice performs all of its loads before its single store and every
  `expected` reads the *initial* state, so aliasing cannot falsify a face.

Side conditions: the staged-tile offsets are injective unconditionally from the
`Fin` bounds (`kvOffset_injective`, `oInterOffset_injective`); the launched `Out`
offset needs `BLOCK_MODEL ≤ e` (`bodyOutOffset_injective`).
-/

namespace VeriTile.Bench.TritonBenchG.LightningAttention

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! **★ Main theorem:** `lightning_attention_output_summary_general` —
shape-general, scoped to a single block index `m` of the forward loop. The whole
loop body (causal-masked `o_intra`, `o_inter`, their sum, and the masked store
into the launched `Out`) has a face; the `NUM_BLOCK` driver threading the carry
does not. -/

/-! # ══════════ CORRECT — genuine / dimension-general (review this) ══════════ -/

section Correct_without_Rounding

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
    ComputeCorrect.Realizes_without_Rounding
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
`KVPrev` holds the genuine `m`-block folded state `kvClosed m`, and the staged
per-block tiles `KTrans`/`VTile` read the genuine block-`m` entries of the
**launched** `K`/`V` tensors, then one loop body — `kvStepSpec`, i.e.
`KVPrev + tl.dot(k_trans, v)` — produces exactly the genuine `(m+1)`-block folded
state `kvClosed (m+1)`.

Note the two distinct roles: `VTile` is the flat per-block staging tile the slice
loads, while `V` is the launched value tensor the closed form `kvClosed` reads
through `fwdVVal`. They must **not** be instantiated with the same region — doing
so makes `hV` read the same region on both sides and drops `V` out of the
advertised "closed form over `K`, `V`" entirely. -/
theorem kvStepSpec_eq_kvClosed_succ
    (s : BlockState) (KVPrev KTrans VTile K V : RegionName)
    (n d e BLOCK BLOCK_MODEL m : Nat)
    (hPrev : ∀ idx : TileIndex [d, BLOCK_MODEL],
      s.readMem KVPrev (kvOffset BLOCK_MODEL idx)
        = kvClosed s K V n d e BLOCK BLOCK_MODEL m idx.1.val idx.2.1.val)
    (hK : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem KTrans (idx.1.val * BLOCK + j.val)
        = fwdKVal s K n d idx.1.val (m * BLOCK + j.val))
    (hV : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem VTile (j.val * BLOCK_MODEL + idx.2.1.val)
        = fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val))
    (idx : TileIndex [d, BLOCK_MODEL]) :
    kvStepSpec s KVPrev KTrans VTile d BLOCK BLOCK_MODEL idx
      = kvClosed s K V n d e BLOCK BLOCK_MODEL (m + 1) idx.1.val idx.2.1.val := by
  rw [kvClosed_succ]
  unfold kvStepSpec tileElem
  rw [hPrev idx]
  congr 1
  rw [Fin.sum_univ_eq_sum_range
    (fun j => s.readMem KTrans (idx.1.val * BLOCK + j) *
      s.readMem VTile (j * BLOCK_MODEL + idx.2.1.val)) BLOCK]
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
    ComputeCorrect.Realizes_without_Rounding
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

/-! ### ════════ The full forward loop body (causal mask + `Out` writeback) ════════

This is the whole `_fwd_kernel` loop body at a fixed block index `m`, over the
**launched** regions `Q`, `K`, `V`, `Out` at their real addresses: the three
masked tile loads, `qk = tl.dot(q, k_trans)`, the causal
`qk = tl.where(index >= 0, qk, 0)`, `o_intra = tl.dot(qk, v)`,
`o_inter = tl.dot(q, kv)`, `o = o_intra + o_inter`, and the masked
`tl.store` into `Out`. Only the carried `kv` register is presented as a
materialized buffer `KVPrev` (it is a register, so it has no address).

Two modelling points that must be read before trusting the face below.

* **The causal predicate is re-spelled, not re-interpreted.** Python computes
  `index = off_block[:, None] - off_block[None, :]` in *signed* int32 and tests
  `index >= 0`. This model's `-` on `.nat` tiles truncates at zero, which would
  make `index >= 0` a tautology — so the slice spells the same predicate as the
  direct comparison `offs_r[:, None] >= offs_r[None, :]`, which is equal to
  `r - j ≥ 0` over ℤ for the non-negative lane indices involved. (Note that
  `lightning_attention_forward_surface` keeps the *literal* Python spelling, and
  therefore models that mask as a no-op; the surface carries only the lowering
  claim, never a semantic one, but a reader comparing the two should know why they
  differ. Note also that Python computes `index` **once, before** the loop, from
  `off_block = tl.arange(0, BLOCK)`, so the predicate is block-relative `j ≤ r`
  and does not drift as `off_block += BLOCK`.)
* **Full-block scope.** `hFullBlock : m·BLOCK + BLOCK ≤ n` restricts the face to
  blocks that lie entirely inside the sequence. Under it all three load masks and
  the store mask are satisfied on every lane, so the loads' `other=0.0` never
  fires. `NUM_BLOCK = cdiv(n, BLOCK)` (`lightning_attention.py:378`), so this
  holds for every block except a partial tail one; the partial tail block is part
  of the trusted boundary, matching the precedent set by the
  `fused_recurrent_retention` step slices. -/

def lightning_attention_forward_body_slice
    (Q K V KVPrev Out : RegionName)
    (m n d e BLOCK BLOCK_MODEL : Nat) : ComputeKernel := triton {
  off_bh = tl.program_id(0)
  off_e = tl.program_id(1)
  qk_offset = off_bh * $(n) * $(d)
  v_offset = off_bh * $(n) * $(e)
  o_offset = off_bh * $(n) * $(e)
  e_offset = off_e * $(BLOCK_MODEL)
  offs_a = tl.arange(0, $(d))
  offs_c = tl.arange(0, $(BLOCK_MODEL))
  offs_r = tl.arange(0, $(BLOCK))
  off_block = $(m) * $(BLOCK) + tl.arange(0, $(BLOCK))
  q = tl.load(Q + qk_offset + off_block[:, None] * $(d) + offs_a[None, :],
    mask=off_block[:, None] < $(n), other=0.0)
  k_trans = tl.load(K + qk_offset + off_block[None, :] * $(d) + offs_a[:, None],
    mask=off_block[None, :] < $(n), other=0.0)
  v = tl.load(V + v_offset + off_block[:, None] * $(e) + e_offset + offs_c[None, :],
    mask=off_block[:, None] < $(n), other=0.0)
  kv = tl.load(KVPrev + offs_a[:, None] * $(BLOCK_MODEL) + offs_c[None, :])
  qk = tl.dot(q, k_trans)
  qk = tl.where(offs_r[:, None] >= offs_r[None, :], qk, 0.0)
  o_intra = tl.dot(qk, v)
  o_inter = tl.dot(q, kv)
  o = o_intra + o_inter
  tl.store(Out + o_offset + off_block[:, None] * $(e) + e_offset + offs_c[None, :],
    (o).to(Out.dtype.element_ty), mask=off_block[:, None] < $(n))
}

theorem lightning_attention_forward_body_slice_toAlgorithm_supported
    (Q K V KVPrev Out : RegionName) (m n d e BLOCK BLOCK_MODEL : Nat) :
    ∃ alg, (lightning_attention_forward_body_slice Q K V KVPrev Out
      m n d e BLOCK BLOCK_MODEL).toAlgorithm? = Except.ok alg := by
  simp [lightning_attention_forward_body_slice, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-- The kernel's store address for block-`m` lane `(r, c)`:
`o_offset + (m·BLOCK + r)·e + e_offset + c`, i.e. global time row `m·BLOCK + r`,
value channel `off_e·BLOCK_MODEL + c`. -/
def bodyOutOffset (s : BlockState) (m n e BLOCK BLOCK_MODEL : Nat)
    (idx : TileIndex [BLOCK, BLOCK_MODEL]) : Nat :=
  s.pids 0 * n * e + (m * BLOCK + idx.1.val) * e + s.pids 1 * BLOCK_MODEL +
    idx.2.1.val

/-- One `qk` entry: the query row at global time `m·BLOCK + r` dotted with the
key row at global time `m·BLOCK + j` (`Σ_a q[·,a]·k[·,a]`). -/
noncomputable def bodyIntraRow (s : BlockState) (Q K : RegionName)
    (m n d BLOCK : Nat) (r j : Fin BLOCK) : ℝ :=
  ∑ a : Fin d,
    fwdKVal s Q n d a.val (m * BLOCK + r.val) *
      fwdKVal s K n d a.val (m * BLOCK + j.val)

/-- One `o_inter` entry: `Σ_a q[m·BLOCK+r, a] · kv[a, c]` against the carried
state materialized in `KVPrev`. -/
noncomputable def bodyInterRow (s : BlockState) (Q KVPrev : RegionName)
    (m n d BLOCK BLOCK_MODEL : Nat) (r : Fin BLOCK) (c : Fin BLOCK_MODEL) : ℝ :=
  ∑ a : Fin d,
    fwdKVal s Q n d a.val (m * BLOCK + r.val) *
      s.readMem KVPrev (a.val * BLOCK_MODEL + c.val)

/-- The stored `o` value in the `WithBot ℝ` cell layer the semantics works in —
`Tile.select`-then-`Tile.dot` for `o_intra`, plus the `o_inter` dot. Written in
the same combinator shape the evaluator produces so the exec walk closes; the
readable `ℝ` form is `bodyStepSpec`, see `bodyStoreValue_eq_bodyStepSpec`. -/
noncomputable def bodyStoreValue (s : BlockState) (Q K V KVPrev : RegionName)
    (m n d e BLOCK BLOCK_MODEL : Nat)
    (idx : TileIndex [BLOCK, BLOCK_MODEL]) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun x y => x + y)
      (@Finset.sum (Fin BLOCK) (WithBot ℝ) _ Finset.univ (fun j =>
        Option.map₂ (fun x y => x * y)
          (if j ≤ idx.1 then
              some (bodyIntraRow s Q K m n d BLOCK idx.1 j)
            else (some (0.0 : ℝ) : WithBot ℝ))
          (some (fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val)))))
      (some (bodyInterRow s Q KVPrev m n d BLOCK BLOCK_MODEL idx.1 idx.2.1)))

/-- **The loop body's arithmetic, in `ℝ`.** `o[r,c] = o_intra + o_inter` with the
causal mask explicit: `Σ_{j} (j ≤ r ? qk[r,j] : 0) · v[j,c] + Σ_a q[r,a]·kv[a,c]`.
This is the clause that was missing entirely — the `tl.where`-masked intra-block
product and the `Out` writeback value. -/
noncomputable def bodyStepSpec (s : BlockState) (Q K V KVPrev : RegionName)
    (m n d e BLOCK BLOCK_MODEL : Nat)
    (idx : TileIndex [BLOCK, BLOCK_MODEL]) : ℝ :=
  (∑ j : Fin BLOCK,
      (if j.val ≤ idx.1.val then bodyIntraRow s Q K m n d BLOCK idx.1 j else 0) *
        fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val))
    + bodyInterRow s Q KVPrev m n d BLOCK BLOCK_MODEL idx.1 idx.2.1

theorem bodyStoreValue_eq_bodyStepSpec
    (s : BlockState) (Q K V KVPrev : RegionName)
    (m n d e BLOCK BLOCK_MODEL : Nat)
    (idx : TileIndex [BLOCK, BLOCK_MODEL]) :
    bodyStoreValue s Q K V KVPrev m n d e BLOCK BLOCK_MODEL idx
      = bodyStepSpec s Q K V KVPrev m n d e BLOCK BLOCK_MODEL idx := by
  have hz : (0.0 : ℝ) = 0 := by norm_num
  unfold bodyStoreValue bodyStepSpec
  have hrow : ∀ j : Fin BLOCK,
      Option.map₂ (fun x y => x * y)
          (if j ≤ idx.1 then
              some (bodyIntraRow s Q K m n d BLOCK idx.1 j)
            else (some (0.0 : ℝ) : WithBot ℝ))
          (some (fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val)))
        = (some ((if j.val ≤ idx.1.val then bodyIntraRow s Q K m n d BLOCK idx.1 j
                else 0) *
              fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val))
            : WithBot ℝ) := by
    intro j
    by_cases hj : j ≤ idx.1
    · have hj' : j.val ≤ idx.1.val := hj
      simp [hj, hj']
    · have hj' : ¬ j.val ≤ idx.1.val := hj
      simp [hj, hj', hz]
  simp only [hrow]
  simp

/-- `bodyOutOffset` is injective given `BLOCK_MODEL ≤ e` (the channel block fits
the value width). Provenance: the launch sets
`BLOCK_MODEL = min(next_power_of_2(e), 32)` (`lightning_attention.py:380`), so at
the benchmark's `e = 128` it is `32 ≤ 128`. -/
theorem bodyOutOffset_injective (s : BlockState) (m n e BLOCK BLOCK_MODEL : Nat)
    (hBM : BLOCK_MODEL ≤ e) :
    Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        bodyOutOffset s m n e BLOCK BLOCK_MODEL idx) := by
  rintro ⟨⟨ra, hra⟩, ⟨ca, hca⟩, _⟩ ⟨⟨rb, hrb⟩, ⟨cb, hcb⟩, _⟩ h
  simp only [bodyOutOffset] at h
  have step : ∀ x y : Nat, x < y → (m * BLOCK + x) * e + e ≤ (m * BLOCK + y) * e := by
    intro x y hxy
    have hmul : (m * BLOCK + x + 1) * e ≤ (m * BLOCK + y) * e :=
      Nat.mul_le_mul_right e (by omega)
    calc (m * BLOCK + x) * e + e = (m * BLOCK + x + 1) * e := by ring
      _ ≤ (m * BLOCK + y) * e := hmul
  have key : ra = rb ∧ ca = cb := by
    rcases Nat.lt_trichotomy ra rb with hlt | heq | hgt
    · have hs := step ra rb hlt
      omega
    · subst heq
      exact ⟨rfl, by omega⟩
    · have hs := step rb ra hgt
      omega
  obtain ⟨hr, hc⟩ := key
  subst hr; subst hc; rfl

theorem lightning_attention_forward_body_slice_correct
    (Q K V KVPrev Out : RegionName) (m n d e BLOCK BLOCK_MODEL : Nat)
    (s : BlockState)
    (hFullBlock : m * BLOCK + BLOCK ≤ n)
    (hBM : BLOCK_MODEL ≤ e) :
    ∀ idx : TileIndex [BLOCK, BLOCK_MODEL],
      (exec (lightning_attention_forward_body_slice Q K V KVPrev Out
          m n d e BLOCK BLOCK_MODEL) s).map
          (·.readMem Out (bodyOutOffset s m n e BLOCK BLOCK_MODEL idx))
        = some (bodyStoreValue s Q K V KVPrev m n d e BLOCK BLOCK_MODEL idx) := by
  intro idx
  have hlt : ∀ r : Fin BLOCK, (m * BLOCK + r.val < n) = True := by
    intro r
    have := r.isLt
    simp only [eq_iff_iff, iff_true]
    omega
  have hOffsetInj : Function.Injective
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        s.pids 0 * n * e + (m * BLOCK + idx.1.val) * e + s.pids 1 * BLOCK_MODEL +
          idx.2.1.val) := by
    simpa [bodyOutOffset] using bodyOutOffset_injective s m n e BLOCK BLOCK_MODEL hBM
  simp [exec, lightning_attention_forward_body_slice, stepStmts, stepStmt,
    evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop,
    Tile.expandDim, Tile.ptrAdd, Tile.dot, Tile.select, Tile.remap,
    NumericDType.add, NumericDType.mul, ComparableDType.ge, ComparableDType.lt,
    ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    bodyOutOffset, bodyStoreValue, bodyIntraRow, bodyInterRow, fwdKVal,
    fwdVVal, hlt, TileShape.dropInsertedIndex]
  rw [BlockState.scatter_readback_nd _ _ _ hOffsetInj idx]
  rfl

/-- **Genuine closed form for the launched output row** — the causal linear
attention `_fwd_kernel` computes, at global time row `t` and value channel
`off_e·BLOCK_MODEL + c`:

```
o[t, c] = Σ_{s ≤ t} (Σ_a q[t,a]·k[s,a]) · v[s,c].
```

A standalone specification over the input regions `Q`, `K`, `V` — never a
read-back of the kernel's own output. -/
noncomputable def fwdOutClosed (s : BlockState) (Q K V : RegionName)
    (n d e BLOCK_MODEL : Nat) (t : Nat) (c : Nat) : ℝ :=
  ∑ keyRow ∈ Finset.range (t + 1),
    (∑ a : Fin d, fwdKVal s Q n d a.val t * fwdKVal s K n d a.val keyRow) *
      fwdVVal s V n e BLOCK_MODEL c keyRow

/-- **The loop body computes the causal linear-attention row.** Under the carry
invariant `KVPrev = kvClosed m`, the body's `o_intra + o_inter` is exactly
`o[t,c] = Σ_{s ≤ t}(Σ_a q[t,a]·k[s,a])·v[s,c]` at `t = m·BLOCK + r`: the causal
mask supplies the in-block keys `s ∈ [m·BLOCK, t]` and the carried state supplies
all earlier ones `s < m·BLOCK`. -/
theorem bodyStepSpec_eq_fwdOutClosed
    (s : BlockState) (Q K V KVPrev : RegionName)
    (m n d e BLOCK BLOCK_MODEL : Nat)
    (hPrev : ∀ idx : TileIndex [d, BLOCK_MODEL],
      s.readMem KVPrev (kvOffset BLOCK_MODEL idx)
        = kvClosed s K V n d e BLOCK BLOCK_MODEL m idx.1.val idx.2.1.val)
    (idx : TileIndex [BLOCK, BLOCK_MODEL]) :
    bodyStepSpec s Q K V KVPrev m n d e BLOCK BLOCK_MODEL idx
      = fwdOutClosed s Q K V n d e BLOCK_MODEL
          (m * BLOCK + idx.1.val) idx.2.1.val := by
  unfold bodyStepSpec fwdOutClosed
  -- the carried state contributes exactly the key rows `< m·BLOCK`
  have hInter :
      bodyInterRow s Q KVPrev m n d BLOCK BLOCK_MODEL idx.1 idx.2.1
        = ∑ keyRow ∈ Finset.range (m * BLOCK),
            (∑ a : Fin d,
                fwdKVal s Q n d a.val (m * BLOCK + idx.1.val) *
                  fwdKVal s K n d a.val keyRow) *
              fwdVVal s V n e BLOCK_MODEL idx.2.1.val keyRow := by
    unfold bodyInterRow
    have h1 : ∀ a : Fin d,
        fwdKVal s Q n d a.val (m * BLOCK + idx.1.val) *
            s.readMem KVPrev (a.val * BLOCK_MODEL + idx.2.1.val)
          = ∑ keyRow ∈ Finset.range (m * BLOCK),
              fwdKVal s Q n d a.val (m * BLOCK + idx.1.val) *
                (fwdKVal s K n d a.val keyRow *
                  fwdVVal s V n e BLOCK_MODEL idx.2.1.val keyRow) := by
      intro a
      have := hPrev (a, idx.2.1, PUnit.unit)
      simp only [kvOffset] at this
      rw [this, kvClosed, Finset.mul_sum]
    rw [Finset.sum_congr rfl (fun a (_ : a ∈ Finset.univ) => h1 a), Finset.sum_comm]
    refine Finset.sum_congr rfl fun keyRow _ => ?_
    rw [Finset.sum_mul]
    refine Finset.sum_congr rfl fun a _ => ?_
    ring
  -- the causal mask contributes exactly the key rows in `[m·BLOCK, t]`
  have hIntra :
      (∑ j : Fin BLOCK,
          (if j.val ≤ idx.1.val then bodyIntraRow s Q K m n d BLOCK idx.1 j
            else 0) *
            fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val))
        = ∑ i ∈ Finset.range (idx.1.val + 1),
            (∑ a : Fin d,
                fwdKVal s Q n d a.val (m * BLOCK + idx.1.val) *
                  fwdKVal s K n d a.val (m * BLOCK + i)) *
              fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + i) := by
    have hg : ∀ j : Fin BLOCK,
        (if j.val ≤ idx.1.val then bodyIntraRow s Q K m n d BLOCK idx.1 j else 0) *
            fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val)
          = (if j.val ≤ idx.1.val then
                (∑ a : Fin d,
                    fwdKVal s Q n d a.val (m * BLOCK + idx.1.val) *
                      fwdKVal s K n d a.val (m * BLOCK + j.val)) *
                  fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val)
              else 0) := by
      intro j
      unfold bodyIntraRow
      split <;> simp
    rw [Finset.sum_congr rfl (fun j (_ : j ∈ Finset.univ) => hg j)]
    rw [Fin.sum_univ_eq_sum_range
      (fun i => if i ≤ idx.1.val then
          (∑ a : Fin d,
              fwdKVal s Q n d a.val (m * BLOCK + idx.1.val) *
                fwdKVal s K n d a.val (m * BLOCK + i)) *
            fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + i)
        else 0) BLOCK]
    rw [← Finset.sum_filter]
    have hfilter : (Finset.range BLOCK).filter (fun i => i ≤ idx.1.val)
        = Finset.range (idx.1.val + 1) := by
      ext i
      simp only [Finset.mem_filter, Finset.mem_range]
      have := idx.1.isLt
      omega
    rw [hfilter]
  -- split the closed form's key range at the block boundary
  have hSplit :
      (∑ keyRow ∈ Finset.range (m * BLOCK + idx.1.val + 1),
          (∑ a : Fin d,
              fwdKVal s Q n d a.val (m * BLOCK + idx.1.val) *
                fwdKVal s K n d a.val keyRow) *
            fwdVVal s V n e BLOCK_MODEL idx.2.1.val keyRow)
        = (∑ keyRow ∈ Finset.range (m * BLOCK),
              (∑ a : Fin d,
                  fwdKVal s Q n d a.val (m * BLOCK + idx.1.val) *
                    fwdKVal s K n d a.val keyRow) *
                fwdVVal s V n e BLOCK_MODEL idx.2.1.val keyRow)
          + ∑ i ∈ Finset.range (idx.1.val + 1),
              (∑ a : Fin d,
                  fwdKVal s Q n d a.val (m * BLOCK + idx.1.val) *
                    fwdKVal s K n d a.val (m * BLOCK + i)) *
                fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + i) := by
    rw [show m * BLOCK + idx.1.val + 1 = m * BLOCK + (idx.1.val + 1) from by omega,
      Finset.sum_range_add]
  rw [hIntra, hInter, hSplit]
  ring

/-- **★ The `Out` writeback face.** Genuine and dimension-general: the launched
masked store of `o = o_intra + o_inter` at block `m` delivers exactly the causal
linear-attention row
`o[t, c] = Σ_{s ≤ t}(Σ_a q[t,a]·k[s,a])·v[s,c]` at global time `t = m·BLOCK + r`
and value channel `off_e·BLOCK_MODEL + c`, over the input regions `Q`, `K`, `V`.

Side conditions, all honest: `hFullBlock` (the block lies inside the sequence, so
every load/store mask lane is live — the partial tail block is trusted boundary),
`hBM` (`BLOCK_MODEL ≤ e`, giving store-offset injectivity; the launch sets
`BLOCK_MODEL = min(next_power_of_2(e), 32)`), and `hPrev` — the *assumed* loop-carry
invariant `KVPrev = kvClosed m`, which is what the unmodeled `NUM_BLOCK` driver
would have to establish. The write map is unmasked because `hFullBlock` makes the
kernel's `mask=off_block[:, None] < n` true on every lane. -/
theorem lightning_attention_forward_body_closed_form_general
    (Q K V KVPrev Out : RegionName) (m n d e BLOCK BLOCK_MODEL : Nat)
    (s : BlockState)
    (hFullBlock : m * BLOCK + BLOCK ≤ n)
    (hBM : BLOCK_MODEL ≤ e)
    (hPrev : ∀ idx : TileIndex [d, BLOCK_MODEL],
      s.readMem KVPrev (kvOffset BLOCK_MODEL idx)
        = kvClosed s K V n d e BLOCK BLOCK_MODEL m idx.1.val idx.2.1.val) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := lightning_attention_forward_body_slice Q K V KVPrev Out
        m n d e BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        some (Out, bodyOutOffset s m n e BLOCK BLOCK_MODEL idx))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        fwdOutClosed s Q K V n d e BLOCK_MODEL
          (m * BLOCK + idx.1.val) idx.2.1.val) := by
  have hfun :
      (fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
          fwdOutClosed s Q K V n d e BLOCK_MODEL
            (m * BLOCK + idx.1.val) idx.2.1.val)
        = fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
            bodyStoreValue s Q K V KVPrev m n d e BLOCK BLOCK_MODEL idx := by
    funext idx
    rw [bodyStoreValue_eq_bodyStepSpec,
      bodyStepSpec_eq_fwdOutClosed s Q K V KVPrev m n d e BLOCK BLOCK_MODEL
        hPrev idx]
  rw [hfun]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [lightning_attention_forward_body_slice, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have h := lightning_attention_forward_body_slice_correct Q K V KVPrev Out
    m n d e BLOCK BLOCK_MODEL s hFullBlock hBM idx
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
      (kernel := lightning_attention_bwd_dq_inter_dot_slice DO KVTrans DQInter
        BLOCK E WIDTH)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK, WIDTH] =>
        some (DQInter, dqInterOffset WIDTH idx))
      (expected := fun idx : TileIndex [BLOCK, WIDTH] =>
        dqInterDotSpec s DO KVTrans BLOCK E WIDTH idx) := by
  unfold ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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
    ComputeCorrect.Realizes_without_Rounding
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

/-- **SCOPE — this is a claim about one block index `m` of the forward loop, not
about the whole launched kernel.** The full launched surfaces appear only in the
lowering clauses; the `NUM_BLOCK` fold that threads the carry across blocks is
outside every clause. The loop *body* — including `o_intra`, the causal mask, and
the masked `Out` writeback — is inside it (clause 4).

**Genuine, shape-general forward step summary** for the lightning-attention
kernel. Symbolic in every dimension (`d e BLOCK NUM_BLOCK BLOCK_MODEL`) and over
an arbitrary loop index `m` and materialized carry buffer; no dimension is
pinned. Exposes:

1. **Surface lowering** of the forward and both backward kernels (every
   dimension symbolic). Lowering is well-formedness, not correctness.
2. **`kv` carry-fold body — genuine closed form.** One `kv += tl.dot(k_trans,
   v)` body realizes its spec `kvStepSpec`, and under the *assumed* loop-carry
   invariant (`KVPrev = kvClosed m`, and `KTrans`/`VTile` staging block `m` of the
   launched `K`/`V`) that spec equals exactly `kvClosed (m+1)` — the running
   `Σ kᵀ·v` over the first `m+1` key blocks. This is a genuine standalone closed
   form over the launched input regions `K` and `V`, never an `exec` read-back.
3. **`o_inter = tl.dot(q, kv)` inter-block producer** realizes its spec
   `oInterDotSpec` — `Σ_a q[r,a]·kv[a,c]` against the carried state — which is
   **one of the two halves** of the causal linear-attention output row.
4. **★ The whole loop body → the launched `Out`.** The causal-masked intra-block
   product `o_intra = tl.dot(where(j ≤ r, qk, 0), v)`, the inter-block product
   `o_inter`, their sum, and the masked `tl.store` into `Out` at the launch's real
   address realize exactly `fwdOutClosed` — the causal linear-attention row
   `o[t,c] = Σ_{s ≤ t}(Σ_a q[t,a]·k[s,a])·v[s,c]` at `t = m·BLOCK + r` — over the
   input regions `Q`, `K`, `V`. The causal mask supplies the in-block keys and the
   carried state (via `hPrev`) supplies all earlier ones. This clause needs
   `hFullBlock` and `hBM`; see the module docstring for both.

What this theorem does **not** say: nothing about the
`NUM_BLOCK` loop that threads the carry, and nothing about the backward
gradients beyond lowering. See the module docstring's modeling boundary.

Side conditions. The staged-tile offsets are injective unconditionally from the
`Fin` bounds; the launched `Out` offset needs `hBM : BLOCK_MODEL ≤ e`, and clause 4
is scoped to full blocks by `hFullBlock : m·BLOCK + BLOCK ≤ n` (both with launch
provenance on their binders). **Load-bearing:** `hPrev`/`hK`/`hV` are *assumed*
loop-carry / staging hypotheses, not lemmas — `hPrev` is what clause 4 needs and
what the unmodeled `NUM_BLOCK` driver would have to establish. -/
specification lightning_attention_output_summary_general
    (Q K V Out DO DQ DK DV KVPrev KTrans VTile KVOut OInter : RegionName)
    (s : BlockState)
    (_b h n d e BLOCK NUM_BLOCK BLOCK_MODEL m : Nat)
    -- block `m` lies inside the sequence; `NUM_BLOCK = cdiv(n, BLOCK)`
    -- (`lightning_attention.py:378`) so only a partial tail block can violate it
    (hFullBlock : m * BLOCK + BLOCK ≤ n)
    -- the channel block fits the value width; the launch sets
    -- `BLOCK_MODEL = min(next_power_of_2(e), 32)` (`lightning_attention.py:380`)
    (hBM : BLOCK_MODEL ≤ e)
    (hPrev : ∀ idx : TileIndex [d, BLOCK_MODEL],
      s.readMem KVPrev (kvOffset BLOCK_MODEL idx)
        = kvClosed s K V n d e BLOCK BLOCK_MODEL m idx.1.val idx.2.1.val)
    (hK : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem KTrans (idx.1.val * BLOCK + j.val)
        = fwdKVal s K n d idx.1.val (m * BLOCK + j.val))
    (hV : ∀ (idx : TileIndex [d, BLOCK_MODEL]) (j : Fin BLOCK),
      s.readMem VTile (j.val * BLOCK_MODEL + idx.2.1.val)
        = fwdVVal s V n e BLOCK_MODEL idx.2.1.val (m * BLOCK + j.val)) :
    -- (1) surface lowering of forward + both backward kernels (symbolic dims)
    (∃ alg, (lightning_attention_forward_surface Q K V Out _b h n d e BLOCK
      NUM_BLOCK BLOCK_MODEL).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (lightning_attention_bwd_intra_surface Q K V DO DQ DK DV
      _b h n d e BLOCK NUM_BLOCK BLOCK NUM_BLOCK).toAlgorithm? = Except.ok alg) ∧
    (∃ alg, (lightning_attention_bwd_inter_surface Q K V DO DQ DK DV
      _b h n d e BLOCK NUM_BLOCK BLOCK NUM_BLOCK).toAlgorithm? = Except.ok alg) ∧
    -- (2) kv carry-fold body realizes kvStepSpec ...
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := lightning_attention_forward_kv_step_slice KVPrev KTrans VTile KVOut
        d BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := fun idx : TileIndex [d, BLOCK_MODEL] =>
        some (KVOut, kvOffset BLOCK_MODEL idx))
      (expected := fun idx : TileIndex [d, BLOCK_MODEL] =>
        kvStepSpec s KVPrev KTrans VTile d BLOCK BLOCK_MODEL idx)) ∧
    -- ... and that spec is the genuine closed-form kvClosed (m+1) under the carry invariant
    (∀ idx : TileIndex [d, BLOCK_MODEL],
      kvStepSpec s KVPrev KTrans VTile d BLOCK BLOCK_MODEL idx
        = kvClosed s K V n d e BLOCK BLOCK_MODEL (m + 1) idx.1.val idx.2.1.val) ∧
    -- (3) o_inter producer realizes its genuine spec oInterDotSpec
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := lightning_attention_forward_o_inter_dot_slice Q KVPrev OInter
        BLOCK d BLOCK_MODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        some (OInter, oInterOffset BLOCK_MODEL idx))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        oInterDotSpec s Q KVPrev BLOCK d BLOCK_MODEL idx)) ∧
    -- (4) ★ the WHOLE loop body — the causal-masked `o_intra`, `o_inter`, their
    --     sum, and the masked `tl.store` into the launched `Out` — realizes the
    --     genuine causal linear-attention output row
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := lightning_attention_forward_body_slice Q K V KVPrev Out
        m n d e BLOCK BLOCK_MODEL)
      (initialState := s)
      (write := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        some (Out, bodyOutOffset s m n e BLOCK BLOCK_MODEL idx))
      (expected := fun idx : TileIndex [BLOCK, BLOCK_MODEL] =>
        fwdOutClosed s Q K V n d e BLOCK_MODEL
          (m * BLOCK + idx.1.val) idx.2.1.val)) := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact lightning_attention_forward_surface_toAlgorithm_supported Q K V Out
      _b h n d e BLOCK NUM_BLOCK BLOCK_MODEL
  · exact lightning_attention_bwd_intra_surface_toAlgorithm_supported Q K V
      DO DQ DK DV _b h n d e BLOCK NUM_BLOCK BLOCK NUM_BLOCK
  · exact lightning_attention_bwd_inter_surface_toAlgorithm_supported Q K V
      DO DQ DK DV _b h n d e BLOCK NUM_BLOCK BLOCK NUM_BLOCK
  · exact lightning_attention_forward_kv_step_slice_compute_correct KVPrev KTrans
      VTile KVOut d BLOCK BLOCK_MODEL s (kvOffset_injective d BLOCK_MODEL)
  · intro idx
    exact kvStepSpec_eq_kvClosed_succ s KVPrev KTrans VTile K V
      n d e BLOCK BLOCK_MODEL m hPrev hK hV idx
  · exact lightning_attention_forward_o_inter_dot_slice_compute_correct Q KVPrev
      OInter BLOCK d BLOCK_MODEL s (oInterOffset_injective BLOCK BLOCK_MODEL)
  · exact lightning_attention_forward_body_closed_form_general Q K V KVPrev Out
      m n d e BLOCK BLOCK_MODEL s hFullBlock hBM hPrev

end Correct_without_Rounding

end VeriTile.Bench.TritonBenchG.LightningAttention

