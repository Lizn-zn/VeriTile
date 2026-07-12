import VeriTile.Triton

/-!
# `attn_fwd_triton` — strict per-kernel correctness

`attn_fwd_triton.py`'s `_attn_fwd` is a (full, `STAGE = 3`) FlashAttention
forward kernel: program `(start_m, off_hz)` streams `K`/`V` blocks for one
`(batch·head, query block)` tile, maintaining the online-softmax running max
`m_i`, denominator `l_i`, and accumulator `acc` (with `q_scale · k_scale`
quantization), then stores the normalized `acc / l_i` to `Out`, masked to the
first 96 head lanes and `offs_m < N_CTX`.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (`_attn_fwd[grid](...)` with `grid = (cdiv(N_CTX,
BLOCK_M), Z·H, 1)`, the scheduling over query blocks and `(batch, head)`, and
how the runtime composes per-program writes into the output buffer) is the
*trusted boundary*, not a proof obligation here. Because the program ids
`start_m`/`off_hz` are universally quantified (via `s`), the per-program
statements cover every program of the grid.

## Proof architecture

```
attn_fwd_triton_output_summary_general                     ← TOP THEOREM
  ├─ attn_fwd_triton_surface_toAlgorithm_supported          surface lowers to the algorithm layer
  └─ aftg_exec_generalG                                     full surface executes; Out = closed form
       ├─ aftgPreLoopG_invariant                            preLoop establishes the ⊥-seed invariant
       ├─ aftgLoopBodyG / aftg_attn_stepG                   per-block streaming loop step (forRange_inv)
       └─ aftgPostLoopG_eval                                postLoop normalizes + stores; Out readback
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float; the `exp2`, the `tl.dot`
`float16` accumulation, and `q_scale · k_scale` quantization are not modeled at
the bit level); `@triton.autotune`/`num_warps`/`num_stages` are not modeled.
The verified result is the **genuine closed form**: the full surface (preLoop +
the inlined `forRange` streaming loop + postLoop) is executed and its `Out` store
is proven equal to `attnFwdTritonOutSpecG` — masked base-2 (`exp2`) per-key-scale
**causal** softmax (`keyScale = aftKeyScale = q_scale · k_scale[block]`,
`keep = causalKeep qStart`, head-active `q`/`k`/`v` masks at `e/d ≥ 96`) — at
every active output lane (`attn_fwd_triton_output_summary_general`). The
inner online-softmax recurrence (`m_i`/`l_i`/`acc` updates, the final `acc / l_i`
normalization) is reconciled to this closed form via the ⊥-seeded `aftgStateBotG`
streaming fold and `aftgStateBotG_full_eq_spec`; the loop is driven by `forRange_inv`
+ `aftg_attn_stepG`. The result is **dimension-general**: it holds for arbitrary
symbolic block/head dimensions at the contiguous layout (`stride_qm = HEAD_DIM`,
`stride_qk = 1`, `stride_kn = HEAD_DIM`). Its side conditions (documented at the
theorem's banner) are: positive blocks (`BLOCK_DMODEL`, `BLOCK_N`, `BLOCK_M > 0`),
`N_CTX = BLOCK_N · numKVBlocks` with `numKVBlocks > 0`, clean input
(`undef ≡ 0`), the `-1e6` `tl.where` mask sentinel being inert
(`aftgScoreBoundG` — every kept score `≥ -1e6`), and output-offset injectivity.
The Python test-shape layout (`Z = 2`, `H = 4`,
`N_CTX = HEAD_DIM = BLOCK_M = BLOCK_DMODEL = 128`, `BLOCK_N = 64`,
`HEAD_ACTIVE = 96`, `STAGE = 3`) is only one instance of this general theorem.

## Translation-surface blocker

Translation-surface blocker: the `_attn_fwd_inner` helper JIT (both call
sites) is inlined into the port's single streaming-loop surface, and the
Python-hard-coded head constants (`tl.arange(0, 128)`, the `< 96` head mask,
`tl.zeros([BLOCK_M, 128])`) are generalized to the `BLOCK_DMODEL` /
`HEAD_ACTIVE` binders — the Python literals are the `128`/`96` instantiation
of the dimension-general top theorem. The Lean surface is therefore not a
line-for-line textual match of the Python `_attn_fwd` body, and the textual
py↔lean scans in `bench/audit_tritonbench_g.sh` exempt this port on this
marker (registered in `proof_blockers.md`).
-/

namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-! **★ Main theorem:** `attn_fwd_triton_output_summary_general` -/

section Correct_without_Rounding

/-- Full Lean port of `attn_fwd_triton.py`'s `_attn_fwd` (`STAGE = 3`).

The upstream kernel runs the K/V streaming-softmax loop through a separate
`@triton.jit` helper `_attn_fwd_inner`, invoked twice: stage `4 - STAGE = 1`
streams the strictly-below-diagonal key blocks `range(0, start_m·BLOCK_M)` with
no causal mask, and stage `2` streams the diagonal block
`range(start_m·BLOCK_M, (start_m+1)·BLOCK_M)` under the causal predicate
`offs_m[:, None] ≥ start_n + offs_n[None, :]`.

The DSL has no cross-`@triton.jit` function-call surface, so the helper body is
inlined here as a single `forRange` loop `range(0, N_CTX, BLOCK_N)` with the
causal `where` `offs_m[:, None] ≥ start_n + offs_n[None, :]` applied to every
block. This faithfully composes the two staged helper calls of `STAGE = 3`: on a
stage-1 block (strictly below the diagonal) every lane satisfies
`offs_m ≥ start_n + offs_n`, so the causal `where` is a no-op there, matching the
unmasked stage-1 helper; on the diagonal block it is the stage-2 mask; and on
the strictly-above-diagonal blocks (which the two-call kernel never visits) the
`where` zeroes every probability (`p = where(mask, p, 0)`), so they contribute
nothing to `acc`/`l_i` — making the full-range loop equal to the kernel's
`range(0, (start_m+1)·BLOCK_M)` traversal. -/
def attn_fwd_triton_surface
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      _stride_kz _stride_kh stride_kn _stride_kk
      _stride_vz _stride_vh _stride_vk _stride_vn
      _stride_oz _stride_oh _stride_om _stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh)
  vk_offset = qvk_offset // $(stride_qm)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + qvk_offset + offs_n[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  O_block_ptr = Out + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
  q_scale = tl.load(Q_scale_ptr)
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    mask = offs_m[:, None] >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk, -1000000.0)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    p = tl.where(mask, p, 0.0)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(HEAD_DIM)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(HEAD_DIM)
  }
  acc = acc / l_i[:, None]
  tl.store(O_block_ptr, (acc).to(Out.type.element_ty),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}

/-- The full staged `attn_fwd_triton` surface lowers to the algorithm layer. -/
theorem attn_fwd_triton_surface_toAlgorithm_supported
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    ∃ alg, (attn_fwd_triton_surface Q K V Q_scale K_scale Out stride_qz
      stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om
      stride_on Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
      STAGE).toAlgorithm? = Except.ok alg := by
  simp [attn_fwd_triton_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Surface transcription/proof-oriented final output-store slice of `attn_fwd_triton.py`'s
`_attn_fwd`.

The full kernel runs staged attention forward loops. This slice starts after those stages have produced a
precomputed normalized `Acc` tile and proves the final masked writeback into
`Out`, preserving the source store address and mask
`(offs_m < N_CTX) & (offs_k < 96)`. The inner `tl.float32` accumulator and
`p.to(tl.float16)` dot-input cast are outside this slice. -/
def attn_fwd_triton_final_store_slice
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat) :
    ComputeKernel := triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  mask = (offs_m[:, None] < $(N_CTX)) & (offs_k[None, :] < $(HEAD_ACTIVE))
  acc = tl.load(Acc + off_z * $(stride_acc_z) + off_h * $(stride_acc_h) +
      offs_m[:, None] * $(stride_acc_m) + offs_k[None, :] * $(stride_acc_k),
      mask=mask, other=0.0)
  tl.store(Out + off_z.to(tl.int64) * $(stride_qz) + off_h.to(tl.int64) * $(stride_qh) +
      offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk),
      (acc).to(Out.dtype.element_ty), mask=mask)
}

def offZ (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 / H

def offH (s : BlockState) (H : Nat) : Nat :=
  s.pids 1 % H

def mIndex (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

def kIndex (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  idx.2.1.val

def active
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Prop :=
  mIndex s BLOCK_M idx.1 < N_CTX ∧ kIndex idx < HEAD_ACTIVE

instance activeDecidable
    (s : BlockState) (N_CTX HEAD_ACTIVE BLOCK_M BLOCK_DMODEL : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) :
    Decidable (active s N_CTX HEAD_ACTIVE BLOCK_M idx) := by
  unfold active
  infer_instance

def accOffset
    (s : BlockState)
    (H stride_acc_z stride_acc_h stride_acc_m stride_acc_k BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_acc_z + offH s H * stride_acc_h +
    mIndex s BLOCK_M idx.1 * stride_acc_m + kIndex idx * stride_acc_k

def outOffset
    (s : BlockState)
    (H stride_qz stride_qh stride_qm stride_qk BLOCK_M : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : Nat :=
  offZ s H * stride_qz + offH s H * stride_qh +
    mIndex s BLOCK_M idx.1 * stride_qm + kIndex idx * stride_qk

/-- Algorithm-layer correctness for the final output store. -/
theorem attn_fwd_triton_final_store_slice_correct
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
      let outAddr := outOffset s H stride_qz stride_qh stride_qm stride_qk
        BLOCK_M idx
      (exec (attn_fwd_triton_final_store_slice Acc Out H N_CTX
            HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL) s).map
          (·.readMem Out outAddr)
        = some (if active s N_CTX HEAD_ACTIVE BLOCK_M idx then
            s.readMem Acc
              (accOffset s H stride_acc_z stride_acc_h stride_acc_m
                stride_acc_k BLOCK_M idx)
          else s.readMem Out outAddr) := by
  intro idx
  simp [exec, attn_fwd_triton_final_store_slice, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, Option.bind, Option.map, Tile.bop, Tile.cop, Tile.expandDim,
        Tile.ptrAdd, NumericDType.add, NumericDType.mul,
        IntegralDType.floorDiv, IntegralDType.mod, ComparableDType.lt,
        offZ, offH, mIndex, kIndex, active, accOffset, outOffset,
        TileShape.dropInsertedIndex]
  let offsetFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx =>
      s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh +
        (s.pids 0 * BLOCK_M + idx.1.val) * stride_qm +
        idx.2.1.val * stride_qk
  let valueFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
    fun idx =>
      WithBot.unbotD 0
        (if s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧
            idx.2.1.val < HEAD_ACTIVE then
          some (s.readMem Acc
            (s.pids 1 / H * stride_acc_z + s.pids 1 % H * stride_acc_h +
              (s.pids 0 * BLOCK_M + idx.1.val) * stride_acc_m +
              idx.2.1.val * stride_acc_k))
        else some (0.0 : ℝ))
  let P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx =>
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧
        idx.2.1.val < HEAD_ACTIVE
  have hOffsetInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, offZ, offH, mIndex, kIndex] using hOutInj
  change (List.foldl
      (fun (acc : BlockState) i =>
        if P i then acc.writeMem Out (offsetFn i) (valueFn i) else acc)
      _ (TileShape.allIndices [BLOCK_M, BLOCK_DMODEL])).readMem Out
        (offsetFn idx) =
    if P idx then
      s.readMem Acc
        (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
          BLOCK_M idx)
    else s.readMem Out (offsetFn idx)
  rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hOffsetInj idx]
  by_cases hActive :
      s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
  · simp [offsetFn, valueFn, P, active, accOffset, outOffset, offZ, offH,
      mIndex, kIndex, hActive]
  · simp [offsetFn, valueFn, P, active, accOffset, outOffset, offZ, offH,
      mIndex, kIndex, hActive]

/-- Compute-facing correctness for the final output store. -/
theorem attn_fwd_triton_final_store_slice_compute_correct
    (Acc Out : RegionName)
    (H N_CTX HEAD_ACTIVE
      stride_acc_z stride_acc_h stride_acc_m stride_acc_k
      stride_qz stride_qh stride_qm stride_qk
      BLOCK_M BLOCK_DMODEL : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attn_fwd_triton_final_store_slice Acc Out H N_CTX
        HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k
        stride_qz stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          active s N_CTX HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh stride_qm stride_qk BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        s.readMem Acc
          (accOffset s H stride_acc_z stride_acc_h stride_acc_m stride_acc_k
            BLOCK_M idx)) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [attn_fwd_triton_final_store_slice]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  have h := attn_fwd_triton_final_store_slice_correct Acc Out H N_CTX
    HEAD_ACTIVE stride_acc_z stride_acc_h stride_acc_m stride_acc_k stride_qz
    stride_qh stride_qm stride_qk BLOCK_M BLOCK_DMODEL s hOutInj idx
  rw [hExec] at h
  simpa [hActive] using Option.some.inj h

/-! ## Python test-shape wrapper

`attn_fwd_triton.py`'s checked tests use `B = 2`, `H = 4`, `N_CTX = 128`,
`HEAD_DIM = 128`, `BLOCK_M = 128`, `BLOCK_N = 64`, and the source mask enables
the first 96 head lanes. Contiguous `[B, H, N_CTX, HEAD_DIM]` tensors have
strides `(65536, 16384, 128, 1)`. -/

end Correct_without_Rounding

section Correct_without_Rounding

/-! ## Genuine closed-form attention spec (exp2, causal)

`attn_fwd_triton.py`'s `_attn_fwd` (`STAGE = 3`) is **base-2** (`tl.math.exp2`)
softmax with a **scalar** score scale `q_scale · k_scale` (loaded once per
program / per key block) and a **causal** mask
`tl.where(offs_m[:, None] ≥ start_n + offs_n[None, :], qk, -inf)`. So the genuine
closed form is the predicate-masked base-2 per-key-scale attention
`attentionRealBase2PerKeyScalePred` (`VeriTile/Triton/Math/Attention.lean`),
instantiated with:

* `keep := causalKeep qStart` — key `j` contributes to query row `i` iff
  `j ≤ qStart + i` (the kernel's `offs_m ≥ start_n + offs_n` mask, with global
  `qStart = start_m · BLOCK_M`);
* a per-key score scale carrier (the scalar `q_scale · k_scale`; here abstracted
  as `keyScaleAFT`, an opaque `Fin S → ℝ`, since the kernel loads the quantization
  scalars from memory rather than fixing them at a constant like triton3).

This routes to **base-2** (NOT FlashAttention1's natural-exp
`attentionRealCausalBlock`), matching the `exp2` kernel; the streaming bridge
`attentionRealBase2PerKeyScalePred_eq_streaming` (sorry-free) delivers the
`osStep` online-softmax fold the exec loop realizes. -/

open VeriTile.Triton (attentionRealBase2PerKeyScalePred attnKeyListPred osStep
  causalKeep)

/-- One ⊥-seeded online-softmax step: like `osStep`, but the running max lives in
`WithBot ℝ` (seeded `⊥`), so `α = realExp2(m ⊖ m')` is `0` on the first block —
faithful to the kernel's `m_i` register (`tl.zeros − inf`) and `l_i`/`acc`
(seeded `0`). Ported from the flash-attn foundation. -/
noncomputable def osStepBot (st : WithBot ℝ × ℝ × ℝ) (sv : ℝ × ℝ) : WithBot ℝ × ℝ × ℝ :=
  let m := st.1; let l := st.2.1; let acc := st.2.2
  let s := sv.1; let v := sv.2
  let m' := m ⊔ ((s : ℝ) : WithBot ℝ)
  let α := (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
  let p := pow2 (s - m'.unbotD 0)
  (m', l * α + p, acc * α + p * v)

/-- The running `max` component of an `osStepBot` fold is the `WithBot ⊔`-fold. -/
theorem aftStateBot_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- **⊥-seeded consistency.** Folding `osStepBot` from a start `(m, l, acc)` whose
`l`/`acc` are anchored to the true (max-free) batch denominator `L` / accumulator
`T` via the `⊥`-aware factor `κ` (`κ ⊥ = 0`, `κ (some r) = pow2(−r)`) keeps that
invariant. Ported from the flash-attn foundation. -/
theorem osStepBot_foldl_consistent (xs : List (ℝ × ℝ)) (m : WithBot ℝ) (l acc T L : ℝ)
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let st := xs.foldl osStepBot (m, l, acc)
    st.2.1 = (st.1.elim 0 (fun r => pow2 (-r))) * (L + (xs.map (fun p => pow2 p.1)).sum) ∧
    st.2.2 = (st.1.elim 0 (fun r => pow2 (-r))) * (T + (xs.map (fun p => pow2 p.1 * p.2)).sum) := by
  induction xs generalizing m l acc T L with
  | nil => simp [hl, hacc]
  | cons x xs ih =>
    obtain ⟨s, v⟩ := x
    set m' : WithBot ℝ := m ⊔ ((s : ℝ) : WithBot ℝ) with hm'
    have hm'r : ∃ r : ℝ, m' = (r : WithBot ℝ) := by
      cases m with
      | bot => exact ⟨s, by rw [hm']; rfl⟩
      | coe a => exact ⟨max a s, by rw [hm']; rw [← WithBot.coe_max]⟩
    obtain ⟨mr, hmr⟩ := hm'r
    have hκm' : m'.elim 0 (fun r => pow2 (-r)) = pow2 (-mr) := by rw [hmr]; rfl
    have hunbot : m'.unbotD 0 = mr := by rw [hmr]; rfl
    have hp : pow2 (s - m'.unbotD 0) = pow2 (-mr) * pow2 s := by
      rw [hunbot, ← pow2_add]; ring_nf
    have hl' : l * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
        + pow2 (s - m'.unbotD 0) = pow2 (-mr) * (L + pow2 s) := by
      cases m with
      | bot =>
        rw [hmL rfl]
        have hz : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a s : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a s := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (pow2 (-a)) * (WithBot.realExp2 (WithBot.realSub (↑a) m')).unbotD 0
            = pow2 (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
          rw [show Real.exp ((a - max a s) * Real.log 2) = pow2 (a - max a s) from by
            simp [pow2, mul_comm], ← pow2_add, hmra]; ring_nf
        rw [hl, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have hacc' : acc * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0
        + pow2 (s - m'.unbotD 0) * v = pow2 (-mr) * (T + pow2 s * v) := by
      cases m with
      | bot =>
        rw [hmT rfl]
        have hz : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) m')).unbotD 0 = 0 := by
          rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
        rw [hz, mul_zero, zero_add, hp]; ring
      | coe a =>
        have hm'a : m' = ((max a s : ℝ) : WithBot ℝ) := by rw [hm']; rw [← WithBot.coe_max]
        have hmra : mr = max a s := by rw [hm'a] at hmr; exact (WithBot.coe_inj.mp hmr.symm)
        have hαa : (pow2 (-a)) * (WithBot.realExp2 (WithBot.realSub (↑a) m')).unbotD 0
            = pow2 (-mr) := by
          rw [hm'a, WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe]
          rw [show Real.exp ((a - max a s) * Real.log 2) = pow2 (a - max a s) from by
            simp [pow2, mul_comm], ← pow2_add, hmra]; ring_nf
        rw [hacc, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [mul_right_comm, hαa, hp]; ring
    have step := ih m'
      (l * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0 + pow2 (s - m'.unbotD 0))
      (acc * (WithBot.realExp2 (WithBot.realSub m m')).unbotD 0 + pow2 (s - m'.unbotD 0) * v)
      (T + pow2 s * v) (L + pow2 s) (by rw [hl', hκm']) (by rw [hacc', hκm'])
      (by rw [hmr]; simp) (by rw [hmr]; simp)
    simpa [List.foldl_cons, osStepBot, hm', List.map_cons, add_assoc] using step

/-- **The block-at-once update equals the key-by-key `osStepBot` fold.** Ported
from the flash-attn foundation (`osStepBot_block_eq`). For a block with max
`M' = m ⊔ blockSup` and a state `(m, l, acc)` anchored via `l = κ(m)·L`,
`acc = κ(m)·T`, the kernel's one-shot rescale-and-add lands on
`block.foldl osStepBot (m, l, acc)`. -/
theorem osStepBot_block_eq (m : WithBot ℝ) (l acc T L : ℝ) (block : List (ℝ × ℝ))
    (hl : l = (m.elim 0 (fun r => pow2 (-r))) * L)
    (hacc : acc = (m.elim 0 (fun r => pow2 (-r))) * T)
    (hmL : m = ⊥ → L = 0) (hmT : m = ⊥ → T = 0) :
    let M' := m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
    (M',
     l * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0))).sum,
     acc * (WithBot.realExp2 (WithBot.realSub m M')).unbotD 0
       + (block.map (fun p => pow2 (p.1 - M'.unbotD 0) * p.2)).sum)
      = block.foldl osStepBot (m, l, acc) := by
  intro M'
  have hfst : (block.foldl osStepBot (m, l, acc)).1 = M' := by
    rw [aftStateBot_fst]
    have gen : ∀ (m0 : WithBot ℝ) (L0 : List (WithBot ℝ)),
        L0.foldl (· ⊔ ·) m0 = m0 ⊔ L0.foldr (· ⊔ ·) ⊥ := by
      intro m0 L0
      induction L0 generalizing m0 with
      | nil => simp
      | cons a t ih => simp only [List.foldl_cons, List.foldr_cons, ih]; rw [max_assoc]
    rw [gen]
  obtain ⟨hfold_l, hfold_acc⟩ := osStepBot_foldl_consistent block m l acc T L hl hacc hmL hmT
  rw [hfst] at hfold_l hfold_acc
  have hM'eq : M' = m ⊔ (block.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := rfl
  cases hM' : M' with
  | bot =>
    have hempty : block = [] := by
      rcases block with _ | ⟨a, t⟩
      · rfl
      · exfalso
        have : ((a.1 : ℝ) : WithBot ℝ) ≤ M' := by
          rw [hM'eq]
          exact le_sup_of_le_right (by simp only [List.map_cons, List.foldr_cons]; exact le_sup_left)
        rw [hM'] at this
        exact absurd (le_bot_iff.mp this) (WithBot.coe_ne_bot)
    have hm0 : m = ⊥ := by
      rw [hM'eq, hempty] at hM'
      simpa only [List.map_nil, List.foldr_nil, sup_bot_eq] using hM'
    have hl0 : l = 0 := by rw [hl, hm0]; simp [hmL hm0]
    have hacc0 : acc = 0 := by rw [hacc, hm0]; simp [hmT hm0]
    subst hempty
    rw [hl0, hacc0]
    simp only [List.foldl_nil, List.map_nil, List.sum_nil, add_zero, mul_zero, zero_mul]
    rw [hm0]
  | coe Mr =>
    rw [hM'] at hfst hfold_l hfold_acc
    have hlα : l * (WithBot.realExp2 (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = pow2 (-Mr) * L := by
      cases hm : m with
      | bot =>
        rw [hl, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = 0 from rfl,
          zero_mul, hmL hm]; ring
      | coe a =>
        rw [hl, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe,
          show Real.exp ((a - Mr) * Real.log 2) = pow2 (a - Mr) from by simp [pow2, mul_comm]]
        rw [mul_right_comm, ← pow2_add]; ring_nf
    have haccα : acc * (WithBot.realExp2 (WithBot.realSub m (↑Mr : WithBot ℝ))).unbotD 0 = pow2 (-Mr) * T := by
      cases hm : m with
      | bot =>
        rw [hacc, hm, show ((⊥ : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = 0 from rfl,
          zero_mul, hmT hm]; ring
      | coe a =>
        rw [hacc, hm, show ((↑a : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-a) from rfl]
        rw [WithBot.realSub_coe_coe, WithBot.realExp2_coe, WithBot.unbotD_coe,
          show Real.exp ((a - Mr) * Real.log 2) = pow2 (a - Mr) from by simp [pow2, mul_comm]]
        rw [mul_right_comm, ← pow2_add]; ring_nf
    have hsumL : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0))).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1)).sum := by
      have := sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun _ => 1)
      simp only [mul_one] at this
      rw [this, WithBot.unbotD_coe]
    have hsumT : (block.map (fun p => pow2 (p.1 - (↑Mr : WithBot ℝ).unbotD 0) * p.2)).sum
        = pow2 (-Mr) * (block.map (fun p => pow2 p.1 * p.2)).sum := by
      rw [sum_map_pow2_sub ((↑Mr : WithBot ℝ).unbotD 0) block (fun p => p.2), WithBot.unbotD_coe]
    refine Prod.ext hfst.symm (Prod.ext ?_ ?_)
    · rw [hfold_l, hlα, hsumL, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring
    · rw [hfold_acc, haccα, hsumT, show ((↑Mr : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-Mr) from rfl]; ring

/-- From a `⊥` running-max seed, the first `osStepBot` step wipes the `l`/`acc`
seed (`α = exp2(⊥ − s) = 0`), so a nonempty fold is seed-independent. -/
theorem osStepBot_bot_seed_indep (xs : List (ℝ × ℝ)) (hne : xs ≠ [])
    (l acc l' acc' : ℝ) :
    xs.foldl osStepBot (⊥, l, acc) = xs.foldl osStepBot (⊥, l', acc') := by
  obtain ⟨x, t, rfl⟩ := List.exists_cons_of_ne_nil hne
  obtain ⟨sv, v⟩ := x
  have hstep : ∀ L A : ℝ, osStepBot (⊥, L, A) (sv, v)
      = (((sv : ℝ) : WithBot ℝ), pow2 (sv - sv), pow2 (sv - sv) * v) := by
    intro L A
    simp only [osStepBot, bot_sup_eq]
    have hα : (WithBot.realExp2 (WithBot.realSub (⊥ : WithBot ℝ) ((sv : ℝ) : WithBot ℝ))).unbotD 0 = 0 := by
      rw [WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
    have hub : (((sv : ℝ) : WithBot ℝ)).unbotD 0 = sv := by rfl
    rw [hα, hub]; simp
  simp only [List.foldl_cons, hstep]

end Correct_without_Rounding

/-! ## FOUNDATION Part 1 — dimension-general body pieces (`aftgPreLoopG` ++ forRange `aftgLoopBodyG` :: postLoop)

The lowered algorithm body of `attn_fwd_triton_surface` is a preLoop / loop / postLoop
list: the deterministic preLoop statements (`aftgPreLoopG`), then the static
`Stmt.forRange` over `aftgLoopBodyG`, then the postLoop statements (`aftgPostLoopG`:
`acc = acc / l_i[:, None]` and the masked `tl.store`). This is a **static** `forRange`
(NOT a `forRangeDyn`), so the `forRange_inv` master invariant principle drives the loop.

The dimension-general split that the top theorem actually consumes is `aftg_body_splitG`
(checked by `rfl`), carrying every block/head dimension as a symbolic parameter rather
than a fixed numeral. -/

namespace AftFoundation

open VeriTile.Triton

end AftFoundation

namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton

open VeriTile.Triton

namespace AftFoundation

open VeriTile.Triton

end AftFoundation

namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton

open VeriTile.Triton

/-- Lift a base-state register readback through a `setReg` to a different name —
used to thread the `s1` head readbacks (`offs_*`/`qvk_offset`) through the
pointer-register `setReg`s accumulated by the earlier tail statements, so the
`expandDim`/`ref` offset rewrites fire on the wrapped state. -/
theorem regs_setReg_chain {d d' : TileDType} {sh sh' : TileShape}
    {n n' : RegName} {s : BlockState} {v : Tile d sh} {w : Tile d' sh'}
    (hne : n ≠ n') (h : s.regs d sh n = some v) :
    (s.setReg n' d' sh' w).regs d sh n = some v := by
  simp only [BlockState.setReg_ne_name, ne_eq, hne, not_false_eq_true, h]

/-! ## FINAL Part 4 (top) — genuine closed-form exec correctness + summary -/

/-- **Generic register-frame lemma.** If every statement in `body`, when stepped,
preserves the register `(d, sh, name)`, then so does the whole `stepStmts` run. -/
theorem stepStmts_regs_preserved {body : List Stmt} {s s' : BlockState}
    {d : TileDType} {sh : TileShape} {name : RegName}
    (hpre : ∀ st s1 s2, st ∈ body → stepStmt st s1 = some s2 → s2.regs d sh name = s1.regs d sh name)
    (h : stepStmts body s = some s') :
    s'.regs d sh name = s.regs d sh name := by
  induction body generalizing s with
  | nil => simp only [stepStmts] at h; rw [← Option.some.inj h]
  | cons st rest ih =>
    unfold stepStmts at h
    cases hst : stepStmt st s with
    | none => rw [hst] at h; simp at h
    | some mid =>
      rw [hst] at h
      have hmid := hpre st s mid (by simp) hst
      rw [ih (fun st' s1 s2 hmem hstep => hpre st' s1 s2 (by simp [hmem]) hstep) h, hmid]

/-- An `assign` to register `name'` preserves any other register `(d, sh, name)`
(`name ≠ name'`), whatever the assigned value. -/
theorem stepStmt_assign_regs_ne {d d' : TileDType} {sh sh' : TileShape}
    {name name' : RegName} {e : Op d' sh'} {s s' : BlockState}
    (hne : name ≠ name') (h : stepStmt (.assign d' sh' name' e) s = some s') :
    s'.regs d sh name = s.regs d sh name := by
  cases hev : evalOp e s with
  | none => rw [show stepStmt (.assign d' sh' name' e) s = none from by simp [stepStmt, hev]] at h
            exact absurd h.symm (Option.some_ne_none s')
  | some v =>
      rw [stepStmt_assign_eq_some hev] at h
      rw [← Option.some.inj h, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ hne]




/-- `evalOp` helper for `tl.math.exp2` (`Op.exp2`). -/
theorem aftg_evalOp_exp2 {shape : TileShape} (a : Op .real shape) (s : BlockState) :
    evalOp (.exp2 a) s = (do let va ← evalOp a s; some (Tile.uop WithBot.realExp2 va)) := by
  simp [evalOp]

/-- `evalOp` helper for the identity `real → real` cast (the `.to(tl.float32)`
fp32 wrapper around `tl.dot`, which is a no-op at the algorithm layer): given the
inner op's value, the cast returns the same tile (`FloatDType.cast real real` is
the identity through `WithBot ℝ`). -/
theorem aftg_castReal_eval {shape : TileShape} (a : Op .real shape) (s : BlockState)
    (va : Tile .real shape) (ha : evalOp a s = some va) :
    evalOp (Op.castFloat FloatDType.real FloatDType.real a) s
      = some ⟨fun i => va.data i⟩ := by
  rw [evalOp_castFloat]
  simp only [FloatDType.toTileDType_real]
  rw [ha]; rfl

/-- `evalOp` helper for the `>=` predicate (`Op.ge`), no `@[simp]` form. -/
theorem aftg_evalOp_ge {dtype a b shape} (h : ComparableDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- `evalOp` helper for `Op.boolAnd`, no `@[simp]` form. -/
theorem aftg_evalOp_boolAnd {a b shape} (bc : Broadcast a b shape)
    (x : Op .bool a) (y : Op .bool b) (s : BlockState) :
    evalOp (.boolAnd bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s;
      some (Tile.bop (fun p q : Bool => p && q) bc vx vy)) := by
  simp [evalOp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- `evalOp (Op.ptrBase R)` — the base pointer tile `(R, 0)` at the empty shape. -/
theorem aftg_evalOp_ptrBase {d : TileDType} (region : Region d) (s : BlockState) :
    evalOp (Op.ptrBase region) s = some (Tile.scalar (Region.cast region, 0)) := by
  simp only [evalOp]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Bind-aware `ptrAdd` reducer.** Given the operand evaluations
`evalOp ptr s = some ptrs` and `evalOp off s = some offs`, fire the `ptrAdd`
reduction inside the `Option.bind` do-block. Threads each `expandDim`-bearing
offset (pre-proved via `evalOp_expandDim_ref_of_regs` / `evalOp_add` / `evalOp_mul`)
into the elementwise `Tile.ptrAdd`. -/
theorem aftg_evalOp_ptrAdd_of {a b shape : TileShape}
    (bc : Broadcast a b shape) (ptr : Op .ptr a) (off : Op .nat b) (s : BlockState)
    (ptrs : Tile .ptr a) (offs : Tile .nat b)
    (hptr : evalOp ptr s = some ptrs) (hoff : evalOp off s = some offs) :
    evalOp (Op.ptrAdd bc ptr off) s = some (Tile.ptrAdd bc ptrs offs) := by
  simp only [evalOp, hptr, hoff, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Bind-aware masked `.ptr` load reducer.** Given the pointer-tile and mask-tile
evaluations, fire the masked `tl.load` over an arbitrary evaluated `.ptr` op (not
just a `ref`): kept lanes read `s.readMem`, masked-out lanes read `s.undef`.
Generalizes `aftg_load_k_eval` to inline (non-`ref`) ptr/mask ops,
unblocking the preLoop `q` load (inline `boolAnd` mask). -/
theorem aftg_evalOp_load_ptr_mask_of {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (s : BlockState)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hptr : evalOp ptrOp s = some ptrs) (hmask : evalOp maskOp s = some masks) :
    evalOp (.load .real (.ptr ptrOp) (.mask maskOp)) s
      = some ⟨fun i : TileIndex shape =>
          if masks.data i then some (s.readMem (ptrs.data i).1 (ptrs.data i).2)
          else some (s.undef (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hptr, hmask, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Bind-aware unmasked `.ptr` load reducer.** Given the pointer-tile evaluation,
fire the unmasked `tl.load` over an arbitrary evaluated `.ptr` op: every lane reads
`s.readMem`. Unblocks the preLoop scalar `q_scale` load (over `Q_scale_ptr`). -/
theorem aftg_evalOp_load_ptr_none_of {shape : TileShape}
    (ptrOp : Op .ptr shape) (s : BlockState) (ptrs : Tile .ptr shape)
    (hptr : evalOp ptrOp s = some ptrs) :
    evalOp (.load .real (.ptr ptrOp) .none) s
      = some ⟨fun i : TileIndex shape =>
          some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hptr, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L3: `k = tl.load(K_ptrs, mask=k_mask)`** — elementwise masked pointer load.
`K_ptrs` is a plain `[BM, BN]` pointer tile (per-lane `(region, offset)`); each
kept lane reads `s.readMem`, each masked-out lane reads `s.undef`. -/
theorem aftg_load_k_eval
    (s : BlockState) (BM BN : Nat) (name maskName : RegName)
    (ptrs : Tile .ptr [BM, BN]) (masks : Tile .bool [BM, BN])
    (hptr : s.regs .ptr [BM, BN] name = some ptrs)
    (hmask : s.regs .bool [BM, BN] maskName = some masks) :
    evalOp (.load .real (.ptr (.ref .ptr [BM, BN] name))
        (.mask (.ref .bool [BM, BN] maskName))) s
      = some ⟨fun i : TileIndex [BM, BN] =>
          if masks.data i then some (s.readMem (ptrs.data i).1 (ptrs.data i).2)
          else some (s.undef (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, evalOp_ref, hptr, hmask, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L4: `k_scale = tl.load(K_scale_ptr)`** — unmasked scalar pointer load. -/
theorem aftg_load_kscale_eval
    (s : BlockState) (name : RegName) (ptr : Tile .ptr [])
    (hptr : s.regs .ptr [] name = some ptr) :
    evalOp (.load .real (.ptr (.ref .ptr [] name)) .none) s
      = some ⟨fun _ : TileIndex [] =>
          some (s.readMem (ptr.data PUnit.unit).1 (ptr.data PUnit.unit).2)⟩ := by
  simp only [evalOp, evalOp_ref, hptr, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real, if_true]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L5: `qk = (castFloat (tl.dot(q, k))) * q_scale * k_scale`** — the `q·k` dot
(through an identity `real → real` fp32 cast) scaled by the two scalar
quantization factors `q_scale`, `k_scale`, each broadcast on the right. -/
theorem aftg_qk_dot_eval (s : BlockState) (BM BN BD : Nat)
    (qtile : Tile .real [BM, BD]) (ktile : Tile .real [BD, BN])
    (qsc ksc : Tile .real [])
    (hq : s.regs .real [BM, BD] "q" = some qtile)
    (hk : s.regs .real [BD, BN] "k" = some ktile)
    (hqsc : s.regs .real [] "q_scale" = some qsc)
    (hksc : s.regs .real [] "k_scale" = some ksc) :
    evalOp (Op.mul .real Broadcast.scalarR
        (Op.mul .real Broadcast.scalarR
          (Op.castFloat FloatDType.real FloatDType.real
            (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k")))
          (Op.ref .real [] "q_scale"))
        (Op.ref .real [] "k_scale")) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.mul Broadcast.scalarR
            ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) ksc) := by
  have hqr : evalOp (Op.ref .real [BM, BD] "q") s = some qtile := by rw [evalOp_ref, hq]
  have hkr : evalOp (Op.ref .real [BD, BN] "k") s = some ktile := by rw [evalOp_ref, hk]
  have hdot : @evalOp TileDType.real [BM, BN]
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k")) s
      = some (Tile.dot [] qtile ktile) := by
    erw [evalOp_dot [] (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"), hqr, hkr]; rfl
  have hcast : @evalOp TileDType.real [BM, BN]
      (Op.castFloat FloatDType.real FloatDType.real
        (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))) s
      = some ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ :=
    aftg_castReal_eval
      (Op.dot (batch := []) (Op.ref .real [BM, BD] "q") (Op.ref .real [BD, BN] "k"))
      s (Tile.dot [] qtile ktile) hdot
  rw [evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, hcast, hqsc, hksc, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L18: `acc += tl.dot(p.to(fp16).to(real), v)`** — numerator accumulation. The
fp16-cast `p` is cast back to real (identity at the algorithm layer) inside the
dot; `v` is real. -/
theorem aftg_acc_eval (s : BlockState) (BM BN BD : Nat)
    (acctile : Tile .real [BM, BD]) (pf16 : Tile .fp16 [BM, BN]) (vtile : Tile .real [BN, BD])
    (hacc : s.regs .real [BM, BD] "acc" = some acctile)
    (hp : s.regs .fp16 [BM, BN] "p" = some pf16)
    (hv : s.regs .real [BN, BD] "v" = some vtile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BM, BD] "acc")
        (Op.dot (batch := [])
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [BM, BN] "p"))
          (Op.ref .real [BN, BD] "v"))) s
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) acctile
          (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ vtile)) := by
  have hpc : evalOp
      (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [BM, BN] "p")) s
      = some ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ := by
    simp only [evalOp_castFloat, FloatDType.toTileDType_fp16, evalOp_ref, hp,
      Option.bind_eq_bind, Option.bind_some]
  have hvr : evalOp (Op.ref .real [BN, BD] "v") s = some vtile := by rw [evalOp_ref, hv]
  have hdot : @evalOp TileDType.real [BM, BD]
      (Op.dot (batch := [])
        (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [BM, BN] "p"))
        (Op.ref .real [BN, BD] "v")) s
      = some (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ vtile) := by
    erw [evalOp_dot []
      (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [BM, BN] "p"))
      (Op.ref .real [BN, BD] "v"), hpc, hvr]; rfl
  rw [evalOp_add]; simp only [evalOp_ref, hacc, hdot, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **L21: `K_scale_ptr += 1`** — advance the scalar K-scale pointer by one key
block. -/
theorem aftg_advance_kscale_eval (s : BlockState) (name : RegName) (ptr : Tile .ptr [])
    (hptr : s.regs .ptr [] name = some ptr) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] name) (Op.constNat 1)) s
      = some (Tile.ptrAdd Broadcast.nil ptr (Tile.scalar 1)) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hptr, Option.bind_eq_bind, Option.bind_some]

/-- The running `max` component of an `osStepBot` fold is the `WithBot ⊔`-fold. -/
theorem aftgStateBot_fst
    (xs : List (ℝ × ℝ)) (m₀ : WithBot ℝ) (l₀ acc₀ : ℝ) :
    (xs.foldl osStepBot (m₀, l₀, acc₀)).1
      = (xs.map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldl (· ⊔ ·) m₀ := by
  induction xs generalizing m₀ l₀ acc₀ with
  | nil => rfl
  | cons x xs ih => simp only [List.foldl_cons, List.map_cons]; rw [ih]; rfl

/-- The `WithBot ⊔`-fold is seed/direction-agnostic. -/
theorem aftg_foldl_sup_bot_eq_foldr (L : List (WithBot ℝ)) :
    L.foldl (· ⊔ ·) (⊥ : WithBot ℝ) = L.foldr (· ⊔ ·) (⊥ : WithBot ℝ) := by
  have gen : ∀ (m : WithBot ℝ), L.foldl (· ⊔ ·) m = m ⊔ L.foldr (· ⊔ ·) ⊥ := by
    induction L with
    | nil => intro m; simp
    | cons a t ih => intro m; simp only [List.foldl_cons, List.foldr_cons, ih]; rw [max_assoc]
  rw [gen ⊥, bot_sup_eq]

/-- Generic threshold-split for a `.val`-ascending list. Ported from flash-attn. -/
private theorem aftg_filterMap_window_split {n : Nat} (l : List (Fin n))
    (hsorted : l.Pairwise (fun a b => a.val < b.val))
    (t hi₂ : Nat) (Q : Fin n → Prop) [DecidablePred Q]
    (g : Fin n → ℝ × ℝ) (hle : t ≤ hi₂) :
    l.filterMap (fun j => if Q j ∧ j.val < hi₂ then some (g j) else none)
      = l.filterMap (fun j => if Q j ∧ j.val < t then some (g j) else none)
        ++ l.filterMap (fun j => if t ≤ j.val ∧ j.val < hi₂ ∧ Q j then some (g j) else none) := by
  induction l with
  | nil => simp
  | cons a tl ih =>
    have htl : tl.Pairwise (fun x y => x.val < y.val) := (List.pairwise_cons.mp hsorted).2
    have hahead : ∀ b ∈ tl, a.val < b.val := (List.pairwise_cons.mp hsorted).1
    rw [List.filterMap_cons, List.filterMap_cons, List.filterMap_cons]
    by_cases hlt : a.val < t
    · rw [ih htl]
      have hnb : ¬ (t ≤ a.val ∧ a.val < hi₂ ∧ Q a) := fun h => (Nat.not_le.mpr hlt) h.1
      rw [if_neg hnb]
      by_cases hQ : Q a
      · have h2 : a.val < hi₂ := lt_of_lt_of_le hlt hle
        rw [if_pos (And.intro hQ h2 : Q a ∧ a.val < hi₂),
          if_pos (And.intro hQ hlt : Q a ∧ a.val < t)]
        rfl
      · rw [if_neg (fun h : Q a ∧ a.val < hi₂ => hQ h.1),
          if_neg (fun h : Q a ∧ a.val < t => hQ h.1)]
    · have hge : t ≤ a.val := Nat.not_lt.mp hlt
      have htail_prefix : tl.filterMap (fun j => if Q j ∧ j.val < t then some (g j) else none) = [] := by
        apply List.filterMap_eq_nil_iff.mpr
        intro b hb
        have hab : a.val < b.val := hahead b hb
        have hbt : ¬ (b.val < t) := by omega
        simp [hbt]
      rw [ih htl, htail_prefix]
      have hnp : ¬ (Q a ∧ a.val < t) := fun h => hlt h.2
      rw [if_neg hnp]
      by_cases hQ : Q a
      · by_cases h2 : a.val < hi₂
        · rw [if_pos (And.intro hQ h2 : Q a ∧ a.val < hi₂),
            if_pos (And.intro hge (And.intro h2 hQ) : t ≤ a.val ∧ a.val < hi₂ ∧ Q a)]
          rfl
        · rw [if_neg (fun h : Q a ∧ a.val < hi₂ => h2 h.2),
            if_neg (fun h : t ≤ a.val ∧ a.val < hi₂ ∧ Q a => h2 h.2.1)]
      · rw [if_neg (fun h : Q a ∧ a.val < hi₂ => hQ h.1),
          if_neg (fun h : t ≤ a.val ∧ a.val < hi₂ ∧ Q a => hQ h.2.2)]

/-- `filterMap`-then-map-and-sum over `finRange n` equals the masked `Finset.sum`. -/
theorem aftg_filterMap_finRange_sum {α : Type*} (n : Nat)
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

/-- Any member of a `WithBot ℝ` list is `≤` its `foldr (⊔) ⊥`. -/
theorem aftg_mem_le_foldr_sup (a : WithBot ℝ) :
    ∀ (L : List (WithBot ℝ)), a ∈ L → a ≤ L.foldr (· ⊔ ·) ⊥ := by
  intro L
  induction L with
  | nil => intro h; simp at h
  | cons x t ih =>
    intro h
    rcases List.mem_cons.mp h with rfl | h
    · exact le_sup_left
    · exact le_trans (ih h) le_sup_right

/-- `filterMap`-then-coe `foldr ⊔ ⊥` over `finRange n` equals the masked `Finset.sup`. -/
theorem aftg_filterMap_foldr_sup (n : Nat) (P : Fin n → Prop) [DecidablePred P]
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

/-- Helper: `WithBot.realSub (some 0) (some 1e6) = some (-1000000)`. -/
theorem aftg_sentinel_eq : WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ)) = some (-1000000.0 : ℝ) := by
  simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map]; norm_num

namespace AftgFoundation

open VeriTile.Triton

/-- General loop body (symbolic `BLOCK_M`/`BLOCK_N`/`BLOCK_DMODEL`/`HEAD_DIM`/
`N_CTX`/`HEAD_ACTIVE`): the streamed per-key-block statements with every dimension
carried as a parameter rather than a fixed numeral. -/
def aftgLoopBodyG (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  [ -- 0: start_n = tl.multiple_of(start_n, BLOCK_N)  (identity)
    Stmt.assign .nat [] "start_n" (Op.ref .nat [] "start_n"),
    -- 1: k_mask
    Stmt.assign .bool [BLOCK_DMODEL, BLOCK_N] "k_mask"
      (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))),
    -- 2: k = tl.load(K_ptrs, mask=k_mask)
    Stmt.assign .real [BLOCK_DMODEL, BLOCK_N] "k"
      (Op.load .real (.ptr (.ref .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs")) (.mask (.ref .bool [BLOCK_DMODEL, BLOCK_N] "k_mask"))),
    -- 3: k_scale = tl.load(K_scale_ptr)
    Stmt.assign .real [] "k_scale"
      (Op.load .real (.ptr (.ref .ptr [] "K_scale_ptr")) .none),
    -- 4: qk = castFloat(q·k) * q_scale * k_scale
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.mul .real Broadcast.scalarR
        (Op.mul .real Broadcast.scalarR
          (Op.castFloat FloatDType.real FloatDType.real
            (Op.dot (batch := []) (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "q") (Op.ref .real [BLOCK_DMODEL, BLOCK_N] "k")))
          (Op.ref .real [] "q_scale"))
        (Op.ref .real [] "k_scale")),
    -- 5: mask = offs_m[:,None] >= start_n + offs_n[None,:]
    Stmt.assign .bool [BLOCK_M, BLOCK_N] "mask"
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))),
    -- 6: qk = tl.where(mask, qk, -1000000.0)
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.where (Op.ref .bool [BLOCK_M, BLOCK_N] "mask")
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [BLOCK_M, BLOCK_N])),
    -- 7: m_ij = maximum(m_i, max(qk,1))
    Stmt.assign .real [BLOCK_M] "m_ij"
      (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
            (Op.ref .real [BLOCK_M, BLOCK_N] "qk")))
        (Op.ref .real [BLOCK_M] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
          (Op.ref .real [BLOCK_M, BLOCK_N] "qk"))),
    -- 8: qk = qk - m_ij[:, None]
    Stmt.assign .real [BLOCK_M, BLOCK_N] "qk"
      (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "m_ij"))),
    -- 9: p = exp2(qk)
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p" (Op.exp2 (Op.ref .real [BLOCK_M, BLOCK_N] "qk")),
    -- 10: p = tl.where(mask, p, 0)
    Stmt.assign .real [BLOCK_M, BLOCK_N] "p"
      (Op.where (Op.ref .bool [BLOCK_M, BLOCK_N] "mask")
        (Op.ref .real [BLOCK_M, BLOCK_N] "p") (Op.broadcast (Op.const 0.0) [BLOCK_M, BLOCK_N])),
    -- 11: l_ij = sum(p, 1)
    Stmt.assign .real [BLOCK_M] "l_ij"
      (Op.reduceSum (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false (Op.ref .real [BLOCK_M, BLOCK_N] "p")),
    -- 12: alpha = exp2(m_i - m_ij)
    Stmt.assign .real [BLOCK_M] "alpha"
      (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij"))),
    -- 13: l_i = l_i * alpha + l_ij
    Stmt.assign .real [BLOCK_M] "l_i"
      (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "l_i") (Op.ref .real [BLOCK_M] "alpha"))
        (Op.ref .real [BLOCK_M] "l_ij")),
    -- 14: acc = acc * alpha[:, None]
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "alpha"))),
    -- 15: v = tl.load(V_ptrs, mask=...)
    Stmt.assign .real [BLOCK_N, BLOCK_DMODEL] "v"
      (Op.load .real (.ptr (.ref .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
            (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))),
    -- 16: p = p.to(fp16)
    Stmt.assign .fp16 [BLOCK_M, BLOCK_N] "p"
      (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BLOCK_M, BLOCK_N] "p")),
    -- 17: acc += dot(p.to(real), v)
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.dot (batch := [])
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref .fp16 [BLOCK_M, BLOCK_N] "p"))
          (Op.ref .real [BLOCK_N, BLOCK_DMODEL] "v"))),
    -- 18: m_i = m_ij
    Stmt.assign .real [BLOCK_M] "m_i" (Op.ref .real [BLOCK_M] "m_ij"),
    -- 19: K_ptrs += BLOCK_N * HEAD_DIM
    Stmt.assign .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_N) (Op.constNat HEAD_DIM))),
    -- 20: K_scale_ptr += 1
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ref .ptr [] "K_scale_ptr") (Op.constNat 1)),
    -- 21: V_ptrs += BLOCK_N * HEAD_DIM
    Stmt.assign .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs")
        (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_N) (Op.constNat HEAD_DIM))) ]

/-- General preLoop (22 deterministic prefix statements), symbolic dims with
contiguous strides (`stride_qm = HEAD_DIM`, `stride_qk = 1`, `stride_kn = HEAD_DIM`):
the deterministic prefix with every dimension carried as a parameter rather than a
fixed numeral. -/
def aftgPreLoopG (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_hz" (Op.programId 1),
    Stmt.assign .nat [] "off_z"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
    Stmt.assign .nat [] "off_h"
      (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
    Stmt.assign .nat [] "qvk_offset"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_qh))),
    Stmt.assign .nat [] "vk_offset"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat HEAD_DIM)),
    Stmt.assign .nat [] "q_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_M)) (Op.constNat 1))
          (Op.constNat BLOCK_M))),
    Stmt.assign .nat [] "k_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_N)) (Op.constNat 1))
          (Op.constNat BLOCK_N))),
    Stmt.assign .nat [BLOCK_M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_DMODEL] "offs_k" (Op.arange BLOCK_DMODEL),
    Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [] "Q_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase QScale)
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))),
    Stmt.assign .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))),
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase KScale) (Op.ref .nat [] "k_scale_offset")),
    Stmt.assign .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .real [BLOCK_M] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLOCK_M] "l_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) (Op.const 1.0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc" (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "q"
      (Op.load .real (.ptr (.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))),
    Stmt.assign .real [] "q_scale"
      (Op.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none) ]

/-- General preLoop head — statements 0–10 of `aftgPreLoopG`. -/
def aftgPreLoopHeadG (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "start_m" (Op.programId 0),
    Stmt.assign .nat [] "off_hz" (Op.programId 1),
    Stmt.assign .nat [] "off_z"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
    Stmt.assign .nat [] "off_h"
      (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)),
    Stmt.assign .nat [] "qvk_offset"
      (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_qh))),
    Stmt.assign .nat [] "vk_offset"
      (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat HEAD_DIM)),
    Stmt.assign .nat [] "q_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_M)) (Op.constNat 1))
          (Op.constNat BLOCK_M))),
    Stmt.assign .nat [] "k_scale_offset"
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_N)) (Op.constNat 1))
          (Op.constNat BLOCK_N))),
    Stmt.assign .nat [BLOCK_M] "offs_m"
      (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)),
    Stmt.assign .nat [BLOCK_N] "offs_n" (Op.arange BLOCK_N),
    Stmt.assign .nat [BLOCK_DMODEL] "offs_k" (Op.arange BLOCK_DMODEL) ]

/-- General preLoop tail — statements 11–21 of `aftgPreLoopG`. -/
def aftgPreLoopTailG (Q K V QScale KScale Out : RegionName)
    (HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Q)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [] "Q_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase QScale)
        (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m"))),
    Stmt.assign .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase K)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))),
    Stmt.assign .ptr [] "K_scale_ptr"
      (Op.ptrAdd Broadcast.nil (Op.ptrBase KScale) (Op.ref .nat [] "k_scale_offset")),
    Stmt.assign .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase V)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Out)
        (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1)))),
    Stmt.assign .real [BLOCK_M] "m_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf),
    Stmt.assign .real [BLOCK_M] "l_i"
      (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) (Op.const 1.0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc" (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)),
    Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "q"
      (Op.load .real (.ptr (.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs"))
        (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))),
    Stmt.assign .real [] "q_scale"
      (Op.load .real (.ptr (.ref .ptr [] "Q_scale_ptr")) .none) ]

theorem aftgPreLoopG_eq_head_tail (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    aftgPreLoopG Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
      = aftgPreLoopHeadG stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL
        ++ aftgPreLoopTailG Q K V QScale KScale Out HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE :=
  rfl

/-- General postLoop (2 statements): `acc /= l_i[:, None]` then masked store to `O_block_ptr`. -/
def aftgPostLoopG (Out : RegionName) (N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  [ Stmt.assign .real [BLOCK_M, BLOCK_DMODEL] "acc"
      (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc") (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i"))),
    Stmt.store .real [BLOCK_M, BLOCK_DMODEL] (.ptr (.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"))
      (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
      (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE))))) ]

/-- General loop-body head (statements 0–10). -/
def aftgLoopBodyHeadG (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  List.take 11 (aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE)

/-- General loop-body tail (statements 11–21). -/
def aftgLoopBodyTailG (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) : List Stmt :=
  List.drop 11 (aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE)

theorem aftgLoopBodyG_eq_head_tail (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
      = aftgLoopBodyHeadG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE
        ++ aftgLoopBodyTailG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE := by
  rw [aftgLoopBodyHeadG, aftgLoopBodyTailG, List.take_append_drop]

end AftgFoundation

section General

open VeriTile.Triton

/-! ### General spec layer (genuine causal closed form over symbolic dims) -/

/-- General per-plane base offset: `off_z·stride_qz + off_h·stride_qh` with
`off_z = pids1 / H`, `off_h = pids1 % H`. -/
def baseOffsetAFT2G (s : BlockState) (stride_qz stride_qh H : Nat) : Nat :=
  s.pids 1 / H * stride_qz + s.pids 1 % H * stride_qh

/-- General query tile: row `i` = global query `pids0·BLOCK_M + i`, head lane `e`
(`stride_qm = HEAD_DIM`, `stride_qk = 1`). -/
noncomputable def qTileAFT2G (s : BlockState) (Q : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, _) => s.readMem Q (baseOffsetAFT2G s stride_qz stride_qh H
    + (s.pids 0 * BLOCK_M + i.val) * HEAD_DIM + e.val)

/-- General key tile: row `j` (global key), head lane `e` (`K[base + j·HEAD_DIM + e]`). -/
noncomputable def kTileAFT2G (s : BlockState) (K : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, e, _) => s.readMem K (baseOffsetAFT2G s stride_qz stride_qh H + j.val * HEAD_DIM + e.val)

/-- General value tile: row `j` (global key), head lane `d` (`V[base + j·HEAD_DIM + d]`). -/
noncomputable def vTileAFT2G (s : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, d, _) => s.readMem V (baseOffsetAFT2G s stride_qz stride_qh H + j.val * HEAD_DIM + d.val)

/-- General global query row for output tile-row `i`. -/
def qStartAFT2G (s : BlockState) (BLOCK_M : Nat) : Nat := s.pids 0 * BLOCK_M

/-- General masked query tile: head-active (`e < HEAD_ACTIVE`) and query-row
boundary (`qStart + i < N_CTX`) masking, mirroring the kernel's `q` load mask. -/
noncomputable def qTileAFT2mG (s : BlockState) (Q : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ :=
  fun (i, e, u) =>
    if qStartAFT2G s BLOCK_M + i.val < N_CTX ∧ e.val < HEAD_ACTIVE then
      qTileAFT2G s Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL (i, e, u) else 0

/-- General masked value tile: head-active (`d < HEAD_ACTIVE`) masking. -/
noncomputable def vTileAFT2mG (s : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL HEAD_ACTIVE : Nat) :
    TileIndex [SEQ, BLOCK_DMODEL] → ℝ :=
  fun (j, d, u) =>
    if d.val < HEAD_ACTIVE then
      vTileAFT2G s V stride_qz stride_qh H HEAD_DIM SEQ BLOCK_DMODEL (j, d, u) else 0

/-- **General genuine closed form** (exp2, causal): predicate-masked base-2
per-key-scale attention with the `causalKeep qStart` mask, over the kernel's
actually-loaded masked q/v tiles. -/
noncomputable def attnFwdTritonOutSpecG
    (s : BlockState) (Q K V : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (idx : TileIndex [BLOCK_M, BLOCK_DMODEL]) : ℝ :=
  attentionRealBase2PerKeyScalePred
    (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
    (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
    (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
    keyScale (fun i j => causalKeep (qStartAFT2G s BLOCK_M) i j) idx

/-- General per-key score scale carrier `q_scale · k_scale` (block `j / BLOCK_N`). -/
noncomputable def keyScaleAFT2G (s : BlockState) (QScale KScale : RegionName)
    (N_CTX BLOCK_M BLOCK_N numKVBlocks : Nat) : Fin (BLOCK_N * numKVBlocks) → ℝ :=
  fun j => s.readMem QScale (s.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M) + s.pids 0)
            * s.readMem KScale (s.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N) + j.val / BLOCK_N)

/-- General streaming bridge: the closed form equals the `osStep` online-softmax
fold over the causal-masked key list. -/
theorem attnFwdTritonOutSpecG_eq_streaming
    (s : BlockState) (Q K V : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    attnFwdTritonOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N
        BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale (i, d, PUnit.unit)
      = (let st := (attnKeyListPred
            (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
            (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
            (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
            keyScale (fun i j => causalKeep (qStartAFT2G s BLOCK_M) i j) i d).foldl
              osStep (0, 0, 0)
         st.2.2 / st.2.1) := by
  simpa [attnFwdTritonOutSpecG] using
    VeriTile.Triton.attentionRealBase2PerKeyScalePred_eq_streaming
      (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
      (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
      (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
      keyScale (fun i j => causalKeep (qStartAFT2G s BLOCK_M) i j) i d

/-! ### General ⊥-seed online-softmax foundation math

The dimension-general `aftgKVG`/`aftgKeysUptoG`/`aftgBlockG`/`aftgRunningMaxG`/
`aftgStateBotG`/`aftgStateBot1G` stack over symbolic `BLOCK_M`(query rows)/
`BLOCK_DMODEL`(channels)/`SEQ`(keys)/`BLOCK_N`(block stride). Reuses the dim-agnostic generic core
(`osStepBot`, `osStepBot_foldl_consistent`, `osStepBot_block_eq`,
`osStepBot_bot_seed_indep`, `aftg_filterMap_window_split`, `aftg_filterMap_foldr_sup`,
`aftg_mem_le_foldr_sup`, `aftg_filterMap_finRange_sum`, `aftg_foldl_sup_bot_eq_foldr`). -/

variable {BLOCK_M BLOCK_DMODEL SEQ : Nat}

/-- General `(score, value)` pair the kernel streams for output `(i, d)` at key `j`. -/
noncomputable def aftgKVG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (j : Fin SEQ) : ℝ × ℝ :=
  (keyScale j * Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit)),
   vT (j, d, PUnit.unit))

/-- General causal per-row key list over `[0, hi)`: keys `j < hi` with `j ≤ qStart + i`. -/
noncomputable def aftgKeysUptoG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : List (ℝ × ℝ) :=
  (List.finRange SEQ).filterMap (fun j : Fin SEQ =>
    if j.val < hi ∧ j.val ≤ qStart + i.val then
      some (aftgKVG qT kT vT keyScale i d j)
    else none)

/-- General block-`c` per-row key list (`c·BLOCK_N ≤ j < (c+1)·BLOCK_N`, causal). -/
noncomputable def aftgBlockG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : List (ℝ × ℝ) :=
  (List.finRange SEQ).filterMap (fun j : Fin SEQ =>
    if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val then
      some (aftgKVG qT kT vT keyScale i d j)
    else none)

/-- General ⊥-seeded running max of the streamed key prefix `[0, hi)`. -/
noncomputable def aftgRunningMaxG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : WithBot ℝ :=
  ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map
    (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥

/-- General ⊥-seeded running `(max, denom, acc)` after streaming `[0, hi)` (seed-`0`). -/
noncomputable def aftgStateBotG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : WithBot ℝ × ℝ × ℝ :=
  (aftgKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)

/-- General ⊥-seeded running state from the kernel's `l_i = 1` seed. -/
noncomputable def aftgStateBot1G
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) : WithBot ℝ × ℝ × ℝ :=
  (aftgKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 1, 0)

variable (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
  (keyScale : Fin SEQ → ℝ)

/-- General: ⊥-seeded running max of `aftgStateBotG` is `aftgRunningMaxG`. -/
theorem aftgStateBotG_fst_eq_runningMax (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).1
      = aftgRunningMaxG qT kT vT keyScale qStart hi i d := by
  rw [aftgStateBotG, aftgStateBot_fst, aftgRunningMaxG, aftg_foldl_sup_bot_eq_foldr]

/-- General ⊥-seeded denominator. -/
theorem aftgStateBotG_snd_fst (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.1
      = ((aftgRunningMaxG qT kT vT keyScale qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * (0 + ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [aftgStateBotG]
  rw [(osStepBot_foldl_consistent (aftgKeysUptoG qT kT vT keyScale qStart hi i d) ⊥ 0 0 0 0
    (by simp) (by simp) (by simp) (by simp)).1]
  rw [show ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)).1
        = aftgRunningMaxG qT kT vT keyScale qStart hi i d from by
    rw [aftgStateBot_fst, aftgRunningMaxG, aftg_foldl_sup_bot_eq_foldr]]

/-- General ⊥-seeded accumulator. -/
theorem aftgStateBotG_snd_snd (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.2
      = ((aftgRunningMaxG qT kT vT keyScale qStart hi i d).elim 0 (fun r => pow2 (-r)))
        * (0 + ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [aftgStateBotG]
  rw [(osStepBot_foldl_consistent (aftgKeysUptoG qT kT vT keyScale qStart hi i d) ⊥ 0 0 0 0
    (by simp) (by simp) (by simp) (by simp)).2]
  rw [show ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).foldl osStepBot (⊥, 0, 0)).1
        = aftgRunningMaxG qT kT vT keyScale qStart hi i d from by
    rw [aftgStateBot_fst, aftgRunningMaxG, aftg_foldl_sup_bot_eq_foldr]]

/-- General ⊥-seeded ratio (max factor cancels) on a nonempty window. -/
theorem aftgStateBotG_ratio_eq (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hne : aftgRunningMaxG qT kT vT keyScale qStart hi i d ≠ ⊥) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.2
        / (aftgStateBotG qT kT vT keyScale qStart hi i d).2.1
      = ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum
        / ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum := by
  rw [aftgStateBotG_snd_fst, aftgStateBotG_snd_snd, zero_add, zero_add]
  cases hM : aftgRunningMaxG qT kT vT keyScale qStart hi i d with
  | bot => exact absurd hM hne
  | coe r =>
    have hκ : ((r : WithBot ℝ).elim 0 (fun r => pow2 (-r))) = pow2 (-r) := rfl
    rw [hκ]
    have hpos : pow2 (-r) ≠ 0 := ne_of_gt (pow2_pos _)
    rw [mul_div_mul_left _ _ hpos]

/-- General ⊥-seeded running max at the empty window is `⊥`. -/
theorem aftgRunningMaxG_zero (qStart : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgRunningMaxG qT kT vT keyScale qStart 0 i d = ⊥ := by
  unfold aftgRunningMaxG aftgKeysUptoG
  rw [show (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (aftgKVG qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- General window split (`hi = c·BLOCK_N`). -/
theorem aftgKeysUptoG_succ (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgKeysUptoG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d
      = aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d
        ++ aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d := by
  unfold aftgKeysUptoG aftgBlockG
  rw [show (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
          then some (aftgKVG qT kT vT keyScale i d j) else none)
      = (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val ≤ qStart + i.val ∧ j.val < (c + 1) * BLOCK_N
          then some (aftgKVG qT kT vT keyScale i d j) else none)
      from List.filterMap_congr (fun j _ => by simp only [and_comm])]
  rw [aftg_filterMap_window_split (List.finRange SEQ) (List.pairwise_lt_finRange SEQ)
    (c * BLOCK_N) ((c + 1) * BLOCK_N) (fun j => j.val ≤ qStart + i.val)
    (fun j => aftgKVG qT kT vT keyScale i d j) (by nlinarith [Nat.zero_le BLOCK_N])]
  refine congrArg₂ (· ++ ·) ?_ ?_
  · apply List.filterMap_congr; intro j _; simp only [and_comm]
  · apply List.filterMap_congr; intro j _
    by_cases h1 : c * BLOCK_N ≤ j.val <;> by_cases h2 : j.val < (c + 1) * BLOCK_N <;>
      by_cases h3 : j.val ≤ qStart + i.val <;> simp [h1, h2, h3, and_assoc]

/-- General one-block advance of `aftgStateBotG`. -/
theorem aftgStateBotG_succ (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d
      = (aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).foldl osStepBot
          (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d) := by
  unfold aftgStateBotG
  rw [aftgKeysUptoG_succ, List.foldl_append]

/-- General running max one-block advance. -/
theorem aftgRunningMaxG_succ (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d
      = aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i d
        ⊔ ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
            (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
  unfold aftgRunningMaxG
  rw [aftgKeysUptoG_succ, List.map_append]
  induction (aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d) with
  | nil => simp
  | cons a t ih => simp only [List.map_cons, List.foldr_cons, List.cons_append, ih, max_assoc]

/-- General `aftgRunningMaxG` is channel-independent. -/
theorem aftgRunningMaxG_eq (qStart hi : Nat) (i : Fin BLOCK_M) (d d' : Fin BLOCK_DMODEL) :
    aftgRunningMaxG qT kT vT keyScale qStart hi i d
      = aftgRunningMaxG qT kT vT keyScale qStart hi i d' := by
  unfold aftgRunningMaxG aftgKeysUptoG
  congr 1
  rw [List.map_filterMap, List.map_filterMap]
  apply List.filterMap_congr
  intro j _
  by_cases hj : j.val < hi ∧ j.val ≤ qStart + i.val <;> simp [aftgKVG, hj]

/-- General `aftgRunningMaxG` over a nonempty causal window ≠ ⊥ (key 0 always kept). -/
theorem aftgRunningMaxG_ne_bot (qStart hi : Nat) (hhi : 1 ≤ hi) (hSEQ : 0 < SEQ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgRunningMaxG qT kT vT keyScale qStart hi i d ≠ ⊥ := by
  unfold aftgRunningMaxG aftgKeysUptoG
  set sc0 : ℝ := keyScale (⟨0, hSEQ⟩ : Fin SEQ) *
      Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) *
        kT (⟨0, hSEQ⟩, e, PUnit.unit)) with hsc0
  have hmem : ((sc0 : ℝ) : WithBot ℝ) ∈
      ((List.finRange SEQ).filterMap (fun j : Fin SEQ =>
        if j.val < hi ∧ j.val ≤ qStart + i.val
        then some (aftgKVG qT kT vT keyScale i d j) else none)).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    rw [List.mem_map]
    refine ⟨aftgKVG qT kT vT keyScale i d ⟨0, hSEQ⟩, ?_, rfl⟩
    rw [List.mem_filterMap]
    refine ⟨⟨0, hSEQ⟩, List.mem_finRange _, ?_⟩
    rw [if_pos ⟨show (0:Nat) < hi from by omega, Nat.zero_le _⟩]
  have hle := aftg_mem_le_foldr_sup _ _ hmem
  intro hbot
  exact absurd (le_bot_iff.mp (hbot ▸ hle)) WithBot.coe_ne_bot

/-- General seed-1 = seed-0 on nonempty windows. -/
theorem aftgStateBot1G_eq_aftgStateBotG (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hne : aftgRunningMaxG qT kT vT keyScale qStart hi i d ≠ ⊥) :
    aftgStateBot1G qT kT vT keyScale qStart hi i d
      = aftgStateBotG qT kT vT keyScale qStart hi i d := by
  have hxs : aftgKeysUptoG qT kT vT keyScale qStart hi i d ≠ [] := by
    intro h; apply hne; unfold aftgRunningMaxG; rw [h]; rfl
  unfold aftgStateBot1G aftgStateBotG
  exact osStepBot_bot_seed_indep _ hxs 1 0 0 0

/-- General seed-1 state at the empty window is `(⊥, 1, 0)`. -/
theorem aftgStateBot1G_zero (qStart : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgStateBot1G qT kT vT keyScale qStart 0 i d = (⊥, 1, 0) := by
  unfold aftgStateBot1G aftgKeysUptoG
  rw [show (List.finRange SEQ).filterMap
        (fun j : Fin SEQ => if j.val < 0 ∧ j.val ≤ qStart + i.val
          then some (aftgKVG qT kT vT keyScale i d j) else none) = [] from by
    apply List.filterMap_eq_nil_iff.mpr; intro j _; simp]
  rfl

/-- General denominator channel-independence. -/
theorem aftgStateBotG_snd_fst_indep (qStart hi : Nat) (i : Fin BLOCK_M) (d d' : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.1
      = (aftgStateBotG qT kT vT keyScale qStart hi i d').2.1 := by
  rw [aftgStateBotG_snd_fst, aftgStateBotG_snd_fst,
    aftgRunningMaxG_eq qT kT vT keyScale qStart hi i d d']
  congr 2
  unfold aftgKeysUptoG
  rw [List.map_filterMap, List.map_filterMap]
  refine congrArg List.sum (List.filterMap_congr ?_)
  intro j _
  by_cases hj : j.val < hi ∧ j.val ≤ qStart + i.val <;> simp [aftgKVG, hj]

/-- General full-window bridge: at `hi = SEQ`, the causal ⊥-seeded key list equals
the predicate-filtered `attnKeyListPred` with `causalKeep qStart`. -/
theorem aftgKeysUptoG_full_eq_pred (qStart : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    aftgKeysUptoG qT kT vT keyScale qStart SEQ i d
      = attnKeyListPred qT kT vT keyScale (fun a b => causalKeep qStart a b) i d := by
  unfold aftgKeysUptoG attnKeyListPred aftgKVG
  apply List.filterMap_congr
  intro j _
  have hjlt : j.val < SEQ := j.isLt
  have hiff : causalKeep qStart i j ↔ j.val ≤ qStart + i.val := by
    unfold causalKeep; omega
  by_cases hc : causalKeep qStart i j
  · rw [if_pos ⟨hjlt, hiff.mp hc⟩, if_pos hc]
  · rw [if_neg (fun hh => hc (hiff.mpr hh.2)), if_neg hc]

/-- General sentinel boundedness side-condition (**causal-only, non-strict**).

The kernel masks every non-causal key with the `-1e6` `tl.where` sentinel, so only
the **kept** (causal) keys `j ≤ qStart + i` need a boundedness assumption, and only
`≥ -1e6` (non-strict) is required: the masked keys are pinned to the sentinel and
never raise the running max. -/
def aftgScoreBoundG
    (qT : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kT vT : TileIndex [SEQ, BLOCK_DMODEL] → ℝ)
    (keyScale : Fin SEQ → ℝ) (qStart : Nat) : Prop :=
  ∀ (j : Fin SEQ) (i : Fin BLOCK_M), j.val ≤ qStart + i.val →
    (0:ℝ) - 1000000.0
      ≤ keyScale j * Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (i, e, PUnit.unit) * kT (j, e, PUnit.unit))

/-- General: under the **causal-only, non-strict** `aftgScoreBoundG`, the running
max over a nonempty causal window is `≥` the `-1e6` sentinel. Key `0` is always
causal-kept (`0 ≤ qStart + i`), so its kept score `≥ -1e6` (the weaker bound) is a
member of the window's `foldr ⊔ ⊥`, hence a lower bound on the running max. -/
theorem aftgRunningMaxG_gt_sentinel (qStart hi : Nat) (hhi : 1 ≤ hi) (hSEQ : 0 < SEQ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hsb : aftgScoreBoundG qT kT vT keyScale qStart) :
    aftgRunningMaxG qT kT vT keyScale qStart hi i d ≥ some (-1000000.0 : ℝ) := by
  unfold aftgRunningMaxG aftgKeysUptoG
  have hkey0 : ((-1000000.0 : ℝ)) ≤ ((aftgKVG qT kT vT keyScale i d ⟨0, hSEQ⟩).1 : ℝ) := by
    have := hsb ⟨0, hSEQ⟩ i (by simp)
    simpa [aftgKVG] using this
  have hmem : ((((aftgKVG qT kT vT keyScale i d ⟨0, hSEQ⟩).1 : ℝ)) : WithBot ℝ) ∈
      ((List.finRange SEQ).filterMap (fun j : Fin SEQ =>
        if j.val < hi ∧ j.val ≤ qStart + i.val
        then some (aftgKVG qT kT vT keyScale i d j) else none)).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ)) := by
    rw [List.mem_map]
    refine ⟨aftgKVG qT kT vT keyScale i d ⟨0, hSEQ⟩, ?_, rfl⟩
    rw [List.mem_filterMap]
    refine ⟨⟨0, hSEQ⟩, List.mem_finRange _, ?_⟩
    rw [if_pos ⟨show (0:Nat) < hi from by omega, Nat.zero_le _⟩]
  have hle := aftg_mem_le_foldr_sup _ _ hmem
  refine le_trans ?_ hle
  exact (WithBot.coe_le_coe).mpr hkey0

/-- General `aftgBlockG` map-and-sum: reindex block `c`'s causal window onto
`Fin BLOCK_N` lanes (global key `c·BLOCK_N + jL`). -/
theorem aftgBlockG_map_sum (qStart BLOCK_N c : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ SEQ) (h : ℝ × ℝ → ℝ) :
    ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map h).sum
      = ∑ jL : Fin BLOCK_N,
          (if (c * BLOCK_N + jL.val) ≤ qStart + i.val then
            h (aftgKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩)
           else 0) := by
  rw [aftgBlockG, aftg_filterMap_finRange_sum SEQ
    (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val)
    (fun j => aftgKVG qT kT vT keyScale i d j) h]
  rw [show (∑ j : Fin SEQ, if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
            then h (aftgKVG qT kT vT keyScale i d j) else 0)
        = ∑ j ∈ Finset.univ.filter (fun j : Fin SEQ => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N),
            (if j.val ≤ qStart + i.val then h (aftgKVG qT kT vT keyScale i d j) else 0) from by
    rw [Finset.sum_filter]
    refine Finset.sum_congr rfl (fun j _ => ?_)
    by_cases hwj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N
    · by_cases hcj : j.val ≤ qStart + i.val
      · rw [if_pos ⟨hwj.1, hwj.2, hcj⟩, if_pos hwj, if_pos hcj]
      · rw [if_neg (fun hh => hcj hh.2.2), if_pos hwj, if_neg hcj]
    · rw [if_neg (fun hh => hwj ⟨hh.1, hh.2.1⟩), if_neg hwj]]
  symm
  have hexp : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := by ring
  refine Finset.sum_bij
    (i := fun jL _ => (⟨c * BLOCK_N + jL.val, by have := jL.isLt; omega⟩ : Fin SEQ)) ?_ ?_ ?_ ?_
  · intro jL _
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    have := jL.isLt; omega
  · intro a _ b _ hab
    apply Fin.ext
    have : c * BLOCK_N + a.val = c * BLOCK_N + b.val := by simpa using congrArg Fin.val hab
    omega
  · intro j hj
    have hj2 : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N := (Finset.mem_filter.mp hj).2
    exact ⟨⟨j.val - c * BLOCK_N, by omega⟩, Finset.mem_univ _, by apply Fin.ext; simp only; omega⟩
  · intro jL _; rfl

/-- General `aftgBlockG` running-sup bridge. -/
theorem aftgBlockG_blockSup (qStart BLOCK_N c : Nat) (i d : Fin BLOCK_M) (d2 : Fin BLOCK_DMODEL)
    (hwin : (c + 1) * BLOCK_N ≤ SEQ) :
    ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d2).map (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥
      = Finset.univ.sup (fun jL : Fin BLOCK_N =>
          if (c * BLOCK_N + jL.val) ≤ qStart + i.val then
            (((aftgKVG qT kT vT keyScale i d2 ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else (⊥ : WithBot ℝ)) := by
  rw [show (aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d2).map (fun p => ((p.1 : ℝ) : WithBot ℝ))
      = ((List.finRange SEQ).filterMap (fun j : Fin SEQ =>
          if c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
          then some ((aftgKVG qT kT vT keyScale i d2 j).1) else none)).map
            (fun x => ((x : ℝ) : WithBot ℝ)) from by
    unfold aftgBlockG
    rw [List.map_filterMap, List.map_filterMap]
    apply List.filterMap_congr
    intro j _
    by_cases hj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val <;> simp [hj]]
  rw [aftg_filterMap_foldr_sup SEQ
    (fun j => c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val)
    (fun j => (aftgKVG qT kT vT keyScale i d2 j).1)]
  have hexp : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := by ring
  apply le_antisymm
  · apply Finset.sup_le; intro j _
    by_cases hj : c * BLOCK_N ≤ j.val ∧ j.val < (c + 1) * BLOCK_N ∧ j.val ≤ qStart + i.val
    · rw [if_pos hj]
      have hjL : j.val - c * BLOCK_N < BLOCK_N := by omega
      have hfin : (⟨c * BLOCK_N + (j.val - c * BLOCK_N), by omega⟩ : Fin SEQ) = j := by apply Fin.ext; simp only; omega
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨j.val - c * BLOCK_N, hjL⟩ : Fin BLOCK_N)))
      simp only
      rw [if_pos (show c * BLOCK_N + (j.val - c * BLOCK_N) ≤ qStart + i.val from by have := hj.2.2; omega)]
      apply le_of_eq
      rw [hfin]
    · rw [if_neg hj]; exact bot_le
  · apply Finset.sup_le; intro jL _
    have hb : c * BLOCK_N + jL.val < SEQ := by have := jL.isLt; omega
    by_cases hkeep : c * BLOCK_N + jL.val ≤ qStart + i.val
    · rw [if_pos hkeep]
      refine le_trans ?_ (Finset.le_sup (Finset.mem_univ (⟨c * BLOCK_N + jL.val, hb⟩ : Fin SEQ)))
      simp only
      rw [if_pos (by have := jL.isLt; exact ⟨by omega, by omega, hkeep⟩)]
    · rw [if_neg hkeep]; exact bot_le

/-! ### General masked-block bridge layer ([BLOCK_M,BLOCK_N], -1e6 sentinel, HEAD_ACTIVE) -/

/-- General canonical axis-1 index of `[BLOCK_M, BLOCK_N]`. -/
abbrev aftgAx1G (BLOCK_M BLOCK_N : Nat) : Fin [BLOCK_M, BLOCK_N].length := ⟨1, by simp⟩

/-- General `reduceMax` row. -/
theorem aftg_reduceMaxDrop_rowG (BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N)
    (qk : Tile .real [BLOCK_M, BLOCK_N]) (rmaxT : Tile .real [BLOCK_M])
    (hrm : Tile.reduceMaxDrop (aftgAx1G BLOCK_M BLOCK_N) qk = some rmaxT)
    (i : Fin BLOCK_M) (g : Fin BLOCK_N → WithBot ℝ)
    (hqk : ∀ jL : Fin BLOCK_N,
        qk.data (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL) = g jL) :
    rmaxT.data (i, PUnit.unit) = Finset.univ.sup g := by
  unfold Tile.reduceMaxDrop at hrm
  rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) from hBN)] at hrm
  rw [← Option.some.inj hrm]
  simp only [Finset.sup'_eq_sup]
  exact Finset.sup_congr rfl (fun jL _ => hqk jL)

/-- General `q·k` score cell (HEAD_ACTIVE). -/
theorem aftg_score_cellG (BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (qStart SN : Nat) (jL : Fin BLOCK_N)
    (i : Fin BLOCK_M) (qsc ksc : ℝ)
    (qm : TileIndex [BLOCK_M, BLOCK_DMODEL] → ℝ) (kread : Fin BLOCK_DMODEL → Fin BLOCK_N → ℝ)
    (qtile : Tile .real [BLOCK_M, BLOCK_DMODEL]) (ktile : Tile .real [BLOCK_DMODEL, BLOCK_N])
    (kscT : Tile .real [])
    (hjLwin : jL.val < SN)
    (hqtile : ∀ e : Fin BLOCK_DMODEL, qtile.data (i, e, PUnit.unit)
        = some (if e.val < HEAD_ACTIVE then qm (i, e, PUnit.unit) else 0))
    (hktile : ∀ e : Fin BLOCK_DMODEL, ktile.data (e, jL, PUnit.unit)
        = some (if jL.val < SN ∧ e.val < HEAD_ACTIVE then kread e jL else 0))
    (hkscT : kscT.data PUnit.unit = some ksc) :
    (Tile.bop NumericDType.real.mul Broadcast.scalarR
        (Tile.bop NumericDType.real.mul Broadcast.scalarR
          ⟨fun ix => (Tile.dot [] qtile ktile).data ix⟩
          (Tile.scalar (some qsc))) kscT).data (i, jL, PUnit.unit)
      = some ((Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
          (if e.val < HEAD_ACTIVE then qm (i, e, PUnit.unit) else 0) * kread e jL)) * qsc * ksc) := by
  have hdot : (Tile.dot [] qtile ktile).data (i, jL, PUnit.unit)
      = some (Finset.univ.sum (fun e : Fin BLOCK_DMODEL =>
          (if e.val < HEAD_ACTIVE then qm (i, e, PUnit.unit) else 0) * kread e jL)) := by
    rw [Tile.dot_nil_data]
    rw [show (@Finset.sum (Fin BLOCK_DMODEL) (WithBot ℝ) _ Finset.univ
          (fun e => Option.map₂ (· * ·) (qtile.data (i, e, PUnit.unit)) (ktile.data (e, jL, PUnit.unit))))
        = @Finset.sum (Fin BLOCK_DMODEL) (WithBot ℝ) _ Finset.univ
          (fun e => (some ((if e.val < HEAD_ACTIVE then qm (i, e, PUnit.unit) else 0) * kread e jL) : WithBot ℝ))
        from Finset.sum_congr rfl (fun e _ => by
          rw [hqtile e, hktile e]
          simp only [Option.map₂, Option.bind, Option.map]
          refine congrArg some ?_
          by_cases he : e.val < HEAD_ACTIVE
          · rw [if_pos he, if_pos ⟨hjLwin, he⟩]
          · rw [if_neg he, if_neg (by simp [he])]; ring)]
    rw [WithBot.sum_someTerm_eq_some]
  simp only [Tile.bop_data, Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    Tile.scalar, NumericDType.mul, hdot, hkscT]
  simp only [WithBot.realMul, Option.map₂, Option.bind, Option.map]

/-! ### General per-statement op-eval recipes -/

set_option maxHeartbeats 1600000 in
/-- General `k_mask` recipe ([BLOCK_DMODEL, BLOCK_N]). -/
theorem aftg_kmask_evalG (s : BlockState) (SN NCTX HA BLOCK_N BLOCK_DMODEL : Nat)
    (offsn : Tile .nat [BLOCK_N])
    (hoffsn : s.regs .nat [BLOCK_N] "offs_n" = some offsn)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.boolAnd (Broadcast.consL (Broadcast.consR Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
          (Op.sub .nat Broadcast.nil (Op.constNat NCTX) (Op.ref .nat [] "start_n")))
        (Op.expandDim ⟨1, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HA)))) s
      = some ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
          (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (NCTX - SN))
            && (ComparableDType.nat.lt idx.1.val HA)⟩ := by
  rw [aftg_evalOp_boolAnd]
  simp only [evalOp_lt, evalOp.eq_def, evalOp_constNat, evalOp_arange, hoffsn, hsn,
    Option.bind_eq_bind, Option.bind_some, Option.bind]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec, Tile.scalar,
    Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
    Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
    TileShape.dropInsertedIndex, NumericDType.sub]

set_option maxHeartbeats 1600000 in
/-- General `mask` (causal keep) recipe. -/
theorem aftg_mask_evalG (s : BlockState) (SN BLOCK_M BLOCK_N : Nat)
    (offsm : Tile .nat [BLOCK_M]) (offsn : Tile .nat [BLOCK_N])
    (hoffsm : s.regs .nat [BLOCK_M] "offs_m" = some offsm)
    (hoffsn : s.regs .nat [BLOCK_N] "offs_n" = some offsn)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN)) :
    evalOp (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m"))
        (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "start_n")
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")))) s
      = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
            (SN + offsn.data (idx.2.1, PUnit.unit))⟩ := by
  rw [aftg_evalOp_ge]
  simp only [evalOp_add, evalOp.eq_def, hoffsm, hoffsn, hsn,
    Option.bind_eq_bind, Option.bind_some, Option.bind]
  refine congrArg some ?_
  ext idx
  simp only [Tile.cop_data, Tile.bop_data, Tile.expandDim_data, Tile.scalar,
    Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
    Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
    TileShape.dropInsertedIndex, NumericDType.add]

set_option maxHeartbeats 1600000 in
/-- General `m_ij = maximum(m_i, max(qk,1))` recipe. -/
theorem aftg_mij_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat)
    (mtile : Tile .real [BLOCK_M]) (qktile : Tile .real [BLOCK_M, BLOCK_N]) (rmaxT : Tile .real [BLOCK_M])
    (hmi : s.regs .real [BLOCK_M] "m_i" = some mtile)
    (hqk : s.regs .real [BLOCK_M, BLOCK_N] "qk" = some qktile)
    (hrm : Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qktile = some rmaxT) :
    evalOp (Op.where
        (Op.gt .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "m_i")
          (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
            (Op.ref .real [BLOCK_M, BLOCK_N] "qk")))
        (Op.ref .real [BLOCK_M] "m_i")
        (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
          (Op.ref .real [BLOCK_M, BLOCK_N] "qk"))) s
      = some (Tile.select
          (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
          mtile rmaxT) := by
  have hrmax : @evalOp TileDType.real [BLOCK_M]
      (Op.reduceMax (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk")) s = some rmaxT := by
    unfold evalOp
    simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]
    exact hrm
  rw [evalOp_where]
  simp only [evalOp_gt, evalOp_ref, hmi, hrmax, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
/-- General `qk = qk - m_ij[:,None]` recipe. -/
theorem aftg_qk_sub_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat) (hax : 1 < [BLOCK_M].length.succ)
    (qktile : Tile .real [BLOCK_M, BLOCK_N]) (mc : Tile .real [BLOCK_M])
    (hqk : s.regs .real [BLOCK_M, BLOCK_N] "qk" = some qktile)
    (hmij : s.regs .real [BLOCK_M] "m_ij" = some mc) :
    evalOp (Op.sub .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "m_ij"))) s
      = some (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          qktile (Tile.expandDim ⟨1, hax⟩ mc)) := by
  have hexp : @evalOp TileDType.real [BLOCK_M, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "m_ij")) s
      = some (Tile.expandDim ⟨1, hax⟩ mc) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hmij
  rw [evalOp_sub]
  simp only [evalOp_ref, hqk, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
/-- General `p = exp2(qk)` recipe. -/
theorem aftg_p_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat) (qktile : Tile .real [BLOCK_M, BLOCK_N])
    (hqk : s.regs .real [BLOCK_M, BLOCK_N] "qk" = some qktile) :
    evalOp (Op.exp2 (Op.ref .real [BLOCK_M, BLOCK_N] "qk")) s
      = some (Tile.uop WithBot.realExp2 qktile) := by
  rw [aftg_evalOp_exp2]; simp only [evalOp_ref, hqk, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
/-- General `p = where(mask, p, 0)` recipe. -/
theorem aftg_p_mask_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat) (masktile : Tile .bool [BLOCK_M, BLOCK_N])
    (ptile : Tile .real [BLOCK_M, BLOCK_N])
    (hmask : s.regs .bool [BLOCK_M, BLOCK_N] "mask" = some masktile)
    (hp : s.regs .real [BLOCK_M, BLOCK_N] "p" = some ptile) :
    evalOp (Op.where (Op.ref .bool [BLOCK_M, BLOCK_N] "mask")
        (Op.ref .real [BLOCK_M, BLOCK_N] "p") (Op.broadcast (Op.const 0.0) [BLOCK_M, BLOCK_N])) s
      = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
          if masktile.data idx then ptile.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩ := by
  have hbcast : @evalOp TileDType.real [BLOCK_M, BLOCK_N] (Op.broadcast (Op.const 0.0) [BLOCK_M, BLOCK_N]) s
      = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => (some (0.0 : ℝ) : WithBot ℝ)⟩ :
          Tile .real [BLOCK_M, BLOCK_N]) := by
    simp only [evalOp, evalOp_const, Option.bind_eq_bind, Option.bind_some]; rfl
  rw [evalOp_where]
  simp only [evalOp_ref, hmask, hp, hbcast, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.select_data, Tile.scalar]

set_option maxHeartbeats 1600000 in
/-- General `l_ij = sum(p, 1)` recipe. -/
theorem aftg_lij_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat) (ptile : Tile .real [BLOCK_M, BLOCK_N])
    (hp : s.regs .real [BLOCK_M, BLOCK_N] "p" = some ptile) :
    evalOp (Op.reduceSum (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) Bool.false
        (Op.ref .real [BLOCK_M, BLOCK_N] "p")) s
      = some (Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) ptile) := by
  rw [evalOp_reduceSum]
  simp only [evalOp_ref, hp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
/-- General `alpha = exp2(m_i - m_ij)` recipe. -/
theorem aftg_alpha_evalG (s : BlockState) (BLOCK_M : Nat) (mi mij : Tile .real [BLOCK_M])
    (hmi : s.regs .real [BLOCK_M] "m_i" = some mi)
    (hmij : s.regs .real [BLOCK_M] "m_ij" = some mij) :
    evalOp (Op.exp2 (Op.sub .real (Broadcast.consSame Broadcast.nil)
        (Op.ref .real [BLOCK_M] "m_i") (Op.ref .real [BLOCK_M] "m_ij"))) s
      = some (Tile.uop WithBot.realExp2
          (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mi mij)) := by
  rw [aftg_evalOp_exp2, evalOp_sub]
  simp only [evalOp_ref, hmi, hmij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
/-- General `l_i = l_i * alpha + l_ij` recipe. -/
theorem aftg_li_evalG (s : BlockState) (BLOCK_M : Nat) (li alpha lij : Tile .real [BLOCK_M])
    (hli : s.regs .real [BLOCK_M] "l_i" = some li)
    (halpha : s.regs .real [BLOCK_M] "alpha" = some alpha)
    (hlij : s.regs .real [BLOCK_M] "l_ij" = some lij) :
    evalOp (Op.add .real (Broadcast.consSame Broadcast.nil)
        (Op.mul .real (Broadcast.consSame Broadcast.nil)
          (Op.ref .real [BLOCK_M] "l_i") (Op.ref .real [BLOCK_M] "alpha"))
        (Op.ref .real [BLOCK_M] "l_ij")) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
          (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) li alpha) lij) := by
  rw [evalOp_add, evalOp_mul]
  simp only [evalOp_ref, hli, halpha, hlij, Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
/-- General `acc = acc * alpha[:,None]` recipe. -/
theorem aftg_acc_rescale_evalG (s : BlockState) (BLOCK_M BLOCK_DMODEL : Nat) (hax : 1 < [BLOCK_M].length.succ)
    (acctile : Tile .real [BLOCK_M, BLOCK_DMODEL]) (alpha : Tile .real [BLOCK_M])
    (hacc : s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acctile)
    (halpha : s.regs .real [BLOCK_M] "alpha" = some alpha) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc") (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "alpha"))) s
      = some (Tile.bop NumericDType.real.mul
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          acctile (Tile.expandDim ⟨1, hax⟩ alpha)) := by
  have hexp : @evalOp TileDType.real [BLOCK_M, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.ref .real [BLOCK_M] "alpha")) s
      = some (Tile.expandDim ⟨1, hax⟩ alpha) := evalOp_expandDim_ref_of_regs _ _ _ _ _ _ halpha
  rw [evalOp_mul]
  simp only [evalOp_ref, hacc, hexp, Option.bind_eq_bind, Option.bind_some]; rfl

set_option maxHeartbeats 1600000 in
/-- General `p = p.to(fp16)` recipe. -/
theorem aftg_p_fp16_evalG (s : BlockState) (BLOCK_M BLOCK_N : Nat) (ptile : Tile .real [BLOCK_M, BLOCK_N])
    (hp : s.regs .real [BLOCK_M, BLOCK_N] "p" = some ptile) :
    evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref .real [BLOCK_M, BLOCK_N] "p")) s
      = some ⟨fun i => FloatDType.real.cast FloatDType.fp16 (ptile.data i)⟩ := by
  simp only [evalOp_castFloat, FloatDType.toTileDType_real, evalOp_ref, hp,
    Option.bind_eq_bind, Option.bind_some]

set_option maxHeartbeats 1600000 in
/-- General `m_i = m_ij` recipe. -/
theorem aftg_mi_carry_evalG (s : BlockState) (BLOCK_M : Nat) (mij : Tile .real [BLOCK_M])
    (hmij : s.regs .real [BLOCK_M] "m_ij" = some mij) :
    evalOp (Op.ref .real [BLOCK_M] "m_ij") s = some mij := by
  rw [evalOp_ref, hmij]

set_option maxHeartbeats 1600000 in
/-- General pointer-advance recipe (second factor symbolic = HEAD_DIM). -/
theorem aftg_advance_ptr_evalG (s : BlockState) (BT BS d e : Nat) (name : RegName)
    (ptrs : Tile .ptr [BT, BS])
    (hptr : s.regs .ptr [BT, BS] name = some ptrs) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BT, BS] name)
        (Op.mul .nat Broadcast.nil (Op.constNat d) (Op.constNat e))) s
      = some (Tile.ptrAdd Broadcast.scalarR ptrs
          (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar d) (Tile.scalar e))) := by
  simp only [evalOp, evalOp_ref, evalOp_constNat, hptr, Option.bind_eq_bind, Option.bind_some]

/-- General `reduceMaxDrop` over axis 1 of `[BLOCK_M, BLOCK_N]` succeeds (`0 < BLOCK_N`). -/
theorem aftg_reduceMaxDrop1_someG (BLOCK_M BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (x : Tile .real [BLOCK_M, BLOCK_N]) :
    ∃ t, Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) x = some t := by
  unfold Tile.reduceMaxDrop
  rw [dif_pos (show 0 < TileShape.axisDim [BLOCK_M, BLOCK_N] (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) from hBN)]
  exact ⟨_, rfl⟩

set_option maxHeartbeats 1600000 in
/-- General `m_ij = aftgRunningMaxG((c+1)·BLOCK_N)` (masked with -1e6 sentinel). -/
theorem aftg_mij_reg_eq_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M)
    (hsb : aftgScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mtile rmaxT mijT : Tile .real [BLOCK_M])
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hrmax : Tile.reduceMaxDrop (aftgAx1G BLOCK_M BLOCK_N) qkSentT = some rmaxT)
    (hmtile : mtile.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩)
    (hmij : mijT = Tile.select
        (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT) mtile rmaxT) :
    mijT.data (i, PUnit.unit)
      = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ := by
  have hSEQ : 0 < SEQ := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  have hrmaxcell : rmaxT.data (i, PUnit.unit)
      = Finset.univ.sup (fun jL : Fin BLOCK_N =>
          if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ)) :=
    aftg_reduceMaxDrop_rowG BLOCK_M BLOCK_N hBN qkSentT rmaxT hrmax i _ hsent
  rw [aftgRunningMaxG_succ qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩]
  rw [aftgBlockG_blockSup qT kT vT keyScale qStart BLOCK_N c i i ⟨0, hBD⟩ hc1]
  set BSk : WithBot ℝ := Finset.univ.sup (fun jL : Fin BLOCK_N =>
      if (c * BLOCK_N + jL.val) ≤ qStart + i.val then
        (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
      else (⊥ : WithBot ℝ)) with hBSk
  set M := aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩ with hMdef
  have hBSk_le : BSk ≤ rmaxT.data (i, PUnit.unit) := by
    rw [hrmaxcell, hBSk]
    apply Finset.sup_le; intro jL _
    split
    · rename_i hk
      refine Finset.le_sup_of_le (Finset.mem_univ jL) ?_
      rw [if_pos (show qStart + i.val ≥ c * BLOCK_N + jL.val from by omega)]
    · exact bot_le
  have hr_le : rmaxT.data (i, PUnit.unit) ≤ BSk ⊔ ((-1000000.0 : ℝ) : WithBot ℝ) := by
    rw [hrmaxcell, hBSk]
    apply Finset.sup_le; intro jL _
    split
    · rename_i hk
      refine le_sup_of_le_left (Finset.le_sup_of_le (Finset.mem_univ jL) ?_)
      rw [if_pos (show c * BLOCK_N + jL.val ≤ qStart + i.val from by omega)]
    · exact le_sup_right
  have hMBSk_sentinel : ((-1000000.0 : ℝ) : WithBot ℝ) ≤ M ⊔ BSk := by
    by_cases hc0 : c = 0
    · subst hc0
      refine le_sup_of_le_right ?_
      rw [hBSk]
      refine Finset.le_sup_of_le (Finset.mem_univ (⟨0, hBN⟩ : Fin BLOCK_N)) ?_
      rw [if_pos (show 0 * BLOCK_N + (⟨0, hBN⟩ : Fin BLOCK_N).val ≤ qStart + i.val from by simp)]
      rw [WithBot.coe_le_coe]
      have hbnd := hsb ⟨0 * BLOCK_N + (⟨0, hBN⟩ : Fin BLOCK_N).val, by simp only [Nat.zero_mul, Nat.add_zero]; exact hSEQ⟩ i (by simp)
      simpa [aftgKVG] using hbnd
    · refine le_sup_of_le_left ?_
      rw [hMdef]
      exact aftgRunningMaxG_gt_sentinel qT kT vT keyScale qStart (c * BLOCK_N) (by
        have : 1 ≤ c := by omega
        calc 1 ≤ c := this
          _ ≤ c * BLOCK_N := Nat.le_mul_of_pos_right c hBN) hSEQ i ⟨0, hBD⟩ hsb
  have hdom : M ⊔ (BSk ⊔ ((-1000000.0 : ℝ) : WithBot ℝ)) = M ⊔ BSk := by
    rw [← sup_assoc, sup_eq_left.mpr hMBSk_sentinel]
  rw [hmij, Tile.select_data, Tile.cop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, ComparableDType.gt, hmtile]
  have hsqueeze : M ⊔ rmaxT.data (i, PUnit.unit) = M ⊔ BSk := by
    apply le_antisymm
    · exact sup_le_sup_left (le_trans hr_le (le_of_eq rfl)) M |>.trans (le_of_eq hdom) |>.trans (le_refl _)
    · exact sup_le_sup_left hBSk_le M
  by_cases hcmp : M > rmaxT.data (i, PUnit.unit)
  · rw [if_pos (by simpa using hcmp)]
    rw [← hsqueeze, max_eq_left (le_of_lt hcmp)]
  · rw [if_neg (by simpa using hcmp)]
    rw [← hsqueeze, max_eq_right (not_lt.mp hcmp)]

set_option maxHeartbeats 1600000 in
/-- General masked `pT` cell. -/
theorem aftg_pmT_cell_maskedG (BLOCK_N : Nat) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (jL : Fin BLOCK_N) (Mc1 : WithBot ℝ)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (kept : Bool)
    (_hkept : kept = decide (qStart + i.val ≥ c * BLOCK_N + jL.val))
    (hsent : qkSentT.data (i, jL, PUnit.unit)
        = if kept then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit) = Mc1)
    (hkeptbot : kept = Bool.true → Mc1 ≠ ⊥)
    (hpT : pT.data (i, jL, PUnit.unit)
        = if kept then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    pT.data (i, jL, PUnit.unit)
      = some (if kept then
          pow2 ((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
            ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
          else 0) := by
  rw [hpT]
  by_cases hk : kept = Bool.true
  · rw [if_pos hk]
    obtain ⟨Mr, hMr⟩ : ∃ Mr : ℝ, Mc1 = (Mr : WithBot ℝ) := by
      cases hh : Mc1 with
      | coe x => exact ⟨x, rfl⟩
      | bot => exact absurd hh (hkeptbot hk)
    simp only [Tile.uop_data, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, TileShape.dropInsertedIndex, NumericDType.sub, hmij, hMr,
      WithBot.unbotD_coe]
    rw [show qkSentT.data (i, jL, PUnit.unit) =
          (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
              ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ) from by
      rw [hsent, if_pos hk]]
    rw [if_pos hk]
    simp only [WithBot.realSub, Option.map₂, Option.bind, Option.map, WithBot.realExp2_some]
    refine congrArg some ?_
    simp only [pow2]; ring_nf
  · rw [if_neg hk, if_neg hk]
    norm_num

set_option maxHeartbeats 1600000 in
/-- General masked `p` reduceSum row. -/
theorem aftg_nume_row_sum_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hsb : aftgScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.reduceSumDrop (aftgAx1G BLOCK_M BLOCK_N) pT).data (i, PUnit.unit)
      = some ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
          (fun p => pow2 (p.1 - (aftgRunningMaxG qT kT vT keyScale
            qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).unbotD 0))).sum := by
  have hc1' : 1 ≤ (c + 1) * BLOCK_N := by
    have : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  set Mc1 := aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hMc1bot : Mc1 ≠ ⊥ := by
    rw [hMc1]; exact aftgRunningMaxG_ne_bot qT kT vT keyScale qStart ((c + 1) * BLOCK_N) hc1' (by omega) i ⟨0, hBD⟩
  rw [Tile.reduceSumDrop_data]
  have hcell : ∀ jL : Fin BLOCK_N,
      pT.data (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL)
        = some (if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            pow2 ((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
            else 0) := by
    intro jL
    rw [show (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (i, PUnit.unit) jL) = (i, jL, PUnit.unit) from rfl]
    have hsentR : qkSentT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ) := by
      rw [hsent jL]; by_cases h : qStart + i.val ≥ c * BLOCK_N + jL.val <;> simp [h]
    have hpTR : pT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ) := by
      rw [hpT jL]; by_cases h : qStart + i.val ≥ c * BLOCK_N + jL.val
      · rw [if_pos h, if_pos (decide_eq_true h)]
      · rw [if_neg h, if_neg (by simp [h])]
    exact aftg_pmT_cell_maskedG qT kT vT keyScale BLOCK_N hBD qStart c hc1 i jL Mc1 qkSentT mijT pT
      (decide (qStart + i.val ≥ c * BLOCK_N + jL.val)) (by rfl) hsentR
      (by rw [hmij]) (fun _ => hMc1bot) hpTR
  simp only [hcell]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [aftgBlockG_map_sum qT kT vT keyScale qStart BLOCK_N c i d hc1
      (fun p => pow2 (p.1 - Mc1.unbotD 0))]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  simp only [decide_eq_true_eq]
  by_cases hkp : (c * BLOCK_N + jL.val) ≤ qStart + i.val
  · rw [if_pos (show qStart + i.val ≥ c * BLOCK_N + jL.val from by omega),
        if_pos (show c * BLOCK_N + jL.val ≤ qStart + i.val from hkp)]
    rfl
  · rw [if_neg (show ¬ qStart + i.val ≥ c * BLOCK_N + jL.val from by omega),
        if_neg (show ¬ c * BLOCK_N + jL.val ≤ qStart + i.val from hkp)]

set_option maxHeartbeats 1600000 in
/-- General masked `dot(p, v)` numerator row. -/
theorem aftg_acc_dot_block_maskedG (BLOCK_N : Nat) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (vtile : Tile .real [BLOCK_N, BLOCK_DMODEL]) (vval : Fin BLOCK_N → ℝ)
    (hMc1bot : aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ ≠ ⊥)
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hmij : mijT.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ))
    (hv : ∀ jL : Fin BLOCK_N, vtile.data (jL, d, PUnit.unit) = some (vval jL))
    (hvval : ∀ jL : Fin BLOCK_N, vval jL
        = (aftgKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).2) :
    (Tile.dot [] pT vtile).data (i, d, PUnit.unit)
      = some ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
          (fun p => pow2 (p.1 - (aftgRunningMaxG qT kT vT keyScale
            qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).unbotD 0) * p.2)).sum := by
  set Mc1 := aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hpcell : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
      = some (if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
          pow2 ((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
          else 0) := by
    intro jL
    have hsentR : qkSentT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ) := by
      rw [hsent jL]; by_cases h : qStart + i.val ≥ c * BLOCK_N + jL.val <;> simp [h]
    have hpTR : pT.data (i, jL, PUnit.unit)
        = if decide (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ) := by
      rw [hpT jL]; by_cases h : qStart + i.val ≥ c * BLOCK_N + jL.val
      · rw [if_pos h, if_pos (decide_eq_true h)]
      · rw [if_neg h, if_neg (by simp [h])]
    have := aftg_pmT_cell_maskedG qT kT vT keyScale BLOCK_N hBD qStart c hc1 i jL Mc1 qkSentT mijT pT
      (decide (qStart + i.val ≥ c * BLOCK_N + jL.val)) (by rfl) hsentR (by rw [hmij]) (fun _ => hMc1bot) hpTR
    rw [this]
  rw [Tile.dot_nil_data]
  have hterm : ∀ jL : Fin BLOCK_N,
      Option.map₂ (· * ·) (pT.data (i, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))
        = some (if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            pow2 ((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
              * vval jL
            else 0) := by
    intro jL
    rw [hpcell jL, hv jL]
    simp only [Option.map₂, Option.bind, Option.map, decide_eq_true_eq]
    refine congrArg some ?_
    by_cases h : qStart + i.val ≥ c * BLOCK_N + jL.val
    · rw [if_pos h, if_pos h]
    · rw [if_neg h, if_neg h]; ring
  rw [show (@Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ
        (fun jL => Option.map₂ (· * ·) (pT.data (i, jL, PUnit.unit)) (vtile.data (jL, d, PUnit.unit))))
      = @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ (fun jL =>
          (some (if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
              pow2 ((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 - Mc1.unbotD 0)
                * vval jL else 0) : WithBot ℝ))
      from Finset.sum_congr rfl (fun jL _ => hterm jL)]
  rw [WithBot.sum_someTerm_eq_some]
  refine congrArg some ?_
  rw [aftgBlockG_map_sum qT kT vT keyScale qStart BLOCK_N c i d hc1
      (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)]
  refine Finset.sum_congr rfl (fun jL _ => ?_)
  by_cases hkp : (c * BLOCK_N + jL.val) ≤ qStart + i.val
  · rw [if_pos (show qStart + i.val ≥ c * BLOCK_N + jL.val from by omega),
        if_pos (show c * BLOCK_N + jL.val ≤ qStart + i.val from hkp)]
    rw [hvval jL,
      show (aftgKVG qT kT vT keyScale i ⟨0, hBD⟩ ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1
          = (aftgKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 from by
        simp only [aftgKVG]]
  · rw [if_neg (show ¬ qStart + i.val ≥ c * BLOCK_N + jL.val from by omega),
        if_neg (show ¬ c * BLOCK_N + jL.val ≤ qStart + i.val from hkp)]

/-- General seed-1 vs seed-0 carry cancellation. -/
theorem aftgStateBot1G_cancel (qStart BLOCK_N c : Nat) (hBN : 0 < BLOCK_N) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (Mc1 : WithBot ℝ)
    (hSEQ : 0 < SEQ) :
    let m := (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).1
    let α := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
    (aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.1 * α
        = (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.1 * α
      ∧ (aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α
        = (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α := by
  intro m α
  by_cases hc0 : c = 0
  · subst hc0
    have hmbot : m = ⊥ := by
      show (aftgStateBotG qT kT vT keyScale qStart (0 * BLOCK_N) i d).1 = ⊥
      rw [aftgStateBotG_fst_eq_runningMax, Nat.zero_mul, aftgRunningMaxG_zero]
    have hα0 : α = 0 := by
      show (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 = 0
      rw [hmbot, WithBot.realSub_bot_left, WithBot.realExp2_bot]; rfl
    rw [hα0]; simp
  · have hne : aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i d ≠ ⊥ :=
      aftgRunningMaxG_ne_bot qT kT vT keyScale qStart (c * BLOCK_N)
        (by calc 1 ≤ c := by omega
              _ ≤ c * BLOCK_N := Nat.le_mul_of_pos_right c hBN) hSEQ i d
    rw [aftgStateBot1G_eq_aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d hne]
    exact ⟨rfl, rfl⟩

/-- General empty-window sum-zero. -/
theorem aftgKeysUptoG_sum_zero_of_bot (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hbot : aftgRunningMaxG qT kT vT keyScale qStart hi i d = ⊥) (h : ℝ × ℝ → ℝ) :
    ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map h).sum = 0 := by
  rw [show aftgKeysUptoG qT kT vT keyScale qStart hi i d = [] from ?_, List.map_nil, List.sum_nil]
  by_contra hne
  obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hne
  have hmem : ((p.1 : ℝ) : WithBot ℝ) ∈
      (aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun q => ((q.1 : ℝ) : WithBot ℝ)) :=
    List.mem_map_of_mem hp
  have := aftg_mem_le_foldr_sup _ _ hmem
  rw [← aftgRunningMaxG, hbot] at this
  exact absurd (le_bot_iff.mp this) WithBot.coe_ne_bot

/-- General denom anchor. -/
theorem aftg_denom_anchorG (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.1
      = ((aftgStateBotG qT kT vT keyScale qStart hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1)).sum) := by
  rw [aftgStateBotG_snd_fst, aftgStateBotG_fst_eq_runningMax]

/-- General acc anchor. -/
theorem aftg_acc_anchorG (qStart hi : Nat) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) :
    (aftgStateBotG qT kT vT keyScale qStart hi i d).2.2
      = ((aftgStateBotG qT kT vT keyScale qStart hi i d).1.elim 0 (fun r => pow2 (-r)))
        * (0 + ((aftgKeysUptoG qT kT vT keyScale qStart hi i d).map (fun p => pow2 p.1 * p.2)).sum) := by
  rw [aftgStateBotG_snd_snd, aftgStateBotG_fst_eq_runningMax]

set_option maxHeartbeats 1600000 in
/-- General `l_i' = aftgStateBotG((c+1)·BLOCK_N).2.1` (masked). -/
theorem aftg_denom_reg_eq_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M)
    (hsb : aftgScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mtile mijT alphaT litile lijT : Tile .real [BLOCK_M])
    (pT : Tile .real [BLOCK_M, BLOCK_N])
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hlitile : litile.data (i, PUnit.unit) = some
        ((aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1))
    (hmtile : mtile.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hlijT : lijT = Tile.reduceSumDrop (aftgAx1G BLOCK_M BLOCK_N) pT)
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ)) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame Broadcast.nil)
        (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT).data (i, PUnit.unit)
      = some ((aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).2.1) := by
  have hSEQ : 0 < SEQ := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  set m := (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).1 with hm_def
  set Mc := aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩ with hMc
  set Mc1 := aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hmMc : m = Mc := by rw [hm_def, hMc, aftgStateBotG_fst_eq_runningMax]
  have hMsucc : Mc1 = m ⊔ ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    rw [hMc1, aftgRunningMaxG_succ, hmMc, ← hMc]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hsum := aftg_nume_row_sum_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 i ⟨0, hBD⟩ hsb qkSentT mijT pT hsent hmij hpT
  have hblockEq := osStepBot_block_eq m
    ((aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1)
    ((aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.2)
    ((aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).map (fun p => pow2 p.1 * p.2)).sum
    ((aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).map (fun p => pow2 p.1)).sum
    (aftgBlockG qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩)
    (by rw [aftg_denom_anchorG, zero_add, hm_def])
    (by rw [aftg_acc_anchorG, zero_add, hm_def])
    (fun hbot => aftgKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩
      (by rw [← aftgStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => aftgKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩
      (by rw [← aftgStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩).2.1
        = (Mc1, (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i ⟨0, hBD⟩).map (fun p => pow2 (p.1 - Mc1.unbotD 0))).sum,
            _).2.1 from by
    rw [aftgStateBotG_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (aftgStateBot1G_cancel qT kT vT keyScale qStart BLOCK_N c hBN i ⟨0, hBD⟩ Mc1 hSEQ).1
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  rw [hlijT]
  erw [hsum]
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul,
    hlitile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1 * α
        = (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩).2.1 * α from by
    have := hcancel; simp only [← hm_def, ← hαdef] at this ⊢; exact this]

set_option maxHeartbeats 1600000 in
/-- General `acc' = aftgStateBotG((c+1)·BLOCK_N).2.2` (masked). -/
theorem aftg_acc_reg_eq_maskedG (BLOCK_N : Nat) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (qStart : Nat) (c : Nat) (hc1 : (c + 1) * BLOCK_N ≤ SEQ) (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hsb : aftgScoreBoundG qT kT vT keyScale qStart)
    (qkSentT : Tile .real [BLOCK_M, BLOCK_N]) (mtile mijT alphaT : Tile .real [BLOCK_M])
    (acctile acc1T : Tile .real [BLOCK_M, BLOCK_DMODEL]) (pT : Tile .real [BLOCK_M, BLOCK_N])
    (vtile : Tile .real [BLOCK_N, BLOCK_DMODEL]) (vval : Fin BLOCK_N → ℝ)
    (hsent : ∀ jL : Fin BLOCK_N, qkSentT.data (i, jL, PUnit.unit)
        = if qStart + i.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale i ⟨0, hBD⟩
                ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ))
    (hacctile : acctile.data (i, d, PUnit.unit) = some
        ((aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2))
    (hmtile : mtile.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩)
    (hmij : mijT.data (i, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩)
    (halpha : alphaT = Tile.uop WithBot.realExp2
        (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
    (hacc1 : acc1T = Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
    (hpT : ∀ jL : Fin BLOCK_N, pT.data (i, jL, PUnit.unit)
        = if (qStart + i.val ≥ c * BLOCK_N + jL.val) then
            (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
              (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data (i, jL, PUnit.unit)
          else (some (0.0 : ℝ) : WithBot ℝ))
    (hv : ∀ jL : Fin BLOCK_N, vtile.data (jL, d, PUnit.unit) = some (vval jL))
    (hvval : ∀ jL : Fin BLOCK_N, vval jL
        = (aftgKVG qT kT vT keyScale i d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).2) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        acc1T (Tile.dot [] pT vtile)).data (i, d, PUnit.unit)
      = some ((aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d).2.2) := by
  have hSEQ : 0 < SEQ := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  have hc1' : 1 ≤ (c + 1) * BLOCK_N := by
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega
  set m := (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).1 with hm_def
  set Mc := aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i ⟨0, hBD⟩ with hMc
  set Mc1 := aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ with hMc1
  have hmMc : m = Mc := by
    rw [hm_def, hMc, aftgStateBotG_fst_eq_runningMax,
      aftgRunningMaxG_eq qT kT vT keyScale qStart (c * BLOCK_N) i d ⟨0, hBD⟩]
  have hmd : m = aftgRunningMaxG qT kT vT keyScale qStart (c * BLOCK_N) i d := by
    rw [hm_def, aftgStateBotG_fst_eq_runningMax]
  have hMsucc : Mc1 = m ⊔ ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map
        (fun p => ((p.1 : ℝ) : WithBot ℝ))).foldr (· ⊔ ·) ⊥ := by
    rw [hMc1, aftgRunningMaxG_eq qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i ⟨0, hBD⟩ d,
      aftgRunningMaxG_succ, hmd]
  have halphaVal : alphaT.data (i, PUnit.unit) = WithBot.realExp2 (WithBot.realSub m Mc1) := by
    rw [halpha]; show WithBot.realExp2 _ = _
    simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hmtile, hmij,
      NumericDType.sub, ← hMc, ← hMc1, hmMc]
  have hMc1bot : Mc1 ≠ ⊥ := by
    rw [hMc1]; exact aftgRunningMaxG_ne_bot qT kT vT keyScale qStart ((c + 1) * BLOCK_N) hc1' hSEQ i ⟨0, hBD⟩
  have hdot := aftg_acc_dot_block_maskedG qT kT vT keyScale BLOCK_N hBM hBD qStart c hc1 i d qkSentT mijT pT vtile vval
    (by rw [← hMc1]; exact hMc1bot) hsent hmij hpT hv hvval
  have hblockEq := osStepBot_block_eq m
    ((aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.1)
    ((aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2)
    ((aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d).map (fun p => pow2 p.1 * p.2)).sum
    ((aftgKeysUptoG qT kT vT keyScale qStart (c * BLOCK_N) i d).map (fun p => pow2 p.1)).sum
    (aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d)
    (by rw [aftg_denom_anchorG, zero_add, hm_def])
    (by rw [aftg_acc_anchorG, zero_add, hm_def])
    (fun hbot => aftgKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i d
      (by rw [← aftgStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
    (fun hbot => aftgKeysUptoG_sum_zero_of_bot qT kT vT keyScale qStart (c * BLOCK_N) i d
      (by rw [← aftgStateBotG_fst_eq_runningMax, ← hm_def]; exact hbot) _)
  rw [← hMsucc] at hblockEq
  rw [show (aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) i d).2.2
        = (Mc1, _,
            (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2
              * (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0
              + ((aftgBlockG qT kT vT keyScale qStart BLOCK_N c i d).map (fun p => pow2 (p.1 - Mc1.unbotD 0) * p.2)).sum).2.2
        from by rw [aftgStateBotG_succ]; rw [← hblockEq]]
  set α : ℝ := (WithBot.realExp2 (WithBot.realSub m Mc1)).unbotD 0 with hαdef
  have hαsome : WithBot.realExp2 (WithBot.realSub m Mc1) = some α := by
    rw [hαdef]; cases WithBot.realSub m Mc1 <;> rfl
  have hcancel := (aftgStateBot1G_cancel qT kT vT keyScale qStart BLOCK_N c hBN i d Mc1 hSEQ).2
  rw [Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex]
  erw [hdot]
  rw [hacc1, Tile.bop_data]
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Tile.expandDim_data,
    TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, hacctile, halphaVal, hαsome]
  simp only [WithBot.realAdd, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  refine congrArg some ?_
  rw [show (aftgStateBot1G qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α
        = (aftgStateBotG qT kT vT keyScale qStart (c * BLOCK_N) i d).2.2 * α from by
    have := hcancel; simp only [← hm_def, ← hαdef] at this ⊢; exact this]

/-- **General ⊥-seeded full-window state reads off the genuine closed-form spec.** -/
theorem aftgStateBotG_full_eq_spec
    (s : BlockState) (Q K V : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (i : Fin BLOCK_M) (d : Fin BLOCK_DMODEL)
    (hne : aftgRunningMaxG
        (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
        (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
        (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
        keyScale (qStartAFT2G s BLOCK_M) (BLOCK_N * numKVBlocks) i d ≠ ⊥) :
    (let st := aftgStateBotG
        (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
        (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
        (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
        keyScale (qStartAFT2G s BLOCK_M) (BLOCK_N * numKVBlocks) i d
     st.2.2 / st.2.1)
      = attnFwdTritonOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N
          BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale (i, d, PUnit.unit) := by
  simp only
  rw [aftgStateBotG_ratio_eq _ _ _ _ _ _ _ _ hne]
  rw [aftgKeysUptoG_full_eq_pred]
  rw [attnFwdTritonOutSpecG_eq_streaming]
  rw [VeriTile.Triton.osStep_foldl_eq_batch]

set_option maxRecDepth 8000 in
/-- **General body split** — the lowered general AFT2 body decomposes as
`take 22 ++ (forRange "start_n" 0 N_CTX BLOCK_N aftgLoopBodyG :: drop 23)`. The
static `forRange` (NOT `forRangeDyn`) sits at index 22 regardless of dimension
values (the AST structure is dim-value-independent), checked by `rfl`. -/
theorem aftg_body_splitG
    (Q K V QScale KScale Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat) :
    (attn_fwd_triton_surface Q K V QScale KScale Out
        sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body
      = (attn_fwd_triton_surface Q K V QScale KScale Out
          sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
          Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body.take 22
        ++ (Stmt.forRange "start_n" 0 N_CTX BLOCK_N
              (AftgFoundation.aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE)
            :: (attn_fwd_triton_surface Q K V QScale KScale Out
                sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
                Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body.drop 23) :=
  rfl

set_option maxRecDepth 8000 in
/-- `body.take 22 = aftgPreLoopG` at the contiguous symbolic layout. Checked by `rfl`. -/
theorem aftgPreLoopG_check (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z : Nat) :
    (attn_fwd_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body.take 22
      = AftgFoundation.aftgPreLoopG Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE :=
  rfl

set_option maxRecDepth 8000 in
/-- `body.drop 23 = aftgPostLoopG` at the contiguous symbolic layout. Checked by `rfl`. -/
theorem aftgPostLoopG_check (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z : Nat) :
    (attn_fwd_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body.drop 23
      = AftgFoundation.aftgPostLoopG Out N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE :=
  rfl

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General preLoop head execution.** -/
theorem aftgPreLoopHeadG_eval
    (s : BlockState) (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s1, stepStmts (AftgFoundation.aftgPreLoopHeadG stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL) s = some s1
      ∧ s1.pids = s.pids ∧ s1.mem = s.mem ∧ (∀ rg o, s1.undef rg o = 0)
      ∧ s1.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s1.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s1.regs .nat [] "qvk_offset" = some (Tile.scalar (baseOffsetAFT2G s stride_qz stride_qh H))
      ∧ s1.regs .nat [] "q_scale_offset"
          = some (Tile.scalar (s.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M)))
      ∧ s1.regs .nat [] "k_scale_offset"
          = some (Tile.scalar (s.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N)))
      ∧ s1.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val))
      ∧ s1.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ s1.regs .nat [BLOCK_DMODEL] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) := by
  unfold AftgFoundation.aftgPreLoopHeadG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)) _
        = some (Tile.scalar (s.pids 1 / H)) from by
      rw [evalOp]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mod .nat Broadcast.nil (Op.ref .nat [] "off_hz") (Op.constNat H)) _
        = some (Tile.scalar (s.pids 1 % H)) from by
      rw [evalOp]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_z") (Op.constNat stride_qz))
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_h") (Op.constNat stride_qh))) _
        = some (Tile.scalar (baseOffsetAFT2G s stride_qz stride_qh H)) from by
      rw [evalOp_add, evalOp_mul, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.floorDiv .nat Broadcast.nil (Op.ref .nat [] "qvk_offset") (Op.constNat HEAD_DIM)) _
        = some (Tile.scalar (baseOffsetAFT2G s stride_qz stride_qh H / HEAD_DIM)) from by
      rw [evalOp]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_M)) (Op.constNat 1))
          (Op.constNat BLOCK_M))) _
        = some (Tile.scalar (s.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M))) from by
      rw [evalOp_mul, evalOp_div, evalOp_sub, evalOp_add]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] "off_hz")
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil (Op.add .nat Broadcast.nil (Op.constNat N_CTX) (Op.constNat BLOCK_N)) (Op.constNat 1))
          (Op.constNat BLOCK_N))) _
        = some (Tile.scalar (s.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N))) from by
      rw [evalOp_mul, evalOp_div, evalOp_sub, evalOp_add]
      simp only [evalOp_ref, evalOp_constNat, BlockState.setReg_same, BlockState.setReg_pids,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        Option.bind_eq_bind, Option.bind_some]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .nat Broadcast.scalarL
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "start_m") (Op.constNat BLOCK_M)) (Op.arange BLOCK_M)) _
        = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val)) from by
      rw [evalOp_add, evalOp_mul]
      simp only [evalOp_ref, evalOp_constNat, evalOp_arange, BlockState.setReg_same,
        BlockState.setReg_pids, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
        not_false_eq_true, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Tile.vec_data, Broadcast.leftIndex,
        Broadcast.rightIndex, NumericDType.add, NumericDType.mul]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_N) _ = some (Tile.vec (fun j : Fin BLOCK_N => j.val)) from
      evalOp_arange BLOCK_N _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.arange BLOCK_DMODEL) _ = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val)) from
      evalOp_arange BLOCK_DMODEL _))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · intro rg o; exact hundef rg o
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]

/-! ### General streamed-pointer closed cell-forms -/

/-- General `K_ptrs` after `c` blocks: `[BLOCK_DMODEL, BLOCK_N]`, cell `(e, jL)` →
`K[base + e + (c·BLOCK_N + jL)·HEAD_DIM]`. -/
noncomputable def kPtrsAFT2G (s0 : BlockState) (K : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) : Tile .ptr [BLOCK_DMODEL, BLOCK_N] :=
  ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
    (K.cast, baseOffsetAFT2G s0 stride_qz stride_qh H + idx.1.val + (c * BLOCK_N + idx.2.1.val) * HEAD_DIM)⟩

/-- General `K_scale_ptr` after `c` blocks. -/
noncomputable def kScalePtrAFT2G (s0 : BlockState) (KScale : RegionName)
    (N_CTX BLOCK_N c : Nat) : Tile .ptr [] :=
  ⟨fun _ : TileIndex [] => (KScale.cast, s0.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N) + c)⟩

/-- General `V_ptrs` after `c` blocks: `[BLOCK_N, BLOCK_DMODEL]`, cell `(jL, d)` →
`V[base + (c·BLOCK_N + jL)·HEAD_DIM + d]`. -/
noncomputable def vPtrsAFT2G (s0 : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) : Tile .ptr [BLOCK_N, BLOCK_DMODEL] :=
  ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
    (V.cast, baseOffsetAFT2G s0 stride_qz stride_qh H + (c * BLOCK_N + idx.1.val) * HEAD_DIM + idx.2.1.val)⟩

/-- General `O_block_ptr` (constant): `[BLOCK_M, BLOCK_DMODEL]`, cell `(i, e)` →
`Out[base + (pids0·BLOCK_M + i)·HEAD_DIM + e]`. -/
noncomputable def oBlockPtrAFT2G (s0 : BlockState) (Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL : Nat) : Tile .ptr [BLOCK_M, BLOCK_DMODEL] :=
  ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
    (Out.cast, baseOffsetAFT2G s0 stride_qz stride_qh H + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val)⟩

theorem kPtrsAFT2G_succ (s0 : BlockState) (K : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile.ptrAdd Broadcast.scalarR (kPtrsAFT2G s0 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c)
        (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM))
      = kPtrsAFT2G s0 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL (c + 1) := by
  ext idx
  · rfl
  · simp only [kPtrsAFT2G, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, Broadcast.leftIndex_nil,
      Broadcast.rightIndex_nil, NumericDType.nat_mul]
    ring

theorem kScalePtrAFT2G_succ (s0 : BlockState) (KScale : RegionName) (N_CTX BLOCK_N c : Nat) :
    Tile.ptrAdd Broadcast.nil (kScalePtrAFT2G s0 KScale N_CTX BLOCK_N c) (Tile.scalar 1)
      = kScalePtrAFT2G s0 KScale N_CTX BLOCK_N (c + 1) := by
  ext idx
  · rfl
  · simp only [kScalePtrAFT2G, Tile.ptrAdd_data, Tile.scalar,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil]
    omega

theorem vPtrsAFT2G_succ (s0 : BlockState) (V : RegionName)
    (stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c : Nat) :
    Tile.ptrAdd Broadcast.scalarR (vPtrsAFT2G s0 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c)
        (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM))
      = vPtrsAFT2G s0 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL (c + 1) := by
  ext idx
  · rfl
  · simp only [vPtrsAFT2G, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR, Broadcast.leftIndex_nil,
      Broadcast.rightIndex_nil, NumericDType.nat_mul]
    ring

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General preLoop tail execution.** -/
theorem aftgPreLoopTailG_eval
    (s1 : BlockState) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (hundef : ∀ rg o, s1.undef rg o = 0)
    (hstartm : s1.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0)))
    (hoffhz : s1.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1)))
    (hqvk : s1.regs .nat [] "qvk_offset" = some (Tile.scalar (baseOffsetAFT2G s1 stride_qz stride_qh H)))
    (hqso : s1.regs .nat [] "q_scale_offset"
        = some (Tile.scalar (s1.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M))))
    (hkso : s1.regs .nat [] "k_scale_offset"
        = some (Tile.scalar (s1.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N))))
    (hoffsm : s1.regs .nat [BLOCK_M] "offs_m"
        = some (Tile.vec (fun r : Fin BLOCK_M => s1.pids 0 * BLOCK_M + r.val)))
    (hoffsn : s1.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val)))
    (hoffsk : s1.regs .nat [BLOCK_DMODEL] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))) :
    ∃ s0, stepStmts (AftgFoundation.aftgPreLoopTailG Q K V QScale KScale Out HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s1 = some s0
      ∧ s0.pids = s1.pids ∧ s0.mem = s1.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0))
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1))
      ∧ s0.regs .real [BLOCK_M] "m_i" = some ⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [BLOCK_M] "l_i" = some ⟨fun _ : TileIndex [BLOCK_M] => (some (1 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => (some (0 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => s1.pids 0 * BLOCK_M + r.val))
      ∧ s0.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ s0.regs .nat [BLOCK_DMODEL] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
      ∧ s0.regs .real [] "q_scale" = some ⟨fun _ : TileIndex [] =>
          some (s1.readMem QScale ((s1.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M) + s1.pids 0)))⟩
      ∧ s0.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some (kPtrsAFT2G s1 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s0.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFT2G s1 KScale N_CTX BLOCK_N 0)
      ∧ s0.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some (vPtrsAFT2G s1 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s0.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = some (oBlockPtrAFT2G s1 Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL)
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          if s1.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
          then some (qTileAFT2G s1 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩ := by
  unfold AftgFoundation.aftgPreLoopTailG
  -- stmt 11: Q_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase Q) _ _ _ _
      (aftg_evalOp_ptrBase Q _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1))) _
          = some _ from by
        rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_mul]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsm,
          evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsk]
        rw [evalOp_ref, hqvk]
        simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 12: Q_scale_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.nil (Op.ptrBase QScale) _ _ _ _
      (aftg_evalOp_ptrBase QScale _)
      (show evalOp (Op.add .nat Broadcast.nil (Op.ref .nat [] "q_scale_offset") (Op.ref .nat [] "start_m")) _
          = some _ from by
        simp only [evalOp_add, evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true, hqso, hstartm, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 13: K_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase K) _ _ _ _
      (aftg_evalOp_ptrBase K _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM))) _
          = some _ from by
        rw [evalOp_add, evalOp_add, evalOp_mul]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsk)),
            evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsn))]
        rw [evalOp_ref, regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hqvk)]
        simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 14: K_scale_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.nil (Op.ptrBase KScale) _ _ _ _
      (aftg_evalOp_ptrBase KScale _)
      (show evalOp (Op.ref .nat [] "k_scale_offset") _ = some _ from by
        simp only [evalOp_ref, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
          not_false_eq_true]
        rw [hkso])))]
  -- stmt 15: V_ptrs
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase V) _ _ _ _
      (aftg_evalOp_ptrBase V _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1))) _
          = some _ from by
        rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_mul]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsn)))),
            evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsk))))]
        rw [evalOp_ref, regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hqvk)))]
        simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 16: O_block_ptr
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_ptrAdd_of Broadcast.scalarL (Op.ptrBase Out) _ _ _ _
      (aftg_evalOp_ptrBase Out _)
      (show evalOp (Op.add .nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "qvk_offset")
            (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat HEAD_DIM)))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_DMODEL] "offs_k")) (Op.constNat 1))) _
          = some _ from by
        rw [evalOp_add, evalOp_add, evalOp_mul, evalOp_mul]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                  (regs_setReg_chain (by decide) hoffsm))))),
            evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                  (regs_setReg_chain (by decide) hoffsk)))))]
        rw [evalOp_ref, regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) hqvk))))]
        simp only [evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 17: m_i = full ⊥
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) Op.negInf) _
        = some (⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩ : Tile .real [BLOCK_M]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, evalOp_negInf, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      rfl))]
  -- stmt 18: l_i = full 1.0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.add .real Broadcast.scalarR (Op.full [BLOCK_M] (Op.const 0)) (Op.const 1.0)) _
        = some (⟨fun _ : TileIndex [BLOCK_M] => (some (1 : ℝ) : WithBot ℝ)⟩ : Tile .real [BLOCK_M]) from by
      rw [evalOp_add]
      simp only [evalOp_full, evalOp_const, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_; ext idx
      simp only [Tile.bop_data, Tile.scalar_data, Broadcast.leftIndex, Broadcast.rightIndex,
        NumericDType.add, WithBot.realAdd, Option.map₂, Option.bind, Option.map]
      norm_num))]
  -- stmt 19: acc = full 0
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.full [BLOCK_M, BLOCK_DMODEL] (Op.const 0)) _
        = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => (some (0 : ℝ) : WithBot ℝ)⟩ : Tile .real [BLOCK_M, BLOCK_DMODEL]) from by
      simp only [evalOp_full, evalOp_const]
      rfl))]
  -- stmt 20: q = masked load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_load_ptr_mask_of (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "Q_ptrs") _ _ _ _
      (by rw [evalOp_ref,
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_same])
      (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
          (Op.expandDim ⟨0, by simp⟩
            (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) _
          = some _ from by
        rw [aftg_evalOp_boolAnd, evalOp_lt]
        erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
            (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                  (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                    (regs_setReg_chain (by decide) hoffsm))))))))),
          evalOp_expandDim]
        simp only [evalOp_lt, evalOp_arange, evalOp_constNat,
          Option.bind_some, Option.bind_eq_bind]
        rfl)))]
  -- stmt 21: q_scale = load
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (aftg_evalOp_load_ptr_none_of (Op.ref .ptr [] "Q_scale_ptr") _ _
      (by rw [evalOp_ref,
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide),
            BlockState.setReg_same])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rfl
  · intro rg o; exact hundef rg o
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids, hstartm]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids, hoffhz]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids, hoffsm]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, hoffsn]
  · simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, hoffsk]
  · -- q_scale
    simp only [BlockState.setReg_same, BlockState.setReg_pids]
    refine congrArg some ?_
    ext _
    simp only [BlockState.readMem, BlockState.setReg_mem, castTile_self,
      Tile.ptrAdd_data, Tile.scalar_data, Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
      Tile.bop_data, Region.cast, NumericDType.nat_add, Nat.zero_add]
  · -- K_ptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [kPtrsAFT2G, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Region.cast]
    · simp only [kPtrsAFT2G, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data, Tile.expandDim_data,
        Tile.vec_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, NumericDType.nat_add,
        NumericDType.nat_mul, baseOffsetAFT2G]
      ring_nf
  · -- K_scale_ptr
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [kScalePtrAFT2G, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_nil,
        Broadcast.rightIndex_nil, Region.cast]
    · simp only [kScalePtrAFT2G, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_nil,
        Broadcast.rightIndex_nil, Nat.zero_add, Nat.add_zero]
  · -- V_ptrs
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [vPtrsAFT2G, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Region.cast]
    · simp only [vPtrsAFT2G, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data, Tile.expandDim_data,
        Tile.vec_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, NumericDType.nat_add,
        NumericDType.nat_mul, baseOffsetAFT2G]
      ring_nf
  · -- O_block_ptr
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same]
    refine congrArg some ?_
    ext idx
    · simp only [oBlockPtrAFT2G, Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Region.cast]
    · simp only [oBlockPtrAFT2G, Tile.ptrAdd_data, Tile.scalar, Tile.bop_data, Tile.expandDim_data,
        Tile.vec_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
        Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
        TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul, NumericDType.nat_add,
        NumericDType.nat_mul, baseOffsetAFT2G]
      ring_nf
  · -- q
    simp only [BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
      BlockState.setReg_same, BlockState.setReg_pids]
    refine congrArg some ?_
    ext idx
    simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
      Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
      TileShape.dropInsertedIndex]
    rw [show (ComparableDType.nat.lt (s1.pids 0 * BLOCK_M + idx.1.val) N_CTX
          && ComparableDType.nat.lt idx.2.1.val HEAD_ACTIVE)
        = decide (s1.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE) from by
      rw [Bool.eq_iff_iff]; simp only [Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
        decide_eq_true_eq]]
    by_cases hk : s1.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
    · rw [if_pos (by simp only [decide_eq_true_eq]; exact hk), if_pos hk]
      refine congrArg some ?_
      simp only [qTileAFT2G, BlockState.readMem, BlockState.setReg_mem, castTile_self,
        Tile.ptrAdd_data, Tile.scalar, Tile.bop_data,
        Tile.expandDim_data, Tile.vec_data, Broadcast.leftIndex_scalarL,
        Broadcast.rightIndex_scalarL, Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
        Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
        Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
        Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, TileShape.dropInsertedIndex,
        NumericDType.add, NumericDType.mul, NumericDType.nat_add, NumericDType.nat_mul,
        Region.cast, baseOffsetAFT2G]
      ring_nf
    · rw [if_neg (by simp only [decide_eq_true_eq]; exact hk), if_neg hk]
      simp only [BlockState.setReg_undef, hundef]
      norm_num

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General preLoop execution** (compose head + tail). -/
theorem aftgPreLoopG_eval
    (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (AftgFoundation.aftgPreLoopG Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s = some s0
      ∧ s0.pids = s.pids ∧ s0.mem = s.mem ∧ (∀ rg o, s0.undef rg o = 0)
      ∧ s0.regs .nat [] "start_m" = some (Tile.scalar (s.pids 0))
      ∧ s0.regs .nat [] "off_hz" = some (Tile.scalar (s.pids 1))
      ∧ s0.regs .real [BLOCK_M] "m_i" = some ⟨fun _ : TileIndex [BLOCK_M] => (⊥ : WithBot ℝ)⟩
      ∧ s0.regs .real [BLOCK_M] "l_i" = some ⟨fun _ : TileIndex [BLOCK_M] => (some (1 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun _ : TileIndex [BLOCK_M, BLOCK_DMODEL] => (some (0 : ℝ) : WithBot ℝ)⟩
      ∧ s0.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => s.pids 0 * BLOCK_M + r.val))
      ∧ s0.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))
      ∧ s0.regs .nat [BLOCK_DMODEL] "offs_k" = some (Tile.vec (fun e : Fin BLOCK_DMODEL => e.val))
      ∧ s0.regs .real [] "q_scale" = some ⟨fun _ : TileIndex [] =>
          some (s.readMem QScale ((s.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M) + s.pids 0)))⟩
      ∧ s0.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some (kPtrsAFT2G s K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s0.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFT2G s KScale N_CTX BLOCK_N 0)
      ∧ s0.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some (vPtrsAFT2G s V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0)
      ∧ s0.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = some (oBlockPtrAFT2G s Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL)
      ∧ s0.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
          if s.pids 0 * BLOCK_M + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE
          then some (qTileAFT2G s Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩ := by
  rw [AftgFoundation.aftgPreLoopG_eq_head_tail]
  obtain ⟨s1, hHead, hpids1, hmem1, hundef1, hstartm1, hoffhz1, hqvk1, hqso1, hkso1,
    hoffsm1, hoffsn1, hoffsk1⟩ := aftgPreLoopHeadG_eval s stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL hundef
  rw [stepStmts.append_some hHead]
  have hstartm1' : s1.regs .nat [] "start_m" = some (Tile.scalar (s1.pids 0)) := by
    rw [hpids1]; exact hstartm1
  have hoffhz1' : s1.regs .nat [] "off_hz" = some (Tile.scalar (s1.pids 1)) := by
    rw [hpids1]; exact hoffhz1
  have hqvk1' : s1.regs .nat [] "qvk_offset" = some (Tile.scalar (baseOffsetAFT2G s1 stride_qz stride_qh H)) := by
    rw [show baseOffsetAFT2G s1 stride_qz stride_qh H = baseOffsetAFT2G s stride_qz stride_qh H from by
      simp only [baseOffsetAFT2G, hpids1]]; exact hqvk1
  have hqso1' : s1.regs .nat [] "q_scale_offset"
      = some (Tile.scalar (s1.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M))) := by
    rw [hpids1]; exact hqso1
  have hkso1' : s1.regs .nat [] "k_scale_offset"
      = some (Tile.scalar (s1.pids 1 * ((N_CTX + BLOCK_N - 1) / BLOCK_N))) := by
    rw [hpids1]; exact hkso1
  have hoffsm1' : s1.regs .nat [BLOCK_M] "offs_m"
      = some (Tile.vec (fun r : Fin BLOCK_M => s1.pids 0 * BLOCK_M + r.val)) := by
    rw [hpids1]; exact hoffsm1
  obtain ⟨s0, hTail, hpids0, hmem0, hundef0, hstartm0, hoffhz0, hmi0, hli0, hacc0,
    hoffsm0, hoffsn0, hoffsk0, hqscale0, hkp0, hksp0, hvp0, hop0, hq0⟩ :=
    aftgPreLoopTailG_eval s1 Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE hundef1
      hstartm1' hoffhz1' hqvk1' hqso1' hkso1' hoffsm1' hoffsn1 hoffsk1
  have hbase : baseOffsetAFT2G s1 stride_qz stride_qh H = baseOffsetAFT2G s stride_qz stride_qh H := by
    simp only [baseOffsetAFT2G, hpids1]
  have hkpEq : kPtrsAFT2G s1 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0 = kPtrsAFT2G s K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0 := by
    simp only [kPtrsAFT2G, hbase]
  have hkspEq : kScalePtrAFT2G s1 KScale N_CTX BLOCK_N 0 = kScalePtrAFT2G s KScale N_CTX BLOCK_N 0 := by
    simp only [kScalePtrAFT2G, hpids1]
  have hvpEq : vPtrsAFT2G s1 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0 = vPtrsAFT2G s V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL 0 := by
    simp only [vPtrsAFT2G, hbase]
  have hopEq : oBlockPtrAFT2G s1 Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL = oBlockPtrAFT2G s Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL := by
    simp only [oBlockPtrAFT2G, hbase, hpids1]
  refine ⟨s0, hTail, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpids0, hpids1]
  · rw [hmem0, hmem1]
  · exact hundef0
  · rw [hstartm0, hpids1]
  · rw [hoffhz0, hpids1]
  · exact hmi0
  · exact hli0
  · exact hacc0
  · rw [hoffsm0, hpids1]
  · exact hoffsn0
  · exact hoffsk0
  · rw [hqscale0]
    refine congrArg some ?_
    ext idx
    simp only [BlockState.readMem, hmem1, hpids1]
  · rw [hkp0, hkpEq]
  · rw [hksp0, hkspEq]
  · rw [hvp0, hvpEq]
  · rw [hop0, hopEq]
  · rw [hq0]
    refine congrArg some ?_
    ext idx
    simp only [hpids1, qTileAFT2G, BlockState.readMem, hmem1, hbase]

/-- **General loop invariant** for the AFT2 streaming loop (counter `i = c·BLOCK_N`).
Binds the running registers to the seed-1 ⊥-state over the first `i` keys, the
static index vectors, loaded `q`/`q_scale`, the three streamed pointers, and
preserves `undef`/`mem`. -/
noncomputable def aftgInvariantG
    (Q K V QScale KScale Out : RegionName) (s0 : BlockState)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ) (hBD : 0 < BLOCK_DMODEL)
    (i : Nat) (s : BlockState) : Prop :=
  let qStart := qStartAFT2G s0 BLOCK_M
  let qT := qTileAFT2mG s0 Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE
  let kT := kTileAFT2G s0 K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL
  let vT := vTileAFT2mG s0 V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE
  s.pids = s0.pids ∧ i % BLOCK_N = 0 ∧ i ≤ N_CTX ∧
  (s.regs .real [BLOCK_M] "m_i" = some ⟨fun r : TileIndex [BLOCK_M] =>
      aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩) ∧
  (s.regs .real [BLOCK_M] "l_i" = some ⟨fun r : TileIndex [BLOCK_M] =>
      ((aftgStateBot1G qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩).2.1 : ℝ)⟩) ∧
  (s.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      ((aftgStateBot1G qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩) ∧
  (s.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => qStart + r.val))) ∧
  (s.regs .nat [BLOCK_N] "offs_n" = some (Tile.vec (fun j : Fin BLOCK_N => j.val))) ∧
  (s.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
      if qStart + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE then
        some (qTileAFT2G s0 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩) ∧
  (s.regs .real [] "q_scale" = some (Tile.scalar
      (some (s0.readMem QScale (s0.pids 1 * ((N_CTX + BLOCK_M - 1) / BLOCK_M) + s0.pids 0))))) ∧
  (s.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some (kPtrsAFT2G s0 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL (i / BLOCK_N))) ∧
  (s.regs .ptr [] "K_scale_ptr" = some (kScalePtrAFT2G s0 KScale N_CTX BLOCK_N (i / BLOCK_N))) ∧
  (s.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some (vPtrsAFT2G s0 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL (i / BLOCK_N))) ∧
  (s.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = some (oBlockPtrAFT2G s0 Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL)) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General preLoop ⇒ invariant base case.** -/
theorem aftgPreLoopG_invariant
    (s : BlockState) (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (hN : N_CTX = BLOCK_N * numKVBlocks) (hBN : 0 < BLOCK_N) (hBD : 0 < BLOCK_DMODEL)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s0, stepStmts (AftgFoundation.aftgPreLoopG Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s = some s0
      ∧ aftgInvariantG Q K V QScale KScale Out s stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale hBD 0 s0 := by
  subst hN
  obtain ⟨s0, hstep, hpids, hmem, hundef0, hstartm, hoffhz, hmi, hli, hacc,
    hoffsm, hoffsn, hoffsk, hqscale, hkp, hksp, hvp, hop, hq⟩ :=
    aftgPreLoopG_eval s Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE hundef
  refine ⟨s0, hstep, ?_⟩
  simp only [aftgInvariantG, qStartAFT2G]
  refine ⟨hpids, by norm_num, by norm_num, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, fun rg o => hundef0 rg o, hmem⟩
  · rw [hmi]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    simp only [aftgRunningMaxG_zero]
  · rw [hli]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    simp only [aftgStateBot1G_zero]; rfl
  · rw [hacc]; refine congrArg some ?_; refine Tile.ext (fun idx => ?_)
    simp only [aftgStateBot1G_zero]; rfl
  · rw [hoffsm]
  · exact hoffsn
  · rw [hq]; rfl
  · rw [hqscale]; rfl
  · rw [hkp, Nat.zero_div]
  · rw [hksp, Nat.zero_div]
  · rw [hvp, Nat.zero_div]
  · rw [hop]

/-! ### General loop-body execution chain -/

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- General loop-body head execution (statements 0–10). -/
theorem aftgLoopBodyHeadG_steps
    (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) (hBN : 0 < BLOCK_N)
    (sin : BlockState) (SN : Nat)
    (offsm : Tile .nat [BLOCK_M]) (offsn : Tile .nat [BLOCK_N])
    (kptrs : Tile .ptr [BLOCK_DMODEL, BLOCK_N]) (ksptr : Tile .ptr [])
    (mtile : Tile .real [BLOCK_M]) (qtile : Tile .real [BLOCK_M, BLOCK_DMODEL]) (qsc : Tile .real [])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsm : sin.regs .nat [BLOCK_M] "offs_m" = some offsm)
    (hoffsn : sin.regs .nat [BLOCK_N] "offs_n" = some offsn)
    (hmi : sin.regs .real [BLOCK_M] "m_i" = some mtile)
    (hkp : sin.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some kptrs)
    (hksp : sin.regs .ptr [] "K_scale_ptr" = some ksptr)
    (hq : sin.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile)
    (hqsc : sin.regs .real [] "q_scale" = some qsc) :
    ∃ s1, stepStmts (AftgFoundation.aftgLoopBodyHeadG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) sin = some s1
      ∧ s1.pids = sin.pids ∧ s1.mem = sin.mem ∧ (∀ rg o, s1.undef rg o = sin.undef rg o)
      ∧ ∃ (kmaskT : Tile .bool [BLOCK_DMODEL, BLOCK_N]) (ktile : Tile .real [BLOCK_DMODEL, BLOCK_N])
          (kscT : Tile .real []) (qkdotT : Tile .real [BLOCK_M, BLOCK_N])
          (maskT : Tile .bool [BLOCK_M, BLOCK_N]) (qkSentT : Tile .real [BLOCK_M, BLOCK_N])
          (rmaxT mijT : Tile .real [BLOCK_M]) (qkShiftT pExpT pT : Tile .real [BLOCK_M, BLOCK_N]),
        (kmaskT = ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
            (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (N_CTX - SN))
              && (ComparableDType.nat.lt idx.1.val HEAD_ACTIVE)⟩)
        ∧ (ktile = ⟨fun i : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
            if kmaskT.data i then some (sin.readMem (kptrs.data i).1 (kptrs.data i).2)
            else some (sin.undef (kptrs.data i).1 (kptrs.data i).2)⟩)
        ∧ (kscT = ⟨fun _ : TileIndex [] =>
            some (sin.readMem (ksptr.data PUnit.unit).1 (ksptr.data PUnit.unit).2)⟩)
        ∧ (qkdotT = Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) kscT)
        ∧ (maskT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
              (SN + offsn.data (idx.2.1, PUnit.unit))⟩)
        ∧ (qkSentT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            if maskT.data idx then qkdotT.data idx
            else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩)
        ∧ (Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkSentT = some rmaxT)
        ∧ (mijT = Tile.select
            (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
            mtile rmaxT)
        ∧ (qkShiftT = Tile.bop NumericDType.real.sub
            (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))
        ∧ (pExpT = Tile.uop WithBot.realExp2 qkShiftT)
        ∧ (pT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            if maskT.data idx then pExpT.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩)
        ∧ s1.regs .real [BLOCK_M] "m_i" = some mtile
        ∧ s1.regs .real [BLOCK_M] "m_ij" = some mijT
        ∧ s1.regs .bool [BLOCK_M, BLOCK_N] "mask" = some maskT
        ∧ s1.regs .real [BLOCK_M, BLOCK_N] "p" = some pT
        ∧ s1.regs .real [BLOCK_M] "l_i" = sin.regs .real [BLOCK_M] "l_i"
        ∧ s1.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = sin.regs .real [BLOCK_M, BLOCK_DMODEL] "acc"
        ∧ s1.regs .real [BLOCK_N, BLOCK_DMODEL] "v" = sin.regs .real [BLOCK_N, BLOCK_DMODEL] "v"
        ∧ s1.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = sin.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs"
        ∧ s1.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some kptrs
        ∧ s1.regs .ptr [] "K_scale_ptr" = some ksptr
        ∧ s1.regs .nat [BLOCK_M] "offs_m" = some offsm
        ∧ s1.regs .nat [BLOCK_N] "offs_n" = some offsn
        ∧ s1.regs .nat [] "start_n" = some (Tile.scalar SN)
        ∧ s1.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile
        ∧ s1.regs .real [] "q_scale" = some qsc := by
  set kmaskT : Tile .bool [BLOCK_DMODEL, BLOCK_N] := ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
      (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (N_CTX - SN))
        && (ComparableDType.nat.lt idx.1.val HEAD_ACTIVE)⟩ with hkmaskT
  set ktile : Tile .real [BLOCK_DMODEL, BLOCK_N] := ⟨fun i : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
      if kmaskT.data i then some (sin.readMem (kptrs.data i).1 (kptrs.data i).2)
      else some (sin.undef (kptrs.data i).1 (kptrs.data i).2)⟩ with hktile
  set kscT : Tile .real [] := ⟨fun _ : TileIndex [] =>
      some (sin.readMem (ksptr.data PUnit.unit).1 (ksptr.data PUnit.unit).2)⟩ with hkscT
  set qkdotT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.mul Broadcast.scalarR
        ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) kscT with hqkdotT
  set maskT : Tile .bool [BLOCK_M, BLOCK_N] := ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
        (SN + offsn.data (idx.2.1, PUnit.unit))⟩ with hmaskT
  set qkSentT : Tile .real [BLOCK_M, BLOCK_N] := ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      if maskT.data idx then qkdotT.data idx
      else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩ with hqkSentT
  obtain ⟨rmaxT, hrm⟩ := aftg_reduceMaxDrop1_someG BLOCK_M BLOCK_N hBN qkSentT
  set mijT : Tile .real [BLOCK_M] := Tile.select
      (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
      mtile rmaxT with hmijT
  set qkShiftT : Tile .real [BLOCK_M, BLOCK_N] := Tile.bop NumericDType.real.sub
      (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT) with hqkShiftT
  set pExpT : Tile .real [BLOCK_M, BLOCK_N] := Tile.uop WithBot.realExp2 qkShiftT with hpExpT
  set pT : Tile .real [BLOCK_M, BLOCK_N] := ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      if maskT.data idx then pExpT.data idx else (some (0.0 : ℝ) : WithBot ℝ)⟩ with hpT
  unfold AftgFoundation.aftgLoopBodyHeadG AftgFoundation.aftgLoopBodyG
  simp only [List.take_succ_cons, List.take_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.ref .nat [] "start_n") sin = some (Tile.scalar SN) from by rw [evalOp_ref, hsn]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some kmaskT from by
      rw [aftg_kmask_evalG _ SN N_CTX HEAD_ACTIVE BLOCK_N BLOCK_DMODEL offsn
        (by simp [BlockState.setReg_ne_name, hoffsn])
        (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some ktile from by
      rw [aftg_load_k_eval _ BLOCK_DMODEL BLOCK_N "K_ptrs" "k_mask" kptrs kmaskT
        (by simp [BlockState.setReg_ne_name, hkp]) (by rw [BlockState.setReg_same])]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some kscT from by
      rw [aftg_load_kscale_eval _ "K_scale_ptr" ksptr
        (by simp [BlockState.setReg_ne_name, hksp])]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some qkdotT from by
      rw [aftg_qk_dot_eval _ BLOCK_M BLOCK_N BLOCK_DMODEL qtile ktile qsc kscT
        (by simp [BlockState.setReg_ne_name, hq]) (by simp [BlockState.setReg_ne_name])
        (by simp [BlockState.setReg_ne_name, hqsc]) (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some maskT from by
      rw [aftg_mask_evalG _ SN BLOCK_M BLOCK_N offsm offsn
        (by simp [BlockState.setReg_ne_name, hoffsm]) (by simp [BlockState.setReg_ne_name, hoffsn])
        (by simp [BlockState.setReg_ne_name, hsn])]))]
  have hbcast6 : ∀ t : BlockState, @evalOp TileDType.real [BLOCK_M, BLOCK_N]
      (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [BLOCK_M, BLOCK_N]) t
      = some (⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] =>
          WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩ : Tile .real [BLOCK_M, BLOCK_N]) := by
    intro t
    simp only [evalOp, evalOp_sub, evalOp_const, Option.bind_eq_bind, Option.bind_some]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.where (Op.ref .bool [BLOCK_M, BLOCK_N] "mask")
        (Op.ref .real [BLOCK_M, BLOCK_N] "qk")
        (Op.broadcast (Op.sub .real Broadcast.nil (Op.const 0.0) (Op.const 1000000.0)) [BLOCK_M, BLOCK_N])) _
        = some qkSentT from by
      rw [evalOp_where]
      simp only [evalOp_ref, BlockState.setReg_same,
        BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
        hbcast6, Option.bind_eq_bind, Option.bind_some]
      refine congrArg some ?_
      ext idx
      simp only [hqkSentT, Tile.select_data, Tile.scalar]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some mijT from by
      rw [aftg_mij_evalG _ BLOCK_M BLOCK_N mtile qkSentT rmaxT
        (by simp [BlockState.setReg_ne_name, hmi]) (by rw [BlockState.setReg_same]) hrm]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some qkShiftT from by
      rw [aftg_qk_sub_evalG _ BLOCK_M BLOCK_N (by simp) qkSentT mijT
        (by simp [BlockState.setReg_ne_name]) (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some pExpT from by
      rw [aftg_p_evalG _ BLOCK_M BLOCK_N qkShiftT (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some pT from by
      rw [aftg_p_mask_evalG _ BLOCK_M BLOCK_N maskT pExpT
        (by simp [BlockState.setReg_ne_name]) (by rw [BlockState.setReg_same])]))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, kmaskT, ktile, kscT, qkdotT, maskT, qkSentT, rmaxT, mijT,
    qkShiftT, pExpT, pT, rfl, rfl, rfl, rfl, rfl, rfl, hrm, rfl, rfl, rfl, rfl,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef]
  · simp [BlockState.setReg_ne_name, hmi]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name]
  · simp [BlockState.setReg_ne_name, hkp]
  · simp [BlockState.setReg_ne_name, hksp]
  · simp [BlockState.setReg_ne_name, hoffsm]
  · simp [BlockState.setReg_ne_name, hoffsn]
  · simp [BlockState.setReg_ne_name, hsn]
  · simp [BlockState.setReg_ne_name, hq]
  · simp [BlockState.setReg_ne_name, hqsc]

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- General loop-body tail execution (statements 11–21). -/
theorem aftgLoopBodyTailG_steps
    (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat)
    (s1 : BlockState) (SN : Nat)
    (offsn : Tile .nat [BLOCK_N])
    (kptrs : Tile .ptr [BLOCK_DMODEL, BLOCK_N]) (ksptr : Tile .ptr []) (vptrs : Tile .ptr [BLOCK_N, BLOCK_DMODEL])
    (mtile mijT : Tile .real [BLOCK_M]) (maskT : Tile .bool [BLOCK_M, BLOCK_N])
    (pT : Tile .real [BLOCK_M, BLOCK_N]) (litile : Tile .real [BLOCK_M]) (acctile : Tile .real [BLOCK_M, BLOCK_DMODEL])
    (hsn : s1.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsn : s1.regs .nat [BLOCK_N] "offs_n" = some offsn)
    (hmi : s1.regs .real [BLOCK_M] "m_i" = some mtile)
    (hmij : s1.regs .real [BLOCK_M] "m_ij" = some mijT)
    (hmask : s1.regs .bool [BLOCK_M, BLOCK_N] "mask" = some maskT)
    (hp : s1.regs .real [BLOCK_M, BLOCK_N] "p" = some pT)
    (hli : s1.regs .real [BLOCK_M] "l_i" = some litile)
    (hacc : s1.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acctile)
    (hvp : s1.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some vptrs)
    (hkp : s1.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some kptrs)
    (hksp : s1.regs .ptr [] "K_scale_ptr" = some ksptr) :
    ∃ sF, stepStmts (AftgFoundation.aftgLoopBodyTailG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s1 = some sF
      ∧ sF.pids = s1.pids ∧ sF.mem = s1.mem ∧ (∀ rg o, sF.undef rg o = s1.undef rg o)
      ∧ sF.regs .nat [BLOCK_M] "offs_m" = s1.regs .nat [BLOCK_M] "offs_m"
      ∧ sF.regs .nat [BLOCK_N] "offs_n" = s1.regs .nat [BLOCK_N] "offs_n"
      ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = s1.regs .real [BLOCK_M, BLOCK_DMODEL] "q"
      ∧ sF.regs .real [] "q_scale" = s1.regs .real [] "q_scale"
      ∧ ∃ (lijT alphaT : Tile .real [BLOCK_M]) (vmaskT : Tile .bool [BLOCK_N, BLOCK_DMODEL])
          (vtile : Tile .real [BLOCK_N, BLOCK_DMODEL]) (pf16 : Tile .fp16 [BLOCK_M, BLOCK_N]),
        (lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pT)
        ∧ (alphaT = Tile.uop WithBot.realExp2
            (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
        ∧ (vmaskT = ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            (ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (N_CTX - SN))
              && (ComparableDType.nat.lt idx.2.1.val HEAD_ACTIVE)⟩)
        ∧ (vtile = ⟨fun i : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            if vmaskT.data i then some (s1.readMem (vptrs.data i).1 (vptrs.data i).2)
            else some (s1.undef (vptrs.data i).1 (vptrs.data i).2)⟩)
        ∧ (pf16 = ⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩)
        ∧ sF.regs .real [BLOCK_M] "m_i" = some mijT
        ∧ sF.regs .real [BLOCK_M] "l_i" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame Broadcast.nil)
            (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT)
        ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
            (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ vtile))
        ∧ sF.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR kptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM)))
        ∧ sF.regs .ptr [] "K_scale_ptr" = some
            (Tile.ptrAdd Broadcast.nil ksptr (Tile.scalar 1))
        ∧ sF.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR vptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM))) := by
  set lijT : Tile .real [BLOCK_M] := Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pT with hlijT
  set alphaT : Tile .real [BLOCK_M] := Tile.uop WithBot.realExp2
      (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT) with halphaT
  set vmaskT : Tile .bool [BLOCK_N, BLOCK_DMODEL] := ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
      (ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (N_CTX - SN))
        && (ComparableDType.nat.lt idx.2.1.val HEAD_ACTIVE)⟩ with hvmaskT
  set vtile : Tile .real [BLOCK_N, BLOCK_DMODEL] := ⟨fun i : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
      if vmaskT.data i then some (s1.readMem (vptrs.data i).1 (vptrs.data i).2)
      else some (s1.undef (vptrs.data i).1 (vptrs.data i).2)⟩ with hvtile
  set pf16 : Tile .fp16 [BLOCK_M, BLOCK_N] := ⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩ with hpf16
  unfold AftgFoundation.aftgLoopBodyTailG AftgFoundation.aftgLoopBodyG
  simp only [List.drop_succ_cons, List.drop_zero]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some lijT from aftg_lij_evalG _ BLOCK_M BLOCK_N pT hp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some alphaT from by
      rw [aftg_alpha_evalG _ BLOCK_M mtile mijT
        (by simp [BlockState.setReg_ne_name, hmi]) (by simp [BlockState.setReg_ne_name, hmij])]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_li_evalG _ BLOCK_M litile alphaT lijT
        (by simp [BlockState.setReg_ne_name, hli]) (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_acc_rescale_evalG _ BLOCK_M BLOCK_DMODEL (by simp) acctile alphaT
        (by simp [BlockState.setReg_ne_name, hacc]) (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some vtile from by
      rw [aftg_evalOp_load_ptr_mask_of (Op.ref .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs") _ _ vptrs vmaskT
        (by rw [evalOp_ref]; simp [BlockState.setReg_ne_name, hvp])
        (show evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_N] "offs_n"))
              (Op.sub .nat Broadcast.nil (Op.constNat N_CTX) (Op.ref .nat [] "start_n")))
            (Op.expandDim ⟨0, by simp⟩
              (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) _
            = some vmaskT from by
          rw [aftg_evalOp_boolAnd, evalOp_lt]
          erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _
              (regs_setReg_chain (by decide) (regs_setReg_chain (by decide)
                (regs_setReg_chain (by decide) (regs_setReg_chain (by decide) hoffsn)))),
            evalOp_expandDim]
          simp only [evalOp_lt, evalOp_sub, evalOp_arange, evalOp_constNat, evalOp_ref,
            BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true,
            BlockState.setReg_same, hsn, Option.bind_some, Option.bind_eq_bind]
          refine congrArg some ?_
          ext idx
          simp only [hvmaskT, Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec,
            Tile.scalar, Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
            Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
            Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
            Broadcast.leftIndex_nil, Broadcast.rightIndex_nil,
            TileShape.dropInsertedIndex, NumericDType.sub])]
      rfl))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some pf16 from
      aftg_p_fp16_evalG _ BLOCK_M BLOCK_N pT (by simp [BlockState.setReg_ne_name, hp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_acc_eval _ BLOCK_M BLOCK_N BLOCK_DMODEL
        (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
        pf16 vtile
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
        (by rw [BlockState.setReg_same])
        (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some mijT from
      aftg_mi_carry_evalG _ BLOCK_M mijT (by simp [BlockState.setReg_ne_name, hmij])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_advance_ptr_evalG _ BLOCK_DMODEL BLOCK_N BLOCK_N HEAD_DIM "K_ptrs" kptrs
        (by simp [BlockState.setReg_ne_name, hkp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_advance_kscale_eval _ "K_scale_ptr" ksptr
        (by simp [BlockState.setReg_ne_name, hksp])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp _ _ = some _ from
      aftg_advance_ptr_evalG _ BLOCK_N BLOCK_DMODEL BLOCK_N HEAD_DIM "V_ptrs" vptrs
        (by simp [BlockState.setReg_ne_name, hvp])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, lijT, alphaT, vmaskT, vtile, pf16,
    rfl, rfl, rfl, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_pids]
  · funext rg o; simp [BlockState.setReg_mem]
  · intro rg o; simp [BlockState.setReg_undef]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]


set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- General loop-body execution chain (22 statements). -/
theorem aftgLoopBody_stepsG
    (N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat) (hBN : 0 < BLOCK_N)
    (sin : BlockState) (SN : Nat)
    (offsm : Tile .nat [BLOCK_M]) (offsn : Tile .nat [BLOCK_N])
    (kptrs : Tile .ptr [BLOCK_DMODEL, BLOCK_N]) (ksptr : Tile .ptr []) (vptrs : Tile .ptr [BLOCK_N, BLOCK_DMODEL])
    (mtile : Tile .real [BLOCK_M]) (qtile : Tile .real [BLOCK_M, BLOCK_DMODEL]) (qsc : Tile .real [])
    (litile : Tile .real [BLOCK_M]) (acctile : Tile .real [BLOCK_M, BLOCK_DMODEL])
    (hsn : sin.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffsm : sin.regs .nat [BLOCK_M] "offs_m" = some offsm)
    (hoffsn : sin.regs .nat [BLOCK_N] "offs_n" = some offsn)
    (hmi : sin.regs .real [BLOCK_M] "m_i" = some mtile)
    (hli : sin.regs .real [BLOCK_M] "l_i" = some litile)
    (hacc : sin.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acctile)
    (hq : sin.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile)
    (hqsc : sin.regs .real [] "q_scale" = some qsc)
    (hkp : sin.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some kptrs)
    (hksp : sin.regs .ptr [] "K_scale_ptr" = some ksptr)
    (hvp : sin.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some vptrs) :
    ∃ sF, stepStmts (AftgFoundation.aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) sin = some sF
      ∧ sF.pids = sin.pids ∧ sF.mem = sin.mem ∧ (∀ rg o, sF.undef rg o = sin.undef rg o)
      ∧ ∃ (kmaskT : Tile .bool [BLOCK_DMODEL, BLOCK_N]) (ktile : Tile .real [BLOCK_DMODEL, BLOCK_N])
          (kscT : Tile .real []) (qkdotT : Tile .real [BLOCK_M, BLOCK_N])
          (maskT : Tile .bool [BLOCK_M, BLOCK_N]) (qkSentT : Tile .real [BLOCK_M, BLOCK_N])
          (rmaxT mijT : Tile .real [BLOCK_M]) (pT : Tile .real [BLOCK_M, BLOCK_N])
          (lijT alphaT : Tile .real [BLOCK_M]) (vmaskT : Tile .bool [BLOCK_N, BLOCK_DMODEL])
          (vtile : Tile .real [BLOCK_N, BLOCK_DMODEL]) (pf16 : Tile .fp16 [BLOCK_M, BLOCK_N]),
        (kmaskT = ⟨fun idx : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
            (ComparableDType.nat.lt (offsn.data (idx.2.1, PUnit.unit)) (N_CTX - SN))
              && (ComparableDType.nat.lt idx.1.val HEAD_ACTIVE)⟩)
        ∧ (ktile = ⟨fun i : TileIndex [BLOCK_DMODEL, BLOCK_N] =>
            if kmaskT.data i then some (sin.readMem (kptrs.data i).1 (kptrs.data i).2)
            else some (sin.undef (kptrs.data i).1 (kptrs.data i).2)⟩)
        ∧ (kscT = ⟨fun _ : TileIndex [] =>
            some (sin.readMem (ksptr.data PUnit.unit).1 (ksptr.data PUnit.unit).2)⟩)
        ∧ (qkdotT = Tile.bop NumericDType.real.mul Broadcast.scalarR
            (Tile.bop NumericDType.real.mul Broadcast.scalarR
              ⟨fun i => (Tile.dot [] qtile ktile).data i⟩ qsc) kscT)
        ∧ (maskT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            ComparableDType.nat.ge (offsm.data (idx.1, PUnit.unit))
              (SN + offsn.data (idx.2.1, PUnit.unit))⟩)
        ∧ (qkSentT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            if maskT.data idx then qkdotT.data idx
            else WithBot.realSub (some (0.0 : ℝ)) (some (1000000.0 : ℝ))⟩)
        ∧ (Tile.reduceMaxDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) qkSentT = some rmaxT)
        ∧ (mijT = Tile.select
            (Tile.cop ComparableDType.real.gt (Broadcast.consSame Broadcast.nil) mtile rmaxT)
            mtile rmaxT)
        ∧ (pT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
            if maskT.data idx then
              (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
                (Broadcast.consSame (Broadcast.consR Broadcast.nil))
                qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data idx
            else (some (0.0 : ℝ) : WithBot ℝ)⟩)
        ∧ (lijT = Tile.reduceSumDrop (⟨1, by simp⟩ : Fin [BLOCK_M, BLOCK_N].length) pT)
        ∧ (alphaT = Tile.uop WithBot.realExp2
            (Tile.bop NumericDType.real.sub (Broadcast.consSame Broadcast.nil) mtile mijT))
        ∧ (vmaskT = ⟨fun idx : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            (ComparableDType.nat.lt (offsn.data (idx.1, PUnit.unit)) (N_CTX - SN))
              && (ComparableDType.nat.lt idx.2.1.val HEAD_ACTIVE)⟩)
        ∧ (vtile = ⟨fun i : TileIndex [BLOCK_N, BLOCK_DMODEL] =>
            if vmaskT.data i then some (sin.readMem (vptrs.data i).1 (vptrs.data i).2)
            else some (sin.undef (vptrs.data i).1 (vptrs.data i).2)⟩)
        ∧ (pf16 = ⟨fun i => FloatDType.real.cast FloatDType.fp16 (pT.data i)⟩)
        ∧ sF.regs .real [BLOCK_M] "m_i" = some mijT
        ∧ sF.regs .real [BLOCK_M] "l_i" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame Broadcast.nil)
            (Tile.bop NumericDType.real.mul (Broadcast.consSame Broadcast.nil) litile alphaT) lijT)
        ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              acctile (Tile.expandDim ⟨1, by simp⟩ alphaT))
            (Tile.dot [] ⟨fun i => FloatDType.fp16.cast FloatDType.real (pf16.data i)⟩ vtile))
        ∧ sF.regs .nat [BLOCK_M] "offs_m" = some offsm
        ∧ sF.regs .nat [BLOCK_N] "offs_n" = some offsn
        ∧ sF.regs .real [BLOCK_M, BLOCK_DMODEL] "q" = some qtile
        ∧ sF.regs .real [] "q_scale" = some qsc
        ∧ sF.regs .ptr [BLOCK_DMODEL, BLOCK_N] "K_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR kptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM)))
        ∧ sF.regs .ptr [] "K_scale_ptr" = some
            (Tile.ptrAdd Broadcast.nil ksptr (Tile.scalar 1))
        ∧ sF.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some
            (Tile.ptrAdd Broadcast.scalarR vptrs
              (Tile.bop NumericDType.nat.mul Broadcast.nil (Tile.scalar BLOCK_N) (Tile.scalar HEAD_DIM))) := by
  rw [AftgFoundation.aftgLoopBodyG_eq_head_tail]
  obtain ⟨s1, hHead, hpids1, hmem1, hundef1, kmaskT, ktile, kscT, qkdotT, maskT, qkSentT,
    rmaxT, mijT, qkShiftT, pExpT, pT, hkmaskT, hktile, hkscT, hqkdotT, hmaskT, hqkSentT,
    hrm, hmijT, hqkShiftT, hpExpT, hpT, hs1mi, hs1mij, hs1mask, hs1p, hs1li, hs1acc,
    hs1v, hs1vp, hs1kp, hs1ksp, hs1offsm, hs1offsn, hs1sn, hs1q, hs1qsc⟩ :=
    aftgLoopBodyHeadG_steps N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE hBN sin SN offsm offsn kptrs ksptr mtile qtile qsc
      hsn hoffsm hoffsn hmi hkp hksp hq hqsc
  rw [stepStmts.append_some hHead]
  have hs1li' : s1.regs .real [BLOCK_M] "l_i" = some litile := by rw [hs1li]; exact hli
  have hs1acc' : s1.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some acctile := by rw [hs1acc]; exact hacc
  have hs1vp' : s1.regs .ptr [BLOCK_N, BLOCK_DMODEL] "V_ptrs" = some vptrs := by rw [hs1vp]; exact hvp
  obtain ⟨sF, hTail, hpidsF, hmemF, hundefF, hFoffsm, hFoffsn, hFq, hFqsc,
    lijT, alphaT, vmaskT, vtile, pf16,
    hlijT, halphaT, hvmaskT, hvtile, hpf16, hFmi, hFli, hFacc, hFkp, hFksp, hFvp⟩ :=
    aftgLoopBodyTailG_steps N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE s1 SN offsn kptrs ksptr vptrs mtile mijT maskT pT litile acctile
      hs1sn hs1offsn hs1mi hs1mij hs1mask hs1p hs1li' hs1acc' hs1vp' hs1kp hs1ksp
  have hpT' : pT = ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      if maskT.data idx then
        (Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub
          (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          qkSentT (Tile.expandDim ⟨1, by simp⟩ mijT))).data idx
      else (some (0.0 : ℝ) : WithBot ℝ)⟩ := by
    rw [hpT, hpExpT, hqkShiftT]
  have hsinmem : s1.mem = sin.mem := hmem1
  have hsinundef : ∀ rg o, s1.undef rg o = sin.undef rg o := hundef1
  refine ⟨sF, hTail, ?_, ?_, ?_, kmaskT, ktile, kscT, qkdotT, maskT, qkSentT, rmaxT, mijT,
    pT, lijT, alphaT, vmaskT, vtile, pf16,
    hkmaskT, hktile, hkscT, hqkdotT, hmaskT, hqkSentT, hrm, hmijT, hpT', hlijT, halphaT,
    ?_, ?_, hpf16, hFmi, hFli, hFacc, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, hpids1]
  · rw [hmemF, hmem1]
  · intro rg o; rw [hundefF, hundef1]
  · rw [hvmaskT]
  · rw [hvtile]
    refine congrArg _ ?_
    ext i
    rw [show s1.readMem (vptrs.data i).1 (vptrs.data i).2
          = sin.readMem (vptrs.data i).1 (vptrs.data i).2 from by
      unfold BlockState.readMem; rw [hsinmem],
      hsinundef]
  · rw [hFoffsm, hs1offsm]
  · rw [hFoffsn, hs1offsn]
  · rw [hFq, hs1q]
  · rw [hFqsc, hs1qsc]
  · exact hFkp
  · exact hFksp
  · exact hFvp

/-- General loop body preserves `O_block_ptr`. -/
theorem aftgLoopBody_preserves_OblockPtrG
    {N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE : Nat} {s s' : BlockState}
    (h : stepStmts (AftgFoundation.aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) s = some s') :
    s'.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = s.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" := by
  refine stepStmts_regs_preserved (fun st s1 s2 hmem hstep => ?_) h
  fin_cases hmem <;> exact stepStmt_assign_regs_ne (by decide) hstep

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General step lemma (AFT2, causal, masked).** -/
theorem aftg_attn_stepG
    (Q K V QScale KScale Out : RegionName)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M) (hBD : 0 < BLOCK_DMODEL)
    (hN : N_CTX = BLOCK_N * numKVBlocks)
    (s0 : BlockState) (i : Nat) (s : BlockState) (hilt : i < N_CTX) (himod : i % BLOCK_N = 0)
    (hsb : aftgScoreBoundG
      (qTileAFT2mG s0 Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
      (kTileAFT2G s0 K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
      (vTileAFT2mG s0 V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
      (keyScaleAFT2G s0 QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) (qStartAFT2G s0 BLOCK_M))
    (hinv : aftgInvariantG Q K V QScale KScale Out s0 stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
      (keyScaleAFT2G s0 QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) hBD i s) :
    ∃ s', stepStmts (AftgFoundation.aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE) (s.setReg "start_n" .nat [] (Tile.scalar i)) = some s'
      ∧ aftgInvariantG Q K V QScale KScale Out s0 stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks
          (keyScaleAFT2G s0 QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) hBD (i + BLOCK_N) s' := by
  subst hN
  set keyScale := keyScaleAFT2G s0 QScale KScale (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N numKVBlocks with hkeyScale
  set qStart := qStartAFT2G s0 BLOCK_M with hqStart
  set qT := qTileAFT2mG s0 Q stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_DMODEL HEAD_ACTIVE with hqT
  set kT := kTileAFT2G s0 K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL with hkT
  set vT := vTileAFT2mG s0 V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE with hvT
  simp only [aftgInvariantG] at hinv
  obtain ⟨hpids, hmod, hile, hmi, hli, hacc, hoffsm, hoffsn,
    hq, hqs, hKp, hKsp, hVp, hOp, hundef, hmem⟩ := hinv
  set c := i / BLOCK_N with hc_def
  have hi : i = c * BLOCK_N := by
    rw [hc_def]; exact (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero himod)).symm
  have hcnum : c < numKVBlocks := by
    have hlt2 : c * BLOCK_N < BLOCK_N * numKVBlocks := by rw [← hi]; exact hilt
    by_contra hge; push_neg at hge
    have : BLOCK_N * numKVBlocks ≤ BLOCK_N * c := Nat.mul_le_mul_left BLOCK_N hge
    rw [Nat.mul_comm BLOCK_N c] at this; omega
  have hc1 : (c + 1) * BLOCK_N ≤ BLOCK_N * numKVBlocks := by
    calc (c + 1) * BLOCK_N = BLOCK_N * (c + 1) := by ring
      _ ≤ BLOCK_N * numKVBlocks := Nat.mul_le_mul_left BLOCK_N (by omega)
  set sin := s.setReg "start_n" .nat [] (Tile.scalar i) with hsin
  have hpids' : sin.pids = s0.pids := by rw [hsin]; simpa using hpids
  have hmem' : sin.mem = s0.mem := by
    rw [hsin]; funext rg o; rw [BlockState.setReg_mem]; exact congrFun (congrFun hmem rg) o
  have hundef' : ∀ rg o, sin.undef rg o = 0 := by
    intro rg o; rw [hsin, BlockState.setReg_undef]; exact hundef rg o
  obtain ⟨sF, hchain, hpidsF, hmemF, hundefF,
      kmaskT, ktile, kscT, qkdotT, maskT, qkSentT, rmaxT, mijT, pT, lijT, alphaT,
      vmaskT, vtile, pf16,
      hkmaskT, hktile, hkscT, hqkdotT, hmaskT, hqkSentT, hrm, hmijT, hpT, hlijT, halphaT,
      hvmaskT, hvtile, hpf16, hFmi, hFli, hFacc, hFoffsm, hFoffsn, hFq, hFqsc, hFKp, hFKsp, hFVp⟩ :=
    aftgLoopBody_stepsG (BLOCK_N * numKVBlocks) HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE hBN
      sin i
      (Tile.vec (fun r : Fin BLOCK_M => qStart + r.val)) (Tile.vec (fun j : Fin BLOCK_N => j.val))
      (kPtrsAFT2G s0 K stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c)
      (kScalePtrAFT2G s0 KScale (BLOCK_N * numKVBlocks) BLOCK_N c)
      (vPtrsAFT2G s0 V stride_qz stride_qh H HEAD_DIM BLOCK_N BLOCK_DMODEL c)
      ⟨fun r : TileIndex [BLOCK_M] => aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if qStart + idx.1.val < BLOCK_N * numKVBlocks ∧ idx.2.1.val < HEAD_ACTIVE then
          some (qTileAFT2G s0 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩
      (Tile.scalar (some (s0.readMem QScale (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + s0.pids 0))))
      ⟨fun r : TileIndex [BLOCK_M] => ((aftgStateBot1G qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩).2.1 : ℝ)⟩
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => ((aftgStateBot1G qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
      (by rw [hsin]; simpa using rfl)
      (by rw [hsin]; simpa using hoffsm)
      (by rw [hsin]; simpa using hoffsn)
      (by rw [hsin]; simpa using hmi)
      (by rw [hsin]; simpa using hli)
      (by rw [hsin]; simpa using hacc)
      (by rw [hsin]; simpa using hq)
      (by rw [hsin]; simpa using hqs)
      (by rw [hsin, hc_def]; simpa using hKp)
      (by rw [hsin, hc_def]; simpa using hKsp)
      (by rw [hsin, hc_def]; simpa using hVp)
  refine ⟨sF, hchain, ?_⟩
  set qsc : ℝ := s0.readMem QScale (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_M - 1) / BLOCK_M) + s0.pids 0) with hqscv
  set ksc : ℝ := s0.readMem KScale (s0.pids 1 * ((BLOCK_N * numKVBlocks + BLOCK_N - 1) / BLOCK_N) + c) with hkscv
  have hkscData : kscT.data PUnit.unit = some ksc := by
    rw [hkscT]; simp only [kScalePtrAFT2G, Region.cast, hkscv]
    refine congrArg some ?_
    unfold BlockState.readMem; rw [hmem']
  have hkeyBlock : ∀ jL : Fin BLOCK_N,
      keyScale (⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩ : Fin (BLOCK_N * numKVBlocks)) = qsc * ksc := by
    intro jL
    have hdiv : (c * BLOCK_N + jL.val) / BLOCK_N = c := by
      rw [Nat.mul_comm c BLOCK_N, Nat.mul_add_div hBN, Nat.div_eq_of_lt jL.isLt, Nat.add_zero]
    simp only [hkeyScale, keyScaleAFT2G, hqscv, hkscv, hdiv]
  have hqcell : ∀ (ir : Fin BLOCK_M) (e : Fin BLOCK_DMODEL),
      (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if qStart + idx.1.val < BLOCK_N * numKVBlocks ∧ idx.2.1.val < HEAD_ACTIVE then
          some (qTileAFT2G s0 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩
          : Tile .real [BLOCK_M, BLOCK_DMODEL]).data (ir, e, PUnit.unit)
        = some (if e.val < HEAD_ACTIVE then qT (ir, e, PUnit.unit) else 0) := by
    intro ir e
    show (if qStart + ir.val < BLOCK_N * numKVBlocks ∧ e.val < HEAD_ACTIVE then some (qTileAFT2G s0 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL (ir, e, PUnit.unit)) else some (0.0 : ℝ))
        = some (if e.val < HEAD_ACTIVE then qT (ir, e, PUnit.unit) else 0)
    rw [hqT]; simp only [qTileAFT2mG, hqStart]
    by_cases he : e.val < HEAD_ACTIVE
    · by_cases hb : qStartAFT2G s0 BLOCK_M + ir.val < BLOCK_N * numKVBlocks
      · rw [if_pos ⟨hb, he⟩, if_pos he, if_pos ⟨hb, he⟩]
      · rw [if_neg (fun h => hb h.1), if_pos he, if_neg (fun h => hb h.1)]; norm_num
    · rw [if_neg (fun h => he h.2), if_neg he]; norm_num
  have hkcell : ∀ (e : Fin BLOCK_DMODEL) (jL : Fin BLOCK_N),
      ktile.data (e, jL, PUnit.unit)
        = some (if jL.val < BLOCK_N * numKVBlocks - i ∧ e.val < HEAD_ACTIVE then
            kT (⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩, e, PUnit.unit) else 0) := by
    intro e jL
    rw [hktile]; simp only [hkmaskT, kPtrsAFT2G, Region.cast, Tile.vec]
    rw [show (ComparableDType.nat.lt jL.val (BLOCK_N * numKVBlocks - i) && ComparableDType.nat.lt e.val HEAD_ACTIVE)
          = decide (jL.val < BLOCK_N * numKVBlocks - i ∧ e.val < HEAD_ACTIVE) from by
      rw [Bool.eq_iff_iff]; simp only [Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
        decide_eq_true_eq]]
    by_cases hb : jL.val < BLOCK_N * numKVBlocks - i ∧ e.val < HEAD_ACTIVE
    · rw [if_pos (by simp only [decide_eq_true_eq]; exact hb), if_pos hb]
      refine congrArg some ?_
      rw [hkT, kTileAFT2G]
      rw [show baseOffsetAFT2G s0 stride_qz stride_qh H + e.val + (c * BLOCK_N + jL.val) * HEAD_DIM
            = baseOffsetAFT2G s0 stride_qz stride_qh H + (c * BLOCK_N + jL.val) * HEAD_DIM + e.val from by omega]
      show sin.readMem _ _ = s0.readMem K _
      unfold BlockState.readMem; rw [hmem']
    · rw [if_neg (by simp only [decide_eq_true_eq]; exact hb), if_neg hb]
      refine congrArg some ?_
      exact hundef' _ _
  have hqTmask : ∀ (ir : Fin BLOCK_M) (e : Fin BLOCK_DMODEL), (if e.val < HEAD_ACTIVE then qT (ir, e, PUnit.unit) else 0) = qT (ir, e, PUnit.unit) := by
    intro ir e
    by_cases he : e.val < HEAD_ACTIVE
    · rw [if_pos he]
    · rw [if_neg he, hqT]; simp only [qTileAFT2mG]; rw [if_neg (fun h => he h.2)]
  have hjLwin : ∀ jL : Fin BLOCK_N, jL.val < BLOCK_N * numKVBlocks - i := by
    intro jL; have := jL.isLt
    have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N
    omega
  have hqkcell : ∀ (ir : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (jL : Fin BLOCK_N),
      qkdotT.data (ir, jL, PUnit.unit)
        = some ((aftgKVG qT kT vT keyScale ir d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1) := by
    intro ir d jL
    rw [hqkdotT]
    have hsc := aftg_score_cellG BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE qStart (BLOCK_N * numKVBlocks - i) jL ir qsc ksc qT
      (fun e jL => kT (⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩, e, PUnit.unit))
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        if qStart + idx.1.val < BLOCK_N * numKVBlocks ∧ idx.2.1.val < HEAD_ACTIVE then
          some (qTileAFT2G s0 Q stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL idx) else some (0.0 : ℝ)⟩
      ktile kscT (hjLwin jL)
      (fun e => hqcell ir e) (fun e => hkcell e jL) hkscData
    rw [hsc]
    refine congrArg some ?_
    simp only [aftgKVG, hkeyBlock jL]
    rw [show (Finset.univ.sum (fun e : Fin BLOCK_DMODEL => (if e.val < HEAD_ACTIVE then qT (ir, e, PUnit.unit) else 0)
            * kT (⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩, e, PUnit.unit)))
          = Finset.univ.sum (fun e : Fin BLOCK_DMODEL => qT (ir, e, PUnit.unit)
            * kT (⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩, e, PUnit.unit))
        from Finset.sum_congr rfl (fun e _ => by rw [hqTmask ir e])]
    ring
  have hmaskcell : ∀ (ir : Fin BLOCK_M) (jL : Fin BLOCK_N),
      maskT.data (ir, jL, PUnit.unit) = decide (qStart + ir.val ≥ c * BLOCK_N + jL.val) := by
    intro ir jL
    rw [hmaskT]; simp only [Tile.vec]
    rw [show (ComparableDType.nat.ge (qStart + ir.val) (i + jL.val))
          = decide (qStart + ir.val ≥ c * BLOCK_N + jL.val) from by
      rw [Bool.eq_iff_iff]; simp only [ComparableDType.nat_ge_eq_true, decide_eq_true_eq]
      omega]
  have hsentcell : ∀ (ir : Fin BLOCK_M) (d : Fin BLOCK_DMODEL) (jL : Fin BLOCK_N),
      qkSentT.data (ir, jL, PUnit.unit)
        = if qStart + ir.val ≥ c * BLOCK_N + jL.val then
            (((aftgKVG qT kT vT keyScale ir d ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).1 : ℝ) : WithBot ℝ)
          else ((-1000000.0 : ℝ) : WithBot ℝ) := by
    intro ir d jL
    rw [hqkSentT]; simp only [hmaskcell ir jL, decide_eq_true_eq]
    by_cases hk : qStart + ir.val ≥ c * BLOCK_N + jL.val
    · rw [if_pos hk, if_pos hk]
      exact hqkcell ir d jL
    · rw [if_neg hk, if_neg hk]
      exact aftg_sentinel_eq
  simp only [aftgInvariantG, ← hqStart, ← hqT, ← hkT, ← hvT]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hpidsF, hpids']
  · rw [Nat.add_mod_right]; exact himod
  · have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N
    rw [hi]; omega
  · rw [hFmi]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    obtain ⟨ir, ⟨⟩⟩ := r
    have hbr := aftg_mij_reg_eq_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 ir hsb
      qkSentT ⟨fun r : TileIndex [BLOCK_M] =>
        aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩ rmaxT mijT
      (fun jL => by
        rw [show (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (ir, PUnit.unit) jL)
              = (ir, jL, PUnit.unit) from rfl]
        rw [hsentcell ir ⟨0, hBD⟩ jL])
      hrm (by simp only [hi]) hmijT
    rw [hbr, show ((c + 1) * BLOCK_N : Nat) = i + BLOCK_N from by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega]
  · rw [hFli]; refine congrArg some ?_; refine Tile.ext (fun r => ?_)
    obtain ⟨ir, ⟨⟩⟩ := r
    have hmijcell : mijT.data (ir, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir ⟨0, hBD⟩ := by
      refine aftg_mij_reg_eq_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 ir hsb
        qkSentT ⟨fun r : TileIndex [BLOCK_M] =>
          aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩ rmaxT mijT
        (fun jL => by
          rw [show (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (ir, PUnit.unit) jL)
                = (ir, jL, PUnit.unit) from rfl]
          rw [hsentcell ir ⟨0, hBD⟩ jL])
        hrm (by simp only [hi]) hmijT
    have hbr := aftg_denom_reg_eq_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 ir hsb
      qkSentT ⟨fun r : TileIndex [BLOCK_M] => aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩
      mijT alphaT ⟨fun r : TileIndex [BLOCK_M] => ((aftgStateBot1G qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩).2.1 : ℝ)⟩
      lijT pT
      (fun jL => by rw [hsentcell ir ⟨0, hBD⟩ jL])
      (by simp only [hi]; rfl) (by simp only [hi]) hmijcell
      halphaT hlijT
      (fun jL => by
        rw [hpT]; simp only [hmaskcell ir jL, decide_eq_true_eq])
    have hne : aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir ⟨0, hBD⟩ ≠ ⊥ :=
      aftgRunningMaxG_ne_bot qT kT vT keyScale qStart ((c + 1) * BLOCK_N) (by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega) (by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega) ir ⟨0, hBD⟩
    show (Tile.bop NumericDType.real.add _ _ _).data _ = _
    rw [hbr, show ((i + BLOCK_N) : Nat) = (c + 1) * BLOCK_N from by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega]
    exact congrArg (fun st : WithBot ℝ × ℝ × ℝ => (st.2.1 : WithBot ℝ))
      (aftgStateBot1G_eq_aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir ⟨0, hBD⟩ hne).symm
  · rw [hFacc, hpf16]
    simp only [FloatDType.cast, FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot,
      FloatDType.fp16_toWithBot, FloatDType.real_ofWithBot]
    refine congrArg some ?_; refine Tile.ext (fun idx => ?_)
    obtain ⟨ir, id, ⟨⟩⟩ := idx
    have hmijcell : mijT.data (ir, PUnit.unit)
        = aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir ⟨0, hBD⟩ := by
      refine aftg_mij_reg_eq_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 ir hsb
        qkSentT ⟨fun r : TileIndex [BLOCK_M] =>
          aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩ rmaxT mijT
        (fun jL => by
          rw [show (TileShape.insertAxisIndex [BLOCK_M, BLOCK_N] (aftgAx1G BLOCK_M BLOCK_N) (ir, PUnit.unit) jL)
                = (ir, jL, PUnit.unit) from rfl]
          rw [hsentcell ir ⟨0, hBD⟩ jL])
        hrm (by simp only [hi]) hmijT
    have hvload : ∀ jL : Fin BLOCK_N,
        vtile.data (jL, id, PUnit.unit)
          = some ((aftgKVG qT kT vT keyScale ir id ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).2) := by
      intro jL
      rw [hvtile]; simp only [hvmaskT, Tile.vec, vPtrsAFT2G, Region.cast]
      rw [show (ComparableDType.nat.lt jL.val (BLOCK_N * numKVBlocks - i) && ComparableDType.nat.lt id.val HEAD_ACTIVE)
            = decide (jL.val < BLOCK_N * numKVBlocks - i ∧ id.val < HEAD_ACTIVE) from by
        rw [Bool.eq_iff_iff]; simp only [Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
          decide_eq_true_eq]]
      simp only [aftgKVG, hvT, vTileAFT2mG]
      by_cases hid : id.val < HEAD_ACTIVE
      · rw [if_pos (by simp only [decide_eq_true_eq]; exact ⟨hjLwin jL, hid⟩), if_pos hid]
        refine congrArg some ?_
        show sin.readMem _ (baseOffsetAFT2G s0 stride_qz stride_qh H + (c * BLOCK_N + jL.val) * HEAD_DIM + id.val) = _
        rw [vTileAFT2G]
        unfold BlockState.readMem; rw [hmem']
      · rw [if_neg (by simp only [decide_eq_true_eq, not_and]; intro _ h; exact hid h), if_neg hid]
        exact congrArg some (hundef' _ _)
    have hbr := aftg_acc_reg_eq_maskedG qT kT vT keyScale BLOCK_N hBN hBM hBD qStart c hc1 ir id hsb
      qkSentT ⟨fun r : TileIndex [BLOCK_M] => aftgRunningMaxG qT kT vT keyScale qStart i r.1 ⟨0, hBD⟩⟩
      mijT alphaT
      ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => ((aftgStateBot1G qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
      (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => ((aftgStateBot1G qT kT vT keyScale qStart i idx.1 idx.2.1).2.2 : ℝ)⟩
        (Tile.expandDim ⟨1, by simp⟩ alphaT))
      pT vtile (fun jL => (aftgKVG qT kT vT keyScale ir id ⟨c * BLOCK_N + jL.val, by have := jL.isLt; have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega⟩).2)
      (fun jL => by rw [hsentcell ir ⟨0, hBD⟩ jL])
      (by simp only [hi]; rfl) (by simp only [hi]) hmijcell halphaT rfl
      (fun jL => by
        rw [hpT]; simp only [hmaskcell ir jL, decide_eq_true_eq])
      hvload (fun jL => rfl)
    have hne : aftgRunningMaxG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir id ≠ ⊥ := by
      rw [aftgRunningMaxG_eq qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir id ⟨0, hBD⟩]
      exact aftgRunningMaxG_ne_bot qT kT vT keyScale qStart ((c + 1) * BLOCK_N) (by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega) (by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega) ir ⟨0, hBD⟩
    show (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) _
      (Tile.dot [] pT vtile)).data _ = _
    rw [hbr, show ((i + BLOCK_N) : Nat) = (c + 1) * BLOCK_N from by have h2 : (c+1)*BLOCK_N = c*BLOCK_N + BLOCK_N := Nat.succ_mul c BLOCK_N; omega]
    exact congrArg (fun st : WithBot ℝ × ℝ × ℝ => (st.2.2 : WithBot ℝ))
      (aftgStateBot1G_eq_aftgStateBotG qT kT vT keyScale qStart ((c + 1) * BLOCK_N) ir id hne).symm
  · rw [hFoffsm]
  · rw [hFoffsn]
  · rw [hFq]
  · rw [hFqsc]
  · rw [hFKp, kPtrsAFT2G_succ, show (i + BLOCK_N) / BLOCK_N = c + 1 from by rw [hi, Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]]
  · rw [hFKsp, kScalePtrAFT2G_succ, show (i + BLOCK_N) / BLOCK_N = c + 1 from by rw [hi, Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]]
  · rw [hFVp, vPtrsAFT2G_succ, show (i + BLOCK_N) / BLOCK_N = c + 1 from by rw [hi, Nat.add_div_right _ hBN, Nat.mul_div_cancel _ hBN]]
  · rw [aftgLoopBody_preserves_OblockPtrG hchain, hsin,
      BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]
    exact hOp
  · intro rg o; rw [hundefF, hundef']
  · rw [hmemF, hmem']

set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **General postLoop evaluation (AFT2, causal).** From a loop-end state satisfying
`aftgInvariantG … N_CTX`, the 2 postLoop statements write the genuine closed form
`attnFwdTritonOutSpecG` to `Out` at every active output lane and preserve `Out` on
inactive lanes. -/
theorem aftgPostLoopG_eval
    (Q K V QScale KScale Out : RegionName) (s0 : BlockState) (s : BlockState)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hN : N_CTX = BLOCK_N * numKVBlocks) (hBN : 0 < BLOCK_N) (hnum : 0 < numKVBlocks)
    (keyScale : Fin (BLOCK_N * numKVBlocks) → ℝ)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s0 H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx))
    (hinv : aftgInvariantG Q K V QScale KScale Out s0 stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale hBD N_CTX s) :
    ∃ sP, stepStmts (AftgFoundation.aftgPostLoopG Out N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE) s = some sP
      ∧ ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
          sP.readMem Out (outOffset s0 H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)
            = if active s0 N_CTX HEAD_ACTIVE BLOCK_M idx then
                attnFwdTritonOutSpecG s0 Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale idx
              else s.readMem Out (outOffset s0 H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx) := by
  have hSEQ : 0 < BLOCK_N * numKVBlocks := Nat.mul_pos hBN hnum
  have hNpos : 0 < N_CTX := by rw [hN]; exact hSEQ
  simp only [aftgInvariantG] at hinv
  obtain ⟨hpids, _, _, _hmi, hli, hacc, hoffsm, _hoffsn,
    _hq, _hqs, _hKp, _hKsp, _hVp, hOp, hundef, hmem⟩ := hinv
  set qStart := qStartAFT2G s0 BLOCK_M with hqStart
  set qT := qTileAFT2mG s0 Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE with hqT
  set kT := kTileAFT2G s0 K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL with hkT
  set vT := vTileAFT2mG s0 V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE with hvT
  set liTile : Tile .real [BLOCK_M] :=
    ⟨fun r : TileIndex [BLOCK_M] => ((aftgStateBot1G qT kT vT keyScale qStart N_CTX r.1 ⟨0, hBD⟩).2.1 : ℝ)⟩
    with hliTile
  set accTile : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => ((aftgStateBot1G qT kT vT keyScale qStart N_CTX idx.1 idx.2.1).2.2 : ℝ)⟩
    with haccTile
  set accFin : Tile .real [BLOCK_M, BLOCK_DMODEL] :=
    Tile.bop NumericDType.real.div (Broadcast.consSame (Broadcast.consR Broadcast.nil)) accTile
      (Tile.expandDim ⟨1, by simp⟩ liTile) with haccFin
  unfold AftgFoundation.aftgPostLoopG
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (show evalOp (Op.div .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BLOCK_M] "l_i"))) s
        = some accFin from by
      have hexp : @evalOp TileDType.real [BLOCK_M, 1]
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.real [BLOCK_M] "l_i")) s
          = some (Tile.expandDim ⟨1, by simp⟩ liTile) :=
        evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hli
      rw [evalOp_div]
      simp only [evalOp_ref, hexp, hacc, Option.bind_eq_bind, Option.bind_some]
      rfl))]
  set s2 := s.setReg "acc" .real [BLOCK_M, BLOCK_DMODEL] accFin with hs2
  have hOp2 : s2.regs .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr" = some (oBlockPtrAFT2G s0 Out stride_qz stride_qh H HEAD_DIM BLOCK_M BLOCK_DMODEL) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hOp
  have hacc2 : s2.regs .real [BLOCK_M, BLOCK_DMODEL] "acc" = some accFin := by
    rw [hs2, BlockState.setReg_same]
  have hoffsm2 : s2.regs .nat [BLOCK_M] "offs_m" = some (Tile.vec (fun r : Fin BLOCK_M => qStart + r.val)) := by
    rw [hs2, BlockState.setReg_ne_name _ _ _ _ _ _ _ _ (by decide)]; exact hoffsm
  set oOffFn : TileIndex [BLOCK_M, BLOCK_DMODEL] → Nat :=
    fun idx => baseOffsetAFT2G s0 stride_qz stride_qh H + (s0.pids 0 * BLOCK_M + idx.1.val) * HEAD_DIM + idx.2.1.val with hoOffFn
  set P : TileIndex [BLOCK_M, BLOCK_DMODEL] → Prop :=
    fun idx => qStart + idx.1.val < N_CTX ∧ idx.2.1.val < HEAD_ACTIVE with hP
  have hopEval : evalOp (Op.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr") s2
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out.cast, oOffFn idx)⟩ : Tile .ptr [BLOCK_M, BLOCK_DMODEL]) := by
    rw [evalOp_ref, hOp2]
    refine congrArg some ?_; ext idx
    · rfl
    · simp only [oBlockPtrAFT2G, hoOffFn]
  have hmaskEval : evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))) s2
      = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => decide (P idx)⟩ : Tile .bool [BLOCK_M, BLOCK_DMODEL]) := by
    rw [aftg_evalOp_boolAnd, evalOp_lt]
    erw [evalOp_expandDim_ref_of_regs _ _ _ _ _ _ hoffsm2, evalOp_expandDim]
    simp only [evalOp_lt, evalOp_arange, evalOp_constNat, Option.bind_some, Option.bind_eq_bind]
    refine congrArg some ?_; ext idx
    simp only [Tile.bop_data, Tile.cop_data, Tile.expandDim_data, Tile.vec_data, Tile.scalar,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
      Broadcast.leftIndex_consL, Broadcast.rightIndex_consL,
      Broadcast.leftIndex_consR, Broadcast.rightIndex_consR,
      Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, TileShape.dropInsertedIndex]
    rw [show (ComparableDType.nat.lt (qStart + idx.1.val) N_CTX && ComparableDType.nat.lt idx.2.1.val HEAD_ACTIVE)
          = decide (P idx) from by
      rw [Bool.eq_iff_iff]; simp only [hP, Bool.and_eq_true, ComparableDType.nat_lt_eq_true,
        decide_eq_true_eq]]
  have hstore : stepStmt (Stmt.store .real [BLOCK_M, BLOCK_DMODEL] (.ptr (.ref .ptr [BLOCK_M, BLOCK_DMODEL] "O_block_ptr"))
      (Op.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
      (.mask (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offs_m")) (Op.constNat N_CTX))
        (Op.expandDim ⟨0, by simp⟩
          (Op.lt ComparableDType.nat Broadcast.scalarR (Op.arange BLOCK_DMODEL) (Op.constNat HEAD_ACTIVE)))))) s2
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_DMODEL]).foldl
          (fun acc idx => if P idx then acc.writeMem Out (oOffFn idx) ((accFin.data idx).unbotD 0) else acc) s2) := by
    simp only [stepStmt, evalOp_ref, hacc2, hopEval, hmaskEval, Option.bind_eq_bind, Option.bind_some,
      Option.map_some, decide_eq_true_eq]
    refine congrArg some ?_
    refine List.foldl_ext _ _ s2 ?_
    intro acc idx _
    by_cases hk : P idx
    · simp only [if_pos hk, Region.cast_id, BlockState.writeMemTyped_real, FloatDType.real_storeValue]
    · simp only [if_neg hk]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  have hinjO : Function.Injective oOffFn := by
    have : oOffFn = (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => outOffset s0 H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx) := by
      funext idx
      simp only [hoOffFn, outOffset, offZ, offH, mIndex, kIndex, baseOffsetAFT2G, Nat.mul_one]
    rw [this]; exact hOutInj
  have houtOff : outOffset s0 H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx = oOffFn idx := by
    simp only [outOffset, offZ, offH, mIndex, kIndex, hoOffFn, baseOffsetAFT2G, Nat.mul_one]
  rw [houtOff]
  rw [BlockState.scatter_readback_prop_masked_nd _ oOffFn
    (fun idx => (accFin.data idx).unbotD 0) P hinjO idx]
  have hactiveP : active s0 N_CTX HEAD_ACTIVE BLOCK_M idx ↔ P idx := by
    simp only [active, mIndex, kIndex, hP, hqStart, qStartAFT2G]
  by_cases hk : P idx
  · rw [if_pos hk, if_pos (hactiveP.mpr hk)]
    obtain ⟨ir, id, ⟨⟩⟩ := idx
    subst hN
    have hne : aftgRunningMaxG qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) ir id ≠ ⊥ :=
      aftgRunningMaxG_ne_bot qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) hSEQ hSEQ ir id
    have hne' : aftgRunningMaxG qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) ir ⟨0, hBD⟩ ≠ ⊥ :=
      aftgRunningMaxG_ne_bot qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) hSEQ hSEQ ir ⟨0, hBD⟩
    simp only [haccFin, Tile.bop_data, Tile.expandDim_data, TileShape.dropInsertedIndex,
      Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.div, WithBot.realDiv,
      Option.map₂, Option.bind, Option.map, haccTile, hliTile, WithBot.unbotD_coe]
    rw [aftgStateBot1G_eq_aftgStateBotG qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) ir id hne,
      aftgStateBot1G_eq_aftgStateBotG qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) ir ⟨0, hBD⟩ hne']
    rw [aftgStateBotG_snd_fst_indep qT kT vT keyScale qStart (BLOCK_N * numKVBlocks) ir ⟨0, hBD⟩ id]
    have hbridge := aftgStateBotG_full_eq_spec s0 Q K V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale ir id
      (by rw [← hqStart, ← hqT, ← hkT, ← hvT]; exact hne)
    rw [← hbridge]
    simp only [hqStart, hqT, hkT, hvT, WithBot.unbotD_some]
  · rw [if_neg hk, if_neg (fun h => hk (hactiveP.mp h)), hs2]
    simp only [BlockState.readMem, BlockState.setReg_mem]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 8000 in
/-- **General full kernel execution (AFT2, causal).** The lowered general AFT2 surface
body (contiguous symbolic layout) steps a clean, score-bounded state through
preLoop + the `forRange` streaming loop (`forRange_inv` with `aftgInvariantG`,
advanced by `aftg_attn_stepG`) + postLoop, leaving the `Out` writeback at every
active lane equal to the genuine causal closed form `attnFwdTritonOutSpecG`. -/
theorem aftg_exec_generalG
    (Q K V QScale KScale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z numKVBlocks : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M)
    (hN : N_CTX = BLOCK_N * numKVBlocks) (hnum : 0 < numKVBlocks)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hsb : aftgScoreBoundG
      (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
      (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
      (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
      (keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) (qStartAFT2G s BLOCK_M)) :
    ∃ sF, stepStmts (attn_fwd_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgKernel.body s = some sF
      ∧ ∀ idx : TileIndex [BLOCK_M, BLOCK_DMODEL],
          active s N_CTX HEAD_ACTIVE BLOCK_M idx →
            sF.readMem Out (outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)
              = attnFwdTritonOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks (keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) idx := by
  set keyScale := keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks with hkeyScale
  rw [aftg_body_splitG, aftgPreLoopG_check, aftgPostLoopG_check]
  obtain ⟨sp, hpre, hinv0⟩ :=
    aftgPreLoopG_invariant s Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hN hBN hBD keyScale hundef
  rw [stepStmts.append_some hpre]
  obtain ⟨final, sL, hloop, hfin, hinvL⟩ :=
    forRange_inv (idx := "start_n") (start := 0) (stop := N_CTX) (step := BLOCK_N)
      (body := AftgFoundation.aftgLoopBodyG N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE)
      (P := fun i st => aftgInvariantG Q K V QScale KScale Out s stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks keyScale hBD i st)
      (s_init := sp)
      (by omega)
      hinv0
      (fun i st hi hP =>
        aftg_attn_stepG Q K V QScale KScale Out stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBN hBM hBD hN s i st hi
          (by simp only [aftgInvariantG] at hP; exact hP.2.1) hsb hP)
  rw [stepStmts.cons_some hloop]
  have hfinal : final = N_CTX := by
    obtain ⟨_, hmod, hle, _⟩ := hinvL
    omega
  rw [hfinal] at hinvL
  obtain ⟨sF, hpost, hO⟩ :=
    aftgPostLoopG_eval Q K V QScale KScale Out s sL stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks hBD hN hBN hnum keyScale hOutInj hinvL
  refine ⟨sF, hpost, ?_⟩
  intro idx hact
  rw [hO idx, if_pos hact]


/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/
set_option maxHeartbeats 1600000 in
set_option maxRecDepth 8000 in
/-- **Dimension-general genuine causal closed-form output summary for `attn_fwd_triton`.**

Removes the Python test-shape pin (`Z = 2`, `H = 4`,
`N_CTX = HEAD_DIM = BLOCK_M = BLOCK_DMODEL = 128`, `BLOCK_N = 64`,
`HEAD_ACTIVE = 96`): for arbitrary symbolic block/head dimensions at the
contiguous layout (`stride_qm = HEAD_DIM`, `stride_qk = 1`, `stride_kn = HEAD_DIM`),
the full causal surface lowers to the algorithm layer and its masked `Out`
writeback realizes the genuine closed-form causal attention
`attnFwdTritonOutSpecG` (= `attentionRealBase2PerKeyScalePred … (causalKeep)` over
INPUT memory) at every active output lane. Side conditions: positive blocks,
`N_CTX = BLOCK_N · numKVBlocks` (`numKVBlocks > 0`), clean input (`undef = 0`), the
`-1e6` sentinel score bound `aftgScoreBoundG`, and output-offset injectivity. -/
specification attn_fwd_triton_output_summary_general
    (Q K V QScale KScale Out : RegionName) (s : BlockState)
    (stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z numKVBlocks : Nat)
    (hBD : 0 < BLOCK_DMODEL) (hBN : 0 < BLOCK_N) (hBM : 0 < BLOCK_M)
    (hN : N_CTX = BLOCK_N * numKVBlocks) (hnum : 0 < numKVBlocks)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (hsb : aftgScoreBoundG
      (qTileAFT2mG s Q stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_DMODEL HEAD_ACTIVE)
      (kTileAFT2G s K stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL)
      (vTileAFT2mG s V stride_qz stride_qh H HEAD_DIM (BLOCK_N * numKVBlocks) BLOCK_DMODEL HEAD_ACTIVE)
      (keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) (qStartAFT2G s BLOCK_M)) :
    (∃ alg, (attn_fwd_triton_surface Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
      (kernel := attn_fwd_triton_surface Q K V QScale KScale Out
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
        Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => active s N_CTX HEAD_ACTIVE BLOCK_M idx)
        (fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] => (Out,
          outOffset s H stride_qz stride_qh HEAD_DIM 1 BLOCK_M idx)))
      (expected := fun idx : TileIndex [BLOCK_M, BLOCK_DMODEL] =>
        attnFwdTritonOutSpecG s Q K V stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE numKVBlocks (keyScaleAFT2G s QScale KScale N_CTX BLOCK_M BLOCK_N numKVBlocks) idx) := by
  refine ⟨?_, ?_⟩
  · exact attn_fwd_triton_surface_toAlgorithm_supported Q K V QScale KScale Out
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      stride_qz stride_qh HEAD_DIM 1 stride_qz stride_qh HEAD_DIM 1
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
  · rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [attn_fwd_triton_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0
    subst s0
    intro idx hActive
    obtain ⟨sF, hstep, hO⟩ := aftg_exec_generalG Q K V QScale KScale Out s
      stride_qz stride_qh H HEAD_DIM N_CTX BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE Z numKVBlocks
      hBD hBN hBM hN hnum hOutInj hundef hsb
    rw [exec] at hExec
    rw [hstep] at hExec
    obtain rfl : sF = s' := Option.some.inj hExec
    exact hO idx hActive

end General


end VeriTile.Bench.TritonBenchG.AttnFwdTriton
