# T3 Scouting — FlashAttention forward feasibility study

**Status:** Phase A deliverable. Date: 2026-04-26.
**Author:** Subagent on behalf of Phase A.

This document evaluates whether the Phase D headline (T3-B: `fa1_eq_fa2`)
is feasible in the program window, by writing the FA forward proof in
pseudocode + sketching the lemma needs, and recommending a mask-handling
option. It is the highest-ROI deliverable of Phase A; its judgment drives
the Gate A → B decision.

The document is research analysis, not Lean code. No Lean theorems are
written here; pseudocode + math + lemma enumeration only.

---

## TL;DR

**Feasibility judgment: CONDITIONAL GO for T3-B.** T3-A (FA-1 forward)
is feasible in 8–12 weeks with medium-high confidence given a clean Phase
B `forLoop_inv` and a chosen mask strategy. T3-B (FA-2 + the headline
`fa1_eq_fa2` corollary) is feasible *only if* (a) Phase C lands the
`(m, l, O)` invariant tooling in re-usable form, (b) the multi-block
`disjointWrites` lemma is bounded to the row-major contiguous output
layout (per the existing PLAN risk register entry), and (c) we accept
descoping the FA-2 mask-skipping optimization to a runtime-equivalent
predicate-without-skip path if it overruns. If any of (a)/(b)/(c) slip,
fall back to T3-A only and run an extended Tier 2 (per Gate A → B rule).

**Mask handling: recommend Option β (mask-predicate denotation).** The
extended-reals route (Option α) is mathematically tidy in isolation but
forces every Phase C op (`Op.dot`, `Op.where`, `Op.reduceMax`,
`Op.reduceSum`) to compose on `WithBot ℝ` / `EReal`, which breaks
well-trodden `Finset.sum`-based proof paths. Option β keeps `ℝ`
everywhere, isolates mask handling at the score-shaping layer, and lets
us re-use the existing softmax/LSE proof skeletons. The added cost is
threading an explicit boolean mask register through the kernel embedding;
that cost is local and well-bounded.

**Top three risks (rank-ordered):**
1. `Op.dot` over `Value.tile2D` plus `Finset.sum`-distribution lemmas:
   simp behavior may resist the way we want, mirroring the P1 `Value.bop`
   pain point (already in PLAN risk register but underweighted).
2. `forLoop_inv` with index-bound-before-body (Phase B prerequisite for
   T3): if its statement form does not generalize to nested loops cleanly,
   FA's Q-block × KV-block double loop pays the cost.
3. Multi-block disjoint-writes proof for FA-2's 3D grid (`batch × heads ×
   Q-blocks`): the PLAN's "200–300 lines" estimate is the most uncertain
   number in the program; a 2× overrun is plausible.

The remainder of this document develops the algorithm, invariant, lemma
inventory, mask-handling comparison, risk register, and feasibility
judgment in detail.

---

## 1. FA forward algorithm in pseudocode

This section writes FlashAttention forward in a style close to our
`triton { ... }` macro, then more verbosely line-by-line, then notes the
deltas for FA-2.

### 1.1 The kernel as Triton-style code

Inputs / shapes (single-head, single-batch for now; FA-2 multi-head/batch
addressed in §1.3):

* `Q : (S × D) ℝ` — queries
* `K : (S × D) ℝ` — keys
* `V : (S × D) ℝ` — values
* `Y : (S × D) ℝ` — output
* `Bq` — Q block size (e.g. 64); `Bk` — KV block size (e.g. 64)
* `S` divisible by both for cleanliness (otherwise tail-block masking; a
  P3+ engineering detail)
* `causal : Bool`
* `softmax_scale : ℝ` — typically `1 / sqrt(D)`

```
# fa_forward (FA-1 style; verbose pseudo-Triton)
@triton.jit
def fa_forward(Q, K, V, Y, S, D, Bq, Bk, scale, causal):
    pid_q   = tl.program_id(0)                       # which Q block
    qoffs   = pid_q * Bq + tl.arange(0, Bq)          # rows we own
    doffs   = tl.arange(0, D)                        # full feature axis
    Q_blk   = tl.load(Q[qoffs, doffs])               # (Bq, D)

    # accumulators per Q row
    m_i = tl.full((Bq,), -inf)                       # running max
    l_i = tl.full((Bq,), 0.0)                        # running denominator
    O_i = tl.full((Bq, D), 0.0)                      # running output

    n_kv_blocks = S // Bk
    for j in range(0, n_kv_blocks):
        koffs = j * Bk + tl.arange(0, Bk)
        K_blk = tl.load(K[koffs, doffs])             # (Bk, D)
        V_blk = tl.load(V[koffs, doffs])             # (Bk, D)

        # scores: (Bq, Bk) = (Bq, D) · (D, Bk)
        S_ij = tl.dot(Q_blk, K_blk.T) * scale

        if causal:
            # mask[r, c] = True ⇔ qoffs[r] < koffs[c]  (forbidden)
            mask = qoffs[:, None] < koffs[None, :]
            S_ij = tl.where(mask, -inf_sentinel, S_ij)

        # online softmax update
        m_new = tl.maximum(m_i, tl.max(S_ij, axis=1))   # (Bq,)
        alpha = tl.exp(m_i - m_new)                     # (Bq,)
        P_ij  = tl.exp(S_ij - m_new[:, None])           # (Bq, Bk)
        l_new = alpha * l_i + tl.sum(P_ij, axis=1)      # (Bq,)
        O_i   = alpha[:, None] * O_i + tl.dot(P_ij, V_blk)
        m_i, l_i = m_new, l_new

    Y_blk = O_i / l_i[:, None]                       # (Bq, D)
    tl.store(Y[qoffs, doffs], Y_blk)
```

Important notes about this rendering:

* `tl.dot` is **new** for Phase C — does not exist in Core.lean yet.
* `tl.where` and `<` are **new** — do not exist in Core.lean yet.
* `tl.maximum` is the elementwise max (analogous to `Op.max2`, which we
  do have); `tl.max(_, axis=1)` is reduceMax along an axis (we have axis-0
  reduce; this is a 2D extension).
* `K_blk.T` (transpose) is implicit in `tl.dot(Q, Kᵀ)` — in production
  Triton there's no explicit transpose op; `tl.dot` accepts a "trans_b"
  flag. For our embedding, we either model `Op.transpose` or bake the
  transpose into a `tl.dotT` flavor.
* The `[:, None]` broadcast notation maps to `Op.broadcastTo` on a
  specific axis.
* `qoffs[:, None] < koffs[None, :]` produces a `(Bq, Bk)` Boolean tile;
  this is where the comparison op shows up.

### 1.2 Same kernel, more verbose (line-by-line annotated)

For the Lean walk-through proof, the kernel will be ~25–30 statements
once expanded. Annotated:

```
# Step 0: identify the Q block we own
pid_q     = tl.program_id(0)
qoffs     = pid_q * Bq + tl.arange(0, Bq)            # (Bq,) of Nat

# Step 1: load Q for our block — needs 2D load
Q_blk     = tl.load(Q[qoffs[:, None] * D + doffs[None, :]])
                                                     # (Bq, D)

# Step 2: initialize accumulators
m_i       = tl.full((Bq,), -inf)                     # masking sentinel
l_i       = tl.full((Bq,), 0.0)
O_i       = tl.full((Bq, D), 0.0)

# Step 3: outer loop over KV blocks
for j in range(0, n_kv_blocks):
    koffs = j * Bk + tl.arange(0, Bk)
    K_blk = tl.load(K[koffs[:, None] * D + doffs[None, :]])   # (Bk, D)
    V_blk = tl.load(V[koffs[:, None] * D + doffs[None, :]])   # (Bk, D)

    # Step 3a: dot product
    QKt   = tl.dot(Q_blk, K_blk, trans_b=True)       # (Bq, Bk)
    S_ij  = QKt * scale                              # (Bq, Bk)

    # Step 3b: causal mask
    if causal:
        cond  = qoffs[:, None] < koffs[None, :]      # (Bq, Bk) bool
        S_ij  = tl.where(cond, -inf_sentinel, S_ij)

    # Step 3c: online softmax
    rowmax = tl.max(S_ij, axis=1)                    # (Bq,)
    m_new  = tl.maximum(m_i, rowmax)                 # (Bq,)
    alpha  = tl.exp(m_i - m_new)                     # (Bq,)
    P_ij   = tl.exp(S_ij - m_new[:, None])           # (Bq, Bk)
    rowsum = tl.sum(P_ij, axis=1)                    # (Bq,)
    l_new  = alpha * l_i + rowsum                    # (Bq,)

    # Step 3d: output rescale + accumulate
    PV     = tl.dot(P_ij, V_blk)                     # (Bq, D)
    O_new  = alpha[:, None] * O_i + PV               # (Bq, D)

    # Step 3e: roll forward
    m_i, l_i, O_i = m_new, l_new, O_new

# Step 4: normalize and store
Y_blk     = O_i / l_i[:, None]                       # (Bq, D)
tl.store(Y[qoffs[:, None] * D + doffs[None, :]], Y_blk)
```

This is what the Lean kernel embedding looks like, modulo the macro's
syntax sugar. It's tractably long: ~25 statements wrapping a single
`forLoop`, plus the inner body which is itself ~10 statements.

### 1.3 What changes for FA-2 (delta only)

FA-2 vs FA-1 — the engineering deltas, not the algorithmic core:

1. **Sequence-length parallelism / 3D grid.** FA-2 launches with a
   `(batch, heads, Q-block)` grid; each program_id processes one Q block
   for one (batch, head). FA-1 typically had a 2D `(batch, heads)` grid
   with the Q loop *inside* the kernel. This means our `multiBlockExec`
   in Phase D needs to support multi-axis program_ids.

2. **Delayed rescaling.** FA-2 does not rescale `O_i` by `alpha` at every
   step. Instead it accumulates an unrescaled `O_i` and tracks the running
   `m_i` plus a separate "to-be-applied" rescale chain; only at the very
   end is `O` divided by `l` (and at certain re-anchor points, multiplied
   by `exp(m_old − m_new)`). The math identity:

   ```
   (∏ alpha_k) · O_at_step_k(unrescaled) = O_at_step_k(rescaled)
   ```

   Saves one fp-multiply per inner iteration. Equivalent at infinite
   precision. Provable via `delayed_rescale_eq` (PLAN already names it).

3. **Block-skipping.** When the entire `S_ij` for a (Q-block, KV-block)
   pair is fully masked (e.g., causal with `pid_q < j_min`), the FA-2
   kernel skips the inner work — no load, no dot, no update. Provable via
   the mask-handling lemma stating "fully-masked block contributes 0 to
   the recurrence", which is exactly the inductive identity step with the
   block's contribution being zero.

4. **(Minor) Better warp-level primitives.** FA-2 uses different
   intra-warp reductions; algorithmically the operational semantics
   doesn't see this distinction.

For Phase D, deltas (1)–(3) are formal, (4) is invisible to us. Note
that delta (1) is the only *new semantic capability* required; (2) and
(3) reuse Phase C tooling.

---

## 2. Per-iteration invariant

### 2.1 The invariant statement

Fix a single Q block index `i`. Index KV blocks by `j ∈ {0, …, K}` where
`K = ⌈S/Bk⌉`. After processing the first `k` KV blocks (i.e., `j ∈
[0, k)`), define:

```
maskedScore(i, j) := { S_ij[i.row, j.col] · scale       if not masked
                     { -∞ (or "masked")                 if masked
```

For each row `r ∈ Fin Bq` of the Q block, let `idxs(r, k) ⊆ Fin S` be
the set of KV-column indices in the first `k` KV blocks that are *not*
masked at row `r`:

```
idxs(r, k) := { c ∈ Fin (k · Bk) | not masked_at(qoffs[r], c) }
```

Then the invariant `Inv_k` says:

```
m_{i,k}[r] = max over c ∈ idxs(r, k) of score(qoffs[r], c)
             (or -∞ if idxs(r, k) is empty)

l_{i,k}[r] = ∑_{c ∈ idxs(r, k)} exp(score(qoffs[r], c) - m_{i,k}[r])

O_{i,k}[r, d] = ∑_{c ∈ idxs(r, k)} exp(score(qoffs[r], c) - m_{i,k}[r])
                                   · V[c, d]
```

For Option α (extended reals), the above can be written uniformly with
`-∞ ↦ 0` under `exp`. For Option β (mask-predicate), the mask is
threaded as the `idxs` filter directly.

After all `K` KV blocks are processed:

```
Y[qoffs[r], d] = O_{i,K}[r, d] / l_{i,K}[r]
```

provided `l_{i,K}[r] > 0` (i.e., the row has at least one unmasked
position; this is always true for non-degenerate causal masks where row
`r` can attend to itself).

### 2.2 Inductive step proof sketch (in math, not Lean)

Base case, `k = 0`: empty index set. `m = -∞`, `l = 0`, `O = 0`. All
hold by the kernel's accumulator init.

Step case, `k → k + 1`: assume `Inv_k`. Let `s_new(r, c) := score(qoffs[r],
koffs[c])` for `c ∈ Fin Bk` ranging over the new KV block (with `koffs[c]
:= k · Bk + c`). Let `mask_new(r, c)` be the new block's mask predicate.
The kernel computes:

```
S_ij[r, c]   = if mask_new(r, c) then -∞ else s_new(r, c)
rowmax[r]    = max over c of S_ij[r, c]                      (1)
m_new[r]     = max(m_i[r], rowmax[r])                        (2)
alpha[r]     = exp(m_i[r] - m_new[r])                        (3)
P_ij[r, c]   = exp(S_ij[r, c] - m_new[r])                    (4)
rowsum[r]    = ∑_c P_ij[r, c]                                (5)
l_new[r]     = alpha[r] · l_i[r] + rowsum[r]                 (6)
PV[r, d]     = ∑_c P_ij[r, c] · V_blk[c, d]                  (7)
O_new[r, d]  = alpha[r] · O_i[r, d] + PV[r, d]               (8)
```

Goal: `(m_new, l_new, O_new)` satisfies `Inv_{k+1}`.

For `m_new[r]`:
* `Inv_k`: `m_i[r] = max over old idxs of score(...)`
* `(1)`: `rowmax[r] = max over new unmasked c of s_new(r, c)`
* `(2)`: `m_new[r] = max of (old max, new max) = max over union`. This
  is `idxs(r, k+1)`'s max. ✓

For `l_new[r]`:
* `Inv_k`: `l_i[r] = ∑_{c ∈ idxs(r, k)} exp(score(...) - m_i[r])`
* `alpha[r] · l_i[r] = ∑_{c ∈ idxs(r, k)} exp(score(...) - m_new[r])`
  by `exp(a - b) · exp(c - a) = exp(c - b)` (key algebraic step).
* `rowsum[r] = ∑_{c ∈ new unmasked} exp(s_new(r, c) - m_new[r])`.
  Masked positions contribute `exp(-∞ - m_new[r])` which we need to
  equal 0. This is the masking lemma. Under Option α it's by extending
  `exp` to ⊥. Under Option β it's by filtering before summing.
* Sum over `idxs(r, k+1)` = sum over old + sum over new. ✓

For `O_new[r, d]`: same algebra as `l_new` but each summand multiplied
by `V[c, d]`. The `PV` step `(7)` factors as `tl.dot(P_ij, V_blk)`,
which by `Op.dot`'s denotation equals `∑_c P_ij[r, c] · V_blk[c, d]` —
exactly what we want.

The conclusion `Y = O_K / l_K` follows from `Inv_K` and division.

### 2.3 Connection to Phase B's online softmax recurrence

PLAN's Phase B theorem #5 — `online_softmax_recurrence_eq_batch` — proves
the `(m, l)` part of this invariant for a 1D streaming case (no `O`, no
masking, no dot). Phase C's job is to:

1. Promote the `(m, l)` recurrence from 1D scalars to per-Q-row tiles,
   i.e., from `m_k : ℝ`, `l_k : ℝ` to `m_k : Fin Bq → ℝ`, `l_k : Fin
   Bq → ℝ`. This is mostly a `Finset`-pointwise lift of the existing
   scalar lemma, not a re-proof.
2. Add the `O` accumulator with the same recurrence shape. The proof
   for `O_k` is structurally identical to `l_k` except multiplied by
   `V[c, d]` per term — a `Finset.sum_mul` argument.
3. Add the masking layer (per Option α / β choice — see §4).

A productivity multiplier: if Phase B's #5 is stated *generically over
the term being summed* — i.e., as the recurrence for `∑ f(c) ·
exp(score(c) - m)` for an arbitrary `f : Fin n → ℝ` — then both `l_k`
(`f = 1`) and `O_k` (`f = V[c, d]` for each fixed `d`) are corollaries.
This is a Phase B-side optimization to flag in the Phase B implementation
plan: write the recurrence for the masked, weighted, online softmax
*sum*, not just for `(m, l)`.

[CHECK: Whether to push the generalization into Phase B or do it lazily
in Phase C is a Phase A → B handoff decision; recommended in §6.]

---

## 3. Lemma needs across phases

This section enumerates every Lean lemma the FA forward proof will
need, marked by status: **(existing)** (in Mathlib or our project),
**(new, easy)** (≤ 30 lines), **(new, medium)** (~50–150 lines),
**(new, hard)** (significant effort, may need creative approach), or
**(UNKNOWN)** (uncertain).

### 3.1 Phase B prerequisites

These must land before Phase C can begin in earnest.

| Lemma / definition | Status | Notes |
|---|---|---|
| `forLoop` operational semantics (mutual `stepStmt`/`stepStmts`) | new, medium | Existing PLAN delivers; ~80 lines |
| `forLoop_inv` with index binding | new, medium | The workhorse; PLAN §Phase B writes the signature; ~50–80 lines |
| `online_softmax_recurrence_eq_batch` (#5) | new, medium | Paper centerpiece, ~80–120 lines per PLAN |
| `Finset.sum_filter` and friends for masked sums | existing | Mathlib `Finset.sum_filter`, `Finset.sum_ite_zero` |
| `Real.exp_sub`, `Real.exp_add`, `Real.exp_pos`, `Real.exp_zero` | existing | Mathlib |
| `Real.exp_le_exp`, `Real.exp_lt_exp` (for max bounds) | existing | Mathlib |
| `Finset.sup'_mono` (max over a superset ≥ max over subset) | existing | Mathlib |
| `welford_kernels_refinement` (#4) — uses `forLoop_inv` | new, medium | Validates `forLoop_inv` shape before T3 |
| `layernorm_kernels_refinement` (#6) | new, medium | Composes Welford + affine |

**Phase B prep recommendation for T3:** state #5 in the
"weighted-and-masked online sum" form so the `O_k` recurrence in Phase C
is a parameter instantiation. See §6 and §2.3.

### 3.2 Phase C new needs

#### 3.2.1 Semantics extensions (in `VeriTile.Triton.Core` / `Semantics`)

| Item | Status | Estimated LoC |
|---|---|---|
| `Value.tile2D : (m n : Nat) → (Fin m → Fin n → ℝ) → Value` constructor | new, easy | 5 |
| `Value.bop2D` pointwise lift on tile2D | new, easy | 15 |
| `Value.uop2D` pointwise lift | new, easy | 5 |
| `Value.reduceMax2D`/`reduceSum2D` (with axis selection) | new, medium | 30 |
| `Op.dot a b` constructor in Core | new, easy | 5 |
| `evalOp .dot` clause: `∑ⱼ A[i,j] · B[j,k]` via Mathlib `Finset.sum` | new, medium | 30 |
| `Op.broadcastTo : Op → Shape2D → Op` (1D→2D promotion) | new, medium | 25 |
| `Op.transpose` (or `Op.dotT` flag on dot) | new, easy | 10 |
| `Op.lt`, `Op.le` returning `Value.tile` of 0/1 ℝ (or boolean) | new, easy | 20 |
| `Op.where(cond, a, b)` element-wise mask | new, medium | 30 |
| `Op.maximum` (elementwise max — `max2` already exists for scalars; needs lifting) | new, easy | 10 |
| Causal indexing helpers: `qoffs`, `koffs`, "this Q-block", "this KV-block" | new, medium | 50 |
| `observeY2D : Option BlockState → Fin S → Fin D → Option ℝ` | new, easy | 15 |
| 2D `tl.load` / `tl.store` semantics (gather/scatter on 2D offsets) | new, medium | 60 |
| `BlockState.scatter_readback_2D` (analog of existing 1D version) | new, medium | 80 (induct over rows × scatter readback per row) |
| 2D-row-disjointness lemma (kept light: pid_q rows owned uniquely) | new, easy | 20 |

**Subtotal: ~410 lines of semantics extensions.** (PLAN's Phase C
"~150 lines" estimate is too low; PLAN's revised "~400–600 lines total"
absorbs the gap into the proof side. We re-allocate: ~400 semantics +
~300 proof + ~100 mask = ~800 lines for Phase C. This is the largest
single Phase delta in the program.)

#### 3.2.2 `tl.dot` proof-side simp lemmas

| Item | Status | Estimated LoC |
|---|---|---|
| `dot_def_unfold` — `(A · B)[i, k] = ∑ⱼ ...` (rfl-style) | new, easy | 5 |
| `dot_distrib_left` — `(A + A') · B = A · B + A' · B` | new, easy | 20 |
| `dot_smul_left` — `(c · A) · B = c · (A · B)` | new, easy | 15 |
| `dot_row` — `(A · B)[i, ·] = A[i, ·] · B` (pulls out a row) | new, easy | 15 |
| `dot_zero_row` — if `A[i, :] = 0` then `(A · B)[i, :] = 0` | new, easy | 15 |
| `dot_sum_distrib` — `(∑_k A_k) · B = ∑_k A_k · B` | new, medium | 30 |
| Custom `simp` set tagging for the above | new, easy | 5 |

**Risk:** as PLAN's risk register flags (and as P1 already saw with
`Value.bop`), `simp` behavior over our `Value`-wrapped `tile2D` may not
be cooperative. Mitigation is custom simp lemmas + `unfold +
induction` fallback. Allocate 80 lines for "simp wrangling" beyond the
mathematical content; this accounts for PLAN's risk register entry but
is not the dominant cost.

#### 3.2.3 Mask handling (depends on Option α / β — see §4)

If Option β chosen (recommended), additional Phase C work:

| Item | Status | Estimated LoC |
|---|---|---|
| `Op.where` denotation (already counted above) | new, medium | (above) |
| `causalMask : Nat → Nat → Bool` (the predicate) | new, easy | 5 |
| `mask_score_zero_when_masked` simp lemma | new, easy | 15 |
| `Finset.sum_filter_eq_sum_ite` to manipulate masked sums | existing | 0 |
| `maskedSoftmax` denotation function | new, easy | 10 |
| `standardAttentionMath Q K V causal` (the spec target) | new, easy | 15 |

#### 3.2.4 The main theorem proof body

| Item | Status | Estimated LoC |
|---|---|---|
| Per-Q-row invariant statement `Inv_k` | new, easy | 25 |
| `forLoop_inv` instantiation for FA inner loop | new, medium | 40 |
| Inductive step: `(m, l, O)` recurrence | new, hard | 80–120 |
| Algebraic identities used in step (`exp_sub` chain, sum splits) | mix | 30 |
| Final composition (post-loop normalization) | new, medium | 30 |
| Tying back to `standardAttentionMath` (the spec) | new, medium | 40 |
| `fa_forward_correct` headline statement + proof glue | new, easy | 30 |

**Subtotal Phase C theorem proof: ~280–340 lines.** Combined with
semantics extensions: **~700–820 lines for Phase C.** This aligns with
PLAN's revised "~400–600 lines" estimate plus the masking work; we end
up at the upper end (or slightly past it) once everything is counted.

[CHECK: PLAN's revised Phase C estimate of "400-600 lines" deserves to
be updated to "600-900 lines" based on this enumeration, since PLAN
seems to count "lines of new content" but does not split semantics
extensions vs proof body — clarify with the Phase A author.]

### 3.3 Phase D new needs

#### 3.3.1 Multi-block infrastructure

| Item | Status | Estimated LoC |
|---|---|---|
| `Grid : Type` — multi-axis program_id space (1D / 2D / 3D) | new, medium | 30 |
| `multiBlockExec : Kernel → InitMem → Grid → FinalMem` | new, medium | 50 |
| Pid extraction: `tl.program_id(axis)` for `axis ∈ {0, 1, 2}` | new, easy | 20 |
| Disjoint-output-region per pid (the spec) | new, medium | 40 |
| `disjointWrites : Kernel → Grid → Prop` predicate | new, medium | 30 |
| `disjointWrites_implies_local_reasoning` lemma | new, hard | 80–120 |
| Stride / row-major contiguous output layout helpers | new, medium | 50 |
| `multiBlockExec_compose` lemma (combine per-pid results) | new, hard | 60 |

**Subtotal: ~360–420 lines.** PLAN's revised "200–300 lines" for this
is on the low side once you count the multi-axis program_id semantics
and the disjoint-writes proof. Likely 300–450 in practice.

[CHECK: The disjoint-writes lemma's exact statement isn't in PLAN; the
shape "writes to disjoint regions ⇒ multi-block result is the ⊔ of
per-block results" is the obvious one but may need refinement once
the kernel embedding lands.]

#### 3.3.2 FA-2-specific lemmas

| Item | Status | Estimated LoC |
|---|---|---|
| FA-2 kernel embedding via `triton { ... }` | new, easy | 60 |
| Multi-axis pid in DSL | new, medium | 30 |
| `delayed_rescale_eq` — the math identity | new, medium | 50 |
| Block-skipping correctness — masked block contributes 0 | new, medium | 40 |
| Reuse of Phase C `(m, l, O)` invariant tooling | (existing) | (rely on Phase C) |
| `fa_2_forward_correct` proof body | new, hard | 150–200 |
| `fa1_eq_fa2` corollary by transitivity | new, easy | 30 |

**Subtotal Phase D theorems: ~360–410 lines.** Plus the multi-block
infrastructure: **~720–830 lines for Phase D.**

PLAN's revised "200–300 lines semantics + 250–400 lines theorems" =
450–700; my estimate is somewhat higher but within the same order. The
gap suggests adding ~100 line cushion to the Phase D budget for the
disjoint-writes proof particularly.

### 3.4 Aggregate

```
Phase B prep work for T3:                        (already in PLAN)
  – generalized online_softmax_recurrence: +30 lines vs vanilla #5
  – `forLoop_inv` shape-validation via #4:       (already in PLAN)

Phase C new Lean LoC estimate:                   ~700–820 lines
  ├── Semantics (tile2D, dot, where, broadcast, …): ~410
  ├── Masking helpers (Option β):                ~50
  ├── Simp lemmas (dot, etc.):                   ~80
  └── Theorem `fa_forward_correct`:              ~280–340

Phase D new Lean LoC estimate:                   ~720–830 lines
  ├── Multi-block infrastructure:                 ~360–420
  ├── FA-2 kernel embedding + DSL extensions:    ~90
  ├── delayed_rescale_eq + block-skipping:       ~90
  ├── Theorem `fa_2_forward_correct`:            ~150–200
  └── Headline `fa1_eq_fa2` corollary:           ~30

Total Phase C+D new LoC:                         ~1420–1650
```

Compare PLAN's estimate (~400–600 + 200–300 + 250–400 = 850–1300). My
estimate is ~15–30% higher; bid the upper bound for risk margin.

---

## 4. Mask handling — Option α vs Option β

PLAN names two options. This section spells each out concretely with
type signatures, estimated LoC, and pain points, then commits.

### 4.1 Option α — extended reals (`WithBot ℝ` or `EReal`)

The idea: scores live in `ℝ ∪ {⊥}` (or `EReal`'s `ℝ ∪ {-∞, +∞}`).
`Real.exp` is extended so `⊥ ↦ 0` (and `+∞ ↦ +∞`, but we don't need
that direction in FA). Masked positions in the kernel produce `⊥`
*semantically*; the `-1e38` float-level sentinel is then a *concretization*
relevant only to differential testing, not formal proof.

**Type signatures:**

```
-- Score tile uses extended reals
Op.dot           : Op → Op → Op   -- semantics: returns Value.tile2D over EReal
Value.tile2D_e   : (m n : Nat) → (Fin m → Fin n → EReal) → Value
maskedSoftmax_α  : (Fin n → EReal) → (Fin n → ℝ)
                 -- the final softmax falls back to ℝ via exp(⊥) := 0
exp_extended    : EReal → ℝ  (or → ℝ≥0; design choice)
fa_forward_correct
  : ... → ∀ i d, observeY2D ... = some (standardAttentionMath_α Q K V causal i d)
```

**Estimated lines (approximate):**
* New `EReal`-wrapped versions of `Value`, `bop`, `uop`, `reduceSum`,
  `reduceMax`, `dot`, etc. — ~150 lines, since most are mechanical lifts.
* `exp_extended` and its lemmas (continuity, monotonicity, `⊥ ↦ 0`) —
  ~30 lines (Mathlib has `EReal.exp`? [CHECK: Mathlib's `EReal`
  exponential support — likely partial; `Real.exp_neg_top` does not
  exist as such]).
* `Finset.sum` over `EReal` — Mathlib supports it but the API surface
  is different (`Finset.sum_eq_top_iff`, etc.) — proofs need care.
  ~50 lines of bridging lemmas.
* `maskedSoftmax_α` denotation — ~20 lines.
* `standardAttentionMath_α` — ~30 lines.

**Subtotal Option α surcharge: ~280 lines** of *additional* Phase C
work versus a non-mask baseline. The same surcharge would propagate
through to T3 directly.

**Pain points specific to α:**

1. **`Finset.sum` over `EReal`.** Mathlib has it but it's noisier:
   `Finset.sum_eq_top_iff` and the `EReal` arithmetic requires more
   case-splits. Our existing softmax/LSE proofs all use `ℝ`-valued
   `Finset.sum` and a clean `field_simp` finish. Lifting these to
   `EReal` would force re-doing the proofs.
2. **`Op.dot` over `EReal`.** The natural denotation `(A · B)[i, k] =
   ∑ⱼ A[i, j] · B[j, k]` requires `EReal` multiplication, which has
   IEEE-style edge cases (`⊥ · 0 = ?`, `+∞ · ⊥`). The score tile only
   contains finite or `⊥`; the values tile `V` is finite. Their product
   in `(P_ij, V_blk)` — fine. Their product in `(Q_blk, K_blk^T)` —
   *Q* and *K* are finite, so dot is fine. So we never multiply a `⊥`
   by anything. **However**, in the *masked* product `P_ij · V_blk`,
   `P_ij` is `exp(...)` which is finite (`exp(⊥) = 0`), so even there
   we're fine. So `EReal` arithmetic edge cases largely don't bite, but
   we still pay the API noise cost.
3. **The kernel embedding still produces a `-1e38` sentinel** at the
   AST level (because real Triton kernels can't write `⊥`), so the
   correctness theorem has to bridge "kernel produces float `-1e38` →
   spec uses `⊥` → these are equivalent under `exp` + `+ rest`". This
   adds an extra "concretization" lemma per masked-score op. ~40 more
   lines.
4. **Mathlib coverage check:** `Mathlib/Data/Real/EReal.lean` exists;
   `Mathlib/Analysis/SpecialFunctions/Exp` has `Real.exp` only. There
   is no `EReal.exp` ready-made [CHECK: confirm in Mathlib search].
   We'd need to define it, though it's trivial (`fun x => match x with
   | ⊥ => 0 | (r : ℝ) => Real.exp r | ⊤ => something`).

### 4.2 Option β — mask-predicate denotation

The idea: scores stay in `ℝ`. The spec is parameterized by an explicit
mask predicate `mask : Fin S → Fin S → Bool` (or `Fin Bq → Fin Bk →
Bool` per block). The kernel embedding threads a *boolean* mask register
and `Op.where` branches on it. The float-level `-1e38` sentinel — if
present at all — is purely a runtime concretization, not visible at the
proof level.

**Type signatures:**

```
maskedSoftmax_β :
  (x : Fin n → ℝ) → (mask : Fin n → Bool) → Fin n → ℝ
maskedSoftmax_β x mask i :=
  if mask i then 0
  else Real.exp (x i) / ∑ j, if mask j then 0 else Real.exp (x j)

standardAttentionMath_β :
  Matrix (Fin S) (Fin D) ℝ → ... → Bool → Matrix (Fin S) (Fin D) ℝ
standardAttentionMath_β Q K V causal :=
  fun i d =>
    let scores := fun j => (Q i ⬝ K j) * scale
    let mask   := fun j => causal && (j > i)
    ∑ j, (if mask j then 0
          else Real.exp (scores j - row_max_unmasked) /
               ∑ k, if mask k then 0 else Real.exp (scores k - row_max_unmasked))
        · V j d

-- Kernel-level: a new Op carrying a boolean tile through Op.where
Op.where : Op → Op → Op → Op   -- where(cond, a, b)
-- semantics: requires `cond` to be a tile of 0/1 ℝ (representing bool)
-- or a dedicated boolean tile constructor

fa_forward_correct
  : ... → ∀ i d, observeY2D (exec FAForwardKernel s) i d
                 = some (standardAttentionMath_β Q K V causal i d)
```

**Estimated lines (approximate):**
* `Op.where` semantics on tile2D (counted in §3.2.1 already) — 0
  marginal.
* `Op.lt` / `Op.le` returning 0/1 ℝ tile (counted in §3.2.1) — 0
  marginal.
* `maskedSoftmax_β` and its lemmas — ~30 lines.
* `standardAttentionMath_β` denotation — ~30 lines.
* "Masked score → kernel `where` → spec `if mask then 0`" bridging
  lemma — ~25 lines.
* Mask-predicate composition lemmas (causal mask, full mask) — ~15
  lines.

**Subtotal Option β surcharge: ~100 lines** of additional Phase C
work. This is ~3× cheaper than Option α.

**Pain points specific to β:**

1. **Boolean tile representation.** The simplest encoding is "tile of
   0/1 ℝ"; `Op.where(cond, a, b)` evaluates `cond_i` and selects `a_i`
   if `cond_i = 1` else `b_i`. This works but is coupled to the `ℝ`
   type. An alternative is a dedicated `Value.tileBool`, but that bloats
   the `Value` enum. Recommend the 0/1 ℝ path; document the convention.
2. **Mask predicate threading.** The kernel must produce the predicate
   *as a runtime value* — i.e., `qoffs[:, None] < koffs[None, :]` is
   computed at kernel run time, not statically. The `Op.lt` semantics
   on tile2D must produce 0/1 values. Ours does (per §3.2.1). Fine.
3. **The proof's masked-vs-unmasked sum split.** When proving
   `Inv_{k+1}` for `l_new`, the inner `rowsum[r]` is `∑_c P_ij[r, c]`
   which equals `∑_c (if mask_new(r, c) then 0 else exp(s_new(r, c) -
   m_new[r]))`, and we need this to equal `∑_{c ∈ unmasked} exp(...)`.
   Mathlib's `Finset.sum_filter` and `Finset.sum_ite_eq` handle this
   cleanly. Should be ~10 lines per use, ~3 uses = 30 lines.
4. **`Op.where` interacts with `simp` cleanly when the predicate
   reduces.** For static predicates (e.g., causal mask with fixed Q
   and K block indices), `simp [if_pos, if_neg]` should work. For
   dynamic predicates we keep the conditional.

### 4.3 Side-by-side comparison

| Dimension | Option α | Option β |
|---|---|---|
| New Lean LoC (Phase C) | ~280 | ~100 |
| Mathlib API smoothness | poor (`EReal` API less developed than `ℝ`) | excellent (`ℝ` only) |
| Reuses existing softmax/LSE proofs | partial (lifted) | full |
| Kernel embedding faithfulness | needs concretization bridge | direct (mask register matches Triton bool) |
| Spec mathematical elegance | very good (uniform `⊥` handling) | good (explicit mask is clear) |
| FA-2 block-skipping proof | natural (whole block ⊥ ⇒ contributes 0) | natural (whole block masked ⇒ filter empty) |
| Risk of unforeseen `EReal` API gaps | medium (Mathlib `EReal.exp` not standard) | low |
| Compatibility with `Op.dot`'s `Finset.sum` | moderate (lift to `EReal`) | trivial (`ℝ`) |

### 4.4 RECOMMENDATION

**Option β.** Reason: ~3× cheaper Phase C surcharge; preserves the
existing `ℝ`-based proof skeletons developed in Phases A and B; isolates
all mask reasoning to a single layer (`Op.where` + spec-level
`maskedSoftmax_β`). Option α's mathematical elegance is real but the
engineering cost — particularly the lift of `Op.dot` and `Finset.sum`
over `EReal`, plus the `EReal.exp` definitional and lemma work that
appears not to be standard in Mathlib — outweighs the elegance gain
within the program window.

**Caveat:** if Phase B work surfaces a use-case where extended reals
genuinely cleans up some other proof (e.g., LayerNorm with infinite
ε), revisit. But for FA forward, β wins.

This recommendation will land as the Phase C implementation plan's
default; Gate B → C reviews it before commitment.

---

## 5. Risk register

This is FA-specific. PLAN's program-level risks already cover the
broader set; this section goes deeper.

| # | Risk | Likelihood | Detection | Mitigation | Phase impact |
|---|---|---|---|---|---|
| R1 | `online_softmax_recurrence_eq_batch` proof (Phase B #5) does not generalize to weighted-and-masked sum cleanly | Medium | Phase B mid-point review; if the generic statement is hard, fall back to scalar `(m, l)` | Bid the generalization in Phase B; if it fails, add a Phase C wrapper lemma that reduces `O_k` proof to scalar `l_k` proof iterated over `d ∈ Fin D` (~+50 lines vs sharing) | Phase C +50 lines |
| R2 | `forLoop_inv` interface design fails to support nested loops needed by FA's Q-loop × KV-loop | Medium-Low | Phase A T3 scout flags it (this doc, §1); Phase B detection in `welford_kernels_refinement` (which is single-loop) doesn't catch it; first surfaces in early Phase C | If `forLoop_inv` requires statement-level mutual recursion to nest, restate it in `Nat.iterate` form; FA only needs *sequential nesting* (outer Q-loop is the only loop in FA-1; KV-loop is the inner one — actually FA-1 has the Q loop *inside* the kernel typically, but in our embedding pid_q is the outer "loop" via grid, and KV is the only `forLoop`). So FA-1 is **single-forLoop** in our embedding. FA-2 same. So nesting is not an immediate issue. | Phase B +30 lines (defensive) |
| R3 | `tl.dot` simp behavior on `Value.tile2D` is bad (P1 already saw with `Value.bop`) | High | First Phase C theorem attempt that uses `Op.dot` | Custom simp lemmas (allocated 80 lines in §3.2.2); if simp won't cooperate, write `unfold + induction` style proofs with explicit `Finset.sum` manipulation | Phase C +80–150 lines |
| R4 | Mask-handling Option β turns out unworkable mid-Phase C (e.g., `Op.where`'s denotation has `simp` issues) | Low-Medium | First Phase C theorem proof gets stuck on a mask predicate | Switch to Option α; cost: redo ~150 lines of Phase C masking work + redo Phase B exp lemmas under `EReal`. Estimated 2-week setback | Phase C +200 lines, +2 weeks |
| R5 | Multi-block disjoint-writes proof is harder than 200–300 lines | Medium-High | Early Phase D when drafting `multiBlockExec` | Restrict to row-major contiguous per-program_id (already in PLAN as a fallback); document the restriction in paper; if even that's too hard, descope to single program_id reasoning + spec-level "obvious" composition (less satisfying but viable) | Phase D +100–200 lines or descope |
| R6 | Causal-indexing arithmetic (Fin manipulation for Q/KV block indices) creates many small lemmas | High | Throughout Phase C | Most of these are Mathlib-trivial (`Fin.val_lt`, `Nat.add_lt_iff`, etc.); plan to spend 1–2 days on a "Fin/Nat hygiene" pass collecting these into a single helper file | Phase C +50 lines |
| R7 | IEEE-754 vs ℝ gap unexpectedly large on representative kernel | Low (FA forward) | Phase C diff-test artifact | Document gap; the formal proof remains valid; PLAN already covers this in §Differential testing | Phase C: artifact only |
| R8 | `/lean4:autoprove` plugin makes minimal dent on FA proofs | Medium | Phase C held-out close rate < 30% | Honest report; split long proofs into sub-lemmas the plugin can close + human-written composition (PLAN already plans this) | Phase C: schedule only; +1–2 weeks for hand-proving |
| R9 | Phase C+D Lean LoC estimate is 30%+ wrong | Medium | Mid-Phase C status check | Bid upper bound (this doc estimates ~1420–1650 lines, 30% more than PLAN's nominal); pre-allocate buffer in timeline | Schedule slip |
| R10 | FA-2 delayed-rescale `delayed_rescale_eq` math identity has subtle precondition not noticed at scout time | Medium | Phase D when drafting #8 | Verify on paper before coding; if precondition is "no all-masked Q rows", document as assumption | Phase D +1 week |
| R11 | `Op.broadcastTo` denotation interacts poorly with `Op.dot` (broadcast-then-dot vs dot-then-broadcast equivalences) | Medium | Phase C kernel walkthrough for `(Bq,) + (Bq, Bk)` shape | Add ~3 simp lemmas pulling broadcast through dot; ~30 lines | Phase C +30 lines |
| R12 | 2D `tl.load` / `tl.store` semantics requires more index-arithmetic lemmas than scalar version | Medium-High | Drafting `evalOp .load` clause for `tile2D` offsets | Allocate 60 lines (already in §3.2.1) plus ~30 lines for `realToNat` over 2D indices; total ~90 | Phase C +30 lines on top of estimate |
| R13 | The mask predicate threading through `Op.where` requires the kernel to pre-compute the mask as a tile register, which the DSL macro doesn't currently support | High | First DSL attempt to write FA forward | Extend DSL: add `<` operator, `tl.where(cond, a, b)`, `tl.broadcast_to`, `tl.full((m, n), val)` — ~80 lines of macro extension | Phase C +80 lines (DSL) |
| R14 | Phase C close rate on `/lean4:autoprove` benchmark is < 30% (PLAN's threshold) | Medium-High | End of Phase C | Per PLAN, lower close rates are expected for longer proofs; honest reporting; the benchmark is non-gating for theorem closure | Reporting only |

The dominant FA-specific risks above PLAN's program-level set are **R3
(simp behavior on dot), R5 (multi-block disjoint writes), R12 (2D load
arithmetic), R13 (DSL gap)**. Together these add ~250 lines to the
PLAN estimate and ~3 weeks to the Phase C+D schedule in expectation.

---

## 6. Feasibility judgment for T3-B

### Bottom line

**T3-B is feasible in the program window with conditional GO**, with
the following conditions and confidence assignment:

* **T3-A (Phase C, FA-1 forward correctness):** medium-high confidence
  (75–80%) feasible in the planned 8–12 weeks given:
  * Phase B closes #5 with the generalized weighted-and-masked
    recurrence (per §2.3 and §3.1 recommendation).
  * Mask Option β (per §4.4) is committed to at Gate B → C.
  * `forLoop_inv` lands as specced in PLAN with index binding before
    body execution.

* **T3-B (Phase D, FA-2 forward correctness + `fa1_eq_fa2`):**
  medium confidence (55–65%) feasible in the planned 6–10 weeks given:
  * Phase C's `(m, l, O)` invariant tooling is reusable (i.e., the
    invariant is parameterized by enough of the kernel's structure that
    FA-2 reuses ~70% of it).
  * Multi-block disjoint-writes is bounded to row-major contiguous
    layout (per PLAN risk register R5 mitigation).
  * If the FA-2 block-skipping optimization proves to require more
    formal infrastructure than budgeted, accept descoping to "FA-2 with
    block-skipping replaced by predicate-where-but-no-skip" — the math
    is identical, the engineering distinction is the only thing lost,
    and that loss is paper-acceptable.

### Why "conditional GO" rather than "GO"

The total Phase C+D Lean budget I estimate (~1420–1650 new lines) is
~30% above PLAN's nominal upper bound of ~1300. This is a yellow
flag, not red: the program has scheduling buffer (PLAN's "8–12 wk"
ranges per phase already absorb 30% variance). But the *combination*
of (a) novel semantics (`Op.dot`, `Op.where`, `tile2D`, multi-block),
(b) a long single-theorem proof (FA forward is the longest single
theorem in the program by ~2×), and (c) the dependence on Phase B
producing a generalizable invariant means there is real risk of a
mid-Phase-C realization "we need to redo Phase B's #5".

### Phase B-side recommendations to make T3 easier

1. **Generalize `online_softmax_recurrence_eq_batch` (#5) to
   weighted-and-masked online sums.** Concretely, prove:

   ```
   For any  f : Fin n → ℝ  (the "weights"),
            x : Fin n → ℝ  (the scores),
            mask : Fin n → Bool  (the mask),
   the streaming recurrence with state (m_k, l_k, S_k) where
     S_k = ∑ over unmasked j ∈ [0, k) of f(j) · exp(x(j) - m_k)
   equals the batch form
     S_n = ∑ over unmasked j ∈ [0, n) of f(j) · exp(x(j) - m_n).
   ```

   With `f = 1` we get `l_k`. With `f = V[c, d]` we get `O_k[d]`. With
   mask = always-false we get the unmasked case. This single lemma is
   ~+30 lines vs the scalar `(m, l)` version, but cuts Phase C's
   inductive-step proof from ~150 lines to ~50.

2. **Validate `forLoop_inv` shape** on `welford_kernels_refinement`
   (#4) explicitly with a multi-step inner body (Welford's recurrence
   updates 3 registers per step). If `forLoop_inv` cooperates, FA's
   inner loop will too.

3. **Keep `Value.bop` simp behavior issue** as a documented
   sub-tooling concern. Phase C will hit similar issues with `tl.dot`;
   the simp-tactic-cascade approach used for `bop` will inform `dot`.

### Earliest checkpoint to bail

**Gate C → D is the decisive checkpoint.** Per PLAN's Gate C → D rules:
* Phase C ≤ 10 wk → full Phase D (T3-B is GO at this point).
* Phase C 11–14 wk → simplified Phase D (drop causal mask, drop
  block-skipping, simplified work partitioning) — this is still a
  publishable T3-B, just less ambitious.
* Phase C > 14 wk → skip Phase D, T3-A only.

If by **Phase C week 6** (mid-phase) the `(m, l, O)` invariant is
not closed, that's the early warning to start descope conversations.
Concretely: if `fa_forward_correct`'s inductive step is still
sorry'd after 6 weeks, accept that Phase D will be simplified and
plan accordingly.

### What if NO-GO?

Consolation prize for T3-A only:
* T3-A (FA-1 forward) is itself ambitious — single-program-id full
  FA forward with online softmax + dot + masking is publishable as
  a FA-1 verification result, which no prior Lean work covers.
* Tier 2 expansion to fill the slot: per Gate A → B rule, drop to
  T3-A only and add 1–2 Tier 2 pairs (e.g., scan reordering, RoPE
  rearrangement) → 3 + 4-or-5 + 1 = 8–9 main theorems.
* Paper venue: still PLDI possible but OOPSLA-spring more comfortable.

Stripped FA-2 alternative (if T3-B is partially feasible):
* Drop causal mask (do non-causal only): saves ~100 lines (skip
  `Op.lt`, mask predicate threading on FA-2 side; reuse FA-1's mask
  infrastructure if causal is preserved on FA-1 side and dropped
  only on FA-2 side — but this breaks `fa1_eq_fa2`'s spec equality
  since the two would differ in causal behavior; not viable).
* Drop block-skipping: reduces FA-2 to "FA-1 with delayed rescale".
  Saves ~80 lines, breaks the FA-1 vs FA-2 distinction since
  block-skipping is one of the three FA-2 deltas.
* Drop multi-axis grid (single-batch, single-head): collapses
  `multiBlockExec` to FA-1's already-handled single program_id.
  Saves ~250 lines; FA-1 vs FA-2 then differs only by delayed-rescale
  and block-skipping. Acceptable for paper.

**Recommended descope path (most palatable):** drop multi-axis grid
on FA-2, keep delayed-rescale and block-skipping. Saves ~250 lines.
`fa1_eq_fa2` still meaningful as both kernels run on a single grid
and produce identical output for the same input.

### Action items for Phase B planning

To set Phase B up so T3 is feasible:

1. **Generalize #5 as proposed in §2.3.** Bid +30 lines in Phase B.
2. **State `forLoop_inv` with a multi-statement inner body** and
   validate on `welford_kernels_refinement` (#4) — if Welford's
   3-register-per-step body works, FA's many-register-per-step body
   works.
3. **Reserve a Phase B-end half-day** for a "T3 prerequisites met?"
   review: confirm #5 generalization landed; confirm `forLoop_inv`
   tested on multi-statement body; commit to mask Option β at Gate
   B → C.

[CHECK: Whether the Phase B implementation plan author wants to bid
the generalization (+30 lines) up front or do it lazily in Phase C
(+50 lines moved into Phase C). My recommendation is up front in
Phase B.]

### Summary

```
Recommendation:  CONDITIONAL GO for T3-B.
Confidence:      Medium (55–65%) for full T3-B.
                 Medium-High (75–80%) for T3-A.
Mask option:     β (mask-predicate denotation).
Bail point:      Gate C → D, with mid-Phase-C (week 6) early warning.
```

---

## Appendix A: Reference notes

Sources drawn upon in writing this document:

* **PLAN.md** of this repository (reading: Phases A–D in full, Risk
  register, Mask Handling section, Decision log).
* **`VeriTile/Examples/SoftmaxEq.lean`** — refinement pattern, kernel
  walkthrough, scatter readback usage.
* **`VeriTile/Examples/LogSumExpEq.lean`** — math identity composition
  with kernel walkthroughs; `tileMax` pattern.
* **`VeriTile/Examples/WelfordMath.lean`** — pure math recurrence
  proof pattern; how `welford_eq_two_pass` is structured.
* **`VeriTile/Triton/Core.lean`** — current AST inventory.
* **`VeriTile/Triton/Semantics.lean`** — `evalOp`, `stepStmt`,
  `Value.bop/uop/reduceMax/reduceSum`; `forLoop` returns `none` (not
  yet implemented) per `stepStmt`; `BlockState.scatter_readback` lemma.
* **`VeriTile/Triton/DSL.lean`** — currently supported macro syntax.
* **FlashAttention paper** (Dao et al., NeurIPS 2022), Algorithm 1
  for forward pass; the (m, l, O) recurrence and online softmax
  formulation.
* **FlashAttention-2 paper** (Dao, 2023), §3.1 (work partitioning),
  §3.2 (delayed rescaling).
* **Reference implementation** (`flash_attn` Python package,
  `flash_attn_triton.py`) — algorithm structure and Triton kernel
  patterns; familiarity from background, not consulted live.
* **Mathlib** — `Real.exp`, `Finset.sum`, `Finset.sup'`,
  `Finset.sum_filter`, `EReal` (limited use in §4.1); standard search
  for available API.

[CHECK markers left in document — for follow-up reading]

* §2.3 — Phase B generalization decision (push generalization into
  Phase B vs lazy in Phase C).
* §3.2.4 — PLAN's Phase C estimate of "400–600 lines" should perhaps
  be revised to "600–900 lines" once semantics are split out
  separately.
* §3.3.1 — Disjoint-writes lemma exact statement not yet specified
  in PLAN; placeholder shape used.
* §4.1 — Mathlib's `EReal.exp` not standard; would need defining if
  Option α chosen.
* §6 — Action item for Phase B planning: where to bid the +30 lines
  for the generalized recurrence.

---

*End of T3 scouting document.*
