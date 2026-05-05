# FA-1 Step 1 — 4D layout + strides

> **Status:** Closed (Tier 3-A). The 4D layout work landed on `main` —
> `fa1_forward_correct_4D` and `fa1_forward_correct_4D_causal` in
> `VeriTile/Examples/FlashAttention1/V0.lean`. This note is kept as an
> implementation record. References to "Phase D" predate the 2026-05-05
> PLAN redirect; see `PLAN.md` for the current roadmap.

Design note for tracking issue #39 step 1. Scopes the data-model and
infrastructure changes needed to lift the FA-1 forward kernel from
`[M, D] / [Bk·N, D]` toy shapes to real Triton `[B, H, S, D]`
tensors with explicit strides.

**Non-goals.** This step does not touch boundary masking (Step 2),
causal masking (Step 3), or multi-instance refinement (Step 4). The
verified spec still only covers one `(batch, head, q_block)` program
instance — the lift is purely about realistic memory layout, not about
proof scaling across program instances.

## Current state

`InputAt` and `observeTileAt` (`VeriTile/Examples/Common.lean:89,96`)
are already fully ND-general:

```lean
def InputAt {shape : TileShape} (s : BlockState) (region : RegionName)
    (offsetFn : TileIndex shape → Nat)
    (xs : TileIndex shape → ℝ) : Prop :=
  ∀ idx : TileIndex shape, s.mem region (offsetFn idx) = xs idx
```

`Offset.strided shape strides base` (`Common.lean:116`) handles arbitrary
rank with arbitrary positional strides. So the layout primitives are
already in place — Step 1 does not require any new ND infrastructure on
that front.

What is **missing** for real-Triton FA-1:

1. **`program_id` is single-axis.** `Op.programId : Op .nat []`
   (`Triton/Core.lean:446`) takes no arguments and reads
   `BlockState.pid : Nat` (`Triton/Semantics.lean:140`). Real Triton FA-1
   uses `program_id(0/1/2)` for `(q_block, head, batch)`.

2. **Kernel signature has no strides, no batch/head dims.**
   `fa1ForwardKernel qReg kReg vReg outReg M D Bk numKVBlocks scale`
   (`FlashAttention1.lean:55`) — pointer arithmetic is hardcoded
   row-major: `q_ptrs := offs_m[:, None] * $(D) + offs_d[None, :]`.

3. **Math signature is 2D.** `attentionReal` (`FlashAttention1.lean:140`)
   is `Q : TileIndex [M, D] → ℝ`, `K V : TileIndex [S, D] → ℝ`. Needs
   to lift to `[B, H, S_q, D] / [B, H, S_k, D]`.

4. **`Offset.strided` injectivity is only proven for 1D / 2D.**
   `linear1D_inj` (`Common.lean:157`) and `rowMajor2D_inj`
   (`Common.lean:168`) cover the existing call sites. A 4D injectivity
   theorem (or a generic `strided_inj` under a non-overlap hypothesis) is
   required for the new layout.

## Target state

### Multi-axis `program_id`

Two reasonable encodings; recommend (b):

**(a) `s.pid : List Nat`, `Op.programId (axis : Nat)`.** Simple but
requires bounds reasoning on the axis index, and `s.pid.get? axis` is
partial.

**(b) `s.pids : Nat → Nat`, `Op.programId (axis : Nat)`.** Total. Out-of-
range axes return `0` (or whatever default the launch model picks). Clean
to thread through `setReg_pid` / `writeMem_pid` lemmas (`Semantics.lean:207,226`).
The launch grid stays a separate piece of metadata — the axis count
(rank of the grid) is not encoded in `BlockState`.

Both encodings are honest about the "Build general ND frameworks" memory:
no axis-3-only hardcoding. Backward compat: `s.pid` becomes
`s.pids 0`, the existing `Op.programId` (no axis) becomes
`Op.programId 0`. The DSL parser at `Triton/DSL.lean:377` already accepts
`tl.program_id($e)` and currently ignores the argument; we wire it
through.

### Kernel signature

```lean
def fa1ForwardKernel
    (qReg kReg vReg outReg : RegionName)
    (B H S_q S_k D Bk numKVBlocks : Nat)  -- numKVBlocks = S_k / Bk (assumes divisibility; Step 2 lifts this)
    (stride_qb stride_qh stride_qs : Nat) -- innermost stride for D-axis is 1
    (stride_kb stride_kh stride_ks : Nat)
    (stride_vb stride_vh stride_vs : Nat)
    (stride_ob stride_oh stride_os : Nat)
    (scale : ℝ) : Kernel := triton {
  pid_qb := tl.program_id(0)
  pid_h  := tl.program_id(1)
  pid_b  := tl.program_id(2)

  q_base := pid_b * $(stride_qb) + pid_h * $(stride_qh) + pid_qb * $(M) * $(stride_qs)
  k_base := pid_b * $(stride_kb) + pid_h * $(stride_kh)
  v_base := pid_b * $(stride_vb) + pid_h * $(stride_vh)
  o_base := pid_b * $(stride_ob) + pid_h * $(stride_oh) + pid_qb * $(M) * $(stride_os)

  -- … same online-softmax body, but pointer arithmetic uses stride_*s
  -- instead of $(D), e.g. q_ptrs := q_base + offs_m[:, None] * $(stride_qs) + offs_d[None, :]
  -- …
}
```

`M` (Q-rows-per-block) becomes a derived constant (`BLOCK_M` in Triton
parlance), no longer the full `S_q`. Step 2 will further generalize `M`
to handle non-divisible `S_q`; for Step 1 we keep the `M ∣ S_q` and
`Bk ∣ S_k` invariants explicit.

### Math model

```lean
noncomputable def attentionReal4D
    {B H S_q S_k D : Nat}
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) : TileIndex [B, H, S_q, D] → ℝ :=
  fun (b, h, i, d, _) => attentionReal
    (fun (i', d', _) => Q (b, h, i', d', PUnit.unit))
    (fun (j, d', _)  => K (b, h, j, d', PUnit.unit))
    (fun (j, d', _)  => V (b, h, j, d', PUnit.unit))
    scale (i, d, PUnit.unit)
```

This delegates back to the existing 2D `attentionReal`, so `Stage A` of
the proof (the streaming math identity `streaming_eq_attentionReal`)
**does not change** — it still operates on the 2D slice.

The kernel-vs-spec theorem becomes:

```lean
theorem fa1_forward_correct_4D
    {B H S_q S_k D Bk numKVBlocks : Nat}
    (hBk : 0 < Bk) (hNumKVBlocks : 0 < numKVBlocks)
    (hSk : S_k = Bk * numKVBlocks)
    -- …region / stride parameters…
    (Q : TileIndex [B, H, S_q, D] → ℝ)
    (K V : TileIndex [B, H, S_k, D] → ℝ)
    (scale : ℝ) (s : BlockState)
    (_hQ : InputAt s qReg (Offset.strided [B, H, S_q, D]
              [stride_qb, stride_qh, stride_qs, 1] 0) Q)
    -- … _hK, _hV similarly …
    : ∀ (i : Fin M) (d : Fin D) (u : PUnit),
        observeTileAt
          (exec (fa1ForwardKernel …) s) outReg
          (Offset.strided [B, H, S_q, D]
              [stride_ob, stride_oh, stride_os, 1] 0)
          (s.pids 2, s.pids 1, ⟨s.pids 0 * M + i.val, _⟩, d, u)
        = some (attentionReal4D Q K V scale (s.pids 2, s.pids 1, …, d, u))
```

Note the read-out and load points target only the `(b, h, q_block)` slice
determined by the three `program_id`s — multiple instances and grid
coverage are out of scope (Phase D, issue #5).

### `Offset.strided` injectivity

Step 1 needs **4D injectivity** for `strided [B, H, S_q, D]
[stride_b, stride_h, stride_s, 1]` under suitable non-overlap hypotheses
(`D ≤ stride_s`, `S_q · stride_s ≤ stride_h`, `H · stride_h ≤ stride_b`).
Two ways:

**(a) Hand-roll 4D-specific theorem.** Mirror `rowMajor2D_inj`'s
div/mod-based recovery, but four nested layers. Verbose, ~100 lines.

**(b) General `strided_inj` for arbitrary rank.** Inductive on `shape`
with hypothesis `∀ i, shape[i+1] · strides[i+1] ≤ strides[i]` and
`strides[last] = 1` (or `shape[last] ≤ strides[last-1]`). Pays once,
covers Step 2's 4D + Step 4's mixed layouts. Recommended.

Per the "Build general ND frameworks" memory, prefer (b). The induction
is real but bounded — the structure mirrors `Offset.strided` itself.

## Migration plan

Three PRs, landed in order:

**PR 1 — `program_id` axis parameter.** `Triton/Core.lean` AST,
`Triton/Semantics.lean` `BlockState`, `Triton/DSL.lean` parser, `evalOp`
case. Update every existing kernel to pass axis `0` (mechanical).
Existing FA-1 proof stays green via the `s.pid = s.pids 0` shim.

**PR 2 — `Offset.strided_inj` general theorem.** `Common.lean`. Add
`Offset.strided4D` as a thin alias for `[B, H, S, D]` if desired. No
proof-side migration needed; existing 1D/2D injectivity calls remain
valid.

**PR 3 — FA-1 4D rewrite.** `Examples/FlashAttention1.lean`. New kernel
signature, new math model `attentionReal4D` delegating to 2D, new
correctness theorem `fa1_forward_correct_4D`. The four-stage proof
structure (`fa1_preLoop_correct` / `fa1_step` / `fa1_postLoop_correct`
+ `streaming_eq_attentionReal`) carries over; the `(b, h)` slice is
threaded as additional static context but the recurrences and
invariants don't change.

## Risk register

- **Backward compat for `s.pid`.** Many existing kernel proofs reference
  `s.pid` directly. Plan: introduce `BlockState.pid` as a notation /
  abbreviation for `s.pids 0`, deprecate over time. Avoid a flag-day
  rename.
- **DSL stride syntax.** Currently the DSL uses `$(M) * $(D)` for
  hardcoded shape arithmetic. Stride parameters need to flow through the
  same `$(...)` antiquote channel. No language extension required, just
  more `Nat` parameters in the kernel signature.
- **Proof time.** The four-stage proof is the dominant cost in Tier 2 +
  Tier 3 builds. Adding `(b, h)` as static parameters threaded through
  every helper lemma should not blow up `simp` cost — the data-flow
  signature changes, but the rewriting work per step is the same.
- **Issue #28 (forLoop_inv nesting).** Not directly required by Step 1
  (still single inner KV loop), but worth re-checking once Step 4 lands
  a baseline FA without the online softmax (which may want a different
  loop nesting).
