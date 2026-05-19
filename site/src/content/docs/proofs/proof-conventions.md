---
title: "VeriTile Proof Conventions"
---

Tactic-level conventions that came out of repeated kernel proofs. These are
not absolute rules — when a proof is uncomfortable, you can deviate, but the
default choices below are what most VeriTile kernel proofs use.

## Carrier bridges and `erw`

VeriTile's kernel-side computation lives in `WithBot ℝ`-valued carriers:
`Tile.reduceSum (Tile.bop mul ...)`, `Option.map₂`, `Option.map`,
`WithBot.realSqrt`, `WithBot.unbotD`. The math layer is in pure `ℝ`. We
bridge between them with theorems in `VeriTile.Triton.Semantics.MaskedReduction`
(and similar `Semantics/` mechanism files):

```lean
theorem reduceSum_masked_sq_eq_some_sum
    (load : Fin BLOCK_N → ℝ) (active : Fin BLOCK_N → Prop) [DecidablePred active] :
    @Finset.sum (Fin BLOCK_N) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·) (if-shape) (if-shape))
      = some (∑ k, if active k then load k * load k else 0)
```

### `rw` / `simp_rw` / `simp` often fail; reach for `erw`

The bridge lemma above looks like a clean rewrite target. In practice,
**`rw [bridge_lemma _ _]`, `simp_rw [bridge_lemma _ _]`, and even
`simp [@simp bridge_lemma _ _]` all fail to unify** the lemma's
`∑ k, Option.map₂ ...` pattern with the kernel goal's `∑ x, Option.map₂ ...`,
despite their being alpha-equivalent. The cause appears to be Lean's tactic-
level pattern matcher not crossing the lambda + `FloatDType.cast` /
coercion wrapper boundary.

**Default fallback**: use `erw` (extended rewrite, allows up-to-defeq
matching). The pattern that works:

```lean
have hcarrier := bridge_lemma (fun k => …kernel-side load…)
                             (fun k => …kernel-side mask…)
simp [bridge_unfolds] at hcarrier
-- Now hcarrier is in a kernel-shaped form
…
simp [..., FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot,
      WithBot.realSqrt, ...] -- expose the bare carrier
erw [hcarrier]                -- bridge to pure-ℝ form
rfl                           -- both sides now structurally equal
```

The `erw` step is what unblocked `bench/l2_norm_triton1` etc. via
`l2VarCarrier_eq_l2NormSqSum`. See that file for a working example.

### Cost of `erw`

`erw` is more expensive than `rw` because it tries definitional unfolding
on every match attempt. It is fine inside a single proof step but **do not**
use `erw` in a global simp set or in a heavy loop — the failed-match cases
compound. Localize `erw` to one rewrite at a time.

If `rw` works, prefer `rw`. Reach for `erw` when the goal has a lambda-bound
variable + cast/wrapper structure that prevents direct unification.

## Open `simp` lists: prefer `tile_elementwise`

Kernel proofs typically open with a "kernel-exec normalization" simp. Use
the `tile_elementwise` simp set (registered in
`VeriTile.Triton.Semantics.BroadcastReshape`) to replace the recurring
8-to-15-name list:

```lean
simp [exec, <kernel_def>, stepStmts, stepStmt, evalOp,
      tile_elementwise] at hExec
```

`tile_elementwise` unfolds `Tile.bop / Tile.cop / Tile.uop / Tile.ptrAdd`,
the reshape ops (`Tile.expandDim / Tile.ofReal / Tile.natToReal / Tile.dot /
Tile.transpose / Tile.select`), the reductions (`Tile.reduceSum* /
Tile.reduceMax*`), shape arithmetic (`TileShape.*`), and the per-dtype
projections (`NumericDType.* / IntegralDType.* / ComparableDType.* /
FloatDType.cast / FloatDType.ofWithBot / FloatDType.toWithBot`).

To extend the set, add `attribute [tile_elementwise]` lines in
`Semantics/BroadcastReshape.lean` (the attribute itself is registered in a
sibling file, `Semantics/BroadcastReshape/Attr.lean`, because Lean 4
disallows declaring and using a simp attribute in the same file).

After `subst s'`, the close usually has another simp with the per-kernel
`*Spec`. If `tile_elementwise` covers what the close needs, the close can
be `simp [hi, *Spec]` with no further dtype/tile names.

**Pre-existing proofs**: kernel files that pre-date `tile_elementwise` keep
their explicit simp list. Migrating one is mechanical: replace the 5-15
`Tile.*` / `NumericDType.*` / `ComparableDType.*` / `FloatDType.*` /
`TileShape.*` names with the single `tile_elementwise` token. See
`bench/swiglu_fwd`, `bench/swiglu_triton`, `bench/swiglu_backward`,
`bench/geglu_tanh_triton`, `bench/masked_add_cuda`, `bench/fused_activation`,
`bench/triton_softmax`, `Examples/FlashAttention1/NaiveKernel.lean` for
worked examples.

## `BlockState.pid_eq` normalization

Kernel proofs often have `s.pid` and `s.pids 0` mixed. They are
definitionally equal but `rw` distinguishes them. Convention: include
`BlockState.pid_eq` in the simp list so all `s.pid` get rewritten to
`s.pids 0` (or vice versa, depending on the lemma direction). Do this
**before** applying any `BlockState.scatter_readback_*` lemma.

## `Function.Injective` boilerplate for scatter readback

Most kernel proofs need an injectivity lemma to feed
`BlockState.scatter_readback_prop_masked_nd` (or its non-masked variant).
The standard form:

```lean
have h_inj : Function.Injective
    (fun idx : TileIndex [BLOCK_N] => s.pids 0 * stride + idx.1.val) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ hab
  obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
  rfl
```

This shows up identically in dozens of bench files. If a kernel deviates
from this shape (e.g. 2D scatter, multi-pid, broadcast), make the deviation
explicit in a comment so the next reader knows why.

## Active-only specs

Kernel correctness theorems use `ComputeCorrect.WriteMap.writeIf` to gate
on the active mask. The `*Spec` is the **active-lane expected value** —
inactive lanes go through the `writeIf`-preserve path and never observe
the spec. This means the spec can be defined at all `Fin BLOCK_N` indices
even though only active ones are observed, and writing
`Math.l2Norm (load_with_masked_zero_tail)` is fine for the inactive lanes
since the writeIf eats the result.

Do **not** add a spec-side `if active then ... else preserve` clause to
mimic the writeIf — let `writeIf` do the masking, keep the spec pure.

## See also

- [`CodeOrganization.md`](/VeriTile/architecture/code-organization/) — three-layer structure
  (Math / Semantics / per-kernel glue)
- [`CorrectnessSurfaces.md`](/VeriTile/proofs/correctness-surfaces/) — the user-facing
  theorem surfaces (`Realizes`, `WriteMap`, `OutputReadable`)
