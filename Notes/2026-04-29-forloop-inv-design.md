# `forLoop_inv` Interface Design

**Status:** Spec, approved 2026-04-29.
**Phase:** B (Tier 2 streaming reductions).
**Files affected:** `VeriTile/Triton/Semantics.lean`, new `VeriTile/Triton/LoopInvariant.lean`, `VeriTile.lean`.

This spec fixes the operational semantics of `Stmt.forLoop` in VeriTile's
embedded Triton subset and the shape of the loop-induction lemma family that
all of Phase B (welford / online softmax / layernorm) and Phase C (FA-1 forward,
single program-id) will rely on. It supersedes the draft signature sketched
in `PLAN.md:83–94`.

## 1. Goal

Provide a reusable loop-induction lemma `forLoop_inv` such that every Tier 2 /
Tier 3-A kernel-pair proof can close its loop by:

1. Picking an invariant `P : Nat → BlockState → Prop` describing the running
   state of the kernel's accumulators after `k` iterations.
2. Discharging `P 0 s_init` (entry).
3. Discharging the step obligation: each body iteration preserves `P` and
   returns `some` (i.e. does not raise a runtime error).
4. Concluding `P n s_final` for some `s_final` that the operational semantics
   produces.

The lemma must compose under nested `forLoop` (FA-1 forward has Q-block ×
KV-block, so Phase C exercises 2-deep nesting).

## 2. Carrier choice (decided)

`P : Nat → BlockState → Prop`.

The invariant talks about the *whole* `BlockState`, not an abstract logical
state `S`. Down-stream users obtain register values from `BlockState` via
`s.regs name` plus `Value.asScalar` / `Value.tile` projections, and any
boilerplate that arises is absorbed by simp lemmas, not by introducing a
parallel abstraction layer.

Rationale: matches the existing P1 style (`BlockState.scatter_readback` is also
`BlockState`-level + simp-driven), avoids the need to maintain abstraction /
concretization round-trip lemmas, and stays directly compatible with the
nested-loop case (the inner `forLoop_inv` invocation just continues to talk
about the same `BlockState`).

## 3. Operational semantics

### 3.1 `mutual` block

`stepStmt` and `stepStmts` move into a `mutual` block together with a new
`stepForLoopAux`:

```lean
mutual
  noncomputable def stepStmt : Stmt → BlockState → Option BlockState
    | .assign name e, s        => -- unchanged
    | .store region off val, s => -- unchanged
    | .forLoop idx n body, s   => stepForLoopAux idx 0 n body s

  noncomputable def stepStmts : List Stmt → BlockState → Option BlockState
    | [], s         => some s
    | st :: rest, s => (stepStmt st s).bind (stepStmts rest)

  noncomputable def stepForLoopAux
      (idx : RegName) (i n : Nat) (body : List Stmt) :
      BlockState → Option BlockState
    | s =>
        if i < n then
          (stepStmts body (s.setReg idx (Value.scalarNat i))).bind
            (stepForLoopAux idx (i+1) n body)
        else some s
end
termination_by
  stepStmt        st _           => (sizeOf st, 0, 0)
  stepStmts       l  _           => (sizeOf l, 0, 0)
  stepForLoopAux  _ i n body _   => (sizeOf body + 1, n - i, 0)
decreasing_by
  -- per-case ordering: stepStmt → stepStmts (body of forLoop body) decreases
  -- on the first component when stmt is forLoop, decreases via list-shrink on
  -- non-forLoop stepStmt → stepStmt; stepForLoopAux self-recursion decreases
  -- on the lex pair (sizeOf body + 1) ↘ unchanged, (n - i) ↘ strict.
  -- Concrete tactic: `decreasing_tactic` for each goal; if it fails on the
  -- aux-self-recursion case, fall back to `simp_wf; omega`.
```

(The `decreasing_by` block above is a sketch; the actual tactic invocation will
be finalized during implementation. The lex pair is what `PLAN.md` already
prescribes.)

### 3.2 Sub-decisions for the semantics

| Decision | Value | Note |
|---|---|---|
| `idx` value channel | `Value.scalarNat i` | Not `Value.scalar (i : ℝ)` as `PLAN.md:88` sketched. The body's offset arithmetic (`pid * BLOCK + idx * STRIDE`) is in the `Nat` channel, so `scalarNat` is the only consistent choice. |
| `idx` post-loop visibility | retained at last-iteration value | Python-like. No scope/shadow mechanism; user is responsible for not aliasing register names across nested loops. |
| `n = 0` behaviour | identity (`stepForLoopAux idx 0 0 body s = some s`) | Natural base case of the `if i < n` guard. |
| Termination measure | `(sizeOf body + 1, n − i)` lex | Per `PLAN.md`. |
| Nested loop name collision | user-guaranteed disjoint `RegName`s | Outer loop `"i"`, inner loop `"j"` etc. — explicitly *not* a scope mechanism. |

## 4. Lemma family

Lives in a new file `VeriTile/Triton/LoopInvariant.lean`. The family has one
master lemma plus a small set of ergonomics corollaries. Corollaries are added
on demand; only the two below are required for Phase B.

### 4.1 Master lemma `forLoop_inv` (Form 1)

```lean
theorem forLoop_inv
    {idx : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx (.scalarNat i)) = some s' ∧
          P (i+1) s') :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      P n s_final
```

Proof outline:

- Generalize to `forLoopAux_inv : ∀ i ≤ n, P i s → ∃ s_final, stepForLoopAux idx i n body s = some s_final ∧ P n s_final`.
- Strong induction on `n − i` (or equivalently `Nat.le_induction` from `i = n`).
- Base `i = n`: `stepForLoopAux idx n n body s = some s` directly.
- Step `i < n`: invoke `h_step i s` to get `s'` with `stepStmts body (s.setReg idx ...) = some s'` and `P (i+1) s'`; combine with the inductive hypothesis at `i+1`.
- Finally `forLoop_inv` is the `i = 0` instance, with `stepStmt (.forLoop ...) s_init = stepForLoopAux idx 0 n body s_init` by definition.

### 4.2 Readout corollary `forLoop_readout_scalar`

```lean
theorem forLoop_readout_scalar
    {idx outReg : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {f : Nat → ℝ}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx (.scalarNat i)) = some s' ∧
          P (i+1) s')
    (h_readout : ∀ s, P n s → s.regs outReg = some (.scalar (f n))) :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      s_final.regs outReg = some (.scalar (f n))
```

For online softmax / layernorm: `outReg` is `"m"`, `"l"`, `"μ"`, `"var"`, etc.;
`f n` is the closed-form value the kernel computes after `n` iterations.

### 4.3 Readout corollary `forLoop_readout_tile`

```lean
theorem forLoop_readout_tile
    {idx outReg : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {len : Nat} {f : Nat → Fin len → ℝ}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx (.scalarNat i)) = some s' ∧
          P (i+1) s')
    (h_readout : ∀ s, P n s → s.regs outReg = some (.tile len (f n))) :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      s_final.regs outReg = some (.tile len (f n))
```

For FA-1 forward: `outReg = "O"`, `len = D`, `f n d = O_n d`.

### 4.4 Deferred corollaries

`forLoop_readout_nat`, `forLoop_readout_tileNat`, `forLoop_readout_two`
(read two registers in one go, e.g. `(m, l)` for online softmax) — added when
the first concrete need arises during Tier 2 implementation. Don't pre-write
them; YAGNI.

### 4.5 Why no `InitInvariant` baked in

An earlier draft burned an `InitInvariant s_init` hypothesis into
`forLoop_readout_scalar` so the user could pass `h_init_state : InitInvariant s_init`
and have `P 0 s_init` derived inside. We rejected this: corollaries should stay
parametric in `P`'s entry condition. Callers that have an `InitInvariant`
predicate in their context just discharge `P 0 s_init` upfront with one local
`have`:

```lean
have h0 : P 0 s_init := h_init_state.toP0
forLoop_readout_scalar h0 h_step h_readout
```

This keeps the corollary reusable across kernels with very different setup
predicates.

## 5. File layout

| File | Status | Responsibility |
|---|---|---|
| `VeriTile/Triton/Semantics.lean` | Modify | Convert `stepStmt` / `stepStmts` to `mutual` block; add `stepForLoopAux`; replace forLoop's `none` placeholder; add `termination_by`. |
| `VeriTile/Triton/LoopInvariant.lean` | Create | `forLoop_inv` master lemma; `forLoop_readout_scalar` and `forLoop_readout_tile` corollaries. |
| `VeriTile.lean` | Modify | Add `import VeriTile.Triton.LoopInvariant`. |

No changes to `Triton/Core.lean` (the `Stmt.forLoop` constructor is already
defined; only its semantics was a placeholder).

## 6. Validation

- `lake build` clean after each file change.
- The existing P1 `simp` walkthroughs in `Examples/SoftmaxEq.lean`,
  `LogSumExpEq.lean`, etc. continue to typecheck (no regressions from the
  `mutual` block changing reduction behaviour).
- A small `example` block in `LoopInvariant.lean` that exercises a 1-line body
  forLoop (e.g. accumulator `acc := acc + idx`) and closes with `forLoop_inv`
  using `P k s := s.regs "acc" = some (.scalar (∑ i ∈ Finset.range k, (i:ℝ)))`.

## 7. Risks (carried from `PLAN.md` Phase B risk register)

| Risk | Mitigation |
|---|---|
| `mutual` block + `decreasing_by` cascade fails to elaborate cleanly | Fall back to a `Nat.iterate`-style `stepForLoop` defined via `Nat.rec`, decoupled from the structural recursion of `stepStmt`. Minor proof-engineering cost; semantics unchanged. |
| `simp` becomes brittle around `stepStmt` after `mutual` change | Add `@[simp]` reduction lemmas exposing the `forLoop`/non-forLoop cases as separate equations; this is local engineering, not interface-affecting. |
| Step hypothesis ergonomics: `∃ s'` form awkward when body uses `.bind` chains | Provide a small simp/`Option.bind` helper if it shows up in 3+ kernel proofs; otherwise leave as the user's responsibility. |
| Down-stream proofs need `forLoop_readout_two` (read 2 registers at once) before Tier 2 closes | Add the corollary on first concrete need, mirroring `forLoop_readout_scalar`'s shape. |

## 8. Out of scope (this spec)

- Multi-program-id grid coordination (Phase D).
- `tl.dot` / 2D tile semantics (Phase C).
- Mask handling (Phase C).
- Backward-pass forLoop (P3+).
- Any change to `Stmt.forLoop`'s constructor signature in `Core.lean`.
