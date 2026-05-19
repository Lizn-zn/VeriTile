---
title: forLoop_inv pitfalls
description: Seven tactical traps that bite when proving kernels with forLoop_inv or forRange_inv.
---

Discovered the hard way while closing the first non-trivial `forLoop_inv`
uses (most painfully during `online_welford_correct`). The simp/unfold
machinery around the `mutual` block in `Triton.Semantics` is brittle in
specific ways. Subagents will rediscover these unless warned.

Apply these as known hazards whenever you dispatch a kernel proof that
wraps a loop with pre/post statements.

## 1. `forLoop_unfold` is too eager for outer `simp only`

A naïve

```lean
simp only [exec, kernel, stepStmts, stepStmt, evalOp, ...]
```

will unfold the `forLoop` arm into `stepForLoopAux` form, after which
`forLoop_inv`'s conclusion (which is in `stepStmt (.forLoop ...)` form)
no longer matches for substitution.

**Workaround:** use private helpers `stmts_cons` / `stmts_nil` proven via

```lean
conv_lhs => unfold stepStmts; rw [h]
```

and chain them statement-by-statement through pre/post. A plain
`unfold stepStmts; rw [h]` leaves a half-reduced match because the
recursive `stepStmts` on the `some` branch also unfolds.

## 2. `stepStmts [] s = some s` is not `rfl`

The mutual block's equation compiler doesn't make this definitionally
equal. `unfold stepStmts` is needed to discharge the empty-tail base case.

## 3. `if_neg h_mv` doesn't auto-fire inside an `∧`-conditional

When `BlockState.writeMem` unfolds, the `if` condition is

```text
region = region ∧ off = off
```

Plain `simp [if_neg h_mv]` can fail to rewrite the conjunction.

**Workaround:** pass `(fun h : meanReg = varReg => h_mv h)` as a simp
lemma so `simp` eliminates the first conjunct directly.

## 4. Value-channel match cascades make per-iteration bodies brittle

Each statement extracts a register via a `match` on `s.regs name`. To step
through (e.g.) five statements in the Welford body required carrying
parallel `have h{name}{i}` facts at every depth, plus explicit
`BlockState.setReg` unfolds.

**Plan budget:** about five `simp` blocks per body statement — not one
global `simp`. The staged "explicit `setReg` chain + per-step `have`"
pattern compiles where one big simp loops or fails.

## 5. Bool-mask vs Prop-if mismatch

When kernel correctness uses masked load/store, `simp` rewrites

```text
if (decide P : Bool) then ...
```

to

```text
if (P : Prop) then ...
```

via `decide_eq_true_eq` automatically. Lemmas stated with a Bool mask —
e.g. `BlockState.scatter_readback_masked` — then no longer match the
post-simp goal.

**Workaround:** also provide a Prop-flavoured corollary
(`scatter_readback_prop_masked`) that takes
`P : Fin n → Prop` with `[DecidablePred P]` and converts Bool→Prop
via `simp only [decide_eq_true_eq]` internally. The bench masked-add
proof was originally stuck on this.

## 6. `List.Disjoint` is the 4-argument form, not the 2-argument one

Mathlib's current `List.Disjoint l₁ l₂` unfolds to

```text
∀ a ∈ l₁, ∀ b ∈ l₂, a ≠ b
```

— the explicit 4-arg form, not the older `∀ a ∈ l₁, a ∉ l₂` shape.

To extract a contradiction from `hl1_disj : Disjoint l₁ (i :: l₂)` and
`hk : i ∈ l₁`, the call is:

```lean
(hl1_disj i hk i List.mem_cons_self) rfl
```

The second `i` and `mem_cons_self` are the `b` and its membership; the
resulting `i ≠ i` contradicts `rfl`.

## 7. `subst` direction surprise

When you have `hki : k = i` with `i` a theorem parameter and `k` an
`intro`'d local, `subst hki` substitutes `i` with `k` — eliminating the
**parameter**, not the local. Downstream references to `i` break.

**Workaround:** use

```lean
rw [hki] at <hyp>
```

to replace `k → i` in just the hypothesis you want, leaving the rest of
the context intact.

---

When dispatching a sub-agent on a `forLoop_inv` kernel proof, include
these as known hazards in the prompt. The Welford proof closed in +297
lines once these patterns were in place; naïve simp-driven attempts loop
or fail.
