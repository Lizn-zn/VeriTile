# Phase B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `forLoop` operational semantics + the `forLoop_inv` lemma family to VeriTile and use them to close three Tier 2 kernel-pair theorems (Welford, online-softmax recurrence — *paper centerpiece* — and LayerNorm), plus a non-gating diff-test artifact and the Phase B LLM-benchmark report.

**Architecture:** Three concentric layers. Layer 1 — operational-semantics + DSL extensions (`mutual` block in `Triton/Semantics.lean` + `tl.for ... { ... }` syntax in `Triton/DSL.lean`). Layer 2 — reusable proof tooling (`VeriTile/Triton/LoopInvariant.lean`: master lemma `forLoop_inv` + readout corollaries). Layer 3 — three kernel-pair theorems following the existing `Examples/SoftmaxEq.lean` template (kernel × 2, spec function × 2, correctness lemma × 2, refinement theorem). Plus an artifact track (`tools/diff_test/`) and a benchmark track (`bench/llm_eval/phase_b/`).

**Tech Stack:** Lean 4 (v4.15.0+, Mathlib), `lake build`, the existing `lean4` Claude Code plugin (`/lean4:autoprove`), Python 3 + Triton + PyTorch (for diff-test only), bash + `jq` (for benchmark scripts).

---

## Scope reference

This plan implements `PLAN.md` §Phase B. It is anchored on the spec
`Notes/2026-04-29-forloop-inv-design.md` (commit d338502) for the operational
semantics and lemma-family interface decisions.

Phase B deliverables:
- `forLoop` operational semantics + `forLoop_inv` lemma family (Workstreams 1, 2)
- 3 Tier 2 kernel-pair theorems: #4 Welford, #5 online-softmax recurrence, #6 LayerNorm (Workstreams 4, 5, 6)
- Diff-test artifact for 1–2 of the 6 closed Tier 1+2 kernels (Workstream 7)
- Phase B LLM benchmark report (Workstreams 3, 8)
- Exit tag `v0.2-tier2` (Workstream 9)

Already complete at start of Phase B (do **not** re-do):
- Tier 1 (#1, #2, #3) closed in `Examples/SoftmaxEq.lean`, `Examples/LogSumExpEq.lean`, `Examples/SoftmaxReciprocal.lean`; main `lake build` clean.
- `welford_eq_two_pass` math lemma closed at `VeriTile/Examples/WelfordMath.lean:204`.
- `scripts/prove.sh`, `bench/llm_eval/`, `Notes/T3_scouting.md`, spec `Notes/2026-04-29-forloop-inv-design.md`.

Out of scope (per `PLAN.md` §Out of scope):
- Phase C work (FA-1 forward, `tl.dot`, masking, 2D layout)
- Phase D (FA-2, multi-block grid, layout)
- Backward pass, IEEE-754, Python lifter

## File structure

**Lean (extends existing library):**

| File | Status | Responsibility |
|---|---|---|
| `VeriTile/Triton/Semantics.lean` | Modify | Convert `stepStmt`/`stepStmts` to a `mutual` block; add `stepForLoopAux`; replace `forLoop` `none` placeholder; lex `termination_by`. |
| `VeriTile/Triton/DSL.lean` | Modify | Add `tl.for $i in $n { … }` syntax + macro expansion to `Stmt.forLoop`. |
| `VeriTile/Triton/LoopInvariant.lean` | Create | `forLoop_inv` master lemma; `forLoop_readout_scalar`, `forLoop_readout_tile` corollaries; sanity example. |
| `VeriTile/Examples/WelfordKernels.lean` | Create | Welford forLoop kernel + two-pass kernel + `welford_kernels_refinement` (#4). |
| `VeriTile/Examples/OnlineSoftmax.lean` | Create | Online softmax recurrence kernel + batch-softmax kernel + `online_softmax_recurrence_eq_batch` (#5; paper centerpiece). |
| `VeriTile/Examples/LayerNormKernels.lean` | Create | Fused (single-pass) LayerNorm kernel + two-pass LayerNorm kernel + `layernorm_kernels_refinement` (#6). |
| `VeriTile.lean` | Modify | Add four new imports (LoopInvariant + 3 Examples files). |

**Held-out evaluation:**

| File | Status | Responsibility |
|---|---|---|
| `bench/llm_eval/phase_b/README.md` | Create | Held-out set documentation (lemmas, allowed context, eval protocol pointer). |
| `bench/llm_eval/phase_b/welford_*.lean` | Create | 1–2 sub-lemmas of #4 with proof body replaced by `sorry`. |
| `bench/llm_eval/phase_b/online_softmax_*.lean` | Create | 2 sub-lemmas of #5 with `sorry`. |
| `bench/llm_eval/phase_b/layernorm_*.lean` | Create | 1–2 sub-lemmas of #6 with `sorry`. |
| `bench/llm_eval/results/phase_b_report.md` | Create | Full benchmark eval report after the 5-trial run. |

**Differential testing artifact (non-gating):**

| File | Status | Responsibility |
|---|---|---|
| `tools/diff_test/README.md` | Create | Side-by-side correspondence convention; tolerance policy; failure-handling. |
| `tools/diff_test/python/welford.py` | Create | (One of the two reps; Phase A picked Welford as canonical Tier 2 representative.) Hand-written Triton + PyTorch reference. |
| `tools/diff_test/python/online_softmax.py` | Create | (Optional second rep. If chosen, hand-written Triton + PyTorch reference.) |
| `tools/diff_test/results/phase_b_report.md` | Create | One-shot numeric agreement report. |

**Meta:**

| File | Status | Responsibility |
|---|---|---|
| `PLAN.md` | Modify | At exit, mark Phase B complete; record `v0.2-tier2`. |

---

## Workstream 1 — `forLoop` operational semantics + DSL (~1 week)

This workstream lifts `Stmt.forLoop` from the `none` placeholder at
`VeriTile/Triton/Semantics.lean:353–363` to a fully evaluating semantics, and
adds DSL surface syntax so kernels can be written as `tl.for i in $(N) { ... }`
rather than raw `Stmt.forLoop "i" N [...]`.

### Task 1.0: Sanity-check baseline build

**Files:**
- Read: `VeriTile/Triton/Semantics.lean:317-371` (current `stepStmt` / `stepStmts`)
- Read: `VeriTile/Triton/DSL.lean:104` (current `tritonStmt` syntax — only `assign` and `tl.store`)

- [ ] **Step 1: Verify clean baseline**

Run:
```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: `Build completed successfully (NNNN jobs).` No warnings.

- [ ] **Step 2: Confirm forLoop is a placeholder**

Run:
```bash
grep -nA 12 "forLoop _idx _n _body" VeriTile/Triton/Semantics.lean
```

Expected: shows the `TODO(P2)` placeholder returning `none`.

- [ ] **Step 3: No commit (read-only verification)**

### Task 1.1: Convert `stepStmt`/`stepStmts` to a `mutual` block (semantics-preserving refactor)

This task does **not** yet implement `forLoop` — it just restructures the
existing definitions into a `mutual` block so a third member (`stepForLoopAux`)
can be added in Task 1.2 without later disruption. We confirm semantics is
unchanged by re-running `lake build`.

**Files:**
- Modify: `VeriTile/Triton/Semantics.lean:317-371`

- [ ] **Step 1: Read the current `stepStmt` and `stepStmts` for exact bytes**

```bash
sed -n '317,371p' VeriTile/Triton/Semantics.lean
```

- [ ] **Step 2: Wrap both in `mutual`**

Replace lines 317–371 of `VeriTile/Triton/Semantics.lean` with:

```lean
mutual

-- Execute one statement.
--
-- `assign` evaluates RHS and stores the value in the named register.
-- `store` writes a scalar or contiguous tile to memory.
-- `forLoop` is implemented in Task 1.2; for now keep the `none` placeholder
-- so this refactor is semantics-preserving.
noncomputable def stepStmt : Stmt → BlockState → Option BlockState
  | .assign name e, s =>
      match evalOp e s with
      | some v => some (s.setReg name v)
      | none   => none
  | .store region off val, s =>
      match evalOp off s, evalOp val s with
      | some voff, some vval =>
          match voff with
          | .scalarNat coff =>
              match vval with
              | .scalar c =>
                  some (s.writeMem region coff c)
              | .tile n f =>
                  some ((List.finRange n).foldl
                          (fun acc i =>
                            acc.writeMem region (coff + i.val) (f i))
                          s)
              | _ => none
          | .tileNat n offs =>
              match vval with
              | .tile m vals =>
                  if h : n = m then
                    some ((List.finRange n).foldl
                            (fun acc i =>
                              acc.writeMem region (offs i)
                                          (vals (Fin.cast h i)))
                            s)
                  else none
              | _ => none
          | _ => none
      | _, _ => none
  | .forLoop _idx _n _body, _s =>
      none  -- Task 1.2 will implement this.

/-- Sequence a list of statements, threading state. -/
noncomputable def stepStmts : List Stmt → BlockState → Option BlockState
  | [], s => some s
  | st :: rest, s =>
      match stepStmt st s with
      | some s' => stepStmts rest s'
      | none    => none

end
```

(The file already has the block-level docstring above this region; preserve it.)

- [ ] **Step 3: Build**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -5
```

Expected: `Build completed successfully`. Lean will choose
`termination_by` automatically because the recursion is structural on `Stmt`
/ `List Stmt` (only the placeholder forLoop case has been added — it doesn't
recurse).

If Lean complains about termination, add an explicit minimal lex measure:

```lean
termination_by
  stepStmt s _   => sizeOf s
  stepStmts l _  => sizeOf l
```

- [ ] **Step 4: Spot-check existing kernel proofs still elaborate**

Run a focused build of one example file to surface any reduction-behaviour
regressions early:

```bash
PATH="$HOME/.elan/bin:$PATH" lake env lean VeriTile/Examples/SoftmaxEq.lean 2>&1 | tail -5
```

Expected: no errors. (This file has the most simp-on-stepStmts pattern, so
any subtle change in unfolding behaviour would surface here first.)

- [ ] **Step 5: Commit**

```bash
git add VeriTile/Triton/Semantics.lean
git commit -m "refactor(semantics): wrap stepStmt/stepStmts in mutual block (no semantics change)"
```

### Task 1.2: Implement `stepForLoopAux` and complete the `forLoop` case

**Files:**
- Modify: `VeriTile/Triton/Semantics.lean` (the `mutual` block from Task 1.1)

- [ ] **Step 1: Add `stepForLoopAux` and route forLoop to it**

Inside the `mutual` block, replace the `forLoop` placeholder line and add a
third definition:

```lean
  | .forLoop idx n body, s =>
      stepForLoopAux idx 0 n body s

/-- The bounded-for-loop driver. Iterates `i` from `start` up to `n`,
    binding the loop counter to register `idx` (as a `Nat` value) before each
    body run. Returns `none` if any body iteration returns `none`. -/
noncomputable def stepForLoopAux
    (idx : RegName) (start n : Nat) (body : List Stmt) :
    BlockState → Option BlockState
  | s =>
      if start < n then
        match stepStmts body (s.setReg idx (Value.scalarNat start)) with
        | some s' => stepForLoopAux idx (start + 1) n body s'
        | none    => none
      else some s
```

(`idx` is bound through `Value.scalarNat`, **not** `Value.scalar (i:ℝ)`, per
spec §3.2 — the body's offset arithmetic is in the `Nat` channel.)

- [ ] **Step 2: Add `termination_by` for the whole `mutual` block**

Immediately after the `end` closing the `mutual` block, add:

```lean
termination_by
  stepStmt        st _              => (sizeOf st, 0)
  stepStmts       l  _              => (sizeOf l,  0)
  stepForLoopAux  _ start n body _  => (sizeOf body + 1, n - start)
```

`stepForLoopAux` calls `stepStmts body ...` (decreases first component:
`sizeOf body < sizeOf body + 1`) and `stepForLoopAux idx (start+1) n body s'`
(first component unchanged, second strictly decreases when `start < n`).
`stepStmt`'s `forLoop` case calls `stepForLoopAux idx 0 n body s` — first
component decreases (`sizeOf body + 1 < sizeOf (Stmt.forLoop _ _ body)`
because `Stmt.forLoop`'s sizeOf = `1 + 0 + 0 + sizeOf body` (constructor head
+ idx/n which are non-recursive + body); confirm via `#eval` if needed).

- [ ] **Step 3: Build; if termination fails, supply `decreasing_by`**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10
```

If Lean cannot derive termination automatically, add:

```lean
decreasing_by
  all_goals first
    | (simp_wf; omega)
    | decreasing_tactic
```

If *that* still fails (the lex pair on `stepForLoopAux` self-recursion is the
likely culprit), use the explicit form:

```lean
decreasing_by
  · -- stepStmt forLoop -> stepForLoopAux: first component drops
    simp_wf; omega
  · -- stepStmts cons -> stepStmt: first component drops
    simp_wf; omega
  · -- stepStmts cons -> stepStmts (rest): first component drops
    simp_wf; omega
  · -- stepForLoopAux self -> stepStmts body: first component drops
    simp_wf; omega
  · -- stepForLoopAux self -> self with start+1: second drops, first unchanged
    simp_wf; omega
```

- [ ] **Step 4: Spot-check existing `Sanity checks` block still elaborates**

`Triton/Semantics.lean` already has a `Sanity checks` block at the bottom
(starting around the line `-- ────────────── Sanity checks ──────────────`).
After the `mutual` block change, those `example`s must still elaborate. Run
the focused build:

```bash
PATH="$HOME/.elan/bin:$PATH" lake env lean VeriTile/Triton/Semantics.lean 2>&1 | tail -10
```

Expected: no errors. The existing examples don't use `forLoop`, so they
exercise only the structural cases of `stepStmt`/`stepStmts`. The new
forLoop case will be exercised by the Task 2.5 sanity example in
`LoopInvariant.lean` — we do not duplicate the check here.

- [ ] **Step 5: If termination cascade keeps fighting, fall back to `Nat.iterate` form**

Per spec §7 risk: replace `stepForLoopAux` with a non-recursive driver:

```lean
noncomputable def stepForLoopAux
    (idx : RegName) (n : Nat) (body : List Stmt) (s : BlockState) :
    Option BlockState :=
  Nat.rec (motive := fun _ => Option BlockState)
    (some s)
    (fun i acc =>
      acc.bind (fun s' =>
        stepStmts body (s'.setReg idx (Value.scalarNat i))))
    n
```

This avoids self-recursion entirely; only `stepStmts body` calls into
`stepStmt`/`stepStmts`, which is structural. Termination is automatic.

(The lex form is preferred for `simp` ergonomics; only fall back if Step 3 +
the `decreasing_by` cascade above cannot close termination after ~1 day of
trying.)

- [ ] **Step 6: Final build clean**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add VeriTile/Triton/Semantics.lean
git commit -m "feat(semantics): forLoop operational semantics via stepForLoopAux"
```

### Task 1.3: DSL syntax `tl.for i in $(n) { … }`

Lets us write Tier 2 kernels in `triton { ... }` form rather than raw
`Stmt.forLoop` constructor calls.

**Files:**
- Modify: `VeriTile/Triton/DSL.lean` (extend `tritonStmt` syntax, `expandStmt`,
  and `stmtRegions`).

- [ ] **Step 1: Read current DSL syntax declarations**

```bash
sed -n '70,115p' VeriTile/Triton/DSL.lean
```

- [ ] **Step 2: Add `tritonStmt` syntax for `tl.for`**

After the `tl.store(...)` syntax declaration (around line 105), add:

```lean
-- `tl.for i in $(n) { stmt* }` — bounded loop.
syntax "tl.for " ident " in " "$(" term ")" " { " tritonStmt* " }" : tritonStmt
-- Convenience form with a numeric literal.
syntax "tl.for " ident " in " num " { " tritonStmt* " }" : tritonStmt
```

- [ ] **Step 3: Add `expandStmt` cases**

Inside `partial def expandStmt` (around line 211), add **before** the existing
catch-all match cases:

```lean
  | `(tritonStmt| tl.for $i:ident in $($n:term) { $stmts:tritonStmt* }) => do
      let body ← stmts.toList.mapM expandStmt
      let bodyList ← `([$body,*])
      `(Stmt.forLoop $(Lean.quote i.getId.toString) $n $bodyList)
  | `(tritonStmt| tl.for $i:ident in $n:num { $stmts:tritonStmt* }) => do
      let body ← stmts.toList.mapM expandStmt
      let bodyList ← `([$body,*])
      `(Stmt.forLoop $(Lean.quote i.getId.toString) $n $bodyList)
```

(Adjust the helper-name `Lean.quote` for ident-name lifting if the existing
DSL uses a different idiom. Check `expandStmt`'s assign case at lines 213–215
for the project convention; copy that style.)

- [ ] **Step 4: Add `stmtRegions` cases (region scanning)**

Around line 277, in `stmtRegions`, mirror the new syntax — for `tl.for`,
recursively walk the inner statements and union their region sets:

```lean
  | `(tritonStmt| tl.for $_:ident in $($_:term) { $stmts:tritonStmt* }) =>
      stmts.toList.foldl
        (fun (acc : List (TSyntax `term) × List (TSyntax `term)) st =>
          let (i, o) := stmtRegions st
          (acc.1 ++ i, acc.2 ++ o)) ([], [])
  | `(tritonStmt| tl.for $_:ident in $_:num { $stmts:tritonStmt* }) =>
      stmts.toList.foldl
        (fun (acc : List (TSyntax `term) × List (TSyntax `term)) st =>
          let (i, o) := stmtRegions st
          (acc.1 ++ i, acc.2 ++ o)) ([], [])
```

- [ ] **Step 5: Smoke-test the DSL**

Add a temporary `example` at the bottom of `DSL.lean`:

```lean
example : Stmt :=
  triton {
    tl.for i in 5 {
      acc := acc + i
    }
  } |>.body.head!  -- hack: pull out the first stmt
```

(or, more cleanly, write it as a full kernel and inspect with `#check`.)

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -5
```

Expected: builds. If macro fails, debug with `#check` and `#print` until the
expansion matches the intended `Stmt.forLoop "i" 5 [...]`.

- [ ] **Step 6: Remove the smoke `example`; final build**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add VeriTile/Triton/DSL.lean
git commit -m "feat(dsl): tl.for i in $(n) { ... } syntax for forLoop"
```

### Task 1.4: Add `@[simp]` reduction lemmas for `stepForLoopAux`

Spec §7 risk: simp brittleness around the new `mutual` block. Provide a small
set of named simp lemmas that callers can use instead of unfolding
`stepForLoopAux` manually.

**Files:**
- Modify: `VeriTile/Triton/Semantics.lean` (add named lemmas at the end of
  the file, before the existing `Sanity checks` section)

- [ ] **Step 1: Add the base-case and step-case simp lemmas**

```lean
namespace stepForLoopAux  -- group under a namespace for discoverability

@[simp] theorem step_eq_self {idx} {n} {body} (s) :
    stepForLoopAux idx n n body s = some s := by
  unfold stepForLoopAux
  simp

@[simp] theorem step_lt {idx} {start n} {body} {s} (h : start < n) :
    stepForLoopAux idx start n body s
      = (stepStmts body (s.setReg idx (Value.scalarNat start))).bind
          (stepForLoopAux idx (start + 1) n body) := by
  unfold stepForLoopAux
  simp [h]
  -- The body is a `match`; rewrite into `Option.bind` form.
  cases hbody : stepStmts body (s.setReg idx (Value.scalarNat start)) <;> rfl

@[simp] theorem forLoop_unfold {idx} {n} {body} {s} :
    stepStmt (.forLoop idx n body) s
      = stepForLoopAux idx 0 n body s := by
  rfl

end stepForLoopAux
```

- [ ] **Step 2: Build**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add VeriTile/Triton/Semantics.lean
git commit -m "feat(semantics): named simp lemmas for stepForLoopAux"
```

---

## Workstream 2 — `forLoop_inv` lemma family (~1 week)

Implements the master lemma + two corollaries from spec §4.1–4.3.

### Task 2.1: Create `LoopInvariant.lean` with imports + namespace

**Files:**
- Create: `VeriTile/Triton/LoopInvariant.lean`

- [ ] **Step 1: Write the file skeleton**

```lean
/-
VeriTile.Triton.LoopInvariant

The forLoop induction lemma family. Master lemma `forLoop_inv` (spec §4.1)
plus ergonomics corollaries `forLoop_readout_scalar` (§4.2) and
`forLoop_readout_tile` (§4.3). See Notes/2026-04-29-forloop-inv-design.md
for the full interface design.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

-- TODO: forLoopAux_inv (Task 2.2)
-- TODO: forLoop_inv (Task 2.3)
-- TODO: forLoop_readout_scalar (Task 2.4)
-- TODO: forLoop_readout_tile (Task 2.4)
-- TODO: sanity example (Task 2.5)

end VeriTile.Triton
```

- [ ] **Step 2: Add to `VeriTile.lean`**

Insert (preserving alphabetical-ish order with siblings):

```lean
import VeriTile.Triton.LoopInvariant
```

after the existing `import VeriTile.Triton.Examples` line.

- [ ] **Step 3: Build**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean (the file is empty of definitions, just has TODO comments).

- [ ] **Step 4: Commit**

```bash
git add VeriTile/Triton/LoopInvariant.lean VeriTile.lean
git commit -m "scaffold(loop-inv): create LoopInvariant.lean skeleton"
```

### Task 2.2: Prove the auxiliary `forLoopAux_inv`

The state-quantified generalization that does the actual induction. Per spec
§4.1 proof outline.

**Files:**
- Modify: `VeriTile/Triton/LoopInvariant.lean`

- [ ] **Step 1: Write the statement with `sorry`**

Replace `-- TODO: forLoopAux_inv (Task 2.2)` with:

```lean
/-- Auxiliary form of `forLoop_inv` quantified over the starting index `i` and
    state `s`. The master `forLoop_inv` is the `i = 0` instance. -/
theorem forLoopAux_inv
    {idx : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop}
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx (Value.scalarNat i)) = some s' ∧
          P (i + 1) s') :
    ∀ i, i ≤ n → ∀ s, P i s →
      ∃ s_final,
        stepForLoopAux idx i n body s = some s_final ∧ P n s_final := by
  sorry
```

- [ ] **Step 2: Build (expect single `declaration uses sorry` warning)**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -5
```

Expected: builds with one `sorry` warning at `forLoopAux_inv`.

- [ ] **Step 3: Prove by strong induction on `n - i`**

Replace the `sorry` with:

```lean
  -- Induction on `n - i`. Equivalent to `Nat.le_induction` from `i = n`
  -- downward, but easier to phrase here.
  intro i hi s hPs
  -- Pull out the measure once.
  have key : ∀ k i, n - i = k → i ≤ n → ∀ s, P i s →
      ∃ s_final, stepForLoopAux idx i n body s = some s_final ∧ P n s_final := by
    intro k
    induction k with
    | zero =>
        -- n - i = 0 means i ≥ n; combined with i ≤ n we get i = n.
        intro i heq hi_le s hP
        have hi_eq : i = n := by omega
        subst hi_eq
        refine ⟨s, ?_, hP⟩
        simp [stepForLoopAux.step_eq_self]
    | succ k ih =>
        -- n - i = k+1 ⇒ i < n.
        intro i heq hi_le s hP
        have hi_lt : i < n := by omega
        obtain ⟨s', h_body, hP'⟩ := h_step i s hi_lt hP
        have heq' : n - (i + 1) = k := by omega
        have hi'_le : i + 1 ≤ n := hi_lt
        obtain ⟨s_final, h_aux, hP_n⟩ := ih (i + 1) heq' hi'_le s' hP'
        refine ⟨s_final, ?_, hP_n⟩
        rw [stepForLoopAux.step_lt hi_lt]
        rw [h_body]
        simpa using h_aux
  exact key (n - i) i rfl hi s hPs
```

- [ ] **Step 4: Build clean**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean. If a tactic step fails (e.g. `simpa` doesn't close because
`Option.bind` reduction doesn't fire), unfold by hand with
`Option.bind_some` / `Option.bind_none`.

- [ ] **Step 5: Commit**

```bash
git add VeriTile/Triton/LoopInvariant.lean
git commit -m "feat(loop-inv): auxiliary forLoopAux_inv (state-quantified)"
```

### Task 2.3: The master `forLoop_inv` (Form 1)

**Files:**
- Modify: `VeriTile/Triton/LoopInvariant.lean`

- [ ] **Step 1: Add `forLoop_inv` deriving from `forLoopAux_inv`**

Replace `-- TODO: forLoop_inv (Task 2.3)` with:

```lean
/-- **Master loop-induction lemma.** For any `forLoop` of body length `n`
    over register `idx`, given an entry invariant `P 0 s_init` and a step
    obligation showing each iteration preserves `P` (and does not error),
    the final state satisfies `P n`.

    Spec: `Notes/2026-04-29-forloop-inv-design.md` §4.1 (Form 1). -/
theorem forLoop_inv
    {idx : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx (Value.scalarNat i)) = some s' ∧
          P (i + 1) s') :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      P n s_final := by
  obtain ⟨s_final, h_aux, hP⟩ :=
    forLoopAux_inv (P := P) h_step 0 (Nat.zero_le _) s_init h_init
  refine ⟨s_final, ?_, hP⟩
  -- stepStmt (forLoop ...) = stepForLoopAux ... 0 n body s_init by definition.
  simpa [stepForLoopAux.forLoop_unfold] using h_aux
```

- [ ] **Step 2: Build**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add VeriTile/Triton/LoopInvariant.lean
git commit -m "feat(loop-inv): forLoop_inv master lemma (Form 1)"
```

### Task 2.4: Readout corollaries (`scalar` and `tile`)

**Files:**
- Modify: `VeriTile/Triton/LoopInvariant.lean`

- [ ] **Step 1: Add `forLoop_readout_scalar`**

Replace `-- TODO: forLoop_readout_scalar (Task 2.4)` with:

```lean
/-- **Scalar-register readout corollary.** Combines `forLoop_inv` with a
    proof that some output register holds a target scalar value when `P n`
    holds; the conclusion is the standard "kernel correctness reads
    register `outReg` and finds `f n`" form used throughout Tier 2 / 3-A. -/
theorem forLoop_readout_scalar
    {idx outReg : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {f : Nat → ℝ}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx (Value.scalarNat i)) = some s' ∧
          P (i + 1) s')
    (h_readout :
      ∀ s, P n s → s.regs outReg = some (Value.scalar (f n))) :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      s_final.regs outReg = some (Value.scalar (f n)) := by
  obtain ⟨s_final, h_eq, hP⟩ := forLoop_inv h_init h_step
  exact ⟨s_final, h_eq, h_readout _ hP⟩
```

- [ ] **Step 2: Add `forLoop_readout_tile`**

Replace `-- TODO: forLoop_readout_tile (Task 2.4)` with:

```lean
/-- **Tile-register readout corollary.** Same as `forLoop_readout_scalar` but
    for an output register containing a `Value.tile len`; used by FA-1
    forward (`O` register) and any kernel writing a tile-valued accumulator. -/
theorem forLoop_readout_tile
    {idx outReg : RegName} {n : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    {len : Nat} {f : Nat → Fin len → ℝ}
    (h_init : P 0 s_init)
    (h_step :
      ∀ i s, i < n → P i s →
        ∃ s',
          stepStmts body (s.setReg idx (Value.scalarNat i)) = some s' ∧
          P (i + 1) s')
    (h_readout :
      ∀ s, P n s → s.regs outReg = some (Value.tile len (f n))) :
    ∃ s_final,
      stepStmt (.forLoop idx n body) s_init = some s_final ∧
      s_final.regs outReg = some (Value.tile len (f n)) := by
  obtain ⟨s_final, h_eq, hP⟩ := forLoop_inv h_init h_step
  exact ⟨s_final, h_eq, h_readout _ hP⟩
```

- [ ] **Step 3: Build**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add VeriTile/Triton/LoopInvariant.lean
git commit -m "feat(loop-inv): readout corollaries (scalar + tile)"
```

### Task 2.5: Sanity example — Nat-channel counter

Validates the master lemma against a concrete tiny kernel; per spec §6.

**Files:**
- Modify: `VeriTile/Triton/LoopInvariant.lean`

- [ ] **Step 1: Add the example**

Replace `-- TODO: sanity example (Task 2.5)` with:

```lean
/-! ### Sanity check

A trivial 1-statement-body forLoop counter. We use `Nat`-channel arithmetic
because the loop index `idx` is `Value.scalarNat` and `Value.bop` rejects
mixing `ℝ` and `Nat` (per `Notes/2026-04-29-forloop-inv-design.md` §6). -/

private def counterBody : List Stmt :=
  [.assign "cnt" (.add (.ref "cnt") (.constNat 1))]

private theorem counter_invariant_step
    (i : Nat) (s : BlockState)
    (_h_lt : i < 5)
    (hP : s.regs "cnt" = some (Value.scalarNat i)) :
    ∃ s', stepStmts counterBody (s.setReg "i" (Value.scalarNat i)) = some s'
        ∧ s'.regs "cnt" = some (Value.scalarNat (i + 1)) := by
  -- Body is a single assign; step it manually.
  refine ⟨_, ?_, ?_⟩
  · simp [stepStmts, stepStmt, evalOp, counterBody, BlockState.setReg, hP,
          Value.bop]
  · simp [BlockState.setReg]

example
    (s_init : BlockState)
    (h_cnt0 : s_init.regs "cnt" = some (Value.scalarNat 0)) :
    ∃ s_final,
      stepStmt (.forLoop "i" 5 counterBody) s_init = some s_final ∧
      s_final.regs "cnt" = some (Value.scalarNat 5) := by
  -- Note: the `i` register is unset in s_init; that's fine — `setReg "i"`
  -- in stepForLoopAux defines it on first use. The invariant talks about
  -- "cnt" only.
  have h_init : s_init.regs "cnt" = some (Value.scalarNat 0) := h_cnt0
  obtain ⟨s_final, h_eq, hP⟩ :=
    forLoop_inv (P := fun k s => s.regs "cnt" = some (Value.scalarNat k))
      h_init
      (fun i s h_lt hP => counter_invariant_step i s h_lt hP)
  exact ⟨s_final, h_eq, hP⟩
```

- [ ] **Step 2: Build**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10
```

Expected: clean. If `counter_invariant_step` fails to elaborate, debug by
unfolding `setReg` / `Value.bop` more explicitly. The expected reduction
chain is:
```
stepStmts [assign "cnt" (.add (.ref "cnt") (.constNat 1))] s_with_i
  = (after evalOp Op.add) some (s_with_i.setReg "cnt" (Value.scalarNat (i+1)))
```

- [ ] **Step 3: Commit**

```bash
git add VeriTile/Triton/LoopInvariant.lean
git commit -m "test(loop-inv): sanity counter example exercises master lemma"
```

---

## Workstream 3 — Phase B held-out set tagging (~0.5 day, mid-Phase B)

Per `PLAN.md` §LLM benchmark protocol: tag the Phase B held-out set
**at the start of Phase B**, before any LLM-tool prompt iteration for Phase B.
We do this once outlines for #4/#5/#6 are written but before any of them is
proven. Concretely: this task fires *after* Workstream 4 / 5 / 6's first task
(skeleton with `sorry`s) and *before* their proof tasks.

### Task 3.1: Pick 5 candidate sub-lemmas

**Files:**
- Read: skeleton-stage `Examples/WelfordKernels.lean`, `Examples/OnlineSoftmax.lean`,
  `Examples/LayerNormKernels.lean` (after their respective Workstream's first task)
- Create: `bench/llm_eval/phase_b/README.md`

- [ ] **Step 1: Inventory all sub-lemmas appearing as `sorry` in the skeletons**

```bash
grep -nE "^theorem|^private theorem|sorry" \
  VeriTile/Examples/WelfordKernels.lean \
  VeriTile/Examples/OnlineSoftmax.lean \
  VeriTile/Examples/LayerNormKernels.lean \
  > /tmp/phase_b_lemmas.txt
cat /tmp/phase_b_lemmas.txt
```

Expected: a list of theorems each appearing on a line near a `sorry` line.

- [ ] **Step 2: Pick 5 candidates per protocol**

Selection criteria (apply in priority):
1. Diverse difficulty: ~2 "easy" (operational walk-through, similar to
   `softmax_naive_correct` in shape), ~2 "medium" (uses `forLoop_inv`),
   ~1 "hard" (the online-softmax recurrence step or LayerNorm composition).
2. Self-contained: each candidate is provable from already-imported
   lemmas + the target file's preceding declarations only (no peeking at
   the target file's own proof of the candidate).
3. Diverse kernel: at least one candidate from each of #4 / #5 / #6 is
   required.

Suggested defaults:
- **#4-A** `twopass_welford_correct` operational walk-through (easy).
- **#4-B** `online_welford_correct` operational walk-through (medium).
- **#5-A** `online_softmax_recurrence_step` math identity (medium).
- **#5-B** `online_softmax_correct` operational walk-through (hard).
- **#6-A** `layernorm_kernels_refinement` final composition (medium).

Replace defaults with whatever the actual outline names are.

- [ ] **Step 3: Create one held-out file per candidate**

For each candidate, create `bench/llm_eval/phase_b/<lemma>.lean` with the
**statement only** (proof body replaced by `sorry`), and the required
imports + namespace alignment. Template:

```lean
/-
HELD-OUT BENCHMARK FILE — Phase B LLM eval target.

Replicates `<TheoremName>` from `<source-file>` with the proof body
replaced by `sorry`. The MAIN LIBRARY copy stays intact and proven; this
file is an LLM eval target only.

DO NOT add this file to the main `VeriTile.lean` import graph.
DO NOT modify the proof body; the LLM tool runs end-to-end on this file.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.SoftmaxEq          -- for shared spec helpers
import VeriTile.Examples.WelfordMath        -- if applicable
-- (add other imports as the candidate requires; do NOT import the source file
-- itself if doing so would expose the candidate's proof.)

namespace VeriTile.Examples.PhaseBHeldOut

open VeriTile.Triton VeriTile.Examples

/-- Held-out copy of `<TheoremName>` for Phase B LLM benchmark. -/
theorem <TheoremName>_held_out
    <hypothesis list verbatim from source file> :
    <conclusion verbatim from source file> := by
  sorry

end VeriTile.Examples.PhaseBHeldOut
```

- [ ] **Step 4: Verify each held-out file elaborates standalone**

```bash
for f in bench/llm_eval/phase_b/*.lean; do
  echo "=== $f ==="
  PATH="$HOME/.elan/bin:$PATH" lake env lean "$f" 2>&1 | tail -3
done
```

Expected: each prints one `declaration uses sorry` warning, no errors.

- [ ] **Step 5: Verify main library is unaffected**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean (no sorry warnings).

- [ ] **Step 6: Write Phase B held-out README**

Create `bench/llm_eval/phase_b/README.md`:

```markdown
# Phase B held-out set

Tagged 2026-... at the start of Phase B, BEFORE any prompt iteration on Tier
2 lemmas. Per `PLAN.md` §LLM benchmark protocol.

## Set

| File | Source theorem | Difficulty | Notes |
|---|---|---|---|
| welford_twopass_correct.lean | twopass_welford_correct | easy | operational walk |
| welford_online_correct.lean | online_welford_correct | medium | uses forLoop_inv |
| online_softmax_recurrence_step.lean | online_softmax_recurrence_step | medium | math identity |
| online_softmax_correct.lean | online_softmax_correct | hard | operational + forLoop_inv |
| layernorm_refinement.lean | layernorm_kernels_refinement | medium | composition |

(Adjust to match Step 2's actual selection.)

## Allowed context

Per `PLAN.md` §LLM benchmark protocol:
- Imports listed at the top of each held-out file
- Mathlib retrieval (the plugin's built-in)
- The held-out file itself
- **NOT allowed**: importing the source file containing the candidate's actual proof; manual prompt hints specific to a held-out lemma

## Eval protocol

5 trials per held-out lemma, recorded in `bench/llm_eval/results/phase_b_report.md`.
```

- [ ] **Step 7: Commit**

```bash
git add bench/llm_eval/phase_b/
git commit -m "feat(bench): Phase B LLM eval held-out set (5 candidates)"
```

---

## Workstream 4 — Tier 2 #4: `welford_kernels_refinement` (~1.5 weeks)

Welford forLoop kernel ≡ two-pass kernel; lifts the existing
`welford_eq_two_pass` math lemma (`Examples/WelfordMath.lean:204`).

### Task 4.1: Skeleton — kernels + spec functions + correctness statements with `sorry`

**Files:**
- Create: `VeriTile/Examples/WelfordKernels.lean`
- Modify: `VeriTile.lean` (add import)

- [ ] **Step 1: Write kernel definitions and skeleton**

Create `VeriTile/Examples/WelfordKernels.lean`:

```lean
/-
VeriTile.Examples.WelfordKernels

Tier 2 kernel-pair: Welford online recurrence kernel ≡ two-pass variance
kernel. Lifts `welford_eq_two_pass` (Examples/WelfordMath.lean:204) to the
operational level using `forLoop_inv`.

Output spec: each output cell holds (μ, var) packed somehow — for this
kernel-pair the simplest form is two separate output regions `meanReg` and
`varReg`, each storing a single scalar at offset `s.pid`. (Production
LayerNorm uses the same pattern.)
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.SoftmaxEq  -- reuse InputLoaded, observeY, etc.
import VeriTile.Examples.WelfordMath  -- the math identity

namespace VeriTile.Examples

open VeriTile.Triton

/-- Two-pass variance kernel: tl.sum twice, then ((Σx² · n) − (Σx)²)/n². -/
def twopassWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange($(blockSize))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / $(blockSize)
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / $(blockSize)
  tl.store($(meanReg), μ)
  tl.store($(varReg), v)
}

/-- Online Welford kernel: forLoop maintains (M, S, n) per iteration. -/
def onlineWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize : Nat) : Kernel := triton {
  pid := tl.program_id(0)
  M   := $(0 : ℝ)        -- running mean (Real channel)
  S   := $(0 : ℝ)        -- running sum-of-squared-deviations
  tl.for i in $(blockSize) {
    xi   := tl.load($(xReg) + (pid * $(blockSize) + i))
    -- Standard Welford update:
    --   M' = M + (xi - M) / (i + 1)
    --   S' = S + (xi - M) * (xi - M')
    delta  := xi - M
    M      := M + delta / ((i : ℝ) + 1)        -- requires Nat→ℝ cast (see Step 2)
    delta2 := xi - M
    S      := S + delta * delta2
  }
  tl.store($(meanReg), M)
  tl.store($(varReg), S / $(blockSize))
}

/-- Shared spec: the kernel-pair both compute the two-pass mean / variance. -/
noncomputable def welfordMeanSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  twoPassMean xs

noncomputable def welfordVarSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  twoPassS xs / N

theorem twopass_welford_correct
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (_hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoaded s xReg blockSize xs) :
    let final := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    final.bind (fun s' => some (s'.readMem meanReg 0))
        = some (welfordMeanSpec xs)
    ∧ final.bind (fun s' => some (s'.readMem varReg 0))
        = some (welfordVarSpec xs) := by
  sorry

theorem online_welford_correct
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoaded s xReg blockSize xs) :
    let final := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    final.bind (fun s' => some (s'.readMem meanReg 0))
        = some (welfordMeanSpec xs)
    ∧ final.bind (fun s' => some (s'.readMem varReg 0))
        = some (welfordVarSpec xs) := by
  sorry

theorem welford_kernels_refinement
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoaded s xReg blockSize xs) :
    let final_2p := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    let final_on := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    final_2p.bind (fun s' => some (s'.readMem meanReg 0))
        = final_on.bind (fun s' => some (s'.readMem meanReg 0))
    ∧ final_2p.bind (fun s' => some (s'.readMem varReg 0))
        = final_on.bind (fun s' => some (s'.readMem varReg 0)) := by
  sorry

end VeriTile.Examples
```

- [ ] **Step 2: Resolve the Nat→ℝ cast inside the DSL**

The line `M := M + delta / ((i : ℝ) + 1)` requires the loop index `i`
(bound by `tl.for` to `Value.scalarNat`) to be cast to ℝ inside the body.
The current `Value.bop` semantics rejects ℝ/Nat mixing.

Two options:

(a) **Add `Op.natToReal`** to `Triton/Core.lean` + `Op.natToReal` evaluation
in `evalOp` (returns `none` on a non-`scalarNat` argument; otherwise produces
`Value.scalar (n : ℝ)`). DSL surface: `tl.cast(idx)` or just `(idx : ℝ)`-style
sugar. ~30 lines.

(b) **Compute the increment in Nat-channel as much as possible**, using
`Value.tile` with `Real`-valued lookup. E.g., precompute `inv_n_plus_1 :
Fin blockSize → ℝ` outside the kernel and pass it as a tile argument; inside,
`gather` the appropriate value.

Choose (a). It's cleaner. Add to `Op.lean`:

```lean
  | natToReal : Op → Op
```

and to `evalOp`:

```lean
  | .natToReal a, s =>
      match evalOp a s with
      | some (.scalarNat n) => some (.scalar (n : ℝ))
      | some (.tileNat n f) => some (.tile n (fun i => (f i : ℝ)))
      | _ => none
```

DSL surface — extend `tritonExpr` syntax in `DSL.lean`:

```lean
syntax "tl.toReal(" tritonExpr ")" : tritonExpr
-- ... in expandExpr:
  | `(tritonExpr| tl.toReal($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.natToReal $e')
```

Then the kernel line becomes:
```
M := M + delta / (tl.toReal(i) + $(1 : ℝ))
```

- [ ] **Step 3: Add `welford_kernels_refinement` to `VeriTile.lean`**

```lean
import VeriTile.Examples.WelfordKernels
```

- [ ] **Step 4: Build (expect 3 sorry warnings)**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10
```

Expected: 3 `declaration uses sorry` warnings (one per theorem). No errors.

- [ ] **Step 5: Commit**

```bash
git add VeriTile/Triton/Core.lean VeriTile/Triton/Semantics.lean \
        VeriTile/Triton/DSL.lean VeriTile/Examples/WelfordKernels.lean \
        VeriTile.lean
git commit -m "scaffold(welford): Tier 2 #4 kernels + Op.natToReal extension (3 sorries)"
```

### Task 4.2: Prove `twopass_welford_correct`

The two-pass kernel has no forLoop; it's a straight operational walk-through
of `tl.sum` × 2 and arithmetic. Mirror the structure of
`softmax_naive_correct` from `Examples/SoftmaxEq.lean:165–189`.

**Files:**
- Modify: `VeriTile/Examples/WelfordKernels.lean`

- [ ] **Step 1: Replace the `sorry` for `twopass_welford_correct`**

The proof structure (sketch — fill out tactic-by-tactic):

```lean
theorem twopass_welford_correct
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoaded s xReg blockSize xs) :
    let final := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    final.bind (fun s' => some (s'.readMem meanReg 0))
        = some (welfordMeanSpec xs)
    ∧ final.bind (fun s' => some (s'.readMem varReg 0))
        = some (welfordVarSpec xs) := by
  -- Symbolic execution of the kernel:
  --   pid is loaded (.scalarNat s.pid),
  --   offs is the standard offset tile,
  --   x is the loaded input tile,
  --   s_x = ∑ x i  by Value.reduceSum,
  --   μ = s_x / blockSize,
  --   d i = x i - μ,
  --   s_d2 = ∑ (d i)² ,
  --   v = s_d2 / blockSize.
  -- Each step reduces by `simp [exec, twopassWelfordKernel, stepStmts,
  --                              stepStmt, evalOp, Value.bop, Value.uop,
  --                              Value.reduceSum, BlockState.setReg,
  --                              BlockState.writeMem, h_x]`.
  -- Then unfold the spec and recognize:
  --   μ = (∑ x_i)/N           — definition of welfordMeanSpec / twoPassMean
  --   v = (∑ (x_i − μ)²)/N    — definition of welfordVarSpec / twoPassS / N
  refine ⟨?_, ?_⟩
  all_goals
    simp [exec, twopassWelfordKernel, stepStmts, stepStmt, evalOp,
          Value.bop, Value.uop, Value.reduceSum,
          BlockState.setReg, BlockState.readMem, BlockState.writeMem,
          welfordMeanSpec, welfordVarSpec, twoPassMean, twoPassS, h_x]
```

If `simp` doesn't close, expand the proof in stages (this is the standard
approach in `Examples/SoftmaxEq.lean`):

```lean
  -- Simplify exec to a closed form.
  have h_exec :
      exec (twopassWelfordKernel ...) s
        = some <closed form here> := by
    simp [exec, ...]
  rw [h_exec]
  refine ⟨?_, ?_⟩
  · -- Mean readout
    simp [Option.bind, BlockState.readMem, ...]
    rfl  -- or `ring` / `field_simp`
  · -- Var readout
    simp [...]
    -- Algebra: simplify (Σ x²·n − (Σx)²)/n² to Σ(x − μ)² / n.
    field_simp
    ring
```

- [ ] **Step 2: Build clean (one fewer sorry warning)**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10
```

Expected: 2 sorry warnings remaining (online_welford_correct, welford_kernels_refinement).

- [ ] **Step 3: Commit**

```bash
git add VeriTile/Examples/WelfordKernels.lean
git commit -m "feat(welford): twopass_welford_correct via simp on operational semantics"
```

### Task 4.3: Prove `online_welford_correct` via `forLoop_inv`

**Files:**
- Modify: `VeriTile/Examples/WelfordKernels.lean`

- [ ] **Step 1: Identify the loop invariant**

The forLoop body is the Welford update. After iteration `i`, registers `M`
and `S` should hold `welfordMean xs (i+1)` and `welfordS xs (i+1)`
respectively (the Welford recurrence at depth `i+1`).

Define the invariant `P : Nat → BlockState → Prop`:

```lean
-- After k iterations:
--   regs "M" = welfordMean xs k
--   regs "S" = welfordS xs k
def P_welford {n : Nat} (xs : Fin n → ℝ) (k : Nat) (s : BlockState) : Prop :=
  s.regs "M" = some (Value.scalar (welfordMean xs k))
  ∧ s.regs "S" = some (Value.scalar (welfordS xs k))
  ∧ s.regs "pid" = some (Value.scalarNat s.pid)  -- pid is also in scope
```

(Add other register-binding clauses as needed: e.g. `xs` is loaded before
the loop, but the loop body only reads from memory, not from a register — so
the "x" register isn't part of the invariant. Same for `delta`, `delta2`,
`xi`: those are local to a single iteration and don't survive across the
loop boundary in any well-defined way; they can be computed fresh each time.)

- [ ] **Step 2: State and prove the per-iteration step lemma**

```lean
private lemma online_welford_step
    {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName) (s : BlockState)
    (h_x : InputLoaded s xReg N xs)
    (i : Nat) (s_pre : BlockState) (h_lt : i < N) :
    P_welford xs i s_pre →
    ∃ s_post,
      stepStmts onlineWelfordBody (s_pre.setReg "i" (Value.scalarNat i))
        = some s_post
      ∧ P_welford xs (i + 1) s_post := by
  -- Where `onlineWelfordBody` is the body list extracted from the kernel's
  -- forLoop. (Either copy-paste the body list, or define it as a `def` and
  -- reuse from the kernel definition.)
  -- Each statement in the body reduces under simp:
  --   xi = .scalar (xs ⟨i, h_lt⟩)             [from tl.load + InputLoaded]
  --   delta = .scalar (xs ⟨i, h_lt⟩ - M_old)
  --   M_new = M_old + delta / (i + 1)         = welfordMean xs (i+1)
  --   delta2 = .scalar (xs ⟨i, h_lt⟩ - M_new)
  --   S_new = S_old + delta · delta2          = welfordS xs (i+1)
  intro ⟨h_M, h_S, h_pid⟩
  refine ⟨_, ?_, ?_, ?_, ?_⟩
  · simp [stepStmts, stepStmt, evalOp, onlineWelfordBody,
          Value.bop, Value.uop, BlockState.setReg, BlockState.readMem,
          h_M, h_S, h_pid, h_x, h_lt]
  · -- Show new "M" equals welfordMean xs (i+1)
    simp [BlockState.setReg, welfordMean, h_lt]
    ring
  · -- Show new "S" equals welfordS xs (i+1)
    simp [BlockState.setReg, welfordS, welfordMean, h_lt]
    ring
  · simp [BlockState.setReg]
```

If the `simp` blocks don't reduce cleanly, fall back to `unfold` + per-stmt
walkthrough.

- [ ] **Step 3: Apply `forLoop_inv` to the loop**

```lean
theorem online_welford_correct
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoaded s xReg blockSize xs) :
    let final := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    final.bind (fun s' => some (s'.readMem meanReg 0))
        = some (welfordMeanSpec xs)
    ∧ final.bind (fun s' => some (s'.readMem varReg 0))
        = some (welfordVarSpec xs) := by
  -- Pre-loop reduction: pid := program_id, M := 0, S := 0.
  -- Apply forLoop_inv to the loop body with invariant P_welford.
  -- Post-loop reduction: tl.store mean, tl.store var.
  -- Use welford_eq_two_pass to convert welfordMean xs blockSize to
  -- welfordMeanSpec, etc.
  have h_init : P_welford xs 0
      ((s.setReg "pid" (Value.scalarNat s.pid))
        .setReg "M" (Value.scalar 0)
        .setReg "S" (Value.scalar 0)) := by
    refine ⟨?_, ?_, ?_⟩
    · simp [BlockState.setReg, welfordMean]
    · simp [BlockState.setReg, welfordS]
    · simp [BlockState.setReg]
  obtain ⟨s_loop_end, h_loop_eq, h_M_end, h_S_end, _⟩ :=
    forLoop_inv h_init
      (fun i s_pre h_lt hP => online_welford_step xs xReg s
        h_x i s_pre h_lt hP)
  -- Now compose with the post-loop tl.store statements and use welford_eq_two_pass.
  refine ⟨?_, ?_⟩
  · -- mean readout
    simp [exec, onlineWelfordKernel, stepStmts, stepStmt, evalOp,
          BlockState.setReg, BlockState.writeMem, BlockState.readMem,
          h_loop_eq, h_M_end, welfordMeanSpec]
    -- welfordMean xs blockSize = twoPassMean xs by welford_eq_two_pass
    exact (welford_eq_two_pass hN xs).1
  · -- var readout
    simp [exec, onlineWelfordKernel, stepStmts, stepStmt, evalOp,
          BlockState.setReg, BlockState.writeMem, BlockState.readMem,
          h_loop_eq, h_S_end, welfordVarSpec]
    -- welfordS xs blockSize = twoPassS xs, then divide by N
    rw [(welford_eq_two_pass hN xs).2]
```

- [ ] **Step 4: Build clean (1 sorry warning remaining)**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10
```

Expected: 1 sorry warning (`welford_kernels_refinement`).

- [ ] **Step 5: Commit**

```bash
git add VeriTile/Examples/WelfordKernels.lean
git commit -m "feat(welford): online_welford_correct via forLoop_inv"
```

### Task 4.4: Compose into `welford_kernels_refinement`

**Files:**
- Modify: `VeriTile/Examples/WelfordKernels.lean`

- [ ] **Step 1: Replace the final sorry**

```lean
theorem welford_kernels_refinement
    (xReg meanReg varReg : RegionName) (blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputLoaded s xReg blockSize xs) :
    let final_2p := exec (twopassWelfordKernel xReg meanReg varReg blockSize) s
    let final_on := exec (onlineWelfordKernel xReg meanReg varReg blockSize) s
    final_2p.bind (fun s' => some (s'.readMem meanReg 0))
        = final_on.bind (fun s' => some (s'.readMem meanReg 0))
    ∧ final_2p.bind (fun s' => some (s'.readMem varReg 0))
        = final_on.bind (fun s' => some (s'.readMem varReg 0)) := by
  obtain ⟨h_2p_mean, h_2p_var⟩ :=
    twopass_welford_correct xReg meanReg varReg blockSize hN s xs h_x
  obtain ⟨h_on_mean, h_on_var⟩ :=
    online_welford_correct xReg meanReg varReg blockSize hN s xs h_x
  exact ⟨h_2p_mean.trans h_on_mean.symm, h_2p_var.trans h_on_var.symm⟩
```

- [ ] **Step 2: Build clean (no sorry warnings)**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add VeriTile/Examples/WelfordKernels.lean
git commit -m "feat(tier2): close welford_kernels_refinement (#4)"
```

---

## Workstream 5 — Tier 2 #5: `online_softmax_recurrence_eq_batch` (~2 weeks)

The paper centerpiece. Streaming softmax `(m_new, l_new)` recurrence over
KV blocks ≡ batch softmax over the whole input. ~80–120 lines Lean expected.

This is independent of #4 / #6 and can run in parallel — start it as soon as
Workstream 2 is done.

### Task 5.1: Skeleton — kernels + spec + statements with `sorry`

**Files:**
- Create: `VeriTile/Examples/OnlineSoftmax.lean`
- Modify: `VeriTile.lean`

- [ ] **Step 1: Write skeleton**

Create `VeriTile/Examples/OnlineSoftmax.lean`:

```lean
/-
VeriTile.Examples.OnlineSoftmax

Tier 2 kernel-pair (PAPER CENTERPIECE): online softmax recurrence ≡ batch
softmax. The streaming form

  m_0 = -∞,  l_0 = 0
  m_{k+1} = max(m_k, max(x_block_k))
  l_{k+1} = exp(m_k − m_{k+1}) · l_k + Σ exp(x_block_k − m_{k+1})

produces the same (m, l) as the one-shot batch form

  m = max(x_0, ..., x_{N-1})
  l = Σ exp(x_i − m)

This is the algorithmic core of FlashAttention; Phase C will reuse the
recurrence at the kernel level.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton

/-- Batch softmax kernel: one tl.max + one tl.sum + per-element divide.
    Identical to `Examples.SoftmaxEq.stableSoftmaxKernel`, restated here for
    clarity. -/
def batchSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel :=
  stableSoftmaxKernel xReg yReg N

/-- Online softmax kernel: maintains (m, l) across blocks of size `Bk`.
    For Phase B we use a degenerate "block size 1" form so that the loop
    iterates over single elements, exposing the recurrence directly. The
    Phase C FA kernel will instantiate this with full Bk-size blocks. -/
def onlineSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel := triton {
  pid := tl.program_id(0)
  m   := $(-1e38 : ℝ)        -- -∞ sentinel; real-valued; unused in the proof
  l   := $(0 : ℝ)
  tl.for i in $(N) {
    xi    := tl.load($(xReg) + (pid * $(N) + i))
    m_new := tl.max(m, xi)                  -- max2 lifted; see DSL note below
    l     := tl.exp(m - m_new) * l + tl.exp(xi - m_new)
    m     := m_new
  }
  -- Then the per-element Y[i] = exp(x_i - m) / l divide; same as stable kernel.
  -- For this Phase B theorem we focus on (m, l), not Y; Y is the Phase C work.
}

/-- Online softmax math: the streaming `(m, l)` recurrence at depth k. -/
noncomputable def onlineSoftmaxM {N : Nat} (xs : Fin N → ℝ) : Nat → ℝ
  | 0     => -1e38
  | k + 1 =>
      if h : k < N then max (onlineSoftmaxM xs k) (xs ⟨k, h⟩) else onlineSoftmaxM xs k

noncomputable def onlineSoftmaxL {N : Nat} (xs : Fin N → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < N then
        let m_old := onlineSoftmaxM xs k
        let m_new := onlineSoftmaxM xs (k + 1)
        Real.exp (m_old - m_new) * onlineSoftmaxL xs k + Real.exp (xs ⟨k, h⟩ - m_new)
      else onlineSoftmaxL xs k

/-- Batch softmax math. -/
noncomputable def batchSoftmaxM {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  tileMax hN xs  -- defined in SoftmaxEq.lean

noncomputable def batchSoftmaxL {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  ∑ i, Real.exp (xs i - batchSoftmaxM hN xs)

/-- **The math identity (paper centerpiece)**: the online recurrence at depth
    N produces the same (m, l) as the batch form. -/
theorem online_softmax_recurrence_eq_batch
    {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) :
    onlineSoftmaxM xs N = batchSoftmaxM hN xs ∧
    onlineSoftmaxL xs N = batchSoftmaxL hN xs := by
  sorry

/-- Operational correctness: the online softmax kernel computes (m, l) at
    each program_id matching `onlineSoftmaxM xs N` and `onlineSoftmaxL xs N`. -/
theorem online_softmax_correct
    (xReg yReg : RegionName) (N : Nat) (hN : 0 < N)
    (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoaded s xReg N xs) :
    -- (m, l) registers at the end of execution match the math definitions.
    let final := exec (onlineSoftmaxKernel xReg yReg N) s
    final.bind (fun s' => s'.regs "m" >>= Value.asScalar)
        = some (onlineSoftmaxM xs N)
    ∧ final.bind (fun s' => s'.regs "l" >>= Value.asScalar)
        = some (onlineSoftmaxL xs N) := by
  sorry

end VeriTile.Examples
```

- [ ] **Step 2: Add to `VeriTile.lean`**

```lean
import VeriTile.Examples.OnlineSoftmax
```

- [ ] **Step 3: Note: `tl.max(m, xi)` requires `Op.max2` on scalars + DSL surface**

Check `Triton/Core.lean:66`: `Op.max2` already exists. DSL surface form? Check
`Triton/DSL.lean` for whether `tl.max(a, b)` parses as the *binary* form or
only the *reduction* form. If only reduction, add a binary form:

```lean
syntax "tl.max(" tritonExpr ", " tritonExpr ")" : tritonExpr
-- in expandExpr:
  | `(tritonExpr| tl.max($a:tritonExpr, $b:tritonExpr)) => do
      let a' ← expandExpr a
      let b' ← expandExpr b
      `(Op.max2 $a' $b')
```

- [ ] **Step 4: Build (expect 2 sorry warnings)**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10
```

Expected: 2 sorry warnings.

- [ ] **Step 5: Commit**

```bash
git add VeriTile/Examples/OnlineSoftmax.lean VeriTile/Triton/DSL.lean VeriTile.lean
git commit -m "scaffold(online-softmax): Tier 2 #5 kernels + statements (2 sorries)"
```

### Task 5.2: Prove the math identity `online_softmax_recurrence_eq_batch`

This is the *math* core. The proof is by induction on a strengthened
"prefix" form.

**Files:**
- Modify: `VeriTile/Examples/OnlineSoftmax.lean`

- [ ] **Step 1: State the prefix-form helper**

```lean
/-- Prefix form of the recurrence-vs-batch identity.

    For any prefix length `k ≤ N`, the online recurrence at depth k matches
    the batch form restricted to the first k inputs:
      onlineSoftmaxM xs k = max over i ∈ Fin k of (xs (castFin h i)) ⊔ (-∞)
      onlineSoftmaxL xs k = ∑ i ∈ Fin k of exp(xs (castFin h i) − onlineSoftmaxM xs k)

    The N-length case is the user-facing `online_softmax_recurrence_eq_batch`. -/
private theorem online_softmax_prefix
    {N : Nat} (xs : Fin N → ℝ) :
    ∀ k : Nat, ∀ (hk : k ≤ N),
      onlineSoftmaxM xs k = (if hk0 : 0 < k then
        ((Finset.univ : Finset (Fin k)).sup'
          (Finset.univ_nonempty_iff.mpr ⟨⟨0, hk0⟩⟩)
          (fun i => xs (castFinHelper hk i))) else -1e38)
      ∧ onlineSoftmaxL xs k = ∑ i : Fin k,
          Real.exp (xs (castFinHelper hk i) - onlineSoftmaxM xs k) := by
  sorry
```

(`castFinHelper` is the analogue of `WelfordMath.castFin`. Either reuse from
`WelfordMath` (lift its visibility from `private`) or duplicate it.)

- [ ] **Step 2: Prove the prefix form by induction on `k`**

The key step at `k → k+1`:
- new max: `onlineSoftmaxM xs (k+1) = max (onlineSoftmaxM xs k) (xs ⟨k, _⟩)`
- new l: factor out `exp(m_k - m_{k+1})`:
  ```
  l_{k+1} = exp(m_k - m_{k+1}) · l_k + exp(x_k - m_{k+1})
          = exp(m_k - m_{k+1}) · ∑_{i<k} exp(x_i - m_k) + exp(x_k - m_{k+1})
          = ∑_{i<k} exp(m_k - m_{k+1}) · exp(x_i - m_k) + exp(x_k - m_{k+1})
          = ∑_{i<k} exp(x_i - m_{k+1}) + exp(x_k - m_{k+1})
          = ∑_{i<k+1} exp(x_i - m_{k+1})
  ```

```lean
private theorem online_softmax_prefix
    {N : Nat} (xs : Fin N → ℝ) :
    ∀ k : Nat, ∀ (hk : k ≤ N), <statement> := by
  intro k
  induction k with
  | zero =>
    intro _
    refine ⟨?_, ?_⟩
    · simp [onlineSoftmaxM]
    · simp [onlineSoftmaxL]
  | succ j ih =>
    intro hk
    have hj : j ≤ N := Nat.le_of_succ_le hk
    have hj_lt : j < N := hk
    obtain ⟨h_M, h_L⟩ := ih hj
    refine ⟨?_, ?_⟩
    · -- Max case: just unfold `onlineSoftmaxM` at `j+1`.
      simp [onlineSoftmaxM, hj_lt]
      sorry  -- Finset.sup' over Fin (j+1) = max (over Fin j) (xs ⟨j, _⟩); standard
    · -- Sum case: the algebraic step above.
      simp [onlineSoftmaxL, hj_lt]
      rw [h_L]
      -- Distribute exp(m_j - m_{j+1}) into the sum;
      -- combine with exp(x_j - m_{j+1}) using Finset.sum_range_succ.
      sorry  -- see math sketch in Step 2 prelude
```

- [ ] **Step 3: Discharge the two inner sorries**

The "max case" follows from `Finset.sup'_succ` (or the equivalent
`Finset.sup'_insert` + `Finset.univ_succ_eq_insert_zero`); look up the exact
Mathlib name with `lean_local_search` or `lean_loogle`.

The "sum case" follows from:
- `Finset.sum_range_succ` (sum over Fin (k+1) = sum over Fin k + element at k)
- `Real.exp_add` (`exp(a+b) = exp(a) * exp(b)`)
- algebra: `exp(m_j - m_{j+1}) * exp(x_i - m_j) = exp(x_i - m_{j+1})`

If the proofs drag past 2 days, escalate to `/lean4:autoprove` on these two
sub-goals. Treat the `prefix` lemma as the "hard math" task of Phase B.

- [ ] **Step 4: Derive the user-facing theorem from prefix form**

```lean
theorem online_softmax_recurrence_eq_batch
    {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) :
    onlineSoftmaxM xs N = batchSoftmaxM hN xs ∧
    onlineSoftmaxL xs N = batchSoftmaxL hN xs := by
  obtain ⟨h_M, h_L⟩ := online_softmax_prefix xs N (le_refl N)
  refine ⟨?_, ?_⟩
  · simp [batchSoftmaxM, tileMax, h_M, hN]
    -- castFinHelper at N is `id`.
    rfl  -- or `congr 1; ext; rfl`
  · simp [batchSoftmaxL, h_L, h_M, hN]
    rfl
```

- [ ] **Step 5: Build clean (one fewer sorry warning)**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10
```

Expected: 1 sorry warning (`online_softmax_correct`).

- [ ] **Step 6: Commit**

```bash
git add VeriTile/Examples/OnlineSoftmax.lean
git commit -m "feat(online-softmax): math identity online_softmax_recurrence_eq_batch (#5 math)"
```

### Task 5.3: Prove `online_softmax_correct` via `forLoop_inv`

**Files:**
- Modify: `VeriTile/Examples/OnlineSoftmax.lean`

- [ ] **Step 1: State the per-iteration step lemma**

```lean
private lemma online_softmax_step
    {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName) (s_outer : BlockState)
    (h_x : InputLoaded s_outer xReg N xs)
    (i : Nat) (s_pre : BlockState) (h_lt : i < N) :
    P_online_softmax xs s_outer.pid i s_pre →
    ∃ s_post,
      stepStmts onlineSoftmaxBody (s_pre.setReg "i" (Value.scalarNat i))
        = some s_post
      ∧ P_online_softmax xs s_outer.pid (i + 1) s_post := by
  sorry  -- analogous to online_welford_step in shape
```

Where `P_online_softmax` is:

```lean
def P_online_softmax {N : Nat} (xs : Fin N → ℝ) (pid : Nat) (k : Nat)
    (s : BlockState) : Prop :=
  s.regs "m" = some (Value.scalar (onlineSoftmaxM xs k))
  ∧ s.regs "l" = some (Value.scalar (onlineSoftmaxL xs k))
  ∧ s.regs "pid" = some (Value.scalarNat pid)
```

Prove by reducing the body's `stepStmts` chain manually (like Welford's
step proof). The body is 4 stmts: `xi := load`, `m_new := max(m, xi)`,
`l := ...`, `m := m_new`. Each is a single simp + setReg.

- [ ] **Step 2: Apply `forLoop_inv`**

```lean
theorem online_softmax_correct
    (xReg yReg : RegionName) (N : Nat) (hN : 0 < N)
    (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoaded s xReg N xs) :
    let final := exec (onlineSoftmaxKernel xReg yReg N) s
    final.bind (fun s' => s'.regs "m" >>= Value.asScalar)
        = some (onlineSoftmaxM xs N)
    ∧ final.bind (fun s' => s'.regs "l" >>= Value.asScalar)
        = some (onlineSoftmaxL xs N) := by
  -- 1. Pre-loop reduction (pid, m, l init).
  have h_init : P_online_softmax xs s.pid 0
      (((s.setReg "pid" (Value.scalarNat s.pid))
          .setReg "m" (Value.scalar (-1e38)))
          .setReg "l" (Value.scalar 0)) := by
    refine ⟨?_, ?_, ?_⟩
    all_goals simp [BlockState.setReg, onlineSoftmaxM, onlineSoftmaxL]
  -- 2. Apply forLoop_inv.
  obtain ⟨s_loop_end, h_loop_eq, hM_end, hL_end, _⟩ :=
    forLoop_inv h_init
      (fun i s_pre h_lt hP => online_softmax_step xs xReg s h_x i s_pre h_lt hP)
  -- 3. Read out (m, l) from final state. The kernel ends after the loop;
  --    no further statements modify "m" or "l".
  refine ⟨?_, ?_⟩
  · simp [exec, onlineSoftmaxKernel, stepStmts, stepStmt, evalOp,
          BlockState.setReg, h_loop_eq, hM_end, Value.asScalar]
  · simp [exec, onlineSoftmaxKernel, stepStmts, stepStmt, evalOp,
          BlockState.setReg, h_loop_eq, hL_end, Value.asScalar]
```

- [ ] **Step 3: Build clean**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add VeriTile/Examples/OnlineSoftmax.lean
git commit -m "feat(tier2): close online_softmax_correct (#5 operational)"
```

---

## Workstream 6 — Tier 2 #6: `layernorm_kernels_refinement` (~1.5 weeks)

Single-pass fused LayerNorm ≡ two-pass LayerNorm. Reuses Welford internally.
Depends on Workstream 4.

### Task 6.1: Skeleton — kernels + spec + statements with `sorry`

**Files:**
- Create: `VeriTile/Examples/LayerNormKernels.lean`
- Modify: `VeriTile.lean`

- [ ] **Step 1: Write skeleton**

Create `VeriTile/Examples/LayerNormKernels.lean`:

```lean
/-
VeriTile.Examples.LayerNormKernels

Tier 2 kernel-pair: fused single-pass LayerNorm ≡ two-pass LayerNorm.
Composes the Welford recurrence (Phase B #4) with the affine
`(x − μ)/√(var+ε) · γ + β` transform.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.SoftmaxEq
import VeriTile.Examples.WelfordKernels
import VeriTile.Examples.WelfordMath

namespace VeriTile.Examples

open VeriTile.Triton

/-- Two-pass LayerNorm kernel: tl.sum twice (mean and var), then affine. -/
def twoPassLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(N) + tl.arange($(N))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / $(N)
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / $(N)
  γ      := tl.load($(γReg) + tl.arange($(N)))
  β      := tl.load($(βReg) + tl.arange($(N)))
  σ_inv  := $(1 : ℝ) / tl.sqrt(v + $(ε))   -- requires Op.sqrt; see below
  y      := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, y)
}

/-- Fused single-pass LayerNorm kernel: Welford forLoop, then affine. -/
def fusedLayerNormKernel
    (xReg γReg βReg yReg : RegionName) (N : Nat) (ε : ℝ) : Kernel := triton {
  pid := tl.program_id(0)
  M   := $(0 : ℝ)
  S   := $(0 : ℝ)
  tl.for i in $(N) {
    xi      := tl.load($(xReg) + (pid * $(N) + i))
    delta   := xi - M
    M       := M + delta / (tl.toReal(i) + $(1 : ℝ))
    delta2  := xi - M
    S       := S + delta * delta2
  }
  μ       := M
  v       := S / $(N)
  σ_inv   := $(1 : ℝ) / tl.sqrt(v + $(ε))
  -- Then a second pass to compute Y. We can't avoid the second read of x for
  -- the residual (x - μ); the "fused" gain is that we computed μ/var in one
  -- pass.
  offs    := pid * $(N) + tl.arange($(N))
  x       := tl.load($(xReg) + offs)
  γ       := tl.load($(γReg) + tl.arange($(N)))
  β       := tl.load($(βReg) + tl.arange($(N)))
  y       := (x - μ) * σ_inv * γ + β
  tl.store($(yReg) + offs, y)
}

/-- LayerNorm spec: y_i = (x_i − μ)/√(var+ε) · γ_i + β_i. -/
noncomputable def layerNormSpec {N : Nat}
    (xs γs βs : Fin N → ℝ) (ε : ℝ) (i : Fin N) : ℝ :=
  let μ : ℝ := twoPassMean xs
  let v : ℝ := twoPassS xs / N
  (xs i - μ) / Real.sqrt (v + ε) * γs i + βs i

theorem twopass_layernorm_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : InputLoaded s xReg N xs)
    (h_γ : InputLoaded s γReg N γs)
    (h_β : InputLoaded s βReg N βs) :
    ∀ i : Fin N,
      observeY (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s) N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  sorry

theorem fused_layernorm_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : InputLoaded s xReg N xs)
    (h_γ : InputLoaded s γReg N γs)
    (h_β : InputLoaded s βReg N βs) :
    ∀ i : Fin N,
      observeY (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s) N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  sorry

theorem layernorm_kernels_refinement
    (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : InputLoaded s xReg N xs)
    (h_γ : InputLoaded s γReg N γs)
    (h_β : InputLoaded s βReg N βs) :
    ∀ i : Fin N,
      observeY (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s) N s.pid i
        = observeY (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s) N s.pid i := by
  sorry

end VeriTile.Examples
```

- [ ] **Step 2: Add `Op.sqrt` to `Triton/Core.lean`**

```lean
  | sqrt : Op → Op
```

In `evalOp`:
```lean
  | .sqrt a, s =>
      match evalOp a s with
      | some va => va.uop Real.sqrt
      | none => none
```

DSL surface in `DSL.lean`:
```lean
syntax "tl.sqrt(" tritonExpr ")" : tritonExpr
-- in expandExpr:
  | `(tritonExpr| tl.sqrt($e:tritonExpr)) => do
      let e' ← expandExpr e
      `(Op.sqrt $e')
```

- [ ] **Step 3: Add to `VeriTile.lean`**

```lean
import VeriTile.Examples.LayerNormKernels
```

- [ ] **Step 4: Build (expect 3 sorry warnings)**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10
```

- [ ] **Step 5: Commit**

```bash
git add VeriTile/Triton/Core.lean VeriTile/Triton/Semantics.lean \
        VeriTile/Triton/DSL.lean \
        VeriTile/Examples/LayerNormKernels.lean VeriTile.lean
git commit -m "scaffold(layernorm): Tier 2 #6 kernels + Op.sqrt extension (3 sorries)"
```

### Task 6.2: Prove `twopass_layernorm_correct`

Operational walk-through; structurally identical to `twopass_welford_correct`
(Workstream 4 Task 4.2) plus a per-element affine transform on the y tile.

**Files:**
- Modify: `VeriTile/Examples/LayerNormKernels.lean`

- [ ] **Step 1: Replace the sorry following the same simp pattern as Task 4.2**

The kernel reduces to a closed form for `Y` via simp on the operational
semantics; the math is identical to `layerNormSpec` per element.

```lean
theorem twopass_layernorm_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : InputLoaded s xReg N xs)
    (h_γ : InputLoaded s γReg N γs)
    (h_β : InputLoaded s βReg N βs) :
    ∀ i : Fin N,
      observeY (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s) N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  intro i
  simp [exec, twoPassLayerNormKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop, Value.reduceSum,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        observeY, h_x, h_γ, h_β,
        layerNormSpec, twoPassMean, twoPassS]
  -- The arithmetic should close by `ring` or `field_simp` after the simp
  -- pass exposes a closed form.
  ring
```

If `ring`/`field_simp` doesn't close, expand the simp set and pass through
intermediate `have ... := by simp [...]` steps.

- [ ] **Step 2: Build (2 sorry warnings remaining)**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add VeriTile/Examples/LayerNormKernels.lean
git commit -m "feat(layernorm): twopass_layernorm_correct"
```

### Task 6.3: Prove `fused_layernorm_correct` via `forLoop_inv` + reuse Welford

**Files:**
- Modify: `VeriTile/Examples/LayerNormKernels.lean`

- [ ] **Step 1: Decompose into "Welford part" + "affine part"**

The fused kernel has a Welford-shaped forLoop followed by the same affine
post-pass as `twoPassLayerNormKernel`. We prove `M`/`S` after the loop equal
`twoPassMean`/`twoPassS` via `forLoop_inv` + `welford_eq_two_pass`, then
hand off to the same simp chain as Task 6.2.

```lean
theorem fused_layernorm_correct
    (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : InputLoaded s xReg N xs)
    (h_γ : InputLoaded s γReg N γs)
    (h_β : InputLoaded s βReg N βs) :
    ∀ i : Fin N,
      observeY (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s) N s.pid i
        = some (layerNormSpec xs γs βs ε i) := by
  intro i
  -- Step 1: Apply forLoop_inv to the Welford forLoop body.
  --   Invariant: regs "M" = welfordMean xs k, regs "S" = welfordS xs k.
  --   This is the same step lemma as in Workstream 4. Reuse via:
  obtain ⟨s_loop_end, h_loop_eq, hM_end, hS_end, _⟩ :=
    -- (Construct the same h_init as in Task 4.3, then apply forLoop_inv.)
    sorry  -- copy / adapt the online_welford_correct proof structure
  -- Step 2: After the loop, M = twoPassMean xs and S/N = twoPassS xs / N.
  rw [(welford_eq_two_pass hN xs).1] at hM_end
  rw [(welford_eq_two_pass hN xs).2] at hS_end
  -- Step 3: Reduce the post-loop statements (μ, v, σ_inv, x reload, γ load,
  --         β load, y compute, y store).
  simp [exec, fusedLayerNormKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop, Value.reduceSum,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        observeY, h_x, h_γ, h_β, h_loop_eq, hM_end, hS_end,
        layerNormSpec, twoPassMean, twoPassS]
  ring
```

The inner `sorry` is filled by an exact copy of the loop reasoning from
`online_welford_correct` (Workstream 4 Task 4.3). If that loop reasoning
was extracted to a private lemma in Workstream 4, reuse it directly here.
Otherwise, extract it now and rebuild Workstream 4 to use the helper.

- [ ] **Step 2: Build (1 sorry warning remaining)**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -10
```

- [ ] **Step 3: Commit**

```bash
git add VeriTile/Examples/LayerNormKernels.lean VeriTile/Examples/WelfordKernels.lean
git commit -m "feat(layernorm): fused_layernorm_correct via reused Welford loop reasoning"
```

### Task 6.4: Compose into `layernorm_kernels_refinement`

**Files:**
- Modify: `VeriTile/Examples/LayerNormKernels.lean`

- [ ] **Step 1: Replace the final sorry**

```lean
theorem layernorm_kernels_refinement
    (xReg γReg βReg yReg : RegionName) (N : Nat) (hN : 0 < N) (ε : ℝ)
    (s : BlockState) (xs γs βs : Fin N → ℝ)
    (h_x : InputLoaded s xReg N xs)
    (h_γ : InputLoaded s γReg N γs)
    (h_β : InputLoaded s βReg N βs) :
    ∀ i : Fin N,
      observeY (exec (twoPassLayerNormKernel xReg γReg βReg yReg N ε) s) N s.pid i
        = observeY (exec (fusedLayerNormKernel xReg γReg βReg yReg N ε) s) N s.pid i := by
  intro i
  rw [twopass_layernorm_correct xReg γReg βReg yReg N hN ε s xs γs βs h_x h_γ h_β i,
      fused_layernorm_correct  xReg γReg βReg yReg N hN ε s xs γs βs h_x h_γ h_β i]
```

- [ ] **Step 2: Build clean**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: clean. All Tier 2 theorems closed.

- [ ] **Step 3: Commit**

```bash
git add VeriTile/Examples/LayerNormKernels.lean
git commit -m "feat(tier2): close layernorm_kernels_refinement (#6)"
```

---

## Workstream 7 — Diff-test artifact (~1 week, parallel)

Non-gating credibility artifact. Pick 1–2 of the 6 closed Tier 1+2 kernels.
Recommend Welford as the canonical Tier 2 representative (interesting
numerical pattern: online recurrence is well-known to differ slightly from
two-pass at finite precision, so the diff-test serves as a direct
demonstration that VeriTile's `ℝ` modelling is correct in the abstract but
*not* a bit-level guarantee — exactly the framing PLAN.md wants).

### Task 7.1: Set up `tools/diff_test/` directory + README

**Files:**
- Create: `tools/diff_test/README.md`
- Create: `tools/diff_test/python/.gitkeep`

- [ ] **Step 1: Create directory + README**

```bash
mkdir -p tools/diff_test/python tools/diff_test/results
```

Create `tools/diff_test/README.md`:

```markdown
# Differential testing artifact

Per `PLAN.md` §Differential testing: this is a credibility supplement, NOT
part of the trusted proof chain. We hand-write Triton kernels in Python that
correspond, statement-by-statement, to the embedded `triton { ... }` AST in
`VeriTile/Examples/`, then compare numeric output against PyTorch reference
on a range of inputs.

## Tolerances

| Tier | Tolerance vs PyTorch (max element-wise abs diff) |
|---|---|
| Tier 1+2 representatives | ≤ 1e-5 |
| FA forward (Phase C) | ≤ 1e-3 |
| FA-2 forward (Phase D) | ≤ 1e-3 |
| FA-1 ↔ FA-2 cross-check (Phase D) | ≤ 1e-3 |

## Failure handling

A diff-test miss is a yellow flag, not a phase gate. Possible causes:
1. IEEE-754 intrinsic difference vs `ℝ` model — acceptable; document.
2. `.py` implementation bug — fix.
3. Side-by-side correspondence asserted incorrectly — re-check.

The Lean proof remains valid in any of the above; only the diff-test artifact
or its scope is updated.

## Side-by-side correspondence

Each `python/<kernel>.py` includes a header comment with line-by-line
correspondence to the Lean kernel definition in `VeriTile/Examples/`. This
is the *human assertion* that the embedded AST matches the runnable code.
A future Python lifter (P3+) would close this gap formally.
```

- [ ] **Step 2: Commit**

```bash
git add tools/diff_test/README.md tools/diff_test/python/.gitkeep
git commit -m "scaffold(diff-test): tools/diff_test/ directory + policy README"
```

### Task 7.2: Hand-write Welford diff-test

**Files:**
- Create: `tools/diff_test/python/welford.py`
- Create: `tools/diff_test/results/welford_phase_b.md`

- [ ] **Step 1: Write the Python script**

Create `tools/diff_test/python/welford.py`:

```python
"""
Welford diff-test.

Side-by-side correspondence to VeriTile/Examples/WelfordKernels.lean:
  twopassWelfordKernel  ↔  twopass_welford_triton (below)
  onlineWelfordKernel   ↔  online_welford_triton (below)

The kernels compute (mean, variance) of a single tile per program_id.
Reference is PyTorch's `torch.var_mean(x, unbiased=False)`.
"""

import argparse
import torch
import triton
import triton.language as tl


@triton.jit
def twopass_welford_triton(
    X_ptr, MEAN_ptr, VAR_ptr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + offs)
    s_x = tl.sum(x, axis=0)
    mu = s_x / BLOCK_SIZE
    d = x - mu
    s_d2 = tl.sum(d * d, axis=0)
    v = s_d2 / BLOCK_SIZE
    tl.store(MEAN_ptr + pid, mu)
    tl.store(VAR_ptr + pid, v)


@triton.jit
def online_welford_triton(
    X_ptr, MEAN_ptr, VAR_ptr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    M = 0.0
    S = 0.0
    for i in range(BLOCK_SIZE):
        xi = tl.load(X_ptr + pid * BLOCK_SIZE + i)
        delta = xi - M
        M = M + delta / (i + 1)
        delta2 = xi - M
        S = S + delta * delta2
    tl.store(MEAN_ptr + pid, M)
    tl.store(VAR_ptr + pid, S / BLOCK_SIZE)


def reference(x: torch.Tensor):
    var, mean = torch.var_mean(x, dim=-1, unbiased=False)
    return mean, var


def run_kernel(kernel_fn, x: torch.Tensor, block_size: int):
    n_blocks = x.shape[0]
    mean = torch.empty(n_blocks, device=x.device, dtype=x.dtype)
    var = torch.empty(n_blocks, device=x.device, dtype=x.dtype)
    grid = (n_blocks,)
    kernel_fn[grid](x, mean, var, BLOCK_SIZE=block_size)
    return mean, var


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = torch.randn(args.n_blocks, args.block_size, device="cuda", dtype=torch.float32)
    x_flat = x.reshape(-1)

    mean_ref, var_ref = reference(x)
    mean_2p, var_2p = run_kernel(twopass_welford_triton, x_flat, args.block_size)
    mean_on, var_on = run_kernel(online_welford_triton, x_flat, args.block_size)

    err_2p_mean = (mean_2p - mean_ref).abs().max().item()
    err_2p_var = (var_2p - var_ref).abs().max().item()
    err_on_mean = (mean_on - mean_ref).abs().max().item()
    err_on_var = (var_on - var_ref).abs().max().item()

    print(f"twopass mean max-abs-diff: {err_2p_mean:.3e}")
    print(f"twopass var  max-abs-diff: {err_2p_var:.3e}")
    print(f"online  mean max-abs-diff: {err_on_mean:.3e}")
    print(f"online  var  max-abs-diff: {err_on_var:.3e}")

    ok = max(err_2p_mean, err_2p_var, err_on_mean, err_on_var) <= args.tol
    print(f"PASS (tol = {args.tol:.0e})" if ok else f"FAIL (tol = {args.tol:.0e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the diff-test on the user's GPU box**

```bash
cd tools/diff_test
python python/welford.py --block-size 64 --n-blocks 128 --seed 0 --tol 1e-5
```

If GPU isn't available locally, document that this step requires a Triton-
capable GPU and skip; the artifact will be generated by an authorized
collaborator. Do not block plan progress on GPU access.

- [ ] **Step 3: Record results**

Create `tools/diff_test/results/welford_phase_b.md`:

```markdown
# Welford diff-test — Phase B Tier 2 representative

Date: 2026-...
Hardware: ...
Triton version: ...
PyTorch version: ...

## Configuration

- BLOCK_SIZE = 64
- N_BLOCKS = 128
- Seed = 0
- Tolerance = 1e-5

## Results

| Kernel | mean max-abs-diff | var max-abs-diff | Pass? |
|---|---|---|---|
| twopass | <fill> | <fill> | <fill> |
| online | <fill> | <fill> | <fill> |

(Seed sweep: also run with seeds 1, 2, 3 and confirm pass; note any
spread.)

## Side-by-side correspondence

The Python kernel `twopass_welford_triton` corresponds line-for-line to
`VeriTile/Examples/WelfordKernels.lean::twopassWelfordKernel`. The Python
kernel `online_welford_triton` corresponds line-for-line to
`VeriTile/Examples/WelfordKernels.lean::onlineWelfordKernel`. Differences
between Python and Lean: only the antiquotation/macro syntax; arithmetic
and control flow are identical.

## Notes

- Online Welford is well-known to be more numerically stable than two-pass
  at very small variances; observed agreement for normally-distributed
  inputs is well within tol.
- Failure outcome handling: per `PLAN.md` §Differential testing.
```

- [ ] **Step 4: Commit**

```bash
git add tools/diff_test/python/welford.py tools/diff_test/results/welford_phase_b.md
git commit -m "feat(diff-test): Welford Phase B Tier 2 representative + report"
```

### Task 7.3: (Optional) Hand-write online-softmax diff-test

If time allows, repeat the pattern with online softmax. Skip if Phase B
schedule is tight; one rep is enough for non-gating.

(Same shape as Task 7.2; replace kernel body and reference with online
softmax / batch softmax. Tolerance ≤ 1e-5.)

---

## Workstream 8 — Phase B LLM benchmark report (~2 days)

Run `scripts/prove.sh` 5 times on each Phase B held-out lemma, aggregate.
Per `PLAN.md` §LLM benchmark protocol; gate target ≥ 50% close rate.

### Task 8.1: Run the trials

**Files:**
- Modify: `bench/llm_eval/results/phase_b_trial_logs/`

- [ ] **Step 1: Run 5 trials per held-out lemma**

```bash
mkdir -p bench/llm_eval/results/phase_b_trial_logs

for lemma in bench/llm_eval/phase_b/*.lean; do
  base=$(basename "$lemma" .lean)
  for trial in 1 2 3 4 5; do
    echo "=== $base trial $trial ==="
    cp "$lemma" "/tmp/${base}_trial_${trial}.lean"
    scripts/prove.sh "/tmp/${base}_trial_${trial}.lean" --max-cycles=10 \
      > "bench/llm_eval/results/phase_b_trial_logs/${base}_trial_${trial}.log" 2>&1
    echo "exit=$?" >> "bench/llm_eval/results/phase_b_trial_logs/${base}_trial_${trial}.log"
  done
done
```

Per `PLAN.md` §LLM benchmark protocol: no manual prompt iteration on the
held-out lemmas. Adjusting `--max-cycles` is allowed and expected (Phase B
proofs are longer than Phase A's `softmax_naive_correct`); start at 10.

If a held-out lemma has 0/5 close rate at `--max-cycles=10`, document the
result and (separately, after the eval) try `--max-cycles=20` to inform
Phase C planning. Don't include the higher-budget result in the canonical
close rate.

- [ ] **Step 2: Tally per-lemma close rates**

```bash
for lemma in bench/llm_eval/phase_b/*.lean; do
  base=$(basename "$lemma" .lean)
  pass=$(grep -c "^exit=0" \
    bench/llm_eval/results/phase_b_trial_logs/${base}_trial_*.log)
  echo "$base: $pass/5"
done
```

- [ ] **Step 3: No commit yet (logs are next-task input)**

### Task 8.2: Aggregate and write report

**Files:**
- Create: `bench/llm_eval/results/phase_b_report.md`

- [ ] **Step 1: Write the report**

Template:

```markdown
# Phase B `scripts/prove.sh` eval report

Date: 2026-...
Wrapper version: v0.1
Plugin: lean4-skills 4.4.9

## Held-out set

(per `bench/llm_eval/phase_b/README.md`, 5 lemmas tagged at start of Phase B)

## Per-lemma results (5 trials each)

| Lemma | Difficulty (estimated) | Close rate | Avg cycles used | Avg wall-clock | Avg API cost USD |
|---|---|---|---|---|---|
| welford_twopass_correct | easy | <X>/5 | ... | ... | ... |
| welford_online_correct | medium | <X>/5 | ... | ... | ... |
| online_softmax_recurrence_step | medium | <X>/5 | ... | ... | ... |
| online_softmax_correct | hard | <X>/5 | ... | ... | ... |
| layernorm_refinement | medium | <X>/5 | ... | ... | ... |

**Aggregate close rate:** total successes / (5 × |held-out|) = <Y>/25 = <Z>%

**Gate target:** ≥ 50% (12.5/25).
**Gate result:** PASS / FAIL.

## Cost summary

| | Total |
|---|---|
| API calls | ... |
| Tokens (input/output) | ... |
| Wall-clock | ... |
| USD | ... |

## Qualitative notes

- Where the plugin closed easily (which proof shapes).
- Where the plugin got stuck (specific tactics, common failure modes).
- Deep-mode escalation rate (count of trials that escalated to `--deep=stuck`).
- Prompt-iteration discipline: confirmed no per-lemma manual hints used.
```

Fill in from `bench/llm_eval/results/phase_b_trial_logs/*.log` (cycles +
wall-clock from `Logs/heldout_*_*.json`; API cost from `result.total_cost_usd`).

- [ ] **Step 2: Commit**

```bash
git add bench/llm_eval/results/phase_b_report.md \
        bench/llm_eval/results/phase_b_trial_logs/
git commit -m "feat(bench): Phase B scripts/prove.sh eval results"
```

---

## Workstream 9 — Exit gate + `v0.2-tier2` tag (~1 day)

### Task 9.1: Verify all gate criteria

**Files:** none (read-only checks)

- [ ] **Step 1: All Tier 2 theorems closed**

```bash
grep -nE "sorry" VeriTile/Examples/WelfordKernels.lean \
                  VeriTile/Examples/OnlineSoftmax.lean \
                  VeriTile/Examples/LayerNormKernels.lean
```

Expected: no matches.

- [ ] **Step 2: `lake build` clean**

```bash
PATH="$HOME/.elan/bin:$PATH" lake build 2>&1 | tail -3
```

Expected: `Build completed successfully (NNNN jobs).` No warnings.

- [ ] **Step 3: Held-out close rate ≥ 50%**

Confirm from `bench/llm_eval/results/phase_b_report.md` that the aggregate is
≥ 12.5/25.

If close rate is below 50%: do not block the tag, but mark in the report
that the gate target was not met (per `PLAN.md` Risk register row "Phase B
held-out close rate is low" mitigation: that's still publishable data —
shifts the paper's narrative).

- [ ] **Step 4: Diff-test artifact in place**

```bash
ls tools/diff_test/python/
ls tools/diff_test/results/
```

Expected: at least one kernel `.py` + one `_phase_b.md` results file.

### Task 9.2: Update `PLAN.md` and tag

**Files:**
- Modify: `PLAN.md`

- [ ] **Step 1: Add a Phase B status section to `PLAN.md`**

Append after the Phase A status section (or at the end of the document):

```markdown
## Phase B status (closed YYYY-MM-DD)

- [x] forLoop operational semantics + `forLoop_inv` lemma family shipped
      (per `Notes/2026-04-29-forloop-inv-design.md`)
- [x] Tier 2 closed: 3 kernel-pair theorems (#4 Welford, #5 online softmax,
      #6 LayerNorm); main `lake build` clean
- [x] Diff-test artifact for <kernel(s)> at `tools/diff_test/`
- [x] Held-out eval close rate: <X>/25 (gate target ≥ 12.5/25)
- [x] Tag `v0.2-tier2`

Next: Gate B → C decision.
```

- [ ] **Step 2: Commit `PLAN.md` update**

```bash
git add PLAN.md
git commit -m "docs(plan): record Phase B exit status"
```

- [ ] **Step 3: Tag the release**

```bash
git tag -a v0.2-tier2 -m "Phase B complete: forLoop semantics + Tier 2 (#4 Welford, #5 online softmax, #6 LayerNorm)"
git push origin v0.2-tier2
```

- [ ] **Step 4: Write release notes on GitHub**

```bash
gh release create v0.2-tier2 --notes "$(cat <<'EOF'
# Phase B: forLoop + Tier 2

Adds `Stmt.forLoop` operational semantics (`mutual` block in
`VeriTile/Triton/Semantics.lean`), the `forLoop_inv` lemma family
(`VeriTile/Triton/LoopInvariant.lean`), and three Tier 2 kernel-pair
theorems.

## New theorems

* `welford_kernels_refinement` (#4) — `Examples/WelfordKernels.lean`
* `online_softmax_recurrence_eq_batch` (#5; *paper centerpiece*) —
  `Examples/OnlineSoftmax.lean`
* `layernorm_kernels_refinement` (#6) — `Examples/LayerNormKernels.lean`

## New tooling

* `Op.natToReal`, `Op.sqrt` — minor `Triton/Core.lean` extensions
* `tl.for i in $(n) { ... }`, `tl.toReal(_)`, `tl.sqrt(_)`,
  binary `tl.max(a, b)` — DSL surface

## Eval results

`scripts/prove.sh` close rate on Phase B held-out: <X>/25 (gate ≥ 12.5/25).

## Diff-test

* Welford kernel pair (`tools/diff_test/python/welford.py`):
  `<max-abs-diff result>` vs PyTorch `torch.var_mean`, tolerance 1e-5.

## Next

Gate B → C: see `PLAN.md` §Phase gate decision rules. Phase C work begins
on `fa_forward_correct` (#7) once the gate decision is made.
EOF
)"
```

- [ ] **Step 5: Confirm release page**

Check: `https://github.com/Lizn-zn/VeriTile/releases/tag/v0.2-tier2`.

---

## Self-review

(Run before declaring Phase B done.)

- [ ] **Spec coverage check** — for each Phase B deliverable in `PLAN.md`,
  confirm a task in this plan implements it:
  - `forLoop` operational semantics → Workstream 1 (Tasks 1.1–1.4)
  - `forLoop_inv` lemma family → Workstream 2 (Tasks 2.1–2.5)
  - 3 Tier 2 kernel-pair theorems → Workstreams 4, 5, 6
  - Diff-test artifact → Workstream 7
  - LLM benchmark + report → Workstreams 3, 8
  - Exit gate → Workstream 9

- [ ] **Type / name consistency check** — function and theorem names used
  across tasks must match:
  - `stepForLoopAux` (Workstreams 1, 2)
  - `forLoop_inv`, `forLoop_readout_scalar`, `forLoop_readout_tile`
    (Workstream 2 + Workstreams 4, 5, 6)
  - `welfordMean`, `welfordS`, `twoPassMean`, `twoPassS`,
    `welford_eq_two_pass` (Workstream 4 + 6, all from
    `Examples/WelfordMath.lean`)
  - `tileMax`, `InputLoaded`, `observeY` (Workstream 4–6, from
    `Examples/SoftmaxEq.lean`)

- [ ] **Placeholder scan** — every step has either real Lean code, a
  specific `lake` command, or an explicit decision criterion. The two
  Step-5 fallbacks in Tasks 1.2 and 5.2 explicitly delineate primary
  approach and fallback to invoke if primary fails after a documented
  amount of effort.

- [ ] **Build commands** — every Lean step uses `PATH="$HOME/.elan/bin:$PATH"
  lake build` or `lake env lean`. Diff-test step uses GPU-Python; clearly
  marked as non-gating.

- [ ] **Dependencies honored** — Workstream order is:
  - 1 before 2 (semantics before lemma family)
  - 2 before 4, 5, 6 (lemma family before kernel theorems)
  - 4 before 6 (Welford before LayerNorm reuse)
  - 5 independent of 4 / 6
  - 3 (held-out tagging) sits between Workstream 4/5/6's *skeleton* tasks
    and their *proof* tasks
  - 7 (diff-test) parallel to anything
  - 8 (benchmark) requires 4, 5, 6 closed AND 3 done
  - 9 (exit) requires all above

If anything is missing, add the task before invoking subagent-driven
development or executing-plans.
