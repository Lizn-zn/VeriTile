# RP1: Pointers vs Named Regions

**Status:** Resolved. Default for Phases A–D: **named regions**.
**Date:** 2026-04-27.
**Owner:** Phase A.
**Revisit conditions:** see §When to revisit.

---

## TL;DR

The embedded Triton DSL models memory accesses as **`Op.load (region :
String) (offset : Op)`** — a static region name chosen at kernel-definition
time + a dynamic offset expression. We do **not** model first-class
pointers (CUDA-style `x_ptr + offsets` as a value).

This is a deliberate verification-side optimization: it makes region
disjointness automatic, eliminates aliasing analysis, removes the need for
memory-safety invariants, and keeps `evalOp` a clean structural recursion.
The cost is a tiny syntactic gap in DSL paste-in (`tl.load(X, offs)` vs
`tl.load(x_ptr + offs)`), which is paper-acceptable and can be closed by
~20 lines of cosmetic macro sugar if/when needed.

None of the 8 main correctness theorems require first-class pointers.

---

## 1. Problem statement

Should the embedded Triton DSL model memory accesses via:

- **(γ) First-class pointers**, as Triton/CUDA does — `x_ptr + offsets` is
  a runtime value of pointer (or pointer-tile) type, can be assigned to a
  register, conditionally selected (`if cond then x_ptr else y_ptr`),
  stored into memory, used for indirect indexing, etc.; *or*
- **Named regions + offsets** — region is a static `String` chosen at
  kernel-definition time, offset is a dynamic `Op`-valued expression; the
  user writes `tl.load(X, offs)` rather than `tl.load(X_ptr + offs)`.

This decision propagates through `Op.load` / `Stmt.store`, `BlockState.mem`,
and every kernel proof in the project.

## 2. Decomposing "support for pointers"

The colloquial phrase "support pointers" conflates three independent
concerns. Only **(γ)** is a real semantic-layer decision; **(α)** is
already supported and **(β)** is a separate cosmetic question.

### (α) Pointer arithmetic as offset arithmetic — already supported

Triton-level expressions like `x_ptr + offs[:, None] * stride + offs[None, :]`
encode an offset computation. In our model, the same computation is
written as the *offset argument* to `tl.load(X, expr)`:

```python
# Triton
x = tl.load(x_ptr + offs[:, None] * stride + offs[None, :])
```
```lean
-- VeriTile DSL (semantically equivalent)
x := tl.load(X, offs[:, None] * stride + offs[None, :])
```

The math is identical; we just spell "X" once instead of carrying a
pointer-typed value. No expressiveness lost.

### (β) Cosmetic syntax `tl.load(X_ptr + offs)` — separate concern

A purely-syntactic extension to the DSL macro that makes the literal text
`tl.load(X_ptr + offs)` parse and lower to `Op.load "X" offs`. This is
~20 lines of macro work, **does not change the semantics**, and exists
purely for paper paste-in friendliness.

Status: **deferred to P3+**, alongside `CONTRIBUTING.md` tutorial work or
paper appendix preparation. No urgency.

### (γ) First-class pointers as values — the real question

Adding pointers as a `Value` constructor (alongside `Value.scalar` and
`Value.tile`):

```lean
-- Hypothetical extension (NOT proposed)
inductive Value where
  | scalar  : ℝ → Value
  | tile    : (n : Nat) → (Fin n → ℝ) → Value
  | pointer : RegionName → Nat → Value          -- base region + offset
  | ptrTile : (n : Nat) → (Fin n → RegionName × Nat) → Value
```

This is what would let users write `ptr := if cond then X else Y` (with
`ptr` a register-typed pointer value), or `tl.load(ptr_array[i])` for
indirect access.

## 3. Why named regions wins on the verification side

Five concrete reasons, each backed by an existing piece of the framework.

### 3.1 Region disjointness is `rfl`-trivial

`BlockState.mem : RegionName → Nat → ℝ` is a two-level function: the first
level is keyed on `String`. By construction:

```lean
(s.writeMem "X" k v).mem "Y" o = s.mem "Y" o    -- by rfl
```

(Definition of `writeMem` has `if r = region ∧ o = offset then v else
s.mem r o`; for `r = "Y" ≠ "X"` the conditional is `false` and we hit the
else branch.) So writes to "X" provably do not affect reads from "Y", with
zero proof effort.

Under (γ): `ptr_x : Value.pointer "X" 0` and `ptr_y : Value.pointer "Y" 0`
*could* alias if their underlying regions / offsets overlap — and the
operational semantics has to express what "alias" means, what the disjoint
case looks like, and how to discharge it per write. This is the territory
of separation logic / Iris / VST. **Whole research direction**.

### 3.2 Aliasing analysis is not required

`BlockState.scatter_readback` (P1's workhorse, ~50 lines) only requires
`Function.Injective offsetFn` *within a single region*. Cross-region
aliasing is impossible by construction, so we never have to consider
"what if this scatter to Y collides with a previous scatter to X?".

Under (γ): the `scatter_readback` lemma's hypothesis becomes "all writes
hit pairwise-disjoint memory cells", which is `(region, offset)` injective
rather than offset-only injective. That's still tractable but doubles the
case-split workload.

### 3.3 No memory safety in scope

`mem` is a total function `RegionName → Nat → ℝ`. Writing to
`"X"` at offset `999999` succeeds and produces a `BlockState` where that
cell now holds the new value — no segfault, no OOB exception, no
valid-pointer invariant to maintain. The kernel's correctness theorem
only cares about the cells the spec actually checks (`observeY` reads at
`pid * N + i.val`); junk writes elsewhere are harmless.

This is **deliberate**, and explicitly out of scope per `PLAN.md` §Out of
scope (alongside IEEE-754 fidelity). Memory safety is a different paper.

Under (γ): pointers carry validity / liveness / range invariants. Each
load/store has a memory-safety obligation (or a defined behavior on
out-of-bounds, which itself becomes part of the spec). The kernel's
correctness theorem has to thread this through.

### 3.4 `evalOp` stays a clean structural recursion

```lean
| .load region off, s => match evalOp off s with
                         | some (.scalar c) => some (.scalar (s.readMem region (realToNat c)))
                         | some (.tile n f) => some (.tile n (fun i => s.readMem region (realToNat (f i))))
                         | _ => none
```

`region` is a macro-time `String`, baked into the AST. `evalOp` does not
need to first compute "what region is this pointer pointing to?" then
unpack `(region, offset)`. The recursion is one-pass on the offset only.

Under (γ): `evalOp .load ptrExpr` would first evaluate `ptrExpr` to a
`Value.pointer`, unpack it into `(region, base_offset)`, then load. One
extra layer of indirection, more sub-goals in proofs, more `Option`-juggling.

### 3.5 No proof tax on existing examples

Tier 1 proofs (`softmax_naive_correct`, `softmax_stable_correct`,
`log_sum_exp_*`, `softmax_reciprocal_*`) are each ~30–50 lines under the
named-region model. The dominant work is symbolic execution + invocation
of `scatter_readback`.

Estimated under (γ), with mandatory disjointness reasoning:
**~100–150 lines per theorem** — a 3× tax. Across Tier 1+2+3 proofs the
program-wide overhead would be 1000+ extra lines.

## 4. When 8 main theorems would actually need (γ)

| Scenario | Need (γ)? | In our 8 theorems? |
|---|---|---|
| `tl.load(X, complex_offset)` | No (α covers it) | All — softmax, LSE, FA |
| 2D / multi-dim load `X[i*sM + j*sN]` | No (α covers it) | FA forward |
| Multiple buffers in one kernel (X, Y, Out) | No (named regions cover it) | add\_kernel, FA, Welford |
| **Conditional buffer**: `ptr := if cond then X else Y; load(ptr)` | Yes | None |
| **Indirect indexing**: `ptr_tile := load(P, idx); val := load(ptr_tile)` | Yes | None |
| **`tl.make_block_ptr` + `tl.advance`** (Triton 2.x block pointer API) | Either (γ) or (α) equivalent rewrite | FA-2 reference uses this style; (α) rewrite is algorithm-equivalent |
| **Pointer as kernel arg** (variable buffer count) | Yes | None |
| **Pointers stored in memory** | Yes (+ heap-typed memory) | None |

**Verdict: zero of the 8 main theorems need (γ).** FA-2's reference
implementation uses block pointers, but the offset-arithmetic rewrite is
provably equivalent (the rewrite is exactly "make explicit what the block
pointer was tracking implicitly") and is what we'll formalize in Phase D.

## 5. The (β) cosmetic syntax decision — **landed**

Independent from the (γ) question. The syntactic gap between Triton and
VeriTile is now closed at the surface layer:

```python
# Triton
x = tl.load(x_ptr + offsets)
tl.store(out_ptr + offsets, value)
```
```lean
-- VeriTile (current)
x := tl.load($(xReg) + offs)
tl.store($(outReg) + offs, value)
```

The macro lowers `tl.load($(R) + offs)` to `Op.load R offs` and
`tl.store($(R) + offs, v)` to `Stmt.store R offs v`. Scalar-pointer
sugar `tl.load($(R))` / `tl.store($(R), v)` desugars to offset `0`.

**Important:** the `+` is a **pure surface-syntax cue**. `$(xReg)` is
still a `RegionName` (`String`), `offs` is still a `Nat`-valued offset
expression, and there is no first-class pointer value being added —
the macro destructures the surface form and emits the existing
region+offset AST. From the operational semantics' point of view this
is identical to the prior comma-separated form `tl.load($(xReg), offs)`;
RP1's (γ)-rejection still holds.

Resolved 2026-04-27. ~30 lines of `expandExpr` / `expandStmt` /
`exprRegions` / `stmtRegions` extension; no semantic edits; all 8
correctness/refinement theorems closed without proof-body changes.

## 6. Comparable work

Similar verification projects targeting GPU / array kernels have
consistently chosen named-buffer abstractions over pointer-arithmetic
semantics:

- **Vero** (PLDI 2024) — Triton lowering verification; uses named
  buffer abstraction at the proof level.
- **ATL** (POPL 2022) — array language with structural reasoning; no
  pointers in the proof obligations.
- **KaTen** / TVM operator verification (CGO 2023) — operator-level
  buffer abstraction.

The reviewer expectation in the PL-verification community is that
pointer / aliasing / memory-safety reasoning is a **separate concern**,
addressed by separation-logic projects (Iris, VST), not bundled with
algorithmic-equivalence verification.

Our choice aligns with the established pattern. Reviewer pushback is
unlikely.

## 7. When to revisit

Re-evaluate the (γ) decision if any of the following becomes true:

- **Pivot to Triton compiler optimization formalization.** If the project
  scope shifts from "algorithmic equivalence of kernel rewrites" to
  "verifying Triton's lowering passes from Triton IR to PTX", pointers
  become first-class in the source language and (γ) is forced.

- **Conditional buffer selection emerges as a target kernel pattern.** If
  some future Tier extends to kernels with `ptr = if cond then X else Y`,
  (γ) is forced. Currently no such kernel is in scope.

- **Block-pointer API (`tl.make_block_ptr`/`tl.advance`) becomes a
  separately-verified rewrite target.** If a future paper wants to verify
  that the block-pointer API is itself a correctness-preserving rewrite
  over offset-arithmetic (i.e., the block-pointer abstraction *is* the
  algorithmic content), (γ) becomes the natural choice for that work.

- **Indirect / gather-of-gather access patterns** (pointer arrays).
  Currently P3+; would force (γ) if pursued.

If none of these fire, named regions remain the chosen abstraction.

## 8. Concrete correspondence (for paper appendix)

Side-by-side mapping of the most common Triton patterns:

| Triton | VeriTile DSL | Resulting AST |
|---|---|---|
| `tl.program_id(0)` | `tl.program_id(0)` | `Op.programId` |
| `tl.arange(0, BLOCK)` | `tl.arange($(BLOCK))` or `tl.arange(0, $(BLOCK))` | `Op.arange BLOCK` |
| `pid * BLOCK + tl.arange(0, BLOCK)` | `pid * $(BLOCK) + tl.arange(0, $(BLOCK))` | `Op.add (Op.mul ...) (Op.arange BLOCK)` |
| `tl.load(x_ptr + offsets)` | `tl.load($(xReg) + offs)` | `Op.load xReg (Op.ref "offs")` |
| `tl.load(x_ptr)` (single scalar) | `tl.load($(xReg))` (sugar) | `Op.load xReg (Op.constNat 0)` |
| `tl.store(out_ptr + offsets, value)` | `tl.store($(outReg) + offs, value)` | `Stmt.store outReg (Op.ref "offs") ...` |
| `tl.store(out_ptr, value)` (single scalar) | `tl.store($(outReg), value)` (sugar) | `Stmt.store outReg (Op.constNat 0) ...` |
| `x_ptr + i*M + j` (2D address) | `tl.load($(xReg) + $(i)*$(M) + $(j))` | `Op.load xReg (Op.add (Op.mul ...) ...)` |

Patterns NOT covered (would need (γ)):

| Triton | VeriTile | Status |
|---|---|---|
| `ptr := if cond then x_ptr else y_ptr; tl.load(ptr + offs)` | not expressible | (γ) only; not needed for 8 theorems |
| `ptr_arr[i]` storing pointers | not expressible | (γ) only; not needed |
| `block_ptr := tl.make_block_ptr(...); x := tl.load(block_ptr); block_ptr := tl.advance(block_ptr, ...)` | rewrite to explicit offset arithmetic | (α) rewrite chosen for FA-2 |

## 9. Decision log

- **2026-04-27** — Discussion of pointer semantics surfaced during
  add\_kernel feasibility review. Confirmed: named-region model retained
  through Phases A–D. (β) cosmetic syntax originally deferred to P3+.
  Recorded as RP1.
- **2026-04-27 (later)** — (β) cosmetic syntax landed earlier than
  planned: `tl.load($(R) + offs)` / `tl.store($(R) + offs, v)` and
  scalar-pointer sugar `tl.load($(R))` / `tl.store($(R), v)`. Pure
  macro-time desugaring; no semantic change; (γ) still rejected.

---

*See `PLAN.md` §Out of scope and §Risk register for related framing
decisions (IEEE-754 fidelity, atomics, multi-stream, Python lifter).*
