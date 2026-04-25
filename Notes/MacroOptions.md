# Macro Options for `triton { ... }` Embedding

**Status:** Tech investigation, P2 planning.
**Audience:** VeriTile implementation team (mostly: future me).
**Date:** 2026-04-25.

---

## The problem

The proposal (§3.3, §11) shows lifted Triton kernels in a `triton { ... }`
syntactic form, e.g.

```lean
def naive_softmax_kernel : TritonKernel := triton {
  @T.jit
  def main(X: Ptr float32, Y: Ptr float32, N: tl.constexpr):
      pid = tl.program_id(0)
      ...
}
```

That form is aspirational. The skeleton committed today (`Examples.naiveSoftmax`)
uses **direct constructor calls** (`.assign "pid" .programId`, etc.) — no
syntactic sugar. This note works through the realistic options for going
from the latter to the former, and recommends an approach.

## What are we actually optimizing for

Important to be clear-eyed: **the user does not write embedded Triton.**
The user writes `.py` Triton; the lifter produces the Lean term. So the
audience for `triton { ... }` syntax is:

1. **The lifter** (Python emitting Lean source). Cares about: predictable,
   easy-to-emit syntax. Does *not* care about brevity for humans.
2. **Test/example files** (hand-written by VeriTile contributors).
   Cares about: readability when debugging, ability to copy-paste from
   real Triton code with minimal edits.
3. **Reviewers / readers of the proposal and paper.** Cares about: visual
   resemblance to actual Triton, so the formalization "looks right."

(2) and (3) are real but secondary. (1) dominates. So a verbose-but-mechanical
form is acceptable; a beautiful-but-fragile parser is not.

## Option A — No macro, raw constructors

Just use `.const`, `.add`, `.assign`, `.store` as Lean would natively.
This is what `Examples.naiveSoftmax` currently does.

```lean
def naiveSoftmax (N : Nat) : Kernel where
  inputs  := ["X"]
  outputs := ["Y"]
  body    := [
    .assign "pid" .programId,
    .assign "offs"
      (.add (.broadcast (.mul .programId (.const (N : ℝ))) N) (.arange N)),
    .assign "m"   (.reduceMax (.ref "x")),
    ...
  ]
```

**Pros:**
* Zero meta-programming work. Writes itself.
* Lifter trivially emits this — it's just constructor application.
* Errors are reported by Lean's normal type checker against `Op` / `Stmt`.
* Refactoring `Op` / `Stmt` automatically propagates.

**Cons:**
* Visually noisy. Triton's `pid * N + tl.arange(0, N)` becomes a deep
  `.add (.broadcast (.mul ...) N) (.arange N)`.
* No parens-saving precedence; everything is fully parenthesized.
* Can't directly copy-paste real Triton into Lean test files.

**Effort:** 0 days. Already done.

## Option B — Lean notation for individual operators

Add Lean `notation`/`infixl` declarations so that arithmetic constructors
look like infix operators. Plus helper functions for the noisier constructors.

```lean
namespace VeriTile.Triton.Notation

scoped infixl:65 " ⊕ " => Op.add
scoped infixl:65 " ⊖ " => Op.sub
scoped infixl:70 " ⊗ " => Op.mul
scoped infixl:70 " ⊘ " => Op.div

scoped notation:max "$" name => Op.ref name
scoped notation:max "ℝ" => fun (c : ℝ) => Op.const c

scoped notation:max "load[" r ", " o "]" => Op.load r o
scoped notation:max "max⟨" e "⟩"  => Op.reduceMax e
scoped notation:max "sum⟨" e "⟩"  => Op.reduceSum e
scoped notation:max "broadcast(" e ", " n ")" => Op.broadcast e n
scoped notation:max "bcast " n " of " e => Op.broadcast e n

end VeriTile.Triton.Notation
```

The example then reads:

```lean
open VeriTile.Triton VeriTile.Triton.Notation in
def naiveSoftmax' (N : Nat) : Kernel where
  inputs  := ["X"]
  outputs := ["Y"]
  body    := [
    .assign "pid" .programId,
    .assign "offs" (bcast N of (Op.programId ⊗ Op.const N) ⊕ Op.arange N),
    -- ...
  ]
```

**Pros:**
* Modest improvement in readability.
* Zero new meta machinery — `notation` is built into Lean 4 since v4.0.
* Lifter trivially emits this (slightly shorter strings).
* Scoped, so it doesn't pollute the global namespace.

**Cons:**
* Doesn't solve the bigger problem: control flow (`forLoop`, `assign`,
  `store`) still looks like Lean, not Triton.
* Custom Unicode operators (⊕ ⊗) can be cute but trade obscurity for
  brevity. Triton-style `+`/`*` would clash with Lean's own — we'd need
  `Op` to live in its own namespace and only "look like" arithmetic when
  unambiguous.
* Still can't paste real Triton.

**Effort:** ~1 day to design + write notation declarations. Mostly safe
because `notation` is non-invasive.

## Option C — Custom syntax block (`triton { ... }`)

Use Lean 4's macro / elaborator infrastructure to parse a Python-like
block as a `Stmt list` or `Kernel`. This is the path the ARM-in-Lean
example takes.

The shape:

```lean
syntax (name := tritonStmt) "triton " "{" tritonBody "}" : term
syntax tritonBody := (tritonStmt)*
syntax tritonStmt :=
  | tritonAssign
  | tritonStore
  | tritonFor

syntax tritonAssign := ident " = " tritonExpr
syntax tritonStore  := "tl.store" "(" ident " + " tritonExpr ", " tritonExpr ")"
syntax tritonFor    := "for " ident " in range(" num "):" tritonBody

-- ... and tritonExpr has its own grammar covering Op constructors.

@[term_elab tritonStmt] def elabTritonStmt : Elab.Term.TermElab := ...
```

The elaborator walks the parsed syntax tree and emits constructor applications
for `Op` / `Stmt` / `Kernel`.

**Pros:**
* Actually looks like Triton. You can copy-paste large chunks from the user's
  `.py` file straight into a `.lean` test fixture.
* Lifter can emit verbatim Triton inside the braces — no constructor mapping.
* Most impressive for paper figures and reviewer visual.

**Cons:**
* **Big upfront cost.** Lean 4 macro/elaborator work is its own skill;
  even the relatively small ARM-in-Lean DSL is a few hundred lines of
  meta code. Triton's surface, even our P1 subset, has ~15-20 forms.
  Realistic estimate: **2-3 weeks for a working subset, ~1 month for
  reasonable coverage**.
* Error messages get worse. Macro expansion errors are notoriously
  opaque; users debugging a parse failure inside `triton { ... }`
  see synthetic line numbers instead of their source.
* Maintenance overhead: every time we extend the `Op` / `Stmt` types
  (and we will, throughout P3-P7), the macro needs corresponding
  parser cases. Constructor calls auto-update; macro parsers do not.
* **Triton syntax has Python-isms our macro can't replicate.** Indentation
  blocks, augmented assignment (`d *= ...`), function definitions inside
  `@triton.jit` decorators, default argument values. The macro must
  define its own simpler grammar that *resembles* Triton without being
  Triton — confusing in subtle ways.
* Lifter still has to convert real `.py` Triton to whatever our macro
  parses, so the macro doesn't reduce lifter complexity.

**Effort:** 2-4 weeks. Real risk of slipping.

## Option D — External format (S-expression / JSON) parsed at compile time

Lifter emits a separate `.triton.sexp` (or `.json`) file. A Lean term
elaborator reads it at compile time and produces the `Kernel` value.

```lean
def naiveSoftmaxKernel : Kernel := loadKernelFromFile "naive_softmax.triton.sexp"
```

**Pros:**
* Decouples the lifter's output format from Lean syntax entirely. The
  external format can be anything we want (S-expr is the natural choice
  for Lisp-style ASTs).
* Lean side is small: just a file reader + AST → constructor mapping.
* The `.triton.sexp` files can be inspected, diffed, and regenerated
  without touching Lean files.

**Cons:**
* Hides the kernel inside a separate file, which hurts readability
  in proposal / paper / docs (you can't quote the kernel inline).
* Build-time file I/O via `compileTimeIO` or similar is supported but
  fiddly in Lean 4; may not play nicely with Lake's incremental build.
* Test fixtures become two-file: the `.lean` test plus the `.sexp` data.
  Annoying for hand-written tests.

**Effort:** 3-5 days. Lean-side reader is simple, Lake integration is
where time goes.

## Comparison table

| Criterion | A: raw | B: notation | C: macro block | D: external file |
|---|---|---|---|---|
| Lifter complexity (Python side) | trivial | trivial | medium (must emit valid macro) | trivial (emit S-expr) |
| Lean meta complexity | none | tiny (~1 day) | high (2-4 wks) | low-medium (3-5 days) |
| Readability of test fixtures | poor | OK | excellent | poor (split file) |
| Readability in paper | poor | medium | excellent | poor |
| Maintenance with `Op` schema changes | auto | auto | manual | auto (S-expr is structural) |
| Error messages | Lean-native (good) | Lean-native (good) | macro-elaboration (poor) | parse-time (medium) |
| Risk of "bikeshedding the surface" | none | low | high | low |
| Time to first working version | 0 | 1 day | 2-4 weeks | 3-5 days |

## Recommendation

**Phase ordering:** A → B → either C or D as a P5+ optional polish.

* **P1-P4: stay on Option A.** The skeleton committed today already does
  this. Constructor calls are ugly but *will work*; we control 100% of
  the shape; refactoring is automatic. Spending P2 on macro work delays
  the actual research (operational semantics + first proof). The pretty
  syntax is irrelevant when the entire goal is "can Lean kernel-check
  the proof at all."

* **P5 (after first end-to-end demo): add Option B.** Once the core
  semantics is stable and we know what shapes recur (e.g., we
  consistently write `pid * N + arange N` for offsets), introduce
  scoped notation for those patterns. Cheap, scoped, removable.

* **P6+ (if reviewers complain or we want a public surface): consider
  Option C or D.** Default to **D (external file)** — lower risk, less
  meta investment, easier to evolve as Op schema changes. Only do
  C if we have explicit demand for in-line Triton in test files.

**Anti-recommendation:** Do not attempt Option C in P1-P4. The macro
hype-cycle is real and seductive — "let's make it look like Triton!" — but
it's a side quest. The research bottleneck is "can the LLM produce a
proof that Lean accepts," not "does the source look pretty in the paper."

## What this means for the lifter (Python)

The lifter's output format follows from this recommendation:

* **P1-P5:** Emit Python-string Lean source using direct constructor
  calls. Roughly:

  ```python
  def emit_op(op_ast) -> str:
      if isinstance(op_ast, Const):
          return f".const ({op_ast.value} : ℝ)"
      elif isinstance(op_ast, BinOp):
          return f"(.{op_ast.op} {emit_op(op_ast.lhs)} {emit_op(op_ast.rhs)})"
      ...
  ```

  Output goes into `VeriTile/Generated/<KernelName>.lean`.

* **P6+ (if we add D):** Lifter emits `.triton.sexp` instead. Same
  structure, just different surface.

## Existing references / prior art

* **arm-in-lean** (the user's earlier example): full Option C for ARM64
  assembly. ~1000+ lines of meta. Worth reading for pattern; their
  scope is smaller (no Python indentation rules, no `@decorator`).
* **Verso**: Lean 4 documentation framework with custom syntax. Heavy
  meta. Overkill for our needs but has good examples of `syntax` /
  `elab` patterns.
* **Mathlib's `calc`**: built-in equational reasoning syntax. Clean
  example of moderate-complexity macro.
* **Lean 4 Manual, "Macros and Elaboration"**: official reference,
  https://lean-lang.org/lean4/doc/macros.html .
* **`leanprover/lean4-samples`** repo: small DSL examples.

## Open questions

* Does our P1 subset have any Python-isms that *won't* survive Option C
  cleanly? (Top suspects: `tl.constexpr` annotations, `for i in range(N)`
  vs `for i in tl.serial(N)` distinction.)
* If we end up with two surface forms (raw + notation), do we want both
  to be supported simultaneously, or migrate?
* For the lifter's emitted Lean files: regenerated each build, or
  checked-in artifacts? (Recommendation: regenerated, but check in
  one example as a known-good fixture.)

---

**Bottom line:** start simple, escalate only when there's a concrete
research/UX reason. Option A is committed; Option B in P5; nothing in
P2.
