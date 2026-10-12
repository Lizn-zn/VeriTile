# Decomposing, proving and disproving program equivalence

The two tactics compose syntax derivations using the existing equivalence
notation. Guarded atoms keep their conditions in those syntax fragments:

```lean
import VeriTile.Meta.FPProve
import VeriTile.Triton.Float.ScalarArithmetic
open VeriTile Triton FP.ScalarArithmetic
open scoped VeriTile.Spec

theorem addition_atom (R : Rules) : [lhs .addCommute] ≡[R] [rhs .addCommute] := by
  equiv_decompose
  fp_prove [admitted R .addCommute (by decide)]
```

Both tactics construct proof terms checked by Lean. Lifting this scalar atom
to a program additionally requires its finite conditions at the actual operand
values. [VectorAdd/FPEquiv.lean](../bench/examples/VectorAdd/FPEquiv.lean) uses
the two tactics to apply the checked `GuardedRewrite.add_commute` bridge; a raw report row
must not be rebound to an unguarded program fragment.

## `equiv_decompose`

Import `VeriTile.Meta.EquivDecompose` when only decomposition is needed. This
module imports the generic specification calculus and Lean; it does not import
Triton, an FP model, numerical rules or a report.

The tactic accepts `lhs ≡[R] rhs`, `Spec.ProgramDerivation` and
`Spec.Derivation` goals. For program goals it exposes syntax through the
existing `ProgramSyntax` view. It retains signature and `sameContext`
obligations that cannot be closed by reflexivity or `True.intro`.

For concrete statement-list spines, it finds a longest common subsequence
using definitional equality. Common statements get reflexivity proofs;
changed fragments become goals. Equal-length changed fragments split into
statement pairs by default. Insertions and deletions remain explicit goals
with an empty side. Use `equiv_decompose (split := false)` when an atomic or
derived lemma concerns several adjacent statements together.

For example, changing `u = a + b; v = c * d` to
`u = b + a; v = d * c` produces two statement-derivation goals. Each retains
its precision and complete expression. You can inspect or manually solve these
goals before calling the second tactic.

Decomposition uses the existing context and transitivity constructors. A
changed loop, branch, cast, mask or memory operation stays in its statement;
there is no numerical simplification or unproved congruence through these
constructs. A symbolic list spine remains an unsplit derivation goal. The
generic path handles the sequential syntax calculus. Semantic decomposition
requires a proved adapter, such as the IO adapter described below.

## `fp_prove`

The tactic uses local hypotheses, imported/local `@[spec_rule]` lemmas and
optional hints, for example `fp_prove [my_derived_lemma]`. It instantiates the
lemmas and solves their premises with bounded lemma application. Precision,
admission and domain premises must match or be proved by these inputs.

For syntax derivations, it can additionally compose concrete fragment rewrites
using symmetry, transitivity and common statement contexts. Search defaults
to eight rewrite edges and at most 256 distinct fragment states, with premise
search depth six. `fp_prove (maxSteps := 16) [my_lemma]` changes the edge bound.
Only fully instantiated intermediate fragments enter the search.

Failure preserves the goal and reports that a rule, premise or structural
lemma may be missing, or that the search bound was reached. Failure does not
prove inequivalence. Report rows alone cannot close `EvidenceValidated`, and
the tactic does not add rules or run numerical experiments. Candidate atom
templates and their scoped rule bindings remain explicit Lean definitions.

## Guarded program rewrites

VectorAdd, FlatVectorAdd, FloatDTypeAdd, TritonBenchVectorAddition,
AdamUpdateGridLaunch and the scalar HyperConnectionsDepth/Width examples use
`GuardedRewrite.Program`. Its domain records a shared prefix and the two actual
operand expressions for each rewrite site. `add_commute` and `mul_commute`
derive contextual execution equality from the guarded scalar atoms. The
conditions include products and exponential results when those are operands.
Shapes remain symbolic. This adapter preserves the complete execution result,
including failure; it does not independently prove successful execution.

Import `VeriTile.Triton.Float.RewriteTactics` to use `equiv_decompose` followed
by `all_goals fp_prove`. The semantic adapter retains the program-point domains
and applies the checked bridge lemmas, including explicit fp32 compute tags.
The scalar HyperConnectionsWidth example composes two sites and prints
`mul_commute` once. It uses the same decomposition path as the matrix examples
below, with the source's explicit precision instead of a default fp32 profile.

## General matrix program rewrites

Import `VeriTile.Triton.Float.RewriteTactics` for the `ProfiledRewrite.Program`
adapter. Both general HyperConnections proofs now use:

```lean
specification mhc_width_matrix_equiv (S T D numIters : Nat) (tau : ℝ) (R : Rules) :
    matrixOriginalProgram S T D numIters tau ≡[R]
      matrixOptimizedProgram S T D numIters tau := by
  equiv_decompose
  all_goals fp_prove
```

`equiv_decompose` compares the aligned source statements, carries the shared
signature and declared domains, and exposes one `Contextual` goal per changed
statement. Every goal retains its complete prefix and suffix. Earlier changes
are present in later prefixes, so a second rewrite is checked against the
actual intermediate program. Loops, casts, masks and stores are kept intact.
Different list lengths require an explicit execution lemma in this adapter.

`fp_prove` selects checked contextual arithmetic lemmas and discharges their
premises from the model and the retained site guards. It does not invent
finite/nonzero facts, discard guards, or change a precision tag. Depth prints
only `add_commute`; Width prints only `div_mul_rcp`. Both leave the matrix
products and normalization loops unchanged.

`Site.divScalar` supports a denominator computed by the kernel, such as a
softmax row sum. Its guards check the numerator tile and scalar denominator
after the exact source prefix, including the nonzero requirement. The
`softmax_triton1/FPEquiv.lean` benchmark uses this path with symbolic masks and
strides; its sole numerical assumption is `div_mul_rcp`.

`Site.addScalar` checks both a tile and a scalar after the actual prefix. Its
contextual lemma swaps `tile + scalar` (scalar-right broadcast) to
`scalar + tile` (scalar-left broadcast), using only `add_commute`. The scalar
may be a literal or a computed value. Both operands must be finite; positivity
and nonzero are not required. The `add_value/FPEquiv.lean` benchmark uses this
path with symbolic element count, block size and scalar value. A guard on the
tile alone, a different scalar, a different prefix, or a different compute
precision does not justify the rewrite.

`Site.mulScalarFP32` handles multiplication inside an explicit fp32 compute
expression, from scalar-right to scalar-left broadcasting. It evaluates both guards in
that same fp32 context, including computed scalars, and uses only
`mul_commute`. Missing operand guards or an fp64 compute tag cannot use this
bridge. The complete `attention_score` source uses it for `qk * qk_scale`.

`Site.mulScalar` similarly lifts `mul_commute` from scalar-left to scalar-right
multiplication. Its domain checks both actual operands, including the scalar
interpreted by the fp32 model. The two `triton_mul2/FPEquiv.lean` proofs retain
ordinary and in-place output separately; the in-place rewrite preserves the
store to the input region. A changed store or explicit fp64 multiplication
cannot be discharged by this arithmetic rule.

Import `VeriTile.Triton.Float.BroadcastRewrite` for multiplication between
two broadcast operands, such as a matrix and an expanded per-row vector.
`Site.mulBroadcast` uses the default fp32 context; `Site.mulBroadcastFP32`
uses explicit fp32 computation. Each checks both actual operand tiles after
the common prefix and uses only `mul_commute`. `equiv_decompose` locates the
changed assignment and `fp_prove` applies the corresponding contextual lemma,
preserving loads, casts, subsequent scans, stores and execution failures.
`bench/tests/FPBroadcastMultiplication.lean` checks both precision paths with
mixed input storage and unchanged cumulative scan/writeback.

The `vector_addition_custom/FPEquiv.lean` benchmark keeps its addition directly
inside `tl.store`. `Source.prefixBeforeStore code occurrence` selects a store
(zero-based, including algorithm-wrapped stores), so extra non-store statements
do not shift its index. `contextual_store_add_commute` checks the operands at
that prefix and preserves the entire memory access, mask and dtype. The
numerical rule rewrites the value only; changed destinations, masks or compute
precision remain unproved obligations.

Import `VeriTile.Triton.Float.ExpressionRewrite` as well for rewrites nested
inside profiled expressions. `OpEquivalent` compares expression evaluation after
the actual prefix. Proved congruence lemmas compose rewrites through addition,
subtraction, multiplication, division and `where`, then lift them through an
unchanged assignment or store. These are structural steps, not new FP atoms.
The `where` congruence checks its condition and both eagerly evaluated arms;
it preserves undefined operands and execution failure. Store congruence keeps
the destination, mask and dtype identical.

The `apply_penalty/FPEquiv.lean` benchmark uses `op_div_scalar_mul_rcp` inside
`tl.where`. `equiv_decompose; fp_prove` proves the complete gather/scatter kernel
rewrite and prints only `div_mul_rcp`. Its guards check the actual loaded logits
and repetition scalar at `rep_logits`; the scalar must be finite and nonzero,
while negative operands are allowed. The FP proof preserves the source store
order even for repeated token ids. Independent per-lane real formulas separately
require distinct active scatter addresses. Registered congruence cannot justify
changed conditions, masks or explicit compute precision.

Site definitions use `Source.prefixBefore source "register" occurrence` instead of
statement numbers; occurrences count from zero. Inserting a shared statement
before a rewrite no longer requires renumbering its site. The selected prefix,
operand expression and precision must still match the contextual lemma.
For example, Adam's `exp_avg` rewrite selects occurrence 1: occurrence 0 loads
the old value. A missing assignment name returns the full source prefix and
cannot justify a rewrite at an earlier point. This locator addresses top-level
assignments; it does not search inside loops or branches.

### Rewrites inside dynamic ranges

The `ProfiledRewrite.Program` adapter also descends through matching dynamic
ranges. `Source.dynamicRange? code "i" occurrence` finds the named top-level
loop and returns its prefix, bounds, body and suffix; a missing loop returns
`none`. Apply the register locator to the returned body to describe a local
rewrite without counting statements.

`Site.range before index start stop stride body inner` requires the inner
site's conditions at every reachable active iteration of the original body,
after binding its index. The inner site then runs its own prefix before
checking operands. A condition checked only at loop entry cannot establish
this premise. Zero stride and empty intervals have no active iterations.

`equiv_decompose` applies the proved range congruence when the index and all
bounds match and the declared domain supplies this range site. It then locates
the body's differences, preserving the iteration guard for `fp_prove`. Nested
ranges use nested sites. Missing guards, changed bounds and changed list lengths
remain proof obligations; the tactic does not synthesize an invariant.

The in-progress `attention_fwd_triton1/Proofs/FP.lean` uses this path for its
complete fp32 sources, with arbitrary dimensions and both branch flags. Its
only numerical atom is `mul_commute` at the loaded query/scale multiplication.
The explicit fp32 cast remains opaque on both sides; removing it or changing
the compute precision is not covered. `bench/tests/FPRangeRewrite.lean` checks
these failure boundaries, nested loops, common inserted statements and an
execution whose first operands are finite but whose next operands are not.

## Observable outputs and reduction permutations

Import `VeriTile.Triton.Float.Tactics` to add the existing `KernelIO₁` adapter.
Its `@[equiv_exec]` summaries prove successful runs, output values and memory
frames. The adapter exposes a `TermEq` goal and can use a checked permutation
with scalar commutation/association premises. Precision, casts and padding are
retained. This algebraic interface has unconditional scalar substitution;
it is not a bridge from guarded experimental atoms.

[RowWiseSum/FPEquiv.lean](../bench/examples/RowWiseSum/FPEquiv.lean) therefore
uses `FP.Kernel₁` and explicit guarded tree derivations in `Proofs/FP.lean`. It constructs paths
from the original and reindexed schedules to a common normal form. Its domain
checks every intermediate operand on both paths. Tree normalization justifies
its zero steps using the existing `add_zero` atom. The statement covers symbolic
row sizes, including zero, successful execution and both memory frames.

```text
FP assumptions used by rowwise_sum_equiv:
  add_commute
  add_assoc
  add_zero
```

Run `python3 -m unittest scripts.test_equiv_tactics` for the tactic regressions.
They cover decomposition, composition, missing admissions, retained guards,
precision, casts, masks, the existing output/reduction adapter, and contextual
matrix rewrites with shifted prefixes and multiple sites. Run
`python3 -m unittest scripts.test_fp_guarded_examples` for program-point domains,
intermediate-value rejection and successful execution witnesses.

## Checked refutation with `refutation` and `disprove`

`VeriTile.Meta.Disprove` supplies the `refutation` declaration and the generic
`disprove` tactic, independently of Triton and floating-point models. Write the
original claim after the colon. The declaration negates it; the tactic searches
checked hints and bounded witnesses to construct a proof for Lean's kernel:

```lean
import VeriTile.Meta.Disprove

refutation positive_numbers_are_one : ∀ n : Nat, 0 < n → n = 1 := by
  disprove

refutation all_numbers_below_seven : ∀ n : Nat, n < 7 := by
  disprove [7]
```

`refutation name (x : α) : P x := proof` expands to the ordinary declaration
`theorem name (x : α) : ¬ (P x) := proof`. Parameters and their hypotheses stay
outside the negation. To refute a universal claim, put `∀ x, P x` after the colon.
The generated name has the negated type when used or printed with `#check`;
the original positive claim is never installed as a theorem. The command always
adds exactly one negation, including when the claim already contains `¬`.
Namespaces, section variables, universe parameters, documentation, attributes
and visibility use the normal theorem machinery. The body may use any ordinary
proof term or tactic. `disprove` remains available in `theorem`/`example` bodies
whose goals are already negated, or are `False`; it does not change their types.

Import `VeriTile.Triton.Float.Disprove` for the FP adapters. They support
numerical counterexamples for `FP.Scheduled.IO₁` and the public `FP.Kernel₁`
interface, plus signature mismatches for any `Spec.ProgramSyntax` view.
StableLogSumExp's historical specification uses the declaration and tactic:

```lean
refutation unconditional_disproved_with_log_rules (R : FP.LogSumExp.Rules) :
    stable "x" "y" 1 ≡[R] direct "x" "y" 1 := by
  disprove [Model.models R, Model.domain_holds, Model.stored_outputs]
```

The three hints prove that the interpretation satisfies the complete admitted
theory, that the input meets the kernel's domain, and that execution produces
the stated stored outputs. The tactic infers the interpreter and input from
these facts, searches an in-bounds output lane, and derives a contradiction
through the public equivalence relation. It preserves the precision profile,
casts, schedules, domains and required comparison support. Execution facts must
refer to the actual kernels; scalar formulas alone cannot discharge the adapter.
The old example-specific `Model.not_equiv` proof is no longer needed.

Hints can also contain explicit model/input/schedule values when erased or
constant preconditions do not determine them. Local hypotheses are searched
automatically, so `disprove` alone works when the necessary facts are already
in context. Conjunctions are expanded and bounded universal hints are
instantiated. An equal output at the first lane causes backtracking to later
lanes. Every finite-index bound is checked.

The tactic uses supplied or registered models; it does not synthesize arbitrary
FP algebras. `@[disprove_hint]` registers reusable facts or witnesses, and
`@[disprove_rule]` registers proved refutation adapters for additional relations
or program interfaces. These attributes are global within the importing module
environment. Additional numerical program views need their own sound adapter;
their source code is not silently interpreted through a different contract.

Search is bounded by `(maxDepth := 24)`, `(maxNodes := 512)` and
`(maxWitnesses := 4)` by default. The last option enumerates natural numbers
and valid `Fin` indices starting at zero; Boolean and unit candidates are also
available. Explicit hints can provide witnesses beyond that range. Failed
branches restore their metavariable assignments. Failure reports that no
checked counterexample was found; neither missing admission nor failure of
`fp_prove` is treated as a disproof. `trace.VeriTile.disprove` exposes the search
obligations when debugging.

Run `python3 -m unittest scripts.test_disprove` for declaration polarity and
scoping, generic search, independent kernel counterexamples,
cast/domain/comparison checks and the StableLogSumExp
integration. The result is a disproof in the stated abstract theory, not a GPU
measurement or a statistical rejection by two-gates.
