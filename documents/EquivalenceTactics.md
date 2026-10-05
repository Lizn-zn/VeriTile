# Decomposing and proving program equivalence

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
the checked `GuardedRewrite.add_commute` bridge for that step; a raw report row
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
AdamUpdateGridLaunch, HyperConnectionsDepth and HyperConnectionsWidth use
`GuardedRewrite.Program`. Its domain records a shared prefix and the two actual
operand expressions for each rewrite site. `add_commute` and `mul_commute`
derive contextual execution equality from the guarded scalar atoms. The
conditions include products and exponential results when those are operands.
Shapes remain symbolic. This adapter preserves the complete execution result,
including failure; it does not independently prove successful execution.

These examples use explicit composition of the checked bridge lemmas. The
syntax tactic cannot discharge program-point domains by discarding them.
HyperConnectionsWidth composes two sites and prints `mul_commute` once.

## Observable outputs and reduction permutations

Import `VeriTile.Triton.Float.Tactics` to add the existing `KernelIO₁` adapter.
Its `@[equiv_exec]` summaries prove successful runs, output values and memory
frames. The adapter exposes a `TermEq` goal and can use a checked permutation
with scalar commutation/association premises. Precision, casts and padding are
retained. This algebraic interface has unconditional scalar substitution;
it is not a bridge from guarded experimental atoms.

[RowWiseSum/FPEquiv.lean](../bench/examples/RowWiseSum/FPEquiv.lean) therefore
uses `Scheduled.IO₁` and explicit guarded tree derivations. It constructs paths
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
precision, casts, masks, and the existing output/reduction adapter. Run
`python3 -m unittest scripts.test_fp_guarded_examples` for program-point domains,
intermediate-value rejection and successful execution witnesses.
