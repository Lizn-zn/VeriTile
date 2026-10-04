# Decomposing and proving program equivalence

The two tactics keep the existing specification and assumption printer:

```lean
import VeriTile.Meta.FPProve

specification vector_equiv (B : Nat) (R : Rules B) :
    originalKernel B ≡[R] optimizedKernel B := by
  equiv_decompose
  all_goals fp_prove

#print_fp_assumptions vector_equiv
```

The real, compilable example is
[VectorAdd/FPEquiv.lean](../bench/examples/VectorAdd/FPEquiv.lean).
Both tactics construct ordinary proof terms checked by Lean. Experimental
admission remains a scoped assumption supplied by `R`.

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
tactic handles the generic sequential syntax calculus; new semantic views
must supply their own justified structure through `ProgramSyntax`.

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

The seven statement-rewrite examples use these tactics: VectorAdd,
FlatVectorAdd, FloatDTypeAdd, TritonBenchVectorAddition, AdamUpdateGridLaunch,
HyperConnectionsDepth and HyperConnectionsWidth. The last decomposes two
rewrite sites and still prints `mul_commute` once. More involved execution,
reduction and loop proofs retain their existing explicit derivations.

Run `python3 -m unittest scripts.test_equiv_tactics` for the focused regressions.
They cover multi-site decomposition, insertions/deletions, block preservation,
symbolic lists, directed and reversed chains, contextual search, metadata,
domain premises, missing admissions, precision, casts and active-lane masks.
