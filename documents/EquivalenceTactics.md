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

The seven statement-rewrite examples use these tactics: VectorAdd,
FlatVectorAdd, FloatDTypeAdd, TritonBenchVectorAddition, AdamUpdateGridLaunch,
HyperConnectionsDepth and HyperConnectionsWidth. The last decomposes two
rewrite sites and still prints `mul_commute` once.

## Observable outputs and reduction permutations

Import `VeriTile.Triton.Float.Tactics` to add the numerical-model adapter. The
generic tactic modules remain independent of Triton and FP semantics.
[RowWiseSum/FPEquiv.lean](../bench/examples/RowWiseSum/FPEquiv.lean) uses it:

```lean
specification rowwise_sum_equiv (nCol B : Nat) (R : Rules) :
    originalIO nCol B ≡[R] reversedIO nCol B := by
  equiv_decompose
  all_goals fp_prove
```

Here `equiv_decompose` uses the `KernelIO₁` numerical contract. Each kernel
supplies a proved `@[equiv_exec]` execution summary: a successful run, the value
written to its output cell, and preservation of every other memory cell.
These are proofs about each implementation, without an equivalence assumption.
The tactic instantiates their inputs from the common initial state and exposes
the relation between their output values. Signature, private-scratch and output
size obligations remain in the proof; they cannot be dropped. The adapter
currently supports one output cell per program, with empty scratch discharged
automatically. It does not synthesize execution summaries for arbitrary kernels.
The explicit `(split := false)` form still selects syntax decomposition.

In this example the remaining `TermEq` goal compares fp32 sums of the original
and reversed vectors of loaded values. `fp_prove` applies the generic reduction
permutation theorem, discharging its scalar addition commutation and association
premises from the admitted rules. This works for symbolic `B`, including zero,
and arbitrary valid reduction schedules. The inputs and any padding zeros keep
their multiplicities. The proof does not call a completed kernel-equivalence
theorem or introduce a whole-reduction atom.

Reduction search tries the identity, reversal, and explicitly supplied or local
permutations (also their inverses). It must prove that the actual input vectors
are related by the chosen permutation. For another index map, supply a typed
`Equiv.Perm (Fin B)` and the input-vector equation. Supported sums have fp32
arithmetic and a one-dimensional input. Common opaque operations, including
loads and casts, can surround the sums; congruence preserves their exact labels
and arguments. Differing casts or precision are not erased. `maxSteps` bounds
this recursive congruence search; the default is eight, with premise depth six.

The resulting printer output remains:

```text
FP assumptions used by rowwise_sum_equiv:
  add_assoc
  add_commute
```

More involved multi-output and loop proofs retain their explicit derivations.

Run `python3 -m unittest scripts.test_equiv_tactics` for the focused regressions.
They cover multi-site decomposition, insertions/deletions, block preservation,
symbolic lists, directed and reversed chains, contextual search, metadata,
domain premises, missing admissions, precision, casts and active-lane masks.
The reduction fixture also checks renamed registers and buffers, an extra
assignment, in-place output, empty rows, arbitrary permutations, missing scalar
rules, duplicated or omitted inputs, extra zeros and changed output contracts.
