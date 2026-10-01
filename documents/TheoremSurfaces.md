# Examples Theorem Surface Style

Correctness against a formula uses real semantics (`Spec.Real claim`).
Implementation equivalence uses `lhs ≡[R] rhs` (`open scoped VeriTile.Spec`):
two-gates admits the atomic assumptions and Lean checks their formal composition.
The rule model `R` holds the experiment configuration and admission records;
those records are not separate public theorem parameters.
Both use `specification`; `#print_spec` exposes the implementations and atom
records. See [CorrectnessSurfaces.md](./CorrectnessSurfaces.md).

Existing unwrapped/abstract-rounding headlines retain their historical meaning;
they are not automatically admitted FP rules. Exact FP facts remain internal
helper lemmas. The library vocabulary below describes legacy mathematical and
abstract-rounding reasoning, which is distinct from the new equivalence calculus.

User-facing theorem surfaces in `VeriTile/Examples/` should start from the
public compute-facing APIs. See
[`CorrectnessSurfaces.md`](./CorrectnessSurfaces.md) for the full user guide.

- **Bench-corpus headlines** (`bench/tritonbench_g/`, `bench/examples/`) are
  stated on a `KernelIO` `⊨` / `⊨[R]` face, not on the surfaces below; that
  triple bundles the addressing, the output values and the frame into one line.
  See [`CorrectnessSurfaces.md`](./CorrectnessSurfaces.md) and
  [`../bench/MAIN_THEOREM_CONVENTIONS.md`](../bench/MAIN_THEOREM_CONVENTIONS.md) §4.
- Single-kernel correctness against a mathematical or algorithmic spec uses
  `ComputeCorrect.Realizes_without_Rounding` (*a kernel realizes a spec*), `ComputeCorrect.Post`,
  or `ComputeCorrect.General`.
- Two-kernel equivalence or rewrite refinement uses `ComputeRefine.Refines`
  (*a kernel refines another* — writes-equality on the two final memories
  outside declared scratch regions), the pointwise `ComputeRefine.RefinesAt`,
  `ComputeRefine.Post`, or `ComputeRefine.General`.
- Rounding-aware single-kernel claims use `ComputeRefine.Realizes`, whose
  expected outputs depend on `RoundingModel`. Two-kernel claims use
  `ComputeRefine.Refines R` or `RefinesAt R` with an explicit model.
  These abstract cast/store contracts do not imply full IEEE-754 semantics.

Projected algorithm lemmas may still mention `Kernel.Correct_without_Rounding`
or `Kernel.Refine` when they are explicitly internal bridge lemmas. Those lemmas
should not be the exported example theorem named in
`scripts/kernel-manifest.tsv`.

## Naming

- Single-kernel correctness theorem: `<name>_correct_view`
- Two-kernel refinement theorem: `<name>_refinement_view`

Execution-only helper lemmas may use an `_exec_view` suffix and can state direct
`exec` equalities. The public theorem should wrap that helper in
`ComputeCorrect.Realizes_without_Rounding` (single-kernel spec) or `ComputeRefine.Refines`
(two-kernel writes-equality) when it is an output observation theorem.

Domain-specific theorem surfaces that are not ordinary single-kernel or
two-kernel example views, such as whole-grid launch facts or specialized
FlashAttention math/trace statements, may keep established descriptive names.
When a public artifact theorem is renamed to follow this style,
`scripts/kernel-manifest.tsv` must be updated in the same commit.
