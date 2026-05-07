# Examples Theorem Surface Style

User-facing theorem surfaces in `VeriTile/Examples/` should start from the
public compute-facing APIs. See
[`CorrectnessSurfaces.md`](./CorrectnessSurfaces.md) for the full user guide.

- Single-kernel correctness against a mathematical or algorithmic spec uses
  `ComputeCorrect.Output*`, `ComputeCorrect.Post`, or
  `ComputeCorrect.General`.
- Two-kernel equivalence or rewrite refinement uses
  `ComputeRefine.Output*Eq`, `ComputeRefine.Post`, or
  `ComputeRefine.General`.

Projected algorithm lemmas may still mention `Kernel.Correct` or
`Kernel.Refine` when they are explicitly internal bridge lemmas. Those lemmas
should not be the exported example theorem named in
`scripts/kernel-manifest.tsv`.

## Naming

- Single-kernel correctness theorem: `<name>_correct_view`
- Two-kernel refinement theorem: `<name>_refinement_view`

Execution-only helper lemmas may use an `_exec_view` suffix and can state direct
`exec` equalities. The public theorem should wrap that helper in
`ComputeCorrect.*` or `ComputeRefine.*`.

Domain-specific theorem surfaces that are not ordinary single-kernel or
two-kernel example views, such as whole-grid launch facts or specialized
FlashAttention math/trace statements, may keep established descriptive names.
When a public artifact theorem is renamed to follow this style,
`scripts/kernel-manifest.tsv` must be updated in the same commit.
