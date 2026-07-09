# Examples Theorem Surface Style

User-facing theorem surfaces in `VeriTile/Examples/` should start from the
public compute-facing APIs. See
[`CorrectnessSurfaces.md`](./CorrectnessSurfaces.md) for the full user guide.

- Single-kernel correctness against a mathematical or algorithmic spec uses
  `ComputeCorrect.Realizes` (*a kernel realizes a spec*), `ComputeCorrect.Post`,
  or `ComputeCorrect.General`.
- Two-kernel equivalence or rewrite refinement uses `ComputeRefine.Refines`
  (*a kernel refines another* — writes-equality on the two final memories
  outside declared scratch regions), the pointwise `ComputeRefine.RefinesAt`,
  `ComputeRefine.Post`, or `ComputeRefine.General`.
- The default `ComputeCorrect.Realizes` / `ComputeRefine.Refines` /
  `RefinesAt` surfaces above are the rounding surfaces, parametric over every
  `RoundingModel`; the exact-ℝ idealizations are their `*_without_Rounding`
  mirrors, which degenerate out of them at the trivial model. Narrow-float
  showcase kernels land on the unqualified rounding surface directly; see
  [`CorrectnessSurfaces.md`](./CorrectnessSurfaces.md) and the showcase
  `bench/examples/FusedSwiglu.lean`.

Projected algorithm lemmas may still mention `Kernel.Correct_without_Rounding`
or `Kernel.Refine` when they are explicitly internal bridge lemmas. Those lemmas
should not be the exported example theorem named in
`scripts/kernel-manifest.tsv`.

## Naming

- Single-kernel correctness theorem: `<name>_correct_view`
- Two-kernel refinement theorem: `<name>_refinement_view`

Execution-only helper lemmas may use an `_exec_view` suffix and can state direct
`exec` equalities. The public theorem should wrap that helper in
`ComputeCorrect.Realizes` (single-kernel spec) or `ComputeRefine.Refines`
(two-kernel writes-equality) when it is an output observation theorem.

Domain-specific theorem surfaces that are not ordinary single-kernel or
two-kernel example views, such as whole-grid launch facts or specialized
FlashAttention math/trace statements, may keep established descriptive names.
When a public artifact theorem is renamed to follow this style,
`scripts/kernel-manifest.tsv` must be updated in the same commit.
