# Fusion Correctness of Multiple Kernels

Resolves
[Define 'fusion correctness of multiple kernels' + pick a running fusion example](https://github.com/Lizn-zn/VeriTile/issues/461)
(property **P2** of the v1 destination, charted by the
[wayfinder map](https://github.com/Lizn-zn/VeriTile/issues/458)).
Decided 2026-07-13 via `/grilling` + `/domain-modeling`. This is a *design*
document: it fixes the definition, the theorem surface, and the running
example. The Lean implementation lands under the follow-up tickets in §7.

---

## 1. The definition

> A fused kernel `C` is **fusion-correct** with respect to stage kernels
> `[A, B, …]` iff `C` exactly implements the compound behavior "run A, then
> B, then …" **on the declared outputs, at the algorithm (ℝ) layer**.

Concretely, as a theorem surface (thin wrapper, follow-up ticket lands it):

```lean
/-- The sequential composition of a stage pipeline: one kernel whose body
    is the concatenation of the stages' bodies. -/
def seqCompose : List ComputeKernel → ComputeKernel
-- with: exec (seqCompose ks) s = ks.foldlM (fun s k => exec k s) s
--       (proved once from the existing `stepStmts_append`)

/-- `fused` exactly implements the pipeline of `stages` on the declared
    outputs (ℝ layer). Definitionally thin over `ComputeRefine.Realizes`. -/
def ComputeRefine.FusionCorrect {ι α : Type} [ComputeCorrect.OutputReadable α]
    (fused : ComputeKernel) (stages : List ComputeKernel)
    (s : BlockState) (write : ComputeCorrect.WriteMap ι) : Prop :=
  ComputeRefine.Realizes fused (seqCompose stages) s
    write write (fun (_ : ι) (x y : α) => x = y)

@[simp] theorem fusionCorrect_iff …   -- the standard unfolding step
```

**The obligation:** whenever both `fused` and `seqCompose stages` execute
successfully from the same per-program initial state, every declared output
cell of the fused kernel is **exactly equal** to the corresponding cell
produced by the stage pipeline, under the real-valued semantics.

This is **kernel-vs-kernel refinement** — the golden reference in a fusion
proof is the *stage pipeline itself*, not a math spec. Spec-equivalence of
the fused kernel against a composed mathematical spec remains available, but
it is property **P1** applied to the fused kernel; it is not what "fusion
correctness" names.

## 2. The decisions, with rationale

Each was put to the collaborator during grilling; recorded here in
dependency order.

### D1 — Core shape: refinement against the stage pipeline

`C` implementing "A then B" is inherently a two-kernel statement. It is what
both existing fused examples already prove (`FusedSiLU` on `main`, the
SwiGLU pilot), it matches the harness's golden-vs-optimized story
([#464](https://github.com/Lizn-zn/VeriTile/issues/464)), and it is
meaningful even for stages with no standalone math spec.

### D2 — Observation: declared outputs only (pointwise write-maps)

Fusion correctness promises **output consistency**, nothing more. The
comparison is pointwise at declared output addresses
(`ComputeCorrect.WriteMap`), exactly `main`'s existing
`ComputeRefine.Realizes` form. Whole-memory equality is *not* part of the
definition: the stage pipeline materializes intermediate buffers that the
fused kernel never writes, so the final memories provably differ there, and
per-address framing is neither important nor necessary for this property.
Framing statements stay where the existing design puts them — separate
frame/preserve theorems (`Memory/Frame/*`).

### D3 — Precision: ℝ-equality only; FP analysis is a separate layer

At the algorithm layer, fused and unfused outputs are **exactly equal** —
unconditionally provable. Under a nontrivial rounding model they are
**provably different**: the pipeline rounds at the intermediate store/load
round-trip (e.g. a bf16 scratch buffer) and fusion removes that rounding
site — that removal is the whole selling point. The definition therefore
asserts *only* ℝ-equality. Whether the eliminated cast/store/load made the
FP-vs-ℝ gap larger or smaller is the job of the FP-analysis layer
(property **P4**, [FpWarningCatalog.md](./FpWarningCatalog.md)), which
consumes the `#447` rounding-event ledgers of both sides
(fused: `SingleRounding`; pipeline: double rounding through the intermediate
dtype). This mirrors the repo's load-bearing two-layer split
([ArchitectureHandoff.md](./ArchitectureHandoff.md) §1).

### D4 — Composition: a first-class N-ary `seqCompose`

"A then B" must exist as a Lean object for the definition to mention.
`seqCompose` is body concatenation, N-ary from day one, with the single
packaged lemma `exec_seqCompose` proved from the existing
`stepStmts_append`. It names the pattern each fused example currently
hand-rolls (`FusedSiLU` proves its own `exec_unfusedSiLUKernel`; the SwiGLU
pilot re-proves the same splitting for its pipeline) so no future example
repeats that boilerplate.

**Stated convention:** concatenation means a later stage executes with the
earlier stage's final *register file* (real launches reset registers; only
memory flows). This is benign for kernels that only read registers they
wrote — every real stage begins with `tl.load` — and is the same convention
`FusedSiLU` already uses. It is a documented property of `seqCompose`, not a
side condition of the definition.

### D5 — Launch scope: per-program, same grid (v1)

The definition fixes one `BlockState`: fused and pipeline run under the same
program-id assignment. This covers the v1 target class — fusions that
preserve tiling (SwiGLU, SiLU pipeline, epilogue fusions). Grid-reshaping
fusion (e.g. LayerNorm+matmul, whose stages disagree on grid shape) waits
for the whole-grid launcher (`launchExec`, a tracked gap); per
[CorrectnessSurfaces.md](./CorrectnessSurfaces.md), the grid quantifier will
wrap the per-program core without changing the theorem shape.

### D6 — Surface: a named, definitionally-thin wrapper

`ComputeRefine.FusionCorrect` follows the existing ergonomic-wrapper
convention (`OutputTileEq` et al. are exactly such wrappers over
`Realizes`): it adds zero proof burden (`fusionCorrect_iff` unfolds it, and
all existing lemma libraries and the `prove.sh` loop apply unchanged) while
giving the manifest, the harness, and the Tilelang frontend (which inherits
property surfaces, [TilelangMapping.md](./TilelangMapping.md)) one stable
identifier for property P2.

### D7 — Manifest: new `fusion` kind

Fusion theorems are registered in `scripts/kernel-manifest.tsv` under a new
kind `fusion` (schema checker gains one allowed value; existing rows
untouched), so property-P2 coverage is directly auditable.

## 3. Running example

**Headline: SwiGLU** — `swiglu_fused` vs `[silu_step, mul_step]`.
Two stages, one intermediate bf16 buffer; the eliminated store/load rounding
site is the canonical input for the P4 FP-analysis layer. Component kernels
and proofs adapt from the `feat/447-swiglu-pilot` branch (§5).

```lean
theorem swiglu_fusion_view :
  ComputeRefine.FusionCorrect swiglu_fused [silu_step, mul_step] s write
```

**N-ary instance: FusedSiLU retrofit** — `fusedSiLU` vs
`[gate_step, silu_step, residual_step]`. Already proven on `main` in the
pre-`seqCompose` style (`silu_kernels_refinement_view`); restating it on
`FusionCorrect` is nearly free and demonstrates the definition is not shaped
around a single two-stage example. The existing theorem remains valid.

Considered and not chosen: attention epilogue (stage kernels don't exist;
turns a definition ticket into a proof effort — it remains a natural later
target on the FlashAttention validation path) and LayerNorm+matmul
(grid-reshaping; excluded by D5 until `launchExec`).

## 4. Compatibility with the existing design

The definition was audited decision-by-decision against
[CorrectnessSurfaces.md](./CorrectnessSurfaces.md) under the constraint
*"v1 builds on the existing implementation on `main`"*:

- D1–D3, D5 **are** the existing design (two-kernel equality helpers;
  pointwise `Realizes` with framing delegated to frame theorems; Real-first
  proofs with `GapPolicy.ignore`; per-program grid statements with the
  documented launcher-wrapping plan).
- D4, D6, D7 are **purely additive** (~40–60 lines of new definitions + one
  manifest-schema value). No existing surface changes name or meaning; the
  ~151 manifest theorems and all bench ports are untouched.

## 5. Disposition of `feat/447-swiglu-pilot`

The pilot branch redefines `ComputeRefine.Realizes` as whole-memory
equality outside a `scratch` list (renaming the pointwise form to
`RealizesAt`). That naming move contradicts D2, and the branch predates the
`#472` layout relocation, so it is **quarried, not merged**:

- **Ported** (re-homed, restated pointwise where needed):
  `ComputeKernel.ExecRefineR` and the pointwise `ComputeRefine.RefinesAtR`
  (the two-kernel `∀R` surfaces `main` lacks — needed by the P4 layer),
  the SwiGLU stage kernels, their component `RealizesR` proofs, and the
  spec algebra (`fusedSpec_ne_unfusedSpec`, the event ledgers).
- **Dropped:** the whole-memory `Realizes` redefinition and the headline
  theorems stated on it. If a whole-memory framing lemma later proves
  useful it may return under a *new* name as an optional strengthening —
  never as the fusion definition.

## 6. What this definition deliberately does not say

- Nothing about memory outside the declared outputs (D2) — framing is a
  separate concern.
- Nothing about FP magnitudes, and no `∀ R` equality claim — false in
  general; the rounding-ledger *difference* is P4's input (D3).
- Nothing about grid-reshaping fusion (D5) — out of v1 scope pending
  `launchExec`.
- No transitivity/vertical-composition theory of `Refine` — not needed for
  the v1 obligation; revisit only if a real target demands stacked
  refinements.

## 7. Follow-up tickets (the "fusion proofs & examples" fog, opened)

1. **Core surface:** `seqCompose` + `exec_seqCompose` +
   `ComputeRefine.FusionCorrect` + `fusionCorrect_iff`; manifest `fusion`
   kind in `check-artifact.sh`; `CorrectnessSurfaces.md` Quick-Pick row.
2. **SwiGLU running example:** port the pilot material per §5 and prove
   `swiglu_fusion_view`.
3. **FusedSiLU retrofit:** restate the existing pipeline proof as
   `fusedsilu_fusion_view` (N-ary instance).
