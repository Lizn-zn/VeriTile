# Trust Audit

**English** | [中文](TrustAudit_zh.md)

A theorem's soundness depends only on (a) the constants in its *statement* (its
type — the kernel checks the proof against it) and (b) its *axiom footprint*. It
does **not** depend on the defs/lemmas used only inside its proof. The trust
audit turns that observation into machine-checkable gates that run on every
elaboration and in CI, so a `sorry` or a smuggled axiom cannot hide in a proof
the corpus calls complete.

## The audit commands

Defined in [`VeriTile/Meta/StatementAudit.lean`](../VeriTile/Meta/StatementAudit.lean)
(module `VeriTile.Meta.StatementAudit`). Importing that module registers these
compile-time commands globally:

| Command | Gate |
|---|---|
| `#axiomsClean T` | fails unless `T`'s transitive axiom footprint ⊆ `{propext, Classical.choice, Quot.sound}`. Rejects `sorryAx` and any smuggled axiom — semantic, stronger than a textual `sorry` grep. |
| `#stmtSurfaceSubset T ⊆ [a, b, …]` | fails if `T`'s *statement* mentions a project constant outside the allowlist (e.g. a spec leaking into a spec-free headline). |
| `#specNonCircular s avoiding [k, …]` | fails if the *definition* of spec `s` transitively references any kernel `k` — a self-referential spec is a circular proof. |
| `#stmtConsts T` / `#auditStmt T` | inspection: enumerate a statement's constants / its non-core project surface. |

`#axiomsClean` is the primary soundness gate. The `SwiGLU` pilot
([`bench/examples/SwigluRoundingInvariance.lean`](../bench/examples/SwigluRoundingInvariance.lean))
embeds all three at its end as the reference usage pattern.

## Two populations, two mechanisms

The corpus splits by importability:

### Library theorems — `VeriTile/Meta/TrustReport.lean`

Every `proven` theorem whose manifest `file` is under `VeriTile/` is importable.
[`VeriTile/Meta/TrustReport.lean`](../VeriTile/Meta/TrustReport.lean) is a
**generated** driver that imports the needed modules and runs `#axiomsClean` on
each (134 theorems across 16 modules). Regenerate it from the manifest with:

```
python3 scripts/gen_trust_report.py
```

Run the gate:

```
lake build VeriTile.Meta.TrustReport      # builds deps + elaborates every #axiomsClean
# equivalently, once deps are built:
lake env lean VeriTile/Meta/TrustReport.lean
```

Exit 0 ⇔ every proven library theorem is axiom-clean. TrustReport lives in the
`VeriTileFull` lakefile lib (not the lite `VeriTile` lib) because it audits
`ApproxGeLU`, which pulls in the heavy `VeriTile.Math.*` analysis chain the lite
target deliberately skips — so a routine `lake build` stays lite.

### Standalone bench corpus — `bench/audit_trust.sh`

The 151 `bench/tritonbench_g/*/*.lean` ports and the `bench/examples/*.lean`
files are compiled solo and are **not** importable. They are audited
*externally*: [`bench/audit_trust.sh`](../bench/audit_trust.sh) emits a temp copy
of each file (via `bench/audit_trust_prep.py`) that adds
`import VeriTile.Meta.StatementAudit` and appends `#axiomsClean` on every
headline theorem (`*_correct`, `*_compute_correct`, `*_output_summary[_general]`,
plus any fully-qualified manifest name for the file) and `#specNonCircular` on
discoverable specs, then compiles the copy with `lake env lean`. The port files
themselves are never modified — the corpus stays clean.

```
bash bench/audit_trust.sh                 # whole bench corpus
bash bench/audit_trust.sh swiglu_fwd ...  # named tritonbench_g kernels
```

Prints `N ok / M fail`; exits nonzero on any failure.

## Running the gates

| Gate | Command |
|---|---|
| Library theorems axiom-clean | `lake build VeriTile.Meta.TrustReport` |
| Bench corpus axiom-clean | `bash bench/audit_trust.sh` |
| Both, inside the corpus audit | `bash bench/audit_tritonbench_g.sh` (extended to run both) |

## Policy

If `#axiomsClean` fails on a theorem the manifest calls `proven`, that is a
**real soundness finding** — a `sorry` or non-standard axiom leaked into a
supposedly-complete proof. Fix the proof; never weaken the gate.
