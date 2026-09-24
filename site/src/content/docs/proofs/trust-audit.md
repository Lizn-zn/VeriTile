---
title: "Trust Audit — how to use"
---

Machine-checkable gates that prove a theorem has no hidden `sorry`, no smuggled
axiom, and no self-referential spec. A theorem's soundness depends only on its
*statement* and its *axiom footprint* — never on the lemmas its proof uses — and
these commands check exactly that.

## Run the gates

```bash
# install the pinned comparator/exporter and sandbox tools once
scripts/setup-comparator.sh /tmp/veritile-proof-tools
export PATH="/tmp/veritile-proof-tools/bin:$PATH"

# comparator replay of all proven LIBRARY manifest targets
python3 scripts/check_comparator.py --library

# audit bench ports, showcases, and infrastructure tests
bash bench/audit_trust.sh                    # whole corpus
bash bench/audit_trust.sh swiglu_fwd         # just named kernels

# both, plus the port-completion checks, in one CI step
bash bench/audit_tritonbench_g.sh
```

Each exits `0` only after every selected file reports a result and all checks
pass. Invalid concurrency, a launcher failure, and missing/duplicate results
are gate failures too. An `#axiomsClean` rejection on a `proven` theorem means
an unapproved axiom reached the proof; an infrastructure failure is reported
separately and does not establish a bad proof.

All three entrypoints require the official comparator. Library targets come
from the proven manifest rows; standalone files export all retained theorem
declarations, including private and macro-generated ones. Comparator checks
their axiom dependencies and replays the proofs in Lean's kernel. Routine
audits use a frozen current-source snapshot, not an old Git specification;
`scripts/prove.sh` additionally compares an agent's edits against the pre-agent
task. Definition-only fixtures explicitly report zero original theorem targets.
See [setup, scope, and recorded evidence](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/README.md#shared-comparator-gate).

The bench audit appends `#auditModuleAxioms` and `#auditModuleSpecs`.
`specification` registers its theorem in Lean's environment; the axiom gate
checks every registered headline and legacy `*_correct` / `*_output_summary`
theorem, including multiline, private, and Unicode declarations. Exact names
from the library manifest supplement this inventory. Each file reports its
actual headline count; zero means no headline axiom checks were performed.

Lean discovers compute kernels
from their elaborated result types (including multiline and parameterized
definitions), and checks every `*Spec` definition against them. Use
`@[kernel_spec]` to register independent mathematical specs with other names.
Use `@[kernel_denotation]` on execution denotations, including declarations
written with `denotation`; these are inventoried separately because they
intentionally depend on the kernel. Do not use that attribute to exempt a
mathematical correctness target. The per-file inventory reports kernel, spec,
and denotation counts; zero independent specs means no independence check was
performed for that file. Discovered specs without any kernel are an error.

## Audit one theorem yourself

`import VeriTile.Meta.StatementAudit`, then:

```lean
#axiomsClean my_theorem
-- ✓ my_theorem: axiom footprint ⊆ standard base
```

The commands:

| Command | Checks |
|---|---|
| `#axiomsClean T` | footprint ⊆ `{propext, Classical.choice, Quot.sound}` — the main gate |
| `#auditModuleAxioms` | discover registered headlines and legacy theorem suffixes in Lean; check their axiom footprints |
| `#stmtSurfaceSubset T ⊆ [a, b, …]` | `T`'s statement mentions no project constant outside the list |
| `#specNonCircular s avoiding [k, …]` | spec `s`'s definition never references a kernel `k` |
| `#auditStmt T` | inspection — lists the project constants in `T`'s statement |
| `#auditModuleSpecs` | discover current-module kernels/specs in Lean, check transitive independence, and report coverage |

## Add a self-audit to a file

Put the checks at the end of the file (see the SwiGLU pilot,
[`bench/examples/FusedSwigluEquiv.lean`](https://github.com/Lizn-zn/VeriTile/blob/main/bench/examples/FusedSwigluEquiv.lean),
for the full pattern). They run at compile time, so the file stops compiling if
any gate is violated:

```lean
#axiomsClean my_main_theorem
#stmtSurfaceSubset my_main_theorem ⊆ [my_kernel, InputLoadedAt, ComputeRefine.Refines]
#specNonCircular my_spec avoiding [my_kernel]
```

## Where things live

- Commands: [`VeriTile/Meta/StatementAudit.lean`](https://github.com/Lizn-zn/VeriTile/blob/main/VeriTile/Meta/StatementAudit.lean).
- Library driver (generated): [`VeriTile/Meta/TrustReport.lean`](https://github.com/Lizn-zn/VeriTile/blob/main/VeriTile/Meta/TrustReport.lean)
  — regenerate from the manifest with `python3 scripts/gen_trust_report.py`.
- Bench driver: [`bench/audit_trust.sh`](https://github.com/Lizn-zn/VeriTile/blob/main/bench/audit_trust.sh).
- Shared comparator driver: [`scripts/check_comparator.py`](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/check_comparator.py).

`TrustReport` lives in the `VeriTileFull` lakefile lib (it audits `ApproxGeLU`,
which pulls the heavy analysis chain), so a routine lite `lake build` stays fast.

## Audit boundary

The statement surface excludes declarations originating in the trusted `Init`,
`Std`, `Lean`, and `Mathlib` modules. Origin comes from Lean's environment,
not from a declaration's name: project definitions named `Nat.wrapper`,
`Real.wrapper`, or `instSomething` are still audited, including after import.
Spec traversal stops at those trusted dependencies; it follows project aliases.

`#axiomsClean` checks transitive axiom dependencies. The statement and spec
checks constrain referenced constants; they do not replace reviewing whether
a specification expresses the intended behavior under appropriate assumptions.
The project axiom whitelist used by `scripts/check-artifact.sh` is a separate
check: it builds every module under `VeriTile/` and enumerates actual axiom
declarations from Lean's environment, including private and macro-generated
declarations. Comments, strings, and source formatting do not affect discovery.
A whitelisted axiom is not thereby allowed in a theorem checked by `#axiomsClean`.
