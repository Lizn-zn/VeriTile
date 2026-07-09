# Trust Audit — how to use

**English** | [中文](TrustAudit_zh.md)

Machine-checkable gates that prove a theorem has no hidden `sorry`, no smuggled
axiom, and no self-referential spec. A theorem's soundness depends only on its
*statement* and its *axiom footprint* — never on the lemmas its proof uses — and
these commands check exactly that.

## Run the gates

```bash
# every proven LIBRARY theorem is axiom-clean (134 theorems)
lake build VeriTile.Meta.TrustReport

# every bench port + showcase is axiom-clean (159 files)
bash bench/audit_trust.sh                    # whole corpus
bash bench/audit_trust.sh swiglu_fwd         # just named kernels

# both, plus the port-completion checks, in one CI step
bash bench/audit_tritonbench_g.sh
```

Each exits `0` iff everything passes. A failure on a `proven` theorem is a
**real soundness finding** (a `sorry`/axiom leaked in) — fix the proof, never
weaken the gate.

## Audit one theorem yourself

`import VeriTile.Meta.StatementAudit`, then:

```lean
#axiomsClean my_theorem
-- ✓ my_theorem: axiom footprint ⊆ standard base
```

The four commands:

| Command | Checks |
|---|---|
| `#axiomsClean T` | footprint ⊆ `{propext, Classical.choice, Quot.sound}` — the main gate |
| `#stmtSurfaceSubset T ⊆ [a, b, …]` | `T`'s statement mentions no project constant outside the list |
| `#specNonCircular s avoiding [k, …]` | spec `s`'s definition never references a kernel `k` |
| `#auditStmt T` | inspection — lists the project constants in `T`'s statement |

## Add a self-audit to a file

Put the checks at the end of the file (see the SwiGLU pilot,
[`bench/examples/FusedSwiglu.lean`](../bench/examples/FusedSwiglu.lean),
for the full pattern). They run at compile time, so the file stops compiling if
any gate is violated:

```lean
#axiomsClean my_main_theorem
#stmtSurfaceSubset my_main_theorem ⊆ [my_kernel, InputLoadedAt, ComputeRefine.Refines]
#specNonCircular my_spec avoiding [my_kernel]
```

## Where things live

- Commands: [`VeriTile/Meta/StatementAudit.lean`](../VeriTile/Meta/StatementAudit.lean).
- Library driver (generated): [`VeriTile/Meta/TrustReport.lean`](../VeriTile/Meta/TrustReport.lean)
  — regenerate from the manifest with `python3 scripts/gen_trust_report.py`.
- Bench driver: [`bench/audit_trust.sh`](../bench/audit_trust.sh).

`TrustReport` lives in the `VeriTileFull` lakefile lib (it audits `ApproxGeLU`,
which pulls the heavy analysis chain), so a routine lite `lake build` stays fast.
