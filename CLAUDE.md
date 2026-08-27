# VeriTile

Lean 4 formal verification of Triton GPU kernels: **173 ported TritonBench-G
kernels** (`bench/tritonbench_g/<kernel>/`, each a faithful `.py` + `.lean`
pair) plus kernel showcases under `bench/examples/` — one self-contained
`.lean` per kernel correctness/equivalence story — and infra smoke tests
(regression gates over library surfaces, no kernel story) under
`bench/tests/` (bench files never import each other). `VeriTile/Examples/`
keeps only shared vocabulary (`Common.lean`), pure-math spec files, and the
multi-file FlashAttention1 / ApproxGeLU stacks.

## Build targets (`lakefile.toml`)

- `VeriTile` — lite target (Triton subset + worked examples); the default.
- `VeriTileMath` — the heavy `VeriTile.Math.*` analysis chain (GeLU error cert).
- `VeriTileFull` — lite + Math + ApproxGeLU; build before release tagging.

`import VeriTile.Triton` pulls the whole subset (Core, Semantics, Memory,
KernelLemmas, Correctness, Float, DSL, Math, Launch, Concurrency).

## Gates

- Library: `lake build` must exit 0. **Judge Lean by exit code, never by
  tail-ing output** (early parse errors scroll off).
- Bench ports: `bash bench/check_ports.sh` → `173 ok, 0 fail` (~23 min on a
  96-core box; bench is standalone, not in any lake target).
- Full bench gate (what CI runs): `bash bench/audit_tritonbench_g.sh` →
  `TritonBench-G audit gates passed`, 35 scans (~55 min). Wraps check_ports +
  the Python↔Lean faithfulness scans + the proof-gap manifest + both trust
  audits. Run it in the background: it exceeds the 10-min foreground cap.
- Artifact gate: `bash scripts/check-artifact.sh` exit 0 (no `sorry`, axiom
  whitelist, kernel-manifest schema, README link drift).
- Showcase: `lake env lean bench/examples/<F>.lean` exit 0, zero `sorry`.

## Conventions (read before non-trivial work)

- `bench/MAIN_THEOREM_CONVENTIONS.md` — main-theorem naming + structure.
- `documents/ProofConventions.md` — proof-style conventions.
- `documents/CodeOrganization.md` — module/directory map + dependency layers.
- `documents/CorrectnessSurfaces.md` — the `Realizes` (kernel-vs-spec) and
  `Refines` (kernel-vs-kernel) surfaces. The unqualified names are the
  rounding-model surfaces; the exact-ℝ idealizations are `*_without_Rounding`.
