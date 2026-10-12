# TritonBench-G work checkpoint

Proof development was stopped at the user's request before committing this
checkpoint. The accepted corpus remains **59 complete / 125 pending**.
See `bench/tritonbench_g/verification_round.md` for the accepted work, scopes,
independent replay records and remaining issues.

This directory preserves the unfinished fifth-order spherical harmonics and
FlashAttention source/proof work that previously lived under ignored `Logs/`.
Neither draft is integrated into the main corpus or included in Lake targets.
The original benchmark directories remain authoritative for corpus status.

`checkpoint.json` records each saved source hash and its actual local-check
status. A passed staging check is not repository acceptance. Missing or stale
receipts are marked unverified; interrupted work is not a pass. The copied
validation logs/receipts may contain original development paths. Historical
README status paragraphs describe earlier development stages; the checkpoint
manifest is authoritative for the saved source revision.

Fifth-order spherical harmonics has full typed sources, four real contracts,
two whole typed FP equivalences and finite real execution witnesses. The
all-storage typed execution checks, final integration, source/layout gates and
independent comparator/trust replay must be finished before acceptance.
Its backward decimal coefficients are not the exact derivatives of its forward
decimal table; the real witness checks the documented 1e-14 discrepancy.

FlashAttention has complete typed source/FP proofs, abstract typed executions,
independent real formulas and substantial real execution helpers. The general
numerical loop invariant, full real IO/finite execution, final integration and
independent replay remain incomplete. The latest entry/pointer helpers may
still require proof repair or fresh validation; consult the manifest.

`development-scripts/` preserves source generators and migration tooling with
their original local paths. Inspect and adapt them before use. They are not a
portable one-command build, and no script is run automatically by this archive.
The draft import roots are the respective `fifth-sph-work/` and
`flash-attn-work/` directories. After dependencies are built, prepend that root
to `LEAN_PATH` and compile modules in import order with Lean's `-R` set to it.
Do not promote a draft by copying source files alone.
