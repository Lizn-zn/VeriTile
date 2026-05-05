# Kernel Manifest

`scripts/kernel-manifest.tsv` is the canonical per-kernel registry for the
artifact. It replaces the old flat theorem and examples lists.

Each non-comment row has this tab-separated schema:

```text
id	file	theorem	kind	status	source	source_ref	config	label	notes
```

## Fields

- `id`: stable machine-readable identifier. Must be unique.
- `file`: Lean source file containing the theorem.
- `theorem`: public theorem symbol guarded by artifact checks.
- `kind`: theorem category.
- `status`: verification status.
- `source`: kernel/source origin, such as `internal`, `tutorial`,
  `paper:<name>`, or `tritonbench:<path>`.
- `source_ref`: source commit, URL, paper anchor, or `-` when not applicable.
- `config`: important static configuration, such as `BLOCK_N=128` or
  `S=T=D=1,numIters=0`.
- `label`: short human-readable name.
- `notes`: simplifications, dropped features, or other caveats.

Allowed `kind` values:

```text
correct refine math launch trace safety frame
```

Allowed `status` values:

```text
proven projected test-gap blocked smoke
```

## Adding an Entry

When adding a new public example or benchmark theorem:

1. Add the Lean theorem using `ComputeKernel.ComputeCorrect` or
   `ComputeKernel.ComputeRefine` for ordinary example surfaces.
2. Add one row to `scripts/kernel-manifest.tsv`.
3. Record the source and static config precisely enough that the port is
   reproducible.
4. Put simplifications or proof-scope limitations in `notes`.
5. Run `scripts/check-artifact.sh`.

The artifact checker validates that every manifest file exists, every theorem
symbol exists, ids are unique, and `kind` / `status` use the allowed
vocabulary.

## Relationship to TritonBench-G

For future TritonBench-G ports, `source` should identify the benchmark entry
and `source_ref` should pin the upstream commit or URL. `config` records the
chosen `BLOCK_*` / dtype / static meta-parameters. `notes` records dropped or
simplified features, such as dropout, unsupported async paths, or deferred IEEE
compute semantics.
