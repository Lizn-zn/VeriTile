# DType Erasure and Compute Kernel Surface

VeriTile uses two kernel layers:

- `Kernel` / `Op` is the algorithm layer. It carries mathematical dtypes:
  `.real`, `.fp32`, `.fp16`, `.bf16`, `.int`, `.nat`, `.bool`, pointer, and
  block-pointer channels.
- `ComputeKernel` is the compute-facing DSL surface. The `triton { ... }`
  macro always produces a `ComputeKernel`.

Today every `ComputeKernel` is `ComputeKernel.fromAlg k` for some algorithm
kernel `k`. This is the PR3a transition scaffold. It lets the DSL move to a
single compute-facing surface before compute-only operations such as
`tl.bitcast` are added.

## Algorithm Projection

`ComputeKernel.toAlgorithm?` is the public projection from the compute surface
to the algorithm layer:

```lean
ComputeKernel.toAlgorithm? : ComputeKernel -> Except EraseDTypeError AlgKernel
```

For the current `.fromAlg` subset this projection always succeeds. Future
compute-only features can fail this projection instead of inventing fake
algorithm meanings.

`ComputeKernel.AlgorithmCorrect` hides the `Except` plumbing:

```lean
ComputeKernel.AlgorithmCorrect ck post :=
  match ck.toAlgorithm? with
  | Except.ok ak => Kernel.Correct ak post
  | Except.error _ => False
```

The public proof surface should use `ComputeKernel.AlgorithmCorrect` for
compute-facing DSL kernels. Internal helper lemmas may still use
`Kernel.Correct` when they are specifically about an erased algorithm kernel.

The transition lemma is:

```lean
ComputeKernel.algorithmCorrect_fromAlg :
  Kernel.Correct k post ->
  ComputeKernel.AlgorithmCorrect (ComputeKernel.fromAlg k) post
```

This keeps existing algorithm proofs reusable while the DSL return type moves
to `ComputeKernel`.

## Naming

Two `eraseDType` families exist by design:

- algorithm-layer `Op.eraseDType` / `Kernel.eraseDType` collapses algorithm
  dtype tags such as `.fp32 -> .real`.
- compute-layer `ComputeKernel.toAlgorithm?` projects a compute-facing kernel
  into the algorithm layer and may fail once compute-only operations exist.

`FloatDType.eraseFloat` is intentionally still named `eraseFloat`; it is a
float-only witness operation, not the global dtype-erasure API.

## Integer Width Spellings

Algorithm correctness treats integer widths as spelling-only:

- `tl.int8`, `tl.int16`, `tl.int32`, `tl.int64` map to `.int`
- `tl.uint8`, `tl.uint16`, `tl.uint32`, `tl.uint64` map to `.nat`

No algorithm-layer wraparound, overflow, signed-width, or bit-vector payload
semantics are implied by those spellings.

## Bitcast Policy

`tl.bitcast` is compute-only. It must not be modeled as numeric `tl.cast`, and
it must not become an `AlgOp.bitcast`.

When compute bit payloads are introduced, supported constant bitcasts may
project to algorithm constants through computable decoders. Runtime bitcasts or
unsupported decode cases should return `Except.error` from
`ComputeKernel.toAlgorithm?`, making `ComputeKernel.AlgorithmCorrect` reduce to
`False`.

Future successful-erasure support for bitcast must include a soundness theorem
showing that successful projection preserves compute semantics modulo the
computable decode lemma.
