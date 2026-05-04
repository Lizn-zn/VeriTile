# DType Erasure and Compute Kernel Surface

VeriTile uses two kernel layers:

- `Kernel` / `Op` is the algorithm layer. It carries mathematical dtypes:
  `.real`, `.fp32`, `.fp16`, `.bf16`, `.int`, `.nat`, `.bool`, pointer, and
  block-pointer channels.
- `ComputeKernel` is the compute-facing DSL surface. The `triton { ... }`
  macro always produces a `ComputeKernel`.

The DSL has a single compute-facing surface. Pure algorithm kernels are emitted
as `ComputeKernel.fromAlg k`; kernels containing compute-facing expressions are
emitted as `ComputeKernel.mk inputs outputs body`, where `body` is a list of
`ComputeStmt`s.

## Algorithm Projection

`ComputeKernel.toAlgorithm?` is the public projection from the compute surface
to the algorithm layer:

```lean
ComputeKernel.toAlgorithm? : ComputeKernel -> Except EraseDTypeError AlgKernel
```

For the `.fromAlg` subset this projection succeeds immediately. For
`ComputeKernel.mk`, projection recursively lowers each `ComputeStmt` /
`ComputeExpr`; compute-only features that cannot be projected must fail this
projection instead of inventing fake algorithm meanings.

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

Successful projection has an explicit soundness bridge for the current
`.fromAlg` subset:

```lean
ComputeKernel.toAlgorithm?_sound :
  ck.toAlgorithm? = Except.ok ak ->
  ∀ s, ck.eval s = exec ak s
```

Future compute-only execution rules must extend this theorem; otherwise
`ComputeKernel.AlgorithmCorrect` would only be a statement about the projected
algorithm kernel, not about compute execution.

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

The current DSL accepts only the narrow algorithm-projectable slice:

- `tl.bitcast(<uint32 numeric literal>, tl.float32)` when the bits decode to a
  finite-normal binary32 value.
- The DSL stores this as `ComputeOp.bitcast .uint32 .fp32 ...`; it does not run
  a separate macro-level IEEE decoder.
- `ComputeKernel.toAlgorithm?` projects the containing `ComputeStmt` through the
  shared computable `Float32Bits.decodeRat` decoder and lowers to an algorithm
  `Op.const`.
- Zero/subnormal, NaN/Inf, non-`tl.float32` destinations, and runtime bitcast
  expressions are rejected at expansion time.

The representative theorem is:

```lean
ComputeOp.oneBitcast_toAlgorithm :
  ComputeOp.constOpToAlgorithm? ComputeOp.oneBitcast = Except.ok (Op.const 1)
```

Full runtime bitcast remains compute-only future work. When it is added,
successful projection must preserve compute semantics through computable decode
lemmas, and unsupported runtime cases must make `ComputeKernel.toAlgorithm?`
return `Except.error`.
