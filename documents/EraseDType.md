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

A definition-unfolding lemma is provided for the projection:

```lean
ComputeKernel.eval_eq_exec_of_toAlgorithm? :
  ck.toAlgorithm? = Except.ok ak ->
  ∀ s, ck.eval s = exec ak s
```

This is **not** a soundness theorem. `ComputeKernel.eval` is *defined* as
`match ck.toAlgorithm? with | Except.ok ak => exec ak s | Except.error _ => none`,
so the lemma above is the definition unfolded for the success case. It is a
convenience for `simp` chains, not a claim about compute-vs-algorithm
semantics.

**By design: there is no internal Lean theorem connecting bit-level compute
execution to algorithm execution.** `ComputeKernel.AlgorithmCorrect` is a
statement about the projected algorithm kernel. The formal compute-to-algorithm
bridge is `ComputeKernel.toAlgorithm?` / `eraseDType`: it maps compute-facing
syntax to the Real / Int / Nat algorithm layer where Lean proofs run. Numeric
compute-layer behavior — IEEE rounding, NaN propagation, denormals,
hardware-dot precision, fast-math, etc. — is validated empirically through the
differential testing pipeline (see PLAN.md "Verification architecture" and
#58), not through a Lean theorem.

Users reading an `AlgorithmCorrect` certificate should interpret it as: "the
projected algorithm structure (over Real / Int / Nat) is Lean-proved correct."
For end-to-end certification, the matching numeric `ComputeCorrect` test result
is also required.

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

The DSL accepts a single narrow algorithm-projectable slice:
`tl.bitcast(<uint32 numeric literal>, tl.float32)` when the bits decode to a
finite-normal binary32 value.

### Two views of the same bitcast

The macro emits two parallel terms for one accepted `tl.bitcast`:

- **Algorithm view** (`EOut.term`): `Op.const ((Float32Bits.decodeRat
  { bits := BitVec.ofNat 32 <bits> }).get (by decide) : ℝ)`. This is the term
  used when the surrounding kernel routes to the legacy `ComputeKernel.fromAlg
  (Kernel.mk ...)` path (no `ComputeStmt` siblings).
- **Compute view** (`EOut.computeTerm`): `ComputeExpr.compute (ComputeOp.bitcast
  ComputeDType.uint32 ComputeDType.fp32 rfl (ComputeOp.const ⟨BitVec.ofNat 32
  <bits>⟩))`. This is what the surrounding kernel uses when it routes to
  `ComputeKernel.mk inputs outputs body`.

Both views call `Float32Bits.decodeRat` as the single authoritative decoder.
There is no second IEEE decoder anywhere; in particular the macro does not
re-implement decoding in `MacroM`. The algorithm view's `Option.get (by decide)`
elaborates the decoder on the concrete `BitVec 32` literal at typechecking
time.

### Macro-time admissibility

`tl.bitcast` admissibility is decided at macro-expansion time, before any
algorithm-side `Op.const` is emitted:

- `dst != tl.float32` → `Macro.throwError`.
- non-numeric-literal source → `Macro.throwError` (runtime bitcast is
  compute-only).
- literal does not fit `uint32` (`bits >= 2^32`) → `Macro.throwError`.
- exponent field `(bits / 2^23) % 256 = 0` (zero/subnormal) → `Macro.throwError`.
- exponent field `(bits / 2^23) % 256 = 255` (NaN/Inf) → `Macro.throwError`.

Any literal that survives all of the above is a finite-normal binary32 pattern.
For such bits, `Float32Bits.decodeRat` is `some _` by construction. The
`Option.get (by decide)` in the emitted algorithm-side `Op.const` is therefore a
*defense-in-depth* assertion: if the macro-time admissibility check ever
regresses, elaboration fails loudly here rather than silently substituting a
placeholder constant. There is no runtime `match` with an `Except.error`
fallback in either view.

### Representative theorem

```lean
ComputeOp.oneBitcast_toAlgorithm :
  ComputeOp.constOpToAlgorithm? ComputeOp.oneBitcast = Except.ok (Op.const 1)
```

Algorithm proofs that do not inspect bitcast values continue to work unchanged;
proofs that inspect a specific decoded value reduce through
`Float32Bits.decodeRat` (e.g., `decide` / `native_decide`).

### Out of scope here

Full runtime bitcast remains compute-only future work. When it is added:

- ComputeOp's representation must remain faithful enough that the testing
  pipeline can lift `ComputeKernel` back to Triton source (this is the only
  bridge to compute-layer semantics — there is no Lean-internal soundness
  theorem covering runtime compute behavior);
- unsupported runtime cases must make `ComputeKernel.toAlgorithm?` return
  `Except.error`, so `AlgorithmCorrect` cannot be claimed for them in Lean;
- the `Except.error` branch must not be inlined into algorithm-side `Op` terms
  with a fallback constant — invariants of the current bitcast macro must be
  preserved end-to-end.
