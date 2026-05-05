# DType Erasure and Compute Kernel Surface

VeriTile uses two kernel layers:

- `Kernel` / `Op` is the algorithm layer. It carries mathematical dtypes:
  `.real`, `.fp32`, `.fp16`, `.bf16`, `.int`, `.nat`, `.bool`, pointer, and
  block-pointer channels.
- `ComputeKernel` is the compute-facing DSL surface. The `triton { ... }`
  macro always produces a `ComputeKernel`.

The DSL has a single compute-facing surface. Every kernel is emitted as
`ComputeKernel.mk inputs outputs body`, where `body` is a list of
`ComputeStmt`s. Pure algorithm statements are represented as `ComputeStmt.alg`
entries; compute-facing syntax uses the other `ComputeStmt` constructors.

## Algorithm Projection

`ComputeKernel.toAlgorithm?` is the public projection from the compute surface
to the algorithm layer:

```lean
ComputeKernel.toAlgorithm? : ComputeKernel -> Except EraseDTypeError AlgKernel
```

Projection recursively lowers each `ComputeStmt` / `ComputeExpr`;
`ComputeStmt.alg` passes through unchanged. Compute-only features that cannot
be projected must fail this projection instead of inventing fake algorithm
meanings.

`ComputeKernel.ComputeCorrect` is the public proof surface and hides the
`Except` plumbing. By default it only requires the projected algorithm proof:

```lean
ComputeKernel.ComputeCorrect ck post (gap := .ignore) :=
  True ∧
    match ck.toAlgorithm? with
    | Except.ok ak => Kernel.Correct ak post
    | Except.error _ => False
```

When a theorem should record an externally validated compute-to-algorithm
gap, use a required gap contract:

```lean
ComputeKernel.ComputeCorrect ck post (gap := .require contract) :=
  ExternalChecked contract ∧
    match ck.toAlgorithm? with
    | Except.ok ak => Kernel.Correct ak post
    | Except.error _ => False
```

`ComputeKernel.ComputeRefine` is the corresponding two-kernel surface with the
same optional `GapPolicy`. Internal helper lemmas may still use
`Kernel.Correct` / `Kernel.Refine` when they are specifically about projected
algorithm kernels.

The transition lemmas are:

```lean
ComputeKernel.computeCorrect_of_toAlgorithm_eq :
  ck.toAlgorithm? = Except.ok ak ->
  Kernel.Correct ak post ->
  ComputeKernel.ComputeCorrect ck post

ComputeKernel.computeRefine_of_toAlgorithm_eq :
  lhs.toAlgorithm? = Except.ok lhsAlg ->
  rhs.toAlgorithm? = Except.ok rhsAlg ->
  Kernel.Refine lhsAlg rhsAlg rel ->
  ComputeKernel.ComputeRefine lhs rhs rel
```

This keeps projected algorithm proofs reusable while making projection success
explicit.

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
execution to algorithm execution.** `ComputeKernel.ComputeCorrect` is a
statement about the projected algorithm kernel plus an optional external gap
obligation. The formal compute-to-algorithm bridge is
`ComputeKernel.toAlgorithm?` / `eraseDType`: it maps compute-facing syntax to
the Real / Int / Nat algorithm layer where Lean proofs run. Numeric
compute-layer behavior — IEEE rounding, NaN propagation, denormals,
hardware-dot precision, fast-math, etc. — is validated empirically through the
external gap checker (see PLAN.md "Verification architecture" and #59), not
through a Lean theorem.

Users reading a `ComputeCorrect` Lean certificate should interpret it as: "the
projected algorithm structure (over Real / Int / Nat) is Lean-proved correct."
If the theorem uses `gap := .require contract`, it also records the trusted
external validation token for the named compute-to-algorithm gap contract.

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

The DSL accepts 32-bit payload reinterpretation among `tl.uint32`, `tl.int32`,
and `tl.float32`. Constant uint32 bit patterns are algorithm-projectable when
the destination can be represented in the mathematical algorithm layer:
`tl.uint32` projects to `.nat`, `tl.int32` projects to `.int`, and `tl.float32`
projects to `.real` only when the bits decode to a finite-normal binary32
value. Runtime bitcasts are represented in `ComputeOp.bitcast`, but
`ComputeKernel.toAlgorithm?` returns `Except.error` for them.

### Two views of the same bitcast

The macro emits two parallel terms for one accepted constant `tl.bitcast`:

- **Algorithm view** (`EOut.term`): `Op.const ((Float32Bits.decodeRat
  { bits := BitVec.ofNat 32 <bits> }).get (by decide) : ℝ)`. This is the
  projected algorithm-side term.
- **Compute view** (`EOut.computeTerm`): `ComputeExpr.compute (ComputeOp.bitcast
  ComputeDType.uint32 <dst> rfl (ComputeOp.const ⟨BitVec.ofNat 32 <bits>⟩))`.
  This is what the surrounding kernel uses when it routes to
  `ComputeKernel.mk inputs outputs body`.

Both views call `Float32Bits.decodeRat` as the single authoritative decoder.
There is no second IEEE decoder anywhere; in particular the macro does not
re-implement decoding in `MacroM`. The algorithm view's `Option.get (by decide)`
elaborates the decoder on the concrete `BitVec 32` literal at typechecking
time.

### Macro-time admissibility

`tl.bitcast` accepts only modeled 32-bit payload destinations at macro
expansion time:

- `dst ∉ {tl.uint32, tl.int32, tl.float32}` → `Macro.throwError`.
- numeric literal source must fit `uint32` (`bits < 2^32`).
- literal `dst = tl.float32` additionally rejects exponent field
  `(bits / 2^23) % 256 = 0` (zero/subnormal) or `255` (NaN/Inf), because those
  values are not yet algorithm-projectable.
- non-literal sources are accepted as runtime `ComputeOp.bitcast`; projection
  to the algorithm layer fails with `Except.error (.requiresComputeSemantics
  "runtime bitcast")`.

Any `tl.float32` literal that survives all of the above is a finite-normal
binary32 pattern. For such bits, `Float32Bits.decodeRat` is `some _` by
construction. The `Option.get (by decide)` in the emitted algorithm-side
`Op.const` is therefore a *defense-in-depth* assertion: if the macro-time
admissibility check ever regresses, elaboration fails loudly here rather than
silently substituting a placeholder constant. Runtime bitcasts have no
algorithm fallback constant.

### Representative theorem

```lean
ComputeOp.oneBitcast_toAlgorithm :
  ComputeOp.constOpToAlgorithm? ComputeOp.oneBitcast = Except.ok (Op.const 1)

ComputeOp.minusOneBitcast_toAlgorithm :
  ComputeOp.constOpToAlgorithm? ComputeOp.minusOneBitcast =
    Except.ok (Op.constInt (-1))
```

Algorithm proofs that do not inspect bitcast values continue to work unchanged;
proofs that inspect a specific decoded value reduce through
`Float32Bits.decodeRat` (e.g., `decide` / `native_decide`).

### Out of scope here

Runtime bitcast is now expressible, but remains compute-only:

- ComputeOp's representation must remain faithful enough that the testing
  pipeline can lift `ComputeKernel` back to Triton source (this is the bridge to
  compute-layer numeric behavior — there is no Lean-internal IEEE soundness
  theorem covering runtime compute behavior);
- unsupported runtime cases make `ComputeKernel.toAlgorithm?` return
  `Except.error`, so `AlgorithmCorrect` cannot be claimed for them in Lean;
- the `Except.error` branch must not be inlined into algorithm-side `Op` terms
  with a fallback constant.
