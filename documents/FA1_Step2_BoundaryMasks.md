# FA-1 Step 2 — Boundary Masks

Tracking note for issue #39 Step 2. Step 1 proved FA-1 over realistic 4D
strided memory, but still assumed full tiles:

```lean
hSk   : Bk * numKVBlocks = S_k
hQBnd : s.pids 0 * M + M <= S_q
```

Step 2 removes those sequence-axis divisibility assumptions by making Q-row
and KV-row boundary masks part of the kernel and proof surface.

## Scope

This step targets the sequence axes:

* Q/output tail: `offs_m < S_q`
* K/V tail: `offs_n < S_k`

The head dimension is still represented by the tile shape parameter `D`.
Supporting a separate `BLOCK_D` with `offs_d < D_head` is a later extension:
it changes the typed `tl.dot` shapes from `[M, D] @ [D, Bk]` to a padded
block-dimension model.

## Kernel Shape

The boundary kernels are:

```lean
fa1ForwardKernelStridedBoundary
fa1ForwardKernelStridedCausalBoundary
```

They take both tile sizes and logical sequence lengths:

```lean
(M D Bk numKVBlocks S_q S_k : Nat)
```

and are also exposed through the public wrappers:

```lean
FA1Layout4D.boundaryKernel
FA1Layout4D.causalBoundaryKernel
FA1Views4D.boundaryKernel
FA1Views4D.causalBoundaryKernel
```

The masks mirror real Triton practice:

* Q load: `tl.load(q_ptrs, mask = offs_m < S_q, other = 0)`
* K/V load: `tl.load(kv_ptrs, mask = offs_n < S_k, other = 0)`
* score mask: invalid KV lanes are rewritten to `-inf` before `tl.exp`
* output store: `tl.store(o_ptrs, out, mask = offs_m < S_q)`

Because the current DSL only coerces scalar masks automatically, the kernel
builds same-shape masks explicitly, for example:

```lean
(offs_m[:, None] + offs_d[None, :] * $(0)) < $(S_q)
```

for an `[M, D]` Q/load-store mask.

## Proof Plan

The existing full-tile proof uses a flat K/V domain
`TileIndex [Bk * numKVBlocks, D]`. Boundary masks require a logical K/V domain
`TileIndex [S_k, D]` plus a padded block view in the loop.

The next proof layer should introduce masked streaming recurrences parameterized
by `S_k`:

```lean
mPartialMasked / lPartialMasked / oPartialMasked
```

where each local `jLocal : Fin Bk` contributes only when
`k * Bk + jLocal.val < S_k`; otherwise the score is `-inf` and the
probability mass is zero. The final theorem should compare the ratio against
`attentionReal4D` / `attentionReal4DCausal` over the logical `[S_k]` domain,
not the padded `[Bk * numKVBlocks]` domain.

The output theorem will observe only in-bounds Q rows:

```lean
s.pids 0 * M + idx.1.val < S_q
```

Out-of-bounds rows are skipped by the store mask and should be stated as
preserving memory, or kept out of the public correctness theorem by requiring
an in-bounds lane hypothesis.

