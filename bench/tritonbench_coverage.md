# TritonBench-G v1 coverage

The external anchor is [TritonBench-G v1][tb] — 184 GitHub-scraped real Triton
kernels (THUNLP / Tsinghua, ACL 2025 Findings; arXiv 2502.14752).

| | Count |
|---|---:|
| Anchor corpus | 184 |
| **Ported** (faithful `.py` + `.lean` pair, compiles, headline proven) | **152** |
| Not yet imported | 32 |
| — of those, expressible with today's DSL surface | 10 |
| — of those, blocked on a missing primitive or an ℝ-model limit | 22 |

## What "expressible" means here, and what it does not

The verdict below is produced by diffing, for each unimported kernel, the set of
`tl.*` forms it uses against the set the DSL's surface syntax accepts
(`VeriTile/Triton/DSL/**`, currently 95 forms). It is a statement about
**primitive coverage only**.

It is *not* a claim about proof feasibility. Several already-ported kernels
(FlashAttention-1 backward, chunked GLA, Mamba SSM) were within the primitive
contract long before their proofs existed. Read `Portable now` as "the surface
will elaborate", not "the port is easy".

### Two blind spots this method had, and a correction

The `tl.*`-name diff misses anything that is not spelled as a `tl.` call, and the
first pass of this table was wrong for six kernels because of it:

* **Python-level operators on tile values.** `b >> shifter`, `x & 0xF`,
  `-b_g`, builtin `min`/`max` carry no `tl.` prefix. Most are fine (the DSL has
  `Op.shiftRight`/`Op.bitAnd`/`Op.where`, and `-x` transcribes as `0.0 - x`), but
  they must be checked, because the DSL models bitwise ops on the **`.nat`
  channel only** — `DSL/Expansion/Compute.lean` refuses signed integer bitwise by
  design.
* **Tile *dtype* keywords.** `tl.zeros(..., dtype=tl.int32)` is a `tl.zeros` call
  either way, so the name diff sees nothing. But `Op.dot` is `.real`-only in the
  AST, so an integer-accumulator `tl.dot` is not expressible at all.
* **Whether the surface elaborates at all.** A kernel can use only supported
  forms and still fail to elaborate, because dtype *inference* is a separate
  matter from dtype *support*. This is what blocked the three `matmul_dequant*`
  kernels until the `PtrElems` threading fix, and no scan of any kind would have
  predicted it. The lesson is not a scan improvement: **elaborate the surface**
  before believing a portability verdict.

The counts and both tables below are the corrected measurement.

It is also not a claim about *semantic* adequacy for concurrency. Kernels whose
correctness depends on inter-program interleaving (spin locks, cooperative
reductions) elaborate fine — the host launch is the trusted boundary — but their
*specification* has to be chosen with that boundary in mind.

## Not yet imported: portable now (10)

Every form these use — including the non-`tl.` ones — is already in the DSL
surface. Each row is one port-sized unit of work: import the `.py` **together
with** its `.lean` port, because `bench/audit_tritonbench_g.sh` enforces
`py_count == lean_count` — the guard that stops a kernel being imported and then
forgotten.

| Kernel | `.py` lines | `@triton.jit` kernels |
|---|---:|---:|
| `bmm_optimized` | 233 | 1 |
| `chunk_gla_fwd` | 369 | 5 |
| `chunk_linear_attn` | 309 | 4 |
| `chunk_retention` | 452 | 4 |
| `chunk_retention_ops` | 364 | 4 |
| `matmul_dequant_int4` | 303 | 2 |
| `matmul_dequantize` | 358 | 3 |
| `matmul_dequantize_int4` | 269 | 1 |
| `parallel_attention` | 481 | 4 |
| `parallel_retention_attention` | 399 | 4 |

The three `matmul_dequant*` rows were briefly listed as blocked. They are not: the
4-bit unpack (`>>` / `& 0xF` on non-negative nibbles), the `Region .nat` packed
container, the `nat → ℝ` crossing at `* scales`, the rank-broadcast load mask,
in-loop pointer advance and the dead `c` binding are all within the surface, and
the one thing that did block them was a `PtrElems` inference bug, now fixed (see
`bench/tests/PtrElemDType.lean`). `matmul_dequantize_int4`'s full surface has been
elaborated end-to-end; the other two share its `matmul4_kernel`, so the same
blocker is gone, but their *additional* jit kernels have not been elaborated yet.

## Not yet imported: blocked on a missing primitive (22)

| Kernel | `.py` lines | missing `tl.*` |
|---|---:|---|
| `int_scaled_matmul` | 304 | `tl.broadcast_to` |
| `matmul_persistent_triton` | 154 | `tl.float8e4nv` |
| `triton_matmul` | 133 | `tl.float8e4nv` |
| `attention_llama` | 174 | `tl.float8e5` |
| `f8_conversion_utils` | 68 | `tl.float8e5` |
| `llama_ff_triton` | 152 | `tl.float8e5` |
| `rms_matmul_rbe` | 279 | `tl.float8e5` |
| `rms_rbe_matmul` | 190 | `tl.float8e5` |
| `fp4_to_bf16` | 214 | `tl.interleave` |
| `fp4_to_bf16_conversion` | 275 | `tl.interleave` |
| `uniform_sampling` | 226 | `tl.philox`, `tl.static_assert`, `tl.uint_to_uniform_float` |
| `layer_norm_fwd` | 218 | `tl.rand` |
| `multinomial_sampling` | 136 | `tl.rand` |
| `seeded_dropout` | 60 | `tl.rand` |
| `int8_matmul_kernel` | 271 | `tl.static_assert` |
| `isfinite_kernel` | 262 | `libdevice.isfinited` / `finitef` — **and** an ℝ-model limit (see below) |
| `int4_matmul` | 252 | signed integer arithmetic: after unpacking, `int_b - int_bzp` ranges over `[-15, 15]` **before** it crosses to `ℝ`, and the `.nat` channel truncates at `0`. Its three `matmul_dequant*` look-alikes multiply by `scales` first, which is why they are portable and this one is not. |
| `int8_dequant_matmul` | 212 | `tl.dot` into a `tl.zeros(..., dtype=tl.int32)` accumulator — `Op.dot` is `.real`-only in the AST |
| `int8_matmul_quantization` | 268 | same int32-accumulator `tl.dot`, plus `.to(tl.int8)` quantization |
| `layer_norm_triton` | 231 | `while tl.atomic_cas(Lock, 0, 1) == 1` **inside** `_layer_norm_bwd_dx_fused` — `Stmt` has `forLoop`/`forRange`/`forRangeDyn` but no `while` |
| `spinning_lock_reduction` | 99 | two `while`s, one a `tl.atomic_cas` spin whose exit depends on another program |
| `streamk_matmul` | 295 | two `while`s plus three atomics |

### Resolved: the `matmul_dequant*` blocker was a `PtrElems` inference bug

Kept as a record, because it cost a wrong published diagnosis before the right
one, and because the failure mode is invisible to both gates.

`DSL/Inference.collectPtrElems` is a whole-body pre-pass recording
`pointer name → element dtype` with prepend; `lookupPtrElem` takes the first
match. On its own that resolves a pointer name to whichever binding comes **last
in the body** — everywhere in the body, *including before that binding*. The
expansion driver passed the map through unchanged while threading `Env` properly,
so:

* `matmul4_kernel` binds `ptr` to `scales_ptrs` (`.real`) and then to
  `zeros_ptrs` (`.nat`), so **both** loads came back `.nat`, `scales` became
  `.nat`, and the following arithmetic failed to typecheck;
* worse, a load could be silently given the **wrong** element dtype rather than
  be rejected — a load before the second binding inherited the second binding's
  dtype and was accepted. Neither gate catches that: proofs are discharged against
  the *elaborated* `Stmt` list, and `audit_tritonbench_g.sh` compares surface text.

Fixed by threading the map per statement (`Inference.ptrElemsAfterStmt`) exactly
as `Env` already was. Checked: the corpus was **not** exposed — of 46 ports with a
non-real region, 39 rebind a pointer name that is then loaded, and all 39 rebind it
to the *same* root region, so the flat and threaded lookups agree everywhere and
`check_ports.sh` stays at 152 ok. `bench/tests/PtrElemDType.lean` pins both
directions and fails without the fix.

An earlier version of this section blamed "the inference environment fixes a name's
dtype at its first binding". That was false — the register environment prepends and
finds first, so value rebinding always worked — and it came from a malformed
minimal repro. The lesson recorded: a reduction has to be well-formed before it can
confirm anything, and a probe that *should* fail and doesn't is worth more than one
that fails.

### Unlock levers, ranked by kernel yield

| Lever | Unlocks | Kernels |
|---|---:|---|
| fp8 dtype channel | 7 | `attention_llama`, `f8_conversion_utils`, `llama_ff_triton`, `matmul_persistent_triton`, `rms_matmul_rbe`, `rms_rbe_matmul`, `triton_matmul` |
| RNG | 4 | `layer_norm_fwd`, `multinomial_sampling`, `seeded_dropout`, `uniform_sampling` |
| tl.interleave | 2 | `fp4_to_bf16`, `fp4_to_bf16_conversion` |
| tl.static_assert (macro no-op) | 2 | `int8_matmul_kernel`, `uniform_sampling` |
| tl.broadcast_to (alias of tl.broadcast) | 1 | `int_scaled_matmul` |
| IEEE special values (inf / NaN) + `libdevice.isfinited`/`finitef` | 1 | `isfinite_kernel` |
| `while` statement in `Stmt` (+ a termination story) | 3 | `layer_norm_triton`, `spinning_lock_reduction`, `streamk_matmul` |
| integer-channel `tl.dot` (int8×int8 → int32 accumulate) | 2 | `int8_dequant_matmul`, `int8_matmul_quantization` |
| signed fixed-width integer arithmetic | 1 | `int4_matmul` |

The column sums to 23, not 22: `uniform_sampling` needs both RNG and
`tl.static_assert`, so it appears under two levers. Every other kernel appears
once.

Two of these are cheap: `tl.static_assert` erases at the algorithm layer (it
constrains `constexpr`s at Triton compile time, so the lowered kernel is
unchanged), and `tl.broadcast_to` is the shape-explicit spelling of the
`tl.broadcast` the DSL already has. Neither should land before its first
consumer, though — an unconsumed surface form is dead code by this repo's own
standard.

The fp8 channel is the largest single lever (7 kernels). It is a real extension:
`FloatDType` would gain `e4m3`/`e5m2` grids, and every `RoundingModel` obligation
(`round_idem`, `round_real = id`) has to hold for them.

### The ℝ-model limit — `isfinite_kernel`

`isfinite_kernel` is the one entry blocked on *semantics* rather than syntax, and
it is worth stating separately because no primitive addition fixes it. VeriTile's
arithmetic is over `ℝ`, which has no infinities and no NaN. A faithful
`isfinite` would therefore be **vacuously true at every lane**, so the kernel's
specification would carry no content: the port would compile and the theorem
would be provable, and it would mean nothing. Making it meaningful requires an
IEEE special-value layer, not a `libdevice` binding.

## Method

Sources are fetched from the documented upstream path
`data/TritonBench_G_v1/<kernel>.py`; every one was checked to contain
`@triton.jit`. The supported-form set is extracted from the DSL's own syntax and
expansion modules rather than from a prose list — the module docstring in
`DSL/Expansion/Main.lean` is out of date and undercounts (it predates `tl.dot`,
`tl.where` and `tl.make_block_ptr`).

**How this scan can be fooled, and how that is checked.** A `tl.*` diff misses
primitives pulled in under an alias — `isfinite_kernel` does
`from triton.language.extra.cuda.libdevice import isfinited as _isfinited` and
then calls `_isfinited(x)`, which matches no `tl.` pattern. Every kernel in the
portable list was therefore also scanned for `from triton… import` lines. Two
have them: `isfinite_kernel` (real — reclassified as blocked) and
`int8_dequant_matmul`, whose `early_config_prune` / `estimate_matmul_time` are
`@triton.autotune` helpers outside the kernel body, and autotune is already the
trusted boundary. A verdict of `portable now` still means "the surface will
elaborate", so read the kernel before starting: this reclassification was found
by reading `isfinite_kernel`, not by the scan.

[tb]: https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1
