# TritonBench-G v1 coverage

The external anchor is [TritonBench-G v1][tb] — 184 GitHub-scraped real Triton
kernels (THUNLP / Tsinghua, ACL 2025 Findings; arXiv 2502.14752).

| | Count |
|---|---:|
| Anchor corpus | 184 |
| **Ported** (faithful `.py` + `.lean` pair, compiles, headline proven) | **162** |
| Not yet imported | 22 |
| — of those, expressible with today's DSL surface | 0 |
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

## Not yet imported: portable now (0)

This bucket is now **empty**: `bmm_optimized` (its last member, 2026-08-13)
is ported. Its 10 `constexpr` branches specialize **whether each `tl.load`
carries a mask** (`mask_a = None` vs `mask_a = mask_m[:, None]`), and
`MaskOpt` is part of `Stmt`, not a value — so the port fixes the constexpr
assignment `DIVISIBLE_M = DIVISIBLE_N = DIVISIBLE_K = False` (the
fully-masked arm, total for arbitrary `M, N, K`; the other arms are
mask-elision optimizations on their divisible domains) rather than emitting
2³ × 2 = 16 surfaces, and transcribes **both** `GROUP_M` CTA-reorder arms
in one surface. Every remaining unimported kernel is blocked on a missing
primitive or an ℝ-model limit.

### Correction: this count was 9, and 7 of those were wrong

The 9 became 2 on 2026-08-10, after `matmul_dequantize_int4` was ported and the
remaining candidates were read kernel-by-kernel instead of scanned. Two distinct
mistakes, both instances of the same root cause — **the verdict had been formed from
one kernel per file, and from `tl.*` names rather than from the code**:

* **The two `matmul_dequant*` siblings are blocked, not portable.** Both contain a
  second jit kernel, `dequantize_kernel` (byte-identical between them), computing
  `(((int32_b >> b_shift) & 0xF) - ((zp_b >> bzp_shift) & 0xF)) * scale_b`. That is
  a subtraction of two nibbles **on the integer channel, before the crossing to
  `ℝ`**, ranging over `[-15, 15]`. `Op.bitAnd` / `Op.shiftRight` are `.nat`-only in
  the AST (`Core/Ast.lean`), the only cross-channel lift is `natToReal`, and `.nat`
  subtraction truncates at `0` — so there is no faithful transcription. This is
  *exactly* the `int4_matmul` blocker, which the previous version of this file cited
  while asserting the siblings were free of it. They are free of it in
  `matmul4_kernel`, which is why `matmul_dequantize_int4` (whose only jit kernel is
  `matmul4_kernel`) did port — and not in the file as a whole.
* **Five kernels contained a descending `for` loop inside a jit kernel**
  (all five now ported via the ascending respelling) —
  `range(NT - 1, -1, -1)`, or `range(hi, lo, -BTS)`. `stepForRangeAux` takes
  `step : Nat` and advances while `cur < stop` (`Semantics/Step.lean`), so a
  descending loop is not expressible at the semantics layer, never mind the surface.
  A `tl.*` name diff cannot see this: the call is `range`, not `tl.range`.

The two survivors were then checked properly: every `tl.*` form each uses is in the
accepted set; the only Python-level operators on tiles are `&` on bool masks
(`Op.boolAnd`), `//` and `%` on `nat` scalars; the only dtype keyword is
`tl.float32` on accumulators; every `for` is ascending; `chunk_gla_fwd`'s
`tl.sum(b, 1)` positional axis is accepted verbatim (`syntax num :
tritonReduceKwarg`).

## Not yet imported: blocked on a missing primitive (22)

The fp8 dtype channel itself **landed 2026-08-13** (`.f8e4`/`.f8e5` in
`TileDType`/`FloatDType`, spelled `tl.float8e4nv`/`tl.float8e5`), with
`f8_conversion_utils` as its first consumer (the 161st port). The six
remaining fp8 rows below lose their *named* blocker, but none has been
seven-point re-checked yet — each next port in the family must run the
full per-jit check first (this table's one-name-per-row format has
under-reported twice; see the correction after the lever table).

| Kernel | `.py` lines | missing `tl.*` |
|---|---:|---|
| `int_scaled_matmul` | 304 | `tl.broadcast_to` — **under-reported**: its second jit also needs an int32-accumulator `tl.dot` (see correction below) |
| `matmul_persistent_triton` | 154 | fp8 channel landed — but seven-point re-checked 2026-08-13: **also** signed integer registers (`ki = -1` sentinel; `tile_id = start_pid - NUM_SMS` goes negative before its first `+= NUM_SMS`) — the persistent-loop idiom puts it in the signed-integer family too |
| `attention_llama` | 174 | fp8 channel landed; port pending |
| `llama_ff_triton` | 152 | fp8 channel landed; port pending |
| `rms_matmul_rbe` | 279 | fp8 channel landed; port pending |
| `rms_rbe_matmul` | 190 | fp8 channel landed; port pending |
| `fp4_to_bf16` | 214 | `tl.interleave` |
| `fp4_to_bf16_conversion` | 275 | `tl.interleave` |
| `uniform_sampling` | 226 | `tl.philox`, `tl.static_assert`, `tl.uint_to_uniform_float` |
| `layer_norm_fwd` | 218 | `tl.rand` |
| `multinomial_sampling` | 136 | `tl.rand` |
| `seeded_dropout` | 60 | `tl.rand` |
| `int8_matmul_kernel` | 271 | `tl.static_assert` — **under-reported**: also an int32-accumulator `tl.dot`, `&`/`>>` on tiles, and a signed int8 subtraction (see correction below) |
| `isfinite_kernel` | 262 | `libdevice.isfinited` / `finitef` — **and** an ℝ-model limit (see below) |
| `int4_matmul` | 252 | signed integer arithmetic: after unpacking, `int_b - int_bzp` ranges over `[-15, 15]` **before** it crosses to `ℝ`, and the `.nat` channel truncates at `0`. |
| `matmul_dequant_int4` | 302 | same signed-nibble subtraction, in its `dequantize_kernel`: `((int32_b >> s) & 0xF) - ((zp_b >> t) & 0xF)`. Its `matmul4_kernel` is fine — that one is what `matmul_dequantize_int4` ports. |
| `matmul_dequantize` | 357 | same `dequantize_kernel` (byte-identical), plus a plain `matmul_kernel` that is fine |
| `int8_dequant_matmul` | 212 | `tl.dot` into a `tl.zeros(..., dtype=tl.int32)` accumulator — `Op.dot` is `.real`-only in the AST |
| `int8_matmul_quantization` | 268 | same int32-accumulator `tl.dot`, plus `.to(tl.int8)` quantization |
| `layer_norm_triton` | 231 | `while tl.atomic_cas(Lock, 0, 1) == 1` **inside** `_layer_norm_bwd_dx_fused` — `Stmt` has `forLoop`/`forRange`/`forRangeDyn` but no `while` |
| `spinning_lock_reduction` | 99 | two `while`s, one a `tl.atomic_cas` spin whose exit depends on another program |
| `streamk_matmul` | 295 | two `while`s plus three atomics |

### Resolved for `matmul4_kernel`: the `PtrElems` inference bug

Kept as a record, because it cost a wrong published diagnosis before the right
one, and because the failure mode is invisible to both gates.

`DSL/Inference.collectPtrElems` is a whole-body pre-pass recording
`pointer name → element dtype` with prepend; `lookupPtrElem` takes the first
match. On its own that resolves a pointer name to whichever binding comes **last
in the body** — everywhere in the body, *including before that binding*. The
expansion driver passed the map through unchanged while threading `Env` properly,
so:

This unblocked `matmul_dequantize_int4`, whose only jit kernel is `matmul4_kernel`.
It did **not** unblock the other two `matmul_dequant*` files; see the correction
above for what actually blocks them.

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
`check_ports.sh` stays at 154 ok. `bench/tests/PtrElemDType.lean` pins both
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
| fp8 dtype channel — **LANDED 2026-08-13** (`f8_conversion_utils` the 161st port; `triton_matmul` the 162nd, the first fp8 **matmul** face) | 4 remaining | `attention_llama`, `llama_ff_triton`, `rms_matmul_rbe`, `rms_rbe_matmul` — each still needs its own seven-point check + port; `matmul_persistent_triton` re-checked and moved (signed-int registers, see its row) |
| RNG | 4 | `layer_norm_fwd`, `multinomial_sampling`, `seeded_dropout`, `uniform_sampling` |
| integer channel (int32-accumulator `tl.dot` + signed fixed-width arithmetic + bitwise-on-int) | 7 | `int8_dequant_matmul`, `int8_matmul_quantization`, `int4_matmul`, `matmul_dequant_int4`, `matmul_dequantize`, `int8_matmul_kernel`, `int_scaled_matmul` — see the correction below for why the last two moved here |
| tl.interleave | 2 | `fp4_to_bf16`, `fp4_to_bf16_conversion` |
| tl.static_assert (macro no-op) | 2 | `int8_matmul_kernel`, `uniform_sampling` |
| tl.broadcast_to (alias of tl.broadcast) | 1 | `int_scaled_matmul` |
| IEEE special values (inf / NaN) + `libdevice.isfinited`/`finitef` | 1 | `isfinite_kernel` |
| `while` statement in `Stmt` (+ a termination story) | 3 | `layer_norm_triton`, `spinning_lock_reduction`, `streamk_matmul` |

Kernels can appear under several levers (`uniform_sampling` under RNG +
static_assert; `int8_matmul_kernel` under integer channel + static_assert;
`int_scaled_matmul` under integer channel + broadcast_to), so the column
does not sum to 23.

### Correction (2026-08-13): two "cheap-lever" rows were under-reported

The pre-fp8 version of this file listed `int8_matmul_kernel` under
`tl.static_assert` *only* and `int_scaled_matmul` under `tl.broadcast_to`
*only* — both wrong, and both instances of the already-recorded root cause
(the missing-`tl.*` column is a **name scan**, and Python-level operators,
dtype keywords, and loop step signs are invisible to it). The seven-point
per-jit check finds:

* **`int8_matmul_kernel`** also has `tl.dot(a, b - 1, out_dtype=tl.int32)`
  into a `tl.zeros(..., dtype=tl.int32)` accumulator (the integer-dot
  blocker), `b_uint8 & mask` / `>> (2*i)` bitwise on tiles, and
  `b - tl.full((1,), 1, dtype=tl.int8)` with `b ∈ [0,3]` — a **signed**
  int8 subtraction before the crossing to ℝ (the `int4_matmul` blocker).
* **`int_scaled_matmul`**'s second jit (`scaled_matmul_kernel_with_block_pointers`)
  also has an `ACC_TYPE = tl.int32` accumulator fed by `tl.dot` over int8
  tiles (integer dot), a `for k in range(K, 0, -BLOCK_K)` descending loop
  (respellable via the closed descending-range recipe), a constexpr
  parameter runtime-rebound (`GROUP_M = min(...)`, respellable), and
  `tl.max_contiguous`/`tl.multiple_of` hints (already in the DSL surface).

So neither is a cheap port: both sit in the **integer channel** family, and
`tl.static_assert`/`tl.broadcast_to` alone unlock nothing (a lever must
land with its first consumer, and these two have none until the integer
channel exists).

The **descending-range** lever is now **closed** — five landed consumers,
zero library change (`forRangeDyn_inv` drives the ascending counter
throughout). `chunk_linear_attn`'s, `chunk_retention`'s, and
`chunk_retention_ops`'s `bwd_kernel_dh` spell `for i_t in range(NT-1, -1,
-1)` as the ascending `for j in range(0, NT)` with `i_t = NT - 1 - j` as
the body's first statement; `parallel_attention`'s
`_parallel_rebased_bwd_dkv` (2026-08-12) landed the `-BTS` variant —
`for j in range(0, cdiv(cdiv(T,BTS)·BTS − (i_c+1)·BTL, BTS))` with
`i = cdiv(T,BTS)·BTS − BTS − j·BTS` (the `±BTS` in Python's `start − stop`
cancels exactly in ℤ, and ℕ-truncated subtraction reproduces Python's empty
range) — and `parallel_retention_attention`'s
`_parallel_retention_bwd_dkv` (2026-08-12) took the identical `-BTS`
respelling, additionally respelling its unary-minus decay index
`-o_k[:, None] + o_q[None, :]` as the subtraction
`o_q[None, :] - o_k[:, None]` (the two agree on every lane the causal
`tl.where` mask keeps). No counting-down `Stmt` constructor or mirror
invariant principle was needed after all.

`tl.static_assert` erases at the algorithm layer (it constrains
`constexpr`s at Triton compile time, so the lowered kernel is unchanged),
and `tl.broadcast_to` is the shape-explicit spelling of the `tl.broadcast`
the DSL already has — but per the correction above, neither is a lever that
unlocks anything *by itself* anymore, and neither should land before its
first consumer (an unconsumed surface form is dead code by this repo's own
standard).

The fp8 channel — formerly the largest single lever — **landed** exactly as
priced: `TileDType`/`FloatDType` gained `.f8e4`/`.f8e5` (Triton spellings
`tl.float8e4nv`/`tl.float8e5`), every match arm follows the `.fp16` pattern
(all float channels share the `WithBot ℝ` carrier), and no per-dtype
`RoundingModel` obligation arose (`round : FloatDType → ℝ → ℝ` is quantified
over the tag, so `triv` and every existing model extend for free). The
remaining big lever by yield is the **integer channel** (7).

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
