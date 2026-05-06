# RP2: Address typing — ℝ-uniform vs Nat-bifurcated `Value`

**Status:** Resolved. Default for Phases A (post-polish)→D: **bifurcated**
(`Value` carries `scalarNat`/`tileNat` separately from `scalar`/`tile`).
**Date:** 2026-04-27.
**Owner:** Phase A polish (between Tier 1 close and Phase B start).
**Revisit conditions:** see §When to revisit.

---

## TL;DR

Originally, `Value` had a single scalar carrier `ℝ`:

```lean
inductive Value where
  | scalar : ℝ → Value
  | tile   : (n : Nat) → (Fin n → ℝ) → Value
```

Memory addresses (e.g. `pid * BLOCK + tl.arange(N)[i]`) were computed as
**`ℝ` arithmetic** at evaluation time, then floored back to `Nat` via
`realToNat : ℝ → Nat := ⌊·⌋₊` when the offset reached
`BlockState.mem`. Every kernel proof carried 8 lines of `hcast` boilerplate
re-establishing that this round-trip was identity for `(pid : ℝ) * (N : ℝ)
+ (k.val : ℝ)`.

We bifurcate `Value` to carry `Nat` and `ℝ` separately. The `realToNat` cast
is removed entirely; address arithmetic stays in `Nat` end-to-end.

---

## 1. Problem statement

Should `Value` (the runtime value type produced by `evalOp` and stored in
registers) use:

- **Uniform `ℝ`** — single scalar carrier; addresses (`pid`, `arange`,
  offsets) are encoded as `ℝ`-valued and floored to `Nat` at the
  `mem`-access boundary; *or*
- **Bifurcated `ℝ` / `Nat`** — separate constructors for `Nat`-valued
  scalars / tiles (used for addresses, sizes, indices) vs `ℝ`-valued
  scalars / tiles (used for data, accumulators, normalized outputs).

This decision propagates through `Op.programId` / `Op.arange` /
`Op.constNat`, `Value.bop`, `Value.uop`, `Value.reduceSum/Max`,
`evalOp.load/.store`, and the structure of every kernel proof.

## 2. Why uniform `ℝ` was the Phase-A default

Three reasons that made sense at bootstrap:

1. **Uniform `Value` → simpler `evalOp` dispatch.** A single scalar carrier
   means `Op.add` always evaluates as `(· + · : ℝ → ℝ → ℝ)`; no need to
   case-split on operand provenance.

2. **`Value.bop` stays at 4 cases.** Pointwise binary lift over scalar/tile
   is 4 combinations. Adding a `Nat` carrier doubles each axis, giving
   16 combinations (most of which are nonsensical mixed-type cases).

3. **Triton data values *are* floats.** `Op.exp`, `Op.div`, `Op.log`, and
   `Op.reduceSum/Max` all operate on `ℝ`. Encoding them inside an `ℝ`-only
   carrier was the obvious choice; addresses got swept along by accident,
   not by design.

The cost (only fully visible after Tier 1 closed) was:

- `realToNat` is a hack. Its only job is undoing the implicit `Nat → ℝ`
  cast that `Op.programId` / `Op.arange` introduced. The semantics file
  explicitly flagged it as `TODO(P1 polish)`.
- Each kernel-correctness proof carries an `hcast` lemma — 8 lines —
  re-proving that `realToNat (↑pid * ↑N + ↑k.val) = pid * N + k.val`.
  Five proofs share this verbatim, ≈ 40 lines of duplicated boilerplate.
- Every `simp`-reduced kernel goal contains `(↑s.pid : ℝ)` casts plus
  a leading `realToNat` wrapper, increasing visual noise.
- `evalOp .exp Op.programId` is **silently well-typed** under the uniform
  model — `exp(pid)` evaluates to a perfectly fine `ℝ`. The semantics is
  too permissive: programs that no Triton author would write are accepted.

## 3. Why bifurcation pays off

After Tier 1 closure, the cost-benefit shifts:

- **`hcast` disappears entirely.** Address arithmetic lives in `Nat`
  end-to-end. The kernel goal arrives at the scatter-readback shape
  directly, with offset `s.pid * N + k.val : Nat`, no cast.
- **Phase B/C/D scaling.** Every new kernel pattern adds new address
  arithmetic — `forLoop` introduces `idx` (a `Nat` counter); FA forward
  adds 2-D layout `i * stride_i + j * stride_j`; FA-2 has multi-axis
  `program_id`. Without bifurcation, each pattern needs its own `hcast`
  lemma — ≈ 10 such lemmas by the end of Phase C. With bifurcation: zero.
- **Type-level guard against nonsensical kernels.** After bifurcation,
  `Op.exp Op.programId` is an `evalOp` error (`none`): `Value.uop Real.exp`
  cannot accept `scalarNat`. The semantics layer rejects programs that
  apply float ops to address values.
- **`Value.bop` at 8 cases is acceptable.** Half the 16 hypothetical
  product cases are ℝ-only (4) or Nat-only (4); the rest are
  mixed-type errors and collapse to a single `_, _ => none` arm.

## 4. The bifurcation, concretely

```lean
inductive Value where
  | scalar    : ℝ   → Value             -- data scalar (for `e / s`, accumulators)
  | scalarNat : Nat → Value             -- address scalar (`pid`, `idx`, sizes)
  | tile      : (n : Nat) → (Fin n → ℝ)   → Value   -- data tile (`tl.exp`, …)
  | tileNat   : (n : Nat) → (Fin n → Nat) → Value   -- offset tile (`arange`, …)
```

`evalOp` mapping:

| Op | Old | New |
|---|---|---|
| `Op.const c`        | `scalar c`                     | `scalar c` (unchanged) |
| `Op.constNat n`     | (didn't exist)                 | `scalarNat n` (new constructor) |
| `Op.programId`      | `scalar (s.pid : ℝ)`           | `scalarNat s.pid` |
| `Op.arange n`       | `tile n (fun i => (i.val : ℝ))`| `tileNat n (fun i => i.val)` |
| `Op.add`/`mul`/`sub`/`div` (Nat × Nat) | unsupported  | `scalarNat`/`tileNat` arithmetic via `bop` |
| `Op.exp`/`log` (on Nat)               | silently casts | `none` (semantic error) |
| `Op.reduceSum/Max` (on Nat tile)      | silently sums  | `none` |
| `Op.load region offsetExpr`           | `realToNat (eval offsetExpr)` | direct `Nat` from `scalarNat`/`tileNat` |
| `Stmt.store region offsetExpr value`  | `realToNat ...` | direct `Nat` |

`realToNat` is **deleted**.

DSL convention update:

| DSL form | Old expansion | New expansion |
|---|---|---|
| `$(t : Nat)` antiquote | `Op.const ((t : Nat) : ℝ)` | `Op.constNat t` |
| Numeric literal `5` | `Op.const ((5 : Nat) : ℝ)` | `Op.const 5` (ℝ) |
| `tl.arange(N)` (literal/`$(N)`) | `Op.arange N` (Nat) | `Op.arange N` (unchanged; produces `tileNat`) |

Convention: **`$(...)` antiquote is the address/size channel (`Nat`);
bare numeric literals are the data channel (`ℝ`).**

## 5. Cost / benefit

| Dimension | Uniform `ℝ` | Bifurcated |
|---|---|---|
| `Value` constructors           | 2          | 4 |
| `Value.bop` cases              | 4          | 8 + 1 mixed-type catch-all |
| `realToNat` exists             | yes        | **no** |
| `hcast` per proof              | 8 lines    | 0 |
| Type-error caught: `exp(pid)`  | no         | **yes** (returns `none`) |
| Phase C 2-D-address proof tax  | × patterns | 0 |

Net: ~+60 lines of new semantics infrastructure; ~−40 lines (plus future
savings) of proof boilerplate; one entire class of weak typing eliminated.

## 6. Implementation log

- Phase-A scaffolding (≈ 2026-04-25) chose uniform `ℝ` deliberately;
  flagged in `Semantics.lean` as `TODO(P1 polish): bifurcate Value into
  .scalarReal / .scalarNat …`.
- 2026-04-27, post Tier 1 close: bifurcation executed in one batch.
  Tier 1 proofs shed their `hcast` lemmas (verbatim deletion in 5
  files); `realToNat` removed; `Value.bop` extended; `evalOp` dispatch
  added.

## 7. When to revisit

- **If we ever need first-class index arithmetic in `ℝ`** (e.g. continuous
  indexing for some non-Triton DSL), the bifurcation may need a third
  carrier. Out of scope for Triton.
- **If `Value.bop`'s case-split becomes a bottleneck** (e.g. Phase C
  introduces many op variants with subtle cross-type semantics), revisit
  whether to inline dispatch into `evalOp` instead.
- **If we ever decide to model IEEE-754 fidelity** (currently out of
  scope), the data carrier likely changes from `ℝ` to a richer type;
  bifurcation only sharpens that choice — addresses still belong in
  `Nat` regardless.

---

*See also `RP1` (`research_problem_pointer_vs_named_region.md`) for the
related decision on region naming. Both decisions push the framework
toward stronger typing of memory operations: regions are statically named
(not first-class pointers), and address arithmetic is statically `Nat`
(not floored `ℝ`).*
