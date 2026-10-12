# Spec sheet — `bench/tritonbench_g/vector_addition_custom/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py`

## Public theorem: `add_commuted_equiv`

<details><summary>docstring</summary>

```
/-- Admitted add_commute preserves the complete masked-store program. -/
```
</details>

**Statement:**
```lean
specification add_commuted_equiv (A B C : RegionName) (size BLOCK : Nat) (R : Rules) :
    original A B C size BLOCK ≡[R] commuted A B C size BLOCK
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `commuted`, `_add_kernel`, `additionSite`, `_add_commuted_kernel`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (A B C : RegionName) (size BLOCK : Nat) : Program where
  kernel := _add_kernel A B C size BLOCK
  domain := [additionSite A B C size BLOCK]
```
</details>

<details><summary><code>commuted</code></summary>

```lean
def commuted (A B C : RegionName) (size BLOCK : Nat) : Program :=
  { original A B C size BLOCK with kernel := _add_commuted_kernel A B C size BLOCK }
```
</details>

<details><summary><code>_add_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `vector_addition_custom.py`'s `_add_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK: tl.constexpr` → Lean `Nat` parameter. -/
```
```lean
def _add_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, a + b, mask=offs < $(size))
}
```
</details>

<details><summary><code>additionSite</code></summary>

```
/-- Both loaded operands must be finite before the first (and only) store,
including masked lanes whose arithmetic is evaluated. Locating the store by
kind keeps the site stable when unrelated register assignments are inserted. -/
```
```lean
def additionSite (A B C : RegionName) (size BLOCK : Nat) : Site :=
  .add (Source.prefixBeforeStore (_add_kernel A B C size BLOCK).surfaceBody) [BLOCK]
    (.ref .real [BLOCK] "a") (.ref .real [BLOCK] "b")
```
</details>

<details><summary><code>_add_commuted_kernel</code></summary>

```
/-- Commute only the value expression of the original masked store. -/
```
```lean
def _add_commuted_kernel
    (A B C : RegionName)
    (size BLOCK : Nat) :
    ComputeKernel := triton {
  prog_id = tl.program_id(0)
  offs = prog_id * $(BLOCK) + tl.arange(0, $(BLOCK))
  a = tl.load(A + offs, mask=offs < $(size))
  b = tl.load(B + offs, mask=offs < $(size))
  tl.store(C + offs, b + a, mask=offs < $(size))
}
```
</details>
