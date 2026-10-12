# Spec sheet — `bench/tritonbench_g/batched_vecmat_mult/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/batched_vecmat_mult/batched_vecmat_mult.py`

## Public theorem: `vecmat_block_first_equiv`

**Statement:**
```lean
specification vecmat_block_first_equiv (A B out : RegionName) (M N K BM BN BK : Nat) (R : Rules) :
    original A B out M N K BM BN BK ≡[R] blockFirst A B out M N K BM BN BK
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `blockFirst`, `requirements`, `blockSum`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (A B out : RegionName) (M N K BM BN BK : Nat) : Program where
  kernel := TypedSource.original A B out M N K BM BN BK
  domain := requirements A B out M N K BM BN BK
```
</details>

<details><summary><code>blockFirst</code></summary>

```lean
def blockFirst (A B out : RegionName) (M N K BM BN BK : Nat) : Program :=
  { original A B out M N K BM BN BK with kernel := TypedSource.blockFirst A B out M N K BM BN BK }
```
</details>

<details><summary><code>requirements</code></summary>

```
/-- Finiteness of the current carry and computed block sum at every reached
addition, after actual loads and broadcast. No reassociation is assumed. -/
```
```lean
def requirements (A B out : RegionName) (M N K BM BN BK : Nat) : List Site :=
  match Source.dynamicRange? (TypedSource.original A B out M N K BM BN BK).surfaceBody "k_index" with
  | none => []
  | some loop => [.range loop.before "k_index" loop.start loop.stop loop.stride loop.body
      (.add (Source.prefixBefore loop.body "vecmat") [BM,BN]
        (.ref .real [BM,BN] "vecmat") (blockSum BM BN BK))]
```
</details>

<details><summary><code>blockSum</code></summary>

```
/-- The actual RHS block product, reduction and transpose. The broadcasted A
register has leading dimension 1; multiplication expands it to [BN,BM,BK]. -/
```
```lean
def blockSum (BM BN BK : Nat) : Op .real [BM,BN] :=
  .transpose (batch := []) (.reduceSum ⟨2, by simp⟩ Bool.false
    (.mul .real (.consL (.consSame (.consSame .nil)))
      (.ref .real [1,BM,BK] "expanded_a") (.ref .real [BN,BM,BK] "b")))
```
</details>
