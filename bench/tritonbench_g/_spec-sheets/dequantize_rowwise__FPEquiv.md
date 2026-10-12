# Spec sheet — `bench/tritonbench_g/dequantize_rowwise/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/dequantize_rowwise/dequantize_rowwise.py`

## Public theorem: `index_commuted_equiv`

**Statement:**
```lean
specification index_commuted_equiv (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) :
    original x scale out inv n block lanes ≡[assumptions] indexCommuted x scale out inv n block lanes
```

**Closed-form spec defs (transitive):** `original`, `assumptions`, `indexCommuted`

<details><summary><code>original</code></summary>

```lean
def original (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) : Program where
  kernel := TypedSource.original x scale out inv n block lanes
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>indexCommuted</code></summary>

```lean
def indexCommuted (x scale out : RegionName) (inv : ℝ) (n block lanes : Nat) : Program where
  kernel := TypedSource.indexCommuted x scale out inv n block lanes
  domain := []
```
</details>
