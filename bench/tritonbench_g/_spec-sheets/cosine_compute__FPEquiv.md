# Spec sheet — `bench/tritonbench_g/cosine_compute/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/cosine_compute/cosine_compute.py`

## Public theorem: `index_commuted_equiv`

**Statement:**
```lean
specification index_commuted_equiv (storage : Storage) (a b : RegionName) (n B : Nat) :
    original storage a b n B ≡[assumptions] indexCommuted storage a b n B
```

**Closed-form spec defs (transitive):** `original`, `assumptions`, `indexCommuted`, `kernel`

<details><summary><code>original</code></summary>

```lean
def original (storage : Storage) (a b : RegionName) (n B : Nat) : Program where
  kernel := CosineCompute.original storage a b n B
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
def indexCommuted (storage : Storage) (a b : RegionName) (n B : Nat) : Program :=
  { original storage a b n B with kernel := CosineCompute.indexCommuted storage a b n B }
```
</details>

<details><summary><code>kernel</code></summary>

```lean
def kernel (storage : Storage) (swapped : Bool) (a b : RegionName) (n B : Nat) :=
  (if swapped then indexCommuted storage else original storage) a b n B
```
</details>
