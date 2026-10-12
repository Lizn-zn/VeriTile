# Spec sheet — `bench/tritonbench_g/chunk_gate_recurrence/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/chunk_gate_recurrence/chunk_gate_recurrence.py`

## Public theorem: `forward_equiv`

**Statement:**
```lean
specification forward_equiv (R : FP.ScalarArithmetic.Rules) :
    forward source gate seed S D O last_kv H N K V BK BV hasSeed ≡[R]
    forwardSumLeft source gate seed S D O last_kv H N K V BK BV hasSeed
```

**Closed-form spec defs (transitive):** `forward`, `forwardSumLeft`

<details><summary><code>forward</code></summary>

```lean
def forward : Program where
  kernel := TypedSource.forward source gate seed S D O last_kv H N K V BK BV hasSeed
  domain := FPEquivProofs.forwardRequirements
    (TypedSource.forward source gate seed S D O last_kv H N K V BK BV hasSeed) source gate BK BV
```
</details>

<details><summary><code>forwardSumLeft</code></summary>

```lean
def forwardSumLeft : Program :=
  { forward source gate seed S D O last_kv H N K V BK BV hasSeed with
    kernel := TypedSource.forwardSumLeft source gate seed S D O last_kv H N K V BK BV hasSeed }
```
</details>

## Public theorem: `backward_equiv`

**Statement:**
```lean
specification backward_equiv (R : FP.ScalarArithmetic.Rules) :
    backward source gate incoming S D DI DG DL DS H N K V BK BV ≡[R]
    backwardSumLeft source gate incoming S D DI DG DL DS H N K V BK BV
```

**Closed-form spec defs (transitive):** `backward`, `backwardSumLeft`

<details><summary><code>backward</code></summary>

```lean
def backward : Program where
  kernel := TypedSource.backward source gate incoming S D DI DG DL DS H N K V BK BV
  domain := FPEquivProofs.backwardRequirements
    (TypedSource.backward source gate incoming S D DI DG DL DS H N K V BK BV) gate incoming BK BV
```
</details>

<details><summary><code>backwardSumLeft</code></summary>

```lean
def backwardSumLeft : Program :=
  { backward source gate incoming S D DI DG DL DS H N K V BK BV with
    kernel := TypedSource.backwardSumLeft source gate incoming S D DI DG DL DS H N K V BK BV }
```
</details>
