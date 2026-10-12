# Spec sheet — `bench/tritonbench_g/chunk_gla_simple/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/chunk_gla_simple/chunk_gla_simple.py`

## Public theorem: `scale_equiv`

**Statement:**
```lean
specification scale_equiv (R : FP.ScalarArithmetic.Rules) :
    original dotStorage valueStorage gateStorage out q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV ≡[R]
    scaleFirst dotStorage valueStorage gateStorage out q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV
```

**Closed-form spec defs (transitive):** `original`, `scaleFirst`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := ChunkGlaSimple.original dotStorage valueStorage gateStorage out q k v h g o
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV
  domain := FPProof.requirements
    (ChunkGlaSimple.original dotStorage valueStorage gateStorage out q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV) valueStorage BT BV scale
```
</details>

<details><summary><code>scaleFirst</code></summary>

```lean
def scaleFirst : Program :=
  { original dotStorage valueStorage gateStorage out q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV with
    kernel := ChunkGlaSimple.scaleFirst dotStorage valueStorage gateStorage out q k v h g o
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV }
```
</details>
