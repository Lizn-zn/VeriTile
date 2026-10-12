# Spec sheet — `bench/tritonbench_g/chunk_bwd_dqkg/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/chunk_bwd_dqkg/chunk_bwd_dqkg.py`

## Public theorem: `chunk_dot_left_equiv`

**Statement:**
```lean
specification chunk_dot_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original qk valueStorage hist dhist q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT ≡[R]
    dotLeft qk valueStorage hist dhist q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT
```

**Closed-form spec defs (transitive):** `original`, `dotLeft`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original qk valueStorage hist dhist q k v h g dout dh dq dk dg
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT
  domain := FPEquivProofs.requirementsFor
    (TypedSource.original qk valueStorage hist dhist q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT) qk BT BK
```
</details>

<details><summary><code>dotLeft</code></summary>

```lean
def dotLeft : Program :=
  { original qk valueStorage hist dhist q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT with
    kernel := TypedSource.dotLeft qk valueStorage hist dhist q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT }
```
</details>
