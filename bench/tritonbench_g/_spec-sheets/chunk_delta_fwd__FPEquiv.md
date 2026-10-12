# Spec sheet — `bench/tritonbench_g/chunk_delta_fwd/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/chunk_delta_fwd/chunk_delta_fwd.py`

## Public theorem: `chunk_sum_left_equiv`

**Statement:**
```lean
specification chunk_sum_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE ≡[R]
    sumLeft kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE
```

**Closed-form spec defs (transitive):** `original`, `sumLeft`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original kd value initial final k v d v_new h initial_state final_state
    s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
    H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE
  domain := FPEquivProofs.requirementsFor
    (TypedSource.original kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE) BK BV
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft : Program :=
  { original kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE with
    kernel := TypedSource.sumLeft kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE }
```
</details>
