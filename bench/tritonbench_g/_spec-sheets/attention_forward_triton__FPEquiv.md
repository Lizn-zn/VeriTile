# Spec sheet — `bench/tritonbench_g/attention_forward_triton/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/attention_forward_triton/attention_forward_triton.py`

## Public theorem: `attention_reciprocal_equiv`

**Statement:**
```lean
specification attention_reciprocal_equiv (R : Rules) :
    original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE ≡[R]
    reciprocal q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `reciprocal`, `divisionSite`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules

section
variable (q k v qScale kScale out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE : Nat)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE
  domain := [divisionSite q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE]
```
</details>

<details><summary><code>reciprocal</code></summary>

```lean
def reciprocal : Program :=
  { original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE with
    kernel := TypedSource.reciprocal q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE }
```
</details>

<details><summary><code>divisionSite</code></summary>

```
/-- Check actual fp32 accumulators and row denominators after the full loop.
Every evaluated lane is finite and every row denominator is nonzero. This adds
no positivity, shape restriction, output equation or dot/reduction identity. -/
```
```lean
def divisionSite : Site :=
  .divBroadcast (Source.prefixBefore
    (TypedSource.original q k v qScale kScale out
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE STAGE).surfaceBody "acc" 1)
    [BLOCK_M, BLOCK_DMODEL] BLOCK_M [1]
    (.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
    (.expandDim ⟨1, by simp⟩ (.ref .real [BLOCK_M] "l_i"))
```
</details>
