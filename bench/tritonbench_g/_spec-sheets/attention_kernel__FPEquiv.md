# Spec sheet — `bench/tritonbench_g/attention_kernel/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/attention_kernel/attention_kernel.py`

## Public theorem: `attention_reciprocal_equiv`

**Statement:**
```lean
specification attention_reciprocal_equiv (R : Rules) :
    original storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N ≡[R]
    reciprocal storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `reciprocal`, `divisionSite`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules

section
variable (storage : TypedSource.Storage)
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N
  domain := [divisionSite storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N]
```
</details>

<details><summary><code>reciprocal</code></summary>

```lean
def reciprocal : Program :=
  { original storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N with
    kernel := TypedSource.reciprocal storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N }
```
</details>

<details><summary><code>divisionSite</code></summary>

```
/-- At the actual normalization after the complete loop, every accumulator lane
and row denominator is finite, and each row denominator is nonzero. This checks
intermediate operands, not an output equation or a restriction to an experiment shape. -/
```
```lean
def divisionSite : Site :=
  .divBroadcast (Source.prefixBefore
    (TypedSource.original storage q k v bias out sm_scale
      stride_qh stride_qm stride_qk stride_kh stride_kn stride_kk
      stride_vh stride_vk stride_vn stride_oh stride_om stride_on
      stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N).surfaceBody "acc" 1)
    [BLOCK_M, BLOCK_DMODEL] BLOCK_M [1]
    (.ref .real [BLOCK_M, BLOCK_DMODEL] "acc")
    (.expandDim ⟨1, by simp⟩ (.ref .real [BLOCK_M] "l_i"))
```
</details>
