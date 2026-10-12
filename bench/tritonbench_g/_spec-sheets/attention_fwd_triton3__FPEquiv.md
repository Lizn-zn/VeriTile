# Spec sheet — `bench/tritonbench_g/attention_fwd_triton3/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/attention_fwd_triton3/attention_fwd_triton3.py`

## Public theorem: `attention_reciprocal_equiv`

**Statement:**
```lean
specification attention_reciprocal_equiv (R : Rules) :
    original Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
    END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size ≡[R]
    reciprocal Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
    END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `reciprocal`, `normalizationSites`, `isNormalization`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules
```
</details>

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
    END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size
  domain := normalizationSites (TypedSource.original Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
    END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size).surfaceBody
    BLOCK_M BLOCK_DMODEL
```
</details>

<details><summary><code>reciprocal</code></summary>

```lean
def reciprocal : Program :=
  { original Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
    Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
    END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size with
    kernel := TypedSource.reciprocal Q K V M Out L sm_scale
      stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N
      END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW sliding_window_offset sliding_window_size }
```
</details>

<details><summary><code>normalizationSites</code></summary>

```
/-- Checks each actual accumulator lane and denominator lane for finiteness,
and each denominator for nonzero, only when END selects normalization. -/
```
```lean
def normalizationSites (code : List ComputeStmt) (BM DM : Nat) : List Site :=
  match Source.conditional? code (fun yes => yes.any isNormalization) with
  | none => []
  | some branch => [.branch branch.before branch.condition Bool.true
      (.divBroadcast (Source.prefixBefore branch.yes "acc") [BM, DM] BM [1]
        (.ref .real [BM, DM] "acc")
        (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i")))]

section
variable (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int)
```
</details>

<details><summary><code>isNormalization</code></summary>

```
/-- Select the final normalization by its operation, not a statement offset. -/
```
```lean
def isNormalization : ComputeStmt → Bool
  | .assign _ _ "acc" (.compute (.alg .fp32 (.div .real _ _ _))) => Bool.true
  | _ => Bool.false
```
</details>
