# Spec sheet — `bench/tritonbench_g/context_attn_llama/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/context_attn_llama/context_attn_llama.py`

## Public theorem: `scaleFirst_equiv`

**Statement:**
```lean
specification scaleFirst_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage Q K V sm_scale Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N ≡[R]
    scaleFirst storage Q K V sm_scale Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N
```

**Closed-form spec defs (transitive):** `original`, `scaleFirst`, `sites`, `scoreOperand`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ContextAttnLlama.original storage Q K V sm_scale Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N
  domain := sites storage (VeriTile.Bench.TritonBenchG.ContextAttnLlama.original storage Q K V sm_scale Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N).surfaceBody
    BLOCK_M BLOCK_N sm_scale
```
</details>

<details><summary><code>scaleFirst</code></summary>

```lean
def scaleFirst : Program :=
  { original storage Q K V sm_scale Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N with
    kernel := VeriTile.Bench.TritonBenchG.ContextAttnLlama.scaleFirst storage Q K V sm_scale Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N }
```
</details>

<details><summary><code>sites</code></summary>

```
/-- Every reached iteration checks finite actual qk/scale operands in every
lane of the eager where arm. The page gather, dot, exp2, reductions, fused
accumulator-dot, casts, masks and stores stay in the common context. -/
```
```lean
def sites (storage : Storage) (code : List ComputeStmt) (BM BN : Nat) (scale : ℝ) : List Site :=
  match Source.intRange? code "start_n" with
  | none => []
  | some loop => [.intRange loop.before "start_n" loop.start loop.stop loop.stride loop.body
      (.mulScalar (Source.prefixBefore loop.body "qk" 1) BM [BN]
        (scoreOperand storage BM BN) (.const scale))]

section
variable (storage : Storage)
    (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
```
</details>

<details><summary><code>scoreOperand</code></summary>

```
/-- The first dot computes fp32 scores on every storage branch. -/
```
```lean
def scoreOperand (storage : Storage) (BM BN : Nat) : Op .real [BM,BN] :=
  match storage with
  | .fp16 | .bf16 => .castFloat .fp32 .real (.ref .fp32 [BM,BN] "qk")
  | .fp32 => .ref .real [BM,BN] "qk"
```
</details>
