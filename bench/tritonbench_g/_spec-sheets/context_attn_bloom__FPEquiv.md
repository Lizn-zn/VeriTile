# Spec sheet — `bench/tritonbench_g/context_attn_bloom/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/context_attn_bloom/context_attn_bloom.py`

## Public theorem: `scaleFirst_equiv`

**Statement:**
```lean
specification scaleFirst_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N ≡[R]
    scaleFirst storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N
```

**Closed-form spec defs (transitive):** `original`, `scaleFirst`, `sites`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ContextAttnBloom.original storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N
  domain := sites (VeriTile.Bench.TritonBenchG.ContextAttnBloom.original storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N).surfaceBody
    BLOCK_M BLOCK_N sm_scale
```
</details>

<details><summary><code>scaleFirst</code></summary>

```lean
def scaleFirst : Program :=
  { original storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
    stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
    kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N with
    kernel := VeriTile.Bench.TritonBenchG.ContextAttnBloom.scaleFirst storage Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx b_prompt_cache_len
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N }
```
</details>

<details><summary><code>sites</code></summary>

```
/-- At each reached signed-loop iteration, the actual qk lanes and scale are
finite at the multiplication. Query/key dot, exp, normalization, precision
casts, gathers and stores remain common context. No exp identity is used. -/
```
```lean
def sites (code : List ComputeStmt) (BM BN : Nat) (scale : ℝ) : List Site :=
  match Source.intRange? code "start_n" with
  | none => []
  | some loop => [.intRange loop.before "start_n" loop.start loop.stop loop.stride loop.body
      (.mulScalarFP32 (Source.prefixBefore loop.body "qk" 2) BM [BN]
        (.ref .real [BM,BN] "qk") (.const scale))]

section
variable (storage : Storage) 
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
```
</details>
