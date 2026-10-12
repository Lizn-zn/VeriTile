# Spec sheet — `bench/tritonbench_g/context_attn_mistral/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/context_attn_mistral/context_attn_mistral.py`

## Public theorem: `scaleFirst_equiv`

**Statement:**
```lean
specification scaleFirst_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage Q K V sm_scale B_Start_Loc B_Seqlen Out
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N ≡[R]
    scaleFirst storage Q K V sm_scale B_Start_Loc B_Seqlen Out
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N
```

**Closed-form spec defs (transitive):** `original`, `scaleFirst`, `sites`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ContextAttnMistral.original storage Q K V sm_scale B_Start_Loc B_Seqlen Out
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N
  domain := sites (VeriTile.Bench.TritonBenchG.ContextAttnMistral.original storage Q K V sm_scale B_Start_Loc B_Seqlen Out
    stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
    stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N).surfaceBody
    BLOCK_M BLOCK_N sm_scale
```
</details>

<details><summary><code>scaleFirst</code></summary>

```lean
def scaleFirst : Program :=
  { original storage Q K V sm_scale B_Start_Loc B_Seqlen Out
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N with
    kernel := VeriTile.Bench.TritonBenchG.ContextAttnMistral.scaleFirst storage Q K V sm_scale B_Start_Loc B_Seqlen Out
      stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N }
```
</details>

<details><summary><code>sites</code></summary>

```
/-- Every reached iteration checks finite actual fp32 qk and scale operands before
the causal/window masks, including all lanes. Dots, natural exp, reductions,
guards, casts and memory effects remain unchanged. -/
```
```lean
def sites (code : List ComputeStmt) (BM BN : Nat) (scale : ℝ) : List Site :=
  match Source.intRange? code "start_n" with
  | none => []
  | some loop => [.intRange loop.before "start_n" loop.start loop.stop loop.stride loop.body
      (.mulScalar (Source.prefixBefore loop.body "qk" 2) BM [BN]
        (.ref .real [BM,BN] "qk") (.const scale))]

section
variable (storage : Storage)
    (Q K V : RegionName) (sm_scale : ℝ) (B_Start_Loc B_Seqlen Out : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od kv_group_num : Nat) (sliding_window : Int)
    (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
```
</details>
