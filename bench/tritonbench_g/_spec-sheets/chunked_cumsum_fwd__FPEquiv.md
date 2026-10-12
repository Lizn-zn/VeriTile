# Spec sheet — `bench/tritonbench_g/chunked_cumsum_fwd/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/chunked_cumsum_fwd/chunked_cumsum_fwd.py`

## Public theorem: `scaleFirst_equiv`

**Statement:**
```lean
specification scaleFirst_equiv (R : FP.ScalarArithmetic.Rules) :
    original biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK ≡[R]
    scaleFirst biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK
```

**Closed-form spec defs (transitive):** `original`, `scaleFirst`

<details><summary><code>original</code></summary>

```
/-- The whole prefix runs through both loads, bias/softplus, both clamps,
remasking and the first output store. The domain checks the actual reached
fp32 dt tile and A row for the only changed multiplication. Bias/A/input
storage stay independent, both outputs are fp32, and scan order is unchanged. -/
```
```lean
def original : Program where
  kernel := VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.original biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK
  domain := [.mulBroadcast
    (Source.prefixBefore (VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.original biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK).surfaceBody "dA")
    [BLOCK_SIZE_H,BLOCK_SIZE_CHUNK] [BLOCK_SIZE_H,1]
    (.ref .real [BLOCK_SIZE_H,BLOCK_SIZE_CHUNK] "dt")
    (.expandDim ⟨1,by simp⟩ (.ref .real [BLOCK_SIZE_H] "A"))]
```
</details>

<details><summary><code>scaleFirst</code></summary>

```lean
def scaleFirst : Program :=
  { original biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK with
    kernel := VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.scaleFirst biasStorage dtype out_dtype dt_ptr A_ptr dt_bias_ptr dt_out_ptr dA_cumsum_ptr batch seqlen nheads chunk_size dt_min dt_max stride_dt_batch stride_dt_seqlen stride_dt_head stride_A_head stride_dt_bias_head stride_dt_out_batch stride_dt_out_chunk stride_dt_out_head stride_dt_out_csize stride_dA_cs_batch stride_dA_cs_chunk stride_dA_cs_head stride_dA_cs_csize DT_SOFTPLUS HAS_DT_BIAS BLOCK_SIZE_H BLOCK_SIZE_CHUNK }
```
</details>
