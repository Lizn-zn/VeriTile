# Spec sheet — `bench/tritonbench_g/bmm_chunk_bwd/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/bmm_chunk_bwd/bmm_chunk_bwd.py`

## Public theorem: `bmm_residual_equiv`

**Statement:**
```lean
specification bmm_residual_equiv (R : FP.ScalarArithmetic.Rules) :
    original aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual ≡[R]
    residualLeft aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual
```

**Closed-form spec defs (transitive):** `original`, `residualLeft`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original aStorage doutStorage dtype out_dtype a dout db res
    seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual
  domain := FPEquivProofs.requirementsFor (TypedSource.original aStorage doutStorage dtype out_dtype a dout db res
    seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual) BM BN
```
</details>

<details><summary><code>residualLeft</code></summary>

```lean
def residualLeft : Program :=
  { original aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual with
    kernel := TypedSource.residualLeft aStorage doutStorage dtype out_dtype a dout db res
      seqlen chunk K groups sab sas sah sak sdb sdc sdh sdm sdn sob sos soh sok srb srs srh srk BM BN BCS hasResidual }
```
</details>
