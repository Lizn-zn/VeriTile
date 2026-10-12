# Spec sheet — `bench/tritonbench_g/bmm_chunk_fwd/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/bmm_chunk_fwd/bmm_chunk_fwd.py`

## Public theorem: `bmm_dot_first_equiv`

**Statement:**
```lean
specification bmm_dot_first_equiv (R : FP.ScalarArithmetic.Rules) :
    original aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq ≡[R]
    dotFirst aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq
```

**Closed-form spec defs (transitive):** `original`, `dotFirst`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original aStorage bStorage out_dtype a b out seq
    seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq
  domain := FPEquivProofs.requirementsFor (TypedSource.original aStorage bStorage out_dtype a b out seq
    seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq)
    aStorage bStorage BM BN BK
```
</details>

<details><summary><code>dotFirst</code></summary>

```lean
def dotFirst : Program :=
  { original aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq with
    kernel := TypedSource.dotFirst aStorage bStorage out_dtype a b out seq
      seqlen chunk K groups sab sas sah sak sbb sbs sbh sbk sob soc soh som son ssb sss BM BN BK causal hasSeq }
```
</details>
