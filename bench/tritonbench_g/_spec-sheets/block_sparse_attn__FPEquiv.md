# Spec sheet — `bench/tritonbench_g/block_sparse_attn/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/block_sparse_attn/block_sparse_attn.py`

## Public theorem: `block_sparse_scaling_equiv`

**Statement:**
```lean
specification block_sparse_scaling_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN ≡[R]
    scalarLeft storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN
```

**Closed-form spec defs (transitive):** `original`, `scalarLeft`

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original storage out Q K V rows cols rs cs layouts scale
    sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN
  domain := FPEquivProofs.requirementsFor (TypedSource.original storage out Q K V rows cols rs cs layouts scale
    sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN) BM BN scale
```
</details>

<details><summary><code>scalarLeft</code></summary>

```lean
def scalarLeft : Program :=
  { original storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN with
    kernel := TypedSource.scalarLeft storage out Q K V rows cols rs cs layouts scale
      sqb sqh sqm skb skh skn svb svh svn sob soh som heads kvHeads N BM BN BD blocks evenM evenN }
```
</details>
