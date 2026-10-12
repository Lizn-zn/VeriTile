# Spec sheet — `bench/tritonbench_g/attention_score/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/attention_score/attention_score.py`

## Public theorem: `score_scaling_equiv`

**Statement:**
```lean
specification score_scaling_equiv (R : Rules) :
    original Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock ≡[R]
    scalarLeft Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `scalarLeft`, `requirements`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules

section
variable (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN : Bool)
    (hBlock : BM = BN)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock
  domain := requirements Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock
```
</details>

<details><summary><code>scalarLeft</code></summary>

```lean
def scalarLeft : Program :=
  { original Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock with
    kernel := TypedSource.scalarLeft Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock }
```
</details>

<details><summary><code>requirements</code></summary>

```
/-- Locate the third qk assignment, after zero initialization and the score
dot. Each reached iteration checks the actual fp32 score tile and scale. -/
```
```lean
def requirements : List Site :=
  match Source.dynamicRange? (TypedSource.original Q K M Out 
      sqz sqh sqm sqk skz skh skn skk soz soh son
      Z H HK N RN NK BM BD BN offset size scale window complement evenM evenN hBlock).surfaceBody "start_m" with
  | none => []
  | some loop => [.range loop.before "start_m" loop.start loop.stop loop.stride loop.body
      (.mulScalarFP32 (Source.prefixBefore loop.body "qk" 2) BM [BN]
        (.ref .real [BM, BN] "qk") (.ref .real [] "qk_scale"))]
```
</details>
