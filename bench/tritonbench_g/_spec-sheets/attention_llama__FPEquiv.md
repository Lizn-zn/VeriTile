# Spec sheet — `bench/tritonbench_g/attention_llama/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/attention_llama/attention_llama.py`

## Public theorem: `attention_carry_first_equiv`

**Statement:**
```lean
specification attention_carry_first_equiv (R : Rules) :
    original Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal ≡[R]
    carryFirst Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `carryFirst`, `requirements`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules

section

variable (Q K V Out : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal : Bool)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
  domain := requirements Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
```
</details>

<details><summary><code>carryFirst</code></summary>

```lean
def carryFirst : Program :=
  { original Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal with
    kernel := TypedSource.carryFirst Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal }
```
</details>

<details><summary><code>requirements</code></summary>

```
/-- Check each row's computed sum(p) and updated carry, after the complete
score/exp prefix in every reachable iteration. No nonzero, positivity, output
formula or reduction identity is assumed by this commutation. -/
```
```lean
def requirements : List Site :=
  match Source.dynamicRange? (TypedSource.original Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal).surfaceBody "start_n" with
  | none => []
  | some loop => [.range loop.before "start_n" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "l_curr") [BM]
        (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "p"))
        (.ref .real [BM] "l_prev"))]
```
</details>

## Public theorem: `attention_fp8_carry_first_equiv`

**Statement:**
```lean
specification attention_fp8_carry_first_equiv (R : Rules) :
    originalFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal ≡[R]
    carryFirstFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
```

**Closed-form spec defs (transitive):** `Rules`, `originalFP8`, `carryFirstFP8`, `requirementsFP8`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules

section

variable (Q K V Out : RegionName) (sm_scale : ℝ)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD : Nat) (causal : Bool)
```
</details>

<details><summary><code>originalFP8</code></summary>

```lean
def originalFP8 : Program where
  kernel := TypedSource.originalFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
  domain := requirementsFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal
```
</details>

<details><summary><code>carryFirstFP8</code></summary>

```lean
def carryFirstFP8 : Program :=
  { originalFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal with
    kernel := TypedSource.carryFirstFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal }
```
</details>

<details><summary><code>requirementsFP8</code></summary>

```
/-- Check each row's computed sum(p) and updated carry, after the complete
score/exp prefix in every reachable iteration. No nonzero, positivity, output
formula or reduction identity is assumed by this commutation. -/
```
```lean
def requirementsFP8 : List Site :=
  match Source.dynamicRange? (TypedSource.originalFP8 Q K V Out sm_scale
      sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      NH H N SP BM BN BD causal).surfaceBody "start_n" with
  | none => []
  | some loop => [.range loop.before "start_n" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "l_curr") [BM]
        (.reduceSum ⟨1, by simp⟩ Bool.false (.ref .real [BM, BN] "p"))
        (.ref .real [BM] "l_prev"))]
```
</details>
