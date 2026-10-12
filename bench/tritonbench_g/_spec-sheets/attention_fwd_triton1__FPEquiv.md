# Spec sheet — `bench/tritonbench_g/attention_fwd_triton1/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/attention_fwd_triton1/attention_fwd_triton1.py`

## Public theorem: `linear_attention_equiv`

<details><summary>docstring</summary>

```
/-- Whole-state equivalence includes both stores, all branches, explicit
fp32 loads/casts, allow_tf32=False dots, and execution failures. -/
```
</details>

**Statement:**
```lean
specification linear_attention_equiv (R : Rules) :
    original q k v h o s_qh s_qt s_qd s_hh s_ht T scale BT BD NT STORE IFCOND ≡[R]
    scalarLeft q k v h o s_qh s_qt s_qd s_hh s_ht T scale BT BD NT STORE IFCOND
```

**Closed-form spec defs (transitive):** `Rules`, `original`, `scalarLeft`, `requirements`

<details><summary><code>Rules</code></summary>

```lean
abbrev Rules := FP.ScalarArithmetic.Rules

section
variable (q k v h o : RegionName)
    (s_qh s_qt s_qd s_hh s_ht T : Nat) (scale : ℝ)
    (BT BD NT : Nat) (STORE IFCOND : Bool)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original : Program where
  kernel := TypedSource.original q k v h o s_qh s_qt s_qd s_hh s_ht T scale
    BT BD NT STORE IFCOND
  domain := requirements q k v h o s_qh s_qt s_qd s_hh s_ht T scale BT BD NT STORE IFCOND
```
</details>

<details><summary><code>scalarLeft</code></summary>

```lean
def scalarLeft : Program :=
  { original q k v h o s_qh s_qt s_qd s_hh s_ht T scale BT BD NT STORE IFCOND with
    kernel := TypedSource.scalarLeft q k v h o s_qh s_qt s_qd s_hh s_ht T scale
      BT BD NT STORE IFCOND }
```
</details>

<details><summary><code>requirements</code></summary>

```
/-- Locate the original loop by its index, then the second write to b_q.
The finite checks apply to the loaded query tile and interpreted fp32 scale
at every reachable iteration. The source has exactly one such range; the
equivalence proof checks its complete body and execution context. -/
```
```lean
def requirements : List Site :=
  match Source.dynamicRange?
      (TypedSource.original q k v h o s_qh s_qt s_qd s_hh s_ht T scale
        BT BD NT STORE IFCOND).surfaceBody "i" with
  | none => []
  | some loop => [.range loop.before "i" loop.start loop.stop loop.stride loop.body
      (.mulScalar (Source.prefixBefore loop.body "b_q" 1) BT [BD]
        (.ref .real [BT, BD] "b_q") (.const scale))]
```
</details>
