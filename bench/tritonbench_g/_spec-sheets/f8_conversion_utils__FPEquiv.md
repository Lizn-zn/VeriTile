# Spec sheet — `bench/tritonbench_g/f8_conversion_utils/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/f8_conversion_utils/f8_conversion_utils.py`

## Public theorem: `f8_to_f16_equiv`

**Statement:**
```lean
specification f8_to_f16_equiv (format : Format) (Y X : RegionName) (N B : Nat) :
    originalToHalf format Y X N B ≡[assumptions] indexCommutedToHalf format Y X N B
```

**Closed-form spec defs (transitive):** `originalToHalf`, `assumptions`, `indexCommutedToHalf`, `toHalf`

<details><summary><code>originalToHalf</code></summary>

```lean
def originalToHalf (format : Format) (Y X : RegionName) (N B : Nat) : Program where
  kernel := toHalf format Bool.false Y X N B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>indexCommutedToHalf</code></summary>

```lean
def indexCommutedToHalf (format : Format) (Y X : RegionName) (N B : Nat) : Program where
  kernel := toHalf format Bool.true Y X N B
  domain := []
```
</details>

<details><summary><code>toHalf</code></summary>

```lean
def toHalf (format : Format) (swapped : Bool) : sourceType :=
  match format,swapped with
  | .e4,Bool.false => f8Forward%[tl.float8e4nv,tl.float16,original]
  | .e4,Bool.true => f8Forward%[tl.float8e4nv,tl.float16,indexCommuted]
  | .e5,Bool.false => f8Forward%[tl.float8e5,tl.float16,original]
  | .e5,Bool.true => f8Forward%[tl.float8e5,tl.float16,indexCommuted]
```
</details>

## Public theorem: `to_fp8_equiv`

**Statement:**
```lean
specification to_fp8_equiv (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) :
    originalToFP8 input format Y X N B ≡[assumptions] indexCommutedToFP8 input format Y X N B
```

**Closed-form spec defs (transitive):** `originalToFP8`, `assumptions`, `indexCommutedToFP8`, `toFP8`

<details><summary><code>originalToFP8</code></summary>

```lean
def originalToFP8 (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) : Program where
  kernel := toFP8 input format Bool.false Y X N B
  domain := []
```
</details>

<details><summary><code>assumptions</code></summary>

```lean
def assumptions : Spec.Assumptions FP.GuardedFragment := []
```
</details>

<details><summary><code>indexCommutedToFP8</code></summary>

```lean
def indexCommutedToFP8 (input : Input) (format : Format) (Y X : RegionName) (N B : Nat) : Program where
  kernel := toFP8 input format Bool.true Y X N B
  domain := []
```
</details>

<details><summary><code>toFP8</code></summary>

```lean
def toFP8 (input : Input) (format : Format) (swapped : Bool) : sourceType :=
  match input,format,swapped with
  | .fp16,.e4,Bool.false => f8Reverse%[tl.float16,tl.float8e4nv,original]
  | .fp16,.e4,Bool.true => f8Reverse%[tl.float16,tl.float8e4nv,indexCommuted]
  | .fp16,.e5,Bool.false => f8Reverse%[tl.float16,tl.float8e5,original]
  | .fp16,.e5,Bool.true => f8Reverse%[tl.float16,tl.float8e5,indexCommuted]
  | .fp32,.e4,Bool.false => f8Reverse%[tl.float32,tl.float8e4nv,original]
  | .fp32,.e4,Bool.true => f8Reverse%[tl.float32,tl.float8e4nv,indexCommuted]
  | .fp32,.e5,Bool.false => f8Reverse%[tl.float32,tl.float8e5,original]
  | .fp32,.e5,Bool.true => f8Reverse%[tl.float32,tl.float8e5,indexCommuted]
```
</details>
