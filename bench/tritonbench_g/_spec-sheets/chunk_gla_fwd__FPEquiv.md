# Spec sheet — `bench/tritonbench_g/chunk_gla_fwd/FPEquiv.lean`

**Python source:** `bench/tritonbench_g/chunk_gla_fwd/chunk_gla_fwd.py`

## Public theorem: `output_equiv`

**Statement:**
```lean
specification output_equiv (R : FP.ScalarArithmetic.Rules) :
    outputOriginal queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV ≡[R]
    outputSumLeft queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV
```

**Closed-form spec defs (transitive):** `outputOriginal`, `outputSumLeft`, `original`, `outputRequirements`, `sumLeft`, `source`, `intraDot`, `Storage`, `scaleFirst`, `loadFirst`

<details><summary><code>outputOriginal</code></summary>

```lean
def outputOriginal : Program where
  kernel := TypedOutput.original queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV
  domain := outputRequirements (TypedOutput.original queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
    s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV) valueStorage BT BV
```
</details>

<details><summary><code>outputSumLeft</code></summary>

```lean
def outputSumLeft : Program :=
  { outputOriginal queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV with
    kernel := TypedOutput.sumLeft queryStorage valueStorage gateStorage dtype out_dtype q v g h o A
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV }
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
```
</details>

<details><summary><code>outputRequirements</code></summary>

```lean
def outputRequirements (kernel : ComputeKernel) (storage : TypedOutput.Storage) (BT BV : Nat) : List Site :=
  [.addFP32 (Source.prefixBefore kernel.surfaceBody "b_o" 1) [BT,BV]
    (.ref .real [BT,BV] "b_o") (intraDot storage BT BV)]

section
variable (queryStorage valueStorage gateStorage : TypedOutput.Storage)
    (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ) (T K V BT BK BV : Nat)
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

<details><summary><code>source</code></summary>

```lean
def source (queryStorage valueStorage gateStorage : Storage) (swapped : Bool) :=
  match queryStorage,valueStorage,gateStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
```
</details>

<details><summary><code>intraDot</code></summary>

```
/-- Finite checks cover the final carry and A·V dot after the complete K loop. -/
```
```lean
def intraDot (storage : TypedOutput.Storage) (BT BV : Nat) : Op .real [BT,BV] :=
  match storage with
  | .fp32 => .dot (batch := []) (.ref .real [BT,BT] "b_A") (.ref .real [BT,BV] "b_v")
      {allowTF32 := some Bool.false}
  | .fp16 => .castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
      (.ref .fp16 [BT,BT] "b_A") (.ref .fp16 [BT,BV] "b_v") {allowTF32 := some Bool.false})
  | .bf16 => .castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
      (.ref .bf16 [BT,BT] "b_A") (.ref .bf16 [BT,BV] "b_v") {allowTF32 := some Bool.false})
```
</details>

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := TypedOutput.Storage
syntax "glaSplitSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaSplitSource%[$qTy:tritonDType, $kTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let scaled : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then `(tritonStmt| b_A = tl.where(o_i >= j, $((scale : ℝ)) * b_A, 0.0))
      else `(tritonStmt| b_A = tl.where(o_i >= j, b_A * $((scale : ℝ)), 0.0))
    `(fun (q k g A : RegionName)
        (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) =>
      ComputeKernel.mk [q,k,g] [A] (triton {
        tl.region q = $qTy, k = $kTy, g = $gTy, A = tl.float32
        i_k = tl.program_id(0)
        i_tc = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_t = i_tc // $(NC)
        i_i = i_tc % $(NC)
        i_j = i_i
        n_bh = tl.num_programs(2)
        if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
        o_i = tl.arange(0, $(BC))
        o_k = i_k * $(BK) + tl.arange(0, $(BK))
        o_A = (i_bh + i_k * n_bh) * $(T) * $(BC) + (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) * $(BC)
        m_k = o_k < $(K)
        m_A = (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) < $(T)
        p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
        b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
        for j in range($(0), min($(BC), $(T) - i_t * $(BT) - i_i * $(BC))) {
          b_A = tl.zeros([$(BC)], dtype=tl.float32)
          b_k = (tl.load(p_k, mask=m_k, other=0)).to(tl.float32)
          b_gk = (tl.load(p_gk, mask=m_k, other=0)).to(tl.float32)
          b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
          $scaled
          tl.store(A + o_A + j, b_A, mask=m_A)
          p_k += $(K)
          p_gk += $(K)
        }
      }).surfaceBody)
```
</details>

<details><summary><code>scaleFirst</code></summary>

```lean
def scaleFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedSplit

/-! Complete source of chunk_gla_fwd_A_kernel_intra_sub_intra_merge. The host
allocates its temporary input in fp32; output storage stays explicit. The source
early return is transcribed directly. Runtime head count,
all NK contributions, omitted padding and output conversion are retained. The
candidate commutes only the fp32 carry/load addition inside the loop. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge
open Lean VeriTile.Triton
syntax "glaMerge%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaMerge%[$version:ident]) => do
    unless version.getId == `original || version.getId == `loadFirst do
      Macro.throwError "expected original or loadFirst"
    let add : TSyntax `tritonStmt ←
      if version.getId == `loadFirst then
        `(tritonStmt| b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat)) + b_A)
      else `(tritonStmt| b_A += tl.load(p_A, boundary_check=([0, 1] : List Nat)))
    `(fun (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) =>
      ComputeKernel.mk [A] [A2] (triton {
        tl.region A = tl.float32, A2 = OUT_DTYPE
        i_t = tl.program_id(0)
        i_c = tl.program_id(1)
        i_bh = tl.program_id(2)
        if i_t * $(BT) + i_c * $(BC) >= $(T) { return }
        n_bh = tl.num_programs(2)
        b_A = tl.zeros([$(BC), $(BC)], dtype=tl.float32)
        for i_k in range($(0), $(NK)) {
          p_A = tl.make_block_ptr(base=A + (i_bh + i_k * n_bh) * $(T) * $(BC),
            shape=($(T), $(BC)), strides=($(BC), $(1)),
            offsets=(i_t * $(BT) + i_c * $(BC), $(0)),
            block_shape=($(BC), $(BC)), order=(1, 0))
          $add
        }
        p_A2 = tl.make_block_ptr(base=A2 + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT) + i_c * $(BC), i_c * $(BC)),
          block_shape=($(BC), $(BC)), order=(1, 0))
        tl.store(p_A2, (b_A).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>loadFirst</code></summary>

```lean
def loadFirst := glaMerge%[loadFirst]
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge

/-! Source: chunk_gla_fwd.py, chunk_gla_fwd_kernel_o.
Complete typed source of the output consumer.
Q, V and gate storage independently select fp32/fp16/bf16. History and output
storage remain explicit dtype parameters; A is fp32, as allocated by the host.
Query scale and exp-gate results are cast back to query storage. H is cast to
that same storage before the default-option inter-chunk dot; A is cast to V
storage before the TF32-disabled final dot. Omitted padding remains unspecified.
The candidate commutes only the final fp32 carry/intra-chunk dot addition.
All dimensions, strides and program IDs are symbolic. This source covers the
output consumer; the four A producers have separate proof obligations. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput
open Lean VeriTile.Triton

syntax "glaOutputSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaOutputSource%[$qTy:tritonDType, $vTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let finalAdd : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| b_o = tl.dot(b_A, b_v, allow_tf32=false) + b_o)
      else `(tritonStmt| b_o += tl.dot(b_A, b_v, allow_tf32=false))
    `(fun (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,v,g,h,A] [o] (triton {
        tl.region q = $qTy, v = $vTy, g = $gTy, h = DTYPE,
          o = OUT_DTYPE, A = tl.float32
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        m_s = tl.arange(0, $(BT))[:, None] >= tl.arange(0, $(BT))[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
            shape=($(K), $(V)), strides=($(s_h_t), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_q = (b_q * $((scale : ℝ))).to($qTy:tritonDType)
          b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
          b_qg = (b_q * tl.exp(b_g)).to($qTy:tritonDType)
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          if i_k >= $(0) {
            b_o += tl.dot(b_qg, (b_h).to($qTy:tritonDType))
          }
        }
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT), $(0)), block_shape=($(BT), $(BT)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
        b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat))
        b_A = (tl.where(m_s, b_A, 0.0)).to($vTy:tritonDType)
        $finalAdd
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

## Public theorem: `inter_equiv`

**Statement:**
```lean
specification inter_equiv (R : FP.ScalarArithmetic.Rules) :
    interOriginal qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC ≡[R]
    interDotFirst qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC
```

**Closed-form spec defs (transitive):** `interOriginal`, `interDotFirst`, `original`, `interRequirements`, `dotFirst`, `source`, `Storage`, `sumLeft`, `scaleFirst`, `loadFirst`

<details><summary><code>interOriginal</code></summary>

```lean
def interOriginal : Program where
  kernel := TypedInter.original qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC
  domain := interRequirements (TypedInter.original qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC) BC BK
```
</details>

<details><summary><code>interDotFirst</code></summary>

```lean
def interDotFirst : Program :=
  { interOriginal qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC with
    kernel := TypedInter.dotFirst qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC }
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
```
</details>

<details><summary><code>interRequirements</code></summary>

```
/-- Finite checks cover the carry and dot in each reached K iteration after both early-return tests. -/
```
```lean
def interRequirements (kernel : ComputeKernel) (BC BK : Nat) : List Site :=
  (do
    let outer ← Source.conditional? kernel.surfaceBody (fun code =>
      (Source.conditional? code (fun body => (Source.dynamicRange? body "i_k").isSome)).isSome)
    let inner ← Source.conditional? outer.yes (fun body => (Source.dynamicRange? body "i_k").isSome)
    let loop ← Source.dynamicRange? inner.yes "i_k"
    return Site.branch outer.before outer.condition Bool.true
      (.branch inner.before inner.condition Bool.true
        (.range loop.before "i_k" loop.start loop.stop loop.stride loop.body
          (.addFP32 (Source.prefixBefore loop.body "b_A") [BC,BC]
            (.ref .real [BC,BC] "b_A")
            (.dot (batch := []) (.ref .real [BC,BK] "b_qg") (.ref .real [BK,BC] "b_kg") {}))))).toList
section
variable (qs ks gs : TypedInter.Storage) (out : FloatDType) (q k g A : RegionName)
  (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat)
```
</details>

<details><summary><code>dotFirst</code></summary>

```lean
def dotFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
```
</details>

<details><summary><code>source</code></summary>

```lean
def source (queryStorage valueStorage gateStorage : Storage) (swapped : Bool) :=
  match queryStorage,valueStorage,gateStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
```
</details>

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := TypedOutput.Storage
syntax "glaSplitSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaSplitSource%[$qTy:tritonDType, $kTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let scaled : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then `(tritonStmt| b_A = tl.where(o_i >= j, $((scale : ℝ)) * b_A, 0.0))
      else `(tritonStmt| b_A = tl.where(o_i >= j, b_A * $((scale : ℝ)), 0.0))
    `(fun (q k g A : RegionName)
        (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) =>
      ComputeKernel.mk [q,k,g] [A] (triton {
        tl.region q = $qTy, k = $kTy, g = $gTy, A = tl.float32
        i_k = tl.program_id(0)
        i_tc = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_t = i_tc // $(NC)
        i_i = i_tc % $(NC)
        i_j = i_i
        n_bh = tl.num_programs(2)
        if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
        o_i = tl.arange(0, $(BC))
        o_k = i_k * $(BK) + tl.arange(0, $(BK))
        o_A = (i_bh + i_k * n_bh) * $(T) * $(BC) + (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) * $(BC)
        m_k = o_k < $(K)
        m_A = (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) < $(T)
        p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
        b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
        for j in range($(0), min($(BC), $(T) - i_t * $(BT) - i_i * $(BC))) {
          b_A = tl.zeros([$(BC)], dtype=tl.float32)
          b_k = (tl.load(p_k, mask=m_k, other=0)).to(tl.float32)
          b_gk = (tl.load(p_gk, mask=m_k, other=0)).to(tl.float32)
          b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
          $scaled
          tl.store(A + o_A + j, b_A, mask=m_A)
          p_k += $(K)
          p_gk += $(K)
        }
      }).surfaceBody)
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

<details><summary><code>scaleFirst</code></summary>

```lean
def scaleFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedSplit

/-! Complete source of chunk_gla_fwd_A_kernel_intra_sub_intra_merge. The host
allocates its temporary input in fp32; output storage stays explicit. The source
early return is transcribed directly. Runtime head count,
all NK contributions, omitted padding and output conversion are retained. The
candidate commutes only the fp32 carry/load addition inside the loop. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge
open Lean VeriTile.Triton
syntax "glaMerge%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaMerge%[$version:ident]) => do
    unless version.getId == `original || version.getId == `loadFirst do
      Macro.throwError "expected original or loadFirst"
    let add : TSyntax `tritonStmt ←
      if version.getId == `loadFirst then
        `(tritonStmt| b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat)) + b_A)
      else `(tritonStmt| b_A += tl.load(p_A, boundary_check=([0, 1] : List Nat)))
    `(fun (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) =>
      ComputeKernel.mk [A] [A2] (triton {
        tl.region A = tl.float32, A2 = OUT_DTYPE
        i_t = tl.program_id(0)
        i_c = tl.program_id(1)
        i_bh = tl.program_id(2)
        if i_t * $(BT) + i_c * $(BC) >= $(T) { return }
        n_bh = tl.num_programs(2)
        b_A = tl.zeros([$(BC), $(BC)], dtype=tl.float32)
        for i_k in range($(0), $(NK)) {
          p_A = tl.make_block_ptr(base=A + (i_bh + i_k * n_bh) * $(T) * $(BC),
            shape=($(T), $(BC)), strides=($(BC), $(1)),
            offsets=(i_t * $(BT) + i_c * $(BC), $(0)),
            block_shape=($(BC), $(BC)), order=(1, 0))
          $add
        }
        p_A2 = tl.make_block_ptr(base=A2 + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT) + i_c * $(BC), i_c * $(BC)),
          block_shape=($(BC), $(BC)), order=(1, 0))
        tl.store(p_A2, (b_A).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>loadFirst</code></summary>

```lean
def loadFirst := glaMerge%[loadFirst]
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge

/-! Source: chunk_gla_fwd.py, chunk_gla_fwd_kernel_o.
Complete typed source of the output consumer.
Q, V and gate storage independently select fp32/fp16/bf16. History and output
storage remain explicit dtype parameters; A is fp32, as allocated by the host.
Query scale and exp-gate results are cast back to query storage. H is cast to
that same storage before the default-option inter-chunk dot; A is cast to V
storage before the TF32-disabled final dot. Omitted padding remains unspecified.
The candidate commutes only the final fp32 carry/intra-chunk dot addition.
All dimensions, strides and program IDs are symbolic. This source covers the
output consumer; the four A producers have separate proof obligations. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput
open Lean VeriTile.Triton

syntax "glaOutputSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaOutputSource%[$qTy:tritonDType, $vTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let finalAdd : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| b_o = tl.dot(b_A, b_v, allow_tf32=false) + b_o)
      else `(tritonStmt| b_o += tl.dot(b_A, b_v, allow_tf32=false))
    `(fun (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,v,g,h,A] [o] (triton {
        tl.region q = $qTy, v = $vTy, g = $gTy, h = DTYPE,
          o = OUT_DTYPE, A = tl.float32
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        m_s = tl.arange(0, $(BT))[:, None] >= tl.arange(0, $(BT))[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
            shape=($(K), $(V)), strides=($(s_h_t), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_q = (b_q * $((scale : ℝ))).to($qTy:tritonDType)
          b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
          b_qg = (b_q * tl.exp(b_g)).to($qTy:tritonDType)
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          if i_k >= $(0) {
            b_o += tl.dot(b_qg, (b_h).to($qTy:tritonDType))
          }
        }
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT), $(0)), block_shape=($(BT), $(BT)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
        b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat))
        b_A = (tl.where(m_s, b_A, 0.0)).to($vTy:tritonDType)
        $finalAdd
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

## Public theorem: `intra_equiv`

**Statement:**
```lean
specification intra_equiv (R : FP.ScalarArithmetic.Rules) :
    intraOriginal qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK ≡[R]
    intraScaleFirst qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK
```

**Closed-form spec defs (transitive):** `intraOriginal`, `intraScaleFirst`, `original`, `intraRequirements`, `scaleFirst`, `source`, `Storage`, `loadFirst`, `sumLeft`

<details><summary><code>intraOriginal</code></summary>

```lean
def intraOriginal : Program where
  kernel := TypedIntra.original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK
  domain := intraRequirements (TypedIntra.original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK) BC scale
```
</details>

<details><summary><code>intraScaleFirst</code></summary>

```lean
def intraScaleFirst : Program :=
  { intraOriginal qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK with
    kernel := TypedIntra.scaleFirst qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK }
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
```
</details>

<details><summary><code>intraRequirements</code></summary>

```
/-- Finite checks cover the sum and scale for each reached column, on all lanes of where. -/
```
```lean
def intraRequirements (kernel : ComputeKernel) (BC : Nat) (scale : ℝ) : List Site :=
  (do
    let active ← Source.conditional? kernel.surfaceBody (fun code => (Source.dynamicRange? code "j").isSome)
    let loop ← Source.dynamicRange? active.yes "j"
    return Site.branch active.before active.condition Bool.true
      (.range loop.before "j" loop.start loop.stop loop.stride loop.body
        (.mulScalar (Source.prefixBefore loop.body "b_A" 1) BC []
          (.ref .real [BC] "b_A") (.const scale)))).toList
section
variable (qs ks gs : TypedIntra.Storage) (q k g A : RegionName)
  (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK : Nat)
```
</details>

<details><summary><code>scaleFirst</code></summary>

```lean
def scaleFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedSplit

/-! Complete source of chunk_gla_fwd_A_kernel_intra_sub_intra_merge. The host
allocates its temporary input in fp32; output storage stays explicit. The source
early return is transcribed directly. Runtime head count,
all NK contributions, omitted padding and output conversion are retained. The
candidate commutes only the fp32 carry/load addition inside the loop. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge
open Lean VeriTile.Triton
syntax "glaMerge%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaMerge%[$version:ident]) => do
    unless version.getId == `original || version.getId == `loadFirst do
      Macro.throwError "expected original or loadFirst"
    let add : TSyntax `tritonStmt ←
      if version.getId == `loadFirst then
        `(tritonStmt| b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat)) + b_A)
      else `(tritonStmt| b_A += tl.load(p_A, boundary_check=([0, 1] : List Nat)))
    `(fun (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) =>
      ComputeKernel.mk [A] [A2] (triton {
        tl.region A = tl.float32, A2 = OUT_DTYPE
        i_t = tl.program_id(0)
        i_c = tl.program_id(1)
        i_bh = tl.program_id(2)
        if i_t * $(BT) + i_c * $(BC) >= $(T) { return }
        n_bh = tl.num_programs(2)
        b_A = tl.zeros([$(BC), $(BC)], dtype=tl.float32)
        for i_k in range($(0), $(NK)) {
          p_A = tl.make_block_ptr(base=A + (i_bh + i_k * n_bh) * $(T) * $(BC),
            shape=($(T), $(BC)), strides=($(BC), $(1)),
            offsets=(i_t * $(BT) + i_c * $(BC), $(0)),
            block_shape=($(BC), $(BC)), order=(1, 0))
          $add
        }
        p_A2 = tl.make_block_ptr(base=A2 + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT) + i_c * $(BC), i_c * $(BC)),
          block_shape=($(BC), $(BC)), order=(1, 0))
        tl.store(p_A2, (b_A).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>source</code></summary>

```lean
def source (queryStorage valueStorage gateStorage : Storage) (swapped : Bool) :=
  match queryStorage,valueStorage,gateStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
```
</details>

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := TypedOutput.Storage
syntax "glaSplitSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaSplitSource%[$qTy:tritonDType, $kTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let scaled : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then `(tritonStmt| b_A = tl.where(o_i >= j, $((scale : ℝ)) * b_A, 0.0))
      else `(tritonStmt| b_A = tl.where(o_i >= j, b_A * $((scale : ℝ)), 0.0))
    `(fun (q k g A : RegionName)
        (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) =>
      ComputeKernel.mk [q,k,g] [A] (triton {
        tl.region q = $qTy, k = $kTy, g = $gTy, A = tl.float32
        i_k = tl.program_id(0)
        i_tc = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_t = i_tc // $(NC)
        i_i = i_tc % $(NC)
        i_j = i_i
        n_bh = tl.num_programs(2)
        if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
        o_i = tl.arange(0, $(BC))
        o_k = i_k * $(BK) + tl.arange(0, $(BK))
        o_A = (i_bh + i_k * n_bh) * $(T) * $(BC) + (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) * $(BC)
        m_k = o_k < $(K)
        m_A = (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) < $(T)
        p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
        b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
        for j in range($(0), min($(BC), $(T) - i_t * $(BT) - i_i * $(BC))) {
          b_A = tl.zeros([$(BC)], dtype=tl.float32)
          b_k = (tl.load(p_k, mask=m_k, other=0)).to(tl.float32)
          b_gk = (tl.load(p_gk, mask=m_k, other=0)).to(tl.float32)
          b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
          $scaled
          tl.store(A + o_A + j, b_A, mask=m_A)
          p_k += $(K)
          p_gk += $(K)
        }
      }).surfaceBody)
```
</details>

<details><summary><code>loadFirst</code></summary>

```lean
def loadFirst := glaMerge%[loadFirst]
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge

/-! Source: chunk_gla_fwd.py, chunk_gla_fwd_kernel_o.
Complete typed source of the output consumer.
Q, V and gate storage independently select fp32/fp16/bf16. History and output
storage remain explicit dtype parameters; A is fp32, as allocated by the host.
Query scale and exp-gate results are cast back to query storage. H is cast to
that same storage before the default-option inter-chunk dot; A is cast to V
storage before the TF32-disabled final dot. Omitted padding remains unspecified.
The candidate commutes only the final fp32 carry/intra-chunk dot addition.
All dimensions, strides and program IDs are symbolic. This source covers the
output consumer; the four A producers have separate proof obligations. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput
open Lean VeriTile.Triton

syntax "glaOutputSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaOutputSource%[$qTy:tritonDType, $vTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let finalAdd : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| b_o = tl.dot(b_A, b_v, allow_tf32=false) + b_o)
      else `(tritonStmt| b_o += tl.dot(b_A, b_v, allow_tf32=false))
    `(fun (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,v,g,h,A] [o] (triton {
        tl.region q = $qTy, v = $vTy, g = $gTy, h = DTYPE,
          o = OUT_DTYPE, A = tl.float32
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        m_s = tl.arange(0, $(BT))[:, None] >= tl.arange(0, $(BT))[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
            shape=($(K), $(V)), strides=($(s_h_t), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_q = (b_q * $((scale : ℝ))).to($qTy:tritonDType)
          b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
          b_qg = (b_q * tl.exp(b_g)).to($qTy:tritonDType)
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          if i_k >= $(0) {
            b_o += tl.dot(b_qg, (b_h).to($qTy:tritonDType))
          }
        }
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT), $(0)), block_shape=($(BT), $(BT)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
        b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat))
        b_A = (tl.where(m_s, b_A, 0.0)).to($vTy:tritonDType)
        $finalAdd
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

## Public theorem: `split_equiv`

**Statement:**
```lean
specification split_equiv (R : FP.ScalarArithmetic.Rules) :
    splitOriginal qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC ≡[R]
    splitScaleFirst qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC
```

**Closed-form spec defs (transitive):** `splitOriginal`, `splitScaleFirst`, `original`, `splitRequirements`, `scaleFirst`, `source`, `Storage`, `loadFirst`, `sumLeft`

<details><summary><code>splitOriginal</code></summary>

```lean
def splitOriginal : Program where
  kernel := TypedSplit.original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC
  domain := splitRequirements (TypedSplit.original qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC) BC scale
```
</details>

<details><summary><code>splitScaleFirst</code></summary>

```lean
def splitScaleFirst : Program :=
  { splitOriginal qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC with
    kernel := TypedSplit.scaleFirst qs ks gs q k g A s_k_h s_k_t scale T K BT BC BK NC }
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
```
</details>

<details><summary><code>splitRequirements</code></summary>

```
/-- Finite checks cover the sum after the source zero/add steps and scale, on all lanes of where. -/
```
```lean
def splitRequirements (kernel : ComputeKernel) (BC : Nat) (scale : ℝ) : List Site :=
  (do
    let active ← Source.conditional? kernel.surfaceBody (fun code => (Source.dynamicRange? code "j").isSome)
    let loop ← Source.dynamicRange? active.yes "j"
    return Site.branch active.before active.condition Bool.true
      (.range loop.before "j" loop.start loop.stop loop.stride loop.body
        (.mulScalar (Source.prefixBefore loop.body "b_A" 2) BC []
          (.ref .real [BC] "b_A") (.const scale)))).toList
section
variable (qs ks gs : TypedSplit.Storage) (q k g A : RegionName)
  (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat)
```
</details>

<details><summary><code>scaleFirst</code></summary>

```lean
def scaleFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedSplit

/-! Complete source of chunk_gla_fwd_A_kernel_intra_sub_intra_merge. The host
allocates its temporary input in fp32; output storage stays explicit. The source
early return is transcribed directly. Runtime head count,
all NK contributions, omitted padding and output conversion are retained. The
candidate commutes only the fp32 carry/load addition inside the loop. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge
open Lean VeriTile.Triton
syntax "glaMerge%" "[" ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaMerge%[$version:ident]) => do
    unless version.getId == `original || version.getId == `loadFirst do
      Macro.throwError "expected original or loadFirst"
    let add : TSyntax `tritonStmt ←
      if version.getId == `loadFirst then
        `(tritonStmt| b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat)) + b_A)
      else `(tritonStmt| b_A += tl.load(p_A, boundary_check=([0, 1] : List Nat)))
    `(fun (out_dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) =>
      ComputeKernel.mk [A] [A2] (triton {
        tl.region A = tl.float32, A2 = OUT_DTYPE
        i_t = tl.program_id(0)
        i_c = tl.program_id(1)
        i_bh = tl.program_id(2)
        if i_t * $(BT) + i_c * $(BC) >= $(T) { return }
        n_bh = tl.num_programs(2)
        b_A = tl.zeros([$(BC), $(BC)], dtype=tl.float32)
        for i_k in range($(0), $(NK)) {
          p_A = tl.make_block_ptr(base=A + (i_bh + i_k * n_bh) * $(T) * $(BC),
            shape=($(T), $(BC)), strides=($(BC), $(1)),
            offsets=(i_t * $(BT) + i_c * $(BC), $(0)),
            block_shape=($(BC), $(BC)), order=(1, 0))
          $add
        }
        p_A2 = tl.make_block_ptr(base=A2 + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT) + i_c * $(BC), i_c * $(BC)),
          block_shape=($(BC), $(BC)), order=(1, 0))
        tl.store(p_A2, (b_A).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>source</code></summary>

```lean
def source (queryStorage valueStorage gateStorage : Storage) (swapped : Bool) :=
  match queryStorage,valueStorage,gateStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
```
</details>

<details><summary><code>Storage</code></summary>

```lean
abbrev Storage := TypedOutput.Storage
syntax "glaSplitSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaSplitSource%[$qTy:tritonDType, $kTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `scaleFirst do
      Macro.throwError "expected original or scaleFirst"
    let scaled : TSyntax `tritonStmt ←
      if version.getId == `scaleFirst then `(tritonStmt| b_A = tl.where(o_i >= j, $((scale : ℝ)) * b_A, 0.0))
      else `(tritonStmt| b_A = tl.where(o_i >= j, b_A * $((scale : ℝ)), 0.0))
    `(fun (q k g A : RegionName)
        (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat) =>
      ComputeKernel.mk [q,k,g] [A] (triton {
        tl.region q = $qTy, k = $kTy, g = $gTy, A = tl.float32
        i_k = tl.program_id(0)
        i_tc = tl.program_id(1)
        i_bh = tl.program_id(2)
        i_t = i_tc // $(NC)
        i_i = i_tc % $(NC)
        i_j = i_i
        n_bh = tl.num_programs(2)
        if i_t * $(BT) + i_i * $(BC) >= $(T) { return }
        o_i = tl.arange(0, $(BC))
        o_k = i_k * $(BK) + tl.arange(0, $(BK))
        o_A = (i_bh + i_k * n_bh) * $(T) * $(BC) + (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) * $(BC)
        m_k = o_k < $(K)
        m_A = (i_t * $(BT) + i_i * $(BC) + tl.arange(0, $(BC))) < $(T)
        p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
          shape=($(T), $(K)), strides=($(s_k_t), $(1)),
          offsets=(i_t * $(BT) + i_i * $(BC), i_k * $(BK)), block_shape=($(BC), $(BK)), order=(1, 0))
        p_k = tl.max_contiguous(tl.multiple_of(k + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        p_gk = tl.max_contiguous(tl.multiple_of(g + i_bh * $(s_k_h) + (i_t * $(BT) + i_j * $(BC)) * $(K) + o_k, $(BK)), $(BK))
        b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
        b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
        for j in range($(0), min($(BC), $(T) - i_t * $(BT) - i_i * $(BC))) {
          b_A = tl.zeros([$(BC)], dtype=tl.float32)
          b_k = (tl.load(p_k, mask=m_k, other=0)).to(tl.float32)
          b_gk = (tl.load(p_gk, mask=m_k, other=0)).to(tl.float32)
          b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
          $scaled
          tl.store(A + o_A + j, b_A, mask=m_A)
          p_k += $(K)
          p_gk += $(K)
        }
      }).surfaceBody)
```
</details>

<details><summary><code>loadFirst</code></summary>

```lean
def loadFirst := glaMerge%[loadFirst]
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge

/-! Source: chunk_gla_fwd.py, chunk_gla_fwd_kernel_o.
Complete typed source of the output consumer.
Q, V and gate storage independently select fp32/fp16/bf16. History and output
storage remain explicit dtype parameters; A is fp32, as allocated by the host.
Query scale and exp-gate results are cast back to query storage. H is cast to
that same storage before the default-option inter-chunk dot; A is cast to V
storage before the TF32-disabled final dot. Omitted padding remains unspecified.
The candidate commutes only the final fp32 carry/intra-chunk dot addition.
All dimensions, strides and program IDs are symbolic. This source covers the
output consumer; the four A producers have separate proof obligations. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput
open Lean VeriTile.Triton

syntax "glaOutputSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaOutputSource%[$qTy:tritonDType, $vTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let finalAdd : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| b_o = tl.dot(b_A, b_v, allow_tf32=false) + b_o)
      else `(tritonStmt| b_o += tl.dot(b_A, b_v, allow_tf32=false))
    `(fun (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,v,g,h,A] [o] (triton {
        tl.region q = $qTy, v = $vTy, g = $gTy, h = DTYPE,
          o = OUT_DTYPE, A = tl.float32
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        m_s = tl.arange(0, $(BT))[:, None] >= tl.arange(0, $(BT))[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
            shape=($(K), $(V)), strides=($(s_h_t), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_q = (b_q * $((scale : ℝ))).to($qTy:tritonDType)
          b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
          b_qg = (b_q * tl.exp(b_g)).to($qTy:tritonDType)
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          if i_k >= $(0) {
            b_o += tl.dot(b_qg, (b_h).to($qTy:tritonDType))
          }
        }
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT), $(0)), block_shape=($(BT), $(BT)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
        b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat))
        b_A = (tl.where(m_s, b_A, 0.0)).to($vTy:tritonDType)
        $finalAdd
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

## Public theorem: `merge_equiv`

**Statement:**
```lean
specification merge_equiv (dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat)
    (R : FP.ScalarArithmetic.Rules) :
    mergeOriginal dtype A A2 T BT BC NK ≡[R] mergeLoadFirst dtype A A2 T BT BC NK
```

**Closed-form spec defs (transitive):** `mergeOriginal`, `mergeLoadFirst`, `original`, `mergeRequirements`, `loadFirst`, `source`, `sumLeft`

<details><summary><code>mergeOriginal</code></summary>

```lean
def mergeOriginal (dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : Program where
  kernel := TypedMerge.original dtype A A2 T BT BC NK
  domain := mergeRequirements (TypedMerge.original dtype A A2 T BT BC NK) BC
```
</details>

<details><summary><code>mergeLoadFirst</code></summary>

```lean
def mergeLoadFirst (dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : Program :=
  { mergeOriginal dtype A A2 T BT BC NK with kernel := TypedMerge.loadFirst dtype A A2 T BT BC NK }
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
```
</details>

<details><summary><code>mergeRequirements</code></summary>

```
/-- Finite checks cover the carry and checked load in every reached NK iteration. -/
```
```lean
def mergeRequirements (kernel : ComputeKernel) (BC : Nat) : List Site :=
  (do
    let branch ← Source.conditional? kernel.surfaceBody (fun body => (Source.dynamicRange? body "i_k").isSome)
    let loop ← Source.dynamicRange? branch.yes "i_k"
    return Site.branch branch.before branch.condition Bool.true
      (.range loop.before "i_k" loop.start loop.stop loop.stride loop.body
        (.addFP32 (Source.prefixBefore loop.body "b_A") [BC,BC]
          (.ref .real [BC,BC] "b_A")
          (.load .real (.blockPtr (.ref .blockPtr [BC,BC] "p_A") [0,1]) .none .unspecified)))).toList
```
</details>

<details><summary><code>loadFirst</code></summary>

```lean
def loadFirst := glaMerge%[loadFirst]
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedMerge

/-! Source: chunk_gla_fwd.py, chunk_gla_fwd_kernel_o.
Complete typed source of the output consumer.
Q, V and gate storage independently select fp32/fp16/bf16. History and output
storage remain explicit dtype parameters; A is fp32, as allocated by the host.
Query scale and exp-gate results are cast back to query storage. H is cast to
that same storage before the default-option inter-chunk dot; A is cast to V
storage before the TF32-disabled final dot. Omitted padding remains unspecified.
The candidate commutes only the final fp32 carry/intra-chunk dot addition.
All dimensions, strides and program IDs are symbolic. This source covers the
output consumer; the four A producers have separate proof obligations. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.TypedOutput
open Lean VeriTile.Triton

syntax "glaOutputSource%" "[" tritonDType "," tritonDType "," tritonDType "," ident "]" : term
set_option hygiene false in
macro_rules
  | `(glaOutputSource%[$qTy:tritonDType, $vTy:tritonDType, $gTy:tritonDType, $version:ident]) => do
    unless version.getId == `original || version.getId == `sumLeft do
      Macro.throwError "expected original or sumLeft"
    let finalAdd : TSyntax `tritonStmt ←
      if version.getId == `sumLeft then `(tritonStmt| b_o = tl.dot(b_A, b_v, allow_tf32=false) + b_o)
      else `(tritonStmt| b_o += tl.dot(b_A, b_v, allow_tf32=false))
    `(fun (dtype out_dtype : FloatDType) (q v g h o A : RegionName)
        (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
        (scale : ℝ) (T K V BT BK BV : Nat) =>
      ComputeKernel.mk [q,v,g,h,A] [o] (triton {
        tl.region q = $qTy, v = $vTy, g = $gTy, h = DTYPE,
          o = OUT_DTYPE, A = tl.float32
        i_v = tl.program_id(0)
        i_t = tl.program_id(1)
        i_bh = tl.program_id(2)
        m_s = tl.arange(0, $(BT))[:, None] >= tl.arange(0, $(BT))[None, :]
        b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
        for i_k in range(tl.cdiv($(K), $(BK))) {
          p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_g = tl.make_block_ptr(base=g + i_bh * $(s_k_h),
            shape=($(T), $(K)), strides=($(s_k_t), $(1)),
            offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
          p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
            shape=($(K), $(V)), strides=($(s_h_t), $(1)),
            offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
          b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
          b_q = (b_q * $((scale : ℝ))).to($qTy:tritonDType)
          b_g = tl.load(p_g, boundary_check=([0, 1] : List Nat))
          b_qg = (b_q * tl.exp(b_g)).to($qTy:tritonDType)
          b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
          if i_k >= $(0) {
            b_o += tl.dot(b_qg, (b_h).to($qTy:tritonDType))
          }
        }
        p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
          shape=($(T), $(V)), strides=($(s_v_t), $(1)),
          offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
        p_A = tl.make_block_ptr(base=A + i_bh * $(T) * $(BT),
          shape=($(T), $(BT)), strides=($(BT), $(1)),
          offsets=(i_t * $(BT), $(0)), block_shape=($(BT), $(BT)), order=(1, 0))
        b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
        b_A = tl.load(p_A, boundary_check=([0, 1] : List Nat))
        b_A = (tl.where(m_s, b_A, 0.0)).to($vTy:tritonDType)
        $finalAdd
        tl.store(p_o, (b_o).to(OUT_DTYPE), boundary_check=([0, 1] : List Nat))
      }).surfaceBody)
```
</details>

<details><summary><code>source</code></summary>

```lean
def source (queryStorage valueStorage gateStorage : Storage) (swapped : Bool) :=
  match queryStorage,valueStorage,gateStorage with
  | .fp32, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float32, original]
  | .fp32, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.float16, original]
  | .fp32, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float32, tl.bfloat16, original]
  | .fp32, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float32, original]
  | .fp32, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.float16, original]
  | .fp32, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.float16, tl.bfloat16, original]
  | .fp32, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float32, original]
  | .fp32, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.float16, original]
  | .fp32, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float32, tl.bfloat16, tl.bfloat16, original]
  | .fp16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float32, original]
  | .fp16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.float16, original]
  | .fp16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float32, tl.bfloat16, original]
  | .fp16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float32, original]
  | .fp16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.float16, original]
  | .fp16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.float16, tl.bfloat16, original]
  | .fp16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float32, original]
  | .fp16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.float16, original]
  | .fp16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.float16, tl.bfloat16, tl.bfloat16, original]
  | .bf16, .fp32, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float32, original]
  | .bf16, .fp32, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.float16, original]
  | .bf16, .fp32, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float32, tl.bfloat16, original]
  | .bf16, .fp16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float32, original]
  | .bf16, .fp16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.float16, original]
  | .bf16, .fp16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.float16, tl.bfloat16, original]
  | .bf16, .bf16, .fp32 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float32, original]
  | .bf16, .bf16, .fp16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.float16, original]
  | .bf16, .bf16, .bf16 =>
      if swapped then glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, sumLeft]
      else glaOutputSource%[tl.bfloat16, tl.bfloat16, tl.bfloat16, original]
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>
