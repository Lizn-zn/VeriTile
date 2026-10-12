# Spec sheet — `bench/tritonbench_g/chunk_gla_fwd/Correct.lean`

**Python source:** `bench/tritonbench_g/chunk_gla_fwd/chunk_gla_fwd.py`

## Public theorem: `output_correct`

**Statement:**
```lean
specification output_correct (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) :
    outputOriginal p r qs vs gs history out scale ⊨ outputFormula p scale
```

**Closed-form spec defs (transitive):** `Storage`, `outputOriginal`, `outputFormula`, `original`, `scaleFirst`, `outputAddress`, `active`, `source`, `loadFirst`, `rowStart`, `enabled`, `sumLeft`

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

<details><summary><code>outputOriginal</code></summary>

```
/-- Bounds cover every active Q/G/H K-block load, A/V load and output store.
Output addresses must be injective; all inputs are read before the store. -/
```
```lean
def outputOriginal (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) : IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (TypedOutput.original qs vs gs history out r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV).eraseDType
  projection := RealOutputIO.projection p r qs vs gs history out scale Bool.false
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.value,s.readMem r.gate,s.readMem r.history,s.readMem r.attention⟩
  valid := fun s => Function.Injective (outputAddress p (launch s))
  bounds := fun s => RealOutputSafety.Bounds s r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey p.T p.K p.V p.BT p.BK p.BV
  out := r.output
  write := outputAddress p
  mask := active p
```
</details>

<details><summary><code>outputFormula</code></summary>

```
/-- Full K-block gated history contraction plus causal A·V, with zero extension
outside the source block pointers' checked parents. -/
```
```lean
def outputFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
  let t := l.chunk*p.BT+i.1.val
  let v := l.value*p.BV+i.2.1.val
  (∑ kb : Fin ((p.K+p.BK-1)/p.BK), ∑ e : Fin p.BK,
    let k := kb.val*p.BK+e.val
    (if t < p.T ∧ k < p.K then
      a.query (l.head*p.keyHead+t*p.keyTime+k) * scale * Real.exp (a.gate (l.head*p.keyHead+t*p.keyTime+k)) else 0) *
    (if k < p.K ∧ v < p.V then
      a.history (l.head*p.historyHead+l.chunk*p.K*p.V+k*p.historyKey+v) else 0)) +
  ∑ j : Fin p.BT,
    (if j.val ≤ i.1.val then (if t < p.T then a.attention (l.head*p.T*p.BT+t*p.BT+j.val) else 0) else 0) *
    (if l.chunk*p.BT+j.val < p.T ∧ v < p.V then
      a.value (l.head*p.valueHead+(l.chunk*p.BT+j.val)*p.valueTime+v) else 0)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
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

<details><summary><code>outputAddress</code></summary>

```lean
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Prop :=
  rowStart p l+i.1.val < p.T ∧ l.subblock*p.BC+i.2.1.val < p.BT
instance (p : Layout) (l : Launch) : Decidable (enabled p l) := inferInstanceAs (Decidable (_ < _))
instance (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Decidable (active p l i) :=
  inferInstanceAs (Decidable (_ ∧ _))
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

<details><summary><code>rowStart</code></summary>

```lean
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
```
</details>

<details><summary><code>enabled</code></summary>

```lean
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

## Public theorem: `output_sumLeft_correct`

**Statement:**
```lean
specification output_sumLeft_correct (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) :
    outputSumLeft p r qs vs gs history out scale ⊨ outputFormula p scale
```

**Closed-form spec defs (transitive):** `Storage`, `outputSumLeft`, `outputFormula`, `original`, `scaleFirst`, `outputOriginal`, `sumLeft`, `source`, `loadFirst`, `outputAddress`, `active`, `rowStart`, `enabled`

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

<details><summary><code>outputSumLeft</code></summary>

```
/-- Commute the final carry/dot addition; retain the same IO contract. -/
```
```lean
def outputSumLeft (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) : IO Launch Inputs (TileIndex [p.BT,p.BV]) :=
  { outputOriginal p r qs vs gs history out scale with
    kernel := (TypedOutput.sumLeft qs vs gs history out r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV).eraseDType
    projection := RealOutputIO.projection p r qs vs gs history out scale Bool.true }
```
</details>

<details><summary><code>outputFormula</code></summary>

```
/-- Full K-block gated history contraction plus causal A·V, with zero extension
outside the source block pointers' checked parents. -/
```
```lean
def outputFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BT,p.BV]) : ℝ :=
  let t := l.chunk*p.BT+i.1.val
  let v := l.value*p.BV+i.2.1.val
  (∑ kb : Fin ((p.K+p.BK-1)/p.BK), ∑ e : Fin p.BK,
    let k := kb.val*p.BK+e.val
    (if t < p.T ∧ k < p.K then
      a.query (l.head*p.keyHead+t*p.keyTime+k) * scale * Real.exp (a.gate (l.head*p.keyHead+t*p.keyTime+k)) else 0) *
    (if k < p.K ∧ v < p.V then
      a.history (l.head*p.historyHead+l.chunk*p.K*p.V+k*p.historyKey+v) else 0)) +
  ∑ j : Fin p.BT,
    (if j.val ≤ i.1.val then (if t < p.T then a.attention (l.head*p.T*p.BT+t*p.BT+j.val) else 0) else 0) *
    (if l.chunk*p.BT+j.val < p.T ∧ v < p.V then
      a.value (l.head*p.valueHead+(l.chunk*p.BT+j.val)*p.valueTime+v) else 0)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
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

<details><summary><code>outputOriginal</code></summary>

```
/-- Bounds cover every active Q/G/H K-block load, A/V load and output store.
Output addresses must be injective; all inputs are read before the store. -/
```
```lean
def outputOriginal (p : Layout) (r : Regions) (qs vs gs : TypedOutput.Storage)
    (history out : FloatDType) (scale : ℝ) : IO Launch Inputs (TileIndex [p.BT,p.BV]) where
  kernel := (TypedOutput.original qs vs gs history out r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey scale p.T p.K p.V p.BT p.BK p.BV).eraseDType
  projection := RealOutputIO.projection p r qs vs gs history out scale Bool.false
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.value,s.readMem r.gate,s.readMem r.history,s.readMem r.attention⟩
  valid := fun s => Function.Injective (outputAddress p (launch s))
  bounds := fun s => RealOutputSafety.Bounds s r.query r.value r.gate r.history r.output r.attention
    p.keyHead p.keyTime p.valueHead p.valueTime p.historyHead p.historyKey p.T p.K p.V p.BT p.BK p.BV
  out := r.output
  write := outputAddress p
  mask := active p
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

<details><summary><code>outputAddress</code></summary>

```lean
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Prop :=
  rowStart p l+i.1.val < p.T ∧ l.subblock*p.BC+i.2.1.val < p.BT
instance (p : Layout) (l : Launch) : Decidable (enabled p l) := inferInstanceAs (Decidable (_ < _))
instance (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Decidable (active p l i) :=
  inferInstanceAs (Decidable (_ ∧ _))
```
</details>

<details><summary><code>rowStart</code></summary>

```lean
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
```
</details>

<details><summary><code>enabled</code></summary>

```lean
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
```
</details>

## Public theorem: `inter_correct`

**Statement:**
```lean
specification inter_correct (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) :
    interOriginal p r qs ks gs out scale ⊨ interFormula p scale
```

**Closed-form spec defs (transitive):** `Storage`, `interOriginal`, `interFormula`, `original`, `scaleFirst`, `outputAddress`, `enabled`, `active`, `matrixValue`, `rowStart`, `columnStart`, `source`, `loadFirst`, `matrixAddress`, `columnBlock`, `sumLeft`, `featureIndex`

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

<details><summary><code>interOriginal</code></summary>

```
/-- Bounds cover Q/G/K, the separate flat gate reference and the final store
on each reached K block. Inputs may alias output because reads precede the store. -/
```
```lean
def interOriginal (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) : IO Launch Inputs (TileIndex [p.BC,p.BC]) where
  kernel := (TypedInter.original qs ks gs out r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC).eraseDType
  projection := RealInterIO.projection p r qs ks gs out scale Bool.false
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun _ => True
  bounds := fun s => RealInterSafety.Bounds p (launch s) r
  out := r.output
  write := fun l i => outputAddress p l i.1.val i.2.1.val
  mask := fun l i => enabled p l ∧ active p l i.1.val i.2.1.val
```
</details>

<details><summary><code>interFormula</code></summary>

```
/-- Full K-block contraction, with zero extension beyond checked input parents. -/
```
```lean
def interFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  ∑ kb ∈ Finset.range ((p.K+p.BK-1)/p.BK), ∑ e : Fin p.BK,
    let k := kb*p.BK+e.val
    matrixValue p l a.query (rowStart p l+i.1.val) k *
      matrixValue p l a.key (columnStart p l+i.2.1.val) k *
      Real.exp (matrixValue p l a.gate (rowStart p l+i.1.val) k -
        matrixValue p l a.gate (columnStart p l+i.2.1.val) k) * scale
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
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

<details><summary><code>outputAddress</code></summary>

```lean
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
```
</details>

<details><summary><code>enabled</code></summary>

```lean
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Prop :=
  rowStart p l+i.1.val < p.T ∧ l.subblock*p.BC+i.2.1.val < p.BT
instance (p : Layout) (l : Launch) : Decidable (enabled p l) := inferInstanceAs (Decidable (_ < _))
instance (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Decidable (active p l i) :=
  inferInstanceAs (Decidable (_ ∧ _))
```
</details>

<details><summary><code>matrixValue</code></summary>

```lean
def matrixValue (p : Layout) (l : Launch) (a : Nat → ℝ) (t k : Nat) : ℝ :=
  if t < p.T ∧ k < p.K then a (matrixAddress p l t k) else 0
```
</details>

<details><summary><code>rowStart</code></summary>

```lean
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
```
</details>

<details><summary><code>columnStart</code></summary>

```lean
def columnStart (p : Layout) (l : Launch) := l.chunk*p.BT+columnBlock p l*p.BC
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

<details><summary><code>matrixAddress</code></summary>

```lean
def matrixAddress (p : Layout) (l : Launch) (i k : Nat) :=
  l.head*p.headStride+(rowStart p l+i)*p.timeStride+featureIndex p l k
```
</details>

<details><summary><code>columnBlock</code></summary>

```lean
def columnBlock (p : Layout) (l : Launch) := l.pair % p.NC
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

<details><summary><code>featureIndex</code></summary>

```lean
def featureIndex (p : Layout) (l : Launch) (e : Nat) := l.feature*p.BK+e
```
</details>

## Public theorem: `inter_dotFirst_correct`

**Statement:**
```lean
specification inter_dotFirst_correct (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) :
    interDotFirst p r qs ks gs out scale ⊨ interFormula p scale
```

**Closed-form spec defs (transitive):** `Storage`, `interDotFirst`, `interFormula`, `original`, `scaleFirst`, `interOriginal`, `dotFirst`, `matrixValue`, `rowStart`, `columnStart`, `source`, `loadFirst`, `outputAddress`, `enabled`, `active`, `matrixAddress`, `columnBlock`, `sumLeft`, `featureIndex`

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

<details><summary><code>interDotFirst</code></summary>

```lean
def interDotFirst (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) : IO Launch Inputs (TileIndex [p.BC,p.BC]) :=
  { interOriginal p r qs ks gs out scale with
    kernel := (TypedInter.dotFirst qs ks gs out r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC).eraseDType
    projection := RealInterIO.projection p r qs ks gs out scale Bool.true }
```
</details>

<details><summary><code>interFormula</code></summary>

```
/-- Full K-block contraction, with zero extension beyond checked input parents. -/
```
```lean
def interFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  ∑ kb ∈ Finset.range ((p.K+p.BK-1)/p.BK), ∑ e : Fin p.BK,
    let k := kb*p.BK+e.val
    matrixValue p l a.query (rowStart p l+i.1.val) k *
      matrixValue p l a.key (columnStart p l+i.2.1.val) k *
      Real.exp (matrixValue p l a.gate (rowStart p l+i.1.val) k -
        matrixValue p l a.gate (columnStart p l+i.2.1.val) k) * scale
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
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

<details><summary><code>interOriginal</code></summary>

```
/-- Bounds cover Q/G/K, the separate flat gate reference and the final store
on each reached K block. Inputs may alias output because reads precede the store. -/
```
```lean
def interOriginal (p : Layout) (r : Regions) (qs ks gs : TypedInter.Storage)
    (out : FloatDType) (scale : ℝ) : IO Launch Inputs (TileIndex [p.BC,p.BC]) where
  kernel := (TypedInter.original qs ks gs out r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC).eraseDType
  projection := RealInterIO.projection p r qs ks gs out scale Bool.false
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun _ => True
  bounds := fun s => RealInterSafety.Bounds p (launch s) r
  out := r.output
  write := fun l i => outputAddress p l i.1.val i.2.1.val
  mask := fun l i => enabled p l ∧ active p l i.1.val i.2.1.val
```
</details>

<details><summary><code>dotFirst</code></summary>

```lean
def dotFirst (qStorage kStorage gStorage : Storage) := source qStorage kStorage gStorage Bool.true
```
</details>

<details><summary><code>matrixValue</code></summary>

```lean
def matrixValue (p : Layout) (l : Launch) (a : Nat → ℝ) (t k : Nat) : ℝ :=
  if t < p.T ∧ k < p.K then a (matrixAddress p l t k) else 0
```
</details>

<details><summary><code>rowStart</code></summary>

```lean
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
```
</details>

<details><summary><code>columnStart</code></summary>

```lean
def columnStart (p : Layout) (l : Launch) := l.chunk*p.BT+columnBlock p l*p.BC
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

<details><summary><code>outputAddress</code></summary>

```lean
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
```
</details>

<details><summary><code>enabled</code></summary>

```lean
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Prop :=
  rowStart p l+i.1.val < p.T ∧ l.subblock*p.BC+i.2.1.val < p.BT
instance (p : Layout) (l : Launch) : Decidable (enabled p l) := inferInstanceAs (Decidable (_ < _))
instance (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Decidable (active p l i) :=
  inferInstanceAs (Decidable (_ ∧ _))
```
</details>

<details><summary><code>matrixAddress</code></summary>

```lean
def matrixAddress (p : Layout) (l : Launch) (i k : Nat) :=
  l.head*p.headStride+(rowStart p l+i)*p.timeStride+featureIndex p l k
```
</details>

<details><summary><code>columnBlock</code></summary>

```lean
def columnBlock (p : Layout) (l : Launch) := l.pair % p.NC
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

<details><summary><code>featureIndex</code></summary>

```lean
def featureIndex (p : Layout) (l : Launch) (e : Nat) := l.feature*p.BK+e
```
</details>

## Public theorem: `intra_correct`

**Statement:**
```lean
specification intra_correct (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hK : p.K ≤ p.BK) :
    intraOriginal p r qs ks gs scale ⊨ intraFormula p scale
```

**Assumptions / layout contracts:**
- `hkey : r.key ≠ r.output`
- `hgate : r.gate ≠ r.output`
- `hK : p.K ≤ p.BK`

**Closed-form spec defs (transitive):** `Storage`, `intraOriginal`, `intraFormula`, `original`, `scaleFirst`, `columns`, `outputAddress`, `enabled`, `rowActive`, `matrixAddress`, `keyAddress`, `source`, `loadFirst`, `rowStart`, `featureIndex`, `sumLeft`

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

<details><summary><code>intraOriginal</code></summary>

```
/-- Bounds cover cached Q/G, each streamed key/gate load and row-masked store.
The column count is at most BT. Key/gate must differ from output; cached Q may alias. -/
```
```lean
def intraOriginal (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) : IO Launch Inputs (Nat × Fin p.BC) where
  kernel := (TypedIntra.original qs ks gs r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK).eraseDType
  projection := RealIntraIO.projection p r qs ks gs scale Bool.false
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun s => columns p (launch s) ≤ p.BT
  bounds := fun s => RealIntraSafety.Bounds p (launch s) r
  out := r.output
  write := fun l i => outputAddress p l i.2.val i.1
  mask := fun l i => enabled p l ∧ i.1 < columns p l ∧ rowActive p l i.2.val
```
</details>

<details><summary><code>intraFormula</code></summary>

```
/-- Causal query/key contraction on the initial arrays; matrix rows use
`timeStride`, streamed key/gate rows use `K`. -/
```
```lean
def intraFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : Nat × Fin p.BC) : ℝ :=
  if i.1 ≤ i.2.val then (∑ k ∈ Finset.range p.K,
      a.query (matrixAddress p l i.2.val k) * a.key (keyAddress p l i.1 k) *
        Real.exp (a.gate (matrixAddress p l i.2.val k)-a.gate (keyAddress p l i.1 k)))*scale else 0
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
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

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (l : Launch) := min p.BC (p.T-l.chunk*p.BT-l.block*p.BC)
```
</details>

<details><summary><code>outputAddress</code></summary>

```lean
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
```
</details>

<details><summary><code>enabled</code></summary>

```lean
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
```
</details>

<details><summary><code>rowActive</code></summary>

```lean
def rowActive (p : Layout) (l : Launch) (i : Nat) : Prop := rowStart p l+i < p.T
```
</details>

<details><summary><code>matrixAddress</code></summary>

```lean
def matrixAddress (p : Layout) (l : Launch) (i k : Nat) :=
  l.head*p.headStride+(rowStart p l+i)*p.timeStride+featureIndex p l k
```
</details>

<details><summary><code>keyAddress</code></summary>

```lean
def keyAddress (p : Layout) (l : Launch) (j k : Nat) :=
  l.head*p.headStride+(rowStart p l+j)*p.K+featureIndex p l k
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

<details><summary><code>rowStart</code></summary>

```lean
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
```
</details>

<details><summary><code>featureIndex</code></summary>

```lean
def featureIndex (p : Layout) (l : Launch) (e : Nat) := l.feature*p.BK+e
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

## Public theorem: `intra_scaleFirst_correct`

**Statement:**
```lean
specification intra_scaleFirst_correct (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) (hK : p.K ≤ p.BK) :
    intraScaleFirst p r qs ks gs scale ⊨ intraFormula p scale
```

**Assumptions / layout contracts:**
- `hkey : r.key ≠ r.output`
- `hgate : r.gate ≠ r.output`
- `hK : p.K ≤ p.BK`

**Closed-form spec defs (transitive):** `Storage`, `intraScaleFirst`, `intraFormula`, `original`, `scaleFirst`, `intraOriginal`, `matrixAddress`, `keyAddress`, `source`, `loadFirst`, `columns`, `outputAddress`, `enabled`, `rowActive`, `rowStart`, `featureIndex`, `sumLeft`

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

<details><summary><code>intraScaleFirst</code></summary>

```lean
def intraScaleFirst (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) : IO Launch Inputs (Nat × Fin p.BC) :=
  { intraOriginal p r qs ks gs scale with
    kernel := (TypedIntra.scaleFirst qs ks gs r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK).eraseDType
    projection := RealIntraIO.projection p r qs ks gs scale Bool.true }
```
</details>

<details><summary><code>intraFormula</code></summary>

```
/-- Causal query/key contraction on the initial arrays; matrix rows use
`timeStride`, streamed key/gate rows use `K`. -/
```
```lean
def intraFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : Nat × Fin p.BC) : ℝ :=
  if i.1 ≤ i.2.val then (∑ k ∈ Finset.range p.K,
      a.query (matrixAddress p l i.2.val k) * a.key (keyAddress p l i.1 k) *
        Real.exp (a.gate (matrixAddress p l i.2.val k)-a.gate (keyAddress p l i.1 k)))*scale else 0
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
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

<details><summary><code>intraOriginal</code></summary>

```
/-- Bounds cover cached Q/G, each streamed key/gate load and row-masked store.
The column count is at most BT. Key/gate must differ from output; cached Q may alias. -/
```
```lean
def intraOriginal (p : Layout) (r : Regions) (qs ks gs : TypedIntra.Storage)
    (scale : ℝ) : IO Launch Inputs (Nat × Fin p.BC) where
  kernel := (TypedIntra.original qs ks gs r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK).eraseDType
  projection := RealIntraIO.projection p r qs ks gs scale Bool.false
  grid := launch
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun s => columns p (launch s) ≤ p.BT
  bounds := fun s => RealIntraSafety.Bounds p (launch s) r
  out := r.output
  write := fun l i => outputAddress p l i.2.val i.1
  mask := fun l i => enabled p l ∧ i.1 < columns p l ∧ rowActive p l i.2.val
```
</details>

<details><summary><code>matrixAddress</code></summary>

```lean
def matrixAddress (p : Layout) (l : Launch) (i k : Nat) :=
  l.head*p.headStride+(rowStart p l+i)*p.timeStride+featureIndex p l k
```
</details>

<details><summary><code>keyAddress</code></summary>

```lean
def keyAddress (p : Layout) (l : Launch) (j k : Nat) :=
  l.head*p.headStride+(rowStart p l+j)*p.K+featureIndex p l k
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

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (l : Launch) := min p.BC (p.T-l.chunk*p.BT-l.block*p.BC)
```
</details>

<details><summary><code>outputAddress</code></summary>

```lean
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
```
</details>

<details><summary><code>enabled</code></summary>

```lean
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
```
</details>

<details><summary><code>rowActive</code></summary>

```lean
def rowActive (p : Layout) (l : Launch) (i : Nat) : Prop := rowStart p l+i < p.T
```
</details>

<details><summary><code>rowStart</code></summary>

```lean
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
```
</details>

<details><summary><code>featureIndex</code></summary>

```lean
def featureIndex (p : Layout) (l : Launch) (e : Nat) := l.feature*p.BK+e
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

## Public theorem: `split_correct`

**Statement:**
```lean
specification split_correct (p : Layout) (r : Regions) (qs ks gs : TypedSplit.Storage)
    (scale : ℝ)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) :
    splitOriginal p r qs ks gs scale ⊨ splitFormula p scale
```

**Assumptions / layout contracts:**
- `hkey : r.key ≠ r.output`
- `hgate : r.gate ≠ r.output`

**Closed-form spec defs (transitive):** `Storage`, `splitOriginal`, `splitFormula`, `original`, `scaleFirst`, `outputAddress`, `enabled`, `columns`, `rowActive`, `featureIndex`, `matrixAddress`, `keyAddress`, `source`, `loadFirst`, `rowStart`, `sumLeft`

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

<details><summary><code>splitOriginal</code></summary>

```
/-- Bounds cover cached Q/G, streamed key/gate and the T-strided temporary
store. Key/gate must differ from output; cached Q may alias. K may exceed BK. -/
```
```lean
def splitOriginal (p : Layout) (r : Regions) (qs ks gs : TypedSplit.Storage)
    (scale : ℝ) : IO Launch Inputs (Nat × Fin p.BC) where
  kernel := (TypedSplit.original qs ks gs r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC).eraseDType
  projection := RealSplitIO.projection p r qs ks gs scale Bool.false
  grid := launch p
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun _ => True
  bounds := fun s => RealSplitSafety.Bounds p (launch p s) r
  out := r.output
  write := fun l i => outputAddress p l i.2.val i.1
  mask := fun l i => enabled p l ∧ i.1 < columns p l ∧ rowActive p l i.2.val
```
</details>

<details><summary><code>splitFormula</code></summary>

```
/-- Causal query/key contraction on the initial arrays; matrix rows use
`timeStride`, streamed key/gate rows use `K`. -/
```
```lean
def splitFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : Nat × Fin p.BC) : ℝ :=
  if i.1 ≤ i.2.val then (∑ e : Fin p.BK, if featureIndex p l e.val < p.K then
      a.query (matrixAddress p l i.2.val e.val) * a.key (keyAddress p l i.1 e.val) *
        Real.exp (a.gate (matrixAddress p l i.2.val e.val)-a.gate (keyAddress p l i.1 e.val)) else 0)*scale else 0
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
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

<details><summary><code>outputAddress</code></summary>

```lean
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
```
</details>

<details><summary><code>enabled</code></summary>

```lean
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
```
</details>

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (l : Launch) := min p.BC (p.T-l.chunk*p.BT-l.block*p.BC)
```
</details>

<details><summary><code>rowActive</code></summary>

```lean
def rowActive (p : Layout) (l : Launch) (i : Nat) : Prop := rowStart p l+i < p.T
```
</details>

<details><summary><code>featureIndex</code></summary>

```lean
def featureIndex (p : Layout) (l : Launch) (e : Nat) := l.feature*p.BK+e
```
</details>

<details><summary><code>matrixAddress</code></summary>

```lean
def matrixAddress (p : Layout) (l : Launch) (i k : Nat) :=
  l.head*p.headStride+(rowStart p l+i)*p.timeStride+featureIndex p l k
```
</details>

<details><summary><code>keyAddress</code></summary>

```lean
def keyAddress (p : Layout) (l : Launch) (j k : Nat) :=
  l.head*p.headStride+(rowStart p l+j)*p.K+featureIndex p l k
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

<details><summary><code>rowStart</code></summary>

```lean
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

## Public theorem: `split_scaleFirst_correct`

**Statement:**
```lean
specification split_scaleFirst_correct (p : Layout) (r : Regions) (qs ks gs : TypedSplit.Storage)
    (scale : ℝ)
    (hkey : r.key ≠ r.output) (hgate : r.gate ≠ r.output) :
    splitScaleFirst p r qs ks gs scale ⊨ splitFormula p scale
```

**Assumptions / layout contracts:**
- `hkey : r.key ≠ r.output`
- `hgate : r.gate ≠ r.output`

**Closed-form spec defs (transitive):** `Storage`, `splitScaleFirst`, `splitFormula`, `original`, `scaleFirst`, `splitOriginal`, `featureIndex`, `matrixAddress`, `keyAddress`, `source`, `loadFirst`, `outputAddress`, `enabled`, `columns`, `rowActive`, `rowStart`, `sumLeft`

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

<details><summary><code>splitScaleFirst</code></summary>

```lean
def splitScaleFirst (p : Layout) (r : Regions) (qs ks gs : TypedSplit.Storage)
    (scale : ℝ) : IO Launch Inputs (Nat × Fin p.BC) :=
  { splitOriginal p r qs ks gs scale with
    kernel := (TypedSplit.scaleFirst qs ks gs r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC).eraseDType
    projection := RealSplitIO.projection p r qs ks gs scale Bool.true }
```
</details>

<details><summary><code>splitFormula</code></summary>

```
/-- Causal query/key contraction on the initial arrays; matrix rows use
`timeStride`, streamed key/gate rows use `K`. -/
```
```lean
def splitFormula (p : Layout) (scale : ℝ) (l : Launch) (a : Inputs) (i : Nat × Fin p.BC) : ℝ :=
  if i.1 ≤ i.2.val then (∑ e : Fin p.BK, if featureIndex p l e.val < p.K then
      a.query (matrixAddress p l i.2.val e.val) * a.key (keyAddress p l i.1 e.val) *
        Real.exp (a.gate (matrixAddress p l i.2.val e.val)-a.gate (keyAddress p l i.1 e.val)) else 0)*scale else 0
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
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

<details><summary><code>splitOriginal</code></summary>

```
/-- Bounds cover cached Q/G, streamed key/gate and the T-strided temporary
store. Key/gate must differ from output; cached Q may alias. K may exceed BK. -/
```
```lean
def splitOriginal (p : Layout) (r : Regions) (qs ks gs : TypedSplit.Storage)
    (scale : ℝ) : IO Launch Inputs (Nat × Fin p.BC) where
  kernel := (TypedSplit.original qs ks gs r.query r.key r.gate r.output
    p.headStride p.timeStride scale p.T p.K p.BT p.BC p.BK p.NC).eraseDType
  projection := RealSplitIO.projection p r qs ks gs scale Bool.false
  grid := launch p
  read := fun s => ⟨s.readMem r.query,s.readMem r.key,s.readMem r.gate⟩
  valid := fun _ => True
  bounds := fun s => RealSplitSafety.Bounds p (launch p s) r
  out := r.output
  write := fun l i => outputAddress p l i.2.val i.1
  mask := fun l i => enabled p l ∧ i.1 < columns p l ∧ rowActive p l i.2.val
```
</details>

<details><summary><code>featureIndex</code></summary>

```lean
def featureIndex (p : Layout) (l : Launch) (e : Nat) := l.feature*p.BK+e
```
</details>

<details><summary><code>matrixAddress</code></summary>

```lean
def matrixAddress (p : Layout) (l : Launch) (i k : Nat) :=
  l.head*p.headStride+(rowStart p l+i)*p.timeStride+featureIndex p l k
```
</details>

<details><summary><code>keyAddress</code></summary>

```lean
def keyAddress (p : Layout) (l : Launch) (j k : Nat) :=
  l.head*p.headStride+(rowStart p l+j)*p.K+featureIndex p l k
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

<details><summary><code>outputAddress</code></summary>

```lean
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
```
</details>

<details><summary><code>enabled</code></summary>

```lean
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
```
</details>

<details><summary><code>columns</code></summary>

```lean
def columns (p : Layout) (l : Launch) := min p.BC (p.T-l.chunk*p.BT-l.block*p.BC)
```
</details>

<details><summary><code>rowActive</code></summary>

```lean
def rowActive (p : Layout) (l : Launch) (i : Nat) : Prop := rowStart p l+i < p.T
```
</details>

<details><summary><code>rowStart</code></summary>

```lean
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

## Public theorem: `merge_correct`

**Statement:**
```lean
specification merge_correct (p : Layout) (A A2 : RegionName) (out : FloatDType) :
    mergeOriginal p A A2 out ⊨ mergeFormula p
```

**Closed-form spec defs (transitive):** `mergeOriginal`, `mergeFormula`, `original`, `outputAddress`, `active`, `inputAddress`, `source`, `rowStart`, `enabled`, `sumLeft`

<details><summary><code>mergeOriginal</code></summary>

```
/-- Bounds cover all NK temporary slices and the final masked store;
NoCollision requires distinct active output addresses. The two regions may alias. -/
```
```lean
def mergeOriginal (p : Layout) (A A2 : RegionName) (out : FloatDType) :
    IO Launch (Nat → ℝ) (TileIndex [p.BC,p.BC]) where
  kernel := (TypedMerge.original out A A2 p.T p.BT p.BC p.NK).eraseDType
  projection := RealMergeIO.projection p A A2 out Bool.false
  grid := launch
  read := fun s => s.readMem A
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => RealMergeSafety.Bounds p (launch s) A A2
  out := A2
  write := outputAddress p
  mask := active p
```
</details>

<details><summary><code>mergeFormula</code></summary>

```lean
def mergeFormula (p : Layout) (l : Launch) (a : Nat → ℝ) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  ∑ kb ∈ Finset.range p.NK, a (inputAddress p l kb i)
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
```
</details>

<details><summary><code>outputAddress</code></summary>

```lean
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Prop :=
  rowStart p l+i.1.val < p.T ∧ l.subblock*p.BC+i.2.1.val < p.BT
instance (p : Layout) (l : Launch) : Decidable (enabled p l) := inferInstanceAs (Decidable (_ < _))
instance (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Decidable (active p l i) :=
  inferInstanceAs (Decidable (_ ∧ _))
```
</details>

<details><summary><code>inputAddress</code></summary>

```lean
def inputAddress (p : Layout) (l : Launch) (kb : Nat) (i : TileIndex [p.BC,p.BC]) :=
  (l.head+kb*l.heads)*p.T*p.BC+(rowStart p l+i.1.val)*p.BC+i.2.1.val
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

<details><summary><code>rowStart</code></summary>

```lean
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
```
</details>

<details><summary><code>enabled</code></summary>

```lean
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

## Public theorem: `merge_loadFirst_correct`

**Statement:**
```lean
specification merge_loadFirst_correct (p : Layout) (A A2 : RegionName) (out : FloatDType) :
    mergeLoadFirst p A A2 out ⊨ mergeFormula p
```

**Closed-form spec defs (transitive):** `mergeLoadFirst`, `mergeFormula`, `mergeOriginal`, `loadFirst`, `inputAddress`, `original`, `outputAddress`, `active`, `sumLeft`, `rowStart`, `source`, `enabled`

<details><summary><code>mergeLoadFirst</code></summary>

```lean
def mergeLoadFirst (p : Layout) (A A2 : RegionName) (out : FloatDType) : IO Launch (Nat → ℝ) (TileIndex [p.BC,p.BC]) :=
  { mergeOriginal p A A2 out with
    kernel := (TypedMerge.loadFirst out A A2 p.T p.BT p.BC p.NK).eraseDType
    projection := RealMergeIO.projection p A A2 out Bool.true }
```
</details>

<details><summary><code>mergeFormula</code></summary>

```lean
def mergeFormula (p : Layout) (l : Launch) (a : Nat → ℝ) (i : TileIndex [p.BC,p.BC]) : ℝ :=
  ∑ kb ∈ Finset.range p.NK, a (inputAddress p l kb i)
```
</details>

<details><summary><code>mergeOriginal</code></summary>

```
/-- Bounds cover all NK temporary slices and the final masked store;
NoCollision requires distinct active output addresses. The two regions may alias. -/
```
```lean
def mergeOriginal (p : Layout) (A A2 : RegionName) (out : FloatDType) :
    IO Launch (Nat → ℝ) (TileIndex [p.BC,p.BC]) where
  kernel := (TypedMerge.original out A A2 p.T p.BT p.BC p.NK).eraseDType
  projection := RealMergeIO.projection p A A2 out Bool.false
  grid := launch
  read := fun s => s.readMem A
  valid := fun s => NoCollision p (launch s)
  bounds := fun s => RealMergeSafety.Bounds p (launch s) A A2
  out := A2
  write := outputAddress p
  mask := active p
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

<details><summary><code>inputAddress</code></summary>

```lean
def inputAddress (p : Layout) (l : Launch) (kb : Nat) (i : TileIndex [p.BC,p.BC]) :=
  (l.head+kb*l.heads)*p.T*p.BC+(rowStart p l+i.1.val)*p.BC+i.2.1.val
```
</details>

<details><summary><code>original</code></summary>

```lean
def original (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.false
```
</details>

<details><summary><code>outputAddress</code></summary>

```lean
def outputAddress (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) :=
  l.head*p.T*p.BT+(rowStart p l+i.1.val)*p.BT+(l.subblock*p.BC+i.2.1.val)
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Prop :=
  rowStart p l+i.1.val < p.T ∧ l.subblock*p.BC+i.2.1.val < p.BT
instance (p : Layout) (l : Launch) : Decidable (enabled p l) := inferInstanceAs (Decidable (_ < _))
instance (p : Layout) (l : Launch) (i : TileIndex [p.BC,p.BC]) : Decidable (active p l i) :=
  inferInstanceAs (Decidable (_ ∧ _))
```
</details>

<details><summary><code>sumLeft</code></summary>

```lean
def sumLeft (queryStorage valueStorage gateStorage : Storage) :=
  source queryStorage valueStorage gateStorage Bool.true
```
</details>

<details><summary><code>rowStart</code></summary>

```lean
def rowStart (p : Layout) (l : Launch) := l.chunk*p.BT+l.subblock*p.BC
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

<details><summary><code>enabled</code></summary>

```lean
def enabled (p : Layout) (l : Launch) : Prop := rowStart p l < p.T
```
</details>
