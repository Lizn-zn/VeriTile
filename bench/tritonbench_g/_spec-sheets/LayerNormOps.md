# Spec sheet — `bench/tritonbench_g/layer_norm_ops/LayerNormOps.lean`

**Python source:** `bench/tritonbench_g/layer_norm_ops/layer_norm_ops.py`

## Public theorem: `layer_norm_ops_fwd_plain_bias_all_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- Plain layer-norm forward, **dimension-general**: exposes the genuine `Y`,
`Mean`, and `Rstd` outputs together for arbitrary feature dim `N`, tile width
`BLOCK_N`, and `Y` row stride `stride_y_row`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_fwd_plain_bias_all_outputs_compute_correct_general
    (ValuePre MeanPre RstdPre Y Mean Rstd : RegionName) (s : BlockState)
    (stride_y_row N BLOCK_N : Nat) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdYStoreSpec s ValuePre stride_y_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_mean_store_slice MeanPre Mean)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, meanRowOffset s))
      (expected := fun _ => meanStoreSpec s MeanPre)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre))
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_fwd_y_store_slice`, `fwdYOffset`, `fwdYStoreSpec`, `layer_norm_ops_fwd_mean_store_slice`, `meanRowOffset`, `meanStoreSpec`, `layer_norm_ops_fwd_rstd_store_slice`, `rstdStoreSpec`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorOffset`, `bwdRowVectorStoreSpec`

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_mean_store_slice</code></summary>

```
/-- Proof-oriented Mean store slice of `layer_norm_ops.py`'s
`_layer_norm_fwd_1pass_kernel`. Takes a precomputed `MeanPre` scalar (per row)
and proves the unmasked scalar writeback into `Mean` at row offset. -/
```
```lean
def layer_norm_ops_fwd_mean_store_slice
    (MeanPre Mean : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  mean = tl.load(MeanPre + row)
  tl.store(Mean + row, mean)
}
```
</details>

<details><summary><code>meanRowOffset</code></summary>

```lean
def meanRowOffset (s : BlockState) : Nat := s.pid
```
</details>

<details><summary><code>meanStoreSpec</code></summary>

```lean
noncomputable def meanStoreSpec (s : BlockState) (MeanPre : RegionName) : ℝ :=
  s.readMem MeanPre s.pid
```
</details>

<details><summary><code>layer_norm_ops_fwd_rstd_store_slice</code></summary>

```
/-- Proof-oriented Rstd store slice. Same scalar-copy pattern. -/
```
```lean
def layer_norm_ops_fwd_rstd_store_slice
    (RstdPre Rstd : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  rstd = tl.load(RstdPre + row)
  tl.store(Rstd + row, rstd)
}
```
</details>

<details><summary><code>rstdStoreSpec</code></summary>

```lean
noncomputable def rstdStoreSpec (s : BlockState) (RstdPre : RegionName) : ℝ :=
  s.readMem RstdPre s.pid
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_fwd_rms_bias_all_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- RMS layer-norm forward with bias, **dimension-general**: exposes the genuine
`Y` and `Rstd` outputs together. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_fwd_rms_bias_all_outputs_compute_correct_general
    (ValuePre RstdPre Y Rstd : RegionName) (s : BlockState)
    (stride_y_row N BLOCK_N : Nat) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdYStoreSpec s ValuePre stride_y_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre))
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_fwd_y_store_slice`, `fwdYOffset`, `fwdYStoreSpec`, `layer_norm_ops_fwd_rstd_store_slice`, `meanRowOffset`, `rstdStoreSpec`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorOffset`, `bwdRowVectorStoreSpec`

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_rstd_store_slice</code></summary>

```
/-- Proof-oriented Rstd store slice. Same scalar-copy pattern. -/
```
```lean
def layer_norm_ops_fwd_rstd_store_slice
    (RstdPre Rstd : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  rstd = tl.load(RstdPre + row)
  tl.store(Rstd + row, rstd)
}
```
</details>

<details><summary><code>meanRowOffset</code></summary>

```lean
def meanRowOffset (s : BlockState) : Nat := s.pid
```
</details>

<details><summary><code>rstdStoreSpec</code></summary>

```lean
noncomputable def rstdStoreSpec (s : BlockState) (RstdPre : RegionName) : ℝ :=
  s.readMem RstdPre s.pid
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_fwd_residual_bias_all_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- Residual plain layer-norm forward with bias, **dimension-general**: exposes
the genuine `RESIDUAL_OUT`, `Y`, `Mean`, and `Rstd` outputs together for
arbitrary `RESIDUAL_OUT`/`Y` row strides, feature dim `N`, and tile width
`BLOCK_N`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_fwd_residual_bias_all_outputs_compute_correct_general
    (ResidualPre ValuePre MeanPre RstdPre RESIDUAL_OUT Y Mean Rstd : RegionName)
    (s : BlockState)
    (stride_res_out_row stride_y_row N BLOCK_N : Nat) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_residual_out_store_slice ResidualPre
        RESIDUAL_OUT stride_res_out_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (RESIDUAL_OUT, fwdResidualOutOffset s stride_res_out_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdResidualOutStoreSpec s ResidualPre stride_res_out_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_y_store_slice ValuePre Y
        stride_y_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (Y, fwdYOffset s stride_y_row i)))
      (expected := fun i : Fin BLOCK_N =>
        fwdYStoreSpec s ValuePre stride_y_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_mean_store_slice MeanPre Mean)
      (initialState := s)
      (write := fun _ : PUnit => some (Mean, meanRowOffset s))
      (expected := fun _ => meanStoreSpec s MeanPre)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_fwd_rstd_store_slice RstdPre Rstd)
      (initialState := s)
      (write := fun _ : PUnit => some (Rstd, meanRowOffset s))
      (expected := fun _ => rstdStoreSpec s RstdPre))
```

**Assumptions / layout contracts:**
- `fun i : Fin BLOCK_N => i.val < N`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_fwd_residual_out_store_slice`, `fwdResidualOutOffset`, `fwdResidualOutStoreSpec`, `layer_norm_ops_fwd_y_store_slice`, `fwdYOffset`, `fwdYStoreSpec`, `layer_norm_ops_fwd_mean_store_slice`, `meanRowOffset`, `meanStoreSpec`, `layer_norm_ops_fwd_rstd_store_slice`, `rstdStoreSpec`, `layer_norm_ops_bwd_row_vector_store_slice`, `bwdRowVectorOffset`, `bwdRowVectorStoreSpec`

<details><summary><code>layer_norm_ops_fwd_residual_out_store_slice</code></summary>

```
/-- Forward residual-output writeback slice for `_layer_norm_fwd_1pass_kernel`.

When `STORE_RESIDUAL_OUT` is true, Python stores the row tile `x` after the
optional residual add into `RESIDUAL_OUT + cols`. The caller supplies that
precomputed row tile in `ValuePre`; this theorem proves the same masked
row-vector writeback shape as the backward row-vector stores. -/
```
```lean
abbrev layer_norm_ops_fwd_residual_out_store_slice
    (ValuePre RESIDUAL_OUT : RegionName)
    (stride_res_out_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre RESIDUAL_OUT
    stride_res_out_row N BLOCK_N
```
</details>

<details><summary><code>fwdResidualOutOffset</code></summary>

```lean
def fwdResidualOutOffset (s : BlockState) (stride_res_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_res_out_row i
```
</details>

<details><summary><code>fwdResidualOutStoreSpec</code></summary>

```lean
noncomputable def fwdResidualOutStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_res_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_res_out_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_y_store_slice</code></summary>

```
/-- Forward output writeback slice for `_layer_norm_fwd_1pass_kernel`.

After the mean/rstd arithmetic and optional bias/residual branches, Python
stores the computed row tile `y` into `Y + cols`. This aliases the generic
row-vector writeback proof with the forward output stride. -/
```
```lean
abbrev layer_norm_ops_fwd_y_store_slice
    (ValuePre Y : RegionName) (stride_y_row N BLOCK_N : Nat) : ComputeKernel :=
  layer_norm_ops_bwd_row_vector_store_slice ValuePre Y stride_y_row N BLOCK_N
```
</details>

<details><summary><code>fwdYOffset</code></summary>

```lean
def fwdYOffset (s : BlockState) (stride_y_row : Nat) (i : Fin BLOCK_N) : Nat :=
  bwdRowVectorOffset s stride_y_row i
```
</details>

<details><summary><code>fwdYStoreSpec</code></summary>

```lean
noncomputable def fwdYStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_y_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  bwdRowVectorStoreSpec s ValuePre stride_y_row i
```
</details>

<details><summary><code>layer_norm_ops_fwd_mean_store_slice</code></summary>

```
/-- Proof-oriented Mean store slice of `layer_norm_ops.py`'s
`_layer_norm_fwd_1pass_kernel`. Takes a precomputed `MeanPre` scalar (per row)
and proves the unmasked scalar writeback into `Mean` at row offset. -/
```
```lean
def layer_norm_ops_fwd_mean_store_slice
    (MeanPre Mean : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  mean = tl.load(MeanPre + row)
  tl.store(Mean + row, mean)
}
```
</details>

<details><summary><code>meanRowOffset</code></summary>

```lean
def meanRowOffset (s : BlockState) : Nat := s.pid
```
</details>

<details><summary><code>meanStoreSpec</code></summary>

```lean
noncomputable def meanStoreSpec (s : BlockState) (MeanPre : RegionName) : ℝ :=
  s.readMem MeanPre s.pid
```
</details>

<details><summary><code>layer_norm_ops_fwd_rstd_store_slice</code></summary>

```
/-- Proof-oriented Rstd store slice. Same scalar-copy pattern. -/
```
```lean
def layer_norm_ops_fwd_rstd_store_slice
    (RstdPre Rstd : RegionName) : ComputeKernel := triton {
  row = tl.program_id(0)
  rstd = tl.load(RstdPre + row)
  tl.store(Rstd + row, rstd)
}
```
</details>

<details><summary><code>rstdStoreSpec</code></summary>

```lean
noncomputable def rstdStoreSpec (s : BlockState) (RstdPre : RegionName) : ℝ :=
  s.readMem RstdPre s.pid
```
</details>

<details><summary><code>layer_norm_ops_bwd_row_vector_store_slice</code></summary>

```
/-- Backward row-vector writeback slice for `DX`, optional `DRESIDUAL_IN`, and
optional recomputed `Y`. The caller supplies the precomputed row tile in
`ValuePre`; the slice proves the masked store to `Out + row * stride + cols`. -/
```
```lean
def layer_norm_ops_bwd_row_vector_store_slice
    (ValuePre Out : RegionName) (stride_out_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  value = tl.load(ValuePre + row * $(stride_out_row) + cols,
    mask=mask, other=0.0)
  tl.store(Out + row * $(stride_out_row) + cols, value, mask=mask)
}
```
</details>

<details><summary><code>bwdRowVectorOffset</code></summary>

```lean
def bwdRowVectorOffset (s : BlockState) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_out_row + i.val
```
</details>

<details><summary><code>bwdRowVectorStoreSpec</code></summary>

```lean
noncomputable def bwdRowVectorStoreSpec
    (s : BlockState) (ValuePre : RegionName) (stride_out_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem ValuePre (bwdRowVectorOffset s stride_out_row i)
```
</details>

## Public theorem: `layer_norm_ops_bwd_rms_core_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- RMS backward, **dimension-general**: exposes the genuine `c1` reduction,
`DX`, and partial `DW` store slices together for arbitrary row strides, feature
dim `N`, and tile width `BLOCK_N`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_bwd_rms_core_outputs_compute_correct_general
    (X Xhat W DY Rstd C1 DX DW : RegionName) (s : BlockState)
    (stride_xhat_row stride_dy_row stride_x_row stride_dx_row N BLOCK_N : Nat)
    (hDWDX : DW ≠ DX) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_c1_reduction_slice Xhat W DY C1
        stride_xhat_row stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C1, s.pid))
      (expected := fun _ : PUnit =>
        bwdC1ReductionSpec s Xhat W DY stride_xhat_row stride_dy_row N BLOCK_N)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_rms_dx_from_c1_slice Xhat W DY Rstd C1 DX
        stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdRmsDXFromC1Spec s Xhat W DY Rstd C1 stride_xhat_row stride_dy_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_rms_one_row X W DY DX DW Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdRmsDWOffset s N i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdRmsDWSpec s X DY Rstd stride_x_row stride_dy_row N BLOCK_N i))
```

**Assumptions / layout contracts:**
- `hDWDX : DW ≠ DX`
- `fun i : Fin BLOCK_N => i.val < N`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_bwd_c1_reduction_slice`, `bwdC1ReductionSpec`, `layer_norm_ops_bwd_rms_dx_from_c1_slice`, `bwdRmsDXOffset`, `bwdRmsDXFromC1Spec`, `layer_norm_ops_bwd_rms_one_row`, `bwdRmsDWOffset`, `bwdRmsDWSpec`, `bwdRecomputeXhatOffset`, `bwdRmsWdyTile`, `bwdRmsDYOffset`, `bwdRmsDYTile`, `bwdRmsXhatTile`, `bwdRmsWTile`, `bwdRmsXTile`, `bwdRmsXOffset`

<details><summary><code>layer_norm_ops_bwd_c1_reduction_slice</code></summary>

```
/-! ## Backward `c1`/`c2` reduction-production slices

The full backward kernel computes `c1 = sum(xhat * (W * DY)) / N` in both RMS
and non-RMS branches, and additionally computes `c2 = sum(W * DY) / N` in the
non-RMS branch. These slices isolate the two row-local reductions and their
scalar stores so the reduction production itself is checked independently of
the surrounding branch/control-flow proof. -/
```
```lean
def layer_norm_ops_bwd_c1_reduction_slice
    (Xhat W DY C1 : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  tl.store(C1 + row, c1)
}
```
</details>

<details><summary><code>bwdC1ReductionSpec</code></summary>

```lean
noncomputable def bwdC1ReductionSpec
    (s : BlockState) (Xhat W DY : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun a => a / (N : ℝ))
      ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
        (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
          { data := fun idx =>
              if idx.1.val < N then
                some (s.readMem Xhat
                  (bwdRecomputeXhatOffset s stride_xhat_row idx.1))
              else some (0.0 : ℝ) }
          (bwdRmsWdyTile s W DY stride_dy_row N BLOCK_N))).data PUnit.unit))
```
</details>

<details><summary><code>layer_norm_ops_bwd_rms_dx_from_c1_slice</code></summary>

```
/-! ## Backward base-DX formula slice

After the RMS branch has produced `xhat` and reduced `c1`, Python computes
`dx = (w * dy - xhat * c1) * rstd`. This slice proves that formula and the
masked `DX` store directly, leaving only the production of `c1` itself outside
the theorem. -/
```
```lean
def layer_norm_ops_bwd_rms_dx_from_c1_slice
    (Xhat W DY Rstd C1 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  c1 = tl.load(C1 + row)
  wdy = w * dy
  dx = (wdy - xhat * c1) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
}
```
</details>

<details><summary><code>bwdRmsDXOffset</code></summary>

```lean
def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val
```
</details>

<details><summary><code>bwdRmsDXFromC1Spec</code></summary>

```lean
noncomputable def bwdRmsDXFromC1Spec
    (s : BlockState) (Xhat W DY Rstd C1 : RegionName)
    (stride_xhat_row stride_dy_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  (s.readMem W i.val *
      s.readMem DY (bwdRmsDYOffset s stride_dy_row i) -
    s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
      s.readMem C1 s.pid) *
    s.readMem Rstd s.pid
```
</details>

<details><summary><code>layer_norm_ops_bwd_rms_one_row</code></summary>

```
/-! ## Backward RMS arithmetic slice

This slice removes one of the narrow-writeback gaps above: it computes the
tested RMS backward arithmetic for one row, not merely a precomputed `DX`/`DW`
copy. It specializes the full backward kernel to the no-bias/no-residual/
no-recompute RMS branch and one row per program. -/
```
```lean
def layer_norm_ops_bwd_rms_one_row
    (X W DY DX DW Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0.0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  xhat = x * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw = dy * xhat
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  dx = (wdy - xhat * c1) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
  tl.store(DW + row * $(N) + cols, dw, mask=mask)
}
```
</details>

<details><summary><code>bwdRmsDWOffset</code></summary>

```lean
def bwdRmsDWOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

<details><summary><code>bwdRmsDWSpec</code></summary>

```lean
noncomputable def bwdRmsDWSpec
    (s : BlockState) (X DY Rstd : RegionName)
    (stride_x_row stride_dy_row N BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun dy xhat => dy * xhat)
      ((bwdRmsDYTile s DY stride_dy_row N BLOCK_N).data (i, PUnit.unit))
      ((bwdRmsXhatTile s X Rstd stride_x_row N BLOCK_N).data (i, PUnit.unit)))
```
</details>

<details><summary><code>bwdRecomputeXhatOffset</code></summary>

```lean
def bwdRecomputeXhatOffset (s : BlockState) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_xhat_row + i.val
```
</details>

<details><summary><code>bwdRmsWdyTile</code></summary>

```lean
noncomputable def bwdRmsWdyTile
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
    (bwdRmsWTile s W N BLOCK_N)
    (bwdRmsDYTile s DY stride_dy_row N BLOCK_N)
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

<details><summary><code>bwdRmsDYTile</code></summary>

```lean
noncomputable def bwdRmsDYTile
    (s : BlockState) (DY : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem DY (bwdRmsDYOffset s stride_dy_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsXhatTile</code></summary>

```lean
noncomputable def bwdRmsXhatTile
    (s : BlockState) (X Rstd : RegionName)
    (stride_x_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        match (bwdRmsXTile s X stride_x_row N BLOCK_N).data idx with
        | some x => some (x * s.readMem Rstd s.pid)
        | none => none
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsWTile</code></summary>

```lean
noncomputable def bwdRmsWTile
    (s : BlockState) (W : RegionName) (N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem W idx.1.val)
      else some (s.undef W idx.1.val) }
```
</details>

<details><summary><code>bwdRmsXTile</code></summary>

```lean
noncomputable def bwdRmsXTile
    (s : BlockState) (X : RegionName) (stride_x_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (bwdRmsXOffset s stride_x_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsXOffset</code></summary>

```lean
def bwdRmsXOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_plain_bias_core_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- Plain+bias backward, **dimension-general**: exposes the genuine `c1`/`c2`
reductions, `DX`, partial `DW`, and partial `DB` store slices together for
arbitrary row strides, feature dim `N`, and tile width `BLOCK_N`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_bwd_plain_bias_core_outputs_compute_correct_general
    (X Xhat W DY DX DW DB Mean Rstd C1 C2 : RegionName) (s : BlockState)
    (stride_xhat_row stride_dy_row stride_x_row stride_dx_row N BLOCK_N : Nat)
    (hDWDX : DW ≠ DX) (hDWDB : DW ≠ DB)
    (hDBDX : DB ≠ DX) (hDBDW : DB ≠ DW) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_c1_reduction_slice Xhat W DY C1
        stride_xhat_row stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C1, s.pid))
      (expected := fun _ : PUnit =>
        bwdC1ReductionSpec s Xhat W DY stride_xhat_row stride_dy_row N BLOCK_N)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_c2_reduction_slice W DY C2
        stride_dy_row N BLOCK_N)
      (initialState := s)
      (write := fun _ : PUnit => some (C2, s.pid))
      (expected := fun _ : PUnit =>
        bwdC2ReductionSpec s W DY stride_dy_row N BLOCK_N)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_plain_dx_from_c1_c2_slice Xhat W DY Rstd
        C1 C2 DX stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdPlainDXFromC1C2Spec s Xhat W DY Rstd C1 C2 stride_xhat_row
          stride_dy_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB Mean Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DW, bwdParamGradOffset s N i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdPlainBiasDWSpec s X DY Mean Rstd stride_x_row stride_dy_row N
          BLOCK_N i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_plain_bias_one_row X W DY DX DW DB Mean Rstd
        stride_x_row stride_dy_row stride_dx_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DB, bwdParamGradOffset s N i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdBiasDBSpec s DY stride_dy_row i))
```

**Assumptions / layout contracts:**
- `hDWDX : DW ≠ DX`
- `hDWDB : DW ≠ DB`
- `hDBDX : DB ≠ DX`
- `hDBDW : DB ≠ DW`
- `fun i : Fin BLOCK_N => i.val < N`
- `fun i : Fin BLOCK_N => i.val < N`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_bwd_c1_reduction_slice`, `bwdC1ReductionSpec`, `layer_norm_ops_bwd_c2_reduction_slice`, `bwdC2ReductionSpec`, `layer_norm_ops_bwd_plain_dx_from_c1_c2_slice`, `bwdRmsDXOffset`, `bwdPlainDXFromC1C2Spec`, `layer_norm_ops_bwd_plain_bias_one_row`, `bwdParamGradOffset`, `bwdPlainBiasDWSpec`, `bwdBiasDBSpec`, `bwdRecomputeXhatOffset`, `bwdRmsWdyTile`, `bwdRmsDYOffset`, `bwdRmsDYTile`, `bwdPlainXhatTile`, `bwdRmsWTile`, `bwdRmsXOffset`

<details><summary><code>layer_norm_ops_bwd_c1_reduction_slice</code></summary>

```
/-! ## Backward `c1`/`c2` reduction-production slices

The full backward kernel computes `c1 = sum(xhat * (W * DY)) / N` in both RMS
and non-RMS branches, and additionally computes `c2 = sum(W * DY) / N` in the
non-RMS branch. These slices isolate the two row-local reductions and their
scalar stores so the reduction production itself is checked independently of
the surrounding branch/control-flow proof. -/
```
```lean
def layer_norm_ops_bwd_c1_reduction_slice
    (Xhat W DY C1 : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  tl.store(C1 + row, c1)
}
```
</details>

<details><summary><code>bwdC1ReductionSpec</code></summary>

```lean
noncomputable def bwdC1ReductionSpec
    (s : BlockState) (Xhat W DY : RegionName)
    (stride_xhat_row stride_dy_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun a => a / (N : ℝ))
      ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
        (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
          { data := fun idx =>
              if idx.1.val < N then
                some (s.readMem Xhat
                  (bwdRecomputeXhatOffset s stride_xhat_row idx.1))
              else some (0.0 : ℝ) }
          (bwdRmsWdyTile s W DY stride_dy_row N BLOCK_N))).data PUnit.unit))
```
</details>

<details><summary><code>layer_norm_ops_bwd_c2_reduction_slice</code></summary>

```lean
def layer_norm_ops_bwd_c2_reduction_slice
    (W DY C2 : RegionName)
    (stride_dy_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  wdy = w * dy
  c2 = tl.sum(wdy, axis=0) / $(N)
  tl.store(C2 + row, c2)
}
```
</details>

<details><summary><code>bwdC2ReductionSpec</code></summary>

```lean
noncomputable def bwdC2ReductionSpec
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : ℝ :=
  WithBot.unbotD 0
    (Option.map (fun a => a / (N : ℝ))
      ((Tile.reduceSum (shape := [BLOCK_N]) ⟨0, by simp⟩ Bool.false
        (bwdRmsWdyTile s W DY stride_dy_row N BLOCK_N)).data PUnit.unit))
```
</details>

<details><summary><code>layer_norm_ops_bwd_plain_dx_from_c1_c2_slice</code></summary>

```
/-- Non-RMS base-DX formula after `c1` and `c2` have been reduced:
`dx = (wdy - (xhat * c1 + c2)) * rstd`. -/
```
```lean
def layer_norm_ops_bwd_plain_dx_from_c1_c2_slice
    (Xhat W DY Rstd C1 C2 DX : RegionName)
    (stride_xhat_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  xhat = tl.load(Xhat + row * $(stride_xhat_row) + cols, mask=mask, other=0.0)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  rstd = tl.load(Rstd + row)
  c1 = tl.load(C1 + row)
  c2 = tl.load(C2 + row)
  wdy = w * dy
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
}
```
</details>

<details><summary><code>bwdRmsDXOffset</code></summary>

```lean
def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val
```
</details>

<details><summary><code>bwdPlainDXFromC1C2Spec</code></summary>

```lean
noncomputable def bwdPlainDXFromC1C2Spec
    (s : BlockState) (Xhat W DY Rstd C1 C2 : RegionName)
    (stride_xhat_row stride_dy_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  (s.readMem W i.val *
      s.readMem DY (bwdRmsDYOffset s stride_dy_row i) -
    (s.readMem Xhat (bwdRecomputeXhatOffset s stride_xhat_row i) *
        s.readMem C1 s.pid +
      s.readMem C2 s.pid)) *
    s.readMem Rstd s.pid
```
</details>

<details><summary><code>layer_norm_ops_bwd_plain_bias_one_row</code></summary>

```
/-! ## Backward non-RMS+bias one-row integrated slice

This is the first non-RMS integrated backward row slice for #133. Unlike the smaller
`DX`/`DW`/`DB` slices above, this kernel keeps the Python row arithmetic together:
`xhat`, `wdy`, partial `dw`, partial `db`, `c1`, `c2`, `dx`, then the three
observable stores `DX`, `DW`, and `DB`. The theorems here prove the integrated
`DW` and `DB` stores; `DX` still needs a reduction-heavy readback theorem. -/
```
```lean
def layer_norm_ops_bwd_plain_bias_one_row
    (X W DY DX DW DB Mean Rstd : RegionName)
    (stride_x_row stride_dy_row stride_dx_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  x = tl.load(X + row * $(stride_x_row) + cols, mask=mask, other=0.0).to(tl.float32)
  dy = tl.load(DY + row * $(stride_dy_row) + cols, mask=mask, other=0.0).to(tl.float32)
  mean = tl.load(Mean + row)
  rstd = tl.load(Rstd + row)
  w = tl.load(W + cols, mask=mask).to(tl.float32)
  xhat = (x - mean) * rstd
  xhat = tl.where(mask, xhat, 0.0)
  wdy = w * dy
  dw = dy * xhat
  db = dy
  c1 = tl.sum(xhat * wdy, axis=0) / $(N)
  c2 = tl.sum(wdy, axis=0) / $(N)
  dx = (wdy - (xhat * c1 + c2)) * rstd
  tl.store(DX + row * $(stride_dx_row) + cols, dx, mask=mask)
  tl.store(DW + row * $(N) + cols, dw, mask=mask)
  tl.store(DB + row * $(N) + cols, db, mask=mask)
}
```
</details>

<details><summary><code>bwdParamGradOffset</code></summary>

```lean
def bwdParamGradOffset (s : BlockState) (N : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * N + i.val
```
</details>

<details><summary><code>bwdPlainBiasDWSpec</code></summary>

```lean
noncomputable def bwdPlainBiasDWSpec
    (s : BlockState) (X DY Mean Rstd : RegionName)
    (stride_x_row stride_dy_row N BLOCK_N : Nat) (i : Fin BLOCK_N) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun dy xhat => dy * xhat)
      ((bwdRmsDYTile s DY stride_dy_row N BLOCK_N).data (i, PUnit.unit))
      ((bwdPlainXhatTile s X Mean Rstd stride_x_row N BLOCK_N).data
        (i, PUnit.unit)))
```
</details>

<details><summary><code>bwdBiasDBSpec</code></summary>

```lean
noncomputable def bwdBiasDBSpec
    (s : BlockState) (DY : RegionName) (stride_dy_row : Nat)
    (i : Fin BLOCK_N) : ℝ :=
  s.readMem DY (bwdRmsDYOffset s stride_dy_row i)
```
</details>

<details><summary><code>bwdRecomputeXhatOffset</code></summary>

```lean
def bwdRecomputeXhatOffset (s : BlockState) (stride_xhat_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_xhat_row + i.val
```
</details>

<details><summary><code>bwdRmsWdyTile</code></summary>

```lean
noncomputable def bwdRmsWdyTile
    (s : BlockState) (W DY : RegionName)
    (stride_dy_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
    (bwdRmsWTile s W N BLOCK_N)
    (bwdRmsDYTile s DY stride_dy_row N BLOCK_N)
```
</details>

<details><summary><code>bwdRmsDYOffset</code></summary>

```lean
def bwdRmsDYOffset (s : BlockState) (stride_dy_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dy_row + i.val
```
</details>

<details><summary><code>bwdRmsDYTile</code></summary>

```lean
noncomputable def bwdRmsDYTile
    (s : BlockState) (DY : RegionName) (stride_dy_row N BLOCK_N : Nat) :
    Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem DY (bwdRmsDYOffset s stride_dy_row idx.1))
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdPlainXhatTile</code></summary>

```lean
noncomputable def bwdPlainXhatTile
    (s : BlockState) (X Mean Rstd : RegionName)
    (stride_x_row N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        match
          match
            if idx.1.val < N then
              some (s.readMem X (bwdRmsXOffset s stride_x_row idx.1))
            else some (0.0 : ℝ)
          with
          | some x => some (x - s.readMem Mean s.pid)
          | none => none
        with
        | some x => some (x * s.readMem Rstd s.pid)
        | none => none
      else some (0.0 : ℝ) }
```
</details>

<details><summary><code>bwdRmsWTile</code></summary>

```lean
noncomputable def bwdRmsWTile
    (s : BlockState) (W : RegionName) (N BLOCK_N : Nat) : Tile .real [BLOCK_N] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem W idx.1.val)
      else some (s.undef W idx.1.val) }
```
</details>

<details><summary><code>bwdRmsXOffset</code></summary>

```lean
def bwdRmsXOffset (s : BlockState) (stride_x_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_x_row + i.val
```
</details>

## Public theorem: `layer_norm_ops_bwd_residual_add_all_outputs_compute_correct_general`

<details><summary>docstring</summary>

```
/-- Backward residual-add, **dimension-general**: exposes both genuine
observable row-vector stores (`DX` and `DRESIDUAL_IN`) produced by the
`dx += dres` branch, for arbitrary row strides, feature dim `N`, and tile width
`BLOCK_N`. -/
```
</details>

**Statement:**
```lean
theorem layer_norm_ops_bwd_residual_add_all_outputs_compute_correct_general
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName) (s : BlockState)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat)
    (hDXDresIn : DX ≠ DRESIDUAL_IN)
    (hDresInDX : DRESIDUAL_IN ≠ DX) :
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DX, bwdRmsDXOffset s stride_dx_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i)) ∧
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := layer_norm_ops_bwd_residual_add_store_slice DXBase DRESIDUAL
        DX DRESIDUAL_IN stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_N => i.val < N)
        (fun i => (DRESIDUAL_IN, bwdDResidualInOffset s stride_dres_in_row i)))
      (expected := fun i : Fin BLOCK_N =>
        bwdResidualAddSpec s DXBase DRESIDUAL stride_dx_row stride_dres_row i))
```

**Assumptions / layout contracts:**
- `hDXDresIn : DX ≠ DRESIDUAL_IN`
- `hDresInDX : DRESIDUAL_IN ≠ DX`
- `fun i : Fin BLOCK_N => i.val < N`
- `fun i : Fin BLOCK_N => i.val < N`

**Closed-form spec defs (transitive):** `layer_norm_ops_bwd_residual_add_store_slice`, `bwdRmsDXOffset`, `bwdResidualAddSpec`, `bwdDResidualInOffset`, `bwdDResidualOffset`

<details><summary><code>layer_norm_ops_bwd_residual_add_store_slice</code></summary>

```
/-! ## Backward residual-add arithmetic slice

When `HAS_DRESIDUAL` is true, the Python backward kernel adds the residual
gradient tile into the already-computed `dx`, optionally stores the combined
value to `DRESIDUAL_IN`, and always stores it to `DX`. This slice proves that
arithmetic/writeback path directly. -/
```
```lean
def layer_norm_ops_bwd_residual_add_store_slice
    (DXBase DRESIDUAL DX DRESIDUAL_IN : RegionName)
    (stride_dx_row stride_dres_row stride_dres_in_row N BLOCK_N : Nat) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  cols = tl.arange(0, $(BLOCK_N))
  mask = cols < $(N)
  dx = tl.load(DXBase + row * $(stride_dx_row) + cols, mask=mask, other=0.0)
  dres = tl.load(DRESIDUAL + row * $(stride_dres_row) + cols, mask=mask, other=0.0)
  out = dx + dres
  tl.store(DRESIDUAL_IN + row * $(stride_dres_in_row) + cols, out, mask=mask)
  tl.store(DX + row * $(stride_dx_row) + cols, out, mask=mask)
}
```
</details>

<details><summary><code>bwdRmsDXOffset</code></summary>

```lean
def bwdRmsDXOffset (s : BlockState) (stride_dx_row : Nat) (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dx_row + i.val
```
</details>

<details><summary><code>bwdResidualAddSpec</code></summary>

```lean
noncomputable def bwdResidualAddSpec
    (s : BlockState) (DXBase DRESIDUAL : RegionName)
    (stride_dx_row stride_dres_row : Nat) (i : Fin BLOCK_N) : ℝ :=
  s.readMem DXBase (bwdRmsDXOffset s stride_dx_row i) +
    s.readMem DRESIDUAL (bwdDResidualOffset s stride_dres_row i)
```
</details>

<details><summary><code>bwdDResidualInOffset</code></summary>

```lean
def bwdDResidualInOffset (s : BlockState) (stride_dres_in_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dres_in_row + i.val
```
</details>

<details><summary><code>bwdDResidualOffset</code></summary>

```lean
def bwdDResidualOffset (s : BlockState) (stride_dres_row : Nat)
    (i : Fin BLOCK_N) : Nat :=
  s.pid * stride_dres_row + i.val
```
</details>

## Also present (pinned special-case summaries)
- `layer_norm_fwd_rms_one_block_y_compute_correct`
- `layer_norm_fwd_rms_bias_one_block_y_compute_correct`
- `layer_norm_fwd_rms_one_block_rstd_compute_correct`
- `layer_norm_fwd_rms_bias_one_block_rstd_compute_correct`
- `layer_norm_fwd_plain_one_block_y_compute_correct`
- `layer_norm_fwd_plain_bias_one_block_y_compute_correct`
- `layer_norm_fwd_plain_residual_one_block_y_compute_correct`
- `layer_norm_fwd_plain_residual_one_block_mean_compute_correct`
- `layer_norm_fwd_plain_residual_one_block_rstd_compute_correct`
- `layer_norm_fwd_rms_residual_one_block_y_compute_correct`
- `layer_norm_fwd_rms_residual_one_block_rstd_compute_correct`
- `layer_norm_fwd_rms_residual_bias_one_block_y_compute_correct`
- `layer_norm_fwd_rms_residual_bias_one_block_rstd_compute_correct`
- `layer_norm_fwd_plain_residual_bias_one_block_y_compute_correct`
- `layer_norm_fwd_plain_residual_bias_one_block_mean_compute_correct`
- `layer_norm_fwd_plain_residual_bias_one_block_rstd_compute_correct`
- `layer_norm_fwd_plain_one_block_mean_compute_correct`
- `layer_norm_fwd_plain_one_block_rstd_compute_correct`
- `layer_norm_fwd_plain_bias_one_block_mean_compute_correct`
- `layer_norm_fwd_plain_bias_one_block_rstd_compute_correct`
- `layer_norm_ops_fwd_mean_store_slice_compute_correct`
- `layer_norm_ops_fwd_rstd_store_slice_compute_correct`
- `layer_norm_ops_bwd_row_vector_store_slice_compute_correct`
- `layer_norm_ops_bwd_dx_store_slice_compute_correct`
- `layer_norm_ops_bwd_dresidual_in_store_slice_compute_correct`
- `layer_norm_ops_bwd_recompute_y_store_slice_compute_correct`
- `layer_norm_ops_fwd_residual_out_store_slice_compute_correct`
- `layer_norm_ops_fwd_y_store_slice_compute_correct`
- `layer_norm_ops_fwd_bias_y_store_slice_compute_correct`
- `layer_norm_ops_fwd_residual_y_store_slice_compute_correct`
- `layer_norm_ops_fwd_rms_bias_y_store_slice_compute_correct`
- `layer_norm_ops_fwd_residual_bias_y_store_slice_compute_correct`
- `layer_norm_ops_bwd_param_grad_store_slice_compute_correct`
- `layer_norm_ops_bwd_dw_store_slice_compute_correct`
- `layer_norm_ops_bwd_db_store_slice_compute_correct`
- `layer_norm_ops_bwd_rms_one_row_dw_compute_correct`
- `layer_norm_ops_bwd_bias_db_one_row_compute_correct`
- `layer_norm_ops_bwd_plain_bias_one_row_dw_compute_correct`
- `layer_norm_ops_bwd_plain_bias_one_row_db_compute_correct`
- `layer_norm_ops_bwd_residual_add_store_slice_dx_compute_correct`
- `layer_norm_ops_bwd_residual_add_store_slice_dresidual_in_compute_correct`
- `layer_norm_ops_bwd_recompute_y_bias_slice_compute_correct`
- `layer_norm_ops_bwd_recompute_y_no_bias_slice_compute_correct`
- `layer_norm_ops_bwd_c1_reduction_slice_compute_correct`
- `layer_norm_ops_bwd_c2_reduction_slice_compute_correct`
- `layer_norm_ops_bwd_rms_dx_from_c1_slice_compute_correct`
- `layer_norm_ops_bwd_plain_dx_from_c1_c2_slice_compute_correct`
- `layer_norm_fwd_1pass_surface_rms_only_y_compute_correct`
- `layer_norm_fwd_1pass_surface_rms_only_rstd_compute_correct`
- `layer_norm_bwd_surface_zero_rows_db_zero_compute_correct`
- `layer_norm_bwd_surface_zero_rows_dw_zero_compute_correct`
