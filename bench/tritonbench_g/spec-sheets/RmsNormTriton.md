# Spec sheet — `bench/tritonbench_g/rms_norm_triton/RmsNormTriton.lean`

**Python source:** `bench/tritonbench_g/rms_norm_triton/rms_norm_triton.py`

## Public theorem: `rms_norm_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `rms_norm_kernel`: the DSL surface lowers to the
algorithm layer, and the masked store to `Y` is compute-correct — every in-bounds
lane holds `rmsNormSpec`, out-of-bounds lanes are preserved. -/
```
</details>

**Statement:**
```lean
theorem rms_norm_kernel_output_summary
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => rmsOutOffset s y_stride_r y_stride_c i)) :
    (∃ alg, (rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
        N eps BLOCK_SIZE).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
        N eps BLOCK_SIZE)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, rmsOutOffset s y_stride_r y_stride_c i)))
      (expected := fun i => rmsNormSpec s X W x_stride_r x_stride_c N BLOCK_SIZE eps i)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => rmsOutOffset s y_stride_r y_stride_c i)`
- `fun i : Fin BLOCK_SIZE => i.val < N`

**Closed-form spec defs (transitive):** `rmsOutOffset`, `rms_norm_kernel`, `rmsNormSpec`, `rmsRrmsCarrier`, `rmsSumCarrier`, `rmsInputTile`

<details><summary><code>rmsOutOffset</code></summary>

```lean
def rmsOutOffset (s : BlockState) (y_stride_r y_stride_c : Nat)
    (i : Fin BLOCK_SIZE) : Nat :=
  s.pids 0 * y_stride_r + i.val * y_stride_c
```
</details>

<details><summary><code>rms_norm_kernel</code></summary>

```
/-- Faithful transcription of `rms_norm_triton.py`'s `rms_norm_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameter. -/
```
```lean
def rms_norm_kernel
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  Y += pid * $(y_stride_r)
  X += pid * $(x_stride_r)
  mask = tl.arange(0, $(BLOCK_SIZE)) < $(N)
  cols = tl.arange(0, $(BLOCK_SIZE))
  x = tl.load(X + cols * $(x_stride_c), mask, other=0.0).to(tl.float32)
  var = tl.sum(x * x, axis=0) / $(N)
  rrms = 1 / tl.sqrt(var + $(eps))
  w = tl.load(W + tl.arange(0, $(BLOCK_SIZE)), mask=mask, other=0.0)
  y = (x * rrms).to(Y.dtype.element_ty) * w
  tl.store(Y + cols * $(y_stride_c), y, mask=mask)
}
```
</details>

<details><summary><code>rmsNormSpec</code></summary>

```lean
noncomputable def rmsNormSpec
    (s : BlockState) (X W : RegionName)
    (x_stride_r x_stride_c N BLOCK_SIZE : Nat) (eps : ℝ) (idx : Fin BLOCK_SIZE) :
    ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun x w => x * w)
      (Option.map₂ (fun x rrms => x * rrms)
        (some (s.readMem X (s.pids 0 * x_stride_r + idx.val * x_stride_c)))
        (rmsRrmsCarrier s X x_stride_r x_stride_c N BLOCK_SIZE eps))
      (some (s.readMem W idx.val)))
```
</details>

<details><summary><code>rmsRrmsCarrier</code></summary>

```lean
noncomputable def rmsRrmsCarrier
    (s : BlockState) (X : RegionName) (x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt (Option.map ((fun a => a + eps) ∘ fun a => a / (N : ℝ))
      (rmsSumCarrier s X x_stride_r x_stride_c N BLOCK_SIZE)))
```
</details>

<details><summary><code>rmsSumCarrier</code></summary>

```lean
noncomputable def rmsSumCarrier
    (s : BlockState) (X : RegionName) (x_stride_r x_stride_c N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  (Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
    (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
      (rmsInputTile s X x_stride_r x_stride_c N BLOCK_SIZE)
      (rmsInputTile s X x_stride_r x_stride_c N BLOCK_SIZE))).data PUnit.unit
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (x_stride_r x_stride_c N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      let off := s.pids 0 * x_stride_r + idx.1.val * x_stride_c
      if idx.1.val < N then some (s.readMem X off) else some (0.0 : ℝ) }
```
</details>

## Also present (pinned special-case summaries)
- `rms_norm_kernel_compute_correct`
