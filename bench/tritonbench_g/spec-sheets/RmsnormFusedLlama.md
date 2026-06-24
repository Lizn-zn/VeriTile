# Spec sheet — `bench/tritonbench_g/rmsnorm_fused_llama/RmsnormFusedLlama.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_fused_llama/rmsnorm_fused_llama.py`

## Public theorem: `rms_norm_fwd_fused_llama_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `_rms_norm_fwd_fused` (Llama): the DSL surface
lowers to the algorithm layer, and the masked fp16 store to `Y` is
compute-correct — every active lane (`i.val < N`) holds the fp16-cast RMSNorm
spec, out-of-bounds lanes are preserved. Stated under the `0 < N ≤ BLOCK_SIZE`
single-block launch precondition chosen by the Python wrapper. -/
```
</details>

**Statement:**
```lean
theorem rms_norm_fwd_fused_llama_output_summary
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ)
    (s : BlockState)
    (hNpos : 0 < N) (hNle : N ≤ BLOCK_SIZE)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)) :
    (∃ alg, (rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_SIZE => i.val < N)
        (fun i => (Y, yOffset s stride i)))
      (expected := fun i =>
        MemCell.of .fp16
          (FloatDType.real.cast FloatDType.fp16
            (some (rmsnormSpec s X W stride N BLOCK_SIZE eps i))))
```

**Assumptions / layout contracts:**
- `hNpos : 0 < N`
- `hNle : N ≤ BLOCK_SIZE`
- `hOutInj : Function.Injective
      (fun i : Fin BLOCK_SIZE => yOffset s stride i)`
- `kernel : = rms_norm_fwd_fused_llama X Y W stride N BLOCK_SIZE eps`
- `initialState : = s`
- `fun i : Fin BLOCK_SIZE => i.val < N`

**Closed-form spec defs (transitive):** `yOffset`, `rms_norm_fwd_fused_llama`, `rmsnormSpec`, `rmsnormCarrierSpec`, `xOffset`, `rmsInvCarrier`, `rmsVarCarrier`, `rmsInputTile`

<details><summary><code>yOffset</code></summary>

```lean
def yOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val
```
</details>

<details><summary><code>rms_norm_fwd_fused_llama</code></summary>

```
/-- Faithful `forRange` transcription of `rmsnorm_fused_llama.py`'s
`_rms_norm_fwd_fused`.

The Python wrapper fixes `BLOCK_SIZE = 16384` after checking `N <= BLOCK_SIZE`,
so the correctness theorem below proves the loop-shaped kernel under that
runtime precondition. Under the precondition both `range(0, N, BLOCK_SIZE)`
loops execute exactly the `off = 0` iteration.

Allowed mechanical Lean-syntax-only changes:
- Python `N` / `BLOCK_SIZE: tl.constexpr` -> Lean `Nat` parameters. -/
```
```lean
def rms_norm_fwd_fused_llama
    (X Y W : RegionName) (stride N BLOCK_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  row = tl.program_id(0)
  Y += row * $(stride)
  X += row * $(stride)
  _var = tl.zeros([$(BLOCK_SIZE)], dtype=tl.float32)
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    x = tl.load(X + cols, mask=cols < $(N), other=0.0).to(tl.float32)
    _var += x * x
  }
  var = tl.sum(_var, axis=0) / $(N)
  rstd = 1 / tl.sqrt(var + $(eps))
  for off in range(0, $(N), $(BLOCK_SIZE)) {
    cols = off + tl.arange(0, $(BLOCK_SIZE))
    mask = cols < $(N)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w
    tl.store(Y + cols, (y).to(tl.float16), mask=mask)
  }
}
```
</details>

<details><summary><code>rmsnormSpec</code></summary>

```lean
noncomputable def rmsnormSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  rmsnormCarrierSpec s X W stride N BLOCK_SIZE eps i
```
</details>

<details><summary><code>rmsnormCarrierSpec</code></summary>

```lean
noncomputable def rmsnormCarrierSpec
    (s : BlockState) (X W : RegionName)
    (stride N BLOCK_SIZE : Nat) (eps : ℝ) (i : Fin BLOCK_SIZE) : ℝ :=
  WithBot.unbotD 0
    (Option.map₂ (fun scaled w => scaled * w)
      (Option.map₂ (fun x inv => x * inv)
        (some (s.readMem X (xOffset s stride i)))
        (rmsInvCarrier s X stride N BLOCK_SIZE eps))
      (some (s.readMem W i.val)))
```
</details>

<details><summary><code>xOffset</code></summary>

```lean
def xOffset (s : BlockState) (stride : Nat) (i : Fin BLOCK_SIZE) : Nat :=
  s.pid * stride + i.val
```
</details>

<details><summary><code>rmsInvCarrier</code></summary>

```lean
noncomputable def rmsInvCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat)
    (eps : ℝ) : WithBot ℝ :=
  Option.map (fun b => b⁻¹)
    (WithBot.realSqrt
      (Option.map (fun a => a + eps)
        (rmsVarCarrier s X stride N BLOCK_SIZE)))
```
</details>

<details><summary><code>rmsVarCarrier</code></summary>

```lean
noncomputable def rmsVarCarrier
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    WithBot ℝ :=
  Option.map₂ (fun a n => a / n)
    ((Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false
      (Tile.bop (NumericDType.mul .real) (Broadcast.consSame Broadcast.nil)
        (rmsInputTile s X stride N BLOCK_SIZE)
        (rmsInputTile s X stride N BLOCK_SIZE))).data PUnit.unit)
    ((Tile.scalar (dtype := .real) (some (N : ℝ) : WithBot ℝ)).data PUnit.unit)
```
</details>

<details><summary><code>rmsInputTile</code></summary>

```lean
noncomputable def rmsInputTile
    (s : BlockState) (X : RegionName) (stride N BLOCK_SIZE : Nat) :
    Tile .real [BLOCK_SIZE] :=
  { data := fun idx =>
      if idx.1.val < N then
        some (s.readMem X (xOffset s stride idx.1))
      else some (0.0 : ℝ) }
```
</details>

## Also present (pinned special-case summaries)
- `rms_norm_fwd_fused_llama_compute_correct`
