# Spec sheet — `bench/tritonbench_g/rmsnorm_triton/RmsnormTriton.lean`

**Python source:** `bench/tritonbench_g/rmsnorm_triton/rmsnorm_triton.py`

## Public theorem: `rmsnorm_full_output_summary`

<details><summary>docstring</summary>

```
/-- **Full output summary**: the surface lowers to the algorithm layer, and the
masked store realizes the genuine multi-block RMS-norm closed form at every
global lane `k < N_SIZE`. Holds for arbitrary `N_SIZE` — no
`N_SIZE ≤ BLOCK_N_SIZE` hypothesis. -/
```
</details>

**Statement:**
```lean
theorem rmsnorm_full_output_summary
    (x w o : RegionName) (sxb sxm sxk srw sob som sok N B : Nat) (eps : ℝ)
    (s : BlockState) (hB : 0 < B) (hNpos : 0 < N)
    (hox : o ≠ x) (how : o ≠ w)
    (hsok : 0 < sok) :
    (∃ alg, (VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := VeriTile.Bench.TritonBenchG.RmsnormTriton.rmsnorm_triton x w o
        sxb sxm sxk srw sob som sok N B eps)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _ : Fin N => True)
        (fun k => (o, outOff s sob som sok k.val)))
      (expected := fun k : Fin N => rmsSpecFull s x w sxb sxm sxk srw N eps k.val)
```

**Assumptions / layout contracts:**
- `hB : 0 < B`
- `hNpos : 0 < N`
- `hox : o ≠ x`
- `how : o ≠ w`
- `hsok : 0 < sok`
- `fun _ : Fin N => True`

**Closed-form spec defs (transitive):** `rmsnorm_triton`, `outOff`, `rmsSpecFull`, `xOff`, `meanSq`

<details><summary><code>rmsnorm_triton</code></summary>

```
/-- Faithful transcription of `rmsnorm_triton.py`'s `rmsnorm_triton`.

Allowed mechanical Lean-syntax-only changes:
- Python `N_SIZE: tl.constexpr` / `eps: tl.constexpr` / `BLOCK_N_SIZE: tl.constexpr`
  -> Lean `Nat` / `ℝ` parameters. -/
```
```lean
def rmsnorm_triton
    (x_ptr rms_w_ptr output_ptr : RegionName)
    (stride_x_batch stride_x_m stride_x_k stride_rms_w
      stride_out_batch stride_out_m stride_out_k : Nat)
    (N_SIZE BLOCK_N_SIZE : Nat) (eps : ℝ) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(0)
  pid_m = tl.program_id(1)
  offs_m = pid_batch * $(stride_x_batch) + pid_m * $(stride_x_m)
  block_N = tl.arange(0, $(BLOCK_N_SIZE))
  var = tl.zeros([$(BLOCK_N_SIZE)], tl.float32)
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0)
    var += tl.extra.cuda.libdevice.pow((x).to(tl.float32), 2)
  }
  var = tl.sum(var, axis=0) / $(N_SIZE)
  rstd = tl.math.rsqrt(var + $(eps))
  for block_n_start_idx in range(0, $(N_SIZE), $(BLOCK_N_SIZE)) {
    offs_n = block_n_start_idx + block_N
    x_ptr_mask = offs_n < $(N_SIZE)
    rms_w = tl.load(rms_w_ptr + offs_n * $(stride_rms_w), mask=x_ptr_mask)
    x = tl.load(x_ptr + offs_m + offs_n * $(stride_x_k), mask=x_ptr_mask, other=0.0).to(tl.float32)
    x_hat = x * rstd
    out = x_hat * rms_w
    out_off = pid_batch * $(stride_out_batch) + pid_m * $(stride_out_m) +
      offs_n * $(stride_out_k)
    tl.store(output_ptr + out_off, out, mask=x_ptr_mask)
  }
}
```
</details>

<details><summary><code>outOff</code></summary>

```lean
def outOff (s : BlockState) (sob som sok : Nat) (k : Nat) : Nat :=
  s.pids 0 * sob + s.pids 1 * som + k * sok
```
</details>

<details><summary><code>rmsSpecFull</code></summary>

```lean
noncomputable def rmsSpecFull (s : BlockState) (x w : RegionName) (sxb sxm sxk srw N : Nat) (eps : ℝ) (k : Nat) : ℝ :=
  s.readMem x (xOff s sxb sxm sxk k) * (WithBot.unbotD 0 (WithBot.realRsqrt (some (meanSq s x sxb sxm sxk N + eps)))) * s.readMem w (k*srw)
```
</details>

<details><summary><code>xOff</code></summary>

```lean
def xOff (s : BlockState) (sxb sxm sxk : Nat) (k : Nat) : Nat :=
  s.pids 0 * sxb + s.pids 1 * sxm + k * sxk
```
</details>

<details><summary><code>meanSq</code></summary>

```lean
noncomputable def meanSq (s : BlockState) (x : RegionName) (sxb sxm sxk N : Nat) : ℝ :=
  (∑ k : Fin N, (s.readMem x (xOff s sxb sxm sxk k.val))^2) / (N:ℝ)
```
</details>

## Also present (pinned special-case summaries)
- `rmsnorm_full_compute_correct`
