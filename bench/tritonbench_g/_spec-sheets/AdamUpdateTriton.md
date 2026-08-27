# Spec sheet — `bench/tritonbench_g/adam_update_triton/AdamUpdateTriton.lean`

**Python source:** `bench/tritonbench_g/adam_update_triton/adam_update_triton.py`

## Public theorem: `update_fn_kernel_correctness`

<details><summary>docstring</summary>

```
/-- **The headline**: `update_fn_kernel` implements the Lion step on its
masked in-place IO signature — for every disjoint flat placement of the
three buffers, every program id whose active lanes are in bounds, and every
launch state whose input windows are loaded at the active lanes, the
translated pointer kernel terminates, every active lane of the parameter
buffer ends up holding `lionParam` and of the momentum buffer
`lionMomentum`, applied to the *originally loaded* windows; every other
flat cell is untouched. The side condition `p_ptr ≠ exp_avg_ptr` rules out
aliasing between the two output buffers (the second masked store would
otherwise clobber the first output). Proof:
`MaskedKernelIO₃ₓ₂.Implements.intro` assembles the region-model triple with
the flat-memory bridge side conditions. -/
```
</details>

**Statement:**
```lean
specification update_fn_kernel_correctness
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (hRegions : p_ptr ≠ exp_avg_ptr) :
    adamIO p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n_elements BLOCK_SIZE
      ⊨ fun p grad expAvg =>
        (fun i => TiledOptimizer.lionParam (p i) (expAvg i) (grad i) lr wd beta1,
         fun i => TiledOptimizer.lionMomentum (expAvg i) (grad i) beta2)
```

**Assumptions / layout contracts:**
- `hRegions : p_ptr ≠ exp_avg_ptr`

**Closed-form spec defs (transitive):** `adamIO`, `update_fn_kernel`

<details><summary><code>adamIO</code></summary>

```
/-- `update_fn_kernel`'s masked in-place **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `bufs` — the allocation list: three buffers, each exactly once;
* `in1`/`in2`/`in3` — parameters, gradient, momentum (the wiring);
* `out1 = in1`, `out2 = in3` — the **in-place** roles: the kernel rewrites
  the parameter and momentum buffers it read;
* `read1..3`/`write1..2` — every window is the same block
  `[pid·BLOCK_SIZE, pid·BLOCK_SIZE + BLOCK_SIZE)` (the launch convention
  `offsets = pid * BLOCK_SIZE + arange`);
* `mask` — program `pid`'s active lanes, `pid * BLOCK_SIZE + j < n_elements`.
  Inactive lanes (the overhang of the last partial block) carry no
  obligations.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer
sizes are not signature content: the headline quantifies over every
allocation whose extents cover the active lanes. -/
```
```lean
def adamIO (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    MaskedKernelIO₃ₓ₂ where
  kernel := update_fn_kernel p_ptr grad_ptr exp_avg_ptr
    lr wd beta1 beta2 n_elements BLOCK_SIZE
  bufs := [p_ptr, grad_ptr, exp_avg_ptr]  -- p and exp_avg are updated in place
  in1 := p_ptr
  in2 := grad_ptr
  in3 := exp_avg_ptr
  out1 := p_ptr          -- = in1: in-place parameter update
  out2 := exp_avg_ptr    -- = in3: in-place momentum update
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  read3 := fun pid => pid * BLOCK_SIZE
  write1 := fun pid => pid * BLOCK_SIZE
  write2 := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements
```
</details>

<details><summary><code>update_fn_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `adam_update_triton.py`'s `update_fn_kernel`. -/
```
```lean
def update_fn_kernel
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  offset_p_ptr = p_ptr + offsets
  offset_grad_ptr = grad_ptr + offsets
  offset_exp_avg_ptr = exp_avg_ptr + offsets
  p = tl.load(offset_p_ptr, mask=mask)
  grad = tl.load(offset_grad_ptr, mask=mask)
  exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)
  p = p * (1 - $((lr : ℝ)) * $((wd : ℝ)))
  diff = exp_avg - grad
  update = diff * $(beta1) + grad
  can_update = update != 0
  update_sign = tl.where(update > 0, -$((lr : ℝ)), $((lr : ℝ)))
  p = p + update_sign * can_update
  exp_avg = diff * $(beta2) + grad
  tl.store(offset_p_ptr, p, mask=mask)
  tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)
}
```
</details>
