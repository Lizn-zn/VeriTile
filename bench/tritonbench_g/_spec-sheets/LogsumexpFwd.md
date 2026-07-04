# Spec sheet — `bench/tritonbench_g/logsumexp_fwd/LogsumexpFwd.lean`

**Python source:** `bench/tritonbench_g/logsumexp_fwd/logsumexp_fwd.py`

## Public theorem: `logsumexp_fwd_kernel_output_summary`

<details><summary>docstring</summary>

```
/-- Per-kernel output summary for `logsumexp_fwd_kernel`: the DSL surface lowers
to the algorithm layer, and in the single-block configuration `D = B = n + 1`
(pid axis-1 = 0) the store to `z[i_n]` is compute-correct — it holds the
standard mathematical row `LSE xs HAS_SCALE scale`. -/
```
</details>

**Statement:**
```lean
theorem logsumexp_fwd_kernel_output_summary
    (x z : RegionName)
    (n : Nat) (HAS_SCALE : Bool) (scale : ℝ)
    (s : BlockState)
    (xs : Fin (n+1) → ℝ)
    (h_x : ∀ j : Fin (n+1), s.readMem x (s.pids 0 * (n+1) + j.val) = xs j)
    (h_pid1 : s.pids 1 = 0) :
    (∃ alg, (logsumexp_fwd_kernel x z (n+1) (n+1) HAS_SCALE scale).toAlgorithm? =
        Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := logsumexp_fwd_kernel x z (n+1) (n+1) HAS_SCALE scale)
      (initialState := s)
      (write := fun _ : PUnit => some (z, s.pids 0))
      (expected := fun _ => LSE xs HAS_SCALE scale)
```

**Assumptions / layout contracts:**
- `xs : Fin (n+1) → ℝ`
- `h_pid1 : s.pids 1 = 0`

**Closed-form spec defs (transitive):** `logsumexp_fwd_kernel`

<details><summary><code>logsumexp_fwd_kernel</code></summary>

```
/-- Faithful 1:1 transcription of `logsumexp_fwd.py`'s `logsumexp_fwd_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `D: tl.constexpr` / `B: tl.constexpr` / `HAS_SCALE: tl.constexpr` →
  Lean parameters; the `tl.constexpr` annotation is implicit on Lean params.
- Python `if cond: body` → `tl.if cond { body }`, the DSL-side gate equivalent.
- `scale` (Lean `ℝ` parameter) injected via `$(...)`. -/
```
```lean
def logsumexp_fwd_kernel
    (x z : RegionName)
    (D B : Nat) (HAS_SCALE : Bool) (scale : ℝ) :
    ComputeKernel := triton {
  i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
  o_d = i_d * $(B) + tl.arange(0, $(B))
  m_d = o_d < $(D)
  b_x = tl.load(x + i_n * $(D) + o_d, mask=m_d, other=-float("inf"))
  if HAS_SCALE {
    b_x = b_x * $(scale)
  }
  b_m = tl.max(b_x, 0)
  b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
  tl.store(z + i_n * tl.cdiv($(D), $(B)) + i_d, b_z)
}
```
</details>

## Also present (pinned special-case summaries)
- `logsumexp_fwd_kernel_compute_correct`
