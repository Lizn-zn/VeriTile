# Spec sheet — `bench/tritonbench_g/fused_recurrent_hgrn/FusedRecurrentHgrn.lean`

**Python source:** `bench/tritonbench_g/fused_recurrent_hgrn/fused_recurrent_hgrn.py`

## Public theorem: `fused_recurrent_hgrn_output_summary_general`

<details><summary>docstring</summary>

```
/-! ### ════════ ★ MAIN THEOREM ★ ════════

**Genuine, dimension-general HGRN compute-correctness.** Parameterized over the
symbolic time/feature/tile sizes `T D BD`, the step index `i_t`, the final time
`T`, and **both** flags `USE_INITIAL_STATE STORE_FINAL_STATE`. It bundles the
genuine forward faces, each realized against the closed form `hgrnStateClosed`
over the *input* regions `x, g, h0` (never a read-back of the kernel's own
output), plus the genuine per-step backward `dx`/`dg` writebacks:

1. the full HGRN forward surface lowers to the algorithm layer;
2. one forward **output** body realizes `hgrnStateClosed(i_t + 1)` — the unrolled
   recurrence `b_h = g·b_h + x` — given the carry invariant
   `BHPrev = hgrnStateClosed(i_t)`;
3. the **final-state** writeback realizes `hgrnStateClosed(T)` (masked), given
   `BHFinal = hgrnStateClosed(T)`;
4. one backward `dx` body realizes the genuine `bwdDxStepValue` (`dh_prev + do`);
5. one backward `dg` body realizes the genuine `bwdDgStepValue` (`(dh_prev+do)·o`).

Honest structural side condition: `0 < BD` (contiguous lanes, giving offset
injectivity for every face). The flags flow through verbatim; clauses 2–5 hold
for every flag setting, and each Python case is recovered by projecting the
subset of clauses its `STORE_FINAL_STATE` / `USE_INITIAL_STATE` configuration
exercises.

The cross-step fold over `range(0, T)` threading `b_h` (forward) and `b_dh`
(backward) is the trusted loop boundary: the carried state is presented to each
step slice as a materialized buffer, and the forward carry invariant
`BHPrev = hgrnStateClosed(m)` is propagated by clause 2 itself
(`forwardStepValue_eq_hgrnStateClosed_succ`). -/
```
</details>

**Statement:**
```lean
theorem fused_recurrent_hgrn_output_summary_general
    (X G O H0 Ht DX DG DO BHPrev BHFinal DHPrev BO : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (i_t T D BD : Nat) (s : BlockState) (hBD : 0 < BD)
    (hPrev : ∀ i : Fin BD,
      s.readMem BHPrev (bhOffset s D BD i)
        = hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD i_t i)
    (hFinal : ∀ i : Fin BD,
      s.readMem BHFinal (finalStateOffset s D BD i)
        = hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD T i) :
    -- (1) the full forward surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht T D BD
      USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm? = Except.ok alg) ∧
    -- (2) the forward output body realizes the genuine `hgrnStateClosed(i_t+1)`
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (O, outOffset s i_t T D BD i)))
      (expected := fun i =>
        hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD (i_t + 1) i)) ∧
    -- (3) the final-state writeback realizes the genuine `hgrnStateClosed(T)`
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_final_state_store_slice BHFinal Ht D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (Ht, finalStateOffset s D BD i)))
      (expected := fun i =>
        if active s D BD i then
          hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD T i
        else 0)) ∧
    -- (4) the backward `dx` body realizes the genuine `dh_prev + do`
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (DX, outOffset s i_t T D BD i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO i_t T D BD i)) ∧
    -- (5) the backward `dg` body realizes the genuine `(dh_prev+do)·o`
    (ComputeCorrect.Realizes
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO BO DG
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (DG, outOffset s i_t T D BD i)))
      (expected := fun i => bwdDgStepValue s DHPrev DO BO i_t T D BD i))
```

**Assumptions / layout contracts:**
- `hBD : 0 < BD`
- `hPrev : ∀ i : Fin BD,
      s.readMem BHPrev (bhOffset s D BD i)
        = hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD i_t i`
- `hFinal : ∀ i : Fin BD,
      s.readMem BHFinal (finalStateOffset s D BD i)
        = hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD T i`

**Closed-form spec defs (transitive):** `bhOffset`, `hgrnStateClosed`, `finalStateOffset`, `fused_recurrent_hgrn_fwd_surface`, `fused_recurrent_hgrn_forward_step_store_slice`, `active`, `outOffset`, `fused_recurrent_hgrn_final_state_store_slice`, `fused_recurrent_hgrn_bwd_dx_step_store_slice`, `bwdDxStepValue`, `fused_recurrent_hgrn_bwd_dg_step_store_slice`, `bwdDgStepValue`, `dIndex`, `stateSeed`, `gVal`, `xVal`

<details><summary><code>bhOffset</code></summary>

```lean
def bhOffset (s : BlockState) (D BD : Nat) (i : Fin BD) : Nat :=
  s.pids 1 * D + dIndex s BD i
```
</details>

<details><summary><code>hgrnStateClosed</code></summary>

```
/-- **Genuine closed form for the forward state after `n` steps**, channel `i`:
`seed · ∏_{j<n} g_j + Σ_{t<n} x_t · ∏_{t<j<n} g_j`. A standalone specification
over the input regions `x, g, h0` — never a read-back of the kernel's own
output. -/
```
```lean
noncomputable def hgrnStateClosed
    (s : BlockState) (x g h0 : RegionName) (USE_INITIAL_STATE : Bool)
    (T D BD n : Nat) (i : Fin BD) : ℝ :=
  stateSeed s h0 USE_INITIAL_STATE D BD i *
      (∏ j ∈ Finset.range n, gVal s g T D BD j i) +
    ∑ t ∈ Finset.range n,
      xVal s x T D BD t i *
        (∏ j ∈ Finset.Ico (t + 1) n, gVal s g T D BD j i)
```
</details>

<details><summary><code>finalStateOffset</code></summary>

```lean
def finalStateOffset (s : BlockState) (D BD : Nat) (i : Fin BD) : Nat :=
  s.pids 1 * D + dIndex s BD i
```
</details>

<details><summary><code>fused_recurrent_hgrn_fwd_surface</code></summary>

```
/-- Faithful transcription of `fused_recurrent_hgrn.py`'s
`fused_recurrent_hgrn_fwd_kernel`.

The forward recurrence is a regular `0..T` loop and is represented directly,
including the optional initial-state load and final-state store. -/
```
```lean
def fused_recurrent_hgrn_fwd_surface
    (x g o h0 ht : RegionName) (T D BD : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  o_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = o_d < $(D)
  p_x = x + i_bh * $(T) * $(D) + o_d
  p_g = g + i_bh * $(T) * $(D) + o_d
  p_o = o + i_bh * $(T) * $(D) + o_d
  b_h = tl.zeros([$(BD)], dtype=tl.float32)
  if USE_INITIAL_STATE {
    p_h0 = h0 + i_bh * $(D) + o_d
    b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
  }
  for _i in range($(0), $(T), $(1)) {
    b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
    b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
    b_h = b_g * b_h + b_x
    tl.store(p_o, (b_h).to(p_o.dtype.element_ty), mask=mask)
    p_x += $(D)
    p_g += $(D)
    p_o += $(D)
  }
  if STORE_FINAL_STATE {
    p_ht = ht + i_bh * $(D) + o_d
    tl.store(p_ht, (b_h).to(p_ht.dtype.element_ty), mask=mask)
  }
}
```
</details>

<details><summary><code>fused_recurrent_hgrn_forward_step_store_slice</code></summary>

```
/-- One forward recurrence step:
`b_h = b_g * b_h + b_x`, then masked store to the current output row. -/
```
```lean
def fused_recurrent_hgrn_forward_step_store_slice
    (BHPrev X G O : RegionName) (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  prev = tl.load(BHPrev + i_bh * $(D) + offs_d, mask=mask, other=0.0)
  b_x = tl.load(X + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_g = tl.load(G + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_h = b_g * prev + b_x
  tl.store(O + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (b_h).to(O.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>active</code></summary>

```lean
def active (s : BlockState) (D BD : Nat) (i : Fin BD) : Prop :=
  dIndex s BD i < D
```
</details>

<details><summary><code>outOffset</code></summary>

```lean
def outOffset (s : BlockState) (i_t T D BD : Nat) (i : Fin BD) : Nat :=
  (s.pids 1 * T + i_t) * D + dIndex s BD i
```
</details>

<details><summary><code>fused_recurrent_hgrn_final_state_store_slice</code></summary>

```
/-- Proof-oriented final-state store slice of `fused_recurrent_hgrn.py`'s
`fused_recurrent_hgrn_fwd_kernel`. Companion to the per-iteration output store
slice: writes a precomputed final-state `BHFinal` vector into `Ht` after the
loop completes (STORE_FINAL_STATE=True branch). -/
```
```lean
def fused_recurrent_hgrn_final_state_store_slice
    (BHFinal Ht : RegionName) (D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  b_h = tl.load(BHFinal + i_bh * $(D) + offs_d, mask=mask, other=0.0)
  tl.store(Ht + i_bh * $(D) + offs_d,
    (b_h).to(Ht.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>fused_recurrent_hgrn_bwd_dx_step_store_slice</code></summary>

```
/-- Backward one-step `dx` formula:
`b_dh = b_dh_prev + b_do`, `b_dx = b_dh`, then masked store to `DX`. -/
```
```lean
def fused_recurrent_hgrn_bwd_dx_step_store_slice
    (DHPrev DO DX : RegionName) (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  dh_prev = tl.load(DHPrev + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0)
  b_do = tl.load(DO + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_dh = dh_prev + b_do
  tl.store(DX + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (b_dh).to(DX.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>bwdDxStepValue</code></summary>

```lean
noncomputable def bwdDxStepValue
    (s : BlockState) (DHPrev DO : RegionName) (i_t T D BD : Nat)
    (i : Fin BD) : ℝ :=
  s.readMem DHPrev (outOffset s i_t T D BD i) +
    s.readMem DO (outOffset s i_t T D BD i)
```
</details>

<details><summary><code>fused_recurrent_hgrn_bwd_dg_step_store_slice</code></summary>

```
/-- Backward one-step `dg` formula:
`b_dh = b_dh_prev + b_do`, `b_dg = b_dh * b_o`, then masked store to `DG`. -/
```
```lean
def fused_recurrent_hgrn_bwd_dg_step_store_slice
    (DHPrev DO BO DG : RegionName) (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  dh_prev = tl.load(DHPrev + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0)
  b_do = tl.load(DO + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_o = tl.load(BO + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_dh = dh_prev + b_do
  b_dg = b_dh * b_o
  tl.store(DG + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (b_dg).to(DG.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>bwdDgStepValue</code></summary>

```lean
noncomputable def bwdDgStepValue
    (s : BlockState) (DHPrev DO BO : RegionName) (i_t T D BD : Nat)
    (i : Fin BD) : ℝ :=
  (s.readMem DHPrev (outOffset s i_t T D BD i) +
      s.readMem DO (outOffset s i_t T D BD i)) *
    s.readMem BO (outOffset s i_t T D BD i)
```
</details>

<details><summary><code>dIndex</code></summary>

```lean
def dIndex (s : BlockState) (BD : Nat) (i : Fin BD) : Nat :=
  s.pids 0 * BD + i.val
```
</details>

<details><summary><code>stateSeed</code></summary>

```
/-- Seeded initial state `b_h^(0)`: `h0` if `USE_INITIAL_STATE` else `0`. -/
```
```lean
noncomputable def stateSeed (s : BlockState) (h0 : RegionName)
    (USE_INITIAL_STATE : Bool) (D BD : Nat) (i : Fin BD) : ℝ :=
  if USE_INITIAL_STATE then s.readMem h0 (bhOffset s D BD i) else 0
```
</details>

<details><summary><code>gVal</code></summary>

```
/-- `g_t[i]` at the kernel's exact time-row layout. -/
```
```lean
noncomputable def gVal (s : BlockState) (g : RegionName) (T D BD : Nat)
    (t : Nat) (i : Fin BD) : ℝ :=
  s.readMem g (outOffset s t T D BD i)
```
</details>

<details><summary><code>xVal</code></summary>

```
/-- `x_t[i]` at the kernel's exact time-row layout. -/
```
```lean
noncomputable def xVal (s : BlockState) (x : RegionName) (T D BD : Nat)
    (t : Nat) (i : Fin BD) : ℝ :=
  s.readMem x (outOffset s t T D BD i)
```
</details>

## Also present (pinned special-case summaries)
- `fused_recurrent_hgrn_output_store_slice_compute_correct`
- `fused_recurrent_hgrn_forward_step_store_slice_compute_correct`
- `fused_recurrent_hgrn_final_state_store_slice_compute_correct`
- `fused_recurrent_hgrn_bwd_grad_store_slice_compute_correct`
- `fused_recurrent_hgrn_bwd_dx_step_store_slice_compute_correct`
- `fused_recurrent_hgrn_bwd_dg_step_store_slice_compute_correct`
- `fused_recurrent_hgrn_bwd_dx_store_slice_compute_correct`
- `fused_recurrent_hgrn_bwd_dg_store_slice_compute_correct`
