# Spec sheet — `bench/tritonbench_g/fused_recurrent_hgrn/FusedRecurrentHgrn.lean`

**Python source:** `bench/tritonbench_g/fused_recurrent_hgrn/fused_recurrent_hgrn.py`

## Public theorem: `fused_recurrent_hgrn_output_summary_general`

<details><summary>docstring</summary>

```
/-! ### ════════ ★ MAIN THEOREM ★ ════════

**SCOPE — this is a claim about four hand-cut slices, not about the launched
kernels.** Clauses 2–6 are `Realizes` facts about
`fused_recurrent_hgrn_forward_step_store_slice`,
`fused_recurrent_hgrn_bwd_dx_step_store_slice`,
`fused_recurrent_hgrn_bwd_dg_step_store_slice` and
`fused_recurrent_hgrn_bwd_dx_two_step_store_slice`; the launched forward surface
appears only in clause 1, which says nothing more than "it lowers to the
algorithm layer", and the launched backward surface does not appear at all. The
`STORE_FINAL_STATE` writeback has **no** correctness face here (see the module
docstring).

Parameterized over the symbolic time/feature/tile sizes `T D BD`, the step index
`i_t`, and both flags `USE_INITIAL_STATE STORE_FINAL_STATE`. The forward face is
realized against the closed form `hgrnStateClosed`, the backward `dx` scan face
against `hgrnBwdDx`, both over the *input* regions (never a read-back of the
kernel's own output):

1. the full HGRN forward surface lowers to the algorithm layer;
2. one forward **output** body realizes `hgrnStateClosed(i_t + 1)` — the unrolled
   recurrence `b_h = g·b_h + x` — given the *assumed* carry invariant
   `BHPrev = hgrnStateClosed(i_t)`;
3. one backward `dx` body realizes the genuine `bwdDxStepValue` (`dh_prev + do`);
4. one backward `dg` body realizes the genuine `bwdDgStepValue`
   `(dh_prev + do) · b_o`, with `b_o` Python's **three-way previous-row branch**
   (`bwdPrevOut`: output row `i_t − 1`, or `h0` at `i_t = 0` under
   `USE_INITIAL_STATE`, else `0`) transcribed as a DSL `if`;
5. **the backward scan step, executed.** Two consecutive reverse iterations —
   iteration `i_t + 1`'s `b_dh = (b_dh + b_do) · b_g` (py:115+118) then iteration
   `i_t`'s `b_dh + b_do` and `dx` store — realize the genuine closed form
   `hgrnBwdDx(i_t) = Σ_{i_t ≤ t < T} do_t · ∏_{i_t < j ≤ t} g_j` over the input
   regions `do`, `g`, given the *assumed* carry entering step `i_t + 1`
   (antecedents local to the clause: `i_t + 1 < T` and that carry);
6. the same two-step body at `i_t = T − 2`, where the assumed carry is only the
   `tl.zeros` **seed** (`DHPrev` row `T − 1` reads `0`, py:104) — so the `dx` row
   `do_{T-2} + do_{T-1}·g_{T-1}` is closed over `do`, `g` with no carry-value
   fiction left (antecedents local to the clause: `2 ≤ T` and the zero seed).

Honest structural side condition: `0 < BD` (contiguous lanes, giving offset
injectivity for every face). The flags flow through verbatim; clauses 2–6 hold
for every flag setting.

**The carry invariants are assumptions, not conclusions.** Nothing in this file
propagates the forward one: clause 2 writes `O` at the time-indexed
`outOffset s i_t`, whereas `hPrev` constrains `BHPrev` at the time-free
`bhOffset` — a different region at a different offset, with no bridging lemma;
likewise nothing proves the base case `hgrnStateClosed(0) = seed`. On the
backward side the carry *values* are now pinned to the closed form
`hgrnBwdCarry` and the fold `hgrnBwdCarry_pred`/`hgrnBwdCarry_init` is proved,
but the region `DHPrev` still only *holds* that value by hypothesis (clause 5) —
except at the loop's first fold, where the hypothesis degenerates to the literal
`tl.zeros` seed (clause 6). Chaining clause 5 across all `T` steps is the
unmodelled reverse fold. -/
```
</details>

**Statement:**
```lean
specification fused_recurrent_hgrn_output_summary_general
    (X G O H0 Ht DX DG DO BHPrev DHPrev : RegionName)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
    (i_t T D BD : Nat) (s : BlockState) (hBD : 0 < BD)
    (hPrev : ∀ i : Fin BD,
      s.readMem BHPrev (bhOffset s D BD i)
        = hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD i_t i) :
    -- (1) the full forward surface lowers to the algorithm layer
    (∃ alg, (fused_recurrent_hgrn_fwd_surface X G O H0 Ht T D BD
      USE_INITIAL_STATE STORE_FINAL_STATE).toAlgorithm? = Except.ok alg) ∧
    -- (2) the forward output body realizes the genuine `hgrnStateClosed(i_t+1)`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_hgrn_forward_step_store_slice BHPrev X G O
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (O, outOffset s i_t T D BD i)))
      (expected := fun i =>
        hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD (i_t + 1) i)) ∧
    -- (3) the backward `dx` body realizes the genuine `dh_prev + do`
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX
        i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (DX, outOffset s i_t T D BD i)))
      (expected := fun i => bwdDxStepValue s DHPrev DO i_t T D BD i)) ∧
    -- (4) the backward `dg` body realizes the genuine `(dh_prev+do)·b_o`, with
    --     Python's three-way previous-row `b_o` branch modeled
    (ComputeCorrect.Realizes_without_Rounding
      (kernel := fused_recurrent_hgrn_bwd_dg_step_store_slice DHPrev DO O H0 DG
        USE_INITIAL_STATE i_t T D BD)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s D BD)
        (fun i => (DG, outOffset s i_t T D BD i)))
      (expected := fun i =>
        bwdDgStepValue s DHPrev DO O H0 USE_INITIAL_STATE i_t T D BD i)) ∧
    -- (5) the backward scan step `b_dh = b_dh * b_g` executed: the two-step body
    --     realizes the genuine closed form `hgrnBwdDx(i_t)` over `do`, `g`
    (i_t + 1 < T →
      (∀ i : Fin BD,
        s.readMem DHPrev (outOffset s (i_t + 1) T D BD i)
          = hgrnBwdCarry s DO G T D BD (i_t + 1) i) →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := fused_recurrent_hgrn_bwd_dx_two_step_store_slice DHPrev DO G DX
          i_t T D BD)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (active s D BD)
          (fun i => (DX, outOffset s i_t T D BD i)))
        (expected := fun i => hgrnBwdDx s DO G T D BD i_t i)) ∧
    -- (6) the same body at the loop's first fold, where the assumed carry is
    --     only the `tl.zeros` seed
    (2 ≤ T →
      (∀ i : Fin BD, s.readMem DHPrev (outOffset s (T - 1) T D BD i) = 0) →
      ComputeCorrect.Realizes_without_Rounding
        (kernel := fused_recurrent_hgrn_bwd_dx_two_step_store_slice DHPrev DO G DX
          (T - 2) T D BD)
        (initialState := s)
        (write := ComputeCorrect.WriteMap.writeIf
          (active s D BD)
          (fun i => (DX, outOffset s (T - 2) T D BD i)))
        (expected := fun i => hgrnBwdDx s DO G T D BD (T - 2) i))
```

**Assumptions / layout contracts:**
- `hBD : 0 < BD`
- `hPrev : ∀ i : Fin BD,
      s.readMem BHPrev (bhOffset s D BD i)
        = hgrnStateClosed s X G H0 USE_INITIAL_STATE T D BD i_t i`

**Closed-form spec defs (transitive):** `bhOffset`, `hgrnStateClosed`, `fused_recurrent_hgrn_fwd_surface`, `fused_recurrent_hgrn_forward_step_store_slice`, `active`, `outOffset`, `fused_recurrent_hgrn_bwd_dx_step_store_slice`, `bwdDxStepValue`, `fused_recurrent_hgrn_bwd_dg_step_store_slice`, `bwdDgStepValue`, `hgrnBwdCarry`, `fused_recurrent_hgrn_bwd_dx_two_step_store_slice`, `hgrnBwdDx`, `dIndex`, `stateSeed`, `gVal`, `xVal`, `bwdPrevOut`, `doVal`

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
/-- Backward one-step `dg` formula: `b_dh = b_dh_prev + b_do`,
`b_dg = b_dh * b_o`, then masked store to `DG`.

`b_o` is Python's three-way branch (py:99, py:108–113): the pointer `p_o` starts
at row `T − 2` and is decremented once per iteration, so at reverse step `i_t` it
addresses the **previous** output row `i_t − 1`; at `i_t = 0` the branch reads
`h0` instead when `USE_INITIAL_STATE`, else uses `tl.zeros`. The branch is
transcribed verbatim as a DSL `if`; because the slice fixes the reverse step
index `i_t`, its condition `i_t > 0` is decided by that index (each concrete
iteration takes exactly one arm), and the row `i_t - 1` is the ℕ-truncated
Python row, reached only under `0 < i_t`. -/
```
```lean
def fused_recurrent_hgrn_bwd_dg_step_store_slice
    (DHPrev DO O H0 DG : RegionName) (USE_INITIAL_STATE : Bool)
    (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  dh_prev = tl.load(DHPrev + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0)
  b_do = tl.load(DO + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  if $(i_t) > $(0) {
    b_o = tl.load(O + (i_bh * $(T) + $(i_t - 1)) * $(D) + offs_d,
      mask=mask, other=0.0).to(tl.float32)
  } else {
    if USE_INITIAL_STATE {
      b_o = tl.load(H0 + i_bh * $(D) + offs_d, mask=mask, other=0.0).to(tl.float32)
    } else {
      b_o = tl.zeros([$(BD)], dtype=tl.float32)
    }
  }
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
    (s : BlockState) (DHPrev DO O H0 : RegionName) (USE_INITIAL_STATE : Bool)
    (i_t T D BD : Nat) (i : Fin BD) : ℝ :=
  (s.readMem DHPrev (outOffset s i_t T D BD i) +
      s.readMem DO (outOffset s i_t T D BD i)) *
    bwdPrevOut s O H0 USE_INITIAL_STATE i_t T D BD i
```
</details>

<details><summary><code>hgrnBwdCarry</code></summary>

```
/-- **Genuine closed form of the backward carry entering reverse step `i_t`**:
`Σ_{i_t < t < T} do_t · ∏_{i_t < j ≤ t} g_j`. A specification over the input
regions `do`, `g` only. -/
```
```lean
noncomputable def hgrnBwdCarry (s : BlockState) (DO G : RegionName)
    (T D BD : Nat) (i_t : Nat) (i : Fin BD) : ℝ :=
  ∑ t ∈ Finset.Ico (i_t + 1) T,
    doVal s DO T D BD t i *
      (∏ j ∈ Finset.Ico (i_t + 1) (t + 1), gVal s G T D BD j i)
```
</details>

<details><summary><code>fused_recurrent_hgrn_bwd_dx_two_step_store_slice</code></summary>

```
/-! ## The backward scan step as an executed slice

The single-step `dx` slice above takes the incoming carry from the fiction
region `DHPrev` and never touches the gate region, so `b_dh = b_dh * b_g`
(py:118) is invisible to it. The **two-step** slice below closes that hole: it
transcribes reverse iteration `i_t + 1`'s carry fold
(`b_dh = b_dh + b_do`, then `b_dh = b_dh * b_g`, both at row `i_t + 1`) followed
by iteration `i_t`'s `b_dh = b_dh + b_do` and `dx` store, so the scan
multiplication is *executed* and the gate region `G` enters the compute face.
Iteration `i_t + 1`'s own `dx`/`dg` stores are omitted from the slice: they
write `DX`/`DG`, which the loop never reads back, so their omission cannot
change the `dx` row this face is about. -/
```
```lean
def fused_recurrent_hgrn_bwd_dx_two_step_store_slice
    (DHPrev DO G DX : RegionName) (i_t T D BD : Nat) :
    ComputeKernel := triton {
  i_d = tl.program_id(0)
  i_bh = tl.program_id(1)
  offs_d = i_d * $(BD) + tl.arange(0, $(BD))
  mask = offs_d < $(D)
  dh_prev = tl.load(DHPrev + (i_bh * $(T) + $(i_t + 1)) * $(D) + offs_d,
    mask=mask, other=0.0)
  b_do_next = tl.load(DO + (i_bh * $(T) + $(i_t + 1)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_g_next = tl.load(G + (i_bh * $(T) + $(i_t + 1)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_dh = dh_prev + b_do_next
  b_dh = b_dh * b_g_next
  b_do = tl.load(DO + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    mask=mask, other=0.0).to(tl.float32)
  b_dh = b_dh + b_do
  tl.store(DX + (i_bh * $(T) + $(i_t)) * $(D) + offs_d,
    (b_dh).to(DX.dtype.element_ty), mask=mask)
}
```
</details>

<details><summary><code>hgrnBwdDx</code></summary>

```
/-- **Genuine closed form of the `dx` row stored at reverse step `i_t`**:
`Σ_{i_t ≤ t < T} do_t · ∏_{i_t < j ≤ t} g_j` — the carry plus this step's
`do`. -/
```
```lean
noncomputable def hgrnBwdDx (s : BlockState) (DO G : RegionName)
    (T D BD : Nat) (i_t : Nat) (i : Fin BD) : ℝ :=
  ∑ t ∈ Finset.Ico i_t T,
    doVal s DO T D BD t i *
      (∏ j ∈ Finset.Ico (i_t + 1) (t + 1), gVal s G T D BD j i)
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

<details><summary><code>bwdPrevOut</code></summary>

```
/-- `b_o` at reverse step `i_t` — Python's three-way branch (py:108–113): the
**previous** output row `i_t − 1` when `i_t > 0`, else `h0` when
`USE_INITIAL_STATE`, else `0`. -/
```
```lean
noncomputable def bwdPrevOut (s : BlockState) (O H0 : RegionName)
    (USE_INITIAL_STATE : Bool) (i_t T D BD : Nat) (i : Fin BD) : ℝ :=
  if 0 < i_t then s.readMem O (outOffset s (i_t - 1) T D BD i)
  else if USE_INITIAL_STATE then s.readMem H0 (bhOffset s D BD i) else 0
```
</details>

<details><summary><code>doVal</code></summary>

```
/-- `do_t[d]` at the kernel's exact time-row layout. -/
```
```lean
noncomputable def doVal (s : BlockState) (DO : RegionName) (T D BD : Nat)
    (t : Nat) (i : Fin BD) : ℝ :=
  s.readMem DO (outOffset s t T D BD i)
```
</details>

## Public theorem: `fused_recurrent_hgrn_bwd_dx_io_correctness`

<details><summary>docstring</summary>

```
/-- **The headline on the IO surface** for `fused_recurrent_hgrn.py`'s backward `dx`
step store: for every disjoint flat placement of the three buffers, every program
coordinate whose active lanes are in bounds, and every launch state whose two source
rows hold `xs` / `ys` at the active lanes, the translated pointer kernel terminates,
every active lane of `DX` holds `xs i + ys i`, and every other memory cell is
unchanged.

Dimension-general in `i_t`, `T`, `D` and `BD`, with **no** side-condition: the row
window is `base + lane`, so the readback's output-injectivity precondition is
discharged by `outOffset_inj` rather than assumed. -/
```
</details>

**Statement:**
```lean
specification fused_recurrent_hgrn_bwd_dx_io_correctness
    (DHPrev DO DX : RegionName) (i_t T D BD : Nat) :
    bwdDxIO DHPrev DO DX i_t T D BD
      ⊨ fun _p₀ _p₁ xs ys o => xs o + ys o
```

**Closed-form spec defs (transitive):** `bwdDxIO`, `fused_recurrent_hgrn_bwd_dx_step_store_slice`

<details><summary><code>bwdDxIO</code></summary>

```
/-- IO signature of the backward `dx` step store: two `[BD]` reads at the row window,
one write at the same window. -/
```
```lean
def bwdDxIO (DHPrev DO DX : RegionName) (i_t T D BD : Nat) :
    MaskedTileShapedKernelIO₂ where
  kernel := fused_recurrent_hgrn_bwd_dx_step_store_slice DHPrev DO DX i_t T D BD
  in1 := DHPrev
  in2 := DO
  out := DX
  shape1 := [BD]
  shape2 := [BD]
  shapeOut := [BD]
  read1 := fun p₀ p₁ i => (p₁ * T + i_t) * D + (p₀ * BD + i.1.val)
  read2 := fun p₀ p₁ i => (p₁ * T + i_t) * D + (p₀ * BD + i.1.val)
  write := fun p₀ p₁ o => (p₁ * T + i_t) * D + (p₀ * BD + o.1.val)
  mask1 := fun p₀ _p₁ i => p₀ * BD + i.1.val < D
  mask2 := fun p₀ _p₁ i => p₀ * BD + i.1.val < D
  writeMask := fun p₀ _p₁ o => p₀ * BD + o.1.val < D
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

## Also present (pinned special-case summaries)
- `fused_recurrent_hgrn_forward_step_store_slice_compute_correct`
- `fused_recurrent_hgrn_bwd_dx_step_store_slice_compute_correct`
- `fused_recurrent_hgrn_bwd_dx_two_step_store_slice_compute_correct`
- `fused_recurrent_hgrn_bwd_dg_step_store_slice_compute_correct`
