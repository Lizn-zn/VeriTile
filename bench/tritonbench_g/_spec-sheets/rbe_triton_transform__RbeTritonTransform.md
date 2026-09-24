# Spec sheet — `bench/tritonbench_g/rbe_triton_transform/RbeTritonTransform.lean`

**Python source:** `bench/tritonbench_g/rbe_triton_transform/rbe_triton_transform.py`

## Public theorem: `rbe_triton_transform_output_summary_general`

<details><summary>docstring</summary>

```
/-- **Dimension-general** correctness summary for `rbe_triton_transform.py`'s
`rbe_triton`, against the **genuine rotary closed form** — a pure function of
INPUT memory and the exact `Real.cos` / `Real.sin` / `Real.rpow`, never a
read-back of the kernel's own output — for arbitrary `M`, `K`, all six
strides, `start_token_position`, `THETA`, `DIM`, `BLOCK_SIZE_M`,
`BLOCK_SIZE_K`, and arbitrary program ids (`s.pids 0` = batch,
`s.pids 1` = the fused `(m, k)` CTA id decomposed in-kernel). It packages:

* the full faithful surface (both `tl.debug_barrier()` calls, both masked
  loads/stores, the inlined `get_freq_multi_tokens`) lowers to the algorithm
  layer;
* the even-offset (`out_real`) store: every `out_real_mask`-active lane
  `(i, j)` holds `x_real·cos(freq) − x_imag·sin(freq)` where
  `freq = (start_token_position + offs_m[i]) / THETA^((offs_n[j] % DIM)/DIM)`
  (`freqSpec_eq_token`) and `x_imag` is the masked imaginary load (`0` on the
  `1 + offs_n[j] ≥ K` boundary lane);
* the odd-offset (`out_imag`) store: every `out_imag_mask`-active lane holds
  `x_real·sin(freq) + x_imag·cos(freq)`.

The final conjunct adds the **`⊨` (`GroupedMasked2DKernelIO.Implements`)** face
of the same two stores: the whole kernel as one grouped masked Hoare triple over
**flat** pointer memory (`nIn = 2` interleaved read channels `x_ptrs` /
`x_ptrs + 1`, `nOut = 2` interleaved write channels `out_ptrs` / `out_ptrs + 1`,
`B = BLOCK_SIZE_M · (BLOCK_SIZE_K / 2)` lanes decoded row-major by `Lane2D.decode`) —
termination, both stored values on their mask-active lanes, and a frame
asserting every cell outside the two output windows is untouched.

Honest side conditions: the even-offset output footprint is injective
(`hOutInj`) and even-offset cells never collide with odd-offset (`+ 1`) cells
(`hRI`) — both hold for the wrapper's contiguous row-major layout. The `⊨`
conjunct quantifies over every launch state, so it needs those same two
conditions at **every** program id (`hOutInjAll` / `hRIAll`, the `∀ s` forms);
`outOff` depends on the state only through `s.pids 0` / `s.pids 1`, so this is
exactly "the layout is collision-free for every CTA", and it is required for
truth — without it the two interleaved scatters alias and no closed form
holds. -/
```
</details>

**Statement:**
```lean
specification rbe_triton_transform_output_summary_general
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx))
    (hRI : ∀ idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1)
    (hOutInjAll : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx))
    (hRIAll : ∀ (s : BlockState)
        (idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]),
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1) :
    -- (1) the full faithful surface lowers to the algorithm layer
    (∃ alg, (rbe_triton_surface x_ptr out_ptr M K stride_x_batch stride_x_m
      stride_x_n stride_out_batch stride_out_m stride_out_n
      start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K).toAlgorithm?
        = Except.ok alg) ∧
    -- (2) even offsets: genuine `x_real·cos − x_imag·sin`
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rbe_triton_surface x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
          activeReal s M K BLOCK_SIZE_M BLOCK_SIZE_K idx)
        (fun idx => (out_ptr,
          outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K idx)))
      (expected := fun idx =>
        rbeOutRealSpec s x_ptr K stride_x_batch stride_x_m stride_x_n
          start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA idx) ∧
    -- (3) odd offsets: genuine `x_real·sin + x_imag·cos`
    ComputeCorrect.Realizes_without_Rounding
      (kernel := rbe_triton_surface x_ptr out_ptr M K stride_x_batch
        stride_x_m stride_x_n stride_out_batch stride_out_m stride_out_n
        start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
          activeImag s M K BLOCK_SIZE_M BLOCK_SIZE_K idx)
        (fun idx => (out_ptr,
          outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K idx + 1)))
      (expected := fun idx =>
        rbeOutImagSpec s x_ptr K stride_x_batch stride_x_m stride_x_n
          start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K THETA idx) ∧
    -- (4) the flat-memory `⊨` face of both interleaved stores
    (rbeTritonIO x_ptr out_ptr M K stride_x_batch stride_x_m stride_x_n
        stride_out_batch stride_out_m stride_out_n start_token_position THETA
        DIM BLOCK_SIZE_M BLOCK_SIZE_K
      ⊨ fun _pid₀ pid₁ xs o j =>
          let xr := xs (⟨0, by decide⟩ : Fin 2) j
          let xi := xs (⟨1, by decide⟩ : Fin 2) j
          let f := rbeFreqP pid₁ K start_token_position DIM BLOCK_SIZE_M
            BLOCK_SIZE_K THETA j
          match o with
          | ⟨0, _⟩ =>
              xr * Real.cos f -
                (if 1 + rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K then xi
                  else 0) * Real.sin f
          | ⟨_ + 1, _⟩ => xr * Real.sin f + xi * Real.cos f)
```

**Assumptions / layout contracts:**
- `hOutInj : Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx)`
- `hRI : ∀ idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2],
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1`
- `hOutInjAll : ∀ s : BlockState, Function.Injective
      (fun idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2] =>
        outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx)`
- `hRIAll : ∀ (s : BlockState)
        (idx k : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]),
      outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx
        ≠ outOff s K stride_out_batch stride_out_m stride_out_n BLOCK_SIZE_M
            BLOCK_SIZE_K k + 1`

**Closed-form spec defs (transitive):** `outOff`, `rbe_triton_surface`, `activeReal`, `rbeOutRealSpec`, `activeImag`, `rbeOutImagSpec`, `rbeTritonIO`, `rbeFreqP`, `rbeColP`, `rowIdx`, `colIdx`, `xOff`, `freqSpec`, `rbeXAddrP`, `rbeRowP`, `rbeOutAddrP`, `kCdiv`, `pidM`, `pidN`

<details><summary><code>outOff</code></summary>

```
/-- Output address of the real part written by lane `(i, j)` (the imaginary
part is stored at `+ 1`). -/
```
```lean
def outOff (s : BlockState)
    (K stride_out_batch stride_out_m stride_out_n
      BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : Nat :=
  s.pids 0 * stride_out_batch +
    stride_out_m * rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K idx.1 +
    stride_out_n * colIdx s K BLOCK_SIZE_K idx.2.1
```
</details>

<details><summary><code>rbe_triton_surface</code></summary>

```
/-- Faithful transcription of `rbe_triton_transform.py`'s `rbe_triton`, with
`get_freq_multi_tokens` inlined at its single call site (see the
Translation-surface blocker preamble for the inlining conventions). -/
```
```lean
def rbe_triton_surface
    (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    ComputeKernel := triton {
  pid_batch = tl.program_id(axis=0)
  pid = tl.program_id(axis=1)
  pid_m = pid // tl.cdiv($(K), $(BLOCK_SIZE_K))
  pid_n = pid % tl.cdiv($(K), $(BLOCK_SIZE_K))

  offs_m = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_n = pid_n * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K) // $(2)) * $(2)
  x_ptrs = x_ptr + (pid_batch * $(stride_x_batch) + $(stride_x_m) * offs_m[:, None] +
    $(stride_x_n) * offs_n[None, :])
  x_real_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(K))
  real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
  x_imag_mask = (offs_m[:, None] < $(M)) & ($((1 : Nat)) + offs_n[None, :] < $(K))
  imag = tl.load(x_ptrs + $(1), mask=x_imag_mask, other=0.0)
  tl.debug_barrier()
  start_block = $((start_token_position : Nat)) + pid_m * $(BLOCK_SIZE_M)
  freqs = offs_n % $(DIM)
  freqs_f = tl.toReal(freqs) / $(DIM)
  freqs_p = tl.extra.cuda.libdevice.pow($((THETA : ℝ)), freqs_f)
  tks = tl.arange(0, $(BLOCK_SIZE_M)) + start_block
  tks_f = tl.toReal(tks)
  freqs_mn = tks_f[:, None] / freqs_p[None, :]
  cos = tl.cos(freqs_mn)
  sin = tl.sin(freqs_mn)

  out_real = real * cos - imag * sin
  out_imag = real * sin + imag * cos
  tl.debug_barrier()
  out_ptrs = out_ptr + (pid_batch * $(stride_out_batch) + $(stride_out_m) * offs_m[:, None] +
    $(stride_out_n) * offs_n[None, :])
  out_real_mask = (offs_m[:, None] < $(M)) & (offs_n[None, :] < $(K))
  tl.store(out_ptrs, out_real, mask=out_real_mask)
  out_imag_mask = (offs_m[:, None] < $(M)) & ($((1 : Nat)) + offs_n[None, :] < $(K))
  tl.store(out_ptrs + $(1), out_imag, mask=out_imag_mask)
}
```
</details>

<details><summary><code>activeReal</code></summary>

```
/-- `x_real_mask` / `out_real_mask` of lane `(i, j)`:
`offs_m[i] < M ∧ offs_n[j] < K`. -/
```
```lean
def activeReal (s : BlockState) (M K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : Prop :=
  rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K idx.1 < M ∧
    colIdx s K BLOCK_SIZE_K idx.2.1 < K
```
</details>

<details><summary><code>rbeOutRealSpec</code></summary>

```
/-- Genuine spec of the even-offset (`out_real`) store on an active lane:
`x_real·cos(freq) − x_imag·sin(freq)`, where `x_imag` is the masked imaginary
load (`0` on the `1 + offs_n[j] ≥ K` boundary lane, matching `x_imag_mask`'s
`other=0.0`). -/
```
```lean
noncomputable def rbeOutRealSpec (s : BlockState) (x_ptr : RegionName)
    (K stride_x_batch stride_x_m stride_x_n
      start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (THETA : ℝ)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : ℝ :=
  s.readMem x_ptr
      (xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M
        BLOCK_SIZE_K idx) *
    Real.cos (freqSpec s K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K
      THETA idx) -
  (if 1 + colIdx s K BLOCK_SIZE_K idx.2.1 < K then
      s.readMem x_ptr
        (xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M
          BLOCK_SIZE_K idx + 1)
    else 0) *
    Real.sin (freqSpec s K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K
      THETA idx)
```
</details>

<details><summary><code>activeImag</code></summary>

```
/-- `x_imag_mask` / `out_imag_mask` of lane `(i, j)`:
`offs_m[i] < M ∧ 1 + offs_n[j] < K`. -/
```
```lean
def activeImag (s : BlockState) (M K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : Prop :=
  rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K idx.1 < M ∧
    1 + colIdx s K BLOCK_SIZE_K idx.2.1 < K
```
</details>

<details><summary><code>rbeOutImagSpec</code></summary>

```
/-- Genuine spec of the odd-offset (`out_imag`) store on an active lane:
`x_real·sin(freq) + x_imag·cos(freq)` (on an `x_imag_mask`-active lane the
real-part load is also in bounds, so both reads are genuine). -/
```
```lean
noncomputable def rbeOutImagSpec (s : BlockState) (x_ptr : RegionName)
    (K stride_x_batch stride_x_m stride_x_n
      start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (THETA : ℝ)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : ℝ :=
  s.readMem x_ptr
      (xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M
        BLOCK_SIZE_K idx) *
    Real.sin (freqSpec s K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K
      THETA idx) +
  s.readMem x_ptr
      (xOff s K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M
        BLOCK_SIZE_K idx + 1) *
    Real.cos (freqSpec s K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K
      THETA idx)
```
</details>

<details><summary><code>rbeTritonIO</code></summary>

```
/-- The **IO signature** of `rbe_triton` — the whole kernel-specific audit
surface of the `⊨` headline.

* `bufs = [x_ptr, out_ptr]`, `nIn = 2`, `nOut = 2`,
  `B = BLOCK_SIZE_M · (BLOCK_SIZE_K / 2)` (the kernel's 2D tile flattened
  row-major by `Lane2D.decode`).
* read channel `0` — `x_ptr` at the even address `x_ptrs`, gated by
  `x_real_mask` (`offs_m < M ∧ offs_n < K`);
  read channel `1` — `x_ptr` at the interleaved odd address `x_ptrs + 1`,
  gated by `x_imag_mask` (`offs_m < M ∧ 1 + offs_n < K`).
* write channel `0` — `out_ptr` at `out_ptrs`, gated by `out_real_mask`;
  write channel `1` — `out_ptr` at `out_ptrs + 1`, gated by `out_imag_mask`.

`cos`/`sin` are *not* read channels: the kernel derives them from index
arithmetic (`get_freq_multi_tokens`, inlined), so they appear in the spec
value as the exact `Real.cos`/`Real.sin` of `rbeFreqP`, not as pinned inputs. -/
```
```lean
def rbeTritonIO (x_ptr out_ptr : RegionName)
    (M K stride_x_batch stride_x_m stride_x_n
      stride_out_batch stride_out_m stride_out_n
      start_token_position : Nat)
    (THETA : ℝ) (DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) :
    GroupedMasked2DKernelIO where
  kernel := rbe_triton_surface x_ptr out_ptr M K stride_x_batch stride_x_m
    stride_x_n stride_out_batch stride_out_m stride_out_n
    start_token_position THETA DIM BLOCK_SIZE_M BLOCK_SIZE_K
  nIn := 2
  nOut := 2
  bufs := [x_ptr, out_ptr]
  inp := fun _ => x_ptr
  out := fun _ => out_ptr
  B := BLOCK_SIZE_M * (BLOCK_SIZE_K / 2)
  read := fun i pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => rbeXAddrP pid₀ pid₁ K stride_x_batch stride_x_m stride_x_n
        BLOCK_SIZE_M BLOCK_SIZE_K j
    | ⟨_ + 1, _⟩ => rbeXAddrP pid₀ pid₁ K stride_x_batch stride_x_m stride_x_n
        BLOCK_SIZE_M BLOCK_SIZE_K j + 1
  readMask := fun i _pid₀ pid₁ j => match i with
    | ⟨0, _⟩ => rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < M ∧
        rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K
    | ⟨_ + 1, _⟩ => rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < M ∧
        1 + rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K
  write := fun o pid₀ pid₁ j => match o with
    | ⟨0, _⟩ => rbeOutAddrP pid₀ pid₁ K stride_out_batch stride_out_m
        stride_out_n BLOCK_SIZE_M BLOCK_SIZE_K j
    | ⟨_ + 1, _⟩ => rbeOutAddrP pid₀ pid₁ K stride_out_batch stride_out_m
        stride_out_n BLOCK_SIZE_M BLOCK_SIZE_K j + 1
  writeMask := fun o _pid₀ pid₁ j => match o with
    | ⟨0, _⟩ => rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < M ∧
        rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K
    | ⟨_ + 1, _⟩ => rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < M ∧
        1 + rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j < K
```
</details>

<details><summary><code>rbeFreqP</code></summary>

```
/-- Rotation angle of flat lane `j`. -/
```
```lean
noncomputable def rbeFreqP (pid₁ K start_token_position DIM
    BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (THETA : ℝ)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) : ℝ :=
  (((Lane2D.decode j).1.val +
      (start_token_position + pid₁ / kCdiv K BLOCK_SIZE_K * BLOCK_SIZE_M)
        : ℕ) : ℝ) /
    Real.rpow THETA
      (((rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j % DIM : ℕ) : ℝ) /
        ((DIM : ℕ) : ℝ))
```
</details>

<details><summary><code>rbeColP</code></summary>

```
/-- Global (even) column `offs_n[j]` covered by flat lane `j`. -/
```
```lean
def rbeColP (pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) : Nat :=
  pid₁ % kCdiv K BLOCK_SIZE_K * BLOCK_SIZE_K +
    (Lane2D.decode j).2.1.val * 2
```
</details>

<details><summary><code>rowIdx</code></summary>

```
/-- Global row `offs_m[i] = pid_m·BLOCK_SIZE_M + i` covered by tile lane `i`. -/
```
```lean
def rowIdx (s : BlockState) (K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (i : Fin BLOCK_SIZE_M) : Nat :=
  pidM s K BLOCK_SIZE_K * BLOCK_SIZE_M + i.val
```
</details>

<details><summary><code>colIdx</code></summary>

```
/-- Global (even) column `offs_n[j] = pid_n·BLOCK_SIZE_K + 2j` covered by tile
lane `j`. -/
```
```lean
def colIdx (s : BlockState) (K BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_K / 2)) : Nat :=
  pidN s K BLOCK_SIZE_K * BLOCK_SIZE_K + j.val * 2
```
</details>

<details><summary><code>xOff</code></summary>

```
/-- Input address of the real part read by lane `(i, j)` (the imaginary part
sits at `+ 1`). -/
```
```lean
def xOff (s : BlockState)
    (K stride_x_batch stride_x_m stride_x_n BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : Nat :=
  s.pids 0 * stride_x_batch +
    stride_x_m * rowIdx s K BLOCK_SIZE_M BLOCK_SIZE_K idx.1 +
    stride_x_n * colIdx s K BLOCK_SIZE_K idx.2.1
```
</details>

<details><summary><code>freqSpec</code></summary>

```
/-- Rotation angle of lane `(i, j)`:
`(i + (start_token_position + pid_m·BLOCK_SIZE_M)) / THETA^((offs_n[j] % DIM)/DIM)`
— the numerator is exactly `start_token_position + offs_m[i]`
(see `freqSpec_eq_token`). -/
```
```lean
noncomputable def freqSpec (s : BlockState)
    (K start_token_position DIM BLOCK_SIZE_M BLOCK_SIZE_K : Nat) (THETA : ℝ)
    (idx : TileIndex [BLOCK_SIZE_M, BLOCK_SIZE_K / 2]) : ℝ :=
  ((idx.1.val + (start_token_position + pidM s K BLOCK_SIZE_K * BLOCK_SIZE_M)
      : ℕ) : ℝ) /
    Real.rpow THETA
      (((colIdx s K BLOCK_SIZE_K idx.2.1 % DIM : ℕ) : ℝ) / ((DIM : ℕ) : ℝ))
```
</details>

<details><summary><code>rbeXAddrP</code></summary>

```
/-- Input address of the real part read by flat lane `j`. -/
```
```lean
def rbeXAddrP (pid₀ pid₁ K stride_x_batch stride_x_m stride_x_n
    BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) : Nat :=
  pid₀ * stride_x_batch +
    stride_x_m * rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j +
    stride_x_n * rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j
```
</details>

<details><summary><code>rbeRowP</code></summary>

```
/-- Global row `offs_m[i]` covered by flat lane `j`. -/
```
```lean
def rbeRowP (pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) : Nat :=
  pid₁ / kCdiv K BLOCK_SIZE_K * BLOCK_SIZE_M +
    (Lane2D.decode j).1.val
```
</details>

<details><summary><code>rbeOutAddrP</code></summary>

```
/-- Output address of the real part written by flat lane `j`. -/
```
```lean
def rbeOutAddrP (pid₀ pid₁ K stride_out_batch stride_out_m stride_out_n
    BLOCK_SIZE_M BLOCK_SIZE_K : Nat)
    (j : Fin (BLOCK_SIZE_M * (BLOCK_SIZE_K / 2))) : Nat :=
  pid₀ * stride_out_batch +
    stride_out_m * rbeRowP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j +
    stride_out_n * rbeColP pid₁ K BLOCK_SIZE_M BLOCK_SIZE_K j
```
</details>

<details><summary><code>kCdiv</code></summary>

```
/-- `tl.cdiv(K, BLOCK_SIZE_K)` at the algorithm layer. -/
```
```lean
def kCdiv (K BLOCK_SIZE_K : Nat) : Nat := (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K
```
</details>

<details><summary><code>pidM</code></summary>

```
/-- `pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)` of this program. -/
```
```lean
def pidM (s : BlockState) (K BLOCK_SIZE_K : Nat) : Nat :=
  s.pids 1 / kCdiv K BLOCK_SIZE_K
```
</details>

<details><summary><code>pidN</code></summary>

```
/-- `pid_n = pid % tl.cdiv(K, BLOCK_SIZE_K)` of this program. -/
```
```lean
def pidN (s : BlockState) (K BLOCK_SIZE_K : Nat) : Nat :=
  s.pids 1 % kCdiv K BLOCK_SIZE_K
```
</details>

## Also present (pinned special-case summaries)
- `rbe_real_compute_correct`
- `rbe_imag_compute_correct`
