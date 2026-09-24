/-
Kernel IO contracts: StreamEmit.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton

/-! ### The per-step emit skins: `StreamEmit*` (streaming genre, style S3)

The second streaming shape: kernels whose store sits **inside** the loop
body — at step `t` the program stores a `C`-lane tile to a `t`-dependent
window, so the output is a `T × C` family of cells rather than one terminal
tile. `Emit` names that capability (a per-step write-window family); the
subscript stays pure data-input × output arity. Everything else is the
`Stream*` genre unchanged: curried per-step streams (never flattened to
`Fin (T*B)`), the single `⊨[R]` surface with the `outDType := .real`
default, and `hrun` as the consumer's `forRange`-invariant obligation.

One skin covers the genre's three value shapes, because the spec `f` eats
the *whole* curried streams and is indexed by the step: an **emit/copy**
kernel's `f t j` uses only the step-`t` tiles, a **scan** kernel's `f t j`
is a prefix fold `(u ≤ t)`, and a **two-pass** kernel's `f t j` combines
the step-`t` tile with a fold over the entire stream (an rmsnorm consumer
writes `xs t j * (√((∑ u, ∑ e, xs u e ^ 2)/N + ε))⁻¹ * ws t j`). The skin
does not see the difference — the loop structure that realizes `f` lives
entirely in the consumer's invariant argument, whose canonical shape is
"windows before the loop counter hold their final spec value, windows at
or past it hold the original memory". -/

/-- IO signature of the **two-stream per-step emit** shape (streaming
genre, style S3: in-loop store): a 2-D pid grid, **two streamed float
input channels** (`inp1`/`inp2`) read in `T` loop steps of `B1`/`B2` lanes
each, and **one output channel** (`out`) written as a **per-step `C`-lane
window family** — step `t` stores `C` lanes at the `t`-indexed window
`write · · t ·`. The spec `f` eats the whole curried streams and returns
the value of output cell `(t, j)` directly; emit, scan and two-pass
consumers differ only in how their `f` uses the streams (see the genre
note). The skin does not prove the loop: the `ImplementsR.intro`
obligation `hrun` is discharged with a `forRange`-invariant argument on
the consumer's side. Intended consumers: the rmsnorm two-pass family and
the streamed normalize/emit relatives.

**Single-surface design.** As for the whole streaming genre, the skin
carries only `⊨[R]` (`ImplementsR`, `execR R`) — no exact-`exec`
`Implements` and no `toU` core embedding; `outDType := .real` (the
default) recovers the exact genre losslessly, and a narrow-float in-loop
store (an fp16 emit) sets its grid so each emitted cell is the ideal ℝ
value rounded **once**. -/
structure StreamEmitMasked2DKernelIO₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First streamed input buffer. -/
  inp1 : RegionName
  /-- Second streamed input buffer. -/
  inp2 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of streaming steps (the `forRange` trip count). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the emitted output window. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  per-step boundary stores. `⊨[R]`'s postcondition reads every emitted
  cell back as an `outDType`-typed cell holding
  `outDType.ofReal (R.round outDType (f … t j))`; an fp16-emitting
  streaming kernel sets `.fp16`. `.real` (the default) is an unrounded
  store — exact under `execR R`, so the default recovers the exact
  genre. -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address. -/
  read2 : Nat → Nat → Fin T → Fin B2 → Nat
  /-- Step `t`, lane `j`'s write address — the per-step emit window. -/
  write : Nat → Nat → Fin T → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`. -/
  mask2 : Nat → Nat → Fin T → Fin B2 → Prop
  /-- The step-`t` store's write-active lanes. -/
  writeMask : Nat → Nat → Fin T → Fin C → Prop

namespace StreamEmitMasked2DKernelIO₂

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the two-stream per-step emit skin, the genre's only
surface (see the `StreamMasked2DKernelIO₂` structure's single-surface
design note). Same full Hoare triple as the family's `ImplementsR`
relations (∀ disjoint allocation, ∀ pids, ∀ launch state with both masked
input streams loaded exact-ℝ step by step), the execution is `execR R`,
and the output readback is the `KernelIO₂.ImplementsR` contract per
**emitted cell**: for every step `t` and write-active lane `j` the cell at
the step-`t` window holds the *ideal* real value `f pid₀ pid₁ xs ys t j`,
quantized **once** at the declared grid `outDType` — read back through
`readMemAs io.outDType` as `io.outDType.ofReal (R.round io.outDType (f …))`.
Frame: every flat cell outside the union of the write-active per-step
windows is untouched. At `R := .triv` and `outDType := .real` every store
is exact and this is the exact streaming contract. -/
def ImplementsR (io : StreamEmitMasked2DKernelIO₂) (R : RoundingModel)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.T → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      io.read1 pid₀ pid₁ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      io.read2 pid₀ pid₁ t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
      io.write pid₀ pid₁ t j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ t j) = ys t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ t j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ xs ys t j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamEmitMasked2DKernelIO₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, mirroring
`StreamMasked2DKernelIO₂.ImplementsR.intro` with the write window lifted
to the per-step family (no core embedding: the streamed rounding surface
is transported directly via `execR_flatten` and `flattenState_readMemAs`).
Obligations in the skin's named vocabulary: `FlattenOk`, the
`TraceSafeR R` safety walk `hts` (fed the two pinned input streams and the
three window-bound groups), and the region-model rounded Hoare triple
`hrun` (termination under `execR R`, the `readMemAs outDType` rounded
readback of every emitted cell, and the per-step-window frame; the `undef`
pin is threaded in for masked loads without an `other=` default). `hrun`
is where the consumer runs its `forRange` invariant argument — the skin
does not prove the loop. -/
theorem ImplementsR.intro (io : StreamEmitMasked2DKernelIO₂)
    {R : RoundingModel}
    {f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.T → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ),
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
        s.readMem io.inp1 (io.read1 (s.pids 0) (s.pids 1) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s.pids 0) (s.pids 1) t j →
        s.readMem io.inp2 (io.read2 (s.pids 0) (s.pids 1) t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
        io.read1 (s.pids 0) (s.pids 1) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s.pids 0) (s.pids 1) t j →
        io.read2 (s.pids 0) (s.pids 1) t j < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask (s.pids 0) (s.pids 1) t j →
        io.write (s.pids 0) (s.pids 1) t j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp1 (io.read1 (s₀.pids 0) (s₀.pids 1) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp2 (io.read2 (s₀.pids 0) (s₀.pids 1) t j) = ys t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ (t : Fin io.T) (j : Fin io.C),
            io.writeMask (s₀.pids 0) (s₀.pids 1) t j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) t j)
              = io.outDType.ofReal
                  (R.round io.outDType (f (s₀.pids 0) (s₀.pids 1) xs ys t j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ (t : Fin io.T) (j : Fin io.C),
                io.writeMask (s₀.pids 0) (s₀.pids 1) t j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) t j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ xs ys s₀ hpid₀ hpid₁ hu hbr1 hbr2 hbw hx hy
  subst hpid₀
  subst hpid₁
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys hu hx hy
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs ys hx hy hbr1 hbr2 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro t j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) t j < A.extent io.out :=
      hbw t j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
    exact hval t j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_)
          by_cases hro : r = io.out
          · subst hro
            refine Or.inr fun t j hj => ?_
            rcases hcond with hflat | hn
            · exact absurd rfl hflat
            · intro hoj
              exact hn t j hj (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamEmitMasked2DKernelIO₂

/-- IO signature of the **single-stream per-step emit** shape (streaming
genre, style S3: in-loop store): the single-stream narrowing of
`StreamEmitMasked2DKernelIO₂` — a 2-D pid grid, **one streamed float input
channel** (`inp1`) read in `T` loop steps of `B1` lanes each, and **one
output channel** (`out`) written as a per-step `C`-lane window family.
Every field is the verbatim `StreamEmitMasked2DKernelIO₂` field with the
`inp2`/`B2`/`read2`/`mask2` channel removed; the skin also carries the
genre's **single-surface design** unchanged (only `⊨[R]`, with the
`outDType := .real` default recovering the exact genre losslessly — see
the `StreamMasked2DKernelIO₂` structure's design note). Intended
consumers: single-input scans (a decay-cumsum consumer's `f t j` is the
prefix sum `∑ u ≤ t` of its stream). -/
structure StreamEmitMasked2DKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- The streamed input buffer. -/
  inp1 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of streaming steps (the `forRange` trip count). -/
  T : Nat
  /-- Per-step tile length of the input channel. -/
  B1 : Nat
  /-- Per-step tile length of the emitted output window. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  per-step boundary stores. `⊨[R]`'s postcondition reads every emitted
  cell back as an `outDType`-typed cell holding
  `outDType.ofReal (R.round outDType (f … t j))`; an fp16-emitting
  streaming kernel sets `.fp16`. `.real` (the default) is an unrounded
  store — exact under `execR R`, so the default recovers the exact
  genre. -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s write address — the per-step emit window. -/
  write : Nat → Nat → Fin T → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Fin T → Fin B1 → Prop
  /-- The step-`t` store's write-active lanes. -/
  writeMask : Nat → Nat → Fin T → Fin C → Prop

namespace StreamEmitMasked2DKernelIO₁

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the single-stream per-step emit skin, the genre's only
surface (see the `StreamMasked2DKernelIO₂` structure's single-surface
design note). Same full Hoare triple as the family's `ImplementsR`
relations (∀ disjoint allocation, ∀ pids, ∀ launch state with the masked
input stream loaded exact-ℝ step by step), the execution is `execR R`,
and the output readback is per **emitted cell**: for every step `t` and
write-active lane `j` the cell at the step-`t` window holds the *ideal*
real value `f pid₀ pid₁ xs t j`, quantized **once** at the declared grid
`outDType` — read back through `readMemAs io.outDType` as
`io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat cell
outside the union of the write-active per-step windows is untouched. At
`R := .triv` and `outDType := .real` every store is exact and this is the
exact streaming contract. -/
def ImplementsR (io : StreamEmitMasked2DKernelIO₁) (R : RoundingModel)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      Fin io.T → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      io.read1 pid₀ pid₁ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
      io.write pid₀ pid₁ t j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ t j) = xs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ t j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ xs t j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamEmitMasked2DKernelIO₁.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the single-stream
narrowing of `StreamEmitMasked2DKernelIO₂.ImplementsR.intro`. Obligations
in the skin's named vocabulary: `FlattenOk`, the `TraceSafeR R` safety
walk `hts` (fed the pinned input stream and the two window-bound groups),
and the region-model rounded Hoare triple `hrun` (termination under
`execR R`, the `readMemAs outDType` rounded readback of every emitted
cell, and the per-step-window frame; the `undef` pin is threaded in for
masked loads without an `other=` default). `hrun` is where the consumer
runs its `forRange` invariant argument — the skin does not prove the
loop. -/
theorem ImplementsR.intro (io : StreamEmitMasked2DKernelIO₁)
    {R : RoundingModel}
    {f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) → Fin io.T → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ),
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
        s.readMem io.inp1 (io.read1 (s.pids 0) (s.pids 1) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
        io.read1 (s.pids 0) (s.pids 1) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask (s.pids 0) (s.pids 1) t j →
        io.write (s.pids 0) (s.pids 1) t j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp1 (io.read1 (s₀.pids 0) (s₀.pids 1) t j) = xs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ (t : Fin io.T) (j : Fin io.C),
            io.writeMask (s₀.pids 0) (s₀.pids 1) t j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) t j)
              = io.outDType.ofReal
                  (R.round io.outDType (f (s₀.pids 0) (s₀.pids 1) xs t j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ (t : Fin io.T) (j : Fin io.C),
                io.writeMask (s₀.pids 0) (s₀.pids 1) t j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) t j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ xs s₀ hpid₀ hpid₁ hu hbr1 hbw hx
  subst hpid₀
  subst hpid₁
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs hu hx
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs hx hbr1 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro t j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) t j < A.extent io.out :=
      hbw t j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
    exact hval t j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_)
          by_cases hro : r = io.out
          · subst hro
            refine Or.inr fun t j hj => ?_
            rcases hcond with hflat | hn
            · exact absurd rfl hflat
            · intro hoj
              exact hn t j hj (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamEmitMasked2DKernelIO₁

/-! ### The grid-stride emit skin: `StreamGridStrideEmitMasked2DKernelIO₁`

`Stream` + `GridStride` + `Emit` + `Masked` + `2D` stacked as capability
prefixes: the **launch-grid-width** widening of
`StreamEmitMasked2DKernelIO₁`. A grid-stride kernel (the GEMS-generated
pointwise wrappers `pow_scalar_tensor` / `relu_strided_buffer` take this
branch whenever `tiles_per_cta > 1`) covers, from a single program, the
whole tile family `pid₀ + j·num_ctas` for `j < tiles_per_cta`, where
`num_ctas = tl.num_programs(0)` — `BlockState.numPids 0` under
`Op.numPrograms` (`Semantics/EvalOp.lean`).

**Why the grid width needs a channel of its own.** Every other member of
the family pins only `pids 0/1/2` and `undef`, leaves `numPids`
universally quantified, and its windows are functions of the pids alone.
Such a signature cannot state a grid-stride contract at all — not just
inconveniently, but provably: instantiate it at two launch states with
identical inputs and `numPids 0 = 1` versus `numPids 0 = 2`. Program `0`
of a `tiles_per_cta = 2` launch writes tiles `{0,1}` in the first state
and tiles `{0,2}` in the second, while the pid-only write-window family
is the *same* set of cells in both. So the cells of tile `1` must hold
the readback value (first state) and be untouched by the frame (second
state) — and the frame's initial contents are themselves universally
quantified, so no `f` satisfies both. Flattening is no escape:
`FlatAlloc.flattenState` copies `numPids` verbatim. Pinning
`tiles_per_cta := 1` (the wrapper's `one_tile_per_cta` branch) is a
silent weakening, not a fix.

So this skin's windows, masks and spec take the grid width `nCtas`
alongside the pids, and `ImplementsR` pins `s₀.numPids 0 = nCtas` exactly
as it pins `s₀.pids 0 = pid₀`: `nCtas` is universally quantified *inside*
the relation, so one `⊨[R]` covers every launch grid and the notation
stays the family's two-hole form (nothing is hidden — `nCtas` is not an
argument of the relation). Only axis 0 is pinned; the widths of the other
axes stay free, because no modeled kernel reads them.

The existing skins are deliberately left alone: a window field cannot
grow an argument without breaking every instance that supplies it as a
lambda, and a pid-only consumer has no use for the pin. The grid-stride
genre therefore gets a sibling skin, and `StreamEmitMasked2DKernelIO₁`
stays the pid-only member (which this skin degenerates to at
`read1 := fun p₀ p₁ _ => …`, i.e. windows ignoring `nCtas`).
-/

/-- IO signature of the **grid-stride per-step emit** shape (streaming
genre, style S3: in-loop store): the grid-width widening of
`StreamEmitMasked2DKernelIO₁` — a 2-D pid grid, **one streamed float
input channel** (`inp1`) read in `T` loop steps of `B1` lanes each, one
output channel (`out`) written as a per-step `C`-lane window family, and
every window additionally parametrized by the **launch grid width**
`nCtas` on axis 0, which `⊨[R]` pins to `s₀.numPids 0`. `T` is the
kernel's `tiles_per_cta`, the grid-stride loop's trip count; its step `t`
handles the flat tile `pid₀ + t·nCtas` (see the genre note above for why
the width cannot be left implicit).

Every field is the verbatim `StreamEmitMasked2DKernelIO₁` field with the
grid-width argument inserted after the pids, plus the family's `pre`
launch-legality field. The genre's **single-surface design** is unchanged
(only `⊨[R]`, with the `outDType := .real` default recovering the exact
genre losslessly — see the `StreamMasked2DKernelIO₂` structure's design
note). -/
structure StreamGridStrideEmitMasked2DKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- The streamed input buffer. -/
  inp1 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of streaming steps: the grid-stride loop's trip count (the
  kernel's `tiles_per_cta`), *not* the number of tiles in the task
  space — one program walks `T` of them, `nCtas` apart. -/
  T : Nat
  /-- Per-step tile length of the input channel. -/
  B1 : Nat
  /-- Per-step tile length of the emitted output window. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  per-step boundary stores. `⊨[R]`'s postcondition reads every emitted
  cell back as an `outDType`-typed cell holding
  `outDType.ofReal (R.round outDType (f … t j))`; an fp16-emitting
  streaming kernel sets `.fp16`. `.real` (the default) is an unrounded
  store — exact under `execR R`, so the default recovers the exact
  genre. -/
  outDType : FloatDType := .real
  /-- The launch-legality precondition on `(pid₀, pid₁, nCtas)`
  (defaulted unconstrained, as elsewhere in the family): the triple is
  claimed only for launches satisfying `pre`. The grid-stride idiom's own
  use is `fun pid₀ _ nCtas => pid₀ < nCtas` — a program's id is below its
  grid's width, hence `0 < nCtas`. `BlockState` carries no such
  invariant between `pids` and `numPids`, so a kernel whose loop bound
  arithmetic needs it has to assume it here rather than pretend it is
  free. -/
  pre : Nat → Nat → Nat → Prop := fun _ _ _ => True
  /-- Step `t`, lane `j`'s `inp1` read address for program `(pid₀, pid₁)`
  in a grid of width `nCtas` on axis 0 — typically the tile
  `pid₀ + t·nCtas` of the flat task space. -/
  read1 : Nat → Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s write address — the per-step emit window, at
  the same grid-stride tile as the read. -/
  write : Nat → Nat → Nat → Fin T → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t` (a boundary-checked
  block-pointer load masks the tail tile). -/
  mask1 : Nat → Nat → Nat → Fin T → Fin B1 → Prop
  /-- The step-`t` store's write-active lanes. -/
  writeMask : Nat → Nat → Nat → Fin T → Fin C → Prop

namespace StreamGridStrideEmitMasked2DKernelIO₁

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the grid-stride per-step emit skin, the genre's only
surface (see the `StreamMasked2DKernelIO₂` structure's single-surface
design note). Same full Hoare triple as
`StreamEmitMasked2DKernelIO₁.ImplementsR` (∀ disjoint allocation, ∀ pids,
∀ launch state with the masked input stream loaded exact-ℝ step by step),
with the **launch grid width** `nCtas` quantified alongside the pids and
pinned by `s₀.numPids 0 = nCtas`, so every window may name it. The
execution is `execR R`; the output readback is per **emitted cell**: for
every step `t` and write-active lane `j` the cell at the step-`t` window
holds the *ideal* real value `f pid₀ pid₁ nCtas xs t j`, quantized
**once** at the declared grid `outDType` — read back through
`readMemAs io.outDType` as
`io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat cell
outside the union of the write-active per-step windows is untouched. At
`R := .triv` and `outDType := .real` every store is exact and this is the
exact grid-stride streaming contract. -/
def ImplementsR (io : StreamGridStrideEmitMasked2DKernelIO₁)
    (R : RoundingModel)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      Fin io.T → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ nCtas : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (s₀ : BlockState),
    io.pre pid₀ pid₁ nCtas →
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.numPids 0 = nCtas →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ nCtas t j →
      io.read1 pid₀ pid₁ nCtas t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ nCtas t j →
      io.write pid₀ pid₁ nCtas t j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ nCtas t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ nCtas t j) = xs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ nCtas t j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ nCtas t j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ nCtas xs t j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (t : Fin io.T) (j : Fin io.C),
              io.writeMask pid₀ pid₁ nCtas t j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ nCtas t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamGridStrideEmitMasked2DKernelIO₁.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the grid-width
widening of `StreamEmitMasked2DKernelIO₁.ImplementsR.intro`. Obligations
in the skin's named vocabulary: `FlattenOk`, the `TraceSafeR R` safety
walk `hts` (fed `pre`, the pinned input stream and the two window-bound
groups), and the region-model rounded Hoare triple `hrun` (termination
under `execR R`, the `readMemAs outDType` rounded readback of every
emitted cell, and the per-step-window frame; the `undef` pin is threaded
in for masked loads without an `other=` default). Both obligations read
the grid width off the state as `s.numPids 0` — the pin is discharged
here, exactly as the pid pins are, so a consumer's loop lemma keeps
stating its addresses in terms of `s.pids 0` and `s.numPids 0` (which is
what evaluating `Op.numPrograms 0` hands it). `hrun` is where the
consumer runs its `forRangeDyn` invariant argument — the skin does not
prove the loop. -/
theorem ImplementsR.intro (io : StreamGridStrideEmitMasked2DKernelIO₁)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      Fin io.T → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ),
      io.pre (s.pids 0) (s.pids 1) (s.numPids 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.numPids 0) t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.numPids 0) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.numPids 0) t j →
        io.read1 (s.pids 0) (s.pids 1) (s.numPids 0) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.C),
        io.writeMask (s.pids 0) (s.pids 1) (s.numPids 0) t j →
        io.write (s.pids 0) (s.pids 1) (s.numPids 0) t j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ),
      io.pre (s₀.pids 0) (s₀.pids 1) (s₀.numPids 0) →
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.numPids 0) t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.numPids 0) t j) = xs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ (t : Fin io.T) (j : Fin io.C),
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.numPids 0) t j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.numPids 0) t j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.numPids 0) xs t j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ (t : Fin io.T) (j : Fin io.C),
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.numPids 0) t j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.numPids 0) t j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ nCtas xs s₀ hpre hpid₀ hpid₁ hnum hu hbr1
    hbw hx
  subst hpid₀
  subst hpid₁
  subst hnum
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs hpre hu hx
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs hpre hx hbr1 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro t j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) (s₀.numPids 0) t j
        < A.extent io.out := hbw t j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
    exact hval t j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_)
          by_cases hro : r = io.out
          · subst hro
            refine Or.inr fun t j hj => ?_
            rcases hcond with hflat | hn
            · exact absurd rfl hflat
            · intro hoj
              exact hn t j hj (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamGridStrideEmitMasked2DKernelIO₁

/-- IO signature of the **three-stream per-step emit** shape (streaming
genre, style S3: in-loop store): the three-stream widening of
`StreamEmitMasked2DKernelIO₂` — a 2-D pid grid, **three streamed float
input channels** (`inp1`/`inp2`/`inp3`) read in `T` loop steps of
`B1`/`B2`/`B3` lanes each, and **one output channel** (`out`) written as a
per-step `C`-lane window family. A channel whose read window ignores `t`
is the degenerate *static* stream (a pre-loop tile re-read each step) — a
diag-ssm consumer streams `x` per step while its initial-state and decay
tiles use `t`-independent windows, so no separate static-input skin is
needed. All other fields and the genre's **single-surface design** are the
verbatim `StreamEmitMasked2DKernelIO₂` shape (only `⊨[R]`, with the
`outDType := .real` default recovering the exact genre losslessly — see
the `StreamMasked2DKernelIO₂` structure's design note). -/
structure StreamEmitMasked2DKernelIO₃ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First streamed input buffer. -/
  inp1 : RegionName
  /-- Second streamed input buffer. -/
  inp2 : RegionName
  /-- Third streamed input buffer. -/
  inp3 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of streaming steps (the `forRange` trip count). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the third input channel. -/
  B3 : Nat
  /-- Per-step tile length of the emitted output window. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  per-step boundary stores. `⊨[R]`'s postcondition reads every emitted
  cell back as an `outDType`-typed cell holding
  `outDType.ofReal (R.round outDType (f … t j))`; an fp16-emitting
  streaming kernel sets `.fp16`. `.real` (the default) is an unrounded
  store — exact under `execR R`, so the default recovers the exact
  genre. -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address. -/
  read2 : Nat → Nat → Fin T → Fin B2 → Nat
  /-- Step `t`, lane `j`'s `inp3` read address. -/
  read3 : Nat → Nat → Fin T → Fin B3 → Nat
  /-- Step `t`, lane `j`'s write address — the per-step emit window. -/
  write : Nat → Nat → Fin T → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`. -/
  mask2 : Nat → Nat → Fin T → Fin B2 → Prop
  /-- `inp3`'s read-active lanes at step `t`. -/
  mask3 : Nat → Nat → Fin T → Fin B3 → Prop
  /-- The step-`t` store's write-active lanes. -/
  writeMask : Nat → Nat → Fin T → Fin C → Prop

namespace StreamEmitMasked2DKernelIO₃

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the three-stream per-step emit skin, the genre's only
surface (see the `StreamMasked2DKernelIO₂` structure's single-surface
design note). Same full Hoare triple as the family's `ImplementsR`
relations (∀ disjoint allocation, ∀ pids, ∀ launch state with all three
masked input streams loaded exact-ℝ step by step), the execution is
`execR R`, and the output readback is per **emitted cell**: for every
step `t` and write-active lane `j` the cell at the step-`t` window holds
the *ideal* real value `f pid₀ pid₁ xs ys zs t j`, quantized **once** at
the declared grid `outDType` — read back through `readMemAs io.outDType`
as `io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat
cell outside the union of the write-active per-step windows is untouched.
At `R := .triv` and `outDType := .real` every store is exact and this is
the exact streaming contract. -/
def ImplementsR (io : StreamEmitMasked2DKernelIO₃) (R : RoundingModel)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.T → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      io.read1 pid₀ pid₁ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      io.read2 pid₀ pid₁ t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ t j →
      io.read3 pid₀ pid₁ t j < A.extent io.inp3) →
    (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
      io.write pid₀ pid₁ t j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ t j) = zs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ t j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ xs ys zs t j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ t j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamEmitMasked2DKernelIO₃.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the three-stream
widening of `StreamEmitMasked2DKernelIO₂.ImplementsR.intro`. Obligations
in the skin's named vocabulary: `FlattenOk`, the `TraceSafeR R` safety
walk `hts` (fed the three pinned input streams and the four window-bound
groups), and the region-model rounded Hoare triple `hrun` (termination
under `execR R`, the `readMemAs outDType` rounded readback of every
emitted cell, and the per-step-window frame; the `undef` pin is threaded
in for masked loads without an `other=` default). `hrun` is where the
consumer runs its `forRange` invariant argument — the skin does not prove
the loop. -/
theorem ImplementsR.intro (io : StreamEmitMasked2DKernelIO₃)
    {R : RoundingModel}
    {f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.T → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
        (zs : Fin io.T → Fin io.B3 → ℝ),
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
        s.readMem io.inp1 (io.read1 (s.pids 0) (s.pids 1) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s.pids 0) (s.pids 1) t j →
        s.readMem io.inp2 (io.read2 (s.pids 0) (s.pids 1) t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 (s.pids 0) (s.pids 1) t j →
        s.readMem io.inp3 (io.read3 (s.pids 0) (s.pids 1) t j) = zs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
        io.read1 (s.pids 0) (s.pids 1) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s.pids 0) (s.pids 1) t j →
        io.read2 (s.pids 0) (s.pids 1) t j < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 (s.pids 0) (s.pids 1) t j →
        io.read3 (s.pids 0) (s.pids 1) t j < bounds io.inp3) →
      (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask (s.pids 0) (s.pids 1) t j →
        io.write (s.pids 0) (s.pids 1) t j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ) (zs : Fin io.T → Fin io.B3 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp1 (io.read1 (s₀.pids 0) (s₀.pids 1) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp2 (io.read2 (s₀.pids 0) (s₀.pids 1) t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp3 (io.read3 (s₀.pids 0) (s₀.pids 1) t j) = zs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ (t : Fin io.T) (j : Fin io.C),
            io.writeMask (s₀.pids 0) (s₀.pids 1) t j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) t j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) xs ys zs t j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ (t : Fin io.T) (j : Fin io.C),
                io.writeMask (s₀.pids 0) (s₀.pids 1) t j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) t j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ xs ys zs s₀ hpid₀ hpid₁ hu hbr1 hbr2 hbr3
    hbw hx hy hz
  subst hpid₀
  subst hpid₁
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys zs hu hx hy hz
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs ys zs hx hy hz hbr1 hbr2 hbr3 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro t j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) t j < A.extent io.out :=
      hbw t j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
    exact hval t j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_)
          by_cases hro : r = io.out
          · subst hro
            refine Or.inr fun t j hj => ?_
            rcases hcond with hflat | hn
            · exact absurd rfl hflat
            · intro hoj
              exact hn t j hj (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamEmitMasked2DKernelIO₃

/-- IO signature of the **single-stream per-step emit** shape on a **3-D
pid grid** (streaming genre, style S3: in-loop store): the three-pid
widening of `StreamEmitMasked2DKernelIO₁` — `3D` names the grid, exactly
as in `Masked3DKernelIO₂ₓ₂`/`StreamMetaMasked3DKernelIO₂`. One streamed
float input channel (`inp1`) read in `T` loop steps of `B1` lanes each,
and one output channel (`out`) written as a per-step `C`-lane window
family; every window and mask eats all three pids. All other fields and
the genre's **single-surface design** are the verbatim
`StreamEmitMasked2DKernelIO₁` shape (only `⊨[R]`, with the
`outDType := .real` default recovering the exact genre losslessly — see
the `StreamMasked2DKernelIO₂` structure's design note). Intended
consumers: the 3-D-grid scan family (a decay-cumsum consumer addresses
its chunk by `(i_k, i_c, i_bh)` and its `f t j` is the prefix sum
`∑ u ≤ t` of its stream). -/
structure StreamEmitMasked3DKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- The streamed input buffer. -/
  inp1 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of streaming steps (the `forRange` trip count). -/
  T : Nat
  /-- Per-step tile length of the input channel. -/
  B1 : Nat
  /-- Per-step tile length of the emitted output window. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  per-step boundary stores. `⊨[R]`'s postcondition reads every emitted
  cell back as an `outDType`-typed cell holding
  `outDType.ofReal (R.round outDType (f … t j))`; an fp16-emitting
  streaming kernel sets `.fp16`. `.real` (the default) is an unrounded
  store — exact under `execR R`, so the default recovers the exact
  genre. -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read1 : Nat → Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s write address — the per-step emit window. -/
  write : Nat → Nat → Nat → Fin T → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Nat → Fin T → Fin B1 → Prop
  /-- The step-`t` store's write-active lanes. -/
  writeMask : Nat → Nat → Nat → Fin T → Fin C → Prop

namespace StreamEmitMasked3DKernelIO₁

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the single-stream per-step emit skin on the 3-D grid, the
genre's only surface (see the `StreamMasked2DKernelIO₂` structure's
single-surface design note). Same full Hoare triple as
`StreamEmitMasked2DKernelIO₁.ImplementsR` with the third pid quantified
and pinned alongside the first two; the output readback is per **emitted
cell**: for every step `t` and write-active lane `j` the cell at the
step-`t` window holds the *ideal* real value `f pid₀ pid₁ pid₂ xs t j`,
quantized **once** at the declared grid `outDType` — read back through
`readMemAs io.outDType` as
`io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat cell
outside the union of the write-active per-step windows is untouched. At
`R := .triv` and `outDType := .real` every store is exact and this is the
exact streaming contract. -/
def ImplementsR (io : StreamEmitMasked3DKernelIO₁) (R : RoundingModel)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      Fin io.T → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      io.read1 pid₀ pid₁ pid₂ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ pid₂ t j →
      io.write pid₀ pid₁ pid₂ t j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ pid₂ t j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ t j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ pid₂ xs t j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (t : Fin io.T) (j : Fin io.C), io.writeMask pid₀ pid₁ pid₂ t j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamEmitMasked3DKernelIO₁.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the three-pid
widening of `StreamEmitMasked2DKernelIO₁.ImplementsR.intro`. Obligations
in the skin's named vocabulary: `FlattenOk`, the `TraceSafeR R` safety
walk `hts` (fed the pinned input stream and the two window-bound groups),
and the region-model rounded Hoare triple `hrun` (termination under
`execR R`, the `readMemAs outDType` rounded readback of every emitted
cell, and the per-step-window frame; the `undef` pin is threaded in for
masked loads without an `other=` default). `hrun` is where the consumer
runs its `forRange` invariant argument — the skin does not prove the
loop. -/
theorem ImplementsR.intro (io : StreamEmitMasked3DKernelIO₁)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      Fin io.T → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ),
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.C),
        io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j) = xs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ (t : Fin io.T) (j : Fin io.C),
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) xs t j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ (t : Fin io.T) (j : Fin io.C),
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ xs s₀ hpid₀ hpid₁ hpid₂ hu hbr1 hbw hx
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs hu hx
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs hx hbr1 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro t j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j
        < A.extent io.out := hbw t j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
    exact hval t j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_)
          by_cases hro : r = io.out
          · subst hro
            refine Or.inr fun t j hj => ?_
            rcases hcond with hflat | hn
            · exact absurd rfl hflat
            · intro hoj
              exact hn t j hj (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamEmitMasked3DKernelIO₁

/-! ### The metadata emit skin: `StreamMetaEmitMasked3DKernelIO₂`

`Stream` + `Meta` + `Emit` + `Masked` + `3D` stack as capability prefixes:
the `StreamMetaMasked3DKernelIO₂` slot mechanism (pre-loop scalar metadata
vector, pid-causal slot windows, sentinel gating through `writeMask`)
composed with the `StreamEmit*` per-step write-window family (style S3:
the store sits inside the loop). The subscript stays pure data-input ×
output arity. -/

/-- IO signature of the **metadata-parametrized two-stream per-step emit**
shape (streaming genre, style S3 on a **3-D pid grid**): before its
`forRange` loop the kernel loads one scalar from each of `nMeta` metadata
regions (an adapter index, a sequence length …), and every read window /
mask, the **per-step** write window / mask and the spec `f` are
parametrized by the loaded slot vector; at step `t` the program stores a
`C`-lane tile to the `t`-indexed window `write · · · m t ·`. Intended
consumers: the LoRA expand family (bgmv/lora expand), whose per-batch
adapter choice is a slot and whose output-block loop emits one disjoint
window per step. A channel whose read window ignores `t` is the genre's
degenerate static stream (a pre-loop tile — register-cached re-use needs
no re-read pins beyond step 0, but the uniform pin is harmless and keeps
the contract one-shaped).

**Slot causality** and **sentinel gating** are inherited verbatim from
`StreamMetaMasked3DKernelIO₂`: slot windows `mwin` eat only the pids
(chained indirection is out of scope), and `writeMask` eats the slot
vector `m`, so the skip-sentinel idiom (adapter index `= -1` ⇒ the
program stores nothing) is an empty write-active window family at the
sentinel value — every readback leg vacuous, the frame keeping the whole
output untouched.

**Single-surface design.** As everywhere in the streaming genre the skin
carries only `⊨[R]` (`ImplementsR`, `execR R`) with the
`outDType := .real` default recovering the exact genre losslessly — see
the `StreamMasked2DKernelIO₂` design note. -/
structure StreamMetaEmitMasked3DKernelIO₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First streamed input buffer. -/
  inp1 : RegionName
  /-- Second streamed input buffer. -/
  inp2 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of scalar metadata slots (a field, never a name subscript). -/
  nMeta : Nat
  /-- Slot `k`'s element type. Slots read back through `(sty k).read` —
  `readMem` for `.float`, the operational `readMemValue` view for the
  typed channels — so `.nat` sequence lengths and `.int` adapter indices
  are both expressible. -/
  sty : Fin nMeta → ChanTy
  /-- Slot `k`'s region (one cell read per program; slot regions may
  pairwise differ). -/
  mbuf : Fin nMeta → RegionName
  /-- Slot `k`'s cell address for program `(pid₀, pid₁, pid₂)` — a
  function of the pids only, never of other slots' values (see the
  slot-causality design note). -/
  mwin : Fin nMeta → Nat → Nat → Nat → Nat
  /-- **Static** step budget: the `forRange` trip count of a
  slot-independent loop, and the value the `steps` default reads. -/
  T : Nat
  /-- Number of streaming steps as a function of the **loaded slot
  vector** — the trip count of a loop whose bound is *read from memory*
  (`for i in range(0, length, BLOCK)` with `length` a per-program metadata
  cell: `steps m = ⌈(m lenSlot) / BLOCK⌉`). Every step-indexed field below
  is indexed by `Fin (steps m)`, so the loop budget is chosen *after* the
  slot vector is quantified.

  Why the budget itself had to move here (and why a `pre` launch-legality
  field would not do): the write-active family is indexed by
  `Fin (steps m) × Fin C`, so under a *constant* budget the permitted
  write set of every launch has at most `T · C` cells, while such a kernel
  writes `⌈length / BLOCK⌉ · BLOCK` of them — a launch with
  `length > T · C` makes the **frame clause false**, and no `writeMask`
  can repair that (a mask only shrinks the permitted write set, never
  grows it). A `pre` field could only *exclude* those launches, i.e.
  record a launch restriction the kernel does not actually have; a
  slot-derived budget states the real contract. Pid-dependent trip counts
  stay `pre`'s business (the `StreamMasked3DKernelIO₃ₓ₃` precedent) —
  this field is for the *memory-loaded* ones, and a slot value already
  carries its own pid dependence through `mwin`.

  The lane widths (`B1`/`B2`/`C`) stay static on purpose: a tile's lane
  count is a `tl.constexpr` in Triton, never a loaded value — only the
  trip count can come out of memory.

  Defaulted to the constant budget `fun _ => T`, so a slot-independent
  consumer never mentions it and its windows keep their `Fin T` shape up
  to unfolding. -/
  steps : (∀ k : Fin nMeta, (sty k).carrier) → Nat := fun _ => T
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the emitted output window. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  per-step boundary stores. `⊨[R]`'s postcondition reads every emitted
  cell back as an `outDType`-typed cell holding
  `outDType.ofReal (R.round outDType (f … t j))`; an fp16-emitting
  streaming kernel sets `.fp16`. `.real` (the default) is an unrounded
  store — exact under `execR R`, so the default recovers the exact
  genre. -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program
  `(pid₀, pid₁, pid₂)`, given the loaded slot vector. -/
  read1 : Nat → Nat → Nat → (m : ∀ k : Fin nMeta, (sty k).carrier) →
    Fin (steps m) → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address, given the loaded slot
  vector. -/
  read2 : Nat → Nat → Nat → (m : ∀ k : Fin nMeta, (sty k).carrier) →
    Fin (steps m) → Fin B2 → Nat
  /-- Step `t`, lane `j`'s write address — the per-step emit window,
  given the loaded slot vector. -/
  write : Nat → Nat → Nat → (m : ∀ k : Fin nMeta, (sty k).carrier) →
    Fin (steps m) → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`, given the loaded slot vector
  (a sequence-length slot bounds the live steps). -/
  mask1 : Nat → Nat → Nat → (m : ∀ k : Fin nMeta, (sty k).carrier) →
    Fin (steps m) → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`, given the loaded slot
  vector. -/
  mask2 : Nat → Nat → Nat → (m : ∀ k : Fin nMeta, (sty k).carrier) →
    Fin (steps m) → Fin B2 → Prop
  /-- The step-`t` store's write-active lanes, given the loaded slot
  vector — the sentinel gate (see the sentinel-gating design note). -/
  writeMask : Nat → Nat → Nat → (m : ∀ k : Fin nMeta, (sty k).carrier) →
    Fin (steps m) → Fin C → Prop

namespace StreamMetaEmitMasked3DKernelIO₂

/-- The pinned slot-value context: one loaded scalar per metadata slot, at
the slot's own `ChanTy` carrier. The structure's window/mask fields spell
this Pi type out verbatim (a field type cannot mention the abbreviation of
the structure it lives in); consumers and the `⊨[R]` surface use the
abbreviation. -/
abbrev Meta (io : StreamMetaEmitMasked3DKernelIO₂) : Type :=
  ∀ k : Fin io.nMeta, (io.sty k).carrier

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the metadata-parametrized two-stream per-step emit skin,
the streaming genre's only surface (see the `StreamMasked2DKernelIO₂`
structure's single-surface design note). Same full Hoare triple as
`StreamMetaMasked3DKernelIO₂.ImplementsR` — ∀ disjoint allocation (the
slot regions listed before the data regions), ∀ pids (three of them),
∀ launch state, the slot vector `m` universally quantified and pinned
cell by cell through each slot's own `ChanTy` readback **before** the
streamed inputs are pinned — with the output readback lifted to the
per-step family: for every step `t` and write-active lane `j` the cell at
the step-`t` window holds the *ideal* real value
`f pid₀ pid₁ pid₂ m xs ys t j`, quantized **once** at the declared grid
`outDType` — read back through `readMemAs io.outDType` as
`io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat cell
outside the union of the write-active per-step windows is untouched. At
`R := .triv` and `outDType := .real` every store is exact and this is the
exact streaming contract.

**Slot-derived step budget.** Every step index ranges over
`Fin (io.steps m)` — the slot vector `m` is quantified *before* the loop
budget is read, so a memory-loaded trip count (`steps m = ⌈(m k)/BLOCK⌉`)
is expressible and its write-active family really covers what the kernel
writes (see the `steps` field docstring for the frame-clause disproof of
the constant-budget spelling). A slot-independent consumer leaves `steps`
at its `fun _ => io.T` default and reads every clause below as the old
`Fin io.T` one. -/
def ImplementsR (io : StreamMetaEmitMasked3DKernelIO₂) (R : RoundingModel)
    (f : ∀ (pid₀ pid₁ pid₂ : Nat) (m : io.Meta),
      (Fin (io.steps m) → Fin io.B1 → ℝ) →
      (Fin (io.steps m) → Fin io.B2 → ℝ) →
      Fin (io.steps m) → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = List.ofFn io.mbuf ++ [io.inp1, io.inp2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (m : io.Meta) (xs : Fin (io.steps m) → Fin io.B1 → ℝ)
    (ys : Fin (io.steps m) → Fin io.B2 → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ k : Fin io.nMeta,
      io.mwin k pid₀ pid₁ pid₂ < A.extent (io.mbuf k)) →
    (∀ (t : Fin (io.steps m)) (j : Fin io.B1),
      io.mask1 pid₀ pid₁ pid₂ m t j →
      io.read1 pid₀ pid₁ pid₂ m t j < A.extent io.inp1) →
    (∀ (t : Fin (io.steps m)) (j : Fin io.B2),
      io.mask2 pid₀ pid₁ pid₂ m t j →
      io.read2 pid₀ pid₁ pid₂ m t j < A.extent io.inp2) →
    (∀ (t : Fin (io.steps m)) (j : Fin io.C),
      io.writeMask pid₀ pid₁ pid₂ m t j →
      io.write pid₀ pid₁ pid₂ m t j < A.extent io.out) →
    (∀ k : Fin io.nMeta,
      (io.sty k).read s₀ (io.mbuf k) (io.mwin k pid₀ pid₁ pid₂) = m k) →
    (∀ (t : Fin (io.steps m)) (j : Fin io.B1),
      io.mask1 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ m t j) = xs t j) →
    (∀ (t : Fin (io.steps m)) (j : Fin io.B2),
      io.mask2 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ m t j) = ys t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (t : Fin (io.steps m)) (j : Fin io.C),
          io.writeMask pid₀ pid₁ pid₂ m t j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ m t j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ pid₂ m xs ys t j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (t : Fin (io.steps m)) (j : Fin io.C),
              io.writeMask pid₀ pid₁ pid₂ m t j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ m t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMetaEmitMasked3DKernelIO₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the per-step-emit
lift of `StreamMetaMasked3DKernelIO₂.ImplementsR.intro`. Obligations in
the skin's named vocabulary: `FlattenOk`, the `TraceSafeR R` safety walk
`hts` (fed the pinned slot vector, the two pinned input streams and the
four window-bound groups — slots first), and the region-model rounded
Hoare triple `hrun` (termination under `execR R`, the
`readMemAs outDType` rounded readback of every emitted cell, and the
per-step-window frame; the `undef` pin is threaded in for masked loads
without an `other=` default). Both `hts` and `hrun` receive the slot
vector `m` with its per-slot `ChanTy` pins — the slot pins are stated on
the *region-model* state, so they pass to the obligations verbatim, no
flattening transport needed. `hrun` is where the consumer runs its
`forRange` invariant argument — the skin does not prove the loop. The
lemma never inspects the step budget: every step index is
`Fin (io.steps m)` with `m` already in scope, so a slot-derived trip
count needs nothing here beyond the `Fin` bound it already carries. -/
theorem ImplementsR.intro (io : StreamMetaEmitMasked3DKernelIO₂)
    {R : RoundingModel}
    {f : ∀ (pid₀ pid₁ pid₂ : Nat) (m : io.Meta),
      (Fin (io.steps m) → Fin io.B1 → ℝ) →
      (Fin (io.steps m) → Fin io.B2 → ℝ) →
      Fin (io.steps m) → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m : io.Meta)
        (xs : Fin (io.steps m) → Fin io.B1 → ℝ)
        (ys : Fin (io.steps m) → Fin io.B2 → ℝ),
      (∀ k : Fin io.nMeta,
        (io.sty k).read s (io.mbuf k)
          (io.mwin k (s.pids 0) (s.pids 1) (s.pids 2)) = m k) →
      (∀ (t : Fin (io.steps m)) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) m t j) = xs t j) →
      (∀ (t : Fin (io.steps m)) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) m t j) = ys t j) →
      (∀ k : Fin io.nMeta,
        io.mwin k (s.pids 0) (s.pids 1) (s.pids 2) < bounds (io.mbuf k)) →
      (∀ (t : Fin (io.steps m)) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.inp1) →
      (∀ (t : Fin (io.steps m)) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.inp2) →
      (∀ (t : Fin (io.steps m)) (j : Fin io.C),
        io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) m t j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (m : io.Meta)
        (xs : Fin (io.steps m) → Fin io.B1 → ℝ)
        (ys : Fin (io.steps m) → Fin io.B2 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ k : Fin io.nMeta,
        (io.sty k).read s₀ (io.mbuf k)
          (io.mwin k (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) = m k) →
      (∀ (t : Fin (io.steps m)) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = xs t j) →
      (∀ (t : Fin (io.steps m)) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = ys t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ (t : Fin (io.steps m)) (j : Fin io.C),
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m xs ys t j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ (t : Fin (io.steps m)) (j : Fin io.C),
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ m xs ys s₀ hpid₀ hpid₁ hpid₂ hu hbm
    hbr1 hbr2 hbw hm hx hy
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ m xs ys hu hm hx hy
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ m xs ys hm hx hy hbm hbr1 hbr2 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by
    rw [hregs]
    exact List.mem_append_right _ (by simp)
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro t j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j
        < A.extent io.out := hbw t j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
    exact hval t j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_)
          by_cases hro : r = io.out
          · subst hro
            refine Or.inr fun t j hj => ?_
            rcases hcond with hflat | hn
            · exact absurd rfl hflat
            · intro hoj
              exact hn t j hj (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamMetaEmitMasked3DKernelIO₂

/-! ### The grouped emit skin: `StreamGroupedEmitMasked3DKernelIO`

`Stream` + `Grouped` + `Emit` + `Masked` + `3D` stack as capability
prefixes; per the `Grouped*` convention there is no arity subscript — the
channel counts are fields. The multi-store streaming kernels (a backward
pass emitting `dq`/`dk`/`dg` rows each iteration, a prepare pass emitting
`qg`/`kg`) own *many* same-length per-step windows; a named-field
`₇ₓ₃`-style emit struct would need one field and one intro leg per
channel, so the grouped treatment takes over exactly as it did for the
rotary/spherical genre. -/

/-- IO signature of the **grouped per-step emit** shape (streaming genre,
style S3 on a **3-D pid grid**): `nIn` float input channels streamed in
`T` loop steps of `B` lanes each, and `nOut` float output channels each
written as a per-step `B`-lane window family, over the decoupled
allocation list `bufs` (an in-place kernel names the same buffer as an
input and an output channel — the decay backward reads and rewrites its
`dq_inter`/`dk_inter` rows). Intended consumers: the multi-store loop
kernels — decay_cumsum's prepare (3 streams in, `qg`/`kg` out) and
backward (7 streams in, `dq_inter`/`dk_inter`/`dg` out, two of them in
place), diag-ssm backward (per-step `grad_x` plus terminal
`grad_s`/`grad_lambda` — a terminal store is the degenerate window family
whose `writeMask` selects a single designated step).

Design notes inherited from the genre: a channel whose read window
ignores `t` is a static stream; a pre-loop tile whose address coincides
with one step's stream cells (decay's `last_decay`/`last_g` row) needs no
extra channel — the spec `f` reads that step of the stream directly. For
an in-place channel pair `f` receives the *old* contents (pinned on the
launch state) and the postcondition asserts the *new* contents — sound
because each emitted cell is written exactly once. **Single-surface
design** as everywhere in the streaming genre: only `⊨[R]`, with the
uniform `outDType := .real` default recovering the exact genre
losslessly (see the `StreamMasked2DKernelIO₂` design note). -/
structure StreamGroupedEmitMasked3DKernelIO where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- Number of input channels (a field, never a name subscript). -/
  nIn : Nat
  /-- Number of output channels. -/
  nOut : Nat
  /-- The allocation list: every buffer the kernel touches, each exactly
  once. The channel buffers below point into this list; for an in-place
  kernel an output channel names the same buffer as an input channel. -/
  bufs : List RegionName
  /-- Input channel `i`'s buffer (channels may share a buffer). -/
  inp : Fin nIn → RegionName
  /-- Output channel `o`'s buffer (channels may share a buffer). -/
  out : Fin nOut → RegionName
  /-- Number of streaming steps (the `forRange` trip count). -/
  T : Nat
  /-- Tile length: every channel owns per-step `B`-lane windows. -/
  B : Nat
  /-- The uniform output dtype — the quantization grid of every per-step
  boundary store. `⊨[R]`'s postcondition reads every emitted cell of
  every output channel back as an `outDType`-typed cell holding
  `outDType.ofReal (R.round outDType (f … o t j))`; `.real` (the default)
  is an unrounded store — exact under `execR R`, so the default recovers
  the exact genre. -/
  outDType : FloatDType := .real
  /-- Input channel `i`'s step-`t`, lane-`j` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read : Fin nIn → Nat → Nat → Nat → Fin T → Fin B → Nat
  /-- Input channel `i`'s read-active lanes at step `t`. -/
  readMask : Fin nIn → Nat → Nat → Nat → Fin T → Fin B → Prop
  /-- Output channel `o`'s step-`t`, lane-`j` write address — the
  per-step emit window family. -/
  write : Fin nOut → Nat → Nat → Nat → Fin T → Fin B → Nat
  /-- Output channel `o`'s step-`t` write-active lanes (a terminal store
  gates on a single designated step). -/
  writeMask : Fin nOut → Nat → Nat → Nat → Fin T → Fin B → Prop

namespace StreamGroupedEmitMasked3DKernelIO

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the grouped per-step emit skin, the streaming genre's
only surface (see the `StreamMasked2DKernelIO₂` structure's
single-surface design note). The pinned inputs are one curried function
`xs : Fin nIn → Fin T → Fin B → ℝ` (channel, step, lane), pinned per
channel on its read-active lanes **on the launch state** (for an in-place
channel pair this is the *old* contents); the spec `f` takes the output
channel as an argument, and for every output channel `o`, step `t` and
write-active lane `j` the cell at channel `o`'s step-`t` window holds the
*ideal* real value `f pid₀ pid₁ pid₂ xs o t j`, quantized **once** at the
declared grid `outDType` — read back through `readMemAs io.outDType` as
`io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat cell
outside the union of all write-active per-step windows of all output
channels is untouched. At `R := .triv` and `outDType := .real` every
store is exact and this is the exact streaming contract. -/
def ImplementsR (io : StreamGroupedEmitMasked3DKernelIO) (R : RoundingModel)
    (f : Nat → Nat → Nat → (Fin io.nIn → Fin io.T → Fin io.B → ℝ) →
      Fin io.nOut → Fin io.T → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = io.bufs →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.nIn → Fin io.T → Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (i : Fin io.nIn) (t : Fin io.T) (j : Fin io.B),
      io.readMask i pid₀ pid₁ pid₂ t j →
      io.read i pid₀ pid₁ pid₂ t j < A.extent (io.inp i)) →
    (∀ (o : Fin io.nOut) (t : Fin io.T) (j : Fin io.B),
      io.writeMask o pid₀ pid₁ pid₂ t j →
      io.write o pid₀ pid₁ pid₂ t j < A.extent (io.out o)) →
    (∀ (i : Fin io.nIn) (t : Fin io.T) (j : Fin io.B),
      io.readMask i pid₀ pid₁ pid₂ t j →
      s₀.readMem (io.inp i) (io.read i pid₀ pid₁ pid₂ t j) = xs i t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (o : Fin io.nOut) (t : Fin io.T) (j : Fin io.B),
          io.writeMask o pid₀ pid₁ pid₂ t j →
          s'.readMemAs io.outDType A.flat
              (A.addr (io.out o) (io.write o pid₀ pid₁ pid₂ t j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ pid₂ xs o t j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ (o : Fin io.nOut) (t : Fin io.T) (j : Fin io.B),
              io.writeMask o pid₀ pid₁ pid₂ t j →
              o' ≠ A.addr (io.out o) (io.write o pid₀ pid₁ pid₂ t j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamGroupedEmitMasked3DKernelIO.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the grouped
sibling of `StreamEmitMasked3DKernelIO₁.ImplementsR.intro` with the
channels indexed by `Fin nIn`/`Fin nOut`, plus the membership side
condition `hout` tying every output channel's buffer into the declared
allocation list. Obligations in the skin's named vocabulary: `FlattenOk`,
the `TraceSafeR R` safety walk `hts` (fed the pinned input channels and
the two window-bound groups), and the region-model rounded Hoare triple
`hrun` (termination under `execR R`, the `readMemAs outDType` rounded
readback of every emitted cell of every channel, and one frame condition
quantified over all output channels; the `undef` pin is threaded in for
masked loads without an `other=` default). `hrun` is where the consumer
runs its `forRange` invariant argument — the skin does not prove the
loop. -/
theorem ImplementsR.intro (io : StreamGroupedEmitMasked3DKernelIO)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → (Fin io.nIn → Fin io.T → Fin io.B → ℝ) →
      Fin io.nOut → Fin io.T → Fin io.B → ℝ}
    (hout : ∀ o, io.out o ∈ io.bufs)
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.nIn → Fin io.T → Fin io.B → ℝ),
      (∀ (i : Fin io.nIn) (t : Fin io.T) (j : Fin io.B),
        io.readMask i (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem (io.inp i)
            (io.read i (s.pids 0) (s.pids 1) (s.pids 2) t j) = xs i t j) →
      (∀ (i : Fin io.nIn) (t : Fin io.T) (j : Fin io.B),
        io.readMask i (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read i (s.pids 0) (s.pids 1) (s.pids 2) t j
          < bounds (io.inp i)) →
      (∀ (o : Fin io.nOut) (t : Fin io.T) (j : Fin io.B),
        io.writeMask o (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.write o (s.pids 0) (s.pids 1) (s.pids 2) t j
          < bounds (io.out o)) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState)
        (xs : Fin io.nIn → Fin io.T → Fin io.B → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (i : Fin io.nIn) (t : Fin io.T) (j : Fin io.B),
        io.readMask i (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem (io.inp i)
            (io.read i (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = xs i t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ (o : Fin io.nOut) (t : Fin io.T) (j : Fin io.B),
            io.writeMask o (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
            s1.readMemAs io.outDType (io.out o)
                (io.write o (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) xs o t j)))
        ∧ (∀ r oo,
            (∀ (o : Fin io.nOut) (t : Fin io.T) (j : Fin io.B),
              io.writeMask o (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
              r = io.out o →
              oo ≠ io.write o (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j) →
            s1.mem r oo = s₀.mem r oo)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ xs s₀ hpid₀ hpid₁ hpid₂ hu hbr hbw hx
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs hu hx
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs hx hbr hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : ∀ o : Fin io.nOut, io.out o ∈ A.regions := by
    intro o
    rw [hregs]
    exact hout o
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro o t j hj
    have hlt : io.write o (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j
        < A.extent (io.out o) := hbw o t j hj
    rw [A.flattenState_readMemAs hd s1 (hmem o) hlt io.outDType]
    exact hval o t j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ?_)
          intro o₀ t j hj hro hoj
          rcases hcond with hflat | hn
          · exact absurd rfl hflat
          · exact hn o₀ t j hj (by rw [hoeq, hro, hoj])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamGroupedEmitMasked3DKernelIO

end VeriTile.Triton
