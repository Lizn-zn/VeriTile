/-
Kernel IO contracts: Stream.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton

/-! ### The streaming genre: `Stream*` skins

The `forRange`-loop kernels stream their inputs: for `t = 0 .. T-1` the
program loads one per-step tile from each input channel, folds it into a
register accumulator, and after the loop stores the `C`-lane result once.
The skins keep the per-step view **curried** — every input channel is
indexed `(t : Fin T, lane : Fin B)`, never flattened to `Fin (T*B)` — so a
consumer's windows, masks and spec all speak per step, exactly the way the
kernel's loop body does. `Stream` names the capability; the subscript stays
pure data-input × output arity (`T` and the lane counts are fields, never
name subscripts). -/

/-- IO signature of the **two-stream fold** shape (streaming genre, style
S1: fold + terminal store): a 2-D pid grid, **two streamed float input
channels** (`inp1`/`inp2`) read in `T` loop steps of `B1`/`B2` lanes each,
and **one output channel** (`out`, `C` lanes) written once after the loop.
The kernel folds the `T` per-step tile pairs into a register accumulator;
the spec `f` eats the *whole* curried streams `(Fin T → Fin B1 → ℝ)` /
`(Fin T → Fin B2 → ℝ)` and returns the terminal value directly — the fold
is expressed inside the consumer's `f` (a GEMM consumer writes
`∑ t, ∑ e, …`), and the skin does not prove the loop for the consumer: the
`ImplementsR.intro` obligation `hrun` is discharged with a
`forRange`-invariant argument on the consumer's side. Intended consumers:
the TritonBench-G matmul family and its streamed-reduction relatives.
Following the family precedent there is no `scratch` field until a consumer
needs one.

**Single-surface design.** The genre carries only the rounding-correctness
relation `⊨[R]` (`ImplementsR`, `execR R`) — no exact-`exec` `Implements`
and no `toU` core embedding. A streaming kernel's terminal store is
routinely a narrow-float typed store (the matmul family stores
`.to(tl.float16)`), and the exact surface's `readMem` readback strictly
mismatches a narrow-float cell; under `execR R` a `.real` store is exact
anyway (`stepStmtR` delegates `.real` writes to the exact semantics), so
the single `⊨[R]` surface with the `outDType := .real` default covers the
exact genre losslessly — the exact surface *is* this relation's
`R := .triv` degeneration, the direct continuation of the
rounding-as-default doctrine. -/
structure StreamMasked2DKernelIO₂ where
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
  /-- Tile length of the terminal output store. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  terminal boundary store. `⊨[R]`'s postcondition reads the output back as
  `outDType`-typed cells holding `outDType.ofReal (R.round outDType (f …))`;
  an fp16-storing streaming kernel sets `.fp16`. `.real` (the default) is an
  unrounded store — exact under `execR R`, so the default recovers the
  exact genre. -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address. -/
  read2 : Nat → Nat → Fin T → Fin B2 → Nat
  /-- Lane `j`'s terminal write address. -/
  write : Nat → Nat → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`. -/
  mask2 : Nat → Nat → Fin T → Fin B2 → Prop
  /-- The terminal store's write-active lanes. -/
  writeMask : Nat → Nat → Fin C → Prop

namespace StreamMasked2DKernelIO₂

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the two-stream fold skin, the genre's only surface (see the
structure's single-surface design note). Same full Hoare triple as the
family's `ImplementsR` relations (∀ disjoint allocation, ∀ pids, ∀ launch
state with both masked input streams loaded exact-ℝ step by step), the
execution is `execR R`, and the single output readback is the
`KernelIO₂.ImplementsR` contract per lane, generalized to the streamed
inputs: the terminal cell holds the *ideal* real value `f pid₀ pid₁ xs ys j`
of the whole `T`-step fold, quantized **once** at the declared grid
`outDType` — read back through `readMemAs io.outDType` as
`io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat cell
outside the write-active output window is untouched. At `R := .triv` and
`outDType := .real` the store is exact and this is the exact streaming
contract. -/
def ImplementsR (io : StreamMasked2DKernelIO₂) (R : RoundingModel)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.C → ℝ) : Prop :=
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
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
      io.write pid₀ pid₁ j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ t j) = ys t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ xs ys j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMasked2DKernelIO₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, mirroring
`MetaMasked2DKernelIO₁ₓ₂.ImplementsR.intro` (no core embedding: the core is
`exec`-based and its typed-output list is `ChanTy`-only, so the streamed
rounding surface is transported here directly via `execR_flatten` and
`flattenState_readMemAs`). Obligations in the skin's named vocabulary:
`FlattenOk`, the `TraceSafeR R` safety walk `hts` (fed the two pinned input
streams and the three window-bound groups), and the region-model rounded
Hoare triple `hrun` (termination under `execR R`, the `readMemAs outDType`
rounded readback of the terminal store, and the single-output frame; the
`undef` pin is threaded in for masked loads without an `other=` default).
`hrun` is where the consumer runs its `forRange` invariant argument — the
skin does not prove the loop. -/
theorem ImplementsR.intro (io : StreamMasked2DKernelIO₂) {R : RoundingModel}
    {f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.C → ℝ}
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
      (∀ j : Fin io.C, io.writeMask (s.pids 0) (s.pids 1) j →
        io.write (s.pids 0) (s.pids 1) j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp1 (io.read1 (s₀.pids 0) (s₀.pids 1) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp2 (io.read2 (s₀.pids 0) (s₀.pids 1) t j) = ys t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) j)
              = io.outDType.ofReal
                  (R.round io.outDType (f (s₀.pids 0) (s₀.pids 1) xs ys j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.C, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) j) →
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
  · intro j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) j < A.extent io.out :=
      hbw j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
    exact hval j hj
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
            refine Or.inr fun j hj => ?_
            rcases hcond with hflat | hn
            · exact absurd rfl hflat
            · intro hoj
              exact hn j hj (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamMasked2DKernelIO₂

/-- IO signature of the **single-stream fold** shape (streaming genre, style
S1: fold + terminal store): the single-stream narrowing of
`StreamMasked2DKernelIO₂` — a 2-D pid grid, **one streamed float input
channel** (`inp1`) read in `T` loop steps of `B1` lanes each, and **one
output channel** (`out`, `C` lanes) written once after the loop. The kernel
folds the `T` per-step tiles into a register accumulator; the spec `f` eats
the *whole* curried stream `(Fin T → Fin B1 → ℝ)` and returns the terminal
value directly — the fold is expressed inside the consumer's `f` (a
reduction consumer writes `∑ t, …`), and the skin does not prove the loop
for the consumer: the `ImplementsR.intro` obligation `hrun` is discharged
with a `forRange`-invariant argument on the consumer's side. Intended
consumers: the single-input streamed-reduction family (mean / sum / norm
over a long axis). Every field is the verbatim `StreamMasked2DKernelIO₂`
field with the `inp2`/`B2`/`read2`/`mask2` channel removed; the skin also
carries the genre's **single-surface design** unchanged (only `⊨[R]`, with
the `outDType := .real` default recovering the exact genre losslessly — see
the ₂ structure's design note). -/
structure StreamMasked2DKernelIO₁ where
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
  /-- Tile length of the terminal output store. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  terminal boundary store. `⊨[R]`'s postcondition reads the output back as
  `outDType`-typed cells holding `outDType.ofReal (R.round outDType (f …))`;
  an fp16-storing streaming kernel sets `.fp16`. `.real` (the default) is an
  unrounded store — exact under `execR R`, so the default recovers the
  exact genre. -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program `(pid₀, pid₁)`. -/
  read1 : Nat → Nat → Fin T → Fin B1 → Nat
  /-- Lane `j`'s terminal write address. -/
  write : Nat → Nat → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Fin T → Fin B1 → Prop
  /-- The terminal store's write-active lanes. -/
  writeMask : Nat → Nat → Fin C → Prop

namespace StreamMasked2DKernelIO₁

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the single-stream fold skin, the genre's only surface (see
the `StreamMasked2DKernelIO₂` structure's single-surface design note). Same
full Hoare triple as the family's `ImplementsR` relations (∀ disjoint
allocation, ∀ pids, ∀ launch state with the masked input stream loaded
exact-ℝ step by step), the execution is `execR R`, and the single output
readback is the `KernelIO₂.ImplementsR` contract per lane, generalized to
the streamed input: the terminal cell holds the *ideal* real value
`f pid₀ pid₁ xs j` of the whole `T`-step fold, quantized **once** at the
declared grid `outDType` — read back through `readMemAs io.outDType` as
`io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat cell
outside the write-active output window is untouched. At `R := .triv` and
`outDType := .real` the store is exact and this is the exact streaming
contract. -/
def ImplementsR (io : StreamMasked2DKernelIO₁) (R : RoundingModel)
    (f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) → Fin io.C → ℝ) : Prop :=
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
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
      io.write pid₀ pid₁ j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ t j) = xs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ xs j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMasked2DKernelIO₁.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the single-stream
narrowing of `StreamMasked2DKernelIO₂.ImplementsR.intro` (no core embedding:
the streamed rounding surface is transported directly via `execR_flatten`
and `flattenState_readMemAs`). Obligations in the skin's named vocabulary:
`FlattenOk`, the `TraceSafeR R` safety walk `hts` (fed the pinned input
stream and the two window-bound groups), and the region-model rounded Hoare
triple `hrun` (termination under `execR R`, the `readMemAs outDType` rounded
readback of the terminal store, and the single-output frame; the `undef`
pin is threaded in for masked loads without an `other=` default). `hrun` is
where the consumer runs its `forRange` invariant argument — the skin does
not prove the loop. -/
theorem ImplementsR.intro (io : StreamMasked2DKernelIO₁) {R : RoundingModel}
    {f : Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ),
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
        s.readMem io.inp1 (io.read1 (s.pids 0) (s.pids 1) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s.pids 0) (s.pids 1) t j →
        io.read1 (s.pids 0) (s.pids 1) t j < bounds io.inp1) →
      (∀ j : Fin io.C, io.writeMask (s.pids 0) (s.pids 1) j →
        io.write (s.pids 0) (s.pids 1) j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 (s₀.pids 0) (s₀.pids 1) t j →
        s₀.readMem io.inp1 (io.read1 (s₀.pids 0) (s₀.pids 1) t j) = xs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) j)
              = io.outDType.ofReal
                  (R.round io.outDType (f (s₀.pids 0) (s₀.pids 1) xs j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.C, io.writeMask (s₀.pids 0) (s₀.pids 1) j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) j) →
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
  · intro j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) j < A.extent io.out :=
      hbw j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
    exact hval j hj
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
            refine Or.inr fun j hj => ?_
            rcases hcond with hflat | hn
            · exact absurd rfl hflat
            · intro hoj
              exact hn j hj (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamMasked2DKernelIO₁

/-! ### The metadata streaming skin: `StreamMetaMasked3DKernelIO₂`

`Stream` + `Meta` + `Masked` + `3D` stack as capability prefixes; the
subscript stays pure data-input × output arity (two streamed data inputs ↦
one output — the slot count `nMeta`, the trip count `T` and the lane counts
are fields, never name subscripts). -/

/-- IO signature of the **metadata-parametrized two-stream fold** shape
(streaming genre, style S1: fold + terminal store, on a **3-D pid grid**):
the `StreamMasked2DKernelIO₂` genre extended with a **pre-loop scalar
metadata slot vector**. Before its `forRange` loop the kernel loads one
scalar from each of `nMeta` metadata regions — a batch start offset, a
sequence length, an adapter index — and every streamed read window / mask,
the terminal write window / mask and the spec `f` are parametrized by the
loaded slot vector. Intended consumers: the LoRA / sequence-metadata gemv
family (sgmv / bgmv shrink-expand), whose per-program work item is
described entirely by such scalars. Slots are heterogeneous: slot `k` has
element type `sty k : ChanTy` (`.nat` lengths and `.int` adapter indices
coexist in one skin) and lives in its own region `mbuf k` (slot regions
may pairwise differ). Following the family precedent there is no `scratch`
field until a consumer needs one.

**Slot causality.** The slot windows `mwin` eat only the three pids —
never other slots' values — so the pinned slot context is well-founded by
construction and no load-order bookkeeping is needed. A kernel whose slot
address depends on a previously loaded slot (chained indirection) is out
of this skin's scope; a `Chain` variant will be added when such a consumer
appears.

**Sentinel gating.** `writeMask` eats the slot vector `m`, so the family's
skip-sentinel idiom (adapter index `= -1` ⇒ the program stores nothing) is
expressed as an empty write-active window at the sentinel value: the
readback leg is then vacuous at that program and the frame keeps the whole
output untouched.

**Single-surface design.** As everywhere in the streaming genre the skin
carries only the rounding-correctness relation `⊨[R]` (`ImplementsR`,
`execR R`) — a streaming kernel's terminal store is routinely a
narrow-float typed store, and with the `outDType := .real` default the
single `⊨[R]` surface covers the exact genre losslessly (the exact surface
is this relation's `R := .triv` degeneration) — see the
`StreamMasked2DKernelIO₂` design note. -/
structure StreamMetaMasked3DKernelIO₂ where
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
  /-- Number of streaming steps (the `forRange` trip count). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Tile length of the terminal output store. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  terminal boundary store. `⊨[R]`'s postcondition reads the output back as
  `outDType`-typed cells holding `outDType.ofReal (R.round outDType (f …))`;
  an fp16-storing streaming kernel sets `.fp16`. `.real` (the default) is an
  unrounded store — exact under `execR R`, so the default recovers the
  exact genre. -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program
  `(pid₀, pid₁, pid₂)`, given the loaded slot vector. -/
  read1 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address, given the loaded slot
  vector. -/
  read2 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B2 → Nat
  /-- Lane `j`'s terminal write address, given the loaded slot vector. -/
  write : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`, given the loaded slot vector
  (a sequence-length slot bounds the live steps). -/
  mask1 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`, given the loaded slot
  vector. -/
  mask2 : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin T → Fin B2 → Prop
  /-- The terminal store's write-active lanes, given the loaded slot
  vector — the sentinel gate (see the sentinel-gating design note). -/
  writeMask : Nat → Nat → Nat → (∀ k : Fin nMeta, (sty k).carrier) →
    Fin C → Prop

namespace StreamMetaMasked3DKernelIO₂

/-- The pinned slot-value context: one loaded scalar per metadata slot, at
the slot's own `ChanTy` carrier. The structure's window/mask fields spell
this Pi type out verbatim (a field type cannot mention the abbreviation of
the structure it lives in); consumers and the `⊨[R]` surface use the
abbreviation. -/
abbrev Meta (io : StreamMetaMasked3DKernelIO₂) : Type :=
  ∀ k : Fin io.nMeta, (io.sty k).carrier

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the metadata-parametrized two-stream fold skin, the
streaming genre's only surface (see the structure's single-surface design
note). Same full Hoare triple as `StreamMasked2DKernelIO₂.ImplementsR` —
∀ disjoint allocation (the slot regions listed before the data regions),
∀ pids (three of them, the 3-D grid), ∀ launch state — extended with the
slot context: the slot vector `m : io.Meta` is universally quantified and
pinned cell by cell through each slot's own `ChanTy` readback **before**
the streamed inputs are pinned, so the streamed windows, the masks and the
spec all speak about the *loaded* metadata. The execution is `execR R` and
the single output readback is the streamed contract per write-active lane:
the terminal cell holds the *ideal* real value `f pid₀ pid₁ pid₂ m xs ys j`
of the whole `T`-step fold, quantized **once** at the declared grid
`outDType` — read back through `readMemAs io.outDType` as
`io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat cell
outside the write-active output window is untouched. At `R := .triv` and
`outDType := .real` the store is exact and this is the exact streaming
contract. -/
def ImplementsR (io : StreamMetaMasked3DKernelIO₂) (R : RoundingModel)
    (f : Nat → Nat → Nat → io.Meta → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = List.ofFn io.mbuf ++ [io.inp1, io.inp2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (m : io.Meta) (xs : Fin io.T → Fin io.B1 → ℝ)
    (ys : Fin io.T → Fin io.B2 → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ k : Fin io.nMeta,
      io.mwin k pid₀ pid₁ pid₂ < A.extent (io.mbuf k)) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ m t j →
      io.read1 pid₀ pid₁ pid₂ m t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ m t j →
      io.read2 pid₀ pid₁ pid₂ m t j < A.extent io.inp2) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
      io.write pid₀ pid₁ pid₂ m j < A.extent io.out) →
    (∀ k : Fin io.nMeta,
      (io.sty k).read s₀ (io.mbuf k) (io.mwin k pid₀ pid₁ pid₂) = m k) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ m t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ m t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ m t j) = ys t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ m j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ pid₂ m xs ys j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ m j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ m j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMetaMasked3DKernelIO₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, mirroring
`StreamMasked2DKernelIO₂.ImplementsR.intro` (no core embedding: the
streamed rounding surface is transported directly via `execR_flatten` and
`flattenState_readMemAs`). Obligations in the skin's named vocabulary:
`FlattenOk`, the `TraceSafeR R` safety walk `hts` (fed the pinned slot
vector, the two pinned input streams and the four window-bound groups —
slots first), and the region-model rounded Hoare triple `hrun`
(termination under `execR R`, the `readMemAs outDType` rounded readback of
the terminal store, and the single-output frame; the `undef` pin is
threaded in for masked loads without an `other=` default). Both `hts` and
`hrun` receive the slot vector `m` with its per-slot `ChanTy` pins — the
slot pins are stated on the *region-model* state, so they pass to the
obligations verbatim, no flattening transport needed. `hrun` is where the
consumer runs its `forRange` invariant argument — the skin does not prove
the loop. -/
theorem ImplementsR.intro (io : StreamMetaMasked3DKernelIO₂)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → io.Meta → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m : io.Meta)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ),
      (∀ k : Fin io.nMeta,
        (io.sty k).read s (io.mbuf k)
          (io.mwin k (s.pids 0) (s.pids 1) (s.pids 2)) = m k) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) m t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) m t j) = ys t j) →
      (∀ k : Fin io.nMeta,
        io.mwin k (s.pids 0) (s.pids 1) (s.pids 2) < bounds (io.mbuf k)) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) m t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) m t j
          < bounds io.inp2) →
      (∀ j : Fin io.C, io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) m j →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) m j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (m : io.Meta)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ k : Fin io.nMeta,
        (io.sty k).read s₀ (io.mbuf k)
          (io.mwin k (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)) = m k) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m t j)
          = ys t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m xs ys j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.C,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j) →
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
  · intro j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) m j
        < A.extent io.out := hbw j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
    exact hval j hj
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
            refine Or.inr fun j hj => ?_
            rcases hcond with hflat | hn
            · exact absurd rfl hflat
            · intro hoj
              exact hn j hj (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamMetaMasked3DKernelIO₂

end VeriTile.Triton
