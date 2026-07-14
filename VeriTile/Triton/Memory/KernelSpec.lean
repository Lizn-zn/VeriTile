/-
VeriTile.Triton.Memory.KernelSpec

The **KernelIO spec surface**: a kernel's headline correctness statement is
`io ⊨ f` — "the kernel described by the IO signature `io` implements the
mathematical function `f`" — a full Hoare triple packaged as one audited
definition:

* **Precondition** (all universally quantified): any program id whose window
  is in bounds; any disjoint flat allocation of the declared buffers (∀ base
  pointers — the allocator contract); any launch state whose two input
  windows hold the input arrays, with **every other allocated cell and every
  register arbitrary** (junk).
* **Postcondition**: the translated pointer kernel terminates; the output
  window holds `f` of the inputs; **every flat cell outside the output
  window is unchanged** (frame).

The per-kernel statement surface is just the `KernelIO₂` value (which buffer
is which argument, buffer lengths, the per-program window) plus `f` — pure
mathematics. `Implements` is the audit-once combinator.

**Modeling boundary (read before trusting)**: the launch state is
`A.flattenState s₀` for an arbitrary region state `s₀`. Concretely this
means: every cell of every *allocated* buffer is arbitrary (in particular
the whole output buffer and the parts of the input buffers outside the
program's window), and the register file is arbitrary (translated). What is
*not* quantified: flat addresses outside every allocated buffer read as the
default cell (unallocated memory is unobservable — the kernel is
trace-safe, so it never touches it), and the `undef` bookkeeping channel is
launch-clean. Strengthening "unallocated cells" to full junk is the
execution-locality upgrade (`Memory/Locality.lean`) and can be layered on
per kernel.
-/

import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Memory.Denotation

namespace VeriTile.Triton

/-- IO signature of a two-input / one-output kernel with uniform tile length
`B`: which buffer is which argument, where each program instance **reads**
its two input tiles, and where it **writes** its output tile. The read
windows are the address half of the precondition, the write window the
address half of the postcondition. Buffer sizes are **not** part of the
signature — they belong to the allocation, and `Implements` quantifies over
every allocation large enough to contain the windows. This is the whole
kernel-specific audit surface of an `io ⊨ f` headline. -/
structure KernelIO₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- First input buffer. -/
  in1 : RegionName
  /-- Second input buffer. -/
  in2 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance reads/writes `B`-element windows. -/
  B : Nat
  /-- Where program `pid` reads its `in1` tile: `[read1 pid, read1 pid + B)`. -/
  read1 : Nat → Nat
  /-- Where program `pid` reads its `in2` tile. -/
  read2 : Nat → Nat
  /-- Where program `pid` writes its output tile. -/
  write : Nat → Nat

namespace KernelIO₂

/-- `io.Implements f` — the kernel of `io` implements the mathematical
function `f` on its declared IO signature. Full Hoare triple; see the module
docstring for exactly what is quantified. -/
def Implements (io : KernelIO₂)
    (f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    -- ∀ base pointers: any disjoint allocation of exactly the three buffers
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    -- the program's window fits inside each allocated buffer
    io.read1 pid + io.B ≤ A.extent io.in1 →
    io.read2 pid + io.B ≤ A.extent io.in2 →
    io.write pid + io.B ≤ A.extent io.out →
  ∀ (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    -- the launch state: program id set, undef launch-clean, input windows
    -- loaded; everything else in s₀ (all buffer cells, registers) arbitrary
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B, s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    ∃ s',
      -- termination of the translated pointer kernel …
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      -- … the output window holds f …
      ∧ (∀ j : Fin io.B,
          s'.readMem A.flat (A.addr io.out (io.write pid + j.val))
            = f xs ys j)
      -- … and every other cell is untouched (frame)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ j : Fin io.B, o' ≠ A.addr io.out (io.write pid + j.val)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => KernelIO₂.Implements

/-- Assembly lemma: `io ⊨ f` from three per-kernel obligations —
`FlattenOk` (bridge fragment membership), `TraceSafe` (the per-execution
safety walk, taking the window-in-bounds contract), and the region-model
Hoare triple `hrun` (termination + output values + frame, all against the
region model — the mathematical core). The flat-memory transport is done
here, once. -/
theorem Implements.intro (io : KernelIO₂)
    {f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read1 s.pid + io.B ≤ bounds io.in1 →
      io.read2 s.pid + io.B ≤ bounds io.in2 →
      io.write s.pid + io.B ≤ bounds io.out →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B, s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B,
            s1.readMem io.out (io.write s₀.pid + j.val) = f xs ys j)
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.B, o ≠ io.write s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  intro A hd hregs hcov pid h1 h2 h3 xs ys s₀ hpid hu hx hy
  subst hpid
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys hx hy
  have hts' : Kernel.TraceSafe A.extent (io.kernel.toAlgKernel) s₀ :=
    hts A.extent s₀ h1 h2 h3
  have hbridge := A.exec_flatten hd hcov _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j
    have hmem : io.out ∈ A.regions := by rw [hregs]; simp
    have hlt : io.write s₀.pid + j.val < A.extent io.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval j
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
            refine Or.inr fun j hoj => ?_
            rcases hcond with hflat | hnadr
            · exact hflat rfl
            · exact hnadr j (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end KernelIO₂

/-- IO signature of a **one-input / one-output** kernel. Unlike `KernelIO₂`,
the input and output tile lengths are independent (`Bin`/`Bout`) — this
covers whole-tile maps (softmax: `Bin = Bout = B`) as well as reductions
that write a single cell per program (LSE, row-wise sum/max:
`Bout = 1`). Same reading as `KernelIO₂`: `read` is the address half of
the precondition, `write` of the postcondition; buffer sizes are not
signature content. -/
structure KernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- Input buffer. -/
  inp : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Input tile length. -/
  Bin : Nat
  /-- Output tile length (`1` for scalar-per-program reductions). -/
  Bout : Nat
  /-- Where program `pid` reads its input tile: `[read pid, read pid + Bin)`. -/
  read : Nat → Nat
  /-- Where program `pid` writes its output tile. -/
  write : Nat → Nat

namespace KernelIO₁

/-- `io.Implements f` — one-input sibling of `KernelIO₂.Implements`; see
the module docstring for exactly what is quantified. -/
def Implements (io : KernelIO₁)
    (f : (Fin io.Bin → ℝ) → Fin io.Bout → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.read pid + io.Bin ≤ A.extent io.inp →
    io.write pid + io.Bout ≤ A.extent io.out →
  ∀ (xs : Fin io.Bin → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.Bin, s₀.readMem io.inp (io.read pid + j.val) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.Bout,
          s'.readMem A.flat (A.addr io.out (io.write pid + j.val))
            = f xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ j : Fin io.Bout, o' ≠ A.addr io.out (io.write pid + j.val)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => KernelIO₁.Implements

/-- Assembly lemma — one-input sibling of `KernelIO₂.Implements.intro`. -/
theorem Implements.intro (io : KernelIO₁)
    {f : (Fin io.Bin → ℝ) → Fin io.Bout → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read s.pid + io.Bin ≤ bounds io.inp →
      io.write s.pid + io.Bout ≤ bounds io.out →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.Bin → ℝ),
      (∀ j : Fin io.Bin, s₀.readMem io.inp (io.read s₀.pid + j.val) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.Bout,
            s1.readMem io.out (io.write s₀.pid + j.val) = f xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.Bout, o ≠ io.write s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  intro A hd hregs hcov pid h1 h2 xs s₀ hpid hu hx
  subst hpid
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs hx
  have hts' : Kernel.TraceSafe A.extent (io.kernel.toAlgKernel) s₀ :=
    hts A.extent s₀ h1 h2
  have hbridge := A.exec_flatten hd hcov _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j
    have hmem : io.out ∈ A.regions := by rw [hregs]; simp
    have hlt : io.write s₀.pid + j.val < A.extent io.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt]
    exact hval j
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
            refine Or.inr fun j hoj => ?_
            rcases hcond with hflat | hnadr
            · exact hflat rfl
            · exact hnadr j (by rw [hoeq, hoj])
          · exact Or.inl hro
    · simp only [FlatAlloc.flattenState, if_neg hr]

end KernelIO₁

end VeriTile.Triton
