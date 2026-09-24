/-
Kernel IO contracts: Base.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.Flatten
import VeriTile.Triton.Memory.FlattenR
import VeriTile.Triton.Memory.Denotation
import VeriTile.Triton.Memory.KernelCore

namespace VeriTile.Triton

namespace UKernelIO

/-! ### The unified core's rounding sibling

`UKernelIO.Implements` (KernelCore) is the exact-ℝ Hoare triple that every
**single-shot** skin in this file assembles through. `ImplementsR` below is
its `execR R` sibling, standing to it exactly as the `Stream*` family's
`ImplementsR` stands to their `Implements`: the kernel runs under the
rounding model, the safety obligation becomes `Kernel.TraceSafeR R`, and
each output cell is read back through `readMemAs` at a declared
`FloatDType` grid holding the *ideal* real value quantized **once**.

Two deliberate restrictions, both stated rather than worked around:

* **Float outputs only.** `f`'s codomain is plain `ℝ` and the readback goes
  through `readMemAs (od o)`, so this relation serves skins whose output
  channels are `ChanTy.float`. Skins that write an integer channel (the
  value-plus-index pair genre) keep the exact `Implements` surface — a
  rounding grid is meaningless there, and faking one would be worse than
  not offering it.
* **The output grid is a parameter, not a field.** Passing `od` here keeps
  `UKernelIO` itself unchanged, so all existing `toU` embeddings and their
  exact `Implements` proofs are untouched.

It lives in this file rather than beside the structure because it needs the
`FlattenR` bridge, which `KernelCore` does not import. -/
def ImplementsR (io : UKernelIO) (R : RoundingModel)
    (od : Fin io.nOut → FloatDType)
    (f : Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → ℝ) :
    Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = io.bufs →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (vals : io.Ctx) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
      io.imask i vals pid₀ pid₁ pid₂ j →
      io.iwin i vals pid₀ pid₁ pid₂ j < A.extent (io.ibuf i)) →
    (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
      io.omask o vals pid₀ pid₁ pid₂ j →
      io.owin o vals pid₀ pid₁ pid₂ j < A.extent (io.obuf o)) →
    (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
      io.smask t vals pid₀ pid₁ pid₂ j →
      io.swin t vals pid₀ pid₁ pid₂ j < A.extent (io.sbuf t)) →
    (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
      io.imask i vals pid₀ pid₁ pid₂ j →
      (io.ity i).read s₀ (io.ibuf i)
          (io.iwin i vals pid₀ pid₁ pid₂ j) = vals i j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
          io.omask o vals pid₀ pid₁ pid₂ j →
          s'.readMemAs (od o) A.flat
              (A.addr (io.obuf o) (io.owin o vals pid₀ pid₁ pid₂ j))
            = (od o).ofReal (R.round (od o) (f pid₀ pid₁ vals o j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
                io.omask o vals pid₀ pid₁ pid₂ j →
                o' ≠ A.addr (io.obuf o) (io.owin o vals pid₀ pid₁ pid₂ j)) ∧
             (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
                io.smask t vals pid₀ pid₁ pid₂ j →
                o' ≠ A.addr (io.sbuf t) (io.swin t vals pid₀ pid₁ pid₂ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

/-- Assembly lemma — the rounding sibling of `UKernelIO.Implements.intro`,
and **the** flattening bridge for every single-shot skin's `⊨[R]` face.
Obligations are the family's usual three: `FlattenOk`, the `TraceSafeR R`
safety walk, and the region-model rounded Hoare triple. -/
theorem ImplementsR.intro (io : UKernelIO) {R : RoundingModel}
    {od : Fin io.nOut → FloatDType}
    {f : Nat → Nat → io.Ctx → (o : Fin io.nOut) → Fin (io.oarity o) → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (vals : io.Ctx),
      (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
        io.imask i vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        (io.ity i).read s (io.ibuf i)
            (io.iwin i vals (s.pids 0) (s.pids 1) (s.pids 2) j) = vals i j) →
      (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
        io.imask i vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.iwin i vals (s.pids 0) (s.pids 1) (s.pids 2) j
          < bounds (io.ibuf i)) →
      (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
        io.omask o vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.owin o vals (s.pids 0) (s.pids 1) (s.pids 2) j
          < bounds (io.obuf o)) →
      (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
        io.smask t vals (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.swin t vals (s.pids 0) (s.pids 1) (s.pids 2) j
          < bounds (io.sbuf t)) →
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (vals : io.Ctx),
      s₀.undef = (fun _ _ => 0) →
      (∀ (i : Fin io.nIn) (j : Fin (io.iarity i)),
        io.imask i vals (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
        (io.ity i).read s₀ (io.ibuf i)
            (io.iwin i vals (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
          = vals i j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ (o : Fin io.nOut) (j : Fin (io.oarity o)),
            io.omask o vals (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs (od o) (io.obuf o)
                (io.owin o vals (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = (od o).ofReal
                  (R.round (od o) (f (s₀.pids 0) (s₀.pids 1) vals o j)))
        ∧ (∀ r o',
            (∀ (oc : Fin io.nOut) (j : Fin (io.oarity oc)),
              io.omask oc vals (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
              r ≠ io.obuf oc ∨
                o' ≠ io.owin oc vals (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) →
            (∀ (t : Fin io.nScr) (j : Fin (io.sarity t)),
              io.smask t vals (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
              r ≠ io.sbuf t ∨
                o' ≠ io.swin t vals (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) →
            s1.mem r o' = s₀.mem r o')) :
    io.ImplementsR R od f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ vals s₀ hpid₀ hpid₁ hpid₂ hu
    hib hob hsb hpins
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ vals hu hpins
  have hts' : Kernel.TraceSafeR R A.extent (io.kernel.toAlgKernel) s₀ :=
    hts A.extent s₀ vals hpins hib hob hsb
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro o j hj
    have hmem : io.obuf o ∈ A.regions := by
      rw [hregs]; exact io.obuf_mem o
    have hlt : io.owin o vals (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
        < A.extent (io.obuf o) := hob o j hj
    rw [A.flattenState_readMemAs hd s1 hmem hlt (od o)]
    exact hval o j hj
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
          refine congrArg A.trCell
            (hframe r o (fun oc j hj => ?_) (fun t j hj => ?_))
          · by_cases hro : r = io.obuf oc
            · subst hro
              refine Or.inr fun hoj => ?_
              rcases hcond with hflat | hn
              · exact hflat rfl
              · exact hn.1 oc j hj (by rw [hoeq, hoj])
            · exact Or.inl hro
          · by_cases hrs : r = io.sbuf t
            · subst hrs
              refine Or.inr fun hoj => ?_
              rcases hcond with hflat | hn
              · exact hflat rfl
              · exact hn.2 t j hj (by rw [hoeq, hoj])
            · exact Or.inl hrs
    · simp only [FlatAlloc.flattenState, if_neg hr]

end UKernelIO

/-- One **private working buffer** of an unmasked kernel: program `pid` may
stage intermediates in the window `[win pid, win pid + len)` of buffer
`buf`. Scratch buffers are allocated and writable, but their post-state is
not part of any contract: `Implements` and `Equiv` exclude them from the
frame, and `Equiv` never compares them. (The masked structs carry their own
scratch shape instead — there the active lane set comes from the kernel's
`mask`.) -/
structure ScratchSpec where
  /-- The buffer. -/
  buf : RegionName
  /-- Where program `pid`'s scratch window starts. -/
  win : Nat → Nat
  /-- The scratch window length. -/
  len : Nat


end VeriTile.Triton
