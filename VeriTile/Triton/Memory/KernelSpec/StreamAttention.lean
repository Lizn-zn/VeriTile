/-
Kernel IO contracts: StreamAttention.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton

/-! ### The attention fold skin: `StreamMasked3DKernelIO₃ₓ₂`

Style S1 (fold + terminal store) at data arity 3×2 on the 3-D pid grid —
the online-softmax attention shape: three streamed float inputs (a static
`Q` tile plus per-step `K`/`V` tiles), a `T`-step register fold
(max/denominator/accumulator), and **two terminal stores of different
widths and different dtypes** (an `O` tile, routinely `.to(tl.float16)`,
and an `L`/`M` statistics row kept unrounded). Named fields, per-output
`C`/`outDType` — the ≤3-output attention genre does not need the grouped
vector-channel treatment.

**Pid-dependent trip counts** (a causal kernel's KV walk stops at
`(pid₀+1)·BLOCK_M`) ride the fixed `T` field as an upper bound: the masks
eat all three pids, so `read`/`mask` simply gate the live steps
per-program — no skin support needed. A channel whose read window ignores
`t` is the genre's static stream. -/

/-- IO signature of the **three-stream fold, two terminal outputs** shape
(streaming genre, style S1 on a **3-D pid grid**): three streamed float
input channels read in `T` loop steps of `B1`/`B2`/`B3` lanes each, and
two output channels written once after the loop — `out1` (`C1` lanes, its
own `out1DType` grid) and `out2` (`C2` lanes, `out2DType`). The spec `f`
eats the whole curried streams and returns both terminal tiles as a pair;
the fold is expressed inside the consumer's `f`, and the skin does not
prove the loop (`ImplementsR.intro`'s `hrun` is the consumer's
`forRange`-invariant obligation). Intended consumers: the online-softmax
attention family (fp16 `O` + unrounded `L`/`M` statistics — mixed output
dtypes are per-field, following the `MetaMasked2DKernelIO₁ₓ₂` precedent).

**Single-surface design** as everywhere in the streaming genre: only
`⊨[R]`, with the `.real` defaults recovering the exact genre losslessly —
see the `StreamMasked2DKernelIO₂` design note. -/
structure StreamMasked3DKernelIO₃ₓ₂ where
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
  /-- First output buffer. -/
  out1 : RegionName
  /-- Second output buffer. -/
  out2 : RegionName
  /-- Number of streaming steps (the `forRange` trip count; a
  pid-dependent walk sets the pid-free upper bound here and gates the
  live steps in the masks). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the third input channel. -/
  B3 : Nat
  /-- Tile length of the first terminal store. -/
  C1 : Nat
  /-- Tile length of the second terminal store. -/
  C2 : Nat
  /-- `out1`'s floating dtype — the quantization grid of its terminal
  boundary store (an fp16-storing attention `O` sets `.fp16`; `.real`,
  the default, is exact under `execR R`). -/
  out1DType : FloatDType := .real
  /-- `out2`'s floating dtype (the `L`/`M` statistics rows are routinely
  unrounded `.real`). -/
  out2DType : FloatDType := .real
  /-- The launch-legality precondition (defaulted unconstrained; the
  `StreamMasked3DKernelIO₃ₓ₃` precedent): the triple is claimed only for
  pids satisfying `pre`. Besides that skin's pid-dependent-trip-count
  case, the other genuine need is a **wrapping store**: a kernel whose
  store mask tests already-wrapped (`% M`) offsets still writes at every
  pid, so an out-of-fit program overwrites in-range cells — folding the
  fit condition into `writeMask` would leave the frame clause false.
  Consumers instantiate `pre` with the host-grid fit facts their exact
  stack already carries (triton_linear_activation's `hFitM`/`hFitN`). -/
  pre : Nat → Nat → Nat → Prop := fun _ _ _ => True
  /-- Step `t`, lane `j`'s `inp1` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read1 : Nat → Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address. -/
  read2 : Nat → Nat → Nat → Fin T → Fin B2 → Nat
  /-- Step `t`, lane `j`'s `inp3` read address. -/
  read3 : Nat → Nat → Nat → Fin T → Fin B3 → Nat
  /-- Lane `j`'s `out1` terminal write address. -/
  write1 : Nat → Nat → Nat → Fin C1 → Nat
  /-- Lane `j`'s `out2` terminal write address. -/
  write2 : Nat → Nat → Nat → Fin C2 → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Nat → Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`. -/
  mask2 : Nat → Nat → Nat → Fin T → Fin B2 → Prop
  /-- `inp3`'s read-active lanes at step `t`. -/
  mask3 : Nat → Nat → Nat → Fin T → Fin B3 → Prop
  /-- `out1`'s write-active lanes. -/
  writeMask1 : Nat → Nat → Nat → Fin C1 → Prop
  /-- `out2`'s write-active lanes. -/
  writeMask2 : Nat → Nat → Nat → Fin C2 → Prop

namespace StreamMasked3DKernelIO₃ₓ₂

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the three-stream two-output fold skin, the genre's only
surface (see the `StreamMasked2DKernelIO₂` structure's single-surface
design note). Same full Hoare triple as the family's `ImplementsR`
relations (∀ disjoint allocation, ∀ pids — three of them, ∀ launch state
with all three masked input streams loaded exact-ℝ step by step), the
execution is `execR R`, and each output readback is the
`KernelIO₂.ImplementsR` contract per lane at its **own** grid: `out1`
holds `io.out1DType.ofReal (R.round io.out1DType ((f …).1 j))` and `out2`
holds `io.out2DType.ofReal (R.round io.out2DType ((f …).2 j))` — each
terminal cell the *ideal* real value of the whole `T`-step fold,
quantized **once**. Frame: every flat cell outside the two write-active
output windows is untouched. At `R := .triv` and `.real` grids both
stores are exact and this is the exact streaming contract. -/
def ImplementsR (io : StreamMasked3DKernelIO₃ₓ₂) (R : RoundingModel)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.C1 → ℝ) × (Fin io.C2 → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ) (s₀ : BlockState),
    io.pre pid₀ pid₁ pid₂ →
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      io.read1 pid₀ pid₁ pid₂ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      io.read2 pid₀ pid₁ pid₂ t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      io.read3 pid₀ pid₁ pid₂ t j < A.extent io.inp3) →
    (∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
      io.write1 pid₀ pid₁ pid₂ j < A.extent io.out1) →
    (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
      io.write2 pid₀ pid₁ pid₂ j < A.extent io.out2) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = zs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
          s'.readMemAs io.out1DType A.flat
              (A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j))
            = io.out1DType.ofReal
                (R.round io.out1DType
                  ((f pid₀ pid₁ pid₂ xs ys zs).1 j)))
      ∧ (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
          s'.readMemAs io.out2DType A.flat
              (A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j))
            = io.out2DType.ofReal
                (R.round io.out2DType
                  ((f pid₀ pid₁ pid₂ xs ys zs).2 j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j)) ∧
             (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMasked3DKernelIO₃ₓ₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the two-output
widening of `StreamMasked2DKernelIO₂.ImplementsR.intro` on the 3-D grid
(no core embedding). Obligations in the skin's named vocabulary:
`FlattenOk`, the `TraceSafeR R` safety walk `hts` (fed the three pinned
input streams and the five window-bound groups), and the region-model
rounded Hoare triple `hrun` (termination under `execR R`, the per-grid
`readMemAs` rounded readback of both terminal stores, and the two-output
frame stated with per-output region guards; the `undef` pin is threaded
in for masked loads without an `other=` default). `hrun` is where the
consumer runs its `forRange` invariant argument — the skin does not
prove the loop. -/
theorem ImplementsR.intro (io : StreamMasked3DKernelIO₃ₓ₂)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.C1 → ℝ) × (Fin io.C2 → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
        (zs : Fin io.T → Fin io.B3 → ℝ),
      io.pre (s.pids 0) (s.pids 1) (s.pids 2) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp3
            (io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j) = zs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp3) →
      (∀ j : Fin io.C1,
        io.writeMask1 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write1 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out1) →
      (∀ j : Fin io.C2,
        io.writeMask2 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write2 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out2) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ) (zs : Fin io.T → Fin io.B3 → ℝ),
      io.pre (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) →
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp3
            (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = zs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C1,
            io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs io.out1DType io.out1
                (io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = io.out1DType.ofReal
                  (R.round io.out1DType
                    ((f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) xs ys zs).1 j)))
        ∧ (∀ j : Fin io.C2,
            io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs io.out2DType io.out2
                (io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = io.out2DType.ofReal
                  (R.round io.out2DType
                    ((f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) xs ys zs).2 j)))
        ∧ (∀ r oo,
            ((∀ j : Fin io.C1,
                io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                r = io.out1 →
                oo ≠ io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) ∧
             (∀ j : Fin io.C2,
                io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                r = io.out2 →
                oo ≠ io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)) →
            s1.mem r oo = s₀.mem r oo)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ xs ys zs s₀ hpre hpid₀ hpid₁ hpid₂ hu
    hbr1 hbr2 hbr3 hbw1 hbw2 hx hy hz
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval1, hval2, hframe⟩ := hrun s₀ xs ys zs hpre hu hx hy hz
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs ys zs hpre hx hy hz hbr1 hbr2 hbr3 hbw1 hbw2
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem1 : io.out1 ∈ A.regions := by rw [hregs]; simp
  have hmem2 : io.out2 ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
        < A.extent io.out1 := hbw1 j hj
    rw [A.flattenState_readMemAs hd s1 hmem1 hlt io.out1DType]
    exact hval1 j hj
  · intro j hj
    have hlt : io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
        < A.extent io.out2 := hbw2 j hj
    rw [A.flattenState_readMemAs hd s1 hmem2 hlt io.out2DType]
    exact hval2 j hj
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
          refine congrArg A.trCell (hframe r o ⟨?_, ?_⟩)
          · intro j hj hro hoj
            rcases hcond with hflat | ⟨hn1, _⟩
            · exact absurd rfl hflat
            · exact hn1 j hj (by rw [hoeq, hro, hoj])
          · intro j hj hro hoj
            rcases hcond with hflat | ⟨_, hn2⟩
            · exact absurd rfl hflat
            · exact hn2 j hj (by rw [hoeq, hro, hoj])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamMasked3DKernelIO₃ₓ₂

/-- IO signature of the **six-stream fold, two terminal outputs** shape
(streaming genre, style S1 on a **3-D pid grid**): the six-input
widening of `StreamMasked3DKernelIO₃ₓ₂`. Every field is the verbatim
₃ₓ₂ field with three more input channels; per-channel widths may be
radically non-uniform and a channel whose window ignores `t` is the
genre's static stream, both exactly as in the narrower members.

**Aliasing.** This is the family's first genre whose consumer routinely
lists the *same* region as both an input and an output — the resumable
attention shape, where `M`/`Out` carry the incoming running state and
are then overwritten with the outgoing one. That is sound and needs no
extra field: `FlatAlloc.base` is a function of the region *name* (not a
prefix-sum over `regions`), so a repeated name denotes the same segment
rather than two; `Disjoint` only constrains `r ≠ r'`; and `decode`
resolves by `find?`. Semantically the input pins read `s₀` while the
readbacks read `s'`, so no obligation is confused. Intended consumer:
attention_fwd_triton3 case 4 (Q/K/V plus the `M`/`L`/`Out` resume
channels). -/
structure StreamMasked3DKernelIO₆ₓ₂ where
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
  /-- Fourth streamed input buffer. -/
  inp4 : RegionName
  /-- Fifth streamed input buffer. -/
  inp5 : RegionName
  /-- Sixth streamed input buffer. -/
  inp6 : RegionName
  /-- First output buffer. -/
  out1 : RegionName
  /-- Second output buffer. -/
  out2 : RegionName
  /-- Number of streaming steps (the loop trip count; a pid-dependent
  walk sets the pid-free upper bound here and gates the live steps in
  the masks). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the third input channel. -/
  B3 : Nat
  /-- Per-step tile length of the fourth input channel. -/
  B4 : Nat
  /-- Per-step tile length of the fifth input channel. -/
  B5 : Nat
  /-- Per-step tile length of the sixth input channel. -/
  B6 : Nat
  /-- Tile length of the first terminal store. -/
  C1 : Nat
  /-- Tile length of the second terminal store. -/
  C2 : Nat
  /-- `out1`'s floating dtype — the quantization grid of its terminal
  boundary store (`.real`, the default, is exact under `execR R`). -/
  out1DType : FloatDType := .real
  /-- `out2`'s floating dtype. -/
  out2DType : FloatDType := .real
  /-- The launch-legality precondition (defaulted unconstrained; the
  `StreamMasked3DKernelIO₃ₓ₂` precedent). -/
  pre : Nat → Nat → Nat → Prop := fun _ _ _ => True
  /-- Step `t`, lane `j`'s `inp1` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read1 : Nat → Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read2 : Nat → Nat → Nat → Fin T → Fin B2 → Nat
  /-- Step `t`, lane `j`'s `inp3` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read3 : Nat → Nat → Nat → Fin T → Fin B3 → Nat
  /-- Step `t`, lane `j`'s `inp4` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read4 : Nat → Nat → Nat → Fin T → Fin B4 → Nat
  /-- Step `t`, lane `j`'s `inp5` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read5 : Nat → Nat → Nat → Fin T → Fin B5 → Nat
  /-- Step `t`, lane `j`'s `inp6` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read6 : Nat → Nat → Nat → Fin T → Fin B6 → Nat
  /-- Lane `j`'s `out1` terminal write address. -/
  write1 : Nat → Nat → Nat → Fin C1 → Nat
  /-- Lane `j`'s `out2` terminal write address. -/
  write2 : Nat → Nat → Nat → Fin C2 → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Nat → Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`. -/
  mask2 : Nat → Nat → Nat → Fin T → Fin B2 → Prop
  /-- `inp3`'s read-active lanes at step `t`. -/
  mask3 : Nat → Nat → Nat → Fin T → Fin B3 → Prop
  /-- `inp4`'s read-active lanes at step `t`. -/
  mask4 : Nat → Nat → Nat → Fin T → Fin B4 → Prop
  /-- `inp5`'s read-active lanes at step `t`. -/
  mask5 : Nat → Nat → Nat → Fin T → Fin B5 → Prop
  /-- `inp6`'s read-active lanes at step `t`. -/
  mask6 : Nat → Nat → Nat → Fin T → Fin B6 → Prop
  /-- `out1`'s write-active lanes. -/
  writeMask1 : Nat → Nat → Nat → Fin C1 → Prop
  /-- `out2`'s write-active lanes. -/
  writeMask2 : Nat → Nat → Nat → Fin C2 → Prop

namespace StreamMasked3DKernelIO₆ₓ₂

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the six-stream two-output fold skin, the genre's only
surface. Identical in shape to `StreamMasked3DKernelIO₃ₓ₂.ImplementsR`
with three more pinned input streams: each terminal cell holds the
*ideal* real value of the whole `T`-step fold at its own output's grid,
quantized **once**; frame outside the two write-active windows. -/
def ImplementsR (io : StreamMasked3DKernelIO₆ₓ₂) (R : RoundingModel)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) →
      (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.T → Fin io.B4 → ℝ) →
      (Fin io.T → Fin io.B5 → ℝ) →
      (Fin io.T → Fin io.B6 → ℝ) →
      (Fin io.C1 → ℝ) × (Fin io.C2 → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions
      = [io.inp1, io.inp2, io.inp3, io.inp4, io.inp5, io.inp6, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀
    (x1s : Fin io.T → Fin io.B1 → ℝ)
    (x2s : Fin io.T → Fin io.B2 → ℝ)
    (x3s : Fin io.T → Fin io.B3 → ℝ)
    (x4s : Fin io.T → Fin io.B4 → ℝ)
    (x5s : Fin io.T → Fin io.B5 → ℝ)
    (x6s : Fin io.T → Fin io.B6 → ℝ)
    (s₀ : BlockState),
    io.pre pid₀ pid₁ pid₂ →
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      io.read1 pid₀ pid₁ pid₂ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      io.read2 pid₀ pid₁ pid₂ t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      io.read3 pid₀ pid₁ pid₂ t j < A.extent io.inp3) →
    (∀ (t : Fin io.T) (j : Fin io.B4), io.mask4 pid₀ pid₁ pid₂ t j →
      io.read4 pid₀ pid₁ pid₂ t j < A.extent io.inp4) →
    (∀ (t : Fin io.T) (j : Fin io.B5), io.mask5 pid₀ pid₁ pid₂ t j →
      io.read5 pid₀ pid₁ pid₂ t j < A.extent io.inp5) →
    (∀ (t : Fin io.T) (j : Fin io.B6), io.mask6 pid₀ pid₁ pid₂ t j →
      io.read6 pid₀ pid₁ pid₂ t j < A.extent io.inp6) →
    (∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
      io.write1 pid₀ pid₁ pid₂ j < A.extent io.out1) →
    (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
      io.write2 pid₀ pid₁ pid₂ j < A.extent io.out2) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = x1s t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = x2s t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = x3s t j) →
    (∀ (t : Fin io.T) (j : Fin io.B4), io.mask4 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp4 (io.read4 pid₀ pid₁ pid₂ t j) = x4s t j) →
    (∀ (t : Fin io.T) (j : Fin io.B5), io.mask5 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp5 (io.read5 pid₀ pid₁ pid₂ t j) = x5s t j) →
    (∀ (t : Fin io.T) (j : Fin io.B6), io.mask6 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp6 (io.read6 pid₀ pid₁ pid₂ t j) = x6s t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
          s'.readMemAs io.out1DType A.flat
              (A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j))
            = io.out1DType.ofReal
                (R.round io.out1DType
                  ((f pid₀ pid₁ pid₂ x1s x2s x3s x4s x5s x6s).1 j)))
      ∧ (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
          s'.readMemAs io.out2DType A.flat
              (A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j))
            = io.out2DType.ofReal
                (R.round io.out2DType
                  ((f pid₀ pid₁ pid₂ x1s x2s x3s x4s x5s x6s).2 j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j)) ∧
             (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMasked3DKernelIO₆ₓ₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the six-stream
widening of `StreamMasked3DKernelIO₃ₓ₂.ImplementsR.intro`. `hrun` is
where the consumer runs its loop-invariant argument — the skin does not
prove the loop. -/
theorem ImplementsR.intro (io : StreamMasked3DKernelIO₆ₓ₂)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) → (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) → (Fin io.T → Fin io.B4 → ℝ) → (Fin io.T → Fin io.B5 → ℝ) → (Fin io.T → Fin io.B6 → ℝ) →
      (Fin io.C1 → ℝ) × (Fin io.C2 → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (x1s : Fin io.T → Fin io.B1 → ℝ)
        (x2s : Fin io.T → Fin io.B2 → ℝ)
        (x3s : Fin io.T → Fin io.B3 → ℝ)
        (x4s : Fin io.T → Fin io.B4 → ℝ)
        (x5s : Fin io.T → Fin io.B5 → ℝ)
        (x6s : Fin io.T → Fin io.B6 → ℝ),
      io.pre (s.pids 0) (s.pids 1) (s.pids 2) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j) = x1s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j) = x2s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp3
            (io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j) = x3s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B4),
        io.mask4 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp4
            (io.read4 (s.pids 0) (s.pids 1) (s.pids 2) t j) = x4s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B5),
        io.mask5 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp5
            (io.read5 (s.pids 0) (s.pids 1) (s.pids 2) t j) = x5s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B6),
        io.mask6 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp6
            (io.read6 (s.pids 0) (s.pids 1) (s.pids 2) t j) = x6s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp3) →
      (∀ (t : Fin io.T) (j : Fin io.B4),
        io.mask4 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read4 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp4) →
      (∀ (t : Fin io.T) (j : Fin io.B5),
        io.mask5 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read5 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp5) →
      (∀ (t : Fin io.T) (j : Fin io.B6),
        io.mask6 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read6 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp6) →
      (∀ j : Fin io.C1,
        io.writeMask1 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write1 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out1) →
      (∀ j : Fin io.C2,
        io.writeMask2 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write2 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out2) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState)
        (x1s : Fin io.T → Fin io.B1 → ℝ)
        (x2s : Fin io.T → Fin io.B2 → ℝ)
        (x3s : Fin io.T → Fin io.B3 → ℝ)
        (x4s : Fin io.T → Fin io.B4 → ℝ)
        (x5s : Fin io.T → Fin io.B5 → ℝ)
        (x6s : Fin io.T → Fin io.B6 → ℝ),
      io.pre (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) →
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = x1s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = x2s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp3
            (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = x3s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B4),
        io.mask4 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp4
            (io.read4 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = x4s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B5),
        io.mask5 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp5
            (io.read5 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = x5s t j) →
      (∀ (t : Fin io.T) (j : Fin io.B6),
        io.mask6 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp6
            (io.read6 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = x6s t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C1,
            io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs io.out1DType io.out1
                (io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = io.out1DType.ofReal
                  (R.round io.out1DType
                    ((f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)
                      x1s x2s x3s x4s x5s x6s).1 j)))
        ∧ (∀ j : Fin io.C2,
            io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs io.out2DType io.out2
                (io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = io.out2DType.ofReal
                  (R.round io.out2DType
                    ((f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)
                      x1s x2s x3s x4s x5s x6s).2 j)))
        ∧ (∀ r oo,
            ((∀ j : Fin io.C1,
                io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                r = io.out1 →
                oo ≠ io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) ∧
             (∀ j : Fin io.C2,
                io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                r = io.out2 →
                oo ≠ io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)) →
            s1.mem r oo = s₀.mem r oo)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ x1s x2s x3s x4s x5s x6s s₀ hpre hpid₀ hpid₁ hpid₂
    hu hbr1 hbr2 hbr3 hbr4 hbr5 hbr6 hbw1 hbw2 hx1 hx2 hx3 hx4 hx5 hx6
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
    hrun s₀ x1s x2s x3s x4s x5s x6s hpre hu hx1 hx2 hx3 hx4 hx5 hx6
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ x1s x2s x3s x4s x5s x6s hpre hx1 hx2 hx3 hx4 hx5 hx6
      hbr1 hbr2 hbr3 hbr4 hbr5 hbr6 hbw1 hbw2
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem1 : io.out1 ∈ A.regions := by rw [hregs]; simp
  have hmem2 : io.out2 ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
        < A.extent io.out1 := hbw1 j hj
    rw [A.flattenState_readMemAs hd s1 hmem1 hlt io.out1DType]
    exact hval1 j hj
  · intro j hj
    have hlt : io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
        < A.extent io.out2 := hbw2 j hj
    rw [A.flattenState_readMemAs hd s1 hmem2 hlt io.out2DType]
    exact hval2 j hj
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s1).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some pr =>
          obtain ⟨r, o⟩ := pr
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s1.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe r o ⟨?_, ?_⟩)
          · intro j hj hro hoj
            rcases hcond with hflat | ⟨hn1, _⟩
            · exact absurd rfl hflat
            · exact hn1 j hj (by rw [hoeq, hro, hoj])
          · intro j hj hro hoj
            rcases hcond with hflat | ⟨_, hn2⟩
            · exact absurd rfl hflat
            · exact hn2 j hj (by rw [hoeq, hro, hoj])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamMasked3DKernelIO₆ₓ₂


/-- IO signature of the **three-stream fold, single terminal output**
shape (streaming genre, style S1 on a **3-D pid grid**): the
single-output narrowing of `StreamMasked3DKernelIO₃ₓ₂` — three streamed
float input channels (a static `Q` tile is the `t`-ignoring degenerate
stream) read in `T` loop steps of `B1`/`B2`/`B3` lanes each, and one
output channel (`out`, `C` lanes, its own `outDType` grid) written once
after the loop. Every field is the verbatim `StreamMasked3DKernelIO₃ₓ₂`
field with the `out2` channel removed; the genre's **single-surface
design** and the pid-dependent-trip-count note (`T` as a pid-free upper
bound, masks gating live steps) carry over unchanged. Intended
consumers: the single-store online-softmax attention family (Q·K·V fold,
one `O` store). -/
structure StreamMasked3DKernelIO₃ where
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
  /-- Number of streaming steps (the `forRange` trip count; a
  pid-dependent walk sets the pid-free upper bound here and gates the
  live steps in the masks). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the third input channel. -/
  B3 : Nat
  /-- Tile length of the terminal output store. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  terminal boundary store (an fp16-storing attention `O` sets `.fp16`;
  `.real`, the default, is exact under `execR R`). -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read1 : Nat → Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address. -/
  read2 : Nat → Nat → Nat → Fin T → Fin B2 → Nat
  /-- Step `t`, lane `j`'s `inp3` read address. -/
  read3 : Nat → Nat → Nat → Fin T → Fin B3 → Nat
  /-- Lane `j`'s terminal write address. -/
  write : Nat → Nat → Nat → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Nat → Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`. -/
  mask2 : Nat → Nat → Nat → Fin T → Fin B2 → Prop
  /-- `inp3`'s read-active lanes at step `t`. -/
  mask3 : Nat → Nat → Nat → Fin T → Fin B3 → Prop
  /-- The terminal store's write-active lanes. -/
  writeMask : Nat → Nat → Nat → Fin C → Prop

namespace StreamMasked3DKernelIO₃

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the three-stream single-output fold skin, the genre's
only surface (see the `StreamMasked2DKernelIO₂` structure's
single-surface design note). Same full Hoare triple as
`StreamMasked3DKernelIO₃ₓ₂.ImplementsR` with the second output removed:
the terminal cell holds the *ideal* real value `f pid₀ pid₁ pid₂ xs ys
zs j` of the whole `T`-step fold, quantized **once** at the declared
grid `outDType` — read back through `readMemAs io.outDType` as
`io.outDType.ofReal (R.round io.outDType (f …))`. Frame: every flat cell
outside the write-active output window is untouched. At `R := .triv` and
`outDType := .real` the store is exact and this is the exact streaming
contract. -/
def ImplementsR (io : StreamMasked3DKernelIO₃) (R : RoundingModel)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      io.read1 pid₀ pid₁ pid₂ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      io.read2 pid₀ pid₁ pid₂ t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      io.read3 pid₀ pid₁ pid₂ t j < A.extent io.inp3) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
      io.write pid₀ pid₁ pid₂ j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = zs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ pid₂ xs ys zs j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMasked3DKernelIO₃.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the single-output
narrowing of `StreamMasked3DKernelIO₃ₓ₂.ImplementsR.intro`. Obligations
in the skin's named vocabulary: `FlattenOk`, the `TraceSafeR R` safety
walk `hts` (fed the three pinned input streams and the four window-bound
groups), and the region-model rounded Hoare triple `hrun` (termination
under `execR R`, the `readMemAs outDType` rounded readback of the
terminal store, and the single-output frame; the `undef` pin is threaded
in for masked loads without an `other=` default). `hrun` is where the
consumer runs its `forRange` invariant argument — the skin does not
prove the loop. -/
theorem ImplementsR.intro (io : StreamMasked3DKernelIO₃)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
        (zs : Fin io.T → Fin io.B3 → ℝ),
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp3
            (io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j) = zs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp3) →
      (∀ j : Fin io.C,
        io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ) (zs : Fin io.T → Fin io.B3 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp3
            (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = zs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) xs ys zs j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.C,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ xs ys zs s₀ hpid₀ hpid₁ hpid₂ hu
    hbr1 hbr2 hbr3 hbw hx hy hz
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys zs hu hx hy hz
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs ys zs hx hy hz hbr1 hbr2 hbr3 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
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

end StreamMasked3DKernelIO₃

/-- IO signature of the **four-stream fold, single terminal output**
shape (streaming genre, style S1 on a **3-D pid grid**): the
`StreamMasked3DKernelIO₃` skin widened by a fourth streamed float input
channel. Every field is the verbatim ₃ field plus
`inp4`/`B4`/`read4`/`mask4`; the single-surface design, the static-stream
degeneration (a `t`-ignoring read window) and the pid-free-`T` note carry
over unchanged. A buffer read at **two windows of different shapes**
(e.g. a prologue full tile plus a per-step column out of the same bias
region) packs into ONE channel by widening the lane space — a
`Lane2D`-style `[rows, cols+1]` pack with the extra column carrying the
per-step read — rather than aliasing two channels to one region.
Intended consumers: the bias-augmented single-store attention family
(attention_kernel / attention_kernel_aligned: Q/K/V + two-window-packed
`B0`). -/
structure StreamMasked3DKernelIO₄ where
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
  /-- Fourth streamed input buffer. -/
  inp4 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of streaming steps (the `forRange` trip count; a
  pid-dependent walk sets the pid-free upper bound here and gates the
  live steps in the masks). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the third input channel. -/
  B3 : Nat
  /-- Per-step tile length of the fourth input channel. -/
  B4 : Nat
  /-- Tile length of the terminal output store. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  terminal boundary store (an fp16-storing attention `O` sets `.fp16`;
  `.real`, the default, is exact under `execR R`). -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read1 : Nat → Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address. -/
  read2 : Nat → Nat → Nat → Fin T → Fin B2 → Nat
  /-- Step `t`, lane `j`'s `inp3` read address. -/
  read3 : Nat → Nat → Nat → Fin T → Fin B3 → Nat
  /-- Step `t`, lane `j`'s `inp4` read address. -/
  read4 : Nat → Nat → Nat → Fin T → Fin B4 → Nat
  /-- Lane `j`'s terminal write address. -/
  write : Nat → Nat → Nat → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Nat → Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`. -/
  mask2 : Nat → Nat → Nat → Fin T → Fin B2 → Prop
  /-- `inp3`'s read-active lanes at step `t`. -/
  mask3 : Nat → Nat → Nat → Fin T → Fin B3 → Prop
  /-- `inp4`'s read-active lanes at step `t`. -/
  mask4 : Nat → Nat → Nat → Fin T → Fin B4 → Prop
  /-- The terminal store's write-active lanes. -/
  writeMask : Nat → Nat → Nat → Fin C → Prop

namespace StreamMasked3DKernelIO₄

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the four-stream single-output fold skin, the genre's
only surface (see the `StreamMasked2DKernelIO₂` structure's
single-surface design note). Same full Hoare triple as
`StreamMasked3DKernelIO₃.ImplementsR` with a fourth input channel
threaded through: the terminal cell holds the *ideal* real value
`f pid₀ pid₁ pid₂ xs ys zs ws j` of the whole `T`-step fold, quantized
**once** at the declared grid `outDType` — read back through
`readMemAs io.outDType` as `io.outDType.ofReal (R.round io.outDType
(f …))`. Frame: every flat cell outside the write-active output window
is untouched. At `R := .triv` and `outDType := .real` the store is exact
and this is the exact streaming contract. -/
def ImplementsR (io : StreamMasked3DKernelIO₄) (R : RoundingModel)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.T → Fin io.B4 → ℝ) → Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.inp4, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ) (ws : Fin io.T → Fin io.B4 → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      io.read1 pid₀ pid₁ pid₂ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      io.read2 pid₀ pid₁ pid₂ t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      io.read3 pid₀ pid₁ pid₂ t j < A.extent io.inp3) →
    (∀ (t : Fin io.T) (j : Fin io.B4), io.mask4 pid₀ pid₁ pid₂ t j →
      io.read4 pid₀ pid₁ pid₂ t j < A.extent io.inp4) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
      io.write pid₀ pid₁ pid₂ j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = zs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B4), io.mask4 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp4 (io.read4 pid₀ pid₁ pid₂ t j) = ws t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ j))
            = io.outDType.ofReal
                (R.round io.outDType (f pid₀ pid₁ pid₂ xs ys zs ws j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMasked3DKernelIO₄.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the four-stream
widening of `StreamMasked3DKernelIO₃.ImplementsR.intro`. Obligations in
the skin's named vocabulary: `FlattenOk`, the `TraceSafeR R` safety walk
`hts` (fed the four pinned input streams and the five window-bound
groups), and the region-model rounded Hoare triple `hrun` (termination
under `execR R`, the `readMemAs outDType` rounded readback of the
terminal store, and the single-output frame; the `undef` pin is threaded
in for masked loads without an `other=` default). `hrun` is where the
consumer runs its `forRange` invariant argument — the skin does not
prove the loop. -/
theorem ImplementsR.intro (io : StreamMasked3DKernelIO₄)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.T → Fin io.B4 → ℝ) → Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
        (zs : Fin io.T → Fin io.B3 → ℝ)
        (ws : Fin io.T → Fin io.B4 → ℝ),
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp3
            (io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j) = zs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B4),
        io.mask4 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp4
            (io.read4 (s.pids 0) (s.pids 1) (s.pids 2) t j) = ws t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp3) →
      (∀ (t : Fin io.T) (j : Fin io.B4),
        io.mask4 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read4 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp4) →
      (∀ j : Fin io.C,
        io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ) (zs : Fin io.T → Fin io.B3 → ℝ)
        (ws : Fin io.T → Fin io.B4 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp3
            (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = zs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B4),
        io.mask4 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp4
            (io.read4 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = ws t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)
                      xs ys zs ws j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.C,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ xs ys zs ws s₀ hpid₀ hpid₁ hpid₂
    hu hbr1 hbr2 hbr3 hbr4 hbw hx hy hz hw
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys zs ws hu hx hy hz hw
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs ys zs ws hx hy hz hw hbr1 hbr2 hbr3 hbr4 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
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

end StreamMasked3DKernelIO₄

/-- IO signature of the **five-stream fold, single terminal output**
shape (streaming genre, style S1 on a **3-D pid grid**): the
`StreamMasked3DKernelIO₄` skin widened by a fifth streamed float input
channel. Channel widths may be radically non-uniform — the intended
consumers pair three 2-D tile channels with two **scalar-width**
channels (`B = 1`, per-tensor/per-block quantization scales, one static
and one advancing per step); a scalar channel's `f` argument is read at
lane `⟨0, _⟩`. Everything else — single-surface design, static-stream
degeneration, pid-free `T` — is inherited verbatim. Intended consumers:
the scale-augmented single-store attention family (attn_fwd_triton /
attn_fwd_causal: Q/K/V + Q_scale/K_scale). -/
structure StreamMasked3DKernelIO₅ where
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
  /-- Fourth streamed input buffer. -/
  inp4 : RegionName
  /-- Fifth streamed input buffer. -/
  inp5 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Number of streaming steps (the `forRange` trip count; a
  pid-dependent walk sets the pid-free upper bound here and gates the
  live steps in the masks). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the third input channel. -/
  B3 : Nat
  /-- Per-step tile length of the fourth input channel. -/
  B4 : Nat
  /-- Per-step tile length of the fifth input channel. -/
  B5 : Nat
  /-- Tile length of the terminal output store. -/
  C : Nat
  /-- The output buffer's floating dtype — the quantization grid of the
  terminal boundary store (an fp16-storing attention `O` sets `.fp16`,
  a bf16 host output sets `.bf16`; `.real`, the default, is exact under
  `execR R`). -/
  outDType : FloatDType := .real
  /-- Step `t`, lane `j`'s `inp1` read address for program
  `(pid₀, pid₁, pid₂)`. -/
  read1 : Nat → Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address. -/
  read2 : Nat → Nat → Nat → Fin T → Fin B2 → Nat
  /-- Step `t`, lane `j`'s `inp3` read address. -/
  read3 : Nat → Nat → Nat → Fin T → Fin B3 → Nat
  /-- Step `t`, lane `j`'s `inp4` read address. -/
  read4 : Nat → Nat → Nat → Fin T → Fin B4 → Nat
  /-- Step `t`, lane `j`'s `inp5` read address. -/
  read5 : Nat → Nat → Nat → Fin T → Fin B5 → Nat
  /-- Lane `j`'s terminal write address. -/
  write : Nat → Nat → Nat → Fin C → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Nat → Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`. -/
  mask2 : Nat → Nat → Nat → Fin T → Fin B2 → Prop
  /-- `inp3`'s read-active lanes at step `t`. -/
  mask3 : Nat → Nat → Nat → Fin T → Fin B3 → Prop
  /-- `inp4`'s read-active lanes at step `t`. -/
  mask4 : Nat → Nat → Nat → Fin T → Fin B4 → Prop
  /-- `inp5`'s read-active lanes at step `t`. -/
  mask5 : Nat → Nat → Nat → Fin T → Fin B5 → Prop
  /-- The terminal store's write-active lanes. -/
  writeMask : Nat → Nat → Nat → Fin C → Prop

namespace StreamMasked3DKernelIO₅

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the five-stream single-output fold skin, the genre's
only surface (see the `StreamMasked2DKernelIO₂` structure's
single-surface design note). Same full Hoare triple as
`StreamMasked3DKernelIO₄.ImplementsR` with a fifth input channel
threaded through: the terminal cell holds the *ideal* real value
`f pid₀ pid₁ pid₂ xs ys zs ws vs j` of the whole `T`-step fold,
quantized **once** at the declared grid `outDType` — read back through
`readMemAs io.outDType` as `io.outDType.ofReal (R.round io.outDType
(f …))`. Frame: every flat cell outside the write-active output window
is untouched. At `R := .triv` and `outDType := .real` the store is exact
and this is the exact streaming contract. -/
def ImplementsR (io : StreamMasked3DKernelIO₅) (R : RoundingModel)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.T → Fin io.B4 → ℝ) → (Fin io.T → Fin io.B5 → ℝ) →
      Fin io.C → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.inp4, io.inp5, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ) (ws : Fin io.T → Fin io.B4 → ℝ)
    (vs : Fin io.T → Fin io.B5 → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      io.read1 pid₀ pid₁ pid₂ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      io.read2 pid₀ pid₁ pid₂ t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      io.read3 pid₀ pid₁ pid₂ t j < A.extent io.inp3) →
    (∀ (t : Fin io.T) (j : Fin io.B4), io.mask4 pid₀ pid₁ pid₂ t j →
      io.read4 pid₀ pid₁ pid₂ t j < A.extent io.inp4) →
    (∀ (t : Fin io.T) (j : Fin io.B5), io.mask5 pid₀ pid₁ pid₂ t j →
      io.read5 pid₀ pid₁ pid₂ t j < A.extent io.inp5) →
    (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
      io.write pid₀ pid₁ pid₂ j < A.extent io.out) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = zs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B4), io.mask4 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp4 (io.read4 pid₀ pid₁ pid₂ t j) = ws t j) →
    (∀ (t : Fin io.T) (j : Fin io.B5), io.mask5 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp5 (io.read5 pid₀ pid₁ pid₂ t j) = vs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ j))
            = io.outDType.ofReal
                (R.round io.outDType
                  (f pid₀ pid₁ pid₂ xs ys zs ws vs j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.C, io.writeMask pid₀ pid₁ pid₂ j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMasked3DKernelIO₅.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the five-stream
widening of `StreamMasked3DKernelIO₄.ImplementsR.intro`. Obligations in
the skin's named vocabulary: `FlattenOk`, the `TraceSafeR R` safety walk
`hts` (fed the five pinned input streams and the six window-bound
groups), and the region-model rounded Hoare triple `hrun` (termination
under `execR R`, the `readMemAs outDType` rounded readback of the
terminal store, and the single-output frame; the `undef` pin is threaded
in for masked loads without an `other=` default). `hrun` is where the
consumer runs its `forRange` invariant argument — the skin does not
prove the loop. -/
theorem ImplementsR.intro (io : StreamMasked3DKernelIO₅)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.T → Fin io.B4 → ℝ) → (Fin io.T → Fin io.B5 → ℝ) →
      Fin io.C → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
        (zs : Fin io.T → Fin io.B3 → ℝ)
        (ws : Fin io.T → Fin io.B4 → ℝ)
        (vs : Fin io.T → Fin io.B5 → ℝ),
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp3
            (io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j) = zs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B4),
        io.mask4 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp4
            (io.read4 (s.pids 0) (s.pids 1) (s.pids 2) t j) = ws t j) →
      (∀ (t : Fin io.T) (j : Fin io.B5),
        io.mask5 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp5
            (io.read5 (s.pids 0) (s.pids 1) (s.pids 2) t j) = vs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp3) →
      (∀ (t : Fin io.T) (j : Fin io.B4),
        io.mask4 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read4 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp4) →
      (∀ (t : Fin io.T) (j : Fin io.B5),
        io.mask5 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read5 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp5) →
      (∀ j : Fin io.C,
        io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ) (zs : Fin io.T → Fin io.B3 → ℝ)
        (ws : Fin io.T → Fin io.B4 → ℝ)
        (vs : Fin io.T → Fin io.B5 → ℝ),
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp3
            (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = zs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B4),
        io.mask4 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp4
            (io.read4 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = ws t j) →
      (∀ (t : Fin io.T) (j : Fin io.B5),
        io.mask5 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp5
            (io.read5 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = vs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs io.outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = io.outDType.ofReal
                  (R.round io.outDType
                    (f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)
                      xs ys zs ws vs j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.C,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ xs ys zs ws vs s₀ hpid₀ hpid₁
    hpid₂ hu hbr1 hbr2 hbr3 hbr4 hbr5 hbw hx hy hz hw hv
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval, hframe⟩ :=
    hrun s₀ xs ys zs ws vs hu hx hy hz hw hv
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs ys zs ws vs hx hy hz hw hv
      hbr1 hbr2 hbr3 hbr4 hbr5 hbw
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem : io.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
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

end StreamMasked3DKernelIO₅

/-- IO signature of the **three-stream fold, three terminal outputs**
shape (streaming genre, style S1 on a **3-D pid grid**): the three-output
widening of `StreamMasked3DKernelIO₃ₓ₂` — three streamed float inputs
folded `T` steps, three terminal outputs each with its own `C` and its
own `outDType` grid. Intended consumer: the `triton_attention` forward
(fp16 `Out` tile + unrounded `L`/`M` statistics rows — the `L`/`M` pair
is only jointly meaningful, so one grouped headline rather than three).
Every design note of `StreamMasked3DKernelIO₃ₓ₂` (static streams,
pid-dependent trip counts as `T`-upper-bound + pid-eating masks,
single-surface `⊨[R]`) carries over verbatim. -/
structure StreamMasked3DKernelIO₃ₓ₃ where
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
  /-- First output buffer. -/
  out1 : RegionName
  /-- Second output buffer. -/
  out2 : RegionName
  /-- Third output buffer. -/
  out3 : RegionName
  /-- Number of streaming steps (pid-free upper bound for pid-dependent
  walks). -/
  T : Nat
  /-- Per-step tile length of the first input channel. -/
  B1 : Nat
  /-- Per-step tile length of the second input channel. -/
  B2 : Nat
  /-- Per-step tile length of the third input channel. -/
  B3 : Nat
  /-- Tile length of the first terminal store. -/
  C1 : Nat
  /-- Tile length of the second terminal store. -/
  C2 : Nat
  /-- Tile length of the third terminal store. -/
  C3 : Nat
  /-- `out1`'s floating dtype grid. -/
  out1DType : FloatDType := .real
  /-- `out2`'s floating dtype grid. -/
  out2DType : FloatDType := .real
  /-- `out3`'s floating dtype grid. -/
  out3DType : FloatDType := .real
  /-- The launch-legality precondition (defaulted unconstrained; the
  `StreamMetaMasked3DKernelIO₃` precedent): a kernel with a
  **pid-dependent trip count and an unmasked terminal store** is outside
  the unguarded `∀ pids` triple for every choice of `T`/masks/`f` — the
  forced-`True` writeMask makes the value claims non-vacuous at every
  extent-satisfiable pid while the walk runs past any pid-free `T`.
  Consumers instantiate `pre` with the host-grid bound their exact stack
  already carries (triton_attention's `hbound`). -/
  pre : Nat → Nat → Nat → Prop := fun _ _ _ => True
  /-- Step `t`, lane `j`'s `inp1` read address. -/
  read1 : Nat → Nat → Nat → Fin T → Fin B1 → Nat
  /-- Step `t`, lane `j`'s `inp2` read address. -/
  read2 : Nat → Nat → Nat → Fin T → Fin B2 → Nat
  /-- Step `t`, lane `j`'s `inp3` read address. -/
  read3 : Nat → Nat → Nat → Fin T → Fin B3 → Nat
  /-- Lane `j`'s `out1` terminal write address. -/
  write1 : Nat → Nat → Nat → Fin C1 → Nat
  /-- Lane `j`'s `out2` terminal write address. -/
  write2 : Nat → Nat → Nat → Fin C2 → Nat
  /-- Lane `j`'s `out3` terminal write address. -/
  write3 : Nat → Nat → Nat → Fin C3 → Nat
  /-- `inp1`'s read-active lanes at step `t`. -/
  mask1 : Nat → Nat → Nat → Fin T → Fin B1 → Prop
  /-- `inp2`'s read-active lanes at step `t`. -/
  mask2 : Nat → Nat → Nat → Fin T → Fin B2 → Prop
  /-- `inp3`'s read-active lanes at step `t`. -/
  mask3 : Nat → Nat → Nat → Fin T → Fin B3 → Prop
  /-- `out1`'s write-active lanes. -/
  writeMask1 : Nat → Nat → Nat → Fin C1 → Prop
  /-- `out2`'s write-active lanes. -/
  writeMask2 : Nat → Nat → Nat → Fin C2 → Prop
  /-- `out3`'s write-active lanes. -/
  writeMask3 : Nat → Nat → Nat → Fin C3 → Prop

namespace StreamMasked3DKernelIO₃ₓ₃

/-- `io.ImplementsR R f` — the **rounding-correctness** relation
`io ⊨[R] f` for the three-stream three-output fold skin (see
`StreamMasked3DKernelIO₃ₓ₂.ImplementsR` — this is its third-output
widening; each terminal cell holds its component of `f`'s triple,
quantized **once** at its own grid; frame outside the three write-active
windows). -/
def ImplementsR (io : StreamMasked3DKernelIO₃ₓ₃) (R : RoundingModel)
    (f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.C1 → ℝ) × (Fin io.C2 → ℝ) × (Fin io.C3 → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp1, io.inp2, io.inp3, io.out1, io.out2, io.out3] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
  ∀ (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
    (zs : Fin io.T → Fin io.B3 → ℝ) (s₀ : BlockState),
    io.pre pid₀ pid₁ pid₂ →
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      io.read1 pid₀ pid₁ pid₂ t j < A.extent io.inp1) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      io.read2 pid₀ pid₁ pid₂ t j < A.extent io.inp2) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      io.read3 pid₀ pid₁ pid₂ t j < A.extent io.inp3) →
    (∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
      io.write1 pid₀ pid₁ pid₂ j < A.extent io.out1) →
    (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
      io.write2 pid₀ pid₁ pid₂ j < A.extent io.out2) →
    (∀ j : Fin io.C3, io.writeMask3 pid₀ pid₁ pid₂ j →
      io.write3 pid₀ pid₁ pid₂ j < A.extent io.out3) →
    (∀ (t : Fin io.T) (j : Fin io.B1), io.mask1 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp1 (io.read1 pid₀ pid₁ pid₂ t j) = xs t j) →
    (∀ (t : Fin io.T) (j : Fin io.B2), io.mask2 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp2 (io.read2 pid₀ pid₁ pid₂ t j) = ys t j) →
    (∀ (t : Fin io.T) (j : Fin io.B3), io.mask3 pid₀ pid₁ pid₂ t j →
      s₀.readMem io.inp3 (io.read3 pid₀ pid₁ pid₂ t j) = zs t j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
          s'.readMemAs io.out1DType A.flat
              (A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j))
            = io.out1DType.ofReal
                (R.round io.out1DType
                  ((f pid₀ pid₁ pid₂ xs ys zs).1 j)))
      ∧ (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
          s'.readMemAs io.out2DType A.flat
              (A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j))
            = io.out2DType.ofReal
                (R.round io.out2DType
                  ((f pid₀ pid₁ pid₂ xs ys zs).2.1 j)))
      ∧ (∀ j : Fin io.C3, io.writeMask3 pid₀ pid₁ pid₂ j →
          s'.readMemAs io.out3DType A.flat
              (A.addr io.out3 (io.write3 pid₀ pid₁ pid₂ j))
            = io.out3DType.ofReal
                (R.round io.out3DType
                  ((f pid₀ pid₁ pid₂ xs ys zs).2.2 j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.C1, io.writeMask1 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out1 (io.write1 pid₀ pid₁ pid₂ j)) ∧
             (∀ j : Fin io.C2, io.writeMask2 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out2 (io.write2 pid₀ pid₁ pid₂ j)) ∧
             (∀ j : Fin io.C3, io.writeMask3 pid₀ pid₁ pid₂ j →
                o' ≠ A.addr io.out3 (io.write3 pid₀ pid₁ pid₂ j)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  StreamMasked3DKernelIO₃ₓ₃.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — direct flat transport, the third-output
widening of `StreamMasked3DKernelIO₃ₓ₂.ImplementsR.intro` (obligations:
`FlattenOk`, the `TraceSafeR R` walk `hts`, and the region-model triple
`hrun` with three per-grid readbacks and the three-output frame stated
with per-output region guards). `hrun` is the consumer's `forRange`
invariant obligation — the skin does not prove the loop. -/
theorem ImplementsR.intro (io : StreamMasked3DKernelIO₃ₓ₃)
    {R : RoundingModel}
    {f : Nat → Nat → Nat → (Fin io.T → Fin io.B1 → ℝ) →
      (Fin io.T → Fin io.B2 → ℝ) → (Fin io.T → Fin io.B3 → ℝ) →
      (Fin io.C1 → ℝ) × (Fin io.C2 → ℝ) × (Fin io.C3 → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (xs : Fin io.T → Fin io.B1 → ℝ) (ys : Fin io.T → Fin io.B2 → ℝ)
        (zs : Fin io.T → Fin io.B3 → ℝ),
      io.pre (s.pids 0) (s.pids 1) (s.pids 2) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp1
            (io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j) = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp2
            (io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j) = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        s.readMem io.inp3
            (io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j) = zs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp1) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp2) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) t j →
        io.read3 (s.pids 0) (s.pids 1) (s.pids 2) t j < bounds io.inp3) →
      (∀ j : Fin io.C1,
        io.writeMask1 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write1 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out1) →
      (∀ j : Fin io.C2,
        io.writeMask2 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write2 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out2) →
      (∀ j : Fin io.C3,
        io.writeMask3 (s.pids 0) (s.pids 1) (s.pids 2) j →
        io.write3 (s.pids 0) (s.pids 1) (s.pids 2) j < bounds io.out3) →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.T → Fin io.B1 → ℝ)
        (ys : Fin io.T → Fin io.B2 → ℝ) (zs : Fin io.T → Fin io.B3 → ℝ),
      io.pre (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) →
      s₀.undef = (fun _ _ => 0) →
      (∀ (t : Fin io.T) (j : Fin io.B1),
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp1
            (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = xs t j) →
      (∀ (t : Fin io.T) (j : Fin io.B2),
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp2
            (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = ys t j) →
      (∀ (t : Fin io.T) (j : Fin io.B3),
        io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j →
        s₀.readMem io.inp3
            (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) t j)
          = zs t j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.C1,
            io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs io.out1DType io.out1
                (io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = io.out1DType.ofReal
                  (R.round io.out1DType
                    ((f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) xs ys zs).1 j)))
        ∧ (∀ j : Fin io.C2,
            io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs io.out2DType io.out2
                (io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = io.out2DType.ofReal
                  (R.round io.out2DType
                    ((f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) xs ys zs).2.1 j)))
        ∧ (∀ j : Fin io.C3,
            io.writeMask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
            s1.readMemAs io.out3DType io.out3
                (io.write3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)
              = io.out3DType.ofReal
                  (R.round io.out3DType
                    ((f (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) xs ys zs).2.2 j)))
        ∧ (∀ r oo,
            ((∀ j : Fin io.C1,
                io.writeMask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                r = io.out1 →
                oo ≠ io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) ∧
             (∀ j : Fin io.C2,
                io.writeMask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                r = io.out2 →
                oo ≠ io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j) ∧
             (∀ j : Fin io.C3,
                io.writeMask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j →
                r = io.out3 →
                oo ≠ io.write3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j)) →
            s1.mem r oo = s₀.mem r oo)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid₀ pid₁ pid₂ xs ys zs s₀ hpre hpid₀ hpid₁ hpid₂ hu
    hbr1 hbr2 hbr3 hbw1 hbw2 hbw3 hx hy hz
  subst hpid₀
  subst hpid₁
  subst hpid₂
  obtain ⟨s1, hexec, hval1, hval2, hval3, hframe⟩ :=
    hrun s₀ xs ys zs hpre hu hx hy hz
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ xs ys zs hpre hx hy hz hbr1 hbr2 hbr3 hbw1 hbw2 hbw3
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  have hmem1 : io.out1 ∈ A.regions := by rw [hregs]; simp
  have hmem2 : io.out2 ∈ A.regions := by rw [hregs]; simp
  have hmem3 : io.out3 ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j hj
    have hlt : io.write1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
        < A.extent io.out1 := hbw1 j hj
    rw [A.flattenState_readMemAs hd s1 hmem1 hlt io.out1DType]
    exact hval1 j hj
  · intro j hj
    have hlt : io.write2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
        < A.extent io.out2 := hbw2 j hj
    rw [A.flattenState_readMemAs hd s1 hmem2 hlt io.out2DType]
    exact hval2 j hj
  · intro j hj
    have hlt : io.write3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) j
        < A.extent io.out3 := hbw3 j hj
    rw [A.flattenState_readMemAs hd s1 hmem3 hlt io.out3DType]
    exact hval3 j hj
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
          refine congrArg A.trCell (hframe r o ⟨?_, ?_, ?_⟩)
          · intro j hj hro hoj
            rcases hcond with hflat | ⟨hn1, _, _⟩
            · exact absurd rfl hflat
            · exact hn1 j hj (by rw [hoeq, hro, hoj])
          · intro j hj hro hoj
            rcases hcond with hflat | ⟨_, hn2, _⟩
            · exact absurd rfl hflat
            · exact hn2 j hj (by rw [hoeq, hro, hoj])
          · intro j hj hro hoj
            rcases hcond with hflat | ⟨_, _, hn3⟩
            · exact absurd rfl hflat
            · exact hn3 j hj (by rw [hoeq, hro, hoj])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end StreamMasked3DKernelIO₃ₓ₃

end VeriTile.Triton
