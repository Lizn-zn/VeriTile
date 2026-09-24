/-
Kernel IO contracts: Basic.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  /-- The output buffer's floating dtype — the quantization grid of the
  boundary store, used only by the rounding-correctness relation `⊨[R]`
  (its postcondition reads the output back as `outDType`-typed cells
  holding `R.round outDType (f …)`). Declared, not parsed: the headline
  proves the kernel's actual store cast matches. `.real` (the default)
  means an unrounded store — the exact relation `⊨` ignores this field. -/
  outDType : FloatDType := .real

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

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(two float tile channels plus three 1-lane bound-witness channels — the
block bounds `w + B ≤ extent` carried as the masked per-lane bounds
`w + B - 1 < extent` gated on `0 < w + B` — one output, no scratch). -/
private def toU (io : KernelIO₂) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 5
  nOut := 1
  nScr := 0
  bufs := [io.in1, io.in2, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .float
    | _ => .nat
  iarity := fun i => match i with
    | ⟨0, _⟩ => io.B
    | ⟨1, _⟩ => io.B
    | _ => 1
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | ⟨1, _⟩ => io.in2
    | ⟨2, _⟩ => io.in1
    | ⟨3, _⟩ => io.in2
    | _ => io.out
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | ⟨1, _⟩ => fun j => io.read2 p₀ + j.val
    | ⟨2, _⟩ => fun _ => io.read1 p₀ + io.B - 1
    | ⟨3, _⟩ => fun _ => io.read2 p₀ + io.B - 1
    | _ => fun _ => io.write p₀ + io.B - 1
  imask := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => 0 < io.read1 p₀ + io.B
    | ⟨3, _⟩ => fun _ => 0 < io.read2 p₀ + io.B
    | _ => fun _ => 0 < io.write p₀ + io.B
  owin := fun _ _ p₀ _ _ j => io.write p₀ + j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun t => t.elim0
  smask := fun t => t.elim0

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
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun _p₀ _p₁ vals _o j =>
        f (fun j' => vals (⟨0, by decide⟩ : Fin 5) j')
          (fun j' => vals (⟨1, by decide⟩ : Fin 5) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib _hob _hsb
      have hb1 : io.read1 (s.pids 0) + io.B ≤ bounds io.in1 := by
        by_cases hpos : 0 < io.read1 (s.pids 0) + io.B
        · have h : io.read1 (s.pids 0) + io.B - 1 < bounds io.in1 :=
            hib (⟨2, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb2 : io.read2 (s.pids 0) + io.B ≤ bounds io.in2 := by
        by_cases hpos : 0 < io.read2 (s.pids 0) + io.B
        · have h : io.read2 (s.pids 0) + io.B - 1 < bounds io.in2 :=
            hib (⟨3, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb3 : io.write (s.pids 0) + io.B ≤ bounds io.out := by
        by_cases hpos : 0 < io.write (s.pids 0) + io.B
        · have h : io.write (s.pids 0) + io.B - 1 < bounds io.out :=
            hib (⟨4, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      exact hts bounds s hb1 hb2 hb3
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 5) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 5) j)
          (fun j => hpins (⟨0, by decide⟩ : Fin 5) j True.intro)
          (fun j => hpins (⟨1, by decide⟩ : Fin 5) j True.intro)
      refine ⟨s1, hexec, fun _o j _ => hval j, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j => ?_
        rcases hoc (⟨0, by decide⟩ : Fin 1) j True.intro with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid h1 h2 h3 xs ys s₀ hpid hu hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | ⟨2, _⟩ => fun _ => ChanTy.read .nat s₀ io.in1 (io.read1 pid + io.B - 1)
        | ⟨3, _⟩ => fun _ => ChanTy.read .nat s₀ io.in2 (io.read2 pid + io.B - 1)
        | ⟨_+4, _⟩ => fun _ => ChanTy.read .nat s₀ io.out (io.write pid + io.B - 1))
      s₀ hpid rfl rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => by
            have hj : j.val < io.B := j.isLt
            have h : io.read1 pid + j.val < A.extent io.in1 := by omega
            exact h
        | ⟨1, _⟩ => fun j _ => by
            have hj : j.val < io.B := j.isLt
            have h : io.read2 pid + j.val < A.extent io.in2 := by omega
            exact h
        | ⟨2, _⟩ => fun _ hm => by
            have hm' : 0 < io.read1 pid + io.B := hm
            have h : io.read1 pid + io.B - 1 < A.extent io.in1 := by omega
            exact h
        | ⟨3, _⟩ => fun _ hm => by
            have hm' : 0 < io.read2 pid + io.B := hm
            have h : io.read2 pid + io.B - 1 < A.extent io.in2 := by omega
            exact h
        | ⟨_+4, _⟩ => fun _ hm => by
            have hm' : 0 < io.write pid + io.B := hm
            have h : io.write pid + io.B - 1 < A.extent io.out := by omega
            exact h)
      (fun _o j _ => by
        have hj : j.val < io.B := j.isLt
        have h : io.write pid + j.val < A.extent io.out := by omega
        exact h)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => hx j
        | ⟨1, _⟩ => fun j _ => hy j
        | ⟨2, _⟩ => fun _ _ => rfl
        | ⟨3, _⟩ => fun _ _ => rfl
        | ⟨_+4, _⟩ => fun _ _ => rfl)
  refine ⟨s', hexec, fun j => hval (⟨0, by decide⟩ : Fin 1) j True.intro, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hout
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j _ => hout j, fun t => t.elim0⟩

/-- `io.ImplementsR R f` — the **rounding-correctness** relation `io ⊨[R] f`:
the kernel of `io` computes the mathematical function `f` exactly and
quantizes it once at the declared output dtype (`io.outDType`) — the
boundary-rounding contract. Same full Hoare triple as `Implements` (same
precondition: ∀ disjoint allocation, ∀ in-bounds pid, ∀ launch state with
exact-ℝ inputs loaded), but the execution is `execR R` and the output
window holds **typed cells**: reading lane `j` back as `outDType` yields
`R.round outDType (f xs ys j)`. Inputs stay exact ℝ — the rounding model
acts at the kernel's cast/store sites, not at loads. At `R := .triv` the
store is exact and this degenerates to the exact surface. -/
def ImplementsR (io : KernelIO₂) (R : RoundingModel)
    (f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.read1 pid + io.B ≤ A.extent io.in1 →
    io.read2 pid + io.B ≤ A.extent io.in2 →
    io.write pid + io.B ≤ A.extent io.out →
  ∀ (xs ys : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B, s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.B,
          s'.readMemAs io.outDType A.flat
              (A.addr io.out (io.write pid + j.val))
            = io.outDType.ofReal (R.round io.outDType (f xs ys j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ j : Fin io.B, o' ≠ A.addr io.out (io.write pid + j.val)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R "] " f =>
  KernelIO₂.ImplementsR io R f

/-- Assembly lemma for `⊨[R]` — the rounding sibling of `Implements.intro`:
`FlattenOk`, the `TraceSafeR R` safety walk, and the region-model rounded
Hoare triple `hrun` (termination under `execR R` + typed output readback +
frame). The flat transport is `execR_flatten` plus the typed readback
transport `flattenState_readMemAs`. -/
theorem ImplementsR.intro (io : KernelIO₂) {R : RoundingModel}
    {f : (Fin io.B → ℝ) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read1 s.pid + io.B ≤ bounds io.in1 →
      io.read2 s.pid + io.B ≤ bounds io.in2 →
      io.write s.pid + io.B ≤ bounds io.out →
      (io.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ (s₀ : BlockState) (xs ys : Fin io.B → ℝ),
      (∀ j : Fin io.B, s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B, s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.B,
            s1.readMemAs io.outDType io.out (io.write s₀.pid + j.val)
              = io.outDType.ofReal (R.round io.outDType (f xs ys j)))
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.B, o ≠ io.write s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.ImplementsR R f := by
  intro A hd hregs hcov pid h1 h2 h3 xs ys s₀ hpid hu hx hy
  subst hpid
  obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀ xs ys hx hy
  have hts' : (io.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts A.extent s₀ h1 h2 h3
  have hbridge := A.execR_flatten hd hcov R _ s₀ hts' hok hu
  refine ⟨A.flattenState s1, ?_, ?_, ?_⟩
  · rw [hbridge, hexec, Option.map_some]
  · intro j
    have hmem : io.out ∈ A.regions := by rw [hregs]; simp
    have hlt : io.write s₀.pid + j.val < A.extent io.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMemAs hd s1 hmem hlt io.outDType]
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
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
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
  /-- This kernel's private working buffers (see `ScratchSpec`). Empty for
  kernels that stage nothing through memory. -/
  scratch : List ScratchSpec := []

namespace KernelIO₁

/-- `io.Implements f` — one-input sibling of `KernelIO₂.Implements`; see
the module docstring for exactly what is quantified. -/
def Implements (io : KernelIO₁)
    (f : (Fin io.Bin → ℝ) → Fin io.Bout → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.out] ++ io.scratch.map (·.buf) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.read pid + io.Bin ≤ A.extent io.inp →
    io.write pid + io.Bout ≤ A.extent io.out →
    (∀ p ∈ io.scratch, p.win pid + p.len ≤ A.extent p.buf) →
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
            ((∀ j : Fin io.Bout,
                o' ≠ A.addr io.out (io.write pid + j.val)) ∧
             (∀ p ∈ io.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => KernelIO₁.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(one float tile channel plus 1-lane bound-witness channels for the input,
output, and every scratch buffer — the block bounds `w + L ≤ extent`
carried as the masked per-lane bounds `w + L - 1 < extent` gated on
`0 < w + L` — one output, scratch as contract-free channels). -/
private def toU (io : KernelIO₁) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 3 + io.scratch.length
  nOut := 1
  nScr := io.scratch.length
  bufs := [io.inp, io.out] ++ io.scratch.map (·.buf)
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | _ => .nat
  iarity := fun i => match i with
    | ⟨0, _⟩ => io.Bin
    | _ => 1
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.inp
    | ⟨1, _⟩ => io.inp
    | ⟨2, _⟩ => io.out
    | ⟨k+3, h⟩ => (io.scratch.get ⟨k, by omega⟩).buf
  oty := fun _ => .float
  oarity := fun _ => io.Bout
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => (io.scratch.get t).len
  sbuf := fun t => (io.scratch.get t).buf
  iwin := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j => io.read p₀ + j.val
    | ⟨1, _⟩ => fun _ => io.read p₀ + io.Bin - 1
    | ⟨2, _⟩ => fun _ => io.write p₀ + io.Bout - 1
    | ⟨k+3, h⟩ => fun _ =>
        (io.scratch.get ⟨k, by omega⟩).win p₀
          + (io.scratch.get ⟨k, by omega⟩).len - 1
  imask := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => 0 < io.read p₀ + io.Bin
    | ⟨2, _⟩ => fun _ => 0 < io.write p₀ + io.Bout
    | ⟨k+3, h⟩ => fun _ =>
        0 < (io.scratch.get ⟨k, by omega⟩).win p₀
          + (io.scratch.get ⟨k, by omega⟩).len
  owin := fun _ _ p₀ _ _ j => io.write p₀ + j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun t _ p₀ _ _ k => (io.scratch.get t).win p₀ + k.val
  smask := fun _ _ _ _ _ _ => True

/-- Assembly lemma — one-input sibling of `KernelIO₂.Implements.intro`. -/
theorem Implements.intro (io : KernelIO₁)
    {f : (Fin io.Bin → ℝ) → Fin io.Bout → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read s.pid + io.Bin ≤ bounds io.inp →
      io.write s.pid + io.Bout ≤ bounds io.out →
      (∀ p ∈ io.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.Bin → ℝ),
      (∀ j : Fin io.Bin, s₀.readMem io.inp (io.read s₀.pid + j.val) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.Bout,
            s1.readMem io.out (io.write s₀.pid + j.val) = f xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.Bout, o ≠ io.write s₀.pid + j.val) →
            (∀ p ∈ io.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun _p₀ _p₁ vals _o j =>
        f (fun j' => vals (⟨0, by omega⟩ : Fin (3 + io.scratch.length)) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib _hob _hsb
      have hb1 : io.read (s.pids 0) + io.Bin ≤ bounds io.inp := by
        by_cases hpos : 0 < io.read (s.pids 0) + io.Bin
        · have h : io.read (s.pids 0) + io.Bin - 1 < bounds io.inp :=
            hib (⟨1, by omega⟩ : Fin (3 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb2 : io.write (s.pids 0) + io.Bout ≤ bounds io.out := by
        by_cases hpos : 0 < io.write (s.pids 0) + io.Bout
        · have h : io.write (s.pids 0) + io.Bout - 1 < bounds io.out :=
            hib (⟨2, by omega⟩ : Fin (3 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      refine hts bounds s hb1 hb2 ?_
      intro q hq
      obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
      subst hu
      show (io.scratch.get u).win (s.pids 0) + (io.scratch.get u).len
          ≤ bounds (io.scratch.get u).buf
      by_cases hpos : 0 < (io.scratch.get u).win (s.pids 0)
          + (io.scratch.get u).len
      · have h : (io.scratch.get u).win (s.pids 0)
            + (io.scratch.get u).len - 1 < bounds (io.scratch.get u).buf :=
          hib (⟨u.val + 3, by have := u.isLt; omega⟩ :
              Fin (3 + io.scratch.length))
            (⟨0, by decide⟩ : Fin 1) hpos
        omega
      · omega
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by omega⟩ : Fin (3 + io.scratch.length)) j)
          (fun j => hpins (⟨0, by omega⟩ : Fin (3 + io.scratch.length)) j
            True.intro)
      refine ⟨s1, hexec, fun _o j _ => hval j, ?_⟩
      intro r o' hoc hsc'
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out
        · subst hro
          refine Or.inr fun j => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 1) j True.intro with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · intro q hq hrq k
        obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
        subst hu
        rcases hsc' u k True.intro with hne | hno
        · exact absurd hrq hne
        · exact hno
  intro A hd hregs hcov pid h1 h2 hsc xs s₀ hpid hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => fun _ => ChanTy.read .nat s₀ io.inp (io.read pid + io.Bin - 1)
        | ⟨2, _⟩ => fun _ => ChanTy.read .nat s₀ io.out (io.write pid + io.Bout - 1)
        | ⟨k+3, h⟩ => fun _ =>
            have hk : k < io.scratch.length := by
              have h' : k + 3 < 3 + io.scratch.length := h
              omega
            ChanTy.read .nat s₀ (io.scratch.get ⟨k, hk⟩).buf
              ((io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len - 1))
      s₀ hpid rfl rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => by
            have hj : j.val < io.Bin := j.isLt
            have h : io.read pid + j.val < A.extent io.inp := by omega
            exact h
        | ⟨1, _⟩ => fun _ hm => by
            have hm' : 0 < io.read pid + io.Bin := hm
            have h : io.read pid + io.Bin - 1 < A.extent io.inp := by omega
            exact h
        | ⟨2, _⟩ => fun _ hm => by
            have hm' : 0 < io.write pid + io.Bout := hm
            have h : io.write pid + io.Bout - 1 < A.extent io.out := by omega
            exact h
        | ⟨k+3, hk3⟩ => fun _ hm => by
            have hk : k < io.scratch.length := by
              have h' : k + 3 < 3 + io.scratch.length := hk3
              omega
            have hm' : 0 < (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len := hm
            have hb : (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len
                ≤ A.extent (io.scratch.get ⟨k, hk⟩).buf :=
              hsc (io.scratch.get ⟨k, hk⟩) (io.scratch.get_mem ⟨k, hk⟩)
            have h : (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len - 1
                < A.extent (io.scratch.get ⟨k, hk⟩).buf := by omega
            exact h)
      (fun _o j _ => by
        have hj : j.val < io.Bout := j.isLt
        have h : io.write pid + j.val < A.extent io.out := by omega
        exact h)
      (fun t k _ => by
        have hb : (io.scratch.get t).win pid + (io.scratch.get t).len
            ≤ A.extent (io.scratch.get t).buf :=
          hsc (io.scratch.get t) (io.scratch.get_mem t)
        have hk : k.val < (io.scratch.get t).len := k.isLt
        have h : (io.scratch.get t).win pid + k.val
            < A.extent (io.scratch.get t).buf := by omega
        exact h)
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => hx j
        | ⟨1, _⟩ => fun _ _ => rfl
        | ⟨2, _⟩ => fun _ _ => rfl
        | ⟨_+3, _⟩ => fun _ _ => rfl)
  refine ⟨s', hexec, fun j => hval (⟨0, by decide⟩ : Fin 1) j True.intro, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hout, hscr⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j _ => hout j,
      fun t k _ => hscr (io.scratch.get t) (io.scratch.get_mem t) k⟩

/-- `io₁ ≡[R] io₂` — kernel equivalence on a shared one-input IO signature;
the one-input sibling of `MaskedKernelIO₂.Equiv`. The interface is read
from `io₁` (instances share it by structure update); `io₂` contributes only
its `kernel` and its private `scratch`. For every disjoint allocation of the
interface buffers plus both scratches, every program id whose windows are in
bounds, and **every** launch state (no input hypotheses — equal inputs are
"the same `s₀`"), both kernels terminate under `execR R`, their output
windows agree, and each frames outside the output window ∪ its own
scratch. -/
def Equiv (io₁ io₂ : KernelIO₁) (R : RoundingModel) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io₁.inp, io₁.out]
      ++ (io₁.scratch.map (·.buf) ++ io₂.scratch.map (·.buf)) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io₁.read pid + io₁.Bin ≤ A.extent io₁.inp →
    io₁.write pid + io₁.Bout ≤ A.extent io₁.out →
    (∀ p ∈ io₁.scratch, p.win pid + p.len ≤ A.extent p.buf) →
    (∀ p ∈ io₂.scratch, p.win pid + p.len ≤ A.extent p.buf) →
  ∀ s₀ : BlockState,
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    ∃ s₁ s₂,
      execR R (A.flattenKernel io₁.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₁
      ∧ execR R (A.flattenKernel io₂.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₂
      ∧ (∀ j : Fin io₁.Bout,
          s₁.readMem A.flat (A.addr io₁.out (io₁.write pid + j.val))
            = s₂.readMem A.flat (A.addr io₁.out (io₁.write pid + j.val)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout,
                o' ≠ A.addr io₁.out (io₁.write pid + j.val)) ∧
             (∀ p ∈ io₁.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₁.mem r' o' = (A.flattenState s₀).mem r' o')
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout,
                o' ≠ A.addr io₁.out (io₁.write pid + j.val)) ∧
             (∀ p ∈ io₂.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₂.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io₁ " ≡[" R "] " io₂ =>
  KernelIO₁.Equiv io₁ io₂ R

/-- Assembly lemma for the one-input `≡[R]` — sibling of
`MaskedKernelIO₂.Equiv.intro`; see there for the reading of the
obligations. -/
theorem Equiv.intro (io₁ io₂ : KernelIO₁) {R : RoundingModel}
    (hok₁ : (io₁.kernel.toAlgKernel).FlattenOk)
    (hok₂ : (io₂.kernel.toAlgKernel).FlattenOk)
    (hts₁ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read s.pid + io₁.Bin ≤ bounds io₁.inp →
      io₁.write s.pid + io₁.Bout ≤ bounds io₁.out →
      (∀ p ∈ io₁.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₁.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hts₂ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read s.pid + io₁.Bin ≤ bounds io₁.inp →
      io₁.write s.pid + io₁.Bout ≤ bounds io₁.out →
      (∀ p ∈ io₂.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₂.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ s₀ : BlockState,
      ∃ s1 s2,
        execR R (io₁.kernel.toAlgKernel) s₀ = some s1
        ∧ execR R (io₂.kernel.toAlgKernel) s₀ = some s2
        ∧ (∀ j : Fin io₁.Bout,
            s1.readMem io₁.out (io₁.write s₀.pid + j.val)
              = s2.readMem io₁.out (io₁.write s₀.pid + j.val))
        ∧ (∀ r o,
            (r ≠ io₁.out ∨
              ∀ j : Fin io₁.Bout, o ≠ io₁.write s₀.pid + j.val) →
            (∀ p ∈ io₁.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s1.mem r o = s₀.mem r o)
        ∧ (∀ r o,
            (r ≠ io₁.out ∨
              ∀ j : Fin io₁.Bout, o ≠ io₁.write s₀.pid + j.val) →
            (∀ p ∈ io₂.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s2.mem r o = s₀.mem r o)) :
    io₁.Equiv io₂ R := by
  intro A hd hregs hcov pid h1 h2 hsc1 hsc2 s₀ hpid hu
  subst hpid
  obtain ⟨s1, s2, hexec1, hexec2, hval, hframe1, hframe2⟩ := hrun s₀
  have hts₁' : (io₁.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₁ A.extent s₀ h1 h2 hsc1
  have hts₂' : (io₂.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₂ A.extent s₀ h1 h2 hsc2
  have hbridge1 := A.execR_flatten hd hcov R _ s₀ hts₁' hok₁ hu
  have hbridge2 := A.execR_flatten hd hcov R _ s₀ hts₂' hok₂ hu
  have hmem : io₁.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, A.flattenState s2, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hbridge1, hexec1, Option.map_some]
  · rw [hbridge2, hexec2, Option.map_some]
  · intro j
    have hlt : io₁.write s₀.pid + j.val < A.extent io₁.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt,
        A.flattenState_readMem hd s2 hmem hlt]
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
          refine congrArg A.trCell (hframe1 r o ?_ ?_)
          · by_cases hro : r = io₁.out
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s2).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s2.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe2 r o ?_ ?_)
          · by_cases hro : r = io₁.out
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end KernelIO₁

/-- IO signature of a **three-input / one-output** kernel. All four tile
lengths are independent (`B1`/`B2`/`B3`/`Bout`), so this covers per-lane
ternary maps as well as kernels that mix per-program tiles with **shared**
buffers: a read window may be constant (e.g. `read2 := fun _ => 0` — every
program reads the same scalar cell). Same reading as `KernelIO₂`: the reads
are the address half of the precondition, the write of the postcondition;
buffer sizes are not signature content. -/
structure KernelIO₃ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First input buffer. -/
  in1 : RegionName
  /-- Second input buffer. -/
  in2 : RegionName
  /-- Third input buffer. -/
  in3 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- First input tile length. -/
  B1 : Nat
  /-- Second input tile length. -/
  B2 : Nat
  /-- Third input tile length. -/
  B3 : Nat
  /-- Output tile length. -/
  Bout : Nat
  /-- Where program `pid` reads its `in1` tile: `[read1 pid, read1 pid + B1)`. -/
  read1 : Nat → Nat
  /-- Where program `pid` reads its `in2` tile. -/
  read2 : Nat → Nat
  /-- Where program `pid` reads its `in3` tile. -/
  read3 : Nat → Nat
  /-- Where program `pid` writes its output tile. -/
  write : Nat → Nat
  /-- This kernel's private working buffers (see `ScratchSpec`). Empty for
  kernels that stage nothing through memory. -/
  scratch : List ScratchSpec := []

namespace KernelIO₃

/-- `io.Implements f` — three-input sibling of `KernelIO₂.Implements`; see
the module docstring for exactly what is quantified. -/
def Implements (io : KernelIO₃)
    (f : (Fin io.B1 → ℝ) → (Fin io.B2 → ℝ) → (Fin io.B3 → ℝ) →
      Fin io.Bout → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.in3, io.out] ++ io.scratch.map (·.buf) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.read1 pid + io.B1 ≤ A.extent io.in1 →
    io.read2 pid + io.B2 ≤ A.extent io.in2 →
    io.read3 pid + io.B3 ≤ A.extent io.in3 →
    io.write pid + io.Bout ≤ A.extent io.out →
    (∀ p ∈ io.scratch, p.win pid + p.len ≤ A.extent p.buf) →
  ∀ (xs : Fin io.B1 → ℝ) (ys : Fin io.B2 → ℝ) (zs : Fin io.B3 → ℝ)
      (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B1, s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B2, s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    (∀ j : Fin io.B3, s₀.readMem io.in3 (io.read3 pid + j.val) = zs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.Bout,
          s'.readMem A.flat (A.addr io.out (io.write pid + j.val))
            = f xs ys zs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.Bout,
                o' ≠ A.addr io.out (io.write pid + j.val)) ∧
             (∀ p ∈ io.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => KernelIO₃.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(three float tile channels plus 1-lane bound-witness channels for every
input, the output, and every scratch buffer — see `KernelIO₁.toU` — one
output, scratch as contract-free channels). -/
private def toU (io : KernelIO₃) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 7 + io.scratch.length
  nOut := 1
  nScr := io.scratch.length
  bufs := [io.in1, io.in2, io.in3, io.out] ++ io.scratch.map (·.buf)
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .float
    | ⟨2, _⟩ => .float
    | _ => .nat
  iarity := fun i => match i with
    | ⟨0, _⟩ => io.B1
    | ⟨1, _⟩ => io.B2
    | ⟨2, _⟩ => io.B3
    | _ => 1
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | ⟨1, _⟩ => io.in2
    | ⟨2, _⟩ => io.in3
    | ⟨3, _⟩ => io.in1
    | ⟨4, _⟩ => io.in2
    | ⟨5, _⟩ => io.in3
    | ⟨6, _⟩ => io.out
    | ⟨k+7, h⟩ => (io.scratch.get ⟨k, by omega⟩).buf
  oty := fun _ => .float
  oarity := fun _ => io.Bout
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => (io.scratch.get t).len
  sbuf := fun t => (io.scratch.get t).buf
  iwin := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | ⟨1, _⟩ => fun j => io.read2 p₀ + j.val
    | ⟨2, _⟩ => fun j => io.read3 p₀ + j.val
    | ⟨3, _⟩ => fun _ => io.read1 p₀ + io.B1 - 1
    | ⟨4, _⟩ => fun _ => io.read2 p₀ + io.B2 - 1
    | ⟨5, _⟩ => fun _ => io.read3 p₀ + io.B3 - 1
    | ⟨6, _⟩ => fun _ => io.write p₀ + io.Bout - 1
    | ⟨k+7, h⟩ => fun _ =>
        (io.scratch.get ⟨k, by omega⟩).win p₀
          + (io.scratch.get ⟨k, by omega⟩).len - 1
  imask := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => True
    | ⟨3, _⟩ => fun _ => 0 < io.read1 p₀ + io.B1
    | ⟨4, _⟩ => fun _ => 0 < io.read2 p₀ + io.B2
    | ⟨5, _⟩ => fun _ => 0 < io.read3 p₀ + io.B3
    | ⟨6, _⟩ => fun _ => 0 < io.write p₀ + io.Bout
    | ⟨k+7, h⟩ => fun _ =>
        0 < (io.scratch.get ⟨k, by omega⟩).win p₀
          + (io.scratch.get ⟨k, by omega⟩).len
  owin := fun _ _ p₀ _ _ j => io.write p₀ + j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun t _ p₀ _ _ k => (io.scratch.get t).win p₀ + k.val
  smask := fun _ _ _ _ _ _ => True

/-- Assembly lemma — three-input sibling of `KernelIO₂.Implements.intro`. -/
theorem Implements.intro (io : KernelIO₃)
    {f : (Fin io.B1 → ℝ) → (Fin io.B2 → ℝ) → (Fin io.B3 → ℝ) →
      Fin io.Bout → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read1 s.pid + io.B1 ≤ bounds io.in1 →
      io.read2 s.pid + io.B2 ≤ bounds io.in2 →
      io.read3 s.pid + io.B3 ≤ bounds io.in3 →
      io.write s.pid + io.Bout ≤ bounds io.out →
      (∀ p ∈ io.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.B1 → ℝ) (ys : Fin io.B2 → ℝ)
        (zs : Fin io.B3 → ℝ),
      (∀ j : Fin io.B1, s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B2, s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      (∀ j : Fin io.B3, s₀.readMem io.in3 (io.read3 s₀.pid + j.val) = zs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.Bout,
            s1.readMem io.out (io.write s₀.pid + j.val) = f xs ys zs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨ ∀ j : Fin io.Bout, o ≠ io.write s₀.pid + j.val) →
            (∀ p ∈ io.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun _p₀ _p₁ vals _o j =>
        f (fun j' => vals (⟨0, by omega⟩ : Fin (7 + io.scratch.length)) j')
          (fun j' => vals (⟨1, by omega⟩ : Fin (7 + io.scratch.length)) j')
          (fun j' => vals (⟨2, by omega⟩ : Fin (7 + io.scratch.length)) j')
          j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib _hob _hsb
      have hb1 : io.read1 (s.pids 0) + io.B1 ≤ bounds io.in1 := by
        by_cases hpos : 0 < io.read1 (s.pids 0) + io.B1
        · have h : io.read1 (s.pids 0) + io.B1 - 1 < bounds io.in1 :=
            hib (⟨3, by omega⟩ : Fin (7 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb2 : io.read2 (s.pids 0) + io.B2 ≤ bounds io.in2 := by
        by_cases hpos : 0 < io.read2 (s.pids 0) + io.B2
        · have h : io.read2 (s.pids 0) + io.B2 - 1 < bounds io.in2 :=
            hib (⟨4, by omega⟩ : Fin (7 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb3 : io.read3 (s.pids 0) + io.B3 ≤ bounds io.in3 := by
        by_cases hpos : 0 < io.read3 (s.pids 0) + io.B3
        · have h : io.read3 (s.pids 0) + io.B3 - 1 < bounds io.in3 :=
            hib (⟨5, by omega⟩ : Fin (7 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb4 : io.write (s.pids 0) + io.Bout ≤ bounds io.out := by
        by_cases hpos : 0 < io.write (s.pids 0) + io.Bout
        · have h : io.write (s.pids 0) + io.Bout - 1 < bounds io.out :=
            hib (⟨6, by omega⟩ : Fin (7 + io.scratch.length))
              (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      refine hts bounds s hb1 hb2 hb3 hb4 ?_
      intro q hq
      obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
      subst hu
      show (io.scratch.get u).win (s.pids 0) + (io.scratch.get u).len
          ≤ bounds (io.scratch.get u).buf
      by_cases hpos : 0 < (io.scratch.get u).win (s.pids 0)
          + (io.scratch.get u).len
      · have h : (io.scratch.get u).win (s.pids 0)
            + (io.scratch.get u).len - 1 < bounds (io.scratch.get u).buf :=
          hib (⟨u.val + 7, by have := u.isLt; omega⟩ :
              Fin (7 + io.scratch.length))
            (⟨0, by decide⟩ : Fin 1) hpos
        omega
      · omega
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀
          (fun j => vals (⟨0, by omega⟩ : Fin (7 + io.scratch.length)) j)
          (fun j => vals (⟨1, by omega⟩ : Fin (7 + io.scratch.length)) j)
          (fun j => vals (⟨2, by omega⟩ : Fin (7 + io.scratch.length)) j)
          (fun j => hpins (⟨0, by omega⟩ : Fin (7 + io.scratch.length)) j
            True.intro)
          (fun j => hpins (⟨1, by omega⟩ : Fin (7 + io.scratch.length)) j
            True.intro)
          (fun j => hpins (⟨2, by omega⟩ : Fin (7 + io.scratch.length)) j
            True.intro)
      refine ⟨s1, hexec, fun _o j _ => hval j, ?_⟩
      intro r o' hoc hsc'
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out
        · subst hro
          refine Or.inr fun j => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 1) j True.intro with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · intro q hq hrq k
        obtain ⟨u, hu⟩ := List.mem_iff_get.mp hq
        subst hu
        rcases hsc' u k True.intro with hne | hno
        · exact absurd hrq hne
        · exact hno
  intro A hd hregs hcov pid h1 h2 h3 h4 hsc xs ys zs s₀ hpid hu hx hy hz
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | ⟨2, _⟩ => zs
        | ⟨3, _⟩ => fun _ => ChanTy.read .nat s₀ io.in1 (io.read1 pid + io.B1 - 1)
        | ⟨4, _⟩ => fun _ => ChanTy.read .nat s₀ io.in2 (io.read2 pid + io.B2 - 1)
        | ⟨5, _⟩ => fun _ => ChanTy.read .nat s₀ io.in3 (io.read3 pid + io.B3 - 1)
        | ⟨6, _⟩ => fun _ => ChanTy.read .nat s₀ io.out (io.write pid + io.Bout - 1)
        | ⟨k+7, h⟩ => fun _ =>
            have hk : k < io.scratch.length := by
              have h' : k + 7 < 7 + io.scratch.length := h
              omega
            ChanTy.read .nat s₀ (io.scratch.get ⟨k, hk⟩).buf
              ((io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len - 1))
      s₀ hpid rfl rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => by
            have hj : j.val < io.B1 := j.isLt
            have h : io.read1 pid + j.val < A.extent io.in1 := by omega
            exact h
        | ⟨1, _⟩ => fun j _ => by
            have hj : j.val < io.B2 := j.isLt
            have h : io.read2 pid + j.val < A.extent io.in2 := by omega
            exact h
        | ⟨2, _⟩ => fun j _ => by
            have hj : j.val < io.B3 := j.isLt
            have h : io.read3 pid + j.val < A.extent io.in3 := by omega
            exact h
        | ⟨3, _⟩ => fun _ hm => by
            have hm' : 0 < io.read1 pid + io.B1 := hm
            have h : io.read1 pid + io.B1 - 1 < A.extent io.in1 := by omega
            exact h
        | ⟨4, _⟩ => fun _ hm => by
            have hm' : 0 < io.read2 pid + io.B2 := hm
            have h : io.read2 pid + io.B2 - 1 < A.extent io.in2 := by omega
            exact h
        | ⟨5, _⟩ => fun _ hm => by
            have hm' : 0 < io.read3 pid + io.B3 := hm
            have h : io.read3 pid + io.B3 - 1 < A.extent io.in3 := by omega
            exact h
        | ⟨6, _⟩ => fun _ hm => by
            have hm' : 0 < io.write pid + io.Bout := hm
            have h : io.write pid + io.Bout - 1 < A.extent io.out := by omega
            exact h
        | ⟨k+7, hk7⟩ => fun _ hm => by
            have hk : k < io.scratch.length := by
              have h' : k + 7 < 7 + io.scratch.length := hk7
              omega
            have hm' : 0 < (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len := hm
            have hb : (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len
                ≤ A.extent (io.scratch.get ⟨k, hk⟩).buf :=
              hsc (io.scratch.get ⟨k, hk⟩) (io.scratch.get_mem ⟨k, hk⟩)
            have h : (io.scratch.get ⟨k, hk⟩).win pid
                + (io.scratch.get ⟨k, hk⟩).len - 1
                < A.extent (io.scratch.get ⟨k, hk⟩).buf := by omega
            exact h)
      (fun _o j _ => by
        have hj : j.val < io.Bout := j.isLt
        have h : io.write pid + j.val < A.extent io.out := by omega
        exact h)
      (fun t k _ => by
        have hb : (io.scratch.get t).win pid + (io.scratch.get t).len
            ≤ A.extent (io.scratch.get t).buf :=
          hsc (io.scratch.get t) (io.scratch.get_mem t)
        have hk : k.val < (io.scratch.get t).len := k.isLt
        have h : (io.scratch.get t).win pid + k.val
            < A.extent (io.scratch.get t).buf := by omega
        exact h)
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => hx j
        | ⟨1, _⟩ => fun j _ => hy j
        | ⟨2, _⟩ => fun j _ => hz j
        | ⟨3, _⟩ => fun _ _ => rfl
        | ⟨4, _⟩ => fun _ _ => rfl
        | ⟨5, _⟩ => fun _ _ => rfl
        | ⟨6, _⟩ => fun _ _ => rfl
        | ⟨_+7, _⟩ => fun _ _ => rfl)
  refine ⟨s', hexec, fun j => hval (⟨0, by decide⟩ : Fin 1) j True.intro, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hout, hscr⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j _ => hout j,
      fun t k _ => hscr (io.scratch.get t) (io.scratch.get_mem t) k⟩

/-- `io₁ ≡[R] io₂` — kernel equivalence on a shared three-input IO
signature; the three-input sibling of `MaskedKernelIO₂.Equiv`. The interface
is read from `io₁` (instances share it by structure update); `io₂`
contributes only its `kernel` and its private `scratch`. No input
hypotheses: equal inputs are "the same `s₀`". -/
def Equiv (io₁ io₂ : KernelIO₃) (R : RoundingModel) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io₁.in1, io₁.in2, io₁.in3, io₁.out]
      ++ (io₁.scratch.map (·.buf) ++ io₂.scratch.map (·.buf)) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io₁.read1 pid + io₁.B1 ≤ A.extent io₁.in1 →
    io₁.read2 pid + io₁.B2 ≤ A.extent io₁.in2 →
    io₁.read3 pid + io₁.B3 ≤ A.extent io₁.in3 →
    io₁.write pid + io₁.Bout ≤ A.extent io₁.out →
    (∀ p ∈ io₁.scratch, p.win pid + p.len ≤ A.extent p.buf) →
    (∀ p ∈ io₂.scratch, p.win pid + p.len ≤ A.extent p.buf) →
  ∀ s₀ : BlockState,
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    ∃ s₁ s₂,
      execR R (A.flattenKernel io₁.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₁
      ∧ execR R (A.flattenKernel io₂.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₂
      ∧ (∀ j : Fin io₁.Bout,
          s₁.readMem A.flat (A.addr io₁.out (io₁.write pid + j.val))
            = s₂.readMem A.flat (A.addr io₁.out (io₁.write pid + j.val)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout,
                o' ≠ A.addr io₁.out (io₁.write pid + j.val)) ∧
             (∀ p ∈ io₁.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₁.mem r' o' = (A.flattenState s₀).mem r' o')
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout,
                o' ≠ A.addr io₁.out (io₁.write pid + j.val)) ∧
             (∀ p ∈ io₂.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₂.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io₁ " ≡[" R "] " io₂ =>
  KernelIO₃.Equiv io₁ io₂ R

/-- Assembly lemma for the three-input `≡[R]` — sibling of
`MaskedKernelIO₂.Equiv.intro`; see there for the reading of the
obligations. -/
theorem Equiv.intro (io₁ io₂ : KernelIO₃) {R : RoundingModel}
    (hok₁ : (io₁.kernel.toAlgKernel).FlattenOk)
    (hok₂ : (io₂.kernel.toAlgKernel).FlattenOk)
    (hts₁ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read1 s.pid + io₁.B1 ≤ bounds io₁.in1 →
      io₁.read2 s.pid + io₁.B2 ≤ bounds io₁.in2 →
      io₁.read3 s.pid + io₁.B3 ≤ bounds io₁.in3 →
      io₁.write s.pid + io₁.Bout ≤ bounds io₁.out →
      (∀ p ∈ io₁.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₁.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hts₂ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read1 s.pid + io₁.B1 ≤ bounds io₁.in1 →
      io₁.read2 s.pid + io₁.B2 ≤ bounds io₁.in2 →
      io₁.read3 s.pid + io₁.B3 ≤ bounds io₁.in3 →
      io₁.write s.pid + io₁.Bout ≤ bounds io₁.out →
      (∀ p ∈ io₂.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₂.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ s₀ : BlockState,
      ∃ s1 s2,
        execR R (io₁.kernel.toAlgKernel) s₀ = some s1
        ∧ execR R (io₂.kernel.toAlgKernel) s₀ = some s2
        ∧ (∀ j : Fin io₁.Bout,
            s1.readMem io₁.out (io₁.write s₀.pid + j.val)
              = s2.readMem io₁.out (io₁.write s₀.pid + j.val))
        ∧ (∀ r o,
            (r ≠ io₁.out ∨
              ∀ j : Fin io₁.Bout, o ≠ io₁.write s₀.pid + j.val) →
            (∀ p ∈ io₁.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s1.mem r o = s₀.mem r o)
        ∧ (∀ r o,
            (r ≠ io₁.out ∨
              ∀ j : Fin io₁.Bout, o ≠ io₁.write s₀.pid + j.val) →
            (∀ p ∈ io₂.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s2.mem r o = s₀.mem r o)) :
    io₁.Equiv io₂ R := by
  intro A hd hregs hcov pid h1 h2 h3 h4 hsc1 hsc2 s₀ hpid hu
  subst hpid
  obtain ⟨s1, s2, hexec1, hexec2, hval, hframe1, hframe2⟩ := hrun s₀
  have hts₁' : (io₁.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₁ A.extent s₀ h1 h2 h3 h4 hsc1
  have hts₂' : (io₂.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₂ A.extent s₀ h1 h2 h3 h4 hsc2
  have hbridge1 := A.execR_flatten hd hcov R _ s₀ hts₁' hok₁ hu
  have hbridge2 := A.execR_flatten hd hcov R _ s₀ hts₂' hok₂ hu
  have hmem : io₁.out ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, A.flattenState s2, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hbridge1, hexec1, Option.map_some]
  · rw [hbridge2, hexec2, Option.map_some]
  · intro j
    have hlt : io₁.write s₀.pid + j.val < A.extent io₁.out := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem hlt,
        A.flattenState_readMem hd s2 hmem hlt]
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
          refine congrArg A.trCell (hframe1 r o ?_ ?_)
          · by_cases hro : r = io₁.out
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s2).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s2.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe2 r o ?_ ?_)
          · by_cases hro : r = io₁.out
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hnout, _⟩
              · exact hflat rfl
              · exact hnout j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end KernelIO₃

/-- IO signature of a **three-input / two-output** kernel (`₃ₓ₂` = "3 × 2").
The five tile lengths are independent; `f` returns the two output tiles as a
pair, in field order (`out1`, `out2`). The frame guarantee covers every cell
outside the **union** of the two output windows — this is why a two-output
kernel needs its own combinator rather than two one-output statements, whose
frames would each (falsely) claim the other output untouched. -/
structure KernelIO₃ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First input buffer. -/
  in1 : RegionName
  /-- Second input buffer. -/
  in2 : RegionName
  /-- Third input buffer. -/
  in3 : RegionName
  /-- First output buffer. -/
  out1 : RegionName
  /-- Second output buffer. -/
  out2 : RegionName
  /-- First input tile length. -/
  B1 : Nat
  /-- Second input tile length. -/
  B2 : Nat
  /-- Third input tile length. -/
  B3 : Nat
  /-- First output tile length. -/
  Bout1 : Nat
  /-- Second output tile length. -/
  Bout2 : Nat
  /-- Where program `pid` reads its `in1` tile: `[read1 pid, read1 pid + B1)`. -/
  read1 : Nat → Nat
  /-- Where program `pid` reads its `in2` tile. -/
  read2 : Nat → Nat
  /-- Where program `pid` reads its `in3` tile. -/
  read3 : Nat → Nat
  /-- Where program `pid` writes its `out1` tile. -/
  write1 : Nat → Nat
  /-- Where program `pid` writes its `out2` tile. -/
  write2 : Nat → Nat

namespace KernelIO₃ₓ₂

/-- `io.Implements f` — three-input / two-output sibling of
`KernelIO₂.Implements`. The postcondition asserts both output windows; the
frame covers every cell outside their union. -/
def Implements (io : KernelIO₃ₓ₂)
    (f : (Fin io.B1 → ℝ) → (Fin io.B2 → ℝ) → (Fin io.B3 → ℝ) →
      (Fin io.Bout1 → ℝ) × (Fin io.Bout2 → ℝ)) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.in3, io.out1, io.out2] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io.read1 pid + io.B1 ≤ A.extent io.in1 →
    io.read2 pid + io.B2 ≤ A.extent io.in2 →
    io.read3 pid + io.B3 ≤ A.extent io.in3 →
    io.write1 pid + io.Bout1 ≤ A.extent io.out1 →
    io.write2 pid + io.Bout2 ≤ A.extent io.out2 →
  ∀ (xs : Fin io.B1 → ℝ) (ys : Fin io.B2 → ℝ) (zs : Fin io.B3 → ℝ)
      (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B1, s₀.readMem io.in1 (io.read1 pid + j.val) = xs j) →
    (∀ j : Fin io.B2, s₀.readMem io.in2 (io.read2 pid + j.val) = ys j) →
    (∀ j : Fin io.B3, s₀.readMem io.in3 (io.read3 pid + j.val) = zs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : Fin io.Bout1,
          s'.readMem A.flat (A.addr io.out1 (io.write1 pid + j.val))
            = (f xs ys zs).1 j)
      ∧ (∀ j : Fin io.Bout2,
          s'.readMem A.flat (A.addr io.out2 (io.write2 pid + j.val))
            = (f xs ys zs).2 j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io.Bout1,
                o' ≠ A.addr io.out1 (io.write1 pid + j.val)) ∧
             (∀ j : Fin io.Bout2,
                o' ≠ A.addr io.out2 (io.write2 pid + j.val)))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => KernelIO₃ₓ₂.Implements

/-- Embed into the unified core — proof plumbing for `Implements.intro`
(three float tile channels plus 1-lane bound-witness channels for every
input and output buffer — see `KernelIO₁.toU` — two outputs, no scratch). -/
private def toU (io : KernelIO₃ₓ₂) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 8
  nOut := 2
  nScr := 0
  bufs := [io.in1, io.in2, io.in3, io.out1, io.out2]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .float
    | ⟨2, _⟩ => .float
    | _ => .nat
  iarity := fun i => match i with
    | ⟨0, _⟩ => io.B1
    | ⟨1, _⟩ => io.B2
    | ⟨2, _⟩ => io.B3
    | _ => 1
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | ⟨1, _⟩ => io.in2
    | ⟨2, _⟩ => io.in3
    | ⟨3, _⟩ => io.in1
    | ⟨4, _⟩ => io.in2
    | ⟨5, _⟩ => io.in3
    | ⟨6, _⟩ => io.out1
    | _ => io.out2
  oty := fun _ => .float
  oarity := fun o => match o with
    | ⟨0, _⟩ => io.Bout1
    | _ => io.Bout2
  obuf := fun o => match o with
    | ⟨0, _⟩ => io.out1
    | _ => io.out2
  obuf_mem := fun o => by fin_cases o <;> simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j => io.read1 p₀ + j.val
    | ⟨1, _⟩ => fun j => io.read2 p₀ + j.val
    | ⟨2, _⟩ => fun j => io.read3 p₀ + j.val
    | ⟨3, _⟩ => fun _ => io.read1 p₀ + io.B1 - 1
    | ⟨4, _⟩ => fun _ => io.read2 p₀ + io.B2 - 1
    | ⟨5, _⟩ => fun _ => io.read3 p₀ + io.B3 - 1
    | ⟨6, _⟩ => fun _ => io.write1 p₀ + io.Bout1 - 1
    | _ => fun _ => io.write2 p₀ + io.Bout2 - 1
  imask := fun i _ p₀ _ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => True
    | ⟨3, _⟩ => fun _ => 0 < io.read1 p₀ + io.B1
    | ⟨4, _⟩ => fun _ => 0 < io.read2 p₀ + io.B2
    | ⟨5, _⟩ => fun _ => 0 < io.read3 p₀ + io.B3
    | ⟨6, _⟩ => fun _ => 0 < io.write1 p₀ + io.Bout1
    | _ => fun _ => 0 < io.write2 p₀ + io.Bout2
  owin := fun o _ p₀ _ _ => match o with
    | ⟨0, _⟩ => fun j => io.write1 p₀ + j.val
    | _ => fun j => io.write2 p₀ + j.val
  omask := fun _ _ _ _ _ _ => True
  swin := fun t => t.elim0
  smask := fun t => t.elim0

/-- Assembly lemma — three-input / two-output sibling of
`KernelIO₂.Implements.intro`. The region-model triple `hrun` takes the frame
as two window conditions (one per output); a cell is untouched when it avoids
both windows. -/
theorem Implements.intro (io : KernelIO₃ₓ₂)
    {f : (Fin io.B1 → ℝ) → (Fin io.B2 → ℝ) → (Fin io.B3 → ℝ) →
      (Fin io.Bout1 → ℝ) × (Fin io.Bout2 → ℝ)}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.read1 s.pid + io.B1 ≤ bounds io.in1 →
      io.read2 s.pid + io.B2 ≤ bounds io.in2 →
      io.read3 s.pid + io.B3 ≤ bounds io.in3 →
      io.write1 s.pid + io.Bout1 ≤ bounds io.out1 →
      io.write2 s.pid + io.Bout2 ≤ bounds io.out2 →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : Fin io.B1 → ℝ) (ys : Fin io.B2 → ℝ)
        (zs : Fin io.B3 → ℝ),
      (∀ j : Fin io.B1, s₀.readMem io.in1 (io.read1 s₀.pid + j.val) = xs j) →
      (∀ j : Fin io.B2, s₀.readMem io.in2 (io.read2 s₀.pid + j.val) = ys j) →
      (∀ j : Fin io.B3, s₀.readMem io.in3 (io.read3 s₀.pid + j.val) = zs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin io.Bout1,
            s1.readMem io.out1 (io.write1 s₀.pid + j.val) = (f xs ys zs).1 j)
        ∧ (∀ j : Fin io.Bout2,
            s1.readMem io.out2 (io.write2 s₀.pid + j.val) = (f xs ys zs).2 j)
        ∧ (∀ r o,
            (r ≠ io.out1 ∨
              ∀ j : Fin io.Bout1, o ≠ io.write1 s₀.pid + j.val) →
            (r ≠ io.out2 ∨
              ∀ j : Fin io.Bout2, o ≠ io.write2 s₀.pid + j.val) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun _p₀ _p₁ vals o => match o with
        | ⟨0, _⟩ => fun j =>
            (f (fun j' => vals (⟨0, by decide⟩ : Fin 8) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 8) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 8) j')).1 j
        | ⟨_+1, _⟩ => fun j =>
            (f (fun j' => vals (⟨0, by decide⟩ : Fin 8) j')
              (fun j' => vals (⟨1, by decide⟩ : Fin 8) j')
              (fun j' => vals (⟨2, by decide⟩ : Fin 8) j')).2 j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib _hob _hsb
      have hb1 : io.read1 (s.pids 0) + io.B1 ≤ bounds io.in1 := by
        by_cases hpos : 0 < io.read1 (s.pids 0) + io.B1
        · have h : io.read1 (s.pids 0) + io.B1 - 1 < bounds io.in1 :=
            hib (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb2 : io.read2 (s.pids 0) + io.B2 ≤ bounds io.in2 := by
        by_cases hpos : 0 < io.read2 (s.pids 0) + io.B2
        · have h : io.read2 (s.pids 0) + io.B2 - 1 < bounds io.in2 :=
            hib (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb3 : io.read3 (s.pids 0) + io.B3 ≤ bounds io.in3 := by
        by_cases hpos : 0 < io.read3 (s.pids 0) + io.B3
        · have h : io.read3 (s.pids 0) + io.B3 - 1 < bounds io.in3 :=
            hib (⟨5, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb4 : io.write1 (s.pids 0) + io.Bout1 ≤ bounds io.out1 := by
        by_cases hpos : 0 < io.write1 (s.pids 0) + io.Bout1
        · have h : io.write1 (s.pids 0) + io.Bout1 - 1 < bounds io.out1 :=
            hib (⟨6, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      have hb5 : io.write2 (s.pids 0) + io.Bout2 ≤ bounds io.out2 := by
        by_cases hpos : 0 < io.write2 (s.pids 0) + io.Bout2
        · have h : io.write2 (s.pids 0) + io.Bout2 - 1 < bounds io.out2 :=
            hib (⟨7, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) hpos
          omega
        · omega
      exact hts bounds s hb1 hb2 hb3 hb4 hb5
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval1, hval2, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨0, by decide⟩ : Fin 8) j)
          (fun j => vals (⟨1, by decide⟩ : Fin 8) j)
          (fun j => vals (⟨2, by decide⟩ : Fin 8) j)
          (fun j => hpins (⟨0, by decide⟩ : Fin 8) j True.intro)
          (fun j => hpins (⟨1, by decide⟩ : Fin 8) j True.intro)
          (fun j => hpins (⟨2, by decide⟩ : Fin 8) j True.intro)
      refine ⟨s1, hexec, fun o => match o with
        | ⟨0, _⟩ => fun j _ => hval1 j
        | ⟨_+1, _⟩ => fun j _ => hval2 j, ?_⟩
      intro r o' hoc _hsc
      refine hframe r o' ?_ ?_
      · by_cases hro : r = io.out1
        · subst hro
          refine Or.inr fun j => ?_
          rcases hoc (⟨0, by decide⟩ : Fin 2) j True.intro with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
      · by_cases hro : r = io.out2
        · subst hro
          refine Or.inr fun j => ?_
          rcases hoc (⟨1, by decide⟩ : Fin 2) j True.intro with hne | hno
          · exact absurd rfl hne
          · exact hno
        · exact Or.inl hro
  intro A hd hregs hcov pid h1 h2 h3 h4 h5 xs ys zs s₀ hpid hu hx hy hz
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => ys
        | ⟨2, _⟩ => zs
        | ⟨3, _⟩ => fun _ => ChanTy.read .nat s₀ io.in1 (io.read1 pid + io.B1 - 1)
        | ⟨4, _⟩ => fun _ => ChanTy.read .nat s₀ io.in2 (io.read2 pid + io.B2 - 1)
        | ⟨5, _⟩ => fun _ => ChanTy.read .nat s₀ io.in3 (io.read3 pid + io.B3 - 1)
        | ⟨6, _⟩ => fun _ =>
            ChanTy.read .nat s₀ io.out1 (io.write1 pid + io.Bout1 - 1)
        | ⟨_+7, _⟩ => fun _ =>
            ChanTy.read .nat s₀ io.out2 (io.write2 pid + io.Bout2 - 1))
      s₀ hpid rfl rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => by
            have hj : j.val < io.B1 := j.isLt
            have h : io.read1 pid + j.val < A.extent io.in1 := by omega
            exact h
        | ⟨1, _⟩ => fun j _ => by
            have hj : j.val < io.B2 := j.isLt
            have h : io.read2 pid + j.val < A.extent io.in2 := by omega
            exact h
        | ⟨2, _⟩ => fun j _ => by
            have hj : j.val < io.B3 := j.isLt
            have h : io.read3 pid + j.val < A.extent io.in3 := by omega
            exact h
        | ⟨3, _⟩ => fun _ hm => by
            have hm' : 0 < io.read1 pid + io.B1 := hm
            have h : io.read1 pid + io.B1 - 1 < A.extent io.in1 := by omega
            exact h
        | ⟨4, _⟩ => fun _ hm => by
            have hm' : 0 < io.read2 pid + io.B2 := hm
            have h : io.read2 pid + io.B2 - 1 < A.extent io.in2 := by omega
            exact h
        | ⟨5, _⟩ => fun _ hm => by
            have hm' : 0 < io.read3 pid + io.B3 := hm
            have h : io.read3 pid + io.B3 - 1 < A.extent io.in3 := by omega
            exact h
        | ⟨6, _⟩ => fun _ hm => by
            have hm' : 0 < io.write1 pid + io.Bout1 := hm
            have h : io.write1 pid + io.Bout1 - 1 < A.extent io.out1 := by omega
            exact h
        | ⟨_+7, _⟩ => fun _ hm => by
            have hm' : 0 < io.write2 pid + io.Bout2 := hm
            have h : io.write2 pid + io.Bout2 - 1 < A.extent io.out2 := by omega
            exact h)
      (fun o => match o with
        | ⟨0, _⟩ => fun j _ => by
            have hj : j.val < io.Bout1 := j.isLt
            have h : io.write1 pid + j.val < A.extent io.out1 := by omega
            exact h
        | ⟨_+1, _⟩ => fun j _ => by
            have hj : j.val < io.Bout2 := j.isLt
            have h : io.write2 pid + j.val < A.extent io.out2 := by omega
            exact h)
      (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j _ => hx j
        | ⟨1, _⟩ => fun j _ => hy j
        | ⟨2, _⟩ => fun j _ => hz j
        | ⟨3, _⟩ => fun _ _ => rfl
        | ⟨4, _⟩ => fun _ _ => rfl
        | ⟨5, _⟩ => fun _ _ => rfl
        | ⟨6, _⟩ => fun _ _ => rfl
        | ⟨_+7, _⟩ => fun _ _ => rfl)
  refine ⟨s', hexec, fun j => hval (⟨0, by decide⟩ : Fin 2) j True.intro,
    fun j => hval (⟨1, by decide⟩ : Fin 2) j True.intro, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | ⟨hn1, hn2⟩
  · exact Or.inl hflat
  · exact Or.inr ⟨fun oc => match oc with
      | ⟨0, _⟩ => fun j _ => hn1 j
      | ⟨_+1, _⟩ => fun j _ => hn2 j,
      fun t => t.elim0⟩

end KernelIO₃ₓ₂

/-- IO signature of a **one-input / two-output** kernel (`₁ₓ₂` = "1 × 2") —
e.g. a statistics kernel producing mean and variance from one tile. Carries
the same field vocabulary as the rest of the family. Its only relation so
far is `Equiv` (the kernel-equivalence surface); the correctness `⊨` form
can be added alongside when a showcase needs it. -/
structure KernelIO₁ₓ₂ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- Input buffer. -/
  inp : RegionName
  /-- First output buffer. -/
  out1 : RegionName
  /-- Second output buffer. -/
  out2 : RegionName
  /-- Input tile length. -/
  Bin : Nat
  /-- First output tile length. -/
  Bout1 : Nat
  /-- Second output tile length. -/
  Bout2 : Nat
  /-- Where program `pid` reads its input tile: `[read pid, read pid + Bin)`. -/
  read : Nat → Nat
  /-- Where program `pid` writes its `out1` tile. -/
  write1 : Nat → Nat
  /-- Where program `pid` writes its `out2` tile. -/
  write2 : Nat → Nat
  /-- This kernel's private working buffers (see `ScratchSpec`). Empty for
  kernels that stage nothing through memory. -/
  scratch : List ScratchSpec := []

namespace KernelIO₁ₓ₂

/-- `io₁ ≡[R] io₂` — kernel equivalence on a shared one-input / two-output
IO signature; sibling of `MaskedKernelIO₂.Equiv`. The interface is read from
`io₁` (instances share it by structure update); `io₂` contributes only its
`kernel` and its private `scratch`. Both output windows must agree; each
side frames outside the **union** of the two output windows ∪ its own
scratch. No input hypotheses: equal inputs are "the same `s₀`". -/
def Equiv (io₁ io₂ : KernelIO₁ₓ₂) (R : RoundingModel) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io₁.inp, io₁.out1, io₁.out2]
      ++ (io₁.scratch.map (·.buf) ++ io₂.scratch.map (·.buf)) →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    io₁.read pid + io₁.Bin ≤ A.extent io₁.inp →
    io₁.write1 pid + io₁.Bout1 ≤ A.extent io₁.out1 →
    io₁.write2 pid + io₁.Bout2 ≤ A.extent io₁.out2 →
    (∀ p ∈ io₁.scratch, p.win pid + p.len ≤ A.extent p.buf) →
    (∀ p ∈ io₂.scratch, p.win pid + p.len ≤ A.extent p.buf) →
  ∀ s₀ : BlockState,
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    ∃ s₁ s₂,
      execR R (A.flattenKernel io₁.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₁
      ∧ execR R (A.flattenKernel io₂.kernel.toAlgKernel) (A.flattenState s₀)
        = some s₂
      ∧ (∀ j : Fin io₁.Bout1,
          s₁.readMem A.flat (A.addr io₁.out1 (io₁.write1 pid + j.val))
            = s₂.readMem A.flat (A.addr io₁.out1 (io₁.write1 pid + j.val)))
      ∧ (∀ j : Fin io₁.Bout2,
          s₁.readMem A.flat (A.addr io₁.out2 (io₁.write2 pid + j.val))
            = s₂.readMem A.flat (A.addr io₁.out2 (io₁.write2 pid + j.val)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout1,
                o' ≠ A.addr io₁.out1 (io₁.write1 pid + j.val)) ∧
             (∀ j : Fin io₁.Bout2,
                o' ≠ A.addr io₁.out2 (io₁.write2 pid + j.val)) ∧
             (∀ p ∈ io₁.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₁.mem r' o' = (A.flattenState s₀).mem r' o')
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ((∀ j : Fin io₁.Bout1,
                o' ≠ A.addr io₁.out1 (io₁.write1 pid + j.val)) ∧
             (∀ j : Fin io₁.Bout2,
                o' ≠ A.addr io₁.out2 (io₁.write2 pid + j.val)) ∧
             (∀ p ∈ io₂.scratch, ∀ k : Fin p.len,
                o' ≠ A.addr p.buf (p.win pid + k.val)))) →
          s₂.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io₁ " ≡[" R "] " io₂ =>
  KernelIO₁ₓ₂.Equiv io₁ io₂ R

/-- Assembly lemma for the one-input / two-output `≡[R]` — sibling of
`MaskedKernelIO₂.Equiv.intro`; see there for the reading of the
obligations. The region-model frames take THREE conditions: one per output
window, one for the respective kernel's scratch. -/
theorem Equiv.intro (io₁ io₂ : KernelIO₁ₓ₂) {R : RoundingModel}
    (hok₁ : (io₁.kernel.toAlgKernel).FlattenOk)
    (hok₂ : (io₂.kernel.toAlgKernel).FlattenOk)
    (hts₁ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read s.pid + io₁.Bin ≤ bounds io₁.inp →
      io₁.write1 s.pid + io₁.Bout1 ≤ bounds io₁.out1 →
      io₁.write2 s.pid + io₁.Bout2 ≤ bounds io₁.out2 →
      (∀ p ∈ io₁.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₁.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hts₂ : ∀ (bounds : RegionBounds) (s : BlockState),
      io₁.read s.pid + io₁.Bin ≤ bounds io₁.inp →
      io₁.write1 s.pid + io₁.Bout1 ≤ bounds io₁.out1 →
      io₁.write2 s.pid + io₁.Bout2 ≤ bounds io₁.out2 →
      (∀ p ∈ io₂.scratch, p.win s.pid + p.len ≤ bounds p.buf) →
      (io₂.kernel.toAlgKernel).TraceSafeR R bounds s)
    (hrun : ∀ s₀ : BlockState,
      ∃ s1 s2,
        execR R (io₁.kernel.toAlgKernel) s₀ = some s1
        ∧ execR R (io₂.kernel.toAlgKernel) s₀ = some s2
        ∧ (∀ j : Fin io₁.Bout1,
            s1.readMem io₁.out1 (io₁.write1 s₀.pid + j.val)
              = s2.readMem io₁.out1 (io₁.write1 s₀.pid + j.val))
        ∧ (∀ j : Fin io₁.Bout2,
            s1.readMem io₁.out2 (io₁.write2 s₀.pid + j.val)
              = s2.readMem io₁.out2 (io₁.write2 s₀.pid + j.val))
        ∧ (∀ r o,
            (r ≠ io₁.out1 ∨
              ∀ j : Fin io₁.Bout1, o ≠ io₁.write1 s₀.pid + j.val) →
            (r ≠ io₁.out2 ∨
              ∀ j : Fin io₁.Bout2, o ≠ io₁.write2 s₀.pid + j.val) →
            (∀ p ∈ io₁.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s1.mem r o = s₀.mem r o)
        ∧ (∀ r o,
            (r ≠ io₁.out1 ∨
              ∀ j : Fin io₁.Bout1, o ≠ io₁.write1 s₀.pid + j.val) →
            (r ≠ io₁.out2 ∨
              ∀ j : Fin io₁.Bout2, o ≠ io₁.write2 s₀.pid + j.val) →
            (∀ p ∈ io₂.scratch, r = p.buf →
              ∀ k : Fin p.len, o ≠ p.win s₀.pid + k.val) →
            s2.mem r o = s₀.mem r o)) :
    io₁.Equiv io₂ R := by
  intro A hd hregs hcov pid h1 h2 h3 hsc1 hsc2 s₀ hpid hu
  subst hpid
  obtain ⟨s1, s2, hexec1, hexec2, hval1, hval2, hframe1, hframe2⟩ := hrun s₀
  have hts₁' : (io₁.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₁ A.extent s₀ h1 h2 h3 hsc1
  have hts₂' : (io₂.kernel.toAlgKernel).TraceSafeR R A.extent s₀ :=
    hts₂ A.extent s₀ h1 h2 h3 hsc2
  have hbridge1 := A.execR_flatten hd hcov R _ s₀ hts₁' hok₁ hu
  have hbridge2 := A.execR_flatten hd hcov R _ s₀ hts₂' hok₂ hu
  have hmem1 : io₁.out1 ∈ A.regions := by rw [hregs]; simp
  have hmem2 : io₁.out2 ∈ A.regions := by rw [hregs]; simp
  refine ⟨A.flattenState s1, A.flattenState s2, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hbridge1, hexec1, Option.map_some]
  · rw [hbridge2, hexec2, Option.map_some]
  · intro j
    have hlt : io₁.write1 s₀.pid + j.val < A.extent io₁.out1 := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem1 hlt,
        A.flattenState_readMem hd s2 hmem1 hlt]
    exact hval1 j
  · intro j
    have hlt : io₁.write2 s₀.pid + j.val < A.extent io₁.out2 := by
      have := j.isLt; omega
    rw [A.flattenState_readMem hd s1 hmem2 hlt,
        A.flattenState_readMem hd s2 hmem2 hlt]
    exact hval2 j
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
          refine congrArg A.trCell (hframe1 r o ?_ ?_ ?_)
          · by_cases hro : r = io₁.out1
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hn1, _, _⟩
              · exact hflat rfl
              · exact hn1 j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · by_cases hro : r = io₁.out2
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨_, hn2, _⟩
              · exact hflat rfl
              · exact hn2 j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, _, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]
  · intro r' o' hcond
    by_cases hr : r' = A.flat
    · subst hr
      show (A.flattenState s2).mem A.flat o'
          = (A.flattenState s₀).mem A.flat o'
      simp only [FlatAlloc.flattenState]
      unfold FlatAlloc.readFlat
      cases hdec : A.decode o' with
      | none => rfl
      | some p =>
          obtain ⟨r, o⟩ := p
          obtain ⟨hrmem, hoeq, holt⟩ := A.decode_sound hdec
          show A.trCell (s2.mem r o) = A.trCell (s₀.mem r o)
          refine congrArg A.trCell (hframe2 r o ?_ ?_ ?_)
          · by_cases hro : r = io₁.out1
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨hn1, _, _⟩
              · exact hflat rfl
              · exact hn1 j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · by_cases hro : r = io₁.out2
            · subst hro
              refine Or.inr fun j hoj => ?_
              rcases hcond with hflat | ⟨_, hn2, _⟩
              · exact hflat rfl
              · exact hn2 j (by rw [hoeq, hoj])
            · exact Or.inl hro
          · intro p hp hrp k hok'
            rcases hcond with hflat | ⟨_, _, hnscr⟩
            · exact hflat rfl
            · exact hnscr p hp k (by rw [hoeq, hrp, hok'])
    · simp only [FlatAlloc.flattenState, if_neg hr]

end KernelIO₁ₓ₂

end VeriTile.Triton
