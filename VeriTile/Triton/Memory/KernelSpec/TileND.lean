/-
Kernel IO contracts: TileND.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.TileIndex

namespace VeriTile.Triton


/-! ## Tile-indexed masked IO over **all three program axes**

`MaskedTileKernelIO₁`'s windows are functions of one program id. Plenty of kernels
own a tile whose address is built from two or three axes — a `(batch, head)` row, a
`(chunk, slice, batch·head)` block — with a single input and a single output.

`Masked3DTileKernelIO₁` is that skin: same one-in / one-out masked tile triple, with
every window and mask taking `pid₀ pid₁ pid₂`. Kernels using fewer axes ignore the
extra arguments, so this subsumes the one-axis skin rather than competing with it;
`MaskedTileKernelIO₁` stays because its narrower signature is what a one-axis
headline should print.

One honest limit, inherited from the core rather than chosen here: the unified
`UKernelIO.Implements`' spec function receives `pid₀` and `pid₁` only, so `f` here
does too. That is enough for every consumer so far (their third axis enters the
*addresses*, not the value), and widening it would mean touching `KernelCore`. -/

/-- One-input / one-output masked tile IO whose windows read all three program
axes. -/
structure Masked3DTileKernelIO₁ where
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
  /-- The tile footprint each program instance owns. -/
  shape : TileShape
  /-- Lane `i`'s read address for program `(pid₀, pid₁, pid₂)`. -/
  read : Nat → Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s write address. -/
  write : Nat → Nat → Nat → TileIndex shape → Nat
  /-- Read-active lanes. -/
  mask : Nat → Nat → Nat → TileIndex shape → Prop
  /-- Write-active lanes; defaults to `mask`. -/
  writeMask : Nat → Nat → Nat → TileIndex shape → Prop := mask

namespace Masked3DTileKernelIO₁

/-- `io.Implements f` — the three-axis sibling of `MaskedTileKernelIO₁.Implements`.
Windows and masks see all three program ids; `f` sees the first two (see the
section docstring for why). -/
def Implements (io : Masked3DTileKernelIO₁)
    (f : Nat → Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ) :
    Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ pid₂ i →
      io.read pid₀ pid₁ pid₂ i < A.extent io.inp) →
    (∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ pid₂ i →
      io.write pid₀ pid₁ pid₂ i < A.extent io.out) →
  ∀ (xs : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ pid₂ i →
      s₀.readMem io.inp (io.read pid₀ pid₁ pid₂ i) = xs i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ pid₂ i →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ pid₂ i))
            = f pid₀ pid₁ xs i)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ pid₂ i →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ i)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Masked3DTileKernelIO₁.Implements

/-- Embed into the unified core. -/
private def toU (io : Masked3DTileKernelIO₁) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 1
  nOut := 1
  nScr := 0
  bufs := [io.inp, io.out]
  ity := fun _ => .float
  iarity := fun _ => (TileShape.allIndices io.shape).length
  ibuf := fun _ => io.inp
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shape).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun _ _ p₀ p₁ p₂ j =>
    io.read p₀ p₁ p₂ ((TileShape.allIndices io.shape).get j)
  imask := fun _ _ p₀ p₁ p₂ j =>
    io.mask p₀ p₁ p₂ ((TileShape.allIndices io.shape).get j)
  owin := fun _ _ p₀ p₁ p₂ j =>
    io.write p₀ p₁ p₂ ((TileShape.allIndices io.shape).get j)
  omask := fun _ _ p₀ p₁ p₂ j =>
    io.writeMask p₀ p₁ p₂ ((TileShape.allIndices io.shape).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma for the three-axis tile family. -/
theorem Implements.intro (io : Masked3DTileKernelIO₁)
    {f : Nat → Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape,
        io.mask (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.read (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.inp) →
      (∀ i : TileIndex io.shape,
        io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : TileIndex io.shape → ℝ),
      (∀ i : TileIndex io.shape,
        io.mask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
          = xs i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ i : TileIndex io.shape,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
              = f (s₀.pids 0) (s₀.pids 1) xs i)
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ i : TileIndex io.shape,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
                o' ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      refine hts bounds s ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨0, by decide⟩ : Fin 1) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape,
          io.mask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
          s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
            = vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 1) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i)) hx
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ i : TileIndex io.shape,
          io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
          r ≠ io.out ∨ o' ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun i hi => ?_
        rcases hoc' i hi with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ pid₂ h1 h2 xs s₀ hp₀ hp₁ hp₂ hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ pid₂
      (fun _ j => xs ((TileShape.allIndices io.shape).get j)) s₀ hp₀ hp₁ hp₂ hu
      (fun _i j hj => h1 _ hj) (fun _o j hj => h2 _ hj) (fun t => t.elim0)
      (fun _i j hj => hx _ hj)
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid₀ pid₁
        (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)
      = f pid₀ pid₁ xs ((TileShape.allIndices io.shape).get j)
    rw [hxs]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

/-! ### The rounding face of the three-axis tile skin -/

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
`io ⊨[R, outDType] f` for the three-axis tile skin: the same Hoare triple as
`Implements`, with the execution under `execR R` and the readback taken through
`readMemAs outDType`, so each written cell holds the *ideal* real value quantized
once at the declared grid. At `R := .triv` and `outDType := .real` the store is
exact and this degenerates to the exact face. -/
def ImplementsR (io : Masked3DTileKernelIO₁) (R : RoundingModel)
    (outDType : FloatDType)
    (f : Nat → Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ) :
    Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ pid₂ i →
      io.read pid₀ pid₁ pid₂ i < A.extent io.inp) →
    (∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ pid₂ i →
      io.write pid₀ pid₁ pid₂ i < A.extent io.out) →
  ∀ (xs : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ pid₂ i →
      s₀.readMem io.inp (io.read pid₀ pid₁ pid₂ i) = xs i) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ pid₂ i →
          s'.readMemAs outDType A.flat
              (A.addr io.out (io.write pid₀ pid₁ pid₂ i))
            = outDType.ofReal (R.round outDType (f pid₀ pid₁ xs i)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ pid₂ i →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ i)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  Masked3DTileKernelIO₁.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`Implements.intro`, riding the same `toU` embedding through the family's rounding
core `UKernelIO.ImplementsR.intro` (at the constant output grid
`fun _ => outDType`). Obligations are the usual three, with the safety walk at
`Kernel.TraceSafeR R` and `hrun` returning a rounded region-model triple. -/
theorem ImplementsR.intro (io : Masked3DTileKernelIO₁) {R : RoundingModel}
    {outDType : FloatDType}
    {f : Nat → Nat → (TileIndex io.shape → ℝ) → TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape,
        io.mask (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.read (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.inp) →
      (∀ i : TileIndex io.shape,
        io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.out) →
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : TileIndex io.shape → ℝ),
      (∀ i : TileIndex io.shape,
        io.mask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
          = xs i) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ i : TileIndex io.shape,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
            s1.readMemAs outDType io.out
                (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
              = outDType.ofReal
                  (R.round outDType (f (s₀.pids 0) (s₀.pids 1) xs i)))
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ i : TileIndex io.shape,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
                o' ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i) →
            s1.mem r o' = s₀.mem r o')) :
    io.ImplementsR R outDType f := by
  have hcore : io.toU.ImplementsR R (fun _ => outDType)
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      refine hts bounds s ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨0, by decide⟩ : Fin 1) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape,
          io.mask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
          s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
            = vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 1) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 1) (tilePos io.shape i)) hx
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ i : TileIndex io.shape,
          io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
          r ≠ io.out ∨ o' ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun i hi => ?_
        rcases hoc' i hi with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ pid₂ h1 h2 xs s₀ hp₀ hp₁ hp₂ hu hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ pid₂
      (fun _ j => xs ((TileShape.allIndices io.shape).get j)) s₀ hp₀ hp₁ hp₂ hu
      (fun _i j hj => h1 _ hj) (fun _o j hj => h2 _ hj) (fun t => t.elim0)
      (fun _i j hj => hx _ hj)
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show outDType.ofReal (R.round outDType (f pid₀ pid₁
        (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)))
      = outDType.ofReal (R.round outDType
          (f pid₀ pid₁ xs ((TileShape.allIndices io.shape).get j)))
    rw [hxs]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end Masked3DTileKernelIO₁


/-! ## Tile-indexed masked IO with **two scalar channels and three tile reads**

The layer-norm normalize genre reads two per-row *scalars* (a mean and a
reciprocal standard deviation), three *tiles* (the row of `X`, and the
column-indexed `W` / `B` parameter vectors), and writes one tile. Six buffers, two
program axes, one tail mask on everything tile-shaped.

`Scalar2Tile3KernelIO` states it. All five input channels are `.float`; what varies
is the **arity** — 1 for the two scalars, the tile's lane count for the other three
— so `toU`'s per-channel matches enumerate every `Fin 5` pattern (a catch-all leaves
`Fin (iarity i)` unreduced, exactly as in `Meta3MaskedTileKernelIO₁`).

Stating the two scalars as *channels* rather than as hypotheses about closed forms is
what makes the resulting headline stronger than the per-write-map summaries this
replaces: those must assume the mean/rstd cells already hold the reduction's closed
form, whereas here they are simply the values the kernel loaded. -/

/-- Two scalar reads / three tile reads / one tile write, over two program axes. -/
structure Scalar2Tile3KernelIO where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First scalar buffer. -/
  sbuf1 : RegionName
  /-- Second scalar buffer. -/
  sbuf2 : RegionName
  /-- First tile-read buffer. -/
  tbuf1 : RegionName
  /-- Second tile-read buffer. -/
  tbuf2 : RegionName
  /-- Third tile-read buffer. -/
  tbuf3 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- The tile footprint each program instance owns. -/
  shape : TileShape
  /-- First scalar's cell address for program `(pid₀, pid₁)`. -/
  swin1 : Nat → Nat → Nat
  /-- Second scalar's cell address. -/
  swin2 : Nat → Nat → Nat
  /-- Lane `i`'s `tbuf1` read address. -/
  read1 : Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s `tbuf2` read address. -/
  read2 : Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s `tbuf3` read address. -/
  read3 : Nat → Nat → TileIndex shape → Nat
  /-- Lane `i`'s write address. -/
  write : Nat → Nat → TileIndex shape → Nat
  /-- Read-active lanes, shared by the three tile channels. -/
  mask : Nat → Nat → TileIndex shape → Prop
  /-- Write-active lanes; defaults to `mask`. -/
  writeMask : Nat → Nat → TileIndex shape → Prop := mask

namespace Scalar2Tile3KernelIO

/-- `io.Implements f` — the normalize-genre Hoare triple. `f` takes both program
ids, the two loaded scalars, and the three loaded tiles. -/
def Implements (io : Scalar2Tile3KernelIO)
    (f : Nat → Nat → ℝ → ℝ → (TileIndex io.shape → ℝ) →
      (TileIndex io.shape → ℝ) → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.sbuf1, io.sbuf2, io.tbuf1, io.tbuf2, io.tbuf3, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
    io.swin1 pid₀ pid₁ < A.extent io.sbuf1 →
    io.swin2 pid₀ pid₁ < A.extent io.sbuf2 →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      io.read1 pid₀ pid₁ i < A.extent io.tbuf1) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      io.read2 pid₀ pid₁ i < A.extent io.tbuf2) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      io.read3 pid₀ pid₁ i < A.extent io.tbuf3) →
    (∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ i →
      io.write pid₀ pid₁ i < A.extent io.out) →
  ∀ (m1 m2 : ℝ) (xs ws bs : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    s₀.readMem io.sbuf1 (io.swin1 pid₀ pid₁) = m1 →
    s₀.readMem io.sbuf2 (io.swin2 pid₀ pid₁) = m2 →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      s₀.readMem io.tbuf1 (io.read1 pid₀ pid₁ i) = xs i) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      s₀.readMem io.tbuf2 (io.read2 pid₀ pid₁ i) = ws i) →
    (∀ i : TileIndex io.shape, io.mask pid₀ pid₁ i →
      s₀.readMem io.tbuf3 (io.read3 pid₀ pid₁ i) = bs i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ i →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ i))
            = f pid₀ pid₁ m1 m2 xs ws bs i)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ i : TileIndex io.shape, io.writeMask pid₀ pid₁ i →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ i)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Scalar2Tile3KernelIO.Implements

/-- Embed into the unified core: five `.float` input channels whose **arities**
differ (1, 1, then the tile's lane count three times), one `.float` output. -/
private def toU (io : Scalar2Tile3KernelIO) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 5
  nOut := 1
  nScr := 0
  bufs := [io.sbuf1, io.sbuf2, io.tbuf1, io.tbuf2, io.tbuf3, io.out]
  ity := fun _ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨2, _⟩ => (TileShape.allIndices io.shape).length
    | ⟨3, _⟩ => (TileShape.allIndices io.shape).length
    | ⟨4, _⟩ => (TileShape.allIndices io.shape).length
    | ⟨_ + 5, h⟩ => absurd h (by omega)
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.sbuf1
    | ⟨1, _⟩ => io.sbuf2
    | ⟨2, _⟩ => io.tbuf1
    | ⟨3, _⟩ => io.tbuf2
    | _ => io.tbuf3
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shape).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.swin1 p₀ p₁
    | ⟨1, _⟩ => fun _ => io.swin2 p₀ p₁
    | ⟨2, _⟩ => fun j =>
        io.read1 p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | ⟨3, _⟩ => fun j =>
        io.read2 p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | ⟨4, _⟩ => fun j =>
        io.read3 p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | ⟨_ + 5, h⟩ => absurd h (by omega)
  imask := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun j => io.mask p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | ⟨3, _⟩ => fun j => io.mask p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | ⟨4, _⟩ => fun j => io.mask p₀ p₁ ((TileShape.allIndices io.shape).get j)
    | ⟨_ + 5, h⟩ => absurd h (by omega)
  owin := fun _ _ p₀ p₁ _ j =>
    io.write p₀ p₁ ((TileShape.allIndices io.shape).get j)
  omask := fun _ _ p₀ p₁ _ j =>
    io.writeMask p₀ p₁ ((TileShape.allIndices io.shape).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma for the normalize genre. -/
theorem Implements.intro (io : Scalar2Tile3KernelIO)
    {f : Nat → Nat → ℝ → ℝ → (TileIndex io.shape → ℝ) →
      (TileIndex io.shape → ℝ) → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      io.swin1 (s.pids 0) (s.pids 1) < bounds io.sbuf1 →
      io.swin2 (s.pids 0) (s.pids 1) < bounds io.sbuf2 →
      (∀ i : TileIndex io.shape, io.mask (s.pids 0) (s.pids 1) i →
        io.read1 (s.pids 0) (s.pids 1) i < bounds io.tbuf1) →
      (∀ i : TileIndex io.shape, io.mask (s.pids 0) (s.pids 1) i →
        io.read2 (s.pids 0) (s.pids 1) i < bounds io.tbuf2) →
      (∀ i : TileIndex io.shape, io.mask (s.pids 0) (s.pids 1) i →
        io.read3 (s.pids 0) (s.pids 1) i < bounds io.tbuf3) →
      (∀ i : TileIndex io.shape, io.writeMask (s.pids 0) (s.pids 1) i →
        io.write (s.pids 0) (s.pids 1) i < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (m1 m2 : ℝ)
        (xs ws bs : TileIndex io.shape → ℝ),
      s₀.readMem io.sbuf1 (io.swin1 (s₀.pids 0) (s₀.pids 1)) = m1 →
      s₀.readMem io.sbuf2 (io.swin2 (s₀.pids 0) (s₀.pids 1)) = m2 →
      (∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.tbuf1 (io.read1 (s₀.pids 0) (s₀.pids 1) i) = xs i) →
      (∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.tbuf2 (io.read2 (s₀.pids 0) (s₀.pids 1) i) = ws i) →
      (∀ i : TileIndex io.shape, io.mask (s₀.pids 0) (s₀.pids 1) i →
        s₀.readMem io.tbuf3 (io.read3 (s₀.pids 0) (s₀.pids 1) i) = bs i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ i : TileIndex io.shape, io.writeMask (s₀.pids 0) (s₀.pids 1) i →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) i)
              = f (s₀.pids 0) (s₀.pids 1) m1 m2 xs ws bs i)
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ i : TileIndex io.shape, io.writeMask (s₀.pids 0) (s₀.pids 1) i →
                o' ≠ io.write (s₀.pids 0) (s₀.pids 1) i) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (vals (⟨0, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1))
          (fun i => vals (⟨2, by decide⟩ : Fin 5) (tilePos io.shape i))
          (fun i => vals (⟨3, by decide⟩ : Fin 5) (tilePos io.shape i))
          (fun i => vals (⟨4, by decide⟩ : Fin 5) (tilePos io.shape i))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      refine hts bounds s
        (hib (⟨0, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨1, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1) trivial)
        ?_ ?_ ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨2, by decide⟩ : Fin 5) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨3, by decide⟩ : Fin 5) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨4, by decide⟩ : Fin 5) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have h2 : ∀ i : TileIndex io.shape,
          io.mask (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.tbuf1 (io.read1 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨2, by decide⟩ : Fin 5) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨2, by decide⟩ : Fin 5) j
      have h3 : ∀ i : TileIndex io.shape,
          io.mask (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.tbuf2 (io.read2 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨3, by decide⟩ : Fin 5) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨3, by decide⟩ : Fin 5) j
      have h4 : ∀ i : TileIndex io.shape,
          io.mask (s₀.pids 0) (s₀.pids 1) i →
          s₀.readMem io.tbuf3 (io.read3 (s₀.pids 0) (s₀.pids 1) i)
            = vals (⟨4, by decide⟩ : Fin 5) (tilePos io.shape i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨4, by decide⟩ : Fin 5) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ _ _ (fun i => vals (⟨2, by decide⟩ : Fin 5) (tilePos io.shape i))
          (fun i => vals (⟨3, by decide⟩ : Fin 5) (tilePos io.shape i))
          (fun i => vals (⟨4, by decide⟩ : Fin 5) (tilePos io.shape i))
          (hpins (⟨0, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1) trivial)
          (hpins (⟨1, by decide⟩ : Fin 5) (⟨0, by decide⟩ : Fin 1) trivial)
          h2 h3 h4
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ i : TileIndex io.shape,
          io.writeMask (s₀.pids 0) (s₀.pids 1) i →
          r ≠ io.out ∨ o' ≠ io.write (s₀.pids 0) (s₀.pids 1) i :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun i hi => ?_
        rcases hoc' i hi with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ hb1 hb2 h1 h2 h3 h4 m1 m2 xs ws bs s₀ hp₀ hp₁
    hu hm1 hm2 hx1 hx2 hx3
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => m1
        | ⟨1, _⟩ => fun _ => m2
        | ⟨2, _⟩ => fun j => xs ((TileShape.allIndices io.shape).get j)
        | ⟨3, _⟩ => fun j => ws ((TileShape.allIndices io.shape).get j)
        | ⟨4, _⟩ => fun j => bs ((TileShape.allIndices io.shape).get j)
        | ⟨_ + 5, h⟩ => absurd h (by simp only [toU]; omega))
      s₀ hp₀ hp₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hb1
        | ⟨1, _⟩ => fun _ _ => hb2
        | ⟨2, _⟩ => fun j hj => h1 _ hj
        | ⟨3, _⟩ => fun j hj => h2 _ hj
        | ⟨4, _⟩ => fun j hj => h3 _ hj
        | ⟨_ + 5, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => h4 _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hm1
        | ⟨1, _⟩ => fun _ _ => hm2
        | ⟨2, _⟩ => fun j hj => hx1 _ hj
        | ⟨3, _⟩ => fun j hj => hx2 _ hj
        | ⟨4, _⟩ => fun j hj => hx3 _ hj
        | ⟨_ + 5, h⟩ => absurd h (by simp only [toU]; omega))
  have hround : ∀ g : TileIndex io.shape → ℝ,
      (fun i => g ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        = g :=
    fun g => funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid₀ pid₁ m1 m2
        (fun i => xs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        (fun i => ws ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        (fun i => bs ((TileShape.allIndices io.shape).get (tilePos io.shape i)))
        ((TileShape.allIndices io.shape).get j)
      = f pid₀ pid₁ m1 m2 xs ws bs ((TileShape.allIndices io.shape).get j)
    rw [hround xs, hround ws, hround bs]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end Scalar2Tile3KernelIO


/-! ## Tile-indexed masked IO with **per-channel shapes over three program axes**

`MaskedTileShapedKernelIO₂` gives two input channels their own tile shapes, which
is what a *contraction* needs — the spec may read both inputs at lanes the output
index does not name. It sees two program ids, though, and the fused-recurrent
genre launches on three (`i_v`, `i_k`, `i_bh`): its per-step slices read a
`[BV, BK]` state tile and a `[BK]` row and write a `[BV]` row, with every address
built from all three axes.

`Masked3DTileShapedKernelIO₂` is that skin at three axes. It is the pointwise
merge of `MaskedTileShapedKernelIO₂` (per-channel shapes) and
`Masked3DTileKernelIO₁` (three axes in the windows, two in `f`); the core needed
no change for either, and needs none for the merge. -/

/-- Two inputs with independent tile shapes, one output with its own, over three
program axes. -/
structure Masked3DTileShapedKernelIO₂ where
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
  /-- First input's tile shape. -/
  shape1 : TileShape
  /-- Second input's tile shape. -/
  shape2 : TileShape
  /-- Output's tile shape. -/
  shapeOut : TileShape
  /-- Lane `i`'s `in1` read address for program `(pid₀, pid₁, pid₂)`. -/
  read1 : Nat → Nat → Nat → TileIndex shape1 → Nat
  /-- Lane `i`'s `in2` read address. -/
  read2 : Nat → Nat → Nat → TileIndex shape2 → Nat
  /-- Lane `o`'s write address. -/
  write : Nat → Nat → Nat → TileIndex shapeOut → Nat
  /-- `in1`'s read-active lanes. -/
  mask1 : Nat → Nat → Nat → TileIndex shape1 → Prop
  /-- `in2`'s read-active lanes. -/
  mask2 : Nat → Nat → Nat → TileIndex shape2 → Prop
  /-- The output's write-active lanes. -/
  writeMask : Nat → Nat → Nat → TileIndex shapeOut → Prop

namespace Masked3DTileShapedKernelIO₂

/-- `io.Implements f` — the three-axis sibling of
`MaskedTileShapedKernelIO₂.Implements`. Windows and masks see all three program
ids; `f` sees the first two, as everywhere in the family (the core's spec
function takes `pid₀ pid₁`). A contraction along the second input's shape is
just a spec that reads both lane functions at indices the output lane does not
name. -/
def Implements (io : Masked3DTileShapedKernelIO₂)
    (f : Nat → Nat → (TileIndex io.shape1 → ℝ) → (TileIndex io.shape2 → ℝ) →
      TileIndex io.shapeOut → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
    (∀ i : TileIndex io.shape1, io.mask1 pid₀ pid₁ pid₂ i →
      io.read1 pid₀ pid₁ pid₂ i < A.extent io.in1) →
    (∀ i : TileIndex io.shape2, io.mask2 pid₀ pid₁ pid₂ i →
      io.read2 pid₀ pid₁ pid₂ i < A.extent io.in2) →
    (∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ pid₂ o →
      io.write pid₀ pid₁ pid₂ o < A.extent io.out) →
  ∀ (xs : TileIndex io.shape1 → ℝ) (ys : TileIndex io.shape2 → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape1, io.mask1 pid₀ pid₁ pid₂ i →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ pid₂ i) = xs i) →
    (∀ i : TileIndex io.shape2, io.mask2 pid₀ pid₁ pid₂ i →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ pid₂ i) = ys i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ pid₂ o →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ pid₂ o))
            = f pid₀ pid₁ xs ys o)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ pid₂ o →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ o)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Masked3DTileShapedKernelIO₂.Implements

/-- Embed into the unified core: two float tile channels with **independent
arities**, one output with its own arity, no scratch. As in
`MaskedTileShapedKernelIO₂`, every per-channel match enumerates all three `Fin 2`
patterns — a catch-all leaves `i` unrefined, so `Fin (iarity i)` does not reduce
and the two different arities cannot typecheck. -/
private def toU (io : Masked3DTileShapedKernelIO₂) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 2
  nOut := 1
  nScr := 0
  bufs := [io.in1, io.in2, io.out]
  ity := fun _ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => (TileShape.allIndices io.shape1).length
    | ⟨1, _⟩ => (TileShape.allIndices io.shape2).length
    | ⟨_ + 2, h⟩ => absurd h (by omega)
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | _ => io.in2
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shapeOut).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ p₂ => match i with
    | ⟨0, _⟩ => fun j =>
        io.read1 p₀ p₁ p₂ ((TileShape.allIndices io.shape1).get j)
    | ⟨1, _⟩ => fun j =>
        io.read2 p₀ p₁ p₂ ((TileShape.allIndices io.shape2).get j)
    | ⟨_ + 2, h⟩ => absurd h (by omega)
  imask := fun i _ p₀ p₁ p₂ => match i with
    | ⟨0, _⟩ => fun j =>
        io.mask1 p₀ p₁ p₂ ((TileShape.allIndices io.shape1).get j)
    | ⟨1, _⟩ => fun j =>
        io.mask2 p₀ p₁ p₂ ((TileShape.allIndices io.shape2).get j)
    | ⟨_ + 2, h⟩ => absurd h (by omega)
  owin := fun _ _ p₀ p₁ p₂ j =>
    io.write p₀ p₁ p₂ ((TileShape.allIndices io.shapeOut).get j)
  omask := fun _ _ p₀ p₁ p₂ j =>
    io.writeMask p₀ p₁ p₂ ((TileShape.allIndices io.shapeOut).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma — the three-axis sibling of
`MaskedTileShapedKernelIO₂.Implements.intro`. -/
theorem Implements.intro (io : Masked3DTileShapedKernelIO₂)
    {f : Nat → Nat → (TileIndex io.shape1 → ℝ) → (TileIndex io.shape2 → ℝ) →
      TileIndex io.shapeOut → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape1,
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.in1) →
      (∀ i : TileIndex io.shape2,
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.in2) →
      (∀ o : TileIndex io.shapeOut,
        io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) o →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) o < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (xs : TileIndex io.shape1 → ℝ)
        (ys : TileIndex io.shape2 → ℝ),
      (∀ i : TileIndex io.shape1,
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
          = xs i) →
      (∀ i : TileIndex io.shape2,
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
          = ys i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ o : TileIndex io.shapeOut,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o)
              = f (s₀.pids 0) (s₀.pids 1) xs ys o)
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ o : TileIndex io.shapeOut,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o →
                o' ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun i => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape1 i))
          (fun i => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape2 i))
          ((TileShape.allIndices io.shapeOut).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      have h1 : ∀ i : TileIndex io.shape1,
          io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) i →
          io.read1 (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.in1 := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hib (⟨0, by decide⟩ : Fin 2) j
      have h2 : ∀ i : TileIndex io.shape2,
          io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) i →
          io.read2 (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.in2 := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hib (⟨1, by decide⟩ : Fin 2) j
      have h3 : ∀ o : TileIndex io.shapeOut,
          io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) o →
          io.write (s.pids 0) (s.pids 1) (s.pids 2) o < bounds io.out := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hob (⟨0, by decide⟩ : Fin 1) j
      exact hts bounds s h1 h2 h3
    · intro s₀ vals _hundef hpins
      have hx : ∀ i : TileIndex io.shape1,
          io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
          s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
            = vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape1 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 2) j
      have hy : ∀ i : TileIndex io.shape2,
          io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
          s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
            = vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape2 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨1, by decide⟩ : Fin 2) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shape1 i))
          (fun i => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape2 i)) hx hy
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ o : TileIndex io.shapeOut,
          io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o →
          r ≠ io.out ∨ o' ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        exact hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun o hoact => ?_
        rcases hoc' o hoact with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ pid₂ h1 h2 h3 xs ys s₀ hp₀ hp₁ hp₂ hu hx hy
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ pid₂
      (fun i => match i with
        | ⟨0, _⟩ => fun j => xs ((TileShape.allIndices io.shape1).get j)
        | ⟨1, _⟩ => fun j => ys ((TileShape.allIndices io.shape2).get j)
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      s₀ hp₀ hp₁ hp₂ hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => h1 _ hj
        | ⟨1, _⟩ => fun j hj => h2 _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => h3 _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx _ hj
        | ⟨1, _⟩ => fun j hj => hy _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
  have hxs :
      (fun i => xs ((TileShape.allIndices io.shape1).get (tilePos io.shape1 i)))
        = xs :=
    funext fun i => by rw [get_tilePos]
  have hys :
      (fun i => ys ((TileShape.allIndices io.shape2).get (tilePos io.shape2 i)))
        = ys :=
    funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid₀ pid₁
        (fun i =>
          xs ((TileShape.allIndices io.shape1).get (tilePos io.shape1 i)))
        (fun i =>
          ys ((TileShape.allIndices io.shape2).get (tilePos io.shape2 i)))
        ((TileShape.allIndices io.shapeOut).get j)
      = f pid₀ pid₁ xs ys ((TileShape.allIndices io.shapeOut).get j)
    rw [hxs, hys]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end Masked3DTileShapedKernelIO₂


/-! ## The same shape at **four** input channels

`Masked3DTileShapedKernelIO₂` covers a two-input contraction. The RWKV-6 state
update needs four: a `[BV, BK]` state tile, the `[BK]` key row, the `[BV]` value
row, and the `[BK]` decay row, contracted into a `[BV, BK]` state tile. The only
thing that changes is the channel count — the arities are still per-channel and
the core is still untouched. -/

/-- Four inputs with independent tile shapes, one output with its own, over three
program axes. -/
structure Masked3DTileShaped4KernelIO where
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
  /-- Fourth input buffer. -/
  in4 : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- First input's tile shape. -/
  shape1 : TileShape
  /-- Second input's tile shape. -/
  shape2 : TileShape
  /-- Third input's tile shape. -/
  shape3 : TileShape
  /-- Fourth input's tile shape. -/
  shape4 : TileShape
  /-- Output's tile shape. -/
  shapeOut : TileShape
  /-- Lane `i`'s `in1` read address for program `(pid₀, pid₁, pid₂)`. -/
  read1 : Nat → Nat → Nat → TileIndex shape1 → Nat
  /-- Lane `i`'s `in2` read address. -/
  read2 : Nat → Nat → Nat → TileIndex shape2 → Nat
  /-- Lane `i`'s `in3` read address. -/
  read3 : Nat → Nat → Nat → TileIndex shape3 → Nat
  /-- Lane `i`'s `in4` read address. -/
  read4 : Nat → Nat → Nat → TileIndex shape4 → Nat
  /-- Lane `o`'s write address. -/
  write : Nat → Nat → Nat → TileIndex shapeOut → Nat
  /-- `in1`'s read-active lanes. -/
  mask1 : Nat → Nat → Nat → TileIndex shape1 → Prop
  /-- `in2`'s read-active lanes. -/
  mask2 : Nat → Nat → Nat → TileIndex shape2 → Prop
  /-- `in3`'s read-active lanes. -/
  mask3 : Nat → Nat → Nat → TileIndex shape3 → Prop
  /-- `in4`'s read-active lanes. -/
  mask4 : Nat → Nat → Nat → TileIndex shape4 → Prop
  /-- The output's write-active lanes. -/
  writeMask : Nat → Nat → Nat → TileIndex shapeOut → Prop

namespace Masked3DTileShaped4KernelIO

/-- `io.Implements f` — the four-channel sibling of
`Masked3DTileShapedKernelIO₂.Implements`. -/
def Implements (io : Masked3DTileShaped4KernelIO)
    (f : Nat → Nat → (TileIndex io.shape1 → ℝ) → (TileIndex io.shape2 → ℝ) →
      (TileIndex io.shape3 → ℝ) → (TileIndex io.shape4 → ℝ) →
      TileIndex io.shapeOut → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.in1, io.in2, io.in3, io.in4, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ pid₂ : Nat,
    (∀ i : TileIndex io.shape1, io.mask1 pid₀ pid₁ pid₂ i →
      io.read1 pid₀ pid₁ pid₂ i < A.extent io.in1) →
    (∀ i : TileIndex io.shape2, io.mask2 pid₀ pid₁ pid₂ i →
      io.read2 pid₀ pid₁ pid₂ i < A.extent io.in2) →
    (∀ i : TileIndex io.shape3, io.mask3 pid₀ pid₁ pid₂ i →
      io.read3 pid₀ pid₁ pid₂ i < A.extent io.in3) →
    (∀ i : TileIndex io.shape4, io.mask4 pid₀ pid₁ pid₂ i →
      io.read4 pid₀ pid₁ pid₂ i < A.extent io.in4) →
    (∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ pid₂ o →
      io.write pid₀ pid₁ pid₂ o < A.extent io.out) →
  ∀ (x1 : TileIndex io.shape1 → ℝ) (x2 : TileIndex io.shape2 → ℝ)
    (x3 : TileIndex io.shape3 → ℝ) (x4 : TileIndex io.shape4 → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.pids 2 = pid₂ →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shape1, io.mask1 pid₀ pid₁ pid₂ i →
      s₀.readMem io.in1 (io.read1 pid₀ pid₁ pid₂ i) = x1 i) →
    (∀ i : TileIndex io.shape2, io.mask2 pid₀ pid₁ pid₂ i →
      s₀.readMem io.in2 (io.read2 pid₀ pid₁ pid₂ i) = x2 i) →
    (∀ i : TileIndex io.shape3, io.mask3 pid₀ pid₁ pid₂ i →
      s₀.readMem io.in3 (io.read3 pid₀ pid₁ pid₂ i) = x3 i) →
    (∀ i : TileIndex io.shape4, io.mask4 pid₀ pid₁ pid₂ i →
      s₀.readMem io.in4 (io.read4 pid₀ pid₁ pid₂ i) = x4 i) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ pid₂ o →
          s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ pid₂ o))
            = f pid₀ pid₁ x1 x2 x3 x4 o)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ o : TileIndex io.shapeOut, io.writeMask pid₀ pid₁ pid₂ o →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ pid₂ o)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => Masked3DTileShaped4KernelIO.Implements

/-- Embed into the unified core: four float tile channels with independent
arities, one output with its own arity, no scratch. -/
private def toU (io : Masked3DTileShaped4KernelIO) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 4
  nOut := 1
  nScr := 0
  bufs := [io.in1, io.in2, io.in3, io.in4, io.out]
  ity := fun _ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => (TileShape.allIndices io.shape1).length
    | ⟨1, _⟩ => (TileShape.allIndices io.shape2).length
    | ⟨2, _⟩ => (TileShape.allIndices io.shape3).length
    | ⟨3, _⟩ => (TileShape.allIndices io.shape4).length
    | ⟨_ + 4, h⟩ => absurd h (by omega)
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.in1
    | ⟨1, _⟩ => io.in2
    | ⟨2, _⟩ => io.in3
    | _ => io.in4
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shapeOut).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i _ p₀ p₁ p₂ => match i with
    | ⟨0, _⟩ => fun j =>
        io.read1 p₀ p₁ p₂ ((TileShape.allIndices io.shape1).get j)
    | ⟨1, _⟩ => fun j =>
        io.read2 p₀ p₁ p₂ ((TileShape.allIndices io.shape2).get j)
    | ⟨2, _⟩ => fun j =>
        io.read3 p₀ p₁ p₂ ((TileShape.allIndices io.shape3).get j)
    | ⟨3, _⟩ => fun j =>
        io.read4 p₀ p₁ p₂ ((TileShape.allIndices io.shape4).get j)
    | ⟨_ + 4, h⟩ => absurd h (by omega)
  imask := fun i _ p₀ p₁ p₂ => match i with
    | ⟨0, _⟩ => fun j =>
        io.mask1 p₀ p₁ p₂ ((TileShape.allIndices io.shape1).get j)
    | ⟨1, _⟩ => fun j =>
        io.mask2 p₀ p₁ p₂ ((TileShape.allIndices io.shape2).get j)
    | ⟨2, _⟩ => fun j =>
        io.mask3 p₀ p₁ p₂ ((TileShape.allIndices io.shape3).get j)
    | ⟨3, _⟩ => fun j =>
        io.mask4 p₀ p₁ p₂ ((TileShape.allIndices io.shape4).get j)
    | ⟨_ + 4, h⟩ => absurd h (by omega)
  owin := fun _ _ p₀ p₁ p₂ j =>
    io.write p₀ p₁ p₂ ((TileShape.allIndices io.shapeOut).get j)
  omask := fun _ _ p₀ p₁ p₂ j =>
    io.writeMask p₀ p₁ p₂ ((TileShape.allIndices io.shapeOut).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma — the four-channel sibling of
`Masked3DTileShapedKernelIO₂.Implements.intro`. -/
theorem Implements.intro (io : Masked3DTileShaped4KernelIO)
    {f : Nat → Nat → (TileIndex io.shape1 → ℝ) → (TileIndex io.shape2 → ℝ) →
      (TileIndex io.shape3 → ℝ) → (TileIndex io.shape4 → ℝ) →
      TileIndex io.shapeOut → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState),
      (∀ i : TileIndex io.shape1,
        io.mask1 (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.read1 (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.in1) →
      (∀ i : TileIndex io.shape2,
        io.mask2 (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.read2 (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.in2) →
      (∀ i : TileIndex io.shape3,
        io.mask3 (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.read3 (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.in3) →
      (∀ i : TileIndex io.shape4,
        io.mask4 (s.pids 0) (s.pids 1) (s.pids 2) i →
        io.read4 (s.pids 0) (s.pids 1) (s.pids 2) i < bounds io.in4) →
      (∀ o : TileIndex io.shapeOut,
        io.writeMask (s.pids 0) (s.pids 1) (s.pids 2) o →
        io.write (s.pids 0) (s.pids 1) (s.pids 2) o < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (x1 : TileIndex io.shape1 → ℝ)
        (x2 : TileIndex io.shape2 → ℝ) (x3 : TileIndex io.shape3 → ℝ)
        (x4 : TileIndex io.shape4 → ℝ),
      (∀ i : TileIndex io.shape1,
        io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
        s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
          = x1 i) →
      (∀ i : TileIndex io.shape2,
        io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
        s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
          = x2 i) →
      (∀ i : TileIndex io.shape3,
        io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
        s₀.readMem io.in3 (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
          = x3 i) →
      (∀ i : TileIndex io.shape4,
        io.mask4 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
        s₀.readMem io.in4 (io.read4 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
          = x4 i) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ o : TileIndex io.shapeOut,
            io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o →
            s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o)
              = f (s₀.pids 0) (s₀.pids 1) x1 x2 x3 x4 o)
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ o : TileIndex io.shapeOut,
                io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o →
                o' ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun i => vals (⟨0, by decide⟩ : Fin 4) (tilePos io.shape1 i))
          (fun i => vals (⟨1, by decide⟩ : Fin 4) (tilePos io.shape2 i))
          (fun i => vals (⟨2, by decide⟩ : Fin 4) (tilePos io.shape3 i))
          (fun i => vals (⟨3, by decide⟩ : Fin 4) (tilePos io.shape4 i))
          ((TileShape.allIndices io.shapeOut).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals _hpins hib hob _hsb
      refine hts bounds s ?_ ?_ ?_ ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨0, by decide⟩ : Fin 4) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨1, by decide⟩ : Fin 4) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨2, by decide⟩ : Fin 4) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨3, by decide⟩ : Fin 4) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have h1 : ∀ i : TileIndex io.shape1,
          io.mask1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
          s₀.readMem io.in1 (io.read1 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
            = vals (⟨0, by decide⟩ : Fin 4) (tilePos io.shape1 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 4) j
      have h2 : ∀ i : TileIndex io.shape2,
          io.mask2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
          s₀.readMem io.in2 (io.read2 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
            = vals (⟨1, by decide⟩ : Fin 4) (tilePos io.shape2 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨1, by decide⟩ : Fin 4) j
      have h3 : ∀ i : TileIndex io.shape3,
          io.mask3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
          s₀.readMem io.in3 (io.read3 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
            = vals (⟨2, by decide⟩ : Fin 4) (tilePos io.shape3 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨2, by decide⟩ : Fin 4) j
      have h4 : ∀ i : TileIndex io.shape4,
          io.mask4 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i →
          s₀.readMem io.in4 (io.read4 (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) i)
            = vals (⟨3, by decide⟩ : Fin 4) (tilePos io.shape4 i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨3, by decide⟩ : Fin 4) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun i => vals (⟨0, by decide⟩ : Fin 4) (tilePos io.shape1 i))
          (fun i => vals (⟨1, by decide⟩ : Fin 4) (tilePos io.shape2 i))
          (fun i => vals (⟨2, by decide⟩ : Fin 4) (tilePos io.shape3 i))
          (fun i => vals (⟨3, by decide⟩ : Fin 4) (tilePos io.shape4 i))
          h1 h2 h3 h4
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ o : TileIndex io.shapeOut,
          io.writeMask (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o →
          r ≠ io.out ∨ o' ≠ io.write (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) o :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun o hoact => ?_
        rcases hoc' o hoact with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ pid₂ hb1 hb2 hb3 hb4 hbo x1 x2 x3 x4 s₀
    hp₀ hp₁ hp₂ hu hx1 hx2 hx3 hx4
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ pid₂
      (fun i => match i with
        | ⟨0, _⟩ => fun j => x1 ((TileShape.allIndices io.shape1).get j)
        | ⟨1, _⟩ => fun j => x2 ((TileShape.allIndices io.shape2).get j)
        | ⟨2, _⟩ => fun j => x3 ((TileShape.allIndices io.shape3).get j)
        | ⟨3, _⟩ => fun j => x4 ((TileShape.allIndices io.shape4).get j)
        | ⟨_ + 4, h⟩ => absurd h (by simp only [toU]; omega))
      s₀ hp₀ hp₁ hp₂ hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hb1 _ hj
        | ⟨1, _⟩ => fun j hj => hb2 _ hj
        | ⟨2, _⟩ => fun j hj => hb3 _ hj
        | ⟨3, _⟩ => fun j hj => hb4 _ hj
        | ⟨_ + 4, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => hbo _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx1 _ hj
        | ⟨1, _⟩ => fun j hj => hx2 _ hj
        | ⟨2, _⟩ => fun j hj => hx3 _ hj
        | ⟨3, _⟩ => fun j hj => hx4 _ hj
        | ⟨_ + 4, h⟩ => absurd h (by simp only [toU]; omega))
  have e1 :
      (fun i => x1 ((TileShape.allIndices io.shape1).get (tilePos io.shape1 i)))
        = x1 := funext fun i => by rw [get_tilePos]
  have e2 :
      (fun i => x2 ((TileShape.allIndices io.shape2).get (tilePos io.shape2 i)))
        = x2 := funext fun i => by rw [get_tilePos]
  have e3 :
      (fun i => x3 ((TileShape.allIndices io.shape3).get (tilePos io.shape3 i)))
        = x3 := funext fun i => by rw [get_tilePos]
  have e4 :
      (fun i => x4 ((TileShape.allIndices io.shape4).get (tilePos io.shape4 i)))
        = x4 := funext fun i => by rw [get_tilePos]
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid₀ pid₁
        (fun i =>
          x1 ((TileShape.allIndices io.shape1).get (tilePos io.shape1 i)))
        (fun i =>
          x2 ((TileShape.allIndices io.shape2).get (tilePos io.shape2 i)))
        (fun i =>
          x3 ((TileShape.allIndices io.shape3).get (tilePos io.shape3 i)))
        (fun i =>
          x4 ((TileShape.allIndices io.shape4).get (tilePos io.shape4 i)))
        ((TileShape.allIndices io.shapeOut).get j)
      = f pid₀ pid₁ x1 x2 x3 x4 ((TileShape.allIndices io.shapeOut).get j)
    rw [e1, e2, e3, e4]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end Masked3DTileShaped4KernelIO

end VeriTile.Triton
