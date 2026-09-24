/-
Kernel IO contracts: TileGather.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.TileIndex

namespace VeriTile.Triton


/-! ## Tile-indexed masked IO for a **row gather**

`Meta1MaskedTileKernelIO₁`'s metadata channel is a single `.nat` cell. An
embedding gather needs a whole `.nat` *tile* of them — one token id per row — and
the data window reads all of them: lane `(row, col)` of `weight` sits at
`ids[row]·stride + col`. The read mask is value-dependent too (Python's
`id_mask = (token_ids >= vob_start) & (token_ids < vob_end)`), while the write
window and its mask are static.

`GatherTileKernelIO` states that: a `.nat` index channel over `shapeIdx`, a float
data channel over `shape` whose window *and* mask eat the loaded index function,
and one static float output over `shape`. The core already allows value-dependent
windows (`iwin` sees the whole pinned context, not just the pid), so this is
another thin wrapper. -/

/-- A `.nat` index tile, a float tile gathered through it, one float tile out. -/
structure GatherTileKernelIO where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- The `.nat` index buffer (a plain region; the `.nat` typing lives on the
  channel, not on the region). -/
  idxbuf : RegionName
  /-- Float data buffer, read through the loaded indices. -/
  inp : RegionName
  /-- Float output buffer. -/
  out : RegionName
  /-- The index tile's footprint. -/
  shapeIdx : TileShape
  /-- The data / output tile's footprint. -/
  shape : TileShape
  /-- Lane `i`'s index read address for program `pid` — static, so the context
  stays causal. -/
  readx : Nat → TileIndex shapeIdx → Nat
  /-- Lane `j`'s data read address, given the loaded index tile. -/
  read : Nat → (TileIndex shapeIdx → Nat) → TileIndex shape → Nat
  /-- Lane `j`'s write address (static). -/
  write : Nat → TileIndex shape → Nat
  /-- The index tile's read-active lanes. -/
  maskx : Nat → TileIndex shapeIdx → Prop
  /-- The data tile's read-active lanes, given the loaded index tile. -/
  readMask : Nat → (TileIndex shapeIdx → Nat) → TileIndex shape → Prop
  /-- The output's write-active lanes (static). -/
  writeMask : Nat → TileIndex shape → Prop

namespace GatherTileKernelIO

/-- `io.Implements f` — the row-gather Hoare triple. The index tile `ids` is
universally quantified and pinned on the index-active lanes; the data tile is
pinned on the (index-dependent) read-active lanes; `f` sees both, so a spec may
branch on the loaded indices exactly as the kernel's `id_mask` does. -/
def Implements (io : GatherTileKernelIO)
    (f : Nat → (TileIndex io.shapeIdx → Nat) → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.idxbuf, io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    (∀ i : TileIndex io.shapeIdx, io.maskx pid i →
      io.readx pid i < A.extent io.idxbuf) →
  ∀ ids : TileIndex io.shapeIdx → Nat,
    (∀ j : TileIndex io.shape, io.readMask pid ids j →
      io.read pid ids j < A.extent io.inp) →
    (∀ j : TileIndex io.shape, io.writeMask pid j →
      io.write pid j < A.extent io.out) →
  ∀ (xs : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shapeIdx, io.maskx pid i →
      s₀.readMemValue .nat io.idxbuf (io.readx pid i) = ids i) →
    (∀ j : TileIndex io.shape, io.readMask pid ids j →
      s₀.readMem io.inp (io.read pid ids j) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : TileIndex io.shape, io.writeMask pid j →
          s'.readMem A.flat (A.addr io.out (io.write pid j)) = f pid ids xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ j : TileIndex io.shape, io.writeMask pid j →
              o' ≠ A.addr io.out (io.write pid j)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => GatherTileKernelIO.Implements

/-- Embed into the unified core: a `.nat` tile channel, a `.float` tile channel
whose window and mask read it, one `.float` output. Both per-channel matches
enumerate every `Fin 2` pattern — type *and* arity differ across the two
channels, so a catch-all would leave both unreduced. -/
private def toU (io : GatherTileKernelIO) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 2
  nOut := 1
  nScr := 0
  bufs := [io.idxbuf, io.inp, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .nat
    | _ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => (TileShape.allIndices io.shapeIdx).length
    | _ => (TileShape.allIndices io.shape).length
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.idxbuf
    | _ => io.inp
  oty := fun _ => .float
  oarity := fun _ => (TileShape.allIndices io.shape).length
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun t => t.elim0
  sbuf := fun t => t.elim0
  iwin := fun i vals p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j =>
        io.readx p₀ ((TileShape.allIndices io.shapeIdx).get j)
    | ⟨1, _⟩ => fun j =>
        io.read p₀
          (fun k => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx k))
          ((TileShape.allIndices io.shape).get j)
    | ⟨_ + 2, h⟩ => absurd h (by omega)
  imask := fun i vals p₀ _ _ => match i with
    | ⟨0, _⟩ => fun j =>
        io.maskx p₀ ((TileShape.allIndices io.shapeIdx).get j)
    | ⟨1, _⟩ => fun j =>
        io.readMask p₀
          (fun k => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx k))
          ((TileShape.allIndices io.shape).get j)
    | ⟨_ + 2, h⟩ => absurd h (by omega)
  owin := fun _ _ p₀ _ _ j =>
    io.write p₀ ((TileShape.allIndices io.shape).get j)
  omask := fun _ _ p₀ _ _ j =>
    io.writeMask p₀ ((TileShape.allIndices io.shape).get j)
  swin := fun t _ _ _ _ _ => t.elim0
  smask := fun t _ _ _ _ _ => t.elim0

/-- Assembly lemma for the row-gather family. -/
theorem Implements.intro (io : GatherTileKernelIO)
    {f : Nat → (TileIndex io.shapeIdx → Nat) → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (ids : TileIndex io.shapeIdx → Nat),
      (∀ i : TileIndex io.shapeIdx, io.maskx s.pid i →
        s.readMemValue .nat io.idxbuf (io.readx s.pid i) = ids i) →
      (∀ i : TileIndex io.shapeIdx, io.maskx s.pid i →
        io.readx s.pid i < bounds io.idxbuf) →
      (∀ j : TileIndex io.shape, io.readMask s.pid ids j →
        io.read s.pid ids j < bounds io.inp) →
      (∀ j : TileIndex io.shape, io.writeMask s.pid j →
        io.write s.pid j < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (ids : TileIndex io.shapeIdx → Nat)
        (xs : TileIndex io.shape → ℝ),
      (∀ i : TileIndex io.shapeIdx, io.maskx s₀.pid i →
        s₀.readMemValue .nat io.idxbuf (io.readx s₀.pid i) = ids i) →
      (∀ j : TileIndex io.shape, io.readMask s₀.pid ids j →
        s₀.readMem io.inp (io.read s₀.pid ids j) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : TileIndex io.shape, io.writeMask s₀.pid j →
            s1.readMem io.out (io.write s₀.pid j) = f s₀.pid ids xs j)
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ j : TileIndex io.shape, io.writeMask s₀.pid j →
                o' ≠ io.write s₀.pid j) →
            s1.mem r o' = s₀.mem r o')) :
    io.Implements f := by
  have hcore : io.toU.Implements
      (fun p₀ _p₁ vals _o j =>
        f p₀ (fun k => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx k))
          (fun k => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape k))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      have hid : ∀ i : TileIndex io.shapeIdx, io.maskx s.pid i →
          s.readMemValue .nat io.idxbuf (io.readx s.pid i)
            = vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 2) j
      refine hts bounds s _ hid ?_ ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨0, by decide⟩ : Fin 2) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨1, by decide⟩ : Fin 2) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have hid : ∀ i : TileIndex io.shapeIdx, io.maskx s₀.pid i →
          s₀.readMemValue .nat io.idxbuf (io.readx s₀.pid i)
            = vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 2) j
      have hx : ∀ j : TileIndex io.shape,
          io.readMask s₀.pid
            (fun k => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx k)) j →
          s₀.readMem io.inp
              (io.read s₀.pid
                (fun k =>
                  vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx k)) j)
            = vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape j) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨1, by decide⟩ : Fin 2) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ _ (fun k => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape k))
          hid hx
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ j : TileIndex io.shape, io.writeMask s₀.pid j →
          r ≠ io.out ∨ o' ≠ io.write s₀.pid j :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j hj => ?_
        rcases hoc' j hj with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid hbx ids hbr hbw xs s₀ hpid hu hidpin hxpin
  have hids :
      (fun k => ids ((TileShape.allIndices io.shapeIdx).get
        (tilePos io.shapeIdx k))) = ids :=
    funext fun k => by rw [get_tilePos]
  have hxs :
      (fun k => xs ((TileShape.allIndices io.shape).get (tilePos io.shape k)))
        = xs :=
    funext fun k => by rw [get_tilePos]
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun j => ids ((TileShape.allIndices io.shapeIdx).get j)
        | ⟨1, _⟩ => fun j => xs ((TileShape.allIndices io.shape).get j)
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      s₀ hpid rfl rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hbx _ hj
        | ⟨1, _⟩ => fun j hj => by
            simp only [toU, hids] at hj ⊢
            exact hbr _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => hbw _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hidpin _ hj
        | ⟨1, _⟩ => fun j hj => by
            simp only [toU, hids] at hj ⊢
            exact hxpin _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show f pid
        (fun k => ids ((TileShape.allIndices io.shapeIdx).get
          (tilePos io.shapeIdx k)))
        (fun k => xs ((TileShape.allIndices io.shape).get (tilePos io.shape k)))
        ((TileShape.allIndices io.shape).get j)
      = f pid ids xs ((TileShape.allIndices io.shape).get j)
    rw [hids, hxs]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

/-- `io.ImplementsR R outDType f` — the **rounding-correctness** relation
`io ⊨[R, outDType] f` for the row-gather family. Verbatim `Implements` with two
changes: the kernel runs under `execR R`, and each active output cell is read
back at `outDType` and must hold `R.round outDType (f pid ids xs j)`.

The index channel is untouched by rounding — it is a `.nat` channel, and
`ChanTy.nat` reads are exact — so only the float output boundary moves. -/
def ImplementsR (io : GatherTileKernelIO) (R : RoundingModel)
    (outDType : FloatDType)
    (f : Nat → (TileIndex io.shapeIdx → Nat) → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.idxbuf, io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid : Nat,
    (∀ i : TileIndex io.shapeIdx, io.maskx pid i →
      io.readx pid i < A.extent io.idxbuf) →
  ∀ ids : TileIndex io.shapeIdx → Nat,
    (∀ j : TileIndex io.shape, io.readMask pid ids j →
      io.read pid ids j < A.extent io.inp) →
    (∀ j : TileIndex io.shape, io.writeMask pid j →
      io.write pid j < A.extent io.out) →
  ∀ (xs : TileIndex io.shape → ℝ) (s₀ : BlockState),
    s₀.pid = pid →
    s₀.undef = (fun _ _ => 0) →
    (∀ i : TileIndex io.shapeIdx, io.maskx pid i →
      s₀.readMemValue .nat io.idxbuf (io.readx pid i) = ids i) →
    (∀ j : TileIndex io.shape, io.readMask pid ids j →
      s₀.readMem io.inp (io.read pid ids j) = xs j) →
    ∃ s',
      execR R (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (∀ j : TileIndex io.shape, io.writeMask pid j →
          s'.readMemAs outDType A.flat (A.addr io.out (io.write pid j))
            = outDType.ofReal (R.round outDType (f pid ids xs j)))
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            ∀ j : TileIndex io.shape, io.writeMask pid j →
              o' ≠ A.addr io.out (io.write pid j)) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped notation:25 io " ⊨[" R ", " outDType "] " f =>
  GatherTileKernelIO.ImplementsR io R outDType f

/-- Assembly lemma for `⊨[R, outDType]` — the rounding sibling of
`Implements.intro`, riding the same `toU` through the family's rounding core
`UKernelIO.ImplementsR.intro` at the constant output grid `fun _ => outDType`. -/
theorem ImplementsR.intro (io : GatherTileKernelIO) {R : RoundingModel}
    {outDType : FloatDType}
    {f : Nat → (TileIndex io.shapeIdx → Nat) → (TileIndex io.shape → ℝ) →
      TileIndex io.shape → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (ids : TileIndex io.shapeIdx → Nat),
      (∀ i : TileIndex io.shapeIdx, io.maskx s.pid i →
        s.readMemValue .nat io.idxbuf (io.readx s.pid i) = ids i) →
      (∀ i : TileIndex io.shapeIdx, io.maskx s.pid i →
        io.readx s.pid i < bounds io.idxbuf) →
      (∀ j : TileIndex io.shape, io.readMask s.pid ids j →
        io.read s.pid ids j < bounds io.inp) →
      (∀ j : TileIndex io.shape, io.writeMask s.pid j →
        io.write s.pid j < bounds io.out) →
      Kernel.TraceSafeR R bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (ids : TileIndex io.shapeIdx → Nat)
        (xs : TileIndex io.shape → ℝ),
      (∀ i : TileIndex io.shapeIdx, io.maskx s₀.pid i →
        s₀.readMemValue .nat io.idxbuf (io.readx s₀.pid i) = ids i) →
      (∀ j : TileIndex io.shape, io.readMask s₀.pid ids j →
        s₀.readMem io.inp (io.read s₀.pid ids j) = xs j) →
      ∃ s1, execR R (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (∀ j : TileIndex io.shape, io.writeMask s₀.pid j →
            s1.readMemAs outDType io.out (io.write s₀.pid j)
              = outDType.ofReal (R.round outDType (f s₀.pid ids xs j)))
        ∧ (∀ r o',
            (r ≠ io.out ∨
              ∀ j : TileIndex io.shape, io.writeMask s₀.pid j →
                o' ≠ io.write s₀.pid j) →
            s1.mem r o' = s₀.mem r o')) :
    io.ImplementsR R outDType f := by
  have hcore : io.toU.ImplementsR R (fun _ => outDType)
      (fun p₀ _p₁ vals _o j =>
        f p₀ (fun k => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx k))
          (fun k => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape k))
          ((TileShape.allIndices io.shape).get j)) := by
    refine UKernelIO.ImplementsR.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib hob _hsb
      have hid : ∀ i : TileIndex io.shapeIdx, io.maskx s.pid i →
          s.readMemValue .nat io.idxbuf (io.readx s.pid i)
            = vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 2) j
      refine hts bounds s _ hid ?_ ?_ ?_
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨0, by decide⟩ : Fin 2) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hib (⟨1, by decide⟩ : Fin 2) j
      · exact (forall_tileIndex_iff _).mp fun j =>
          hob (⟨0, by decide⟩ : Fin 1) j
    · intro s₀ vals _hundef hpins
      have hid : ∀ i : TileIndex io.shapeIdx, io.maskx s₀.pid i →
          s₀.readMemValue .nat io.idxbuf (io.readx s₀.pid i)
            = vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx i) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨0, by decide⟩ : Fin 2) j
      have hx : ∀ j : TileIndex io.shape,
          io.readMask s₀.pid
            (fun k => vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx k)) j →
          s₀.readMem io.inp
              (io.read s₀.pid
                (fun k =>
                  vals (⟨0, by decide⟩ : Fin 2) (tilePos io.shapeIdx k)) j)
            = vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape j) := by
        refine (forall_tileIndex_iff _).mp ?_
        intro j
        rw [tilePos_get]
        exact hpins (⟨1, by decide⟩ : Fin 2) j
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ _ (fun k => vals (⟨1, by decide⟩ : Fin 2) (tilePos io.shape k))
          hid hx
      refine ⟨s1, hexec, fun _o j hj => hval _ hj, ?_⟩
      intro r o' hoc _hsc
      have hoc' : ∀ j : TileIndex io.shape, io.writeMask s₀.pid j →
          r ≠ io.out ∨ o' ≠ io.write s₀.pid j :=
        (forall_tileIndex_iff _).mp fun j => hoc (⟨0, by decide⟩ : Fin 1) j
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j hj => ?_
        rcases hoc' j hj with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid hbx ids hbr hbw xs s₀ hpid hu hidpin hxpin
  have hids :
      (fun k => ids ((TileShape.allIndices io.shapeIdx).get
        (tilePos io.shapeIdx k))) = ids :=
    funext fun k => by rw [get_tilePos]
  have hxs :
      (fun k => xs ((TileShape.allIndices io.shape).get (tilePos io.shape k)))
        = xs :=
    funext fun k => by rw [get_tilePos]
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid (s₀.pids 1) (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun j => ids ((TileShape.allIndices io.shapeIdx).get j)
        | ⟨1, _⟩ => fun j => xs ((TileShape.allIndices io.shape).get j)
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      s₀ hpid rfl rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hbx _ hj
        | ⟨1, _⟩ => fun j hj => by
            simp only [toU, hids] at hj ⊢
            exact hbr _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
      (fun _o j hj => hbw _ hj) (fun t => t.elim0)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hidpin _ hj
        | ⟨1, _⟩ => fun j hj => by
            simp only [toU, hids] at hj ⊢
            exact hxpin _ hj
        | ⟨_ + 2, h⟩ => absurd h (by simp only [toU]; omega))
  refine ⟨s', hexec, ?_, ?_⟩
  · refine (forall_tileIndex_iff _).mp ?_
    intro j hj
    refine (hval (⟨0, by decide⟩ : Fin 1) j hj).trans ?_
    show outDType.ofReal (R.round outDType (f pid
        (fun k => ids ((TileShape.allIndices io.shapeIdx).get
          (tilePos io.shapeIdx k)))
        (fun k => xs ((TileShape.allIndices io.shape).get (tilePos io.shape k)))
        ((TileShape.allIndices io.shape).get j)))
      = outDType.ofReal (R.round outDType
          (f pid ids xs ((TileShape.allIndices io.shape).get j)))
    rw [hids, hxs]
  · intro r' o' hcond
    refine hframe r' o' ?_
    rcases hcond with hflat | hout
    · exact Or.inl hflat
    · exact Or.inr ⟨fun _o j hj => hout _ hj, fun t => t.elim0⟩

end GatherTileKernelIO

end VeriTile.Triton
