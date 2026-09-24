/-
Kernel IO contracts: Scatter.

See `VeriTile.Triton.Memory.KernelSpec` for the shared contract and modeling
boundary. This module preserves the public `VeriTile.Triton` namespaces.
-/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton


/-! ### The scatter genre: `Scatter*` skins

Data-dependent **write** addresses. A `Scatter*` struct declares an
**index channel** — a `.nat` tile (a named field, never part of the
arity subscript) whose pinned per-lane values `ids` enter the write
window and the write gate. Scatter readback is only meaningful when no
two write-active lanes collide, so the readback leg inside `Implements`
is guarded by a per-context `WriteInj` antecedent, while the frame and
all bounds stay unconditional: the kernel writes the raw scatter cells
whether or not they collide, so in the core embedding those cells are
shadowed by a scratch channel (same buffer, same window, ungated mask),
which carries both the unconditional frame exclusion and the
unconditional trace-safety write bound. -/

/-- IO signature of a **compaction-scatter** masked one-input /
one-output kernel with a `.bool` gate tile: one float data tile (`inp`),
one bool select tile (`mbuf`), and one `.nat` **index tile** (`idxbuf`)
whose loaded values `ids` give each lane's scatter destination
`write … ids j`; the store is gated by `writeMask … bs ids` (typically
`mask ∧ bs j = true`). Intended consumer: the TritonBench-G compaction
scatter `masked_select` (data tile + select-mask tile + prefix-sum tile
whose values give the destinations `prefix_sum[j] − 1`).

**Injectivity design.** The readback leg of `Implements` is guarded by
the *per-pinned-context* antecedent `WriteInj pid₀ pid₁ bs ids` (no two
write-active lanes share a destination) — it is **not** a hypothesis of
`Implements.intro`: a `∀`-quantified intro hypothesis would demand
injectivity for arbitrary index-buffer contents (false for, e.g.,
duplicated indices), whereas the per-context antecedent is exactly the
host-side no-duplicate-destination guarantee (the consumers' `hOutInj` /
`hUniq` side conditions), which `hrun` receives as the antecedent of its
own value leg and threads to its scatter-readback lemma. The frame and
the trace-safety write bound are stated at the *ungated* `writeMask`
lanes, so they hold with or without injectivity. -/
structure BoolScatterMasked2DKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- Input buffer (ℝ channel). -/
  inp : RegionName
  /-- Boolean input buffer (`.bool` channel — the select gate). -/
  mbuf : RegionName
  /-- Index buffer (`.nat` channel — the scatter destinations). -/
  idxbuf : RegionName
  /-- Output buffer. -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- Lane `j`'s `inp` read address for program `(pid₀, pid₁)`. -/
  read : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `mbuf` read address (the bool tile's window). -/
  readm : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `idxbuf` read address (the index tile's window). -/
  readx : Nat → Nat → Fin B → Nat
  /-- Lane `j`'s write address given the loaded index tile — the
  data-dependent scatter destination. -/
  write : Nat → Nat → (Fin B → Nat) → Fin B → Nat
  /-- Program `(pid₀, pid₁)`'s **read-active** lanes — the bounds /
  trace-safety superset. -/
  mask : Nat → Nat → Fin B → Prop
  /-- **Write-active** lanes given the loaded bool and index tiles;
  defaults to the static `mask`. -/
  writeMask : Nat → Nat → (Fin B → Bool) → (Fin B → Nat) → Fin B → Prop :=
    fun p₀ p₁ _ _ j => mask p₀ p₁ j

namespace BoolScatterMasked2DKernelIO₁

/-- No two write-active lanes share a scatter destination — the
per-pinned-context injectivity that scatter readback needs. For
`masked_select` this is the host prefix-sum's no-duplicate-destination
guarantee. -/
def WriteInj (io : BoolScatterMasked2DKernelIO₁) (p₀ p₁ : Nat)
    (bs : Fin io.B → Bool) (ids : Fin io.B → Nat) : Prop :=
  ∀ j k : Fin io.B, io.writeMask p₀ p₁ bs ids j →
    io.writeMask p₀ p₁ bs ids k →
    io.write p₀ p₁ ids j = io.write p₀ p₁ ids k → j = k

/-- `io.Implements f` — the compaction-scatter masked Hoare triple. The
bool tile `bs` and the index tile `ids` are quantified alongside the data
tile and pinned on the read-active lanes; the write addresses eat the
*loaded* `ids`. The readback leg is guarded by `WriteInj` (see the
struct docstring); the frame excludes the raw (ungated) scatter cells. -/
def Implements (io : BoolScatterMasked2DKernelIO₁)
    (f : Nat → Nat → (Fin io.B → Bool) → (Fin io.B → Nat) →
      (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.inp, io.mbuf, io.idxbuf, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (bs : Fin io.B → Bool) (ids : Fin io.B → Nat) (xs : Fin io.B → ℝ)
    (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.read pid₀ pid₁ j < A.extent io.inp) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.readm pid₀ pid₁ j < A.extent io.mbuf) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      io.readx pid₀ pid₁ j < A.extent io.idxbuf) →
    (∀ j : Fin io.B, io.writeMask pid₀ pid₁ bs ids j →
      io.write pid₀ pid₁ ids j < A.extent io.out) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMem io.inp (io.read pid₀ pid₁ j) = xs j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMemValue .bool io.mbuf (io.readm pid₀ pid₁ j) = bs j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ j →
      s₀.readMemValue .nat io.idxbuf (io.readx pid₀ pid₁ j) = ids j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.WriteInj pid₀ pid₁ bs ids →
          ∀ j : Fin io.B, io.writeMask pid₀ pid₁ bs ids j →
            s'.readMem A.flat (A.addr io.out (io.write pid₀ pid₁ ids j))
              = f pid₀ pid₁ bs ids xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.B, io.writeMask pid₀ pid₁ bs ids j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ ids j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => BoolScatterMasked2DKernelIO₁.Implements

/-- Embed into the unified core: channel 0 is the float data tile,
channel 1 the `.bool` select tile, channel 2 the `.nat` index tile.
The single output's window eats the index channel's pinned values and
its mask is the write gate **conjoined with `WriteInj`**; a scratch
channel shadows the same cells with the *ungated* write gate, carrying
the unconditional frame exclusion and write bound. -/
private def toU (io : BoolScatterMasked2DKernelIO₁) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 3
  nOut := 1
  nScr := 1
  bufs := [io.inp, io.mbuf, io.idxbuf, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .bool
    | ⟨2, _⟩ => .nat
  iarity := fun _ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.inp
    | ⟨1, _⟩ => io.mbuf
    | ⟨2, _⟩ => io.idxbuf
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun _ => io.out
  iwin := fun i _ p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun j => io.read p₀ p₁ j
    | ⟨1, _⟩ => fun j => io.readm p₀ p₁ j
    | ⟨2, _⟩ => fun j => io.readx p₀ p₁ j
  imask := fun _ _ p₀ p₁ _ j => io.mask p₀ p₁ j
  owin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (fun j' => vals (⟨2, by decide⟩ : Fin 3) j') j
  omask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
      (fun j' => vals (⟨2, by decide⟩ : Fin 3) j') j
    ∧ io.WriteInj p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
        (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')
  swin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (fun j' => vals (⟨2, by decide⟩ : Fin 3) j') j
  smask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
      (fun j' => vals (⟨2, by decide⟩ : Fin 3) j') j

/-- Assembly lemma: obligations in the skin's named vocabulary — the
bool/index tiles enter `hts`/`hrun` as pinned tiles, and `hrun`'s value
leg receives the per-context `WriteInj` antecedent (the consumer threads
its `hOutInj`/`hUniq` side condition there); `hrun`'s frame and the
write bound are at the ungated `writeMask` lanes. -/
theorem Implements.intro (io : BoolScatterMasked2DKernelIO₁)
    {f : Nat → Nat → (Fin io.B → Bool) → (Fin io.B → Nat) →
      (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState)
        (bs : Fin io.B → Bool) (ids : Fin io.B → Nat),
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        s.readMemValue .bool io.mbuf (io.readm (s.pids 0) (s.pids 1) j)
          = bs j) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        s.readMemValue .nat io.idxbuf (io.readx (s.pids 0) (s.pids 1) j)
          = ids j) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.read (s.pids 0) (s.pids 1) j < bounds io.inp) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.readm (s.pids 0) (s.pids 1) j < bounds io.mbuf) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) j →
        io.readx (s.pids 0) (s.pids 1) j < bounds io.idxbuf) →
      (∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) bs ids j →
        io.write (s.pids 0) (s.pids 1) ids j < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (bs : Fin io.B → Bool)
        (ids : Fin io.B → Nat) (xs : Fin io.B → ℝ),
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMem io.inp (io.read (s₀.pids 0) (s₀.pids 1) j) = xs j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMemValue .bool io.mbuf (io.readm (s₀.pids 0) (s₀.pids 1) j)
          = bs j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) j →
        s₀.readMemValue .nat io.idxbuf (io.readx (s₀.pids 0) (s₀.pids 1) j)
          = ids j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (io.WriteInj (s₀.pids 0) (s₀.pids 1) bs ids →
            ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) bs ids j →
              s1.readMem io.out (io.write (s₀.pids 0) (s₀.pids 1) ids j)
                = f (s₀.pids 0) (s₀.pids 1) bs ids xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B, io.writeMask (s₀.pids 0) (s₀.pids 1) bs ids j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) ids j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁ (fun j' => vals (⟨1, by decide⟩ : Fin 3) j')
          (fun j' => vals (⟨2, by decide⟩ : Fin 3) j')
          (fun j' => vals (⟨0, by decide⟩ : Fin 3) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib _hob hsb
      exact hts bounds s
        (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
        (fun j => vals (⟨2, by decide⟩ : Fin 3) j)
        (fun j hj => hpins (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hpins (⟨2, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨0, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨1, by decide⟩ : Fin 3) j hj)
        (fun j hj => hib (⟨2, by decide⟩ : Fin 3) j hj)
        (fun j hj => hsb (⟨0, by decide⟩ : Fin 1) j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ :=
        hrun s₀ (fun j => vals (⟨1, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨2, by decide⟩ : Fin 3) j)
          (fun j => vals (⟨0, by decide⟩ : Fin 3) j)
          (fun j hj => hpins (⟨0, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨1, by decide⟩ : Fin 3) j hj)
          (fun j hj => hpins (⟨2, by decide⟩ : Fin 3) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval hj.2 j hj.1, ?_⟩
      intro r o' _hoc hsc
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j hj => ?_
        rcases hsc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ bs ids xs s₀ hpid₀ hpid₁ hu hbr hbm hbx
    hbw hx hb hi
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => xs
        | ⟨1, _⟩ => bs
        | ⟨2, _⟩ => ids)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hbr j hj
        | ⟨1, _⟩ => fun j hj => hbm j hj
        | ⟨2, _⟩ => fun j hj => hbx j hj)
      (fun _o j hj => hbw j hj.1)
      (fun _t j hj => hbw j hj)
      (fun i => match i with
        | ⟨0, _⟩ => fun j hj => hx j hj
        | ⟨1, _⟩ => fun j hj => hb j hj
        | ⟨2, _⟩ => fun j hj => hi j hj)
  refine ⟨s', hexec,
    fun hinj j hj => hval (⟨0, by decide⟩ : Fin 1) j ⟨hj, hinj⟩, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hn j hj.1, fun _t j hj => hn j hj⟩

end BoolScatterMasked2DKernelIO₁

/-- IO signature of the **penalty gather–scatter shape** (`Meta` slots +
`Scatter` writes): three per-program float scalar slots (`fbuf1`–`fbuf3`,
cells `fwin1`–`fwin3`), two per-program `.nat` scalar slots on **one**
metadata buffer (`mbuf`, cells `mwin1`/`mwin2` — the cumsum window
bounds), a `.nat` **index tile** (`idbuf`, window `readi`, values `ids` —
the gathered token ids), a `.nat` payload tile (`cntbuf`, window `readc`,
values `cnts` — the token counts), and one float data channel (`inp`)
**gather-read** at the index-dependent window `read … ids` and
**scatter-written** at `write … ids` (the in-place consumer instantiates
`out := inp` — duplicate-region wiring is supported by `bufs`). The
subscript is pure float-data arity (one gathered input ↦ one scattered
output); slots and `.nat` tiles are capability fields. Intended consumer:
the TritonBench-G LightLLM penalty kernel `apply_penalty`
(per-`cur_batch` presence/frequency/repetition penalties applied in place
to `Logits` at the gathered token positions).

**Injectivity design.** Same as `BoolScatterMasked2DKernelIO₁`: the
readback leg of `Implements` is guarded by the per-pinned-context
`WriteInj pid₀ pid₁ m₁ m₂ ids` antecedent (the consumer's `hUniq`
distinct-active-token-ids side condition) rather than by an
`Implements.intro` hypothesis, and the frame / trace-safety write bound
stay at the ungated `writeMask` lanes via the core scratch shadow. -/
structure MetaScatterMasked2DKernelIO₁ where
  /-- The kernel being specified. -/
  kernel : ComputeKernel
  /-- The contract executes only a successfully projected kernel. The default
  discharges transparent supported kernels; abstract kernels need a witness. -/
  projection : kernel.toAlgorithm? = Except.ok kernel.toAlgKernel := by
    first | rfl | simp [ComputeKernel.toAlgKernel]
  /-- First float scalar slot's buffer (one ℝ cell per program). -/
  fbuf1 : RegionName
  /-- Second float scalar slot's buffer. -/
  fbuf2 : RegionName
  /-- Third float scalar slot's buffer. -/
  fbuf3 : RegionName
  /-- The `.nat` metadata buffer carrying both scalar slots (the
  consumer's cumulative-sequence-length array). -/
  mbuf : RegionName
  /-- Index buffer (`.nat` tile — the gather/scatter token ids). -/
  idbuf : RegionName
  /-- Payload buffer (`.nat` tile — the per-token counts). -/
  cntbuf : RegionName
  /-- Float data buffer, gather-read at the index-dependent window. -/
  inp : RegionName
  /-- Output buffer, scatter-written (the in-place consumer sets
  `out := inp`). -/
  out : RegionName
  /-- Tile length: each program instance owns `B`-lane windows. -/
  B : Nat
  /-- First float slot's cell address for program `(pid₀, pid₁)`. -/
  fwin1 : Nat → Nat → Nat
  /-- Second float slot's cell address. -/
  fwin2 : Nat → Nat → Nat
  /-- Third float slot's cell address. -/
  fwin3 : Nat → Nat → Nat
  /-- First `.nat` slot's cell address in `mbuf`. -/
  mwin1 : Nat → Nat → Nat
  /-- Second `.nat` slot's cell address in `mbuf`. -/
  mwin2 : Nat → Nat → Nat
  /-- Lane `j`'s `idbuf` read address at `(pid₀, pid₁, m₁, m₂)` — the
  loaded `.nat` scalars are ordinary named arguments. -/
  readi : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `cntbuf` read address. -/
  readc : Nat → Nat → Nat → Nat → Fin B → Nat
  /-- Lane `j`'s `inp` **gather** read address, given the loaded index
  tile. -/
  read : Nat → Nat → Nat → Nat → (Fin B → Nat) → Fin B → Nat
  /-- Read-active lanes, given the loaded `.nat` scalars (the consumer's
  `m₁ + j < m₂` batch window). -/
  mask : Nat → Nat → Nat → Nat → Fin B → Prop
  /-- Lane `j`'s **scatter** write address, given the loaded index tile. -/
  write : Nat → Nat → Nat → Nat → (Fin B → Nat) → Fin B → Nat
  /-- Write-active lanes given the loaded index tile; defaults to `mask`. -/
  writeMask : Nat → Nat → Nat → Nat → (Fin B → Nat) → Fin B → Prop :=
    fun p₀ p₁ m₁ m₂ _ j => mask p₀ p₁ m₁ m₂ j

namespace MetaScatterMasked2DKernelIO₁

/-- No two write-active lanes share a scatter destination — the
per-pinned-context injectivity that scatter readback needs. For
`apply_penalty` this is the distinct-active-token-ids side condition
(`hUniq`). -/
def WriteInj (io : MetaScatterMasked2DKernelIO₁) (p₀ p₁ m₁ m₂ : Nat)
    (ids : Fin io.B → Nat) : Prop :=
  ∀ j k : Fin io.B, io.writeMask p₀ p₁ m₁ m₂ ids j →
    io.writeMask p₀ p₁ m₁ m₂ ids k →
    io.write p₀ p₁ m₁ m₂ ids j = io.write p₀ p₁ m₁ m₂ ids k → j = k

/-- `io.Implements f` — the penalty gather–scatter masked Hoare triple.
The float slots `g₁ g₂ g₃`, the `.nat` slots `m₁ m₂`, the index tile
`ids`, the payload tile `cnts`, and the gathered data tile `xs` are all
universally quantified and pinned inside the memory precondition, so the
window/mask/value contracts speak about the *loaded* values throughout.
The readback leg is guarded by `WriteInj` (see the struct docstring);
the frame excludes the raw (ungated) scatter cells. -/
def Implements (io : MetaScatterMasked2DKernelIO₁)
    (f : Nat → Nat → ℝ → ℝ → ℝ → Nat → Nat → (Fin io.B → Nat) →
      (Fin io.B → Nat) → (Fin io.B → ℝ) → Fin io.B → ℝ) : Prop :=
  ∀ A : FlatAlloc,
    A.Disjoint →
    A.regions = [io.fbuf1, io.fbuf2, io.fbuf3, io.mbuf, io.idbuf,
      io.cntbuf, io.inp, io.out] →
    (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ pid₀ pid₁ : Nat,
  ∀ (g₁ g₂ g₃ : ℝ) (m₁ m₂ : Nat) (ids cnts : Fin io.B → Nat)
    (xs : Fin io.B → ℝ) (s₀ : BlockState),
    s₀.pids 0 = pid₀ →
    s₀.pids 1 = pid₁ →
    s₀.undef = (fun _ _ => 0) →
    io.fwin1 pid₀ pid₁ < A.extent io.fbuf1 →
    io.fwin2 pid₀ pid₁ < A.extent io.fbuf2 →
    io.fwin3 pid₀ pid₁ < A.extent io.fbuf3 →
    io.mwin1 pid₀ pid₁ < A.extent io.mbuf →
    io.mwin2 pid₀ pid₁ < A.extent io.mbuf →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      io.readi pid₀ pid₁ m₁ m₂ j < A.extent io.idbuf) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      io.readc pid₀ pid₁ m₁ m₂ j < A.extent io.cntbuf) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      io.read pid₀ pid₁ m₁ m₂ ids j < A.extent io.inp) →
    (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ ids j →
      io.write pid₀ pid₁ m₁ m₂ ids j < A.extent io.out) →
    s₀.readMem io.fbuf1 (io.fwin1 pid₀ pid₁) = g₁ →
    s₀.readMem io.fbuf2 (io.fwin2 pid₀ pid₁) = g₂ →
    s₀.readMem io.fbuf3 (io.fwin3 pid₀ pid₁) = g₃ →
    s₀.readMemValue .nat io.mbuf (io.mwin1 pid₀ pid₁) = m₁ →
    s₀.readMemValue .nat io.mbuf (io.mwin2 pid₀ pid₁) = m₂ →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      s₀.readMemValue .nat io.idbuf (io.readi pid₀ pid₁ m₁ m₂ j) = ids j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      s₀.readMemValue .nat io.cntbuf (io.readc pid₀ pid₁ m₁ m₂ j)
        = cnts j) →
    (∀ j : Fin io.B, io.mask pid₀ pid₁ m₁ m₂ j →
      s₀.readMem io.inp (io.read pid₀ pid₁ m₁ m₂ ids j) = xs j) →
    ∃ s',
      exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s₀)
        = some s'
      ∧ (io.WriteInj pid₀ pid₁ m₁ m₂ ids →
          ∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ ids j →
            s'.readMem A.flat
                (A.addr io.out (io.write pid₀ pid₁ m₁ m₂ ids j))
              = f pid₀ pid₁ g₁ g₂ g₃ m₁ m₂ ids cnts xs j)
      ∧ (∀ r' o',
          (r' ≠ A.flat ∨
            (∀ j : Fin io.B, io.writeMask pid₀ pid₁ m₁ m₂ ids j →
              o' ≠ A.addr io.out (io.write pid₀ pid₁ m₁ m₂ ids j))) →
          s'.mem r' o' = (A.flattenState s₀).mem r' o')

@[inherit_doc] scoped infix:25 " ⊨ " => MetaScatterMasked2DKernelIO₁.Implements

/-- Embed into the unified core: channels 0–2 are the 1-lane float
slots, channels 3–4 the 1-lane `.nat` slots (both on `mbuf`), channel 5
the `.nat` index tile and channel 6 the `.nat` payload tile (windows
reading the slots' pinned values), channel 7 the float gather tile
(window reading slots *and* the index channel). The output window eats
the index channel and its mask is the write gate conjoined with
`WriteInj`; a scratch channel shadows the same cells ungated. -/
private def toU (io : MetaScatterMasked2DKernelIO₁) : UKernelIO where
  kernel := io.kernel
  projection := io.projection
  nIn := 8
  nOut := 1
  nScr := 1
  bufs := [io.fbuf1, io.fbuf2, io.fbuf3, io.mbuf, io.idbuf, io.cntbuf,
    io.inp, io.out]
  ity := fun i => match i with
    | ⟨0, _⟩ => .float
    | ⟨1, _⟩ => .float
    | ⟨2, _⟩ => .float
    | ⟨3, _⟩ => .nat
    | ⟨4, _⟩ => .nat
    | ⟨5, _⟩ => .nat
    | ⟨6, _⟩ => .nat
    | ⟨7, _⟩ => .float
  iarity := fun i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨2, _⟩ => 1
    | ⟨3, _⟩ => 1
    | ⟨4, _⟩ => 1
    | ⟨5, _⟩ => io.B
    | ⟨6, _⟩ => io.B
    | ⟨7, _⟩ => io.B
  ibuf := fun i => match i with
    | ⟨0, _⟩ => io.fbuf1
    | ⟨1, _⟩ => io.fbuf2
    | ⟨2, _⟩ => io.fbuf3
    | ⟨3, _⟩ => io.mbuf
    | ⟨4, _⟩ => io.mbuf
    | ⟨5, _⟩ => io.idbuf
    | ⟨6, _⟩ => io.cntbuf
    | ⟨7, _⟩ => io.inp
  oty := fun _ => .float
  oarity := fun _ => io.B
  obuf := fun _ => io.out
  obuf_mem := fun _ => by simp
  sarity := fun _ => io.B
  sbuf := fun _ => io.out
  iwin := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => io.fwin1 p₀ p₁
    | ⟨1, _⟩ => fun _ => io.fwin2 p₀ p₁
    | ⟨2, _⟩ => fun _ => io.fwin3 p₀ p₁
    | ⟨3, _⟩ => fun _ => io.mwin1 p₀ p₁
    | ⟨4, _⟩ => fun _ => io.mwin2 p₀ p₁
    | ⟨5, _⟩ => fun j => io.readi p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨6, _⟩ => fun j => io.readc p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨7, _⟩ => fun j => io.read p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (fun j' => vals (⟨5, by decide⟩ : Fin 8) j') j
  imask := fun i vals p₀ p₁ _ => match i with
    | ⟨0, _⟩ => fun _ => True
    | ⟨1, _⟩ => fun _ => True
    | ⟨2, _⟩ => fun _ => True
    | ⟨3, _⟩ => fun _ => True
    | ⟨4, _⟩ => fun _ => True
    | ⟨5, _⟩ => fun j => io.mask p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨6, _⟩ => fun j => io.mask p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1)) j
    | ⟨7, _⟩ => fun j => io.mask p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1)) j
  owin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
    (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
    (fun j' => vals (⟨5, by decide⟩ : Fin 8) j') j
  omask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁
      (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
      (fun j' => vals (⟨5, by decide⟩ : Fin 8) j') j
    ∧ io.WriteInj p₀ p₁
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (fun j' => vals (⟨5, by decide⟩ : Fin 8) j')
  swin := fun _ vals p₀ p₁ _ j => io.write p₀ p₁
    (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
    (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
    (fun j' => vals (⟨5, by decide⟩ : Fin 8) j') j
  smask := fun _ vals p₀ p₁ _ j =>
    io.writeMask p₀ p₁
      (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
      (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
      (fun j' => vals (⟨5, by decide⟩ : Fin 8) j') j

/-- Assembly lemma: obligations in the skin's named vocabulary — all
scalar slots enter `hts`/`hrun` as pinned named values (no
lane-constancy plumbing; slots are 1-lane channels), and `hrun`'s value
leg receives the per-context `WriteInj` antecedent (the consumer threads
its `hUniq` side condition there); `hrun`'s frame and the write bound
are at the ungated `writeMask` lanes. -/
theorem Implements.intro (io : MetaScatterMasked2DKernelIO₁)
    {f : Nat → Nat → ℝ → ℝ → ℝ → Nat → Nat → (Fin io.B → Nat) →
      (Fin io.B → Nat) → (Fin io.B → ℝ) → Fin io.B → ℝ}
    (hok : (io.kernel.toAlgKernel).FlattenOk)
    (hts : ∀ (bounds : RegionBounds) (s : BlockState) (m₁ m₂ : Nat)
        (ids : Fin io.B → Nat),
      s.readMemValue .nat io.mbuf (io.mwin1 (s.pids 0) (s.pids 1)) = m₁ →
      s.readMemValue .nat io.mbuf (io.mwin2 (s.pids 0) (s.pids 1)) = m₂ →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        s.readMemValue .nat io.idbuf
          (io.readi (s.pids 0) (s.pids 1) m₁ m₂ j) = ids j) →
      io.fwin1 (s.pids 0) (s.pids 1) < bounds io.fbuf1 →
      io.fwin2 (s.pids 0) (s.pids 1) < bounds io.fbuf2 →
      io.fwin3 (s.pids 0) (s.pids 1) < bounds io.fbuf3 →
      io.mwin1 (s.pids 0) (s.pids 1) < bounds io.mbuf →
      io.mwin2 (s.pids 0) (s.pids 1) < bounds io.mbuf →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.readi (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.idbuf) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.readc (s.pids 0) (s.pids 1) m₁ m₂ j < bounds io.cntbuf) →
      (∀ j : Fin io.B, io.mask (s.pids 0) (s.pids 1) m₁ m₂ j →
        io.read (s.pids 0) (s.pids 1) m₁ m₂ ids j < bounds io.inp) →
      (∀ j : Fin io.B, io.writeMask (s.pids 0) (s.pids 1) m₁ m₂ ids j →
        io.write (s.pids 0) (s.pids 1) m₁ m₂ ids j < bounds io.out) →
      Kernel.TraceSafe bounds (io.kernel.toAlgKernel) s)
    (hrun : ∀ (s₀ : BlockState) (g₁ g₂ g₃ : ℝ) (m₁ m₂ : Nat)
        (ids cnts : Fin io.B → Nat) (xs : Fin io.B → ℝ),
      s₀.readMem io.fbuf1 (io.fwin1 (s₀.pids 0) (s₀.pids 1)) = g₁ →
      s₀.readMem io.fbuf2 (io.fwin2 (s₀.pids 0) (s₀.pids 1)) = g₂ →
      s₀.readMem io.fbuf3 (io.fwin3 (s₀.pids 0) (s₀.pids 1)) = g₃ →
      s₀.readMemValue .nat io.mbuf (io.mwin1 (s₀.pids 0) (s₀.pids 1)) = m₁ →
      s₀.readMemValue .nat io.mbuf (io.mwin2 (s₀.pids 0) (s₀.pids 1)) = m₂ →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMemValue .nat io.idbuf
          (io.readi (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) = ids j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMemValue .nat io.cntbuf
          (io.readc (s₀.pids 0) (s₀.pids 1) m₁ m₂ j) = cnts j) →
      (∀ j : Fin io.B, io.mask (s₀.pids 0) (s₀.pids 1) m₁ m₂ j →
        s₀.readMem io.inp
          (io.read (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j) = xs j) →
      ∃ s1, exec (io.kernel.toAlgKernel) s₀ = some s1
        ∧ (io.WriteInj (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids →
            ∀ j : Fin io.B,
              io.writeMask (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j →
              s1.readMem io.out
                  (io.write (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j)
                = f (s₀.pids 0) (s₀.pids 1) g₁ g₂ g₃ m₁ m₂ ids cnts xs j)
        ∧ (∀ r o,
            (r ≠ io.out ∨
              ∀ j : Fin io.B,
                io.writeMask (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j →
                o ≠ io.write (s₀.pids 0) (s₀.pids 1) m₁ m₂ ids j) →
            s1.mem r o = s₀.mem r o)) :
    io.Implements f := by
  -- assemble the unified-core triple once, then convert it back into the
  -- family statement; the flattening bridge lives in
  -- `UKernelIO.Implements.intro`
  have hcore : io.toU.Implements
      (fun p₀ p₁ vals _o j =>
        f p₀ p₁
          (vals (⟨0, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨1, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨2, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
          (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
          (fun j' => vals (⟨5, by decide⟩ : Fin 8) j')
          (fun j' => vals (⟨6, by decide⟩ : Fin 8) j')
          (fun j' => vals (⟨7, by decide⟩ : Fin 8) j') j) := by
    refine UKernelIO.Implements.intro _ hok ?_ ?_
    · intro bounds s vals hpins hib _hob hsb
      exact hts bounds s
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (fun j' => vals (⟨5, by decide⟩ : Fin 8) j')
        (hpins (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hpins (⟨5, by decide⟩ : Fin 8) j hj)
        (hib (⟨0, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨1, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨2, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hib (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hib (⟨5, by decide⟩ : Fin 8) j hj)
        (fun j hj => hib (⟨6, by decide⟩ : Fin 8) j hj)
        (fun j hj => hib (⟨7, by decide⟩ : Fin 8) j hj)
        (fun j hj => hsb (⟨0, by decide⟩ : Fin 1) j hj)
    · intro s₀ vals _hundef hpins
      obtain ⟨s1, hexec, hval, hframe⟩ := hrun s₀
        (vals (⟨0, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨1, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨2, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (vals (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1))
        (fun j => vals (⟨5, by decide⟩ : Fin 8) j)
        (fun j => vals (⟨6, by decide⟩ : Fin 8) j)
        (fun j => vals (⟨7, by decide⟩ : Fin 8) j)
        (hpins (⟨0, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨1, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨2, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨3, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (hpins (⟨4, by decide⟩ : Fin 8) (⟨0, by decide⟩ : Fin 1) trivial)
        (fun j hj => hpins (⟨5, by decide⟩ : Fin 8) j hj)
        (fun j hj => hpins (⟨6, by decide⟩ : Fin 8) j hj)
        (fun j hj => hpins (⟨7, by decide⟩ : Fin 8) j hj)
      refine ⟨s1, hexec, fun _o j hj => hval hj.2 j hj.1, ?_⟩
      intro r o' _hoc hsc
      refine hframe r o' ?_
      by_cases hro : r = io.out
      · subst hro
        refine Or.inr fun j hj => ?_
        rcases hsc (⟨0, by decide⟩ : Fin 1) j hj with hne | hno
        · exact absurd rfl hne
        · exact hno
      · exact Or.inl hro
  intro A hd hregs hcov pid₀ pid₁ g₁ g₂ g₃ m₁ m₂ ids cnts xs s₀ hpid₀ hpid₁
    hu hbf1 hbf2 hbf3 hbm1 hbm2 hbi hbc hbr hbw hg1 hg2 hg3 hm1 hm2 hid
    hcnt hx
  obtain ⟨s', hexec, hval, hframe⟩ :=
    hcore A hd hregs hcov pid₀ pid₁ (s₀.pids 2)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ => g₁
        | ⟨1, _⟩ => fun _ => g₂
        | ⟨2, _⟩ => fun _ => g₃
        | ⟨3, _⟩ => fun _ => m₁
        | ⟨4, _⟩ => fun _ => m₂
        | ⟨5, _⟩ => ids
        | ⟨6, _⟩ => cnts
        | ⟨7, _⟩ => xs)
      s₀ hpid₀ hpid₁ rfl hu
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hbf1
        | ⟨1, _⟩ => fun _ _ => hbf2
        | ⟨2, _⟩ => fun _ _ => hbf3
        | ⟨3, _⟩ => fun _ _ => hbm1
        | ⟨4, _⟩ => fun _ _ => hbm2
        | ⟨5, _⟩ => fun j hj => hbi j hj
        | ⟨6, _⟩ => fun j hj => hbc j hj
        | ⟨7, _⟩ => fun j hj => hbr j hj)
      (fun _o j hj => hbw j hj.1)
      (fun _t j hj => hbw j hj)
      (fun i => match i with
        | ⟨0, _⟩ => fun _ _ => hg1
        | ⟨1, _⟩ => fun _ _ => hg2
        | ⟨2, _⟩ => fun _ _ => hg3
        | ⟨3, _⟩ => fun _ _ => hm1
        | ⟨4, _⟩ => fun _ _ => hm2
        | ⟨5, _⟩ => fun j hj => hid j hj
        | ⟨6, _⟩ => fun j hj => hcnt j hj
        | ⟨7, _⟩ => fun j hj => hx j hj)
  refine ⟨s', hexec,
    fun hinj j hj => hval (⟨0, by decide⟩ : Fin 1) j ⟨hj, hinj⟩, ?_⟩
  intro r' o' hcond
  refine hframe r' o' ?_
  rcases hcond with hflat | hn
  · exact Or.inl hflat
  · exact Or.inr ⟨fun _o j hj => hn j hj.1, fun _t j hj => hn j hj⟩

end MetaScatterMasked2DKernelIO₁

end VeriTile.Triton
