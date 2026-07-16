/-
bench/examples/FloatDTypeCorrect

**Float dtypes at the memory boundary — the float-facing *correctness*
policy, on the `⊨[R]` surface.** Pilot consumer of `KernelIO₂.ImplementsR`.

One kernel (`floatAddKernel`), two float-annotation fates — and three fp32
spellings, each with its own meaning:

* **Loads, `dtype=tl.float32` — erased.** `tl.float32` load annotations live
  on VeriTile's bit-accurate compute channel (`ComputeDType.fp32`), and the
  compute→algorithm projection rewrites every fp32 load to the Real channel
  before any semantics sees it (`floatAdd_toAlg`: the projected loads are
  `Op.load .real`) — inputs are consumed as exact ℝ. Documented by the
  honest region-typing negation below. Likewise `.to(tl.float32)` is the
  **working-precision** cast: it routes to the same compute channel and is
  inert on the projected ℝ surface (the abstraction models fp32 working
  precision as ℝ — the ports' proofs rely on it).
* **Store, `.round_to(tl.float32)` — modeled.** `round_to` is the explicit
  **quantization-event** spelling: it always emits a `FloatDType`
  rounding-channel cast, which the projection preserves — the projected
  store is fp32-typed and its value carries `Op.castFloat .real .fp32`.
  Under `execR R` the value is rounded at the cast (site 1) and quantized
  again at the typed store (site 2), and `R.round_idem` collapses the pair
  to one boundary round.

The headline is the **boundary-rounding contract**: for every rounding model
`R`, the output window holds the fp32-typed cells
`R.round .fp32 (xs i + ys i)`. At `R := .triv` the round is the identity and
the statement is exact — the file's previous erasure headline (`⊨` with
`(floatAddKernel …).eraseDType` in the IO signature) is retired, and
`.eraseDType` no longer appears anywhere in the audited statement. The
refinement counterpart of the erasure policy stays in
`bench/examples/FloatDTypeEquiv.lean`.

Four parts:

1. **The kernel** — `floatAddKernel`, elementwise add with fp32-annotated
   loads and an fp32-quantized (`round_to`) store.
2. **Literal projected body + region-model Hoare triple** —
   `floatAdd_toAlg` pins the kernel's algorithm projection to its literal
   six-statement list (load annotations erased, store cast preserved), then
   the rounded region run `floatAdd_region_runR` (termination under
   `execR R`, typed fp32 output readback, frame). Also here, the honest
   region-typing facts: the all-fp32 typing is a proved **negation** (the
   load annotations die in projection), while the mixed typing — inputs
   `.real`, output `.fp32` — is the true contract (the store dtype
   survives).
3. **Flat-memory bridge side conditions** — `Kernel.TraceSafeR` and
   `FlattenOk`, discharged on the literal body.
4. **The spec** — the file's single `specification`:

       float_add_correctness : floatAddIO B ⊨[R] fun xs ys i => xs i + ys i

   `floatAddIO` is the kernel's **IO signature**: which buffer is which
   argument, where each program reads its input tiles, where it writes its
   output tile, and — new on this surface — the declared output dtype
   `outDType := .fp32`. `⊨[R]` is the audit-once rounding-correctness
   combinator (`KernelIO₂.ImplementsR`, `VeriTile.Triton.Memory.KernelSpec`);
   spelled out, the headline says: for **every** rounding model `R`,
   **every** disjoint placement of the three buffers in flat memory,
   **every** program id whose windows are in bounds, and **every** launch
   state whose input windows hold `xs`/`ys` exactly — the translated pointer
   kernel terminates under `execR R`, its output window holds the fp32-typed
   cells `R.round .fp32 (xs i + ys i)`, and every other memory cell is
   unchanged.
-/

import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FloatDTypeCorrect

open VeriTile.Triton
open VeriTile.Triton.KernelIO₂ (ImplementsR)
open scoped VeriTile.Triton.KernelIO₂

/-- Local reduction helper: the algorithm-projection traversals
(`ComputeStmt.toAlgorithm?` and friends) are written in `do` notation over
`Except`, so their equation lemmas leave `Except.ok _ >>= f` redexes behind.
This discharges them for `simp`. -/
@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-! ## Part 1 — the kernel -/

/-- Elementwise add with fp32-annotated input memory and an fp32-quantized
store: the loads carry Triton-like `dtype=tl.float32` information, and the
stored value crosses an explicit `.round_to(tl.float32)` quantization cast.
The store-side fp32 is **modeled** by the headline below (the output cells
are fp32-typed and the stored value is rounded at the fp32 grid); the
load-side fp32 is erased by the compute→algorithm projection (inputs are
consumed as exact ℝ), as documented by the region-typing facts below. -/
def floatAddKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  y    := tl.load($(yReg) + offs, dtype=tl.float32)
  out  := x + y
  tl.store($(outReg) + offs, (out).round_to(tl.float32))
}

/-! ## Part 2 — literal projected body, region-model Hoare triple, typing facts -/

/-- The kernel's projected body, pinned to its literal statement list: six
statements (`pid`, `offs`, the two loads, `out = x + y`, the cast store).
The projection erases the loads' fp32 annotations to the Real channel but
**preserves the store cast**: the store is fp32-typed and its value is
`Op.castFloat .real .fp32 out` — the rounding site the headline models.
Every obligation below rewrites the kernel's projection along this equation
first, then runs the computational walk on the literal body. -/
private theorem floatAdd_toAlg (xReg yReg outReg : RegionName) (B : Nat) :
    (floatAddKernel xReg yReg outReg B).toAlgKernel
      = Kernel.mk [xReg, yReg] [outReg]
          [Stmt.assign TileDType.nat [] "pid" (Op.programId 0),
           Stmt.assign TileDType.nat [B] "offs"
             (Op.add NumericDType.nat Broadcast.scalarL
               (Op.mul NumericDType.nat Broadcast.nil
                 (Op.ref TileDType.nat [] "pid") (Op.constNat B))
               (Op.arange B)),
           Stmt.assign TileDType.real [B] "x"
             (Op.load TileDType.real
               (MemAccess.region xReg (Op.ref TileDType.nat [B] "offs"))
               MaskOpt.none),
           Stmt.assign TileDType.real [B] "y"
             (Op.load TileDType.real
               (MemAccess.region yReg (Op.ref TileDType.nat [B] "offs"))
               MaskOpt.none),
           Stmt.assign TileDType.real [B] "out"
             (Op.add NumericDType.real Broadcast.nil.consSame
               (Op.ref TileDType.real [B] "x") (Op.ref TileDType.real [B] "y")),
           Stmt.store TileDType.fp32 [B]
             (MemAccess.region (Region.cast outReg) (Op.ref TileDType.nat [B] "offs"))
             (Op.castFloat FloatDType.real FloatDType.fp32
               (Op.ref TileDType.real [B] "out"))
             MaskOpt.none] := by
  simp [floatAddKernel, ComputeKernel.toAlgKernel, ComputeStmt.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType]
  try rfl

/-- The original claim `Kernel.RespectsRegionTyping (fun _ => .fp32)
(floatAddKernel …)` is **false** under today's definitions and is therefore
recorded as a proved negation. `Kernel.RespectsRegionTyping` receives the
kernel through the `ComputeKernel → AlgKernel` coercion (`toAlgKernel`), and
the compute→algorithm projection (`ComputeOp.toAlgorithm?`) rewrites every
`tl.float32` load to `Op.load ComputeDType.fp32.eraseDType = Op.load .real`
before the typing contract ever sees it. The contract's load cases then
demand `Γ region = .real`, i.e. `TileDType.fp32 = TileDType.real`, which
reduces to `False`. -/
private theorem float_add_not_respects_fp32_regions
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    ¬ Kernel.RespectsRegionTyping (fun _ => .fp32)
      (floatAddKernel xReg yReg outReg blockSize) := by
  simp [floatAddKernel, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.RespectsRegionTyping, StmtList.RespectsRegionTyping,
    Stmt.RespectsRegionTyping, Op.RespectsRegionTyping,
    MemAccess.RespectsRegionTyping, MaskOpt.RespectsRegionTyping]

/-- The true region-typing contract on today's projected surface: the loads'
fp32 annotations are erased (the input regions type as `.real`), but the
store's fp32 dtype **survives** the projection (it entered through the
rounding-channel `round_to`, not the erased compute channel) — the output
region types as `.fp32`. Stated at the headline's concrete buffers so the
mixed typing is a plain function of the region name. (The old all-`.real`
typing fact died with the raw store: the output region is now genuinely
fp32-typed.) -/
private theorem float_add_respects_mixed_regions (blockSize : Nat) :
    Kernel.RespectsRegionTyping
      (fun r => if r = ⟨"out"⟩ then .fp32 else .real)
      (floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ blockSize) := by
  simp [floatAddKernel, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.RespectsRegionTyping, StmtList.RespectsRegionTyping,
    Stmt.RespectsRegionTyping, Op.RespectsRegionTyping.eq_def,
    MemAccess.RespectsRegionTyping, MaskOpt.RespectsRegionTyping]

/-- Readback of an unmasked `writeMemAsR` scatter at lane `i`'s offset, given
injective offsets: the `R`-rounded typed cell. Unmasked corollary of the
library `BlockState.scatter_memcell_R_prop_masked_nd`. -/
private theorem scatter_writeMemAsR_readback {shape : TileShape}
    (R : RoundingModel) (dtype : FloatDType) {region : RegionName}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier dtype.toTileDType)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => acc.writeMemAsR R dtype region (offsetFn k) (valueFn k))
       s).mem region (offsetFn i)
      = MemCell.of dtype.toTileDType
          (dtype.ofReal (R.storeValue dtype (valueFn i))) := by
  have h := BlockState.scatter_memcell_R_prop_masked_nd R dtype
    (region := region) s offsetFn valueFn (fun _ => True) h_inj i
  simpa using h

/-- **The rounded region-model Hoare triple** — termination under `execR R`,
typed fp32 output readback, and frame, from any launch state whose input
windows are loaded (exact ℝ). The stored value is rounded twice — once at
the `.round_to(tl.float32)` cast (site 1), once at the fp32-typed store
(site 2) — and `R.round_idem` collapses the pair to the single boundary
round the headline states. This is what the `⊨[R]` headline transports to
flat memory. -/
private theorem floatAdd_region_runR (R : RoundingModel) (B : Nat)
    (s₀ : BlockState) (xs ys : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.readMem ⟨"x"⟩ (s₀.pid * B + j.val) = xs j)
    (hy : ∀ j : Fin B, s₀.readMem ⟨"y"⟩ (s₀.pid * B + j.val) = ys j) :
    ∃ s1, execR R ((floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).toAlgKernel) s₀
        = some s1
      ∧ (∀ j : Fin B,
          s1.readMemAs .fp32 ⟨"out"⟩ (s₀.pid * B + j.val)
            = FloatDType.fp32.ofReal (R.round .fp32 (xs j + ys j)))
      ∧ (∀ r o,
          (r ≠ ⟨"out"⟩ ∨ ∀ j : Fin B, o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  simp only [BlockState.pid_eq] at hx hy
  rw [floatAdd_toAlg]
  simp [execR, stepStmtsR, stepStmtR, evalOpR.eq_def, Tile.bop,
    NumericDType.add, NumericDType.mul]
  have h_inj : Function.Injective
      (fun idx : TileIndex [B] => s₀.pids 0 * B + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  refine ⟨fun j => ?_, fun r o hcond => ?_⟩
  · -- value: single-lane readback of the R-store scatter, then the
    -- cast/store value chain collapses by `round_idem`
    unfold BlockState.readMemAs
    rw [scatter_writeMemAsR_readback R .fp32 _ _ _ h_inj (j, PUnit.unit)]
    simp [MemCell.readAs_of_same, FloatDType.ofReal, FloatDType.storeValue,
      RoundingModel.storeValue, RoundingModel.cast, hx j, hy j, R.round_idem]
  · -- frame: the scatter misses (r, o), and `setReg` never touches memory
    refine Eq.trans
      (BlockState.foldl_writeMemAsR_preserve_cell R .fp32 _ _ r o _ ?_ _) rfl
    intro k _ hc
    rcases hcond with hne | hno
    · exact hne hc.1.symm
    · exact hno k.1 hc.2.symm

/-! ## Part 3 — flat-memory bridge side conditions

The projected kernel is register-indirect (`offs := ...; tl.load(x + offs)`),
so no ∀-state contract covers it; the flat-memory bridge (v1.2, `execR`
flavor) takes the per-execution `Kernel.TraceSafeR` contract instead, plus
`FlattenOk` (bridge fragment membership). Because the kernel is **unmasked**,
`MaskOpt.ActiveR` makes every lane active, and the bounds obligation is the
aligned contract `pid * B + B ≤ bounds reg` — the whole program tile in
bounds — for each of the three regions; the cast is memory-silent. -/

/-- The projected kernel sits inside the bridge's covered fragment. -/
private theorem floatAdd_flattenOk (xReg yReg outReg : RegionName) (B : Nat) :
    ((floatAddKernel xReg yReg outReg B).toAlgKernel).FlattenOk := by
  rw [floatAdd_toAlg]
  unfold Kernel.FlattenOk
  simp [StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in
/-- Per-execution safety walk under `R`: one computational unfold walks all
six statements of the literal body, discharging every memory-silent
`SafeAtR` (including the store's cast) and reducing the three unmasked
accesses' address obligations to the whole-tile bounds hypotheses. -/
private theorem floatAdd_traceSafeR (xReg yReg outReg : RegionName) (B : Nat)
    (R : RoundingModel) (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg)
    (hout : s.pid * B + B ≤ bounds outReg) :
    Kernel.TraceSafeR R bounds
      ((floatAddKernel xReg yReg outReg B).toAlgKernel) s := by
  rw [floatAdd_toAlg]
  unfold Kernel.TraceSafeR
  simp only [BlockState.pid_eq] at hx hy hout
  simp [Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
    MaskOpt.SafeAtR, MemAccess.SafeAtR, stepStmtR, evalOpR.eq_def,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR,
    MaskOpt.ActiveR, BlockState.setReg,
    Tile.bop, NumericDType.add, NumericDType.mul]
  exact ⟨fun i => lt_of_lt_of_le (Nat.add_lt_add_left i.isLt _) hx,
    fun i => lt_of_lt_of_le (Nat.add_lt_add_left i.isLt _) hy,
    fun i => lt_of_lt_of_le (Nat.add_lt_add_left i.isLt _) hout⟩

/-! ## Part 4 — the spec: `floatAddIO ⊨[R]` pointwise addition -/

/-- The float add kernel's **IO signature** — the whole kernel-specific
audit surface of the headline:

* `kernel` — `floatAddKernel` itself: no erasure; the fp32 store is modeled;
* `in1`/`in2`/`out` — which buffer is which argument (the wiring);
* `B` — the tile length;
* `read1`/`read2` — where program `pid` reads its input tiles (the address
  half of the *pre*condition);
* `write` — where program `pid` writes its output tile (the address half
  of the *post*condition: rounded fp32 values land there, frame holds
  everywhere else);
* `outDType := .fp32` — the declared quantization grid of the boundary
  store; the headline **proves** the kernel's actual store cast matches.

The windows are declared, not parsed from the kernel: they formalize the
host-side launch convention (`offsets = pid * BLOCK_SIZE + arange`), and
the headline **proves** the kernel's actual addressing matches them.
Buffer sizes are not signature content: the headline quantifies over every
allocation large enough for the windows. -/
def floatAddIO (B : Nat) : KernelIO₂ where
  kernel := floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B
  in1 := ⟨"x"⟩
  in2 := ⟨"y"⟩
  out := ⟨"out"⟩
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  outDType := .fp32

/-- **The headline**: the float-annotated add kernel implements pointwise
addition on its IO signature under the boundary-rounding contract, for
**every** rounding model — see the module docstring for the full Hoare
triple `⊨[R]` unfolds to (output window = fp32-typed cells holding
`R.round .fp32 (xs i + ys i)`; at `R := .triv` this is the exact statement).
Proof: `ImplementsR.intro` assembles the rounded region-model triple
(Part 2) with the bridge side conditions (Part 3), each stated on the
kernel's literal projected body (`floatAdd_toAlg`). -/
specification float_add_correctness (R : RoundingModel) (B : Nat) :
    floatAddIO B ⊨[R] fun xs ys i => xs i + ys i := by
  refine KernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact floatAdd_flattenOk ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B
  · intro bounds s h1 h2 h3
    exact floatAdd_traceSafeR ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B R bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    show ∃ s1,
      execR R ((floatAddKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).toAlgKernel) s₀ = some s1
        ∧ (∀ j : Fin B,
            s1.readMemAs .fp32 ⟨"out"⟩ (s₀.pid * B + j.val)
              = FloatDType.fp32.ofReal (R.round .fp32 (xs j + ys j)))
        ∧ (∀ r o,
            (r ≠ ⟨"out"⟩ ∨ ∀ j : Fin B, o ≠ s₀.pid * B + j.val) →
            s1.mem r o = s₀.mem r o)
    exact floatAdd_region_runR R B s₀ xs ys hx hy

/-! ## Trust gates -/

-- No `sorry`, no smuggled axiom, in the headline's transitive proof.
#axiomsClean float_add_correctness

/- The headline's statement surface is the IO signature plus the audit-once
rounding-correctness combinator and the rounding-model interface — no other
project constant. The output-dtype policy (`outDType := .fp32`) lives inside
`floatAddIO`'s definition, one definition-click away from the headline. -/
#stmtSurfaceSubset float_add_correctness ⊆
  [floatAddIO, VeriTile.Triton.KernelIO₂.ImplementsR,
   VeriTile.Triton.KernelIO₂.B, RoundingModel]

end VeriTile.Bench.Examples.FloatDTypeCorrect
