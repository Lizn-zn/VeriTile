/-
VeriTile.Triton.Memory.Locality

Execution locality (#487 step 2): a trace-safe kernel only observes memory
inside the in-bounds window of its `RegionBounds`, so running it from any
state that agrees with the reference state **inside** the window — and on
every non-mem component — produces the same in-window results. Junk outside
the kernel's windows cannot change the answer.

The denotation combinator (`VeriTile.Triton.Memory.Denotation`) runs the
flattened kernel from a canonical start state whose memory is `0` outside
the loaded slots. This module removes that idealization: `exec_agreeOn`
shows the run is insensitive to out-of-window memory content, so the
canonical-state denotation transports to *any* real start state agreeing
with it inside the allocated windows.

Structure (mirrors the flat-memory bridge walk in `Flatten.lean`, which
solved the same mutual-recursion shape):

* `BlockState.AgreeOn` — the agreement relation; `RegionBounds.Window` — the
  in-bounds window of a bounds map.
* Read/write congruence helpers: reads at agreed cells are equal; writes of
  equal values at equal addresses preserve agreement (scatter-fold and
  atomic-fold versions).
* `evalOp_agreeOn` — expression evaluation is **equal** on agreeing states
  (loads only read active in-window cells, the `undef` channel, or the
  `other` operand; inactive lanes never read memory).
* `stepAtomicRMW_agreeOn` — the atomicRMW statement, via the
  `rmwExtras`/`rmwActive`/`rmwDispatch`/`rmwCommit` decomposition.
* `stepStmt_agreeOn` / `stepStmts_agreeOn` / loop auxiliaries /
  `exec_agreeOn` — the mutual walk over statements, threading
  `Stmt.TraceSafe` along the reference execution.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Memory.Bounds
import VeriTile.Triton.Memory.Flatten

namespace VeriTile.Triton

/-! ## The agreement relation -/

/-- The in-bounds window of a bounds map: region `r` at offsets `< bounds r`.
A trace-safe kernel touches memory only inside this window. -/
abbrev RegionBounds.Window (bounds : RegionBounds) (r : RegionName) (o : Nat) :
    Prop :=
  o < bounds r

/-- Two states agree inside the window `P` and on every non-mem component.
Out-of-window memory (`¬ P r o`) is unconstrained — that is the junk the
locality theorems tolerate. -/
def BlockState.AgreeOn (P : RegionName → Nat → Prop) (s₁ s₂ : BlockState) :
    Prop :=
  (∀ r o, P r o → s₁.mem r o = s₂.mem r o) ∧ s₁.regs = s₂.regs ∧
  s₁.pids = s₂.pids ∧ s₁.undef = s₂.undef ∧ s₁.numPids = s₂.numPids

namespace BlockState.AgreeOn

variable {P : RegionName → Nat → Prop} {s₁ s₂ : BlockState}

theorem mem (h : s₁.AgreeOn P s₂) {r : RegionName} {o : Nat} (hP : P r o) :
    s₁.mem r o = s₂.mem r o := h.1 r o hP

theorem regs (h : s₁.AgreeOn P s₂) : s₁.regs = s₂.regs := h.2.1

theorem pids (h : s₁.AgreeOn P s₂) : s₁.pids = s₂.pids := h.2.2.1

theorem undef (h : s₁.AgreeOn P s₂) : s₁.undef = s₂.undef := h.2.2.2.1

theorem numPids (h : s₁.AgreeOn P s₂) : s₁.numPids = s₂.numPids := h.2.2.2.2

theorem refl (P : RegionName → Nat → Prop) (s : BlockState) :
    s.AgreeOn P s :=
  ⟨fun _ _ _ => rfl, rfl, rfl, rfl, rfl⟩

/-- Agreement is preserved by writing the same register value on both
sides. -/
theorem setReg (h : s₁.AgreeOn P s₂) (name : RegName) (dtype : TileDType)
    (shape : TileShape) (v : Tile dtype shape) :
    (s₁.setReg name dtype shape v).AgreeOn P (s₂.setReg name dtype shape v) := by
  obtain ⟨hmem, hregs, hpids, hundef, hnum⟩ := h
  refine ⟨fun r o hP => hmem r o hP, ?_, hpids, hundef, hnum⟩
  simp only [BlockState.setReg, hregs]

/-- Agreement is preserved by resetting `pids` to equal maps on both sides. -/
theorem withPids (h : s₁.AgreeOn P s₂) {p₁ p₂ : Nat → Nat} (hp : p₁ = p₂) :
    ({ s₁ with pids := p₁ } : BlockState).AgreeOn P { s₂ with pids := p₂ } := by
  obtain ⟨hmem, hregs, _, hundef, hnum⟩ := h
  exact ⟨fun r o hP => hmem r o hP, hregs, hp, hundef, hnum⟩

end BlockState.AgreeOn

/-! ## Read congruence

Every operational read factors through the single cell `s.mem r o`, so
agreement at that cell gives equal reads. -/

namespace BlockState

/-- `readMem` congruence at an agreed cell. -/
theorem readMem_congr {s₁ s₂ : BlockState} {r : RegionName} {o : Nat}
    (h : s₁.mem r o = s₂.mem r o) :
    s₁.readMem r o = s₂.readMem r o := by
  unfold readMem
  rw [h]

/-- `readMemAs` congruence at an agreed cell. -/
theorem readMemAs_congr {s₁ s₂ : BlockState} (dtype : FloatDType)
    {r : RegionName} {o : Nat} (h : s₁.mem r o = s₂.mem r o) :
    s₁.readMemAs dtype r o = s₂.readMemAs dtype r o := by
  unfold readMemAs
  rw [h]

/-- `readMemTyped` congruence at an agreed cell. -/
theorem readMemTyped_congr {s₁ s₂ : BlockState} (dtype : TileDType)
    {r : RegionName} {o : Nat} (h : s₁.mem r o = s₂.mem r o) :
    s₁.readMemTyped dtype r o = s₂.readMemTyped dtype r o := by
  unfold readMemTyped
  rw [h]

/-- The operational `tl.load` cell read is equal at an agreed cell. -/
theorem readMemValue_congr {s₁ s₂ : BlockState} (dtype : TileDType)
    {r : RegionName} {o : Nat} (h : s₁.mem r o = s₂.mem r o) :
    s₁.readMemValue dtype r o = s₂.readMemValue dtype r o := by
  cases dtype <;>
    simp only [readMemValue, readMemAs_congr _ h, readMemTyped_congr _ h]

end BlockState

/-! ## Write congruence

Both runs execute the same stores with equal operands, so they write the
same values at the same addresses: agreement is preserved on **every**
window (written cells hold the same new value; unwritten cells are
untouched). -/

namespace BlockState

/-- Writing the same typed value at the same address on both sides leaves
agreed cells agreed. -/
theorem writeMemTyped_mem_congr {s₁ s₂ : BlockState} {r : RegionName} {o : Nat}
    (dtype : TileDType) (region : RegionName) (offset : Nat)
    (v : TileCarrier dtype) (h : s₁.mem r o = s₂.mem r o) :
    (s₁.writeMemTyped dtype region offset v).mem r o
      = (s₂.writeMemTyped dtype region offset v).mem r o := by
  cases dtype <;>
    (show (if r = region ∧ o = offset then _ else s₁.mem r o)
        = (if r = region ∧ o = offset then _ else s₂.mem r o)) <;>
    (by_cases hc : r = region ∧ o = offset
     · rw [if_pos hc, if_pos hc]
     · rw [if_neg hc, if_neg hc, h])

/-- Agreement is preserved by a single typed write of the same value at the
same address on both sides. -/
theorem AgreeOn.writeMemTyped {P : RegionName → Nat → Prop}
    {s₁ s₂ : BlockState} (h : s₁.AgreeOn P s₂) (dtype : TileDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype) :
    (s₁.writeMemTyped dtype region offset v).AgreeOn P
      (s₂.writeMemTyped dtype region offset v) := by
  obtain ⟨hmem, hregs, hpids, hundef, hnum⟩ := h
  refine ⟨fun r o hP => writeMemTyped_mem_congr dtype region offset v
      (hmem r o hP), ?_, ?_, ?_, ?_⟩
  · funext d sh nm
    rw [writeMemTyped_regs, writeMemTyped_regs, hregs]
  · rw [writeMemTyped_pids, writeMemTyped_pids, hpids]
  · funext r o
    rw [writeMemTyped_undef, writeMemTyped_undef, hundef]
  · rw [writeMemTyped_numPids, writeMemTyped_numPids, hnum]

/-- Writing the same raw cell at the same address on both sides leaves
agreed cells agreed. -/
theorem writeCell_mem_congr {s₁ s₂ : BlockState} {r : RegionName} {o : Nat}
    (region : RegionName) (offset : Nat) (cell : MemCell)
    (h : s₁.mem r o = s₂.mem r o) :
    (s₁.writeCell region offset cell).mem r o
      = (s₂.writeCell region offset cell).mem r o := by
  show (if r = region ∧ o = offset then cell else s₁.mem r o)
      = (if r = region ∧ o = offset then cell else s₂.mem r o)
  by_cases hc : r = region ∧ o = offset
  · rw [if_pos hc, if_pos hc]
  · rw [if_neg hc, if_neg hc, h]

/-- Agreement is preserved by a single raw-cell write of the same cell at
the same address on both sides. -/
theorem AgreeOn.writeCell {P : RegionName → Nat → Prop} {s₁ s₂ : BlockState}
    (h : s₁.AgreeOn P s₂) (region : RegionName) (offset : Nat)
    (cell : MemCell) :
    (s₁.writeCell region offset cell).AgreeOn P
      (s₂.writeCell region offset cell) := by
  obtain ⟨hmem, hregs, hpids, hundef, hnum⟩ := h
  exact ⟨fun r o hP => writeCell_mem_congr region offset cell (hmem r o hP),
    hregs, hpids, hundef, hnum⟩

end BlockState

/-! ## Scatter-fold congruence -/

/-- A masked typed scatter with the same regions/offsets/values/mask on both
sides preserves agreement (no in-bounds hypothesis needed: identical writes
land on identical cells). Covers the `.region`, `.ptr`, and `.blockPtr`
store folds. -/
theorem foldl_writeMemTyped_agreeOn {α : Type} (dtype : TileDType)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype) (active : α → Bool)
    (P : RegionName → Nat → Prop) :
    ∀ (l : List α) (s₁ s₂ : BlockState), s₁.AgreeOn P s₂ →
      (l.foldl (fun acc i =>
          if active i then
            acc.writeMemTyped dtype (regionFn i) (offsetFn i) (valueFn i)
          else acc) s₁).AgreeOn P
        (l.foldl (fun acc i =>
          if active i then
            acc.writeMemTyped dtype (regionFn i) (offsetFn i) (valueFn i)
          else acc) s₂)
  | [], s₁, s₂, h => h
  | i :: rest, s₁, s₂, h => by
      rw [List.foldl_cons, List.foldl_cons]
      by_cases hact : active i = true
      · rw [if_pos hact, if_pos hact]
        exact foldl_writeMemTyped_agreeOn dtype regionFn offsetFn valueFn
          active P rest _ _ (h.writeMemTyped dtype _ _ _)
      · rw [if_neg hact, if_neg hact]
        exact foldl_writeMemTyped_agreeOn dtype regionFn offsetFn valueFn
          active P rest _ _ h

/-- The atomic-add scatter: the written value reads the accumulator's target
cell, so active cells must sit inside the agreement window (they do — that
is exactly the `ActiveAddressSafe` obligation of `Stmt.TraceSafe`). -/
theorem foldl_atomicAdd_agreeOn {α : Type} {dtype : TileDType}
    (hnum : NumericDType dtype) (regionFn : α → RegionName)
    (offsetFn : α → Nat) (inputFn : α → TileCarrier dtype)
    (active : α → Bool) (P : RegionName → Nat → Prop) :
    ∀ (l : List α) (s₁ s₂ : BlockState), s₁.AgreeOn P s₂ →
      (∀ i ∈ l, active i = true → P (regionFn i) (offsetFn i)) →
      (l.foldl (fun acc i =>
          if active i then
            acc.writeMemTyped dtype (regionFn i) (offsetFn i)
              (hnum.add (acc.readMemValue dtype (regionFn i) (offsetFn i))
                (inputFn i))
          else acc) s₁).AgreeOn P
        (l.foldl (fun acc i =>
          if active i then
            acc.writeMemTyped dtype (regionFn i) (offsetFn i)
              (hnum.add (acc.readMemValue dtype (regionFn i) (offsetFn i))
                (inputFn i))
          else acc) s₂)
  | [], s₁, s₂, h, _ => h
  | i :: rest, s₁, s₂, h, hwin => by
      rw [List.foldl_cons, List.foldl_cons]
      by_cases hact : active i = true
      · rw [if_pos hact, if_pos hact]
        rw [BlockState.readMemValue_congr dtype
          (h.mem (hwin i List.mem_cons_self hact))]
        exact foldl_atomicAdd_agreeOn hnum regionFn offsetFn inputFn active P
          rest _ _ (h.writeMemTyped dtype _ _ _)
          (fun j hj => hwin j (List.mem_cons_of_mem i hj))
      · rw [if_neg hact, if_neg hact]
        exact foldl_atomicAdd_agreeOn hnum regionFn offsetFn inputFn active P
          rest _ _ h (fun j hj => hwin j (List.mem_cons_of_mem i hj))

/-! ## Optional-result agreement -/

/-- Both computations are stuck, or both produce results related by `R` —
the conclusion shape of the locality walk (execution from agreeing states
gets stuck in lockstep). -/
inductive OptionRel {α β : Type} (R : α → β → Prop) : Option α → Option β → Prop
  | none : OptionRel R none none
  | some {a : α} {b : β} : R a b → OptionRel R (some a) (some b)

/-- Agreement relation on optional states: both stuck, or both step to
states agreeing on `P`. -/
abbrev BlockState.AgreeOnO (P : RegionName → Nat → Prop) :
    Option BlockState → Option BlockState → Prop :=
  OptionRel (BlockState.AgreeOn P)

/-! ## Expression evaluation is equal on agreeing states

`evalOp` reads memory only in the `.load` case, and only at **active**
in-window addresses (`Op.SafeAt`'s `ActiveAddressSafe` obligation). Inactive
mask lanes never read memory: they read the `undef` channel (equal by
`AgreeOn`), the dtype default, or the `other` operand. So evaluation on
agreeing states returns **equal** tiles — not merely related ones. -/

/-- `mapM`-level congruence for dynamic block-pointer offset lists. -/
private theorem mapM_data_agreeOn (s₁ s₂ : BlockState) :
    ∀ (l : List (Op TileDType.nat [])),
      (∀ off ∈ l, evalOp off s₁ = evalOp off s₂) →
      l.mapM (fun off => do
          let v ← evalOp off s₁
          some (v.data PUnit.unit))
        = l.mapM (fun off => do
            let v ← evalOp off s₂
            some (v.data PUnit.unit))
  | [], _ => rfl
  | o :: rest, h => by
      simp only [List.mapM_cons]
      rw [h o (List.mem_cons_self ..)]
      cases evalOp o s₂ with
      | none => rfl
      | some v =>
          rw [mapM_data_agreeOn s₁ s₂ rest
            (fun off hoff => h off (List.mem_cons_of_mem _ hoff))]

set_option maxHeartbeats 3200000 in
/-- **Op-level locality**: on states agreeing inside the window of `bounds`
(and on every non-mem component), an op that is safe at the first state
evaluates identically at both. -/
theorem evalOp_agreeOn (bounds : RegionBounds) :
    ∀ {d : TileDType} {sh : TileShape} (e : Op d sh) (s₁ s₂ : BlockState),
      s₁.AgreeOn bounds.Window s₂ → e.SafeAt bounds s₁ →
      evalOp e s₁ = evalOp e s₂
  | _, _, .const _, _, _, _, _ => by simp only [evalOp]
  | _, _, .constFloat _ _, _, _, _, _ => by simp only [evalOp]
  | _, _, .constNat _, _, _, _, _ => by simp only [evalOp]
  | _, _, .constInt _, _, _, _, _ => by simp only [evalOp]
  | _, _, .constBool _, _, _, _, _ => by simp only [evalOp]
  | _, _, .negInf, _, _, _, _ => by simp only [evalOp]
  | _, _, .arange _, _, _, _, _ => by simp only [evalOp]
  | _, _, .ptrBase _, _, _, _, _ => by simp only [evalOp]
  | _, _, .makeBlockPtr _ _ _ _ _ _, _, _, _, _ => by simp only [evalOp]
  | _, _, .programId ax, s₁, s₂, hag, _ => by
      simp only [evalOp, hag.pids]
  | _, _, .numPrograms ax, s₁, s₂, hag, _ => by
      simp only [evalOp, hag.numPids]
  | _, _, .ref d sh n, s₁, s₂, hag, _ => by
      simp only [evalOp, hag.regs]
  | _, _, .broadcast e sh, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds e s₁ s₂ hag hms]
  | _, _, .full sh e, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds e s₁ s₂ hag hms]
  | _, _, .castFloat src dst a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .castNatToInt a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .castIntToNat a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .castRealToInt8 a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .add nd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .sub nd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .mul nd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .div nd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .floorDiv nd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .mod nd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .bitAnd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .bitOr bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .bitXor bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .shiftLeft bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .shiftRight bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .exp a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .exp2 a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .log a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .log2 a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .sigmoid a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .sqrt a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .rsqrt a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .tanh a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .sin a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .cos a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .tan a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .atan a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .cosh a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .sinh a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .erf a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .lt cd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .le cd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .eq cd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .gt cd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .ge cd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .ne cd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .boolAnd bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .boolOr bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .boolNot a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .max2 bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .pow bc a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .where c a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds c s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds a s₁ s₂ hag hms.2.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2.2]
  | _, _, .ite c a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds c s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds a s₁ s₂ hag hms.2.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2.2]
  | _, _, .reduceMax ax kd a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .reduceMaxNat ax kd a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .reduceSum ax kd a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .scan op ax dir a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .argMax ax a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .argMin ax a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .sort ax a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .dot a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .dotInt a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .transpose a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .reshape outSh a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .remap outSh f a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .join a b, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds b s₁ s₂ hag hms.2]
  | _, _, .split side a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .expandDim ax a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .ptrAdd bc p o, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds p s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds o s₁ s₂ hag hms.2]
  | _, _, .ptrSub bc p o, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds p s₁ s₂ hag hms.1,
        evalOp_agreeOn bounds o s₁ s₂ hag hms.2]
  | _, _, .makeBlockPtrDyn r bo ps bs strides offs, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds bo s₁ s₂ hag hms]
  | _, _, .makeBlockPtrDynOffsets r bo ps bs strides offs, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      obtain ⟨hmb, hmoffs⟩ := hms
      have hmem : ∀ off ∈ offs, evalOp off s₁ = evalOp off s₂ :=
        fun off hoff => evalOp_agreeOn bounds off s₁ s₂ hag (hmoffs off hoff)
      simp only [evalOp, evalOp_agreeOn bounds bo s₁ s₂ hag hmb,
        mapM_data_agreeOn s₁ s₂ offs hmem]
  | _, _, .advanceBlockPtr p ds, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds p s₁ s₂ hag hms]
  | _, _, .natToReal a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .intToReal a, s₁, s₂, hag, hms => by
      simp only [Op.SafeAt] at hms
      simp only [evalOp, evalOp_agreeOn bounds a s₁ s₂ hag hms]
  | _, _, .load d mem mask, s₁, s₂, hag, hms => by
      cases mem with
      | region r off =>
          cases mask with
          | none =>
              simp only [Op.SafeAt] at hms
              obtain ⟨hmsmem, -, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe] at haddr
              have IHoff := evalOp_agreeOn bounds off s₁ s₂ hag hmsmem
              cases hoffs : evalOp off s₂ with
              | none => simp [IHoff, hoffs]
              | some offs =>
                  simp only [evalOp, IHoff, hoffs]
                  refine congrArg some (Tile.ext fun i => ?_)
                  simp only [reduceIte]
                  exact BlockState.readMemValue_congr d
                    (hag.mem (haddr offs (IHoff.trans hoffs) i trivial))
          | mask m =>
              simp only [Op.SafeAt] at hms
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe] at haddr
              have IHoff := evalOp_agreeOn bounds off s₁ s₂ hag hmsmem
              have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask
              cases hoffs : evalOp off s₂ with
              | none => simp [evalOp, IHoff, hoffs]
              | some offs =>
                  cases hm : evalOp m s₂ with
                  | none => simp [evalOp, IHoff, hoffs, IHm, hm]
                  | some ms =>
                      simp only [evalOp, IHoff, hoffs, IHm, hm]
                      refine congrArg some (Tile.ext fun i => ?_)
                      simp only [reduceIte]
                      by_cases hact : ms.data i = true
                      · simp only [hact, if_true]
                        exact BlockState.readMemValue_congr d
                          (hag.mem (haddr offs (IHoff.trans hoffs) i
                            ⟨ms, IHm.trans hm, hact⟩))
                      · simp only [Bool.not_eq_true] at hact
                        simp only [hact, Bool.false_eq_true, if_false]
                        cases d <;> simp [hag.undef]
          | maskOther m other =>
              simp only [Op.SafeAt] at hms
              obtain ⟨hmsmem, ⟨hmm, hmo⟩, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe] at haddr
              have IHoff := evalOp_agreeOn bounds off s₁ s₂ hag hmsmem
              have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmm
              have IHo := evalOp_agreeOn bounds other s₁ s₂ hag hmo
              cases hoffs : evalOp off s₂ with
              | none => simp [evalOp, IHoff, hoffs]
              | some offs =>
                  cases hm : evalOp m s₂ with
                  | none => simp [evalOp, IHoff, hoffs, IHm, hm]
                  | some ms =>
                      cases ho : evalOp other s₂ with
                      | none =>
                          simp [evalOp, IHoff, hoffs, IHm, hm, IHo, ho]
                      | some os =>
                          simp [evalOp, IHoff, hoffs, IHm, hm, IHo, ho]
                          funext i
                          by_cases hact : ms.data i = true
                          · simp only [hact, if_true]
                            exact BlockState.readMemValue_congr d
                              (hag.mem (haddr offs (IHoff.trans hoffs) i
                                ⟨ms, IHm.trans hm, hact⟩))
                          · simp only [Bool.not_eq_true] at hact
                            simp only [hact, Bool.false_eq_true, if_false]
      | ptr p =>
          cases mask with
          | none =>
              simp only [Op.SafeAt] at hms
              obtain ⟨hmsmem, -, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
              cases hps : evalOp p s₂ with
              | none => simp [evalOp, IHp, hps]
              | some ps =>
                  simp only [evalOp, IHp, hps]
                  refine congrArg some (Tile.ext fun i => ?_)
                  simp only [reduceIte]
                  exact BlockState.readMemValue_congr d
                    (hag.mem (haddr.2 ps (IHp.trans hps) i trivial))
          | mask m =>
              simp only [Op.SafeAt] at hms
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
              have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask
              cases hps : evalOp p s₂ with
              | none => simp [evalOp, IHp, hps]
              | some ps =>
                  cases hm : evalOp m s₂ with
                  | none => simp [evalOp, IHp, hps, IHm, hm]
                  | some ms =>
                      simp only [evalOp, IHp, hps, IHm, hm]
                      refine congrArg some (Tile.ext fun i => ?_)
                      simp only [reduceIte]
                      by_cases hact : ms.data i = true
                      · simp only [hact, if_true]
                        exact BlockState.readMemValue_congr d
                          (hag.mem (haddr.2 ps (IHp.trans hps) i
                            ⟨ms, IHm.trans hm, hact⟩))
                      · simp only [Bool.not_eq_true] at hact
                        simp only [hact, Bool.false_eq_true, if_false]
                        cases d <;> simp [hag.undef]
          | maskOther m other =>
              simp only [Op.SafeAt] at hms
              obtain ⟨hmsmem, ⟨hmm, hmo⟩, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
              have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmm
              have IHo := evalOp_agreeOn bounds other s₁ s₂ hag hmo
              cases hps : evalOp p s₂ with
              | none => simp [evalOp, IHp, hps]
              | some ps =>
                  cases hm : evalOp m s₂ with
                  | none => simp [evalOp, IHp, hps, IHm, hm]
                  | some ms =>
                      cases ho : evalOp other s₂ with
                      | none =>
                          simp [evalOp, IHp, hps, IHm, hm, IHo, ho]
                      | some os =>
                          simp [evalOp, IHp, hps, IHm, hm, IHo, ho]
                          funext i
                          by_cases hact : ms.data i = true
                          · simp only [hact, if_true]
                            exact BlockState.readMemValue_congr d
                              (hag.mem (haddr.2 ps (IHp.trans hps) i
                                ⟨ms, IHm.trans hm, hact⟩))
                          · simp only [Bool.not_eq_true] at hact
                            simp only [hact, Bool.false_eq_true, if_false]
      | blockPtr p bc =>
          cases mask with
          | none =>
              simp only [Op.SafeAt] at hms
              obtain ⟨hmsmem, -, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe] at haddr
              have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
              cases hps : evalOp p s₂ with
              | none => simp [evalOp, IHp, hps]
              | some ps =>
                  simp only [evalOp, IHp, hps]
                  refine congrArg some (Tile.ext fun i => ?_)
                  by_cases hib : (ps.data i).inBounds
                      (TileShape.indexToList _ i) bc = true
                  · simp only [hib, if_true]
                    exact BlockState.readMemValue_congr d
                      (hag.mem (haddr.2 ps (IHp.trans hps) i trivial hib))
                  · simp only [Bool.not_eq_true] at hib
                    simp only [hib, Bool.false_eq_true, if_false]
          | mask m =>
              simp only [Op.SafeAt] at hms
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe] at haddr
              have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
              have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask
              cases hps : evalOp p s₂ with
              | none => simp [evalOp, IHp, hps]
              | some ps =>
                  cases hm : evalOp m s₂ with
                  | none => simp [evalOp, IHp, hps, IHm, hm]
                  | some ms =>
                      simp only [evalOp, IHp, hps, IHm, hm]
                      refine congrArg some (Tile.ext fun i => ?_)
                      by_cases hact : ms.data i = true
                      · simp only [hact, if_true]
                        by_cases hib : (ps.data i).inBounds
                            (TileShape.indexToList _ i) bc = true
                        · simp only [hib, if_true]
                          exact BlockState.readMemValue_congr d
                            (hag.mem (haddr.2 ps (IHp.trans hps) i
                              ⟨ms, IHm.trans hm, hact⟩ hib))
                        · simp only [Bool.not_eq_true] at hib
                          simp only [hib, Bool.false_eq_true, if_false]
                      · simp only [Bool.not_eq_true] at hact
                        simp only [hact, Bool.false_eq_true, if_false]
                        cases d <;> simp [hag.undef]
          | maskOther m other =>
              simp only [Op.SafeAt] at hms
              obtain ⟨hmsmem, ⟨hmm, hmo⟩, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe] at haddr
              have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
              have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmm
              have IHo := evalOp_agreeOn bounds other s₁ s₂ hag hmo
              cases hps : evalOp p s₂ with
              | none => simp [evalOp, IHp, hps]
              | some ps =>
                  cases hm : evalOp m s₂ with
                  | none => simp [evalOp, IHp, hps, IHm, hm]
                  | some ms =>
                      cases ho : evalOp other s₂ with
                      | none =>
                          simp [evalOp, IHp, hps, IHm, hm, IHo, ho]
                      | some os =>
                          simp [evalOp, IHp, hps, IHm, hm, IHo, ho]
                          funext i
                          by_cases hact : ms.data i = true
                          · simp only [hact, if_true]
                            by_cases hib : (ps.data i).inBounds
                                (TileShape.indexToList _ i) bc = true
                            · simp only [hib, if_true]
                              exact BlockState.readMemValue_congr d
                                (hag.mem (haddr.2 ps (IHp.trans hps) i
                                  ⟨ms, IHm.trans hm, hact⟩ hib))
                            · simp only [Bool.not_eq_true] at hib
                              simp only [hib, Bool.false_eq_true, if_false]
                          · simp only [Bool.not_eq_true] at hact
                            simp only [hact, Bool.false_eq_true, if_false]

/-! ## The atomicRMW statement

Mirrors the `stepAtomicRMW` decomposition: `rmwExtras`/`rmwActive` are
plain expression evaluations (equal on agreeing states), the
`foldAtomicRMWIndices` fold reads and rewrites only active in-window cells,
and `rmwCommit` is a register write of the equal returned tile. -/

/-- Active-lane monotonicity of the address-safety contract. -/
theorem MemAccess.ActiveAddressSafe.mono {dtype : TileDType}
    {shape : TileShape} {bounds : RegionBounds} {mem : MemAccess dtype shape}
    {s : BlockState} {active active' : TileIndex shape → Prop}
    (hsub : ∀ i, active' i → active i)
    (h : mem.ActiveAddressSafe bounds s active) :
    mem.ActiveAddressSafe bounds s active' := by
  cases mem with
  | region r off =>
      exact fun offsets hoffs i hi => h offsets hoffs i (hsub i hi)
  | ptr p =>
      exact ⟨h.1, fun ptrs hps i hi => h.2 ptrs hps i (hsub i hi)⟩
  | blockPtr p bc =>
      exact ⟨h.1, fun ptrs hps i hi => h.2 ptrs hps i (hsub i hi)⟩

/-- The optional CAS replacement operand evaluates identically. -/
theorem rmwExtras_agreeOn (bounds : RegionBounds) {dtype : TileDType}
    {shape : TileShape} (extraInput : Option (Op dtype shape))
    (s₁ s₂ : BlockState) (hag : s₁.AgreeOn bounds.Window s₂)
    (hms : extraInput.elim True (·.SafeAt bounds s₁)) :
    rmwExtras extraInput s₁ = rmwExtras extraInput s₂ := by
  cases extraInput with
  | none => rfl
  | some extra =>
      simp only [rmwExtras, evalOp_agreeOn bounds extra s₁ s₂ hag hms]

/-- The RMW activity map evaluates identically. -/
theorem rmwActive_agreeOn (bounds : RegionBounds) {dtype : TileDType}
    {shape : TileShape} (mask : MaskOpt dtype shape) (s₁ s₂ : BlockState)
    (hag : s₁.AgreeOn bounds.Window s₂) (hms : mask.SafeAt bounds s₁) :
    rmwActive mask s₁ = rmwActive mask s₂ := by
  cases mask with
  | none => rfl
  | mask m =>
      simp only [MaskOpt.SafeAt] at hms
      simp only [rmwActive, evalOp_agreeOn bounds m s₁ s₂ hag hms]
  | maskOther m other =>
      simp only [MaskOpt.SafeAt] at hms
      simp only [rmwActive, evalOp_agreeOn bounds m s₁ s₂ hag hms.1]

/-- A single-cell RMW at an agreed cell: both runs observe the same old
cell, so they commit the same new cell (agreement preserved) and return the
same value and event. -/
theorem BlockState.atomicRMWAt_agreeOn {P : RegionName → Nat → Prop}
    {s₁ s₂ : BlockState} (hag : s₁.AgreeOn P s₂) (op : RMWOp)
    (dtype : TileDType) (region : RegionName) (offset : Nat)
    (input : TileCarrier dtype) (extraInput : Option (TileCarrier dtype))
    (hcell : s₁.mem region offset = s₂.mem region offset) :
    OptionRel (fun p₁ p₂ : BlockState × TileCarrier dtype × RMWEvent =>
        p₁.1.AgreeOn P p₂.1 ∧ p₁.2 = p₂.2)
      (s₁.atomicRMWAt op dtype region offset input extraInput)
      (s₂.atomicRMWAt op dtype region offset input extraInput) := by
  unfold BlockState.atomicRMWAt
  rw [hcell]
  cases happ : RMWOp.apply op (s₂.mem region offset)
      { cell := (region, offset), op := op, input := MemCell.of dtype input,
        extraInput := extraInput.map (MemCell.of dtype) } with
  | none => simp only [happ]; exact .none
  | some out =>
      obtain ⟨nextCell, event'⟩ := out
      simp only [happ]
      exact .some ⟨hag.writeCell region offset nextCell, rfl⟩

/-- The RMW index fold from agreeing states: active cells sit inside the
window, so both runs observe/commit the same cells and accumulate the same
returned tile. -/
theorem foldAtomicRMWIndices_agreeOn {dtype : TileDType} {shape : TileShape}
    (op : RMWOp) (regionFn : TileIndex shape → RegionName)
    (offsetFn : TileIndex shape → Nat)
    (inputFn : TileIndex shape → TileCarrier dtype)
    (extraFn : Option (TileIndex shape → TileCarrier dtype))
    (active : TileIndex shape → Bool) (P : RegionName → Nat → Prop) :
    ∀ (l : List (TileIndex shape)) (s₁ s₂ : BlockState)
      (retFn : TileIndex shape → TileCarrier dtype),
      s₁.AgreeOn P s₂ →
      (∀ i ∈ l, active i = true → P (regionFn i) (offsetFn i)) →
      OptionRel (fun p₁ p₂ : BlockState × Tile dtype shape =>
          p₁.1.AgreeOn P p₂.1 ∧ p₁.2 = p₂.2)
        (foldAtomicRMWIndices op regionFn offsetFn inputFn extraFn active
          l s₁ retFn)
        (foldAtomicRMWIndices op regionFn offsetFn inputFn extraFn active
          l s₂ retFn)
  | [], s₁, s₂, retFn, hag, _ => by
      simp only [foldAtomicRMWIndices]
      exact .some ⟨hag, rfl⟩
  | i :: rest, s₁, s₂, retFn, hag, hwin => by
      simp only [foldAtomicRMWIndices]
      by_cases hact : active i = true
      · rw [if_pos hact, if_pos hact]
        have hAt := BlockState.atomicRMWAt_agreeOn hag op dtype (regionFn i)
          (offsetFn i) (inputFn i) (extraFn.map fun f => f i)
          (hag.mem (hwin i List.mem_cons_self hact))
        cases h1 : s₁.atomicRMWAt op dtype (regionFn i) (offsetFn i)
            (inputFn i) (extraFn.map fun f => f i) with
        | none =>
            cases h2 : s₂.atomicRMWAt op dtype (regionFn i) (offsetFn i)
                (inputFn i) (extraFn.map fun f => f i) with
            | none => exact .none
            | some p2 => rw [h1, h2] at hAt; cases hAt
        | some p1 =>
            obtain ⟨s₁', ret₁, event₁⟩ := p1
            cases h2 : s₂.atomicRMWAt op dtype (regionFn i) (offsetFn i)
                (inputFn i) (extraFn.map fun f => f i) with
            | none => rw [h1, h2] at hAt; cases hAt
            | some p2 =>
                obtain ⟨s₂', ret₂, event₂⟩ := p2
                rw [h1, h2] at hAt
                cases hAt with
                | some hrel =>
                    obtain ⟨hag', hret⟩ := hrel
                    injection hret with hret₁ hret₂
                    subst hret₁
                    exact foldAtomicRMWIndices_agreeOn op regionFn offsetFn
                      inputFn extraFn active P rest s₁' s₂' _ hag'
                      (fun j hj => hwin j (List.mem_cons_of_mem i hj))
      · rw [if_neg hact, if_neg hact]
        exact foldAtomicRMWIndices_agreeOn op regionFn offsetFn inputFn
          extraFn active P rest s₁ s₂ retFn hag
          (fun j hj => hwin j (List.mem_cons_of_mem i hj))

/-- The RMW address-form dispatch from agreeing states. -/
theorem rmwDispatch_agreeOn (bounds : RegionBounds) (op : RMWOp)
    {dtype : TileDType} {shape : TileShape} (mem : MemAccess dtype shape)
    (inputs : Tile dtype shape)
    (extraFn : Option (TileIndex shape → TileCarrier dtype))
    (active : TileIndex shape → Bool) (s₁ s₂ : BlockState)
    (hag : s₁.AgreeOn bounds.Window s₂) (hmem : mem.SafeAt bounds s₁)
    (haddr : mem.ActiveAddressSafe bounds s₁ (fun i => active i = true)) :
    OptionRel (fun p₁ p₂ : BlockState × Tile dtype shape =>
        p₁.1.AgreeOn bounds.Window p₂.1 ∧ p₁.2 = p₂.2)
      (rmwDispatch op mem inputs extraFn active s₁)
      (rmwDispatch op mem inputs extraFn active s₂) := by
  cases mem with
  | region r off =>
      simp only [MemAccess.SafeAt] at hmem
      simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
        at haddr
      have IHoff := evalOp_agreeOn bounds off s₁ s₂ hag hmem
      simp only [rmwDispatch, IHoff]
      cases hoffs : evalOp off s₂ with
      | none => exact .none
      | some offs =>
          exact foldAtomicRMWIndices_agreeOn op _ _ _ _ _ _
            (TileShape.allIndices shape) s₁ s₂ _ hag
            (fun i _ hact => haddr offs (IHoff.trans hoffs) i hact)
  | ptr p =>
      simp only [MemAccess.SafeAt] at hmem
      simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
        Op.PointerAddressesSafeOn] at haddr
      have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmem
      simp only [rmwDispatch, IHp]
      cases hps : evalOp p s₂ with
      | none => exact .none
      | some ps =>
          exact foldAtomicRMWIndices_agreeOn op _ _ _ _ _ _
            (TileShape.allIndices shape) s₁ s₂ _ hag
            (fun i _ hact => haddr.2 ps (IHp.trans hps) i hact)
  | blockPtr p bc =>
      simp only [MemAccess.SafeAt] at hmem
      simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
        at haddr
      have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmem
      simp only [rmwDispatch, IHp]
      cases hps : evalOp p s₂ with
      | none => exact .none
      | some ps =>
          refine foldAtomicRMWIndices_agreeOn op _ _ _ _ _ _
            (TileShape.allIndices shape) s₁ s₂ _ hag
            (fun i _ hact => ?_)
          simp only [Bool.and_eq_true] at hact
          exact haddr.2 ps (IHp.trans hps) i hact.1 hact.2

/-- The raw atomicRMW step from agreeing states. -/
theorem stepAtomicRMWRaw_agreeOn (bounds : RegionBounds) (op : RMWOp)
    {dtype : TileDType} {shape : TileShape} (mem : MemAccess dtype shape)
    (input : Op dtype shape) (extraInput : Option (Op dtype shape))
    (mask : MaskOpt dtype shape) (dest : Option RegName)
    (s₁ s₂ : BlockState) (hag : s₁.AgreeOn bounds.Window s₂)
    (hmem : mem.SafeAt bounds s₁) (hin : input.SafeAt bounds s₁)
    (hextra : extraInput.elim True (·.SafeAt bounds s₁))
    (hmask : mask.SafeAt bounds s₁)
    (haddr : mem.ActiveAddressSafe bounds s₁ (mask.Active s₁)) :
    OptionRel (BlockState.AgreeOn bounds.Window)
      (stepAtomicRMWRaw op dtype shape mem input extraInput mask dest s₁)
      (stepAtomicRMWRaw op dtype shape mem input extraInput mask dest s₂) := by
  have IHin := evalOp_agreeOn bounds input s₁ s₂ hag hin
  have hex := rmwExtras_agreeOn bounds extraInput s₁ s₂ hag hextra
  have hac := rmwActive_agreeOn bounds mask s₁ s₂ hag hmask
  simp only [stepAtomicRMWRaw, IHin, hex, hac]
  cases hinv : evalOp input s₂ with
  | none => exact .none
  | some inputs =>
      cases hexv : rmwExtras extraInput s₂ with
      | none => exact .none
      | some extraFn =>
          cases hacv : rmwActive mask s₂ with
          | none => exact .none
          | some active =>
              have hdisp := rmwDispatch_agreeOn bounds op mem inputs extraFn
                active s₁ s₂ hag hmem
                (haddr.mono fun i hi =>
                  rmwActive_active (hac.trans hacv) hi)
              show OptionRel (BlockState.AgreeOn bounds.Window)
                ((rmwDispatch op mem inputs extraFn active s₁).bind
                  fun p => rmwCommit dest p.1 p.2)
                ((rmwDispatch op mem inputs extraFn active s₂).bind
                  fun p => rmwCommit dest p.1 p.2)
              cases h1 : rmwDispatch op mem inputs extraFn active s₁ with
              | none =>
                  cases h2 : rmwDispatch op mem inputs extraFn active s₂ with
                  | none => exact .none
                  | some p2 => rw [h1, h2] at hdisp; cases hdisp
              | some p1 =>
                  obtain ⟨s₁', returns₁⟩ := p1
                  cases h2 : rmwDispatch op mem inputs extraFn active s₂ with
                  | none => rw [h1, h2] at hdisp; cases hdisp
                  | some p2 =>
                      obtain ⟨s₂', returns₂⟩ := p2
                      rw [h1, h2] at hdisp
                      cases hdisp with
                      | some hrel =>
                          obtain ⟨hag', hret⟩ := hrel
                          replace hret : returns₁ = returns₂ := hret
                          subst hret
                          cases dest with
                          | none => exact .some hag'
                          | some name =>
                              exact .some (hag'.setReg name dtype shape
                                returns₁)

/-- The atomicRMW statement step from agreeing states (the raw step plus
the `pids` restore, which both sides perform with the equal original
maps). -/
theorem stepAtomicRMW_agreeOn (bounds : RegionBounds) (op : RMWOp)
    {dtype : TileDType} {shape : TileShape} (mem : MemAccess dtype shape)
    (input : Op dtype shape) (extraInput : Option (Op dtype shape))
    (mask : MaskOpt dtype shape) (dest : Option RegName)
    (s₁ s₂ : BlockState) (hag : s₁.AgreeOn bounds.Window s₂)
    (hmem : mem.SafeAt bounds s₁) (hin : input.SafeAt bounds s₁)
    (hextra : extraInput.elim True (·.SafeAt bounds s₁))
    (hmask : mask.SafeAt bounds s₁)
    (haddr : mem.ActiveAddressSafe bounds s₁ (mask.Active s₁)) :
    OptionRel (BlockState.AgreeOn bounds.Window)
      (stepAtomicRMW op dtype shape mem input extraInput mask dest s₁)
      (stepAtomicRMW op dtype shape mem input extraInput mask dest s₂) := by
  have hraw := stepAtomicRMWRaw_agreeOn bounds op mem input extraInput mask
    dest s₁ s₂ hag hmem hin hextra hmask haddr
  unfold stepAtomicRMW
  cases h1 : stepAtomicRMWRaw op dtype shape mem input extraInput mask
      dest s₁ with
  | none =>
      cases h2 : stepAtomicRMWRaw op dtype shape mem input extraInput mask
          dest s₂ with
      | none => exact .none
      | some t2 => rw [h1, h2] at hraw; cases hraw
  | some t1 =>
      cases h2 : stepAtomicRMWRaw op dtype shape mem input extraInput mask
          dest s₂ with
      | none => rw [h1, h2] at hraw; cases hraw
      | some t2 =>
          rw [h1, h2] at hraw
          cases hraw with
          | some hrel =>
              exact .some (hrel.withPids hag.pids)


/-! ## The statement walk -/

/-- Cons-step of `stepStmts` when the head statement gets stuck. -/
theorem stepStmts.cons_none {st : Stmt} {rest : List Stmt} {s : BlockState}
    (h : stepStmt st s = none) : stepStmts (st :: rest) s = none := by
  conv_lhs => unfold stepStmts
  rw [h]

set_option maxHeartbeats 1600000 in
mutual

/-- **Statement-level locality**: a statement that is trace-safe from `s₁`
steps from agreeing states to agreeing states (or gets stuck in both). -/
theorem stepStmt_agreeOn (bounds : RegionBounds) :
    ∀ (st : Stmt) (s₁ s₂ : BlockState),
      st.TraceSafe bounds s₁ → s₁.AgreeOn bounds.Window s₂ →
      OptionRel (BlockState.AgreeOn bounds.Window)
        (stepStmt st s₁) (stepStmt st s₂)
  | .assign d sh n e, s₁, s₂, hms, hag => by
      simp only [Stmt.TraceSafe] at hms
      have IH := evalOp_agreeOn bounds e s₁ s₂ hag hms
      simp only [stepStmt, IH]
      cases hv : evalOp e s₂ with
      | none => exact .none
      | some v => exact .some (hag.setReg n d sh v)
  | .store d sh mem val mask, s₁, s₂, hms, hag => by
      simp only [Stmt.TraceSafe] at hms
      rcases mask with _ | m | ⟨m, o⟩ <;> rcases mem with ⟨r, off⟩ | p | ⟨p, bc⟩
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHoff := evalOp_agreeOn bounds off s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHoff]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hoffs : evalOp off s₂ with
            | none => exact .none
            | some offs =>
                exact .some (foldl_writeMemTyped_agreeOn d
                  (fun _ => Region.cast r) (fun i => offs.data i)
                  (fun i => values.data i) (fun _ => true)
                  bounds.Window (TileShape.allIndices sh) s₁ s₂ hag)
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hps : evalOp p s₂ with
            | none => exact .none
            | some ps =>
                exact .some (foldl_writeMemTyped_agreeOn d
                  (fun i => (ps.data i).1) (fun i => (ps.data i).2)
                  (fun i => values.data i) (fun _ => true)
                  bounds.Window (TileShape.allIndices sh) s₁ s₂ hag)
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hps : evalOp p s₂ with
            | none => exact .none
            | some ps =>
                exact .some (foldl_writeMemTyped_agreeOn d
                  (fun i => (ps.data i).region) (fun i => (ps.data i).address (TileShape.indexToList sh i))
                  (fun i => values.data i) (fun i => true && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                  bounds.Window (TileShape.allIndices sh) s₁ s₂ hag)
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask
        have IHoff := evalOp_agreeOn bounds off s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHoff]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hoffs : evalOp off s₂ with
                | none => exact .none
                | some offs =>
                    exact .some (foldl_writeMemTyped_agreeOn d
                      (fun _ => Region.cast r) (fun i => offs.data i)
                      (fun i => values.data i) (fun i => ms.data i)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag)
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hps : evalOp p s₂ with
                | none => exact .none
                | some ps =>
                    exact .some (foldl_writeMemTyped_agreeOn d
                      (fun i => (ps.data i).1) (fun i => (ps.data i).2)
                      (fun i => values.data i) (fun i => ms.data i)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag)
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hps : evalOp p s₂ with
                | none => exact .none
                | some ps =>
                    exact .some (foldl_writeMemTyped_agreeOn d
                      (fun i => (ps.data i).region) (fun i => (ps.data i).address (TileShape.indexToList sh i))
                      (fun i => values.data i) (fun i => ms.data i && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag)
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask.1
        have IHoff := evalOp_agreeOn bounds off s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHoff]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hoffs : evalOp off s₂ with
                | none => exact .none
                | some offs =>
                    exact .some (foldl_writeMemTyped_agreeOn d
                      (fun _ => Region.cast r) (fun i => offs.data i)
                      (fun i => values.data i) (fun i => ms.data i)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag)
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask.1
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hps : evalOp p s₂ with
                | none => exact .none
                | some ps =>
                    exact .some (foldl_writeMemTyped_agreeOn d
                      (fun i => (ps.data i).1) (fun i => (ps.data i).2)
                      (fun i => values.data i) (fun i => ms.data i)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag)
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask.1
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hps : evalOp p s₂ with
                | none => exact .none
                | some ps =>
                    exact .some (foldl_writeMemTyped_agreeOn d
                      (fun i => (ps.data i).region) (fun i => (ps.data i).address (TileShape.indexToList sh i))
                      (fun i => values.data i) (fun i => ms.data i && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag)
  | .atomicAdd hnum sh mem val mask, s₁, s₂, hms, hag => by
      simp only [Stmt.TraceSafe] at hms
      rcases mask with _ | m | ⟨m, o⟩ <;> rcases mem with ⟨r, off⟩ | p | ⟨p, bc⟩
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MemAccess.ActiveAddressSafe,
          memAccessActiveAddressSafe] at haddr
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHoff := evalOp_agreeOn bounds off s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHoff]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hoffs : evalOp off s₂ with
            | none => exact .none
            | some offs =>
                refine .some (foldl_atomicAdd_agreeOn hnum
                  (fun _ => Region.cast r) (fun i => offs.data i)
                  (fun i => values.data i) (fun _ => true)
                  bounds.Window (TileShape.allIndices sh) s₁ s₂ hag
                  (fun i _ hact => haddr offs (IHoff.trans hoffs) i trivial))
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MemAccess.ActiveAddressSafe,
          memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hps : evalOp p s₂ with
            | none => exact .none
            | some ps =>
                refine .some (foldl_atomicAdd_agreeOn hnum
                  (fun i => (ps.data i).1) (fun i => (ps.data i).2)
                  (fun i => values.data i) (fun _ => true)
                  bounds.Window (TileShape.allIndices sh) s₁ s₂ hag
                  (fun i _ hact => haddr.2 ps (IHp.trans hps) i trivial))
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MemAccess.ActiveAddressSafe,
          memAccessActiveAddressSafe] at haddr
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hps : evalOp p s₂ with
            | none => exact .none
            | some ps =>
                refine .some (foldl_atomicAdd_agreeOn hnum
                  (fun i => (ps.data i).region) (fun i => (ps.data i).address (TileShape.indexToList sh i))
                  (fun i => values.data i) (fun i => true && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                  bounds.Window (TileShape.allIndices sh) s₁ s₂ hag
                  (fun i _ hact => by
                    simp only [Bool.and_eq_true] at hact
                    exact haddr.2 ps (IHp.trans hps) i trivial hact.2))
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        simp only [MemAccess.ActiveAddressSafe,
          memAccessActiveAddressSafe] at haddr
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask
        have IHoff := evalOp_agreeOn bounds off s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHoff]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hoffs : evalOp off s₂ with
                | none => exact .none
                | some offs =>
                    refine .some (foldl_atomicAdd_agreeOn hnum
                      (fun _ => Region.cast r) (fun i => offs.data i)
                      (fun i => values.data i) (fun i => ms.data i)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag
                      (fun i _ hact => haddr offs (IHoff.trans hoffs) i ⟨ms, IHm.trans hm, hact⟩))
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        simp only [MemAccess.ActiveAddressSafe,
          memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hps : evalOp p s₂ with
                | none => exact .none
                | some ps =>
                    refine .some (foldl_atomicAdd_agreeOn hnum
                      (fun i => (ps.data i).1) (fun i => (ps.data i).2)
                      (fun i => values.data i) (fun i => ms.data i)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag
                      (fun i _ hact => haddr.2 ps (IHp.trans hps) i ⟨ms, IHm.trans hm, hact⟩))
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        simp only [MemAccess.ActiveAddressSafe,
          memAccessActiveAddressSafe] at haddr
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hps : evalOp p s₂ with
                | none => exact .none
                | some ps =>
                    refine .some (foldl_atomicAdd_agreeOn hnum
                      (fun i => (ps.data i).region) (fun i => (ps.data i).address (TileShape.indexToList sh i))
                      (fun i => values.data i) (fun i => ms.data i && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag
                      (fun i _ hact => by
                        simp only [Bool.and_eq_true] at hact
                        exact haddr.2 ps (IHp.trans hps) i ⟨ms, IHm.trans hm, hact.1⟩ hact.2))
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        simp only [MemAccess.ActiveAddressSafe,
          memAccessActiveAddressSafe] at haddr
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask.1
        have IHoff := evalOp_agreeOn bounds off s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHoff]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hoffs : evalOp off s₂ with
                | none => exact .none
                | some offs =>
                    refine .some (foldl_atomicAdd_agreeOn hnum
                      (fun _ => Region.cast r) (fun i => offs.data i)
                      (fun i => values.data i) (fun i => ms.data i)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag
                      (fun i _ hact => haddr offs (IHoff.trans hoffs) i ⟨ms, IHm.trans hm, hact⟩))
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        simp only [MemAccess.ActiveAddressSafe,
          memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask.1
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hps : evalOp p s₂ with
                | none => exact .none
                | some ps =>
                    refine .some (foldl_atomicAdd_agreeOn hnum
                      (fun i => (ps.data i).1) (fun i => (ps.data i).2)
                      (fun i => values.data i) (fun i => ms.data i)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag
                      (fun i _ hact => haddr.2 ps (IHp.trans hps) i ⟨ms, IHm.trans hm, hact⟩))
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        simp only [MemAccess.SafeAt] at hmsmem
        simp only [MaskOpt.SafeAt] at hmsmask
        simp only [MemAccess.ActiveAddressSafe,
          memAccessActiveAddressSafe] at haddr
        have IHval := evalOp_agreeOn bounds val s₁ s₂ hag hmsval
        have IHm := evalOp_agreeOn bounds m s₁ s₂ hag hmsmask.1
        have IHp := evalOp_agreeOn bounds p s₁ s₂ hag hmsmem
        simp only [stepStmt, IHval, IHm, IHp]
        cases hv : evalOp val s₂ with
        | none => exact .none
        | some values =>
            cases hm : evalOp m s₂ with
            | none => exact .none
            | some ms =>
                cases hps : evalOp p s₂ with
                | none => exact .none
                | some ps =>
                    refine .some (foldl_atomicAdd_agreeOn hnum
                      (fun i => (ps.data i).region) (fun i => (ps.data i).address (TileShape.indexToList sh i))
                      (fun i => values.data i) (fun i => ms.data i && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                      bounds.Window (TileShape.allIndices sh) s₁ s₂ hag
                      (fun i _ hact => by
                        simp only [Bool.and_eq_true] at hact
                        exact haddr.2 ps (IHp.trans hps) i ⟨ms, IHm.trans hm, hact.1⟩ hact.2))
  | .atomicRMW op d sh mem input extra mask dest, s₁, s₂, hms, hag => by
      simp only [Stmt.TraceSafe] at hms
      obtain ⟨hmem, hin, hextra, hmask, haddr⟩ := hms
      simp only [stepStmt]
      exact stepAtomicRMW_agreeOn bounds op mem input extra mask dest s₁ s₂
        hag hmem hin hextra hmask haddr
  | .forLoop idx n body, s₁, s₂, hms, hag => by
      simp only [Stmt.TraceSafe] at hms
      simp only [stepForLoopAux.forLoop_unfold]
      exact stepForLoopAux_agreeOn bounds idx 0 n body s₁ s₂ hms hag
  | .forRange idx start stop step body, s₁, s₂, hms, hag => by
      simp only [Stmt.TraceSafe] at hms
      simp only [stepForRangeAux.forRange_unfold]
      exact stepForRangeAux_agreeOn bounds idx start stop step body s₁ s₂
        hms hag
  | .forRangeDyn idx start stop step body, s₁, s₂, hms, hag => by
      simp only [Stmt.TraceSafe] at hms
      obtain ⟨hmstart, hmstop, hmstep, hnext⟩ := hms
      have IHstart := evalOp_agreeOn bounds start s₁ s₂ hag hmstart
      have IHstop := evalOp_agreeOn bounds stop s₁ s₂ hag hmstop
      have IHstep := evalOp_agreeOn bounds step s₁ s₂ hag hmstep
      simp only [stepForRangeAux.forRangeDyn_unfold, IHstart, IHstop, IHstep]
      cases hstart : evalOp start s₂ with
      | none => exact .none
      | some a =>
          cases hstop : evalOp stop s₂ with
          | none => exact .none
          | some b =>
              cases hstep : evalOp step s₂ with
              | none => exact .none
              | some c =>
                  rw [IHstart.trans hstart, IHstop.trans hstop,
                    IHstep.trans hstep] at hnext
                  replace hnext : Stmt.forRangeTraceSafe bounds idx
                      (a.data PUnit.unit) (b.data PUnit.unit)
                      (c.data PUnit.unit) body s₁ := hnext
                  exact stepForRangeAux_agreeOn bounds idx _ _ _ body s₁ s₂
                    hnext hag
  | .ifThen c body, s₁, s₂, hms, hag => by
      simp only [Stmt.TraceSafe] at hms
      have IHc := evalOp_agreeOn bounds c s₁ s₂ hag hms.1
      have hbody := hms.2
      simp only [stepStmt, IHc]
      cases hc : evalOp c s₂ with
      | none => exact .none
      | some vc =>
          rw [IHc.trans hc] at hbody
          replace hbody : (if vc.data PUnit.unit = true
              then Stmt.TraceSafeList bounds body s₁ else True) := hbody
          show OptionRel (BlockState.AgreeOn bounds.Window)
            (if vc.data PUnit.unit = true then stepStmts body s₁ else some s₁)
            (if vc.data PUnit.unit = true then stepStmts body s₂ else some s₂)
          by_cases hb : vc.data PUnit.unit = true
          · rw [if_pos hb, if_pos hb]
            rw [if_pos hb] at hbody
            exact stepStmts_agreeOn bounds body s₁ s₂ hbody hag
          · rw [if_neg hb, if_neg hb]
            exact .some hag
  | .ifThenElse c tb eb, s₁, s₂, hms, hag => by
      simp only [Stmt.TraceSafe] at hms
      have IHc := evalOp_agreeOn bounds c s₁ s₂ hag hms.1
      have hbody := hms.2
      simp only [stepStmt, IHc]
      cases hc : evalOp c s₂ with
      | none => exact .none
      | some vc =>
          rw [IHc.trans hc] at hbody
          replace hbody : (if vc.data PUnit.unit = true
              then Stmt.TraceSafeList bounds tb s₁
              else Stmt.TraceSafeList bounds eb s₁) := hbody
          show OptionRel (BlockState.AgreeOn bounds.Window)
            (if vc.data PUnit.unit = true then stepStmts tb s₁
              else stepStmts eb s₁)
            (if vc.data PUnit.unit = true then stepStmts tb s₂
              else stepStmts eb s₂)
          by_cases hb : vc.data PUnit.unit = true
          · rw [if_pos hb, if_pos hb]
            rw [if_pos hb] at hbody
            exact stepStmts_agreeOn bounds tb s₁ s₂ hbody hag
          · rw [if_neg hb, if_neg hb]
            rw [if_neg hb] at hbody
            exact stepStmts_agreeOn bounds eb s₁ s₂ hbody hag
  termination_by st _ _ _ _ => (sizeOf st, 0)
  decreasing_by
    all_goals simp_wf
    all_goals (try omega)
    all_goals (have : 0 < sizeOf idx := by cases idx; simp)
    all_goals omega

/-- List version of the statement walk. -/
theorem stepStmts_agreeOn (bounds : RegionBounds) :
    ∀ (l : List Stmt) (s₁ s₂ : BlockState),
      Stmt.TraceSafeList bounds l s₁ → s₁.AgreeOn bounds.Window s₂ →
      OptionRel (BlockState.AgreeOn bounds.Window)
        (stepStmts l s₁) (stepStmts l s₂)
  | [], s₁, s₂, _, hag => by
      simp only [stepStmts.nil]
      exact .some hag
  | st :: rest, s₁, s₂, hms, hag => by
      simp only [Stmt.TraceSafeList] at hms
      have hstep := stepStmt_agreeOn bounds st s₁ s₂ hms.1 hag
      have hrest := hms.2
      cases h1 : stepStmt st s₁ with
      | none =>
          cases h2 : stepStmt st s₂ with
          | none =>
              rw [stepStmts.cons_none h1, stepStmts.cons_none h2]
              exact .none
          | some t2 => rw [h1, h2] at hstep; cases hstep
      | some t1 =>
          cases h2 : stepStmt st s₂ with
          | none => rw [h1, h2] at hstep; cases hstep
          | some t2 =>
              rw [h1, h2] at hstep
              cases hstep with
              | some hrel =>
                  rw [stepStmts.cons_some h1, stepStmts.cons_some h2]
                  rw [h1] at hrest
                  replace hrest : Stmt.TraceSafeList bounds rest t1 := hrest
                  exact stepStmts_agreeOn bounds rest t1 t2 hrest hrel
  termination_by l _ _ _ _ => (sizeOf l, 0)
  decreasing_by all_goals (simp_wf; omega)

/-- `forLoop` auxiliary walk. -/
theorem stepForLoopAux_agreeOn (bounds : RegionBounds) :
    ∀ (idx : RegName) (start n : Nat) (body : List Stmt)
      (s₁ s₂ : BlockState),
      Stmt.forLoopTraceSafe bounds idx start n body s₁ →
      s₁.AgreeOn bounds.Window s₂ →
      OptionRel (BlockState.AgreeOn bounds.Window)
        (stepForLoopAux idx start n body s₁)
        (stepForLoopAux idx start n body s₂)
  | idx, start, n, body, s₁, s₂, hms, hag => by
      rw [stepForLoopAux, stepForLoopAux]
      rw [Stmt.forLoopTraceSafe] at hms
      by_cases hlt : start < n
      · rw [if_pos hlt, if_pos hlt]
        rw [if_pos hlt] at hms
        obtain ⟨hbody, hnext⟩ := hms
        have hstep := stepStmts_agreeOn bounds body _ _ hbody
          (hag.setReg idx .nat [] (Tile.scalar start))
        cases h1 : stepStmts body (s₁.setReg idx .nat [] (Tile.scalar start)) with
        | none =>
            cases h2 : stepStmts body
                (s₂.setReg idx .nat [] (Tile.scalar start)) with
            | none => exact .none
            | some t2 => rw [h1, h2] at hstep; cases hstep
        | some t1 =>
            cases h2 : stepStmts body
                (s₂.setReg idx .nat [] (Tile.scalar start)) with
            | none => rw [h1, h2] at hstep; cases hstep
            | some t2 =>
                rw [h1, h2] at hstep
                cases hstep with
                | some hrel =>
                    rw [h1] at hnext
                    replace hnext : Stmt.forLoopTraceSafe bounds idx
                        (start + 1) n body t1 := hnext
                    exact stepForLoopAux_agreeOn bounds idx (start + 1) n
                      body t1 t2 hnext hrel
      · rw [if_neg hlt, if_neg hlt]
        exact .some hag
  termination_by idx start n body _ _ _ _ => (sizeOf body + 1, n - start)
  decreasing_by all_goals omega

/-- `forRange` auxiliary walk. -/
theorem stepForRangeAux_agreeOn (bounds : RegionBounds) :
    ∀ (idx : RegName) (cur stop step : Nat) (body : List Stmt)
      (s₁ s₂ : BlockState),
      Stmt.forRangeTraceSafe bounds idx cur stop step body s₁ →
      s₁.AgreeOn bounds.Window s₂ →
      OptionRel (BlockState.AgreeOn bounds.Window)
        (stepForRangeAux idx cur stop step body s₁)
        (stepForRangeAux idx cur stop step body s₂)
  | idx, cur, stop, step, body, s₁, s₂, hms, hag => by
      rw [stepForRangeAux, stepForRangeAux]
      rw [Stmt.forRangeTraceSafe] at hms
      by_cases hz : step = 0
      · rw [if_pos hz, if_pos hz]
        exact .some hag
      · rw [if_neg hz, if_neg hz]
        rw [if_neg hz] at hms
        by_cases hlt : cur < stop
        · rw [if_pos hlt, if_pos hlt]
          rw [if_pos hlt] at hms
          obtain ⟨hbody, hnext⟩ := hms
          have hstep := stepStmts_agreeOn bounds body _ _ hbody
            (hag.setReg idx .nat [] (Tile.scalar cur))
          cases h1 : stepStmts body
              (s₁.setReg idx .nat [] (Tile.scalar cur)) with
          | none =>
              cases h2 : stepStmts body
                  (s₂.setReg idx .nat [] (Tile.scalar cur)) with
              | none => exact .none
              | some t2 => rw [h1, h2] at hstep; cases hstep
          | some t1 =>
              cases h2 : stepStmts body
                  (s₂.setReg idx .nat [] (Tile.scalar cur)) with
              | none => rw [h1, h2] at hstep; cases hstep
              | some t2 =>
                  rw [h1, h2] at hstep
                  cases hstep with
                  | some hrel =>
                      rw [h1] at hnext
                      replace hnext : Stmt.forRangeTraceSafe bounds idx
                          (cur + step) stop step body t1 := hnext
                      exact stepForRangeAux_agreeOn bounds idx (cur + step)
                        stop step body t1 t2 hnext hrel
        · rw [if_neg hlt, if_neg hlt]
          exact .some hag
  termination_by idx cur stop step body _ _ _ _ =>
    (sizeOf body + 1, stop - cur)
  decreasing_by all_goals omega

end

/-- **Execution locality**: a kernel that is trace-safe from `s₁` (all its
memory traffic stays inside the window of `bounds`) computes, from any state
`s₂` agreeing with `s₁` inside that window, the same in-window results —
both runs get stuck together, or both succeed with final states again
agreeing inside the window. Junk outside the window cannot change the
answer. -/
theorem exec_agreeOn (bounds : RegionBounds) (k : Kernel)
    (s₁ s₂ : BlockState) (hms : k.TraceSafe bounds s₁)
    (hag : s₁.AgreeOn bounds.Window s₂) :
    OptionRel (BlockState.AgreeOn bounds.Window) (exec k s₁) (exec k s₂) :=
  stepStmts_agreeOn bounds k.body s₁ s₂ hms hag

end VeriTile.Triton
