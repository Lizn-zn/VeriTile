/-
VeriTile.Triton.Memory.FlattenTolerant

**Dead-write-tolerant flat bridge** (Phase 2 of the ptrSub bridge).

`FlattenR` bridges `execR R` runs whose every statement is trace-safe
(`Stmt.TraceSafeR`), which at `.ptrSub` demands per-state no-underflow. Loop
kernels that walk pointers backwards (`p_x -= DK` at the end of every
iteration, e.g. `bwd_decay_global_cumsum`) violate that exactly once: the
trailing self-decrements of the *final* iteration underflow — the truncated
`Nat` subtraction lands the region-model pointer at a different offset than
its flat image would. Those writes are **dead**: the registers they produce
are never read again (the loop is the last statement of the kernel), so the
divergence is confined to register *values*. This file weakens the bridge
conclusion from state equality to **observational agreement** (`ObsAgree`:
same `mem`/`pids`/`numPids`/`undef`) and threads register *domain* agreement
(`RegsDefEq`) to keep both runs succeeding in lockstep.

Layers:
* `BlockState.ObsAgree`, `RegsDefEq` — the tolerant agreement vocabulary,
  with `readMem`/`readMemAs`/`readMemValue`/`readMemTyped` transports.
* `Stmt.PtrDecStmt` / `StmtList.PtrDecTail` — the narrow dead-tail shape:
  a register self-decrement `n := ptrSub (ref n) (constNat k)`.
* Lemma A `stepStmtsR_ptrDecTail_frame` — a PtrDec tail is a register-only
  frame (mem/pids/numPids/undef untouched).
* Lemma B `FlatAlloc.stepStmtsR_ptrDecTail_tolerant` — the flat run of a
  PtrDec tail succeeds from any `RegsDefEq`/`ObsAgree` sibling state and
  re-establishes both (no safety hypothesis: `evalOpR` of
  ptrSub-of-ref-of-constNat is total once the ref's slot is defined).
* Lemma C `FlatAlloc.stepForRangeAuxR_flatten_deadPtrTail` — the tolerant
  loop bridge: strict per-statement commutation until the final iteration,
  then strict core + tolerant tail; invariant-style hypotheses mirroring
  `Stmt.forRangeTraceSafeR_inv`.
* Lemma D `FlatAlloc.stepStmtR_forRangeDyn_flatten_deadPtrTail` and
  `FlatAlloc.execR_flatten_deadPtrTail` — the `forRangeDyn` wrapper and the
  top-level bridge for kernels of shape `pre ++ [forRangeDyn … (core ++
  tail)]` (loop last), concluding `∃ s', execR … = some s' ∧
  s'.ObsAgree (flattenState s1)`.
-/

import VeriTile.Triton.Memory.FlattenR

namespace VeriTile.Triton

/-! ## Observational agreement -/

namespace BlockState

/-- Two states agree on everything a kernel *observation* can see: memory,
program ids, grid dimensions, and the undef backing store. Register values
may differ (the tolerant bridge gives up exactly the dead trailing pointer
registers). -/
def ObsAgree (s t : BlockState) : Prop :=
  s.mem = t.mem ∧ s.pids = t.pids ∧ s.numPids = t.numPids ∧ s.undef = t.undef

theorem ObsAgree.refl (s : BlockState) : s.ObsAgree s := ⟨rfl, rfl, rfl, rfl⟩

theorem ObsAgree.mem {s t : BlockState} (h : s.ObsAgree t) : s.mem = t.mem :=
  h.1

theorem ObsAgree.pids {s t : BlockState} (h : s.ObsAgree t) : s.pids = t.pids :=
  h.2.1

theorem ObsAgree.numPids {s t : BlockState} (h : s.ObsAgree t) :
    s.numPids = t.numPids := h.2.2.1

theorem ObsAgree.undef {s t : BlockState} (h : s.ObsAgree t) :
    s.undef = t.undef := h.2.2.2

/-- `readMem` is a function of `mem` only, so it transports along
`ObsAgree`. -/
theorem ObsAgree.readMem_eq {s t : BlockState} (h : s.ObsAgree t)
    (r : RegionName) (o : Nat) : s.readMem r o = t.readMem r o := by
  unfold BlockState.readMem
  rw [h.1]

/-- `readMemAs` is a function of `mem` only, so it transports along
`ObsAgree`. -/
theorem ObsAgree.readMemAs_eq {s t : BlockState} (h : s.ObsAgree t)
    (dtype : FloatDType) (r : RegionName) (o : Nat) :
    s.readMemAs dtype r o = t.readMemAs dtype r o := by
  unfold BlockState.readMemAs
  rw [h.1]

/-- `readMemTyped` is a function of `mem` only, so it transports along
`ObsAgree`. -/
theorem ObsAgree.readMemTyped_eq {s t : BlockState} (h : s.ObsAgree t)
    (dtype : TileDType) (r : RegionName) (o : Nat) :
    s.readMemTyped dtype r o = t.readMemTyped dtype r o := by
  unfold BlockState.readMemTyped
  rw [h.1]

/-- `readMemValue` is a function of `mem` only, so it transports along
`ObsAgree`. -/
theorem ObsAgree.readMemValue_eq {s t : BlockState} (h : s.ObsAgree t)
    (dtype : TileDType) (r : RegionName) (o : Nat) :
    s.readMemValue dtype r o = t.readMemValue dtype r o := by
  cases dtype <;>
    simp only [BlockState.readMemValue, h.readMemAs_eq, h.readMemTyped_eq]

end BlockState

/-! ## Register-domain agreement -/

/-- Register **domain** agreement: the same slots are defined (values may
differ). This is what transports `evalOpR` success of a pure register
self-update between the region run and its flat sibling once their register
values have diverged. -/
def RegsDefEq (s t : BlockState) : Prop :=
  ∀ (d : TileDType) (sh : TileShape) (n : RegName),
    (s.regs d sh n).isSome = (t.regs d sh n).isSome

theorem RegsDefEq.refl (s : BlockState) : RegsDefEq s s := fun _ _ _ => rfl

/-- Writing the *same* slot on both sides (with possibly different values)
preserves domain agreement: `setReg` fills the written slot and clears the
same stale aliases on both sides. -/
theorem RegsDefEq.setReg {u s : BlockState} (h : RegsDefEq u s) (n : RegName)
    (d : TileDType) (sh : TileShape) (v w : Tile d sh) :
    RegsDefEq (u.setReg n d sh v) (s.setReg n d sh w) := by
  intro d' sh' n'
  by_cases hn : n' = n
  · by_cases hd' : d' = d
    · by_cases hs' : sh' = sh
      · subst hn; subst hd'; subst hs'
        rw [BlockState.setReg_same, BlockState.setReg_same]
        rfl
      · rw [BlockState.setReg_ne_shape u n n' d d' sh sh' v hn hd' hs',
          BlockState.setReg_ne_shape s n n' d d' sh sh' w hn hd' hs']
    · rw [BlockState.setReg_ne_dtype u n n' d d' sh sh' v hn hd',
        BlockState.setReg_ne_dtype s n n' d d' sh sh' w hn hd']
  · rw [BlockState.setReg_ne_name u n n' d d' sh sh' v hn,
      BlockState.setReg_ne_name s n n' d d' sh sh' w hn]
    exact h d' sh' n'

/-- The flat image defines exactly the source's register slots (`trRegs` is
an `Option.map`). -/
theorem FlatAlloc.regsDefEq_flattenState (A : FlatAlloc) (s : BlockState) :
    RegsDefEq (A.flattenState s) s := by
  intro d sh n
  show ((s.regs d sh n).map A.trTile).isSome = (s.regs d sh n).isSome
  cases s.regs d sh n <;> rfl

/-! ## The dead-tail shape -/

/-- The narrow self-decrement shape tolerated by the bridge: assign a `.ptr`
register the `ptrSub` of *itself* by a constant. Deliberately narrow — widen
only when a consumer needs more. -/
def Stmt.PtrDecStmt (st : Stmt) : Prop :=
  ∃ (sh : TileShape) (n : RegName) (bc : Broadcast sh [] sh) (k : Nat),
    st = Stmt.assign .ptr sh n
      (Op.ptrSub bc (Op.ref .ptr sh n) (Op.constNat k))

/-- Every statement of the list is a pointer self-decrement. -/
def StmtList.PtrDecTail (L : List Stmt) : Prop := ∀ st ∈ L, st.PtrDecStmt

/-- Pointer self-decrements are in the `R`-bridge fragment (the whole point
of `FlattenOkR`'s structural `ptrSub` clause). -/
theorem Stmt.PtrDecStmt.flattenOkR {st : Stmt} (h : st.PtrDecStmt) :
    st.FlattenOkR := by
  obtain ⟨sh, n, bc, k, rfl⟩ := h
  simp only [Stmt.FlattenOkR, Op.FlattenOkR]
  exact ⟨trivial, trivial⟩

/-- List version of `Stmt.PtrDecStmt.flattenOkR`. -/
theorem StmtList.PtrDecTail.flattenOkR :
    ∀ {L : List Stmt}, StmtList.PtrDecTail L → StmtList.FlattenOkR L
  | [], _ => by simp only [StmtList.FlattenOkR]
  | st :: rest, h => by
      simp only [StmtList.FlattenOkR]
      exact ⟨(h st List.mem_cons_self).flattenOkR,
        StmtList.PtrDecTail.flattenOkR
          (fun st' hst' => h st' (List.mem_cons_of_mem st hst'))⟩

/-- `StmtList.FlattenOkR` closes under concatenation. -/
theorem StmtList.FlattenOkR.append :
    ∀ {xs ys : List Stmt}, StmtList.FlattenOkR xs → StmtList.FlattenOkR ys →
      StmtList.FlattenOkR (xs ++ ys)
  | [], ys, _, hys => by simpa using hys
  | x :: xs, ys, hxs, hys => by
      simp only [StmtList.FlattenOkR] at hxs
      simp only [List.cons_append, StmtList.FlattenOkR]
      exact ⟨hxs.1, StmtList.FlattenOkR.append hxs.2 hys⟩

/-- Flattening is the identity on a pointer self-decrement (`flattenOp` is a
congruence at `ptrSub`/`ref`/`constNat`). -/
theorem FlatAlloc.flattenStmt_ptrDec (A : FlatAlloc) {st : Stmt}
    (h : st.PtrDecStmt) : A.flattenStmt st = st := by
  obtain ⟨sh, n, bc, k, rfl⟩ := h
  simp only [FlatAlloc.flattenStmt, FlatAlloc.flattenOp]

/-- Flattening is the identity on a PtrDec tail. -/
theorem FlatAlloc.flattenStmts_ptrDecTail (A : FlatAlloc) :
    ∀ (L : List Stmt), StmtList.PtrDecTail L → A.flattenStmts L = L
  | [], _ => by simp only [FlatAlloc.flattenStmts]
  | st :: rest, h => by
      simp only [FlatAlloc.flattenStmts]
      rw [A.flattenStmt_ptrDec (h st List.mem_cons_self),
        A.flattenStmts_ptrDecTail rest
          (fun st' hst' => h st' (List.mem_cons_of_mem st hst'))]

/-- `flattenStmts` distributes over concatenation. -/
theorem FlatAlloc.flattenStmts_append (A : FlatAlloc) :
    ∀ (xs ys : List Stmt),
      A.flattenStmts (xs ++ ys) = A.flattenStmts xs ++ A.flattenStmts ys
  | [], ys => by simp only [FlatAlloc.flattenStmts, List.nil_append]
  | x :: xs, ys => by
      simp only [List.cons_append, FlatAlloc.flattenStmts,
        A.flattenStmts_append xs ys]

/-- `evalOpR` of a pointer self-decrement is the mapped register lookup:
total once the slot is defined, regardless of underflow. -/
theorem evalOpR_ptrDec (R : RoundingModel) {sh : TileShape}
    (bc : Broadcast sh [] sh) (n : RegName) (k : Nat) (s : BlockState) :
    evalOpR R (Op.ptrSub bc (Op.ref .ptr sh n) (Op.constNat k)) s
      = (s.regs .ptr sh n).map (fun p => Tile.ptrSub bc p (Tile.scalar k)) := by
  cases h : s.regs .ptr sh n with
  | none => simp only [evalOpR, h, Option.map_none]; rfl
  | some p => simp only [evalOpR, h, Option.map_some]; rfl

/-! ## Lemma A: a PtrDec tail is a register-only frame -/

/-- **Region-tail frame**: a successful run of a PtrDec tail touches only
registers — memory, program ids, grid dims and undef all survive verbatim.
(No `RegsDefEq s1 s` conclusion: `setReg` clears stale same-name slots at
*other* dtypes/shapes, so the domain claim is not provable without a
no-stale-alias hypothesis; the tolerant bridge threads domain agreement in
Lemma B instead, where both runs write the same slot in lockstep.) -/
theorem stepStmtsR_ptrDecTail_frame (R : RoundingModel) :
    ∀ (L : List Stmt) (s s1 : BlockState), StmtList.PtrDecTail L →
      stepStmtsR R L s = some s1 →
      s1.mem = s.mem ∧ s1.pids = s.pids ∧ s1.numPids = s.numPids ∧
        s1.undef = s.undef
  | [], s, s1, _, h => by
      simp only [stepStmtsR, Option.some.injEq] at h
      subst h
      exact ⟨rfl, rfl, rfl, rfl⟩
  | st :: rest, s, s1, hpd, h => by
      obtain ⟨sh, n, bc, k, rfl⟩ := hpd st List.mem_cons_self
      simp only [stepStmtsR] at h
      cases hst : stepStmtR R
          (Stmt.assign .ptr sh n
            (Op.ptrSub bc (Op.ref .ptr sh n) (Op.constNat k))) s with
      | none => rw [hst] at h; exact absurd h (by simp)
      | some s' =>
          rw [hst] at h
          replace h : stepStmtsR R rest s' = some s1 := h
          obtain ⟨v, _, rfl⟩ := stepStmtR_assign_inv hst
          obtain ⟨hm, hp, hn2, hu2⟩ := stepStmtsR_ptrDecTail_frame R rest _ s1
            (fun st' hst' => hpd st' (List.mem_cons_of_mem _ hst')) h
          exact ⟨hm, hp, hn2, hu2⟩

/-! ## Lemma B: tolerant tail bridge -/

/-- **Tolerant tail bridge**: if the region run steps a PtrDec tail to `s1`,
then *any* state `u` that domain-agrees with `s` and observationally agrees
with `s`'s flat image also steps the tail (the flat spelling of a PtrDec
statement is itself), landing in observational agreement with `s1`'s flat
image and domain agreement with `s1`. No trace-safety hypothesis: `evalOpR`
of ptrSub-of-ref-of-constNat succeeds on any state whose ref slot is
defined, underflowing or not. -/
theorem FlatAlloc.stepStmtsR_ptrDecTail_tolerant (A : FlatAlloc)
    (R : RoundingModel) :
    ∀ (L : List Stmt) (s s1 u : BlockState), StmtList.PtrDecTail L →
      stepStmtsR R L s = some s1 → RegsDefEq u s →
      u.ObsAgree (A.flattenState s) →
      ∃ u', stepStmtsR R L u = some u' ∧
        u'.ObsAgree (A.flattenState s1) ∧ RegsDefEq u' s1
  | [], s, s1, u, _, h, hdef, hobs => by
      simp only [stepStmtsR, Option.some.injEq] at h
      subst h
      exact ⟨u, by simp only [stepStmtsR], hobs, hdef⟩
  | st :: rest, s, s1, u, hpd, h, hdef, hobs => by
      obtain ⟨sh, n, bc, k, rfl⟩ := hpd st List.mem_cons_self
      simp only [stepStmtsR] at h
      cases hst : stepStmtR R
          (Stmt.assign .ptr sh n
            (Op.ptrSub bc (Op.ref .ptr sh n) (Op.constNat k))) s with
      | none => rw [hst] at h; exact absurd h (by simp)
      | some s' =>
          rw [hst] at h
          replace h : stepStmtsR R rest s' = some s1 := h
          obtain ⟨v, hv, rfl⟩ := stepStmtR_assign_inv hst
          -- the region eval pins the source slot as defined
          rw [evalOpR_ptrDec] at hv
          cases hreg : s.regs .ptr sh n with
          | none => rw [hreg] at hv; exact absurd hv (by simp)
          | some p =>
              rw [hreg, Option.map_some] at hv
              -- the sibling slot is defined via domain agreement
              have husome : (u.regs .ptr sh n).isSome = true := by
                rw [hdef .ptr sh n, hreg]; rfl
              obtain ⟨q, hq⟩ := Option.isSome_iff_exists.mp husome
              have hstu : stepStmtR R
                  (Stmt.assign .ptr sh n
                    (Op.ptrSub bc (Op.ref .ptr sh n) (Op.constNat k))) u
                  = some (u.setReg n .ptr sh (Tile.ptrSub bc q (Tile.scalar k))) :=
                stepStmtR_assign_eq_some (by rw [evalOpR_ptrDec, hq]; rfl)
              obtain ⟨u', hu', hobs1, hdef1⟩ :=
                A.stepStmtsR_ptrDecTail_tolerant R rest
                  (s.setReg n .ptr sh v) s1
                  (u.setReg n .ptr sh (Tile.ptrSub bc q (Tile.scalar k)))
                  (fun st' hst' => hpd st' (List.mem_cons_of_mem _ hst'))
                  h (hdef.setReg n .ptr sh _ _) hobs
              refine ⟨u', ?_, hobs1, hdef1⟩
              simp only [stepStmtsR]
              rw [hstu]
              exact hu'

/-! ## Lemma C: the tolerant `forRange` auxiliary bridge -/

/-- **Tolerant loop bridge** for a body `core ++ tail` with a PtrDec tail.

Invariant-style hypotheses mirroring `Stmt.forRangeTraceSafeR_inv`: supply
`P` (indexed by the loop counter) such that every in-range iteration started
from a `P`-state with the counter set has (i) a trace-safe `core`, (ii) a
successful full-body step re-establishing `P`, and (iii) — for **non-final**
iterations only — a trace-safe `tail` from whatever state `core` reaches.
The final iteration's tail may underflow freely: the bridge switches from
strict per-statement commutation to the tolerant tail bridge there, and the
loop (hence the divergence window) ends immediately after.

Conclusion: the flat run of the flattened body reaches a state that
observationally agrees with the flat image of the region result, with
register domains agreeing. -/
theorem FlatAlloc.stepForRangeAuxR_flatten_deadPtrTail (A : FlatAlloc)
    (hd : A.Disjoint) (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (R : RoundingModel) (idx : RegName) (stop step : Nat)
    (core tail : List Stmt) (htail : StmtList.PtrDecTail tail)
    (hokCore : StmtList.FlattenOkR core)
    (P : Nat → BlockState → Prop)
    (hstep : ∀ c s, c < stop → P c s →
      Stmt.TraceSafeListR R A.extent core
        (s.setReg idx .nat [] (Tile.scalar c)) ∧
      (∃ s', stepStmtsR R (core ++ tail)
          (s.setReg idx .nat [] (Tile.scalar c)) = some s' ∧
        P (c + step) s') ∧
      (c + step < stop → ∀ sMid,
        stepStmtsR R core (s.setReg idx .nat [] (Tile.scalar c)) = some sMid →
        Stmt.TraceSafeListR R A.extent tail sMid)) :
    ∀ (cur : Nat) (s : BlockState), P cur s → s.undef = (fun _ _ => 0) →
      ∀ s_fin, stepForRangeAuxR R idx cur stop step (core ++ tail) s
          = some s_fin →
      ∃ t, stepForRangeAuxR R idx cur stop step
            (A.flattenStmts (core ++ tail)) (A.flattenState s) = some t ∧
        t.ObsAgree (A.flattenState s_fin) ∧ RegsDefEq t s_fin
  | cur, s, hP, hu, s_fin, hrun => by
      rw [stepForRangeAuxR] at hrun
      by_cases hz : step = 0
      · rw [if_pos hz, Option.some.injEq] at hrun
        subst hrun
        exact ⟨A.flattenState s, by rw [stepForRangeAuxR, if_pos hz],
          BlockState.ObsAgree.refl _, A.regsDefEq_flattenState s⟩
      · rw [if_neg hz] at hrun
        by_cases hlt : cur < stop
        · rw [if_pos hlt] at hrun
          obtain ⟨hsafeCore, ⟨s'w, hs'w, hPnext⟩, htailSafe⟩ :=
            hstep cur s hlt hP
          cases hbody : stepStmtsR R (core ++ tail)
              (s.setReg idx .nat [] (Tile.scalar cur)) with
          | none => rw [hbody] at hrun; exact absurd hrun (by simp)
          | some s1 =>
              rw [hbody] at hrun
              replace hrun : stepForRangeAuxR R idx (cur + step) stop step
                  (core ++ tail) s1 = some s_fin := hrun
              have hPn1 : P (cur + step) s1 := by
                have h12 : s'w = s1 := Option.some.inj (hs'w.symm.trans hbody)
                exact h12 ▸ hPnext
              have hu1 : s1.undef = (fun _ _ => 0) := by
                rw [stepStmtsR_undef R (core ++ tail) _ s1
                  (hokCore.append htail.flattenOkR) hbody]
                exact hu
              have hset : (A.flattenState s).setReg idx .nat []
                    (Tile.scalar cur)
                  = A.flattenState (s.setReg idx .nat [] (Tile.scalar cur)) :=
                (A.flattenState_setReg s idx .nat [] (Tile.scalar cur)).symm
              by_cases hfin : cur + step < stop
              · -- non-final iteration: strict per-statement commutation
                have hsafeBody : Stmt.TraceSafeListR R A.extent (core ++ tail)
                    (s.setReg idx .nat [] (Tile.scalar cur)) :=
                  Stmt.TraceSafeListR.append_intro core _ hsafeCore
                    (fun sMid hsMid => htailSafe hfin sMid hsMid)
                have hbridge := A.stepStmtsR_flatten hd hcov R (core ++ tail)
                  (s.setReg idx .nat [] (Tile.scalar cur)) hsafeBody
                  (hokCore.append htail.flattenOkR) hu
                rw [hbody, Option.map_some] at hbridge
                obtain ⟨t, ht, hobs, hdef⟩ :=
                  A.stepForRangeAuxR_flatten_deadPtrTail hd hcov R idx stop
                    step core tail htail hokCore P hstep (cur + step) s1
                    hPn1 hu1 s_fin hrun
                refine ⟨t, ?_, hobs, hdef⟩
                rw [stepForRangeAuxR, if_neg hz, if_pos hlt, hset, hbridge]
                exact ht
              · -- final iteration: strict core, tolerant tail; loop ends
                rw [stepForRangeAuxR, if_neg hz, if_neg hfin] at hrun
                have hs1fin : s1 = s_fin := Option.some.inj hrun
                rw [stepStmtsR_append R core tail] at hbody
                cases hcore : stepStmtsR R core
                    (s.setReg idx .nat [] (Tile.scalar cur)) with
                | none => rw [hcore] at hbody; exact absurd hbody (by simp)
                | some sMid =>
                    rw [hcore] at hbody
                    replace hbody : stepStmtsR R tail sMid = some s1 :=
                      hbody
                    have hcoreBridge := A.stepStmtsR_flatten hd hcov R core
                      (s.setReg idx .nat [] (Tile.scalar cur)) hsafeCore
                      hokCore hu
                    rw [hcore, Option.map_some] at hcoreBridge
                    obtain ⟨u', hu'run, hobs', hdef'⟩ :=
                      A.stepStmtsR_ptrDecTail_tolerant R tail sMid s1
                        (A.flattenState sMid) htail hbody
                        (A.regsDefEq_flattenState sMid)
                        (BlockState.ObsAgree.refl _)
                    have hflatBody : stepStmtsR R
                        (A.flattenStmts (core ++ tail))
                        (A.flattenState
                          (s.setReg idx .nat [] (Tile.scalar cur)))
                        = some u' := by
                      rw [A.flattenStmts_append core tail,
                        stepStmtsR_append R (A.flattenStmts core)
                          (A.flattenStmts tail) _,
                        hcoreBridge]
                      show stepStmtsR R (A.flattenStmts tail)
                          (A.flattenState sMid) = some u'
                      rw [A.flattenStmts_ptrDecTail tail htail]
                      exact hu'run
                    refine ⟨u', ?_, hs1fin ▸ hobs', hs1fin ▸ hdef'⟩
                    rw [stepForRangeAuxR, if_neg hz, if_pos hlt, hset,
                      hflatBody]
                    show stepForRangeAuxR R idx (cur + step) stop step
                        (A.flattenStmts (core ++ tail)) u' = some u'
                    rw [stepForRangeAuxR, if_neg hz, if_neg hfin]
        · rw [if_neg hlt, Option.some.injEq] at hrun
          subst hrun
          exact ⟨A.flattenState s,
            by rw [stepForRangeAuxR, if_neg hz, if_neg hlt],
            BlockState.ObsAgree.refl _, A.regsDefEq_flattenState s⟩
  termination_by cur _ _ _ _ _ => stop - cur
  decreasing_by omega

/-! ## Lemma D: the `forRangeDyn` wrapper and the top-level bridge -/

/-- Tolerant bridge for a single `forRangeDyn` statement whose body ends in
a PtrDec tail. The bound ops bridge strictly (their `SafeAtR`/`FlattenOkR`),
then the loop delegates to `stepForRangeAuxR_flatten_deadPtrTail`. The loop
obligations are stated relative to the *evaluated* bounds, keeping them in
the consumer's `hts`-obligation vocabulary. -/
theorem FlatAlloc.stepStmtR_forRangeDyn_flatten_deadPtrTail (A : FlatAlloc)
    (hd : A.Disjoint) (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (R : RoundingModel) (idx : RegName) (bOp eOp stOp : Op .nat [])
    (core tail : List Stmt) (htail : StmtList.PtrDecTail tail)
    (hokCore : StmtList.FlattenOkR core)
    (hokB : bOp.FlattenOkR) (hokE : eOp.FlattenOkR) (hokS : stOp.FlattenOkR)
    (s : BlockState) (hu : s.undef = (fun _ _ => 0))
    (hmsB : bOp.SafeAtR R A.extent s) (hmsE : eOp.SafeAtR R A.extent s)
    (hmsS : stOp.SafeAtR R A.extent s)
    (P : Nat → BlockState → Prop)
    (hloop : ∀ bv ev sv, evalOpR R bOp s = some bv →
      evalOpR R eOp s = some ev → evalOpR R stOp s = some sv →
      P (bv.data PUnit.unit) s ∧
      ∀ c s', c < ev.data PUnit.unit → P c s' →
        Stmt.TraceSafeListR R A.extent core
          (s'.setReg idx .nat [] (Tile.scalar c)) ∧
        (∃ s'', stepStmtsR R (core ++ tail)
            (s'.setReg idx .nat [] (Tile.scalar c)) = some s'' ∧
          P (c + sv.data PUnit.unit) s'') ∧
        (c + sv.data PUnit.unit < ev.data PUnit.unit → ∀ sMid,
          stepStmtsR R core (s'.setReg idx .nat [] (Tile.scalar c))
            = some sMid →
          Stmt.TraceSafeListR R A.extent tail sMid)) :
    ∀ s_fin, stepStmtR R (.forRangeDyn idx bOp eOp stOp (core ++ tail)) s
        = some s_fin →
    ∃ t, stepStmtR R
          (A.flattenStmt (.forRangeDyn idx bOp eOp stOp (core ++ tail)))
          (A.flattenState s) = some t ∧
      t.ObsAgree (A.flattenState s_fin) ∧ RegsDefEq t s_fin := by
  intro s_fin hrun
  simp only [stepStmtR] at hrun
  cases hb : evalOpR R bOp s with
  | none =>
      rw [hb] at hrun
      replace hrun : (none : Option BlockState) = some s_fin := hrun
      exact absurd hrun (by simp)
  | some bv =>
      rw [hb] at hrun
      cases he : evalOpR R eOp s with
      | none =>
          rw [he] at hrun
          replace hrun : (none : Option BlockState) = some s_fin := hrun
          exact absurd hrun (by simp)
      | some ev =>
          rw [he] at hrun
          cases hs : evalOpR R stOp s with
          | none =>
              rw [hs] at hrun
              replace hrun : (none : Option BlockState) = some s_fin := hrun
              exact absurd hrun (by simp)
          | some sv =>
              rw [hs] at hrun
              replace hrun : stepForRangeAuxR R idx (bv.data PUnit.unit)
                  (ev.data PUnit.unit) (sv.data PUnit.unit) (core ++ tail) s
                  = some s_fin := hrun
              obtain ⟨hP0, hstep⟩ := hloop bv ev sv hb he hs
              obtain ⟨t, ht, hobs, hdef⟩ :=
                A.stepForRangeAuxR_flatten_deadPtrTail hd hcov R idx
                  (ev.data PUnit.unit) (sv.data PUnit.unit) core tail htail
                  hokCore P hstep (bv.data PUnit.unit) s hP0 hu s_fin hrun
              refine ⟨t, ?_, hobs, hdef⟩
              simp only [FlatAlloc.flattenStmt, stepStmtR]
              rw [show evalOpR R (A.flattenOp bOp) (A.flattenState s)
                    = evalOpR R bOp s from by
                  rw [A.evalOpR_flatten hd hcov R bOp s hmsB hokB hu,
                    A.trTileFun_data (d := .nat) (by decide) (by decide),
                    Option.map_id, id_eq],
                hb,
                show evalOpR R (A.flattenOp eOp) (A.flattenState s)
                    = evalOpR R eOp s from by
                  rw [A.evalOpR_flatten hd hcov R eOp s hmsE hokE hu,
                    A.trTileFun_data (d := .nat) (by decide) (by decide),
                    Option.map_id, id_eq],
                he,
                show evalOpR R (A.flattenOp stOp) (A.flattenState s)
                    = evalOpR R stOp s from by
                  rw [A.evalOpR_flatten hd hcov R stOp s hmsS hokS hu,
                    A.trTileFun_data (d := .nat) (by decide) (by decide),
                    Option.map_id, id_eq],
                hs]
              exact ht

/-- **Top-level dead-write-tolerant flat bridge** for kernels of the shape
`pre ++ [forRangeDyn idx b e st (core ++ tail)]` — a straight-line prefix
followed by a *final* dynamic loop whose body trails off in pointer
self-decrements (`bwd_decay_global_cumsum`'s shape). The prefix and the
loop's core bridge strictly (their trace safety is hypothesised); the tail
is tolerated in the last iteration, so the conclusion is observational
agreement with the flat image of the region result rather than equality —
which is all any memory-level consumer (readback + frame legs) needs. -/
theorem FlatAlloc.execR_flatten_deadPtrTail (A : FlatAlloc)
    (hd : A.Disjoint) (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    (R : RoundingModel) (k : Kernel) (pre : List Stmt) (idx : RegName)
    (bOp eOp stOp : Op .nat []) (core tail : List Stmt)
    (hbody : k.body
      = pre ++ [Stmt.forRangeDyn idx bOp eOp stOp (core ++ tail)])
    (htail : StmtList.PtrDecTail tail)
    (hokPre : StmtList.FlattenOkR pre) (hokCore : StmtList.FlattenOkR core)
    (hokB : bOp.FlattenOkR) (hokE : eOp.FlattenOkR) (hokS : stOp.FlattenOkR)
    (s₀ : BlockState) (hu : s₀.undef = (fun _ _ => 0))
    (hmsPre : Stmt.TraceSafeListR R A.extent pre s₀)
    (P : Nat → BlockState → Prop)
    (hloop : ∀ sP, stepStmtsR R pre s₀ = some sP →
      (bOp.SafeAtR R A.extent sP ∧ eOp.SafeAtR R A.extent sP ∧
        stOp.SafeAtR R A.extent sP) ∧
      ∀ bv ev sv, evalOpR R bOp sP = some bv →
        evalOpR R eOp sP = some ev → evalOpR R stOp sP = some sv →
        P (bv.data PUnit.unit) sP ∧
        ∀ c s', c < ev.data PUnit.unit → P c s' →
          Stmt.TraceSafeListR R A.extent core
            (s'.setReg idx .nat [] (Tile.scalar c)) ∧
          (∃ s'', stepStmtsR R (core ++ tail)
              (s'.setReg idx .nat [] (Tile.scalar c)) = some s'' ∧
            P (c + sv.data PUnit.unit) s'') ∧
          (c + sv.data PUnit.unit < ev.data PUnit.unit → ∀ sMid,
            stepStmtsR R core (s'.setReg idx .nat [] (Tile.scalar c))
              = some sMid →
            Stmt.TraceSafeListR R A.extent tail sMid)) :
    ∀ s1, execR R k s₀ = some s1 →
    ∃ s', execR R (A.flattenKernel k) (A.flattenState s₀) = some s' ∧
      s'.ObsAgree (A.flattenState s1) := by
  intro s1 hrun
  simp only [execR, hbody] at hrun
  rw [stepStmtsR_append R pre _ s₀] at hrun
  cases hpre : stepStmtsR R pre s₀ with
  | none =>
      rw [hpre] at hrun
      replace hrun : (none : Option BlockState) = some s1 := hrun
      exact absurd hrun (by simp)
  | some sP =>
      rw [hpre] at hrun
      replace hrun : stepStmtsR R
          [Stmt.forRangeDyn idx bOp eOp stOp (core ++ tail)] sP
          = some s1 := hrun
      simp only [stepStmtsR] at hrun
      cases hstmt : stepStmtR R
          (Stmt.forRangeDyn idx bOp eOp stOp (core ++ tail)) sP with
      | none => rw [hstmt] at hrun; exact absurd hrun (by simp)
      | some sL =>
          rw [hstmt] at hrun
          replace hrun : some sL = some s1 := hrun
          have hsl : sL = s1 := Option.some.inj hrun
          have huP : sP.undef = (fun _ _ => 0) := by
            rw [stepStmtsR_undef R pre s₀ sP hokPre hpre]
            exact hu
          obtain ⟨⟨hmsB, hmsE, hmsS⟩, hloop'⟩ := hloop sP hpre
          obtain ⟨t, ht, hobs, hdef⟩ :=
            A.stepStmtR_forRangeDyn_flatten_deadPtrTail hd hcov R idx bOp
              eOp stOp core tail htail hokCore hokB hokE hokS sP huP hmsB
              hmsE hmsS P hloop' sL hstmt
          refine ⟨t, ?_, hsl ▸ hobs⟩
          simp only [execR, FlatAlloc.flattenKernel, hbody]
          rw [A.flattenStmts_append pre _,
            stepStmtsR_append R (A.flattenStmts pre) _ _,
            A.stepStmtsR_flatten hd hcov R pre s₀ hmsPre hokPre hu, hpre,
            Option.map_some]
          show stepStmtsR R (A.flattenStmts
              [Stmt.forRangeDyn idx bOp eOp stOp (core ++ tail)])
              (A.flattenState sP) = some t
          simp only [FlatAlloc.flattenStmts, stepStmtsR]
          rw [ht]

end VeriTile.Triton
