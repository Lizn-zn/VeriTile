/-
VeriTile.Triton.Float.Pipeline

Honest n-launch pipeline semantics (#447 Phase C.5).

`ComputeKernel.seq` (Core.Ast) concatenates stage bodies into one kernel.
That concatenation is a purely syntactic surrogate for "launch the stages one
after another": it silently lets register bindings flow across the stage
seams, which no real multi-launch execution allows. This file provides:

* `execPipelineR` — the honest n-launch semantics: every stage starts from a
  fresh register file (`BlockState.resetRegs`), including the first (a real
  launch never inherits registers).
* `execR_toAlgKernel_seq` — the syntactic split: the concatenated kernel's
  execution is the *no-reset* left fold of the stage executions.
* `MemPidsEq` — state agreement up to registers (memory, program ids, undef,
  grid dimensions all equal), with `MemPidsEqO` its `Option` lift (both
  executions fail, or both succeed with `MemPidsEq` results).
* `ComputeKernel.RegClosed` — the semantic "never reads an unwritten
  register" condition: execution from `MemPidsEq` states yields
  `MemPidsEqO` outcomes.
* `ComputeKernel.execR_seq_rel_execPipelineR` — **the bridge theorem**: if
  every stage is `RegClosed`, the concatenated kernel and the honest
  pipeline agree on everything but the final (dead) register file. This
  discharges the register-leakage gap of concatenation.
-/

import VeriTile.Triton.Float.Refine

namespace VeriTile.Triton

/-! ## Honest launch semantics -/

/-- The honest n-launch pipeline semantics: run the stage kernels in order,
resetting the register file before **every** launch (a fresh launch never
inherits registers — including the first stage). Memory, program ids, undef,
and grid dimensions persist across the seams. -/
noncomputable def execPipelineR (R : RoundingModel) :
    List ComputeKernel → BlockState → Option BlockState
  | [], s => some s
  | k :: ks, s =>
      (execR R k.toAlgKernel s.resetRegs).bind fun s' => execPipelineR R ks s'

@[simp] theorem execPipelineR_nil (R : RoundingModel) (s : BlockState) :
    execPipelineR R [] s = some s := rfl

@[simp] theorem execPipelineR_cons (R : RoundingModel) (k : ComputeKernel)
    (ks : List ComputeKernel) (s : BlockState) :
    execPipelineR R (k :: ks) s =
      (execR R k.toAlgKernel s.resetRegs).bind fun s' => execPipelineR R ks s' := rfl

/-- Degeneration at the trivial model: each stage of the honest pipeline runs
under the exact base semantics `exec`. -/
theorem execPipelineR_triv_cons (k : ComputeKernel) (ks : List ComputeKernel)
    (s : BlockState) :
    execPipelineR .triv (k :: ks) s =
      (exec k.toAlgKernel s.resetRegs).bind fun s' => execPipelineR .triv ks s' := by
  rw [execPipelineR_cons, execR_triv]

/-! ## The syntactic split of the concatenated kernel -/

private theorem foldl_execR_none (R : RoundingModel) (ks : List ComputeKernel) :
    ks.foldl (fun acc k => acc.bind fun s' => execR R k.toAlgKernel s') none = none := by
  induction ks with
  | nil => rfl
  | cons k ks ih => simpa using ih

/-- Executing the concatenated kernel `ComputeKernel.seq inputs outputs ks`
is the **no-reset** left fold of the stage executions: registers flow freely
across the stage seams. Contrast with `execPipelineR`, which resets them. -/
theorem execR_toAlgKernel_seq (R : RoundingModel)
    (inputs outputs : List RegionName) (ks : List ComputeKernel) (s : BlockState) :
    execR R (ComputeKernel.seq inputs outputs ks).toAlgKernel s =
      ks.foldl (fun acc k => acc.bind fun s' => execR R k.toAlgKernel s') (some s) := by
  rw [ComputeKernel.toAlgKernel_seq]
  show stepStmtsR R (ks.flatMap ComputeKernel.body) s = _
  induction ks generalizing s with
  | nil => simp [stepStmtsR]
  | cons k ks ih =>
      rw [List.flatMap_cons, stepStmtsR_append]
      have hk : execR R k.toAlgKernel s = stepStmtsR R k.body s := rfl
      cases h : stepStmtsR R k.body s with
      | none =>
          simp only [List.foldl_cons, hk, h, Option.bind_none, Option.bind_some]
          exact (foldl_execR_none R ks).symm
      | some s1 =>
          simp only [List.foldl_cons, hk, h, Option.bind_some]
          exact ih s1

/-! ## State agreement up to registers -/

/-- `s` and `t` agree on everything but the register file: memory, program
ids, the undef environment, and the launch-grid dimensions are all equal. -/
def MemPidsEq (s t : BlockState) : Prop :=
  s.mem = t.mem ∧ s.pids = t.pids ∧ s.undef = t.undef ∧ s.numPids = t.numPids

namespace MemPidsEq

variable {s t u : BlockState}

theorem mem_eq (h : MemPidsEq s t) : s.mem = t.mem := h.1
theorem pids_eq (h : MemPidsEq s t) : s.pids = t.pids := h.2.1
theorem undef_eq (h : MemPidsEq s t) : s.undef = t.undef := h.2.2.1
theorem numPids_eq (h : MemPidsEq s t) : s.numPids = t.numPids := h.2.2.2

@[refl] theorem refl (s : BlockState) : MemPidsEq s s := ⟨rfl, rfl, rfl, rfl⟩

theorem symm (h : MemPidsEq s t) : MemPidsEq t s :=
  ⟨h.1.symm, h.2.1.symm, h.2.2.1.symm, h.2.2.2.symm⟩

theorem trans (h₁ : MemPidsEq s t) (h₂ : MemPidsEq t u) : MemPidsEq s u :=
  ⟨h₁.1.trans h₂.1, h₁.2.1.trans h₂.2.1, h₁.2.2.1.trans h₂.2.2.1,
   h₁.2.2.2.trans h₂.2.2.2⟩

/-- `MemPidsEq` states have literally equal fresh-launch resets: `resetRegs`
erases the only field on which they may differ. -/
theorem resetRegs_eq (h : MemPidsEq s t) : s.resetRegs = t.resetRegs := by
  obtain ⟨hm, hp, hu, hn⟩ := h
  unfold BlockState.resetRegs
  rw [hm, hp, hu, hn]

/-! ### Rewriting the register-independent observations -/

theorem readMem_eq (h : MemPidsEq s t) : t.readMem = s.readMem := by
  funext r o
  unfold BlockState.readMem
  rw [← h.mem_eq]

theorem readMemAs_eq (h : MemPidsEq s t) : t.readMemAs = s.readMemAs := by
  funext dt r o
  unfold BlockState.readMemAs
  rw [← h.mem_eq]

theorem readMemTyped_eq (h : MemPidsEq s t) : t.readMemTyped = s.readMemTyped := by
  funext dt r o
  unfold BlockState.readMemTyped
  rw [← h.mem_eq]

theorem readMemValue_eq (h : MemPidsEq s t) : t.readMemValue = s.readMemValue := by
  funext dt r o
  cases dt <;>
    simp only [BlockState.readMemValue, h.readMemAs_eq, h.readMemTyped_eq]

/-! ### Congruence under the register/memory writes -/

/-- `setReg` only touches registers, so it preserves `MemPidsEq` pairs. -/
theorem setReg (h : MemPidsEq s t) (name : RegName) (dtype : TileDType)
    (shape : TileShape) (v : Tile dtype shape) :
    MemPidsEq (s.setReg name dtype shape v) (t.setReg name dtype shape v) :=
  ⟨h.1, h.2.1, h.2.2.1, h.2.2.2⟩

/-- The rounding-aware store writes the same cell with the same value on both
sides of a `MemPidsEq` pair. -/
theorem writeMemAsR (h : MemPidsEq s t) (R : RoundingModel) (dtype : FloatDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype.toTileDType) :
    MemPidsEq (s.writeMemAsR R dtype region offset v)
      (t.writeMemAsR R dtype region offset v) := by
  obtain ⟨hm, hp, hu, hn⟩ := h
  refine ⟨?_, hp, hu, hn⟩
  funext r o
  rw [BlockState.writeMemAsR_mem, BlockState.writeMemAsR_mem, hm]

/-- `MemPidsEq` is a congruence for any fold whose step preserves it. -/
theorem foldl_congr {α : Type*} {f : BlockState → α → BlockState}
    (hf : ∀ ⦃a b : BlockState⦄, MemPidsEq a b → ∀ x : α, MemPidsEq (f a x) (f b x)) :
    ∀ (l : List α) {a b : BlockState}, MemPidsEq a b →
      MemPidsEq (l.foldl f a) (l.foldl f b)
  | [], _, _, h => h
  | x :: l, a, b, h => by
      rw [List.foldl_cons, List.foldl_cons]
      exact foldl_congr hf l (hf h x)

end MemPidsEq

/-- Every state agrees with its own fresh-launch reset up to registers. -/
theorem memPidsEq_resetRegs (s : BlockState) : MemPidsEq s s.resetRegs :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- `Option` lift of `MemPidsEq`: both executions fail, or both succeed with
`MemPidsEq`-related results. -/
def MemPidsEqO : Option BlockState → Option BlockState → Prop
  | none, none => True
  | some s, some t => MemPidsEq s t
  | _, _ => False

@[simp] theorem memPidsEqO_none_none : MemPidsEqO none none := trivial

@[simp] theorem memPidsEqO_some_some (s t : BlockState) :
    MemPidsEqO (some s) (some t) ↔ MemPidsEq s t := Iff.rfl

@[simp] theorem memPidsEqO_none_some (t : BlockState) :
    ¬ MemPidsEqO none (some t) := fun h => h

@[simp] theorem memPidsEqO_some_none (s : BlockState) :
    ¬ MemPidsEqO (some s) none := fun h => h

/-! ## Register-closed kernels and the bridge theorem -/

namespace ComputeKernel

/-- Semantic register-closedness: the kernel's execution never depends on
pre-existing register bindings. From any two states agreeing up to registers,
the two executions both fail or both succeed with results again agreeing up
to registers. No syntactic analysis — this is exactly the property the
seam of a concatenated pipeline needs. -/
def RegClosed (k : ComputeKernel) : Prop :=
  ∀ (R : RoundingModel) (s t : BlockState), MemPidsEq s t →
    MemPidsEqO (execR R k.toAlgKernel s) (execR R k.toAlgKernel t)

end ComputeKernel

private theorem foldl_execR_rel_execPipelineR (R : RoundingModel) :
    ∀ (ks : List ComputeKernel), (∀ k ∈ ks, k.RegClosed) →
      ∀ (s t : BlockState), MemPidsEq s t →
        MemPidsEqO
          (ks.foldl (fun acc k => acc.bind fun s' => execR R k.toAlgKernel s') (some s))
          (execPipelineR R ks t)
  | [], _, s, t, h => h
  | k :: ks, hks, s, t, h => by
      have hst : MemPidsEq s t.resetRegs := h.trans (memPidsEq_resetRegs t)
      have hk := hks k List.mem_cons_self R s t.resetRegs hst
      rw [List.foldl_cons, execPipelineR_cons, Option.bind_some]
      cases h1 : execR R k.toAlgKernel s with
      | none =>
          rw [h1] at hk
          cases h2 : execR R k.toAlgKernel t.resetRegs with
          | none =>
              rw [foldl_execR_none R ks, Option.bind_none]
              exact memPidsEqO_none_none
          | some t1 =>
              rw [h2] at hk
              exact absurd hk (memPidsEqO_none_some t1)
      | some s1 =>
          rw [h1] at hk
          cases h2 : execR R k.toAlgKernel t.resetRegs with
          | none =>
              rw [h2] at hk
              exact absurd hk (memPidsEqO_some_none s1)
          | some t1 =>
              rw [h2] at hk
              rw [Option.bind_some]
              exact foldl_execR_rel_execPipelineR R ks
                (fun k' hk' => hks k' (List.mem_cons_of_mem _ hk')) s1 t1 hk

/-- **The bridge theorem** (#447 Phase C.5): if every stage of the pipeline
is `RegClosed`, the syntactic concatenation `ComputeKernel.seq` and the
honest per-launch semantics `execPipelineR` are indistinguishable up to the
final (dead) register file: both fail, or both succeed with `MemPidsEq`
results. This discharges the register-leakage gap of concatenation. -/
theorem ComputeKernel.execR_seq_rel_execPipelineR (R : RoundingModel)
    (inputs outputs : List RegionName) {ks : List ComputeKernel}
    (hks : ∀ k ∈ ks, k.RegClosed) (s : BlockState) :
    MemPidsEqO
      (execR R (ComputeKernel.seq inputs outputs ks).toAlgKernel s)
      (execPipelineR R ks s) := by
  rw [execR_toAlgKernel_seq]
  exact foldl_execR_rel_execPipelineR R ks hks s s (MemPidsEq.refl s)

end VeriTile.Triton
