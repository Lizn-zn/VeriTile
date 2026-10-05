import bench.examples.OnlineSoftmax.Kernels
/- Use libdevice.exp for exp-sub rewrites: the measured fp32 tl.exp relation
has B = 0.1608954387 ULP > 0.05 under the configured Normal(1,1) probe.
That intrinsic relation failed admission; the libdevice EXP-SUB instance passed. -/
/- Opaque FP execution of the original online softmax recurrence. The source
has no store: the result lives in m/l, and all memory is preserved. Equating
this recurrence to the batch result still needs the admitted numerical atoms. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Control

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPExecution
open VeriTile.Bench.Examples.OnlineSoftmax.Kernels
open VeriTile Triton FP.Structural

set_option maxHeartbeats 1600000


private def ref (n : RegName) : Op .real [] := .ref .real [] n

def body (xReg : RegionName) (N : Nat) : List ComputeStmt :=
  [.assign .real [] "xi" (.alg (.load .real (.region xReg
      (.add .nat .nil (.mul .nat .nil (.ref .nat [] "pid") (.constNat N))
        (.ref .nat [] "i"))) .none)),
   .assign .real [] "m_new" (.alg (.max2 .nil (ref "m") (ref "xi"))),
   .assign .real [] "l" (.alg (.add .real .nil
      (.mul .real .nil (.libdeviceExp (.sub .real .nil (ref "m") (ref "m_new"))) (ref "l"))
      (.libdeviceExp (.sub .real .nil (ref "xi") (ref "m_new"))))),
   .assign .real [] "m" (.alg (ref "m_new"))]

def initialCode : List ComputeStmt :=
  [.assign .nat [] "pid" (.alg (.programId 0)),
   .assign .real [] "m" (.alg .negInf),
   .assign .real [] "l" (.alg (.const 0))]

theorem kernel_body (x y : RegionName) (N : Nat) :
    (onlineNormalizerKernel x y N).surfaceBody =
      initialCode ++ [.forLoop "i" N (body x N)] := rfl

def update {α : Type} (M : Algebra α) (x : α) (acc : α × α) : α × α :=
  let m := M.binary none .real .max acc.1 x
  (m, M.binary none .real .add
    (M.binary none .real .mul
      (M.unary none .libdeviceExp (M.binary none .real .sub acc.1 m)) acc.2)
    (M.unary none .libdeviceExp (M.binary none .real .sub x m)))

def recurrence {α : Type} (M : Algebra α) (xs : Fin N → α) : Nat → α × α
  | 0 => (M.negInf, M.literal none .real 0)
  | i + 1 => if h : i < N then update M (xs ⟨i, h⟩) (recurrence M xs i)
      else recurrence M xs i

private def Invariant {α : Type} [Inhabited α] (M : Algebra α) (xs : Fin N → α)
    (origin : State α) (i : Nat) (s : State α) : Prop :=
  s.regs .real [] "m" = some (fun _ => (recurrence M xs i).1) ∧
  s.regs .real [] "l" = some (fun _ => (recurrence M xs i).2) ∧
  s.regs .nat [] "pid" = some (fun _ => origin.pids 0) ∧
  s.mem = origin.mem ∧ s.pids = origin.pids

private theorem iteration {α : Type} [Inhabited α] (M : Algebra α)
    (xReg : RegionName) (xs : Fin N → α) (origin : State α)
    (hx : ∀ i : Fin N, (origin.mem xReg (origin.pids 0 * N + i.val)).read .real = xs i)
    (i : Nat) (s : State α) (hi : i < N) (hs : Invariant M xs origin i s) :
    ∃ t, run M (body xReg N) (s.setReg "i" .nat [] (fun _ => i)) = some t ∧
      Invariant M xs origin (i + 1) t := by
  rcases hs with ⟨hm, hl, hpid, hmem, hpids⟩
  have hxi := hx ⟨i, hi⟩
  simp [body, run, step, evalExpr, evalOp_unfold, numeric, FP.Structural.bop,
    ref, State.setReg, hmem, hm, hl, hpid, hxi, Region.cast,
    Invariant, recurrence, hi, update, hpids]
  exact ⟨rfl, rfl⟩

/-- For any floating interpretation, the original source terminates with its
recurrence in m/l. Even -inf and exp remain opaque; there is no hidden identity
for initialization, no Real projection and no invented output store. -/
theorem online_run {α : Type} [Inhabited α] (M : Algebra α)
    (xReg yReg : RegionName) (xs : Fin N → α) (origin : State α)
    (hx : ∀ i : Fin N, (origin.mem xReg (origin.pids 0 * N + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (onlineNormalizerKernel xReg yReg N) origin = some t ∧
      t.regs .real [] "m" = some (fun _ => (recurrence M xs N).1) ∧
      t.regs .real [] "l" = some (fun _ => (recurrence M xs N).2) ∧
      t.mem = origin.mem ∧ t.pids = origin.pids := by
  let s := ((origin.setReg "pid" .nat [] (fun _ => origin.pids 0)).setReg "m" .real []
    (fun _ => M.negInf)).setReg "l" .real [] (fun _ => M.literal none .real 0)
  have hs : Invariant M xs origin 0 s := by simp [Invariant, recurrence, s, State.setReg]
  obtain ⟨t, ht, hm, hl, _, hmem, hpids⟩ := forLoop_invariant M hs (iteration M xReg xs origin hx)
  refine ⟨t, ?_, hm, hl, hmem, hpids⟩
  have he : FP.Structural.exec M (onlineNormalizerKernel xReg yReg N) origin =
      run M (initialCode ++ [.forLoop "i" N (body xReg N)]) origin :=
    congrArg (fun code => run M code origin) (kernel_body xReg yReg N)
  rw [he, run_append]
  simpa [initialCode, run, step, evalExpr, evalOp_unfold, s] using ht

/-- The source's second pass, isolated only to compose its execution proof. -/
def normalizationBody (x y : RegionName) (N : Nat) : List ComputeStmt :=
  (onlineSoftmaxKernel x y N).surfaceBody.drop 4

theorem full_kernel_body (x y : RegionName) (N : Nat) :
    (onlineSoftmaxKernel x y N).surfaceBody =
      (onlineNormalizerKernel x y N).surfaceBody ++ normalizationBody x y N := rfl

def normalizedValue {α : Type} (M : Algebra α) (x m l : α) : α :=
  M.binary none .real .div (M.unary none .libdeviceExp
    (M.binary none .real .sub x m)) l

/-- Successful execution of the actual vector load and store. The load
precedes the store, including when x and y name the same region. -/
theorem normalization_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (N : Nat) (xs : Fin N → α) (s : State α) (m l : α)
    (hm : s.regs .real [] "m" = some (fun _ => m))
    (hl : s.regs .real [] "l" = some (fun _ => l))
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * N + i.val)).read .real = xs i) :
    ∃ t, run M (normalizationBody x y N) s = some t ∧
      (∀ i : Fin N, t.mem y (s.pids 0 * N + i.val) =
        .mk .real (normalizedValue M (xs i) m l)) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * N + i.val) → t.mem r o = s.mem r o) := by
  have hinj : Function.Injective (fun i : TileIndex [N] => s.pids 0 * N + i.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [normalizationBody, onlineSoftmaxKernel, ComputeKernel.surfaceBody, List.drop, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, hm, hl, hx, Region.cast, normalizedValue]
  refine ⟨fun i => ?_, fun r o hmiss => ?_⟩
  · rw [State.scatter_readback _ _ _ _ (i, PUnit.unit) hinj]
  · apply (State.scatter_frame _ _ _ _ r o ?_ _).trans rfl
    rcases hmiss with hr | ho
    · exact Or.inl hr
    · exact Or.inr fun k _ => ho k.1

/-- Complete online kernel: final output cells and the memory frame. -/
theorem online_output_run {α : Type} [Inhabited α] (M : Algebra α)
    (x y : RegionName) (N : Nat) (xs : Fin N → α) (s : State α)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * N + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (onlineSoftmaxKernel x y N) s = some t ∧
      (∀ i : Fin N, t.mem y (s.pids 0 * N + i.val) =
        .mk .real (normalizedValue M (xs i) (recurrence M xs N).1 (recurrence M xs N).2)) ∧
      (∀ r o, (r ≠ y ∨ ∀ i : Fin N, o ≠ s.pids 0 * N + i.val) → t.mem r o = s.mem r o) := by
  obtain ⟨a, ha, hm, hl, hmem, hpids⟩ := online_run M x y xs s hx
  obtain ⟨b, hb, hout, hframe⟩ := normalization_run M x y N xs a _ _ hm hl
    (by simpa [hmem, hpids] using hx)
  refine ⟨b, ?_, ?_, ?_⟩
  · change run M (onlineSoftmaxKernel x y N).surfaceBody s = some b
    rw [full_kernel_body, run_append]
    change (FP.Structural.exec M (onlineNormalizerKernel x y N) s).bind _ = _
    rw [ha]
    exact hb
  · simpa [hpids] using hout
  · simpa [hpids, hmem] using hframe

end VeriTile.Bench.Examples.OnlineSoftmaxFPExecution
