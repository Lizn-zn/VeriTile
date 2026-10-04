/- Use libdevice.exp for exp-sub rewrites: the measured fp32 tl.exp relation
has B = 0.1608954387 ULP > 0.05 under the configured Normal(1,1) probe.
That intrinsic relation failed admission; the libdevice EXP-SUB instance passed. -/
/- Opaque FP execution of the original online softmax recurrence. The source
has no store: the result lives in m/l, and all memory is preserved. Equating
this recurrence to the batch result still needs the admitted numerical atoms. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Control

namespace VeriTile.Bench.Examples.OnlineSoftmaxFPExecution
open VeriTile Triton FP.Structural

set_option maxHeartbeats 1600000

def onlineSoftmaxKernel (xReg _yReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  m   := -inf
  l   := 0
  tl.for i in $(N) {
    xi    := tl.load($(xReg) + (pid * $(N) + i))
    m_new := tl.max(m, xi)
    l     := libdevice.exp(m - m_new) * l + libdevice.exp(xi - m_new)
    m     := m_new
  }
}

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
    (onlineSoftmaxKernel x y N).surfaceBody =
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
    ∃ t, FP.Structural.exec M (onlineSoftmaxKernel xReg yReg N) origin = some t ∧
      t.regs .real [] "m" = some (fun _ => (recurrence M xs N).1) ∧
      t.regs .real [] "l" = some (fun _ => (recurrence M xs N).2) ∧
      t.mem = origin.mem ∧ t.pids = origin.pids := by
  let s := ((origin.setReg "pid" .nat [] (fun _ => origin.pids 0)).setReg "m" .real []
    (fun _ => M.negInf)).setReg "l" .real [] (fun _ => M.literal none .real 0)
  have hs : Invariant M xs origin 0 s := by simp [Invariant, recurrence, s, State.setReg]
  obtain ⟨t, ht, hm, hl, _, hmem, hpids⟩ := forLoop_invariant M hs (iteration M xReg xs origin hx)
  refine ⟨t, ?_, hm, hl, hmem, hpids⟩
  have he : FP.Structural.exec M (onlineSoftmaxKernel xReg yReg N) origin =
      run M (initialCode ++ [.forLoop "i" N (body xReg N)]) origin :=
    congrArg (fun code => run M code origin) (kernel_body xReg yReg N)
  rw [he, run_append]
  simpa [initialCode, run, step, evalExpr, evalOp_unfold, s] using ht

end VeriTile.Bench.Examples.OnlineSoftmaxFPExecution
