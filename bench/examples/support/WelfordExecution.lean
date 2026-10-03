/- FP execution of the original Welford sources, before the numerical
comparison with the two-pass kernel. Operations, count conversion and output
casts remain opaque. This file does not assert the pending FP equivalence. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Control

namespace VeriTile.Bench.Examples.WelfordFPExecution
open VeriTile Triton FP.Structural

set_option maxHeartbeats 1600000

def twopassWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize rowStride : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(rowStride) + tl.arange($(blockSize))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(blockSize))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(blockSize))
  tl.store($(meanReg), (μ).to(tl.bfloat16))
  tl.store($(varReg), (v).to(tl.bfloat16))
}

def onlineWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize rowStride : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(blockSize) {
    xi     := tl.load($(xReg) + (pid * $(rowStride) + i))
    delta  := xi - M
    M      := M + delta / (tl.toReal(i) + 1)
    delta2 := xi - M
    S      := S + delta * delta2
  }
  tl.store($(meanReg), (M).to(tl.bfloat16))
  tl.store($(varReg), (S / tl.toReal($(blockSize))).to(tl.bfloat16))
}

private def realRef (n : RegName) : Op .real [] := .ref .real [] n

def body (xReg : RegionName) (rowStride : Nat) : List ComputeStmt :=
  [.assign .real [] "xi" (.alg (.load .real (.region xReg
      (.add .nat .nil (.mul .nat .nil (.ref .nat [] "pid") (.constNat rowStride))
        (.ref .nat [] "i"))) .none)),
   .assign .real [] "delta" (.alg (.sub .real .nil (realRef "xi") (realRef "M"))),
   .assign .real [] "M" (.alg (.add .real .nil (realRef "M")
      (.div .real .nil (realRef "delta")
        (.add .real .nil (.natToReal (.ref .nat [] "i")) (.const 1))))),
   .assign .real [] "delta2" (.alg (.sub .real .nil (realRef "xi") (realRef "M"))),
   .assign .real [] "S" (.alg (.add .real .nil (realRef "S")
      (.mul .real .nil (realRef "delta") (realRef "delta2"))))]

def initialCode : List ComputeStmt :=
  [.assign .nat [] "pid" (.alg (.programId 0)),
   .assign .real [] "M" (.alg (.const 0)),
   .assign .real [] "S" (.alg (.const 0))]

def outputCode (meanReg varReg : RegionName) (N : Nat) : List ComputeStmt :=
  [.store .bf16 [] (.region ⟨meanReg.name⟩ (.constNat 0))
      (.alg (.castFloat .real .bf16 (realRef "M"))) .none,
   .store .bf16 [] (.region ⟨varReg.name⟩ (.constNat 0))
      (.alg (.castFloat .real .bf16
        (.div .real .nil (realRef "S") (.natToReal (.constNat N))))) .none]

theorem kernel_body (x m v : RegionName) (N stride : Nat) :
    (onlineWelfordKernel x m v N stride).surfaceBody =
      initialCode ++ [.forLoop "i" N (body x stride)] ++ outputCode m v N := rfl

def update {α : Type} (M : Algebra α) (i : Nat) (x : α) (acc : α × α) : α × α :=
  let delta := M.binary none .real .sub x acc.1
  let mean := M.binary none .real .add acc.1
    (M.binary none .real .div delta
      (M.binary none .real .add (M.fromNat none i) (M.literal none .real 1)))
  (mean, M.binary none .real .add acc.2
    (M.binary none .real .mul delta (M.binary none .real .sub x mean)))

def recurrence {α : Type} (M : Algebra α) (xs : Fin N → α) : Nat → α × α
  | 0 => (M.literal none .real 0, M.literal none .real 0)
  | i + 1 => if h : i < N then update M i (xs ⟨i, h⟩) (recurrence M xs i)
      else recurrence M xs i

private def Invariant {α : Type} [Inhabited α] (M : Algebra α) (xs : Fin N → α)
    (origin : State α) (i : Nat) (s : State α) : Prop :=
  s.regs .real [] "M" = some (fun _ => (recurrence M xs i).1) ∧
  s.regs .real [] "S" = some (fun _ => (recurrence M xs i).2) ∧
  s.regs .nat [] "pid" = some (fun _ => origin.pids 0) ∧
  s.mem = origin.mem ∧ s.pids = origin.pids

private theorem iteration {α : Type} [Inhabited α] (M : Algebra α)
    (xReg : RegionName) (stride : Nat) (xs : Fin N → α) (origin : State α)
    (hx : ∀ i : Fin N, (origin.mem xReg (origin.pids 0 * stride + i.val)).read .real = xs i)
    (i : Nat) (s : State α) (hi : i < N) (hs : Invariant M xs origin i s) :
    ∃ t, run M (body xReg stride) (s.setReg "i" .nat [] (fun _ => i)) = some t ∧
      Invariant M xs origin (i + 1) t := by
  rcases hs with ⟨hm, hv, hpid, hmem, hpids⟩
  have hxi := hx ⟨i, hi⟩
  simp [body, run, step, evalExpr, evalOp_unfold, numeric, FP.Structural.bop,
    realRef, State.setReg, hmem, hm, hv, hpid, hxi, Region.cast,
    Invariant, recurrence, hi, update, hpids]
  exact ⟨rfl, rfl⟩

/-- The original loop successfully computes its exact opaque recurrence. No
law about arithmetic, count conversion or finite values is needed to execute. -/
theorem loop_run {α : Type} [Inhabited α] (M : Algebra α)
    (xReg : RegionName) (stride : Nat) (xs : Fin N → α) (origin : State α)
    (hx : ∀ i : Fin N, (origin.mem xReg (origin.pids 0 * stride + i.val)).read .real = xs i) :
    ∃ t, run M (initialCode ++ [.forLoop "i" N (body xReg stride)]) origin = some t ∧
      t.regs .real [] "M" = some (fun _ => (recurrence M xs N).1) ∧
      t.regs .real [] "S" = some (fun _ => (recurrence M xs N).2) ∧
      t.mem = origin.mem ∧ t.pids = origin.pids := by
  let s := ((origin.setReg "pid" .nat [] (fun _ => origin.pids 0)).setReg "M" .real []
    (fun _ => M.literal none .real 0)).setReg "S" .real [] (fun _ => M.literal none .real 0)
  have hs : Invariant M xs origin 0 s := by simp [Invariant, recurrence, s, State.setReg]
  obtain ⟨t, ht, hm, hv, _, hmem, hpids⟩ := forLoop_invariant M hs (iteration M xReg stride xs origin hx)
  refine ⟨t, ?_, hm, hv, hmem, hpids⟩
  rw [run_append]
  simpa [initialCode, run, step, evalExpr, evalOp_unfold, s] using ht

def meanValue {α : Type} (M : Algebra α) (xs : Fin N → α) : α :=
  M.cast none .real .bf16 (recurrence M xs N).1

def varianceValue {α : Type} (M : Algebra α) (xs : Fin N → α) : α :=
  M.cast none .real .bf16 (M.binary none .real .div (recurrence M xs N).2 (M.fromNat none N))

/-- Both original bf16 output cells and every untouched cell are accounted
for. Output regions must differ; the input may alias either output because all
reads finish before the two stores. Empty rows still execute the original code. -/
theorem online_run {α : Type} [Inhabited α] (M : Algebra α)
    (xReg meanReg varReg : RegionName) (stride : Nat) (xs : Fin N → α) (s : State α)
    (hd : meanReg ≠ varReg)
    (hx : ∀ i : Fin N, (s.mem xReg (s.pids 0 * stride + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (onlineWelfordKernel xReg meanReg varReg N stride) s = some t ∧
      t.mem meanReg 0 = .mk .bf16 (meanValue M xs) ∧
      t.mem varReg 0 = .mk .bf16 (varianceValue M xs) ∧
      (∀ r o, (r ≠ meanReg ∨ o ≠ 0) → (r ≠ varReg ∨ o ≠ 0) → t.mem r o = s.mem r o) := by
  obtain ⟨u, hu, hm, hv, hmem, _⟩ := loop_run M xReg stride xs s hx
  have he : FP.Structural.exec M (onlineWelfordKernel xReg meanReg varReg N stride) s =
      run M (initialCode ++ [.forLoop "i" N (body xReg stride)] ++ outputCode meanReg varReg N) s :=
    congrArg (fun code => run M code s) (kernel_body xReg meanReg varReg N stride)
  let t := (u.write meanReg 0 (.mk .bf16 (meanValue M xs))).write varReg 0
    (.mk .bf16 (varianceValue M xs))
  refine ⟨t, ?_, ?_, ?_, ?_⟩
  · rw [he, run_append, hu]
    simp [outputCode, run, step, evalExpr, evalOp_unfold, realRef, hm, hv, numeric,
      FP.Structural.bop, store, TileShape.allIndices, Region.cast, meanValue, varianceValue,
      t, State.write, ofFloat, toFloat]
  · exact (State.write_other _ varReg meanReg 0 0 _ (Or.inl hd)).trans
      (State.write_same _ _ _ _)
  · exact State.write_same _ _ _ _
  · intro r o h₁ h₂
    rw [State.write_other _ varReg r 0 o _ h₂, State.write_other _ meanReg r 0 o _ h₁, hmem]

def sumValue {α : Type} (M : Algebra α) (xs : Fin N → α) : α :=
  M.reduceSum none (shape := [N]) ⟨0, by simp⟩ Bool.false (fun i => xs i.1) PUnit.unit

def twopassMean {α : Type} (M : Algebra α) (xs : Fin N → α) : α :=
  M.binary none .real .div (sumValue M xs) (M.fromNat none N)

def twopassVariance {α : Type} (M : Algebra α) (xs : Fin N → α) : α :=
  let delta := fun i => M.binary none .real .sub (xs i) (twopassMean M xs)
  M.binary none .real .div
    (sumValue M (fun i => M.binary none .real .mul (delta i) (delta i))) (M.fromNat none N)

/-- Execution of the two-pass side preserves the original two reductions,
integer-count conversions and bf16 stores; no fold identity is assumed. -/
theorem twopass_run {α : Type} [Inhabited α] (M : Algebra α)
    (xReg meanReg varReg : RegionName) (stride : Nat) (xs : Fin N → α) (s : State α)
    (hd : meanReg ≠ varReg)
    (hx : ∀ i : Fin N, (s.mem xReg (s.pids 0 * stride + i.val)).read .real = xs i) :
    ∃ t, FP.Structural.exec M (twopassWelfordKernel xReg meanReg varReg N stride) s = some t ∧
      t.mem meanReg 0 = .mk .bf16 (M.cast none .real .bf16 (twopassMean M xs)) ∧
      t.mem varReg 0 = .mk .bf16 (M.cast none .real .bf16 (twopassVariance M xs)) ∧
      (∀ r o, (r ≠ meanReg ∨ o ≠ 0) → (r ≠ varReg ∨ o ≠ 0) → t.mem r o = s.mem r o) := by
  simp [twopassWelfordKernel, FP.Structural.exec, run, step, evalExpr, evalOp_unfold,
    numeric, FP.Structural.bop, store, State.write, TileShape.allIndices,
    TileShape.eraseAxis, Region.cast, ofFloat, toFloat, hx, hd,
    sumValue, twopassMean, twopassVariance]
  refine ⟨rfl, ?_⟩
  intro r o h₁ h₂
  simp only [if_neg (by tauto : ¬ (r = varReg ∧ o = 0)),
    if_neg (by tauto : ¬ (r = meanReg ∧ o = 0))]

end VeriTile.Bench.Examples.WelfordFPExecution
