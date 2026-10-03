/- FP execution of the original Welford sources, before the numerical
comparison with the two-pass kernel. Operations, count conversion and output
casts remain opaque. This file does not assert the pending FP equivalence. -/
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Control
import VeriTile.Triton.Float.ExecutionProfile
import VeriTile.Triton.Float.WelfordInduction

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

/-- Connect the scalar-derived mean step to the exact original update under
an fp32 execution profile. The integer conversion is retained on both sides;
this theorem does not identify fromNat(i+1) with fromNat(i)+1. -/
theorem fp32_mean_step {α : Type} [Inhabited α] (R : FP.ScalarArithmetic.Rules)
    (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : FP.Guarded.Models R.assumptions M D) (s : State α)
    (i : Nat) (x mean variance : α)
    (hd : FP.Welford.MeanStepDomain M D x mean (M.fromNat (some .fp32) i)) :
    FP.ScalarArithmetic.mul M
      (update (M.withDefaultPrecision .fp32) i x (mean, variance)).1
      (FP.Welford.nextCount M (M.fromNat (some .fp32) i)) =
      FP.ScalarArithmetic.add M
        (FP.ScalarArithmetic.mul M mean (M.fromNat (some .fp32) i)) x :=
  FP.Welford.mean_step R M D hM s x mean (M.fromNat (some .fp32) i) hd

/-- The original S update splits into the new sample's squared residual and
the old-count center-shift contribution, using only admitted scalar atoms. -/
theorem fp32_variance_step {α : Type} [Inhabited α] (R : FP.ScalarArithmetic.Rules)
    (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : FP.Guarded.Models R.assumptions M D) (s : State α)
    (i : Nat) (x mean variance : α)
    (hd : FP.Welford.VarianceStepDomain M D x mean (M.fromNat (some .fp32) i)) :
    (update (M.withDefaultPrecision .fp32) i x (mean, variance)).2 =
      FP.ScalarArithmetic.add M variance
        (FP.ScalarArithmetic.add M
          (FP.Welford.square M (FP.Welford.residual M x mean (M.fromNat (some .fp32) i)))
          (FP.ScalarArithmetic.mul M (M.fromNat (some .fp32) i)
            (FP.Welford.square M (FP.Welford.correction M x mean (M.fromNat (some .fp32) i))))) :=
  congrArg (FP.ScalarArithmetic.add M variance)
    (FP.Welford.variance_step R M D hM s x mean (M.fromNat (some .fp32) i) hd)

def recurrence {α : Type} (M : Algebra α) (xs : Fin N → α) : Nat → α × α
  | 0 => (M.literal none .real 0, M.literal none .real 0)
  | i + 1 => if h : i < N then update M i (xs ⟨i, h⟩) (recurrence M xs i)
      else recurrence M xs i

/-- The scalar induction is the exact arithmetic of the original update at
its fp32 default precision, including the natural-index conversion. -/
theorem fp32_update {α : Type} (M : Algebra α) (i : Nat) (x : α) (acc : α × α) :
    update (M.withDefaultPrecision .fp32) i x acc =
      FP.WelfordInduction.update M i x acc := rfl

theorem fp32_recurrence_prefix {α : Type} (M : Algebra α) (xs : Nat → α)
    (N k : Nat) (hk : k ≤ N) :
    recurrence (M.withDefaultPrecision .fp32) (FP.WelfordInduction.rowPrefix xs N) k =
      FP.WelfordInduction.state M xs k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [recurrence, dif_pos (by omega), ih (by omega), fp32_update]
    rfl

/-- All original loop iterations now inherit the scalar-derived statistics
invariant. The two primitive count laws are explicit unadmitted premises;
no whole-row or whole-kernel equality is supplied as an assumption. -/
theorem fp32_recurrence_statistics {α : Type} [Inhabited α]
    (R : FP.ScalarArithmetic.Rules) (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : FP.Guarded.Models R.assumptions M D) (s : State α)
    (xs : Nat → α) (empty : FP.Equational.ReductionTree 0) (N : Nat) (hN : 0 < N)
    (hc : FP.WelfordInduction.CountConversion M N)
    (hd : FP.WelfordInduction.IterationDomain M D xs empty N) :
    recurrence (M.withDefaultPrecision .fp32) (FP.WelfordInduction.rowPrefix xs N) N =
      FP.WelfordInduction.statistics M (FP.WelfordInduction.rowPrefix xs N)
        (FP.WelfordInduction.tree empty N) := by
  rw [fp32_recurrence_prefix M xs N N le_rfl]
  exact FP.WelfordInduction.state_statistics R M D hM s xs empty N hN hc hd

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

/-- The original implementations expose both statistics, with symbolic row
length and stride. These contracts add no numerical relation between them. -/
def onlineIO (x mean variance : RegionName) (N stride : Nat) : KernelIO₁ₓ₂ where
  kernel := onlineWelfordKernel x mean variance N stride
  projection := by rfl
  inp := x
  out1 := mean
  out2 := variance
  Bin := N
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * stride
  write1 := fun _ => 0
  write2 := fun _ => 0

def twopassIO (x mean variance : RegionName) (N stride : Nat) : KernelIO₁ₓ₂ :=
  { onlineIO x mean variance N stride with
    kernel := twopassWelfordKernel x mean variance N stride
    projection := by rfl }

theorem io_same_signature (x mean variance : RegionName) (N stride : Nat) :
    io₁ₓ₂Signature (onlineIO x mean variance N stride) =
      io₁ₓ₂Signature (twopassIO x mean variance N stride) := rfl

private theorem scalar_outputs_frame {α : Type} {io : KernelIO₁ₓ₂} {s t : State α}
    (h₁ : io.Bout1 = 1) (h₂ : io.Bout2 = 1)
    (hw₁ : io.write1 (s.pids 0) = 0) (hw₂ : io.write2 (s.pids 0) = 0)
    (h : ∀ r o, (r ≠ io.out1 ∨ o ≠ 0) → (r ≠ io.out2 ∨ o ≠ 0) →
      t.mem r o = s.mem r o) : IO₁ₓ₂Frame io s t := by
  intro r o hm hv _
  apply h r o
  · rcases hm with hm | hm
    · exact Or.inl hm
    · exact Or.inr (by simpa [hw₁] using hm ⟨0, by omega⟩)
  · rcases hv with hv | hv
    · exact Or.inl hv
    · exact Or.inr (by simpa [hw₂] using hv ⟨0, by omega⟩)

/-- Successful execution supplies both typed output cells and the public
two-output frame. In particular, proving only the mean is insufficient. -/
theorem online_io_run {α : Type} [Inhabited α] (M : Algebra α)
    (x mean variance : RegionName) (stride : Nat) (xs : Fin N → α) (s : State α)
    (hd : mean ≠ variance)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * stride + i.val)).read .real = xs i) :
    IO₁ₓ₂PrivateScratch (onlineIO x mean variance N stride) ∧
    ∃ t, FP.Structural.exec M (onlineIO x mean variance N stride).kernel s = some t ∧
      t.mem mean 0 = .mk .bf16 (meanValue M xs) ∧
      t.mem variance 0 = .mk .bf16 (varianceValue M xs) ∧
      IO₁ₓ₂Frame (onlineIO x mean variance N stride) s t := by
  obtain ⟨t, ht, hm, hv, hf⟩ := online_run M x mean variance stride xs s hd hx
  refine ⟨by simp [IO₁ₓ₂PrivateScratch, onlineIO], t, ht, hm, hv, ?_⟩
  exact scalar_outputs_frame rfl rfl rfl rfl hf

theorem twopass_io_run {α : Type} [Inhabited α] (M : Algebra α)
    (x mean variance : RegionName) (stride : Nat) (xs : Fin N → α) (s : State α)
    (hd : mean ≠ variance)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * stride + i.val)).read .real = xs i) :
    IO₁ₓ₂PrivateScratch (twopassIO x mean variance N stride) ∧
    ∃ t, FP.Structural.exec M (twopassIO x mean variance N stride).kernel s = some t ∧
      t.mem mean 0 = .mk .bf16 (M.cast none .real .bf16 (twopassMean M xs)) ∧
      t.mem variance 0 = .mk .bf16 (M.cast none .real .bf16 (twopassVariance M xs)) ∧
      IO₁ₓ₂Frame (twopassIO x mean variance N stride) s t := by
  obtain ⟨t, ht, hm, hv, hf⟩ := twopass_run M x mean variance stride xs s hd hx
  refine ⟨by simp [IO₁ₓ₂PrivateScratch, twopassIO, onlineIO], t, ht, hm, hv, ?_⟩
  exact scalar_outputs_frame rfl rfl rfl rfl hf

/-- Conditional numerical result for the original online kernel, including
both bf16 stores and the full memory frame. The count laws are not admitted
yet, and the prefix tree is not assumed equal to an arbitrary batch schedule. -/
theorem fp32_online_statistics_run {α : Type} [Inhabited α]
    (R : FP.ScalarArithmetic.Rules) (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : FP.Guarded.Models R.assumptions M D) (s : State α)
    (x mean variance : RegionName) (N stride : Nat) (hN : 0 < N) (hdistinct : mean ≠ variance)
    (xs : Nat → α) (empty : FP.Equational.ReductionTree 0)
    (hx : ∀ i : Fin N, (s.mem x (s.pids 0 * stride + i.val)).read .real = xs i.val)
    (hc : FP.WelfordInduction.CountConversion M N)
    (hd : FP.WelfordInduction.IterationDomain M D xs empty N) :
    IO₁ₓ₂PrivateScratch (onlineIO x mean variance N stride) ∧
    ∃ t, FP.Structural.exec (M.withDefaultPrecision .fp32)
        (onlineIO x mean variance N stride).kernel s = some t ∧
      t.mem mean 0 = .mk .bf16 (M.cast (some .fp32) .real .bf16
        (FP.WelfordInduction.statistics M (FP.WelfordInduction.rowPrefix xs N)
          (FP.WelfordInduction.tree empty N)).1) ∧
      t.mem variance 0 = .mk .bf16 (M.cast (some .fp32) .real .bf16
        (FP.ScalarArithmetic.div M
          (FP.WelfordInduction.statistics M (FP.WelfordInduction.rowPrefix xs N)
            (FP.WelfordInduction.tree empty N)).2
          (FP.WelfordReduction.count M (FP.WelfordInduction.tree empty N)))) ∧
      IO₁ₓ₂Frame (onlineIO x mean variance N stride) s t := by
  obtain ⟨hp, t, ht, hm, hv, hf⟩ := online_io_run (M.withDefaultPrecision .fp32)
    x mean variance stride (FP.WelfordInduction.rowPrefix xs N) s hdistinct hx
  have hs := fp32_recurrence_statistics R M D hM s xs empty N hN hc hd
  have hn := FP.WelfordInduction.converted_count R M D hM s empty N hc hd.initial.zero N le_rfl
  refine ⟨hp, t, ht, ?_, ?_, hf⟩
  · change t.mem mean 0 = .mk .bf16 (M.cast (some .fp32) .real .bf16
      (recurrence (M.withDefaultPrecision .fp32) (FP.WelfordInduction.rowPrefix xs N) N).1) at hm
    exact hm.trans (congrArg (fun acc : α × α => Cell.mk .bf16
      (M.cast (some .fp32) .real .bf16 acc.1)) hs)
  · change t.mem variance 0 = .mk .bf16 (M.cast (some .fp32) .real .bf16
      (FP.ScalarArithmetic.div M
        (recurrence (M.withDefaultPrecision .fp32) (FP.WelfordInduction.rowPrefix xs N) N).2
        (M.fromNat (some .fp32) N))) at hv
    simpa only [hs, hn] using hv

end VeriTile.Bench.Examples.WelfordFPExecution
