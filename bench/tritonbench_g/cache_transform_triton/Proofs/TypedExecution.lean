import bench.tritonbench_g.cache_transform_triton.Proofs.FP
import VeriTile.Triton.Float.ProfiledSoundness

/-! Successful complete-source executions in every floating algebra. Prefill
selects the second sequence; decode executes the full 4×2 tile with only its
first row active. The inactive metadata value is deliberately negative.
Payloads remain arbitrary: no Real or IEEE arithmetic is used for them. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def payloadCell {α : Type} (dtype : TypedSource.Storage) (x : α) : Cell α :=
  match dtype with
  | .fp32 => .mk .real x
  | .fp16 => .mk .fp16 x
  | .bf16 => .mk .bf16 x

def prefillInitial {α : Type} [Inhabited α] (cosType sinType : TypedSource.Storage)
    (payload : RegionName → Nat → α) : State α where
  mem := fun r offset => if r = "metadata" then .mk .int (if offset = 0 then 3 else 7)
    else payloadCell (if r = "cos" ∨ r = "cosOut" then cosType else sinType) (payload r offset)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun dim => if dim = 0 then 2 else 4
  undef := fun d _ _ => defaultValue d

def decodeInitial {α : Type} [Inhabited α] (cosType sinType : TypedSource.Storage)
    (payload : RegionName → Nat → α) : State α where
  mem := fun r _offset => if r = "metadata" then .mk .int 2
    else payloadCell (if r = "cos" ∨ r = "cosOut" then cosType else sinType) (payload r _offset)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => match d with
    | .int => -99
    | d => defaultValue d

def prefillOriginal (cosType sinType : TypedSource.Storage) :=
  FPEquivProofs.prefillOriginal cosType sinType "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut" 2 1 6 2 2 4
def prefillCandidate (cosType sinType : TypedSource.Storage) :=
  FPEquivProofs.prefillCommuted cosType sinType "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut" 2 1 6 2 2 4
def decodeOriginal (cosType sinType : TypedSource.Storage) :=
  FPEquivProofs.decodeOriginal cosType sinType "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut" 2 1 2 5 4
def decodeCandidate (cosType sinType : TypedSource.Storage) :=
  FPEquivProofs.decodeCommuted cosType sinType "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut" 2 1 2 5 4

/- Complete FP execution at arbitrary shape and state. Only the nonempty
integer-reduction axis is required; floating operations remain uninterpreted. -/
section
variable {α : Type} [Inhabited α] (M : Algebra α) (cosType sinType : TypedSource.Storage)
    (cos sin : RegionName) (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride total H N B : Nat) (s : State α)

def prefillRun (hN : 0 < N) :
    {t : State α // FP.Structural.exec (engine M)
      (FPEquivProofs.prefillOriginal cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride total H N B).kernel
      s = some t} := by
  cases cosType <;> cases sinType
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (config := { maxSteps := 100000 }) [FPEquivProofs.prefillOriginal,
      TypedSource.prefill, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, numericLt, numericLe,
      numericEq, natLt, bop, State.setReg, Cell.read, ofFloat,
      toFloat, TileShape.axisDim, Region.cast, FP.Structural.store, State.masked_scatter_eq_filter,
      TileShape.discreteMaximum, hN]
    try rfl

def decodeRun :
    {t : State α // FP.Structural.exec (engine M)
      (FPEquivProofs.decodeOriginal cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride H N B).kernel
      s = some t} := by
  cases cosType <;> cases sinType
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (config := { maxSteps := 100000 }) [FPEquivProofs.decodeOriginal,
      TypedSource.decode, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType, numeric, numericLt, numericLe,
      numericEq, natLt, bop, State.setReg, Cell.read, ofFloat,
      toFloat, TileShape.axisDim, Region.cast, FP.Structural.store, State.masked_scatter_eq_filter]
    try rfl
end

/-- Concrete observations below specialize the complete execution theorem. -/
def prefillFinish {α : Type} [Inhabited α] (M : Algebra α) (cosType sinType : TypedSource.Storage)
    (payload : RegionName → Nat → α) :
    {t : State α // FP.Structural.exec (engine M) (prefillOriginal cosType sinType).kernel
      (prefillInitial cosType sinType payload) = some t} :=
  prefillRun M cosType sinType "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut" 2 1 6 2 2 4
    (prefillInitial cosType sinType payload) (by decide)

def decodeFinish {α : Type} [Inhabited α] (M : Algebra α) (cosType sinType : TypedSource.Storage)
    (payload : RegionName → Nat → α) :
    {t : State α // FP.Structural.exec (engine M) (decodeOriginal cosType sinType).kernel
      (decodeInitial cosType sinType payload) = some t} :=
  decodeRun M cosType sinType "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut" 2 1 2 5 4
    (decodeInitial cosType sinType payload)

/-- An empty atomic theory imposes no condition on the floating algebra. -/
theorem empty_rules_model {α : Type} [Inhabited α] (M : Algebra α) (D : FP.Guarded.Domain α) :
    FP.Guarded.Models FPEquivProofs.assumptions M D := by
  have same {lhs rhs : List FP.GuardedFragment}
      (h : Spec.Derivation FPEquivProofs.assumptions lhs rhs) : lhs = rhs := by
    induction h with
    | refl => rfl
    | atom entry member _ => simp [FPEquivProofs.assumptions] at member
    | symm _ ih => exact ih.symm
    | trans _ _ first second => exact first.trans second
    | frame before after _ ih => rw [ih]
  intro lhs rhs h s _ _ a b ha hb
  have eq : lhs = rhs := List.singleton_injective (same h)
  subst rhs
  rw [ha] at hb
  cases Option.some.inj hb
  rfl

/-- Successful full prefill execution on both sides at every modeled shape.
No floating guard or assumed whole-program equation is an input. -/
theorem prefill_complete_executes {α : Type} [Inhabited α]
    (M : Algebra α) (cosType sinType : TypedSource.Storage)
    (cos sin : RegionName) (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride total H N B : Nat) (s : State α) (hN : 0 < N) :
    ∃ t, FP.Structural.exec (engine M)
        (FPEquivProofs.prefillOriginal cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride total H N B).kernel s = some t ∧
      FP.Structural.exec (engine M)
        (FPEquivProofs.prefillCommuted cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride total H N B).kernel s = some t := by
  let result := prefillRun M cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride total H N B s hN
  refine ⟨result.val, result.property, ?_⟩
  have h := FP.ProfiledRewrite.sound (FPEquivProofs.prefill_equiv cosType sinType
    cos sin metadata cosOut sinOut rowStride hiddenStride total H N B)
  exact (h α M (fun _ _ => True) (empty_rules_model M _) s
    (by simp [FPEquivProofs.prefillOriginal])).symm.trans result.property

theorem decode_complete_executes {α : Type} [Inhabited α]
    (M : Algebra α) (cosType sinType : TypedSource.Storage)
    (cos sin : RegionName) (metadata : Region .int) (cosOut sinOut : RegionName)
    (rowStride hiddenStride H N B : Nat) (s : State α) :
    ∃ t, FP.Structural.exec (engine M)
        (FPEquivProofs.decodeOriginal cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride H N B).kernel s = some t ∧
      FP.Structural.exec (engine M)
        (FPEquivProofs.decodeCommuted cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride H N B).kernel s = some t := by
  let result := decodeRun M cosType sinType cos sin metadata cosOut sinOut rowStride hiddenStride H N B s
  refine ⟨result.val, result.property, ?_⟩
  have h := FP.ProfiledRewrite.sound (FPEquivProofs.decode_equiv cosType sinType
    cos sin metadata cosOut sinOut rowStride hiddenStride H N B)
  exact (h α M (fun _ _ => True) (empty_rules_model M _) s
    (by simp [FPEquivProofs.decodeOriginal])).symm.trans result.property

theorem prefill_original_executes {α : Type} [Inhabited α] (M : Algebra α)
    (cosType sinType : TypedSource.Storage) (payload : RegionName → Nat → α) :
    FP.Structural.exec (engine M) (prefillOriginal cosType sinType).kernel
      (prefillInitial cosType sinType payload) = some (prefillFinish M cosType sinType payload).val :=
  (prefillFinish M cosType sinType payload).property

theorem decode_original_executes {α : Type} [Inhabited α] (M : Algebra α)
    (cosType sinType : TypedSource.Storage) (payload : RegionName → Nat → α) :
    FP.Structural.exec (engine M) (decodeOriginal cosType sinType).kernel
      (decodeInitial cosType sinType payload) = some (decodeFinish M cosType sinType payload).val :=
  (decodeFinish M cosType sinType payload).property

theorem prefill_candidate_executes {α : Type} [Inhabited α] (M : Algebra α)
    (cosType sinType : TypedSource.Storage) (payload : RegionName → Nat → α) :
    FP.Structural.exec (engine M) (prefillCandidate cosType sinType).kernel
      (prefillInitial cosType sinType payload) = some (prefillFinish M cosType sinType payload).val := by
  have h := FP.ProfiledRewrite.sound (FPEquivProofs.prefill_equiv cosType sinType
    "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut" 2 1 6 2 2 4)
  exact (h α M (fun _ _ => True) (empty_rules_model M _) (prefillInitial cosType sinType payload)
    (by simp [FPEquivProofs.prefillOriginal])).symm.trans (prefill_original_executes M cosType sinType payload)

theorem decode_candidate_executes {α : Type} [Inhabited α] (M : Algebra α)
    (cosType sinType : TypedSource.Storage) (payload : RegionName → Nat → α) :
    FP.Structural.exec (engine M) (decodeCandidate cosType sinType).kernel
      (decodeInitial cosType sinType payload) = some (decodeFinish M cosType sinType payload).val := by
  have h := FP.ProfiledRewrite.sound (FPEquivProofs.decode_equiv cosType sinType
    "cos" "sin" ⟨"metadata"⟩ "cosOut" "sinOut" 2 1 2 5 4)
  exact (h α M (fun _ _ => True) (empty_rules_model M _) (decodeInitial cosType sinType payload)
    (by simp [FPEquivProofs.decodeOriginal])).symm.trans (decode_original_executes M cosType sinType payload)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.TypedExecution
