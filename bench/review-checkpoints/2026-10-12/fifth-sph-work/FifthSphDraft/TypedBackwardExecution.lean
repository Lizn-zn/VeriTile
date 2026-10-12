import FifthSphDraft.TypedModel
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- Successful nonempty typed execution for every independent storage choice.
Only the abstract singleton algebra is used; memory tags and masks are retained. -/
theorem backward_executes (input destination upstream : Storage) :
    ∃ v,FP.Structural.exec (engine model)
      (backward input destination upstream Bool.false "x" "dx" "g" 2 9 36 1 12) (initial input destination upstream)=some v := by
  cases input <;> cases destination <;> cases upstream
  all_goals
    simp only [FP.Structural.exec,backward,backward_fp16_fp16_fp16,backward_fp16_fp16_bf16,backward_fp16_fp16_fp32,backward_fp16_bf16_fp16,backward_fp16_bf16_bf16,backward_fp16_bf16_fp32,backward_fp16_fp32_fp16,backward_fp16_fp32_bf16,backward_fp16_fp32_fp32,backward_bf16_fp16_fp16,backward_bf16_fp16_bf16,backward_bf16_fp16_fp32,backward_bf16_bf16_fp16,backward_bf16_bf16_bf16,backward_bf16_bf16_fp32,backward_bf16_fp32_fp16,backward_bf16_fp32_bf16,backward_bf16_fp32_fp32,backward_fp32_fp16_fp16,backward_fp32_fp16_bf16,backward_fp32_fp16_fp32,backward_fp32_bf16_fp16,backward_fp32_bf16_bf16,backward_fp32_bf16_fp32,backward_fp32_fp32_fp16,backward_fp32_fp32_bf16,backward_fp32_fp32_fp32,ComputeKernel.surfaceBody,Bool.false_eq_true,↓reduceIte]
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    apply assigns_exists
    case he =>
      simp only [↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast]
      first | exact ⟨_,rfl⟩ | exact ⟨default,True.intro⟩
    intro
    simp (maxSteps := 3000000) [run,step,FP.Structural.store,↓evalExpr,↓evalComputeOp,↓evalOp_unfold,↓State.setReg_same,↓State.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte,engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,numeric,natLt,numericLt,numericLe,numericEq,bop,storageType,ofFloat,toFloat,Region.cast,initial,↓masked_scatter_regs]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution
