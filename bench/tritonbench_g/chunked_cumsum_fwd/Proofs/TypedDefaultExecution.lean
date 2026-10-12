import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.TypedExecution
import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! An inhabited typed execution with the default positive-infinite upper bound. The model's
scan is identity and its transcendental operations return one; this is a
consistency witness, not a claim about GPU numerical values. Nonzero program
IDs visit both head and sequence tails, with mixed storage and both flags. -/
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.TypedDefaultExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open FPModel (model domain model_satisfies_rules)
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

abbrev initial := TypedExecution.initial

def original := FPProof.original .bf16 .bf16 .fp16 "dt" "A" "bias" "out" "sum"
  2 7 3 4 0 .posInf 21 3 1 1 1 24 4 8 1 24 4 8 1 Bool.true Bool.true 2 4

def candidate := FPProof.scaleFirst .bf16 .bf16 .fp16 "dt" "A" "bias" "out" "sum"
  2 7 3 4 0 .posInf 21 3 1 1 1 24 4 8 1 24 4 8 1 Bool.true Bool.true 2 4

def finish : {t : State ℝ // FP.Structural.exec (engine model) original.kernel initial = some t} := by
  refine ⟨?_,?_⟩
  swap
  simp only [original,FPProof.original,VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.original,source,finiteSource,infiniteSource,
    ComputeKernel.surfaceBody,↓reduceIte]
  simp (config := {maxSteps := 150000}) [FP.Structural.exec,
    run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    engine,Algebra.withDefaultPrecision,resolvePrecision,model,ComputeDType.eraseDType,
    numeric,natLt,natLe,numericLt,numericLe,bop,initial,TypedExecution.initial,ofFloat,toFloat,TileShape.axisDim,Region.cast,State.setReg,
    FP.Structural.store,integralFloorDiv,integralMod]
  try rfl

theorem domain_holds : ∀ site ∈ original.domain, site.Holds model domain initial := by
  simp [original,FPProof.original,VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.original,source,finiteSource,infiniteSource,
    ComputeKernel.surfaceBody,Source.prefixBefore,Site.Holds,domain]

private theorem read_rate_cell (P : Prop) [Decidable P] (h : ¬ P) :
    Cell.read (α := ℝ) .fp16 (if P then Cell.mk .bf16 1 else Cell.mk .fp16 2) = 2 := by
  rw [if_neg h]
  rfl

theorem values :
    (FP.Structural.exec (engine model) original.kernel initial).map
      (fun t => (t.mem "out" 44,t.mem "out" 47,t.mem "sum" 44,t.mem "sum" 47,t.mem "sum" 48)) =
      some (.mk .real 1,.mk .real 0,.mk .real 2,.mk .real 0,.mk .real 99) := by
  rw [finish.property]
  simp only [Option.map_some,Option.some.injEq,Prod.mk.injEq]
  repeat' apply And.intro
  all_goals
    (conv_lhs => whnf) <;>
    norm_num [bop,Function.comp_def,numeric,ofFloat,toFloat,Broadcast.leftIndex,Broadcast.rightIndex,
      TileShape.dropInsertedIndex,State.write,State.setReg,TileShape.allIndices,List.finRange,
      List.ofFn_succ,String.reduceEq]
  all_goals
    rw [read_rate_cell _ (by decide)]

theorem candidate_execution (R : FP.ScalarArithmetic.Rules) :
    FP.Structural.exec (engine model) candidate.kernel initial = some finish.val := by
  have he := FP.ProfiledRewrite.sound (FPProof.equivalent .bf16 .bf16 .fp16 "dt" "A" "bias" "out" "sum"
    2 7 3 4 0 .posInf 21 3 1 1 1 24 4 8 1 24 4 8 1 Bool.true Bool.true 2 4 R)
  exact (he ℝ model domain (model_satisfies_rules R) initial domain_holds).symm.trans finish.property

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.TypedDefaultExecution
