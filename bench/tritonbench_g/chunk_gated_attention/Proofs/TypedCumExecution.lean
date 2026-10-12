import bench.tritonbench_g.chunk_gated_attention.Proofs.FP
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.FPModel
import VeriTile.Triton.Float.ProfiledSoundness

/-! Nonempty typed execution of both complete cumulative sources. The shared
abstract model returns one for dot. Mixed input/output storage, a partial tile
and nonzero program IDs are tested; real prefix values are proved separately. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.TypedCumExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open BmmChunkBwd.FPModel (domain)
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- Supply the option-aware dot used by the source with allow_tf32=false. -/
def model : Algebra ℝ :=
  { BmmChunkBwd.FPModel.model with
    dotWithOptions := fun _ _ {_} {_} {_} {_} _ _ => some (fun _ => 1) }

def initial : State ℝ where
  mem := fun r _ => if r = "s" then .mk .fp16 1 else .mk .bf16 99
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

def cumulative := FPEquivProofs.cumulative .fp16 .bf16 "s" "o" 15 3 1 5 3 4 2

def result : {t : State ℝ // FP.Structural.exec (engine model) cumulative.kernel initial = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp only [cumulative, FPEquivProofs.cumulative, TypedSource.cumOriginal,
    FP.Structural.exec, ComputeKernel.surfaceBody]
  simp (config := { maxSteps := 150000 }) [run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, Option.bind, engine, Algebra.withDefaultPrecision,
    resolvePrecision, model, BmmChunkBwd.FPModel.model, ComputeDType.eraseDType, numeric, natLe, bop, initial,
    ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg,
    FP.Structural.store, integralFloorDiv, integralMod]
  rfl

theorem values :
    (FP.Structural.exec (engine model) cumulative.kernel initial).map
      (fun t => (t.mem "o" 29, t.mem "o" 30, t.mem "s" 29)) =
      some (.mk .bf16 1, .mk .bf16 99, .mk .fp16 1) := by
  rw [result.property]
  simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq]
  repeat' apply And.intro
  all_goals (conv_lhs => whnf)

theorem domain_holds : ∀ site ∈ cumulative.domain, site.Holds model domain initial := by
  simp [cumulative, FPEquivProofs.cumulative]

theorem candidate_execution :
    FP.Structural.exec (engine model)
      (FPEquivProofs.cumulativeIndexFirst .fp16 .bf16 "s" "o" 15 3 1 5 3 4 2).kernel initial =
      some result.val := by
  have he := FP.ProfiledRewrite.sound
    (FPEquivProofs.cumulative_equiv .fp16 .bf16 "s" "o" 15 3 1 5 3 4 2)
  exact (he ℝ model domain (by apply FP.models_of_atoms; intro e h; simp at h) initial domain_holds).symm.trans result.property

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.TypedCumExecution
