import FifthSphDraft.FP
import VeriTile.Triton.Float.ProfiledSoundness
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite TypedSource
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- An abstract singleton-valued backend. Storage tags, integer addresses and
masks remain observable; numerical operations and casts return the sole value.
There are no FP atoms to satisfy. This only witnesses nonempty typed execution,
not physical rounding, numerical accuracy or the real polynomial result. -/
def model : Algebra PUnit where
  literal := fun _ _ _ => PUnit.unit
  negInf := PUnit.unit
  binary := fun _ _ _ _ _ => PUnit.unit
  atomicAdd := fun _ => some (fun _ _ => PUnit.unit)
  unary := fun _ _ _ => PUnit.unit
  compareLe := fun _ _ => some (fun _ _ => Bool.true)
  compareLt := fun _ _ => some (fun _ _ => Bool.false)
  compareEq := fun _ _ => some (fun _ _ => Bool.true)
  cast := fun _ _ _ _ => PUnit.unit
  fromNat := fun _ _ => PUnit.unit
  fromInt := fun _ _ => PUnit.unit
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => PUnit.unit
  reduceSum := fun _ {_} _ _ _ _ => PUnit.unit
  reduceSumTyped := fun _ d {_} _ _ _ => some (fun _ => ofFloat d PUnit.unit)
  scan := fun _ {_} _ _ _ v => some v
  dot := fun _ {_} {_} {_} {_} _ _ => some (fun _ => PUnit.unit)
  dotAcc := fun _ _ _ out _ {_} {_} {_} {_} _ _ _ => some (fun _ => ofFloat out PUnit.unit)
  dotTyped := fun _ _ _ out _ {_} {_} {_} {_} _ _ => some (fun _ => ofFloat out PUnit.unit)


def storageType : Storage → FloatDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def initial (input destination upstream : Storage) : State PUnit where
  mem := fun r _ => if r="x" then .mk (storageType input).toTileDType (ofFloat (storageType input) PUnit.unit)
    else if r="g" then .mk (storageType upstream).toTileDType (ofFloat (storageType upstream) PUnit.unit)
    else .mk (storageType destination).toTileDType (ofFloat (storageType destination) PUnit.unit)
  regs := fun _ _ _ => none
  pids := fun ax => if ax=0 then 1 else 0
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

@[simp] theorem masked_scatter_regs {ι : Type} (l : List ι) (s : State PUnit)
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell PUnit)
    (active : ι → Prop) [DecidablePred active] :
    (l.foldl (fun t i => if active i then t.write (region i) (offset i) (value i) else t) s).regs=s.regs := by
  induction l generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons,ih]
    split <;> rfl

theorem assigns_exists {d sh name} {e : ComputeExpr d sh} {s : State PUnit} {rest : List ComputeStmt}
    (he : ∃ v,evalExpr (engine model) e s=some v)
    (hr : ∀ v,∃ t,run (engine model) rest (s.setReg name d sh v)=some t) :
    ∃ t,run (engine model) (.assign d sh name e :: rest) s=some t := by
  obtain ⟨v,hv⟩ := he
  obtain ⟨t,ht⟩ := hr v
  refine ⟨t,?_⟩
  simpa only [run,step,hv,Option.bind_some] using ht


#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.TypedExecution
