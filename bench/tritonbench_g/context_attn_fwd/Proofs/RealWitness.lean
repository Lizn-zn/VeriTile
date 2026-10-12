import bench.tritonbench_g.context_attn_fwd.Proofs.RealIO
namespace VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealWitness
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false

/-- A nonzero batch/head/block, grouped KV heads, a cached prefix, three visited
key blocks and a partial final key block. These constants establish feasibility;
they do not restrict the symbolic source theorem. -/
def layout : Layout where
  BM := 16
  BN := 16
  D := 16
  heads := 4
  kvGroup := 2
  qRow := 64
  qHead := 16
  qDim := 1
  kBatch := 1024
  kRow := 32
  kHead := 16
  kDim := 1
  vBatch := 1024
  vRow := 32
  vHead := 16
  vDim := 1
  outRow := 64
  outHead := 16
  outDim := 1

def regions : Regions := ⟨"q","k","v","out","start","sequence","prompt"⟩

def initial : BlockState where
  mem := fun r _ =>
    if r = "start" ∨ r = "prompt" then .of .int 1
    else if r = "sequence" then .of .int 33
    else .real 1
  regs := fun _ _ _ => none
  pids := fun d => if d = 0 then 1 else if d = 1 then 7 else 0
  numPids := fun d => if d = 0 then 2 else if d = 1 then 8 else 1
  undef := fun _ _ => 7

@[simp] theorem start_value (i : Nat) : (inputs initial regions).start i = 1 := rfl
@[simp] theorem prompt_value (i : Nat) : (inputs initial regions).prompt i = 1 := rfl
@[simp] theorem sequence_value (i : Nat) : (inputs initial regions).sequence i = 33 := rfl
@[simp] theorem launch_value : launch initial = ⟨1,7⟩ := rfl

def arrays : Inputs where
  q := fun _ => 1
  k := fun _ => 1
  v := fun _ => 1
  start := fun _ => 1
  sequence := fun _ => 33
  prompt := fun _ => 1

theorem input_values : inputs initial regions = arrays := rfl

def allocation : FlatAlloc where
  flat := "flat"
  regions := ["q","k","v","start","sequence","out","prompt"]
  base := fun r =>
    if r = "q" then 0 else if r = "k" then 4096 else if r = "v" then 8192
    else if r = "start" then 12288 else if r = "sequence" then 16384 else if r = "out" then 20480
    else 24576
  extent := fun r => if r ∈ (["q","k","v","start","sequence","out","prompt"] : List RegionName) then 4096 else 0

theorem disjoint : allocation.Disjoint := by
  intro r hr s hs hne
  simp only [allocation,List.mem_cons,List.not_mem_nil,or_false] at hr hs
  rcases hr with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  all_goals first | exact False.elim (hne rfl) | decide

theorem coverage : ∀ r, r ∉ allocation.regions → allocation.extent r = 0 := by
  intro r hr
  simpa only [allocation] using (if_neg hr : (if r ∈ allocation.regions then 4096 else 0) = 0)

theorem valid : RealIO.valid layout (launch initial) (inputs initial regions) := by
  intro i j hi hj he
  obtain ⟨i,d,⟨⟩⟩ := i
  obtain ⟨j,e,⟨⟩⟩ := j
  have hi := i.isLt
  have hj := j.isLt
  have hd := d.isLt
  have he' := e.isLt
  change i.val < 16 at hi
  change j.val < 16 at hj
  change d.val < 16 at hd
  change e.val < 16 at he'
  norm_num [RealIO.write,AddressBounds.outputAddress,queryRow,Formula.batch,Formula.head,layout] at he
  have hij : i.val = j.val := by omega
  have hde : d.val = e.val := by omega
  simp only [Fin.ext hij,Fin.ext hde]

theorem bounds : AddressBounds.Valid layout regions (launch initial) (inputs initial regions) allocation.extent := by
  rw [input_values]
  constructor
  all_goals
    norm_num [AddressBounds.inside,AddressBounds.queryAddress,
      AddressBounds.keyAddress,AddressBounds.valueAddress,AddressBounds.outputAddress,
      Formula.sequence,Formula.limit,Formula.count,Formula.batch,Formula.head,queryRow,layout,regions,allocation,arrays,String.reduceEq]
  all_goals intros
  all_goals simp only [layout] at *
  all_goals omega

/-- Complete finite flat execution exists for both source variants and all
storage types. No preinitialized registers or zero-undefined premise is used. -/
theorem executes (storage : Storage) (swapped : Bool) :
    ∃ final, exec (allocation.flattenKernel (kernel layout regions storage swapped 1).eraseDType.toAlgKernel)
      (allocation.flattenState initial) = some final := by
  obtain ⟨final,he,_,_⟩ := Flat.correct allocation disjoint coverage layout regions storage swapped 1 initial
    (by decide) valid bounds
  exact ⟨final,he⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnFwd.RealWitness
