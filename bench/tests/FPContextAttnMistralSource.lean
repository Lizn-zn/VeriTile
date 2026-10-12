import bench.tritonbench_g.context_attn_mistral.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.ContextAttnMistral.SourceTypes
open VeriTile.Triton
set_option maxRecDepth 20000
set_option maxHeartbeats 1000000

def selected (storage : Storage) (candidate : Bool) (BM BN D : Nat) (window : Int) :=
  (if candidate then scaleFirst storage else original storage)
    "q" "k" "v" 2 "start" "sequence" "out"
    8 2 1 8 2 1 8 2 1 8 2 1 2 window BM D BN

def dtype : Storage → TileDType
  | .fp16 => .fp16
  | .bf16 => .bf16
  | .fp32 => .real

def assignType? (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) : Option TileDType :=
  match code with
  | [] => none
  | .assign d _ out _ :: rest =>
    if out = name then match occurrence with
      | 0 => some d
      | k+1 => assignType? rest name k
    else assignType? rest name occurrence
  | _ :: rest => assignType? rest name occurrence

def loopType? (code : List ComputeStmt) (name : RegName) (occurrence : Nat := 0) :=
  (Source.intRange? code "start_n").bind (fun lp => assignType? lp.body name occurrence)
def storeType? (code : List ComputeStmt) := code.getLast?.bind fun st => match st with
  | .store d _ _ _ _ => some d
  | _ => none

/-- Both sources preserve all lower-precision loads, the probability conversion
before the second dot, and the accumulator conversion at the output store. -/
theorem floating_storage (storage : Storage) (candidate : Bool) (BM BN D : Nat) (window : Int) :
    let code := (selected storage candidate BM BN D window).surfaceBody
    assignType? code "q" = some (dtype storage) ∧
    loopType? code "k" = some (dtype storage) ∧
    loopType? code "v" = some (dtype storage) ∧
    loopType? code "p" 2 = some (dtype storage) ∧
    storeType? code = some (dtype storage) := by
  cases storage <;> cases candidate <;> exact ⟨rfl,rfl,rfl,rfl,rfl⟩

def assignedLoad? (code : List ComputeStmt) (name : RegName) : Option ComputeDType :=
  (code.drop (Source.prefixBefore code name).length).head?.bind fun st => match st with
  | .assign _ _ _ (.compute (.load precision _ _ _)) => some precision
  | _ => none

/-- The real carrier on the fp32 branch still carries fp32 load annotations. -/
theorem fp32_load_precision (candidate : Bool) (BM BN D : Nat) (window : Int) :
    let code := (selected .fp32 candidate BM BN D window).surfaceBody
    assignedLoad? code "q" = some .fp32 ∧
    ((Source.intRange? code "start_n").bind fun lp => assignedLoad? lp.body "k") = some .fp32 ∧
    ((Source.intRange? code "start_n").bind fun lp => assignedLoad? lp.body "v") = some .fp32 := by
  cases candidate <;> exact ⟨rfl,rfl,rfl⟩

/-- Metadata and the loop counter are signed on every storage branch. -/
theorem signed_metadata (storage : Storage) (candidate : Bool) (BM BN D : Nat) (window : Int) :
    let code := (selected storage candidate BM BN D window).surfaceBody
    assignType? code "cur_batch_in_all_start_index" = some .int ∧
    assignType? code "cur_batch_seq_len" = some .int ∧
    loopType? code "start_n" = some .int := by
  cases storage <;> cases candidate <;> exact ⟨rfl,rfl,rfl⟩

/-- Window subtraction happens in Int before the comparison, including when
queryRow < window. A Nat subtraction here would incorrectly clip to zero. -/
theorem signed_window (storage : Storage) (candidate : Bool) (BM BN D : Nat) (window : Int) :
    ((Source.intRange? (selected storage candidate BM BN D window).surfaceBody "start_n").bind
      fun lp => (lp.body.drop (Source.prefixBefore lp.body "qk" 4).length).head?) =
    some (.assign .real [BM,BN] "qk" (.alg
      (.where (.gt .int Broadcast.nil.consR.consL
        (.add .int .scalarL (.ref .int [] "start_n")
          (Op.expandDim ⟨0,by simp⟩ (.ref .nat [BN] "offs_n")).castNatToInt)
        (.sub .int .scalarR
          (Op.expandDim ⟨1,by simp⟩ (.ref .nat [BM] "offs_m")).castNatToInt (.constInt window)))
        (.ref .real [BM,BN] "qk")
        (.broadcast (.sub .real .nil (.const (0.0 : ℝ)) (.const (1e9 : ℝ))) [BM,BN])))) := by
  cases storage <;> cases candidate <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnMistral.SourceTypes
