import FlashAttnDraft.Kernels
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FlashAttn.SourceChecks
open VeriTile.Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 15000

def storageType : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .real

def types (body : List ComputeStmt) (name : RegName) : List TileDType :=
  body.filterMap fun st => match st with
    | .assign d _ n _ => if n=name then some d else none
    | _ => none

def storeTypes (body : List ComputeStmt) : List TileDType :=
  body.filterMap fun st => match st with
    | .store d _ _ _ _ => some d
    | _ => none

def loopBody (input output : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) (swapped : Bool) : List ComputeStmt :=
  ((Source.dynamicRange? (kernel input output swapped Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal).surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩).body

/-- Q is loaded in its storage type, scaled, then cast to fp16. Both FP32
accumulators and both destination storage types remain observable. -/
theorem precision (input output : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) (swapped : Bool) :
    (types (kernel input output swapped Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal).surfaceBody "q",
     types (kernel input output swapped Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal).surfaceBody "max",
     types (kernel input output swapped Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal).surfaceBody "denom",
     types (kernel input output swapped Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal).surfaceBody "out_buffer",
     storeTypes (kernel input output swapped Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal).surfaceBody)=
      ([storageType input,.fp16],[.real],[.real],[.real,.real],[.real,storageType output]) := by
  cases input <;> cases output <;> cases swapped <;> rfl

theorem loop_precision (input output : Storage) (Q K V L O : RegionName) (sm_scale : ℝ)
    (stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
      stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
      stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
      stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
      BS HEAD SEQLEN BLOCK_M DIM BLOCK_N : Nat) (causal : Bool) (swapped : Bool) :
    (types (loopBody input output Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal swapped) "k",
     types (loopBody input output Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal swapped) "v",
     types (loopBody input output Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal swapped) "out_buffer",
     types (loopBody input output Q K V L O sm_scale stride_q_bs stride_q_head stride_q_seqlen stride_q_dim
    stride_k_bs stride_k_head stride_k_seqlen stride_k_dim
    stride_v_bs stride_v_head stride_v_seqlen stride_v_dim
    stride_o_bs stride_o_head stride_o_seqlen stride_o_dim
    BS HEAD SEQLEN BLOCK_M DIM BLOCK_N causal swapped) "qk")=
      ([.fp16],[.fp16],[.real,.real],[.real,.real]) := by
  cases input <;> cases output <;> cases swapped <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.SourceChecks
