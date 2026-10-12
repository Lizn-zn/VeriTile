from pathlib import Path
import ast
w=Path(__file__).resolve().parent;p=w/'FlashAttnDraft'
a=ast.parse((w/'prepare_fp.py').read_text());c={n.targets[0].id:ast.literal_eval(n.value) for n in a.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.targets[0],ast.Name)}
params=c['params'];args=c['args'];call=f'kernel input output swapped {args}'
s='''import FlashAttnDraft.Kernels
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
'''
s+=f'''\ndef loopBody {params} (swapped : Bool) : List ComputeStmt :=
  ((Source.dynamicRange? ({call}).surfaceBody "start_n").getD
    ⟨[],.constNat 0,.constNat 0,.constNat 0,[],[]⟩).body

/-- Q is loaded in its storage type, scaled, then cast to fp16. Both FP32
accumulators and both destination storage types remain observable. -/
theorem precision {params} (swapped : Bool) :
    (types ({call}).surfaceBody "q",
     types ({call}).surfaceBody "max",
     types ({call}).surfaceBody "denom",
     types ({call}).surfaceBody "out_buffer",
     storeTypes ({call}).surfaceBody)=
      ([storageType input,.fp16],[.real],[.real],[.real,.real],[.real,storageType output]) := by
  cases input <;> cases output <;> cases swapped <;> rfl

theorem loop_precision {params} (swapped : Bool) :
    (types (loopBody input output {args} swapped) "k",
     types (loopBody input output {args} swapped) "v",
     types (loopBody input output {args} swapped) "out_buffer",
     types (loopBody input output {args} swapped) "qk")=
      ([.fp16],[.fp16],[.real,.real],[.real,.real]) := by
  cases input <;> cases output <;> cases swapped <;> rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.SourceChecks
'''
(p/'SourceChecks.lean').write_text(s)
