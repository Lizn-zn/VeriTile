import bench.tritonbench_g.diag_ssm_triton.Proofs.Erasure
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardSource
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000

/-- The complete source prefix: column index/mask and four static loads.
The decomposition below checks that this is exactly the prefix of every
sequence length and every choice of the remaining buffer names. -/
def prelude (s l : RegionName) (batch dim block : Nat) : List Stmt :=
  (Erasure.complexForward Bool.false s "input" "output" l 0 batch dim block).toAlgKernel.body.take 7

/-- Extract the full source loop, including both stores and the final tuple
assignment; no arithmetic expression is replaced during this decomposition. -/
def body (x y : RegionName) (batch dim block : Nat) : List Stmt :=
  match (Erasure.complexForward Bool.false "initial" x y "transition" 0 batch dim block).toAlgKernel.body.drop 7 with
  | [.forLoop _ _ loop] => loop
  | _ => []

theorem source_body (s x y l : RegionName) (length batch dim block : Nat) :
    (Erasure.complexForward Bool.false s x y l length batch dim block).toAlgKernel.body =
      prelude s l batch dim block ++ [.forLoop "t" length (body x y batch dim block)] := by
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexForwardSource
