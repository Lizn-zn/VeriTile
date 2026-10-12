import bench.tritonbench_g.diag_ssm_triton.Proofs.Erasure
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardSource
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000

/-- Complete static prefix: indices/mask, two transition loads and four
zero accumulators. Every fragment is extracted from the full source below. -/
def prelude (l : RegionName) (batch dim B : Nat) : List Stmt :=
  (Erasure.complexBackward Bool.false "s" l "y" "gs" "gx" "gl" "gy" 0 batch dim B).toAlgKernel.body.take 9

def body (s y gx gy : RegionName) (length batch dim B : Nat) : List Stmt :=
  match (Erasure.complexBackward Bool.false s "l" y "gs" gx "gl" gy length batch dim B).toAlgKernel.body.drop 9 with
  | .forLoop _ _ loop :: _ => loop
  | _ => []

def postlude (gs gl : RegionName) (B : Nat) : List Stmt :=
  (Erasure.complexBackward Bool.false "s" "l" "y" gs "gx" gl "gy" 0 0 0 B).toAlgKernel.body.drop 10

theorem source_body (s l y gs gx gl gy : RegionName) (length batch dim B : Nat) :
    (Erasure.complexBackward Bool.false s l y gs gx gl gy length batch dim B).toAlgKernel.body =
      prelude l batch dim B ++ [.forLoop "i" length (body s y gx gy length batch dim B)] ++
        postlude gs gl B := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexBackwardSource
