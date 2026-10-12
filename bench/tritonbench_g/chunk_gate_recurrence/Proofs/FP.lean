import bench.tritonbench_g.chunk_gate_recurrence.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Complete recurrence equivalence. Every reached multiply-add checks the
actual product and source tile in fp32, after storage-to-compute conversion.
Backward also checks its final DL update after the entire original loop.
The whole-tile DG reduction is unchanged. No product, reduction or cross-chunk
algebraic identity is assumed. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000

/-- The implicit storage conversion at an fp32 arithmetic operand. -/
def operand (storage : TypedSource.Storage) (shape : TileShape) (name : RegName) : Op .real shape :=
  match storage with
  | .fp32 => .ref .real shape name
  | .fp16 => .castFloat .fp16 .real (.ref .fp16 shape name)
  | .bf16 => .castFloat .bf16 .real (.ref .bf16 shape name)

def product (gate : TypedSource.Storage) (BK BV : Nat) (carry : RegName) : Op .real [BK,BV] :=
  .mul .real .scalarR (.ref .real [BK,BV] carry) (operand gate [] "d_i")

def forwardRequirements (kernel : ComputeKernel) (source gate : TypedSource.Storage)
    (BK BV : Nat) : List Site :=
  (do
    let loop ← Source.dynamicRange? kernel.surfaceBody "_i"
    return Site.range loop.before "_i" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "acc") [BK,BV]
        (product gate BK BV "acc") (operand source [BK,BV] "S_i"))).toList

def backwardRequirements (kernel : ComputeKernel) (gate incoming : TypedSource.Storage)
    (BK BV : Nat) : List Site :=
  (do
    let loop ← Source.dynamicRange? kernel.surfaceBody "_i"
    return Site.range loop.before "_i" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "Dacc") [BK,BV]
        (product gate BK BV "Dacc") (operand incoming [BK,BV] "DS_i"))).toList ++
  [.addFP32 (Source.prefixBefore kernel.surfaceBody "Dacc" 1) [BK,BV]
    (product gate BK BV "Dacc") (operand incoming [BK,BV] "DS_i")]

section
variable (source gate seed : TypedSource.Storage) (S D O last_kv : RegionName)
    (H N K V BK BV : Nat) (hasSeed : Bool)

def forward : Program where
  kernel := TypedSource.forward source gate seed S D O last_kv H N K V BK BV hasSeed
  domain := forwardRequirements
    (TypedSource.forward source gate seed S D O last_kv H N K V BK BV hasSeed) source gate BK BV

def forwardSumLeft : Program :=
  { forward source gate seed S D O last_kv H N K V BK BV hasSeed with
    kernel := TypedSource.forwardSumLeft source gate seed S D O last_kv H N K V BK BV hasSeed }

theorem forward_domain_present :
    (forward source gate seed S D O last_kv H N K V BK BV hasSeed).domain.length = 1 := by
  cases source <;> cases gate <;> rfl

theorem forward_equiv (R : FP.ScalarArithmetic.Rules) :
    forward source gate seed S D O last_kv H N K V BK BV hasSeed ≡[R]
    forwardSumLeft source gate seed S D O last_kv H N K V BK BV hasSeed := by
  cases source <;> cases gate <;> equiv_decompose <;> fp_prove
end

section
variable (source gate incoming : TypedSource.Storage) (S D DI DG DL DS : RegionName)
    (H N K V BK BV : Nat)

def backward : Program where
  kernel := TypedSource.backward source gate incoming S D DI DG DL DS H N K V BK BV
  domain := backwardRequirements
    (TypedSource.backward source gate incoming S D DI DG DL DS H N K V BK BV) gate incoming BK BV

def backwardSumLeft : Program :=
  { backward source gate incoming S D DI DG DL DS H N K V BK BV with
    kernel := TypedSource.backwardSumLeft source gate incoming S D DI DG DL DS H N K V BK BV }

theorem backward_domain_present :
    (backward source gate incoming S D DI DG DL DS H N K V BK BV).domain.length = 2 := by
  cases gate <;> cases incoming <;> rfl

theorem backward_equiv (R : FP.ScalarArithmetic.Rules) :
    backward source gate incoming S D DI DG DL DS H N K V BK BV ≡[R]
    backwardSumLeft source gate incoming S D DI DG DL DS H N K V BK BV := by
  cases gate <;> cases incoming <;> equiv_decompose (reverse := true) <;> fp_prove
end

#print_fp_assumptions forward_equiv
#print_fp_assumptions backward_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.FPEquivProofs
