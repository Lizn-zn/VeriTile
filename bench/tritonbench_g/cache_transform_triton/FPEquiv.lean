import bench.tritonbench_g.cache_transform_triton.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.Layout
import bench.tritonbench_g.cache_transform_triton.Proofs.FP
import VeriTile.Meta.StatementAudit

/-! Whole-program FP equivalence for both cache gathers. The candidate only
reverses the exact integer multiplication in token indexing. Both output
dtypes, signed metadata, masks and unspecified values are preserved. Floating
payloads remain opaque: no numerical atom or numerical guard is required.
Proofs/TypedExecution.lean also establishes successful complete executions
for arbitrary opaque payloads and all nine independent storage pairs. -/
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []

section
variable (p : Layout) (cosType sinType : TypedSource.Storage) (total N : Nat)

def prefillOriginal : Program where
  kernel := p.prefill cosType sinType Bool.false total N
  domain := []

def prefillIndexCommuted : Program :=
  { prefillOriginal p cosType sinType total N with
    kernel := p.prefill cosType sinType Bool.true total N }

def decodeOriginal : Program where
  kernel := p.decode cosType sinType Bool.false N
  domain := []

def decodeIndexCommuted : Program :=
  { decodeOriginal p cosType sinType N with kernel := p.decode cosType sinType Bool.true N }

specification prefill_index_commuted_equiv :
    prefillOriginal p cosType sinType total N ≡[assumptions]
    prefillIndexCommuted p cosType sinType total N := by
  exact FPEquivProofs.prefill_equiv cosType sinType p.cos p.sin p.metadata p.cosOut p.sinOut
    p.rowStride p.hiddenStride total p.H N p.B

specification decode_index_commuted_equiv :
    decodeOriginal p cosType sinType N ≡[assumptions] decodeIndexCommuted p cosType sinType N := by
  exact FPEquivProofs.decode_equiv cosType sinType p.cos p.sin p.metadata p.cosOut p.sinOut
    p.rowStride p.hiddenStride p.H N p.B
end

#print_fp_assumptions prefill_index_commuted_equiv
#print_fp_assumptions decode_index_commuted_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.Equivalence
