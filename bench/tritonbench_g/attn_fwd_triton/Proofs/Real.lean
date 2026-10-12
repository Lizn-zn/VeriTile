import bench.tritonbench_g.attn_fwd_triton.Proofs.RealProjection

/-! Transfer the complete real IO contract through equality of both entire
precision-erased sources, retaining input windows, masks, successful execution,
independent weighted-sum formula and every cell outside active output writes. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealProofs
open VeriTile.Triton
noncomputable section
abbrev Parameters := AttnFwdCausal.RealContract.Parameters

def kernel (BM BN BD : Nat) (p : Parameters) (candidate : Bool) : ComputeKernel :=
  (RealProjection.source BM BN BD p candidate).eraseDType

theorem projection (BM BN BD : Nat) (p : Parameters) (candidate : Bool) :
    (kernel BM BN BD p candidate).toAlgorithm? = .ok (kernel BM BN BD p candidate).toAlgKernel := by
  rw [kernel, RealProjection.full_erased_source]
  exact AttnFwdCausal.RealContract.projection BM BN BD p candidate

def io (BM BN BD : Nat) (p : Parameters) (candidate : Bool) : AttnFwdCausal.RealIO.IO BM BN BD :=
  { AttnFwdCausal.RealContract.io BM BN BD p candidate with
    kernel := kernel BM BN BD p candidate
    projection := projection BM BN BD p candidate }

theorem implements (BM BN BD : Nat) (p : Parameters) (candidate : Bool) (hBN : 0 < BN) :
    (io BM BN BD p candidate).Implements
      (fun startM _ xs => AttnFwdCausal.RealContract.expected BM BN BD startM p xs) := by
  simpa only [io, kernel, RealProjection.full_erased_source] using
    AttnFwdCausal.RealContract.implements BM BN BD p candidate hBN

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealProofs
