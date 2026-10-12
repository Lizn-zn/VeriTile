import bench.tritonbench_g.bmm_optimized.Kernels
import bench.tritonbench_g.bmm_optimized.Proofs.RealContract
import VeriTile.Meta.StatementAudit

/-! Real correctness of the complete batched matrix multiply and its dot-first
candidate, after precision erasure. A/B/O share fp32, fp16 or bf16 storage in
the typed source. The mathematical target uses independent initial arrays.

The per-program contract pins both runtime grid dimensions to the host's
ceiling-divided launch and admits only valid grid coordinates. It includes
active allocation bounds, successful flat execution and preservation of all
cells outside the output tile. The batch ID is limited by allocation bounds.
Positive tile/group sizes and divisibility justify every omitted mask.

The source's masked loads omit `other`. This real contract explicitly uses
zero unspecified load values, inherited from the shared IO contract; the
source does not guarantee that behavior. Proofs/UndefinedTail.lean refutes
removing that premise on a supported 32×32 tile with K=33. Address arithmetic
is mathematical; fixed-width overflow and launch-wide races are separate. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.Correctness
open VeriTile.Triton
open scoped VeriTile.Bench.TritonBenchG.BmmOptimized.RealIO
noncomputable section

/-- Buffer names, logical M/N/K, CTA grouping and all three mask-elision flags. -/
abbrev Parameters := RealProgram.Parameters
abbrev Inputs := RealIO.Inputs

/-- Each input window covers exactly K entries for the selected output rows
or columns. Tail rows/columns are masked. The contract requires disjoint A/B/O
allocations and zero unspecified load values; both actual num_programs values
must match the displayed grid. -/
def originalIO (dtype : TypedSource.Storage) (BM BN BK : Nat) (p : Parameters) : RealIO.IO BM BN p.K where
  kernel := (RealProgram.source dtype Bool.false BM BN BK p).eraseDType
  projection := by
    rw [RealProgram.storage_erasure]
    exact (RealViews.io Bool.false BM BN BK p).projection
  A := p.A
  B := p.B
  O := p.O
  gridX := BmmChunkBwd.RealMath.blocks p.M BM
  gridY := BmmChunkBwd.RealMath.blocks p.N BN
  aWindow := fun px py pb i => pb * p.M * p.K + RealViews.row BM BN p px py i.1 * p.K + i.2.1.val
  bWindow := fun px py pb i => pb * p.K * p.N + i.1.val * p.N + RealViews.column BM BN p px py i.2.1
  outputWindow := fun px py pb i => pb * p.M * p.N + RealViews.row BM BN p px py i.1 * p.N + RealViews.column BM BN p px py i.2.1
  aMask := fun px py _ i => RealViews.row BM BN p px py i.1 < p.M
  bMask := fun px py _ i => RealViews.column BM BN p px py i.2.1 < p.N
  outputMask := fun px py _ i => RealViews.row BM BN p px py i.1 < p.M ∧ RealViews.column BM BN p px py i.2.1 < p.N

/-- Commute the fp32 carry and dot result in each actual K-loop iteration. -/
def dotFirstIO (dtype : TypedSource.Storage) (BM BN BK : Nat) (p : Parameters) : RealIO.IO BM BN p.K :=
  { originalIO dtype BM BN BK p with
    kernel := (RealProgram.source dtype Bool.true BM BN BK p).eraseDType
    projection := by
      rw [RealProgram.storage_erasure]
      exact (RealViews.io Bool.true BM BN BK p).projection }

/-- Ordinary matrix multiplication on initial input arrays. -/
def expected {BM BN K : Nat} (xs : Inputs BM BN K) (i : TileIndex [BM,BN]) : ℝ :=
  ∑ k : Fin K, xs.a i.1 k * xs.b k i.2.1

section
variable (dtype : TypedSource.Storage) (BM BN BK : Nat) (p : Parameters)
    (hBM : 0 < BM) (hBN : 0 < BN) (hBK : 0 < BK) (hGroup : 0 < p.groupM)
    (hDivM : p.divM = Bool.true → BM ∣ p.M)
    (hDivN : p.divN = Bool.true → BN ∣ p.N)
    (hDivK : p.divK = Bool.true → BK ∣ p.K)

include hBM hBN hBK hGroup hDivM hDivN hDivK

specification bmm_optimized_correctness :
    Spec.Real (originalIO dtype BM BN BK p ⊨ expected) := by
  simpa only [originalIO, RealProgram.storage_erasure] using
    RealContract.implements Bool.false BM BN BK p ⟨hBM,hBN,hBK,hGroup,hDivM,hDivN,hDivK⟩

specification bmm_dot_first_correctness :
    Spec.Real (dotFirstIO dtype BM BN BK p ⊨ expected) := by
  simpa only [dotFirstIO, originalIO, RealProgram.storage_erasure] using
    RealContract.implements Bool.true BM BN BK p ⟨hBM,hBN,hBK,hGroup,hDivM,hDivN,hDivK⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.Correctness
