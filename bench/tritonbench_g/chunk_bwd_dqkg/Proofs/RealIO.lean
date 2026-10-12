import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealFlat

/-! Reader-facing IO contract for all three gradients. Initial-memory arrays
feed an independent mathematical function; successful flat execution, active
outputs and full-cell frame are conclusions of the implementation relation. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealIO
open VeriTile.Triton RealMath
noncomputable section
set_option maxHeartbeats 500000

structure Outputs (p : Layout) where
  dq : Fin p.BT → Fin p.BK → ℝ
  dk : Fin p.BT → Fin p.BK → ℝ
  dg : Fin p.BT → ℝ

/-- Inputs are read from the INITIAL memory. Bounds describe source access
windows; output masks and addresses also determine the exact cell frame. -/
structure IO (p : Layout) where
  kernel : ComputeKernel
  projection : kernel.toAlgorithm? = .ok kernel.toAlgKernel
  read : BlockState → Inputs
  bounds : BlockState → RegionBounds → Prop
  dq : RegionName
  dk : RegionName
  dg : RegionName
  qkWindow : Launch → TileIndex [p.BT,p.BK] → Nat
  dgWindow : Launch → TileIndex [p.BT] → Nat
  qkMask : Launch → TileIndex [p.BT,p.BK] → Prop
  dgMask : Launch → TileIndex [p.BT] → Prop

def IO.writes {p : Layout} (io : IO p) (A : FlatAlloc) (l : Launch) : WriteFootprint :=
  fun addr => addr.1 = A.flat ∧
    ((∃ i, io.qkMask l i ∧ A.addr io.dq (io.qkWindow l i) = addr.2) ∨
     (∃ i, io.qkMask l i ∧ A.addr io.dk (io.qkWindow l i) = addr.2) ∨
     (∃ i, io.dgMask l i ∧ A.addr io.dg (io.dgWindow l i) = addr.2))

/-- Successful flat execution, all three mathematical outputs, and equality
of every other full tagged cell. Disjointness is between physical allocation
intervals; region inputs may share names when the same allocation is read. -/
def IO.Implements {p : Layout} (io : IO p) (f : Launch → Inputs → Outputs p) : Prop :=
  ∀ A : FlatAlloc, A.Disjoint → (∀ r, r ∉ A.regions → A.extent r = 0) →
  ∀ s : BlockState, s.undef = (fun _ _ => 0) → io.bounds s A.extent →
    ∃ final, exec (A.flattenKernel io.kernel.toAlgKernel) (A.flattenState s) = some final ∧
      (∀ i, io.qkMask (RealLoop.launch s) i →
        final.readMem A.flat (A.addr io.dq (io.qkWindow (RealLoop.launch s) i)) =
          (f (RealLoop.launch s) (io.read s)).dq i.1 i.2.1) ∧
      (∀ i, io.qkMask (RealLoop.launch s) i →
        final.readMem A.flat (A.addr io.dk (io.qkWindow (RealLoop.launch s) i)) =
          (f (RealLoop.launch s) (io.read s)).dk i.1 i.2.1) ∧
      (∀ i, io.dgMask (RealLoop.launch s) i →
        final.readMem A.flat (A.addr io.dg (io.dgWindow (RealLoop.launch s) i)) =
          (f (RealLoop.launch s) (io.read s)).dg i.1) ∧
      BlockState.WriteWithin (io.writes A (RealLoop.launch s)) (A.flattenState s) final

scoped notation:25 io " ⊨ " f => IO.Implements io f

section
variable (p : Layout) (qk valueStorage hist dhist : TypedSource.Storage) (swapped : Bool)
    (q k v h g dout dh dqOut dkOut dgOut : RegionName)

theorem projection :
    (RealSource.source qk valueStorage hist dhist swapped q k v h g dout dh dqOut dkOut dgOut
      p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
      p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType.toAlgorithm? =
    .ok (RealSource.source qk valueStorage hist dhist swapped q k v h g dout dh dqOut dkOut dgOut
      p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
      p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, RealSource.projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

def sourceIO : IO p where
  kernel := (RealSource.source qk valueStorage hist dhist swapped q k v h g dout dh dqOut dkOut dgOut
    p.s_k_h p.s_k_t p.s_v_h p.s_v_t p.s_h_h p.s_h_t p.scale
    p.T p.K p.V p.BT p.BK p.BV p.NT).eraseDType
  projection := projection p qk valueStorage hist dhist swapped q k v h g dout dh dqOut dkOut dgOut
  read := fun s => RealPost.inputs s q k v h g dout dh
  bounds := fun s => RealSafety.Bounds s p q k v h g dout dh dqOut dkOut dgOut
  dq := dqOut
  dk := dkOut
  dg := dgOut
  qkWindow := RealFrame.qkAddress p
  dgWindow := RealFrame.dgAddress p
  qkMask := RealFrame.qkActive p
  dgMask := RealFrame.dgActive p

def expected (p : Layout) (l : Launch) (a : Inputs) : Outputs p where
  dq := RealMath.dq p l a
  dk := RealMath.dk p l a
  dg := RealMath.dg p l a

theorem implements (hBV : 0 < p.BV) (hstride : p.K ≤ p.s_k_t)
    (hDqDk : dqOut ≠ dkOut) (hDqDg : dqOut ≠ dgOut) (hDkDg : dkOut ≠ dgOut) :
    sourceIO p qk valueStorage hist dhist swapped q k v h g dout dh dqOut dkOut dgOut ⊨ expected p := by
  intro A hd hcov s hu hb
  exact RealFlat.source_executes A hd hcov s p swapped qk valueStorage hist dhist
    q k v h g dout dh dqOut dkOut dgOut hu hBV hDqDk hDqDg hDkDg
    (RealFrame.qk_injective_on p (RealLoop.launch s) hstride) hb

end
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealIO
