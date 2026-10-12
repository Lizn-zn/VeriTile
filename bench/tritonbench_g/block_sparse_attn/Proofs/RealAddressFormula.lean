import bench.tritonbench_g.block_sparse_attn.Proofs.RealLoop

/-! Address arithmetic for the independent initial-array contract. Signed
CSR visits remain signed; only selected key blocks must have valid starts. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealAddressFormula
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def schedule (s : BlockState) (p : RealProgram.Parameters) : CSRMath.Schedule where
  layout := RealSetup.head s p % p.layouts
  rowStride := p.rowStride
  columnStride := p.columnStride
  queryBlock := s.pids 0

def channel (BD : Nat) (second : Bool) (d : Fin BD) : Nat :=
  if second then BD + d.val else d.val

theorem query_address (s : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (r : Fin BM) (d : Fin BD) :
    RealQuery.address (RealSetup.queryPointers s BM BD p) second r d =
      (p.Q, RealSetup.qBase s p + (s.pids 0 * BM + r.val) * p.qRow + channel BD second d) := by
  cases second <;> simp [RealQuery.address, RealSetup.queryPointers, RealSetup.rows,
    channel, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

theorem key_address (s : BlockState) (BN BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (start : Int) (hstart : 0 ≤ start) (j : Fin BN) (d : Fin BD) :
    RealKey.address (RealSetup.keyPointers s BN BD p) start p.kRow second d j =
      (p.K, RealSetup.kBase s p + (start + (j.val : Int)).toNat * p.kRow + channel BD second d) := by
  have hj : (0 : Int) ≤ j.val := by omega
  cases second <;> simp [RealKey.address, RealSetup.keyPointers, channel,
    Int.toNat_mul hstart (Int.natCast_nonneg p.kRow), Int.toNat_add hstart hj, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

theorem value_address (s : BlockState) (BN BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (start : Int) (hstart : 0 ≤ start) (j : Fin BN) (d : Fin BD) :
    RealValueLoad.address (RealSetup.valuePointers s BN BD p) start p.vRow second j d =
      (p.V, RealSetup.vBase s p + (start + (j.val : Int)).toNat * p.vRow + channel BD second d) := by
  have hj : (0 : Int) ≤ j.val := by omega
  cases second <;> simp [RealValueLoad.address, RealSetup.valuePointers, channel,
    Int.toNat_mul hstart (Int.natCast_nonneg p.vRow), Int.toNat_add hstart hj, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealAddressFormula
