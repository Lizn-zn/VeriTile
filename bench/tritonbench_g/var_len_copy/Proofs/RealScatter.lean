import bench.tritonbench_g.var_len_copy.Proofs.RealStep
import bench.tritonbench_g.var_len_copy.Proofs.Math

namespace VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal
open VeriTile Triton

 theorem address_nat {B : Nat} (start current : Nat) (i : TileIndex [B]) :
    address (B := B) start current i = start + current + i.1.val := by
  simp only [address, ← Int.natCast_add, Int.toNat_natCast]

 theorem address_injective {B : Nat} (start current : Nat) :
    Function.Injective (address (B := B) start current) := by
  intro i j h
  simp only [address_nat] at h
  apply Prod.ext
  · exact Fin.ext (Nat.add_left_cancel h)
  · exact Subsingleton.elim _ _

 theorem scatter_value {B : Nat} (s : BlockState) (dst : RegionName)
    (start current : Nat) (length : Int) (v : TileIndex [B] → ℝ)
    (i : TileIndex [B]) (hi : (i.1.val : Int) < length) :
    (scatter s dst start current length v).readMem dst (start + current + i.1.val) = v i := by
  rw [← address_nat start current i]
  apply BlockState.scatter_readback_prop_masked_nd_of_true s (address start current) v
    (fun j ↦ active length j = Bool.true) i (by simpa [active] using hi)
  intro j _ he
  exact address_injective start current he

/-- Each iteration writes only its actual active lanes. -/
 theorem scatter_frame {B : Nat} (s : BlockState) (dst : RegionName)
    (start current : Nat) (length : Int) (v : TileIndex [B] → ℝ) :
    BlockState.WriteWithin
      (fun addr ↦ addr.1 = dst ∧ ∃ lane, lane < B ∧ (lane : Int) < length ∧
        addr.2 = start + current + lane) s (scatter s dst start current length v) := by
  apply BlockState.foldl_writeMem_prop_writeWithin dst (address start current) v
    (fun i ↦ active length i = Bool.true) (TileShape.allIndices [B]) s
  intro i _ hi
  exact ⟨rfl, i.1.val, i.1.isLt, of_decide_eq_true hi, address_nat start current i⟩

 theorem scatter_state {B : Nat} (s : BlockState) (dst : RegionName)
    (start current length : Int) (v : TileIndex [B] → ℝ) :
    (scatter s dst start current length v).regs = s.regs ∧
    (scatter s dst start current length v).pids = s.pids ∧
    (scatter s dst start current length v).numPids = s.numPids ∧
    (scatter s dst start current length v).undef = s.undef := by
  unfold scatter
  generalize TileShape.allIndices [B] = lanes
  induction lanes generalizing s with
  | nil => exact ⟨rfl,rfl,rfl,rfl⟩
  | cons i tail ih =>
    simp only [List.foldl_cons]
    by_cases hi : active length i = Bool.true
    · simpa only [if_pos hi, BlockState.writeMem_regs, BlockState.writeMem_pids,
        BlockState.writeMem_numPids, BlockState.writeMem_undef] using
        ih (s.writeMem dst (address start current i) (v i))
    · simpa only [if_neg hi] using ih s

end VeriTile.Bench.TritonBenchG.VarLenCopy.FullReal

#guard_msgs (drop info) in
#auditModuleAxioms
