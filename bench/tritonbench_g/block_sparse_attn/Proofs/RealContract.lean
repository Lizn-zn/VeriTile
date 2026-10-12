import bench.tritonbench_g.block_sparse_attn.Proofs.RealIO
import bench.tritonbench_g.block_sparse_attn.Proofs.RealFlat

/-! Connect the named IO windows to the checked complete source and transport
both channel branches into one contiguous output-channel contract. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealContract
open VeriTile.Triton
noncomputable section

def schedule (p : RealProgram.Parameters) (pid₀ pid₁ : Nat) : CSRMath.Schedule where
  layout := pid₁ % p.heads % p.layouts
  rowStride := p.rowStride
  columnStride := p.columnStride
  queryBlock := pid₀

/-- A negative signed visit is allowed if the complete column slot is valid.
Selected block indices are nonnegative; no first-visible-key premise occurs. -/
def validCSR (p : RealProgram.Parameters) (pid₀ pid₁ : Nat) (xs : RealIO.Inputs) : Prop :=
  let sched := schedule p pid₀ pid₁
  let slot := fun visit : Nat => ((sched.layout * p.columnStride : Nat) : Int) + CSRMath.first sched xs + (visit : Int)
  (∀ visit, visit < CSRMath.count sched xs → 0 ≤ slot visit) ∧
  (∀ visit, visit < CSRMath.count sched xs → 0 ≤ xs.columnIndex (slot visit))

theorem projection (storage : TypedSource.Storage) (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    (RealProgram.source storage swapped BM BN BD p).eraseDType.toAlgorithm? =
      .ok (RealProgram.source storage swapped BM BN BD p).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, RealProgram.source_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

def io (storage : TypedSource.Storage) (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) : RealIO.IO BM where
  kernel := (RealProgram.source storage swapped BM BN BD p).eraseDType
  projection := projection storage swapped BM BN BD p
  Q := p.Q
  K := p.K
  V := p.V
  rows := p.rows
  columns := p.cols
  output := p.Out
  context := p.context
  width := if 2 ≤ p.blocks then 2 * BD else BD
  evenM := p.evenM
  evenN := p.evenN
  rowSlot := fun pid₀ pid₁ next =>
    (pid₁ % p.heads % p.layouts) * p.rowStride + pid₀ + if next then 1 else 0
  columnSlot := fun pid₀ pid₁ xs visit =>
    (((pid₁ % p.heads % p.layouts) * p.columnStride : Nat) : Int) +
      xs.rowIndex (((pid₁ % p.heads % p.layouts) * p.rowStride + pid₀ : Nat) : Int) + (visit : Int)
  queryWindow := fun pid₁ r d => pid₁ / p.heads * p.qBatch + pid₁ % p.heads * p.qHead + r * p.qRow + d
  keyWindow := fun pid₁ r d => pid₁ / p.heads * p.kBatch + (pid₁ % p.heads / (p.heads / p.kvHeads)) * p.kHead + r.toNat * p.kRow + d
  valueWindow := fun pid₁ r d => pid₁ / p.heads * p.vBatch + (pid₁ % p.heads / (p.heads / p.kvHeads)) * p.vHead + r.toNat * p.vRow + d
  outputWindow := fun pid₀ pid₁ r d => pid₁ / p.heads * p.oBatch + pid₁ % p.heads * p.oHead + (pid₀ * BM + r.val) * p.oRow + d
  domain := validCSR p

def expected (BM BN BD : Nat) (p : RealProgram.Parameters) (pid₀ pid₁ : Nat)
    (xs : RealIO.Inputs) (r : Fin BM) (d : Nat) : ℝ :=
  CSRMath.attention (schedule p pid₀ pid₁) xs BN BD p.blocks (pid₀ * BM + r.val) d p.scale

theorem input_pins (storage : TypedSource.Storage) (swapped : Bool) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (xs : RealIO.Inputs)
    (h : (io storage swapped BM BN BD p).Pins s xs) : RealInputs.Memory s BD p xs :=
  ⟨h.rows, h.columns, h.query, h.key, h.value⟩

theorem valid_indices (storage : TypedSource.Storage) (swapped : Bool) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (xs : RealIO.Inputs)
    (h : (io storage swapped BM BN BD p).domain (s.pids 0) (s.pids 1) xs) : RealInputs.Valid s p xs :=
  ⟨h.1, h.2⟩

theorem address_bounds (storage : TypedSource.Storage) (swapped : Bool) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (xs : RealIO.Inputs) (bounds : RegionBounds)
    (h : (io storage swapped BM BN BD p).Bounds BN bounds (s.pids 0) (s.pids 1) xs) :
    RealWindows.Bounds bounds s BM BN BD p xs :=
  ⟨h.rows, h.columns, h.query, h.key, h.value, h.output⟩

/-- The actual source for each supported storage dtype implements the same
independent real formula. Precision erasure is confined to this real claim. -/
theorem implements (storage : TypedSource.Storage) (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (hN : 0 < BN) (hstride : BD ≤ p.oRow) (htwo : 2 ≤ p.blocks → 2 * BD ≤ p.oRow) :
    (io storage swapped BM BN BD p).Implements BN (expected BM BN BD p) := by
  intro A hd hcov s xs hv hu hw hm
  obtain ⟨t, he, hy⟩ := RealFlat.source_correct A hd hcov swapped s BM BN BD p xs
    (input_pins storage swapped s BM BN BD p xs hm) (valid_indices storage swapped s BM BN BD p xs hv)
    (address_bounds storage swapped s BM BN BD p xs A.extent hw) hu hN hstride htwo
  refine ⟨t, ?_, ?_, ?_⟩
  · change exec (A.flattenKernel (RealProgram.source storage swapped BM BN BD p).eraseDType.toAlgKernel)
      (A.flattenState s) = some t
    rw [RealProgram.storage_erasure]
    exact he
  · intro r d hd hr
    change d < RealInputs.width BD p at hd
    by_cases hfirst : d < BD
    · exact hy.first r ⟨d, hfirst⟩ hr
    · have hb : 2 ≤ p.blocks := by
        by_contra h
        simp only [RealInputs.width, h, ↓reduceIte] at hd
        omega
      have htail : d - BD < BD := by
        simp only [RealInputs.width, hb, ↓reduceIte] at hd
        omega
      have heq : BD + (d - BD) = d := by omega
      have h := hy.second hb r ⟨d - BD, htail⟩ hr
      simp only [heq] at h
      exact h
  · intro region address houtside
    apply hy.frame region address
    · intro r d hr hhit
      exact houtside r d.val (RealInputs.channel_lt BD p Bool.false d (by simp)) hr hhit
    · intro hb r d hr hhit
      exact houtside r (BD + d.val) (RealInputs.channel_lt BD p Bool.true d (fun _ => hb)) hr hhit

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealContract
