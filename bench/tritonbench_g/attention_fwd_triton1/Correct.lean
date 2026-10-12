import bench.tritonbench_g.attention_fwd_triton1.Kernels
import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealIO
import VeriTile.Meta.StatementAudit

/-! Real correctness of the complete sources in Kernels.lean.
This is chunked linear attention, without softmax. For chunk c:
  H_c = Σ_{j<c} K_jᵀ V_j
  O_c = ((scale Q_c) K_cᵀ) V_c + (scale Q_c) H_c.
STORE saves H_c before its update; IFCOND skips its zero contribution at c=0.
Both switches remain parameters and both sources satisfy the same formulas.

Scope: contiguous head lanes (s_qt=BD, s_qd=1), T=NT*BT, BT>0 and BD≤s_ht.
The source has no boundary checks, so the flat allocation must cover every
read and write. O and H are separate from each other and Q/K/V. The contract
includes successful execution, active-address safety and the memory frame
for one program. Host grid composition and IEEE arithmetic are outside it. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.Correctness
open VeriTile.Triton
open scoped StreamMasked3DKernelIO₃ₓ₂

/-- Three input streams and two aggregate output windows, indexed by chunk
and matrix cell. H's write window is active only when STORE is enabled;
otherwise the frame preserves all H cells. -/
def originalIO (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) : StreamMasked3DKernelIO₃ₓ₂ where
  kernel := (TypedSource.original Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
    scale BT BD NT STORE IFCOND).eraseDType
  projection := RealProjection.original_projection_supported ..
  inp1 := Q
  inp2 := K
  inp3 := V
  out1 := O
  out2 := H
  T := NT
  B1 := BT * BD
  B2 := BD * BT
  B3 := BT * BD
  C1 := NT * (BT * BD)
  C2 := NT * (BD * BD)
  out1DType := .real
  out2DType := .real
  pre := fun _ _ _ => Q ≠ O ∧ K ≠ O ∧ V ≠ O ∧ Q ≠ H ∧ K ≠ H ∧ V ≠ H ∧ H ≠ O
  read1 := fun p _ _ c j => p * s_qh + (c.val * BT + j.val / BD) * BD + j.val % BD
  read2 := fun p _ _ c j => p * s_qh + j.val / BT + (c.val * BT + j.val % BT) * BD
  read3 := fun p _ _ c j => p * s_qh + (c.val * BT + j.val / BD) * BD + j.val % BD
  write1 := fun p _ _ j =>
    let chunk := Lane2D.decode j
    let cell := Lane2D.decode chunk.2.1
    p * s_qh + (chunk.1.val * BT + cell.1.val) * BD + cell.2.1.val
  write2 := fun p _ _ j =>
    let chunk := Lane2D.decode j
    let cell := Lane2D.decode chunk.2.1
    p * s_hh + (chunk.1.val * BD + cell.1.val) * s_ht + cell.2.1.val
  mask1 := fun _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ => True
  mask3 := fun _ _ _ _ _ => True
  writeMask1 := fun _ _ _ _ => True
  writeMask2 := fun _ _ _ _ => STORE = Bool.true

def scalarLeftIO (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) : StreamMasked3DKernelIO₃ₓ₂ :=
  { originalIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND with
    kernel := (TypedSource.scalarLeft Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType
    projection := RealProjection.scalarLeft_projection_supported .. }

/-- H contains the sum of KᵀV from chunks strictly before this one. -/
noncomputable def expectedCarry (NT BT BD : Nat)
    (ys : Fin NT → Fin (BD * BT) → ℝ) (zs : Fin NT → Fin (BT * BD) → ℝ)
    (c : Fin NT) (i j : Fin BD) : ℝ :=
  ∑ past : Fin c.val, ∑ t : Fin BT,
    ys (Fin.castLE (Nat.le_of_lt c.isLt) past) (Lane2D.encode (i, t, PUnit.unit)) *
      zs (Fin.castLE (Nat.le_of_lt c.isLt) past) (Lane2D.encode (t, j, PUnit.unit))

/-- O is the local contribution plus Q times the pre-update carry. -/
noncomputable def expectedOutput (scale : ℝ) (NT BT BD : Nat)
    (xs : Fin NT → Fin (BT * BD) → ℝ) (ys : Fin NT → Fin (BD * BT) → ℝ)
    (zs : Fin NT → Fin (BT * BD) → ℝ) (c : Fin NT) (i : Fin BT) (j : Fin BD) : ℝ :=
  (∑ t : Fin BT,
    (∑ e : Fin BD, (xs c (Lane2D.encode (i, e, PUnit.unit)) * scale) *
      ys c (Lane2D.encode (e, t, PUnit.unit))) * zs c (Lane2D.encode (t, j, PUnit.unit))) +
  ∑ e : Fin BD, (xs c (Lane2D.encode (i, e, PUnit.unit)) * scale) *
    expectedCarry NT BT BD ys zs c e j

noncomputable def expected (scale : ℝ) (NT BT BD : Nat)
    (xs : Fin NT → Fin (BT * BD) → ℝ) (ys : Fin NT → Fin (BD * BT) → ℝ)
    (zs : Fin NT → Fin (BT * BD) → ℝ) :
    (Fin (NT * (BT * BD)) → ℝ) × (Fin (NT * (BD * BD)) → ℝ) :=
  (fun lane =>
    let chunk := Lane2D.decode lane
    let cell := Lane2D.decode chunk.2.1
    expectedOutput scale NT BT BD xs ys zs chunk.1 cell.1 cell.2.1,
   fun lane =>
    let chunk := Lane2D.decode lane
    let cell := Lane2D.decode chunk.2.1
    expectedCarry NT BT BD ys zs chunk.1 cell.1 cell.2.1)

/-- The original implements the two independent formulas for all flag combinations. -/
specification linear_attention_correctness (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool)
    (hBT : 0 < BT) (hstride : BD ≤ s_ht) :
    Spec.Real (originalIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND ⊨
      fun _ _ _ xs ys zs => expected scale NT BT BD xs ys zs) := by
  exact RealIO.original_correct Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND
    hBT hstride

/-- The scalar-left candidate implements the same formulas and output masks. -/
specification linear_attention_scalar_left_correctness (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool)
    (hBT : 0 < BT) (hstride : BD ≤ s_ht) :
    Spec.Real (scalarLeftIO Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND ⊨
      fun _ _ _ xs ys zs => expected scale NT BT BD xs ys zs) := by
  exact RealIO.scalarLeft_correct Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND
    hBT hstride

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.Correctness
