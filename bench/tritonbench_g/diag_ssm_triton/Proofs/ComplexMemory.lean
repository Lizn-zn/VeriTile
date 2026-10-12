import bench.tritonbench_g.diag_ssm_triton.Proofs.Reference
import VeriTile.Triton.KernelLemmas.ScatterStore
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexMemory
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 1000000

/-- Interleaved complex read: adjacent real cells form one complex value. -/
def read (u : BlockState) (r : RegionName) (off : Nat) : ℂ :=
  ⟨u.readMem r (off*2),u.readMem r (off*2+1)⟩


@[simp] theorem read_setReg (u : BlockState) (name : RegName) (d : TileDType)
    (sh : TileShape) (v : Tile d sh) (r : RegionName) (off : Nat) :
    read (u.setReg name d sh v) r off = read u r off := rfl

/-- The source stores every real component first, then every imaginary
component. Values and masks are fixed before either scatter. -/
def store {B : Nat} (r : RegionName) (off : TileIndex [B] → Nat)
    (values : TileIndex [B] → ℂ) (mask : TileIndex [B] → Bool) (u : BlockState) : BlockState :=
  let realPart := (TileShape.allIndices [B]).foldl
    (fun acc j => if mask j then acc.writeMem r (off j*2) (values j).re else acc) u
  (TileShape.allIndices [B]).foldl
    (fun acc j => if mask j then acc.writeMem r (off j*2+1) (values j).im else acc) realPart

@[simp] theorem store_regs {B : Nat} (r : RegionName) (off : TileIndex [B] → Nat)
    (values : TileIndex [B] → ℂ) (mask : TileIndex [B] → Bool) (u : BlockState)
    (d : TileDType) (sh : TileShape) (name : RegName) :
    (store r off values mask u).regs d sh name = u.regs d sh name := by
  simp [store]

@[simp] theorem store_pids {B : Nat} (r : RegionName) (off : TileIndex [B] → Nat)
    (values : TileIndex [B] → ℂ) (mask : TileIndex [B] → Bool) (u : BlockState) :
    (store r off values mask u).pids = u.pids := by
  simp only [store,foldl_store_pids]

/-- Parity separates the two components; injectivity is needed only for
same-component lanes, and is supplied by the source column indices. -/
theorem store_read {B : Nat} (r : RegionName) (off : TileIndex [B] → Nat)
    (values : TileIndex [B] → ℂ) (mask : TileIndex [B] → Bool) (u : BlockState)
    (hi : Function.Injective off) (j : TileIndex [B]) (hm : mask j = true) :
    read (store r off values mask u) r (off j) = values j := by
  apply Complex.ext
  · change (store r off values mask u).readMem r (off j*2) = (values j).re
    unfold store
    rw [foldl_store_preserve _ _ _ _ _ _ (by intro k _ _; omega)]
    apply foldl_store_at (a := j)
    · exact TileShape.mem_allIndices [B] j
    · exact hm
    · rfl
    · intro k _ _ he; exact hi (by omega)
    · exact TileShape.allIndices_nodup [B]
  · change (store r off values mask u).readMem r (off j*2+1) = (values j).im
    unfold store
    apply foldl_store_at (a := j)
    · exact TileShape.mem_allIndices [B] j
    · exact hm
    · rfl
    · intro k _ _ he; exact hi (by omega)
    · exact TileShape.allIndices_nodup [B]

/-- Preserve full tagged cells away from both active component addresses. -/
theorem store_frame {B : Nat} (r : RegionName) (off : TileIndex [B] → Nat)
    (values : TileIndex [B] → ℂ) (mask : TileIndex [B] → Bool) (u : BlockState)
    (q : RegionName) (o : Nat)
    (h : q ≠ r ∨ ∀ j, mask j = true → off j*2 ≠ o ∧ off j*2+1 ≠ o) :
    (store r off values mask u).mem q o = u.mem q o := by
  unfold store
  refine (foldl_writeMem_frame _ _ _ q o _ ?_ _).trans
    (foldl_writeMem_frame _ _ _ q o _ ?_ _)
  · rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (fun j _ hm => (h j hm).2)
  · rcases h with h | h
    · exact Or.inl h
    · exact Or.inr (fun j _ hm => (h j hm).1)

theorem read_of_mem {s t : BlockState} {r : RegionName} {o : Nat}
    (hre : s.mem r (o*2) = t.mem r (o*2))
    (him : s.mem r (o*2+1) = t.mem r (o*2+1)) : read s r o = read t r o := by
  apply Complex.ext
  · change s.readMem r (o*2) = t.readMem r (o*2)
    unfold BlockState.readMem; rw [hre]
  · change s.readMem r (o*2+1) = t.readMem r (o*2+1)
    unfold BlockState.readMem; rw [him]

/-- Exact lane addresses are injective for any block size. -/
theorem lane_injective (B start : Nat) :
    Function.Injective (fun j : TileIndex [B] => start+j.1.val) := by
  intro ⟨j,jz⟩ ⟨k,kz⟩ h
  change start+j.val=start+k.val at h
  have hval : j=k := Fin.ext (by omega)
  subst k
  rw [Subsingleton.elim jz kz]

/-- Active lanes, rather than a block-size restriction, separate rows. -/
theorem time_separated (width t u a b : Nat)
    (ha : a<width) (hb : b<width) (htu : t≠u) : t*width+a ≠ u*width+b := by
  intro he
  rcases lt_or_gt_of_ne htu with hlt | hgt
  · have h := Nat.mul_le_mul_right width (Nat.succ_le_of_lt hlt)
    rw [Nat.succ_mul] at h
    omega
  · have h := Nat.mul_le_mul_right width (Nat.succ_le_of_lt hgt)
    rw [Nat.succ_mul] at h
    omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.ComplexMemory
