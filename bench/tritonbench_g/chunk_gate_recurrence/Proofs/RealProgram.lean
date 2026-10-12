import bench.tritonbench_g.chunk_gate_recurrence.Proofs.RealMath
import VeriTile.Meta.StatementAudit

/-! Shared real addresses, tiles and input arrays for the complete recurrence
proofs. Dimensions and launch indices remain symbolic. These are mathematical
definitions; no carried register or kernel output is supplied as an input. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealProgram
open VeriTile.Triton
noncomputable section

structure Layout where
  N : Nat
  K : Nat
  V : Nat
  BK : Nat
  BV : Nat

structure Launch where
  head : Nat
  key : Nat
  value : Nat

def launch (u : BlockState) : Launch := ⟨u.pids 0, u.pids 1, u.pids 2⟩

abbrev Values (p : Layout) := Fin p.BK → Fin p.BV → ℝ
abbrev Inputs (p : Layout) := RealMath.Inputs p.BK p.BV

def address (p : Layout) (l : Launch) (c e j : Nat) : Nat :=
  l.head * p.N * p.K * p.V + l.key * p.V * p.BK + e * p.V +
    l.value * p.BV + j + c * p.K * p.V

def seedAddress (p : Layout) (l : Launch) (e j : Nat) : Nat :=
  l.head * p.K * p.V + l.key * p.V * p.BK + e * p.V + l.value * p.BV + j

def gateAddress (p : Layout) (l : Launch) (c : Nat) : Nat := l.head * p.N + c

def pointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) : Tile .ptr [p.BK,p.BV] :=
  ⟨fun i => (r, address p l c i.1.val i.2.1.val)⟩

def seedPointer (p : Layout) (l : Launch) (r : RegionName) : Tile .ptr [p.BK,p.BV] :=
  ⟨fun i => (r, seedAddress p l i.1.val i.2.1.val)⟩

def gatePointer (p : Layout) (l : Launch) (r : RegionName) (c : Nat) : Tile .ptr [] :=
  Tile.scalar (r, gateAddress p l c)

def tile (p : Layout) (f : Values p) : Tile .real [p.BK,p.BV] :=
  Tile.mat fun e j => some (f e j)

def scatter (p : Layout) (l : Launch) (r : RegionName) (c : Nat)
    (f : Values p) (u : BlockState) : BlockState :=
  (TileShape.allIndices [p.BK,p.BV]).foldl (fun st i =>
    st.writeMem r (address p l c i.1.val i.2.1.val) (f i.1 i.2.1)) u

theorem address_succ (p : Layout) (l : Launch) (c e j : Nat) :
    address p l (c+1) e j = address p l c e j + p.K * p.V := by
  simp only [address]; ring

theorem gateAddress_succ (p : Layout) (l : Launch) (c : Nat) :
    gateAddress p l (c+1) = gateAddress p l c + 1 := by
  simp only [gateAddress]; omega

theorem pointer_advance (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :
    Tile.ptrAdd .scalarR (pointer p l r c) (Tile.scalar (p.K * p.V)) =
      pointer p l r (c+1) := by
  apply Tile.ext
  rintro ⟨e,j,⟨⟩⟩
  simp [Tile.ptrAdd, Broadcast.leftIndex, Broadcast.rightIndex, pointer, address_succ]

theorem gatePointer_advance (p : Layout) (l : Launch) (r : RegionName) (c : Nat) :
    Tile.ptrAdd .nil (gatePointer p l r c) (Tile.scalar 1) = gatePointer p l r (c+1) := by
  apply Tile.ext
  intro i
  simp [Tile.ptrAdd, Broadcast.leftIndex, Broadcast.rightIndex, gatePointer, gateAddress_succ]

/-- Distinct chunks and tile lanes must not overwrite one another. The
ordinary unmasked block-fit conditions imply this property below. -/
def NoCollision (p : Layout) (l : Launch) : Prop :=
  ∀ (c d : Nat) (e k : Fin p.BK) (j v : Fin p.BV),
    address p l c e.val j.val = address p l d k.val v.val → c = d ∧ e = k ∧ j = v

private theorem radix_unique {b a c x y : Nat} (hx : x < b) (hy : y < b)
    (h : a*b+x = c*b+y) : a = c ∧ x = y := by
  have hb : 0 < b := by omega
  have hm := congrArg (fun n => n % b) h
  simp only [Nat.add_mod, Nat.mul_mod_left, Nat.zero_add,
    Nat.mod_eq_of_lt hx, Nat.mod_eq_of_lt hy] at hm
  subst y
  exact ⟨Nat.eq_of_mul_eq_mul_right hb (Nat.add_right_cancel h), rfl⟩

theorem noCollision (p : Layout) (l : Launch) (hk : p.BK ≤ p.K) (hv : p.BV ≤ p.V) :
    NoCollision p l := by
  intro c d e k j v heq
  have hj : j.val < p.V := lt_of_lt_of_le j.isLt hv
  have hval : v.val < p.V := lt_of_lt_of_le v.isLt hv
  have he : e.val < p.K := lt_of_lt_of_le e.isLt hk
  have hkey : k.val < p.K := lt_of_lt_of_le k.isLt hk
  have h : (c*p.K+e.val)*p.V+j.val = (d*p.K+k.val)*p.V+v.val := by
    dsimp [address] at heq
    nlinarith [heq]
  obtain ⟨hrows,hjv⟩ := radix_unique hj hval h
  obtain ⟨hcd,hek⟩ := radix_unique he hkey hrows
  exact ⟨hcd, Fin.ext hek, Fin.ext hjv⟩

theorem scatter_read (p : Layout) (l : Launch) (r : RegionName) (c : Nat)
    (f : Values p) (u : BlockState) (hi : NoCollision p l) (e : Fin p.BK) (j : Fin p.BV) :
    (scatter p l r c f u).readMem r (address p l c e.val j.val) = f e j := by
  apply BlockState.scatter_readback_nd (shape := [p.BK,p.BV]) u
    (fun i => address p l c i.1.val i.2.1.val) (fun i => f i.1 i.2.1) ?_ (e,j,PUnit.unit)
  rintro ⟨e₁,j₁,⟨⟩⟩ ⟨e₂,j₂,⟨⟩⟩ heq
  obtain ⟨_,he,hj⟩ := hi c c e₁ e₂ j₁ j₂ heq
  subst e₂; subst j₂; rfl

theorem scatter_other_chunk (p : Layout) (l : Launch) (r : RegionName) (c d : Nat)
    (f : Values p) (u : BlockState) (hi : NoCollision p l) (hcd : c ≠ d)
    (e : Fin p.BK) (j : Fin p.BV) :
    (scatter p l r c f u).readMem r (address p l d e.val j.val) =
      u.readMem r (address p l d e.val j.val) := by
  have h := BlockState.scatter_prop_masked_preserves_other_offset r
    (fun i : TileIndex [p.BK,p.BV] => address p l c i.1.val i.2.1.val)
    (fun i => f i.1 i.2.1) (fun _ => True) (address p l d e.val j.val)
    (fun i _ heq => hcd (hi c d i.1 e i.2.1 j heq).1)
    (TileShape.allIndices [p.BK,p.BV]) u
  simpa only [scatter, ite_true] using h

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkGateRecurrence.RealProgram
