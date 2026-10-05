import VeriTile.Triton.Math.Sinkhorn
import VeriTile.Triton.Memory.TraceProofs

/-! Sinkhorn normalization touches only numerical temporaries. Address
registers and all memory are preserved through every finite iteration. -/
namespace VeriTile.Triton.LogSinkhorn
open TraceProofs

private theorem body_safe (A B : Nat) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (body A B) s := by
  unfold body
  apply assign_cons (by simp [Op.SafeAt.eq_def])
  intro u
  apply assign_cons (by simp [Op.SafeAt.eq_def])
  intro v
  exact .nil_intro

private theorem body_reg (A B : Nat) (s t : BlockState)
    (h : stepStmts (body A B) s = some t) (d sh name)
    (hu : name ≠ "u") (hv : name ≠ "v") : t.regs d sh name = s.regs d sh name := by
  unfold body at h
  obtain ⟨a, ha, hrest⟩ := cons_inv h
  obtain ⟨b, hb, hnil⟩ := cons_inv hrest
  have hbt : b = t := by simpa using hnil
  obtain ⟨u, _, rfl⟩ := assign_inv ha
  obtain ⟨v, _, rfl⟩ := assign_inv hb
  subst t
  simp [BlockState.setReg, hu, hv]

theorem loop_traceSafe (A B N : Nat) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafe bounds (.forLoop "iter" N (body A B)) s := by
  rw [Stmt.TraceSafe]
  exact loop_safe (n := N) (P := fun _ => True) (fun _ _ _ _ => body_safe A B bounds _)
    (fun _ _ _ _ _ _ => trivial) N 0 s (by omega) trivial

theorem loop_register (A B N : Nat) (s t : BlockState)
    (h : stepStmt (.forLoop "iter" N (body A B)) s = some t)
    (d sh name) (hu : name ≠ "u") (hv : name ≠ "v") (hi : name ≠ "iter") :
    t.regs d sh name = s.regs d sh name := by
  apply loop_preserves (P := fun t => t.regs d sh name = s.regs d sh name)
    (n := N) (idx := "iter") (body := body A B) ?_ N 0 s t (by omega) rfl
    (by simpa only [stepForLoopAux.forLoop_unfold] using h)
  intro i a b _ ha hb
  rw [body_reg A B _ _ hb d sh name hu hv]
  simpa [BlockState.setReg, hi] using ha

/-- Row-major conversion between the public flat input vector and a matrix. -/
def matrixIndex (i : Fin (A * B)) : TileIndex [A, B] := (i.divNat, i.modNat, PUnit.unit)

def matrixTile (xs : Fin (A * B) → ℝ) : Tile .real [A, B] :=
  ⟨fun i => (xs (finProdFinEquiv (i.1, i.2.1)) : WithBot ℝ)⟩

@[simp] theorem matrixIndex_address (i : Fin (A * B)) :
    (matrixIndex i).1.val * B + (matrixIndex i).2.1.val = i.val := by
  simpa [matrixIndex, Fin.divNat, Fin.modNat, Nat.mul_comm, Nat.add_comm] using Nat.mod_add_div i.val B

theorem readMatrix_eq_tile (s : BlockState) (r : RegionName) (base A B : Nat)
    (xs : Fin (A * B) → ℝ) (hx : ∀ i, s.readMem r (base + i.val) = xs i) :
    readMatrix s r base A B = matrixTile xs := by
  ext i
  simpa [readMatrix, matrixTile, finProdFinEquiv, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm] using congrArg
    (fun x : ℝ => (x : WithBot ℝ)) (hx (finProdFinEquiv (i.1, i.2.1)))

theorem matrix_frame {s t : BlockState} {r : RegionName} {base A B : Nat}
    (hf : ∀ q o, (q ≠ r ∨ ∀ i : TileIndex [A, B], o ≠ base + i.1.val * B + i.2.1.val) →
      t.mem q o = s.mem q o) (q o)
    (h : q ≠ r ∨ ∀ i : Fin (A * B), o ≠ base + i.val) : t.mem q o = s.mem q o := by
  apply hf q o
  rcases h with h | h
  · exact Or.inl h
  · exact Or.inr (fun i => by simpa [finProdFinEquiv, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, Nat.mul_comm] using h (finProdFinEquiv (i.1, i.2.1)))

end VeriTile.Triton.LogSinkhorn
