import bench.tritonbench_g.bmm_optimized.Proofs.RealLoads
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealDot

/-! Dot and accumulation consume the actual loaded values, including unspecified
masked lanes. Zero-fill assumptions enter only the separate matrix-sum lemma. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealDot
open VeriTile.Triton RealLoads
noncomputable section
set_option linter.unusedSimpArgs false
open BmmChunkBwd.RealDot (finite_dot)

def value {BM BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (i : Fin BM) (j : Fin BN) : ℝ :=
  ∑ k : Fin BK, aElement s p ap rows ks i k * bElement s p bp cols ks k j

theorem loaded_dot (BM BN BK : Nat) (s : BlockState) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat) :
    Tile.dot [] (aValues s p ap rows ks) (bValues s p bp cols ks) =
      Tile.mat (fun i j => some (value s p ap bp rows cols ks i j)) := by
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  exact finite_dot BM BK BN _ _ i j _ _ (by intro e; rfl) (by intro e; rfl)

/-- Register provenance is explicit, independently of any memory-domain assumption. -/
theorem accumulation_eval (swapped : Bool) (BM BN BK : Nat) (s origin : BlockState)
    (p : RealProgram.Parameters) (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (carry : Fin BM → Fin BN → ℝ)
    (hz : s.regs .real [BM,BN] "o" = some (Tile.mat fun i j => some (carry i j)))
    (ha : s.regs .real [BM,BK] "a" = some (aValues origin p ap rows ks))
    (hb : s.regs .real [BK,BN] "b" = some (bValues origin p bp cols ks)) :
    evalOp (RealLoads.update swapped BM BN BK) s =
      some (Tile.mat fun i j => some (carry i j + value origin p ap bp rows cols ks i j)) := by
  have hdot : evalOp (.dot (batch := []) (.ref .real [BM,BK] "a")
      (.ref .real [BK,BN] "b") { allowTF32 := some Bool.false }) s =
      some (Tile.mat fun i j => some (value origin p ap bp rows cols ks i j)) := by
    rw [evalOp_dot]
    simp [ha, hb, loaded_dot]
  cases swapped
  · simp only [RealLoads.update, Bool.false_eq_true, ↓reduceIte, evalOp_add,
      evalOp_ref, hz, bind, Option.bind_some]
    erw [hdot]
    rfl
  · simp only [RealLoads.update, ↓reduceIte, evalOp_add]
    erw [hdot]
    simp only [evalOp_ref, hz, bind, Option.bind_some]
    apply congrArg some
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    simp [Tile.bop, Tile.mat, NumericDType.add, Broadcast.leftIndex,
      Broadcast.rightIndex, add_comm]

/-- Masked loads have no `other` in the source. This premise is consequently
needed to identify their real value with a zero-padded contraction. -/
theorem value_zero_fill (BM BN BK : Nat) (s : BlockState) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (hu : ∀ r o, s.undef r o = 0) (i : Fin BM) (j : Fin BN) :
    value s p ap bp rows cols ks i j =
      ∑ k : Fin BK,
        (if RealMath.active p.divM (rows i) p.M ∧ RealMath.active p.divK (ks k) p.K
         then s.readMem (ap.data (i,k,PUnit.unit)).1 (ap.data (i,k,PUnit.unit)).2 else 0) *
        (if RealMath.active p.divK (ks k) p.K ∧ RealMath.active p.divN (cols j) p.N
         then s.readMem (bp.data (k,j,PUnit.unit)).1 (bp.data (k,j,PUnit.unit)).2 else 0) := by
  simp only [value, aElement, bElement, hu]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealDot
