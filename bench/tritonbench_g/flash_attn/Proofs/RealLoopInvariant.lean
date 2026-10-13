import bench.tritonbench_g.flash_attn.Proofs.RealInvariantStep
import VeriTile.Triton.KernelLemmas.LoopInvariant

/-! Complete source traversal, including a final partial logical key block.
The interpreter runs full physical tiles up to the ceiling of the stop bound. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula
noncomputable section

theorem loop_invariant (p : Layout) (s₀ : BlockState) (Q K V : RegionName)
    (t : BlockState) (hBN : 0 < p.BLOCK_N) (h : PrefixInvariant p s₀ Q K V 0 t) :
    ∃ u, stepStmt (.forRangeDyn "start_n" (.ref .nat [] "lo") (.ref .nat [] "hi")
      (.constNat p.BLOCK_N) (body p)) t = some u ∧
        PrefixInvariant p s₀ Q K V (Geometry.tiles (hi p (s₀.pids 0)) p.BLOCK_N) u := by
  let count := Geometry.tiles (hi p (s₀.pids 0)) p.BLOCK_N
  let P := fun n st => ∃ c, n = c*p.BLOCK_N ∧ c ≤ count ∧ PrefixInvariant p s₀ Q K V c st
  have hinit : P 0 t := ⟨0, by simp, Nat.zero_le _, h⟩
  have hstep : ∀ n st, n < hi p (s₀.pids 0) → P n st →
      ∃ u, stepStmts (body p) (st.setReg "start_n" .nat [] (Tile.scalar n)) = some u ∧
        P (n+p.BLOCK_N) u := by
    intro n st hn ⟨c,hc,hle,hinv⟩
    subst n
    have hclt : c < count := (Geometry.tile_start_lt _ _ _ hBN).mpr hn
    have hset := hinv.set_start_n p s₀ Q K V c st (c*p.BLOCK_N)
    obtain ⟨u,hu,hnext⟩ := invariant_body_step p s₀ Q K V c
      (st.setReg "start_n" .nat [] (Tile.scalar (c*p.BLOCK_N))) hBN hset (by simp)
    refine ⟨u,hu,c+1,?_,Nat.succ_le_of_lt hclt,hnext⟩
    simp [Nat.add_mul]
  obtain ⟨n,u,hu,hstop,c,hc,hle,hinv⟩ := forRangeDyn_inv
    (idx := "start_n") (startOp := .ref .nat [] "lo") (stopOp := .ref .nat [] "hi")
    (stepOp := .constNat p.BLOCK_N) (P := P)
    (by simpa only [evalOp_ref] using h.lo)
    (by simpa only [evalOp_ref] using h.hi)
    (by simp only [evalOp_constNat]) (Nat.ne_of_gt hBN) hinit hstep
  have hceq : c = count := by
    apply le_antisymm hle
    by_contra hnot
    have hlt : c < count := Nat.lt_of_not_ge hnot
    have hnlt := (Geometry.tile_start_lt (hi p (s₀.pids 0)) p.BLOCK_N c hBN).mp hlt
    omega
  subst c
  exact ⟨u,hu,hinv⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
