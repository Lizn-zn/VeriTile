import bench.tritonbench_g.bmm_optimized.Proofs.RealDot

/-! One complete source iteration, including the masks, two loads, lane and
pointer advances, then the dot/add. No loop execution, dot result or zero-fill
premise is assumed. The origin of every loaded value remains explicit. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealStep
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

def advance {shape : TileShape} (ptrs : Tile .ptr shape) (offset : Nat) : Tile .ptr shape :=
  ⟨fun i => ((ptrs.data i).1, (ptrs.data i).2 + offset)⟩

def result {BM BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (carry : Fin BM → Fin BN → ℝ) : BlockState :=
  let t := RealLoads.result s p ap bp rows cols ks
  let t := t.setReg "offs_k" .nat [BK] (Tile.vec fun k => ks k + BK)
  let t := t.setReg "a_ptrs" .ptr [BM,BK] (advance ap BK)
  let t := t.setReg "b_ptrs" .ptr [BK,BN] (advance bp (BK * p.N))
  t.setReg "o" .real [BM,BN] (Tile.mat fun i j => some
    (carry i j + RealDot.value s p ap bp rows cols ks i j))

theorem executes (swapped : Bool) (BM BN BK : Nat) (s : BlockState) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (carry : Fin BM → Fin BN → ℝ)
    (hm : p.divM = Bool.false → s.regs .bool [BM] "mask_m" = some (Tile.vec fun i => decide (rows i < p.M)))
    (hn : p.divN = Bool.false → s.regs .bool [BN] "mask_n" = some (Tile.vec fun j => decide (cols j < p.N)))
    (hk : s.regs .nat [BK] "offs_k" = some (Tile.vec ks))
    (ha : s.regs .ptr [BM,BK] "a_ptrs" = some ap)
    (hb : s.regs .ptr [BK,BN] "b_ptrs" = some bp)
    (hz : s.regs .real [BM,BN] "o" = some (Tile.mat fun i j => some (carry i j))) :
    stepStmts (RealLoads.body swapped BM BN BK p) s =
      some (result s p ap bp rows cols ks carry) := by
  let sl := RealLoads.result s p ap bp rows cols ks
  let sk := sl.setReg "offs_k" .nat [BK] (Tile.vec fun k => ks k + BK)
  let sa := sk.setReg "a_ptrs" .ptr [BM,BK] (advance ap BK)
  let sb := sa.setReg "b_ptrs" .ptr [BK,BN] (advance bp (BK * p.N))
  have hfixed : ∀ dtype shape name, name ∉ ["mask_k","mask_a","mask_b","a","b"] →
      sl.regs dtype shape name = s.regs dtype shape name :=
    RealLoads.register_frame s p ap bp rows cols ks
  have hk' : sl.regs .nat [BK] "offs_k" = some (Tile.vec ks) :=
    (hfixed _ _ _ (by decide)).trans hk
  have ha' : sl.regs .ptr [BM,BK] "a_ptrs" = some ap :=
    (hfixed _ _ _ (by decide)).trans ha
  have hb' : sl.regs .ptr [BK,BN] "b_ptrs" = some bp :=
    (hfixed _ _ _ (by decide)).trans hb
  have hz' : sl.regs .real [BM,BN] "o" = some (Tile.mat fun i j => some (carry i j)) :=
    (hfixed _ _ _ (by decide)).trans hz
  have hks : evalOp (.add .nat .scalarR (.ref .nat [BK] "offs_k") (.constNat BK)) sl =
      some (Tile.vec fun k => ks k + BK) := by
    simp [evalOp.eq_def, hk', bind, Option.bind]
    rfl
  have has : evalOp (.ptrAdd .scalarR (.ref .ptr [BM,BK] "a_ptrs") (.constNat BK)) sk =
      some (advance ap BK) := by
    simp [evalOp.eq_def, sk, ha', bind, Option.bind]
    rfl
  have hbs : evalOp (.ptrAdd .scalarR (.ref .ptr [BK,BN] "b_ptrs")
      (.mul .nat .nil (.constNat BK) (.constNat p.N))) sa = some (advance bp (BK * p.N)) := by
    simp [evalOp.eq_def, sa, sk, hb', bind, Option.bind]
    rfl
  have hadd : evalOp (RealLoads.update swapped BM BN BK) sb =
      some (Tile.mat fun i j => some (carry i j + RealDot.value s p ap bp rows cols ks i j)) := by
    apply RealDot.accumulation_eval swapped BM BN BK sb s p ap bp rows cols ks carry
    · simpa [sb, sa, sk] using hz'
    · simp [sb, sa, sk, sl, RealLoads.result]
    · simp [sb, sa, sk, sl, RealLoads.result]
  rw [RealLoads.body,
    stepStmts.append_some (RealLoads.executes s BM BN BK p ap bp rows cols ks hm hn hk ha hb)]
  dsimp only [sb, sa, sk, sl] at hks has hbs hadd
  simp only [RealLoads.tailCode, stepStmts, stepStmt, hks, has, hbs, hadd, bind, Option.bind_some]
  rfl

theorem state_frame {BM BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (carry : Fin BM → Fin BN → ℝ) :
    {result s p ap bp rows cols ks carry with regs := s.regs} = s :=
  RealLoads.state_frame s p ap bp rows cols ks

def writes : List RegName := ["mask_k","mask_a","mask_b","a","b","offs_k","a_ptrs","b_ptrs","o"]

theorem register_frame {BM BN BK : Nat} (s : BlockState) (p : RealProgram.Parameters)
    (ap : Tile .ptr [BM,BK]) (bp : Tile .ptr [BK,BN])
    (rows : Fin BM → Nat) (cols : Fin BN → Nat) (ks : Fin BK → Nat)
    (carry : Fin BM → Fin BN → ℝ) (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ writes) :
    (result s p ap bp rows cols ks carry).regs dtype shape name = s.regs dtype shape name := by
  simp only [writes, List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  rcases hn with ⟨hk, hma, hmb, ha, hb, hoff, hap, hbp, ho⟩
  simp only [result, BlockState.setReg, hoff, hap, hbp, ho, ↓reduceIte]
  exact RealLoads.register_frame s p ap bp rows cols ks dtype shape name (by simp [hk,hma,hmb,ha,hb])

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealStep
