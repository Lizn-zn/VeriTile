import bench.tritonbench_g.bmm_optimized.Proofs.RealStep

/-! The actual K loop starts at the complete source prefix and runs all
ceiling-divided iterations. The contraction formula refers to initial memory.
Because the source omits `other`, zero unspecified values are an explicit
premise of this formula theorem, not a change to the program. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealLoop
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

def aPointers (s₀ : BlockState) (BM BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) : Tile .ptr [BM,BK] :=
  Tile.mat fun i k => (p.A, RealSetup.aBase s₀ p + RealSetup.rows s₀ BM p i * p.K + (c * BK + k.val))
def bPointers (s₀ : BlockState) (BN BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) : Tile .ptr [BK,BN] :=
  Tile.mat fun k j => (p.B, RealSetup.bBase s₀ p + (c * BK + k.val) * p.N + RealSetup.columns s₀ BN p j)

def aEntry (s₀ : BlockState) (BM : Nat) (p : RealProgram.Parameters)
    (i : Fin BM) (k : Nat) : ℝ :=
  s₀.readMem p.A (RealSetup.aBase s₀ p + RealSetup.rows s₀ BM p i * p.K + k)
def bEntry (s₀ : BlockState) (BN : Nat) (p : RealProgram.Parameters)
    (j : Fin BN) (k : Nat) : ℝ :=
  s₀.readMem p.B (RealSetup.bBase s₀ p + k * p.N + RealSetup.columns s₀ BN p j)

def carry (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) (i : Fin BM) (j : Fin BN) : ℝ :=
  RealMath.partialSum p.M p.N (RealSetup.rows s₀ BM p i) (RealSetup.columns s₀ BN p j)
    p.K BK c p.divM p.divN (aEntry s₀ BM p i) (bEntry s₀ BN p j)

def writes : List RegName := "_i" :: RealStep.writes

structure Holds (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) (s : BlockState) : Prop where
  count : c ≤ BmmChunkBwd.RealMath.blocks p.K BK
  mem : s.mem = s₀.mem
  undef : s.undef = s₀.undef
  pids : s.pids = s₀.pids
  numPids : s.numPids = s₀.numPids
  fixed : ∀ dtype shape name, name ∉ writes → s.regs dtype shape name =
    (RealSetup.result s₀ BM BN BK p).regs dtype shape name
  a : s.regs .ptr [BM,BK] "a_ptrs" = some (aPointers s₀ BM BK p c)
  b : s.regs .ptr [BK,BN] "b_ptrs" = some (bPointers s₀ BN BK p c)
  k : s.regs .nat [BK] "offs_k" = some (Tile.vec fun k => c * BK + k.val)
  acc : s.regs .real [BM,BN] "o" = some (Tile.mat fun i j => some (carry s₀ BM BN BK p c i j))

theorem initial (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    Holds s₀ BM BN BK p 0 (RealSetup.result s₀ BM BN BK p) := by
  have hf := RealSetup.state_frame s₀ BM BN BK p
  refine ⟨Nat.zero_le _, (by have hx := congrArg BlockState.mem hf; exact hx),
    (by have hx := congrArg BlockState.undef hf; exact hx),
    (by have hx := congrArg BlockState.pids hf; exact hx),
    (by have hx := congrArg BlockState.numPids hf; exact hx), ?_, ?_, ?_, ?_, ?_⟩
  · intros; rfl
  · simp [RealSetup.result, aPointers, RealSetup.aPointers]
  · simp [RealSetup.result, bPointers, RealSetup.bPointers]
  · cases hm : p.divM <;> cases hn : p.divN <;> simp [RealSetup.result, hm, hn]
  · simp [RealSetup.result, carry, RealMath.partial_zero]

theorem a_advance (s₀ : BlockState) (BM BK : Nat) (p : RealProgram.Parameters) (c : Nat) :
    RealStep.advance (aPointers s₀ BM BK p c) BK = aPointers s₀ BM BK p (c + 1) := by
  apply Tile.ext
  rintro ⟨i,k,⟨⟩⟩
  simp only [RealStep.advance, aPointers, Tile.mat, Prod.mk.injEq, true_and]
  ring

theorem b_advance (s₀ : BlockState) (BN BK : Nat) (p : RealProgram.Parameters) (c : Nat) :
    RealStep.advance (bPointers s₀ BN BK p c) (BK * p.N) = bPointers s₀ BN BK p (c + 1) := by
  apply Tile.ext
  rintro ⟨k,j,⟨⟩⟩
  simp only [RealStep.advance, bPointers, Tile.mat, Prod.mk.injEq, true_and]
  ring

theorem carry_succ (s₀ s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (c : Nat) (hBK : 0 < BK) (hdiv : p.divK = Bool.true → BK ∣ p.K)
    (hc : c < BmmChunkBwd.RealMath.blocks p.K BK)
    (hm : s.mem = s₀.mem) (hu : ∀ r o, s.undef r o = 0) (i : Fin BM) (j : Fin BN) :
    carry s₀ BM BN BK p c i j +
      RealDot.value s p (aPointers s₀ BM BK p c) (bPointers s₀ BN BK p c)
        (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p) (fun k => c * BK + k.val) i j =
      carry s₀ BM BN BK p (c+1) i j := by
  simp only [carry, RealMath.partial_succ]
  congr 1
  rw [RealDot.value_zero_fill BM BN BK s p (aPointers s₀ BM BK p c)
    (bPointers s₀ BN BK p c) (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p)
    (fun k => c * BK + k.val) hu]
  convert RealMath.masked_dot p.M p.N (RealSetup.rows s₀ BM p i) (RealSetup.columns s₀ BN p j)
    p.K BK c p.divM p.divN p.divK hBK hdiv hc (aEntry s₀ BM p i) (bEntry s₀ BN p j) using 1
  simp only [aPointers, bPointers, Tile.mat, aEntry, bEntry, BlockState.readMem, hm]

theorem iteration (swapped : Bool) (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (hdiv : p.divK = Bool.true → BK ∣ p.K) (hu : ∀ r o, s₀.undef r o = 0)
    (c : Nat) (s : BlockState) (hc : c < BmmChunkBwd.RealMath.blocks p.K BK)
    (h : Holds s₀ BM BN BK p c s) :
    ∃ t, stepStmts (RealLoads.body swapped BM BN BK p) (s.setReg "_i" .nat [] (Tile.scalar c)) = some t ∧
      Holds s₀ BM BN BK p (c+1) t := by
  let entry := s.setReg "_i" .nat [] (Tile.scalar c)
  have hm : p.divM = Bool.false → entry.regs .bool [BM] "mask_m" =
      some (Tile.vec fun i => decide (RealSetup.rows s₀ BM p i < p.M)) := by
    intro hdm
    have hv := h.fixed .bool [BM] "mask_m" (by decide)
    cases hdn : p.divN <;> simpa [entry, RealSetup.result, hdm, hdn] using hv
  have hn : p.divN = Bool.false → entry.regs .bool [BN] "mask_n" =
      some (Tile.vec fun j => decide (RealSetup.columns s₀ BN p j < p.N)) := by
    intro hdn
    have hv := h.fixed .bool [BN] "mask_n" (by decide)
    simpa [entry, RealSetup.result, hdn] using hv
  have he := RealStep.executes swapped BM BN BK entry p (aPointers s₀ BM BK p c)
    (bPointers s₀ BN BK p c) (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p)
    (fun k => c * BK + k.val) (carry s₀ BM BN BK p c) hm hn
    (by simpa [entry] using h.k) (by simpa [entry] using h.a)
    (by simpa [entry] using h.b) (by simpa [entry] using h.acc)
  have hf := RealStep.state_frame entry p (aPointers s₀ BM BK p c)
    (bPointers s₀ BN BK p c) (RealSetup.rows s₀ BM p) (RealSetup.columns s₀ BN p)
    (fun k => c * BK + k.val) (carry s₀ BM BN BK p c)
  refine ⟨_, he, Nat.succ_le_of_lt hc, (by have hx := congrArg BlockState.mem hf; exact hx.trans h.mem),
    (by have hx := congrArg BlockState.undef hf; exact hx.trans h.undef), (by have hx := congrArg BlockState.pids hf; exact hx.trans h.pids),
    (by have hx := congrArg BlockState.numPids hf; exact hx.trans h.numPids), ?_, ?_, ?_, ?_, ?_⟩
  · intro dtype shape name hname
    have hs := hname
    simp only [writes, List.mem_cons, not_or] at hs
    rw [RealStep.register_frame _ _ _ _ _ _ _ _ _ _ _ hs.2]
    simpa [entry, BlockState.setReg, hs.1] using h.fixed dtype shape name hname
  · simp [RealStep.result, a_advance]
  · simp [RealStep.result, b_advance]
  · simp only [RealStep.result, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, BlockState.setReg_same]
    congr 1
    apply Tile.ext
    rintro ⟨k,⟨⟩⟩
    simp only [Tile.vec]
    ring
  · simp only [RealStep.result, BlockState.setReg_same]
    congr 1
    apply Tile.ext
    rintro ⟨i,j,⟨⟩⟩
    exact congrArg some (carry_succ s₀ entry BM BN BK p c hBK hdiv hc h.mem
      (by intro r o; change s.undef r o = 0; rw [h.undef]; exact hu r o) i j)

def statement (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) : Stmt :=
  .forRangeDyn "_i" (.constNat 0) (.ref .nat [] "num_iters") (.constNat 1)
    (RealLoads.body swapped BM BN BK p)

theorem executes (swapped : Bool) (s₀ : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (hdiv : p.divK = Bool.true → BK ∣ p.K) (hu : ∀ r o, s₀.undef r o = 0) :
    ∃ t, stepStmt (statement swapped BM BN BK p) (RealSetup.result s₀ BM BN BK p) = some t ∧
      Holds s₀ BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK) t := by
  have resolve : stepStmt (statement swapped BM BN BK p) (RealSetup.result s₀ BM BN BK p) =
      stepForRangeAux "_i" 0 (BmmChunkBwd.RealMath.blocks p.K BK) 1
        (RealLoads.body swapped BM BN BK p) (RealSetup.result s₀ BM BN BK p) := by
    simp [statement, stepForRangeAux.forRangeDyn_unfold, RealSetup.result,
      evalOp.eq_def, bind, Option.bind]
  rw [resolve]
  obtain ⟨final,t,he,hf,ht⟩ := forRangeAux_inv
    (idx := "_i") (stop := BmmChunkBwd.RealMath.blocks p.K BK) (step := 1)
    (P := Holds s₀ BM BN BK p) (by decide)
    (fun c s hc h => iteration swapped s₀ BM BN BK p hBK hdiv hu c s hc h)
    0 (RealSetup.result s₀ BM BN BK p) (initial s₀ BM BN BK p)
  have final_eq : final = BmmChunkBwd.RealMath.blocks p.K BK := Nat.le_antisymm ht.count hf
  subst final
  exact ⟨t,he,ht⟩

/-- The finished carry is the independent matrix product over initial memory. -/
theorem full_sum (s₀ s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (hBK : 0 < BK) (h : Holds s₀ BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK) s) :
    s.regs .real [BM,BN] "o" = some (Tile.mat fun i j => some
      (if RealMath.active p.divM (RealSetup.rows s₀ BM p i) p.M ∧
          RealMath.active p.divN (RealSetup.columns s₀ BN p j) p.N then
        RealMath.contraction p.K (aEntry s₀ BM p i) (bEntry s₀ BN p j) else 0)) := by
  rw [h.acc]
  simp only [carry, RealMath.partial_complete _ _ _ _ _ _ _ _ hBK]

/-- Full source prefix and loop, from the initial state without prepared registers. -/
theorem prefix_executes (swapped : Bool) (s₀ : BlockState) (BM BN BK : Nat)
    (p : RealProgram.Parameters) (hBK : 0 < BK) (hdiv : p.divK = Bool.true → BK ∣ p.K)
    (hu : ∀ r o, s₀.undef r o = 0) :
    ∃ t, stepStmts
      (RealProgram.realPart (RealProgram.parts swapped BM BN BK p).before ++
        [.forRangeDyn "_i" (.constNat 0) (.ref .nat [] "num_iters") (.constNat 1)
          (RealProgram.realPart (RealProgram.parts swapped BM BN BK p).body)]) s₀ = some t ∧
      Holds s₀ BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK) t := by
  obtain ⟨t,he,ht⟩ := executes swapped s₀ BM BN BK p hBK hdiv hu
  refine ⟨t,?_,ht⟩
  rw [RealSetup.original_candidate_setup, RealLoads.matches_source,
    stepStmts.append_some (RealSetup.executes s₀ BM BN BK p)]
  dsimp only [statement] at he
  rw [stepStmts.cons_some he]
  exact stepStmts.nil

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealLoop
