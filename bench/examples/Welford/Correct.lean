import bench.examples.Welford.Kernels
/- Real correctness of the original two-pass and online Welford kernels.
Both compute the population mean and variance, including empty input rows.
Output bf16 casts are retained in the source and erased for real correctness. -/
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.WelfordCorrect
open VeriTile.Bench.Examples.Welford.Kernels
open VeriTile Triton
open VeriTile.Triton.TiledReduction.WelfordRec
open VeriTile.Examples (InputRowLoadedAt onlineWelfordLoopBody)
open scoped VeriTile.Triton.KernelIO₁ₓ₂

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl


/-- Executable real counterpart of the two-pass source. -/

def twopassMathKernel (xReg meanReg varReg : RegionName)
    (blockSize rowStride : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(rowStride) + tl.arange($(blockSize))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(blockSize))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(blockSize))
  tl.store($(meanReg), μ)
  tl.store($(varReg), v)
}


def onlineMathKernel (xReg meanReg varReg : RegionName)
    (blockSize rowStride : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(blockSize) {
    xi     := tl.load($(xReg) + (pid * $(rowStride) + i))
    delta  := xi - M
    M      := M + delta / (tl.toReal(i) + 1)
    delta2 := xi - M
    S      := S + delta * delta2
  }
  tl.store($(meanReg), M)
  tl.store($(varReg), S / tl.toReal($(blockSize)))
}

variable (rowStride : Nat)

private def P_welford {N : Nat} (rowStride : Nat) (xs : Fin N → ℝ)
    (xReg : RegionName)
    (origPid : Nat) (k : Nat) (s : BlockState) : Prop :=
  s.regs .real [] "M" = some (Tile.scalar (welfordMean xs k))
  ∧ s.regs .real [] "S" = some (Tile.scalar (welfordS xs k))
  ∧ s.regs .nat [] "pid" = some (Tile.scalar origPid)
  ∧ s.pid = origPid
  ∧ InputRowLoadedAt s xReg rowStride N xs

private theorem online_welford_step
    {N : Nat} (xs : Fin N → ℝ) (xReg : RegionName) (origPid i : Nat)
    (s : BlockState) (hi : i < N)
    (hP : P_welford rowStride xs xReg origPid i s) :
    ∃ s',
      stepStmts (onlineWelfordLoopBody xReg rowStride)
        (s.setReg "i" .nat [] (Tile.scalar i)) = some s' ∧
      P_welford rowStride xs xReg origPid (i + 1) s' := by
  rcases hP with ⟨hM, hS, hpidReg, hpid, hX⟩
  let xi : ℝ := s.readMem xReg (origPid * rowStride + i)
  have hxi : xi = xs ⟨i, hi⟩ := by
    have hx := hX ⟨i, hi⟩
    rw [hpid] at hx
    exact hx
  let m : ℝ := welfordMean xs i
  let ssum : ℝ := welfordS xs i
  let delta : ℝ := xi - m
  let m' : ℝ := m + delta / ((i : ℝ) + 1)
  let delta2 : ℝ := xi - m'
  let ssum' : ℝ := ssum + delta * delta2
  let s' :=
    (((((s.setReg "i" .nat [] (Tile.scalar i)).setReg
      "xi" .real [] (Tile.scalar xi)).setReg
      "delta" .real [] (Tile.scalar delta)).setReg
      "M" .real [] (Tile.scalar m')).setReg
      "delta2" .real [] (Tile.scalar delta2)).setReg
      "S" .real [] (Tile.scalar ssum')
  refine ⟨s', ?_, ?_⟩
  · -- Reduce loop body via simp; remaining goal is structural equality of
    -- WithBot ℝ arithmetic terms (↑a + ↑b vs ↑(a + b) etc.) which matches via
    -- WithBot.coe_add / coe_mul in reverse, plus rfl on the outer setReg shell.
    simp [onlineWelfordLoopBody, stepStmts, stepStmt, Tile.bop,
      Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
      NumericDType.div, BlockState.readMem, hM, hS, hpidReg,
      xi, m, ssum, delta, m', delta2, ssum', s',
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
      BlockState.setReg]
    rfl
  · simp [P_welford, s', InputRowLoadedAt, welfordMean, welfordS, hi, xi, m,
      ssum, delta, m', delta2, ssum', hpidReg, hpid, hxi]
    intro j
    have hx := hX j
    rw [hpid] at hx
    exact hx

/-- A successful `assign` step only touches registers, never memory. -/
private theorem stepStmt_assign_mem {dtype : TileDType} {shape : TileShape}
    {name : RegName} {e : Op dtype shape} {t t' : BlockState}
    (h : stepStmt (Stmt.assign dtype shape name e) t = some t')
    (r : RegionName) (o : Nat) : t'.mem r o = t.mem r o := by
  simp only [stepStmt] at h
  cases hv : evalOp e t with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      injection h with h
      subst h
      rfl

/-- A successful run of a list of `assign` statements leaves memory unchanged. -/
private theorem stepStmts_assigns_mem :
    ∀ (l : List Stmt),
      (∀ st ∈ l, ∃ (dtype : TileDType) (shape : TileShape) (name : RegName)
        (e : Op dtype shape), st = Stmt.assign dtype shape name e) →
      ∀ t t' : BlockState, stepStmts l t = some t' →
        ∀ (r : RegionName) (o : Nat), t'.mem r o = t.mem r o := by
  intro l
  induction l with
  | nil =>
      intro _ t t' h r o
      simp only [stepStmts.nil, Option.some.injEq] at h
      subst h
      rfl
  | cons hd tl ih =>
      intro hall t t' h r o
      unfold stepStmts at h
      cases hhd : stepStmt hd t with
      | none => simp [hhd] at h
      | some tmid =>
          simp only [hhd] at h
          obtain ⟨dt, sh, nm, e, heq⟩ := hall hd (by simp)
          subst heq
          rw [ih (fun st hst => hall st (List.mem_cons_of_mem _ hst)) tmid t' h r o,
            stepStmt_assign_mem hhd r o]

/-- Every statement of the online Welford loop body is an `assign`. -/
private theorem onlineWelfordLoopBody_assigns (xReg : RegionName)
    (stride : Nat) :
    ∀ st ∈ onlineWelfordLoopBody xReg stride,
      ∃ (dtype : TileDType) (shape : TileShape) (name : RegName)
        (e : Op dtype shape), st = Stmt.assign dtype shape name e := by
  intro st hst
  simp only [onlineWelfordLoopBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl <;> exact ⟨_, _, _, _, rfl⟩

theorem welford_eq_two_pass_total {n : Nat} (x : Fin n → ℝ) :
    welfordMean x n = twoPassMean x ∧ welfordS x n = twoPassS x := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact ⟨by simp [welfordMean, twoPassMean], by simp [welfordS, twoPassS]⟩
  · exact welford_eq_two_pass hn x

/-- The three register initializations preceding the shared online loop. -/
def loopInitial (s : BlockState) : BlockState :=
  ((s.setReg "pid" .nat [] (Tile.scalar (s.pids 0))).setReg
    "M" .real [] (Tile.scalar (some 0))).setReg "S" .real [] (Tile.scalar (some 0))

/-- The online loop computes the two-pass statistics and leaves all memory
unchanged. This also supplies the loop proof for real LayerNorm correctness. -/
theorem welford_loop_run (N rowStride : Nat) (xReg : RegionName) (s : BlockState)
    (xs : Fin N → ℝ) (hx : InputRowLoadedAt s xReg rowStride N xs) :
    ∃ t, stepForLoopAux "i" 0 N (onlineWelfordLoopBody xReg rowStride) (loopInitial s) = some t ∧
      t.regs .real [] "M" = some (Tile.scalar (twoPassMean xs)) ∧
      t.regs .real [] "S" = some (Tile.scalar (twoPassS xs)) ∧
      t.regs .nat [] "pid" = some (Tile.scalar s.pid) ∧
      t.pid = s.pid ∧ ∀ (r : RegionName) o, t.mem r o = s.mem r o := by
  have hinit : P_welford rowStride xs xReg s.pid 0 (loopInitial s) ∧
      ∀ (r : RegionName) o, (loopInitial s).mem r o = s.mem r o := by
    refine ⟨?_, fun r o => rfl⟩
    simp [P_welford, loopInitial, welfordMean, welfordS]
    exact ⟨rfl, hx⟩
  obtain ⟨t, he, hp, hm⟩ := forLoop_inv
    (idx := "i") (n := N) (body := onlineWelfordLoopBody xReg rowStride)
    (P := fun k t => P_welford rowStride xs xReg s.pid k t ∧
      ∀ (r : RegionName) o, t.mem r o = s.mem r o)
    (s_init := loopInitial s) hinit (fun i t hi hp => by
      obtain ⟨hp, hm⟩ := hp
      obtain ⟨t', he, hp'⟩ := online_welford_step rowStride xs xReg s.pid i t hi hp
      refine ⟨t', he, hp', fun r o => ?_⟩
      rw [stepStmts_assigns_mem _ (onlineWelfordLoopBody_assigns xReg rowStride) _ _ he r o]
      exact hm r o)
  rcases hp with ⟨hM, hS, hpid, hpids, _⟩
  have hstats := welford_eq_two_pass_total xs
  refine ⟨t, ?_, ?_, ?_, hpid, hpids, hm⟩
  · simpa [stepForLoopAux.forLoop_unfold] using he
  · simpa only [hstats.1] using hM
  · simpa only [hstats.2] using hS

theorem twopass_projection (xReg meanReg varReg : RegionName) (N stride : Nat) :
    (twopassWelfordKernel xReg meanReg varReg N stride).eraseDType.toAlgorithm? =
      Except.ok (twopassMathKernel xReg meanReg varReg N stride).toAlgKernel := by
  simp [twopassWelfordKernel, twopassMathKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType]

theorem online_projection (xReg meanReg varReg : RegionName) (N stride : Nat) :
    (onlineWelfordKernel xReg meanReg varReg N stride).eraseDType.toAlgorithm? =
      Except.ok (onlineMathKernel xReg meanReg varReg N stride).toAlgKernel := by
  simp [onlineWelfordKernel, onlineMathKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeExpr.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [Op.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType]

/-- Direct mean/variance computation, including the totalized empty-row case. -/
theorem twopass_region_run (N stride : Nat) (s : BlockState) (xs : Fin N → ℝ)
    (hx : InputRowLoadedAt s "x" stride N xs) :
    ∃ t, exec (twopassMathKernel "x" "mean" "var" N stride).toAlgKernel s = some t ∧
      t.readMem "mean" 0 = twoPassMean xs ∧
      t.readMem "var" 0 = twoPassS xs / N ∧
      (∀ (r : RegionName) o, (r ≠ "mean" ∨ o ≠ 0) → (r ≠ "var" ∨ o ≠ 0) →
        t.mem r o = s.mem r o) := by
  unfold InputRowLoadedAt at hx
  simp only [BlockState.pid_eq] at hx ⊢
  simp [twopassMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Tile.bop, Tile.natToReal, NumericDType.add, NumericDType.mul, NumericDType.sub,
    NumericDType.div, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  refine ⟨?_, ?_, ?_⟩
  · simp [hx, twoPassMean]
    rfl
  · simp [hx, twoPassS, twoPassMean, pow_two]
    rfl
  · intro r o hm hv
    have hm' : ¬ (r = "mean" ∧ o = 0) := by tauto
    have hv' : ¬ (r = "var" ∧ o = 0) := by tauto
    simp [BlockState.writeMem_mem, hm', hv']

/-- The online program's two stores expose the statistics established by the
loop invariant. The full cell frame is retained, not only output readback. -/
theorem online_region_run (N stride : Nat) (s : BlockState) (xs : Fin N → ℝ)
    (hx : InputRowLoadedAt s "x" stride N xs) :
    ∃ t, exec (onlineMathKernel "x" "mean" "var" N stride).toAlgKernel s = some t ∧
      t.readMem "mean" 0 = twoPassMean xs ∧
      t.readMem "var" 0 = twoPassS xs / N ∧
      (∀ (r : RegionName) o, (r ≠ "mean" ∨ o ≠ 0) → (r ≠ "var" ∨ o ≠ 0) →
        t.mem r o = s.mem r o) := by
  obtain ⟨t, he, hM, hS, _, _, hmem⟩ := welford_loop_run N stride "x" s xs hx
  let result := (t.writeMem "mean" 0 (twoPassMean xs)).writeMem "var" 0 (twoPassS xs / N)
  refine ⟨result, ?_, ?_, ?_, ?_⟩
  · simp [onlineMathKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.natToReal]
    simp only [loopInitial, onlineWelfordLoopBody] at he
    rw [he]
    simp [NumericDType.div, hM, hS, result]
  · simp [result, BlockState.writeMem_readMem]
  · simp [result, BlockState.writeMem_readMem]
  · intro r o hm hv
    have hm' : ¬ (r = "mean" ∧ o = 0) := by tauto
    have hv' : ¬ (r = "var" ∧ o = 0) := by tauto
    simp [result, BlockState.writeMem_mem, hm', hv', hmem]

section Safety
variable (xReg meanReg varReg : RegionName) (blockSize rowStride : Nat) (s : BlockState)

theorem stepStmt_assign_inv {d : TileDType}
    {sh : TileShape} {nm : RegName} {e : Op d sh} {t t' : BlockState}
    (h : stepStmt  (.assign d sh nm e) t = some t') :
    ∃ v, evalOp  e t = some v ∧ t' = t.setReg nm d sh v := by
  simp only [stepStmt] at h
  cases hv : evalOp  e t with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      replace h : some (t.setReg nm d sh v) = some t' := h
      exact ⟨v, rfl, (Option.some_inj.mp h).symm⟩


theorem stepStmts_cons_inv {st : Stmt}
    {rest : List Stmt} {t t' : BlockState}
    (h : stepStmts  (st :: rest) t = some t') :
    ∃ t1, stepStmt  st t = some t1 ∧ stepStmts  rest t1 = some t' := by
  rw [stepStmts] at h
  cases h1 : stepStmt  st t with
  | none => rw [h1] at h; exact absurd h (by simp)
  | some t1 => rw [h1] at h; exact ⟨t1, rfl, h⟩

theorem stepStmts_nil_inv {t t' : BlockState}
    (h : stepStmts  [] t = some t') : t' = t := by
  rw [stepStmts] at h
  exact (Option.some_inj.mp h).symm


private theorem offs_activeAddressSafe (bounds : RegionBounds) (pid₀ : Nat) (t : BlockState)
    (active : TileIndex [blockSize] → Prop)
    (hread : t.regs .nat [blockSize] "offs"
      = some ⟨fun i => pid₀ * rowStride + i.1.val⟩)
    (reg : RegionName) (hreg : pid₀ * rowStride + blockSize ≤ bounds reg) :
    memAccessActiveAddressSafe  bounds
      (MemAccess.region reg (Op.ref .nat [blockSize] "offs")) t active := by
  simp only [memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp  (Op.ref .nat [blockSize] "offs") t
      = some ⟨fun i => pid₀ * rowStride + i.1.val⟩ from by
    simp [evalOp.eq_def, hread]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hreg


private theorem const0_activeAddressSafe (bounds : RegionBounds) (t : BlockState)
    (active : TileIndex [] → Prop)
    (reg : RegionName) (hreg : 0 < bounds reg) :
    memAccessActiveAddressSafe  bounds
      (MemAccess.region reg (Op.constNat 0)) t active := by
  simp only [memAccessActiveAddressSafe]
  intro offsets hoffs i _
  rw [show evalOp  (Op.constNat 0) t = some (Tile.scalar 0) from by
    simp [evalOp.eq_def]] at hoffs
  obtain rfl := Option.some_inj.mp hoffs
  simp only [Region.cast_self]
  exact hreg


theorem twopass_flattenOk :
    ((twopassMathKernel xReg meanReg varReg blockSize rowStride
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [twopassMathKernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]


theorem online_flattenOk :
    ((onlineMathKernel xReg meanReg varReg blockSize rowStride
      ).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [onlineMathKernel, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, StmtList.FlattenOk, Stmt.FlattenOk,
    Op.FlattenOk.eq_def]

set_option maxHeartbeats 1600000 in

theorem twopass_traceSafe (bounds : RegionBounds)
    (hxb : s.pids 0 * rowStride + blockSize ≤ bounds xReg)
    (hmb : 0 < bounds meanReg) (hvb : 0 < bounds varReg) :
    Kernel.TraceSafe  bounds
      ((twopassMathKernel xReg meanReg varReg blockSize rowStride
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp  (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by
    simp [evalOp.eq_def]] at hv1
  obtain rfl := Option.some_inj.mp hv1
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  rw [show evalOp  (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat rowStride))
      (Op.arange blockSize)) (s.setReg "pid" .nat [] (Tile.scalar (s.pids 0)))
      = some ⟨fun i => s.pids 0 * rowStride + i.1.val⟩ from by
    simp [evalOp.eq_def, Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec, BlockState.setReg]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    exact ⟨trivial, trivial,
      offs_activeAddressSafe blockSize rowStride  bounds (s.pids 0) _ _
        (by simp [BlockState.setReg]) xReg hxb⟩
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro s7 hs7
  obtain ⟨v7, hv7, rfl⟩ := stepStmt_assign_inv hs7
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s8 hs8
  obtain ⟨v8, hv8, rfl⟩ := stepStmt_assign_inv hs8
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt]
    exact ⟨trivial, by simp, trivial,
      const0_activeAddressSafe  bounds _ _ (Region.cast meanReg)
        (by simpa [Region.cast_self] using hmb)⟩
  intro s9 hs9
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt]
  exact ⟨trivial, by simp, trivial,
    const0_activeAddressSafe  bounds _ _ (Region.cast varReg)
      (by simpa [Region.cast_self] using hvb)⟩


private theorem welfordBody_traceSafe (xReg : RegionName)
    (blockSize rowStride : Nat)
    (bounds : RegionBounds) (pid₀ k : Nat)
    (hk : k < blockSize)
    (hxb : pid₀ * rowStride + blockSize ≤ bounds xReg)
    (t : BlockState) (hpid : t.regs .nat [] "pid" = some (Tile.scalar pid₀)) :
    Stmt.TraceSafeList  bounds (onlineWelfordLoopBody xReg rowStride)
      (t.setReg "i" .nat [] (Tile.scalar k)) := by
  unfold onlineWelfordLoopBody
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, Op.SafeAt]
    refine ⟨by simp, trivial, ?_⟩
    simp only [memAccessActiveAddressSafe]
    intro offsets hoffs i _
    rw [show evalOp  (Op.add .nat .nil
        (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat rowStride))
        (Op.ref .nat [] "i")) (t.setReg "i" .nat [] (Tile.scalar k))
        = some (Tile.scalar (pid₀ * rowStride + k)) from by
      simp [evalOp.eq_def, Tile.bop, NumericDType.nat_add,
        NumericDType.nat_mul, BlockState.setReg, hpid]] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    simp only [Region.cast_self]
    exact lt_of_lt_of_le (Nat.add_lt_add_left hk _) hxb
  intro t1 ht1
  obtain ⟨w1, hw1, rfl⟩ := stepStmt_assign_inv ht1
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro t2 ht2
  obtain ⟨w2, hw2, rfl⟩ := stepStmt_assign_inv ht2
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro t3 ht3
  obtain ⟨w3, hw3, rfl⟩ := stepStmt_assign_inv ht3
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) ?_
  intro t4 ht4
  obtain ⟨w4, hw4, rfl⟩ := stepStmt_assign_inv ht4
  refine Stmt.TraceSafeList.cons_intro
    (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def]) (fun _ _ => .nil_intro)


theorem welfordBody_pid_preserved (xReg : RegionName) (rowStride : Nat)
    {t t' : BlockState}
    (h : stepStmts  (onlineWelfordLoopBody xReg rowStride) t = some t') :
    t'.regs .nat [] "pid" = t.regs .nat [] "pid" := by
  unfold onlineWelfordLoopBody at h
  obtain ⟨t1, h1, h⟩ := stepStmts_cons_inv h
  obtain ⟨w1, _, rfl⟩ := stepStmt_assign_inv h1
  obtain ⟨t2, h2, h⟩ := stepStmts_cons_inv h
  obtain ⟨w2, _, rfl⟩ := stepStmt_assign_inv h2
  obtain ⟨t3, h3, h⟩ := stepStmts_cons_inv h
  obtain ⟨w3, _, rfl⟩ := stepStmt_assign_inv h3
  obtain ⟨t4, h4, h⟩ := stepStmts_cons_inv h
  obtain ⟨w4, _, rfl⟩ := stepStmt_assign_inv h4
  obtain ⟨t5, h5, h⟩ := stepStmts_cons_inv h
  obtain ⟨w5, _, rfl⟩ := stepStmt_assign_inv h5
  obtain rfl := stepStmts_nil_inv h
  simp [BlockState.setReg]


theorem welfordLoop_traceSafe (xReg : RegionName)
    (blockSize rowStride : Nat)
    (bounds : RegionBounds) (pid₀ : Nat)
    (hxb : pid₀ * rowStride + blockSize ≤ bounds xReg) :
    ∀ (fuel k : Nat) (t : BlockState), blockSize - k ≤ fuel →
      t.regs .nat [] "pid" = some (Tile.scalar pid₀) →
      Stmt.forLoopTraceSafe  bounds "i" k blockSize
        (onlineWelfordLoopBody xReg rowStride) t
  | 0, k, t, hf, hpid => by
      rw [Stmt.forLoopTraceSafe]
      rw [if_neg (by omega)]
      trivial
  | fuel + 1, k, t, hf, hpid => by
      rw [Stmt.forLoopTraceSafe]
      split
      next hlt =>
        refine ⟨welfordBody_traceSafe xReg blockSize rowStride  bounds pid₀ k
          hlt hxb t hpid, ?_⟩
        split
        next s' hs' =>
          refine welfordLoop_traceSafe xReg blockSize rowStride  bounds pid₀
            hxb fuel (k + 1) s' (by omega) ?_
          rw [welfordBody_pid_preserved xReg rowStride hs']
          simp [BlockState.setReg, hpid]
        next => trivial
      next => trivial

set_option maxHeartbeats 1600000 in

theorem online_traceSafe (bounds : RegionBounds)
    (hxb : s.pids 0 * rowStride + blockSize ≤ bounds xReg)
    (hmb : 0 < bounds meanReg) (hvb : 0 < bounds varReg) :
    Kernel.TraceSafe  bounds
      ((onlineMathKernel xReg meanReg varReg blockSize rowStride
        ).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp  (Op.programId 0) s = some (Tile.scalar (s.pids 0)) from by
    simp [evalOp.eq_def]] at hv1
  obtain rfl := Option.some_inj.mp hv1
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s3 hs3
  obtain ⟨v3, hv3, rfl⟩ := stepStmt_assign_inv hs3
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe]
    exact welfordLoop_traceSafe xReg blockSize rowStride  bounds (s.pids 0)
      hxb blockSize 0 _ (by omega) (by simp [BlockState.setReg])
  intro s4 hs4
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt]
    exact ⟨trivial, by simp, trivial,
      const0_activeAddressSafe  bounds _ _ (Region.cast meanReg)
        (by simpa [Region.cast_self] using hmb)⟩
  intro s5 hs5
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt]
  exact ⟨trivial, by simp, trivial,
    const0_activeAddressSafe  bounds _ _ (Region.cast varReg)
      (by simpa [Region.cast_self] using hvb)⟩

end Safety

def twopassIO (N stride : Nat) : KernelIO₁ₓ₂ where
  kernel := (twopassWelfordKernel "x" "mean" "var" N stride).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, twopass_projection]
  inp := "x"
  out1 := "mean"
  out2 := "var"
  Bin := N
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * stride
  write1 := fun _ => 0
  write2 := fun _ => 0

specification twopass_welford_correct (N stride : Nat) :
    Spec.Real (twopassIO N stride ⊨ fun xs =>
      ((fun _ => (∑ i, xs i) / N),
       (fun _ => (∑ i, (xs i - (∑ j, xs j) / N) ^ 2) / N))) := by
  have halg : (twopassIO N stride).kernel.toAlgKernel =
      (twopassMathKernel "x" "mean" "var" N stride).toAlgKernel := by
    simp [twopassIO, ComputeKernel.toAlgKernel, twopass_projection]
  refine KernelIO₁ₓ₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact twopass_flattenOk "x" "mean" "var" N stride
  · intro bounds s hx hm hv _
    rw [halg]; exact twopass_traceSafe "x" "mean" "var" N stride s bounds hx hm hv
  · intro s xs hx
    rw [halg]
    obtain ⟨t, he, hm, hv, hf⟩ := twopass_region_run N stride s xs hx
    refine ⟨t, he, ?_, ?_, ?_⟩
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassMean] using hm
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassS, twoPassMean] using hv
    · intro r o hm hv _
      apply hf r o
      · rcases hm with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)
      · rcases hv with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)

def onlineIO (N stride : Nat) : KernelIO₁ₓ₂ where
  kernel := (onlineWelfordKernel "x" "mean" "var" N stride).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, online_projection]
  inp := "x"
  out1 := "mean"
  out2 := "var"
  Bin := N
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * stride
  write1 := fun _ => 0
  write2 := fun _ => 0

specification online_welford_correct (N stride : Nat) :
    Spec.Real (onlineIO N stride ⊨ fun xs =>
      ((fun _ => (∑ i, xs i) / N),
       (fun _ => (∑ i, (xs i - (∑ j, xs j) / N) ^ 2) / N))) := by
  have halg : (onlineIO N stride).kernel.toAlgKernel =
      (onlineMathKernel "x" "mean" "var" N stride).toAlgKernel := by
    simp [onlineIO, ComputeKernel.toAlgKernel, online_projection]
  refine KernelIO₁ₓ₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact online_flattenOk "x" "mean" "var" N stride
  · intro bounds s hx hm hv _
    rw [halg]; exact online_traceSafe "x" "mean" "var" N stride s bounds hx hm hv
  · intro s xs hx
    rw [halg]
    obtain ⟨t, he, hm, hv, hf⟩ := online_region_run N stride s xs hx
    refine ⟨t, he, ?_, ?_, ?_⟩
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassMean] using hm
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassS, twoPassMean] using hv
    · intro r o hm hv _
      apply hf r o
      · rcases hm with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)
      · rcases hv with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.WelfordCorrect
