import bench.examples.VectorAdd.Kernels
import VeriTile.Triton
import VeriTile.Triton.Memory.Flatten
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-! Real execution, arithmetic and memory lemmas for VectorAdd.
The public IO contracts and mathematical specifications are in ../Correct.lean. -/

namespace VeriTile.Bench.Examples.VectorAdd
open VeriTile.Bench.Examples.VectorAdd.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Triton
open VeriTile.Triton.KernelIO₂ (Implements)
open scoped VeriTile.Triton.KernelIO₂
open VeriTile.Examples

/-- Value half: from any state with the inputs loaded, every output lane
holds the elementwise sum. -/
theorem add_kernel_correct
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (_hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs)
    (_h_y : InputLoadedAt s yReg blockSize ys) :
    ∀ i : Fin blockSize,
      observeAt (exec (addKernel xReg yReg outReg blockSize) s) outReg blockSize s.pid i
        = some (xs i + ys i) := by
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [blockSize] => s.pid * blockSize + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, addKernel, stepStmts, stepStmt, Tile.bop,
        NumericDType.add, NumericDType.mul]
  unfold InputLoadedAt at _h_x _h_y
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x, _h_y]

/-- A scatter-store `foldl` leaves every memory cell it does not hit
unchanged (cell-level frame for the unmasked store). -/
private theorem foldl_store_preserve_cell {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ)
    (r : RegionName) (o : Nat) (l : List α) (s : BlockState)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    (l.foldl (fun acc k =>
        acc.writeMem region (offsetFn k) (valueFn k)) s).mem r o
      = s.mem r o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons,
        ih _ (fun k hk => hnot k (List.mem_cons_of_mem hd hk)),
        BlockState.writeMem_mem]
      exact if_neg (fun hc =>
        hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

/-- Frame half: every memory cell other than the output window is preserved
by the run. -/
private theorem addKernel_frame (xReg yReg outReg : RegionName)
    (B : Nat) (s s1 : BlockState)
    (hExec : exec ((addKernel xReg yReg outReg B).toAlgKernel) s = some s1)
    (r : RegionName) (o : Nat)
    (hmiss : ∀ i : Fin B, ¬(outReg = r ∧ s.pid * B + i.val = o)) :
    s1.mem r o = s.mem r o := by
  simp [exec, addKernel, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
    Tile.bop, NumericDType.add, NumericDType.mul] at hExec
  subst hExec
  refine Eq.trans (foldl_store_preserve_cell _ _ r o _ _ ?_) rfl
  intro k _ hc
  exact hmiss k.1 hc

/-- **The region-model Hoare triple** — termination, output-window values,
and frame, from any launch state whose input windows are loaded. This is
what the `⊨` headline transports to flat memory. -/
theorem addKernel_region_run (B : Nat) (hB : 0 < B)
    (s₀ : BlockState) (xs ys : Fin B → ℝ)
    (hx : ∀ j : Fin B, s₀.readMem ⟨"x"⟩ (s₀.pid * B + j.val) = xs j)
    (hy : ∀ j : Fin B, s₀.readMem ⟨"y"⟩ (s₀.pid * B + j.val) = ys j) :
    ∃ s1, exec ((addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).toAlgKernel) s₀ = some s1
      ∧ (∀ j : Fin B,
          s1.readMem ⟨"out"⟩ (s₀.pid * B + j.val) = xs j + ys j)
      ∧ (∀ r o,
          (r ≠ ⟨"out"⟩ ∨ ∀ j : Fin B, o ≠ s₀.pid * B + j.val) →
          s1.mem r o = s₀.mem r o) := by
  have hobs := add_kernel_correct ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B hB s₀ xs ys hx hy
  rw [show exec (addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B) s₀
      = exec ((addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).toAlgKernel) s₀ from rfl]
    at hobs
  cases hsrc : exec ((addKernel ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B).toAlgKernel) s₀ with
  | none =>
      have := hobs ⟨0, hB⟩
      rw [hsrc] at this
      simp [observeAt] at this
  | some s1 =>
      refine ⟨s1, rfl, fun j => ?_, fun r o hcond => ?_⟩
      · have := hobs j
        rw [hsrc] at this
        simpa [observeAt] using this
      · refine addKernel_frame ⟨"x"⟩ ⟨"y"⟩ ⟨"out"⟩ B s₀ s1 hsrc r o
          (fun i ⟨hr, ho⟩ => ?_)
        rcases hcond with hne | hno
        · exact hne hr.symm
        · exact hno i ho.symm

/-- Inversion for a successful `assign` step. -/
private theorem stepStmt_assign_inv {d : TileDType} {sh : TileShape}
    {nm : RegName} {e : Op d sh} {s s' : BlockState}
    (h : stepStmt (.assign d sh nm e) s = some s') :
    ∃ v, evalOp e s = some v ∧ s' = s.setReg nm d sh v := by
  simp only [stepStmt] at h
  cases hv : evalOp e s with
  | none => rw [hv] at h; exact absurd h (by simp)
  | some v =>
      rw [hv] at h
      replace h : some (s.setReg nm d sh v) = some s' := h
      exact ⟨v, rfl, (Option.some_inj.mp h).symm⟩

set_option maxHeartbeats 1600000 in
/-- The aligned vector add is trace-safe: each of its six statements is safe
in the state the execution actually reaches. Unmasked loads/store touch all
`B` lanes of the program tile, so each region's bound must cover the whole
tile: `s.pid * B + B ≤ bounds reg`. -/
theorem addKernel_traceSafe (xReg yReg outReg : RegionName)
    (B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hx : s.pid * B + B ≤ bounds xReg) (hy : s.pid * B + B ≤ bounds yReg)
    (hout : s.pid * B + B ≤ bounds outReg) :
    Kernel.TraceSafe bounds
      ((addKernel xReg yReg outReg B).toAlgKernel)
      s := by
  unfold Kernel.TraceSafe
  -- statement 1: pid := program_id(0)
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s1 hs1
  obtain ⟨v1, hv1, rfl⟩ := stepStmt_assign_inv hs1
  rw [show evalOp (Op.programId 0) s
      = some (Tile.scalar (dtype := .nat) (s.pids 0)) from by simp] at hv1
  obtain rfl := Option.some_inj.mp hv1
  -- statement 2: offs := pid * B + arange B
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s2 hs2
  obtain ⟨v2, hv2, rfl⟩ := stepStmt_assign_inv hs2
  rw [show evalOp (Op.add .nat .scalarL
      (Op.mul .nat .nil (Op.ref .nat [] "pid") (Op.constNat B))
      (Op.arange B))
      (s.setReg "pid" .nat [] (Tile.scalar (dtype := .nat) (s.pids 0)))
      = some ⟨fun i => s.pids 0 * B + i.1.val⟩ from by
    simp [Tile.bop, NumericDType.nat_add, NumericDType.nat_mul,
      Tile.vec]] at hv2
  obtain rfl := Option.some_inj.mp hv2
  -- shared register-readback fact at the state before the loads
  set sm := (s.setReg "pid" .nat []
        (Tile.scalar (dtype := .nat) (s.pids 0))).setReg "offs" .nat [B]
      ⟨fun i => s.pids 0 * B + i.1.val⟩
      with hsm
  have hAAS : ∀ (t : BlockState),
      (∀ dt sh nm, nm = "offs" → t.regs dt sh nm = sm.regs dt sh nm) →
      ∀ (reg : RegionName), s.pid * B + B ≤ bounds reg →
      MemAccess.ActiveAddressSafe bounds
        (MemAccess.region reg (Op.ref .nat [B] "offs")) t
        ((MaskOpt.none (dtype := .real)).Active t) := by
    intro t hframe reg hreg
    simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe]
    intro offsets hoffs i _
    rw [show evalOp (Op.ref .nat [B] "offs") t
        = some ⟨fun i => s.pids 0 * B + i.1.val⟩ from by
      simp [hframe .nat [B] "offs" rfl, hsm, BlockState.setReg]] at hoffs
    obtain rfl := Option.some_inj.mp hoffs
    simp only [Region.cast_self]
    exact lt_of_lt_of_le (Nat.add_lt_add_left i.1.isLt _) hreg
  -- statement 3: x := load(xReg + offs)   (unmasked: all lanes active)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    exact ⟨by simp, trivial, hAAS sm (fun _ _ _ _ => rfl) xReg hx⟩
  intro s4 hs4
  obtain ⟨v4, hv4, rfl⟩ := stepStmt_assign_inv hs4
  -- statement 4: y := load(yReg + offs)
  refine Stmt.TraceSafeList.cons_intro ?_ ?_
  · simp only [ComputeDType.eraseDType, Stmt.TraceSafe, Op.SafeAt]
    refine ⟨by simp, trivial, hAAS _ (fun dt sh nm hnm => ?_) yReg hy⟩
    subst hnm; simp [BlockState.setReg]
  intro s5 hs5
  obtain ⟨v5, hv5, rfl⟩ := stepStmt_assign_inv hs5
  -- statement 5: out := x + y
  refine Stmt.TraceSafeList.cons_intro (by simp [Stmt.TraceSafe, Op.SafeAt]) ?_
  intro s6 hs6
  obtain ⟨v6, hv6, rfl⟩ := stepStmt_assign_inv hs6
  -- statement 6: store(outReg + offs, out)   (unmasked)
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
  simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt]
  refine ⟨by simp [Op.SafeAt], by simp [Op.SafeAt], trivial,
    hAAS _ (fun dt sh nm hnm => ?_) outReg hout⟩
  subst hnm; simp [BlockState.setReg]

/-- The aligned kernel sits inside the bridge's covered fragment. -/
theorem addKernel_flattenOk (xReg yReg outReg : RegionName) (B : Nat) :
    ((addKernel xReg yReg outReg B).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [addKernel, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

theorem optimized_exec (B : Nat) (s : BlockState) :
    exec (optimizedKernel B).toAlgKernel s =
      exec (addKernel "x" "y" "out" B).toAlgKernel s := by
  simp [optimizedKernel, addKernel, exec, stepStmts, stepStmt, evalOp.eq_def,
    Region.cast, Tile.bop, NumericDType.add, NumericDType.mul, add_comm]

end Optimized

end VeriTile.Bench.Examples.VectorAdd
