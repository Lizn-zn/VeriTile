import bench.tritonbench_g.block_sparse_attn.Proofs.RealProgram
import bench.tritonbench_g.block_sparse_attn.Proofs.RealNormalizationValues
import VeriTile.Triton.Memory.TraceProofs

/-! Actual prologue Q loads for both EVEN_M branches and both channel blocks.
Masked query rows retain arbitrary undefined values. The pointer tile and row
indices are supplied by the earlier address setup; no q/q2 register is assumed.
The initial-array correspondence is required only on stored query rows. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealQuery
open VeriTile.Triton RealNormalizationValues
set_option linter.unusedSimpArgs false

def pointer (BM BD : Nat) (second : Bool) : Op .ptr [BM, BD] :=
  if second then .ptrAdd .scalarR (.ref .ptr [BM, BD] "q_ptrs") (.constNat BD)
    else .ref .ptr [BM, BD] "q_ptrs"

def mask (BM BD : Nat) : Op .bool [BM, BD] :=
  .remap [BM, BD] (Broadcast.consSame (Broadcast.consL Broadcast.nil)).leftIndex
    (.lt .nat .scalarR (.expandDim 1 (.ref .nat [BM] "offs_m")) (.ref .nat [] "q_seq_len"))

def load (BM BD : Nat) (even second : Bool) : Stmt :=
  .assign .real [BM, BD] (if second then "q2" else "q")
    (.load .real (.ptr (pointer BM BD second)) (if even then .none else .mask (mask BM BD)))

def branch (BM BD blocks : Nat) (even : Bool) : List Stmt :=
  [load BM BD even Bool.false,
   .ifThen (.constBool (decide (2 ≤ blocks))) [load BM BD even Bool.true]]

def code (BM BD blocks : Nat) (even : Bool) : List Stmt :=
  [.ifThenElse (.constBool even) (branch BM BD blocks Bool.true) (branch BM BD blocks Bool.false)]

def writesQuery (stmts : List ComputeStmt) : Bool := stmts.any fun st =>
  match st with
  | .assign _ _ name _ => name == "q"
  | _ => Bool.false

/-- The top-level query-load conditional, located by its q write. -/
def source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) : List ComputeStmt :=
  let before := (RealProgram.parts swapped BM BN BD p).before
  let located := (Source.conditional? before writesQuery).getD ⟨[], .constBool Bool.false, [], [], []⟩
  (before.drop located.before.length).take 1

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem matches_source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (source swapped BM BN BD p) = code BM BD p.blocks p.evenM := by
  cases swapped
  all_goals
    simp [source, code, branch, load, pointer, mask, writesQuery,
      RealProgram.realPart, RealProgram.parts, RealProgram.source, TypedSource.specialize,
      Source.intRange?, Source.prefixBefore, Source.conditional?, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    rfl

noncomputable section
open Classical

def address {BM BD : Nat} (ptrs : Tile .ptr [BM, BD]) (second : Bool)
    (r : Fin BM) (d : Fin BD) : RegionName × Nat :=
  let p := ptrs.data (r, d, PUnit.unit)
  (p.1, p.2 + if second then BD else 0)

def active {BM : Nat} (rows : Fin BM → Nat) (context : Nat) (even : Bool) (r : Fin BM) : Prop :=
  even = Bool.true ∨ rows r < context

def loaded {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context : Nat) (even second : Bool) (r : Fin BM) (d : Fin BD) : ℝ :=
  let a := address ptrs second r d
  if active rows context even r then s.readMem a.1 a.2 else s.undef a.1 a.2

theorem pointer_eval {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD]) (second : Bool)
    (hp : s.regs .ptr [BM, BD] "q_ptrs" = some ptrs) :
    evalOp (pointer BM BD second) s = some (Tile.mat (address ptrs second)) := by
  cases second
  · simp only [pointer, Bool.false_eq_true, ↓reduceIte, evalOp_ref, hp]
    congr 1
  · simp [pointer, evalOp.eq_def, hp, bind, Option.bind]
    rfl

theorem mask_eval {BM BD : Nat} (s : BlockState) (rows : Fin BM → Nat) (context : Nat)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hc : s.regs .nat [] "q_seq_len" = some (Tile.scalar context)) :
    evalOp (mask BM BD) s = some (Tile.mat (fun r (_ : Fin BD) => decide (rows r < context))) := by
  simp [mask, evalOp.eq_def, hm, hc, bind, Option.bind]
  rfl

theorem load_eval {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BM, BD] "q_ptrs" = some ptrs)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hc : s.regs .nat [] "q_seq_len" = some (Tile.scalar context)) :
    evalOp (.load .real (.ptr (pointer BM BD second)) (if even then .none else .mask (mask BM BD))) s =
      some (matrix (loaded s ptrs rows context even second)) := by
  have hp' := pointer_eval s ptrs second hp
  have hmask := mask_eval (BD := BD) s rows context hm hc
  cases even
  all_goals
    rw [evalOp.eq_def]
    simp [hp', hmask, bind, Option.bind]
    apply Tile.ext
    rintro ⟨r, d, ⟨⟩⟩
    by_cases h : rows r < context <;> simp [matrix, finite, loaded, active, h]

theorem load_executes {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BM, BD] "q_ptrs" = some ptrs)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hc : s.regs .nat [] "q_seq_len" = some (Tile.scalar context)) :
    stepStmt (load BM BD even second) s =
      some (s.setReg (if second then "q2" else "q") .real [BM, BD]
        (matrix (loaded s ptrs rows context even second))) :=
  stepStmt_assign_eq_some (load_eval s ptrs rows context even second hp hm hc)

@[simp] theorem loaded_setReg {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context : Nat) (even second : Bool)
    (name : RegName) (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape) :
    loaded (s.setReg name dtype shape v) ptrs rows context even second =
      loaded s ptrs rows context even second := by
  funext r d
  simp [loaded]

def result {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context blocks : Nat) (even : Bool) : BlockState :=
  let t := s.setReg "q" .real [BM, BD] (matrix (loaded s ptrs rows context even Bool.false))
  if 2 ≤ blocks then t.setReg "q2" .real [BM, BD]
    (matrix (loaded s ptrs rows context even Bool.true)) else t

theorem branch_executes {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context blocks : Nat) (even : Bool)
    (hp : s.regs .ptr [BM, BD] "q_ptrs" = some ptrs)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hc : s.regs .nat [] "q_seq_len" = some (Tile.scalar context)) :
    stepStmts (branch BM BD blocks even) s = some (result s ptrs rows context blocks even) := by
  have hf := load_executes s ptrs rows context even Bool.false hp hm hc
  unfold branch
  rw [stepStmts.cons_some hf]
  let t := s.setReg "q" .real [BM, BD] (matrix (loaded s ptrs rows context even Bool.false))
  change stepStmts [.ifThen (.constBool (decide (2 ≤ blocks))) [load BM BD even Bool.true]] t = _
  by_cases hb : 2 ≤ blocks
  · have hs := load_executes t ptrs rows context even Bool.true
      (by simpa [t] using hp) (by simpa [t] using hm) (by simpa [t] using hc)
    simp [stepStmts, stepStmt, evalOp.eq_def, hb, hs, result, t]
  · simp [stepStmts, stepStmt, evalOp.eq_def, hb, result, t]

theorem executes {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context blocks : Nat) (even : Bool)
    (hp : s.regs .ptr [BM, BD] "q_ptrs" = some ptrs)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hc : s.regs .nat [] "q_seq_len" = some (Tile.scalar context)) :
    stepStmts (code BM BD blocks even) s = some (result s ptrs rows context blocks even) := by
  have ht := branch_executes s ptrs rows context blocks Bool.true hp hm hc
  have hf := branch_executes s ptrs rows context blocks Bool.false hp hm hc
  cases even <;> simp [code, stepStmts, stepStmt, evalOp.eq_def, ht, hf]

theorem state_frame {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context blocks : Nat) (even : Bool) :
    {result s ptrs rows context blocks even with regs := s.regs} = s := by
  by_cases hb : 2 ≤ blocks <;> simp [result, hb, BlockState.setReg]

theorem register_frame {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context blocks : Nat) (even : Bool)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hq : name ≠ "q") (hq2 : name ≠ "q2") :
    (result s ptrs rows context blocks even).regs dtype shape name = s.regs dtype shape name := by
  by_cases hb : 2 ≤ blocks <;> simp [result, hb, hq, hq2]

theorem readbacks {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context blocks : Nat) (even : Bool) :
    (result s ptrs rows context blocks even).regs .real [BM, BD] "q" =
      some (matrix (loaded s ptrs rows context even Bool.false)) ∧
    (2 ≤ blocks → (result s ptrs rows context blocks even).regs .real [BM, BD] "q2" =
      some (matrix (loaded s ptrs rows context even Bool.true))) := by
  by_cases hb : 2 ≤ blocks <;> simp [result, hb]

/-- On a stored query row the load is active, regardless of EVEN_M. Other
query rows need no relation to the independent input array. -/
theorem loaded_eq_input {BM BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context : Nat) (even second : Bool)
    (query : Nat → Nat → ℝ) (r : Fin BM) (d : Fin BD)
    (hr : rows r < context)
    (hq : s.readMem (address ptrs second r d).1 (address ptrs second r d).2 =
      query (rows r) (if second then BD + d.val else d.val)) :
    loaded s ptrs rows context even second r d = query (rows r) (if second then BD + d.val else d.val) := by
  simpa [loaded, active, hr] using hq

def Bounds {BM BD : Nat} (bounds : RegionBounds) (ptrs : Tile .ptr [BM, BD])
    (rows : Fin BM → Nat) (context : Nat) (even second : Bool) : Prop :=
  ∀ r d, active rows context even r →
    (address ptrs second r d).2 < bounds (address ptrs second r d).1

theorem mask_active {BM BD : Nat} (s : BlockState) (rows : Fin BM → Nat)
    (context : Nat) (even : Bool)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hc : s.regs .nat [] "q_seq_len" = some (Tile.scalar context)) (r : Fin BM) (d : Fin BD) :
    (if even then MaskOpt.none (dtype := .real) else .mask (mask BM BD)).Active s (r, d, PUnit.unit) ↔
      active rows context even r := by
  cases even <;> simp [MaskOpt.Active, mask_eval s rows context hm hc, active]

theorem load_safe {BM BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BM, BD]) (rows : Fin BM → Nat) (context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BM, BD] "q_ptrs" = some ptrs)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hc : s.regs .nat [] "q_seq_len" = some (Tile.scalar context))
    (hb : Bounds bounds ptrs rows context even second) :
    Stmt.TraceSafe bounds (load BM BD even second) s := by
  have haddr : (MemAccess.ptr (d := .real) (pointer BM BD second)).ActiveAddressSafe bounds s
      ((if even then MaskOpt.none (dtype := .real) else .mask (mask BM BD)).Active s) := by
    refine ⟨?_, ?_⟩
    · cases second <;> simp [pointer, Op.MemorySafe]
    · intro tile ht
      rw [pointer_eval s ptrs second hp] at ht
      cases ht
      rintro ⟨r, d, ⟨⟩⟩ ha
      exact hb r d ((mask_active s rows context even hm hc r d).mp ha)
  rw [load, Stmt.TraceSafe, Op.SafeAt.eq_def]
  refine ⟨?_, ?_, haddr⟩
  · cases second <;> simp [pointer, Op.SafeAt.eq_def]
  · cases even <;> simp [mask, Op.SafeAt.eq_def]

theorem branch_safe {BM BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BM, BD]) (rows : Fin BM → Nat) (context blocks : Nat) (even : Bool)
    (hp : s.regs .ptr [BM, BD] "q_ptrs" = some ptrs)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hc : s.regs .nat [] "q_seq_len" = some (Tile.scalar context))
    (hb : Bounds bounds ptrs rows context even Bool.false)
    (hb2 : 2 ≤ blocks → Bounds bounds ptrs rows context even Bool.true) :
    Stmt.TraceSafeList bounds (branch BM BD blocks even) s := by
  refine Stmt.TraceSafeList.cons_intro (load_safe bounds s ptrs rows context even Bool.false hp hm hc hb) ?_
  intro t ht
  rw [load_executes s ptrs rows context even Bool.false hp hm hc] at ht
  cases ht
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  rw [Stmt.TraceSafe]
  simp only [evalOp.eq_def, Op.SafeAt.eq_def, Tile.scalar_data, true_and]
  by_cases htwo : 2 ≤ blocks
  · simp only [htwo, decide_true, ↓reduceIte]
    refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
    apply load_safe (ptrs := ptrs) (rows := rows) (context := context)
    · simpa using hp
    · simpa using hm
    · simpa using hc
    · exact hb2 htwo
  · simp [htwo]

theorem code_safe {BM BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BM, BD]) (rows : Fin BM → Nat) (context blocks : Nat) (even : Bool)
    (hp : s.regs .ptr [BM, BD] "q_ptrs" = some ptrs)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hc : s.regs .nat [] "q_seq_len" = some (Tile.scalar context))
    (hb : Bounds bounds ptrs rows context even Bool.false)
    (hb2 : 2 ≤ blocks → Bounds bounds ptrs rows context even Bool.true) :
    Stmt.TraceSafeList bounds (code BM BD blocks even) s := by
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  rw [Stmt.TraceSafe]
  simp only [evalOp.eq_def, Op.SafeAt.eq_def, Tile.scalar_data, true_and]
  cases even <;> simp only [Bool.false_eq_true, ↓reduceIte]
  all_goals exact branch_safe bounds s ptrs rows context blocks _ hp hm hc hb hb2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealQuery
