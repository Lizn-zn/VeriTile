import bench.tritonbench_g.block_sparse_attn.Proofs.RealColumn
import bench.tritonbench_g.block_sparse_attn.Proofs.RealKey
import bench.tritonbench_g.block_sparse_attn.Proofs.RealNormalizationMath

/-! Actual CSR-column and score-prefix execution: masked K loads, one or two
matrix dots, scaling and the causal mask. Q tiles, base pointers and layout /
visit registers are supplied by the still-separate prologue proof. The score
is related to independent initial Q/K arrays on stored query rows only; unused
query rows and masked K values remain arbitrary. Local access safety and all
state frames are proved here, without assuming whole-loop execution. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealScore
open VeriTile.Triton RealNormalizationValues
open scoped BigOperators
noncomputable section
set_option linter.unusedSimpArgs false

def dot (BM BN BD : Nat) (query : RegName) : Stmt :=
  .assign .real [BM, BN] "qk"
    (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "qk")
      (.dot (batch := []) (.ref .real [BM, BD] query) (.ref .real [BD, BN] "k")))

def scaling (BM BN : Nat) (scale : ℝ) (swapped : Bool) : Stmt :=
  .assign .real [BM, BN] "qk" (if swapped then
    .mul .real .scalarL (.const scale) (.ref .real [BM, BN] "qk")
    else .mul .real .scalarR (.ref .real [BM, BN] "qk") (.const scale))

def causal (BM BN : Nat) : Op .bool [BM, BN] :=
  .ge .int (.consR (.consL .nil))
    (.castNatToInt (.expandDim 1 (.ref .nat [BM] "offs_m")))
    (.add .int .scalarL (.ref .int [] "start_n")
      (.castNatToInt (.expandDim 0 (.ref .nat [BN] "offs_n"))))

def masking (BM BN : Nat) : Stmt :=
  .assign .real [BM, BN] "qk"
    (.add .real (.consSame (.consSame .nil)) (.ref .real [BM, BN] "qk")
      (.where (causal BM BN) ((Op.const (0 : ℝ)).broadcast [BM, BN])
        (Op.negInf.broadcast [BM, BN])))

def code (BM BN BD stride context blocks : Nat) (scale : ℝ) (even swapped : Bool) : List Stmt :=
  RealKey.code BN BD stride context even Bool.false ++
    [.assign .real [BM, BN] "qk" (.full [BM, BN] (.const 0)),
     dot BM BN BD "q",
     .ifThen (.constBool (decide (2 ≤ blocks)))
       (RealKey.code BN BD stride context even Bool.true ++ [dot BM BN BD "q2"]),
     scaling BM BN scale swapped,
     masking BM BN]

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem prefix_matches (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.normalizationPrefix swapped BM BN BD p) =
      RealColumn.code BN p.cols p.columnStride ++
        code BM BN BD p.kRow p.context p.blocks p.scale p.evenN swapped := by
  cases swapped
  all_goals
    simp [code, dot, scaling, masking, causal, RealColumn.code, RealColumn.offset,
      RealKey.code, RealKey.pointer, RealKey.mask,
      RealProgram.realPart, RealProgram.normalizationPrefix, RealProgram.parts,
      RealProgram.source, TypedSource.specialize, Source.intRange?, Source.prefixBefore,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    exact ⟨rfl, rfl, rfl⟩

def dotValues {BM BN BD : Nat} (q : Fin BM → Fin BD → ℝ) (k : Fin BD → Fin BN → ℝ) :
    Fin BM → Fin BN → ℝ := fun r j => ∑ d : Fin BD, q r d * k d j

theorem dot_executes {BM BN BD : Nat} (s : BlockState) (name : RegName)
    (q : Fin BM → Fin BD → ℝ) (k : Fin BD → Fin BN → ℝ) (old : Fin BM → Fin BN → ℝ)
    (hq : s.regs .real [BM, BD] name = some (matrix q))
    (hk : s.regs .real [BD, BN] "k" = some (matrix k))
    (ho : s.regs .real [BM, BN] "qk" = some (matrix old)) :
    stepStmt (dot BM BN BD name) s =
      some (s.setReg "qk" .real [BM, BN] (matrix (fun r j => old r j + dotValues q k r j))) := by
  apply stepStmt_assign_eq_some
  simp [dot, evalOp.eq_def, hq, hk, ho, bind, Option.bind,
    RealNormalizationMath.matrix_dot, RealNormalizationMath.matrix_add, dotValues]

theorem scaling_executes {BM BN : Nat} (s : BlockState) (scale : ℝ) (swapped : Bool)
    (old : Fin BM → Fin BN → ℝ)
    (ho : s.regs .real [BM, BN] "qk" = some (matrix old)) :
    stepStmt (scaling BM BN scale swapped) s =
      some (s.setReg "qk" .real [BM, BN] (matrix (fun r j => old r j * scale))) := by
  apply stepStmt_assign_eq_some
  cases swapped
  · simp [scaling, evalOp.eq_def, ho, bind, Option.bind]
    rfl
  · simp [scaling, evalOp.eq_def, ho, bind, Option.bind]
    apply Tile.ext
    rintro ⟨r, j, ⟨⟩⟩
    change (some (scale * old r j) : WithBot ℝ) = some (old r j * scale)
    rw [mul_comm]

theorem causal_eval {BM BN : Nat} (s : BlockState) (start : Int) (rows : Fin BM → Nat)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val)) :
    evalOp (causal BM BN) s =
      some (Tile.mat (fun r j => decide (start + (j.val : Int) ≤ (rows r : Int)))) := by
  simp [causal, evalOp.eq_def, hs, hm, hn, bind, Option.bind]
  rfl

def masked {BM BN : Nat} (old : Fin BM → Fin BN → ℝ) (start : Int) (rows : Fin BM → Nat) :
    Fin BM → Fin BN → WithBot ℝ :=
  fun r j => if start + (j.val : Int) ≤ (rows r : Int) then (old r j : WithBot ℝ) else ⊥

theorem masking_executes {BM BN : Nat} (s : BlockState) (start : Int) (rows : Fin BM → Nat)
    (old : Fin BM → Fin BN → ℝ)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (ho : s.regs .real [BM, BN] "qk" = some (matrix old)) :
    stepStmt (masking BM BN) s =
      some (s.setReg "qk" .real [BM, BN] (extendedMatrix (masked old start rows))) := by
  have hc := causal_eval s start rows hs hm hn
  have hw : evalOp (.where (causal BM BN) ((Op.const (0 : ℝ)).broadcast [BM, BN])
      (Op.negInf.broadcast [BM, BN])) s =
      some (extendedMatrix (masked (fun _ _ => 0) start rows)) := by
    simp only [evalOp_where, hc, evalOp, bind, Option.bind]
    apply congrArg some
    apply Tile.ext
    rintro ⟨r, j, ⟨⟩⟩
    by_cases h : start + (j.val : Int) ≤ (rows r : Int) <;>
      simp [Tile.select, Tile.mat, extendedMatrix, masked, h]
    all_goals rfl
  unfold masking
  apply stepStmt_assign_eq_some
  simp only [evalOp_add, evalOp_ref, ho, hw, bind, Option.bind]
  apply congrArg some
  apply Tile.ext
  rintro ⟨r, j, ⟨⟩⟩
  change WithBot.realAdd (some (old r j))
      (if start + (j.val : Int) ≤ (rows r : Int) then some 0 else none) =
    (if start + (j.val : Int) ≤ (rows r : Int) then some (old r j) else none)
  by_cases h : start + (j.val : Int) ≤ (rows r : Int) <;>
    simp [masked, WithBot.realAdd, h]

def loadDot (BM BN BD stride context : Nat) (even second : Bool) : List Stmt :=
  RealKey.code BN BD stride context even second ++
    (if second then [] else [.assign .real [BM, BN] "qk" (.full [BM, BN] (.const 0))]) ++
    [dot BM BN BD (if second then "q2" else "q")]

def dotResult {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context : Nat) (even second : Bool)
    (q : Fin BM → Fin BD → ℝ) (old : Fin BM → Fin BN → ℝ) : BlockState :=
  let k := RealKey.loaded s ptrs start stride context even second
  let t := s.setReg "k" .real [BD, BN] (matrix k)
  let t := if second then t else t.setReg "qk" .real [BM, BN] (matrix (fun _ _ => 0))
  t.setReg "qk" .real [BM, BN]
    (matrix (fun r j => (if second then old r j else 0) + dotValues q k r j))

theorem load_dot_executes {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context : Nat) (even second : Bool)
    (q : Fin BM → Fin BD → ℝ) (old : Fin BM → Fin BN → ℝ)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BD] (if second then "q2" else "q") = some (matrix q))
    (ho : second = Bool.true → s.regs .real [BM, BN] "qk" = some (matrix old)) :
    stepStmts (loadDot BM BN BD stride context even second) s =
      some (dotResult s ptrs start stride context even second q old) := by
  have hl := RealKey.executes s ptrs start stride context even second hp hs hn
  unfold loadDot
  rw [List.append_assoc, stepStmts.append_some hl]
  let t := s.setReg "k" .real [BD, BN] (matrix (RealKey.loaded s ptrs start stride context even second))
  change stepStmts ((if second then [] else
    [.assign .real [BM, BN] "qk" (.full [BM, BN] (.const 0))]) ++
    [dot BM BN BD (if second then "q2" else "q")]) t = _
  cases second
  · simp only [Bool.false_eq_true, ↓reduceIte, List.singleton_append]
    have hz : stepStmt (.assign .real [BM, BN] "qk" (.full [BM, BN] (.const 0))) t =
        some (t.setReg "qk" .real [BM, BN] (matrix (fun _ _ => 0))) := by
      apply stepStmt_assign_eq_some
      simp only [evalOp_full, evalOp_const, bind, Option.bind]
      rfl
    rw [stepStmts.cons_some hz]
    have hd := dot_executes (t.setReg "qk" .real [BM, BN] (matrix (fun _ _ => 0))) "q"
      q (RealKey.loaded s ptrs start stride context even Bool.false) (fun _ _ => 0)
      (by simpa [t] using hq) (by simp [t]) (by simp)
    rw [stepStmts.cons_some hd, stepStmts.nil]
    rfl
  · simp only [↓reduceIte, List.nil_append]
    have hd := dot_executes t "q2" q (RealKey.loaded s ptrs start stride context even Bool.true) old
      (by simpa [t] using hq) (by simp [t]) (by simpa [t] using ho rfl)
    rw [stepStmts.cons_some hd, stepStmts.nil]
    rfl

def unscaled {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context blocks : Nat) (even : Bool)
    (q q2 : Fin BM → Fin BD → ℝ) : Fin BM → Fin BN → ℝ :=
  fun r j => dotValues q (RealKey.loaded s ptrs start stride context even Bool.false) r j +
    if 2 ≤ blocks then dotValues q2 (RealKey.loaded s ptrs start stride context even Bool.true) r j else 0

def dotsResult {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context blocks : Nat) (even : Bool)
    (q q2 : Fin BM → Fin BD → ℝ) : BlockState :=
  let t := dotResult s ptrs start stride context even Bool.false q (fun _ _ => 0)
  if 2 ≤ blocks then dotResult t ptrs start stride context even Bool.true q2
    (dotValues q (RealKey.loaded s ptrs start stride context even Bool.false)) else t

theorem dots_readback {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context blocks : Nat) (even : Bool) (q q2 : Fin BM → Fin BD → ℝ) :
    (dotsResult s ptrs start stride context blocks even q q2).regs .real [BM, BN] "qk" =
      some (matrix (unscaled s ptrs start stride context blocks even q q2)) := by
  unfold unscaled
  by_cases hb : 2 ≤ blocks <;> simp [dotsResult, dotResult, hb]

def dotsCode (BM BN BD stride context blocks : Nat) (even : Bool) : List Stmt :=
  loadDot BM BN BD stride context even Bool.false ++
    [.ifThen (.constBool (decide (2 ≤ blocks))) (loadDot BM BN BD stride context even Bool.true)]

theorem dots_execute {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context blocks : Nat) (even : Bool) (q q2 : Fin BM → Fin BD → ℝ)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BD] "q" = some (matrix q))
    (hq2 : 2 ≤ blocks → s.regs .real [BM, BD] "q2" = some (matrix q2)) :
    stepStmts (dotsCode BM BN BD stride context blocks even) s =
      some (dotsResult s ptrs start stride context blocks even q q2) := by
  have hf := load_dot_executes s ptrs start stride context even Bool.false q (fun _ _ => 0)
    hp hs hn hq (by simp)
  unfold dotsCode
  rw [stepStmts.append_some hf]
  let t := dotResult s ptrs start stride context even Bool.false q (fun _ _ => 0)
  change stepStmts [.ifThen (.constBool (decide (2 ≤ blocks)))
      (loadDot BM BN BD stride context even Bool.true)] t = _
  by_cases hb : 2 ≤ blocks
  · have hsecond := load_dot_executes t ptrs start stride context even Bool.true q2
      (dotValues q (RealKey.loaded s ptrs start stride context even Bool.false))
      (by simpa [t, dotResult] using hp) (by simpa [t, dotResult] using hs)
      (by simpa [t, dotResult] using hn) (by simpa [t, dotResult] using hq2 hb)
      (by simp [t, dotResult])
    have hi : stepStmt (.ifThen (.constBool (decide (2 ≤ blocks)))
        (loadDot BM BN BD stride context even Bool.true)) t =
        some (dotResult t ptrs start stride context even Bool.true q2
          (dotValues q (RealKey.loaded s ptrs start stride context even Bool.false))) := by
      simpa only [stepStmt, evalOp.eq_def, hb, decide_true, Tile.scalar_data] using hsecond
    rw [stepStmts.cons_some hi, stepStmts.nil]
    simp only [dotsResult, hb, ↓reduceIte, t]
  · have hi : stepStmt (.ifThen (.constBool (decide (2 ≤ blocks)))
        (loadDot BM BN BD stride context even Bool.true)) t = some t := by
      simp [stepStmt, evalOp.eq_def, hb]
    rw [stepStmts.cons_some hi, stepStmts.nil]
    simp only [dotsResult, hb, ↓reduceIte, t]

theorem code_split (BM BN BD stride context blocks : Nat) (scale : ℝ) (even swapped : Bool) :
    code BM BN BD stride context blocks scale even swapped =
      dotsCode BM BN BD stride context blocks even ++ [scaling BM BN scale swapped, masking BM BN] := by
  simp [code, dotsCode, loadDot, List.append_assoc]

def result {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context blocks : Nat) (scale : ℝ) (even : Bool)
    (rows : Fin BM → Nat) (q q2 : Fin BM → Fin BD → ℝ) : BlockState :=
  let t := dotsResult s ptrs start stride context blocks even q q2
  let score := fun r j => unscaled s ptrs start stride context blocks even q q2 r j * scale
  (t.setReg "qk" .real [BM, BN] (matrix score)).setReg "qk" .real [BM, BN]
    (extendedMatrix (masked score start rows))

theorem executes {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context blocks : Nat) (scale : ℝ) (even swapped : Bool)
    (rows : Fin BM → Nat) (q q2 : Fin BM → Fin BD → ℝ)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BD] "q" = some (matrix q))
    (hq2 : 2 ≤ blocks → s.regs .real [BM, BD] "q2" = some (matrix q2)) :
    stepStmts (code BM BN BD stride context blocks scale even swapped) s =
      some (result s ptrs start stride context blocks scale even rows q q2) := by
  rw [code_split, stepStmts.append_some (dots_execute s ptrs start stride context blocks even q q2 hp hs hn hq hq2)]
  let t := dotsResult s ptrs start stride context blocks even q q2
  have hscaled := scaling_executes t scale swapped (unscaled s ptrs start stride context blocks even q q2)
    (dots_readback s ptrs start stride context blocks even q q2)
  rw [stepStmts.cons_some hscaled]
  let u := t.setReg "qk" .real [BM, BN]
    (matrix (fun r j => unscaled s ptrs start stride context blocks even q q2 r j * scale))
  have hmask := masking_executes u start rows
    (fun r j => unscaled s ptrs start stride context blocks even q q2 r j * scale)
    (by by_cases hb : 2 ≤ blocks <;> simpa [u, t, dotsResult, dotResult, hb] using hs)
    (by by_cases hb : 2 ≤ blocks <;> simpa [u, t, dotsResult, dotResult, hb] using hm)
    (by by_cases hb : 2 ≤ blocks <;> simpa [u, t, dotsResult, dotResult, hb] using hn)
    (by simp [u])
  rw [stepStmts.cons_some hmask, stepStmts.nil]
  rfl

theorem readback {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context blocks : Nat) (scale : ℝ) (even : Bool)
    (rows : Fin BM → Nat) (q q2 : Fin BM → Fin BD → ℝ) :
    (result s ptrs start stride context blocks scale even rows q q2).regs .real [BM, BN] "qk" =
      some (extendedMatrix (masked
        (fun r j => unscaled s ptrs start stride context blocks even q q2 r j * scale) start rows)) := by
  simp [result]

theorem state_frame {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context blocks : Nat) (scale : ℝ) (even : Bool)
    (rows : Fin BM → Nat) (q q2 : Fin BM → Fin BD → ℝ) :
    {result s ptrs start stride context blocks scale even rows q q2 with regs := s.regs} = s := by
  by_cases hb : 2 ≤ blocks <;> simp [result, dotsResult, dotResult, hb, BlockState.setReg]

theorem register_frame {BM BN BD : Nat} (s : BlockState) (ptrs : Tile .ptr [BD, BN])
    (start : Int) (stride context blocks : Nat) (scale : ℝ) (even : Bool)
    (rows : Fin BM → Nat) (q q2 : Fin BM → Fin BD → ℝ)
    (dtype : TileDType) (shape : TileShape) (name : RegName) (hk : name ≠ "k") (hqk : name ≠ "qk") :
    (result s ptrs start stride context blocks scale even rows q q2).regs dtype shape name =
      s.regs dtype shape name := by
  by_cases hb : 2 ≤ blocks <;> simp [result, dotsResult, dotResult, hb, hk, hqk]

theorem source_executes (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (s : BlockState) (layout : Nat) (visit : Int) (ptrs : Tile .ptr [BD, BN])
    (rows : Fin BM → Nat) (q q2 : Fin BM → Fin BD → ℝ)
    (hl : s.regs .nat [] "layout_h" = some (Tile.scalar layout))
    (hi : s.regs .int [] "col_idx_idx" = some (Tile.scalar visit))
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hm : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BD] "q" = some (matrix q))
    (hq2 : 2 ≤ p.blocks → s.regs .real [BM, BD] "q2" = some (matrix q2)) :
    stepStmts (RealProgram.realPart (RealProgram.normalizationPrefix swapped BM BN BD p)) s =
      some (result (RealColumn.result s BN p.cols layout p.columnStride visit) ptrs
        (RealColumn.column s p.cols layout p.columnStride visit * (BN : Int))
        p.kRow p.context p.blocks p.scale p.evenN rows q q2) := by
  rw [prefix_matches, stepStmts.append_some (RealColumn.executes s BN p.cols layout p.columnStride visit hl hi)]
  apply executes
  · simpa [RealColumn.result] using hp
  · exact (RealColumn.readbacks s BN p.cols layout p.columnStride visit).2
  · simpa [RealColumn.result] using hm
  · simpa [RealColumn.result] using hn
  · simpa [RealColumn.result] using hq
  · intro hb
    simpa [RealColumn.result] using hq2 hb

/-- The initial-array score is recovered on a stored row. Causal visibility
forces the actual upper-bound K mask to be active; invisible lanes become
bottom even when their arbitrary loaded values differ from the input array.
The Q producers and active K address pins remain prefix/layout obligations. -/
theorem score_eq_formula (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs)
    (BM BN BD blocks context visit stride : Nat) (scale : ℝ) (even : Bool)
    (s : BlockState) (ptrs : Tile .ptr [BD, BN]) (q q2 : Fin BM → Fin BD → ℝ)
    (r : Fin BM) (j : Fin BN)
    (hrow : schedule.queryBlock * BM + r.val < context)
    (hq : ∀ d, q r d = xs.query (schedule.queryBlock * BM + r.val) d.val)
    (hq2 : 2 ≤ blocks → ∀ d, q2 r d = xs.query (schedule.queryBlock * BM + r.val) (BD + d.val))
    (hk : ∀ d, RealKey.active (RealValueFormula.keyStart schedule xs BN visit) context even j →
      s.readMem (RealKey.address ptrs (RealValueFormula.keyStart schedule xs BN visit) stride Bool.false d j).1
        (RealKey.address ptrs (RealValueFormula.keyStart schedule xs BN visit) stride Bool.false d j).2 =
      xs.key (CSRMath.selectedKey schedule xs BN visit j) d.val)
    (hk2 : 2 ≤ blocks → ∀ d, RealKey.active (RealValueFormula.keyStart schedule xs BN visit) context even j →
      s.readMem (RealKey.address ptrs (RealValueFormula.keyStart schedule xs BN visit) stride Bool.true d j).1
        (RealKey.address ptrs (RealValueFormula.keyStart schedule xs BN visit) stride Bool.true d j).2 =
      xs.key (CSRMath.selectedKey schedule xs BN visit j) (BD + d.val)) :
    masked (fun r j => unscaled s ptrs (RealValueFormula.keyStart schedule xs BN visit)
      stride context blocks even q q2 r j * scale)
      (RealValueFormula.keyStart schedule xs BN visit) (fun r => schedule.queryBlock * BM + r.val) r j =
        CSRMath.score schedule xs BN BD blocks (schedule.queryBlock * BM + r.val) scale visit j := by
  let start := RealValueFormula.keyStart schedule xs BN visit
  have hkey : CSRMath.selectedKey schedule xs BN visit j = start + (j.val : Int) := rfl
  change masked (fun r j => unscaled s ptrs start stride context blocks even q q2 r j * scale)
    start (fun r => schedule.queryBlock * BM + r.val) r j = _
  by_cases hv : start + (j.val : Int) ≤ (schedule.queryBlock * BM + r.val : Nat)
  · have ha : RealKey.active start context even j := by
      right
      omega
    have hf : dotValues q (RealKey.loaded s ptrs start stride context even Bool.false) r j =
        ∑ d : Fin BD, xs.query (schedule.queryBlock * BM + r.val) d.val *
          xs.key (CSRMath.selectedKey schedule xs BN visit j) d.val := by
      apply Finset.sum_congr rfl
      intro d _
      simp only [RealKey.loaded, ha, ↓reduceIte, hq d]
      exact congrArg (fun x => xs.query (schedule.queryBlock * BM + r.val) d.val * x) (hk d ha)
    have hvalue : unscaled s ptrs start stride context blocks even q q2 r j * scale =
        CSRMath.dotScore xs BD blocks (schedule.queryBlock * BM + r.val)
          (CSRMath.selectedKey schedule xs BN visit j) scale := by
      unfold unscaled CSRMath.dotScore
      rw [hf]
      by_cases hb : 2 ≤ blocks
      · have hsecond : dotValues q2 (RealKey.loaded s ptrs start stride context even Bool.true) r j =
            ∑ d : Fin BD, xs.query (schedule.queryBlock * BM + r.val) (BD + d.val) *
              xs.key (CSRMath.selectedKey schedule xs BN visit j) (BD + d.val) := by
          apply Finset.sum_congr rfl
          intro d _
          simp only [RealKey.loaded, ha, ↓reduceIte, hq2 hb d]
          exact congrArg (fun x => xs.query (schedule.queryBlock * BM + r.val) (BD + d.val) * x) (hk2 hb d ha)
        simp only [hb, ↓reduceIte, hsecond]
      · simp only [hb, ↓reduceIte]
    simpa only [masked, CSRMath.score, hkey, hv, ↓reduceIte] using
      congrArg (fun x : ℝ => (x : WithBot ℝ)) hvalue
  · simp only [masked, CSRMath.score, hkey, hv, ↓reduceIte]

theorem load_dot_safe {BM BN BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BD, BN]) (start : Int) (stride context : Nat) (even second : Bool)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hb : RealKey.Bounds bounds ptrs start stride context even second) :
    Stmt.TraceSafeList bounds (loadDot BM BN BD stride context even second) s := by
  unfold loadDot
  rw [List.append_assoc]
  apply TraceProofs.append_safe _ s (RealKey.code_safe bounds s ptrs start stride context even second hp hs hn hb)
  intro t _
  apply TraceProofs.all_safe
  intro st hst u
  cases second
  · simp only [Bool.false_eq_true, ↓reduceIte, List.singleton_append, List.mem_cons, List.not_mem_nil, or_false] at hst
    rcases hst with rfl | rfl <;> simp [Stmt.TraceSafe, dot, Op.SafeAt.eq_def]
  · simp only [↓reduceIte, List.nil_append, List.mem_singleton] at hst
    subst st
    simp [Stmt.TraceSafe, dot, Op.SafeAt.eq_def]

theorem dots_safe {BM BN BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BD, BN]) (start : Int) (stride context blocks : Nat) (even : Bool)
    (q : Fin BM → Fin BD → ℝ)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BD] "q" = some (matrix q))
    (hb : RealKey.Bounds bounds ptrs start stride context even Bool.false)
    (hb2 : 2 ≤ blocks → RealKey.Bounds bounds ptrs start stride context even Bool.true) :
    Stmt.TraceSafeList bounds (dotsCode BM BN BD stride context blocks even) s := by
  unfold dotsCode
  apply TraceProofs.append_safe _ s (load_dot_safe bounds s ptrs start stride context even Bool.false hp hs hn hb)
  intro t ht
  have he := load_dot_executes s ptrs start stride context even Bool.false q (fun _ _ => 0)
    hp hs hn hq (by simp)
  rw [he] at ht
  cases ht
  refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => Stmt.TraceSafeList.nil_intro)
  rw [Stmt.TraceSafe]
  simp only [evalOp.eq_def, Op.SafeAt.eq_def, Tile.scalar_data, true_and]
  by_cases htwo : 2 ≤ blocks
  · simp only [htwo, decide_true, ↓reduceIte]
    apply load_dot_safe (ptrs := ptrs) (start := start)
    · simpa [dotResult] using hp
    · simpa [dotResult] using hs
    · simpa [dotResult] using hn
    · exact hb2 htwo
  · simp [htwo]

theorem code_safe {BM BN BD : Nat} (bounds : RegionBounds) (s : BlockState)
    (ptrs : Tile .ptr [BD, BN]) (start : Int) (stride context blocks : Nat)
    (scale : ℝ) (even swapped : Bool) (q : Fin BM → Fin BD → ℝ)
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hs : s.regs .int [] "start_n" = some (Tile.scalar start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BD] "q" = some (matrix q))
    (hb : RealKey.Bounds bounds ptrs start stride context even Bool.false)
    (hb2 : 2 ≤ blocks → RealKey.Bounds bounds ptrs start stride context even Bool.true) :
    Stmt.TraceSafeList bounds (code BM BN BD stride context blocks scale even swapped) s := by
  rw [code_split]
  apply TraceProofs.append_safe _ s (dots_safe bounds s ptrs start stride context blocks even q hp hs hn hq hb hb2)
  intro t _
  apply TraceProofs.all_safe
  intro st hst u
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl
  · cases swapped <;> simp [Stmt.TraceSafe, scaling, Op.SafeAt.eq_def]
  · simp [Stmt.TraceSafe, masking, causal, Op.SafeAt.eq_def]

theorem source_safe (bounds : RegionBounds) (swapped : Bool) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (s : BlockState) (layout : Nat) (visit : Int)
    (ptrs : Tile .ptr [BD, BN]) (q : Fin BM → Fin BD → ℝ)
    (hl : s.regs .nat [] "layout_h" = some (Tile.scalar layout))
    (hi : s.regs .int [] "col_idx_idx" = some (Tile.scalar visit))
    (hp : s.regs .ptr [BD, BN] "k_ptrs" = some ptrs)
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec Fin.val))
    (hq : s.regs .real [BM, BD] "q" = some (matrix q))
    (hc : RealColumn.address layout p.columnStride visit < bounds (Region.cast p.cols))
    (hb : RealKey.Bounds bounds ptrs (RealColumn.column s p.cols layout p.columnStride visit * (BN : Int))
      p.kRow p.context p.evenN Bool.false)
    (hb2 : 2 ≤ p.blocks → RealKey.Bounds bounds ptrs
      (RealColumn.column s p.cols layout p.columnStride visit * (BN : Int)) p.kRow p.context p.evenN Bool.true) :
    Stmt.TraceSafeList bounds (RealProgram.realPart (RealProgram.normalizationPrefix swapped BM BN BD p)) s := by
  rw [prefix_matches]
  apply TraceProofs.append_safe _ s (RealColumn.code_safe bounds s BN p.cols layout p.columnStride visit hl hi hc)
  intro t ht
  rw [RealColumn.executes s BN p.cols layout p.columnStride visit hl hi] at ht
  cases ht
  apply code_safe (ptrs := ptrs)
    (start := RealColumn.column s p.cols layout p.columnStride visit * (BN : Int)) (q := q)
  · simpa [RealColumn.result] using hp
  · exact (RealColumn.readbacks s BN p.cols layout p.columnStride visit).2
  · simpa [RealColumn.result] using hn
  · simpa [RealColumn.result] using hq
  · exact hb
  · exact hb2

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealScore
