import bench.tritonbench_g.block_sparse_attn.Proofs.RealLoop
import VeriTile.Triton.KernelLemmas.ScatterStore

/-! The actual final output stores. Both channel blocks always mask query
rows, independently of EVEN_M. The explicit scatter result exposes active
writes and preserves complete memory cells everywhere else. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealStores
open VeriTile.Triton RealNormalizationValues
noncomputable section
set_option linter.unusedSimpArgs false

def offsetOp (BM BD : Nat) (p : RealProgram.Parameters) : Op .nat [BM, BD] :=
  .add .nat (.consR (.consL .nil))
    (.add .nat .scalarL
      (.add .nat .nil
        (.mul .nat .nil (.ref .nat [] "off_b") (.constNat p.oBatch))
        (.mul .nat .nil (.ref .nat [] "off_h") (.constNat p.oHead)))
      (.mul .nat .scalarR (.expandDim 1 (.ref .nat [BM] "offs_m")) (.constNat p.oRow)))
    (.expandDim 0 (.ref .nat [BD] "offs_d"))

def pointer (BM BD : Nat) (second : Bool) : Op .ptr [BM, BD] :=
  if second then .ptrAdd .scalarR (.ref .ptr [BM, BD] "out_ptrs") (.constNat BD)
    else .ref .ptr [BM, BD] "out_ptrs"

def store (BM BD : Nat) (second : Bool) : Stmt :=
  .store .real [BM, BD] (.ptr (pointer BM BD second))
    (.ref .real [BM, BD] (if second then "acc2" else "acc")) (.mask (RealQuery.mask BM BD))

def code (BM BD : Nat) (p : RealProgram.Parameters) : List Stmt :=
  [.assign .nat [BM, BD] "off_o" (offsetOp BM BD p),
   .assign .ptr [BM, BD] "out_ptrs" (.ptrAdd .scalarL (.ptrBase p.Out) (.ref .nat [BM, BD] "off_o")),
   store BM BD Bool.false,
   .ifThen (.constBool (decide (2 ≤ p.blocks))) [store BM BD Bool.true]]

set_option maxRecDepth 22000 in
set_option maxHeartbeats 4000000 in
theorem matches_source (swapped : Bool) (BM BN BD : Nat) (p : RealProgram.Parameters) :
    RealProgram.realPart (RealProgram.parts swapped BM BN BD p).after = code BM BD p := by
  cases swapped
  all_goals
    simp [code, offsetOp, store, pointer, RealQuery.mask,
      RealProgram.realPart, RealProgram.parts, RealProgram.source, TypedSource.specialize,
      Source.intRange?, Source.prefixBefore, ComputeKernel.surfaceBody,
      ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    exact ⟨⟨rfl, rfl⟩, rfl⟩

def offset (s : BlockState) (BM BD : Nat) (p : RealProgram.Parameters) (second : Bool)
    (r : Fin BM) (d : Fin BD) : Nat :=
  RealSetup.batch s p * p.oBatch + RealSetup.head s p * p.oHead +
    RealSetup.rows s BM r * p.oRow + d.val + if second then BD else 0

def pointers (s : BlockState) (BM BD : Nat) (p : RealProgram.Parameters) : Tile .ptr [BM, BD] :=
  Tile.mat fun r d => (p.Out, offset s BM BD p Bool.false r d)

def prepared (s t : BlockState) (BM BD : Nat) (p : RealProgram.Parameters) : BlockState :=
  (t.setReg "off_o" .nat [BM, BD] (Tile.mat (offset s BM BD p Bool.false))).setReg
    "out_ptrs" .ptr [BM, BD] (pointers s BM BD p)

/-- Each store is a row-masked scatter, including when EVEN_M is true. -/
def write (s t : BlockState) (BM BD : Nat) (p : RealProgram.Parameters) (second : Bool)
    (values : Fin BM → Fin BD → ℝ) : BlockState :=
  (TileShape.allIndices [BM, BD]).foldl (fun u i =>
    if RealSetup.rows s BM i.1 < p.context then
      u.writeMem p.Out (offset s BM BD p second i.1 i.2.1) (values i.1 i.2.1) else u) t

@[simp] theorem write_register (s t : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (values : Fin BM → Fin BD → ℝ) (dtype shape name) :
    (write s t BM BD p second values).regs dtype shape name = t.regs dtype shape name := by
  simpa only [write, decide_eq_true_eq] using foldl_store_regs
    (fun i : TileIndex [BM, BD] => offset s BM BD p second i.1 i.2.1)
    (fun i => values i.1 i.2.1) (fun i => decide (RealSetup.rows s BM i.1 < p.context))
    (TileShape.allIndices [BM, BD]) t dtype shape name

theorem write_control (s t : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (values : Fin BM → Fin BD → ℝ) :
    {write s t BM BD p second values with mem := t.mem} = t := by
  unfold write
  generalize TileShape.allIndices [BM, BD] = indices
  induction indices generalizing t with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons]
    split_ifs
    · have h := congrArg (fun u => {u with mem := t.mem})
        (ih (t.writeMem p.Out (offset s BM BD p second i.1 i.2.1) (values i.1 i.2.1)))
      exact h
    · exact ih t

theorem write_frame (s t : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (values : Fin BM → Fin BD → ℝ) (region : RegionName) (address : Nat)
    (h : ∀ r d, RealSetup.rows s BM r < p.context →
      ¬(p.Out = region ∧ offset s BM BD p second r d = address)) :
    (write s t BM BD p second values).mem region address = t.mem region address := by
  exact foldl_store_preserve_cell _ _ _ region address _ t (fun i _ hi => h i.1 i.2.1 hi)

theorem write_readback (s t : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (values : Fin BM → Fin BD → ℝ) (r : Fin BM) (d : Fin BD)
    (hr : RealSetup.rows s BM r < p.context)
    (hinj : Function.Injective (fun i : TileIndex [BM, BD] => offset s BM BD p second i.1 i.2.1)) :
    (write s t BM BD p second values).readMem p.Out (offset s BM BD p second r d) = values r d := by
  exact BlockState.scatter_readback_prop_masked_nd_of_true t
    (fun i : TileIndex [BM, BD] => offset s BM BD p second i.1 i.2.1)
    (fun i => values i.1 i.2.1) (fun i => RealSetup.rows s BM i.1 < p.context)
    (r, d, PUnit.unit) hr (fun _ _ heq => hinj heq)

theorem offset_eval (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (h : RealLoop.Fixed s BM BN BD p t) :
    evalOp (offsetOp BM BD p) t = some (Tile.mat (offset s BM BD p Bool.false)) := by
  simp [offsetOp, evalOp.eq_def, h.batch, h.head, h.rows, h.channels, bind, Option.bind]
  rfl

theorem pointer_eval (s t : BlockState) (BM BD : Nat) (p : RealProgram.Parameters) (second : Bool)
    (hp : t.regs .ptr [BM, BD] "out_ptrs" = some (pointers s BM BD p)) :
    evalOp (pointer BM BD second) t =
      some (Tile.mat (fun r d => (p.Out, offset s BM BD p second r d))) := by
  cases second <;> simp [pointer, evalOp.eq_def, hp, bind, Option.bind]
  all_goals rfl

theorem store_executes (s t : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (second : Bool) (values : Fin BM → Fin BD → ℝ)
    (hp : t.regs .ptr [BM, BD] "out_ptrs" = some (pointers s BM BD p))
    (hr : t.regs .nat [BM] "offs_m" = some (Tile.vec (RealSetup.rows s BM)))
    (hctx : t.regs .nat [] "q_seq_len" = some (Tile.scalar p.context))
    (hv : t.regs .real [BM, BD] (if second then "acc2" else "acc") = some (matrix values)) :
    stepStmt (store BM BD second) t = some (write s t BM BD p second values) := by
  have hm := RealQuery.mask_eval (BD := BD) t (RealSetup.rows s BM) p.context hr hctx
  rw [store, stepStmt]
  simp only [pointer_eval s t BM BD p second hp, hm, evalOp_ref, hv,
    bind, Option.bind_some, Option.map_some, Tile.mat, decide_eq_true_eq, BlockState.writeMemTyped_real]
  rfl

def result (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (x : Inputs BM BN BD) : BlockState :=
  let u := write s (prepared s t BM BD p) BM BD p Bool.false x.accumulator
  if 2 ≤ p.blocks then write s u BM BD p Bool.true x.secondAccumulator else u

theorem executes (s t : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (x : Inputs BM BN BD) (hf : RealLoop.Fixed s BM BN BD p t) (hc : RealLoop.CarryAt x p.blocks t) :
    stepStmts (code BM BD p) t = some (result s t BM BN BD p x) := by
  have ho := stepStmt_assign_eq_some (name := "off_o") (offset_eval s t BM BN BD p hf)
  have hp : stepStmt (.assign .ptr [BM, BD] "out_ptrs"
      (.ptrAdd .scalarL (.ptrBase p.Out) (.ref .nat [BM, BD] "off_o")))
      (t.setReg "off_o" .nat [BM, BD] (Tile.mat (offset s BM BD p Bool.false))) =
      some (prepared s t BM BD p) := by
    apply stepStmt_assign_eq_some
    simp [evalOp.eq_def, bind, Option.bind, pointers]
    apply Tile.ext
    rintro ⟨r, d, ⟨⟩⟩
    simp only [Tile.ptrAdd, Tile.mat, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, Nat.zero_add]
  have hfirst := store_executes s (prepared s t BM BD p) BM BD p Bool.false x.accumulator
    (by simp [prepared]) (by simpa [prepared] using hf.rows)
    (by simpa [prepared] using hf.context) (by simpa [prepared] using hc.accumulator)
  rw [code, stepStmts.cons_some ho, stepStmts.cons_some hp, stepStmts.cons_some hfirst]
  by_cases hb : 2 ≤ p.blocks
  · have hsecond := store_executes s
      (write s (prepared s t BM BD p) BM BD p Bool.false x.accumulator) BM BD p Bool.true x.secondAccumulator
      (by simp [prepared]) (by simpa [prepared] using hf.rows)
      (by simpa [prepared] using hf.context) (by simpa [prepared] using hc.secondAccumulator hb)
    simp [stepStmts, stepStmt, evalOp.eq_def, hb, hsecond, result]
  · simp [stepStmts, stepStmt, evalOp.eq_def, hb, result]

theorem source_executes (swapped : Bool) (s t : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (x : Inputs BM BN BD)
    (hf : RealLoop.Fixed s BM BN BD p t) (hc : RealLoop.CarryAt x p.blocks t) :
    stepStmts (RealProgram.realPart (RealProgram.parts swapped BM BN BD p).after) t =
      some (result s t BM BN BD p x) := by
  rw [matches_source]
  exact executes s t BM BN BD p x hf hc

/-- The whole real source executes, including the masked final stores.
This execution theorem still uses the source-derived history; its independent
attention formula and allocation contract are separate obligations. -/
theorem kernel_executes (swapped : Bool) (s : BlockState) (BM BN BD : Nat)
    (p : RealProgram.Parameters) (hN : 0 < BN) :
    ∃ t, RealLoop.Fixed s BM BN BD p t ∧
      stepStmts (RealProgram.kernel swapped BM BN BD p).toAlgKernel.body s =
        some (result s t BM BN BD p (RealLoop.carry s BM BN BD p (RealLoop.count s p))) := by
  obtain ⟨t, ht, hf, hc⟩ := RealLoop.prologue_loop_executes swapped s BM BN BD p hN
  refine ⟨t, hf, ?_⟩
  rw [RealProgram.body_split, stepStmts.append_some ht]
  exact source_executes swapped s t BM BN BD p _ hf hc

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealStores
