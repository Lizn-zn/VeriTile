import bench.tritonbench_g.attn_fwd_causal.Proofs.RealCoordinates

/-! Source index vectors and pointer expressions. Query, value and output
share the source Q strides, while the key uses stride_kn and an unscaled
head lane. The unused source K/V/Out stride parameters are not substituted. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPointers
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def rowExpr (BM : Nat) : Op .nat [BM] :=
  .add .nat .scalarL (.mul .nat .nil (.ref .nat [] "start_m") (.constNat BM)) (.arange BM)

def rowPointerExpr (R : RegionName) (BR BD strideRow strideDim : Nat) (rows : RegName) : Op .ptr [BR,BD] :=
  .ptrAdd .scalarL (.ptrBase R)
    (.add .nat (.consR (.consL .nil))
      (.add .nat .scalarL (.ref .nat [] "qvk_offset")
        (.mul .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BR] rows)) (.constNat strideRow)))
      (.mul .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BD] "offs_k")) (.constNat strideDim)))

def keyExpr (K : RegionName) (BD BN strideKN : Nat) : Op .ptr [BD,BN] :=
  .ptrAdd .scalarL (.ptrBase K)
    (.add .nat (.consR (.consL .nil))
      (.add .nat .scalarL (.ref .nat [] "qvk_offset") (.expandDim ⟨1, by simp⟩ (.ref .nat [BD] "offs_k")))
      (.mul .nat .scalarR (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n")) (.constNat strideKN)))

def queryScaleExpr (QS : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase QS) (.add .nat .nil (.ref .nat [] "q_scale_offset") (.ref .nat [] "start_m"))

def keyScaleExpr (KS : RegionName) : Op .ptr [] :=
  .ptrAdd .nil (.ptrBase KS) (.ref .nat [] "k_scale_offset")

def code (c : Config) (Out : RegionName) (BM BN BD : Nat) : List Stmt :=
  [.assign .nat [BM] "offs_m" (rowExpr BM),
   .assign .nat [BN] "offs_n" (.arange BN),
   .assign .nat [BD] "offs_k" (.arange BD),
   .assign .ptr [BM,BD] "Q_ptrs" (rowPointerExpr c.Q BM BD c.strideM c.strideD "offs_m"),
   .assign .ptr [] "Q_scale_ptr" (queryScaleExpr c.QS),
   .assign .ptr [BD,BN] "K_ptrs" (keyExpr c.K BD BN c.strideKN),
   .assign .ptr [] "K_scale_ptr" (keyScaleExpr c.KS),
   .assign .ptr [BN,BD] "V_ptrs" (rowPointerExpr c.V BN BD c.strideM c.strideD "offs_n"),
   .assign .ptr [BM,BD] "O_block_ptr" (rowPointerExpr Out BM BD c.strideM c.strideD "offs_m")]

def rows (BR start : Nat) : Tile .nat [BR] := ⟨fun i => start+i.1.val⟩
def rowPointer (R : RegionName) (BR BD base start strideRow strideDim : Nat) : Tile .ptr [BR,BD] :=
  ⟨fun i => (R.cast,base+(start+i.1.val)*strideRow+i.2.1.val*strideDim)⟩
def queryPointer (c : Config) (BM BD : Nat) : Tile .ptr [BM,BD] :=
  rowPointer c.Q BM BD c.base (c.startM*BM) c.strideM c.strideD
def outputPointer (c : Config) (Out : RegionName) (BM BD : Nat) : Tile .ptr [BM,BD] :=
  rowPointer Out BM BD c.base (c.startM*BM) c.strideM c.strideD
def queryScalePointer (c : Config) : Tile .ptr [] := Tile.scalar (c.QS.cast,c.queryScaleBase+c.startM)

theorem eval_rows (s : BlockState) (BM startM : Nat)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar startM)) :
    evalOp (rowExpr BM) s = some (rows BM (startM*BM)) := by
  simp only [rowExpr, evalOp.eq_def, hm, bind, Option.bind]
  rfl

theorem eval_row_pointer (s : BlockState) (R : RegionName) (BR BD base start strideRow strideDim : Nat)
    (name : RegName)
    (hb : s.regs .nat [] "qvk_offset" = some (Tile.scalar base))
    (hr : s.regs .nat [BR] name = some (rows BR start))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val))) :
    evalOp (rowPointerExpr R BR BD strideRow strideDim name) s =
      some (rowPointer R BR BD base start strideRow strideDim) := by
  simp only [rowPointerExpr, evalOp.eq_def, hb, hr, hk, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨r,d,⟨⟩⟩
  simp [rowPointer, rows, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.expandDim_data,
    Tile.vec, TileShape.dropInsertedIndex, NumericDType.mul, NumericDType.add, -Tile.scalar_eta]

theorem eval_key_pointer (s : BlockState) (c : Config) (BD BN : Nat)
    (hb : s.regs .nat [] "qvk_offset" = some (Tile.scalar c.base))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hk : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val))) :
    evalOp (keyExpr c.K BD BN c.strideKN) s = some (keyPointer c BD BN 0) := by
  simp only [keyExpr, evalOp.eq_def, hb, hn, hk, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨d,j,⟨⟩⟩
  simp [keyPointer, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, Tile.expandDim_data,
    Tile.vec, TileShape.dropInsertedIndex, NumericDType.mul, NumericDType.add, -Tile.scalar_eta]

theorem eval_query_scale_pointer (s : BlockState) (c : Config)
    (hb : s.regs .nat [] "q_scale_offset" = some (Tile.scalar c.queryScaleBase))
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar c.startM)) :
    evalOp (queryScaleExpr c.QS) s = some (queryScalePointer c) := by
  simp only [queryScaleExpr, evalOp.eq_def, hb, hm, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨⟩
  simp [queryScalePointer, Tile.ptrAdd_data, Tile.bop_data, Tile.scalar, NumericDType.add, -Tile.scalar_eta]

theorem eval_key_scale_pointer (s : BlockState) (c : Config) (BN : Nat)
    (hb : s.regs .nat [] "k_scale_offset" = some (Tile.scalar c.keyScaleBase)) :
    evalOp (keyScaleExpr c.KS) s = some (keyScalePointer c BN 0) := by
  simp only [keyScaleExpr, evalOp.eq_def, hb, bind, Option.bind]
  congr 1
  apply Tile.ext
  rintro ⟨⟩
  simp [keyScalePointer, Tile.scalar, -Tile.scalar_eta]

def writes : List RegName := ["offs_m","offs_n","offs_k","Q_ptrs","Q_scale_ptr","K_ptrs","K_scale_ptr","V_ptrs","O_block_ptr"]

structure Registers (c : Config) (Out : RegionName) (BM BN BD : Nat) (s : BlockState) : Prop where
  rows : s.regs .nat [BM] "offs_m" = some (rows BM (c.startM*BM))
  keys : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val))
  head : s.regs .nat [BD] "offs_k" = some (Tile.vec (fun i => i.val))
  query : s.regs .ptr [BM,BD] "Q_ptrs" = some (queryPointer c BM BD)
  queryScale : s.regs .ptr [] "Q_scale_ptr" = some (queryScalePointer c)
  caller : RealTwoLoops.Caller s c BN BD
  output : s.regs .ptr [BM,BD] "O_block_ptr" = some (outputPointer c Out BM BD)

theorem executes (s : BlockState) (c : Config) (Out : RegionName) (BM BN BD : Nat)
    (hm : s.regs .nat [] "start_m" = some (Tile.scalar c.startM))
    (hb : s.regs .nat [] "qvk_offset" = some (Tile.scalar c.base))
    (hqbase : s.regs .nat [] "q_scale_offset" = some (Tile.scalar c.queryScaleBase))
    (hkbase : s.regs .nat [] "k_scale_offset" = some (Tile.scalar c.keyScaleBase)) :
    ∃ t, stepStmts (code c Out BM BN BD) s = some t ∧ Registers c Out BM BN BD t ∧ Frame writes s t := by
  let s1 := s.setReg "offs_m" .nat [BM] (rows BM (c.startM*BM))
  let s2 := s1.setReg "offs_n" .nat [BN] (Tile.vec (fun i => i.val))
  let s3 := s2.setReg "offs_k" .nat [BD] (Tile.vec (fun i => i.val))
  let s4 := s3.setReg "Q_ptrs" .ptr [BM,BD] (queryPointer c BM BD)
  let s5 := s4.setReg "Q_scale_ptr" .ptr [] (queryScalePointer c)
  let s6 := s5.setReg "K_ptrs" .ptr [BD,BN] (keyPointer c BD BN 0)
  let s7 := s6.setReg "K_scale_ptr" .ptr [] (keyScalePointer c BN 0)
  let s8 := s7.setReg "V_ptrs" .ptr [BN,BD] (valuePointer c BN BD 0)
  have hq := eval_row_pointer s3 c.Q BM BD c.base (c.startM*BM) c.strideM c.strideD "offs_m"
    (by simp [s3, s2, s1, hb]) (by simp [s3, s2, s1]) (by simp [s3])
  change evalOp (rowPointerExpr c.Q BM BD c.strideM c.strideD "offs_m") s3 = some (queryPointer c BM BD) at hq
  have hqs := eval_query_scale_pointer s4 c (by simp [s4, s3, s2, s1, hqbase]) (by simp [s4, s3, s2, s1, hm])
  have hk := eval_key_pointer s5 c BD BN (by simp [s5, s4, s3, s2, s1, hb])
    (by simp [s5, s4, s3, s2]) (by simp [s5, s4, s3])
  have hks := eval_key_scale_pointer s6 c BN (by simp [s6, s5, s4, s3, s2, s1, hkbase])
  have hv : evalOp (rowPointerExpr c.V BN BD c.strideM c.strideD "offs_n") s7 = some (valuePointer c BN BD 0) := by
    have h := eval_row_pointer s7 c.V BN BD c.base 0 c.strideM c.strideD "offs_n"
      (by simp [s7, s6, s5, s4, s3, s2, s1, hb])
      (by simp [s7, s6, s5, s4, s3, s2, rows, Tile.vec]) (by simp [s7, s6, s5, s4, s3])
    rw [h]
    congr 1
    apply Tile.ext
    intro i
    simp [rowPointer, valuePointer]
  have ho := eval_row_pointer s8 Out BM BD c.base (c.startM*BM) c.strideM c.strideD "offs_m"
    (by simp [s8, s7, s6, s5, s4, s3, s2, s1, hb])
    (by simp [s8, s7, s6, s5, s4, s3, s2, s1]) (by simp [s8, s7, s6, s5, s4, s3])
  change evalOp (rowPointerExpr Out BM BD c.strideM c.strideD "offs_m") s8 = some (outputPointer c Out BM BD) at ho
  simp only [code]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (eval_rows s BM c.startM hm)),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (.arange BN) s1 = some (Tile.vec (fun i => i.val)) from by simp [evalOp.eq_def])),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (show evalOp (.arange BD) s2 = some (Tile.vec (fun i => i.val)) from by simp [evalOp.eq_def])),
    stepStmts.cons_some (stepStmt_assign_eq_some hq), stepStmts.cons_some (stepStmt_assign_eq_some hqs),
    stepStmts.cons_some (stepStmt_assign_eq_some hk), stepStmts.cons_some (stepStmt_assign_eq_some hks),
    stepStmts.cons_some (stepStmt_assign_eq_some hv), stepStmts.cons_some (stepStmt_assign_eq_some ho), stepStmts.nil]
  refine ⟨_, rfl, ?_, rfl, rfl, rfl, ?_⟩
  · refine ⟨?_, ?_, ?_, ?_, ?_, ⟨?_, ?_, ?_, ?_⟩, ?_⟩ <;>
      simp [s8, s7, s6, s5, s4, s3, s2, s1, hm, queryPointer, outputPointer]
  · intro dt shape name hn
    simp only [writes, List.mem_cons, not_or] at hn
    simp [s8, s7, s6, s5, s4, s3, s2, s1, hn.1, hn.2.1, hn.2.2.1, hn.2.2.2.1,
      hn.2.2.2.2.1, hn.2.2.2.2.2.1, hn.2.2.2.2.2.2.1, hn.2.2.2.2.2.2.2.1, hn.2.2.2.2.2.2.2.2.1]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPointers
