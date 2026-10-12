import bench.tritonbench_g.attention_llama.Proofs.FP8RealSource
import bench.tritonbench_g.attention_llama.Proofs.FP8RealMemory

/-! One complete FP8 attention iteration, related to the same real recurrence
on decoded input memory. The relation preserves all state channels and failure.
Only the actual masked K/V lanes need finite signed-byte encodings. -/
namespace VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealIteration
open VeriTile.Triton RealKernel FP8RealMemory
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

private theorem decode_cons (K V : RegionName) (s : BlockState) (name : RegName)
    (ptr : Op .ptr sh) (mask : Op .bool sh) (ptrs : Tile .ptr sh) (masks : Tile .bool sh)
    (lhs rhs : List Stmt)
    (hp : evalOp ptr s = some ptrs) (hm : evalOp mask s = some masks)
    (hpure : ptr.SafeAt (fun _ => 0) s) (mpure : mask.SafeAt (fun _ => 0) s)
    (hregions : ∀ i, masks.data i = Bool.true → (ptrs.data i).1 = K ∨ (ptrs.data i).1 = V)
    (hfinite : ∀ i, masks.data i = Bool.true →
      FiniteByte (s.readMemValue .int (ptrs.data i).1 (ptrs.data i).2))
    (htail : ∀ tile : Tile .real sh,
      (stepStmts lhs (s.setReg name .real sh tile)).map (decoded K V) =
        stepStmts rhs (decoded K V (s.setReg name .real sh tile))) :
    (stepStmts (.assign .real sh name (.decodeFloat8E5 .real
      (.load .int (.ptr ptr) (.maskOther mask (.broadcast (.constInt 0) sh)))) :: lhs) s).map
        (decoded K V) =
      stepStmts (.assign .real sh name
        (.load .real (.ptr ptr) (.maskOther mask (.broadcast (.const 0) sh))) :: rhs)
        (decoded K V s) := by
  have he := masked_decode K V s ptr mask ptrs masks hp hm hpure mpure hregions hfinite
  simp only [stepStmts, stepStmt]
  rw [← he]
  cases hv : @evalOp .real sh (.decodeFloat8E5 .real
      (.load .int (.ptr ptr) (.maskOther mask (.broadcast (.constInt 0) sh)))) s with
  | none => simp [hv]
  | some tile => simpa [hv, decoded_setReg] using htail tile

/-- The source mask, shared by the K and V loads, tests the global key index. -/
def keyMask (N BN BD : Nat) : Op .bool [BN, BD] :=
  .remap [BN, BD] Broadcast.nil.consL.consSame.leftIndex
    (.lt .nat Broadcast.scalarR
      (.expandDim ⟨1, by simp⟩ (.ref .nat [BN] "block_n_offs")) (.constNat N))

private theorem load_cons (K V R : RegionName) (s : BlockState)
    (name ptrName : RegName) (base srow scol SN N BN BD : Nat) (lhs rhs : List Stmt)
    (hregion : R = K ∨ R = V)
    (hptr : s.regs .ptr [BN, BD] ptrName = some (alPtrTile R base srow scol SN BN BD))
    (hrows : s.regs .nat [BN] "block_n_offs" =
      some (Tile.vec (fun j : Fin BN => SN + j.val)))
    (hbytes : ∀ (j : Fin BN) (d : Fin BD), SN + j.val < N →
      FiniteByte (s.readMemValue .int R (base + (SN + j.val) * srow + d.val * scol)))
    (htail : ∀ tile : Tile .real [BN, BD],
      (stepStmts lhs (s.setReg name .real [BN, BD] tile)).map (decoded K V) =
        stepStmts rhs (decoded K V (s.setReg name .real [BN, BD] tile))) :
    (stepStmts (.assign .real [BN, BD] name (.decodeFloat8E5 .real
      (.load .int (.ptr (.ref .ptr [BN, BD] ptrName))
        (.maskOther (keyMask N BN BD) (.broadcast (.constInt 0) [BN, BD])))) :: lhs) s).map
        (decoded K V) =
      stepStmts (.assign .real [BN, BD] name
        (.load .real (.ptr (.ref .ptr [BN, BD] ptrName))
          (.maskOther (keyMask N BN BD) (.broadcast (.const 0) [BN, BD]))) :: rhs)
        (decoded K V s) := by
  apply decode_cons K V s name _ _ (alPtrTile R base srow scol SN BN BD)
    (⟨fun i => decide (SN + i.1.val < N)⟩ : Tile .bool [BN, BD]) lhs rhs
  · simpa only [evalOp_ref] using hptr
  · exact al_colmask_eval s SN N hrows
  · simp [Op.SafeAt.eq_def]
  · simp [keyMask, Op.SafeAt.eq_def]
  · intro i _; exact hregion
  · intro i hi
    exact hbytes i.1 i.2.1 (by simpa using hi)
  · exact htail

/-- The unchanged real recurrence, including the constant causal branch. -/
def referenceLoop (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal candidate : Bool) : List Stmt :=
  alLoopPrefixG scale N BM BN BD ++
    (if causal then [alCausalWhereStmt SP BM BN] else []) ++
      alLoopTailG skn svk N BM BN BD candidate

/-- Inline only the source's constant causal branch; the byte loads remain. -/
def inlineLoop (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal candidate : Bool) : List Stmt :=
  (alLoopPrefixG scale N BM BN BD).flatMap (FP8RealSource.lowerLoad Bool.true) ++
    (if causal then [alCausalWhereStmt SP BM BN] else []) ++
      (alLoopTailG skn svk N BM BN BD candidate).flatMap (FP8RealSource.lowerLoad Bool.true)

theorem source_inline (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal candidate : Bool) (s : BlockState) :
    stepStmts (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s =
      stepStmts (inlineLoop scale skn svk N SP BM BN BD causal candidate) s := by
  simp only [FP8RealSource.loopBody, inlineLoop, ProgramRewrite.append, ProgramRewrite.constant_branch]
  cases stepStmts ((alLoopPrefixG scale N BM BN BD).flatMap
      (FP8RealSource.lowerLoad Bool.true)) s <;> rfl

/-- A complete source iteration simulates the same real recurrence on decoded
K/V inputs. The real arithmetic registers may contain arbitrary values; this
preserves evaluation failure as well as success. The finite-byte hypotheses
cover precisely the key lanes selected by the actual source mask. -/
theorem iteration (K V : RegionName) (s : BlockState) (scale : ℝ)
    (kBase vBase skn skk svk svn SN N SP BM BN BD : Nat) (causal candidate : Bool)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar SN))
    (hoffn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun j : Fin BN => j.val)))
    (hkp : s.regs .ptr [BN, BD] "k_ptrs" = some (alPtrTile K kBase skn skk SN BN BD))
    (hvp : s.regs .ptr [BN, BD] "v_ptrs" = some (alPtrTile V vBase svk svn SN BN BD))
    (hkbytes : ∀ (j : Fin BN) (d : Fin BD), SN + j.val < N →
      FiniteByte (s.readMemValue .int K (kBase + (SN + j.val) * skn + d.val * skk)))
    (hvbytes : ∀ (j : Fin BN) (d : Fin BD), SN + j.val < N →
      FiniteByte (s.readMemValue .int V (vBase + (SN + j.val) * svk + d.val * svn))) :
    (stepStmts (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s).map
      (decoded K V) = stepStmts (referenceLoop scale skn svk N SP BM BN BD causal candidate)
        (decoded K V s) := by
  have hoff : evalOp (.add .nat Broadcast.scalarL (.ref .nat [] "start_n")
      (.ref .nat [BN] "offs_n")) s = some (Tile.vec (fun j : Fin BN => SN + j.val)) := by
    rw [evalOp_add]
    simp only [evalOp_ref, hsn, hoffn, Option.bind_eq_bind, Option.bind_some]
    apply congrArg some
    ext i
    rfl
  rw [source_inline]
  cases causal <;> cases candidate
  all_goals
    simp only [inlineLoop, referenceLoop, alLoopPrefixG, alLoopTailG,
      FP8RealSource.lowerLoad, FP8RealSource.byteLoad?,
      List.flatMap_cons, List.flatMap_nil, List.cons_append, List.nil_append,
      Bool.false_eq_true, ↓reduceIte, Bool.or_eq_true, beq_iff_eq, String.reduceEq,
      false_or, or_false, show (0.0 : ℝ) = 0 from by norm_num]
    apply evaluated_cons (tile := Tile.vec (fun j : Fin BN => SN + j.val))
      (hpure := by simp [Op.SafeAt.eq_def]) (he := hoff)
    apply load_cons (R := K) (base := kBase) (srow := skn) (scol := skk) (SN := SN)
      (hregion := Or.inl rfl)
    · simpa using hkp
    · simp
    · simpa only [BlockState.setReg_readMemValue] using hkbytes
    · intro kt
      repeat' (apply pure_cons (hpure := by simp [Op.SafeAt.eq_def, alCausalWhereStmt]); intro)
      apply load_cons (R := V) (base := vBase) (srow := svk) (scol := svn) (SN := SN)
        (hregion := Or.inr rfl)
      · simpa using hvp
      · simp
      · simpa only [BlockState.setReg_readMemValue] using hvbytes
      · intro vt
        repeat' (apply pure_cons (hpure := by simp [Op.SafeAt.eq_def]); intro)
        simp only [stepStmts.nil, Option.map_some]

/-- The entire source iteration writes registers only. Preserve the raw byte
cells themselves, not merely their possibly non-injective decoded values. -/
theorem memory_frame (scale : ℝ) (skn svk N SP BM BN BD : Nat)
    (causal candidate : Bool) (s t : BlockState)
    (hrun : stepStmts (FP8RealSource.loopBody Bool.true scale skn svk N SP BM BN BD causal candidate) s = some t) :
    t.mem = s.mem := by
  apply storeFree_stepStmts_mem _ s t _ hrun
  cases candidate <;>
    simp [FP8RealSource.loopBody, alLoopPrefixG, alLoopTailG, alCausalWhereStmt,
      FP8RealSource.lowerLoad, FP8RealSource.byteLoad?, storeFree]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionLlama.FP8RealIteration
