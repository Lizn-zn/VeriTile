import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealFoundation
import bench.tritonbench_g.bgmv_expand_slice.Proofs.RealSource
import VeriTile.Triton.Semantics.ProgramRewrite

/-! Real semantics of an output-loop block, including old-output accumulation.
The reduction is an independent finite sum. Input lanes are arbitrary here:
this supports both masked and unmasked input loading, whose correspondence to
the initial input vector is a separate setup obligation. No FP rule is used. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealBlock
open VeriTile.Triton
open RealFoundation
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- The actual reduction over the loaded input/weight tiles. -/
def reduction (BN BK : Nat) : Op .real [BN] :=
  .reduceSum ⟨1, by simp⟩ Bool.false
    (.mul .real (.leadL (.consSame .nil)) (.ref .real [BK] "tiled_a")
      (.ref .real [BN,BK] "tiled_b"))

def outputAddress (stride BN : Nat) : Op .ptr [BN] :=
  .ptrAdd .scalarL (.ref .ptr [] "c_ptr")
    (.mul .nat .scalarR (.ref .nat [BN] "current_n") (.constNat stride))

def oldOutputLoad (stride BN : Nat) : Op .real [BN] :=
  .load .real (.ptr (outputAddress stride BN)) (.mask (.ref .bool [BN] "c_mask"))

def accumulated (BN BK : Nat) (swapped : Bool) : Op .real [BN] :=
  if swapped then .add .real (.consSame .nil) (.ref .real [BN] "tiled_out") (reduction BN BK)
  else .add .real (.consSame .nil) (reduction BN BK) (.ref .real [BN] "tiled_out")

def update (stride BN BK : Nat) (add swapped : Bool) : Stmt :=
  .ifThenElse (.constBool add)
    [.assign .real [BN] "tiled_out" (oldOutputLoad stride BN),
     .assign .real [BN] "accumulator" (accumulated BN BK swapped)]
    [.assign .real [BN] "accumulator" (reduction BN BK)]

def outputStore (stride BN : Nat) : Stmt :=
  .store .real [BN] (.ptr (outputAddress stride BN))
    (.ref .real [BN] "accumulator") (.mask (.ref .bool [BN] "c_mask"))

def tail (stride BN BK : Nat) (add swapped : Bool) : List Stmt :=
  [update stride BN BK add swapped, outputStore stride BN]

/-- Independent masked dot product. `w` denotes the original weight memory;
`a` is the input tile loaded before the output loop. -/
def dot (a : Fin BK → ℝ) (w : Nat → ℝ) (base rowStride colStride K g : Nat) : ℝ :=
  ∑ k : Fin BK, a k * (if k.val < K then w (base + g*rowStride + k.val*colStride) else 0)

/-- Both EVEN_K input-load choices give the same masked rank reduction.
Unmasked extra input lanes may be nonzero; the weight mask removes them. -/
theorem dot_input_mask (a : Fin BK → ℝ) (w : Nat → ℝ)
    (base rowStride colStride K g : Nat) (EVEN_K : Bool) :
    dot (fun k => if EVEN_K then a k else if k.val < K then a k else 0)
      w base rowStride colStride K g = dot a w base rowStride colStride K g := by
  unfold dot
  apply Finset.sum_congr rfl
  intro k _
  cases EVEN_K <;> by_cases hk : k.val < K <;> simp [hk]

/-- The actual loaded weight tile, including both masks. -/
def weights (w : Nat → ℝ) (base rowStride colStride K snl n BN BK : Nat) : Tile .real [BN,BK] :=
  ⟨fun i => some (if n + i.1.val < snl ∧ i.2.1.val < K then
    w (base + (n+i.1.val)*rowStride + i.2.1.val*colStride) else 0)⟩

def sums (a : Fin BK → ℝ) (w : Nat → ℝ) (base rowStride colStride K snl n BN : Nat) : Fin BN → ℝ :=
  fun i => if n+i.val < snl then dot a w base rowStride colStride K (n+i.val) else 0

theorem reduction_eval (s : BlockState) (a : Fin BK → ℝ) (w : Nat → ℝ)
    (base rowStride colStride K snl n BN : Nat)
    (ha : s.regs .real [BK] "tiled_a" = some ⟨fun i => some (a i.1)⟩)
    (hb : s.regs .real [BN,BK] "tiled_b" = some (weights w base rowStride colStride K snl n BN BK)) :
    evalOp (reduction BN BK) s = some ⟨fun i => some (sums a w base rowStride colStride K snl n BN i.1)⟩ := by
  unfold reduction
  erw [evalOp_reduceSum]
  rw [evalOp_mul, evalOp_ref, ha, evalOp_ref, hb]
  apply congrArg some
  ext i
  simp only [Tile.reduceSum_false, Tile.reduceSumDrop_data, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, Tile.bop_data,
    Broadcast.leftIndex, Broadcast.rightIndex, weights]
  simp only [NumericDType.mul, WithBot.realMul, Option.map₂, Option.bind, Option.map]
  erw [withBot_sum_some]
  apply congrArg some
  by_cases hn : n+i.1.val < snl <;> simp [sums, dot, hn]

/-- Masked lanes retain the source's unspecified-load convention. Only active
lanes are used by the store; no zero-undef premise is needed in this lemma. -/
def oldValues (s : BlockState) (out : RegionName) (base stride snl n BN : Nat) : Fin BN → ℝ :=
  fun i => if n+i.val < snl then s.readMem out (base+(n+i.val)*stride)
    else s.undef out (base+(n+i.val)*stride)

theorem old_output_eval (s : BlockState) (out : RegionName)
    (base stride snl BN c : Nat)
    (hp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out,base)))
    (hn : s.regs .nat [BN] "current_n" = some (Tile.vec (fun i => c*BN+i.val)))
    (hm : s.regs .bool [BN] "c_mask" = some (Tile.vec (fun i => decide (c*BN+i.val < snl)))) :
    evalOp (oldOutputLoad stride BN) s =
      some ⟨fun i => some (oldValues s out base stride snl (c*BN) BN i.1)⟩ := by
  simp only [oldOutputLoad, outputAddress, evalOp,
    cptrs_eval out base stride BN c s hp hn, hm]
  apply congrArg some
  ext i
  by_cases h : c*BN+i.1.val < snl <;> simp [Tile.vec, oldValues, BlockState.readMemValue_real, h]

theorem accumulated_eval (s : BlockState) (sum old : Fin BN → ℝ) (BK : Nat) (swapped : Bool)
    (hs : evalOp (reduction BN BK) s = some ⟨fun i => some (sum i.1)⟩)
    (ho : s.regs .real [BN] "tiled_out" = some ⟨fun i => some (old i.1)⟩) :
    evalOp (accumulated BN BK swapped) s = some ⟨fun i => some (sum i.1 + old i.1)⟩ := by
  cases swapped
  all_goals
    simp only [accumulated, Bool.false_eq_true, ↓reduceIte, evalOp_add, evalOp_ref, hs, ho]
    apply congrArg some
    ext i
    simp [Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.add, WithBot.realAdd, add_comm]

private theorem reduction_set_old (s : BlockState) (old : Fin BN → ℝ) (BK : Nat) :
    evalOp (reduction BN BK) (s.setReg "tiled_out" .real [BN] ⟨fun i => some (old i.1)⟩) =
    evalOp (reduction BN BK) s := by
  unfold reduction
  erw [evalOp_reduceSum, evalOp_reduceSum]
  simp

private theorem eval_bool (b : Bool) (s : BlockState) :
    evalOp (.constBool b) s = some (Tile.scalar b) := by simp [evalOp]

/-- Both addition orders have this same real result. -/
def results (sum old : Fin BN → ℝ) (add : Bool) : Fin BN → ℝ :=
  fun i => if add then sum i + old i else sum i

def afterUpdate (s : BlockState) (sum old : Fin BN → ℝ) (add : Bool) : BlockState :=
  let loaded := if add then s.setReg "tiled_out" .real [BN] ⟨fun i => some (old i.1)⟩ else s
  loaded.setReg "accumulator" .real [BN] ⟨fun i => some (results sum old add i.1)⟩

theorem update_exec (s : BlockState) (sum : Fin BN → ℝ) (out : RegionName)
    (base stride snl c BK : Nat) (add swapped : Bool)
    (hp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out,base)))
    (hn : s.regs .nat [BN] "current_n" = some (Tile.vec (fun i => c*BN+i.val)))
    (hm : s.regs .bool [BN] "c_mask" = some (Tile.vec (fun i => decide (c*BN+i.val < snl))))
    (hs : evalOp (reduction BN BK) s = some ⟨fun i => some (sum i.1)⟩) :
    stepStmt (update stride BN BK add swapped) s =
      some (afterUpdate s sum (oldValues s out base stride snl (c*BN) BN) add) := by
  cases add
  · simp [update, stepStmt, stepStmts, eval_bool, Tile.scalar_data,
      hs, afterUpdate, results]
  · have hs' : evalOp (reduction BN BK)
        (s.setReg "tiled_out" .real [BN]
          ⟨fun i => some (oldValues s out base stride snl (c*BN) BN i.1)⟩) =
        some ⟨fun i => some (sum i.1)⟩ := by
      rw [reduction_set_old, hs]
    simp only [update, stepStmt, eval_bool, Tile.scalar_data,
      bind, Option.bind, ↓reduceIte, stepStmts, old_output_eval s out base stride snl BN c hp hn]
    erw [old_output_eval s out base stride snl BN c hp hn]
    simp only [Option.bind]
    erw [accumulated_eval _ sum (oldValues s out base stride snl (c*BN) BN) BK swapped hs' (by simp)]
    rfl
    all_goals assumption

def scatter (s : BlockState) (out : RegionName) (base stride snl n : Nat)
    (vals : Fin BN → ℝ) : BlockState :=
  (TileShape.allIndices [BN]).foldl (fun st i =>
    if decide (n+i.1.val < snl) then st.writeMem out (base+(n+i.1.val)*stride) (vals i.1)
    else st) s

def afterTail (s : BlockState) (sum : Fin BN → ℝ) (out : RegionName)
    (base stride snl n : Nat) (add : Bool) : BlockState :=
  let old := oldValues s out base stride snl n BN
  scatter (afterUpdate s sum old add) out base stride snl n (results sum old add)

/-- Complete update-and-store suffix, for symbolic tiles and arbitrary old
output memory, in either source version and both accumulation branches. -/
theorem tail_exec (s : BlockState) (sum : Fin BN → ℝ) (out : RegionName)
    (base stride snl c BK : Nat) (add swapped : Bool)
    (hp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out,base)))
    (hn : s.regs .nat [BN] "current_n" = some (Tile.vec (fun i => c*BN+i.val)))
    (hm : s.regs .bool [BN] "c_mask" = some (Tile.vec (fun i => decide (c*BN+i.val < snl))))
    (hs : evalOp (reduction BN BK) s = some ⟨fun i => some (sum i.1)⟩) :
    stepStmts (tail stride BN BK add swapped) s = some (afterTail s sum out base stride snl (c*BN) add) := by
  rw [tail, stepStmts.cons_some (update_exec s sum out base stride snl c BK add swapped hp hn hm hs)]
  simp only [outputStore, outputAddress]
  rw [stepStmts.cons_some (store_step out base stride BN c
    (afterUpdate s sum (oldValues s out base stride snl (c*BN) BN) add)
    (results sum (oldValues s out base stride snl (c*BN) BN) add)
    (fun i => decide (c*BN+i.val < snl))
    (by cases add <;> simpa [afterUpdate] using hp)
    (by cases add <;> simpa [afterUpdate] using hn)
    (by cases add <;> simp [afterUpdate])
    (by cases add <;> simpa [afterUpdate] using hm))]
  rw [stepStmts.nil]
  rfl

theorem scatter_frame (s : BlockState) (out r : RegionName) (base stride snl n o : Nat)
    (vals : Fin BN → ℝ)
    (h : ∀ i : Fin BN, n+i.val < snl → ¬(out = r ∧ base+(n+i.val)*stride = o)) :
    (scatter s out base stride snl n vals).mem r o = s.mem r o := by
  unfold scatter
  simp only [decide_eq_true_eq]
  exact foldl_store_preserve_cell _ _ _ r o _ s (fun i _ hi => h i.1 hi)

theorem scatter_active (s : BlockState) (out : RegionName) (base stride snl n : Nat)
    (vals : Fin BN → ℝ) (hstride : 0 < stride) (i : Fin BN) (hi : n+i.val < snl) :
    (scatter s out base stride snl n vals).readMem out (base+(n+i.val)*stride) = vals i := by
  unfold scatter
  refine foldl_store_at
    (fun j : TileIndex [BN] => base+(n+j.1.val)*stride)
    (fun j : TileIndex [BN] => vals j.1)
    (fun j : TileIndex [BN] => decide (n+j.1.val < snl))
    (base+(n+i.val)*stride) (TileShape.allIndices [BN]) s (i,PUnit.unit)
    (TileShape.mem_allIndices _ _) (by simpa) rfl ?_ (TileShape.allIndices_nodup _)
  intro b _ _ heq
  have hv : b.1.val = i.val := by
    change base+(n+b.1.val)*stride = base+(n+i.val)*stride at heq
    exact Nat.add_left_cancel (Nat.eq_of_mul_eq_mul_right hstride (Nat.add_left_cancel heq))
  apply Prod.ext
  · exact Fin.ext hv
  · rfl

/-- Active lanes contain the reduction result plus the output value from
before this block, or just the reduction when accumulation is disabled. -/
theorem tail_readback (s : BlockState) (sum : Fin BN → ℝ) (out : RegionName)
    (base stride snl n : Nat) (add : Bool) (hstride : 0 < stride)
    (i : Fin BN) (hi : n+i.val < snl) :
    (afterTail s sum out base stride snl n add).readMem out (base+(n+i.val)*stride) =
      if add then sum i + s.readMem out (base+(n+i.val)*stride) else sum i := by
  rw [afterTail, scatter_active _ _ _ _ _ _ _ hstride i hi]
  simp [results, oldValues, hi]

/-- The full-cell frame preserves all memory outside the actual active stores,
including typed metadata and unread output lanes. -/
theorem tail_frame (s : BlockState) (sum : Fin BN → ℝ) (out r : RegionName)
    (base stride snl n o : Nat) (add : Bool)
    (h : ∀ i : Fin BN, n+i.val < snl → ¬(out = r ∧ base+(n+i.val)*stride = o)) :
    (afterTail s sum out base stride snl n add).mem r o = s.mem r o := by
  rw [afterTail, scatter_frame _ _ _ _ _ _ _ _ _ h]
  cases add <;> simp [afterUpdate]

/-- The block's update/store suffix preserves every other register. This is
needed to retain the input tile and source pointer/split registers in the loop. -/
theorem tail_register (s : BlockState) (sum : Fin BN → ℝ) (out : RegionName)
    (base stride snl n : Nat) (add : Bool) (dtype : TileDType) (shape : TileShape)
    (name : RegName) (h1 : name ≠ "tiled_out") (h2 : name ≠ "accumulator") :
    (afterTail s sum out base stride snl n add).regs dtype shape name = s.regs dtype shape name := by
  unfold afterTail scatter
  rw [foldl_store_regs]
  cases add <;> simp [afterUpdate, BlockState.setReg, h1, h2]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealBlock
