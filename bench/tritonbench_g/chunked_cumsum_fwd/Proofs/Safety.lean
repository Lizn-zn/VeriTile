import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.AccessSafety
namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Safety
open VeriTile.Triton Math Program Initialization Tiles Loads Preparation Execution
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

theorem bias_branch_safe (p : Layout) (l : Launch) (r : Regions) (u : BlockState)
    (hasBias : Bool) (bounds : RegionBounds)
    (hh : u.regs .nat [p.BH] "offs_h" = some (heads p l))
    (hb : hasBias = Bool.true → ∀ i : Fin p.BH, head p l i.val < p.heads → head p l i.val*p.biasHead < bounds r.bias) :
    Stmt.TraceSafe bounds (.ifThen (.constBool hasBias)
      [.assign .real [p.BH] "dt_bias" (loadBias p r),.assign .real [p.BH,p.BC] "dt" (addBias p)]) u := by
  cases hasBias
  · simp [Stmt.TraceSafe,Op.SafeAt.eq_def,evalOp.eq_def]
  · simp only [Stmt.TraceSafe,Op.SafeAt.eq_def,evalOp.eq_def,Tile.scalar_data,true_and,↓reduceIte]
    apply TraceProofs.assign_cons (bias_safe p l r u bounds hh (hb rfl))
    intro v
    apply TraceProofs.all_safe
    simp [Stmt.TraceSafe,addBias,dtOp,Op.SafeAt.eq_def]

theorem preparation_safe (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (u : BlockState)
    (h : Ready p l r u) (hd : ∀ off, u.readMem r.dt off = a.dt off)
    (hbias : ∀ off, u.readMem r.bias off = a.bias off)
    (bounds : RegionBounds) (hb : Bounds p l r hasBias bounds) :
    Stmt.TraceSafeList bounds (preparation p r hasBias useSoftplus lower upper) u := by
  let d := u.setReg "dt" .real [p.BH,p.BC] (matrix p (inputValues p l a))
  have hdr : Ready p l r d := h.setDt _
  unfold preparation
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (loadDt_eval p l r a u h.dt
    (inputMask_eval p l u h.head h.column h.limit) hd))]
  refine ⟨by simpa only [Stmt.TraceSafe] using dt_safe p l r u bounds h hb.dt,?_⟩
  change Stmt.TraceSafeList bounds _ d
  rw [TraceProofs.cons_of_run_iff (bias_executes p l r a hasBias d hdr.head
    (by simpa [d] using hbias) (by simp [d]))]
  refine ⟨bias_branch_safe p l r d hasBias bounds hdr.head hb.bias,?_⟩
  apply TraceProofs.all_safe
  intro st hs t
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl
  · simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp.eq_def, Tile.scalar_data, true_and]
    split
    · apply TraceProofs.all_safe
      simp [Stmt.TraceSafe, softplusOp, dtOp, Op.SafeAt.eq_def]
    · trivial
  · cases upper <;> simp [Stmt.TraceSafe,clamp,lowerClamp,dtOp,Op.SafeAt.eq_def]
  · simp [Stmt.TraceSafe,inputMask,mask,headOp,columnOp,dtOp,zero,Op.SafeAt.eq_def]

theorem accumulation_safe (p : Layout) (l : Launch) (r : Regions) (u : BlockState)
    (swapped : Bool) (bounds : RegionBounds) (h : Ready p l r u)
    (hRate : ∀ i : Fin p.BH, head p l i.val < p.heads → head p l i.val*p.rateHead < bounds r.rate)
    (hSum : ∀ (i : Fin p.BH) (j : Fin p.BC), activeOutput p l i.val j.val → sumAddress p l i.val j.val < bounds r.sum) :
    Stmt.TraceSafeList bounds (accumulation p swapped) u := by
  unfold accumulation
  apply TraceProofs.assign_cons (rate_safe p l r u bounds h hRate)
  intro a
  apply TraceProofs.assign_cons
  · cases swapped <;> simp [product,dtOp,Op.SafeAt.eq_def]
  intro d
  apply TraceProofs.assign_cons
  · simp [Op.SafeAt.eq_def]
  intro c
  apply Stmt.TraceSafeList.cons_intro
  · exact store_safe p l r.sum (sumOffsets p l) "dA_cs_ptrs" "dA_cs" _ bounds
      (by simpa using h.sum) (by simpa using h.head) (by simpa using h.column) hSum
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

theorem tail_safe (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasBias useSoftplus swapped : Bool) (lower : ℝ) (upper : UpperBound) (u : BlockState)
    (h : Ready p l r u) (hd : ∀ off, u.readMem r.dt off = a.dt off)
    (hbias : ∀ off, u.readMem r.bias off = a.bias off)
    (bounds : RegionBounds) (hb : Bounds p l r hasBias bounds) :
    Stmt.TraceSafeList bounds (preparation p r hasBias useSoftplus lower upper ++ [storeDt p] ++ accumulation p swapped) u := by
  simp only [List.append_assoc]
  apply TraceProofs.append_safe
  · exact preparation_safe p l r a hasBias useSoftplus lower upper u h hd hbias bounds hb
  · intro v he
    rw [Preparation.executes p l r a hasBias useSoftplus lower upper u h hd hbias] at he
    cases he
    let pre := Preparation.result p l a hasBias useSoftplus lower upper u
    have hr : Ready p l r pre := Preparation.result_ready h a hasBias useSoftplus lower upper
    have hstore := Stores.store_eval p l r.out (outOffsets p l) (dtValues p l a hasBias useSoftplus lower upper)
      "dt_out_ptrs" "dt" pre hr.out (Preparation.result_register p l a hasBias useSoftplus lower upper u)
      (outputMask_eval p l pre hr.head hr.column)
    change Stmt.TraceSafeList bounds (storeDt p :: accumulation p swapped) pre
    rw [storeDt,dtOp,TraceProofs.cons_of_run_iff hstore]
    refine ⟨store_safe p l r.out (outOffsets p l) "dt_out_ptrs" "dt" pre bounds hr.out hr.head hr.column hb.out,?_⟩
    exact accumulation_safe p l r _ swapped bounds (Stores.ready hr _ _ _) hb.rate hb.sum

theorem source_safe (p : Layout) (r : Regions) (bias : BiasStorage) (swapped : Bool)
    (dtype out_dtype : FloatDType) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (u : BlockState) (bounds : RegionBounds) (hb : Bounds p (launch u) r hasBias bounds) :
    (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).eraseDType.toAlgKernel.TraceSafe bounds u := by
  rw [Kernel.TraceSafe,source_body]
  unfold body
  simp only [List.append_assoc]
  apply TraceProofs.append_safe
  · exact initialization_safe p r bounds u
  · intro v he
    rw [Initialization.executes p r u] at he
    cases he
    exact tail_safe p (launch u) r ⟨u.readMem r.dt,u.readMem r.rate,u.readMem r.bias⟩
      hasBias useSoftplus swapped lower upper (initial p r u) (Ready.initial p r u)
      (by intro off; rfl) (by intro off; rfl) bounds hb

theorem source_flatten_ok (p : Layout) (r : Regions) (bias : BiasStorage) (swapped : Bool)
    (dtype out_dtype : FloatDType) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :
    (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).eraseDType.toAlgKernel.FlattenOk := by
  rw [Kernel.FlattenOk,source_body]
  cases upper <;> cases swapped
  all_goals unfold body initialization preparation accumulation
  all_goals simp only [List.cons_append, List.nil_append]
  all_goals simp [storeDt,storeSum,offset,dtOffset,
    pointer,limit,remaining,headOp,columnOp,mask,inputMask,outputMask,dtOp,zero,
    loadDt,loadBias,headMask,addBias,softplusOp,clamp,lowerClamp,loadRate,product,
    StmtList.FlattenOk,Stmt.FlattenOk, Op.FlattenOk]
  all_goals simp [Op.FlattenOk.eq_def]

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Safety
