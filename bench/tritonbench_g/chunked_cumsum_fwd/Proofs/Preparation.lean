import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Loads

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Preparation
open VeriTile.Triton Math Program Initialization Tiles Loads
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

structure Ready (p : Layout) (l : Launch) (r : Regions) (u : BlockState) : Prop where
  head : u.regs .nat [p.BH] "offs_h" = some (heads p l)
  column : u.regs .nat [p.BC] "offs_c" = some (columns p)
  limit : u.regs .nat [] "chunk_size_limit" = some (Tile.scalar (min p.chunk (p.sequence-l.chunk*p.chunk)))
  dt : u.regs .ptr [p.BH,p.BC] "dt_ptrs" = some (dtPointers p l r)
  rate : u.regs .ptr [p.BH] "A_ptrs" = some (ratePointers p l r)
  out : u.regs .ptr [p.BH,p.BC] "dt_out_ptrs" = some (outPointers p l r)
  sum : u.regs .ptr [p.BH,p.BC] "dA_cs_ptrs" = some (sumPointers p l r)

theorem Ready.initial (p : Layout) (r : Regions) (u : BlockState) :
    Ready p (launch u) r (initial p r u) := by
  constructor <;> simp [Initialization.initial]

theorem Ready.setDt (h : Ready p l r u) (f : Fin p.BH → Fin p.BC → ℝ) :
    Ready p l r (u.setReg "dt" .real [p.BH,p.BC] (matrix p f)) := by
  rcases h with ⟨hh,hc,hl,hd,ha,ho,hs⟩
  constructor <;> simp_all

theorem Ready.setBias (h : Ready p l r u) (b : Fin p.BH → ℝ) :
    Ready p l r (u.setReg "dt_bias" .real [p.BH] (row p b)) := by
  rcases h with ⟨hh,hc,hl,hd,ha,ho,hs⟩
  constructor <;> simp_all

def biased (p : Layout) (l : Launch) (a : Inputs) (hasBias : Bool) (i : Fin p.BH) (j : Fin p.BC) : ℝ :=
  inputValues p l a i j + if hasBias then biasValues p l a i else 0
def softened (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool) (i : Fin p.BH) (j : Fin p.BC) : ℝ :=
  if useSoftplus then softplus (biased p l a hasBias i j) else biased p l a hasBias i j

def biasState (p : Layout) (l : Launch) (a : Inputs) (hasBias : Bool) (u : BlockState) : BlockState :=
  if hasBias then
    (u.setReg "dt_bias" .real [p.BH] (row p (biasValues p l a))).setReg
      "dt" .real [p.BH,p.BC] (matrix p (biased p l a hasBias))
  else u

def softState (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool) (u : BlockState) : BlockState :=
  if useSoftplus then u.setReg "dt" .real [p.BH,p.BC] (matrix p (softened p l a hasBias useSoftplus)) else u

def result (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (u : BlockState) : BlockState :=
  let loaded := u.setReg "dt" .real [p.BH,p.BC] (matrix p (inputValues p l a))
  let biased := biasState p l a hasBias loaded
  let softened := softState p l a hasBias useSoftplus biased
  let clamped := softened.setReg "dt" .real [p.BH,p.BC]
    (matrix p (fun i j => upper.cap (max (Preparation.softened p l a hasBias useSoftplus i j) lower)))
  clamped.setReg "dt" .real [p.BH,p.BC]
    (matrix p (fun i j => prepared p l a hasBias useSoftplus lower upper i.val j.val))

theorem remask_formula (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (i : Fin p.BH) (j : Fin p.BC) :
    (if activeInput p l i.val j.val then upper.cap (max (softened p l a hasBias useSoftplus i j) lower) else 0) =
      prepared p l a hasBias useSoftplus lower upper i.val j.val := by
  by_cases h : activeInput p l i.val j.val
  · have hh := h.1
    simp [prepared,softened,biased,inputValues,biasValues,h,hh]
  · simp [prepared,h]

theorem remask_eval (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (u : BlockState)
    (hm : evalOp (inputMask p) u = some (Tile.mat (fun i j => decide (activeInput p l i.val j.val))))
    (hd : u.regs .real [p.BH,p.BC] "dt" = some
      (matrix p (fun i j => upper.cap (max (softened p l a hasBias useSoftplus i j) lower)))) :
    evalOp (.where (inputMask p) (dtOp p) (zero p)) u =
      some (matrix p (fun i j => prepared p l a hasBias useSoftplus lower upper i.val j.val)) := by
  simp only [evalOp_where,hm,dtOp,evalOp_ref,hd,zero,evalOp.eq_def,Option.bind_eq_bind,Option.bind_some]
  apply congrArg some
  apply Tile.ext
  rintro ⟨i,j,⟨⟩⟩
  simp only [Tile.select,Tile.mat_data,matrix,decide_eq_true_eq]
  split_ifs with h
  · exact congrArg some (by simpa only [if_pos h] using remask_formula p l a hasBias useSoftplus lower upper i j)
  · exact congrArg some (by simpa only [if_neg h] using remask_formula p l a hasBias useSoftplus lower upper i j)

theorem bias_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (hasBias : Bool) (u : BlockState)
    (hh : u.regs .nat [p.BH] "offs_h" = some (heads p l))
    (ha : ∀ off, u.readMem r.bias off = a.bias off)
    (hd : u.regs .real [p.BH,p.BC] "dt" = some (matrix p (inputValues p l a))) :
    stepStmt (.ifThen (.constBool hasBias) [.assign .real [p.BH] "dt_bias" (loadBias p r),
      .assign .real [p.BH,p.BC] "dt" (addBias p)]) u = some (biasState p l a hasBias u) := by
  cases hasBias
  · simp [stepStmt,evalOp.eq_def,biasState]
  · simp only [stepStmt,evalOp.eq_def,Option.bind_eq_bind,Option.bind_some,Tile.scalar_data,↓reduceIte]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (loadBias_eval p l r a u hh ha))]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (addBias_eval p _ (inputValues p l a) (biasValues p l a)
      (by simpa using hd) (by simp))),stepStmts.nil]
    simp [biasState,biased,matrix]

theorem bias_ready (h : Ready p l r u) (a : Inputs) (hasBias : Bool) :
    Ready p l r (biasState p l a hasBias u) := by
  cases hasBias
  · exact h
  · exact (h.setBias _).setDt _

theorem bias_register (p : Layout) (l : Launch) (a : Inputs) (hasBias : Bool) (u : BlockState)
    (hd : u.regs .real [p.BH,p.BC] "dt" = some (matrix p (inputValues p l a))) :
    (biasState p l a hasBias u).regs .real [p.BH,p.BC] "dt" = some (matrix p (biased p l a hasBias)) := by
  cases hasBias <;> simp [biasState,biased,hd,matrix]

theorem soft_executes (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool) (u : BlockState)
    (hd : u.regs .real [p.BH,p.BC] "dt" = some (matrix p (biased p l a hasBias))) :
    stepStmt (.ifThen (.constBool useSoftplus) [.assign .real [p.BH,p.BC] "dt" (softplusOp p)]) u =
      some (softState p l a hasBias useSoftplus u) := by
  cases useSoftplus
  · simp [stepStmt,evalOp.eq_def,softState]
  · simp only [stepStmt,evalOp.eq_def,Option.bind_eq_bind,Option.bind_some,Tile.scalar_data,↓reduceIte]
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (softplus_eval p u (biased p l a hasBias) hd)),stepStmts.nil]
    simp [softState,softened,matrix]

theorem soft_ready (h : Ready p l r u) (a : Inputs) (hasBias useSoftplus : Bool) :
    Ready p l r (softState p l a hasBias useSoftplus u) := by
  cases useSoftplus
  · exact h
  · exact h.setDt _

theorem soft_register (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool) (u : BlockState)
    (hd : u.regs .real [p.BH,p.BC] "dt" = some (matrix p (biased p l a hasBias))) :
    (softState p l a hasBias useSoftplus u).regs .real [p.BH,p.BC] "dt" =
      some (matrix p (softened p l a hasBias useSoftplus)) := by
  cases useSoftplus <;> simp [softState,softened,hd,matrix]

theorem executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (u : BlockState) (h : Ready p l r u)
    (hd : ∀ off, u.readMem r.dt off = a.dt off)
    (hb : ∀ off, u.readMem r.bias off = a.bias off) :
    stepStmts (preparation p r hasBias useSoftplus lower upper) u =
      some (result p l a hasBias useSoftplus lower upper u) := by
  let d := u.setReg "dt" .real [p.BH,p.BC] (matrix p (inputValues p l a))
  let b := biasState p l a hasBias d
  let s := softState p l a hasBias useSoftplus b
  let c := s.setReg "dt" .real [p.BH,p.BC]
    (matrix p (fun i j => upper.cap (max (softened p l a hasBias useSoftplus i j) lower)))
  have hdReady : Ready p l r d := h.setDt _
  have hbReady : Ready p l r b := bias_ready hdReady a hasBias
  have hsReady : Ready p l r s := soft_ready hbReady a hasBias useSoftplus
  have hcReady : Ready p l r c := hsReady.setDt _
  have hbReg := bias_register p l a hasBias d (by simp [d])
  have hsReg := soft_register p l a hasBias useSoftplus b hbReg
  unfold preparation
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (loadDt_eval p l r a u h.dt
    (inputMask_eval p l u h.head h.column h.limit) hd))]
  change stepStmts _ d = _
  rw [stepStmts.cons_some (bias_executes p l r a hasBias d hdReady.head
    (by simpa [d] using hb) (by simp [d]))]
  change stepStmts _ b = _
  rw [stepStmts.cons_some (soft_executes p l a hasBias useSoftplus b hbReg)]
  change stepStmts _ s = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (clamp_eval p s _ lower upper hsReg))]
  change stepStmts _ c = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (remask_eval p l a hasBias useSoftplus lower upper c
    (inputMask_eval p l c hcReady.head hcReady.column hcReady.limit) (by simp [c]))),stepStmts.nil]
  rfl

theorem result_ready (h : Ready p l r u) (a : Inputs) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) :
    Ready p l r (result p l a hasBias useSoftplus lower upper u) :=
  ((soft_ready (bias_ready (h.setDt _) a hasBias) a hasBias useSoftplus).setDt _).setDt _

theorem result_mem (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (u : BlockState) :
    (result p l a hasBias useSoftplus lower upper u).mem = u.mem := by
  cases hasBias <;> cases useSoftplus <;> rfl

theorem result_register (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (u : BlockState) :
    (result p l a hasBias useSoftplus lower upper u).regs .real [p.BH,p.BC] "dt" =
      some (matrix p (fun i j => prepared p l a hasBias useSoftplus lower upper i.val j.val)) := by
  simp [result]

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Preparation
