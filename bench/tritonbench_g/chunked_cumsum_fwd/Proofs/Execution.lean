import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.Stores

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Execution
open VeriTile.Triton Math Program Initialization Tiles Loads Preparation
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def dtValues (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) : Stores.Value p :=
  fun i j => prepared p l a hasBias useSoftplus lower upper i.val j.val
def sums (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) : Stores.Value p :=
  cumulative p l a hasBias useSoftplus lower upper
def outOffsets (p : Layout) (l : Launch) : Stores.Address p := fun i j => outAddress p l i.val j.val
def sumOffsets (p : Layout) (l : Launch) : Stores.Address p := fun i j => sumAddress p l i.val j.val

def computed (p : Layout) (l : Launch) (a : Inputs) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (u : BlockState) : BlockState :=
  let u := u.setReg "A" .real [p.BH] (row p (fun i => rate p l a i.val))
  let u := u.setReg "dA" .real [p.BH,p.BC]
    (matrix p (fun i j => dtValues p l a hasBias useSoftplus lower upper i j * rate p l a i.val))
  u.setReg "dA_cs" .real [p.BH,p.BC] (matrix p (sums p l a hasBias useSoftplus lower upper))

def result (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (u : BlockState) : BlockState :=
  let pre := Preparation.result p l a hasBias useSoftplus lower upper u
  let stored := Stores.scatter p l r.out (outOffsets p l) (dtValues p l a hasBias useSoftplus lower upper) pre
  Stores.scatter p l r.sum (sumOffsets p l) (sums p l a hasBias useSoftplus lower upper)
    (computed p l a hasBias useSoftplus lower upper stored)

def footprint (p : Layout) (l : Launch) (r : Regions) : WriteFootprint := fun cell =>
  Stores.footprint p l r.out (outOffsets p l) cell ∨ Stores.footprint p l r.sum (sumOffsets p l) cell

theorem accumulation_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasBias useSoftplus swapped : Bool) (lower : ℝ) (upper : UpperBound) (u : BlockState)
    (h : Ready p l r u) (ha : ∀ off, u.readMem r.rate off = a.rate off)
    (hd : u.regs .real [p.BH,p.BC] "dt" = some (matrix p (dtValues p l a hasBias useSoftplus lower upper))) :
    stepStmts (accumulation p swapped) u = some
      (Stores.scatter p l r.sum (sumOffsets p l) (sums p l a hasBias useSoftplus lower upper)
        (computed p l a hasBias useSoftplus lower upper u)) := by
  let v := u.setReg "A" .real [p.BH] (row p (fun i => rate p l a i.val))
  let w := v.setReg "dA" .real [p.BH,p.BC]
    (matrix p (fun i j => dtValues p l a hasBias useSoftplus lower upper i j * rate p l a i.val))
  let z := computed p l a hasBias useSoftplus lower upper u
  have hv : v.regs .real [p.BH,p.BC] "dt" = some (matrix p (dtValues p l a hasBias useSoftplus lower upper)) := by
    simpa [v] using hd
  have heproduct := product_eval p v swapped (dtValues p l a hasBias useSoftplus lower upper)
    (fun i => rate p l a i.val) hv (by simp [v])
  have hescan := Scan.eval_scan p w
    (fun i j => dtValues p l a hasBias useSoftplus lower upper i j * rate p l a i.val) (by simp [w])
  have heout := Stores.store_eval p l r.sum (sumOffsets p l) (sums p l a hasBias useSoftplus lower upper)
    "dA_cs_ptrs" "dA_cs" z
    (by simpa [z,computed,sumPointers,sumOffsets] using h.sum)
    (by simp [z,computed])
    (outputMask_eval p l z (by simpa [z,computed] using h.head) (by simpa [z,computed] using h.column))
  unfold accumulation
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (loadRate_eval p l r a u h.rate
    (headMask_eval p l u h.head) ha))]
  change stepStmts _ v = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some heproduct)]
  change stepStmts _ w = _
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hescan)]
  change stepStmts [storeSum p] z = _
  rw [storeSum,stepStmts.cons_some heout,stepStmts.nil]

theorem tail_executes (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasBias useSoftplus swapped : Bool) (lower : ℝ) (upper : UpperBound) (u : BlockState)
    (h : Ready p l r u) (hd : ∀ off, u.readMem r.dt off = a.dt off)
    (hb : ∀ off, u.readMem r.bias off = a.bias off)
    (ha : ∀ off, u.readMem r.rate off = a.rate off) (hRA : r.rate ≠ r.out) :
    stepStmts (preparation p r hasBias useSoftplus lower upper ++ [storeDt p] ++ accumulation p swapped) u =
      some (result p l r a hasBias useSoftplus lower upper u) := by
  let pre := Preparation.result p l a hasBias useSoftplus lower upper u
  let v := Stores.scatter p l r.out (outOffsets p l) (dtValues p l a hasBias useSoftplus lower upper) pre
  have hpre : Ready p l r pre := Preparation.result_ready h a hasBias useSoftplus lower upper
  have hpreReg := Preparation.result_register p l a hasBias useSoftplus lower upper u
  have hstore := Stores.store_eval p l r.out (outOffsets p l) (dtValues p l a hasBias useSoftplus lower upper)
    "dt_out_ptrs" "dt" pre hpre.out hpreReg (outputMask_eval p l pre hpre.head hpre.column)
  have hv : Ready p l r v := Stores.ready hpre _ _ _
  have hva (off : Nat) : v.readMem r.rate off = a.rate off := by
    rw [Stores.other _ _ _ _ _ _ _ hRA]
    unfold BlockState.readMem
    rw [Preparation.result_mem]
    exact ha off
  have hvd : v.regs .real [p.BH,p.BC] "dt" = some (matrix p (dtValues p l a hasBias useSoftplus lower upper)) := by
    simpa only [v,Stores.registers] using hpreReg
  simp only [List.append_assoc]
  rw [stepStmts.append_some (Preparation.executes p l r a hasBias useSoftplus lower upper u h hd hb)]
  change stepStmts (storeDt p :: _) pre = _
  rw [storeDt,dtOp,stepStmts.cons_some hstore]
  exact accumulation_executes p l r a hasBias useSoftplus swapped lower upper v hv hva hvd

theorem outputs (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (u : BlockState)
    (hOS : r.out ≠ r.sum)
    (hiO : Stores.NoCollision p l (outOffsets p l)) (hiS : Stores.NoCollision p l (sumOffsets p l)) :
    ∀ (i : Fin p.BH) (j : Fin p.BC), activeOutput p l i.val j.val →
      (result p l r a hasBias useSoftplus lower upper u).readMem r.out (outOffsets p l i j) =
        dtValues p l a hasBias useSoftplus lower upper i j ∧
      (result p l r a hasBias useSoftplus lower upper u).readMem r.sum (sumOffsets p l i j) =
        sums p l a hasBias useSoftplus lower upper i j := by
  intro i j hm
  constructor
  · rw [result,Stores.other _ _ _ _ _ _ _ hOS]
    change (Stores.scatter p l r.out _ _ _).readMem _ _ = _
    exact Stores.readback p l r.out _ _ _ hiO i j hm
  · exact Stores.readback p l r.sum _ _ _ hiS i j hm

theorem frame (p : Layout) (l : Launch) (r : Regions) (a : Inputs)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (u : BlockState) :
    BlockState.WriteWithin (footprint p l r) u (result p l r a hasBias useSoftplus lower upper u) := by
  intro region off hn
  have ho := Stores.frame p l r.out (outOffsets p l) (dtValues p l a hasBias useSoftplus lower upper)
    (Preparation.result p l a hasBias useSoftplus lower upper u) region off (fun h => hn (Or.inl h))
  have hs := Stores.frame p l r.sum (sumOffsets p l) (sums p l a hasBias useSoftplus lower upper)
    (computed p l a hasBias useSoftplus lower upper
      (Stores.scatter p l r.out (outOffsets p l) (dtValues p l a hasBias useSoftplus lower upper)
        (Preparation.result p l a hasBias useSoftplus lower upper u))) region off (fun h => hn (Or.inr h))
  change u.mem region off = _
  rw [← Preparation.result_mem p l a hasBias useSoftplus lower upper u]
  exact ho.trans hs

theorem source_executes (p : Layout) (r : Regions) (bias : BiasStorage) (swapped : Bool)
    (dtype out_dtype : FloatDType) (a : Inputs) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (u : BlockState) (hd : ∀ off, u.readMem r.dt off = a.dt off)
    (hb : ∀ off, u.readMem r.bias off = a.bias off)
    (ha : ∀ off, u.readMem r.rate off = a.rate off) (hRA : r.rate ≠ r.out) :
    exec (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).eraseDType.toAlgKernel u =
      some (result p (launch u) r a hasBias useSoftplus lower upper (initial p r u)) := by
  rw [exec,source_body]
  unfold body
  simp only [List.append_assoc]
  rw [stepStmts.append_some (Initialization.executes p r u)]
  exact tail_executes p (launch u) r a hasBias useSoftplus swapped lower upper (initial p r u)
    (Ready.initial p r u) (by simpa [initial] using hd) (by simpa [initial] using hb)
    (by simpa [initial] using ha) hRA

/-- Complete real execution and both initial-array outputs, with a whole-cell
frame. Bounds/flat allocation and the host's infinite-limit default are
separate obligations; this theorem is not yet the public benchmark contract. -/
theorem correct (p : Layout) (r : Regions) (bias : BiasStorage) (swapped : Bool)
    (dtype out_dtype : FloatDType) (a : Inputs) (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (u : BlockState) (hd : ∀ off, u.readMem r.dt off = a.dt off)
    (hb : ∀ off, u.readMem r.bias off = a.bias off)
    (ha : ∀ off, u.readMem r.rate off = a.rate off)
    (hRA : r.rate ≠ r.out) (hOS : r.out ≠ r.sum)
    (hiO : Stores.NoCollision p (launch u) (outOffsets p (launch u)))
    (hiS : Stores.NoCollision p (launch u) (sumOffsets p (launch u))) :
    ∃ v, exec (kernel p r bias swapped dtype out_dtype hasBias useSoftplus lower upper).eraseDType.toAlgKernel u = some v ∧
      (∀ (i : Fin p.BH) (j : Fin p.BC), activeOutput p (launch u) i.val j.val →
        v.readMem r.out (outAddress p (launch u) i.val j.val) =
          prepared p (launch u) a hasBias useSoftplus lower upper i.val j.val ∧
        v.readMem r.sum (sumAddress p (launch u) i.val j.val) =
          cumulative p (launch u) a hasBias useSoftplus lower upper i j) ∧
      BlockState.WriteWithin (footprint p (launch u) r) u v := by
  refine ⟨result p (launch u) r a hasBias useSoftplus lower upper (initial p r u),
    source_executes p r bias swapped dtype out_dtype a hasBias useSoftplus lower upper u hd hb ha hRA,
    outputs p (launch u) r a hasBias useSoftplus lower upper (initial p r u) hOS hiO hiS,?_⟩
  exact frame p (launch u) r a hasBias useSoftplus lower upper (initial p r u)

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Execution
