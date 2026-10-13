import bench.tritonbench_g.flash_attn.Proofs.RealLoop

/-! Numeric contents of the three live loop registers after one entire source
iteration. These definitions retain the source's denom * 0 + alpha operation. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula NormalizedSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

@[simp] private theorem real_zero_add (x : WithBot ℝ) :
    NumericDType.real.add (some 0) x = x := by
  cases x <;> simp [NumericDType.add, WithBot.realAdd]
  rfl

@[simp] private theorem real_bot_add (x : WithBot ℝ) :
    NumericDType.real.add none x = ⊥ := by
  rfl

@[simp] private theorem nat_add (a b : Nat) : NumericDType.nat.add a b = a + b := rfl
@[simp] private theorem nat_ge (a b : Nat) : ComparableDType.nat.ge a b = decide (b ≤ a) := rfl

@[simp] private theorem select_max (x y : WithBot ℝ) [Decidable (y < x)] :
    (if y < x then x else y) = max x y := by
  by_cases h : y < x
  · simp [h, max_eq_left h.le]
  · simp [h, max_eq_right (le_of_not_gt h)]

@[simp] private theorem max_select {shape : TileShape} (x y : Tile .real shape)
    (b : Broadcast shape shape shape) :
    (Tile.cop ComparableDType.real.gt b x y).select x y =
      ⟨fun i => if ComparableDType.real.gt
        (x.data (b.leftIndex i)) (y.data (b.rightIndex i)) then x.data i else y.data i⟩ := rfl

@[simp] private theorem real_max_select (x y : WithBot ℝ) :
    (if ComparableDType.real.gt x y then x else y) = x ⊔ y := by
  by_cases h : x ≤ y
  · simp [ComparableDType.gt, not_lt_of_ge h, sup_eq_right.mpr h]
  · simp [ComparableDType.gt, lt_of_not_ge h, sup_eq_left.mpr (le_of_not_ge h)]

@[simp] private theorem insert_row (BM BN : Nat) (i : TileIndex [BM]) (j : Fin BN) :
    TileShape.insertAxisIndex [BM, BN] ⟨1, by simp⟩ i j = (i.1,j,PUnit.unit) := rfl

def loaded {shape : TileShape} (s : BlockState) (ptr : Tile .blockPtr shape) :
    Tile .real shape :=
  ⟨fun i => s.readMemValue .real (ptr.data i).region
    ((ptr.data i).address (TileShape.indexToList shape i))⟩

def scores (p : Layout) (s : BlockState) (sn : Nat)
    (offm : Tile .nat [p.BLOCK_M]) (offn : Tile .nat [p.BLOCK_N])
    (q : Tile .real [p.BLOCK_M, p.DIM]) (kp : Tile .blockPtr [p.DIM, p.BLOCK_N]) :
    Tile .real [p.BLOCK_M, p.BLOCK_N] :=
  ⟨fun i => NumericDType.real.add
    (if p.causal then
      if sn + offn.data (i.2.1, PUnit.unit) ≤ offm.data (i.1, PUnit.unit)
        then some 0 else none
      else some 0)
    ((Tile.dot [] q (loaded s kp)).data i)⟩

def updatedMax (p : Layout) (m : Tile .real [p.BLOCK_M])
    (sc : Tile .real [p.BLOCK_M, p.BLOCK_N]) : Tile .real [p.BLOCK_M] :=
  ⟨fun i => m.data i ⊔ Finset.univ.sup (fun j : Fin p.BLOCK_N => sc.data (i.1,j,PUnit.unit))⟩

def rescale (p : Layout) (m mn : Tile .real [p.BLOCK_M]) : Tile .real [p.BLOCK_M] :=
  Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame .nil) m mn)

def probabilities (p : Layout) (sc : Tile .real [p.BLOCK_M,p.BLOCK_N])
    (mn : Tile .real [p.BLOCK_M]) : Tile .real [p.BLOCK_M,p.BLOCK_N] :=
  Tile.uop WithBot.realExp2 (Tile.bop NumericDType.real.sub (.consSame (.consR .nil))
    sc (Tile.expandDim ⟨1, by simp⟩ mn))

def updatedDenom (p : Layout) (den alpha : Tile .real [p.BLOCK_M])
    (prob : Tile .real [p.BLOCK_M,p.BLOCK_N]) : Tile .real [p.BLOCK_M] :=
  Tile.bop NumericDType.real.add (.consSame .nil)
    (Tile.bop NumericDType.real.mul (.consSame .nil) den alpha)
    (Tile.reduceSumDrop ⟨1, by simp⟩ prob)

def updatedAcc (p : Layout) (s : BlockState) (den alpha : Tile .real [p.BLOCK_M])
    (acc : Tile .real [p.BLOCK_M,p.DIM]) (prob : Tile .real [p.BLOCK_M,p.BLOCK_N])
    (vp : Tile .blockPtr [p.BLOCK_N,p.DIM]) : Tile .real [p.BLOCK_M,p.DIM] :=
  let scale := Tile.bop NumericDType.real.add (.consSame .nil)
    (Tile.bop NumericDType.real.mul .scalarR den (Tile.scalar (some 0))) alpha
  Tile.bop NumericDType.real.add (.consSame (.consSame .nil))
    (Tile.bop NumericDType.real.mul (.consSame (.consR .nil)) acc
      (Tile.expandDim ⟨1, by simp⟩ scale))
    (Tile.dot [] prob (loaded s vp))

theorem iteration_numeric (p : Layout) (s : BlockState) (sn : Nat)
    (offm : Tile .nat [p.BLOCK_M]) (offn : Tile .nat [p.BLOCK_N])
    (q : Tile .real [p.BLOCK_M,p.DIM])
    (m den : Tile .real [p.BLOCK_M]) (acc : Tile .real [p.BLOCK_M,p.DIM])
    (kp : Tile .blockPtr [p.DIM,p.BLOCK_N]) (vp : Tile .blockPtr [p.BLOCK_N,p.DIM])
    (hBN : 0 < p.BLOCK_N)
    (hsn : s.regs .nat [] "start_n" = some (Tile.scalar sn))
    (hom : s.regs .nat [p.BLOCK_M] "off_m" = some offm)
    (hon : s.regs .nat [p.BLOCK_N] "off_n" = some offn)
    (hq : s.regs .real [p.BLOCK_M,p.DIM] "q" = some q)
    (hm : s.regs .real [p.BLOCK_M] "max" = some m)
    (hd : s.regs .real [p.BLOCK_M] "denom" = some den)
    (ha : s.regs .real [p.BLOCK_M,p.DIM] "out_buffer" = some acc)
    (hkp : s.regs .blockPtr [p.DIM,p.BLOCK_N] "K_block_ptr" = some kp)
    (hvp : s.regs .blockPtr [p.BLOCK_N,p.DIM] "V_block_ptr" = some vp) :
    let t := (iteration p s sn offm offn q m den acc kp vp
      hBN hsn hom hon hq hm hd ha hkp hvp).val
    let sc := scores p s sn offm offn q kp
    let mn := updatedMax p m sc
    let alpha := rescale p m mn
    let prob := probabilities p sc mn
    t.regs .real [p.BLOCK_M] "max" = some mn ∧
    t.regs .real [p.BLOCK_M] "denom" = some (updatedDenom p den alpha prob) ∧
    t.regs .real [p.BLOCK_M,p.DIM] "out_buffer" =
      some (updatedAcc p s den alpha acc prob vp) := by
  rcases p with ⟨sqh,sqs,sqd,sks,skd,svd,sos,sod,seq,bm,dim,bn,c,scale⟩
  cases c <;> simp [-Tile.scalar_eta, iteration, scores, updatedMax, rescale, probabilities,
    updatedDenom, updatedAcc, loaded, Finset.sup'_eq_sup,
    Tile.bop, Tile.cop, Tile.select, Tile.expandDim, Tile.scalar,
    Broadcast.leftIndex, Broadcast.rightIndex, TileShape.insertAxisIndex]
  all_goals
    refine ⟨?_, ?_, ?_⟩
    · funext ⟨i, ⟨⟩⟩
      simp only [← Nat.not_lt, ite_not, select_max]
      rfl
    · funext ⟨i, ⟨⟩⟩
      simp only [← Nat.not_lt, ite_not, select_max]
      rfl
    · funext ⟨i, j, ⟨⟩⟩
      simp only [← Nat.not_lt, ite_not, select_max]
      rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
