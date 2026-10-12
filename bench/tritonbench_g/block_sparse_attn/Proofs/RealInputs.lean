import bench.tritonbench_g.block_sparse_attn.Proofs.RealAddressFormula

/-! Initial memory represents independent Q/K/V arrays and the visited CSR
metadata. The contract contains only input cells and index validity, not an
execution, output equation or precomputed score. Partial query blocks and
undefined masked loads are allowed. Row/column strides stay independent. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealInputs
open VeriTile.Triton RealAddressFormula
noncomputable section
set_option linter.unusedSimpArgs false

def width (BD : Nat) (p : RealProgram.Parameters) : Nat :=
  if 2 ≤ p.blocks then 2 * BD else BD

def rowAddress (s : BlockState) (p : RealProgram.Parameters) (next : Bool) : Nat :=
  (schedule s p).layout * p.rowStride + s.pids 0 + if next then 1 else 0

def columnSlot (s : BlockState) (p : RealProgram.Parameters) (xs : CSRMath.Inputs) (visit : Nat) : Int :=
  ((schedule s p).layout * p.columnStride : Nat) + CSRMath.first (schedule s p) xs + (visit : Int)

structure Memory (s : BlockState) (BD : Nat) (p : RealProgram.Parameters) (xs : CSRMath.Inputs) : Prop where
  rows : ∀ next, s.readMemValue .int (Region.cast p.rows) (rowAddress s p next) =
    xs.rowIndex (rowAddress s p next)
  columns : ∀ visit, visit < CSRMath.count (schedule s p) xs →
    s.readMemValue .int (Region.cast p.cols) (columnSlot s p xs visit).toNat =
      xs.columnIndex (columnSlot s p xs visit)
  query : ∀ r d, r < p.context → d < width BD p →
    s.readMem p.Q (RealSetup.qBase s p + r * p.qRow + d) = xs.query r d
  key : ∀ r d, 0 ≤ r → r < (p.context : Int) → d < width BD p →
    s.readMem p.K (RealSetup.kBase s p + r.toNat * p.kRow + d) = xs.key r d
  value : ∀ r d, 0 ≤ r → r < (p.context : Int) → d < width BD p →
    s.readMem p.V (RealSetup.vBase s p + r.toNat * p.vRow + d) = xs.value r d

/-- Signed visits may be negative when the complete CSR column slot is
nonnegative. Selected blocks themselves are ordinary nonnegative key rows.
Allocation upper bounds and padding for unmasked loads are checked separately. -/
structure Valid (s : BlockState) (p : RealProgram.Parameters) (xs : CSRMath.Inputs) : Prop where
  slots : ∀ visit, visit < CSRMath.count (schedule s p) xs → 0 ≤ columnSlot s p xs visit
  columns : ∀ visit, visit < CSRMath.count (schedule s p) xs →
    0 ≤ xs.columnIndex (columnSlot s p xs visit)

theorem channel_lt (BD : Nat) (p : RealProgram.Parameters) (second : Bool) (d : Fin BD)
    (hb : second = Bool.true → 2 ≤ p.blocks) : channel BD second d < width BD p := by
  have hd := d.isLt
  cases second <;> by_cases h : 2 ≤ p.blocks <;> simp [channel, width, h] at * <;> omega

theorem interval_matches (s : BlockState) (BD : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hm : Memory s BD p xs) :
    RealLoop.first s p = CSRMath.first (schedule s p) xs ∧
      RealLoop.last s p = CSRMath.last (schedule s p) xs := by
  simpa [RealLoop.first, RealLoop.last, RealRows.bound, RealRows.address,
    rowAddress, schedule, CSRMath.first, CSRMath.last] using And.intro (hm.rows Bool.false) (hm.rows Bool.true)

theorem count_matches (s : BlockState) (BD : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hm : Memory s BD p xs) :
    RealLoop.count s p = CSRMath.count (schedule s p) xs := by
  simp only [RealLoop.count, CSRMath.count, (interval_matches s BD p xs hm).1,
    (interval_matches s BD p xs hm).2]

theorem start_matches (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hm : Memory s BD p xs) (visit : Nat)
    (hv : visit < CSRMath.count (schedule s p) xs) :
    RealVisit.start s (RealLoop.operands s BM BN BD p visit) p =
      RealValueFormula.keyStart (schedule s p) xs BN visit := by
  have hc := hm.columns visit hv
  simpa [RealVisit.start, RealLoop.operands, RealColumn.column, RealColumn.address,
    (interval_matches s BD p xs hm).1, columnSlot, schedule, RealValueFormula.keyStart,
    Int.add_assoc] using congrArg (fun c : Int => c * (BN : Int)) hc

theorem start_nonnegative (s : BlockState) (BN : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hvalid : Valid s p xs) (visit : Nat)
    (hv : visit < CSRMath.count (schedule s p) xs) :
    0 ≤ RealValueFormula.keyStart (schedule s p) xs BN visit :=
  mul_nonneg (hvalid.columns visit hv) (Int.natCast_nonneg BN)

theorem key_nonnegative (s : BlockState) (BN : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hvalid : Valid s p xs) (visit : Nat)
    (hv : visit < CSRMath.count (schedule s p) xs) (j : Fin BN) :
    0 ≤ CSRMath.selectedKey (schedule s p) xs BN visit j := by
  have hs := start_nonnegative s BN p xs hvalid visit hv
  change 0 ≤ RealValueFormula.keyStart (schedule s p) xs BN visit + (j.val : Int)
  omega

theorem query_matches (s : BlockState) (BM BD : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hm : Memory s BD p xs) (second : Bool)
    (hb : second = Bool.true → 2 ≤ p.blocks) (r : Fin BM) (d : Fin BD)
    (hr : RealSetup.rows s BM r < p.context) :
    RealQuery.loaded s (RealSetup.queryPointers s BM BD p) (RealSetup.rows s BM) p.context p.evenM second r d =
      xs.query (RealSetup.rows s BM r) (channel BD second d) := by
  apply RealQuery.loaded_eq_input s _ _ _ _ _ xs.query r d hr
  rw [query_address]
  exact hm.query _ _ hr (channel_lt BD p second d hb)

theorem key_matches (s : BlockState) (BN BD : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hm : Memory s BD p xs) (hvalid : Valid s p xs)
    (second : Bool) (hb : second = Bool.true → 2 ≤ p.blocks) (visit : Nat)
    (hv : visit < CSRMath.count (schedule s p) xs) (j : Fin BN) (d : Fin BD)
    (hk : CSRMath.selectedKey (schedule s p) xs BN visit j < (p.context : Int)) :
    s.readMem (RealKey.address (RealSetup.keyPointers s BN BD p)
      (RealValueFormula.keyStart (schedule s p) xs BN visit) p.kRow second d j).1
      (RealKey.address (RealSetup.keyPointers s BN BD p)
        (RealValueFormula.keyStart (schedule s p) xs BN visit) p.kRow second d j).2 =
      xs.key (CSRMath.selectedKey (schedule s p) xs BN visit j) (channel BD second d) := by
  rw [key_address s BN BD p second _ (start_nonnegative s BN p xs hvalid visit hv)]
  exact hm.key _ _ (key_nonnegative s BN p xs hvalid visit hv j) hk (channel_lt BD p second d hb)

/-- Causally visible values on a stored query row are necessarily active
loads. The independent array need not describe any masked undefined values. -/
theorem value_matches (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hm : Memory s BD p xs) (hvalid : Valid s p xs)
    (second : Bool) (hb : second = Bool.true → 2 ≤ p.blocks) (visit : Nat)
    (hv : visit < CSRMath.count (schedule s p) xs) (j : Fin BN) (d : Fin BD)
    (hk : CSRMath.selectedKey (schedule s p) xs BN visit j < (p.context : Int)) :
    RealLoop.value s BM BN BD p second visit j d =
      xs.value (CSRMath.selectedKey (schedule s p) xs BN visit j) (channel BD second d) := by
  simp only [RealLoop.value, start_matches s BM BN BD p xs hm visit hv]
  have ha : RealValueLoad.active (RealValueFormula.keyStart (schedule s p) xs BN visit) p.context p.evenN j := by
    right
    change RealValueFormula.keyStart (schedule s p) xs BN visit + (j.val : Int) < (p.context : Int) at hk
    omega
  simp only [RealValueLoad.loaded, ha, ↓reduceIte, RealLoop.operands]
  rw [value_address s BN BD p second _ (start_nonnegative s BN p xs hvalid visit hv)]
  exact hm.value _ _ (key_nonnegative s BN p xs hvalid visit hv j) hk (channel_lt BD p second d hb)

/-- The actual Q/K reads and dots yield the independent score, only on the
observed query row. Out-of-context physical query rows remain unconstrained. -/
theorem score_matches (s : BlockState) (BM BN BD : Nat) (p : RealProgram.Parameters)
    (xs : CSRMath.Inputs) (hm : Memory s BD p xs) (hvalid : Valid s p xs)
    (visit : Nat) (hv : visit < CSRMath.count (schedule s p) xs) (r : Fin BM) (j : Fin BN)
    (hr : RealSetup.rows s BM r < p.context) :
    RealLoop.score s BM BN BD p visit r j =
      CSRMath.score (schedule s p) xs BN BD p.blocks (RealSetup.rows s BM r) p.scale visit j := by
  simp only [RealLoop.score, start_matches s BM BN BD p xs hm visit hv]
  by_cases hvisible : CSRMath.selectedKey (schedule s p) xs BN visit j ≤ (RealSetup.rows s BM r : Int)
  · have hk : CSRMath.selectedKey (schedule s p) xs BN visit j < (p.context : Int) := by omega
    apply RealScore.score_eq_formula
    · exact hr
    · intro d
      exact query_matches s BM BD p xs hm Bool.false (by simp) r d hr
    · intro hb d
      exact query_matches s BM BD p xs hm Bool.true (fun _ => hb) r d hr
    · intro d _
      exact key_matches s BN BD p xs hm hvalid Bool.false (by simp) visit hv j d hk
    · intro hb d _
      exact key_matches s BN BD p xs hm hvalid Bool.true (fun _ => hb) visit hv j d hk
  · simpa only [RealScore.masked, CSRMath.score, RealLoop.operands, RealSetup.rows, schedule,
      CSRMath.selectedKey, RealValueFormula.keyStart] using
      (show (if CSRMath.selectedKey (schedule s p) xs BN visit j ≤ (RealSetup.rows s BM r : Int)
          then _ else (⊥ : WithBot ℝ)) =
        (if CSRMath.selectedKey (schedule s p) xs BN visit j ≤ (RealSetup.rows s BM r : Int)
          then _ else (⊥ : WithBot ℝ)) from by simp only [hvisible, ↓reduceIte])

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealInputs
