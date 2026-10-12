import VeriTile.Triton
import VeriTile.Triton.Semantics.BlockPadding
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.chunk_gla_simple.Proofs.RealProgram

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
/-- No-mask 2D block-pointer load through a *bound register* `name` holding the
block-pointer tile produced by `makeBlockPtrDynOffsets`. Lane `(i,j)` reads the
genuine memory cell when in-bounds, else `0`. -/
theorem load_bp_2d_ref (rg : RegionName) (s : BlockState) (name : RegName)
    (base rows cols BT BS strideT strideS rowOff colOff : Nat)
    (hreg : s.regs TileDType.blockPtr [BT, BS] name = some
      ⟨fun _ => BlockPtr.mk rg base [rows, cols] [BT, BS] [strideT, strideS]
        [rowOff, colOff]⟩) (hu : s.undef = fun _ _ => 0) :
    evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT, BS] name) [0, 1]) MaskOpt.none .unspecified) s
    = some ⟨fun idx : TileIndex [BT, BS] =>
        if (rowOff + idx.1.val < rows ∧ colOff + idx.2.1.val < cols) then
          some (s.readMem rg (base + (rowOff + idx.1.val) * strideT
            + (colOff + idx.2.1.val) * strideS))
        else some 0⟩ := by
  rw [evalOp_load_padding_of_undef_zero _ _ _ _ s hu]
  simp only [evalOp, evalOp_ref, hreg, List.mapM, List.mapM.loop, bind, Option.bind, Tile.scalar,
    List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append, List.cons_append]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, j, rest⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
    BlockState.readMemValue_real]
  by_cases h : rowOff + i.val < rows ∧ colOff + j.val < cols
  · simp only [h, and_self, decide_true, if_true, and_true, true_and]
  · simp only [h, decide_false, if_false, BlockState.defaultCarrier, if_neg]
    rfl

/-- No-mask 1D block-pointer load through a bound register. -/
theorem load_bp_1d_ref (rg : RegionName) (s : BlockState) (name : RegName)
    (base len BT stride off : Nat)
    (hreg : s.regs TileDType.blockPtr [BT] name = some
      ⟨fun _ => BlockPtr.mk rg base [len] [BT] [stride] [off]⟩) (hu : s.undef = fun _ _ => 0) :
    evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT] name) [0]) MaskOpt.none .unspecified) s
    = some ⟨fun idx : TileIndex [BT] =>
        if (off + idx.1.val < len) then
          some (s.readMem rg (base + (off + idx.1.val) * stride))
        else some 0⟩ := by
  rw [evalOp_load_padding_of_undef_zero _ _ _ _ s hu]
  simp only [evalOp, evalOp_ref, hreg, List.mapM, List.mapM.loop, bind, Option.bind, Tile.scalar,
    List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append, List.cons_append]
  refine congrArg some ?_
  ext idx
  obtain ⟨i, rest⟩ := idx
  simp only [TileShape.indexToList, BlockPtr.address_1d, BlockPtr.inBounds_1d,
    BlockState.readMemValue_real]
  by_cases h : off + i.val < len
  · simp only [h, decide_true, if_true]
  · simp only [h, decide_false, if_false, BlockState.defaultCarrier, if_neg]
    rfl

/-- Evaluation unfolding for the `≥` comparison op. -/
theorem evalOp_ge_def {dtype : TileDType} {a b shape : TileShape}
    (h : ComparableDType dtype) (bc : Broadcast a b shape)
    (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.ge h bc x y) s = (do
      let vx ← evalOp x s
      let vy ← evalOp y s
      some (Tile.cop h.ge bc vx vy)) := by
  simp [evalOp]

/-- **2D `makeBlockPtrDynOffsets` eval recipe.** Given the base / two offset ops
evaluate to scalars, the block-pointer op evaluates to the constant tile holding
`BlockPtr.mk rg base parentShape blockShape strides [rowOff, colOff]`. This is the
register value `load_bp_2d_ref` consumes. -/
theorem makeBlockPtr_2d_eval (rg : RegionName) (s : BlockState)
    (baseOp : Op .nat []) (rowOffOp colOffOp : Op .nat [])
    (parentShape blockShape strides : List Nat)
    (base rowOff colOff : Nat)
    (hbase : evalOp baseOp s = some (Tile.scalar base))
    (hrow : evalOp rowOffOp s = some (Tile.scalar rowOff))
    (hcol : evalOp colOffOp s = some (Tile.scalar colOff)) :
    evalOp (Op.makeBlockPtrDynOffsets rg baseOp parentShape blockShape strides
        [rowOffOp, colOffOp]) s
      = some (⟨fun _ => BlockPtr.mk rg base parentShape blockShape strides
          [rowOff, colOff]⟩ : Tile .blockPtr blockShape) := by
  simp only [evalOp, hbase, hrow, hcol, List.mapM, List.mapM.loop, bind, Option.bind,
    Tile.scalar, List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append,
    List.cons_append]

/-- **1D `makeBlockPtrDynOffsets` eval recipe.** -/
theorem makeBlockPtr_1d_eval (rg : RegionName) (s : BlockState)
    (baseOp : Op .nat []) (offOp : Op .nat [])
    (parentShape blockShape strides : List Nat)
    (base off : Nat)
    (hbase : evalOp baseOp s = some (Tile.scalar base))
    (hoff : evalOp offOp s = some (Tile.scalar off)) :
    evalOp (Op.makeBlockPtrDynOffsets rg baseOp parentShape blockShape strides
        [offOp]) s
      = some (⟨fun _ => BlockPtr.mk rg base parentShape blockShape strides
          [off]⟩ : Tile .blockPtr blockShape) := by
  simp only [evalOp, hbase, hoff, List.mapM, List.mapM.loop, bind, Option.bind,
    Tile.scalar, List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append,
    List.cons_append]

/-! ## Per-statement op-eval recipes (body + post-loop) -/

/-- **`b_o += dot(b_q, b_h)` (and `b_s += dot(b_q, b_k)`) recipe.** -/
theorem accDot_op_eval (s : BlockState) (M K N : Nat) (accName qName hName : RegName)
    (acctile : Tile .real [M, N]) (qtile : Tile .real [M, K]) (htile : Tile .real [K, N])
    (hacc : s.regs .real [M, N] accName = some acctile)
    (hq : s.regs .real [M, K] qName = some qtile)
    (hh : s.regs .real [K, N] hName = some htile) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] accName)
        (Op.dot (batch := []) (Op.ref .real [M, K] qName) (Op.ref .real [K, N] hName) {allowTF32 := some Bool.false})) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          acctile (Tile.dot [] qtile htile)) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] qName) (Op.ref .real [K, N] hName) {allowTF32 := some Bool.false}) s
      = some (Tile.dot [] qtile htile) := by rw [evalOp_dot]; simp [hq, hh]
  have hdot2 : @evalOp TileDType.real [M, N]
      (Op.dot (batch := []) (Op.ref .real [M, K] qName) (Op.ref .real [K, N] hName) {allowTF32 := some Bool.false}) s
      = some (Tile.dot [] qtile htile) := hdot
  rw [evalOp_add]
  simp only [evalOp_ref, hacc, hdot2, Option.bind_eq_bind, Option.bind_some, Option.bind]

/-- **`b_o *= exp(b_g)[:, None]` recipe** (post-loop gate decay on the output).
The `expandDim` of `exp(b_g)` is proved naturally then defeq-coerced to `[M,1]`. -/
theorem mulExpCol_op_eval (s : BlockState) (M N : Nat) (hax : 1 < [M].length.succ)
    (oName gName : RegName) (otile : Tile .real [M, N]) (gtile : Tile .real [M])
    (ho : s.regs .real [M, N] oName = some otile)
    (hg : s.regs .real [M] gName = some gtile) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [M, N] oName)
        (Op.expandDim ⟨1, hax⟩ (Op.exp (Op.ref .real [M] gName)))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          otile (Tile.expandDim ⟨1, hax⟩ (Tile.uop WithBot.realExp gtile))) := by
  have hexpN : evalOp (Op.expandDim ⟨1, hax⟩ (Op.exp (Op.ref .real [M] gName))) s
      = some (Tile.expandDim ⟨1, hax⟩ (Tile.uop WithBot.realExp gtile)) := by
    rw [evalOp_expandDim, evalOp_exp]; simp [hg]
  have hexp2 : @evalOp TileDType.real [M, 1]
      (Op.expandDim ⟨1, hax⟩ (Op.exp (Op.ref .real [M] gName))) s
      = some (Tile.expandDim ⟨1, hax⟩ (Tile.uop WithBot.realExp gtile)) := hexpN
  rw [evalOp_mul]
  simp only [evalOp_ref, ho, hexp2, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- **`b_s *= exp(b_g[:, None] − b_g[None, :])` recipe** (post-loop score gate decay).
`b_g[:,None]` is `expandDim` axis 1 (`[M,1]`), `b_g[None,:]` is axis 0 (`[1,M]`),
both proved naturally then defeq-coerced. -/
theorem mulExpOuter_op_eval (s : BlockState) (M : Nat)
    (hax1 : 1 < [M].length.succ) (hax0 : 0 < [M].length.succ)
    (sName gName : RegName) (stile : Tile .real [M, M]) (gtile : Tile .real [M])
    (hs : s.regs .real [M, M] sName = some stile)
    (hg : s.regs .real [M] gName = some gtile) :
    evalOp (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, M] sName)
        (Op.exp (Op.sub .real (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, hax1⟩ (Op.ref .real [M] gName))
          (Op.expandDim ⟨0, hax0⟩ (Op.ref .real [M] gName))))) s
      = some (Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          stile (Tile.uop WithBot.realExp
            (Tile.bop NumericDType.real.sub (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Tile.expandDim ⟨1, hax1⟩ gtile) (Tile.expandDim ⟨0, hax0⟩ gtile)))) := by
  have hexp1N : evalOp (Op.expandDim ⟨1, hax1⟩ (Op.ref .real [M] gName)) s
      = some (Tile.expandDim ⟨1, hax1⟩ gtile) := by rw [evalOp_expandDim]; simp [hg]
  have hexp1 : @evalOp TileDType.real [M, 1] (Op.expandDim ⟨1, hax1⟩ (Op.ref .real [M] gName)) s
      = some (Tile.expandDim ⟨1, hax1⟩ gtile) := hexp1N
  have hexp0N : evalOp (Op.expandDim ⟨0, hax0⟩ (Op.ref .real [M] gName)) s
      = some (Tile.expandDim ⟨0, hax0⟩ gtile) := by rw [evalOp_expandDim]; simp [hg]
  have hexp0 : @evalOp TileDType.real [1, M] (Op.expandDim ⟨0, hax0⟩ (Op.ref .real [M] gName)) s
      = some (Tile.expandDim ⟨0, hax0⟩ gtile) := hexp0N
  have hsubN : evalOp (Op.sub .real (Broadcast.consR (Broadcast.consL Broadcast.nil))
      (Op.expandDim ⟨1, hax1⟩ (Op.ref .real [M] gName))
      (Op.expandDim ⟨0, hax0⟩ (Op.ref .real [M] gName))) s
      = some (Tile.bop NumericDType.real.sub (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Tile.expandDim ⟨1, hax1⟩ gtile) (Tile.expandDim ⟨0, hax0⟩ gtile)) := by
    rw [evalOp_sub]
    simp only [hexp1, hexp0, Option.bind, Option.bind_eq_bind, Option.bind_some]
  rw [evalOp_mul, evalOp_exp]
  simp only [evalOp_ref, hs, hsubN, Option.bind, Option.bind_eq_bind, Option.bind_some]

/-- **`b_s = where(m_s, b_s, broadcast 0)` recipe** (causal mask).
`(Op.const 0).broadcast [M,M]` evaluates to the all-`some 0` tile. -/
theorem whereMask_op_eval (s : BlockState) (M : Nat)
    (mName sName : RegName) (mtile : Tile .bool [M, M]) (stile : Tile .real [M, M])
    (hm : s.regs .bool [M, M] mName = some mtile)
    (hs : s.regs .real [M, M] sName = some stile) :
    evalOp (Op.where (Op.ref .bool [M, M] mName) (Op.ref .real [M, M] sName)
        (Op.broadcast (Op.const 0) [M, M])) s
      = some (Tile.select mtile stile (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [M, M])) := by
  have hbc : evalOp (Op.broadcast (Op.const 0) [M, M]) s
      = some (⟨fun _ => some (0 : ℝ)⟩ : Tile .real [M, M]) := by
    simp [evalOp]
  rw [evalOp_where]
  simp only [evalOp_ref, hm, hs, hbc, Option.bind_eq_bind, Option.bind_some]

/-- **`m_s = o_i[:, None] >= o_i[None, :]` recipe** (causal-mask boolean). -/
theorem msGe_op_eval (s : BlockState) (BT : Nat)
    (hax1 : 1 < [BT].length.succ) (hax0 : 0 < [BT].length.succ)
    (oiName : RegName) (oitile : Tile .nat [BT])
    (ho : s.regs .nat [BT] oiName = some oitile) :
    evalOp (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, hax1⟩ (Op.ref .nat [BT] oiName))
        (Op.expandDim ⟨0, hax0⟩ (Op.ref .nat [BT] oiName))) s
      = some (Tile.cop ComparableDType.nat.ge (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Tile.expandDim ⟨1, hax1⟩ oitile) (Tile.expandDim ⟨0, hax0⟩ oitile)) := by
  have he1N : evalOp (Op.expandDim ⟨1, hax1⟩ (Op.ref .nat [BT] oiName)) s
      = some (Tile.expandDim ⟨1, hax1⟩ oitile) := by rw [evalOp_expandDim]; simp [ho]
  have he1 : @evalOp TileDType.nat [BT, 1] (Op.expandDim ⟨1, hax1⟩ (Op.ref .nat [BT] oiName)) s
      = some (Tile.expandDim ⟨1, hax1⟩ oitile) := he1N
  have he0N : evalOp (Op.expandDim ⟨0, hax0⟩ (Op.ref .nat [BT] oiName)) s
      = some (Tile.expandDim ⟨0, hax0⟩ oitile) := by rw [evalOp_expandDim]; simp [ho]
  have he0 : @evalOp TileDType.nat [1, BT] (Op.expandDim ⟨0, hax0⟩ (Op.ref .nat [BT] oiName)) s
      = some (Tile.expandDim ⟨0, hax0⟩ oitile) := he0N
  rw [evalOp_ge_def, he1, he0]; rfl

/-- **`b_o = (b_o + dot(b_s, b_v)) * scale` recipe** (final accumulation + scale). -/
theorem accScale_op_eval (s : BlockState) (M Tt N : Nat) (sc : ℝ)
    (oName sName vName : RegName)
    (otile : Tile .real [M, N]) (stile : Tile .real [M, Tt]) (vtile : Tile .real [Tt, N])
    (ho : s.regs .real [M, N] oName = some otile)
    (hs : s.regs .real [M, Tt] sName = some stile)
    (hv : s.regs .real [Tt, N] vName = some vtile) :
    evalOp (Op.mul .real Broadcast.scalarR
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [M, N] oName)
          (Op.dot (batch := []) (Op.ref .real [M, Tt] sName) (Op.ref .real [Tt, N] vName) {allowTF32 := some Bool.false}))
        (Op.const sc)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            otile (Tile.dot [] stile vtile))
          (Tile.scalar (some sc : WithBot ℝ))) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [M, Tt] sName) (Op.ref .real [Tt, N] vName) {allowTF32 := some Bool.false}) s
      = some (Tile.dot [] stile vtile) := by rw [evalOp_dot]; simp [hs, hv]
  have hdot2 : @evalOp TileDType.real [M, N]
      (Op.dot (batch := []) (Op.ref .real [M, Tt] sName) (Op.ref .real [Tt, N] vName) {allowTF32 := some Bool.false}) s
      = some (Tile.dot [] stile vtile) := hdot
  rw [evalOp_mul, evalOp_add]
  simp only [evalOp_ref, evalOp_const, ho, hdot2, Option.bind_eq_bind, Option.bind_some, Option.bind]

/-! ## Matmul (dot) element primitives -/

/-- A `WithBot ℝ` sum of `some`-valued cells is `some` of the real sum. -/
theorem withBot_sum_some {N : Nat} (g : Fin N → ℝ) :
    @Finset.sum (Fin N) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ)) = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

/-- **2D dot element lemma.** For all-`some` operand tiles `a : [M,K]`, `b : [K,N]`,
the `(m, n)` cell of `dot a b` is `Σ_e a[m,e]·b[e,n]`. The generic matmul-readout
primitive used for `b_o += q·h`, `b_s += q·k`, and `b_o += b_s·v`. -/
theorem dot2d_elem {M K N : Nat} (a : Tile .real [M, K]) (b : Tile .real [K, N])
    (m : Fin M) (n : Fin N) (fa fb : Fin K → ℝ)
    (ha : ∀ e : Fin K, a.data (m, e, PUnit.unit) = some (fa e))
    (hb : ∀ e : Fin K, b.data (e, n, PUnit.unit) = some (fb e)) :
    (Tile.dot [] a b).data (m, n, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin K => fa e * fb e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (a.data (m, e, PUnit.unit)) (b.data (e, n, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ (fun e => (some (fa e * fb e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [ha e, hb e]; rfl)]
  exact withBot_sum_some _

/-- Scalar offset op `name * c` evaluates to `scalar (val * c)` given `name = val`. -/
theorem mulConst_eval (s : BlockState) (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOp (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  rw [evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hr, Option.bind_eq_bind, Option.bind_some]
  rfl


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
/-- Loaded-tile abbreviations: the masked block-pointer loads of `q`, `k`, `h`.
Stated exactly as `load_bp_2d_ref` emits them (offset `kb * BK` on the key axis). -/
noncomputable def bqTile (s : BlockState) (q : RegionName) (s_k_h s_k_t T K BT BK kb : Nat) :
    Tile .real [BT, BK] :=
  ⟨fun idx => if (s.pids 1 * BT + idx.1.val < T ∧ kb * BK + idx.2.1.val < K) then
      some (s.readMem q (s.pids 2 * s_k_h + (s.pids 1 * BT + idx.1.val) * s_k_t
        + (kb * BK + idx.2.1.val) * 1)) else some 0⟩

noncomputable def bkTile (s : BlockState) (k : RegionName) (s_k_h s_k_t T K BT BK kb : Nat) :
    Tile .real [BK, BT] :=
  ⟨fun idx => if (kb * BK + idx.1.val < K ∧ s.pids 1 * BT + idx.2.1.val < T) then
      some (s.readMem k (s.pids 2 * s_k_h + (kb * BK + idx.1.val) * 1
        + (s.pids 1 * BT + idx.2.1.val) * s_k_t)) else some 0⟩

noncomputable def bhTile (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BT BK BV kb : Nat) :
    Tile .real [BK, BV] :=
  ⟨fun idx => if (kb * BK + idx.1.val < K ∧ s.pids 0 * BV + idx.2.1.val < V) then
      some (s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + (kb * BK + idx.1.val) * s_h_t
        + (s.pids 0 * BV + idx.2.1.val) * 1)) else some 0⟩

/-- **Loop-body execution at key block `kb`.** Given the prologue register
readbacks on the loop-entry state `sl` (program ids, `i_k = kb`, the current
`b_o`/`b_s`), the 8-statement loop body steps to a final state whose `b_o`/`b_s`
registers hold the two matmul accumulations of the masked `q`/`k`/`h` loads. -/
theorem chunkGlaLoopBody_steps (q k v h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t T K V BT BK BV : Nat) (kb : Nat) (sl : BlockState)
    (boTile : Tile .real [BT, BV]) (bsTile : Tile .real [BT, BT])
    (hiv : sl.regs .nat [] "i_v" = some (Tile.scalar (sl.pids 0)))
    (hit : sl.regs .nat [] "i_t" = some (Tile.scalar (sl.pids 1)))
    (hibh : sl.regs .nat [] "i_bh" = some (Tile.scalar (sl.pids 2)))
    (hik : sl.regs .nat [] "i_k" = some (Tile.scalar kb))
    (hbo : sl.regs .real [BT, BV] "b_o" = some boTile)
    (hbs : sl.regs .real [BT, BT] "b_s" = some bsTile)
    (hu : sl.undef = fun _ _ => 0) :
    ∃ sF, stepStmts (chunkGlaLoopBody q k v h s_k_h s_k_t s_h_h s_h_t T K V BT BK BV) sl = some sF
      ∧ sF.pids = sl.pids ∧ sF.mem = sl.mem ∧ sF.undef = sl.undef
      ∧ sF.regs .nat [] "i_v" = some (Tile.scalar (sl.pids 0))
      ∧ sF.regs .nat [] "i_t" = some (Tile.scalar (sl.pids 1))
      ∧ sF.regs .nat [] "i_bh" = some (Tile.scalar (sl.pids 2))
      ∧ sF.regs .bool [BT, BT] "m_s" = sl.regs .bool [BT, BT] "m_s"
      ∧ sF.regs .real [BT, BV] "b_o" = some
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            boTile (Tile.dot [] (bqTile sl q s_k_h s_k_t T K BT BK kb)
              (bhTile sl h s_h_h s_h_t K V BT BK BV kb)))
      ∧ sF.regs .real [BT, BT] "b_s" = some
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            bsTile (Tile.dot [] (bqTile sl q s_k_h s_k_t T K BT BK kb)
              (bkTile sl k s_k_h s_k_t T K BT BK kb))) := by
  -- abbreviations for the makeBlockPtr base/offset scalar evals (proved by simp,
  -- peeling the accumulated setReg chain via the @[simp 1100] ref/setReg lemmas)
  have hbaseH : ∀ st : BlockState, st.regs .nat [] "i_bh" = some (Tile.scalar (sl.pids 2)) →
      st.regs .nat [] "i_t" = some (Tile.scalar (sl.pids 1)) →
      evalOp (Op.add .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
        (Op.mul .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K)) (Op.constNat V))) st
      = some (Tile.scalar (sl.pids 2 * s_h_h + sl.pids 1 * K * V)) := by
    intro st hbh ht
    rw [evalOp_add, mulConst_eval st "i_bh" _ _ hbh, evalOp_mul,
      mulConst_eval st "i_t" (sl.pids 1) K ht]
    simp only [evalOp_constNat, Option.bind_eq_bind, Option.bind_some]
    rfl
  -- step p_q, p_k, p_h
  unfold chunkGlaLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval q sl _ _ _ [T, K] [BT, BK] [s_k_t, 1] (sl.pids 2 * s_k_h)
      (sl.pids 1 * BT) (kb * BK)
      (mulConst_eval sl "i_bh" _ _ hibh) (mulConst_eval sl "i_t" _ _ hit)
      (mulConst_eval sl "i_k" _ _ hik)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval k _ _ _ _ [K, T] [BK, BT] [1, s_k_t] (sl.pids 2 * s_k_h)
      (kb * BK) (sl.pids 1 * BT)
      (mulConst_eval _ "i_bh" _ _ (by simp [hibh])) (mulConst_eval _ "i_k" _ _ (by simp [hik]))
      (mulConst_eval _ "i_t" _ _ (by simp [hit]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval h _ _ _ _ [K, V] [BK, BV] [s_h_t, 1]
      (sl.pids 2 * s_h_h + sl.pids 1 * K * V) (kb * BK) (sl.pids 0 * BV)
      (hbaseH _ (by simp [hibh]) (by simp [hit])) (mulConst_eval _ "i_k" _ _ (by simp [hik]))
      (mulConst_eval _ "i_v" _ _ (by simp [hiv]))))]
  -- step loads b_q, b_k, b_h
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref q _ "p_q" (sl.pids 2 * s_k_h) T K BT BK s_k_t 1 (sl.pids 1 * BT) (kb * BK)
      (by simp) (by simpa using hu)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref k _ "p_k" (sl.pids 2 * s_k_h) K T BK BT 1 s_k_t (kb * BK) (sl.pids 1 * BT)
      (by simp) (by simpa using hu)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref h _ "p_h" (sl.pids 2 * s_h_h + sl.pids 1 * K * V) K V BK BV s_h_t 1
      (kb * BK) (sl.pids 0 * BV) (by simp) (by simpa using hu)))]
  -- step b_o += dot(b_q, b_h) and b_s += dot(b_q, b_k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accDot_op_eval _ BT BK BV "b_o" "b_q" "b_h" boTile (bqTile sl q s_k_h s_k_t T K BT BK kb)
      (bhTile sl h s_h_h s_h_t K V BT BK BV kb) (by simp [hbo]) (by simp [bqTile]) (by simp [bhTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accDot_op_eval _ BT BK BT "b_s" "b_q" "b_k" bsTile (bqTile sl q s_k_h s_k_t T K BT BK kb)
      (bkTile sl k s_k_h s_k_t T K BT BK kb) (by simp [hbs]) (by simp [bqTile]) (by simp [bkTile])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, rfl, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · simp [hiv]
  · simp [hit]
  · simp [hibh]
  · simp
  · simp
  · simp


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Real
