import VeriTile.Triton

/-!
# `chunk_gla_simple` — closed-form correctness

`chunk_simple_gla_fwd_kernel_o` is the chunked output pass of simple gated
linear attention. Program `(i_v, i_t, i_bh)`:

* accumulates, over key blocks, the inter-chunk term `b_o += q · h` (with the
  precomputed chunk state `h`) and the intra-chunk score `b_s += q · k`;
* applies the gate decay `b_o *= exp(b_g)`, `b_s[i,j] *= exp(b_g_i - b_g_j)`;
* masks `b_s` to the causal lower triangle (`m_s = i ≥ j`);
* adds the intra-chunk contribution `b_s · v`, scales by `scale`;
* stores the `[BT, BV]` result into `o`.

This file proves the **full per-program kernel** correct against a genuine
mathematical closed form `glaOutput` (NOT the kernel's own emitted value):
every output lane `(i, p)` of the produced tile equals

```
  ( (Σ_e q[i,e]·h[e,p]) · exp(g_i)
    + Σ_j (if i ≥ j then (Σ_e q[i,e]·k[e,j]) · exp(g_i − g_j) else 0) · v[j,p]
  ) · scale
```

over `ℝ`, with the kernel's exact block-pointer layouts.

## Scope

This verifies the per-program `@triton.jit` body. The host launch (3-D grid over
value blocks / time chunks / batch·head rows, host-computed `BK/BV`) is the
trusted boundary; the per-program statement is universally quantified over the
`BlockState`, so it covers every program of the grid.

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` are not modeled. The closed form is proved for the single-key-block
regime `K = BK` (`ceil(K/BK) = 1`), which is exactly every checked Python case
(`K=BK=64` for cases 1–3, `K=BK=32` for case 4). The `.to(...)` casts erase to
the identity at the algorithm layer. Output store offsets are injective on the
active region (the per-case side condition).

## Proof architecture

The closed form is derived **end-to-end from `q/k/v/h/g`** by executing the
genuine surface — no producer hypothesis. Every body statement gets a per-statement
`*_op_eval` recipe taking abstract register-readback hypotheses; the body is stepped
with `stepStmts.cons_some (stepStmt_assign_eq_some recipe)`, keeping the `BlockState`
symbolic (readbacks peel via the `setReg`-name-inequality `@[simp]` set, never `whnf`).

```
chunk_gla_simple_output_summary_general                    ← TOP THEOREM (dimension-general)
  ├─ chunk_gla_simple_fwd_surface_toAlgorithm_supported     full surface lowers
  └─ chunk_gla_simple_exec_glaOutput                        exec readback o = glaOutput
       ├─ chunk_gla_simple_body_split                       body = front ++ [store]
       ├─ chunk_gla_simple_front_exec                       front → b_o = glaOutput
       │    ├─ prologue recipes (programId/arange/ge/full)
       │    ├─ chunkGlaLoopBody_steps (single key block: 8 stmts)
       │    │    └─ makeBlockPtr_2d_eval / load_bp_2d_ref / accDot_op_eval / dot2d_elem
       │    └─ post-loop recipes (mulExpCol/mulExpOuter/whereMask/accScale)
       └─ scatter_readback_prop_masked_nd_of_true           injective masked store
```
-/

namespace VeriTile.Bench.TritonBenchG.ChunkGlaSimple

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option maxHeartbeats 4000000

/-- No-mask 2D block-pointer load through a *bound register* `name` holding the
block-pointer tile produced by `makeBlockPtrDynOffsets`. Lane `(i,j)` reads the
genuine memory cell when in-bounds, else `0`. -/
theorem load_bp_2d_ref (rg : RegionName) (s : BlockState) (name : RegName)
    (base rows cols BT BS strideT strideS rowOff colOff : Nat)
    (hreg : s.regs TileDType.blockPtr [BT, BS] name = some
      ⟨fun _ => BlockPtr.mk rg base [rows, cols] [BT, BS] [strideT, strideS]
        [rowOff, colOff]⟩) :
    evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT, BS] name) [0, 1]) MaskOpt.none) s
    = some ⟨fun idx : TileIndex [BT, BS] =>
        if (rowOff + idx.1.val < rows ∧ colOff + idx.2.1.val < cols) then
          some (s.readMem rg (base + (rowOff + idx.1.val) * strideT
            + (colOff + idx.2.1.val) * strideS))
        else some 0⟩ := by
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
      ⟨fun _ => BlockPtr.mk rg base [len] [BT] [stride] [off]⟩) :
    evalOp (Op.load TileDType.real
      (MemAccess.blockPtr (Op.ref TileDType.blockPtr [BT] name) [0]) MaskOpt.none) s
    = some ⟨fun idx : TileIndex [BT] =>
        if (off + idx.1.val < len) then
          some (s.readMem rg (base + (off + idx.1.val) * stride))
        else some 0⟩ := by
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
        (Op.dot (batch := []) (Op.ref .real [M, K] qName) (Op.ref .real [K, N] hName))) s
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          acctile (Tile.dot [] qtile htile)) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] qName) (Op.ref .real [K, N] hName)) s
      = some (Tile.dot [] qtile htile) := by rw [evalOp_dot]; simp [hq, hh]
  have hdot2 : @evalOp TileDType.real [M, N]
      (Op.dot (batch := []) (Op.ref .real [M, K] qName) (Op.ref .real [K, N] hName)) s
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
        (Op.broadcast (Op.const 0.0) [M, M])) s
      = some (Tile.select mtile stile (⟨fun _ => some (0.0 : ℝ)⟩ : Tile .real [M, M])) := by
  have hbc : evalOp (Op.broadcast (Op.const 0.0) [M, M]) s
      = some (⟨fun _ => some (0.0 : ℝ)⟩ : Tile .real [M, M]) := by
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
          (Op.dot (batch := []) (Op.ref .real [M, Tt] sName) (Op.ref .real [Tt, N] vName)))
        (Op.const sc)) s
      = some (Tile.bop NumericDType.real.mul Broadcast.scalarR
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            otile (Tile.dot [] stile vtile))
          (Tile.scalar (some sc : WithBot ℝ))) := by
  have hdot : evalOp (Op.dot (batch := []) (Op.ref .real [M, Tt] sName) (Op.ref .real [Tt, N] vName)) s
      = some (Tile.dot [] stile vtile) := by rw [evalOp_dot]; simp [hs, hv]
  have hdot2 : @evalOp TileDType.real [M, N]
      (Op.dot (batch := []) (Op.ref .real [M, Tt] sName) (Op.ref .real [Tt, N] vName)) s
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

/-! ## Kernel surface (faithful transcription) -/

/-- Faithful transcription of `chunk_gla_simple.py`'s
`chunk_simple_gla_fwd_kernel_o`. -/
def chunk_gla_simple_fwd_surface
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ComputeKernel := triton {
  i_v = tl.program_id(0)
  i_t = tl.program_id(1)
  i_bh = tl.program_id(2)
  o_i = tl.arange(0, $(BT))
  m_s = o_i[:, None] >= o_i[None, :]
  b_o = tl.zeros([$(BT), $(BV)], dtype=tl.float32)
  b_s = tl.zeros([$(BT), $(BT)], dtype=tl.float32)
  for i_k in range($(0), tl.cdiv($(K), $(BK)), $(1)) {
    p_q = tl.make_block_ptr(base=q + i_bh * $(s_k_h),
      shape=($(T), $(K)), strides=($(s_k_t), $(1)),
      offsets=(i_t * $(BT), i_k * $(BK)), block_shape=($(BT), $(BK)), order=(1, 0))
    p_k = tl.make_block_ptr(base=k + i_bh * $(s_k_h),
      shape=($(K), $(T)), strides=($(1), $(s_k_t)),
      offsets=(i_k * $(BK), i_t * $(BT)), block_shape=($(BK), $(BT)), order=(0, 1))
    p_h = tl.make_block_ptr(base=h + i_bh * $(s_h_h) + i_t * $(K) * $(V),
      shape=($(K), $(V)), strides=($(s_h_t), $(1)),
      offsets=(i_k * $(BK), i_v * $(BV)), block_shape=($(BK), $(BV)), order=(1, 0))
    b_q = tl.load(p_q, boundary_check=([0, 1] : List Nat))
    b_k = tl.load(p_k, boundary_check=([0, 1] : List Nat))
    b_h = tl.load(p_h, boundary_check=([0, 1] : List Nat))
    b_o += tl.dot(b_q, b_h, allow_tf32=false)
    b_s += tl.dot(b_q, b_k, allow_tf32=false)
  }
  p_g = tl.make_block_ptr(base=g + i_bh * $(T), shape=($(T)),
    strides=($(1)), offsets=(i_t * $(BT)), block_shape=($(BT)), order=(0))
  b_g = tl.load(p_g, boundary_check=([0] : List Nat))
  b_o = b_o * tl.exp(b_g)[:, None]
  b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
  b_s = tl.where(m_s, b_s, 0.0)
  p_v = tl.make_block_ptr(base=v + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  b_v = tl.load(p_v, boundary_check=([0, 1] : List Nat))
  b_o = (b_o + tl.dot((b_s).to(b_v.dtype), b_v, allow_tf32=false)) * $(scale)
  p_o = tl.make_block_ptr(base=o + i_bh * $(s_v_h),
    shape=($(T), $(V)), strides=($(s_v_t), $(1)),
    offsets=(i_t * $(BT), i_v * $(BV)), block_shape=($(BT), $(BV)), order=(1, 0))
  tl.store(p_o, (b_o).to(p_o.dtype.element_ty), boundary_check=([0, 1] : List Nat))
}

/-- The full simple-GLA forward output surface lowers to the algorithm layer. -/
theorem chunk_gla_simple_fwd_surface_toAlgorithm_supported
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    ∃ alg, (chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h
      s_v_t s_h_h s_h_t scale T K V BT BK BV).toAlgorithm? =
        Except.ok alg := by
  simp [chunk_gla_simple_fwd_surface, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?]

/-! ## Compiled body decomposition -/

/-- The compiled (algorithm-lowered) loop body of the key-block loop: the three
block-pointer constructions, the three loads, and the two matmul accumulations. -/
def chunkGlaLoopBody (q k v h : RegionName) (s_k_h s_k_t s_h_h s_h_t T K V BT BK BV : Nat) :
    List Stmt :=
  [ Stmt.assign .blockPtr [BT, BK] "p_q"
      (Op.makeBlockPtrDynOffsets q
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h)) [T, K]
        [BT, BK] [s_k_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK)]),
    Stmt.assign .blockPtr [BK, BT] "p_k"
      (Op.makeBlockPtrDynOffsets k
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h)) [K, T]
        [BK, BT] [1, s_k_t]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .blockPtr [BK, BV] "p_h"
      (Op.makeBlockPtrDynOffsets h
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
          (Op.mul .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K)) (Op.constNat V)))
        [K, V] [BK, BV] [s_h_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .real [BT, BK] "b_q"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT, BK] "p_q") [0, 1]) .none),
    Stmt.assign .real [BK, BT] "b_k"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BK, BT] "p_k") [0, 1]) .none),
    Stmt.assign .real [BK, BV] "b_h"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BK, BV] "p_h") [0, 1]) .none),
    Stmt.assign .real [BT, BV] "b_o"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BV] "b_o")
        (Op.dot (batch := []) (Op.ref .real [BT, BK] "b_q") (Op.ref .real [BK, BV] "b_h"))),
    Stmt.assign .real [BT, BT] "b_s"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BT] "b_s")
        (Op.dot (batch := []) (Op.ref .real [BT, BK] "b_q") (Op.ref .real [BK, BT] "b_k"))) ]

/-- Loaded-tile abbreviations: the masked block-pointer loads of `q`, `k`, `h`.
Stated exactly as `load_bp_2d_ref` emits them (offset `0 * BK` on the key axis). -/
noncomputable def bqTile (s : BlockState) (q : RegionName) (s_k_h s_k_t T K BT BK : Nat) :
    Tile .real [BT, BK] :=
  ⟨fun idx => if (s.pids 1 * BT + idx.1.val < T ∧ 0 * BK + idx.2.1.val < K) then
      some (s.readMem q (s.pids 2 * s_k_h + (s.pids 1 * BT + idx.1.val) * s_k_t
        + (0 * BK + idx.2.1.val) * 1)) else some 0⟩

noncomputable def bkTile (s : BlockState) (k : RegionName) (s_k_h s_k_t T K BT BK : Nat) :
    Tile .real [BK, BT] :=
  ⟨fun idx => if (0 * BK + idx.1.val < K ∧ s.pids 1 * BT + idx.2.1.val < T) then
      some (s.readMem k (s.pids 2 * s_k_h + (0 * BK + idx.1.val) * 1
        + (s.pids 1 * BT + idx.2.1.val) * s_k_t)) else some 0⟩

noncomputable def bhTile (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BT BK BV : Nat) :
    Tile .real [BK, BV] :=
  ⟨fun idx => if (0 * BK + idx.1.val < K ∧ s.pids 0 * BV + idx.2.1.val < V) then
      some (s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + (0 * BK + idx.1.val) * s_h_t
        + (s.pids 0 * BV + idx.2.1.val) * 1)) else some 0⟩

/-- **Loop-body execution (single key block).** Given the prologue register
readbacks on the loop-entry state `sl` (program ids, `i_k = 0`, the zero-init
`b_o`/`b_s`), the 8-statement loop body steps to a final state whose `b_o`/`b_s`
registers hold the two matmul accumulations of the masked `q`/`k`/`h` loads. -/
theorem chunkGlaLoopBody_steps (q k v h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t T K V BT BK BV : Nat) (sl : BlockState)
    (boTile : Tile .real [BT, BV]) (bsTile : Tile .real [BT, BT])
    (hiv : sl.regs .nat [] "i_v" = some (Tile.scalar (sl.pids 0)))
    (hit : sl.regs .nat [] "i_t" = some (Tile.scalar (sl.pids 1)))
    (hibh : sl.regs .nat [] "i_bh" = some (Tile.scalar (sl.pids 2)))
    (hik : sl.regs .nat [] "i_k" = some (Tile.scalar 0))
    (hbo : sl.regs .real [BT, BV] "b_o" = some boTile)
    (hbs : sl.regs .real [BT, BT] "b_s" = some bsTile) :
    ∃ sF, stepStmts (chunkGlaLoopBody q k v h s_k_h s_k_t s_h_h s_h_t T K V BT BK BV) sl = some sF
      ∧ sF.pids = sl.pids ∧ sF.mem = sl.mem
      ∧ sF.regs .nat [] "i_v" = some (Tile.scalar (sl.pids 0))
      ∧ sF.regs .nat [] "i_t" = some (Tile.scalar (sl.pids 1))
      ∧ sF.regs .nat [] "i_bh" = some (Tile.scalar (sl.pids 2))
      ∧ sF.regs .bool [BT, BT] "m_s" = sl.regs .bool [BT, BT] "m_s"
      ∧ sF.regs .real [BT, BV] "b_o" = some
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            boTile (Tile.dot [] (bqTile sl q s_k_h s_k_t T K BT BK)
              (bhTile sl h s_h_h s_h_t K V BT BK BV)))
      ∧ sF.regs .real [BT, BT] "b_s" = some
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            bsTile (Tile.dot [] (bqTile sl q s_k_h s_k_t T K BT BK)
              (bkTile sl k s_k_h s_k_t T K BT BK))) := by
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
      (sl.pids 1 * BT) (0 * BK)
      (mulConst_eval sl "i_bh" _ _ hibh) (mulConst_eval sl "i_t" _ _ hit)
      (mulConst_eval sl "i_k" _ _ hik)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval k _ _ _ _ [K, T] [BK, BT] [1, s_k_t] (sl.pids 2 * s_k_h)
      (0 * BK) (sl.pids 1 * BT)
      (mulConst_eval _ "i_bh" _ _ (by simp [hibh])) (mulConst_eval _ "i_k" _ _ (by simp [hik]))
      (mulConst_eval _ "i_t" _ _ (by simp [hit]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval h _ _ _ _ [K, V] [BK, BV] [s_h_t, 1]
      (sl.pids 2 * s_h_h + sl.pids 1 * K * V) (0 * BK) (sl.pids 0 * BV)
      (hbaseH _ (by simp [hibh]) (by simp [hit])) (mulConst_eval _ "i_k" _ _ (by simp [hik]))
      (mulConst_eval _ "i_v" _ _ (by simp [hiv]))))]
  -- step loads b_q, b_k, b_h
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref q _ "p_q" (sl.pids 2 * s_k_h) T K BT BK s_k_t 1 (sl.pids 1 * BT) (0 * BK)
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref k _ "p_k" (sl.pids 2 * s_k_h) K T BK BT 1 s_k_t (0 * BK) (sl.pids 1 * BT)
      (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref h _ "p_h" (sl.pids 2 * s_h_h + sl.pids 1 * K * V) K V BK BV s_h_t 1
      (0 * BK) (sl.pids 0 * BV) (by simp)))]
  -- step b_o += dot(b_q, b_h) and b_s += dot(b_q, b_k)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accDot_op_eval _ BT BK BV "b_o" "b_q" "b_h" boTile (bqTile sl q s_k_h s_k_t T K BT BK)
      (bhTile sl h s_h_h s_h_t K V BT BK BV) (by simp [hbo]) (by simp [bqTile]) (by simp [bhTile])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accDot_op_eval _ BT BK BT "b_s" "b_q" "b_k" bsTile (bqTile sl q s_k_h s_k_t T K BT BK)
      (bkTile sl k s_k_h s_k_t T K BT BK) (by simp [hbs]) (by simp [bqTile]) (by simp [bkTile])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp
  · funext region offset; simp
  · simp [hiv]
  · simp [hit]
  · simp [hibh]
  · simp
  · simp
  · simp

/-- Global time (row) index of tile lane `i`: `i_t · BT + i`. -/
def tIndex (s : BlockState) (BT : Nat) (i : Fin BT) : Nat := s.pids 1 * BT + i.val

/-- Global value (column) index of tile lane `p`: `i_v · BV + p`. -/
def vIndex (s : BlockState) (BV : Nat) (p : Fin BV) : Nat := s.pids 0 * BV + p.val

/-- A tile lane is *active* when it maps inside the `T × V` output window. -/
def active (s : BlockState) (T V BT BV : Nat) (idx : TileIndex [BT, BV]) : Prop :=
  tIndex s BT idx.1 < T ∧ vIndex s BV idx.2.1 < V

instance activeDecidable (s : BlockState) (T V BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Decidable (active s T V BT BV idx) := by
  unfold active; infer_instance

/-- `q[i, e]` element: `q` at `i_bh·s_k_h + (i_t·BT + i)·s_k_t + e`. -/
noncomputable def qElem (s : BlockState) (q : RegionName) (s_k_h s_k_t BT : Nat)
    (i : Fin BT) (e : Nat) : ℝ :=
  s.readMem q (s.pids 2 * s_k_h + (s.pids 1 * BT + i.val) * s_k_t + e)

/-- `k[e, j]` element (block ptr layout `(K,T)` strides `(1, s_k_t)`):
`k` at `i_bh·s_k_h + e + (i_t·BT + j)·s_k_t`. -/
noncomputable def kElem (s : BlockState) (k : RegionName) (s_k_h s_k_t BT : Nat)
    (j : Fin BT) (e : Nat) : ℝ :=
  s.readMem k (s.pids 2 * s_k_h + e * 1 + (s.pids 1 * BT + j.val) * s_k_t)

/-- `h[e, p]` element (chunk state, base `h + i_bh·s_h_h + i_t·K·V`):
`h` at `i_bh·s_h_h + i_t·K·V + e·s_h_t + (i_v·BV + p)`. -/
noncomputable def hElem (s : BlockState) (h : RegionName) (s_h_h s_h_t K V BV : Nat)
    (p : Fin BV) (e : Nat) : ℝ :=
  s.readMem h (s.pids 2 * s_h_h + s.pids 1 * K * V + e * s_h_t + (s.pids 0 * BV + p.val) * 1)

/-- `v[j, p]` element: `v` at `i_bh·s_v_h + (i_t·BT + j)·s_v_t + (i_v·BV + p)`. -/
noncomputable def vElem (s : BlockState) (v : RegionName) (s_v_h s_v_t BT BV : Nat)
    (j : Fin BT) (p : Fin BV) : ℝ :=
  s.readMem v (s.pids 2 * s_v_h + (s.pids 1 * BT + j.val) * s_v_t + (s.pids 0 * BV + p.val) * 1)

/-- `g[i]` element (gate): `g` at `i_bh·T + (i_t·BT + i)`. -/
noncomputable def gElem (s : BlockState) (g : RegionName) (T BT : Nat) (i : Fin BT) : ℝ :=
  s.readMem g (s.pids 2 * T + (s.pids 1 * BT + i.val) * 1)

/-- Inter-chunk term lane `(i,p)`: `(Σ_e q[i,e]·h[e,p]) · exp(g_i)`. -/
noncomputable def interTerm (s : BlockState) (q h : RegionName)
    (s_k_h s_k_t s_h_h s_h_t : Nat) (T K V BT BV BK : Nat)
    (g : RegionName) (i : Fin BT) (p : Fin BV) : ℝ :=
  (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
      * hElem s h s_h_h s_h_t K V BV p e.val)
    * Real.exp (gElem s g T BT i)

/-- Masked, decayed score lane `(i,j)`: `if i ≥ j then (Σ_e q·k) · exp(g_i−g_j) else 0`. -/
noncomputable def scoreTerm (s : BlockState) (q k : RegionName)
    (s_k_h s_k_t : Nat) (T BT BK : Nat)
    (g : RegionName) (i j : Fin BT) : ℝ :=
  if (j.val ≤ i.val) then
    (Finset.univ.sum fun e : Fin BK => qElem s q s_k_h s_k_t BT i e.val
        * kElem s k s_k_h s_k_t BT j e.val)
      * Real.exp (gElem s g T BT i - gElem s g T BT j)
  else 0

/-- **Genuine GLA output closed form** for lane `(i, p)`. -/
noncomputable def glaOutput
    (s : BlockState) (q k v h g : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (i : Fin BT) (p : Fin BV) : ℝ :=
  (interTerm s q h s_k_h s_k_t s_h_h s_h_t T K V BT BV BK g i p
    + Finset.univ.sum fun j : Fin BT =>
        scoreTerm s q k s_k_h s_k_t T BT BK g i j
          * vElem s v s_v_h s_v_t BT BV j p) * scale

/-! ## Output store address and the masked store value -/

/-- The output store address for lane `(i, p)`:
`i_bh·s_v_h + (i_t·BT + i)·s_v_t + (i_v·BV + p)`. -/
def outOffset (s : BlockState) (s_v_h s_v_t BT BV : Nat)
    (idx : TileIndex [BT, BV]) : Nat :=
  s.pids 2 * s_v_h + tIndex s BT idx.1 * s_v_t + vIndex s BV idx.2.1 * 1

/-! ## Full-kernel exec → `glaOutput` (recipe-architecture derivation) -/

/-- The post-loop block-pointer base/offset scalar op for `p_g`, `p_v`, `p_o`
reduces (`mulConst_eval`). The `b_v` loaded tile. -/
noncomputable def bvTile (s : BlockState) (v : RegionName) (s_v_h s_v_t T V BT BV : Nat) :
    Tile .real [BT, BV] :=
  ⟨fun idx => if (s.pids 1 * BT + idx.1.val < T ∧ s.pids 0 * BV + idx.2.1.val < V) then
      some (s.readMem v (s.pids 2 * s_v_h + (s.pids 1 * BT + idx.1.val) * s_v_t
        + (s.pids 0 * BV + idx.2.1.val) * 1)) else some 0⟩

/-- The `b_g` loaded tile (1D gate load). -/
noncomputable def bgTile (s : BlockState) (g : RegionName) (T BT : Nat) : Tile .real [BT] :=
  ⟨fun idx => if (s.pids 1 * BT + idx.1.val < T) then
      some (s.readMem g (s.pids 2 * T + (s.pids 1 * BT + idx.1.val) * 1)) else some 0⟩

/-! ## Full-kernel exec produces `glaOutput` -/

/-- The compiled (algorithm-lowered) front of the kernel: everything except the
final block-pointer store. -/
def chunkGlaFront (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV : Nat) (scale : ℝ) : List Stmt :=
  [ Stmt.assign .nat [] "i_v" (Op.programId 0),
    Stmt.assign .nat [] "i_t" (Op.programId 1),
    Stmt.assign .nat [] "i_bh" (Op.programId 2),
    Stmt.assign .nat [BT] "o_i" (Op.arange BT),
    Stmt.assign .bool [BT, BT] "m_s"
      (Op.ge ComparableDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BT] "o_i"))
        (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BT] "o_i"))),
    Stmt.assign .real [BT, BV] "b_o" (Op.full [BT, BV] (Op.const 0)),
    Stmt.assign .real [BT, BT] "b_s" (Op.full [BT, BT] (Op.const 0)),
    Stmt.forRangeDyn "i_k" (Op.constNat 0)
      (Op.div .nat Broadcast.nil
        (Op.sub .nat Broadcast.nil
          (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
        (Op.constNat BK))
      (Op.constNat 1)
      (chunkGlaLoopBody q k v h s_k_h s_k_t s_h_h s_h_t T K V BT BK BV),
    Stmt.assign .blockPtr [BT] "p_g"
      (Op.makeBlockPtrDynOffsets g
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat T)) [T] [BT] [1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]),
    Stmt.assign .real [BT] "b_g"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT] "p_g") [0]) .none),
    Stmt.assign .real [BT, BV] "b_o"
      (Op.mul .real (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ref .real [BT, BV] "b_o")
        (Op.expandDim ⟨1, by simp⟩ (Op.exp (Op.ref .real [BT] "b_g")))),
    Stmt.assign .real [BT, BT] "b_s"
      (Op.mul .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [BT, BT] "b_s")
        (Op.exp (Op.sub .real (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref .real [BT] "b_g"))
          (Op.expandDim ⟨0, by simp⟩ (Op.ref .real [BT] "b_g"))))),
    Stmt.assign .real [BT, BT] "b_s"
      (Op.where (Op.ref .bool [BT, BT] "m_s") (Op.ref .real [BT, BT] "b_s")
        (Op.broadcast (Op.const 0.0) [BT, BT])),
    Stmt.assign .blockPtr [BT, BV] "p_v"
      (Op.makeBlockPtrDynOffsets v
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h)) [T, V]
        [BT, BV] [s_v_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]),
    Stmt.assign .real [BT, BV] "b_v"
      (Op.load .real (.blockPtr (Op.ref .blockPtr [BT, BV] "p_v") [0, 1]) .none),
    Stmt.assign .real [BT, BV] "b_o"
      (Op.mul .real Broadcast.scalarR
        (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          (Op.ref .real [BT, BV] "b_o")
          (Op.dot (batch := []) (Op.ref .real [BT, BT] "b_s") (Op.ref .real [BT, BV] "b_v")))
        (Op.const scale)),
    Stmt.assign .blockPtr [BT, BV] "p_o"
      (Op.makeBlockPtrDynOffsets o
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h)) [T, V]
        [BT, BV] [s_v_t, 1]
        [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT),
          Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV)]) ]

/-- The final block-pointer store statement. -/
def chunkGlaStoreStmt (o : RegionName) (BT BV : Nat) : Stmt :=
  Stmt.store .real [BT, BV] (.blockPtr (Op.ref .blockPtr [BT, BV] "p_o") [0, 1])
    (Op.ref .real [BT, BV] "b_o") .none

/-- The compiled body splits as `front ++ [store]`. By `rfl`. -/
theorem chunk_gla_simple_body_split
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) :
    (chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale T K V BT BK BV).toAlgKernel.body
      = chunkGlaFront q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t T K V BT BK BV scale
          ++ [chunkGlaStoreStmt o BT BV] := by
  rfl

set_option maxHeartbeats 2000000 in
/-- **Front execution.** Stepping the prologue (program ids, `o_i`, `m_s`,
zero-init `b_o`/`b_s`), the single key-block loop, and the post-loop gate decay /
causal mask / `b_s·v` accumulation, the surface front reaches a state `sF` whose
`p_o` block-pointer register and final `b_o` register are the genuine kernel
values; `b_o` equals `glaOutput` at every active lane. -/
theorem chunk_gla_simple_front_exec
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (s : BlockState)
    (hKBK : K = BK) (hBK : 0 < BK) (hBT : 0 < BT)
    (hundef : ∀ rg off, s.undef rg off = 0) :
    ∃ sF, stepStmts (chunkGlaFront q k v h g o s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t T K V BT BK BV scale) s = some sF
      ∧ sF.pids = s.pids ∧ (∀ rg off, sF.mem rg off = s.mem rg off)
      ∧ sF.regs .blockPtr [BT, BV] "p_o" = some
          (⟨fun _ => BlockPtr.mk o (s.pids 2 * s_v_h) [T, V] [BT, BV] [s_v_t, 1]
            [s.pids 1 * BT, s.pids 0 * BV]⟩ : Tile .blockPtr [BT, BV])
      ∧ ∃ boF : Tile .real [BT, BV],
          sF.regs .real [BT, BV] "b_o" = some boF
          ∧ ∀ (i : Fin BT) (p : Fin BV),
              active s T V BT BV (i, p, PUnit.unit) →
              boF.data (i, p, PUnit.unit)
                = some (glaOutput s q k v h g s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
                    scale T K V BT BK BV i p) := by
  have hax1 : 1 < [BT].length.succ := by simp
  have hax0 : 0 < [BT].length.succ := by simp
  -- zero-init tiles
  set boZero : Tile .real [BT, BV] := ⟨fun _ => some (0 : ℝ)⟩ with hboz
  set bsZero : Tile .real [BT, BT] := ⟨fun _ => some (0 : ℝ)⟩ with hbsz
  set oiTile : Tile .nat [BT] := Tile.vec (fun i : Fin BT => i.val) with hoi
  set msTile : Tile .bool [BT, BT] := Tile.cop ComparableDType.nat.ge
    (Broadcast.consR (Broadcast.consL Broadcast.nil))
    (Tile.expandDim ⟨1, hax1⟩ oiTile) (Tile.expandDim ⟨0, hax0⟩ oiTile) with hms
  -- step the prologue
  unfold chunkGlaFront
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.programId 0) s
    = some (Tile.scalar (s.pids 0)) from by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.programId 1) _
    = some (Tile.scalar (s.pids 1)) from by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.programId 2) _
    = some (Tile.scalar (s.pids 2)) from by simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.arange BT) _
    = some oiTile from by simp [hoi]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (msGe_op_eval _ BT hax1 hax0 "o_i" oiTile (by simp [hoi])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.full [BT, BV] (Op.const 0)) _
    = some boZero from by simp [hboz]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.full [BT, BT] (Op.const 0)) _
    = some bsZero from by simp [hbsz]))]
  -- the single-iteration key-block loop
  have hcdiv : (K + BK - 1) / BK = 1 := by
    subst hKBK
    have he : K + K - 1 = K + (K - 1) := by omega
    rw [he, Nat.add_div_left _ hBK, Nat.div_eq_of_lt (by omega)]
  set s7 : BlockState :=
    (((((((s.setReg "i_v" .nat [] (Tile.scalar (s.pids 0))).setReg "i_t" .nat []
      (Tile.scalar (s.pids 1))).setReg "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg
      "o_i" .nat [BT] oiTile).setReg "m_s" .bool [BT, BT] msTile).setReg "b_o" .real [BT, BV]
      boZero).setReg "b_s" .real [BT, BT] bsZero) with hs7
  obtain ⟨sBody, hbody, hbpids, hbmem, hbiv, hbit, hbibh, hbms, hbbo, hbbs⟩ :=
    chunkGlaLoopBody_steps q k v h s_k_h s_k_t s_h_h s_h_t T K V BT BK BV
      (s7.setReg "i_k" .nat [] (Tile.scalar 0)) boZero bsZero
      (by simp [hs7]) (by simp [hs7]) (by simp [hs7]) (by simp) (by simp [hs7]) (by simp [hs7])
  have hstopE : evalOp (Op.div .nat Broadcast.nil
      (Op.sub .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
      (Op.constNat BK)) s7
      = some (Tile.scalar 1) := by
    simp only [evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat, Option.bind_eq_bind,
      Option.bind_some]
    refine congrArg some ?_
    simp only [Tile.bop, NumericDType.div, NumericDType.sub, NumericDType.add, Tile.scalar]
    refine congrArg _ ?_
    funext _; exact hcdiv
  rw [stepStmts.cons_some (forRangeDyn_single_step
    (start := 0) (stop := 1) (step := 1)
    (by simp) hstopE (by simp)
    (by norm_num) (by norm_num) (by norm_num) hbody)]
  -- post-loop register values
  set boLoop : Tile .real [BT, BV] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      boZero (Tile.dot [] (bqTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) q s_k_h s_k_t T K BT BK)
        (bhTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) h s_h_h s_h_t K V BT BK BV)) with hboLoop
  set bsLoop : Tile .real [BT, BT] :=
    Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      bsZero (Tile.dot [] (bqTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) q s_k_h s_k_t T K BT BK)
        (bkTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) k s_k_h s_k_t T K BT BK)) with hbsLoop
  have hpidsB : sBody.pids = s.pids := by rw [hbpids]; simp [hs7]
  have hrmemB : ∀ (R : RegionName) (oo : Nat), sBody.readMem R oo = s.readMem R oo := by
    intro R oo; unfold BlockState.readMem; rw [hbmem]; simp [hs7]
  have hmsB : sBody.regs .bool [BT, BT] "m_s" = some msTile := by rw [hbms]; simp [hs7]
  have hbgT : sBody.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [hbibh]; simp [hs7]
  have hbtT : sBody.regs .nat [] "i_t" = some (Tile.scalar (s.pids 1)) := by rw [hbit]; simp [hs7]
  have hbvT : sBody.regs .nat [] "i_v" = some (Tile.scalar (s.pids 0)) := by rw [hbiv]; simp [hs7]
  -- step p_g, b_g
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_1d_eval g sBody _ _ [T] [BT] [1] (s.pids 2 * T) (s.pids 1 * BT)
      (mulConst_eval _ "i_bh" _ _ hbgT) (mulConst_eval _ "i_t" _ _ hbtT)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_1d_ref g _ "p_g" (s.pids 2 * T) T BT 1 (s.pids 1 * BT) (by simp)))]
  -- b_o *= exp(b_g)[:, None]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mulExpCol_op_eval _ BT BV hax1 "b_o" "b_g" boLoop (bgTile s g T BT)
      (by simp [hbbo, hboLoop]) (by simp only [BlockState.setReg_same]; ext idx; simp [bgTile, hrmemB])))]
  -- b_s *= exp(b_g[:, None] − b_g[None, :])
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mulExpOuter_op_eval _ BT hax1 hax0 "b_s" "b_g" bsLoop (bgTile s g T BT)
      (by simp [hbbs, hbsLoop]) (by ext idx; simp [bgTile, hrmemB])))]
  -- b_s = where(m_s, b_s, 0)
  set bsExp : Tile .real [BT, BT] :=
    Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      bsLoop (Tile.uop WithBot.realExp
        (Tile.bop NumericDType.real.sub (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Tile.expandDim ⟨1, hax1⟩ (bgTile s g T BT))
          (Tile.expandDim ⟨0, hax0⟩ (bgTile s g T BT)))) with hbsExp
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (whereMask_op_eval _ BT "m_s" "b_s" msTile bsExp (by simp [hmsB]) (by simp [hbsExp])))]
  -- p_v, b_v
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval v _ _ _ _ [T, V] [BT, BV] [s_v_t, 1] (s.pids 2 * s_v_h)
      (s.pids 1 * BT) (s.pids 0 * BV)
      (mulConst_eval _ "i_bh" _ _ (by simp [hbgT])) (mulConst_eval _ "i_t" _ _ (by simp [hbtT]))
      (mulConst_eval _ "i_v" _ _ (by simp [hbvT]))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (load_bp_2d_ref v _ "p_v" (s.pids 2 * s_v_h) T V BT BV s_v_t 1 (s.pids 1 * BT)
      (s.pids 0 * BV) (by simp)))]
  -- b_o = (b_o + dot(b_s, b_v)) * scale
  set boExp : Tile .real [BT, BV] :=
    Tile.bop NumericDType.real.mul (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      boLoop (Tile.expandDim ⟨1, hax1⟩ (Tile.uop WithBot.realExp (bgTile s g T BT))) with hboExp
  set bsMasked : Tile .real [BT, BT] :=
    Tile.select msTile bsExp (⟨fun _ => some (0.0 : ℝ)⟩ : Tile .real [BT, BT]) with hbsMasked
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (accScale_op_eval _ BT BT BV scale "b_o" "b_s" "b_v" boExp bsMasked (bvTile s v s_v_h s_v_t T V BT BV)
      (by simp [hboExp]) (by simp [hbsMasked]) (by ext idx; simp [bvTile, hrmemB])))]
  -- p_o
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (makeBlockPtr_2d_eval o _ _ _ _ [T, V] [BT, BV] [s_v_t, 1] (s.pids 2 * s_v_h)
      (s.pids 1 * BT) (s.pids 0 * BV)
      (mulConst_eval _ "i_bh" _ _ (by simp [hbgT])) (mulConst_eval _ "i_t" _ _ (by simp [hbtT]))
      (mulConst_eval _ "i_v" _ _ (by simp [hbvT]))))]
  rw [stepStmts.nil]
  -- the final accumulated b_o tile
  set boFin : Tile .real [BT, BV] :=
    Tile.bop NumericDType.real.mul Broadcast.scalarR
      (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        boExp (Tile.dot [] bsMasked (bvTile s v s_v_h s_v_t T V BT BV)))
      (Tile.scalar (some scale : WithBot ℝ)) with hboFin
  -- the loop-entry state shares pids / readMem with `s`
  have hpidsE : (s7.setReg "i_k" .nat [] (Tile.scalar 0)).pids = s.pids := by simp [hs7]
  have hrmemE : ∀ (R : RegionName) (oo : Nat),
      (s7.setReg "i_k" .nat [] (Tile.scalar 0)).readMem R oo = s.readMem R oo := by
    intro R oo; unfold BlockState.readMem; simp [hs7]
  -- element equalities (active lanes): the masked loads equal the genuine reads
  have hbqE : ∀ (i : Fin BT) (e : Fin BK), s.pids 1 * BT + i.val < T →
      (bqTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) q s_k_h s_k_t T K BT BK).data (i, e, PUnit.unit)
        = some (qElem s q s_k_h s_k_t BT i e.val) := by
    intro i e hi
    have heK : 0 * BK + e.val < K := by subst hKBK; have := e.isLt; omega
    show (if _ then _ else _) = _
    simp only [bqTile, hpidsE, hrmemE]
    rw [if_pos (And.intro hi heK), qElem,
      show (0 * BK + e.val) * 1 = e.val by simp]
  have hbhE : ∀ (p : Fin BV) (e : Fin BK), s.pids 0 * BV + p.val < V →
      (bhTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) h s_h_h s_h_t K V BT BK BV).data (e, p, PUnit.unit)
        = some (hElem s h s_h_h s_h_t K V BV p e.val) := by
    intro p e hp
    have heK : 0 * BK + e.val < K := by subst hKBK; have := e.isLt; omega
    show (if _ then _ else _) = _
    simp only [bhTile, hpidsE, hrmemE]
    rw [if_pos (And.intro heK hp), hElem,
      show (0 * BK + e.val) * s_h_t = e.val * s_h_t by simp]
  have hbkE : ∀ (j : Fin BT) (e : Fin BK), s.pids 1 * BT + j.val < T →
      (bkTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) k s_k_h s_k_t T K BT BK).data (e, j, PUnit.unit)
        = some (kElem s k s_k_h s_k_t BT j e.val) := by
    intro j e hj
    have heK : 0 * BK + e.val < K := by subst hKBK; have := e.isLt; omega
    show (if _ then _ else _) = _
    simp only [bkTile, hpidsE, hrmemE]
    rw [if_pos (And.intro heK hj), kElem,
      show (0 * BK + e.val) * 1 = e.val * 1 by simp]
  have hbvE : ∀ (j : Fin BT) (p : Fin BV), s.pids 1 * BT + j.val < T → s.pids 0 * BV + p.val < V →
      (bvTile s v s_v_h s_v_t T V BT BV).data (j, p, PUnit.unit)
        = some (vElem s v s_v_h s_v_t BT BV j p) := by
    intro j p hj hp; show (if _ then _ else _) = _
    rw [if_pos (And.intro hj hp), vElem]
  have hbgE : ∀ (i : Fin BT), s.pids 1 * BT + i.val < T →
      (bgTile s g T BT).data (i, PUnit.unit) = some (gElem s g T BT i) := by
    intro i hi; show (if _ then _ else _) = _
    rw [if_pos hi, gElem]
  -- the matmul `q·h` and the interTerm
  have hdotQH : ∀ (i : Fin BT) (p : Fin BV), s.pids 1 * BT + i.val < T → s.pids 0 * BV + p.val < V →
      (Tile.dot [] (bqTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) q s_k_h s_k_t T K BT BK)
          (bhTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) h s_h_h s_h_t K V BT BK BV)).data
          (i, p, PUnit.unit)
        = some (Finset.univ.sum fun e : Fin BK =>
            qElem s q s_k_h s_k_t BT i e.val * hElem s h s_h_h s_h_t K V BV p e.val) :=
    fun i p hi hp => dot2d_elem _ _ i p _ _ (fun e => hbqE i e hi) (fun e => hbhE p e hp)
  -- the matmul `q·k` row
  have hdotQK : ∀ (i j : Fin BT), s.pids 1 * BT + i.val < T → s.pids 1 * BT + j.val < T →
      (Tile.dot [] (bqTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) q s_k_h s_k_t T K BT BK)
          (bkTile (s7.setReg "i_k" .nat [] (Tile.scalar 0)) k s_k_h s_k_t T K BT BK)).data
          (i, j, PUnit.unit)
        = some (Finset.univ.sum fun e : Fin BK =>
            qElem s q s_k_h s_k_t BT i e.val * kElem s k s_k_h s_k_t BT j e.val) :=
    fun i j hi hj => dot2d_elem _ _ i j _ _ (fun e => hbqE i e hi) (fun e => hbkE j e hj)
  -- msTile(i,j) = decide (j ≤ i)
  have hmsData : ∀ (i j : Fin BT), msTile.data (i, j, PUnit.unit) = decide (j.val ≤ i.val) := by
    intro i j
    simp only [hms, Tile.cop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      Tile.expandDim_data, TileShape.dropInsertedIndex, hoi, Tile.vec, ComparableDType.ge,
      ge_iff_le, decide_eq_decide]
  -- bsMasked(i,j) = scoreTerm(i,j) for active row i (all columns j)
  have hbsM : ∀ (i j : Fin BT), s.pids 1 * BT + i.val < T →
      bsMasked.data (i, j, PUnit.unit) = some (scoreTerm s q k s_k_h s_k_t T BT BK g i j) := by
    intro i j hi
    simp only [hbsMasked, Tile.select_data, hmsData]
    by_cases hji : j.val ≤ i.val
    · -- active column j (since j ≤ i and i active)
      have hj : s.pids 1 * BT + j.val < T := by omega
      simp only [decide_eq_true hji, if_pos, scoreTerm, if_pos hji]
      simp only [hbsExp, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Tile.uop_data, Tile.expandDim_data, TileShape.dropInsertedIndex, hbsLoop,
        hbsz, hdotQK i j hi hj, hbgE i hi, hbgE j hj]
      simp only [NumericDType.mul, NumericDType.add, NumericDType.sub, WithBot.realMul,
        WithBot.realAdd, WithBot.realSub, WithBot.realExp, Option.map₂, Option.bind, Option.map]
      rw [zero_add]
    · rw [if_neg (by simp [hji]), scoreTerm, if_neg hji]; norm_num
  refine ⟨_, rfl, ?_, ?_, ?_, boFin, ?_, ?_⟩
  · simp [hpidsB]
  · intro rg off; simp [hbmem, hs7]
  · simp
  · simp
  · intro i p hActive
    obtain ⟨hi, hp⟩ := hActive
    simp only [active, tIndex, vIndex] at hi hp
    -- boExp(i,p) = interTerm
    have hboExpE : boExp.data (i, p, PUnit.unit)
        = some (interTerm s q h s_k_h s_k_t s_h_h s_h_t T K V BT BV BK g i p) := by
      simp only [hboExp, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
        Tile.expandDim_data, TileShape.dropInsertedIndex, Tile.uop_data, hboLoop, hboz,
        hdotQH i p hi hp, hbgE i hi, interTerm]
      simp only [NumericDType.mul, NumericDType.add, WithBot.realMul, WithBot.realAdd,
        WithBot.realExp, Option.map₂, Option.bind, Option.map]
      rw [zero_add]
    -- the intra-chunk matmul `bsMasked · bvTile`
    set fb : Fin BT → ℝ := fun j =>
      if (s.pids 1 * BT + j.val < T ∧ s.pids 0 * BV + p.val < V) then
        vElem s v s_v_h s_v_t BT BV j p else 0 with hfb
    have hbvData : ∀ j : Fin BT, (bvTile s v s_v_h s_v_t T V BT BV).data (j, p, PUnit.unit)
        = some (fb j) := by
      intro j; by_cases hj : s.pids 1 * BT + j.val < T ∧ s.pids 0 * BV + p.val < V
      · simp only [hfb, if_pos hj]; exact hbvE j p hj.1 hj.2
      · simp only [hfb, if_neg hj]; simp only [bvTile, if_neg hj]
    have hdotSV : (Tile.dot [] bsMasked (bvTile s v s_v_h s_v_t T V BT BV)).data (i, p, PUnit.unit)
        = some (Finset.univ.sum fun j : Fin BT =>
            scoreTerm s q k s_k_h s_k_t T BT BK g i j * vElem s v s_v_h s_v_t BT BV j p) := by
      rw [dot2d_elem _ _ i p (fun j => scoreTerm s q k s_k_h s_k_t T BT BK g i j) fb
        (fun j => hbsM i j hi) hbvData]
      refine congrArg some (Finset.sum_congr rfl (fun j _ => ?_))
      by_cases hsc : j.val ≤ i.val
      · have hj : s.pids 1 * BT + j.val < T := by omega
        simp only [hfb, if_pos (And.intro hj hp)]
      · have : scoreTerm s q k s_k_h s_k_t T BT BK g i j = 0 := by
          simp only [scoreTerm, if_neg hsc]
        rw [this, zero_mul, zero_mul]
    -- assemble
    simp only [hboFin, Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex,
      hboExpE, hdotSV, glaOutput, Tile.scalar]
    simp only [NumericDType.mul, NumericDType.add, WithBot.realMul, WithBot.realAdd,
      Option.map₂, Option.bind, Option.map]

set_option maxHeartbeats 2000000 in
/-- **Full-kernel exec realizes `glaOutput`.** Executing the genuine GLA forward
surface, every active output lane of `o` equals `glaOutput` — the closed form,
NOT the kernel's own emitted value. The masked block-pointer store is an injective
scatter (`scatter_readback_prop_masked_nd_of_true`) over the active region. -/
theorem chunk_gla_simple_exec_glaOutput
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (s : BlockState)
    (hKBK : K = BK) (hBK : 0 < BK) (hBT : 0 < BT)
    (hundef : ∀ rg off, s.undef rg off = 0)
    (hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx))
    (idx : TileIndex [BT, BV]) (hActive : active s T V BT BV idx) :
    (exec (chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale T K V BT BK BV) s).map
        (·.readMem o (outOffset s s_v_h s_v_t BT BV idx))
      = some (glaOutput s q k v h g s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
          scale T K V BT BK BV idx.1 idx.2.1) := by
  obtain ⟨sF, hfront, hpids, hmemF, hpo, boF, hboF, hboData⟩ :=
    chunk_gla_simple_front_exec q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
      scale T K V BT BK BV s hKBK hBK hBT hundef
  show (stepStmts (chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
      s_h_h s_h_t scale T K V BT BK BV).toAlgKernel.body s).map _ = _
  rw [chunk_gla_simple_body_split, stepStmts.append_some hfront]
  -- step the store
  have hstore : stepStmts [chunkGlaStoreStmt o BT BV] sF
      = some ((TileShape.allIndices [BT, BV]).foldl
          (fun acc i =>
            if (s.pids 1 * BT + i.1.val < T ∧ s.pids 0 * BV + i.2.1.val < V) then
              acc.writeMem o (s.pids 2 * s_v_h + (s.pids 1 * BT + i.1.val) * s_v_t
                + (s.pids 0 * BV + i.2.1.val) * 1)
                (FloatDType.real.storeValue (boF.data i))
            else acc) sF) := by
    rw [stepStmts.cons_some (st := chunkGlaStoreStmt o BT BV) (s' := _) ?_, stepStmts.nil]
    show stepStmt (chunkGlaStoreStmt o BT BV) sF = some _
    unfold chunkGlaStoreStmt stepStmt
    simp only [evalOp_ref, hboF, hpo, Option.bind, Option.map]
    refine congrArg some ?_
    congr 1
    funext acc i
    simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
      Bool.true_and, BlockState.writeMemTyped_real, decide_eq_true_eq]
  rw [hstore]
  simp only [Option.map_some]
  -- readback at the active lane
  set offsetFn : TileIndex [BT, BV] → Nat :=
    fun i => s.pids 2 * s_v_h + (s.pids 1 * BT + i.1.val) * s_v_t
      + (s.pids 0 * BV + i.2.1.val) * 1 with hoffdef
  set P : TileIndex [BT, BV] → Prop :=
    fun i => s.pids 1 * BT + i.1.val < T ∧ s.pids 0 * BV + i.2.1.val < V with hPdef
  have hoffInj : Function.Injective offsetFn := by
    simpa [offsetFn, outOffset, tIndex, vIndex] using hInj
  have hPidx : P idx := by simpa [P, active, tIndex, vIndex] using hActive
  have hcollision : ∀ kk, P kk → offsetFn kk = offsetFn idx → kk = idx :=
    fun kk _ heq => hoffInj heq
  rw [show outOffset s s_v_h s_v_t BT BV idx = offsetFn idx from by
    simp [outOffset, offsetFn, tIndex, vIndex]]
  rw [BlockState.scatter_readback_prop_masked_nd_of_true (region := o) sF offsetFn
    (fun i => FloatDType.real.storeValue (boF.data i)) P idx hPidx hcollision]
  rw [hboData idx.1 idx.2.1 (by simpa [active, tIndex, vIndex] using hActive)]
  rfl


/-! ## Public dimension-general output summary -/

/-- **Public dimension-general output summary.** Over *symbolic* strides, scale,
and dimensions `T K V BT BK BV` (with `K = BK`, `BK,BT > 0`, the `undef`-free and
output-offset-injective side conditions), the full simple-GLA forward surface

* lowers to the algorithm layer, and
* executes so that every active output lane of `o` equals the genuine GLA closed
  form `glaOutput` (read off the kernel's actual store; the `glaOutput` spec reads
  the *input* memory `q/k/v/h/g`, NOT a self-referential exec read-back).

The masked write map matches the kernel's store mask (`active`), and every active
output lane of `o` equals the genuine GLA closed form `glaOutput` (read off the
kernel's actual store; the `glaOutput` spec reads the *input* memory `q/k/v/h/g`,
NOT a self-referential exec read-back). Discharged via
`chunk_gla_simple_fwd_surface_toAlgorithm_supported` (lowering) and
`chunk_gla_simple_exec_glaOutput` (per-lane readback). The pinned per-Python-case
summaries are specializations of this theorem at their literal dimensions. -/
specification chunk_gla_simple_output_summary_general
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (s : BlockState)
    (hKBK : K = BK) (hBK : 0 < BK) (hBT : 0 < BT)
    (hundef : ∀ rg off, s.undef rg off = 0)
    (hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx)) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale T K V BT BK BV)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BT, BV] => active s T V BT BV idx)
        (fun idx => (o, outOffset s s_v_h s_v_t BT BV idx)))
      (expected := fun idx : TileIndex [BT, BV] =>
        glaOutput s q k v h g s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
          scale T K V BT BK BV idx.1 idx.2.1) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [chunk_gla_simple_fwd_surface, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst s0
  intro idx hActive
  simp only [ComputeCorrect.OutputReadable.read_real]
  have h := chunk_gla_simple_exec_glaOutput q k v h g o s_k_h s_k_t s_v_h s_v_t
    s_h_h s_h_t scale T K V BT BK BV s hKBK hBK hBT hundef hInj idx hActive
  rw [hExec] at h
  simpa using Option.some.inj h

/-! # ══════════ The `⊨[R]` io face ══════════

The `StreamMasked3DKernelIO₅` face of `chunk_simple_gla_fwd_kernel_o`.
Everything below is purely additive; the exact surfaces above are untouched.

**Genre fit.** The kernel streams three in-loop block-pointer channels
(`q`, `k`, `h`, advanced by the `i_k` key-block counter) and reads two more
after the loop (`g`, `v`), then performs one terminal masked store to `o` —
five input channels of radically different widths (`BT·BK`, `BK·BT`,
`BK·BV`, `BT`, `BT·BV`) feeding one `BT·BV` output, which is exactly the
₅ contract. In the file's declared `K = BK` regime the key-block loop is a
single step, so the step count is `T = 1` and all five channels degenerate
to static (step-independent) streams — the legal degeneration the skin's
section note describes.

**Terminal store dtype (`outDType := .real`).** The surface writes
`tl.store(p_o, b_o.to(p_o.dtype.element_ty), …)`, but that cast **erases at
translation**: the lowered statement is `chunkGlaStoreStmt`, a bare
`Stmt.store .real [BT, BV] …` carrying `Op.ref .real [BT, BV] "b_o"` with no
`Op.castFloat` node — and `chunk_gla_simple_body_split` matches it to the
surface by `rfl`. The same erasure removes the in-loop
`b_s.to(b_v.dtype)` (the final dot's left operand is a bare
`Op.ref .real [BT, BT] "b_s"`). So the whole body is cast-free, `outDType`
is the exact `.real` grid, and **no `R.round .fp16 = id` hypothesis is
needed or admissible** — an `.fp16` readback claim would be unprovable
here. -/

section IOFace

open scoped VeriTile.Triton.StreamMasked3DKernelIO₅

/-! ## Cast-free collapse

No `Op.castFloat` node survives translation and the terminal store is
`.real`-typed, so `stepStmtsR R` collapses onto the exact stepper for every
rounding model `R` and the whole exact stack above is reused verbatim under
`execR R`. -/

/-- `.real` stores never round: `writeMemTypedR` delegates to the exact write. -/
private theorem cgsIO_wmtR_real (R : RoundingModel) (s : BlockState)
    (region : RegionName) (o : Nat) (x : TileCarrier .real) :
    s.writeMemTypedR R .real region o x = s.writeMemTyped .real region o x := rfl

set_option maxHeartbeats 4000000 in
/-- The key-block loop body collapses onto the exact stepper (three
`makeBlockPtrDynOffsets` assigns, three block-pointer loads, two dots — all
cast-free; the `mapM` over the literal offset list is unfolded so the
per-offset `evalOpR` reductions can fire). -/
private theorem cgsIO_loopBody_castFree (R : RoundingModel)
    (q k v h : RegionName) (s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV : Nat)
    (t : BlockState) :
    stepStmtsR R (chunkGlaLoopBody q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV) t
      = stepStmts (chunkGlaLoopBody q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV) t := by
  simp only [chunkGlaLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, List.mapM, List.mapM.loop, bind, Option.bind]
  rfl

set_option maxHeartbeats 4000000 in
/-- The whole front (prologue, the `forRangeDyn` key-block loop, the
post-loop gate decay / causal mask / `b_s·v` accumulation) collapses onto the
exact stepper: the loop's `.nat` bounds are cast-free and its body collapses
by `cgsIO_loopBody_castFree` (transported through
`stepForRangeAuxR_castFree`). -/
private theorem cgsIO_front_castFree (R : RoundingModel)
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t Tt K V BT BK BV : Nat) (scale : ℝ)
    (t : BlockState) :
    stepStmtsR R (chunkGlaFront q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
        Tt K V BT BK BV scale) t
      = stepStmts (chunkGlaFront q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
        Tt K V BT BK BV scale) t := by
  simp only [chunkGlaFront, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, List.mapM, List.mapM.loop, bind, Option.bind,
    stepForRangeAuxR_castFree R _
      (cgsIO_loopBody_castFree R q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV) "i_k"]
  rfl

/-- The terminal block-pointer store collapses: it is `.real`-typed, so
`writeMemTypedR R .real` **is** the exact typed write. -/
private theorem cgsIO_store_castFree (R : RoundingModel) (o : RegionName) (BT BV : Nat)
    (t : BlockState) :
    stepStmtsR R [chunkGlaStoreStmt o BT BV] t = stepStmts [chunkGlaStoreStmt o BT BV] t := by
  simp only [chunkGlaStoreStmt, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def, bind, Option.bind, cgsIO_wmtR_real R]
  rfl

/-- `stepStmts` splits over list concatenation (the exact-side mirror of
`stepStmtsR_append`). -/
private theorem cgsIO_stepStmts_append (xs ys : List Stmt) (s : BlockState) :
    stepStmts (xs ++ ys) s = (stepStmts xs s).bind (fun s' => stepStmts ys s') := by
  induction xs generalizing s with
  | nil => simp [stepStmts]
  | cons st rest ih =>
      simp only [List.cons_append, stepStmts]
      cases stepStmt st s <;> simp [ih]

set_option maxHeartbeats 4000000 in
/-- **The whole kernel is cast-free**: `execR R` *is* `exec`, for every
rounding model `R`. -/
private theorem cgsIO_execR_eq (R : RoundingModel)
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (Tt K V BT BK BV : Nat) (s : BlockState) :
    execR R (chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale Tt K V BT BK BV).toAlgKernel s
      = exec (chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale Tt K V BT BK BV).toAlgKernel s := by
  unfold execR exec
  rw [chunk_gla_simple_body_split, stepStmtsR_append, cgsIO_stepStmts_append,
    cgsIO_front_castFree R q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t Tt K V BT BK BV scale]
  cases hf : stepStmts (chunkGlaFront q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
      Tt K V BT BK BV scale) s with
  | none => rfl
  | some sF => exact cgsIO_store_castFree R o BT BV sF

set_option maxHeartbeats 4000000 in
/-- The `chunk_gla_simple` surface sits inside the flat-memory bridge's covered
fragment (no `ptrSub`, no atomics; the block-pointer ops are structurally
covered). -/
private theorem cgsIO_flattenOk (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (Tt K V BT BK BV : Nat) :
    ((chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
      s_h_h s_h_t scale Tt K V BT BK BV).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [chunk_gla_simple_body_split]
  simp [chunkGlaFront, chunkGlaLoopBody, chunkGlaStoreStmt, StmtList.FlattenOk,
    Stmt.FlattenOk, Op.FlattenOk]
  simp [Op.FlattenOk.eq_def]

/-! ## IO signature, streamed tiles, and the closed-form spec `f`

Window transcription (`Tt` = the kernel's sequence length `T`; the skin's own
`T` field is the **step count**, which is `1` here — see the section note):

* `read1` (`q`, block ptr `(Tt, K)` strides `(s_k_t, 1)`, offsets
  `(p₁·BT, 0)`): lane `j = (i, e)` row-major over `[BT, BK]` reads
  `p₂·s_k_h + (p₁·BT + i)·s_k_t + e`, masked by the `boundary_check=(0,1)`
  window `p₁·BT + i < Tt ∧ e < K`.
* `read2` (`k`, block ptr `(K, Tt)` strides `(1, s_k_t)`, offsets
  `(0, p₁·BT)`): lane `j = (e, jj)` over `[BK, BT]` reads
  `p₂·s_k_h + e·1 + (p₁·BT + jj)·s_k_t`, masked `e < K ∧ p₁·BT + jj < Tt`.
* `read3` (`h`, chunk state, base `p₂·s_h_h + p₁·K·V`, strides `(s_h_t, 1)`,
  offsets `(0, p₀·BV)`): lane `j = (e, p)` over `[BK, BV]` reads
  `p₂·s_h_h + p₁·K·V + e·s_h_t + (p₀·BV + p)·1`, masked
  `e < K ∧ p₀·BV + p < V`.
* `read4` (`g`, the post-loop `BT`-vector gate): lane `i` reads
  `p₂·Tt + (p₁·BT + i)·1`, masked `p₁·BT + i < Tt`.
* `read5` (`v`, the post-loop `[BT, BV]` value tile): lane `j = (jj, p)`
  reads `p₂·s_v_h + (p₁·BT + jj)·s_v_t + (p₀·BV + p)·1`, masked
  `p₁·BT + jj < Tt ∧ p₀·BV + p < V`.
* `write` (`o`): lane `j = (i, p)` writes
  `p₂·s_v_h + (p₁·BT + i)·s_v_t + (p₀·BV + p)·1` under the genuine store mask
  `p₁·BT + i < Tt ∧ p₀·BV + p < V` (= the exact stack's `active`).
* `outDType := .real` — the terminal store lowers to `Stmt.store .real`
  (see the section header), so the boundary grid is exact. -/

/-- **Streaming IO signature** of `chunk_simple_gla_fwd_kernel_o` on the
five-stream single-output fold skin. The step count is `1`: in the file's
declared `K = BK` regime the key-block loop `for i_k in range(cdiv(K, BK))`
runs exactly once, so all five channels are static one-step streams. -/
def chunkGlaSimpleIO (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (Tt K V BT BK BV : Nat) : StreamMasked3DKernelIO₅ where
  kernel := chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
    s_h_h s_h_t scale Tt K V BT BK BV
  inp1 := q
  inp2 := k
  inp3 := h
  inp4 := g
  inp5 := v
  out := o
  T := 1
  B1 := BT * BK
  B2 := BK * BT
  B3 := BK * BV
  B4 := BT
  B5 := BT * BV
  C := BT * BV
  outDType := .real
  read1 := fun _ p₁ p₂ _ j => p₂ * s_k_h + (p₁ * BT + j.val / BK) * s_k_t + j.val % BK
  read2 := fun _ p₁ p₂ _ j => p₂ * s_k_h + (j.val / BT) * 1 + (p₁ * BT + j.val % BT) * s_k_t
  read3 := fun p₀ p₁ p₂ _ j =>
    p₂ * s_h_h + p₁ * K * V + (j.val / BV) * s_h_t + (p₀ * BV + j.val % BV) * 1
  read4 := fun _ p₁ p₂ _ j => p₂ * Tt + (p₁ * BT + j.val) * 1
  read5 := fun p₀ p₁ p₂ _ j =>
    p₂ * s_v_h + (p₁ * BT + j.val / BV) * s_v_t + (p₀ * BV + j.val % BV) * 1
  write := fun p₀ p₁ p₂ j =>
    p₂ * s_v_h + (p₁ * BT + j.val / BV) * s_v_t + (p₀ * BV + j.val % BV) * 1
  mask1 := fun _ p₁ _ _ j => p₁ * BT + j.val / BK < Tt ∧ j.val % BK < K
  mask2 := fun _ p₁ _ _ j => j.val / BT < K ∧ p₁ * BT + j.val % BT < Tt
  mask3 := fun p₀ _ _ _ j => j.val / BV < K ∧ p₀ * BV + j.val % BV < V
  mask4 := fun _ p₁ _ _ j => p₁ * BT + j.val < Tt
  mask5 := fun p₀ p₁ _ _ j => p₁ * BT + j.val / BV < Tt ∧ p₀ * BV + j.val % BV < V
  writeMask := fun p₀ p₁ _ j => p₁ * BT + j.val / BV < Tt ∧ p₀ * BV + j.val % BV < V

/-! ### The five streamed tiles (all at the single step `0`) -/

/-- `q[i, e]` off the first stream. -/
noncomputable def cgsIOqT (BT BK : Nat) (xs : Fin 1 → Fin (BT * BK) → ℝ)
    (i : Fin BT) (e : Fin BK) : ℝ := xs 0 (Lane2D.encode (i, e, PUnit.unit))

/-- `k[e, jj]` off the second stream. -/
noncomputable def cgsIOkT (BT BK : Nat) (ys : Fin 1 → Fin (BK * BT) → ℝ)
    (e : Fin BK) (jj : Fin BT) : ℝ := ys 0 (Lane2D.encode (e, jj, PUnit.unit))

/-- `h[e, p]` off the third stream. -/
noncomputable def cgsIOhT (BK BV : Nat) (zs : Fin 1 → Fin (BK * BV) → ℝ)
    (e : Fin BK) (p : Fin BV) : ℝ := zs 0 (Lane2D.encode (e, p, PUnit.unit))

/-- `g[i]` off the fourth stream. -/
noncomputable def cgsIOgT (BT : Nat) (ws : Fin 1 → Fin BT → ℝ) (i : Fin BT) : ℝ := ws 0 i

/-- `v[jj, p]` off the fifth stream. -/
noncomputable def cgsIOvT (BT BV : Nat) (vs : Fin 1 → Fin (BT * BV) → ℝ)
    (jj : Fin BT) (p : Fin BV) : ℝ := vs 0 (Lane2D.encode (jj, p, PUnit.unit))

/-- Masked, decayed intra-chunk score lane `(i, jj)` on the streams — the
stream restatement of `scoreTerm`. -/
noncomputable def cgsIOScoreTerm (BT BK : Nat) (xs : Fin 1 → Fin (BT * BK) → ℝ)
    (ys : Fin 1 → Fin (BK * BT) → ℝ) (ws : Fin 1 → Fin BT → ℝ)
    (i jj : Fin BT) : ℝ :=
  if jj.val ≤ i.val then
    (Finset.univ.sum fun e : Fin BK => cgsIOqT BT BK xs i e * cgsIOkT BT BK ys e jj)
      * Real.exp (cgsIOgT BT ws i - cgsIOgT BT ws jj)
  else 0

/-- **The GLA closed form on the streams** — `glaOutput` restated over the
five streamed tiles, at output lane `j = (i, p)` row-major over `[BT, BV]`. -/
noncomputable def chunkGlaSimpleIOOutSpec (scale : ℝ) (BT BK BV : Nat)
    (xs : Fin 1 → Fin (BT * BK) → ℝ) (ys : Fin 1 → Fin (BK * BT) → ℝ)
    (zs : Fin 1 → Fin (BK * BV) → ℝ) (ws : Fin 1 → Fin BT → ℝ)
    (vs : Fin 1 → Fin (BT * BV) → ℝ) (j : Fin (BT * BV)) : ℝ :=
  ((Finset.univ.sum fun e : Fin BK =>
        cgsIOqT BT BK xs (Lane2D.decode j).1 e
          * cgsIOhT BK BV zs e (Lane2D.decode j).2.1)
      * Real.exp (cgsIOgT BT ws (Lane2D.decode j).1)
    + Finset.univ.sum fun jj : Fin BT =>
        cgsIOScoreTerm BT BK xs ys ws (Lane2D.decode j).1 jj
          * cgsIOvT BT BV vs jj (Lane2D.decode j).2.1) * scale

/-! ### Stream-pin element bridges

Under the skin's input pins, each kernel-side memory element **is** the
corresponding streamed tile element. Every bridge carries exactly the window
hypothesis its channel's mask asks for. -/

/-- `qElem` = the first stream, at an active row. -/
private theorem cgsIOqT_elem (s₀ : BlockState) (q : RegionName)
    (s_k_h s_k_t Tt K BT BK : Nat) (hKBK : K = BK)
    (xs : Fin 1 → Fin (BT * BK) → ℝ)
    (hx : ∀ (t : Fin 1) (j : Fin (BT * BK)),
      (s₀.pids 1 * BT + j.val / BK < Tt ∧ j.val % BK < K) →
      s₀.readMem q (s₀.pids 2 * s_k_h + (s₀.pids 1 * BT + j.val / BK) * s_k_t + j.val % BK)
        = xs t j)
    (i : Fin BT) (e : Fin BK) (hi : s₀.pids 1 * BT + i.val < Tt) :
    qElem s₀ q s_k_h s_k_t BT i e.val = cgsIOqT BT BK xs i e := by
  rw [cgsIOqT, ← hx 0 (Lane2D.encode (i, e, PUnit.unit))
    (by simp only [Lane2D.encode_div, Lane2D.encode_mod]
        exact ⟨hi, by rw [hKBK]; exact e.isLt⟩)]
  simp only [Lane2D.encode_div, Lane2D.encode_mod, qElem]

/-- `kElem` = the second stream, at an active column. -/
private theorem cgsIOkT_elem (s₀ : BlockState) (k : RegionName)
    (s_k_h s_k_t Tt K BT BK : Nat) (hKBK : K = BK)
    (ys : Fin 1 → Fin (BK * BT) → ℝ)
    (hy : ∀ (t : Fin 1) (j : Fin (BK * BT)),
      (j.val / BT < K ∧ s₀.pids 1 * BT + j.val % BT < Tt) →
      s₀.readMem k (s₀.pids 2 * s_k_h + (j.val / BT) * 1
          + (s₀.pids 1 * BT + j.val % BT) * s_k_t) = ys t j)
    (jj : Fin BT) (e : Fin BK) (hjj : s₀.pids 1 * BT + jj.val < Tt) :
    kElem s₀ k s_k_h s_k_t BT jj e.val = cgsIOkT BT BK ys e jj := by
  rw [cgsIOkT, ← hy 0 (Lane2D.encode (e, jj, PUnit.unit))
    (by simp only [Lane2D.encode_div, Lane2D.encode_mod]
        exact ⟨by rw [hKBK]; exact e.isLt, hjj⟩)]
  simp only [Lane2D.encode_div, Lane2D.encode_mod, kElem]

/-- `hElem` = the third stream, at an active value column. -/
private theorem cgsIOhT_elem (s₀ : BlockState) (h : RegionName)
    (s_h_h s_h_t K V BK BV : Nat) (hKBK : K = BK)
    (zs : Fin 1 → Fin (BK * BV) → ℝ)
    (hz : ∀ (t : Fin 1) (j : Fin (BK * BV)),
      (j.val / BV < K ∧ s₀.pids 0 * BV + j.val % BV < V) →
      s₀.readMem h (s₀.pids 2 * s_h_h + s₀.pids 1 * K * V + (j.val / BV) * s_h_t
          + (s₀.pids 0 * BV + j.val % BV) * 1) = zs t j)
    (p : Fin BV) (e : Fin BK) (hp : s₀.pids 0 * BV + p.val < V) :
    hElem s₀ h s_h_h s_h_t K V BV p e.val = cgsIOhT BK BV zs e p := by
  rw [cgsIOhT, ← hz 0 (Lane2D.encode (e, p, PUnit.unit))
    (by simp only [Lane2D.encode_div, Lane2D.encode_mod]
        exact ⟨by rw [hKBK]; exact e.isLt, hp⟩)]
  simp only [Lane2D.encode_div, Lane2D.encode_mod, hElem]

/-- `gElem` = the fourth stream, at an active row. -/
private theorem cgsIOgT_elem (s₀ : BlockState) (g : RegionName) (Tt BT : Nat)
    (ws : Fin 1 → Fin BT → ℝ)
    (hw : ∀ (t : Fin 1) (j : Fin BT), s₀.pids 1 * BT + j.val < Tt →
      s₀.readMem g (s₀.pids 2 * Tt + (s₀.pids 1 * BT + j.val) * 1) = ws t j)
    (i : Fin BT) (hi : s₀.pids 1 * BT + i.val < Tt) :
    gElem s₀ g Tt BT i = cgsIOgT BT ws i := by
  rw [cgsIOgT, ← hw 0 i hi]; simp only [gElem]

/-- `vElem` = the fifth stream, at an active lane. -/
private theorem cgsIOvT_elem (s₀ : BlockState) (v : RegionName)
    (s_v_h s_v_t Tt V BT BV : Nat)
    (vs : Fin 1 → Fin (BT * BV) → ℝ)
    (hv : ∀ (t : Fin 1) (j : Fin (BT * BV)),
      (s₀.pids 1 * BT + j.val / BV < Tt ∧ s₀.pids 0 * BV + j.val % BV < V) →
      s₀.readMem v (s₀.pids 2 * s_v_h + (s₀.pids 1 * BT + j.val / BV) * s_v_t
          + (s₀.pids 0 * BV + j.val % BV) * 1) = vs t j)
    (jj : Fin BT) (p : Fin BV) (hjj : s₀.pids 1 * BT + jj.val < Tt)
    (hp : s₀.pids 0 * BV + p.val < V) :
    vElem s₀ v s_v_h s_v_t BT BV jj p = cgsIOvT BT BV vs jj p := by
  rw [cgsIOvT, ← hv 0 (Lane2D.encode (jj, p, PUnit.unit))
    (by simp only [Lane2D.encode_div, Lane2D.encode_mod]; exact ⟨hjj, hp⟩)]
  simp only [Lane2D.encode_div, Lane2D.encode_mod, vElem]

set_option maxHeartbeats 1000000 in
/-- **The closed form on the streams equals the closed form on memory** at
every write-active output lane. Off-diagonal columns `jj > i` need no `k`/`v`
pin: the causal mask zeroes them on both sides; columns `jj ≤ i` are active
because the row `i` is (`p₁·BT + jj ≤ p₁·BT + i < Tt`). -/
private theorem cgsIO_glaOutput_eq (s₀ : BlockState) (q k v h g : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (Tt K V BT BK BV : Nat) (hKBK : K = BK)
    (xs : Fin 1 → Fin (BT * BK) → ℝ) (ys : Fin 1 → Fin (BK * BT) → ℝ)
    (zs : Fin 1 → Fin (BK * BV) → ℝ) (ws : Fin 1 → Fin BT → ℝ)
    (vs : Fin 1 → Fin (BT * BV) → ℝ)
    (hx : ∀ (t : Fin 1) (j : Fin (BT * BK)),
      (s₀.pids 1 * BT + j.val / BK < Tt ∧ j.val % BK < K) →
      s₀.readMem q (s₀.pids 2 * s_k_h + (s₀.pids 1 * BT + j.val / BK) * s_k_t + j.val % BK)
        = xs t j)
    (hy : ∀ (t : Fin 1) (j : Fin (BK * BT)),
      (j.val / BT < K ∧ s₀.pids 1 * BT + j.val % BT < Tt) →
      s₀.readMem k (s₀.pids 2 * s_k_h + (j.val / BT) * 1
          + (s₀.pids 1 * BT + j.val % BT) * s_k_t) = ys t j)
    (hz : ∀ (t : Fin 1) (j : Fin (BK * BV)),
      (j.val / BV < K ∧ s₀.pids 0 * BV + j.val % BV < V) →
      s₀.readMem h (s₀.pids 2 * s_h_h + s₀.pids 1 * K * V + (j.val / BV) * s_h_t
          + (s₀.pids 0 * BV + j.val % BV) * 1) = zs t j)
    (hw : ∀ (t : Fin 1) (j : Fin BT), s₀.pids 1 * BT + j.val < Tt →
      s₀.readMem g (s₀.pids 2 * Tt + (s₀.pids 1 * BT + j.val) * 1) = ws t j)
    (hv : ∀ (t : Fin 1) (j : Fin (BT * BV)),
      (s₀.pids 1 * BT + j.val / BV < Tt ∧ s₀.pids 0 * BV + j.val % BV < V) →
      s₀.readMem v (s₀.pids 2 * s_v_h + (s₀.pids 1 * BT + j.val / BV) * s_v_t
          + (s₀.pids 0 * BV + j.val % BV) * 1) = vs t j)
    (j : Fin (BT * BV))
    (hact : s₀.pids 1 * BT + j.val / BV < Tt ∧ s₀.pids 0 * BV + j.val % BV < V) :
    glaOutput s₀ q k v h g s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale Tt K V BT BK BV
        (Lane2D.decode j).1 (Lane2D.decode j).2.1
      = chunkGlaSimpleIOOutSpec scale BT BK BV xs ys zs ws vs j := by
  obtain ⟨hi, hp⟩ := hact
  unfold glaOutput chunkGlaSimpleIOOutSpec interTerm
  refine congrArg (fun z => z * scale) ?_
  refine congrArg₂ (· + ·) ?_ ?_
  · refine congrArg₂ (· * ·) (Finset.sum_congr rfl fun e _ => ?_) ?_
    · rw [cgsIOqT_elem s₀ q s_k_h s_k_t Tt K BT BK hKBK xs hx _ e hi,
        cgsIOhT_elem s₀ h s_h_h s_h_t K V BK BV hKBK zs hz _ e hp]
    · rw [cgsIOgT_elem s₀ g Tt BT ws hw _ hi]
  · refine Finset.sum_congr rfl fun jj _ => ?_
    by_cases hjj : jj.val ≤ (Lane2D.decode j).1.val
    · have hjjT : s₀.pids 1 * BT + jj.val < Tt := by
        simp only [Lane2D.decode_row] at hjj; omega
      rw [cgsIOvT_elem s₀ v s_v_h s_v_t Tt V BT BV vs hv jj _ hjjT hp]
      refine congrArg (fun z => z * cgsIOvT BT BV vs jj (Lane2D.decode j).2.1) ?_
      simp only [scoreTerm, cgsIOScoreTerm, if_pos hjj]
      refine congrArg₂ (· * ·) (Finset.sum_congr rfl fun e _ => ?_) ?_
      · rw [cgsIOqT_elem s₀ q s_k_h s_k_t Tt K BT BK hKBK xs hx _ e hi,
          cgsIOkT_elem s₀ k s_k_h s_k_t Tt K BT BK hKBK ys hy jj e hjjT]
      · rw [cgsIOgT_elem s₀ g Tt BT ws hw _ hi, cgsIOgT_elem s₀ g Tt BT ws hw jj hjjT]
    · rw [show scoreTerm s₀ q k s_k_h s_k_t Tt BT BK g (Lane2D.decode j).1 jj = 0 from by
          simp only [scoreTerm, if_neg hjj],
        show cgsIOScoreTerm BT BK xs ys ws (Lane2D.decode j).1 jj = 0 from by
          simp only [cgsIOScoreTerm, if_neg hjj],
        zero_mul, zero_mul]

/-! ## The rounded Hoare triple: run, readback, frame -/

/-- A masked single-region `writeMem` scatter preserves every cell it does not
hit. -/
private theorem cgsIO_foldl_writeMem_frame {α : Type} (region : RegionName)
    (P : α → Prop) [DecidablePred P] (offFn : α → Nat) (valFn : α → ℝ) :
    ∀ (l : List α) (s : BlockState) (r : RegionName) (oo : Nat),
      (r = region → ∀ x ∈ l, P x → offFn x ≠ oo) →
      ((l.foldl (fun acc x =>
          if P x then acc.writeMem region (offFn x) (valFn x) else acc) s).mem r oo
        = s.mem r oo)
  | [], _, _, _, _ => rfl
  | x :: rest, s, r, oo, hh => by
      rw [List.foldl_cons]
      by_cases hx : P x
      · rw [if_pos hx,
          cgsIO_foldl_writeMem_frame region P offFn valFn rest _ r oo
            (fun hr y hy => hh hr y (List.mem_cons_of_mem _ hy)),
          BlockState.writeMem_mem,
          if_neg (fun hro => hh hro.1 x List.mem_cons_self hx hro.2.symm)]
      · rw [if_neg hx]
        exact cgsIO_foldl_writeMem_frame region P offFn valFn rest _ r oo
          (fun hr y hy => hh hr y (List.mem_cons_of_mem _ hy))

set_option maxHeartbeats 2000000 in
/-- **Full-kernel run with a frame.** Executing the surface from any
`undef`-free launch state terminates; every write-active output lane of `o`
carries the closed form `glaOutput`, and every cell outside the write window
is untouched. Refines `chunk_gla_simple_exec_glaOutput` with the final state
made explicit (the skin's `hrun` needs the frame, which the `.map` readback
form cannot state). -/
private theorem cgsIO_exec_run (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (Tt K V BT BK BV : Nat) (s : BlockState)
    (hKBK : K = BK) (hBK : 0 < BK) (hBT : 0 < BT)
    (hundef : ∀ rg off, s.undef rg off = 0)
    (hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx)) :
    ∃ sF, exec (chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale Tt K V BT BK BV).toAlgKernel s = some sF
      ∧ (∀ idx : TileIndex [BT, BV], active s Tt V BT BV idx →
          sF.readMem o (outOffset s s_v_h s_v_t BT BV idx)
            = glaOutput s q k v h g s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
                scale Tt K V BT BK BV idx.1 idx.2.1)
      ∧ (∀ (r : RegionName) (oo : Nat),
          (r ≠ o ∨ ∀ idx : TileIndex [BT, BV], active s Tt V BT BV idx →
            oo ≠ outOffset s s_v_h s_v_t BT BV idx) →
          sF.mem r oo = s.mem r oo) := by
  obtain ⟨sFront, hfront, hpids, hmemF, hpo, boF, hboF, hboData⟩ :=
    chunk_gla_simple_front_exec q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
      scale Tt K V BT BK BV s hKBK hBK hBT hundef
  have hstore : stepStmts [chunkGlaStoreStmt o BT BV] sFront
      = some ((TileShape.allIndices [BT, BV]).foldl
          (fun acc i =>
            if (s.pids 1 * BT + i.1.val < Tt ∧ s.pids 0 * BV + i.2.1.val < V) then
              acc.writeMem o (s.pids 2 * s_v_h + (s.pids 1 * BT + i.1.val) * s_v_t
                + (s.pids 0 * BV + i.2.1.val) * 1)
                (FloatDType.real.storeValue (boF.data i))
            else acc) sFront) := by
    rw [stepStmts.cons_some (st := chunkGlaStoreStmt o BT BV) (s' := _) ?_, stepStmts.nil]
    show stepStmt (chunkGlaStoreStmt o BT BV) sFront = some _
    unfold chunkGlaStoreStmt stepStmt
    simp only [evalOp_ref, hboF, hpo, Option.bind, Option.map]
    refine congrArg some ?_
    congr 1
    funext acc i
    simp only [TileShape.indexToList, BlockPtr.address_2d_offsets, BlockPtr.inBounds_2d_offsets,
      Bool.true_and, BlockState.writeMemTyped_real, decide_eq_true_eq]
  have hexec : exec (chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
        s_h_h s_h_t scale Tt K V BT BK BV).toAlgKernel s
      = some ((TileShape.allIndices [BT, BV]).foldl
          (fun acc i =>
            if (s.pids 1 * BT + i.1.val < Tt ∧ s.pids 0 * BV + i.2.1.val < V) then
              acc.writeMem o (s.pids 2 * s_v_h + (s.pids 1 * BT + i.1.val) * s_v_t
                + (s.pids 0 * BV + i.2.1.val) * 1)
                (FloatDType.real.storeValue (boF.data i))
            else acc) sFront) := by
    show stepStmts (chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t
      s_h_h s_h_t scale Tt K V BT BK BV).toAlgKernel.body s = _
    rw [chunk_gla_simple_body_split, stepStmts.append_some hfront]
    exact hstore
  refine ⟨_, hexec, ?_, ?_⟩
  · intro idx hActive
    set offsetFn : TileIndex [BT, BV] → Nat :=
      fun i => s.pids 2 * s_v_h + (s.pids 1 * BT + i.1.val) * s_v_t
        + (s.pids 0 * BV + i.2.1.val) * 1 with hoffdef
    set Pw : TileIndex [BT, BV] → Prop :=
      fun i => s.pids 1 * BT + i.1.val < Tt ∧ s.pids 0 * BV + i.2.1.val < V with hPdef
    have hoffInj : Function.Injective offsetFn := by
      simpa [offsetFn, outOffset, tIndex, vIndex] using hInj
    have hPidx : Pw idx := by simpa [Pw, active, tIndex, vIndex] using hActive
    have hcollision : ∀ kk, Pw kk → offsetFn kk = offsetFn idx → kk = idx :=
      fun kk _ heq => hoffInj heq
    rw [show outOffset s s_v_h s_v_t BT BV idx = offsetFn idx from by
      simp [outOffset, offsetFn, tIndex, vIndex]]
    rw [BlockState.scatter_readback_prop_masked_nd_of_true (region := o) sFront offsetFn
      (fun i => FloatDType.real.storeValue (boF.data i)) Pw idx hPidx hcollision]
    rw [hboData idx.1 idx.2.1 (by simpa [active, tIndex, vIndex] using hActive)]
    rfl
  · intro r oo hcond
    set offsetFn : TileIndex [BT, BV] → Nat :=
      fun i => s.pids 2 * s_v_h + (s.pids 1 * BT + i.1.val) * s_v_t
        + (s.pids 0 * BV + i.2.1.val) * 1 with hoffdef
    set Pw : TileIndex [BT, BV] → Prop :=
      fun i => s.pids 1 * BT + i.1.val < Tt ∧ s.pids 0 * BV + i.2.1.val < V with hPdef
    have hmiss : r = o → ∀ x ∈ TileShape.allIndices [BT, BV], Pw x → offsetFn x ≠ oo := by
      intro hr idx _ hPidx
      rcases hcond with hne | hno
      · exact absurd hr hne
      · intro hoff
        refine hno idx (by simpa [Pw, active, tIndex, vIndex] using hPidx) ?_
        rw [show outOffset s s_v_h s_v_t BT BV idx = offsetFn idx from by
          simp [outOffset, offsetFn, tIndex, vIndex]]
        exact hoff.symm
    rw [cgsIO_foldl_writeMem_frame o Pw offsetFn
      (fun i => FloatDType.real.storeValue (boF.data i)) (TileShape.allIndices [BT, BV])
      sFront r oo hmiss]
    exact hmemF r oo

/-! ## The safety walk

The skin's `hts` obligation quantifies over **arbitrary** launch states, so
the walk re-derives the surface's block-pointer registers statement by
statement (`R`-mirrors of the exact recipes above); every loaded/stored lane's
address is then literally the skin's `read*`/`write` window, in bounds by the
window hypotheses. -/

/-- `R`-mirror of the `programId` evaluation. -/
private theorem cgsIO_programId_evalR (R : RoundingModel) (n : Nat) (s : BlockState) :
    evalOpR R (Op.programId n) s = some (Tile.scalar (s.pids n)) := by
  simp only [evalOpR]

/-- `programId` at a state whose pid vector is already known. -/
private theorem cgsIO_programId_evalR' (R : RoundingModel) (n : Nat) (s : BlockState)
    (a : Nat) (hp : s.pids n = a) :
    evalOpR R (Op.programId n) s = some (Tile.scalar a) := by
  rw [cgsIO_programId_evalR, hp]

/-- `R`-mirror of `mulConst_eval`. -/
private theorem cgsIO_mulConst_evalR (R : RoundingModel) (s : BlockState)
    (name : RegName) (val c : Nat)
    (hr : s.regs .nat [] name = some (Tile.scalar val)) :
    evalOpR R (Op.mul .nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat c)) s
      = some (Tile.scalar (val * c)) := by
  simp only [evalOpR, evalOpR_ref, hr, Option.bind_eq_bind, Option.bind_some]
  rfl

/-- `R`-mirror of `makeBlockPtr_2d_eval`. -/
private theorem cgsIO_mbpdo2_evalR (R : RoundingModel) (rg : RegionName) (s : BlockState)
    (baseOp rowOffOp colOffOp : Op .nat [])
    (parentShape blockShape strides : List Nat) (base rowOff colOff : Nat)
    (hbase : evalOpR R baseOp s = some (Tile.scalar base))
    (hrow : evalOpR R rowOffOp s = some (Tile.scalar rowOff))
    (hcol : evalOpR R colOffOp s = some (Tile.scalar colOff)) :
    evalOpR R (Op.makeBlockPtrDynOffsets rg baseOp parentShape blockShape strides
        [rowOffOp, colOffOp]) s
      = some (⟨fun _ => BlockPtr.mk rg base parentShape blockShape strides
          [rowOff, colOff]⟩ : Tile .blockPtr blockShape) := by
  simp only [evalOpR, hbase, hrow, hcol, List.mapM, List.mapM.loop, bind, Option.bind,
    Tile.scalar, List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append,
    List.cons_append]

/-- `R`-mirror of `makeBlockPtr_1d_eval`. -/
private theorem cgsIO_mbpdo1_evalR (R : RoundingModel) (rg : RegionName) (s : BlockState)
    (baseOp offOp : Op .nat [])
    (parentShape blockShape strides : List Nat) (base off : Nat)
    (hbase : evalOpR R baseOp s = some (Tile.scalar base))
    (hoff : evalOpR R offOp s = some (Tile.scalar off)) :
    evalOpR R (Op.makeBlockPtrDynOffsets rg baseOp parentShape blockShape strides
        [offOp]) s
      = some (⟨fun _ => BlockPtr.mk rg base parentShape blockShape strides [off]⟩
          : Tile .blockPtr blockShape) := by
  simp only [evalOpR, hbase, hoff, List.mapM, List.mapM.loop, bind, Option.bind,
    Tile.scalar, List.reverse_cons, List.reverse_nil, List.append_nil, List.nil_append,
    List.cons_append]

/-- `R`-mirror of the `p_h` base evaluation `i_bh·s_h_h + i_t·K·V`. -/
private theorem cgsIO_hBase_evalR (R : RoundingModel) (s : BlockState)
    (s_h_h K V a b : Nat)
    (hbh : s.regs .nat [] "i_bh" = some (Tile.scalar a))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar b)) :
    evalOpR R (Op.add .nat Broadcast.nil
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
      (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K)) (Op.constNat V))) s
      = some (Tile.scalar (a * s_h_h + b * K * V)) := by
  simp only [evalOpR, evalOpR_ref, hbh, ht, Option.bind_eq_bind, Option.bind_some]
  rfl

set_option maxHeartbeats 4000000 in
/-- The key-block loop statement is cast-free (its `.nat` bounds never round
and its body collapses through `stepForRangeAuxR_castFree`). -/
private theorem cgsIO_loopStmt_castFree (R : RoundingModel)
    (q k v h : RegionName) (s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV : Nat)
    (u : BlockState) :
    stepStmtR R (Stmt.forRangeDyn "i_k" (Op.constNat 0)
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
          (Op.constNat BK))
        (Op.constNat 1)
        (chunkGlaLoopBody q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV)) u
      = stepStmt (Stmt.forRangeDyn "i_k" (Op.constNat 0)
        (Op.div .nat Broadcast.nil
          (Op.sub .nat Broadcast.nil
            (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
          (Op.constNat BK))
        (Op.constNat 1)
        (chunkGlaLoopBody q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV)) u := by
  simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def, bind, Option.bind,
    stepForRangeAuxR_castFree R _
      (cgsIO_loopBody_castFree R q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV) "i_k"]

/-- The compiled `cdiv(K, BK)` trip count is `1` in the `K = BK` regime
(the `R`-side mirror of the exact stack's `hstopE`). -/
private theorem cgsIO_stopOp_evalR (R : RoundingModel) (K BK : Nat)
    (hKBK : K = BK) (hBK : 0 < BK) (s : BlockState) :
    evalOpR R (Op.div .nat Broadcast.nil
      (Op.sub .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
      (Op.constNat BK)) s
      = some (Tile.scalar 1) := by
  have hcdiv : (K + BK - 1) / BK = 1 := by
    subst hKBK
    have he : K + K - 1 = K + (K - 1) := by omega
    rw [he, Nat.add_div_left _ hBK, Nat.div_eq_of_lt (by omega)]
  simp only [evalOpR, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  simp only [Tile.bop, NumericDType.div, NumericDType.sub, NumericDType.add, Tile.scalar]
  refine congrArg _ ?_
  funext _; exact hcdiv

/-- The exact-side mirror of `cgsIO_stopOp_evalR` (needed to resolve the
`forRangeDyn` bounds when threading the loop's successor state). -/
private theorem cgsIO_stopOp_eval (K BK : Nat) (hKBK : K = BK) (hBK : 0 < BK)
    (s : BlockState) :
    evalOp (Op.div .nat Broadcast.nil
      (Op.sub .nat Broadcast.nil
        (Op.add .nat Broadcast.nil (Op.constNat K) (Op.constNat BK)) (Op.constNat 1))
      (Op.constNat BK)) s
      = some (Tile.scalar 1) := by
  have hcdiv : (K + BK - 1) / BK = 1 := by
    subst hKBK
    have he : K + K - 1 = K + (K - 1) := by omega
    rw [he, Nat.add_div_left _ hBK, Nat.div_eq_of_lt (by omega)]
  simp only [evalOp_div, evalOp_sub, evalOp_add, evalOp_constNat, Option.bind_eq_bind,
    Option.bind_some]
  refine congrArg some ?_
  simp only [Tile.bop, NumericDType.div, NumericDType.sub, NumericDType.add, Tile.scalar]
  refine congrArg _ ?_
  funext _; exact hcdiv

/-- A register read survives an assignment to a **different** name. Stated so
that a walk can chain it without ever having to spell out the intermediate
`setReg` towers (every implicit is resolved by the rewrite it feeds). -/
private theorem cgsIO_reg_step {dt dt' : TileDType} {sh sh' : TileShape}
    (nm nm2 : RegName) {u : BlockState} {X : Tile dt sh} {Y : Tile dt' sh'}
    (hne : nm2 ≠ nm) (hu : u.regs dt' sh' nm2 = some Y) :
    (u.setReg nm dt sh X).regs dt' sh' nm2 = some Y := by
  rw [BlockState.setReg_ne_name u nm nm2 dt dt' sh sh' X hne, hu]

/-- Address safety of the surface's `make_block_ptr` assigns: the base and the
offsets are `.nat` register arithmetic. Kept as a lemma so the walk never feeds
`Op.SafeAtR.eq_def` a `∀ off ∈ offsets` binder (where its equation diverges on
the bound variable). -/
private theorem cgsIO_mbpdo_safeR (R : RoundingModel) (bounds : RegionBounds) (s : BlockState)
    (rg : RegionName) (nm : RegName) (base : Op .nat []) (ps : List Nat) (bs : TileShape)
    (strides : List Nat) (offs : List (Op .nat []))
    (hb : Op.SafeAtR R bounds s base)
    (ho : ∀ off ∈ offs, Op.SafeAtR R bounds s off) :
    Stmt.TraceSafeR R bounds
      (Stmt.assign .blockPtr bs nm (Op.makeBlockPtrDynOffsets rg base ps bs strides offs)) s := by
  simp only [Stmt.TraceSafeR, Op.SafeAtR]
  exact ⟨hb, ho⟩

/-- The `.nat` scalar `ref * const` (every offset the surface builds) is
address-safe at every state. -/
private theorem cgsIO_mulRef_safeR (R : RoundingModel) (bounds : RegionBounds) (s : BlockState)
    (nm : RegName) (c : Nat) :
    Op.SafeAtR R bounds s (Op.mul .nat Broadcast.nil (Op.ref .nat [] nm) (Op.constNat c)) := by
  simp only [Op.SafeAtR, and_self]

/-- The `p_h` base `i_bh * s_h_h + i_t * K * V` is address-safe at every state. -/
private theorem cgsIO_hBase_safeR (R : RoundingModel) (bounds : RegionBounds) (s : BlockState)
    (s_h_h K V : Nat) :
    Op.SafeAtR R bounds s (Op.add .nat Broadcast.nil
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
      (Op.mul .nat Broadcast.nil
        (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K)) (Op.constNat V))) := by
  simp only [Op.SafeAtR, and_self]

set_option maxHeartbeats 8000000 in
set_option maxRecDepth 4000 in
/-- **The `TraceSafeR` walk for the whole kernel.** Every load and the terminal
store is a `boundary_check`-ed block-pointer access whose in-bounds lanes land
exactly on the skin's `read*` / `write` windows, so the six window hypotheses
are precisely the skin's own bound obligations. The walk needs no `undef` pin
(nothing here reads an undefined cell) and holds at an arbitrary launch
state. -/
private theorem cgsIO_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (Tt K V BT BK BV : Nat) (s : BlockState)
    (hKBK : K = BK) (hBK : 0 < BK)
    (hbq : ∀ j : Fin (BT * BK), s.pids 1 * BT + j.val / BK < Tt → j.val % BK < K →
      s.pids 2 * s_k_h + (s.pids 1 * BT + j.val / BK) * s_k_t + j.val % BK < bounds q)
    (hbk : ∀ j : Fin (BK * BT), j.val / BT < K → s.pids 1 * BT + j.val % BT < Tt →
      s.pids 2 * s_k_h + (j.val / BT) * 1 + (s.pids 1 * BT + j.val % BT) * s_k_t < bounds k)
    (hbh : ∀ j : Fin (BK * BV), j.val / BV < K → s.pids 0 * BV + j.val % BV < V →
      s.pids 2 * s_h_h + s.pids 1 * K * V + (j.val / BV) * s_h_t
        + (s.pids 0 * BV + j.val % BV) * 1 < bounds h)
    (hbg : ∀ j : Fin BT, s.pids 1 * BT + j.val < Tt →
      s.pids 2 * Tt + (s.pids 1 * BT + j.val) * 1 < bounds g)
    (hbv : ∀ j : Fin (BT * BV), s.pids 1 * BT + j.val / BV < Tt →
      s.pids 0 * BV + j.val % BV < V →
      s.pids 2 * s_v_h + (s.pids 1 * BT + j.val / BV) * s_v_t
        + (s.pids 0 * BV + j.val % BV) * 1 < bounds v)
    (hbo : ∀ j : Fin (BT * BV), s.pids 1 * BT + j.val / BV < Tt →
      s.pids 0 * BV + j.val % BV < V →
      s.pids 2 * s_v_h + (s.pids 1 * BT + j.val / BV) * s_v_t
        + (s.pids 0 * BV + j.val % BV) * 1 < bounds o) :
    ((chunk_gla_simple_fwd_surface q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
      scale Tt K V BT BK BV).toAlgKernel).TraceSafeR R bounds s := by
  unfold Kernel.TraceSafeR
  rw [chunk_gla_simple_body_split]
  simp only [chunkGlaFront, List.cons_append, List.nil_append]
  -- (1)(2)(3) the three program ids
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u1 h1 => ?_)
  rw [stepStmtR_assign_eq_some (cgsIO_programId_evalR R 0 s)] at h1
  obtain rfl := Option.some.inj h1
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u2 h2 => ?_)
  rw [stepStmtR_assign_eq_some (cgsIO_programId_evalR' R 1
    (s.setReg "i_v" .nat [] (Tile.scalar (s.pids 0))) (s.pids 1) (by simp))] at h2
  obtain rfl := Option.some.inj h2
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u3 h3 => ?_)
  rw [stepStmtR_assign_eq_some (cgsIO_programId_evalR' R 2
    ((s.setReg "i_v" .nat [] (Tile.scalar (s.pids 0))).setReg "i_t" .nat []
      (Tile.scalar (s.pids 1))) (s.pids 2) (by simp))] at h3
  obtain rfl := Option.some.inj h3
  -- (4)(5)(6)(7) o_i / m_s / b_o / b_s: register-only, values irrelevant
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u4 h4 => ?_)
  obtain ⟨oiT, -, rfl⟩ := stepStmtR_assign_inv h4
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u5 h5 => ?_)
  obtain ⟨msT, -, rfl⟩ := stepStmtR_assign_inv h5
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u6 h6 => ?_)
  obtain ⟨boT, -, rfl⟩ := stepStmtR_assign_inv h6
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u7 h7 => ?_)
  obtain ⟨bsT, -, rfl⟩ := stepStmtR_assign_inv h7
  -- the loop-entry state and its pinned scalars
  set s7 : BlockState :=
    (((((((s.setReg "i_v" .nat [] (Tile.scalar (s.pids 0))).setReg "i_t" .nat []
      (Tile.scalar (s.pids 1))).setReg "i_bh" .nat [] (Tile.scalar (s.pids 2))).setReg
      "o_i" .nat [BT] oiT).setReg "m_s" .bool [BT, BT] msT).setReg "b_o" .real [BT, BV]
      boT).setReg "b_s" .real [BT, BT] bsT) with hs7
  set t0 : BlockState := s7.setReg "i_k" .nat [] (Tile.scalar 0) with ht0
  have hpids0 : t0.pids = s.pids := by rw [ht0, hs7]; simp
  have hiv0 : t0.regs .nat [] "i_v" = some (Tile.scalar (s.pids 0)) := by
    rw [ht0, hs7]; simp
  have hit0 : t0.regs .nat [] "i_t" = some (Tile.scalar (s.pids 1)) := by
    rw [ht0, hs7]; simp
  have hibh0 : t0.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [ht0, hs7]; simp
  have hik0 : t0.regs .nat [] "i_k" = some (Tile.scalar 0) := by rw [ht0]; simp
  -- the loop body actually runs (single key block)
  obtain ⟨sBody, hbody, hbpids, hbmem, hbiv, hbit, hbibh, hbms, hbbo, hbbs⟩ :=
    chunkGlaLoopBody_steps q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV t0 boT bsT
      (by rw [hpids0]; exact hiv0) (by rw [hpids0]; exact hit0) (by rw [hpids0]; exact hibh0)
      hik0 (by rw [ht0, hs7]; simp) (by rw [ht0, hs7]; simp)
  have hbivS : sBody.regs .nat [] "i_v" = some (Tile.scalar (s.pids 0)) := by
    rw [hbiv, hpids0]
  have hbitS : sBody.regs .nat [] "i_t" = some (Tile.scalar (s.pids 1)) := by
    rw [hbit, hpids0]
  have hbibhS : sBody.regs .nat [] "i_bh" = some (Tile.scalar (s.pids 2)) := by
    rw [hbibh, hpids0]
  -- (8) the key-block loop
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun u8 h8 => ?_)
  · simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR.eq_def], by simp [Op.SafeAtR.eq_def],
      by simp [Op.SafeAtR.eq_def], ?_⟩
    rw [show evalOpR R (Op.constNat 0) s7 = some (Tile.scalar 0) from by simp only [evalOpR],
      cgsIO_stopOp_evalR R K BK hKBK hBK s7,
      show evalOpR R (Op.constNat 1) s7 = some (Tile.scalar 1) from by simp only [evalOpR]]
    show Stmt.forRangeTraceSafeR R bounds "i_k" 0 1 1
      (chunkGlaLoopBody q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV) s7
    refine Stmt.forRangeTraceSafeR_inv R bounds "i_k" 1 1
      (chunkGlaLoopBody q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV)
      (fun c u => (c = 0 ∧ u = s7) ∨ 1 ≤ c) ?_ 0 s7 (Or.inl ⟨rfl, rfl⟩)
    rintro c u hc (⟨rfl, rfl⟩ | hge)
    · refine ⟨?_, sBody, ?_, Or.inr (by norm_num)⟩
      · -- the body's own trace safety, at `t0`
        rw [← ht0]
        unfold chunkGlaLoopBody
        -- b1: p_q
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun c1 hc1 => ?_)
        · refine cgsIO_mbpdo_safeR R bounds _ q "p_q" _ _ _ _ _
            (cgsIO_mulRef_safeR _ _ _ "i_bh" s_k_h) ?_
          intro off hoff
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hoff
          rcases hoff with rfl | rfl
          · exact cgsIO_mulRef_safeR _ _ _ "i_t" BT
          · exact cgsIO_mulRef_safeR _ _ _ "i_k" BK
        rw [stepStmtR_assign_eq_some (cgsIO_mbpdo2_evalR R q t0
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK))
          [Tt, K] [BT, BK] [s_k_t, 1] (s.pids 2 * s_k_h) (s.pids 1 * BT) (0 * BK)
          (cgsIO_mulConst_evalR R t0 "i_bh" (s.pids 2) s_k_h hibh0)
          (cgsIO_mulConst_evalR R t0 "i_t" (s.pids 1) BT hit0)
          (cgsIO_mulConst_evalR R t0 "i_k" 0 BK hik0))] at hc1
        obtain rfl := Option.some.inj hc1
        -- b2: p_k
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun c2 hc2 => ?_)
        · refine cgsIO_mbpdo_safeR R bounds _ k "p_k" _ _ _ _ _
            (cgsIO_mulRef_safeR _ _ _ "i_bh" s_k_h) ?_
          intro off hoff
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hoff
          rcases hoff with rfl | rfl
          · exact cgsIO_mulRef_safeR _ _ _ "i_k" BK
          · exact cgsIO_mulRef_safeR _ _ _ "i_t" BT
        rw [stepStmtR_assign_eq_some (cgsIO_mbpdo2_evalR R k _
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_k_h))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
          [K, Tt] [BK, BT] [1, s_k_t] (s.pids 2 * s_k_h) (0 * BK) (s.pids 1 * BT)
          (cgsIO_mulConst_evalR R _ "i_bh" (s.pids 2) s_k_h
            (cgsIO_reg_step "p_q" "i_bh" (by simp) hibh0))
          (cgsIO_mulConst_evalR R _ "i_k" 0 BK
            (cgsIO_reg_step "p_q" "i_k" (by simp) hik0))
          (cgsIO_mulConst_evalR R _ "i_t" (s.pids 1) BT
            (cgsIO_reg_step "p_q" "i_t" (by simp) hit0)))] at hc2
        obtain rfl := Option.some.inj hc2
        -- b3: p_h
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun c3 hc3 => ?_)
        · refine cgsIO_mbpdo_safeR R bounds _ h "p_h" _ _ _ _ _
            (cgsIO_hBase_safeR _ _ _ s_h_h K V) ?_
          intro off hoff
          simp only [List.mem_cons, List.not_mem_nil, or_false] at hoff
          rcases hoff with rfl | rfl
          · exact cgsIO_mulRef_safeR _ _ _ "i_k" BK
          · exact cgsIO_mulRef_safeR _ _ _ "i_v" BV
        rw [stepStmtR_assign_eq_some (cgsIO_mbpdo2_evalR R h _
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_h_h))
            (Op.mul .nat Broadcast.nil
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat K))
              (Op.constNat V)))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_k") (Op.constNat BK))
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV))
          [K, V] [BK, BV] [s_h_t, 1] (s.pids 2 * s_h_h + s.pids 1 * K * V)
          (0 * BK) (s.pids 0 * BV)
          (cgsIO_hBase_evalR R _ s_h_h K V (s.pids 2) (s.pids 1)
            (cgsIO_reg_step "p_k" "i_bh" (by simp)
              (cgsIO_reg_step "p_q" "i_bh" (by simp) hibh0))
            (cgsIO_reg_step "p_k" "i_t" (by simp)
              (cgsIO_reg_step "p_q" "i_t" (by simp) hit0)))
          (cgsIO_mulConst_evalR R _ "i_k" 0 BK
            (cgsIO_reg_step "p_k" "i_k" (by simp)
              (cgsIO_reg_step "p_q" "i_k" (by simp) hik0)))
          (cgsIO_mulConst_evalR R _ "i_v" (s.pids 0) BV
            (cgsIO_reg_step "p_k" "i_v" (by simp)
              (cgsIO_reg_step "p_q" "i_v" (by simp) hiv0))))] at hc3
        obtain rfl := Option.some.inj hc3
        -- b4: b_q = load p_q  (`read1` window)
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun c4 hc4 => ?_)
        · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
            MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
          refine ⟨trivial, trivial, ?_⟩
          intro ptrs hptrs idx _
          rw [evalOpR_ref, cgsIO_reg_step "p_h" "p_q" (by simp)
            (cgsIO_reg_step "p_k" "p_q" (by simp)
              (BlockState.setReg_same _ "p_q" _ _ _))] at hptrs
          obtain rfl := Option.some.inj hptrs
          intro hin
          simp only [TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
            decide_eq_true_eq] at hin
          simp only [TileShape.indexToList, BlockPtr.address_2d_offsets]
          have hb := hbq (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
            (by simp only [Lane2D.encode_div]; exact hin.1)
            (by simp only [Lane2D.encode_mod]; simpa using hin.2)
          simp only [Lane2D.encode_div, Lane2D.encode_mod] at hb
          simpa using hb
        obtain ⟨bqT, -, rfl⟩ := stepStmtR_assign_inv hc4
        -- b5: b_k = load p_k  (`read2` window)
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun c5 hc5 => ?_)
        · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
            MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
          refine ⟨trivial, trivial, ?_⟩
          intro ptrs hptrs idx _
          rw [evalOpR_ref, cgsIO_reg_step "b_q" "p_k" (by simp)
            (cgsIO_reg_step "p_h" "p_k" (by simp)
              (BlockState.setReg_same _ "p_k" _ _ _))] at hptrs
          obtain rfl := Option.some.inj hptrs
          intro hin
          simp only [TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
            decide_eq_true_eq] at hin
          simp only [TileShape.indexToList, BlockPtr.address_2d_offsets]
          have hb := hbk (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
            (by simp only [Lane2D.encode_div]; simpa using hin.1)
            (by simp only [Lane2D.encode_mod]; exact hin.2)
          simp only [Lane2D.encode_div, Lane2D.encode_mod] at hb
          simpa using hb
        obtain ⟨bkT, -, rfl⟩ := stepStmtR_assign_inv hc5
        -- b6: b_h = load p_h  (`read3` window)
        refine Stmt.TraceSafeListR.cons_intro ?_ (fun c6 hc6 => ?_)
        · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
            MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
          refine ⟨trivial, trivial, ?_⟩
          intro ptrs hptrs idx _
          rw [evalOpR_ref, cgsIO_reg_step "b_k" "p_h" (by simp)
            (cgsIO_reg_step "b_q" "p_h" (by simp)
              (BlockState.setReg_same _ "p_h" _ _ _))] at hptrs
          obtain rfl := Option.some.inj hptrs
          intro hin
          simp only [TileShape.indexToList, BlockPtr.inBounds_2d_offsets,
            decide_eq_true_eq] at hin
          simp only [TileShape.indexToList, BlockPtr.address_2d_offsets]
          have hb := hbh (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
            (by simp only [Lane2D.encode_div]; simpa using hin.1)
            (by simp only [Lane2D.encode_mod]; exact hin.2)
          simp only [Lane2D.encode_div, Lane2D.encode_mod] at hb
          simpa using hb
        obtain ⟨bhT, -, rfl⟩ := stepStmtR_assign_inv hc6
        -- b7 / b8: the two dots (register-only)
        refine Stmt.TraceSafeListR.cons_intro
          (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun c7 hc7 => ?_)
        obtain ⟨w7, -, rfl⟩ := stepStmtR_assign_inv hc7
        exact Stmt.TraceSafeListR.cons_intro
          (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def])
          (fun _ _ => Stmt.TraceSafeListR.nil_intro)
      · rw [cgsIO_loopBody_castFree R q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV,
          ← ht0]
        exact hbody
    · exact absurd hc (by omega)
  -- the loop's successor state is `sBody`
  rw [cgsIO_loopStmt_castFree R q k v h s_k_h s_k_t s_h_h s_h_t Tt K V BT BK BV s7,
    forRangeDyn_single_step (start := 0) (stop := 1) (step := 1)
      (by simp) (cgsIO_stopOp_eval K BK hKBK hBK s7) (by simp)
      (by norm_num) (by norm_num) (by norm_num) (by rw [← ht0]; exact hbody)] at h8
  obtain rfl := Option.some.inj h8
  -- (9) p_g
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun u9 h9 => ?_)
  · refine cgsIO_mbpdo_safeR R bounds _ g "p_g" _ _ _ _ _
      (cgsIO_mulRef_safeR _ _ _ "i_bh" Tt) ?_
    intro off hoff
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hoff
    rcases hoff with rfl
    exact cgsIO_mulRef_safeR _ _ _ "i_t" BT
  rw [stepStmtR_assign_eq_some (cgsIO_mbpdo1_evalR R g sBody
    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat Tt))
    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
    [Tt] [BT] [1] (s.pids 2 * Tt) (s.pids 1 * BT)
    (cgsIO_mulConst_evalR R sBody "i_bh" (s.pids 2) Tt hbibhS)
    (cgsIO_mulConst_evalR R sBody "i_t" (s.pids 1) BT hbitS))] at h9
  obtain rfl := Option.some.inj h9
  -- (10) b_g = load p_g  (`read4` window)
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun u10 h10 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs idx _
    rw [evalOpR_ref, BlockState.setReg_same _ "p_g" _ _ _] at hptrs
    obtain rfl := Option.some.inj hptrs
    intro hin
    simp only [TileShape.indexToList, BlockPtr.inBounds_1d, decide_eq_true_eq] at hin
    simp only [TileShape.indexToList, BlockPtr.address_1d]
    exact hbg idx.1 hin
  obtain ⟨bgT, -, rfl⟩ := stepStmtR_assign_inv h10
  -- (11)(12)(13) gate decay + causal mask (register-only)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u11 h11 => ?_)
  obtain ⟨w11, -, rfl⟩ := stepStmtR_assign_inv h11
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u12 h12 => ?_)
  obtain ⟨w12, -, rfl⟩ := stepStmtR_assign_inv h12
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u13 h13 => ?_)
  obtain ⟨w13, -, rfl⟩ := stepStmtR_assign_inv h13
  -- (14) p_v
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun u14 h14 => ?_)
  · refine cgsIO_mbpdo_safeR R bounds _ v "p_v" _ _ _ _ _
      (cgsIO_mulRef_safeR _ _ _ "i_bh" s_v_h) ?_
    intro off hoff
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hoff
    rcases hoff with rfl | rfl
    · exact cgsIO_mulRef_safeR _ _ _ "i_t" BT
    · exact cgsIO_mulRef_safeR _ _ _ "i_v" BV
  rw [stepStmtR_assign_eq_some (cgsIO_mbpdo2_evalR R v _
    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h))
    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV))
    [Tt, V] [BT, BV] [s_v_t, 1] (s.pids 2 * s_v_h) (s.pids 1 * BT) (s.pids 0 * BV)
    (cgsIO_mulConst_evalR R _ "i_bh" (s.pids 2) s_v_h
      (cgsIO_reg_step "b_s" "i_bh" (by simp) (cgsIO_reg_step "b_s" "i_bh" (by simp)
        (cgsIO_reg_step "b_o" "i_bh" (by simp) (cgsIO_reg_step "b_g" "i_bh" (by simp)
          (cgsIO_reg_step "p_g" "i_bh" (by simp) hbibhS))))))
    (cgsIO_mulConst_evalR R _ "i_t" (s.pids 1) BT
      (cgsIO_reg_step "b_s" "i_t" (by simp) (cgsIO_reg_step "b_s" "i_t" (by simp)
        (cgsIO_reg_step "b_o" "i_t" (by simp) (cgsIO_reg_step "b_g" "i_t" (by simp)
          (cgsIO_reg_step "p_g" "i_t" (by simp) hbitS))))))
    (cgsIO_mulConst_evalR R _ "i_v" (s.pids 0) BV
      (cgsIO_reg_step "b_s" "i_v" (by simp) (cgsIO_reg_step "b_s" "i_v" (by simp)
        (cgsIO_reg_step "b_o" "i_v" (by simp) (cgsIO_reg_step "b_g" "i_v" (by simp)
          (cgsIO_reg_step "p_g" "i_v" (by simp) hbivS)))))))] at h14
  obtain rfl := Option.some.inj h14
  -- (15) b_v = load p_v  (`read5` window)
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun u15 h15 => ?_)
  · simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR,
      MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs idx _
    rw [evalOpR_ref, BlockState.setReg_same _ "p_v" _ _ _] at hptrs
    obtain rfl := Option.some.inj hptrs
    intro hin
    simp only [TileShape.indexToList, BlockPtr.inBounds_2d_offsets, decide_eq_true_eq] at hin
    simp only [TileShape.indexToList, BlockPtr.address_2d_offsets]
    have hb := hbv (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
      (by simp only [Lane2D.encode_div]; exact hin.1)
      (by simp only [Lane2D.encode_mod]; exact hin.2)
    simp only [Lane2D.encode_div, Lane2D.encode_mod] at hb
    simpa using hb
  obtain ⟨bvT, -, rfl⟩ := stepStmtR_assign_inv h15
  -- (16) the final accumulation (register-only)
  refine Stmt.TraceSafeListR.cons_intro
    (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun u16 h16 => ?_)
  obtain ⟨w16, -, rfl⟩ := stepStmtR_assign_inv h16
  -- (17) p_o
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun u17 h17 => ?_)
  · refine cgsIO_mbpdo_safeR R bounds _ o "p_o" _ _ _ _ _
      (cgsIO_mulRef_safeR _ _ _ "i_bh" s_v_h) ?_
    intro off hoff
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hoff
    rcases hoff with rfl | rfl
    · exact cgsIO_mulRef_safeR _ _ _ "i_t" BT
    · exact cgsIO_mulRef_safeR _ _ _ "i_v" BV
  rw [stepStmtR_assign_eq_some (cgsIO_mbpdo2_evalR R o _
    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat s_v_h))
    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT))
    (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_v") (Op.constNat BV))
    [Tt, V] [BT, BV] [s_v_t, 1] (s.pids 2 * s_v_h) (s.pids 1 * BT) (s.pids 0 * BV)
    (cgsIO_mulConst_evalR R _ "i_bh" (s.pids 2) s_v_h
      (cgsIO_reg_step "b_o" "i_bh" (by simp) (cgsIO_reg_step "b_v" "i_bh" (by simp)
        (cgsIO_reg_step "p_v" "i_bh" (by simp) (cgsIO_reg_step "b_s" "i_bh" (by simp)
          (cgsIO_reg_step "b_s" "i_bh" (by simp) (cgsIO_reg_step "b_o" "i_bh" (by simp)
            (cgsIO_reg_step "b_g" "i_bh" (by simp)
              (cgsIO_reg_step "p_g" "i_bh" (by simp) hbibhS)))))))))
    (cgsIO_mulConst_evalR R _ "i_t" (s.pids 1) BT
      (cgsIO_reg_step "b_o" "i_t" (by simp) (cgsIO_reg_step "b_v" "i_t" (by simp)
        (cgsIO_reg_step "p_v" "i_t" (by simp) (cgsIO_reg_step "b_s" "i_t" (by simp)
          (cgsIO_reg_step "b_s" "i_t" (by simp) (cgsIO_reg_step "b_o" "i_t" (by simp)
            (cgsIO_reg_step "b_g" "i_t" (by simp)
              (cgsIO_reg_step "p_g" "i_t" (by simp) hbitS)))))))))
    (cgsIO_mulConst_evalR R _ "i_v" (s.pids 0) BV
      (cgsIO_reg_step "b_o" "i_v" (by simp) (cgsIO_reg_step "b_v" "i_v" (by simp)
        (cgsIO_reg_step "p_v" "i_v" (by simp) (cgsIO_reg_step "b_s" "i_v" (by simp)
          (cgsIO_reg_step "b_s" "i_v" (by simp) (cgsIO_reg_step "b_o" "i_v" (by simp)
            (cgsIO_reg_step "b_g" "i_v" (by simp)
              (cgsIO_reg_step "p_g" "i_v" (by simp) hbivS))))))))))] at h17
  obtain rfl := Option.some.inj h17
  -- (18) the terminal store  (`write` window)
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  simp only [chunkGlaStoreStmt, Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR,
    Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
    memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, trivial, ?_⟩
  intro ptrs hptrs idx _
  rw [evalOpR_ref, BlockState.setReg_same _ "p_o" _ _ _] at hptrs
  obtain rfl := Option.some.inj hptrs
  intro hin
  simp only [TileShape.indexToList, BlockPtr.inBounds_2d_offsets, decide_eq_true_eq] at hin
  simp only [TileShape.indexToList, BlockPtr.address_2d_offsets]
  have hb := hbo (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
    (by simp only [Lane2D.encode_div]; exact hin.1)
    (by simp only [Lane2D.encode_mod]; exact hin.2)
  simp only [Lane2D.encode_div, Lane2D.encode_mod] at hb
  simpa using hb

/-! ### ════════ ★ MAIN THEOREM ★ ════════ -/

set_option maxHeartbeats 4000000 in
/-- **The `chunk_gla_simple` `⊨[R]` io headline.** For every rounding model
`R`, `chunk_simple_gla_fwd_kernel_o` implements, on its
`StreamMasked3DKernelIO₅` signature, the **ideal-ℝ chunked gated-linear-attention
output fold** over its five streamed input channels: output lane `j = (i, p)`
of `o` holds

```
  ( (Σ_e q[i,e]·h[e,p]) · exp(g_i)
    + Σ_jj (if jj ≤ i then (Σ_e q[i,e]·k[e,jj]) · exp(g_i − g_jj) else 0) · v[jj,p]
  ) · scale
```

(`chunkGlaSimpleIOOutSpec` = the exact stack's `glaOutput` closed form restated
on the streams), and every flat cell outside the write window is untouched.

The output grid is the `.real` default: the surface's
`tl.store(p_o, b_o.to(p_o.dtype.element_ty), …)` cast **erases at translation**
(the lowered statement is `chunkGlaStoreStmt = Stmt.store .real …`, matched to
the surface by `chunk_gla_simple_body_split`'s `rfl`), and so does the pre-store
`b_s.to(b_v.dtype)`. The whole body is therefore cast-free — machine-checked by
`cgsIO_execR_eq : execR R … = exec …` — so **no `R.round .fp16 = id` modeling
boundary is carried**, and at every `R` the terminal cells hold the exact fold
values.

**Hypothesis provenance** (all truth-forced, inherited from the exact headline
`chunk_gla_simple_output_summary_general`):

* `hKBK : K = BK` — the single-key-block regime: it makes the surface's
  `for i_k in range(cdiv(K, BK))` a one-iteration loop, which is what the skin's
  step count `T = 1` records (and what every checked Python case runs:
  `K = BK = 64` for cases 1–3, `K = BK = 32` for case 4).
* `hBK : 0 < BK` — needed to evaluate the compiled `(K + BK − 1) / BK` trip
  count; a zero-wide key block is not a launch.
* `hBT : 0 < BT` — nonempty time block, as on the exact stack.
* `hInj` — the terminal store is a masked scatter, so its per-lane readback is
  only well defined when distinct output lanes hit distinct offsets; this is the
  ∀-pids form of the exact headline's per-program `hInj`.

The exact headline's `hundef` is **not** a hypothesis here — the skin's Hoare
triple carries the `undef` pin itself.

**Scope inherited from the port**: arithmetic is over `ℝ` (not bit-accurate
IEEE); `@triton.autotune` / `num_warps` are not modeled; the host launch (the
3-D grid and the host-computed `BK`/`BV`) is the trusted boundary, and the
statement is universally quantified over the launch state, so it covers every
program of the grid. -/
specification chunk_gla_simple_io_correctness (R : RoundingModel)
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat) (scale : ℝ)
    (Tt K V BT BK BV : Nat)
    (hKBK : K = BK) (hBK : 0 < BK) (hBT : 0 < BT)
    (hInj : ∀ p₀ p₁ p₂ : Nat, Function.Injective
      (fun idx : TileIndex [BT, BV] =>
        p₂ * s_v_h + (p₁ * BT + idx.1.val) * s_v_t + (p₀ * BV + idx.2.1.val) * 1)) :
    chunkGlaSimpleIO q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      Tt K V BT BK BV ⊨[R]
      fun _ _ _ xs ys zs ws vs j =>
        chunkGlaSimpleIOOutSpec scale BT BK BV xs ys zs ws vs j := by
  refine StreamMasked3DKernelIO₅.ImplementsR.intro _ ?_ ?_ ?_
  · exact cgsIO_flattenOk q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
      Tt K V BT BK BV
  · -- the safety walk
    intro bounds s xs ys zs ws vs _hx _hy _hz _hw _hv hbr1 hbr2 hbr3 hbr4 hbr5 hbw
    simp only [chunkGlaSimpleIO] at hbr1 hbr2 hbr3 hbr4 hbr5 hbw ⊢
    exact cgsIO_traceSafeR R bounds q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t
      scale Tt K V BT BK BV s hKBK hBK
      (fun j h1 h2 => hbr1 0 j ⟨h1, h2⟩) (fun j h1 h2 => hbr2 0 j ⟨h1, h2⟩)
      (fun j h1 h2 => hbr3 0 j ⟨h1, h2⟩) (fun j h1 => hbr4 0 j h1)
      (fun j h1 h2 => hbr5 0 j ⟨h1, h2⟩) (fun j h1 h2 => hbw j ⟨h1, h2⟩)
  · -- the rounded Hoare triple
    intro s₀ xs ys zs ws vs hu hx hy hz hw hv
    simp only [chunkGlaSimpleIO] at hx hy hz hw hv ⊢
    have hundef' : ∀ rg oo, s₀.undef rg oo = 0 := fun rg oo => by rw [hu]
    obtain ⟨sF, hexec, hval, hframe⟩ :=
      cgsIO_exec_run q k v h g o s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
        Tt K V BT BK BV s₀ hKBK hBK hBT hundef'
        (by simpa [outOffset, tIndex, vIndex] using hInj (s₀.pids 0) (s₀.pids 1) (s₀.pids 2))
    refine ⟨sF, ?_, ?_, ?_⟩
    · -- termination under `execR R` (the whole body is cast-free)
      rw [cgsIO_execR_eq]; exact hexec
    · -- the readback = the streamed closed form
      intro j hj
      have hoff : outOffset s₀ s_v_h s_v_t BT BV (Lane2D.decode j)
          = s₀.pids 2 * s_v_h + (s₀.pids 1 * BT + j.val / BV) * s_v_t
            + (s₀.pids 0 * BV + j.val % BV) * 1 := by
        simp [outOffset, tIndex, vIndex]
      rw [BlockState.readMemAs_real, ← hoff,
        hval (Lane2D.decode j) (by simpa [active, tIndex, vIndex] using hj),
        cgsIO_glaOutput_eq s₀ q k v h g s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale
          Tt K V BT BK BV hKBK xs ys zs ws vs
          (fun t j' hj' => hx t j' hj') (fun t j' hj' => hy t j' hj')
          (fun t j' hj' => hz t j' hj') (fun t j' hj' => hw t j' hj')
          (fun t j' hj' => hv t j' hj') j hj]
      simp [FloatDType.ofReal]
    · -- the frame: cells outside the write window are untouched
      intro r oo hcond
      refine hframe r oo ?_
      rcases hcond with hne | hno
      · exact Or.inl hne
      · refine Or.inr fun idx hact => ?_
        have := hno (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
          (by simp only [Lane2D.encode_div, Lane2D.encode_mod]
              simpa [active, tIndex, vIndex] using hact)
        simpa [outOffset, tIndex, vIndex, Lane2D.encode_div, Lane2D.encode_mod] using this

end IOFace
