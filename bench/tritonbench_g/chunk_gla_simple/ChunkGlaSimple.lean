import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel

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
theorem chunk_gla_simple_output_summary_general
    (q k v h g o : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV : Nat) (s : BlockState)
    (hKBK : K = BK) (hBK : 0 < BK) (hBT : 0 < BT)
    (hundef : ∀ rg off, s.undef rg off = 0)
    (hInj : Function.Injective
      (fun idx : TileIndex [BT, BV] => outOffset s s_v_h s_v_t BT BV idx)) :
    ComputeCorrect.Realizes
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
