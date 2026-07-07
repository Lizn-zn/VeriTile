import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Examples.AttentionForwardClosedForm

/-!
# `rope_backward_transform` — strict per-kernel correctness

`_triton_rope` (here exercised on the `BACKWARD_PASS = true` branch) applies the
rotary position embedding's backward rotation in place to fused Q/K gradient
buffers: each program owns one row (`program_id(0)`), loads the per-row
`cos`/`sin` half-dim vectors, and for every head rewrites the two rotary halves of
both `q_ptr` and `k_ptr` via the sign-flipped pair `(t1*cos + t2*sin,
t2*cos - t1*sin)` — the transpose of the forward rotation.

## Scope

This file verifies **the Triton kernel itself** — the per-program `@triton.jit`
body. The host launch (the `(n_row,)` grid, the `next_power_of_2` padding choices
`pad_n_qh`/`pad_n_kh`/`pad_hd`, `BLOCK_SIZE`, the contiguity/transpose bookkeeping
in `rope_backward`, and how the runtime composes per-program writes into one
buffer) is the *trusted boundary*, not a proof obligation here. Because the row
position and per-head/per-dim indices are universally quantified, the per-program
statement covers every program of the grid.

## Proof architecture

```
rope_backward_python_backward_output_summary_general  ← TOP THEOREM (genuine)
  a conjunction of four `ComputeCorrect.Realizes` (one per stored output half),
  each with a `WriteMap.writeIf` masking the store to its active lanes; the
  projection-succeeds fact and the per-run `exec` quantification are internalized
  by `Realizes` (discharged via `computeCorrect_of_toAlgKernel` + the surface's
  `toAlgorithm?` reduction). Each conjunct closes on the raw per-store lemma:
  ├─ rope_backward_q0_backward_correct  → ropeBackwardKernelQ0Spec  (q1·cos + q2·sin)
  ├─ rope_backward_q1_backward_correct  → ropeBackwardKernelQ1Spec  (q2·cos − q1·sin)
  ├─ rope_backward_k0_backward_correct  → ropeBackwardKernelK0Spec  (k1·cos + k2·sin)
  └─ rope_backward_k1_backward_correct  → ropeBackwardKernelK1Spec  (k2·cos − k1·sin)
       each reads back one full `[pad_n_qh, pad_hd/2]` store from the REAL
       `triton_rope_surface` (`BACKWARD_PASS = true`) via `rope_backward_body_steps`
       (`triton_rope_surface_toAlgorithm_supported` records the same surface-lowers fact)

rope_backward_body_steps (the load-bearing lemma)
  ├─ triton_rope_surface_backward_body : (surface).body = ropeBackwardBody  (by rfl)
  ├─ 22 prologue assigns stepped via the standalone offset/mask/load recipes
  │    (firstHalf_offsets_eval, firstMask_eval, ptr_plus_offsets_eval,
  │     load_ptr_maskOther_real, …; expandDim wall handled by the *_arange recipes)
  ├─ ifThenElse (constBool true).boolNot takes the else-branch (4 store pairs)
  └─ readback: cross-region peel (q_ptr ≠ k_ptr) + same-region disjoint peel
       (first-half vs second-half offsets) + scatter_readback_prop_masked_nd_of_true

supporting head-slice + per-store track (one Q/K head, one row):
  ├─ rope_backward_q0_head_compute_correct → rope_backward_q0_head_correct
  ├─ rope_backward_q1_head_compute_correct → rope_backward_q1_head_correct
  ├─ rope_backward_k0_head_compute_correct
  └─ rope_backward_k1_head_compute_correct
```

Full-kernel offset disjointness within a half pair is discharged inline (by
`omega`) in the readback peel.

## Modeling boundary

Arithmetic is over `ℝ`, not bit-accurate IEEE float; the `.to(sin_row.dtype)`
register casts erase to the identity at the algorithm layer (post-erasure all
dtypes unify to `ℝ`). `cos`/`sin` are modeled as **precomputed inputs** loaded
from memory, not computed; the backward spec uses them with the
`BACKWARD_PASS = true` sign convention. The top summary is the **full-surface,
full-tile** backward result: every active lane of each of the four
Python-observable stores (Q/K first and second halves) reads back to the genuine
rotary-backward closed form, NOT the kernel's own executed value, against the real
`triton_rope_surface` kernel (no head-slice gap). It is dimension-general over all
strides / shapes / program ids; the concrete Python benchmark shape (`batch=2`,
`seq=4`, `n_qh = n_kh = 8`, `head_dim = 16`, so Q/K row stride `128`, cos/sin row
stride `8`) is just one instantiation. The host launch / `next_power_of_2` padding
(`pad_n_qh`, `pad_hd`, `BLOCK_SIZE`, the contiguity/transpose bookkeeping, grid
composition) remains the trusted boundary, as does `q_ptr ≠ k_ptr` (distinct Q/K
buffers). The `BACKWARD_PASS` flag is modeled by the `Bool` argument;
`@triton.autotune` is not modeled.
-/

namespace VeriTile.Bench.TritonBenchG.RopeBackwardTransform

open VeriTile
open VeriTile.Examples.AttentionForwardClosedForm

set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-! ## Standalone eval recipes for the full-kernel backward proof

The full `triton_rope_surface` kernel builds its tile offsets/masks from the 2D
`tl.arange[:, None]`/`tl.arange[None, :]` (`expandDim`) composition. The generic
`evalOp_expandDim` simp lemma does not fire on `Op.expandDim _ (Op.arange _)`
inside a nested `Option.bind` (the "expandDim wall"), so we fix the concrete
output shapes up front in these standalone recipes and chain them by `simp only`
when stepping the body. These mirror the just-proven forward sibling
(`rope_transform`, PR #276); the backward proof differs only in the executed
`if`-branch (`BACKWARD_PASS = true` ⇒ else-branch) and the per-store sign
convention. -/

/-- `expandDim` axis-1 of an `arange` (the `tl.arange(0, PN)[:, None]` column
broadcast), output shape pinned to `[PN, 1]`. -/
@[simp] theorem evalOp_expandDim_one_arange {PN : Nat} (s : BlockState) :
    @evalOp .nat [PN, 1] (Op.expandDim ⟨1, by simp⟩ (Op.arange PN)) s =
      some ({ data := fun i : TileIndex [PN, 1] => i.1.val } : Tile .nat [PN, 1]) := by
  unfold evalOp; simp [Tile.expandDim, Tile.vec, TileShape.dropInsertedIndex]; rfl

/-- `expandDim` axis-0 of an `arange` (the `tl.arange(0, PH)[None, :]` row
broadcast), output shape pinned to `[1, PH]`. -/
@[simp] theorem evalOp_expandDim_zero_arange {PH : Nat} (s : BlockState) :
    @evalOp .nat [1, PH] (Op.expandDim ⟨0, by simp⟩ (Op.arange PH)) s =
      some ({ data := fun i : TileIndex [1, PH] => i.2.1.val } : Tile .nat [1, PH]) := by
  unfold evalOp; simp [Tile.expandDim, Tile.vec, TileShape.dropInsertedIndex]; rfl

/-- First-half offsets recipe: `arange(PN)[:,None]·hd + arange(PH)[None,:]`
evaluates, lane `idx`, to `idx.row·hd + idx.col`. -/
theorem firstHalf_offsets_eval (s : BlockState) (PN PH hd : Nat) :
    evalOp (Op.add NumericDType.nat Broadcast.nil.consL.consR
      (Op.mul NumericDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.arange PN))
        (Op.constNat hd))
      (Op.expandDim ⟨0, by simp⟩ (Op.arange PH))) s
      = some (⟨fun idx : TileIndex [PN, PH] => idx.1.val * hd + idx.2.1.val⟩ : Tile .nat [PN, PH]) := by
  rw [evalOp_add, evalOp_mul]
  simp only [evalOp_expandDim_one_arange, evalOp_expandDim_zero_arange, evalOp_constNat,
    Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, NumericDType.add, NumericDType.mul, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex]

/-- Second-half offsets recipe: `first_half_offsets + hd//2`. -/
theorem secondHalf_offsets_eval (s : BlockState) (PN PH hd : Nat) (name : RegName)
    (first : Tile .nat [PN, PH])
    (hfirst : s.regs .nat [PN, PH] name = some first) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarR
      (Op.ref .nat [PN, PH] name)
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2))) s
      = some (⟨fun idx : TileIndex [PN, PH] => (first.data idx) + hd / 2⟩ : Tile .nat [PN, PH]) := by
  rw [evalOp_add, evalOp_floorDiv]
  simp only [evalOp_ref, evalOp_constNat, hfirst, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Broadcast.rightIndex, Broadcast.leftIndex, NumericDType.add,
    IntegralDType.floorDiv, Tile.scalar]

/-- First-half mask recipe: `(arange(PN)[:,None] < n) & (arange(PH)[None,:] < hd//2)`. -/
theorem firstMask_eval (s : BlockState) (PN PH n hd : Nat) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.arange PN))
        (Op.constNat n))
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.arange PH))
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2)))) s
      = some (⟨fun idx : TileIndex [PN, PH] =>
          decide (idx.1.val < n) && decide (idx.2.1.val < hd / 2)⟩ : Tile .bool [PN, PH]) := by
  rw [evalOp_boolAnd, evalOp_lt, evalOp_lt, evalOp_floorDiv]
  simp only [evalOp_expandDim_one_arange, evalOp_expandDim_zero_arange, evalOp_constNat,
    Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.cop, Broadcast.rightIndex, Broadcast.leftIndex, Tile.scalar,
    ComparableDType.lt, IntegralDType.floorDiv]
  rfl

/-- `q_ptr + offsets` recipe: a scalar base pointer broadcast-added to a 2D
offset tile yields the per-lane address tile `(region, base + offset)`. -/
theorem ptr_plus_offsets_eval (s : BlockState) (Q : RegionName) (PN PH : Nat)
    (base : Nat) (offs : Tile .nat [PN, PH]) (pname oname : RegName)
    (hp : s.regs .ptr [] pname = some (Tile.scalar (Q.cast, base)))
    (ho : s.regs .nat [PN, PH] oname = some offs) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] pname) (Op.ref .nat [PN, PH] oname)) s
      = some (⟨fun i => (Q.cast, base + offs.data i)⟩ : Tile .ptr [PN, PH]) := by
  rw [evalOp_ptrAdd]
  simp only [evalOp_ref, hp, ho, Option.bind_some]
  refine congrArg some ?_
  ext i
  · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex]
  · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]

/-- Prologue base-pointer recipe: `base_ptr + pid·stride`. -/
theorem ptrbase_mul_eval (s : BlockState) (Q : RegionName) (pidv stride : Nat)
    (hpid : s.regs .nat [] "pid" = some (Tile.scalar pidv)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Q)
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid") (Op.constNat stride))) s
      = some (Tile.scalar (Q.cast, pidv * stride)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase, evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hpid, Option.bind_some]
  refine congrArg some ?_
  ext i
  · simp only [Tile.ptrAdd_data, Tile.scalar, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.mul]
  · simp only [Tile.ptrAdd_data, Tile.scalar, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.mul, Nat.zero_add]

/-- Cos/sin base-pointer recipe: `base_ptr + cos_row_idx·stride`. -/
theorem ptrbase_mul_ref_eval (s : BlockState) (Q : RegionName) (idxv stride : Nat)
    (name : RegName) (hidx : s.regs .nat [] name = some (Tile.scalar idxv)) :
    evalOp (Op.ptrAdd Broadcast.nil (Op.ptrBase Q)
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] name) (Op.constNat stride))) s
      = some (Tile.scalar (Q.cast, idxv * stride)) := by
  rw [evalOp_ptrAdd, evalOp_ptrBase, evalOp_mul]
  simp only [evalOp_ref, evalOp_constNat, hidx, Option.bind_some]
  refine congrArg some ?_
  ext i
  · simp only [Tile.ptrAdd_data, Tile.scalar, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.mul]
  · simp only [Tile.ptrAdd_data, Tile.scalar, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      NumericDType.mul, Nat.zero_add]

/-- `cos_row_idx = pid % sl` recipe. -/
theorem mod_eval (s : BlockState) (pidv sl : Nat)
    (hpid : s.regs .nat [] "pid" = some (Tile.scalar pidv)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid") (Op.constNat sl)) s
      = some (Tile.scalar (pidv % sl)) := by
  rw [evalOp_mod]
  simp only [evalOp_ref, evalOp_constNat, hpid, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.scalar, Tile.bop, Broadcast.leftIndex, Broadcast.rightIndex, IntegralDType.mod]

/-- 1D `cos/sin + cos_offsets` recipe: scalar base broadcast-added to a 1D
offset tile. -/
theorem ptr_plus_offsets1d_eval (s : BlockState) (Q : RegionName) (PH : Nat)
    (base : Nat) (offs : Tile .nat [PH]) (pname oname : RegName)
    (hp : s.regs .ptr [] pname = some (Tile.scalar (Q.cast, base)))
    (ho : s.regs .nat [PH] oname = some offs) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] pname) (Op.ref .nat [PH] oname)) s
      = some (⟨fun i => (Q.cast, base + offs.data i)⟩ : Tile .ptr [PH]) := by
  rw [evalOp_ptrAdd]
  simp only [evalOp_ref, hp, ho, Option.bind_some]
  refine congrArg some ?_
  ext i
  · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex]
  · simp only [Tile.ptrAdd_data, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]

/-- 1D cos/sin mask recipe: `cos_offsets < hd//2`. -/
theorem cosMask_eval (s : BlockState) (PH hd : Nat) (offs : Tile .nat [PH]) (oname : RegName)
    (ho : s.regs .nat [PH] oname = some offs) :
    evalOp (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [PH] oname)
      (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2))) s
      = some (⟨fun i => decide (offs.data i < hd / 2)⟩ : Tile .bool [PH]) := by
  rw [evalOp_lt, evalOp_floorDiv]
  simp only [evalOp_ref, evalOp_constNat, ho, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.cop, Tile.bop, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    ComparableDType.lt, IntegralDType.floorDiv]
  rfl

/-- Masked-other `.ptr` load recipe (the `tl.load(..., other=0)` form used for
all six rope loads): per lane, reads memory where the mask holds, else `0`. -/
theorem load_ptr_maskOther_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (s : BlockState)
    (ptrs : Tile .ptr shape) (masks : Tile .bool shape)
    (hp : evalOp ptrOp s = some ptrs) (hm : evalOp maskOp s = some masks) :
    evalOp (.load .real (.ptr ptrOp) (.maskOther maskOp ((Op.const 0).broadcast shape))) s
      = some ⟨fun i => if masks.data i then
          some (s.readMem (ptrs.data i).1 (ptrs.data i).2) else some 0⟩ := by
  simp only [evalOp, hp, hm]
  refine congrArg some ?_
  ext i
  simp only [BlockState.readMemValue_real]
  cases hmi : masks.data i <;> simp [hmi]

/-- `if not BACKWARD_PASS` with `BACKWARD_PASS = true` takes the else-branch. -/
theorem ifThenElse_notTrue_step (thenB elseB : List Stmt) (st : BlockState) :
    stepStmt (Stmt.ifThenElse (Op.constBool Bool.true).boolNot thenB elseB) st
      = stepStmts elseB st := by
  unfold stepStmt
  simp only [evalOp, Tile.uop, Tile.scalar]
  norm_num

/-- Singleton-list form: the trailing `if`-statement of the body takes its
else-branch (the `BACKWARD_PASS = true` path). -/
theorem stepStmts_singleton_ifThenElse_notTrue (thenB elseB : List Stmt) (st : BlockState) :
    stepStmts [Stmt.ifThenElse (Op.constBool Bool.true).boolNot thenB elseB] st
      = stepStmts elseB st := by
  conv_lhs => unfold stepStmts
  rw [ifThenElse_notTrue_step]
  cases stepStmts elseB st <;> simp

/-- `new_*_tile_1 = t1·cos + t2·sin` value recipe (first-half backward store). -/
theorem newAdd_value_eval (s : BlockState) (PN PH : Nat)
    (t1 t2 : Tile .real [PN, PH]) (cosrow sinrow : Tile .real [PH])
    (n1 n2 cn sn : RegName)
    (h1 : s.regs .real [PN, PH] n1 = some t1)
    (h2 : s.regs .real [PN, PH] n2 = some t2)
    (hc : s.regs .real [PH] cn = some cosrow)
    (hsr : s.regs .real [PH] sn = some sinrow) :
    evalOp (Op.add NumericDType.real Broadcast.nil.consSame.consSame
      (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
        (Op.ref .real [PN, PH] n1) (Op.ref .real [PH] cn))
      (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
        (Op.ref .real [PN, PH] n2) (Op.ref .real [PH] sn))) s
      = some (⟨fun idx : TileIndex [PN, PH] =>
          NumericDType.real.add
            (NumericDType.real.mul (t1.data idx) (cosrow.data (idx.2.1, idx.2.2)))
            (NumericDType.real.mul (t2.data idx) (sinrow.data (idx.2.1, idx.2.2)))⟩
            : Tile .real [PN, PH]) := by
  rw [evalOp_add, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, h1, h2, hc, hsr, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
    Broadcast.leftIndex_leadR, Broadcast.rightIndex_leadR]

/-- `new_*_tile_2 = t1·cos − t2·sin` value recipe (second-half backward store). -/
theorem newSub_value_eval (s : BlockState) (PN PH : Nat)
    (t1 t2 : Tile .real [PN, PH]) (cosrow sinrow : Tile .real [PH])
    (n1 n2 cn sn : RegName)
    (h1 : s.regs .real [PN, PH] n1 = some t1)
    (h2 : s.regs .real [PN, PH] n2 = some t2)
    (hc : s.regs .real [PH] cn = some cosrow)
    (hsr : s.regs .real [PH] sn = some sinrow) :
    evalOp (Op.sub NumericDType.real Broadcast.nil.consSame.consSame
      (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
        (Op.ref .real [PN, PH] n1) (Op.ref .real [PH] cn))
      (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
        (Op.ref .real [PN, PH] n2) (Op.ref .real [PH] sn))) s
      = some (⟨fun idx : TileIndex [PN, PH] =>
          NumericDType.real.sub
            (NumericDType.real.mul (t1.data idx) (cosrow.data (idx.2.1, idx.2.2)))
            (NumericDType.real.mul (t2.data idx) (sinrow.data (idx.2.1, idx.2.2)))⟩
            : Tile .real [PN, PH]) := by
  rw [evalOp_sub, evalOp_mul, evalOp_mul]
  simp only [evalOp_ref, h1, h2, hc, hsr, Option.bind_some]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop_data, Broadcast.leftIndex_consSame, Broadcast.rightIndex_consSame,
    Broadcast.leftIndex_leadR, Broadcast.rightIndex_leadR]

/-- Masked `.ptr` store step: produces the masked `writeMemTyped` foldl. -/
theorem stepStmt_store_ptr_mask_eq {shape : TileShape}
    (ptrOp : Op .ptr shape) (valOp : Op .real shape) (maskOp : Op .bool shape)
    (s : BlockState) (ptrs : Tile .ptr shape) (vals : Tile .real shape) (masks : Tile .bool shape)
    (hv : evalOp valOp s = some vals)
    (hp : evalOp ptrOp s = some ptrs)
    (hm : evalOp maskOp s = some masks) :
    stepStmt (Stmt.store .real shape (MemAccess.ptr ptrOp) valOp (MaskOpt.mask maskOp)) s
      = some ((TileShape.allIndices shape).foldl
          (fun acc i =>
            if masks.data i then
              acc.writeMemTyped .real (ptrs.data i).1 (ptrs.data i).2 (vals.data i)
            else acc) s) := by
  unfold stepStmt
  simp only [hv, hp, hm, Option.map_some]
  rfl

/-- Bool-predicate intra-region offset-disjointness readback (the rope-style
first-half read commuting past the second-half masked store-foldl). -/
theorem foldl_writeMem_same_region_disjoint_offsets_readMem_bool {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (P : α → Bool) (l : List α) (s : BlockState)
    (off : Nat) (hOff : ∀ k, k ∈ l → P k → off ≠ offsetFn k) :
    BlockState.readMem ((l.foldl
        (fun acc k =>
          if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s))
      region off
      = s.readMem region off := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      have htl : ∀ k, k ∈ tl → P k → off ≠ offsetFn k :=
        fun k hk hPk => hOff k (List.mem_cons_of_mem hd hk) hPk
      by_cases hhd : P hd = Bool.true
      · simp only [hhd, ite_true]
        rw [ih _ htl, BlockState.writeMem_readMem, if_neg]
        rintro ⟨_, hOffEq⟩
        exact (hOff hd List.mem_cons_self hhd) hOffEq
      · rw [Bool.not_eq_true] at hhd
        simp only [hhd, Bool.false_eq_true]
        exact ih _ htl

/-- Faithful transcription of `rope_backward_transform.py`'s `_triton_rope`. -/
def triton_rope_surface
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) :
    ComputeKernel := triton {
    pid = tl.program_id(0)
    q_ptr = q_ptr + pid * $(q_row_stride)
    k_ptr = k_ptr + pid * $(k_row_stride)
    cos_row_idx = pid % $(sl)
    cos = cos + cos_row_idx * $(cos_row_stride)
    sin = sin + cos_row_idx * $(sin_row_stride)
    cos_offsets = tl.arange(0, $(pad_hd) // $(2))
    cos_mask = cos_offsets < $(hd) // $(2)
    cos_row = tl.load(cos + cos_offsets, mask=cos_mask, other=0)
    sin_row = tl.load(sin + cos_offsets, mask=cos_mask, other=0)
    first_half_q_offsets = tl.arange(0, $(pad_n_qh))[:, None] * $(hd) +
      tl.arange(0, $(pad_hd) // $(2))[None, :]
    first_half_k_offsets = tl.arange(0, $(pad_n_kh))[:, None] * $(hd) +
      tl.arange(0, $(pad_hd) // $(2))[None, :]
    first_q_mask = (tl.arange(0, $(pad_n_qh))[:, None] < $(n_qh)) &
      (tl.arange(0, $(pad_hd) // $(2))[None, :] < $(hd) // $(2))
    first_k_mask = (tl.arange(0, $(pad_n_kh))[:, None] < $(n_kh)) &
      (tl.arange(0, $(pad_hd) // $(2))[None, :] < $(hd) // $(2))
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask,
      other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask,
      other=0).to(sin_row.dtype)
    second_half_q_offsets = first_half_q_offsets + $(hd) // $(2)
    second_half_k_offsets = first_half_k_offsets + $(hd) // $(2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask,
      other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask,
      other=0).to(sin_row.dtype)
    if not BACKWARD_PASS {
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    } else {
    new_q_tile_1 = q_tile_1 * cos_row + q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row - q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
    new_k_tile_1 = k_tile_1 * cos_row + k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row - k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    }
}

/-- The full backward RoPE transform surface lowers to the algorithm layer,
including Q/K, both halves, and the `BACKWARD_PASS` branch. -/
theorem triton_rope_surface_toAlgorithm_supported
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (BACKWARD_PASS : Bool) :
    ∃ alg, (triton_rope_surface q_ptr k_ptr cos sin q_row_stride k_row_stride
      cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
      pad_hd BLOCK_SIZE BACKWARD_PASS).toAlgorithm? = Except.ok alg := by
  simp [triton_rope_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-- Proof-oriented one-Q-head first-half backward slice of
`rope_backward_transform.py`'s `_triton_rope`.

This fixes one Q head and one row in the `BACKWARD_PASS=True` branch and proves
the first-half store: `q0' = q0 * cos + q1 * sin`. -/
def rope_backward_q0_head
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  dim = tl.arange(0, $(BLOCK_HALF))
  q_base = Q + pid * $(q_row_stride) + $(HEAD_IDX) * $(hd)
  cos_base = COS + $(COS_ROW_IDX) * $(cos_row_stride)
  sin_base = SIN + $(COS_ROW_IDX) * $(sin_row_stride)
  q0 = tl.load(q_base + dim,
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)), other=0).to(sin_row.dtype)
  q1 = tl.load(q_base + dim + $(HEAD_HALF),
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)), other=0).to(sin_row.dtype)
  cos_row = tl.load(cos_base + dim, mask=dim < $(HEAD_HALF), other=0)
  sin_row = tl.load(sin_base + dim, mask=dim < $(HEAD_HALF), other=0)
  out = q0 * cos_row + q1 * sin_row
  tl.store(q_base + dim, out,
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)))
}

def rowIndex (s : BlockState) : Nat :=
  s.pids 0

def dimIndex (i : Fin BLOCK_HALF) : Nat :=
  i.val

def active (HEAD_IDX n_qh HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : Prop :=
  HEAD_IDX < n_qh ∧ dimIndex i < HEAD_HALF

instance activeDecidable (HEAD_IDX n_qh HEAD_HALF : Nat) (i : Fin BLOCK_HALF) :
    Decidable (active HEAD_IDX n_qh HEAD_HALF i) := by
  unfold active
  infer_instance

def qOffset
    (s : BlockState) (HEAD_IDX q_row_stride hd : Nat) (dim : Nat) : Nat :=
  rowIndex s * q_row_stride + HEAD_IDX * hd + dim

def cosOffset (COS_ROW_IDX cos_row_stride : Nat) (i : Fin BLOCK_HALF) : Nat :=
  COS_ROW_IDX * cos_row_stride + dimIndex i

def sinOffset (COS_ROW_IDX sin_row_stride : Nat) (i : Fin BLOCK_HALF) : Nat :=
  COS_ROW_IDX * sin_row_stride + dimIndex i

noncomputable def ropeBackwardQ0Spec
    (s : BlockState) (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) *
    s.readMem COS (cosOffset COS_ROW_IDX cos_row_stride i) +
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i + HEAD_HALF)) *
    s.readMem SIN (sinOffset COS_ROW_IDX sin_row_stride i)

/-- Algorithm-layer correctness for the one-Q-head backward RoPE store. -/
theorem rope_backward_q0_head_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s HEAD_IDX q_row_stride hd (dimIndex i)))
    (hExec : exec (rope_backward_q0_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) =
        if active HEAD_IDX n_qh HEAD_HALF i then
          ropeBackwardQ0Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
            cos_row_stride sin_row_stride hd HEAD_HALF i
        else
          s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 0 * q_row_stride + HEAD_IDX * hd + idx.1.val) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qOffset, rowIndex, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rope_backward_q0_head, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [qOffset, rowIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : HEAD_IDX < n_qh
    · by_cases hDim : i.val < HEAD_HALF
      · simp [active, ropeBackwardQ0Spec, qOffset, cosOffset, sinOffset,
              rowIndex, dimIndex, hHead, hDim, Option.bind, Option.map]
        left
        rw [Nat.add_assoc]
      · simp [active, dimIndex, hHead, hDim]
    · simp [active, hHead]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the one-Q-head backward RoPE store. -/
theorem rope_backward_q0_head_compute_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s HEAD_IDX q_row_stride hd (dimIndex i))) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_q0_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_qh HEAD_HALF i)
        (fun i => (Q, qOffset s HEAD_IDX q_row_stride hd (dimIndex i))))
      (expected := fun i =>
        ropeBackwardQ0Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_backward_q0_head]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_backward_q0_head_correct Q COS SIN HEAD_IDX COS_ROW_IDX
    q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF BLOCK_HALF
    s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## Second-half backward Q store (`new_q_tile_2`) -/

/-- Slice for the backward second-half Q store:
`q1' = q1 * cos - q0 * sin` at `second_half_q_offsets`. -/
def rope_backward_q1_head
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(0)
  dim = tl.arange(0, $(BLOCK_HALF))
  q_base = Q + pid * $(q_row_stride) + $(HEAD_IDX) * $(hd)
  cos_base = COS + $(COS_ROW_IDX) * $(cos_row_stride)
  sin_base = SIN + $(COS_ROW_IDX) * $(sin_row_stride)
  q0 = tl.load(q_base + dim,
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)), other=0).to(sin_row.dtype)
  q1 = tl.load(q_base + dim + $(HEAD_HALF),
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)), other=0).to(sin_row.dtype)
  cos_row = tl.load(cos_base + dim, mask=dim < $(HEAD_HALF), other=0)
  sin_row = tl.load(sin_base + dim, mask=dim < $(HEAD_HALF), other=0)
  out = q1 * cos_row - q0 * sin_row
  tl.store(q_base + dim + $(HEAD_HALF), out,
    mask=($(HEAD_IDX) < $(n_qh)) and (dim < $(HEAD_HALF)))
}

def q1WriteOffset
    (s : BlockState) (HEAD_IDX q_row_stride hd HEAD_HALF : Nat)
    (i : Fin BLOCK_HALF) : Nat :=
  s.pid * q_row_stride + HEAD_IDX * hd + dimIndex i + HEAD_HALF

noncomputable def ropeBackwardQ1Spec
    (s : BlockState) (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      HEAD_HALF : Nat) (i : Fin BLOCK_HALF) : ℝ :=
  s.readMem Q (q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i) *
    s.readMem COS (cosOffset COS_ROW_IDX cos_row_stride i) -
  s.readMem Q (qOffset s HEAD_IDX q_row_stride hd (dimIndex i)) *
    s.readMem SIN (sinOffset COS_ROW_IDX sin_row_stride i)

theorem rope_backward_q1_head_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i))
    (hExec : exec (rope_backward_q1_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF) s = some s') :
    ∀ i : Fin BLOCK_HALF,
      s'.readMem Q (q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i) =
        if active HEAD_IDX n_qh HEAD_HALF i then
          ropeBackwardQ1Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
            cos_row_stride sin_row_stride hd HEAD_HALF i
        else
          s.readMem Q (q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i) := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [BLOCK_HALF] =>
        s.pids 0 * q_row_stride + HEAD_IDX * hd + idx.1.val + HEAD_HALF) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simp only [q1WriteOffset, rowIndex, dimIndex] at *
      exact h
    cases a; cases b; simp only at hab; cases hab; rfl
  by_cases hBH : 0 < BLOCK_HALF
  · simp [exec, rope_backward_q1_head, stepStmts, stepStmt, evalOp, evalOp.eq_def,
          Option.bind, Option.map, Tile.bop, Tile.ptrAdd, Tile.uop,
          NumericDType.add, NumericDType.mul, NumericDType.sub,
          ComparableDType.lt, hBH] at hExec
    rw [← hExec]
    simp only [q1WriteOffset, rowIndex, dimIndex]
    rw [BlockState.scatter_readback_prop_masked_nd _ _ _ _ hRawInj
      (i, PUnit.unit)]
    by_cases hHead : HEAD_IDX < n_qh
    · by_cases hDim : i.val < HEAD_HALF
      · simp [active, ropeBackwardQ1Spec, q1WriteOffset, qOffset, cosOffset,
              sinOffset, rowIndex, dimIndex, hHead, hDim, Option.bind, Option.map]
      · simp [active, dimIndex, hHead, hDim]
    · simp [active, hHead]
  · exact False.elim (hBH (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

theorem rope_backward_q1_head_compute_correct
    (Q COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX q_row_stride cos_row_stride sin_row_stride hd
      n_qh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i)) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_q1_head Q COS SIN HEAD_IDX COS_ROW_IDX
        q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF
        BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_qh HEAD_HALF i)
        (fun i => (Q, q1WriteOffset s HEAD_IDX q_row_stride hd HEAD_HALF i)))
      (expected := fun i =>
        ropeBackwardQ1Spec s Q COS SIN HEAD_IDX COS_ROW_IDX q_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [rope_backward_q1_head]
  intro s0 s' hExec hs0
  subst s0
  intro i hActive
  have h := rope_backward_q1_head_correct Q COS SIN HEAD_IDX COS_ROW_IDX
    q_row_stride cos_row_stride sin_row_stride hd n_qh HEAD_HALF BLOCK_HALF
    s s' hOutInj hExec i
  simpa [hActive] using h

/-! ## K-side backward coverage (alias of Q-side, with K region and `n_kh`) -/

abbrev rope_backward_k0_head
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat) :
    ComputeKernel :=
  rope_backward_q0_head K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
    cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF

abbrev rope_backward_k1_head
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat) :
    ComputeKernel :=
  rope_backward_q1_head K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
    cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF

theorem rope_backward_k0_head_compute_correct
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF => qOffset s HEAD_IDX k_row_stride hd (dimIndex i))) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_k0_head K COS SIN HEAD_IDX COS_ROW_IDX
        k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_kh HEAD_HALF i)
        (fun i => (K, qOffset s HEAD_IDX k_row_stride hd (dimIndex i))))
      (expected := fun i =>
        ropeBackwardQ0Spec s K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) :=
  rope_backward_q0_head_compute_correct K COS SIN HEAD_IDX COS_ROW_IDX
    k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF s
    hOutInj

theorem rope_backward_k1_head_compute_correct
    (K COS SIN : RegionName)
    (HEAD_IDX COS_ROW_IDX k_row_stride cos_row_stride sin_row_stride hd
      n_kh HEAD_HALF BLOCK_HALF : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin BLOCK_HALF =>
        q1WriteOffset s HEAD_IDX k_row_stride hd HEAD_HALF i)) :
    ComputeCorrect.Realizes
      (kernel := rope_backward_k1_head K COS SIN HEAD_IDX COS_ROW_IDX
        k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun i : Fin BLOCK_HALF => active HEAD_IDX n_kh HEAD_HALF i)
        (fun i => (K, q1WriteOffset s HEAD_IDX k_row_stride hd HEAD_HALF i)))
      (expected := fun i =>
        ropeBackwardQ1Spec s K COS SIN HEAD_IDX COS_ROW_IDX k_row_stride
          cos_row_stride sin_row_stride hd HEAD_HALF i) :=
  rope_backward_q1_head_compute_correct K COS SIN HEAD_IDX COS_ROW_IDX
    k_row_stride cos_row_stride sin_row_stride hd n_kh HEAD_HALF BLOCK_HALF s
    hOutInj

/-! ## Full-kernel Q first-half store correctness (`BACKWARD_PASS = true`)

Per the #139 audit, the slice proofs above (one Q head, one row) aren't
sufficient. This section closes the Q first-half store for the entire
`triton_rope_surface` kernel under the `BACKWARD_PASS = true` branch.

The kernel issues four stores in sequence:
  1. Q at `first_half_q_offsets`  (the target whose readback we prove)
  2. Q at `second_half_q_offsets = first_half_q_offsets + hd / 2`
  3. K at `first_half_k_offsets`
  4. K at `second_half_k_offsets = first_half_k_offsets + hd / 2`

For the Q first-half readback we
* strip the K-side foldls via `foldl_writeMem_const_region_bool_masked_readMem_other`
  (cross-region: `Q ≠ K`);
* strip the Q second-half foldl via
  `foldl_writeMem_same_region_disjoint_offsets_readMem_bool` (intra-region,
  disjoint offsets thanks to the `+ hd / 2` shift);
* finally apply `scatter_readback_prop_masked_nd_of_true` to the Q first-half foldl.

Disjointness uses `hd / 2 + hd / 2 ≤ hd` and the active-region bound
`d < hd / 2` to rule out wrap-around between adjacent heads. -/

/-- Tile-level Q first-half offset in the full `triton_rope_surface` kernel.
Tile shape is `[pad_n_qh, pad_hd / 2]`. -/
def qFullFirstOffset
    (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Nat :=
  s.pids 0 * q_row_stride + idx.1.val * hd + idx.2.1.val

/-- Tile-level Q second-half offset (writes go here in store #2). -/
def qFullSecondOffset
    (s : BlockState) (q_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Nat :=
  s.pids 0 * q_row_stride + idx.1.val * hd + idx.2.1.val + hd / 2

/-- Cos offset for the full kernel's first-half Q store. -/
def cosFullFirstOffset
    (s : BlockState) (sl cos_row_stride : Nat)
    (idx : TileIndex [pad_hd_half]) : Nat :=
  s.pids 0 % sl * cos_row_stride + idx.1.val

/-- Sin offset for the full kernel's first-half Q store. -/
def sinFullFirstOffset
    (s : BlockState) (sl sin_row_stride : Nat)
    (idx : TileIndex [pad_hd_half]) : Nat :=
  s.pids 0 % sl * sin_row_stride + idx.1.val

/-- Active predicate for the Q first-half store of `triton_rope_surface`. -/
def activeQFull (n_qh hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : Prop :=
  idx.1.val < n_qh ∧ idx.2.1.val < hd / 2

instance activeQFullDecidable (n_qh hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) :
    Decidable (activeQFull n_qh hd idx) := by
  unfold activeQFull
  infer_instance

/-- Specification for the full kernel's Q first-half output, under
`BACKWARD_PASS = true` (i.e. `new_q_tile_1 = q_tile_1 * cos + q_tile_2 * sin`). -/
noncomputable def ropeBackwardKernelQ0Spec
    (s : BlockState) (q_ptr cos sin : RegionName)
    (q_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : ℝ :=
  s.readMem q_ptr (qFullFirstOffset s q_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) +
  s.readMem q_ptr (qFullSecondOffset s q_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-! ## Full-kernel Q second-half store correctness (`BACKWARD_PASS = true`)

Mirrors the Q first-half proof: target offset is `qFullSecondOffset`. The
foldl-stack is `Q1 . Q2 . K1 . K2` (innermost to outermost); we peel K2, K1
(cross-region), then apply `scatter_readback_prop_masked_nd_of_true` to Q2, and in
the inactive case peel Q1 via offset-disjointness. -/

/-- Spec for the full kernel's Q second-half output under
`BACKWARD_PASS = true`: `new_q_tile_2 = q_tile_2 * cos - q_tile_1 * sin`. -/
noncomputable def ropeBackwardKernelQ1Spec
    (s : BlockState) (q_ptr cos sin : RegionName)
    (q_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_qh, pad_hd_half]) : ℝ :=
  s.readMem q_ptr (qFullSecondOffset s q_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) -
  s.readMem q_ptr (qFullFirstOffset s q_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-! ## Full-kernel K first-half store correctness (`BACKWARD_PASS = true`)

Target region is `k_ptr` at `kFullFirstOffset` (= the `first_half_k_offsets`
write). Foldl-stack is `Q1 . Q2 . K1 . K2`; for reading at K1 offsets we
peel K2 via intra-region offset disjointness (K2 = K1 + hd/2), then apply
scatter to K1, and strip Q1, Q2 in the inactive case via cross-region. -/

/-- Tile-level K first-half offset (target of store #3). -/
def kFullFirstOffset
    (s : BlockState) (k_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Nat :=
  s.pids 0 * k_row_stride + idx.1.val * hd + idx.2.1.val

/-- Tile-level K second-half offset (target of store #4). -/
def kFullSecondOffset
    (s : BlockState) (k_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Nat :=
  s.pids 0 * k_row_stride + idx.1.val * hd + idx.2.1.val + hd / 2

/-- Active predicate for the K-side stores. -/
def activeKFull (n_kh hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : Prop :=
  idx.1.val < n_kh ∧ idx.2.1.val < hd / 2

instance activeKFullDecidable (n_kh hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) :
    Decidable (activeKFull n_kh hd idx) := by
  unfold activeKFull
  infer_instance

/-- Spec for the full kernel's K first-half output, under
`BACKWARD_PASS = true`: `new_k_tile_1 = k_tile_1 * cos + k_tile_2 * sin`. -/
noncomputable def ropeBackwardKernelK0Spec
    (s : BlockState) (k_ptr cos sin : RegionName)
    (k_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : ℝ :=
  s.readMem k_ptr (kFullFirstOffset s k_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) +
  s.readMem k_ptr (kFullSecondOffset s k_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))

/-! ## Full-kernel K second-half store correctness (`BACKWARD_PASS = true`)

Target offset is `kFullSecondOffset` (= store #4, outermost foldl). We
apply scatter to K2 directly, then peel K1, Q2, Q1 in the inactive case. -/

/-- Spec for the full kernel's K second-half output, under
`BACKWARD_PASS = true`: `new_k_tile_2 = k_tile_2 * cos - k_tile_1 * sin`. -/
noncomputable def ropeBackwardKernelK1Spec
    (s : BlockState) (k_ptr cos sin : RegionName)
    (k_row_stride sl cos_row_stride sin_row_stride hd : Nat)
    (idx : TileIndex [pad_n_kh, pad_hd_half]) : ℝ :=
  s.readMem k_ptr (kFullSecondOffset s k_row_stride hd idx) *
    s.readMem cos (cosFullFirstOffset s sl cos_row_stride (idx.2.1, idx.2.2)) -
  s.readMem k_ptr (kFullFirstOffset s k_row_stride hd idx) *
    s.readMem sin (sinFullFirstOffset s sl sin_row_stride (idx.2.1, idx.2.2))
/-! ## Full-kernel backward body and store-peeling steps

The explicit `ropeBackwardBody` is definitionally the body of
`triton_rope_surface … Bool.true`. `rope_backward_body_steps` steps the entire
prologue via the standalone recipes above, takes the `BACKWARD_PASS = true`
else-branch (four `(assign, store)` pairs), and reads back each of the four
Python-observable stores through the foldl-stack via cross-region
(`q_ptr ≠ k_ptr`) and intra-region disjoint-offset peeling. -/

def ropeBackwardBody
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl n_qh n_kh hd pad_n_qh pad_n_kh pad_hd : Nat) : List Stmt :=
  [ Stmt.assign .nat [] "pid" (Op.programId 0),
    Stmt.assign .ptr [] "q_ptr" (Op.ptrAdd Broadcast.nil (Op.ptrBase q_ptr)
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid") (Op.constNat q_row_stride))),
    Stmt.assign .ptr [] "k_ptr" (Op.ptrAdd Broadcast.nil (Op.ptrBase k_ptr)
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "pid") (Op.constNat k_row_stride))),
    Stmt.assign .nat [] "cos_row_idx"
      (Op.mod IntegralDType.nat Broadcast.nil (Op.ref .nat [] "pid") (Op.constNat sl)),
    Stmt.assign .ptr [] "cos" (Op.ptrAdd Broadcast.nil (Op.ptrBase cos)
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "cos_row_idx") (Op.constNat cos_row_stride))),
    Stmt.assign .ptr [] "sin" (Op.ptrAdd Broadcast.nil (Op.ptrBase sin)
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref .nat [] "cos_row_idx") (Op.constNat sin_row_stride))),
    Stmt.assign .nat [pad_hd / 2] "cos_offsets" (Op.arange (pad_hd / 2)),
    Stmt.assign .bool [pad_hd / 2] "cos_mask"
      (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref .nat [pad_hd / 2] "cos_offsets")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2))),
    Stmt.assign .real [pad_hd / 2] "cos_row"
      (Op.load .real (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "cos")
          (Op.ref .nat [pad_hd / 2] "cos_offsets")))
        (MaskOpt.maskOther (Op.ref .bool [pad_hd / 2] "cos_mask") ((Op.const 0).broadcast [pad_hd / 2]))),
    Stmt.assign .real [pad_hd / 2] "sin_row"
      (Op.load .real (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "sin")
          (Op.ref .nat [pad_hd / 2] "cos_offsets")))
        (MaskOpt.maskOther (Op.ref .bool [pad_hd / 2] "cos_mask") ((Op.const 0).broadcast [pad_hd / 2]))),
    Stmt.assign .nat [pad_n_qh, pad_hd / 2] "first_half_q_offsets"
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.arange pad_n_qh))
          (Op.constNat hd))
        (Op.expandDim ⟨0, by simp⟩ (Op.arange (pad_hd / 2)))),
    Stmt.assign .nat [pad_n_kh, pad_hd / 2] "first_half_k_offsets"
      (Op.add NumericDType.nat Broadcast.nil.consL.consR
        (Op.mul NumericDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.arange pad_n_kh))
          (Op.constNat hd))
        (Op.expandDim ⟨0, by simp⟩ (Op.arange (pad_hd / 2)))),
    Stmt.assign .bool [pad_n_qh, pad_hd / 2] "first_q_mask"
      (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.arange pad_n_qh))
          (Op.constNat n_qh))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.arange (pad_hd / 2)))
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2)))),
    Stmt.assign .bool [pad_n_kh, pad_hd / 2] "first_k_mask"
      (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.arange pad_n_kh))
          (Op.constNat n_kh))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.arange (pad_hd / 2)))
          (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2)))),
    Stmt.assign .real [pad_n_qh, pad_hd / 2] "q_tile_1"
      (Op.load .real (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "q_ptr")
          (Op.ref .nat [pad_n_qh, pad_hd / 2] "first_half_q_offsets")))
        (MaskOpt.maskOther (Op.ref .bool [pad_n_qh, pad_hd / 2] "first_q_mask")
          ((Op.const 0).broadcast [pad_n_qh, pad_hd / 2]))),
    Stmt.assign .real [pad_n_kh, pad_hd / 2] "k_tile_1"
      (Op.load .real (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "k_ptr")
          (Op.ref .nat [pad_n_kh, pad_hd / 2] "first_half_k_offsets")))
        (MaskOpt.maskOther (Op.ref .bool [pad_n_kh, pad_hd / 2] "first_k_mask")
          ((Op.const 0).broadcast [pad_n_kh, pad_hd / 2]))),
    Stmt.assign .nat [pad_n_qh, pad_hd / 2] "second_half_q_offsets"
      (Op.add NumericDType.nat Broadcast.scalarR
        (Op.ref .nat [pad_n_qh, pad_hd / 2] "first_half_q_offsets")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2))),
    Stmt.assign .nat [pad_n_kh, pad_hd / 2] "second_half_k_offsets"
      (Op.add NumericDType.nat Broadcast.scalarR
        (Op.ref .nat [pad_n_kh, pad_hd / 2] "first_half_k_offsets")
        (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat hd) (Op.constNat 2))),
    Stmt.assign .bool [pad_n_qh, pad_hd / 2] "second_q_mask"
      (Op.ref .bool [pad_n_qh, pad_hd / 2] "first_q_mask"),
    Stmt.assign .bool [pad_n_kh, pad_hd / 2] "second_k_mask"
      (Op.ref .bool [pad_n_kh, pad_hd / 2] "first_k_mask"),
    Stmt.assign .real [pad_n_qh, pad_hd / 2] "q_tile_2"
      (Op.load .real (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "q_ptr")
          (Op.ref .nat [pad_n_qh, pad_hd / 2] "second_half_q_offsets")))
        (MaskOpt.maskOther (Op.ref .bool [pad_n_qh, pad_hd / 2] "second_q_mask")
          ((Op.const 0).broadcast [pad_n_qh, pad_hd / 2]))),
    Stmt.assign .real [pad_n_kh, pad_hd / 2] "k_tile_2"
      (Op.load .real (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "k_ptr")
          (Op.ref .nat [pad_n_kh, pad_hd / 2] "second_half_k_offsets")))
        (MaskOpt.maskOther (Op.ref .bool [pad_n_kh, pad_hd / 2] "second_k_mask")
          ((Op.const 0).broadcast [pad_n_kh, pad_hd / 2]))),
    Stmt.ifThenElse (Op.constBool Bool.true).boolNot
      [ Stmt.assign .real [pad_n_qh, pad_hd / 2] "new_q_tile_1"
          (Op.sub NumericDType.real Broadcast.nil.consSame.consSame
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_qh, pad_hd / 2] "q_tile_1") (Op.ref .real [pad_hd / 2] "cos_row"))
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_qh, pad_hd / 2] "q_tile_2") (Op.ref .real [pad_hd / 2] "sin_row"))),
        Stmt.store .real [pad_n_qh, pad_hd / 2]
          (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "q_ptr")
            (Op.ref .nat [pad_n_qh, pad_hd / 2] "first_half_q_offsets")))
          (Op.ref .real [pad_n_qh, pad_hd / 2] "new_q_tile_1")
          (MaskOpt.mask (Op.ref .bool [pad_n_qh, pad_hd / 2] "first_q_mask")),
        Stmt.assign .real [pad_n_qh, pad_hd / 2] "new_q_tile_2"
          (Op.add NumericDType.real Broadcast.nil.consSame.consSame
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_qh, pad_hd / 2] "q_tile_2") (Op.ref .real [pad_hd / 2] "cos_row"))
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_qh, pad_hd / 2] "q_tile_1") (Op.ref .real [pad_hd / 2] "sin_row"))),
        Stmt.store .real [pad_n_qh, pad_hd / 2]
          (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "q_ptr")
            (Op.ref .nat [pad_n_qh, pad_hd / 2] "second_half_q_offsets")))
          (Op.ref .real [pad_n_qh, pad_hd / 2] "new_q_tile_2")
          (MaskOpt.mask (Op.ref .bool [pad_n_qh, pad_hd / 2] "second_q_mask")),
        Stmt.assign .real [pad_n_kh, pad_hd / 2] "new_k_tile_1"
          (Op.sub NumericDType.real Broadcast.nil.consSame.consSame
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_kh, pad_hd / 2] "k_tile_1") (Op.ref .real [pad_hd / 2] "cos_row"))
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_kh, pad_hd / 2] "k_tile_2") (Op.ref .real [pad_hd / 2] "sin_row"))),
        Stmt.store .real [pad_n_kh, pad_hd / 2]
          (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "k_ptr")
            (Op.ref .nat [pad_n_kh, pad_hd / 2] "first_half_k_offsets")))
          (Op.ref .real [pad_n_kh, pad_hd / 2] "new_k_tile_1")
          (MaskOpt.mask (Op.ref .bool [pad_n_kh, pad_hd / 2] "first_k_mask")),
        Stmt.assign .real [pad_n_kh, pad_hd / 2] "new_k_tile_2"
          (Op.add NumericDType.real Broadcast.nil.consSame.consSame
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_kh, pad_hd / 2] "k_tile_2") (Op.ref .real [pad_hd / 2] "cos_row"))
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_kh, pad_hd / 2] "k_tile_1") (Op.ref .real [pad_hd / 2] "sin_row"))),
        Stmt.store .real [pad_n_kh, pad_hd / 2]
          (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "k_ptr")
            (Op.ref .nat [pad_n_kh, pad_hd / 2] "second_half_k_offsets")))
          (Op.ref .real [pad_n_kh, pad_hd / 2] "new_k_tile_2")
          (MaskOpt.mask (Op.ref .bool [pad_n_kh, pad_hd / 2] "second_k_mask")) ]
      [ Stmt.assign .real [pad_n_qh, pad_hd / 2] "new_q_tile_1"
          (Op.add NumericDType.real Broadcast.nil.consSame.consSame
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_qh, pad_hd / 2] "q_tile_1") (Op.ref .real [pad_hd / 2] "cos_row"))
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_qh, pad_hd / 2] "q_tile_2") (Op.ref .real [pad_hd / 2] "sin_row"))),
        Stmt.store .real [pad_n_qh, pad_hd / 2]
          (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "q_ptr")
            (Op.ref .nat [pad_n_qh, pad_hd / 2] "first_half_q_offsets")))
          (Op.ref .real [pad_n_qh, pad_hd / 2] "new_q_tile_1")
          (MaskOpt.mask (Op.ref .bool [pad_n_qh, pad_hd / 2] "first_q_mask")),
        Stmt.assign .real [pad_n_qh, pad_hd / 2] "new_q_tile_2"
          (Op.sub NumericDType.real Broadcast.nil.consSame.consSame
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_qh, pad_hd / 2] "q_tile_2") (Op.ref .real [pad_hd / 2] "cos_row"))
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_qh, pad_hd / 2] "q_tile_1") (Op.ref .real [pad_hd / 2] "sin_row"))),
        Stmt.store .real [pad_n_qh, pad_hd / 2]
          (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "q_ptr")
            (Op.ref .nat [pad_n_qh, pad_hd / 2] "second_half_q_offsets")))
          (Op.ref .real [pad_n_qh, pad_hd / 2] "new_q_tile_2")
          (MaskOpt.mask (Op.ref .bool [pad_n_qh, pad_hd / 2] "second_q_mask")),
        Stmt.assign .real [pad_n_kh, pad_hd / 2] "new_k_tile_1"
          (Op.add NumericDType.real Broadcast.nil.consSame.consSame
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_kh, pad_hd / 2] "k_tile_1") (Op.ref .real [pad_hd / 2] "cos_row"))
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_kh, pad_hd / 2] "k_tile_2") (Op.ref .real [pad_hd / 2] "sin_row"))),
        Stmt.store .real [pad_n_kh, pad_hd / 2]
          (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "k_ptr")
            (Op.ref .nat [pad_n_kh, pad_hd / 2] "first_half_k_offsets")))
          (Op.ref .real [pad_n_kh, pad_hd / 2] "new_k_tile_1")
          (MaskOpt.mask (Op.ref .bool [pad_n_kh, pad_hd / 2] "first_k_mask")),
        Stmt.assign .real [pad_n_kh, pad_hd / 2] "new_k_tile_2"
          (Op.sub NumericDType.real Broadcast.nil.consSame.consSame
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_kh, pad_hd / 2] "k_tile_2") (Op.ref .real [pad_hd / 2] "cos_row"))
            (Op.mul NumericDType.real Broadcast.nil.consSame.leadR
              (Op.ref .real [pad_n_kh, pad_hd / 2] "k_tile_1") (Op.ref .real [pad_hd / 2] "sin_row"))),
        Stmt.store .real [pad_n_kh, pad_hd / 2]
          (MemAccess.ptr (Op.ptrAdd Broadcast.scalarL (Op.ref .ptr [] "k_ptr")
            (Op.ref .nat [pad_n_kh, pad_hd / 2] "second_half_k_offsets")))
          (Op.ref .real [pad_n_kh, pad_hd / 2] "new_k_tile_2")
          (MaskOpt.mask (Op.ref .bool [pad_n_kh, pad_hd / 2] "second_k_mask")) ] ]

/-- The backward `triton_rope_surface` lowers to `ropeBackwardBody`. -/
theorem triton_rope_surface_backward_body
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat) :
    (triton_rope_surface q_ptr k_ptr cos sin q_row_stride k_row_stride
      cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
      pad_hd BLOCK_SIZE Bool.true).body
      = ropeBackwardBody q_ptr k_ptr cos sin q_row_stride k_row_stride
          cos_row_stride sin_row_stride sl n_qh n_kh hd pad_n_qh pad_n_kh pad_hd := rfl

theorem rope_backward_body_steps
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl n_qh n_kh hd pad_n_qh pad_n_kh pad_hd : Nat)
    (s : BlockState)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ sfin, stepStmts (ropeBackwardBody q_ptr k_ptr cos sin q_row_stride k_row_stride
      cos_row_stride sin_row_stride sl n_qh n_kh hd pad_n_qh pad_n_kh pad_hd) s = some sfin
      ∧ (q_ptr ≠ k_ptr →
          ∀ idx : TileIndex [pad_n_qh, pad_hd/2], idx.1.val < n_qh → idx.2.1.val < hd/2 →
            sfin.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val)) =
              s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val)) *
                s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val) +
              s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2)) *
                s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val))
      ∧ (q_ptr ≠ k_ptr →
          ∀ idx : TileIndex [pad_n_qh, pad_hd/2], idx.1.val < n_qh → idx.2.1.val < hd/2 →
            sfin.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2)) =
              s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2)) *
                s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val) -
              s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val)) *
                s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val))
      ∧ (∀ idx : TileIndex [pad_n_kh, pad_hd/2], idx.1.val < n_kh → idx.2.1.val < hd/2 →
            sfin.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val)) =
              s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val)) *
                s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val) +
              s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2)) *
                s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val))
      ∧ (∀ idx : TileIndex [pad_n_kh, pad_hd/2], idx.1.val < n_kh → idx.2.1.val < hd/2 →
            sfin.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2)) =
              s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2)) *
                s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val) -
              s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val)) *
                s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val)) := by
  unfold ropeBackwardBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by rw [evalOp_programId]
        : evalOp (Op.programId 0) s = some (Tile.scalar (s.pids 0))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (ptrbase_mul_eval _ q_ptr (s.pids 0) q_row_stride (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (ptrbase_mul_eval _ k_ptr (s.pids 0) k_row_stride (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (mod_eval _ (s.pids 0) sl (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (ptrbase_mul_ref_eval _ cos (s.pids 0 % sl) cos_row_stride "cos_row_idx" (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (ptrbase_mul_ref_eval _ sin (s.pids 0 % sl) sin_row_stride "cos_row_idx" (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by rw [evalOp_arange]
        : evalOp (Op.arange (pad_hd/2)) _ = some (Tile.vec (fun i : Fin (pad_hd/2) => i.val))))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (cosMask_eval _ (pad_hd/2) hd (Tile.vec (fun i : Fin (pad_hd/2) => i.val)) "cos_offsets" (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_real _ (Op.ref .bool [pad_hd/2] "cos_mask") _ _ _
          (ptr_plus_offsets1d_eval _ cos (pad_hd/2) (s.pids 0 % sl * cos_row_stride)
            (Tile.vec (fun i : Fin (pad_hd/2) => i.val)) "cos" "cos_offsets" (by simp) (by simp))
          (show evalOp (Op.ref .bool [pad_hd/2] "cos_mask") _
              = some (⟨fun i => decide (i.1.val < hd/2)⟩ : Tile .bool [pad_hd/2]) from by
            rw [evalOp_ref]; simp; rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_real _ (Op.ref .bool [pad_hd/2] "cos_mask") _ _ _
          (ptr_plus_offsets1d_eval _ sin (pad_hd/2) (s.pids 0 % sl * sin_row_stride)
            (Tile.vec (fun i : Fin (pad_hd/2) => i.val)) "sin" "cos_offsets" (by simp) (by simp))
          (show evalOp (Op.ref .bool [pad_hd/2] "cos_mask") _
              = some (⟨fun i => decide (i.1.val < hd/2)⟩ : Tile .bool [pad_hd/2]) from by
            rw [evalOp_ref]; simp; rfl)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (firstHalf_offsets_eval _ pad_n_qh (pad_hd/2) hd))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (firstHalf_offsets_eval _ pad_n_kh (pad_hd/2) hd))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (firstMask_eval _ pad_n_qh (pad_hd/2) n_qh hd))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (firstMask_eval _ pad_n_kh (pad_hd/2) n_kh hd))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_real _ (Op.ref .bool [pad_n_qh, pad_hd/2] "first_q_mask") _ _ _
          (ptr_plus_offsets_eval _ q_ptr pad_n_qh (pad_hd/2) (s.pids 0 * q_row_stride)
            (⟨fun idx => idx.1.val * hd + idx.2.1.val⟩ : Tile .nat [pad_n_qh, pad_hd/2])
            "q_ptr" "first_half_q_offsets" (by simp) (by simp))
          (show evalOp (Op.ref .bool [pad_n_qh, pad_hd/2] "first_q_mask") _
              = some (⟨fun idx => decide (idx.1.val < n_qh) && decide (idx.2.1.val < hd/2)⟩
                : Tile .bool [pad_n_qh, pad_hd/2]) from by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_real _ (Op.ref .bool [pad_n_kh, pad_hd/2] "first_k_mask") _ _ _
          (ptr_plus_offsets_eval _ k_ptr pad_n_kh (pad_hd/2) (s.pids 0 * k_row_stride)
            (⟨fun idx => idx.1.val * hd + idx.2.1.val⟩ : Tile .nat [pad_n_kh, pad_hd/2])
            "k_ptr" "first_half_k_offsets" (by simp) (by simp))
          (show evalOp (Op.ref .bool [pad_n_kh, pad_hd/2] "first_k_mask") _
              = some (⟨fun idx => decide (idx.1.val < n_kh) && decide (idx.2.1.val < hd/2)⟩
                : Tile .bool [pad_n_kh, pad_hd/2]) from by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (secondHalf_offsets_eval _ pad_n_qh (pad_hd/2) hd "first_half_q_offsets" (⟨fun idx => idx.1.val * hd + idx.2.1.val⟩ : Tile .nat [pad_n_qh, pad_hd/2]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (secondHalf_offsets_eval _ pad_n_kh (pad_hd/2) hd "first_half_k_offsets" (⟨fun idx => idx.1.val * hd + idx.2.1.val⟩ : Tile .nat [pad_n_kh, pad_hd/2]) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.ref .bool [pad_n_qh, pad_hd/2] "first_q_mask") _
            = some (⟨fun idx => decide (idx.1.val < n_qh) && decide (idx.2.1.val < hd/2)⟩
              : Tile .bool [pad_n_qh, pad_hd/2]) from by rw [evalOp_ref]; simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.ref .bool [pad_n_kh, pad_hd/2] "first_k_mask") _
            = some (⟨fun idx => decide (idx.1.val < n_kh) && decide (idx.2.1.val < hd/2)⟩
              : Tile .bool [pad_n_kh, pad_hd/2]) from by rw [evalOp_ref]; simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_real _ (Op.ref .bool [pad_n_qh, pad_hd/2] "second_q_mask") _ _ _
          (ptr_plus_offsets_eval _ q_ptr pad_n_qh (pad_hd/2) (s.pids 0 * q_row_stride)
            (⟨fun idx => idx.1.val * hd + idx.2.1.val + hd/2⟩ : Tile .nat [pad_n_qh, pad_hd/2])
            "q_ptr" "second_half_q_offsets" (by simp) (by simp))
          (show evalOp (Op.ref .bool [pad_n_qh, pad_hd/2] "second_q_mask") _
              = some (⟨fun idx => decide (idx.1.val < n_qh) && decide (idx.2.1.val < hd/2)⟩
                : Tile .bool [pad_n_qh, pad_hd/2]) from by rw [evalOp_ref]; simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_maskOther_real _ (Op.ref .bool [pad_n_kh, pad_hd/2] "second_k_mask") _ _ _
          (ptr_plus_offsets_eval _ k_ptr pad_n_kh (pad_hd/2) (s.pids 0 * k_row_stride)
            (⟨fun idx => idx.1.val * hd + idx.2.1.val + hd/2⟩ : Tile .nat [pad_n_kh, pad_hd/2])
            "k_ptr" "second_half_k_offsets" (by simp) (by simp))
          (show evalOp (Op.ref .bool [pad_n_kh, pad_hd/2] "second_k_mask") _
              = some (⟨fun idx => decide (idx.1.val < n_kh) && decide (idx.2.1.val < hd/2)⟩
                : Tile .bool [pad_n_kh, pad_hd/2]) from by rw [evalOp_ref]; simp)))]
  rw [stepStmts_singleton_ifThenElse_notTrue]
  -- elseBody: 4 (assign, store) pairs (BACKWARD_PASS = true)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (newAdd_value_eval _ pad_n_qh (pad_hd/2) (⟨fun i => if (decide (i.1.val < n_qh) && decide (i.2.1.val < hd/2)) then some (s.readMem q_ptr (s.pids 0 * q_row_stride + (i.1.val * hd + i.2.1.val))) else some 0⟩ : Tile .real [pad_n_qh, pad_hd/2]) (⟨fun i => if (decide (i.1.val < n_qh) && decide (i.2.1.val < hd/2)) then some (s.readMem q_ptr (s.pids 0 * q_row_stride + (i.1.val * hd + i.2.1.val + hd/2))) else some 0⟩ : Tile .real [pad_n_qh, pad_hd/2]) (⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem cos (s.pids 0 % sl * cos_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2]) (⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem sin (s.pids 0 % sl * sin_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2]) "q_tile_1" "q_tile_2" "cos_row" "sin_row" (by simp) (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_store_ptr_mask_eq _ (Op.ref .real [pad_n_qh, pad_hd/2] "new_q_tile_1") _ _ _ (⟨fun idx => NumericDType.real.add (NumericDType.real.mul (((⟨fun i => if (decide (i.1.val < n_qh) && decide (i.2.1.val < hd/2)) then some (s.readMem q_ptr (s.pids 0 * q_row_stride + (i.1.val * hd + i.2.1.val))) else some 0⟩ : Tile .real [pad_n_qh, pad_hd/2])).data idx) (((⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem cos (s.pids 0 % sl * cos_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2])).data (idx.2.1, idx.2.2))) (NumericDType.real.mul (((⟨fun i => if (decide (i.1.val < n_qh) && decide (i.2.1.val < hd/2)) then some (s.readMem q_ptr (s.pids 0 * q_row_stride + (i.1.val * hd + i.2.1.val + hd/2))) else some 0⟩ : Tile .real [pad_n_qh, pad_hd/2])).data idx) (((⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem sin (s.pids 0 % sl * sin_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2])).data (idx.2.1, idx.2.2)))⟩ : Tile .real [pad_n_qh, pad_hd/2]) _ (by rw [evalOp_ref]; simp)
        (ptr_plus_offsets_eval _ q_ptr pad_n_qh (pad_hd/2) (s.pids 0 * q_row_stride)
          (⟨fun idx => idx.1.val * hd + idx.2.1.val⟩ : Tile .nat [pad_n_qh, pad_hd/2])
          "q_ptr" "first_half_q_offsets" (by simp) (by simp))
        (show evalOp (Op.ref .bool [pad_n_qh, pad_hd/2] "first_q_mask") _
            = some (⟨fun idx => decide (idx.1.val < n_qh) && decide (idx.2.1.val < hd/2)⟩
              : Tile .bool [pad_n_qh, pad_hd/2]) from by rw [evalOp_ref]; simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (newSub_value_eval _ pad_n_qh (pad_hd/2) (⟨fun i => if (decide (i.1.val < n_qh) && decide (i.2.1.val < hd/2)) then some (s.readMem q_ptr (s.pids 0 * q_row_stride + (i.1.val * hd + i.2.1.val + hd/2))) else some 0⟩ : Tile .real [pad_n_qh, pad_hd/2]) (⟨fun i => if (decide (i.1.val < n_qh) && decide (i.2.1.val < hd/2)) then some (s.readMem q_ptr (s.pids 0 * q_row_stride + (i.1.val * hd + i.2.1.val))) else some 0⟩ : Tile .real [pad_n_qh, pad_hd/2]) (⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem cos (s.pids 0 % sl * cos_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2]) (⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem sin (s.pids 0 % sl * sin_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2]) "q_tile_2" "q_tile_1" "cos_row" "sin_row" (by simp) (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_store_ptr_mask_eq _ (Op.ref .real [pad_n_qh, pad_hd/2] "new_q_tile_2") _ _ _ (⟨fun idx => NumericDType.real.sub (NumericDType.real.mul (((⟨fun i => if (decide (i.1.val < n_qh) && decide (i.2.1.val < hd/2)) then some (s.readMem q_ptr (s.pids 0 * q_row_stride + (i.1.val * hd + i.2.1.val + hd/2))) else some 0⟩ : Tile .real [pad_n_qh, pad_hd/2])).data idx) (((⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem cos (s.pids 0 % sl * cos_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2])).data (idx.2.1, idx.2.2))) (NumericDType.real.mul (((⟨fun i => if (decide (i.1.val < n_qh) && decide (i.2.1.val < hd/2)) then some (s.readMem q_ptr (s.pids 0 * q_row_stride + (i.1.val * hd + i.2.1.val))) else some 0⟩ : Tile .real [pad_n_qh, pad_hd/2])).data idx) (((⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem sin (s.pids 0 % sl * sin_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2])).data (idx.2.1, idx.2.2)))⟩ : Tile .real [pad_n_qh, pad_hd/2]) _ (by rw [evalOp_ref]; simp)
        (ptr_plus_offsets_eval _ q_ptr pad_n_qh (pad_hd/2) (s.pids 0 * q_row_stride)
          (⟨fun idx => idx.1.val * hd + idx.2.1.val + hd/2⟩ : Tile .nat [pad_n_qh, pad_hd/2])
          "q_ptr" "second_half_q_offsets" (by simp) (by simp))
        (show evalOp (Op.ref .bool [pad_n_qh, pad_hd/2] "second_q_mask") _
            = some (⟨fun idx => decide (idx.1.val < n_qh) && decide (idx.2.1.val < hd/2)⟩
              : Tile .bool [pad_n_qh, pad_hd/2]) from by rw [evalOp_ref]; simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (newAdd_value_eval _ pad_n_kh (pad_hd/2) (⟨fun i => if (decide (i.1.val < n_kh) && decide (i.2.1.val < hd/2)) then some (s.readMem k_ptr (s.pids 0 * k_row_stride + (i.1.val * hd + i.2.1.val))) else some 0⟩ : Tile .real [pad_n_kh, pad_hd/2]) (⟨fun i => if (decide (i.1.val < n_kh) && decide (i.2.1.val < hd/2)) then some (s.readMem k_ptr (s.pids 0 * k_row_stride + (i.1.val * hd + i.2.1.val + hd/2))) else some 0⟩ : Tile .real [pad_n_kh, pad_hd/2]) (⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem cos (s.pids 0 % sl * cos_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2]) (⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem sin (s.pids 0 % sl * sin_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2]) "k_tile_1" "k_tile_2" "cos_row" "sin_row" (by simp) (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_store_ptr_mask_eq _ (Op.ref .real [pad_n_kh, pad_hd/2] "new_k_tile_1") _ _ _ (⟨fun idx => NumericDType.real.add (NumericDType.real.mul (((⟨fun i => if (decide (i.1.val < n_kh) && decide (i.2.1.val < hd/2)) then some (s.readMem k_ptr (s.pids 0 * k_row_stride + (i.1.val * hd + i.2.1.val))) else some 0⟩ : Tile .real [pad_n_kh, pad_hd/2])).data idx) (((⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem cos (s.pids 0 % sl * cos_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2])).data (idx.2.1, idx.2.2))) (NumericDType.real.mul (((⟨fun i => if (decide (i.1.val < n_kh) && decide (i.2.1.val < hd/2)) then some (s.readMem k_ptr (s.pids 0 * k_row_stride + (i.1.val * hd + i.2.1.val + hd/2))) else some 0⟩ : Tile .real [pad_n_kh, pad_hd/2])).data idx) (((⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem sin (s.pids 0 % sl * sin_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2])).data (idx.2.1, idx.2.2)))⟩ : Tile .real [pad_n_kh, pad_hd/2]) _ (by rw [evalOp_ref]; simp)
        (ptr_plus_offsets_eval _ k_ptr pad_n_kh (pad_hd/2) (s.pids 0 * k_row_stride)
          (⟨fun idx => idx.1.val * hd + idx.2.1.val⟩ : Tile .nat [pad_n_kh, pad_hd/2])
          "k_ptr" "first_half_k_offsets" (by simp) (by simp))
        (show evalOp (Op.ref .bool [pad_n_kh, pad_hd/2] "first_k_mask") _
            = some (⟨fun idx => decide (idx.1.val < n_kh) && decide (idx.2.1.val < hd/2)⟩
              : Tile .bool [pad_n_kh, pad_hd/2]) from by rw [evalOp_ref]; simp))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (newSub_value_eval _ pad_n_kh (pad_hd/2) (⟨fun i => if (decide (i.1.val < n_kh) && decide (i.2.1.val < hd/2)) then some (s.readMem k_ptr (s.pids 0 * k_row_stride + (i.1.val * hd + i.2.1.val + hd/2))) else some 0⟩ : Tile .real [pad_n_kh, pad_hd/2]) (⟨fun i => if (decide (i.1.val < n_kh) && decide (i.2.1.val < hd/2)) then some (s.readMem k_ptr (s.pids 0 * k_row_stride + (i.1.val * hd + i.2.1.val))) else some 0⟩ : Tile .real [pad_n_kh, pad_hd/2]) (⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem cos (s.pids 0 % sl * cos_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2]) (⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem sin (s.pids 0 % sl * sin_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2]) "k_tile_2" "k_tile_1" "cos_row" "sin_row" (by simp) (by simp) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_store_ptr_mask_eq _ (Op.ref .real [pad_n_kh, pad_hd/2] "new_k_tile_2") _ _ _ (⟨fun idx => NumericDType.real.sub (NumericDType.real.mul (((⟨fun i => if (decide (i.1.val < n_kh) && decide (i.2.1.val < hd/2)) then some (s.readMem k_ptr (s.pids 0 * k_row_stride + (i.1.val * hd + i.2.1.val + hd/2))) else some 0⟩ : Tile .real [pad_n_kh, pad_hd/2])).data idx) (((⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem cos (s.pids 0 % sl * cos_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2])).data (idx.2.1, idx.2.2))) (NumericDType.real.mul (((⟨fun i => if (decide (i.1.val < n_kh) && decide (i.2.1.val < hd/2)) then some (s.readMem k_ptr (s.pids 0 * k_row_stride + (i.1.val * hd + i.2.1.val))) else some 0⟩ : Tile .real [pad_n_kh, pad_hd/2])).data idx) (((⟨fun i => if decide (i.1.val < hd/2) then some (s.readMem sin (s.pids 0 % sl * sin_row_stride + i.1.val)) else some 0⟩ : Tile .real [pad_hd/2])).data (idx.2.1, idx.2.2)))⟩ : Tile .real [pad_n_kh, pad_hd/2]) _ (by rw [evalOp_ref]; simp)
        (ptr_plus_offsets_eval _ k_ptr pad_n_kh (pad_hd/2) (s.pids 0 * k_row_stride)
          (⟨fun idx => idx.1.val * hd + idx.2.1.val + hd/2⟩ : Tile .nat [pad_n_kh, pad_hd/2])
          "k_ptr" "second_half_k_offsets" (by simp) (by simp))
        (show evalOp (Op.ref .bool [pad_n_kh, pad_hd/2] "second_k_mask") _
            = some (⟨fun idx => decide (idx.1.val < n_kh) && decide (idx.2.1.val < hd/2)⟩
              : Tile .bool [pad_n_kh, pad_hd/2]) from by rw [evalOp_ref]; simp))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_, ?_, ?_, ?_⟩
  · -- Q0 readback: read q first-half = q1*cos + q2*sin
    intro hqk idx hrow hcol
    simp only [BlockState.writeMemTyped_real, Region.cast_id, BlockState.setReg_readMem]
    rw [BlockState.foldl_writeMem_const_region_bool_masked_readMem_other (region := k_ptr) (R := q_ptr) (hRR := hqk)]
    simp only [BlockState.setReg_readMem]
    rw [BlockState.foldl_writeMem_const_region_bool_masked_readMem_other (region := k_ptr) (R := q_ptr) (hRR := hqk)]
    simp only [BlockState.setReg_readMem]
    rw [foldl_writeMem_same_region_disjoint_offsets_readMem_bool (region := q_ptr)
          (off := s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val))]
    · simp only [BlockState.setReg_readMem]
      rw [BlockState.scatter_readback_prop_masked_nd_of_true (region := q_ptr)
            (offsetFn := fun i : TileIndex [pad_n_qh, pad_hd/2] => s.pids 0 * q_row_stride + (i.1.val * hd + i.2.1.val))
            (P := fun i : TileIndex [pad_n_qh, pad_hd/2] => (decide (i.1.val < n_qh) && decide (i.2.1.val < hd/2)) = Bool.true)
            (i := idx)
            (hPi := by simp [hrow, hcol])
            (h_no_collision := by
              intro kk hPk heq
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hPk
              have hcancel : kk.1.val * hd + kk.2.1.val = idx.1.val * hd + idx.2.1.val :=
                Nat.add_left_cancel heq
              have hkc := hPk.2
              rcases lt_trichotomy kk.1.val idx.1.val with hlt | heqn | hgt
              · exfalso
                have hb : kk.1.val * hd + hd ≤ idx.1.val * hd := by
                  have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hlt); simpa [Nat.succ_mul] using this
                omega
              · have hrr : kk.1 = idx.1 := Fin.ext heqn
                have hcc0 : kk.2.1.val = idx.2.1.val := by
                  have : kk.1.val * hd = idx.1.val * hd := by rw [heqn]
                  omega
                have hcc : kk.2.1 = idx.2.1 := Fin.ext hcc0
                obtain ⟨k1,k2,k3⟩ := kk; obtain ⟨i1,i2,i3⟩ := idx
                simp_all
              · exfalso
                have hb : idx.1.val * hd + hd ≤ kk.1.val * hd := by
                  have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hgt); simpa [Nat.succ_mul] using this
                omega)]
      have h1 : decide (idx.1.val < n_qh) = Bool.true := by simp [hrow]
      have h2 : decide (idx.2.1.val < hd/2) = Bool.true := by simp [hcol]
      simp only [h1, h2, Bool.and_self, Bool.true_and, if_true, FloatDType.real_storeValue,
        NumericDType.add, NumericDType.mul]
      rw [show (some (s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val))) : WithBot ℝ)
            = ((s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val)) : ℝ) : WithBot ℝ) from rfl,
          show (some (s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val)) : WithBot ℝ)
            = ((s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val) : ℝ) : WithBot ℝ) from rfl,
          show (some (s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2))) : WithBot ℝ)
            = ((s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2)) : ℝ) : WithBot ℝ) from rfl,
          show (some (s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val)) : WithBot ℝ)
            = ((s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val) : ℝ) : WithBot ℝ) from rfl]
      rw [WithBot.realMul_coe_coe, WithBot.realMul_coe_coe, WithBot.realAdd_coe_coe, WithBot.unbotD_coe]
    · intro kk _ hp
      have hkc : kk.2.1.val < hd/2 := by
        by_contra h
        simp only [decide_eq_true_eq, Bool.and_eq_true, decide_eq_true_eq] at hp
        omega
      have hhalf : hd/2 + hd/2 ≤ hd := by omega
      intro heq
      have hc : idx.1.val * hd + idx.2.1.val = kk.1.val * hd + kk.2.1.val + hd/2 := by omega
      rcases lt_trichotomy idx.1.val kk.1.val with hlt | heqn | hgt
      · have : idx.1.val * hd + hd ≤ kk.1.val * hd := by
          have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hlt)
          simpa [Nat.succ_mul] using this
        omega
      · rw [heqn] at hc; omega
      · have : kk.1.val * hd + hd ≤ idx.1.val * hd := by
          have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hgt)
          simpa [Nat.succ_mul] using this
        omega
  · -- Q1 readback: read q second-half = q2*cos - q1*sin
    intro hqk idx hrow hcol
    simp only [BlockState.writeMemTyped_real, Region.cast_id, BlockState.setReg_readMem]
    rw [BlockState.foldl_writeMem_const_region_bool_masked_readMem_other (region := k_ptr) (R := q_ptr) (hRR := hqk)]
    simp only [BlockState.setReg_readMem]
    rw [BlockState.foldl_writeMem_const_region_bool_masked_readMem_other (region := k_ptr) (R := q_ptr) (hRR := hqk)]
    simp only [BlockState.setReg_readMem]
    rw [BlockState.scatter_readback_prop_masked_nd_of_true (region := q_ptr)
          (offsetFn := fun i : TileIndex [pad_n_qh, pad_hd/2] => s.pids 0 * q_row_stride + (i.1.val * hd + i.2.1.val + hd/2))
          (P := fun i : TileIndex [pad_n_qh, pad_hd/2] => (decide (i.1.val < n_qh) && decide (i.2.1.val < hd/2)) = Bool.true)
          (i := idx)
          (hPi := by simp [hrow, hcol])
          (h_no_collision := by
            intro kk hPk heq
            simp only [Bool.and_eq_true, decide_eq_true_eq] at hPk
            have hcancel : kk.1.val * hd + kk.2.1.val = idx.1.val * hd + idx.2.1.val := by
              have := Nat.add_left_cancel heq; omega
            have hkc := hPk.2
            rcases lt_trichotomy kk.1.val idx.1.val with hlt | heqn | hgt
            · exfalso
              have hb : kk.1.val * hd + hd ≤ idx.1.val * hd := by
                have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hlt); simpa [Nat.succ_mul] using this
              omega
            · have hrr : kk.1 = idx.1 := Fin.ext heqn
              have hcc0 : kk.2.1.val = idx.2.1.val := by
                have : kk.1.val * hd = idx.1.val * hd := by rw [heqn]
                omega
              have hcc : kk.2.1 = idx.2.1 := Fin.ext hcc0
              obtain ⟨k1,k2,k3⟩ := kk; obtain ⟨i1,i2,i3⟩ := idx
              simp_all
            · exfalso
              have hb : idx.1.val * hd + hd ≤ kk.1.val * hd := by
                have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hgt); simpa [Nat.succ_mul] using this
              omega)]
    have h1 : decide (idx.1.val < n_qh) = Bool.true := by simp [hrow]
    have h2 : decide (idx.2.1.val < hd/2) = Bool.true := by simp [hcol]
    simp only [h1, h2, Bool.and_self, Bool.true_and, if_true, FloatDType.real_storeValue,
      NumericDType.sub, NumericDType.mul]
    rw [show (some (s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2))) : WithBot ℝ)
          = ((s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2)) : ℝ) : WithBot ℝ) from rfl,
        show (some (s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val)) : WithBot ℝ)
          = ((s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val) : ℝ) : WithBot ℝ) from rfl,
        show (some (s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val))) : WithBot ℝ)
          = ((s.readMem q_ptr (s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val)) : ℝ) : WithBot ℝ) from rfl,
        show (some (s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val)) : WithBot ℝ)
          = ((s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val) : ℝ) : WithBot ℝ) from rfl]
    rw [WithBot.realMul_coe_coe, WithBot.realMul_coe_coe, WithBot.realSub_coe_coe, WithBot.unbotD_coe]
  · -- K0 readback: read k first-half = k1*cos + k2*sin
    intro idx hrow hcol
    simp only [BlockState.writeMemTyped_real, Region.cast_id, BlockState.setReg_readMem]
    rw [foldl_writeMem_same_region_disjoint_offsets_readMem_bool (region := k_ptr)
          (off := s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val))]
    · simp only [BlockState.setReg_readMem]
      rw [BlockState.scatter_readback_prop_masked_nd_of_true (region := k_ptr)
            (offsetFn := fun i : TileIndex [pad_n_kh, pad_hd/2] => s.pids 0 * k_row_stride + (i.1.val * hd + i.2.1.val))
            (P := fun i : TileIndex [pad_n_kh, pad_hd/2] => (decide (i.1.val < n_kh) && decide (i.2.1.val < hd/2)) = Bool.true)
            (i := idx)
            (hPi := by simp [hrow, hcol])
            (h_no_collision := by
              intro kk hPk heq
              simp only [Bool.and_eq_true, decide_eq_true_eq] at hPk
              have hcancel : kk.1.val * hd + kk.2.1.val = idx.1.val * hd + idx.2.1.val :=
                Nat.add_left_cancel heq
              have hkc := hPk.2
              rcases lt_trichotomy kk.1.val idx.1.val with hlt | heqn | hgt
              · exfalso
                have hb : kk.1.val * hd + hd ≤ idx.1.val * hd := by
                  have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hlt); simpa [Nat.succ_mul] using this
                omega
              · have hrr : kk.1 = idx.1 := Fin.ext heqn
                have hcc0 : kk.2.1.val = idx.2.1.val := by
                  have : kk.1.val * hd = idx.1.val * hd := by rw [heqn]
                  omega
                have hcc : kk.2.1 = idx.2.1 := Fin.ext hcc0
                obtain ⟨k1,k2,k3⟩ := kk; obtain ⟨i1,i2,i3⟩ := idx
                simp_all
              · exfalso
                have hb : idx.1.val * hd + hd ≤ kk.1.val * hd := by
                  have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hgt); simpa [Nat.succ_mul] using this
                omega)]
      have h1 : decide (idx.1.val < n_kh) = Bool.true := by simp [hrow]
      have h2 : decide (idx.2.1.val < hd/2) = Bool.true := by simp [hcol]
      simp only [h1, h2, Bool.and_self, Bool.true_and, if_true, FloatDType.real_storeValue,
        NumericDType.add, NumericDType.mul]
      rw [show (some (s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val))) : WithBot ℝ)
            = ((s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val)) : ℝ) : WithBot ℝ) from rfl,
          show (some (s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val)) : WithBot ℝ)
            = ((s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val) : ℝ) : WithBot ℝ) from rfl,
          show (some (s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2))) : WithBot ℝ)
            = ((s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2)) : ℝ) : WithBot ℝ) from rfl,
          show (some (s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val)) : WithBot ℝ)
            = ((s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val) : ℝ) : WithBot ℝ) from rfl]
      rw [WithBot.realMul_coe_coe, WithBot.realMul_coe_coe, WithBot.realAdd_coe_coe, WithBot.unbotD_coe]
    · intro kk _ hp
      have hkc : kk.2.1.val < hd/2 := by
        by_contra h
        simp only [decide_eq_true_eq, Bool.and_eq_true, decide_eq_true_eq] at hp
        omega
      have hhalf : hd/2 + hd/2 ≤ hd := by omega
      intro heq
      have hc : idx.1.val * hd + idx.2.1.val = kk.1.val * hd + kk.2.1.val + hd/2 := by omega
      rcases lt_trichotomy idx.1.val kk.1.val with hlt | heqn | hgt
      · have : idx.1.val * hd + hd ≤ kk.1.val * hd := by
          have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hlt); simpa [Nat.succ_mul] using this
        omega
      · rw [heqn] at hc; omega
      · have : kk.1.val * hd + hd ≤ idx.1.val * hd := by
          have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hgt); simpa [Nat.succ_mul] using this
        omega
  · -- K1 readback: read k second-half = k2*cos - k1*sin (K1 outermost)
    intro idx hrow hcol
    simp only [BlockState.writeMemTyped_real, Region.cast_id, BlockState.setReg_readMem]
    rw [BlockState.scatter_readback_prop_masked_nd_of_true (region := k_ptr)
          (offsetFn := fun i : TileIndex [pad_n_kh, pad_hd/2] => s.pids 0 * k_row_stride + (i.1.val * hd + i.2.1.val + hd/2))
          (P := fun i : TileIndex [pad_n_kh, pad_hd/2] => (decide (i.1.val < n_kh) && decide (i.2.1.val < hd/2)) = Bool.true)
          (i := idx)
          (hPi := by simp [hrow, hcol])
          (h_no_collision := by
            intro kk hPk heq
            simp only [Bool.and_eq_true, decide_eq_true_eq] at hPk
            have hcancel : kk.1.val * hd + kk.2.1.val = idx.1.val * hd + idx.2.1.val := by
              have := Nat.add_left_cancel heq; omega
            have hkc := hPk.2
            rcases lt_trichotomy kk.1.val idx.1.val with hlt | heqn | hgt
            · exfalso
              have hb : kk.1.val * hd + hd ≤ idx.1.val * hd := by
                have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hlt); simpa [Nat.succ_mul] using this
              omega
            · have hrr : kk.1 = idx.1 := Fin.ext heqn
              have hcc0 : kk.2.1.val = idx.2.1.val := by
                have : kk.1.val * hd = idx.1.val * hd := by rw [heqn]
                omega
              have hcc : kk.2.1 = idx.2.1 := Fin.ext hcc0
              obtain ⟨k1,k2,k3⟩ := kk; obtain ⟨i1,i2,i3⟩ := idx
              simp_all
            · exfalso
              have hb : idx.1.val * hd + hd ≤ kk.1.val * hd := by
                have := Nat.mul_le_mul_right hd (Nat.succ_le_of_lt hgt); simpa [Nat.succ_mul] using this
              omega)]
    have h1 : decide (idx.1.val < n_kh) = Bool.true := by simp [hrow]
    have h2 : decide (idx.2.1.val < hd/2) = Bool.true := by simp [hcol]
    simp only [h1, h2, Bool.and_self, Bool.true_and, if_true, FloatDType.real_storeValue,
      NumericDType.sub, NumericDType.mul]
    rw [show (some (s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2))) : WithBot ℝ)
          = ((s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val + hd/2)) : ℝ) : WithBot ℝ) from rfl,
        show (some (s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val)) : WithBot ℝ)
          = ((s.readMem cos (s.pids 0 % sl * cos_row_stride + idx.2.1.val) : ℝ) : WithBot ℝ) from rfl,
        show (some (s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val))) : WithBot ℝ)
          = ((s.readMem k_ptr (s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val)) : ℝ) : WithBot ℝ) from rfl,
        show (some (s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val)) : WithBot ℝ)
          = ((s.readMem sin (s.pids 0 % sl * sin_row_stride + idx.2.1.val) : ℝ) : WithBot ℝ) from rfl]
    rw [WithBot.realMul_coe_coe, WithBot.realMul_coe_coe, WithBot.realSub_coe_coe, WithBot.unbotD_coe]

/-! ## Genuine per-store backward correctness against `triton_rope_surface`

Each theorem reads back one of the four Python-observable backward stores from
the real `triton_rope_surface` kernel (`BACKWARD_PASS = true`) and shows the
active lanes equal the genuine rotary-backward closed form
(`ropeBackwardKernel{Q0,Q1,K0,K1}Spec`), NOT the kernel's own executed value. The
host launch / padding choices remain the trusted boundary. -/

private theorem rope_exec_eq (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (s sfin : BlockState)
    (h : stepStmts (ropeBackwardBody q_ptr k_ptr cos sin q_row_stride k_row_stride
      cos_row_stride sin_row_stride sl n_qh n_kh hd pad_n_qh pad_n_kh pad_hd) s = some sfin) :
    exec (triton_rope_surface q_ptr k_ptr cos sin q_row_stride k_row_stride
      cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
      pad_hd BLOCK_SIZE Bool.true) s = some sfin := by
  rw [exec]
  rw [show (triton_rope_surface q_ptr k_ptr cos sin q_row_stride k_row_stride
      cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
      pad_hd BLOCK_SIZE Bool.true).toAlgKernel.body
      = ropeBackwardBody q_ptr k_ptr cos sin q_row_stride k_row_stride
          cos_row_stride sin_row_stride sl n_qh n_kh hd pad_n_qh pad_n_kh pad_hd
      from triton_rope_surface_backward_body q_ptr k_ptr cos sin q_row_stride k_row_stride
        cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE]
  exact h

/-- Q first-half backward store correctness: `new_q_tile_1 = q1·cos + q2·sin`. -/
theorem rope_backward_q0_backward_correct
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) (hqk : q_ptr ≠ k_ptr)
    (idx : TileIndex [pad_n_qh, pad_hd/2])
    (hActive : activeQFull (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2) n_qh hd idx) :
    (match exec (triton_rope_surface q_ptr k_ptr cos sin q_row_stride k_row_stride
        cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
        pad_hd BLOCK_SIZE Bool.true) s with
      | some s' => s'.readMem q_ptr (qFullFirstOffset (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2) s q_row_stride hd idx)
      | none => (0.0 : ℝ)) =
      ropeBackwardKernelQ0Spec (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2)
        s q_ptr cos sin q_row_stride sl cos_row_stride sin_row_stride hd idx := by
  obtain ⟨sfin, hstep, hq0, _, _, _⟩ := rope_backward_body_steps q_ptr k_ptr cos sin
    q_row_stride k_row_stride cos_row_stride sin_row_stride sl n_qh n_kh hd
    pad_n_qh pad_n_kh pad_hd s hundef
  rw [rope_exec_eq (s := s) (sfin := sfin) (h := hstep)]
  simp only [qFullFirstOffset, qFullSecondOffset, cosFullFirstOffset, sinFullFirstOffset,
    ropeBackwardKernelQ0Spec]
  rw [Nat.add_assoc (s.pids 0 * q_row_stride) (idx.1.val * hd) (idx.2.1.val),
      show s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val) + hd / 2
        = s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val + hd / 2) by ring]
  exact hq0 hqk idx hActive.1 hActive.2

/-- Q second-half backward store correctness: `new_q_tile_2 = q2·cos − q1·sin`. -/
theorem rope_backward_q1_backward_correct
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) (hqk : q_ptr ≠ k_ptr)
    (idx : TileIndex [pad_n_qh, pad_hd/2])
    (hActive : activeQFull (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2) n_qh hd idx) :
    (match exec (triton_rope_surface q_ptr k_ptr cos sin q_row_stride k_row_stride
        cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
        pad_hd BLOCK_SIZE Bool.true) s with
      | some s' => s'.readMem q_ptr (qFullSecondOffset (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2) s q_row_stride hd idx)
      | none => (0.0 : ℝ)) =
      ropeBackwardKernelQ1Spec (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2)
        s q_ptr cos sin q_row_stride sl cos_row_stride sin_row_stride hd idx := by
  obtain ⟨sfin, hstep, _, hq1, _, _⟩ := rope_backward_body_steps q_ptr k_ptr cos sin
    q_row_stride k_row_stride cos_row_stride sin_row_stride sl n_qh n_kh hd
    pad_n_qh pad_n_kh pad_hd s hundef
  rw [rope_exec_eq (s := s) (sfin := sfin) (h := hstep)]
  simp only [qFullFirstOffset, qFullSecondOffset, cosFullFirstOffset, sinFullFirstOffset,
    ropeBackwardKernelQ1Spec]
  rw [Nat.add_assoc (s.pids 0 * q_row_stride) (idx.1.val * hd) (idx.2.1.val),
      show s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val) + hd / 2
        = s.pids 0 * q_row_stride + (idx.1.val * hd + idx.2.1.val + hd / 2) by ring]
  exact hq1 hqk idx hActive.1 hActive.2

/-- K first-half backward store correctness: `new_k_tile_1 = k1·cos + k2·sin`. -/
theorem rope_backward_k0_backward_correct
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [pad_n_kh, pad_hd/2])
    (hActive : activeKFull (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2) n_kh hd idx) :
    (match exec (triton_rope_surface q_ptr k_ptr cos sin q_row_stride k_row_stride
        cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
        pad_hd BLOCK_SIZE Bool.true) s with
      | some s' => s'.readMem k_ptr (kFullFirstOffset (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2) s k_row_stride hd idx)
      | none => (0.0 : ℝ)) =
      ropeBackwardKernelK0Spec (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2)
        s k_ptr cos sin k_row_stride sl cos_row_stride sin_row_stride hd idx := by
  obtain ⟨sfin, hstep, _, _, hk0, _⟩ := rope_backward_body_steps q_ptr k_ptr cos sin
    q_row_stride k_row_stride cos_row_stride sin_row_stride sl n_qh n_kh hd
    pad_n_qh pad_n_kh pad_hd s hundef
  rw [rope_exec_eq (s := s) (sfin := sfin) (h := hstep)]
  simp only [kFullFirstOffset, kFullSecondOffset, cosFullFirstOffset, sinFullFirstOffset,
    ropeBackwardKernelK0Spec]
  rw [Nat.add_assoc (s.pids 0 * k_row_stride) (idx.1.val * hd) (idx.2.1.val),
      show s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val) + hd / 2
        = s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val + hd / 2) by ring]
  exact hk0 idx hActive.1 hActive.2

/-- K second-half backward store correctness: `new_k_tile_2 = k2·cos − k1·sin`. -/
theorem rope_backward_k1_backward_correct
    (q_ptr k_ptr cos sin : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [pad_n_kh, pad_hd/2])
    (hActive : activeKFull (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2) n_kh hd idx) :
    (match exec (triton_rope_surface q_ptr k_ptr cos sin q_row_stride k_row_stride
        cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
        pad_hd BLOCK_SIZE Bool.true) s with
      | some s' => s'.readMem k_ptr (kFullSecondOffset (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2) s k_row_stride hd idx)
      | none => (0.0 : ℝ)) =
      ropeBackwardKernelK1Spec (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2)
        s k_ptr cos sin k_row_stride sl cos_row_stride sin_row_stride hd idx := by
  obtain ⟨sfin, hstep, _, _, _, hk1⟩ := rope_backward_body_steps q_ptr k_ptr cos sin
    q_row_stride k_row_stride cos_row_stride sin_row_stride sl n_qh n_kh hd
    pad_n_qh pad_n_kh pad_hd s hundef
  rw [rope_exec_eq (s := s) (sfin := sfin) (h := hstep)]
  simp only [kFullFirstOffset, kFullSecondOffset, cosFullFirstOffset, sinFullFirstOffset,
    ropeBackwardKernelK1Spec]
  rw [Nat.add_assoc (s.pids 0 * k_row_stride) (idx.1.val * hd) (idx.2.1.val),
      show s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val) + hd / 2
        = s.pids 0 * k_row_stride + (idx.1.val * hd + idx.2.1.val + hd / 2) by ring]
  exact hk1 idx hActive.1 hActive.2

/-- **Dimension-general Python-path backward output summary.** For arbitrary
strides, `sl`/`bs`/`n_qh`/`n_kh`/`hd`/`pad_n_qh`/`pad_n_kh`/`pad_hd`/`BLOCK_SIZE`
(and any program ids in `s`), the full backward surface lowers and every active
lane of each of the four `BACKWARD_PASS = true` Python-observable stores (Q/K
first and second halves) reads back to the genuine rotary-backward closed form,
NOT the kernel's own executed value, against the real `triton_rope_surface`
kernel — under the honest `undef`-zero and `q_ptr ≠ k_ptr` side conditions.

Stated in the standard `ComputeCorrect.Realizes` trust surface: the conclusion is
a conjunction of one `Realizes` per stored output half. Each `Realizes` bundles
the projection-succeeds and per-run `exec` quantification internally; the
`WriteMap.writeIf` masks the store to exactly its active lanes and the
`expected` map is the genuine rotary-backward closed form over INPUT memory.
Concrete Python benchmark shapes are instantiations of this. -/
theorem rope_backward_python_backward_output_summary_general
    (Q K COS SIN : RegionName)
    (q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0) (hqk : Q ≠ K) :
    ComputeCorrect.Realizes
      (kernel := triton_rope_surface Q K COS SIN q_row_stride k_row_stride
        cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
        pad_hd BLOCK_SIZE Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [pad_n_qh, pad_hd/2] =>
          activeQFull (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2) n_qh hd idx)
        (fun idx => (Q, qFullFirstOffset (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2) s q_row_stride hd idx)))
      (expected := fun idx =>
        ropeBackwardKernelQ0Spec (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2) s Q COS SIN q_row_stride sl cos_row_stride sin_row_stride hd idx) ∧
    ComputeCorrect.Realizes
      (kernel := triton_rope_surface Q K COS SIN q_row_stride k_row_stride
        cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
        pad_hd BLOCK_SIZE Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [pad_n_qh, pad_hd/2] =>
          activeQFull (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2) n_qh hd idx)
        (fun idx => (Q, qFullSecondOffset (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2) s q_row_stride hd idx)))
      (expected := fun idx =>
        ropeBackwardKernelQ1Spec (pad_n_qh := pad_n_qh) (pad_hd_half := pad_hd/2) s Q COS SIN q_row_stride sl cos_row_stride sin_row_stride hd idx) ∧
    ComputeCorrect.Realizes
      (kernel := triton_rope_surface Q K COS SIN q_row_stride k_row_stride
        cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
        pad_hd BLOCK_SIZE Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [pad_n_kh, pad_hd/2] =>
          activeKFull (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2) n_kh hd idx)
        (fun idx => (K, kFullFirstOffset (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2) s k_row_stride hd idx)))
      (expected := fun idx =>
        ropeBackwardKernelK0Spec (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2) s K COS SIN k_row_stride sl cos_row_stride sin_row_stride hd idx) ∧
    ComputeCorrect.Realizes
      (kernel := triton_rope_surface Q K COS SIN q_row_stride k_row_stride
        cos_row_stride sin_row_stride sl bs n_qh n_kh hd pad_n_qh pad_n_kh
        pad_hd BLOCK_SIZE Bool.true)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [pad_n_kh, pad_hd/2] =>
          activeKFull (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2) n_kh hd idx)
        (fun idx => (K, kFullSecondOffset (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2) s k_row_stride hd idx)))
      (expected := fun idx =>
        ropeBackwardKernelK1Spec (pad_n_kh := pad_n_kh) (pad_hd_half := pad_hd/2) s K COS SIN k_row_stride sl cos_row_stride sin_row_stride hd idx) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_rope_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0; subst s0; intro idx hact
    simp only [ComputeCorrect.OutputReadable.read_real]
    have h := rope_backward_q0_backward_correct Q K COS SIN
      q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE s hundef hqk idx hact
    rw [hExec] at h
    exact h
  · rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_rope_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0; subst s0; intro idx hact
    simp only [ComputeCorrect.OutputReadable.read_real]
    have h := rope_backward_q1_backward_correct Q K COS SIN
      q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE s hundef hqk idx hact
    rw [hExec] at h
    exact h
  · rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_rope_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0; subst s0; intro idx hact
    simp only [ComputeCorrect.OutputReadable.read_real]
    have h := rope_backward_k0_backward_correct Q K COS SIN
      q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE s hundef idx hact
    rw [hExec] at h
    exact h
  · rw [ComputeCorrect.realizes_writeIf_iff]
    apply ComputeKernel.computeCorrect_of_toAlgKernel
    · simp [triton_rope_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    intro s0 s' hExec hs0; subst s0; intro idx hact
    simp only [ComputeCorrect.OutputReadable.read_real]
    have h := rope_backward_k1_backward_correct Q K COS SIN
      q_row_stride k_row_stride cos_row_stride sin_row_stride
      sl bs n_qh n_kh hd pad_n_qh pad_n_kh pad_hd BLOCK_SIZE s hundef idx hact
    rw [hExec] at h
    exact h

end VeriTile.Bench.TritonBenchG.RopeBackwardTransform
