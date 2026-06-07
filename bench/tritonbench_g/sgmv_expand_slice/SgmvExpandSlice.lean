import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant

/-!
# `sgmv_expand_slice` — full closed-form correctness

`_sgmv_expand_slice_kernel` is a segmented (multi-sequence) LoRA *expand* GEMM.
Program `(pid, cur_batch)` decomposes its CTA into an `(pid_m, pid_n)` tile,
gathers the batch's sequence metadata (`seq_lens[cur_batch]`,
`b_seq_start_loc[cur_batch]`) and LoRA index (`lora_indices[cur_batch]`, with the
signed `-1` sentinel skipping the batch), then accumulates a
`BLOCK_M × BLOCK_N` output tile by looping over the rank dimension `K` with
`accumulator += tl.dot(tiled_a, tiled_b)`, and finally **masked-stores** the tile
into the output slice at `(cur_seq_start + row, n + slice_offset)`.

This file proves the **full K-loop** correct against a genuine mathematical
contraction: every active output cell equals

```
out[cur_seq_start + offset_m, offset_n + slice_offset]
  = Σ_{k < K}  input[cur_seq_start + (offset_m % M), k]
             · loraB[lora_index, offset_n % N, k]
```

over `ℝ`, where `K = BLOCK_K · numKBlocks` is the rank, `offset_m = pid_m·BLOCK_M + i`,
`offset_n = pid_n·BLOCK_N + j`, and the `% M` / `% N` gathers
(`ram`/`rbn` = `tl.max_contiguous(tl.multiple_of(offset % size, BLOCK))`) are the
kernel's own address arithmetic. This is the independent closed-form `Σ_k a·b`
reference derived from the loaded `input`/`loraB` tiles — NOT the kernel's own
emitted value.

## Proof architecture (mirrors `matmul_triton1`/`AttentionForwardClosedForm`)

```
sgmv_expand_slice_closed_form_correct            ← TOP THEOREM (ComputeCorrect.Realizes)
  └─ sgmv_exec_closed_form                        ← exec-side closed form (every active cell = Σ_k a·b)
       ├─ sgmv_preLoop      (P 0: acc = 0, pointers/metadata seeded)
       ├─ sgmv_step         (one K-block: acc += dot advances the partial sum)
       ├─ sgmv_postLoop     (final masked store = the closed form on active lanes)
       └─ forRange_inv      (loop-invariant principle, drives the K-loop)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float); `@triton.autotune` /
`num_warps` are not modeled. The host launch (grid, the `(pid, cur_batch)` ↦
`(pid_m, pid_n)` linearization, the `pid_m·BLOCK_M > M` early-return, and the
`lora_index == -1` sentinel skip) is the *trusted boundary*; the per-program
statement is universally quantified over `s`, so it covers every program of the
grid. The proven path is `ADD_INPUTS = false`, `CAST_TYPE = false`, `EVEN_K`
load (`K = BLOCK_K · numKBlocks`) — the kernel's main numeric path. The metadata
(`cur_seq_start`, `lora_index`, `M`) are read data-dependently from the
`b_seq_start_loc` / `lora_indices` / `seq_lens` regions exactly as the kernel
loads them. `tl.max_contiguous` / `tl.multiple_of` are layout hints erased into
the same `% M` / `% N` value expression. The masked store is scoped to *active*
lanes (`offset_m < M` and `offset_n < N`); an output-offset injectivity
hypothesis `hInj` (distinct active lanes hit distinct addresses) is a side
condition, discharged for the Python test shape.
-/

namespace VeriTile.Bench.TritonBenchG.SgmvExpandSlice

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Faithful transcription of `sgmv_expand_slice.py`'s `_sgmv_expand_slice_kernel`
core (the `ADD_INPUTS = false`, `CAST_TYPE = false`, `EVEN_K` numeric path).

Program ids: `pid_m` (axis 0), `pid_n` (axis 1), `cur_batch` (axis 2) — the host
grid linearizes `(pid, cur_batch) ↦ (pid_m, pid_n)`; that linearization and the
`pid_m·BLOCK_M > M` / `lora_index == -1` early returns are the trusted boundary.

The metadata loads, the `ram = offset_m % M` / `rbn = offset_n % N` gathers
(`tl.max_contiguous`/`tl.multiple_of` are layout hints erased to the same value),
the K-block `tl.dot` accumulation loop, and the final masked store are all
transcribed. -/
def sgmv_expand_slice_surface
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ComputeKernel := triton {
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  cur_batch = tl.program_id(2)
  M = tl.load(seq_lens + cur_batch)
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch)
  lora_index = tl.load(lora_indices + cur_batch)
  offset_m = tl.arange(0, $(BLOCK_M)) + pid_m * $(BLOCK_M)
  offset_n = tl.arange(0, $(BLOCK_N)) + pid_n * $(BLOCK_N)
  offset_k = tl.arange(0, $(BLOCK_K))
  ram = offset_m % M
  rbn = offset_n % $(N)
  a_ptr = input_ptr + cur_seq_start * $(xm_stride) +
    ram[:, None] * $(xm_stride) + offset_k[None, :] * $(xk_stride)
  b_ptr = lora_ptr + $(l0_stride) * lora_index +
    offset_k[:, None] * $(lora_n_stride) + rbn[None, :] * $(lora_k_stride)
  accumulator = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
  for kk in range($(0), $(K), $(BLOCK_K)) {
    tiled_a = tl.load(a_ptr)
    tiled_b = tl.load(b_ptr)
    accumulator += tl.dot(tiled_a, tiled_b)
    a_ptr += $(BLOCK_K) * $(xk_stride)
    b_ptr += $(BLOCK_K) * $(lora_n_stride)
  }
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + $(slice_offset)
  c_ptr = out_ptr + offset_cm[:, None] * $(cm_stride) +
    offset_cn[None, :] * $(cn_stride)
  c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < $(N))
  tl.store(c_ptr, accumulator, mask=c_mask)
}

/-- The full SGMV expand-slice surface lowers to the algorithm layer. -/
theorem sgmv_expand_slice_surface_toAlgorithm_supported
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ∃ alg, (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc
      seq_lens lora_indices N K xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N
      BLOCK_K).toAlgorithm? = Except.ok alg := by
  simp [sgmv_expand_slice_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

/-! ## exec-stepping helpers (ported from `MatmulTriton1`/`AttentionForwardClosedForm`) -/

theorem evalOp_mod' {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_expandDim_zero_nat {D : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [1, D] (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [D] name)) s =
      (s.regs .nat [D] name).bind (fun v =>
        some ({ data := fun i : TileIndex [1, D] => v.data (i.2.1, PUnit.unit) } : Tile .nat [1, D])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

@[simp] theorem evalOp_expandDim_one_nat {M : Nat} (name : RegName) (s : BlockState) :
    @evalOp .nat [M, 1] (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] name)) s =
      (s.regs .nat [M] name).bind (fun v =>
        some ({ data := fun i : TileIndex [M, 1] => v.data (i.1, PUnit.unit) } : Tile .nat [M, 1])) := by
  unfold evalOp; simp [Tile.expandDim]; rfl

theorem evalOp_ptrAdd' {a b shape} (bc : Broadcast a b shape)
    (ptr : Op .ptr a) (off : Op .nat b) (s : BlockState) :
    evalOp (.ptrAdd bc ptr off) s = (do
      let ptrs ← evalOp ptr s; let offs ← evalOp off s;
      some (Tile.ptrAdd bc ptrs offs)) := by simp [evalOp]

theorem evalOp_ptrBase' (region : RegionName) (s : BlockState) :
    evalOp (.ptrBase region) s = some (Tile.scalar (region.cast, 0)) := by simp [evalOp]

/-- No-mask `.ptr` load: reads `readMem` at each pointer (clean `undef`). -/
theorem load_ptr_none_real {shape : TileShape}
    (ptrOp : Op .ptr shape) (s : BlockState) (ptrs : Tile .ptr shape)
    (hp : evalOp ptrOp s = some ptrs) :
    evalOp (.load .real (.ptr ptrOp) .none) s
      = some ⟨fun i => some (s.readMem (ptrs.data i).1 (ptrs.data i).2)⟩ := by
  simp only [evalOp, hp]
  refine congrArg some ?_
  ext i
  simp [BlockState.readMemValue_real]

/-! ## SGMV closed-form spec -/

/-- `M = seq_lens[cur_batch]` (this program's sequence length). -/
def seqLen (s : BlockState) (seq_lens : Region .nat) : Nat :=
  s.readMemValue .nat seq_lens.cast (s.pids 2)

/-- `cur_seq_start = b_seq_start_loc[cur_batch]` (this program's token offset). -/
def seqStart (s : BlockState) (b_seq_start_loc : Region .nat) : Nat :=
  s.readMemValue .nat b_seq_start_loc.cast (s.pids 2)

/-- `lora_index = lora_indices[cur_batch]` (this program's LoRA slot). -/
def loraIdx (s : BlockState) (lora_indices : Region .nat) : Nat :=
  s.readMemValue .nat lora_indices.cast (s.pids 2)

/-- Global row index `offset_m = pid_m·BLOCK_M + i` of tile lane `i`. -/
def rowG (s : BlockState) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  s.pids 0 * BLOCK_M + i.val

/-- Global col index `offset_n = pid_n·BLOCK_N + j` of tile lane `j`. -/
def colG (s : BlockState) (BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  s.pids 1 * BLOCK_N + j.val

/-- Gathered input row `ram = offset_m % M` (`tl.max_contiguous`/`tl.multiple_of`
are layout hints erased to this `%`). -/
def ramRow (s : BlockState) (seq_lens : Region .nat) (BLOCK_M : Nat) (i : Fin BLOCK_M) : Nat :=
  rowG s BLOCK_M i % seqLen s seq_lens

/-- Gathered LoRA col `rbn = offset_n % N`. -/
def rbnCol (s : BlockState) (N BLOCK_N : Nat) (j : Fin BLOCK_N) : Nat :=
  colG s BLOCK_N j % N

/-- `input[gathered_row, k]` at the kernel's row-major flattened address
`input + cur_seq_start·xm + ram·xm + k·xk`. -/
noncomputable def aElem (s : BlockState) (input_ptr : RegionName)
    (b_seq_start_loc seq_lens : Region .nat)
    (xm_stride xk_stride BLOCK_M : Nat) (i : Fin BLOCK_M) (k : Nat) : ℝ :=
  s.readMem input_ptr
    (seqStart s b_seq_start_loc * xm_stride + ramRow s seq_lens BLOCK_M i * xm_stride
      + k * xk_stride)

/-- `loraB[lora_index, gathered_col, k]` at the kernel's address
`lora + l0·lora_index + k·lora_n + rbn·lora_k`. -/
noncomputable def bElem (s : BlockState) (lora_ptr : RegionName)
    (lora_indices : Region .nat)
    (N l0_stride lora_k_stride lora_n_stride BLOCK_N : Nat) (j : Fin BLOCK_N) (k : Nat) : ℝ :=
  s.readMem lora_ptr
    (l0_stride * loraIdx s lora_indices + k * lora_n_stride
      + rbnCol s N BLOCK_N j * lora_k_stride)

/-- **Genuine SGMV spec**: `out[i,j] = Σ_{k < BLOCK_K·numKBlocks} aElem i k · bElem j k`. -/
noncomputable def sgmvSpec (s : BlockState)
    (input_ptr lora_ptr : RegionName) (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (i : Fin BLOCK_M) (j : Fin BLOCK_N) : ℝ :=
  (Finset.range (BLOCK_K * numKBlocks)).sum
    (fun k => aElem s input_ptr b_seq_start_loc seq_lens xm_stride xk_stride BLOCK_M i k
      * bElem s lora_ptr lora_indices N l0_stride lora_k_stride lora_n_stride BLOCK_N j k)

/-- Partial accumulator after `c` K-blocks: `Σ_{k < c·BLOCK_K} aElem · bElem`. -/
noncomputable def accPartial (s : BlockState)
    (input_ptr lora_ptr : RegionName) (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K : Nat) (i : Fin BLOCK_M) (j : Fin BLOCK_N) (c : Nat) : ℝ :=
  (Finset.range (c * BLOCK_K)).sum
    (fun k => aElem s input_ptr b_seq_start_loc seq_lens xm_stride xk_stride BLOCK_M i k
      * bElem s lora_ptr lora_indices N l0_stride lora_k_stride lora_n_stride BLOCK_N j k)

/-- One-block step of the partial accumulator: the new block's dot is over the
`BLOCK_K` keys `c·BLOCK_K + e`. -/
theorem accPartial_succ (s : BlockState)
    (input_ptr lora_ptr : RegionName) (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K : Nat) (i : Fin BLOCK_M) (j : Fin BLOCK_N) (c : Nat) :
    accPartial s input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K i j (c + 1)
      = accPartial s input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K i j c
        + (Finset.univ.sum fun e : Fin BLOCK_K =>
            aElem s input_ptr b_seq_start_loc seq_lens xm_stride xk_stride BLOCK_M i (c * BLOCK_K + e.val)
              * bElem s lora_ptr lora_indices N l0_stride lora_k_stride lora_n_stride BLOCK_N j (c * BLOCK_K + e.val)) := by
  unfold accPartial
  have h : (c + 1) * BLOCK_K = c * BLOCK_K + BLOCK_K := by ring
  rw [h, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_range fun e =>
    aElem s input_ptr b_seq_start_loc seq_lens xm_stride xk_stride BLOCK_M i (c * BLOCK_K + e)
      * bElem s lora_ptr lora_indices N l0_stride lora_k_stride lora_n_stride BLOCK_N j (c * BLOCK_K + e)]

/-- The output store address for tile lane `(i,j)`:
`(cur_seq_start + offset_m)·cm + (offset_n + slice_offset)·cn`. -/
def cOffset (s : BlockState) (b_seq_start_loc : Region .nat)
    (cm_stride cn_stride slice_offset BLOCK_M BLOCK_N : Nat) (idx : TileIndex [BLOCK_M, BLOCK_N]) : Nat :=
  (seqStart s b_seq_start_loc + rowG s BLOCK_M idx.1) * cm_stride
    + (colG s BLOCK_N idx.2.1 + slice_offset) * cn_stride

/-- A lane is *active* iff `offset_m < M` and `offset_n < N`. -/
def activeLane (s : BlockState) (seq_lens : Region .nat) (N BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Prop :=
  rowG s BLOCK_M idx.1 < seqLen s seq_lens ∧ colG s BLOCK_N idx.2.1 < N

instance activeLaneDecidable (s : BlockState) (seq_lens : Region .nat) (N BLOCK_M BLOCK_N : Nat)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) : Decidable (activeLane s seq_lens N BLOCK_M BLOCK_N idx) := by
  unfold activeLane; infer_instance

/-! ## pointer-tile eval lemmas -/

/-- `a_ptr` eval: cell `(i,e) = (input, css·xm + ram i·xm + offk e·xk)`. -/
theorem aptr_eval (s : BlockState) (input_ptr : RegionName)
    (M BK XM XK css : Nat) (gram : Fin M → Nat)
    (hram : s.regs .nat [M] "ram" = some (Tile.vec gram))
    (hcss : s.regs .nat [] "cur_seq_start" = some (Tile.scalar css))
    (hk : s.regs .nat [BK] "offset_k" = some (Tile.vec (fun e : Fin BK => e.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase input_ptr)
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_seq_start") (Op.constNat XM))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "ram")) (Op.constNat XM)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BK] "offset_k")) (Op.constNat XK)))) s
      = some (⟨fun idx : TileIndex [M, BK] => (input_ptr.cast, css * XM + gram idx.1 * XM + idx.2.1.val * XK)⟩ : Tile .ptr [M, BK]) := by
  rw [evalOp_ptrAdd', evalOp_ptrBase']
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hram, hcss, hk, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `b_ptr` eval: cell `(e,j) = (lora, L0·li + offk e·LN + rbn j·LK)`. -/
theorem bptr_eval (s : BlockState) (lora_ptr : RegionName)
    (BK N L0 LK LN li : Nat) (grbn : Fin N → Nat)
    (hli : s.regs .nat [] "lora_index" = some (Tile.scalar li))
    (hk : s.regs .nat [BK] "offset_k" = some (Tile.vec (fun e : Fin BK => e.val)))
    (hrbn : s.regs .nat [N] "rbn" = some (Tile.vec grbn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase lora_ptr)
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.add .nat Broadcast.scalarL
          (Op.mul .nat Broadcast.nil (Op.constNat L0) (Op.ref .nat [] "lora_index"))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BK] "offset_k")) (Op.constNat LN)))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "rbn")) (Op.constNat LK)))) s
      = some (⟨fun idx : TileIndex [BK, N] => (lora_ptr.cast, L0 * li + idx.1.val * LN + grbn idx.2.1 * LK)⟩ : Tile .ptr [BK, N]) := by
  rw [evalOp_ptrAdd', evalOp_ptrBase']
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hli, hk, hrbn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `c_ptr` eval: cell `(i,j) = (out, offcm i·CM + offcn j·CN)`. -/
theorem cptr_eval (s : BlockState) (out_ptr : RegionName)
    (M N CM CN : Nat) (gcm : Fin M → Nat) (gcn : Fin N → Nat)
    (hcm : s.regs .nat [M] "offset_cm" = some (Tile.vec gcm))
    (hcn : s.regs .nat [N] "offset_cn" = some (Tile.vec gcn)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase out_ptr)
      (Op.add .nat Broadcast.nil.consL.consR
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offset_cm")) (Op.constNat CM))
        (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [N] "offset_cn")) (Op.constNat CN)))) s
      = some (⟨fun idx : TileIndex [M, N] => (out_ptr.cast, gcm idx.1 * CM + gcn idx.2.1 * CN)⟩ : Tile .ptr [M, N]) := by
  rw [evalOp_ptrAdd', evalOp_ptrBase']
  simp only [evalOp_add, evalOp_mul, evalOp_constNat, evalOp_ref, evalOp_expandDim_one_nat,
    evalOp_expandDim_zero_nat, hcm, hcn, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul]

/-- `a_ptr += BLOCK_K · xk_stride`. -/
theorem aptr_adv_eval (s : BlockState) (M BK XK BLOCK_K : Nat) (ap : Tile .ptr [M, BK])
    (ha : s.regs .ptr [M, BK] "a_ptr" = some ap) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, BK] "a_ptr")
      (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_K) (Op.constNat XK))) s
      = some (Tile.ptrAdd Broadcast.scalarR ap (Tile.scalar (BLOCK_K * XK))) := by
  rw [evalOp_ptrAdd']
  simp [evalOp_ref, ha, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- `b_ptr += BLOCK_K · lora_n_stride`. -/
theorem bptr_adv_eval (s : BlockState) (BK N LN BLOCK_K : Nat) (bp : Tile .ptr [BK, N])
    (hb : s.regs .ptr [BK, N] "b_ptr" = some bp) :
    evalOp (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, N] "b_ptr")
      (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_K) (Op.constNat LN))) s
      = some (Tile.ptrAdd Broadcast.scalarR bp (Tile.scalar (BLOCK_K * LN))) := by
  rw [evalOp_ptrAdd']
  simp [evalOp_ref, hb, evalOp_mul, evalOp_constNat, NumericDType.mul, Tile.bop]

/-- `accumulator` init: `tl.zeros` → the all-`0` tile. -/
theorem acc_init_eval (s : BlockState) (M N : Nat) :
    evalOp (Op.full [M, N] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [M, N] => some (0 : ℝ)⟩ : Tile .real [M, N]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- The masked dot of two all-`some` loaded tiles, lane `(i,j)`. -/
theorem dot_ab (M K N : Nat) (x : Tile .real [M, K]) (y : Tile .real [K, N])
    (i : Fin M) (j : Fin N) (fx : Fin K → ℝ) (fy : Fin K → ℝ)
    (hx : ∀ e : Fin K, x.data (i, e, PUnit.unit) = some (fx e))
    (hy : ∀ e : Fin K, y.data (e, j, PUnit.unit) = some (fy e)) :
    (Tile.dot [] x y).data (i, j, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin K => fx e * fy e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (x.data (i, e, PUnit.unit)) (y.data (e, j, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ (fun e => (some (fx e * fy e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [hx e, hy e]; rfl)]
  show (Finset.univ.sum fun e => ((fx e * fy e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]; rfl

/-- **`accumulator = accumulator + tl.dot(tiled_a, tiled_b)` eval.** -/
theorem accdot_op_eval (M K N : Nat) (st : BlockState)
    (zt : Tile .real [M, N]) (xt : Tile .real [M, K]) (yt : Tile .real [K, N])
    (hz : st.regs .real [M, N] "accumulator" = some zt)
    (hx : st.regs .real [M, K] "tiled_a" = some xt)
    (hy : st.regs .real [K, N] "tiled_b" = some yt) :
    evalOp (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] "accumulator")
        (Op.dot (batch := []) (Op.ref .real [M, K] "tiled_a") (Op.ref .real [K, N] "tiled_b"))) st
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          zt (Tile.dot [] xt yt)) := by
  have hd : evalOp (Op.dot (batch := []) (Op.ref .real [M, K] "tiled_a")
        (Op.ref .real [K, N] "tiled_b")) st = some (Tile.dot [] xt yt) := by
    rw [evalOp_dot]; simp [hx, hy]
  rw [evalOp_add]
  simp only [evalOp_ref, hz, bind, Option.bind_some]
  erw [hd]
  rfl

/-- `acc + dot` lane `(i,j)`: `some (zv + dv)`. -/
theorem accadd_eval (M N : Nat) (zt dt : Tile .real [M, N]) (i : Fin M) (j : Fin N) (zv dv : ℝ)
    (hz : zt.data (i, j, PUnit.unit) = some zv) (hd : dt.data (i, j, PUnit.unit) = some dv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) zt dt).data
        (i, j, PUnit.unit) = some (zv + dv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hz, hd, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-! ## Body decomposition -/

/-- The 5-statement K-loop body, transcribed. -/
def sgmvLoopBody (M N BK XK LN BLOCK_K : Nat) : List Stmt :=
  [ Stmt.assign .real [M, BK] "tiled_a"
      (Op.load .real (.ptr (Op.ref .ptr [M, BK] "a_ptr")) .none),
    Stmt.assign .real [BK, N] "tiled_b"
      (Op.load .real (.ptr (Op.ref .ptr [BK, N] "b_ptr")) .none),
    Stmt.assign .real [M, N] "accumulator"
      (Op.add .real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref .real [M, N] "accumulator")
        (Op.dot (batch := []) (Op.ref .real [M, BK] "tiled_a") (Op.ref .real [BK, N] "tiled_b"))),
    Stmt.assign .ptr [M, BK] "a_ptr"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [M, BK] "a_ptr")
        (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_K) (Op.constNat XK))),
    Stmt.assign .ptr [BK, N] "b_ptr"
      (Op.ptrAdd Broadcast.scalarR (Op.ref .ptr [BK, N] "b_ptr")
        (Op.mul .nat Broadcast.nil (Op.constNat BLOCK_K) (Op.constNat LN))) ]

/-- Post-loop masked store: the four offset/ptr/mask statements plus the store.
`BN` is the tile column dim (`BLOCK_N`), `Nb` the mask bound (`N`). -/
def sgmvStoreTail (out_ptr : RegionName) (M BN Nb CM CN slice_offset : Nat) : List Stmt :=
  [ Stmt.assign .nat [M] "offset_cm"
      (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_seq_start") (Op.ref .nat [M] "offset_m")),
    Stmt.assign .nat [BN] "offset_cn"
      (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "offset_n") (Op.constNat slice_offset)),
    Stmt.assign .ptr [M, BN] "c_ptr"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase out_ptr)
        (Op.add .nat Broadcast.nil.consL.consR
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offset_cm")) (Op.constNat CM))
          (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offset_cn")) (Op.constNat CN)))),
    Stmt.assign .bool [M, BN] "c_mask"
      (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [M] "offset_m")) (Op.ref .nat [] "M"))
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offset_n")) (Op.constNat Nb))),
    Stmt.store .real [M, BN] (.ptr (Op.ref .ptr [M, BN] "c_ptr"))
      (Op.ref .real [M, BN] "accumulator") (.mask (Op.ref .bool [M, BN] "c_mask")) ]

/-- Body decomposition: prologue (14) ++ [for-loop] ++ store tail (5). By `rfl`. -/
theorem sgmv_body_split (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) :
    (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body
      = (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
          lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
          lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body.take 14
        ++ (Stmt.forRange "kk" 0 (BLOCK_K * numKBlocks) BLOCK_K
              (sgmvLoopBody BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K)
            :: sgmvStoreTail out_ptr BLOCK_M BLOCK_N N cm_stride cn_stride slice_offset) := by
  simp only [sgmv_expand_slice_surface, ComputeKernel.toAlgKernel_mk,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Except.ok.injEq,
    sgmvLoopBody, sgmvStoreTail, List.take, List.append, List.cons_append, List.nil_append]
  rfl

/-! ## Loop invariant -/

/-- **Loop invariant** (counter `i = c · BLOCK_K`).

After `c` K-blocks: program ids, `mem`/`undef` fixed; the prologue scalars
(`cur_seq_start`/`lora_index`/`M`) and index vectors (`offset_m`/`offset_n`/
`offset_k`/`ram`/`rbn`) seeded; `accumulator` equals the partial accumulator
`accPartial … c`; and `a_ptr`/`b_ptr` advanced by `c` blocks. -/
noncomputable def sgmvInvariant
    (input_ptr lora_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s0 : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (i : Nat) (s : BlockState) : Prop :=
  let c := i / BLOCK_K
  s.pids = s0.pids ∧ i = c * BLOCK_K ∧ c ≤ numKBlocks ∧
  (s.regs .real [BLOCK_M, BLOCK_N] "accumulator" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      some (accPartial s0 input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 c)⟩) ∧
  (s.regs .nat [BLOCK_M] "offset_m" = some (Tile.vec (fun r : Fin BLOCK_M => rowG s0 BLOCK_M r))) ∧
  (s.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => colG s0 BLOCK_N j))) ∧
  (s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun e : Fin BLOCK_K => e.val))) ∧
  (s.regs .nat [BLOCK_M] "ram" = some (Tile.vec (fun r : Fin BLOCK_M => ramRow s0 seq_lens BLOCK_M r))) ∧
  (s.regs .nat [BLOCK_N] "rbn" = some (Tile.vec (fun j : Fin BLOCK_N => rbnCol s0 N BLOCK_N j))) ∧
  (s.regs .nat [] "cur_seq_start" = some (Tile.scalar (seqStart s0 b_seq_start_loc))) ∧
  (s.regs .nat [] "lora_index" = some (Tile.scalar (loraIdx s0 lora_indices))) ∧
  (s.regs .nat [] "M" = some (Tile.scalar (seqLen s0 seq_lens))) ∧
  (s.regs .ptr [BLOCK_M, BLOCK_K] "a_ptr" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_K] =>
      (input_ptr.cast, seqStart s0 b_seq_start_loc * xm_stride
        + ramRow s0 seq_lens BLOCK_M idx.1 * xm_stride + (idx.2.1.val + c * BLOCK_K) * xk_stride)⟩) ∧
  (s.regs .ptr [BLOCK_K, BLOCK_N] "b_ptr" = some ⟨fun idx : TileIndex [BLOCK_K, BLOCK_N] =>
      (lora_ptr.cast, l0_stride * loraIdx s0 lora_indices
        + (idx.1.val + c * BLOCK_K) * lora_n_stride + rbnCol s0 N BLOCK_N idx.2.1 * lora_k_stride)⟩) ∧
  (∀ rg o, s.undef rg o = 0) ∧ (s.mem = s0.mem)

/-- **preLoop scalars** (statements 0–10): the 6 metadata scalars + the 5 index
vectors (`offset_m`/`offset_n`/`offset_k`/`ram`/`rbn`). -/
theorem preLoop_scalars (s : BlockState)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N BLOCK_M BLOCK_N BLOCK_K : Nat) :
    ∃ s11, stepStmts
        [ Stmt.assign .nat [] "pid_m" (Op.programId 0),
          Stmt.assign .nat [] "pid_n" (Op.programId 1),
          Stmt.assign .nat [] "cur_batch" (Op.programId 2),
          Stmt.assign .nat [] "M" (Op.load .nat (.region seq_lens (Op.ref .nat [] "cur_batch")) .none),
          Stmt.assign .nat [] "cur_seq_start" (Op.load .nat (.region b_seq_start_loc (Op.ref .nat [] "cur_batch")) .none),
          Stmt.assign .nat [] "lora_index" (Op.load .nat (.region lora_indices (Op.ref .nat [] "cur_batch")) .none),
          Stmt.assign .nat [BLOCK_M] "offset_m"
            (Op.add .nat Broadcast.scalarR (Op.arange BLOCK_M)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BLOCK_M))),
          Stmt.assign .nat [BLOCK_N] "offset_n"
            (Op.add .nat Broadcast.scalarR (Op.arange BLOCK_N)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BLOCK_N))),
          Stmt.assign .nat [BLOCK_K] "offset_k" (Op.arange BLOCK_K),
          Stmt.assign .nat [BLOCK_M] "ram"
            (Op.mod .nat Broadcast.scalarR (Op.ref .nat [BLOCK_M] "offset_m") (Op.ref .nat [] "M")),
          Stmt.assign .nat [BLOCK_N] "rbn"
            (Op.mod .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_n") (Op.constNat N)) ] s = some s11
      ∧ s11.pids = s.pids
      ∧ s11.regs .nat [] "cur_seq_start" = some (Tile.scalar (seqStart s b_seq_start_loc))
      ∧ s11.regs .nat [] "lora_index" = some (Tile.scalar (loraIdx s lora_indices))
      ∧ s11.regs .nat [] "M" = some (Tile.scalar (seqLen s seq_lens))
      ∧ s11.regs .nat [BLOCK_M] "offset_m" = some (Tile.vec (fun r : Fin BLOCK_M => rowG s BLOCK_M r))
      ∧ s11.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => colG s BLOCK_N j))
      ∧ s11.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun e : Fin BLOCK_K => e.val))
      ∧ s11.regs .nat [BLOCK_M] "ram" = some (Tile.vec (fun r : Fin BLOCK_M => ramRow s seq_lens BLOCK_M r))
      ∧ s11.regs .nat [BLOCK_N] "rbn" = some (Tile.vec (fun j : Fin BLOCK_N => rbnCol s N BLOCK_N j))
      ∧ s11.undef = s.undef
      ∧ s11.mem = s.mem := by
  simp only [rowG, colG, ramRow, rbnCol, seqStart, loraIdx, seqLen]
  simp only [stepStmts, stepStmt, evalOp_mod', evalOp_load_region_none,
    evalOp_programId, evalOp_arange, evalOp_add, evalOp_mul, evalOp_constNat,
    Option.bind_some, bind, Option.bind_eq_bind,
    BlockState.setReg_readMemTyped, BlockState.setReg_readMemAs,
    BlockState.setReg_pids,
    evalOp_ref_setReg, String.reduceEq, reduceIte, dite_true, dite_false,
    Tile.bop, Tile.vec, NumericDType.add, NumericDType.mul, IntegralDType.mod,
    BlockState.readMemValue]
  refine ⟨_, rfl, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- cur_seq_start
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, castTile, Region.cast_id, Region.cast_cast]
    refine congrArg some ?_; ext idx
    simp [seqStart, BlockState.readMemValue]
  · -- lora_index
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, castTile, Region.cast_id, Region.cast_cast]
    refine congrArg some ?_; ext idx
    simp [loraIdx, BlockState.readMemValue]
  · -- M
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, castTile, Region.cast_id, Region.cast_cast]
    refine congrArg some ?_; ext idx
    simp [seqLen, BlockState.readMemValue]
  · -- offset_m
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, castTile]
    refine congrArg some ?_; ext idx
    simp [Tile.vec, rowG, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]
    ring
  · -- offset_n
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, castTile]
    refine congrArg some ?_; ext idx
    simp [Tile.vec, colG, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.mul]
    ring
  · -- offset_k
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, castTile, Tile.vec]
  · -- ram
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, castTile]
    refine congrArg some ?_; ext idx
    simp only [Tile.vec, ramRow, rowG, seqLen, BlockState.readMemValue,
      Broadcast.leftIndex, Broadcast.rightIndex, IntegralDType.mod, Tile.scalar,
      Region.cast_id, Region.cast_cast]
    rw [Nat.add_comm]
  · -- rbn
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff, castTile]
    refine congrArg some ?_; ext idx
    simp only [Tile.vec, rbnCol, colG, Broadcast.leftIndex, Broadcast.rightIndex, IntegralDType.mod,
      Tile.scalar]
    rw [Nat.add_comm]
  · -- undef
    funext rg o; simp only [BlockState.setReg_undef]
  · -- mem
    rfl

set_option maxHeartbeats 2000000 in
/-- **preLoop** (statements 0–13): from a clean input state (`undef = 0`), the
prologue steps to a state satisfying `sgmvInvariant … 0` (the base case:
`accumulator = 0`, pointers/metadata seeded). -/
theorem sgmv_preLoop (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ∃ s', stepStmts
        ((sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
            lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
            lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body.take 14) s = some s'
      ∧ sgmvInvariant input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K numKBlocks 0 s' := by
  obtain ⟨s11, h11, hpids, hcss, hli, hM, hom, hon, hok, hram, hrbn, huf, hmem⟩ :=
    preLoop_scalars s b_seq_start_loc seq_lens lora_indices N BLOCK_M BLOCK_N BLOCK_K
  rw [show ((sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body.take 14)
      = [ Stmt.assign .nat [] "pid_m" (Op.programId 0),
          Stmt.assign .nat [] "pid_n" (Op.programId 1),
          Stmt.assign .nat [] "cur_batch" (Op.programId 2),
          Stmt.assign .nat [] "M" (Op.load .nat (.region seq_lens (Op.ref .nat [] "cur_batch")) .none),
          Stmt.assign .nat [] "cur_seq_start" (Op.load .nat (.region b_seq_start_loc (Op.ref .nat [] "cur_batch")) .none),
          Stmt.assign .nat [] "lora_index" (Op.load .nat (.region lora_indices (Op.ref .nat [] "cur_batch")) .none),
          Stmt.assign .nat [BLOCK_M] "offset_m"
            (Op.add .nat Broadcast.scalarR (Op.arange BLOCK_M)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_m") (Op.constNat BLOCK_M))),
          Stmt.assign .nat [BLOCK_N] "offset_n"
            (Op.add .nat Broadcast.scalarR (Op.arange BLOCK_N)
              (Op.mul .nat Broadcast.nil (Op.ref .nat [] "pid_n") (Op.constNat BLOCK_N))),
          Stmt.assign .nat [BLOCK_K] "offset_k" (Op.arange BLOCK_K),
          Stmt.assign .nat [BLOCK_M] "ram"
            (Op.mod .nat Broadcast.scalarR (Op.ref .nat [BLOCK_M] "offset_m") (Op.ref .nat [] "M")),
          Stmt.assign .nat [BLOCK_N] "rbn"
            (Op.mod .nat Broadcast.scalarR (Op.ref .nat [BLOCK_N] "offset_n") (Op.constNat N)) ]
      ++ [ Stmt.assign .ptr [BLOCK_M, BLOCK_K] "a_ptr"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase input_ptr)
              (Op.add .nat Broadcast.nil.consL.consR
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil (Op.ref .nat [] "cur_seq_start") (Op.constNat xm_stride))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "ram")) (Op.constNat xm_stride)))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k")) (Op.constNat xk_stride)))),
          Stmt.assign .ptr [BLOCK_K, BLOCK_N] "b_ptr"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase lora_ptr)
              (Op.add .nat Broadcast.nil.consL.consR
                (Op.add .nat Broadcast.scalarL
                  (Op.mul .nat Broadcast.nil (Op.constNat l0_stride) (Op.ref .nat [] "lora_index"))
                  (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_K] "offset_k")) (Op.constNat lora_n_stride)))
                (Op.mul .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "rbn")) (Op.constNat lora_k_stride)))),
          Stmt.assign .real [BLOCK_M, BLOCK_N] "accumulator" (Op.full [BLOCK_M, BLOCK_N] (Op.const 0)) ] from by
      simp only [sgmv_expand_slice_surface, ComputeKernel.toAlgKernel_mk,
        ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?, Except.bind, bind, Except.ok.injEq, List.take, List.append,
        List.cons_append, List.nil_append]
      rfl,
    stepStmts.append_some h11,
    stepStmts.cons_some (stepStmt_assign_eq_some
      (aptr_eval s11 input_ptr BLOCK_M BLOCK_K xm_stride xk_stride (seqStart s b_seq_start_loc)
        (fun r => ramRow s seq_lens BLOCK_M r) (by simpa using hram) (by simpa using hcss) (by simpa using hok))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (bptr_eval _ lora_ptr BLOCK_K BLOCK_N l0_stride lora_k_stride lora_n_stride (loraIdx s lora_indices)
        (fun j : Fin BLOCK_N => rbnCol s N BLOCK_N j) (by simp [hli]) (by simp [hok]) (by simp [hrbn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval _ BLOCK_M BLOCK_N)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [sgmvInvariant, Nat.zero_div]
  refine ⟨by simp [hpids], by simp, by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- accumulator = accPartial 0 = 0
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_; ext idx
    simp only [accPartial, Nat.zero_mul, Finset.range_zero, Finset.sum_empty]
  · simp [hom]
  · simp [hon]
  · simp [hok]
  · simp [hram]
  · simp [hrbn]
  · simp [hcss]
  · simp [hli]
  · simp [hM]
  · -- a_ptr (c = 0)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_; ext idx <;> simp [Nat.zero_mul]
  · -- b_ptr (c = 0)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_; ext idx <;> simp [Nat.zero_mul]
  · intro rg o
    show s11.undef rg o = 0
    rw [huf, hundef]
  · show s11.mem = s.mem
    exact hmem

set_option maxHeartbeats 2000000 in
/-- **Step lemma**: one K-loop body iteration advances the invariant by one block
(`acc += tl.dot(tiled_a, tiled_b)` adds the `c`-th block's dot to the partial
accumulator; the `a_ptr`/`b_ptr` pointers advance one step). -/
theorem sgmv_step (input_ptr lora_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s0 : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (i : Nat) (s : BlockState) (hilt : i < BLOCK_K * numKBlocks)
    (hinv : sgmvInvariant input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s0
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K numKBlocks i s) :
    ∃ s', stepStmts (sgmvLoopBody BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K)
        (s.setReg "kk" .nat [] (Tile.scalar i)) = some s'
      ∧ sgmvInvariant input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s0
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K numKBlocks (i + BLOCK_K) s' := by
  have hc : i / BLOCK_K < numKBlocks := (Nat.div_lt_iff_lt_mul hBK).mpr (by rw [Nat.mul_comm]; exact hilt)
  have hc1 : (i + BLOCK_K) / BLOCK_K = i / BLOCK_K + 1 := Nat.add_div_right i hBK
  simp only [sgmvInvariant] at hinv
  obtain ⟨hpids, hi, hcle, hacc, hom, hon, hok, hram, hrbn, hcss, hli, hM, hap, hbp, hundef, hmem⟩ := hinv
  set c := i / BLOCK_K with hcdef
  set apT : Tile .ptr [BLOCK_M, BLOCK_K] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_K] =>
      (input_ptr.cast, seqStart s0 b_seq_start_loc * xm_stride
        + ramRow s0 seq_lens BLOCK_M idx.1 * xm_stride + (idx.2.1.val + c * BLOCK_K) * xk_stride)⟩ with hapT
  set bpT : Tile .ptr [BLOCK_K, BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_K, BLOCK_N] =>
      (lora_ptr.cast, l0_stride * loraIdx s0 lora_indices
        + (idx.1.val + c * BLOCK_K) * lora_n_stride + rbnCol s0 N BLOCK_N idx.2.1 * lora_k_stride)⟩ with hbpT
  set accT : Tile .real [BLOCK_M, BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      some (accPartial s0 input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 c)⟩ with haccT
  set sk := s.setReg "kk" .nat [] (Tile.scalar i) with hsk
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  have hapk : sk.regs .ptr [BLOCK_M, BLOCK_K] "a_ptr" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BLOCK_K, BLOCK_N] "b_ptr" = some bpT := by simp [hsk, hbp, hbpT]
  have hacck : sk.regs .real [BLOCK_M, BLOCK_N] "accumulator" = some accT := by simp [hsk, hacc, haccT]
  set asub : Tile .real [BLOCK_M, BLOCK_K] :=
    ⟨fun idx => some (sk.readMem (apT.data idx).1 (apT.data idx).2)⟩ with hasub
  set bsub : Tile .real [BLOCK_K, BLOCK_N] :=
    ⟨fun idx => some (sk.readMem (bpT.data idx).1 (bpT.data idx).2)⟩ with hbsub
  unfold sgmvLoopBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_none_real (Op.ref .ptr [BLOCK_M, BLOCK_K] "a_ptr") _ apT (by rw [evalOp_ref]; simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_none_real (Op.ref .ptr [BLOCK_K, BLOCK_N] "b_ptr") _ bpT (by rw [evalOp_ref]; simp [hbpk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (accdot_op_eval BLOCK_M BLOCK_K BLOCK_N _ accT asub bsub
          (by simp [hacck, hasub, hbsub, BlockState.setReg_readMem])
          (by simp [hasub, BlockState.setReg_readMem])
          (by simp [hbsub, BlockState.setReg_readMem])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (aptr_adv_eval _ BLOCK_M BLOCK_K xk_stride BLOCK_K apT (by simp [hapk])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (bptr_adv_eval _ BLOCK_K BLOCK_N lora_n_stride BLOCK_K bpT (by simp [hbpk])))]
  rw [stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [sgmvInvariant, hc1]
  refine ⟨by simp [hsk, hpids], by rw [hi]; ring, by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- accumulator = accPartial (c+1)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_; ext idx
    have has : ∀ e : Fin BLOCK_K, asub.data (idx.1, e, PUnit.unit)
        = some (aElem s0 input_ptr b_seq_start_loc seq_lens xm_stride xk_stride BLOCK_M idx.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hasub, hapT, hrmem, aElem]
      congr 2
      ring
    have hbs : ∀ e : Fin BLOCK_K, bsub.data (e, idx.2.1, PUnit.unit)
        = some (bElem s0 lora_ptr lora_indices N l0_stride lora_k_stride lora_n_stride BLOCK_N idx.2.1 (c * BLOCK_K + e.val)) := by
      intro e
      simp only [hbsub, hbpT, hrmem, bElem]
      congr 2
      ring
    rw [accadd_eval BLOCK_M BLOCK_N accT (Tile.dot [] asub bsub) idx.1 idx.2.1
        (accPartial s0 input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 c)
        (Finset.univ.sum fun e : Fin BLOCK_K =>
          aElem s0 input_ptr b_seq_start_loc seq_lens xm_stride xk_stride BLOCK_M idx.1 (c * BLOCK_K + e.val)
            * bElem s0 lora_ptr lora_indices N l0_stride lora_k_stride lora_n_stride BLOCK_N idx.2.1 (c * BLOCK_K + e.val))
        (by rw [haccT])
        (dot_ab BLOCK_M BLOCK_K BLOCK_N asub bsub idx.1 idx.2.1 _ _ has hbs)]
    show some _ = some (accPartial s0 input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
      N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 (c + 1))
    rw [accPartial_succ]
  · simp [hsk, hom]
  · simp [hsk, hon]
  · simp [hsk, hok]
  · simp [hsk, hram]
  · simp [hsk, hrbn]
  · simp [hsk, hcss]
  · simp [hsk, hli]
  · simp [hsk, hM]
  · -- a_ptr advanced
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_; ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex,
      Broadcast.rightIndex, Tile.scalar, hapT, NumericDType.add]
    ring
  · -- b_ptr advanced
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_; ext idx <;> simp only [Tile.ptrAdd, Tile.bop, Broadcast.leftIndex,
      Broadcast.rightIndex, Tile.scalar, hbpT, NumericDType.add]
    ring
  · intro rg o; simp [hsk, hundef]
  · show _ = s0.mem
    rw [← hmem, hsk]; rfl

/-! ## post-loop store-tail eval lemmas -/

/-- `offset_cm = cur_seq_start + offset_m` eval. -/
theorem offcm_eval (s : BlockState) (M css : Nat) (gom : Fin M → Nat)
    (hcss : s.regs .nat [] "cur_seq_start" = some (Tile.scalar css))
    (hom : s.regs .nat [M] "offset_m" = some (Tile.vec gom)) :
    evalOp (Op.add .nat Broadcast.scalarL (Op.ref .nat [] "cur_seq_start") (Op.ref .nat [M] "offset_m")) s
      = some (Tile.vec (fun i : Fin M => css + gom i)) := by
  rw [evalOp_add]; simp [evalOp_ref, hcss, hom, Tile.bop, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]

/-- `offset_cn = offset_n + slice_offset` eval. -/
theorem offcn_eval (s : BlockState) (BN so : Nat) (gon : Fin BN → Nat)
    (hon : s.regs .nat [BN] "offset_n" = some (Tile.vec gon)) :
    evalOp (Op.add .nat Broadcast.scalarR (Op.ref .nat [BN] "offset_n") (Op.constNat so)) s
      = some (Tile.vec (fun j : Fin BN => gon j + so)) := by
  rw [evalOp_add]; simp [evalOp_ref, hon, evalOp_constNat, Tile.bop, Tile.vec,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add]

set_option maxHeartbeats 2000000 in
/-- **postLoop**: from the invariant at `numKBlocks` blocks, the masked store of
`accumulator` writes the genuine SGMV value `sgmvSpec` at every *active* output
lane (`offset_m < M ∧ offset_n < N`), under output-offset injectivity. -/
theorem sgmv_postLoop (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s0 : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N))
    (st : BlockState)
    (hinv : sgmvInvariant input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s0
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K numKBlocks (BLOCK_K * numKBlocks) st) :
    ∃ sfin, stepStmts (sgmvStoreTail out_ptr BLOCK_M BLOCK_N N cm_stride cn_stride slice_offset) st = some sfin
      ∧ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
          sfin.readMem out_ptr
              (cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx)
            = if activeLane s0 seq_lens N BLOCK_M BLOCK_N idx then
                sgmvSpec s0 input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
                  N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
                  BLOCK_M BLOCK_N BLOCK_K numKBlocks idx.1 idx.2.1
              else
                st.readMem out_ptr
                  (cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx) := by
  have hcN : (BLOCK_K * numKBlocks) / BLOCK_K = numKBlocks := by
    rw [Nat.mul_comm, Nat.mul_div_cancel _ hBK]
  simp only [sgmvInvariant, hcN] at hinv
  obtain ⟨hpids, hi, hcle, hacc, hom, hon, hok, hram, hrbn, hcss, hli, hM, hap, hbp, hundef, hmem⟩ := hinv
  unfold sgmvStoreTail
  -- step offset_cm, offset_cn
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (offcm_eval st BLOCK_M (seqStart s0 b_seq_start_loc) (fun r => rowG s0 BLOCK_M r) hcss hom))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (offcn_eval _ BLOCK_N slice_offset (fun j => colG s0 BLOCK_N j) (by simp [hon])))]
  -- c_ptr eval (offset_cm/offset_cn now in regs)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (cptr_eval _ out_ptr BLOCK_M BLOCK_N cm_stride cn_stride
          (fun i => seqStart s0 b_seq_start_loc + rowG s0 BLOCK_M i)
          (fun j => colG s0 BLOCK_N j + slice_offset) (by simp) (by simp)))]
  -- c_mask eval
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.boolAnd Broadcast.nil.consL.consR
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offset_m")) (Op.ref .nat [] "M"))
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n")) (Op.constNat N))) _
          = some (⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
              decide (rowG s0 BLOCK_M idx.1 < seqLen s0 seq_lens) &&
                decide (colG s0 BLOCK_N idx.2.1 < N)⟩ : Tile .bool [BLOCK_M, BLOCK_N]) from by
          simp only [evalOp, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat,
            BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
          simp only [hom, hon, hM, Option.bind_some, bind, Option.bind_eq_bind]
          refine congrArg some ?_; ext idx
          simp only [Tile.bop, Tile.cop, Tile.vec, Broadcast.leftIndex, Broadcast.rightIndex,
            ComparableDType.lt, seqLen, rowG, colG]
          rfl))]
  -- the store
  set cmaskT : Tile .bool [BLOCK_M, BLOCK_N] :=
    ⟨fun idx => decide (rowG s0 BLOCK_M idx.1 < seqLen s0 seq_lens) && decide (colG s0 BLOCK_N idx.2.1 < N)⟩ with hcmaskT
  set cptrT : Tile .ptr [BLOCK_M, BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      (out_ptr.cast, (seqStart s0 b_seq_start_loc + rowG s0 BLOCK_M idx.1) * cm_stride
        + (colG s0 BLOCK_N idx.2.1 + slice_offset) * cn_stride)⟩ with hcptrT
  set accT : Tile .real [BLOCK_M, BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      some (accPartial s0 input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 numKBlocks)⟩ with haccT
  have hstore : stepStmt (Stmt.store .real [BLOCK_M, BLOCK_N]
        (.ptr (Op.ref .ptr [BLOCK_M, BLOCK_N] "c_ptr"))
        (Op.ref .real [BLOCK_M, BLOCK_N] "accumulator") (.mask (Op.ref .bool [BLOCK_M, BLOCK_N] "c_mask")))
        ((((st.setReg "offset_cm" .nat [BLOCK_M] (Tile.vec (fun i : Fin BLOCK_M => seqStart s0 b_seq_start_loc + rowG s0 BLOCK_M i))).setReg
            "offset_cn" .nat [BLOCK_N] (Tile.vec (fun j : Fin BLOCK_N => colG s0 BLOCK_N j + slice_offset))).setReg
            "c_ptr" .ptr [BLOCK_M, BLOCK_N] cptrT).setReg "c_mask" .bool [BLOCK_M, BLOCK_N] cmaskT)
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_N]).foldl
          (fun acc idx =>
            if (activeLane s0 seq_lens N BLOCK_M BLOCK_N idx) then
              acc.writeMem out_ptr
                (cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx)
                (accPartial s0 input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
                  N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 numKBlocks)
            else acc)
          ((((st.setReg "offset_cm" .nat [BLOCK_M] (Tile.vec (fun i : Fin BLOCK_M => seqStart s0 b_seq_start_loc + rowG s0 BLOCK_M i))).setReg
              "offset_cn" .nat [BLOCK_N] (Tile.vec (fun j : Fin BLOCK_N => colG s0 BLOCK_N j + slice_offset))).setReg
              "c_ptr" .ptr [BLOCK_M, BLOCK_N] cptrT).setReg "c_mask" .bool [BLOCK_M, BLOCK_N] cmaskT)) := by
    simp only [stepStmt]
    rw [show evalOp (Op.ref .real [BLOCK_M, BLOCK_N] "accumulator") _ = some accT from by
          rw [evalOp_ref]; simp [hacc, haccT]]
    rw [show evalOp (Op.ref .ptr [BLOCK_M, BLOCK_N] "c_ptr") _ = some cptrT from by
          rw [evalOp_ref]; simp [hcptrT]]
    rw [show evalOp (Op.ref .bool [BLOCK_M, BLOCK_N] "c_mask") _ = some cmaskT from by
          rw [evalOp_ref]; simp [hcmaskT]]
    simp only [bind, Option.bind_some, Option.map]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc idx _ => ?_))
    simp only [cmaskT, hcmaskT, cptrT, hcptrT, accT, haccT, cOffset, activeLane, seqLen,
      Bool.and_eq_true, decide_eq_true_eq, Region.cast_id, BlockState.writeMemTyped_real,
      FloatDType.real_storeValue, WithBot.unbotD_some]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [BlockState.scatter_readback_prop_masked_nd (region := out_ptr)
      (offsetFn := cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)
      (valueFn := fun idx => accPartial s0 input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 numKBlocks)
      (P := fun idx => activeLane s0 seq_lens N BLOCK_M BLOCK_N idx)
      (h_inj := hInj) (i := idx)]
  by_cases ha : activeLane s0 seq_lens N BLOCK_M BLOCK_N idx
  · rw [if_pos ha, if_pos ha]
    simp only [sgmvSpec, accPartial, Nat.mul_comm numKBlocks BLOCK_K]
  · rw [if_neg ha, if_neg ha]
    -- the readMem on the seeded state equals readMem on st (setReg preserves mem)
    show _ = st.readMem out_ptr _
    rfl

/-! ## Composition: full exec closed form -/

set_option maxHeartbeats 2000000 in
/-- **Top exec reduction**: composes `sgmv_preLoop` + `sgmv_step` (driven by
`forRange_inv`) + `sgmv_postLoop` into the full `exec` result. Every *active*
output lane equals the genuine SGMV value `sgmvSpec`. -/
theorem sgmv_exec_closed_form (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (cOffset s b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N))
    (hundef : ∀ rg o, s.undef rg o = 0)
    (idx : TileIndex [BLOCK_M, BLOCK_N]) :
    (match exec (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K) s with
      | some s' => s'.readMem out_ptr (cOffset s b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx)
      | none => (0.0 : ℝ)) =
      if activeLane s seq_lens N BLOCK_M BLOCK_N idx then
        sgmvSpec s input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K numKBlocks idx.1 idx.2.1
      else
        s.readMem out_ptr (cOffset s b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx) := by
  obtain ⟨s0, hpre_eq, hP0⟩ := sgmv_preLoop input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
    lora_indices s N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
    cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks hundef
  obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
    forRange_inv (idx := "kk") (start := 0) (stop := BLOCK_K * numKBlocks) (step := BLOCK_K)
      (by omega) hP0
      (fun i st hlt hinv => sgmv_step input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K numKBlocks
        hBK i st hlt hinv)
  have hfinalEq : final = BLOCK_K * numKBlocks := by
    have hle : final ≤ BLOCK_K * numKBlocks := by
      simp only [sgmvInvariant] at hPLoop
      obtain ⟨_, hieq, hcle, _⟩ := hPLoop
      calc final = final / BLOCK_K * BLOCK_K := hieq
        _ ≤ numKBlocks * BLOCK_K := Nat.mul_le_mul_right _ hcle
        _ = BLOCK_K * numKBlocks := Nat.mul_comm _ _
    exact le_antisymm hle hfinal
  subst hfinalEq
  obtain ⟨sfin, hTail, hpost⟩ := sgmv_postLoop input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
    lora_indices s N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
    cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks hBK hInj sLoop hPLoop
  have hexec : exec (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K) s = some sfin := by
    rw [exec, sgmv_body_split input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
      N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset
      BLOCK_M BLOCK_N BLOCK_K numKBlocks,
      stepStmts.append_some hpre_eq, stepStmts.cons_some hLoopStmt, hTail]
  have hsLoopMem : sLoop.mem = s.mem := by
    simp only [sgmvInvariant] at hPLoop
    exact hPLoop.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  rw [hexec]
  show sfin.readMem out_ptr _ = _
  rw [hpost idx]
  by_cases ha : activeLane s seq_lens N BLOCK_M BLOCK_N idx
  · rw [if_pos ha, if_pos ha]
  · rw [if_neg ha, if_neg ha]
    show sLoop.readMem out_ptr _ = s.readMem out_ptr _
    unfold BlockState.readMem; rw [hsLoopMem]

/-- **Closed-form correctness for `sgmv_expand_slice` (general statement).**

For arbitrary tile dims `BLOCK_M`/`BLOCK_N`, rank-block size `BLOCK_K`, and
rank-block count `numKBlocks` (so the rank `K = BLOCK_K · numKBlocks`), every
*active* output cell of the computed tile equals the genuine SGMV contraction

```
out[cur_seq_start + offset_m, offset_n + slice_offset]
  = Σ_{k < BLOCK_K·numKBlocks}  input[cur_seq_start + (offset_m % M), k]
                              · loraB[lora_index, offset_n % N, k]
```

(over ℝ) of the loaded `input`/`loraB` tiles — NOT the kernel's own executed
value. Preconditions: `0 < BLOCK_K`, output-offset injectivity, clean initial
`undef`. -/
theorem sgmv_expand_slice_closed_form_correct
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (cOffset s b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    ComputeCorrect.Realizes
      (kernel := sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_N] => activeLane s seq_lens N BLOCK_M BLOCK_N idx)
        (fun idx => (out_ptr, cOffset s b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx)))
      (expected := fun idx =>
        sgmvSpec s input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K numKBlocks idx.1 idx.2.1) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [sgmv_expand_slice_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  have hmain := sgmv_exec_closed_form input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
    lora_indices s0 N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
    cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks hBK hInj hundef idx
  have hExec2 : exec (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
      lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
      lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K) s0 = some s' := hExec
  rw [hExec2] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_real, hActive, if_true] using hmain

/-- **Public per-kernel output summary** (replaces the former slice/self-ref
proof-gap spec with a genuine contraction): the full SGMV expand-slice surface
(with the K-loop `tl.dot` accumulator and the masked store) lowers to the
algorithm layer **and** is compute-correct against the *genuine* rank-`K`
contraction `sgmvSpec` on every active output lane — under the
no-duplicate-destination hypothesis `hInj`. This is the closed-form GEMV
reference `Σ_{k<K} input·loraB`, not the kernel's own emitted value. -/
theorem sgmv_expand_slice_one_row_block_output_summary
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (cOffset s b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun idx : TileIndex [BLOCK_M, BLOCK_N] => activeLane s seq_lens N BLOCK_M BLOCK_N idx)
        (fun idx => (out_ptr, cOffset s b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx)))
      (expected := fun idx =>
        sgmvSpec s input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K numKBlocks idx.1 idx.2.1) :=
  ⟨sgmv_expand_slice_surface_toAlgorithm_supported input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
      lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K,
    sgmv_expand_slice_closed_form_correct input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
      lora_indices s N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks hBK hInj hundef⟩

end VeriTile.Bench.TritonBenchG.SgmvExpandSlice
