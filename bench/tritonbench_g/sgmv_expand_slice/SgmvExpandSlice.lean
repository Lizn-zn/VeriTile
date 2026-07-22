import VeriTile.Triton

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
sgmv_expand_slice_closed_form_correct            ← TOP THEOREM (ComputeCorrect.Realizes_without_Rounding)
  └─ sgmv_exec_closed_form                        ← exec-side closed form (every active cell = Σ_k a·b)
       ├─ sgmv_preLoop      (P 0: acc = 0, pointers/metadata seeded)
       ├─ sgmv_step         (one K-block: acc += dot advances the partial sum)
       ├─ sgmv_postLoop     (final masked store = the closed form on active lanes)
       └─ forRange_inv      (loop-invariant principle, drives the K-loop)
```

## Modeling boundary

Arithmetic is over `ℝ` (not bit-accurate IEEE float). The host launch (grid, the `(pid, cur_batch)` ↦
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
hypothesis `hInj` (distinct active lanes hit distinct addresses) is carried as an
open side condition all the way up to the top theorem (not discharged here).

## Translation-surface blocker

Translation-surface blocker: the Lean surface is the kernel's main numeric
path — `EVEN_K` (unmasked K-block) loads with `ADD_INPUTS = false` and
`CAST_TYPE = false` — so those three constexpr parameters and their branches
(the masked tail loads, the `tl.load(c)` residual add, and the dtype casts,
which collapse to the real carrier anyway) are dropped from the surface. The
in-kernel `pid → (pid_m, pid_n)` CTA decomposition (`cta_n_num =
tl.cdiv(N, BLOCK_N)`) is supplied via separate program-id axes — the
linearization is the trusted host boundary — and `tl.max_contiguous` /
`tl.multiple_of` layout hints are erased into the same `% M` / `% N` value
expression. The textual py↔lean scans in `bench/audit_tritonbench_g.sh`
exempt this port on this marker (registered in `proof_blockers.md`).
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
    ComputeCorrect.Realizes_without_Rounding
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
specification sgmv_expand_slice_one_row_block_output_summary
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (cOffset s b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N))
    (hundef : ∀ rg o, s.undef rg o = 0) :
    (∃ alg, (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes_without_Rounding
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

/-! ## The `⊨[R]` streaming metadata headline (wave-5 S1 fold genre)

Everything below is purely additive; the kernel surface and the exact stack
above are untouched. Structure of the `execR R` story: this kernel has
**zero rounding events** — the three metadata loads are `.nat`, the K-loop
loads and the `tl.dot` arithmetic are all at `.real`, and the terminal
masked store is a `.real` store (`stepStmtR` delegates a `.real` write to
the exact `writeMemTyped`; `RoundingModel.storeValue_real`). The prologue
and the whole K-loop are therefore cast-free and collapse verbatim onto the
exact stepper (`stepForRangeAuxR_castFree`), reusing the proven
`sgmv_preLoop` / `sgmv_step` / `forRange_inv` invariant stack unchanged;
only the masked terminal store is re-proved on the `R` side
(`sgmv_postLoopR`, via `BlockState.scatter_memcell_R_prop_masked_nd`). The
skin's readback contract then carries `R.round .real`, the identity by the
model's defining `round_real` — the `⊨[R]` face at the `.real` grid is the
exact streaming contract, stated once for every `R`. -/

open scoped VeriTile.Triton.StreamMetaMasked3DKernelIO₂

/-! ### Prologue decomposition (named segments) -/

/-- Statements 0–2: the three `tl.program_id` reads. -/
private def sgmvPidStmts : List Stmt :=
  [ Stmt.assign .nat [] "pid_m" (Op.programId 0),
    Stmt.assign .nat [] "pid_n" (Op.programId 1),
    Stmt.assign .nat [] "cur_batch" (Op.programId 2) ]

/-- Statements 3–5: the three metadata slot loads (`M` / `cur_seq_start` /
`lora_index`), each a single scalar cell at address `cur_batch`. -/
private def sgmvLoadStmts (b_seq_start_loc seq_lens lora_indices : Region .nat) : List Stmt :=
  [ Stmt.assign .nat [] "M" (Op.load .nat (.region seq_lens (Op.ref .nat [] "cur_batch")) .none),
    Stmt.assign .nat [] "cur_seq_start" (Op.load .nat (.region b_seq_start_loc (Op.ref .nat [] "cur_batch")) .none),
    Stmt.assign .nat [] "lora_index" (Op.load .nat (.region lora_indices (Op.ref .nat [] "cur_batch")) .none) ]

/-- Statements 6–10: the five index vectors (`offset_m`/`offset_n`/
`offset_k`/`ram`/`rbn`), all register-only. -/
private def sgmvIdxStmts (N BLOCK_M BLOCK_N BLOCK_K : Nat) : List Stmt :=
  [ Stmt.assign .nat [BLOCK_M] "offset_m"
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

/-- Statements 11–13: the two pointer tiles and the zero accumulator,
register-only. -/
private def sgmvPtrStmts (input_ptr lora_ptr : RegionName)
    (xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BLOCK_M, BLOCK_K] "a_ptr"
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
    Stmt.assign .real [BLOCK_M, BLOCK_N] "accumulator" (Op.full [BLOCK_M, BLOCK_N] (Op.const 0)) ]

/-- The 14-statement prologue as the concatenation of its four segments. -/
private def sgmvPrologue (input_ptr lora_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K : Nat) : List Stmt :=
  sgmvPidStmts ++ sgmvLoadStmts b_seq_start_loc seq_lens lora_indices
    ++ sgmvIdxStmts N BLOCK_M BLOCK_N BLOCK_K
    ++ sgmvPtrStmts input_ptr lora_ptr xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride BLOCK_M BLOCK_N BLOCK_K

/-- The store tail's four register-only offset/ptr/mask assigns. -/
private def sgmvTailAssigns (out_ptr : RegionName) (M BN Nb CM CN slice_offset : Nat) : List Stmt :=
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
        (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BN] "offset_n")) (Op.constNat Nb))) ]

/-- The terminal masked `.real` store statement. -/
private def sgmvStoreStmt (M BN : Nat) : Stmt :=
  Stmt.store .real [M, BN] (.ptr (Op.ref .ptr [M, BN] "c_ptr"))
    (Op.ref .real [M, BN] "accumulator") (.mask (Op.ref .bool [M, BN] "c_mask"))

private theorem sgmvStoreTail_eq (out_ptr : RegionName) (M BN Nb CM CN slice_offset : Nat) :
    sgmvStoreTail out_ptr M BN Nb CM CN slice_offset
      = sgmvTailAssigns out_ptr M BN Nb CM CN slice_offset ++ [sgmvStoreStmt M BN] := rfl

/-- `body.take 14` **is** the named prologue. -/
private theorem sgmv_take14_eq (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) :
    (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body.take 14
      = sgmvPrologue input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K := by
  simp only [sgmv_expand_slice_surface, ComputeKernel.toAlgKernel_mk,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Except.ok.injEq, List.take,
    sgmvPrologue, sgmvPidStmts, sgmvLoadStmts, sgmvIdxStmts, sgmvPtrStmts,
    List.cons_append, List.nil_append]
  rfl

/-- `sgmv_body_split` with the prologue named. -/
private theorem sgmv_body_split' (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) :
    (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgKernel.body
      = sgmvPrologue input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K
        ++ (Stmt.forRange "kk" 0 (BLOCK_K * numKBlocks) BLOCK_K
              (sgmvLoopBody BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K)
            :: sgmvStoreTail out_ptr BLOCK_M BLOCK_N N cm_stride cn_stride slice_offset) := by
  rw [sgmv_body_split input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
      N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks,
    sgmv_take14_eq input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
      N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks]

/-! ### Cast-free collapses (no `castFloat`, no narrow-float store anywhere
outside the terminal statement) -/

private theorem sgmvPidStmts_castFree (R : RoundingModel) (t : BlockState) :
    stepStmtsR R sgmvPidStmts t = stepStmts sgmvPidStmts t := by
  simp only [sgmvPidStmts, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 4000000 in
/-- The whole 14-statement prologue is cast-free (`.nat` loads and
register-only arithmetic step identically under `stepStmtsR R`). -/
private theorem sgmvPrologue_castFree (R : RoundingModel) (input_ptr lora_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K : Nat) (t : BlockState) :
    stepStmtsR R (sgmvPrologue input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K) t
      = stepStmts (sgmvPrologue input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K) t := by
  simp only [sgmvPrologue, sgmvPidStmts, sgmvLoadStmts, sgmvIdxStmts, sgmvPtrStmts,
    List.cons_append, List.nil_append, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

set_option maxHeartbeats 1000000 in
/-- The K-loop body is cast-free (`.real` loads, `tl.dot`+add, nat pointer
advances): it steps identically under `stepStmtsR R`. -/
private theorem sgmvBody_castFree (R : RoundingModel) (M N BK XK LN BKs : Nat)
    (t : BlockState) :
    stepStmtsR R (sgmvLoopBody M N BK XK LN BKs) t
      = stepStmts (sgmvLoopBody M N BK XK LN BKs) t := by
  simp only [sgmvLoopBody, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

private theorem sgmvTailAssigns_castFree (R : RoundingModel) (out_ptr : RegionName)
    (M BN Nb CM CN slice_offset : Nat) (t : BlockState) :
    stepStmtsR R (sgmvTailAssigns out_ptr M BN Nb CM CN slice_offset) t
      = stepStmts (sgmvTailAssigns out_ptr M BN Nb CM CN slice_offset) t := by
  simp only [sgmvTailAssigns, stepStmtsR, stepStmts, stepStmtR, stepStmt,
    evalOpR.eq_def, evalOp.eq_def]
  rfl

/-! ### `.real` R-write plumbing (copied shape from `matmul_triton1`) -/

/-- A `.real`-typed rounded write **is** the `writeMemAsR R .real` write
(`RoundingModel.storeValue_real`): lets the masked `.real` terminal store
reuse the `writeMemAsR` scatter readback/frame lemma family. -/
private theorem writeMemTypedR_real_eq (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier TileDType.real) :
    s.writeMemTypedR R .real region offset v
      = s.writeMemAsR R .real region offset v := by
  show s.writeMemTyped .real region offset v = _
  simp only [BlockState.writeMemTyped, BlockState.writeMemAs, BlockState.writeMemAsR,
    RoundingModel.storeValue_real]

/-- Tag-exact readback of a stored `.real` cell through `readMemAs .real`. -/
private theorem readMemAs_real_of_cell {s : BlockState} {region : RegionName}
    {offset : Nat} {x : ℝ}
    (h : s.mem region offset
      = MemCell.of FloatDType.real.toTileDType (FloatDType.real.ofReal x)) :
    s.readMemAs .real region offset = FloatDType.real.ofReal x := by
  simp [BlockState.readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

/-! ### The store tail's assign block, once -/

/-- The `c_ptr` tile the tail assigns compute. -/
private def sgmvCPtrT (out_ptr : RegionName) (b_seq_start_loc : Region .nat)
    (s0 : BlockState) (cm_stride cn_stride slice_offset BLOCK_M BLOCK_N : Nat) :
    Tile .ptr [BLOCK_M, BLOCK_N] :=
  ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
    (out_ptr.cast, (seqStart s0 b_seq_start_loc + rowG s0 BLOCK_M idx.1) * cm_stride
      + (colG s0 BLOCK_N idx.2.1 + slice_offset) * cn_stride)⟩

/-- The `c_mask` tile the tail assigns compute. -/
private def sgmvCMaskT (seq_lens : Region .nat) (s0 : BlockState)
    (N BLOCK_M BLOCK_N : Nat) : Tile .bool [BLOCK_M, BLOCK_N] :=
  ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
    decide (rowG s0 BLOCK_M idx.1 < seqLen s0 seq_lens)
      && decide (colG s0 BLOCK_N idx.2.1 < N)⟩

/-- The state after the four tail assigns. -/
private def sgmvTailState (out_ptr : RegionName) (b_seq_start_loc seq_lens : Region .nat)
    (s0 st : BlockState) (N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N : Nat) :
    BlockState :=
  (((st.setReg "offset_cm" .nat [BLOCK_M]
        (Tile.vec (fun i : Fin BLOCK_M => seqStart s0 b_seq_start_loc + rowG s0 BLOCK_M i))).setReg
      "offset_cn" .nat [BLOCK_N]
        (Tile.vec (fun j : Fin BLOCK_N => colG s0 BLOCK_N j + slice_offset))).setReg
      "c_ptr" .ptr [BLOCK_M, BLOCK_N]
        (sgmvCPtrT out_ptr b_seq_start_loc s0 cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)).setReg
      "c_mask" .bool [BLOCK_M, BLOCK_N] (sgmvCMaskT seq_lens s0 N BLOCK_M BLOCK_N)

set_option maxHeartbeats 2000000 in
/-- Running the four tail assigns (exact stepper) computes `sgmvTailState`. -/
private theorem sgmvTailAssigns_run (out_ptr : RegionName)
    (b_seq_start_loc seq_lens : Region .nat) (s0 st : BlockState)
    (N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N : Nat)
    (hom : st.regs .nat [BLOCK_M] "offset_m" = some (Tile.vec (fun r : Fin BLOCK_M => rowG s0 BLOCK_M r)))
    (hon : st.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => colG s0 BLOCK_N j)))
    (hcss : st.regs .nat [] "cur_seq_start" = some (Tile.scalar (seqStart s0 b_seq_start_loc)))
    (hM : st.regs .nat [] "M" = some (Tile.scalar (seqLen s0 seq_lens))) :
    stepStmts (sgmvTailAssigns out_ptr BLOCK_M BLOCK_N N cm_stride cn_stride slice_offset) st
      = some (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
          N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N) := by
  unfold sgmvTailAssigns
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (offcm_eval st BLOCK_M (seqStart s0 b_seq_start_loc) (fun r => rowG s0 BLOCK_M r) hcss hom))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (offcn_eval _ BLOCK_N slice_offset (fun j => colG s0 BLOCK_N j) (by simp [hon])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (cptr_eval _ out_ptr BLOCK_M BLOCK_N cm_stride cn_stride
          (fun i => seqStart s0 b_seq_start_loc + rowG s0 BLOCK_M i)
          (fun j => colG s0 BLOCK_N j + slice_offset) (by simp) (by simp)))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (show evalOp (Op.boolAnd Broadcast.nil.consL.consR
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref .nat [BLOCK_M] "offset_m")) (Op.ref .nat [] "M"))
            (Op.lt .nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref .nat [BLOCK_N] "offset_n")) (Op.constNat N))) _
          = some (sgmvCMaskT seq_lens s0 N BLOCK_M BLOCK_N) from by
          simp only [evalOp, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat,
            BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
            not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
          simp only [hom, hon, hM, Option.bind_some, bind, Option.bind_eq_bind]
          refine congrArg some ?_
          ext idx
          simp only [sgmvCMaskT, Tile.bop, Tile.cop, Tile.vec, Broadcast.leftIndex,
            Broadcast.rightIndex, ComparableDType.lt, seqLen, rowG, colG]
          rfl))]
  rw [stepStmts.nil]
  rfl

set_option maxHeartbeats 2000000 in
/-- **R-postLoop**: from the exact invariant at `numKBlocks` blocks, the
`execR R` store tail terminates; every *active* output lane holds the exact-ℝ
cell `MemCell.of .real (sgmvSpec …)` — the masked `.real` store has no
rounding event — with a per-cell frame outside the active output window. -/
private theorem sgmv_postLoopR (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s0 : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hBK : 0 < BLOCK_K)
    (hInj : Function.Injective (cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N))
    (st : BlockState)
    (hinv : sgmvInvariant input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s0
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K numKBlocks (BLOCK_K * numKBlocks) st) :
    ∃ sfin, stepStmtsR R (sgmvStoreTail out_ptr BLOCK_M BLOCK_N N cm_stride cn_stride slice_offset) st
        = some sfin
      ∧ (∀ idx : TileIndex [BLOCK_M, BLOCK_N],
          activeLane s0 seq_lens N BLOCK_M BLOCK_N idx →
          sfin.mem out_ptr (cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx)
            = MemCell.of FloatDType.real.toTileDType
                (FloatDType.real.ofReal
                  (sgmvSpec s0 input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
                    N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
                    BLOCK_M BLOCK_N BLOCK_K numKBlocks idx.1 idx.2.1)))
      ∧ (∀ r o, (r ≠ out_ptr ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
            activeLane s0 seq_lens N BLOCK_M BLOCK_N idx →
            o ≠ cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx) →
          sfin.mem r o = st.mem r o) := by
  have hcN : (BLOCK_K * numKBlocks) / BLOCK_K = numKBlocks := by
    rw [Nat.mul_comm, Nat.mul_div_cancel _ hBK]
  simp only [sgmvInvariant, hcN] at hinv
  obtain ⟨hpids, hi, hcle, hacc, hom, hon, hok, hram, hrbn, hcss, hli, hM, hap, hbp, hundef, hmem⟩ := hinv
  set accT : Tile .real [BLOCK_M, BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
      some (accPartial s0 input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K idx.1 idx.2.1 numKBlocks)⟩ with haccT
  have hacc4 : (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
      N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N).regs .real [BLOCK_M, BLOCK_N] "accumulator"
      = some accT := by
    simp [sgmvTailState, hacc, haccT]
  have hcp4 : (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
      N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N).regs .ptr [BLOCK_M, BLOCK_N] "c_ptr"
      = some (sgmvCPtrT out_ptr b_seq_start_loc s0 cm_stride cn_stride slice_offset BLOCK_M BLOCK_N) := by
    simp [sgmvTailState]
  have hcm4 : (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
      N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N).regs .bool [BLOCK_M, BLOCK_N] "c_mask"
      = some (sgmvCMaskT seq_lens s0 N BLOCK_M BLOCK_N) := by
    simp [sgmvTailState]
  have hstore : stepStmtR R (sgmvStoreStmt BLOCK_M BLOCK_N)
      (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
        N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)
      = some ((TileShape.allIndices [BLOCK_M, BLOCK_N]).foldl
          (fun acc idx =>
            if activeLane s0 seq_lens N BLOCK_M BLOCK_N idx then
              acc.writeMemAsR R .real out_ptr
                (cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx)
                (accT.data idx)
            else acc)
          (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
            N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)) := by
    simp only [sgmvStoreStmt, stepStmtR]
    rw [show evalOpR R (Op.ref .real [BLOCK_M, BLOCK_N] "accumulator")
          (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
            N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N) = some accT from by
        rw [evalOpR_ref]; exact hacc4]
    rw [show evalOpR R (Op.ref .bool [BLOCK_M, BLOCK_N] "c_mask")
          (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
            N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)
          = some (sgmvCMaskT seq_lens s0 N BLOCK_M BLOCK_N) from by
        rw [evalOpR_ref]; exact hcm4]
    rw [show evalOpR R (Op.ref .ptr [BLOCK_M, BLOCK_N] "c_ptr")
          (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
            N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)
          = some (sgmvCPtrT out_ptr b_seq_start_loc s0 cm_stride cn_stride slice_offset BLOCK_M BLOCK_N) from by
        rw [evalOpR_ref]; exact hcp4]
    simp only [Option.map_some, bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ _ (fun acc idx _ => ?_))
    simp only [sgmvCMaskT, sgmvCPtrT, Bool.and_eq_true, decide_eq_true_eq, activeLane,
      cOffset, seqLen, Region.cast_id, writeMemTypedR_real_eq]
  rw [sgmvStoreTail_eq, stepStmtsR_append,
    sgmvTailAssigns_castFree R out_ptr BLOCK_M BLOCK_N N cm_stride cn_stride slice_offset st,
    sgmvTailAssigns_run out_ptr b_seq_start_loc seq_lens s0 st
      N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N hom hon hcss hM,
    Option.bind_some, stepStmtsR_cons_some hstore, stepStmtsR_nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx hact
    rw [BlockState.scatter_memcell_R_prop_masked_nd R .real (region := out_ptr)
        (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
          N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)
        (cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)
        (fun idx => accT.data idx)
        (fun idx => activeLane s0 seq_lens N BLOCK_M BLOCK_N idx)
        hInj idx]
    rw [if_pos hact]
    simp only [haccT, RoundingModel.storeValue_real, FloatDType.real_storeValue,
      WithBot.unbotD_some, sgmvSpec, accPartial, Nat.mul_comm numKBlocks BLOCK_K]
  · intro r o hcond
    by_cases hr : r = out_ptr
    · rcases hcond with hne | hno
      · exact absurd hr hne
      · rw [hr]
        rw [BlockState.foldl_writeMemAsR_preserve_masked_prop R .real
            (cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)
            (fun idx => accT.data idx)
            (fun idx => activeLane s0 seq_lens N BLOCK_M BLOCK_N idx) o
            (TileShape.allIndices [BLOCK_M, BLOCK_N])
            (fun k _ hk he => hno k hk he.symm)
            (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
              N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)]
        rfl
    · rw [BlockState.foldl_writeMemAsR_preserve_other_region R .real
          (cOffset s0 b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)
          (fun idx => accT.data idx)
          (fun idx => activeLane s0 seq_lens N BLOCK_M BLOCK_N idx) r hr o
          (TileShape.allIndices [BLOCK_M, BLOCK_N])
          (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
            N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)]
      rfl

/-! ### Safety-walk invariant (weak shape half of `sgmvInvariant`)

Needed because the `⊨[R]` skin's `hts` obligation quantifies over arbitrary
launch states, so the safety walk cannot assume the clean-`undef`
precondition that `sgmv_preLoop`'s full invariant needs. -/

/-- Safety-walk loop invariant: the *shape* half of `sgmvInvariant` (counter
`i = c · BLOCK_K`; *some* accumulator tile, the tail's four seed registers,
and the exact `a_ptr` / `b_ptr` address shapes) with no `undef` / `mem` /
accumulator-value pins. -/
private def sgmvSafeInv (input_ptr lora_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s0 : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (i : Nat) (s : BlockState) : Prop :=
  let c := i / BLOCK_K
  i = c * BLOCK_K ∧ c ≤ numKBlocks ∧
  (∃ accT : Tile .real [BLOCK_M, BLOCK_N],
    s.regs .real [BLOCK_M, BLOCK_N] "accumulator" = some accT) ∧
  (s.regs .nat [BLOCK_M] "offset_m" = some (Tile.vec (fun r : Fin BLOCK_M => rowG s0 BLOCK_M r))) ∧
  (s.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => colG s0 BLOCK_N j))) ∧
  (s.regs .nat [] "cur_seq_start" = some (Tile.scalar (seqStart s0 b_seq_start_loc))) ∧
  (s.regs .nat [] "M" = some (Tile.scalar (seqLen s0 seq_lens))) ∧
  (s.regs .ptr [BLOCK_M, BLOCK_K] "a_ptr" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_K] =>
      (input_ptr.cast, seqStart s0 b_seq_start_loc * xm_stride
        + ramRow s0 seq_lens BLOCK_M idx.1 * xm_stride + (idx.2.1.val + c * BLOCK_K) * xk_stride)⟩) ∧
  (s.regs .ptr [BLOCK_K, BLOCK_N] "b_ptr" = some ⟨fun idx : TileIndex [BLOCK_K, BLOCK_N] =>
      (lora_ptr.cast, l0_stride * loraIdx s0 lora_indices
        + (idx.1.val + c * BLOCK_K) * lora_n_stride + rbnCol s0 N BLOCK_N idx.2.1 * lora_k_stride)⟩)

set_option maxHeartbeats 2000000 in
/-- Weak `preLoop`: from an **arbitrary** state the prologue steps to a state
satisfying `sgmvSafeInv … 0` (no clean-`undef` hypothesis; the value half of
`sgmv_preLoop` is dropped). -/
private theorem sgmv_preLoopW (input_ptr lora_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) :
    ∃ s', stepStmts (sgmvPrologue input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K) s = some s'
      ∧ sgmvSafeInv input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K numKBlocks 0 s' := by
  obtain ⟨s11, h11, hpids, hcss, hli, hM, hom, hon, hok, hram, hrbn, huf, hmem⟩ :=
    preLoop_scalars s b_seq_start_loc seq_lens lora_indices N BLOCK_M BLOCK_N BLOCK_K
  rw [show sgmvPrologue input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K
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
      ++ sgmvPtrStmts input_ptr lora_ptr xm_stride xk_stride l0_stride lora_k_stride
          lora_n_stride BLOCK_M BLOCK_N BLOCK_K from rfl,
    stepStmts.append_some h11]
  unfold sgmvPtrStmts
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
      (aptr_eval s11 input_ptr BLOCK_M BLOCK_K xm_stride xk_stride (seqStart s b_seq_start_loc)
        (fun r => ramRow s seq_lens BLOCK_M r) (by simpa using hram) (by simpa using hcss) (by simpa using hok))),
    stepStmts.cons_some (stepStmt_assign_eq_some
      (bptr_eval _ lora_ptr BLOCK_K BLOCK_N l0_stride lora_k_stride lora_n_stride (loraIdx s lora_indices)
        (fun j : Fin BLOCK_N => rbnCol s N BLOCK_N j) (by simp [hli]) (by simp [hok]) (by simp [hrbn]))),
    stepStmts.cons_some (stepStmt_assign_eq_some (acc_init_eval _ BLOCK_M BLOCK_N)),
    stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  simp only [sgmvSafeInv, Nat.zero_div]
  refine ⟨by simp, by simp, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨⟨fun _ : TileIndex [BLOCK_M, BLOCK_N] => some (0 : ℝ)⟩, ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp [hom]
  · simp [hon]
  · simp [hcss]
  · simp [hM]
  · -- a_ptr (c = 0)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_; ext idx <;> simp [Nat.zero_mul]
  · -- b_ptr (c = 0)
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
    refine congrArg some ?_; ext idx <;> simp [Nat.zero_mul]

set_option maxHeartbeats 2000000 in
/-- Weak step lemma: one body iteration from `sgmvSafeInv i` steps
successfully (exact stepper; the body is cast-free) and re-establishes
`sgmvSafeInv (i + BLOCK_K)` — the shape half of `sgmv_step`, valid from
arbitrary launch states. -/
private theorem sgmv_stepW (input_ptr lora_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s0 : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) (hBK : 0 < BLOCK_K)
    (i : Nat) (s : BlockState) (hilt : i < BLOCK_K * numKBlocks)
    (hinv : sgmvSafeInv input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s0
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K numKBlocks i s) :
    ∃ s', stepStmts (sgmvLoopBody BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K)
        (s.setReg "kk" .nat [] (Tile.scalar i)) = some s'
      ∧ sgmvSafeInv input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s0
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K numKBlocks (i + BLOCK_K) s' := by
  have hc : i / BLOCK_K < numKBlocks := (Nat.div_lt_iff_lt_mul hBK).mpr (by rw [Nat.mul_comm]; exact hilt)
  have hc1 : (i + BLOCK_K) / BLOCK_K = i / BLOCK_K + 1 := Nat.add_div_right i hBK
  simp only [sgmvSafeInv] at hinv
  obtain ⟨hi, hcle, ⟨accT, hacc⟩, hom, hon, hcss, hM, hap, hbp⟩ := hinv
  set c := i / BLOCK_K with hcdef
  set apT : Tile .ptr [BLOCK_M, BLOCK_K] :=
    ⟨fun idx : TileIndex [BLOCK_M, BLOCK_K] =>
      (input_ptr.cast, seqStart s0 b_seq_start_loc * xm_stride
        + ramRow s0 seq_lens BLOCK_M idx.1 * xm_stride + (idx.2.1.val + c * BLOCK_K) * xk_stride)⟩ with hapT
  set bpT : Tile .ptr [BLOCK_K, BLOCK_N] :=
    ⟨fun idx : TileIndex [BLOCK_K, BLOCK_N] =>
      (lora_ptr.cast, l0_stride * loraIdx s0 lora_indices
        + (idx.1.val + c * BLOCK_K) * lora_n_stride + rbnCol s0 N BLOCK_N idx.2.1 * lora_k_stride)⟩ with hbpT
  set sk := s.setReg "kk" .nat [] (Tile.scalar i) with hsk
  have hapk : sk.regs .ptr [BLOCK_M, BLOCK_K] "a_ptr" = some apT := by simp [hsk, hap, hapT]
  have hbpk : sk.regs .ptr [BLOCK_K, BLOCK_N] "b_ptr" = some bpT := by simp [hsk, hbp, hbpT]
  have hacck : sk.regs .real [BLOCK_M, BLOCK_N] "accumulator" = some accT := by simp [hsk, hacc]
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
  simp only [sgmvSafeInv, hc1]
  refine ⟨by rw [hi]; ring, by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · refine ⟨Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      accT (Tile.dot [] asub bsub), ?_⟩
    simp only [BlockState.setReg_same, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, IsEmpty.forall_iff]
  · simp [hsk, hom]
  · simp [hsk, hon]
  · simp [hsk, hcss]
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

/-! ### The `TraceSafeR` walk -/

/-- Safety of one scalar metadata load (`tl.load(buf + cur_batch)`): the only
active address is the pinned `cur_batch` register value, in bounds by the
skin's slot window. -/
private theorem sgmv_load_safeR (R : RoundingModel) (bounds : RegionBounds)
    (buf : Region .nat) (name : RegName) (t : BlockState) (a : Nat)
    (hcur : t.regs .nat [] "cur_batch" = some (Tile.scalar a))
    (hb : a < bounds buf.cast) :
    Stmt.TraceSafeR R bounds
      (Stmt.assign .nat [] name (Op.load .nat (.region buf (Op.ref .nat [] "cur_batch")) .none)) t := by
  simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, ?_⟩
  intro offs hoffs i _
  rw [evalOpR_ref, hcur] at hoffs
  obtain rfl := Option.some.inj hoffs
  exact hb

set_option maxHeartbeats 1000000 in
/-- Per-iteration `TraceSafeListR` for the K-loop body: the two unmasked
loads' addresses are the invariant's pointer shapes, in bounds by the skin's
`read1`/`read2` windows (instantiated at block `c`); the three register-only
assigns are unconditionally safe. -/
private theorem sgmv_bodySafeR (R : RoundingModel) (bounds : RegionBounds)
    (input_ptr lora_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s0 : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (c : Nat) (hc : c < numKBlocks) (sk : BlockState)
    (hap : sk.regs .ptr [BLOCK_M, BLOCK_K] "a_ptr" = some ⟨fun idx : TileIndex [BLOCK_M, BLOCK_K] =>
        (input_ptr.cast, seqStart s0 b_seq_start_loc * xm_stride
          + ramRow s0 seq_lens BLOCK_M idx.1 * xm_stride + (idx.2.1.val + c * BLOCK_K) * xk_stride)⟩)
    (hbp : sk.regs .ptr [BLOCK_K, BLOCK_N] "b_ptr" = some ⟨fun idx : TileIndex [BLOCK_K, BLOCK_N] =>
        (lora_ptr.cast, l0_stride * loraIdx s0 lora_indices
          + (idx.1.val + c * BLOCK_K) * lora_n_stride + rbnCol s0 N BLOCK_N idx.2.1 * lora_k_stride)⟩)
    (hbA : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_M * BLOCK_K)),
      seqStart s0 b_seq_start_loc * xm_stride
        + ((s0.pids 0 * BLOCK_M + j.val / BLOCK_K) % seqLen s0 seq_lens) * xm_stride
        + (t.val * BLOCK_K + j.val % BLOCK_K) * xk_stride < bounds input_ptr)
    (hbB : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_K * BLOCK_N)),
      l0_stride * loraIdx s0 lora_indices
        + (t.val * BLOCK_K + j.val / BLOCK_N) * lora_n_stride
        + ((s0.pids 1 * BLOCK_N + j.val % BLOCK_N) % N) * lora_k_stride < bounds lora_ptr) :
    Stmt.TraceSafeListR R bounds
      (sgmvLoopBody BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K) sk := by
  unfold sgmvLoopBody
  refine Stmt.TraceSafeListR.cons_intro ?_ ?_
  · -- load tiled_a
    simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, ?_⟩
    intro ptrs hptrs i _
    rw [evalOpR_ref, hap] at hptrs
    obtain rfl := Option.some.inj hptrs
    show seqStart s0 b_seq_start_loc * xm_stride + ramRow s0 seq_lens BLOCK_M i.1 * xm_stride
        + (i.2.1.val + c * BLOCK_K) * xk_stride < bounds (Region.cast input_ptr)
    have h' : seqStart s0 b_seq_start_loc * xm_stride
        + ((s0.pids 0 * BLOCK_M + i.1.val) % seqLen s0 seq_lens) * xm_stride
        + (c * BLOCK_K + i.2.1.val) * xk_stride < bounds input_ptr := by
      simpa using hbA ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
    simp only [Region.cast_id]
    calc seqStart s0 b_seq_start_loc * xm_stride + ramRow s0 seq_lens BLOCK_M i.1 * xm_stride
          + (i.2.1.val + c * BLOCK_K) * xk_stride
        = seqStart s0 b_seq_start_loc * xm_stride
            + ((s0.pids 0 * BLOCK_M + i.1.val) % seqLen s0 seq_lens) * xm_stride
            + (c * BLOCK_K + i.2.1.val) * xk_stride := by
          unfold ramRow rowG; ring
      _ < bounds input_ptr := h'
  · intro s1 h1
    obtain ⟨v1, -, rfl⟩ := stepStmtR_assign_inv h1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · -- load tiled_b
      simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro ptrs hptrs i _
      rw [evalOpR_ref] at hptrs
      rw [show (sk.setReg "tiled_a" .real [BLOCK_M, BLOCK_K] v1).regs .ptr [BLOCK_K, BLOCK_N] "b_ptr"
          = some (⟨fun idx : TileIndex [BLOCK_K, BLOCK_N] =>
              (lora_ptr.cast, l0_stride * loraIdx s0 lora_indices
                + (idx.1.val + c * BLOCK_K) * lora_n_stride
                + rbnCol s0 N BLOCK_N idx.2.1 * lora_k_stride)⟩ : Tile .ptr [BLOCK_K, BLOCK_N])
          from by simp [hbp]] at hptrs
      obtain rfl := Option.some.inj hptrs
      show l0_stride * loraIdx s0 lora_indices + (i.1.val + c * BLOCK_K) * lora_n_stride
          + rbnCol s0 N BLOCK_N i.2.1 * lora_k_stride < bounds (Region.cast lora_ptr)
      have h' : l0_stride * loraIdx s0 lora_indices + (c * BLOCK_K + i.1.val) * lora_n_stride
          + ((s0.pids 1 * BLOCK_N + i.2.1.val) % N) * lora_k_stride < bounds lora_ptr := by
        simpa using hbB ⟨c, hc⟩ (Lane2D.encode (i.1, i.2.1, PUnit.unit))
      simp only [Region.cast_id]
      calc l0_stride * loraIdx s0 lora_indices + (i.1.val + c * BLOCK_K) * lora_n_stride
            + rbnCol s0 N BLOCK_N i.2.1 * lora_k_stride
          = l0_stride * loraIdx s0 lora_indices + (c * BLOCK_K + i.1.val) * lora_n_stride
              + ((s0.pids 1 * BLOCK_N + i.2.1.val) % N) * lora_k_stride := by
            unfold rbnCol colG; ring
        _ < bounds lora_ptr := h'
    · intro s2 h2
      obtain ⟨v2, -, rfl⟩ := stepStmtR_assign_inv h2
      refine Stmt.TraceSafeListR.of_forall _ _ ?_
      intro st hst s'
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
      rcases hst with rfl | rfl | rfl <;>
        simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]

set_option maxHeartbeats 1000000 in
/-- `TraceSafeListR` for the store tail: the four assigns are register-only;
the masked `.real` store's *active* lanes (`c_mask`) hit exactly the skin's
write window, in bounds by `hbC`. -/
private theorem sgmv_tailSafeR (R : RoundingModel) (bounds : RegionBounds)
    (out_ptr : RegionName) (b_seq_start_loc seq_lens : Region .nat) (s0 : BlockState)
    (N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N : Nat) (st : BlockState)
    (hom : st.regs .nat [BLOCK_M] "offset_m" = some (Tile.vec (fun r : Fin BLOCK_M => rowG s0 BLOCK_M r)))
    (hon : st.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun j : Fin BLOCK_N => colG s0 BLOCK_N j)))
    (hcss : st.regs .nat [] "cur_seq_start" = some (Tile.scalar (seqStart s0 b_seq_start_loc)))
    (hM : st.regs .nat [] "M" = some (Tile.scalar (seqLen s0 seq_lens)))
    (hbC : ∀ j : Fin (BLOCK_M * BLOCK_N),
      s0.pids 0 * BLOCK_M + j.val / BLOCK_N < seqLen s0 seq_lens →
      s0.pids 1 * BLOCK_N + j.val % BLOCK_N < N →
      (seqStart s0 b_seq_start_loc + (s0.pids 0 * BLOCK_M + j.val / BLOCK_N)) * cm_stride
        + ((s0.pids 1 * BLOCK_N + j.val % BLOCK_N) + slice_offset) * cn_stride < bounds out_ptr) :
    Stmt.TraceSafeListR R bounds
      (sgmvStoreTail out_ptr BLOCK_M BLOCK_N N cm_stride cn_stride slice_offset) st := by
  rw [sgmvStoreTail_eq]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hstmt s'
    simp only [sgmvTailAssigns, List.mem_cons, List.not_mem_nil, or_false] at hstmt
    rcases hstmt with rfl | rfl | rfl | rfl <;>
      simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · intro s4 hs4
    rw [sgmvTailAssigns_castFree R out_ptr BLOCK_M BLOCK_N N cm_stride cn_stride slice_offset st,
      sgmvTailAssigns_run out_ptr b_seq_start_loc seq_lens s0 st
        N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N hom hon hcss hM] at hs4
    obtain rfl := Option.some.inj hs4
    refine Stmt.TraceSafeListR.cons_intro ?_ (fun s' _ => Stmt.TraceSafeListR.nil_intro)
    simp only [sgmvStoreStmt, Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR,
      Op.SafeAtR.eq_def, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
      memAccessActiveAddressSafeR]
    refine ⟨trivial, trivial, trivial, ?_⟩
    intro ptrs hptrs i hact
    rw [evalOpR_ref] at hptrs
    rw [show (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
          N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N).regs .ptr [BLOCK_M, BLOCK_N] "c_ptr"
        = some (sgmvCPtrT out_ptr b_seq_start_loc s0 cm_stride cn_stride slice_offset BLOCK_M BLOCK_N)
        from by simp [sgmvTailState]] at hptrs
    obtain rfl := Option.some.inj hptrs
    obtain ⟨masks, hmasks, hmi⟩ := hact
    rw [evalOpR_ref] at hmasks
    rw [show (sgmvTailState out_ptr b_seq_start_loc seq_lens s0 st
          N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N).regs .bool [BLOCK_M, BLOCK_N] "c_mask"
        = some (sgmvCMaskT seq_lens s0 N BLOCK_M BLOCK_N)
        from by simp [sgmvTailState]] at hmasks
    obtain rfl := Option.some.inj hmasks
    have hmi' : rowG s0 BLOCK_M i.1 < seqLen s0 seq_lens ∧ colG s0 BLOCK_N i.2.1 < N := by
      have := hmi
      simp only [sgmvCMaskT, Bool.and_eq_true, decide_eq_true_eq] at this
      exact this
    show (seqStart s0 b_seq_start_loc + rowG s0 BLOCK_M i.1) * cm_stride
        + (colG s0 BLOCK_N i.2.1 + slice_offset) * cn_stride < bounds (Region.cast out_ptr)
    have h' := hbC (Lane2D.encode (i.1, i.2.1, PUnit.unit))
      (by rw [Lane2D.encode_div]; exact hmi'.1)
      (by rw [Lane2D.encode_mod]; exact hmi'.2)
    rw [Lane2D.encode_div, Lane2D.encode_mod] at h'
    simp only [Region.cast_id]
    exact h'

set_option maxHeartbeats 2000000 in
/-- **The `TraceSafeR` walk for the whole kernel** — driven by
`Stmt.forRangeTraceSafeR_inv` over the weak `sgmvSafeInv`, with the counter
advancing by the loop's stride `BLOCK_K` (not 1). The three scalar bounds
are the skin's slot windows; the three tile bound groups are the skin's
`read1`/`read2`/`write` windows (`write` guarded by the `c_mask` lanes). -/
private theorem sgmv_traceSafeR (R : RoundingModel) (bounds : RegionBounds)
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hBK : 0 < BLOCK_K) (s : BlockState)
    (hbSL : s.pids 2 < bounds seq_lens.cast)
    (hbBS : s.pids 2 < bounds b_seq_start_loc.cast)
    (hbLI : s.pids 2 < bounds lora_indices.cast)
    (hbA : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_M * BLOCK_K)),
      seqStart s b_seq_start_loc * xm_stride
        + ((s.pids 0 * BLOCK_M + j.val / BLOCK_K) % seqLen s seq_lens) * xm_stride
        + (t.val * BLOCK_K + j.val % BLOCK_K) * xk_stride < bounds input_ptr)
    (hbB : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_K * BLOCK_N)),
      l0_stride * loraIdx s lora_indices
        + (t.val * BLOCK_K + j.val / BLOCK_N) * lora_n_stride
        + ((s.pids 1 * BLOCK_N + j.val % BLOCK_N) % N) * lora_k_stride < bounds lora_ptr)
    (hbC : ∀ j : Fin (BLOCK_M * BLOCK_N),
      s.pids 0 * BLOCK_M + j.val / BLOCK_N < seqLen s seq_lens →
      s.pids 1 * BLOCK_N + j.val % BLOCK_N < N →
      (seqStart s b_seq_start_loc + (s.pids 0 * BLOCK_M + j.val / BLOCK_N)) * cm_stride
        + ((s.pids 1 * BLOCK_N + j.val % BLOCK_N) + slice_offset) * cn_stride < bounds out_ptr) :
    ((sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgKernel).TraceSafeR
      R bounds s := by
  unfold Kernel.TraceSafeR
  rw [sgmv_body_split' input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
    slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks]
  have hstep : ∀ i s', i < BLOCK_K * numKBlocks →
      sgmvSafeInv input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K numKBlocks i s' →
      Stmt.TraceSafeListR R bounds (sgmvLoopBody BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K)
        (s'.setReg "kk" .nat [] (Tile.scalar i)) ∧
      ∃ s'', stepStmtsR R (sgmvLoopBody BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K)
          (s'.setReg "kk" .nat [] (Tile.scalar i)) = some s'' ∧
        sgmvSafeInv input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K numKBlocks (i + BLOCK_K) s'' := by
    intro i s' hi hP
    have hPd := hP
    simp only [sgmvSafeInv] at hPd
    obtain ⟨hieq, hcle, haccE, homI, honI, hcssI, hMI, hapI, hbpI⟩ := hPd
    have hcT : i / BLOCK_K < numKBlocks :=
      (Nat.div_lt_iff_lt_mul hBK).mpr (by rw [Nat.mul_comm]; exact hi)
    refine ⟨sgmv_bodySafeR R bounds input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K numKBlocks (i / BLOCK_K) hcT _
        (by simp [hapI]) (by simp [hbpI]) hbA hbB, ?_⟩
    obtain ⟨s'', hs'', hP''⟩ :=
      sgmv_stepW input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K numKBlocks hBK i s' hi hP
    exact ⟨s'', by rw [sgmvBody_castFree]; exact hs'', hP''⟩
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- prologue trace safety, segment by segment
    refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
    · refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
      · refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
        · -- pid reads
          refine Stmt.TraceSafeListR.of_forall _ _ ?_
          intro st hst s'
          simp only [sgmvPidStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
          rcases hst with rfl | rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
        · -- the three metadata loads, from the computed pid state
          intro s1 hs1
          rw [sgmvPidStmts_castFree R s,
            show stepStmts sgmvPidStmts s
              = some (((s.setReg "pid_m" .nat [] (Tile.scalar (s.pids 0))).setReg
                  "pid_n" .nat [] (Tile.scalar (s.pids 1))).setReg
                  "cur_batch" .nat [] (Tile.scalar (s.pids 2))) from by
              simp [sgmvPidStmts, stepStmts, stepStmt, bind]] at hs1
          obtain rfl := Option.some.inj hs1
          refine Stmt.TraceSafeListR.cons_intro
            (sgmv_load_safeR R bounds seq_lens "M" _ (s.pids 2) (by simp) hbSL) ?_
          intro s2 hs2
          obtain ⟨vM, -, rfl⟩ := stepStmtR_assign_inv hs2
          refine Stmt.TraceSafeListR.cons_intro
            (sgmv_load_safeR R bounds b_seq_start_loc "cur_seq_start" _ (s.pids 2) (by simp) hbBS) ?_
          intro s3 hs3
          obtain ⟨vCS, -, rfl⟩ := stepStmtR_assign_inv hs3
          exact Stmt.TraceSafeListR.cons_intro
            (sgmv_load_safeR R bounds lora_indices "lora_index" _ (s.pids 2) (by simp) hbLI)
            (fun s4 _ => Stmt.TraceSafeListR.nil_intro)
      · -- index vectors: register-only
        intro s2 _
        refine Stmt.TraceSafeListR.of_forall _ _ ?_
        intro st hst s'
        simp only [sgmvIdxStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
        rcases hst with rfl | rfl | rfl | rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
    · -- pointer seeds: register-only
      intro s3 _
      refine Stmt.TraceSafeListR.of_forall _ _ ?_
      intro st hst s'
      simp only [sgmvPtrStmts, List.mem_cons, List.not_mem_nil, or_false] at hst
      rcases hst with rfl | rfl | rfl <;> simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  · -- after the prologue: the K-loop, then the store tail
    intro s1 hs1
    obtain ⟨s1x, hpre, hP0⟩ :=
      sgmv_preLoopW input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K numKBlocks
    rw [sgmvPrologue_castFree R input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K s,
      hpre] at hs1
    obtain rfl := Option.some.inj hs1
    refine Stmt.TraceSafeListR.cons_intro ?_ ?_
    · simp only [Stmt.TraceSafeR]
      exact Stmt.forRangeTraceSafeR_inv R bounds "kk" (BLOCK_K * numKBlocks) BLOCK_K
        (sgmvLoopBody BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K)
        (sgmvSafeInv input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
          BLOCK_M BLOCK_N BLOCK_K numKBlocks)
        hstep 0 s1x hP0
    · intro s2 hs2
      obtain ⟨final, sLoop, hLoopStmt, hfinal, hPL⟩ :=
        forRange_inv (idx := "kk") (start := 0) (stop := BLOCK_K * numKBlocks) (step := BLOCK_K)
          (body := sgmvLoopBody BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K)
          (by omega) hP0
          (fun i st hlt hinv =>
            sgmv_stepW input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s
              N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
              BLOCK_M BLOCK_N BLOCK_K numKBlocks hBK i st hlt hinv)
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _
          (sgmvBody_castFree R BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K) "kk",
        ← stepForRangeAux.forRange_unfold, hLoopStmt] at hs2
      obtain rfl := Option.some.inj hs2
      simp only [sgmvSafeInv] at hPL
      obtain ⟨-, -, -, homL, honL, hcssL, hML, -, -⟩ := hPL
      exact sgmv_tailSafeR R bounds out_ptr b_seq_start_loc seq_lens s
        N cm_stride cn_stride slice_offset BLOCK_M BLOCK_N sLoop homL honL hcssL hML hbC

/-- The full sgmv surface sits inside the flat-memory bridge's covered
fragment (`FlattenOk`; the `forRange` clause recurses into the cast-free
body). -/
private theorem sgmv_flattenOk (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) :
    ((sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
        lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
        lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [sgmv_body_split' input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
    slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks]
  simp [sgmvPrologue, sgmvPidStmts, sgmvLoadStmts, sgmvIdxStmts, sgmvPtrStmts,
    sgmvLoopBody, sgmvStoreTail, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

/-! ### IO signature, lane bridges, spec bridge -/

/-- The `input`-stream lane feeding output lane `l` at inner key `e`: the row
of `l` (row-major over the `[BLOCK_M, BLOCK_N]` output tile) paired with `e`
over the `[BLOCK_M, BLOCK_K]` per-step `a`-tile, via the shared `Lane2D`
bridge. -/
def aLane (BLOCK_M BLOCK_N BLOCK_K : Nat) (l : Fin (BLOCK_M * BLOCK_N))
    (e : Fin BLOCK_K) : Fin (BLOCK_M * BLOCK_K) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)

/-- The `loraB`-stream lane feeding output lane `l` at inner key `e`: `e`
paired with the column of `l` over the `[BLOCK_K, BLOCK_N]` per-step
`b`-tile. -/
def bLane (BLOCK_M BLOCK_N BLOCK_K : Nat) (l : Fin (BLOCK_M * BLOCK_N))
    (e : Fin BLOCK_K) : Fin (BLOCK_K * BLOCK_N) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)

set_option maxHeartbeats 1000000 in
/-- Under the two stream pins (stated in launch-state vocabulary), `sgmvSpec`
at the decoded output lane **is** the skin-level double fold
`∑ t, ∑ e, xs · ys` (`StreamLane.sum_range_mul` + address identity of the
windows with `aElem`/`bElem`). -/
private theorem sgmvSpec_eq_streamSum (input_ptr lora_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat) (s₀ : BlockState)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (xs : Fin numKBlocks → Fin (BLOCK_M * BLOCK_K) → ℝ)
    (ys : Fin numKBlocks → Fin (BLOCK_K * BLOCK_N) → ℝ)
    (hx : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_M * BLOCK_K)),
      s₀.readMem input_ptr (seqStart s₀ b_seq_start_loc * xm_stride
        + ((s₀.pids 0 * BLOCK_M + j.val / BLOCK_K) % seqLen s₀ seq_lens) * xm_stride
        + (t.val * BLOCK_K + j.val % BLOCK_K) * xk_stride) = xs t j)
    (hy : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_K * BLOCK_N)),
      s₀.readMem lora_ptr (l0_stride * loraIdx s₀ lora_indices
        + (t.val * BLOCK_K + j.val / BLOCK_N) * lora_n_stride
        + ((s₀.pids 1 * BLOCK_N + j.val % BLOCK_N) % N) * lora_k_stride) = ys t j)
    (l : Fin (BLOCK_M * BLOCK_N)) :
    sgmvSpec s₀ input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K numKBlocks (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = ∑ t : Fin numKBlocks, ∑ e : Fin BLOCK_K,
          xs t (aLane BLOCK_M BLOCK_N BLOCK_K l e) * ys t (bLane BLOCK_M BLOCK_N BLOCK_K l e) := by
  unfold sgmvSpec
  rw [Nat.mul_comm BLOCK_K numKBlocks, StreamLane.sum_range_mul]
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => ?_
  have hxa : aElem s₀ input_ptr b_seq_start_loc seq_lens xm_stride xk_stride BLOCK_M
      (Lane2D.decode l).1 (t.val * BLOCK_K + e.val)
      = xs t (aLane BLOCK_M BLOCK_N BLOCK_K l e) := by
    rw [← hx t (aLane BLOCK_M BLOCK_N BLOCK_K l e)]
    simp only [aLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  have hyb : bElem s₀ lora_ptr lora_indices N l0_stride lora_k_stride lora_n_stride BLOCK_N
      (Lane2D.decode l).2.1 (t.val * BLOCK_K + e.val)
      = ys t (bLane BLOCK_M BLOCK_N BLOCK_K l e) := by
    rw [← hy t (bLane BLOCK_M BLOCK_N BLOCK_K l e)]
    simp only [bLane, Lane2D.encode_div, Lane2D.encode_mod]
    rfl
  rw [hxa, hyb]

/-- Slot-region table of the three per-batch metadata slots, in the kernel's
own load order: slot `0` = `seq_lens` (the sequence length `M`), slot `1` =
`b_seq_start_loc` (the token offset `cur_seq_start`), slot `2` =
`lora_indices` (the adapter slot `lora_index`). A shared def, never an
inline `match` in a window/spec position. -/
def sgmvMetaBuf (b_seq_start_loc seq_lens lora_indices : Region .nat) : Fin 3 → RegionName
  | ⟨0, _⟩ => seq_lens.cast
  | ⟨1, _⟩ => b_seq_start_loc.cast
  | ⟨_ + 2, _⟩ => lora_indices.cast

/-- **Streaming metadata IO signature** of `sgmv_expand_slice` on the
metadata-parametrized two-stream fold skin (S1: fold + terminal masked
store, 3-D pid grid). The three `.nat` metadata slots are the kernel's own
per-batch scalars, all loaded at cell `cur_batch = pid₂` of their own
regions (`sgmvMetaBuf`; no chained slot indirection). Step `c` of the K-loop
reads the `[BLOCK_M, BLOCK_K]` `input`-tile and the `[BLOCK_K, BLOCK_N]`
`loraB`-tile **unmasked** (the transcribed path is the `EVEN_K` shape,
`K = BLOCK_K · numKBlocks` baked into the loop structure, so `mask1`/`mask2`
are `True`); after the loop one `[BLOCK_M, BLOCK_N]` output tile is
masked-stored at the **`.real`** grid (`outDType` default — the store is an
untyped `tl.store` at `.real`, no quantization event). The windows
transcribe the kernel's pointer arithmetic exactly, with the loaded slot
vector `m` in place of the in-state metadata reads
(`m 0 = M`, `m 1 = cur_seq_start`, `m 2 = lora_index`):

* `read1` lane `j = (i, e)` (row-major over `[BLOCK_M, BLOCK_K]`), step `t`:
  `m 1·xm + ((pid₀·BLOCK_M + i) % m 0)·xm + (t·BLOCK_K + e)·xk` — the
  invariant's `a_ptr` cell after `t` advances (the `ram = offset_m % M`
  gather geometry, slot-parametrized).
* `read2` lane `j = (e, n)` (row-major over `[BLOCK_K, BLOCK_N]`), step `t`:
  `l0·m 2 + (t·BLOCK_K + e)·ln + ((pid₁·BLOCK_N + n) % N)·lk` — the `b_ptr`
  cell (the `rbn = offset_n % N` gather and the `lora_index` bank select).
* `write` lane `j = (i, n)`:
  `(m 1 + (pid₀·BLOCK_M + i))·cm + ((pid₁·BLOCK_N + n) + slice_offset)·cn`
  — the kernel's `c_ptr` (= `cOffset` in pid/slot form).
* `writeMask` lane `j = (i, n)`:
  `pid₀·BLOCK_M + i < m 0 ∧ pid₁·BLOCK_N + n < N` — the kernel's `c_mask`
  (`offset_m < M & offset_n < N`), slot-parametrized. -/
def sgmvExpandSliceIO (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat) :
    StreamMetaMasked3DKernelIO₂ where
  kernel := sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
    lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
    lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K
  inp1 := input_ptr
  inp2 := lora_ptr
  out := out_ptr
  nMeta := 3
  sty := fun _ => ChanTy.nat
  mbuf := sgmvMetaBuf b_seq_start_loc seq_lens lora_indices
  mwin := fun _ _ _ pid₂ => pid₂
  T := numKBlocks
  B1 := BLOCK_M * BLOCK_K
  B2 := BLOCK_K * BLOCK_N
  C := BLOCK_M * BLOCK_N
  read1 := fun pid₀ _ _ m t j =>
    m (⟨1, by omega⟩ : Fin 3) * xm_stride
      + ((pid₀ * BLOCK_M + j.val / BLOCK_K) % m (⟨0, by omega⟩ : Fin 3)) * xm_stride
      + (t.val * BLOCK_K + j.val % BLOCK_K) * xk_stride
  read2 := fun _ pid₁ _ m t j =>
    l0_stride * m (⟨2, by omega⟩ : Fin 3)
      + (t.val * BLOCK_K + j.val / BLOCK_N) * lora_n_stride
      + ((pid₁ * BLOCK_N + j.val % BLOCK_N) % N) * lora_k_stride
  write := fun pid₀ pid₁ _ m j =>
    (m (⟨1, by omega⟩ : Fin 3) + (pid₀ * BLOCK_M + j.val / BLOCK_N)) * cm_stride
      + ((pid₁ * BLOCK_N + j.val % BLOCK_N) + slice_offset) * cn_stride
  mask1 := fun _ _ _ _ _ _ => True
  mask2 := fun _ _ _ _ _ _ => True
  writeMask := fun pid₀ pid₁ _ m j =>
    pid₀ * BLOCK_M + j.val / BLOCK_N < m (⟨0, by omega⟩ : Fin 3)
      ∧ pid₁ * BLOCK_N + j.val % BLOCK_N < N

/-! ### The headline -/

set_option maxHeartbeats 2000000 in
/-- **The `⊨[R]` streaming metadata headline (wave-5 S1 fold genre).** For
every rounding model `R`, the faithful `sgmv_expand_slice` surface
implements, on its `StreamMetaMasked3DKernelIO₂` signature, the **ideal ℝ
SGMV fold** over the streamed tiles: every write-active output lane
`l = (i, n)` holds `∑ t, ∑ e, a-tile[t](i,e) · b-tile[t](e,n)` — exact real
arithmetic; the slot vector `m` enters the spec only through the gather
geometry of the windows (`% m 0` row gather, `m 1` token offset, `m 2`
adapter bank) and the sentinel-shaped `writeMask`. The kernel has **no
rounding events** (`.nat` metadata loads, `.real` loads/dot, `.real` masked
terminal store), so the skin's boundary quantization degenerates: the
readback's `R.round .real` is the identity (`round_real`), and the `.real`
store is exact under `execR R` (`RoundingModel.storeValue_real`).

Layer map: the prologue and the whole K-loop are cast-free, so under
`execR R` they collapse verbatim onto the exact stepper and the proven
`sgmv_preLoop` / `sgmv_step` / `forRange_inv` stack above is reused
unchanged; only the masked terminal store is re-proved on the `R` side
(`sgmv_postLoopR`).

Both hypotheses are truth-forced:

* `hBK : 0 < BLOCK_K` — the surface's K-loop steps by `BLOCK_K`
  (`range(0, K, BLOCK_K)`); at `BLOCK_K = 0` the loop never advances and the
  block-index arithmetic `i / BLOCK_K` is meaningless — the same hypothesis
  the exact headline carries. It holds for every real launch.
* `hInj` — output-offset injectivity of the write window
  `(start + pid₀·BLOCK_M + i)·cm + ((pid₁·BLOCK_N + n) + slice)·cn`,
  the pid/slot-parametrized spelling of the exact headline's open side
  condition `hInj : Function.Injective (cOffset …)` (the skin quantifies the
  launch state internally, so the state-indexed spelling is not expressible
  here); with colliding output lanes the per-lane readback would be
  last-writer-wins and the statement false. As in the exact headline it is
  carried as an open side condition, not discharged.

Inherited modeling boundary (unchanged from the exact surface, see the file
docstring): `lora_indices` is Python `int32` with a `-1` skip sentinel; this
port's surface has always erased it to `.nat` and delegated the two host
guards (`pid_m·BLOCK_M > M` early return and the `-1` sentinel skip) to the
trusted launch boundary, so the slot channel here is `.nat` and the sentinel
skip is *not* expressed as an empty `writeMask` — the statement covers the
programs the host actually launches, exactly as the exact headline does.

Relation to the exact surface: the exact headline
`sgmv_expand_slice_one_row_block_output_summary`
(`Realizes_without_Rounding`) above is retained unchanged; this `⊨[R]` face
restates the same SGMV contraction on the streaming metadata skin, for every
`R` at once (at the `.real` grid the two faces carry the same exact cell).
Both faces are kept per the rounding-as-default doctrine. -/
specification sgmv_expand_slice_io_correctness (R : RoundingModel)
    (input_ptr lora_ptr out_ptr : RegionName)
    (b_seq_start_loc seq_lens lora_indices : Region .nat)
    (N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks : Nat)
    (hBK : 0 < BLOCK_K)
    (hInj : ∀ pid₀ pid₁ start : Nat,
      Function.Injective (fun idx : TileIndex [BLOCK_M, BLOCK_N] =>
        (start + (pid₀ * BLOCK_M + idx.1.val)) * cm_stride
          + ((pid₁ * BLOCK_N + idx.2.1.val) + slice_offset) * cn_stride)) :
    sgmvExpandSliceIO input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks ⊨[R]
      fun _ _ _ _ xs ys l =>
        ∑ t : Fin numKBlocks, ∑ e : Fin BLOCK_K,
          xs t (aLane BLOCK_M BLOCK_N BLOCK_K l e)
            * ys t (bLane BLOCK_M BLOCK_N BLOCK_K l e) := by
  refine StreamMetaMasked3DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact sgmv_flattenOk input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
      N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
      slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks
  · -- safety walk
    intro bounds s m xs ys hm _hx _hy hbm hbr1 hbr2 hbw
    simp only [sgmvExpandSliceIO] at hm hbm hbr1 hbr2 hbw
    have hm0 : seqLen s seq_lens = m (⟨0, by omega⟩ : Fin 3) := hm (⟨0, by omega⟩ : Fin 3)
    have hm1 : seqStart s b_seq_start_loc = m (⟨1, by omega⟩ : Fin 3) := hm (⟨1, by omega⟩ : Fin 3)
    have hm2 : loraIdx s lora_indices = m (⟨2, by omega⟩ : Fin 3) := hm (⟨2, by omega⟩ : Fin 3)
    refine sgmv_traceSafeR R bounds input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
      lora_indices N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks hBK s
      (hbm (⟨0, by omega⟩ : Fin 3)) (hbm (⟨1, by omega⟩ : Fin 3)) (hbm (⟨2, by omega⟩ : Fin 3)) ?_ ?_ ?_
    · intro t j
      have h := hbr1 t j trivial
      rw [← hm0, ← hm1] at h
      exact h
    · intro t j
      have h := hbr2 t j trivial
      rw [← hm2] at h
      exact h
    · intro j hj1 hj2
      rw [hm0] at hj1
      have h := hbw j ⟨hj1, hj2⟩
      rw [← hm1] at h
      exact h
  · -- the rounded Hoare triple
    intro s₀ m xs ys hundef hm hx hy
    simp only [sgmvExpandSliceIO] at hm hx hy ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hundef]
    have hm0 : seqLen s₀ seq_lens = m (⟨0, by omega⟩ : Fin 3) := hm (⟨0, by omega⟩ : Fin 3)
    have hm1 : seqStart s₀ b_seq_start_loc = m (⟨1, by omega⟩ : Fin 3) := hm (⟨1, by omega⟩ : Fin 3)
    have hm2 : loraIdx s₀ lora_indices = m (⟨2, by omega⟩ : Fin 3) := hm (⟨2, by omega⟩ : Fin 3)
    have hInj' : Function.Injective
        (cOffset s₀ b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N) :=
      hInj (s₀.pids 0) (s₀.pids 1) (seqStart s₀ b_seq_start_loc)
    have hx' : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_M * BLOCK_K)),
        s₀.readMem input_ptr (seqStart s₀ b_seq_start_loc * xm_stride
          + ((s₀.pids 0 * BLOCK_M + j.val / BLOCK_K) % seqLen s₀ seq_lens) * xm_stride
          + (t.val * BLOCK_K + j.val % BLOCK_K) * xk_stride) = xs t j := by
      intro t j
      rw [hm0, hm1]
      exact hx t j trivial
    have hy' : ∀ (t : Fin numKBlocks) (j : Fin (BLOCK_K * BLOCK_N)),
        s₀.readMem lora_ptr (l0_stride * loraIdx s₀ lora_indices
          + (t.val * BLOCK_K + j.val / BLOCK_N) * lora_n_stride
          + ((s₀.pids 1 * BLOCK_N + j.val % BLOCK_N) % N) * lora_k_stride) = ys t j := by
      intro t j
      rw [hm2]
      exact hy t j trivial
    -- exact preLoop + K-loop (cast-free, so they are the `execR` run too)
    obtain ⟨s1, hpre, hP0⟩ := sgmv_preLoop input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
      lora_indices s₀ N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
      cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks hundef'
    obtain ⟨final, sLoop, hLoopStmt, hfinal, hPLoop⟩ :=
      forRange_inv (idx := "kk") (start := 0) (stop := BLOCK_K * numKBlocks) (step := BLOCK_K)
        (by omega) hP0
        (fun i st hlt hinv => sgmv_step input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s₀
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K
          numKBlocks hBK i st hlt hinv)
    have hfinalEq : final = BLOCK_K * numKBlocks := by
      have hle : final ≤ BLOCK_K * numKBlocks := by
        simp only [sgmvInvariant] at hPLoop
        obtain ⟨_, hieq, hcle, _⟩ := hPLoop
        calc final = final / BLOCK_K * BLOCK_K := hieq
          _ ≤ numKBlocks * BLOCK_K := Nat.mul_le_mul_right _ hcle
          _ = BLOCK_K * numKBlocks := Nat.mul_comm _ _
      exact le_antisymm hle hfinal
    subst hfinalEq
    have hmem0 : sLoop.mem = s₀.mem := by
      simp only [sgmvInvariant] at hPLoop
      exact hPLoop.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
    -- R-side masked store tail
    obtain ⟨sfin, hTailR, hval, hframe⟩ :=
      sgmv_postLoopR R input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices s₀
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks hBK hInj' sLoop hPLoop
    have hLoopR : stepStmtR R (Stmt.forRange "kk" 0 (BLOCK_K * numKBlocks) BLOCK_K
        (sgmvLoopBody BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K)) s1 = some sLoop := by
      rw [stepStmtR_forRange,
        stepForRangeAuxR_castFree R _
          (sgmvBody_castFree R BLOCK_M BLOCK_N BLOCK_K xk_stride lora_n_stride BLOCK_K) "kk",
        ← stepForRangeAux.forRange_unfold]
      exact hLoopStmt
    have hpre' : stepStmts (sgmvPrologue input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K) s₀
        = some s1 := by
      rw [← sgmv_take14_eq input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
        slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks]
      exact hpre
    refine ⟨sfin, ?_, ?_, ?_⟩
    · show execR R (sgmv_expand_slice_surface input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens
          lora_indices N (BLOCK_K * numKBlocks) xm_stride xk_stride l0_stride lora_k_stride
          lora_n_stride cm_stride cn_stride slice_offset BLOCK_M BLOCK_N BLOCK_K).toAlgKernel s₀
        = some sfin
      unfold execR
      rw [sgmv_body_split' input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride
          slice_offset BLOCK_M BLOCK_N BLOCK_K numKBlocks,
        stepStmtsR_append,
        sgmvPrologue_castFree R input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices
          N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride BLOCK_M BLOCK_N BLOCK_K s₀,
        hpre', Option.bind_some, stepStmtsR_cons_some hLoopR]
      exact hTailR
    · intro j hj
      have hact : activeLane s₀ seq_lens N BLOCK_M BLOCK_N (Lane2D.decode j) := by
        refine ⟨?_, hj.2⟩
        rw [hm0]
        exact hj.1
      rw [← hm1]
      have haddr : (seqStart s₀ b_seq_start_loc + (s₀.pids 0 * BLOCK_M + j.val / BLOCK_N)) * cm_stride
            + ((s₀.pids 1 * BLOCK_N + j.val % BLOCK_N) + slice_offset) * cn_stride
          = cOffset s₀ b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N
              (Lane2D.decode j) := rfl
      rw [haddr, readMemAs_real_of_cell (hval (Lane2D.decode j) hact), R.round_real_apply]
      congr 1
      exact sgmvSpec_eq_streamSum input_ptr lora_ptr b_seq_start_loc seq_lens lora_indices s₀
        N xm_stride xk_stride l0_stride lora_k_stride lora_n_stride
        BLOCK_M BLOCK_N BLOCK_K numKBlocks xs ys hx' hy' j
    · intro r o hcond
      have hcond' : r ≠ out_ptr ∨ ∀ idx : TileIndex [BLOCK_M, BLOCK_N],
          activeLane s₀ seq_lens N BLOCK_M BLOCK_N idx →
          o ≠ cOffset s₀ b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx := by
        rcases hcond with hne | hno
        · exact Or.inl hne
        · refine Or.inr fun idx hact => ?_
          have h := hno (Lane2D.encode idx)
            ⟨by rw [Lane2D.encode_div, ← hm0]; exact hact.1,
             by rw [Lane2D.encode_mod]; exact hact.2⟩
          rw [← hm1] at h
          have haddr : (seqStart s₀ b_seq_start_loc
                + (s₀.pids 0 * BLOCK_M + (Lane2D.encode idx).val / BLOCK_N)) * cm_stride
              + ((s₀.pids 1 * BLOCK_N + (Lane2D.encode idx).val % BLOCK_N) + slice_offset) * cn_stride
              = cOffset s₀ b_seq_start_loc cm_stride cn_stride slice_offset BLOCK_M BLOCK_N idx := by
            simp only [Lane2D.encode_div, Lane2D.encode_mod]
            rfl
          rw [haddr] at h
          exact h
      rw [hframe r o hcond', hmem0]

end VeriTile.Bench.TritonBenchG.SgmvExpandSlice
