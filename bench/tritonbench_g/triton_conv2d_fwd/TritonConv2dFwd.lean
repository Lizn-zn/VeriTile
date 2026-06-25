import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Kernel

/-!
# `triton_conv2d_fwd` — closed-form conv2d-forward correctness

`conv2d_forward_kernel` is a grouped 2D convolution forward pass realised as an
implicit GEMM / im2col dot-accumulator. Each program covers a flattened
batch/height/width tile (`BLOCK_BHW`) and an output-feature tile
(`BLOCK_OUT_FEAT`), and accumulates

```
accum[i, j] += tl.dot(input_block, weight_block)
```

inside a nested `kernel_height × kernel_width × in_feat` loop, with masked input
and weight loads (padding/boundary aware) and a final masked store of the
accumulator into `Output`.

This file proves the **full nested loop** correct against a genuine mathematical
convolution reference: every output cell of the computed tile equals

```
out[i, j] = Σ_{h<KH} Σ_{w<KW} Σ_{c<in_group_dim}
              maskedInput(h, w, c, i) · maskedWeight(h, w, c, j)
```

over `ℝ`, where `maskedInput`/`maskedWeight` are the loaded input/weight values
zeroed out exactly where the kernel's per-lane masks are false (image/channel
boundaries — i.e. zero padding). This is NOT the kernel's own executed value: it
is the independent closed-form im2col convolution reference, derived from the
loaded `Input`/`Weight` tiles.

## Proof architecture

```
conv2d_closed_form_correct                    ← TOP THEOREM (ComputeCorrect.Realizes)
  └─ conv2d_exec_closed_form                  ← exec-side closed form (active cells = the conv sum)
       ├─ conv2d_preLoop      (P 0,0,0 : accum = 0, Input/Weight seeded)
       ├─ conv2d_c_step / c_loop  (inner in-feat dot loop: accum += masked dot)
       ├─ conv2d_w_step / w_loop  (middle kernel-width loop)
       ├─ conv2d_h_loop           (outer kernel-height loop)
       ├─ conv2d_postLoop         (masked 2D writeback = the closed form)
       └─ forRange(Aux/Dyn)_inv   (loop-invariant principle drives each loop)
```

## Modeling boundary

Arithmetic is over `ℝ` (the in-model fp16 cast is the identity, so the
`fp16=true` round-trip on the dot operands is a no-op). `@triton.autotune` /
`num_warps` and grouped-launch scheduling are not modeled. The host launch (the
3D grid, the scheduling, and how the runtime composes per-program writes) is the
trusted boundary; the per-program statement is universally quantified over the
initial state `s`, so it covers every program of the grid. The flattened BHW
index decomposition (`//`/`%` into batch/height/width) is transcribed faithfully
and appears in both the store offset and the masks.

Preconditions for the general theorem: `fp16 = true` (the DSL always wraps the
dot operands in a fp16→real cast; only the `fp16` branch makes the body
well-typed — under ℝ the cast is the identity); `0 < BLOCK_IN`; `in_group_dim`
divisible into `numCBlocks` full input-feature blocks; clean initial `undef`
(so a masked-out load lane reads `0`, i.e. zero padding); output-address
injectivity (distinct active lanes hit distinct `Output` addresses).
-/

namespace VeriTile.Bench.TritonBenchG.TritonConv2dFwd

open VeriTile.Triton

set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-- Faithful transcription of `triton_conv2d_fwd.py`'s `conv2d_forward_kernel`. -/
def conv2d_forward_surface
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      input_batch_stride input_in_feat_stride input_height_stride input_width_stride
      weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      kernel_height kernel_width stride_height stride_width padding_height padding_width groups : Nat)
    (_fp16 _tf32 : Bool)
    (BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat) :
    ComputeKernel := triton {
  batch_height_width_pid = tl.program_id(0)
  out_feat_pid = tl.program_id(1)
  group_pid = tl.program_id(2)
  in_group_dim = $(in_feat_dim) // $(groups)
  out_group_dim = $(out_feat_dim) // $(groups)
  batch_height_width_offset =
    batch_height_width_pid * $(BLOCK_SIZE_BATCH_HEIGHT_WIDTH) +
      tl.arange(0, $(BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
  batch_height_offset = batch_height_width_offset // $(out_width)
  batch_offset = batch_height_offset // $(out_height)
  output_feat_offset = out_feat_pid * $(BLOCK_SIZE_OUT_FEAT) +
    tl.arange(0, $(BLOCK_SIZE_OUT_FEAT))
  output_height_offset = batch_height_offset % $(out_height)
  output_width_offset = batch_height_width_offset % $(out_width)
    Input +=
      ($(input_batch_stride) * batch_offset +
        $(input_in_feat_stride) * group_pid * in_group_dim)[:, None]
    Weight +=
      ($(weight_out_feat_stride) * output_feat_offset +
        $(weight_out_feat_stride) * group_pid * out_group_dim)[None, :]
  accum = tl.zeros([$(BLOCK_SIZE_BATCH_HEIGHT_WIDTH), $(BLOCK_SIZE_OUT_FEAT)], dtype=tl.float32)
  for h in range($(0), $(kernel_height), $(1)) {
    for w in range($(0), $(kernel_width), $(1)) {
      for c in range($(0), in_group_dim, $(BLOCK_SIZE_IN_FEAT)) {
        input_feat_offset = c + tl.arange(0, $(BLOCK_SIZE_IN_FEAT))
        input_height_offset = h - $((padding_height : Int)) +
          $(stride_height) * output_height_offset
        input_width_offset = w - $((padding_width : Int)) +
          $(stride_width) * output_width_offset
          curr_input_pointer = Input +
            ($(input_in_feat_stride) * input_feat_offset)[None, :] +
            ($(input_height_stride) * input_height_offset)[:, None] +
            ($(input_width_stride) * input_width_offset)[:, None]
          curr_weight_pointer = Weight +
            ($(weight_in_feat_stride) * input_feat_offset)[:, None] +
            $(weight_height_stride) * h + $(weight_width_stride) * w
        input_mask = (batch_offset[:, None] < $(batch_dim)) &
          (input_feat_offset[None, :] < in_group_dim) &
          ($((0 : Int)) <= input_height_offset[:, None]) &
          (input_height_offset[:, None] < $(in_height)) &
          ($((0 : Int)) <= input_width_offset[:, None]) &
          (input_width_offset[:, None] < $(in_width))
        weight_mask = (input_feat_offset[:, None] < in_group_dim) &
          (output_feat_offset[None, :] < out_group_dim)
          input_block = tl.load(curr_input_pointer, mask=input_mask)
          weight_block = tl.load(curr_weight_pointer, mask=weight_mask)
        if _fp16 {
          input_block = (input_block).to(tl.float16)
          weight_block = (weight_block).to(tl.float16)
        }
        accum += tl.dot(input_block, weight_block, allow_tf32=_tf32)
      }
    }
  }
    Output += $(output_batch_stride) * batch_offset[:, None] +
      $(output_out_feat_stride) * (group_pid * out_group_dim + output_feat_offset)[None, :] +
      $(output_height_stride) * output_height_offset[:, None] +
      $(output_width_stride) * output_width_offset[:, None]
    output_mask = (batch_offset[:, None] < $(batch_dim)) &
      (output_feat_offset[None, :] < out_group_dim) &
      (output_height_offset[:, None] < $(out_height)) &
      (output_width_offset[:, None] < $(out_width))
    tl.store(Output, accum, mask=output_mask)
}

/-- The full conv2d forward surface lowers to the algorithm layer. -/
theorem conv2d_forward_surface_toAlgorithm_supported
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      input_batch_stride input_in_feat_stride input_height_stride input_width_stride
      weight_out_feat_stride weight_in_feat_stride weight_height_stride weight_width_stride
      output_batch_stride output_out_feat_stride output_height_stride output_width_stride
      kernel_height kernel_width stride_height stride_width padding_height padding_width groups : Nat)
    (_fp16 _tf32 : Bool)
    (BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT BLOCK_SIZE_OUT_FEAT : Nat) :
    ∃ alg,
      (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height
        in_width out_feat_dim out_height out_width input_batch_stride
        input_in_feat_stride input_height_stride input_width_stride
        weight_out_feat_stride weight_in_feat_stride weight_height_stride
        weight_width_stride output_batch_stride output_out_feat_stride
        output_height_stride output_width_stride kernel_height kernel_width
        stride_height stride_width padding_height padding_width groups _fp16 _tf32
        BLOCK_SIZE_BATCH_HEIGHT_WIDTH BLOCK_SIZE_IN_FEAT
        BLOCK_SIZE_OUT_FEAT).toAlgorithm? = Except.ok alg := by
  simp [conv2d_forward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]

@[simp] theorem evalOp_floorDiv {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.floorDiv h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.floorDiv bc vx vy)) := by
  simp [evalOp]

@[simp] theorem evalOp_mod {dtype a b shape} (h : IntegralDType dtype)
    (bc : Broadcast a b shape) (x : Op dtype a) (y : Op dtype b) (s : BlockState) :
    evalOp (.mod h bc x y) s = (do
      let vx ← evalOp x s; let vy ← evalOp y s; some (Tile.bop h.mod bc vx vy)) := by
  simp [evalOp]

/-- A masked `.ptr` load (no `other`) with clean `undef`: lane `(i)` reads
`readMem` at the pointer when the mask is true, else `0` (zero padding). -/
theorem load_ptr_mask_clean {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape)
    (s : BlockState) (ptrs : Tile .ptr shape) (mtile : Tile .bool shape)
    (hp : evalOp ptrOp s = some ptrs)
    (hm : evalOp maskOp s = some mtile)
    (hundef : ∀ rg o, s.undef rg o = 0) :
    evalOp (.load .real (.ptr ptrOp) (.mask maskOp)) s
      = some ⟨fun i => if mtile.data i then some (s.readMem (ptrs.data i).1 (ptrs.data i).2) else some 0⟩ := by
  unfold evalOp
  simp only [hp, hm, Option.bind_eq_bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  by_cases hb : mtile.data i
  · simp only [hb, if_true, BlockState.readMemValue_real]
  · simp only [hb, if_false, Bool.false_eq_true, hundef, if_true]

/-- The masked dot of two loaded tiles whose lanes are `some (f …)`/`some (g …)`,
lane `(i,j)`, equals `some (Σ_e f e · g e)`. -/
theorem dot_mask (BHW BIN OF : Nat) (x : Tile .real [BHW, BIN]) (y : Tile .real [BIN, OF])
    (i : Fin BHW) (j : Fin OF) (fx : Fin BIN → ℝ) (fy : Fin BIN → ℝ)
    (hx : ∀ e : Fin BIN, x.data (i, e, PUnit.unit) = some (fx e))
    (hy : ∀ e : Fin BIN, y.data (e, j, PUnit.unit) = some (fy e)) :
    (Tile.dot [] x y).data (i, j, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin BIN => fx e * fy e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin BIN) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (x.data (i, e, PUnit.unit)) (y.data (e, j, PUnit.unit))))
      = @Finset.sum (Fin BIN) (WithBot ℝ) _ Finset.univ (fun e => (some (fx e * fy e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [hx e, hy e]; rfl)]
  show (Finset.univ.sum fun e => ((fx e * fy e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]; rfl

/-- `accum + dot` lane `(i,j)`: `some (zv + dv)`. -/
theorem accadd_eval (BHW OF : Nat) (zt dt : Tile .real [BHW, OF]) (i : Fin BHW) (j : Fin OF) (zv dv : ℝ)
    (hz : zt.data (i, j, PUnit.unit) = some zv) (hd : dt.data (i, j, PUnit.unit) = some dv) :
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) zt dt).data
        (i, j, PUnit.unit) = some (zv + dv) := by
  simp only [Tile.bop_data, Broadcast.leftIndex, Broadcast.rightIndex, hz, hd, NumericDType.add,
    WithBot.realAdd, Option.map₂, Option.bind, Option.map]

/-! ## Index / address / mask / spec layer

All quantities are functions of the initial state `s0` (program ids fixed) and
the strides/dims; they mirror exactly the kernel's own pointer arithmetic and
masks. Throughout, `i : Fin BHW` indexes a BHW lane, `j : Fin OF` an
output-feature lane, `e : Fin BIN` an input-feature lane within a block. -/

/-- Flattened BHW index of lane `i`: `pid0 · BHW + i`. -/
def bhwIdx (s0 : BlockState) (BHW : Nat) (i : Fin BHW) : Nat := s0.pids 0 * BHW + i.val

/-- `batch_height_offset = bhwIdx // out_width`. -/
def bhIdx (s0 : BlockState) (OW BHW : Nat) (i : Fin BHW) : Nat := bhwIdx s0 BHW i / OW

/-- `batch_offset = bhIdx // out_height`. -/
def batchIdx (s0 : BlockState) (OH OW BHW : Nat) (i : Fin BHW) : Nat := bhIdx s0 OW BHW i / OH

/-- `output_height_offset = bhIdx % out_height`. -/
def heightIdx (s0 : BlockState) (OH OW BHW : Nat) (i : Fin BHW) : Nat := bhIdx s0 OW BHW i % OH

/-- `output_width_offset = bhwIdx % out_width`. -/
def widthIdx (s0 : BlockState) (OW BHW : Nat) (i : Fin BHW) : Nat := bhwIdx s0 BHW i % OW

/-- `output_feat_offset = pid1 · OF + j`. -/
def featIdx (s0 : BlockState) (OF : Nat) (j : Fin OF) : Nat := s0.pids 1 * OF + j.val

/-- Per-lane input row base (the `Input +=` prefix), lane `i`:
`input_batch_stride · batchIdx i + input_in_feat_stride · pid2 · in_group_dim`. -/
def inputBase (s0 : BlockState) (IBS IIFS OH OW BHW IGD : Nat) (i : Fin BHW) : Nat :=
  IBS * batchIdx s0 OH OW BHW i + IIFS * s0.pids 2 * IGD

/-- Per-lane weight col base (the `Weight +=` prefix), lane `j`:
`weight_out_feat_stride · featIdx j + weight_out_feat_stride · pid2 · out_group_dim`. -/
def weightBase (s0 : BlockState) (WOFS OF OGD : Nat) (j : Fin OF) : Nat :=
  WOFS * featIdx s0 OF j + WOFS * s0.pids 2 * OGD

/-- The kernel's `input_height_offset` (an `Int`): `h - padding_height + stride_height · heightIdx i`. -/
def inHeightOff (s0 : BlockState) (OH OW BHW SH PH : Nat) (h : Nat) (i : Fin BHW) : Int :=
  (h : Int) - (PH : Int) + (SH : Int) * (heightIdx s0 OH OW BHW i : Int)

/-- The kernel's `input_width_offset` (an `Int`): `w - padding_width + stride_width · widthIdx i`. -/
def inWidthOff (s0 : BlockState) (OW BHW SW PW : Nat) (w : Nat) (i : Fin BHW) : Int :=
  (w : Int) - (PW : Int) + (SW : Int) * (widthIdx s0 OW BHW i : Int)

/-- The kernel's input load address at lane `(i,e)`, block `(h,w,c)`:
`inputBase i + IIFS·(c+e) + (IHS·inHeightOff).toNat + (IWS·inWidthOff).toNat`. -/
def inAddr (s0 : BlockState)
    (IBS IIFS IHS IWS OH OW BHW IGD SH SW PH PW : Nat) (h w c : Nat) (i : Fin BHW) (e : Fin BIN) : Nat :=
  inputBase s0 IBS IIFS OH OW BHW IGD i + IIFS * (c + e.val)
    + ((IHS : Int) * inHeightOff s0 OH OW BHW SH PH h i).toNat
    + ((IWS : Int) * inWidthOff s0 OW BHW SW PW w i).toNat

/-- The kernel's input mask Prop at lane `(i,e)`, block `(h,w,c)`. -/
def inMask (s0 : BlockState)
    (batch_dim in_height in_width OH OW BHW IGD SH SW PH PW : Nat)
    (h w c : Nat) (i : Fin BHW) (e : Fin BIN) : Prop :=
  ((((batchIdx s0 OH OW BHW i < batch_dim ∧ c + e.val < IGD) ∧
      (0 : Int) ≤ inHeightOff s0 OH OW BHW SH PH h i) ∧
      inHeightOff s0 OH OW BHW SH PH h i < (in_height : Int)) ∧
      (0 : Int) ≤ inWidthOff s0 OW BHW SW PW w i) ∧
      inWidthOff s0 OW BHW SW PW w i < (in_width : Int)

instance (s0 : BlockState) (batch_dim in_height in_width OH OW BHW IGD SH SW PH PW h w c : Nat)
    (i : Fin BHW) (e : Fin BIN) :
    Decidable (inMask s0 batch_dim in_height in_width OH OW BHW IGD SH SW PH PW h w c i e) := by
  unfold inMask; infer_instance

/-- The kernel's weight load address at lane `(e,j)`, block `(h,w,c)`:
`weightBase j + WIFS·(c+e) + WHS·h + WWS·w`. -/
def wAddr (s0 : BlockState)
    (WOFS WIFS WHS WWS OF OGD : Nat) (h w c : Nat) (e : Fin BIN) (j : Fin OF) : Nat :=
  weightBase s0 WOFS OF OGD j + WIFS * (c + e.val) + WHS * h + WWS * w

/-- The kernel's weight mask Prop at lane `(e,j)`, block `(h,w,c)`. -/
def wMask (s0 : BlockState) (IGD OGD OF c : Nat) (e : Fin BIN) (j : Fin OF) : Prop :=
  c + e.val < IGD ∧ featIdx s0 OF j < OGD

instance (s0 : BlockState) (IGD OGD OF c : Nat) (e : Fin BIN) (j : Fin OF) :
    Decidable (wMask s0 IGD OGD OF c e j) := by unfold wMask; infer_instance

/-- The masked input value at lane `(i,e)`, block `(h,w,c)`: the loaded `Input`
cell when the input mask holds, else `0` (zero padding under clean `undef`). -/
noncomputable def miVal (s0 : BlockState)
    (Input : RegionName) (BHW BIN batch_dim in_height in_width
      IBS IIFS IHS IWS OH OW IGD SH SW PH PW : Nat)
    (h w c : Nat) (i : Fin BHW) (e : Fin BIN) : ℝ :=
  if inMask s0 batch_dim in_height in_width OH OW BHW IGD SH SW PH PW h w c i e then
    s0.readMem Input (inAddr s0 IBS IIFS IHS IWS OH OW BHW IGD SH SW PH PW h w c i e)
  else 0

/-- The masked weight value at lane `(e,j)`, block `(h,w,c)`. -/
noncomputable def mwVal (s0 : BlockState)
    (Weight : RegionName) (BIN OF IGD OGD WOFS WIFS WHS WWS : Nat)
    (h w c : Nat) (e : Fin BIN) (j : Fin OF) : ℝ :=
  if wMask s0 IGD OGD OF c e j then
    s0.readMem Weight (wAddr s0 WOFS WIFS WHS WWS OF OGD h w c e j)
  else 0

/-- The per-block masked dot at output lane `(i,j)`, block `(h,w,c)`:
`Σ_{e<BIN} miVal(h,w,c,i,e) · mwVal(h,w,c,e,j)`. This is one `tl.dot`'s worth. -/
noncomputable def blockDot (s0 : BlockState)
    (Input Weight : RegionName) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW : Nat)
    (h w c : Nat) (i : Fin BHW) (j : Fin OF) : ℝ :=
  Finset.univ.sum fun e : Fin BIN =>
    miVal s0 Input BHW BIN batch_dim in_height in_width IBS IIFS IHS IWS OH OW IGD SH SW PH PW h w c i e
      * mwVal s0 Weight BIN OF IGD OGD WOFS WIFS WHS WWS h w c e j

/-- Accumulator after `cbCount` complete input-feature blocks within the
`(h, w)` iteration (the innermost-loop partial value, on top of the `accHW`
prefix). The c-loop index for block `cb` is `cb · BIN`. -/
noncomputable def accC (s0 : BlockState)
    (Input Weight : RegionName) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW : Nat)
    (h w cbCount : Nat) (i : Fin BHW) (j : Fin OF) : ℝ :=
  (Finset.range cbCount).sum fun cb =>
    blockDot s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW h w (cb * BIN) i j

/-- Accumulator after `wCount` complete kernel-width iterations within outer
iteration `h` (each width iteration runs the full `numCBlocks`-block c-loop). -/
noncomputable def accW (s0 : BlockState)
    (Input Weight : RegionName) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW numCBlocks : Nat)
    (h wCount : Nat) (i : Fin BHW) (j : Fin OF) : ℝ :=
  (Finset.range wCount).sum fun w =>
    accC s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW h w numCBlocks i j

/-- Accumulator after `hCount` complete kernel-height iterations (each runs the
full kernel-width loop). The full convolution value is `accH … KH`. -/
noncomputable def accH (s0 : BlockState)
    (Input Weight : RegionName) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW KW numCBlocks : Nat)
    (hCount : Nat) (i : Fin BHW) (j : Fin OF) : ℝ :=
  (Finset.range hCount).sum fun h =>
    accW s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW numCBlocks h KW i j

/-- **Genuine conv2d spec**: every output cell equals the full im2col convolution
sum over the kernel window and input-feature axis. -/
noncomputable def convSpec (s0 : BlockState)
    (Input Weight : RegionName) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW KH KW numCBlocks : Nat)
    (i : Fin BHW) (j : Fin OF) : ℝ :=
  accH s0 Input Weight BHW BIN OF batch_dim in_height in_width
    IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW KW numCBlocks KH i j

/-- One-block step of the innermost accumulator. -/
theorem accC_succ (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW : Nat)
    (h w cbCount : Nat) (i : Fin BHW) (j : Fin OF) :
    accC s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW h w (cbCount + 1) i j
      = accC s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW h w cbCount i j
        + blockDot s0 Input Weight BHW BIN OF batch_dim in_height in_width
            IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW h w (cbCount * BIN) i j := by
  unfold accC
  rw [Finset.sum_range_succ]

/-- One-iteration step of the middle (kernel-width) accumulator: a full
`numCBlocks`-block c-loop. -/
theorem accW_succ (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW numCBlocks : Nat)
    (h wCount : Nat) (i : Fin BHW) (j : Fin OF) :
    accW s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW numCBlocks h (wCount + 1) i j
      = accW s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW numCBlocks h wCount i j
        + accC s0 Input Weight BHW BIN OF batch_dim in_height in_width
            IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW h wCount numCBlocks i j := by
  unfold accW
  rw [Finset.sum_range_succ]

/-- One-iteration step of the outer (kernel-height) accumulator: a full
kernel-width loop. -/
theorem accH_succ (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW KW numCBlocks : Nat)
    (hCount : Nat) (i : Fin BHW) (j : Fin OF) :
    accH s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW KW numCBlocks (hCount + 1) i j
      = accH s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW KW numCBlocks hCount i j
        + accW s0 Input Weight BHW BIN OF batch_dim in_height in_width
            IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW numCBlocks hCount KW i j := by
  unfold accH
  rw [Finset.sum_range_succ]

/-! ## Inner-body eval lemmas -/

/-- `pid · N + arange` (scalar-times block plus index range). -/
theorem pidblock_eval (s : BlockState) (N P : Nat) (regName : RegName)
    (hp : s.regs .nat [] regName = some (Tile.scalar P)) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarL
      (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] regName) (Op.constNat N)) (Op.arange N')) s
      = some (Tile.vec (fun i : Fin N' => P * N + i.val)) := by
  simp only [evalOp_add, evalOp_mul, evalOp_ref, hp, evalOp_constNat, evalOp_arange, Option.bind,
    Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.mul]

/-- `floorDiv` of a seeded nat vector by a scalar. -/
theorem floordiv_vec_eval (s : BlockState) (N D : Nat) (regName : RegName) (g : Fin N → Nat)
    (hg : s.regs .nat [N] regName = some (Tile.vec g)) :
    evalOp (Op.floorDiv IntegralDType.nat Broadcast.scalarR (Op.ref TileDType.nat [N] regName) (Op.constNat D)) s
      = some (Tile.vec (fun i : Fin N => IntegralDType.floorDiv IntegralDType.nat (g i) D)) := by
  simp only [evalOp_floorDiv, evalOp_ref, hg, evalOp_constNat, Option.bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]

/-- `mod` of a seeded nat vector by a scalar. -/
theorem mod_vec_eval (s : BlockState) (N D : Nat) (regName : RegName) (g : Fin N → Nat)
    (hg : s.regs .nat [N] regName = some (Tile.vec g)) :
    evalOp (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref TileDType.nat [N] regName) (Op.constNat D)) s
      = some (Tile.vec (fun i : Fin N => IntegralDType.mod IntegralDType.nat (g i) D)) := by
  simp only [evalOp_mod, evalOp_ref, hg, evalOp_constNat, Option.bind, Option.bind_some]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex]

/-- `Input +=` eval: per-row base `IBS·batch_offset i + IIFS·group_pid·in_group_dim`. -/
theorem inputbase_eval (s : BlockState) (BHW IBS IIFS IGD : Nat) (I : RegionName) (gp : Nat)
    (bo : Fin BHW → Nat)
    (hbo : s.regs .nat [BHW] "batch_offset" = some (Tile.vec bo))
    (hgp : s.regs .nat [] "group_pid" = some (Tile.scalar gp))
    (hgd : s.regs .nat [] "in_group_dim" = some (Tile.scalar IGD)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase I)
      (Op.expandDim ⟨1, by simp⟩
        (Op.add NumericDType.nat Broadcast.scalarR
          (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat IBS) (Op.ref TileDType.nat [BHW] "batch_offset"))
          (Op.mul NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.constNat IIFS) (Op.ref TileDType.nat [] "group_pid"))
            (Op.ref TileDType.nat [] "in_group_dim"))))) s
      = some (⟨fun idx : TileIndex [BHW, 1] => (I.cast, IBS * bo idx.1 + IIFS * gp * IGD)⟩ : Tile .ptr [BHW, 1]) := by
  simp only [evalOp.eq_def, hbo, hgp, hgd, Option.bind, Option.bind_eq_bind]
  refine congrArg some ?_
  ext idx
  · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex]
  · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul]
    ring

/-- `Weight +=` eval: per-col base `WOFS·output_feat_offset j + WOFS·group_pid·out_group_dim`. -/
theorem weightbase_eval (s : BlockState) (OF WOFS OGD : Nat) (W : RegionName) (gp : Nat)
    (fo : Fin OF → Nat)
    (hfo : s.regs .nat [OF] "output_feat_offset" = some (Tile.vec fo))
    (hgp : s.regs .nat [] "group_pid" = some (Tile.scalar gp))
    (hod : s.regs .nat [] "out_group_dim" = some (Tile.scalar OGD)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase W)
      (Op.expandDim ⟨0, by simp⟩
        (Op.add NumericDType.nat Broadcast.scalarR
          (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat WOFS) (Op.ref TileDType.nat [OF] "output_feat_offset"))
          (Op.mul NumericDType.nat Broadcast.nil
            (Op.mul NumericDType.nat Broadcast.nil (Op.constNat WOFS) (Op.ref TileDType.nat [] "group_pid"))
            (Op.ref TileDType.nat [] "out_group_dim"))))) s
      = some (⟨fun idx : TileIndex [1, OF] => (W.cast, WOFS * fo idx.2.1 + WOFS * gp * OGD)⟩ : Tile .ptr [1, OF]) := by
  simp only [evalOp.eq_def, hfo, hgp, hod, Option.bind, Option.bind_eq_bind]
  refine congrArg some ?_
  ext idx
  · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex]
  · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul]
    ring

/-- `accum = tl.zeros` eval. -/
theorem accum_init_eval (s : BlockState) (BHW OF : Nat) :
    evalOp (Op.full [BHW, OF] (Op.const 0)) s
      = some (⟨fun _ : TileIndex [BHW, OF] => some (0 : ℝ)⟩ : Tile .real [BHW, OF]) := by
  simp [evalOp_full, evalOp_const, Option.bind]

/-- `input_feat_offset = c + arange` eval. -/
theorem input_feat_offset_eval (s : BlockState) (BIN c : Nat)
    (hc : s.regs .nat [] "c" = some (Tile.scalar c)) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "c") (Op.arange BIN)) s
      = some (Tile.vec (fun e : Fin BIN => c + e.val)) := by
  simp only [evalOp_add, evalOp_ref, hc, evalOp_arange, Option.bind, Option.bind_some]
  refine congrArg some ?_
  ext e
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add]

/-- `input_height_offset` / `input_width_offset` eval (Int): `t - P + S · off`. -/
theorem int_offset_eval (s : BlockState) (BHW P S : Nat) (off : Fin BHW → Nat) (t : Nat)
    (regOff : RegName) (regT : RegName)
    (htv : s.regs .nat [] regT = some (Tile.scalar t))
    (hoff : s.regs .nat [BHW] regOff = some (Tile.vec off)) :
    evalOp (Op.add NumericDType.int Broadcast.scalarL
      (Op.sub NumericDType.int Broadcast.nil (Op.castNatToInt (Op.ref TileDType.nat [] regT))
        (Op.constInt (Int.ofNat P)))
      (Op.mul NumericDType.int Broadcast.scalarL (Op.constInt (Int.ofNat S))
        (Op.castNatToInt (Op.ref TileDType.nat [BHW] regOff)))) s
      = some (Tile.vec (fun i : Fin BHW => (t : Int) - (P : Int) + (S : Int) * (off i : Int))) := by
  simp only [evalOp.eq_def, htv, hoff, Option.bind, Option.bind_eq_bind]
  refine congrArg some ?_
  ext i
  simp only [Tile.bop, Tile.vec, Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex,
    NumericDType.add, NumericDType.sub, NumericDType.mul, Int.ofNat_eq_natCast]

/-- `curr_input_pointer` eval: lane `(i,e)` points at
`baseI i + IIFS·(c+e) + (IHS·ihoff i).toNat + (IWS·iwoff i).toNat`. -/
theorem curr_input_pointer_eval (BHW BIN : Nat) (I : RegionName) (baseI : Fin BHW → Nat)
    (c IIFS IHS IWS : Nat) (ihoff iwoff : Fin BHW → Int) (s : BlockState)
    (hI : s.regs .ptr [BHW,1] "Input" = some ⟨fun idx : TileIndex [BHW,1] => (I.cast, baseI idx.1)⟩)
    (hf : s.regs .nat [BIN] "input_feat_offset" = some (Tile.vec (fun e : Fin BIN => c + e.val)))
    (hh : s.regs .int [BHW] "input_height_offset" = some (Tile.vec ihoff))
    (hw : s.regs .int [BHW] "input_width_offset" = some (Tile.vec iwoff)) :
    evalOp (Op.ptrAdd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.ptrAdd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ptrAdd (Broadcast.consR (Broadcast.consL Broadcast.nil)) (Op.ref TileDType.ptr [BHW, 1] "Input")
          (Op.expandDim ⟨0, by simp⟩
            (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat IIFS)
              (Op.ref TileDType.nat [BIN] "input_feat_offset"))))
        (Op.castIntToNat
          (Op.expandDim ⟨1, by simp⟩
            (Op.mul NumericDType.int Broadcast.scalarL (Op.castNatToInt (Op.constNat IHS))
              (Op.ref TileDType.int [BHW] "input_height_offset")))))
      (Op.castIntToNat
        (Op.expandDim ⟨1, by simp⟩
          (Op.mul NumericDType.int Broadcast.scalarL (Op.castNatToInt (Op.constNat IWS))
            (Op.ref TileDType.int [BHW] "input_width_offset"))))) s
    = some (⟨fun idx : TileIndex [BHW, BIN] =>
        (I.cast, baseI idx.1 + IIFS * (c + idx.2.1.val)
          + ((IHS : Int) * ihoff idx.1).toNat + ((IWS : Int) * iwoff idx.1).toNat)⟩ : Tile .ptr [BHW, BIN]) := by
  simp only [evalOp.eq_def, hI, hf, hh, hw, Option.bind, Option.bind_eq_bind]
  refine congrArg some ?_
  ext idx
  · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex]
  · simp [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex, NumericDType.mul, NumericDType.add]

/-- `curr_weight_pointer` eval: lane `(e,j)` points at
`baseW j + WIFS·(c+e) + WHS·h + WWS·w`. -/
theorem curr_weight_pointer_eval (BIN OF WIFS WHS WWS : Nat) (W : RegionName) (baseW : Fin OF → Nat)
    (c hh ww : Nat) (s : BlockState)
    (hW : s.regs .ptr [1,OF] "Weight" = some ⟨fun idx : TileIndex [1,OF] => (W.cast, baseW idx.2.1)⟩)
    (hf : s.regs .nat [BIN] "input_feat_offset" = some (Tile.vec (fun e : Fin BIN => c + e.val)))
    (hhv : s.regs .nat [] "h" = some (Tile.scalar hh))
    (hwv : s.regs .nat [] "w" = some (Tile.scalar ww)) :
    evalOp (Op.ptrAdd Broadcast.scalarR
      (Op.ptrAdd Broadcast.scalarR
        (Op.ptrAdd (Broadcast.consL (Broadcast.consR Broadcast.nil)) (Op.ref TileDType.ptr [1, OF] "Weight")
          (Op.expandDim ⟨1, by simp⟩
            (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat WIFS)
              (Op.ref TileDType.nat [BIN] "input_feat_offset"))))
        (Op.mul NumericDType.nat Broadcast.nil (Op.constNat WHS) (Op.ref TileDType.nat [] "h")))
      (Op.mul NumericDType.nat Broadcast.nil (Op.constNat WWS) (Op.ref TileDType.nat [] "w"))) s
    = some (⟨fun idx : TileIndex [BIN, OF] =>
        (W.cast, baseW idx.2.1 + WIFS * (c + idx.1.val) + WHS * hh + WWS * ww)⟩ : Tile .ptr [BIN, OF]) := by
  simp only [evalOp.eq_def, hW, hf, hhv, hwv, Option.bind, Option.bind_eq_bind]
  refine congrArg some ?_
  ext idx
  · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex]
  · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex, NumericDType.mul]

/-- `input_mask` eval: matches the `inMask` predicate cellwise. -/
theorem input_mask_eval (BHW BIN batch_dim in_height in_width IGD : Nat) (s : BlockState)
    (bo : Fin BHW → Nat) (c : Nat) (ihoff iwoff : Fin BHW → Int)
    (hbo : s.regs .nat [BHW] "batch_offset" = some (Tile.vec bo))
    (hf : s.regs .nat [BIN] "input_feat_offset" = some (Tile.vec (fun e : Fin BIN => c + e.val)))
    (hh : s.regs .int [BHW] "input_height_offset" = some (Tile.vec ihoff))
    (hw : s.regs .int [BHW] "input_width_offset" = some (Tile.vec iwoff))
    (hgd : s.regs .nat [] "in_group_dim" = some (Tile.scalar IGD)) :
    evalOp (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.lt ComparableDType.nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "batch_offset"))
                (Op.constNat batch_dim))
              (Op.lt ComparableDType.nat Broadcast.scalarR
                (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BIN] "input_feat_offset"))
                (Op.ref TileDType.nat [] "in_group_dim")))
            (Op.le ComparableDType.int Broadcast.scalarL (Op.constInt (Int.ofNat 0))
              (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_height_offset"))))
          (Op.lt ComparableDType.int Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_height_offset"))
            (Op.castNatToInt (Op.constNat in_height))))
        (Op.le ComparableDType.int Broadcast.scalarL (Op.constInt (Int.ofNat 0))
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_width_offset"))))
      (Op.lt ComparableDType.int Broadcast.scalarR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_width_offset"))
        (Op.castNatToInt (Op.constNat in_width)))) s
    = some (⟨fun idx : TileIndex [BHW, BIN] =>
        decide (((((bo idx.1 < batch_dim ∧ c + idx.2.1.val < IGD) ∧
          (0:Int) ≤ ihoff idx.1) ∧ ihoff idx.1 < (in_height:Int)) ∧
          (0:Int) ≤ iwoff idx.1) ∧ iwoff idx.1 < (in_width:Int))⟩ : Tile .bool [BHW, BIN]) := by
  simp only [evalOp.eq_def, hbo, hf, hh, hw, hgd, Option.bind, Option.bind_eq_bind]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt, ComparableDType.le, TileShape.dropInsertedIndex,
    Int.ofNat_eq_natCast, Bool.decide_and, Bool.and_assoc]
  push_cast
  rfl

/-- `weight_mask` eval: matches the `wMask` predicate cellwise. -/
theorem weight_mask_eval (BIN OF IGD OGD : Nat) (s : BlockState) (c : Nat) (ofo : Fin OF → Nat)
    (hf : s.regs .nat [BIN] "input_feat_offset" = some (Tile.vec (fun e : Fin BIN => c + e.val)))
    (ho : s.regs .nat [OF] "output_feat_offset" = some (Tile.vec ofo))
    (hgd : s.regs .nat [] "in_group_dim" = some (Tile.scalar IGD))
    (hod : s.regs .nat [] "out_group_dim" = some (Tile.scalar OGD)) :
    evalOp (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BIN] "input_feat_offset"))
        (Op.ref TileDType.nat [] "in_group_dim"))
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [OF] "output_feat_offset"))
        (Op.ref TileDType.nat [] "out_group_dim"))) s
    = some (⟨fun idx : TileIndex [BIN, OF] => decide (c + idx.1.val < IGD ∧ ofo idx.2.1 < OGD)⟩ : Tile .bool [BIN, OF]) := by
  simp only [evalOp.eq_def, hf, ho, hgd, hod, Option.bind, Option.bind_eq_bind]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt, TileShape.dropInsertedIndex]
  rw [Bool.decide_and]
  rfl

/-! ## Inner-loop body and invariant

The innermost `c`-loop body (11 statements), transcribed from the algorithm
layer (with `fp16 = true`, so the conditional fp16 cast runs and the dot's
fp16→real casts compose to the identity under ℝ). -/

def convInnerBody (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat) :
    List Stmt :=
  [ Stmt.assign .nat [BIN] "input_feat_offset"
      (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "c") (Op.arange BIN)),
    Stmt.assign .int [BHW] "input_height_offset"
      (Op.add NumericDType.int Broadcast.scalarL
        (Op.sub NumericDType.int Broadcast.nil (Op.castNatToInt (Op.ref TileDType.nat [] "h"))
          (Op.constInt (Int.ofNat PH)))
        (Op.mul NumericDType.int Broadcast.scalarL (Op.constInt (Int.ofNat SH))
          (Op.castNatToInt (Op.ref TileDType.nat [BHW] "output_height_offset")))),
    Stmt.assign .int [BHW] "input_width_offset"
      (Op.add NumericDType.int Broadcast.scalarL
        (Op.sub NumericDType.int Broadcast.nil (Op.castNatToInt (Op.ref TileDType.nat [] "w"))
          (Op.constInt (Int.ofNat PW)))
        (Op.mul NumericDType.int Broadcast.scalarL (Op.constInt (Int.ofNat SW))
          (Op.castNatToInt (Op.ref TileDType.nat [BHW] "output_width_offset")))),
    Stmt.assign .ptr [BHW, BIN] "curr_input_pointer"
      (Op.ptrAdd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.ptrAdd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.ptrAdd (Broadcast.consR (Broadcast.consL Broadcast.nil)) (Op.ref TileDType.ptr [BHW, 1] "Input")
            (Op.expandDim ⟨0, by simp⟩
              (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat IIFS)
                (Op.ref TileDType.nat [BIN] "input_feat_offset"))))
          (Op.castIntToNat
            (Op.expandDim ⟨1, by simp⟩
              (Op.mul NumericDType.int Broadcast.scalarL (Op.castNatToInt (Op.constNat IHS))
                (Op.ref TileDType.int [BHW] "input_height_offset")))))
        (Op.castIntToNat
          (Op.expandDim ⟨1, by simp⟩
            (Op.mul NumericDType.int Broadcast.scalarL (Op.castNatToInt (Op.constNat IWS))
              (Op.ref TileDType.int [BHW] "input_width_offset"))))),
    Stmt.assign .ptr [BIN, OF] "curr_weight_pointer"
      (Op.ptrAdd Broadcast.scalarR
        (Op.ptrAdd Broadcast.scalarR
          (Op.ptrAdd (Broadcast.consL (Broadcast.consR Broadcast.nil)) (Op.ref TileDType.ptr [1, OF] "Weight")
            (Op.expandDim ⟨1, by simp⟩
              (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat WIFS)
                (Op.ref TileDType.nat [BIN] "input_feat_offset"))))
          (Op.mul NumericDType.nat Broadcast.nil (Op.constNat WHS) (Op.ref TileDType.nat [] "h")))
        (Op.mul NumericDType.nat Broadcast.nil (Op.constNat WWS) (Op.ref TileDType.nat [] "w"))),
    Stmt.assign .bool [BHW, BIN] "input_mask"
      (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
              (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "batch_offset"))
                  (Op.constNat batch_dim))
                (Op.lt ComparableDType.nat Broadcast.scalarR
                  (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BIN] "input_feat_offset"))
                  (Op.ref TileDType.nat [] "in_group_dim")))
              (Op.le ComparableDType.int Broadcast.scalarL (Op.constInt (Int.ofNat 0))
                (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_height_offset"))))
            (Op.lt ComparableDType.int Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_height_offset"))
              (Op.castNatToInt (Op.constNat in_height))))
          (Op.le ComparableDType.int Broadcast.scalarL (Op.constInt (Int.ofNat 0))
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_width_offset"))))
        (Op.lt ComparableDType.int Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.int [BHW] "input_width_offset"))
          (Op.castNatToInt (Op.constNat in_width)))),
    Stmt.assign .bool [BIN, OF] "weight_mask"
      (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BIN] "input_feat_offset"))
          (Op.ref TileDType.nat [] "in_group_dim"))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [OF] "output_feat_offset"))
          (Op.ref TileDType.nat [] "out_group_dim"))),
    Stmt.assign .real [BHW, BIN] "input_block"
      (Op.load .real (MemAccess.ptr (Op.ref TileDType.ptr [BHW, BIN] "curr_input_pointer"))
        (MaskOpt.mask (Op.ref TileDType.bool [BHW, BIN] "input_mask"))),
    Stmt.assign .real [BIN, OF] "weight_block"
      (Op.load .real (MemAccess.ptr (Op.ref TileDType.ptr [BIN, OF] "curr_weight_pointer"))
        (MaskOpt.mask (Op.ref TileDType.bool [BIN, OF] "weight_mask"))),
    Stmt.ifThen (Op.constBool Bool.true)
      [Stmt.assign .fp16 [BHW, BIN] "input_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BHW, BIN] "input_block")),
        Stmt.assign .fp16 [BIN, OF] "weight_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BIN, OF] "weight_block"))],
    Stmt.assign .real [BHW, OF] "accum"
      (Op.add NumericDType.real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref TileDType.real [BHW, OF] "accum")
        (Op.dot (batch := [])
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BHW, BIN] "input_block"))
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BIN, OF] "weight_block")))) ]

/-- Post-loop tail: `Output +=`, `output_mask`, masked store (3 statements). -/
def convPostBody (Input Weight Output : RegionName)
    (batch_dim out_height out_width OBS OOFS OHS OWS OGD BHW OF : Nat) : List Stmt :=
  [ Stmt.assign .ptr [BHW, OF] "Output"
      (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Output)
        (Op.add NumericDType.nat (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.add NumericDType.nat (Broadcast.consSame (Broadcast.consR Broadcast.nil))
            (Op.add NumericDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
              (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OBS)
                (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "batch_offset")))
              (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OOFS)
                (Op.expandDim ⟨0, by simp⟩
                  (Op.add NumericDType.nat Broadcast.scalarL
                    (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "group_pid")
                      (Op.ref TileDType.nat [] "out_group_dim"))
                    (Op.ref TileDType.nat [OF] "output_feat_offset")))))
            (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OHS)
              (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_height_offset"))))
          (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OWS)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_width_offset"))))),
    Stmt.assign .bool [BHW, OF] "output_mask"
      (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "batch_offset")) (Op.constNat batch_dim))
            (Op.lt ComparableDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [OF] "output_feat_offset"))
              (Op.ref TileDType.nat [] "out_group_dim")))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_height_offset"))
            (Op.constNat out_height)))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_width_offset"))
          (Op.constNat out_width))),
    Stmt.store .real [BHW, OF] (MemAccess.ptr (Op.ref TileDType.ptr [BHW, OF] "Output"))
      (Op.ref TileDType.real [BHW, OF] "accum") (MaskOpt.mask (Op.ref TileDType.bool [BHW, OF] "output_mask")) ]

/-- **Body decomposition** (locks `convInnerBody`/`convPostBody` to the kernel). -/
theorem conv2d_body_split
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF : Nat) :
    (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgKernel.body
      = (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
          out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
          KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgKernel.body.take 14
        ++ (Stmt.forRange "h" 0 KH 1
              [Stmt.forRange "w" 0 KW 1
                [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
                  (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]]
            :: convPostBody Input Weight Output batch_dim out_height out_width OBS OOFS OHS OWS
                (in_feat_dim / groups) BHW OF) := by
  rfl

/-- The seeded-register context shared by all three loops: program ids and
`mem`/`undef` are fixed at `s0`, and the prefix-derived registers hold their
closed forms. Bundling these keeps the per-loop invariants readable. -/
structure ConvCtx (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim out_height out_width
      IBS IIFS WOFS OH OW IGD OGD : Nat) (s : BlockState) : Prop where
  pids : s.pids = s0.pids
  undef : ∀ rg o, s.undef rg o = 0
  mem : s.mem = s0.mem
  group_pid : s.regs .nat [] "group_pid" = some (Tile.scalar (s0.pids 2))
  in_group_dim : s.regs .nat [] "in_group_dim" = some (Tile.scalar IGD)
  out_group_dim : s.regs .nat [] "out_group_dim" = some (Tile.scalar OGD)
  batch_offset : s.regs .nat [BHW] "batch_offset" =
    some (Tile.vec (fun i : Fin BHW => batchIdx s0 OH OW BHW i))
  output_height_offset : s.regs .nat [BHW] "output_height_offset" =
    some (Tile.vec (fun i : Fin BHW => heightIdx s0 OH OW BHW i))
  output_width_offset : s.regs .nat [BHW] "output_width_offset" =
    some (Tile.vec (fun i : Fin BHW => widthIdx s0 OW BHW i))
  output_feat_offset : s.regs .nat [OF] "output_feat_offset" =
    some (Tile.vec (fun j : Fin OF => featIdx s0 OF j))
  inputReg : s.regs .ptr [BHW, 1] "Input" =
    some ⟨fun idx : TileIndex [BHW, 1] => (Input.cast, inputBase s0 IBS IIFS OH OW BHW IGD idx.1)⟩
  weightReg : s.regs .ptr [1, OF] "Weight" =
    some ⟨fun idx : TileIndex [1, OF] => (Weight.cast, weightBase s0 WOFS OF OGD idx.2.1)⟩

/-- The fp16 round-trip on a loaded `.real` tile is the identity: casting the
tile to `.fp16` then reading it back through `castFloat .fp16 .real` recovers the
original cell values (the in-model fp16 cast is the identity). -/
theorem fp16_roundtrip_eval {shape : TileShape} (regName : RegName) (s : BlockState) (t : Tile .real shape)
    (hreg : s.regs .fp16 shape regName =
      some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (t.data i)⟩ : Tile .fp16 shape)) :
    evalOp (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 shape regName)) s
      = some t := by
  rw [evalOp_castFloat, Option.bind_eq_bind]
  simp only [FloatDType.toTileDType_fp16]
  have href : evalOp (Op.ref TileDType.fp16 shape regName) s
      = some (⟨fun i => FloatDType.real.cast FloatDType.fp16 (t.data i)⟩ : Tile .fp16 shape) := by
    rw [evalOp_ref]; exact hreg
  conv_lhs => arg 1; rw [href]
  refine congrArg some ?_
  ext i
  show FloatDType.fp16.cast FloatDType.real (FloatDType.real.cast FloatDType.fp16 (t.data i)) = t.data i
  simp only [FloatDType.cast, FloatDType.real_toWithBot, FloatDType.fp16_ofWithBot,
    FloatDType.fp16_toWithBot, FloatDType.real_ofWithBot]

/-- The innermost-loop invariant: `ConvCtx` holds, `h`/`w` are pinned to the
current outer iteration values, and `accum` equals `accBase` plus the partial
`accC` after `cbCount` blocks (where `cbCount = i / BIN`, the c-loop having
counter `i = cbCount · BIN`). -/
noncomputable def convCInv
    (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW IGD OGD SH SW PH PW : Nat)
    (hc wc : Nat) (accBase : Fin BHW → Fin OF → ℝ)
    (i : Nat) (s : BlockState) : Prop :=
  ConvCtx s0 Input Weight BHW BIN OF batch_dim 0 0 IBS IIFS WOFS OH OW IGD OGD s ∧
  i = i / BIN * BIN ∧ i ≤ IGD ∧
  s.regs .nat [] "h" = some (Tile.scalar hc) ∧
  s.regs .nat [] "w" = some (Tile.scalar wc) ∧
  s.regs .real [BHW, OF] "accum" = some ⟨fun idx : TileIndex [BHW, OF] =>
    some (accBase idx.1 idx.2.1
      + accC s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW hc wc (i / BIN) idx.1 idx.2.1)⟩

/-- `readMem` is insensitive to a `Region.cast` on the region name. -/
theorem readMem_cast (s : BlockState) (R : RegionName) (o : Nat) :
    s.readMem (Region.cast (d := .real) (d' := .real) R) o = s.readMem R o := rfl

set_option maxHeartbeats 2000000 in
/-- **Inner-loop step**: one c-block iteration advances `convCInv` by one block.
The masked loads produce the `miVal`/`mwVal` padded values, the fp16 round-trip
is the identity, and `accum += tl.dot` adds `blockDot` (= `accC`'s next term). -/
theorem conv2d_c_step
    (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW IGD OGD SH SW PH PW : Nat)
    (hBIN : 0 < BIN) (hdvd : BIN ∣ IGD) (hc wc : Nat) (accBase : Fin BHW → Fin OF → ℝ)
    (i : Nat) (s : BlockState) (hilt : i < IGD)
    (hinv : convCInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW IGD OGD SH SW PH PW hc wc accBase i s) :
    ∃ s', stepStmts (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)
        (s.setReg "c" .nat [] (Tile.scalar i)) = some s'
      ∧ convCInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW IGD OGD SH SW PH PW hc wc accBase (i + BIN) s' := by
  obtain ⟨hctx, hi, hile, hh, hw, hacc⟩ := hinv
  -- i + BIN ≤ IGD from i < IGD, i aligned, and BIN | IGD
  have hiBINle : i + BIN ≤ IGD := by
    obtain ⟨q, hq⟩ := hdvd
    have hlt : i / BIN < q := by
      have h1 : i / BIN * BIN < BIN * q := by rw [← hi]; rw [← hq]; exact hilt
      rw [Nat.mul_comm] at h1; exact Nat.lt_of_mul_lt_mul_left h1
    have h2 : i + BIN = (i / BIN + 1) * BIN := by rw [Nat.succ_mul, ← hi]
    have h3 : (i / BIN + 1) * BIN ≤ q * BIN := Nat.mul_le_mul_right _ hlt
    rw [hq, Nat.mul_comm]; omega
  obtain ⟨hpids, hundef, hmem, hgp, hgd, hod, hbo, hoh, how, hof, hIreg, hWreg⟩ := hctx
  -- the loop-set state and its register readbacks (all ConvCtx regs survive the `c` set)
  set sk := s.setReg "c" .nat [] (Tile.scalar i) with hsk
  have hck : sk.regs .nat [] "c" = some (Tile.scalar i) := by simp [hsk]
  have hrmem : ∀ (R : RegionName) (o : Nat), sk.readMem R o = s0.readMem R o := by
    intro R o; simp only [hsk, BlockState.setReg_readMem]; unfold BlockState.readMem; rw [hmem]
  -- abbreviations
  set ihoff : Fin BHW → Int := fun ii => inHeightOff s0 OH OW BHW SH PH hc ii with hihoff
  set iwoff : Fin BHW → Int := fun ii => inWidthOff s0 OW BHW SW PW wc ii with hiwoff
  unfold convInnerBody
  -- stmt 0: input_feat_offset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (input_feat_offset_eval sk BIN i (by simp [hsk])))]
  set s1 := sk.setReg "input_feat_offset" .nat [BIN] (Tile.vec (fun e : Fin BIN => i + e.val)) with hs1
  -- stmt 1: input_height_offset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (int_offset_eval s1 BHW PH SH (fun ii => heightIdx s0 OH OW BHW ii) hc
          "output_height_offset" "h" (by simp [hs1, hsk, hh]) (by simp [hs1, hsk, hoh])))]
  set s2 := s1.setReg "input_height_offset" .int [BHW]
    (Tile.vec (fun ii : Fin BHW => (hc : Int) - (PH : Int) + (SH : Int) * (heightIdx s0 OH OW BHW ii : Int))) with hs2
  -- stmt 2: input_width_offset
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (int_offset_eval s2 BHW PW SW (fun ii => widthIdx s0 OW BHW ii) wc
          "output_width_offset" "w" (by simp [hs2, hs1, hsk, hw]) (by simp [hs2, hs1, hsk, how])))]
  set s3 := s2.setReg "input_width_offset" .int [BHW]
    (Tile.vec (fun ii : Fin BHW => (wc : Int) - (PW : Int) + (SW : Int) * (widthIdx s0 OW BHW ii : Int))) with hs3
  -- stmt 3: curr_input_pointer
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (curr_input_pointer_eval BHW BIN Input
          (fun ii => inputBase s0 IBS IIFS OH OW BHW IGD ii) i IIFS IHS IWS
          (fun ii => inHeightOff s0 OH OW BHW SH PH hc ii)
          (fun ii => inWidthOff s0 OW BHW SW PW wc ii) s3
          (by simp [hs3, hs2, hs1, hsk, hIreg])
          (by simp [hs3, hs2, hs1])
          (by simp only [hs3, hs2, BlockState.setReg_ne_name, BlockState.setReg_same,
                ne_eq, String.reduceEq, not_false_eq_true]; rfl)
          (by simp only [hs3, BlockState.setReg_same]; rfl)))]
  set cipT : Tile .ptr [BHW, BIN] := ⟨fun idx : TileIndex [BHW, BIN] =>
    (Input.cast, inputBase s0 IBS IIFS OH OW BHW IGD idx.1 + IIFS * (i + idx.2.1.val)
      + ((IHS : Int) * inHeightOff s0 OH OW BHW SH PH hc idx.1).toNat
      + ((IWS : Int) * inWidthOff s0 OW BHW SW PW wc idx.1).toNat)⟩ with hcipT
  set s4 := s3.setReg "curr_input_pointer" .ptr [BHW, BIN] cipT with hs4
  -- stmt 4: curr_weight_pointer
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (curr_weight_pointer_eval BIN OF WIFS WHS WWS Weight
          (fun jj => weightBase s0 WOFS OF OGD jj) i hc wc s4
          (by simp [hs4, hs3, hs2, hs1, hsk, hWreg])
          (by simp [hs4, hs3, hs2, hs1])
          (by simp [hs4, hs3, hs2, hs1, hsk, hh])
          (by simp [hs4, hs3, hs2, hs1, hsk, hw])))]
  set cwpT : Tile .ptr [BIN, OF] := ⟨fun idx : TileIndex [BIN, OF] =>
    (Weight.cast, weightBase s0 WOFS OF OGD idx.2.1 + WIFS * (i + idx.1.val) + WHS * hc + WWS * wc)⟩ with hcwpT
  set s5 := s4.setReg "curr_weight_pointer" .ptr [BIN, OF] cwpT with hs5
  -- stmt 5: input_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (input_mask_eval BHW BIN batch_dim in_height in_width IGD s5
          (fun ii => batchIdx s0 OH OW BHW ii) i
          (fun ii => inHeightOff s0 OH OW BHW SH PH hc ii)
          (fun ii => inWidthOff s0 OW BHW SW PW wc ii)
          (by simp [hs5, hs4, hs3, hs2, hs1, hsk, hbo])
          (by simp [hs5, hs4, hs3, hs2, hs1])
          (by simp only [hs5, hs4, hs3, hs2, BlockState.setReg_ne_name, BlockState.setReg_same,
                ne_eq, String.reduceEq, not_false_eq_true]; rfl)
          (by simp only [hs5, hs4, hs3, BlockState.setReg_ne_name, BlockState.setReg_same,
                ne_eq, String.reduceEq, not_false_eq_true]; rfl)
          (by simp [hs5, hs4, hs3, hs2, hs1, hsk, hgd])))]
  set imT : Tile .bool [BHW, BIN] := ⟨fun idx : TileIndex [BHW, BIN] =>
    decide (((((batchIdx s0 OH OW BHW idx.1 < batch_dim ∧ i + idx.2.1.val < IGD) ∧
      (0:Int) ≤ inHeightOff s0 OH OW BHW SH PH hc idx.1) ∧ inHeightOff s0 OH OW BHW SH PH hc idx.1 < (in_height:Int)) ∧
      (0:Int) ≤ inWidthOff s0 OW BHW SW PW wc idx.1) ∧ inWidthOff s0 OW BHW SW PW wc idx.1 < (in_width:Int))⟩ with himT
  set s6 := s5.setReg "input_mask" .bool [BHW, BIN] imT with hs6
  -- stmt 6: weight_mask
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (weight_mask_eval BIN OF IGD OGD s6 i (fun jj => featIdx s0 OF jj)
          (by simp [hs6, hs5, hs4, hs3, hs2, hs1])
          (by simp [hs6, hs5, hs4, hs3, hs2, hs1, hsk, hof])
          (by simp [hs6, hs5, hs4, hs3, hs2, hs1, hsk, hgd])
          (by simp [hs6, hs5, hs4, hs3, hs2, hs1, hsk, hod])))]
  set wmT : Tile .bool [BIN, OF] := ⟨fun idx : TileIndex [BIN, OF] =>
    decide (i + idx.1.val < IGD ∧ featIdx s0 OF idx.2.1 < OGD)⟩ with hwmT
  set s7 := s6.setReg "weight_mask" .bool [BIN, OF] wmT with hs7
  -- undef preserved through s7
  have hundef7 : ∀ rg o, s7.undef rg o = 0 := by
    intro rg o; simp only [hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, BlockState.setReg]; exact hundef rg o
  -- stmt 7: input_block masked load → miVal tile
  set inBlock : Tile .real [BHW, BIN] :=
    ⟨fun idx => if imT.data idx then some (s7.readMem (cipT.data idx).1 (cipT.data idx).2) else some 0⟩ with hinBlock
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_mask_clean (Op.ref TileDType.ptr [BHW, BIN] "curr_input_pointer")
          (Op.ref TileDType.bool [BHW, BIN] "input_mask") s7 cipT imT
          (by rw [evalOp_ref]; simp [hs7, hs6, hs5, hs4])
          (by rw [evalOp_ref]; simp [hs7, hs6])
          hundef7))]
  set s8 := s7.setReg "input_block" .real [BHW, BIN] inBlock with hs8
  have hundef8 : ∀ rg o, s8.undef rg o = 0 := by
    intro rg o; simp only [hs8, BlockState.setReg]; exact hundef7 rg o
  -- stmt 8: weight_block masked load → mwVal tile
  set wBlock : Tile .real [BIN, OF] :=
    ⟨fun idx => if wmT.data idx then some (s8.readMem (cwpT.data idx).1 (cwpT.data idx).2) else some 0⟩ with hwBlock
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (load_ptr_mask_clean (Op.ref TileDType.ptr [BIN, OF] "curr_weight_pointer")
          (Op.ref TileDType.bool [BIN, OF] "weight_mask") s8 cwpT wmT
          (by rw [evalOp_ref]; simp [hs8, hs7, hs6, hs5])
          (by rw [evalOp_ref]; simp [hs8, hs7, hs6])
          hundef8))]
  set s9 := s8.setReg "weight_block" .real [BIN, OF] wBlock with hs9
  -- stmt 9: ifThen (constBool true) [cast input_block→fp16, cast weight_block→fp16]
  have hif : stepStmt (Stmt.ifThen (Op.constBool Bool.true)
      [Stmt.assign .fp16 [BHW, BIN] "input_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BHW, BIN] "input_block")),
        Stmt.assign .fp16 [BIN, OF] "weight_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BIN, OF] "weight_block"))]) s9
      = stepStmts [Stmt.assign .fp16 [BHW, BIN] "input_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BHW, BIN] "input_block")),
        Stmt.assign .fp16 [BIN, OF] "weight_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BIN, OF] "weight_block"))] s9 := by
    simp only [stepStmt, evalOp]
    rfl
  -- evaluate the two fp16 casts
  have hcastI : evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BHW, BIN] "input_block")) s9
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (inBlock.data idx)⟩ : Tile .fp16 [BHW, BIN]) := by
    rw [evalOp_castFloat, Option.bind_eq_bind]
    simp only [FloatDType.toTileDType_real]
    have hrefI : evalOp (Op.ref TileDType.real [BHW, BIN] "input_block") s9 = some inBlock := by
      rw [evalOp_ref]; simp [hs9, hs8]
    conv_lhs => arg 1; rw [hrefI]
    rfl
  set inFp : Tile .fp16 [BHW, BIN] := ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (inBlock.data idx)⟩ with hinFp
  set s10 := s9.setReg "input_block" .fp16 [BHW, BIN] inFp with hs10
  have hcastW : evalOp (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BIN, OF] "weight_block")) s10
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (wBlock.data idx)⟩ : Tile .fp16 [BIN, OF]) := by
    rw [evalOp_castFloat, Option.bind_eq_bind]
    simp only [FloatDType.toTileDType_real]
    have hrefW : evalOp (Op.ref TileDType.real [BIN, OF] "weight_block") s10 = some wBlock := by
      rw [evalOp_ref]; simp [hs10, hs9]
    conv_lhs => arg 1; rw [hrefW]
    rfl
  set wFp : Tile .fp16 [BIN, OF] := ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (wBlock.data idx)⟩ with hwFp
  set s11 := s10.setReg "weight_block" .fp16 [BIN, OF] wFp with hs11
  -- the ifThen body runs both casts, ending at s11
  have hifbody : stepStmts [Stmt.assign .fp16 [BHW, BIN] "input_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BHW, BIN] "input_block")),
        Stmt.assign .fp16 [BIN, OF] "weight_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BIN, OF] "weight_block"))] s9
      = some s11 := by
    rw [stepStmts.cons_some (stepStmt_assign_eq_some (dtype := TileDType.fp16) hcastI),
        stepStmts.cons_some (stepStmt_assign_eq_some (dtype := TileDType.fp16) hcastW), stepStmts.nil]
  have hif' : stepStmt (Stmt.ifThen (Op.constBool Bool.true)
      [Stmt.assign .fp16 [BHW, BIN] "input_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BHW, BIN] "input_block")),
        Stmt.assign .fp16 [BIN, OF] "weight_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BIN, OF] "weight_block"))]) s9
      = some s11 := by rw [hif, hifbody]
  rw [stepStmts.cons_some hif']
  -- stmt 10: accum += tl.dot(fp16→real input_block, fp16→real weight_block)
  -- the two dot operands recover the real loaded tiles (fp16 round-trip = id)
  have hdotI : evalOp (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BHW, BIN] "input_block")) s11
      = some inBlock :=
    fp16_roundtrip_eval "input_block" s11 inBlock (by simp [hs11, hs10, hinFp])
  have hdotW : evalOp (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BIN, OF] "weight_block")) s11
      = some wBlock :=
    fp16_roundtrip_eval "weight_block" s11 wBlock (by simp [hs11, hwFp])
  -- the dot evaluates to Tile.dot inBlock wBlock
  have hdot : evalOp (Op.dot (batch := [])
      (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BHW, BIN] "input_block"))
      (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BIN, OF] "weight_block"))) s11
      = some (Tile.dot [] inBlock wBlock) := by
    have key : ∀ (a : Option (Tile .real [BHW, BIN])) (b : Option (Tile .real [BIN, OF])),
        a = some inBlock → b = some wBlock →
        (a.bind (fun vx => b.bind (fun vy => some (Tile.dot [] vx vy)))) = some (Tile.dot [] inBlock wBlock) := by
      rintro a b rfl rfl; rfl
    rw [evalOp_dot]
    exact key _ _ hdotI hdotW
  -- accum (held in s11) is the accBase + accC tile
  have haccS11 : s11.regs .real [BHW, OF] "accum" = some ⟨fun idx : TileIndex [BHW, OF] =>
      some (accBase idx.1 idx.2.1 + accC s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW hc wc (i / BIN) idx.1 idx.2.1)⟩ := by
    simp [hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hacc]
  -- accum + dot eval
  have haccdot : evalOp (Op.add NumericDType.real (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref TileDType.real [BHW, OF] "accum")
        (Op.dot (batch := [])
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BHW, BIN] "input_block"))
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BIN, OF] "weight_block")))) s11
      = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          ⟨fun idx : TileIndex [BHW, OF] => some (accBase idx.1 idx.2.1 + accC s0 Input Weight BHW BIN OF batch_dim
            in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW hc wc (i / BIN) idx.1 idx.2.1)⟩
          (Tile.dot [] inBlock wBlock)) := by
    rw [evalOp_add]
    set zt : Tile .real [BHW, OF] := ⟨fun idx : TileIndex [BHW, OF] => some (accBase idx.1 idx.2.1 +
      accC s0 Input Weight BHW BIN OF batch_dim in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS
        OH OW IGD OGD SH SW PH PW hc wc (i / BIN) idx.1 idx.2.1)⟩ with hzt
    have hrefacc : evalOp (Op.ref TileDType.real [BHW, OF] "accum") s11 = some zt := by
      rw [evalOp_ref]; exact haccS11
    have key : ∀ (a : Option (Tile .real [BHW, OF])) (b : Option (Tile .real [BHW, OF])),
        a = some zt → b = some (Tile.dot [] inBlock wBlock) →
        (a.bind (fun vz => b.bind (fun vd => some
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) vz vd))))
        = some (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
            zt (Tile.dot [] inBlock wBlock)) := by
      rintro a b rfl rfl; rfl
    exact key _ _ hrefacc hdot
  rw [stepStmts.cons_some (stepStmt_assign_eq_some haccdot), stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  set s12 := s11.setReg "accum" .real [BHW, OF]
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      ⟨fun idx : TileIndex [BHW, OF] => some (accBase idx.1 idx.2.1 + accC s0 Input Weight BHW BIN OF batch_dim
        in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW hc wc (i / BIN) idx.1 idx.2.1)⟩
      (Tile.dot [] inBlock wBlock)) with hs12
  -- mem and undef preserved through the whole inner body
  have hmem12 : s12.mem = s0.mem := by
    simp only [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, BlockState.setReg]
    exact hmem
  have hundef12 : ∀ rg o, s12.undef rg o = 0 := by
    intro rg o
    simp only [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, BlockState.setReg]
    exact hundef rg o
  have hrmem12 : ∀ (R : RegionName) (o : Nat), s12.readMem R o = s0.readMem R o := by
    intro R o; unfold BlockState.readMem; rw [hmem12]
  -- readMem on the intermediate states equals s0 (all are pure register updates)
  have hrmem7 : ∀ (R : RegionName) (o : Nat), s7.readMem R o = s0.readMem R o := by
    intro R o
    simp only [hs7, hs6, hs5, hs4, hs3, hs2, hs1, BlockState.setReg_readMem]; exact hrmem R o
  have hrmem8 : ∀ (R : RegionName) (o : Nat), s8.readMem R o = s0.readMem R o := by
    intro R o; rw [hs8, BlockState.setReg_readMem]; exact hrmem7 R o
  -- the loaded tiles equal the masked spec values
  have hinB : ∀ (ii : Fin BHW) (e : Fin BIN), inBlock.data (ii, e, PUnit.unit)
      = some (miVal s0 Input BHW BIN batch_dim in_height in_width IBS IIFS IHS IWS OH OW IGD SH SW PH PW
          hc wc i ii e) := by
    intro ii e
    simp only [hinBlock, himT, hcipT, miVal, inMask, inAddr, readMem_cast]
    by_cases hm : ((((batchIdx s0 OH OW BHW ii < batch_dim ∧ i + e.val < IGD) ∧
        (0:Int) ≤ inHeightOff s0 OH OW BHW SH PH hc ii) ∧ inHeightOff s0 OH OW BHW SH PH hc ii < (in_height:Int)) ∧
        (0:Int) ≤ inWidthOff s0 OW BHW SW PW wc ii) ∧ inWidthOff s0 OW BHW SW PW wc ii < (in_width:Int)
    · rw [if_pos (by simp [hm]), if_pos hm, hrmem7]
    · rw [if_neg (by simp [hm]), if_neg hm]
  have hwB : ∀ (e : Fin BIN) (j : Fin OF), wBlock.data (e, j, PUnit.unit)
      = some (mwVal s0 Weight BIN OF IGD OGD WOFS WIFS WHS WWS hc wc i e j) := by
    intro e j
    simp only [hwBlock, hwmT, hcwpT, mwVal, wMask, wAddr, readMem_cast]
    by_cases hm : i + e.val < IGD ∧ featIdx s0 OF j < OGD
    · rw [if_pos (by simp [hm]), if_pos hm, hrmem8]
    · rw [if_neg (by simp [hm]), if_neg hm]
  -- assemble convCInv
  refine ⟨⟨?_, hundef12, hmem12, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_, ?_, ?_, ?_, ?_⟩
  · -- pids
    simp only [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, BlockState.setReg_pids]
    exact hpids
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hgp]
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hgd]
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hod]
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hbo]
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hoh]
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, how]
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hof]
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hIreg]
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hWreg]
  · -- (i+BIN) = (i+BIN)/BIN * BIN
    have : (i + BIN) / BIN = i / BIN + 1 := by rw [Nat.add_div_right _ hBIN]
    rw [this]; rw [Nat.succ_mul]; omega
  · -- i + BIN ≤ IGD
    exact hiBINle
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hh]
  · simp [hs12, hs11, hs10, hs9, hs8, hs7, hs6, hs5, hs4, hs3, hs2, hs1, hsk, hw]
  · -- accum = accBase + accC((i+BIN)/BIN)
    have hs12acc : s12.regs .real [BHW, OF] "accum" = some
        (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
          ⟨fun idx : TileIndex [BHW, OF] => some (accBase idx.1 idx.2.1 + accC s0 Input Weight BHW BIN OF batch_dim
            in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW hc wc (i / BIN) idx.1 idx.2.1)⟩
          (Tile.dot [] inBlock wBlock)) := by rw [hs12]; exact BlockState.setReg_same _ _ _ _ _
    rw [hs12acc]
    refine congrArg some ?_
    ext idx
    have hcc : (i + BIN) / BIN = i / BIN + 1 := by rw [Nat.add_div_right _ hBIN]
    rw [hcc]
    -- the dot lane equals blockDot
    have hdotlane : (Tile.dot [] inBlock wBlock).data (idx.1, idx.2.1, PUnit.unit)
        = some (blockDot s0 Input Weight BHW BIN OF batch_dim in_height in_width
            IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW hc wc ((i / BIN) * BIN) idx.1 idx.2.1) := by
      rw [dot_mask BHW BIN OF inBlock wBlock idx.1 idx.2.1 _ _ (fun e => hinB idx.1 e) (fun e => hwB e idx.2.1)]
      unfold blockDot
      rw [show i / BIN * BIN = i from hi.symm]
    rw [accadd_eval BHW OF _ (Tile.dot [] inBlock wBlock) idx.1 idx.2.1
        (accBase idx.1 idx.2.1 + accC s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW hc wc (i / BIN) idx.1 idx.2.1)
        (blockDot s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW hc wc ((i / BIN) * BIN) idx.1 idx.2.1)
        rfl hdotlane]
    show _ = some (accBase idx.1 idx.2.1 + accC s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW hc wc (i / BIN + 1) idx.1 idx.2.1)
    rw [accC_succ s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW hc wc (i / BIN) idx.1 idx.2.1]
    ring_nf

set_option maxHeartbeats 1000000 in
/-- **Inner c-loop driver**: the dynamic `forRangeDyn "c" 0 in_group_dim BIN`
runs the full `numCBlocks`-block accumulation, advancing `convCInv` from
`cbCount = 0` to `cbCount = numCBlocks` (with `IGD = numCBlocks · BIN`). -/
theorem conv2d_c_loop
    (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW numCBlocks : Nat)
    (hBIN : 0 < BIN) (hc wc : Nat) (accBase : Fin BHW → Fin OF → ℝ)
    (s : BlockState)
    (hP0 : convCInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW hc wc accBase 0 s) :
    ∃ s', stepStmt (Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
        (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)) s = some s'
      ∧ convCInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW hc wc accBase
          (BIN * numCBlocks) s' := by
  set IGD := BIN * numCBlocks with hIGD
  -- the dynamic stop resolves to IGD via in_group_dim
  have hgd : s.regs .nat [] "in_group_dim" = some (Tile.scalar IGD) := hP0.1.in_group_dim
  have hresolve : stepStmt (Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
      (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)) s
      = stepForRangeAux "c" 0 IGD BIN
          (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW) s := by
    rw [stepForRangeAux.forRangeDyn_unfold]
    simp only [evalOp_constNat, evalOp_ref, hgd, Tile.scalar, Option.bind_some, bind]
  rw [hresolve]
  obtain ⟨final, s_final, haux, hfinal, hPfinal⟩ :=
    forRangeAux_inv (idx := "c") (stop := IGD) (step := BIN)
      (P := convCInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW IGD OGD SH SW PH PW hc wc accBase)
      (by omega)
      (fun ii st hlt hinv => conv2d_c_step s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW IGD OGD SH SW PH PW hBIN ⟨numCBlocks, hIGD⟩ hc wc accBase ii st hlt hinv)
      0 s hP0
  -- the loop exits exactly at IGD (BIN-aligned, bounded by IGD)
  have hfinalEq : final = IGD := by
    obtain ⟨_, _, hile, _⟩ := hPfinal
    have : IGD ≤ final := hfinal
    omega
  subst hfinalEq
  exact ⟨s_final, haux, hPfinal⟩

/-- The middle kernel-width loop invariant: `ConvCtx` holds, `h` is pinned, and
`accum` equals `accBaseH` plus the partial `accW` after `wCount` width
iterations (each running a full c-loop). -/
noncomputable def convWInv
    (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW IGD OGD SH SW PH PW numCBlocks : Nat)
    (hc : Nat) (accBaseH : Fin BHW → Fin OF → ℝ)
    (wCount : Nat) (s : BlockState) : Prop :=
  ConvCtx s0 Input Weight BHW BIN OF batch_dim 0 0 IBS IIFS WOFS OH OW IGD OGD s ∧
  s.regs .nat [] "h" = some (Tile.scalar hc) ∧
  s.regs .real [BHW, OF] "accum" = some ⟨fun idx : TileIndex [BHW, OF] =>
    some (accBaseH idx.1 idx.2.1
      + accW s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW numCBlocks hc wCount idx.1 idx.2.1)⟩

set_option maxHeartbeats 1000000 in
/-- **Width-loop step**: one kernel-width iteration runs a full c-loop, advancing
`convWInv` by one width block (`accW … (wCount+1) = accW … wCount + accC …`). -/
theorem conv2d_w_step
    (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW numCBlocks : Nat)
    (hBIN : 0 < BIN) (hc : Nat) (accBaseH : Fin BHW → Fin OF → ℝ)
    (wCount : Nat) (s : BlockState)
    (hinv : convWInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW numCBlocks
        hc accBaseH wCount s) :
    ∃ s', stepStmts [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
        (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]
        (s.setReg "w" .nat [] (Tile.scalar wCount)) = some s'
      ∧ convWInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW numCBlocks
          hc accBaseH (wCount + 1) s' := by
  obtain ⟨hctx, hh, hacc⟩ := hinv
  -- the accumulator base for this width iteration
  set accBaseW : Fin BHW → Fin OF → ℝ := fun ii j => accBaseH ii j +
    accW s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW (BIN * numCBlocks) OGD SH SW PH PW numCBlocks hc wCount ii j with haccBaseW
  set sw := s.setReg "w" .nat [] (Tile.scalar wCount) with hsw
  -- convCInv at cbCount = 0 holds on sw, with hc and wCount pinned
  have hP0 : convCInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW hc wCount accBaseW 0 sw := by
    obtain ⟨hpids, hundef, hmem, hgp, hgd, hod, hbo, hoh, how, hof, hIreg, hWreg⟩ := hctx
    refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, by simp, by omega, ?_, by simp [hsw], ?_⟩
    · simp [hsw, BlockState.setReg_pids, hpids]
    · intro rg o; simp [hsw, BlockState.setReg]; exact hundef rg o
    · simp [hsw, BlockState.setReg]; exact hmem
    · simp [hsw, hgp]
    · simp [hsw, hgd]
    · simp [hsw, hod]
    · simp [hsw, hbo]
    · simp [hsw, hoh]
    · simp [hsw, how]
    · simp [hsw, hof]
    · simp [hsw, hIreg]
    · simp [hsw, hWreg]
    · simp [hsw, hh]
    · -- accum at cbCount 0 = accBaseW (accC 0 = 0)
      rw [show sw.regs .real [BHW, OF] "accum" = s.regs .real [BHW, OF] "accum" from by simp [hsw], hacc]
      refine congrArg some ?_
      ext idx
      simp only [haccBaseW, accC, Nat.zero_div, Finset.range_zero, Finset.sum_empty, add_zero]
  obtain ⟨s', hstep, hPfin⟩ := conv2d_c_loop s0 Input Weight BHW BIN OF batch_dim in_height in_width
    IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW numCBlocks hBIN hc wCount accBaseW sw hP0
  refine ⟨s', by rw [stepStmts.cons_some hstep, stepStmts.nil], ?_⟩
  obtain ⟨hctx', hieq', hile', hh', hw', hacc'⟩ := hPfin
  refine ⟨hctx', hh', ?_⟩
  rw [hacc']
  refine congrArg some ?_
  ext idx
  simp only [haccBaseW]
  rw [show BIN * numCBlocks / BIN = numCBlocks from by rw [Nat.mul_div_cancel_left _ hBIN]]
  rw [accW_succ s0 Input Weight BHW BIN OF batch_dim in_height in_width
    IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW (BIN * numCBlocks) OGD SH SW PH PW numCBlocks hc wCount idx.1 idx.2.1]
  ring_nf

/-- For a unit-step static `forRange [0, stop)`, the loop runs exactly `stop`
iterations and exits with the invariant at `stop` (no overshoot). -/
theorem forRange_step1_exact {idx : RegName} {stop : Nat} {body : List Stmt}
    {P : Nat → BlockState → Prop} {s_init : BlockState}
    (h_init : P 0 s_init)
    (h_step : ∀ i s, i < stop → P i s →
      ∃ s', stepStmts body (s.setReg idx .nat [] (Tile.scalar i)) = some s' ∧ P (i + 1) s') :
    ∃ s_final, stepStmt (.forRange idx 0 stop 1 body) s_init = some s_final ∧ P stop s_final := by
  obtain ⟨final, s_final, haux, hfin, hP⟩ := forRange_inv (idx := idx) (start := 0) (stop := stop) (step := 1)
    (P := fun i s => P i s ∧ i ≤ stop)
    (by norm_num) ⟨h_init, Nat.zero_le _⟩
    (fun i s hlt hinv => by
      obtain ⟨hPi, hile⟩ := hinv
      obtain ⟨s', hs', hP'⟩ := h_step i s hlt hPi
      exact ⟨s', hs', hP', by omega⟩)
  obtain ⟨hPf, hfle⟩ := hP
  have : final = stop := by omega
  subst this
  exact ⟨s_final, haux, hPf⟩

/-- **Width-loop driver**: `forRange "w" 0 KW 1` runs `KW` width iterations,
advancing `convWInv` from `0` to `KW`. -/
theorem conv2d_w_loop
    (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW numCBlocks KW : Nat)
    (hBIN : 0 < BIN) (hc : Nat) (accBaseH : Fin BHW → Fin OF → ℝ)
    (s : BlockState)
    (hP0 : convWInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW numCBlocks hc accBaseH 0 s) :
    ∃ s', stepStmt (Stmt.forRange "w" 0 KW 1
        [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
          (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]) s = some s'
      ∧ convWInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW numCBlocks hc accBaseH KW s' := by
  exact forRange_step1_exact (idx := "w")
    (P := convWInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW numCBlocks hc accBaseH)
    hP0
    (fun wCount st _ hinv => conv2d_w_step s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW numCBlocks hBIN hc accBaseH wCount st hinv)

/-- The outer kernel-height loop invariant: `ConvCtx` holds and `accum` equals
the partial `accH` after `hCount` height iterations (each a full width loop). -/
noncomputable def convHInv
    (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW IGD OGD SH SW PH PW KW numCBlocks : Nat)
    (hCount : Nat) (s : BlockState) : Prop :=
  ConvCtx s0 Input Weight BHW BIN OF batch_dim 0 0 IBS IIFS WOFS OH OW IGD OGD s ∧
  s.regs .real [BHW, OF] "accum" = some ⟨fun idx : TileIndex [BHW, OF] =>
    some (accH s0 Input Weight BHW BIN OF batch_dim in_height in_width
            IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW KW numCBlocks hCount idx.1 idx.2.1)⟩

set_option maxHeartbeats 1000000 in
/-- **Height-loop step**: one kernel-height iteration runs a full width loop,
advancing `convHInv` by one (`accH … (hCount+1) = accH … hCount + accW …`). -/
theorem conv2d_h_step
    (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW KW numCBlocks : Nat)
    (hBIN : 0 < BIN) (hCount : Nat) (s : BlockState)
    (hinv : convHInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KW numCBlocks hCount s) :
    ∃ s', stepStmts [Stmt.forRange "w" 0 KW 1
        [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
          (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]]
        (s.setReg "h" .nat [] (Tile.scalar hCount)) = some s'
      ∧ convHInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KW numCBlocks (hCount + 1) s' := by
  obtain ⟨hctx, hacc⟩ := hinv
  set accBaseH : Fin BHW → Fin OF → ℝ := fun ii j =>
    accH s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KW numCBlocks hCount ii j with haccBaseH
  set sh := s.setReg "h" .nat [] (Tile.scalar hCount) with hsh
  have hP0 : convWInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW numCBlocks hCount accBaseH 0 sh := by
    obtain ⟨hpids, hundef, hmem, hgp, hgd, hod, hbo, hoh, how, hof, hIreg, hWreg⟩ := hctx
    refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, by simp [hsh], ?_⟩
    · simp [hsh, BlockState.setReg_pids, hpids]
    · intro rg o; simp [hsh, BlockState.setReg]; exact hundef rg o
    · simp [hsh, BlockState.setReg]; exact hmem
    · simp [hsh, hgp]
    · simp [hsh, hgd]
    · simp [hsh, hod]
    · simp [hsh, hbo]
    · simp [hsh, hoh]
    · simp [hsh, how]
    · simp [hsh, hof]
    · simp [hsh, hIreg]
    · simp [hsh, hWreg]
    · rw [show sh.regs .real [BHW, OF] "accum" = s.regs .real [BHW, OF] "accum" from by simp [hsh], hacc]
      refine congrArg some ?_
      ext idx
      simp only [haccBaseH, accW, Finset.range_zero, Finset.sum_empty, add_zero]
  obtain ⟨s', hstep, hPfin⟩ := conv2d_w_loop s0 Input Weight BHW BIN OF batch_dim in_height in_width
    IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW numCBlocks KW hBIN hCount accBaseH sh hP0
  refine ⟨s', by rw [stepStmts.cons_some hstep, stepStmts.nil], ?_⟩
  obtain ⟨hctx', hh', hacc'⟩ := hPfin
  refine ⟨hctx', ?_⟩
  rw [hacc']
  refine congrArg some ?_
  ext idx
  simp only [haccBaseH]
  rw [accH_succ s0 Input Weight BHW BIN OF batch_dim in_height in_width
    IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KW numCBlocks hCount idx.1 idx.2.1]

/-- **Height-loop driver**: `forRange "h" 0 KH 1` runs `KH` height iterations,
advancing `convHInv` from `0` to `KH`. -/
theorem conv2d_h_loop
    (s0 : BlockState) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW KH KW numCBlocks : Nat)
    (hBIN : 0 < BIN) (s : BlockState)
    (hP0 : convHInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KW numCBlocks 0 s) :
    ∃ s', stepStmt (Stmt.forRange "h" 0 KH 1
        [Stmt.forRange "w" 0 KW 1
          [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
            (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]]) s = some s'
      ∧ convHInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KW numCBlocks KH s' := by
  exact forRange_step1_exact (idx := "h")
    (P := convHInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KW numCBlocks)
    hP0
    (fun hCount st _ hinv => conv2d_h_step s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW KW numCBlocks hBIN hCount st hinv)

/-- The output write address for tile lane `(i,j)`. -/
def outputOffset (s0 : BlockState) (BHW OF OH OW OBS OOFS OHS OWS OGD : Nat)
    (idx : TileIndex [BHW, OF]) : Nat :=
  OBS * batchIdx s0 OH OW BHW idx.1 +
    OOFS * (s0.pids 2 * OGD + featIdx s0 OF idx.2.1) +
    OHS * heightIdx s0 OH OW BHW idx.1 + OWS * widthIdx s0 OW BHW idx.1

/-- The output store-mask predicate for tile lane `(i,j)`. -/
def active (s0 : BlockState) (BHW OF batch_dim OH OW OGD : Nat) (idx : TileIndex [BHW, OF]) : Prop :=
  ((batchIdx s0 OH OW BHW idx.1 < batch_dim ∧ featIdx s0 OF idx.2.1 < OGD) ∧
    heightIdx s0 OH OW BHW idx.1 < OH) ∧ widthIdx s0 OW BHW idx.1 < OW

instance (s0 : BlockState) (BHW OF batch_dim OH OW OGD : Nat) (idx : TileIndex [BHW, OF]) :
    Decidable (active s0 BHW OF batch_dim OH OW OGD idx) := by unfold active; infer_instance

set_option maxHeartbeats 2000000 in
/-- **postLoop**: from `convHInv` at `KH` (so `accum = convSpec`), the three
post-statements (`Output +=`, `output_mask`, masked `tl.store`) write the genuine
convolution value at every active output lane and preserve the rest. -/
theorem conv2d_postLoop
    (Input Weight Output : RegionName)
    (s0 : BlockState) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW KH KW numCBlocks : Nat)
    (hOutInj : Function.Injective (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD))
    (st : BlockState)
    (hinv : convHInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KW numCBlocks KH st) :
    ∃ sfin, stepStmts (convPostBody Input Weight Output batch_dim OH OW OBS OOFS OHS OWS OGD BHW OF) st = some sfin
      ∧ ∀ idx : TileIndex [BHW, OF],
          sfin.readMem Output (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD idx)
            = if active s0 BHW OF batch_dim OH OW OGD idx then
                convSpec s0 Input Weight BHW BIN OF batch_dim in_height in_width
                  IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KH KW numCBlocks idx.1 idx.2.1
              else st.readMem Output (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD idx) := by
  obtain ⟨hctx, hacc⟩ := hinv
  obtain ⟨hpids, hundef, hmem, hgp, hgd, hod, hbo, hoh, how, hof, hIreg, hWreg⟩ := hctx
  -- the accumulator holds convSpec
  have haccSpec : st.regs .real [BHW, OF] "accum" = some ⟨fun idx : TileIndex [BHW, OF] =>
      some (convSpec s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KH KW numCBlocks idx.1 idx.2.1)⟩ := by
    rw [hacc]; rfl
  -- abbreviations for the post-body tiles
  set valueFn : TileIndex [BHW, OF] → ℝ := fun idx =>
    convSpec s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW (BIN * numCBlocks) OGD SH SW PH PW KH KW numCBlocks idx.1 idx.2.1 with hvalueFn
  -- step 0: Output += (the output pointer tile)
  set opT : Tile .ptr [BHW, OF] := ⟨fun idx : TileIndex [BHW, OF] =>
    (Output.cast, outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD idx)⟩ with hopT
  have hOutEval : evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Output)
      (Op.add NumericDType.nat (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.add NumericDType.nat (Broadcast.consSame (Broadcast.consR Broadcast.nil))
          (Op.add NumericDType.nat (Broadcast.consR (Broadcast.consL Broadcast.nil))
            (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OBS)
              (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "batch_offset")))
            (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OOFS)
              (Op.expandDim ⟨0, by simp⟩
                (Op.add NumericDType.nat Broadcast.scalarL
                  (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "group_pid")
                    (Op.ref TileDType.nat [] "out_group_dim"))
                  (Op.ref TileDType.nat [OF] "output_feat_offset")))))
          (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OHS)
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_height_offset"))))
        (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat OWS)
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_width_offset"))))) st
      = some opT := by
    simp only [evalOp.eq_def, hbo, hof, hoh, how, hgp, hod, Option.bind, Option.bind_eq_bind]
    refine congrArg some ?_
    ext idx
    · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, TileShape.dropInsertedIndex, hopT]
    · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
        Broadcast.rightIndex, TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
        hopT, outputOffset, batchIdx, bhIdx, bhwIdx, heightIdx, widthIdx, featIdx,
        IntegralDType.nat_floorDiv, IntegralDType.nat_mod]
      ring
  erw [stepStmts.cons_some (stepStmt_assign_eq_some hOutEval)]
  set s1 := st.setReg "Output" .ptr [BHW, OF] opT with hs1
  -- step 1: output_mask
  set mT : Tile .bool [BHW, OF] := ⟨fun idx : TileIndex [BHW, OF] =>
    decide (active s0 BHW OF batch_dim OH OW OGD idx)⟩ with hmT
  have hMaskEval : evalOp (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "batch_offset")) (Op.constNat batch_dim))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [OF] "output_feat_offset"))
            (Op.ref TileDType.nat [] "out_group_dim")))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_height_offset"))
          (Op.constNat OH)))
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_width_offset"))
        (Op.constNat OW))) s1 = some mT := by
    simp only [evalOp.eq_def, hs1, BlockState.setReg_ne_name, ne_eq, String.reduceEq,
      not_false_eq_true, hbo, hof, hoh, how, hod, Option.bind, Option.bind_eq_bind]
    refine congrArg some ?_
    ext idx
    simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, ComparableDType.lt, TileShape.dropInsertedIndex, hmT, active,
      batchIdx, bhIdx, bhwIdx, heightIdx, widthIdx, featIdx, IntegralDType.nat_floorDiv,
      IntegralDType.nat_mod, Bool.decide_and]
    rfl
  rw [stepStmts.cons_some (stepStmt_assign_eq_some hMaskEval)]
  set s2 := s1.setReg "output_mask" .bool [BHW, OF] mT with hs2
  -- step 2: the masked store
  have haccS2 : s2.regs .real [BHW, OF] "accum" = some ⟨fun idx => some (valueFn idx)⟩ := by
    simp only [hs2, hs1, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    rw [haccSpec]
  have hopS2 : s2.regs .ptr [BHW, OF] "Output" = some opT := by
    simp only [hs2, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hs1,
      BlockState.setReg_same]
  have hmS2 : s2.regs .bool [BHW, OF] "output_mask" = some mT := by
    simp only [hs2, BlockState.setReg_same]
  have hstore : stepStmt (Stmt.store .real [BHW, OF] (MemAccess.ptr (Op.ref TileDType.ptr [BHW, OF] "Output"))
      (Op.ref TileDType.real [BHW, OF] "accum") (MaskOpt.mask (Op.ref TileDType.bool [BHW, OF] "output_mask"))) s2
      = some ((TileShape.allIndices [BHW, OF]).foldl
          (fun acc idx => if active s0 BHW OF batch_dim OH OW OGD idx then
            acc.writeMem Output (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD idx) (valueFn idx) else acc) s2) := by
    simp only [stepStmt, evalOp_ref, haccS2, hopS2, hmS2, bind, Option.bind_some]
    refine congrArg some (List.foldl_ext _ _ s2 (fun acc idx _ => ?_))
    by_cases hb : active s0 BHW OF batch_dim OH OW OGD idx
    · simp only [hmT, hb, decide_true, if_true, hopT, BlockState.writeMemTyped_real, Region.cast_id,
        FloatDType.real_storeValue, WithBot.unbotD_some]
    · simp only [hmT, hb, decide_false, if_false, Bool.false_eq_true]
  rw [stepStmts.cons_some hstore, stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  intro idx
  rw [BlockState.scatter_readback_prop_masked_nd (region := Output) (s := s2)
      (offsetFn := outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD)
      (valueFn := valueFn) (P := active s0 BHW OF batch_dim OH OW OGD) hOutInj idx]
  -- s2.readMem Output = st.readMem Output (Output/mask are register-only updates)
  have hreadback : s2.readMem Output (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD idx)
      = st.readMem Output (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD idx) := by
    simp only [hs2, hs1, BlockState.setReg_readMem]
  rw [hreadback]

set_option maxHeartbeats 2000000 in
/-- **preLoop**: from a clean initial state, the 14 prefix statements set up the
`ConvCtx` registers (the flattened BHW index decomposition, group-local
input/weight bases) and initialise `accum = 0`, establishing `convHInv … 0`. -/
theorem conv2d_preLoop
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF numCBlocks : Nat)
    (s : BlockState) (hundef : ∀ rg o, s.undef rg o = 0)
    (hIGD : in_feat_dim / groups = BIN * numCBlocks) :
    ∃ s', stepStmts ((conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgKernel.body.take 14) s = some s'
      ∧ convHInv s Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS out_height out_width (BIN * numCBlocks)
          (out_feat_dim / groups) SH SW PH PW KW numCBlocks 0 s' := by
  rw [show ((conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgKernel.body.take 14)
      = [ Stmt.assign .nat [] "batch_height_width_pid" (Op.programId 0),
          Stmt.assign .nat [] "out_feat_pid" (Op.programId 1),
          Stmt.assign .nat [] "group_pid" (Op.programId 2),
          Stmt.assign .nat [] "in_group_dim"
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat in_feat_dim) (Op.constNat groups)),
          Stmt.assign .nat [] "out_group_dim"
            (Op.floorDiv IntegralDType.nat Broadcast.nil (Op.constNat out_feat_dim) (Op.constNat groups)),
          Stmt.assign .nat [BHW] "batch_height_width_offset"
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "batch_height_width_pid") (Op.constNat BHW))
              (Op.arange BHW)),
          Stmt.assign .nat [BHW] "batch_height_offset"
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BHW] "batch_height_width_offset")
              (Op.constNat out_width)),
          Stmt.assign .nat [BHW] "batch_offset"
            (Op.floorDiv IntegralDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BHW] "batch_height_offset")
              (Op.constNat out_height)),
          Stmt.assign .nat [OF] "output_feat_offset"
            (Op.add NumericDType.nat Broadcast.scalarL
              (Op.mul NumericDType.nat Broadcast.nil (Op.ref TileDType.nat [] "out_feat_pid") (Op.constNat OF)) (Op.arange OF)),
          Stmt.assign .nat [BHW] "output_height_offset"
            (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BHW] "batch_height_offset")
              (Op.constNat out_height)),
          Stmt.assign .nat [BHW] "output_width_offset"
            (Op.mod IntegralDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BHW] "batch_height_width_offset")
              (Op.constNat out_width)),
          Stmt.assign .ptr [BHW, 1] "Input"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Input)
              (Op.expandDim ⟨1, by simp⟩
                (Op.add NumericDType.nat Broadcast.scalarR
                  (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat IBS) (Op.ref TileDType.nat [BHW] "batch_offset"))
                  (Op.mul NumericDType.nat Broadcast.nil
                    (Op.mul NumericDType.nat Broadcast.nil (Op.constNat IIFS) (Op.ref TileDType.nat [] "group_pid"))
                    (Op.ref TileDType.nat [] "in_group_dim"))))),
          Stmt.assign .ptr [1, OF] "Weight"
            (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Weight)
              (Op.expandDim ⟨0, by simp⟩
                (Op.add NumericDType.nat Broadcast.scalarR
                  (Op.mul NumericDType.nat Broadcast.scalarL (Op.constNat WOFS) (Op.ref TileDType.nat [OF] "output_feat_offset"))
                  (Op.mul NumericDType.nat Broadcast.nil
                    (Op.mul NumericDType.nat Broadcast.nil (Op.constNat WOFS) (Op.ref TileDType.nat [] "group_pid"))
                    (Op.ref TileDType.nat [] "out_group_dim"))))),
          Stmt.assign .real [BHW, OF] "accum" (Op.full [BHW, OF] (Op.const 0)) ] from rfl]
  -- thread the 14 statements via their eval lemmas
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by rw [evalOp_programId] : evalOp (Op.programId 0) s = _)),
      stepStmts.cons_some (stepStmt_assign_eq_some (by rw [evalOp_programId] : evalOp (Op.programId 1) _ = _)),
      stepStmts.cons_some (stepStmt_assign_eq_some (by rw [evalOp_programId] : evalOp (Op.programId 2) _ = _))]
  -- after the 3 program ids, group_pid = s.pids 2
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.constNat in_feat_dim) (Op.constNat groups)) _ = some (Tile.scalar (in_feat_dim / groups)) from by
      simp [evalOp_floorDiv, evalOp_constNat, Tile.bop, IntegralDType.floorDiv]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.floorDiv IntegralDType.nat Broadcast.nil
        (Op.constNat out_feat_dim) (Op.constNat groups)) _ = some (Tile.scalar (out_feat_dim / groups)) from by
      simp [evalOp_floorDiv, evalOp_constNat, Tile.bop, IntegralDType.floorDiv]))]
  -- batch_height_width_offset (uses batch_height_width_pid = s.pids 0)
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (pidblock_eval _ BHW (s.pids 0) "batch_height_width_pid" (by simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (floordiv_vec_eval _ BHW out_width "batch_height_width_offset"
          (fun i : Fin BHW => s.pids 0 * BHW + i.val) (by simp [BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (floordiv_vec_eval _ BHW out_height "batch_height_offset"
          (fun i : Fin BHW => IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BHW + i.val) out_width)
          (by simp [BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (pidblock_eval _ OF (s.pids 1) "out_feat_pid"
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (mod_vec_eval _ BHW out_height "batch_height_offset"
          (fun i : Fin BHW => IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BHW + i.val) out_width)
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (mod_vec_eval _ BHW out_width "batch_height_width_offset"
          (fun i : Fin BHW => s.pids 0 * BHW + i.val)
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
        (inputbase_eval _ BHW IBS IIFS (in_feat_dim / groups) Input (s.pids 2)
          (fun i : Fin BHW => IntegralDType.floorDiv IntegralDType.nat
            (IntegralDType.floorDiv IntegralDType.nat (s.pids 0 * BHW + i.val) out_width) out_height)
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  erw [stepStmts.cons_some (stepStmt_assign_eq_some
        (weightbase_eval _ OF WOFS (out_feat_dim / groups) Weight (s.pids 2)
          (fun j : Fin OF => s.pids 1 * OF + j.val)
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same])))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (accum_init_eval _ BHW OF)), stepStmts.nil]
  refine ⟨_, rfl, ?_⟩
  -- establish convHInv on the final state
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_⟩
  · simp [BlockState.setReg_pids]
  · intro rg o; simp only [BlockState.setReg]; exact hundef rg o
  · simp only [BlockState.setReg]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids]
  · rw [hIGD]; simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, Option.some.injEq]
    ext idx; simp only [batchIdx, bhIdx, bhwIdx, Tile.vec, IntegralDType.nat_floorDiv]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, Option.some.injEq]
    ext idx; simp only [heightIdx, bhIdx, bhwIdx, Tile.vec, IntegralDType.nat_floorDiv, IntegralDType.nat_mod]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, Option.some.injEq]
    ext idx; simp only [widthIdx, bhwIdx, Tile.vec, IntegralDType.nat_mod]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, Option.some.injEq]
    ext idx; simp only [featIdx, Tile.vec]
  · simp only [TileShape.insertAxis, BlockState.setReg_ne_name, BlockState.setReg_same,
      BlockState.setReg_ne_dtype, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
      Option.some.injEq]
    ext idx
    · rfl
    · simp only [inputBase, batchIdx, bhIdx, bhwIdx, hIGD, IntegralDType.nat_floorDiv]
  · simp only [TileShape.insertAxis, BlockState.setReg_ne_name, BlockState.setReg_same,
      BlockState.setReg_ne_dtype, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
      Option.some.injEq]
    ext idx
    · rfl
    · simp only [weightBase, featIdx]
  · -- accum = accH 0 = 0
    simp only [BlockState.setReg_same, Option.some.injEq]
    ext idx; simp only [accH, Finset.range_zero, Finset.sum_empty]

set_option maxHeartbeats 4000000 in
/-- **Top exec closed form**: composing `preLoop` + the height/width/in-feat loop
nest + `postLoop` gives the full `exec`. Every active output lane equals the
genuine convolution value `convSpec`; out-of-bounds lanes are preserved. -/
theorem conv2d_exec_closed_form
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF numCBlocks : Nat)
    (s : BlockState) (hBIN : 0 < BIN) (hundef : ∀ rg o, s.undef rg o = 0)
    (hIGD : in_feat_dim / groups = BIN * numCBlocks)
    (hOutInj : Function.Injective (outputOffset s BHW OF out_height out_width OBS OOFS OHS OWS (out_feat_dim / groups)))
    (idx : TileIndex [BHW, OF]) :
    (match exec (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF) s with
      | some s' => s'.readMem Output (outputOffset s BHW OF out_height out_width OBS OOFS OHS OWS (out_feat_dim / groups) idx)
      | none => (0.0 : ℝ)) =
      if active s BHW OF batch_dim out_height out_width (out_feat_dim / groups) idx then
        convSpec s Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS out_height out_width (BIN * numCBlocks) (out_feat_dim / groups)
          SH SW PH PW KH KW numCBlocks idx.1 idx.2.1
      else s.readMem Output (outputOffset s BHW OF out_height out_width OBS OOFS OHS OWS (out_feat_dim / groups) idx) := by
  set OGD := out_feat_dim / groups with hOGD
  obtain ⟨s0, hpre, hP0⟩ := conv2d_preLoop Input Weight Output batch_dim in_feat_dim in_height in_width
    out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
    KH KW SH SW PH PW groups tf32 BHW BIN OF numCBlocks s hundef hIGD
  obtain ⟨s1, hloop, hPloop⟩ := conv2d_h_loop s Input Weight BHW BIN OF batch_dim in_height in_width
    IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS out_height out_width OGD SH SW PH PW KH KW numCBlocks hBIN s0 hP0
  obtain ⟨sfin, hpost, hpostval⟩ := conv2d_postLoop Input Weight Output s BHW BIN OF batch_dim in_height in_width
    IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS out_height out_width OGD SH SW PH PW KH KW numCBlocks hOutInj s1 hPloop
  have hexec : exec (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
      out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF) s = some sfin := by
    rw [exec, conv2d_body_split Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups tf32 BHW BIN OF]
    rw [stepStmts.append_some hpre, stepStmts.cons_some hloop]
    have : (in_feat_dim / groups) = BIN * numCBlocks := hIGD
    rw [this]
    exact hpost
  rw [hexec]
  have hs1mem : s1.readMem Output (outputOffset s BHW OF out_height out_width OBS OOFS OHS OWS OGD idx)
      = s.readMem Output (outputOffset s BHW OF out_height out_width OBS OOFS OHS OWS OGD idx) := by
    unfold BlockState.readMem; rw [hPloop.1.mem]
  show sfin.readMem Output _ = _
  rw [hpostval idx, hs1mem]

/-- **Closed-form correctness for `conv2d_forward` (general statement).**

For arbitrary dims/strides/groups, kernel window `KH × KW`, in-feat block size
`BIN` and block count `numCBlocks` (so `in_group_dim = BIN · numCBlocks`), every
**active** output cell of the computed `BHW × OF` tile equals the genuine im2col
convolution value `convSpec` — the masked dot-accumulation
`Σ_{h<KH} Σ_{w<KW} Σ_{c<in_group_dim} maskedInput(h,w,c)·maskedWeight(h,w,c)` over
ℝ (with the kernel's padding/boundary masks zeroing out-of-range lanes). Out-of-
bounds output lanes are preserved. This is NOT the kernel's executed value: it is
the independent closed-form convolution reference derived from the loaded tiles.

Preconditions: `fp16 = true` (the DSL always wraps the dot operands in a fp16→
real cast, so only this branch is well-typed; under ℝ the cast is the identity);
`0 < BIN`; `in_feat_dim / groups = BIN · numCBlocks` (the in-feat loop tiles
exactly); clean initial `undef` (so masked-out loads read `0` — zero padding);
output-address injectivity (distinct active lanes hit distinct `Output`
addresses). -/
theorem conv2d_closed_form_correct
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF numCBlocks : Nat)
    (s : BlockState) (hBIN : 0 < BIN) (hundef : ∀ rg o, s.undef rg o = 0)
    (hIGD : in_feat_dim / groups = BIN * numCBlocks)
    (hOutInj : Function.Injective (outputOffset s BHW OF out_height out_width OBS OOFS OHS OWS (out_feat_dim / groups))) :
    ComputeCorrect.Realizes
      (kernel := conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s BHW OF batch_dim out_height out_width (out_feat_dim / groups))
        (fun idx => (Output, outputOffset s BHW OF out_height out_width OBS OOFS OHS OWS (out_feat_dim / groups) idx)))
      (expected := fun idx : TileIndex [BHW, OF] =>
        convSpec s Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS out_height out_width (BIN * numCBlocks) (out_feat_dim / groups)
          SH SW PH PW KH KW numCBlocks idx.1 idx.2.1) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [conv2d_forward_surface, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
  intro s0 s' hExec hs0
  subst hs0
  intro idx hActive
  have hmain := conv2d_exec_closed_form Input Weight Output batch_dim in_feat_dim in_height in_width
    out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
    KH KW SH SW PH PW groups tf32 BHW BIN OF numCBlocks s0 hBIN hundef hIGD hOutInj idx
  rw [hExec] at hmain
  simp only [hActive, if_true] at hmain
  simpa only [ComputeCorrect.OutputReadable.read_real] using hmain

/-- The full conv2d forward surface lowers to the algorithm layer and realizes
the genuine convolution `convSpec` on every active output lane. The convolution
dot-accumulator, the per-block padding/boundary masking, and the masked
writeback are all proven; only the host launch / scheduling is trusted. -/
theorem conv2d_output_summary
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF numCBlocks : Nat)
    (s : BlockState) (hBIN : 0 < BIN) (hundef : ∀ rg o, s.undef rg o = 0)
    (hIGD : in_feat_dim / groups = BIN * numCBlocks)
    (hOutInj : Function.Injective (outputOffset s BHW OF out_height out_width OBS OOFS OHS OWS (out_feat_dim / groups))) :
    (∃ alg, (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgorithm? = Except.ok alg) ∧
    ComputeCorrect.Realizes
      (kernel := conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (active s BHW OF batch_dim out_height out_width (out_feat_dim / groups))
        (fun idx => (Output, outputOffset s BHW OF out_height out_width OBS OOFS OHS OWS (out_feat_dim / groups) idx)))
      (expected := fun idx : TileIndex [BHW, OF] =>
        convSpec s Input Weight BHW BIN OF batch_dim in_height in_width
          IBS IIFS IHS IWS WOFS WIFS WHS WWS out_height out_width (BIN * numCBlocks) (out_feat_dim / groups)
          SH SW PH PW KH KW numCBlocks idx.1 idx.2.1) :=
  ⟨conv2d_forward_surface_toAlgorithm_supported Input Weight Output batch_dim in_feat_dim in_height
      in_width out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF,
    conv2d_closed_form_correct Input Weight Output batch_dim in_feat_dim in_height in_width
      out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups tf32 BHW BIN OF numCBlocks s hBIN hundef hIGD hOutInj⟩

end VeriTile.Bench.TritonBenchG.TritonConv2dFwd

