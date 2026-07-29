import VeriTile.Triton

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
conv2d_closed_form_correct                    ← TOP THEOREM (ComputeCorrect.Realizes_without_Rounding)
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
    ComputeCorrect.Realizes_without_Rounding
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
specification conv2d_output_summary
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
    ComputeCorrect.Realizes_without_Rounding
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


/-! ## The `⊨[R]` streaming headline (wave-5 S1 fold genre, flattened nest)

Everything below is purely additive; the kernel surface and the exact stack
above are untouched.

**The flattening idea.** `conv2d_forward_kernel`'s accumulation is a *triple*
nested loop `for h < KH { for w < KW { for c in range(0, in_group_dim,
BLOCK_IN) } } }`, but the streaming skin `StreamMetaMasked3DKernelIO₂` never
mentions the loop: `ImplementsR` is a Hoare triple over `execR R` of the
**whole** kernel plus per-step read/write windows, and the nest lives entirely
inside the `hrun` obligation, where the file's existing three-level invariant
tower (`conv2d_c_loop` / `conv2d_w_loop` / `conv2d_h_loop`) discharges it
unchanged. Because the nest is a **perfect rectangle** — no inner bound
depends on an outer index (`KH`, `KW` are constants and the `c`-loop's stop
register `in_group_dim` is pid-free) — the whole nest is one stream of
`T = KH · KW · numCBlocks` steps, with step `t` decoded by
`convStepH`/`convStepW`/`convStepCB` (`t / NC / KW`, `t / NC % KW`, `t % NC`;
the first is `t / (KW · NC)` by `Nat.div_div_eq_div_mul`). Every window and
mask is expressible under that decode with zero information loss, so **no
nested-loop skin genre is needed**: a nest of perfect rectangles is a stream.

**Rounding events.** The kernel's only `castFloat` sites are the four in the
inner body — the `if fp16:` branch's `input_block`/`weight_block` widening to
`tl.float16` and the two `fp16 → real` re-widenings inside `tl.dot`. Under
`execR R` these become `R.cast` sites, so the headline carries
`hfp16 : R.round .fp16 = id`, the file's declared fp16 modeling boundary made
explicit (see the file docstring's *Modeling boundary*). With it the three
`convIO_Rcast_*` collapses fire and the entire nest is cast-free, so it steps
identically under `stepStmtR R` and the exact stepper and the proven
invariant tower is reused verbatim. The terminal store is `.real`-typed (no
quantization event), so only it is re-proved on the `R` side
(`conv2d_postLoopR`). -/

section IOFace

open scoped VeriTile.Triton.StreamMetaMasked3DKernelIO₂

/-! ### Cast-free collapse under `R.round .fp16 = id` -/

/-- The `R`-cast on the `.real` channel never rounds. -/
private theorem convIO_Rcast_real_real (R : RoundingModel) :
    R.cast .real .real = FloatDType.cast .real .real := by
  funext x
  unfold RoundingModel.cast FloatDType.cast
  rw [show R.roundW .real = id from by
    funext y; cases y <;> simp [RoundingModel.roundW]]
  rfl

/-- With `R.round .fp16 = id`, the `R`-cast into the fp16 grid is the exact
cast (the in-loop rounding-event site collapses). -/
private theorem convIO_Rcast_real_fp16 (R : RoundingModel) (hfp16 : R.round .fp16 = id) :
    R.cast .real .fp16 = FloatDType.cast .real .fp16 := by
  funext x
  unfold RoundingModel.cast FloatDType.cast
  rw [show R.roundW .fp16 = id from by
    funext y; cases y <;> simp [RoundingModel.roundW, hfp16]]
  rfl

/-- The `R`-cast out of the fp16 grid lands on the `.real` channel, which
never rounds. -/
private theorem convIO_Rcast_fp16_real (R : RoundingModel) :
    R.cast .fp16 .real = FloatDType.cast .fp16 .real := by
  funext x
  unfold RoundingModel.cast FloatDType.cast
  rw [show R.roundW .real = id from by
    funext y; cases y <;> simp [RoundingModel.roundW]]
  rfl

/-- `.real` stores never round: `writeMemTypedR` delegates to the exact write. -/
private theorem convIO_wmtR_real (R : RoundingModel) (s : BlockState)
    (region : RegionName) (o : Nat) (v : TileCarrier .real) :
    s.writeMemTypedR R .real region o v = s.writeMemTyped .real region o v := rfl

/-- A `.real`-typed rounded write **is** the `writeMemAsR R .real` write
(`RoundingModel.storeValue_real`): lets the masked `.real` terminal store
reuse the `writeMemAsR` scatter readback / frame lemma family. -/
private theorem convIO_writeMemTypedR_real_eq (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier TileDType.real) :
    s.writeMemTypedR R .real region offset v
      = s.writeMemAsR R .real region offset v := by
  show s.writeMemTyped .real region offset v = _
  simp only [BlockState.writeMemTyped, BlockState.writeMemAs, BlockState.writeMemAsR,
    RoundingModel.storeValue_real]

/-- Tag-exact readback of a stored `.real` cell through `readMemAs .real`. -/
private theorem convIO_readMemAs_real_of_cell {s : BlockState} {region : RegionName}
    {offset : Nat} {x : ℝ}
    (h : s.mem region offset
      = MemCell.of FloatDType.real.toTileDType (FloatDType.real.ofReal x)) :
    s.readMemAs .real region offset = FloatDType.real.ofReal x := by
  simp [BlockState.readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

/-- Per-statement cast-free collapse lifts to statement lists (walks the
actual successor chain; a failing step collapses on both sides). -/
private theorem convIO_stepStmtsR_castFree_of_stmts (R : RoundingModel) :
    ∀ (l : List Stmt), (∀ st ∈ l, ∀ u, stepStmtR R st u = stepStmt st u) →
      ∀ s, stepStmtsR R l s = stepStmts l s
  | [], _, s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, h, s => by
      simp only [stepStmtsR, stepStmts, h st List.mem_cons_self s]
      cases stepStmt st s with
      | none => rfl
      | some s' =>
          exact convIO_stepStmtsR_castFree_of_stmts R rest
            (fun st' h' u => h st' (List.mem_cons_of_mem _ h') u) s'

/-! ### Named body decomposition

`conv2d_body_split` above splits the kernel as `body.take 14 ++ (nest ::
convPostBody …)`. The `⊨[R]` walk needs the prefix as an explicit statement
list (to run `TraceSafeListR` and the cast-free collapse over it), so
`convPreBody` names those 14 statements and `conv2d_take14_eq` locks it to
the surface. -/

/-- The kernel's 14-statement prologue (pids, group dims, the flattened BHW
index decomposition, the `Input`/`Weight` row/col bases, `accum = 0`) —
`body.take 14`, named. -/
private def convPreBody (Input Weight : RegionName)
    (in_feat_dim out_feat_dim out_height out_width IBS IIFS WOFS groups BHW OF : Nat) :
    List Stmt :=
  [ Stmt.assign .nat [] "batch_height_width_pid" (Op.programId 0),
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
    Stmt.assign .real [BHW, OF] "accum" (Op.full [BHW, OF] (Op.const 0)) ]

/-- `body.take 14` **is** the named prologue. -/
private theorem conv2d_take14_eq (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF : Nat) :
    (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgKernel.body.take 14
      = convPreBody Input Weight in_feat_dim out_feat_dim out_height out_width
          IBS IIFS WOFS groups BHW OF := rfl

/-- `conv2d_body_split` with the prologue named. -/
private theorem conv2d_body_split' (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF : Nat) :
    (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgKernel.body
      = convPreBody Input Weight in_feat_dim out_feat_dim out_height out_width
            IBS IIFS WOFS groups BHW OF
        ++ (Stmt.forRange "h" 0 KH 1
              [Stmt.forRange "w" 0 KW 1
                [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
                  (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]]
            :: convPostBody Input Weight Output batch_dim out_height out_width OBS OOFS OHS OWS
                (in_feat_dim / groups) BHW OF) := by
  rw [conv2d_body_split Input Weight Output batch_dim in_feat_dim in_height in_width
      out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups tf32 BHW BIN OF,
    conv2d_take14_eq Input Weight Output batch_dim in_feat_dim in_height in_width
      out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups tf32 BHW BIN OF]

/-! ### Cast-free collapse of every segment -/

set_option maxHeartbeats 4000000 in
/-- Every prologue statement is cast-free (`.nat` index arithmetic and two
`.ptr` base advances — no rounding site, no memory access). -/
private theorem convIO_preBody_stmt_castFree (R : RoundingModel) (Input Weight : RegionName)
    (in_feat_dim out_feat_dim out_height out_width IBS IIFS WOFS groups BHW OF : Nat) :
    ∀ st ∈ convPreBody Input Weight in_feat_dim out_feat_dim out_height out_width
        IBS IIFS WOFS groups BHW OF,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [convPreBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl <;>
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 4000000 in
/-- Every inner-body statement is cast-free **given `R.round .fp16 = id`**:
the four `castFloat` sites (the `if fp16:` branch's two widenings and the two
`fp16 → real` re-widenings inside `tl.dot`) collapse via the three
`convIO_Rcast_*` lemmas; the two masked loads are `.real`. -/
private theorem convIO_innerBody_stmt_castFree (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat) :
    ∀ st ∈ convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [convInnerBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp only [stepStmtR, stepStmt, stepStmtsR, stepStmts, evalOpR.eq_def, evalOp.eq_def,
      convIO_Rcast_real_real R, convIO_Rcast_real_fp16 R hfp16, convIO_Rcast_fp16_real R]
  all_goals rfl

set_option maxHeartbeats 4000000 in
/-- Both post-loop assigns and the terminal masked `.real` store are cast-free
(`convIO_wmtR_real`). -/
private theorem convIO_postBody_stmt_castFree (R : RoundingModel)
    (Input Weight Output : RegionName)
    (batch_dim out_height out_width OBS OOFS OHS OWS OGD BHW OF : Nat) :
    ∀ st ∈ convPostBody Input Weight Output batch_dim out_height out_width OBS OOFS OHS OWS OGD BHW OF,
      ∀ u, stepStmtR R st u = stepStmt st u := by
  intro st hst u
  simp only [convPostBody, List.mem_cons, List.not_mem_nil, or_false] at hst
  rcases hst with rfl | rfl | rfl <;>
    simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def, convIO_wmtR_real R]

/-- The prologue collapses onto the exact stepper. -/
private theorem convIO_preBody_castFree (R : RoundingModel) (Input Weight : RegionName)
    (in_feat_dim out_feat_dim out_height out_width IBS IIFS WOFS groups BHW OF : Nat)
    (t : BlockState) :
    stepStmtsR R (convPreBody Input Weight in_feat_dim out_feat_dim out_height out_width
        IBS IIFS WOFS groups BHW OF) t
      = stepStmts (convPreBody Input Weight in_feat_dim out_feat_dim out_height out_width
        IBS IIFS WOFS groups BHW OF) t :=
  convIO_stepStmtsR_castFree_of_stmts R _
    (convIO_preBody_stmt_castFree R Input Weight in_feat_dim out_feat_dim out_height out_width
      IBS IIFS WOFS groups BHW OF) t

/-- The inner `c`-loop body collapses onto the exact stepper (under `hfp16`). -/
private theorem convIO_innerBody_castFree (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW : Nat)
    (t : BlockState) :
    stepStmtsR R (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW) t
      = stepStmts (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW) t :=
  convIO_stepStmtsR_castFree_of_stmts R _
    (convIO_innerBody_stmt_castFree R hfp16 BHW BIN OF IIFS IHS IWS WIFS WHS WWS
      batch_dim in_height in_width PH PW SH SW) t

/-- The post-loop tail collapses onto the exact stepper. -/
private theorem convIO_postBody_castFree (R : RoundingModel)
    (Input Weight Output : RegionName)
    (batch_dim out_height out_width OBS OOFS OHS OWS OGD BHW OF : Nat) (t : BlockState) :
    stepStmtsR R (convPostBody Input Weight Output batch_dim out_height out_width OBS OOFS OHS OWS OGD BHW OF) t
      = stepStmts (convPostBody Input Weight Output batch_dim out_height out_width OBS OOFS OHS OWS OGD BHW OF) t :=
  convIO_stepStmtsR_castFree_of_stmts R _
    (convIO_postBody_stmt_castFree R Input Weight Output batch_dim out_height out_width
      OBS OOFS OHS OWS OGD BHW OF) t

/-- **The missing `forRangeDyn` cast-free lift.** `StepR` ships
`stepStmtR_forRange` for the *static* loop but no `forRangeDyn` mirror, so the
inner `c`-loop's collapse is proved here, straight from `stepStmtR`'s
definition: the three bound expressions (`0`, `in_group_dim`, `BLOCK_IN`) are
a `constNat`/`ref`/`constNat` triple — cast-free by construction, evaluating
identically under `evalOpR R` and `evalOp` — after which the strided fold is
`stepForRangeAuxR_castFree`. -/
private theorem convIO_cLoopStmt_castFree (R : RoundingModel) (BIN : Nat) (body : List Stmt)
    (hbody : ∀ t : BlockState, stepStmtsR R body t = stepStmts body t) (u : BlockState) :
    stepStmtR R (Stmt.forRangeDyn "c" (Op.constNat 0)
        (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN) body) u
      = stepStmt (Stmt.forRangeDyn "c" (Op.constNat 0)
        (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN) body) u := by
  simp only [stepStmtR, stepStmt, evalOpR.eq_def, evalOp.eq_def, bind, Option.bind_eq_bind]
  cases hgd : u.regs .nat [] "in_group_dim" with
  | none => simp only [Option.bind_some, Option.bind_none]
  | some v =>
      simp only [Option.bind_some]
      exact stepForRangeAuxR_castFree R body hbody "c" 0 (v.data PUnit.unit) BIN u

/-- The middle `w`-loop statement is cast-free: a static `forRange` whose body
is the single `c`-loop statement. -/
private theorem convIO_wLoopStmt_castFree (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW KW : Nat)
    (u : BlockState) :
    stepStmtR R (Stmt.forRange "w" 0 KW 1
        [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
          (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]) u
      = stepStmt (Stmt.forRange "w" 0 KW 1
        [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
          (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]) u := by
  rw [stepStmtR_forRange,
    stepForRangeAuxR_castFree R _
      (convIO_stepStmtsR_castFree_of_stmts R _ (by
        intro st hst t
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
        subst hst
        exact convIO_cLoopStmt_castFree R BIN _
          (convIO_innerBody_castFree R hfp16 BHW BIN OF IIFS IHS IWS WIFS WHS WWS
            batch_dim in_height in_width PH PW SH SW) t)) "w",
    ← stepForRangeAux.forRange_unfold]

/-- The outer `h`-loop statement — the whole nest — is cast-free. -/
private theorem convIO_hLoopStmt_castFree (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW KH KW : Nat)
    (u : BlockState) :
    stepStmtR R (Stmt.forRange "h" 0 KH 1
        [Stmt.forRange "w" 0 KW 1
          [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
            (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]]) u
      = stepStmt (Stmt.forRange "h" 0 KH 1
        [Stmt.forRange "w" 0 KW 1
          [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
            (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]]) u := by
  rw [stepStmtR_forRange,
    stepForRangeAuxR_castFree R _
      (convIO_stepStmtsR_castFree_of_stmts R _ (by
        intro st hst t
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
        subst hst
        exact convIO_wLoopStmt_castFree R hfp16 BHW BIN OF IIFS IHS IWS WIFS WHS WWS
          batch_dim in_height in_width PH PW SH SW KW t)) "h",
    ← stepForRangeAux.forRange_unfold]

/-! ### The flattening decode

`T = KH · KW · numCBlocks` (parsed `(KH · KW) · numCBlocks`), and step `t` is
the row-major encoding of the nest's iteration triple `(h, w, cb)`. The three
decodes below invert it; `convStepH` is `t / (KW · numCBlocks)` by
`Nat.div_div_eq_div_mul`, and the split-by-`numCBlocks`-first spelling is the
one that composes with the `Lane2D` row-major bridge (used twice: once for
`(h, w) ↦ h · KW + w`, once for `((h,w), cb) ↦ (h · KW + w) · NC + cb`). -/

/-- Outer kernel-height index of flat step `t` (`= t / (KW · NC)`). -/
private def convStepH (KW NC t : Nat) : Nat := t / NC / KW

/-- Middle kernel-width index of flat step `t`. -/
private def convStepW (KW NC t : Nat) : Nat := t / NC % KW

/-- Inner in-feature *block* index of flat step `t` (the `c`-loop counter is
`convStepCB … t · BLOCK_IN`). -/
private def convStepCB (NC t : Nat) : Nat := t % NC

/-- The flat step of nest iteration `(h, w, cb)` — the row-major encode that
`convStepH`/`convStepW`/`convStepCB` invert. -/
private def convStepIdx {KH KW NC : Nat} (h : Fin KH) (w : Fin KW) (cb : Fin NC) :
    Fin (KH * KW * NC) :=
  Lane2D.encode (Lane2D.encode (h, w, PUnit.unit), cb, PUnit.unit)

@[simp] private theorem convStepIdx_cb {KH KW NC : Nat} (h : Fin KH) (w : Fin KW) (cb : Fin NC) :
    convStepCB NC (convStepIdx h w cb).val = cb.val :=
  Lane2D.encode_mod (M := KH * KW) (N := NC) _

@[simp] private theorem convStepIdx_w {KH KW NC : Nat} (h : Fin KH) (w : Fin KW) (cb : Fin NC) :
    convStepW KW NC (convStepIdx h w cb).val = w.val := by
  show (convStepIdx h w cb).val / NC % KW = w.val
  simp only [convStepIdx]
  rw [Lane2D.encode_div (M := KH * KW) (N := NC)]
  exact Lane2D.encode_mod (M := KH) (N := KW) _

@[simp] private theorem convStepIdx_h {KH KW NC : Nat} (h : Fin KH) (w : Fin KW) (cb : Fin NC) :
    convStepH KW NC (convStepIdx h w cb).val = h.val := by
  show (convStepIdx h w cb).val / NC / KW = h.val
  simp only [convStepIdx]
  rw [Lane2D.encode_div (M := KH * KW) (N := NC)]
  exact Lane2D.encode_div (M := KH) (N := KW) _

/-- The `Input`-stream lane feeding output lane `l` at inner key `e`: the BHW
row of `l` (row-major over the `[BHW, OF]` output tile) paired with `e` over
the `[BHW, BLOCK_IN]` per-step input tile. -/
def convInLane (BHW BIN OF : Nat) (l : Fin (BHW * OF)) (e : Fin BIN) : Fin (BHW * BIN) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)

/-- The `Weight`-stream lane feeding output lane `l` at inner key `e`: `e`
paired with the out-feature column of `l` over the `[BLOCK_IN, OF]` per-step
weight tile. -/
def convWLane (BHW BIN OF : Nat) (l : Fin (BHW * OF)) (e : Fin BIN) : Fin (BIN * OF) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)

/-! ### pid-form index / address / mask vocabulary

The exact stack's `bhwIdx`/`inAddr`/`inMask`/… are indexed by the launch
state `s0` (they read `s0.pids`). The skin quantifies the launch state
internally and hands its window fields the three pids as plain `Nat`s, so the
same quantities are restated here in pid form. Every pid-form definition is
**definitionally** the `s0`-form at `pid_k := s0.pids k` (the bridges below
are `rfl`), so no faithfulness is added or lost. -/

/-- pid-form `batch_height_width_offset` lane `i`: `pid₀ · BHW + i`. -/
private def pBhw (pid₀ BHW i : Nat) : Nat := pid₀ * BHW + i

/-- pid-form `batch_height_offset`. -/
private def pBh (pid₀ OW BHW i : Nat) : Nat := pBhw pid₀ BHW i / OW

/-- pid-form `batch_offset`. -/
private def pBatch (pid₀ OH OW BHW i : Nat) : Nat := pBh pid₀ OW BHW i / OH

/-- pid-form `output_height_offset`. -/
private def pHeight (pid₀ OH OW BHW i : Nat) : Nat := pBh pid₀ OW BHW i % OH

/-- pid-form `output_width_offset`. -/
private def pWidth (pid₀ OW BHW i : Nat) : Nat := pBhw pid₀ BHW i % OW

/-- pid-form `output_feat_offset` lane `j`: `pid₁ · OF + j`. -/
private def pFeat (pid₁ OF j : Nat) : Nat := pid₁ * OF + j

/-- pid-form per-row `Input +=` base. -/
private def pInputBase (pid₀ pid₂ IBS IIFS OH OW BHW IGD i : Nat) : Nat :=
  IBS * pBatch pid₀ OH OW BHW i + IIFS * pid₂ * IGD

/-- pid-form per-col `Weight +=` base. -/
private def pWeightBase (pid₁ pid₂ WOFS OF OGD j : Nat) : Nat :=
  WOFS * pFeat pid₁ OF j + WOFS * pid₂ * OGD

/-- pid-form `input_height_offset` (an `Int`; negative under padding). -/
private def pInH (pid₀ OH OW BHW SH PH h i : Nat) : Int :=
  (h : Int) - (PH : Int) + (SH : Int) * (pHeight pid₀ OH OW BHW i : Int)

/-- pid-form `input_width_offset` (an `Int`; negative under padding). -/
private def pInW (pid₀ OW BHW SW PW w i : Nat) : Int :=
  (w : Int) - (PW : Int) + (SW : Int) * (pWidth pid₀ OW BHW i : Int)

/-- pid-form `curr_input_pointer` cell at lane `(i, e)`, nest iteration
`(h, w, c)`. **Honest sentinel address**: `pInH` may be negative under padding,
so `((IHS : Int) * pInH …).toNat` is garbage on the masked-off lanes — exactly
as the kernel's own `tl.load` pointer arithmetic is. Harmless: every skin
obligation about `read1` is guarded by `mask1 →`. -/
private def pInAddr (pid₀ pid₂ IBS IIFS IHS IWS OH OW BHW IGD SH SW PH PW h w c i e : Nat) : Nat :=
  pInputBase pid₀ pid₂ IBS IIFS OH OW BHW IGD i + IIFS * (c + e)
    + ((IHS : Int) * pInH pid₀ OH OW BHW SH PH h i).toNat
    + ((IWS : Int) * pInW pid₀ OW BHW SW PW w i).toNat

/-- pid-form `input_mask` at lane `(i, e)`, nest iteration `(h, w, c)`. -/
private def pInMask (pid₀ batch_dim in_height in_width OH OW BHW IGD SH SW PH PW
    h w c i e : Nat) : Prop :=
  ((((pBatch pid₀ OH OW BHW i < batch_dim ∧ c + e < IGD) ∧
      (0 : Int) ≤ pInH pid₀ OH OW BHW SH PH h i) ∧
      pInH pid₀ OH OW BHW SH PH h i < (in_height : Int)) ∧
      (0 : Int) ≤ pInW pid₀ OW BHW SW PW w i) ∧
      pInW pid₀ OW BHW SW PW w i < (in_width : Int)

instance (pid₀ batch_dim in_height in_width OH OW BHW IGD SH SW PH PW h w c i e : Nat) :
    Decidable (pInMask pid₀ batch_dim in_height in_width OH OW BHW IGD SH SW PH PW h w c i e) := by
  unfold pInMask; infer_instance

/-- pid-form `curr_weight_pointer` cell at lane `(e, j)`. -/
private def pWAddr (pid₁ pid₂ WOFS WIFS WHS WWS OF OGD h w c e j : Nat) : Nat :=
  pWeightBase pid₁ pid₂ WOFS OF OGD j + WIFS * (c + e) + WHS * h + WWS * w

/-- pid-form `weight_mask` at lane `(e, j)`. -/
private def pWMask (pid₁ IGD OGD OF c e j : Nat) : Prop :=
  c + e < IGD ∧ pFeat pid₁ OF j < OGD

instance (pid₁ IGD OGD OF c e j : Nat) : Decidable (pWMask pid₁ IGD OGD OF c e j) := by
  unfold pWMask; infer_instance

/-- pid-form terminal `Output` store address at lane `(i, j)`. -/
private def pOutAddr (pid₀ pid₁ pid₂ BHW OF OH OW OBS OOFS OHS OWS OGD i j : Nat) : Nat :=
  OBS * pBatch pid₀ OH OW BHW i + OOFS * (pid₂ * OGD + pFeat pid₁ OF j)
    + OHS * pHeight pid₀ OH OW BHW i + OWS * pWidth pid₀ OW BHW i

/-- pid-form `output_mask` at lane `(i, j)`. -/
private def pActive (pid₀ pid₁ BHW OF batch_dim OH OW OGD i j : Nat) : Prop :=
  ((pBatch pid₀ OH OW BHW i < batch_dim ∧ pFeat pid₁ OF j < OGD) ∧
    pHeight pid₀ OH OW BHW i < OH) ∧ pWidth pid₀ OW BHW i < OW

instance (pid₀ pid₁ BHW OF batch_dim OH OW OGD i j : Nat) :
    Decidable (pActive pid₀ pid₁ BHW OF batch_dim OH OW OGD i j) := by
  unfold pActive; infer_instance

/-! ### `s0`-form ↔ pid-form bridges (all `rfl`) -/

private theorem pBatch_eq (s0 : BlockState) (OH OW BHW : Nat) (i : Fin BHW) :
    batchIdx s0 OH OW BHW i = pBatch (s0.pids 0) OH OW BHW i.val := rfl

private theorem pHeight_eq (s0 : BlockState) (OH OW BHW : Nat) (i : Fin BHW) :
    heightIdx s0 OH OW BHW i = pHeight (s0.pids 0) OH OW BHW i.val := rfl

private theorem pWidth_eq (s0 : BlockState) (OW BHW : Nat) (i : Fin BHW) :
    widthIdx s0 OW BHW i = pWidth (s0.pids 0) OW BHW i.val := rfl

private theorem pFeat_eq (s0 : BlockState) (OF : Nat) (j : Fin OF) :
    featIdx s0 OF j = pFeat (s0.pids 1) OF j.val := rfl

private theorem pInAddr_eq (s0 : BlockState)
    (IBS IIFS IHS IWS OH OW BHW IGD SH SW PH PW : Nat) {BIN : Nat} (h w c : Nat)
    (i : Fin BHW) (e : Fin BIN) :
    inAddr s0 IBS IIFS IHS IWS OH OW BHW IGD SH SW PH PW h w c i e
      = pInAddr (s0.pids 0) (s0.pids 2) IBS IIFS IHS IWS OH OW BHW IGD SH SW PH PW
          h w c i.val e.val := rfl

private theorem pInMask_iff (s0 : BlockState)
    (batch_dim in_height in_width OH OW BHW IGD SH SW PH PW : Nat) {BIN : Nat} (h w c : Nat)
    (i : Fin BHW) (e : Fin BIN) :
    inMask s0 batch_dim in_height in_width OH OW BHW IGD SH SW PH PW h w c i e
      ↔ pInMask (s0.pids 0) batch_dim in_height in_width OH OW BHW IGD SH SW PH PW
          h w c i.val e.val := Iff.rfl

private theorem pWAddr_eq (s0 : BlockState) (WOFS WIFS WHS WWS OF OGD : Nat) {BIN : Nat}
    (h w c : Nat) (e : Fin BIN) (j : Fin OF) :
    wAddr s0 WOFS WIFS WHS WWS OF OGD h w c e j
      = pWAddr (s0.pids 1) (s0.pids 2) WOFS WIFS WHS WWS OF OGD h w c e.val j.val := rfl

private theorem pWMask_iff (s0 : BlockState) (IGD OGD OF c : Nat) {BIN : Nat}
    (e : Fin BIN) (j : Fin OF) :
    wMask s0 IGD OGD OF c e j ↔ pWMask (s0.pids 1) IGD OGD OF c e.val j.val := Iff.rfl

private theorem pOutAddr_eq (s0 : BlockState) (BHW OF OH OW OBS OOFS OHS OWS OGD : Nat)
    (idx : TileIndex [BHW, OF]) :
    outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD idx
      = pOutAddr (s0.pids 0) (s0.pids 1) (s0.pids 2) BHW OF OH OW OBS OOFS OHS OWS OGD
          idx.1.val idx.2.1.val := rfl

private theorem pActive_iff (s0 : BlockState) (BHW OF batch_dim OH OW OGD : Nat)
    (idx : TileIndex [BHW, OF]) :
    active s0 BHW OF batch_dim OH OW OGD idx
      ↔ pActive (s0.pids 0) (s0.pids 1) BHW OF batch_dim OH OW OGD idx.1.val idx.2.1.val :=
  Iff.rfl

/-! ### IO signature and the streamed spec -/

/-- **The streamed conv2d spec**: the ideal ℝ value of output lane `l = (i, n)`
as a single `T`-step fold over the flattened nest,

`∑_{t < KH·KW·NC} ∑_{e < BLOCK_IN} maskedInput(t)[i, e] · maskedWeight(t)[e, n]`,

where the two masked factors are the streamed tiles zeroed exactly where the
kernel's own per-lane masks are false (`pInMask` = image/channel boundaries,
i.e. zero padding; `pWMask` = in-feature / out-feature group boundaries). This
is the `⊨[R]` face of `convSpec`: the same im2col convolution reference,
re-expressed on the streaming skin's per-step tiles instead of on `s0`-indexed
`readMem`s. -/
noncomputable def convStreamSum
    (pid₀ pid₁ : Nat)
    (batch_dim in_height in_width OH OW OGD SH SW PH PW
      KH KW BHW BIN OF numCBlocks : Nat)
    (xs : Fin (KH * KW * numCBlocks) → Fin (BHW * BIN) → ℝ)
    (ys : Fin (KH * KW * numCBlocks) → Fin (BIN * OF) → ℝ)
    (l : Fin (BHW * OF)) : ℝ :=
  ∑ t : Fin (KH * KW * numCBlocks), ∑ e : Fin BIN,
    (if pInMask pid₀ batch_dim in_height in_width OH OW BHW (BIN * numCBlocks) SH SW PH PW
          (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
          (convStepCB numCBlocks t.val * BIN) (Lane2D.decode l).1.val e.val then
        xs t (convInLane BHW BIN OF l e)
      else 0)
      * (if pWMask pid₁ (BIN * numCBlocks) OGD OF
            (convStepCB numCBlocks t.val * BIN) e.val (Lane2D.decode l).2.1.val then
          ys t (convWLane BHW BIN OF l e)
        else 0)

/-- **Streaming IO signature** of `conv2d_forward` on the metadata-parametrized
two-stream fold skin (S1: fold + terminal masked store, 3-D pid grid), at
`nMeta := 0` — this kernel loads **no** scalar metadata, so the slot vector is
empty (`sty`/`mbuf`/`mwin` are `Fin.elim0`) and every window is a function of
the three pids alone. The 3-D grid is the kernel's own: `pid₀` = flattened
batch·height·width tile, `pid₁` = out-feature tile, `pid₂` = group.

The **triple nest is flattened into one stream** of `T = KH · KW · numCBlocks`
steps (see the section docstring): step `t` is nest iteration
`(convStepH t, convStepW t, convStepCB t)` with `c`-loop counter
`convStepCB t · BLOCK_IN`. Per step the kernel reads a `[BHW, BLOCK_IN]` input
tile and a `[BLOCK_IN, OF]` weight tile, both **masked** with no `other=`
default (the zero padding comes from clean `undef`, which the skin's triple
pins); after the nest one `[BHW, OF]` output tile is masked-stored at the
**`.real`** grid (`outDType` default — the transcribed `tl.store` is untyped,
no quantization event).

* `read1` lane `j = (i, e)` (row-major over `[BHW, BLOCK_IN]`): the kernel's
  `curr_input_pointer` cell, `pInAddr` — note its `((IHS : Int) · input_height_offset).toNat`
  is a garbage address on masked-off lanes (padding makes the `Int` offset
  negative), exactly as in the kernel; harmless because every obligation is
  `mask1`-guarded.
* `read2` lane `j = (e, n)` (row-major over `[BLOCK_IN, OF]`): the
  `curr_weight_pointer` cell, `pWAddr`.
* `mask1` / `mask2`: the kernel's `input_mask` (6 conjuncts) / `weight_mask`.
* `write` / `writeMask` lane `j = (i, n)`: the terminal `Output +=` address
  and `output_mask`. -/
def triton_conv2d_fwd_IO (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF numCBlocks : Nat) :
    StreamMetaMasked3DKernelIO₂ where
  kernel := conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
    out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
    KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF
  inp1 := Input
  inp2 := Weight
  out := Output
  nMeta := 0
  sty := Fin.elim0
  mbuf := Fin.elim0
  mwin := Fin.elim0
  T := KH * KW * numCBlocks
  B1 := BHW * BIN
  B2 := BIN * OF
  C := BHW * OF
  outDType := .real
  read1 := fun pid₀ _ pid₂ _ t j =>
    pInAddr pid₀ pid₂ IBS IIFS IHS IWS out_height out_width BHW (BIN * numCBlocks) SH SW PH PW
      (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
      (convStepCB numCBlocks t.val * BIN) (j.val / BIN) (j.val % BIN)
  read2 := fun _ pid₁ pid₂ _ t j =>
    pWAddr pid₁ pid₂ WOFS WIFS WHS WWS OF (out_feat_dim / groups)
      (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
      (convStepCB numCBlocks t.val * BIN) (j.val / OF) (j.val % OF)
  write := fun pid₀ pid₁ pid₂ _ j =>
    pOutAddr pid₀ pid₁ pid₂ BHW OF out_height out_width OBS OOFS OHS OWS
      (out_feat_dim / groups) (j.val / OF) (j.val % OF)
  mask1 := fun pid₀ _ _ _ t j =>
    pInMask pid₀ batch_dim in_height in_width out_height out_width BHW (BIN * numCBlocks)
      SH SW PH PW (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
      (convStepCB numCBlocks t.val * BIN) (j.val / BIN) (j.val % BIN)
  mask2 := fun _ pid₁ _ _ t j =>
    pWMask pid₁ (BIN * numCBlocks) (out_feat_dim / groups) OF
      (convStepCB numCBlocks t.val * BIN) (j.val / OF) (j.val % OF)
  writeMask := fun pid₀ pid₁ _ _ j =>
    pActive pid₀ pid₁ BHW OF batch_dim out_height out_width (out_feat_dim / groups)
      (j.val / OF) (j.val % OF)

/-! ### The spec bridge: `convSpec` **is** the flattened stream fold

The one genuinely new proof obligation of the flattening: `convSpec`'s
`range KH × range KW × range numCBlocks` triple sum has to become the skin's
single `∑ t : Fin (KH · KW · numCBlocks)`. `convStreamSum_reindex` does it with
two applications of `StreamLane.sum_univ_mul` (the row-major block
decomposition), matching `convStepIdx`'s two `Lane2D` encodes; the decode
identities `convStepIdx_h` / `_w` / `_cb` close the per-term step. -/

set_option maxHeartbeats 1000000 in
/-- **The flattening reindex.** A flat `KH · KW · numCBlocks`-step sum of a
per-step quantity that depends on the step only through its decoded nest
triple *is* the nest's triple sum. -/
private theorem convStreamSum_reindex (KH KW NC BIN : Nat) (g : Nat → Nat → Nat → Fin BIN → ℝ) :
    (∑ t : Fin (KH * KW * NC), ∑ e : Fin BIN,
        g (convStepH KW NC t.val) (convStepW KW NC t.val) (convStepCB NC t.val) e)
      = ∑ h ∈ Finset.range KH, ∑ w ∈ Finset.range KW, ∑ cb ∈ Finset.range NC,
          ∑ e : Fin BIN, g h w cb e := by
  have step1 : (∑ t : Fin (KH * KW * NC), ∑ e : Fin BIN,
        g (convStepH KW NC t.val) (convStepW KW NC t.val) (convStepCB NC t.val) e)
      = ∑ a : Fin (KH * KW), ∑ b : Fin NC, ∑ e : Fin BIN,
          g (convStepH KW NC (a.val * NC + b.val)) (convStepW KW NC (a.val * NC + b.val))
            (convStepCB NC (a.val * NC + b.val)) e :=
    StreamLane.sum_univ_mul (KH * KW) NC
      (fun tv => ∑ e : Fin BIN, g (convStepH KW NC tv) (convStepW KW NC tv) (convStepCB NC tv) e)
  have step2 : (∑ a : Fin (KH * KW), ∑ b : Fin NC, ∑ e : Fin BIN,
          g (convStepH KW NC (a.val * NC + b.val)) (convStepW KW NC (a.val * NC + b.val))
            (convStepCB NC (a.val * NC + b.val)) e)
      = ∑ h : Fin KH, ∑ w : Fin KW, ∑ b : Fin NC, ∑ e : Fin BIN,
          g (convStepH KW NC ((h.val * KW + w.val) * NC + b.val))
            (convStepW KW NC ((h.val * KW + w.val) * NC + b.val))
            (convStepCB NC ((h.val * KW + w.val) * NC + b.val)) e :=
    StreamLane.sum_univ_mul KH KW
      (fun av => ∑ b : Fin NC, ∑ e : Fin BIN,
        g (convStepH KW NC (av * NC + b.val)) (convStepW KW NC (av * NC + b.val))
          (convStepCB NC (av * NC + b.val)) e)
  have step3 : ∀ (h : Fin KH) (w : Fin KW) (b : Fin NC),
      (∑ e : Fin BIN, g (convStepH KW NC ((h.val * KW + w.val) * NC + b.val))
            (convStepW KW NC ((h.val * KW + w.val) * NC + b.val))
            (convStepCB NC ((h.val * KW + w.val) * NC + b.val)) e)
        = ∑ e : Fin BIN, g h.val w.val b.val e := by
    intro h w b
    rw [show (h.val * KW + w.val) * NC + b.val = (convStepIdx h w b).val from rfl,
      convStepIdx_h, convStepIdx_w, convStepIdx_cb]
  rw [step1, step2]
  simp only [step3]
  rw [Fin.sum_univ_eq_sum_range
    (fun hv => ∑ w : Fin KW, ∑ b : Fin NC, ∑ e : Fin BIN, g hv w.val b.val e) KH]
  refine Finset.sum_congr rfl fun h _ => ?_
  rw [Fin.sum_univ_eq_sum_range (fun wv => ∑ b : Fin NC, ∑ e : Fin BIN, g h wv b.val e) KW]
  refine Finset.sum_congr rfl fun w _ => ?_
  exact Fin.sum_univ_eq_sum_range (fun bv => ∑ e : Fin BIN, g h w bv e) NC

set_option maxHeartbeats 1000000 in
/-- **`convSpec` = `convStreamSum`.** Under the two stream pins (stated in the
exact stack's `s0` vocabulary), the genuine im2col convolution reference at
output lane `l = (i, n)` is the skin-level flattened fold: the masked loaded
values `miVal` / `mwVal` *are* the streamed tiles `xs` / `ys` gated by the
skin's `mask1` / `mask2` (both zero exactly where the kernel zero-pads), and
the nest's triple sum is the single `T`-step sum by `convStreamSum_reindex`. -/
private theorem convSpec_eq_streamSum (s₀ : BlockState) (Input Weight : RegionName)
    (batch_dim in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW OGD SH SW PH PW
      KH KW BHW BIN OF numCBlocks : Nat)
    (xs : Fin (KH * KW * numCBlocks) → Fin (BHW * BIN) → ℝ)
    (ys : Fin (KH * KW * numCBlocks) → Fin (BIN * OF) → ℝ)
    (hx : ∀ (t : Fin (KH * KW * numCBlocks)) (i : Fin BHW) (e : Fin BIN),
      inMask s₀ batch_dim in_height in_width OH OW BHW (BIN * numCBlocks) SH SW PH PW
          (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
          (convStepCB numCBlocks t.val * BIN) i e →
      s₀.readMem Input (inAddr s₀ IBS IIFS IHS IWS OH OW BHW (BIN * numCBlocks) SH SW PH PW
          (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
          (convStepCB numCBlocks t.val * BIN) i e)
        = xs t (Lane2D.encode (i, e, PUnit.unit)))
    (hy : ∀ (t : Fin (KH * KW * numCBlocks)) (e : Fin BIN) (n : Fin OF),
      wMask s₀ (BIN * numCBlocks) OGD OF (convStepCB numCBlocks t.val * BIN) e n →
      s₀.readMem Weight (wAddr s₀ WOFS WIFS WHS WWS OF OGD
          (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
          (convStepCB numCBlocks t.val * BIN) e n)
        = ys t (Lane2D.encode (e, n, PUnit.unit)))
    (l : Fin (BHW * OF)) :
    convSpec s₀ Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW (BIN * numCBlocks) OGD SH SW PH PW
        KH KW numCBlocks (Lane2D.decode l).1 (Lane2D.decode l).2.1
      = convStreamSum (s₀.pids 0) (s₀.pids 1) batch_dim in_height in_width OH OW OGD
          SH SW PH PW KH KW BHW BIN OF numCBlocks xs ys l := by
  have hterm : ∀ (t : Fin (KH * KW * numCBlocks)) (e : Fin BIN),
      (if pInMask (s₀.pids 0) batch_dim in_height in_width OH OW BHW (BIN * numCBlocks)
            SH SW PH PW (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
            (convStepCB numCBlocks t.val * BIN) (Lane2D.decode l).1.val e.val then
          xs t (convInLane BHW BIN OF l e)
        else 0)
        * (if pWMask (s₀.pids 1) (BIN * numCBlocks) OGD OF
              (convStepCB numCBlocks t.val * BIN) e.val (Lane2D.decode l).2.1.val then
            ys t (convWLane BHW BIN OF l e)
          else 0)
      = miVal s₀ Input BHW BIN batch_dim in_height in_width IBS IIFS IHS IWS OH OW
            (BIN * numCBlocks) SH SW PH PW (convStepH KW numCBlocks t.val)
            (convStepW KW numCBlocks t.val) (convStepCB numCBlocks t.val * BIN)
            (Lane2D.decode l).1 e
        * mwVal s₀ Weight BIN OF (BIN * numCBlocks) OGD WOFS WIFS WHS WWS
            (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
            (convStepCB numCBlocks t.val * BIN) e (Lane2D.decode l).2.1 := by
    intro t e
    have hfac1 :
        (if pInMask (s₀.pids 0) batch_dim in_height in_width OH OW BHW (BIN * numCBlocks)
              SH SW PH PW (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
              (convStepCB numCBlocks t.val * BIN) (Lane2D.decode l).1.val e.val then
            xs t (convInLane BHW BIN OF l e)
          else 0)
        = miVal s₀ Input BHW BIN batch_dim in_height in_width IBS IIFS IHS IWS OH OW
            (BIN * numCBlocks) SH SW PH PW (convStepH KW numCBlocks t.val)
            (convStepW KW numCBlocks t.val) (convStepCB numCBlocks t.val * BIN)
            (Lane2D.decode l).1 e := by
      unfold miVal
      by_cases hm : pInMask (s₀.pids 0) batch_dim in_height in_width OH OW BHW (BIN * numCBlocks)
          SH SW PH PW (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
          (convStepCB numCBlocks t.val * BIN) (Lane2D.decode l).1.val e.val
      · rw [if_pos hm,
          if_pos ((pInMask_iff s₀ batch_dim in_height in_width OH OW BHW (BIN * numCBlocks)
            SH SW PH PW _ _ _ (Lane2D.decode l).1 e).mpr hm)]
        exact (hx t (Lane2D.decode l).1 e
          ((pInMask_iff s₀ batch_dim in_height in_width OH OW BHW (BIN * numCBlocks)
            SH SW PH PW _ _ _ (Lane2D.decode l).1 e).mpr hm)).symm
      · rw [if_neg hm,
          if_neg (fun hc => hm ((pInMask_iff s₀ batch_dim in_height in_width OH OW BHW
            (BIN * numCBlocks) SH SW PH PW _ _ _ (Lane2D.decode l).1 e).mp hc))]
    have hfac2 :
        (if pWMask (s₀.pids 1) (BIN * numCBlocks) OGD OF
              (convStepCB numCBlocks t.val * BIN) e.val (Lane2D.decode l).2.1.val then
            ys t (convWLane BHW BIN OF l e)
          else 0)
        = mwVal s₀ Weight BIN OF (BIN * numCBlocks) OGD WOFS WIFS WHS WWS
            (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
            (convStepCB numCBlocks t.val * BIN) e (Lane2D.decode l).2.1 := by
      unfold mwVal
      by_cases hm : pWMask (s₀.pids 1) (BIN * numCBlocks) OGD OF
          (convStepCB numCBlocks t.val * BIN) e.val (Lane2D.decode l).2.1.val
      · rw [if_pos hm,
          if_pos ((pWMask_iff s₀ (BIN * numCBlocks) OGD OF _ e (Lane2D.decode l).2.1).mpr hm)]
        exact (hy t e (Lane2D.decode l).2.1
          ((pWMask_iff s₀ (BIN * numCBlocks) OGD OF _ e (Lane2D.decode l).2.1).mpr hm)).symm
      · rw [if_neg hm,
          if_neg (fun hc => hm ((pWMask_iff s₀ (BIN * numCBlocks) OGD OF _ e
            (Lane2D.decode l).2.1).mp hc))]
    rw [hfac1, hfac2]
  unfold convStreamSum
  rw [show (∑ t : Fin (KH * KW * numCBlocks), ∑ e : Fin BIN,
        (if pInMask (s₀.pids 0) batch_dim in_height in_width OH OW BHW (BIN * numCBlocks)
              SH SW PH PW (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
              (convStepCB numCBlocks t.val * BIN) (Lane2D.decode l).1.val e.val then
            xs t (convInLane BHW BIN OF l e)
          else 0)
          * (if pWMask (s₀.pids 1) (BIN * numCBlocks) OGD OF
                (convStepCB numCBlocks t.val * BIN) e.val (Lane2D.decode l).2.1.val then
              ys t (convWLane BHW BIN OF l e)
            else 0))
      = ∑ t : Fin (KH * KW * numCBlocks), ∑ e : Fin BIN,
          miVal s₀ Input BHW BIN batch_dim in_height in_width IBS IIFS IHS IWS OH OW
              (BIN * numCBlocks) SH SW PH PW (convStepH KW numCBlocks t.val)
              (convStepW KW numCBlocks t.val) (convStepCB numCBlocks t.val * BIN)
              (Lane2D.decode l).1 e
            * mwVal s₀ Weight BIN OF (BIN * numCBlocks) OGD WOFS WIFS WHS WWS
              (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
              (convStepCB numCBlocks t.val * BIN) e (Lane2D.decode l).2.1
      from Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun e _ => hterm t e]
  rw [convStreamSum_reindex KH KW numCBlocks BIN
    (fun hv wv cbv e =>
      miVal s₀ Input BHW BIN batch_dim in_height in_width IBS IIFS IHS IWS OH OW
          (BIN * numCBlocks) SH SW PH PW hv wv (cbv * BIN) (Lane2D.decode l).1 e
        * mwVal s₀ Weight BIN OF (BIN * numCBlocks) OGD WOFS WIFS WHS WWS
          hv wv (cbv * BIN) e (Lane2D.decode l).2.1)]
  rfl

/-! ### Flat-memory coverage

`StmtList.FlattenOk` over the whole body (with the nest inlined) is one
enormous conjunction, so it is discharged segment by segment through the two
structural combinators below rather than in a single `simp`. The three
shape-indexed `Op` arms (`expandDim`'s `insertAxis`, `castFloat`'s
`toTileDType`, `dot`'s `batch ++ [M, N]`) are the canonical simp blind spot:
their `Op.FlattenOk` equations match only up to unfolding the *type index*, so
they are supplied as recipe lemmas **applied** (`exact`/`apply`), never put in
a simp set. -/

private theorem convIO_flattenOk_expandDim {dtype : TileDType} {shape : TileShape}
    (ax : Fin (shape.length + 1)) (a : Op dtype shape) :
    (Op.expandDim ax a).FlattenOk ↔ a.FlattenOk := by simp [Op.FlattenOk]

private theorem convIO_flattenOk_castFloat {shape : TileShape} (src dst : FloatDType)
    (a : Op src.toTileDType shape) :
    (Op.castFloat src dst a).FlattenOk ↔ a.FlattenOk := by simp [Op.FlattenOk]

private theorem convIO_flattenOk_dot {batch : TileShape} {M K N : Nat}
    (a : Op .real (batch ++ [M, K])) (b : Op .real (batch ++ [K, N])) :
    (Op.dot a b).FlattenOk ↔ a.FlattenOk ∧ b.FlattenOk := by simp [Op.FlattenOk]

private theorem convIO_flattenOk_cons (st : Stmt) (l : List Stmt)
    (h1 : Stmt.FlattenOk st) (h2 : StmtList.FlattenOk l) :
    StmtList.FlattenOk (st :: l) := ⟨h1, h2⟩

private theorem convIO_flattenOk_append : ∀ (l1 l2 : List Stmt),
    StmtList.FlattenOk l1 → StmtList.FlattenOk l2 → StmtList.FlattenOk (l1 ++ l2)
  | [], _, _, h2 => h2
  | _ :: rest, l2, h1, h2 => ⟨h1.1, convIO_flattenOk_append rest l2 h1.2 h2⟩

set_option maxHeartbeats 4000000 in
/-- The full conv2d surface sits inside the flat-memory bridge's covered
fragment (plain `ptrAdd` walks only — no `ptrSub`, no atomics, no block
pointers; the `forRange` / `forRangeDyn` clauses recurse into the nest). -/
private theorem conv2d_flattenOk (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF : Nat) :
    ((conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  rw [conv2d_body_split' Input Weight Output batch_dim in_feat_dim in_height in_width
    out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
    KH KW SH SW PH PW groups tf32 BHW BIN OF]
  refine convIO_flattenOk_append _ _ ?_ (convIO_flattenOk_cons _ _ ?_ ?_)
  · simp [convPreBody, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
    repeat' first
      | exact trivial
      | apply And.intro
      | exact (convIO_flattenOk_expandDim _ _).mpr (by simp [Op.FlattenOk])
  · simp [Stmt.FlattenOk, StmtList.FlattenOk, convInnerBody, Op.FlattenOk]
    repeat' first
      | exact trivial
      | apply And.intro
      | exact (convIO_flattenOk_expandDim _ _).mpr (by simp [Op.FlattenOk])
      | apply (convIO_flattenOk_castFloat _ _ _).mpr
      | apply (convIO_flattenOk_dot _ _).mpr
      | simp [Op.FlattenOk]
  · simp [convPostBody, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk]
    repeat' first
      | exact trivial
      | apply And.intro
      | exact (convIO_flattenOk_expandDim _ _).mpr (by simp [Op.FlattenOk])

/-! ### The terminal masked store on the `R` side

`conv2d_postLoop` already runs the three post-loop statements under the exact
stepper, but the skin needs (a) the `readMemAs .real` readback at the `MemCell`
layer and (b) the full outside-the-window frame, neither of which the exact
lemma exposes. Since the store is `.real`-typed the `R` run is the same run
(`convIO_writeMemTypedR_real_eq`); the scatter is re-derived here through the
`writeMemAsR` readback/frame family. -/

/-- The two post-loop address/mask expressions are cast-free, so they evaluate
identically under `evalOpR R` — this lets the exact eval computations of
`conv2d_postLoop` be reused verbatim on the `R` side. -/
private theorem convIO_evalR_outPtr (R : RoundingModel) (Output : RegionName)
    (OBS OOFS OHS OWS BHW OF : Nat) (u : BlockState) :
    evalOpR R (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Output)
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
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_width_offset"))))) u
      = evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Output)
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
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_width_offset"))))) u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

private theorem convIO_evalR_outMask (R : RoundingModel) (batch_dim OH OW BHW OF : Nat)
    (u : BlockState) :
    evalOpR R (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
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
        (Op.constNat OW))) u
      = evalOp (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
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
        (Op.constNat OW))) u := by
  simp only [evalOpR.eq_def, evalOp.eq_def]

set_option maxHeartbeats 4000000 in
/-- **R-postLoop**: from `convHInv` at `KH` (so `accum = convSpec`), the three
post statements terminate under `execR R`; every *active* output lane holds the
exact-ℝ cell `MemCell.of .real (convSpec …)` — a `.real` store has no rounding
event — with a per-cell frame outside the active output window. -/
private theorem conv2d_postLoopR (R : RoundingModel)
    (Input Weight Output : RegionName)
    (s0 : BlockState) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW OGD SH SW PH PW KH KW numCBlocks : Nat)
    (hOutInj : Function.Injective (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD))
    (st : BlockState)
    (hinv : convHInv s0 Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS OH OW (BIN * numCBlocks) OGD
        SH SW PH PW KW numCBlocks KH st) :
    ∃ sfin, stepStmtsR R (convPostBody Input Weight Output batch_dim OH OW OBS OOFS OHS OWS OGD BHW OF) st
        = some sfin
      ∧ (∀ idx : TileIndex [BHW, OF], active s0 BHW OF batch_dim OH OW OGD idx →
          sfin.mem Output (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD idx)
            = MemCell.of FloatDType.real.toTileDType
                (FloatDType.real.ofReal
                  (convSpec s0 Input Weight BHW BIN OF batch_dim in_height in_width
                    IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW (BIN * numCBlocks) OGD
                    SH SW PH PW KH KW numCBlocks idx.1 idx.2.1)))
      ∧ (∀ r o, (r ≠ Output ∨ ∀ idx : TileIndex [BHW, OF],
            active s0 BHW OF batch_dim OH OW OGD idx →
            o ≠ outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD idx) →
          sfin.mem r o = st.mem r o) := by
  obtain ⟨hctx, hacc⟩ := hinv
  obtain ⟨hpids, hundef, hmem, hgp, hgd, hod, hbo, hoh, how, hof, hIreg, hWreg⟩ := hctx
  set valueFn : TileIndex [BHW, OF] → ℝ := fun idx =>
    convSpec s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW (BIN * numCBlocks) OGD SH SW PH PW
      KH KW numCBlocks idx.1 idx.2.1 with hvalueFn
  set accT : Tile .real [BHW, OF] := ⟨fun idx : TileIndex [BHW, OF] => some (valueFn idx)⟩ with haccT
  have haccSpec : st.regs .real [BHW, OF] "accum" = some accT := by rw [hacc]; rfl
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
  unfold convPostBody
  erw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
    (by rw [convIO_evalR_outPtr]; exact hOutEval))]
  set s1 := st.setReg "Output" .ptr [BHW, OF] opT with hs1
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
  rw [stepStmtsR_cons_some (stepStmtR_assign_eq_some
    (by rw [convIO_evalR_outMask]; exact hMaskEval))]
  set s2 := s1.setReg "output_mask" .bool [BHW, OF] mT with hs2
  have haccS2 : s2.regs .real [BHW, OF] "accum" = some accT := by
    simp only [hs2, hs1, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true]
    rw [haccSpec]
  have hopS2 : s2.regs .ptr [BHW, OF] "Output" = some opT := by
    simp only [hs2, BlockState.setReg_ne_name, ne_eq, String.reduceEq, not_false_eq_true, hs1,
      BlockState.setReg_same]
  have hmS2 : s2.regs .bool [BHW, OF] "output_mask" = some mT := by
    simp only [hs2, BlockState.setReg_same]
  have hstore : stepStmtR R (Stmt.store .real [BHW, OF]
      (MemAccess.ptr (Op.ref TileDType.ptr [BHW, OF] "Output"))
      (Op.ref TileDType.real [BHW, OF] "accum")
      (MaskOpt.mask (Op.ref TileDType.bool [BHW, OF] "output_mask"))) s2
      = some ((TileShape.allIndices [BHW, OF]).foldl
          (fun acc idx => if active s0 BHW OF batch_dim OH OW OGD idx then
            acc.writeMemAsR R .real Output
              (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD idx) (accT.data idx)
          else acc) s2) := by
    simp only [stepStmtR, evalOpR_ref, haccS2, hopS2, hmS2, bind, Option.bind_some,
      Option.map_some]
    refine congrArg some (List.foldl_ext _ _ s2 (fun acc idx _ => ?_))
    by_cases hb : active s0 BHW OF batch_dim OH OW OGD idx
    · simp only [hmT, hb, decide_true, if_true, hopT, Region.cast_id,
        convIO_writeMemTypedR_real_eq]
    · simp only [hmT, hb, decide_false, if_false, Bool.false_eq_true]
  rw [stepStmtsR_cons_some hstore, stepStmtsR_nil]
  refine ⟨_, rfl, ?_, ?_⟩
  · intro idx hact
    rw [BlockState.scatter_memcell_R_prop_masked_nd R .real (region := Output) s2
        (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD)
        (fun idx => accT.data idx) (fun idx => active s0 BHW OF batch_dim OH OW OGD idx)
        hOutInj idx,
      if_pos hact]
    simp only [haccT, RoundingModel.storeValue_real, FloatDType.real_storeValue,
      WithBot.unbotD_some, hvalueFn]
  · intro r o hcond
    by_cases hr : r = Output
    · rcases hcond with hne | hno
      · exact absurd hr hne
      · rw [hr,
          BlockState.foldl_writeMemAsR_preserve_masked_prop R .real
            (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD)
            (fun idx => accT.data idx) (fun idx => active s0 BHW OF batch_dim OH OW OGD idx) o
            (TileShape.allIndices [BHW, OF]) (fun k _ hk he => hno k hk he.symm) s2]
        rfl
    · rw [BlockState.foldl_writeMemAsR_preserve_other_region R .real
        (outputOffset s0 BHW OF OH OW OBS OOFS OHS OWS OGD)
        (fun idx => accT.data idx) (fun idx => active s0 BHW OF batch_dim OH OW OGD idx) r hr o
        (TileShape.allIndices [BHW, OF]) s2]
      rfl

/-! ### The safety-walk context (weak, `undef`-free)

The skin's `hts` obligation quantifies over an **arbitrary** launch state, so
the trace-safety walk cannot assume the clean-`undef` precondition that
`ConvCtx` (and hence `conv2d_preLoop` / `conv2d_c_step`) carries. `ConvSafeCtx`
is the shape half of `ConvCtx`: the prefix-derived registers in pid form, plus
"there is *some* accumulator tile" (needed only for the inner body's steps to
succeed), with no `undef` / `mem` / value pins. Trace safety never needs a
loaded value — only the address and mask tiles, which are `undef`-independent. -/

private structure ConvSafeCtx (pid₀ pid₁ pid₂ : Nat) (Input Weight : RegionName)
    (BHW OF OH OW IGD OGD IBS IIFS WOFS : Nat) (s : BlockState) : Prop where
  group_pid : s.regs .nat [] "group_pid" = some (Tile.scalar pid₂)
  in_group_dim : s.regs .nat [] "in_group_dim" = some (Tile.scalar IGD)
  out_group_dim : s.regs .nat [] "out_group_dim" = some (Tile.scalar OGD)
  batch_offset : s.regs .nat [BHW] "batch_offset" =
    some (Tile.vec (fun i : Fin BHW => pBatch pid₀ OH OW BHW i.val))
  output_height_offset : s.regs .nat [BHW] "output_height_offset" =
    some (Tile.vec (fun i : Fin BHW => pHeight pid₀ OH OW BHW i.val))
  output_width_offset : s.regs .nat [BHW] "output_width_offset" =
    some (Tile.vec (fun i : Fin BHW => pWidth pid₀ OW BHW i.val))
  output_feat_offset : s.regs .nat [OF] "output_feat_offset" =
    some (Tile.vec (fun j : Fin OF => pFeat pid₁ OF j.val))
  inputReg : s.regs .ptr [BHW, 1] "Input" =
    some ⟨fun idx : TileIndex [BHW, 1] =>
      (Input.cast, pInputBase pid₀ pid₂ IBS IIFS OH OW BHW IGD idx.1.val)⟩
  weightReg : s.regs .ptr [1, OF] "Weight" =
    some ⟨fun idx : TileIndex [1, OF] =>
      (Weight.cast, pWeightBase pid₁ pid₂ WOFS OF OGD idx.2.1.val)⟩
  accum : ∃ accT : Tile .real [BHW, OF], s.regs .real [BHW, OF] "accum" = some accT

set_option maxHeartbeats 2000000 in
/-- **Weak preLoop**: from an *arbitrary* state the 14 prologue statements step
successfully to a `ConvSafeCtx` state. The value half of `conv2d_preLoop`
(clean `undef`, `mem = s.mem`, `accum = 0`) is dropped. -/
private theorem conv2d_preLoopW (Input Weight : RegionName)
    (in_feat_dim out_feat_dim out_height out_width IBS IIFS WOFS groups BHW OF BIN numCBlocks : Nat)
    (s : BlockState) (hIGD : in_feat_dim / groups = BIN * numCBlocks) :
    ∃ s', stepStmts (convPreBody Input Weight in_feat_dim out_feat_dim out_height out_width
        IBS IIFS WOFS groups BHW OF) s = some s'
      ∧ ConvSafeCtx (s.pids 0) (s.pids 1) (s.pids 2) Input Weight BHW OF out_height out_width
          (BIN * numCBlocks) (out_feat_dim / groups) IBS IIFS WOFS s' := by
  unfold convPreBody
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (by rw [evalOp_programId] :
        evalOp (Op.programId 0) s = _)),
      stepStmts.cons_some (stepStmt_assign_eq_some (by rw [evalOp_programId] :
        evalOp (Op.programId 1) _ = _)),
      stepStmts.cons_some (stepStmt_assign_eq_some (by rw [evalOp_programId] :
        evalOp (Op.programId 2) _ = _))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.floorDiv IntegralDType.nat
        Broadcast.nil (Op.constNat in_feat_dim) (Op.constNat groups)) _
        = some (Tile.scalar (in_feat_dim / groups)) from by
      simp [evalOp_floorDiv, evalOp_constNat, Tile.bop, IntegralDType.floorDiv]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some (show evalOp (Op.floorDiv IntegralDType.nat
        Broadcast.nil (Op.constNat out_feat_dim) (Op.constNat groups)) _
        = some (Tile.scalar (out_feat_dim / groups)) from by
      simp [evalOp_floorDiv, evalOp_constNat, Tile.bop, IntegralDType.floorDiv]))]
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
        (pidblock_eval _ BHW (s.pids 0) "batch_height_width_pid"
          (by simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids])))]
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
  refine ⟨_, rfl, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same, BlockState.setReg_pids]
  · rw [hIGD]; simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp [BlockState.setReg_ne_name, BlockState.setReg_same]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, Option.some.injEq]
    ext idx; simp only [pBatch, pBh, pBhw, Tile.vec, IntegralDType.nat_floorDiv]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, Option.some.injEq]
    ext idx
    simp only [pHeight, pBh, pBhw, Tile.vec, IntegralDType.nat_floorDiv, IntegralDType.nat_mod]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, Option.some.injEq]
    ext idx; simp only [pWidth, pBhw, Tile.vec, IntegralDType.nat_mod]
  · simp only [BlockState.setReg_ne_name, BlockState.setReg_same, ne_eq, String.reduceEq,
      not_false_eq_true, reduceCtorEq, Option.some.injEq]
    ext idx; simp only [pFeat, Tile.vec]
  · simp only [TileShape.insertAxis, BlockState.setReg_ne_name, BlockState.setReg_same,
      BlockState.setReg_ne_dtype, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
      Option.some.injEq]
    ext idx
    · rfl
    · simp only [pInputBase, pBatch, pBh, pBhw, hIGD, IntegralDType.nat_floorDiv]
  · simp only [TileShape.insertAxis, BlockState.setReg_ne_name, BlockState.setReg_same,
      BlockState.setReg_ne_dtype, ne_eq, String.reduceEq, not_false_eq_true, reduceCtorEq,
      Option.some.injEq]
    ext idx
    · rfl
    · simp only [pWeightBase, pFeat]
  · refine ⟨⟨fun _ : TileIndex [BHW, OF] => some (0 : ℝ)⟩, ?_⟩
    simp only [BlockState.setReg_same]

/-! ### Trace-safety helpers

Same story as `Op.FlattenOk`: the three shape-indexed `Op` arms need `SafeAtR`
recipes **applied**, never simp'd. `convIO_tsl_cons` is the cons form of
`Stmt.TraceSafeListR` specialised to a statement whose successor is already
known from the exact walk (every statement of this kernel is cast-free, so
the `R` successor *is* the exact one). -/

private theorem convIO_safeAtR_expandDim (R : RoundingModel) (bounds : RegionBounds)
    (s : BlockState) {dtype : TileDType} {shape : TileShape}
    (ax : Fin (shape.length + 1)) (a : Op dtype shape) :
    (Op.expandDim ax a).SafeAtR R bounds s ↔ a.SafeAtR R bounds s := by simp [Op.SafeAtR]

private theorem convIO_safeAtR_castFloat (R : RoundingModel) (bounds : RegionBounds)
    (s : BlockState) {shape : TileShape} (src dst : FloatDType) (a : Op src.toTileDType shape) :
    (Op.castFloat src dst a).SafeAtR R bounds s ↔ a.SafeAtR R bounds s := by simp [Op.SafeAtR]

private theorem convIO_safeAtR_dot (R : RoundingModel) (bounds : RegionBounds)
    (s : BlockState) {batch : TileShape} {M K N : Nat}
    (a : Op .real (batch ++ [M, K])) (b : Op .real (batch ++ [K, N])) :
    (Op.dot a b).SafeAtR R bounds s ↔ a.SafeAtR R bounds s ∧ b.SafeAtR R bounds s := by
  simp [Op.SafeAtR]

/-- `TraceSafeListR` cons with a **known** successor: the head statement is
cast-free, so its `R`-successor is the exact one, which the walk already
computed. -/
private theorem convIO_tsl_cons {R : RoundingModel} {bounds : RegionBounds} {st : Stmt}
    {rest : List Stmt} {s s' : BlockState}
    (hcf : ∀ u, stepStmtR R st u = stepStmt st u)
    (hstep : stepStmt st s = some s')
    (h1 : Stmt.TraceSafeR R bounds st s)
    (h2 : Stmt.TraceSafeListR R bounds rest s') :
    Stmt.TraceSafeListR R bounds (st :: rest) s :=
  Stmt.TraceSafeListR.cons_intro h1 (fun s'' hs'' => by
    rw [hcf s, hstep] at hs''
    obtain rfl := Option.some.inj hs''
    exact h2)

/-- A masked `.ptr` load with no `other` always evaluates once its pointer and
mask tiles do (the masked-off lanes fall back to `undef`, whatever it is) —
the `undef`-free sibling of `load_ptr_mask_clean`, for the safety walk. -/
private theorem convIO_load_mask_some {shape : TileShape}
    (ptrOp : Op .ptr shape) (maskOp : Op .bool shape) (s : BlockState)
    (ptrs : Tile .ptr shape) (mtile : Tile .bool shape)
    (hp : evalOp ptrOp s = some ptrs) (hm : evalOp maskOp s = some mtile) :
    ∃ v : Tile .real shape,
      evalOp (Op.load .real (MemAccess.ptr ptrOp) (MaskOpt.mask maskOp)) s = some v := by
  unfold evalOp
  simp only [hp, hm, Option.bind_eq_bind, Option.bind_some]
  exact ⟨_, rfl⟩

set_option maxHeartbeats 4000000 in
/-- **Weak inner-body step + trace safety.** One `c`-block iteration of the
innermost loop, from an *arbitrary* `ConvSafeCtx` state with `h`/`w` pinned:
the eleven statements are trace-safe (the only two memory accesses are the two
masked loads, whose *active* addresses are the skin's `read1` / `read2` windows
at flat step `convStepIdx hc wc cb`) and step successfully, re-establishing the
weak context. This is the shape half of `conv2d_c_step`, valid without the
clean-`undef` precondition. -/
private theorem conv2d_c_bodyW (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (bounds : RegionBounds) (pid₀ pid₁ pid₂ : Nat) (Input Weight : RegionName)
    (BHW BIN OF batch_dim in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS
      OH OW OGD SH SW PH PW KH KW numCBlocks : Nat)
    (hc wc cb : Nat) (hcLt : hc < KH) (hwLt : wc < KW) (hcbLt : cb < numCBlocks)
    (s : BlockState)
    (hctx : ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD
      IBS IIFS WOFS s)
    (hh : s.regs .nat [] "h" = some (Tile.scalar hc))
    (hw : s.regs .nat [] "w" = some (Tile.scalar wc))
    (hbr1 : ∀ (t : Fin (KH * KW * numCBlocks)) (j : Fin (BHW * BIN)),
      pInMask pid₀ batch_dim in_height in_width OH OW BHW (BIN * numCBlocks) SH SW PH PW
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / BIN) (j.val % BIN) →
      pInAddr pid₀ pid₂ IBS IIFS IHS IWS OH OW BHW (BIN * numCBlocks) SH SW PH PW
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / BIN) (j.val % BIN) < bounds Input)
    (hbr2 : ∀ (t : Fin (KH * KW * numCBlocks)) (j : Fin (BIN * OF)),
      pWMask pid₁ (BIN * numCBlocks) OGD OF (convStepCB numCBlocks t.val * BIN)
        (j.val / OF) (j.val % OF) →
      pWAddr pid₁ pid₂ WOFS WIFS WHS WWS OF OGD
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / OF) (j.val % OF) < bounds Weight) :
    Stmt.TraceSafeListR R bounds
        (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)
        (s.setReg "c" .nat [] (Tile.scalar (cb * BIN)))
      ∧ ∃ s', stepStmts (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS
              batch_dim in_height in_width PH PW SH SW)
            (s.setReg "c" .nat [] (Tile.scalar (cb * BIN))) = some s'
          ∧ ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD
              IBS IIFS WOFS s'
          ∧ s'.regs .nat [] "h" = some (Tile.scalar hc)
          ∧ s'.regs .nat [] "w" = some (Tile.scalar wc) := by
  obtain ⟨hgp, hgd, hod, hbo, hoh, how, hof, hIreg, hWreg, ⟨accT, hacc⟩⟩ := hctx
  -- per-lane window bounds at this nest iteration
  have hb1 : ∀ (ii : Fin BHW) (e : Fin BIN),
      pInMask pid₀ batch_dim in_height in_width OH OW BHW (BIN * numCBlocks) SH SW PH PW
        hc wc (cb * BIN) ii.val e.val →
      pInAddr pid₀ pid₂ IBS IIFS IHS IWS OH OW BHW (BIN * numCBlocks) SH SW PH PW
        hc wc (cb * BIN) ii.val e.val < bounds Input := by
    intro ii e hm
    have hI := hbr1 (convStepIdx ⟨hc, hcLt⟩ ⟨wc, hwLt⟩ ⟨cb, hcbLt⟩)
      (Lane2D.encode (ii, e, PUnit.unit))
    simp only [convStepIdx_h, convStepIdx_w, convStepIdx_cb, Lane2D.encode_div,
      Lane2D.encode_mod] at hI
    exact hI hm
  have hb2 : ∀ (e : Fin BIN) (jj : Fin OF),
      pWMask pid₁ (BIN * numCBlocks) OGD OF (cb * BIN) e.val jj.val →
      pWAddr pid₁ pid₂ WOFS WIFS WHS WWS OF OGD hc wc (cb * BIN) e.val jj.val < bounds Weight := by
    intro e jj hm
    have hW := hbr2 (convStepIdx ⟨hc, hcLt⟩ ⟨wc, hwLt⟩ ⟨cb, hcbLt⟩)
      (Lane2D.encode (e, jj, PUnit.unit))
    simp only [convStepIdx_h, convStepIdx_w, convStepIdx_cb, Lane2D.encode_div,
      Lane2D.encode_mod] at hW
    exact hW hm
  -- the loop-set state
  set u0 := s.setReg "c" .nat [] (Tile.scalar (cb * BIN)) with hu0
  have hck : u0.regs .nat [] "c" = some (Tile.scalar (cb * BIN)) := by simp [hu0]
  -- statement 0: input_feat_offset
  have E0 := stepStmt_assign_eq_some (name := "input_feat_offset")
    (input_feat_offset_eval u0 BIN (cb * BIN) hck)
  set u1 := u0.setReg "input_feat_offset" .nat [BIN]
    (Tile.vec (fun e : Fin BIN => cb * BIN + e.val)) with hu1
  -- statement 1: input_height_offset
  have E1 := stepStmt_assign_eq_some (name := "input_height_offset")
    (int_offset_eval u1 BHW PH SH (fun ii : Fin BHW => pHeight pid₀ OH OW BHW ii.val) hc
      "output_height_offset" "h" (by simp [hu1, hu0, hh]) (by simp [hu1, hu0, hoh]))
  set u2 := u1.setReg "input_height_offset" .int [BHW]
    (Tile.vec (fun ii : Fin BHW =>
      (hc : Int) - (PH : Int) + (SH : Int) * (pHeight pid₀ OH OW BHW ii.val : Int))) with hu2
  -- statement 2: input_width_offset
  have E2 := stepStmt_assign_eq_some (name := "input_width_offset")
    (int_offset_eval u2 BHW PW SW (fun ii : Fin BHW => pWidth pid₀ OW BHW ii.val) wc
      "output_width_offset" "w" (by simp [hu2, hu1, hu0, hw]) (by simp [hu2, hu1, hu0, how]))
  set u3 := u2.setReg "input_width_offset" .int [BHW]
    (Tile.vec (fun ii : Fin BHW =>
      (wc : Int) - (PW : Int) + (SW : Int) * (pWidth pid₀ OW BHW ii.val : Int))) with hu3
  -- statement 3: curr_input_pointer
  have E3 := stepStmt_assign_eq_some (name := "curr_input_pointer")
    (curr_input_pointer_eval BHW BIN Input
      (fun ii : Fin BHW => pInputBase pid₀ pid₂ IBS IIFS OH OW BHW (BIN * numCBlocks) ii.val)
      (cb * BIN) IIFS IHS IWS
      (fun ii : Fin BHW => pInH pid₀ OH OW BHW SH PH hc ii.val)
      (fun ii : Fin BHW => pInW pid₀ OW BHW SW PW wc ii.val) u3
      (by simp [hu3, hu2, hu1, hu0, hIreg])
      (by simp [hu3, hu2, hu1])
      (by simp only [hu3, hu2, BlockState.setReg_ne_name, BlockState.setReg_same,
            ne_eq, String.reduceEq, not_false_eq_true]; rfl)
      (by simp only [hu3, BlockState.setReg_same]; rfl))
  set cipT : Tile .ptr [BHW, BIN] := ⟨fun idx : TileIndex [BHW, BIN] =>
    (Input.cast, pInputBase pid₀ pid₂ IBS IIFS OH OW BHW (BIN * numCBlocks) idx.1.val
      + IIFS * (cb * BIN + idx.2.1.val)
      + ((IHS : Int) * pInH pid₀ OH OW BHW SH PH hc idx.1.val).toNat
      + ((IWS : Int) * pInW pid₀ OW BHW SW PW wc idx.1.val).toNat)⟩ with hcipT
  set u4 := u3.setReg "curr_input_pointer" .ptr [BHW, BIN] cipT with hu4
  -- statement 4: curr_weight_pointer
  have E4 := stepStmt_assign_eq_some (name := "curr_weight_pointer")
    (curr_weight_pointer_eval BIN OF WIFS WHS WWS Weight
      (fun jj : Fin OF => pWeightBase pid₁ pid₂ WOFS OF OGD jj.val) (cb * BIN) hc wc u4
      (by simp [hu4, hu3, hu2, hu1, hu0, hWreg])
      (by simp [hu4, hu3, hu2, hu1])
      (by simp [hu4, hu3, hu2, hu1, hu0, hh])
      (by simp [hu4, hu3, hu2, hu1, hu0, hw]))
  set cwpT : Tile .ptr [BIN, OF] := ⟨fun idx : TileIndex [BIN, OF] =>
    (Weight.cast, pWeightBase pid₁ pid₂ WOFS OF OGD idx.2.1.val
      + WIFS * (cb * BIN + idx.1.val) + WHS * hc + WWS * wc)⟩ with hcwpT
  set u5 := u4.setReg "curr_weight_pointer" .ptr [BIN, OF] cwpT with hu5
  -- statement 5: input_mask
  have E5 := stepStmt_assign_eq_some (name := "input_mask")
    (input_mask_eval BHW BIN batch_dim in_height in_width (BIN * numCBlocks) u5
      (fun ii : Fin BHW => pBatch pid₀ OH OW BHW ii.val) (cb * BIN)
      (fun ii : Fin BHW => pInH pid₀ OH OW BHW SH PH hc ii.val)
      (fun ii : Fin BHW => pInW pid₀ OW BHW SW PW wc ii.val)
      (by simp [hu5, hu4, hu3, hu2, hu1, hu0, hbo])
      (by simp [hu5, hu4, hu3, hu2, hu1])
      (by simp only [hu5, hu4, hu3, hu2, BlockState.setReg_ne_name, BlockState.setReg_same,
            ne_eq, String.reduceEq, not_false_eq_true]; rfl)
      (by simp only [hu5, hu4, hu3, BlockState.setReg_ne_name, BlockState.setReg_same,
            ne_eq, String.reduceEq, not_false_eq_true]; rfl)
      (by simp [hu5, hu4, hu3, hu2, hu1, hu0, hgd]))
  set imT : Tile .bool [BHW, BIN] := ⟨fun idx : TileIndex [BHW, BIN] =>
    decide (((((pBatch pid₀ OH OW BHW idx.1.val < batch_dim ∧
        cb * BIN + idx.2.1.val < BIN * numCBlocks) ∧
      (0 : Int) ≤ pInH pid₀ OH OW BHW SH PH hc idx.1.val) ∧
      pInH pid₀ OH OW BHW SH PH hc idx.1.val < (in_height : Int)) ∧
      (0 : Int) ≤ pInW pid₀ OW BHW SW PW wc idx.1.val) ∧
      pInW pid₀ OW BHW SW PW wc idx.1.val < (in_width : Int))⟩ with himT
  set u6 := u5.setReg "input_mask" .bool [BHW, BIN] imT with hu6
  -- statement 6: weight_mask
  have E6 := stepStmt_assign_eq_some (name := "weight_mask")
    (weight_mask_eval BIN OF (BIN * numCBlocks) OGD u6 (cb * BIN)
      (fun jj : Fin OF => pFeat pid₁ OF jj.val)
      (by simp [hu6, hu5, hu4, hu3, hu2, hu1])
      (by simp [hu6, hu5, hu4, hu3, hu2, hu1, hu0, hof])
      (by simp [hu6, hu5, hu4, hu3, hu2, hu1, hu0, hgd])
      (by simp [hu6, hu5, hu4, hu3, hu2, hu1, hu0, hod]))
  set wmT : Tile .bool [BIN, OF] := ⟨fun idx : TileIndex [BIN, OF] =>
    decide (cb * BIN + idx.1.val < BIN * numCBlocks ∧ pFeat pid₁ OF idx.2.1.val < OGD)⟩ with hwmT
  set u7 := u6.setReg "weight_mask" .bool [BIN, OF] wmT with hu7
  -- register readbacks at the load states
  have hcip7 : u7.regs .ptr [BHW, BIN] "curr_input_pointer" = some cipT := by
    simp [hu7, hu6, hu5, hu4]
  have him7 : u7.regs .bool [BHW, BIN] "input_mask" = some imT := by simp [hu7, hu6]
  -- statement 7: the masked input load
  obtain ⟨inBlock, hinBlock⟩ :=
    convIO_load_mask_some (Op.ref TileDType.ptr [BHW, BIN] "curr_input_pointer")
      (Op.ref TileDType.bool [BHW, BIN] "input_mask") u7 cipT imT
      (by rw [evalOp_ref]; exact hcip7) (by rw [evalOp_ref]; exact him7)
  have E7 := stepStmt_assign_eq_some (name := "input_block") hinBlock
  set u8 := u7.setReg "input_block" .real [BHW, BIN] inBlock with hu8
  have hcwp8 : u8.regs .ptr [BIN, OF] "curr_weight_pointer" = some cwpT := by
    simp [hu8, hu7, hu6, hu5]
  have hwm8 : u8.regs .bool [BIN, OF] "weight_mask" = some wmT := by simp [hu8, hu7]
  -- statement 8: the masked weight load
  obtain ⟨wBlock, hwBlock⟩ :=
    convIO_load_mask_some (Op.ref TileDType.ptr [BIN, OF] "curr_weight_pointer")
      (Op.ref TileDType.bool [BIN, OF] "weight_mask") u8 cwpT wmT
      (by rw [evalOp_ref]; exact hcwp8) (by rw [evalOp_ref]; exact hwm8)
  have E8 := stepStmt_assign_eq_some (name := "weight_block") hwBlock
  set u9 := u8.setReg "weight_block" .real [BIN, OF] wBlock with hu9
  -- statement 9: the pinned `if fp16:` branch — two `.real → .fp16` casts
  have hcastI : evalOp (Op.castFloat FloatDType.real FloatDType.fp16
      (Op.ref TileDType.real [BHW, BIN] "input_block")) u9
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (inBlock.data idx)⟩
              : Tile .fp16 [BHW, BIN]) := by
    rw [evalOp_castFloat, Option.bind_eq_bind]
    simp only [FloatDType.toTileDType_real]
    have hrefI : evalOp (Op.ref TileDType.real [BHW, BIN] "input_block") u9 = some inBlock := by
      rw [evalOp_ref]; simp [hu9, hu8]
    conv_lhs => arg 1; rw [hrefI]
    rfl
  set inFp : Tile .fp16 [BHW, BIN] :=
    ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (inBlock.data idx)⟩ with hinFp
  set u10 := u9.setReg "input_block" .fp16 [BHW, BIN] inFp with hu10
  have hcastW : evalOp (Op.castFloat FloatDType.real FloatDType.fp16
      (Op.ref TileDType.real [BIN, OF] "weight_block")) u10
      = some (⟨fun idx => FloatDType.real.cast FloatDType.fp16 (wBlock.data idx)⟩
              : Tile .fp16 [BIN, OF]) := by
    rw [evalOp_castFloat, Option.bind_eq_bind]
    simp only [FloatDType.toTileDType_real]
    have hrefW : evalOp (Op.ref TileDType.real [BIN, OF] "weight_block") u10 = some wBlock := by
      rw [evalOp_ref]; simp [hu10, hu9]
    conv_lhs => arg 1; rw [hrefW]
    rfl
  set wFp : Tile .fp16 [BIN, OF] :=
    ⟨fun idx => FloatDType.real.cast FloatDType.fp16 (wBlock.data idx)⟩ with hwFp
  set u11 := u10.setReg "weight_block" .fp16 [BIN, OF] wFp with hu11
  have E9 : stepStmt (Stmt.ifThen (Op.constBool Bool.true)
      [Stmt.assign .fp16 [BHW, BIN] "input_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BHW, BIN] "input_block")),
        Stmt.assign .fp16 [BIN, OF] "weight_block"
          (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BIN, OF] "weight_block"))])
      u9 = some u11 := by
    have hif : stepStmt (Stmt.ifThen (Op.constBool Bool.true)
        [Stmt.assign .fp16 [BHW, BIN] "input_block"
            (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BHW, BIN] "input_block")),
          Stmt.assign .fp16 [BIN, OF] "weight_block"
            (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BIN, OF] "weight_block"))]) u9
        = stepStmts [Stmt.assign .fp16 [BHW, BIN] "input_block"
            (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BHW, BIN] "input_block")),
          Stmt.assign .fp16 [BIN, OF] "weight_block"
            (Op.castFloat FloatDType.real FloatDType.fp16 (Op.ref TileDType.real [BIN, OF] "weight_block"))] u9 := by
      simp only [stepStmt, evalOp]
      rfl
    rw [hif, stepStmts.cons_some (stepStmt_assign_eq_some (dtype := TileDType.fp16) hcastI),
      stepStmts.cons_some (stepStmt_assign_eq_some (dtype := TileDType.fp16) hcastW), stepStmts.nil]
  -- statement 10: accum += tl.dot (the fp16 round trip is the identity at ℝ)
  have hdotI : evalOp (Op.castFloat FloatDType.fp16 FloatDType.real
      (Op.ref TileDType.fp16 [BHW, BIN] "input_block")) u11 = some inBlock :=
    fp16_roundtrip_eval "input_block" u11 inBlock (by simp [hu11, hu10, hinFp])
  have hdotW : evalOp (Op.castFloat FloatDType.fp16 FloatDType.real
      (Op.ref TileDType.fp16 [BIN, OF] "weight_block")) u11 = some wBlock :=
    fp16_roundtrip_eval "weight_block" u11 wBlock (by simp [hu11, hwFp])
  have hdot : evalOp (Op.dot (batch := [])
      (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BHW, BIN] "input_block"))
      (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BIN, OF] "weight_block"))) u11
      = some (Tile.dot [] inBlock wBlock) := by
    have key : ∀ (a : Option (Tile .real [BHW, BIN])) (b : Option (Tile .real [BIN, OF])),
        a = some inBlock → b = some wBlock →
        (a.bind (fun vx => b.bind (fun vy => some (Tile.dot [] vx vy))))
          = some (Tile.dot [] inBlock wBlock) := by
      rintro a b rfl rfl; rfl
    rw [evalOp_dot]
    exact key _ _ hdotI hdotW
  have haccU11 : u11.regs .real [BHW, OF] "accum" = some accT := by
    simp [hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hacc]
  have haccdot : evalOp (Op.add NumericDType.real
        (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
        (Op.ref TileDType.real [BHW, OF] "accum")
        (Op.dot (batch := [])
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BHW, BIN] "input_block"))
          (Op.castFloat FloatDType.fp16 FloatDType.real (Op.ref TileDType.fp16 [BIN, OF] "weight_block")))) u11
      = some (Tile.bop NumericDType.real.add
          (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) accT (Tile.dot [] inBlock wBlock)) := by
    rw [evalOp_add]
    have hrefacc : evalOp (Op.ref TileDType.real [BHW, OF] "accum") u11 = some accT := by
      rw [evalOp_ref]; exact haccU11
    have key : ∀ (a b : Option (Tile .real [BHW, OF])),
        a = some accT → b = some (Tile.dot [] inBlock wBlock) →
        (a.bind (fun vz => b.bind (fun vd => some
          (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) vz vd))))
        = some (Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) accT (Tile.dot [] inBlock wBlock)) := by
      rintro a b rfl rfl; rfl
    exact key _ _ hrefacc hdot
  have E10 := stepStmt_assign_eq_some (name := "accum") haccdot
  set u12 := u11.setReg "accum" .real [BHW, OF]
    (Tile.bop NumericDType.real.add (Broadcast.consSame (Broadcast.consSame Broadcast.nil))
      accT (Tile.dot [] inBlock wBlock)) with hu12
  -- the cast-free collapse, statement by statement
  have castFreeAll := convIO_innerBody_stmt_castFree R hfp16 BHW BIN OF IIFS IHS IWS WIFS WHS WWS
    batch_dim in_height in_width PH PW SH SW
  constructor
  · -- trace safety of the eleven statements
    unfold convInnerBody
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E0 (by
      simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E1 (by
      simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E2 (by
      simp [Stmt.TraceSafeR, Op.SafeAtR]) ?_
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E3 ?_ ?_
    · simp only [Stmt.TraceSafeR]
      repeat' first
        | exact trivial
        | apply And.intro
        | exact (convIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
        | simp [Op.SafeAtR]
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E4 ?_ ?_
    · simp only [Stmt.TraceSafeR]
      repeat' first
        | exact trivial
        | apply And.intro
        | exact (convIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
        | simp [Op.SafeAtR]
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E5 ?_ ?_
    · simp only [Stmt.TraceSafeR]
      repeat' first
        | exact trivial
        | apply And.intro
        | exact (convIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
        | simp [Op.SafeAtR]
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E6 ?_ ?_
    · simp only [Stmt.TraceSafeR]
      repeat' first
        | exact trivial
        | apply And.intro
        | exact (convIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
        | simp [Op.SafeAtR]
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E7 ?_ ?_
    · -- the masked input load: active addresses are the skin's `read1` window
      simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro ptrs hptrs idx hact
      rw [evalOpR_ref, hcip7] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hmasks, hmi⟩ := hact
      rw [evalOpR_ref, him7] at hmasks
      obtain rfl := Option.some.inj hmasks
      show pInputBase pid₀ pid₂ IBS IIFS OH OW BHW (BIN * numCBlocks) idx.1.val
          + IIFS * (cb * BIN + idx.2.1.val)
          + ((IHS : Int) * pInH pid₀ OH OW BHW SH PH hc idx.1.val).toNat
          + ((IWS : Int) * pInW pid₀ OW BHW SW PW wc idx.1.val).toNat
        < bounds (Region.cast Input)
      simp only [Region.cast_id]
      exact hb1 idx.1 idx.2.1 (of_decide_eq_true hmi)
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E8 ?_ ?_
    · -- the masked weight load: active addresses are the skin's `read2` window
      simp only [Stmt.TraceSafeR, Op.SafeAtR, MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR,
        memAccessActiveAddressSafeR]
      refine ⟨trivial, trivial, ?_⟩
      intro ptrs hptrs idx hact
      rw [evalOpR_ref, hcwp8] at hptrs
      obtain rfl := Option.some.inj hptrs
      obtain ⟨masks, hmasks, hmi⟩ := hact
      rw [evalOpR_ref, hwm8] at hmasks
      obtain rfl := Option.some.inj hmasks
      show pWeightBase pid₁ pid₂ WOFS OF OGD idx.2.1.val + WIFS * (cb * BIN + idx.1.val)
          + WHS * hc + WWS * wc < bounds (Region.cast Weight)
      simp only [Region.cast_id]
      exact hb2 idx.1 idx.2.1 (of_decide_eq_true hmi)
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E9 ?_ ?_
    · -- the pinned `if fp16:` branch: two register-only casts
      simp only [Stmt.TraceSafeR]
      refine ⟨by simp [Op.SafeAtR], ?_⟩
      simp only [evalOpR, Tile.scalar, if_true]
      refine Stmt.TraceSafeListR.of_forall _ _ ?_
      intro st hst t
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hst
      rcases hst with rfl | rfl <;>
        simp only [Stmt.TraceSafeR] <;>
        refine (convIO_safeAtR_castFloat R bounds t _ _ _).mpr ?_ <;>
        simp [Op.SafeAtR]
    refine convIO_tsl_cons (castFreeAll _ (by unfold convInnerBody; simp)) E10 ?_ ?_
    · simp only [Stmt.TraceSafeR]
      repeat' first
        | exact trivial
        | apply And.intro
        | apply (convIO_safeAtR_castFloat R bounds _ _ _ _).mpr
        | apply (convIO_safeAtR_dot R bounds _ _ _).mpr
        | simp [Op.SafeAtR]
    exact Stmt.TraceSafeListR.nil_intro
  · -- the step itself
    refine ⟨u12, ?_, ?_, ?_, ?_⟩
    · unfold convInnerBody
      rw [stepStmts.cons_some E0, stepStmts.cons_some E1, stepStmts.cons_some E2,
        stepStmts.cons_some E3, stepStmts.cons_some E4, stepStmts.cons_some E5,
        stepStmts.cons_some E6, stepStmts.cons_some E7, stepStmts.cons_some E8,
        stepStmts.cons_some E9, stepStmts.cons_some E10, stepStmts.nil]
    · refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hgp]
      · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hgd]
      · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hod]
      · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hbo]
      · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hoh]
      · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, how]
      · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hof]
      · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hIreg]
      · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hWreg]
      · refine ⟨Tile.bop NumericDType.real.add
            (Broadcast.consSame (Broadcast.consSame Broadcast.nil)) accT (Tile.dot [] inBlock wBlock), ?_⟩
        simp only [hu12, BlockState.setReg_same]
    · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hh]
    · simp [hu12, hu11, hu10, hu9, hu8, hu7, hu6, hu5, hu4, hu3, hu2, hu1, hu0, hw]

/-! ### Weak loop drivers and the nested `TraceSafeR` walk

Three levels, each the same two-part obligation: the level's body is
trace-safe, and it steps successfully re-establishing the weak invariant. The
`c` level is `forRangeDyn` (stop register `in_group_dim`, pid-free), the `w`
and `h` levels are unit-step static `forRange`s. The nest's safety invariants
are *shape-only* — no `undef`, no value pins — which is what makes them
establishable from the skin's arbitrary launch state. -/

section NestSafety

variable (R : RoundingModel) (bounds : RegionBounds) (pid₀ pid₁ pid₂ : Nat)
  (Input Weight : RegionName)
  (BHW BIN OF batch_dim in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS
    OH OW OGD SH SW PH PW KH KW numCBlocks : Nat)

set_option maxHeartbeats 1000000 in
/-- **Weak inner `c`-loop driver**: the dynamic `forRangeDyn "c" 0 in_group_dim
BLOCK_IN` runs to completion from any `ConvSafeCtx` state and preserves it. -/
private theorem conv2d_c_loopW (hfp16 : R.round .fp16 = id) (hBIN : 0 < BIN)
    (hc wc : Nat) (hcLt : hc < KH) (hwLt : wc < KW) (s : BlockState)
    (hctx : ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD
      IBS IIFS WOFS s)
    (hh : s.regs .nat [] "h" = some (Tile.scalar hc))
    (hw : s.regs .nat [] "w" = some (Tile.scalar wc))
    (hbr1 : ∀ (t : Fin (KH * KW * numCBlocks)) (j : Fin (BHW * BIN)),
      pInMask pid₀ batch_dim in_height in_width OH OW BHW (BIN * numCBlocks) SH SW PH PW
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / BIN) (j.val % BIN) →
      pInAddr pid₀ pid₂ IBS IIFS IHS IWS OH OW BHW (BIN * numCBlocks) SH SW PH PW
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / BIN) (j.val % BIN) < bounds Input)
    (hbr2 : ∀ (t : Fin (KH * KW * numCBlocks)) (j : Fin (BIN * OF)),
      pWMask pid₁ (BIN * numCBlocks) OGD OF (convStepCB numCBlocks t.val * BIN)
        (j.val / OF) (j.val % OF) →
      pWAddr pid₁ pid₂ WOFS WIFS WHS WWS OF OGD
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / OF) (j.val % OF) < bounds Weight) :
    Stmt.TraceSafeR R bounds (Stmt.forRangeDyn "c" (Op.constNat 0)
        (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
        (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)) s
      ∧ ∃ s', stepStmt (Stmt.forRangeDyn "c" (Op.constNat 0)
          (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
          (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)) s
          = some s'
        ∧ ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD
            IBS IIFS WOFS s'
        ∧ s'.regs .nat [] "h" = some (Tile.scalar hc)
        ∧ s'.regs .nat [] "w" = some (Tile.scalar wc) := by
  -- the shape-only loop invariant
  set P : Nat → BlockState → Prop := fun i st =>
    ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD IBS IIFS WOFS st ∧
    i = i / BIN * BIN ∧ i ≤ BIN * numCBlocks ∧
    st.regs .nat [] "h" = some (Tile.scalar hc) ∧
    st.regs .nat [] "w" = some (Tile.scalar wc) with hP
  have hP0 : P 0 s := ⟨hctx, by simp, by omega, hh, hw⟩
  -- one iteration: safe and stepping
  have hbody : ∀ i st, i < BIN * numCBlocks → P i st →
      Stmt.TraceSafeListR R bounds
        (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)
        (st.setReg "c" .nat [] (Tile.scalar i)) ∧
      ∃ st', stepStmts (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS
            batch_dim in_height in_width PH PW SH SW)
          (st.setReg "c" .nat [] (Tile.scalar i)) = some st' ∧ P (i + BIN) st' := by
    intro i st hilt hinv
    obtain ⟨hctxi, hi, hile, hhi, hwi⟩ := hinv
    have hcbLt : i / BIN < numCBlocks := by
      rcases Nat.lt_or_ge (i / BIN) numCBlocks with hlt | hge
      · exact hlt
      · exfalso
        have hge' : BIN * numCBlocks ≤ i := by
          calc BIN * numCBlocks = numCBlocks * BIN := Nat.mul_comm _ _
            _ ≤ i / BIN * BIN := Nat.mul_le_mul_right _ hge
            _ = i := hi.symm
        omega
    have hieq : i / BIN * BIN = i := hi.symm
    have hstep := conv2d_c_bodyW R hfp16 bounds pid₀ pid₁ pid₂ Input Weight
      BHW BIN OF batch_dim in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS
      OH OW OGD SH SW PH PW KH KW numCBlocks hc wc (i / BIN) hcLt hwLt hcbLt st hctxi hhi hwi
      hbr1 hbr2
    rw [hieq] at hstep
    obtain ⟨hsafe, st', hrun, hctx', hh', hw'⟩ := hstep
    refine ⟨hsafe, st', hrun, hctx', ?_, ?_, hh', hw'⟩
    · have hdiv : (i + BIN) / BIN = i / BIN + 1 := Nat.add_div_right i hBIN
      rw [hdiv, Nat.succ_mul]
      omega
    · calc i + BIN = (i / BIN + 1) * BIN := by rw [Nat.succ_mul]; omega
        _ ≤ numCBlocks * BIN := Nat.mul_le_mul_right _ hcbLt
        _ = BIN * numCBlocks := Nat.mul_comm _ _
  -- resolve the dynamic stop register
  have hgd : s.regs .nat [] "in_group_dim" = some (Tile.scalar (BIN * numCBlocks)) :=
    hctx.in_group_dim
  constructor
  · simp only [Stmt.TraceSafeR]
    refine ⟨by simp [Op.SafeAtR], by simp [Op.SafeAtR], by simp [Op.SafeAtR], ?_⟩
    rw [show evalOpR R (Op.constNat 0) s = some (Tile.scalar 0) from by simp only [evalOpR],
      show evalOpR R (Op.ref TileDType.nat [] "in_group_dim") s
          = some (Tile.scalar (BIN * numCBlocks)) from by rw [evalOpR_ref]; exact hgd,
      show evalOpR R (Op.constNat BIN) s = some (Tile.scalar BIN) from by simp only [evalOpR]]
    exact Stmt.forRangeTraceSafeR_inv R bounds "c" (BIN * numCBlocks) BIN _ P
      (fun i st hlt hPi => by
        obtain ⟨hsafe, st', hrun, hP'⟩ := hbody i st hlt hPi
        exact ⟨hsafe, st', by
          rw [convIO_innerBody_castFree R hfp16 BHW BIN OF IIFS IHS IWS WIFS WHS WWS
            batch_dim in_height in_width PH PW SH SW]
          exact hrun, hP'⟩)
      0 s hP0
  · have hresolve : stepStmt (Stmt.forRangeDyn "c" (Op.constNat 0)
        (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
        (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)) s
        = stepForRangeAux "c" 0 (BIN * numCBlocks) BIN
            (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW) s := by
      rw [stepForRangeAux.forRangeDyn_unfold]
      simp only [evalOp_constNat, evalOp_ref, hgd, Tile.scalar, Option.bind_some, bind]
    rw [hresolve]
    obtain ⟨final, s_final, haux, hfinal, hPfinal⟩ :=
      forRangeAux_inv (idx := "c") (stop := BIN * numCBlocks) (step := BIN) (P := P)
        (by omega) (fun i st hlt hPi => (hbody i st hlt hPi).2) 0 s hP0
    obtain ⟨hctxF, _, hileF, hhF, hwF⟩ := hPfinal
    exact ⟨s_final, haux, hctxF, hhF, hwF⟩

set_option maxHeartbeats 1000000 in
/-- **Weak middle `w`-loop driver + safety**: `forRange "w" 0 KW 1` over the
single `c`-loop statement. -/
private theorem conv2d_w_loopW (hfp16 : R.round .fp16 = id) (hBIN : 0 < BIN)
    (hc : Nat) (hcLt : hc < KH) (s : BlockState)
    (hctx : ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD
      IBS IIFS WOFS s)
    (hh : s.regs .nat [] "h" = some (Tile.scalar hc))
    (hbr1 : ∀ (t : Fin (KH * KW * numCBlocks)) (j : Fin (BHW * BIN)),
      pInMask pid₀ batch_dim in_height in_width OH OW BHW (BIN * numCBlocks) SH SW PH PW
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / BIN) (j.val % BIN) →
      pInAddr pid₀ pid₂ IBS IIFS IHS IWS OH OW BHW (BIN * numCBlocks) SH SW PH PW
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / BIN) (j.val % BIN) < bounds Input)
    (hbr2 : ∀ (t : Fin (KH * KW * numCBlocks)) (j : Fin (BIN * OF)),
      pWMask pid₁ (BIN * numCBlocks) OGD OF (convStepCB numCBlocks t.val * BIN)
        (j.val / OF) (j.val % OF) →
      pWAddr pid₁ pid₂ WOFS WIFS WHS WWS OF OGD
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / OF) (j.val % OF) < bounds Weight) :
    Stmt.TraceSafeR R bounds (Stmt.forRange "w" 0 KW 1
        [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
          (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]) s
      ∧ ∃ s', stepStmt (Stmt.forRange "w" 0 KW 1
          [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
            (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]) s
          = some s'
        ∧ ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD
            IBS IIFS WOFS s'
        ∧ s'.regs .nat [] "h" = some (Tile.scalar hc) := by
  set Q : Nat → BlockState → Prop := fun _ st =>
    ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD IBS IIFS WOFS st ∧
    st.regs .nat [] "h" = some (Tile.scalar hc) with hQ
  have hQ0 : Q 0 s := ⟨hctx, hh⟩
  have hbody : ∀ wc st, wc < KW → Q wc st →
      Stmt.TraceSafeListR R bounds
        [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
          (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]
        (st.setReg "w" .nat [] (Tile.scalar wc)) ∧
      ∃ st', stepStmts
          [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
            (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]
          (st.setReg "w" .nat [] (Tile.scalar wc)) = some st' ∧ Q (wc + 1) st' := by
    intro wc st hwLt hQi
    obtain ⟨hctxi, hhi⟩ := hQi
    set sw := st.setReg "w" .nat [] (Tile.scalar wc) with hsw
    have hctxw : ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD
        IBS IIFS WOFS sw := by
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · simp [hsw, hctxi.group_pid]
      · simp [hsw, hctxi.in_group_dim]
      · simp [hsw, hctxi.out_group_dim]
      · simp [hsw, hctxi.batch_offset]
      · simp [hsw, hctxi.output_height_offset]
      · simp [hsw, hctxi.output_width_offset]
      · simp [hsw, hctxi.output_feat_offset]
      · simp [hsw, hctxi.inputReg]
      · simp [hsw, hctxi.weightReg]
      · obtain ⟨accT, hacc⟩ := hctxi.accum
        exact ⟨accT, by simp [hsw, hacc]⟩
    obtain ⟨hsafe, s', hrun, hctx', hh', hw'⟩ :=
      conv2d_c_loopW R bounds pid₀ pid₁ pid₂ Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW OGD SH SW PH PW KH KW numCBlocks hfp16 hBIN
        hc wc hcLt hwLt sw hctxw (by simp [hsw, hhi]) (by simp [hsw]) hbr1 hbr2
    refine ⟨Stmt.TraceSafeListR.cons_intro hsafe (fun _ _ => Stmt.TraceSafeListR.nil_intro),
      s', ?_, hctx', hh'⟩
    rw [stepStmts.cons_some hrun, stepStmts.nil]
  constructor
  · simp only [Stmt.TraceSafeR]
    refine Stmt.forRangeTraceSafeR_inv R bounds "w" KW 1 _ Q (fun wc st hlt hQi => ?_) 0 s hQ0
    obtain ⟨hsafe, st', hrun, hQ'⟩ := hbody wc st hlt hQi
    refine ⟨hsafe, st', ?_, hQ'⟩
    rw [convIO_stepStmtsR_castFree_of_stmts R _ (by
      intro stmt hstmt t
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hstmt
      subst hstmt
      exact convIO_cLoopStmt_castFree R BIN _
        (convIO_innerBody_castFree R hfp16 BHW BIN OF IIFS IHS IWS WIFS WHS WWS
          batch_dim in_height in_width PH PW SH SW) t)]
    exact hrun
  · obtain ⟨s', hrun, hQ'⟩ := forRange_step1_exact (idx := "w") (P := Q) hQ0
      (fun wc st hlt hQi => (hbody wc st hlt hQi).2)
    exact ⟨s', hrun, hQ'.1, hQ'.2⟩

set_option maxHeartbeats 1000000 in
/-- **Weak outer `h`-loop driver + safety**: `forRange "h" 0 KH 1` over the
single `w`-loop statement — the whole nest. -/
private theorem conv2d_h_loopW (hfp16 : R.round .fp16 = id) (hBIN : 0 < BIN)
    (s : BlockState)
    (hctx : ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD
      IBS IIFS WOFS s)
    (hbr1 : ∀ (t : Fin (KH * KW * numCBlocks)) (j : Fin (BHW * BIN)),
      pInMask pid₀ batch_dim in_height in_width OH OW BHW (BIN * numCBlocks) SH SW PH PW
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / BIN) (j.val % BIN) →
      pInAddr pid₀ pid₂ IBS IIFS IHS IWS OH OW BHW (BIN * numCBlocks) SH SW PH PW
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / BIN) (j.val % BIN) < bounds Input)
    (hbr2 : ∀ (t : Fin (KH * KW * numCBlocks)) (j : Fin (BIN * OF)),
      pWMask pid₁ (BIN * numCBlocks) OGD OF (convStepCB numCBlocks t.val * BIN)
        (j.val / OF) (j.val % OF) →
      pWAddr pid₁ pid₂ WOFS WIFS WHS WWS OF OGD
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / OF) (j.val % OF) < bounds Weight) :
    Stmt.TraceSafeR R bounds (Stmt.forRange "h" 0 KH 1
        [Stmt.forRange "w" 0 KW 1
          [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
            (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]]) s
      ∧ ∃ s', stepStmt (Stmt.forRange "h" 0 KH 1
          [Stmt.forRange "w" 0 KW 1
            [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
              (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]]) s
          = some s'
        ∧ ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD
            IBS IIFS WOFS s' := by
  set S : Nat → BlockState → Prop := fun _ st =>
    ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD IBS IIFS WOFS st
    with hS
  have hbody : ∀ hc st, hc < KH → S hc st →
      Stmt.TraceSafeListR R bounds
        [Stmt.forRange "w" 0 KW 1
          [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
            (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]]
        (st.setReg "h" .nat [] (Tile.scalar hc)) ∧
      ∃ st', stepStmts
          [Stmt.forRange "w" 0 KW 1
            [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim") (Op.constNat BIN)
              (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width PH PW SH SW)]]
          (st.setReg "h" .nat [] (Tile.scalar hc)) = some st' ∧ S (hc + 1) st' := by
    intro hc st hcLt hSi
    set sh := st.setReg "h" .nat [] (Tile.scalar hc) with hsh
    have hctxh : ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW (BIN * numCBlocks) OGD
        IBS IIFS WOFS sh := by
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
      · simp [hsh, hSi.group_pid]
      · simp [hsh, hSi.in_group_dim]
      · simp [hsh, hSi.out_group_dim]
      · simp [hsh, hSi.batch_offset]
      · simp [hsh, hSi.output_height_offset]
      · simp [hsh, hSi.output_width_offset]
      · simp [hsh, hSi.output_feat_offset]
      · simp [hsh, hSi.inputReg]
      · simp [hsh, hSi.weightReg]
      · obtain ⟨accT, hacc⟩ := hSi.accum
        exact ⟨accT, by simp [hsh, hacc]⟩
    obtain ⟨hsafe, s', hrun, hctx', hh'⟩ :=
      conv2d_w_loopW R bounds pid₀ pid₁ pid₂ Input Weight BHW BIN OF batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW OGD SH SW PH PW KH KW numCBlocks hfp16 hBIN
        hc hcLt sh hctxh (by simp [hsh]) hbr1 hbr2
    refine ⟨Stmt.TraceSafeListR.cons_intro hsafe (fun _ _ => Stmt.TraceSafeListR.nil_intro),
      s', ?_, hctx'⟩
    rw [stepStmts.cons_some hrun, stepStmts.nil]
  constructor
  · simp only [Stmt.TraceSafeR]
    refine Stmt.forRangeTraceSafeR_inv R bounds "h" KH 1 _ S (fun hc st hlt hSi => ?_) 0 s hctx
    obtain ⟨hsafe, st', hrun, hS'⟩ := hbody hc st hlt hSi
    refine ⟨hsafe, st', ?_, hS'⟩
    rw [convIO_stepStmtsR_castFree_of_stmts R _ (by
      intro stmt hstmt t
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hstmt
      subst hstmt
      exact convIO_wLoopStmt_castFree R hfp16 BHW BIN OF IIFS IHS IWS WIFS WHS WWS
        batch_dim in_height in_width PH PW SH SW KW t)]
    exact hrun
  · obtain ⟨s', hrun, hS'⟩ := forRange_step1_exact (idx := "h") (P := S) hctx
      (fun hc st hlt hSi => (hbody hc st hlt hSi).2)
    exact ⟨s', hrun, hS'⟩

end NestSafety

/-! ### The terminal store's safety, and the whole-kernel `TraceSafeR` -/

set_option maxHeartbeats 2000000 in
/-- pid-form evaluation of the post-loop `Output +=` pointer tile. -/
private theorem convIO_outPtrEvalW (Output : RegionName)
    (pid₀ pid₁ pid₂ BHW OF OH OW OBS OOFS OHS OWS OGD : Nat) (st : BlockState)
    (hbo : st.regs .nat [BHW] "batch_offset" =
      some (Tile.vec (fun i : Fin BHW => pBatch pid₀ OH OW BHW i.val)))
    (hoh : st.regs .nat [BHW] "output_height_offset" =
      some (Tile.vec (fun i : Fin BHW => pHeight pid₀ OH OW BHW i.val)))
    (how : st.regs .nat [BHW] "output_width_offset" =
      some (Tile.vec (fun i : Fin BHW => pWidth pid₀ OW BHW i.val)))
    (hof : st.regs .nat [OF] "output_feat_offset" =
      some (Tile.vec (fun j : Fin OF => pFeat pid₁ OF j.val)))
    (hgp : st.regs .nat [] "group_pid" = some (Tile.scalar pid₂))
    (hod : st.regs .nat [] "out_group_dim" = some (Tile.scalar OGD)) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ptrBase Output)
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
      = some (⟨fun idx : TileIndex [BHW, OF] =>
          (Output.cast, pOutAddr pid₀ pid₁ pid₂ BHW OF OH OW OBS OOFS OHS OWS OGD
            idx.1.val idx.2.1.val)⟩ : Tile .ptr [BHW, OF]) := by
  simp only [evalOp.eq_def, hbo, hof, hoh, how, hgp, hod, Option.bind, Option.bind_eq_bind]
  refine congrArg some ?_
  ext idx
  · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex]
  · simp only [Tile.ptrAdd, Tile.bop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
      Broadcast.rightIndex, TileShape.dropInsertedIndex, NumericDType.add, NumericDType.mul,
      pOutAddr, pBatch, pBh, pBhw, pHeight, pWidth, pFeat]
    ring

set_option maxHeartbeats 2000000 in
/-- pid-form evaluation of the post-loop `output_mask` tile. -/
private theorem convIO_outMaskEvalW
    (pid₀ pid₁ BHW OF batch_dim OH OW OGD : Nat) (st : BlockState)
    (hbo : st.regs .nat [BHW] "batch_offset" =
      some (Tile.vec (fun i : Fin BHW => pBatch pid₀ OH OW BHW i.val)))
    (hoh : st.regs .nat [BHW] "output_height_offset" =
      some (Tile.vec (fun i : Fin BHW => pHeight pid₀ OH OW BHW i.val)))
    (how : st.regs .nat [BHW] "output_width_offset" =
      some (Tile.vec (fun i : Fin BHW => pWidth pid₀ OW BHW i.val)))
    (hof : st.regs .nat [OF] "output_feat_offset" =
      some (Tile.vec (fun j : Fin OF => pFeat pid₁ OF j.val)))
    (hod : st.regs .nat [] "out_group_dim" = some (Tile.scalar OGD)) :
    evalOp (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
      (Op.boolAnd (Broadcast.consSame (Broadcast.consR Broadcast.nil))
        (Op.boolAnd (Broadcast.consR (Broadcast.consL Broadcast.nil))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "batch_offset"))
            (Op.constNat batch_dim))
          (Op.lt ComparableDType.nat Broadcast.scalarR
            (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [OF] "output_feat_offset"))
            (Op.ref TileDType.nat [] "out_group_dim")))
        (Op.lt ComparableDType.nat Broadcast.scalarR
          (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_height_offset"))
          (Op.constNat OH)))
      (Op.lt ComparableDType.nat Broadcast.scalarR
        (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BHW] "output_width_offset"))
        (Op.constNat OW))) st
      = some (⟨fun idx : TileIndex [BHW, OF] =>
          decide (pActive pid₀ pid₁ BHW OF batch_dim OH OW OGD idx.1.val idx.2.1.val)⟩
          : Tile .bool [BHW, OF]) := by
  simp only [evalOp.eq_def, hbo, hof, hoh, how, hod, Option.bind, Option.bind_eq_bind]
  refine congrArg some ?_
  ext idx
  simp only [Tile.bop, Tile.cop, Tile.expandDim, Tile.vec, Tile.scalar, Broadcast.leftIndex,
    Broadcast.rightIndex, ComparableDType.lt, TileShape.dropInsertedIndex, pActive,
    pBatch, pBh, pBhw, pHeight, pWidth, pFeat, Bool.decide_and]
  rfl

set_option maxHeartbeats 2000000 in
/-- **Post-loop trace safety**: the two register-only assigns plus the terminal
masked `.real` store, whose *active* lanes hit exactly the skin's `write`
window (in bounds by `hbw`). -/
private theorem conv2d_postSafeR (R : RoundingModel) (bounds : RegionBounds)
    (Input Weight Output : RegionName)
    (pid₀ pid₁ pid₂ BHW OF batch_dim OH OW OBS OOFS OHS OWS IGD OGD IBS IIFS WOFS : Nat)
    (st : BlockState)
    (hctx : ConvSafeCtx pid₀ pid₁ pid₂ Input Weight BHW OF OH OW IGD OGD IBS IIFS WOFS st)
    (hbw : ∀ j : Fin (BHW * OF),
      pActive pid₀ pid₁ BHW OF batch_dim OH OW OGD (j.val / OF) (j.val % OF) →
      pOutAddr pid₀ pid₁ pid₂ BHW OF OH OW OBS OOFS OHS OWS OGD (j.val / OF) (j.val % OF)
        < bounds Output) :
    Stmt.TraceSafeListR R bounds
      (convPostBody Input Weight Output batch_dim OH OW OBS OOFS OHS OWS OGD BHW OF) st := by
  obtain ⟨hgp, hgd, hod, hbo, hoh, how, hof, hIreg, hWreg, hacc⟩ := hctx
  set opT : Tile .ptr [BHW, OF] := ⟨fun idx : TileIndex [BHW, OF] =>
    (Output.cast, pOutAddr pid₀ pid₁ pid₂ BHW OF OH OW OBS OOFS OHS OWS OGD
      idx.1.val idx.2.1.val)⟩ with hopT
  have E0 := stepStmt_assign_eq_some (name := "Output")
    (convIO_outPtrEvalW Output pid₀ pid₁ pid₂ BHW OF OH OW OBS OOFS OHS OWS OGD st
      hbo hoh how hof hgp hod)
  set v1 := st.setReg "Output" .ptr [BHW, OF] opT with hv1
  set mT : Tile .bool [BHW, OF] := ⟨fun idx : TileIndex [BHW, OF] =>
    decide (pActive pid₀ pid₁ BHW OF batch_dim OH OW OGD idx.1.val idx.2.1.val)⟩ with hmT
  have E1 := stepStmt_assign_eq_some (name := "output_mask")
    (convIO_outMaskEvalW pid₀ pid₁ BHW OF batch_dim OH OW OGD v1
      (by simp [hv1, hbo]) (by simp [hv1, hoh]) (by simp [hv1, how]) (by simp [hv1, hof])
      (by simp [hv1, hod]))
  set v2 := v1.setReg "output_mask" .bool [BHW, OF] mT with hv2
  have hcastAll := convIO_postBody_stmt_castFree R Input Weight Output batch_dim OH OW
    OBS OOFS OHS OWS OGD BHW OF
  unfold convPostBody
  refine convIO_tsl_cons (hcastAll _ (by unfold convPostBody; simp)) E0 ?_ ?_
  · simp only [Stmt.TraceSafeR]
    repeat' first
      | exact trivial
      | apply And.intro
      | exact (convIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
      | simp [Op.SafeAtR]
  refine convIO_tsl_cons (hcastAll _ (by unfold convPostBody; simp)) E1 ?_ ?_
  · simp only [Stmt.TraceSafeR]
    repeat' first
      | exact trivial
      | apply And.intro
      | exact (convIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
      | simp [Op.SafeAtR]
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR, MemAccess.SafeAtR, MaskOpt.SafeAtR, Op.SafeAtR,
    MaskOpt.ActiveR, MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR]
  refine ⟨trivial, trivial, trivial, ?_⟩
  intro ptrs hptrs idx hact
  rw [evalOpR_ref, show v2.regs .ptr [BHW, OF] "Output" = some opT from by
    simp [hv2, hv1]] at hptrs
  obtain rfl := Option.some.inj hptrs
  obtain ⟨masks, hmasks, hmi⟩ := hact
  rw [evalOpR_ref, show v2.regs .bool [BHW, OF] "output_mask" = some mT from by
    simp [hv2]] at hmasks
  obtain rfl := Option.some.inj hmasks
  show pOutAddr pid₀ pid₁ pid₂ BHW OF OH OW OBS OOFS OHS OWS OGD idx.1.val idx.2.1.val
    < bounds (Region.cast Output)
  simp only [Region.cast_id]
  have hb := hbw (Lane2D.encode (idx.1, idx.2.1, PUnit.unit))
  simp only [Lane2D.encode_div, Lane2D.encode_mod] at hb
  exact hb (of_decide_eq_true hmi)

set_option maxHeartbeats 4000000 in
/-- **The whole-kernel `TraceSafeR` walk.** Prologue (register-only), the
flattened nest (three nested `forRangeTraceSafeR_inv`s over the shape-only
invariants), then the terminal masked store. The three bound groups are the
skin's `read1` / `read2` / `write` windows, `read1` / `read2` guarded by
`mask1` / `mask2` and `write` by `writeMask`. -/
private theorem conv2d_traceSafeR (R : RoundingModel) (hfp16 : R.round .fp16 = id)
    (bounds : RegionBounds) (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF numCBlocks : Nat)
    (hBIN : 0 < BIN) (hIGD : in_feat_dim / groups = BIN * numCBlocks) (s : BlockState)
    (hbr1 : ∀ (t : Fin (KH * KW * numCBlocks)) (j : Fin (BHW * BIN)),
      pInMask (s.pids 0) batch_dim in_height in_width out_height out_width BHW
        (BIN * numCBlocks) SH SW PH PW (convStepH KW numCBlocks t.val)
        (convStepW KW numCBlocks t.val) (convStepCB numCBlocks t.val * BIN)
        (j.val / BIN) (j.val % BIN) →
      pInAddr (s.pids 0) (s.pids 2) IBS IIFS IHS IWS out_height out_width BHW
        (BIN * numCBlocks) SH SW PH PW (convStepH KW numCBlocks t.val)
        (convStepW KW numCBlocks t.val) (convStepCB numCBlocks t.val * BIN)
        (j.val / BIN) (j.val % BIN) < bounds Input)
    (hbr2 : ∀ (t : Fin (KH * KW * numCBlocks)) (j : Fin (BIN * OF)),
      pWMask (s.pids 1) (BIN * numCBlocks) (out_feat_dim / groups) OF
        (convStepCB numCBlocks t.val * BIN) (j.val / OF) (j.val % OF) →
      pWAddr (s.pids 1) (s.pids 2) WOFS WIFS WHS WWS OF (out_feat_dim / groups)
        (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
        (convStepCB numCBlocks t.val * BIN) (j.val / OF) (j.val % OF) < bounds Weight)
    (hbw : ∀ j : Fin (BHW * OF),
      pActive (s.pids 0) (s.pids 1) BHW OF batch_dim out_height out_width
        (out_feat_dim / groups) (j.val / OF) (j.val % OF) →
      pOutAddr (s.pids 0) (s.pids 1) (s.pids 2) BHW OF out_height out_width
        OBS OOFS OHS OWS (out_feat_dim / groups) (j.val / OF) (j.val % OF) < bounds Output) :
    ((conv2d_forward_surface Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups Bool.true tf32 BHW BIN OF).toAlgKernel).TraceSafeR
      R bounds s := by
  unfold Kernel.TraceSafeR
  rw [conv2d_body_split' Input Weight Output batch_dim in_feat_dim in_height in_width
    out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
    KH KW SH SW PH PW groups tf32 BHW BIN OF]
  refine Stmt.TraceSafeListR.append_intro _ _ ?_ ?_
  · -- the prologue: register-only at every state
    refine Stmt.TraceSafeListR.of_forall _ _ ?_
    intro stmt hstmt u
    simp only [convPreBody, List.mem_cons, List.not_mem_nil, or_false] at hstmt
    rcases hstmt with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl <;>
      · simp only [Stmt.TraceSafeR]
        repeat' first
          | exact trivial
          | apply And.intro
          | exact (convIO_safeAtR_expandDim R bounds _ _ _).mpr (by simp [Op.SafeAtR])
          | simp [Op.SafeAtR]
  · -- after the prologue: the nest, then the store tail
    intro s1 hs1
    obtain ⟨s1x, hpre, hctx0⟩ := conv2d_preLoopW Input Weight in_feat_dim out_feat_dim
      out_height out_width IBS IIFS WOFS groups BHW OF BIN numCBlocks s hIGD
    rw [convIO_preBody_castFree R Input Weight in_feat_dim out_feat_dim out_height out_width
        IBS IIFS WOFS groups BHW OF s, hpre] at hs1
    obtain rfl := Option.some.inj hs1
    obtain ⟨hsafeNest, sLoop, hrunNest, hctxL⟩ :=
      conv2d_h_loopW R bounds (s.pids 0) (s.pids 1) (s.pids 2) Input Weight
        BHW BIN OF batch_dim in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS
        out_height out_width (out_feat_dim / groups) SH SW PH PW KH KW numCBlocks
        hfp16 hBIN s1x hctx0 hbr1 hbr2
    refine convIO_tsl_cons (convIO_hLoopStmt_castFree R hfp16 BHW BIN OF IIFS IHS IWS
      WIFS WHS WWS batch_dim in_height in_width PH PW SH SW KH KW) hrunNest hsafeNest ?_
    exact conv2d_postSafeR R bounds Input Weight Output (s.pids 0) (s.pids 1) (s.pids 2)
      BHW OF batch_dim out_height out_width OBS OOFS OHS OWS (BIN * numCBlocks)
      (out_feat_dim / groups) IBS IIFS WOFS sLoop hctxL hbw

/-! ### The headline -/

set_option maxHeartbeats 4000000 in
/-- **The `⊨[R]` streaming headline (wave-5 S1 fold genre, flattened nest).**
For every rounding model `R` with `R.round .fp16 = id`, the faithful
`conv2d_forward` surface implements, on its `StreamMetaMasked3DKernelIO₂`
signature at `nMeta := 0`, the **ideal ℝ im2col convolution fold** over the
streamed tiles: every write-active output lane `l = (i, n)` holds

`∑_{t < KH·KW·numCBlocks} ∑_{e < BLOCK_IN} maskedInput(t)[i,e] · maskedWeight(t)[e,n]`

— exact real arithmetic, the two factors zeroed exactly where the kernel's own
padding / boundary masks are false. The triple nest is **flattened onto one
stream**: step `t` is nest iteration `(t / NC / KW, t / NC % KW, t % NC)` and
the skin never sees a loop (see the section docstring).

Layer map: the prologue and the whole nest are cast-free under `hfp16`, so
under `execR R` they collapse verbatim onto the exact stepper and the proven
`conv2d_preLoop` / `conv2d_c_loop` / `conv2d_w_loop` / `conv2d_h_loop`
invariant tower above is reused unchanged; only the masked terminal store is
re-proved on the `R` side (`conv2d_postLoopR`). The store is `.real`-typed, so
the skin's boundary quantization degenerates: the readback's `R.round .real` is
the identity (`round_real`).

Hypotheses, each inherited from the exact headline
`conv2d_output_summary` (see its docstring) except the first:

* `hfp16 : R.round .fp16 = id` — the file's declared fp16 modeling boundary
  (docstring *Modeling boundary*: "the in-model fp16 cast is the identity"),
  made explicit on the `⊨[R]` surface. The four `Op.castFloat` sites of the
  inner body (the pinned `if fp16:` branch's two widenings and the two
  `fp16 → real` re-widenings inside `tl.dot`) are `R.cast` rounding events under
  `execR R`; at a model that does not quantize the fp16 grid they collapse to
  the exact casts and the nest is cast-free. This is the *whole* fp16 content
  of the port: the exact surface already assumed it silently by working over ℝ.
* `hBIN : 0 < BIN` — the in-feat loop steps by `BLOCK_SIZE_IN_FEAT`; at `0` the
  loop never advances and the block index `i / BIN` is meaningless. Exact
  headline's `hBIN`.
* `hIGD : in_feat_dim / groups = BIN · numCBlocks` — the in-feat loop tiles the
  group-local channel axis exactly. Exact headline's `hIGD`.
* `hOutInj` — output-address injectivity of the write window, the pid-form
  spelling of the exact headline's `hOutInj : Function.Injective (outputOffset
  s …)` (the skin quantifies the launch state internally, so the state-indexed
  spelling is not expressible here). With colliding output lanes the per-lane
  readback would be last-writer-wins and the statement false. As in the exact
  headline it is carried as an open side condition, not discharged.

The exact headline's `hundef` is **not** a hypothesis here: the skin's triple
already pins `s₀.undef = fun _ _ => 0`, which is exactly the zero padding the
two `other=`-less masked loads rely on.

Relation to the exact surface: `conv2d_output_summary`
(`Realizes_without_Rounding`) above is retained unchanged; this `⊨[R]` face
restates the same convolution on the streaming skin, for every `R` at once (at
the `.real` grid the two faces carry the same exact cell). Both are kept per
the rounding-as-default doctrine. -/
specification triton_conv2d_fwd_io_correctness (R : RoundingModel)
    (Input Weight Output : RegionName)
    (batch_dim in_feat_dim in_height in_width out_feat_dim out_height out_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups : Nat) (tf32 : Bool) (BHW BIN OF numCBlocks : Nat)
    (hfp16 : R.round .fp16 = id) (hBIN : 0 < BIN)
    (hIGD : in_feat_dim / groups = BIN * numCBlocks)
    (hOutInj : ∀ pid₀ pid₁ pid₂ : Nat,
      Function.Injective (fun idx : TileIndex [BHW, OF] =>
        pOutAddr pid₀ pid₁ pid₂ BHW OF out_height out_width OBS OOFS OHS OWS
          (out_feat_dim / groups) idx.1.val idx.2.1.val)) :
    triton_conv2d_fwd_IO Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS
        OBS OOFS OHS OWS KH KW SH SW PH PW groups tf32 BHW BIN OF numCBlocks ⊨[R]
      fun pid₀ pid₁ _ _ xs ys l =>
        convStreamSum pid₀ pid₁ batch_dim in_height in_width out_height out_width
          (out_feat_dim / groups) SH SW PH PW KH KW BHW BIN OF numCBlocks xs ys l := by
  refine StreamMetaMasked3DKernelIO₂.ImplementsR.intro _ ?_ ?_ ?_
  · exact conv2d_flattenOk Input Weight Output batch_dim in_feat_dim in_height in_width
      out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      KH KW SH SW PH PW groups tf32 BHW BIN OF
  · -- the safety walk
    intro bounds s m xs ys _hm _hx _hy _hbm hbr1 hbr2 hbw
    simp only [triton_conv2d_fwd_IO] at hbr1 hbr2 hbw
    exact conv2d_traceSafeR R hfp16 bounds Input Weight Output batch_dim in_feat_dim
      in_height in_width out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS
      OBS OOFS OHS OWS KH KW SH SW PH PW groups tf32 BHW BIN OF numCBlocks hBIN hIGD s
      hbr1 hbr2 hbw
  · -- the rounded Hoare triple
    intro s₀ m xs ys hundef _hm hx hy
    simp only [triton_conv2d_fwd_IO] at hx hy ⊢
    have hundef' : ∀ rg o, s₀.undef rg o = 0 := fun rg o => by rw [hundef]
    have hOutInj' : Function.Injective (outputOffset s₀ BHW OF out_height out_width
        OBS OOFS OHS OWS (out_feat_dim / groups)) :=
      hOutInj (s₀.pids 0) (s₀.pids 1) (s₀.pids 2)
    -- the exact prologue and the exact nest (both cast-free, so they *are* the `R` run)
    obtain ⟨s1, hpre, hP0⟩ := conv2d_preLoop Input Weight Output batch_dim in_feat_dim
      in_height in_width out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS
      OBS OOFS OHS OWS KH KW SH SW PH PW groups tf32 BHW BIN OF numCBlocks s₀ hundef' hIGD
    obtain ⟨sLoop, hloop, hPloop⟩ := conv2d_h_loop s₀ Input Weight BHW BIN OF batch_dim
      in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
      out_height out_width (out_feat_dim / groups) SH SW PH PW KH KW numCBlocks hBIN s1 hP0
    obtain ⟨sfin, hTailR, hval, hframe⟩ := conv2d_postLoopR R Input Weight Output s₀
      BHW BIN OF batch_dim in_height in_width IBS IIFS IHS IWS WOFS WIFS WHS WWS
      OBS OOFS OHS OWS out_height out_width (out_feat_dim / groups) SH SW PH PW KH KW
      numCBlocks hOutInj' sLoop hPloop
    have hpre' : stepStmts (convPreBody Input Weight in_feat_dim out_feat_dim out_height
        out_width IBS IIFS WOFS groups BHW OF) s₀ = some s1 := by
      rw [← conv2d_take14_eq Input Weight Output batch_dim in_feat_dim in_height in_width
        out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS OBS OOFS OHS OWS
        KH KW SH SW PH PW groups tf32 BHW BIN OF]
      exact hpre
    have hLoopR : stepStmtR R (Stmt.forRange "h" 0 KH 1
        [Stmt.forRange "w" 0 KW 1
          [Stmt.forRangeDyn "c" (Op.constNat 0) (Op.ref TileDType.nat [] "in_group_dim")
            (Op.constNat BIN)
            (convInnerBody BHW BIN OF IIFS IHS IWS WIFS WHS WWS batch_dim in_height in_width
              PH PW SH SW)]]) s1 = some sLoop := by
      rw [convIO_hLoopStmt_castFree R hfp16 BHW BIN OF IIFS IHS IWS WIFS WHS WWS
        batch_dim in_height in_width PH PW SH SW KH KW]
      exact hloop
    refine ⟨sfin, ?_, ?_, ?_⟩
    · show execR R (conv2d_forward_surface Input Weight Output batch_dim in_feat_dim
          in_height in_width out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS
          WHS WWS OBS OOFS OHS OWS KH KW SH SW PH PW groups Bool.true tf32
          BHW BIN OF).toAlgKernel s₀ = some sfin
      unfold execR
      rw [conv2d_body_split' Input Weight Output batch_dim in_feat_dim in_height in_width
          out_feat_dim out_height out_width IBS IIFS IHS IWS WOFS WIFS WHS WWS
          OBS OOFS OHS OWS KH KW SH SW PH PW groups tf32 BHW BIN OF,
        stepStmtsR_append,
        convIO_preBody_castFree R Input Weight in_feat_dim out_feat_dim out_height out_width
          IBS IIFS WOFS groups BHW OF s₀,
        hpre', Option.bind_some, stepStmtsR_cons_some hLoopR]
      exact hTailR
    · -- the readback: `convSpec` at the decoded lane is the flattened stream fold
      intro j hj
      have hact : active s₀ BHW OF batch_dim out_height out_width (out_feat_dim / groups)
          (Lane2D.decode j) :=
        (pActive_iff s₀ BHW OF batch_dim out_height out_width (out_feat_dim / groups)
          (Lane2D.decode j)).mpr hj
      have haddr : pOutAddr (s₀.pids 0) (s₀.pids 1) (s₀.pids 2) BHW OF out_height out_width
            OBS OOFS OHS OWS (out_feat_dim / groups) (j.val / OF) (j.val % OF)
          = outputOffset s₀ BHW OF out_height out_width OBS OOFS OHS OWS
              (out_feat_dim / groups) (Lane2D.decode j) := rfl
      rw [haddr, convIO_readMemAs_real_of_cell (hval (Lane2D.decode j) hact),
        R.round_real_apply]
      refine congrArg _ ?_
      have hx' : ∀ (t : Fin (KH * KW * numCBlocks)) (i : Fin BHW) (e : Fin BIN),
          inMask s₀ batch_dim in_height in_width out_height out_width BHW (BIN * numCBlocks)
              SH SW PH PW (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
              (convStepCB numCBlocks t.val * BIN) i e →
          s₀.readMem Input (inAddr s₀ IBS IIFS IHS IWS out_height out_width BHW
              (BIN * numCBlocks) SH SW PH PW (convStepH KW numCBlocks t.val)
              (convStepW KW numCBlocks t.val) (convStepCB numCBlocks t.val * BIN) i e)
            = xs t (Lane2D.encode (i, e, PUnit.unit)) := by
        intro t i e hm
        have h := hx t (Lane2D.encode (i, e, PUnit.unit))
        simp only [Lane2D.encode_div, Lane2D.encode_mod] at h
        exact h ((pInMask_iff s₀ batch_dim in_height in_width out_height out_width BHW
          (BIN * numCBlocks) SH SW PH PW _ _ _ i e).mp hm)
      have hy' : ∀ (t : Fin (KH * KW * numCBlocks)) (e : Fin BIN) (n : Fin OF),
          wMask s₀ (BIN * numCBlocks) (out_feat_dim / groups) OF
              (convStepCB numCBlocks t.val * BIN) e n →
          s₀.readMem Weight (wAddr s₀ WOFS WIFS WHS WWS OF (out_feat_dim / groups)
              (convStepH KW numCBlocks t.val) (convStepW KW numCBlocks t.val)
              (convStepCB numCBlocks t.val * BIN) e n)
            = ys t (Lane2D.encode (e, n, PUnit.unit)) := by
        intro t e n hm
        have h := hy t (Lane2D.encode (e, n, PUnit.unit))
        simp only [Lane2D.encode_div, Lane2D.encode_mod] at h
        exact h ((pWMask_iff s₀ (BIN * numCBlocks) (out_feat_dim / groups) OF _ e n).mp hm)
      exact convSpec_eq_streamSum s₀ Input Weight batch_dim in_height in_width
        IBS IIFS IHS IWS WOFS WIFS WHS WWS out_height out_width (out_feat_dim / groups)
        SH SW PH PW KH KW BHW BIN OF numCBlocks xs ys hx' hy' j
    · -- the frame
      intro r o hcond
      have hcond' : r ≠ Output ∨ ∀ idx : TileIndex [BHW, OF],
          active s₀ BHW OF batch_dim out_height out_width (out_feat_dim / groups) idx →
          o ≠ outputOffset s₀ BHW OF out_height out_width OBS OOFS OHS OWS
            (out_feat_dim / groups) idx := by
        rcases hcond with hne | hno
        · exact Or.inl hne
        · refine Or.inr fun idx hactive => ?_
          have h := hno (Lane2D.encode idx) (by
            simp only [Lane2D.encode_div, Lane2D.encode_mod]
            exact (pActive_iff s₀ BHW OF batch_dim out_height out_width
              (out_feat_dim / groups) idx).mp hactive)
          simpa only [Lane2D.encode_div, Lane2D.encode_mod] using h
      rw [hframe r o hcond', hPloop.1.mem]

end IOFace

end VeriTile.Bench.TritonBenchG.TritonConv2dFwd
