# Spec sheet — `bench/tritonbench_g/triton_conv2d_fwd/TritonConv2dFwd.lean`

**Python source:** `bench/tritonbench_g/triton_conv2d_fwd/triton_conv2d_fwd.py`

## Public theorem: `conv2d_output_summary`

<details><summary>docstring</summary>

```
/-- The full conv2d forward surface lowers to the algorithm layer and realizes
the genuine convolution `convSpec` on every active output lane. The convolution
dot-accumulator, the per-block padding/boundary masking, and the masked
writeback are all proven; only the host launch / scheduling is trusted. -/
```
</details>

**Statement:**
```lean
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
          SH SW PH PW KH KW numCBlocks idx.1 idx.2.1)
```

**Assumptions / layout contracts:**
- `hBIN : 0 < BIN`
- `hundef : ∀ rg o, s.undef rg o = 0`
- `hIGD : in_feat_dim / groups = BIN * numCBlocks`

**Closed-form spec defs (transitive):** `outputOffset`, `conv2d_forward_surface`, `active`, `convSpec`, `batchIdx`, `featIdx`, `heightIdx`, `widthIdx`, `accH`, `bhIdx`, `bhwIdx`, `accW`, `accC`, `blockDot`, `miVal`, `mwVal`, `inMask`, `inAddr`, `wMask`, `wAddr`, `inHeightOff`, `inWidthOff`, `inputBase`, `weightBase`

<details><summary><code>outputOffset</code></summary>

```
/-- The output write address for tile lane `(i,j)`. -/
```
```lean
def outputOffset (s0 : BlockState) (BHW OF OH OW OBS OOFS OHS OWS OGD : Nat)
    (idx : TileIndex [BHW, OF]) : Nat :=
  OBS * batchIdx s0 OH OW BHW idx.1 +
    OOFS * (s0.pids 2 * OGD + featIdx s0 OF idx.2.1) +
    OHS * heightIdx s0 OH OW BHW idx.1 + OWS * widthIdx s0 OW BHW idx.1
```
</details>

<details><summary><code>conv2d_forward_surface</code></summary>

```
/-- Faithful transcription of `triton_conv2d_fwd.py`'s `conv2d_forward_kernel`. -/
```
```lean
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
```
</details>

<details><summary><code>active</code></summary>

```
/-- The output store-mask predicate for tile lane `(i,j)`. -/
```
```lean
def active (s0 : BlockState) (BHW OF batch_dim OH OW OGD : Nat) (idx : TileIndex [BHW, OF]) : Prop :=
  ((batchIdx s0 OH OW BHW idx.1 < batch_dim ∧ featIdx s0 OF idx.2.1 < OGD) ∧
    heightIdx s0 OH OW BHW idx.1 < OH) ∧ widthIdx s0 OW BHW idx.1 < OW

instance (s0 : BlockState) (BHW OF batch_dim OH OW OGD : Nat) (idx : TileIndex [BHW, OF]) :
    Decidable (active s0 BHW OF batch_dim OH OW OGD idx) := by unfold active; infer_instance
```
</details>

<details><summary><code>convSpec</code></summary>

```
/-- **Genuine conv2d spec**: every output cell equals the full im2col convolution
sum over the kernel window and input-feature axis. -/
```
```lean
noncomputable def convSpec (s0 : BlockState)
    (Input Weight : RegionName) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW KH KW numCBlocks : Nat)
    (i : Fin BHW) (j : Fin OF) : ℝ :=
  accH s0 Input Weight BHW BIN OF batch_dim in_height in_width
    IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW KW numCBlocks KH i j
```
</details>

<details><summary><code>batchIdx</code></summary>

```
/-- `batch_offset = bhIdx // out_height`. -/
```
```lean
def batchIdx (s0 : BlockState) (OH OW BHW : Nat) (i : Fin BHW) : Nat := bhIdx s0 OW BHW i / OH
```
</details>

<details><summary><code>featIdx</code></summary>

```
/-- `output_feat_offset = pid1 · OF + j`. -/
```
```lean
def featIdx (s0 : BlockState) (OF : Nat) (j : Fin OF) : Nat := s0.pids 1 * OF + j.val
```
</details>

<details><summary><code>heightIdx</code></summary>

```
/-- `output_height_offset = bhIdx % out_height`. -/
```
```lean
def heightIdx (s0 : BlockState) (OH OW BHW : Nat) (i : Fin BHW) : Nat := bhIdx s0 OW BHW i % OH
```
</details>

<details><summary><code>widthIdx</code></summary>

```
/-- `output_width_offset = bhwIdx % out_width`. -/
```
```lean
def widthIdx (s0 : BlockState) (OW BHW : Nat) (i : Fin BHW) : Nat := bhwIdx s0 BHW i % OW
```
</details>

<details><summary><code>accH</code></summary>

```
/-- Accumulator after `hCount` complete kernel-height iterations (each runs the
full kernel-width loop). The full convolution value is `accH … KH`. -/
```
```lean
noncomputable def accH (s0 : BlockState)
    (Input Weight : RegionName) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW KW numCBlocks : Nat)
    (hCount : Nat) (i : Fin BHW) (j : Fin OF) : ℝ :=
  (Finset.range hCount).sum fun h =>
    accW s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW numCBlocks h KW i j
```
</details>

<details><summary><code>bhIdx</code></summary>

```
/-- `batch_height_offset = bhwIdx // out_width`. -/
```
```lean
def bhIdx (s0 : BlockState) (OW BHW : Nat) (i : Fin BHW) : Nat := bhwIdx s0 BHW i / OW
```
</details>

<details><summary><code>bhwIdx</code></summary>

```
/-- Flattened BHW index of lane `i`: `pid0 · BHW + i`. -/
```
```lean
def bhwIdx (s0 : BlockState) (BHW : Nat) (i : Fin BHW) : Nat := s0.pids 0 * BHW + i.val
```
</details>

<details><summary><code>accW</code></summary>

```
/-- Accumulator after `wCount` complete kernel-width iterations within outer
iteration `h` (each width iteration runs the full `numCBlocks`-block c-loop). -/
```
```lean
noncomputable def accW (s0 : BlockState)
    (Input Weight : RegionName) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW numCBlocks : Nat)
    (h wCount : Nat) (i : Fin BHW) (j : Fin OF) : ℝ :=
  (Finset.range wCount).sum fun w =>
    accC s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW h w numCBlocks i j
```
</details>

<details><summary><code>accC</code></summary>

```
/-- Accumulator after `cbCount` complete input-feature blocks within the
`(h, w)` iteration (the innermost-loop partial value, on top of the `accHW`
prefix). The c-loop index for block `cb` is `cb · BIN`. -/
```
```lean
noncomputable def accC (s0 : BlockState)
    (Input Weight : RegionName) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW : Nat)
    (h w cbCount : Nat) (i : Fin BHW) (j : Fin OF) : ℝ :=
  (Finset.range cbCount).sum fun cb =>
    blockDot s0 Input Weight BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW h w (cb * BIN) i j
```
</details>

<details><summary><code>blockDot</code></summary>

```
/-- The per-block masked dot at output lane `(i,j)`, block `(h,w,c)`:
`Σ_{e<BIN} miVal(h,w,c,i,e) · mwVal(h,w,c,e,j)`. This is one `tl.dot`'s worth. -/
```
```lean
noncomputable def blockDot (s0 : BlockState)
    (Input Weight : RegionName) (BHW BIN OF batch_dim in_height in_width
      IBS IIFS IHS IWS WOFS WIFS WHS WWS OH OW IGD OGD SH SW PH PW : Nat)
    (h w c : Nat) (i : Fin BHW) (j : Fin OF) : ℝ :=
  Finset.univ.sum fun e : Fin BIN =>
    miVal s0 Input BHW BIN batch_dim in_height in_width IBS IIFS IHS IWS OH OW IGD SH SW PH PW h w c i e
      * mwVal s0 Weight BIN OF IGD OGD WOFS WIFS WHS WWS h w c e j
```
</details>

<details><summary><code>miVal</code></summary>

```
/-- The masked input value at lane `(i,e)`, block `(h,w,c)`: the loaded `Input`
cell when the input mask holds, else `0` (zero padding under clean `undef`). -/
```
```lean
noncomputable def miVal (s0 : BlockState)
    (Input : RegionName) (BHW BIN batch_dim in_height in_width
      IBS IIFS IHS IWS OH OW IGD SH SW PH PW : Nat)
    (h w c : Nat) (i : Fin BHW) (e : Fin BIN) : ℝ :=
  if inMask s0 batch_dim in_height in_width OH OW BHW IGD SH SW PH PW h w c i e then
    s0.readMem Input (inAddr s0 IBS IIFS IHS IWS OH OW BHW IGD SH SW PH PW h w c i e)
  else 0
```
</details>

<details><summary><code>mwVal</code></summary>

```
/-- The masked weight value at lane `(e,j)`, block `(h,w,c)`. -/
```
```lean
noncomputable def mwVal (s0 : BlockState)
    (Weight : RegionName) (BIN OF IGD OGD WOFS WIFS WHS WWS : Nat)
    (h w c : Nat) (e : Fin BIN) (j : Fin OF) : ℝ :=
  if wMask s0 IGD OGD OF c e j then
    s0.readMem Weight (wAddr s0 WOFS WIFS WHS WWS OF OGD h w c e j)
  else 0
```
</details>

<details><summary><code>inMask</code></summary>

```
/-- The kernel's input mask Prop at lane `(i,e)`, block `(h,w,c)`. -/
```
```lean
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
```
</details>

<details><summary><code>inAddr</code></summary>

```
/-- The kernel's input load address at lane `(i,e)`, block `(h,w,c)`:
`inputBase i + IIFS·(c+e) + (IHS·inHeightOff).toNat + (IWS·inWidthOff).toNat`. -/
```
```lean
def inAddr (s0 : BlockState)
    (IBS IIFS IHS IWS OH OW BHW IGD SH SW PH PW : Nat) (h w c : Nat) (i : Fin BHW) (e : Fin BIN) : Nat :=
  inputBase s0 IBS IIFS OH OW BHW IGD i + IIFS * (c + e.val)
    + ((IHS : Int) * inHeightOff s0 OH OW BHW SH PH h i).toNat
    + ((IWS : Int) * inWidthOff s0 OW BHW SW PW w i).toNat
```
</details>

<details><summary><code>wMask</code></summary>

```
/-- The kernel's weight mask Prop at lane `(e,j)`, block `(h,w,c)`. -/
```
```lean
def wMask (s0 : BlockState) (IGD OGD OF c : Nat) (e : Fin BIN) (j : Fin OF) : Prop :=
  c + e.val < IGD ∧ featIdx s0 OF j < OGD

instance (s0 : BlockState) (IGD OGD OF c : Nat) (e : Fin BIN) (j : Fin OF) :
    Decidable (wMask s0 IGD OGD OF c e j) := by unfold wMask; infer_instance
```
</details>

<details><summary><code>wAddr</code></summary>

```
/-- The kernel's weight load address at lane `(e,j)`, block `(h,w,c)`:
`weightBase j + WIFS·(c+e) + WHS·h + WWS·w`. -/
```
```lean
def wAddr (s0 : BlockState)
    (WOFS WIFS WHS WWS OF OGD : Nat) (h w c : Nat) (e : Fin BIN) (j : Fin OF) : Nat :=
  weightBase s0 WOFS OF OGD j + WIFS * (c + e.val) + WHS * h + WWS * w
```
</details>

<details><summary><code>inHeightOff</code></summary>

```
/-- The kernel's `input_height_offset` (an `Int`): `h - padding_height + stride_height · heightIdx i`. -/
```
```lean
def inHeightOff (s0 : BlockState) (OH OW BHW SH PH : Nat) (h : Nat) (i : Fin BHW) : Int :=
  (h : Int) - (PH : Int) + (SH : Int) * (heightIdx s0 OH OW BHW i : Int)
```
</details>

<details><summary><code>inWidthOff</code></summary>

```
/-- The kernel's `input_width_offset` (an `Int`): `w - padding_width + stride_width · widthIdx i`. -/
```
```lean
def inWidthOff (s0 : BlockState) (OW BHW SW PW : Nat) (w : Nat) (i : Fin BHW) : Int :=
  (w : Int) - (PW : Int) + (SW : Int) * (widthIdx s0 OW BHW i : Int)
```
</details>

<details><summary><code>inputBase</code></summary>

```
/-- Per-lane input row base (the `Input +=` prefix), lane `i`:
`input_batch_stride · batchIdx i + input_in_feat_stride · pid2 · in_group_dim`. -/
```
```lean
def inputBase (s0 : BlockState) (IBS IIFS OH OW BHW IGD : Nat) (i : Fin BHW) : Nat :=
  IBS * batchIdx s0 OH OW BHW i + IIFS * s0.pids 2 * IGD
```
</details>

<details><summary><code>weightBase</code></summary>

```
/-- Per-lane weight col base (the `Weight +=` prefix), lane `j`:
`weight_out_feat_stride · featIdx j + weight_out_feat_stride · pid2 · out_group_dim`. -/
```
```lean
def weightBase (s0 : BlockState) (WOFS OF OGD : Nat) (j : Fin OF) : Nat :=
  WOFS * featIdx s0 OF j + WOFS * s0.pids 2 * OGD
```
</details>

## Public theorem: `triton_conv2d_fwd_io_correctness`

<details><summary>docstring</summary>

```
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
```
</details>

**Statement:**
```lean
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
          (out_feat_dim / groups) SH SW PH PW KH KW BHW BIN OF numCBlocks xs ys l
```

**Assumptions / layout contracts:**
- `hfp16 : R.round .fp16 = id`
- `hBIN : 0 < BIN`
- `hIGD : in_feat_dim / groups = BIN * numCBlocks`

**Closed-form spec defs (transitive):** `pOutAddr`, `triton_conv2d_fwd_IO`, `convStreamSum`, `pBatch`, `pFeat`, `pHeight`, `pWidth`, `conv2d_forward_surface`, `pInAddr`, `convStepH`, `convStepW`, `convStepCB`, `pWAddr`, `pInMask`, `pWMask`, `pActive`, `convInLane`, `convWLane`, `pBh`, `pBhw`, `pInputBase`, `pInH`, `pInW`, `pWeightBase`

<details><summary><code>pOutAddr</code></summary>

```
/-- pid-form terminal `Output` store address at lane `(i, j)`. -/
```
```lean
private def pOutAddr (pid₀ pid₁ pid₂ BHW OF OH OW OBS OOFS OHS OWS OGD i j : Nat) : Nat :=
  OBS * pBatch pid₀ OH OW BHW i + OOFS * (pid₂ * OGD + pFeat pid₁ OF j)
    + OHS * pHeight pid₀ OH OW BHW i + OWS * pWidth pid₀ OW BHW i
```
</details>

<details><summary><code>triton_conv2d_fwd_IO</code></summary>

```
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
```
```lean
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
```
</details>

<details><summary><code>convStreamSum</code></summary>

```
/-- **The streamed conv2d spec**: the ideal ℝ value of output lane `l = (i, n)`
as a single `T`-step fold over the flattened nest,

`∑_{t < KH·KW·NC} ∑_{e < BLOCK_IN} maskedInput(t)[i, e] · maskedWeight(t)[e, n]`,

where the two masked factors are the streamed tiles zeroed exactly where the
kernel's own per-lane masks are false (`pInMask` = image/channel boundaries,
i.e. zero padding; `pWMask` = in-feature / out-feature group boundaries). This
is the `⊨[R]` face of `convSpec`: the same im2col convolution reference,
re-expressed on the streaming skin's per-step tiles instead of on `s0`-indexed
`readMem`s. -/
```
```lean
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
```
</details>

<details><summary><code>pBatch</code></summary>

```
/-- pid-form `batch_offset`. -/
```
```lean
private def pBatch (pid₀ OH OW BHW i : Nat) : Nat := pBh pid₀ OW BHW i / OH
```
</details>

<details><summary><code>pFeat</code></summary>

```
/-- pid-form `output_feat_offset` lane `j`: `pid₁ · OF + j`. -/
```
```lean
private def pFeat (pid₁ OF j : Nat) : Nat := pid₁ * OF + j
```
</details>

<details><summary><code>pHeight</code></summary>

```
/-- pid-form `output_height_offset`. -/
```
```lean
private def pHeight (pid₀ OH OW BHW i : Nat) : Nat := pBh pid₀ OW BHW i % OH
```
</details>

<details><summary><code>pWidth</code></summary>

```
/-- pid-form `output_width_offset`. -/
```
```lean
private def pWidth (pid₀ OW BHW i : Nat) : Nat := pBhw pid₀ BHW i % OW
```
</details>

<details><summary><code>conv2d_forward_surface</code></summary>

```
/-- Faithful transcription of `triton_conv2d_fwd.py`'s `conv2d_forward_kernel`. -/
```
```lean
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
```
</details>

<details><summary><code>pInAddr</code></summary>

```
/-- pid-form `curr_input_pointer` cell at lane `(i, e)`, nest iteration
`(h, w, c)`. **Honest sentinel address**: `pInH` may be negative under padding,
so `((IHS : Int) * pInH …).toNat` is garbage on the masked-off lanes — exactly
as the kernel's own `tl.load` pointer arithmetic is. Harmless: every skin
obligation about `read1` is guarded by `mask1 →`. -/
```
```lean
private def pInAddr (pid₀ pid₂ IBS IIFS IHS IWS OH OW BHW IGD SH SW PH PW h w c i e : Nat) : Nat :=
  pInputBase pid₀ pid₂ IBS IIFS OH OW BHW IGD i + IIFS * (c + e)
    + ((IHS : Int) * pInH pid₀ OH OW BHW SH PH h i).toNat
    + ((IWS : Int) * pInW pid₀ OW BHW SW PW w i).toNat
```
</details>

<details><summary><code>convStepH</code></summary>

```
/-- Outer kernel-height index of flat step `t` (`= t / (KW · NC)`). -/
```
```lean
private def convStepH (KW NC t : Nat) : Nat := t / NC / KW
```
</details>

<details><summary><code>convStepW</code></summary>

```
/-- Middle kernel-width index of flat step `t`. -/
```
```lean
private def convStepW (KW NC t : Nat) : Nat := t / NC % KW
```
</details>

<details><summary><code>convStepCB</code></summary>

```
/-- Inner in-feature *block* index of flat step `t` (the `c`-loop counter is
`convStepCB … t · BLOCK_IN`). -/
```
```lean
private def convStepCB (NC t : Nat) : Nat := t % NC
```
</details>

<details><summary><code>pWAddr</code></summary>

```
/-- pid-form `curr_weight_pointer` cell at lane `(e, j)`. -/
```
```lean
private def pWAddr (pid₁ pid₂ WOFS WIFS WHS WWS OF OGD h w c e j : Nat) : Nat :=
  pWeightBase pid₁ pid₂ WOFS OF OGD j + WIFS * (c + e) + WHS * h + WWS * w
```
</details>

<details><summary><code>pInMask</code></summary>

```
/-- pid-form `input_mask` at lane `(i, e)`, nest iteration `(h, w, c)`. -/
```
```lean
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
```
</details>

<details><summary><code>pWMask</code></summary>

```
/-- pid-form `weight_mask` at lane `(e, j)`. -/
```
```lean
private def pWMask (pid₁ IGD OGD OF c e j : Nat) : Prop :=
  c + e < IGD ∧ pFeat pid₁ OF j < OGD

instance (pid₁ IGD OGD OF c e j : Nat) : Decidable (pWMask pid₁ IGD OGD OF c e j) := by
  unfold pWMask; infer_instance
```
</details>

<details><summary><code>pActive</code></summary>

```
/-- pid-form `output_mask` at lane `(i, j)`. -/
```
```lean
private def pActive (pid₀ pid₁ BHW OF batch_dim OH OW OGD i j : Nat) : Prop :=
  ((pBatch pid₀ OH OW BHW i < batch_dim ∧ pFeat pid₁ OF j < OGD) ∧
    pHeight pid₀ OH OW BHW i < OH) ∧ pWidth pid₀ OW BHW i < OW

instance (pid₀ pid₁ BHW OF batch_dim OH OW OGD i j : Nat) :
    Decidable (pActive pid₀ pid₁ BHW OF batch_dim OH OW OGD i j) := by
  unfold pActive; infer_instance
```
</details>

<details><summary><code>convInLane</code></summary>

```
/-- The `Input`-stream lane feeding output lane `l` at inner key `e`: the BHW
row of `l` (row-major over the `[BHW, OF]` output tile) paired with `e` over
the `[BHW, BLOCK_IN]` per-step input tile. -/
```
```lean
def convInLane (BHW BIN OF : Nat) (l : Fin (BHW * OF)) (e : Fin BIN) : Fin (BHW * BIN) :=
  Lane2D.encode ((Lane2D.decode l).1, e, PUnit.unit)
```
</details>

<details><summary><code>convWLane</code></summary>

```
/-- The `Weight`-stream lane feeding output lane `l` at inner key `e`: `e`
paired with the out-feature column of `l` over the `[BLOCK_IN, OF]` per-step
weight tile. -/
```
```lean
def convWLane (BHW BIN OF : Nat) (l : Fin (BHW * OF)) (e : Fin BIN) : Fin (BIN * OF) :=
  Lane2D.encode (e, (Lane2D.decode l).2.1, PUnit.unit)
```
</details>

<details><summary><code>pBh</code></summary>

```
/-- pid-form `batch_height_offset`. -/
```
```lean
private def pBh (pid₀ OW BHW i : Nat) : Nat := pBhw pid₀ BHW i / OW
```
</details>

<details><summary><code>pBhw</code></summary>

```
/-- pid-form `batch_height_width_offset` lane `i`: `pid₀ · BHW + i`. -/
```
```lean
private def pBhw (pid₀ BHW i : Nat) : Nat := pid₀ * BHW + i
```
</details>

<details><summary><code>pInputBase</code></summary>

```
/-- pid-form per-row `Input +=` base. -/
```
```lean
private def pInputBase (pid₀ pid₂ IBS IIFS OH OW BHW IGD i : Nat) : Nat :=
  IBS * pBatch pid₀ OH OW BHW i + IIFS * pid₂ * IGD
```
</details>

<details><summary><code>pInH</code></summary>

```
/-- pid-form `input_height_offset` (an `Int`; negative under padding). -/
```
```lean
private def pInH (pid₀ OH OW BHW SH PH h i : Nat) : Int :=
  (h : Int) - (PH : Int) + (SH : Int) * (pHeight pid₀ OH OW BHW i : Int)
```
</details>

<details><summary><code>pInW</code></summary>

```
/-- pid-form `input_width_offset` (an `Int`; negative under padding). -/
```
```lean
private def pInW (pid₀ OW BHW SW PW w i : Nat) : Int :=
  (w : Int) - (PW : Int) + (SW : Int) * (pWidth pid₀ OW BHW i : Int)
```
</details>

<details><summary><code>pWeightBase</code></summary>

```
/-- pid-form per-col `Weight +=` base. -/
```
```lean
private def pWeightBase (pid₁ pid₂ WOFS OF OGD j : Nat) : Nat :=
  WOFS * pFeat pid₁ OF j + WOFS * pid₂ * OGD
```
</details>

## Also present (pinned special-case summaries)
- `conv2d_closed_form_correct`
