import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Generic tile, mask, dynamic-pointer and store lemmas used by the current
complete GEMV proof. Base-pointer production is proved separately from the
actual source in RealSetup. The obsolete static-address model is retained
only in LegacyClippedOffset and is not imported here. -/
namespace VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealFoundation
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 8000
set_option linter.unusedSimpArgs false

-- matmul-style expandDim helpers
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

-- current_n eval
theorem currentn_eval (BLOCK_N c : Nat) (s : BlockState)
    (hon : s.regs .nat [BLOCK_N] "offset_n" = some (Tile.vec (fun i : Fin BLOCK_N => i.val))) :
    evalOp (Op.add NumericDType.nat Broadcast.scalarL (Op.ref TileDType.nat [] "n")
        (Op.ref TileDType.nat [BLOCK_N] "offset_n"))
      (s.setReg "n" .nat [] (Tile.scalar (c*BLOCK_N)))
      = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)) := by
  rw [evalOp_add, evalOp_ref_setReg_same,
    evalOp_ref_setReg_ne_name _ _ _ _ _ _ _ _ (show ("offset_n":RegName) ≠ "n" by decide),
    evalOp_ref, hon]
  apply congrArg some
  ext i
  simp only [Tile.bop_data, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
    Tile.scalar, Tile.vec, NumericDType.add]

-- c_mask eval (1D)
theorem cmask_eval (split_n_length BLOCK_N c : Nat) (s : BlockState)
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val))) :
    evalOp (Op.lt ComparableDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat split_n_length)) s
      = some (Tile.vec (fun i : Fin BLOCK_N => decide (c*BLOCK_N + i.val < split_n_length))) := by
  rw [evalOp_lt, evalOp_ref, hcn, evalOp_constNat]
  apply congrArg some
  ext i
  simp only [Tile.cop_data, Tile.vec_data, Tile.scalar_data, Broadcast.leftIndex_scalarR,
    Broadcast.rightIndex_scalarR, ComparableDType.lt, decide_eq_decide]


-- b_ptr_mask eval (2D)
theorem bmask_eval (K split_n_length BLOCK_N BLOCK_K c : Nat) (s : BlockState)
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val))) :
    evalOp (Op.boolAnd Broadcast.nil.consL.consR
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n"))
          (Op.constNat split_n_length))
        (Op.lt ComparableDType.nat Broadcast.scalarR (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k"))
          (Op.constNat K))) s
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          (decide (c*BLOCK_N + idx.1.val < split_n_length) && decide (idx.2.1.val < K))⟩ := by
  unfold evalOp
  simp only [evalOp_lt, evalOp_constNat, evalOp_expandDim_zero_nat, evalOp_expandDim_one_nat,
    hcn, hok, Option.bind_eq_bind, Option.bind_some, Option.bind]
  apply congrArg some
  ext idx
  simp only [Tile.bop_data, Tile.cop_data, Tile.scalar_data, Tile.vec,
    Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR,
    Broadcast.leftIndex_consL, Broadcast.leftIndex_consR,
    Broadcast.rightIndex_consL, Broadcast.rightIndex_consR,
    Broadcast.leftIndex_nil, Broadcast.rightIndex_nil, ComparableDType.lt]
  rfl


-- tiled_b address tile eval (2D ptr)
theorem bptrs_eval (lora_ptr : RegionName) (lbase lora_k_stride lora_n_stride BLOCK_N BLOCK_K c : Nat) (s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr" = some (Tile.scalar (lora_ptr, lbase)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val))) :
    evalOp (Op.ptrAdd Broadcast.nil.consL.consR
        (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "b_ptr")
          (Op.mul NumericDType.nat Broadcast.scalarR
            (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n")) (Op.constNat lora_k_stride)))
        (Op.mul NumericDType.nat Broadcast.scalarR
          (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k")) (Op.constNat lora_n_stride))) s
      = some (⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          (lora_ptr, lbase + (c*BLOCK_N + idx.1.val)*lora_k_stride + idx.2.1.val*lora_n_stride)⟩ : Tile .ptr [BLOCK_N, BLOCK_K]) := by
  rw [evalOp_ptrAdd, evalOp_ptrAdd, evalOp_ref, hbp]
  simp only [evalOp_mul, evalOp_constNat, evalOp_expandDim_one_nat, evalOp_expandDim_zero_nat,
    hcn, hok, Option.bind_eq_bind, Option.bind_some, Option.bind]
  apply congrArg some
  ext idx
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar_data,
      Broadcast.leftIndex_consL, Broadcast.leftIndex_consR, Broadcast.rightIndex_consL,
      Broadcast.rightIndex_consR, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar_data, NumericDType.add, NumericDType.mul,
      Broadcast.leftIndex_consL, Broadcast.leftIndex_consR, Broadcast.rightIndex_consL,
      Broadcast.rightIndex_consR, Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]


-- tiled_b load eval (2D masked ptr)
theorem tiledb_eval (lora_ptr : RegionName) (lbase K split_n_length lora_k_stride lora_n_stride BLOCK_N BLOCK_K c : Nat) (s : BlockState)
    (hbp : s.regs .ptr [] "b_ptr" = some (Tile.scalar (lora_ptr, lbase)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)))
    (hok : s.regs .nat [BLOCK_K] "offset_k" = some (Tile.vec (fun j : Fin BLOCK_K => j.val)))
    (hbm : s.regs .bool [BLOCK_N, BLOCK_K] "b_ptr_mask"
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] => (decide (c*BLOCK_N + idx.1.val < split_n_length) && decide (idx.2.1.val < K))⟩) :
    evalOp (Op.load TileDType.real
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.nil.consL.consR
            (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "b_ptr")
              (Op.mul NumericDType.nat Broadcast.scalarR
                (Op.expandDim ⟨1, by simp⟩ (Op.ref TileDType.nat [BLOCK_N] "current_n")) (Op.constNat lora_k_stride)))
            (Op.mul NumericDType.nat Broadcast.scalarR
              (Op.expandDim ⟨0, by simp⟩ (Op.ref TileDType.nat [BLOCK_K] "offset_k")) (Op.constNat lora_n_stride))))
        (MaskOpt.maskOther (Op.ref TileDType.bool [BLOCK_N, BLOCK_K] "b_ptr_mask")
          ((Op.const (0.0:ℝ)).broadcast [BLOCK_N, BLOCK_K]))) s
      = some ⟨fun idx : TileIndex [BLOCK_N, BLOCK_K] =>
          some (if c*BLOCK_N + idx.1.val < split_n_length ∧ idx.2.1.val < K then
            s.readMem lora_ptr (lbase + (c*BLOCK_N + idx.1.val)*lora_k_stride + idx.2.1.val*lora_n_stride) else 0.0)⟩ := by
  simp only [evalOp, bptrs_eval lora_ptr lbase lora_k_stride lora_n_stride BLOCK_N BLOCK_K c s hbp hcn hok, hbm]
  apply congrArg some
  ext idx
  simp only [Tile.scalar_data, BlockState.readMemValue_real]
  by_cases h1 : c*BLOCK_N + idx.1.val < split_n_length <;> by_cases h2 : idx.2.1.val < K <;>
    simp [h1, h2]

-- c_ptrs address tile eval (1D ptr)
theorem cptrs_eval (out_ptr : RegionName) (cbase cn_stride BLOCK_N c : Nat) (s : BlockState)
    (hcp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out_ptr, cbase)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val))) :
    evalOp (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "c_ptr")
        (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat cn_stride))) s
      = some (Tile.vec (fun i : Fin BLOCK_N => (out_ptr, cbase + (c*BLOCK_N + i.val)*cn_stride))) := by
  rw [evalOp_ptrAdd, evalOp_ref, hcp, evalOp_mul, evalOp_ref, hcn, evalOp_constNat]
  apply congrArg some
  ext i
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar_data,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]
  · simp [Tile.ptrAdd, Tile.bop, Tile.vec, Tile.scalar_data, NumericDType.add, NumericDType.mul,
      Broadcast.leftIndex_scalarL, Broadcast.rightIndex_scalarL,
      Broadcast.leftIndex_scalarR, Broadcast.rightIndex_scalarR]


-- store step: the masked ptr store folds to writeMem over lanes
theorem store_step (out_ptr : RegionName) (cbase cn_stride BLOCK_N c : Nat) (s : BlockState)
    (vals : Fin BLOCK_N → ℝ) (mask : Fin BLOCK_N → Bool)
    (hcp : s.regs .ptr [] "c_ptr" = some (Tile.scalar (out_ptr, cbase)))
    (hcn : s.regs .nat [BLOCK_N] "current_n" = some (Tile.vec (fun i : Fin BLOCK_N => c*BLOCK_N + i.val)))
    (hacc : s.regs .real [BLOCK_N] "accumulator" = some ⟨fun i : TileIndex [BLOCK_N] => some (vals i.1)⟩)
    (hcm : s.regs .bool [BLOCK_N] "c_mask" = some (Tile.vec mask)) :
    stepStmt (Stmt.store TileDType.real [BLOCK_N]
        (MemAccess.ptr
          (Op.ptrAdd Broadcast.scalarL (Op.ref TileDType.ptr [] "c_ptr")
            (Op.mul NumericDType.nat Broadcast.scalarR (Op.ref TileDType.nat [BLOCK_N] "current_n") (Op.constNat cn_stride))))
        (Op.ref TileDType.real [BLOCK_N] "accumulator") (MaskOpt.mask (Op.ref TileDType.bool [BLOCK_N] "c_mask"))) s
      = some ((TileShape.allIndices [BLOCK_N]).foldl
          (fun acc (i : TileIndex [BLOCK_N]) =>
            if mask i.1 then acc.writeMem out_ptr (cbase + (c*BLOCK_N + i.1.val)*cn_stride) (vals i.1) else acc) s) := by
  simp only [stepStmt, evalOp_ref, hacc, hcm,
    cptrs_eval out_ptr cbase cn_stride BLOCK_N c s hcp hcn,
    Option.bind_eq_bind, Option.bind_some, Option.map,
    Tile.vec, BlockState.writeMemTyped_real, FloatDType.real_storeValue, WithBot.unbotD_some]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvExpandSlice.RealFoundation
