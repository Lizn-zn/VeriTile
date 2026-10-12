import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! A load followed by a cast must use the pointer's storage dtype at that
program point. These checks retain casts, masks and pointer rebinding. -/
namespace LoadCastStorageTests
open VeriTile.Triton

private def assignment (k : ComputeKernel) (name : RegName) :=
  (k.surfaceBody.drop (Source.prefixBefore k.surfaceBody name).length).head?

private def source (X Y : RegionName) (N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float16, Y = tl.bfloat16
  offsets = tl.arange(0, $(N))
  ptr = X + offsets
  first = tl.load(ptr).to(tl.float32)
  ptr = Y + offsets
  second = (tl.load(ptr, mask=offsets < $(N), other=0.0)).to(tl.float16)
  third = tl.load(ptr, offsets < $(N), other=0.0).to(tl.float32)
}

theorem first_load_reads_fp16 (X Y : RegionName) (N : Nat) :
    assignment (source X Y N) "first" = some
      (.assign .real [N] "first" (.compute (.alg .fp32
        (.castFloat .fp16 .real (.load .fp16 (.ptr (.ref .ptr [N] "ptr")) .none))))) := by
  rfl

theorem rebound_load_reads_bf16 (X Y : RegionName) (N : Nat) :
    ∃ access mask, assignment (source X Y N) "second" = some
      (.assign .fp16 [N] "second" (.alg (.castFloat .bf16 .fp16 (.load .bf16 access mask)))) := by
  exact ⟨_, _, rfl⟩

theorem positional_mask_retains_storage (X Y : RegionName) (N : Nat) :
    ∃ access mask other, assignment (source X Y N) "third" = some
      (.assign .real [N] "third" (.compute (.alg .fp32
        (.castFloat .bf16 .real (.load .bf16 access (.maskOther mask other)))))) := by
  exact ⟨_, _, _, rfl⟩

private def blockSource (X : RegionName) (N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float16
  ptr = tl.make_block_ptr(base=X, shape=($(N), 1), strides=(1, 1), offsets=(0, 0),
    block_shape=($(N), 1), order=(1, 0))
  value = tl.load(ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
}

theorem checked_block_load_retains_storage (X : RegionName) (N : Nat) :
    assignment (blockSource X N) "value" = some
      (.assign .real [N, 1] "value" (.compute (.alg .fp32
        (.castFloat .fp16 .real
          (.load .fp16 (.blockPtr (.ref .blockPtr [N, 1] "ptr") [0, 1]) .none))))) := by
  rfl

private def parameterSource (dtype : FloatDType) (X : RegionName) (N : Nat) : ComputeKernel := triton {
  tl.region X = DTYPE
  ptr = X + tl.arange(0, $(N))
  value = (tl.load(ptr)).to(tl.float32)
}

theorem parameter_storage_precedes_cast (dtype : FloatDType) (X : RegionName) (N : Nat) :
    assignment (parameterSource dtype X N) "value" = some
      (.assign .real [N] "value" (.compute (.alg .fp32
        (.castFloat dtype .real (.load dtype.toTileDType (.ptr (.ref .ptr [N] "ptr")) .none))))) := by
  rfl

/-- In-place arithmetic uses the operand's storage, including a pointer
rebound before the second pair of updates. The fp32 accumulator is separate. -/
private def updates (dtype : FloatDType) (X Y : RegionName) (N : Nat) : ComputeKernel := triton {
  tl.region X = DTYPE, Y = tl.bfloat16
  ptr = X + tl.arange(0, $(N))
  a = tl.zeros([$(N)], dtype=tl.float32)
  a += tl.load(ptr).to(tl.float32)
  b = tl.zeros([$(N)], dtype=tl.float32)
  b -= (tl.load(ptr)).to(tl.float32)
  ptr = Y + tl.arange(0, $(N))
  c = tl.zeros([$(N)], dtype=tl.float32)
  c *= tl.load(ptr).to(tl.float32)
  d = tl.zeros([$(N)], dtype=tl.float32)
  d /= (tl.load(ptr)).to(tl.float32)
}

private def updateAt (k : ComputeKernel) (name : RegName) :=
  (k.surfaceBody.drop (Source.prefixBefore k.surfaceBody name 1).length).head?

theorem update_add_retains_storage (dtype : FloatDType) (X Y : RegionName) (N : Nat) :
    updateAt (updates dtype X Y N) "a" = some
      (.assign .real [N] "a" (.compute (.alg .fp32
        (.add .real (.consSame .nil) (.ref .real [N] "a")
          (.castFloat dtype .real (.load dtype.toTileDType (.ptr (.ref .ptr [N] "ptr")) .none)))))) := rfl

theorem update_sub_retains_storage (dtype : FloatDType) (X Y : RegionName) (N : Nat) :
    updateAt (updates dtype X Y N) "b" = some
      (.assign .real [N] "b" (.compute (.alg .fp32
        (.sub .real (.consSame .nil) (.ref .real [N] "b")
          (.castFloat dtype .real (.load dtype.toTileDType (.ptr (.ref .ptr [N] "ptr")) .none)))))) := rfl

theorem update_mul_retains_storage (dtype : FloatDType) (X Y : RegionName) (N : Nat) :
    updateAt (updates dtype X Y N) "c" = some
      (.assign .real [N] "c" (.compute (.alg .fp32
        (.mul .real (.consSame .nil) (.ref .real [N] "c")
          (.castFloat .bf16 .real (.load .bf16 (.ptr (.ref .ptr [N] "ptr")) .none)))))) := rfl

theorem update_div_retains_storage (dtype : FloatDType) (X Y : RegionName) (N : Nat) :
    updateAt (updates dtype X Y N) "d" = some
      (.assign .real [N] "d" (.compute (.alg .fp32
        (.div .real (.consSame .nil) (.ref .real [N] "d")
          (.castFloat .bf16 .real (.load .bf16 (.ptr (.ref .ptr [N] "ptr")) .none)))))) := rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end LoadCastStorageTests
