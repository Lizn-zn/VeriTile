import VeriTile.Triton.Float.EvalOpR

/-! Changing a load's boundary padding is justified by an explicit oracle
condition. These lemmas do not add a condition to the original source; IO
contracts that choose zero unspecified values must supply it themselves. -/
namespace VeriTile.Triton

theorem evalOp_load_padding_of_undef_zero (dtype : TileDType)
    (mem : MemAccess dtype shape) (mask : MaskOpt dtype shape)
    (padding : PaddingOption) (s : BlockState) (hu : s.undef = fun _ _ => 0) :
    evalOp (.load dtype mem mask padding) s = evalOp (.load dtype mem mask .zero) s := by
  cases mem <;> cases mask <;>
    simp only [evalOp, BlockState.paddingValue_of_undef_zero _ _ _ _ _ hu]

theorem evalOpR_load_padding_of_undef_zero (R : RoundingModel) (dtype : TileDType)
    (mem : MemAccess dtype shape) (mask : MaskOpt dtype shape)
    (padding : PaddingOption) (s : BlockState) (hu : s.undef = fun _ _ => 0) :
    evalOpR R (.load dtype mem mask padding) s = evalOpR R (.load dtype mem mask .zero) s := by
  cases mem <;> cases mask <;>
    simp only [evalOpR, BlockState.paddingValue_of_undef_zero _ _ _ _ _ hu]

end VeriTile.Triton
