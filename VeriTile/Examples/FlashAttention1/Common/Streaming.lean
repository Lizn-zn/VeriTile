/-
VeriTile.Examples.FlashAttention1.Common.Streaming

Compatibility re-export for the promoted streaming accumulator module.
-/

import VeriTile.Triton.Semantics.StreamingAccumulator

namespace VeriTile.Examples

namespace FA1Math

export VeriTile.Triton.StreamingAccumulator
  ( scaledScore softmaxRow blockIndex mPartial alphaPartial
    alphaPartial_toWithBot lPartial oPartial lFree oFree lFree_succ
    oFree_succ mPartial_succ_ne_bot mPartial_succ_of_lt
    lPartial_succ_of_lt oPartial_succ_of_lt lPartial_eq_mShifted
    oPartial_eq_mShifted blockIndexEquiv blockIndex_blockIndexEquiv
    lFree_eq_flat oFree_eq_flat qkT_data_eq scaled_data_eq
    softmaxRow_scaled_data_eq qkT_data_eq' scaled_data_eq'
    softmaxRow_scaled_data_eq' block_qkT_data_eq block_scaled_data_eq
    block_mBlock_data_eq block_p_data_eq block_p_toWithBot
    block_p_rowSum_eq block_pv_data_eq block_mNew_tile_eq
    option_max_eq_withbot_max sup'_proof_irrel block_lNew_tile_eq
    block_oAcc_tile_eq oFree_div_lFree_eq_attentionReal
    streaming_eq_attentionReal lFree_final_pos lPartial_final_ne_zero )

end FA1Math

end VeriTile.Examples
