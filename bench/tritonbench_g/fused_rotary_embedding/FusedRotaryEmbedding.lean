import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding

open VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- Surface transcription of the unconditional Q rotary part of
`fused_rotary_embedding.py`'s `decoding_fused_rotary_embedding_kernel`.

The full Python kernel also conditionally rotates K and fills K/V caches when
`cur_head_idx % KV_GROUP_NUM == 0`. That branch depends on context-length
metadata and cache block tables, so this surface covers the unconditional Q
updates that every program instance performs: both the first and second rotary
halves are written. -/
def decoding_fused_rotary_embedding_q_surface
    (Q Cos Sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      _HEAD_DIM HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(Q + off_q0)
  loaded_q1 = tl.load(Q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(Cos + off_cos_sin)
  loaded_sin = tl.load(Sin + off_cos_sin)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
  tl.store(Q + off_q0, out_q0)
  tl.store(Q + off_q1, out_q1)
}

/-- Proof-oriented Q first-half slice of `fused_rotary_embedding.py`'s
`decoding_fused_rotary_embedding_kernel`.

The full kernel updates Q, conditionally rotates K, and fills K/V caches. This
slice captures the unconditional Q first-half rotary writeback:
`q0 * cos - q1 * sin`. -/
def decoding_fused_rotary_embedding_q_first_half
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat) :
    ComputeKernel := triton {
  cur_head_idx = tl.program_id(0)
  cur_token_idx = tl.program_id(1)
  dim_range0 = tl.arange(0, $(HALF_DIM))
  dim_range1 = tl.arange(0, $(HALF_DIM)) + $(HALF_DIM)
  off_q = cur_token_idx * $(q_token_stride) + cur_head_idx * $(q_head_stride)
  off_q0 = off_q + dim_range0 * $(head_dim_stride)
  off_q1 = off_q + dim_range1 * $(head_dim_stride)
  loaded_q0 = tl.load(q + off_q0)
  loaded_q1 = tl.load(q + off_q1)
  off_cos_sin = cur_token_idx * $(cos_token_stride) + dim_range0 * $(cos_stride)
  loaded_cos = tl.load(cos + off_cos_sin)
  loaded_sin = tl.load(sin + off_cos_sin)
  out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
  tl.store(q + off_q0, out_q0)
}

def dimIndex (i : Fin HALF_DIM) : Nat :=
  i.val

def qBase (s : BlockState) (q_token_stride q_head_stride : Nat) : Nat :=
  s.pids 1 * q_token_stride + s.pids 0 * q_head_stride

def qFirstOffset
    (s : BlockState) (q_token_stride q_head_stride head_dim_stride : Nat)
    (i : Fin HALF_DIM) : Nat :=
  qBase s q_token_stride q_head_stride + dimIndex i * head_dim_stride

def qSecondOffset
    (s : BlockState)
    (q_token_stride q_head_stride head_dim_stride HALF_DIM : Nat)
    (i : Fin HALF_DIM) : Nat :=
  qBase s q_token_stride q_head_stride + (dimIndex i + HALF_DIM) * head_dim_stride

def cosOffset
    (s : BlockState) (cos_token_stride cos_stride : Nat) (i : Fin HALF_DIM) : Nat :=
  s.pids 1 * cos_token_stride + dimIndex i * cos_stride

def sinOffset
    (s : BlockState) (cos_token_stride cos_stride : Nat) (i : Fin HALF_DIM) : Nat :=
  s.pids 1 * cos_token_stride + dimIndex i * cos_stride

noncomputable def qFirstSpec
    (s : BlockState) (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (i : Fin HALF_DIM) : ℝ :=
  s.readMem q (qFirstOffset s q_token_stride q_head_stride head_dim_stride i) *
    s.readMem cos (cosOffset s cos_token_stride cos_stride i) -
  s.readMem q
      (qSecondOffset s q_token_stride q_head_stride head_dim_stride HALF_DIM i) *
    s.readMem sin (sinOffset s cos_token_stride cos_stride i)

/-- Algorithm-layer correctness for the Q first-half rotary writeback. -/
theorem decoding_fused_rotary_embedding_q_first_half_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s s' : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qFirstOffset s q_token_stride q_head_stride head_dim_stride i))
    (hExec : exec (decoding_fused_rotary_embedding_q_first_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM) s = some s') :
    ∀ i : Fin HALF_DIM,
      s'.readMem q (qFirstOffset s q_token_stride q_head_stride head_dim_stride i) =
        qFirstSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i := by
  intro i
  have hRawInj : Function.Injective
      (fun idx : TileIndex [HALF_DIM] =>
        s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
          idx.1.val * head_dim_stride) := by
    intro a b h
    have hab : a.1 = b.1 := by
      apply hOutInj
      simpa [qFirstOffset, qBase, dimIndex] using h
    cases a
    cases b
    simp only at hab
    cases hab
    rfl
  by_cases hHalf : 0 < HALF_DIM
  · simp [exec, decoding_fused_rotary_embedding_q_first_half, stepStmts, stepStmt,
          evalOp, Option.bind, Option.map, Tile.bop, Tile.ptrAdd,
          NumericDType.add, NumericDType.mul, NumericDType.sub, hHalf] at hExec
    rw [← hExec]
    simp only [qFirstOffset, qBase, dimIndex]
    have hScatter :=
      (BlockState.scatter_readback_prop_masked_nd
        (region := q)
        (shape := [HALF_DIM])
        (s := (s.setReg "cur_head_idx" TileDType.nat [] (Tile.scalar (s.pids 0))
          |>.setReg "cur_token_idx" TileDType.nat [] (Tile.scalar (s.pids 1))
          |>.setReg "dim_range0" TileDType.nat [HALF_DIM] (Tile.vec fun i => i.val)
          |>.setReg "dim_range1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => i.val + HALF_DIM)
          |>.setReg "off_q" TileDType.nat []
            (Tile.scalar (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride))
          |>.setReg "off_q0" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                i.val * head_dim_stride)
          |>.setReg "off_q1" TileDType.nat [HALF_DIM]
            (Tile.vec fun i =>
              s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                (i.val + HALF_DIM) * head_dim_stride)
          |>.setReg "loaded_q0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem q
                (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                  i.1.val * head_dim_stride)) }
          |>.setReg "loaded_q1" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem q
                (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                  (i.1.val + HALF_DIM) * head_dim_stride)) }
          |>.setReg "off_cos_sin" TileDType.nat [HALF_DIM]
            (Tile.vec fun i => s.pids 1 * cos_token_stride + i.val * cos_stride)
          |>.setReg "loaded_cos" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "loaded_sin" TileDType.real [HALF_DIM]
            { data := fun i =>
              some (s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }
          |>.setReg "out_q0" TileDType.real [HALF_DIM]
            { data := fun i =>
              some
                (s.readMem q
                    (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                      i.1.val * head_dim_stride) *
                  s.readMem cos (s.pids 1 * cos_token_stride + i.1.val * cos_stride) -
                s.readMem q
                    (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                      (i.1.val + HALF_DIM) * head_dim_stride) *
                  s.readMem sin (s.pids 1 * cos_token_stride + i.1.val * cos_stride)) }))
        (offsetFn := fun idx : TileIndex [HALF_DIM] =>
          s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
            idx.1.val * head_dim_stride)
        (valueFn := fun idx : TileIndex [HALF_DIM] =>
          s.readMem q
              (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                idx.1.val * head_dim_stride) *
            s.readMem cos (s.pids 1 * cos_token_stride + idx.1.val * cos_stride) -
          s.readMem q
              (s.pids 1 * q_token_stride + s.pids 0 * q_head_stride +
                (idx.1.val + HALF_DIM) * head_dim_stride) *
            s.readMem sin (s.pids 1 * cos_token_stride + idx.1.val * cos_stride))
        (P := fun _idx : TileIndex [HALF_DIM] => True)
        hRawInj (i, PUnit.unit))
    simpa [qFirstSpec, qFirstOffset, qSecondOffset, cosOffset, sinOffset, qBase,
      dimIndex, Tile.vec] using hScatter
  · exact False.elim (hHalf (Nat.lt_of_le_of_lt (Nat.zero_le _) i.isLt))

/-- Compute-facing correctness for the Q first-half rotary writeback. -/
theorem decoding_fused_rotary_embedding_q_first_half_compute_correct
    (q cos sin : RegionName)
    (q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
      HALF_DIM : Nat)
    (s : BlockState)
    (hOutInj : Function.Injective
      (fun i : Fin HALF_DIM =>
        qFirstOffset s q_token_stride q_head_stride head_dim_stride i)) :
    ComputeCorrect.Realizes
      (kernel := decoding_fused_rotary_embedding_q_first_half q cos sin
        q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
        HALF_DIM)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.writeIf
        (fun _i : Fin HALF_DIM => True)
        (fun i => (q, qFirstOffset s q_token_stride q_head_stride head_dim_stride i)))
      (expected := fun i =>
        qFirstSpec s q cos sin q_token_stride q_head_stride head_dim_stride
          cos_token_stride cos_stride HALF_DIM i) := by
  rw [ComputeCorrect.realizes_writeIf_iff]
  apply ComputeKernel.computeCorrect_of_toAlgKernel
  · simp [decoding_fused_rotary_embedding_q_first_half]
  intro s0 s' hExec hs0
  subst s0
  intro i _hActive
  exact decoding_fused_rotary_embedding_q_first_half_correct q cos sin
    q_token_stride q_head_stride head_dim_stride cos_token_stride cos_stride
    HALF_DIM s s' hOutInj hExec i

end VeriTile.Bench.TritonBenchG.FusedRotaryEmbedding
