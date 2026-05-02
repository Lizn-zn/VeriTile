/-
VeriTile.Triton.MemoryTyping.Region

Basic region-dtype contracts for Triton memory typing.
-/

import VeriTile.Triton.Core

namespace VeriTile.Triton

/-- User-supplied dtype assignment for named memory regions. -/
abbrev RegionTyping := RegionName → TileDType

namespace FloatDType

/-- The concrete dtype witnessed by a floating dtype proof. -/
def dtype : {dtype : TileDType} → FloatDType dtype → TileDType
  | _, .real => .real
  | _, .fp32 => .fp32
  | _, .fp16 => .fp16
  | _, .bf16 => .bf16

end FloatDType

end VeriTile.Triton
