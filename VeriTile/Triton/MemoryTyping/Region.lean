/-
VeriTile.Triton.MemoryTyping.Region

Basic region-dtype contracts for Triton memory typing.
-/

import VeriTile.Triton.Core

namespace VeriTile.Triton

/-- User-supplied dtype assignment for named memory regions. -/
abbrev RegionTyping := RegionName → TileDType

end VeriTile.Triton
