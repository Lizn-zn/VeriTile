/-
VeriTile.Memory.Typing.Region

Basic region-dtype contracts for Triton memory typing.
-/

import VeriTile.Core

namespace VeriTile

/-- User-supplied dtype assignment for named memory regions. -/
abbrev RegionTyping := RegionName → TileDType

end VeriTile
