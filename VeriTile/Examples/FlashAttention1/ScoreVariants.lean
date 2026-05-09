/-
VeriTile.Examples.FlashAttention1.ScoreVariants

Score-level FA-1 realism references for issue #40: ALiBi, sliding-window
masks, and softcap. These are mathematical spec surfaces; the online-softmax
kernel recurrence can target them by changing only the score expression fed
to max/exp.
-/

import VeriTile.Examples.FlashAttention1.ScoreVariants.Forward
import VeriTile.Examples.FlashAttention1.ScoreVariants.Backward

namespace VeriTile.Examples

open VeriTile.Triton

namespace FA1Score


end FA1Score

end VeriTile.Examples
