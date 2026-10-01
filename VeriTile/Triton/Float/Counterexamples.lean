/- Kernel-checked witnesses for the concrete software value profile.
These refute unconditional exact rules, not two-gates acceptance. -/
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Triton.FP

set_option maxRecDepth 4096 in
theorem bf16_add_assoc_witness :
    let a := Value.ofBits .bf16 0x4380 -- 256
    let b := Value.ofBits .bf16 0x3f80 -- 1
    let z := Value.ofBits .bf16 0xc380 -- -256
    (add Config.ieeeValue (add Config.ieeeValue a b) z).toRat? = some 0 ∧
    (add Config.ieeeValue a (add Config.ieeeValue b z)).toRat? = some 1 := by decide +kernel

set_option maxRecDepth 4096 in
theorem fp32_add_assoc_witness :
    let a := Value.ofBits .fp32 0x4b800000 -- 2^24
    let b := Value.ofBits .fp32 0x3f800000 -- 1
    let z := Value.ofBits .fp32 0xcb800000 -- -2^24
    (add Config.ieeeValue (add Config.ieeeValue a b) z).toRat? = some 0 ∧
    (add Config.ieeeValue a (add Config.ieeeValue b z)).toRat? = some 1 := by decide +kernel

set_option maxRecDepth 4096 in
theorem bf16_distrib_witness :
    let a := Value.ofBits .bf16 0x4040 -- 3
    let b := Value.ofBits .bf16 0x4380 -- 256
    let z := Value.ofBits .bf16 0x3f80 -- 1
    (mul Config.ieeeValue a (add Config.ieeeValue b z)).toRat? = some 768 ∧
    (add Config.ieeeValue (mul Config.ieeeValue a b) (mul Config.ieeeValue a z)).toRat? = some 772 := by decide +kernel

set_option maxRecDepth 4096 in
theorem fp32_distrib_witness :
    let a := Value.ofBits .fp32 0x40400000 -- 3
    let b := Value.ofBits .fp32 0x4b800000 -- 2^24
    let z := Value.ofBits .fp32 0x3f800000 -- 1
    (mul Config.ieeeValue a (add Config.ieeeValue b z)).toRat? = some 50331648 ∧
    (add Config.ieeeValue (mul Config.ieeeValue a b) (mul Config.ieeeValue a z)).toRat? = some 50331652 := by decide +kernel

set_option maxRecDepth 4096 in
theorem bf16_fma_witness :
    let a := Value.ofBits .bf16 0x3f81 -- 129/128
    let b := Value.ofBits .bf16 0x3f7e -- 127/128
    let z := Value.ofBits .bf16 0xbf80 -- -1
    (add Config.ieeeValue (mul Config.ieeeValue a b) z).toRat? = some 0 ∧
    (fma Config.ieeeValue a b z).toRat? = some (mkRat (-1) 16384) := by decide +kernel

set_option maxRecDepth 4096 in
theorem fp32_fma_witness :
    let a := Value.ofBits .fp32 0x3f800400 -- 8193/8192
    let b := Value.ofBits .fp32 0x3f7ff800 -- 8191/8192
    let z := Value.ofBits .fp32 0xbf800000 -- -1
    (add Config.ieeeValue (mul Config.ieeeValue a b) z).toRat? = some 0 ∧
    (fma Config.ieeeValue a b z).toRat? = some (mkRat (-1) 67108864) := by decide +kernel

end VeriTile.Triton.FP
