/-
VeriTile.Math.GeluTaylor20PolyDef

Taylor-20 polynomial definition for the midrange approximate-GeLU
certificate (`GeluTaylor20Cert`). Kept axiom-free of erf/sigmoid so the
polynomial-bound proofs in `GeluTaylor20PolyLowerCases` /
`GeluTaylor20PolyUpperCases` only need real arithmetic.
-/

import Mathlib.Data.Real.Basic

namespace VeriTile.Math

/-- Degree-20 Taylor polynomial used by the midrange certificate, expanded at
`463/200 = 2.315`. -/
noncomputable def geluError_mid_taylor20_forCert (x : ℝ) : ℝ :=
  let t := x - 463 / 200
  58799578250044 / 168873512033139177
    + (490511376410848 / 765861081911736471) * t
    + (-(69595973957346 / 98732046708052927)) * t^2
    + (-(318812859576406 / 603001279934315985)) * t^3
    + (95323914281662 / 166041057487226167) * t^4
    + (83409703730901 / 541893669526039966) * t^5
    + (-(131670005603011 / 523709508683584689)) * t^6
    + (-(3688290726053 / 972548317670397000)) * t^7
    + (45774012660025 / 681762334304929013) * t^8
    + (-(7348721185503 / 706588070591977423)) * t^9
    + (-(5731745181647 / 501999458364499667)) * t^10
    + (12817416488 / 3546237897524503) * t^11
    + (1104703829666 / 959634314348063215) * t^12
    + (-(638985140807 / 953651943452319460)) * t^13
    + (-(20677130095 / 708673697371547448)) * t^14
    + (23268802103 / 302557557620128678) * t^15
    + (-(3607948043 / 331131972419363373)) * t^16
    + (-(2089268122 / 454157679889909063)) * t^17
    + (577867222 / 307044885316340023) * t^18
    + (-(15956207 / 139052182026179153)) * t^19
    + (-(30752279 / 251067520891536832)) * t^20

end VeriTile.Math
