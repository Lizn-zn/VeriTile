import bench.tritonbench_g.fast_rope_embedding.Proofs.Erasure
import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Reference
open VeriTile.Triton
noncomputable section

def first (row QR H head lane : Nat) := row*QR+head*H+lane
def second (row QR H head lane : Nat) := first row QR H head lane+H/2
def coefficientAddress (row stride seq lane : Nat) := (row%seq)*stride+lane

def signedSine (backwards : Bool) (x : ℝ) := if backwards then -x else x

def firstValue (backwards : Bool) (a b c s : ℝ) := a*c-b*signedSine backwards s
def secondValue (backwards : Bool) (a b c s : ℝ) := b*c+a*signedSine backwards s

def start (group : Nat) := group*4
def stop (group heads : Nat) := min (start group+4) heads

def Written (row group QR H heads B o : Nat) : Prop :=
  ∃ k,start group≤k ∧ k<stop group heads ∧ ∃ j : Fin B,j.val<H/2 ∧
    (o=first row QR H k j.val ∨ o=second row QR H k j.val)

def coefficient (u : BlockState) (R : RegionName) (stride seq H B : Nat) (j : Fin B) : ℝ :=
  if j.val<H/2 then u.readMem R (coefficientAddress u.pid stride seq j.val) else 0

theorem halves_disjoint (row QR H k : Nat) (i j : Fin B) (hi : i.val<H/2) :
    first row QR H k i.val≠second row QR H k j.val := by
  simp only [first,second]
  omega

theorem earlier_head (row QR H a b : Nat) (i j : Fin B) (hab : a<b)
    (hi : i.val<H/2) : second row QR H a i.val<first row QR H b j.val := by
  have hm := Nat.mul_le_mul_right H (show a+1≤b by omega)
  simp only [Nat.add_mul,Nat.one_mul] at hm
  simp only [first,second]
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Reference
