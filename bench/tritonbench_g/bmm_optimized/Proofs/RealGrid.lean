import bench.tritonbench_g.bmm_optimized.Proofs.RealMath

/-! CTA coordinates depend on both grid dimensions. These closed forms match
the source's identity/grouped branches, including its final shortened group.
They remain separate from array values and numerical assumptions. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid

def groupSize (gridX groupId groupM : Nat) : Nat :=
  if groupId * groupM + groupM > gridX then gridX % groupM else groupM

def rowTile (px py gridX gridY groupM : Nat) : Nat :=
  if groupM = 1 then px else
    let pid := px + py * gridX
    let count := gridY * groupM
    let group := pid / count
    group * groupM + pid % count % groupSize gridX group groupM

def columnTile (px py gridX gridY groupM : Nat) : Nat :=
  if groupM = 1 then py else
    let pid := px + py * gridX
    let count := gridY * groupM
    let group := pid / count
    pid % count / groupSize gridX group groupM

theorem identity (px py gx gy : Nat) :
    rowTile px py gx gy 1 = px ∧ columnTile px py gx gy 1 = py := by
  simp [rowTile, columnTile]

theorem linear_index_bound (px py gx gy : Nat) (hx : px < gx) (hy : py < gy) :
    px + py * gx < gx * gy := by
  have h := Nat.mul_le_mul_right gx (Nat.succ_le_of_lt hy)
  nlinarith

theorem group_base_bound (pid gx gy gm : Nat)
    (hpid : pid < gx * gy) : pid / (gy * gm) * gm < gx := by
  have h := Nat.div_mul_le_self pid (gy * gm)
  nlinarith

/-- A final short group uses precisely the number of rows still in the grid. -/
theorem tail_size (gx group gm : Nat) (hgm : 0 < gm)
    (hbase : group * gm < gx) (htail : gx < group * gm + gm) :
    gx % gm = gx - group * gm := by
  have hq : gx / gm = group := by
    have hlow := (Nat.le_div_iff_mul_le hgm).mpr (Nat.le_of_lt hbase)
    have hhigh : gx / gm < group + 1 := by
      apply (Nat.div_lt_iff_lt_mul hgm).mpr
      nlinarith
    omega
  have h := Nat.mod_add_div gx gm
  rw [hq, Nat.mul_comm gm group] at h
  omega

theorem group_size_positive (gx group gm : Nat) (hgm : 0 < gm) (hbase : group * gm < gx) :
    0 < groupSize gx group gm := by
  unfold groupSize
  split
  · rw [tail_size gx group gm hgm hbase (by omega)]
    omega
  · exact hgm

theorem group_span (gx group gm : Nat) (hgm : 0 < gm) (hbase : group * gm < gx) :
    group * gm + groupSize gx group gm ≤ gx := by
  unfold groupSize
  split
  · rw [tail_size gx group gm hgm hbase (by omega)]
    omega
  · omega

theorem inner_bound (pid gx gy gm : Nat) (hgy : 0 < gy) (hgm : 0 < gm)
    (hpid : pid < gx * gy) :
    pid % (gy * gm) < gy * groupSize gx (pid / (gy * gm)) gm := by
  have hbase := group_base_bound pid gx gy gm hpid
  unfold groupSize
  split
  · rw [tail_size gx (pid / (gy * gm)) gm hgm hbase (by omega)]
    have h := Nat.mod_add_div pid (gy * gm)
    have hs := Nat.sub_add_cancel (Nat.le_of_lt hbase)
    nlinarith
  · exact Nat.mod_lt pid (Nat.mul_pos hgy hgm)

theorem coordinates_in_grid (px py gx gy gm : Nat) (hgm : 0 < gm)
    (hx : px < gx) (hy : py < gy) :
    rowTile px py gx gy gm < gx ∧ columnTile px py gx gy gm < gy := by
  by_cases hg : gm = 1
  · simpa [rowTile, columnTile, hg] using And.intro hx hy
  · let pid := px + py * gx
    let group := pid / (gy * gm)
    have hpid : pid < gx * gy := linear_index_bound px py gx gy hx hy
    have hbase := group_base_bound pid gx gy gm hpid
    have hsize : 0 < groupSize gx group gm := group_size_positive gx group gm hgm hbase
    have hspan := group_span gx group gm hgm hbase
    have hinner := inner_bound pid gx gy gm (by omega) hgm hpid
    have hrow := Nat.mod_lt (pid % (gy * gm)) hsize
    have hcol := (Nat.div_lt_iff_lt_mul hsize).mpr hinner
    simp only [rowTile, columnTile, hg, ↓reduceIte]
    change group * gm + pid % (gy * gm) % groupSize gx group gm < gx ∧
      pid % (gy * gm) / groupSize gx group gm < gy
    exact ⟨by omega, hcol⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealGrid
