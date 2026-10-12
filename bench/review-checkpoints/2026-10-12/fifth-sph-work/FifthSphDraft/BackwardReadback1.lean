import FifthSphDraft.BackwardReadbackCommon
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback
open VeriTile.Triton BackwardBase
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
theorem component1 (X dX g : RegionName) (B N M off stride : Nat) (u v : BlockState)
    (he : exec (source X dX g B N M off stride) u=some v) (j : Fin B)
    (hcoords : ∀ d : Fin 3,coord u.pid B j.val d.val<N)
    (hg : ∀ d : Fin 11,output u.pid B stride off j.val d.val<M) :
    v.readMem dX (coord u.pid B j.val 1)=
      RealMath.gradient 1 (u.readMem X (coord u.pid B j.val 0))
        (u.readMem X (coord u.pid B j.val 1)) (u.readMem X (coord u.pid B j.val 2))
        (u.readMem dX (coord u.pid B j.val 1))
        (fun d => u.readMem g (output u.pid B stride off j.val d.val)) := by
  change exec (NormalizedSource.original X dX g B N M off stride) u=some v at he
  rw [NormalizedSource.original_eq_algorithm] at he
  change stepStmts (NormalizedSource.algorithm X dX g B N M off stride).body u=some v at he
  obtain ⟨t0,ht0,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht0
  subst t0
  obtain ⟨t1,ht1,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht1
  subst t1
  obtain ⟨t2,ht2,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht2
  subst t2
  obtain ⟨t3,ht3,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht3
  subst t3
  obtain ⟨t4,ht4,he⟩ := assigns_inv he
  obtain ⟨t5,ht5,he⟩ := assigns_inv he
  obtain ⟨t6,ht6,he⟩ := assigns_inv he
  obtain ⟨t7,ht7,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht7
  subst t7
  obtain ⟨t8,ht8,he⟩ := assigns_inv he
  simp [evalOp.eq_def,tile_elementwise] at ht8
  subst t8
  obtain ⟨t9,ht9,he⟩ := assigns_inv he
  obtain ⟨t10,ht10,he⟩ := assigns_inv he
  obtain ⟨t11,ht11,he⟩ := assigns_inv he
  obtain ⟨t12,ht12,he⟩ := assigns_inv he
  obtain ⟨t13,ht13,he⟩ := assigns_inv he
  obtain ⟨t14,ht14,he⟩ := assigns_inv he
  obtain ⟨t15,ht15,he⟩ := assigns_inv he
  obtain ⟨t16,ht16,he⟩ := assigns_inv he
  obtain ⟨t17,ht17,he⟩ := assigns_inv he
  obtain ⟨t18,ht18,he⟩ := assigns_inv he
  obtain ⟨t19,ht19,he⟩ := assigns_inv he
  obtain ⟨t20,ht20,he⟩ := assigns_inv he
  obtain ⟨t21,ht21,he⟩ := assigns_inv he
  obtain ⟨t22,ht22,he⟩ := assigns_inv he
  obtain ⟨t23,ht23,he⟩ := assigns_inv he
  obtain ⟨t24,ht24,he⟩ := assigns_inv he
  obtain ⟨t25,ht25,he⟩ := assigns_inv he
  obtain ⟨t26,ht26,he⟩ := assigns_inv he
  obtain ⟨t27,ht27,he⟩ := assigns_inv he
  obtain ⟨t28,ht28,he⟩ := assigns_inv he
  obtain ⟨t29,ht29,he⟩ := assigns_inv he
  obtain ⟨t30,ht30,he⟩ := assigns_inv he
  obtain ⟨t31,ht31,he⟩ := assigns_inv he
  obtain ⟨t32,ht32,he⟩ := assigns_inv he
  obtain ⟨t33,ht33,he⟩ := assigns_inv he
  obtain ⟨t34,ht34,he⟩ := assigns_inv he
  obtain ⟨t35,ht35,he⟩ := assigns_inv he
  obtain ⟨t36,ht36,he⟩ := assigns_inv he
  obtain ⟨t37,ht37,he⟩ := assigns_inv he
  obtain ⟨t38,ht38,he⟩ := assigns_inv he
  obtain ⟨t39,ht39,he⟩ := assigns_inv he
  obtain ⟨t40,ht40,he⟩ := assigns_inv he
  obtain ⟨t41,ht41,he⟩ := assigns_inv he
  obtain ⟨t42,ht42,he⟩ := assigns_inv he
  obtain ⟨t43,ht43,he⟩ := assigns_inv he
  obtain ⟨t44,ht44,he⟩ := assigns_inv he
  obtain ⟨t45,ht45,he⟩ := assigns_inv he
  obtain ⟨t46,ht46,he⟩ := assigns_inv he
  obtain ⟨t47,ht47,he⟩ := assigns_inv he
  obtain ⟨t48,ht48,he⟩ := assigns_inv he
  obtain ⟨t49,ht49,he⟩ := assigns_inv he
  obtain ⟨t50,ht50,he⟩ := assigns_inv he
  obtain ⟨t51,ht51,he⟩ := assigns_inv he
  obtain ⟨t52,ht52,he⟩ := assigns_inv he
  obtain ⟨t53,ht53,he⟩ := assigns_inv he
  obtain ⟨t54,ht54,he⟩ := assigns_inv he
  obtain ⟨t55,ht55,he⟩ := assigns_inv he
  obtain ⟨t56,ht56,he⟩ := assigns_inv he
  obtain ⟨t57,ht57,he⟩ := assigns_inv he
  obtain ⟨t58,ht58,he⟩ := assigns_inv he
  obtain ⟨t59,ht59,he⟩ := assigns_inv he
  obtain ⟨t60,ht60,he⟩ := assigns_inv he
  obtain ⟨t61,ht61,he⟩ := assigns_inv he
  obtain ⟨t62,ht62,he⟩ := assigns_inv he
  obtain ⟨t63,ht63,he⟩ := assigns_inv he
  obtain ⟨t64,ht64,he⟩ := assigns_inv he
  obtain ⟨t65,ht65,he⟩ := assigns_inv he
  obtain ⟨t66,ht66,he⟩ := assigns_inv he
  obtain ⟨t67,ht67,he⟩ := assigns_inv he
  obtain ⟨t68,ht68,he⟩ := assigns_inv he
  obtain ⟨t69,ht69,he⟩ := assigns_inv he
  obtain ⟨t70,ht70,he⟩ := assigns_inv he
  obtain ⟨t71,ht71,he⟩ := assigns_inv he
  obtain ⟨t72,ht72,he⟩ := assigns_inv he
  obtain ⟨t73,ht73,he⟩ := assigns_inv he
  obtain ⟨t74,ht74,he⟩ := assigns_inv he
  obtain ⟨t75,ht75,he⟩ := assigns_inv he
  obtain ⟨t76,ht76,he⟩ := assigns_inv he
  obtain ⟨t77,ht77,he⟩ := assigns_inv he
  obtain ⟨t78,ht78,he⟩ := assigns_inv he
  obtain ⟨t79,ht79,he⟩ := assigns_inv he
  obtain ⟨t80,ht80,he⟩ := assigns_inv he
  obtain ⟨t81,ht81,he⟩ := assigns_inv he
  obtain ⟨t82,ht82,he⟩ := assigns_inv he
  obtain ⟨t83,ht83,he⟩ := assigns_inv he
  obtain ⟨t84,ht84,he⟩ := assigns_inv he
  obtain ⟨t85,ht85,he⟩ := assigns_inv he
  obtain ⟨t86,ht86,he⟩ := assigns_inv he
  clear ht86
  obtain ⟨t87,ht87,he⟩ := assigns_inv he
  obtain ⟨t88,ht88,he⟩ := assigns_inv he
  clear ht88
  simp (maxSteps := 3000000) [stepStmts,stepStmt,evalOp.eq_def,tile_elementwise] at he
  subst v
  simp only [coord,Nat.add_zero,BlockState.pid]
  rw [BlockState.foldl_writeMem_same_region_disjoint_offsets_readMem
    (region := dX) (l := TileShape.allIndices [B])
    (off := j.val*3+B*3*u.pids 0+1)
    (P := fun idx : TileIndex [B] => idx.1.val*3+B*3*u.pids 0+2<N)
    (offsetFn := fun idx : TileIndex [B] => idx.1.val*3+B*3*u.pids 0+2)
    (hOff := fun k _ _ => by simpa [coord,BlockState.pid] using coord_disjoint u.pid B j k 1 2 (by decide))]
  rw [BlockState.scatter_readback_prop_masked_nd (region := dX) _
    (fun idx : TileIndex [B] => idx.1.val*3+B*3*u.pids 0+1) _ _
    (by simpa only [Nat.add_zero,BlockState.pid] using coord_injective u.pid B 1) (j,PUnit.unit)]
  have hc0 := hcoords 0
  have hc1 := hcoords 1
  have hc2 := hcoords 2
  have hg0 := hg 0
  have hg1 := hg 1
  have hg2 := hg 2
  have hg3 := hg 3
  have hg4 := hg 4
  have hg5 := hg 5
  have hg6 := hg 6
  have hg7 := hg 7
  have hg8 := hg 8
  have hg9 := hg 9
  have hg10 := hg 10
  norm_num [coord,output,BlockState.pid] at hc0 hc1 hc2 hg0 hg1 hg2 hg3 hg4 hg5 hg6 hg7 hg8 hg9 hg10
  simp only [hc0,hc1,hc2,if_true]
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht4
  have h4 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht4)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 4"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10] at h4
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht5
  have h5 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht5)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 5"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4] at h5
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht6
  have h6 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht6)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 6"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5] at h6
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht9
  have h9 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht9)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 9"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6] at h9
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht10
  have h10 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht10)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 10"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9] at h10
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht11
  have h11 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht11)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 11"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10] at h11
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht12
  have h12 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht12)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 12"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11] at h12
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht13
  have h13 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht13)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 13"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12] at h13
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht14
  have h14 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht14)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 14"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13] at h14
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht15
  have h15 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht15)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 15"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14] at h15
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht16
  have h16 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht16)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 16"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15] at h16
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht17
  have h17 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht17)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 17"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16] at h17
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht18
  have h18 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht18)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 18"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17] at h18
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht19
  have h19 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht19)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 19"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18] at h19
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht20
  have h20 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht20)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 20"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19] at h20
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht21
  have h21 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht21)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 21"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20] at h21
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht22
  have h22 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht22)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 22"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21] at h22
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht23
  have h23 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht23)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 23"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22] at h23
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht24
  have h24 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht24)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 24"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23] at h24
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht25
  have h25 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht25)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 25"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24] at h25
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht26
  have h26 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht26)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 26"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25] at h26
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht27
  have h27 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht27)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 27"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26] at h27
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht28
  have h28 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht28)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 28"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27] at h28
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht29
  have h29 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht29)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 29"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28] at h29
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht30
  have h30 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht30)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 30"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29] at h30
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht31
  have h31 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht31)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 31"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30] at h31
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht32
  have h32 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht32)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 32"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31] at h32
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht33
  have h33 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht33)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 33"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32] at h33
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht34
  have h34 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht34)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 34"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33] at h34
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht35
  have h35 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht35)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 35"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34] at h35
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht36
  have h36 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht36)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 36"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35] at h36
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht37
  have h37 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht37)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 37"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36] at h37
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht38
  have h38 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht38)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 38"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37] at h38
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht39
  have h39 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht39)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 39"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38] at h39
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht40
  have h40 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht40)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 40"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39] at h40
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht41
  have h41 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht41)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 41"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40] at h41
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht42
  have h42 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht42)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 42"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41] at h42
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht43
  have h43 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht43)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 43"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42] at h43
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht44
  have h44 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht44)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 44"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43] at h44
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht45
  have h45 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht45)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 45"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44] at h45
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht46
  have h46 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht46)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 46"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45] at h46
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht47
  have h47 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht47)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 47"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46] at h47
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht48
  have h48 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht48)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 48"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47] at h48
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht49
  have h49 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht49)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 49"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48] at h49
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht50
  have h50 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht50)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 50"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49] at h50
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht51
  have h51 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht51)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 51"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50] at h51
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht52
  have h52 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht52)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 52"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51] at h52
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht53
  have h53 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht53)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 53"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52] at h53
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht54
  have h54 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht54)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 54"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53] at h54
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht55
  have h55 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht55)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 55"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54] at h55
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht56
  have h56 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht56)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 56"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55] at h56
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht57
  have h57 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht57)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 57"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56] at h57
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht58
  have h58 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht58)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 58"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57] at h58
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht59
  have h59 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht59)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 59"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58] at h59
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht60
  have h60 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht60)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 60"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59] at h60
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht61
  have h61 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht61)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 61"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60] at h61
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht62
  have h62 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht62)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 62"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61] at h62
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht63
  have h63 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht63)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 63"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62] at h63
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht64
  have h64 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht64)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 64"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63] at h64
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht65
  have h65 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht65)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 65"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64] at h65
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht66
  have h66 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht66)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 66"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65] at h66
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht67
  have h67 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht67)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 67"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66] at h67
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht68
  have h68 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht68)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 68"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67] at h68
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht69
  have h69 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht69)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 69"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68] at h69
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht70
  have h70 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht70)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 70"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69] at h70
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht71
  have h71 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht71)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 71"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70] at h71
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht72
  have h72 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht72)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 72"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71] at h72
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht73
  have h73 := congrArg (fun v => v.data PUnit.unit) (Option.some.inj ht73)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 73"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72] at h73
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht74
  have h74 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht74)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 74"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73] at h74
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht75
  have h75 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht75)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 75"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74] at h75
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht76
  have h76 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht76)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 76"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75] at h76
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht77
  have h77 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht77)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 77"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75,←h76] at h77
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht78
  have h78 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht78)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 78"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75,←h76,←h77] at h78
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht79
  have h79 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht79)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 79"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75,←h76,←h77,←h78] at h79
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht80
  have h80 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht80)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 80"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75,←h76,←h77,←h78,←h79] at h80
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht81
  have h81 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht81)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 81"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75,←h76,←h77,←h78,←h79,←h80] at h81
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht82
  have h82 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht82)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 82"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75,←h76,←h77,←h78,←h79,←h80,←h81] at h82
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht83
  have h83 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht83)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 83"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75,←h76,←h77,←h78,←h79,←h80,←h81,←h82] at h83
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht84
  have h84 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht84)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 84"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75,←h76,←h77,←h78,←h79,←h80,←h81,←h82,←h83] at h84
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht85
  have h85 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht85)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 85"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75,←h76,←h77,←h78,←h79,←h80,←h81,←h82,←h83,←h84] at h85
  simp only [↓evalOp.eq_def,↓BlockState.setReg_same,↓BlockState.setReg_ne_name,Option.bind_some,ne_eq,String.reduceEq,not_false_eq_true,↓reduceDIte,↓reduceIte] at ht87
  have h87 := congrArg (fun v => v.data (j,PUnit.unit)) (Option.some.inj ht87)
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "lane 86"
  simp (maxSteps := 3000000) [↓BlockState.setReg_readMemValue,↓BlockState.setReg_readMem,↓BlockState.setReg_undef,↓BlockState.setReg_mem,evalOp.eq_def,tile_elementwise,NumericDType.mul,NumericDType.add,NumericDType.sub,WithBot.realSub,WithBot.realAdd,WithBot.realMul,Option.map₂,Option.map,WithBot.unbotD,Option.bind,Option.bind_some,WithBot.recBotCoe,Region.cast,hc0,hc1,hc2,hg0,hg1,hg2,hg3,hg4,hg5,hg6,hg7,hg8,hg9,hg10,←h4,←h5,←h6,←h9,←h10,←h11,←h12,←h13,←h14,←h15,←h16,←h17,←h18,←h19,←h20,←h21,←h22,←h23,←h24,←h25,←h26,←h27,←h28,←h29,←h30,←h31,←h32,←h33,←h34,←h35,←h36,←h37,←h38,←h39,←h40,←h41,←h42,←h43,←h44,←h45,←h46,←h47,←h48,←h49,←h50,←h51,←h52,←h53,←h54,←h55,←h56,←h57,←h58,←h59,←h60,←h61,←h62,←h63,←h64,←h65,←h66,←h67,←h68,←h69,←h70,←h71,←h72,←h73,←h74,←h75,←h76,←h77,←h78,←h79,←h80,←h81,←h82,←h83,←h84,←h85] at h87
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "polynomial equation derived"
  rw [←h87]
  simp only [WithBot.unbotD,WithBot.recBotCoe]
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "readback substituted"
  norm_num [RealMath.gradient,RealMath.backwardWeight,Fin.sum_univ_succ,coord,output,BlockState.pid]
  run_tac do IO.FS.writeFile "/tmp/fifth-sph-readback-1-progress.txt" "normalization complete"
  ring

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.BackwardReadback
