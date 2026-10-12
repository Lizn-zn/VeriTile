from pathlib import Path
D=Path('/tmp/FifthSphDraft')
math=(D/'RealMath.lean').read_text();math=math[math.index('def output'):math.index('/-- At (1,0,0)')]
s='''import FifthSphDraft.ForwardIO
import FifthSphDraft.BackwardIO
/-! Complete original/candidate real IO contracts for both source JITs.
The polynomial tables use the source decimal coefficients as exact rationals.
Backward adds the source update to the old coordinate gradient. Its rounded
coefficient table is not the exact derivative of the forward table; the
formal coefficient discrepancy is in Proofs/RealMath. Host launch composition,
fixed-width overflow, races and physical floating-point accuracy are excluded.
-/
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Correctness
open VeriTile VeriTile.Triton TypedSource
noncomputable section
'''+math+'\n'
f=(D/'ForwardIO.lean').read_text();defs=f[f.index('def coord'):f.index('theorem polynomial_eq')];defs=defs.replace('structure Bounds','structure ForwardBounds').replace('Bounds X Y','ForwardBounds X Y').replace('def Cover','def ForwardCover').replace('→ Cover B','→ ForwardCover B').replace('def Outside','def forwardOutside').replace('Outside A u Y','forwardOutside A u Y').replace('def Implements','def forward_correct_target').replace('RealMath.output','output');s+=defs
for swapped,label,proof in [('Bool.false','','typed_correct'),('Bool.true','index_commuted_','candidate_correct')]:
 s+=f'''specification forward_{label}correctness (input outputStorage : Storage) (X Y : RegionName) (B N M off stride : Nat) :
    Spec.Real (forward_correct_target (forward input outputStorage {swapped} X Y B N M off stride).eraseDType.toAlgKernel X Y B N M off stride) := by
  intro A hd ha u hu hb hstride hcover
  have hb' : ForwardIO.Bounds X Y B N M off stride u A.extent := ⟨hb.coordinates,hb.output⟩
  exact ForwardIO.{proof} input outputStorage X Y B N M off stride A hd ha u hu hb' hstride hcover

'''
b=(D/'BackwardIO.lean').read_text();defs=b[b.index('/-- Every active'):b.index('theorem canonical_correct')];defs=defs.replace('def Cover','def BackwardCover').replace('  Cover u.pid','  BackwardCover u.pid').replace('def Outside','def backwardOutside').replace('Outside A u dX','backwardOutside A u dX').replace('def Implements','def backward_correct_target').replace('Bounds X dX','BackwardBounds X dX').replace('RealMath.gradient','gradient').replace('output pid','outputAddress pid').replace('output u.pid','outputAddress u.pid')
bounds=(D/'BackwardBase.lean').read_text();bounds=bounds[bounds.index('structure Bounds'):bounds.index('theorem trace_safe')].replace('structure Bounds','structure BackwardBounds').replace('output u.pid','outputAddress u.pid');s+=bounds+'\n'+defs
for swapped,label,proof in [('Bool.false','','typed_correct'),('Bool.true','index_commuted_','candidate_correct')]:
 s+=f'''specification backward_{label}correctness (input gradientStorage upstream : Storage) (X dX g : RegionName) (B N M off stride : Nat) :
    Spec.Real (backward_correct_target (backward input gradientStorage upstream {swapped} X dX g B N M off stride).eraseDType.toAlgKernel X dX g B N M off stride) := by
  intro A hd ha u hu hb hcover
  have hb' : BackwardBase.Bounds X dX g B N M off stride u A.extent := ⟨hb.coordinates,hb.gradient,hb.upstream⟩
  exact BackwardIO.{proof} input gradientStorage upstream X dX g B N M off stride A hd ha u hu hb' hcover

'''
s+='''#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Correctness
''';(D/'Correct.lean').write_text(s)
f=(D/'FP.lean').read_text();f=f.replace('import FifthSphDraft.Kernels\nimport bench.tritonbench_g.destindex_copy.Proofs.IndexRewrite\nimport VeriTile.Meta.StatementAudit','''import FifthSphDraft.FP
/-! Whole typed-state/failure equivalence. The only change commutes the natural
coordinate-block product. Original and candidate retain all source statements,
fp16/bf16/fp32 input/output choices and independent backward gradient/upstream
types. Input powers retain input precision, named decimal coefficients are fp32
scalar tensors, coefficient expressions promote to fp32, and each store casts
to its destination storage. Backward retains load-then-add of the old gradient.
The input domain and numerical FP atom list are empty. -/''').replace('FifthOrderSphHarmonics.FPProof','FifthOrderSphHarmonics.Equivalence').replace('theorem forward_equivalent','specification forward_equiv').replace('theorem backward_equivalent','specification backward_equiv').replace('  cases input <;> cases output <;> equiv_decompose <;> fp_prove','  exact FPProof.forward_equivalent input output X Y B N M off stride').replace('  cases input <;> cases gradient <;> cases upstream <;> equiv_decompose <;> fp_prove','  exact FPProof.backward_equivalent input gradient upstream X dX g B N M off stride').replace('#print_fp_assumptions forward_equivalent','#print_fp_assumptions forward_equiv').replace('#print_fp_assumptions backward_equivalent','#print_fp_assumptions backward_equiv');(D/'FPEquiv.lean').write_text(f)
