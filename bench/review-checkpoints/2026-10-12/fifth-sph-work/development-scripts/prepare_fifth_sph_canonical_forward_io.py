from pathlib import Path
import re,subprocess,json
root=Path('/tmp/FifthSphCanonical');draft=Path('/tmp/FifthSphDraft')
s=(draft/'Erasure.lean').read_text().replace('import FifthSphDraft.Kernels','import FifthSphCanonical.Kernels')
s=re.sub(r'(?:forward|backward)_(?:fp16|bf16|fp32)(?:_(?:fp16|bf16|fp32)){1,2},',lambda m: m[0] if m[0] in ['forward_fp32_fp32,','backward_fp32_fp32_fp32,'] else '',s)
(root/'Erasure.lean').write_text(s)
for module in ['Candidate','ForwardIO']:
 s=(draft/(module+'.lean')).read_text()
 for dep in ['Erasure','Candidate','ForwardReal']:s=s.replace('import FifthSphDraft.'+dep,'import FifthSphCanonical.'+dep)
 (root/(module+'.lean')).write_text(s)
s=(draft/'Correct.lean').read_text().split('structure BackwardBounds')[0]
s=s.replace('import FifthSphDraft.ForwardIO','import FifthSphCanonical.ForwardIO').replace('import FifthSphDraft.BackwardIO\n','')
s+='\n#guard_msgs (drop info) in\n#auditModuleAxioms\nend\nend VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Correctness\n'
(root/'ForwardCorrect.lean').write_text(s)
s=(draft/'Execution.lean').read_text().split('def BackwardResult')[0].replace('import FifthSphDraft.Correct','import FifthSphCanonical.ForwardCorrect')
s+='\n#guard_msgs (drop info) in\n#auditModuleAxioms\nend\nend VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.Execution\n'
(root/'ForwardExecution.lean').write_text(s)
rows=[]
for module,label in [('Erasure','erasure1'),('Candidate','candidate1'),('ForwardIO','forward-io1'),('ForwardCorrect','forward-correct1'),('ForwardExecution','forward-execution1')]:
 r=subprocess.run(['python3','/tmp/check_fifth_sph_canonical.py',module,label]);rows.append(dict(module=module,label=label,code=r.returncode))
 Path('/tmp/fifth-sph-canonical-forward-io-results.json').write_text(json.dumps(rows,indent=2)+'\n');assert r.returncode==0,module
