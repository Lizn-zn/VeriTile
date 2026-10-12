from pathlib import Path
import datetime,hashlib,json
w=Path(__file__).resolve().parent;p=w/'FlashAttnDraft';record=json.loads((w/'review.json').read_text())
files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.glob('*.lean')) if f.stem not in ['ErasureFast'] and not f.stem.endswith(('Probe','Debug'))}
checks=[]
for name,sha in files.items():
 r=w/'validation'/(Path(name).stem+'.result.json')
 if r.exists():
  result=json.loads(r.read_text())
  if result['code']==0 and result['source_sha256']==sha:checks.append(str(r.relative_to(w.parent.parent)))
record.update(files=files,passing_checks=checks,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted=False,repository_integrated=False)
record['status']='Full typed source, source comparison, whole-state FP equivalence, abstract typed nonempty executions, independent real formulas, online-softmax algebra and source erasure passed local checks. General real execution/IO proof, finite real witness, integration and independent replay remain pending.'
record['pending']=['Connect the full symbolic real loop execution to the independent actual-source output and LSE formulas','Prove actual unmasked read/write bounds and preservation of every other tagged cell under finite-flat execution','Validate a nonempty finite real-memory execution','Repository integration and independent comparator/trust replay']
(w/'review.json').write_text(json.dumps(record,indent=2)+'\n');print(len(checks),'current-source local Lean checks pass;',len(files),'selected Lean files; acceptance remains pending')
