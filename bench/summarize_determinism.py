#!/usr/bin/env python3
"""Exact before/after token gate and measured throughput; no tolerance acceptance."""
import hashlib,json,statistics
from pathlib import Path
from determinism import OUT, compare

def first(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), None)

if __name__=='__main__':
    rows=json.loads((OUT/'runs.json').read_text())
    groups={g:[r for r in rows if r['name'].startswith(g+'-A-')] for g in ['before','after','identity']}
    gates={g:compare([r['name'] for r in rs]) for g,rs in groups.items()}
    old=json.loads((OUT/'identity-reference-ids.json').read_text())
    new=json.loads((OUT/'logs'/ (groups['identity'][0]['name']+'-ids.json')).read_text())
    census=json.loads((OUT/'census-before.json').read_text())
    actual={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in census}
    (OUT/'census-after.json').write_text(json.dumps(actual,indent=2)+'\n')
    timings={g:dict(emitted_convention_tps=[r['emitted_decode_tps'] for r in rs],median=statistics.median(r['emitted_decode_tps'] for r in rs),forward_median=statistics.median(r['decode_forward_tps'] for r in rs),wall_median=statistics.median(255/(r['wall_seconds']-r['prefill_ms']/1000) for r in rs)) for g,rs in groups.items()}
    timings['reference_tps']=8.855
    timings['change_vs_reference_pct']=100*(timings['after']['median']/8.855-1)
    timings['change_vs_current_before_pct']=100*(timings['after']['median']/timings['before']['median']-1)
    # Identical generation flags, ignoring executable pathname only.
    same_flags=len({tuple(r['command'][1:]) for g in ['before','after'] for r in groups[g]})==1
    proof=(len(groups['before'])>=4 and len(groups['after'])>=8 and len(groups['identity'])>=4
           and not gates['before']['passed'] and gates['after']['passed'] and gates['identity']['passed']
           and all(r['valid'] and r['exit']==0 and r['token_count']==256 for rs in groups.values() for r in rs)
           and len({r['binary_sha256'] for g in ['after','identity'] for r in groups[g]})==1
           and all(r['scope_limits']==['107374182400','118111600640','1073741824'] for rs in groups.values() for r in rs)
           and same_flags and actual==census)
    summary=dict(proven=proof,gates=gates,timings=timings,identical_flags=same_flags,census_unchanged=actual==census,identity_reference_sha256='204b072b7633fe7dd380b44530ded033da71208f1e8b67f87f3550ce5c27fd86',identity_sha256=groups['identity'][0]['ids_sha256'],identity_first_difference=first(old,new))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='gates'},indent=2))
    raise SystemExit(0 if proof else 1)
