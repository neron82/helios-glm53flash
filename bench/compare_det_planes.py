#!/usr/bin/env python3
"""Compare ordered row fingerprints; these are diagnostics, not a byte-equality gate."""
import argparse,json,re
from pathlib import Path

def parse(path):
    return [dict(pos=int(p),layer=int(l),name=n,row_bytes=int(b),hashes=h.split(','))
            for p,l,n,b,h in re.findall(r'\[det-plane\] pos=(\d+) layer=(\d+) name=(\w+) row_bytes=(\d+) hashes=([0-9a-f,]+)',Path(path).read_text())]

def compare(a,b):
    x,y=parse(a),parse(b); assert len(x)==len(y) and x, (len(x),len(y))
    differences=[]
    for i,(u,v) in enumerate(zip(x,y)):
        assert {k:u[k] for k in u if k!='hashes'}=={k:v[k] for k in v if k!='hashes'}
        assert len(u['hashes'])==len(v['hashes'])
        bad=[r for r,(c,d) in enumerate(zip(u['hashes'],v['hashes'])) if c!=d]
        if bad: differences.append(dict(observation=i,**{k:u[k] for k in u if k!='hashes'},first_row=bad[0],differing_rows=len(bad)))
    return dict(a=a,b=b,observations=len(x),first_difference=next(iter(differences),None),first_decode_difference=next((d for d in differences if d['pos']==4096),None),differences=differences,limitation='FNV-1a 64-bit row fingerprints; no claim of exact byte equality from matching fingerprints.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('a');p.add_argument('b');p.add_argument('output');args=p.parse_args()
    result=compare(args.a,args.b);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='differences'},indent=2))
