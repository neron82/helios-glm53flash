#!/usr/bin/env python3
"""Sequential, memory-gated measurements; checkpoint mounts are read-only."""
import argparse, hashlib, json, os, pathlib, re, statistics, subprocess, threading, time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'bench/results/determinism'
LAUNCHER = '/home/neron/models/helios_glm53flash_abl_server.sh'
MODELS = {'A': '/home/neron/models/glm53flash_abl', 'B': '/home/neron/models/glm53flash'}

def call(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout

def sha(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()

def health():
    try:
        return json.loads(call(['curl', '-fsS', '--max-time', '2', 'http://127.0.0.1:8080/health'])) == {'status': 'ok'}
    except Exception:
        return False

def restore():
    unit = 'helios-prod-restore-det-' + str(time.time_ns())
    call(['systemd-run', '--user', '--unit=' + unit, '--collect', '-p', 'Type=oneshot',
          '-p', 'RemainAfterExit=yes', '-p', 'KillMode=none', '--', LAUNCHER, 'start'])
    deadline = time.monotonic() + 900
    while not health():
        if time.monotonic() > deadline:
            raise RuntimeError('Production restoration timeout: ' + unit)
        time.sleep(1)
    pid = pathlib.Path(os.environ.get('XDG_RUNTIME_DIR', '/tmp') + '/helios-glm53flash-abl-8080.pid').read_text().split()[0]
    result = {'unit': unit, 'health': json.loads(call(['curl', '-fsS', 'http://127.0.0.1:8080/health'])),
              'models': json.loads(call(['curl', '-fsS', 'http://127.0.0.1:8080/v1/models'])),
              'binary_sha256': sha('/proc/' + pid + '/exe'),
              'omnivoice': call(['systemctl', '--user', 'is-active', 'omnivoice.service']).strip(), 'pid': int(pid)}
    (OUT / 'production-state.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['binary_sha256'] == '1a9cf7dc0557207a717746d796d224a5a94d4e99bf837c95f839061752c7a813'
    assert any(m['id'] == 'glm-5.3-flash-exl3' and m['context_length'] == 262144 for m in result['models']['data'])
    result['bge_health'] = json.loads(call(['curl', '-fsS', 'http://127.0.0.1:7777/health']))
    assert pathlib.Path('/proc/26170').exists()
    result['bge_listener'] = call(['ss', '-H', '-ltnp', 'sport = :7777']).strip()
    assert 'pid=26170,' in result['bge_listener']
    result['omnivoice_health'] = json.loads(call(['curl', '-fsS', 'http://127.0.0.1:8880/health']))
    result['bge_pid'] = 26170
    (OUT / 'production-state.json').write_text(json.dumps(result, indent=2) + '\n')
    return result

def sample(stop, path):
    with path.open('w') as f:
        while not stop.is_set():
            raw = call(['nvidia-smi', '--query-gpu=index,pci.bus_id,memory.free', '--format=csv,noheader,nounits'])
            gpus = [{'index': int(a), 'pci': b.strip(), 'free_mib': int(c)}
                    for a, b, c in (line.split(',') for line in raw.strip().splitlines())]
            f.write(json.dumps({'time': time.time(), 'gpus': gpus}) + '\n'); f.flush()
            stop.wait(1)

def run(name, binary, model, prompt):
    assert not health(), 'Production must be stopped before model runs'
    assert call(['systemctl', '--user', 'is-active', 'omnivoice.service']).strip() == 'active'
    assert 'llama' in call(['curl', '-fsS', 'http://127.0.0.1:7777/health']) or json.loads(call(['curl', '-fsS', 'http://127.0.0.1:7777/health']))['status'] == 'ok'
    command = [str(ROOT / binary), 'gen', MODELS[model], '--cap', '262144', '--chunk', '4096',
               '--tokens', '256', '--temp', '0', '--ignore-eos', '--prefix-snap-mb', '0',
               '--prompt-ids', str(OUT / f'prompt-{prompt}.json')]
    # Check limits in the exact scope that execs the model. No checkpoint/census writes possible.
    shell = 'cg=$(cut -d: -f3 /proc/self/cgroup); cat /sys/fs/cgroup"$cg"/memory.{high,max,swap.max}; exec "$@"'
    wrapped = ['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemoryHigh=100G', '-p',
               'MemoryMax=110G', '-p', 'MemorySwapMax=1G', '--', 'bash', '-c', shell, 'm1-gate',
               '/usr/bin/time', '-v', 'bwrap', '--ro-bind', '/', '/', '--dev-bind', '/dev', '/dev',
               '--proc', '/proc', '--', 'env', 'HELIOS_MTP_TOKENS=1', 'HELIOS_BENCH_PHASES=1', *command]
    logpath = OUT / 'logs' / (name + '.log')
    stop = threading.Event(); thread = threading.Thread(target=sample, args=(stop, OUT / 'logs' / (name + '-vram.jsonl')))
    thread.start(); start = time.monotonic()
    try:
        with logpath.open('w') as log:
            proc = subprocess.run(wrapped, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    finally:
        stop.set(); thread.join()
    raw = logpath.read_text(errors='replace')
    tokens = [int(x) for x in re.findall(r'\[tok\] (\d+)', raw)]
    (OUT / 'logs' / (name + '-ids.json')).write_text(json.dumps(tokens) + '\n')
    records = json.loads((OUT / 'runs.json').read_text()) if (OUT / 'runs.json').exists() else []
    data = {'name': name, 'model': model, 'prompt': prompt, 'binary': binary, 'binary_sha256': sha(ROOT / binary),
            'command': command, 'gate_command': wrapped, 'exit': proc.returncode,
            'process_wall_seconds': time.monotonic() - start, 'log': str(logpath.relative_to(ROOT)),
            'default_census': True, 'census_read_only': True, 'token_count': len(tokens)}
    bench = re.findall(r'\[bench-json\] (\{[^\n]+\})', raw)
    if bench:
        data.update(json.loads(bench[-1]))
        data['emitted_decode_tps'] = data['generated_tokens'] * 1000 / data['decode_ms']
        data['prefill_tps'] = data['prefill_tokens'] * 1000 / data['prefill_ms']
    for phase, lk, hits, streams, evictions, byte_count in re.findall(
            r'\[bench-slots\] (prefill|decode) lookups=(\d+) resident_hits=(\d+) streams=(\d+) evictions=(\d+) h2d_bytes=(\d+)', raw):
        data[phase + '_slots'] = dict(zip(['lookups', 'resident_hits', 'streams', 'evictions', 'h2d_bytes'],
                                        map(int, [lk, hits, streams, evictions, byte_count])))
    for phase, layer, streams in re.findall(r'\[bench-layer\] (prefill|decode) layer=(\d+) streams=(\d+)', raw):
        data.setdefault(phase + '_layer_streams', {})[layer] = int(streams)
    for label, pattern in [('cache_line', r'^\[cache\].*'), ('model_line', r'^\[model\] jobs=.*'), ('slots_line', r'^\[slots\] lookups=.*')]:
        matches = re.findall(pattern, raw, re.M); data[label] = matches[-1] if matches else None
    rss = re.search(r'Maximum resident set size \(kbytes\): (\d+)', raw)
    data['peak_rss_bytes'] = int(rss[1]) * 1024 if rss else None
    samples = [json.loads(line) for line in (OUT / 'logs' / (name + '-vram.jsonl')).read_text().splitlines()]
    data['vram_free_mib'] = {str(i): {'min': min(v), 'median': statistics.median(v)} for i in [0, 1]
                           for v in [[g['free_mib'] for s in samples for g in s['gpus'] if g['index'] == i]]}
    data['ids_sha256'] = sha(OUT / 'logs' / (name + '-ids.json'))
    data['decode_forward_tps'] = data.get('decode_steps', 0) * 1000 / data.get('decode_ms', 1)
    data['scope_limits'] = re.findall(r'^\d+$', raw, re.M)[:3]
    data['valid'] = proc.returncode == 0 and len(tokens) == 256 and bool(bench) and not re.search(
        r'continuing pageable|\[verify(?:-gpu)?\] MISMATCH|\bDEGRADED:|\[moe\].*missing=|CUDA error', raw)
    records.append(data); (OUT / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(data), flush=True)
    if not data['valid']:
        raise RuntimeError('Invalid measurement ' + name)

def compare(names):
    ids = {n: json.loads((OUT / 'logs' / (n + '-ids.json')).read_text()) for n in names}
    pairs = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            x, y = ids[a], ids[b]
            first = next((k for k, (u,v) in enumerate(zip(x,y)) if u != v), None)
            if first is None and len(x) != len(y): first = min(len(x),len(y))
            pairs.append(dict(a=a,b=b,first_difference=first))
    byte_identical = len({(OUT / 'logs' / (n + '-ids.json')).read_bytes() for n in names}) == 1
    result = dict(names=names, pairs=pairs, byte_identical=byte_identical, passed=byte_identical and all(p['first_difference'] is None for p in pairs))
    (OUT / (names[0].split('-')[0] + '-comparison.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result

def checks():
    for label,command in [('attention',['build-determinism/attn/test_attn_parity']),('aux',['build-determinism/aux/test_aux']),('reductions',['build-determinism/helios_deterministic_reductions_test'])]:
        with (OUT/'logs'/('check-'+label+'.log')).open('w') as f:
            p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
        assert p.returncode == 0, label

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase', choices=['before','after','identity','diagnostic','checks','restore'])
    parser.add_argument('--start',type=int,default=1)
    parser.add_argument('--count',type=int,default=4)
    args=parser.parse_args()
    if args.phase=='restore': print(json.dumps(restore())); raise SystemExit
    binary = 'build-determinism/helios-before' if args.phase=='before' else 'build-determinism/helios'
    if args.phase == 'diagnostic': binary = 'build-determinism/helios-before-diagnostic'
    windows=json.loads((OUT/'outages.json').read_text()) if (OUT/'outages.json').exists() else []
    window=dict(phase=args.phase,start=args.start,count=args.count,stop_started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    windows.append(window); (OUT/'outages.json').write_text(json.dumps(windows,indent=2)+'\n')
    try:
        call([LAUNCHER,'stop'])
        window['stopped']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        if args.phase == 'checks': checks()
        elif args.phase == 'diagnostic':
            os.environ['HELIOS_DET_PLANES']='1'
            for repeat in range(1,3): run(f'diagnostic-before-r{repeat}',binary,'A',4096)
        else:
            names=[]
            for repeat in range(args.start,args.start+args.count):
                name=f'identity-A-r{repeat}' if args.phase=='identity' else f'{args.phase}-A-4096-r{repeat}'
                run(name,binary,'A','identity' if args.phase=='identity' else 4096); names.append(name)
            all_names=[r['name'] for r in json.loads((OUT/'runs.json').read_text()) if r['name'].startswith(args.phase+'-A-') and r['valid']]
            comparison=compare(all_names)
            if args.phase=='before' and comparison['passed'] and len(all_names)==4:
                window['flaky_repro']=True
                for repeat in range(5,9):
                    name=f'before-A-4096-r{repeat}'; run(name,binary,'A',4096); all_names.append(name)
                compare(all_names)
            if args.phase=='identity': checks()
    finally:
        window['restore_started']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        print(json.dumps(restore()),flush=True)
        window['healthy']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
        (OUT/'outages.json').write_text(json.dumps(windows,indent=2)+'\n')
