#!/usr/bin/env python3
"""Isolate the two prefill cliffs the users feel:

  1. cold start: the FIRST forward() in a server process vs the 2nd/3rd (same ctx,
     same chunk size, prefix cache OFF, different prompts) -> the "first token takes
     forever" complaint.
  2. long prompt: prefill tok/s for a ~2k vs a ~9k prompt, and the effect of the
     prefill chunk size (serve defaults to 128 tokens, so a 9k prompt becomes ~71
     chunks that each re-dequantise the whole model).

It drives the real OpenAI-compatible server (same path the user hits) and reads the
server-reported timings: prompt_tokens, prefill_ms, cache_hit_tokens, prefilled_tokens.

Usage:
  python tools/prefill_cliff_probe.py --model D:/p0/models/Qwen3.5-9B-Q5_K_M.gguf \
      --ctx 32768 --rows 128 --port 7071 --tag cold-128 \
      --cases 2000,2000,2000,9100
  (each case is a target prompt length in tokens; use a distinct nonce per request)

  --rows 0 means "do not pass --prefill-rows": keep the serve default (128).
  --prefix-cache 0 keeps the cache off (default in this probe).
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILLER = ('The quick brown fox jumps over the lazy dog while the engine keeps '
          'prefilling tokens as fast as it can. ')
CH_FILLER = ('智能推理引擎正在把很长的文档切分成许多块并逐块计算注意力，'
             '这是一段用来制造长提示词的填充文本。')


def free_port(start):
    for p in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', p))
                return p
            except OSError:
                continue
    raise RuntimeError('no free port')


def wait_health(port, timeout=240.0, proc=None):
    t0 = time.time()
    url = f'http://127.0.0.1:{port}/health'
    while time.time() - t0 < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f'server exited early (rc={proc.returncode})')
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError('server did not become healthy')


def build_prompt(tokens, nonce, lang):
    """Approximate text that tokenizes to ~`tokens` tokens (actual count is read
    back from the server response, so the estimate only needs to be close)."""
    filler = CH_FILLER if lang == 'zh' else FILLER
    # ~1 token per 4 ASCII chars, ~1 token per 1.6 CJK chars; these fillers are
    # measured empirically to land within ~5%.
    per_rep = 16 if lang == 'zh' else 19
    reps = max(1, int(tokens / per_rep))
    head = f'[{nonce}] ' + ('请阅读下面的长文档。' if lang == 'zh' else 'Please read the document below. ')
    return head + filler * reps


def post(port, prompt, max_tokens=1):
    body = json.dumps({
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': 0,
        'stream': False,
        'cache_prompt': False,
    }).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                 data=body, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read().decode())
    wall = (time.time() - t0) * 1000.0
    return out, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--ctx', type=int, default=32768)
    ap.add_argument('--rows', type=int, default=0,
                    help='0 = leave serve default (128); else fixed prefill chunk')
    ap.add_argument('--rows-max', type=int, default=0)
    ap.add_argument('--port', type=int, default=7071)
    ap.add_argument('--cases', default='2000,2000,2000')
    ap.add_argument('--lang', default='en')
    ap.add_argument('--tag', default='run')
    ap.add_argument('--prefix-cache', type=int, default=0)
    ap.add_argument('--max-tokens', type=int, default=1)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    ctx = args.ctx
    cases = [int(x) for x in args.cases.split(',') if x.strip()]
    port = free_port(args.port)
    log_path = os.path.join(ROOT, 'build', f'prefill_cliff_{args.tag}_{ctx}_{port}.log')
    log = open(log_path, 'w', encoding='utf-8', errors='replace')
    cmd = [os.path.join(ROOT, 'build', 'engine.exe'), 'serve', '--model', args.model,
           '--ctx', str(ctx), '--port', str(port),
           '--prefix-cache', str(args.prefix_cache)]
    if args.rows > 0:
        cmd += ['--prefill-rows', str(args.rows)]
        cmd += ['--prefill-rows-max', str(args.rows_max or args.rows)]
    print(f'[{args.tag}] start: {" ".join(cmd)}', flush=True)
    t_start = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        ready_s = wait_health(port, proc=proc)
        load_s = time.time() - t_start
        rows.append({'tag': args.tag, 'ctx': ctx, 'rows': args.rows,
                     'phase': 'startup', 'startup_s': round(load_s, 2)})
        print(f'[{args.tag}] healthy after {load_s:.1f} s (log {log_path})', flush=True)
        for i, target in enumerate(cases):
            prompt = build_prompt(target, f'{args.tag}-{i}', args.lang)
            try:
                out, wall = post(port, prompt, args.max_tokens)
            except urllib.error.HTTPError as e:
                rows.append({'tag': args.tag, 'i': i, 'target': target,
                             'error': e.read().decode()[:400]})
                print(f'  req{i} target={target}: HTTP {e.code}', flush=True)
                continue
            u = out.get('usage', {})
            t = out.get('timings', {})
            pt = int(u.get('prompt_tokens', 0))
            pre = float(t.get('prefill_ms', 0.0))
            pref_tok = int(t.get('prefilled_tokens', pt))
            hit = int(t.get('cache_hit_tokens', 0))
            r = {'tag': args.tag, 'i': i, 'target': target, 'prompt_tokens': pt,
                 'prefilled_tokens': pref_tok, 'cache_hit_tokens': hit,
                 'prefill_ms': round(pre, 1),
                 'prefill_tok_s': round(pref_tok * 1000.0 / pre, 1) if pre > 0 else 0.0,
                 'client_wall_ms': round(wall, 1),
                 'decode_ms': round(float(t.get('decode_ms', 0.0)), 2)}
            rows.append(r)
            print(f"  req{i} prompt={pt} prefilled={pref_tok} hit={hit} "
                  f"prefill={pre:.0f} ms -> {r['prefill_tok_s']:.0f} tok/s "
                  f"(client wall {wall:.0f} ms)", flush=True)
    finally:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/shutdown', timeout=3):
                pass
        except Exception:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    out_path = args.out or os.path.join(ROOT, 'build', f'prefill_cliff_{args.tag}.jsonl')
    with open(out_path, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'[{args.tag}] wrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
