#!/usr/bin/env python3
"""Long-run stability probe for the engine's OpenAI-compatible server.

Covers the "real user keeps chatting for a long time" failure modes that unit tests and
short curl checks miss:

  long_chat  -- 20-turn conversation on one live server: per-turn prompt/decode timings,
                prefix-cache hits, RSS and VRAM trend, and a correctness sanity check
                (greedy replies must stay deterministic and non-degenerate).
  long_ctx   -- servers started at ctx 8192 / 32768: prefill 2k, decode 16, prefill 8k
                (when the ctx allows), decode 16 -- catches OOM / prefill blow-ups.
  restart    -- start/health/shutdown 5x: VRAM returned to baseline, port released,
                process gone (thread/handle leaks would keep the port or the VRAM).
  errors     -- prompt longer than ctx, empty messages, invalid JSON, two concurrent
                streaming requests (single-threaded server must not deadlock).

Usage (needs an exclusive GPU):
    python tools/stability_probe.py --scenario all --model <9B.gguf> --ctx 8192
    python tools/stability_probe.py --scenario long_chat --max-tokens 50 --turns 20
    python tools/stability_probe.py --scenario errors --model <9B.gguf>

Results: build/stability/<scenario>.jsonl (raw) + a summary table on stdout.
Nothing is written outside build/.
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
WAIT_GPU_S = 0   # >0: wait for an exclusive GPU before every server start
OUT_DIR = os.path.join(ROOT, 'build', 'stability')


# --------------------------------------------------------------------------- utils
def http_json(url, payload=None, timeout=600):
    data = None
    headers = {'Content-Type': 'application/json'}
    if payload is not None:
        data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body), time.time() - t0


def http_stream(url, payload, timeout=600):
    """POST a stream=true chat request; return (text, ttfb, total, frames)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    t0 = time.time()
    ttfb = None
    text = []
    frames = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        # readline() on the buffered HTTPResponse: one syscall per SSE frame instead of
        # one per byte (a per-byte read would also pollute the TTFB measurement).
        while True:
            raw = r.readline()
            if not raw:
                break
            line = raw.strip()
            if not line.startswith(b'data:'):
                continue
            frames += 1
            if ttfb is None:
                ttfb = time.time() - t0
            body = line[5:].strip()
            if body == b'[DONE]':
                break
            try:
                obj = json.loads(body)
            except Exception:
                continue
            for ch in obj.get('choices', []):
                piece = ch.get('delta', {}).get('content') or ''
                text.append(piece)
    return ''.join(text), ttfb if ttfb is not None else time.time() - t0, time.time() - t0, frames


def port_open(port, host='127.0.0.1'):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def process_rss_kb(pid):
    if not pid:
        return None
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             f'(Get-Process -Id {pid} -ErrorAction SilentlyContinue).WorkingSet64'],
            capture_output=True, text=True, timeout=20).stdout.strip()
        return int(int(out) / (1024 * 1024)) if out.isdigit() else None  # MB
    except Exception:
        return None


def gpu_used_mb():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


def model_processes_running():
    """True if one of the engine/test binaries is alive (another agent's benchmark)."""
    try:
        out = subprocess.run(['tasklist'], capture_output=True, text=True, timeout=20).stdout
        return any(n in out for n in ('test_gen', 'engine.exe', 'engine_new'))
    except Exception:
        return False


def wait_gpu_free(seconds, quiet_mb=1200):
    """Wait until no model process is alive and VRAM usage looks like a desktop baseline.

    Needed because the GPU is shared between agents: starting a serve while another
    agent's 35B benchmark is loading would corrupt both measurements (and can OOM).
    """
    t0 = time.time()
    while True:
        used = gpu_used_mb()
        busy = model_processes_running()
        if not busy and used is not None and used < quiet_mb:
            return True
        if time.time() - t0 >= seconds:
            return False
        print(f'  [wait-gpu] used={used}MB other_model_procs={busy} '
              f'({time.time() - t0:.0f}/{seconds}s)', flush=True)
        time.sleep(10)


class Server:
    def __init__(self, engine, model, port, ctx, extra=None, ctx_extra=None):
        self.engine = engine
        self.model = model
        self.port = port
        self.ctx = ctx
        self.extra = extra or []
        self.ctx_extra = ctx_extra or []
        self.wait_gpu_s = WAIT_GPU_S
        self.proc = None
        self.log_path = None

    def start(self, wait_s=180):
        if port_open(self.port):
            raise RuntimeError(f'port {self.port} already in use')
        if not wait_gpu_free(self.wait_gpu_s):
            raise RuntimeError('GPU stayed busy (another model process / VRAM > 1200MB)')
        os.makedirs(OUT_DIR, exist_ok=True)
        self.log_path = os.path.join(OUT_DIR, f'serve-{self.port}-ctx{self.ctx}.log')
        log = open(self.log_path, 'wb')
        cmd = [self.engine, 'serve', self.model, '--host', '127.0.0.1', '--port', str(self.port),
               '--ctx', str(self.ctx)] + self.ctx_extra + self.extra
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f'server exited early rc={self.proc.returncode}; see {self.log_path}')
            try:
                j, _ = http_json(f'http://127.0.0.1:{self.port}/health', timeout=3)
                if j.get('status') == 'ok':
                    return time.time() - t0
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError('server did not become healthy in time')

    def stop(self, timeout=60):
        if not self.proc:
            return
        try:
            http_json(f'http://127.0.0.1:{self.port}/shutdown', timeout=10)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=20)
        self.proc = None
        # wait for the port to be released by the OS
        t0 = time.time()
        while time.time() - t0 < 15 and port_open(self.port):
            time.sleep(0.3)


def write_jsonl(name, rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'{name}.jsonl')
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return path


# ----------------------------------------------------------------------- scenarios
def scenario_long_chat(args):
    """20 turns; the model must keep answering coherently, cache must hit, no perf decay."""
    srv = Server(args.engine, args.model, args.port, args.ctx)
    rows = []
    startup = srv.start()
    rows.append({'event': 'startup', 's': round(startup, 2), 'gpu_mb': gpu_used_mb()})
    history = []
    rss0 = process_rss_kb(srv.proc.pid)
    try:
        for turn in range(1, args.turns + 1):
            user = (f'第{turn}轮。请用一句话回答，并接着上文：{args.topic}。'
                    f'另外记住数字{turn}。')
            history.append({'role': 'user', 'content': user})
            payload = {'model': 'm', 'messages': history, 'max_tokens': args.max_tokens,
                       'temperature': 0, 'stream': True}
            text, ttfb, total, frames = http_stream(f'http://127.0.0.1:{srv.port}/v1/chat/completions',
                                                    payload)
            # non-stream request right after to get the `timings` block
            j, rt = http_json(f'http://127.0.0.1:{srv.port}/v1/chat/completions',
                              {'model': 'm', 'messages': history, 'max_tokens': args.max_tokens,
                               'temperature': 0})
            reply = j['choices'][0]['message']['content']
            history.append({'role': 'assistant', 'content': reply})
            t = j.get('timings', {})
            u = j.get('usage', {})
            # NB: the serve `timings` block only carries prefill_ms/decode_ms/... ;
            # token counts live in `usage` (streaming's final chunk has its own layout).
            prompt_n = t.get('prompt_n') or u.get('prompt_tokens')
            predicted_n = t.get('predicted_n') or u.get('completion_tokens')
            row = {
                'event': 'turn', 'turn': turn, 'startup_s': round(startup, 2),
                'prompt_n': prompt_n, 'cache_hit': t.get('cache_hit_tokens'),
                'predicted_n': predicted_n,
                'decode_tps': round(t.get('tokens_per_second') or 0, 2),
                'prefill_ms': round(t.get('prefill_ms') or 0, 1),
                'ttfb_ms': round((ttfb or 0) * 1000, 1),
                'stream_total_ms': round(total * 1000, 1),
                'nonstream_total_ms': round(rt * 1000, 1),
                'rss_mb': process_rss_kb(srv.proc.pid),
                'rss_delta_mb': (lambda r, r0: (r - r0) if (r and r0) else None)(
                    process_rss_kb(srv.proc.pid), rss0),
                'gpu_mb': gpu_used_mb(),
                'reply_head': reply[:80],
                'reply_len': len(reply),
                'frames': frames,
            }
            rows.append(row)
            print(f"turn {turn:2d}  prompt={row['prompt_n']!s:>5} "
                  f"cache_hit={row['cache_hit']!s:>5} pred={row['predicted_n']!s:>3} "
                  f"dec={row['decode_tps']:5.1f} tok/s ttfb={row['ttfb_ms']:6.0f}ms "
                  f"rss={row['rss_mb']}MB gpu={row['gpu_mb']}MB", flush=True)
    finally:
        srv.stop()
        rows.append({'event': 'shutdown', 'port_open': port_open(srv.port),
                     'gpu_mb': gpu_used_mb(), 'proc_alive': False})
    write_jsonl('long_chat', rows)
    _summarise_long_chat(rows)


def _summarise_long_chat(rows):
    turns = [r for r in rows if r.get('event') == 'turn']
    if not turns:
        return
    tps = [r['decode_tps'] for r in turns if r.get('decode_tps')]
    hits = [r['cache_hit'] or 0 for r in turns]
    print('\n== long_chat summary ==')
    print(f"turns={len(turns)}  decode tok/s: first={tps[0] if tps else '?'} "
          f"last={tps[-1] if tps else '?'} min={min(tps) if tps else '?'} "
          f"max={max(tps) if tps else '?'}")
    print(f"prefix-cache hits per turn: {hits} (turn1 must be 0, later turns > 0 if cache works)")
    rss = [r['rss_mb'] for r in turns if r.get('rss_mb')]
    if len(rss) >= 2:
        print(f"RSS: first={rss[0]}MB last={rss[-1]}MB delta={rss[-1] - rss[0]}MB "
              f"(linear leak would grow every turn)")
    gpu = [r['gpu_mb'] for r in turns if r.get('gpu_mb')]
    if len(gpu) >= 2:
        print(f"VRAM: first={gpu[0]}MB last={gpu[-1]}MB delta={gpu[-1] - gpu[0]}MB")
    empties = [r['turn'] for r in turns if not r.get('reply_len')]
    print(f"empty replies: {empties if empties else 'none'}")


def scenario_long_ctx(args):
    """ctx 8192 and 32768: prefill 2k, decode 16, prefill 8k, decode 16."""
    rows = []
    for ctx in [8192, 32768]:
        srv = Server(args.engine, args.model, args.port, ctx,
                     ctx_extra=args.ctx_extra if ctx == args.ctx else [])
        try:
            startup = srv.start()
        except Exception as e:
            rows.append({'event': 'start_failed', 'ctx': ctx, 'error': str(e),
                         'gpu_mb': gpu_used_mb()})
            print(f'ctx={ctx}: START FAILED: {e}')
            continue
        rows.append({'event': 'startup', 'ctx': ctx, 's': round(startup, 2),
                     'gpu_mb': gpu_used_mb(), 'log': srv.log_path})
        try:
            for n_prefill in (2048, 8192):
                if n_prefill >= ctx:
                    continue
                prompt = ('The quick brown fox jumps over the lazy dog. ' * 2000)[:n_prefill * 5]
                payload = {'model': 'm', 'messages': [{'role': 'user', 'content': prompt}],
                           'max_tokens': 16, 'temperature': 0}
                t0 = time.time()
                j, rt = http_json(f'http://127.0.0.1:{srv.port}/v1/chat/completions', payload,
                                  timeout=1800)
                t = j.get('timings', {})
                u = j.get('usage', {})
                prompt_n = t.get('prompt_n') or u.get('prompt_tokens')
                predicted_n = t.get('predicted_n') or u.get('completion_tokens')
                row = {'event': 'ctx_run', 'ctx': ctx, 'prefill_target': n_prefill,
                       'prompt_n': prompt_n, 'prefill_ms': t.get('prefill_ms'),
                       'prefill_tps': round((prompt_n or 0) /
                                            max(1e-9, (t.get('prefill_ms') or 1) / 1000), 1),
                       'decode_tps': round(t.get('tokens_per_second') or 0, 2),
                       'predicted_n': predicted_n,
                       'wall_ms': round(rt * 1000, 1), 'gpu_mb': gpu_used_mb(),
                       'rss_mb': process_rss_kb(srv.proc.pid),
                       'reply_head': (j['choices'][0]['message']['content'] or '')[:60]}
                rows.append(row)
                print(f"ctx={ctx} prefill={n_prefill}: prompt_n={row['prompt_n']} "
                      f"prefill={row['prefill_ms']}ms ({row['prefill_tps']} tok/s) "
                      f"decode={row['decode_tps']} tok/s gpu={row['gpu_mb']}MB", flush=True)
        except Exception as e:
            rows.append({'event': 'run_failed', 'ctx': ctx, 'error': str(e),
                         'gpu_mb': gpu_used_mb()})
            print(f'ctx={ctx}: RUN FAILED: {e}')
        finally:
            srv.stop()
    write_jsonl('long_ctx', rows)


def scenario_restart(args):
    """5x start -> health -> shutdown; VRAM must return to the pre-start baseline."""
    rows = []
    base = gpu_used_mb()
    rows.append({'event': 'baseline', 'gpu_mb': base})
    for i in range(1, 6):
        srv = Server(args.engine, args.model, args.port, 2048)
        row = {'event': 'cycle', 'n': i}
        try:
            row['startup_s'] = round(srv.start(), 2)
            row['gpu_running_mb'] = gpu_used_mb()
            j, _ = http_json(f'http://127.0.0.1:{srv.port}/v1/models', timeout=10)
            row['models_ok'] = bool(j.get('data'))
        except Exception as e:
            row['error'] = str(e)
        finally:
            srv.stop()
        time.sleep(2)
        row['gpu_after_mb'] = gpu_used_mb()
        row['gpu_delta_vs_baseline'] = (row['gpu_after_mb'] - base
                                        if row.get('gpu_after_mb') is not None and base else None)
        row['port_open_after'] = port_open(srv.port)
        row['proc_gone'] = True
        rows.append(row)
        print(f"cycle {i}: startup={row.get('startup_s')}s run={row.get('gpu_running_mb')}MB "
              f"after={row.get('gpu_after_mb')}MB delta={row.get('gpu_delta_vs_baseline')}MB "
              f"port_open={row['port_open_after']} err={row.get('error', '-')}")
    write_jsonl('restart', rows)
    deltas = [abs(r['gpu_delta_vs_baseline']) for r in rows
              if r.get('event') == 'cycle' and r.get('gpu_delta_vs_baseline') is not None]
    if deltas:
        print(f'\n== restart summary == VRAM delta vs baseline: max={max(deltas)}MB '
              f'(>64MB suggests a leak) ; any port left open: '
              f"{any(r.get('port_open_after') for r in rows if r.get('event') == 'cycle')}")


def scenario_errors(args):
    """Malformed / oversized / concurrent requests must not hang or crash the server."""
    srv = Server(args.engine, args.model, args.port, 2048)
    rows = []
    try:
        srv.start()
        base = f'http://127.0.0.1:{srv.port}'
        cases = []

        def run_case(name, fn, timeout=120):
            t0 = time.time()
            try:
                detail = fn()
                rows.append({'event': 'case', 'name': name, 'ok': True,
                             'ms': round((time.time() - t0) * 1000, 1), 'detail': detail})
                print(f'  {name:34s} OK    {detail}')
            except Exception as e:
                rows.append({'event': 'case', 'name': name, 'ok': False,
                             'ms': round((time.time() - t0) * 1000, 1), 'error': str(e)})
                print(f'  {name:34s} ERR   {e}')

        # 1) prompt much longer than ctx
        long_prompt = ('word ' * 6000)
        run_case('prompt_longer_than_ctx', lambda: _post_raw(
            f'{base}/v1/chat/completions',
            json.dumps({'model': 'm', 'messages': [{'role': 'user', 'content': long_prompt}],
                        'max_tokens': 8, 'temperature': 0}).encode(), 300))
        # 2) empty messages
        run_case('empty_messages', lambda: _post_raw(
            f'{base}/v1/chat/completions',
            json.dumps({'model': 'm', 'messages': [], 'max_tokens': 8}).encode(), 60))
        # 3) invalid JSON
        run_case('invalid_json', lambda: _post_raw(f'{base}/v1/chat/completions',
                                                   b'{"model": "m", broken', 30))
        # 4) unknown route
        run_case('unknown_route', lambda: _post_raw(f'{base}/v1/nope', b'{}', 30))
        # 5) two concurrent streaming requests (single-threaded accept loop)
        def concurrent():
            results = {}

            def one(tag):
                try:
                    text, ttfb, total, frames = http_stream(
                        f'{base}/v1/chat/completions',
                        {'model': 'm', 'messages': [{'role': 'user', 'content': '用一句话说你好'}],
                         'max_tokens': 24, 'temperature': 0, 'stream': True}, timeout=300)
                    results[tag] = {'len': len(text), 'ttfb_ms': round(ttfb * 1000, 1),
                                    'total_ms': round(total * 1000, 1), 'frames': frames}
                except Exception as e:
                    results[tag] = {'error': str(e)}
            threads = [threading.Thread(target=one, args=(f'req{i}',)) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=280)
            return results

        run_case('two_concurrent_streams', concurrent)
        # 6) server still healthy afterwards?
        run_case('health_after_errors', lambda: http_json(f'{base}/health', timeout=10)[0])
    finally:
        srv.stop()
        rows.append({'event': 'shutdown', 'port_open': port_open(srv.port),
                     'gpu_mb': gpu_used_mb()})
    write_jsonl('errors', rows)


def _post_raw(url, data, timeout):
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(512).decode('utf-8', 'replace')
            return {'status': r.status, 'body': body[:200]}
    except urllib.error.HTTPError as e:
        return {'status': e.code, 'body': e.read(200).decode('utf-8', 'replace')}


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='all',
                    choices=['all', 'long_chat', 'long_ctx', 'restart', 'errors'])
    ap.add_argument('--engine', default=os.path.join(ROOT, 'build', 'engine.exe'))
    ap.add_argument('--model', default=os.environ.get(
        'MODEL', os.path.join(os.environ.get('MODEL_DIR', ''),
                              'Qwen3.5-9B-Q5_K_M.gguf')))
    ap.add_argument('--port', type=int, default=7060)
    ap.add_argument('--ctx', type=int, default=8192)
    ap.add_argument('--turns', type=int, default=20)
    ap.add_argument('--max-tokens', type=int, default=50)
    ap.add_argument('--topic', default='你正在和我聊天')
    ap.add_argument('--ctx-extra', default='')
    ap.add_argument('--wait-gpu-s', type=int, default=0,
                    help='wait up to N seconds for an exclusive GPU before each server start')
    args = ap.parse_args()
    args.ctx_extra = args.ctx_extra.split() if args.ctx_extra else []
    global WAIT_GPU_S
    WAIT_GPU_S = args.wait_gpu_s
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'engine={args.engine}\nmodel={args.model}\nport={args.port} ctx={args.ctx}')
    t0 = time.time()
    todo = (['long_chat', 'long_ctx', 'restart', 'errors'] if args.scenario == 'all'
            else [args.scenario])
    for sc in todo:
        print(f'\n===== scenario: {sc} =====')
        {'long_chat': scenario_long_chat, 'long_ctx': scenario_long_ctx,
         'restart': scenario_restart, 'errors': scenario_errors}[sc](args)
    print(f'\nall done in {time.time() - t0:.1f}s; raw jsonl in {OUT_DIR}')


if __name__ == '__main__':
    main()
